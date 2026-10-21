//! MLA weight container and the three projections
//! (Q_a/Q_b, KV_a + latent RMSNorm, Wv/Wo output groups).
//!
//! Ground truth: DeepSeek-V4.1-Flash / FlashMLA README —
//! `hidden --Q_a--> [q_lora 1536] --(no Q-norm in V4.1)-- Q_b-->
//! [n_heads 64, head_dim 512]` where `Q_b` is the ABSORBED query
//! projection (`W_UK` folded in upstream); `hidden --KV_a-->
//! [512] = [latent 448 | rope 64]` with RMSNorm applied to the latent
//! slice only; attention output `[s, 64, 512]` reshaped to 8 groups of
//! `(64/8 heads * 512 = 4096)`, projected per group by
//! `Wv [512, 4096]`, concatenated to `[4096]` and mapped back to
//! `hidden` by `Wo [hidden, 4096]`.
//!
//! RoPE is deliberately NOT applied inside the projections: the caller
//! rotates the rope slice (the LAST `rope_dim` dims of each row) via
//! [`crate::MlaWeights::rope`] before cache append / query use.

use crate::config::{MlaConfig, PAGE_SIZE};
use deepinfer_core::math::dot;
use deepinfer_core::norm::rmsnorm_into;
use deepinfer_core::rng::Philox;
use deepinfer_core::rope::{Rope, RopeStyle};
use deepinfer_core::{ensure, Error, Result};

/// RMSNorm epsilon used by the MLA norm layers. `MlaWeights` carries no
/// config (the interface pins its public fields), so the V4.1 value
/// (1e-4, FlashMLA / DeepSeek-V4.1 configuration) is used.
pub(crate) const V41_RMS_EPS: f32 = 1e-4;

/// Ceiling on the RoPE position table built by [`MlaWeights::from_config`].
const ROPE_MAX_POSITIONS: usize = 65_536;
/// Floor for the same table.
const ROPE_MIN_POSITIONS: usize = 1024;

/// The MLA attention weights. All matrices are row-major `[rows, cols]`
/// and applied as `y = W @ x`.
///
/// The layer dimensions are derived (and validated) from the field sizes
/// at every projection call — see [`MlaWeights::project_q`] — so weights
/// loaded from a checkpoint are self-describing:
/// `head_dim = kv_norm.len() + rope.dim`,
/// `hidden = kv_a_proj.len() / head_dim`,
/// `q_lora = q_a_proj.len() / hidden`,
/// `n_heads = q_b_proj.len() / (q_lora * head_dim)`,
/// `o_lora = wv_proj.len() / (n_heads * head_dim)` and
/// `o_groups = wo_proj.len() / (hidden * o_lora)`.
#[derive(Debug, Clone)]
pub struct MlaWeights {
    /// Query-LoRA down-projection `[q_lora_rank, hidden_size]`.
    pub q_a_proj: Vec<f32>,
    /// Absorbed query up-projection `[n_heads * head_dim, q_lora_rank]`
    /// (V4.1 folds `W_UK` into `Q_b`, so queries live in the 512-dim
    /// latent+rope space of the cache).
    pub q_b_proj: Vec<f32>,
    /// KV down-projection `[head_dim, hidden_size]`; rows are
    /// `[latent (head_dim - rope_dim) | rope (rope_dim)]`.
    pub kv_a_proj: Vec<f32>,
    /// Grouped output projection `[o_groups * o_lora_rank,
    /// (n_heads / o_groups) * head_dim]`.
    pub wv_proj: Vec<f32>,
    /// Final output projection `[hidden_size, o_groups * o_lora_rank]`.
    pub wo_proj: Vec<f32>,
    /// V4-only Q-norm over the query-LoRA slice; `None` for V4.1.
    pub q_norm: Option<Vec<f32>>,
    /// RMSNorm gains over the latent part of the KV projection
    /// (`[head_dim - rope_dim]`).
    pub kv_norm: Vec<f32>,
    /// Optional per-head attention sink logits `[n_heads]` (FlashMLA
    /// `attn_sink`); applied by the attention kernels, not here.
    pub attn_sink: Option<Vec<f32>>,
    /// RoPE built from the config (interleaved style; the rope slice is
    /// the LAST `rope_dim` dims of each head row).
    pub rope: Rope,
}

/// Layer dimensions derived from the weight shapes.
struct Dims {
    hidden: usize,
    q_lora: usize,
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    o_groups: usize,
    o_lora: usize,
}

/// Allocates `len` zeroed f32s, or an empty vector when the reservation
/// fails — allocation failure must not abort the process, and degenerate
/// (empty) weights are then rejected by the projection validators.
fn alloc_or_empty(len: usize) -> Vec<f32> {
    let mut v = Vec::new();
    if v.try_reserve_exact(len).is_ok() {
        v.resize(len, 0.0);
    }
    v
}

/// `y = W @ x` with `W` row-major `[y.len(), x.len()]` and sequential
/// (deterministic) accumulation.
fn mat_vec(w: &[f32], x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(w.len(), x.len() * y.len());
    let cols = x.len();
    for (r, yr) in y.iter_mut().enumerate() {
        *yr = dot(&w[r * cols..(r + 1) * cols], x);
    }
}

impl MlaWeights {
    /// Random initialization for tests: every matrix is filled
    /// uniformly in `[-0.02, 0.02]` (Philox, deterministic consumption
    /// order), `kv_norm` is set to ones (identity norm), `q_norm` is
    /// `None` (V4.1 has no Q-norm) and `attn_sink` is `None`.
    ///
    /// `from_config` is infallible by interface contract, so degenerate
    /// config values are deterministically sanitized (never panicking):
    /// dimensions are clamped to sane test-scale bounds, `rope_dim` is
    /// made even and kept strictly below `head_dim`, an `o_groups` that
    /// does not divide `n_heads` falls back to 1, and a `rope_base <= 1`
    /// falls back to 10000. The RoPE table covers
    /// `max_pages * 64` positions clamped to `[1024, 65536]` (callers can
    /// replace the public `rope` field with a longer table if needed).
    pub fn from_config(cfg: &MlaConfig, rng: &mut Philox) -> MlaWeights {
        let hidden = cfg.hidden_size.clamp(1, 8192);
        let q_lora = cfg.q_lora_rank.clamp(1, 2048);
        let n_heads = cfg.n_heads.clamp(1, 128);
        let rope_dim = cfg.rope_dim.clamp(2, 512) & !1;
        let head_dim = cfg.head_dim.clamp(rope_dim.saturating_add(2).max(4), 1024);
        let o_groups = if cfg.o_groups >= 1
            && cfg.o_groups <= 64
            && n_heads % cfg.o_groups == 0
        {
            cfg.o_groups
        } else {
            1
        };
        let o_lora = cfg.o_lora_rank.clamp(1, 2048);
        let base = if cfg.rope_base > 1.0 {
            cfg.rope_base
        } else {
            10_000.0
        };
        let positions = ((cfg.max_pages as u64 * PAGE_SIZE as u64)
            .clamp(ROPE_MIN_POSITIONS as u64, ROPE_MAX_POSITIONS as u64))
            as usize;
        // The sanitized arguments provably satisfy Rope::new's contract
        // (even dim in [2, 512], base > 1, positions >= 1024).
        let rope = Rope::new(rope_dim, RopeStyle::Interleaved, base, cfg.yarn, positions)
            .expect("sanitized rope parameters are valid by construction");
        let mut w = MlaWeights {
            q_a_proj: alloc_or_empty(q_lora.checked_mul(hidden).unwrap_or(0)),
            q_b_proj: alloc_or_empty(
                n_heads
                    .checked_mul(head_dim)
                    .and_then(|v| v.checked_mul(q_lora))
                    .unwrap_or(0),
            ),
            kv_a_proj: alloc_or_empty(head_dim.checked_mul(hidden).unwrap_or(0)),
            wv_proj: alloc_or_empty(
                o_groups
                    .checked_mul(o_lora)
                    .and_then(|v| v.checked_mul(n_heads / o_groups))
                    .and_then(|v| v.checked_mul(head_dim))
                    .unwrap_or(0),
            ),
            wo_proj: alloc_or_empty(
                hidden
                    .checked_mul(o_groups)
                    .and_then(|v| v.checked_mul(o_lora))
                    .unwrap_or(0),
            ),
            q_norm: None,
            kv_norm: {
                let mut v = alloc_or_empty(head_dim - rope_dim);
                v.fill(1.0);
                v
            },
            attn_sink: None,
            rope,
        };
        rng.fill_uniform(&mut w.q_a_proj, -0.02, 0.02);
        rng.fill_uniform(&mut w.q_b_proj, -0.02, 0.02);
        rng.fill_uniform(&mut w.kv_a_proj, -0.02, 0.02);
        rng.fill_uniform(&mut w.wv_proj, -0.02, 0.02);
        rng.fill_uniform(&mut w.wo_proj, -0.02, 0.02);
        w
    }

    /// Derives and validates the layer dimensions from the weight shapes.
    fn derive_dims(&self) -> Result<Dims> {
        let rope_dim = self.rope.dim;
        let latent = self.kv_norm.len();
        ensure!(
            latent >= 1,
            DimensionMismatch,
            "kv_norm must carry at least one latent weight (got {latent})"
        );
        let head_dim = latent
            .checked_add(rope_dim)
            .ok_or_else(|| Error::DimensionMismatch("head_dim overflows".into()))?;
        ensure!(
            head_dim > 0 && !self.kv_a_proj.is_empty(),
            DimensionMismatch,
            "kv_a_proj is empty"
        );
        ensure!(
            self.kv_a_proj.len() % head_dim == 0,
            DimensionMismatch,
            "kv_a_proj length {} is not divisible by head_dim {head_dim}",
            self.kv_a_proj.len()
        );
        let hidden = self.kv_a_proj.len() / head_dim;
        ensure!(
            !self.q_a_proj.is_empty() && self.q_a_proj.len() % hidden == 0,
            DimensionMismatch,
            "q_a_proj length {} is not divisible by hidden {hidden}",
            self.q_a_proj.len()
        );
        let q_lora = self.q_a_proj.len() / hidden;
        ensure!(
            !self.q_b_proj.is_empty() && self.q_b_proj.len() % q_lora == 0,
            DimensionMismatch,
            "q_b_proj length {} is not divisible by q_lora {q_lora}",
            self.q_b_proj.len()
        );
        let qh = self.q_b_proj.len() / q_lora;
        ensure!(
            qh % head_dim == 0,
            DimensionMismatch,
            "q_b_proj row count {qh} is not a multiple of head_dim {head_dim}"
        );
        let n_heads = qh / head_dim;
        let prod = n_heads
            .checked_mul(head_dim)
            .ok_or_else(|| Error::DimensionMismatch("n_heads*head_dim overflows".into()))?;
        ensure!(
            !self.wv_proj.is_empty() && self.wv_proj.len() % prod == 0,
            DimensionMismatch,
            "wv_proj length {} is not divisible by n_heads*head_dim ({prod})",
            self.wv_proj.len()
        );
        let o_lora = self.wv_proj.len() / prod;
        let ol = hidden
            .checked_mul(o_lora)
            .ok_or_else(|| Error::DimensionMismatch("hidden*o_lora overflows".into()))?;
        ensure!(
            !self.wo_proj.is_empty() && self.wo_proj.len() % ol == 0,
            DimensionMismatch,
            "wo_proj length {} is not divisible by hidden*o_lora ({ol})",
            self.wo_proj.len()
        );
        let o_groups = self.wo_proj.len() / ol;
        ensure!(
            o_groups >= 1 && n_heads % o_groups == 0,
            DimensionMismatch,
            "derived o_groups {o_groups} does not divide n_heads {n_heads}"
        );
        if let Some(qn) = &self.q_norm {
            ensure!(
                qn.len() == q_lora,
                DimensionMismatch,
                "q_norm length {} does not match q_lora {q_lora}",
                qn.len()
            );
        }
        if let Some(sk) = &self.attn_sink {
            ensure!(
                sk.len() == n_heads,
                DimensionMismatch,
                "attn_sink length {} does not match n_heads {n_heads}",
                sk.len()
            );
        }
        Ok(Dims {
            hidden,
            q_lora,
            n_heads,
            head_dim,
            rope_dim,
            o_groups,
            o_lora,
        })
    }

    /// Query projection: `Q_a` (+ optional Q-norm) then the absorbed
    /// `Q_b`. `hidden` is `[s, hidden_size]`; `out` is
    /// `[s, n_heads, head_dim]` with RoPE NOT yet applied — the caller
    /// rotates the last `rope_dim` dims of each head row with
    /// `self.rope` before use.
    ///
    /// # Errors
    /// [`Error::DimensionMismatch`] on buffer/weight shape mismatch,
    /// [`Error::Numerical`] on non-finite hidden states.
    pub fn project_q(&self, hidden: &[f32], s: usize, out: &mut [f32]) -> Result<()> {
        let dm = self.derive_dims()?;
        let expect_h = s
            .checked_mul(dm.hidden)
            .ok_or_else(|| Error::DimensionMismatch("s*hidden_size overflows".into()))?;
        ensure!(
            hidden.len() == expect_h,
            DimensionMismatch,
            "hidden has {} elems, expected s*hidden_size = {expect_h}",
            hidden.len()
        );
        let expect_o = s
            .checked_mul(dm.n_heads)
            .and_then(|v| v.checked_mul(dm.head_dim))
            .ok_or_else(|| Error::DimensionMismatch("s*n_heads*head_dim overflows".into()))?;
        ensure!(
            out.len() == expect_o,
            DimensionMismatch,
            "out has {} elems, expected s*n_heads*head_dim = {expect_o}",
            out.len()
        );
        ensure!(
            hidden.iter().all(|v| v.is_finite()),
            Numerical,
            "hidden states contain non-finite values"
        );
        let mut lora = vec![0.0f32; dm.q_lora];
        let row = dm.n_heads * dm.head_dim;
        for t in 0..s {
            let hvec = &hidden[t * dm.hidden..(t + 1) * dm.hidden];
            mat_vec(&self.q_a_proj, hvec, &mut lora);
            if let Some(qn) = &self.q_norm {
                deepinfer_core::norm::rmsnorm(&mut lora, qn, V41_RMS_EPS);
            }
            mat_vec(
                &self.q_b_proj,
                &lora,
                &mut out[t * row..(t + 1) * row],
            );
        }
        Ok(())
    }

    /// KV projection: `KV_a` then RMSNorm over the latent slice ONLY.
    /// `hidden` is `[s, hidden_size]`; `out` is `[s, head_dim]` with rows
    /// `[latent | rope-raw]` — the rope slice is left un-rotated; the
    /// caller applies `self.rope` (offset `head_dim - rope_dim`) before
    /// appending to the cache.
    ///
    /// # Errors
    /// [`Error::DimensionMismatch`] on buffer/weight shape mismatch,
    /// [`Error::Numerical`] on non-finite hidden states.
    pub fn project_kv(&self, hidden: &[f32], s: usize, out: &mut [f32]) -> Result<()> {
        let dm = self.derive_dims()?;
        let expect_h = s
            .checked_mul(dm.hidden)
            .ok_or_else(|| Error::DimensionMismatch("s*hidden_size overflows".into()))?;
        ensure!(
            hidden.len() == expect_h,
            DimensionMismatch,
            "hidden has {} elems, expected s*hidden_size = {expect_h}",
            hidden.len()
        );
        let expect_o = s
            .checked_mul(dm.head_dim)
            .ok_or_else(|| Error::DimensionMismatch("s*head_dim overflows".into()))?;
        ensure!(
            out.len() == expect_o,
            DimensionMismatch,
            "out has {} elems, expected s*head_dim = {expect_o}",
            out.len()
        );
        ensure!(
            hidden.iter().all(|v| v.is_finite()),
            Numerical,
            "hidden states contain non-finite values"
        );
        let latent = dm.head_dim - dm.rope_dim;
        let mut row = vec![0.0f32; dm.head_dim];
        for t in 0..s {
            let hvec = &hidden[t * dm.hidden..(t + 1) * dm.hidden];
            mat_vec(&self.kv_a_proj, hvec, &mut row);
            let orow = &mut out[t * dm.head_dim..(t + 1) * dm.head_dim];
            rmsnorm_into(&row, &self.kv_norm, V41_RMS_EPS, orow, 0, latent);
            orow[latent..].copy_from_slice(&row[latent..]);
        }
        Ok(())
    }

    /// Output projection: `attn_out` is `[s, n_heads, head_dim]` (after
    /// the O-RoPE conjugate); the row is reshaped to
    /// `[o_groups, (n_heads/o_groups) * head_dim]`, each group is
    /// projected by its `Wv [o_lora_rank, ...]` slice, the results are
    /// concatenated and mapped to `[s, hidden_size]` by `Wo`.
    ///
    /// # Errors
    /// [`Error::DimensionMismatch`] on buffer/weight shape mismatch,
    /// [`Error::Numerical`] on non-finite inputs.
    pub fn project_o(&self, attn_out: &[f32], s: usize, out: &mut [f32]) -> Result<()> {
        let dm = self.derive_dims()?;
        let expect_i = s
            .checked_mul(dm.n_heads)
            .and_then(|v| v.checked_mul(dm.head_dim))
            .ok_or_else(|| Error::DimensionMismatch("s*n_heads*head_dim overflows".into()))?;
        ensure!(
            attn_out.len() == expect_i,
            DimensionMismatch,
            "attn_out has {} elems, expected s*n_heads*head_dim = {expect_i}",
            attn_out.len()
        );
        let expect_o = s
            .checked_mul(dm.hidden)
            .ok_or_else(|| Error::DimensionMismatch("s*hidden_size overflows".into()))?;
        ensure!(
            out.len() == expect_o,
            DimensionMismatch,
            "out has {} elems, expected s*hidden_size = {expect_o}",
            out.len()
        );
        ensure!(
            attn_out.iter().all(|v| v.is_finite()),
            Numerical,
            "attn_out contains non-finite values"
        );
        let hpg = dm.n_heads / dm.o_groups;
        let r = hpg * dm.head_dim; // Wv group input width
        let lora_len = dm.o_groups * dm.o_lora;
        let mut lora = vec![0.0f32; lora_len];
        for t in 0..s {
            let x = &attn_out[t * dm.n_heads * dm.head_dim..(t + 1) * dm.n_heads * dm.head_dim];
            for g in 0..dm.o_groups {
                let xg = &x[g * r..(g + 1) * r];
                let wg = &self.wv_proj[g * dm.o_lora * r..(g + 1) * dm.o_lora * r];
                mat_vec(wg, xg, &mut lora[g * dm.o_lora..(g + 1) * dm.o_lora]);
            }
            mat_vec(&self.wo_proj, &lora, &mut out[t * dm.hidden..(t + 1) * dm.hidden]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CacheFormat;
    use crate::testutil::small_cfg;
    use deepinfer_core::prop::{assert_close, random_vec};

    #[test]
    fn from_config_is_deterministic_and_well_shaped() {
        let cfg = small_cfg(CacheFormat::Bf16, 8);
        let mut rng1 = Philox::new(42, 1);
        let mut rng2 = Philox::new(42, 1);
        let a = MlaWeights::from_config(&cfg, &mut rng1);
        let b = MlaWeights::from_config(&cfg, &mut rng2);
        assert_eq!(a.q_a_proj, b.q_a_proj);
        assert_eq!(a.q_b_proj, b.q_b_proj);
        assert_eq!(a.kv_a_proj, b.kv_a_proj);
        assert_eq!(a.wv_proj, b.wv_proj);
        assert_eq!(a.wo_proj, b.wo_proj);
        assert_eq!(a.kv_norm, b.kv_norm);
        assert_eq!(a.q_a_proj.len(), cfg.q_lora_rank * cfg.hidden_size);
        assert_eq!(a.q_b_proj.len(), cfg.n_heads * cfg.head_dim * cfg.q_lora_rank);
        assert_eq!(a.kv_a_proj.len(), cfg.head_dim * cfg.hidden_size);
        assert_eq!(a.wv_proj.len(), cfg.o_groups * cfg.o_lora_rank * (cfg.n_heads / cfg.o_groups) * cfg.head_dim);
        assert_eq!(a.wo_proj.len(), cfg.hidden_size * cfg.o_groups * cfg.o_lora_rank);
        assert_eq!(a.kv_norm, vec![1.0f32; cfg.head_dim - cfg.rope_dim]);
        assert!(a.q_norm.is_none());
        assert!(a.attn_sink.is_none());
        assert!(a.q_a_proj.iter().all(|v| v.abs() <= 0.02));
        assert_eq!(a.rope.dim, cfg.rope_dim);
    }

    #[test]
    fn project_q_matches_manual_reference() {
        let cfg = small_cfg(CacheFormat::Bf16, 8);
        let mut rng = Philox::new(7, 3);
        let w = MlaWeights::from_config(&cfg, &mut rng);
        let s = 4usize;
        let hidden = random_vec(&mut rng, s * cfg.hidden_size, 1.0);
        let mut out = vec![0.0f32; s * cfg.n_heads * cfg.head_dim];
        w.project_q(&hidden, s, &mut out).unwrap();
        for t in 0..s {
            let hv = &hidden[t * cfg.hidden_size..(t + 1) * cfg.hidden_size];
            for hi in 0..cfg.n_heads {
                for d in 0..cfg.head_dim {
                    let row = hi * cfg.head_dim + d;
                    let mut lora = vec![0.0f32; cfg.q_lora_rank];
                    for (l, lv) in lora.iter_mut().enumerate() {
                        let mut acc = 0.0;
                        for c in 0..cfg.hidden_size {
                            acc += w.q_a_proj[l * cfg.hidden_size + c] * hv[c];
                        }
                        *lv = acc;
                    }
                    let mut acc = 0.0;
                    for c in 0..cfg.q_lora_rank {
                        acc += w.q_b_proj[row * cfg.q_lora_rank + c] * lora[c];
                    }
                    let got = out[t * cfg.n_heads * cfg.head_dim + row];
                    assert_close(got, acc, 1e-4, "project_q");
                }
            }
        }
    }

    #[test]
    fn project_kv_rmsnorms_latent_only() {
        let cfg = small_cfg(CacheFormat::Bf16, 8);
        let mut rng = Philox::new(9, 4);
        let w = MlaWeights::from_config(&cfg, &mut rng);
        let s = 3usize;
        let hidden = random_vec(&mut rng, s * cfg.hidden_size, 1.0);
        let mut out = vec![0.0f32; s * cfg.head_dim];
        w.project_kv(&hidden, s, &mut out).unwrap();
        let latent = cfg.head_dim - cfg.rope_dim;
        for t in 0..s {
            let hv = &hidden[t * cfg.hidden_size..(t + 1) * cfg.hidden_size];
            let orow = &out[t * cfg.head_dim..(t + 1) * cfg.head_dim];
            // Raw KV_a row.
            let mut raw = vec![0.0f32; cfg.head_dim];
            for r in 0..cfg.head_dim {
                let mut acc = 0.0;
                for c in 0..cfg.hidden_size {
                    acc += w.kv_a_proj[r * cfg.hidden_size + c] * hv[c];
                }
                raw[r] = acc;
            }
            // Rope slice copied verbatim (rope NOT applied here).
            for j in latent..cfg.head_dim {
                assert_close(orow[j], raw[j], 1e-12, "rope slice raw");
            }
            // Latent slice is RMSNorm(raw_latent) * kv_norm.
            let ms: f32 = raw[..latent].iter().map(|v| v * v).sum::<f32>() / latent as f32;
            let inv = 1.0 / (ms + V41_RMS_EPS).sqrt();
            for j in 0..latent {
                let expect = raw[j] * inv * w.kv_norm[j];
                assert_close(orow[j], expect, 1e-4, "latent rmsnorm");
            }
        }
    }

    #[test]
    fn project_o_matches_manual_einsum() {
        let cfg = small_cfg(CacheFormat::Bf16, 8);
        let mut rng = Philox::new(11, 2);
        let w = MlaWeights::from_config(&cfg, &mut rng);
        let s = 3usize;
        let attn = random_vec(&mut rng, s * cfg.n_heads * cfg.head_dim, 1.0);
        let mut out = vec![0.0f32; s * cfg.hidden_size];
        w.project_o(&attn, s, &mut out).unwrap();
        let hpg = cfg.n_heads / cfg.o_groups;
        let r = hpg * cfg.head_dim;
        let lora_len = cfg.o_groups * cfg.o_lora_rank;
        for t in 0..s {
            for hs in 0..cfg.hidden_size {
                let mut acc = 0.0;
                for ol in 0..lora_len {
                    let g = ol / cfg.o_lora_rank;
                    let mut lora_v = 0.0;
                    for c in 0..r {
                        lora_v += w.wv_proj[ol * r + c]
                            * attn[t * cfg.n_heads * cfg.head_dim + g * r + c];
                    }
                    acc += w.wo_proj[hs * lora_len + ol] * lora_v;
                }
                assert_close(out[t * cfg.hidden_size + hs], acc, 1e-4, "project_o");
            }
        }
    }

    #[test]
    fn projections_validate_shapes() {
        let cfg = small_cfg(CacheFormat::Bf16, 8);
        let mut rng = Philox::new(13, 5);
        let w = MlaWeights::from_config(&cfg, &mut rng);
        let hidden = vec![0.1f32; 2 * cfg.hidden_size];
        let mut q = vec![0.0f32; 2 * cfg.n_heads * cfg.head_dim];
        let mut kv = vec![0.0f32; 2 * cfg.head_dim];
        assert!(w.project_q(&hidden, 3, &mut q).is_err());
        let qn = q.len() - 1;
        assert!(w.project_q(&hidden, 2, &mut q[..qn]).is_err());
        assert!(w.project_kv(&hidden, 3, &mut kv).is_err());
        let kn = kv.len() - 1;
        assert!(w.project_kv(&hidden, 2, &mut kv[..kn]).is_err());
        let mut o = vec![0.0f32; 2 * cfg.hidden_size];
        assert!(w.project_o(&q, 3, &mut o).is_err());
        let on = o.len() - 1;
        assert!(w.project_o(&q, 2, &mut o[..on]).is_err());
        let nan_hidden = vec![f32::NAN; cfg.hidden_size];
        assert!(w.project_q(&nan_hidden, 1, &mut q).is_err());
        assert!(w.project_kv(&nan_hidden, 1, &mut kv).is_err());
        assert!(w.project_o(&vec![f32::INFINITY; cfg.n_heads * cfg.head_dim], 1, &mut o).is_err());
    }

    #[test]
    fn derive_dims_rejects_inconsistent_weights() {
        let cfg = small_cfg(CacheFormat::Bf16, 8);
        let mut rng = Philox::new(17, 6);
        let w = MlaWeights::from_config(&cfg, &mut rng);
        // Empty kv_norm -> no derivable head_dim.
        let mut bad = w.clone();
        bad.kv_norm = Vec::new();
        assert!(bad.project_q(&vec![0.0; 32], 1, &mut vec![0.0; 4 * 64]).is_err());
        // kv_a_proj not divisible by head_dim.
        let mut bad = w.clone();
        bad.kv_a_proj.push(0.1);
        assert!(bad.project_kv(&vec![0.0; 32], 1, &mut vec![0.0; 64]).is_err());
        // o_groups that does not divide n_heads (wo len implies 3 groups).
        let mut bad = w.clone();
        bad.wo_proj = vec![0.01f32; cfg.hidden_size * 3 * cfg.o_lora_rank];
        assert!(bad.project_o(&vec![0.0; 4 * 64], 1, &mut vec![0.0; 32]).is_err());
        // q_norm length mismatch.
        let mut bad = w.clone();
        bad.q_norm = Some(vec![1.0f32; cfg.q_lora_rank + 1]);
        assert!(bad.project_q(&vec![0.0; 32], 1, &mut vec![0.0; 4 * 64]).is_err());
    }

    #[test]
    fn degenerate_configs_are_sanitized_not_panicking() {
        let mut weird = small_cfg(CacheFormat::Bf16, 8);
        weird.rope_dim = 63; // odd
        weird.head_dim = 10; // smaller than rope_dim
        weird.o_groups = 3; // does not divide n_heads
        weird.rope_base = 0.5; // invalid
        weird.n_heads = 0;
        weird.hidden_size = 0;
        let mut rng = Philox::new(1, 1);
        let w = MlaWeights::from_config(&weird, &mut rng);
        // Derived dims stay consistent: projections still work.
        let dm = w.derive_dims().unwrap();
        let hidden = random_vec(&mut rng, 2 * dm.hidden, 1.0);
        assert!(dm.head_dim > dm.rope_dim);
        assert!(dm.rope_dim % 2 == 0);
        assert_eq!(dm.n_heads % dm.o_groups, 0);
        let mut q = vec![0.0f32; 2 * dm.n_heads * dm.head_dim];
        w.project_q(&hidden, 2, &mut q).unwrap();
        assert!(q.iter().all(|v| v.is_finite()));
    }
}
