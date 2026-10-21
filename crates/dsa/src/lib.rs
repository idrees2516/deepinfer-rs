//! # deepinfer-dsa
//!
//! DeepSeek Sparse Attention (DSA) lightning indexer — the CPU-reference
//! port of the indexer described in the DSA paper (arXiv:2509.07152) and
//! shipped with DeepSeek-V3.2 (see also DeepGEMM's `fp8_fp4_mqa_logits`
//! kernel semantics and FlashMLA's DSA sections).
//!
//! The indexer scores every (query, key) pair with two complementary
//! branches and keeps only the top `topk_tokens` keys per query:
//!
//! * **thunder** ([`thunder_scores`]) — the semantic branch with the exact
//!   semantics of DeepGEMM's V3.2 MQA-logits kernel:
//!   `score(i,j) = sum_h relu(qtilde_i^h . ktilde_j^h) * w_i^h`, computed
//!   on RMSNormed, lightning-FP8-quantized indexer vectors (ReLU applied
//!   before weighting, no `1/sqrt(d)` scaling);
//! * **cascade conv** ([`conv_scores`]) — the spatial branch: causal
//!   depthwise convolutions over the key sequence followed by a gated
//!   RMSNorm;
//! * **coefficient gating** ([`gate_and_combine`]) — per-query sigmoid
//!   gates blend the two branches;
//! * **selection** ([`select_top_k`]) — block-wise pre-selection followed
//!   by a global top-k with deterministic tie-breaking, plus the V3.2
//!   [`mscale`] constant and context extrapolation.
//!
//! [`indexer_forward`] chains all stages end-to-end. The causal clip keeps
//! keys `j <= i` (the V3.2 "current token" fix); masked entries carry the
//! `-inf` score sentinel and are never selected.
//!
//! ## Fidelity notes / modeling choices
//!
//! * RMSNorm granularity is per (token, head) `head_dim` slice (eps 1e-6,
//!   no weight), per the port contract; upstream V3.2 normalizes over the
//!   full index dim.
//! * Lightning quantization covers only COMPLETE `quant_block`-token
//!   chunks along the sequence axis (one E8M0 scale per block over all
//!   heads/dims, `amax/448` rounded up); trailing partial chunks stay in
//!   f32, so `quant_block > sequence length` disables quantization
//!   entirely.
//! * The conv branch's gated-RMSNorm gain is derived from the key
//!   projection (`wk` row RMS) because the SPEC-mandated weight struct has
//!   no dedicated `conv_norm_gain` field.
//! * Gating normalizes by `g_thunder + g_conv` (falling back to the thunder
//!   score when both gates underflow to zero); `-inf` (masked) entries
//!   propagate through the blend.
//! * Score matrices are validated to be finite or exactly `-inf`; `NaN`
//!   and `+inf` are rejected as numerical overflow.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod config;
pub mod conv;
pub mod gate;
pub mod select;
pub mod thunder;
pub mod weights;

pub use config::LightningIndexerConfig;
pub use conv::conv_scores;
pub use gate::gate_and_combine;
pub use select::{mscale, select_top_k};
pub use thunder::thunder_scores;
pub use weights::LightningIndexerWeights;

use deepinfer_core::math::dot;
use deepinfer_core::{ensure, Error, Result};

/// RMSNorm epsilon shared by the indexer normalizations (the thunder
/// per-head norm and the conv branch's gated RMSNorm).
pub(crate) const NORM_EPS: f32 = 1e-6;

/// Numerically stable logistic function.
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Checked multiplication for allocation-size math (DoS guard).
pub(crate) fn checked_dim(a: usize, b: usize, what: &str) -> Result<usize> {
    a.checked_mul(b).ok_or_else(|| {
        Error::CapacityExceeded(format!("{what}: {a} * {b} overflows usize"))
    })
}

/// Ensures every element of `x` is finite.
pub(crate) fn ensure_finite(x: &[f32], name: &str) -> Result<()> {
    for (i, &v) in x.iter().enumerate() {
        ensure!(v.is_finite(), Numerical, "{name}[{i}] is not finite ({v})");
    }
    Ok(())
}

/// Ensures every element is finite or exactly `-inf` (the causal-mask
/// sentinel).
pub(crate) fn ensure_scores_valid(x: &[f32], name: &str) -> Result<()> {
    for (i, &v) in x.iter().enumerate() {
        ensure!(
            v.is_finite() || v == f32::NEG_INFINITY,
            Numerical,
            "{name}[{i}] must be finite or -inf (got {v})"
        );
    }
    Ok(())
}

/// End-to-end lightning-indexer forward pass.
///
/// Projects `q_hidden` (`[s_q, hidden]`) and `k_hidden` (`[s_kv, hidden]`)
/// through `wq`/`wk` into index-dim vectors, computes the per-query head
/// weights `w_i = weights_proj @ q_i`, scores with the thunder and conv
/// branches (both causally clipped), blends them with coefficient gating
/// (driven by the RAW hidden queries and their sequence mean), and selects
/// the top-k key positions with [`select_top_k`] (including any
/// `extrapolation_window` restriction).
///
/// Returns `[s_q, K]` with `K = min(topk_tokens, s_kv)`: per query, the
/// selected key positions sorted ascending, padded with `-1`.
pub fn indexer_forward(
    w: &LightningIndexerWeights,
    cfg: &LightningIndexerConfig,
    q_hidden: &[f32],
    k_hidden: &[f32],
    s_q: usize,
    s_kv: usize,
) -> Result<Vec<i32>> {
    ensure!(cfg.hidden_size >= 1, InvalidArgument, "hidden_size must be >= 1");
    ensure!(cfg.num_heads >= 1, InvalidArgument, "num_heads must be >= 1");
    ensure!(cfg.head_dim >= 1, InvalidArgument, "head_dim must be >= 1");
    ensure!(cfg.quant_block >= 1, InvalidArgument, "quant_block must be >= 1");
    if let Some(xw) = cfg.extrapolation_window {
        ensure!(xw >= 1, InvalidArgument, "extrapolation_window must be >= 1");
    }
    let hidden = cfg.hidden_size;
    let index_dim = checked_dim(cfg.num_heads, cfg.head_dim, "index dim")?;
    let proj_len = checked_dim(index_dim, hidden, "wq/wk length")?;
    let wp_len = checked_dim(cfg.num_heads, index_dim, "weights_proj length")?;
    let gate_len = checked_dim(hidden, 4, "gate_proj length")?;
    ensure!(
        w.wq.len() == proj_len,
        DimensionMismatch,
        "wq has {} elements, expected index_dim*hidden = {proj_len}",
        w.wq.len()
    );
    ensure!(
        w.wk.len() == proj_len,
        DimensionMismatch,
        "wk has {} elements, expected index_dim*hidden = {proj_len}",
        w.wk.len()
    );
    ensure!(
        w.weights_proj.len() == wp_len,
        DimensionMismatch,
        "weights_proj has {} elements, expected num_heads*index_dim = {wp_len}",
        w.weights_proj.len()
    );
    ensure!(
        w.gate_proj.len() == gate_len,
        DimensionMismatch,
        "gate_proj has {} elements, expected 2*2*hidden = {gate_len}",
        w.gate_proj.len()
    );
    ensure!(
        w.conv_weights.len() == cfg.conv_kernels.len(),
        DimensionMismatch,
        "conv_weights has {} cascades, config lists {}",
        w.conv_weights.len(),
        cfg.conv_kernels.len()
    );
    for (c, &ks) in cfg.conv_kernels.iter().enumerate() {
        ensure!(
            w.conv_weights[c].len() == ks,
            DimensionMismatch,
            "conv_weights[{c}] has {} taps, expected {ks}",
            w.conv_weights[c].len()
        );
    }
    let q_len = checked_dim(s_q, hidden, "q_hidden length")?;
    ensure!(
        q_hidden.len() == q_len,
        DimensionMismatch,
        "q_hidden has {} elements, expected s_q*hidden = {q_len}",
        q_hidden.len()
    );
    let k_len = checked_dim(s_kv, hidden, "k_hidden length")?;
    ensure!(
        k_hidden.len() == k_len,
        DimensionMismatch,
        "k_hidden has {} elements, expected s_kv*hidden = {k_len}",
        k_hidden.len()
    );
    ensure_finite(&w.wq, "wq")?;
    ensure_finite(&w.wk, "wk")?;
    ensure_finite(&w.weights_proj, "weights_proj")?;
    ensure_finite(&w.gate_proj, "gate_proj")?;
    for (c, kw) in w.conv_weights.iter().enumerate() {
        ensure_finite(kw, &format!("conv_weights[{c}]"))?;
    }
    ensure_finite(q_hidden, "q_hidden")?;
    ensure_finite(k_hidden, "k_hidden")?;

    // Projections: q = q_hidden @ wq^T, k = k_hidden @ wk^T.
    let mut q = vec![0.0f32; checked_dim(s_q, index_dim, "q length")?];
    for i in 0..s_q {
        let x = &q_hidden[i * hidden..(i + 1) * hidden];
        for r in 0..index_dim {
            q[i * index_dim + r] = dot(&w.wq[r * hidden..r * hidden + hidden], x);
        }
    }
    let mut k = vec![0.0f32; checked_dim(s_kv, index_dim, "k length")?];
    for j in 0..s_kv {
        let x = &k_hidden[j * hidden..(j + 1) * hidden];
        for r in 0..index_dim {
            k[j * index_dim + r] = dot(&w.wk[r * hidden..r * hidden + hidden], x);
        }
    }

    // Per-query head weights: w_i = weights_proj @ q_i.
    let mut hw = vec![0.0f32; checked_dim(s_q, cfg.num_heads, "head weights length")?];
    for i in 0..s_q {
        let qi = &q[i * index_dim..(i + 1) * index_dim];
        for h in 0..cfg.num_heads {
            hw[i * cfg.num_heads + h] =
                dot(&w.weights_proj[h * index_dim..h * index_dim + index_dim], qi);
        }
    }

    // Thunder + conv branches (causally clipped), then coefficient gating.
    let mut thunder = thunder_scores(
        &q,
        &k,
        &hw,
        s_q,
        s_kv,
        cfg.num_heads,
        cfg.head_dim,
        cfg.quant_block,
        true,
    )?;
    let gain = weights::conv_norm_gain_from_wk(&w.wk, index_dim, hidden);
    let conv = conv_scores(&q, &k, s_q, s_kv, index_dim, &w.conv_weights, &gain, true)?;
    gate_and_combine(q_hidden, s_q, hidden, &w.gate_proj, &mut thunder, &conv)?;

    select_top_k(&thunder, s_q, s_kv, cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepinfer_core::prop::{prop, random_vec};
    use deepinfer_core::rng::Philox;

    fn small_cfg() -> LightningIndexerConfig {
        LightningIndexerConfig {
            hidden_size: 16,
            num_heads: 4,
            head_dim: 6,
            block_size: 8,
            topk_tokens: 10,
            preselect_per_block: 3,
            quant_block: 5,
            conv_kernels: vec![3, 5],
            extrapolation_window: Some(12),
        }
    }

    #[test]
    fn forward_matches_componentwise() {
        let cfg = small_cfg();
        let mut rng = Philox::new(0xD5A, 3);
        let w = LightningIndexerWeights::from_config(&cfg, &mut rng);
        let (s_q, s_kv) = (7, 9);
        let qh = random_vec(&mut rng, s_q * cfg.hidden_size, 1.0);
        let kh = random_vec(&mut rng, s_kv * cfg.hidden_size, 1.0);
        let got = indexer_forward(&w, &cfg, &qh, &kh, s_q, s_kv).unwrap();

        // Component-wise replication through the public API.
        let d = cfg.num_heads * cfg.head_dim;
        let hidden = cfg.hidden_size;
        let mut q = vec![0.0f32; s_q * d];
        for i in 0..s_q {
            for r in 0..d {
                q[i * d + r] = dot(&w.wq[r * hidden..r * hidden + hidden], &qh[i * hidden..(i + 1) * hidden]);
            }
        }
        let mut k = vec![0.0f32; s_kv * d];
        for j in 0..s_kv {
            for r in 0..d {
                k[j * d + r] = dot(&w.wk[r * hidden..r * hidden + hidden], &kh[j * hidden..(j + 1) * hidden]);
            }
        }
        let mut hw = vec![0.0f32; s_q * cfg.num_heads];
        for i in 0..s_q {
            for h in 0..cfg.num_heads {
                hw[i * cfg.num_heads + h] =
                    dot(&w.weights_proj[h * d..h * d + d], &q[i * d..(i + 1) * d]);
            }
        }
        let mut thunder =
            thunder_scores(&q, &k, &hw, s_q, s_kv, cfg.num_heads, cfg.head_dim, cfg.quant_block, true)
                .unwrap();
        let gain = weights::conv_norm_gain_from_wk(&w.wk, d, hidden);
        let conv = conv_scores(&q, &k, s_q, s_kv, d, &w.conv_weights, &gain, true).unwrap();
        gate_and_combine(&qh, s_q, hidden, &w.gate_proj, &mut thunder, &conv).unwrap();
        let want = select_top_k(&thunder, s_q, s_kv, &cfg).unwrap();
        assert_eq!(got, want);
    }

    #[test]
    fn forward_output_invariants() {
        let cfg = small_cfg();
        let mut rng = Philox::new(0xD5A, 4);
        let w = LightningIndexerWeights::from_config(&cfg, &mut rng);
        let (s_q, s_kv) = (6, 11); // s_q < s_kv (MTP-style draft block)
        let qh = random_vec(&mut rng, s_q * cfg.hidden_size, 1.0);
        let kh = random_vec(&mut rng, s_kv * cfg.hidden_size, 1.0);
        let out = indexer_forward(&w, &cfg, &qh, &kh, s_q, s_kv).unwrap();
        let k_out = cfg.topk_tokens.min(s_kv);
        assert_eq!(out.len(), s_q * k_out);
        for i in 0..s_q {
            let row = &out[i * k_out..(i + 1) * k_out];
            let split = row.iter().position(|&v| v == -1).unwrap_or(row.len());
            let (valid, pad) = (&row[..split], &row[split..]);
            assert!(pad.iter().all(|&v| v == -1));
            assert!(valid.windows(2).all(|w| w[0] < w[1]), "row {i}: {valid:?}");
            for &p in valid {
                let p = p as usize;
                assert!(p < s_kv);
                assert!(p <= i, "causal violation: key {p} for query {i}");
                // extrapolation window of 12 covers everything here (s_kv small)
                assert!(p + cfg.extrapolation_window.unwrap() > i);
            }
        }
        // Determinism.
        let again = indexer_forward(&w, &cfg, &qh, &kh, s_q, s_kv).unwrap();
        assert_eq!(out, again);
    }

    #[test]
    fn forward_v32_default_smoke() {
        let cfg = LightningIndexerConfig::v32_default(64);
        let mut rng = Philox::new(1, 2);
        let w = LightningIndexerWeights::from_config(&cfg, &mut rng);
        let s = 4usize;
        let qh = random_vec(&mut rng, s * 64, 1.0);
        let kh = random_vec(&mut rng, s * 64, 1.0);
        let out = indexer_forward(&w, &cfg, &qh, &kh, s, s).unwrap();
        assert_eq!(out.len(), s * s); // K = min(2048, 4) = 4
        for i in 0..s {
            let row = &out[i * s..(i + 1) * s];
            let valid = row.iter().filter(|&&v| v != -1).count();
            assert_eq!(valid, i + 1, "row {i}: {row:?}"); // causal, single block
        }
        let again = indexer_forward(&w, &cfg, &qh, &kh, s, s).unwrap();
        assert_eq!(out, again);
    }

    #[test]
    fn forward_validates_inputs() {
        let cfg = small_cfg();
        let mut rng = Philox::new(2, 2);
        let w = LightningIndexerWeights::from_config(&cfg, &mut rng);
        let qh = vec![0.0f32; 2 * cfg.hidden_size];
        let kh = vec![0.0f32; 2 * cfg.hidden_size];
        assert!(indexer_forward(&w, &cfg, &qh, &kh, 2, 2).is_ok());
        assert!(indexer_forward(&w, &cfg, &qh, &kh, 3, 2).is_err()); // q_hidden len
        assert!(indexer_forward(&w, &cfg, &qh, &kh, 2, 3).is_err()); // k_hidden len
        let mut bad = LightningIndexerWeights::from_config(&cfg, &mut rng);
        bad.wq.pop();
        assert!(indexer_forward(&bad, &cfg, &qh, &kh, 2, 2).is_err());
        let mut bad_cfg = cfg.clone();
        bad_cfg.quant_block = 0;
        assert!(indexer_forward(&w, &bad_cfg, &qh, &kh, 2, 2).is_err());
        let nan_qh = vec![f32::NAN; 2 * cfg.hidden_size];
        assert!(indexer_forward(&w, &cfg, &nan_qh, &kh, 2, 2).is_err());
    }

    #[test]
    fn property_forward_deterministic_and_causal() {
        prop("dsa_indexer_forward", 0xD5B, 100, |_i, rng| {
            let hidden = rng.uniform_usize(2, 6);
            let cfg = LightningIndexerConfig {
                hidden_size: hidden,
                num_heads: rng.uniform_usize(1, 3),
                head_dim: rng.uniform_usize(1, 4),
                block_size: rng.uniform_usize(1, 6),
                topk_tokens: rng.uniform_usize(1, 7),
                preselect_per_block: rng.uniform_usize(1, 4),
                quant_block: rng.uniform_usize(1, 4),
                conv_kernels: vec![rng.uniform_usize(1, 4), rng.uniform_usize(1, 4)],
                extrapolation_window: if rng.next_f32() < 0.5 {
                    Some(rng.uniform_usize(1, 8))
                } else {
                    None
                },
            };
            let mut wrng = Philox::new(0xD00D, _i as u64);
            let w = LightningIndexerWeights::from_config(&cfg, &mut wrng);
            let s_q = rng.uniform_usize(1, 5);
            let s_kv = rng.uniform_usize(1, 8);
            let qh = random_vec(rng, s_q * hidden, 1.0);
            let kh = random_vec(rng, s_kv * hidden, 1.0);
            let a = indexer_forward(&w, &cfg, &qh, &kh, s_q, s_kv)?;
            let b = indexer_forward(&w, &cfg, &qh, &kh, s_q, s_kv)?;
            ensure!(a == b, Internal, "non-deterministic");
            let k_out = cfg.topk_tokens.min(s_kv);
            ensure!(a.len() == s_q * k_out, Internal, "output length");
            for i in 0..s_q {
                let row = &a[i * k_out..(i + 1) * k_out];
                let split = row.iter().position(|&v| v == -1).unwrap_or(row.len());
                ensure!(
                    row[..split].windows(2).all(|w| w[0] < w[1]),
                    Internal,
                    "row {i} not ascending"
                );
                ensure!(row[split..].iter().all(|&v| v == -1), Internal, "pad");
                for &p in &row[..split] {
                    ensure!(p >= 0 && (p as usize) < s_kv, Internal, "range");
                    ensure!(p as usize <= i, Internal, "causal");
                    if let Some(xw) = cfg.extrapolation_window {
                        ensure!(p as usize + xw > i, Internal, "extrapolation");
                    }
                }
            }
            Ok(())
        });
    }
}
