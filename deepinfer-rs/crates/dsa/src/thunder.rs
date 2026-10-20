//! The DSA "thunder" branch: semantic (content-based) scoring of
//! (query, key) pairs.
//!
//! This implements the exact semantics of DeepGEMM's V3.2
//! `fp8_fp4_mqa_logits` kernel (DeepGEMM README, "MQA logits" section, as
//! used by DeepSeek Sparse Attention — arXiv:2509.07152):
//!
//! ```text
//! score(i, j) = sum_h relu( qtilde_i^h . ktilde_j^h ) * w_i^h
//! ```
//!
//! where the ReLU is applied to the per-head dot product **before** the
//! per-head weighting and there is **no** `1/sqrt(d)` scaling. `qtilde` /
//! `ktilde` are the projected indexer vectors after:
//!
//! 1. per-(token, head) RMSNorm over the `head_dim` slice (eps 1e-6, no
//!    weight);
//! 2. "lightning quant": FP8 E4M3 quantization with E8M0 power-of-two
//!    scales per `quant_block` tokens along the sequence axis — one scale
//!    per block covering all heads and dims, computed as
//!    `2^ceil(log2(amax/448))` from the block's absolute maximum.
//!
//! Fidelity notes (documented modeling choices, per the port contract):
//! * Only COMPLETE `quant_block`-token chunks are quantized; a trailing
//!   partial chunk stays in f32. Consequently `quant_block > sequence
//!   length` disables quantization entirely — the hook used by the
//!   exactness tests.
//! * Upstream V3.2 normalizes the indexer q/k over the full index dim;
//!   this port normalizes each `head_dim` slice.

use deepinfer_core::format::{e4m3_to_f32, e8m0_to_f32, f32_to_e4m3_rne, f32_to_e8m0, E8M0Round};
use deepinfer_core::math::dot;
use deepinfer_core::norm::rms_mean_square;
use deepinfer_core::{ensure, Result};

use crate::{checked_dim, ensure_finite, NORM_EPS};

/// Largest finite E4M3 magnitude — the quantization target range.
const E4M3_MAX: f32 = 448.0;

/// Computes the thunder-branch score matrix `[s_q, s_kv]`.
///
/// * `q`: `[s_q, num_heads*head_dim]` projected indexer queries.
/// * `k`: `[s_kv, num_heads*head_dim]` projected indexer keys.
/// * `w`: `[s_q, num_heads]` per-query head weights (`weights_proj @ q_i`).
/// * `quant_block`: lightning-quant block size in tokens (>= 1).
/// * `causal`: when `true`, entries with `j > i` are set to `-inf` (the
///   causal clip; `j == i` is kept — the V3.2 "current token" fix).
///
/// Determinism: fixed sequential f32 accumulation (heads in ascending
/// order, `deepinfer_core::math::dot` over the head slice), so the same
/// inputs always produce bit-identical output.
pub fn thunder_scores(
    q: &[f32],
    k: &[f32],
    w: &[f32],
    s_q: usize,
    s_kv: usize,
    num_heads: usize,
    head_dim: usize,
    quant_block: usize,
    causal: bool,
) -> Result<Vec<f32>> {
    ensure!(
        num_heads >= 1,
        InvalidArgument,
        "num_heads must be >= 1 (got {num_heads})"
    );
    ensure!(
        head_dim >= 1,
        InvalidArgument,
        "head_dim must be >= 1 (got {head_dim})"
    );
    ensure!(
        quant_block >= 1,
        InvalidArgument,
        "quant_block must be >= 1 (got {quant_block})"
    );
    let index_dim = checked_dim(num_heads, head_dim, "index dim")?;
    let q_len = checked_dim(s_q, index_dim, "q length")?;
    ensure!(
        q.len() == q_len,
        DimensionMismatch,
        "q has {} elements, expected s_q*index_dim = {q_len}",
        q.len()
    );
    let k_len = checked_dim(s_kv, index_dim, "k length")?;
    ensure!(
        k.len() == k_len,
        DimensionMismatch,
        "k has {} elements, expected s_kv*index_dim = {k_len}",
        k.len()
    );
    let w_len = checked_dim(s_q, num_heads, "w length")?;
    ensure!(
        w.len() == w_len,
        DimensionMismatch,
        "w has {} elements, expected s_q*num_heads = {w_len}",
        w.len()
    );
    ensure_finite(q, "q")?;
    ensure_finite(k, "k")?;
    ensure_finite(w, "w")?;

    // RMSNorm each (token, head) slice, then lightning-quantize per block of
    // tokens along the sequence axis.
    let mut qt = q.to_vec();
    let mut kt = k.to_vec();
    rmsnorm_per_head(&mut qt, s_q, num_heads, head_dim);
    rmsnorm_per_head(&mut kt, s_kv, num_heads, head_dim);
    lightning_quant(&mut qt, s_q, index_dim, quant_block)?;
    lightning_quant(&mut kt, s_kv, index_dim, quant_block)?;

    let out_len = checked_dim(s_q, s_kv, "score matrix length")?;
    let mut out = vec![0.0f32; out_len];
    for i in 0..s_q {
        for j in 0..s_kv {
            let mut acc = 0.0f32;
            for h in 0..num_heads {
                let qs = &qt[i * index_dim + h * head_dim..i * index_dim + (h + 1) * head_dim];
                let ks = &kt[j * index_dim + h * head_dim..j * index_dim + (h + 1) * head_dim];
                let d = dot(qs, ks);
                acc += (if d > 0.0 { d } else { 0.0 }) * w[i * num_heads + h];
            }
            out[i * s_kv + j] = acc;
        }
    }
    if causal {
        for i in 0..s_q {
            for j in (i + 1)..s_kv {
                out[i * s_kv + j] = f32::NEG_INFINITY;
            }
        }
    }
    Ok(out)
}

/// RMSNorms every `(token, head)` slice of `x` (layout `[rows, nh*hd]`).
fn rmsnorm_per_head(x: &mut [f32], rows: usize, num_heads: usize, head_dim: usize) {
    let d = num_heads * head_dim;
    for r in 0..rows {
        for h in 0..num_heads {
            let s = &mut x[r * d + h * head_dim..r * d + (h + 1) * head_dim];
            let inv = 1.0 / (rms_mean_square(s) + NORM_EPS).sqrt();
            for v in s.iter_mut() {
                *v *= inv;
            }
        }
    }
}

/// Lightning quantization: FP8 E4M3 with one E8M0 scale per COMPLETE
/// `quant_block`-token chunk (the scale covers all heads and dims of the
/// chunk). Trailing partial chunks are left in f32.
fn lightning_quant(x: &mut [f32], rows: usize, dim: usize, quant_block: usize) -> Result<()> {
    for b in 0..rows / quant_block {
        let start = b * quant_block * dim;
        let end = start + quant_block * dim;
        let blk = &mut x[start..end];
        let mut amax = 0.0f32;
        for &v in blk.iter() {
            let a = v.abs();
            if a > amax {
                amax = a;
            }
        }
        let ratio = amax / E4M3_MAX;
        if ratio > 0.0 {
            let scale = e8m0_to_f32(f32_to_e8m0(ratio, E8M0Round::Ceil)?);
            for v in blk.iter_mut() {
                *v = e4m3_to_f32(f32_to_e4m3_rne(*v / scale, true)) * scale;
            }
        } else {
            // All-zero block, or values so small that amax/448 underflowed:
            // nothing on the E8M0-scaled E4M3 lattice can represent them.
            for v in blk.iter_mut() {
                *v = 0.0;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepinfer_core::prop::{prop, random_vec};
    use deepinfer_core::rng::Philox;

    /// Test-side re-implementation of the lightning quant (same semantics:
    /// complete blocks only, E8M0 ceil scale, E4M3 RNE with saturation).
    fn naive_quant(x: &[f32], rows: usize, dim: usize, quant_block: usize) -> Vec<f32> {
        let mut out = x.to_vec();
        for b in 0..rows / quant_block {
            let s = b * quant_block * dim;
            let e = s + quant_block * dim;
            let mut amax = 0.0f32;
            for &v in &out[s..e] {
                let a = v.abs();
                if a > amax {
                    amax = a;
                }
            }
            let ratio = amax / 448.0;
            if ratio > 0.0 {
                let scale = e8m0_to_f32(f32_to_e8m0(ratio, E8M0Round::Ceil).unwrap());
                for v in out[s..e].iter_mut() {
                    *v = e4m3_to_f32(f32_to_e4m3_rne(*v / scale, true)) * scale;
                }
            } else {
                for v in out[s..e].iter_mut() {
                    *v = 0.0;
                }
            }
        }
        out
    }

    /// Naive double-loop transcription of the thunder semantics.
    fn naive_thunder(
        q: &[f32],
        k: &[f32],
        w: &[f32],
        s_q: usize,
        s_kv: usize,
        nh: usize,
        hd: usize,
        quant_block: usize,
        causal: bool,
    ) -> Vec<f32> {
        let d = nh * hd;
        let norm = |x: &[f32], rows: usize| -> Vec<f32> {
            let mut t = x.to_vec();
            for r in 0..rows {
                for h in 0..nh {
                    let s = &mut t[r * d + h * hd..r * d + (h + 1) * hd];
                    let inv = 1.0 / (rms_mean_square(s) + 1e-6).sqrt();
                    for v in s.iter_mut() {
                        *v *= inv;
                    }
                }
            }
            t
        };
        let mut qt = norm(q, s_q);
        let mut kt = norm(k, s_kv);
        qt = naive_quant(&qt, s_q, d, quant_block);
        kt = naive_quant(&kt, s_kv, d, quant_block);
        let mut out = vec![0.0f32; s_q * s_kv];
        for i in 0..s_q {
            for j in 0..s_kv {
                let mut acc = 0.0f32;
                for h in 0..nh {
                    let dv = dot(
                        &qt[i * d + h * hd..i * d + (h + 1) * hd],
                        &kt[j * d + h * hd..j * d + (h + 1) * hd],
                    );
                    acc += (if dv > 0.0 { dv } else { 0.0 }) * w[i * nh + h];
                }
                out[i * s_kv + j] = acc;
            }
        }
        if causal {
            for i in 0..s_q {
                for j in (i + 1)..s_kv {
                    out[i * s_kv + j] = f32::NEG_INFINITY;
                }
            }
        }
        out
    }

    #[test]
    fn relu_applied_before_weight() {
        // nh = 2, hd = 1, single (i, j); quantization disabled (quant_block > s_kv).
        // Head 0: q.k < 0 -> relu(dot) = 0, so the weight -10 must NOT
        // resurrect it. Head 1: unit vectors after RMSNorm -> dot ~ 1.
        let q = [2.0f32, 3.0];
        let k = [-1.0f32, 2.0];
        let w = [-10.0f32, 0.5];
        let out = thunder_scores(&q, &k, &w, 1, 1, 2, 1, 8, false).unwrap();
        assert!((out[0] - 0.5).abs() < 1e-4, "got {}", out[0]);
        // relu-after-weight instead would give ~10.5.
    }

    #[test]
    fn matches_naive_without_quant() {
        // quant_block > s_kv disables lightning quantization entirely.
        let mut rng = Philox::new(0x7A, 1);
        let (s_q, s_kv, nh, hd) = (3, 5, 4, 3);
        let q = random_vec(&mut rng, s_q * nh * hd, 2.0);
        let k = random_vec(&mut rng, s_kv * nh * hd, 2.0);
        let w = random_vec(&mut rng, s_q * nh, 1.0);
        let got = thunder_scores(&q, &k, &w, s_q, s_kv, nh, hd, 64, false).unwrap();
        let want = naive_thunder(&q, &k, &w, s_q, s_kv, nh, hd, 64, false);
        assert_eq!(got, want, "bit-identical: same operation order");
    }

    #[test]
    fn matches_naive_with_quant() {
        let mut rng = Philox::new(0x7A, 2);
        let (s_q, s_kv, nh, hd) = (4, 7, 3, 2);
        let q = random_vec(&mut rng, s_q * nh * hd, 2.0);
        let k = random_vec(&mut rng, s_kv * nh * hd, 2.0);
        let w = random_vec(&mut rng, s_q * nh, 1.0);
        // quant_block = 2 (with a trailing partial block) and 1 (per token).
        for qb in [1usize, 2, 3, 7] {
            let got = thunder_scores(&q, &k, &w, s_q, s_kv, nh, hd, qb, true).unwrap();
            let want = naive_thunder(&q, &k, &w, s_q, s_kv, nh, hd, qb, true);
            assert_eq!(got, want, "quant_block = {qb}");
        }
    }

    #[test]
    fn causal_clip_pattern() {
        let mut rng = Philox::new(0x7A, 3);
        let (s_q, s_kv, nh, hd) = (4, 6, 2, 3);
        let q = random_vec(&mut rng, s_q * nh * hd, 2.0);
        let k = random_vec(&mut rng, s_kv * nh * hd, 2.0);
        let w = random_vec(&mut rng, s_q * nh, 1.0);
        let causal = thunder_scores(&q, &k, &w, s_q, s_kv, nh, hd, 64, true).unwrap();
        let full = thunder_scores(&q, &k, &w, s_q, s_kv, nh, hd, 64, false).unwrap();
        for i in 0..s_q {
            for j in 0..s_kv {
                if j > i {
                    assert_eq!(causal[i * s_kv + j], f32::NEG_INFINITY, "({i},{j})");
                } else {
                    // j == i included (V3.2 "current token" fix).
                    assert_eq!(causal[i * s_kv + j], full[i * s_kv + j], "({i},{j})");
                }
            }
        }
    }

    #[test]
    fn deterministic() {
        let mut rng = Philox::new(0x7A, 4);
        let (s_q, s_kv, nh, hd) = (2, 4, 3, 2);
        let q = random_vec(&mut rng, s_q * nh * hd, 2.0);
        let k = random_vec(&mut rng, s_kv * nh * hd, 2.0);
        let w = random_vec(&mut rng, s_q * nh, 1.0);
        let a = thunder_scores(&q, &k, &w, s_q, s_kv, nh, hd, 2, true).unwrap();
        let b = thunder_scores(&q, &k, &w, s_q, s_kv, nh, hd, 2, true).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn validates_inputs() {
        assert!(thunder_scores(&[1.0], &[1.0, 1.0], &[1.0, 1.0], 1, 1, 2, 1, 8, false).is_err());
        assert!(thunder_scores(&[1.0, 1.0], &[1.0], &[1.0, 1.0], 1, 1, 2, 1, 8, false).is_err());
        assert!(thunder_scores(&[1.0, 1.0], &[1.0, 1.0], &[1.0], 1, 1, 2, 1, 8, false).is_err());
        assert!(
            thunder_scores(&[1.0, 1.0], &[1.0, 1.0], &[1.0, 1.0], 1, 1, 0, 1, 8, false).is_err()
        );
        assert!(
            thunder_scores(&[1.0, 1.0], &[1.0, 1.0], &[1.0, 1.0], 1, 1, 1, 1, 0, false).is_err()
        );
        assert!(thunder_scores(
            &[f32::NAN, 1.0],
            &[1.0, 1.0],
            &[1.0, 1.0],
            1,
            1,
            2,
            1,
            8,
            false
        )
        .is_err());
    }

    #[test]
    fn property_matches_naive() {
        prop("dsa_thunder_matches_naive", 0x7A2, 120, |_i, rng| {
            let s_q = rng.uniform_usize(1, 5);
            let s_kv = rng.uniform_usize(1, 7);
            let nh = rng.uniform_usize(1, 4);
            let hd = rng.uniform_usize(1, 4);
            let q = random_vec(rng, s_q * nh * hd, 3.0);
            let k = random_vec(rng, s_kv * nh * hd, 3.0);
            let w = random_vec(rng, s_q * nh, 2.0);
            let causal = rng.next_f32() < 0.5;
            let qb = rng.uniform_usize(1, 4);
            let got = thunder_scores(&q, &k, &w, s_q, s_kv, nh, hd, qb, causal)?;
            let want = naive_thunder(&q, &k, &w, s_q, s_kv, nh, hd, qb, causal);
            ensure!(got == want, Internal, "thunder mismatch (qb={qb}, causal={causal})");
            Ok(())
        });
    }

    #[test]
    fn property_causal_clip() {
        prop("dsa_thunder_causal_clip", 0x7A3, 120, |_i, rng| {
            let s_q = rng.uniform_usize(1, 5);
            let s_kv = rng.uniform_usize(1, 7);
            let nh = rng.uniform_usize(1, 3);
            let hd = rng.uniform_usize(1, 3);
            let q = random_vec(rng, s_q * nh * hd, 3.0);
            let k = random_vec(rng, s_kv * nh * hd, 3.0);
            let w = random_vec(rng, s_q * nh, 2.0);
            let causal = thunder_scores(&q, &k, &w, s_q, s_kv, nh, hd, 64, true)?;
            let full = thunder_scores(&q, &k, &w, s_q, s_kv, nh, hd, 64, false)?;
            for i in 0..s_q {
                for j in 0..s_kv {
                    let c = causal[i * s_kv + j];
                    if j > i {
                        ensure!(c == f32::NEG_INFINITY, Internal, "({i},{j}) not clipped");
                    } else {
                        ensure!(c == full[i * s_kv + j], Internal, "({i},{j}) differs");
                    }
                }
            }
            Ok(())
        });
    }
}
