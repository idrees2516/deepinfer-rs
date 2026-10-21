//! The DSA "cascade conv" branch: spatial (locality-based) scoring.
//!
//! DeepSeek Sparse Attention (arXiv:2509.07152, "Lightning Indexer")
//! complements the semantic thunder branch with a cascade of CAUSAL
//! depthwise convolutions over the key sequence, capturing local n-gram
//! patterns that pure content matching misses.
//!
//! For each cascade kernel `kw` of size `ks` the feature accumulator is
//! the causal, zero-padded moving sum
//!
//! ```text
//! convfeat_j[d] += sum_{t=0..ks-1} kw[t] * k_{j-t}[d]
//! ```
//!
//! The indexer key is one vector per token, so each cascade is depthwise
//! with a SINGLE channel: the taps are scalars shared across dims. The
//! cascade outputs are summed elementwise. The accumulated features are
//! then normalized with a gated RMSNorm
//!
//! ```text
//! convfeat_j = convfeat_j * sigmoid(norm_gain . convfeat_j)
//!              / sqrt(mean(convfeat_j^2) + 1e-6)
//! ```
//!
//! (modeling choice: `norm_gain` is a per-dim learnable vector — upstream's
//! dedicated `conv_norm_gain`; in the end-to-end path it is derived from the
//! key projection, see [`crate::weights`]) and scored as
//! `score(i, j) = q_i . convfeat_j`.

use deepinfer_core::math::dot;
use deepinfer_core::norm::rms_mean_square;
use deepinfer_core::{ensure, Result};

use crate::{checked_dim, ensure_finite, sigmoid, NORM_EPS};

/// Computes the conv-branch score matrix `[s_q, s_kv]`.
///
/// * `q`: `[s_q, hidden]` projected indexer queries — `hidden` here is the
///   row width of `q`/`k` (in the end-to-end indexer this is the index dim
///   `num_heads * head_dim`).
/// * `k`: `[s_kv, hidden]` projected indexer keys the convolutions run over.
/// * `kernels`: one tap-weight vector per cascade entry (`kernels[c].len()`
///   is that cascade's kernel size); taps are scalars shared across dims.
/// * `norm_gain`: per-dim gated-RMSNorm gain `[hidden]`.
/// * `causal`: when `true`, entries with `j > i` are set to `-inf` (`j == i`
///   is kept).
///
/// Determinism: features accumulate cascade-major, then tap, then dim; all
/// reductions are sequential f32.
pub fn conv_scores(
    q: &[f32],
    k: &[f32],
    s_q: usize,
    s_kv: usize,
    hidden: usize,
    kernels: &[Vec<f32>],
    norm_gain: &[f32],
    causal: bool,
) -> Result<Vec<f32>> {
    ensure!(
        hidden >= 1,
        InvalidArgument,
        "hidden (feature dim) must be >= 1 (got {hidden})"
    );
    ensure!(
        norm_gain.len() == hidden,
        DimensionMismatch,
        "norm_gain has {} entries, expected {hidden}",
        norm_gain.len()
    );
    for (c, kw) in kernels.iter().enumerate() {
        ensure!(!kw.is_empty(), InvalidArgument, "conv kernel {c} is empty");
    }
    let q_len = checked_dim(s_q, hidden, "q length")?;
    ensure!(
        q.len() == q_len,
        DimensionMismatch,
        "q has {} elements, expected s_q*hidden = {q_len}",
        q.len()
    );
    let k_len = checked_dim(s_kv, hidden, "k length")?;
    ensure!(
        k.len() == k_len,
        DimensionMismatch,
        "k has {} elements, expected s_kv*hidden = {k_len}",
        k.len()
    );
    ensure_finite(q, "q")?;
    ensure_finite(k, "k")?;
    ensure_finite(norm_gain, "norm_gain")?;
    for (c, kw) in kernels.iter().enumerate() {
        ensure_finite(kw, &format!("conv kernel {c}"))?;
    }

    // Cascade causal depthwise convolutions (zero padding for j - t < 0).
    let feat_len = checked_dim(s_kv, hidden, "conv feature length")?;
    let mut feat = vec![0.0f32; feat_len];
    for j in 0..s_kv {
        for kw in kernels {
            for (t, &wt) in kw.iter().enumerate() {
                if t > j {
                    break; // causal: k[j - t] does not exist
                }
                let src = (j - t) * hidden;
                let dst = j * hidden;
                for d in 0..hidden {
                    feat[dst + d] += wt * k[src + d];
                }
            }
        }
    }

    // Gated RMSNorm per key position.
    for j in 0..s_kv {
        let row = &mut feat[j * hidden..(j + 1) * hidden];
        let g = sigmoid(dot(row, norm_gain));
        let inv = 1.0 / (rms_mean_square(row) + NORM_EPS).sqrt();
        for v in row.iter_mut() {
            *v = (*v) * g * inv;
        }
    }

    let out_len = checked_dim(s_q, s_kv, "score matrix length")?;
    let mut out = vec![0.0f32; out_len];
    for i in 0..s_q {
        let qs = &q[i * hidden..(i + 1) * hidden];
        for j in 0..s_kv {
            out[i * s_kv + j] = dot(qs, &feat[j * hidden..(j + 1) * hidden]);
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

#[cfg(test)]
mod tests {
    use super::*;
    use deepinfer_core::prop::{prop, random_vec};
    use deepinfer_core::rng::Philox;

    /// Naive transcription of the conv branch (independent ordering of the
    /// same math; used for exact comparison).
    fn naive_conv(
        q: &[f32],
        k: &[f32],
        s_q: usize,
        s_kv: usize,
        hidden: usize,
        kernels: &[Vec<f32>],
        gain: &[f32],
        causal: bool,
    ) -> Vec<f32> {
        let mut feat = vec![0.0f32; s_kv * hidden];
        for j in 0..s_kv {
            for kw in kernels {
                for t in 0..kw.len() {
                    if t <= j {
                        for d in 0..hidden {
                            feat[j * hidden + d] += kw[t] * k[(j - t) * hidden + d];
                        }
                    }
                }
            }
        }
        for j in 0..s_kv {
            let mut gv = 0.0f32;
            for d in 0..hidden {
                gv += feat[j * hidden + d] * gain[d];
            }
            let g = 1.0 / (1.0 + (-gv).exp());
            let mut ms = 0.0f32;
            for d in 0..hidden {
                ms += feat[j * hidden + d] * feat[j * hidden + d];
            }
            ms /= hidden as f32;
            let inv = 1.0 / (ms + 1e-6).sqrt();
            for d in 0..hidden {
                feat[j * hidden + d] = feat[j * hidden + d] * g * inv;
            }
        }
        let mut out = vec![0.0f32; s_q * s_kv];
        for i in 0..s_q {
            for j in 0..s_kv {
                let mut acc = 0.0f32;
                for d in 0..hidden {
                    acc += q[i * hidden + d] * feat[j * hidden + d];
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
    fn hand_computed_case() {
        // One cascade, kernel [1, 2]; hidden = 2; gain = [1, 0]; no quant.
        // feat_0 = [1,0]; feat_1 = [2,1]; feat_2 = [2,1]; feat_3 = [5,-1].
        // Gated RMSNorm (sigma(1), sigma(2), sigma(2), sigma(5) gates, RMS
        // denominators) then dot with q_0 = [1,0], q_1 = [0,1].
        let q = [1.0f32, 0.0, 0.0, 1.0];
        let k = [1.0f32, 0.0, 0.0, 1.0, 2.0, -1.0, 1.0, 1.0];
        let kernels = vec![vec![1.0f32, 2.0]];
        let gain = [1.0f32, 0.0];
        let out = conv_scores(&q, &k, 2, 4, 2, &kernels, &gain, false).unwrap();
        // Hand-computed: g0 = sigma(1) = .7310586, inv0 = 1/sqrt(.5+1e-6)
        // g1 = sigma(2) = .8807971, inv1 = 1/sqrt(2.5+1e-6)
        // g3 = sigma(5) = .9933072, inv3 = 1/sqrt(13+1e-6)
        let s00 = 1.0 * 0.731_058_6 * (1.0f32 / (0.5f32 + 1e-6).sqrt());
        let s01 = 2.0 * 0.880_797_1 * (1.0f32 / (2.5f32 + 1e-6).sqrt());
        let s03 = 5.0 * 0.993_307_2 * (1.0f32 / (13.0f32 + 1e-6).sqrt());
        assert!((out[0] - s00).abs() < 1e-4, "s(0,0) {}", out[0]);
        assert!((out[1] - s01).abs() < 1e-4, "s(0,1) {}", out[1]);
        assert!((out[2] - s01).abs() < 1e-4, "s(0,2) {}", out[2]);
        assert!((out[3] - s03).abs() < 1e-4, "s(0,3) {}", out[3]);
        assert!(out[4].abs() < 1e-6, "s(1,0) {}", out[4]);
        let s12 = 1.0 * 0.880_797_1 * (1.0f32 / (2.5f32 + 1e-6).sqrt());
        assert!((out[5] - s12).abs() < 1e-4, "s(1,1) {}", out[5]);
        assert!((out[6] - s12).abs() < 1e-4, "s(1,2) {}", out[6]);
        // s(1,3) = dot([0,1], out_3) = -(g3 * inv3) with g3 = sigma(5).
        assert!(
            (out[7] + 0.993_307_2 * (1.0f32 / (13.0f32 + 1e-6).sqrt())).abs() < 1e-4,
            "s(1,3) {}",
            out[7]
        );

        // Naive comparison must be bit-identical.
        let want = naive_conv(&q, &k, 2, 4, 2, &kernels, &gain, false);
        assert_eq!(out, want);
    }

    #[test]
    fn zero_padding_at_sequence_start() {
        // With kernel [1, 2, 3], position 0 sees only tap 0: k_{-1} and
        // k_{-2} are zero-padded, so feat_0 (and thus score(*, 0)) depends
        // ONLY on k[0].
        let kernels = vec![vec![1.0f32, 2.0, 3.0]];
        let gain = [0.0f32, 0.0]; // gate = sigma(0) = 0.5
        let q = [1.0f32, 0.0];
        let k = [2.0f32, 5.0, 7.0, 11.0];
        let out = conv_scores(&q, &k, 1, 2, 2, &kernels, &gain, false).unwrap();
        // feat_0 = [2, 5] -> score(0,0) = 2 * 0.5 / sqrt(14.5 + eps)
        assert!(
            (out[0] - 1.0f32 / (14.5f32 + 1e-6).sqrt()).abs() < 1e-4,
            "s(0,0) {}",
            out[0]
        );
        // feat_1 = [7 + 2*2, 11 + 2*5] = [11, 21] -> score(0,1) = 11 * 0.5 / sqrt(281 + eps)
        assert!(
            (out[1] - 5.5f32 / (281.0f32 + 1e-6).sqrt()).abs() < 1e-4,
            "s(0,1) {}",
            out[1]
        );
        // Zero-padding causality: perturbing every token after the first
        // must not change score(*, 0) at all.
        let mut k2 = k;
        k2[2] = -42.0;
        k2[3] = 0.25;
        let out2 = conv_scores(&q, &k2, 1, 2, 2, &kernels, &gain, false).unwrap();
        assert_eq!(out2[0], out[0], "score(0,0) must depend only on k[0]");
        assert_ne!(out2[1], out[1]);
        // Naive comparison must be bit-identical.
        let want = naive_conv(&q, &k, 1, 2, 2, &kernels, &gain, false);
        assert_eq!(out, want);
    }

    #[test]
    fn causal_clip_pattern() {
        let mut rng = Philox::new(0xC0, 1);
        let (s_q, s_kv, hidden) = (3, 5, 4);
        let q = random_vec(&mut rng, s_q * hidden, 2.0);
        let k = random_vec(&mut rng, s_kv * hidden, 2.0);
        let kernels = vec![vec![0.5f32, -0.25], vec![1.0f32, 0.5, 0.25]];
        let gain = random_vec(&mut rng, hidden, 1.0);
        let causal = conv_scores(&q, &k, s_q, s_kv, hidden, &kernels, &gain, true).unwrap();
        let full = conv_scores(&q, &k, s_q, s_kv, hidden, &kernels, &gain, false).unwrap();
        for i in 0..s_q {
            for j in 0..s_kv {
                if j > i {
                    assert_eq!(causal[i * s_kv + j], f32::NEG_INFINITY);
                } else {
                    assert_eq!(causal[i * s_kv + j], full[i * s_kv + j]);
                }
            }
        }
    }

    #[test]
    fn validates_inputs() {
        let q = [1.0f32, 1.0];
        let k = [1.0f32, 1.0];
        let kernels = vec![vec![1.0f32]];
        let gain = [1.0f32, 1.0];
        assert!(conv_scores(&q, &k, 1, 1, 2, &kernels, &[1.0], false).is_err()); // gain len
        assert!(conv_scores(&q, &k, 1, 1, 2, &[], &gain, false).is_ok()); // empty cascade allowed
        assert!(conv_scores(&q, &k, 1, 1, 2, &[vec![]], &gain, false).is_err()); // empty kernel
        assert!(conv_scores(&q, &k, 2, 1, 2, &kernels, &gain, false).is_err()); // q len
        assert!(conv_scores(&q, &k, 1, 2, 2, &kernels, &gain, false).is_err()); // k len
        assert!(conv_scores(&[f32::NAN, 1.0], &k, 1, 1, 2, &kernels, &gain, false).is_err());
    }

    #[test]
    fn property_matches_naive() {
        prop("dsa_conv_matches_naive", 0xC02, 120, |_i, rng| {
            let s_q = rng.uniform_usize(1, 4);
            let s_kv = rng.uniform_usize(1, 7);
            let hidden = rng.uniform_usize(1, 5);
            let n_cascade = rng.uniform_usize(0, 3);
            let mut kernels = Vec::new();
            for _ in 0..n_cascade {
                let ks = rng.uniform_usize(1, 5);
                kernels.push(random_vec(rng, ks, 1.5));
            }
            let gain = random_vec(rng, hidden, 1.0);
            let q = random_vec(rng, s_q * hidden, 2.0);
            let k = random_vec(rng, s_kv * hidden, 2.0);
            let causal = rng.next_f32() < 0.5;
            let got = conv_scores(&q, &k, s_q, s_kv, hidden, &kernels, &gain, causal)?;
            let want = naive_conv(&q, &k, s_q, s_kv, hidden, &kernels, &gain, causal);
            ensure!(got == want, Internal, "conv mismatch");
            Ok(())
        });
    }
}
