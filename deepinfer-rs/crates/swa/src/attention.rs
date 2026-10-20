//! Sliding-window causal attention over YOCO decoder KV rows.
//!
//! In YOCO ("You Only Cache Once", Sun et al., arXiv:2405.05254) the
//! self-attention half of the decoder consumes keys/values only inside a
//! sliding window of the most recent positions; DeepInfer ships this as
//! the SWA decoder layers fed by the bounded KV replay (upstream README:
//! "YOCO SWA bounded replay"). This module is the CPU-reference kernel
//! [`swa_attention`]: for every query it selects the causal,
//! within-window keys and applies a numerically stable (max-subtracted)
//! softmax, accumulating in ascending key order so that the same inputs
//! always produce bit-identical outputs.

use deepinfer_core::math::dot;
use deepinfer_core::{ensure, Error, Result};

/// Computes sliding-window causal attention of a query block against
/// windowed KV rows (YOCO decoder self-attention; Sun et al.,
/// arXiv:2405.05254).
///
/// * `q` — `[s_q, h, d]` row-major query block; query `i` sits at the
///   absolute position `q_offset + i`.
/// * `kv` — `[n_kv, d]` row-major key/value rows (single-head latent-style
///   rows shared by all heads); row `j` sits at the absolute position
///   `kv_positions[j]`.
/// * `kv_positions` — absolute, non-decreasing positions of the `kv` rows.
/// * `window` — SWA window: query `(i, head)` attends key `j` iff
///   `kv_positions[j] <= q_offset + i` (causal) **and**
///   `q_offset + i - kv_positions[j] < window` (window), i.e. the maximum
///   attendable key distance is `window - 1` and `window == 0` masks
///   everything.
/// * `sm_scale` — softmax scale (typically `1/sqrt(d)`).
///
/// Returns the `[s_q, h, d]` output: for each query/head the softmax over
/// that query's valid key set applied to the `kv` rows, with the
/// weighted sum accumulated in ascending `j` order and rescaled once at
/// the end (flash-attention style). Queries with no valid keys produce
/// zero rows. Non-finite `q` / `kv` / `sm_scale` values propagate
/// deterministically (garbage in, garbage out), matching upstream kernels.
///
/// Determinism: sequential `j` and `k` accumulation, `f32` throughout —
/// identical inputs give bit-identical outputs.
///
/// # Errors
/// * [`Error::DimensionMismatch`] when `q.len() != s_q * h * d` or
///   `kv.len() != kv_positions.len() * d` (or either product overflows
///   `usize`).
/// * [`Error::InvalidArgument`] when `kv_positions` decreases, or when
///   `q_offset + s_q - 1` overflows `usize`.
pub fn swa_attention(
    q: &[f32],
    kv: &[f32],
    kv_positions: &[usize],
    q_offset: usize,
    s_q: usize,
    h: usize,
    d: usize,
    window: usize,
    sm_scale: f32,
) -> Result<Vec<f32>> {
    // ---- shape validation (checked arithmetic; never panics) ----
    let q_len = s_q
        .checked_mul(h)
        .and_then(|x| x.checked_mul(d))
        .ok_or_else(|| {
            Error::DimensionMismatch(format!(
                "s_q ({}) * h ({}) * d ({}) overflows usize",
                s_q, h, d
            ))
        })?;
    ensure!(
        q.len() == q_len,
        DimensionMismatch,
        "q has {} elements, expected s_q ({}) * h ({}) * d ({}) = {}",
        q.len(),
        s_q,
        h,
        d,
        q_len
    );
    let n_kv = kv_positions.len();
    let kv_len = n_kv.checked_mul(d).ok_or_else(|| {
        Error::DimensionMismatch(format!("n_kv ({}) * d ({}) overflows usize", n_kv, d))
    })?;
    ensure!(
        kv.len() == kv_len,
        DimensionMismatch,
        "kv has {} elements, expected n_kv ({}) * d ({}) = {}",
        kv.len(),
        n_kv,
        d,
        kv_len
    );
    for j in 1..n_kv {
        ensure!(
            kv_positions[j] >= kv_positions[j - 1],
            InvalidArgument,
            "kv_positions must be non-decreasing: positions[{}] = {} < positions[{}] = {}",
            j,
            kv_positions[j],
            j - 1,
            kv_positions[j - 1]
        );
    }
    ensure!(
        s_q == 0 || q_offset.checked_add(s_q - 1).is_some(),
        InvalidArgument,
        "q_offset ({}) + s_q ({}) - 1 overflows usize",
        q_offset,
        s_q
    );

    let mut out = vec![0.0f32; q_len];
    // Score scratch indexed by key j — sized by the real n_kv, so the
    // allocation is bounded by the input.
    let mut scores = vec![0.0f32; n_kv];

    for i in 0..s_q {
        let q_pos = q_offset + i;
        // Causality: with non-decreasing positions the keys at position
        // <= q_pos are exactly the prefix [0, hi).
        let mut hi = 0usize;
        while hi < n_kv && kv_positions[hi] <= q_pos {
            hi += 1;
        }
        // Window: the leading keys at distance >= window drop out; the
        // valid set is the contiguous range [lo, hi).
        let mut lo = 0usize;
        while lo < hi && q_pos - kv_positions[lo] >= window {
            lo += 1;
        }
        if lo == hi {
            continue; // no valid keys: the output row stays zero
        }
        for head in 0..h {
            let base = (i * h + head) * d;
            let qh = &q[base..base + d];
            // Pass 1: scores and their max (numerically stable softmax).
            let mut m = f32::NEG_INFINITY;
            for j in lo..hi {
                let s = dot(qh, &kv[j * d..j * d + d]) * sm_scale;
                scores[j] = s;
                if s > m {
                    m = s;
                }
            }
            // Pass 2: softmax weights times kv rows, accumulated in
            // ascending j order and rescaled once at the end.
            let mut denom = 0.0f32;
            for j in lo..hi {
                let p = (scores[j] - m).exp();
                denom += p;
                let kj = &kv[j * d..j * d + d];
                for k in 0..d {
                    out[base + k] += p * kj[k];
                }
            }
            for k in 0..d {
                out[base + k] /= denom;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepinfer_core::prop::{assert_close_slice, prop, random_vec};
    use deepinfer_core::rng::Philox;
    use deepinfer_core::Error;

    /// Naive dense reference: direct per-key causal+window test, `f64`
    /// arithmetic, max-subtracted softmax with per-key normalization.
    /// Deliberately structured differently from the implementation.
    #[allow(clippy::too_many_arguments)]
    fn dense_reference(
        q: &[f32],
        kv: &[f32],
        kv_positions: &[usize],
        q_offset: usize,
        s_q: usize,
        h: usize,
        d: usize,
        window: usize,
        sm_scale: f32,
    ) -> Vec<f32> {
        let n_kv = kv_positions.len();
        let mut out = vec![0.0f64; s_q * h * d];
        for i in 0..s_q {
            let q_pos = q_offset + i;
            for head in 0..h {
                let base = (i * h + head) * d;
                let mut js: Vec<usize> = Vec::new();
                let mut scores: Vec<f64> = Vec::new();
                for j in 0..n_kv {
                    let pos = kv_positions[j];
                    if pos <= q_pos && q_pos - pos < window {
                        let mut s = 0.0f64;
                        for k in 0..d {
                            s += q[base + k] as f64 * kv[j * d + k] as f64;
                        }
                        js.push(j);
                        scores.push(s * sm_scale as f64);
                    }
                }
                if js.is_empty() {
                    continue;
                }
                let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let denom: f64 = scores.iter().map(|s| (s - m).exp()).sum();
                for (idx, &j) in js.iter().enumerate() {
                    let p = (scores[idx] - m).exp() / denom;
                    for k in 0..d {
                        out[base + k] += p * kv[j * d + k] as f64;
                    }
                }
            }
        }
        out.into_iter().map(|x| x as f32).collect()
    }

    #[test]
    fn uniform_sliding_selection_is_exact() {
        // q = 0 makes every score 0, so each query's output is the exact
        // mean of its valid kv rows — a crisp check of which keys were
        // selected. Queries sit at positions 1 and 2 with window 2:
        // query 0 -> keys {0, 1}, query 1 -> keys {1, 2}.
        let q = [0.0f32, 0.0f32];
        let kv = [1.0f32, 2.0f32, 3.0f32];
        let positions = [0usize, 1, 2];
        let out = swa_attention(&q, &kv, &positions, 1, 2, 1, 1, 2, 1.0).unwrap();
        // (1 + 2) / 2 and (2 + 3) / 2 — exact in f32.
        assert_eq!(out, vec![1.5, 2.5]);
    }

    #[test]
    fn window_boundary_included_at_minus_one_excluded_at_window() {
        // Query at position 100, window 4: distance 3 (= window - 1) is
        // attendable, distance 4 (= window) is not.
        let q = vec![1.0f32, 1.0f32];
        // Row at position 96 (distance 4, excluded), row at 97 (distance
        // 3, included). Positions must be non-decreasing.
        let kv = vec![9.0f32, 9.0f32, 3.0f32, 7.0f32];
        let positions = [96usize, 97];
        let out = swa_attention(&q, &kv, &positions, 100, 1, 1, 2, 4, 1.0).unwrap();
        // Single valid key -> probability exactly 1 -> output == that row.
        assert_eq!(out, vec![3.0, 7.0]);

        // window == 1 keeps only the key at the query position itself.
        let positions = [99usize, 100, 101];
        let kv = vec![1.0f32, 2.0f32, 4.0f32, 8.0f32, 16.0f32, 32.0f32];
        let out = swa_attention(&q, &kv, &positions, 100, 1, 1, 2, 1, 1.0).unwrap();
        assert_eq!(out, vec![4.0, 8.0]);
    }

    #[test]
    fn no_valid_keys_yields_zero_rows() {
        // (a) empty kv cache.
        let out = swa_attention(&[1.0f32; 6], &[], &[], 10, 2, 3, 1, 8, 1.0).unwrap();
        assert_eq!(out, vec![0.0f32; 6]);
        // (b) keys only in the future of every query.
        let out = swa_attention(&[1.0f32; 2], &[1.0f32, 2.0f32], &[50], 10, 1, 1, 2, 8, 1.0)
            .unwrap();
        assert_eq!(out, vec![0.0f32, 0.0f32]);
        // (c) keys too far in the past.
        let out = swa_attention(&[1.0f32; 2], &[1.0f32, 2.0f32], &[0], 100, 1, 1, 2, 8, 1.0)
            .unwrap();
        assert_eq!(out, vec![0.0f32, 0.0f32]);
        // (d) window == 0 masks everything, even the query's own position.
        let out = swa_attention(&[1.0f32; 2], &[1.0f32, 2.0f32], &[5], 5, 1, 1, 2, 0, 1.0)
            .unwrap();
        assert_eq!(out, vec![0.0f32, 0.0f32]);
    }

    #[test]
    fn future_keys_never_leak() {
        // Query at position 2; the key at position 3 is in the future.
        let q = [2.0f32];
        let kv = [1.0f32, 5.0f32];
        let positions = [2usize, 3];
        let out = swa_attention(&q, &kv, &positions, 2, 1, 1, 1, 64, 1.0).unwrap();
        // Single valid key -> output == that row exactly.
        assert_eq!(out, vec![1.0f32]);
        // Perturbing the future key must not change the output.
        let kv2 = [1.0f32, 999.0f32];
        let out2 = swa_attention(&q, &kv2, &positions, 2, 1, 1, 1, 64, 1.0).unwrap();
        assert_eq!(out, out2);
    }

    #[test]
    fn two_key_softmax_hand_computed() {
        // d = 1, one query at position 5, keys at positions 4 and 5.
        // Scores: 1*2 = 2 and 1*0 = 0.
        let q = [1.0f32];
        let kv = [2.0f32, 0.0f32];
        let positions = [4usize, 5];
        let out = swa_attention(&q, &kv, &positions, 5, 1, 1, 1, 128, 1.0).unwrap();
        let e2 = 2.0f32.exp();
        let p0 = e2 / (e2 + 1.0);
        assert_close_slice(&out, &[p0 * 2.0], 1e-6, "hand-computed softmax");
    }

    #[test]
    fn duplicate_positions_are_both_attended() {
        // Two keys share position 4 (non-decreasing allows duplicates);
        // both must contribute (uniform because q is orthogonal to row 0).
        let q = [0.0f32, 1.0f32];
        let kv = [1.0f32, 1.0f32, 3.0f32, 5.0f32];
        let positions = [4usize, 4];
        let out = swa_attention(&q, &kv, &positions, 4, 1, 1, 2, 8, 1.0).unwrap();
        // Scores: 1 and 5 -> softmax(1, 5).
        let e1 = 1.0f32.exp();
        let e5 = 5.0f32.exp();
        let p0 = e1 / (e1 + e5);
        let p1 = e5 / (e1 + e5);
        let want = [p0 * 1.0 + p1 * 3.0, p0 * 1.0 + p1 * 5.0];
        assert_close_slice(&out, &want, 1e-5, "duplicate positions");
    }

    #[test]
    fn heads_are_independent_slices() {
        // A [s_q, 2, d] call must equal two [s_q, 1, d] calls on the
        // per-head query slices.
        let mut rng = Philox::new(0x5A11, 3);
        let (s_q, h, d) = (3usize, 2usize, 3usize);
        let q_offset = 4usize;
        let window = 6usize;
        let positions = [1usize, 3, 4, 5, 6, 9];
        let n_kv = positions.len();
        let q = random_vec(&mut rng, s_q * h * d, 1.5);
        let kv = random_vec(&mut rng, n_kv * d, 1.5);
        let out = swa_attention(&q, &kv, &positions, q_offset, s_q, h, d, window, 0.5).unwrap();
        assert_eq!(out.len(), s_q * h * d);
        for head in 0..h {
            let mut qh = Vec::with_capacity(s_q * d);
            for i in 0..s_q {
                let base = (i * h + head) * d;
                qh.extend_from_slice(&q[base..base + d]);
            }
            let want =
                swa_attention(&qh, &kv, &positions, q_offset, s_q, 1, d, window, 0.5).unwrap();
            for i in 0..s_q {
                let base = (i * h + head) * d;
                assert_eq!(&out[base..base + d], &want[i * d..i * d + d], "head {head}");
            }
        }
    }

    #[test]
    fn validates_shapes_positions_and_overflow() {
        // q length mismatch.
        assert!(matches!(
            swa_attention(&[1.0f32; 5], &[1.0f32; 4], &[0, 1], 0, 2, 2, 1, 8, 1.0),
            Err(Error::DimensionMismatch(_))
        ));
        // kv length vs positions count.
        assert!(matches!(
            swa_attention(&[1.0f32; 2], &[1.0f32; 3], &[0, 1], 0, 1, 1, 2, 8, 1.0),
            Err(Error::DimensionMismatch(_))
        ));
        // Decreasing positions are rejected; equal positions are fine.
        assert!(matches!(
            swa_attention(&[1.0f32; 2], &[1.0f32; 4], &[5, 4], 0, 1, 1, 2, 8, 1.0),
            Err(Error::InvalidArgument(_))
        ));
        assert!(swa_attention(&[1.0f32; 2], &[1.0f32; 4], &[5, 5], 5, 1, 1, 2, 8, 1.0).is_ok());
        // q_offset + s_q - 1 must not overflow usize.
        assert!(matches!(
            swa_attention(&[1.0f32; 2], &[1.0f32], &[0], usize::MAX, 2, 1, 1, 8, 1.0),
            Err(Error::InvalidArgument(_))
        ));
        // Degenerate but consistent shapes are accepted.
        let out = swa_attention(&[], &[1.0f32], &[0], 3, 1, 0, 1, 8, 1.0).unwrap();
        assert!(out.is_empty());
        let out = swa_attention(&[], &[1.0f32], &[0], 3, 0, 4, 1, 8, 1.0).unwrap();
        assert!(out.is_empty());
        // A product that overflows usize cannot match any real slice.
        assert!(matches!(
            swa_attention(&[], &[], &[], 0, usize::MAX, usize::MAX, usize::MAX, 8, 1.0),
            Err(Error::DimensionMismatch(_))
        ));
    }

    #[test]
    fn deterministic_across_runs() {
        let mut rng = Philox::new(0x5A77, 11);
        let (s_q, h, d) = (4usize, 3usize, 5usize);
        let q_offset = 7usize;
        let window = 5usize;
        let mut positions = Vec::new();
        let mut p = rng.uniform_usize(0, 12);
        for _ in 0..14 {
            positions.push(p);
            p += rng.uniform_usize(0, 3);
        }
        let q = random_vec(&mut rng, s_q * h * d, 1.5);
        let kv = random_vec(&mut rng, positions.len() * d, 1.5);
        let a = swa_attention(&q, &kv, &positions, q_offset, s_q, h, d, window, 0.7).unwrap();
        let b = swa_attention(&q, &kv, &positions, q_offset, s_q, h, d, window, 0.7).unwrap();
        assert_eq!(a, b, "bit-identical across runs");
        assert!(a.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn property_matches_dense_reference() {
        prop("swa_attention_dense_equiv", 0x5A77, 150, |_i, rng| {
            let s_q = rng.uniform_usize(1, 5);
            let h = rng.uniform_usize(1, 3);
            let d = rng.uniform_usize(1, 5);
            let window = rng.uniform_usize(0, 14);
            let q_offset = rng.uniform_usize(0, 20);
            let n_kv = rng.uniform_usize(0, 18);
            // Non-decreasing positions (steps may repeat -> duplicates),
            // mixing past, present and future keys.
            let mut kv_positions = Vec::with_capacity(n_kv);
            let mut p = rng.uniform_usize(0, q_offset + s_q + 8);
            for _ in 0..n_kv {
                kv_positions.push(p);
                p += rng.uniform_usize(0, 3);
            }
            let q = random_vec(rng, s_q * h * d, 1.5);
            let kv = random_vec(rng, n_kv * d, 1.5);
            let sm_scale = if _i % 2 == 0 {
                1.0
            } else {
                rng.uniform_f32(0.1, 1.0)
            };
            let got = swa_attention(
                &q, &kv, &kv_positions, q_offset, s_q, h, d, window, sm_scale,
            )?;
            let want = dense_reference(
                &q, &kv, &kv_positions, q_offset, s_q, h, d, window, sm_scale,
            );
            assert_close_slice(&got, &want, 1e-4, "swa_attention vs dense window-mask reference");
            Ok(())
        });
    }
}
