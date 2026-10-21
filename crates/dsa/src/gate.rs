//! Coefficient gating for the DSA lightning indexer.
//!
//! DSA (arXiv:2509.07152, "Lightning Indexer") blends the semantic
//! (thunder) and spatial (cascade-conv) score branches with per-query
//! sigmoid gates computed from a small projection of the query and the
//! query-sequence mean:
//!
//! ```text
//! raw      = gate_proj (2 x 2*hidden) @ [q_i ; mean_j(q_j)]
//! g_thunder = sigmoid(raw[0]),  g_conv = sigmoid(raw[1])
//! combined  = (g_thunder * thunder + g_conv * conv) / (g_thunder + g_conv)
//! ```
//!
//! with a fallback to the raw thunder score when both gates underflow to
//! zero. Division by the gate sum (documented deviation from the SPEC's
//! plain `g_t * thunder + g_c * conv` form) keeps the blended score on the
//! same scale as the two branches — the pair of sigmoids does not sum to
//! one by construction.
//!
//! `q` is the RAW hidden-state query block `[s_q, hidden]` (not the
//! projected indexer vectors), and `mean_j(q_j)` is the mean over the `s_q`
//! query rows. Entries carrying the causal-mask sentinel `-inf` (both
//! branches share the same clip) stay `-inf` after the blend.

use deepinfer_core::{ensure, Result};

use crate::{checked_dim, ensure_finite, ensure_scores_valid, sigmoid};

/// Blends `thunder` and `conv` score matrices in place with per-query
/// coefficient gating.
///
/// * `q`: `[s_q, hidden]` raw hidden-state queries.
/// * `gate_proj`: `[2, 2*hidden]` gating projection.
/// * `thunder`: `[s_q, cols]` thunder scores (mutated; becomes the combined
///   score). `cols` is inferred as `thunder.len() / s_q`.
/// * `conv`: `[s_q, cols]` conv scores.
///
/// Score entries must be finite or exactly `-inf`; `NaN` / `+inf` are
/// rejected as numerical overflow.
pub fn gate_and_combine(
    q: &[f32],
    s_q: usize,
    hidden: usize,
    gate_proj: &[f32],
    thunder: &mut [f32],
    conv: &[f32],
) -> Result<()> {
    ensure!(
        hidden >= 1,
        InvalidArgument,
        "hidden must be >= 1 (got {hidden})"
    );
    let q_len = checked_dim(s_q, hidden, "q length")?;
    ensure!(
        q.len() == q_len,
        DimensionMismatch,
        "q has {} elements, expected s_q*hidden = {q_len}",
        q.len()
    );
    let gate_len = checked_dim(hidden, 4, "gate_proj length")?; // [2, 2*hidden]
    ensure!(
        gate_proj.len() == gate_len,
        DimensionMismatch,
        "gate_proj has {} elements, expected 2*2*hidden = {gate_len}",
        gate_proj.len()
    );
    ensure!(
        thunder.len() == conv.len(),
        DimensionMismatch,
        "thunder has {} entries but conv has {}",
        thunder.len(),
        conv.len()
    );
    ensure_finite(q, "q")?;
    ensure_finite(gate_proj, "gate_proj")?;
    ensure_scores_valid(thunder, "thunder")?;
    ensure_scores_valid(conv, "conv")?;
    if s_q == 0 {
        ensure!(
            thunder.is_empty(),
            DimensionMismatch,
            "s_q == 0 but thunder has {} entries",
            thunder.len()
        );
        return Ok(());
    }
    let cols = thunder.len() / s_q;
    ensure!(
        cols * s_q == thunder.len(),
        DimensionMismatch,
        "thunder length {} is not divisible by s_q = {s_q}",
        thunder.len()
    );

    // mean over the query rows (fixed i-ascending accumulation per dim).
    let mut mean_q = vec![0.0f32; hidden];
    for i in 0..s_q {
        for d in 0..hidden {
            mean_q[d] += q[i * hidden + d];
        }
    }
    for m in mean_q.iter_mut() {
        *m /= s_q as f32;
    }

    let row2 = 2 * hidden;
    for i in 0..s_q {
        // raw_t / raw_c = gate_proj rows dotted with [q_i ; mean_q].
        let mut raw_t = 0.0f32;
        let mut raw_c = 0.0f32;
        for u in 0..row2 {
            let x = if u < hidden {
                q[i * hidden + u]
            } else {
                mean_q[u - hidden]
            };
            raw_t += gate_proj[u] * x;
            raw_c += gate_proj[row2 + u] * x;
        }
        ensure!(
            !raw_t.is_nan() && !raw_c.is_nan(),
            Numerical,
            "gate logits overflowed to NaN at query {i}"
        );
        let g_t = sigmoid(raw_t);
        let g_c = sigmoid(raw_c);
        let denom = g_t + g_c;
        let base = i * cols;
        for j in 0..cols {
            let th = thunder[base + j];
            let cv = conv[base + j];
            // Both branches share the causal clip: -inf must survive the
            // blend (and 0 * -inf would be NaN, so handle masked entries
            // explicitly).
            if th == f32::NEG_INFINITY || cv == f32::NEG_INFINITY {
                thunder[base + j] = f32::NEG_INFINITY;
            } else if denom > 0.0 {
                thunder[base + j] = (g_t * th + g_c * cv) / denom;
            } else {
                // Both gates underflowed to zero: keep the thunder score.
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepinfer_core::prop::{prop, random_vec};

    #[test]
    fn gates_are_in_unit_interval_and_blend_hand_case() {
        // hidden = 2, s_q = 2; gate_proj rows isolate q_i[0] and mean_q[1].
        let q = [1.0f32, 0.0, 0.0, 1.0]; // mean_q = [0.5, 0.5]
        let gate_proj = [
            1.0f32, 0.0, 0.0, 0.0, // row 0: raw_t = q_i[0]
            0.0, 0.0, 0.0, 2.0, // row 1: raw_c = 2 * mean_q[1] = 1
        ];
        let mut thunder = [10.0f32, 20.0, 30.0, 40.0];
        let conv = [0.0f32; 4];
        gate_and_combine(&q, 2, 2, &gate_proj, &mut thunder, &conv).unwrap();
        // i = 0: g_t = sigma(1), g_c = sigma(1) -> pure average of (10, 0), (20, 0).
        assert!((thunder[0] - 5.0).abs() < 1e-6, "{}", thunder[0]);
        assert!((thunder[1] - 10.0).abs() < 1e-6, "{}", thunder[1]);
        // i = 1: g_t = sigma(0) = 0.5, g_c = sigma(1) -> weighted average.
        let denom = 0.5f32 + 0.731_058_6;
        assert!((thunder[2] - 30.0 * 0.5 / denom).abs() < 1e-4, "{}", thunder[2]);
        assert!((thunder[3] - 40.0 * 0.5 / denom).abs() < 1e-4, "{}", thunder[3]);
    }

    #[test]
    fn masked_entries_stay_masked() {
        let q = [1.0f32, 0.0];
        let gate_proj = [0.0f32; 8]; // gates sigma(0) = 0.5 each
        let mut thunder = [f32::NEG_INFINITY, 4.0];
        let conv = [f32::NEG_INFINITY, 2.0];
        gate_and_combine(&q, 1, 2, &gate_proj, &mut thunder, &conv).unwrap();
        assert_eq!(thunder[0], f32::NEG_INFINITY);
        assert!((thunder[1] - 3.0).abs() < 1e-6, "{}", thunder[1]); // equal gates -> average
        assert!(!thunder.iter().any(|v| v.is_nan()));
    }

    #[test]
    fn zero_denominator_falls_back_to_thunder() {
        let q = [1.0f32, 1.0];
        let gate_proj = [-1000.0f32; 8]; // sigma(-2000) underflows to 0
        let mut thunder = [7.0f32, f32::NEG_INFINITY];
        let conv = [3.0f32, 9.0];
        gate_and_combine(&q, 1, 2, &gate_proj, &mut thunder, &conv).unwrap();
        assert_eq!(thunder[0], 7.0);
        assert_eq!(thunder[1], f32::NEG_INFINITY);
    }

    #[test]
    fn validates_inputs() {
        let mut th = [0.0f32; 2];
        let cv = [0.0f32; 2];
        assert!(gate_and_combine(&[1.0], 1, 2, &[0.0; 8], &mut th, &cv).is_err()); // q len
        assert!(gate_and_combine(&[1.0, 1.0], 1, 2, &[0.0; 7], &mut th, &cv).is_err()); // gp len
        assert!(gate_and_combine(&[1.0, 1.0], 1, 2, &[0.0; 8], &mut th, &[0.0; 3]).is_err()); // len mismatch
        let mut bad = [f32::NAN; 2];
        assert!(gate_and_combine(&[1.0, 1.0], 1, 2, &[0.0; 8], &mut bad, &cv).is_err()); // NaN score
        let mut pos_inf = [f32::INFINITY; 2];
        assert!(gate_and_combine(&[1.0, 1.0], 1, 2, &[0.0; 8], &mut pos_inf, &cv).is_err()); // +inf
        let mut odd = [0.0f32; 3];
        assert!(gate_and_combine(&[1.0, 1.0, 1.0, 1.0], 2, 2, &[0.0; 8], &mut odd, &[0.0; 3]).is_err()); // not divisible
    }

    #[test]
    fn property_combined_is_convex_mix() {
        prop("dsa_gate_convex_mix", 0x6A7, 120, |_i, rng| {
            let hidden = rng.uniform_usize(1, 5);
            let s_q = rng.uniform_usize(1, 4);
            let cols = rng.uniform_usize(1, 6);
            let q = random_vec(rng, s_q * hidden, 2.0);
            let gate_proj = random_vec(rng, 4 * hidden, 2.0);
            let mut th = random_vec(rng, s_q * cols, 5.0);
            let cv = random_vec(rng, s_q * cols, 5.0);
            let orig = th.clone();
            gate_and_combine(&q, s_q, hidden, &gate_proj, &mut th, &cv)?;
            for idx in 0..s_q * cols {
                let lo = orig[idx].min(cv[idx]) - 1e-5;
                let hi = orig[idx].max(cv[idx]) + 1e-5;
                ensure!(
                    th[idx] >= lo && th[idx] <= hi,
                    Internal,
                    "combined {} not within [{lo}, {hi}] at {idx}",
                    th[idx]
                );
                ensure!(th[idx].is_finite(), Internal, "non-finite at {idx}");
            }
            Ok(())
        });
    }
}
