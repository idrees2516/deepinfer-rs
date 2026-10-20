//! DSA top-k token selection and the V3.2 `mscale` constant.
//!
//! DeepSeek Sparse Attention (arXiv:2509.07152) keeps only the
//! `topk_tokens` highest-scoring keys per query. The efficient two-stage
//! selection of the V3.2 config (selection block 64, 8 pre-selected
//! survivors per block, global top-k 2048):
//!
//! 1. split the key positions `[0, s_kv)` into blocks of `block_size`;
//! 2. keep the top `preselect_per_block` entries per block (ties: higher
//!    score, then LOWER position);
//! 3. take the global top `K = min(topk_tokens, s_kv)` over the union of
//!    survivors (same tie rules);
//! 4. return positions sorted ascending, padded with `-1` to exactly `K`
//!    entries per query.
//!
//! Entries with the `-inf` causal-mask sentinel are never selected (they
//! surface as `-1` padding instead), and the context-extrapolation window
//! (`extrapolation_window = Some(w)`) restricts the candidates of query
//! `i` to the last `w` positions ending at `i` (inclusive) before
//! selection.
//!
//! ## Exactness of the two-stage selection
//!
//! The two-stage result equals the true global top-K iff no single block
//! contains more than `preselect_per_block` members of the true top-K. A
//! necessary condition is
//! `preselect_per_block * ceil(s_kv / block_size) >= K`; it becomes
//! (probabilistically) sufficient for uncorrelated scores when the
//! expected number of true-top-K members per block,
//! `K * block_size / s_kv`, sits well below `preselect_per_block`. For the
//! V3.2 defaults (block 64, preselect 8, topk 2048) at
//! `s_kv = 262144` this expectation is `0.5 << 8` and equality holds
//! (verified by a fixed-seed test below); near the boundary
//! (`s_kv = 16384`, expectation 8) the pre-selection is intentionally
//! lossy — the algorithm trades exact top-K for a spatially spread token
//! budget, which is the upstream behaviour.

use crate::config::LightningIndexerConfig;
use crate::{checked_dim, ensure_scores_valid};
use deepinfer_core::{ensure, Result};

/// DSA mscale for sparse attention: `0.1 * ln(topk_tokens) + 1`
/// (the V3.2 "msacle" constant used to rescale sparse-attention logits).
///
/// `topk_tokens == 0` yields `-inf` (`ln 0`); callers pass `>= 1`.
pub fn mscale(topk_tokens: usize) -> f32 {
    (0.1 * (topk_tokens as f64).ln() + 1.0) as f32
}

/// Selects the top-k key positions per query from a `[s_q, s_kv]` score
/// matrix, using the two-stage block pre-selection described in the module
/// docs.
///
/// Returns `[s_q, K]` with `K = min(topk_tokens, s_kv)`: per query, the
/// selected positions sorted ascending, padded with `-1` to exactly `K`
/// entries. Scores must be finite or exactly `-inf` (the causal-mask
/// sentinel; masked entries are never selected). When
/// `cfg.extrapolation_window == Some(w)`, the candidate keys of query `i`
/// are restricted to `[max(i+1, w) - w, min(i, s_kv - 1)]`.
///
/// Determinism: ties are broken by the total order
/// `(score desc, position asc)` and the output is position-sorted, so the
/// result is a pure function of the inputs.
pub fn select_top_k(
    scores: &[f32],
    s_q: usize,
    s_kv: usize,
    cfg: &LightningIndexerConfig,
) -> Result<Vec<i32>> {
    let total = checked_dim(s_q, s_kv, "scores length")?;
    ensure!(
        scores.len() == total,
        DimensionMismatch,
        "scores has {} elements, expected s_q*s_kv = {total}",
        scores.len()
    );
    ensure!(
        cfg.block_size >= 1,
        InvalidArgument,
        "block_size must be >= 1 (got {})",
        cfg.block_size
    );
    ensure!(
        cfg.preselect_per_block >= 1,
        InvalidArgument,
        "preselect_per_block must be >= 1 (got {})",
        cfg.preselect_per_block
    );
    ensure!(
        cfg.topk_tokens >= 1,
        InvalidArgument,
        "topk_tokens must be >= 1 (got {})",
        cfg.topk_tokens
    );
    if let Some(w) = cfg.extrapolation_window {
        ensure!(w >= 1, InvalidArgument, "extrapolation_window must be >= 1 (got {w})");
    }
    ensure!(
        s_kv <= i32::MAX as usize,
        CapacityExceeded,
        "s_kv ({s_kv}) exceeds i32::MAX; positions are not representable"
    );
    ensure_scores_valid(scores, "scores")?;

    let k_out = cfg.topk_tokens.min(s_kv);
    let mut out = vec![-1i32; checked_dim(s_q, k_out, "selection length")?];
    if k_out == 0 {
        return Ok(out);
    }
    for i in 0..s_q {
        let (lo, hi) = candidate_window(i, s_kv, cfg.extrapolation_window);
        let first_block = lo / cfg.block_size;
        let last_block = hi / cfg.block_size;
        let mut survivors: Vec<(f32, u32)> = Vec::new();
        for b in first_block..=last_block {
            let bs = b * cfg.block_size;
            let be = (b + 1).saturating_mul(cfg.block_size).min(s_kv);
            let s = bs.max(lo);
            let e = be.min(hi + 1);
            if s >= e {
                continue;
            }
            let mut cands: Vec<(f32, u32)> = (s..e)
                .filter(|&j| scores[i * s_kv + j].is_finite())
                .map(|j| (scores[i * s_kv + j], j as u32))
                .collect();
            cands.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            cands.truncate(cfg.preselect_per_block);
            survivors.extend(cands);
        }
        survivors.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        survivors.truncate(k_out);
        let mut pos: Vec<i32> = survivors.iter().map(|&(_, j)| j as i32).collect();
        pos.sort_unstable();
        out[i * k_out..i * k_out + pos.len()].copy_from_slice(&pos);
    }
    Ok(out)
}

/// Candidate key range `[lo, hi]` (inclusive) for query `i`: the whole key
/// axis by default, or the last `w` positions ending at `i` (clamped to the
/// sequence start) under context extrapolation.
fn candidate_window(i: usize, s_kv: usize, extrapolation: Option<usize>) -> (usize, usize) {
    let hi = if s_kv == 0 { 0 } else { s_kv - 1 };
    match extrapolation {
        Some(w) => ((i + 1).saturating_sub(w), i.min(hi)),
        None => (0, hi),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepinfer_core::prop::{prop, random_vec};
    use deepinfer_core::rng::Philox;

    /// Reference: full-sort top-K over the finite candidates of one row.
    fn full_sort_topk(scores: &[f32], s_kv: usize, i: usize, k: usize) -> Vec<i32> {
        let mut cands: Vec<(f32, u32)> = (0..s_kv)
            .filter(|&j| scores[i * s_kv + j].is_finite())
            .map(|j| (scores[i * s_kv + j], j as u32))
            .collect();
        cands.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        cands.truncate(k);
        let mut pos: Vec<i32> = cands.into_iter().map(|(_, j)| j as i32).collect();
        pos.sort_unstable();
        pos
    }

    fn small_cfg(
        block_size: usize,
        topk: usize,
        preselect: usize,
        extrapolation: Option<usize>,
    ) -> LightningIndexerConfig {
        LightningIndexerConfig {
            hidden_size: 8,
            num_heads: 2,
            head_dim: 4,
            block_size,
            topk_tokens: topk,
            preselect_per_block: preselect,
            quant_block: 1024,
            conv_kernels: vec![3],
            extrapolation_window: extrapolation,
        }
    }

    #[test]
    fn mscale_values() {
        // 0.1 * ln(2048) + 1 = 1.7624619...
        assert!((mscale(2048) - 1.762_461_9).abs() < 1e-6, "{}", mscale(2048));
        assert_eq!(mscale(1), 1.0); // ln(1) = 0
        assert!((mscale(2) - (1.0 + 0.1 * 2f64.ln() as f32)).abs() < 1e-7);
    }

    #[test]
    fn hand_case_two_stage_selection() {
        // blocks of 4, preselect 2, topk 3 over 10 keys.
        let cfg = small_cfg(4, 3, 2, None);
        let scores = [5.0f32, 1.0, 1.0, 1.0, 9.0, 8.0, 0.0, 0.0, 7.0, f32::NEG_INFINITY];
        let sel = select_top_k(&scores, 1, 10, &cfg).unwrap();
        // Block [0,4): 5@0, 1@1 (ties -> lower positions).
        // Block [4,8): 9@4, 8@5. Block [8,10): 7@8 (-inf never selected).
        // Global top-3: 9@4, 8@5, 7@8.
        assert_eq!(sel, vec![4, 5, 8]);
    }

    #[test]
    fn ties_break_to_lower_positions() {
        let cfg = small_cfg(64, 4, 2, None);
        let scores = [2.0f32; 5];
        let sel = select_top_k(&scores, 1, 5, &cfg).unwrap();
        assert_eq!(sel, vec![0, 1, -1, -1]);
    }

    #[test]
    fn masked_entries_are_padded_not_selected() {
        let cfg = small_cfg(8, 4, 8, None);
        let scores = [1.0f32, f32::NEG_INFINITY, f32::NEG_INFINITY, 3.0];
        let sel = select_top_k(&scores, 1, 4, &cfg).unwrap();
        assert_eq!(sel, vec![0, 3, -1, -1]);
    }

    #[test]
    fn true_topk_when_preselection_is_lossless() {
        // preselect >= block_size: per-block pre-selection keeps everything,
        // so the result is the true top-K even for adversarial placements
        // (all top-K members inside one block).
        let cfg = small_cfg(8, 5, 8, None);
        let mut scores = vec![0.5f32; 32];
        for j in 0..5 {
            scores[j] = 10.0 - j as f32; // top-5 all inside block 0
        }
        scores[20] = 5.9; // 6th-best globally: must NOT displace the top-5
        let sel = select_top_k(&scores, 1, 32, &cfg).unwrap();
        assert_eq!(sel, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn pads_when_survivors_below_k() {
        // V3.2 defaults, s_kv = 128: 2 blocks x 8 survivors = 16 < K = 128.
        let cfg = LightningIndexerConfig::v32_default(8);
        let mut rng = Philox::new(0x5E, 1);
        let scores = random_vec(&mut rng, 128, 10.0);
        let sel = select_top_k(&scores, 1, 128, &cfg).unwrap();
        assert_eq!(sel.len(), 128);
        let valid = sel.iter().filter(|&&v| v != -1).count();
        assert_eq!(valid, 16, "2 blocks x preselect 8");
        // Ascending positions, then -1 padding.
        let split = sel.iter().position(|&v| v == -1).unwrap();
        assert!(sel[..split].windows(2).all(|w| w[0] < w[1]));
        assert!(sel[split..].iter().all(|&v| v == -1));
        assert!(sel[..split].iter().all(|&v| (0..128).contains(&v)));
    }

    #[test]
    fn extrapolation_window_restricts_candidates() {
        let cfg = small_cfg(64, 3, 8, Some(2));
        let mut scores = vec![0.0f32; 36]; // [6, 6]
        // Row i: only positions within [i+1-2, i] are candidates.
        for (i, row) in scores.chunks_mut(6).enumerate() {
            row[0] = 100.0; // highest everywhere, must be excluded for windows not covering 0
            row[i] = 60.0;
            if i >= 1 {
                row[i - 1] = 50.0;
            }
        }
        let sel = select_top_k(&scores, 6, 6, &cfg).unwrap();
        assert_eq!(&sel[0..3], &[0, -1, -1]); // window {0}
        assert_eq!(&sel[3..6], &[0, 1, -1]); // window {0, 1}
        assert_eq!(&sel[6..9], &[1, 2, -1]); // window {0, 1, 2}
        assert_eq!(&sel[9..12], &[2, 3, -1]);
        assert_eq!(&sel[12..15], &[3, 4, -1]);
        // Query 5: window {4, 5}; position 0's 100 must not be selected.
        assert_eq!(&sel[15..18], &[4, 5, -1]);
    }

    #[test]
    fn matches_full_sort_on_v32_default_config() {
        // Fixed-seed empirical verification on the default V3.2 config:
        // s_kv = 262144 -> 4096 blocks x 8 = 32768 survivors >= K = 2048,
        // expected true-top-K members per block = 2048*64/262144 = 0.5 << 8,
        // so the two-stage selection equals the full sort.
        let cfg = LightningIndexerConfig::v32_default(8);
        let s_kv = 262_144usize;
        for seed in [0x5EED_0001u64, 0x5EED_0002] {
            let mut rng = Philox::new(seed, 7);
            let scores = random_vec(&mut rng, s_kv, 10.0);
            let sel = select_top_k(&scores, 1, s_kv, &cfg).unwrap();
            assert_eq!(sel.len(), 2048);
            assert!(sel.iter().all(|&v| v != -1));
            let want = full_sort_topk(&scores, s_kv, 0, 2048);
            assert_eq!(sel, want, "seed {seed:#x}");
        }
    }

    #[test]
    fn deterministic() {
        let cfg = small_cfg(4, 5, 2, None);
        let mut rng = Philox::new(0x5E, 2);
        let scores = random_vec(&mut rng, 3 * 16, 5.0);
        let a = select_top_k(&scores, 3, 16, &cfg).unwrap();
        let b = select_top_k(&scores, 3, 16, &cfg).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn validates_inputs() {
        let cfg = small_cfg(4, 4, 2, None);
        assert!(select_top_k(&[0.0; 3], 2, 2, &cfg).is_err()); // len mismatch
        assert!(select_top_k(&[f32::NAN; 4], 1, 4, &cfg).is_err()); // NaN
        let mut inf = vec![0.0f32; 4];
        inf[0] = f32::INFINITY;
        assert!(select_top_k(&inf, 1, 4, &cfg).is_err()); // +inf rejected
        assert!(select_top_k(&[0.0; 4], 1, 4, &small_cfg(0, 4, 2, None)).is_err());
        assert!(select_top_k(&[0.0; 4], 1, 4, &small_cfg(4, 0, 2, None)).is_err());
        assert!(select_top_k(&[0.0; 4], 1, 4, &small_cfg(4, 4, 0, None)).is_err());
        assert!(select_top_k(&[0.0; 4], 1, 4, &small_cfg(4, 4, 2, Some(0))).is_err());
    }

    #[test]
    fn property_matches_full_sort_with_margin() {
        // Scaled config with a comfortable margin: block 8, preselect 6,
        // topk 8, s_kv 512 -> 64 blocks x 6 = 384 survivors >= 8, expected
        // top-K members per block = 8*8/512 = 0.125 << 6.
        prop("dsa_select_matches_full_sort", 0x5E1, 120, |_i, rng| {
            let cfg = small_cfg(8, 8, 6, None);
            let s_kv = 512usize;
            let s_q = 3usize;
            let mut scores = random_vec(rng, s_q * s_kv, 10.0);
            // causal clip (j > i -> -inf), as produced by the score branches
            for i in 0..s_q {
                for j in (i + 1)..s_kv {
                    scores[i * s_kv + j] = f32::NEG_INFINITY;
                }
            }
            let sel = select_top_k(&scores, s_q, s_kv, &cfg)?;
            for i in 0..s_q {
                let want = full_sort_topk(&scores, s_kv, i, 8);
                let mut padded = want.clone();
                padded.resize(8, -1);
                let row = &sel[i * 8..(i + 1) * 8];
                ensure!(row == padded, Internal, "row {i}: {row:?} vs {padded:?}");
            }
            Ok(())
        });
    }

    #[test]
    fn property_output_invariants() {
        prop("dsa_select_invariants", 0x5E2, 120, |_i, rng| {
            let block = rng.uniform_usize(1, 9);
            let preselect = rng.uniform_usize(1, 9);
            let topk = rng.uniform_usize(1, 12);
            let extrapolation = if rng.next_f32() < 0.5 {
                Some(rng.uniform_usize(1, 10))
            } else {
                None
            };
            let cfg = small_cfg(block, topk, preselect, extrapolation);
            let s_kv = rng.uniform_usize(1, 60);
            let s_q = rng.uniform_usize(1, 5);
            let mut scores = random_vec(rng, s_q * s_kv, 10.0);
            for i in 0..s_q {
                for j in (i + 1)..s_kv {
                    scores[i * s_kv + j] = f32::NEG_INFINITY; // causal clip
                }
            }
            let sel = select_top_k(&scores, s_q, s_kv, &cfg)?;
            let k_out = topk.min(s_kv);
            ensure!(sel.len() == s_q * k_out, Internal, "output length");
            for i in 0..s_q {
                let row = &sel[i * k_out..(i + 1) * k_out];
                let split = row.iter().position(|&v| v == -1).unwrap_or(row.len());
                let (valid, pad) = (&row[..split], &row[split..]);
                ensure!(pad.iter().all(|&v| v == -1), Internal, "pad not all -1");
                ensure!(
                    valid.windows(2).all(|w| w[0] < w[1]),
                    Internal,
                    "row {i} not ascending: {valid:?}"
                );
                for &p in valid {
                    let p = p as usize;
                    ensure!(p < s_kv, Internal, "position {p} out of range");
                    ensure!(p <= i, Internal, "causal violation {p} > {i}");
                    if let Some(w) = extrapolation {
                        ensure!(p + w > i, Internal, "extrapolation violation at {p}");
                    }
                }
                // Determinism: rerun must be identical.
                let again = select_top_k(&scores, s_q, s_kv, &cfg)?;
                ensure!(again == sel, Internal, "non-deterministic");
            }
            Ok(())
        });
    }
}
