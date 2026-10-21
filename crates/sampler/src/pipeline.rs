//! The deterministic sampling pipeline and the [`Sampler`] driver.
//!
//! Implements DeepInfer's deterministic sampling (upstream README:
//! "deterministic sampling" + "batch-invariant inference") as a fixed
//! filter chain over the logits:
//!
//! `mask -> temperature -> top-k -> softmax -> top-p -> min-p ->
//! renormalize -> categorical(rng)`
//!
//! with total-order tie-breaking (probability descending, token id
//! ascending) and sequential `f32` accumulation, so a draw is a pure
//! function of `(logits, params, seed, seq_id)`. Greedy decoding
//! (`temperature <= 0`, or any magnitude below
//! [`GREEDY_TEMPERATURE_EPS`]) short-circuits to argmax with ties
//! resolved to the lowest token id and consumes no RNG draws.
//!
//! Numerical hardening: non-finite logits (`±∞`, `NaN`) are mapped to
//! `-inf` and therefore never sampled; an all-`-inf` row, an empty logit
//! vector, or a mask that allows no token is an error — never a panic,
//! never a NaN.

use deepinfer_core::mask::TokenMask;
use deepinfer_core::math::{argmax, sample_categorical, softmax_inplace, top_k_desc};
use deepinfer_core::rng::Philox;
use deepinfer_core::{ensure, Error, Result};

use crate::params::SamplingParams;

/// Seed base used when no explicit `seq_seed` is supplied: XORed into the
/// sequence id before mixing (see [`Sampler::seq_rng`]).
pub const DEFAULT_SEQ_SEED: u64 = 0x5EED_0000_0000_0001;

/// Temperatures whose magnitude falls below this threshold select the
/// greedy (argmax) path instead of the categorical path.
pub const GREEDY_TEMPERATURE_EPS: f32 = 1e-8;

/// Deterministic, batch-invariant sampler configured by
/// [`SamplingParams`].
pub struct Sampler {
    params: SamplingParams,
}

impl Sampler {
    /// Creates a sampler with the given parameters.
    pub fn new(params: SamplingParams) -> Sampler {
        Sampler { params }
    }

    /// The sampler's parameters.
    pub fn params(&self) -> &SamplingParams {
        &self.params
    }

    /// Deterministic sample of one token from `logits` (length = the
    /// effective vocabulary).
    ///
    /// Pipeline (fixed order): apply `mask` (masked-out logits → `-inf`)
    /// → temperature scaling (`logits / temperature`) → top-k (keep the
    /// `k` highest, ties → lower token id, rest → `-inf`) → softmax →
    /// top-p (keep the smallest descending-probability prefix with
    /// cumulative mass `>= top_p`, always at least the top token) → min-p
    /// (drop probabilities `< min_p * max_prob`) → renormalize →
    /// categorical draw with `u = rng.next_f32()` via
    /// [`deepinfer_core::math::sample_categorical`].
    ///
    /// Greedy path (`temperature <= 0` or `|temperature| <
    /// [`GREEDY_TEMPERATURE_EPS`]`): argmax with ties → lowest token id;
    /// consumes **no** RNG draws.
    ///
    /// Validation: `logits` non-empty and at most `u32::MAX + 1` entries;
    /// mask vocabulary must equal `logits.len()` and allow at least one
    /// token; params must pass [`SamplingParams::validate`] against
    /// `logits.len()`; at least one finite logit must survive masking.
    /// Masked-out and non-finite (`±∞`/`NaN`) logits are never returned.
    pub fn sample(&self, logits: &[f32], mask: Option<&TokenMask>, rng: &mut Philox) -> Result<u32> {
        ensure!(
            !logits.is_empty(),
            InvalidArgument,
            "sample: logits must be non-empty"
        );
        ensure!(
            logits.len() <= u32::MAX as usize,
            InvalidArgument,
            "sample: logits length {} exceeds the u32 token-id space",
            logits.len()
        );
        if let Some(m) = mask {
            ensure!(
                m.vocab_size() == logits.len(),
                DimensionMismatch,
                "sample: mask vocabulary {} does not match logits length {}",
                m.vocab_size(),
                logits.len()
            );
            ensure!(
                m.any(),
                InvalidArgument,
                "sample: token mask allows no tokens"
            );
        }
        self.params.validate(logits.len())?;

        // Stage 1 — sanitize (non-finite => -inf) and apply the mask.
        let mut work: Vec<f32> = logits
            .iter()
            .map(|&v| if v.is_finite() { v } else { f32::NEG_INFINITY })
            .collect();
        if let Some(m) = mask {
            for (i, w) in work.iter_mut().enumerate() {
                if !m.test(i as u32) {
                    *w = f32::NEG_INFINITY;
                }
            }
        }
        ensure!(
            work.iter().any(|v| v.is_finite()),
            Numerical,
            "sample: no finite logit survives masking (no valid distribution)"
        );

        // Stage 2 — greedy short-circuit (no RNG consumed).
        let t = self.params.temperature;
        if !t.is_finite() || t <= 0.0 || t.abs() < GREEDY_TEMPERATURE_EPS {
            let idx = argmax(&work)
                .ok_or_else(|| Error::Internal("sample: argmax over non-empty logits".into()))?;
            return Ok(idx as u32);
        }

        // Stage 3 — temperature scaling of the logits.
        for w in work.iter_mut() {
            *w /= t;
        }

        // Stage 4 — top-k on the scaled logits (ties -> lower token id).
        let k = self.params.top_k;
        if k > 0 && k < work.len() {
            let kept = top_k_desc(&work, k);
            let mut filtered = vec![f32::NEG_INFINITY; work.len()];
            for &i in &kept.indices {
                filtered[i as usize] = work[i as usize];
            }
            work = filtered;
        }

        // Stage 5 — softmax -> probabilities (-inf entries become exactly 0).
        softmax_inplace(&mut work);

        // Stage 6 — top-p nucleus on the probabilities.
        let p = self.params.top_p;
        if p < 1.0 {
            let mut order: Vec<u32> = (0..work.len() as u32).collect();
            order.sort_by(|&a, &b| {
                work[b as usize]
                    .total_cmp(&work[a as usize])
                    .then(a.cmp(&b))
            });
            let mut cum = 0.0f32;
            let mut keep_len = order.len(); // never reached => keep everything
            for (rank, &i) in order.iter().enumerate() {
                cum += work[i as usize];
                if cum >= p {
                    keep_len = rank + 1;
                    break;
                }
            }
            for &i in order.iter().skip(keep_len) {
                work[i as usize] = 0.0;
            }
        }

        // Stage 7 — min-p relative to the surviving maximum probability.
        let mp = self.params.min_p;
        if mp > 0.0 {
            let max_prob = work.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if max_prob > 0.0 {
                let threshold = mp * max_prob;
                for w in work.iter_mut() {
                    if *w < threshold {
                        *w = 0.0;
                    }
                }
            }
        }

        // Stage 8 — renormalize the surviving mass.
        let total: f32 = work.iter().sum();
        ensure!(
            total > 0.0 && total.is_finite(),
            Numerical,
            "sample: filtered distribution has no positive probability mass"
        );
        for w in work.iter_mut() {
            *w /= total;
        }

        // Stage 9 — categorical draw from the shared per-sequence stream.
        let u = rng.next_f32();
        let idx = sample_categorical(&work, u).ok_or_else(|| {
            Error::Numerical("sample: categorical draw failed on a valid distribution".into())
        })?;
        Ok(idx as u32)
    }

    /// Per-sequence RNG derivation — the core of batch invariance.
    ///
    /// * `seed = Some(s)`: the Philox key is `s` verbatim and the stream
    ///   (counter block) is `seq_id`.
    /// * `seed = None`: the key is derived from the sequence id by
    ///
    ///   ```text
    ///   key = splitmix64_finalize(seq_id ^ DEFAULT_SEQ_SEED)
    ///
    ///   splitmix64_finalize(z):
    ///       z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    ///       z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    ///       return z ^ (z >> 31)        (wrapping u64 arithmetic)
    ///   ```
    ///
    /// In both cases the stream depends only on `(seed, seq_id)`, so a
    /// sequence's draws are invariant to batch composition and scheduling
    /// order.
    pub fn seq_rng(seed: Option<u64>, seq_id: u64) -> Philox {
        let key = seed.unwrap_or_else(|| derive_seq_seed(seq_id));
        Philox::new(key, seq_id)
    }
}

/// Stable per-sequence seed derivation for `seq_seed = None`.
fn derive_seq_seed(seq_id: u64) -> u64 {
    splitmix64_finalize(seq_id ^ DEFAULT_SEQ_SEED)
}

/// SplitMix64 output mixer (finalizer).
fn splitmix64_finalize(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepinfer_core::prop::prop;
    use std::collections::BTreeSet;

    fn draw_n(s: &Sampler, logits: &[f32], rng: &mut Philox, n: usize) -> Vec<u32> {
        (0..n).map(|_| s.sample(logits, None, rng).unwrap()).collect()
    }

    fn seen_set(s: &Sampler, logits: &[f32], rng: &mut Philox, n: usize) -> BTreeSet<u32> {
        let mut set = BTreeSet::new();
        for _ in 0..n {
            set.insert(s.sample(logits, None, rng).unwrap());
        }
        set
    }

    fn probs_logits(probs: &[f32]) -> Vec<f32> {
        probs.iter().map(|p| p.ln()).collect()
    }

    #[test]
    fn deterministic_across_runs() {
        let s = Sampler::new(SamplingParams::default());
        let logits: Vec<f32> = (0..24).map(|i| ((i as f32) * 0.7).sin() - 0.3).collect();
        let a = draw_n(&s, &logits, &mut Philox::new(99, 1), 100);
        let b = draw_n(&s, &logits, &mut Philox::new(99, 1), 100);
        assert_eq!(a, b);
        // A second sampler instance with identical params agrees bitwise.
        let c = draw_n(
            &Sampler::new(SamplingParams::default()),
            &logits,
            &mut Philox::new(99, 1),
            100,
        );
        assert_eq!(a, c);
        assert!(!a.is_empty());
    }

    #[test]
    fn batch_invariance_interleaving() {
        // A sequence's draws must not depend on the rest of the batch:
        // sample the target sequence alone vs. interleaved with dummy
        // sequences drawing from their own (independent) streams.
        let s = Sampler::new(SamplingParams::default());
        let target: Vec<f32> = vec![0.5, 1.5, -0.5, 2.0, 0.25, -1.0, 0.75, 0.0];
        let noise: Vec<f32> = vec![-1.0, 0.5, 1.0, -0.5, 2.0, -2.0, 0.0, 1.5];

        let alone = draw_n(&s, &target, &mut Sampler::seq_rng(None, 42), 64);

        let mut rng_b = Sampler::seq_rng(None, 42);
        let mut others: Vec<Philox> = (0..5u64).map(|i| Sampler::seq_rng(None, 500 + i)).collect();
        let mut mixed = Vec::with_capacity(64);
        for _ in 0..64 {
            for o in others.iter_mut() {
                let _ = s.sample(&noise, None, o).unwrap();
            }
            mixed.push(s.sample(&target, None, &mut rng_b).unwrap());
        }
        assert_eq!(alone, mixed);
    }

    #[test]
    fn greedy_temperature_zero_is_argmax() {
        let s = Sampler::new(SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        });
        let mut rng = Philox::new(1, 1);
        let logits = vec![0.5, 2.5, 2.5, -1.0, 3.25];
        assert_eq!(s.sample(&logits, None, &mut rng).unwrap(), 4);
        // Ties -> lowest token id.
        let tied = vec![7.0, 7.0, 7.0];
        assert_eq!(s.sample(&tied, None, &mut rng).unwrap(), 0);
        // Near-zero temperature is greedy too.
        let s_eps = Sampler::new(SamplingParams {
            temperature: 1e-9,
            ..SamplingParams::default()
        });
        assert_eq!(s_eps.sample(&logits, None, &mut rng).unwrap(), 4);
    }

    #[test]
    fn greedy_consumes_no_rng() {
        let s = Sampler::new(SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        });
        let logits = vec![1.0, 3.0, 2.0];
        let mut plain = Philox::new(3, 3);
        let u1 = plain.next_f32();
        let u2 = plain.next_f32();

        let mut with_sample = Philox::new(3, 3);
        let v1 = with_sample.next_f32();
        let _ = s.sample(&logits, None, &mut with_sample).unwrap();
        let v2 = with_sample.next_f32();
        assert_eq!((u1, u2), (v1, v2));
    }

    #[test]
    fn degenerate_logits_error_or_never_sampled() {
        let s = Sampler::new(SamplingParams::default());
        let mut rng = Philox::new(4, 4);
        assert!(s.sample(&[], None, &mut rng).is_err(), "empty logits");
        assert!(
            s.sample(&[f32::NEG_INFINITY; 6], None, &mut rng).is_err(),
            "all -inf"
        );
        // NaN / +inf are treated as -inf: never sampled, no NaN escapes.
        for bad in [f32::NAN, f32::INFINITY] {
            let logits = vec![bad, 3.0, 1.0];
            let seen = seen_set(&s, &logits, &mut rng, 300);
            let expect: BTreeSet<u32> = [1u32, 2].into_iter().collect();
            assert_eq!(seen, expect, "logit[0] = {bad}");
        }
        // Greedy over NaN picks the finite argmax.
        let g = Sampler::new(SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        });
        assert_eq!(g.sample(&[f32::NAN, 3.0, 1.0], None, &mut rng).unwrap(), 1);
    }

    #[test]
    fn sample_rejects_invalid_params() {
        let mut rng = Philox::new(1, 1);
        let s = Sampler::new(SamplingParams {
            temperature: -1.0,
            ..SamplingParams::default()
        });
        assert!(s.sample(&[1.0, 2.0], None, &mut rng).is_err());
        let s = Sampler::new(SamplingParams {
            top_k: 5,
            ..SamplingParams::default()
        });
        assert!(
            s.sample(&[1.0, 2.0], None, &mut rng).is_err(),
            "top_k > vocab"
        );
    }

    #[test]
    fn mask_enforced_and_validated() {
        let g = Sampler::new(SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        });
        let mut rng = Philox::new(6, 6);
        let logits = vec![9.0, 8.0, 7.0, 6.0, 5.0, 1.0, 0.0, 2.0];

        let mut mask = TokenMask::zeros(8);
        mask.set(5);
        mask.set(7);
        assert_eq!(g.sample(&logits, Some(&mask), &mut rng).unwrap(), 7);
        mask.clear(7);
        assert_eq!(g.sample(&logits, Some(&mask), &mut rng).unwrap(), 5);

        let empty = TokenMask::zeros(8);
        assert!(g.sample(&logits, Some(&empty), &mut rng).is_err());
        let small = TokenMask::all_ones(4);
        assert!(
            g.sample(&logits, Some(&small), &mut rng).is_err(),
            "mask vocab mismatch"
        );
        // Mask allowing only -inf logits: no valid distribution.
        let mut m2 = TokenMask::zeros(3);
        m2.set(0);
        assert!(g.sample(&[f32::NEG_INFINITY, 1.0, 2.0], Some(&m2), &mut rng).is_err());
    }

    #[test]
    fn top_k_restricts_support() {
        let logits = vec![0.1, 3.0, -0.5, 2.5, 2.5, 0.0, -3.0, 1.0];
        // Top-3 by (value desc, id asc): 1 (3.0), 3 (2.5), 4 (2.5).
        let s = Sampler::new(SamplingParams {
            top_k: 3,
            ..SamplingParams::default()
        });
        let seen = seen_set(&s, &logits, &mut Philox::new(21, 4), 400);
        let expect: BTreeSet<u32> = [1u32, 3, 4].into_iter().collect();
        assert_eq!(seen, expect);
    }

    #[test]
    fn top_k_uniform_logits_uniform_draws() {
        let logits = vec![1.25f32; 50];
        let s = Sampler::new(SamplingParams {
            top_k: 10,
            ..SamplingParams::default()
        });
        let mut rng = Philox::new(2, 2);
        let mut counts = vec![0usize; 50];
        for _ in 0..5000 {
            counts[s.sample(&logits, None, &mut rng).unwrap() as usize] += 1;
        }
        for (i, &c) in counts.iter().enumerate() {
            if i < 10 {
                assert!(
                    (350..=650).contains(&c),
                    "top-k id {i} drawn {c}/5000 times (expected ~500)"
                );
            } else {
                assert_eq!(c, 0, "id {i} sampled outside the top-k set");
            }
        }
    }

    #[test]
    fn top_p_hand_computed_prefixes() {
        // probs ~ [0.5, 0.3, 0.15, 0.05]:
        //   top_p 0.45 -> prefix [0]        (0.5 >= 0.45)
        //   top_p 0.75 -> prefix [0, 1]     (0.5 < 0.75 <= 0.8)
        //   top_p 0.93 -> prefix [0, 1, 2]  (0.8 < 0.93 <= 0.95)
        let logits = probs_logits(&[0.5, 0.3, 0.15, 0.05]);
        for (top_p, allowed) in [
            (0.45f32, vec![0u32]),
            (0.75f32, vec![0u32, 1]),
            (0.93f32, vec![0u32, 1, 2]),
        ] {
            let s = Sampler::new(SamplingParams {
                top_p,
                ..SamplingParams::default()
            });
            let seen = seen_set(&s, &logits, &mut Philox::new(5, 5), 600);
            let expect: BTreeSet<u32> = allowed.into_iter().collect();
            assert_eq!(seen, expect, "top_p = {top_p}");
        }
        // top_p = 1 is off: the full support survives.
        let s = Sampler::new(SamplingParams {
            top_p: 1.0,
            ..SamplingParams::default()
        });
        assert_eq!(seen_set(&s, &logits, &mut Philox::new(5, 6), 2000).len(), 4);
    }

    #[test]
    fn min_p_drops_tail() {
        let logits = probs_logits(&[0.5, 0.3, 0.15, 0.05]);
        // min_p 0.2 -> threshold 0.1: keep {0.5, 0.3, 0.15}, drop 0.05.
        let s = Sampler::new(SamplingParams {
            min_p: 0.2,
            ..SamplingParams::default()
        });
        let seen = seen_set(&s, &logits, &mut Philox::new(7, 7), 600);
        let expect: BTreeSet<u32> = [0u32, 1, 2].into_iter().collect();
        assert_eq!(seen, expect);
        // min_p 0.4 -> threshold 0.2: keep {0.5, 0.3}.
        let s = Sampler::new(SamplingParams {
            min_p: 0.4,
            ..SamplingParams::default()
        });
        let seen = seen_set(&s, &logits, &mut Philox::new(7, 8), 600);
        let expect: BTreeSet<u32> = [0u32, 1].into_iter().collect();
        assert_eq!(seen, expect);
    }

    #[test]
    fn top_k_then_top_p_pipeline_order() {
        // probs ~ [0.5, 0.3, 0.15, 0.05]; top-k 3 keeps ids {0,1,2}; the
        // softmax after top-k renormalizes them to ~[0.526, 0.316, 0.158];
        // top-p 0.6 then keeps the smallest prefix with cumsum >= 0.6,
        // which is {0, 1} (0.526 < 0.6 <= 0.842).
        let logits = probs_logits(&[0.5, 0.3, 0.15, 0.05]);
        let s = Sampler::new(SamplingParams {
            top_k: 3,
            top_p: 0.6,
            ..SamplingParams::default()
        });
        let seen = seen_set(&s, &logits, &mut Philox::new(11, 11), 600);
        let expect: BTreeSet<u32> = [0u32, 1].into_iter().collect();
        assert_eq!(seen, expect);
    }

    #[test]
    fn default_params_keep_full_support() {
        let logits: Vec<f32> = (0..8).map(|i| (i as f32) * 0.3 - 1.0).collect();
        let s = Sampler::new(SamplingParams::default());
        let seen = seen_set(&s, &logits, &mut Philox::new(12, 12), 2000);
        assert_eq!(seen.len(), 8);
    }

    #[test]
    fn temperature_sharpens_or_flattens() {
        let logits = probs_logits(&[0.7, 0.3]);
        let hot = Sampler::new(SamplingParams {
            temperature: 0.2,
            ..SamplingParams::default()
        });
        let cold = Sampler::new(SamplingParams {
            temperature: 3.0,
            ..SamplingParams::default()
        });
        let mut rh = Philox::new(8, 8);
        let mut rc = Philox::new(8, 8);
        let (mut h1, mut c1) = (0usize, 0usize);
        for _ in 0..1000 {
            if hot.sample(&logits, None, &mut rh).unwrap() == 1 {
                h1 += 1;
            }
            if cold.sample(&logits, None, &mut rc).unwrap() == 1 {
                c1 += 1;
            }
        }
        // temp 0.2: p(1) ~ 0.014; temp 3.0: p(1) ~ 0.43.
        assert!(h1 < 80, "low-temperature tail drawn {h1}/1000 times");
        assert!(c1 > 250, "high-temperature tail drawn {c1}/1000 times");
    }

    #[test]
    fn seq_rng_stream_properties() {
        // An explicit seed is used verbatim as the Philox key.
        let mut f = Sampler::seq_rng(Some(1234), 77);
        let mut g = Philox::new(1234, 77);
        for _ in 0..32 {
            assert_eq!(f.next_u32(), g.next_u32());
        }
        // Deterministic across calls.
        let mut a = Sampler::seq_rng(None, 9);
        let mut b = Sampler::seq_rng(None, 9);
        for _ in 0..64 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
        // Different sequences get independent streams.
        let mut c = Sampler::seq_rng(None, 10);
        let mut diffs = 0;
        for _ in 0..64 {
            if a.next_u32() != c.next_u32() {
                diffs += 1;
            }
        }
        assert!(diffs > 32, "streams for seq 9 and 10 look identical");
        // The None-derivation actually mixes (not the raw DEFAULT_SEQ_SEED).
        let mut d = Philox::new(DEFAULT_SEQ_SEED, 9);
        let mut e = Sampler::seq_rng(None, 9);
        let mut diffs = 0;
        for _ in 0..64 {
            if d.next_u32() != e.next_u32() {
                diffs += 1;
            }
        }
        assert!(diffs > 32, "derived seed equals the raw DEFAULT_SEQ_SEED");
    }

    #[test]
    fn prop_greedy_matches_argmax() {
        prop("sampler-greedy-argmax", 0x6EA0, 150, |_, rng| {
            let n = 5 + (rng.next_u32() as usize % 20);
            let logits: Vec<f32> = (0..n).map(|_| rng.normal_f32()).collect();
            let s = Sampler::new(SamplingParams {
                temperature: 0.0,
                ..SamplingParams::default()
            });
            let mut r = Philox::new(rng.next_u64_from_two(), rng.next_u64_from_two());
            let tok = s.sample(&logits, None, &mut r)?;
            ensure!((tok as usize) < n, Internal, "token {tok} out of range");
            let am = argmax(&logits).unwrap_or(0) as u32;
            ensure!(tok == am, Internal, "greedy token {tok} != argmax {am}");
            Ok(())
        });
    }

    #[test]
    fn prop_mask_never_violated() {
        prop("sampler-mask-enforcement", 0x35C1, 200, |_, rng| {
            let n = 8 + (rng.next_u32() as usize % 40);
            let logits: Vec<f32> = (0..n).map(|_| rng.normal_f32()).collect();
            let mut mask = TokenMask::zeros(n);
            for t in 0..n as u32 {
                if rng.next_f32() < 0.35 {
                    mask.set(t);
                }
            }
            if !mask.any() {
                mask.set(rng.next_u32() % n as u32);
            }
            let s = Sampler::new(SamplingParams {
                temperature: 0.9,
                ..SamplingParams::default()
            });
            let mut r = Philox::new(rng.next_u64_from_two(), 7);
            for step in 0..25 {
                let tok = s.sample(&logits, Some(&mask), &mut r)?;
                ensure!(
                    mask.test(tok),
                    Internal,
                    "step {step}: sampled disallowed token {tok}"
                );
            }
            Ok(())
        });
    }

    #[test]
    fn prop_top_p_minimal_prefix() {
        prop("sampler-topp-minimal-prefix", 0x70D5, 150, |_, rng| {
            let n = 4 + (rng.next_u32() as usize % 20);
            let logits: Vec<f32> = (0..n).map(|_| rng.normal_f32()).collect();
            let top_p = 0.05 + rng.next_f32() * 0.9;

            // Independent transcription of the spec: sort probs descending
            // (ties -> lower id), keep the smallest prefix with cumsum >= top_p.
            let mut probs = logits.clone();
            softmax_inplace(&mut probs);
            let mut order: Vec<u32> = (0..n as u32).collect();
            order.sort_by(|&a, &b| {
                probs[b as usize]
                    .total_cmp(&probs[a as usize])
                    .then(a.cmp(&b))
            });
            let mut cum = 0.0f32;
            let mut keep = BTreeSet::new();
            let mut cum_before_last_kept = 0.0f32;
            for &i in &order {
                cum_before_last_kept = cum;
                keep.insert(i);
                cum += probs[i as usize];
                if cum >= top_p {
                    break;
                }
            }
            ensure!(!keep.is_empty(), Internal, "empty top-p keep set");
            // Minimality: without the last kept token the prefix misses top_p.
            if keep.len() < n {
                ensure!(
                    cum_before_last_kept < top_p,
                    Internal,
                    "top-p prefix not minimal (cum {cum_before_last_kept} >= {top_p})"
                );
            }

            let s = Sampler::new(SamplingParams {
                top_p,
                ..SamplingParams::default()
            });
            let mut r = Philox::new(rng.next_u64_from_two(), rng.next_u64_from_two());
            for step in 0..40 {
                let tok = s.sample(&logits, None, &mut r)?;
                ensure!(
                    keep.contains(&tok),
                    Internal,
                    "step {step}: token {tok} outside the top-p keep set"
                );
            }
            Ok(())
        });
    }

    #[test]
    fn prop_min_p_support() {
        prop("sampler-min-p-support", 0x31F7, 150, |_, rng| {
            let n = 4 + (rng.next_u32() as usize % 20);
            let logits: Vec<f32> = (0..n).map(|_| rng.normal_f32()).collect();
            let min_p = rng.next_f32() * 0.9;

            let mut probs = logits.clone();
            softmax_inplace(&mut probs);
            let max = probs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let threshold = min_p * max;
            let keep: BTreeSet<u32> = probs
                .iter()
                .enumerate()
                .filter(|(_, p)| **p >= threshold)
                .map(|(i, _)| i as u32)
                .collect();

            let s = Sampler::new(SamplingParams {
                min_p,
                ..SamplingParams::default()
            });
            let mut r = Philox::new(rng.next_u64_from_two(), rng.next_u64_from_two());
            for step in 0..40 {
                let tok = s.sample(&logits, None, &mut r)?;
                ensure!(
                    keep.contains(&tok),
                    Internal,
                    "step {step}: token {tok} outside the min-p keep set"
                );
            }
            Ok(())
        });
    }

    #[test]
    fn prop_top_k_support() {
        prop("sampler-topk-support", 0x70B, 150, |i, rng| {
            let n = 6 + (i as usize % 30);
            let logits: Vec<f32> = (0..n).map(|_| rng.normal_f32()).collect();
            let k = 1 + rng.next_u32() as usize % n;
            let keep: BTreeSet<u32> = top_k_desc(&logits, k).indices.into_iter().collect();
            ensure!(keep.len() == k.min(n), Internal, "top-k set size");

            let s = Sampler::new(SamplingParams {
                top_k: k,
                ..SamplingParams::default()
            });
            let mut r = Philox::new(rng.next_u64_from_two(), rng.next_u64_from_two());
            for step in 0..40 {
                let tok = s.sample(&logits, None, &mut r)?;
                ensure!(
                    keep.contains(&tok),
                    Internal,
                    "step {step}: token {tok} outside the top-k set"
                );
            }
            Ok(())
        });
    }

    #[test]
    fn prop_distribution_skew() {
        // With a skewed distribution the head token must be drawn more
        // often than the tail token at temperature 1.
        prop("sampler-distribution-sanity", 0x0FF, 120, |_, rng| {
            let logits = probs_logits(&[0.8, 0.2]);
            let s = Sampler::new(SamplingParams::default());
            let mut r = Philox::new(rng.next_u64_from_two(), rng.next_u64_from_two());
            let (mut head, mut tail) = (0usize, 0usize);
            for _ in 0..500 {
                if s.sample(&logits, None, &mut r).unwrap() == 0 {
                    head += 1;
                } else {
                    tail += 1;
                }
            }
            ensure!(head > tail, Internal, "head {head} <= tail {tail}");
            ensure!(head > 300, Internal, "head {head}/500 too low for p = 0.8");
            Ok(())
        });
    }

    #[test]
    fn seq_rng_derivation_golden_values() {
        // Regression pins for the documented None-seed derivation:
        // key = splitmix64_finalize(seq_id ^ DEFAULT_SEQ_SEED). These
        // values must never change — downstream determinism depends on
        // them.
        assert_eq!(derive_seq_seed(0), 0xa7d3_19ac_150b_e659);
        assert_eq!(derive_seq_seed(42), 0x8ec7_418b_f5f9_f01c);
        assert!((Sampler::seq_rng(None, 0).next_f32() - 0.62999576).abs() < 1e-7);
        assert!((Sampler::seq_rng(None, 42).next_f32() - 0.57701564).abs() < 1e-7);
    }
}
