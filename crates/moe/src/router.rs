//! DeepSeekMoE router: sigmoid scoring, optional `e_score_correction_bias`
//! ("noaux-tc") for top-k selection, optional V3.2-style grouped experts, and
//! bias-free weight renormalization.
//!
//! Upstream: DeepSeek-V3 technical report §2.1.1 ("Auxiliary-Loss-Free Load
//! Balancing" strategy): the per-expert bias adjusts the **selection** only,
//! while the routing weights are the plain sigmoid affinities of the selected
//! experts, renormalized to sum to one.

use crate::{ExpertWeights, MoeConfig, Routing};
use deepinfer_core::math::top_k_desc;
use deepinfer_core::rng::Philox;
use deepinfer_core::{ensure, Error, Result};

/// Full weight set of one DeepSeekMoE layer (router + shared + routed
/// experts).
#[derive(Debug, Clone, PartialEq)]
pub struct MoeWeights {
    /// Router gate matrix, row-major `[num_routed, hidden]` (`gate · x` are
    /// the pre-activation scores).
    pub gate: Vec<f32>,
    /// `e_score_correction_bias`, `[num_routed]` — participates in top-k
    /// **selection only** when `cfg.use_noaux_tc_bias` (never in the weights).
    pub gate_bias: Vec<f32>,
    /// Shared (always-active) experts, added unweighted by
    /// [`MoeWeights::shared_forward`].
    pub shared: Vec<ExpertWeights>,
    /// Routed experts, indexed by expert id.
    pub routed: Vec<ExpertWeights>,
}

impl MoeWeights {
    /// Synthetic initialization for tests and reference runs: every matrix is
    /// filled i.i.d. uniform in `±1/sqrt(hidden)` (deterministic given
    /// `rng`).
    ///
    /// Infallible by SPEC: dimension products that overflow `usize` degenerate
    /// to empty tensors, which later fail shape validation with a clear error
    /// instead of panicking or aborting.
    pub fn from_config(cfg: &MoeConfig, rng: &mut Philox) -> MoeWeights {
        let s = if cfg.hidden > 0 {
            1.0 / (cfg.hidden as f32).sqrt()
        } else {
            0.0
        };
        let dim = |a: usize, b: usize| a.checked_mul(b).unwrap_or(0);
        let uniform = |rng: &mut Philox, n: usize| -> Vec<f32> {
            let mut v = vec![0.0f32; n];
            rng.fill_uniform(&mut v, -s, s);
            v
        };
        let gate = uniform(rng, dim(cfg.num_routed, cfg.hidden));
        let gate_bias = uniform(rng, cfg.num_routed);
        let mut shared = Vec::with_capacity(cfg.num_shared);
        for _ in 0..cfg.num_shared {
            shared.push(ExpertWeights {
                w_gate_up: uniform(rng, dim(2 * cfg.intermediate, cfg.hidden)),
                w_down: uniform(rng, dim(cfg.hidden, cfg.intermediate)),
            });
        }
        let mut routed = Vec::with_capacity(cfg.num_routed);
        for _ in 0..cfg.num_routed {
            routed.push(ExpertWeights {
                w_gate_up: uniform(rng, dim(2 * cfg.intermediate, cfg.hidden)),
                w_down: uniform(rng, dim(cfg.hidden, cfg.intermediate)),
            });
        }
        MoeWeights {
            gate,
            gate_bias,
            shared,
            routed,
        }
    }

    /// Routes a single token (see the module docs for the full semantics).
    ///
    /// Errors on shape mismatches, non-finite `x`, or an invalid
    /// [`MoeConfig`].
    pub fn route(&self, cfg: &MoeConfig, x: &[f32]) -> Result<Routing> {
        validate_config(cfg)?;
        self.validate_against(cfg)?;
        ensure!(
            x.len() == cfg.hidden,
            DimensionMismatch,
            "route: x.len() = {} but hidden = {}",
            x.len(),
            cfg.hidden
        );
        ensure!(
            x.iter().all(|v| v.is_finite()),
            InvalidArgument,
            "route: x contains non-finite values"
        );
        Ok(self.route_validated(cfg, x))
    }

    /// Batch-routes `tokens` row-major rows of `xs`. Deterministic; ties by
    /// expert id ascending (see [`Routing`]).
    pub fn route_batch(&self, cfg: &MoeConfig, xs: &[f32], tokens: usize) -> Result<Vec<Routing>> {
        validate_config(cfg)?;
        self.validate_against(cfg)?;
        let need = tokens.checked_mul(cfg.hidden).ok_or_else(|| {
            Error::CapacityExceeded(format!(
                "route_batch: tokens ({tokens}) * hidden ({}) overflows usize",
                cfg.hidden
            ))
        })?;
        ensure!(
            xs.len() == need,
            DimensionMismatch,
            "route_batch: xs.len() = {} but tokens*hidden = {need}",
            xs.len()
        );
        ensure!(
            xs.iter().all(|v| v.is_finite()),
            InvalidArgument,
            "route_batch: xs contains non-finite values"
        );
        let mut out = Vec::with_capacity(tokens);
        for t in 0..tokens {
            out.push(self.route_validated(cfg, &xs[t * cfg.hidden..(t + 1) * cfg.hidden]));
        }
        Ok(out)
    }

    /// Core routing for one already-validated token row.
    fn route_validated(&self, cfg: &MoeConfig, x: &[f32]) -> Routing {
        let h = cfg.hidden;
        let n = cfg.num_routed;
        // Bias-free sigmoid scores (used for the weights) and selection
        // scores (bias added only when noaux-tc is enabled).
        let mut nobias = vec![0.0f32; n];
        let mut sel = vec![0.0f32; n];
        for e in 0..n {
            let row = &self.gate[e * h..(e + 1) * h];
            let z: f32 = row.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
            let s = sigmoid(z);
            nobias[e] = s;
            sel[e] = if cfg.use_noaux_tc_bias {
                s + self.gate_bias[e]
            } else {
                s
            };
        }
        let selected: Vec<u32> = if cfg.group_size > 1 {
            let g = cfg.group_size;
            let num_groups = n / g;
            let k_groups = cfg.num_activated / g;
            // Group score = best selection score among its members.
            let mut gscore = vec![f32::NEG_INFINITY; num_groups];
            for e in 0..n {
                let gi = e / g;
                if sel[e] > gscore[gi] {
                    gscore[gi] = sel[e];
                }
            }
            let top = top_k_desc(&gscore, k_groups);
            let mut out = Vec::with_capacity(cfg.num_activated);
            for &gi in &top.indices {
                let base = gi as usize * g;
                for e in base..base + g {
                    out.push(e as u32);
                }
            }
            out
        } else {
            top_k_desc(&sel, cfg.num_activated).indices
        };
        // Weights: bias-free sigmoids of the selected experts, renormalized.
        // If every selected score underflowed to exactly zero, fall back to a
        // uniform distribution (documented; never NaN).
        let sum: f32 = selected.iter().map(|&e| nobias[e as usize]).sum();
        let topk_weights: Vec<f32> = if sum > 0.0 && sum.is_finite() {
            selected
                .iter()
                .map(|&e| nobias[e as usize] / sum)
                .collect()
        } else {
            vec![1.0 / selected.len() as f32; selected.len()]
        };
        Routing {
            topk_idx: selected,
            topk_weights,
        }
    }

    /// Validates that this weight set matches `cfg`'s declared shapes.
    pub(crate) fn validate_against(&self, cfg: &MoeConfig) -> Result<()> {
        let gate_len = cfg.num_routed.checked_mul(cfg.hidden).ok_or_else(|| {
            Error::CapacityExceeded(format!(
                "MoeWeights: num_routed ({}) * hidden ({}) overflows usize",
                cfg.num_routed, cfg.hidden
            ))
        })?;
        ensure!(
            self.gate.len() == gate_len,
            DimensionMismatch,
            "MoeWeights: gate.len() = {} but num_routed*hidden = {gate_len}",
            self.gate.len()
        );
        if cfg.use_noaux_tc_bias {
            ensure!(
                self.gate_bias.len() == cfg.num_routed,
                DimensionMismatch,
                "MoeWeights: gate_bias.len() = {} but num_routed = {}",
                self.gate_bias.len(),
                cfg.num_routed
            );
        }
        ensure!(
            self.routed.len() == cfg.num_routed,
            DimensionMismatch,
            "MoeWeights: routed.len() = {} but num_routed = {}",
            self.routed.len(),
            cfg.num_routed
        );
        ensure!(
            self.shared.len() == cfg.num_shared,
            DimensionMismatch,
            "MoeWeights: shared.len() = {} but num_shared = {}",
            self.shared.len(),
            cfg.num_shared
        );
        let i_len = cfg.intermediate.checked_mul(cfg.hidden).ok_or_else(|| {
            Error::CapacityExceeded(format!(
                "MoeWeights: intermediate ({}) * hidden ({}) overflows usize",
                cfg.intermediate, cfg.hidden
            ))
        })?;
        let gu_len = i_len
            .checked_mul(2)
            .ok_or_else(|| Error::CapacityExceeded("MoeWeights: 2*I*H overflows usize".into()))?;
        let check = |label: &str, ew: &ExpertWeights| -> Result<()> {
            ensure!(
                ew.w_gate_up.len() == gu_len && ew.w_down.len() == i_len,
                DimensionMismatch,
                "MoeWeights: {label} has w_gate_up.len() = {} and w_down.len() = {} \
                 but expects [{},{}] and [{},{}]",
                ew.w_gate_up.len(),
                ew.w_down.len(),
                gu_len,
                i_len,
                cfg.hidden,
                cfg.intermediate
            );
            Ok(())
        };
        for (i, ew) in self.routed.iter().enumerate() {
            check(&format!("routed[{i}]"), ew)?;
        }
        for (i, ew) in self.shared.iter().enumerate() {
            check(&format!("shared[{i}]"), ew)?;
        }
        Ok(())
    }
}

/// Numerical sigmoid `1 / (1 + e^-z)`.
fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// Validates the structural invariants of a [`MoeConfig`].
pub(crate) fn validate_config(cfg: &MoeConfig) -> Result<()> {
    ensure!(cfg.hidden >= 1, InvalidArgument, "MoeConfig: hidden must be >= 1");
    ensure!(
        cfg.intermediate >= 1,
        InvalidArgument,
        "MoeConfig: intermediate must be >= 1"
    );
    ensure!(
        cfg.num_routed >= 1,
        InvalidArgument,
        "MoeConfig: num_routed must be >= 1"
    );
    ensure!(
        cfg.num_activated >= 1,
        InvalidArgument,
        "MoeConfig: num_activated must be >= 1"
    );
    ensure!(
        cfg.num_activated <= cfg.num_routed,
        InvalidArgument,
        "MoeConfig: num_activated ({}) must be <= num_routed ({})",
        cfg.num_activated,
        cfg.num_routed
    );
    ensure!(
        cfg.group_size >= 1,
        InvalidArgument,
        "MoeConfig: group_size must be >= 1"
    );
    if cfg.group_size > 1 {
        ensure!(
            cfg.num_routed % cfg.group_size == 0,
            InvalidArgument,
            "MoeConfig: num_routed ({}) must be divisible by group_size ({})",
            cfg.num_routed,
            cfg.group_size
        );
        ensure!(
            cfg.num_activated % cfg.group_size == 0,
            InvalidArgument,
            "MoeConfig: num_activated ({}) must be divisible by group_size ({})",
            cfg.num_activated,
            cfg.group_size
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepinfer_core::prop::{prop, random_vec};

    fn sigmoid64(z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    /// Shape-valid placeholder experts (routing-only tests never run them).
    fn dummy_experts(n: usize, hidden: usize, intermediate: usize) -> Vec<ExpertWeights> {
        (0..n)
            .map(|_| ExpertWeights {
                w_gate_up: vec![0.0; 2 * intermediate * hidden],
                w_down: vec![0.0; hidden * intermediate],
            })
            .collect()
    }

    /// Builds a router with explicit gate rows so sigmoid scores are known.
    fn hand_router() -> (MoeConfig, MoeWeights) {
        // hidden = 4; expert e has score sigmoid(e_row · x).
        let cfg = MoeConfig {
            hidden: 4,
            intermediate: 2,
            num_routed: 4,
            num_activated: 2,
            num_shared: 0,
            group_size: 1,
            use_noaux_tc_bias: true,
        };
        let mut w = MoeWeights {
            gate: vec![0.0; 4 * 4],
            gate_bias: vec![0.0; 4],
            shared: Vec::new(),
            routed: dummy_experts(4, 4, 2),
        };
        // x = [1, 0, 0, 0] → scores sigmoid(z_e) with z = gate[e][0].
        let zs = [2.0f32, 1.5f32, 1.0f32, 0.0f32];
        for (e, &z) in zs.iter().enumerate() {
            w.gate[e * 4] = z;
        }
        (cfg, w)
    }

    #[test]
    fn noaux_tc_bias_selects_but_never_weights() {
        let (cfg, mut w) = hand_router();
        let x = vec![1.0, 0.0, 0.0, 0.0];
        // Without bias: top-2 = experts 0, 1 (scores .881, .818).
        let plain = w.route(&cfg, &x).unwrap();
        assert_eq!(plain.topk_idx, vec![0, 1]);
        let s0: f64 = sigmoid64(2.0);
        let s1: f64 = sigmoid64(1.5);
        assert!((plain.topk_weights[0] as f64 - s0 / (s0 + s1)).abs() < 1e-6);
        assert!((plain.topk_weights[1] as f64 - s1 / (s0 + s1)).abs() < 1e-6);

        // With a large bias on expert 3 the selection changes …
        w.gate_bias[3] = 5.0;
        let biased = w.route(&cfg, &x).unwrap();
        assert_eq!(biased.topk_idx, vec![3, 0]);
        // … but the weights stay the bias-free renormalized sigmoids.
        let s3: f64 = sigmoid64(0.0);
        assert!((biased.topk_weights[0] as f64 - s3 / (s0 + s3)).abs() < 1e-6);
        assert!((biased.topk_weights[1] as f64 - s0 / (s0 + s3)).abs() < 1e-6);
        // The biased selection score itself never appears in the weights.
        assert!(biased.topk_weights[0] < 0.75);
    }

    #[test]
    fn weights_renormalize_to_one() {
        prop("routing renormalization", 0x1207, 100, |i, mut rng| {
            let cfg = MoeConfig {
                hidden: rng.uniform_usize(1, 8),
                intermediate: 2,
                num_routed: rng.uniform_usize(2, 8),
                num_activated: 0, // patched below
                num_shared: 0,
                group_size: 1,
                use_noaux_tc_bias: rng.next_f32() < 0.5,
            };
            let cfg = MoeConfig {
                num_activated: rng.uniform_usize(1, cfg.num_routed + 1),
                ..cfg
            };
            let mut w = MoeWeights::from_config(&cfg, &mut rng);
            for b in w.gate_bias.iter_mut() {
                *b = rng.uniform_f32(-1.0, 1.0);
            }
            let x = random_vec(&mut rng, cfg.hidden, 2.0);
            let r = w.route(&cfg, &x)?;
            assert_eq!(r.topk_idx.len(), cfg.num_activated);
            assert_eq!(r.topk_weights.len(), cfg.num_activated);
            let sum: f32 = r.topk_weights.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "weights sum to {sum}, expected 1"
            );
            assert!(
                r.topk_weights.iter().all(|&v| v >= 0.0 && v.is_finite()),
                "weights must be finite and non-negative"
            );
            let mut sorted = r.topk_idx.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(
                sorted.len(),
                r.topk_idx.len(),
                "selected experts must be distinct"
            );
            let _ = i;
            Ok(())
        });
    }

    #[test]
    fn group_mode_selects_whole_groups() {
        // 8 experts, groups of 4: group 0 = {0..4}, group 1 = {4..8}.
        let cfg = MoeConfig {
            hidden: 2,
            intermediate: 2,
            num_routed: 8,
            num_activated: 4,
            num_shared: 0,
            group_size: 4,
            use_noaux_tc_bias: false,
        };
        let mut w = MoeWeights {
            gate: vec![0.0; 8 * 2],
            gate_bias: vec![0.0; 8],
            shared: Vec::new(),
            routed: dummy_experts(8, 2, 2),
        };
        // Group scores are the max member score: make group 1 win.
        w.gate[4 * 2] = 3.0; // expert 4 → sigmoid(3) ≈ 0.953
        w.gate[0 * 2] = 1.0; // expert 0 → sigmoid(1) ≈ 0.731
        let x = vec![1.0, 0.0];
        let r = w.route(&cfg, &x).unwrap();
        assert_eq!(r.topk_idx, vec![4, 5, 6, 7], "all of group 1 selected");
        // Weights: sigmoids of all four experts of group 1, renormalized.
        let s: Vec<f64> = [4, 5, 6, 7].iter().map(|&e| sigmoid64(w.gate[e * 2] as f64)).collect();
        let tot: f64 = s.iter().sum();
        for j in 0..4 {
            assert!((r.topk_weights[j] as f64 - s[j] / tot).abs() < 1e-6);
        }
        let sum: f32 = r.topk_weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Group selection follows the bias when noaux-tc is on.
        let mut cfg_b = cfg.clone();
        cfg_b.use_noaux_tc_bias = true;
        w.gate_bias[0] = 10.0; // boosts group 0's max selection score
        let r2 = w.route(&cfg_b, &x).unwrap();
        assert_eq!(r2.topk_idx, vec![0, 1, 2, 3]);
    }

    #[test]
    fn group_mode_rejects_bad_divisibility() {
        let mut cfg = MoeConfig {
            hidden: 2,
            intermediate: 2,
            num_routed: 8,
            num_activated: 3,
            num_shared: 0,
            group_size: 2,
            use_noaux_tc_bias: false,
        };
        assert!(validate_config(&cfg).is_err(), "k=3 not divisible by g=2");
        cfg.num_activated = 4;
        cfg.num_routed = 7;
        assert!(validate_config(&cfg).is_err(), "num_routed=7 not divisible by g=2");
    }

    #[test]
    fn route_is_deterministic_and_ties_break_low() {
        let (cfg, mut w) = hand_router();
        // Equal scores for experts 0 and 1 → tie broken by lower id.
        w.gate[0 * 4] = 1.5;
        let x = vec![1.0, 0.0, 0.0, 0.0];
        let cfg_no_bias = MoeConfig {
            use_noaux_tc_bias: false,
            ..cfg.clone()
        };
        let a = w.route(&cfg_no_bias, &x).unwrap();
        let b = w.route(&cfg_no_bias, &x).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.topk_idx, vec![0, 1], "tie must pick lower expert id first");
    }

    #[test]
    fn route_validates_shapes() {
        let (cfg, w) = hand_router();
        assert!(w.route(&cfg, &[0.0; 3]).is_err());
        assert!(w.route(&cfg, &[f32::INFINITY, 0.0, 0.0, 0.0]).is_err());
        assert!(w.route_batch(&cfg, &[0.0; 4], 2).is_err());
        let mut broken = w.clone();
        broken.gate.truncate(3);
        assert!(broken.route(&cfg, &[0.0; 4]).is_err());
    }

    #[test]
    fn route_batch_matches_route() {
        prop("route_batch == route", 0x1208, 100, |i, mut rng| {
            let cfg = MoeConfig {
                hidden: rng.uniform_usize(1, 6),
                intermediate: 2,
                num_routed: rng.uniform_usize(2, 6),
                num_activated: 0,
                num_shared: 0,
                group_size: 1,
                use_noaux_tc_bias: true,
            };
            let cfg = MoeConfig {
                num_activated: rng.uniform_usize(1, cfg.num_routed + 1),
                ..cfg
            };
            let mut w = MoeWeights::from_config(&cfg, &mut rng);
            for b in w.gate_bias.iter_mut() {
                *b = rng.uniform_f32(-1.0, 1.0);
            }
            let tokens = rng.uniform_usize(1, 6);
            let xs = random_vec(&mut rng, tokens * cfg.hidden, 2.0);
            let batch = w.route_batch(&cfg, &xs, tokens)?;
            for t in 0..tokens {
                let one = w.route(&cfg, &xs[t * cfg.hidden..(t + 1) * cfg.hidden])?;
                assert_eq!(batch[t], one, "token {t} routing differs");
            }
            let _ = i;
            Ok(())
        });
    }
}
