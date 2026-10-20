//! # deepinfer-moe
//!
//! DeepSeekMoE routing and expert compute for DeepInfer-rs — the CPU
//! reference port of the Mixture-of-Experts layer described in the
//! DeepSeek-V3 technical report §2.1.1 (fine-grained expert segmentation +
//! shared experts + auxiliary-loss-free "noaux-tc" balancing), including the
//! V3.2-style grouped-expert routing extension.
//!
//! * **Router** ([`MoeWeights::route`]): `scores = sigmoid(gate · x)`. With
//!   `use_noaux_tc_bias = true` the `e_score_correction_bias` is added to the
//!   scores **for top-k selection only** — the routing weights are the
//!   bias-free sigmoids of the selected experts, renormalized to sum to one
//!   (V3 "noaux-tc" semantics). Ties resolve to the lower expert id.
//! * **Grouped experts** (`group_size = g > 1`): experts are partitioned into
//!   `num_routed / g` consecutive groups; routing selects the top
//!   `num_activated / g` groups by their members' best selection score and
//!   activates **all** `g` experts of every selected group.
//! * **Experts** ([`MoeWeights::expert_forward`]): SwiGLU FFN
//!   `y = (silu(x·Wg^T) ⊙ (x·Wu^T)) · Wd^T` with the fused gate/up layout
//!   `w_gate_up: [2I, hidden]` (gate rows first, then up rows) and
//!   `w_down: [hidden, I]`.
//! * **Shared experts** ([`MoeWeights::shared_forward`]) are computed like
//!   routed experts and added **unweighted** (V3 semantics); only routed
//!   experts carry routing weights: `moe_forward` computes
//!   `y = shared(x) + Σ_k w_k · routed_k(x)`.
//!
//! Determinism: selections use a total order (score desc, expert id asc),
//! weight renormalization accumulates in expert order, the fused matmuls
//! accumulate sequentially over k, and `moe_forward` accumulates per token in
//! ascending k — identical inputs give bit-identical outputs.
//!
//! All matmuls use `deepinfer_gemm::f32_gemm_nt` (the workspace's golden
//! reference NT GEMM).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod expert;
pub mod router;

pub use router::MoeWeights;

use deepinfer_core::{ensure, Error, Result};

/// Configuration of one DeepSeekMoE layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeConfig {
    /// Hidden size (router input and expert output width).
    pub hidden: usize,
    /// Per-expert FFN intermediate size `I` (gate and up branches each have
    /// `I` rows in `w_gate_up`).
    pub intermediate: usize,
    /// Number of routed experts (256 for V3-class models).
    pub num_routed: usize,
    /// Number of activated experts per token (`k`).
    pub num_activated: usize,
    /// Number of shared (always-active) experts (1 for V3-class models).
    pub num_shared: usize,
    /// Expert group size: `1` disables grouping; `g > 1` partitions experts
    /// into `num_routed / g` consecutive groups and routes whole groups
    /// (V3.2-style grouped experts). `num_routed` and `num_activated` must be
    /// divisible by `g`.
    pub group_size: usize,
    /// V3 "noaux-tc": `gate_bias` affects top-k **selection only**; the
    /// routing weights use the bias-free sigmoid scores.
    pub use_noaux_tc_bias: bool,
}

impl MoeConfig {
    /// V4.1-class reference defaults for a given hidden size: 256 routed
    /// experts, 8 activated, 1 shared expert, noaux-tc bias on, grouping off
    /// (V3.2 grouped experts are opt-in via `group_size`).
    ///
    /// `intermediate` is set to `hidden / 4` (at least 1) as a compact
    /// reference default — the real value comes from the checkpoint config.
    pub fn v41_default(hidden: usize) -> Self {
        MoeConfig {
            hidden,
            intermediate: (hidden / 4).max(1),
            num_routed: 256,
            num_activated: 8,
            num_shared: 1,
            group_size: 1,
            use_noaux_tc_bias: true,
        }
    }
}

/// SwiGLU weights of a single expert (shared or routed).
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertWeights {
    /// Fused gate/up projection, row-major `[2 * intermediate, hidden]`:
    /// rows `[0, I)` are the gate branch, rows `[I, 2I)` the up branch.
    pub w_gate_up: Vec<f32>,
    /// Down projection, row-major `[hidden, intermediate]`.
    pub w_down: Vec<f32>,
}

/// Routing result for one token.
#[derive(Debug, Clone, PartialEq)]
pub struct Routing {
    /// Selected expert ids. Normal mode: selection-score descending, ties to
    /// the lower expert id. Group mode: selected groups in descending group
    /// score (ties to the lower group id), experts ascending within each
    /// selected group.
    pub topk_idx: Vec<u32>,
    /// Routing weights parallel to `topk_idx`: the bias-free sigmoid scores of
    /// the selected experts, renormalized to sum to one.
    pub topk_weights: Vec<f32>,
}

/// Full local MoE forward (no expert parallelism):
/// `y = shared(x) + Σ_k w_k · routed_k(x)` — shared experts contribute
/// unweighted, routed experts with their renormalized routing weights
/// (DeepSeek-V3 §2.1.1).
///
/// Accumulation order is fixed: shared experts first (ascending expert id),
/// then the routed contributions of each token in ascending k —
/// bit-deterministic for identical inputs.
pub fn moe_forward(w: &MoeWeights, cfg: &MoeConfig, x: &[f32], tokens: usize) -> Result<Vec<f32>> {
    router::validate_config(cfg)?;
    w.validate_against(cfg)?;
    let h = cfg.hidden;
    let need = tokens.checked_mul(h).ok_or_else(|| {
        Error::CapacityExceeded(format!("moe_forward: tokens ({tokens}) * hidden ({h}) overflows usize"))
    })?;
    ensure!(
        x.len() == need,
        DimensionMismatch,
        "moe_forward: x.len() = {} but tokens*hidden = {need}",
        x.len()
    );
    ensure!(
        x.iter().all(|v| v.is_finite()),
        InvalidArgument,
        "moe_forward: x contains non-finite values"
    );
    let mut y = w.shared_forward(cfg, x, tokens)?;
    let routings = w.route_batch(cfg, x, tokens)?;
    for t in 0..tokens {
        let xt = &x[t * h..(t + 1) * h];
        for (&e, &wgt) in routings[t].topk_idx.iter().zip(routings[t].topk_weights.iter()) {
            let ew = w.routed.get(e as usize).ok_or_else(|| {
                Error::OutOfRange(format!(
                    "moe_forward: routing selected expert {e} but only {} routed experts exist",
                    w.routed.len()
                ))
            })?;
            let row = expert::expert_rows(ew, xt, 1, h, cfg.intermediate)?;
            for d in 0..h {
                y[t * h + d] += wgt * row[d];
            }
        }
    }
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepinfer_core::prop::{assert_close_slice, prop, random_vec};
    use deepinfer_core::rng::Philox;

    fn small_cfg(rng: &mut Philox) -> MoeConfig {
        let hidden = rng.uniform_usize(1, 9);
        let num_routed = rng.uniform_usize(2, 7);
        let k = rng.uniform_usize(1, num_routed + 1);
        let group = if num_routed % 2 == 0 && k % 2 == 0 && rng.next_f32() < 0.5 {
            2
        } else {
            1
        };
        MoeConfig {
            hidden,
            intermediate: rng.uniform_usize(1, 5),
            num_routed,
            num_activated: k,
            num_shared: rng.uniform_usize(0, 3),
            group_size: group,
            use_noaux_tc_bias: rng.next_f32() < 0.5,
        }
    }

    #[test]
    fn v41_defaults() {
        let cfg = MoeConfig::v41_default(7168);
        assert_eq!(cfg.num_routed, 256);
        assert_eq!(cfg.num_activated, 8);
        assert_eq!(cfg.num_shared, 1);
        assert_eq!(cfg.group_size, 1);
        assert!(cfg.use_noaux_tc_bias);
        assert_eq!(cfg.intermediate, 1792);
        assert!(router::validate_config(&cfg).is_ok());
    }

    #[test]
    fn moe_forward_matches_manual_loop() {
        prop("moe_forward == manual", 0xB0E1, 100, |i, mut rng| {
            let cfg = small_cfg(&mut rng);
            let tokens = rng.uniform_usize(1, 5);
            let mut w = MoeWeights::from_config(&cfg, &mut rng);
            // Deterministic bias strong enough to actually re-rank selections.
            for e in 0..cfg.num_routed {
                w.gate_bias[e] = rng.uniform_f32(-0.5, 0.5);
            }
            let x = random_vec(&mut rng, tokens * cfg.hidden, 1.0);
            let y = moe_forward(&w, &cfg, &x, tokens)?;
            let h = cfg.hidden;
            let mut y_ref = w.shared_forward(&cfg, &x, tokens)?;
            let routings = w.route_batch(&cfg, &x, tokens)?;
            assert_eq!(routings.len(), tokens);
            for t in 0..tokens {
                assert_eq!(routings[t].topk_idx.len(), cfg.num_activated);
                for (&e, &wgt) in routings[t]
                    .topk_idx
                    .iter()
                    .zip(routings[t].topk_weights.iter())
                {
                    let row = w.expert_forward(e as usize, &x[t * h..(t + 1) * h], 1)?;
                    for d in 0..h {
                        y_ref[t * h + d] += wgt * row[d];
                    }
                }
            }
            assert_close_slice(&y, &y_ref, 1e-5, "moe_forward vs manual loop");
            let _ = i;
            Ok(())
        });
    }

    #[test]
    fn moe_forward_is_deterministic() {
        prop("moe_forward determinism", 0xB0E2, 50, |i, mut rng| {
            let cfg = small_cfg(&mut rng);
            let tokens = rng.uniform_usize(1, 4);
            let w = MoeWeights::from_config(&cfg, &mut rng);
            let x = random_vec(&mut rng, tokens * cfg.hidden, 1.0);
            let a = moe_forward(&w, &cfg, &x, tokens)?;
            let b = moe_forward(&w, &cfg, &x, tokens)?;
            assert_eq!(a, b, "moe_forward must be bit-deterministic");
            let _ = i;
            Ok(())
        });
    }

    #[test]
    fn shared_expert_is_added_unweighted() {
        // A single shared expert with identity-like magnitudes must appear in
        // the output exactly once, independent of any routing weights.
        let mut rng = Philox::new(0x5EED, 1);
        let cfg = MoeConfig {
            hidden: 4,
            intermediate: 3,
            num_routed: 2,
            num_activated: 1,
            num_shared: 1,
            group_size: 1,
            use_noaux_tc_bias: false,
        };
        let w = MoeWeights::from_config(&cfg, &mut rng);
        let x = random_vec(&mut rng, cfg.hidden, 1.0);
        let tokens = 1;
        let shared = w.shared_forward(&cfg, &x, tokens).unwrap();
        // num_activated == 1: y = shared + w_0 * routed_0(x).
        let y = moe_forward(&w, &cfg, &x, tokens).unwrap();
        let r = w.route(&cfg, &x).unwrap();
        assert_eq!(r.topk_idx.len(), 1);
        let routed = w.expert_forward(r.topk_idx[0] as usize, &x, tokens).unwrap();
        for d in 0..cfg.hidden {
            let expect = shared[d] + r.topk_weights[0] * routed[d];
            assert!((y[d] - expect).abs() < 1e-5, "y[{d}] = {} vs {expect}", y[d]);
        }
    }

    #[test]
    fn moe_forward_validates_inputs() {
        let mut rng = Philox::new(0xE22, 7);
        let cfg = MoeConfig {
            hidden: 4,
            intermediate: 2,
            num_routed: 3,
            num_activated: 2,
            num_shared: 1,
            group_size: 1,
            use_noaux_tc_bias: false,
        };
        let w = MoeWeights::from_config(&cfg, &mut rng);
        assert!(moe_forward(&w, &cfg, &[0.0; 3], 1).is_err()); // wrong x len
        assert!(moe_forward(&w, &cfg, &[f32::NAN; 4], 1).is_err()); // NaN x
        let mut bad_cfg = cfg.clone();
        bad_cfg.num_activated = 0;
        assert!(moe_forward(&w, &bad_cfg, &[0.0; 4], 1).is_err());
        let mut bad_group = cfg.clone();
        bad_group.group_size = 2; // num_activated (2) ok, but 3 % 2 != 0
        assert!(moe_forward(&w, &bad_group, &[0.0; 4], 1).is_err());
    }
}
