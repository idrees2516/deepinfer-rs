//! SwiGLU expert compute for DeepSeekMoE: the per-expert FFN
//! `y = (silu(x·Wg^T) ⊙ (x·Wu^T)) · Wd^T` and the unweighted shared-expert
//! sum.
//!
//! Upstream: DeepSeek-V3 technical report §2.1.1 (expert FFN with SiLU-gated
//! activation; the fused `w_gate_up = [2I, hidden]` layout stores the gate
//! rows first, then the up rows). All matmuls go through
//! `deepinfer_gemm::f32_gemm_nt` (NT layout, sequential f32 accumulation in
//! k — the golden reference GEMM of the workspace).

use crate::{ExpertWeights, MoeConfig, MoeWeights};
use deepinfer_core::{ensure, Error, Result};
use deepinfer_gemm::f32_gemm_nt;

impl MoeWeights {
    /// Runs one routed expert on a batch of token rows:
    /// `y[t] = (silu(x[t]·Wg^T) ⊙ (x[t]·Wu^T)) · Wd^T`.
    ///
    /// `x` is `[tokens, hidden]` row-major; the output is `[tokens, hidden]`.
    /// The hidden size is inferred from the router gate matrix
    /// (`gate.len() / routed.len()`) and the intermediate size from
    /// `w_down.len() / hidden`; mismatches yield `Err`, never a panic.
    pub fn expert_forward(&self, expert: usize, x: &[f32], tokens: usize) -> Result<Vec<f32>> {
        ensure!(
            expert < self.routed.len(),
            OutOfRange,
            "expert_forward: expert index {expert} >= num_routed {}",
            self.routed.len()
        );
        let hidden = self.infer_hidden()?;
        let ew = &self.routed[expert];
        let intermediate = infer_intermediate(ew, hidden)?;
        expert_rows(ew, x, tokens, hidden, intermediate)
    }

    /// Shared-expert output: the **unweighted** sum over all shared experts
    /// (V3 semantics — shared experts do not carry routing weights),
    /// `[tokens, hidden]`.
    pub fn shared_forward(&self, cfg: &MoeConfig, x: &[f32], tokens: usize) -> Result<Vec<f32>> {
        crate::router::validate_config(cfg)?;
        ensure!(
            self.shared.len() == cfg.num_shared,
            DimensionMismatch,
            "shared_forward: shared.len() = {} but num_shared = {}",
            self.shared.len(),
            cfg.num_shared
        );
        let h = cfg.hidden;
        let need = tokens.checked_mul(h).ok_or_else(|| {
            Error::CapacityExceeded(format!(
                "shared_forward: tokens ({tokens}) * hidden ({h}) overflows usize"
            ))
        })?;
        ensure!(
            x.len() == need,
            DimensionMismatch,
            "shared_forward: x.len() = {} but tokens*hidden = {need}",
            x.len()
        );
        ensure!(
            x.iter().all(|v| v.is_finite()),
            InvalidArgument,
            "shared_forward: x contains non-finite values"
        );
        let mut y = vec![0.0f32; need];
        for (si, ew) in self.shared.iter().enumerate() {
            let i = infer_intermediate(ew, h).map_err(|e| {
                Error::DimensionMismatch(format!("shared expert {si}: {e}"))
            })?;
            ensure!(
                ew.w_gate_up.len() == 2 * i * h,
                DimensionMismatch,
                "shared_forward: shared[{si}].w_gate_up.len() = {} but 2*I*H = {}",
                ew.w_gate_up.len(),
                2 * i * h
            );
            let out = expert_rows(ew, x, tokens, h, i)?;
            for (yv, &ov) in y.iter_mut().zip(out.iter()) {
                *yv += ov;
            }
        }
        Ok(y)
    }

    /// Hidden size inferred from the router gate (`[num_routed, hidden]`).
    fn infer_hidden(&self) -> Result<usize> {
        ensure!(
            !self.routed.is_empty(),
            InvalidArgument,
            "expert_forward: cannot infer hidden size from an empty routed set"
        );
        ensure!(
            self.gate.len() >= self.routed.len(),
            DimensionMismatch,
            "expert_forward: gate.len() = {} < routed.len() = {} (hidden would be 0)",
            self.gate.len(),
            self.routed.len()
        );
        ensure!(
            self.gate.len() % self.routed.len() == 0,
            DimensionMismatch,
            "expert_forward: gate.len() = {} is not a multiple of routed.len() = {}",
            self.gate.len(),
            self.routed.len()
        );
        Ok(self.gate.len() / self.routed.len())
    }
}

/// Intermediate size inferred from an expert's down projection
/// (`w_down: [hidden, I]`).
fn infer_intermediate(ew: &ExpertWeights, hidden: usize) -> Result<usize> {
    ensure!(
        hidden >= 1,
        InvalidArgument,
        "infer_intermediate: hidden must be >= 1"
    );
    ensure!(
        ew.w_down.len() % hidden == 0,
        DimensionMismatch,
        "w_down.len() = {} is not a multiple of hidden = {}",
        ew.w_down.len(),
        hidden
    );
    let i = ew.w_down.len() / hidden;
    ensure!(
        ew.w_gate_up.len() == 2 * i * hidden,
        DimensionMismatch,
        "w_gate_up.len() = {} but 2*I*H = {} (I = {i})",
        ew.w_gate_up.len(),
        2 * i * hidden
    );
    Ok(i)
}

/// SwiGLU forward over `[tokens, hidden]` rows with known dims.
pub(crate) fn expert_rows(
    ew: &ExpertWeights,
    x: &[f32],
    tokens: usize,
    hidden: usize,
    intermediate: usize,
) -> Result<Vec<f32>> {
    let need = tokens.checked_mul(hidden).ok_or_else(|| {
        Error::CapacityExceeded(format!(
            "expert_rows: tokens ({tokens}) * hidden ({hidden}) overflows usize"
        ))
    })?;
    ensure!(
        x.len() == need,
        DimensionMismatch,
        "expert_rows: x.len() = {} but tokens*hidden = {need}",
        x.len()
    );
    ensure!(
        x.iter().all(|v| v.is_finite()),
        InvalidArgument,
        "expert_rows: x contains non-finite values"
    );
    // gate_up: [tokens, 2I] = x @ w_gate_up^T
    let gate_up = f32_gemm_nt(x, &ew.w_gate_up, tokens, 2 * intermediate, hidden)?;
    // act: [tokens, I] = silu(gate) * up
    let mut act = vec![0.0f32; tokens * intermediate];
    for t in 0..tokens {
        for c in 0..intermediate {
            let g = gate_up[t * 2 * intermediate + c];
            let u = gate_up[t * 2 * intermediate + intermediate + c];
            act[t * intermediate + c] = silu(g) * u;
        }
    }
    // y: [tokens, hidden] = act @ w_down^T
    f32_gemm_nt(&act, &ew.w_down, tokens, hidden, intermediate)
}

/// SiLU (swish) activation: `x · sigmoid(x)`.
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MoeConfig, MoeWeights};
    use deepinfer_core::prop::{assert_close_slice, prop, random_vec};
    use deepinfer_core::rng::Philox;

    /// Independent SwiGLU reference (row-wise, explicit loops).
    fn manual_swiglu(ew: &ExpertWeights, x: &[f32], tokens: usize, h: usize, i: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; tokens * h];
        for t in 0..tokens {
            for c in 0..i {
                let mut g = 0.0f32;
                let mut u = 0.0f32;
                for d in 0..h {
                    let xv = x[t * h + d];
                    g += ew.w_gate_up[c * h + d] * xv;
                    u += ew.w_gate_up[(i + c) * h + d] * xv;
                }
                let act = (g / (1.0 + (-g).exp())) * u;
                for d in 0..h {
                    y[t * h + d] += ew.w_down[d * i + c] * act;
                }
            }
        }
        y
    }

    fn weights_with_experts(hidden: usize, intermediate: usize, routed: usize) -> MoeWeights {
        let mut rng = Philox::new(0xE9, 3);
        let cfg = MoeConfig {
            hidden,
            intermediate,
            num_routed: routed,
            num_activated: 1,
            num_shared: 0,
            group_size: 1,
            use_noaux_tc_bias: false,
        };
        MoeWeights::from_config(&cfg, &mut rng)
    }

    #[test]
    fn expert_forward_matches_manual_swiglu() {
        prop("expert SwiGLU", 0xE11, 100, |i, mut rng| {
            let h = rng.uniform_usize(1, 7);
            let it = rng.uniform_usize(1, 6);
            let tokens = rng.uniform_usize(1, 5);
            let w = weights_with_experts(h, it, 3);
            let x = random_vec(&mut rng, tokens * h, 1.0);
            let e = rng.uniform_usize(0, 3);
            let y = w.expert_forward(e, &x, tokens)?;
            let expect = manual_swiglu(&w.routed[e], &x, tokens, h, it);
            assert_close_slice(&y, &expect, 1e-4, "expert_forward vs manual SwiGLU");
            let _ = i;
            Ok(())
        });
    }

    #[test]
    fn expert_forward_batch_equals_rows() {
        prop("expert batch == rows", 0xE12, 50, |i, mut rng| {
            let h = rng.uniform_usize(1, 6);
            let it = rng.uniform_usize(1, 5);
            let tokens = rng.uniform_usize(1, 5);
            let w = weights_with_experts(h, it, 2);
            let x = random_vec(&mut rng, tokens * h, 1.5);
            let batch = w.expert_forward(1, &x, tokens)?;
            for t in 0..tokens {
                let row = w.expert_forward(1, &x[t * h..(t + 1) * h], 1)?;
                assert_close_slice(
                    &batch[t * h..(t + 1) * h],
                    &row,
                    1e-6,
                    "batched row must match single-row compute",
                );
            }
            let _ = i;
            Ok(())
        });
    }

    #[test]
    fn shared_forward_sums_all_shared_experts() {
        let mut rng = Philox::new(0x5A, 5);
        let cfg = MoeConfig {
            hidden: 3,
            intermediate: 2,
            num_routed: 1,
            num_activated: 1,
            num_shared: 2,
            group_size: 1,
            use_noaux_tc_bias: false,
        };
        let w = MoeWeights::from_config(&cfg, &mut rng);
        let x = random_vec(&mut rng, cfg.hidden, 1.0);
        let y = w.shared_forward(&cfg, &x, 1).unwrap();
        let a = manual_swiglu(&w.shared[0], &x, 1, 3, 2);
        let b = manual_swiglu(&w.shared[1], &x, 1, 3, 2);
        for d in 0..3 {
            assert!((y[d] - (a[d] + b[d])).abs() < 1e-5);
        }
        // Zero shared experts → zero output, no error.
        let cfg0 = MoeConfig {
            num_shared: 0,
            ..cfg.clone()
        };
        let w0 = MoeWeights {
            gate: w.gate.clone(),
            gate_bias: w.gate_bias.clone(),
            shared: Vec::new(),
            routed: w.routed.clone(),
        };
        assert_eq!(w0.shared_forward(&cfg0, &x, 1).unwrap(), vec![0.0f32; 3]);
    }

    #[test]
    fn expert_forward_validates() {
        let w = weights_with_experts(4, 2, 2);
        assert!(w.expert_forward(2, &[0.0; 4], 1).is_err()); // expert OOB
        assert!(w.expert_forward(0, &[0.0; 3], 1).is_err()); // x len
        assert!(w.expert_forward(0, &[f32::NAN; 4], 1).is_err()); // NaN
        assert!(w.expert_forward(0, &[0.0; 8], 1).is_err()); // tokens mismatch
        // Malformed weights: w_down inconsistent with w_gate_up.
        let mut bad = w.clone();
        bad.routed[0].w_down.push(0.0);
        assert!(bad.expert_forward(0, &[0.0; 4], 1).is_err());
    }

    #[test]
    fn f32_gemm_nt_known_values() {
        // a = [1, 2; 3, 4], b = [5, 6; 7, 8] → out = a·b^T = [17, 23; 39, 53]
        let out = f32_gemm_nt(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], 2, 2, 2).unwrap();
        assert_eq!(out, vec![17.0, 23.0, 39.0, 53.0]);
        assert!(f32_gemm_nt(&[0.0; 3], &[0.0; 4], 2, 2, 2).is_err());
    }
}
