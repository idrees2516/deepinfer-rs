//! Expert-parallelism configuration (DeepEP `Buffer`-level parameters).

use deepinfer_core::{ensure, Result};

/// Configuration of one expert-parallelism domain, mirroring the parameters
/// DeepEP's `Buffer` is configured with (`num_ranks`, `num_experts`,
/// `hidden`, `num_max_tokens_per_rank`) plus the DeepGEMM contiguous-layout
/// `expert_alignment`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpConfig {
    /// Number of ranks in the simulated world.
    pub world_size: usize,
    /// Total (logical) number of experts across all ranks.
    pub num_experts: usize,
    /// Per-sender token-slot capacity of the dispatch buffers: each rank may
    /// dispatch at most this many `(token, k)` slots to any single receiver
    /// (DeepEP sizes its normal-mode channel buffers per sender-receiver
    /// pair). Enforced by [`dispatch`](crate::dispatch) and
    /// [`recv`](crate::recv) with `CapacityExceeded`.
    pub num_max_tokens_per_rank: usize,
    /// Hidden dimension of the dispatched token rows.
    pub hidden: usize,
    /// Row-count alignment for the grouped-contiguous GEMM layout
    /// (DeepGEMM: 128). Each expert's received row segment is padded to a
    /// multiple of this value with zero rows.
    pub expert_alignment: usize,
}

impl EpConfig {
    /// Number of local experts per rank: `num_experts / world_size`
    /// (must divide evenly — see [`EpConfig::validate`]).
    ///
    /// Total and panic-free: returns `0` for degenerate configurations
    /// (`world_size == 0`) instead of dividing by zero; call
    /// [`EpConfig::validate`] to reject them up front.
    pub fn num_local_experts(&self) -> usize {
        if self.world_size == 0 {
            0
        } else {
            self.num_experts / self.world_size
        }
    }

    /// Owner rank of `expert`: `expert * world_size / num_experts`
    /// (block-contiguous ownership — DeepEP's expert placement contract:
    /// rank `r` owns experts `[r * num_local, (r+1) * num_local)`).
    ///
    /// Precondition: `expert < num_experts` (enforced by every public entry
    /// point of this crate). The implementation is total and panic-free
    /// regardless: out-of-range experts clamp to the last rank and
    /// degenerate configurations return `0`.
    pub fn owner_rank(&self, expert: u32) -> usize {
        if self.num_experts == 0 || self.world_size == 0 {
            return 0;
        }
        let v = (expert as u128 * self.world_size as u128) / self.num_experts as u128;
        let v = v.min(self.world_size as u128 - 1);
        v as usize
    }

    /// Validates the structural invariants:
    ///
    /// * `world_size >= 1`, `num_experts >= 1`, `num_max_tokens_per_rank >= 1`,
    ///   `hidden >= 1`, `expert_alignment >= 1`;
    /// * `num_experts % world_size == 0` (experts divide evenly across
    ///   ranks — DeepEP's uniform expert placement);
    /// * `num_experts <= u32::MAX` (expert ids are `u32` in the routing
    ///   tables).
    ///
    /// Note `hidden % FP8_QUANT_GROUP == 0` is **not** required here: it is
    /// only needed for FP8 payloads and is enforced (with a clear error) by
    /// [`dispatch`](crate::dispatch) when `fp8 = true`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`](deepinfer_core::Error::InvalidArgument) /
    /// [`Error::OutOfRange`](deepinfer_core::Error::OutOfRange) on the
    /// violations above.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.world_size >= 1,
            InvalidArgument,
            "EpConfig: world_size must be >= 1, got {}",
            self.world_size
        );
        ensure!(
            self.num_experts >= 1,
            InvalidArgument,
            "EpConfig: num_experts must be >= 1, got {}",
            self.num_experts
        );
        ensure!(
            self.num_max_tokens_per_rank >= 1,
            InvalidArgument,
            "EpConfig: num_max_tokens_per_rank must be >= 1, got {}",
            self.num_max_tokens_per_rank
        );
        ensure!(
            self.hidden >= 1,
            InvalidArgument,
            "EpConfig: hidden must be >= 1, got {}",
            self.hidden
        );
        ensure!(
            self.expert_alignment >= 1,
            InvalidArgument,
            "EpConfig: expert_alignment must be >= 1, got {}",
            self.expert_alignment
        );
        ensure!(
            self.num_experts % self.world_size == 0,
            InvalidArgument,
            "EpConfig: num_experts ({}) must be divisible by world_size ({})",
            self.num_experts,
            self.world_size
        );
        ensure!(
            self.num_experts <= u32::MAX as usize,
            OutOfRange,
            "EpConfig: num_experts ({}) must be <= u32::MAX (expert ids are u32)",
            self.num_experts
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_experts_and_owner_rank_block_contiguity() {
        let cfg = EpConfig {
            world_size: 4,
            num_experts: 12,
            num_max_tokens_per_rank: 8,
            hidden: 16,
            expert_alignment: 128,
        };
        assert_eq!(cfg.num_local_experts(), 3);
        // Block-contiguous ownership: rank r owns [r*3, (r+1)*3).
        for e in 0..12u32 {
            assert_eq!(cfg.owner_rank(e), (e * 4 / 12) as usize);
        }
        assert_eq!(cfg.owner_rank(0), 0);
        assert_eq!(cfg.owner_rank(2), 0);
        assert_eq!(cfg.owner_rank(3), 1);
        assert_eq!(cfg.owner_rank(11), 3);
        // Monotone across the whole range.
        let mut owners: Vec<usize> = (0..12u32).map(|e| cfg.owner_rank(e)).collect();
        assert!(owners.windows(2).all(|w| w[0] <= w[1]));
        owners.dedup();
        assert_eq!(owners, vec![0, 1, 2, 3]);
    }

    #[test]
    fn owner_rank_is_total_for_degenerate_configs() {
        let zero_world = EpConfig {
            world_size: 0,
            num_experts: 4,
            num_max_tokens_per_rank: 1,
            hidden: 1,
            expert_alignment: 1,
        };
        assert_eq!(zero_world.num_local_experts(), 0);
        assert_eq!(zero_world.owner_rank(3), 0);
        let zero_experts = EpConfig {
            world_size: 2,
            num_experts: 0,
            num_max_tokens_per_rank: 1,
            hidden: 1,
            expert_alignment: 1,
        };
        assert_eq!(zero_experts.owner_rank(0), 0);
        // Out-of-range experts clamp to the last rank (documented).
        let cfg = EpConfig {
            world_size: 2,
            num_experts: 4,
            num_max_tokens_per_rank: 1,
            hidden: 1,
            expert_alignment: 1,
        };
        assert_eq!(cfg.owner_rank(999), 1);
    }

    #[test]
    fn validate_rejects_bad_configs() {
        let base = EpConfig {
            world_size: 2,
            num_experts: 8,
            num_max_tokens_per_rank: 4,
            hidden: 64,
            expert_alignment: 128,
        };
        assert!(base.validate().is_ok());
        let mut c = base.clone();
        c.world_size = 0;
        assert!(c.validate().is_err());
        let mut c = base.clone();
        c.num_experts = 0;
        assert!(c.validate().is_err());
        let mut c = base.clone();
        c.num_experts = 7; // 7 % 2 != 0
        assert!(c.validate().is_err());
        let mut c = base.clone();
        c.num_max_tokens_per_rank = 0;
        assert!(c.validate().is_err());
        let mut c = base.clone();
        c.hidden = 0;
        assert!(c.validate().is_err());
        let mut c = base.clone();
        c.expert_alignment = 0;
        assert!(c.validate().is_err());
        let mut c = base.clone();
        c.num_experts = u32::MAX as usize + 1;
        assert!(c.validate().is_err());
        // hidden not divisible by the FP8 group is still a *valid* config
        // (only FP8 dispatch rejects it).
        let mut c = base.clone();
        c.hidden = 100;
        assert!(c.validate().is_ok());
    }
}
