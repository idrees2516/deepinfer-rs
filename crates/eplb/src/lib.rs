//! # deepinfer-eplb
//!
//! Expert-Parallelism Load Balancer for **DeepInfer-rs** — the pure-Rust
//! reference port of DeepSeek's [EPLB](https://github.com/deepseek-ai/EPLB)
//! redundant-experts + rebalancing logic (upstream `eplb/rebalance.py`).
//!
//! * **Redundant experts** ([`get_redundant_experts`]): greedy replication —
//!   `num_replicas` times, clone the logical expert with the highest
//!   load-per-current-copy (`weight[e] / copies[e]`, ties to the lower expert
//!   id). This is the upstream `get_redundant_experts` greedy.
//! * **Rebalancing** ([`rebalance_experts`]): instantiate every logical
//!   expert `copies[e]` times (each instance carrying `weight[e]/copies[e]`
//!   of the load) and place the physical instances onto GPUs:
//!   * *hierarchical policy* when the upstream divisibility asserts hold
//!     (`num_groups % num_nodes == 0`, `num_gpus % num_nodes == 0`,
//!     `num_gpus >= num_nodes`, `num_groups / num_nodes >= 1`) — expert
//!     groups (contiguous blocks of logical experts) are LPT-packed onto
//!     nodes (exactly `num_groups / num_nodes` groups per node), then each
//!     node's instances are LPT-packed onto its `num_gpus / num_nodes` GPUs;
//!   * *global policy* otherwise — all instances LPT-packed onto all GPUs.
//! * **Statistics** ([`ExpertStats`]): sliding-window moving average of
//!   per-expert batch counts — the load signal `rebalance_experts` consumes.
//!
//! Determinism contract: every selection/packing uses a total order
//! (load descending, then lower id) and load ties resolve toward the lower
//! bin id; no hash-ordered iteration participates in any output. Identical
//! inputs produce bit-identical [`EplbMaps`].
//!
//! Deviation from upstream (documented): upstream EPLB returns a per-GPU
//! `physical_to_logical_map` tensor; this port returns the flat
//! `phy2log` / `log2phy` / `logcnt` maps mandated by SPEC.md. The `phy2log`
//! layout is **GPU-major**: physical expert ids are assigned by walking GPUs
//! in ascending id and appending each GPU's instances in assignment order,
//! so experts co-located on a GPU are contiguous in `phy2log`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod rebalance;
pub mod stats;

pub use rebalance::{get_redundant_experts, rebalance_experts, EplbMaps};
pub use stats::ExpertStats;

use deepinfer_core::{ensure, Error, Result};

/// Configuration of one EPLB rebalancing problem (upstream signature:
/// `rebalance_experts(weight, num_replicas, num_groups, num_nodes,
/// n_gpus)`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EplbConfig {
    /// Number of distinct logical experts.
    pub num_logical: usize,
    /// Extra physical copies to add on top of the logical experts
    /// (`num_physical = num_logical + num_replicas`).
    pub num_replicas: usize,
    /// Number of expert groups for the hierarchical policy. Group `g`
    /// contains the logical experts `[g * num_logical / num_groups,
    /// (g+1) * num_logical / num_groups)`.
    pub num_groups: usize,
    /// Number of nodes in the GPU topology.
    pub num_nodes: usize,
    /// Total number of GPUs across all nodes.
    pub num_gpus: usize,
}

impl EplbConfig {
    /// Total number of physical experts: `num_logical + num_replicas`
    /// (saturating on overflow; overflow is rejected by
    /// [`EplbConfig::validate`]).
    pub fn num_physical(&self) -> usize {
        self.num_logical.saturating_add(self.num_replicas)
    }

    /// Validates positivity and the divisibility rules both placement
    /// policies rely on:
    ///
    /// * `num_logical`, `num_groups`, `num_nodes`, `num_gpus` all `>= 1`;
    /// * `num_logical + num_replicas` does not overflow `usize`
    ///   (denial-of-service guard);
    /// * `num_logical % num_groups == 0` — groups evenly partition the
    ///   logical experts (upstream EPLB asserts
    ///   `n_experts_per_group * num_groups == n_logical_experts`);
    /// * `num_physical >= num_gpus` — every GPU hosts at least one physical
    ///   expert.
    ///
    /// Configurations that fail only the *hierarchical* divisibility rules
    /// (`num_groups % num_nodes`, `num_gpus % num_nodes`) are **valid**:
    /// [`rebalance_experts`] falls back to the global LPT policy for them.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] for the violations above and
    /// [`Error::CapacityExceeded`] when `num_logical + num_replicas`
    /// overflows.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.num_logical >= 1,
            InvalidArgument,
            "EplbConfig: num_logical must be >= 1, got {}",
            self.num_logical
        );
        ensure!(
            self.num_groups >= 1,
            InvalidArgument,
            "EplbConfig: num_groups must be >= 1, got {}",
            self.num_groups
        );
        ensure!(
            self.num_nodes >= 1,
            InvalidArgument,
            "EplbConfig: num_nodes must be >= 1, got {}",
            self.num_nodes
        );
        ensure!(
            self.num_gpus >= 1,
            InvalidArgument,
            "EplbConfig: num_gpus must be >= 1, got {}",
            self.num_gpus
        );
        let physical = self.num_logical.checked_add(self.num_replicas).ok_or_else(|| {
            Error::CapacityExceeded(format!(
                "EplbConfig: num_logical ({}) + num_replicas ({}) overflows usize",
                self.num_logical, self.num_replicas
            ))
        })?;
        ensure!(
            self.num_logical % self.num_groups == 0,
            InvalidArgument,
            "EplbConfig: num_logical ({}) must be divisible by num_groups ({}) \
             (EPLB: groups evenly partition the logical experts)",
            self.num_logical,
            self.num_groups
        );
        ensure!(
            physical >= self.num_gpus,
            InvalidArgument,
            "EplbConfig: num_physical ({physical}) must be >= num_gpus ({}) \
             so every GPU hosts at least one physical expert",
            self.num_gpus
        );
        Ok(())
    }

    /// Whether the hierarchical placement policy applies to this config.
    ///
    /// Mirrors the branch condition of upstream EPLB's `rebalance_experts`:
    /// hierarchical iff `num_groups >= 1`, `num_nodes >= 1`,
    /// `num_gpus >= num_nodes`, `num_groups % num_nodes == 0`,
    /// `num_gpus % num_nodes == 0` and `num_groups / num_nodes >= 1`.
    /// Note the degenerate case `num_groups == 1 && num_nodes == 1` is
    /// hierarchical but reduces to exactly the global packing (one node
    /// holding every instance).
    pub fn is_hierarchical(&self) -> bool {
        self.num_groups >= 1
            && self.num_nodes >= 1
            && self.num_gpus >= self.num_nodes
            && self.num_groups % self.num_nodes == 0
            && self.num_gpus % self.num_nodes == 0
            && self.num_groups / self.num_nodes >= 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(logical: usize, replicas: usize, groups: usize, nodes: usize, gpus: usize) -> EplbConfig {
        EplbConfig {
            num_logical: logical,
            num_replicas: replicas,
            num_groups: groups,
            num_nodes: nodes,
            num_gpus: gpus,
        }
    }

    #[test]
    fn num_physical_is_logical_plus_replicas() {
        assert_eq!(cfg(8, 4, 2, 1, 4).num_physical(), 12);
        assert_eq!(cfg(8, 0, 2, 1, 4).num_physical(), 8);
    }

    #[test]
    fn validate_rejects_broken_configs() {
        assert!(cfg(0, 0, 1, 1, 1).validate().is_err()); // num_logical = 0
        assert!(cfg(4, 0, 0, 1, 1).validate().is_err()); // num_groups = 0
        assert!(cfg(4, 0, 1, 0, 1).validate().is_err()); // num_nodes = 0
        assert!(cfg(4, 0, 1, 1, 0).validate().is_err()); // num_gpus = 0
        assert!(cfg(5, 0, 2, 1, 2).validate().is_err()); // 5 % 2 != 0
        assert!(cfg(2, 0, 1, 1, 4).validate().is_err()); // 2 physical < 4 gpus
        assert!(cfg(usize::MAX, usize::MAX, 1, 1, 1)
            .validate()
            .is_err()); // logical + replicas overflows
        assert!(cfg(4, 1, 2, 2, 4).validate().is_ok());
        assert!(cfg(4, 1, 2, 3, 4).validate().is_ok()); // global fallback is valid
    }

    #[test]
    fn hierarchical_branch_condition() {
        // Divisible topologies -> hierarchical.
        assert!(cfg(8, 0, 4, 2, 4).is_hierarchical());
        assert!(cfg(8, 0, 2, 1, 4).is_hierarchical());
        assert!(cfg(8, 0, 4, 4, 8).is_hierarchical());
        // Degenerate single group / single node == hierarchical (and then
        // reduces to the global packing).
        assert!(cfg(8, 0, 1, 1, 4).is_hierarchical());
        // Non-divisible or inverted topologies -> global fallback.
        assert!(!cfg(8, 0, 4, 3, 4).is_hierarchical()); // 4 % 3 != 0
        assert!(!cfg(8, 0, 1, 2, 4).is_hierarchical()); // 1 % 2 != 0
        assert!(!cfg(8, 0, 4, 4, 6).is_hierarchical()); // 6 % 4 != 0
        assert!(!cfg(8, 0, 2, 4, 4).is_hierarchical()); // gpus < nodes
        assert!(!cfg(8, 0, 6, 4, 4).is_hierarchical()); // gpus < nodes
    }
}
