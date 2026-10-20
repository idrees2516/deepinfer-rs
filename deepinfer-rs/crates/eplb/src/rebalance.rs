//! EPLB redundant-expert selection and expert placement
//! (upstream: `deepseek-ai/EPLB`, `eplb/rebalance.py`).
//!
//! Two placement policies, selected exactly like upstream `rebalance_experts`:
//!
//! * **Hierarchical** (when [`EplbConfig::is_hierarchical`] holds): expert
//!   groups — group `g` is the contiguous logical range
//!   `[g * L/G, (g+1) * L/G)` — are sorted by total group load (descending,
//!   ties to the lower group id) and LPT-scheduled onto nodes, where each
//!   node takes exactly `G/N` groups (capacity-capped LPT; the least-loaded
//!   node with remaining group capacity wins, ties to the lower node id).
//!   Each node's physical instances are then sorted by
//!   (load desc, logical asc, instance ordinal asc) and LPT-scheduled onto
//!   the node's `num_gpus / num_nodes` GPUs.
//! * **Global** (fallback): all physical instances sorted by
//!   (load desc, logical asc, instance ordinal asc) and LPT-scheduled onto
//!   all `num_gpus` GPUs (least-loaded GPU wins, ties to the lower gpu id).
//!
//! The produced `phy2log` is GPU-major (GPU 0's instances in assignment
//! order, then GPU 1's, ...), so co-located experts stay contiguous.

use crate::EplbConfig;
use deepinfer_core::{ensure, Result};

/// The logical ↔ physical expert maps produced by [`rebalance_experts`]
/// (upstream EPLB returns the equivalent `physical_to_logical_map` /
/// `logical_count_by_physical_expert` pair).
#[derive(Debug, Clone, PartialEq)]
pub struct EplbMaps {
    /// Logical expert id of every physical expert, GPU-major
    /// (each GPU's instances contiguous, in assignment order). Length
    /// `num_physical`.
    pub phy2log: Vec<u32>,
    /// Ascending list of physical expert ids hosting each logical expert.
    /// Length `num_logical`.
    pub log2phy: Vec<Vec<u32>>,
    /// Number of physical copies of each logical expert (= the output of
    /// [`get_redundant_experts`]). Length `num_logical`.
    pub logcnt: Vec<u32>,
}

impl EplbMaps {
    /// Structural consistency check: the maps agree in both directions and
    /// the copy counts match. Concretely, for a valid `cfg`:
    ///
    /// * `phy2log.len() == cfg.num_physical()`, `log2phy.len() ==
    ///   logcnt.len() == cfg.num_logical`;
    /// * every `phy2log` entry is a valid logical id;
    /// * `log2phy[e]` equals exactly the ascending list of physical ids `p`
    ///   with `phy2log[p] == e` (round-trip);
    /// * `logcnt[e] == log2phy[e].len()` (counts match).
    ///
    /// This is deliberately a *structural* check — it does not assert
    /// rebalancing-quality properties (every logical expert placed at least
    /// once, load balance); those are properties of [`rebalance_experts`]
    /// covered by its tests. Returns `false` for invalid configs.
    pub fn is_consistent(&self, cfg: &EplbConfig) -> bool {
        if cfg.validate().is_err() {
            return false;
        }
        let n_log = cfg.num_logical;
        let n_phy = cfg.num_physical();
        if self.phy2log.len() != n_phy || self.log2phy.len() != n_log || self.logcnt.len() != n_log
        {
            return false;
        }
        // Expected per-logical physical id lists, ascending by construction.
        let mut expected: Vec<Vec<u32>> = vec![Vec::new(); n_log];
        for (p, &e) in self.phy2log.iter().enumerate() {
            if e as usize >= n_log {
                return false;
            }
            expected[e as usize].push(p as u32);
        }
        for e in 0..n_log {
            if self.logcnt[e] as usize != self.log2phy[e].len() {
                return false;
            }
            // Equality against the ascending `expected` list simultaneously
            // forces `log2phy[e]` to be ascending and duplicate-free.
            if self.log2phy[e] != expected[e] {
                return false;
            }
        }
        true
    }
}

/// Greedy redundant-expert selection (upstream `get_redundant_experts`).
///
/// Starts with one copy per logical expert and, `num_replicas` times, adds a
/// copy to the expert `e` maximizing `weight[e] / copies[e]` (ties to the
/// lower expert id). Returns the per-logical copy counts, so
/// `copies[e] >= 1` and `sum(copies) == weight.len() + num_replicas`.
///
/// Deterministic and total for any input (a `NaN` weight is treated as
/// minimal, so it never wins a selection unless every weight is `NaN`, in
/// which case expert 0 is cloned). An empty `weight` yields an empty result.
pub fn get_redundant_experts(weight: &[f64], num_replicas: usize) -> Vec<u32> {
    let n = weight.len();
    let mut copies = vec![1u32; n];
    for _ in 0..num_replicas {
        if n == 0 {
            break;
        }
        let mut best = 0usize;
        let mut best_ratio = f64::NEG_INFINITY;
        for e in 0..n {
            let ratio = weight[e] / copies[e] as f64;
            let ratio = if ratio.is_nan() {
                f64::NEG_INFINITY
            } else {
                ratio
            };
            if ratio > best_ratio {
                best = e;
                best_ratio = ratio;
            }
        }
        copies[best] += 1;
    }
    copies
}

/// One physical expert instance: a copy of logical expert `logical`
/// carrying `weight[logical] / copies[logical]` of its load.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Instance {
    /// Logical expert id.
    pub(crate) logical: usize,
    /// Load of this instance (`weight / copies`).
    pub(crate) load: f64,
}

/// Expands logical weights + copy counts into the physical instance list,
/// ordered by (logical id asc, instance ordinal asc).
pub(crate) fn build_instances(weight: &[f64], copies: &[u32]) -> Vec<Instance> {
    let total: usize = copies.iter().map(|&c| c as usize).sum();
    let mut out = Vec::with_capacity(total);
    for (e, &w) in weight.iter().enumerate() {
        let load = w / copies[e] as f64;
        for _ in 0..copies[e] {
            out.push(Instance { logical: e, load });
        }
    }
    out
}

/// Sorts `idx` by (load desc, logical asc, original index asc) — the total
/// LPT feed order shared by both policies.
fn sort_by_load_desc(idx: &[usize], instances: &[Instance]) -> Vec<usize> {
    let mut v = idx.to_vec();
    v.sort_by(|&a, &b| {
        instances[b]
            .load
            .total_cmp(&instances[a].load)
            .then(instances[a].logical.cmp(&instances[b].logical))
            .then(a.cmp(&b))
    });
    v
}

/// LPT list-scheduling of `loads` (in feed order) onto `num_bins` bins:
/// each load goes to the least-loaded bin, ties to the lower bin id.
/// Returns, per bin, the *positions* into `loads` that were assigned, in
/// assignment order. When `capacity` is set, a bin never receives more than
/// `capacity` items (full bins are skipped).
fn lpt_schedule(loads: &[f64], num_bins: usize, capacity: usize) -> Vec<Vec<usize>> {
    let mut bins: Vec<Vec<usize>> = vec![Vec::new(); num_bins];
    let mut bin_load = vec![0.0f64; num_bins];
    for (pos, &l) in loads.iter().enumerate() {
        let mut best = usize::MAX;
        for b in 0..num_bins {
            if bins[b].len() >= capacity {
                continue;
            }
            if best == usize::MAX || bin_load[b] < bin_load[best] {
                best = b;
            }
        }
        if best == usize::MAX {
            // Unreachable for the callers in this module (total capacity is
            // always >= item count); kept total and panic-free regardless.
            continue;
        }
        bins[best].push(pos);
        bin_load[best] += l;
    }
    bins
}

/// Global policy: LPT of all instances onto all GPUs.
fn place_global(instances: &[Instance], cfg: &EplbConfig) -> Vec<Vec<usize>> {
    let order = sort_by_load_desc(&(0..instances.len()).collect::<Vec<_>>(), instances);
    let loads: Vec<f64> = order.iter().map(|&i| instances[i].load).collect();
    lpt_schedule(&loads, cfg.num_gpus, usize::MAX)
        .iter()
        .map(|bin| bin.iter().map(|&pos| order[pos]).collect())
        .collect()
}

/// Hierarchical policy: groups → nodes (capped LPT), instances → node GPUs.
fn place_hierarchical(weight: &[f64], instances: &[Instance], cfg: &EplbConfig) -> Vec<Vec<usize>> {
    let n_groups = cfg.num_groups;
    let n_nodes = cfg.num_nodes;
    let groups_per_node = n_groups / n_nodes;
    let gpus_per_node = cfg.num_gpus / n_nodes;
    let experts_per_group = cfg.num_logical / n_groups;
    // Total load per group = sum of its logical experts' weights (expert-id
    // order). This equals the sum of the group's instance loads up to
    // floating-point rounding; the exact weight sum is the intended
    // group-load definition.
    let mut group_load = vec![0.0f64; n_groups];
    for (e, &w) in weight.iter().enumerate() {
        group_load[e / experts_per_group] += w;
    }
    // Groups by (load desc, group id asc).
    let mut groups: Vec<usize> = (0..n_groups).collect();
    groups.sort_by(|&a, &b| group_load[b].total_cmp(&group_load[a]).then(a.cmp(&b)));
    let loads: Vec<f64> = groups.iter().map(|&g| group_load[g]).collect();
    // Capped LPT: exactly `groups_per_node` groups land on every node.
    let node_bins = lpt_schedule(&loads, n_nodes, groups_per_node);
    // Per node: collect its instances, feed them to the node's GPUs by LPT.
    let mut out: Vec<Vec<usize>> = Vec::with_capacity(cfg.num_gpus);
    for node in 0..n_nodes {
        let mut node_instances: Vec<usize> = Vec::new();
        for &pos in &node_bins[node] {
            let g = groups[pos];
            // Instance indices of this group's logical experts (instances
            // are laid out logical-major, so a simple scan keeps them in
            // (logical, ordinal) order).
            for (i, inst) in instances.iter().enumerate() {
                if inst.logical / experts_per_group == g {
                    node_instances.push(i);
                }
            }
        }
        let order = sort_by_load_desc(&node_instances, instances);
        let gloads: Vec<f64> = order.iter().map(|&i| instances[i].load).collect();
        let gpu_bins = lpt_schedule(&gloads, gpus_per_node, usize::MAX);
        for bin in gpu_bins {
            out.push(bin.iter().map(|&pos| order[pos]).collect());
        }
    }
    out
}

/// Full placement: per GPU (global id, node-major), the assigned instance
/// indices in assignment order. Exposed `pub(crate)` for in-crate tests of
/// the load-balance guarantees.
pub(crate) fn place(weight: &[f64], instances: &[Instance], cfg: &EplbConfig) -> Vec<Vec<usize>> {
    if cfg.is_hierarchical() {
        place_hierarchical(weight, instances, cfg)
    } else {
        place_global(instances, cfg)
    }
}

/// Full EPLB rebalance: greedy redundant experts + policy-selected placement
/// (upstream `rebalance_experts`). Deterministic: identical inputs yield
/// bit-identical [`EplbMaps`].
///
/// The returned `phy2log` is GPU-major as documented on [`EplbMaps`];
/// `log2phy[e]` is the ascending list of physical ids hosting logical `e`;
/// `logcnt` is the copy count vector from [`get_redundant_experts`].
///
/// # Errors
///
/// * [`deepinfer_core::Error::InvalidArgument`] / [`Error::CapacityExceeded`]
///   propagated from [`EplbConfig::validate`];
/// * [`Error::DimensionMismatch`] if `weight.len() != cfg.num_logical`;
/// * [`Error::Numerical`] if any weight is non-finite.
///
/// [`Error::DimensionMismatch`]: deepinfer_core::Error::DimensionMismatch
/// [`Error::InvalidArgument`]: deepinfer_core::Error::InvalidArgument
/// [`Error::CapacityExceeded`]: deepinfer_core::Error::CapacityExceeded
/// [`Error::Numerical`]: deepinfer_core::Error::Numerical
pub fn rebalance_experts(weight: &[f64], cfg: &EplbConfig) -> Result<EplbMaps> {
    cfg.validate()?;
    ensure!(
        weight.len() == cfg.num_logical,
        DimensionMismatch,
        "rebalance_experts: weight.len() = {} but num_logical = {}",
        weight.len(),
        cfg.num_logical
    );
    ensure!(
        weight.iter().all(|w| w.is_finite()),
        Numerical,
        "rebalance_experts: weight contains non-finite values"
    );
    let copies = get_redundant_experts(weight, cfg.num_replicas);
    let instances = build_instances(weight, &copies);
    let per_gpu = place(weight, &instances, cfg);
    let mut phy2log = Vec::with_capacity(cfg.num_physical());
    for gpu in &per_gpu {
        for &i in gpu {
            phy2log.push(instances[i].logical as u32);
        }
    }
    // Physical ids are appended in ascending `p`, so each log2phy list is
    // ascending by construction.
    let mut log2phy: Vec<Vec<u32>> = vec![Vec::new(); cfg.num_logical];
    for (p, &e) in phy2log.iter().enumerate() {
        log2phy[e as usize].push(p as u32);
    }
    Ok(EplbMaps {
        phy2log,
        log2phy,
        logcnt: copies,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepinfer_core::prop::prop;
    use deepinfer_core::rng::Philox;

    fn cfg(logical: usize, replicas: usize, groups: usize, nodes: usize, gpus: usize) -> EplbConfig {
        EplbConfig {
            num_logical: logical,
            num_replicas: replicas,
            num_groups: groups,
            num_nodes: nodes,
            num_gpus: gpus,
        }
    }

    fn rand_weights(rng: &mut Philox, n: usize) -> Vec<f64> {
        (0..n).map(|_| rng.uniform_f32(0.0, 100.0) as f64).collect()
    }

    /// Random *valid* config (both policies reachable).
    fn rand_cfg(rng: &mut Philox) -> EplbConfig {
        let groups = rng.uniform_usize(1, 5);
        let epg = rng.uniform_usize(1, 4);
        let logical = groups * epg;
        let replicas = rng.uniform_usize(0, 6);
        let nodes = rng.uniform_usize(1, 4);
        let gpus = rng.uniform_usize(1, 9).min(logical + replicas).max(1);
        cfg(logical, replicas, groups, nodes, gpus)
    }

    // ------------------------------------------------------------------
    // get_redundant_experts
    // ------------------------------------------------------------------

    #[test]
    fn greedy_redundant_spot() {
        // (2,1) R=3: pick 2 -> e0; tie (1,1) -> e0; (2/3,1) -> e1.
        assert_eq!(get_redundant_experts(&[2.0, 1.0], 3), vec![3, 2]);
        // Ties always clone the lower id first, then the load shifts.
        assert_eq!(get_redundant_experts(&[3.0, 3.0], 2), vec![2, 2]);
        // All-zero weights: every ratio is 0, ties clone expert 0.
        assert_eq!(get_redundant_experts(&[0.0, 0.0], 2), vec![3, 1]);
        // Degenerate cases.
        assert!(get_redundant_experts(&[], 5).is_empty());
        assert_eq!(get_redundant_experts(&[7.0, 1.0], 0), vec![1, 1]);
        // A dominant expert absorbs replicas until its per-copy load drops
        // to the runner-up: 5 -> 2.5 -> 1.67 -> 1.25 -> 1 (tie keeps e0) ->
        // only then e1 is cloned.
        assert_eq!(get_redundant_experts(&[5.0, 1.0], 4), vec![5, 1]);
        assert_eq!(get_redundant_experts(&[5.0, 1.0], 6), vec![6, 2]);
    }

    #[test]
    fn greedy_equalizes_load_per_copy() {
        prop("eplb greedy equalization", 0xE9_1B, 200, |_i, rng| {
            let n = rng.uniform_usize(1, 12);
            let replicas = rng.uniform_usize(0, 2 * n);
            let weight = rand_weights(rng, n);
            let copies = get_redundant_experts(&weight, replicas);
            assert_eq!(copies.len(), n);
            let total: usize = copies.iter().map(|&c| c as usize).sum();
            assert_eq!(total, n + replicas, "copy count must be exact");
            assert!(copies.iter().all(|&c| c >= 1), "every expert keeps >= 1 copy");
            let ratio = |e: usize| weight[e] / copies[e] as f64;
            let max_ratio = (0..n).map(ratio).fold(f64::NEG_INFINITY, f64::max);
            // Experts that were actually cloned (greedy picks) must end
            // within a factor 2 of the final maximum load-per-copy.
            let picked: Vec<usize> = (0..n).filter(|&e| copies[e] > 1).collect();
            if replicas > 0 {
                assert!(!picked.is_empty(), "replicas must be placed somewhere");
                let min_picked = picked.iter().map(|&e| ratio(e)).fold(f64::INFINITY, f64::min);
                assert!(
                    max_ratio <= 2.0 * min_picked + 1e-9,
                    "max ratio {max_ratio} vs 2*min picked {min_picked}"
                );
            }
            Ok(())
        });
    }

    #[test]
    fn greedy_uniform_weights_spread_copies() {
        // With uniform weights the copies must differ by at most one.
        assert_eq!(get_redundant_experts(&[1.0, 1.0, 1.0], 4), vec![3, 2, 2]);
        prop("eplb uniform spread", 0xE9_1C, 100, |_i, rng| {
            let n = rng.uniform_usize(1, 10);
            let replicas = rng.uniform_usize(0, 3 * n);
            let weight = vec![1.0f64; n];
            let copies = get_redundant_experts(&weight, replicas);
            let (lo, hi) = copies.iter().fold((u32::MAX, 0u32), |(lo, hi), &c| {
                (lo.min(c), hi.max(c))
            });
            assert!(hi - lo <= 1, "copies spread {copies:?} exceeds 1");
            Ok(())
        });
    }

    // ------------------------------------------------------------------
    // rebalance_experts: hand-computed placements
    // ------------------------------------------------------------------

    #[test]
    fn global_placement_spot() {
        // L=4, R=2, G=1, N=2 (1 % 2 != 0 -> global), gpus=2.
        // copies = [3,1,1,1]; instances: e0 x3 @ 4/3, e1 @ 2, e2 @ 1, e3 @ 1.
        // Sorted desc: e1(2), e0(4/3)x3, e2(1), e3(1).
        // LPT: e1->g0(2); e0->g1(4/3); e0->g1(8/3); e0->g0(10/3); e2->g1;
        // e3->g0. g0=[e1,e0,e3], g1=[e0,e0,e2].
        let c = cfg(4, 2, 1, 2, 2);
        assert!(!c.is_hierarchical());
        let maps = rebalance_experts(&[4.0, 2.0, 1.0, 1.0], &c).unwrap();
        assert_eq!(maps.phy2log, vec![1, 0, 3, 0, 0, 2]);
        assert_eq!(maps.log2phy, vec![vec![1, 3, 4], vec![0], vec![5], vec![2]]);
        assert_eq!(maps.logcnt, vec![3, 1, 1, 1]);
        assert!(maps.is_consistent(&c));
    }

    #[test]
    fn hierarchical_placement_spot() {
        // L=4 (G=2: group0={0,1}, group1={2,3}), N=2, gpus=4 -> hierarchical.
        // Group loads: g0=2, g1=6 -> g1 to node0, g0 to node1.
        // node0: e2(4)->gpu0, e3(2)->gpu1; node1: e0(1)->gpu2, e1(1)->gpu3.
        let c = cfg(4, 0, 2, 2, 4);
        assert!(c.is_hierarchical());
        let maps = rebalance_experts(&[1.0, 1.0, 4.0, 2.0], &c).unwrap();
        assert_eq!(maps.phy2log, vec![2, 3, 0, 1]);
        assert_eq!(
            maps.log2phy,
            vec![vec![2], vec![3], vec![0], vec![1]]
        );
        assert!(maps.is_consistent(&c));

        // Same topology with one replica: copies=[1,1,2,1]; group loads
        // unchanged (g1=6 -> node0, g0=2 -> node1). node0 instances all
        // carry load 2, ordered e2#a, e2#b, e3 (ties -> lower logical id):
        // e2#a->gpu0, e2#b->gpu1, e3->gpu0 (tie). gpu0=[e2,e3], gpu1=[e2].
        let c2 = cfg(4, 1, 2, 2, 4);
        let maps2 = rebalance_experts(&[1.0, 1.0, 4.0, 2.0], &c2).unwrap();
        assert_eq!(maps2.phy2log, vec![2, 3, 2, 0, 1]);
        assert_eq!(
            maps2.log2phy,
            vec![vec![3], vec![4], vec![0, 2], vec![1]]
        );
        assert_eq!(maps2.logcnt, vec![1, 1, 2, 1]);
        assert!(maps2.is_consistent(&c2));
    }

    #[test]
    fn degenerate_hierarchical_equals_global() {
        // G=1, N=1 is hierarchical but degenerates to the global packing;
        // G=1, N=2 fails 1 % 2 and takes the global path. Same maps.
        let weight = [4.0, 2.0, 1.0, 1.0, 3.0];
        let hier = rebalance_experts(&weight, &cfg(5, 3, 1, 1, 2)).unwrap();
        let glob = rebalance_experts(&weight, &cfg(5, 3, 1, 2, 2)).unwrap();
        assert_eq!(hier, glob);
    }

    #[test]
    fn hierarchical_gives_exact_groups_per_node() {
        // Skewed group loads: capped LPT still lands exactly G/N groups per
        // node (pure uncapped LPT would give 1 vs 3 here).
        let c = cfg(8, 0, 4, 2, 2);
        assert!(c.is_hierarchical());
        let weight = [100.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let maps = rebalance_experts(&weight, &c).unwrap();
        assert!(maps.is_consistent(&c));
        let copies = get_redundant_experts(&weight, 0);
        let instances = build_instances(&weight, &copies);
        let per_gpu = place(&weight, &instances, &c);
        assert_eq!(per_gpu.len(), 2);
        for node in 0..2 {
            let mut groups: Vec<usize> = per_gpu[node]
                .iter()
                .map(|&i| instances[i].logical / (8 / 4))
                .collect();
            groups.sort_unstable();
            groups.dedup();
            assert_eq!(groups.len(), 2, "node {node} must host 2 groups");
        }
    }

    // ------------------------------------------------------------------
    // rebalance_experts: properties
    // ------------------------------------------------------------------

    #[test]
    fn maps_are_consistent_and_complete() {
        prop("eplb maps consistency", 0xE9_2A, 200, |_i, rng| {
            let c = rand_cfg(rng);
            let weight = rand_weights(rng, c.num_logical);
            let maps = rebalance_experts(&weight, &c)?;
            assert!(maps.is_consistent(&c), "maps must be consistent");
            assert_eq!(maps.phy2log.len(), c.num_physical());
            for e in 0..c.num_logical {
                assert!(maps.logcnt[e] >= 1, "logical {e} not placed");
                assert!(!maps.log2phy[e].is_empty());
                let mut sorted = maps.log2phy[e].clone();
                sorted.sort_unstable();
                assert_eq!(sorted, maps.log2phy[e], "log2phy[{e}] must be ascending");
            }
            Ok(())
        });
    }

    #[test]
    fn rebalance_is_deterministic() {
        prop("eplb determinism", 0xE9_2B, 150, |_i, rng| {
            let c = rand_cfg(rng);
            let weight = rand_weights(rng, c.num_logical);
            let a = rebalance_experts(&weight, &c)?;
            let b = rebalance_experts(&weight, &c)?;
            assert_eq!(a, b, "rebalance must be bit-deterministic");
            Ok(())
        });
    }

    #[test]
    fn global_gpu_balance_bound() {
        // List-scheduling guarantee for the global policy: the most loaded
        // GPU is at most `mean + max instance load`. (The hierarchical
        // policy intentionally trades this global bound for topology-aware
        // packing — see the module docs.)
        prop("eplb global balance", 0xE9_2C, 200, |_i, rng| {
            let mut c = rand_cfg(rng);
            let mut tries = 0;
            while c.is_hierarchical() && tries < 64 {
                c = rand_cfg(rng);
                tries += 1;
            }
            if c.is_hierarchical() {
                // Force a global config deterministically.
                c = cfg(c.num_logical, c.num_replicas, 1, 2, 2.min(c.num_physical()));
            }
            assert!(!c.is_hierarchical());
            let weight = rand_weights(rng, c.num_logical);
            let copies = get_redundant_experts(&weight, c.num_replicas);
            let instances = build_instances(&weight, &copies);
            let per_gpu = place(&weight, &instances, &c);
            assert_eq!(per_gpu.len(), c.num_gpus);
            let loads: Vec<f64> = per_gpu
                .iter()
                .map(|g| g.iter().map(|&i| instances[i].load).sum::<f64>())
                .collect();
            let total: f64 = loads.iter().sum();
            let mean = total / c.num_gpus as f64;
            let max_inst = instances
                .iter()
                .map(|x| x.load)
                .fold(f64::NEG_INFINITY, f64::max);
            let max_load = loads.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            assert!(
                max_load <= mean + max_inst + 1e-9,
                "max gpu load {max_load} exceeds mean {mean} + max instance {max_inst}"
            );
            // Every instance is placed exactly once.
            let placed: usize = per_gpu.iter().map(|g| g.len()).sum();
            assert_eq!(placed, instances.len());
            Ok(())
        });
    }

    #[test]
    fn hierarchical_node_balance_bound() {
        // The node-level LPT guarantee: max node load <= mean node load +
        // max group load. Node loads are measured as the sum of the
        // instances placed on the node's GPUs.
        prop("eplb node balance", 0xE9_2D, 200, |_i, rng| {
            let mut c = rand_cfg(rng);
            let mut tries = 0;
            while !c.is_hierarchical() && tries < 64 {
                c = rand_cfg(rng);
                tries += 1;
            }
            if !c.is_hierarchical() {
                c = cfg(c.num_logical, c.num_replicas, 2, 2, 2.min(c.num_physical()));
            }
            let weight = rand_weights(rng, c.num_logical);
            let epg = c.num_logical / c.num_groups;
            let mut group_load = vec![0.0f64; c.num_groups];
            for (e, &w) in weight.iter().enumerate() {
                group_load[e / epg] += w;
            }
            let max_group = group_load.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let copies = get_redundant_experts(&weight, c.num_replicas);
            let instances = build_instances(&weight, &copies);
            let per_gpu = place(&weight, &instances, &c);
            let gpus_per_node = c.num_gpus / c.num_nodes;
            let node_loads: Vec<f64> = (0..c.num_nodes)
                .map(|node| {
                    per_gpu[node * gpus_per_node..(node + 1) * gpus_per_node]
                        .iter()
                        .flat_map(|g| g.iter().map(|&i| instances[i].load))
                        .sum()
                })
                .collect();
            let total: f64 = node_loads.iter().sum();
            let mean = total / c.num_nodes as f64;
            let max_load = node_loads.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            assert!(
                max_load <= mean + max_group + 1e-9,
                "max node load {max_load} exceeds mean {mean} + max group {max_group}"
            );
            Ok(())
        });
    }

    #[test]
    fn rebalance_validates_inputs() {
        let c = cfg(4, 0, 2, 2, 2);
        assert!(rebalance_experts(&[1.0, 2.0], &c).is_err()); // wrong length
        assert!(rebalance_experts(&[1.0, 2.0, f64::NAN, 0.0], &c).is_err()); // NaN
        assert!(rebalance_experts(&[1.0, 2.0, 0.0, f64::INFINITY], &c).is_err());
        let bad = cfg(3, 0, 2, 2, 2); // 3 % 2 != 0
        assert!(rebalance_experts(&[1.0, 2.0, 0.0], &bad).is_err());
    }

    // ------------------------------------------------------------------
    // EplbMaps::is_consistent
    // ------------------------------------------------------------------

    #[test]
    fn is_consistent_rejects_broken_maps() {
        // Replicas guarantee some log2phy entry has >= 2 physical ids.
        let c = cfg(4, 2, 2, 1, 2);
        let maps = rebalance_experts(&[1.0, 2.0, 3.0, 4.0], &c).unwrap();
        assert!(maps.is_consistent(&c));
        let multi = (0..c.num_logical)
            .find(|&e| maps.log2phy[e].len() >= 2)
            .expect("replicas must create a multi-copy logical expert");

        let mut m = maps.clone();
        m.phy2log[0] = 99;
        assert!(!m.is_consistent(&c)); // invalid logical id

        let mut m = maps.clone();
        m.phy2log[0] = m.phy2log[1];
        assert!(!m.is_consistent(&c)); // round-trip broken

        let mut m = maps.clone();
        m.log2phy[0].push(0);
        assert!(!m.is_consistent(&c)); // counts no longer match

        let mut m = maps.clone();
        m.logcnt[0] += 1;
        assert!(!m.is_consistent(&c));

        let mut m = maps.clone();
        m.log2phy[multi].reverse();
        assert!(!m.is_consistent(&c)); // must be ascending

        let mut m = maps.clone();
        m.log2phy.truncate(3);
        assert!(!m.is_consistent(&c)); // wrong lengths

        assert!(!maps.is_consistent(&cfg(4, 0, 3, 1, 2))); // invalid cfg
    }
}
