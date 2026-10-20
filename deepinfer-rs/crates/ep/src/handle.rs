//! Dispatch handles: the routing bookkeeping that travels with (or mirrors)
//! a payload.

use crate::hook::EpHook;

/// Routing bookkeeping for one dispatch, mirroring DeepEP's dispatch
/// metadata (source rank, per-row token slots and top-k routing, per-expert
/// receive counts, completion hook).
///
/// * [`dispatch`](crate::dispatch) returns the **sender-side** handle:
///     `token_slots` lists **all** `(token, k)` pairs of the sender's batch
///     in the sender's row order (`(t asc, k asc)` — row-major over
///     `[tokens, k]`), so slot `i` corresponds to `topk_idx[i]` /
///     `topk_weights[i]`, and `recv_counts_per_expert[e]` counts the
///     sender's slots routed to expert `e` (length `num_experts`).
/// * [`recv`](crate::recv) returns a **receiver-side mirror** inside
///     [`DispatchResult`](crate::DispatchResult): `src_rank` is the
///     receiving rank, the slots are the true (unpadded) received rows in
///     receive order, and `recv_counts_per_expert` holds the true
///     per-expert receive counts.
/// * [`low_latency_dispatch`](crate::low_latency_dispatch) returns the
///     receiver-side mirror for the masked layout:
///     `recv_counts_per_expert` has `num_local_experts` entries (the
///     `masked_m` counts).
///
/// `token_slots` is `Vec<(u32, u32)>` — `(token index in the sender batch,
/// k index)` per row — which is the pair encoding the SPEC's field comment
/// describes ("(token idx in sender batch, k) per received row").
///
/// [`combine`](crate::combine) consumes a handle whose slots index the rows
/// of the `expert_out` buffer it is given; see its docs for the exact
/// contract.
#[derive(Debug)]
pub struct DispatchHandle {
    /// Rank that produced the slots (sender for dispatch handles, receiver
    /// for the mirrors inside [`DispatchResult`](crate::DispatchResult) /
    /// [`LowLatencyDispatchResult`](crate::LowLatencyDispatchResult)).
    pub src_rank: usize,
    /// `(token index in the sender batch, k index)` per row, in the
    /// handle's row order.
    pub token_slots: Vec<(u32, u32)>,
    /// Routed expert id per row (parallel to `token_slots`).
    pub topk_idx: Vec<u32>,
    /// Routing weight per row (parallel to `token_slots`).
    pub topk_weights: Vec<f32>,
    /// Receive counts: normal mode `[num_experts]`; low-latency mode
    /// `[num_local_experts]` (the `masked_m`).
    pub recv_counts_per_expert: Vec<usize>,
    /// Completion hook (always fulfilled in this simulation).
    pub hook: EpHook,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handle_holds_parallel_bookkeeping() {
        let h = DispatchHandle {
            src_rank: 3,
            token_slots: vec![(0, 0), (0, 1), (2, 0)],
            topk_idx: vec![4, 5, 4],
            topk_weights: vec![0.2, 0.3, 0.5],
            recv_counts_per_expert: vec![0, 0, 0, 0, 2, 1],
            hook: EpHook::new(),
        };
        assert_eq!(h.src_rank, 3);
        assert_eq!(h.token_slots.len(), h.topk_idx.len());
        assert_eq!(h.token_slots.len(), h.topk_weights.len());
        assert!(h.hook.is_fulfilled());
        assert_eq!(h.recv_counts_per_expert.iter().sum::<usize>(), h.token_slots.len());
    }
}
