//! Normal-mode dispatch and receive (DeepEP `intranode_dispatch` /
//! `internode_dispatch` + `recv` semantics, simulated in-process).

use crate::config::EpConfig;
use crate::handle::DispatchHandle;
use crate::hook::EpHook;
use crate::payload::{decode_payload, quantize_payload, DispatchPayload};
use crate::align_up;
use deepinfer_core::{ensure, Error, Result};

/// Result of [`recv`]: the rows destined for one rank, plus the
/// receiver-side handle mirror.
#[derive(Debug)]
pub struct DispatchResult {
    /// Received token rows, row-major `[rows, hidden]`, **dequantized** to
    /// `f32`. Rows are grouped by expert (ascending expert id); within an
    /// expert segment the order is `(sender rank, token index, k)`; each
    /// expert's segment is padded to a multiple of
    /// `cfg.expert_alignment` with **zero rows** appended at the end of the
    /// segment (the true counts live in `handle.recv_counts_per_expert`).
    pub recv_x: Vec<f32>,
    /// Original token index (in the sender's batch) per row. Padding rows
    /// carry `0` — their metadata is meaningless by construction.
    pub recv_src_idx: Vec<u32>,
    /// Routed expert id per row. Padding rows carry their segment's expert
    /// id, so the tensor doubles as a grouped-GEMM `m_indices` input.
    pub recv_topk_idx: Vec<u32>,
    /// Routing weight per row. Padding rows carry `0.0`.
    pub recv_topk_weights: Vec<f32>,
    /// Receiver-side handle mirror: `src_rank` is the receiving rank,
    /// `token_slots` lists the true (unpadded) received rows in receive
    /// order, and `recv_counts_per_expert` holds the true per-expert
    /// receive counts.
    pub handle: DispatchHandle,
}

/// Validates the routing inputs shared by [`dispatch`] and
/// [`low_latency_dispatch`].
pub(crate) fn validate_routing(
    cfg: &EpConfig,
    x: &[f32],
    topk_idx: &[u32],
    topk_weights: &[f32],
    tokens: usize,
    k: usize,
) -> Result<()> {
    cfg.validate()?;
    ensure!(k >= 1, InvalidArgument, "k must be >= 1, got {k}");
    ensure!(
        tokens <= u32::MAX as usize,
        OutOfRange,
        "tokens ({tokens}) must be <= u32::MAX"
    );
    ensure!(
        k <= u32::MAX as usize,
        OutOfRange,
        "k ({k}) must be <= u32::MAX"
    );
    let pairs = tokens.checked_mul(k).ok_or_else(|| {
        Error::CapacityExceeded(format!("dispatch: tokens ({tokens}) * k ({k}) overflows usize"))
    })?;
    ensure!(
        topk_idx.len() == pairs,
        DimensionMismatch,
        "topk_idx.len() = {} but tokens*k = {pairs}",
        topk_idx.len()
    );
    ensure!(
        topk_weights.len() == pairs,
        DimensionMismatch,
        "topk_weights.len() = {} but tokens*k = {pairs}",
        topk_weights.len()
    );
    let cells = tokens.checked_mul(cfg.hidden).ok_or_else(|| {
        Error::CapacityExceeded(format!(
            "dispatch: tokens ({tokens}) * hidden ({}) overflows usize",
            cfg.hidden
        ))
    })?;
    ensure!(
        x.len() == cells,
        DimensionMismatch,
        "x.len() = {} but tokens*hidden = {cells}",
        x.len()
    );
    ensure!(
        x.iter().all(|v| v.is_finite()),
        Numerical,
        "x contains non-finite values"
    );
    ensure!(
        topk_weights.iter().all(|v| v.is_finite()),
        Numerical,
        "topk_weights contains non-finite values"
    );
    ensure!(
        topk_idx.iter().all(|&e| (e as usize) < cfg.num_experts),
        OutOfRange,
        "topk_idx contains an expert >= num_experts ({})",
        cfg.num_experts
    );
    Ok(())
}

/// Normal-mode dispatch: routes each `(token, k)` pair of `rank`'s batch to
/// the owner rank of its expert, returning the payload to "send" and the
/// sender-side handle.
///
/// The payload carries the whole batch — raw `f32` rows, or (when `fp8`)
/// per-token-group E4M3 + UE8M0 quantization with granularity
/// [`FP8_QUANT_GROUP`](crate::FP8_QUANT_GROUP) = 128 via `deepinfer-fp8`
/// (requires `hidden % 128 == 0`). The handle records, in the sender's row
/// order (`t` asc, `k` asc): the `(token, k)` slots, their experts and
/// weights, the per-expert counts `[num_experts]`, and an (immediately
/// fulfilled) [`EpHook`].
///
/// Capacity contract: no receiver may be handed more than
/// `cfg.num_max_tokens_per_rank` slots by this sender (DeepEP's per-pair
/// channel buffer sizing) — violated configurations error with
/// `CapacityExceeded` instead of overrunning a buffer.
///
/// # Errors
///
/// * `InvalidArgument` / `OutOfRange` from [`EpConfig::validate`] or
///   `k < 1` / `rank >= world_size`;
/// * `DimensionMismatch` on any length mismatch;
/// * `Numerical` on non-finite inputs;
/// * `CapacityExceeded` on overflow or per-receiver oversubscription;
/// * FP8 quantization errors (e.g. `hidden % 128 != 0`) from
///   `deepinfer-fp8`.
pub fn dispatch(
    cfg: &EpConfig,
    rank: usize,
    x: &[f32],
    topk_idx: &[u32],
    topk_weights: &[f32],
    tokens: usize,
    k: usize,
    fp8: bool,
) -> Result<(DispatchPayload, DispatchHandle)> {
    cfg.validate()?;
    ensure!(
        rank < cfg.world_size,
        OutOfRange,
        "dispatch: rank ({rank}) must be < world_size ({})",
        cfg.world_size
    );
    validate_routing(cfg, x, topk_idx, topk_weights, tokens, k)?;
    // Per-receiver channel capacity (DeepEP sizes normal-mode buffers per
    // sender-receiver pair).
    let mut per_receiver = vec![0usize; cfg.world_size];
    for &e in topk_idx.iter() {
        per_receiver[cfg.owner_rank(e)] += 1;
    }
    for (r, &c) in per_receiver.iter().enumerate() {
        ensure!(
            c <= cfg.num_max_tokens_per_rank,
            CapacityExceeded,
            "dispatch: rank {rank} routes {c} token slots to rank {r} but \
             num_max_tokens_per_rank = {}",
            cfg.num_max_tokens_per_rank
        );
    }
    let payload = if fp8 {
        quantize_payload(x, cfg)?
    } else {
        DispatchPayload::F32(x.to_vec())
    };
    let pairs = tokens * k;
    let mut token_slots = Vec::with_capacity(pairs);
    let mut slot_topk_idx = Vec::with_capacity(pairs);
    let mut slot_topk_weights = Vec::with_capacity(pairs);
    let mut counts = vec![0usize; cfg.num_experts];
    for t in 0..tokens {
        for ki in 0..k {
            let i = t * k + ki;
            token_slots.push((t as u32, ki as u32));
            slot_topk_idx.push(topk_idx[i]);
            slot_topk_weights.push(topk_weights[i]);
            counts[topk_idx[i] as usize] += 1;
        }
    }
    let handle = DispatchHandle {
        src_rank: rank,
        token_slots,
        topk_idx: slot_topk_idx,
        topk_weights: slot_topk_weights,
        recv_counts_per_expert: counts,
        hook: EpHook::new(),
    };
    Ok((payload, handle))
}

/// Validates the internal consistency of a (possibly caller-assembled)
/// handle.
fn validate_handle(cfg: &EpConfig, handle: &DispatchHandle) -> Result<()> {
    ensure!(
        handle.src_rank < cfg.world_size,
        OutOfRange,
        "recv: handle.src_rank ({}) must be < world_size ({})",
        handle.src_rank,
        cfg.world_size
    );
    let n = handle.token_slots.len();
    ensure!(
        handle.topk_idx.len() == n && handle.topk_weights.len() == n,
        DimensionMismatch,
        "recv: handle has {} slots but topk_idx.len() = {} and topk_weights.len() = {}",
        n,
        handle.topk_idx.len(),
        handle.topk_weights.len()
    );
    ensure!(
        handle.topk_idx.iter().all(|&e| (e as usize) < cfg.num_experts),
        OutOfRange,
        "recv: handle.topk_idx contains an expert >= num_experts ({})",
        cfg.num_experts
    );
    ensure!(
        handle.topk_weights.iter().all(|v| v.is_finite()),
        Numerical,
        "recv: handle.topk_weights contains non-finite values"
    );
    Ok(())
}

/// One collected receive row (before ordering).
struct PendingRow {
    expert: u32,
    sender: usize,
    token: u32,
    ki: u32,
    weight: f32,
    /// Index into `pending` (the decoded source batch to copy the row from).
    payload_idx: usize,
}

/// Delivers every dispatched row destined for `rank` (in-process mailbox
/// drain of the collective: all ranks must have called [`dispatch`] and
/// their `(payload, handle)` pairs passed in `pending`).
///
/// Row order is the deterministic total order `(expert, sender rank, token
/// index, k)`; each expert's segment is then padded to a multiple of
/// `cfg.expert_alignment` with zero rows (so `recv_x` feeds DeepGEMM's
/// grouped-contiguous layout; the **true** counts remain available in
/// `handle.recv_counts_per_expert`). Padding rows carry `recv_src_idx = 0`,
/// their segment's expert id and weight `0.0`.
///
/// The result is independent of the order of `pending` (the sort key
/// includes the sender rank). Enforces the per-sender half of the
/// `num_max_tokens_per_rank` capacity contract on the received traffic.
///
/// # Errors
///
/// * `OutOfRange` for `rank >= world_size`, a malformed handle
///   (`src_rank`, expert ids) or a slot referencing a token row beyond its
///   payload;
/// * `DimensionMismatch` for inconsistent handle/payload shapes;
/// * `Numerical` for non-finite weights;
/// * `CapacityExceeded` for size overflows or per-sender
///   oversubscription;
/// * FP8 dequantization errors for malformed FP8 payloads.
pub fn recv(
    cfg: &EpConfig,
    rank: usize,
    pending: &[(DispatchPayload, DispatchHandle)],
) -> Result<DispatchResult> {
    cfg.validate()?;
    ensure!(
        rank < cfg.world_size,
        OutOfRange,
        "recv: rank ({rank}) must be < world_size ({})",
        cfg.world_size
    );
    let hidden = cfg.hidden;
    // Decode every payload once (validates the buffers, then reuse rows).
    let mut decoded: Vec<Vec<f32>> = Vec::with_capacity(pending.len());
    for (payload, _) in pending.iter() {
        decoded.push(decode_payload(payload, cfg)?);
    }
    // Collect the rows destined for `rank`.
    let mut rows: Vec<PendingRow> = Vec::new();
    let mut per_sender = vec![0usize; cfg.world_size];
    for (pi, ((_, handle), batch)) in pending.iter().zip(decoded.iter()).enumerate() {
        validate_handle(cfg, handle)?;
        let batch_rows = batch.len() / hidden;
        for (si, &(t, ki)) in handle.token_slots.iter().enumerate() {
            ensure!(
                (t as usize) < batch_rows,
                OutOfRange,
                "recv: handle from rank {} references token {t} but its payload \
                 only has {batch_rows} rows",
                handle.src_rank
            );
            let e = handle.topk_idx[si];
            if cfg.owner_rank(e) == rank {
                per_sender[handle.src_rank] += 1;
                rows.push(PendingRow {
                    expert: e,
                    sender: handle.src_rank,
                    token: t,
                    ki,
                    weight: handle.topk_weights[si],
                    payload_idx: pi,
                });
            }
        }
    }
    for (s, &c) in per_sender.iter().enumerate() {
        ensure!(
            c <= cfg.num_max_tokens_per_rank,
            CapacityExceeded,
            "recv: rank {s} delivered {c} token slots to rank {rank} but \
             num_max_tokens_per_rank = {}",
            cfg.num_max_tokens_per_rank
        );
    }
    // Deterministic total order: (expert, sender, token, k).
    rows.sort_by(|a, b| {
        a.expert
            .cmp(&b.expert)
            .then(a.sender.cmp(&b.sender))
            .then(a.token.cmp(&b.token))
            .then(a.ki.cmp(&b.ki))
    });
    // Per-expert true counts and padded segment starts.
    let mut counts = vec![0usize; cfg.num_experts];
    for r in rows.iter() {
        counts[r.expert as usize] += 1;
    }
    let mut seg_start = vec![0usize; cfg.num_experts];
    let mut total_rows = 0usize;
    for e in 0..cfg.num_experts {
        seg_start[e] = total_rows;
        total_rows = total_rows
            .checked_add(align_up(counts[e], cfg.expert_alignment)?)
            .ok_or_else(|| {
                Error::CapacityExceeded("recv: padded row count overflows usize".into())
            })?;
    }
    let recv_len = total_rows.checked_mul(hidden).ok_or_else(|| {
        Error::CapacityExceeded(format!(
            "recv: {total_rows} rows * hidden ({hidden}) overflows usize"
        ))
    })?;
    let mut recv_x = vec![0.0f32; recv_len];
    let mut recv_src_idx = vec![0u32; total_rows];
    let mut recv_topk_idx = vec![0u32; total_rows];
    let mut recv_topk_weights = vec![0.0f32; total_rows];
    // Padding rows inherit their segment's expert id (usable as
    // m_indices); everything else about them is zeroed by initialization.
    for e in 0..cfg.num_experts {
        let padded = align_up(counts[e], cfg.expert_alignment)?;
        for j in counts[e]..padded {
            recv_topk_idx[seg_start[e] + j] = e as u32;
        }
    }
    // Receiver-side handle mirror over the true rows (receive order).
    let mut slots = Vec::with_capacity(rows.len());
    let mut slot_topk_idx = Vec::with_capacity(rows.len());
    let mut slot_topk_weights = Vec::with_capacity(rows.len());
    let mut cursor = seg_start.clone();
    for r in rows.iter() {
        let e = r.expert as usize;
        let row = cursor[e];
        cursor[e] += 1;
        let batch = &decoded[r.payload_idx];
        let t = r.token as usize;
        recv_x[row * hidden..(row + 1) * hidden]
            .copy_from_slice(&batch[t * hidden..(t + 1) * hidden]);
        recv_src_idx[row] = r.token;
        recv_topk_idx[row] = r.expert;
        recv_topk_weights[row] = r.weight;
        slots.push((r.token, r.ki));
        slot_topk_idx.push(r.expert);
        slot_topk_weights.push(r.weight);
    }
    let handle = DispatchHandle {
        src_rank: rank,
        token_slots: slots,
        topk_idx: slot_topk_idx,
        topk_weights: slot_topk_weights,
        recv_counts_per_expert: counts,
        hook: EpHook::new(),
    };
    Ok(DispatchResult {
        recv_x,
        recv_src_idx,
        recv_topk_idx,
        recv_topk_weights,
        handle,
    })
}
