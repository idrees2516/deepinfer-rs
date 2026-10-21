//! Low-latency dispatch (DeepEP `low_latency_dispatch` semantics: fixed
//! per-expert capacity, masked layout, overflow drops).

use crate::config::EpConfig;
use crate::dispatch::validate_routing;
use crate::handle::DispatchHandle;
use crate::hook::EpHook;
use crate::payload::{decode_payload, quantize_payload};
use deepinfer_core::{ensure, Error, Result};

/// Result of [`low_latency_dispatch`].
#[derive(Debug)]
pub struct LowLatencyDispatchResult {
    /// Received rows in the masked layout, row-major
    /// `[num_local_experts * capacity, hidden]`, **dequantized** to `f32`.
    /// Rows of local expert `g` occupy `[g * capacity, g * capacity +
    /// masked_m[g])`; every row beyond the count is **zero**.
    pub recv_x: Vec<f32>,
    /// True row counts per local expert (`[num_local_experts]`) — the
    /// DeepGEMM masked-grouped-GEMM `masked_m` input.
    pub masked_m: Vec<usize>,
    /// Receiver-side handle mirror: `token_slots` lists the placed rows in
    /// masked-layout order (expert segment major, fill order within the
    /// segment), `recv_counts_per_expert` is `masked_m`, and the hook is
    /// immediately fulfilled.
    pub handle: DispatchHandle,
    /// **Documented SPEC extension**: `(token, k)` pairs dropped because
    /// their local expert's capacity was exhausted, in deterministic
    /// `(token, k)` order (single sender per call; the multi-rank fill
    /// order would be `(sender rank, token, k)`).
    pub dropped: Vec<(u32, u32)>,
}

/// Low-latency dispatch with a fixed capacity per local expert.
///
/// Mirrors DeepEP's `low_latency_dispatch`: rows of local expert `g` live
/// at `[g * capacity, g * capacity + masked_m[g])` of the returned
/// `recv_x`, rows beyond the count are zero, and pairs that overflow an
/// expert's capacity are **dropped and reported** in
/// [`LowLatencyDispatchResult::dropped`] rather than erroring (the SPEC's
/// "documented choice").
///
/// In-process simulation note (documented deviation): this single-rank call
/// sees only `rank`'s own batch, so the local buffers are filled with the
/// **self-routed** `(token, k)` pairs — pairs owned by peer ranks are
/// neither dropped nor placed (in the real transport they land in the
/// peers' own buffers, whose fills happen in the peers' calls). Pairs are
/// placed in `(token, k)` order, the single-sender instance of DeepEP's
/// `(sender rank, token, k)` fill order.
///
/// `capacity` is DeepEP's `num_max_tokens_per_rank` buffer depth (passed
/// explicitly here); it must be `>= 1`.
///
/// # Errors
///
/// Same validation family as [`dispatch`](crate::dispatch) (config, rank,
/// shapes, finiteness, expert range) plus `InvalidArgument` for
/// `capacity == 0`, `CapacityExceeded` for layout overflows, and FP8
/// quantization errors (e.g. `hidden % 128 != 0`).
pub fn low_latency_dispatch(
    cfg: &EpConfig,
    rank: usize,
    x: &[f32],
    topk_idx: &[u32],
    topk_weights: &[f32],
    tokens: usize,
    k: usize,
    capacity: usize,
    fp8: bool,
) -> Result<LowLatencyDispatchResult> {
    cfg.validate()?;
    ensure!(
        rank < cfg.world_size,
        OutOfRange,
        "low_latency_dispatch: rank ({rank}) must be < world_size ({})",
        cfg.world_size
    );
    ensure!(
        capacity >= 1,
        InvalidArgument,
        "low_latency_dispatch: capacity must be >= 1, got {capacity}"
    );
    validate_routing(cfg, x, topk_idx, topk_weights, tokens, k)?;
    let local = cfg.num_local_experts();
    let hidden = cfg.hidden;
    // Quantize the whole outgoing batch (mirrors the FP8 wire format), then
    // decode back so the placed rows carry the quantization error exactly
    // like a real FP8 round trip.
    let decoded: Vec<f32> = if fp8 {
        let payload = quantize_payload(x, cfg)?;
        decode_payload(&payload, cfg)?
    } else {
        x.to_vec()
    };
    let first_local = rank * local;
    // Place the self-routed pairs, (token, k) order.
    let mut masked_m = vec![0usize; local];
    let mut dropped: Vec<(u32, u32)> = Vec::new();
    let mut placed: Vec<Vec<(u32, u32, f32)>> = vec![Vec::new(); local];
    for t in 0..tokens {
        for ki in 0..k {
            let i = t * k + ki;
            let e = topk_idx[i] as usize;
            if cfg.owner_rank(topk_idx[i]) != rank {
                continue; // routed to a peer rank: not dropped, not placed
            }
            let g = e - first_local;
            if masked_m[g] < capacity {
                masked_m[g] += 1;
                placed[g].push((t as u32, ki as u32, topk_weights[i]));
            } else {
                dropped.push((t as u32, ki as u32));
            }
        }
    }
    let recv_len = local
        .checked_mul(capacity)
        .and_then(|v| v.checked_mul(hidden))
        .ok_or_else(|| {
            Error::CapacityExceeded(format!(
                "low_latency_dispatch: num_local_experts ({local}) * capacity ({capacity}) \
                 * hidden ({hidden}) overflows usize"
            ))
        })?;
    let mut recv_x = vec![0.0f32; recv_len];
    let n_placed: usize = masked_m.iter().sum();
    let mut slots = Vec::with_capacity(n_placed);
    let mut slot_topk_idx = Vec::with_capacity(n_placed);
    let mut slot_topk_weights = Vec::with_capacity(n_placed);
    for g in 0..local {
        for (j, &(t, ki, w)) in placed[g].iter().enumerate() {
            let row = g * capacity + j;
            let base = t as usize * hidden;
            recv_x[row * hidden..(row + 1) * hidden]
                .copy_from_slice(&decoded[base..base + hidden]);
            slots.push((t, ki));
            slot_topk_idx.push((first_local + g) as u32);
            slot_topk_weights.push(w);
        }
    }
    let handle = DispatchHandle {
        src_rank: rank,
        token_slots: slots,
        topk_idx: slot_topk_idx,
        topk_weights: slot_topk_weights,
        recv_counts_per_expert: masked_m.clone(),
        hook: EpHook::new(),
    };
    Ok(LowLatencyDispatchResult {
        recv_x,
        masked_m,
        handle,
        dropped,
    })
}
