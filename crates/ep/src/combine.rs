//! Combine: weighted sum of expert outputs back to the sender's original
//! token order (DeepEP `combine` / `low_latency_combine` semantics).

use crate::config::EpConfig;
use crate::handle::DispatchHandle;
use deepinfer_core::{ensure, Error, Result};

/// Combines expert outputs back into the sender's token order:
/// `y[t] = Σ_k topk_weights[t,k] · expert_out[t,k]`.
///
/// `handle` is a handle whose `token_slots` index the rows of `expert_out`
/// — slot `i` corresponds to `expert_out` row `i`. The intended input is
/// the **sender-side** handle returned by [`dispatch`](crate::dispatch)
/// (slots in the sender's row order, one per `(token, k)` pair), with
/// `expert_out` gathered by the caller from the receiving ranks in that
/// same slot order. The receiver-side mirrors inside
/// [`DispatchResult`](crate::DispatchResult) /
/// [`LowLatencyDispatchResult`](crate::LowLatencyDispatchResult) list only
/// the true received rows, so they can be combined after stripping the
/// alignment padding from the expert outputs.
///
/// Deterministic accumulation: contributions of one token are summed in
/// ascending `(k, slot index)` order, tokens in ascending index order —
/// sequential `f32` throughout.
///
/// The output has one row per token of `handle.src_rank`'s batch:
/// `max(token) + 1` rows (`0` when the handle has no slots). Trailing
/// tokens without slots cannot be inferred from a handle and are not
/// appended; `dispatch`-produced handles always cover every token
/// (`k >= 1`).
///
/// # Errors
///
/// * `DimensionMismatch` when the handle's parallel arrays disagree, when
///   `expert_out.len()` is not a multiple of `hidden`, or when
///   `expert_out` does not have exactly one row per handle slot;
/// * `Numerical` for non-finite weights or expert outputs;
/// * `CapacityExceeded` on output-size overflow;
/// * the [`EpConfig::validate`] family.
pub fn combine(cfg: &EpConfig, handle: &DispatchHandle, expert_out: &[f32]) -> Result<Vec<f32>> {
    cfg.validate()?;
    let hidden = cfg.hidden;
    let slots = handle.token_slots.len();
    ensure!(
        handle.topk_idx.len() == slots && handle.topk_weights.len() == slots,
        DimensionMismatch,
        "combine: handle has {slots} slots but topk_idx.len() = {} and \
         topk_weights.len() = {}",
        handle.topk_idx.len(),
        handle.topk_weights.len()
    );
    ensure!(
        expert_out.len() % hidden == 0,
        DimensionMismatch,
        "combine: expert_out.len() = {} is not a multiple of hidden ({hidden})",
        expert_out.len()
    );
    let rows = expert_out.len() / hidden;
    ensure!(
        rows == slots,
        DimensionMismatch,
        "combine: expert_out has {rows} rows but the handle lists {slots} slots"
    );
    ensure!(
        handle.topk_weights.iter().all(|w| w.is_finite()),
        Numerical,
        "combine: handle.topk_weights contains non-finite values"
    );
    ensure!(
        expert_out.iter().all(|v| v.is_finite()),
        Numerical,
        "combine: expert_out contains non-finite values"
    );
    let tokens = handle
        .token_slots
        .iter()
        .map(|&(t, _)| t as usize)
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);
    let out_len = tokens.checked_mul(hidden).ok_or_else(|| {
        Error::CapacityExceeded(format!(
            "combine: tokens ({tokens}) * hidden ({hidden}) overflows usize"
        ))
    })?;
    let mut y = vec![0.0f32; out_len];
    // Accumulation order: per token, ascending (k, slot index).
    let mut order: Vec<usize> = (0..slots).collect();
    order.sort_by(|&a, &b| {
        handle.token_slots[a]
            .0
            .cmp(&handle.token_slots[b].0)
            .then(handle.token_slots[a].1.cmp(&handle.token_slots[b].1))
            .then(a.cmp(&b))
    });
    for &i in order.iter() {
        let (t, _ki) = handle.token_slots[i];
        let base = t as usize * hidden;
        let w = handle.topk_weights[i];
        for d in 0..hidden {
            y[base + d] += w * expert_out[i * hidden + d];
        }
    }
    Ok(y)
}
