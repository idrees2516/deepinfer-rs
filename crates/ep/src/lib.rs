//! # deepinfer-ep
//!
//! DeepEP-style expert-parallelism dispatch/combine for **DeepInfer-rs** —
//! an in-process, single-threaded simulation of DeepSeek's
//! [DeepEP](https://github.com/deepseek-ai/DeepEP) communication kernels,
//! mirroring the upstream Python API shapes (normal + low-latency modes,
//! FP8 payloads, deferred-completion hooks).
//!
//! * **Normal dispatch** ([`dispatch`]): every `(token, k)` routing pair of
//!   a rank's batch is routed to the *owner rank* of its expert
//!   (`expert * world_size / num_experts`, block-contiguous ownership).
//!   The payload carries the token rows either raw (BF32/FP32 mode) or
//!   quantized per-token-group E4M3 + UE8M0 scales (FP8 mode, granularity
//!   `1 x 128` along hidden, via `deepinfer-fp8`).
//! * **Receive** ([`recv`]): assembles the rows destined for one rank from
//!   all pending dispatches in the deterministic order
//!   `(expert, sender rank, token index, k)` — the order DeepEP's
//!   `dispatch`/`return_recv_hook` reference implementation documents — and
//!   pads each expert's row segment up to a multiple of
//!   `expert_alignment` with zero rows, so the result feeds
//!   DeepGEMM's `m_grouped_contiguous` layout directly.
//! * **Low-latency dispatch** ([`low_latency_dispatch`]): fixed capacity per
//!   local expert, masked `[num_local_experts, capacity, hidden]` layout
//!   with `masked_m` counts; over-capacity pairs are dropped and reported.
//! * **Combine** ([`combine`]): weighted sum of expert outputs back to the
//!   sender's original token order (`y[t] = Σ_k w_k · expert_out[t, k]`,
//!   accumulated in ascending `k`).
//!
//! In-process simulation notes (documented deviations from the
//! multi-process upstream): dispatch/recv are decoupled through explicit
//! `(payload, handle)` "mailbox" slices instead of NVLink/RDMA buffers, and
//! [`EpHook`] is always fulfilled at construction — it exists to mirror
//! DeepEP's `EventOverlap` / `return_recv_hook` structure for future async
//! transports.
//!
//! Determinism contract: all row orders are total orders, no hash-ordered
//! iteration participates in any output, and accumulation is sequential
//! `f32` — identical inputs give bit-identical outputs.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod combine;
pub mod config;
pub mod dispatch;
pub mod handle;
pub mod hook;
pub mod low_latency;
pub mod payload;

pub use combine::combine;
pub use config::EpConfig;
pub use dispatch::{dispatch, recv, DispatchResult};
pub use handle::DispatchHandle;
pub use hook::EpHook;
pub use low_latency::{low_latency_dispatch, LowLatencyDispatchResult};
pub use payload::{DispatchPayload, FP8_QUANT_GROUP};

use deepinfer_core::Error;

/// Pads a token count up to `alignment`: `ceil(n / alignment) * alignment`
/// (checked; overflow is an error, not a panic).
pub(crate) fn align_up(n: usize, alignment: usize) -> deepinfer_core::Result<usize> {
    if alignment <= 1 {
        return Ok(n);
    }
    n.div_ceil(alignment)
        .checked_mul(alignment)
        .ok_or_else(|| Error::CapacityExceeded(format!("align_up({n}, {alignment}) overflows usize")))
}

#[cfg(test)]
mod tests;
