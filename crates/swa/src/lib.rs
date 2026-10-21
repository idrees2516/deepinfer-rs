//! # deepinfer-swa
//!
//! YOCO decoder layout, sliding-window attention and bounded-replay
//! prefill planning for DeepInfer-rs, mirroring the decoder half of
//! DeepSeek's [DeepInfer] inference framework (upstream README: "YOCO SWA
//! bounded replay: reduce decoder prefill work by replaying only a
//! bounded suffix").
//!
//! YOCO ("You Only Cache Once", Sun et al., arXiv:2405.05254) splits the
//! decoder into self-attention layers — each retaining only a
//! sliding-window KV cache ([`SwaRingCache`], [`swa_attention`]) — and
//! cross-attention layers that read a single shared encoder-side KV cache,
//! so the decoder half of prefill only needs a bounded suffix of the
//! prompt ([`BoundedReplayPlanner`]). The DeepInfer env contract exposes
//! this as `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE` (`full` / `compact`)
//! and `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS` (V4.1 default 140); in
//! compact mode the retained suffix must cover the SWA window plus MTP
//! backtracking headroom, which [`SwaConfig::validate`] enforces.
//!
//! [DeepInfer]: https://github.com/deepseek-ai/DeepInfer

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod attention;
pub mod planner;
pub mod ring;

pub use attention::swa_attention;
pub use deepinfer_core::env::YocoTruncationMode;
pub use planner::{BoundedReplayPlanner, ReplayPlan, SwaConfig, YocoLayout};
pub use ring::SwaRingCache;
