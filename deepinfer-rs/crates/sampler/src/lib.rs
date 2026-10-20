//! # deepinfer-sampler
//!
//! Deterministic, batch-invariant token sampling for DeepInfer-rs,
//! mirroring the `SamplingParams` semantics of DeepSeek's
//! [DeepInfer](https://github.com/deepseek-ai/DeepInfer) inference
//! framework (upstream README: "deterministic sampling" +
//! "batch-invariant inference") — vLLM-style `temperature` / `top_p` /
//! `top_k` / `min_p` filters plus a per-sequence `seq_seed` that keys a
//! Philox stream by `(seed, seq_id)`.
//!
//! Pipeline (fixed order, matching the upstream filter chain):
//! `mask -> temperature -> top-k -> softmax -> top-p -> min-p ->
//! renormalize -> categorical(rng)`. Greedy decoding
//! (`temperature <= 0`, or below [`GREEDY_TEMPERATURE_EPS`]) short-circuits
//! to argmax with ties resolved to the lowest token id.
//!
//! ## Determinism and batch invariance
//!
//! A sequence's draws depend only on `(logits, params, seed, seq_id)`:
//! each sequence owns a [`deepinfer_core::rng::Philox`] stream derived by
//! [`Sampler::seq_rng`], so interleaving other sequences in a batch never
//! perturbs it. All filtering uses total-order tie-breaking (probability
//! descending, token id ascending) and sequential `f32` accumulation.
//!
//! ## Numerical hardening
//!
//! Non-finite logits (`±∞`, `NaN`) are treated as `-inf` (never sampled);
//! an all-`-inf` row, an empty logit vector, or a mask that allows no
//! token is an error — never a panic, never a NaN.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod params;
pub mod pipeline;

pub use params::{SamplingParams, SchemaType, StructuredOutputArgs};
pub use pipeline::{Sampler, DEFAULT_SEQ_SEED, GREEDY_TEMPERATURE_EPS};
