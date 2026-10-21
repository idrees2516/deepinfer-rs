//! # deepinfer-core
//!
//! Foundation crate for **DeepInfer-rs**, a pure-Rust, CPU-reference
//! implementation of DeepSeek's [DeepInfer](https://github.com/deepseek-ai/DeepInfer)
//! large-scale inference framework, following the DeepSeek-V4.1-Flash
//! architecture (DSA sparse attention + FlashMLA-style latent attention +
//! YOCO/SWA decoders + DeepGEMM FP8 recipes + DeepEP expert parallelism +
//! EPLB + DSpark/MTP speculative decoding).
//!
//! This crate provides the shared primitives every other `deepinfer-*`
//! crate builds on:
//!
//! * [`DType`] / [`Tensor`] — validated, dtype-tagged dense tensors.
//! * [`format`] — bit-exact BF16/F16/E4M3/E5M2/E8M0/E2M1 codecs with
//!   round-to-nearest-even semantics matching the OCP FP8 / MX specs used
//!   by DeepGEMM and FlashMLA.
//! * [`rng::Philox`] — Philox4x32-10 counter-based RNG (deterministic,
//!   platform-independent — the basis of deterministic sampling).
//! * [`math`] — stable softmax, log-sum-exp, deterministic top-k and
//!   categorical sampling.
//! * [`rope::Rope`] — interleaved / half-split RoPE with YaRN scaling and
//!   attention-conjugate rotation (O-RoPE, as fused into FlashMLA's V4.1
//!   kernel).
//! * [`norm`] — RMSNorm.
//! * [`mask::TokenMask`] — vocabulary bitsets for structured outputs.
//! * [`json::Json`] — hardened, allocation-bounded JSON parser.
//! * [`env::Env`] — the `DEEPINFER_*` environment contract.
//! * [`safetensors`] — fully validated checkpoint readers.
//! * [`tokenizer`] — byte-fallback and GPT-2-style BPE tokenizers.
//! * [`sync`] — thread pool with deterministic disjoint-range parallelism.
//! * [`prop`] — dependency-free property-testing harness.
//!
//! ## Security posture
//!
//! * `#![forbid(unsafe_code)]` — zero unsafe blocks, workspace-wide.
//! * Zero external dependencies (std only): no supply-chain surface.
//! * All parsing (JSON, safetensors, tokenizer, env) is bounds-checked,
//!   depth-limited and panic-free; malformed input yields `Err`, never
//!   undefined behaviour.
//! * Determinism contracts (fixed accumulation order, total-order
//!   tie-breaking) make outputs reproducible and auditable.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod dtype;
pub mod env;
pub mod error;
pub mod format;
pub mod json;
pub mod mask;
pub mod math;
pub mod norm;
pub mod prop;
pub mod rng;
pub mod rope;
pub mod safetensors;
pub mod sync;
pub mod tensor;
pub mod tokenizer;

// Crate-root re-exports so `crate::Error` / `crate::ensure!` resolve
// everywhere (including the `ensure!` macro body).
pub use error::{Error, Result};

/// Commonly used items.
pub mod prelude {
    pub use crate::dtype::DType;
    pub use crate::env::{Accelerator, Env, YocoTruncationMode};
    pub use crate::error::{Error, Result};
    pub use crate::json::Json;
    pub use crate::mask::TokenMask;
    pub use crate::math::{
        argmax, dot, logsumexp, sample_categorical, softmax_inplace, top_k_desc, TopK,
    };
    pub use crate::prop::{assert_close, assert_close_slice, prop};
    pub use crate::rng::Philox;
    pub use crate::rope::{Rope, RopeStyle, YarnConfig};
    pub use crate::tensor::Tensor;
    pub use crate::tokenizer::{ByteTokenizer, Tokenizer};
}
