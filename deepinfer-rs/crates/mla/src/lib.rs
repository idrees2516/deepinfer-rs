//! # deepinfer-mla
//!
//! FlashMLA-equivalent Multi-head Latent Attention (MLA) for
//! DeepSeek-V4.1-Flash, following DeepSeek's FlashMLA reference
//! implementation (github.com/deepseek-ai/FlashMLA): absorbed MQA over a
//! compressed 512-dim KV row (448 latent + 64 interleaved rope dims), a
//! paged KV cache with 64-token pages and Bf16 / FP8 (528 B) / FP4
//! (288 B) V4.1 row formats, sparse decode with FlashMLA's split-K
//! online-softmax structure, an exact transcription of the README's
//! sparse-prefill pseudocode (DSA top-k selection, `topk_length`,
//! `attn_sink`), causal dense prefill, the O-RoPE output conjugate, and
//! the fused e4m3+e8m0 output cast consumed by the DeepGEMM-style
//! epilogue einsum.
//!
//! Fidelity notes:
//! * [`flash_mla_decode`] processes the gathered KV rows in 64-row chunks
//!   (the FlashMLA page size) with a running `(m, l, o)` state merged
//!   across chunks exactly like flash-decoding split-K.
//! * [`flash_mla_sparse_prefill`] reproduces the README pseudocode
//!   literally, including the empty-selection edge case
//!   (`max_logits = -inf`, `lse = +inf`, zero output) and the `attn_sink`
//!   epilogue `out *= exp(lse) / (exp(lse) + exp(sink))` (computed in a
//!   numerically stable form; the returned logits are unchanged).
//! * The V4.1 FP8/FP4 KV-cache codecs currently live in the private
//!   `fp8cache` module as byte-identical mirrors of the `deepinfer-fp8`
//!   crate's specified API (that crate is still a placeholder); see the
//!   worklog.
//! * All kernels are sequential with fixed accumulation order, so outputs
//!   are bit-identical across runs (determinism contract).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod cache;
mod config;
mod decode;
mod fp8cache;
mod fused;
mod orope;
mod prefill;
mod weights;

#[cfg(test)]
mod testutil;

pub use cache::MlaPagedCache;
pub use config::{CacheFormat, MlaConfig};
pub use decode::flash_mla_decode;
pub use fused::fused_attention_out_fp8;
pub use orope::o_rope_conjugate;
pub use prefill::{flash_mla_dense_prefill, flash_mla_sparse_prefill};
pub use weights::MlaWeights;
