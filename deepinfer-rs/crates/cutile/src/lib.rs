//! Pure-Rust GPU kernels for DeepInfer-rs, built on [CUDA Rust] /
//! [cuTile Rust] (`cutile`).
//!
//! # Why cuTile
//!
//! Every kernel in this crate is authored in Rust and compiled by the cuTile
//! proc-macro into CUDA Tile IR, which NVIDIA's `tileiras` assembler lowers to
//! GPU code for the target architecture at first launch (JIT). There is no
//! C/C++ source, no NVRTC, no `nvcc`: the *entire* inference kernel stack —
//! tensor-core GEMMs, attention, quantization, routing — is Rust that type
//! checks on the host and JITs on the device. This supersedes the
//! NVRTC/hipRTC-based `deepinfer-kernels` crate for NVIDIA targets, which
//! remains available as the AMD MI335X (HIP) fallback.
//!
//! # Operator inventory (DeepSeek V4.1-Flash)
//!
//! | Area | Kernels |
//! |---|---|
//! | Elementwise / norm | `add_bf16`, `silu_mul_bf16`, `sigmoid_mul_f32`, `rms_norm_bf16`, `add_rms_norm_bf16` |
//! | RoPE | `rope_interleaved` (MLA q/k, non-neox pairs), `o_rope_conjugate` (output latents) |
//! | Quantization | `quant_fp8_block128` (1×128 UE8M0, ×4-replicated to 32-lane granularity), `quant_fp8_kv_cache` (per-32), `quant_fp4_kv_cache` (per-16 E4M3) |
//! | GEMM | `bf16_gemm_nt`, `fp8_scaled_gemm_nt` (`mmaf_scaled`), `fp8_grouped_gemm_nt_contiguous` (m_indices), `fp8_grouped_gemm_nt_masked` |
//! | Attention | `mla_prefill` (latent-512 + rope-64 flash, causal, attn-sink, O-RoPE epilogue), `mla_decode_splitkv` (paged, FP8/FP4 caches, split-K + `mla_split_merge`) |
//! | DSA | `thunder_logits` (weighted-ReLU MQA), `dsa_block_preselect`, `dsa_topk_select` (exact (score desc, pos asc) ties), `gather_kv_rows` |
//! | MoE | `route_sigmoid_topk` (noaux-tc), `dispatch_build`, `combine_gather` |
//! | KV cache | `kv_append_bf16`, `kv_append_fp8`, `kv_append_fp4` |
//! | Sampling | `argmax_blocks`, `argmax_final` (ties → lowest index) |
//!
//! # Precision policy
//!
//! * FP8 block-scaled GEMM uses `mmaf_scaled` (Blackwell block-scaled MMA,
//!   E4M3 data + UE8M0 scales). On sm_100/sm_103 (GB200/GB300) this is the
//!   native tcgen05 MXFP8 path — the same numeric contract as DeepGEMM's SM100
//!   recipe. On sm_80/sm_90 the launcher degrades to the BF16 path with an
//!   explicit error unless [`arch::FORCE_BF16_FALLBACK`] is set.
//! * Attention accumulates in f32 with `exp2`-based online softmax; the FP8/FP4
//!   KV caches are dequantized in-register before the MMA.
//!
//! # Safety model
//!
//! Hot kernels use the Tile-IR `unchecked_accesses` waiver (generated as
//! `unsafe fn`); every host entry point in [`ops`] re-validates shapes,
//! strides and divisibility before launching, mirroring the CPU crates'
//! validate-then-compute structure. The crate-level lint is
//! `unsafe_code = "deny"` with per-module `#[allow(unsafe_code)]` and SAFETY
//! proofs, identical to [`deepinfer_gpu`]'s audited-boundary policy. All
//! public APIs are safe to call.
//!
//! # Build / run
//!
//! * Build: CUDA 13.x toolkit headers + libclang (bindgen, build-time only).
//!   `source scripts/gpu-env.sh` sets up a hermetic prefix.
//! * Run: NVIDIA GPU with CUDA 13.3+ driver; sm_80 … sm_121 supported,
//!   tuned for sm_100 (GB200 NVL72) and sm_103 (GB300 / B300 NVL72).
//!
//! [CUDA Rust]: https://github.com/NVIDIA/cuda-rust
//! [cuTile Rust]: https://nvidia.github.io/cuda-rust/cutile/

#![deny(unsafe_op_in_unsafe_fn)]

pub mod arch;
pub mod pack;

#[cfg(feature = "cuda")]
pub mod device;
#[cfg(feature = "cuda")]
#[allow(unsafe_code, missing_docs)]
pub mod kernels;
#[cfg(feature = "cuda")]
// SAFETY: the only `unsafe` blocks in this module call cuTile kernel
// launchers generated with the `unchecked_accesses` Tile-IR waiver; every
// public wrapper re-validates the shapes and divisibility preconditions
// the waiver relies on before launching.
#[allow(unsafe_code)]
pub mod ops;
