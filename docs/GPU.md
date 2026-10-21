# GPU inference layer (deepinfer-gpu + deepinfer-kernels)

This document specifies the GPU stack added to DeepInfer-rs: what it
implements, how it maps onto the three target platforms, its safety and
determinism model, and the exact upgrade paths that remain.

## Target platforms

| Platform | Chip | Arch (JIT target) | Notes |
|---|---|---|---|
| NVIDIA GB200 NVL72 | 72x B200 | `sm_100` (CC 10.0) | CUDA >= 12.4; 12.8+ recommended; NVLink5 1.8 TB/s/GPU (130 TB/s rack bisection); FP8 dense 5 PF, FP4 dense 10 PF per GPU; 8 TB/s HBM3e |
| NVIDIA GB300 NVL72 (Blackwell Ultra) | 72x B300 | `sm_103` (CC 10.3) | CUDA >= 12.9, R580+ driver, open kernel modules; 288 GB HBM3e; 15 PF dense NVFP4; 2x SFU softmax throughput |
| AMD Instinct MI335X (and MI300/MI350 families) | CDNA | `gfx…` from `gcnArchName` (e.g. `gfx942`, `gfx950`) | 288 GB HBM3e, 8 TB/s; ROCm 6/7; arch is detected at runtime from the device itself, so one binary serves whatever Instinct it meets |

The arch string drives NVRTC (`--gpu-architecture=sm_100` / `sm_103`) or
hipRTC (`-D__HIP_PLATFORM_AMD__ --gpu-architecture=gfx942`). When the local
NVRTC is older than the device (e.g. CUDA 12.8 tooling on a B300), the JIT
automatically falls back to `compute_XXX` PTX and lets the driver JIT it.

## Architecture

```
deepinfer-runtime (model loop, scheduler, sampling)      [host, unchanged]
        |
deepinfer-kernels  (typed launchers + embedded CUDA/HIP C++ sources)
        |
deepinfer-gpu      (dlopen + CUDA driver / HIP + NVRTC / hipRTC + JIT cache)
        |
libcuda.so.1 / libnvrtc.so  ·  libamdhip64.so / libhiprtc.so   [runtime only]
```

Design decisions (validated against DeepGEMM's DeepJIT, the `deepgemm-rs`
Rust port, and the `kern` runtime, which all use the same model):

1. **Zero build-time CUDA/HIP.** All vendor libraries are `dlopen`ed at
   runtime; kernels are compiled from embedded C++ strings by NVRTC /
   hipRTC on first use and cached on disk keyed by
   `SHA-256(backend, arch, name, options, source)`. Binaries build and test
   on machines with no GPU stack (this repository's CI plane is exactly
   that). Cache/env knobs mirror DeepGEMM's `DG_JIT_*`:
   `DEEPINFER_JIT_CACHE=0` disables, `DEEPINFER_JIT_DEBUG=1` traces,
   `DEEPINFER_CACHE_DIR` overrides the cache root
   (`~/.cache/deepinfer-rs/jit` by default).
2. **One source, two compilers.** `crates/kernels/kernels/*.cu` compiles
   unchanged under NVRTC and hipRTC; `#if defined(__HIP__)` selects the
   vendor-specific core (warp size 64 vs 32, MMA path). All kernels use 1-D
   blocks and warp-size-parametric reductions.
3. **Golden-tested against the CPU reference.** The CPU crates from the
   reference implementation are the numerical oracle: integration tests in
   `crates/kernels/tests/golden.rs` run both sides on identical inputs
   whenever a device is present and skip (with a note) otherwise. Bit-exact
   expectations are used where the math is elementwise (codecs, RoPE,
   integer selections); tolerances where reduction order legitimately
   differs (GEMM trees, softmax merges).

## What is implemented (operator inventory)

| Operator | Kernel(s) | CPU golden |
|---|---|---|
| RMSNorm | `di_rmsnorm_kernel` | `core` norm semantics |
| Interleaved RoPE + O-RoPE conjugate | `di_rope_kernel` (host-built cos/sin table; bit-exact) | `core::rope` |
| SiLU-mul, residual add, f32<->bf16 | `di_silu_mul`, `di_add`, casts | `core::format` |
| Activation quant (1x128 E4M3+UE8M0) | `di_quant_per_token_group_kernel` | `deepinfer-fp8` |
| KV-cache encode, FP8 528 B / FP4 288 B | `di_kv_encode_fp8/fp4_kernel` | `deepinfer-fp8` (FlashMLA V4.1 layouts) |
| BF16 NT GEMM | `di_gemm_bf16_nt_kernel` | `deepinfer-gemm` |
| FP8 block-scaled NT GEMM (dense) | `di_gemm_fp8_nt_kernel` | `deepinfer-gemm::fp8_gemm_nt` |
| Grouped-contiguous FP8 GEMM (MoE) | `di_gemm_fp8_grouped_nt_kernel` | `m_grouped_..._contiguous` |
| MLA decode, split-KV + combine, all 3 cache formats, attn_sink | `di_mla_decode_partial/combine_kernel` | `deepinfer-mla::flash_mla_decode` |
| MLA prefill (dense causal / sparse) | `di_mla_prefill_kernel` | `flash_mla_dense/sparse_prefill` |
| DSA: per-head RMSNorm, lightning quant, thunder logits | `di_indexer_rmsnorm_per_head`, `di_lightning_*`, `di_thunder_logits_kernel` | `deepinfer-dsa::thunder_scores` |
| DSA: cascade conv, gated norm, conv scores | `di_conv_*`, `di_dot_scores_kernel` | `conv_scores` |
| DSA: gate + blend, causal clip | `di_qmean`, `di_gate_*`, `di_causal_clip` | `gate_and_combine` |
| DSA: block top-K selection | `di_block_preselect`, `di_topk_merge/scatter_kernel` (packed-key bitonic; exact tie-break) | `select_top_k` (exact integer match) |
| MoE routing (sigmoid + noaux-tc) | `di_moe_route_kernel` | `deepinfer-moe::route_batch` (exact idx match) |
| MoE dispatch/combine (aligned segments) | `di_moe_count/prefix/scatter/combine_kernel` | `deepinfer-ep` layout contract |
| Peer copy / multi-GPU | `Device::enable_peer_access`, `memcpy_peer_async` | NVL72 / Infinity Fabric EP primitives |

## Precision and determinism policy

* **Codecs are bit-exact ports.** e4m3/e5m2/e8m0/e2m1/bf16 conversions in
  `common.cuh` are transcriptions of `deepinfer-core::format` (same RNE,
  saturation and NaN policies), so quantized payloads are byte-identical to
  the CPU's (the only documented edge: the UE8M0 ceil uses device `f64`
  `log2`; its 1e-9 exact-power-of-two guard absorbs ULP differences, and
  the golden test compares dequantized values).
* **Determinism.** Fixed launch geometry + fixed reduction trees + fixed
  accumulation order (heads/dims ascending, splits ascending, k ascending)
  mean the same binary and shapes give identical results run-to-run on the
  same device. GPU results are *not* bit-identical to the CPU reference
  (different association order) — the goldens use per-operator tolerances.
* **MoE dispatch atomics.** Expert ranks come from atomic counters, so the
  *permuted row order* can vary; per-token outputs do not depend on it
  (each token's row is computed independently and combined in fixed k
  order), which preserves end-to-end determinism.
* **FP8 GEMM semantics** follow DeepGEMM SM90: per-128-k-block rescale,
  f32 accumulation between blocks, activation scales 1x128, weight scales
  128x128, UE8M0. (SM100's packed-UE8M0/TMA layout requirement applies only
  to the tcgen05 path — see upgrades.)

## Safety model

The workspace keeps `#![forbid(unsafe_code)]` everywhere except
`deepinfer-gpu`, which uses `deny` + module-level `#[allow(unsafe_code)]`
(`forbid` cannot be overridden). The audited surface is exactly:

* `dl.rs` — dlopen/dlsym wrapper. Libraries are never unloaded (eliminating
  use-after-dlclose); handles are Send/Sync; symbol resolution copies the
  dlsym-returned address into pointer-sized types only.
* `ffi/{cuda,nvrtc,hip,hiprtc}.rs` — declarations transcribed from the
  vendor headers; every field is a typed function pointer.
* `device.rs` / `jit.rs` — call sites, each carrying a SAFETY comment.

No vendor library is a *build* dependency and no C toolchain is required:
the supply-chain surface remains the Rust workspace itself.

## Performance profile and upgrade paths (honest engineering cuts)

The v1 kernels favor *verifiable correctness* over peak throughput. Every
cut below is isolated in one place and has a concrete upgrade:

1. **NVIDIA GEMM runs bf16 `mma.sync.m16n8k16` tensor cores** (fragment
   mapping per the PTX ISA tables); FP8 operands are decoded to bf16 during
   the global->shared staging, so memory traffic stays FP8 while compute
   runs at bf16 tensor rate (2.5 PF/GPU on B200 vs 5 PF FP8 peak).
   Upgrades: (a) e4m3 `mma.sync.m16n8k32` fragments — one function;
   (b) tcgen05 + TMA + UE8M0 packed scales on `sm_100a`/`sm_103a` for full
   DeepGEMM SM100 parity. The fragment mapping shipped here is validated by
   the GEMM goldens (any mapping error produces loud garbage, not silent
   drift).
2. **AMD GEMM runs LDS-tiled scalar f32 FMA** — correct and
   bandwidth-efficient (decode GEMMs are bandwidth-bound at 8 TB/s), but
   prefill leaves MFMA throughput unused. The MFMA upgrade requires the
   exact per-lane register layouts (`v_mfma_f32_16x16x16_bf16` and the
   gfx942 FP8 variants `v_mfma_f32_16x16x128_fp8_fp8`) to be validated on
   hardware; we deliberately do not ship unverifiable ISA asm. On-device
   validation is a half-day with the golden harness once an Instinct is
   available.
3. **Attention / DSA / MoE kernels are bandwidth-oriented scalar-warp
   designs** (warp-per-(head,query) online softmax; serial-per-thread
   dots). They match the algorithmic structure of FlashMLA / DeepGEMM's
   MQA-logits (split-KV, online rescale, deterministic merges) without
   claiming FlashMLA's occupancy tuning. TMA/tcgen05-style SMEM pipelines
   are the follow-up.
4. **NVLink / Infinity Fabric EP** exposes P2P copy primitives
   (`enable_peer_access`, `memcpy_peer_async`). Production NVL72 serving
   would layer NVSHMEM-style symmetric buffers (DeepEP / DeepGEMM Mega-MoE
   style) on top; the Rust primitives here are the substrate.

## Model-loop integration blueprint (next mechanical step)

The operator set above is complete for the V4.1-Flash forward pass. Wiring
it into `deepinfer-runtime` as a `DeviceModel` is mechanical:

1. Load weights (safetensors, unchanged), quantize on host with
   `deepinfer-fp8` (weights 128x128, per-layer), upload bytes + scales.
2. Per decode step, per layer (all on-device, one stream):
   `rmsnorm` -> `quant_per_token_group` -> `gemm_fp8_nt` (q_a|kv_a fused
   rows) -> `rmsnorm` (q slice) -> `gemm_fp8_nt` (q_b) -> `rope` (q/k
   slices, host cos/sin table uploaded once) -> `kv_encode_*` append ->
   DSA layers: `gemm_fp8_nt` (indexer wq/wk) ->
   `indexer_rmsnorm_per_head` -> `lightning_quant` -> `thunder_logits` ->
   `conv_feat` (new rows only) -> `conv_gated_norm` -> `dot_scores` ->
   `qmean`/`gate_logits`/`gate_blend` -> `causal_clip` -> `select_top_k` ->
   `mla_decode_partial` + `mla_decode_combine` -> `rope` (conjugate) ->
   `gemm_fp8_nt` (wv|wo) -> `add` -> `rmsnorm` -> `gemm_bf16_nt` (router)
   -> `moe_route` -> `moe_dispatch` -> `gemm_fp8_grouped_nt` (gate_up) ->
   `silu_mul` -> `gemm_fp8_grouped_nt` (down) -> `moe_combine` -> `add`.
3. Head GEMM + D2H of the final logits row; sampling stays on host
   (`deepinfer-sampler` is latency-trivial and batch-invariant by design).
4. Multi-GPU: expert-parallel placement with `deepinfer-eplb` maps,
   `memcpy_peer_async` for dispatch/combine (single-process simulation
   first, matching the CPU EP crate's model).

## Running on hardware

```bash
cargo test -p deepinfer-kernels --test golden   # CPU-vs-GPU goldens
DEEPINFER_JIT_DEBUG=1 cargo test ...            # trace JIT compiles
```

Requirements: NVIDIA driver >= 550 (R580+ for B300) with `libnvrtc`
(CUDA 12.4+; 12.9+ for `sm_103` cubins), or ROCm 6/7 with `libamdhip64` +
`libhiprtc`. Nothing needs to be installed to *build*.
