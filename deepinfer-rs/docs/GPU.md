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
deepinfer-runtime  (DeepSeekModel = CPU golden · DeviceModel = GPU forward,
                    decode graphs, sampling, scheduler)        [host]
        |
deepinfer-kernels  (typed launchers + embedded CUDA/HIP C++ sources:
                    GEMMs incl. the tcgen05/TMA SM100 path, attention, DSA,
                    MoE, the _dev decode family, the paged serve family,
                    stream-copy peer EP + symmetric-buffer LL EP)
        |
deepinfer-gpu      (dlopen + CUDA driver / HIP + NVRTC / hipRTC + JIT cache
                    + events + stream-capture graphs + peer access)
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
| F32 reference NT GEMM | `di_gemm_f32_nt_kernel` (scalar core, both vendors) | `deepinfer-gemm::f32_gemm_nt` association order |
| Masked grouped GEMM (graph MoE / low-latency EP) | `di_gemm_grouped_masked_nt_kernel` (f32/bf16/fp8, `masked_m` on device) | DeepGEMM `m_grouped_nt_masked` semantics |
| **Dedicated FP4-cache decode** | `di_mla_decode_fp4_partial_kernel` (scale-decode per 16-dim group, unrolled nibble unpack, ILP=2 — bit-identical to the generic fmt-2 path) | `flash_mla_decode` over an FP4 cache |
| Device-scalar dense prefill | `di_mla_prefill_dense_dev_kernel` (q_offset / nkeys from device memory) | moving cross-attention windows in graphs |
| Sliding-window attention (D = hidden) | `di_swa_attention_dev_kernel` (CTA-per-query, shared-memory accumulator) | `deepinfer_swa::swa_attention` |
| Ring / window gather | `di_gather_rows_kernel` (wrap-aware, device-read start/count) | cross-attn window staging in graphs |
| Graph-replayable decode family | `di_step_params`, `di_pos_fill`, `di_kv_encode_*_dev`, `di_append_*`, `di_lightning_*_dev`, `di_thunder_logits_dev`, `di_dot_scores_dev`, `di_gate_blend_dev`, `di_causal_window_clip_dev`, `di_block_preselect_dev`, `di_conv_*_dev` | the CPU crates' semantics with every varying scalar device-resident |
| Masked MoE dispatch / combine | `di_moe_scatter_masked_kernel`, `di_silu_mul_masked_kernel` (fused gate/up layout) | DeepEP low-latency capacity semantics |
| Indexed row gather | `di_gather_rows_indexed_kernel` | embedding lookup, EP send packing |
| **tcgen05 FP8 GEMM (SM100)** | `di_gemm_fp8_nt_sm100a_kernel` — `tcgen05.mma.kind::mxf8f6f4.block_scale` with UE8M0 scale factors in Tensor Memory, `cp.async.bulk.tensor` TMA loads (SWIZZLE_128B), warp-specialized 3-stage pipeline (TMA / MMA / SF-loader / Layout-D epilogue), `sm_100a`/`sm_103a` targets | DeepGEMM SM100 `fp8_fp4_gemm_1d1d` semantics (same 1x128 / 128x128 ue8m0 recipe) |
| **Paged multi-sequence serving** | the `di_serve_*` family: fanout cells, paged kv encode / row appends / per-head norm, cascade conv, lightning quant, thunder + conv scores, gate blend, causal clip, block preselect + top-k, split-KV decode partial/combine, sparse + dense-causal prefill, SWA rings — every row addressed through `tbl[r/64]*64 + r%64` page tables, one launch per op per batch | the arena kernels they port (all CPU-validated) |
| AMD MFMA FP8 GEMM (opt-in) | `di_gemm_fp8_nt_mfma_kernel` (`v_mfma_f32_32x32x16_fp8_fp8`, gfx940-class; `DEEPINFER_AMD_MFMA=1`) | same DeepGEMM recipe; validated via the golden harness |
| **Symmetric-buffer EP (NVSHMEM-style)** | `di_ep_{reset,dispatch,dispatch_done,group,combine,reduce}_ll_kernel` — device-initiated one-sided writes into peer symmetric heaps over NVLink with `atom/red.release.sys` + `ld.acquire.sys` progress counters; no host readback, no library | DeepEP low-latency dispatch/combine protocol |

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

1. **NVIDIA GEMM below compute capability 10.0 runs bf16
   `mma.sync.m16n8k16` tensor cores** (fragment mapping per the PTX ISA
   tables); FP8 operands are decoded to bf16 during the global->shared
   staging, so memory traffic stays FP8 while compute runs at bf16 tensor
   rate. On Blackwell (`sm_100a`/`sm_103a`) the **tcgen05 path takes over
   automatically** (see the inventory row above): e4m3 operands feed
   `tcgen05.mma.kind::mxf8f6f4.block_scale` directly from 128B-swizzled
   shared memory (TMA-loaded), the UE8M0 scales ride Tensor Memory as
   packed x4 words (the DeepGEMM UTCCP-transposer pattern with
   `tcgen05.cp.32x128b.warpx4`), and the f32 accumulator never leaves
   Tensor Memory until the Layout-D epilogue. `DeviceModel` picks it
   transparently (`DEEPINFER_TCGEN05=0` pins the portable kernel); the
   first driver/JIT rejection falls back stickily, so mixed fleets keep
   running. Known follow-ups (tracked): 2-CTA cluster MMA (cta_group::2,
   M=256), persistent tile scheduling, and swap-AB for decode-shaped GEMMs
   (M=1 wastes 127 accumulator lanes today — bandwidth-bound, so the cost
   is compute-idle, not traffic).
2. **AMD GEMM defaults to the LDS-tiled scalar f32 FMA core** — correct
   and bandwidth-efficient (decode GEMMs are bandwidth-bound at 8 TB/s).
   The **MFMA FP8 path is implemented and opt-in**
   (`DEEPINFER_AMD_MFMA=1`): `di_gemm_fp8_nt_mfma_kernel` drives
   `v_mfma_f32_32x32x16_fp8_fp8` (CDNA3+, compiled for gfx940-class
   targets) with the fragment mappings transcribed from AMD's matrix-core
   documentation — A: 8 contiguous k-bytes of row `lane%32`, k-block
   `8*(lane/32)`; B: `B[k + 8*(lane/32)][lane%32]`; C: rows
   `8i + j + 4*(lane/32)`, col `lane%32` — and the same per-128-k ue8m0
   rescale as the scalar core. Because we cannot execute CDNA asm on the
   CI plane, the path ships **disabled by default**: enable it, then run
   `amd_mfma_matches_scalar` or one `golden_run --record`/`--verify`
   round on the Instinct — the goldens compare against the CPU reference,
   so a wrong lane mapping fails loudly, not silently. Remaining MFMA
   follow-ups: the bf16 16x16x16 variant (trivial once the FP8 path is
   validated) and LDS-bank-conflict tuning of the fragment loads.
3. **Attention / DSA / MoE kernels are bandwidth-oriented scalar-warp
   designs** (warp-per-(head,query) online softmax; serial-per-thread
   dots). They match the algorithmic structure of FlashMLA / DeepGEMM's
   MQA-logits (split-KV, online rescale, deterministic merges) without
   claiming FlashMLA's occupancy tuning. TMA/tcgen05-style SMEM pipelines
   are the follow-up.
4. **NVLink / Infinity Fabric EP** ships the single-process peer
   collectives (see "NVLink peer collectives for EP" above): send plans,
   packed peer copies with event-mesh ordering, and combine-back through
   the standard kernel. The remaining upgrade is NVSHMEM-style symmetric
   buffers with sender-side writes directly into the aligned layout
   (DeepEP / DeepGEMM Mega-MoE style), which removes the receive-side
   repack and the per-step `topk_idx` read-back.

## The DeviceModel (runtime integration — implemented)

`deepinfer-runtime::DeviceModel` wires the operator layer above into the
model loop per the original blueprint, plus the three tracked follow-ups:

1. **Weights**: loaded once (safetensors, unchanged loader), quantized on
   the host with `deepinfer-fp8` (weights 128x128, per expert for grouped
   matrices) and uploaded. [`Precision::F32`] keeps every GEMM in f32 (the
   CPU-golden test mode); [`Precision::Auto`] selects FP8 whenever
   `k % 128 == 0` (all GEMMs at V4.1 dims) and BF16 otherwise. The
   `q_a|kv_a` projection is fused into one GEMM (rows `[q_lora | kv_a]`).
2. **Caches**: two coexisting engines.
   *Single-sequence* (the CUDA-graph decode path): per self-decoder
   layer, flat append-only arenas — MLA KV rows in the config's cache
   format (f32 / 528 B FP8 / 288 B FP4), the DSA indexer's raw k rows
   (conv branch), the per-head-normed k rows with their block-quantized
   FP8 copy (thunder branch, quantized incrementally as blocks complete),
   and the cascade conv features. Rollback is exact: rows are
   position-keyed, so truncating the logical length leaves surviving rows
   intact and re-appended rows overwrite their slots (bit-identical to
   never having appended — tested).
   *Multi-sequence* (the paged serve engine, below): the same five
   per-layer caches become **pools of 64-token pages** (the DSA
   sparse-index page granularity) with per-sequence device page tables;
   sequences attach/prefill/decode/rollback/detach independently through
   `serve_attach` / `serve_prefill` / `serve_decode(batch)` /
   `serve_rollback` / `serve_detach`, sharing one free list.
3. **Prefill** (multi-token, eager kernels): rmsnorm -> fused q_a|kv_a ->
   latent norm -> RoPE -> cache encode -> DSA (projections from the raw
   residual, per-head norm, lightning quant, thunder + conv + gating +
   selection) -> sparse MLA prefill -> O-RoPE -> wv/wo -> MoE
   (aligned-contiguous dispatch, `MOE_CHUNK_TOKENS`-chunked, shared
   experts dense) -> residual. Cross layers add the SWA ring (device
   rings, wrapped appends) and dense cross-attention over the last self
   layer's KV window.
4. **Decode** (the uniform `_dev` path): one host upload per step
   (the embedded row + a `{pos, start}` i32 pair); everything else is
   kernel launches reading the device-resident parameter block — so the
   identical function serves eager execution and graph capture.
5. **Sampling** stays host-side (`deepinfer-sampler`); `generate()` is a
   thin prefill/decode loop. MTP drafting stays host-side; verification
   re-enters through the multi-token prefill path.

## CUDA-graph decode capture (implemented)

`DeviceModel::enable_decode_graph(true)` captures the whole decode step
into a CUDA/HIP graph on the first decode of a sequence and replays it
for every subsequent token — one launch instead of hundreds. The design
rule that makes this work: **no scalar that changes between steps may be
a kernel argument**. The `decode.cu` family reads the sequence length,
cache write rows, quant-block completion, attention window starts and
ring slots from a device parameter cell that `di_step_params_kernel`
fans out from the single host-written `{pos, start}` pair; all launch
grids derive from fixed maxima. The MoE runs the masked capacity layout
(`masked_m` on device, capacity 1 per expert — a token's top-k experts
are distinct, so nothing drops), which needs no host readback. Graphs are
per-sequence (cache-window pointers are baked at capture) and survive
rollback (the graph re-reads the position cell every replay). Replay is
**bit-identical** to the eager launches — same kernels, same grids, same
device cells — and the test suite asserts exactly that. The substrate
lives in `deepinfer-gpu::sync` (`CaptureGuard` / `GraphExec`, RELAXED
capture mode, `cuGraphInstantiateWithFlags` / `hipGraphInstantiateWithFlags`,
optionally resolved so older drivers still load).

## NVLink peer collectives for EP (implemented)

`deepinfer-kernels::peer` provides the single-process multi-GPU expert
parallelism layer for the NVL72 / Infinity Fabric domain:

* `PeerMesh::new` enables all-pairs peer access and owns one EP stream +
  barrier event per rank;
* `build_send_plan` (pure, host-side, unit-tested) computes each rank's
  sends in the DeepEP-documented `(destination, token asc, k asc)` order
  with block-contiguous expert ownership `owner(e) = e * world / E`;
* senders pack per-destination buffers with the indexed row gather
  (device-resident index lists);
* `all_to_all_dispatch_fp8` moves the FP8 payloads + scales with one
  contiguous peer copy per (destination, sender) pair, ordered by the
  event mesh and closed by a full barrier;
* `all_to_all_combine_f32` reverses it, feeding the standard
  `moe_combine` (fixed k-ascending accumulation).

The receive side keeps the deterministic `(sender, token, k)` row order
with per-row expert metadata; the stream-copy path remains the fallback
for stacks without peer atomics.

**Symmetric-buffer EP (implemented).** `deepinfer-kernels::peer_ll`
rebuilds the NVSHMEM model on raw driver primitives: every rank owns one
`SymmetricHeap` with an identical layout (dispatch channels keyed by
routing slot, per-home combine slots, progress counters), kernels write
peers' heaps directly over NVLink (in-process peer pointers, or
IPC-imported pointers through the file-rendezvous `Session` for
multi-process serving), and correctness rides system-scope
acquire/release atomics (`red.release.sys` after each payload write,
`ld.acquire.sys` spins on the receivers). The step protocol —
reset (event-ordered) -> dispatch (remote slot reservation + payload +
release-publish) -> dispatch_done (per-rank barrier) -> group (receiver
scatters into the masked per-expert layout after the device barrier) ->
[caller's expert GEMMs] -> combine (owner writes home combine slots) ->
reduce (fixed k-ascending weighted sum) — has **no host readback and no
per-step allocation**, exactly the DeepEP low-latency contract. Channel
row order varies run-to-run (reservation races) but per-token outputs are
deterministic; capacity drops are counted (`drops`) for the caller's
backpressure.

## Paged device cache and multi-sequence serving (implemented)

`DeviceModel` grows a second cache engine alongside the flat arenas:

* `serve_configure(max_seqs, max_len, pages_per_layer)` allocates, per
  self-decoder layer, five paged pools (kv / k_raw / k_norm / k_u8 /
  feat) of `pages_per_layer` x 64 rows plus the batch-sized serve
  workspace (activations for `max_seqs` decode rows, score matrices for
  the prefill chunk, split-KV partials, masked-MoE staging).
* Page ids are global: one shared free list, one physical id addressing
  the same logical page in every layer's pools (identical geometry).
* One batched decode step = **one launch per operator per layer** across
  all sequences: the `di_serve_*` kernels take `grid.z = slots` (or fold
  the slot into `blockIdx`), read the per-slot geometry from a device
  cell block that `di_serve_fanout_kernel` expands from a single small
  host upload (`{active, pos, len, sq, sqnb}` per slot), and translate
  every row through `tbl[r / 64] * 64 + r % 64`. No varying scalar is a
  kernel argument — the same graph-safety rule as the decode family.
* Prefill runs through the same kernels with one slot and `sq = chunk`
  (512-token chunks, aligned-contiguous MoE); decode runs with
  `sq = 1` per slot and the masked-capacity MoE (`cap = batch`), both
  graph-safe.
* Rollback truncates the logical length; page tables keep their entries
  and re-appended rows overwrite their slots (the arena semantics,
  position-keyed). Detach returns pages to the free list.
* The single-sequence engine (arenas + CUDA-graph decode capture) is
  untouched and shares weights, the JIT library, and the eager kernels
  with the serve engine.
* **DSpark speculative serving** (host-drafted, device-verified): the
  serve engine exposes `serve_verify` (one chunked multi-row forward
  per draft chain — the same `di_serve_*` family with `sq = chain`),
  `serve_decode_out` / `serve_prefill_out` (logits **plus** the
  residual hidden anchor readback that feeds the host-side MTP draft
  step — `DeviceModel::draft_step` reads one embedding row back per
  draft, the only D2H traffic), and `serve_rollback`. The server's
  engine thread drafts on the host, verifies on the device, rolls
  rejected tails back (logical-length truncation, O(1)) and batches
  every round's bonus decode into one `serve_decode` launch per step.

Golden: `serve_engine_multi_sequence` (auto-skipped without a GPU)
compares two interleaved sequences against the single-sequence device
path — itself CPU-validated — through prefill, four interleaved decode
steps, an exact rollback, and a detach/re-attach cycle;
`serve_spec_decode_surface` exercises the spec path (draft → chunked
verify vs sequential decodes → rollback exactness).

## On-hardware golden runs (the acceptance battery)

`crates/runtime/examples/golden_run.rs` is the machine-facing acceptance
harness:

```bash
cargo run --release -p deepinfer-runtime --example golden_run -- --list
cargo run --release -p deepinfer-runtime --example golden_run -- --record golden-<site>.json
cargo run --release -p deepinfer-runtime --example golden_run -- --verify golden-<site>.json
```

`--record` writes a JSON artifact with the capability report (backend,
device, arch, SM count, memory, NVRTC version, tcgen05-family and graph
flags) plus one entry per check: the F32 model vs the CPU golden, the FP8
Auto smoke, CUDA-graph replay determinism and rollback exactness (SHA
hashes of the logits), the tcgen05-vs-portable GEMM agreement (Blackwell
only), the paged serve engine vs the single-sequence path, the
symmetric-buffer EP protocol (world 1, and multi-rank when >= 2 devices
are visible), and decode/prefill timings. `--verify` re-runs the battery
against a recorded artifact (same machine/driver) and fails on any
regression. Without a GPU the harness prints the capability report and
exits cleanly — the auto-skip contract of the test suite.

## Serving on hardware

The `deepinfer-server` binary (`crates/server`) exposes the model over
an OpenAI-compatible HTTP API (SSE streaming, continuous batching,
grammar-constrained decoding, **DSpark speculative decoding**,
Prometheus metrics — zero external
dependencies). `--gpu` selects the paged serve engine through
`GpuTokenEngine`: `prepare` sizes the page pools from the scheduler's
batch width (`DEEPINFER_SERVE_PAGES` overrides), `attach`/`detach` map
onto `serve_attach`/`serve_detach`, one batched decode step drives the
`serve_decode_out` batch flush (plain decodes and spec bonus tokens
share a single launch), and speculative rounds run
`serve_verify` + `serve_rollback` per request with host-side MTP
drafting (`DEEPINFER_SERVER_SPEC_DRAFT_LEN`, default 5). See
`docs/UZU-COMPARISON.md` and the `deepinfer-server` crate docs.

## Running on hardware

```bash
cargo test -p deepinfer-kernels --test golden   # CPU-vs-GPU goldens
DEEPINFER_JIT_DEBUG=1 cargo test ...            # trace JIT compiles
```

Requirements: NVIDIA driver >= 550 (R580+ for B300) with `libnvrtc`
(CUDA 12.4+; 12.9+ for `sm_103` cubins), or ROCm 6/7 with `libamdhip64` +
`libhiprtc`. Nothing needs to be installed to *build*.

## GPU prefix caching (implemented)

The paged serve engine grows a **block-granular radix tree over the
shared page pools** — the vLLM automatic-prefix-caching / SGLang
RadixAttention analogue for this engine (`runtime/src/prefix_cache.rs`
+ the serve.rs integration):

* **Reference-counted page sharing.** Every page carries a lease count
  (`PagedState::refs`); a page is in the free list iff its count is 0.
  Lease holders are attached sequences *and* radix-tree nodes, so a
  page referenced by a live sequence and the tree is freed only when
  both release it — the exact upgrade the shared-free-list layout was
  designed for. `serve_detach` drops leases instead of blindly freeing.
* **One tree node per 64-token block**, keyed by a chained FNV-1a hash
  of the token blocks (the vLLM hash-chain scheme) and verified against
  the exact block tokens on every walk (hash collisions cannot alias
  two prompts silently). A node holds its block's leased page, the
  `k_sf` lightning-scale bytes that *complete* inside its block (a byte
  belongs to the block where its `quant_block`-row window closes, so
  concatenating a path's segments reproduces the scale prefix exactly),
  and a device snapshot of the SWA rings at the block's end — the only
  state the pools do not hold.
* **Restore** (inside `serve_prefill`/`serve_prefill_out` on a fresh
  sequence): match the deepest ring-bearing path, adopt its pages into
  the new sequence's page table (leases bumped, one table upload),
  write the scale prefix, and device-to-device copy the ring snapshot —
  pointer writes and D2D copies replacing a compute-bound prefill. The
  restored state is bit-identical to a cold prefill at the same
  boundary (the golden test asserts the logits agree).
* **Insert** happens after every committed prefill chunk and decode
  step — never after speculative-verify chunks (their rows are
  reversible). Blocks another sequence already cached are skipped.
  Ring-allocation failure degrades the node to a walk-through (not a
  match endpoint) instead of failing the step.
* **Copy-on-write.** Tree-leased pages are frozen: a rollback that
  re-enters the frozen region (spec rejection rewinding deep, or the
  scheduler's prefill rewind) triggers full-page D2D copies into fresh
  private pages before any rewrite, so snapshots never see overwritten
  content. Identical rewrites (the rewind re-prefilling the same
  prompt) are detected against the cached tokens and skip the copy.
* **Eviction.** LRU over leaves, bounded by a token budget converted to
  page-units (`serve_prefix_configure`; the cost counts the ring
  snapshots honestly at `ring_bytes / page_bytes`). The allocation path
  runs a pressure valve: when the free list runs dry, tree leaves are
  evicted before a grant fails — live sequences always outrank cached
  prefixes.
* **Surface.** `serve_prefix_configure` / `serve_prefix_stats` /
  `serve_prefix_reset`; the server wires its `prefix_cache_tokens`
  budget through `GpuTokenEngine::with_prefix_tokens` and exports the
  counters as the existing `deepinfer_prefix_cache_*` metrics (plus
  `..._evictions_total`). The CPU golden engine keeps its
  snapshot-based index (`server/src/prefix.rs`) — the two never mix.

Golden: `serve_prefix_cache_reuse` and `serve_prefix_cache_pressure_valve`
(auto-skipped without a GPU) cover the cold/warm/CoW/reset cycle and
the pressure valve; the `golden_run --record/--verify` battery gained a
`serve_prefix_cache` check.

Known follow-up (tracked): the SWA rings are snapshotted per node
rather than paged into the pools — a hybrid vLLM-style paged ring would
remove the per-node ring cost and make the budget purely page-based.

## Device-side sampling (implemented — the P0 batch, Nov 2026)

`di_sample_topp_kernel` (`kernels/sampler.cu`) is the serve engine's
on-device sampler — vLLM-V1's "sampler on device" shape. One CTA per
logits row reproduces the `deepinfer-sampler` pipeline in place:
cap-mask → sanitize → temperature → greedy argmax (ties → lowest id,
bit-identical to the CPU rule) → top-k → softmax → top-p → min-p →
renormalize → categorical over a host-pre-drawn uniform. The per-row
parameters ride a `[B, 6]` f32 block (`SampleParams`); the uniform is
pre-drawn from the request's own Philox stream host-side, so
batch-invariance and RNG parity with the CPU path are preserved exactly.
Threshold selection needs no sort: a monotone f32→u32 key map turns
"the k-th value" and "the smallest descending prefix with mass ≥ p" into
fixed-order bit-descents (32 passes each, tie boundaries by id-descent),
with the `-inf` sentinel contributing zero mass — the CPU fill's exact
semantics.

Stochastic rows match the CPU sampler up to `expf` rounding (the
documented tolerance; golden tests assert distributional agreement ≥ 90%
plus replay bit-determinism, greedy rows exactly).

**Runtime surface**: `serve_decode_sampled` returns `(token, anchor
hidden)` per row — at DeepSeek vocabulary the per-step D2H drops from
`[B, vocab]` f32 (~516 KB/seq) to `[B, hidden + 2]` (~28 KB/seq). The
server's decode flush takes this fast path whenever every row is a
plain grammar-free decode; grammar rows and spec bonus anchors keep the
logits path (the vLLM/SGLang fallback split).

## Async step loop (implemented)

`serve_decode_sampled_stage` / `serve_decode_sampled_finish` split the
sampled decode into launch and consume halves (P0-5): the slim outputs
return through event-guarded asynchronous D2H copies, and the engine
thread queues the step's prefill chunks between staging and consuming —
decode-result processing overlaps prefill GPU time (one step of the
vLLM-V1 overlap shape, no speculative scheduling: the next plan still
waits for this step's tokens). The server's `execute()` runs decodes
first, stages flush-1, prefills, runs the prefills' bonus flush-2, then
consumes flush-1.

## Serve graphs — prefill, verify and decode buckets (implemented)

`enable_serve_graph(true)` (env `DEEPINFER_SERVE_GRAPH=1` at the
server) captures the paged serve forward into CUDA/HIP graphs at
bucketed shapes and replays them as single launches — SGLang's
"breakable prefill graphs" applied to the paged engine:

* **Prefill chunks** at `{64, 128, 256, 512}` tokens (the
  `SERVE_CHUNK` ladder) and **DSpark verify chains** at `sq ∈ [2, 8]`.
* **Decode batches** at bucketed widths (every width ≤ 8, powers of two
  above, plus `max_seqs`); sub-bucket batches pad with inert cells
  (`active = 0`) — every pool-writing kernel guards on the flag and pad
  slots borrow slot 0's page tables, so no pointer is ever stale.
* Graph safety rests on the existing kernel contract (per-step scalars
  in device cells) plus a **readback-free MoE prefill**: the
  aligned-contiguous dispatch launches at the worst-case row bound
  (`moe_dispatch_bound`), zeroes the persistent staging buffers
  (`mpf_*` in the serve workspace), and the grouped GEMMs
  (`di_gemm_grouped_nt_bound_kernel` bf16/f32, the fp8 contiguous
  kernel at the bound) exit per-segment from the device-resident
  offsets — the host never sees `m_total` at all.
* Token/cell uploads stay outside the captured region; rollbacks stay
  exact (graphs read positions and page tables from device memory every
  replay).

## Fused DSpark drafting (implemented)

`draft_step_fused` (P0-3, SGLang's `MarkovSampleStep` shape) runs the
whole MTP module on the GPU — eh_proj GEMV, the shared/depth norms, the
degenerate single-head block (the length-1 context's 1×1 softmax is
exactly 1, so the Q/K projections cancel bit-exactly), the SwiGLU MLP,
and the fused head pick (`di_draft_pick_part/final`: an
online-rescaled argmax + top-1 confidence over the tied head, no
`[vocab]` logits ever materialized). Eight bytes come back. The server
drafts through it whenever no grammar mask or vocabulary cap needs the
draft logits; `DEEPINFER_SERVER_SPEC_GREEDY=1` forces the one-hot argmax
verification rule even for stochastic requests
(`ServeTuning::spec_greedy`), and `verify_greedy_blockwise` segments
verification for bucketed chain graphs.

## NVFP4 weight-only precision (implemented)

`Precision::Nvfp4` (`--precision nvfp4`) quantizes every GEMM's B side
to the NVFP4 weight layout — e2m1 nibbles (two per byte, low nibble =
even k) with one E4M3 scale per 16-element group
(`deepinfer_fp8::quantize_weight_nvfp4`) — while the A side stays f32:
weight traffic halves versus FP8 and the numerics are CPU-golden
(`dequantize_weight_nvfp4`). The portable kernels decode B tiles on the
fly: dense `di_gemm_nvfp4_nt_kernel`, the bound grouped kernel, and the
masked grouped kernel (dtype 3) — dense GEMMs, the arena MoE paths and
the serve MoE paths all route through them.

Follow-up (tracked): the tcgen05 `kind::mxf4nvf4` variant — e2m1
operands natively with UE4M3 16-element scales in Tensor Memory — needs
on-hardware validation on a Blackwell device before it replaces the
scalar core (the AMD MFMA precedent: opt-in env until the first golden
run).

## Hierarchical KV — host-memory L2 tier (implemented)

The radix tree's evictions now **spill** instead of dropping (P1-6, the
LMCache/Mooncake shape): `serve_prefix_evict_to_budget` reads each LRU
leaf's page content (all five pools, every self layer), `k_sf` segment
and ring snapshots back once and stores them under the block's chained
hash in the `L2Tier` — a byte-budget-bounded host arena
(`DEEPINFER_PREFIX_L2_BYTES`, default 25% of the token budget's device
footprint). A VRAM miss (or short match) walks the hash chain deeper
into the tier and **pages in** the surviving blocks — fresh pages, H2D
restores, re-created tree nodes — before falling back to a cold
prefill. Counters surface as `l2_spills` / `l2_pageins` /
`l2_blobs` / `l2_bytes` in `PrefixCacheStats`.

## Parallel checkpoint loader (implemented)

`load_weights` decodes per-layer tensors on a scoped worker pool
(`parallel_map`: `available_parallelism` threads over contiguous
chunks, zero-dependency `std::thread::scope`, zero unsafe) — the MLA
projections, indexer tensors and the routed experts that dominate the
tensor count read in parallel from the page cache (P1-10). Order and
error semantics are preserved (first failing chunk wins).

## Offline CUDA compile gate (implemented)

`scripts/check-cuda.py` compiles every embedded kernel translation unit
exactly the way the runtime JIT does (COMMON + module source, `--std=c++17
-default-device`) through NVRTC and then assembles the emitted PTX with
ptxas for `sm_80`, `sm_90`, `sm_100` (plus `sm_100a`/`sm_103a` for the
tcgen05 module). Syntax, semantics **and inline PTX** (the `cp.async`,
`atom.sys`, `mma.sync` strings) are validated without a GPU present; the
gate caught three latent breakages on its first run (an invalid
`--use_fast_math=off` NVRTC flag, the header-only `NAN`/`INFINITY` macros,
and `ATT_WARPS` leaking across translation units) — all of which would
have failed every kernel JIT on hardware. Run it with
`python3 scripts/check-cuda.py` (requires the `nvidia-cuda-nvrtc-cu12`
and `nvidia-cuda-nvcc-cu12` pip wheels).

## cp.async-pipelined attention and vectorized DSA kernels (implemented)

The hot serve walks gained a second kernel family (P1-8, the FA4
technique adapted to the page-table indirection):

* `di_serve_mla_decode_pipe_kernel` and
  `di_serve_prefill_dense_pipe_kernel` stage their KV rows through a
  3-deep shared-memory pipeline of 16-byte `cp.async` copies
  (`di_cp16`/`di_cp_commit`/`di_cp_wait<N>` in common.cuh). The whole CTA
  consumes ONE staged copy of each row (the direct-load family reads
  every row once per warp — 8x), and tile t+1 is in flight while tile t
  is consumed. The per-row online-softmax order and every arithmetic
  expression are identical to the direct-load kernels — **bit-identical
  outputs**, pinned by the `serve_pipe_vs_legacy` golden entry (hash
  compare). On HIP/pre-Ampere the copy helper degrades to synchronous
  loads (same bytes, no overlap). `DEEPINFER_SERVE_PIPE=0` forces the
  legacy family for A/B.
* `di_serve_thunder_logits_vec_kernel` and
  `di_serve_dot_scores_vec_kernel` switch the DSA score kernels from one
  thread per key row (32-way uncoalesced row streams) to one WARP per key
  row with packed loads — u32 for the e4m3 lightning rows, float4 for the
  f32 tail and the conv feat rows — so every load instruction becomes a
  full 128-byte transaction. Lane-sliced d-ascending partials merge
  through the fixed warp tree: deterministic, tolerance-golden versus the
  scalar family (`DEEPINFER_SERVE_VEC=0` to disable).

Tile geometry lives in the launcher (`SV_PIPE_SMEM` = 32 KiB budget, 3
stages, rows-per-stage by cache format — unit-tested in Rust).

## Fused symmetric-heap EP and EPLB feedback (implemented)

The low-latency EP path gained a fused shape (P1-9, the MegaMoE/DeepEP-v2
technique on the existing symmetric heaps):

* The masked per-expert workspace (aq/asf/meta2/counters) plus the expert
  outputs `y` and a **completion table** moved INTO the heap
  (`SymmetricEp::new_with_experts`); `fused_buffers` hands the slices
  straight to `gemm_grouped_masked_nt` — the grouped GEMM computes
  directly over the symmetric heap layout, no staging copies between
  communication and compute.
* `di_ep_publish_ll_kernel` replaces the row-copying combine: per valid
  expert row the owner writes one 8-byte completion entry
  `(rank << 48) | y_row_byte_offset` into the home's table (a tiny remote
  write instead of a `hidden`-f32 row) and release-increments the home's
  `combine_done`; `di_ep_reduce_remote_ll_kernel` reads the y rows
  directly from the peers' heaps through the table. Per-(token,k)
  traffic drops from a full f32 row write+read to 8 bytes.
* EPLB feedback: `Kernels::topk_histogram` accumulates a routing
  histogram on device (integer atomics — order-independent, hence
  deterministic); the host drains it into `deepinfer_eplb`'s
  `ExpertStats`, whose moving average drives `rebalance_experts`, and the
  resulting placement becomes the dispatch tables through
  `deepinfer_eplb::ep_routing` (`rank_map` + per-rank `slot_map` — the
  dispatch kernel takes them as device-resident tables, replacing the
  hard-coded `e * world / num_experts` formula).

## Persistent self-layer megakernel (implemented — phase 1)

`di_mega_self_block_kernel` (P1-7, `DEEPINFER_MEGAKERNEL=1`, BS <= 4)
replaces the per-layer chain
`rmsnorm -> GEMV(fused_qkv) -> copy_cols x2 -> latent-norm -> rope(kv) ->
paged KV encode -> GEMV(q_b) -> rope(q)`
— eight kernels and four global round-trips of the intermediate rows —
with ONE launch (grid = rows x layers): each CTA keeps `h/x/qa/kv`
resident in shared memory, streams both weight matrices through a
2-deep `cp.async` tile pipeline, quantizes the activations per-token-group
in-kernel for the FP8 weight path (bit-exact codecs), applies the latent
norm, both RoPEs (the q-pair lives in adjacent lanes — one shuffle) and
writes the encoded KV row straight into the paged pool. All four weight
dtypes (f32/bf16/fp8/nvfp4) are decoded in-kernel. Every reduction is a
fixed tree and every GEMV accumulates k-ascending — deterministic; versus
the tiled GEMM kernels the association differs, so outputs are
tolerance-goldens (`serve_megakernel_vs_unfused` in golden_run). The
launcher validates the geometry (16 B row alignment, even latent,
hd % 32, smem budget) and the engine falls back to the unfused chain
whenever anything is unsupported.
