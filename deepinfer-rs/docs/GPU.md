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
  with the serve engine; MTP drafting stays on the single-sequence path
  (documented; the serve batch loop re-enters through prefill for
  verification).

Golden: `serve_engine_multi_sequence` (auto-skipped without a GPU)
compares two interleaved sequences against the single-sequence device
path — itself CPU-validated — through prefill, four interleaved decode
steps, an exact rollback, and a detach/re-attach cycle.

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

## Running on hardware

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
  with the serve engine; MTP drafting stays on the single-sequence path
  (documented; the serve batch loop re-enters through prefill for
  verification).

Golden: `serve_engine_multi_sequence` (auto-skipped without a GPU)
compares two interleaved sequences against the single-sequence device
path — itself CPU-validated — through prefill, four interleaved decode
steps, an exact rollback, and a detach/re-attach cycle.

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

## Running on hardware

```bash
cargo test -p deepinfer-kernels --test golden   # CPU-vs-GPU goldens
DEEPINFER_JIT_DEBUG=1 cargo test ...            # trace JIT compiles
```

Requirements: NVIDIA driver >= 550 (R580+ for B300) with `libnvrtc`
(CUDA 12.4+; 12.9+ for `sm_103` cubins), or ROCm 6/7 with `libamdhip64` +
`libhiprtc`. Nothing needs to be installed to *build*.
