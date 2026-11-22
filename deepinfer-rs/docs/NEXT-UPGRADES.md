# Next upgrades: research report and editing plan (2026-11)

Grounding for the next implementation phases. Sources: SGLang
v0.5.21–v0.5.23 release notes (Oct–Nov 2026) and the 2026-Q2/Q4
roadmaps, vLLM V1 architecture blog + v0.18 docs, FlashMLA and
DeepGEMM repositories (Sept–Oct 2026 updates), FlashInfer v0.7.4
release highlights, NVIDIA Wide-EP/NVL72 and NVFP4 material,
LMCache/Mooncake tech reports, DistServe retrospective, and the 2026
speculative-decoding literature (MLSys 2026). Every recommendation
below is tied to a concrete file-level edit in this repository.

---

## 1. Where the market is (Nov 2026)

| System | Version | Position |
|---|---|---|
| SGLang | v0.5.23 (Nov 2, 2026) | DeepSeek-V4.1-Flash reference stack: megakernel/monokernel low-latency mode, breakable prefill CUDA graphs, Rust-core prefix cache, DSpark markov drafts, HiCache L3, Paged Experts, elastic EP, TPU7x |
| vLLM | v0.18 (V1 engine) | Async scheduling (CPU schedules step N+1 under step N's GPU work), zero-overhead hash-chain prefix caching, piecewise CUDA graphs, FlashInfer as default Blackwell attention backend, NIXL/Mooncake/LMCache P/D connectors |
| TensorRT-LLM | 1.0 | PyTorch-based stable architecture, Blackwell "model-specific + agentic-workload" optimizations |
| FlashMLA | 2026.09.30 | **Blackwell-only** (Hopper support removed): sparse prefill 1350 TF, sparse decode 1024 TF on B200 (CUDA 13.3), fused norm-RoPE-attn-RoPE-cast 1460/950 TF, V4.1 cache formats (FP8 528 B e4m3+UE8M0, FP4 288 B e2m1+E4M3 scales), Ascend 950 port |
| FlashInfer | v0.7.4 (Nov 10, 2026) | DeepSeek-V4 support, **batch-invariant mode for GEMMs and quantizers on Blackwell and Rubin**, is SGLang/vLLM's Blackwell attention backend |
| FlashAttention-4 | Apr 2026 | Redesigned async pipeline for Blackwell (softmax interleaved with MMA, TMA-driven), the attention reference for dense heads |

Three industry-wide currents matter to us:

1. **The stack is collapsing into fewer, bigger kernels and more
   graphs.** SGLang's headline decode win (44% on GB300, 75% on
   MI350X) is a *fused persistent megakernel*; its prefill win
   (+33–37% saturated) is *breakable prefill CUDA graphs* replayed at
   any context length.
2. **KV is becoming hierarchical and disaggregated.** LMCache
   (CPU/SSD tiers, 101-citation tech report), Mooncake Store, SGLang
   HiCache L3, vLLM connectors — the radix/page cache is the substrate
   that P/D transfer builds on.
3. **The control plane is moving to Rust.** SGLang's prefix cache now
   runs on a Rust core by default, with an RFC for a native Rust gRPC
   server and a gradual scheduler/Rust migration. We are already
   there; our advantage is a *zero-dependency, single-process* core.

DeepSeek's own line: V4.1-Flash runs the DSA family (lightning
indexer + sparse attention + SWA cross layers) on FlashMLA V4.1
formats — exactly the layouts our engine implements — and the
FlashMemory/LSA report sketches a *neural memory indexer* beyond DSA
(watch item, not actionable yet).

---

## 2. Gap matrix (us vs the market)

| Capability | vLLM v0.18 | SGLang v0.5.23 | deepinfer-rs today | Verdict |
|---|---|---|---|---|
| Prefix caching | hash-chain, <1% overhead | radix (Rust core) | **CPU snapshots + GPU block-radix page sharing** (just landed) | parity on GPU; see P1-6 for hierarchy |
| Continuous batching + chunked prefill | yes | yes + mixed-chunk | yes (`scheduler`) | parity |
| CUDA-graph decode | piecewise | full + breakable | **single-sequence decode graph only** | gap → P0-2 |
| CUDA-graph prefill | piecewise | breakable, any length, +33-37% | **none** | gap → P0-2 |
| Speculative decoding | EAGLE-3, NGPUhead | DSpark markov heads, draft graph tails, block verify, per-request adaptive | DSpark host-drafted, batched verify+rollback, adaptive depth | partial → P0-3 |
| Sampling location | on-device (V1) | on-device | **host, full-vocab logits readback per step** | gap → P0-1 (largest single win) |
| CPU-GPU overlap | async scheduler (V1) | overlap scheduler | coordinator/engine threads, but engine blocks on GPU each step | partial → P0-5 |
| Weight quant | FP8, NVFP4 (cutlass) | FP8, NVFP4, MXFP4/8 (MegaMoE) | FP8 block-scale (tcgen05 path) | gap → P0-4 (NVFP4) |
| KV formats | FP8, (FP4 via FlashMLA) | FP8/FP4 V4.1 | **FP8 528 B / FP4 288 B V4.1** ✓ | parity |
| Attention kernels | FlashInfer/FA4 | FlashInfer/FA4/TRT | bandwidth-oriented scalar-warp, golden-exact | gap → P1-8 |
| MoE comms | DeepEP v2, Mooncake EP | DeepEP v2, MegaMoE, wide-EP + EPLB | peer-copy + symmetric-buffer LL EP (NVSHMEM-style) | parity-ish → P1-9 |
| Hierarchical KV (host/SSD tier) | LMCache/Mooncake connectors | HiCache L2/L3 native | none | gap → P1-6 |
| P/D disaggregation | NIXL/Mooncake | NIXL, on-the-fly role switch | none (single-process by design) | P2 watch |
| LoRA | yes | on quantized MoE (FP8/NVFP4) | none | P2 |
| Batch invariance | (deterministic modes) | DeepEP-v2 BF16 batch-invariant | deterministic by construction (fixed trees/launches) | document + pin → P2 |
| Structured output | xgrammar | xgrammar + others | EBNF→DFA, byte-safe, in-house | parity |
| Load speed | — | InstantTensor (10x) | safetensors mmap | P1-10 |

---

## 3. Priority plan (specific edits)

### P0-1 Device-side sampling + slim readback (largest single win)

**Why.** `serve_decode_out` synchronizes and reads back
`[B, vocab]` f32 logits **and** `[B, hidden]` per step, then samples
on the host (`deepinfer-sampler` at the server layer). At
DeepSeek vocab (≈129k) that is ~516 KB/seq/step of D2H plus a full
host sync — the engine thread cannot overlap anything. vLLM V1's
sampler runs on device; its async scheduler exists *because* the
per-step host dependency is gone.

**Edit.**
- `crates/cutile/src/kernels/sampler.rs` already has
  `argmax_blocks`/`argmax_final`: add `sample_topp`
  (temperature + top-k/top-p, single-block kernel per row, seeded by
  the per-sequence `seq_seed` we already thread through the
  sampler crate).
- `crates/runtime/src/serve.rs`: new
  `serve_decode_sampled(batch, params) -> Vec<SampledTok>` that (a)
  launches the serve forward as today, (b) launches the device
  sampler on `ws.logits`, (c) reads back only `[B]` token ids plus
  the spec-anchor hidden rows (keep hidden readback — MTP drafting
  is host-side today; 28 KB/seq is fine), (d) syncs once on an
  event, not the full stream.
- `crates/server/src/engine.rs` `execute()`: plain-decode slots use
  `serve_decode_sampled`; only grammar-constrained and spec requests
  take the logits path (grammar masking stays host-side — same
  fallback split vLLM/SGLang use).
- Metrics: `decode` path bytes-readback drops ~130x at production
  vocab.

**Effort**: M. **Expected**: removes the dominant per-step stall;
multi-ms savings per step at real vocab, prerequisite for P0-5.

### P0-2 Prefill/verify CUDA-graph capture (breakable graphs)

**Why.** Every `di_serve_*` kernel already obeys the graph-safety
rule (no varying scalar as kernel argument; per-slot geometry from
device cells) — we capture *decode* only. SGLang replays prefill
graphs at any context length ("breakable", +33–37% saturated
prefill on GB300; MLA prefill capture default = 8192-token chunks).

**Edit.**
- `crates/runtime/src/device_model.rs`: generalize
  `enable_decode_graph` into a `ServeGraphRunner` with two modes:
  decode (existing) and prefill buckets {512, 1024, 2048, 4096,
  8192} tokens (our `SERVE_CHUNK` chunking already aligns).
  "Breakable" here is natural for us: the chunked forward is a
  fixed launch sequence once `sq` is bucketed — pad the tail chunk
  and mask (the `active` cell flag already supports it).
- `crates/runtime/src/serve.rs` `serve_chunk_rows`: take the graph
  path when `sq` matches a bucket and `B == 1`; keep eager fallback
  (mixed batches, first chunk).
- Rollback interplay is already safe: graphs re-read the position
  cell each replay, and page tables live in device buffers the
  kernels indirection through (`tbl[r/64]`).
- Verify-path capture: SGLang captures draft-extend graphs for spec
  (1.5x decode at BS1) — same runner, `sq = draft_len` buckets.

**Effort**: M-L. **Expected**: +20–40% prefill throughput on
GB300-class; removes ~hundreds of launches per chunk.

### P0-3 DSpark to SGLang-v0.5.23 level

**Why.** Their Nov-2026 stack: draft CUDA-graph tail (drafting inside
the graph), fused draft sampling (`MarkovSampleStep` = one op for
add+sample+pick), greedy-draft mode (argmax accepted as one-hot),
block verification (verify per 64-token block), and "keep the DSpark
draft SWA out of the radix tree" (we already never insert
verify-chunk blocks — same insight, our tests pin it).

**Edit.**
- `crates/mtp/src/draft.rs`: `MarkovSampleStep` equivalent = a tiny
  cutile kernel doing embedding-dot + temperature sample in one
  launch (draft step is GEMV-shaped: `hidden × W_embᵀ`), replacing
  the host dot product; keep host fallback.
- `crates/server/src/spec.rs`: greedy-draft mode
  (`DEEPINFER_SERVER_SPEC_GREEDY=1`) — argmax proposal accepted as
  one-hot in verification (their `SGLANG_DSPARK_USE_GREEDY`).
- Block verification: `crates/mtp/src/verify.rs` gains a
  `verify_blockwise` that verifies at 64-token granularity inside
  one forward — pairs with P0-2's prefill buckets (chain length ==
  bucket → one graph replay).
- Anchor plumbing: keep the hidden readback from P0-1 (that is
  exactly their "capture aux hidden states on the target" item).

**Effort**: M. **Expected**: their measured wins for the same
components are 1.5x BS1 decode (draft graphs) and fused-sampling
overhead removal; our adaptive-depth controller already matches
their "adaptive spec configs".

### P0-4 NVFP4 weights on the tcgen05 path

**Why.** tcgen05 `mxf8f6f4.block_scale` takes fp4 (e2m1) operands
natively; FlashMLA V4.1's FP4 KV is e2m1 + **E4M3** scales (288 B) —
note the scale dtype differs from the FP8 path's UE8M0. NVIDIA
reports 2–2.3x throughput at iso-accuracy for NVFP4 GEMMs; SGLang
runs NVFP4/MXFP4 experts through MegaMoE; our GEMM already does
e4m3+UE8M0 on SM100.

**Edit.**
- `crates/kernels/kernels/gemm_sm100.cu`: add an e2m1 variant of
  the tcgen05 path (decode nibble pairs, E4M3 scale factors in
  Tensor Memory instead of packed UE8M0 x4) — the TMA swizzle and
  warp-specialization structure carries over.
- `crates/core/src/format.rs` + `crates/fp8/src/lib.rs`: e2m1
  codec exists (FP4 KV); add the NVFP4 weight layout
  (16-element blocks, e4m3 scales, 32-lane replication for
  mxf8f6f4).
- `crates/runtime/src/loader.rs` + `weights.rs`: NVFP4 quantize-on-
  load (like today's FP8 path) behind `Precision::Nvfp4`;
  DEEPINFER_PRECISION=nvfp4.
- MoE first (routed experts dominate FLOPs), dense GEMMs second.

**Effort**: L. **Expected**: ~2x MoE GEMM throughput at 4-bit
weights; halves weight memory vs FP8.

### P0-5 Async step loop in the engine thread

**Why.** vLLM V1's async scheduler prepares step N+1 while step N
runs on GPU; SGLang's overlap scheduler does the same and their
Q2-26 roadmap item is "hide CPU overhead … for all cases". Our
coordinator already runs concurrently, but `execute()` serializes:
launch → sync → readback → sample → respond.

**Edit.** After P0-1 (device sampling) this becomes tractable:
- `crates/server/src/engine.rs`: split `execute()` into
  `launch(plan) -> StepHandle` and `complete(handle) -> results`;
  keep one step of lookahead: launch N+1's forward before
  consuming N's outputs (double-buffered `ws.logits`-equivalents —
  the serve workspace is single, so add a second `ServeWs`-lite
  output region or ping-pong the logits buffer).
- Event-based readback: record an event after the sampler kernel;
  `complete()` waits on the event (not `stream.synchronize()`).
- The scheduler's optimistic prefill commit (already present at the
  coordinator) plus this change gives the full vLLM-V1 shape.

**Effort**: M. **Expected**: hides 100% of host sampling/queueing
under GPU time at high batch; the last big throughput lever in the
single-process path.

---

### P1-6 Hierarchical KV cache (host-memory L2 for the radix tree)

**Why.** LMCache tech report (101 citations) and Mooncake: extending
the cache across GPU→DRAM→SSD multiplies hit rates for agentic and
multi-tenant workloads; SGLang made "hierarchical cache + hybrid
attention" a *native* feature (HiCache L3 file storage, HiSparse
CPU-memory KV with spec support). Our new GPU block-radix tree is
already page-granular — the right substrate.

**Edit.**
- `crates/runtime/src/prefix_cache.rs`: eviction gains a *spill*
  hook: instead of dropping an LRU leaf, serialize its page bytes +
  k_sf + rings to a host-side arena (`Vec<u8>` slots, LRU-bounded by
  a byte budget) keyed by the node's chained hash; the tree keeps a
  "cold" tombstone pointing at the host slot.
- `serve_prefix_restore`: a miss in VRAM checks the host tier, then
  page-ins (H2D copies into fresh pages, rings restored) before the
  suffix prefill — TTFT still beats a cold prefill by far.
- Page-out traffic is one contiguous copy per pool page
  (page-granular, exactly what Mooncake/NIXL transfer looks like —
  this is also the groundwork for P2-11).
- Config: `DEEPINFER_PREFIX_L2_BYTES` (default 25% of the token
  budget's footprint).

**Effort**: M. **Expected**: LMCache-class hit-rate gains for
long-horizon agents (their number: up to 3x throughput on
cache-heavy workloads).

### P1-7 Persistent megakernel decode (low-latency mode)

**Why.** SGLang v0.5.23: NVIDIA megakernel / AMD monokernel plugins
run DeepSeek-V4.1-Flash decode as fused persistent kernels — 44%
faster single-user decode on GB300, 75% on MI350X. Our decode step
is ~5 launches × n_self layers + MoE — CUDA graphs amortize the
launch cost but not the memory round-trips between kernels.

**Edit.**
- Phase it per layer-group: start with a fused "self-layer block"
  persistent kernel (`crates/kernels/kernels/serve.cu` or a cutile
  kernel) that keeps `(h, x, q, kv_rows)` tiles resident in SMEM/
  registers across rmsnorm→fused-qkv→rope→encode for the whole CTA
  wave, one CTA per (slot, layer-group) with grid-sync between MoE
  stages (or the masked-MoE in the same kernel at small batch).
- Ship behind `DEEPINFER_MEGAKERNEL=1`, goldens vs the graph path
  (tolerances, like the tcgen05-vs-portable check).
- Target the BS ≤ 4 regime (their 44% is single-user); the graph
  path remains the throughput default.

**Effort**: L (research-y, high payoff at low batch).

### P1-8 TMA + SMEM-pipelined attention/DSA kernels

**Why.** FlashMLA sits at 1350/1024 TF sparse on B200; our
attention/DSA kernels are deliberately bandwidth-oriented scalar-warp
designs (documented cut in `docs/GPU.md`). FA4's techniques
(softmax-MMA interleave, async TMA pipelines) are the template; the
NVFP4-GEMM worklogs (238 µs → 23.8 µs via TMA + warp-spec + cluster
multicast) show the same order of gains for grouped GEMM.

**Edit.**
- `di_serve_mla_decode_partial/combine` (kernels/attention.cu):
  convert the KV walk to `cp.async.bulk` 128 B-swizzled tiles with a
  3-stage pipeline; keep the exact reduction order (determinism
  contract) by fixing tile/split boundaries to the current ones.
- The DSA `di_serve_thunder_logits`/`dot_scores` fusions: pack the
  per-head MQA dots into `mma` fragments (the DeepGEMM
  MQA-logits structure).
- Benchmark against the existing golden_run timing entries;
  regressions in determinism tests must stay zero.

**Effort**: L, incremental per kernel. **Expected**: 2-5x on the
attention/DSA fraction of decode.

### P1-9 EP upgrades: MegaMoE-style fusion + EPLB feedback

**Why.** SGLang routes MoE through fused dispatch-GEMM-combine
(MegaMoE/DeepGEMM) and their Q2-26 kernel list is
"communication-compute overlap kernels" + nvfp4/mxfp8. We have
symmetric-buffer LL EP (no host readback) — the next cut is fusing
the per-expert GEMM into the dispatch/combine protocol.

**Edit.** `crates/kernels/src/peer_ll.rs` + `kernels/moe.cu`: after
`dispatch_done`/`group`, launch the grouped GEMM directly over the
symmetric heap layout (masked-m layout already matches
`gemm_grouped_masked_nt`); combine reads expert outputs from the
peers' home slots without the staging copy. Add EPLB rebalance
triggers from live serve-router statistics (`crates/eplb/stats.rs`
fed by a periodic `topk_idx` histogram from the serve path).

**Effort**: M-L.

### P1-10 Fast parallel checkpoint loader

**Why.** SGLang's InstantTensor: 10x model load (parallel
decompress + prefetch + overlap H2D). Our loader is a serial
safetensors mmap walk.

**Edit.** `crates/runtime/src/loader.rs`: thread-parallel tensor
decode (per-tensor tasks into a small rayon-less worker pool —
`std::thread` scoped, zero-dep) + double-buffered H2D upload on a
second stream; keep the deterministic weight-hash check at the end.

**Effort**: S-M. **Ops win** (deploy velocity, OOM-restart speed).

---

### P2 watch list (do not implement yet)

- **P2-11 P/D disaggregation.** The paged pools + block-radix tree
  are the right substrate (page-granular KV transfer = NIXL/
  Mooncake's model); a first cut could be two processes of our own
  server with a shared-memory page transfer channel. Only worth it
  once P0-1/2/5 land (the market data says monolithic colocation is
  still fine below rack scale).
- **P2-12 DeepSeek-V4 LSA / neural memory indexer** (FlashMemory
  report): a new model family beyond DSA — needs published kernels;
  track FlashMLA releases.
- **P2-13 Multi-LoRA** (SGLang serves LoRA on FP8/NVFP4 MoE).
- **P2-14 Spec-decoding research to fold in**: *Speculative
  Decoding: Performance or Illusion?* (MLSys 2026 — argues
  per-workload acceptance modeling; our adaptive-depth controller
  is the right hook), block verification (P0-3), P-EAGLE-style
  parallel drafting (out-of-family but the abstraction fits our
  `TokenEngine` spec surface).
- **P2-15 Batch-invariant determinism**: FlashInfer v0.7.4 ships a
  batch-invariant mode on Blackwell/Rubin; we are deterministic by
  construction — add a golden_run check pinning cross-batch-size
  equality and document it (cheap marketing win for RL rollouts —
  SGLang's Miles work treats determinism as a first-class RL
  requirement).
- **P2-16 gRPC API** (SGLang RFC #22558, Rust): conflicts with our
  zero-dependency rule unless hand-rolled HTTP/2; defer, keep the
  OpenAI-compatible HTTP surface best-in-class (SSE keep-alives,
  /metrics, health).
- **Security note**: the JIL attack (Oct 2026) shows length-
  prediction schedulers are manipulable; we do length prediction
  nowhere — keep it that way when adding admission heuristics.

---

## 4. Recommended order

1. **P0-1** device sampling (unblocks P0-5, biggest steady-state win)
2. **P0-2** prefill/verify graphs (biggest prefill win, reuses
   graph-safe kernels)
3. **P0-3** DSpark upgrades (rides on P0-2's buckets)
4. **P0-5** async step loop (rides on P0-1)
5. **P0-4** NVFP4 (independent; biggest kernel-level win)
6. **P1-6** hierarchical KV (rides on the new radix tree)
7. **P1-8** attention pipelines, **P1-7** megakernel, **P1-9** EP
   fusion, **P1-10** loader — independent tracks.

Each lands with the house contract: CPU-golden unit tests,
auto-skipping device goldens, a golden_run `--record/--verify`
entry, and docs/GPU.md updated.

## 5. Source register

- SGLang releases v0.5.21–23: github.com/sgl-project/sglang/releases
  (Oct 2 / Oct 19 / Nov 2, 2026)
- SGLang roadmap 2026 Q2: github.com/sgl-project/sglang/issues/22949
- vLLM V1 alpha blog: vllm.ai/blog/2025-01-27-v1-alpha-release;
  vLLM prefix-caching docs: docs.vllm.ai/en/stable/design/prefix_caching
- vLLM DeepSeek-V3.2 DSA post: vllm.ai/blog/2025-09-29-deepseek-v3-2
- FlashMLA: github.com/deepseek-ai/FlashMLA (news 2026.09.10 /
  2026.09.30: V4.1 kernels, 1350/1024 TF sparse, FP4 288 B layout,
  Hopper removal)
- FlashInfer v0.7.4 highlights (Nov 10, 2026): DeepSeek-V4,
  batch-invariant GEMM/quant on Blackwell+Rubin
- FlashAttention-4 (Apr 2026): Blackwell async-pipeline attention
- NVIDIA: Wide-EP on NVL72 + EPLB (developer.nvidia.com, Oct 2025);
  NVFP4 introduction (Jun 2025); NVFP4 characterization
  (arXiv 2606.06527)
- LMCache tech report (Jul 2026, 101 citations);
  Mooncake (KVCache-centric disaggregation, Kimi)
- DistServe retrospective "Disaggregated Inference: 18 Months Later"
  (Nov 2025); PD-disagg survey (Mitra 2026)
- Speculative decoding: *Performance or Illusion?* (MLSys 2026);
  P-EAGLE (Mar 2026); DDD dynamic-depth (2024→)
- SGLang DSpark confidence-driven spec (blog); DFlash/KDA/QSA
  families (v0.5.21–23 notes)
