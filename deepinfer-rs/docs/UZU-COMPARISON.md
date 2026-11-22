# DeepInfer-rs vs. uzu — an engineering comparison

This document compares DeepInfer-rs with **uzu** in both of its senses,
and records what we adopted from each while making this repository
production-ready and servable.

Two different projects carry the name:

1. **NVIDIA uzu** — "a minimal research kernel for building fast
   inference engines", open-sourced by NVIDIA's GPUs team in October
   2025. It was a *teaching artifact*: a vertical slice of an inference
   engine written directly on CUDA (tcgen05 MMA + TMA on Blackwell,
   correctness tests against PyTorch, B200 benchmarks), with a thin
   Python host. Its public repository is no longer accessible at
   `github.com/NVIDIA/uzu` (404 as of this writing), so the comparison
   below is based on its published release scope; where a claim cannot
   be re-verified today, it is marked as such.
2. **uzu by Mirai (`trymirai/uzu`)** — a live, production-shaped local
   inference engine (Rust core, Metal-first, CPU/Vulkan/WebGPU
   backends, Swift/Python/TypeScript/Rust bindings, model catalog and
   downloader, chat sessions, grammar support). This one *is* the
   "complete and production ready, model servable from this" reference
   shape, and its repository was surveyed directly for this comparison.

## What each project is for

| | NVIDIA uzu (research kernel) | trymirai/uzu (local engine) | DeepInfer-rs |
|---|---|---|---|
| Goal | Teach how to build inference kernels on raw CUDA | Deploy AI in-app with zero latency on Apple devices | Full DeepSeek-V4.1-Flash lineage: research-grade algorithms **and** a servable stack |
| Host language | C++/CUDA + Python glue | Rust | Rust (pure, `#![forbid(unsafe_code)]` everywhere except the audited GPU FFI shims) |
| Hardware | NVIDIA Blackwell (SM100/SM103), Hopper fallback | Apple Silicon (Metal), CPU, Vulkan, WebGPU | NVIDIA GB200/GB300 NVL72 (sm_100/sm_103 via NVRTC JIT), AMD MI335X (HIP), CPU golden path |
| Model coverage | One small transformer end-to-end | Broad catalog (Qwen, Llama, Gemma, ...) | DeepSeek-V4.1-Flash architecture: MLA + DSA sparse attention + SWA + DeepSeekMoE + MTP |
| Serving surface | None (benchmarks/tests only) | In-process chat API + language bindings | **OpenAI-compatible HTTP API with SSE streaming** (`deepinfer-server`), continuous batching, Prometheus metrics |

## Feature comparison

### Kernels and compute

| Capability | NVIDIA uzu | trymirai/uzu | DeepInfer-rs |
|---|---|---|---|
| Tensor-core GEMM | tcgen05 + TMA (the reference implementation of the pattern) | Metal Performance Shaders / custom Metal kernels | `tcgen05.mma.kind::mxf8f6f4.block_scale` + `cp.async.bulk.tensor` TMA, warp-specialized 3-stage pipeline; Hopper/Ampere `mma.sync` fallback; AMD MFMA FP8 (opt-in) |
| FP8 block-scaled GEMM | Yes (DeepGEMM-recipe) | N/A (consumer quant formats) | Dense + grouped-contiguous + masked, UE8M0 1x128/128x128 recipe, CPU-golden-validated |
| Attention | Decode/prefill kernels | Metal attention | MLA split-KV decode + combine (F32/FP8-528B/FP4-288B caches), sparse + dense prefill, DSA lightning indexer with exact top-K |
| CUDA graphs | Demonstrated | N/A | Full decode-step capture, bit-identical replay, rollback-safe |
| Multi-GPU | No | No | NVLink peer collectives + symmetric-buffer EP (NVSHMEM-style protocol on raw driver primitives) |

DeepInfer-rs adopted NVIDIA uzu's central lesson — *write the kernels
yourself so the whole stack is legible* — but went further in scope:
every operator also has a CPU reference implementation that serves as
the numerical oracle for on-hardware golden runs.

### Runtime and serving

| Capability | NVIDIA uzu | trymirai/uzu | DeepInfer-rs |
|---|---|---|---|
| Continuous batching | No | Yes (in-process sessions) | Yes — vLLM-style chunked prefill, preemption, batch-invariant admission |
| Streaming | No | Yes (async session API) | Yes — SSE `chat.completion.chunk` deltas, UTF-8-safe incremental detokenization |
| Cancellation | N/A | Session-level | Per-request abort on client disconnect, stop-string match, or deadline; slot recycling with exact cache teardown |
| Structured output | No | Yes (grammar in the language model path) | EBNF / JSON Schema / regex → DFA token masking on the batched serving path (same engine as offline `generate`) |
| Speculative decoding | No | Partial (speculators module) | MTP/DSPark drafting + verification on **both** the offline path and the batched serve path: continuous-batching DSpark rounds with per-request rollback of rejected tails, grammar-aware draft/verify masking, batched bonus decodes, spec budget accounting in the scheduler and `deepinfer_spec_*` Prometheus series |
| HTTP API | No | No (bindings, not network) | OpenAI-compatible: `/v1/chat/completions` (stream + non-stream), `/v1/completions`, `/v1/models`, `/health`, `/metrics`, `/admin/shutdown` |
| Auth / limits | No | N/A | Bearer auth, body/header/connection caps, per-request deadlines, `max_tokens` ceiling, degraded-mode health |
| Metrics | Benchmarks only | App-level | Prometheus text: request counters, TTFT + latency histograms, token throughput, queue depth, engine health |
| Model management | Single baked model | Catalog + downloader + progress API | safetensors checkpoint loading (host), `--tiny` deterministic self-check mode |

### Correctness methodology

| | NVIDIA uzu | trymirai/uzu | DeepInfer-rs |
|---|---|---|---|
| Oracle | PyTorch | "Traceable computations against source-of-truth" | The CPU reference implementation itself (414+ unit tests) + golden-run artifacts (`--record`/`--verify`) |
| Determinism | Where practical | Not a headline | A contract: batch-invariant scheduling, per-sequence Philox streams, fixed reduction orders; the serving engine's stream is asserted **bit-identical to `generate()`** in CI |
| Supply chain | Vendor CUDA headers | Rust + Metal crates | Zero external crate dependencies for the CPU path; GPU layer dlopens `libcuda`/`libnvrtc`/HIP at runtime — no build-time vendor requirement |

### Safety

* NVIDIA uzu: raw CUDA C++ (memory safety is on the author).
* trymirai/uzu: Rust with a Metal FFI boundary.
* DeepInfer-rs: `#![forbid(unsafe_code)]` across the workspace; the
  single audited unsafe surface is `deepinfer-gpu`'s dlopen/FFI shims
  (`deny` + per-module `#[allow]` with SAFETY proofs). The new
  `deepinfer-server` crate — including its hand-rolled HTTP/1.1 parser
  — is `forbid(unsafe_code)` and fuzzed by its own hostile-input tests
  (oversize bodies, malformed request lines, bad percent-escapes,
  mid-stream disconnects).

## What this comparison changed in DeepInfer-rs

Making the repository "complete, production ready, and servable"
closed the gaps the uzu projects highlighted:

1. **A serving surface** (`crates/server`, ~4,300 lines): the OpenAI
   HTTP API, SSE streaming, cancellation, auth, limits and metrics
   described above — with zero external dependencies, keeping the
   workspace's supply-chain posture intact.
2. **Production request lifecycle**: continuous-batching engine loop
   with per-request sampling state, grammar masking, stop strings,
   deadlines, and per-request error isolation (a capacity-overflowing
   request fails alone; the engine keeps serving).
3. **Engine-thread architecture** mirroring the scheduler's
   zero-overlap pattern: coordinator thread (scheduler + event fan-out)
   and engine thread (model + sampling), with optimistic prefill
   commits and a strict one-plan-in-flight gate — the bug class that
   duplicate-schedules work is covered by a dedicated regression test.
4. **Chat template + streaming detokenization**: the official
   DeepSeek conversation format, and a delta decoder that only emits
   complete UTF-8 sequences (multi-byte characters split across tokens
   are held back, matching vLLM/SGLang behavior).
5. **Operations**: `/health` with degraded mode after engine failure,
   Prometheus `/metrics`, graceful `/admin/shutdown` drain, config
   file + `DEEPINFER_SERVER_*` env overrides, `--tiny` self-check.
6. **DSpark speculative decoding on the serve path** (`crates/server/src/spec.rs`):
   MTP drafting is wired through continuous batching — every decode step
   runs per-request draft → verify → rollback → emit rounds, with the
   rejected tails rolled back out of the paged caches (page recycling,
   per-sequence isolation), grammar-aware draft-time and verification
   masking (beyond the offline path, which disables speculation under a
   grammar), bonus decodes batched into one `serve_decode` launch per
   step on the GPU engine, and `deepinfer_spec_*` Prometheus series.
   Greedy parity is pinned by tests: spec serve == plain serve ==
   offline `generate`, including under grammars and under batching.
7. **The serving-efficiency state of the art, integrated** (the
   2025-2026 engine advancements, adapted to this architecture):
   * **Prompt prefix caching** (`crates/server/src/prefix.rs`, the
     SGLang-RadixAttention / vLLM-automatic-prefix-caching feature):
     an LRU-bounded radix tree over token ids holding exact engine
     state snapshots; admission-time lookup skips the recomputed
     prefix (`Scheduler::skip_prefill`), the engine restores the state
     and prefills only the suffix. Bit-identity at the chunk boundary
     is pinned by tests; the restore-fallback and scheduler-eviction
     rewind paths are exact.
   * **Adaptive speculative depth** (vLLM dynamic speculative
     decoding): a per-request acceptance EMA retargets the draft depth
     so speculation stays profitable as concurrency rises; greedy
     parity is preserved by construction.
   * **Token-budget admission control**: `max_prompt_tokens` rejects
     oversized prompts with a clear `400` before they reach the
     scheduler (count tokens, not requests).
   * **The byte tokenizer as a first-class serving mode** with the
     DeepSeek chat pieces as single tokens and a capped sampling
     surface (`--tokenizer byte`).

## Honest remaining gaps (tracked)

* **GPU-engine prefix caching.** The prefix cache is implemented on
  the CPU golden engine (state snapshots). The GPU paged-serve engine
  reports "unsupported" today; sharing there wants reference-counted
  pages in the serve pools rather than snapshots — tracked.
* **Chat-template special tokens with a plain byte tokenizer.**
  Resolved: the serving byte tokenizer carries the DeepSeek pieces as
  single tokens (`ByteTokenizer::chat_specials`, ids 256-259), and
  `--tokenizer byte` makes byte-mode serving a first-class mode on
  any checkpoint.
* **Single-process serving.** Resolved as the documented deployment
  shape: one process, engine threads on `std::thread`, no forks, no
  IPC — multi-rank symmetric EP stays an offline operator-layer
  capability.
* **NVIDIA uzu re-verification.** The NVIDIA repository's current
  unavailability means its post-release evolution could not be
  re-surveyed; the comparison row for it reflects the October 2025
  release scope.
