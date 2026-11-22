# DeepInfer-rs

**A pure-Rust, CPU-reference implementation of [DeepInfer](https://github.com/deepseek-ai/DeepInfer) — DeepSeek's high-performance, large-scale inference framework — targeting the DeepSeek-V4.1-Flash architecture.**

This workspace ports the full DeepInfer release feature set to dependency-free, `unsafe`-free Rust, following the published DeepSeek ground truth (FlashMLA, DSA, DeepGEMM, DeepEP, EPLB repositories and the DeepInfer README) at algorithmic fidelity: every component implements the *exact* upstream math and protocol semantics on CPU, with deterministic numerics.

```
cargo test --workspace     # 415 tests, all green
cargo run -p deepinfer-runtime --example offline_inference -- --tiny
```

## Feature matrix (vs. the upstream DeepInfer release)

| Upstream feature | Crate | What is implemented |
|---|---|---|
| FlashMLA attention | `deepinfer-mla` | V4.1 geometry (64 heads × 512 head-dim = 448 latent + 64 interleaved rope, q-LoRA 1536, Wv in 8×8 groups → o-LoRA 512), paged KV cache (page 64) with the exact V4.1 **FP8 (528 B/token: 512 B e4m3 + 16 B e8m0 per-32)** and **FP4 (288 B/token: packed e2m1 + e4m3 scales per-16)** layouts, sparse decode with FlashMLA's exact semantics (invalid = −1, empty-selection ⇒ max-logit −∞ / lse +∞ / zero output, `attn_sink` epilogue `out *= exp(lse)/(exp(lse)+exp(sink))`), chunked split-K online-softmax merge, dense causal prefill, O-RoPE (conjugate) epilogue, fused attention→FP8 epilogue for `fp8_einsum("bhr,hdr->bhd")` |
| DSA (DeepSeek Sparse Attention) | `deepinfer-dsa` | Lightning indexer: **thunder** weighted-ReLU semantic scores with lightning FP8 quantization (RMSNorm → per-2048-block e4m3/e8m0) exactly matching DeepGEMM's `fp8_fp4_mqa_logits` kernel; **cascade conv** spatial branch with gated RMSNorm; coefficient gating; causal clip (current token included); block-wise top-k pre-selection (top-8 per 64-block → global top-2048, ascending, −1 padded); context extrapolation window; `mscale = 0.1·ln(topk)+1` |
| YOCO + SWA bounded replay | `deepinfer-swa` | YOCO self/cross-decoder layout, `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE={full,compact}` / `KEEP_TOKENS=140` replay planning (suffix must cover SWA window + MTP backtracking), SWA ring cache with `rollback(n)` for speculative backtracking, windowed causal attention |
| DeepGEMM FP8 GEMMs | `deepinfer-fp8` + `deepinfer-gemm` | Activation 1×128 / weight 128×128 tile scaling with UE8M0 scales (+ packed-×4 layout), `fp8_gemm_nt` (D = C + A·Bᵀ), `m_grouped_fp8_gemm_nt_contiguous` (m_indices, M-alignment 128), `m_grouped_fp8_gemm_nt_masked` (masked_m), `fp8_einsum_bhr_hdr_bhd` (recipe (1,1,32)), BF16/F32 reference GEMMs, deterministic parallelism over disjoint outputs |
| DeepSeekMoE | `deepinfer-moe` | Sigmoid routing + **noaux-tc** (e_score_correction_bias affects selection only; weights exclude it), top-k renormalization, optional grouped experts, SwiGLU expert FFNs, shared experts |
| DeepEP expert parallelism | `deepinfer-ep` | Normal-mode dispatch/recv/combine with deterministic `(expert, sender, token, k)` ordering, expert-alignment padding for contiguous grouped GEMMs, per-token-group FP8 payloads, low-latency mode (fixed capacity, `masked_m`, dropped-pair reporting), `EventOverlap`-shaped hooks |
| EPLB | `deepinfer-eplb` | Greedy redundant-expert cloning (maximize load-per-copy), hierarchical + global LPT placement, `phy2log`/`log2phy`/`logcnt` maps, sliding-window expert statistics |
| DSpark (MTP + confidence-scheduled verification) | `deepinfer-mtp` | V3-paper MTP module (eh_proj → shared norm → block → depth norm → shared head), draft chains up to `DEEPINFER_USE_NEXTN_DECODE` (default 5), confidence-scheduled verification (early-stop below `conf_lo`, verify when confident/complete), greedy + stochastic (shared-RNG resample) verification, backtracking math |
| Deterministic sampling | `deepinfer-sampler` | `SamplingParams` with `seq_seed` (the upstream API), pipeline mask → temperature → top-k → top-p → min-p → categorical, per-sequence Philox streams (batch-invariant by construction), greedy tie-break to lowest id |
| Structured outputs | `deepinfer-grammar` | EBNF / JSON Schema / Regex → byte-level Thompson NFA → capped DFA → **XGrammar-style adaptive token masks** (liveness-pruned, per-state cached), strict UTF-8 + surrogate rules, `so_args = {"schema_type": ..., "schema": ...}` |
| Asynchronous scheduling | `deepinfer-scheduler` | Continuous batching, chunked prefill, decode-priority budgets, preemption (recompute), two-thread zero-overlap engine (`EngineOps` boundary, std::mpsc lock-free fast paths, one-step overlap window), batch-invariant determinism |
| Model runtime | `deepinfer-runtime` | V4.1-Flash model (YOCO split: DSA+MLA self-decoder → cross-attn + SWA cross-decoder, MoE FFNs), HF-style `config.json`, safetensors checkpoint loading with full validation, BPE/byte tokenizers, MTP-integrated `generate()`, structured-output-constrained decoding, `examples/offline_inference.rs` (mirrors upstream `offline_inference.py`) |

## Workspace layout

```
crates/
├── core       dtypes, tensors, FP8/BF16/FP4 codecs, Philox RNG, RoPE+YaRN,
│              RMSNorm, JSON, DEEPINFER_* env, safetensors, tokenizers,
│              thread pool, property-test harness
├── fp8        quantization recipes (DeepGEMM / FlashMLA V4.1 cache formats)
├── gemm       FP8/BF16/F32 GEMMs: dense, grouped-contiguous, grouped-masked, einsum
├── mla        FlashMLA attention + paged quantized KV cache
├── dsa        lightning indexer + top-k selection + mscale
├── swa        YOCO layout, bounded-replay planner, SWA ring + attention
├── moe        DeepSeekMoE router + experts
├── eplb       redundant experts + balanced placement
├── ep         DeepEP dispatch/combine (normal + low-latency)
├── mtp        MTP draft + DSpark verification
├── sampler    deterministic batch-invariant sampling
├── grammar    EBNF/JSON-Schema/Regex → DFA → adaptive token masks
├── scheduler  continuous batching + zero-overlap async engine
├── runtime    the DeepSeek-V4.1-Flash model + generate + checkpoint + example
│              (+ DeviceModel: the GPU forward path, decode graphs, paged serve)
├── gpu        dlopen CUDA/HIP driver + NVRTC/hipRTC JIT layer
├── kernels    embedded CUDA/HIP kernels + typed launchers (the operator layer)
└── server     OpenAI-compatible HTTP serving: SSE streaming, continuous
               batching, DSpark speculative decoding with per-request
               rollback, prefix caching, adaptive draft depth, grammar-
               constrained decoding, Prometheus metrics
```

## Design principles

1. **Zero dependencies, zero `unsafe`.** The workspace builds with the standard library only — `#![forbid(unsafe_code)]` is enforced workspace-wide — eliminating the entire supply-chain attack surface. (Upstream DeepInfer's open-source release is offline inference; no async runtime is needed, so none is pulled in.)
2. **Determinism everywhere.** Fixed accumulation order, total-order tie-breaking, counter-based Philox RNG, scheduling independent of wall-clock: identical inputs produce bit-identical outputs — the CPU realization of upstream's *batch-invariant inference* and *deterministic sampling*.
3. **Security as a boundary.** Every parser (JSON, safetensors, tokenizer, grammar, env) is depth/size-bounded and panic-free on hostile input: malformed data yields `Err`, never unbounded allocation. The grammar compiler (untrusted user input by design) caps NFA nodes (200k), DFA states (65 536), repetition counts and recursion depth — `a{1000000}` and `(a|aa){40}`-style bombs fail fast with `CapacityExceeded`.
4. **Fidelity over speed.** Each crate documents its upstream source (paper section / repo file) and the exact deviations (e.g. CPU accumulation order vs GPU MMA order; documented sound-but-conservative JSON-Schema choices).

## Getting started

```bash
# Tiny deterministic self-test (random-init weights — no checkpoint):
cargo run -p deepinfer-runtime --example offline_inference -- --tiny

# Full checkpoint inference (DeepSeek-V4.1-Flash layout):
DEEPINFER_HF_MODEL_DIR=/path/to/DeepSeek-V4.1-Flash-1017 \
  cargo run --release -p deepinfer-runtime --example offline_inference
```

Environment contract (mirrors upstream): `DEEPINFER_ACCELERATOR`, `DEEPINFER_HF_MODEL_DIR`, `DEEPINFER_ENABLE_MTP_VARLEN`, `DEEPINFER_USE_NEXTN_DECODE`, `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE`, `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS`, `DEEPINFER_BATCH_INVARIANT`.

## Serving (OpenAI-compatible HTTP API)

The `deepinfer-server` binary serves the model over the wire — still
with **zero external dependencies** (hand-rolled HTTP/1.1 + SSE on
`std::net`, the workspace JSON, the workspace scheduler):

```bash
# Tiny self-check (deterministic random weights; proves the whole wire path):
cargo run -p deepinfer-server --bin deepinfer-server -- --tiny --port 8080

# Serve a checkpoint on the CPU golden engine:
cargo run --release -p deepinfer-server --bin deepinfer-server -- \
    --model /path/to/DeepSeek-V4.1-Flash-1017

# Serve on the GPU (paged serve engine, FP8 auto):
cargo run --release -p deepinfer-server --bin deepinfer-server -- \
    --model /path/to/DeepSeek-V4.1-Flash-1017 --gpu
```

Then any OpenAI client works:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepinfer-rs","messages":[{"role":"user","content":"Hi"}],"max_tokens":64}'

# Streaming:
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hi"}],"stream":true}'

# Grammar-constrained output (JSON):
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Give me JSON"}],"response_format":{"type":"json_object"}}'
```

Endpoints: `GET /health` (engine status, degraded 503 after a failure),
`GET /v1/models`, `POST /v1/chat/completions` (stream + non-stream,
stop strings, `response_format` JSON Schema, native
`structured_output` for EBNF/regex), `POST /v1/completions` (text or
token-id prompts), `GET /metrics` (Prometheus: TTFT/latency
histograms, token throughput, queue depth), `POST /admin/shutdown`
(graceful drain).

Operational knobs: config file (`--config`),
`DEEPINFER_SERVER_{HOST,PORT,API_KEY,MODEL_NAME,MAX_TOKENS_CAP,MAX_PROMPT_TOKENS,REQUEST_TIMEOUT_SECS,MAX_CONNECTIONS,MAX_BODY_BYTES,SCHED_*,SPEC_DRAFT_LEN,SPEC_ADAPTIVE,PREFIX_CACHE_TOKENS}`.
Bearer auth gates `/v1/*` when `DEEPINFER_SERVER_API_KEY` is set. A
Dockerfile is provided (`docker run --init` for clean SIGTERM drain).

**Single-process serving.** The server is one OS process, end to end:
weights load, tokenizer, engine threads (coordinator + engine over
`std::thread`), the accept loop and every connection worker live in the
same address space — no forks, no subprocesses, no IPC rendezvous, no
external runtime. (The multi-rank symmetric-EP paths exist at the
operator layer for offline multi-GPU runs; the HTTP server never uses
them.) This is the deployment shape the `--tiny` self-check exercises:
`deepinfer-server --tiny` proves the whole wire path in one process,
and the same binary serves a full checkpoint.

**The byte tokenizer.** `--tokenizer auto` (default) loads the
checkpoint's `tokenizer.json`; `--tokenizer byte` forces byte-mode
serving: a 256-token byte-level alphabet plus the DeepSeek
conversation pieces (`<｜begin▁of▁sentence｜>`, `<｜end▁of▁sentence｜>`,
`<｜User｜>`, `<｜Assistant｜>`) as single tokens (ids 256-259), so the
chat template costs 4 tokens instead of ~64 bytes of context. Any
checkpoint serves without a `tokenizer.json` — the sampling surface is
capped to the tokenizer's alphabet (the engine masks unreachable ids
out of plain sampling, spec drafting and verification), and
structured-output grammar masking uses exact per-token bytes (`0xFF`
masks as the byte `0xFF`, never U+FFFD). Quality follows the model's
training tokenizer — byte mode is the universal, always-works path
(the direction Meta's Byte Latent Transformer research validates for
byte-level models).

**Prefix caching** (SGLang RadixAttention / vLLM automatic prefix
caching, adapted to this architecture): prompts completing prefill are
snapshotted (exact f32 KV shadow rows, DSA indexer rows, SWA rings)
into an LRU-bounded radix tree over token ids; a later request sharing
a prefix is admitted past it (`Scheduler::skip_prefill`) and the engine
restores the cached state before prefilling only the suffix — a
memory-bound copy instead of a compute-bound forward, bit-identical at
the same chunk boundary (pinned by the runtime
`prefix_snapshot_restore_is_bit_identical` test). Multi-turn chats,
few-shot prompts with shared system preambles and agentic loops are
the workloads that win. Budget: `DEEPINFER_SERVER_PREFIX_CACHE_TOKENS`
(default 4096 tokens, 0 = off); observability:
`deepinfer_prefix_cache_{hits,reused_tokens,inserts,entries,tokens}`.
A scheduler eviction mid-prefill is handled exactly (the engine rolls
the surplus rows back before re-prefilling — the rewind guard also
closes a latent double-append on prefill restarts). On the GPU
(`--gpu`), the same budget drives an engine-internal prefix cache
instead: a block-granular radix tree over the paged serve pools with
reference-counted page sharing, copy-on-write on deep rollbacks, and
an allocation pressure valve (see `docs/GPU.md`) — metrics land on the
same `deepinfer_prefix_cache_*` family plus
`deepinfer_prefix_cache_evictions_total`.

**Adaptive speculative depth** (vLLM dynamic-speculative-decoding-style):
each request tracks an acceptance EMA and retargets its draft depth
toward `round(ema·(depth+1)) − 1` — a rejecting workload decays to
single-token rounds (speculation stops costing throughput as
concurrency rises), an accepting one keeps the full chain. Greedy
parity is unaffected (verification is prefix-exact for any depth);
`DEEPINFER_SERVER_SPEC_ADAPTIVE=false` pins the fixed depth. The
scheduler still bills the configured `depth + 1` rows per step
(conservative budget accounting).

The serving engine is the same continuous-batching scheduler as the
offline path, with per-request sampling state, grammar masking,
cancellation on client disconnect, per-request error isolation, and
slot recycling with exact cache teardown — CI asserts the served
greedy stream is **bit-identical to `generate()`**.
See `docs/UZU-COMPARISON.md` for how this stacks up against uzu, and
the `deepinfer-server` crate docs for the architecture.

**DSpark speculative serving.** When the model carries MTP (NEXTN)
weights, every decode step runs a speculative round per request:
draft up to `DEEPINFER_SERVER_SPEC_DRAFT_LEN` (default 5, 0 = off)
tokens with the MTP head under the DSpark confidence scheduler,
verify the chain against the target in one call, roll the rejected
tail back out of the paged caches (per-sequence, page-recycling), and
emit the accepted prefix plus the bonus token; the bonus decodes of
the whole batch share one `serve_decode` launch on the GPU engine.
Structured-output requests get grammar-aware speculation (draft-time
masking + masked verification rows + DFA state snapshots) — beyond
the offline path, which disables speculation under a grammar. Greedy
parity is pinned in CI: **spec serve == plain serve == `generate()`**,
with and without grammars, under batching. `deepinfer_spec_*`
Prometheus series track rounds, drafts, accepted/bonus tokens and
rollbacks.

## Tests

534 tests across 17 crates: unit + property-based (deterministic seeded harness), including golden-numerics checks against paper formulas, exact-semantics checks against the FlashMLA README pseudocode, brute-force grammar/mask validation, regex-vs-naive-backtracker properties, dispatch/combine round-trips, MTP greedy-consistency (speculative ON == speculative OFF under temperature 0, offline **and** on the batched serve path), and end-to-end generation determinism.

## Scope notes

* CPU reference implementation: performance is deliberately secondary to exact semantics; the kernel *structure* (tile scaling, chunked split-K merges, paged layouts, masked grouped GEMMs) mirrors upstream so a GPU backend can slot in behind the same APIs.
* Cross-decoder attention is capped to a 512-token recent window of the self-decoder KV (documented CPU-tractability cap).
* Expert parallelism runs single-process (in-process mailbox transport); the protocol semantics (ordering, alignment, capacities, hooks) match DeepEP.
* Not included (same as the upstream release): serving-layer KV-cache system, prefill/decode disaggregation, cross-replica load balancing.

## License

Apache License 2.0, matching upstream DeepInfer.

## Citation

```bibtex
@misc{deepinfer2026,
  title  = {DeepInfer: High-performance, large-scale inference framework for DeepSeek models},
  author = {DeepSeek-AI},
  year   = {2026},
  howpublished = {\url{https://github.com/deepseek-ai/DeepInfer}},
}
```

## GPU inference (GB200 NVL72 / GB300 / AMD MI335X)

`deepinfer-gpu` + `deepinfer-kernels` add the GPU plane on top of the CPU
reference: runtime-dlopened CUDA/HIP (zero build-time GPU deps), NVRTC/
hipRTC JIT with a content-addressed disk cache, and the complete operator
inventory for V4.1-Flash inference (FP8/BF16 GEMMs incl. grouped-contiguous
MoE, FlashMLA decode split-KV with FP8/FP4 KV caches, DSA lightning indexer
+ block top-K, MoE routing/dispatch/combine, RoPE/O-RoPE, quantization).
Numerics are golden-tested against the CPU crates on-device. Targets:
sm_100 (B200), sm_103 (B300), gfx942/gfx950 (Instinct). Details, safety
model and upgrade paths: [docs/GPU.md](docs/GPU.md).

### GPU execution (deepinfer-gpu / deepinfer-kernels / DeviceModel)

`deepinfer-runtime::DeviceModel` runs the full V4.1-Flash forward pass on
CUDA (GB200 NVL72 `sm_100`, GB300 `sm_103`) or HIP (MI335X-class) with
runtime-JIT-compiled kernels, per-GEMM F32/FP8/BF16/NVFP4 precision, FP8/FP4 KV
caches (including the dedicated FP4 decode kernel), CUDA-graph decode
capture/replay, and single-process NVLink peer collectives for expert
parallelism. The serving plane adds device-side sampling, prefill/verify/
decode serve graphs, a block-radix **GPU prefix cache** with a host-memory
L2 tier, **cp.async-pipelined attention** (bit-identical to the direct
walks) and vectorized DSA score kernels, the **fused symmetric-heap EP**
(grouped masked GEMM directly over the heaps, completion-table combine,
EPLB-driven routing), and an opt-in **persistent self-layer megakernel**
(`DEEPINFER_MEGAKERNEL=1`). Kernel sources are validated offline by the
NVRTC+ptxas compile gate (`scripts/check-cuda.py`, sm_80/90/100/103).
Binaries build and test on GPU-less machines (device tests auto-skip).
See `docs/GPU.md` for the architecture and operator inventory.

### On-hardware acceptance (golden runs)

```bash
cargo run --release -p deepinfer-runtime --example golden_run -- --list
cargo run --release -p deepinfer-runtime --example golden_run -- --record site.json
cargo run --release -p deepinfer-runtime --example golden_run -- --verify site.json
```

Records the capability report plus the full check battery (CPU-vs-GPU
model agreement, CUDA-graph replay/rollback exactness, tcgen05 vs portable
GEMM, the paged multi-sequence serve engine, the symmetric-buffer EP, and
timings) as a JSON artifact; `--verify` re-runs against it.
