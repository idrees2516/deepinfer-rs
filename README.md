# DeepInfer-rs

**A pure-Rust, CPU-reference implementation of [DeepInfer](https://github.com/deepseek-ai/DeepInfer) — DeepSeek's high-performance, large-scale inference framework — targeting the DeepSeek-V4.1-Flash architecture.**

This workspace ports the full DeepInfer release feature set to dependency-free, `unsafe`-free Rust, following the published DeepSeek ground truth (FlashMLA, DSA, DeepGEMM, DeepEP, EPLB repositories and the DeepInfer README) at algorithmic fidelity: every component implements the *exact* upstream math and protocol semantics on CPU, with deterministic numerics.

```
cargo test --workspace     # 445 tests, all green
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
└── runtime    the DeepSeek-V4.1-Flash model + generate + checkpoint + example
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

## Tests

415 tests across 14 crates: unit + property-based (deterministic seeded harness), including golden-numerics checks against paper formulas, exact-semantics checks against the FlashMLA README pseudocode, brute-force grammar/mask validation, regex-vs-naive-backtracker properties, dispatch/combine round-trips, MTP greedy-consistency (speculative ON == speculative OFF under temperature 0), and end-to-end generation determinism.

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

### Pure-Rust kernels: `deepinfer-cutile` (primary NVIDIA path)

Every inference operator is also implemented **entirely in Rust** on
[NVIDIA CUDA Rust](https://github.com/NVIDIA/cuda-rust) / [cuTile Rust]
(https://nvidia.github.io/cuda-rust/cutile/): kernels are `#[cutile::module]`
DSL functions compiled to CUDA Tile IR and JIT-assembled on the target GPU —
no C/C++, no NVRTC, no `nvcc`. Targets sm_80 … sm_121, tuned for sm_100
(GB200 NVL72) and sm_103 (GB300/B300 NVL72). Highlights: FP8 block-scaled
GEMMs via `mmaf_scaled` (the Blackwell tcgen05 MXFP8 path, DeepGEMM's
1×128/128×128 UE8M0 recipe), MLA prefill flash + paged split-KV decode
(BF16 and FP8 KV caches, attn-sink, exact LSE semantics), DSA thunder
indexer with exact `(score desc, pos asc)` top-K, noaux-tc MoE
route/dispatch/combine, FP4/FP8 KV appends, and ties-lowest argmax.
Two-plane validated like everything else: `cargo test -p deepinfer-cutile`
green without any GPU; `--features cuda` adds the kernels (build-time
bindgen needs CUDA 13 headers + libclang — see `scripts/gpu-env.sh` for a
rootless recipe); `--test golden` runs the CPU-reference golden tests on a
real GPU.

### C++-via-JIT fallback: `deepinfer-gpu` + `deepinfer-kernels`

The earlier GPU plane — runtime-dlopened CUDA/HIP (zero build-time GPU
deps), NVRTC/hipRTC JIT with a content-addressed disk cache, and the same
complete operator inventory for V4.1-Flash. This remains the **AMD MI335X**
path (HIP has no CUDA Rust story). Numerics are golden-tested against the
CPU crates on-device; targets sm_100/sm_103 and gfx942/gfx950. Details,
safety model and upgrade paths: [docs/GPU.md](docs/GPU.md).
