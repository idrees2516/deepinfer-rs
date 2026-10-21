# DeepInfer-rs Interface Specification (contract for all crates)

This document is **binding**: every crate must expose exactly the public APIs
described here (signatures + semantics). Internal helpers are free-form.
When in doubt, follow the DeepSeek upstream references cited per crate.

## Global conventions (ALL crates)

1. **`#![forbid(unsafe_code)]`** — already enforced workspace-wide. No `unsafe`, ever.
2. **Zero external dependencies.** Only `deepinfer-core` (+ listed sibling crates) and `std`.
3. **Errors**: return `deepinfer_core::Result<T>`; use `deepinfer_core::ensure!(cond, Variant, "fmt")`.
   NEVER panic on user input. Panics are acceptable only in `#[cfg(test)]` code.
4. **Determinism**: fixed accumulation order (sequential f32 unless stated), total-order
   tie-breaking, no timing/hash-order dependencies. Same input → bit-identical output.
5. **Validation**: every public fn validates shapes/indices/ranges and returns `Err` on misuse.
   Guard against DoS: bound all allocations by checked math.
6. **Docs**: every public item has a doc comment (`#![warn(missing_docs)]` is on). Cite the
   upstream source of the algorithm (paper section / repo file) in the module docs.
7. **Tests**: unit tests + property tests using `deepinfer_core::prop::prop`
   (e.g. `prop("mymath", 0xSEED, 200, |i, rng| {...})`) and
   `assert_close_slice` from `deepinfer_core::prop`. Every crate must `cargo test -p <crate>` green.
8. **Layout**: crate `deepinfer-X` lives in `crates/X`. Add sibling deps to `Cargo.toml` only as
   listed below (never create cycles).
9. Fill in the real `description` in each crate's Cargo.toml.
10. Module doc header format:
    ```rust
    //! <One-paragraph what this implements, citing the upstream paper/repo.>
    //!
    //! <Optional: fidelity notes / deviations.>
    ```

## deepinfer-core (DONE — do not modify)

You may use anything from `deepinfer_core::prelude` (re-exports below) plus:

- `DType`, `Tensor` (`from_f32`, `from_raw`, `zeros`, `as_f32_flat`, `get_f32`, `shape`, `numel`, `bytes`, `rows`)
- `Error`, `Result`, `ensure!` macro; `Error::{invalid, internal, checkpoint}`
- `format`: `bf16_to_f32, f32_to_bf16_rne, f16_to_f32, f32_to_f16_rne, e4m3_to_f32, f32_to_e4m3_rne(x, saturate),
  e5m2_to_f32, f32_to_e5m2_rne(x, saturate), e8m0_to_f32, f32_to_e8m0(x, E8M0Round::{Floor,Ceil}) -> Result<u8>,
  e2m1_to_f32, f32_to_e2m1_rne, E2M1_MAGNITUDES`
- `rng::Philox`: `new(seed: u64, stream: u64)`, `next_u32`, `next_f32` ([0,1)), `next_f64`,
  `uniform_f32(lo,hi)`, `uniform_usize(lo,hi)`, `normal_f32`, `fill_uniform(&mut [f32], lo, hi)`,
  `fill_normal`, `next_u64_from_two`
- `math`: `softmax_inplace(&mut [f32])`, `logsumexp(&[f32]) -> f32`, `dot(&[f32], &[f32]) -> f32`,
  `argmax(&[f32]) -> Option<usize>` (ties → lowest), `top_k_desc(&[f32], k) -> TopK{indices: Vec<u32>, values: Vec<f32>}`
  (order: value desc, index asc), `sample_categorical(&[f32], u: f32) -> Option<usize>`,
  `axpy(alpha, x, &mut y)`, `scale(&mut [f32], s)`
- `rope`: `RopeStyle::{Interleaved, HalfSplit}`, `YarnConfig{factor, original_max_position_embeddings,
  beta_fast, beta_slow, mscale, mscale_all_dim}` (Default = V3 values), `Rope::new(dim, style, base,
  scaling: Option<YarnConfig>, max_position) -> Result<Rope>`; methods `apply(&[u32], &mut [f32], row_len)`,
  `apply_at(positions, x, row_len, offset)`, `apply_conjugate_at(...)` (O-RoPE), `attention_factor() -> f64`,
  `max_position()`
- `norm`: `rmsnorm(&mut [f32], w: &[f32], eps)`, `rmsnorm_into(x, w, eps, out, offset, len)`,
  `rms_mean_square(&[f32]) -> f32`
- `mask::TokenMask`: `all_ones(vocab)`, `zeros(vocab)`, `set(u32)`, `clear(u32)`, `test(u32) -> bool`,
  `intersect(&mut, &TokenMask)`, `count_ones()`, `any()`, `iter_ones()`, `vocab_size()`
- `json::Json` (parse/serialize, `get`, `entries`, `as_str/bool/f64/u64/i64/arr`, `str_field`,
  `u64_field`, `f64_field`, `bool_field`)
- `env::{Env, EnvReport, Accelerator, YocoTruncationMode}` (`Env::from_env()`, `Env::default()`:
  nextn=5, keep_tokens=140, mode=Full)
- `safetensors::{SafeTensorsFile (open, names, meta, tensor), CheckpointDir (open, names, tensor, shard_count, dir)}`
- `tokenizer::{Tokenizer (vocab_size/encode/decode/eos_token_id/bos_token_id), ByteTokenizer::new(eos), BpeTokenizer::from_tokenizer_json(&Json)}`
- `sync::{ThreadPool::new(n), spawn, par_for(total, min_chunk, Fn(usize, usize) + Send + Sync + 'static), global_pool()}`
- `prop::{prop(label, seed, iters, Fn(u32, &mut Philox) -> Result<()>), assert_close, assert_close_slice,
  random_matrix(rng, rows, cols, mag), random_vec(rng, n, mag)}`

## deepinfer-fp8  (deps: core)

Quantization recipes mirroring DeepGEMM / tile-kernels `quant` and the FlashMLA V4.1 KV-cache
formats. Reference: DeepGEMM README (scaling layouts), FlashMLA README (V4.1 528B/288B formats).

```rust
/// Per-token-group E4M3 quantization with UE8M0 scales (DeepGEMM activation recipe,
/// granularity 1x128 by default). Returns (values, scales) where scales.len() == rows*ceil(k/group).
pub fn quantize_per_token_group(x: &[f32], cols: usize, group: usize)
    -> Result<(Vec<u8>, Vec<u8>)>;   // row-major [rows][k/group], e8m0 bytes (scale = 2^ceil(log2(amax/448)))
/// Dequantize the above. out[i] = e4m3_to_f32(v[i]) * e8m0_to_f32(sf[i/group]).
pub fn dequantize_per_token_group(v: &[u8], sf: &[u8], cols: usize, group: usize) -> Result<Vec<f32>>;

/// Per-block weight quantization (DeepGEMM weight recipe, granularity 128x128 by default).
/// b is [n, k] row-major; scales are [(n/block_n).ceil()][ceil(k/block_k)] e8m0.
pub fn quantize_weight_per_block(b: &[f32], n: usize, k: usize, block_n: usize, block_k: usize)
    -> Result<(Vec<u8>, Vec<u8>)>;
pub fn dequantize_weight_per_block(v: &[u8], sf: &[u8], n: usize, k: usize, block_n: usize, block_k: usize)
    -> Result<Vec<f32>>;

/// Pack 4 UE8M0 bytes into one u32 (DeepGEMM SM100 packed-sf layout, little-endian order).
pub fn pack_ue8m0_x4(scales: &[u8]) -> Result<Vec<u32>>;
pub fn unpack_ue8m0_x4(packed: &[u32]) -> Vec<u8>;

/// FlashMLA V4.1 FP8 KV-cache row: 512 f32 dims in, 528 bytes out
/// (512 e4m3 values then 16 e8m0 scales, one per 32 consecutive dims).
pub fn quantize_kv_cache_v41_fp8(dims: &[f32]) -> Result<[u8; 528]>;
pub fn dequantize_kv_cache_v41_fp8(bytes: &[u8; 528]) -> [f32; 512];

/// FlashMLA V4.1 FP4 KV-cache row: 512 f32 dims in, 288 bytes out
/// (256 bytes of packed e2m1, even-index value in the LOW nibble, then 32 e4m3
/// scales, one per 16 consecutive values). FP4 scale selection: amax/6, rounded
/// to E4M3 with RNE, floor to at least the smallest positive normal.
pub fn quantize_kv_cache_v41_fp4(dims: &[f32]) -> Result<[u8; 288]>;
pub fn dequantize_kv_cache_v41_fp4(bytes: &[u8; 288]) -> [f32; 512];

/// Per-token cast used by the fused attention epilogue (tile_kernels.quant.per_token_cast
/// semantics, granularity `group`, e4m3 + e8m0 scales) — convenience wrapper over
/// quantize_per_token_group for a single row.
pub fn per_token_cast_row(x: &[f32], group: usize) -> Result<(Vec<u8>, Vec<u8>)>;
```

Tests: byte-layout exactness vs the FlashMLA spec (528/288 sizes, scale positions, nibble order),
round-trip error bounds (fp8 rel err <= ~0.13 on random data, fp4 abs err <= 3.0 + rel bounds),
group/edge cases (non-multiple-of-group lengths → Err), determinism, NaN/Inf saturation
(quantize clamps, never panics), property tests with prop().

## deepinfer-gemm  (deps: core, fp8)

DeepGEMM-equivalent GEMMs, NT layout (`D = C + A @ B^T`), FP8 with fine-grained scaling
(activations 1x128, weights 128x128, UE8M0), f32 accumulation, plus BF16 reference.

```rust
/// The M/K alignment required for the contiguous grouped layout.
pub fn get_mk_alignment_for_contiguous_layout() -> usize; // 128

/// Dense FP8 NT GEMM. a: [m, k] e4m3 bytes; a_sf: [m, ceil(k/128)] e8m0;
/// b: [n, k] e4m3; b_sf: [ceil(n/128), ceil(k/128)] e8m0; c: optional [m, n] addend.
/// out[i][j] = c + sum over k-blocks of  sf_a(i,kb) * sf_b(jb,kb) * sum_{k in block} deq(a)*deq(b).
pub fn fp8_gemm_nt(a: &[u8], a_sf: &[u8], b: &[u8], b_sf: &[u8],
                   m: usize, n: usize, k: usize, c: Option<&[f32]>) -> Result<Vec<f32>>;

/// BF16 NT GEMM with f32 accumulation (reference path). a/b are bf16 bit patterns.
pub fn bf16_gemm_nt(a: &[u16], b: &[u16], m: usize, n: usize, k: usize,
                    c: Option<&[f32]>) -> Result<Vec<f32>>;

/// F32 NT GEMM (naive, sequential accumulation — the golden reference).
pub fn f32_gemm_nt(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Result<Vec<f32>>;

/// Grouped-contiguous FP8 GEMM (MoE prefill): groups stacked along M.
/// m_indices[i] = expert (group) id of row i; rows of the same group must be contiguous
/// and each group's row count must be a multiple of the alignment (pad, verify + Err).
/// b: [num_groups, n, k]; b_sf: [num_groups, ceil(n/128), ceil(k/128)].
pub fn m_grouped_fp8_gemm_nt_contiguous(a: &[u8], a_sf: &[u8], b: &[u8], b_sf: &[u8],
        m_indices: &[u32], m: usize, n: usize, k: usize, num_groups: usize) -> Result<Vec<f32>>;

/// Grouped-masked FP8 GEMM (MoE decode / CUDA-graph style): fixed capacity per group.
/// a: [num_groups, m_max, k] (a_sf: [num_groups, m_max, ceil(k/128)]),
/// b: [num_groups, n, k], masked_m[g] = valid rows of group g (rest ignored).
/// Computes rows [0, masked_m[g]) of each group; output [num_groups, m_max, n] (zeros beyond).
pub fn m_grouped_fp8_gemm_nt_masked(a: &[u8], a_sf: &[u8], b: &[u8], b_sf: &[u8],
        masked_m: &[usize], m_max: usize, n: usize, k: usize, num_groups: usize) -> Result<Vec<f32>>;

/// The einsum used by the FlashMLA fused-kernel epilogue: "bhr,hdr->bhd".
/// q: [b, h, r] e4m3 with q_sf: [b, h, ceil(r/32)] e8m0 (recipe (1,1,32));
/// w: [h, d, r] e4m3 with w_sf: [h, d, ceil(r/32)] e4m3-as-f32-dequantized? NO —
/// w_sf is e4m3 bytes per (h, d, r/32) (FlashMLA permuted Wv uses e4m3 scales).
/// out[b][h][d] = sum over r-blocks of sf_q * sf_w * sum_{r in block} deq(q)*deq(w).
pub fn fp8_einsum_bhr_hdr_bhd(q: &[u8], q_sf: &[u8], w: &[u8], w_sf: &[u8],
        b: usize, h: usize, d: usize, r: usize, group: usize) -> Result<Vec<f32>>;

/// Pad token count to the contiguous-layout alignment.
pub fn pad_to_alignment(tokens: usize, alignment: usize) -> usize;
```

Semantics notes:
- Per-tile scaling: for each output element, the k axis is split into blocks of 128;
  `a` scales vary per (row, k-block); `b` scales vary per (n-block of 128, k-block).
  Multiply the dequantized products by `2^(sa+sb)` per block, accumulate in f32.
- Determinism: parallelize ONLY over disjoint output rows via `deepinfer_core::sync::global_pool()
  .par_for(...)`; within an output element accumulate sequentially in fixed k order.
- Validate: all lens match products; k % group == 0 (else Err); n % 128 == 0 for weight scales
  (or ceil semantics documented); masked_m[g] <= m_max.

Tests: fp8 vs f32 reference within 2-3% rel tolerance on random data; grouped-contiguous vs
loop of dense GEMMs; masked vs dense-with-zeros; einsum vs f32 reference; determinism (run twice,
bit-identical); property tests.

## deepinfer-mla  (deps: core, fp8, gemm)

FlashMLA-equivalent Multi-head Latent Attention for DeepSeek-V4.1
(h_q=64, head_dim/cache dim=512 = 448 latent + 64 rope, q_lora=1536, n_wv_group=8,
o_lora_rank=512, rope interleaved, page 64, FP8/FP4 caches, DSA indices, attn_sink).

```rust
pub enum CacheFormat { Bf16, Fp8Group32, Fp4Group16 }

pub struct MlaConfig {
    pub hidden_size: usize,      // e.g. 4096
    pub q_lora_rank: usize,      // 1536 (V4.1)
    pub n_heads: usize,          // 64
    pub head_dim: usize,         // 512 (absorbed q per head = latent 448 + rope 64)
    pub rope_dim: usize,         // 64
    pub o_groups: usize,         // 8
    pub o_lora_rank: usize,      // 512
    pub rope_base: f64,          // 10000
    pub yarn: Option<YarnConfig>,
    pub rms_eps: f32,            // 1e-4 for V4.1
    pub sm_scale: f32,           // default 1/sqrt(head_dim) * mscale when set by caller
    pub cache: CacheFormat,
    pub max_pages: usize,
}
impl MlaConfig { pub fn v41_default(max_position: usize) -> MlaConfig; }

pub struct MlaWeights {
    // [q_lora, hidden]
    pub q_a_proj: Vec<f32>,
    // [n_heads * head_dim, q_lora] — Q_b is the ABSORBED query projection (V4.1 folds W_UK here)
    pub q_b_proj: Vec<f32>,
    // [(kv_lora + rope) = head_dim, hidden]  (latent 448 | rope 64)
    pub kv_a_proj: Vec<f32>,
    // [o_groups * o_lora_rank, (n_heads / o_groups) * head_dim]
    pub wv_proj: Vec<f32>,
    // [hidden, o_groups * o_lora_rank]
    pub wo_proj: Vec<f32>,
    // V4-only Q norm; None for V4.1
    pub q_norm: Option<Vec<f32>>,
    // [head_dim - rope_dim] RMSNorm over the latent part of the KV projection
    pub kv_norm: Vec<f32>,
    // optional per-head attention sink logits [n_heads] (FlashMLA `attn_sink`)
    pub attn_sink: Option<Vec<f32>>,
    pub rope: Rope, // built from config (interleaved, rope slice = LAST rope_dim of head_dim)
}

impl MlaWeights {
    pub fn from_config(cfg: &MlaConfig, rng: &mut Philox) -> MlaWeights; // random init for tests
    /// Q_a projection + optional norm + Q_b: returns [s, n_heads, head_dim] (rope NOT yet applied)
    pub fn project_q(&self, hidden: &[f32], s: usize, out: &mut [f32]) -> Result<()>;
    /// KV projection: returns [s, head_dim] rows = [latent | rope-raw] with RMSNorm applied
    /// to the latent slice only (rope applied later by caller via self.rope).
    pub fn project_kv(&self, hidden: &[f32], s: usize, out: &mut [f32]) -> Result<()>;
    /// Wv + Wo: input attn out [s, n_heads, head_dim] (after O-RoPE conjugate) -> [s, hidden]
    pub fn project_o(&self, attn_out: &[f32], s: usize, out: &mut [f32]) -> Result<()>;
}

/// Paged KV cache. Pages hold [page_size, bytes_per_token] for the configured format;
/// per-sequence page lists give logical order.
pub struct MlaPagedCache { /* ... */ }
impl MlaPagedCache {
    pub fn new(format: CacheFormat, page_size: usize, max_pages: usize) -> Self; // page_size = 64
    pub fn format(&self) -> CacheFormat;
    pub fn page_size(&self) -> usize;
    /// Appends one token row ([head_dim] f32, rope already applied) for `seq`, allocating pages
    /// as needed; returns Err (CapacityExceeded) when out of pages.
    pub fn append(&mut self, seq: u32, row: &[f32]) -> Result<()>;
    /// Number of tokens stored for `seq`.
    pub fn seq_len(&self, seq: u32) -> usize;
    /// Block table (page ids in logical order).
    pub fn block_table(&self, seq: u32) -> &[u32];
    /// Encodes a logical token position of `seq` into the FlashMLA physical index
    /// `page_id * page_size + offset`. Returns Err if pos >= len.
    pub fn physical_index(&self, seq: u32, pos: usize) -> Result<i32>;
    /// Dequantizes one stored token row by physical index.
    pub fn row_f32(&self, phys: i32) -> Result<Vec<f32>>; // [head_dim]
    /// Total stored tokens.
    pub fn total_tokens(&self) -> usize;
}

/// Sparse decode: EXACT FlashMLA semantics.
/// q: [b][s_q][n_heads][head_dim] with rope already applied (and mscale folded in sm_scale).
/// indices: [b][s_q][topk] physical indices (-1 invalid).
/// Returns out [b][s_q][n_heads][head_dim] (attention over latents; O-RoPE NOT applied here)
/// and lse [b][n_heads][s_q]. Queries with no valid index: zero output, lse = +inf.
/// Implementation must process KV in page-aligned chunks with running-max online softmax
/// (flash-decoding style) and merge partials — matching FlashMLA's split-K structure.
pub fn flash_mla_decode(q: &[f32], cache: &MlaPagedCache,
        indices: &[i32], b: usize, s_q: usize, topk: usize,
        sm_scale: f32, attn_sink: Option<&[f32]>,
        out: &mut [f32], lse: &mut [f32]) -> Result<()>;

/// Sparse prefill: EXACT equivalent of the FlashMLA README pseudocode:
/// focused_kv = kv[indices]; P = (Q @ focused_kv^T) * sm_scale; invalid -> -inf;
/// out = softmax(P) @ focused_kv; plus attn_sink epilogue out *= exp(lse)/(exp(lse)+exp(sink)).
/// q: [s_q, h, d_qk]; kv: [s_kv, d_qk]; indices: [s_q, topk] (int32; -1 or >= s_kv invalid);
/// topk_length: optional per-query valid prefix length.
/// Returns (out [s_q*h*d_qk], max_logits [s_q, h], lse [s_q, h]).
pub fn flash_mla_sparse_prefill(q: &[f32], kv: &[f32], indices: &[i32],
        s_q: usize, h: usize, d_qk: usize, s_kv: usize, topk: usize,
        sm_scale: f32, attn_sink: Option<&[f32]>, topk_length: Option<&[i32]>)
    -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)>;

/// Dense (non-sparse) causal prefill over the full KV — used by self-decoder layers
/// without DSA. Same layout as flash_mla_sparse_prefill but attends to all s_kv keys
/// with causal masking (key pos <= query pos + kv_offset).
pub fn flash_mla_dense_prefill(q: &[f32], kv: &[f32], s_q: usize, h: usize,
        d_qk: usize, s_kv: usize, q_offset: usize, sm_scale: f32,
        out: &mut [f32]) -> Result<()>;

/// O-RoPE (conjugate) on the rope slice of each head's output row.
pub fn o_rope_conjugate(rope: &Rope, positions: &[u32], x: &mut [f32],
        rows: usize, row_len: usize) -> Result<()>; // rotates [row_len - rope_dim, row_len)

/// Fused path equivalent (composition): norm-free Q-RoPE already applied by caller; this
/// runs sparse attention then quantizes output to e4m3 + e8m0 per-32 scales (num_per_channels=32)
/// for direct consumption by gemm::fp8_einsum_bhr_hdr_bhd. Returns (data [s*h*d], sf [s*h*(d/32)]).
pub fn fused_attention_out_fp8(...)  // agent picks exact signature; document it
```

Tests: decode vs naive gather+softmax reference (bit-close, tol 1e-4), prefill vs the PyTorch
pseudocode transcription, empty-selection edge case (max=-inf, lse=+inf, zero out), attn_sink
epilogue math, FP8/FP4 cache round-trip through quantization (tolerance per format), online
softmax merge == single-pass, paged layout with wrapped pages, determinism, property tests.

## deepinfer-dsa  (deps: core, fp8)

DeepSeek Sparse Attention lightning indexer + top-k selection + mscale.
References: DSA paper (arXiv:2509.07152), DeepGEMM MQA logits kernel semantics,
FlashMLA DSA sections, V3.2 config (topk 2048, block 64).

```rust
pub struct LightningIndexerConfig {
    pub hidden_size: usize,     // model hidden
    pub num_heads: usize,       // heads for the MQA-form scoring (e.g. 64)
    pub head_dim: usize,        // index dim / num_heads (e.g. 112 -> index dim 7168)
    pub block_size: usize,      // pre-selection block (64)
    pub topk_tokens: usize,     // 2048
    pub preselect_per_block: usize, // 8
    pub quant_block: usize,     // lightning-quant FP8 block (2048)
    pub conv_kernels: Vec<usize>, // cascade depthwise conv kernels, default [3,7,15,31,63,127,255,511]
    pub extrapolation_window: Option<usize>, // None = off; Some(w) restricts candidates to last w keys
}
impl LightningIndexerConfig { pub fn v32_default(hidden_size: usize) -> Self; }

pub struct LightningIndexerWeights {
    pub wq: Vec<f32>,        // [num_heads*head_dim, hidden]
    pub wk: Vec<f32>,        // [num_heads*head_dim, hidden]
    pub weights_proj: Vec<f32>, // [num_heads, num_heads*head_dim] (per-query head weights)
    pub gate_proj: Vec<f32>,    // [2, 2*hidden] -> produces (g_thunder, g_conv) after sigmoid
    pub conv_weights: Vec<Vec<f32>>, // one depthwise kernel per cascade entry
}
impl LightningIndexerWeights { pub fn from_config(cfg: &LightningIndexerConfig, rng: &mut Philox) -> Self; }

/// The "thunder" semantic score (DeepGEMM MQA logits semantics):
/// score(i,j) = sum_h relu( qtilde_i^h . ktilde_j^h ) * w_i^h, where qtilde/ktilde are
/// RMSNormed then FP8-quantized per quant_block tokens with e8m0 scales (lightning quant),
/// and w_i = weights_proj(q_i). Output [s_q, s_kv] (causal clip applied by caller or flag).
pub fn thunder_scores(q: &[f32], k: &[f32], w: &[f32], s_q: usize, s_kv: usize,
        num_heads: usize, head_dim: usize, quant_block: usize, causal: bool) -> Result<Vec<f32>>;

/// The "cascade conv" spatial score: causal depthwise convs over K (sum over cascade kernels),
/// then score(i,j) = <q_i, convfeat_j> (convfeat normalized by gated RMSNorm:
/// convfeat * sigmoid(a . convfeat) / rms(convfeat) — implement gated-RMSNorm with a
/// learnable per-dim gain `conv_norm_gain` derived from weights; document choice).
pub fn conv_scores(q: &[f32], k: &[f32], s_q: usize, s_kv: usize, hidden: usize,
        kernels: &[Vec<f32>], norm_gain: &[f32], causal: bool) -> Result<Vec<f32>>;

/// Coefficient gating: g_i = sigmoid(gate_proj . [q_i; mean_q]) per (thunder, conv);
/// combined = g_t * thunder + g_c * conv (gates sum to 1 via sigmoid pair).
pub fn gate_and_combine(q: &[f32], s_q: usize, hidden: usize, gate_proj: &[f32],
        thunder: &mut [f32], conv: &[f32]) -> Result<()>;

/// Top-k token selection with block-wise pre-selection: per block of block_size keys keep the
/// top preselect_per_block by score, then global top-K (K = topk_tokens, capped at s_kv).
/// Ties: higher score first, then LOWER key position. Returns sorted-ascending positions.
pub fn select_top_k(scores: &[f32], s_q: usize, s_kv: usize, cfg: &LightningIndexerConfig)
    -> Result<Vec<i32>>; // [s_q, topk'] with -1 padding, topk' = min(topk_tokens, s_kv)

/// DSA mscale for sparse attention: 0.1 * ln(topk_tokens) + 1 (V3.2 "msacle").
pub fn mscale(topk_tokens: usize) -> f32;

/// End-to-end indexer forward: q_hidden [s_q, hidden], k_hidden [s_kv, hidden] ->
/// selected key positions [s_q, topk'] (ascending, -1 padded).
pub fn indexer_forward(w: &LightningIndexerWeights, cfg: &LightningIndexerConfig,
        q_hidden: &[f32], k_hidden: &[f32], s_q: usize, s_kv: usize) -> Result<Vec<i32>>;
```

Tests: thunder matches a naive double loop exactly (on small dims with quantization disabled
via huge quant_block); causal clip; gate in [0,1] and combination; select_top_k picks the true
top-k (vs sort); tie-breaking; extrapolation window; determinism; property tests.

## deepinfer-swa  (deps: core)

YOCO decoder layout + SWA bounded replay (DeepInfer env contract:
DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE / _KEEP_TOKENS).

```rust
pub struct YocoLayout { pub self_layers: usize, pub cross_layers: usize }
impl YocoLayout { pub fn total(&self) -> usize; }

pub struct SwaConfig {
    pub window: usize,        // SWA window (e.g. 128)
    pub max_backtrack: usize, // MTP backtracking headroom
    pub keep_tokens: usize,   // retained suffix (V4.1 default 140)
    pub mode: YocoTruncationMode, // Full | Compact
}
impl SwaConfig {
    pub fn from_env(env: &deepinfer_core::env::Env, window: usize, max_backtrack: usize) -> SwaConfig;
    /// keep_tokens must cover window + max_backtrack (Err otherwise when mode == Compact).
    pub fn validate(&self) -> Result<()>;
}

/// Prefill replay plan: which suffix of the prompt must be processed by the decoder half.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayPlan { pub full_len: usize, pub replay_start: usize }
impl ReplayPlan { pub fn replay_len(&self) -> usize; }

pub struct BoundedReplayPlanner { config: SwaConfig }
impl BoundedReplayPlanner {
    pub fn new(config: SwaConfig) -> Result<Self>;
    /// Full mode: replay everything. Compact mode: replay [len - keep, len).
    pub fn plan_prefill(&self, prompt_len: usize) -> ReplayPlan;
}

/// Ring buffer of recent decoder tokens (per layer/per sequence), supporting MTP rollback.
pub struct SwaRingCache { /* window-sized ring over [dim] rows */ }
impl SwaRingCache {
    pub fn new(window: usize, dim: usize) -> Self;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn push(&mut self, row: &[f32]) -> Result<()>;      // evicts oldest beyond window
    pub fn rollback(&mut self, n: usize) -> Result<()>;      // drops last n rows (Err if n > len)
    /// Rows in logical order (oldest..newest), up to `window`.
    pub fn window_rows(&self) -> Vec<Vec<f32>>;
    /// Position of the oldest retained row given the absolute position of the newest.
    pub fn start_position(&self, newest_pos: usize) -> usize;
}

/// Sliding-window causal attention over a q block against the ring contents.
/// q: [s_q, h, d]; kv rows: window rows with their ABSOLUTE positions (pos <= q_pos + q_offset
/// and q_pos - pos < window). Returns out [s_q, h, d].
pub fn swa_attention(q: &[f32], kv: &[f32], kv_positions: &[usize], q_offset: usize,
        s_q: usize, h: usize, d: usize, window: usize, sm_scale: f32) -> Result<Vec<f32>>;
```

Tests: planner modes; validate() rejects keep < window + backtrack; ring push/evict/rollback
round-trips; swa_attention equals dense attention with window mask (property test);
determinism.

## deepinfer-moe  (deps: core, gemm)

DeepSeekMoE router (sigmoid + noaux-tc bias, optional grouped experts) + expert GLU compute.

```rust
pub struct MoeConfig {
    pub hidden: usize,
    pub intermediate: usize,       // per-expert FFN half-size (gate/up each)
    pub num_routed: usize,         // 256
    pub num_activated: usize,      // 8
    pub num_shared: usize,         // 1
    pub group_size: usize,         // 1 = off; else experts grouped (route to top-k/g groups)
    pub use_noaux_tc_bias: bool,   // true: bias affects SELECTION only, not weights
}
impl MoeConfig { pub fn v41_default(hidden: usize) -> Self; }

pub struct ExpertWeights { pub w_gate_up: Vec<f32> /* [2I, hidden] */, pub w_down: Vec<f32> /* [hidden, I] */ }

pub struct MoeWeights {
    pub gate: Vec<f32>,        // [num_routed, hidden]
    pub gate_bias: Vec<f32>,   // [num_routed] e_score_correction_bias
    pub shared: Vec<ExpertWeights>,
    pub routed: Vec<ExpertWeights>,
}
impl MoeWeights { pub fn from_config(cfg: &MoeConfig, rng: &mut Philox) -> MoeWeights; }

/// Routing result for one token.
#[derive(Debug, Clone, PartialEq)]
pub struct Routing { pub topk_idx: Vec<u32>, pub topk_weights: Vec<f32> }

impl MoeWeights {
    /// Sigmoid scores (+ bias for selection when noaux_tc); top-k; group mode: select top-k/g
    /// GROUPS then all experts in each group; renormalize weights to sum 1 over selected.
    pub fn route(&self, cfg: &MoeConfig, x: &[f32]) -> Result<Routing>;
    /// Batch route. Deterministic; ties by expert id ascending.
    pub fn route_batch(&self, cfg: &MoeConfig, xs: &[f32], tokens: usize) -> Result<Vec<Routing>>;
    /// One expert forward: silu(x@w_gate^T) * (x@w_up^T) @ w_down^T (SwiGLU; use f32_gemm_nt).
    pub fn expert_forward(&self, expert: usize, x: &[f32], tokens: usize) -> Result<Vec<f32>>;
    /// Shared expert output (sum over shared experts), added before combine weighting.
    pub fn shared_forward(&self, cfg: &MoeConfig, x: &[f32], tokens: usize) -> Result<Vec<f32>>;
}

/// Full local MoE forward (no EP): y = shared(x) + sum_k w_k * routed_k(x).
pub fn moe_forward(w: &MoeWeights, cfg: &MoeConfig, x: &[f32], tokens: usize) -> Result<Vec<f32>>;
```

Tests: bias affects selection not weights (noaux-tc semantics); renormalization sums to 1;
group mode selects whole groups; moe_forward == manual loop over routing; determinism; property
tests (weights positive, k distinct experts, etc).

## deepinfer-eplb  (deps: core)

EPLB: redundant experts + balanced placement (EPLB repo semantics: phy2log/log2phy/logcnt).

```rust
pub struct EplbConfig {
    pub num_logical: usize,
    pub num_replicas: usize, // extra physical copies
    pub num_groups: usize,
    pub num_nodes: usize,
    pub num_gpus: usize,
}
impl EplbConfig {
    pub fn num_physical(&self) -> usize; // logical + replicas
    pub fn validate(&self) -> Result<()>; // divisibility & positivity checks
}

#[derive(Debug, Clone, PartialEq)]
pub struct EplbMaps {
    pub phy2log: Vec<u32>,          // [num_physical]
    pub log2phy: Vec<Vec<u32>>,     // [num_logical] ascending
    pub logcnt: Vec<u32>,           // [num_logical] copies
}
impl EplbMaps {
    pub fn is_consistent(&self, cfg: &EplbConfig) -> bool; // maps agree both ways, counts match
}

/// Greedy redundant-expert selection: repeatedly clone the expert with the highest
/// load-per-current-copy (weight / copies), until num_replicas added. Returns copy counts.
pub fn get_redundant_experts(weight: &[f64], num_replicas: usize) -> Vec<u32>;

/// Full rebalance. Hierarchical policy when num_groups divides evenly across nodes
/// (pack groups -> nodes, replicate within node, pack to GPUs); otherwise global
/// LPT-style (longest processing time) assignment of physical experts to GPUs.
/// Deterministic: ties broken by lower expert id / lower gpu id.
pub fn rebalance_experts(weight: &[f64], cfg: &EplbConfig) -> Result<EplbMaps>;

/// Sliding-window expert statistics (moving average of historical loads).
pub struct ExpertStats { window: usize, history: Vec<Vec<f64>>, counts: Vec<f64> }
impl ExpertStats {
    pub fn new(num_experts: usize, window: usize) -> Self;
    pub fn record(&mut self, batch_counts: &[u64]);
    pub fn average(&self) -> Vec<f64>;
}
```

Tests: maps consistency; all logical experts placed; per-GPU load balance (max load <= mean +
max single physical load); determinism; stats moving average; property tests with random loads.

## deepinfer-ep  (deps: core, fp8, moe)

DeepEP-equivalent dispatch/combine over in-process channels (normal + low-latency modes, FP8
payloads, hooks). API shape mirrors the DeepEP python API.

```rust
pub struct EpConfig {
    pub world_size: usize,
    pub num_experts: usize,        // total (logical) experts across ranks
    pub num_max_tokens_per_rank: usize,
    pub hidden: usize,
    pub expert_alignment: usize,   // 128 (DeepGEMM contiguous alignment)
}
impl EpConfig {
    pub fn num_local_experts(&self) -> usize; // num_experts / world_size (must divide evenly)
    pub fn owner_rank(&self, expert: u32) -> usize; // expert * world_size / num_experts
    pub fn validate(&self) -> Result<()>;
}

/// Deferred-completion handle (DeepEP EventOverlap / return_recv_hook analogue).
pub struct EpHook { /* fulfilled flag + waker channel */ }
impl EpHook { pub fn wait(&self); pub fn is_fulfilled(&self) -> bool; }

pub struct DispatchHandle {
    pub src_rank: usize,
    pub token_slots: Vec<u32>,     // (token idx in sender batch, k) per received row
    pub topk_idx: Vec<u32>,        // expert per received row
    pub topk_weights: Vec<f32>,
    pub recv_counts_per_expert: Vec<usize>, // normal mode: [num_experts]; LL: per local expert
    pub hook: EpHook,
}

pub enum DispatchPayload { F32(Vec<f32>), Fp8 { data: Vec<u8>, sf: Vec<u8>, group: usize } }

/// Normal-mode dispatch: route each (token, k) to the owner rank; receivers get tokens
/// grouped by expert in arrival order. Deterministic ordering: sort by (expert, sender rank,
/// token index, k). FP8 payload: quantize per-token-group(128) e4m3+e8m0 before "send".
pub fn dispatch(cfg: &EpConfig, rank: usize, x: &[f32], topk_idx: &[u32],
        topk_weights: &[f32], tokens: usize, k: usize, fp8: bool) -> Result<(DispatchPayload, DispatchHandle)>;

/// Deliver dispatched payloads for `rank` (in-process mailbox drain; all ranks must have
/// called dispatch — single-process simulation of the collective).
pub fn recv(cfg: &EpConfig, rank: usize, pending: &[(DispatchPayload, DispatchHandle)])
    -> Result<DispatchResult>;

pub struct DispatchResult {
    pub recv_x: Vec<f32>,          // [tokens_recv, hidden] dequantized
    pub recv_src_idx: Vec<u32>,    // original token index (in sender batch)
    pub recv_topk_idx: Vec<u32>,
    pub recv_topk_weights: Vec<f32>,
    pub handle: DispatchHandle,
}

/// Low-latency dispatch: fixed capacity per (local expert), masked layout
/// [num_local_experts, capacity, hidden]; rows beyond per-expert count are zero-padded.
/// Returns recv_x (padded), plus per-expert counts. Tokens dropped when over capacity
/// must be reported (Err or dropped list — document choice).
pub fn low_latency_dispatch(cfg: &EpConfig, rank: usize, x: &[f32], topk_idx: &[u32],
        topk_weights: &[f32], tokens: usize, k: usize, capacity: usize, fp8: bool)
    -> Result<LowLatencyDispatchResult>;
pub struct LowLatencyDispatchResult {
    pub recv_x: Vec<f32>,          // [num_local_experts * capacity, hidden]
    pub masked_m: Vec<usize>,      // [num_local_experts]
    pub handle: DispatchHandle,
}

/// Combine: weighted sum of expert outputs back to senders, unpermuted to original token
/// order (y[token] = sum_k w_k * expert_out[token,k]). Deterministic summation order:
/// ascending (sender rank, k).
pub fn combine(cfg: &EpConfig, handle: &DispatchHandle, expert_out: &[f32]) -> Result<Vec<f32>>;
```

Tests: dispatch+combine round-trip equals local computation with the same routing; capacity
overflow behavior; expert_alignment padding respected (recv_x row counts per expert multiple
of alignment in normal mode — pad with zero rows and report counts); FP8 error bounds;
determinism; property tests.

## deepinfer-mtp  (deps: core, sampler)

DSpark: MTP draft + confidence-scheduled verification + backtracking (DeepInfer env:
DEEPINFER_ENABLE_MTP_VARLEN, DEEPINFER_USE_NEXTN_DECODE default 5).

```rust
pub struct MtpConfig { pub hidden: usize, pub vocab: usize, pub draft_len: usize /* 5 */ }
impl MtpConfig { pub fn from_env(env: &deepinfer_core::env::Env, hidden: usize, vocab: usize) -> Self; }

/// V3-paper MTP module weights (depth 1 = V4.1 NEXTN):
/// h' = rmsnorm(W_h @ [h ; emb(t_next)]); block(h'); logits = head(rmsnorm_depth(h'')).
pub struct MtpWeights {
    pub eh_proj: Vec<f32>,     // [hidden, 2*hidden]
    pub shared_norm: Vec<f32>, // [hidden]
    pub depth_norm: Vec<f32>,  // [hidden]
    // transformer block weights (minimal: qkv-less single attention-free MLP block is NOT
    // faithful — implement a small full block: attn + mlp weights) OR accept a closure
    // `block` provided by the runtime; pick ONE and document.
    pub block: MtpBlockWeights,
    pub head: Vec<f32>,        // [vocab, hidden]
}
pub struct MtpBlockWeights { /* attn + mlp weight matrices, your layout */ }
impl MtpWeights { pub fn from_config(cfg: &MtpConfig, rng: &mut Philox) -> MtpWeights; }

#[derive(Debug, Clone)]
pub struct DraftToken { pub token: u32, pub confidence: f32, pub logits: Vec<f32> }

pub trait DraftModel {
    /// One draft step from (last accepted token, its hidden state).
    fn draft_step(&mut self, last_token: u32, hidden: &[f32]) -> Result<DraftToken>;
    fn config(&self) -> &MtpConfig;
}

/// Confidence-scheduling decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Decision { ContinueDraft, VerifyNow }

pub struct DsparkScheduler {
    pub max_draft: usize,       // from env DEEPINFER_USE_NEXTN_DECODE
    pub conf_hi: f32,           // e.g. 0.9
    pub conf_lo: f32,           // e.g. 0.5
}
impl DsparkScheduler {
    /// Verify when: draft full, OR running-min confidence < conf_lo (early stop),
    /// OR mean confidence >= conf_hi with draft >= 2 (confident extension not needed).
    pub fn should_verify(&self, drafts: &[DraftToken]) -> Decision;
    /// Number of extra positions the target must evaluate (draft + 1 for the next token).
    pub fn verify_positions(&self, drafts: &[DraftToken]) -> usize;
}

pub struct VerificationOutcome {
    pub accepted: usize,
    pub next_token: u32,          // token after the accepted prefix (from target logits)
    pub target_logits: Vec<Vec<f32>>, // per verified position (for runtime reuse)
}

/// Greedy verification: accept draft[i] while it equals argmax(target_logits[i]).
pub fn verify_greedy(drafts: &[DraftToken], target_logits: &[Vec<f32>]) -> Result<VerificationOutcome>;

/// Stochastic (resample) verification with a SHARED per-sequence RNG stream:
/// sample t_i ~ target_logits[i] (via sampler); accept while t_i == draft[i].token;
/// on first mismatch the sampled t_i becomes the corrected next token.
pub fn verify_stochastic(drafts: &[DraftToken], target_logits: &[Vec<f32>],
        rng: &mut Philox, temperature: f32) -> Result<VerificationOutcome>;

/// Backtracking requirement: positions to roll back = drafts.len() - accepted.
pub fn backtrack_steps(drafts: &[DraftToken], outcome: &VerificationOutcome) -> usize;
```

Tests: greedy accept prefix correctness; scheduler policy table; backtrack math; stochastic
verify uses the RNG deterministically (same seed → same acceptance); property tests.

## deepinfer-sampler  (deps: core)

Deterministic, batch-invariant sampling (DeepInfer SamplingParams semantics with seq_seed).

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemaType { Regex, Json, Ebnf }

#[derive(Debug, Clone, PartialEq)]
pub struct StructuredOutputArgs { pub schema_type: SchemaType, pub schema: String }

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,     // 0 = greedy
    pub top_p: f32,           // 1 = off
    pub top_k: usize,         // 0 = off
    pub min_p: f32,           // 0 = off
    pub seq_seed: Option<u64>,
    pub so_args: Option<StructuredOutputArgs>,
    pub max_tokens: usize,
}
impl Default for SamplingParams {
    fn default() -> Self { temperature 1.0, top_p 1.0, top_k 0, min_p 0.0, seq_seed None, so_args None, max_tokens 64 }
}
impl SamplingParams { pub fn validate(&self, vocab: usize) -> Result<()>; }

pub struct Sampler { params: SamplingParams }
impl Sampler {
    pub fn new(params: SamplingParams) -> Sampler;
    pub fn params(&self) -> &SamplingParams;
    /// Deterministic sample from logits (len == vocab). mask: optional allowed-token bitset
    /// (structured outputs). Pipeline: mask -> temperature -> top-k -> top-p -> min-p ->
    /// categorical(rng). Greedy when temperature <= 0 (argmax, ties -> lowest id).
    /// Masked-to-zero tokens never sampled; empty mask -> Err.
    pub fn sample(&self, logits: &[f32], mask: Option<&TokenMask>, rng: &mut Philox) -> Result<u32>;
    /// Per-sequence RNG derivation: Philox::new(seq_seed or derived, seq_id) — batch invariant.
    pub fn seq_rng(seed: Option<u64>, seq_id: u64) -> Philox;
}
```

Tests: determinism (same logits+seed → same token); batch invariance (RNG independent of
other sequences); top-p keeps minimal prefix >= p; top-k/min-p filtering; mask enforcement;
greedy ties; temperature 0; property tests (sampled id always in mask, distribution sanity).

## deepinfer-grammar  (deps: core)

XGrammar-equivalent structured outputs: EBNF / JSON Schema / Regex -> byte-level DFA ->
adaptive token masks. (so_args = {"schema_type": "regex"|"json"|"ebnf", "schema": "..."}).

```rust
pub use deepinfer_sampler_types /* NO — avoid cycles: define SchemaType/StructuredOutputArgs
   LOCALLY as well (sampler has its own copy; runtime converts). */;

/// Compiled byte-level DFA with a hard state cap (DoS guard, default 1 << 16 states).
pub struct Grammar { /* ... */ }
impl Grammar {
    pub fn compile_ebnf(text: &str) -> Result<Grammar>;
    pub fn compile_json_schema(schema: &deepinfer_core::json::Json) -> Result<Grammar>;
    pub fn compile_regex(pattern: &str) -> Result<Grammar>; // full-match semantics
    pub fn from_so_args(schema_type: &str, schema: &str) -> Result<Grammar>; // "regex"/"json"/"ebnf"
    pub fn initial_state(&self) -> u32;
    pub fn is_accepting(&self, state: u32) -> bool;
    /// None = dead state (no accepting continuation).
    pub fn advance(&self, state: u32, byte: u8) -> Option<u32>;
    pub fn num_states(&self) -> usize;
}

/// Adaptive token-mask computation over a vocabulary (token -> bytes).
pub struct TokenMaskComputer { vocab: Vec<Vec<u8>>, cache: std::collections::HashMap<u32, TokenMask> }
impl TokenMaskComputer {
    pub fn new(vocab_bytes: Vec<Vec<u8>>) -> Self;
    pub fn vocab_size(&self) -> usize;
    /// A token is allowed from `state` iff consuming all its bytes stays on live states AND
    /// either (a) the final state is accepting (token can end the generation), or
    /// (b) the final state has a continuation (more input possible). EOS handling is the
    /// runtime's job: empty-byte tokens are allowed iff is_accepting(state).
    pub fn compute_mask(&mut self, grammar: &Grammar, state: u32) -> deepinfer_core::mask::TokenMask;
    /// Advance state over a token's bytes (None = dead).
    pub fn advance_token(&self, grammar: &Grammar, state: u32, token: u32) -> Option<u32>;
    pub fn cache_size(&self) -> usize;
}
```

EBNF subset (xgrammar-style): rules `name ::= alt;`, alternatives `|`, sequences, optional
`?`, star `*`, plus `+`, grouping `()`, ranges `{m, n}` optional, literals `"abc"` and `'abc'`
with escapes `\n \t \r \\ \" \' \u{XXXX}`, character classes `[a-z0-9_]` with negation `^`,
dot `.` (any byte except `\n`), comments `(* ... *)`. Reject unsupported constructs with Err.

JSON Schema subset: type (string/number/integer/boolean/null/object/array or arrays of those),
properties/required/additionalProperties (bool or schema), items, prefixItems, enum, const,
anyOf/oneOf, allOf (merge), $ref to #/$defs (local), minItems/maxItems, minLength/maxLength,
minimum/maximum (numeric bounds -> digit-count grammar), patternProperties unsupported -> Err,
format ignored, exclusiveMinimum/Maximum supported. Numbers: JSON number grammar (optional -,
integer digits without leading zeros, optional fraction, optional exponent).

Regex subset: literals, `.` (any byte but `\n`), classes `[...]` with ranges/negation and
escapes, escapes `\d \D \w \W \s \S \n \t \r \\ \. \* ...`, alternation, groups (non-capturing
ok, capture treated as plain), quantifiers `* + ? {m} {m,} {m,n}` (greedy == lazy for
full-match DFA), anchors `^ $` (treated as no-ops under full-match semantics; document),
unicode escapes `\uXXXX` in literals. REJECT with clear Err: backreferences, lookaround,
backreferences `\1`, named groups are OK to treat as plain groups. Word boundaries `\b`: Err.

Tests: EBNF parse/compile; JSON schema -> generate strings by walking the DFA (all sampled
strings validate against a hand validator); regex full-match vs a naive backtracking matcher
(property test on the supported subset); mask correctness vs brute-force per-token check;
state-cap DoS guard (e.g. `(a|b){200}` nesting or `a{1000000}` → Err, not OOM); EOS rule;
caching; determinism.

## deepinfer-scheduler  (deps: core; NO dep on runtime)

Continuous batching + chunked prefill + preemption + zero-overlap async engine loop.

```rust
pub struct SchedulerConfig {
    pub max_running_requests: usize,  // e.g. 64
    pub max_batch_tokens: usize,      // token budget per step (prefill+decode), e.g. 8192
    pub chunked_prefill_size: usize,  // max prompt tokens per step per request, e.g. 2048
    pub batch_invariant: bool,        // DEEPINFER_BATCH_INVARIANT
}
impl SchedulerConfig { pub fn from_env(env: &Env) -> SchedulerConfig; }

#[derive(Debug, Clone)]
pub struct Request {
    pub id: u64,
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
    pub arrival_order: u64,
}

/// Per-request execution state owned by the scheduler.
#[derive(Debug, Clone, PartialEq)]
pub enum Phase {
    Waiting,
    Prefilling { chunk: usize },   // prompt tokens already prefilled
    Decoding { generated: usize },
    Finished { StopReason },
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason { Length, Eos, Aborted }

/// One execution step's plan handed to the engine.
pub enum BatchPlan {
    Idle,
    Step { decode_reqs: Vec<u64>, prefill_reqs: Vec<(u64, usize, usize)> /* id, chunk_start, chunk_len */ },
}

pub struct Scheduler { /* ... */ }
impl Scheduler {
    pub fn new(cfg: SchedulerConfig) -> Scheduler;
    pub fn add_request(&mut self, req: Request) -> u64; // assigns id if 0
    pub fn num_active(&self) -> usize;
    /// Compute the next batch from current states. Admission: FCFS by (arrival_order, id);
    /// decode-first token budget; chunk prefill when prompt > chunked_prefill_size;
    /// preemption: evict the NEWEST prefilling requests (recompute on re-admission) when
    /// decode budget is squeezed. Batch-invariant mode: admission decisions depend only on
    /// request set + step counter, never on wall-clock.
    pub fn schedule(&mut self) -> BatchPlan;
    /// Report step results: (req_id, generated_token). Advances phases; EOS/length finish.
    pub fn report_tokens(&mut self, tokens: &[(u64, u32)], eos_id: u32);
    pub fn finished(&self) -> Vec<(u64, StopReason)>;
    pub fn is_done(&self) -> bool;
}

/// Engine interface implemented by the runtime (keeps this crate decoupled).
pub trait EngineOps: Send {
    /// Execute one step plan; returns (req_id, token) per decode position, in plan order.
    fn execute_step(&mut self, plan: &BatchPlan, eos_id: u32) -> deepinfer_core::Result<Vec<(u64, u32)>>;
}

/// Two-thread overlapping engine: the scheduler builds step N+1 while the engine executes
/// step N (DeepInfer "asynchronous scheduling (zero-overhead batch scheduler)").
pub struct AsyncEngine<E: EngineOps> { /* ... */ }
impl<E: EngineOps + 'static> AsyncEngine<E> {
    pub fn new(cfg: SchedulerConfig, engine: E) -> AsyncEngine<E>;
    pub fn submit(&mut self, req: Request);
    /// Runs until all requests finish (or `max_steps`); returns finish reasons.
    pub fn run(&mut self, eos_id: u32, max_steps: usize) -> deepinfer_core::Result<Vec<(u64, StopReason)>>;
    pub fn steps_executed(&self) -> u64;
}
```

Tests: all requests complete in mock engine; chunked prefill splits long prompts; preemption
recovery (mock engine alternates budgets); determinism of schedule() sequence (same inputs →
same plans); overlap correctness (AsyncEngine with a counting mock == sequential execution);
property tests.

## deepinfer-runtime  (deps: ALL above; wave 2)

Model config (HF-style JSON), checkpoint loading, the YOCO-layered DeepSeek-V4.1-Flash model,
generate() API, offline_inference example. Exact API to be defined by the wave-2 agent in its
own doc; REQUIRED surface:

```rust
pub struct ModelConfig { /* from JSON: hidden_size, n_layers, yoco split, heads, mla fields,
   dsa fields, moe fields, mtp fields, vocab_size, rms_eps, rope, max_position_embeddings */ }
impl ModelConfig { pub fn from_json(&Json) -> Result<Self>; pub fn v41_tiny() -> Self; }

pub struct DeepSeekModel { /* weights + layer stack + caches */ }
impl DeepSeekModel {
    pub fn load_checkpoint(dir: &Path, cfg: &ModelConfig) -> Result<Self>; // safetensors
    pub fn random_init(cfg: &ModelConfig, seed: u64) -> Result<Self>;      // --tiny self-test
    /// One decode/prefill step producing next-token logits per sequence.
    pub fn forward(&mut self, ...) -> ...;
}

pub struct GenerateRequest { pub prompt: Vec<u32>, pub params: SamplingParams }
pub fn generate(model, tokenizer, requests, env) -> Result<Vec<GenerateOutput>>;
```

Plus `examples/offline_inference.rs` mirroring upstream `examples/offline_inference.py`
(prompts, DEEPINFER_HF_MODEL_DIR support, prints completions; `--tiny` runs a random-init
self-test without a checkpoint).

## Milestone & reporting

Each wave-1 agent: implement crate(s) + tests until `cargo test -p <crate>` and
`cargo build --workspace` are green, then append to `/home/z/my-project/worklog.md`:

```
---
Task ID: <your id>
Agent: <subagent type>
Task: <crate names>

Work Log:
- <steps>

Stage Summary:
- <public API implemented, tests passing count, deviations from spec (if any)>
```

## Appendix: GPU crates (wave 3)

### deepinfer-gpu (deps: core)

Runtime-loaded device layer. dlopen + typed FFI (CUDA driver, NVRTC, HIP,
hipRTC), unified `Gpu/Device/Stream/Buffer/Module/Function/KernelArgs`,
arch detection (NV cc -> `sm_XXX`; AMD gcnArchName buffer-scan ->
`gfxXXXX`, `DEEPINFER_HIP_ARCH` override), PTX fallback for old-NVRTC/new-
GPU, SHA-256 content-addressed JIT disk cache (`DEEPINFER_JIT_CACHE`,
`DEEPINFER_JIT_DEBUG`, `DEEPINFER_CACHE_DIR`), peer access + async peer
copy. The ONLY crate with `unsafe` (audited modules: dl, ffi/*, device,
jit — `deny` + `#[allow(unsafe_code)]` with SAFETY proofs, since workspace
`forbid` cannot be overridden).

```rust
pub fn probe_backends() -> Vec<Backend>;
impl Gpu { pub fn new(Backend) -> Result<Gpu>; pub fn device_count(); pub fn device(usize) -> Result<Device>; }
impl Device { props(); stream(); alloc(); alloc_from(); write(); read(); load_module(); enable_peer_access(); memcpy_peer_async(); }
pub fn compile(device, source, name, arch, opts) -> Result<KernelImage>;
```

### deepinfer-kernels (deps: core, gpu; dev: fp8, gemm, mla, moe, dsa)

Operator library: embedded CUDA/HIP C++ (`kernels/*.cu`, one source for
both compilers) + typed launchers `Kernels::{rmsnorm, rope, silu_mul, add,
quant_per_token_group, kv_encode_fp8/fp4, gemm_bf16_nt, gemm_fp8_nt,
gemm_fp8_grouped_nt, mla_decode_partial/combine, mla_prefill,
indexer_rmsnorm_per_head, lightning_quant, thunder_logits, conv_feat,
conv_gated_norm, dot_scores, qmean, gate_logits, gate_blend, causal_clip,
select_top_k, moe_route, moe_dispatch, moe_combine, fill_zero}`.
Golden integration tests vs the CPU crates (auto-skip without a device).
See docs/GPU.md for the full contract, precision policy and upgrade paths.
