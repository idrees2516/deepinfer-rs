//! The DeepSeek-V4.1-Flash runtime model: the YOCO-layered decoder stack,
//! paged KV caches, prefill/decode entry points and MTP rollback.
//!
//! Architecture (DeepInfer upstream README + FlashMLA README + YOCO paper,
//! Sun et al., arXiv:2405.05254):
//!
//! * The first [`ModelConfig::n_self_layers`] layers form the
//!   **self-decoder**: MLA attention (absorbed 512-wide latent+rope rows,
//!   DSA sparse top-k selection with the V3.2 mscale folded into the
//!   softmax scale) over a per-layer paged KV cache, then a DeepSeekMoE
//!   FFN. The DSA indexer consumes the layer's residual-stream input and
//!   scores it against a per-layer cache of the same rows, so selection
//!   stays causal and deterministic.
//! * The remaining layers form the **cross-decoder**: each cross layer
//!   has its own `q_proj`/`o_proj` and attends (with RoPE + the O-RoPE
//!   conjugate) into the **last self-decoder layer's KV cache** — the
//!   YOCO "you only cache once" trick — plus a sliding-window local
//!   self-attention over a ring of recent normed hidden rows
//!   (`deepinfer_swa::swa_attention`); the two attention outputs are
//!   summed before the MoE FFN.
//! * The final RMSNorm + LM head produce next-token logits.
//!
//! Documented CPU-reference simplifications:
//! * Cross-attention is **dense over the most recent
//!   [`ModelConfig::cross_attn_window`] (512) self-decoder KV rows**
//!   instead of the full encoder cache — a tractability cap.
//! * MoE runs **locally** (`deepinfer_moe::moe_forward`, no expert
//!   parallelism) — the single-process CPU path; the `deepinfer-ep`
//!   crate provides dispatch/combine for distributed deployments.
//! * MTP rollback rebuilds the per-layer paged caches from the exact f32
//!   shadow rows (the public `MlaPagedCache` has no removal API), so a
//!   rejected draft leaves the model bit-identical to the state the
//!   plain autoregressive path would have reached.

use std::collections::BTreeMap;
use std::path::Path;

use deepinfer_core::norm::rmsnorm;
use deepinfer_core::tokenizer::Tokenizer;
use deepinfer_core::{ensure, Error, Result};
use deepinfer_dsa::mscale;
use deepinfer_mla::{flash_mla_decode, flash_mla_sparse_prefill, o_rope_conjugate, MlaPagedCache};
use deepinfer_core::math::dot;
use deepinfer_core::rope::Rope;
use deepinfer_swa::{swa_attention, SwaRingCache};

use crate::config::{ModelConfig, PAGE_SIZE};
use crate::loader;
use crate::weights::{
    matvec, matvec_into, zeros, CrossLayerWeights, ModelWeights, SelfLayerWeights,
};

/// Maximum number of tokens one `forward` call may process (CPU reference
/// chunk bound; longer prompts must be chunked by the caller — the
/// scheduler crate implements chunked prefill for service deployments).
pub const MAX_PREFILL_TOKENS: usize = 8192;

/// Per-sequence runtime state: the logical length, per-self-layer caches
/// (exact f32 KV shadow rows for rollback + residual rows feeding the DSA
/// indexer) and per-cross-layer SWA rings.
struct SeqState {
    len: usize,
    /// `[n_self][len * head_dim]` — exact f32 KV rows as appended (the
    /// pre-quantization source of truth for cache rebuilds).
    kv_shadow: Vec<Vec<f32>>,
    /// `[n_self][len * hidden]` — residual-stream rows for the DSA
    /// indexer's key side.
    indexer_k: Vec<Vec<f32>>,
    /// `[n_cross]` — SWA rings over the normed hidden rows.
    rings: Vec<SwaRingCache>,
}

impl SeqState {
    fn new(cfg: &ModelConfig) -> SeqState {
        SeqState {
            len: 0,
            kv_shadow: vec![Vec::new(); cfg.n_self_layers],
            indexer_k: vec![Vec::new(); cfg.n_self_layers],
            rings: (0..cfg.n_cross_layers)
                .map(|_| SwaRingCache::new(cfg.swa_window, cfg.hidden_size))
                .collect(),
        }
    }
}

/// The DeepSeek-V4.1-Flash model: weights, per-layer caches and
/// per-sequence state. Tokenizer-independent — structured-output vocab
/// bytes can be attached separately via [`DeepSeekModel::set_vocab_bytes`].
pub struct DeepSeekModel {
    /// Model configuration.
    config: ModelConfig,
    /// Full weight set.
    pub(crate) weights: ModelWeights,
    /// One paged KV cache per self-decoder layer (each layer owns its own
    /// page pool; sequences allocate pages from their layer's pool).
    self_caches: Vec<MlaPagedCache>,
    /// Per-sequence state, keyed by sequence id (deterministic order).
    seqs: BTreeMap<u32, SeqState>,
    /// Optional vocabulary byte table for structured-output masking.
    vocab_bytes: Option<Vec<Vec<u8>>>,
}

impl DeepSeekModel {
    /// Loads a model from a checkpoint directory (safetensors shards +
    /// optional `config.json` cross-check). See [`loader`] for the
    /// accepted tensor-name mapping.
    ///
    /// # Errors
    /// [`Error::Checkpoint`] when the directory or tensors are missing or
    /// shaped inconsistently with `cfg`.
    pub fn load_checkpoint(dir: &Path, cfg: &ModelConfig) -> Result<Self> {
        cfg.validate()?;
        let config_path = dir.join("config.json");
        if config_path.is_file() {
            let text = std::fs::read_to_string(&config_path).map_err(|e| {
                Error::Checkpoint(format!("reading {}: {e}", config_path.display()))
            })?;
            let json = deepinfer_core::json::Json::parse(&text)?;
            let disk = ModelConfig::from_json(&json)?;
            ensure!(
                disk.hidden_size == cfg.hidden_size,
                Checkpoint,
                "config.json hidden_size {} disagrees with the provided config ({})",
                disk.hidden_size,
                cfg.hidden_size
            );
            ensure!(
                disk.vocab_size == cfg.vocab_size,
                Checkpoint,
                "config.json vocab_size {} disagrees with the provided config ({})",
                disk.vocab_size,
                cfg.vocab_size
            );
            ensure!(
                disk.n_layers == cfg.n_layers,
                Checkpoint,
                "config.json n_layers {} disagrees with the provided config ({})",
                disk.n_layers,
                cfg.n_layers
            );
        }
        let cp = deepinfer_core::safetensors::CheckpointDir::open(dir)?;
        let weights = loader::load_weights(&cp, cfg)?;
        Self::from_parts(cfg.clone(), weights)
    }

    /// Deterministic random initialization (self-test path; see
    /// [`crate::weights`]). `cfg` is validated and the total parameter
    /// count is bounded before any allocation.
    ///
    /// # Errors
    /// [`Error::CapacityExceeded`] when the configuration is too large
    /// for a random init; validation errors from
    /// [`ModelConfig::validate`].
    pub fn random_init(cfg: &ModelConfig, seed: u64) -> Result<Self> {
        cfg.validate()?;
        let weights = crate::weights::random_init(cfg, seed)?;
        Self::from_parts(cfg.clone(), weights)
    }

    fn from_parts(config: ModelConfig, weights: ModelWeights) -> Result<Self> {
        let self_caches = (0..config.n_self_layers)
            .map(|_| MlaPagedCache::new(config.cache, PAGE_SIZE, config.max_pages()))
            .collect();
        Ok(DeepSeekModel {
            config,
            weights,
            self_caches,
            seqs: BTreeMap::new(),
            vocab_bytes: None,
        })
    }

    /// The model configuration.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Number of tokens cached for `seq` (0 for unknown sequences).
    pub fn seq_len(&self, seq: u32) -> usize {
        self.seqs.get(&seq).map(|s| s.len).unwrap_or(0)
    }

    /// Total tokens in the first self-decoder layer's paged cache (all
    /// layers stay in lockstep; 0 when there is no self layer).
    pub fn total_cached_tokens(&self) -> usize {
        self.self_caches
            .first()
            .map(|c| c.total_tokens())
            .unwrap_or(0)
    }

    /// Whether the model carries MTP (NEXTN) draft weights.
    pub fn has_mtp(&self) -> bool {
        self.weights.mtp.is_some()
    }

    /// Attaches a vocabulary byte table (`token id -> bytes`) used for
    /// structured-output masking. The EOS token should have empty bytes
    /// (see [`DeepSeekModel::vocab_bytes_from_tokenizer`], which arranges
    /// this).
    pub fn set_vocab_bytes(&mut self, vocab_bytes: Vec<Vec<u8>>) {
        self.vocab_bytes = Some(vocab_bytes);
    }

    /// The attached vocabulary byte table, if any.
    pub fn vocab_bytes(&self) -> Option<&[Vec<u8>]> {
        self.vocab_bytes.as_deref()
    }

    /// Derives a vocabulary byte table from a tokenizer:
    /// * `vocab == 256` — byte-level table (`id t` ↔ byte `t`), matching
    ///   [`deepinfer_core::tokenizer::ByteTokenizer`];
    /// * otherwise — the decoded bytes of each single-token id (matching
    ///   byte-level BPE).
    ///
    /// The EOS token's bytes are emptied so the grammar mask can gate EOS
    /// on accepting states only.
    ///
    /// # Errors
    /// [`Error::OutOfRange`] on a bad `eos_id`; tokenizer decode errors
    /// are wrapped as [`Error::Checkpoint`].
    pub fn vocab_bytes_from_tokenizer(tok: &dyn Tokenizer, eos_id: u32) -> Result<Vec<Vec<u8>>> {
        let vocab = tok.vocab_size();
        ensure!(
            vocab >= 1 && (eos_id as usize) < vocab,
            OutOfRange,
            "eos_id {eos_id} outside vocab [0, {vocab})"
        );
        let mut out = Vec::with_capacity(vocab);
        if vocab == 256 {
            for t in 0..256u32 {
                out.push(vec![t as u8]);
            }
        } else {
            for t in 0..vocab as u32 {
                let s = tok.decode(&[t]).map_err(|e| {
                    Error::Checkpoint(format!("decoding token {t} for vocab bytes: {e}"))
                })?;
                out.push(s.into_bytes());
            }
        }
        out[eos_id as usize] = Vec::new();
        Ok(out)
    }

    /// Clears all sequence state: per-sequence caches, SWA rings and the
    /// paged KV caches. Weights and the vocab byte table are kept.
    pub fn reset(&mut self) {
        self.seqs.clear();
        let (format, pages) = (self.config.cache, self.config.max_pages());
        for cache in &mut self.self_caches {
            *cache = MlaPagedCache::new(format, PAGE_SIZE, pages);
        }
    }

    /// Prefills `tokens` for `seq` (appending to any existing state) and
    /// returns the next-token logits `[vocab]` for the final position.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] on an empty token slice;
    /// [`Error::OutOfRange`] on tokens outside the vocabulary;
    /// [`Error::CapacityExceeded`] when the position or chunk bound would
    /// be exceeded.
    pub fn prefill(&mut self, seq: u32, tokens: &[u32]) -> Result<Vec<f32>> {
        ensure!(
            !tokens.is_empty(),
            InvalidArgument,
            "prefill requires at least one prompt token"
        );
        let (logits, _) = self.forward_tokens(seq, tokens)?;
        Ok(logits)
    }

    /// Decodes one `token` for `seq` (appends to the caches) and returns
    /// the next-token logits `[vocab]`.
    ///
    /// # Errors
    /// [`Error::OutOfRange`] when `token` is outside the vocabulary;
    /// [`Error::CapacityExceeded`] when the position bound would be
    /// exceeded.
    pub fn decode_step(&mut self, seq: u32, token: u32) -> Result<Vec<f32>> {
        let (logits, _) = self.forward_tokens(seq, &[token])?;
        Ok(logits)
    }

    /// Shared forward path: embeds `tokens`, runs the layer stack and the
    /// final norm + LM head. Returns `(next-token logits, final residual
    /// hidden state of the last position)` — the hidden state feeds the
    /// MTP draft module.
    pub(crate) fn forward_tokens(
        &mut self,
        seq: u32,
        tokens: &[u32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let cfg = &self.config;
        let hidden = cfg.hidden_size;
        let s = tokens.len();
        ensure!(
            s >= 1,
            InvalidArgument,
            "forward_tokens requires at least one token"
        );
        ensure!(
            s <= MAX_PREFILL_TOKENS,
            CapacityExceeded,
            "forward chunk of {s} tokens exceeds the {MAX_PREFILL_TOKENS}-token CPU bound; \
             split the prefill"
        );
        for &t in tokens {
            ensure!(
                (t as usize) < cfg.vocab_size,
                OutOfRange,
                "token {t} is outside the vocabulary ({})",
                cfg.vocab_size
            );
        }
        self.seqs.entry(seq).or_insert_with(|| SeqState::new(cfg));
        let pos0 = self
            .seqs
            .get(&seq)
            .map(|st| st.len)
            .ok_or_else(|| Error::internal("sequence state missing after insertion"))?;
        let end = pos0
            .checked_add(s)
            .ok_or_else(|| Error::CapacityExceeded("position overflow".into()))?;
        ensure!(
            end <= cfg.max_position_embeddings,
            CapacityExceeded,
            "position {end} would exceed max_position_embeddings ({})",
            cfg.max_position_embeddings
        );

        // Embedding lookup.
        let h_len = s
            .checked_mul(hidden)
            .ok_or_else(|| Error::CapacityExceeded("s*hidden overflows".into()))?;
        let mut h = zeros(h_len)?;
        for (i, &t) in tokens.iter().enumerate() {
            let src = t as usize * hidden;
            h[i * hidden..(i + 1) * hidden].copy_from_slice(&self.weights.embed[src..src + hidden]);
        }

        // Self-decoder layers (weights / caches / sequence state are
        // disjoint fields, so the split borrows below are legal).
        {
            let st = self
                .seqs
                .get_mut(&seq)
                .ok_or_else(|| Error::internal("sequence state missing after insertion"))?;
            for l in 0..cfg.n_self_layers {
                let w = &self.weights.self_layers[l];
                let cache = &mut self.self_caches[l];
                let k_rows = &mut st.indexer_k[l];
                let kv_shadow = &mut st.kv_shadow[l];
                h = self_layer_forward(cfg, w, cache, k_rows, kv_shadow, seq, &h, s, pos0)?;
            }
        }
        // Cross-decoder layers (attending to the LAST self layer's cache).
        if cfg.n_cross_layers > 0 {
            let last = cfg.n_self_layers - 1;
            let st = self
                .seqs
                .get_mut(&seq)
                .ok_or_else(|| Error::internal("sequence state missing after insertion"))?;
            for l in 0..cfg.n_cross_layers {
                let w = &self.weights.cross_layers[l];
                let rope = &self.weights.rope;
                let ring = &mut st.rings[l];
                h = cross_layer_forward(
                    cfg,
                    w,
                    rope,
                    &self.self_caches[last],
                    ring,
                    seq,
                    &h,
                    s,
                    pos0,
                )?;
            }
        }

        // Final norm + LM head over the last position.
        let mut normed = h[(s - 1) * hidden..s * hidden].to_vec();
        rmsnorm(&mut normed, &self.weights.final_norm, cfg.rms_eps);
        let logits = matvec(&self.weights.lm_head, &normed, cfg.vocab_size, hidden)?;
        // The residual (pre-norm) hidden state feeds the MTP module.
        let residual = h[(s - 1) * hidden..s * hidden].to_vec();
        self.seqs
            .get_mut(&seq)
            .ok_or_else(|| Error::internal("sequence state missing after forward"))?
            .len += s;
        Ok((logits, residual))
    }

    /// Rolls back the newest `n` tokens of `seq` (MTP rejection): drops
    /// shadow/indexer rows, rolls back the SWA rings and rebuilds the
    /// paged caches from the surviving exact f32 rows so the model is
    /// bit-identical to the state before the rejected tokens.
    pub(crate) fn rollback_seq(&mut self, seq: u32, n: usize) -> Result<()> {
        let cfg = &self.config;
        let hd = cfg.head_dim;
        let hidden = cfg.hidden_size;
        let st = self
            .seqs
            .get_mut(&seq)
            .ok_or_else(|| Error::invalid("rollback on unknown sequence {seq}"))?;
        ensure!(
            n <= st.len,
            InvalidArgument,
            "cannot roll back {n} tokens: sequence {seq} only holds {}",
            st.len
        );
        st.len -= n;
        for l in 0..cfg.n_self_layers {
            st.kv_shadow[l].truncate(st.len * hd);
            st.indexer_k[l].truncate(st.len * hidden);
        }
        for l in 0..cfg.n_cross_layers {
            st.rings[l].rollback(n)?;
        }
        // Rebuild the paged caches from the shadows (same rows in the
        // same order ⇒ identical quantized contents; only the physical
        // page layout may differ, which no kernel observes).
        let mut fresh = (0..cfg.n_self_layers)
            .map(|_| MlaPagedCache::new(cfg.cache, PAGE_SIZE, cfg.max_pages()))
            .collect::<Vec<_>>();
        for (&sid, state) in &self.seqs {
            for l in 0..cfg.n_self_layers {
                for t in 0..state.len {
                    fresh[l].append(sid, &state.kv_shadow[l][t * hd..(t + 1) * hd])?;
                }
            }
        }
        self.self_caches = fresh;
        Ok(())
    }

    /// The MTP weights (speculative path).
    pub(crate) fn mtp(&self) -> Option<&deepinfer_mtp::MtpWeights> {
        self.weights.mtp.as_ref()
    }

    /// Resolves the vocabulary byte table for structured outputs: the
    /// attached table, or the byte-level fallback for 256-entry
    /// vocabularies.
    pub(crate) fn resolve_vocab_bytes(&self, eos_id: u32) -> Result<Vec<Vec<u8>>> {
        if let Some(vb) = &self.vocab_bytes {
            ensure!(
                vb.len() == self.config.vocab_size,
                InvalidArgument,
                "attached vocab byte table has {} entries but the model vocab is {}",
                vb.len(),
                self.config.vocab_size
            );
            return Ok(vb.clone());
        }
        ensure!(
            self.config.vocab_size == 256,
            InvalidArgument,
            "structured outputs need a vocabulary byte table: call \
             DeepSeekModel::set_vocab_bytes (or vocab_bytes_from_tokenizer)"
        );
        let mut vb: Vec<Vec<u8>> = (0..256u32).map(|t| vec![t as u8]).collect();
        vb[eos_id as usize] = Vec::new();
        Ok(vb)
    }
}

/// One self-decoder layer forward (MLA + DSA + MoE) over `s` tokens
/// starting at absolute position `pos0`.
fn self_layer_forward(
    cfg: &ModelConfig,
    w: &SelfLayerWeights,
    cache: &mut MlaPagedCache,
    k_rows: &mut Vec<f32>,
    kv_shadow: &mut Vec<f32>,
    seq: u32,
    h_in: &[f32],
    s: usize,
    pos0: usize,
) -> Result<Vec<f32>> {
    let hidden = cfg.hidden_size;
    let hd = cfg.head_dim;
    let latent = hd - cfg.rope_dim;
    let rope = &w.mla.rope;

    // 1. Pre-attention RMSNorm.
    let mut x = h_in.to_vec();
    for i in 0..s {
        rmsnorm(&mut x[i * hidden..(i + 1) * hidden], &w.attn_norm, cfg.rms_eps);
    }

    // 2. KV projection (latent slice normed inside), rope, cache append.
    let mut kv = zeros(s * hd)?;
    w.mla.project_kv(&x, s, &mut kv)?;
    let kv_positions: Vec<u32> = (pos0..pos0 + s).map(|p| p as u32).collect();
    rope.apply_at(&kv_positions, &mut kv, hd, latent)?;
    for i in 0..s {
        let row = kv[i * hd..(i + 1) * hd].to_vec();
        cache.append(seq, &row)?;
        kv_shadow.extend_from_slice(&row);
        // DSA indexer key side: the raw residual rows.
        k_rows.extend_from_slice(&h_in[i * hidden..(i + 1) * hidden]);
    }

    // 3. Q projection + rope on each head's rope slice.
    let mut q = zeros(s * cfg.n_heads * hd)?;
    w.mla.project_q(&x, s, &mut q)?;
    let q_positions: Vec<u32> = (0..s * cfg.n_heads)
        .map(|r| (pos0 + r / cfg.n_heads) as u32)
        .collect();
    rope.apply_at(&q_positions, &mut q, hd, latent)?;

    // 4. DSA selection over (current residual rows, all cached rows).
    let total = pos0 + s;
    let sel = deepinfer_dsa::indexer_forward(&w.indexer, &cfg.dsa, h_in, k_rows, s, total)?;
    let topk = cfg.dsa.topk_tokens.min(total);
    ensure!(
        sel.len() == s * topk,
        Internal,
        "indexer returned {} indices, expected s*topk = {}",
        sel.len(),
        s * topk
    );

    // 5. Sparse attention (mscale folded into the softmax scale, DSA V3.2).
    let mut attn = zeros(s * cfg.n_heads * hd)?;
    let sm_eff = cfg.sm_scale() * mscale(cfg.dsa.topk_tokens);
    if s == 1 {
        let mut phys = vec![-1i32; topk];
        for (t, &v) in sel.iter().enumerate() {
            if v >= 0 {
                phys[t] = cache.physical_index(seq, v as usize)?;
            }
        }
        let mut lse = zeros(cfg.n_heads)?;
        flash_mla_decode(
            &q,
            cache,
            &phys,
            1,
            1,
            topk,
            sm_eff,
            w.mla.attn_sink.as_deref(),
            &mut attn,
            &mut lse,
        )?;
    } else {
        let mut kv_all = zeros(total * hd)?;
        for p in 0..total {
            let phys = cache.physical_index(seq, p)?;
            let row = cache.row_f32(phys)?;
            kv_all[p * hd..(p + 1) * hd].copy_from_slice(&row);
        }
        let (out, _max_logits, _lse) = flash_mla_sparse_prefill(
            &q, &kv_all, &sel, s, cfg.n_heads, hd, total, topk, sm_eff,
            w.mla.attn_sink.as_deref(), None,
        )?;
        attn.copy_from_slice(&out);
    }

    // 6. O-RoPE conjugate + output projection + residual.
    o_rope_conjugate(rope, &q_positions, &mut attn, s * cfg.n_heads, hd)?;
    let mut proj = zeros(s * hidden)?;
    w.mla.project_o(&attn, s, &mut proj)?;
    let mut h_out = h_in.to_vec();
    for (o, p) in h_out.iter_mut().zip(proj.iter()) {
        *o += p;
    }

    // 7. Pre-FFN norm + local MoE (no expert parallelism on the CPU path)
    //    + residual.
    let mut x2 = h_out.clone();
    for i in 0..s {
        rmsnorm(&mut x2[i * hidden..(i + 1) * hidden], &w.ffn_norm, cfg.rms_eps);
    }
    let moe_out = deepinfer_moe::moe_forward(&w.moe, &cfg.moe, &x2, s)?;
    for (o, m) in h_out.iter_mut().zip(moe_out.iter()) {
        *o += m;
    }
    Ok(h_out)
}

/// One cross-decoder layer forward: SWA local self-attention + dense
/// cross-attention into the last self-decoder layer's KV cache, summed,
/// then the MoE FFN.
fn cross_layer_forward(
    cfg: &ModelConfig,
    w: &CrossLayerWeights,
    rope: &Rope,
    self_cache: &MlaPagedCache,
    ring: &mut SwaRingCache,
    seq: u32,
    h_in: &[f32],
    s: usize,
    pos0: usize,
) -> Result<Vec<f32>> {
    let hidden = cfg.hidden_size;
    let hd = cfg.head_dim;
    let latent = hd - cfg.rope_dim;
    let total = pos0 + s;

    // 1. Pre-attention RMSNorm (the ring stores the normed rows).
    let mut x = h_in.to_vec();
    for i in 0..s {
        rmsnorm(&mut x[i * hidden..(i + 1) * hidden], &w.attn_norm, cfg.rms_eps);
    }

    // 2. SWA local self-attention over the ring.
    for i in 0..s {
        ring.push(&x[i * hidden..(i + 1) * hidden])?;
    }
    let rows = ring.window_rows();
    let n = rows.len();
    let start = ring.start_position(total - 1);
    let kv_positions: Vec<usize> = (start..start + n).collect();
    let kv_flat: Vec<f32> = rows.concat();
    let swa_out = swa_attention(
        &x,
        &kv_flat,
        &kv_positions,
        pos0,
        s,
        1,
        hidden,
        cfg.swa_window,
        1.0 / (hidden as f32).sqrt(),
    )?;

    // 3. Cross-attention: q from the layer's own projection, keys/values
    //    from the self-decoder's (last layer) KV cache, dense over the
    //    most recent `cross_attn_window` positions (documented cap).
    let qhd = cfg.n_heads * hd;
    let mut q = zeros(s * qhd)?;
    for i in 0..s {
        matvec_into(
            &w.q_proj,
            &x[i * hidden..(i + 1) * hidden],
            qhd,
            hidden,
            &mut q[i * qhd..(i + 1) * qhd],
        )?;
    }
    let q_positions: Vec<u32> = (0..s * cfg.n_heads)
        .map(|r| (pos0 + r / cfg.n_heads) as u32)
        .collect();
    rope.apply_at(&q_positions, &mut q, hd, latent)?;

    let lo = total.saturating_sub(cfg.cross_attn_window);
    let nkeys = total - lo;
    let mut kv_keys = zeros(nkeys * hd)?;
    for (j, p) in (lo..total).enumerate() {
        let phys = self_cache.physical_index(seq, p)?;
        let row = self_cache.row_f32(phys)?;
        kv_keys[j * hd..(j + 1) * hd].copy_from_slice(&row);
    }
    let key_positions: Vec<usize> = (lo..total).collect();
    let q_pos_usize: Vec<usize> = (0..s).map(|i| pos0 + i).collect();
    let mut attn = zeros(s * qhd)?;
    cross_attention_dense(
        &q,
        &kv_keys,
        &key_positions,
        &q_pos_usize,
        cfg.n_heads,
        hd,
        cfg.sm_scale(),
        &mut attn,
    )?;
    o_rope_conjugate(rope, &q_positions, &mut attn, s * cfg.n_heads, hd)?;

    let mut proj = zeros(s * hidden)?;
    for i in 0..s {
        matvec_into(
            &w.o_proj,
            &attn[i * qhd..(i + 1) * qhd],
            hidden,
            qhd,
            &mut proj[i * hidden..(i + 1) * hidden],
        )?;
    }

    // 4. Sum the two attention outputs, add the residual.
    let mut h_out = h_in.to_vec();
    for i in 0..s * hidden {
        h_out[i] += proj[i] + swa_out[i];
    }

    // 5. Pre-FFN norm + MoE + residual.
    let mut x2 = h_out.clone();
    for i in 0..s {
        rmsnorm(&mut x2[i * hidden..(i + 1) * hidden], &w.ffn_norm, cfg.rms_eps);
    }
    let moe_out = deepinfer_moe::moe_forward(&w.moe, &cfg.moe, &x2, s)?;
    for (o, m) in h_out.iter_mut().zip(moe_out.iter()) {
        *o += m;
    }
    Ok(h_out)
}

/// Dense causal attention of `q [s, h, d]` rows against `kv [n, d]` rows
/// at absolute `key_positions` / `q_positions` (per-query causal mask:
/// key `j` is visible to query `i` iff `key_positions[j] <= q_positions[i]`).
/// Two-pass max-subtracted softmax, ascending key order — deterministic.
fn cross_attention_dense(
    q: &[f32],
    kv: &[f32],
    key_positions: &[usize],
    q_positions: &[usize],
    h: usize,
    d: usize,
    sm_scale: f32,
    out: &mut [f32],
) -> Result<()> {
    let s = q_positions.len();
    let n = key_positions.len();
    let q_len = s
        .checked_mul(h)
        .and_then(|v| v.checked_mul(d))
        .ok_or_else(|| Error::DimensionMismatch("s*h*d overflows".into()))?;
    ensure!(
        q.len() == q_len,
        DimensionMismatch,
        "q has {} elems, expected s*h*d = {q_len}",
        q.len()
    );
    let kv_len = n
        .checked_mul(d)
        .ok_or_else(|| Error::DimensionMismatch("n*d overflows".into()))?;
    ensure!(
        kv.len() == kv_len,
        DimensionMismatch,
        "kv has {} elems, expected n*d = {kv_len}",
        kv.len()
    );
    ensure!(
        out.len() == q.len(),
        DimensionMismatch,
        "out has {} elems, expected {}",
        out.len(),
        q.len()
    );
    let mut acc = vec![0.0f32; d];
    for i in 0..s {
        let qpos = q_positions[i];
        for hh in 0..h {
            let qrow = &q[(i * h + hh) * d..(i * h + hh + 1) * d];
            let orow = &mut out[(i * h + hh) * d..(i * h + hh + 1) * d];
            // Pass 1: max of the scaled scores over the causal window.
            let mut max = f32::NEG_INFINITY;
            for (j, &kp) in key_positions.iter().enumerate() {
                if kp > qpos {
                    continue;
                }
                let score = dot(qrow, &kv[j * d..(j + 1) * d]) * sm_scale;
                if score > max {
                    max = score;
                }
            }
            if max == f32::NEG_INFINITY {
                orow.fill(0.0);
                continue;
            }
            // Pass 2: softmax weights and the weighted KV sum.
            let mut sum = 0.0f32;
            acc.iter_mut().for_each(|a| *a = 0.0);
            for (j, &kp) in key_positions.iter().enumerate() {
                if kp > qpos {
                    continue;
                }
                let score = dot(qrow, &kv[j * d..(j + 1) * d]) * sm_scale;
                let wgt = (score - max).exp();
                sum += wgt;
                let krow = &kv[j * d..(j + 1) * d];
                for (a, k) in acc.iter_mut().zip(krow.iter()) {
                    *a += wgt * k;
                }
            }
            let inv = 1.0 / sum;
            for (o, a) in orow.iter_mut().zip(acc.iter()) {
                *o = a * inv;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_model(seed: u64) -> DeepSeekModel {
        DeepSeekModel::random_init(&ModelConfig::v41_tiny(), seed).unwrap()
    }

    #[test]
    fn prefill_decode_cache_growth() {
        let mut m = tiny_model(42);
        let prompt: Vec<u32> = (0..4u32).map(|t| (t * 37) % 250).collect();
        let logits = m.prefill(7, &prompt).unwrap();
        assert_eq!(logits.len(), 256);
        assert!(logits.iter().all(|v| v.is_finite()));
        assert_eq!(m.seq_len(7), 4);
        assert_eq!(m.total_cached_tokens(), 4);
        for l in 0..m.config.n_self_layers {
            assert_eq!(m.self_caches[l].seq_len(7), 4);
        }
        for _ in 0..3 {
            m.decode_step(7, 200).unwrap();
        }
        assert_eq!(m.seq_len(7), 7);
        assert_eq!(m.total_cached_tokens(), 7);
        // Unknown sequences read as empty.
        assert_eq!(m.seq_len(999), 0);
    }

    #[test]
    fn reset_clears_state() {
        let mut m = tiny_model(42);
        m.prefill(0, &[1, 2, 3]).unwrap();
        m.decode_step(0, 4).unwrap();
        assert_eq!(m.seq_len(0), 4);
        m.reset();
        assert_eq!(m.seq_len(0), 0);
        assert_eq!(m.total_cached_tokens(), 0);
        // The model is usable again and independent of the old state.
        let a = m.prefill(0, &[1, 2, 3]).unwrap();
        let mut fresh = tiny_model(42);
        let b = fresh.prefill(0, &[1, 2, 3]).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn prefill_validates_inputs() {
        let mut m = tiny_model(42);
        assert!(m.prefill(0, &[]).is_err());
        assert!(m.prefill(0, &[256]).is_err()); // vocab
        assert!(m.decode_step(0, 999).is_err()); // vocab
        // Position bound: max_position_embeddings = 256.
        let long: Vec<u32> = vec![1; 250];
        m.prefill(0, &long).unwrap();
        for _ in 0..6 {
            m.decode_step(0, 1).unwrap(); // 251..=256
        }
        assert!(m.decode_step(0, 1).is_err()); // 257 > 256
    }

    #[test]
    fn forward_is_bit_deterministic() {
        let prompt: Vec<u32> = (0..6u32).map(|t| (t * 61) % 256).collect();
        let mut a = tiny_model(7);
        let mut b = tiny_model(7);
        let la = a.prefill(0, &prompt).unwrap();
        let lb = b.prefill(0, &prompt).unwrap();
        assert_eq!(la, lb);
        let mut ta = 5u32;
        let mut tb = 5u32;
        for _ in 0..4 {
            let xa = a.decode_step(0, ta).unwrap();
            let xb = b.decode_step(0, tb).unwrap();
            assert_eq!(xa, xb);
            ta = deepinfer_core::math::argmax(&xa).unwrap() as u32;
            tb = deepinfer_core::math::argmax(&xb).unwrap() as u32;
            assert_eq!(ta, tb);
        }
        // Different seeds differ.
        let mut c = tiny_model(8);
        assert_ne!(c.prefill(0, &prompt).unwrap(), la);
    }

    #[test]
    fn rollback_restores_exact_state() {
        // Appending 3 tokens then rolling back 2 must equal appending 1:
        // the next decode after rollback reproduces the pre-rollback
        // logits bit-for-bit.
        let prompt: Vec<u32> = (0..5u32).map(|t| (t * 17) % 256).collect();
        let mut m = tiny_model(3);
        m.prefill(0, &prompt).unwrap();
        let l1 = m.decode_step(0, 11).unwrap();
        let l2 = m.decode_step(0, 12).unwrap();
        let l3 = m.decode_step(0, 13).unwrap();
        assert_eq!(m.seq_len(0), 8);
        m.rollback_seq(0, 2).unwrap();
        assert_eq!(m.seq_len(0), 6);
        let after = m.decode_step(0, 12).unwrap();
        assert_eq!(after, l2, "rollback must restore the exact state");
        assert_ne!(l1, l3);
        // Rolling back more than stored is an error; state is preserved.
        assert!(m.rollback_seq(0, 100).is_err());
        assert_eq!(m.seq_len(0), 7);
        // Rolling back the whole sequence empties it.
        m.rollback_seq(0, 7).unwrap();
        assert_eq!(m.seq_len(0), 0);
        assert!(m.rollback_seq(0, 1).is_err());
    }

    #[test]
    fn cross_attention_dense_is_causal_and_normalized() {
        // 1 head, d = 2; keys at positions [0, 1, 2], queries at [1, 2].
        let q = vec![1.0, 0.0, 1.0, 0.0];
        let kv = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut out = vec![0.0f32; 4];
        cross_attention_dense(&q, &kv, &[0, 1, 2], &[1, 2], 1, 2, 1.0, &mut out).unwrap();
        // Query at pos 1 sees keys 0 (score 1) and 1 (score 0):
        // weights [e/(e+1), 1/(e+1)] over rows [1,0] and [0,1].
        let e = std::f32::consts::E;
        assert!((out[0] - e / (e + 1.0)).abs() < 1e-5);
        assert!((out[1] - 1.0 / (e + 1.0)).abs() < 1e-5);
        // Query at pos 2 sees keys 0,1,2 (scores 1, 0, 1): weights
        // [p, w, p] with p = e/(2e+1), w = 1/(2e+1).
        let p = e / (2.0 * e + 1.0);
        let w = 1.0 / (2.0 * e + 1.0);
        assert!((out[2] - 2.0 * p).abs() < 1e-5);
        assert!((out[3] - (w + p)).abs() < 1e-5);
        // A query with no visible keys produces a zero row.
        let mut z = vec![0.0f32; 2];
        cross_attention_dense(&q[..2], &kv[2..], &[1, 2], &[0], 1, 2, 1.0, &mut z).unwrap();
        assert_eq!(z, vec![0.0f32; 2]);
        // Shape validation.
        assert!(
            cross_attention_dense(&q, &kv, &[0, 1, 2], &[1, 2], 1, 3, 1.0, &mut out).is_err()
        );
    }

    #[test]
    fn vocab_bytes_helpers() {
        let tok = deepinfer_core::tokenizer::ByteTokenizer::new(0).unwrap();
        let vb = DeepSeekModel::vocab_bytes_from_tokenizer(&tok, 0).unwrap();
        assert_eq!(vb.len(), 256);
        assert_eq!(vb[65], vec![b'A']);
        assert!(vb[0].is_empty(), "EOS bytes must be empty");
        let mut m = tiny_model(1);
        assert!(m.vocab_bytes().is_none());
        m.set_vocab_bytes(vb.clone());
        assert_eq!(m.vocab_bytes().unwrap().len(), 256);
        assert_eq!(m.resolve_vocab_bytes(0).unwrap(), vb);
        assert!(DeepSeekModel::vocab_bytes_from_tokenizer(&tok, 256).is_err());
    }
}
