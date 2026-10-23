//! The paged multi-sequence serving engine (`impl DeviceModel`).
//!
//! The single-sequence paths carve contiguous ranges out of flat arenas;
//! this engine gives the same model a **paged device cache** — per-layer
//! pools of 64-token pages shared by every attached sequence, per-sequence
//! page tables on the device, and the batched `di_serve_*` kernel family
//! (one launch covers the whole batch; per-slot geometry comes from a
//! device cell block, so nothing here needs a host readback). Sequences
//! attach, prefill, decode together, roll back and detach independently:
//!
//! ```text
//! serve_attach(1); serve_prefill(1, prompt);
//! serve_attach(2); serve_prefill(2, prompt);
//! loop { serve_decode(&[(1, tok1), (2, tok2)])?; sample(); }
//! serve_detach(1);   // pages return to the pool
//! ```
//!
//! Page indices are global: every layer's five pools (kv, k_raw, k_norm,
//! k_u8, feat) have identical page geometry, so one physical page id
//! addresses the same logical page in each of them and one shared free
//! list tracks ownership.
//!
//! The engine keeps the single-sequence path untouched (its CUDA-graph
//! decode capture operates on the flat arenas); both share weights, the
//! JIT library and the eager kernels. Numerics: every serve kernel is a
//! port of the corresponding arena kernel with the page indirection added,
//! so the CPU reference remains the golden oracle (the multi-sequence
//! golden test compares against the single-sequence device path, which is
//! itself CPU-validated).

use std::collections::BTreeMap;

use deepinfer_core::{ensure, Error, Result};
use deepinfer_gpu::Buffer;
use deepinfer_kernels::{CacheFormat as DevCacheFormat, GemmDType, ServeCellIn};

use crate::device_model::{
    kv_row_bytes, DeviceModel, MoeW, Precision, DEVICE_MAX_PREFILL_TOKENS, MOE_ALIGN,
    MOE_CHUNK_TOKENS,
};

/// Page size (kept in lockstep with the kernels).
const PAGE: usize = deepinfer_kernels::SERVE_PAGE;
/// Prefill chunk of the serve engine (matches the MoE prefill chunk).
pub const SERVE_CHUNK: usize = MOE_CHUNK_TOKENS;

/// One pool: `[total_pages, PAGE, row_bytes]`.
struct Pool {
    buf: Buffer,
    #[allow(dead_code)]
    row_bytes: usize,
    #[allow(dead_code)]
    total_pages: usize,
}

impl Pool {
    fn new(
        dev: &deepinfer_gpu::Device,
        total_pages: usize,
        row_bytes: usize,
    ) -> Result<Pool> {
        ensure!(total_pages > 0, InvalidArgument, "pool needs pages");
        let bytes = total_pages
            .checked_mul(PAGE)
            .and_then(|r| r.checked_mul(row_bytes))
            .ok_or_else(|| Error::CapacityExceeded("pool size overflows".into()))?;
        let buf = dev
            .alloc(bytes)
            .map_err(|e| Error::CapacityExceeded(format!("paged pool ({bytes} B): {e}")))?;
        Ok(Pool {
            buf,
            row_bytes,
            total_pages,
        })
    }
}

/// Per self-layer paged caches.
struct LayerPools {
    kv: Pool,
    k_raw: Pool,
    k_norm: Pool,
    k_u8: Pool,
    feat: Pool,
}

/// One attached sequence.
struct ServeSeq {
    len: usize,
    /// Physical page ids (one per logical 64-row page).
    pages: Vec<u32>,
    tbl: Buffer,
    /// Lightning scales `[max_blocks]` u8.
    k_sf: Buffer,
    /// Per cross layer: `[swa_window, hidden]` f32 ring.
    rings: Vec<Buffer>,
}

/// Serve scratch (fixed maxima; every buffer is allocated once).
struct ServeWs {
    h: Buffer,
    x: Buffer,
    fused: Buffer,
    qa: Buffer,
    kv_rows: Buffer,
    q: Buffer,
    q_idx: Buffer,
    q_idx_n: Buffer,
    q_idx_u8: Buffer,
    q_idx_sf: Buffer,
    k_new: Buffer,
    hw: Buffer,
    mean: Buffer,
    gt: Buffer,
    gc: Buffer,
    attn: Buffer,
    proj: Buffer,
    wv_tmp: Buffer,
    wv_g: Buffer,
    wv_out: Buffer,
    swa_out: Buffer,
    logits: Buffer,
    tokens: Buffer,
    pos_kv: Buffer,
    pos_q: Buffer,
    cells_in: Buffer,
    cells: Buffer,
    tbl_ptrs: Buffer,
    sf_ptrs: Buffer,
    ring_ptrs: Buffer,
    /// Score matrices `[score_rows, max_len]`.
    scores: Buffer,
    conv: Buffer,
    cands: Buffer,
    sel: Buffer,
    /// Decode split-KV partials `[rows*H, splits(, D)]`.
    pm: Buffer,
    pl: Buffer,
    po: Buffer,
    lse: Buffer,
    lightning_bits: Buffer,
    // Batched masked MoE (cap = decode batch bound) + aligned prefill.
    moe_counters: Buffer,
    moe_offsets: Buffer,
    moe_rank: Buffer,
    moe_drops: Buffer,
    moe_perm: Buffer,
    moe_map: Buffer,
    moe_aq: Option<Buffer>,
    moe_asf: Option<Buffer>,
    moe_act: Buffer,
    moe_silu: Buffer,
    moe_aq2: Option<Buffer>,
    moe_asf2: Option<Buffer>,
    moe_a16: Option<Buffer>,
    moe_down: Buffer,
    moe_y: Buffer,
    topk_w: Buffer,
    topk_idx: Buffer,
    router_logits: Buffer,
}

/// The paged-cache state of a [`DeviceModel`].
pub(crate) struct PagedState {
    layers: Vec<LayerPools>,
    /// The one shared free list (page ids are global across layers/kinds).
    free: Vec<u32>,
    seqs: BTreeMap<u32, ServeSeq>,
    max_seqs: usize,
    max_len: usize,
    max_pages_per_seq: usize,
    total_pages: usize,
    ws: Option<ServeWs>,
}

impl PagedState {
    pub(crate) fn empty() -> PagedState {
        PagedState {
            layers: Vec::new(),
            free: Vec::new(),
            seqs: BTreeMap::new(),
            max_seqs: 0,
            max_len: 0,
            max_pages_per_seq: 0,
            total_pages: 0,
            ws: None,
        }
    }
}

impl DeviceModel {
    // ------------------------------------------------------------------
    // Configuration and sequence lifecycle
    // ------------------------------------------------------------------

    /// Configure the paged serving caches: `max_seqs` concurrent
    /// sequences, each up to `max_len` tokens, sharing per-layer pools of
    /// `pages_per_layer` 64-token pages (per cache kind).
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] on degenerate geometry or when sequences
    /// are still attached; [`Error::CapacityExceeded`] when an allocation
    /// fails.
    pub fn serve_configure(
        &mut self,
        max_seqs: usize,
        max_len: usize,
        pages_per_layer: usize,
    ) -> Result<()> {
        ensure!(max_seqs >= 1, InvalidArgument, "max_seqs must be >= 1");
        ensure!(max_len >= 1, InvalidArgument, "max_len must be >= 1");
        ensure!(pages_per_layer >= 1, InvalidArgument, "pages_per_layer must be >= 1");
        ensure!(
            self.paged.seqs.is_empty(),
            InvalidArgument,
            "detach all sequences before reconfiguring the paged caches"
        );
        let cfg = self.cfg.clone();
        let hd = cfg.head_dim;
        let idx_dim = cfg.dsa.num_heads * cfg.dsa.head_dim;
        let kv_rb = kv_row_bytes(self.fmt, hd);
        let layers: Result<Vec<LayerPools>> = (0..cfg.n_self_layers)
            .map(|_| {
                Ok(LayerPools {
                    kv: Pool::new(&self.dev, pages_per_layer, kv_rb)?,
                    k_raw: Pool::new(&self.dev, pages_per_layer, idx_dim * 4)?,
                    k_norm: Pool::new(&self.dev, pages_per_layer, idx_dim * 4)?,
                    k_u8: Pool::new(&self.dev, pages_per_layer, idx_dim)?,
                    feat: Pool::new(&self.dev, pages_per_layer, idx_dim * 4)?,
                })
            })
            .collect();
        let layers = layers?;
        self.paged = PagedState {
            free: (0..pages_per_layer as u32).rev().collect(),
            layers,
            seqs: BTreeMap::new(),
            max_seqs,
            max_len,
            max_pages_per_seq: max_len.div_ceil(PAGE),
            total_pages: pages_per_layer,
            ws: None,
        };
        let ws = self.build_serve_ws(max_seqs)?;
        self.paged.ws = Some(ws);
        Ok(())
    }

    /// Allocate the serve scratch (`rows` = decode batch bound; the score
    /// matrices cover the prefill chunk).
    fn build_serve_ws(&self, rows: usize) -> Result<ServeWs> {
        let cfg = &self.cfg;
        let dev = &self.dev;
        let hidden = cfg.hidden_size;
        let hd = cfg.head_dim;
        let nh = cfg.dsa.num_heads;
        let ihd = cfg.dsa.head_dim;
        let idx_dim = nh * ihd;
        let qb = cfg.dsa.quant_block;
        let topk = self.geo.topk;
        let vocab = cfg.vocab_size;
        let max_len = self.paged.max_len;
        let score_rows = rows.max(SERVE_CHUNK);
        let moe = &cfg.moe;
        let inter = moe.intermediate;
        let e = moe.num_routed;
        let ki = moe.num_activated;
        let cap = rows.max(1); // decode bound: a token's k experts are distinct
        let fp8 = self.prec == Precision::Auto && hidden % 128 == 0 && inter % 128 == 0;
        let f = |n: usize| -> Result<Buffer> {
            dev.alloc(n.max(1) * 4)
                .map_err(|er| Error::CapacityExceeded(format!("serve workspace: {er}")))
        };
        let b = |n: usize| -> Result<Buffer> {
            dev.alloc(n.max(1))
                .map_err(|er| Error::CapacityExceeded(format!("serve workspace: {er}")))
        };
        let splits = topk.div_ceil(deepinfer_kernels::DECODE_CHUNK).max(1);
        Ok(ServeWs {
            h: f(score_rows * hidden)?,
            x: f(score_rows * hidden)?,
            fused: f(score_rows * (cfg.q_lora_rank + hd))?,
            qa: f(score_rows * cfg.q_lora_rank)?,
            kv_rows: f(score_rows * hd)?,
            q: f(score_rows * cfg.n_heads * hd)?,
            q_idx: f(score_rows * idx_dim)?,
            q_idx_n: f(score_rows * idx_dim)?,
            q_idx_u8: b(score_rows * idx_dim)?,
            q_idx_sf: b((score_rows / qb).max(1))?,
            k_new: f(score_rows * idx_dim)?,
            hw: f(score_rows * nh)?,
            mean: f(score_rows)?,
            gt: f(score_rows)?,
            gc: f(score_rows)?,
            attn: f(score_rows * cfg.n_heads * hd)?,
            proj: f(score_rows * hidden)?,
            wv_tmp: f(score_rows * (cfg.n_heads / cfg.o_groups) * hd)?,
            wv_g: f(score_rows * cfg.o_lora_rank)?,
            wv_out: f(score_rows * cfg.o_groups * cfg.o_lora_rank)?,
            swa_out: f(score_rows * hidden)?,
            logits: f(score_rows * vocab)?,
            tokens: b(score_rows * 4)?,
            pos_kv: b(score_rows * 4)?,
            pos_q: b(score_rows * cfg.n_heads * 4)?,
            cells_in: b(rows * 5 * 4)?,
            cells: b(rows * deepinfer_kernels::SERVE_CELL_FIELDS * 4)?,
            tbl_ptrs: b(rows * 8)?,
            sf_ptrs: b(rows * 8)?,
            ring_ptrs: b(cfg.n_cross_layers * rows * 8)?,
            scores: f(score_rows * max_len)?,
            conv: f(score_rows * max_len)?,
            cands: b(score_rows * self.geo.cands_cap * 8)?,
            sel: b(score_rows * topk * 4)?,
            pm: f(rows * cfg.n_heads * splits)?,
            pl: f(rows * cfg.n_heads * splits)?,
            po: f(rows * cfg.n_heads * splits * hd)?,
            lse: f(rows * cfg.n_heads)?,
            lightning_bits: f((score_rows / qb + 1).max(1))?,
            moe_counters: b(e * 4)?,
            moe_offsets: b((e + 1) * 4)?,
            moe_rank: b(score_rows * ki * 4)?,
            moe_drops: b(4)?,
            moe_perm: f(e * cap * hidden)?,
            moe_map: b(score_rows * ki * 4)?,
            moe_aq: if fp8 { Some(b(e * cap * hidden)?) } else { None },
            moe_asf: if fp8 {
                Some(b(e * cap * (hidden / 128))?)
            } else {
                None
            },
            moe_act: f(e * cap * 2 * inter)?,
            moe_silu: f(e * cap * inter)?,
            moe_aq2: if fp8 { Some(b(e * cap * inter)?) } else { None },
            moe_asf2: if fp8 {
                Some(b(e * cap * (inter / 128).max(1))?)
            } else {
                None
            },
            moe_a16: if fp8 { None } else { Some(f(e * cap * 2 * inter)?) },
            moe_down: f(e * cap * hidden)?,
            moe_y: f(score_rows * hidden)?,
            topk_w: f(score_rows * ki)?,
            topk_idx: b(score_rows * ki * 4)?,
            router_logits: f(score_rows * e)?,
        })
    }

    /// Attach a sequence to the paged engine (idempotent).
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] before [`DeviceModel::serve_configure`];
    /// [`Error::CapacityExceeded`] when the batch is full or an allocation
    /// fails.
    pub fn serve_attach(&mut self, seq: u32) -> Result<()> {
        ensure!(
            !self.paged.layers.is_empty(),
            InvalidArgument,
            "call serve_configure first"
        );
        if self.paged.seqs.contains_key(&seq) {
            return Ok(());
        }
        ensure!(
            self.paged.seqs.len() < self.paged.max_seqs,
            CapacityExceeded,
            "serve batch is full ({} sequences)",
            self.paged.max_seqs
        );
        let cfg = &self.cfg;
        let qb = cfg.dsa.quant_block;
        let max_blocks = (self.paged.max_len / qb).max(1);
        let tbl = self.dev.alloc(self.paged.max_pages_per_seq.max(1) * 4)?;
        let k_sf = self.dev.alloc(max_blocks)?;
        let rings: Result<Vec<Buffer>> = (0..cfg.n_cross_layers)
            .map(|_| self.dev.alloc(cfg.swa_window.max(1) * cfg.hidden_size * 4))
            .collect();
        self.paged.seqs.insert(
            seq,
            ServeSeq {
                len: 0,
                pages: Vec::new(),
                tbl,
                k_sf,
                rings: rings?,
            },
        );
        Ok(())
    }

    /// Detach a sequence; its pages return to the shared free list.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] for an unknown sequence.
    pub fn serve_detach(&mut self, seq: u32) -> Result<()> {
        let s = self
            .paged
            .seqs
            .remove(&seq)
            .ok_or_else(|| Error::invalid(format!("sequence {seq} is not attached")))?;
        let mut pages = s.pages;
        self.paged.free.extend_from_slice(&pages);
        pages.clear();
        Ok(())
    }

    /// Cached length of an attached sequence (0 when unknown).
    pub fn serve_len(&self, seq: u32) -> usize {
        self.paged.seqs.get(&seq).map(|s| s.len).unwrap_or(0)
    }

    /// Whether the sequence is attached to the paged engine.
    pub fn serve_attached(&self, seq: u32) -> bool {
        self.paged.seqs.contains_key(&seq)
    }

    /// Roll back `n` cached rows of an attached sequence (page tables keep
    /// their entries; re-appended rows overwrite their slots, exactly like
    /// the arena rollback).
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] for an unknown sequence or `n > len`.
    pub fn serve_rollback(&mut self, seq: u32, n: usize) -> Result<()> {
        let s = self
            .paged
            .seqs
            .get_mut(&seq)
            .ok_or_else(|| Error::invalid(format!("sequence {seq} is not attached")))?;
        ensure!(n <= s.len, InvalidArgument, "rollback beyond cached length");
        s.len -= n;
        Ok(())
    }

    /// Grow `seq`'s page table to cover `total` rows and upload it.
    fn serve_ensure_pages(&mut self, seq: u32, total: usize) -> Result<()> {
        let need = total.div_ceil(PAGE);
        ensure!(
            need <= self.paged.max_pages_per_seq,
            CapacityExceeded,
            "sequence would need {need} pages (budget {})",
            self.paged.max_pages_per_seq
        );
        let have = {
            let s = self.paged.seqs.get(&seq).ok_or_else(|| {
                Error::invalid(format!("sequence {seq} is not attached"))
            })?;
            s.pages.len()
        };
        if need > have {
            let take = need - have;
            ensure!(
                self.paged.free.len() >= take,
                CapacityExceeded,
                "paged cache exhausted ({} free of {} pages); raise the pool budget",
                self.paged.free.len(),
                self.paged.total_pages
            );
            let granted: Vec<u32> = (0..take)
                .map(|_| self.paged.free.pop().expect("checked above"))
                .collect();
            let tbl_words: Vec<i32> = {
                let s = self.paged.seqs.get_mut(&seq).expect("checked above");
                s.pages.extend_from_slice(&granted);
                s.pages.iter().map(|&p| p as i32).collect()
            };
            let view = self.paged.seqs[&seq]
                .tbl
                .slice(0, tbl_words.len() * 4)?;
            self.stream.write_async(&self.dev, &view, &tbl_words)?;
        }
        Ok(())
    }

    /// Upload the per-slot pointer tables (`tbl_ptrs`, `sf_ptrs`,
    /// `ring_ptrs [n_cross][B]`) and the host cells, then run the fanout.
    fn serve_begin_step(&self, slots: &[(u32, ServeCellIn)]) -> Result<()> {
        let ws = self
            .paged
            .ws
            .as_ref()
            .ok_or_else(|| Error::internal("serve workspace missing"))?;
        let b = slots.len();
        ensure!(b <= ws_rows_bound(self), InvalidArgument, "batch exceeds max_seqs");
        let mut in_cells = Vec::with_capacity(b * 5);
        let mut tbl = Vec::with_capacity(b);
        let mut sf = Vec::with_capacity(b);
        let mut rings = vec![0u64; self.cfg.n_cross_layers * b];
        for (i, (seq, cell)) in slots.iter().enumerate() {
            in_cells.extend_from_slice(&cell.words());
            let s = self
                .paged
                .seqs
                .get(seq)
                .ok_or_else(|| Error::invalid(format!("sequence {seq} is not attached")))?;
            tbl.push(s.tbl.ptr());
            sf.push(s.k_sf.ptr());
            for (l, ring) in s.rings.iter().enumerate() {
                rings[l * b + i] = ring.ptr();
            }
        }
        self.stream.write_async(&self.dev, &ws.cells_in, &in_cells)?;
        self.stream.write_async(&self.dev, &ws.tbl_ptrs, &tbl)?;
        self.stream.write_async(&self.dev, &ws.sf_ptrs, &sf)?;
        self.stream.write_async(&self.dev, &ws.ring_ptrs, &rings)?;
        let cfg = &self.cfg;
        self.k.serve_fanout(
            &ws.cells_in,
            &ws.cells,
            b,
            cfg.cross_attn_window,
            cfg.swa_window,
            cfg.dsa.quant_block,
            self.geo.topk,
            &self.stream,
        )
    }

    // ------------------------------------------------------------------
    // Prefill (one sequence, chunked; caches written through pages)
    // ------------------------------------------------------------------

    /// Prefill an attached sequence with `tokens`, returning the final
    /// position's logits. Chunks of [`SERVE_CHUNK`] tokens; the caches grow
    /// page by page.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] for unknown sequences / bad tokens;
    /// [`Error::CapacityExceeded`] when the page budget or max_len is hit.
    pub fn serve_prefill(&mut self, seq: u32, tokens: &[u32]) -> Result<Vec<f32>> {
        ensure!(!tokens.is_empty(), InvalidArgument, "prefill needs tokens");
        ensure!(
            tokens.len() <= DEVICE_MAX_PREFILL_TOKENS,
            CapacityExceeded,
            "prefill of {} tokens exceeds the device bound ({DEVICE_MAX_PREFILL_TOKENS})",
            tokens.len()
        );
        for &t in tokens {
            ensure!(
                (t as usize) < self.cfg.vocab_size,
                OutOfRange,
                "token {t} outside the vocabulary"
            );
        }
        let mut logits = Vec::new();
        let mut done = 0usize;
        while done < tokens.len() {
            let take = SERVE_CHUNK.min(tokens.len() - done);
            let chunk = &tokens[done..done + take];
            let l = self.serve_prefill_chunk(seq, chunk)?;
            if done + take == tokens.len() {
                logits = l;
            }
            done += take;
        }
        Ok(logits)
    }

    /// One prefill chunk (B = 1 slot, sq = chunk).
    fn serve_prefill_chunk(&mut self, seq: u32, tokens: &[u32]) -> Result<Vec<f32>> {
        let cfg = self.cfg.clone();
        let s = take_seq_len(&self.paged, seq)?;
        let pos0 = s;
        let sq = tokens.len();
        let total = pos0 + sq;
        ensure!(
            total <= self.paged.max_len,
            CapacityExceeded,
            "position {total} exceeds the serve cache bound ({})",
            self.paged.max_len
        );
        ensure!(
            total <= cfg.max_position_embeddings,
            CapacityExceeded,
            "position {total} exceeds max_position_embeddings"
        );
        self.serve_ensure_pages(seq, total)?;
        // Whole-history re-quant per chunk (the CPU prefill semantics).
        let sqnb = 0i32;
        self.serve_begin_step(&[(
            seq,
            ServeCellIn {
                active: 1,
                pos: pos0 as i32,
                len: pos0 as i32,
                sq: sq as i32,
                sqnb,
            },
        )])?;
        let ws = self.paged.ws.as_ref().expect("configured");
        let b = 1usize;
        self.run_serve_forward(sq, b, Some(tokens), seq)?;
        // Read the last row's logits.
        self.stream.synchronize()?;
        let mut logits = vec![0.0f32; cfg.vocab_size];
        let row = ws.logits.slice((sq - 1) * cfg.vocab_size * 4, cfg.vocab_size * 4)?;
        self.dev.read(&row, &mut logits)?;
        self.paged.seqs.get_mut(&seq).expect("attached").len = total;
        Ok(logits)
    }

    // ------------------------------------------------------------------
    // Decode (the batched hot loop)
    // ------------------------------------------------------------------

    /// One batched decode step: every `(sequence, token)` pair advances by
    /// one position; returns the per-sequence logits in batch order.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] on unknown sequences / empty batch /
    /// bad tokens; [`Error::CapacityExceeded`] when the page budget is hit.
    pub fn serve_decode(&mut self, batch: &[(u32, u32)]) -> Result<Vec<Vec<f32>>> {
        ensure!(!batch.is_empty(), InvalidArgument, "decode batch is empty");
        let cfg = self.cfg.clone();
        for &(seq, t) in batch {
            ensure!(
                self.paged.seqs.contains_key(&seq),
                InvalidArgument,
                "sequence {seq} is not attached to the serve engine"
            );
            ensure!(
                (t as usize) < cfg.vocab_size,
                OutOfRange,
                "token {t} outside the vocabulary"
            );
        }
        // Deduplicate sequences (one row per sequence per step).
        let mut slots: Vec<(u32, ServeCellIn)> = Vec::with_capacity(batch.len());
        for &(seq, _) in batch {
            if slots.iter().any(|(s, _)| *s == seq) {
                continue;
            }
            let len = self.paged.seqs[&seq].len;
            slots.push((
                seq,
                ServeCellIn {
                    active: 1,
                    pos: len as i32,
                    len: len as i32,
                    sq: 1,
                    sqnb: (len / cfg.dsa.quant_block) as i32,
                },
            ));
        }
        let b = slots.len();
        for (seq, cell) in &slots {
            let total = cell.len as usize + 1;
            ensure!(
                total <= self.paged.max_len,
                CapacityExceeded,
                "sequence {seq} would exceed the serve cache bound"
            );
            ensure!(
                total <= cfg.max_position_embeddings,
                CapacityExceeded,
                "sequence {seq} exceeds max_position_embeddings"
            );
            self.serve_ensure_pages(*seq, total)?;
        }
        self.serve_begin_step(&slots)?;
        let tokens: Vec<u32> = slots.iter().map(|(seq, _)| {
            batch.iter().find(|&&(s, _)| s == *seq).expect("present").1
        })
        .collect();
        self.run_serve_forward(1, b, Some(&tokens), slots[0].0)?;
        // Read per-sequence logits.
        self.stream.synchronize()?;
        let ws = self.paged.ws.as_ref().expect("configured");
        let mut out = Vec::with_capacity(b);
        for i in 0..b {
            let mut logits = vec![0.0f32; cfg.vocab_size];
            let row = ws.logits.slice(i * cfg.vocab_size * 4, cfg.vocab_size * 4)?;
            self.dev.read(&row, &mut logits)?;
            out.push(logits);
        }
        for (seq, cell) in &slots {
            let s = self.paged.seqs.get_mut(seq).expect("attached");
            s.len = cell.len as usize + 1;
        }
        Ok(out)
    }

    // ------------------------------------------------------------------
    // The shared forward (sq rows per slot, B slots)
    // ------------------------------------------------------------------

    /// The serve forward over the paged caches. `tokens` (already staged
    /// or given) are embedded into `ws.h`; layers run exactly like the
    /// single-sequence paths but with `di_serve_*` kernels and the shared
    /// pools. `anchor` is any attached sequence (unused except for the
    /// cross-layer pools which read the LAST self layer of each slot's own
    /// caches — the page tables carry the indirection).
    #[allow(clippy::too_many_arguments)]
    fn run_serve_forward(
        &self,
        sq: usize,
        b: usize,
        tokens: Option<&[u32]>,
        _anchor: u32,
    ) -> Result<()> {
        let cfg = &self.cfg;
        let s = &self.stream;
        let ws = self.paged.ws.as_ref().ok_or_else(|| Error::internal("serve ws"))?;
        let rows = sq * b;
        let hidden = cfg.hidden_size;
        let hd = cfg.head_dim;
        let latent = hd - cfg.rope_dim;
        let half = cfg.rope_dim / 2;
        let nh = cfg.dsa.num_heads;
        let ihd = cfg.dsa.head_dim;
        let idx_dim = nh * ihd;
        let qb = cfg.dsa.quant_block;
        let window = cfg.dsa.extrapolation_window.unwrap_or(0);
        let topk = self.geo.topk;
        let sm_eff = cfg.sm_scale() * deepinfer_dsa::mscale(cfg.dsa.topk_tokens);
        let max_cols = self.paged.max_len;

        // Embedding gather + positions.
        if let Some(toks) = tokens {
            self.stream.write_async(&self.dev, &ws.tokens, toks)?;
        }
        self.k.gather_indexed(
            &self.root.embed,
            &ws.h,
            &ws.tokens,
            rows,
            hidden * 4,
            s,
        )?;
        self.k.serve_pos_fill(&ws.cells, &ws.pos_kv, &ws.pos_q, b, sq, cfg.n_heads, s)?;

        for l in 0..cfg.n_self_layers {
            let (w, pools) = (&self.self_layers[l].0, &self.paged.layers[l]);
            self.k
                .rmsnorm(&ws.h, &w.attn_norm, &ws.x, rows, hidden, cfg.rms_eps, s)?;
            self.gemm(&w.fused_qkv, &ws.x, &ws.fused, rows)?;
            self.k.copy_cols(
                &ws.fused,
                &ws.qa,
                rows,
                cfg.q_lora_rank + hd,
                cfg.q_lora_rank,
                0,
                0,
                cfg.q_lora_rank,
                s,
            )?;
            self.k.copy_cols(
                &ws.fused,
                &ws.kv_rows,
                rows,
                cfg.q_lora_rank + hd,
                hd,
                cfg.q_lora_rank,
                0,
                hd,
                s,
            )?;
            self.k.rmsnorm_latent(
                &ws.kv_rows,
                &w.kv_norm,
                rows,
                hd,
                latent,
                crate::weights::V41_RMS_EPS,
                s,
            )?;
            self.k.rope(
                &self.root.rope_cs,
                &ws.pos_kv,
                &ws.kv_rows,
                rows,
                hd,
                latent,
                half,
                false,
                s,
            )?;
            self.gemm(&w.q_b, &ws.qa, &ws.q, rows)?;
            self.k.rope(
                &self.root.rope_cs,
                &ws.pos_q,
                &ws.q,
                rows * cfg.n_heads,
                hd,
                latent,
                half,
                false,
                s,
            )?;
            // Paged cache writes.
            let fmt = match self.fmt {
                DevCacheFormat::F32 => deepinfer_kernels::CacheFormat::F32,
                DevCacheFormat::Fp8Group32 => deepinfer_kernels::CacheFormat::Fp8Group32,
                DevCacheFormat::Fp4Group16 => deepinfer_kernels::CacheFormat::Fp4Group16,
            };
            self.k
                .serve_kv_encode(&ws.kv_rows, &pools.kv.buf, &ws.tbl_ptrs, &ws.cells, hd, fmt, sq, b, s)?;
            // DSA indexer.
            self.gemm(&w.wq, &ws.h, &ws.q_idx, rows)?;
            self.gemm(&w.wk, &ws.h, &ws.k_new, rows)?;
            self.k
                .serve_append_rows(&ws.k_new, &pools.k_raw.buf, &ws.tbl_ptrs, &ws.cells, idx_dim, sq, b, s)?;
            self.k.serve_append_norm_rows(
                &ws.k_new,
                &pools.k_norm.buf,
                &ws.tbl_ptrs,
                &ws.cells,
                nh,
                ihd,
                crate::weights::V41_RMS_EPS,
                sq,
                b,
                s,
            )?;
            self.k.serve_conv_feat(
                &pools.k_raw.buf,
                &pools.feat.buf,
                &ws.tbl_ptrs,
                &ws.cells,
                &w.conv_w,
                &w.conv_off,
                w.conv_kernels,
                idx_dim,
                sq,
                b,
                s,
            )?;
            self.k.serve_conv_gated_norm(
                &pools.feat.buf,
                &ws.tbl_ptrs,
                &ws.cells,
                &w.conv_gain,
                idx_dim,
                sq,
                b,
                s,
            )?;
            let max_blocks = (self.paged.max_len / qb).max(1);
            self.k.serve_lightning_quant(
                &pools.k_norm.buf,
                &pools.k_u8.buf,
                &ws.tbl_ptrs,
                &ws.sf_ptrs,
                &ws.cells,
                idx_dim,
                qb,
                b,
                max_blocks,
                s,
            )?;
            self.gemm(&w.weights_proj, &ws.q_idx, &ws.hw, rows)?;
            {
                let bytes = rows * idx_dim * 4;
                let dst = ws.q_idx_n.slice(0, bytes)?;
                let src = ws.q_idx.slice(0, bytes)?;
                self.dev.memcpy_dtod_async(&dst, &src, s)?;
            }
            self.k
                .indexer_rmsnorm_per_head(&ws.q_idx_n, rows, nh, ihd, s)?;
            let q_nblocks = sq / qb;
            self.k.lightning_quant(
                &ws.q_idx_n,
                &ws.q_idx_u8,
                &ws.q_idx_sf,
                &ws.lightning_bits,
                rows,
                idx_dim,
                qb,
                s,
            )?;
            self.k.serve_thunder_logits(
                &ws.q_idx_n,
                &ws.q_idx_u8,
                &ws.q_idx_sf,
                &pools.k_norm.buf,
                &pools.k_u8.buf,
                &ws.tbl_ptrs,
                &ws.sf_ptrs,
                &ws.cells,
                &ws.hw,
                &ws.scores,
                nh,
                ihd,
                qb,
                q_nblocks,
                max_cols,
                sq,
                b,
                s,
            )?;
            self.k.serve_dot_scores(
                &ws.q_idx,
                &pools.feat.buf,
                &ws.tbl_ptrs,
                &ws.cells,
                &ws.conv,
                idx_dim,
                max_cols,
                sq,
                b,
                s,
            )?;
            self.k.qmean(&ws.h, &ws.mean, rows, hidden, s)?;
            self.k.gate_logits(
                &ws.h,
                &ws.mean,
                &w.gate_proj,
                &ws.gt,
                &ws.gc,
                rows,
                hidden,
                s,
            )?;
            self.k.serve_gate_blend(&ws.gt, &ws.gc, &ws.conv, &ws.scores, &ws.cells, max_cols, sq, b, s)?;
            self.k.serve_causal_clip(&ws.scores, &ws.cells, max_cols, window, sq, b, s)?;
            self.k.fill_zero(&ws.cands, rows * self.geo.cands_cap * 8, s)?;
            self.k.serve_block_preselect(
                &ws.scores,
                &ws.cands,
                &ws.cells,
                self.geo.block,
                self.geo.per,
                self.geo.num_blocks,
                max_cols,
                sq,
                b,
                s,
            )?;
            self.k
                .serve_topk(&ws.cands, &ws.sel, &ws.cells, self.geo.num_blocks, self.geo.per, topk, sq, b, s)?;
            // Attention: sparse prefill (sq > 1) or split-KV decode (sq == 1).
            let last = cfg.n_self_layers - 1;
            let kv_pool = &self.paged.layers[last].kv.buf;
            if sq > 1 {
                self.k.serve_prefill_sparse(
                    &ws.q,
                    kv_pool,
                    &ws.tbl_ptrs,
                    &ws.cells,
                    &ws.sel,
                    &ws.attn,
                    None,
                    cfg.n_heads,
                    hd,
                    topk,
                    sm_eff,
                    fmt,
                    sq,
                    b,
                    s,
                )?;
            } else {
                let chunk = deepinfer_kernels::DECODE_CHUNK;
                let splits = topk.div_ceil(chunk).max(1);
                self.k.serve_mla_decode_partial(
                    &ws.q,
                    kv_pool,
                    &ws.tbl_ptrs,
                    &ws.cells,
                    &ws.sel,
                    &ws.pm,
                    &ws.pl,
                    &ws.po,
                    cfg.n_heads,
                    hd,
                    topk,
                    chunk,
                    sm_eff,
                    fmt,
                    sq,
                    b,
                    s,
                )?;
                self.k.serve_mla_decode_combine(
                    &ws.pm,
                    &ws.pl,
                    &ws.po,
                    w.sink.as_ref(),
                    &ws.attn,
                    Some(&ws.lse),
                    &ws.cells,
                    cfg.n_heads,
                    hd,
                    splits,
                    sq,
                    b,
                    s,
                )?;
            }
            // O-RoPE + output projections + residual.
            self.k.rope(
                &self.root.rope_cs,
                &ws.pos_q,
                &ws.attn,
                rows * cfg.n_heads,
                hd,
                latent,
                half,
                true,
                s,
            )?;
            // Grouped wv: per group, gather the head columns, project into
            // `wv_g`, then scatter into the group-strided `wv_out` columns
            // (the same dance as the arena prefill, batch-general).
            let r = (cfg.n_heads / cfg.o_groups) * hd;
            for g in 0..cfg.o_groups {
                self.k.copy_cols(
                    &ws.attn,
                    &ws.wv_tmp,
                    rows,
                    cfg.n_heads * hd,
                    r,
                    g * r,
                    0,
                    r,
                    s,
                )?;
                self.gemm(&w.wv[g], &ws.wv_tmp, &ws.wv_g, rows)?;
                self.k.copy_cols(
                    &ws.wv_g,
                    &ws.wv_out,
                    rows,
                    cfg.o_lora_rank,
                    cfg.o_groups * cfg.o_lora_rank,
                    0,
                    g * cfg.o_lora_rank,
                    cfg.o_lora_rank,
                    s,
                )?;
            }
            self.gemm(&w.wo, &ws.wv_out, &ws.proj, rows)?;
            self.k.add(&ws.proj, &ws.h, rows * hidden, s)?;
            self.k
                .rmsnorm(&ws.h, &w.ffn_norm, &ws.x, rows, hidden, cfg.rms_eps, s)?;
            if sq > 1 {
                // Prefill: the aligned-contiguous MoE path on the shared
                // activations (chunk <= MOE_CHUNK_TOKENS by construction).
                self.serve_moe_prefill(&w.moe, rows, ws)?;
            } else {
                self.serve_moe_decode(&w.moe, b, ws)?;
            }
            self.k.add(&ws.moe_y, &ws.h, rows * hidden, s)?;
        }

        // Cross-decoder layers.
        if cfg.n_cross_layers > 0 {
            let last = cfg.n_self_layers - 1;
            let kv_pool = &self.paged.layers[last].kv.buf;
            let fmt = match self.fmt {
                DevCacheFormat::F32 => deepinfer_kernels::CacheFormat::F32,
                DevCacheFormat::Fp8Group32 => deepinfer_kernels::CacheFormat::Fp8Group32,
                DevCacheFormat::Fp4Group16 => deepinfer_kernels::CacheFormat::Fp4Group16,
            };
            let sm = 1.0f32 / (hidden as f32).sqrt();
            for l in 0..cfg.n_cross_layers {
                let w = &self.cross_layers[l];
                self.k
                    .rmsnorm(&ws.h, &w.attn_norm, &ws.x, rows, hidden, cfg.rms_eps, s)?;
                self.k.serve_ring_append(
                    &ws.x,
                    &ws.ring_ptrs,
                    &ws.cells,
                    hidden,
                    cfg.swa_window,
                    cfg.n_cross_layers,
                    sq,
                    b,
                    s,
                )?;
                // SWA per cross layer (layer-major ring view).
                let ring_view = ws
                    .ring_ptrs
                    .slice(l * b * 8, b * 8)?;
                let f = self.k.f("di_serve_swa_attention_kernel")?;
                f.launch(
                    &self.dev,
                    deepinfer_gpu::Launch {
                        grid: (rows as u32, 1, 1),
                        block: (256, 1, 1),
                        shared: (hidden * 4 + 2 * 256 * 4) as u32,
                    },
                    &deepinfer_gpu::KernelArgs::new()
                        .ptr(&ws.x)
                        .ptr(&ring_view)
                        .ptr(&ws.cells)
                        .ptr(&ws.swa_out)
                        .i32(hidden as i32)
                        .i32(cfg.swa_window as i32)
                        .f32(sm)
                        .i32(sq as i32)
                        .i32(b as i32),
                    s,
                )?;
                self.gemm(&w.q_proj, &ws.x, &ws.q, rows)?;
                self.k.rope(
                    &self.root.rope_cs,
                    &ws.pos_q,
                    &ws.q,
                    rows * cfg.n_heads,
                    hd,
                    latent,
                    half,
                    false,
                    s,
                )?;
                self.k.serve_prefill_dense(
                    &ws.q,
                    kv_pool,
                    &ws.tbl_ptrs,
                    &ws.cells,
                    &ws.attn,
                    cfg.n_heads,
                    hd,
                    cfg.sm_scale(),
                    fmt,
                    sq,
                    b,
                    s,
                )?;
                self.k.rope(
                    &self.root.rope_cs,
                    &ws.pos_q,
                    &ws.attn,
                    rows * cfg.n_heads,
                    hd,
                    latent,
                    half,
                    true,
                    s,
                )?;
                self.gemm(&w.o_proj, &ws.attn, &ws.proj, rows)?;
                self.k.add(&ws.proj, &ws.h, rows * hidden, s)?;
                self.k.add(&ws.swa_out, &ws.h, rows * hidden, s)?;
                self.k
                    .rmsnorm(&ws.h, &w.ffn_norm, &ws.x, rows, hidden, cfg.rms_eps, s)?;
                if sq > 1 {
                    self.serve_moe_prefill(&w.moe, rows, ws)?;
                } else {
                    self.serve_moe_decode(&w.moe, b, ws)?;
                }
                self.k.add(&ws.moe_y, &ws.h, rows * hidden, s)?;
            }
        }

        // Final norm + LM head.
        self.k.rmsnorm(
            &ws.h,
            &self.root.final_norm,
            &ws.x,
            rows,
            hidden,
            cfg.rms_eps,
            s,
        )?;
        self.gemm(&self.root.lm_head, &ws.x, &ws.logits, rows)?;
        Ok(())
    }

    // ------------------------------------------------------------------
    // MoE over the serve workspace
    // ------------------------------------------------------------------

    /// Aligned-contiguous MoE for prefill rows (the arena `moe_prefill`
    /// flow — router, aligned dispatch with a segment readback, grouped
    /// FP8 or per-expert dense GEMMs, combine — over the serve workspace;
    /// staging is per call, prefill not being the hot loop).
    #[allow(clippy::too_many_arguments)]
    fn serve_moe_prefill(&self, m: &MoeW, rows: usize, ws: &ServeWs) -> Result<()> {
        let cfg = &self.cfg;
        let hidden = cfg.moe.hidden;
        let inter = cfg.moe.intermediate;
        let e = cfg.moe.num_routed;
        let ki = cfg.moe.num_activated;
        let s = &self.stream;
        let align = MOE_ALIGN;
        self.gemm(&m.router, &ws.x, &ws.router_logits, rows)?;
        self.k.moe_route(
            &ws.router_logits,
            m.router_bias.as_ref(),
            &ws.topk_w,
            &ws.topk_idx,
            rows,
            e,
            ki,
            s,
        )?;
        let bound = rows
            .checked_mul(ki)
            .and_then(|v| v.checked_add(127 * v.min(e)))
            .ok_or_else(|| Error::CapacityExceeded("moe row bound overflows".into()))?
            .next_multiple_of(align);
        let perm = self.dev.alloc(bound.max(1) * hidden * 4)?;
        let m_total = self.k.moe_dispatch(
            &ws.x,
            &ws.topk_idx,
            &ws.moe_counters,
            &ws.moe_offsets,
            &ws.moe_rank,
            &perm,
            &ws.moe_map,
            rows,
            hidden,
            ki,
            e,
            align,
            s,
        )?;
        self.stream.synchronize()?;
        let mut counts = vec![0i32; e];
        self.dev.read(&ws.moe_counters, &mut counts)?;
        let act = self.dev.alloc(m_total.max(1) * 2 * inter * 4)?;
        let silu = self.dev.alloc(m_total.max(1) * inter * 4)?;
        let down = self.dev.alloc(m_total.max(1) * hidden * 4)?;
        match m.routed_gate_up.dtype {
            GemmDType::Fp8 => {
                let aq = self.dev.alloc(m_total.max(1) * hidden)?;
                let asf = self.dev.alloc(m_total.max(1) * (hidden / 128).max(1))?;
                self.k.quant_per_token_group(&perm, &aq, &asf, m_total, hidden, 128, s)?;
                self.k.gemm_fp8_grouped_nt(
                    &aq,
                    &asf,
                    m.routed_gate_up.q.as_ref().unwrap(),
                    m.routed_gate_up.sf.as_ref().unwrap(),
                    &ws.moe_offsets,
                    &act,
                    m_total,
                    2 * inter,
                    hidden,
                    e,
                    s,
                )?;
            }
            dtype => {
                let mut off = 0usize;
                for ex in 0..e {
                    let cnt = counts[ex].max(0) as usize;
                    if cnt > 0 {
                        let a = perm.slice(off * hidden * 4, cnt * hidden * 4)?;
                        let d = act.slice(off * 2 * inter * 4, cnt * 2 * inter * 4)?;
                        match dtype {
                            GemmDType::Fp8 => unreachable!(),
                            GemmDType::Bf16 => {
                                let a16 = self.dev.alloc(cnt * hidden * 2)?;
                                self.k.f32_to_bf16(&a, &a16, cnt * hidden, s)?;
                                self.k.gemm_bf16_nt(
                                    &a16,
                                    m.routed_gate_up.raw.as_ref().unwrap(),
                                    None,
                                    &d,
                                    cnt,
                                    2 * inter,
                                    hidden,
                                    s,
                                )?;
                            }
                            GemmDType::F32 => {
                                self.k.gemm_f32_nt(
                                    &a,
                                    m.routed_gate_up.raw.as_ref().unwrap(),
                                    None,
                                    &d,
                                    cnt,
                                    2 * inter,
                                    hidden,
                                    s,
                                )?;
                            }
                        }
                    }
                    off += cnt.next_multiple_of(align);
                }
            }
        }
        self.k.silu_mul_fused(&act, &silu, m_total, inter, s)?;
        match m.routed_down.dtype {
            GemmDType::Fp8 => {
                let aq = self.dev.alloc(m_total.max(1) * inter)?;
                let asf = self.dev.alloc(m_total.max(1) * (inter / 128).max(1))?;
                self.k.quant_per_token_group(&silu, &aq, &asf, m_total, inter, 128, s)?;
                self.k.gemm_fp8_grouped_nt(
                    &aq,
                    &asf,
                    m.routed_down.q.as_ref().unwrap(),
                    m.routed_down.sf.as_ref().unwrap(),
                    &ws.moe_offsets,
                    &down,
                    m_total,
                    hidden,
                    inter,
                    e,
                    s,
                )?;
            }
            dtype => {
                let mut off = 0usize;
                for ex in 0..e {
                    let cnt = counts[ex].max(0) as usize;
                    if cnt > 0 {
                        let a = silu.slice(off * inter * 4, cnt * inter * 4)?;
                        let d = down.slice(off * hidden * 4, cnt * hidden * 4)?;
                        match dtype {
                            GemmDType::Fp8 => unreachable!(),
                            GemmDType::Bf16 => {
                                let a16 = self.dev.alloc(cnt * inter * 2)?;
                                self.k.f32_to_bf16(&a, &a16, cnt * inter, s)?;
                                self.k.gemm_bf16_nt(
                                    &a16,
                                    m.routed_down.raw.as_ref().unwrap(),
                                    None,
                                    &d,
                                    cnt,
                                    hidden,
                                    inter,
                                    s,
                                )?;
                            }
                            GemmDType::F32 => {
                                self.k.gemm_f32_nt(
                                    &a,
                                    m.routed_down.raw.as_ref().unwrap(),
                                    None,
                                    &d,
                                    cnt,
                                    hidden,
                                    inter,
                                    s,
                                )?;
                            }
                        }
                    }
                    off += cnt.next_multiple_of(align);
                }
            }
        }
        self.k.moe_combine(
            &down,
            &ws.moe_map,
            &ws.topk_w,
            &ws.moe_y,
            rows,
            hidden,
            ki,
            s,
        )?;
        for (gu, dn) in &m.shared {
            let su = self.dev.alloc(rows * 2 * inter * 4)?;
            let so = self.dev.alloc(rows * inter * 4)?;
            let sd = self.dev.alloc(rows * hidden * 4)?;
            self.gemm(gu, &ws.x, &su, rows)?;
            self.k.silu_mul_fused(&su, &so, rows, inter, s)?;
            self.gemm(dn, &so, &sd, rows)?;
            self.k.add(&sd, &ws.moe_y, rows * hidden, s)?;
        }
        Ok(())
    }

    /// Masked-capacity MoE for the decode batch (graph-safe, no readback).
    fn serve_moe_decode(&self, m: &MoeW, tokens: usize, ws: &ServeWs) -> Result<()> {
        let cfg = &self.cfg;
        let hidden = cfg.moe.hidden;
        let inter = cfg.moe.intermediate;
        let e = cfg.moe.num_routed;
        let ki = cfg.moe.num_activated;
        let cap = tokens; // a token's k experts are distinct -> <= tokens per expert
        let rows = e * cap;
        let s = &self.stream;
        self.gemm(&m.router, &ws.x, &ws.router_logits, tokens)?;
        self.k.moe_route(
            &ws.router_logits,
            m.router_bias.as_ref(),
            &ws.topk_w,
            &ws.topk_idx,
            tokens,
            e,
            ki,
            s,
        )?;
        self.k.moe_dispatch_masked(
            &ws.x,
            &ws.topk_idx,
            &ws.moe_counters,
            &ws.moe_perm,
            &ws.moe_map,
            &ws.moe_drops,
            tokens,
            hidden,
            ki,
            e,
            cap,
            s,
        )?;
        match m.routed_gate_up.dtype {
            GemmDType::Fp8 => {
                let aq = ws.moe_aq.as_ref().unwrap();
                let asf = ws.moe_asf.as_ref().unwrap();
                self.k.quant_per_token_group(&ws.moe_perm, aq, asf, rows, hidden, 128, s)?;
                self.k.gemm_grouped_masked_nt(
                    aq,
                    Some(asf),
                    m.routed_gate_up.q.as_ref().unwrap(),
                    m.routed_gate_up.sf.as_ref(),
                    &ws.moe_counters,
                    &ws.moe_act,
                    cap,
                    2 * inter,
                    hidden,
                    e,
                    GemmDType::Fp8,
                    s,
                )?;
            }
            GemmDType::Bf16 => {
                let a16 = ws.moe_a16.as_ref().unwrap();
                self.k.f32_to_bf16(&ws.moe_perm, a16, rows * hidden, s)?;
                self.k.gemm_grouped_masked_nt(
                    a16,
                    None,
                    m.routed_gate_up.raw.as_ref().unwrap(),
                    None,
                    &ws.moe_counters,
                    &ws.moe_act,
                    cap,
                    2 * inter,
                    hidden,
                    e,
                    GemmDType::Bf16,
                    s,
                )?;
            }
            GemmDType::F32 => {
                self.k.gemm_grouped_masked_nt(
                    &ws.moe_perm,
                    None,
                    m.routed_gate_up.raw.as_ref().unwrap(),
                    None,
                    &ws.moe_counters,
                    &ws.moe_act,
                    cap,
                    2 * inter,
                    hidden,
                    e,
                    GemmDType::F32,
                    s,
                )?;
            }
        }
        self.k.silu_mul_masked(&ws.moe_act, &ws.moe_silu, &ws.moe_counters, cap, e, inter, s)?;
        match m.routed_down.dtype {
            GemmDType::Fp8 => {
                let aq = ws.moe_aq2.as_ref().unwrap();
                let asf = ws.moe_asf2.as_ref().unwrap();
                self.k.quant_per_token_group(&ws.moe_silu, aq, asf, rows, inter, 128, s)?;
                self.k.gemm_grouped_masked_nt(
                    aq,
                    Some(asf),
                    m.routed_down.q.as_ref().unwrap(),
                    m.routed_down.sf.as_ref(),
                    &ws.moe_counters,
                    &ws.moe_down,
                    cap,
                    hidden,
                    inter,
                    e,
                    GemmDType::Fp8,
                    s,
                )?;
            }
            GemmDType::Bf16 => {
                let a16_full = ws
                    .moe_a16
                    .as_ref()
                    .ok_or_else(|| Error::internal("BF16 staging missing for serve down"))?;
                let a16 = a16_full.slice(0, rows * inter * 2)?;
                self.k.f32_to_bf16(&ws.moe_silu, &a16, rows * inter, s)?;
                self.k.gemm_grouped_masked_nt(
                    &a16,
                    None,
                    m.routed_down.raw.as_ref().unwrap(),
                    None,
                    &ws.moe_counters,
                    &ws.moe_down,
                    cap,
                    hidden,
                    inter,
                    e,
                    GemmDType::Bf16,
                    s,
                )?;
            }
            GemmDType::F32 => {
                self.k.gemm_grouped_masked_nt(
                    &ws.moe_silu,
                    None,
                    m.routed_down.raw.as_ref().unwrap(),
                    None,
                    &ws.moe_counters,
                    &ws.moe_down,
                    cap,
                    hidden,
                    inter,
                    e,
                    GemmDType::F32,
                    s,
                )?;
            }
        }
        self.k.moe_combine(
            &ws.moe_down,
            &ws.moe_map,
            &ws.topk_w,
            &ws.moe_y,
            tokens,
            hidden,
            ki,
            s,
        )?;
        for (gu, dn) in &m.shared {
            let su = ws.moe_act.slice(0, tokens * 2 * inter * 4)?;
            let so = ws.moe_silu.slice(0, tokens * inter * 4)?;
            let sd = ws.moe_down.slice(0, tokens * hidden * 4)?;
            self.gemm(gu, &ws.x, &su, tokens)?;
            self.k.silu_mul_fused(&su, &so, tokens, inter, s)?;
            self.gemm(dn, &so, &sd, tokens)?;
            self.k.add(&sd, &ws.moe_y, tokens * hidden, s)?;
        }
        Ok(())
    }
}

/// Bound check helper for the batch size.
fn ws_rows_bound(m: &DeviceModel) -> usize {
    m.paged.max_seqs
}

fn take_seq_len(p: &PagedState, seq: u32) -> Result<usize> {
    p.seqs
        .get(&seq)
        .map(|s| s.len)
        .ok_or_else(|| Error::invalid(format!("sequence {seq} is not attached")))
}
