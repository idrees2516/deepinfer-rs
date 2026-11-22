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
use crate::prefix_cache::{ksf_seg_range, GpuPrefixTree, PrefixCacheStats};

/// Page size (kept in lockstep with the kernels).
const PAGE: usize = deepinfer_kernels::SERVE_PAGE;
/// Prefill chunk of the serve engine (matches the MoE prefill chunk).
pub const SERVE_CHUNK: usize = MOE_CHUNK_TOKENS;

/// One device-sampled decode result: the sampled token plus the
/// speculative-decoding anchor hidden row (the `[vocab]` logits never
/// reach the host — see [`DeviceModel::serve_decode_sampled`]).
#[derive(Debug, Clone, PartialEq)]
pub struct SampledOut {
    /// The sampled token id.
    pub token: u32,
    /// Residual hidden state at the sampled position (the MTP draft
    /// input; `[hidden]`).
    pub hidden: Vec<f32>,
}

/// An in-flight device-sampled decode step (P0-5): the slim outputs are
/// on their way back through asynchronous D2H copies guarded by an
/// event, so the engine thread can queue further GPU work (the step's
/// prefill chunks) before consuming them — the vLLM-V1 overlap shape,
/// one step deep, without speculative scheduling. Produced by
/// [`DeviceModel::serve_decode_sampled_stage`], consumed by
/// [`DeviceModel::serve_decode_sampled_finish`].
pub struct SampledStep {
    toks: Vec<u32>,
    status: Vec<i32>,
    hidden: Vec<f32>,
    event: deepinfer_gpu::Event,
    slots: Vec<(u32, deepinfer_kernels::ServeCellIn)>,
    tokens: Vec<u32>,
}

/// Worst-case aligned-contiguous MoE staging rows for `rows` tokens:
/// every pair lands in some expert segment, each segment is padded up to
/// `align` — the bound the graph path preallocates and the grouped GEMMs
/// are launched at.
fn moe_prefill_bound(rows: usize, ki: usize, e: usize, align: usize) -> usize {
    let pairs = rows * ki;
    (pairs + 127 * pairs.min(e)).next_multiple_of(align).max(align)
}

/// One pool: `[total_pages, PAGE, row_bytes]`.
struct Pool {
    buf: Buffer,
    #[allow(dead_code)]
    row_bytes: usize,
    #[allow(dead_code)]
    total_pages: usize,
}

impl Pool {
    fn new(dev: &deepinfer_gpu::Device, total_pages: usize, row_bytes: usize) -> Result<Pool> {
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
    /// Full token history (the prefix-cache key; truncated on
    /// rollback exactly like `len`).
    tokens: Vec<u32>,
    /// Rows `[0..shared)` sit on tree-leased pages: content frozen,
    /// writes trigger copy-on-write (see `serve_prefix_cow`).
    shared: usize,
    /// Rows `[0..inserted)` of the current token history are already
    /// radix-tree nodes (block-aligned; the insert starting point).
    inserted: usize,
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
    // Device sampler workspace (P0-1): per-row parameter block, sampled
    // tokens and statuses.
    samp_params: Buffer,
    samp_tok: Buffer,
    samp_status: Buffer,
    // Graph-safe prefill-MoE staging (P0-2): bound-sized persistent
    // buffers so `serve_moe_prefill` never allocates or reads back —
    // capturable end to end.
    mpf_perm: Buffer,
    mpf_aq: Option<Buffer>,
    mpf_asf: Option<Buffer>,
    mpf_a16: Option<Buffer>,
    mpf_act: Buffer,
    mpf_silu: Buffer,
    mpf_aq2: Option<Buffer>,
    mpf_asf2: Option<Buffer>,
    mpf_down: Buffer,
}

/// The paged-cache state of a [`DeviceModel`].
pub(crate) struct PagedState {
    layers: Vec<LayerPools>,
    /// The one shared free list (page ids are global across layers/kinds).
    free: Vec<u32>,
    /// Per-page lease counts: a page is in `free` iff its count is 0.
    /// Lease holders: attached sequences and radix-tree nodes.
    refs: Vec<u32>,
    /// The prefix cache (block-granular radix tree over the pools).
    prefix: GpuPrefixTree,
    /// Device bytes one page id holds (all layers, all five pools) —
    /// the ring-snapshot cost converter for the tree budget.
    page_units_bytes: usize,
    seqs: BTreeMap<u32, ServeSeq>,
    max_seqs: usize,
    max_len: usize,
    max_pages_per_seq: usize,
    total_pages: usize,
    ws: Option<ServeWs>,
    /// P0-2: serve-graph switch and the captured `(sq, b)` bucket table.
    serve_graph_on: bool,
    graphs: std::collections::HashMap<(usize, usize), deepinfer_gpu::GraphExec>,
    /// P1-6: the host-memory (L2) spill tier of the prefix cache.
    pub(crate) l2: crate::prefix_cache::L2Tier,
}

impl PagedState {
    pub(crate) fn empty() -> PagedState {
        PagedState {
            layers: Vec::new(),
            free: Vec::new(),
            refs: Vec::new(),
            prefix: GpuPrefixTree::new(0, 0),
            page_units_bytes: 1,
            seqs: BTreeMap::new(),
            max_seqs: 0,
            max_len: 0,
            max_pages_per_seq: 0,
            total_pages: 0,
            ws: None,
            serve_graph_on: false,
            graphs: std::collections::HashMap::new(),
            l2: crate::prefix_cache::L2Tier::new(0),
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
        ensure!(
            pages_per_layer >= 1,
            InvalidArgument,
            "pages_per_layer must be >= 1"
        );
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
        // One cached page id holds rows in every layer's five pools; the
        // honest per-page device cost drives the prefix-cache budget.
        let idx_dim = cfg.dsa.num_heads * cfg.dsa.head_dim;
        let row_bytes_per_layer = kv_rb + idx_dim * 4 + idx_dim * 4 + idx_dim + idx_dim * 4;
        let page_units_bytes = row_bytes_per_layer * PAGE * cfg.n_self_layers;
        self.paged = PagedState {
            free: (0..pages_per_layer as u32).rev().collect(),
            refs: vec![0; pages_per_layer],
            prefix: GpuPrefixTree::new(0, 0),
            page_units_bytes,
            layers,
            seqs: BTreeMap::new(),
            max_seqs,
            max_len,
            max_pages_per_seq: max_len.div_ceil(PAGE),
            total_pages: pages_per_layer,
            ws: None,
            serve_graph_on: self.paged.serve_graph_on,
            graphs: std::collections::HashMap::new(),
            l2: crate::prefix_cache::L2Tier::new(0),
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
        // Prefill-MoE staging bound over the largest chunk this workspace
        // can serve (score_rows tokens).
        let mpf_bound = moe_prefill_bound(score_rows, ki, e, MOE_ALIGN);
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
            moe_aq: if fp8 {
                Some(b(e * cap * hidden)?)
            } else {
                None
            },
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
            moe_a16: if fp8 {
                None
            } else {
                Some(f(e * cap * 2 * inter)?)
            },
            moe_down: f(e * cap * hidden)?,
            moe_y: f(score_rows * hidden)?,
            topk_w: f(score_rows * ki)?,
            topk_idx: b(score_rows * ki * 4)?,
            router_logits: f(score_rows * e)?,
            samp_params: f(rows * 6)?,
            samp_tok: b(rows * 4)?,
            samp_status: b(rows * 4)?,
            // Graph-safe prefill-MoE staging at the worst-case bound of
            // the largest forward the workspace serves (score_rows).
            mpf_perm: f(mpf_bound * hidden)?,
            mpf_aq: if fp8 {
                Some(b(mpf_bound * hidden)?)
            } else {
                None
            },
            mpf_asf: if fp8 {
                Some(b(mpf_bound * (hidden / 128).max(1))?)
            } else {
                None
            },
            mpf_a16: if fp8 {
                None
            } else {
                Some(f(mpf_bound * hidden)?)
            },
            mpf_act: f(mpf_bound * 2 * inter)?,
            mpf_silu: f(mpf_bound * inter)?,
            mpf_aq2: if fp8 {
                Some(b(mpf_bound * inter)?)
            } else {
                None
            },
            mpf_asf2: if fp8 {
                Some(b(mpf_bound * (inter / 128).max(1))?)
            } else {
                None
            },
            mpf_down: f(mpf_bound * hidden)?,
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
                tokens: Vec::new(),
                shared: 0,
                inserted: 0,
            },
        );
        Ok(())
    }

    /// Detach a sequence; its page leases drop (pages shared with the
    /// radix tree or other sequences survive until their last lease
    /// ends; only then do they return to the free list).
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] for an unknown sequence.
    pub fn serve_detach(&mut self, seq: u32) -> Result<()> {
        let s = self
            .paged
            .seqs
            .remove(&seq)
            .ok_or_else(|| Error::invalid(format!("sequence {seq} is not attached")))?;
        for p in s.pages {
            self.release_page(p);
        }
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
    /// the arena rollback). Tree-leased rows stay frozen: if a later append
    /// re-enters the shared region the affected pages are copied-on-write,
    /// so the radix tree's snapshots never see overwritten content.
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
        // Blocks at or beyond the new length will be re-appended with
        // different tokens: they are no longer tree members of this
        // history. (Block-aligned clamp; rows below keep their tokens.)
        // `shared` is deliberately NOT clamped: the tree nodes for the
        // old history still lease those pages, so the copy-on-write
        // guard must keep protecting them.
        let t = (s.len / PAGE) * PAGE;
        s.inserted = s.inserted.min(t);
        s.tokens.truncate(s.len);
        Ok(())
    }

    /// Grow `seq`'s page table to cover `total` rows and upload it.
    /// Page grants run through the pressure valve: when the free list
    /// runs dry, radix-tree leaves are evicted before the call fails.
    fn serve_ensure_pages(&mut self, seq: u32, total: usize) -> Result<()> {
        let need = total.div_ceil(PAGE);
        ensure!(
            need <= self.paged.max_pages_per_seq,
            CapacityExceeded,
            "sequence would need {need} pages (budget {})",
            self.paged.max_pages_per_seq
        );
        let have = {
            let s = self
                .paged
                .seqs
                .get(&seq)
                .ok_or_else(|| Error::invalid(format!("sequence {seq} is not attached")))?;
            s.pages.len()
        };
        if need > have {
            let take = need - have;
            let mut granted = Vec::with_capacity(take);
            for _ in 0..take {
                match self.serve_alloc_page() {
                    Ok(p) => granted.push(p),
                    Err(e) => {
                        // Return what we took before failing.
                        for p in granted {
                            self.release_page(p);
                        }
                        return Err(e);
                    }
                }
            }
            let tbl_words: Vec<i32> = {
                let s = self.paged.seqs.get_mut(&seq).expect("checked above");
                s.pages.extend_from_slice(&granted);
                s.pages.iter().map(|&p| p as i32).collect()
            };
            let view = self.paged.seqs[&seq].tbl.slice(0, tbl_words.len() * 4)?;
            self.stream.write_async(&self.dev, &view, &tbl_words)?;
        }
        Ok(())
    }

    /// Pop a page from the free list; when empty, evict radix-tree
    /// leaves (the prefix cache is the pressure valve — live sequences
    /// always outrank cached prefixes) until a page frees.
    ///
    /// # Errors
    /// [`Error::CapacityExceeded`] when the pools are exhausted even
    /// after evicting the whole tree.
    fn serve_alloc_page(&mut self) -> Result<u32> {
        if let Some(p) = self.paged.free.pop() {
            return Ok(p);
        }
        while self.paged.prefix.holds_pages() {
            let Some(p) = self.paged.prefix.evict_lru_leaf() else {
                break;
            };
            if self.release_page(p) {
                return Ok(p);
            }
        }
        Err(Error::CapacityExceeded(format!(
            "paged cache exhausted ({} pages); raise the pool budget",
            self.paged.total_pages
        )))
    }

    /// Drop one lease on `p`; the page returns to the free list iff
    /// that was the last one. Returns whether the page was freed.
    fn release_page(&mut self, p: u32) -> bool {
        let r = &mut self.paged.refs[p as usize];
        if *r > 0 {
            *r -= 1;
        }
        if *r == 0 {
            self.paged.free.push(p);
            true
        } else {
            false
        }
    }

    // ------------------------------------------------------------------
    // GPU prefix caching (block radix tree over the shared pools)
    // ------------------------------------------------------------------

    /// Configure the GPU prefix cache: at most `budget_tokens` prompt
    /// tokens of shared pages plus ring snapshots (0 disables; the
    /// default after [`DeviceModel::serve_configure`]). The budget is
    /// honest about device bytes: a node costs its page plus its ring
    /// snapshot, both counted in page-units, so ring-heavy configs
    /// cache proportionally fewer blocks. Call after every
    /// `serve_configure` (reconfiguration resets the budget to 0).
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] before [`DeviceModel::serve_configure`].
    pub fn serve_prefix_configure(&mut self, budget_tokens: usize) -> Result<()> {
        ensure!(
            !self.paged.layers.is_empty(),
            InvalidArgument,
            "call serve_configure first"
        );
        let ring_bytes =
            self.cfg.n_cross_layers * self.cfg.swa_window.max(1) * self.cfg.hidden_size * 4;
        let ring_units = ring_bytes / self.paged.page_units_bytes.max(1);
        self.paged
            .prefix
            .set_budget(budget_tokens.div_ceil(PAGE), ring_units);
        // P1-6: the host-memory spill tier — `DEEPINFER_PREFIX_L2_BYTES`
        // overrides; the default is 25% of the token budget's device
        // footprint (the LMCache-style CPU-memory L2).
        let default_l2 = (budget_tokens.div_ceil(PAGE)) * self.paged.page_units_bytes / 4;
        let l2_budget = std::env::var("DEEPINFER_PREFIX_L2_BYTES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(default_l2);
        self.paged.l2.set_budget(l2_budget);
        for p in self.paged.prefix.evict_to_budget() {
            self.release_page(p);
        }
        Ok(())
    }

    /// The prefix-cache counters (hits, reused tokens, inserts,
    /// evictions, live nodes, budget) — the GPU analogue of the CPU
    /// server's `PrefixIndex` gauges.
    pub fn serve_prefix_stats(&self) -> PrefixCacheStats {
        let mut st = self.paged.prefix.stats();
        let (spills, pageins) = self.paged.l2.counters();
        st.l2_spills = spills;
        st.l2_pageins = pageins;
        st.l2_blobs = self.paged.l2.len();
        st.l2_bytes = self.paged.l2.bytes();
        st.l2_budget_bytes = self.paged.l2.budget();
        st
    }

    /// Drop every cached prefix (tree leases end; pages shared with
    /// live sequences survive them). The budget and the allocation
    /// pressure valve manage the tree automatically — this is for
    /// benchmarks and cache-off measurements.
    pub fn serve_prefix_reset(&mut self) {
        for p in self.paged.prefix.clear() {
            self.release_page(p);
        }
    }

    /// Enables or disables **serve-graph capture** (P0-2): decode steps
    /// at a bucketed batch width and single-slot prefill/verify chunks at
    /// a bucketed length are captured once into a CUDA/HIP graph on
    /// first use and replayed as a single launch afterwards — SGLang's
    /// "breakable prefill graphs" shape, applied to the paged engine.
    ///
    /// Graph safety rests on the existing kernel contract: every
    /// per-step scalar (positions, lengths, window bounds, page tables)
    /// is read from device cells the kernels dereference at replay time,
    /// the MoE prefill block stages into persistent bound-sized buffers
    /// with device-resident segment offsets (no host readback), and the
    /// token/cell uploads stay outside the captured region. Decode
    /// batches are padded with inert cells (`active = 0`) up to the
    /// bucket width — every pool-writing kernel guards on the active
    /// flag, and pad slots borrow slot 0's page tables so no pointer is
    /// ever stale.
    ///
    /// Non-bucketed shapes (mixed-batch prefill, tail chunks) keep the
    /// eager path. Rollbacks stay exact: graphs read the position and
    /// page tables from device memory every replay.
    ///
    /// # Errors
    /// [`Error::Internal`] when enabling on a driver without graph
    /// capture support.
    pub fn enable_serve_graph(&mut self, on: bool) -> Result<()> {
        if on && !deepinfer_gpu::graph_supported(&self.dev) {
            return Err(Error::internal(
                "serve graphs requested but the driver does not expose CUDA/HIP graph capture",
            ));
        }
        self.paged.serve_graph_on = on;
        if !on {
            self.paged.graphs.clear();
        }
        Ok(())
    }

    /// Whether serve-graph capture is enabled (see
    /// [`DeviceModel::enable_serve_graph`]).
    pub fn serve_graph_enabled(&self) -> bool {
        self.paged.serve_graph_on
    }

    /// Number of captured serve graphs (the `(sq, b)` bucket table size —
    /// a capacity/telemetry gauge).
    pub fn serve_graphs_captured(&self) -> usize {
        self.paged.graphs.len()
    }

    /// Match and adopt a cached prefix for a **fresh** sequence: shares
    /// the radix path's pages (bumping their leases), uploads the page
    /// table, restores the `k_sf` scale prefix and copies the ring
    /// snapshot — a handful of pointer writes and device-to-device
    /// copies replacing a compute-bound prefill. Returns the restored
    /// row count (a page-aligned 0 when the cache misses).
    ///
    /// The restored state is bit-identical to a cold prefill of the
    /// same tokens at the same chunk boundary: the pages hold the
    /// donor's exact rows and the rings its exact window.
    fn serve_prefix_restore(&mut self, seq: u32, tokens: &[u32]) -> Result<usize> {
        if !self.paged.prefix.enabled() || tokens.is_empty() {
            return Ok(0);
        }
        // At least one token must remain for the suffix prefill.
        let limit = (tokens.len() - 1) / PAGE;
        if limit == 0 {
            return Ok(0);
        }
        let mut m = self.paged.prefix.match_blocks(&tokens[..limit * PAGE]);
        if m < limit && self.paged.l2.enabled() && !self.paged.l2.is_empty() {
            // P1-6: a VRAM miss (or a short match) consults the host
            // tier — the deepest surviving spilled chain pages back in
            // as fresh tree nodes, then the match runs again.
            let restored = self.serve_prefix_l2_page_in(tokens, m, limit);
            if restored > m {
                m = self.paged.prefix.match_blocks(&tokens[..limit * PAGE]);
            }
        }
        if m == 0 {
            return Ok(0);
        }
        let rows = m * PAGE;
        let (pages, ksf) = self.paged.prefix.restore_payload(tokens, m);
        if pages.len() != m {
            return Ok(0); // evicted between match and collect: cold path
        }
        // Adopt the pages (disjoint-field borrows: prefix vs seqs vs refs).
        for &p in &pages {
            self.paged.refs[p as usize] += 1;
        }
        let tbl_words: Vec<i32>;
        {
            let s = self
                .paged
                .seqs
                .get_mut(&seq)
                .ok_or_else(|| Error::invalid(format!("sequence {seq} is not attached")))?;
            s.pages = pages;
            s.tokens = tokens[..rows].to_vec();
            s.shared = rows;
            s.inserted = rows;
            s.len = rows;
            tbl_words = s.pages.iter().map(|&p| p as i32).collect();
        }
        let view = self.paged.seqs[&seq].tbl.slice(0, tbl_words.len() * 4)?;
        self.stream.write_async(&self.dev, &view, &tbl_words)?;
        if !ksf.is_empty() {
            let sf = self.paged.seqs[&seq].k_sf.slice(0, ksf.len())?;
            self.stream.write_async(&self.dev, &sf, &ksf)?;
        }
        // Ring snapshot -> the sequence's rings (stream-ordered D2D).
        if let Some(src) = self.paged.prefix.endpoint_rings(tokens, m) {
            let s = &self.paged.seqs[&seq];
            for (l, ring) in s.rings.iter().enumerate() {
                if let Some(from) = src.get(l) {
                    self.dev.memcpy_dtod_async(ring, from, &self.stream)?;
                }
            }
        }
        Ok(rows)
    }

    /// Insert the sequence's newly completed blocks into the radix
    /// tree: pages leased (frozen), `k_sf` segments read back, ring
    /// snapshots copied. Called after every committed prefill chunk and
    /// decode step; speculative-verify chunks never insert (their rows
    /// are reversible). Blocks another sequence already cached are
    /// skipped (LRU-touched, no new lease). Ring allocation failures
    /// degrade the node to a walk-through (not a match endpoint)
    /// rather than failing the step.
    fn serve_prefix_insert(&mut self, seq: u32) -> Result<()> {
        if !self.paged.prefix.enabled() {
            return Ok(());
        }
        // Cheap no-op check first: the decode hot path calls this every
        // step, and blocks complete only once per PAGE tokens.
        let (len, inserted) = {
            let Some(s) = self.paged.seqs.get(&seq) else {
                return Ok(());
            };
            (s.len, s.inserted)
        };
        let nb = len / PAGE;
        if nb <= inserted / PAGE {
            if self.paged.prefix.holds_pages() {
                self.serve_prefix_evict_to_budget()?;
            }
            return Ok(());
        }
        let (pages, n_rings, ksf_len) = {
            let Some(s) = self.paged.seqs.get(&seq) else {
                return Ok(());
            };
            (s.pages.clone(), s.rings.len(), s.k_sf.len())
        };
        let tokens = {
            let Some(s) = self.paged.seqs.get(&seq) else {
                return Ok(());
            };
            s.tokens.clone()
        };
        if tokens.len() < len {
            return Ok(()); // history desync: never cache wrong tokens
        }
        // Leading blocks already present (ours or another sequence's).
        let start = self
            .paged
            .prefix
            .shared_blocks(&tokens)
            .max(inserted / PAGE);
        let qb = self.cfg.dsa.quant_block;
        let ring_bytes = self.cfg.swa_window.max(1) * self.cfg.hidden_size * 4;
        let mut done = start * PAGE;
        for b in start..nb {
            if b >= pages.len() {
                break;
            }
            // The k_sf bytes completing inside this block.
            let (s0, s1) = ksf_seg_range(b, qb);
            let mut seg = Vec::new();
            if s1 > s0 && s1 <= ksf_len {
                self.stream.synchronize()?;
                let view = self.paged.seqs[&seq].k_sf.slice(s0, s1 - s0)?;
                seg = vec![0u8; s1 - s0];
                self.dev.read(&view, &mut seg)?;
            }
            // The ring snapshot (device-owned; freed when the node is).
            let rings = if n_rings == 0 {
                Some(Vec::new()) // zero cross layers: a valid endpoint
            } else {
                let bufs: Result<Vec<Buffer>> =
                    (0..n_rings).map(|_| self.dev.alloc(ring_bytes)).collect();
                match bufs {
                    Ok(dst) => {
                        let s = &self.paged.seqs[&seq];
                        for (l, d) in dst.iter().enumerate() {
                            if let Some(src) = s.rings.get(l) {
                                self.dev.memcpy_dtod_async(d, src, &self.stream)?;
                            }
                        }
                        Some(dst)
                    }
                    // VRAM pressure: a walk-through node still shares
                    // the page for deeper matches.
                    Err(_) => None,
                }
            };
            let p = pages[b];
            self.paged.refs[p as usize] += 1; // the tree lease
            if self
                .paged
                .prefix
                .insert_block(&tokens, b, p, seg, rings)
                .is_err()
            {
                // Never fail the serving step for a cache insert.
                self.release_page(p);
                break;
            }
            done = (b + 1) * PAGE;
        }
        if let Some(s) = self.paged.seqs.get_mut(&seq) {
            s.inserted = done;
            // Older-history nodes may still lease pages beyond `done`
            // (rollback kept them frozen); the freeze range only grows.
            s.shared = s.shared.max(done);
        }
        self.serve_prefix_evict_to_budget()?;
        Ok(())
    }

    /// Evict LRU leaves down to the tree budget, **spilling** each
    /// victim to the host-memory L2 tier first (P1-6): the page content
    /// of all five pools for every self layer, the `k_sf` segment and
    /// the ring snapshots are read back once and stored under the
    /// block's chained hash — a later VRAM miss pages them back in
    /// instead of recomputing the prefill. A disabled tier drops the
    /// pages exactly as before.
    fn serve_prefix_evict_to_budget(&mut self) -> Result<()> {
        while self.paged.prefix.over_budget() {
            let Some(victim) = self.paged.prefix.lru_leaf_idx() else {
                break;
            };
            if self.paged.l2.enabled() {
                self.spill_l2(victim)?;
            }
            let Some(p) = self.paged.prefix.evict_lru_leaf() else {
                break;
            };
            self.release_page(p);
        }
        Ok(())
    }

    /// Read one victim node's page content back and store it in the L2
    /// tier under its chained hash.
    fn spill_l2(&mut self, victim: usize) -> Result<()> {
        let hash = self.paged.prefix.node_hash(victim);
        let page = self.paged.prefix.node_page(victim) as usize;
        let block = self.paged.prefix.node_block(victim).to_vec();
        let k_sf = self.paged.prefix.node_ksf(victim).to_vec();
        // Pool content, layer-major, five kinds per layer.
        self.stream.synchronize()?;
        let mut pools: Vec<Vec<u8>> = Vec::with_capacity(self.paged.layers.len() * 5);
        for lp in &self.paged.layers {
            for pool in [&lp.kv, &lp.k_raw, &lp.k_norm, &lp.k_u8, &lp.feat] {
                let bytes = PAGE * pool.row_bytes;
                let view = pool.buf.slice(page * bytes, bytes)?;
                let mut dst = vec![0u8; bytes];
                self.dev.read(&view, &mut dst)?;
                pools.push(dst);
            }
        }
        // Ring snapshots (endpoints only).
        let rings = match self.paged.prefix.node_rings(victim) {
            Some(src) => {
                let mut rs = Vec::with_capacity(src.len());
                for buf in src {
                    let mut dst = vec![0u8; buf.len()];
                    self.dev.read(buf, &mut dst)?;
                    rs.push(dst);
                }
                Some(rs)
            }
            None => None,
        };
        self.paged.l2.store(
            hash,
            crate::prefix_cache::L2Blob {
                block,
                k_sf,
                pools,
                rings,
                last_used: 0,
            },
        );
        Ok(())
    }

    /// Page the deepest surviving L2 chain of `tokens` (starting at
    /// block `from`, up to `limit` blocks) back into fresh VRAM pages,
    /// re-creating the radix nodes so the normal restore path matches
    /// them. Returns the number of blocks restored into the tree.
    fn serve_prefix_l2_page_in(&mut self, tokens: &[u32], from: usize, limit: usize) -> usize {
        if !self.paged.l2.enabled() || from >= limit {
            return from;
        }
        let mut h = crate::prefix_cache::FNV_OFFSET;
        for b in 0..from {
            h = crate::prefix_cache::chain_block(h, &tokens[b * PAGE..(b + 1) * PAGE]);
        }
        let mut restored = from;
        let mut blobs: Vec<crate::prefix_cache::L2Blob> = Vec::new();
        for _b in from..limit {
            let block = &tokens[_b * PAGE..(_b + 1) * PAGE];
            h = crate::prefix_cache::chain_block(h, block);
            let Some(blob) = self.paged.l2.get(h).cloned() else {
                break;
            };
            if blob.block != block {
                break; // hash collision: not the same tokens
            }
            blobs.push(blob);
        }
        if blobs.is_empty() {
            return from;
        }
        let ring_bytes = self.cfg.swa_window.max(1) * self.cfg.hidden_size * 4;
        for (i, blob) in blobs.into_iter().enumerate() {
            let b = from + i;
            // A fresh page per block; the tree holds its lease.
            let Ok(page) = self.serve_alloc_page() else {
                break;
            };
            // Restore the pool content.
            let mut ok = true;
            for (li, lp) in self.paged.layers.iter().enumerate() {
                for (ki, pool) in [&lp.kv, &lp.k_raw, &lp.k_norm, &lp.k_u8, &lp.feat]
                    .into_iter()
                    .enumerate()
                {
                    let idx = li * 5 + ki;
                    let Some(bytes) = blob.pools.get(idx) else {
                        ok = false;
                        break;
                    };
                    let want = PAGE * pool.row_bytes;
                    if bytes.len() != want {
                        ok = false;
                        break;
                    }
                    let view = match pool.buf.slice(page as usize * want, want) {
                        Ok(v) => v,
                        Err(_) => {
                            ok = false;
                            break;
                        }
                    };
                    if self
                        .stream
                        .write_async(&self.dev, &view, bytes)
                        .is_err()
                    {
                        ok = false;
                        break;
                    }
                }
                if !ok {
                    break;
                }
            }
            if !ok {
                self.release_page(page);
                break;
            }
            // The ring snapshots (device buffers, H2D).
            let rings = match &blob.rings {
                None => None,
                Some(rs) => {
                    let bufs: Result<Vec<Buffer>> =
                        (0..rs.len()).map(|_| self.dev.alloc(ring_bytes)).collect();
                    match bufs {
                        Ok(dst) => {
                            for (d, bytes) in dst.iter().zip(rs.iter()) {
                                if self.stream.write_async(&self.dev, d, bytes).is_err() {
                                    ok = false;
                                    break;
                                }
                            }
                            if ok {
                                Some(dst)
                            } else {
                                None
                            }
                        }
                        Err(_) => None,
                    }
                }
            };
            if !ok {
                self.release_page(page);
                break;
            }
            if self
                .paged
                .prefix
                .insert_block(tokens, b, page, blob.k_sf.clone(), rings)
                .is_err()
            {
                self.release_page(page);
                break;
            }
            restored = b + 1;
        }
        restored
    }

    /// Copy-on-write guard: appends starting at `pos0` that would
    /// rewrite tree-frozen rows get fresh private pages (full-page
    /// device copies; the tail rows are stale but always rewritten
    /// before any reader can see them). The old pages stay leased by
    /// the radix tree with their frozen content intact. Blocks whose
    /// frozen content the re-append reproduces exactly (the scheduler
    /// rewind case: same prompt re-prefilled) skip the copy.
    fn serve_prefix_cow(&mut self, seq: u32, pos0: usize) -> Result<()> {
        let (shared_blocks, tokens) = {
            let Some(s) = self.paged.seqs.get(&seq) else {
                return Ok(());
            };
            (s.shared / PAGE, s.tokens.clone())
        };
        let fb = pos0 / PAGE;
        if fb >= shared_blocks {
            return Ok(());
        }
        // Leading blocks whose cached content the current history
        // reproduces exactly: rewrites there are no-ops on the pages.
        let matched = self
            .paged
            .prefix
            .matching_blocks(&tokens)
            .min(shared_blocks);
        let from = fb.max(matched);
        if from >= shared_blocks {
            return Ok(()); // identical rewrite (rewind): nothing to do
        }
        // Per-kind row bytes (the pool geometry from serve_configure).
        let idx_dim = self.cfg.dsa.num_heads * self.cfg.dsa.head_dim;
        let kv_rb = kv_row_bytes(self.fmt, self.cfg.head_dim);
        let rbs = [kv_rb, idx_dim * 4, idx_dim * 4, idx_dim, idx_dim * 4];
        for b in from..shared_blocks {
            let old = self.paged.seqs[&seq].pages[b];
            let fresh = self.serve_alloc_page()?;
            {
                // Enqueue the five per-layer page copies (immutable
                // borrows; mutations follow once the stream has them).
                for l in &self.paged.layers {
                    let bufs = [
                        (&l.kv.buf, rbs[0]),
                        (&l.k_raw.buf, rbs[1]),
                        (&l.k_norm.buf, rbs[2]),
                        (&l.k_u8.buf, rbs[3]),
                        (&l.feat.buf, rbs[4]),
                    ];
                    for (buf, rb) in bufs {
                        let bytes = rb * PAGE;
                        let dst = buf.slice(fresh as usize * bytes, bytes)?;
                        let src = buf.slice(old as usize * bytes, bytes)?;
                        self.dev.memcpy_dtod_async(&dst, &src, &self.stream)?;
                    }
                }
            }
            let s = self.paged.seqs.get_mut(&seq).expect("attached");
            s.pages[b] = fresh;
            self.release_page(old);
        }
        // Re-upload the page table and drop the frozen range.
        let tbl_words: Vec<i32> = {
            let s = self.paged.seqs.get_mut(&seq).expect("attached");
            s.shared = from * PAGE;
            s.inserted = s.inserted.min(from * PAGE);
            s.pages.iter().map(|&p| p as i32).collect()
        };
        let view = self.paged.seqs[&seq].tbl.slice(0, tbl_words.len() * 4)?;
        self.stream.write_async(&self.dev, &view, &tbl_words)?;
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
        ensure!(
            b <= ws_rows_bound(self),
            InvalidArgument,
            "batch exceeds max_seqs"
        );
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
        self.stream
            .write_async(&self.dev, &ws.cells_in, &in_cells)?;
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
        // Fresh sequence: adopt any cached prefix (page sharing) before
        // the first chunk; the suffix prefill starts at `skip`.
        let skip = if take_seq_len(&self.paged, seq)? == 0 {
            self.serve_prefix_restore(seq, tokens)?
        } else {
            0
        };
        let mut done = skip;
        while done < tokens.len() {
            let take = SERVE_CHUNK.min(tokens.len() - done);
            let chunk = &tokens[done..done + take];
            let rows = self.serve_chunk_rows(seq, chunk)?;
            self.serve_prefix_insert(seq)?;
            if done + take == tokens.len() {
                logits = rows.into_iter().next_back().expect("non-empty chunk");
            }
            done += take;
        }
        Ok(logits)
    }

    /// [`DeviceModel::serve_prefill`] with the full speculative-decoding
    /// anchor: the final position's logits **and** residual hidden state
    /// (see [`crate::model::ForwardOut`]).
    ///
    /// # Errors
    /// Same as [`DeviceModel::serve_prefill`].
    pub fn serve_prefill_out(
        &mut self,
        seq: u32,
        tokens: &[u32],
    ) -> Result<crate::model::ForwardOut> {
        let mut last = None;
        // Fresh sequence: adopt any cached prefix before chunking.
        let skip = if take_seq_len(&self.paged, seq)? == 0 {
            self.serve_prefix_restore(seq, tokens)?
        } else {
            0
        };
        let mut done = skip;
        while done < tokens.len() {
            let take = SERVE_CHUNK.min(tokens.len() - done);
            let chunk = &tokens[done..done + take];
            let (logits, hidden) = self.serve_chunk_rows_out(seq, chunk)?;
            self.serve_prefix_insert(seq)?;
            if done + take == tokens.len() {
                last = Some(crate::model::ForwardOut { logits, hidden });
            }
            done += take;
        }
        last.ok_or_else(|| Error::invalid("serve_prefill_out requires tokens"))
    }

    /// One prefill/verify chunk (B = 1 slot, sq = chunk): runs the shared
    /// serve forward and reads **every** position's logit row plus the
    /// last position's residual hidden state.
    fn serve_chunk_rows_out(&mut self, seq: u32, tokens: &[u32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let rows = self.serve_chunk_rows(seq, tokens)?;
        let hidden = self.read_last_hidden(tokens.len())?;
        let logits = rows.into_iter().next_back().expect("non-empty chunk");
        Ok((logits, hidden))
    }

    /// Reads the residual hidden state of the last row from the serve
    /// workspace (`ws.h` keeps the pre-final-norm residual; the final
    /// norm writes into `ws.x`).
    fn read_last_hidden(&self, sq: usize) -> Result<Vec<f32>> {
        let hidden = self.cfg.hidden_size;
        let ws = self.paged.ws.as_ref().expect("configured");
        self.stream.synchronize()?;
        let mut out = vec![0.0f32; hidden];
        let row = ws.h.slice((sq - 1) * hidden * 4, hidden * 4)?;
        self.dev.read(&row, &mut out)?;
        Ok(out)
    }

    /// Speculative-verification forward for the paged serve engine: the
    /// whole draft chain `tokens` runs as **one chunked multi-row
    /// forward** (B = 1 slot, sq = chain length) and every per-position
    /// logit row is read back. Row `j` is the distribution over the token
    /// after `tokens[j]` — the verification contract
    /// `deepinfer_mtp::verify_greedy` / `verify_stochastic` expect when
    /// prefixed with the anchor row. The sequence advances by
    /// `tokens.len()`; rejected tails roll back with
    /// [`DeviceModel::serve_rollback`].
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] for unknown sequences / bad tokens /
    /// an empty chain; [`Error::CapacityExceeded`] over the page or
    /// position bounds.
    pub fn serve_verify(&mut self, seq: u32, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        ensure!(
            !tokens.is_empty(),
            InvalidArgument,
            "verify needs draft tokens"
        );
        for &t in tokens {
            ensure!(
                (t as usize) < self.cfg.vocab_size,
                OutOfRange,
                "token {t} outside the vocabulary"
            );
        }
        self.serve_chunk_rows(seq, tokens)
    }

    /// One prefill/verify chunk (B = 1 slot, sq = chunk): runs the shared
    /// serve forward and reads **every** position's logit row.
    fn serve_chunk_rows(&mut self, seq: u32, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        let cfg = self.cfg.clone();
        let s = take_seq_len(&self.paged, seq)?;
        let pos0 = s;
        let sq = tokens.len();
        let total = pos0 + sq;
        // Copy-on-write: a rewound sequence re-entering its tree-frozen
        // region must not overwrite radix-shared pages.
        self.serve_prefix_cow(seq, pos0)?;
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
        let b = 1usize;
        self.stage_serve_tokens(tokens)?;
        self.serve_forward_step(sq, b)?;
        // Read every row's logits.
        self.stream.synchronize()?;
        let ws = self.paged.ws.as_ref().expect("configured");
        let mut out = Vec::with_capacity(sq);
        for j in 0..sq {
            let mut logits = vec![0.0f32; cfg.vocab_size];
            let row = ws
                .logits
                .slice(j * cfg.vocab_size * 4, cfg.vocab_size * 4)?;
            self.dev.read(&row, &mut logits)?;
            out.push(logits);
        }
        self.paged.seqs.get_mut(&seq).expect("attached").len = total;
        self.paged
            .seqs
            .get_mut(&seq)
            .expect("attached")
            .tokens
            .extend_from_slice(tokens);
        Ok(out)
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
        Ok(self
            .serve_decode_out(batch)?
            .into_iter()
            .map(|o| o.logits)
            .collect())
    }

    /// [`DeviceModel::serve_decode`] with the full speculative-decoding
    /// anchor: per-sequence `(logits, residual hidden)` in batch order
    /// (see [`crate::model::ForwardOut`]) — the hidden row feeds the
    /// host-side MTP draft step between decodes.
    ///
    /// # Errors
    /// Same as [`DeviceModel::serve_decode`].
    pub fn serve_decode_out(
        &mut self,
        batch: &[(u32, u32)],
    ) -> Result<Vec<crate::model::ForwardOut>> {
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
            // Copy-on-write guard (a rewound sequence re-entering its
            // tree-frozen region) before any page is granted.
            self.serve_prefix_cow(*seq, cell.len as usize)?;
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
        // Graph padding: inert cells up to the bucketed width.
        let b_eff = self.serve_graph_pad_width(b);
        let mut padded = slots.clone();
        let anchor_seq = padded[0].0;
        while padded.len() < b_eff {
            padded.push((
                anchor_seq,
                ServeCellIn {
                    active: 0,
                    pos: 0,
                    len: 0,
                    sq: 1,
                    sqnb: 0,
                },
            ));
        }
        self.serve_begin_step(&padded)?;
        let mut tokens: Vec<u32> = slots
            .iter()
            .map(|(seq, _)| batch.iter().find(|&&(s, _)| s == *seq).expect("present").1)
            .collect();
        tokens.resize(b_eff, 0);
        self.stage_serve_tokens(&tokens)?;
        self.serve_forward_step(1, b_eff)?;
        // Read per-sequence logits and the anchor hidden rows.
        self.stream.synchronize()?;
        let ws = self.paged.ws.as_ref().expect("configured");
        let mut out = Vec::with_capacity(b);
        for i in 0..b {
            let mut logits = vec![0.0f32; cfg.vocab_size];
            let row = ws
                .logits
                .slice(i * cfg.vocab_size * 4, cfg.vocab_size * 4)?;
            self.dev.read(&row, &mut logits)?;
            let mut hidden = vec![0.0f32; cfg.hidden_size];
            let hrow = ws.h.slice(i * cfg.hidden_size * 4, cfg.hidden_size * 4)?;
            self.dev.read(&hrow, &mut hidden)?;
            out.push(crate::model::ForwardOut { logits, hidden });
        }
        for ((seq, cell), tok) in slots.iter().zip(tokens.iter()) {
            let s = self.paged.seqs.get_mut(seq).expect("attached");
            s.len = cell.len as usize + 1;
            s.tokens.push(*tok);
        }
        // Blocks completed by this step enter the radix tree.
        for (seq, _) in &slots {
            self.serve_prefix_insert(*seq)?;
        }
        Ok(out)
    }

    /// One batched decode step with **on-device sampling** (P0-1): the
    /// forward runs exactly like [`DeviceModel::serve_decode_out`], then
    /// the device sampler consumes `ws.logits` in place and only the
    /// `[B]` token ids, statuses and the per-sequence anchor hidden rows
    /// come back — at DeepSeek vocabulary the per-step D2H traffic drops
    /// from `[B, vocab]` f32 (~516 KB/seq) to `[B, hidden + 2]` (~28
    /// KB/seq), removing the dominant host stall of the plain decode
    /// path.
    ///
    /// The per-row `params` carry `{temperature, top_p, min_p, u, cap,
    /// top_k}`; the uniform `u` must be pre-drawn by the caller from the
    /// request's Philox stream (greedy rows should pass any value — no
    /// host RNG is consumed for them, matching the CPU sampler's
    /// contract). Grammar-constrained requests cannot use this path
    /// (their masks live on the host); the caller falls back to
    /// [`DeviceModel::serve_decode_out`].
    ///
    /// # Errors
    /// Same as [`DeviceModel::serve_decode_out`], plus
    /// [`Error::Numerical`] when a row's distribution is degenerate (the
    /// CPU sampler's "no finite logit" / "no positive mass" cases).
    pub fn serve_decode_sampled(
        &mut self,
        batch: &[(u32, u32)],
        params: &[deepinfer_kernels::SampleParams],
    ) -> Result<Vec<SampledOut>> {
        let step = self.serve_decode_sampled_stage(batch, params)?;
        self.serve_decode_sampled_finish(step)
    }

    /// [`DeviceModel::serve_decode_sampled`] split in two (P0-5): stage
    /// launches the forward, the device sampler and the event-guarded
    /// slim readback, then returns **without waiting** — the caller may
    /// queue more GPU work (this step's prefill chunks) before
    /// [`DeviceModel::serve_decode_sampled_finish`] consumes the
    /// results. Dropping the staged step is safe (the granted pages are
    /// simply re-used by the next forward), but the sequence lengths
    /// only advance at finish.
    ///
    /// # Errors
    /// Same as [`DeviceModel::serve_decode_sampled`], raised before any
    /// GPU work for validation failures.
    pub fn serve_decode_sampled_stage(
        &mut self,
        batch: &[(u32, u32)],
        params: &[deepinfer_kernels::SampleParams],
    ) -> Result<SampledStep> {
        ensure!(!batch.is_empty(), InvalidArgument, "decode batch is empty");
        ensure!(
            params.len() == batch.len(),
            InvalidArgument,
            "serve_decode_sampled: {} param rows for a {}-row batch",
            params.len(),
            batch.len()
        );
        let cfg = self.cfg.clone();
        for (i, &(seq, t)) in batch.iter().enumerate() {
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
            let p = &params[i];
            // CPU `SamplingParams::validate` invariants (the host sampler
            // rejects these before touching the stream).
            ensure!(
                p.top_p.is_finite() && p.top_p > 0.0 && p.top_p <= 1.0,
                InvalidArgument,
                "serve_decode_sampled: top_p must be in (0, 1] (got {})",
                p.top_p
            );
            ensure!(
                p.min_p.is_finite() && p.min_p >= 0.0 && p.min_p < 1.0,
                InvalidArgument,
                "serve_decode_sampled: min_p must be in [0, 1) (got {})",
                p.min_p
            );
            ensure!(
                p.cap <= cfg.vocab_size,
                InvalidArgument,
                "serve_decode_sampled: vocab cap {} exceeds the vocabulary ({})",
                p.cap,
                cfg.vocab_size
            );
            if !p.is_greedy() {
                ensure!(
                    p.u.is_finite() && p.u >= 0.0 && p.u < 1.0,
                    InvalidArgument,
                    "serve_decode_sampled: the uniform must lie in [0, 1) (got {})",
                    p.u
                );
            }
        }
        // Same slot preparation as `serve_decode_out` (deduplicated rows,
        // copy-on-write guard, page bounds).
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
        // Params follow the deduplicated slot order (one row per sequence).
        let mut slot_params: Vec<deepinfer_kernels::SampleParams> =
            Vec::with_capacity(b);
        for (seq, _) in &slots {
            let idx = batch.iter().position(|&(s, _)| s == *seq).expect("present");
            slot_params.push(params[idx]);
        }
        for (seq, cell) in &slots {
            let total = cell.len as usize + 1;
            self.serve_prefix_cow(*seq, cell.len as usize)?;
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
        // Graph padding: inert cells up to the bucketed width.
        let b_eff = self.serve_graph_pad_width(b);
        let mut padded = slots.clone();
        let anchor_seq = padded[0].0;
        while padded.len() < b_eff {
            padded.push((
                anchor_seq,
                ServeCellIn {
                    active: 0,
                    pos: 0,
                    len: 0,
                    sq: 1,
                    sqnb: 0,
                },
            ));
        }
        self.serve_begin_step(&padded)?;
        let mut tokens: Vec<u32> = Vec::with_capacity(b_eff);
        for (seq, _) in slots.iter() {
            tokens.push(batch.iter().find(|&&(s, _)| s == *seq).expect("present").1);
        }
        tokens.resize(b_eff, 0);
        self.stage_serve_tokens(&tokens)?;
        self.serve_forward_step(1, b_eff)?;
        // Device sampler over the logits + the slim asynchronous readback
        // (P0-5): the [B] tokens, statuses and anchor hidden rows travel
        // back through event-guarded D2H copies, so the engine thread can
        // queue the step's prefill work before consuming them.
        {
            let ws = self.paged.ws.as_ref().expect("configured");
            let mut words = Vec::with_capacity(b * 6);
            for p in &slot_params {
                words.extend_from_slice(&p.words());
            }
            self.stream.write_async(&self.dev, &ws.samp_params, &words)?;
            deepinfer_kernels::sampler::sample_topp(
                &self.k,
                &ws.logits,
                &ws.samp_params,
                &ws.samp_tok,
                &ws.samp_status,
                b,
                cfg.vocab_size,
                &self.stream,
            )?;
            let mut toks = vec![0u32; b];
            let mut status = vec![0i32; b];
            let mut hidden = vec![0.0f32; b * cfg.hidden_size];
            self.stream
                .read_async(&self.dev, &ws.samp_tok.slice(0, b * 4)?, &mut toks)?;
            self.stream
                .read_async(&self.dev, &ws.samp_status.slice(0, b * 4)?, &mut status)?;
            for (i, chunk) in hidden.chunks_mut(cfg.hidden_size).enumerate() {
                let hrow = ws.h.slice(i * cfg.hidden_size * 4, cfg.hidden_size * 4)?;
                self.stream.read_async(&self.dev, &hrow, chunk)?;
            }
            let event = self.dev.event()?;
            event.record(&self.stream)?;
            Ok(SampledStep {
                toks,
                status,
                hidden,
                event,
                slots,
                tokens,
            })
        }
    }

    /// The consuming half of [`DeviceModel::serve_decode_sampled_stage`]:
    /// wait on the readback event, validate the sampler statuses,
    /// assemble the results and apply the sequence bookkeeping.
    ///
    /// # Errors
    /// [`deepinfer_core::Error::Numerical`] on a degenerate row (the
    /// CPU sampler's error cases).
    pub fn serve_decode_sampled_finish(&mut self, step: SampledStep) -> Result<Vec<SampledOut>> {
        let SampledStep {
            toks,
            status,
            hidden,
            event,
            slots,
            tokens,
        } = step;
        let cfg = self.cfg.clone();
        let b = slots.len();
        event.synchronize()?;
        for &st in status.iter() {
            match st {
                0 => {}
                1 => {
                    return Err(Error::Numerical(
                        "serve_decode_sampled: no finite logit survives masking".into(),
                    ))
                }
                2 => {
                    return Err(Error::Numerical(
                        "serve_decode_sampled: filtered distribution has no positive mass"
                            .into(),
                    ))
                }
                3 => {
                    return Err(Error::Numerical(
                        "serve_decode_sampled: invalid uniform".into(),
                    ))
                }
                other => {
                    return Err(Error::internal(format!(
                        "serve_decode_sampled: unknown sampler status {other}"
                    )))
                }
            }
        }
        let mut out = Vec::with_capacity(b);
        for (i, chunk) in hidden.chunks(cfg.hidden_size).enumerate().take(b) {
            out.push(SampledOut {
                token: toks[i],
                hidden: chunk.to_vec(),
            });
        }
        for ((seq, cell), tok) in slots.iter().zip(tokens.iter()) {
            let s = self.paged.seqs.get_mut(seq).expect("attached");
            s.len = cell.len as usize + 1;
            s.tokens.push(*tok);
        }
        for (seq, _) in &slots {
            self.serve_prefix_insert(*seq)?;
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
    /// Stage this step's input tokens into the workspace (outside any
    /// graph capture — an H2D memcpy must not be baked into a replay).
    fn stage_serve_tokens(&self, tokens: &[u32]) -> Result<()> {
        let ws = self
            .paged
            .ws
            .as_ref()
            .ok_or_else(|| Error::internal("serve ws"))?;
        self.stream.write_async(&self.dev, &ws.tokens, tokens)
    }

    /// Whether `(sq, b)` has a serve-graph bucket: decode steps at a
    /// bucketed batch width (powers of two above 8, every width up to 8,
    /// plus `max_seqs`), and single-slot prefill/verify chunks at a
    /// bucketed `sq` (the `SERVE_CHUNK` ladder plus the DSpark chain
    /// lengths). Everything else runs eager.
    fn serve_graph_bucket(sq: usize, b: usize, max_seqs: usize) -> bool {
        if sq == 1 {
            // Decode family: b is its own bucket when small, a power of
            // two when large, or exactly max_seqs.
            if b <= 8 || b == max_seqs {
                return true;
            }
            return b > 8 && b.is_power_of_two() && b < max_seqs;
        }
        if b != 1 {
            return false;
        }
        // Prefill chunks (the SERVE_CHUNK ladder) and DSpark verify
        // chains (2..=8 tokens).
        const PREFILL_BUCKETS: [usize; 4] = [64, 128, 256, 512];
        sq == 2
            || sq == 3
            || sq == 4
            || sq == 5
            || sq == 6
            || sq == 7
            || sq == 8
            || PREFILL_BUCKETS.contains(&sq)
    }

    /// The decode batch width to run at when serve graphs are on: the
    /// smallest bucketed width `>= b` (padding cells are inert — every
    /// pool-writing kernel guards on `active == 0`, and the pad slots
    /// borrow slot 0's page tables so no pointer is ever stale).
    fn serve_graph_pad_width(&self, b: usize) -> usize {
        if !self.paged.serve_graph_on || b > self.paged.max_seqs {
            return b;
        }
        if b <= 8 || b == self.paged.max_seqs {
            return b;
        }
        let mut w = b.next_power_of_two();
        if w > self.paged.max_seqs {
            w = self.paged.max_seqs;
        }
        w
    }

    /// Run one serve forward, replaying the captured graph for `(sq, b)`
    /// when one exists and capturing it on first use (the capture call
    /// itself records; the immediate launch executes it). Cell, page-table
    /// and token staging must already be on the device — only kernel
    /// launches live in the graph.
    fn serve_forward_step(&mut self, sq: usize, b: usize) -> Result<()> {
        if self.paged.serve_graph_on
            && Self::serve_graph_bucket(sq, b, self.paged.max_seqs)
            && deepinfer_gpu::graph_supported(&self.dev)
        {
            let key = (sq, b);
            if let Some(exec) = self.paged.graphs.get(&key) {
                exec.launch(&self.stream)?;
                return Ok(());
            }
            let guard = self.dev.begin_capture(&self.stream)?;
            self.run_serve_forward(sq, b)?;
            let exec = guard.end()?;
            exec.launch(&self.stream)?;
            self.paged.graphs.insert(key, exec);
            return Ok(());
        }
        self.run_serve_forward(sq, b)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_serve_forward(&self, sq: usize, b: usize) -> Result<()> {
        let cfg = &self.cfg;
        let s = &self.stream;
        let ws = self
            .paged
            .ws
            .as_ref()
            .ok_or_else(|| Error::internal("serve ws"))?;
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

        // Embedding gather + positions (tokens already staged by the
        // caller — the staging memcpy stays outside graph captures).
        self.k
            .gather_indexed(&self.root.embed, &ws.h, &ws.tokens, rows, hidden * 4, s)?;
        self.k
            .serve_pos_fill(&ws.cells, &ws.pos_kv, &ws.pos_q, b, sq, cfg.n_heads, s)?;

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
            self.k.serve_kv_encode(
                &ws.kv_rows,
                &pools.kv.buf,
                &ws.tbl_ptrs,
                &ws.cells,
                hd,
                fmt,
                sq,
                b,
                s,
            )?;
            // DSA indexer.
            self.gemm(&w.wq, &ws.h, &ws.q_idx, rows)?;
            self.gemm(&w.wk, &ws.h, &ws.k_new, rows)?;
            self.k.serve_append_rows(
                &ws.k_new,
                &pools.k_raw.buf,
                &ws.tbl_ptrs,
                &ws.cells,
                idx_dim,
                sq,
                b,
                s,
            )?;
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
            self.k.serve_gate_blend(
                &ws.gt, &ws.gc, &ws.conv, &ws.scores, &ws.cells, max_cols, sq, b, s,
            )?;
            self.k
                .serve_causal_clip(&ws.scores, &ws.cells, max_cols, window, sq, b, s)?;
            self.k
                .fill_zero(&ws.cands, rows * self.geo.cands_cap * 8, s)?;
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
            self.k.serve_topk(
                &ws.cands,
                &ws.sel,
                &ws.cells,
                self.geo.num_blocks,
                self.geo.per,
                topk,
                sq,
                b,
                s,
            )?;
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
                let ring_view = ws.ring_ptrs.slice(l * b * 8, b * 8)?;
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
    /// flow — router, aligned dispatch, grouped GEMMs, combine) over the
    /// **persistent bound-sized staging** in the serve workspace: no
    /// per-call allocations and no host readback, so the whole block is
    /// CUDA/HIP-graph capturable (P0-2). The dispatch zeroes the perm
    /// buffer up to the worst-case bound; the grouped GEMMs launch at the
    /// same bound and exit per-segment from the device-resident offsets,
    /// so replays never depend on the actual per-expert counts.
    #[allow(clippy::too_many_arguments)]
    fn serve_moe_prefill(&self, m: &MoeW, rows: usize, ws: &ServeWs) -> Result<()> {
        let cfg = &self.cfg;
        let hidden = cfg.moe.hidden;
        let inter = cfg.moe.intermediate;
        let e = cfg.moe.num_routed;
        let ki = cfg.moe.num_activated;
        let s = &self.stream;
        let align = MOE_ALIGN;
        let bound = moe_prefill_bound(rows, ki, e, align);
        let perm = ws.mpf_perm.slice(0, bound * hidden * 4)?;
        let act = ws.mpf_act.slice(0, bound * 2 * inter * 4)?;
        let silu = ws.mpf_silu.slice(0, bound * inter * 4)?;
        let down = ws.mpf_down.slice(0, bound * hidden * 4)?;
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
        self.k.moe_dispatch_bound(
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
            bound,
            s,
        )?;
        match m.routed_gate_up.dtype {
            GemmDType::Fp8 => {
                let aq = ws.mpf_aq.as_ref().unwrap();
                let asf = ws.mpf_asf.as_ref().unwrap();
                self.k
                    .quant_per_token_group(&perm, aq, asf, bound, hidden, 128, s)?;
                self.k.gemm_fp8_grouped_nt(
                    aq,
                    asf,
                    m.routed_gate_up.q.as_ref().unwrap(),
                    m.routed_gate_up.sf.as_ref().unwrap(),
                    &ws.moe_offsets,
                    &act,
                    bound,
                    2 * inter,
                    hidden,
                    e,
                    s,
                )?;
            }
            GemmDType::Nvfp4 => {
                self.k.gemm_grouped_nt_bound(
                    &perm,
                    m.routed_gate_up.q.as_ref().unwrap(),
                    m.routed_gate_up.sf.as_ref(),
                    &ws.moe_offsets,
                    &act,
                    bound,
                    2 * inter,
                    hidden,
                    e,
                    GemmDType::Nvfp4,
                    s,
                )?;
            }
            GemmDType::Bf16 => {
                let a16 = ws.mpf_a16.as_ref().unwrap();
                self.k.f32_to_bf16(&perm, a16, bound * hidden, s)?;
                self.k.gemm_grouped_nt_bound(
                    a16,
                    m.routed_gate_up.raw.as_ref().unwrap(),
                    None,
                    &ws.moe_offsets,
                    &act,
                    bound,
                    2 * inter,
                    hidden,
                    e,
                    GemmDType::Bf16,
                    s,
                )?;
            }
            GemmDType::F32 => {
                self.k.gemm_grouped_nt_bound(
                    &perm,
                    m.routed_gate_up.raw.as_ref().unwrap(),
                    None,
                    &ws.moe_offsets,
                    &act,
                    bound,
                    2 * inter,
                    hidden,
                    e,
                    GemmDType::F32,
                    s,
                )?;
            }
        }
        self.k.silu_mul_fused(&act, &silu, bound, inter, s)?;
        match m.routed_down.dtype {
            GemmDType::Fp8 => {
                let aq = ws.mpf_aq2.as_ref().unwrap();
                let asf = ws.mpf_asf2.as_ref().unwrap();
                self.k
                    .quant_per_token_group(&silu, aq, asf, bound, inter, 128, s)?;
                self.k.gemm_fp8_grouped_nt(
                    aq,
                    asf,
                    m.routed_down.q.as_ref().unwrap(),
                    m.routed_down.sf.as_ref().unwrap(),
                    &ws.moe_offsets,
                    &down,
                    bound,
                    hidden,
                    inter,
                    e,
                    s,
                )?;
            }
            GemmDType::Nvfp4 => {
                self.k.gemm_grouped_nt_bound(
                    &silu,
                    m.routed_down.q.as_ref().unwrap(),
                    m.routed_down.sf.as_ref(),
                    &ws.moe_offsets,
                    &down,
                    bound,
                    hidden,
                    inter,
                    e,
                    GemmDType::Nvfp4,
                    s,
                )?;
            }
            GemmDType::Bf16 => {
                let a16 = ws.mpf_a16.as_ref().unwrap();
                let a16_down = a16.slice(0, bound * inter * 2)?;
                self.k.f32_to_bf16(&silu, &a16_down, bound * inter, s)?;
                self.k.gemm_grouped_nt_bound(
                    &a16_down,
                    m.routed_down.raw.as_ref().unwrap(),
                    None,
                    &ws.moe_offsets,
                    &down,
                    bound,
                    hidden,
                    inter,
                    e,
                    GemmDType::Bf16,
                    s,
                )?;
            }
            GemmDType::F32 => {
                self.k.gemm_grouped_nt_bound(
                    &silu,
                    m.routed_down.raw.as_ref().unwrap(),
                    None,
                    &ws.moe_offsets,
                    &down,
                    bound,
                    hidden,
                    inter,
                    e,
                    GemmDType::F32,
                    s,
                )?;
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
        // Shared experts reuse the routed staging's head rows as scratch
        // (combine has already consumed them).
        for (gu, dn) in &m.shared {
            let su = act.slice(0, rows * 2 * inter * 4)?;
            let so = silu.slice(0, rows * inter * 4)?;
            let sd = down.slice(0, rows * hidden * 4)?;
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
            GemmDType::Nvfp4 => {
                self.k.gemm_grouped_masked_nt(
                    &ws.moe_perm,
                    None,
                    m.routed_gate_up.q.as_ref().unwrap(),
                    m.routed_gate_up.sf.as_ref(),
                    &ws.moe_counters,
                    &ws.moe_act,
                    cap,
                    2 * inter,
                    hidden,
                    e,
                    GemmDType::Nvfp4,
                    s,
                )?;
            }
            GemmDType::Fp8 => {
                let aq = ws.moe_aq.as_ref().unwrap();
                let asf = ws.moe_asf.as_ref().unwrap();
                self.k
                    .quant_per_token_group(&ws.moe_perm, aq, asf, rows, hidden, 128, s)?;
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
        self.k.silu_mul_masked(
            &ws.moe_act,
            &ws.moe_silu,
            &ws.moe_counters,
            cap,
            e,
            inter,
            s,
        )?;
        match m.routed_down.dtype {
            GemmDType::Nvfp4 => {
                self.k.gemm_grouped_masked_nt(
                    &ws.moe_silu,
                    None,
                    m.routed_down.q.as_ref().unwrap(),
                    m.routed_down.sf.as_ref(),
                    &ws.moe_counters,
                    &ws.moe_down,
                    cap,
                    hidden,
                    inter,
                    e,
                    GemmDType::Nvfp4,
                    s,
                )?;
            }
            GemmDType::Fp8 => {
                let aq = ws.moe_aq2.as_ref().unwrap();
                let asf = ws.moe_asf2.as_ref().unwrap();
                self.k
                    .quant_per_token_group(&ws.moe_silu, aq, asf, rows, inter, 128, s)?;
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
