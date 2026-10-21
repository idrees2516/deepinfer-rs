//! DSA (lightning indexer / conv / gate / top-K) and MoE launchers.

use deepinfer_core::{ensure, Result};
use deepinfer_gpu::{Buffer, KernelArgs, Launch, Stream};

use crate::gemm::ceil_div;
use crate::Kernels;

const MERGE_CHUNK: usize = 2048; // bitonic entries per merge CTA

impl Kernels {
    // ------------------------------------------------------------------
    // DSA: thunder branch
    // ------------------------------------------------------------------

    /// In-place per-(token, head) RMSNorm of `x` `[rows, nh*hd]`.
    pub fn indexer_rmsnorm_per_head(
        &self,
        x: &Buffer,
        rows: usize,
        nh: usize,
        hd: usize,
        stream: &Stream,
    ) -> Result<()> {
        self.launch1d(
            "di_indexer_rmsnorm_per_head_kernel",
            rows * nh,
            256,
            0,
            KernelArgs::new()
                .ptr(x)
                .i32(rows as i32)
                .i32(nh as i32)
                .i32(hd as i32),
            stream,
        )
    }

    /// Lightning quantization of `x_f32` (`[rows, dim]`, already per-head
    /// normed) into `q_u8` + `sf_u8`, using `amax_bits` (`[nblocks]` u32) as
    /// scratch. Only COMPLETE `quant_block` chunks are quantized (the CPU
    /// contract); trailing rows stay f32 and are consumed by
    /// [`Kernels::thunder_logits`].
    pub fn lightning_quant(
        &self,
        x_f32: &Buffer,
        q_u8: &Buffer,
        sf_u8: &Buffer,
        amax_bits: &Buffer,
        rows: usize,
        dim: usize,
        quant_block: usize,
        stream: &Stream,
    ) -> Result<()> {
        let nblocks = rows / quant_block;
        if nblocks == 0 {
            return Ok(()); // nothing to quantize (all rows stay f32)
        }
        // 1) zero the atomic amax accumulator
        self.fill_zero(amax_bits, nblocks * 4, stream)?;
        // 2) per-(block, chunk) partial amax -> atomicMax ordered bits
        let total_elems = nblocks * quant_block * dim;
        let chunk_elems = 1 << 15; // 32k elements per CTA
        let chunks = ceil_div(total_elems, chunk_elems);
        let elems_last = total_elems - (chunks - 1) * chunk_elems;
        let _ = elems_last;
        let f = self.f("di_lightning_amax_kernel")?;
        f.launch(
            self.device(),
            Launch {
                grid: (nblocks as u32, chunks as u32, 1),
                block: (256, 1, 1),
                shared: 0,
            },
            &KernelArgs::new()
                .ptr(x_f32)
                .ptr(amax_bits)
                .i32(dim as i32)
                .i32(quant_block as i32)
                .u64(chunk_elems as u64),
            stream,
        )?;
        // 3) amax bits -> ue8m0 scale bytes (device-side; ceil policy with
        //    the exact-power-of-two guard — same formula as the CPU)
        self.launch1d(
            "di_lightning_finalize_kernel",
            nblocks,
            256,
            0,
            KernelArgs::new().ptr(amax_bits).ptr(sf_u8).i32(nblocks as i32),
            stream,
        )?;
        // 4) quantize the complete blocks
        self.launch1d(
            "di_lightning_quant_kernel",
            nblocks * quant_block * dim,
            256,
            0,
            KernelArgs::new()
                .ptr(x_f32)
                .ptr(q_u8)
                .ptr(sf_u8)
                .i32(nblocks as i32)
                .i32(dim as i32)
                .i32(quant_block as i32),
            stream,
        )
    }

    /// Thunder logits: `scores[i][j] = sum_h relu(qtilde_i^h . ktilde_j^h)
    /// * w_i^h` over the quantized-or-f32 representations.
    #[allow(clippy::too_many_arguments)]
    pub fn thunder_logits(
        &self,
        q_f32: &Buffer,
        q_u8: &Buffer,
        q_sf: &Buffer,
        k_f32: &Buffer,
        k_u8: &Buffer,
        k_sf: &Buffer,
        w: &Buffer,
        scores: &Buffer,
        s_q: usize,
        s_kv: usize,
        nh: usize,
        hd: usize,
        q_nblocks: usize,
        k_nblocks: usize,
        quant_block: usize,
        stream: &Stream,
    ) -> Result<()> {
        let dim = nh * hd;
        let f = self.f("di_thunder_logits_kernel")?;
        f.launch(
            self.device(),
            Launch {
                grid: (ceil_div(s_kv, 256) as u32, s_q as u32, 1),
                block: (256, 1, 1),
                shared: (dim * 4) as u32,
            },
            &KernelArgs::new()
                .ptr(q_f32)
                .ptr(q_u8)
                .ptr(q_sf)
                .ptr(k_f32)
                .ptr(k_u8)
                .ptr(k_sf)
                .ptr(w)
                .ptr(scores)
                .i32(s_q as i32)
                .i32(s_kv as i32)
                .i32(nh as i32)
                .i32(hd as i32)
                .i32(q_nblocks as i32)
                .i32(k_nblocks as i32)
                .i32(quant_block as i32),
            stream,
        )
    }

    // ------------------------------------------------------------------
    // DSA: conv branch + gate
    // ------------------------------------------------------------------

    /// Cascade causal depthwise conv features for rows `[j0, j1)`.
    /// `kw`: flattened taps; `kw_off`: `[num_kernels + 1]` i32 offsets.
    pub fn conv_feat(
        &self,
        k: &Buffer,
        feat: &Buffer,
        kw: &Buffer,
        kw_off: &Buffer,
        num_kernels: usize,
        j0: usize,
        j1: usize,
        s_kv: usize,
        hidden: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(j1 > j0, InvalidArgument, "conv needs j1 > j0");
        self.launch1d(
            "di_conv_feat_kernel",
            (j1 - j0) * hidden,
            256,
            0,
            KernelArgs::new()
                .ptr(k)
                .ptr(feat)
                .ptr(kw)
                .ptr(kw_off)
                .i32(num_kernels as i32)
                .i32(j0 as i32)
                .i32(j1 as i32)
                .i32(s_kv as i32)
                .i32(hidden as i32),
            stream,
        )
    }

    /// Gated RMSNorm over `feat` (`[rows, hidden]`, in place).
    pub fn conv_gated_norm(
        &self,
        feat: &Buffer,
        gain: &Buffer,
        rows: usize,
        hidden: usize,
        stream: &Stream,
    ) -> Result<()> {
        self.launch1d(
            "di_conv_gated_norm_kernel",
            rows,
            256,
            0,
            KernelArgs::new()
                .ptr(feat)
                .ptr(gain)
                .i32(rows as i32)
                .i32(hidden as i32),
            stream,
        )
    }

    /// `out[i][j] = q_i . feat_j` (serial dot per element).
    pub fn dot_scores(
        &self,
        q: &Buffer,
        feat: &Buffer,
        out: &Buffer,
        s_q: usize,
        s_kv: usize,
        hidden: usize,
        stream: &Stream,
    ) -> Result<()> {
        let f = self.f("di_dot_scores_kernel")?;
        f.launch(
            self.device(),
            Launch {
                grid: (ceil_div(s_kv, 256) as u32, s_q as u32, 1),
                block: (256, 1, 1),
                shared: 0,
            },
            &KernelArgs::new()
                .ptr(q)
                .ptr(feat)
                .ptr(out)
                .i32(s_q as i32)
                .i32(s_kv as i32)
                .i32(hidden as i32),
            stream,
        )
    }

    /// Column mean of `q` (`[s_q, hidden]`) — i-ascending accumulation.
    pub fn qmean(
        &self,
        q: &Buffer,
        mean: &Buffer,
        s_q: usize,
        hidden: usize,
        stream: &Stream,
    ) -> Result<()> {
        self.launch1d(
            "di_qmean_kernel",
            hidden,
            256,
            0,
            KernelArgs::new()
                .ptr(q)
                .ptr(mean)
                .i32(s_q as i32)
                .i32(hidden as i32),
            stream,
        )
    }

    /// Gate logits per query row: `gt`/`gc` buffers (`[s_q]`).
    pub fn gate_logits(
        &self,
        q: &Buffer,
        mean: &Buffer,
        gate_proj: &Buffer,
        gt: &Buffer,
        gc: &Buffer,
        s_q: usize,
        hidden: usize,
        stream: &Stream,
    ) -> Result<()> {
        self.launch1d(
            "di_gate_logits_kernel",
            s_q,
            256,
            0,
            KernelArgs::new()
                .ptr(q)
                .ptr(mean)
                .ptr(gate_proj)
                .ptr(gt)
                .ptr(gc)
                .i32(s_q as i32)
                .i32(hidden as i32),
            stream,
        )
    }

    /// Blend thunder and conv scores with the gates (in place on `thunder`).
    pub fn gate_blend(
        &self,
        gt: &Buffer,
        gc: &Buffer,
        conv: &Buffer,
        thunder: &Buffer,
        s_q: usize,
        cols: usize,
        stream: &Stream,
    ) -> Result<()> {
        self.launch1d(
            "di_gate_blend_kernel",
            s_q * cols,
            256,
            0,
            KernelArgs::new()
                .ptr(gt)
                .ptr(gc)
                .ptr(conv)
                .ptr(thunder)
                .i32(s_q as i32)
                .i32(cols as i32),
            stream,
        )
    }

    /// Causal clip: `scores[i][j] = -inf` for `j > i`.
    pub fn causal_clip(
        &self,
        scores: &Buffer,
        s_q: usize,
        s_kv: usize,
        stream: &Stream,
    ) -> Result<()> {
        self.launch1d(
            "di_causal_clip_kernel",
            s_q * s_kv,
            256,
            0,
            KernelArgs::new()
                .ptr(scores)
                .i32(s_q as i32)
                .i32(s_kv as i32),
            stream,
        )
    }

    // ------------------------------------------------------------------
    // DSA: top-K selection (block preselect + merge tree + scatter)
    // ------------------------------------------------------------------

    /// Full selection: `scores` `[s_q, s_kv]` -> `out` `[s_q, k_out]` i32
    /// positions ascending, -1 padded. `cands_a`/`cands_b`: scratch
    /// `[s_q, cap]` u64 key buffers with `cap >= num_blocks * per`.
    #[allow(clippy::too_many_arguments)]
    pub fn select_top_k(
        &self,
        scores: &Buffer,
        cands_a: &Buffer,
        cands_b: &Buffer,
        cands_cap: usize,
        out: &Buffer,
        s_q: usize,
        s_kv: usize,
        block: usize,
        per: usize,
        k: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(s_q > 0, InvalidArgument, "select needs s_q > 0");
        ensure!(block <= 1024, InvalidArgument, "preselect block must be <= 1024");
        let num_blocks = ceil_div(s_kv, block);
        ensure!(
            cands_cap >= num_blocks * per,
            InvalidArgument,
            "cands scratch too small"
        );
        // 1) zero the candidate buffer (padding keys must be 0)
        self.fill_zero(cands_a, s_q * cands_cap * 8, stream)?;
        // 2) per-block preselect
        let f = self.f("di_block_preselect_kernel")?;
        f.launch(
            self.device(),
            Launch {
                grid: (num_blocks as u32, s_q as u32, 1),
                block: (block as u32, 1, 1),
                shared: (block * 8) as u32,
            },
            &KernelArgs::new()
                .ptr(scores)
                .ptr(cands_a)
                .i32(s_q as i32)
                .i32(s_kv as i32)
                .i32(block as i32)
                .i32(per as i32),
            stream,
        )?;
        // 3) merge rounds until <= MERGE_CHUNK candidates, then final top-K
        let mut cur = &cands_a;
        let mut nxt = &cands_b;
        let mut n = num_blocks * per;
        while n > MERGE_CHUNK {
            let keep = n / 2; // halving each round
            let f = self.f("di_topk_merge_kernel")?;
            f.launch(
                self.device(),
                Launch {
                    grid: (ceil_div(n, MERGE_CHUNK) as u32, s_q as u32, 1),
                    block: (1024, 1, 1),
                    shared: (MERGE_CHUNK * 8) as u32,
                },
                &KernelArgs::new()
                    .ptr(cur)
                    .ptr(nxt)
                    .i32(n as i32)
                    .i32(keep as i32)
                    .i32(MERGE_CHUNK as i32),
                stream,
            )?;
            std::mem::swap(&mut cur, &mut nxt);
            n = keep;
        }
        // 4) final top-K merge (keep = k)
        {
            let f = self.f("di_topk_merge_kernel")?;
            f.launch(
                self.device(),
                Launch {
                    grid: (1, s_q as u32, 1),
                    block: (1024, 1, 1),
                    shared: (MERGE_CHUNK * 8) as u32,
                },
                &KernelArgs::new()
                    .ptr(cur)
                    .ptr(nxt)
                    .i32(n as i32)
                    .i32(k as i32)
                    .i32(MERGE_CHUNK as i32),
                stream,
            )?;
        }
        // 5) scatter to ascending positions with -1 padding
        let k_pow2 = k.next_power_of_two();
        let f = self.f("di_topk_scatter_kernel")?;
        f.launch(
            self.device(),
            Launch {
                grid: (1, s_q as u32, 1),
                block: (256, 1, 1),
                shared: (k_pow2 * 4) as u32,
            },
            &KernelArgs::new()
                .ptr(nxt)
                .ptr(out)
                .i32(k as i32)
                .i32(k as i32),
            stream,
        )
    }

    // ------------------------------------------------------------------
    // MoE
    // ------------------------------------------------------------------

    /// Routing: `logits` `[tokens, E]` (pre-sigmoid GEMM output) ->
    /// `topk_w` `[tokens, K]` f32 (renormalized, noaux-tc semantics) and
    /// `topk_idx` `[tokens, K]` i32. Requires `E <= 1024` and
    /// group-free routing (V4.1 default).
    pub fn moe_route(
        &self,
        logits: &Buffer,
        bias: Option<&Buffer>,
        topk_w: &Buffer,
        topk_idx: &Buffer,
        tokens: usize,
        e: usize,
        k: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(tokens > 0 && e > 0 && k > 0, InvalidArgument, "route needs positive dims");
        ensure!(e <= 1024, InvalidArgument, "route supports up to 1024 experts");
        ensure!(k <= e, InvalidArgument, "k must be <= E");
        let pow2 = e.next_power_of_two();
        let shared = pow2 * 8 + e * 4;
        let f = self.f("di_moe_route_kernel")?;
        f.launch(
            self.device(),
            Launch {
                grid: (tokens as u32, 1, 1),
                block: (256, 1, 1),
                shared: shared as u32,
            },
            &KernelArgs::new()
                .ptr(logits)
                .u64(bias.map(|b| b.ptr()).unwrap_or(0))
                .ptr(topk_w)
                .ptr(topk_idx)
                .i32(tokens as i32)
                .i32(e as i32)
                .i32(k as i32)
                .i32(bias.is_some() as i32),
            stream,
        )
    }

    /// Dispatch orchestration: counts -> aligned offsets -> scatter. All
    /// scratch buffers are caller-allocated:
    /// * `counts`, `rank_counters`: `[E]` i32 (zeroed here)
    /// * `offsets`: `[E + 1]` i32
    /// * `perm`: `[m_total_padded, hidden]` f32 (zeroed here)
    /// * `map`: `[tokens, K]` i32
    /// Returns `m_total_padded` (the grouped GEMM M).
    pub fn moe_dispatch(
        &self,
        x: &Buffer,
        topk_idx: &Buffer,
        counts: &Buffer,
        offsets: &Buffer,
        rank_counters: &Buffer,
        perm: &Buffer,
        map: &Buffer,
        tokens: usize,
        hidden: usize,
        k: usize,
        e: usize,
        align: usize,
        stream: &Stream,
    ) -> Result<usize> {
        ensure!(align % 128 == 0 || align == 1, InvalidArgument, "align must be a multiple of 128");
        self.fill_zero(counts, e * 4, stream)?;
        self.fill_zero(rank_counters, e * 4, stream)?;
        self.launch1d(
            "di_moe_count_kernel",
            tokens * k,
            256,
            0,
            KernelArgs::new()
                .ptr(topk_idx)
                .ptr(counts)
                .i32(tokens as i32)
                .i32(k as i32)
                .i32(e as i32),
            stream,
        )?;
        // offsets need counts on device -> read back the padded total
        self.launch1d(
            "di_moe_prefix_kernel",
            256,
            256,
            0,
            KernelArgs::new()
                .ptr(counts)
                .ptr(offsets)
                .i32(e as i32)
                .i32(align as i32),
            stream,
        )?;
        let mut off_host = vec![0i32; e + 1];
        stream.synchronize()?;
        self.device().read(offsets, &mut off_host)?;
        let m_total = off_host[e] as usize;
        // zero the perm buffer (alignment padding rows must be zero)
        self.fill_zero(perm, m_total * hidden * 4, stream)?;
        let f = self.f("di_moe_scatter_kernel")?;
        f.launch(
            self.device(),
            Launch {
                grid: ((tokens * k) as u32, 1, 1),
                block: (256, 1, 1),
                shared: 0,
            },
            &KernelArgs::new()
                .ptr(x)
                .ptr(topk_idx)
                .ptr(offsets)
                .ptr(rank_counters)
                .ptr(perm)
                .ptr(map)
                .i32(tokens as i32)
                .i32(hidden as i32)
                .i32(k as i32),
            stream,
        )?;
        Ok(m_total)
    }

    /// Combine: `y[t] = sum_k w_k * expert_out[map(t,k)]` (k-ascending).
    pub fn moe_combine(
        &self,
        perm_out: &Buffer,
        map: &Buffer,
        topk_w: &Buffer,
        y: &Buffer,
        tokens: usize,
        hidden: usize,
        k: usize,
        stream: &Stream,
    ) -> Result<()> {
        let f = self.f("di_moe_combine_kernel")?;
        f.launch(
            self.device(),
            Launch {
                grid: (tokens as u32, 1, 1),
                block: (256, 1, 1),
                shared: 0,
            },
            &KernelArgs::new()
                .ptr(perm_out)
                .ptr(map)
                .ptr(topk_w)
                .ptr(y)
                .i32(tokens as i32)
                .i32(hidden as i32)
                .i32(k as i32),
            stream,
        )
    }

    /// Zero `n` bytes of a device buffer.
    pub fn fill_zero(&self, p: &Buffer, n: usize, stream: &Stream) -> Result<()> {
        self.launch1d(
            "di_fill_zero_kernel",
            n.max(1),
            256,
            0,
            KernelArgs::new().raw_ptr(p.ptr()).u64(n as u64),
            stream,
        )
    }

    /// Elementwise sigmoid (router logits -> scores).
    pub fn sigmoid(&self, z: &Buffer, s: &Buffer, n: usize, stream: &Stream) -> Result<()> {
        self.launch1d(
            "di_sigmoid_kernel",
            n.max(1),
            256,
            0,
            KernelArgs::new().ptr(z).ptr(s).i32(n as i32),
            stream,
        )
    }
}
