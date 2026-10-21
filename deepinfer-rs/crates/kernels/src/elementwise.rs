//! Elementwise / norm / RoPE / quantization launchers.

use deepinfer_core::{ensure, Result};
use deepinfer_gpu::{Buffer, KernelArgs, Stream};

use crate::Kernels;

/// Kernel entry points defined in `kernels/elementwise.cu` (used by the
/// source-name sanity test).
#[cfg(test)]
pub(crate) const KERNEL_NAMES: &[&str] = &[
    "di_rmsnorm_kernel",
    "di_rope_kernel",
    "di_silu_mul_kernel",
    "di_add_kernel",
    "di_quant_per_token_group_kernel",
    "di_kv_encode_fp8_kernel",
    "di_kv_encode_fp4_kernel",
    "di_f32_to_bf16_kernel",
    "di_bf16_to_f32_kernel",
    "di_latent_norm_kernel",
    "di_append_rows_kernel",
    "di_append_norm_rows_kernel",
    "di_copy_cols_kernel",
];

/// KV cache row format ids (attention.cu contract).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheFormat {
    /// f32 rows `[rows, D]`.
    F32,
    /// FlashMLA V4.1 FP8 rows (528 B per 512-dim token).
    Fp8Group32,
    /// FlashMLA V4.1 FP4 rows (288 B per 512-dim token).
    Fp4Group16,
}

impl CacheFormat {
    /// Kernel id.
    pub fn id(self) -> i32 {
        match self {
            CacheFormat::F32 => 0,
            CacheFormat::Fp8Group32 => 1,
            CacheFormat::Fp4Group16 => 2,
        }
    }

    /// Bytes per cached token for `dims` dimensions.
    pub fn bytes_per_token(self, dims: usize) -> usize {
        match self {
            CacheFormat::F32 => dims * 4,
            CacheFormat::Fp8Group32 => dims + dims / 32,
            CacheFormat::Fp4Group16 => dims / 2 + dims / 16,
        }
    }
}

impl Kernels {
    /// Host-side warp size of the active backend (needed for warp-grouped
    /// launch geometry).
    pub fn warp_size(&self) -> u32 {
        match self.device().backend() {
            deepinfer_gpu::Backend::Cuda => 32,
            deepinfer_gpu::Backend::Hip => 64,
        }
    }

    /// `out = rmsnorm(x) * gain` for `[rows, cols]` f32.
    pub fn rmsnorm(
        &self,
        x: &Buffer,
        gain: &Buffer,
        out: &Buffer,
        rows: usize,
        cols: usize,
        eps: f32,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(rows > 0 && cols > 0, InvalidArgument, "rmsnorm needs rows/cols > 0");
        self.launch1d(
            "di_rmsnorm_kernel",
            rows,
            256,
            0,
            KernelArgs::new()
                .ptr(x)
                .ptr(gain)
                .ptr(out)
                .i32(rows as i32)
                .i32(cols as i32)
                .f32(eps),
            stream,
        )
    }

    /// Interleaved RoPE (or its conjugate) over `half` pairs starting at
    /// `offset` in each row of `x` (`rows` rows of `row_len`).
    /// `cs` is the host-built cos/sin table `[max_pos][half][2]`.
    pub fn rope(
        &self,
        cs: &Buffer,
        pos: &Buffer,
        x: &Buffer,
        rows: usize,
        row_len: usize,
        offset: usize,
        half: usize,
        conjugate: bool,
        stream: &Stream,
    ) -> Result<()> {
        let total = rows * half;
        self.launch1d(
            "di_rope_kernel",
            total.max(1),
            256,
            0,
            KernelArgs::new()
                .ptr(cs)
                .ptr(pos)
                .ptr(x)
                .i32(rows as i32)
                .i32(row_len as i32)
                .i32(offset as i32)
                .i32(half as i32)
                .i32(conjugate as i32),
            stream,
        )
    }

    /// `out = silu(g) * u`.
    pub fn silu_mul(
        &self,
        g: &Buffer,
        u: &Buffer,
        out: &Buffer,
        n: usize,
        stream: &Stream,
    ) -> Result<()> {
        self.launch1d(
            "di_silu_mul_kernel",
            n.max(1),
            256,
            0,
            KernelArgs::new().ptr(g).ptr(u).ptr(out).i32(n as i32),
            stream,
        )
    }

    /// `y += x`.
    pub fn add(&self, x: &Buffer, y: &Buffer, n: usize, stream: &Stream) -> Result<()> {
        self.launch1d(
            "di_add_kernel",
            n.max(1),
            256,
            0,
            KernelArgs::new().ptr(x).ptr(y).i32(n as i32),
            stream,
        )
    }

    /// Elementwise f32 -> bf16 (RNE) cast.
    pub fn f32_to_bf16(&self, x: &Buffer, out: &Buffer, n: usize, stream: &Stream) -> Result<()> {
        self.launch1d(
            "di_f32_to_bf16_kernel",
            n.max(1),
            256,
            0,
            KernelArgs::new().ptr(x).ptr(out).i32(n as i32),
            stream,
        )
    }

    /// Elementwise bf16 -> f32 cast (exact).
    pub fn bf16_to_f32(&self, x: &Buffer, out: &Buffer, n: usize, stream: &Stream) -> Result<()> {
        self.launch1d(
            "di_bf16_to_f32_kernel",
            n.max(1),
            256,
            0,
            KernelArgs::new().ptr(x).ptr(out).i32(n as i32),
            stream,
        )
    }

    /// Per-token-group E4M3 + UE8M0 quantization (DeepGEMM activation
    /// recipe). `x`: `[rows, cols]` f32; `q`: `[rows, cols]` u8;
    /// `sf`: `[rows, cols/group]` u8. Requires `cols % group == 0`.
    pub fn quant_per_token_group(
        &self,
        x: &Buffer,
        q: &Buffer,
        sf: &Buffer,
        rows: usize,
        cols: usize,
        group: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(cols % group == 0, InvalidArgument, "cols must be a multiple of group");
        let warps_per_block = 256 / self.warp_size() as usize;
        let total_warps = rows * (cols / group);
        let grid = total_warps.div_ceil(warps_per_block);
        let f = self.f("di_quant_per_token_group_kernel")?;
        f.launch(
            self.device(),
            deepinfer_gpu::Launch {
                grid: (grid.max(1) as u32, 1, 1),
                block: (256, 1, 1),
                shared: 0,
            },
            &KernelArgs::new()
                .ptr(x)
                .ptr(q)
                .ptr(sf)
                .i32(rows as i32)
                .i32(cols as i32)
                .i32(group as i32),
            stream,
        )
    }

    /// FlashMLA V4.1 FP8 KV-cache encode: `x` `[tokens, 512]` f32 ->
    /// `out` `[tokens, 528]` u8.
    pub fn kv_encode_fp8(
        &self,
        x: &Buffer,
        out: &Buffer,
        tokens: usize,
        dims: usize,
        stream: &Stream,
    ) -> Result<()> {
        self.kv_encode(x, out, tokens, dims, "di_kv_encode_fp8_kernel", stream)
    }

    /// FlashMLA V4.1 FP4 KV-cache encode: `x` `[tokens, 512]` f32 ->
    /// `out` `[tokens, 288]` u8.
    pub fn kv_encode_fp4(
        &self,
        x: &Buffer,
        out: &Buffer,
        tokens: usize,
        dims: usize,
        stream: &Stream,
    ) -> Result<()> {
        self.kv_encode(x, out, tokens, dims, "di_kv_encode_fp4_kernel", stream)
    }

    fn kv_encode(
        &self,
        x: &Buffer,
        out: &Buffer,
        tokens: usize,
        dims: usize,
        kernel: &'static str,
        stream: &Stream,
    ) -> Result<()> {
        let group = if kernel.ends_with("fp8_kernel") { 32 } else { 16 };
        let warps_per_block = 64 / self.warp_size() as usize;
        let total_warps = tokens * (dims / group);
        let grid = total_warps.div_ceil(warps_per_block);
        let f = self.f(kernel)?;
        f.launch(
            self.device(),
            deepinfer_gpu::Launch {
                grid: (grid.max(1) as u32, 1, 1),
                block: (64, 1, 1),
                shared: 0,
            },
            &KernelArgs::new()
                .ptr(x)
                .ptr(out)
                .i32(tokens as i32)
                .i32(dims as i32),
            stream,
        )
    }
}

impl Kernels {
    /// In-place latent-prefix RMSNorm of `kv` (`[rows, hd]` rows; the first
    /// `latent` dims of each row are normed with `gain`, the rope tail is
    /// left raw) — the MLA KV latent norm.
    pub fn rmsnorm_latent(
        &self,
        kv: &Buffer,
        gain: &Buffer,
        rows: usize,
        hd: usize,
        latent: usize,
        eps: f32,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(rows > 0 && hd > 0 && latent > 0 && latent <= hd, InvalidArgument, "latent norm needs 0 < latent <= hd");
        self.launch1d(
            "di_latent_norm_kernel",
            rows,
            256,
            0,
            KernelArgs::new().ptr(kv).ptr(gain).i32(rows as i32).i32(hd as i32).i32(latent as i32).f32(eps),
            stream,
        )
    }

    /// Batched row append `dst[row0 + i] = src[i]` for `i in [0, rows)`
    /// (row stride `floats`).
    pub fn append_rows(
        &self,
        src: &Buffer,
        dst: &Buffer,
        row0: usize,
        rows: usize,
        floats: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(rows > 0 && floats > 0, InvalidArgument, "append_rows needs rows/floats > 0");
        self.launch1d(
            "di_append_rows_kernel",
            rows * floats,
            256,
            0,
            KernelArgs::new()
                .ptr(src)
                .ptr(dst)
                .i32(row0 as i32)
                .i32(rows as i32)
                .i32(floats as i32),
            stream,
        )
    }

    /// Batched per-(token, head) RMSNormed row append (thunder-side k cache).
    pub fn append_norm_rows(
        &self,
        src: &Buffer,
        dst: &Buffer,
        row0: usize,
        rows: usize,
        nh: usize,
        hd: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(rows > 0 && nh > 0 && hd > 0, InvalidArgument, "append_norm_rows needs positive dims");
        self.launch1d(
            "di_append_norm_rows_kernel",
            rows * nh,
            256,
            0,
            KernelArgs::new()
                .ptr(src)
                .ptr(dst)
                .i32(row0 as i32)
                .i32(rows as i32)
                .i32(nh as i32)
                .i32(hd as i32),
            stream,
        )
    }

    /// Strided column-block copy
    /// `dst[i][dst_col0 + c] = src[i][src_col0 + c]` (strides in floats).
    #[allow(clippy::too_many_arguments)]
    pub fn copy_cols(
        &self,
        src: &Buffer,
        dst: &Buffer,
        rows: usize,
        src_stride: usize,
        dst_stride: usize,
        src_col0: usize,
        dst_col0: usize,
        cols: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(rows > 0 && cols > 0, InvalidArgument, "copy_cols needs rows/cols > 0");
        self.launch1d(
            "di_copy_cols_kernel",
            rows * cols,
            256,
            0,
            KernelArgs::new()
                .ptr(src)
                .ptr(dst)
                .i32(rows as i32)
                .u64(src_stride as u64)
                .u64(dst_stride as u64)
                .u64(src_col0 as u64)
                .u64(dst_col0 as u64)
                .i32(cols as i32),
            stream,
        )
    }
}

impl Kernels {
    /// Indexed row gather `dst[i] = src[idx[i]]` for `i in [0, rows)`
    /// (whole rows of `row_bytes`; host-built index lists uploaded by the
    /// caller). Used for the embedding table gather and the EP send-buffer
    /// packing.
    pub fn gather_indexed(
        &self,
        src: &Buffer,
        dst: &Buffer,
        idx: &Buffer,
        rows: usize,
        row_bytes: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(rows >= 1, InvalidArgument, "gather needs rows >= 1");
        self.launch1d(
            "di_gather_rows_indexed_kernel",
            rows * row_bytes,
            256,
            0,
            KernelArgs::new()
                .ptr(src)
                .ptr(dst)
                .ptr(idx)
                .i32(row_bytes as i32)
                .u64(rows as u64),
            stream,
        )
    }
}
