//! Elementwise / norm / RoPE / quantization launchers.

use deepinfer_core::{ensure, Result};
use deepinfer_gpu::{Buffer, KernelArgs, Stream};

use crate::Kernels;

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
