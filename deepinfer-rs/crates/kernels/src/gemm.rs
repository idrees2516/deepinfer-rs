//! GEMM launchers (DeepGEMM semantics; see kernels/gemm.cu).

use deepinfer_core::{ensure, Result};
use deepinfer_gpu::{Buffer, KernelArgs, Launch, Stream};

use crate::Kernels;

const BM: usize = 64;
const BN: usize = 64;

/// Kernel entry points defined in `kernels/gemm.cu` (used by the
/// source-name sanity test).
#[cfg(test)]
pub(crate) const KERNEL_NAMES: &[&str] = &[
    "di_gemm_bf16_nt_kernel",
    "di_gemm_f32_nt_kernel",
    "di_gemm_fp8_nt_kernel",
    "di_gemm_fp8_grouped_nt_kernel",
    "di_gemm_grouped_masked_nt_kernel",
];

/// Operand dtype ids of the device GEMM kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmDType {
    /// f32 operands, f32 accumulate (the reference-precision path).
    F32 = 0,
    /// bf16 operands (decoded during staging), f32 accumulate.
    Bf16 = 1,
    /// FP8 e4m3 operands with ue8m0 block scales (DeepGEMM recipe).
    Fp8 = 2,
}

impl Kernels {
    /// Dense BF16 NT GEMM: `D = C + A @ B^T`. `a`: `[m, k]` bf16 bits,
    /// `b`: `[n, k]` bf16 bits, `c`: optional `[m, n]` f32 addend,
    /// `d`: `[m, n]` f32 output.
    pub fn gemm_bf16_nt(
        &self,
        a: &Buffer,
        b: &Buffer,
        c: Option<&Buffer>,
        d: &Buffer,
        m: usize,
        n: usize,
        k: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(m > 0 && n > 0 && k > 0, InvalidArgument, "gemm needs m/n/k > 0");
        let f = self.f("di_gemm_bf16_nt_kernel")?;
        f.launch(
            self.device(),
            gemm_launch(n, m),
            &KernelArgs::new()
                .ptr(a)
                .ptr(b)
                .u64(c.map(|b| b.ptr()).unwrap_or(0))
                .ptr(d)
                .i32(m as i32)
                .i32(n as i32)
                .i32(k as i32),
            stream,
        )
    }

    /// Dense F32 NT GEMM: `D = C + A @ B^T` — the reference-precision path
    /// (scalar core, k-ascending sequential f32 accumulation on both
    /// vendors; the association order matches the CPU `dot`).
    pub fn gemm_f32_nt(
        &self,
        a: &Buffer,
        b: &Buffer,
        c: Option<&Buffer>,
        d: &Buffer,
        m: usize,
        n: usize,
        k: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(m > 0 && n > 0 && k > 0, InvalidArgument, "gemm needs m/n/k > 0");
        let f = self.f("di_gemm_f32_nt_kernel")?;
        f.launch(
            self.device(),
            gemm_launch(n, m),
            &KernelArgs::new()
                .ptr(a)
                .ptr(b)
                .u64(c.map(|b| b.ptr()).unwrap_or(0))
                .ptr(d)
                .i32(m as i32)
                .i32(n as i32)
                .i32(k as i32),
            stream,
        )
    }

    /// Dense FP8 block-scaled NT GEMM (activations 1x128 + weights 128x128,
    /// ue8m0 scales). Requires `k % 128 == 0` and `a_sf`/`b_sf` layouts per
    /// the DeepGEMM contract (see kernels/gemm.cu header).
    pub fn gemm_fp8_nt(
        &self,
        a: &Buffer,
        a_sf: &Buffer,
        b: &Buffer,
        b_sf: &Buffer,
        d: &Buffer,
        m: usize,
        n: usize,
        k: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(m > 0 && n > 0 && k > 0, InvalidArgument, "gemm needs m/n/k > 0");
        ensure!(k % 128 == 0, InvalidArgument, "fp8 gemm requires k % 128 == 0");
        let f = self.f("di_gemm_fp8_nt_kernel")?;
        f.launch(
            self.device(),
            gemm_launch(n, m),
            &KernelArgs::new()
                .ptr(a)
                .ptr(a_sf)
                .ptr(b)
                .ptr(b_sf)
                .ptr(d)
                .i32(m as i32)
                .i32(n as i32)
                .i32(k as i32),
            stream,
        )
    }

    /// Grouped-contiguous FP8 GEMM (MoE prefill; DeepGEMM
    /// `m_grouped_gemm_nt_contiguous` semantics). `m_offsets`: `[g+1]` i32
    /// cumulative segment starts (each segment a multiple of 128);
    /// `b`: `[g, n, k]`; `b_sf`: `[g, ceil(n/128), k/128]`; `d`:
    /// `[m_total, n]` f32.
    pub fn gemm_fp8_grouped_nt(
        &self,
        a: &Buffer,
        a_sf: &Buffer,
        b: &Buffer,
        b_sf: &Buffer,
        m_offsets: &Buffer,
        d: &Buffer,
        m_total: usize,
        n: usize,
        k: usize,
        g: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(g > 0 && n > 0 && k > 0 && k % 128 == 0, InvalidArgument, "grouped gemm bad shape");
        ensure!(m_total % 128 == 0, InvalidArgument, "m_total must be 128-aligned");
        let f = self.f("di_gemm_fp8_grouped_nt_kernel")?;
        f.launch(
            self.device(),
            gemm_launch(n, m_total.max(1)),
            &KernelArgs::new()
                .ptr(a)
                .ptr(a_sf)
                .ptr(b)
                .ptr(b_sf)
                .ptr(m_offsets)
                .ptr(d)
                .i32(m_total as i32)
                .i32(n as i32)
                .i32(k as i32)
                .i32(g as i32),
            stream,
        )
    }

    /// **Masked grouped NT GEMM** — the CUDA-graph / low-latency-EP MoE
    /// path (DeepGEMM `m_grouped_gemm_nt_masked` semantics).
    ///
    /// `a`: `[g * cap, k]` (capacity-strided groups); `a_sf`:
    /// `[g * cap, k / 128]` (fp8 only, else `None`); `b`: `[g, n, k]`;
    /// `b_sf`: `[g, ceil(n / 128), k / 128]` (fp8 only); `masked_m`:
    /// `[g]` i32 read **on device** (no host readback — capturable);
    /// `d`: `[g * cap, n]` f32. Only rows `r < min(masked_m[g], cap)` of
    /// each group are computed.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_grouped_masked_nt(
        &self,
        a: &Buffer,
        a_sf: Option<&Buffer>,
        b: &Buffer,
        b_sf: Option<&Buffer>,
        masked_m: &Buffer,
        d: &Buffer,
        cap: usize,
        n: usize,
        k: usize,
        g: usize,
        dtype: GemmDType,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            g > 0 && n > 0 && k > 0 && cap > 0,
            InvalidArgument,
            "masked grouped gemm needs g/n/k/cap > 0"
        );
        ensure!(
            g.checked_mul(cap).is_some() && (g as u64) * (cap as u64) <= i32::MAX as u64,
            InvalidArgument,
            "g*cap overflows the i32 row space"
        );
        if dtype == GemmDType::Fp8 {
            ensure!(k % 128 == 0, InvalidArgument, "fp8 gemm requires k % 128 == 0");
            ensure!(a_sf.is_some() && b_sf.is_some(), InvalidArgument, "fp8 needs scale buffers");
        }
        let m_tiles = cap.div_ceil(BM);
        let grid_y = g
            .checked_mul(m_tiles)
            .filter(|&v| v <= u32::MAX as usize)
            .ok_or_else(|| {
                deepinfer_core::Error::CapacityExceeded("masked gemm grid.y overflows".into())
            })?;
        let f = self.f("di_gemm_grouped_masked_nt_kernel")?;
        f.launch(
            self.device(),
            Launch {
                grid: (n.div_ceil(BN) as u32, grid_y as u32, 1),
                block: (256, 1, 1),
                shared: 0,
            },
            &KernelArgs::new()
                .ptr(a)
                .u64(a_sf.map(|s| s.ptr()).unwrap_or(0))
                .ptr(b)
                .u64(b_sf.map(|s| s.ptr()).unwrap_or(0))
                .ptr(masked_m)
                .ptr(d)
                .i32(cap as i32)
                .i32(n as i32)
                .i32(k as i32)
                .i32(g as i32)
                .i32(dtype as i32),
            stream,
        )
    }
}

fn gemm_launch(n: usize, m: usize) -> Launch {
    Launch {
        grid: (ceil_div(n, BN) as u32, ceil_div(m, BM) as u32, 1),
        block: (256, 1, 1),
        shared: 0,
    }
}

pub(crate) fn ceil_div(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}
