//! GEMM launchers (DeepGEMM semantics; see kernels/gemm.cu).

use deepinfer_core::{ensure, Result};
use deepinfer_gpu::{Buffer, KernelArgs, Launch, Stream};

use crate::Kernels;

const BM: usize = 64;
const BN: usize = 64;

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
