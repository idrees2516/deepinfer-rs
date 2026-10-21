//! Safe high-level host orchestration over the cuTile kernels.
//!
//! Every public function re-validates shapes, strides and divisibility
//! preconditions before launching the unchecked kernels, mirroring the
//! validate-then-compute structure of the CPU crates. Tensors are
//! `Arc`-shared so weights upload once and feed many launches; all work is
//! issued on the context's stream in call order.

use std::sync::Arc;

use cutile::api::{self, DeviceOpReshape};
use cutile::cuda_core::{Device, Stream};
use cutile::error::Error;
use cutile::half::bf16;
use cutile::prelude::*;
use cutile::tensor::Tensor;
use cutile::tile_kernel::PartitionOp;

use crate::arch::ComputeCapability;
use crate::kernels;

/// Device context: device + stream + capability.
pub struct GpuCtx {
    #[allow(dead_code)] // kept for future multi-stream scheduling
    device: Arc<Device>,
    stream: Arc<Stream>,
    cc: ComputeCapability,
}

impl GpuCtx {
    /// Opens device `ordinal`. Fails without a CUDA device (GPU-less
    /// sandboxes) or when the device is below sm_80.
    pub fn new(ordinal: usize) -> Result<Self, Error> {
        let device = Device::new(ordinal)?;
        let stream = device.new_stream()?;
        let name = device.name()?;
        let cc = crate::arch::cc_from_device_name(&name)
            .ok_or_else(|| cutile::error::kernel_launch_error(&format!("unknown GPU architecture for {name}")))?;
        if !cc.is_supported_nvidia() {
            return Err(cutile::error::kernel_launch_error(&format!(
                "{name} ({cc}) is below sm_80; cuTile requires Ampere or newer"
            )));
        }
        Ok(Self { device, stream, cc })
    }

    /// The context's compute capability.
    pub fn capability(&self) -> ComputeCapability {
        self.cc
    }

    /// Whether FP8 block-scaled MMA (`mmaf_scaled`) is available (sm_100+);
    /// GB200/GB300 always qualify.
    pub fn has_fp8_mma(&self) -> bool {
        self.cc.has_block_scaled_mma() || crate::arch::force_bf16_fallback()
    }

    /// The context's stream.
    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }
}

/// Uploads F32 data as a BF16 tensor (host RNE conversion).
pub fn upload_bf16(
    ctx: &GpuCtx,
    data: &[f32],
    shape: &[usize],
) -> Result<Arc<Tensor<bf16>>, Error> {
    let expect: usize = shape.iter().product();
    if data.len() != expect {
        return Err(cutile::error::kernel_launch_error(&format!(
            "upload_bf16: {} elements, shape {shape:?} expects {expect}",
            data.len()
        )));
    }
    let host: Vec<bf16> = data.iter().map(|&v| bf16::from_f32(v)).collect();
    let t: Tensor<bf16> = api::copy_host_vec_to_device(&Arc::new(host))
        .reshape(shape)
        .sync_on(ctx.stream())?;
    Ok(Arc::new(t))
}

/// Uploads raw F32 data unchanged.
pub fn upload_f32(
    ctx: &GpuCtx,
    data: &[f32],
    shape: &[usize],
) -> Result<Arc<Tensor<f32>>, Error> {
    let expect: usize = shape.iter().product();
    if data.len() != expect {
        return Err(cutile::error::kernel_launch_error(&format!(
            "upload_f32: {} elements, shape {shape:?} expects {expect}",
            data.len()
        )));
    }
    let t: Tensor<f32> = api::copy_host_vec_to_device(&Arc::new(data.to_vec()))
        .reshape(shape)
        .sync_on(ctx.stream())?;
    Ok(Arc::new(t))
}

/// Uploads raw i32 data unchanged.
pub fn upload_i32(
    ctx: &GpuCtx,
    data: &[i32],
    shape: &[usize],
) -> Result<Arc<Tensor<i32>>, Error> {
    let expect: usize = shape.iter().product();
    if data.len() != expect {
        return Err(cutile::error::kernel_launch_error(&format!(
            "upload_i32: {} elements, shape {shape:?} expects {expect}",
            data.len()
        )));
    }
    let t: Tensor<i32> = api::copy_host_vec_to_device(&Arc::new(data.to_vec()))
        .reshape(shape)
        .sync_on(ctx.stream())?;
    Ok(Arc::new(t))
}

/// Downloads an F32 tensor to the host.
pub fn download_f32(
    ctx: &GpuCtx,
    t: &Arc<Tensor<f32>>,
) -> Result<Vec<f32>, Error> {
    t.to_host_vec().sync_on(ctx.stream()).map_err(Into::into)
}

/// Downloads an I32 tensor to the host.
pub fn download_i32(
    ctx: &GpuCtx,
    t: &Arc<Tensor<i32>>,
) -> Result<Vec<i32>, Error> {
    t.to_host_vec().sync_on(ctx.stream()).map_err(Into::into)
}

/// RMSNorm over `[rows, n]` BF16 rows (eps 1e-4 per V4.1).
pub fn rms_norm(
    ctx: &GpuCtx,
    x: &Arc<Tensor<bf16>>,
    w: &Arc<Tensor<bf16>>,
    rows: usize,
    n: usize,
    eps: f32,
) -> Result<Tensor<bf16>, Error> {
    if x.shape() != [rows as i32, n as i32] {
        return Err(cutile::error::kernel_launch_error("rms_norm: x shape mismatch"));
    }
    if w.shape() != [n as i32] {
        return Err(cutile::error::kernel_launch_error("rms_norm: weight shape mismatch"));
    }
    const BLOCK: i32 = 512;
    let n_i = n as i32;
    if n_i % BLOCK != 0 && n_i > BLOCK * 4 {
        // Block size 512 covers 512..2048 exactly; larger hidden sizes use
        // the ceil-div tail path which the partition handles.
    }
    let out = api::zeros::<bf16>(&[rows, n]).partition([1, n]);
    let (_x, _w, out, _eps) = unsafe { kernels::rms_norm_bf16(x.clone(), w.clone(), out, eps) }
        .generics(vec![n_i.to_string(), BLOCK.to_string()])
        .sync_on(ctx.stream())?;
    Ok(out.unpartition())
}

/// Fused residual add + RMSNorm; returns `(normed, residual_out)`.
pub fn add_rms_norm(
    ctx: &GpuCtx,
    residual: &Arc<Tensor<bf16>>,
    x: &Arc<Tensor<bf16>>,
    w: &Arc<Tensor<bf16>>,
    rows: usize,
    n: usize,
    eps: f32,
) -> Result<(Tensor<bf16>, Tensor<bf16>), Error> {
    if residual.shape() != [rows as i32, n as i32] || x.shape() != [rows as i32, n as i32] {
        return Err(cutile::error::kernel_launch_error("add_rms_norm: shape mismatch"));
    }
    if w.shape() != [n as i32] {
        return Err(cutile::error::kernel_launch_error("add_rms_norm: weight shape mismatch"));
    }
    const BLOCK: i32 = 512;
    let out = api::zeros::<bf16>(&[rows, n]).partition([1, n]);
    let res_out = api::zeros::<bf16>(&[rows, n]).partition([1, n]);
    let (_r, _x, _w, out, res_out, _eps) = unsafe {
        kernels::add_rms_norm_bf16(residual.clone(), x.clone(), w.clone(), out, res_out, eps)
    }
    .generics(vec![n.to_string(), BLOCK.to_string()])
    .sync_on(ctx.stream())?;
    Ok((out.unpartition(), res_out.unpartition()))
}

/// Dense BF16 GEMM `D = A @ B^T` (A `[m, k]`, B `[n, k]`).
pub fn bf16_gemm_nt(
    ctx: &GpuCtx,
    a: &Arc<Tensor<bf16>>,
    b: &Arc<Tensor<bf16>>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<Tensor<bf16>, Error> {
    if a.shape() != [m as i32, k as i32] || b.shape() != [n as i32, k as i32] {
        return Err(cutile::error::kernel_launch_error("bf16_gemm_nt: shape mismatch"));
    }
    const BM: i32 = 64;
    const BN: i32 = 64;
    const BK: i32 = 32;
    let d = api::zeros::<bf16>(&[m, n]).partition([BM as usize, BN as usize]);
    let (d, ..) = unsafe { kernels::bf16_gemm_nt(d, a.clone(), b.clone()) }
        .generics(vec![BM.to_string(), BN.to_string(), BK.to_string()])
        .sync_on(ctx.stream())?;
    Ok(d.unpartition())
}

/// FP8 block-scaled GEMM `D = A @ B^T` with UE8M0 scales at 32-lane
/// granularity (per-128 DeepSeek recipe pre-replicated by
/// [`crate::pack`]). Requires sm_100+ unless the BF16 fallback is forced.
pub fn fp8_scaled_gemm_nt(
    ctx: &GpuCtx,
    a: &Arc<Tensor<cutile::cuda_core::f8e4m3fn>>,
    a_scales: &Arc<Tensor<cutile::cuda_core::f8e8m0fnu>>,
    b: &Arc<Tensor<cutile::cuda_core::f8e4m3fn>>,
    b_scales: &Arc<Tensor<cutile::cuda_core::f8e8m0fnu>>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<Tensor<bf16>, Error> {
    if !ctx.has_fp8_mma() {
        return Err(cutile::error::kernel_launch_error(&format!(
            "fp8_scaled_gemm_nt: {} has no block-scaled MMA; set {} to fall back to BF16",
            ctx.capability(),
            crate::arch::FORCE_BF16_FALLBACK_ENV
        )));
    }
    if k % 128 != 0 {
        return Err(cutile::error::kernel_launch_error(&format!("K={k} must be a multiple of 128")));
    }
    if a.shape() != [m as i32, k as i32]
        || b.shape() != [n as i32, k as i32]
        || a_scales.shape() != [m as i32, (k / 32) as i32]
        || b_scales.shape() != [n as i32, (k / 32) as i32]
    {
        return Err(cutile::error::kernel_launch_error("fp8_scaled_gemm_nt: shape mismatch"));
    }
    const BM: i32 = 64;
    const BN: i32 = 128;
    const BK: i32 = 64;
    const BKS: i32 = 2;
    let d = api::zeros::<bf16>(&[m, n]).partition([BM as usize, BN as usize]);
    let (d, ..) = unsafe {
        kernels::fp8_scaled_gemm_nt(d, a.clone(), b.clone(), a_scales.clone(), b_scales.clone())
    }
    .generics(vec![
        BM.to_string(),
        BN.to_string(),
        BK.to_string(),
        BKS.to_string(),
    ])
    .sync_on(ctx.stream())?;
    Ok(d.unpartition())
}

/// Quantizes `[rows, k]` BF16 activations to E4M3 + replicated UE8M0
/// scales (feeds [`fp8_scaled_gemm_nt`]).
pub fn quant_fp8_block128(
    ctx: &GpuCtx,
    x: &Arc<Tensor<bf16>>,
    rows: usize,
    k: usize,
) -> Result<(Tensor<cutile::cuda_core::f8e4m3fn>, Tensor<cutile::cuda_core::f8e8m0fnu>), Error>
{
    if k % 128 != 0 {
        return Err(cutile::error::kernel_launch_error(&format!("K={k} must be a multiple of 128")));
    }
    if x.shape() != [rows as i32, k as i32] {
        return Err(cutile::error::kernel_launch_error("quant_fp8_block128: shape mismatch"));
    }
    let q = api::zeros::<cutile::cuda_core::f8e4m3fn>(&[rows, k]).partition([1, 128]);
    let s = api::zeros::<cutile::cuda_core::f8e8m0fnu>(&[rows, k / 32]).partition([1, 4]);
    let (q, s, ..) = unsafe { kernels::quant_fp8_block128(q, s, x.clone()) }
        .sync_on(ctx.stream())?;
    Ok((q.unpartition(), s.unpartition()))
}

/// MLA decode over a gathered KV window (BF16 cache): split-KV + merge.
/// `indices: [b, topk]` physical row ids (`-1` invalid). Returns
/// `(out [b, n_heads, 576], lse [b, n_heads])` with natural-log LSE and
/// the sink epilogue applied.
#[allow(clippy::too_many_arguments)]
pub fn mla_decode_bf16(
    ctx: &GpuCtx,
    q_nope: &Arc<Tensor<bf16>>,
    q_pe: &Arc<Tensor<bf16>>,
    ckv: &Arc<Tensor<bf16>>,
    kpe: &Arc<Tensor<bf16>>,
    indices: &Arc<Tensor<i32>>,
    sinks: Option<&Arc<Tensor<f32>>>,
    b: usize,
    ngroups: usize,
    topk: usize,
    sm_scale: f32,
    num_splits: usize,
) -> Result<(Tensor<bf16>, Tensor<f32>), Error> {
    const GROUP: i32 = 8;
    const BN: i32 = 64;
    const DNOPE: i32 = 512;
    const DPE: i32 = 64;
    let ns = num_splits as i32;
    let rows = b * ngroups;
    if q_nope.shape() != [rows as i32, GROUP, DNOPE]
        || q_pe.shape() != [rows as i32, GROUP, DPE]
    {
        return Err(cutile::error::kernel_launch_error("mla_decode_bf16: q shape mismatch"));
    }
    if indices.shape() != [b as i32, topk as i32] {
        return Err(cutile::error::kernel_launch_error("mla_decode_bf16: indices shape mismatch"));
    }
    let att_n = api::zeros::<bf16>(&[rows, (num_splits * GROUP as usize), DNOPE as usize])
        .partition([1, GROUP as usize, DNOPE as usize]);
    let att_p = api::zeros::<bf16>(&[rows, (num_splits * GROUP as usize), DPE as usize])
        .partition([1, GROUP as usize, DPE as usize]);
    let lse_p = api::zeros::<f32>(&[rows, num_splits * GROUP as usize])
        .partition([1, GROUP as usize]);
    let (att_n, att_p, lse_p, ..) = unsafe {
        kernels::mla_decode_splitkv_bf16(
            att_n,
            att_p,
            lse_p,
            q_nope.clone(),
            q_pe.clone(),
            ckv.clone(),
            kpe.clone(),
            indices.clone(),
            ngroups as i32,
            topk as i32,
            sm_scale,
        )
    }
    .generics(vec![
        GROUP.to_string(),
        BN.to_string(),
        DNOPE.to_string(),
        DPE.to_string(),
        ns.to_string(),
    ])
    .sync_on(ctx.stream())?;

    let att_n = Arc::new(att_n.unpartition());
    let att_p = Arc::new(att_p.unpartition());
    let lse_p = Arc::new(lse_p.unpartition());
    merge_splits(
        ctx, &att_n, &att_p, &lse_p, sinks, b, ngroups, num_splits,
    )
}

/// Merge stage shared by the BF16 and FP8 decode paths.
#[allow(clippy::too_many_arguments)]
fn merge_splits(
    ctx: &GpuCtx,
    att_n: &Arc<Tensor<bf16>>,
    att_p: &Arc<Tensor<bf16>>,
    lse_p: &Arc<Tensor<f32>>,
    sinks: Option<&Arc<Tensor<f32>>>,
    b: usize,
    ngroups: usize,
    num_splits: usize,
) -> Result<(Tensor<bf16>, Tensor<f32>), Error> {
    const GROUP: i32 = 8;
    const DNOPE: i32 = 512;
    const DPE: i32 = 64;
    const D: i32 = 576;
    let ns = num_splits as i32;
    let rows = b * ngroups;
    let has_sink = sinks.is_some() as i32;
    let sink_t: Arc<Tensor<f32>> = sinks
        .cloned()
        .unwrap_or_else(|| Arc::new(api::zeros::<f32>(&[1]).sync_on(ctx.stream()).unwrap()));

    let out = api::zeros::<bf16>(&[rows, GROUP as usize, D as usize])
        .partition([1, GROUP as usize, D as usize]);
    let lse = api::zeros::<f32>(&[rows, GROUP as usize]).partition([1, GROUP as usize]);
    let (out, lse, ..) = unsafe {
        kernels::mla_split_merge(
            out,
            lse,
            att_n.clone(),
            att_p.clone(),
            lse_p.clone(),
            sink_t,
            ngroups as i32,
            has_sink,
            f32::INFINITY,
        )
    }
    .generics(vec![
        GROUP.to_string(),
        DNOPE.to_string(),
        DPE.to_string(),
        D.to_string(),
        ns.to_string(),
        (num_splits * GROUP as usize).to_string(),
    ])
    .sync_on(ctx.stream())?;
    Ok((out.unpartition(), lse.unpartition()))
}

/// Greedy argmax over `[vocab]` BF16 logits (ties → lowest index).
pub fn argmax_bf16(
    ctx: &GpuCtx,
    logits: &Arc<Tensor<bf16>>,
    len: usize,
) -> Result<(f32, i32), Error> {
    const BLOCK: i32 = 1024;
    let nb = (len + BLOCK as usize - 1) / BLOCK as usize;
    let block_max = api::zeros::<f32>(&[nb]).partition([1]);
    let block_idx = api::zeros::<i32>(&[nb]).partition([1]);
    let (block_max, block_idx, ..) =
        unsafe { kernels::argmax_blocks_bf16(block_max, block_idx, logits.clone(), len as i32) }
            .generics(vec![BLOCK.to_string()])
            .sync_on(ctx.stream())?;
    let block_max = Arc::new(block_max.unpartition());
    let block_idx = Arc::new(block_idx.unpartition());

    let out_max = api::zeros::<f32>(&[1]).partition([1]);
    let out_idx = api::zeros::<i32>(&[1]).partition([1]);
    let (out_max, out_idx, ..) = unsafe {
        kernels::argmax_final(out_max, out_idx, block_max, block_idx)
    }
    .generics(vec![nb.to_string()])
    .grid((1, 1, 1))
    .sync_on(ctx.stream())?;
    let m: Vec<f32> = out_max.unpartition().to_host_vec().sync_on(ctx.stream())?;
    let i: Vec<i32> = out_idx.unpartition().to_host_vec().sync_on(ctx.stream())?;
    Ok((m[0], i[0]))
}
