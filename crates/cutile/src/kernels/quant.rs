//! Quantization kernels: DeepGEMM's 1×128 activation recipe for
//! block-scaled MMA, and the V4.1 KV-cache formats (per-32 UE8M0 FP8,
//! per-16 E4M3 FP4).

/// BF16 → FP8 E4M3 with per-1×128 UE8M0 scales, emitted at 32-lane
/// granularity (each 128-block's scale byte replicated ×4) so the output
/// feeds `mmaf_scaled` directly.
#[cutile::module]
pub mod quant_fp8_block128_module {
    use crate::kernels::helpers::scale_helpers;
    use cutile::core::*;

    /// Quantizes one `[1, 128]` block: scale `2^ceil(log2(amax/448))`,
    /// `data = e4m3(x / scale)`, scale byte stored into four consecutive
    /// 32-lane slots.
    #[cutile::entry(print_ir = false,
                    unchecked_accesses = true,
                    optimization_hints = (
                      sm_100 = (max_divisibility = 16,),
                      sm_103 = (max_divisibility = 16,),
                    ))]
    unsafe fn quant_fp8_block128(
        q: &mut Tensor<f8e4m3fn, { [1, 128] }>,
        scales: &mut Tensor<f8e8m0fnu, { [1, 4] }>,
        x: &Tensor<bf16, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;
        let kblock = pid.1;

        let x_part: Partition<bf16, { [1, 128] }> = x.partition(const_shape![1, 128]);
        let xb: Tile<bf16, { [1, 128] }> = x_part.load([row, kblock]);
        let xf: Tile<f32, { [1, 128] }> = convert_tile(xb);

        let amax_1: Tile<f32, { [1] }> = reduce_max(absf(xf), 1i32);
        let amax: Tile<f32, { [] }> = amax_1.reshape(const_shape![]);

        let scale: Tile<f32, { [] }> = scale_helpers::e8m0_value_from_amax(amax);
        let byte: Tile<f8e8m0fnu, { [] }> = scale_helpers::e8m0_byte_from_amax(amax);

        let scale_s: f32 = tile_to_scalar(scale);
        let inv: Tile<f32, { [1, 128] }> =
            broadcast_scalar(1.0f32 / scale_s, const_shape![1, 128]);
        let data: Tile<f32, { [1, 128] }> = xf * inv;
        q.store(convert_tile(data));
        let byte_s: f8e8m0fnu = tile_to_scalar(byte);
        scales.store(byte_s.broadcast(const_shape![1, 4]));
    }
}

pub use quant_fp8_block128_module::quant_fp8_block128;

/// BF16 → the V4.1 FP8 KV-cache layout: E4M3 data with one UE8M0 scale per
/// 32 dims (`2^ceil(log2(amax/448))`, byte 0 for all-zero groups).
#[cutile::module]
pub mod quant_fp8_group32_module {
    use crate::kernels::helpers::scale_helpers;
    use cutile::core::*;

    /// Quantizes `[tokens, D]` into `q[tokens, D]` E4M3 plus
    /// `scales[tokens, D/32]` UE8M0, one CTA tile per (token, group).
    #[cutile::entry(print_ir = false,
                    unchecked_accesses = true,
                    optimization_hints = (
                      sm_100 = (max_divisibility = 8,),
                      sm_103 = (max_divisibility = 8,),
                    ))]
    unsafe fn quant_fp8_group32<const D: i32>(
        q: &mut Tensor<f8e4m3fn, { [1, 32] }>,
        scales: &mut Tensor<f8e8m0fnu, { [1, 1] }>,
        x: &Tensor<bf16, { [-1, D] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;
        let group = pid.1;

        let x_part: Partition<bf16, { [1, 32] }> = x.partition(const_shape![1, 32]);
        let xb: Tile<bf16, { [1, 32] }> = x_part.load([row, group]);
        let xf: Tile<f32, { [1, 32] }> = convert_tile(xb);

        let amax_1: Tile<f32, { [1] }> = reduce_max(absf(xf), 1i32);
        let amax: Tile<f32, { [] }> = amax_1.reshape(const_shape![]);

        let scale: Tile<f32, { [] }> = scale_helpers::e8m0_value_from_amax(amax);
        let byte: Tile<f8e8m0fnu, { [] }> = scale_helpers::e8m0_byte_from_amax(amax);

        let scale_s: f32 = tile_to_scalar(scale);
        let inv: Tile<f32, { [1, 32] }> = broadcast_scalar(1.0f32 / scale_s, const_shape![1, 32]);
        q.store(convert_tile(xf * inv));
        scales.store(byte.reshape(const_shape![1, 1]));
    }
}

pub use quant_fp8_group32_module::quant_fp8_group32;

/// BF16 → the V4.1 FP4 KV-cache layout: packed E2M1 (two values per byte,
/// even index in the low nibble) with one E4M3 scale per 16 dims
/// (`amax/6` rounded to E4M3, floored at the smallest positive normal).
#[cutile::module]
pub mod quant_fp4_group16_module {
    use crate::kernels::helpers::scale_helpers;
    use cutile::core::*;

    /// Quantizes `[tokens, D]` into packed `q[tokens, D/2]` E2M1 plus
    /// `scales[tokens, D/16]` E4M3.
    #[cutile::entry(print_ir = false,
                    unchecked_accesses = true,
                    optimization_hints = (
                      sm_100 = (max_divisibility = 8,),
                      sm_103 = (max_divisibility = 8,),
                    ))]
    unsafe fn quant_fp4_group16<const D: i32>(
        q: &mut Tensor<f4e2m1fnx2, { [1, 8] }>,
        scales: &mut Tensor<f8e4m3fn, { [1, 1] }>,
        x: &Tensor<bf16, { [-1, D] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;
        let group = pid.1;

        let x_part: Partition<bf16, { [1, 16] }> = x.partition(const_shape![1, 16]);
        let xb: Tile<bf16, { [1, 16] }> = x_part.load([row, group]);
        let xf: Tile<f32, { [1, 16] }> = convert_tile(xb);

        let amax_1: Tile<f32, { [1] }> = reduce_max(absf(xf), 1i32);
        let amax: Tile<f32, { [] }> = amax_1.reshape(const_shape![]);

        let scale: Tile<f32, { [] }> = scale_helpers::fp4_scale_from_amax(amax);
        let scale_s: f32 = tile_to_scalar(scale);
        let inv: Tile<f32, { [1, 16] }> = broadcast_scalar(1.0f32 / scale_s, const_shape![1, 16]);
        let data: Tile<f32, { [1, 16] }> = xf * inv;

        let nibs: Tile<f4e2m1fn, { [16] }> = convert_tile(data.reshape(const_shape![16]));
        let packed: Tile<f4e2m1fnx2, { [1, 8] }> = nibs.pack(const_shape![1, 8]);
        q.store(packed);

        let s_f8: Tile<f8e4m3fn, { [] }> = convert_tile(scale);
        scales.store(s_f8.reshape(const_shape![1, 1]));
    }
}

pub use quant_fp4_group16_module::quant_fp4_group16;

/// FP8 KV-cache → BF16 (dequantization for verification and fallback paths).
#[cutile::module]
pub mod dequant_fp8_group32_module {
    use cutile::core::*;

    /// Multiplies E4M3 data by its per-32 UE8M0 scale value, emitting BF16
    /// rows of `[tokens, D]`.
    #[cutile::entry(print_ir = false,
                    unchecked_accesses = true,
                    optimization_hints = (
                      sm_100 = (max_divisibility = 8,),
                      sm_103 = (max_divisibility = 8,),
                    ))]
    unsafe fn dequant_fp8_group32<const D: i32>(
        out: &mut Tensor<bf16, { [1, 32] }>,
        q: &Tensor<f8e4m3fn, { [-1, D] }>,
        scales: &Tensor<f8e8m0fnu, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;
        let group = pid.1;

        let q_part: Partition<f8e4m3fn, { [1, 32] }> = q.partition(const_shape![1, 32]);
        let s_part: Partition<f8e8m0fnu, { [1, 1] }> = scales.partition(const_shape![1, 1]);
        let qb: Tile<f8e4m3fn, { [1, 32] }> = q_part.load([row, group]);
        let sb: Tile<f8e8m0fnu, { [1, 1] }> = s_part.load([row, group]);

        // E8M0 scale value = 2^(byte - 127).
        let sb_u8: Tile<u8, { [1, 1] }> = bitcast(sb);
        let sb_i: Tile<i32, { [1, 1] }> = exti(sb_u8);
        let biased: Tile<f32, { [1, 1] }> = convert_tile(sb_i);
        let exp_t: Tile<f32, { [1, 1] }> =
            biased - broadcast_scalar(127.0f32, const_shape![1, 1]);
        let scale_s: f32 = tile_to_scalar(exp2(exp_t.reshape(const_shape![]), ftz::Disabled));

        let qf: Tile<f32, { [1, 32] }> = convert_tile(qb);
        let s_b: Tile<f32, { [1, 32] }> = broadcast_scalar(scale_s, const_shape![1, 32]);
        out.store(convert_tile(qf * s_b));
    }
}

pub use dequant_fp8_group32_module::dequant_fp8_group32;
