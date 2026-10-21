//! Paged KV-cache append kernels for the three V4.1 cache formats.
//!
//! Physical row of token `p` (page size 64): `page_id = block_table[b,
//! p / 64]`, `row = page_id * 64 + p % 64`. Positions are per-sequence
//! (`position_starts: [b]`). All loads/stores use computed (runtime)
//! indices, so hosts launch with an explicit `.grid((b * s, groups, 1))`.

/// BF16 cache append (no quantization): one CTA per (token, tensor-pair).
#[cutile::module]
pub mod kv_append_bf16_module {
    use cutile::core::*;

    /// Copies `new_ckv[bs, :]` / `new_kpe[bs, :]` (bs = `b * s + s`) into
    /// the flat cache tensors at the paged rows for positions
    /// `position_starts[b] + s`. Grid `(b * s, 1, 1)`.
    #[cutile::entry(print_ir = false,
                    unchecked_accesses = true,
                    optimization_hints = (
                      sm_100 = (max_divisibility = 8,),
                      sm_103 = (max_divisibility = 8,),
                    ))]
    unsafe fn kv_append_bf16<const PAGE: i32, const DNOPE: i32, const DPE: i32>(
        ckv_cache: &mut Tensor<bf16, { [1, DNOPE] }>,
        kpe_cache: &mut Tensor<bf16, { [1, DPE] }>,
        new_ckv: &Tensor<bf16, { [-1, DNOPE] }>,
        new_kpe: &Tensor<bf16, { [-1, DPE] }>,
        block_table: &Tensor<i32, { [-1, -1] }>,
        position_starts: &Tensor<i32, { [-1] }>,
        seq_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let bs: i32 = pid.0;
        let b: i32 = bs / seq_len;
        let s: i32 = bs % seq_len;
        let pos0: i32 = tile_to_scalar(
            position_starts
                .partition(const_shape![1])
                .load([b])
                .reshape(const_shape![]),
        );
        let pos: i32 = pos0 + s;
        let page: i32 = tile_to_scalar(
            block_table
                .partition(const_shape![1, 1])
                .load([b, pos / PAGE])
                .reshape(const_shape![]),
        );
        let row: i32 = page * PAGE + pos % PAGE;

        let row_ckv: Tile<bf16, { [1, DNOPE] }> = new_ckv
            .partition(const_shape![1, DNOPE])
            .load([bs, 0i32]);
        let row_kpe: Tile<bf16, { [1, DPE] }> =
            new_kpe.partition(const_shape![1, DPE]).load([bs, 0i32]);

        let mut cache_n: PartitionMut<bf16, { [1, DNOPE] }> =
            ckv_cache.partition_mut(const_shape![1, DNOPE]);
        let mut cache_p: PartitionMut<bf16, { [1, DPE] }> =
            kpe_cache.partition_mut(const_shape![1, DPE]);
        cache_n.store(row_ckv, [row, 0i32]);
        cache_p.store(row_kpe, [row, 0i32]);
    }
}

pub use kv_append_bf16_module::kv_append_bf16;

/// FP8 cache append: one CTA per (token, 32-lane group), quantizing
/// in-register with the exact CPU scale recipe.
#[cutile::module]
pub mod kv_append_fp8_module {
    use crate::kernels::helpers::scale_helpers;
    use cutile::core::*;

    /// Quantizes one 32-lane group of the new rows to E4M3 + UE8M0
    /// (`2^ceil(log2(amax/448))`) and writes it at the paged row.
    /// `pid.1 < GNOPE` selects the latent slice, otherwise the rope slice.
    /// `DNOPE`/`DPE` must be multiples of 32. Grid
    /// `(b * s, GNOPE + GPE, 1)`.
    #[cutile::entry(print_ir = false,
                    unchecked_accesses = true,
                    optimization_hints = (
                      sm_100 = (max_divisibility = 8,),
                      sm_103 = (max_divisibility = 8,),
                    ))]
    unsafe fn kv_append_fp8<
        const PAGE: i32,
        const DNOPE: i32,
        const DPE: i32,
        const GNOPE: i32,
        const GPE: i32,
    >(
        ckv_cache: &mut Tensor<f8e4m3fn, { [1, 32] }>,
        kpe_cache: &mut Tensor<f8e4m3fn, { [1, 32] }>,
        ckv_scales: &mut Tensor<f8e8m0fnu, { [1, 1] }>,
        kpe_scales: &mut Tensor<f8e8m0fnu, { [1, 1] }>,
        new_ckv: &Tensor<bf16, { [-1, DNOPE] }>,
        new_kpe: &Tensor<bf16, { [-1, DPE] }>,
        block_table: &Tensor<i32, { [-1, -1] }>,
        position_starts: &Tensor<i32, { [-1] }>,
        seq_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let bs: i32 = pid.0;
        let g: i32 = pid.1;
        let b: i32 = bs / seq_len;
        let s: i32 = bs % seq_len;
        let pos0: i32 = tile_to_scalar(
            position_starts
                .partition(const_shape![1])
                .load([b])
                .reshape(const_shape![]),
        );
        let pos: i32 = pos0 + s;
        let page: i32 = tile_to_scalar(
            block_table
                .partition(const_shape![1, 1])
                .load([b, pos / PAGE])
                .reshape(const_shape![]),
        );
        let row: i32 = page * PAGE + pos % PAGE;

        if g < GNOPE {
            let x: Tile<bf16, { [1, 32] }> = new_ckv
                .partition(const_shape![1, 32])
                .load([bs, g]);
            let xf: Tile<f32, { [1, 32] }> = convert_tile(x);
            let amax: Tile<f32, { [1] }> = reduce_max(absf(xf), 1i32);
            let amax0: Tile<f32, { [] }> = amax.reshape(const_shape![]);
            let scale: Tile<f32, { [] }> = scale_helpers::e8m0_value_from_amax(amax0);
            let byte: Tile<f8e8m0fnu, { [] }> = scale_helpers::e8m0_byte_from_amax(amax0);
            let scale_s: f32 = tile_to_scalar(scale);
            let inv: Tile<f32, { [1, 32] }> =
                broadcast_scalar(1.0f32 / scale_s, const_shape![1, 32]);
            let mut cache: PartitionMut<f8e4m3fn, { [1, 32] }> =
                ckv_cache.partition_mut(const_shape![1, 32]);
            let mut scales: PartitionMut<f8e8m0fnu, { [1, 1] }> =
                ckv_scales.partition_mut(const_shape![1, 1]);
            cache.store(convert_tile(xf * inv), [row, g]);
            scales.store(byte.reshape(const_shape![1, 1]), [row, g]);
        } else {
            let grp: i32 = g - GNOPE;
            let x: Tile<bf16, { [1, 32] }> =
                new_kpe.partition(const_shape![1, 32]).load([bs, grp]);
            let xf: Tile<f32, { [1, 32] }> = convert_tile(x);
            let amax: Tile<f32, { [1] }> = reduce_max(absf(xf), 1i32);
            let amax0: Tile<f32, { [] }> = amax.reshape(const_shape![]);
            let scale: Tile<f32, { [] }> = scale_helpers::e8m0_value_from_amax(amax0);
            let byte: Tile<f8e8m0fnu, { [] }> = scale_helpers::e8m0_byte_from_amax(amax0);
            let scale_s: f32 = tile_to_scalar(scale);
            let inv: Tile<f32, { [1, 32] }> =
                broadcast_scalar(1.0f32 / scale_s, const_shape![1, 32]);
            let mut cache: PartitionMut<f8e4m3fn, { [1, 32] }> =
                kpe_cache.partition_mut(const_shape![1, 32]);
            let mut scales: PartitionMut<f8e8m0fnu, { [1, 1] }> =
                kpe_scales.partition_mut(const_shape![1, 1]);
            cache.store(convert_tile(xf * inv), [row, grp]);
            scales.store(byte.reshape(const_shape![1, 1]), [row, grp]);
        }
    }
}

pub use kv_append_fp8_module::kv_append_fp8;

/// FP4 cache append: one CTA per (token, 16-lane group), E2M1 packed
/// (even value in the low nibble) + E4M3 scales (`amax/6`, floored at
/// 2^-6).
#[cutile::module]
pub mod kv_append_fp4_module {
    use crate::kernels::helpers::scale_helpers;
    use cutile::core::*;

    /// Quantizes one 16-lane group to packed E2M1 + one E4M3 scale byte
    /// and writes it at the paged row. `pid.1 < GNOPE` selects the latent
    /// slice. `DNOPE`/`DPE` must be multiples of 16. Grid
    /// `(b * s, GNOPE + GPE, 1)` with 16-lane groups.
    #[cutile::entry(print_ir = false,
                    unchecked_accesses = true,
                    optimization_hints = (
                      sm_100 = (max_divisibility = 8,),
                      sm_103 = (max_divisibility = 8,),
                    ))]
    unsafe fn kv_append_fp4<
        const PAGE: i32,
        const DNOPE: i32,
        const DPE: i32,
        const GNOPE: i32,
        const GPE: i32,
    >(
        ckv_cache: &mut Tensor<f4e2m1fnx2, { [1, 8] }>,
        kpe_cache: &mut Tensor<f4e2m1fnx2, { [1, 8] }>,
        ckv_scales: &mut Tensor<f8e4m3fn, { [1, 1] }>,
        kpe_scales: &mut Tensor<f8e4m3fn, { [1, 1] }>,
        new_ckv: &Tensor<bf16, { [-1, DNOPE] }>,
        new_kpe: &Tensor<bf16, { [-1, DPE] }>,
        block_table: &Tensor<i32, { [-1, -1] }>,
        position_starts: &Tensor<i32, { [-1] }>,
        seq_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let bs: i32 = pid.0;
        let g: i32 = pid.1;
        let b: i32 = bs / seq_len;
        let s: i32 = bs % seq_len;
        let pos0: i32 = tile_to_scalar(
            position_starts
                .partition(const_shape![1])
                .load([b])
                .reshape(const_shape![]),
        );
        let pos: i32 = pos0 + s;
        let page: i32 = tile_to_scalar(
            block_table
                .partition(const_shape![1, 1])
                .load([b, pos / PAGE])
                .reshape(const_shape![]),
        );
        let row: i32 = page * PAGE + pos % PAGE;

        if g < GNOPE {
            let x: Tile<bf16, { [1, 16] }> = new_ckv
                .partition(const_shape![1, 16])
                .load([bs, g]);
            let xf: Tile<f32, { [1, 16] }> = convert_tile(x);
            let amax: Tile<f32, { [1] }> = reduce_max(absf(xf), 1i32);
            let amax0: Tile<f32, { [] }> = amax.reshape(const_shape![]);
            let scale: Tile<f32, { [] }> = scale_helpers::fp4_scale_from_amax(amax0);
            let scale_s: f32 = tile_to_scalar(scale);
            let inv: Tile<f32, { [1, 16] }> =
                broadcast_scalar(1.0f32 / scale_s, const_shape![1, 16]);
            let nibs: Tile<f4e2m1fn, { [16] }> =
                convert_tile((xf * inv).reshape(const_shape![16]));
            let packed: Tile<f4e2m1fnx2, { [1, 8] }> = nibs.pack(const_shape![1, 8]);
            let mut cache: PartitionMut<f4e2m1fnx2, { [1, 8] }> =
                ckv_cache.partition_mut(const_shape![1, 8]);
            let mut scales: PartitionMut<f8e4m3fn, { [1, 1] }> =
                ckv_scales.partition_mut(const_shape![1, 1]);
            cache.store(packed, [row, g]);
            let s_f8: Tile<f8e4m3fn, { [] }> = convert_tile(scale);
            scales.store(s_f8.reshape(const_shape![1, 1]), [row, g]);
        } else {
            let grp: i32 = g - GNOPE;
            let x: Tile<bf16, { [1, 16] }> =
                new_kpe.partition(const_shape![1, 16]).load([bs, grp]);
            let xf: Tile<f32, { [1, 16] }> = convert_tile(x);
            let amax: Tile<f32, { [1] }> = reduce_max(absf(xf), 1i32);
            let amax0: Tile<f32, { [] }> = amax.reshape(const_shape![]);
            let scale: Tile<f32, { [] }> = scale_helpers::fp4_scale_from_amax(amax0);
            let scale_s: f32 = tile_to_scalar(scale);
            let inv: Tile<f32, { [1, 16] }> =
                broadcast_scalar(1.0f32 / scale_s, const_shape![1, 16]);
            let nibs: Tile<f4e2m1fn, { [16] }> =
                convert_tile((xf * inv).reshape(const_shape![16]));
            let packed: Tile<f4e2m1fnx2, { [1, 8] }> = nibs.pack(const_shape![1, 8]);
            let mut cache: PartitionMut<f4e2m1fnx2, { [1, 8] }> =
                kpe_cache.partition_mut(const_shape![1, 8]);
            let mut scales: PartitionMut<f8e4m3fn, { [1, 1] }> =
                kpe_scales.partition_mut(const_shape![1, 1]);
            cache.store(packed, [row, grp]);
            let s_f8: Tile<f8e4m3fn, { [] }> = convert_tile(scale);
            scales.store(s_f8.reshape(const_shape![1, 1]), [row, grp]);
        }
    }
}

pub use kv_append_fp4_module::kv_append_fp4;
