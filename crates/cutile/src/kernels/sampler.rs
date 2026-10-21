//! Sampling kernels: two-stage exact argmax with ties → lowest index,
//! mirroring `deepinfer-sampler`'s greedy semantics.

/// Stage 1: per-block argmax over BF16 logits.
#[cutile::module]
pub mod argmax_blocks_bf16_module {
    use cutile::core::*;

    /// Per CTA (block of `BLOCK_SIZE` logits): writes the block max and the
    /// lowest matching index. Out-of-range lanes score `-inf`.
    #[cutile::entry(print_ir = false,
                    unchecked_accesses = true,
                    optimization_hints = (
                      sm_100 = (max_divisibility = 8,),
                      sm_103 = (max_divisibility = 8,),
                    ))]
    unsafe fn argmax_blocks_bf16<const BLOCK_SIZE: i32>(
        block_max: &mut Tensor<f32, { [1] }>,
        block_idx: &mut Tensor<i32, { [1] }>,
        logits: &Tensor<bf16, { [-1] }>,
        len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let block = pid.0;

        let lg_bf: Tile<bf16, { [BLOCK_SIZE] }> = logits
            .partition(const_shape![BLOCK_SIZE])
            .load([block]);
        let lg: Tile<f32, { [BLOCK_SIZE] }> = convert_tile(lg_bf);

        let base: i32 = block * BLOCK_SIZE;
        let indices: Tile<i32, { [BLOCK_SIZE] }> =
            iota(const_shape![BLOCK_SIZE]) + broadcast_scalar(base, const_shape![BLOCK_SIZE]);
        let valid: Tile<bool, { [BLOCK_SIZE] }> =
            lt_tile(indices, broadcast_scalar(len, const_shape![BLOCK_SIZE]));
        let neg_inf: Tile<f32, { [BLOCK_SIZE] }> =
            constant(0.0f32, const_shape![BLOCK_SIZE])
                - constant(1.0e30f32, const_shape![BLOCK_SIZE]);
        let masked: Tile<f32, { [BLOCK_SIZE] }> = select(valid, lg, neg_inf);

        let m: Tile<f32, { [1] }> = reduce_max(masked, 0i32);
        let m_s: f32 = tile_to_scalar(m.reshape(const_shape![]));
        let is_max: Tile<bool, { [BLOCK_SIZE] }> =
            eq_tile(masked, broadcast_scalar(m_s, const_shape![BLOCK_SIZE]));
        let int_max: Tile<i32, { [BLOCK_SIZE] }> =
            broadcast_scalar(2147483647i32, const_shape![BLOCK_SIZE]);
        let cand: Tile<i32, { [BLOCK_SIZE] }> = select(is_max, indices, int_max);
        let idx: Tile<i32, { [1] }> = reduce_min(cand, 0i32);
        let idx_s: i32 = tile_to_scalar(idx.reshape(const_shape![]));

        block_max.store(scalar_to_tile(m_s).reshape(const_shape![1]));
        block_idx.store(scalar_to_tile(idx_s).reshape(const_shape![1]));
    }
}

pub use argmax_blocks_bf16_module::argmax_blocks_bf16;

/// Stage 2: final argmax over per-block results (single CTA).
#[cutile::module]
pub mod argmax_final_module {
    use cutile::core::*;

    /// Reduces `NB` `(block_max, block_idx)` pairs into the global argmax;
    /// ties resolve to the lowest index. Launch grid `(1, 1, 1)`.
    #[cutile::entry(print_ir = false,
                    unchecked_accesses = true,
                    optimization_hints = (
                      sm_100 = (max_divisibility = 8,),
                      sm_103 = (max_divisibility = 8,),
                    ))]
    unsafe fn argmax_final<const NB: i32>(
        out_max: &mut Tensor<f32, { [1] }>,
        out_idx: &mut Tensor<i32, { [1] }>,
        block_max: &Tensor<f32, { [-1] }>,
        block_idx: &Tensor<i32, { [-1] }>,
    ) {
        let m: Tile<f32, { [NB] }> = block_max.partition(const_shape![NB]).load([0i32]);
        let idx: Tile<i32, { [NB] }> = block_idx.partition(const_shape![NB]).load([0i32]);

        let gmax: Tile<f32, { [1] }> = reduce_max(m, 0i32);
        let gmax_s: f32 = tile_to_scalar(gmax.reshape(const_shape![]));
        let is_max: Tile<bool, { [NB] }> =
            eq_tile(m, broadcast_scalar(gmax_s, const_shape![NB]));
        let int_max: Tile<i32, { [NB] }> = broadcast_scalar(2147483647i32, const_shape![NB]);
        let cand: Tile<i32, { [NB] }> = select(is_max, idx, int_max);
        let gidx: Tile<i32, { [1] }> = reduce_min(cand, 0i32);
        let gidx_s: i32 = tile_to_scalar(gidx.reshape(const_shape![]));

        out_max.store(scalar_to_tile(gmax_s).reshape(const_shape![1]));
        out_idx.store(scalar_to_tile(gidx_s).reshape(const_shape![1]));
    }
}

pub use argmax_final_module::argmax_final;
