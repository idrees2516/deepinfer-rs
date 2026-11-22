//! RMSNorm kernels (DeepSeek V4.1 uses RMSNorm with eps 1e-4 on BF16
//! activations; all accumulation happens in F32).

/// Plain RMSNorm over the last dimension of `[rows, N]` BF16.
#[cutile::module]
pub mod rms_norm_bf16_module {
    use cutile::core::*;

    /// `out = x / rms(x) * w` row-wise, two passes over `[1, BLOCK_SIZE]`
    /// tiles with an F32 accumulator (matches `deepinfer_core::norm`).
    #[cutile::entry(print_ir = false,
                    unchecked_accesses = true,
                    optimization_hints = (
                      sm_100 = (max_divisibility = 8,),
                      sm_103 = (max_divisibility = 8,),
                    ))]
    unsafe fn rms_norm_bf16<const N: i32, const BLOCK_SIZE: i32>(
        x: &Tensor<bf16, { [-1, N] }>,
        w: &Tensor<bf16, { [N] }>,
        out: &mut Tensor<bf16, { [1, N] }>,
        eps: f32,
    ) {
        let tile_shape: Shape<{ [1, BLOCK_SIZE] }> = const_shape![1, BLOCK_SIZE];
        let num_tiles: i32 = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;

        let x_part: Partition<bf16, { [1, BLOCK_SIZE] }> = x.partition(tile_shape);
        let mut rms: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0f32, tile_shape);
        for j in 0i32..num_tiles {
            let tx_bf: Tile<bf16, { [1, BLOCK_SIZE] }> = x_part.load([row, j]);
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tx_bf);
            rms = rms + tx * tx;
        }
        let rms: Tile<f32, { [1] }> = reduce_sum(rms, 1i32);
        let rms: Tile<f32, { [] }> = rms.reshape(const_shape![]);
        let rms: f32 = tile_to_scalar(rms);
        let n: f32 = convert_scalar(N);
        let inv_rms: f32 = rms / n + eps;
        let inv_rms: Tile<f32, { [] }> = rsqrt(scalar_to_tile(inv_rms), ftz::Disabled);
        let inv_rms: f32 = tile_to_scalar(inv_rms);
        let inv_rms: Tile<f32, { [1, BLOCK_SIZE] }> = inv_rms.broadcast(tile_shape);

        let w_part: Partition<bf16, { [BLOCK_SIZE] }> = w.partition(const_shape![BLOCK_SIZE]);
        let mut out_part: PartitionMut<bf16, { [1, BLOCK_SIZE] }> = out.partition_mut(tile_shape);
        for j in 0i32..num_tiles {
            let tx_bf: Tile<bf16, { [1, BLOCK_SIZE] }> = x_part.load([row, j]);
            let tw_bf: Tile<bf16, { [BLOCK_SIZE] }> = w_part.load([j]);
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tx_bf);
            let tw: Tile<f32, { [1, BLOCK_SIZE] }> =
                convert_tile(tw_bf.reshape(const_shape![1, BLOCK_SIZE]));
            let tout: Tile<f32, { [1, BLOCK_SIZE] }> = tx * inv_rms * tw;
            out_part.store(convert_tile(tout), [0i32, j]);
        }
    }
}

pub use rms_norm_bf16_module::rms_norm_bf16;

/// Fused residual-add + RMSNorm, also emitting the updated residual.
#[cutile::module]
pub mod add_rms_norm_bf16_module {
    use cutile::core::*;

    /// `residual_out = residual + x; out = residual_out / rms(residual_out) * w`
    /// in one kernel — the transformer-block epilogue/prologue fusion.
    #[cutile::entry(print_ir = false,
                    unchecked_accesses = true,
                    optimization_hints = (
                      sm_100 = (max_divisibility = 8,),
                      sm_103 = (max_divisibility = 8,),
                    ))]
    unsafe fn add_rms_norm_bf16<const N: i32, const BLOCK_SIZE: i32>(
        residual: &Tensor<bf16, { [-1, N] }>,
        x: &Tensor<bf16, { [-1, N] }>,
        w: &Tensor<bf16, { [N] }>,
        out: &mut Tensor<bf16, { [1, N] }>,
        residual_out: &mut Tensor<bf16, { [1, N] }>,
        eps: f32,
    ) {
        let tile_shape: Shape<{ [1, BLOCK_SIZE] }> = const_shape![1, BLOCK_SIZE];
        let num_tiles: i32 = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;

        let residual_part: Partition<bf16, { [1, BLOCK_SIZE] }> = residual.partition(tile_shape);
        let x_part: Partition<bf16, { [1, BLOCK_SIZE] }> = x.partition(tile_shape);

        let mut rms: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0f32, tile_shape);
        for j in 0i32..num_tiles {
            let tr_bf: Tile<bf16, { [1, BLOCK_SIZE] }> = residual_part.load([row, j]);
            let tx_bf: Tile<bf16, { [1, BLOCK_SIZE] }> = x_part.load([row, j]);
            let tr: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tr_bf);
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tx_bf);
            let combined: Tile<f32, { [1, BLOCK_SIZE] }> = tr + tx;
            rms = rms + combined * combined;
        }
        let rms: Tile<f32, { [1] }> = reduce_sum(rms, 1i32);
        let rms: Tile<f32, { [] }> = rms.reshape(const_shape![]);
        let rms: f32 = tile_to_scalar(rms);
        let n: f32 = convert_scalar(N);
        let inv_rms: f32 = rms / n + eps;
        let inv_rms: Tile<f32, { [] }> = rsqrt(scalar_to_tile(inv_rms), ftz::Disabled);
        let inv_rms: f32 = tile_to_scalar(inv_rms);
        let inv_rms: Tile<f32, { [1, BLOCK_SIZE] }> = inv_rms.broadcast(tile_shape);

        let w_part: Partition<bf16, { [BLOCK_SIZE] }> = w.partition(const_shape![BLOCK_SIZE]);
        let mut out_part: PartitionMut<bf16, { [1, BLOCK_SIZE] }> = out.partition_mut(tile_shape);
        let mut res_out_part: PartitionMut<bf16, { [1, BLOCK_SIZE] }> =
            residual_out.partition_mut(tile_shape);
        for j in 0i32..num_tiles {
            let tr_bf: Tile<bf16, { [1, BLOCK_SIZE] }> = residual_part.load([row, j]);
            let tx_bf: Tile<bf16, { [1, BLOCK_SIZE] }> = x_part.load([row, j]);
            let tw_bf: Tile<bf16, { [BLOCK_SIZE] }> = w_part.load([j]);
            let tr: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tr_bf);
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tx_bf);
            let tw: Tile<f32, { [1, BLOCK_SIZE] }> =
                convert_tile(tw_bf.reshape(const_shape![1, BLOCK_SIZE]));
            let combined: Tile<f32, { [1, BLOCK_SIZE] }> = tr + tx;
            let normed: Tile<f32, { [1, BLOCK_SIZE] }> = combined * inv_rms * tw;
            out_part.store(convert_tile(normed), [0i32, j]);
            res_out_part.store(convert_tile(combined), [0i32, j]);
        }
    }
}

pub use add_rms_norm_bf16_module::add_rms_norm_bf16;
