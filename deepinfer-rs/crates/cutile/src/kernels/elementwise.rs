//! Elementwise kernels (BF16 activations, F32 routing logits).

/// Residual add over `[rows, cols]` BF16 tensors.
#[cutile::module]
pub mod add_bf16_module {
    use cutile::core::*;

    /// `out = lhs + rhs` elementwise on `[rows, cols]` BF16 tiles.
    #[cutile::entry(print_ir = false,
                    optimization_hints = (
                      sm_100 = (max_divisibility = 16,),
                      sm_103 = (max_divisibility = 16,),
                    ))]
    fn add_bf16<const BLOCK_SIZE: i32>(
        out: &mut Tensor<bf16, { [1, BLOCK_SIZE] }>,
        lhs: &Tensor<bf16, { [-1, -1] }>,
        rhs: &Tensor<bf16, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;
        let col = pid.1;
        let lhs_part: Partition<bf16, { [1, BLOCK_SIZE] }> =
            lhs.partition(const_shape![1, BLOCK_SIZE]);
        let rhs_part: Partition<bf16, { [1, BLOCK_SIZE] }> =
            rhs.partition(const_shape![1, BLOCK_SIZE]);
        let lhs_tile: Tile<bf16, { [1, BLOCK_SIZE] }> = lhs_part.load([row, col]);
        let rhs_tile: Tile<bf16, { [1, BLOCK_SIZE] }> = rhs_part.load([row, col]);
        out.store(lhs_tile + rhs_tile);
    }
}

pub use add_bf16_module::add_bf16;

/// Fused SwiGLU activation `silu(gate) * up` over `[rows, cols]` BF16.
#[cutile::module]
pub mod silu_mul_bf16_module {
    use cutile::core::*;

    /// `out = gate * sigmoid(gate) * up`, computed in F32 and stored BF16.
    #[cutile::entry(print_ir = false,
                    optimization_hints = (
                      sm_100 = (max_divisibility = 16,),
                      sm_103 = (max_divisibility = 16,),
                    ))]
    fn silu_mul_bf16<const BLOCK_SIZE: i32>(
        out: &mut Tensor<bf16, { [1, BLOCK_SIZE] }>,
        gate: &Tensor<bf16, { [-1, -1] }>,
        up: &Tensor<bf16, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;
        let col = pid.1;
        let gate_part: Partition<bf16, { [1, BLOCK_SIZE] }> =
            gate.partition(const_shape![1, BLOCK_SIZE]);
        let up_part: Partition<bf16, { [1, BLOCK_SIZE] }> =
            up.partition(const_shape![1, BLOCK_SIZE]);
        let gate_bf: Tile<bf16, { [1, BLOCK_SIZE] }> = gate_part.load([row, col]);
        let up_bf: Tile<bf16, { [1, BLOCK_SIZE] }> = up_part.load([row, col]);
        let gate_f: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(gate_bf);
        let up_f: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(up_bf);
        let one: Tile<f32, { [1, BLOCK_SIZE] }> = constant(1.0f32, const_shape![1, BLOCK_SIZE]);
        let zero: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0f32, const_shape![1, BLOCK_SIZE]);
        let neg_gate: Tile<f32, { [1, BLOCK_SIZE] }> = zero - gate_f;
        let sigmoid: Tile<f32, { [1, BLOCK_SIZE] }> = true_div(one, one + exp(neg_gate));
        let y: Tile<f32, { [1, BLOCK_SIZE] }> = sigmoid * gate_f * up_f;
        out.store(convert_tile(y));
    }
}

pub use silu_mul_bf16_module::silu_mul_bf16;

/// Routing-gate multiply `sigmoid(x) * y` over `[rows, cols]` F32 (DSA gate).
#[cutile::module]
pub mod sigmoid_mul_f32_module {
    use cutile::core::*;

    /// `out = sigmoid(x) * y` elementwise on F32 tiles (DSA conv-gate).
    #[cutile::entry(print_ir = false,
                    optimization_hints = (
                      sm_100 = (max_divisibility = 16,),
                      sm_103 = (max_divisibility = 16,),
                    ))]
    fn sigmoid_mul_f32<const BLOCK_SIZE: i32>(
        out: &mut Tensor<f32, { [1, BLOCK_SIZE] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        y: &Tensor<f32, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;
        let col = pid.1;
        let x_part: Partition<f32, { [1, BLOCK_SIZE] }> = x.partition(const_shape![1, BLOCK_SIZE]);
        let y_part: Partition<f32, { [1, BLOCK_SIZE] }> = y.partition(const_shape![1, BLOCK_SIZE]);
        let x_t: Tile<f32, { [1, BLOCK_SIZE] }> = x_part.load([row, col]);
        let y_t: Tile<f32, { [1, BLOCK_SIZE] }> = y_part.load([row, col]);
        let one: Tile<f32, { [1, BLOCK_SIZE] }> = constant(1.0f32, const_shape![1, BLOCK_SIZE]);
        let zero: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0f32, const_shape![1, BLOCK_SIZE]);
        let sigmoid: Tile<f32, { [1, BLOCK_SIZE] }> = true_div(one, one + exp(zero - x_t));
        out.store(sigmoid * y_t);
    }
}

pub use sigmoid_mul_f32_module::sigmoid_mul_f32;

/// Row gather: `out[i, :] = src[indices[i], :]` over BF16 `[rows, cols]`
/// (invalid indices gather row 0 and the caller masks them away).
#[cutile::module]
pub mod gather_rows_bf16_module {
    use cutile::core::*;

    /// Gathers `indices.len()` rows from `src` into `out`. Invalid
    /// (negative) indices gather a zero row: the caller passes
    /// `indices` pre-sanitized to `0` for invalid lanes.
    #[cutile::entry(print_ir = false,
                    unchecked_accesses = true,
                    optimization_hints = (
                      sm_100 = (max_divisibility = 16,),
                      sm_103 = (max_divisibility = 16,),
                    ))]
    unsafe fn gather_rows_bf16<const BLOCK_SIZE: i32>(
        out: &mut Tensor<bf16, { [1, BLOCK_SIZE] }>,
        src: &Tensor<bf16, { [-1, -1] }>,
        indices: &Tensor<i32, { [-1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;
        let col = pid.1;
        let idx_part = indices.partition(const_shape![1]);
        let idx: i32 = tile_to_scalar(idx_part.load([row]).reshape(const_shape![]));
        let src_part: Partition<bf16, { [1, BLOCK_SIZE] }> =
            src.partition(const_shape![1, BLOCK_SIZE]);
        let tile: Tile<bf16, { [1, BLOCK_SIZE] }> = src_part.load([idx, col]);
        out.store(tile);
    }
}

pub use gather_rows_bf16_module::gather_rows_bf16;
