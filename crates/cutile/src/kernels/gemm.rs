//! Tensor-core GEMM kernels (DeepGEMM NT convention: `D = A @ B^T` with
//! `A: [M, K]`, `B: [N, K]`).
//!
//! The FP8 kernels use `mmaf_scaled` — Blackwell block-scaled MMA with E4M3
//! operands and UE8M0 scales at 32-lane granularity, the same numeric
//! contract as DeepGEMM's SM100 recipe (1×128 activation / 128×128 weight
//! scales, replicated ×4 into 32-lane slots by [`crate::pack`] on the host
//! for weights and by `quant_fp8_block128` on the device for activations).

/// Dense BF16 GEMM, f32 accumulation.
#[cutile::module]
pub mod bf16_gemm_nt_module {
    use cutile::core::*;

    /// `D = A @ B^T` over `[M, K] x [N, K]` BF16 with `[BM, BN]` tiles and
    /// `BK`-sized K steps. Tail blocks are zero-padded on load and clipped
    /// on store by the partition views.
    #[cutile::entry(print_ir = false,
                    unchecked_accesses = true,
                    optimization_hints = (
                      sm_100 = (max_divisibility = 16,),
                      sm_103 = (max_divisibility = 16,),
                    ))]
    unsafe fn bf16_gemm_nt<const BM: i32, const BN: i32, const BK: i32>(
        d: &mut Tensor<bf16, { [BM, BN] }>,
        a: &Tensor<bf16, { [-1, -1] }>,
        b: &Tensor<bf16, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let part_a = a.partition(const_shape![BM, BK]);
        let part_b = b.partition(const_shape![BN, BK]);
        let k_tiles: i32 = ceil_div(a.shape()[1], BK);

        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
        for k in 0i32..k_tiles {
            let tile_a: Tile<bf16, { [BM, BK] }> = part_a.load([pid.0, k]);
            let tile_b: Tile<bf16, { [BN, BK] }> = part_b.load([pid.1, k]);
            let tile_bt: Tile<bf16, { [BK, BN] }> = tile_b.transpose();
            acc = mma(tile_a, tile_bt, acc);
        }
        d.store(convert_tile(acc));
    }
}

pub use bf16_gemm_nt_module::bf16_gemm_nt;

/// Dense FP8 block-scaled GEMM (`mmaf_scaled`), f32 accumulation, BF16 out.
#[cutile::module]
pub mod fp8_scaled_gemm_nt_module {
    use cutile::core::*;

    /// `D = A @ B^T` with E4M3 data and UE8M0 scales:
    /// `a_scales: [M, K/32]`, `b_scales: [N, K/32]` (per-128 DeepSeek recipe
    /// pre-replicated ×4). `BKS = BK / 32`.
    #[cutile::entry(print_ir = false,
                    unchecked_accesses = true,
                    optimization_hints = (
                      sm_100 = (max_divisibility = 16,),
                      sm_103 = (max_divisibility = 16,),
                    ))]
    unsafe fn fp8_scaled_gemm_nt<const BM: i32, const BN: i32, const BK: i32, const BKS: i32>(
        d: &mut Tensor<bf16, { [BM, BN] }>,
        a: &Tensor<f8e4m3fn, { [-1, -1] }>,
        b: &Tensor<f8e4m3fn, { [-1, -1] }>,
        a_scales: &Tensor<f8e8m0fnu, { [-1, -1] }>,
        b_scales: &Tensor<f8e8m0fnu, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let part_a = a.partition(const_shape![BM, BK]);
        let part_b = b.partition(const_shape![BN, BK]);
        let part_as = a_scales.partition(const_shape![BM, BKS]);
        let part_bs = b_scales.partition(const_shape![BN, BKS]);
        let k_tiles: i32 = ceil_div(a.shape()[1], BK);

        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
        for k in 0i32..k_tiles {
            let tile_a: Tile<f8e4m3fn, { [BM, BK] }> = part_a.load([pid.0, k]);
            let tile_b: Tile<f8e4m3fn, { [BN, BK] }> = part_b.load([pid.1, k]);
            let tile_bt: Tile<f8e4m3fn, { [BK, BN] }> = tile_b.transpose();
            let tile_as: Tile<f8e8m0fnu, { [BM, BKS] }> = part_as.load([pid.0, k]);
            let tile_bs: Tile<f8e8m0fnu, { [BN, BKS] }> = part_bs.load([pid.1, k]);
            let tile_bst: Tile<f8e8m0fnu, { [BKS, BN] }> = tile_bs.transpose();
            acc = mmaf_scaled(tile_a, tile_bt, acc, tile_as, tile_bst);
        }
        d.store(convert_tile(acc));
    }
}

pub use fp8_scaled_gemm_nt_module::fp8_scaled_gemm_nt;

/// Grouped FP8 GEMM, contiguous layout (MoE prefill): activations are
/// already expert-sorted, `m_indices[m]` gives the owning expert (or -1 for
/// padding rows, which produce zero output).
#[cutile::module]
pub mod fp8_grouped_gemm_nt_contiguous_module {
    use cutile::core::*;

    /// `D[m, :] = A[m, :] @ B[expert*n .., :]^T` with `expert =
    /// m_indices[BM * pid.0]`. `b` stacks all experts' `[N, K]` weights.
    /// `n` (per-expert output width) must be a multiple of `BN`.
    #[cutile::entry(print_ir = false,
                    unchecked_accesses = true,
                    optimization_hints = (
                      sm_100 = (max_divisibility = 16,),
                      sm_103 = (max_divisibility = 16,),
                    ))]
    unsafe fn fp8_grouped_gemm_nt_contiguous<
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const BKS: i32,
    >(
        d: &mut Tensor<bf16, { [BM, BN] }>,
        a: &Tensor<f8e4m3fn, { [-1, -1] }>,
        b: &Tensor<f8e4m3fn, { [-1, -1] }>,
        a_scales: &Tensor<f8e8m0fnu, { [-1, -1] }>,
        b_scales: &Tensor<f8e8m0fnu, { [-1, -1] }>,
        m_indices: &Tensor<i32, { [-1] }>,
        n: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row0: i32 = pid.0 * BM;
        let expert: i32 = tile_to_scalar(
            m_indices
                .partition(const_shape![1])
                .load([row0])
                .reshape(const_shape![]),
        );

        if expert < 0 {
            // Padding rows: deterministic zero output.
            let zero: Tile<bf16, { [BM, BN] }> = convert_tile(constant(0.0f32, const_shape![BM, BN]));
            d.store(zero);
        } else {
            let part_a = a.partition(const_shape![BM, BK]);
            let part_b = b.partition(const_shape![BN, BK]);
            let part_as = a_scales.partition(const_shape![BM, BKS]);
            let part_bs = b_scales.partition(const_shape![BN, BKS]);
            let k_tiles: i32 = ceil_div(a.shape()[1], BK);
            let w_row_tile: i32 = expert * (n / BN) + pid.1;

            let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            for k in 0i32..k_tiles {
                let tile_a: Tile<f8e4m3fn, { [BM, BK] }> = part_a.load([pid.0, k]);
                let tile_b: Tile<f8e4m3fn, { [BN, BK] }> = part_b.load([w_row_tile, k]);
                let tile_bt: Tile<f8e4m3fn, { [BK, BN] }> = tile_b.transpose();
                let tile_as: Tile<f8e8m0fnu, { [BM, BKS] }> = part_as.load([pid.0, k]);
                let tile_bs: Tile<f8e8m0fnu, { [BN, BKS] }> = part_bs.load([w_row_tile, k]);
                let tile_bst: Tile<f8e8m0fnu, { [BKS, BN] }> = tile_bs.transpose();
                acc = mmaf_scaled(tile_a, tile_bt, acc, tile_as, tile_bst);
            }
            d.store(convert_tile(acc));
        }
    }
}

pub use fp8_grouped_gemm_nt_contiguous_module::fp8_grouped_gemm_nt_contiguous;

/// Grouped FP8 GEMM, masked layout (MoE decode, CUDA-graph friendly):
/// per-expert padded activations `[E, max_m, K]` with `masked_m[e]` valid
/// rows; blocks entirely beyond `masked_m` store zeros.
#[cutile::module]
pub mod fp8_grouped_gemm_nt_masked_module {
    use cutile::core::*;

    /// `D[e, m, :] = A[e, m, :] @ B[e*n .., :]^T` for `m < masked_m[e]`.
    /// Rows at or beyond `masked_m[e]` are zero (the activation buffer is
    /// expected to be zero-initialized so partially-valid blocks are
    /// deterministic).
    #[cutile::entry(print_ir = false,
                    unchecked_accesses = true,
                    optimization_hints = (
                      sm_100 = (max_divisibility = 16,),
                      sm_103 = (max_divisibility = 16,),
                    ))]
    unsafe fn fp8_grouped_gemm_nt_masked<
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const BKS: i32,
    >(
        d: &mut Tensor<bf16, { [1, BM, BN] }>,
        a: &Tensor<f8e4m3fn, { [-1, -1, -1] }>,
        b: &Tensor<f8e4m3fn, { [-1, -1] }>,
        a_scales: &Tensor<f8e8m0fnu, { [-1, -1, -1] }>,
        b_scales: &Tensor<f8e8m0fnu, { [-1, -1] }>,
        masked_m: &Tensor<i32, { [-1] }>,
        n: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let e: i32 = pid.0;
        let m0: i32 = pid.1 * BM;
        let valid: i32 = tile_to_scalar(
            masked_m
                .partition(const_shape![1])
                .load([e])
                .reshape(const_shape![]),
        );

        if m0 < valid {
            let part_a = a.partition(const_shape![1, BM, BK]);
            let part_as = a_scales.partition(const_shape![1, BM, BKS]);
            let part_b = b.partition(const_shape![BN, BK]);
            let part_bs = b_scales.partition(const_shape![BN, BKS]);
            let k_tiles: i32 = ceil_div(a.shape()[2], BK);
            let w_row_tile: i32 = e * (n / BN) + pid.2;

            let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            for k in 0i32..k_tiles {
                let tile_a3: Tile<f8e4m3fn, { [1, BM, BK] }> = part_a.load([e, pid.1, k]);
                let tile_a: Tile<f8e4m3fn, { [BM, BK] }> =
                    tile_a3.reshape(const_shape![BM, BK]);
                let tile_b: Tile<f8e4m3fn, { [BN, BK] }> = part_b.load([w_row_tile, k]);
                let tile_bt: Tile<f8e4m3fn, { [BK, BN] }> = tile_b.transpose();
                let tile_as3: Tile<f8e8m0fnu, { [1, BM, BKS] }> = part_as.load([e, pid.1, k]);
                let tile_as: Tile<f8e8m0fnu, { [BM, BKS] }> =
                    tile_as3.reshape(const_shape![BM, BKS]);
                let tile_bs: Tile<f8e8m0fnu, { [BN, BKS] }> = part_bs.load([w_row_tile, k]);
                let tile_bst: Tile<f8e8m0fnu, { [BKS, BN] }> = tile_bs.transpose();
                acc = mmaf_scaled(tile_a, tile_bt, acc, tile_as, tile_bst);
            }
            let out3: Tile<bf16, { [1, BM, BN] }> =
                convert_tile(acc).reshape(const_shape![1, BM, BN]);
            d.store(out3);
        } else {
            let zero: Tile<bf16, { [1, BM, BN] }> =
                convert_tile(constant(0.0f32, const_shape![1, BM, BN]));
            d.store(zero);
        }
    }
}

pub use fp8_grouped_gemm_nt_masked_module::fp8_grouped_gemm_nt_masked;
