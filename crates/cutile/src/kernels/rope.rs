//! Rotary position embedding kernels for MLA.
//!
//! V4.1-Flash uses *interleaved* (non-neox) RoPE pairs over `rope_dim = 64`:
//! lane `2i` pairs with lane `2i+1`. Both kernels take the host-precomputed
//! `inv_freq` table from `deepinfer_core::rope::Rope` (which carries any
//! YaRN scaling), so device and CPU numerics agree bit-for-bit on the
//! rotation angles.

/// Interleaved RoPE over a `[seq, heads, ROPE]` tensor (MLA `q_pe` / `k_pe`).
#[cutile::module]
pub mod rope_interleaved_module {
    use cutile::core::*;

    /// `out[s, h, 2i]   = x[s, h, 2i] * cos(θ_i) - x[s, h, 2i+1] * sin(θ_i)`
    /// `out[s, h, 2i+1] = x[s, h, 2i+1] * cos(θ_i) + x[s, h, 2i] * sin(θ_i)`
    /// with `θ_i = (position_start + s) * inv_freq[i]`. Grid: `(seq, heads)`.
    #[cutile::entry(print_ir = false,
                    unchecked_accesses = true,
                    optimization_hints = (
                      sm_100 = (max_divisibility = 8,),
                      sm_103 = (max_divisibility = 8,),
                    ))]
    unsafe fn rope_interleaved<const ROPE: i32, const HALF: i32>(
        out: &mut Tensor<bf16, { [1, 1, ROPE] }>,
        x: &Tensor<bf16, { [-1, -1, ROPE] }>,
        inv_freq: &Tensor<f32, { [HALF] }>,
        position_start: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let seq_idx = pid.0;
        let head_idx = pid.1;
        let heads: i32 = x.shape()[1];
        let row_off: i32 = (seq_idx * heads + head_idx) * ROPE;

        // Strided (interleaved) accesses via raw pointers. Loads offset from
        // the whole-tensor base by the row; stores are slab-relative (the
        // output partition hands each CTA its own row).
        let xb: PointerTile<*const bf16, { [] }> = pointer_to_tile(x.as_ptr());
        let ob: PointerTile<*mut bf16, { [] }> = pointer_to_tile(cast_mut(out.as_ptr()));
        let pair: Tile<i32, { [HALF] }> = iota(const_shape![HALF]);
        let two: Tile<i32, { [HALF] }> = broadcast_scalar(2i32, const_shape![HALF]);
        let even_off: Tile<i32, { [HALF] }> = pair * two;
        let odd_off: Tile<i32, { [HALF] }> = even_off + broadcast_scalar(1i32, const_shape![HALF]);
        let row_off_t: Tile<i32, { [HALF] }> = broadcast_scalar(row_off, const_shape![HALF]);
        let even_off_x: Tile<i32, { [HALF] }> = even_off + row_off_t;
        let odd_off_x: Tile<i32, { [HALF] }> = odd_off + row_off_t;

        let xb_b: PointerTile<*const bf16, { [HALF] }> =
            xb.reshape(const_shape![1]).broadcast(const_shape![HALF]);
        let ob_b: PointerTile<*mut bf16, { [HALF] }> =
            ob.reshape(const_shape![1]).broadcast(const_shape![HALF]);
        let (lo_bf, tok): (Tile<bf16, { [HALF] }>, Token) = unsafe {
            load_ptr_tko(
                xb_b.offset_tile(even_off_x),
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None::<Token>,
                Latency::<0>,
            )
        };
        let (hi_bf, _): (Tile<bf16, { [HALF] }>, Token) = unsafe {
            load_ptr_tko(
                xb_b.offset_tile(odd_off_x),
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None::<Token>,
                Latency::<0>,
            )
        };

        // θ = (position_start + seq) * inv_freq
        let freq: Tile<f32, { [HALF] }> = inv_freq.partition(const_shape![HALF]).load([0i32]);
        let pos: f32 = convert_scalar(position_start + seq_idx);
        let theta: Tile<f32, { [HALF] }> = freq * broadcast_scalar(pos, const_shape![HALF]);
        let cos_t: Tile<f32, { [HALF] }> = cos(theta);
        let sin_t: Tile<f32, { [HALF] }> = sin(theta);

        let lo: Tile<f32, { [HALF] }> = convert_tile(lo_bf);
        let hi: Tile<f32, { [HALF] }> = convert_tile(hi_bf);
        let out_lo: Tile<f32, { [HALF] }> = lo * cos_t - hi * sin_t;
        let out_hi: Tile<f32, { [HALF] }> = hi * cos_t + lo * sin_t;

        let _: Token = unsafe {
            store_ptr_tko(
                ob_b.offset_tile(even_off),
                convert_tile(out_lo),
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                Some(tok),
                Latency::<0>,
            )
        };
        let _: Token = unsafe {
            store_ptr_tko(
                ob_b.offset_tile(odd_off),
                convert_tile(out_hi),
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None::<Token>,
                Latency::<0>,
            )
        };
    }
}

pub use rope_interleaved_module::rope_interleaved;

/// O-RoPE conjugate: rotates the trailing `ROPE` lanes of each output row
/// in place by the conjugate of that row's query position, so the output
/// latents live in the query's relative frame (FlashMLA V4.1 "O-RoPE
/// (conjugate)").
#[cutile::module]
pub mod o_rope_conjugate_module {
    use cutile::core::*;

    /// For each row `r` of a `[rows, ROW_LEN]` tensor (laid out
    /// `[token, head]`, `heads` rows per token), the slice
    /// `[ROW_LEN - ROPE, ROW_LEN)` is rotated by `-θ` where
    /// `θ = (position_start + r / heads) * inv_freq[i]`. In-place.
    #[cutile::entry(print_ir = false,
                    unchecked_accesses = true,
                    optimization_hints = (
                      sm_100 = (max_divisibility = 8,),
                      sm_103 = (max_divisibility = 8,),
                    ))]
    unsafe fn o_rope_conjugate<const ROW_LEN: i32, const ROPE: i32, const HALF: i32>(
        x: &mut Tensor<bf16, { [1, ROW_LEN] }>,
        inv_freq: &Tensor<f32, { [HALF] }>,
        position_start: i32,
        heads: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;
        let base: i32 = ROW_LEN - ROPE;

        let xb: PointerTile<*mut bf16, { [] }> = pointer_to_tile(cast_mut(x.as_ptr()));
        let pair: Tile<i32, { [HALF] }> = iota(const_shape![HALF]);
        let two: Tile<i32, { [HALF] }> = broadcast_scalar(2i32, const_shape![HALF]);
        let even_off: Tile<i32, { [HALF] }> = pair * two + broadcast_scalar(base, const_shape![HALF]);
        let odd_off: Tile<i32, { [HALF] }> = even_off + broadcast_scalar(1i32, const_shape![HALF]);
        let xb_b: PointerTile<*mut bf16, { [HALF] }> =
            xb.reshape(const_shape![1]).broadcast(const_shape![HALF]);

        let (lo_bf, tok): (Tile<bf16, { [HALF] }>, Token) = unsafe {
            load_ptr_tko(
                xb_b.offset_tile(even_off),
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None::<Token>,
                Latency::<0>,
            )
        };
        let (hi_bf, _): (Tile<bf16, { [HALF] }>, Token) = unsafe {
            load_ptr_tko(
                xb_b.offset_tile(odd_off),
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None::<Token>,
                Latency::<0>,
            )
        };

        // Conjugate rotation: θ -> -θ.
        let freq: Tile<f32, { [HALF] }> = inv_freq.partition(const_shape![HALF]).load([0i32]);
        let pos: f32 = convert_scalar(position_start + row / heads);
        let theta: Tile<f32, { [HALF] }> = freq * broadcast_scalar(pos, const_shape![HALF]);
        let cos_t: Tile<f32, { [HALF] }> = cos(theta);
        let sin_t: Tile<f32, { [HALF] }> = sin(theta);

        let lo: Tile<f32, { [HALF] }> = convert_tile(lo_bf);
        let hi: Tile<f32, { [HALF] }> = convert_tile(hi_bf);
        let out_lo: Tile<f32, { [HALF] }> = lo * cos_t + hi * sin_t;
        let out_hi: Tile<f32, { [HALF] }> = hi * cos_t - lo * sin_t;

        let _: Token = unsafe {
            store_ptr_tko(
                xb_b.offset_tile(even_off),
                convert_tile(out_lo),
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                Some(tok),
                Latency::<0>,
            )
        };
        let _: Token = unsafe {
            store_ptr_tko(
                xb_b.offset_tile(odd_off),
                convert_tile(out_hi),
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None::<Token>,
                Latency::<0>,
            )
        };
    }
}

pub use o_rope_conjugate_module::o_rope_conjugate;
