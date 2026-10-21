//! Shared device functions (inlined into entry points at JIT time).
//!
//! Inter-module device-function calls are supported by the cuTile macro:
//! importing from this module records it as a JIT dependency and inlines
//! the functions into the importing entry point.

/// Scale/quantization math shared by the quantization and KV-cache kernels.
#[cutile::module]
pub mod scale_helpers {
    use cutile::core::*;

    /// UE8M0 scale byte for one block from its amax, matching the CPU
    /// reference (`f32_to_e8m0(amax/448, Ceil)`; DeepGEMM `round_sf` Ceil
    /// policy): `2^ceil(log2(amax/448))`, exact powers of two not rounded
    /// up, all-zero blocks → byte 0 (2^-127).
    pub fn e8m0_byte_from_amax(amax: Tile<f32, { [] }>) -> Tile<f8e8m0fnu, { [] }> {
        let pos: Tile<bool, { [] }> = gt_tile(amax, scalar_to_tile(0.0f32));
        // amax == 0 → ratio 2^-133 → clamped to 2^-127 below (byte 0).
        let tiny: Tile<f32, { [] }> = scalar_to_tile(1.0e-40f32);
        let ratio: Tile<f32, { [] }> = select(pos, true_div(amax, scalar_to_tile(448.0f32)), tiny);
        let l2: Tile<f32, { [] }> = log2(ratio);
        let e_ceil: Tile<f32, { [] }> = ceil(l2);
        // Exact powers of two must not be rounded up by Ceil.
        let near: Tile<bool, { [] }> =
            lt_tile(absf(e_ceil - l2), scalar_to_tile(1e-9f32));
        let e: Tile<f32, { [] }> = select(near, floor(l2), e_ceil);
        let e: Tile<f32, { [] }> =
            maxf(e, scalar_to_tile(-127.0f32), nan::Disabled, ftz::Disabled);
        let e: Tile<f32, { [] }> =
            minf(e, scalar_to_tile(127.0f32), nan::Disabled, ftz::Disabled);
        let biased: Tile<f32, { [] }> = e + scalar_to_tile(127.0f32);
        let e_i: Tile<i32, { [] }> = ftoi(biased, rounding::NearestIntToZero);
        let e_u8: Tile<u8, { [] }> = trunci(e_i, overflow::None);
        bitcast(e_u8)
    }

    /// The UE8M0 scale *value* `2^e` for one block (reciprocal of the
    /// quantization multiplier).
    pub fn e8m0_value_from_amax(amax: Tile<f32, { [] }>) -> Tile<f32, { [] }> {
        let byte: Tile<f8e8m0fnu, { [] }> = e8m0_byte_from_amax(amax);
        let b_u8: Tile<u8, { [] }> = bitcast(byte);
        let b_i: Tile<i32, { [] }> = exti(b_u8);
        let biased: Tile<f32, { [] }> = convert_tile(b_i);
        exp2(biased - scalar_to_tile(127.0f32), ftz::Disabled)
    }

    /// E4M3 scale value for one FP4 group: `e4m3(amax/6)` rounded to the
    /// lattice, floored at the smallest positive normal (2^-6) — which also
    /// covers the all-zero group.
    pub fn fp4_scale_from_amax(amax: Tile<f32, { [] }>) -> Tile<f32, { [] }> {
        let raw: Tile<f32, { [] }> = true_div(amax, scalar_to_tile(6.0f32));
        let s_f8: Tile<f8e4m3fn, { [] }> = convert_tile(raw);
        let s_f32: Tile<f32, { [] }> = convert_tile(s_f8);
        maxf(s_f32, scalar_to_tile(0.015625f32), nan::Disabled, ftz::Disabled)
    }
}
