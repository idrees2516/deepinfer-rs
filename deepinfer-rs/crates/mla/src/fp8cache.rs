//! Private KV-cache row codecs for the three [`CacheFormat`] variants.
//!
//! The `deepinfer-fp8` crate (which owns the V4.1 KV-cache codecs per the
//! workspace SPEC: `quantize_kv_cache_v41_fp8` / `dequantize_kv_cache_v41_fp8`
//! and the `_fp4` pair) is still a placeholder at the time of writing, so
//! this module carries private mirrors with byte-identical semantics
//! (FlashMLA README, V4.1 cache formats):
//!
//! * **FP8** (`Fp8Group32`, 528 B per 512-dim row): one E4M3 value per dim
//!   followed by one UE8M0 scale per 32 dims, with scale =
//!   `2^ceil(log2(amax / 448))` (DeepGEMM `round_sf` Ceil policy) and
//!   saturating RNE value casts.
//! * **FP4** (`Fp4Group16`, 288 B per 512-dim row): packed E2M1 nibbles
//!   (the even-index value in the LOW nibble) followed by one E4M3 scale
//!   per 16 dims, with scale = `amax / 6` rounded to E4M3 with RNE and
//!   floored to at least the smallest positive E4M3 normal (2^-6).
//! * **BF16**: little-endian BF16 per dim.
//!
//! The helpers are dimension-generic (the canonical 512-dim row yields
//! exactly 528 / 288 / 1024 bytes); switching to the `deepinfer-fp8` APIs
//! once that crate lands is a drop-in replacement.

use crate::config::CacheFormat;
use deepinfer_core::format::{
    bf16_to_f32, e2m1_to_f32, e4m3_to_f32, e8m0_to_f32, f32_to_bf16_rne, f32_to_e2m1_rne,
    f32_to_e4m3_rne, f32_to_e8m0, E8M0Round,
};
use deepinfer_core::{ensure, Error, Result};

/// FP8 scale-group size (dims per UE8M0 scale byte).
pub(crate) const FP8_GROUP: usize = 32;
/// FP4 scale-group size (dims per E4M3 scale byte).
pub(crate) const FP4_GROUP: usize = 16;
/// Smallest positive normal E4M3 value (2^-6) — the FP4 scale floor.
pub(crate) const E4M3_MIN_NORMAL: f32 = 0.015_625;

/// Byte size of one quantized KV-cache row, or `None` on overflow.
pub(crate) fn row_bytes(format: CacheFormat, head_dim: usize) -> Option<usize> {
    match format {
        CacheFormat::Bf16 => head_dim.checked_mul(2),
        CacheFormat::Fp8Group32 => head_dim.checked_add(head_dim / FP8_GROUP),
        CacheFormat::Fp4Group16 => (head_dim / 2).checked_add(head_dim / FP4_GROUP),
    }
}

/// UE8M0 scale byte for one group: `2^ceil(log2(amax / 448))`, `0`
/// (2^-127) for an all-zero group. Mirrors DeepGEMM's activation recipe.
fn fp8_group_scale(group: &[f32]) -> Result<u8> {
    let mut amax = 0.0f32;
    for &v in group {
        let a = v.abs();
        if a > amax {
            amax = a;
        }
    }
    ensure!(
        amax.is_finite(),
        Numerical,
        "non-finite value in FP8 quantization group"
    );
    if amax == 0.0 {
        return Ok(0);
    }
    f32_to_e8m0(amax / 448.0, E8M0Round::Ceil)
}

/// E4M3 scale value for one FP4 group: `amax / 6` rounded to the E4M3
/// lattice with RNE, floored to the smallest positive normal.
fn fp4_group_scale(group: &[f32]) -> Result<f32> {
    let mut amax = 0.0f32;
    for &v in group {
        let a = v.abs();
        if a > amax {
            amax = a;
        }
    }
    ensure!(
        amax.is_finite(),
        Numerical,
        "non-finite value in FP4 quantization group"
    );
    if amax == 0.0 {
        return Ok(E4M3_MIN_NORMAL);
    }
    let s = e4m3_to_f32(f32_to_e4m3_rne(amax / 6.0, true));
    Ok(if s < E4M3_MIN_NORMAL {
        E4M3_MIN_NORMAL
    } else {
        s
    })
}

/// Quantizes one KV-cache token row (`dims`) into `out` per `format`.
pub(crate) fn quantize_row(format: CacheFormat, dims: &[f32], out: &mut [u8]) -> Result<()> {
    let expect = row_bytes(format, dims.len()).ok_or_else(|| {
        Error::CapacityExceeded(format!("row byte size overflows for {} dims", dims.len()))
    })?;
    ensure!(
        out.len() == expect,
        DimensionMismatch,
        "quantize buffer has {} bytes, expected {expect}",
        out.len()
    );
    match format {
        CacheFormat::Bf16 => {
            for (i, &v) in dims.iter().enumerate() {
                out[2 * i..2 * i + 2].copy_from_slice(&f32_to_bf16_rne(v).to_le_bytes());
            }
        }
        CacheFormat::Fp8Group32 => {
            let d = dims.len();
            for g in 0..d / FP8_GROUP {
                let group = &dims[g * FP8_GROUP..(g + 1) * FP8_GROUP];
                let scale_byte = fp8_group_scale(group)?;
                let scale = e8m0_to_f32(scale_byte);
                let values = &mut out[g * FP8_GROUP..(g + 1) * FP8_GROUP];
                for (b, &v) in values.iter_mut().zip(group.iter()) {
                    *b = f32_to_e4m3_rne(v / scale, true);
                }
                out[d + g] = scale_byte;
            }
        }
        CacheFormat::Fp4Group16 => {
            let d = dims.len();
            for g in 0..d / FP4_GROUP {
                let group = &dims[g * FP4_GROUP..(g + 1) * FP4_GROUP];
                let scale = fp4_group_scale(group)?;
                let packed = &mut out[g * FP4_GROUP / 2..(g + 1) * FP4_GROUP / 2];
                // Even-index value in the LOW nibble, odd-index in the high.
                for p in 0..FP4_GROUP / 2 {
                    let lo = f32_to_e2m1_rne(group[2 * p] / scale);
                    let hi = f32_to_e2m1_rne(group[2 * p + 1] / scale);
                    packed[p] = lo | (hi << 4);
                }
                out[d / 2 + g] = f32_to_e4m3_rne(scale, true);
            }
        }
    }
    Ok(())
}

/// Dequantizes one KV-cache token row from `bytes` into `out` (values are
/// `dequant(v) * scale` per group). Sizes must match
/// [`row_bytes`] — enforced by the caller.
pub(crate) fn dequantize_row(format: CacheFormat, bytes: &[u8], out: &mut [f32]) {
    debug_assert_eq!(
        bytes.len(),
        row_bytes(format, out.len()).unwrap_or(usize::MAX)
    );
    match format {
        CacheFormat::Bf16 => {
            for (i, v) in out.iter_mut().enumerate() {
                *v = bf16_to_f32(u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]));
            }
        }
        CacheFormat::Fp8Group32 => {
            let d = out.len();
            for g in 0..d / FP8_GROUP {
                let s = e8m0_to_f32(bytes[d + g]);
                for j in 0..FP8_GROUP {
                    out[g * FP8_GROUP + j] = e4m3_to_f32(bytes[g * FP8_GROUP + j]) * s;
                }
            }
        }
        CacheFormat::Fp4Group16 => {
            let d = out.len();
            for g in 0..d / FP4_GROUP {
                let s = e4m3_to_f32(bytes[d / 2 + g]);
                for j in 0..FP4_GROUP {
                    let byte = bytes[g * FP4_GROUP / 2 + j / 2];
                    let nib = if j % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                    out[g * FP4_GROUP + j] = e2m1_to_f32(nib) * s;
                }
            }
        }
    }
}

/// Per-row E4M3 + UE8M0 cast with groups of `group` consecutive dims
/// (DeepGEMM per-token recipe, granularity 1x`group`). `values` receives
/// the E4M3 bytes and `scales` the UE8M0 bytes. Mirrors the fp8 crate's
/// specified `per_token_cast_row`; used by the fused attention epilogue.
pub(crate) fn quantize_e4m3_e8m0_groups(
    dims: &[f32],
    group: usize,
    values: &mut [u8],
    scales: &mut [u8],
) -> Result<()> {
    ensure!(
        group > 0 && dims.len() % group == 0,
        InvalidArgument,
        "row length {} is not divisible by group {group}",
        dims.len()
    );
    ensure!(
        values.len() == dims.len() && scales.len() == dims.len() / group,
        DimensionMismatch,
        "cast buffers have sizes {} / {}, expected {} / {}",
        values.len(),
        scales.len(),
        dims.len(),
        dims.len() / group
    );
    for g in 0..dims.len() / group {
        let slice = &dims[g * group..(g + 1) * group];
        let scale_byte = fp8_group_scale(slice)?;
        let s = e8m0_to_f32(scale_byte);
        let out_vals = &mut values[g * group..(g + 1) * group];
        for (b, &v) in out_vals.iter_mut().zip(slice.iter()) {
            *b = f32_to_e4m3_rne(v / s, true);
        }
        scales[g] = scale_byte;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_row_sizes_match_flashmla() {
        assert_eq!(row_bytes(CacheFormat::Bf16, 512), Some(1024));
        assert_eq!(row_bytes(CacheFormat::Fp8Group32, 512), Some(528));
        assert_eq!(row_bytes(CacheFormat::Fp4Group16, 512), Some(288));
        // Dimension-generic sizes.
        assert_eq!(row_bytes(CacheFormat::Fp8Group32, 64), Some(66));
        assert_eq!(row_bytes(CacheFormat::Fp4Group16, 64), Some(36));
        assert_eq!(row_bytes(CacheFormat::Bf16, 3), Some(6));
    }

    #[test]
    fn fp8_layout_scale_position_and_exactness() {
        // Values on the E4M3 lattice with amax = 448 give scale exactly 1
        // (448/448 = 2^0, Ceil keeps exact powers) -> lossless round trip.
        let mut dims = [0.0f32; 512];
        for (i, v) in dims.iter_mut().enumerate() {
            *v = [0.5f32, -1.0, 2.25, -448.0, 3.5, 0.0][i % 6];
        }
        let mut bytes = [0u8; 528];
        quantize_row(CacheFormat::Fp8Group32, &dims, &mut bytes).unwrap();
        // Scale bytes occupy [512, 528).
        for g in 0..16 {
            assert_eq!(bytes[512 + g], 127, "group {g} scale must be 2^0");
        }
        let mut back = [0.0f32; 512];
        dequantize_row(CacheFormat::Fp8Group32, &bytes, &mut back);
        assert_eq!(&back[..], &dims[..]);
    }

    #[test]
    fn fp8_scale_is_ceil_log2_amax_over_448() {
        // amax = 100 -> 100/448 = 0.223.. -> ceil(log2) = -2 -> scale 2^-2.
        let mut group = [0.0f32; 32];
        group[0] = 100.0;
        group[7] = -50.0;
        assert_eq!(fp8_group_scale(&group).unwrap(), 125); // 127 - 2
        // amax = 449 -> 449/448 = 1.0022 -> ceil(log2) = 1 -> scale 2^1.
        group[0] = 449.0;
        assert_eq!(fp8_group_scale(&group).unwrap(), 128);
        // All-zero group -> byte 0 (2^-127); zeros dequantize to zeros.
        let zeros = [0.0f32; 32];
        assert_eq!(fp8_group_scale(&zeros).unwrap(), 0);
        let mut bytes = [0u8; 528];
        quantize_row(CacheFormat::Fp8Group32, &[0.0f32; 512], &mut bytes).unwrap();
        assert!(bytes.iter().all(|&b| b == 0));
        let mut back = [0.0f32; 512];
        dequantize_row(CacheFormat::Fp8Group32, &bytes, &mut back);
        assert!(back.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn fp4_nibble_order_and_scale_floor() {
        // amax = 6 -> scale = e4m3(6/6) = 1.0; lattice values round-trip.
        let mut dims = [0.0f32; 16];
        dims[0] = 6.0; // even index -> LOW nibble, code 7
        dims[1] = 0.5; // odd index -> high nibble, code 1
        let mut bytes = [0u8; 288];
        quantize_row(CacheFormat::Fp4Group16, &{
            let mut v = [0.0f32; 512];
            v[..16].copy_from_slice(&dims);
            v
        }, &mut bytes)
        .unwrap();
        assert_eq!(bytes[0], 7 | (1 << 4));
        assert_eq!(bytes[256], 0x38); // e4m3(1.0)
        // Tiny amax: 0.03/6 = 0.005 -> RNE -> subnormal -> floored to 2^-6.
        let tiny = [0.03f32; 16];
        assert!((fp4_group_scale(&tiny).unwrap() - E4M3_MIN_NORMAL).abs() < 1e-12);
    }

    #[test]
    fn round_trip_error_bounds_random_rows() {
        let mut rng = deepinfer_core::rng::Philox::new(0xF48, 1);
        for _ in 0..25 {
            let dims: Vec<f32> = (0..512).map(|_| rng.uniform_f32(-1.0, 1.0)).collect();
            let amax = dims.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            for (format, tol) in [
                (CacheFormat::Fp8Group32, 0.13 * 1.0 + 1e-6),
                (CacheFormat::Fp4Group16, 3.0),
            ] {
                let n = row_bytes(format, 512).unwrap();
                let mut bytes = vec![0u8; n];
                quantize_row(format, &dims, &mut bytes).unwrap();
                let mut back = vec![0.0f32; 512];
                dequantize_row(format, &bytes, &mut back);
                for (a, b) in dims.iter().zip(back.iter()) {
                    assert!(
                        (a - b).abs() <= tol,
                        "{format:?}: |{a} - {b}| > {tol} (amax {amax})"
                    );
                }
            }
        }
    }

    #[test]
    fn quantization_is_deterministic() {
        let dims: Vec<f32> = (0..64).map(|i| ((i % 13) as f32 - 6.0) * 0.37).collect();
        for format in [
            CacheFormat::Bf16,
            CacheFormat::Fp8Group32,
            CacheFormat::Fp4Group16,
        ] {
            let n = row_bytes(format, 64).unwrap();
            let mut a = vec![0u8; n];
            let mut b = vec![1u8; n];
            quantize_row(format, &dims, &mut a).unwrap();
            quantize_row(format, &dims, &mut b).unwrap();
            assert_eq!(a, b, "{format:?}");
        }
    }

    #[test]
    fn bf16_round_trip_close() {
        let dims: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1 - 3.2).collect();
        let mut bytes = vec![0u8; 128];
        quantize_row(CacheFormat::Bf16, &dims, &mut bytes).unwrap();
        let mut back = vec![0.0f32; 64];
        dequantize_row(CacheFormat::Bf16, &bytes, &mut back);
        for (a, b) in dims.iter().zip(back.iter()) {
            // bf16 mantissa step at |x| in [2,4) is 2^-6 -> max RNE err 2^-7.
            assert!((a - b).abs() <= 0.008, "{a} vs {b}");
        }
    }

    #[test]
    fn group_cast_matches_manual_recipe() {
        let row = [100.0f32, -50.0, 25.0, 0.0, 449.0, -1.0, 2.0, 3.0];
        let mut values = [0u8; 8];
        let mut scales = [0u8; 1];
        quantize_e4m3_e8m0_groups(&row, 8, &mut values, &mut scales).unwrap();
        assert_eq!(scales[0], 128); // amax 449 -> ceil(log2(449/448)) = 1
        let s = e8m0_to_f32(128);
        for (b, &v) in values.iter().zip(row.iter()) {
            assert_eq!(*b, f32_to_e4m3_rne(v / s, true));
        }
        // Non-multiple lengths are rejected.
        assert!(quantize_e4m3_e8m0_groups(&row, 3, &mut values, &mut scales).is_err());
    }
}
