//! Quantization recipes: DeepGEMM activation/weight casting and the
//! FlashMLA V4.1 KV-cache row formats.
//!
//! All recipes share two conventions:
//!
//! * **Scale selection** — for a group with maximum absolute value `amax`,
//!   the UE8M0 scale is `2^ceil(log2(amax / 448))` (byte via
//!   [`deepinfer_core::format::f32_to_e8m0`] with `E8M0Round::Ceil`),
//!   clamped to `[2^-127, 2^127]`; `amax == 0` yields `2^-127` (byte 0).
//!   The ceil policy guarantees `amax / scale <= 448`, the E4M3 maximum.
//! * **Non-finite inputs** — `NaN` maps to `+0` (no NaN encodings in the
//!   payload), `±Inf` saturates to `±448` after scaling. Never panics.

use deepinfer_core::format::{
    e2m1_to_f32, e4m3_to_f32, e8m0_to_f32, f32_to_e2m1_rne, f32_to_e4m3_rne, f32_to_e8m0,
    E8M0Round,
};
use deepinfer_core::{ensure, Error, Result};

/// FlashMLA V4.1 FP8 KV-cache row size in bytes (512 e4m3 + 16 e8m0).
const KV_FP8_BYTES: usize = 528;
/// FlashMLA V4.1 FP8 KV-cache latent dimension count.
const KV_DIMS: usize = 512;
/// FlashMLA V4.1 FP8 KV-cache scale group (dims per e8m0 scale byte).
const KV_FP8_GROUP: usize = 32;
/// FlashMLA V4.1 FP4 KV-cache row size in bytes (256 e2m1 + 32 e4m3).
const KV_FP4_BYTES: usize = 288;
/// FlashMLA V4.1 FP4 KV-cache scale group (dims per e4m3 scale byte).
const KV_FP4_GROUP: usize = 16;
/// Largest E4M3 magnitude (OCP E4M3FN has no infinities).
const E4M3_MAX: f32 = 448.0;
/// Largest E2M1 magnitude.
const E2M1_MAX: f32 = 6.0;
/// Smallest positive E4M3 magnitude (min subnormal, `2^-9`).
const E4M3_MIN_SUBNORMAL: f32 = 0.001_953_125;

/// Maximum absolute value of `seg`, with `NaN` mapped to `0.0` (a single
/// NaN must not poison the group scale). May return `+inf` when the group
/// contains an infinite entry; callers clamp before scale selection.
fn group_amax(seg: &[f32]) -> f32 {
    let mut amax = 0.0f32;
    for &v in seg {
        let a = if v.is_nan() { 0.0 } else { v.abs() };
        if a > amax {
            amax = a;
        }
    }
    amax
}

/// UE8M0 scale byte for a group amax: `2^ceil(log2(amax/448))` clamped to
/// `[2^-127, 2^127]`. `amax <= 0` (all-zero group) yields byte 0 (`2^-127`).
/// A non-finite amax (group containing `±Inf`) is clamped to `f32::MAX`, so
/// the scale stays finite and infinite entries saturate to `±448` after
/// scaling instead of producing NaN encodings.
fn ue8m0_scale_byte(amax: f32) -> u8 {
    let amax = if amax.is_finite() { amax } else { f32::MAX };
    if amax <= 0.0 {
        return 0; // 2^-127
    }
    // amax / 448 is finite and positive for every finite amax > 0, so the
    // Err branch is unreachable; keep it as a defensive no-panic fallback.
    match f32_to_e8m0(amax / E4M3_MAX, E8M0Round::Ceil) {
        Ok(byte) => byte,
        Err(_) => 0,
    }
}

/// Maps `NaN` to `+0` before value quantization (documented no-panic
/// policy: quantized payloads never contain the E4M3 NaN encoding).
#[inline]
fn sanitize(v: f32) -> f32 {
    if v.is_nan() {
        0.0
    } else {
        v
    }
}

/// Per-token-group E4M3 quantization with UE8M0 scales (DeepGEMM activation
/// recipe, granularity `1 x group` — `128` by default).
///
/// `x` is `rows x cols` row-major with `cols` = K. For every (row, group)
/// pair the scale is `2^ceil(log2(amax/448))` as an e8m0 byte and each value
/// is `f32_to_e4m3_rne(x / scale, /* saturate */ true)`.
///
/// Returns `(values, scales)` where `values.len() == x.len()` and
/// `scales.len() == rows * (cols / group)` laid out row-major as
/// `[row][group]` (the scale of value `i` is `scales[i / group]`).
///
/// # Errors
///
/// - `group == 0`, `cols == 0`, `x.len() % cols != 0`, or
///   `cols % group != 0` (DeepGEMM requires K aligned to the granularity).
/// - The scale-buffer length overflows `usize` (DoS guard).
pub fn quantize_per_token_group(x: &[f32], cols: usize, group: usize) -> Result<(Vec<u8>, Vec<u8>)> {
    ensure!(group > 0, InvalidArgument, "group must be > 0, got {group}");
    ensure!(cols > 0, InvalidArgument, "cols must be > 0, got {cols}");
    ensure!(
        x.len() % cols == 0,
        DimensionMismatch,
        "x has {} elements, not a multiple of cols = {cols}",
        x.len()
    );
    ensure!(
        cols % group == 0,
        InvalidArgument,
        "cols = {cols} must be a multiple of group = {group}"
    );
    let rows = x.len() / cols;
    let groups_per_row = cols / group;
    let n_scales = row_checked_mul(rows, groups_per_row, "scale count")?;
    let mut values = vec![0u8; x.len()];
    let mut scales = vec![0u8; n_scales];
    for r in 0..rows {
        for g in 0..groups_per_row {
            let base = r * cols + g * group;
            let seg = &x[base..base + group];
            let scale_byte = ue8m0_scale_byte(group_amax(seg));
            let scale = e8m0_to_f32(scale_byte);
            scales[r * groups_per_row + g] = scale_byte;
            for (kk, &v) in seg.iter().enumerate() {
                values[base + kk] = f32_to_e4m3_rne(sanitize(v) / scale, true);
            }
        }
    }
    Ok((values, scales))
}

/// Dequantizes [`quantize_per_token_group`] output:
/// `out[i] = e4m3_to_f32(v[i]) * e8m0_to_f32(sf[i / group])`.
///
/// # Errors
///
/// - `group == 0`, `cols == 0`, `v.len() % cols != 0`, `cols % group != 0`,
///   or `sf.len() != (v.len() / cols) * (cols / group)`.
pub fn dequantize_per_token_group(
    v: &[u8],
    sf: &[u8],
    cols: usize,
    group: usize,
) -> Result<Vec<f32>> {
    ensure!(group > 0, InvalidArgument, "group must be > 0, got {group}");
    ensure!(cols > 0, InvalidArgument, "cols must be > 0, got {cols}");
    ensure!(
        v.len() % cols == 0,
        DimensionMismatch,
        "v has {} elements, not a multiple of cols = {cols}",
        v.len()
    );
    ensure!(
        cols % group == 0,
        InvalidArgument,
        "cols = {cols} must be a multiple of group = {group}"
    );
    let rows = v.len() / cols;
    let groups_per_row = cols / group;
    let n_scales = row_checked_mul(rows, groups_per_row, "scale count")?;
    ensure!(
        sf.len() == n_scales,
        DimensionMismatch,
        "sf has {} scales, expected rows * (cols/group) = {n_scales}",
        sf.len()
    );
    let mut out = vec![0.0f32; v.len()];
    // cols % group == 0 implies flat index / group == row * (cols/group) + col/group.
    for i in 0..v.len() {
        out[i] = e4m3_to_f32(v[i]) * e8m0_to_f32(sf[i / group]);
    }
    Ok(out)
}

/// Per-block weight quantization (DeepGEMM weight recipe, granularity
/// `block_n x block_k` — `128 x 128` by default).
///
/// `b` is `[n, k]` row-major. For every `block_n x block_k` tile the scale
/// is `2^ceil(log2(amax/448))` as an e8m0 byte and each value is
/// `f32_to_e4m3_rne(b / scale, true)`. Edge tiles are partial (ceil
/// semantics): the scale grid is `[ceil(n/block_n)][ceil(k/block_k)]`
/// row-major, so the scale of value `(r, c)` is
/// `sf[(r / block_n) * ceil(k/block_k) + c / block_k]`.
///
/// Returns `(values, scales)` with `values.len() == n * k` (same `[n, k]`
/// layout as the input).
///
/// # Errors
///
/// - `block_n == 0` or `block_k == 0`.
/// - `b.len() != n * k` or the scale count overflows `usize`.
pub fn quantize_weight_per_block(
    b: &[f32],
    n: usize,
    k: usize,
    block_n: usize,
    block_k: usize,
) -> Result<(Vec<u8>, Vec<u8>)> {
    ensure!(block_n > 0, InvalidArgument, "block_n must be > 0, got {block_n}");
    ensure!(block_k > 0, InvalidArgument, "block_k must be > 0, got {block_k}");
    let nk = row_checked_mul(n, k, "n*k")?;
    ensure!(
        b.len() == nk,
        DimensionMismatch,
        "b has {} elements, expected n*k = {nk}",
        b.len()
    );
    let blocks_n = n.div_ceil(block_n);
    let blocks_k = k.div_ceil(block_k);
    let n_scales = row_checked_mul(blocks_n, blocks_k, "weight scale count")?;
    let mut values = vec![0u8; nk];
    let mut scales = vec![0u8; n_scales];
    for br in 0..blocks_n {
        let r0 = br * block_n;
        let r1 = (r0 + block_n).min(n);
        for bc in 0..blocks_k {
            let c0 = bc * block_k;
            let c1 = (c0 + block_k).min(k);
            let mut amax = 0.0f32;
            for r in r0..r1 {
                for &v in &b[r * k + c0..r * k + c1] {
                    let a = if v.is_nan() { 0.0 } else { v.abs() };
                    if a > amax {
                        amax = a;
                    }
                }
            }
            let scale_byte = ue8m0_scale_byte(amax);
            let scale = e8m0_to_f32(scale_byte);
            scales[br * blocks_k + bc] = scale_byte;
            for r in r0..r1 {
                for c in c0..c1 {
                    values[r * k + c] = f32_to_e4m3_rne(sanitize(b[r * k + c]) / scale, true);
                }
            }
        }
    }
    Ok((values, scales))
}

/// Dequantizes [`quantize_weight_per_block`] output:
/// `out[r][c] = e4m3_to_f32(v[r][c]) *
/// e8m0_to_f32(sf[(r/block_n) * ceil(k/block_k) + c/block_k])`.
///
/// # Errors
///
/// - `block_n == 0` or `block_k == 0`.
/// - `v.len() != n * k` or `sf.len() != ceil(n/block_n) * ceil(k/block_k)`.
pub fn dequantize_weight_per_block(
    v: &[u8],
    sf: &[u8],
    n: usize,
    k: usize,
    block_n: usize,
    block_k: usize,
) -> Result<Vec<f32>> {
    ensure!(block_n > 0, InvalidArgument, "block_n must be > 0, got {block_n}");
    ensure!(block_k > 0, InvalidArgument, "block_k must be > 0, got {block_k}");
    let nk = row_checked_mul(n, k, "n*k")?;
    ensure!(
        v.len() == nk,
        DimensionMismatch,
        "v has {} elements, expected n*k = {nk}",
        v.len()
    );
    let blocks_k = k.div_ceil(block_k);
    let n_scales = row_checked_mul(n.div_ceil(block_n), blocks_k, "weight scale count")?;
    ensure!(
        sf.len() == n_scales,
        DimensionMismatch,
        "sf has {} scales, expected ceil(n/block_n)*ceil(k/block_k) = {n_scales}",
        sf.len()
    );
    let mut out = vec![0.0f32; nk];
    for r in 0..n {
        let sf_row = &sf[(r / block_n) * blocks_k..(r / block_n + 1) * blocks_k];
        for c in 0..k {
            out[r * k + c] = e4m3_to_f32(v[r * k + c]) * e8m0_to_f32(sf_row[c / block_k]);
        }
    }
    Ok(out)
}

/// Packs 4 UE8M0 scale bytes into one `u32` (DeepGEMM SM100 packed-sf
/// layout): the first byte goes into the low 8 bits (little-endian order),
/// i.e. `packed[i] = s[4i] | s[4i+1] << 8 | s[4i+2] << 16 | s[4i+3] << 24`.
///
/// # Errors
///
/// `scales.len() % 4 != 0`.
pub fn pack_ue8m0_x4(scales: &[u8]) -> Result<Vec<u32>> {
    ensure!(
        scales.len() % 4 == 0,
        InvalidArgument,
        "scale count {} is not a multiple of 4",
        scales.len()
    );
    Ok(scales
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Reverses [`pack_ue8m0_x4`]: each `u32` expands to its four bytes in
/// little-endian order. Infallible by construction.
pub fn unpack_ue8m0_x4(packed: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(packed.len() * 4);
    for &p in packed {
        out.extend_from_slice(&p.to_le_bytes());
    }
    out
}

/// FlashMLA V4.1 FP8 KV-cache row: 512 f32 dims in, 528 bytes out.
///
/// Layout (exact, FlashMLA README "V4.1 FP8 = 512B e4m3 + 16B e8m0 scales
/// per 32"): `bytes[0..512]` are the e4m3-quantized dims (per-32-group
/// scaling, scale = `2^ceil(log2(amax/448))` e8m0, min `2^-127`) and
/// `bytes[512..528]` are the 16 e8m0 scale bytes in group order.
///
/// # Errors
///
/// `dims.len() != 512`.
pub fn quantize_kv_cache_v41_fp8(dims: &[f32]) -> Result<[u8; KV_FP8_BYTES]> {
    ensure!(
        dims.len() == KV_DIMS,
        DimensionMismatch,
        "FP8 KV row expects {KV_DIMS} dims, got {}",
        dims.len()
    );
    let mut out = [0u8; KV_FP8_BYTES];
    for g in 0..KV_DIMS / KV_FP8_GROUP {
        let seg = &dims[g * KV_FP8_GROUP..(g + 1) * KV_FP8_GROUP];
        let scale_byte = ue8m0_scale_byte(group_amax(seg));
        let scale = e8m0_to_f32(scale_byte);
        out[KV_DIMS + g] = scale_byte;
        for (kk, &v) in seg.iter().enumerate() {
            out[g * KV_FP8_GROUP + kk] = f32_to_e4m3_rne(sanitize(v) / scale, true);
        }
    }
    Ok(out)
}

/// Reverses [`quantize_kv_cache_v41_fp8`] exactly:
/// `out[i] = e4m3_to_f32(bytes[i]) * e8m0_to_f32(bytes[512 + i/32])`.
pub fn dequantize_kv_cache_v41_fp8(bytes: &[u8; KV_FP8_BYTES]) -> [f32; KV_DIMS] {
    let mut out = [0.0f32; KV_DIMS];
    for i in 0..KV_DIMS {
        out[i] = e4m3_to_f32(bytes[i]) * e8m0_to_f32(bytes[KV_DIMS + i / KV_FP8_GROUP]);
    }
    out
}

/// FlashMLA V4.1 FP4 KV-cache row: 512 f32 dims in, 288 bytes out.
///
/// Layout (exact, FlashMLA README "V4.1 FP4 = 256B e2m1 (2/byte, even low
/// nibble) + 32B e4m3 scales per 16"): the first 256 bytes hold the 512
/// e2m1 nibbles with the even-indexed dim (0, 2, 4, ...) in the LOW nibble
/// and the odd-indexed dim in the HIGH nibble of each byte. Quantization is
/// per 16 dims: `scale = e4m3_rne(amax / 6)` clamped to `[2^-9, 448]`
/// (`2^-9` is the smallest positive E4M3, `6` the largest E2M1 magnitude),
/// and the nibble is `f32_to_e2m1_rne(x / e4m3_to_f32(scale))`.
/// `bytes[256..288]` are the 32 e4m3 scale bytes in group order.
///
/// # Errors
///
/// `dims.len() != 512`.
pub fn quantize_kv_cache_v41_fp4(dims: &[f32]) -> Result<[u8; KV_FP4_BYTES]> {
    ensure!(
        dims.len() == KV_DIMS,
        DimensionMismatch,
        "FP4 KV row expects {KV_DIMS} dims, got {}",
        dims.len()
    );
    let mut out = [0u8; KV_FP4_BYTES];
    for g in 0..KV_DIMS / KV_FP4_GROUP {
        let seg = &dims[g * KV_FP4_GROUP..(g + 1) * KV_FP4_GROUP];
        let amax = group_amax(seg);
        // amax / 6 with clamping; NaN cannot occur (amax is 0, finite or +inf,
        // and f32::INFINITY.clamp(lo, hi) == hi).
        let scale_val = (amax / E2M1_MAX).clamp(E4M3_MIN_SUBNORMAL, E4M3_MAX);
        let scale_byte = f32_to_e4m3_rne(scale_val, true);
        let scale = e4m3_to_f32(scale_byte);
        out[256 + g] = scale_byte;
        for (kk, &v) in seg.iter().enumerate() {
            let nib = f32_to_e2m1_rne(sanitize(v) / scale);
            let idx = g * KV_FP4_GROUP + kk;
            if idx % 2 == 0 {
                out[idx / 2] = nib; // low nibble (byte was zero-initialized)
            } else {
                out[idx / 2] |= nib << 4; // high nibble
            }
        }
    }
    Ok(out)
}

/// Reverses [`quantize_kv_cache_v41_fp4`] exactly:
/// `out[i] = e2m1_to_f32(nibble(i)) * e4m3_to_f32(bytes[256 + i/16])` with
/// the even-indexed dim read from the low nibble of `bytes[i/2]`.
pub fn dequantize_kv_cache_v41_fp4(bytes: &[u8; KV_FP4_BYTES]) -> [f32; KV_DIMS] {
    let mut out = [0.0f32; KV_DIMS];
    for i in 0..KV_DIMS {
        let nib = if i % 2 == 0 {
            bytes[i / 2] & 0x0F
        } else {
            bytes[i / 2] >> 4
        };
        out[i] = e2m1_to_f32(nib) * e4m3_to_f32(bytes[256 + i / KV_FP4_GROUP]);
    }
    out
}

/// Per-token cast used by the fused attention epilogue
/// (tile-kernels `quant.per_token_cast` semantics): single-row convenience
/// wrapper over [`quantize_per_token_group`] with `cols = x.len()` and the
/// requested granularity, returning `(e4m3 values, e8m0 scales)`.
///
/// # Errors
///
/// `x` is empty (a token must have at least one dim) or `group == 0` or
/// `x.len() % group != 0`.
pub fn per_token_cast_row(x: &[f32], group: usize) -> Result<(Vec<u8>, Vec<u8>)> {
    ensure!(!x.is_empty(), InvalidArgument, "per-token cast needs a non-empty row");
    quantize_per_token_group(x, x.len(), group)
}

/// Checked `usize` multiplication with a capacity error (DoS guard).
fn row_checked_mul(a: usize, b: usize, what: &str) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| Error::CapacityExceeded(format!("{what} overflows: {a} * {b}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepinfer_core::prop::{prop, random_vec};
    use deepinfer_core::rng::Philox;

    // ------------------------------------------------------------------
    // Activation recipe (DeepGEMM 1x128)
    // ------------------------------------------------------------------

    #[test]
    fn activation_recipe_exact_bytes_and_scale_positions() {
        // 2 rows x 256 cols, group 128.
        let mut x = vec![0.0f32; 512];
        // Row 0, group 0: amax 224 -> scale 2^-1 (byte 126); 224/0.5 = 448.
        x[0] = 224.0;
        for i in 1..128 {
            x[i] = -224.0;
        }
        // Row 0, group 1: amax 448 -> scale 2^0 (byte 127).
        for i in 128..256 {
            x[i] = 448.0;
        }
        // Row 1: all zeros -> scale 2^-127 (byte 0).
        let (v, sf) = quantize_per_token_group(&x, 256, 128).unwrap();
        assert_eq!(sf, vec![126u8, 127, 0, 0], "scale bytes / positions");
        assert_eq!(v[0], 0x7E, "+224 / 0.5 = +448");
        assert_eq!(v[1], 0xFE, "-224 / 0.5 = -448");
        assert_eq!(v[127], 0xFE);
        for i in 128..256 {
            assert_eq!(v[i], 0x7E);
        }
        for i in 256..512 {
            assert_eq!(v[i], 0x00);
        }
        // Round trip is exact for these representable values.
        let deq = dequantize_per_token_group(&v, &sf, 256, 128).unwrap();
        for i in 0..512 {
            assert_eq!(deq[i], x[i], "index {i}");
        }
    }

    #[test]
    fn activation_amax_above_range_saturates_gracefully() {
        // amax 897: 897/448 is in (2, 4) -> ceil(log2) = 2 -> scale 4 (byte 129).
        let mut x = vec![896.0f32; 128];
        x[0] = 897.0;
        let (v, sf) = quantize_per_token_group(&x, 128, 128).unwrap();
        assert_eq!(sf[0], 129);
        // 897/4 = 224.25 and 896/4 = 224 -> nearest e4m3 is 224.
        assert_eq!(e4m3_to_f32(v[0]), 224.0);
        assert_eq!(e4m3_to_f32(v[1]), 224.0);
        let deq = dequantize_per_token_group(&v, &sf, 128, 128).unwrap();
        let err = (deq[0] - 897.0).abs() / 897.0;
        assert!(err <= 0.13, "relative error {err}");
    }

    #[test]
    fn activation_validation_errors() {
        assert!(quantize_per_token_group(&[1.0; 5], 2, 4).is_err()); // len % cols
        assert!(quantize_per_token_group(&[1.0; 4], 3, 1).is_err()); // len % cols
        assert!(quantize_per_token_group(&[1.0; 4], 0, 4).is_err()); // cols == 0
        assert!(quantize_per_token_group(&[1.0; 4], 4, 0).is_err()); // group == 0
        assert!(quantize_per_token_group(&[1.0; 130], 130, 128).is_err()); // cols % group
        assert!(dequantize_per_token_group(&[0; 4], &[0; 1], 4, 2).is_err()); // sf len
        assert!(dequantize_per_token_group(&[0; 4], &[0; 2], 3, 1).is_err()); // len % cols
    }

    #[test]
    fn activation_empty_batch_is_ok() {
        let (v, sf) = quantize_per_token_group(&[], 128, 128).unwrap();
        assert!(v.is_empty() && sf.is_empty());
        let deq = dequantize_per_token_group(&[], &[], 128, 128).unwrap();
        assert!(deq.is_empty());
    }

    #[test]
    fn activation_deterministic_bitwise() {
        let mut rng = Philox::new(0xACC, 1);
        let x = random_vec(&mut rng, 5 * 256, 400.0);
        let (v1, s1) = quantize_per_token_group(&x, 256, 128).unwrap();
        let (v2, s2) = quantize_per_token_group(&x, 256, 128).unwrap();
        assert_eq!(v1, v2);
        assert_eq!(s1, s2);
    }

    #[test]
    fn prop_activation_round_trip_bound() {
        prop("fp8-activation-roundtrip", 0xF00D, 120, |_i, rng| {
            let rows = rng.uniform_usize(1, 4);
            let cols = 128 * rng.uniform_usize(1, 3);
            let x = random_vec(rng, rows * cols, 400.0);
            let (v, sf) = quantize_per_token_group(&x, cols, 128)?;
            ensure!(sf.len() == rows * cols / 128, Internal, "scale count");
            let deq = dequantize_per_token_group(&v, &sf, cols, 128)?;
            for (j, (&d, &orig)) in deq.iter().zip(x.iter()).enumerate() {
                let s = e8m0_to_f32(sf[j / 128]);
                // Normal-region relative error <= 2^-4; the subnormal region
                // adds an absolute floor of one subnormal step.
                let bound = (0.13 * orig.abs()).max(s * 0.001_953_125);
                ensure!(
                    (d - orig).abs() <= bound,
                    Internal,
                    "elem {j}: |{d} - {orig}| > {bound} (scale {s})"
                );
            }
            Ok(())
        });
    }

    // ------------------------------------------------------------------
    // Weight recipe (DeepGEMM 128x128)
    // ------------------------------------------------------------------

    #[test]
    fn weight_recipe_layout_and_partial_blocks() {
        // n = 130, k = 128, 128x128 blocks -> ceil grid 2 x 1; the second
        // block row covers only rows 128..130.
        let mut b = vec![0.0f32; 130 * 128];
        for c in 0..128 {
            b[c] = 448.0; // block (0,0): amax 448 -> scale 2^0
        }
        for c in 0..128 {
            b[128 * 128 + c] = 224.0; // block (1,0): amax 224 -> scale 2^-1
            b[129 * 128 + c] = -224.0;
        }
        let (v, sf) = quantize_weight_per_block(&b, 130, 128, 128, 128).unwrap();
        assert_eq!(sf.len(), 2);
        assert_eq!(sf[0], 127);
        assert_eq!(sf[1], 126);
        assert_eq!(v[0], 0x7E);
        assert_eq!(v[128 * 128], 0x7E, "224 / 0.5 = 448");
        assert_eq!(v[129 * 128], 0xFE, "-224 / 0.5 = -448");
        let deq = dequantize_weight_per_block(&v, &sf, 130, 128, 128, 128).unwrap();
        assert_eq!(deq, b, "exact round trip on representable values");
    }

    #[test]
    fn weight_validation_errors() {
        assert!(quantize_weight_per_block(&[0.0; 10], 5, 2, 0, 1).is_err()); // block_n == 0
        assert!(quantize_weight_per_block(&[0.0; 10], 5, 2, 1, 0).is_err()); // block_k == 0
        assert!(quantize_weight_per_block(&[0.0; 9], 5, 2, 4, 4).is_err()); // len != n*k
        assert!(dequantize_weight_per_block(&[0; 4], &[0; 1], 2, 2, 1, 1).is_err()); // sf len
        assert!(dequantize_weight_per_block(&[0; 5], &[0; 4], 2, 2, 1, 1).is_err()); // len != n*k
    }

    #[test]
    fn prop_weight_round_trip_bound() {
        prop("fp8-weight-roundtrip", 0xB10C, 40, |_i, rng| {
            let n = 128 * rng.uniform_usize(1, 2);
            let k = 128 * rng.uniform_usize(1, 2);
            let b = random_vec(rng, n * k, 100.0);
            let (v, sf) = quantize_weight_per_block(&b, n, k, 128, 128)?;
            let deq = dequantize_weight_per_block(&v, &sf, n, k, 128, 128)?;
            let kb = k / 128;
            for (j, (&d, &orig)) in deq.iter().zip(b.iter()).enumerate() {
                let r = j / k;
                let c = j % k;
                let s = e8m0_to_f32(sf[(r / 128) * kb + c / 128]);
                let bound = (0.13 * orig.abs()).max(s * 0.001_953_125);
                ensure!(
                    (d - orig).abs() <= bound,
                    Internal,
                    "elem ({r},{c}): |{d} - {orig}| > {bound} (scale {s})"
                );
            }
            Ok(())
        });
    }

    // ------------------------------------------------------------------
    // UE8M0 x4 packing
    // ------------------------------------------------------------------

    #[test]
    fn pack_unpack_layout_and_round_trip() {
        assert_eq!(pack_ue8m0_x4(&[0x01, 0x02, 0x03, 0x04]).unwrap(), vec![0x0403_0201u32]);
        assert_eq!(unpack_ue8m0_x4(&[0x0403_0201]), vec![0x01, 0x02, 0x03, 0x04]);
        assert_eq!(pack_ue8m0_x4(&[]).unwrap(), Vec::<u32>::new());
        assert_eq!(unpack_ue8m0_x4(&[]), Vec::<u8>::new());
        assert!(pack_ue8m0_x4(&[1, 2, 3]).is_err());
        let bytes: Vec<u8> = (0..64u8).collect();
        let packed = pack_ue8m0_x4(&bytes).unwrap();
        assert_eq!(packed.len(), 16);
        assert_eq!(unpack_ue8m0_x4(&packed), bytes);
    }

    #[test]
    fn prop_pack_round_trip() {
        prop("fp8-pack-roundtrip", 0x9A4C, 100, |_i, rng| {
            let n = 4 * rng.uniform_usize(1, 8);
            let mut scales = vec![0u8; n];
            for s in scales.iter_mut() {
                *s = (rng.next_u32() & 0xFF) as u8;
            }
            let packed = pack_ue8m0_x4(&scales)?;
            ensure!(packed.len() == n / 4, Internal, "packed len");
            ensure!(unpack_ue8m0_x4(&packed) == scales, Internal, "round trip");
            Ok(())
        });
    }

    // ------------------------------------------------------------------
    // FlashMLA V4.1 KV-cache rows
    // ------------------------------------------------------------------

    #[test]
    fn kv_fp8_exact_byte_layout() {
        let mut dims = [0.0f32; 512];
        dims[0] = 448.0; // group 0: amax 448 -> scale byte 127, value 448 -> 0x7E
        dims[31] = -448.0;
        dims[160] = 224.0; // group 5: amax 224 -> scale byte 126, 224/0.5 = 448 -> 0x7E
        let bytes = quantize_kv_cache_v41_fp8(&dims).unwrap();
        assert_eq!(bytes.len(), 528);
        assert_eq!(bytes[0], 0x7E);
        assert_eq!(bytes[31], 0xFE);
        assert_eq!(bytes[160], 0x7E);
        for i in 0..512 {
            if ![0usize, 31, 160].contains(&i) {
                assert_eq!(bytes[i], 0x00, "value byte {i}");
            }
        }
        assert_eq!(bytes[512], 127, "group 0 scale position");
        assert_eq!(bytes[512 + 5], 126, "group 5 scale position");
        for g in 0..16 {
            if g != 0 && g != 5 {
                assert_eq!(bytes[512 + g], 0, "zero group scale {g}");
            }
        }
        let deq = dequantize_kv_cache_v41_fp8(&bytes);
        assert_eq!(deq[0], 448.0);
        assert_eq!(deq[31], -448.0);
        assert_eq!(deq[160], 224.0);
        assert_eq!(deq[161], 0.0);
        // Length validation.
        assert!(quantize_kv_cache_v41_fp8(&dims[..511]).is_err());
        assert!(quantize_kv_cache_v41_fp8(&[0.0; 513]).is_err());
    }

    #[test]
    fn kv_fp4_exact_byte_layout_and_nibble_order() {
        let mut dims = [0.0f32; 512];
        // Group 0 (dims 0..16): amax 6 -> scale e4m3(6/6) = e4m3(1.0) = 0x38.
        dims[0] = 6.0;
        dims[1] = 0.0;
        dims[2] = 3.0;
        dims[3] = -1.5;
        dims[4] = -6.0;
        dims[5] = 0.25; // below the 0.25 tie -> nibble 0
        let bytes = quantize_kv_cache_v41_fp4(&dims).unwrap();
        assert_eq!(bytes.len(), 288);
        // Even-indexed dim in the LOW nibble, odd-indexed in the HIGH nibble.
        assert_eq!(bytes[0], 0x7, "dims0=6 (low) | dims1=0 (high)");
        assert_eq!(bytes[1], 0x5 | (0xB << 4), "dims2=3 (low) | dims3=-1.5 (high)");
        assert_eq!(bytes[2], 0xF, "dims4=-6 (low) | dims5->0 (high)");
        // Scale bytes: group 0 -> 0x38; zero groups -> clamped to 2^-9 = 0x01.
        assert_eq!(bytes[256], 0x38);
        for g in 1..32 {
            assert_eq!(bytes[256 + g], 0x01, "zero-group scale {g}");
        }
        for i in 8..256 {
            assert_eq!(bytes[i], 0x00, "value byte {i}");
        }
        let deq = dequantize_kv_cache_v41_fp4(&bytes);
        assert_eq!(deq[0], 6.0);
        assert_eq!(deq[2], 3.0);
        assert_eq!(deq[3], -1.5);
        assert_eq!(deq[4], -6.0);
        // Dequant nibble order (regression for even-low / odd-high).
        let mut probe = [0.0f32; 512];
        probe[0] = 3.0; // even
        probe[1] = 1.5; // odd
        let b = quantize_kv_cache_v41_fp4(&probe).unwrap();
        // amax 3 -> scale e4m3(0.5) = 0x30; 3/0.5 = 6 -> code 7; 1.5/0.5 = 3 -> code 5.
        assert_eq!(b[256], 0x30);
        assert_eq!(b[0], 0x7 | (0x5 << 4));
        assert_eq!(dequantize_kv_cache_v41_fp4(&b)[0], 3.0);
        assert_eq!(dequantize_kv_cache_v41_fp4(&b)[1], 1.5);
        // Length validation.
        assert!(quantize_kv_cache_v41_fp4(&dims[..100]).is_err());
        assert!(quantize_kv_cache_v41_fp4(&[0.0; 513]).is_err());
    }

    #[test]
    fn kv_fp4_scale_clamps() {
        let mut dims = [0.0f32; 512];
        dims[0] = 1e-6; // amax/6 < 2^-9 -> scale clamps up to 2^-9 (byte 0x01)
        let b = quantize_kv_cache_v41_fp4(&dims).unwrap();
        assert_eq!(b[256], 0x01);
        assert_eq!(b[0], 0x0);
        let deq = dequantize_kv_cache_v41_fp4(&b);
        assert_eq!(deq[0], 0.0);
        assert!((deq[0] - 1e-6).abs() <= 3.1 * e4m3_to_f32(0x01));
        // Huge amax: scale clamps at 448 (byte 0x7E); values saturate to 6.
        let mut big = [0.0f32; 512];
        big[7] = 1e6;
        let bb = quantize_kv_cache_v41_fp4(&big).unwrap();
        assert_eq!(bb[256], 0x7E);
        assert_eq!(bb[3] >> 4, 0x7, "odd dim 7 saturates to 6 in the high nibble");
        assert_eq!(bb[3] & 0x0F, 0x0, "even dim 6 stays 0 in the low nibble");
    }

    #[test]
    fn kv_fp8_all_zeros() {
        let bytes = quantize_kv_cache_v41_fp8(&[0.0; 512]).unwrap();
        assert!(bytes.iter().all(|&b| b == 0));
        assert!(dequantize_kv_cache_v41_fp8(&bytes).iter().all(|&v| v == 0.0));
    }

    #[test]
    fn nan_inf_inputs_never_panic_and_saturate() {
        let mut x = vec![0.0f32; 128];
        x[0] = f32::NAN;
        x[1] = f32::INFINITY;
        x[2] = -f32::INFINITY;
        let (v, sf) = quantize_per_token_group(&x, 128, 128).unwrap();
        // amax: NaN -> 0, Inf -> f32::MAX -> scale 2^120 (byte 247).
        assert_eq!(sf[0], 247);
        assert_eq!(v[0], 0x00, "NaN maps to +0");
        assert_eq!(v[1], 0x7E, "+Inf saturates to +448 after scaling");
        assert_eq!(v[2], 0xFE, "-Inf saturates to -448 after scaling");
        let deq = dequantize_per_token_group(&v, &sf, 128, 128).unwrap();
        // No NaN ever; ±Inf saturates to the largest finite quantized
        // magnitude (448 * 2^120), which exceeds the f32 range on purpose.
        assert!(deq.iter().all(|z| !z.is_nan()));
        assert_eq!(deq[0], 0.0);
        assert_eq!(deq[1], f32::INFINITY);
        assert_eq!(deq[2], f32::NEG_INFINITY);
        assert!(v.iter().all(|&b| b != 0x7F && b != 0xFF), "no NaN encodings");

        let mut w = vec![0.0f32; 256];
        w[0] = f32::NAN;
        w[255] = f32::INFINITY;
        let (wv, wsf) = quantize_weight_per_block(&w, 2, 128, 128, 128).unwrap();
        assert_eq!(wsf[0], 247);
        assert_eq!(wv[0], 0x00);
        assert_eq!(wv[255], 0x7E);
        assert!(wv.iter().all(|&b| b != 0x7F && b != 0xFF), "no NaN encodings");
        assert!(dequantize_weight_per_block(&wv, &wsf, 2, 128, 128, 128)
            .unwrap()
            .iter()
            .all(|z| !z.is_nan()));

        let mut dims = [0.0f32; 512];
        dims[0] = f32::INFINITY;
        dims[1] = f32::NAN;
        let b8 = quantize_kv_cache_v41_fp8(&dims).unwrap();
        assert!(b8[..512].iter().all(|&b| b != 0x7F && b != 0xFF));
        let d8 = dequantize_kv_cache_v41_fp8(&b8);
        assert!(d8.iter().all(|z| !z.is_nan()));
        assert_eq!(d8[0], f32::INFINITY);
        assert_eq!(d8[1], 0.0);
        let b4 = quantize_kv_cache_v41_fp4(&dims).unwrap();
        // FP4 scales are clamped to <= 448, so FP4 rows always dequantize
        // to finite values (max 6 * 448).
        let d4 = dequantize_kv_cache_v41_fp4(&b4);
        assert!(d4.iter().all(|z| z.is_finite()));
        assert_eq!(d4[0], 6.0 * 448.0);
        assert_eq!(d4[1], 0.0);
    }

    #[test]
    fn prop_kv_fp8_round_trip_bound() {
        prop("fp8-kv-roundtrip", 0x528, 100, |_i, rng| {
            let dims = random_vec(rng, 512, 1000.0);
            let bytes = quantize_kv_cache_v41_fp8(&dims)?;
            let deq = dequantize_kv_cache_v41_fp8(&bytes);
            for i in 0..512 {
                let s = e8m0_to_f32(bytes[512 + i / 32]);
                let bound = (0.13 * dims[i].abs()).max(s * 0.001_953_125);
                ensure!(
                    (deq[i] - dims[i]).abs() <= bound,
                    Internal,
                    "dim {i}: |{} - {}| > {bound} (scale {s})",
                    deq[i],
                    dims[i]
                );
            }
            Ok(())
        });
    }

    #[test]
    fn prop_kv_fp4_round_trip_bound() {
        prop("fp4-kv-roundtrip", 0x288, 100, |_i, rng| {
            let dims = random_vec(rng, 512, 1000.0);
            let bytes = quantize_kv_cache_v41_fp4(&dims)?;
            let deq = dequantize_kv_cache_v41_fp4(&bytes);
            for i in 0..512 {
                let s = e4m3_to_f32(bytes[256 + i / 16]);
                ensure!(
                    (deq[i] - dims[i]).abs() <= 3.1 * s,
                    Internal,
                    "dim {i}: |{} - {}| > {} (scale {s})",
                    deq[i],
                    dims[i],
                    3.1 * s
                );
            }
            Ok(())
        });
    }

    // ------------------------------------------------------------------
    // per_token_cast_row
    // ------------------------------------------------------------------

    #[test]
    fn per_token_cast_row_matches_group_quantizer() {
        let mut rng = Philox::new(42, 0);
        let x = random_vec(&mut rng, 256, 10.0);
        let (v1, s1) = per_token_cast_row(&x, 128).unwrap();
        let (v2, s2) = quantize_per_token_group(&x, 256, 128).unwrap();
        assert_eq!(v1, v2);
        assert_eq!(s1, s2);
        assert_eq!(s1.len(), 2);
        assert!(per_token_cast_row(&[], 128).is_err());
        assert!(per_token_cast_row(&x, 0).is_err());
        assert!(per_token_cast_row(&x[..100], 128).is_err()); // 100 % 128 != 0
    }
}
