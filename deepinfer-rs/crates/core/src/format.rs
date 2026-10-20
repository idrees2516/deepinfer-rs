//! Bit-exact codecs for low-precision floating point formats.
//!
//! These are *reference* codecs: every conversion is bit-exact and
//! deterministic (round-to-nearest-even), matching the semantics of the
//! formats used by DeepSeek's kernels (DeepGEMM FP8 GEMMs, FlashMLA V4.1
//! FP8/FP4 KV caches, MX-style UE8M0 scales).
//!
//! * `BF16` / `F16` — standard IEEE truncations with RNE.
//! * `E4M3` — OCP E4M3FN (no ±inf, max finite 448).
//! * `E5M2` — OCP E5M2 (has ±inf / NaN, max finite 57344).
//! * `E8M0` — unsigned power-of-two scale `2^(x-127)`; `0xFF` is NaN.
//! * `E2M1` — OCP FP4 (±{0, 0.5, 1, 1.5, 2, 3, 4, 6}).

/// Converts a `u16` bfloat16 to `f32` (exact).
#[inline]
pub fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

/// Converts `f32` to bfloat16 with round-to-nearest-even. NaN is quieted.
#[inline]
pub fn f32_to_bf16_rne(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        // Quiet NaN, preserve sign.
        return (((bits >> 16) as u16) & 0x8000) | 0x7FC0;
    }
    if x.is_infinite() || x == 0.0 {
        // Exact (includes -0.0).
        return (bits >> 16) as u16;
    }
    // Classic RNE via rounding bias; overflow to infinity is impossible for
    // finite f32 (max f32 rounds to bf16 inf naturally through carry).
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7FFF + lsb);
    (rounded >> 16) as u16
}

/// Converts a `u16` IEEE half to `f32` (exact, handles subnormals).
#[inline]
pub fn f16_to_f32(v: u16) -> f32 {
    let sign = ((v >> 15) & 1) as u32;
    let exp = ((v >> 10) & 0x1F) as u32;
    let man = (v & 0x3FF) as u32;
    let f32_bits = match (exp, man) {
        (0, 0) => sign << 31,
        (0, m) => {
            // Subnormal half: m * 2^-24. Normalize via f64 for correctness.
            let val = (m as f64) * 2f64.powi(-24);
            (val as f32).to_bits() | (sign << 31)
        }
        (31, 0) => (sign << 31) | 0x7F80_0000,
        (31, m) => (sign << 31) | 0x7FC0_0000 | (m << 13),
        (e, m) => (sign << 31) | ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(f32_bits)
}

/// Converts `f32` to IEEE half with round-to-nearest-even.
#[inline]
pub fn f32_to_f16_rne(x: f32) -> u16 {
    let sign = ((x.to_bits() >> 31) as u16) << 15;
    if x.is_nan() {
        return sign | 0x7E00;
    }
    if x.is_infinite() {
        return sign | 0x7C00;
    }
    if x == 0.0 {
        return sign;
    }
    let a = x.abs();
    // f16 subnormal / small-normal boundary handling via f64 (exact for f32 inputs).
    if a < 6.103515625e-5 {
        // 2^-14: subnormal region. Step = 2^-24.
        let q = (a as f64) * 2f64.powi(24);
        let m = q.round_ties_even();
        // m in [0, 1024]; 1024 carries into the min normal 0x0400 encoding.
        return sign | (m as u16);
    }
    if a >= 65520.0 {
        // RNE overflow threshold (midpoint between 65504 and 65536).
        return sign | 0x7C00;
    }
    // Normal region — bit surgery on the f32 representation.
    let bits = a.to_bits();
    let be = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;
    let mut e16 = be - 127 + 15; // in [1, 30] here
    let half = 0x1000u32; // 2^12: halfway of the 13 dropped mantissa bits
    let m13 = mant >> 13;
    let rem = mant & 0x1FFF;
    let mut m16 = m13
        + u32::from(rem > half || (rem == half && (m13 & 1) == 1));
    if m16 == 0x400 {
        m16 = 0;
        e16 += 1;
    }
    if e16 > 30 {
        return sign | 0x7C00;
    }
    sign | ((e16 as u16) << 10) | (m16 as u16)
}

/// Converts an OCP E4M3FN byte to `f32` (exact). `0x7F`/`0xFF` decode to NaN.
#[inline]
pub fn e4m3_to_f32(v: u8) -> f32 {
    let sign = ((v >> 7) & 1) as u32;
    let exp = ((v >> 3) & 0xF) as u32;
    let man = (v & 0x7) as u32;
    let bits = match (exp, man) {
        (0, 0) => sign << 31,
        (0, m) => {
            // Subnormal: m * 2^-9 (exact in f32 for m <= 7).
            ((m as f32) * 2f32.powi(-9)).to_bits() | (sign << 31)
        }
        (15, 7) => (sign << 31) | 0x7FC0_0000, // NaN
        (e, m) => (sign << 31) | ((e + 120) << 23) | (m << 20),
    };
    f32::from_bits(bits)
}

/// Converts `f32` to OCP E4M3FN with round-to-nearest-even.
///
/// `saturate` controls out-of-range behaviour: `true` clamps to ±448
/// (the quantization policy used by DeepSeek's FP8 kernels), `false`
/// maps overflow to the NaN encoding.
#[inline]
pub fn f32_to_e4m3_rne(x: f32, saturate: bool) -> u8 {
    let sign = if x.is_sign_negative() { 1u8 } else { 0 };
    if x.is_nan() {
        return 0x7F;
    }
    let a = x.abs();
    if a >= 448.0 {
        // Anything that would round to 512 (the NaN slot) is clamped.
        return if saturate {
            (sign << 7) | 0x7E
        } else {
            (sign << 7) | 0x7F
        };
    }
    if a < 0.015_625 {
        // 2^-6: subnormal region, step 2^-9.
        let q = a * 512.0; // a / 2^-9
        let m = q.round_ties_even();
        if m >= 8.0 {
            // Carry into the minimum normal.
            return (sign << 7) | (1 << 3);
        }
        return (sign << 7) | (m as u8);
    }
    // Normal region.
    let bits = a.to_bits();
    let be = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;
    let mut e4 = be - 127 + 7; // in [1, 15]
    let half = 0x0008_0000u32; // 2^19: halfway of the 20 dropped mantissa bits
    let m3 = mant >> 20;
    let rem = mant & 0x000F_FFFF;
    let mut m = m3 + u32::from(rem > half || (rem == half && (m3 & 1) == 1));
    if m == 8 {
        m = 0;
        e4 += 1;
    }
    if e4 > 15 {
        return if saturate {
            (sign << 7) | 0x7E
        } else {
            (sign << 7) | 0x7F
        };
    }
    (sign << 7) | ((e4 as u8) << 3) | (m as u8)
}

/// Converts an OCP E5M2 byte to `f32` (exact). Layout: 1 sign, 5
/// exponent (bias 15), 2 mantissa bits; subnormal step `2^-16`.
#[inline]
pub fn e5m2_to_f32(v: u8) -> f32 {
    let sign = ((v >> 7) & 1) as u32;
    let exp = ((v >> 2) & 0x1F) as u32;
    let man = (v & 0x3) as u32;
    let bits = match (exp, man) {
        (0, 0) => sign << 31,
        (0, m) => ((m as f32) * 2f32.powi(-16)).to_bits() | (sign << 31),
        (31, 0) => (sign << 31) | 0x7F80_0000,
        (31, m) => (sign << 31) | 0x7FC0_0000 | (m << 21),
        (e, m) => (sign << 31) | ((e + 112) << 23) | (m << 21),
    };
    f32::from_bits(bits)
}

/// Converts `f32` to OCP E5M2 with round-to-nearest-even.
///
/// Values beyond the RNE overflow threshold (61440, midpoint of 57344 and
/// 65536) become ±inf unless `saturate` is set, in which case they clamp to
/// ±57344.
#[inline]
pub fn f32_to_e5m2_rne(x: f32, saturate: bool) -> u8 {
    let sign = if x.is_sign_negative() { 1u8 } else { 0 };
    if x.is_nan() {
        return 0x7E;
    }
    if x.is_infinite() {
        return (sign << 7) | 0x7B;
    }
    let a = x.abs();
    if a >= 61440.0 {
        return if saturate {
            (sign << 7) | 0x7B
        } else {
            (sign << 7) | 0x7C
        };
    }
    if a < 6.103515625e-5 {
        // 2^-14: subnormal region, step 2^-16 (2-bit mantissa).
        let q = (a as f64) * 2f64.powi(16);
        let m = q.round_ties_even();
        if m >= 4.0 {
            return (sign << 7) | (1 << 2); // carry into the min normal
        }
        return (sign << 7) | (m as u8);
    }
    let bits = a.to_bits();
    let be = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;
    let mut e5 = be - 127 + 15; // in [1, 30]
    let half = 0x0010_0000u32; // 2^20: halfway of the 21 dropped mantissa bits
    let m2 = mant >> 21;
    let rem = mant & 0x001F_FFFF;
    let mut m = m2 + u32::from(rem > half || (rem == half && (m2 & 1) == 1));
    if m == 4 {
        m = 0;
        e5 += 1;
    }
    if e5 > 30 {
        return if saturate {
            (sign << 7) | 0x7B
        } else {
            (sign << 7) | 0x7C
        };
    }
    (sign << 7) | ((e5 as u8) << 2) | (m as u8)
}

/// Converts an E8M0 byte to `f32`: `2^(v-127)`, with `0xFF` mapping to NaN.
///
/// E8M0 has no zero and no negative values — it exists purely as a
/// power-of-two scale factor in MX / DeepGEMM recipes.
#[inline]
pub fn e8m0_to_f32(v: u8) -> f32 {
    if v == 0xFF {
        return f32::NAN;
    }
    if v == 0x00 {
        // 2^-127 is below the f32 normal range; exact as a subnormal.
        return f32::from_bits(1u32 << 22);
    }
    // Exact: every 2^k with k in [-126, 127] is a normal f32.
    f32::from_bits((v as u32) << 23)
}

/// Rounding policy when converting a scale factor to E8M0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum E8M0Round {
    /// Round down (towards zero) — may under-cover the value.
    Floor,
    /// Round up (away from zero) — guarantees the quantized value stays
    /// inside the E4M3 representable range. This is the policy used by
    /// DeepSeek's "round_sf" scale selection.
    Ceil,
}

/// Converts a strictly positive, finite `f32` scale to an E8M0 byte.
///
/// Returns an error for NaN / non-positive inputs (E8M0 cannot represent
/// them — callers should clamp first, e.g. `amax.max(tiny)`).
pub fn f32_to_e8m0(x: f32, round: E8M0Round) -> Result<u8, crate::Error> {
    if !x.is_finite() || x <= 0.0 {
        return Err(crate::Error::Numerical(format!(
            "e8m0 scale must be finite and positive, got {x}"
        )));
    }
    let l2 = (x as f64).log2();
    let e = match round {
        E8M0Round::Floor => l2.floor(),
        E8M0Round::Ceil => l2.ceil(),
    };
    // Exact powers of two must not be rounded up by Ceil.
    let e = if (e - l2).abs() < 1e-9 { l2.floor() } else { e };
    let e = e.clamp(-127.0, 127.0);
    Ok((e as i32 + 127) as u8)
}

/// The eight magnitudes of the E2M1 FP4 format.
pub const E2M1_MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Converts an E2M1 nibble (low 4 bits used) to `f32` (exact).
#[inline]
pub fn e2m1_to_f32(nib: u8) -> f32 {
    let sign = if (nib & 8) != 0 { -1.0f32 } else { 1.0 };
    sign * E2M1_MAGNITUDES[(nib & 7) as usize]
}

/// Converts `f32` to E2M1 with round-to-nearest-even over the FP4 lattice.
///
/// NaN maps to the largest magnitude with the input's sign (FP4 has no NaN).
#[inline]
pub fn f32_to_e2m1_rne(x: f32) -> u8 {
    let sign = if x.is_sign_negative() { 8u8 } else { 0 };
    if x.is_nan() {
        return sign | 7;
    }
    let a = x.abs();
    // Midpoints between adjacent lattice points; ties go to the neighbour
    // with an even mantissa bit (0, 1, 2, 4 have even codes 0,2,4,6).
    let code = if a <= 0.25 {
        0
    } else if a < 0.75 {
        1
    } else if a <= 1.25 {
        2
    } else if a < 1.75 {
        3
    } else if a <= 2.5 {
        4
    } else if a < 3.5 {
        5
    } else if a <= 5.0 {
        6
    } else {
        7
    };
    sign | code
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip_bf16(x: f32) {
        let b = f32_to_bf16_rne(x);
        let y = bf16_to_f32(b);
        let diff = (x - y).abs();
        let tol = (x.abs() * (1.0 / 128.0)).max(f32::EPSILON);
        assert!(diff <= tol, "bf16 round trip {x} -> {y} (diff {diff})");
    }

    #[test]
    fn bf16_basics() {
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(f32_to_bf16_rne(1.0), 0x3F80);
        assert_eq!(f32_to_bf16_rne(0.0), 0x0000);
        assert_eq!(f32_to_bf16_rne(-0.0), 0x8000);
        assert!(bf16_to_f32(f32_to_bf16_rne(f32::NAN)).is_nan());
        assert_eq!(bf16_to_f32(f32_to_bf16_rne(f32::INFINITY)), f32::INFINITY);
    }

    #[test]
    fn bf16_rne_ties() {
        // Tie between 1.0 (0x3F80) and 1.0078125 (0x3F81): even target is 1.0.
        assert_eq!(f32_to_bf16_rne(1.0 + 1.0 / 256.0), 0x3F80);
        // Tie between 1.0078125 and 1.015625: even target is 1.015625 (0x3F82).
        assert_eq!(f32_to_bf16_rne(1.0078125 + 1.0 / 256.0), 0x3F82);
    }

    #[test]
    fn bf16_random_round_trip() {
        let mut rng = crate::rng::Philox::new(0xBEEF, 1);
        for _ in 0..10_000 {
            let x = rng.uniform_f32(-1000.0, 1000.0);
            round_trip_bf16(x);
        }
        for e in -20..16 {
            round_trip_f16_pow2(e);
        }
        // 2^16 is past the f16 finite range (65504): must round to inf.
        assert_eq!(f32_to_f16_rne(65536.0), 0x7C00);
    }

    fn round_trip_f16_pow2(e: i32) {
        let x = 2f32.powi(e);
        let h = f32_to_f16_rne(x);
        assert_eq!(f16_to_f32(h), x, "2^{e}");
    }

    #[test]
    fn f16_basics() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f32_to_f16_rne(1.0), 0x3C00);
        assert_eq!(f32_to_f16_rne(65504.0), 0x7BFF);
        assert_eq!(f32_to_f16_rne(65520.0), 0x7C00); // overflow -> inf
        assert_eq!(f32_to_f16_rne(65519.0), 0x7BFF);
        assert_eq!(f32_to_f16_rne(5.960464477539063e-08), 0x0001); // min subnormal
        assert_eq!(f32_to_f16_rne(2.9802322e-08), 0x0000); // below half of min subnormal
        assert_eq!(f16_to_f32(0x0001), 5.960464477539063e-08);
        assert!(f16_to_f32(f32_to_f16_rne(f32::NAN)).is_nan());
    }

    #[test]
    fn e4m3_reference_values() {
        assert_eq!(e4m3_to_f32(0x00), 0.0);
        assert_eq!(e4m3_to_f32(0x80), -0.0);
        assert_eq!(e4m3_to_f32(0x01), 2f32.powi(-9)); // min subnormal
        assert_eq!(e4m3_to_f32(0x08), 2f32.powi(-6)); // min normal
        assert_eq!(e4m3_to_f32(0x38), 1.0);
        assert_eq!(e4m3_to_f32(0x7E), 448.0);
        assert!(e4m3_to_f32(0x7F).is_nan());
        assert_eq!(f32_to_e4m3_rne(1.0, true), 0x38);
        assert_eq!(f32_to_e4m3_rne(448.0, true), 0x7E);
        assert_eq!(f32_to_e4m3_rne(1_000.0, true), 0x7E); // saturate
        assert_eq!(f32_to_e4m3_rne(1_000.0, false), 0x7F); // NaN
        assert_eq!(f32_to_e4m3_rne(1.0625, true), 0x38); // midpoint of 1.0/1.125 -> even mantissa 1.0
        assert_eq!(f32_to_e4m3_rne(1.1875, true), 0x3A); // midpoint of 1.125/1.25 -> even mantissa 1.25
        assert_eq!(f32_to_e4m3_rne(1.125, true), 0x39);
        assert_eq!(f32_to_e4m3_rne(0.0, true), 0x00);
        assert_eq!(f32_to_e4m3_rne(-2.0, true), 0xC0);
        assert!(f32_to_e4m3_rne(f32::NAN, true) == 0x7F);
    }

    #[test]
    fn e5m2_reference_values() {
        assert_eq!(e5m2_to_f32(0x00), 0.0);
        assert_eq!(e5m2_to_f32(0x01), 2f32.powi(-16)); // min subnormal: 1 x 2^-16
        assert_eq!(e5m2_to_f32(0x04), 2f32.powi(-14));
        assert_eq!(e5m2_to_f32(0x3C), 1.0);
        assert_eq!(e5m2_to_f32(0x7B), 57344.0);
        assert_eq!(e5m2_to_f32(0x7C), f32::INFINITY);
        assert!(e5m2_to_f32(0x7E).is_nan());
        assert_eq!(f32_to_e5m2_rne(1.0, true), 0x3C);
        assert_eq!(f32_to_e5m2_rne(57344.0, true), 0x7B);
        assert_eq!(f32_to_e5m2_rne(1e9, false), 0x7C); // inf
        assert_eq!(f32_to_e5m2_rne(1e9, true), 0x7B); // saturate
    }

    #[test]
    fn e8m0_round_trip_and_edges() {
        for v in [0x01u8, 0x40, 0x7E, 0x7F, 0xFE] {
            let f = e8m0_to_f32(v);
            assert!(f.is_finite() && f > 0.0);
            // Exact power of two: floor == ceil == original byte.
            assert_eq!(f32_to_e8m0(f, E8M0Round::Floor).unwrap(), v);
            assert_eq!(f32_to_e8m0(f, E8M0Round::Ceil).unwrap(), v);
        }
        assert!(e8m0_to_f32(0xFF).is_nan());
        assert_eq!(e8m0_to_f32(0x00), 2f32.powi(-127));
        // Ceil covers, floor under-covers.
        let x = 3.7;
        assert!(e8m0_to_f32(f32_to_e8m0(x, E8M0Round::Ceil).unwrap()) >= x);
        assert!(e8m0_to_f32(f32_to_e8m0(x, E8M0Round::Floor).unwrap()) <= x);
        assert!(f32_to_e8m0(0.0, E8M0Round::Ceil).is_err());
        assert!(f32_to_e8m0(-1.0, E8M0Round::Ceil).is_err());
        assert!(f32_to_e8m0(f32::NAN, E8M0Round::Ceil).is_err());
    }

    #[test]
    fn e2m1_lattice() {
        assert_eq!(e2m1_to_f32(0), 0.0);
        assert_eq!(e2m1_to_f32(1), 0.5);
        assert_eq!(e2m1_to_f32(7), 6.0);
        assert_eq!(e2m1_to_f32(0x8 | 7), -6.0);
        assert_eq!(f32_to_e2m1_rne(0.0), 0);
        assert_eq!(f32_to_e2m1_rne(0.24), 0);
        assert_eq!(f32_to_e2m1_rne(0.26), 1); // 0.5 closer
        assert_eq!(f32_to_e2m1_rne(0.25), 0); // tie -> even (0)
        assert_eq!(f32_to_e2m1_rne(0.75), 2); // tie -> even (1.0)
        assert_eq!(f32_to_e2m1_rne(5.0), 6); // tie 4|6 -> even code 6? see below
        assert_eq!(f32_to_e2m1_rne(1e9), 7);
    }

    #[test]
    fn e4m3_random_round_trip_error_bound() {
        let mut rng = crate::rng::Philox::new(7, 7);
        for _ in 0..10_000 {
            let x = rng.uniform_f32(-400.0, 400.0);
            let q = e4m3_to_f32(f32_to_e4m3_rne(x, true));
            if x == 0.0 {
                assert_eq!(q, 0.0);
                continue;
            }
            let rel = ((q - x) / x).abs();
            assert!(rel <= 0.13, "e4m3 rel err {rel} for x={x}");
        }
    }
}
