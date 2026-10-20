//! Tensor element data types.
//!
//! The quantized formats (`E4M3`, `E5M2`, `E8M0`, `E2M1`) follow the OCP FP8
//! and MX micro-scaling specifications, matching the formats used by
//! DeepSeek's DeepGEMM / FlashMLA kernels — including the DeepSeek-V4.1
//! FP8 (528 B/token) and FP4 (288 B/token) KV-cache layouts.

/// Element data types supported across the DeepInfer-rs stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DType {
    /// IEEE 754 binary32.
    F32,
    /// IEEE 754 binary16.
    F16,
    /// bfloat16.
    BF16,
    /// OCP FP8 E4M3FN — no infinities, largest finite value 448, NaN at `0x7F`/`0xFF`.
    E4M3,
    /// OCP FP8 E5M2 — has infinities and NaN, largest finite value 57344.
    E5M2,
    /// Unsigned 8-bit exponent `2^(x-127)` (MX scale format; `0xFF` is NaN).
    E8M0,
    /// OCP FP4 E2M1 — values `±{0, 0.5, 1, 1.5, 2, 3, 4, 6}`.
    E2M1,
    /// Signed 8-bit integer.
    I8,
    /// Signed 16-bit integer.
    I16,
    /// Signed 32-bit integer.
    I32,
    /// Signed 64-bit integer.
    I64,
    /// Unsigned 8-bit integer.
    U8,
    /// Unsigned 16-bit integer.
    U16,
    /// Boolean stored as one byte (`0` or `1`).
    Bool,
}

impl DType {
    /// Number of bytes per element when stored inside a [`crate::Tensor`].
    ///
    /// `E2M1` is a 4-bit format; inside tensors it is stored unpacked (one
    /// byte per element). Packed representations (two values per byte, as
    /// used by the FlashMLA V4.1 FP4 KV cache) are handled explicitly by the
    /// `deepinfer-fp8` crate.
    pub const fn size_in_bytes(self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::I64 => 8,
            DType::F16 | DType::BF16 | DType::I16 | DType::U16 => 2,
            DType::E4M3 | DType::E5M2 | DType::E8M0 | DType::E2M1 | DType::I8 | DType::U8
            | DType::Bool => 1,
        }
    }

    /// `true` for floating / floating-quantized formats.
    pub const fn is_float(self) -> bool {
        matches!(
            self,
            DType::F32 | DType::F16 | DType::BF16 | DType::E4M3 | DType::E5M2 | DType::E8M0 | DType::E2M1
        )
    }

    /// Canonical short name (used in logs and errors).
    pub const fn as_str(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::E4M3 => "e4m3",
            DType::E5M2 => "e5m2",
            DType::E8M0 => "e8m0",
            DType::E2M1 => "e2m1",
            DType::I8 => "i8",
            DType::I16 => "i16",
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::U8 => "u8",
            DType::U16 => "u16",
            DType::Bool => "bool",
        }
    }

    /// Safetensors header dtype string (round-trips with [`DType::from_safetensors_str`]).
    pub const fn to_safetensors_str(self) -> &'static str {
        match self {
            DType::F32 => "F32",
            DType::F16 => "F16",
            DType::BF16 => "BF16",
            DType::E4M3 => "F8_E4M3",
            DType::E5M2 => "F8_E5M2",
            DType::E8M0 => "F8_E8M0",
            DType::E2M1 => "F4_E2M1",
            DType::I8 => "I8",
            DType::I16 => "I16",
            DType::I32 => "I32",
            DType::I64 => "I64",
            DType::U8 => "U8",
            DType::U16 => "U16",
            DType::Bool => "BOOL",
        }
    }

    /// Parses a safetensors header dtype string.
    pub fn from_safetensors_str(s: &str) -> Option<DType> {
        Some(match s {
            "F32" => DType::F32,
            "F16" => DType::F16,
            "BF16" => DType::BF16,
            "F8_E4M3" => DType::E4M3,
            "F8_E5M2" => DType::E5M2,
            "F8_E8M0" => DType::E8M0,
            "F4_E2M1" => DType::E2M1,
            "I8" => DType::I8,
            "I16" => DType::I16,
            "I32" => DType::I32,
            "I64" => DType::I64,
            "U8" => DType::U8,
            "U16" => DType::U16,
            "BOOL" => DType::Bool,
            _ => return None,
        })
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_match_formats() {
        assert_eq!(DType::F32.size_in_bytes(), 4);
        assert_eq!(DType::BF16.size_in_bytes(), 2);
        assert_eq!(DType::E4M3.size_in_bytes(), 1);
        assert_eq!(DType::E2M1.size_in_bytes(), 1);
        assert_eq!(DType::E8M0.size_in_bytes(), 1);
    }

    #[test]
    fn safetensors_round_trip() {
        for d in [
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::E4M3,
            DType::E5M2,
            DType::E8M0,
            DType::E2M1,
            DType::I8,
            DType::I16,
            DType::I32,
            DType::I64,
            DType::U8,
            DType::U16,
            DType::Bool,
        ] {
            assert_eq!(DType::from_safetensors_str(d.to_safetensors_str()), Some(d));
        }
        assert_eq!(DType::from_safetensors_str("NOPE"), None);
    }
}
