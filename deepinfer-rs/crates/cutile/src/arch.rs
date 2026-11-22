//! Target-architecture mapping for the cuTile backend.
//!
//! The target machines of record for DeepInfer-rs are the NVIDIA GB200 NVL72
//! (sm_100, 72-GPU NVLink domain) and GB300 / B300 NVL72 (Blackwell Ultra,
//! sm_103, 288 GB HBM3e), plus AMD Instinct MI335X — which has no CUDA Rust
//! path and is served by the HIP backend in `deepinfer-kernels` instead.

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

/// Selected GPU backend for a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// cuTile Rust (this crate): sm_80 … sm_121, JIT through CUDA Tile IR.
    Cutile,
    /// HIP / hipRTC (`deepinfer-kernels`): AMD Instinct, including MI335X.
    Hip,
    /// Unknown / unsupported device.
    Unsupported,
}

/// A CUDA compute capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd)]
pub struct ComputeCapability {
    /// Major version (e.g. 10 for Blackwell sm_100).
    pub major: u8,
    /// Minor version.
    pub minor: u8,
}

impl ComputeCapability {
    /// Constructs a compute capability from its (major, minor) pair.
    pub fn new(major: u8, minor: u8) -> Self {
        Self { major, minor }
    }

    /// Combined `sm` number (e.g. `103` for sm_103).
    pub fn sm(&self) -> u32 {
        u32::from(self.major) * 10 + u32::from(self.minor)
    }

    /// True for GB200-class datacenter Blackwell (sm_100).
    pub fn is_gb200(&self) -> bool {
        self.major == 10 && self.minor == 0
    }

    /// True for GB300 / B300 Blackwell Ultra (sm_103).
    pub fn is_gb300(&self) -> bool {
        self.major == 10 && self.minor == 3
    }

    /// Blackwell or newer: block-scaled MMA (MXFP8 / MXFP4) and FP4 packing
    /// are available (Tile IR 13.3 + sm_100+).
    pub fn has_block_scaled_mma(&self) -> bool {
        self.major >= 10
    }

    /// Ampere and later — the oldest family cuTile supports.
    pub fn is_supported_nvidia(&self) -> bool {
        self.major >= 8
    }
}

impl fmt::Display for ComputeCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sm_{}{}", self.major, self.minor)
    }
}

/// Maps a CUDA device name (as reported by the driver) to its compute
/// capability when the naming is unambiguous; `None` otherwise (the driver's
/// reported attribute is authoritative and should be preferred). AMD
/// Instinct names map to `None` — they carry a `gcnArchName`, not a CC, and
/// select the HIP backend in [`backend_for`].
pub fn cc_from_device_name(name: &str) -> Option<ComputeCapability> {
    let cc = match name {
        "NVIDIA GB200" | "NVIDIA B200" | "NVIDIA B100" => ComputeCapability::new(10, 0),
        "NVIDIA GB300" | "NVIDIA B300" => ComputeCapability::new(10, 3),
        "NVIDIA DGX Spark" | "NVIDIA GB10" => ComputeCapability::new(12, 1),
        "NVIDIA GeForce RTX 5090" | "NVIDIA GeForce RTX 5080" => ComputeCapability::new(12, 0),
        "NVIDIA H100" | "NVIDIA H100 80GB HBM3" | "NVIDIA H200" => ComputeCapability::new(9, 0),
        "NVIDIA A100-SXM4-40GB" | "NVIDIA A100-SXM4-80GB" | "NVIDIA A100" => {
            ComputeCapability::new(8, 0)
        }
        _ => return None,
    };
    Some(cc)
}

/// Chooses the backend for a device.
///
/// `cc` is `None` for non-CUDA (AMD) devices, where the driver reports a
/// `gcnArchName` instead.
pub fn backend_for(name: &str, cc: Option<ComputeCapability>) -> Backend {
    if let Some(cc) = cc {
        if cc.is_supported_nvidia() {
            return Backend::Cutile;
        }
        return Backend::Unsupported;
    }
    if name.contains("AMD Instinct") || name.contains("gfx") {
        return Backend::Hip;
    }
    Backend::Unsupported
}

/// Env override name for [`force_bf16_fallback`].
pub const FORCE_BF16_FALLBACK_ENV: &str = "DEEPINFER_FORCE_BF16_GEMM";

static FORCE_BF16: AtomicBool = AtomicBool::new(false);
static INITIALIZED: AtomicU8 = AtomicU8::new(0);

/// When enabled, FP8 block-scaled GEMM requests degrade to the BF16 path
/// (dequantize weights on host) instead of erroring on architectures without
/// block-scaled MMA. Initialized once from `DEEPINFER_FORCE_BF16_GEMM=1`.
pub fn force_bf16_fallback() -> bool {
    if INITIALIZED
        .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
    {
        let v = std::env::var(FORCE_BF16_FALLBACK_ENV)
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        FORCE_BF16.store(v, Ordering::Release);
    }
    FORCE_BF16.load(Ordering::Acquire)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cc_mapping() {
        assert_eq!(
            cc_from_device_name("NVIDIA GB200"),
            Some(ComputeCapability::new(10, 0))
        );
        assert_eq!(
            cc_from_device_name("NVIDIA GB300"),
            Some(ComputeCapability::new(10, 3))
        );
        assert_eq!(
            cc_from_device_name("NVIDIA B100"),
            Some(ComputeCapability::new(10, 0))
        );
        assert_eq!(
            cc_from_device_name("NVIDIA H100 80GB HBM3"),
            Some(ComputeCapability::new(9, 0))
        );
        assert!(cc_from_device_name("Strawberry").is_none());
        assert!(cc_from_device_name("AMD Instinct MI335X").is_none());
    }

    #[test]
    fn backend_selection() {
        assert_eq!(
            backend_for("NVIDIA GB200", Some(ComputeCapability::new(10, 0))),
            Backend::Cutile
        );
        assert_eq!(backend_for("AMD Instinct MI335X", None), Backend::Hip);
        assert_eq!(
            backend_for("Old GPU", Some(ComputeCapability::new(6, 1))),
            Backend::Unsupported
        );
    }

    #[test]
    fn features() {
        let gb200 = ComputeCapability::new(10, 0);
        let gb300 = ComputeCapability::new(10, 3);
        assert!(gb200.has_block_scaled_mma() && gb300.has_block_scaled_mma());
        assert!(!ComputeCapability::new(9, 0).has_block_scaled_mma());
        assert!(gb200.is_gb200() && !gb300.is_gb200());
        assert!(gb300.is_gb300() && !gb200.is_gb300());
        assert_eq!(gb300.to_string(), "sm_103");
        assert_eq!(gb300.sm(), 103);
    }
}
