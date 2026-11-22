//! Device bootstrap helpers (cuTile / CUDA Rust runtime).

use cutile::cuda_core::Device;

/// Returns the device at `ordinal` if a CUDA device is present.
pub fn try_device(ordinal: usize) -> Option<std::sync::Arc<Device>> {
    Device::new(ordinal).ok()
}

/// Number of CUDA devices visible to the driver (0 in GPU-less sandboxes).
pub fn device_count() -> usize {
    Device::device_count().map(|n| n as usize).unwrap_or(0)
}
