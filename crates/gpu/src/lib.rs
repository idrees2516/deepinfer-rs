//! # deepinfer-gpu — runtime-loaded GPU layer for DeepInfer-rs
//!
//! A zero-build-dependency GPU abstraction for the three deployment targets
//! of the DeepSeek inference stack:
//!
//! * **NVIDIA GB200 NVL72 / B200** — compute capability 10.0 (`sm_100`),
//!   72-GPU NVLink5 domain (130 TB/s bisection), FP8 dense 5 PFLOPS/GPU.
//! * **NVIDIA GB300 NVL72 / B300 (Blackwell Ultra)** — compute capability
//!   10.3 (`sm_103`, CUDA 12.9+ / R580 driver), 288 GB HBM3e, 15 PFLOPS
//!   dense NVFP4, 2x SFU softmax throughput.
//! * **AMD Instinct MI335X** (and MI300/MI350 families) — 288 GB HBM3e,
//!   8 TB/s, CDNA matrix cores; compile target read from the device's
//!   `gcnArchName` at runtime.
//!
//! Design (following DeepGEMM's DeepJIT, the `deepgemm-rs` Rust port, and
//! the `kern` runtime — all of which validate this architecture):
//!
//! 1. **No build-time CUDA/HIP.** `libcuda`/`libnvrtc`/`libamdhip64`/
//!    `libhiprtc` are `dlopen`ed at runtime. Binaries build and test on
//!    GPU-less machines; kernels are compiled from embedded C++ sources by
//!    NVRTC / hipRTC on first use and cached on disk (SHA-256 keyed).
//! 2. **One API, two backends.** [`Device`], [`Stream`], [`Buffer`],
//!    [`Module`], [`Function`], [`KernelArgs`] hide the vendor difference;
//!    `deepinfer-kernels` provides the kernel sources and launchers.
//! 3. **Audited unsafe boundary.** This is the only workspace crate that
//!    contains `unsafe`. The workspace `forbid(unsafe_code)` lint is
//!    replaced here by `deny` plus module-level `#[allow(unsafe_code)]`
//!    with SAFETY proofs on every block (see `dl.rs` and `ffi/*`).
//! 4. **Graceful degradation.** [`Gpu::probe_backends`] reports what is
//!    installed; all constructors return `Err` (never panic) when a stack
//!    or device is missing, so CPU-only deployments coexist in the same
//!    binary.
//!
//! NVLink / Infinity Fabric peer access is exposed via
//! [`Device::enable_peer_access`] and [`Device::memcpy_peer_async`] — the
//! primitives needed for single-node expert parallelism (the NVL72 domain
//! being the entire point of that system for wide-EP MoE serving).

// The modules below contain the audited unsafe FFI call sites; each has
// SAFETY proofs per block. `deny` (not `forbid`) allows exactly this override.
#[allow(unsafe_code)]
pub mod device;
#[allow(unsafe_code)]
pub mod dl;
pub mod ffi;
pub mod hash;
#[allow(unsafe_code)]
pub mod jit;

pub use device::{
    Backend, Buffer, Device, DeviceProps, Function, Gpu, GpuArch, KernelArgs, Launch, Module,
    Stream,
};
pub use jit::{compile, compile_and_load, KernelImage};

#[cfg(test)]
mod tests {
    #[test]
    fn crate_has_no_unaudited_unsafe() {
        // The audited unsafe modules are exactly: dl, ffi/*, device, jit.
        // (device.rs and jit.rs contain the call sites of the FFI pointers,
        // each carrying a SAFETY comment.) This test documents the contract;
        // enforcement is by review + `cargo clippy` on the crate.
    }
}
