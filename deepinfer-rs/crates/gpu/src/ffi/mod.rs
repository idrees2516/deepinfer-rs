//! Vendor FFI shims (CUDA driver, NVRTC, HIP, hipRTC).
//!
//! This module and its children contain the audited unsafe code of the
//! workspace: every `extern "C"` block, pointer cast and call site carries a
//! SAFETY comment. See `dl.rs` for the loader contract and docs/GPU.md for
//! the full safety model.
#![allow(unsafe_code)]

pub mod cuda;
pub mod hip;
pub mod hiprtc;
pub mod nvrtc;
