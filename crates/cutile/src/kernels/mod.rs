//! cuTile Rust kernel modules for DeepInfer-rs.
//!
//! Each submodule is a `#[cutile::module]` block whose `#[cutile::entry]`
//! functions are GPU kernels authored in the cuTile DSL and JIT-compiled
//! through CUDA Tile IR at first launch. Kernel semantics are ported
//! one-to-one from the CPU reference crates (`deepinfer-mla`,
//! `deepinfer-gemm`, `deepinfer-dsa`, `deepinfer-moe`, …), which remain the
//! golden models for the GPU-gated tests in `tests/golden.rs`.
//!
//! Hot kernels use the `unchecked_accesses` Tile-IR safety waiver (generated
//! as `unsafe fn`): the host wrappers in [`crate::ops`] re-validate every
//! shape, stride and divisibility precondition before launching, mirroring
//! the validate-then-compute structure of the CPU crates.

pub mod attention;
pub mod dsa;
pub mod elementwise;
pub mod helpers;
pub mod gemm;
pub mod kvcache;
pub mod moe;
pub mod norm;
pub mod quant;
pub mod rope;
pub mod sampler;
pub use attention::*;
pub use dsa::*;
pub use elementwise::*;
pub use gemm::*;
pub use kvcache::*;
pub use moe::*;
pub use norm::*;
pub use quant::*;
pub use rope::*;
pub use sampler::*;
