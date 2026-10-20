//! # deepinfer-runtime
//!
//! The integration crate of **DeepInfer-rs**: the DeepSeek-V4.1-Flash
//! model (YOCO self/cross decoder split, MLA + DSA sparse attention,
//! DeepSeekMoE experts, optional MTP/DSpark speculative decoding),
//! checkpoint loading, structured-output-constrained generation, and the
//! offline-inference example mirroring upstream
//! `examples/offline_inference.py`.
//!
//! Layering: [`generate`] is a deliberately simple, deterministic
//! sequential loop (batch-invariant by construction on the CPU path);
//! the `deepinfer-scheduler` crate provides the continuous-batching
//! engine for service deployments on top of [`DeepSeekModel`].
//!
//! See `SPEC.md` for the workspace-wide contracts and the per-crate
//! ground-truth references (FlashMLA README, DeepInfer README, DSA /
//! DeepGEMM / DeepEP / EPLB repositories).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod config;
pub mod generate;
pub mod loader;
pub mod model;
pub mod weights;

pub use config::ModelConfig;
pub use generate::{generate, GenerateOutput, GenerateRequest};
pub use model::DeepSeekModel;
