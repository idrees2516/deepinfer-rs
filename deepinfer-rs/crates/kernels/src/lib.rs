//! # deepinfer-kernels — GPU kernels for the DeepSeek inference stack
//!
//! Shared CUDA/HIP C++ sources (in `kernels/`), JIT-compiled at runtime by
//! [`deepinfer_gpu::jit`] for the exact device in use, plus typed Rust
//! launchers for every operator the DeepInfer model forward pass needs:
//!
//! * elementwise: RMSNorm, interleaved RoPE (+ O-RoPE conjugate), SiLU-mul,
//!   residual add, f32↔bf16 casts
//! * quantization: per-token-group E4M3+UE8M0 (DeepGEMM activation recipe),
//!   FlashMLA V4.1 FP8 (528 B) and FP4 (288 B) KV-cache row encoders
//! * GEMM: BF16 NT and FP8 block-scaled NT (dense + grouped-contiguous),
//!   DeepGEMM semantics (activations 1×128, weights 128×128)
//! * attention: FlashMLA decode (split-KV online softmax + combine, all
//!   three KV cache formats, attn_sink) and prefill (dense causal / sparse)
//! * DSA: lightning indexer (per-head RMSNorm, lightning quant, thunder
//!   logits), cascade conv branch, gating, deterministic block top-K
//! * MoE: sigmoid + noaux-tc routing, aligned dispatch (permute), combine
//!
//! Numerics are ports of the CPU reference crates (`deepinfer-core`
//! bit-exact codecs; `deepinfer-gemm` / `-mla` / `-dsa` / `-moe` semantics)
//! and are validated by golden tests that run the CPU reference on the same
//! inputs whenever a GPU is present (auto-skipped otherwise).

use std::collections::HashMap;
use std::sync::Mutex;

use deepinfer_core::{ensure, Error, Result};
use deepinfer_gpu::{compile_and_load, Backend, Buffer, Device, Function, Module, Stream};

mod elementwise;
mod gemm;
mod attention;
mod dsamoe;

pub use elementwise::CacheFormat;

/// Embedded kernel sources (one translation unit per group).
pub mod source {
    /// Shared device foundation (codecs, reductions, bitonic sort).
    pub const COMMON: &str = include_str!("../kernels/common.cuh");
    /// Norms, RoPE, activations, quantization, KV-cache encode.
    pub const ELEMENTWISE: &str = include_str!("../kernels/elementwise.cu");
    /// Dense / grouped FP8 + BF16 NT GEMMs.
    pub const GEMM: &str = include_str!("../kernels/gemm.cu");
    /// MLA decode (split-KV + combine) and prefill.
    pub const ATTENTION: &str = include_str!("../kernels/attention.cu");
    /// DSA lightning indexer, conv branch, gate, top-K.
    pub const DSA: &str = include_str!("../kernels/dsa.cu");
    /// MoE routing, dispatch, combine, fills.
    pub const MOE: &str = include_str!("../kernels/moe.cu");
}

/// Compile a module group for `device` (goes through the JIT disk cache).
fn build_module(device: &Device, name: &str, src: &str) -> Result<Module> {
    let arch = device.props().arch.clone();
    let extra: Vec<String> = match device.backend() {
        // Match the CPU reference's rounding closely enough for golden
        // tolerances; FP contraction stays ON for GEMM throughput.
        Backend::Cuda => vec!["--use_fast_math=off".to_string()],
        Backend::Hip => vec![],
    };
    compile_and_load(device, src, name, &arch, &extra)
}

/// Compiled kernel library bound to one device.
///
/// All kernel entry points are resolved once at construction. Launches are
/// enqueued on caller-provided streams; nothing synchronizes implicitly.
pub struct Kernels {
    device: Device,
    m_elem: Module,
    m_gemm: Module,
    m_att: Module,
    m_dsa: Module,
    m_moe: Module,
    cache: Mutex<HashMap<&'static str, Function>>,
}

impl std::fmt::Debug for Kernels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kernels")
            .field("device", &self.device.props().name)
            .field("arch", &self.device.props().arch.as_str())
            .finish()
    }
}

impl Kernels {
    /// JIT-compile (or cache-load) all five module groups.
    pub fn new(device: &Device) -> Result<Kernels> {
        let cu = format!("{}\n{}", source::COMMON, source::ELEMENTWISE);
        let cg = format!("{}\n{}", source::COMMON, source::GEMM);
        let ca = format!("{}\n{}", source::COMMON, source::ATTENTION);
        let cd = format!("{}\n{}", source::COMMON, source::DSA);
        let cm = format!("{}\n{}", source::COMMON, source::MOE);
        let k = Kernels {
            m_elem: build_module(device, "elementwise", &cu)?,
            m_gemm: build_module(device, "gemm", &cg)?,
            m_att: build_module(device, "attention", &ca)?,
            m_dsa: build_module(device, "dsa", &cd)?,
            m_moe: build_module(device, "moe", &cm)?,
            device: device.clone(),
            cache: Mutex::new(HashMap::new()),
        };
        Ok(k)
    }

    /// The device this library runs on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Resolve (and memoize) a kernel entry point by name. Memoized lookups
    /// cost one uncontended mutex lock; misses do one driver lookup.
    pub(crate) fn f(&self, name: &'static str) -> Result<Function> {
        if let Some(f) = self.cache.lock().expect("kernel cache poisoned").get(name) {
            return Ok(f.clone());
        }
        let module = match name {
            n if n.starts_with("di_rmsnorm")
                || n.starts_with("di_rope")
                || n.starts_with("di_silu")
                || n.starts_with("di_add")
                || n.starts_with("di_quant")
                || n.starts_with("di_kv")
                || n.starts_with("di_f32_to")
                || n.starts_with("di_bf16_to") =>
            {
                &self.m_elem
            }
            n if n.starts_with("di_gemm") => &self.m_gemm,
            n if n.starts_with("di_mla") => &self.m_att,
            n if n.starts_with("di_indexer")
                || n.starts_with("di_lightning")
                || n.starts_with("di_thunder")
                || n.starts_with("di_conv")
                || n.starts_with("di_dot_scores")
                || n.starts_with("di_qmean")
                || n.starts_with("di_gate")
                || n.starts_with("di_causal")
                || n.starts_with("di_block_preselect")
                || n.starts_with("di_topk") => &self.m_dsa,
            n if n.starts_with("di_moe") || n.starts_with("di_fill") || n.starts_with("di_sigmoid") => {
                &self.m_moe
            }
            _ => return Err(Error::internal(format!("unknown kernel `{name}`"))),
        };
        let f = module.function(name)?;
        self.cache
            .lock()
            .expect("kernel cache poisoned")
            .insert(name, f.clone());
        Ok(f)
    }

    /// Launch helper: grid = ceil(total / block) on x.
    pub(crate) fn launch1d(
        &self,
        name: &'static str,
        total: usize,
        block: u32,
        shared: u32,
        args: deepinfer_gpu::KernelArgs,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(block > 0, InvalidArgument, "block must be > 0");
        let grid = (total as u64).div_ceil(block as u64) as u32;
        let f = self.f(name)?;
        f.launch(
            &self.device,
            deepinfer_gpu::Launch {
                grid: (grid.max(1), 1, 1),
                block: (block, 1, 1),
                shared,
            },
            &args,
            stream,
        )
    }
}

/// Allocate a zeroed device buffer of `n` elements of `T`.
pub fn alloc_zeroed<T: Copy>(dev: &Device, n: usize) -> Result<Buffer> {
    let buf = dev.alloc(n * std::mem::size_of::<T>())?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sources_are_nonempty_and_balanced() {
        assert!(source::COMMON.len() > 1000);
        for (name, s) in [
            ("elementwise", source::ELEMENTWISE),
            ("gemm", source::GEMM),
            ("attention", source::ATTENTION),
            ("dsa", source::DSA),
            ("moe", source::MOE),
        ] {
            assert!(s.len() > 1000, "{name} source suspiciously short");
            assert!(s.contains("extern \"C\" __global__"), "{name} has no kernels");
        }
    }

    #[test]
    fn no_size_t_in_device_sources() {
        // NVRTC / hipRTC compile without headers; `size_t` must not appear.
        for s in [
            source::COMMON,
            source::ELEMENTWISE,
            source::GEMM,
            source::ATTENTION,
            source::DSA,
            source::MOE,
        ] {
            assert!(!s.contains("size_t"), "size_t leaked into device source");
        }
    }
}
