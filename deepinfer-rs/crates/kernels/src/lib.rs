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
//! * GEMM: BF16 / **F32 reference** / FP8 block-scaled NT (dense +
//!   grouped-contiguous + **grouped-masked** for CUDA-graph MoE),
//!   DeepGEMM semantics (activations 1×128, weights 128×128)
//! * attention: FlashMLA decode (split-KV online softmax + combine, all
//!   three KV cache formats, attn_sink, and a **dedicated FP4-cache decode
//!   kernel**), prefill (dense causal / sparse / device-scalar dense)
//! * DSA: lightning indexer (per-head RMSNorm, lightning quant, thunder
//!   logits), cascade conv branch, gating, deterministic block top-K
//! * MoE: sigmoid + noaux-tc routing, aligned dispatch (permute) and
//!   masked capacity dispatch (graph-safe), combine
//! * decode: the **`_dev` kernel family** — every scalar that changes
//!   between decode steps (sequence length, cache write rows, attention
//!   window starts) is read from device memory, so a whole decode step can
//!   be captured into a CUDA/HIP graph and replayed across sequence growth
//!   (bit-identical to the eager launches)
//! * peer: NVLink / Infinity Fabric **peer collectives for expert
//!   parallelism** — all-to-all FP8 dispatch and weighted combine over
//!   `memcpy_peer_async` with event barriers
//!
//! Numerics are ports of the CPU reference crates (`deepinfer-core`
//! bit-exact codecs; `deepinfer-gemm` / `-mla` / `-dsa` / `-moe` semantics)
//! and are validated by golden tests that run the CPU reference on the same
//! inputs whenever a GPU is present (auto-skipped otherwise).

use std::collections::HashMap;
use std::sync::Mutex;

/// Cache key of a tcgen05 tensor-map pair.
type TmKey = (u64, u64, usize, usize, usize);
/// The cached (A, B) tensor maps of one GEMM site.
type TmPair = (TensorMap, TensorMap);

use deepinfer_core::{ensure, Error, Result};
use deepinfer_gpu::{
    compile_and_load, Backend, Buffer, Device, Function, GpuArch, Module, Stream, TensorMap,
};

mod attention;
mod decode;
mod dsamoe;
mod elementwise;
mod gemm;
mod peer;
mod peer_ll;
mod serve;

pub use attention::DECODE_CHUNK;
pub use decode::{fanout_params, DecodeParams, StepGeometry};
pub use elementwise::CacheFormat;
pub use gemm::GemmDType;
pub use peer::{build_send_plan, PeerMesh, SendPlan};
pub use peer_ll::{SymmetricEp, SymmetricHeap};
pub use serve::{ServeCellIn, SERVE_CELL_FIELDS, SERVE_PAGE};

/// Embedded kernel sources (one translation unit per group).
pub mod source {
    /// Shared device foundation (codecs, reductions, bitonic sort).
    pub const COMMON: &str = include_str!("../kernels/common.cuh");
    /// Norms, RoPE, activations, quantization, KV-cache encode.
    pub const ELEMENTWISE: &str = include_str!("../kernels/elementwise.cu");
    /// Dense / grouped FP8 + BF16 + F32 NT GEMMs (incl. the masked
    /// graph-safe grouped GEMM).
    pub const GEMM: &str = include_str!("../kernels/gemm.cu");
    /// MLA decode (split-KV + combine, dedicated FP4 path, device-scalar
    /// dense) and prefill (dense causal / sparse).
    pub const ATTENTION: &str = include_str!("../kernels/attention.cu");
    /// DSA lightning indexer, conv branch, gate, top-K.
    pub const DSA: &str = include_str!("../kernels/dsa.cu");
    /// MoE routing, dispatch (aligned + masked), combine, fills.
    pub const MOE: &str = include_str!("../kernels/moe.cu");
    /// Device-resident decode parameter block layout (ABI with the Rust
    /// mirror in `decode.rs`).
    pub const DECODE_PARAMS: &str = include_str!("../kernels/decode_params.h");
    /// The graph-replayable `_dev` decode kernel family.
    pub const DECODE: &str = include_str!("../kernels/decode.cu");
    /// tcgen05 + TMA FP8 GEMM (sm_100a / sm_103a only; CUDA only).
    pub const GEMM_SM100: &str = include_str!("../kernels/gemm_sm100.cu");
    /// NVSHMEM-style symmetric-buffer EP (device-initiated peer writes).
    pub const PEER_LL: &str = include_str!("../kernels/peer_ll.cu");
    /// The paged-cache multi-sequence serving kernel family.
    pub const SERVE: &str = include_str!("../kernels/serve.cu");
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

/// Build the tcgen05 module when `device` is a Blackwell-class CUDA device.
///
/// The `a`-suffixed targets carry the tcgen05 instructions (the portable
/// `sm_100` target rejects them); the matching `compute_…a` PTX is the
/// fallback when the local NVRTC predates the cubin target.
fn sm100_module(device: &Device) -> Result<Option<Module>> {
    if device.backend() != Backend::Cuda {
        return Ok(None);
    }
    let (major, minor) = match device.props().arch {
        GpuArch::Sm(maj, min) => (maj, min),
        GpuArch::Gfx(_) => return Ok(None),
    };
    if major < 10 {
        return Ok(None);
    }
    // Family targets: 10.0/10.1 -> sm_100a, 10.3+ -> sm_103a (B300).
    let (sm, compute) = if minor >= 3 {
        ("sm_103a", "compute_103a")
    } else {
        ("sm_100a", "compute_100a")
    };
    let src = format!("{}\n{}", source::COMMON, source::GEMM_SM100);
    let extra = vec!["--use_fast_math=off".to_string()];
    let m = deepinfer_gpu::compile_and_load_with_arch(
        device,
        &src,
        "gemm_sm100",
        &format!("--gpu-architecture={sm}"),
        Some(&format!("--gpu-architecture={compute}")),
        &extra,
    )?;
    Ok(Some(m))
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
    m_dec: Module,
    /// tcgen05 module — present only on CUDA devices with compute
    /// capability >= 10.0 (compiled for the `sm_100a`/`sm_103a` family
    /// target). Everything else keeps the portable module set above.
    m_gemm100: Option<Module>,
    /// Low-latency symmetric-buffer EP module (any backend; the kernels
    /// degrade to in-rank copies when world == 1).
    m_peer_ll: Module,
    /// Paged multi-sequence serving module.
    m_serve: Module,
    cache: Mutex<HashMap<&'static str, Function>>,
    /// Tensor maps for the tcgen05 GEMM, keyed by
    /// `(a_ptr, b_ptr, m, n, k)`; bounded, cleared on overflow.
    tm_cache: Mutex<HashMap<TmKey, TmPair>>,
    /// Whether the tcgen05 kernel's dynamic-smem attribute was already
    /// raised (per-function, sticky).
    sm100_smem_set: Mutex<bool>,
}

impl std::fmt::Debug for Kernels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kernels")
            .field("device", &self.device.props().name)
            .field("arch", &self.device.props().arch.as_str())
            .field("tcgen05", &self.m_gemm100.is_some())
            .finish()
    }
}

impl Kernels {
    /// JIT-compile (or cache-load) all six module groups.
    pub fn new(device: &Device) -> Result<Kernels> {
        let cu = format!("{}\n{}", source::COMMON, source::ELEMENTWISE);
        let cg = format!("{}\n{}", source::COMMON, source::GEMM);
        let ca = format!("{}\n{}", source::COMMON, source::ATTENTION);
        let cd = format!("{}\n{}", source::COMMON, source::DSA);
        let cm = format!("{}\n{}", source::COMMON, source::MOE);
        let ck = format!(
            "{}\n{}\n{}",
            source::COMMON,
            source::DECODE_PARAMS,
            source::DECODE
        );
        let cpl = format!("{}\n{}", source::COMMON, source::PEER_LL);
        let csv = format!("{}\n{}", source::COMMON, source::SERVE);
        // The tcgen05 module targets the architecture-specific family
        // (`sm_100a` / `sm_103a`); it is built only on Blackwell-class CUDA
        // devices, and its absence is not an error anywhere else.
        let m_gemm100 = sm100_module(device)?;
        let k = Kernels {
            m_elem: build_module(device, "elementwise", &cu)?,
            m_gemm: build_module(device, "gemm", &cg)?,
            m_att: build_module(device, "attention", &ca)?,
            m_dsa: build_module(device, "dsa", &cd)?,
            m_moe: build_module(device, "moe", &cm)?,
            m_dec: build_module(device, "decode", &ck)?,
            m_peer_ll: build_module(device, "peer_ll", &cpl)?,
            m_serve: build_module(device, "serve", &csv)?,
            m_gemm100,
            device: device.clone(),
            cache: Mutex::new(HashMap::new()),
            tm_cache: Mutex::new(HashMap::new()),
            sm100_smem_set: Mutex::new(false),
        };
        Ok(k)
    }

    /// Whether the tcgen05 (5th-gen tensor core) GEMM module is available
    /// on this device (CUDA with compute capability >= 10.0).
    pub fn tcgen05_available(&self) -> bool {
        self.m_gemm100.is_some()
    }

    /// The device this library runs on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Resolve (and memoize) a kernel entry point by name. Memoized lookups
    /// cost one uncontended mutex lock; misses do one driver lookup.
    pub fn f(&self, name: &'static str) -> Result<Function> {
        if let Some(f) = self.cache.lock().expect("kernel cache poisoned").get(name) {
            return Ok(f.clone());
        }
        let module = match name {
            // The graph-replayable decode family lives in its own module.
            // Matched FIRST: several of its names share prefixes with the
            // prefill kernels (di_kv_encode_*, di_lightning_*, di_conv_*).
            n if n.starts_with("di_step_params")
                || n.starts_with("di_pos_fill")
                || n.starts_with("di_append")
                || n.ends_with("_dev_kernel") =>
            {
                &self.m_dec
            }
            n if n.starts_with("di_gather_rows") => &self.m_att,
            n if n.starts_with("di_silu_mul_masked") => &self.m_moe,
            n if n.starts_with("di_gemm_fp8_nt_sm100a") => {
                self.m_gemm100.as_ref().ok_or_else(|| {
                    Error::invalid("the tcgen05 GEMM needs CUDA compute capability >= 10.0")
                })?
            }
            n if n.starts_with("di_ep_") => &self.m_peer_ll,
            n if n.starts_with("di_serve_") => &self.m_serve,
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
                || n.starts_with("di_topk") =>
            {
                &self.m_dsa
            }
            n if n.starts_with("di_moe")
                || n.starts_with("di_fill")
                || n.starts_with("di_sigmoid") =>
            {
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
            ("decode", source::DECODE),
        ] {
            assert!(s.len() > 1000, "{name} source suspiciously short");
            assert!(
                s.contains("extern \"C\" __global__"),
                "{name} has no kernels"
            );
        }
        assert!(source::DECODE_PARAMS.contains("DP_COUNT"));
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
            source::DECODE_PARAMS,
            source::DECODE,
        ] {
            assert!(!s.contains("size_t"), "size_t leaked into device source");
        }
    }

    #[test]
    fn decode_module_routing_is_unambiguous() {
        // Every kernel defined in decode.cu must route to the decode module
        // under the match rules in `Kernels::f` (suffix `_dev_kernel` or the
        // three dev-specific prefixes); shared-name prefill kernels must
        // keep routing to their original modules.
        for k in [
            "di_step_params_kernel",
            "di_pos_fill_kernel",
            "di_kv_encode_fp8_dev_kernel",
            "di_kv_encode_fp4_dev_kernel",
            "di_append_row_kernel",
            "di_append_norm_row_kernel",
            "di_lightning_block_zero_dev_kernel",
            "di_lightning_amax_dev_kernel",
            "di_lightning_finalize_dev_kernel",
            "di_lightning_quant_dev_kernel",
            "di_thunder_logits_dev_kernel",
            "di_dot_scores_dev_kernel",
            "di_gate_blend_dev_kernel",
            "di_causal_window_clip_dev_kernel",
            "di_block_preselect_dev_kernel",
            "di_conv_feat_dev_kernel",
            "di_conv_gated_norm_dev_kernel",
        ] {
            assert!(source::DECODE.contains(k), "decode.cu lost {k}");
            let routed = k.starts_with("di_step_params")
                || k.starts_with("di_pos_fill")
                || k.starts_with("di_append")
                || k.ends_with("_dev_kernel");
            assert!(routed, "{k} would not route to the decode module");
        }
        // Prefill twins keep their original routing.
        for (k, is_dsa) in [
            ("di_kv_encode_fp8_kernel", false),
            ("di_lightning_quant_kernel", true),
            ("di_conv_feat_kernel", true),
            ("di_causal_clip_kernel", true),
            ("di_block_preselect_kernel", true),
            ("di_dot_scores_kernel", true),
        ] {
            let routed_dev = k.starts_with("di_step_params")
                || k.starts_with("di_pos_fill")
                || k.starts_with("di_append")
                || k.ends_with("_dev_kernel");
            assert!(!routed_dev, "{k} would be misrouted to the decode module");
            let _ = is_dsa;
        }
        // The masked-silu / gather kernels route where they are defined.
        assert!(source::MOE.contains("di_silu_mul_masked_kernel"));
        assert!(source::ATTENTION.contains("di_gather_rows_kernel"));
    }

    #[test]
    fn all_launched_kernel_names_are_defined() {
        // Every kernel name referenced by the launcher modules must exist
        // in exactly one embedded source (guards against typos drifting in
        // between the .cu files and the Rust sides).
        let rust_sources = [
            (elementwise::KERNEL_NAMES, source::ELEMENTWISE),
            (gemm::KERNEL_NAMES, source::GEMM),
            (attention::KERNEL_NAMES, source::ATTENTION),
            (dsamoe::KERNEL_NAMES, source::DSA),
            (dsamoe::MOE_KERNEL_NAMES, source::MOE),
            (decode::KERNEL_NAMES, source::DECODE),
            (gemm::KERNEL_NAMES_SM100, source::GEMM_SM100),
            (gemm::KERNEL_NAMES_MFMA, source::GEMM),
            (peer_ll::KERNEL_NAMES, source::PEER_LL),
            (serve::KERNEL_NAMES, source::SERVE),
        ];
        for (names, src) in rust_sources {
            for k in names {
                assert!(src.contains(k), "{k} missing from its module source");
            }
        }
    }
}
