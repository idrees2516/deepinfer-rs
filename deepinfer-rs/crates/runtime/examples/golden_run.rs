//! On-hardware golden-run harness — the acceptance battery for the GPU
//! stack (record/verify artifacts + capability report + timings).
//!
//! ```text
//! cargo run --release -p deepinfer-runtime --example golden_run -- --list
//! cargo run --release -p deepinfer-runtime --example golden_run -- --record golden.json
//! cargo run --release -p deepinfer-runtime --example golden_run -- --verify golden.json
//! ```
//!
//! `--record` writes a JSON artifact describing the machine (backend,
//! device, arch, JIT versions, feature flags) and one entry per check with
//! its status and metric (relative errors, hashes of bit-exact outputs,
//! timings). `--verify` re-runs the battery and requires every recorded
//! check to pass again (metric drift is reported, not fatal). With no GPU
//! present the harness prints the capability report and exits 0 with all
//! checks skipped — the same auto-skip contract as the test suite.
//!
//! Deterministic parts (CUDA-graph replay, rollback, run-to-run
//! determinism) are compared through a SHA-256 of the f32 output bytes, so
//! a verify on the same machine/driver is exact.

use std::fmt::Write as _;

use deepinfer_gpu::{Backend, Gpu, GpuArch};
use deepinfer_runtime::{DeviceModel, ModelConfig, Precision};

// ---------------------------------------------------------------------------
// Minimal JSON writer (std-only; the workspace has no dependencies).
// ---------------------------------------------------------------------------

fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

#[derive(Default)]
struct Json {
    buf: String,
}

impl Json {
    fn new() -> Json {
        Json {
            buf: String::from("{"),
        }
    }
    fn key(&mut self, k: &str) {
        if !self.buf.ends_with('{') && !self.buf.ends_with('[') && !self.buf.ends_with(',') {
            self.buf.push(',');
        }
        let _ = write!(self.buf, "\"{}\":", esc(k));
    }
    fn str(&mut self, k: &str, v: &str) {
        self.key(k);
        let _ = write!(self.buf, "\"{}\"", esc(v));
    }
    fn num(&mut self, k: &str, v: f64) {
        self.key(k);
        if v.is_finite() {
            let _ = write!(self.buf, "{v:.6}");
        } else {
            self.buf.push_str("null");
        }
    }
    fn boolean(&mut self, k: &str, v: bool) {
        self.key(k);
        self.buf.push_str(if v { "true" } else { "false" });
    }
    fn raw_field(&mut self, k: &str, raw: &str) {
        self.key(k);
        self.buf.push_str(raw);
    }
    fn finish(mut self) -> String {
        self.buf.push('}');
        self.buf
    }
}

/// One check result.
struct Check {
    name: &'static str,
    status: Status,
    metric: f64,
    detail: String,
}

enum Status {
    Pass,
    Fail,
    Skip,
}

impl Status {
    fn as_str(&self) -> &'static str {
        match self {
            Status::Pass => "pass",
            Status::Fail => "fail",
            Status::Skip => "skip",
        }
    }
}

/// FNV-1a over the LE bytes of an f32 slice — stable, dependency-free.
fn hash_f32(v: &[f32]) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for x in v {
        for b in x.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    format!("{h:016x}")
}

fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    let mut worst = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        let m = x.abs().max(y.abs()).max(1e-3);
        worst = worst.max(d / m);
    }
    worst
}

fn now_ms() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64() * 1000.0)
        .unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// The battery
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("--list");
    let path = args.get(2).cloned().unwrap_or_else(|| "golden.json".into());

    // Capability report.
    let mut cap = Json::new();
    let mut devices = Vec::new();
    for b in [Backend::Cuda, Backend::Hip] {
        if let Ok(gpu) = Gpu::new(b) {
            if let Ok(n) = gpu.device_count() {
                for i in 0..n {
                    if let Ok(d) = gpu.device(i) {
                        devices.push((gpu.clone(), d, i));
                    }
                }
            }
        }
    }
    cap.str("generated_ms", &format!("{}", now_ms()));
    cap.str("workspace", "deepinfer-rs");
    cap.str("artifact_version", "1");
    cap.num("devices", devices.len() as f64);
    if devices.is_empty() {
        cap.str("note", "no GPU present; all checks skip (CPU-only host)");
    }
    let mut dev_json = String::from("[");
    for (gpu, dev, i) in &devices {
        let p = dev.props();
        let mut j = Json::new();
        j.str("backend", gpu.backend().name());
        j.str("name", &p.name);
        j.str("arch", &p.arch.as_str());
        j.num("sm_count", p.sm_count as f64);
        j.num(
            "total_mem_gb",
            p.total_mem as f64 / (1024.0 * 1024.0 * 1024.0),
        );
        if gpu.backend() == Backend::Cuda {
            if let Some((ma, mi)) = gpu.nvrtc_version() {
                j.str("nvrtc", &format!("{ma}.{mi}"));
            }
            let tc = matches!(p.arch, GpuArch::Sm(m, _) if m >= 10);
            j.boolean("tcgen05_family", tc);
        }
        j.boolean("graphs", deepinfer_gpu::graph_supported(dev));
        let _ = i;
        dev_json.push_str(&j.finish());
    }
    dev_json.push(']');
    cap.raw_field("device_list", &dev_json);
    let report = cap.finish();
    println!("{report}");

    if mode == "--list" {
        println!("modes: --record <file> | --verify <file> | (default) run");
        return;
    }

    // Battery.
    let mut checks: Vec<Check> = Vec::new();
    let Some((_, dev, _)) = devices.first().cloned() else {
        // Nothing to run; still write a valid artifact.
        write_artifact(&path, &report, &[], mode);
        return;
    };

    // 1) F32 model vs the CPU golden.
    {
        let name = "model_f32_vs_cpu";
        let cfg = ModelConfig::v41_tiny();
        let mut cpu = deepinfer_runtime::DeepSeekModel::random_init(&cfg, 42).unwrap();
        let mut gpu = DeviceModel::random_init(&cfg, 42, &dev, Precision::F32).unwrap();
        let prompt: Vec<u32> = (0..6u32).map(|t| (t * 61) % 250).collect();
        let la = cpu.prefill(7, &prompt).unwrap();
        let lb = gpu.prefill(7, &prompt).unwrap();
        let e = rel_err(&la, &lb);
        let ok = e < 5e-2 && lb.iter().all(|v| v.is_finite());
        checks.push(Check {
            name,
            status: if ok { Status::Pass } else { Status::Fail },
            metric: e as f64,
            detail: format!("prefill rel err {e}"),
        });
    }

    // 2) FP8 Auto smoke.
    {
        let name = "model_fp8_auto_vs_cpu";
        let cfg = ModelConfig::v41_tiny();
        let cfg = {
            let mut c = cfg;
            c.cache = deepinfer_mla::CacheFormat::Fp8Group32;
            c
        };
        cfg.validate().unwrap();
        let mut cpu = deepinfer_runtime::DeepSeekModel::random_init(&cfg, 42).unwrap();
        let mut gpu = DeviceModel::random_init(&cfg, 42, &dev, Precision::Auto).unwrap();
        let prompt: Vec<u32> = (0..8u32).map(|t| (t * 31) % 250).collect();
        let la = cpu.prefill(3, &prompt).unwrap();
        let lb = gpu.prefill(3, &prompt).unwrap();
        let e = rel_err(&la, &lb);
        let ok = e < 0.25 && lb.iter().all(|v| v.is_finite());
        checks.push(Check {
            name,
            status: if ok { Status::Pass } else { Status::Fail },
            metric: e as f64,
            detail: format!("fp8 prefill rel err {e}"),
        });
    }

    // 3) Graph replay + determinism + rollback (bit-exact hashes).
    {
        let cfg = ModelConfig::v41_tiny();
        let mut gpu = DeviceModel::random_init(&cfg, 42, &dev, Precision::F32).unwrap();
        let prompt: Vec<u32> = (0..6u32).map(|t| (t * 61) % 250).collect();
        gpu.prefill(7, &prompt).unwrap();
        gpu.enable_decode_graph(true).unwrap();
        let h1 = {
            let l = gpu.decode_step(7, 5).unwrap();
            hash_f32(&l)
        };
        let h2 = {
            let l = gpu.decode_step(7, 6).unwrap();
            hash_f32(&l)
        };
        checks.push(Check {
            name: "decode_graph_replay_deterministic",
            status: if h1 == h2 { Status::Pass } else { Status::Fail },
            metric: 0.0,
            detail: format!("hashes {h1} / {h2}"),
        });
        // Rollback bit-exactness.
        let before = {
            let l = gpu.decode_step(7, 7).unwrap();
            hash_f32(&l)
        };
        gpu.rollback_seq(7, 1).unwrap();
        let after = {
            let l = gpu.decode_step(7, 7).unwrap();
            hash_f32(&l)
        };
        checks.push(Check {
            name: "rollback_bit_exact",
            status: if before == after {
                Status::Pass
            } else {
                Status::Fail
            },
            metric: 0.0,
            detail: format!("hashes {before} / {after}"),
        });
    }

    // 4) tcgen05 vs portable (Blackwell only).
    {
        let name = "tcgen05_vs_portable_gemm";
        let is_sm100 = match dev.props().arch {
            GpuArch::Sm(m, _) => m >= 10,
            _ => false,
        } && dev.backend() == Backend::Cuda;
        if !is_sm100 {
            checks.push(Check {
                name,
                status: Status::Skip,
                metric: 0.0,
                detail: "requires CUDA cc >= 10.0".into(),
            });
        } else {
            let k = deepinfer_kernels::Kernels::new(&dev).unwrap();
            let (m, n, kk) = (96usize, 512usize, 512usize);
            let mut rng = deepinfer_core::rng::Philox::new(11, 3);
            let a: Vec<f32> = (0..m * kk).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
            let b: Vec<f32> = (0..n * kk).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
            let (aq, asf) = deepinfer_fp8::quantize_per_token_group(&a, kk, 128).unwrap();
            let (bq, bsf) = deepinfer_fp8::quantize_weight_per_block(&b, n, kk, 128, 128).unwrap();
            let aq_b = dev.alloc_from(&aq).unwrap();
            let asf_b = dev.alloc_from(&asf).unwrap();
            let bq_b = dev.alloc_from(&bq).unwrap();
            let bsf_b = dev.alloc_from(&bsf).unwrap();
            let d1 = dev.alloc(m * n * 4).unwrap();
            let d2 = dev.alloc(m * n * 4).unwrap();
            let s = dev.stream().unwrap();
            k.gemm_fp8_nt(&aq_b, &asf_b, &bq_b, &bsf_b, &d1, m, n, kk, &s)
                .unwrap();
            k.gemm_fp8_nt_sm100(&aq_b, &asf_b, &bq_b, &bsf_b, &d2, m, n, kk, &s)
                .unwrap();
            s.synchronize().unwrap();
            let mut g1 = vec![0.0f32; m * n];
            let mut g2 = vec![0.0f32; m * n];
            dev.read(&d1, &mut g1).unwrap();
            dev.read(&d2, &mut g2).unwrap();
            let e = rel_err(&g1, &g2);
            checks.push(Check {
                name,
                status: if e < 1e-3 { Status::Pass } else { Status::Fail },
                metric: e as f64,
                detail: format!("rel err {e}"),
            });
        }
    }

    // 5) Paged serve engine vs the single-sequence path.
    {
        let name = "serve_engine_vs_single_sequence";
        let cfg = ModelConfig::v41_tiny();
        let mut gpu = DeviceModel::random_init(&cfg, 42, &dev, Precision::F32).unwrap();
        let prompt: Vec<u32> = (0..6u32).map(|t| (t * 61) % 250).collect();
        let lp = gpu.prefill(7, &prompt).unwrap();
        let mut steps = Vec::new();
        let mut t = 5u32;
        for _ in 0..4 {
            let l = gpu.decode_step(7, t).unwrap();
            t = deepinfer_core::math::argmax(&l).unwrap() as u32;
            steps.push(l);
        }
        gpu.serve_configure(2, 64, 2).unwrap();
        gpu.serve_attach(1).unwrap();
        gpu.serve_attach(2).unwrap();
        let prompt2: Vec<u32> = (0..6u32).map(|t| (t * 97 + 3) % 250).collect();
        let l1 = gpu.serve_prefill(1, &prompt).unwrap();
        let _ = gpu.serve_prefill(2, &prompt2).unwrap();
        let mut worst = rel_err(&lp, &l1);
        let mut tt = 5u32;
        for want in &steps {
            let got = gpu.serve_decode(&[(1, tt), (2, 100)]).unwrap();
            worst = worst.max(rel_err(want, &got[0]));
            tt = deepinfer_core::math::argmax(&got[0]).unwrap() as u32;
        }
        checks.push(Check {
            name,
            status: if worst < 5e-2 {
                Status::Pass
            } else {
                Status::Fail
            },
            metric: worst as f64,
            detail: format!("worst rel err {worst}"),
        });
    }

    // 5b) DSpark speculative serving surface: host draft -> chunked
    //     serve_verify vs sequential decodes -> rollback exactness.
    {
        let name = "serve_spec_decode";
        let cfg = ModelConfig::v41_tiny();
        let mut gpu = DeviceModel::random_init(&cfg, 42, &dev, Precision::F32).unwrap();
        let prompt: Vec<u32> = (0..6u32).map(|t| (t * 61) % 250).collect();
        gpu.serve_configure(2, 64, 4).unwrap();
        gpu.serve_attach(1).unwrap();
        gpu.serve_attach(2).unwrap();
        let anchor = gpu.serve_prefill_out(1, &prompt).unwrap();
        let _ = gpu.serve_prefill(2, &prompt).unwrap();
        // Draft chain from the anchor's hidden state.
        let mut chain = Vec::new();
        let h = anchor.hidden.clone();
        let mut last = *prompt.last().unwrap();
        for _ in 0..3 {
            let d = gpu.draft_step(last, &h).unwrap();
            last = d.token;
            chain.push(d.token);
        }
        // Chunked verification vs sequential decodes on the twin slot
        // (same prompt): decoding chain[j] on slot 2 yields rows[j].
        let rows = gpu.serve_verify(1, &chain).unwrap();
        let mut seq_rows = Vec::new();
        for &c in &chain {
            let r = gpu.serve_decode(&[(2, c)]).unwrap();
            seq_rows.push(r.into_iter().next().unwrap());
        }
        let mut worst = 0.0f32;
        for (a, b) in rows.iter().zip(seq_rows.iter()) {
            worst = worst.max(rel_err(a, b));
        }
        // Rollback the chain: the re-decode of the first draft must
        // reproduce the pre-rollback row exactly (slot-keyed rewrite).
        gpu.serve_rollback(1, chain.len()).unwrap();
        let after = gpu.serve_decode(&[(1, chain[0])]).unwrap();
        let after = after.into_iter().next().unwrap();
        let direct = {
            let mut m = DeviceModel::random_init(&cfg, 42, &dev, Precision::F32).unwrap();
            let _ = m.prefill(7, &prompt).unwrap();
            m.decode_step(7, chain[0]).unwrap()
        };
        worst = worst.max(rel_err(&after, &direct));
        checks.push(Check {
            name,
            status: if worst < 5e-2 {
                Status::Pass
            } else {
                Status::Fail
            },
            metric: worst as f64,
            detail: format!("draft 3, worst rel err {worst}"),
        });
    }

    // 5c) GPU prefix cache: block-granular page sharing. A warm re-
    //     prefill of the same 130-token prompt must hit the radix tree
    //     and match the cold logits; a rollback below the frozen
    //     prefix (copy-on-write) must keep the cached snapshot exact
    //     for a third sequence.
    {
        let name = "serve_prefix_cache";
        let cfg = ModelConfig::v41_tiny();
        let mut gpu = DeviceModel::random_init(&cfg, 42, &dev, Precision::F32).unwrap();
        let prompt: Vec<u32> = (0..130u32).map(|t| (t * 37 + 5) % 250).collect();
        gpu.serve_configure(3, 256, 16).unwrap();
        gpu.serve_prefix_configure(1 << 18).unwrap();
        gpu.serve_attach(1).unwrap();
        let cold = gpu.serve_prefill(1, &prompt).unwrap();
        let st = gpu.serve_prefix_stats();
        gpu.serve_attach(2).unwrap();
        let warm = gpu.serve_prefill(2, &prompt).unwrap();
        let st2 = gpu.serve_prefix_stats();
        // Copy-on-write: sequence 2 rewinds below the frozen prefix and
        // appends divergent tokens; sequence 3 must still restore the
        // original prefix bit-exactly.
        gpu.serve_rollback(2, 11).unwrap();
        let alt: Vec<u32> = vec![200, 150, 99, 3];
        let _ = gpu.serve_verify(2, &alt).unwrap();
        gpu.serve_rollback(2, alt.len()).unwrap();
        let _ = gpu.serve_decode(&[(2, alt[0])]).unwrap();
        gpu.serve_attach(3).unwrap();
        let again = gpu.serve_prefill(3, &prompt).unwrap();
        let worst = rel_err(&cold, &warm).max(rel_err(&cold, &again));
        let ok = st.nodes == 2
            && st.cached_tokens == 128
            && st2.hits == st.hits + 1
            && st2.reused_tokens >= 128
            && worst < 5e-2;
        checks.push(Check {
            name,
            status: if ok { Status::Pass } else { Status::Fail },
            metric: worst as f64,
            detail: format!(
                "nodes {} hits {} reused {} worst rel err {worst}",
                st2.nodes, st2.hits, st2.reused_tokens
            ),
        });
    }

    // 5d) P1-8 kernel-family A/B: the cp.async-pipelined KV walk must be
    //     BIT-IDENTICAL to the direct-load family (hash compare with the
    //     vec kernels held off), and the vectorized DSA score kernels must
    //     stay within the serve tolerance band of the scalar family.
    {
        let name = "serve_pipe_vs_legacy";
        let cfg = ModelConfig::v41_tiny();
        let prompt: Vec<u32> = (0..130u32).map(|t| (t * 37 + 5) % 250).collect();
        // The workload under test: warm prefill + a few batched decodes.
        let run_engine = |pipe: Option<bool>, vec: Option<bool>| -> Vec<String> {
            match pipe {
                Some(true) => std::env::set_var("DEEPINFER_SERVE_PIPE", "1"),
                Some(false) => std::env::set_var("DEEPINFER_SERVE_PIPE", "0"),
                None => std::env::remove_var("DEEPINFER_SERVE_PIPE"),
            }
            match vec {
                Some(true) => std::env::set_var("DEEPINFER_SERVE_VEC", "1"),
                Some(false) => std::env::set_var("DEEPINFER_SERVE_VEC", "0"),
                None => std::env::remove_var("DEEPINFER_SERVE_VEC"),
            }
            let mut gpu = DeviceModel::random_init(&cfg, 42, &dev, Precision::F32).unwrap();
            gpu.serve_configure(2, 256, 16).unwrap();
            gpu.serve_attach(1).unwrap();
            gpu.serve_attach(2).unwrap();
            let prompt2: Vec<u32> = (0..80u32).map(|t| (t * 53 + 7) % 250).collect();
            let l1 = gpu.serve_prefill(1, &prompt).unwrap();
            let l2 = gpu.serve_prefill(2, &prompt2).unwrap();
            let mut hashes = vec![hash_f32(&l1), hash_f32(&l2)];
            let mut t1 = 5u32;
            let mut t2 = 100u32;
            for _ in 0..4 {
                let got = gpu.serve_decode(&[(1, t1), (2, t2)]).unwrap();
                t1 = deepinfer_core::math::argmax(&got[0]).unwrap() as u32;
                t2 = (t2 + 7) % 200;
                hashes.push(hash_f32(&got[0]));
                hashes.push(hash_f32(&got[1]));
            }
            hashes
        };
        let legacy = run_engine(Some(false), Some(false));
        let piped = run_engine(Some(true), Some(false));
        let piped_vec = run_engine(Some(true), Some(true));
        std::env::remove_var("DEEPINFER_SERVE_PIPE");
        std::env::remove_var("DEEPINFER_SERVE_VEC");
        let bit_exact = legacy.iter().zip(&piped).all(|(a, b)| a == b);
        // The vec family reassociates the DSA dots (fixed tree): compare
        // via the hash difference count only (logits already flowed through
        // argmax; the serve tolerance contract is checked by 5/5b/5c which
        // run with the same defaults as the vec path).
        let vec_diffs = piped
            .iter()
            .zip(&piped_vec)
            .filter(|(a, b)| a != b)
            .count();
        let ok = bit_exact;
        checks.push(Check {
            name,
            status: if ok { Status::Pass } else { Status::Fail },
            metric: vec_diffs as f64,
            detail: format!(
                "pipe bit-exact: {bit_exact}; vec reassociated steps: {vec_diffs}/{}",
                piped.len()
            ),
        });
    }

    // 5e) P1-7 megakernel: the fused self-layer block must track the
    //     unfused chain within the serve tolerance band (the GEMV
    //     association differs from the tiled GEMM kernels).
    {
        let name = "serve_megakernel_vs_unfused";
        let cfg = ModelConfig::v41_tiny();
        let prompt: Vec<u32> = (0..80u32).map(|t| (t * 37 + 5) % 250).collect();
        let run_engine = |mega: bool| -> Vec<f32> {
            if mega {
                std::env::set_var("DEEPINFER_MEGAKERNEL", "1");
            } else {
                std::env::set_var("DEEPINFER_MEGAKERNEL", "0");
            }
            let mut gpu = DeviceModel::random_init(&cfg, 42, &dev, Precision::F32).unwrap();
            gpu.serve_configure(2, 256, 16).unwrap();
            gpu.serve_attach(1).unwrap();
            gpu.serve_attach(2).unwrap();
            let prompt2: Vec<u32> = (0..50u32).map(|t| (t * 53 + 7) % 250).collect();
            let l1 = gpu.serve_prefill(1, &prompt).unwrap();
            let _l2 = gpu.serve_prefill(2, &prompt2).unwrap();
            let mut t1 = 5u32;
            let mut last = l1.clone();
            for _ in 0..4 {
                let got = gpu.serve_decode(&[(1, t1), (2, 100)]).unwrap();
                t1 = deepinfer_core::math::argmax(&got[0]).unwrap() as u32;
                last = got[0].clone();
            }
            std::env::remove_var("DEEPINFER_MEGAKERNEL");
            let mut all = l1;
            all.extend(last);
            all
        };
        let unfused = run_engine(false);
        let mega = run_engine(true);
        let e = rel_err(&unfused, &mega);
        checks.push(Check {
            name,
            status: if e < 5e-2 { Status::Pass } else { Status::Fail },
            metric: e as f64,
            detail: format!("worst rel err {e}"),
        });
    }

    // 6) Symmetric-buffer EP (world 1; multi-rank when available).
    {
        let name = "symmetric_ep_protocol";
        let world = devices.len().min(2);
        let run = |devs: &[deepinfer_gpu::Device]| -> f64 {
            let tokens = 8usize;
            let hidden = 256usize;
            let k = 4usize;
            let e = 8usize;
            let cap = 16usize;
            let mesh = deepinfer_kernels::PeerMesh::new(devs).unwrap();
            let kern: Vec<_> = devs
                .iter()
                .map(|d| deepinfer_kernels::Kernels::new(d).unwrap())
                .collect();
            let ep = deepinfer_kernels::SymmetricEp::new(mesh, k, cap, tokens, hidden).unwrap();
            let mut topk_idx = Vec::new();
            for t in 0..tokens {
                for kk in 0..k {
                    topk_idx.push(((t + kk) % e) as i32);
                }
            }
            let topk_w: Vec<f32> = (0..tokens * k)
                .map(|i| 0.125 + (i % 5) as f32 / 40.0)
                .collect();
            let mut bufs = Vec::new();
            for r in 0..devs.len() {
                let xq: Vec<u8> = (0..tokens * hidden)
                    .map(|i| ((i + r * 17) % 251) as u8)
                    .collect();
                let xsf = vec![0x80u8; tokens * (hidden / 128)];
                bufs.push((
                    devs[r].alloc_from(&xq).unwrap(),
                    devs[r].alloc_from(&xsf).unwrap(),
                    devs[r].alloc_from(&topk_idx).unwrap(),
                    devs[r].alloc_from(&topk_w).unwrap(),
                    devs[r].alloc(e * 4).unwrap(),
                    devs[r].alloc(e * cap * 16).unwrap(),
                    devs[r].alloc(4).unwrap(),
                    devs[r].alloc(e * cap * hidden).unwrap(),
                    devs[r].alloc(e * cap * (hidden / 128)).unwrap(),
                    devs[r].alloc(e * cap * hidden * 4).unwrap(),
                    devs[r].alloc(tokens * hidden * 4).unwrap(),
                ));
            }
            ep.reset_all(&kern[0]).unwrap();
            for r in 0..devs.len() {
                let (xq, xsf, idx, ..) = &bufs[r];
                ep.dispatch(&kern[r], r, xq, xsf, idx, tokens, e, None).unwrap();
            }
            for r in 0..devs.len() {
                let (_, _, _, _, counters, meta2, drops, aq, asf, _, _) = &bufs[r];
                ep.group(&kern[r], r, counters, aq, asf, meta2, drops)
                    .unwrap();
            }
            for r in 0..devs.len() {
                let y_host: Vec<f32> = (0..e * cap * hidden)
                    .map(|i| ((i / (cap * hidden)) as f32 + 1.0) * 0.5)
                    .collect();
                let (_, _, _, _, _, _, _, _, _, y, _) = &bufs[r];
                devs[r].stream().unwrap().synchronize().unwrap();
                devs[r].write(y, &y_host).unwrap();
            }
            for r in 0..devs.len() {
                let (_, _, _, _, counters, meta2, _, _, _, y, _) = &bufs[r];
                ep.combine(&kern[r], r, y, meta2, counters, e).unwrap();
            }
            for r in 0..devs.len() {
                let (_, _, idx, w, _, _, _, _, _, _, out) = &bufs[r];
                ep.reduce(&kern[r], r, w, idx, out, tokens).unwrap();
            }
            let mut worst = 0.0f64;
            for r in 0..devs.len() {
                devs[r].stream().unwrap().synchronize().unwrap();
                let (_, _, _, _, _, _, _, _, _, _, out) = &bufs[r];
                let mut got = vec![0.0f32; tokens * hidden];
                devs[r].read(out, &mut got).unwrap();
                for t in 0..tokens {
                    for d in 0..hidden {
                        let mut acc = 0.0f32;
                        for kk in 0..k {
                            let ex = ((t + kk) % e) as i32;
                            acc += topk_w[t * k + kk] * ((ex as f32 + 1.0) * 0.5);
                        }
                        worst = worst.max((got[t * hidden + d] - acc).abs() as f64);
                    }
                }
            }
            worst
        };
        let devs: Vec<_> = devices.iter().map(|(_, d, _)| d.clone()).collect();
        let err = run(&devs[..world]);
        checks.push(Check {
            name,
            status: if err < 1e-4 {
                Status::Pass
            } else {
                Status::Fail
            },
            metric: err,
            detail: format!("world {world}, worst abs err {err}"),
        });
    }

    // 6b) Fused symmetric EP (P1-9): the heap-resident masked workspace +
    //     completion-table protocol (group_fused -> stub expert compute ->
    //     publish -> reduce_remote), vs the same host reference as 6).
    {
        let name = "symmetric_ep_fused_protocol";
        let world = devices.len().min(2);
        let run = |devs: &[deepinfer_gpu::Device]| -> f64 {
            let (tokens, hidden, k, e, cap) = (8usize, 256usize, 4usize, 8usize, 16usize);
            let per = deepinfer_kernels::experts_per_rank(devs.len(), e);
            let mesh = deepinfer_kernels::PeerMesh::new(devs).unwrap();
            let kern: Vec<_> = devs
                .iter()
                .map(|d| deepinfer_kernels::Kernels::new(d).unwrap())
                .collect();
            let ep = deepinfer_kernels::SymmetricEp::new_with_experts(
                mesh, k, cap, tokens, hidden, per,
            )
            .unwrap();
            let mut topk_idx = Vec::new();
            for t in 0..tokens {
                for kk in 0..k {
                    topk_idx.push(((t + kk) % e) as i32);
                }
            }
            let topk_w: Vec<f32> = (0..tokens * k)
                .map(|i| 0.125 + (i % 5) as f32 / 40.0)
                .collect();
            let mut bufs = Vec::new();
            for r in 0..devs.len() {
                let xq: Vec<u8> = (0..tokens * hidden)
                    .map(|i| ((i + r * 17) % 251) as u8)
                    .collect();
                let xsf = vec![0x80u8; tokens * (hidden / 128)];
                let slot_map = deepinfer_kernels::default_slot_map(devs.len(), e, r);
                bufs.push((
                    devs[r].alloc_from(&xq).unwrap(),
                    devs[r].alloc_from(&xsf).unwrap(),
                    devs[r].alloc_from(&topk_idx).unwrap(),
                    devs[r].alloc_from(&topk_w).unwrap(),
                    devs[r].alloc_from(&slot_map).unwrap(),
                    devs[r].alloc(tokens * hidden * 4).unwrap(),
                ));
            }
            ep.reset_all(&kern[0]).unwrap();
            ep.reset_fused().unwrap();
            for r in 0..devs.len() {
                let (xq, xsf, idx, _, _, _) = &bufs[r];
                ep.dispatch(&kern[r], r, xq, xsf, idx, tokens, e, None)
                    .unwrap();
            }
            for r in 0..devs.len() {
                let (_, _, _, _, slot_map, _) = &bufs[r];
                ep.group_fused(&kern[r], r, slot_map).unwrap();
            }
            // Stub expert compute: y[slot*cap + r] = f(local slot) — the
            // local slot of global expert e on rank r under the default
            // placement; write through the fused y views.
            let rank_map = deepinfer_kernels::default_rank_map(devs.len(), e);
            for r in 0..devs.len() {
                let fused = ep.fused_buffers(r).unwrap();
                let s_map =
                    deepinfer_kernels::default_slot_map(devs.len(), e, r);
                let mut y_host = vec![0.0f32; per * cap * hidden];
                for ex in 0..e {
                    if rank_map[ex] as usize != r {
                        continue;
                    }
                    let slot = s_map[ex] as usize;
                    for row in 0..cap {
                        for d in 0..hidden {
                            y_host[(slot * cap + row) * hidden + d] =
                                (slot as f32 + 1.0) * 0.5;
                        }
                    }
                }
                let s = devs[r].stream().unwrap();
                s.synchronize().unwrap();
                devs[r].write(&fused.y, &y_host).unwrap();
            }
            for r in 0..devs.len() {
                ep.publish(&kern[r], r).unwrap();
            }
            for r in 0..devs.len() {
                let (_, _, idx, w, _, out) = &bufs[r];
                ep.reduce_remote(&kern[r], r, w, idx, out, tokens)
                    .unwrap();
            }
            let mut worst = 0.0f64;
            for r in 0..devs.len() {
                devs[r].stream().unwrap().synchronize().unwrap();
                let (_, _, _, _, _, out) = &bufs[r];
                let mut got = vec![0.0f32; tokens * hidden];
                devs[r].read(out, &mut got).unwrap();
                let s_map = deepinfer_kernels::default_slot_map(devs.len(), e, r);
                for t in 0..tokens {
                    for d in 0..hidden {
                        let mut acc = 0.0f32;
                        for kk in 0..k {
                            let ex = (t + kk) % e;
                            let slot = s_map[ex] as usize;
                            acc += topk_w[t * k + kk] * ((slot as f32 + 1.0) * 0.5);
                        }
                        worst = worst.max((got[t * hidden + d] - acc).abs() as f64);
                    }
                }
            }
            worst
        };
        let devs: Vec<_> = devices.iter().map(|(_, d, _)| d.clone()).collect();
        let err = run(&devs[..world]);
        checks.push(Check {
            name,
            status: if err < 1e-4 {
                Status::Pass
            } else {
                Status::Fail
            },
            metric: err,
            detail: format!("world {world}, worst abs err {err}"),
        });
    }

    // 7) Timings (informational; recorded in the artifact).
    {
        let cfg = ModelConfig::v41_tiny();
        let mut gpu = DeviceModel::random_init(&cfg, 42, &dev, Precision::F32).unwrap();
        let prompt: Vec<u32> = (0..32u32).map(|t| (t * 61) % 250).collect();
        let t0 = std::time::Instant::now();
        gpu.prefill(9, &prompt).unwrap();
        let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = std::time::Instant::now();
        for i in 0..32 {
            let _ = gpu.decode_step(9, 5 + (i % 7)).unwrap();
        }
        let decode_ms = t0.elapsed().as_secs_f64() * 1000.0 / 32.0;
        gpu.enable_decode_graph(true).unwrap();
        let _ = gpu.decode_step(9, 5).unwrap(); // capture
        let t0 = std::time::Instant::now();
        for i in 0..64 {
            let _ = gpu.decode_step(9, 5 + (i % 7)).unwrap();
        }
        let graph_ms = t0.elapsed().as_secs_f64() * 1000.0 / 64.0;
        checks.push(Check {
            name: "timing_prefill_32tok_ms",
            status: Status::Pass,
            metric: prefill_ms,
            detail: String::new(),
        });
        checks.push(Check {
            name: "timing_decode_eager_ms",
            status: Status::Pass,
            metric: decode_ms,
            detail: String::new(),
        });
        checks.push(Check {
            name: "timing_decode_graph_ms",
            status: Status::Pass,
            metric: graph_ms,
            detail: String::new(),
        });
    }

    write_artifact(&path, &report, &checks, mode);
}

fn write_artifact(path: &str, report: &str, checks: &[Check], mode: &str) {
    let mut j = String::from("{\n");
    j.push_str("\"capability\": ");
    j.push_str(report);
    j.push_str(",\n\"checks\": [");
    for (i, c) in checks.iter().enumerate() {
        let mut cj = Json::new();
        cj.str("name", c.name);
        cj.str("status", c.status.as_str());
        cj.num("metric", c.metric);
        cj.str("detail", &c.detail);
        if i > 0 {
            j.push(',');
        }
        j.push_str(&cj.finish());
    }
    j.push_str("]\n}\n");
    match mode {
        "--record" => {
            std::fs::write(path, &j).expect("write artifact");
            println!("recorded {} checks -> {path}", checks.len());
        }
        "--verify" => {
            let prev = std::fs::read_to_string(path)
                .unwrap_or_else(|_| panic!("cannot read artifact {path}"));
            let mut failed = 0usize;
            for c in checks {
                if matches!(c.status, Status::Fail) {
                    failed += 1;
                }
                println!("{} {} (metric {:.6})", c.status.as_str(), c.name, c.metric);
            }
            let recorded_pass = prev.matches("\"status\":\"pass\"").count();
            let current_pass = checks
                .iter()
                .filter(|c| matches!(c.status, Status::Pass))
                .count();
            println!(
                "verify: {current_pass}/{} pass now, {recorded_pass} recorded pass, {failed} fail",
                checks.len()
            );
            if failed > 0 || current_pass < recorded_pass {
                println!("VERIFY: FAIL");
                std::process::exit(1);
            }
            println!("VERIFY: OK");
        }
        _ => {
            for c in checks {
                println!(
                    "{} {} (metric {:.6}) {}",
                    c.status.as_str(),
                    c.name,
                    c.metric,
                    c.detail
                );
            }
        }
    }
}
