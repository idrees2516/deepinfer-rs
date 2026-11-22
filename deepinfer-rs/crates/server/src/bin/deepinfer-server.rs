//! `deepinfer-server` — the serving binary.
//!
//! ```text
//! # tiny self-check (random weights, deterministic, no checkpoint):
//! deepinfer-server --tiny --port 8080
//!
//! # serve a DeepSeek-V4.1-Flash checkpoint (CPU engine):
//! deepinfer-server --model /path/to/DeepSeek-V4.1-Flash-1017
//!
//! # serve on the GPU (paged serve engine, FP8 auto):
//! deepinfer-server --model /path/to/checkpoint --gpu --precision auto
//!
//! # configuration file + env overrides:
//! deepinfer-server --config server.json
//! DEEPINFER_SERVER_PORT=9000 deepinfer-server --model ...
//! ```

use std::path::PathBuf;
use std::time::Instant;

use deepinfer_core::json::Json;
use deepinfer_core::tokenizer::{ByteTokenizer, Tokenizer};
use deepinfer_runtime::{DeepSeekModel, DeviceModel, ModelConfig, Precision};
use deepinfer_server::config::ServerConfig;
use deepinfer_server::engine::{CpuTokenEngine, GpuTokenEngine};
use deepinfer_server::metrics;
use deepinfer_server::Server;

/// Builds the vocabulary byte table for a model, padding to the model's
/// vocabulary when the tokenizer's is smaller (byte-mode serving on a
/// full checkpoint): ids beyond the tokenizer's alphabet are unreachable
/// — the engine's sampling cap masks them out — and carry empty bytes
/// so the structured-output mask stays consistent.
fn padded_vocab_bytes(tok: &dyn Tokenizer, eos: u32, model_vocab: usize) -> Vec<Vec<u8>> {
    let mut vb = DeepSeekModel::vocab_bytes_from_tokenizer(tok, eos).unwrap_or_else(|e| {
        eprintln!("error: cannot build the vocabulary byte table: {e}");
        std::process::exit(2);
    });
    if vb.len() < model_vocab {
        vb.resize(model_vocab, Vec::new());
    }
    vb
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| args.iter().any(|a| a == name);
    let value = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };

    if flag("--help") || flag("-h") {
        print_help();
        return;
    }
    let tiny = flag("--tiny");
    let gpu = flag("--gpu");
    let model_dir: Option<PathBuf> = value("--model").map(PathBuf::from);
    let config_path: Option<PathBuf> = value("--config").map(PathBuf::from);
    let seed: u64 = value("--seed").and_then(|v| v.parse().ok()).unwrap_or(42);
    // Tokenizer selection: `auto` (default) loads tokenizer.json when
    // present and falls back to the byte tokenizer with the DeepSeek
    // chat pieces; `byte` forces byte-mode serving (any checkpoint,
    // no tokenizer.json needed — output quality follows only if the
    // model was trained at the byte level).
    let tokenizer_mode = value("--tokenizer").unwrap_or_else(|| "auto".to_string());
    if !matches!(tokenizer_mode.as_str(), "auto" | "byte") {
        eprintln!("error: --tokenizer must be 'auto' or 'byte'");
        std::process::exit(2);
    }

    if !tiny && model_dir.is_none() {
        eprintln!("error: pass --tiny (random weights self-check) or --model <dir> (checkpoint)");
        eprintln!("usage: deepinfer-server [--tiny | --model DIR] [--gpu] [--config F] [--host H] [--port P] [--seed N] [--tokenizer auto|byte]");
        std::process::exit(2);
    }

    // ---- configuration --------------------------------------------------------
    let mut cfg = config_path
        .as_ref()
        .map(|p| match ServerConfig::from_file(p) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("error: cannot load config {}: {e}", p.display());
                std::process::exit(2);
            }
        })
        .unwrap_or_default();
    if let Some(h) = value("--host") {
        cfg.host = h;
    }
    if let Some(p) = value("--port").and_then(|v| v.parse::<u16>().ok()) {
        cfg.port = p;
    }
    if tiny || model_dir.is_none() {
        // Self-check defaults: low caps, fast deadlines.
        cfg.max_tokens_cap = cfg.max_tokens_cap.min(256);
    }
    cfg = cfg.apply_env();
    if let Err(e) = cfg.validate() {
        eprintln!("error: invalid server configuration: {e}");
        std::process::exit(2);
    }

    let started = Instant::now();
    let m = metrics::shared();

    // ---- model + tokenizer ------------------------------------------------------
    enum Loaded {
        Cpu(Box<DeepSeekModel>, Box<dyn Tokenizer>, u32, Vec<Vec<u8>>),
        Gpu(Box<DeviceModel>, Box<dyn Tokenizer>, u32, Vec<Vec<u8>>),
    }

    let loaded: Loaded = if tiny {
        if gpu {
            eprintln!("error: --gpu needs --model (random-init GPU weights are supported only through golden_run)");
            std::process::exit(2);
        }
        println!("--tiny self-check: random-init weights, seed {seed}");
        // The byte tokenizer with the DeepSeek chat pieces: the
        // conversation control tokens are single ids (256-259), so the
        // template costs 4 tokens instead of ~64 bytes of context.
        let tok: Box<dyn Tokenizer> =
            Box::new(ByteTokenizer::chat_specials().expect("chat specials"));
        let eos = tok.eos_token_id();
        let mut cfg = ModelConfig::v41_tiny();
        cfg.vocab_size = tok.vocab_size();
        cfg.mtp.vocab = tok.vocab_size();
        let model = match DeepSeekModel::random_init(&cfg, seed) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("error: random init failed: {e}");
                std::process::exit(2);
            }
        };
        let vocab_bytes =
            DeepSeekModel::vocab_bytes_from_tokenizer(tok.as_ref(), eos).expect("vocab bytes");
        let mut model = Box::new(model);
        model.set_vocab_bytes(vocab_bytes.clone());
        Loaded::Cpu(model, tok, eos, vocab_bytes)
    } else {
        let dir = model_dir.expect("checked");
        let cfg_path = dir.join("config.json");
        let cfg_text = match std::fs::read_to_string(&cfg_path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("error: cannot read {}: {e}", cfg_path.display());
                std::process::exit(2);
            }
        };
        let model_cfg = match Json::parse(&cfg_text)
            .map_err(|e| e.to_string())
            .and_then(|j| ModelConfig::from_json(&j).map_err(|e| e.to_string()))
        {
            Ok(c) => c,
            Err(e) => {
                eprintln!("error: invalid config.json: {e}");
                std::process::exit(2);
            }
        };
        // Tokenizer selection: `--tokenizer byte` forces byte-mode
        // serving; `auto` loads tokenizer.json (GPT-2-style BPE) when
        // present and falls back to the byte tokenizer with the
        // DeepSeek chat pieces. EOS comes from the tokenizer (the
        // offline example's contract).
        let tok: Box<dyn Tokenizer> = if tokenizer_mode == "byte" {
            println!("tokenizer: byte mode (forced; chat pieces as single tokens)");
            Box::new(ByteTokenizer::chat_specials().expect("chat specials"))
        } else {
            match std::fs::read_to_string(dir.join("tokenizer.json")) {
                Ok(text) => match Json::parse(&text) {
                    Ok(j) => match deepinfer_core::tokenizer::BpeTokenizer::from_tokenizer_json(&j)
                    {
                        Ok(b) => Box::new(b),
                        Err(e) => {
                            eprintln!("warning: tokenizer.json unreadable ({e}); falling back to the byte tokenizer");
                            Box::new(ByteTokenizer::chat_specials().expect("chat specials"))
                        }
                    },
                    Err(e) => {
                        eprintln!("warning: tokenizer.json invalid ({e}); falling back to the byte tokenizer");
                        Box::new(ByteTokenizer::chat_specials().expect("chat specials"))
                    }
                },
                Err(_) => {
                    eprintln!(
                        "warning: no tokenizer.json; using the byte tokenizer with chat specials"
                    );
                    Box::new(ByteTokenizer::chat_specials().expect("chat specials"))
                }
            }
        };
        let eos = tok.eos_token_id();
        if gpu {
            let dev = match DeviceModel::probe_device() {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("error: no GPU available: {e}");
                    std::process::exit(2);
                }
            };
            println!("loading checkpoint onto the device ...");
            let prec = match value("--precision").as_deref() {
                Some("f32") => Precision::F32,
                _ => Precision::Auto,
            };
            let mut model = match DeviceModel::load_checkpoint(&dir, &model_cfg, &dev, prec) {
                Ok(m) => Box::new(m),
                Err(e) => {
                    eprintln!("error: checkpoint load failed: {e}");
                    std::process::exit(2);
                }
            };
            let vocab_bytes = padded_vocab_bytes(tok.as_ref(), eos, model_cfg.vocab_size);
            model.set_vocab_bytes(vocab_bytes.clone());
            Loaded::Gpu(model, tok, eos, vocab_bytes)
        } else {
            println!("loading checkpoint from {} ...", dir.display());
            let mut model = match DeepSeekModel::load_checkpoint(&dir, &model_cfg) {
                Ok(m) => Box::new(m),
                Err(e) => {
                    eprintln!("error: checkpoint load failed: {e}");
                    std::process::exit(2);
                }
            };
            let vocab_bytes = padded_vocab_bytes(tok.as_ref(), eos, model_cfg.vocab_size);
            model.set_vocab_bytes(vocab_bytes.clone());
            Loaded::Cpu(model, tok, eos, vocab_bytes)
        }
    };

    // ---- start the server -----------------------------------------------------
    let server = match loaded {
        Loaded::Cpu(model, tok, eos, vocab_bytes) => {
            let engine = CpuTokenEngine::new(*model);
            Server::start(cfg, Box::new(engine), tok, eos, vocab_bytes, m, "cpu")
        }
        Loaded::Gpu(model, tok, eos, vocab_bytes) => {
            let engine = GpuTokenEngine::new(*model).with_prefix_tokens(cfg.prefix_cache_tokens);
            Server::start(cfg, Box::new(engine), tok, eos, vocab_bytes, m, "gpu")
        }
    };
    let mut server = match server {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: server start failed: {e}");
            std::process::exit(2);
        }
    };

    println!(
        "deepinfer-server listening on http://{} (engine: {}, model: {}, ready in {:.1?})",
        server.local_addr(),
        if server.engine().failed() {
            "DEGRADED"
        } else {
            "up"
        },
        server.config().model_name,
        started.elapsed()
    );
    let spec = if server.config().spec_draft_len > 0 {
        format!(
            "DSpark speculative decoding: draft {} per step",
            server.config().spec_draft_len
        )
    } else {
        "DSpark speculative decoding: off".to_string()
    };
    println!("endpoints: /health /v1/models /v1/chat/completions /v1/completions /metrics");
    println!("{spec}");
    println!("press ctrl-c to stop (in-flight requests drain)");

    server.wait();
    println!("accept loop exited; server stopped");
}

fn print_help() {
    println!(
        "deepinfer-server — OpenAI-compatible serving for DeepInfer-rs

USAGE:
    deepinfer-server --tiny [FLAGS]                  random-weight self-check
    deepinfer-server --model <DIR> [FLAGS]            serve a checkpoint

FLAGS:
    --gpu                    use the GPU DeviceModel serve engine (needs a CUDA/HIP device)
    --precision f32|auto     GPU precision (default auto = FP8 where dims allow)
    --tokenizer auto|byte    tokenizer selection (default auto: tokenizer.json when
                             present, else the byte tokenizer; byte forces byte-mode
                             serving — any checkpoint serves without tokenizer.json)
    --config <FILE>          server JSON configuration
    --host <HOST>            bind host (default 0.0.0.0; env DEEPINFER_SERVER_HOST)
    --port <PORT>            bind port (default 8080; env DEEPINFER_SERVER_PORT)
    --seed <N>               --tiny random-init seed (default 42)
    --help                   this help

ENVIRONMENT:
    DEEPINFER_SERVER_API_KEY                require 'Authorization: Bearer <key>' on /v1/*
    DEEPINFER_SERVER_MAX_TOKENS_CAP         per-request max_tokens ceiling
    DEEPINFER_SERVER_MAX_PROMPT_TOKENS      per-request prompt-token ceiling (400 past it)
    DEEPINFER_SERVER_REQUEST_TIMEOUT_SECS   per-request wall-clock deadline
    DEEPINFER_SERVER_SCHED_*                scheduler knobs (see ServerConfig docs)
    DEEPINFER_SERVER_SPEC_DRAFT_LEN         DSpark draft tokens per step (0 = off; default 5)
    DEEPINFER_SERVER_SPEC_ADAPTIVE          adapt draft depth to acceptance (default true)
    DEEPINFER_SERVER_PREFIX_CACHE_TOKENS    prefix-cache budget in tokens (0 = off; default 4096)"
    );
}
