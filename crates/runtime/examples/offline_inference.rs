//! Offline inference for DeepInfer-rs — the Rust mirror of upstream
//! DeepInfer's `examples/offline_inference.py`.
//!
//! ```text
//! # Tiny self-test (random-init weights, no checkpoint needed):
//! cargo run -p deepinfer-runtime --example offline_inference -- --tiny
//!
//! # Full checkpoint inference (DeepSeek-V4.1-Flash layout):
//! DEEPINFER_HF_MODEL_DIR=/path/to/DeepSeek-V4.1-Flash-1017 \
//!   cargo run -p deepinfer-runtime --example offline_inference --release
//! ```
//!
//! The tiny mode is fully deterministic: identical seeds produce
//! identical completions bit-for-bit.

use deepinfer_core::env::Env;
use deepinfer_core::tokenizer::{ByteTokenizer, Tokenizer};
use deepinfer_runtime::{generate, DeepSeekModel, GenerateRequest, ModelConfig};
use deepinfer_sampler::SamplingParams;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let tiny = args.iter().any(|a| a == "--tiny");
    let model_dir = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from);

    // --- environment contract (DEEPINFER_*) ---
    let report = Env::from_env();
    let env = report.env.clone();
    println!("=== DeepInfer-rs offline inference ===");
    println!("accelerator: {:?}", env.accelerator);
    println!("use_nextn_decode (MTP draft len): {}", env.use_nextn_decode);
    println!("enable_mtp_varlen: {}", env.enable_mtp_varlen);
    println!(
        "yoco prefill truncation: {:?} (keep {})",
        env.yoco_prefill_truncation_mode, env.yoco_prefill_keep_tokens
    );
    println!("batch_invariant: {}", env.batch_invariant);
    for w in &report.warnings {
        eprintln!("warning: {w}");
    }

    let prompts: Vec<String> = if tiny {
        vec![
            "Once upon a time".to_string(),
            "The capital of France is".to_string(),
        ]
    } else {
        vec![
            "Once upon a time".to_string(),
            "The capital of France is".to_string(),
            "Explain the theory of relativity in simple terms:".to_string(),
        ]
    };

    let start = Instant::now();
    let (mut model, tokenizer, eos_id) = if tiny {
        println!("\n--tiny self-test: random-init weights, seed 42--");
        let cfg = ModelConfig::v41_tiny();
        let model = DeepSeekModel::random_init(&cfg, 42).expect("random init");
        let tok: Box<dyn Tokenizer> = Box::new(ByteTokenizer::new(0).expect("byte tokenizer"));
        (model, tok, 0u32)
    } else {
        let dir = model_dir
            .or_else(|| env.hf_model_dir.clone())
            .unwrap_or_else(|| {
                eprintln!(
                    "no model directory: pass --model <dir> or set DEEPINFER_HF_MODEL_DIR"
                );
                std::process::exit(2);
            });
        println!("loading checkpoint from {} ...", dir.display());
        let cfg_path = dir.join("config.json");
        let cfg_text = std::fs::read_to_string(&cfg_path).unwrap_or_else(|e| {
            eprintln!("cannot read {}: {e}", cfg_path.display());
            std::process::exit(2);
        });
        let cfg_json = deepinfer_core::json::Json::parse(&cfg_text)
            .unwrap_or_else(|e| {
                eprintln!("invalid config.json: {e}");
                std::process::exit(2);
            });
        let cfg = ModelConfig::from_json(&cfg_json)
            .unwrap_or_else(|e| {
                eprintln!("config error: {e}");
                std::process::exit(2);
            });
        let model = DeepSeekModel::load_checkpoint(&dir, &cfg).unwrap_or_else(|e| {
            eprintln!("checkpoint error: {e}");
            std::process::exit(2);
        });
        // tokenizer.json (GPT-2-style BPE) when present, else bytes.
        let tok: Box<dyn Tokenizer> =
            match std::fs::read_to_string(dir.join("tokenizer.json")) {
                Ok(text) => match deepinfer_core::json::Json::parse(&text) {
                    Ok(json) => Box::new(
                        deepinfer_core::tokenizer::BpeTokenizer::from_tokenizer_json(&json)
                            .unwrap_or_else(|e| {
                                eprintln!("tokenizer error: {e}");
                                std::process::exit(2);
                            }),
                    ),
                    Err(e) => {
                        eprintln!("invalid tokenizer.json ({e}); falling back to bytes");
                        Box::new(ByteTokenizer::new(0).expect("byte tokenizer"))
                    }
                },
                Err(_) => {
                    println!("no tokenizer.json; using byte tokenizer");
                    Box::new(ByteTokenizer::new(0).expect("byte tokenizer"))
                }
            };
        let eos = tok.eos_token_id();
        (model, tok, eos)
    };
    println!("model ready in {:?}", start.elapsed());

    let params = SamplingParams {
        temperature: if tiny { 0.0 } else { 0.7 },
        top_p: 0.95,
        top_k: 0,
        min_p: 0.0,
        seq_seed: Some(42),
        so_args: None,
        max_tokens: if tiny { 12 } else { 48 },
    };

    let requests: Vec<GenerateRequest> = prompts
        .iter()
        .map(|p| GenerateRequest {
            prompt: tokenizer.encode(p).expect("encode"),
            params: params.clone(),
        })
        .collect();

    let start = Instant::now();
    let outputs = generate(&mut model, requests, eos_id, &env).unwrap_or_else(|e| {
        eprintln!("generation failed: {e}");
        std::process::exit(3);
    });
    println!("generation finished in {:?}", start.elapsed());

    for (i, out) in outputs.iter().enumerate() {
        let text = tokenizer.decode(&out.tokens).unwrap_or_else(|_| "<decode error>".into());
        println!("\n--- prompt #{i}: {:?} ---", prompts.get(i).map(String::as_str).unwrap_or(""));
        println!("tokens: {:?}", out.tokens);
        println!("finish: {}", out.finish);
        println!("text:   {text}");
    }

    if tiny {
        // Determinism check: a second identical run must match exactly.
        let mut model2 = DeepSeekModel::random_init(&ModelConfig::v41_tiny(), 42).unwrap();
        let requests2: Vec<GenerateRequest> = prompts
            .iter()
            .map(|p| GenerateRequest {
                prompt: tokenizer.encode(p).unwrap(),
                params: params.clone(),
            })
            .collect();
        let outputs2 = generate(&mut model2, requests2, eos_id, &env).unwrap();
        let same = outputs
            .iter()
            .zip(outputs2.iter())
            .all(|(a, b)| a.tokens == b.tokens && a.finish == b.finish);
        println!("\ndeterminism self-check: {}", if same { "PASS" } else { "FAIL" });
        assert!(same, "tiny self-test must be deterministic");
    }
}
