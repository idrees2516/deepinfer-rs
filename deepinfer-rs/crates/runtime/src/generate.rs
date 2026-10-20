//! The `generate()` API: deterministic sequential generation with
//! DSpark/DSpark speculative decoding and structured outputs.
//!
//! Mirrors the offline-inference entry point of DeepSeek's DeepInfer
//! release (upstream `examples/offline_inference.py`; README: "DSpark MTP
//! + confidence-scheduled verification", "deterministic sampling",
//! "batch-invariant inference", "structured outputs").
//!
//! Layering note: this path is deliberately **simple and sequential** —
//! one request at a time, each on a freshly reset cache, in arrival
//! order. That trivially guarantees batch invariance on the CPU
//! reference: a request's result depends only on `(model, prompt,
//! params, eos_id)`. The `deepinfer-scheduler` crate provides the
//! continuous-batching / chunked-prefill / preemption engine for service
//! deployments; it sits *on top of* an engine abstraction, not under
//! this function.
//!
//! MTP/DSpark (enabled when `env.use_nextn_decode > 1` and the model has
//! MTP weights): draft up to N tokens with the MTP head from the last
//! accepted `(token, hidden)` anchor, verify against the target's
//! `decode_step` logits (greedy prefix acceptance at temperature 0,
//! seeded resampling otherwise), accept the prefix, and roll the
//! rejected tail back out of the caches. Because every accepted token
//! was produced by the same `decode_step` call the plain path would have
//! made — and rollback restores the caches bit-exactly — **greedy
//! generation with MTP on is token-identical to greedy generation with
//! MTP off** (the property pinned by the tests).
//!
//! Structured outputs: `params.so_args` compiles a
//! [`deepinfer_grammar::Grammar`] and computes per-step token masks over
//! the vocabulary bytes; the sampler respects the mask, and EOS is only
//! allowed from accepting states. When a grammar is active the
//! speculative path is disabled (the draft head is grammar-unaware —
//! combining the two requires draft-time masking, which the upstream
//! DSpark contract does not define).

use deepinfer_core::env::Env;
use deepinfer_core::math::{argmax, softmax_inplace};
use deepinfer_core::{ensure, Error, Result};
use deepinfer_grammar::{Grammar, TokenMaskComputer};
use deepinfer_mtp::{
    backtrack_steps, verify_greedy, verify_stochastic, Decision, DraftToken, DsparkScheduler,
};
use deepinfer_sampler::SchemaType;
use deepinfer_sampler::{Sampler, SamplingParams, GREEDY_TEMPERATURE_EPS};

use crate::model::DeepSeekModel;

/// One generation request: prompt token ids plus sampling parameters.
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    /// Prompt token ids (non-empty; every id must be `< vocab_size`).
    pub prompt: Vec<u32>,
    /// Sampling parameters for this request.
    pub params: SamplingParams,
}

/// One generation result.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerateOutput {
    /// Generated token ids (the prompt is not included).
    pub tokens: Vec<u32>,
    /// Finish reason: `"stop"` (EOS sampled), `"length"` (max_tokens
    /// reached), or `"dead"` (the structured-output grammar mask
    /// admitted no continuation).
    pub finish: String,
}

/// Generates completions for `requests` on `model`, sequentially in
/// arrival order (batch-invariant by construction on the CPU path — see
/// the module docs). Each request runs on a freshly reset cache with its
/// own sequence id (`0..n`) and its own Philox stream derived via
/// [`Sampler::seq_rng`], so `params.seq_seed` makes results
/// reproducible.
///
/// # Errors
/// [`Error::OutOfRange`] when `eos_id` or a prompt token is outside the
/// vocabulary; parameter-validation errors from
/// [`SamplingParams::validate`]; [`Error::Grammar`] when a structured
/// output schema does not compile; forward-pass errors (cache capacity,
/// non-finite states) bubble up unchanged.
pub fn generate(
    model: &mut DeepSeekModel,
    requests: Vec<GenerateRequest>,
    eos_id: u32,
    env: &Env,
) -> Result<Vec<GenerateOutput>> {
    let vocab = model.config().vocab_size;
    ensure!(
        (eos_id as usize) < vocab,
        OutOfRange,
        "eos_id {eos_id} is outside the vocabulary ({vocab})"
    );
    for (i, req) in requests.iter().enumerate() {
        req.params.validate(vocab).map_err(|e| {
            Error::invalid(format!("request {i}: invalid sampling params: {e}"))
        })?;
        ensure!(
            !req.prompt.is_empty(),
            InvalidArgument,
            "request {i}: prompt must contain at least one token"
        );
        for &t in &req.prompt {
            ensure!(
                (t as usize) < vocab,
                OutOfRange,
                "request {i}: prompt token {t} is outside the vocabulary ({vocab})"
            );
        }
    }
    // Vocabulary bytes for structured outputs (resolved once).
    let vocab_bytes = if requests.iter().any(|r| r.params.so_args.is_some()) {
        Some(model.resolve_vocab_bytes(eos_id)?)
    } else {
        None
    };

    let mut outputs = Vec::with_capacity(requests.len());
    for (i, req) in requests.into_iter().enumerate() {
        // Fresh, isolated caches per request (sequential CPU path).
        model.reset();
        outputs.push(generate_one(
            model,
            i as u32,
            req,
            eos_id,
            env,
            vocab_bytes.as_deref(),
        )?);
    }
    Ok(outputs)
}

/// Runs one request on sequence `seq`.
fn generate_one(
    model: &mut DeepSeekModel,
    seq: u32,
    req: GenerateRequest,
    eos_id: u32,
    env: &Env,
    vocab_bytes: Option<&[Vec<u8>]>,
) -> Result<GenerateOutput> {
    let GenerateRequest { prompt, params } = req;
    let max_tokens = params.max_tokens;
    let (mut logits, mut hidden) = model.forward_tokens(seq, &prompt)?;
    let sampler = Sampler::new(params.clone());
    let mut rng = Sampler::seq_rng(params.seq_seed, seq as u64);
    let greedy = params.temperature <= 0.0 || params.temperature.abs() < GREEDY_TEMPERATURE_EPS;

    // Structured-output setup.
    let mut grammar = None;
    let mut masker = None;
    let mut gstate;
    if let Some(so) = &params.so_args {
        let type_str = match so.schema_type {
            SchemaType::Regex => "regex",
            SchemaType::Json => "json",
            SchemaType::Ebnf => "ebnf",
        };
        let g = Grammar::from_so_args(type_str, &so.schema)?;
        let vb = vocab_bytes.ok_or_else(|| {
            Error::invalid("structured output requires vocabulary bytes")
        })?;
        gstate = g.initial_state();
        masker = Some(TokenMaskComputer::new(vb.to_vec()));
        grammar = Some(g);
    } else {
        gstate = 0;
    }

    // Speculative path: MTP weights present, nextn > 1, no grammar.
    let use_mtp = env.use_nextn_decode > 1 && model.has_mtp() && grammar.is_none();
    let mut tokens: Vec<u32> = Vec::new();
    let mut finish = String::from("length");

    loop {
        let remaining = max_tokens - tokens.len();
        if remaining == 0 {
            break;
        }
        if use_mtp {
            let stop = speculative_round(
                model,
                seq,
                &prompt,
                &mut tokens,
                &mut logits,
                &mut hidden,
                &sampler,
                &mut rng,
                greedy,
                &params,
                eos_id,
                env,
            )?;
            if let Some(reason) = stop {
                finish = reason;
                break;
            }
            continue;
        }
        // ---- plain autoregressive step ----
        let mask = if let (Some(g), Some(m)) = (&grammar, &mut masker) {
            let mut mk = m.compute_mask(g, gstate);
            if !g.is_accepting(gstate) {
                mk.clear(eos_id);
            }
            if !mk.any() {
                finish = String::from("dead");
                break;
            }
            Some(mk)
        } else {
            None
        };
        let tok = sampler.sample(&logits, mask.as_ref(), &mut rng)?;
        if let (Some(g), Some(m)) = (&grammar, &masker) {
            match m.advance_token(g, gstate, tok) {
                Some(next) => gstate = next,
                None => {
                    // Unreachable for mask-allowed tokens (the pruned
                    // DFA guarantees liveness); defensive stop.
                    finish = String::from("dead");
                    break;
                }
            }
        }
        tokens.push(tok);
        if tok == eos_id {
            finish = String::from("stop");
            break;
        }
        if tokens.len() >= max_tokens {
            break;
        }
        let (l, h) = model.forward_tokens(seq, &[tok])?;
        logits = l;
        hidden = h;
    }
    Ok(GenerateOutput { tokens, finish })
}

/// Result of one speculative round: `Some(finish_reason)` stops the
/// request, `None` continues with `(logits, hidden)` advanced past the
/// round's accepted tokens.
#[allow(clippy::too_many_arguments)]
fn speculative_round(
    model: &mut DeepSeekModel,
    seq: u32,
    prompt: &[u32],
    tokens: &mut Vec<u32>,
    logits: &mut Vec<f32>,
    hidden: &mut Vec<f32>,
    sampler: &Sampler,
    rng: &mut deepinfer_core::rng::Philox,
    greedy: bool,
    params: &SamplingParams,
    eos_id: u32,
    env: &Env,
) -> Result<Option<String>> {
    let max_tokens = params.max_tokens;
    let mtp = model
        .mtp()
        .ok_or_else(|| Error::internal("speculative_round without MTP weights"))?;
    let vocab = model.config().vocab_size;
    let hidden_len = model.config().hidden_size;
    let sched = DsparkScheduler::new(env.use_nextn_decode);
    let max_draft = env.use_nextn_decode.min(max_tokens - tokens.len());

    // ---- draft chain from the last accepted anchor ----
    let mut drafts: Vec<DraftToken> = Vec::new();
    let mut last_tok = if let Some(&t) = tokens.last() {
        t
    } else {
        *prompt
            .last()
            .ok_or_else(|| Error::internal("prompt cannot be empty here"))?
    };
    while drafts.len() < max_draft && sched.should_verify(&drafts) == Decision::ContinueDraft {
        let d = draft_token(mtp, &model.weights.embed, vocab, hidden_len, last_tok, hidden)?;
        last_tok = d.token;
        drafts.push(d);
    }

    // ---- verify against the target (decode through the drafts) ----
    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(drafts.len() + 1);
    rows.push(logits.clone());
    for d in &drafts {
        let (l, h) = model.forward_tokens(seq, &[d.token])?;
        rows.push(l);
        *hidden = h;
    }
    let outcome = if greedy {
        verify_greedy(&drafts, &rows)?
    } else {
        verify_stochastic(&drafts, &rows, rng, params.temperature)?
    };
    let accepted: Vec<u32> = drafts[..outcome.accepted].iter().map(|d| d.token).collect();

    // EOS inside the accepted prefix ends the request there.
    if let Some(e) = accepted.iter().position(|&t| t == eos_id) {
        let keep = e + 1;
        tokens.extend_from_slice(&accepted[..keep]);
        model.rollback_seq(seq, drafts.len() - keep)?;
        return Ok(Some(String::from("stop")));
    }
    tokens.extend_from_slice(&accepted);
    if tokens.len() >= max_tokens {
        model.rollback_seq(seq, backtrack_steps(&drafts, &outcome))?;
        return Ok(Some(String::from("length")));
    }
    // Roll the rejected tail out of the caches.
    let bt = backtrack_steps(&drafts, &outcome);
    if bt > 0 {
        model.rollback_seq(seq, bt)?;
    }
    // Bonus token: the target's continuation after the accepted prefix.
    let bonus = outcome.next_token;
    tokens.push(bonus);
    if bonus == eos_id {
        return Ok(Some(String::from("stop")));
    }
    if tokens.len() >= max_tokens {
        return Ok(Some(String::from("length")));
    }
    let (l, h) = model.forward_tokens(seq, &[bonus])?;
    *logits = l;
    *hidden = h;
    let _ = sampler; // sampling happens in the plain path / verify_stochastic
    Ok(None)
}

/// One MTP draft step — the exact semantics of
/// `deepinfer_mtp::NextNetDraft::draft_step` (the concrete draft network
/// cannot be assembled from checkpoint weights because its constructor
/// only offers random init, so the runtime replicates the 10-line
/// greedy-argmax + softmax-confidence step over
/// [`deepinfer_mtp::MtpWeights::forward_logits`] and the model's shared
/// embedding).
fn draft_token(
    mtp: &deepinfer_mtp::MtpWeights,
    embed: &[f32],
    vocab: usize,
    hidden: usize,
    last_token: u32,
    hidden_state: &[f32],
) -> Result<DraftToken> {
    ensure!(
        (last_token as usize) < vocab,
        OutOfRange,
        "draft anchor token {last_token} is outside the vocabulary ({vocab})"
    );
    let emb = &embed[last_token as usize * hidden..(last_token as usize + 1) * hidden];
    let logits = mtp.forward_logits(hidden_state, emb)?;
    let idx = argmax(&logits).ok_or_else(|| Error::internal("empty MTP logits"))?;
    let mut probs = logits.clone();
    softmax_inplace(&mut probs);
    Ok(DraftToken {
        token: idx as u32,
        confidence: probs[idx],
        logits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use deepinfer_core::env::Env;
    use deepinfer_core::prop::prop;
    use deepinfer_sampler::StructuredOutputArgs;

    fn env_nextn(n: usize) -> Env {
        Env {
            use_nextn_decode: n,
            ..Env::default()
        }
    }

    fn seeded_params(max_tokens: usize) -> SamplingParams {
        SamplingParams {
            temperature: 0.7,
            top_p: 0.95,
            seq_seed: Some(42),
            max_tokens,
            ..Default::default()
        }
    }

    #[test]
    fn tiny_generate_is_deterministic_across_runs() {
        let prompts: Vec<Vec<u32>> = vec![
            b"Once upon a time".iter().map(|&b| b as u32).collect(),
            b"The quick brown fox".iter().map(|&b| b as u32).collect(),
        ];
        let run = || {
            let mut model =
                DeepSeekModel::random_init(&ModelConfig::v41_tiny(), 42).unwrap();
            let reqs: Vec<GenerateRequest> = prompts
                .iter()
                .cloned()
                .map(|p| GenerateRequest {
                    prompt: p,
                    params: seeded_params(8),
                })
                .collect();
            generate(&mut model, reqs, 0, &env_nextn(1)).unwrap()
        };
        let a = run();
        let b = run();
        assert_eq!(a, b);
        assert_eq!(a.len(), 2);
        for out in &a {
            assert_eq!(out.tokens.len(), 8, "{out:?}");
            assert_eq!(out.finish, "length");
            assert!(out.tokens.iter().all(|&t| t < 256));
        }
        // Different sequence ids derive different RNG streams: the two
        // seeded requests differ (with overwhelming probability).
        assert_ne!(a[0].tokens, a[1].tokens);
    }

    #[test]
    fn greedy_mtp_on_equals_mtp_off() {
        // THE speculative-decoding correctness property: greedy (temp 0)
        // generation with MTP enabled produces the exact same tokens as
        // the plain autoregressive path.
        let prompts: Vec<Vec<u32>> = vec![
            vec![9, 8, 7, 6],
            vec![1, 2, 3, 4, 5],
            vec![200, 150, 100],
        ];
        let greedy_params = |n| SamplingParams {
            temperature: 0.0,
            max_tokens: n,
            ..Default::default()
        };
        for nextn in [1usize, 2, 3, 5] {
            let mut model =
                DeepSeekModel::random_init(&ModelConfig::v41_tiny(), 2024).unwrap();
            assert!(model.has_mtp());
            let reqs: Vec<GenerateRequest> = prompts
                .iter()
                .cloned()
                .map(|p| GenerateRequest {
                    prompt: p,
                    params: greedy_params(12),
                })
                .collect();
            let outs = generate(&mut model, reqs, 0, &env_nextn(nextn)).unwrap();
            for (i, out) in outs.iter().enumerate() {
                assert_eq!(out.tokens.len(), 12, "nextn={nextn} prompt {i}: {out:?}");
                assert_eq!(out.finish, "length");
            }
            if nextn == 1 {
                // Reference: the plain path defines the expectation.
                for (i, out) in outs.iter().enumerate() {
                    assert!(
                        !out.tokens.is_empty(),
                        "nextn={nextn} prompt {i}: {out:?}"
                    );
                }
                // Compare against a second plain run to pin the tokens.
                let mut again =
                    DeepSeekModel::random_init(&ModelConfig::v41_tiny(), 2024).unwrap();
                let reqs: Vec<GenerateRequest> = prompts
                    .iter()
                    .cloned()
                    .map(|p| GenerateRequest {
                        prompt: p,
                        params: greedy_params(12),
                    })
                    .collect();
                let plain = generate(&mut again, reqs, 0, &env_nextn(1)).unwrap();
                assert_eq!(outs, plain);
                // Stash for the MTP comparisons below.
                PLAIN_TOKENS.with(|c| *c.borrow_mut() = Some(plain));
            } else {
                PLAIN_TOKENS.with(|c| {
                    let plain = c.borrow().clone().unwrap();
                    assert_eq!(
                        outs, plain,
                        "greedy tokens diverged with MTP nextn={nextn}"
                    );
                });
            }
        }
    }

    thread_local! {
        static PLAIN_TOKENS: std::cell::RefCell<Option<Vec<GenerateOutput>>> =
            const { std::cell::RefCell::new(None) };
    }

    #[test]
    fn mtp_seeded_sampling_is_deterministic() {
        let run = || {
            let mut model =
                DeepSeekModel::random_init(&ModelConfig::v41_tiny(), 7).unwrap();
            generate(
                &mut model,
                vec![GenerateRequest {
                    prompt: vec![5, 6, 7],
                    params: seeded_params(10),
                }],
                0,
                &env_nextn(3),
            )
            .unwrap()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn mtp_path_exercises_rollback_safely() {
        // With a random draft head, rejections happen; the loop must
        // still terminate and stay in-vocab (state restored exactly).
        let mut model = DeepSeekModel::random_init(&ModelConfig::v41_tiny(), 31337).unwrap();
        let out = generate(
            &mut model,
            vec![GenerateRequest {
                prompt: vec![11, 22, 33],
                params: SamplingParams {
                    temperature: 0.0,
                    max_tokens: 20,
                    ..Default::default()
                },
            }],
            0,
            &Env::default(), // use_nextn_decode = 5 -> MTP on
        )
        .unwrap();
        assert_eq!(out[0].tokens.len(), 20);
        assert!(out[0].tokens.iter().all(|&t| t < 256));
        assert_eq!(out[0].finish, "length");
    }

    #[test]
    fn structured_output_regex_digits() {
        let mut model = DeepSeekModel::random_init(&ModelConfig::v41_tiny(), 5).unwrap();
        let out = generate(
            &mut model,
            vec![GenerateRequest {
                prompt: vec![1, 2, 3],
                params: SamplingParams {
                    temperature: 0.0,
                    max_tokens: 8,
                    so_args: Some(StructuredOutputArgs {
                        schema_type: SchemaType::Regex,
                        schema: "[0-9]{3}".to_string(),
                    }),
                    ..Default::default()
                },
            }],
            0,
            &env_nextn(1),
        )
        .unwrap()
        .pop()
        .unwrap();
        assert_eq!(out.finish, "stop", "{out:?}");
        // 3 digit tokens + the appended EOS control token.
        assert_eq!(out.tokens.len(), 4, "{out:?}");
        assert_eq!(*out.tokens.last().unwrap(), 0, "EOS appended: {out:?}");
        let digits = &out.tokens[..3];
        let bytes: Vec<u8> = digits.iter().map(|&t| t as u8).collect();
        assert!(bytes.iter().all(|b| b.is_ascii_digit()), "{bytes:?}");
        // Walk the grammar: every emitted digit keeps a live state and
        // the final state is accepting (EOS was allowed).
        let g = Grammar::compile_regex("[0-9]{3}").unwrap();
        let mut state = g.initial_state();
        for &t in digits {
            state = g
                .advance(state, t as u8)
                .expect("generated token must stay on a live state");
        }
        assert!(g.is_accepting(state));
    }

    #[test]
    fn structured_output_prefix_never_dead() {
        // Property: with a live grammar, generation never walks into a
        // dead state and never emits a masked token.
        prop("runtime_grammar_prefix", 0xC0FFEE, 12, |_i, rng| {
            let mut model = DeepSeekModel::random_init(&ModelConfig::v41_tiny(), 77).unwrap();
            let prompt: Vec<u32> = (0..rng.uniform_usize(1, 5))
                .map(|_| rng.uniform_usize(0, 256) as u32)
                .collect();
            let out = generate(
                &mut model,
                vec![GenerateRequest {
                    prompt,
                    params: SamplingParams {
                        temperature: 0.0,
                        max_tokens: 6,
                        so_args: Some(StructuredOutputArgs {
                            schema_type: SchemaType::Regex,
                            schema: "[0-9]{2,4}".to_string(),
                        }),
                        ..Default::default()
                    },
                }],
                0,
                &env_nextn(1),
            )?
            .pop()
            .unwrap();
            let g = Grammar::compile_regex("[0-9]{2,4}")?;
            let mut state = g.initial_state();
            for &t in &out.tokens {
                if t == 0 {
                    // EOS: allowed exactly at accepting states.
                    ensure!(
                        g.is_accepting(state),
                        Internal,
                        "EOS emitted at a non-accepting grammar state"
                    );
                    break;
                }
                let b = t as u8;
                ensure!(b.is_ascii_digit(), Internal, "non-digit token {t}");
                state = g
                    .advance(state, b)
                    .ok_or_else(|| Error::internal("dead grammar state"))?;
            }
            ensure!(
                out.finish == "stop" || out.finish == "length" || out.finish == "dead",
                Internal,
                "bad finish {:?}",
                out.finish
            );
            ensure!(out.tokens.len() <= 6, Internal, "max_tokens exceeded");
            Ok(())
        });
    }

    #[test]
    fn property_generation_always_terminates() {
        // One model, random prompts, greedy: every request terminates
        // within max_tokens with in-vocab tokens.
        let mut model = DeepSeekModel::random_init(&ModelConfig::v41_tiny(), 123).unwrap();
        prop("runtime_terminates", 0x7E2A, 20, |i, rng| {
            let prompt: Vec<u32> = (0..rng.uniform_usize(1, 8))
                .map(|_| rng.uniform_usize(0, 256) as u32)
                .collect();
            let max_tokens = rng.uniform_usize(1, 7);
            let temp = if i % 2 == 0 { 0.0 } else { 0.9 };
            let out = generate(
                &mut model,
                vec![GenerateRequest {
                    prompt,
                    params: SamplingParams {
                        temperature: temp,
                        top_p: 0.9,
                        seq_seed: Some(1234),
                        max_tokens,
                        ..Default::default()
                    },
                }],
                0,
                &env_nextn(if i % 3 == 0 { 1 } else { 3 }),
            )?
            .pop()
            .unwrap();
            ensure!(
                out.tokens.len() <= max_tokens,
                Internal,
                "generated {} > max_tokens {max_tokens}",
                out.tokens.len()
            );
            ensure!(
                out.tokens.iter().all(|&t| (t as usize) < 256),
                Internal,
                "out-of-vocab token"
            );
            ensure!(
                matches!(out.finish.as_str(), "stop" | "length" | "dead"),
                Internal,
                "bad finish {:?}",
                out.finish
            );
            Ok(())
        });
    }

    #[test]
    fn generate_validates_requests() {
        let mut model = DeepSeekModel::random_init(&ModelConfig::v41_tiny(), 1).unwrap();
        let mk = |prompt: Vec<u32>, temp: f32| GenerateRequest {
            prompt,
            params: SamplingParams {
                temperature: temp,
                max_tokens: 2,
                ..Default::default()
            },
        };
        // eos out of vocab.
        assert!(generate(&mut model, vec![mk(vec![1], 0.0)], 256, &Env::default()).is_err());
        // empty prompt.
        assert!(generate(&mut model, vec![mk(vec![], 0.0)], 0, &Env::default()).is_err());
        // out-of-vocab prompt token.
        assert!(generate(
            &mut model,
            vec![mk(vec![1, 999], 0.0)],
            0,
            &Env::default()
        )
        .is_err());
        // invalid params (top_p = 0).
        assert!(generate(
            &mut model,
            vec![GenerateRequest {
                prompt: vec![1],
                params: SamplingParams {
                    top_p: 0.0,
                    max_tokens: 2,
                    ..Default::default()
                },
            }],
            0,
            &Env::default()
        )
        .is_err());
        // Empty request list is fine.
        assert_eq!(
            generate(&mut model, vec![], 0, &Env::default()).unwrap(),
            Vec::<GenerateOutput>::new()
        );
    }

    #[test]
    fn batch_invariant_flag_is_a_documented_noop() {
        // The CPU path is sequential regardless of the flag, so both
        // settings produce identical results.
        let prompts: Vec<Vec<u32>> = vec![vec![4, 5, 6], vec![7, 8]];
        let run = |inv: bool| {
            let mut model = DeepSeekModel::random_init(&ModelConfig::v41_tiny(), 64).unwrap();
            let env = Env {
                batch_invariant: inv,
                ..Env::default()
            };
            generate(
                &mut model,
                prompts
                    .iter()
                    .cloned()
                    .map(|p| GenerateRequest {
                        prompt: p,
                        params: seeded_params(6),
                    })
                    .collect(),
                0,
                &env,
            )
            .unwrap()
        };
        assert_eq!(run(true), run(false));
    }

    #[test]
    fn default_env_runs_mtp_path() {
        // Env::default() has use_nextn_decode = 5 and random-init models
        // always carry MTP weights: the default path IS speculative.
        let mut model = DeepSeekModel::random_init(&ModelConfig::v41_tiny(), 11).unwrap();
        let out = generate(
            &mut model,
            vec![GenerateRequest {
                prompt: vec![3, 1, 4],
                params: SamplingParams {
                    temperature: 0.0,
                    max_tokens: 6,
                    ..Default::default()
                },
            }],
            0,
            &Env::default(),
        )
        .unwrap();
        assert_eq!(out[0].tokens.len(), 6);
        assert_eq!(out[0].finish, "length");
    }

    #[test]
    fn eos_stops_generation() {
        // A model whose logits spike EOS: craft via zero lm_head except
        // the EOS row.
        let mut model = DeepSeekModel::random_init(&ModelConfig::v41_tiny(), 2).unwrap();
        {
            // Zero lm_head: every logit ties at 0 and the greedy argmax
            // tie-break (lowest token id) deterministically picks EOS.
            let vocab = model.config().vocab_size;
            let h = model.config().hidden_size;
            model.weights.lm_head = vec![0.0f32; vocab * h];
        }
        let out = generate(
            &mut model,
            vec![GenerateRequest {
                prompt: vec![1, 2],
                params: SamplingParams {
                    temperature: 0.0,
                    max_tokens: 5,
                    ..Default::default()
                },
            }],
            0,
            &env_nextn(1),
        )
        .unwrap();
        assert_eq!(out[0].tokens, vec![0]);
        assert_eq!(out[0].finish, "stop");
    }
}
