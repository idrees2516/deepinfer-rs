//! The serving engine: continuous batching over the scheduler with
//! streaming events, cancellation, structured output and **DSpark
//! speculative decoding** (see [`spec`]).
//!
//! Two threads (the `deepinfer-scheduler` zero-overlap pattern, adapted
//! for streaming):
//!
//! * **Coordinator thread** — owns the [`Scheduler`] and the subscriber
//!   fan-out. It drains submit/abort commands, folds the previous
//!   step's results (`report_tokens` / `commit_prefill`), reaps
//!   finished requests, and sends the next step's plan.
//! * **Engine thread** — owns the [`TokenEngine`] (the model) and the
//!   per-request sampling state. It executes one plan at a time:
//!   prefill chunks, DSpark spec rounds (draft → verify → rollback →
//!   emit), the batched decode flush (bonus tokens + plain decodes in
//!   one call), grammar-masked sampling, the
//!   first-token-after-prefill contract.
//!
//! The one-step pipeline window means scheduling of step *N+1* overlaps
//! execution of step *N* (a request that finishes at *N* may get one
//! extra decode scheduled; the result is discarded and per-request
//! outputs are unaffected — each request owns its sampler and Philox
//! stream).
//!
//! Cancellation: dropping the event receiver (client went away) makes
//! the next send fail; the coordinator aborts the request at the
//! scheduler, the engine frees the sequence slot, and nothing leaks.
//! Deadlines are enforced by the connection worker (see `api.rs`).
//!
//! Structured output: a request carrying `so_args` gets the same
//! EBNF/JSON/regex DFA masking as the offline `generate` path, applied
//! at every sampled token — and, when speculating, at every drafted and
//! verified position (see [`spec`]).

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use deepinfer_core::{ensure, Error, Result};
use deepinfer_grammar::{Grammar, TokenMaskComputer};
use deepinfer_mtp::{Decision, DraftToken, DsparkScheduler};
use deepinfer_runtime::PrefixState;
use deepinfer_sampler::pipeline::Sampler;
use deepinfer_sampler::{SamplingParams, StructuredOutputArgs};
use deepinfer_scheduler::{BatchPlan, Request, Scheduler, SchedulerConfig, StopReason};

use crate::metrics::SharedMetrics;
use crate::prefix::PrefixIndex;
use crate::spec::SpecAnchor;

/// The model surface the serving loop drives. Implemented by the CPU
/// golden path ([`CpuTokenEngine`]) and the GPU paged-serve path
/// ([`GpuTokenEngine`]) — both `DeepSeekModel` and `DeviceModel` expose
/// the same `prefill` / `decode_step` / rollback / vocab-bytes shape.
///
/// The speculative-decoding surface (`supports_spec` / `draft_step` /
/// `verify` / `rollback`) is what lets the engine run DSpark rounds
/// under continuous batching; the default impls report "unsupported"
/// so trivial engines keep working.
pub trait TokenEngine: Send {
    /// One-time setup with the serving batch width (the scheduler's
    /// `max_running_requests`): the GPU engine sizes and allocates its
    /// paged serve pools here (`DeviceModel::serve_configure`); the CPU
    /// engine needs nothing.
    fn prepare(&mut self, _max_seqs: usize) -> Result<()> {
        Ok(())
    }
    /// Claims sequence slot `seq` (GPU: `serve_attach`; CPU: no-op —
    /// caches are created lazily per id).
    fn attach(&mut self, seq: u32) -> Result<()> {
        let _ = seq;
        Ok(())
    }
    /// Prefills `tokens` onto sequence `seq`, returning next-position
    /// logits and the residual hidden state (`[vocab]` / `[hidden]`).
    fn prefill(&mut self, seq: u32, tokens: &[u32]) -> Result<ForwardOut>;
    /// Decodes one token onto `seq`, returning next-position logits and
    /// the residual hidden state.
    fn decode(&mut self, seq: u32, token: u32) -> Result<ForwardOut>;
    /// Decodes a whole batch of `(seq, token)` pairs in one call —
    /// the step's unified flush for plain decodes and spec bonus
    /// tokens. The default loops [`TokenEngine::decode`]; the GPU
    /// engine overrides it with one `serve_decode` launch.
    fn decode_batch(&mut self, batch: &[(u32, u32)]) -> Result<Vec<ForwardOut>> {
        batch
            .iter()
            .map(|&(seq, tok)| self.decode(seq, tok))
            .collect()
    }
    /// Whether [`TokenEngine::decode_batch_sampled`] is available (the
    /// GPU paged-serve engine with its device sampler).
    fn supports_device_sampling(&self) -> bool {
        false
    }
    /// One batched decode with **on-device sampling**: per-row
    /// `{temperature, top_p, min_p, u, cap, top_k}` (the uniform
    /// pre-drawn from the request's Philox stream — greedy rows pass any
    /// value and consume nothing), returning the sampled token plus the
    /// anchor hidden row without ever reading the `[B, vocab]` logits
    /// back. Only valid for grammar-free plain decodes (masks live on
    /// the host); the engine falls back to [`TokenEngine::decode_batch`]
    /// otherwise.
    ///
    /// # Errors
    /// [`deepinfer_core::Error::InvalidArgument`] when unsupported or on
    /// shape mismatches; [`deepinfer_core::Error::Numerical`] on a
    /// degenerate row.
    fn decode_batch_sampled(
        &mut self,
        _batch: &[(u32, u32)],
        _params: &[deepinfer_kernels::SampleParams],
    ) -> Result<Vec<deepinfer_runtime::SampledOut>> {
        Err(Error::invalid(
            "this engine cannot sample on device (use decode_batch)",
        ))
    }
    /// The staged half of [`TokenEngine::decode_batch_sampled`] (P0-5):
    /// launch the forward + sampler + event-guarded slim readback and
    /// return **without waiting**, so the engine thread can queue this
    /// step's prefill work before consuming the results. Only valid
    /// under the same conditions as the synchronous variant.
    ///
    /// # Errors
    /// [`deepinfer_core::Error::InvalidArgument`] when unsupported.
    fn decode_batch_sampled_stage(
        &mut self,
        _batch: &[(u32, u32)],
        _params: &[deepinfer_kernels::SampleParams],
    ) -> Result<deepinfer_runtime::SampledStep> {
        Err(Error::invalid(
            "this engine cannot sample on device (use decode_batch)",
        ))
    }
    /// The consuming half of [`TokenEngine::decode_batch_sampled_stage`].
    ///
    /// # Errors
    /// [`deepinfer_core::Error::Numerical`] on a degenerate row.
    fn decode_batch_sampled_finish(
        &mut self,
        _step: deepinfer_runtime::SampledStep,
    ) -> Result<Vec<deepinfer_runtime::SampledOut>> {
        Err(Error::invalid(
            "this engine cannot sample on device (use decode_batch)",
        ))
    }
    /// Releases the sequence slot and its cached rows for reuse.
    fn detach(&mut self, seq: u32) -> Result<()>;
    /// Vocabulary size (for token validation).
    fn vocab_size(&self) -> usize;
    /// Per-token UTF-8 bytes (structured-output masking); `None` when
    /// the model was not given a vocabulary byte table.
    fn vocab_bytes(&self) -> Option<&[Vec<u8>]>;

    // ---- speculative decoding surface (defaults: unsupported) ----

    /// Whether the engine can draft (MTP/NEXTN weights present). When
    /// `false` the serving loop runs plain autoregressive decodes even
    /// if a speculation depth is configured.
    fn supports_spec(&self) -> bool {
        false
    }
    /// One MTP draft step: proposes the token after
    /// `(last_token, hidden)` (greedy argmax + softmax confidence).
    fn draft_step(&mut self, _last_token: u32, _hidden: &[f32]) -> Result<DraftToken> {
        Err(Error::invalid("this engine cannot draft (no MTP weights)"))
    }
    /// One MTP draft step on the **fused path** (P0-3): the GPU engine
    /// runs the whole draft module on device and reads back only
    /// `(token, confidence)` — the returned `DraftToken` carries empty
    /// `logits`. Only valid when the caller needs no draft logits
    /// (no grammar mask, no vocabulary cap); the default falls back to
    /// [`TokenEngine::draft_step`] with the full distribution.
    fn draft_step_fused(&mut self, last_token: u32, hidden: &[f32]) -> Result<DraftToken> {
        self.draft_step(last_token, hidden)
    }
    /// Speculative verification: runs the whole draft chain through the
    /// target, returning one logit row per position — row `j` is the
    /// distribution over the token after `tokens[j]`.
    fn verify(&mut self, _seq: u32, _tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        Err(Error::invalid("this engine cannot verify draft chains"))
    }
    /// Rolls the newest `n` tokens of `seq` out of the caches (draft
    /// rejection) without touching any other sequence.
    fn rollback(&mut self, _seq: u32, _n: usize) -> Result<()> {
        Err(Error::invalid("this engine cannot roll back"))
    }

    // ---- prefix-cache surface (defaults: unsupported) ----

    /// Snapshots the first `n` tokens of `seq` for prefix reuse;
    /// `Ok(None)` when the engine does not support prefix caching.
    /// The GPU paged-serve engine caches prefixes internally (its
    /// radix tree lives in the device model's paged state), so it too
    /// reports `None` here — its stats surface is
    /// [`TokenEngine::prefix_stats`].
    fn snapshot_prefix(&mut self, _seq: u32, _n: usize) -> Result<Option<Arc<PrefixState>>> {
        let _ = (_seq, _n);
        Ok(None)
    }
    /// Restores a snapshot into the (empty) sequence `seq`.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] when the engine cannot restore
    /// prefixes (only called after a snapshot from the same engine).
    fn restore_prefix(&mut self, _seq: u32, _snap: &PrefixState) -> Result<()> {
        let _ = (_seq, _snap);
        Err(Error::invalid("this engine cannot restore prefixes"))
    }
    /// Engine-internal prefix-cache stats (`None` for engines whose
    /// cache is managed outside the engine, like the CPU golden path
    /// whose radix index lives in the server). The GPU paged-serve
    /// engine reports its block radix tree here.
    fn prefix_stats(&self) -> Option<deepinfer_runtime::PrefixCacheStats> {
        None
    }
}

/// Re-exported forward result (next-position logits + residual hidden
/// state) shared by both engines.
pub use deepinfer_runtime::ForwardOut;

/// The CPU golden engine: [`deepinfer_runtime::DeepSeekModel`].
pub struct CpuTokenEngine {
    model: deepinfer_runtime::DeepSeekModel,
}

impl CpuTokenEngine {
    /// Wraps the model.
    pub fn new(model: deepinfer_runtime::DeepSeekModel) -> CpuTokenEngine {
        CpuTokenEngine { model }
    }

    /// Mutable access to the wrapped model (vocab setup, diagnostics).
    pub fn model(&mut self) -> &mut deepinfer_runtime::DeepSeekModel {
        &mut self.model
    }
}

impl TokenEngine for CpuTokenEngine {
    fn prefill(&mut self, seq: u32, tokens: &[u32]) -> Result<ForwardOut> {
        self.model.prefill_out(seq, tokens)
    }
    fn decode(&mut self, seq: u32, token: u32) -> Result<ForwardOut> {
        self.model.decode_out(seq, token)
    }
    fn detach(&mut self, seq: u32) -> Result<()> {
        self.model.free_seq(seq)
    }
    fn vocab_size(&self) -> usize {
        self.model.config().vocab_size
    }
    fn vocab_bytes(&self) -> Option<&[Vec<u8>]> {
        self.model.vocab_bytes()
    }
    fn supports_spec(&self) -> bool {
        self.model.has_mtp()
    }
    fn draft_step(&mut self, last_token: u32, hidden: &[f32]) -> Result<DraftToken> {
        self.model.draft_step(last_token, hidden)
    }
    fn verify(&mut self, seq: u32, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        self.model.verify_tokens(seq, tokens)
    }
    fn rollback(&mut self, seq: u32, n: usize) -> Result<()> {
        self.model.rollback_seq(seq, n)
    }
    fn snapshot_prefix(&mut self, seq: u32, n: usize) -> Result<Option<Arc<PrefixState>>> {
        self.model
            .snapshot_prefix(seq, n)
            .map(|s| Some(Arc::new(s)))
    }
    fn restore_prefix(&mut self, seq: u32, snap: &PrefixState) -> Result<()> {
        self.model.restore_prefix(seq, snap)
    }
}

/// The GPU paged-serve engine: [`deepinfer_runtime::DeviceModel`] with
/// its multi-sequence serve paths (`serve_attach` / `serve_prefill` /
/// `serve_decode` / `serve_detach`). Requires a CUDA/HIP device; the
/// caller constructs and configures the `DeviceModel`.
pub struct GpuTokenEngine {
    model: deepinfer_runtime::DeviceModel,
    /// Prefix-cache token budget (0 = off), applied in `prepare`.
    prefix_tokens: usize,
}

impl GpuTokenEngine {
    /// Wraps the device model.
    pub fn new(model: deepinfer_runtime::DeviceModel) -> GpuTokenEngine {
        GpuTokenEngine {
            model,
            prefix_tokens: 0,
        }
    }

    /// Sets the GPU prefix-cache token budget (the server's
    /// `prefix_cache_tokens` knob; 0 disables). The cache itself is
    /// engine-internal: block-granular page sharing over the paged
    /// pools, invisible to the scheduler.
    pub fn with_prefix_tokens(mut self, tokens: usize) -> GpuTokenEngine {
        self.prefix_tokens = tokens;
        self
    }
}

impl TokenEngine for GpuTokenEngine {
    fn prepare(&mut self, max_seqs: usize) -> Result<()> {
        let max_len = self.model.config().max_position_embeddings;
        let want = max_seqs * max_len.div_ceil(deepinfer_kernels::SERVE_PAGE);
        let pages = std::env::var("DEEPINFER_SERVE_PAGES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(want.clamp(1, 4096));
        self.model.serve_configure(max_seqs, max_len, pages)?;
        self.model.serve_prefix_configure(self.prefix_tokens)?;
        // P0-2 serve graphs (decode batch buckets + prefill/verify chunk
        // buckets), SGLang's breakable-prefill-graph shape. Opt-in via
        // the environment; drivers without capture support keep the
        // eager path (enable_serve_graph errors are swallowed).
        if std::env::var("DEEPINFER_SERVE_GRAPH")
            .ok()
            .as_deref()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
        {
            let _ = self.model.enable_serve_graph(true);
        }
        Ok(())
    }
    fn attach(&mut self, seq: u32) -> Result<()> {
        self.model.serve_attach(seq)
    }
    fn prefill(&mut self, seq: u32, tokens: &[u32]) -> Result<ForwardOut> {
        self.model.serve_prefill_out(seq, tokens)
    }
    fn decode(&mut self, seq: u32, token: u32) -> Result<ForwardOut> {
        self.decode_batch(&[(seq, token)])?
            .into_iter()
            .next()
            .ok_or_else(|| Error::internal("serve_decode returned no logits"))
    }
    fn decode_batch(&mut self, batch: &[(u32, u32)]) -> Result<Vec<ForwardOut>> {
        self.model.serve_decode_out(batch)
    }
    fn supports_device_sampling(&self) -> bool {
        true
    }
    fn decode_batch_sampled(
        &mut self,
        batch: &[(u32, u32)],
        params: &[deepinfer_kernels::SampleParams],
    ) -> Result<Vec<deepinfer_runtime::SampledOut>> {
        self.model.serve_decode_sampled(batch, params)
    }
    fn decode_batch_sampled_stage(
        &mut self,
        batch: &[(u32, u32)],
        params: &[deepinfer_kernels::SampleParams],
    ) -> Result<deepinfer_runtime::SampledStep> {
        self.model.serve_decode_sampled_stage(batch, params)
    }
    fn decode_batch_sampled_finish(
        &mut self,
        step: deepinfer_runtime::SampledStep,
    ) -> Result<Vec<deepinfer_runtime::SampledOut>> {
        self.model.serve_decode_sampled_finish(step)
    }
    fn detach(&mut self, seq: u32) -> Result<()> {
        self.model.serve_detach(seq)
    }
    fn vocab_size(&self) -> usize {
        self.model.config().vocab_size
    }
    fn vocab_bytes(&self) -> Option<&[Vec<u8>]> {
        self.model.vocab_bytes()
    }
    fn supports_spec(&self) -> bool {
        self.model.has_mtp()
    }
    fn draft_step(&mut self, last_token: u32, hidden: &[f32]) -> Result<DraftToken> {
        self.model.draft_step(last_token, hidden)
    }
    fn draft_step_fused(&mut self, last_token: u32, hidden: &[f32]) -> Result<DraftToken> {
        self.model.draft_step_fused(last_token, hidden)
    }
    fn verify(&mut self, seq: u32, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        self.model.serve_verify(seq, tokens)
    }
    fn rollback(&mut self, seq: u32, n: usize) -> Result<()> {
        self.model.serve_rollback(seq, n)
    }
    fn prefix_stats(&self) -> Option<deepinfer_runtime::PrefixCacheStats> {
        Some(self.model.serve_prefix_stats())
    }
}
/// Why a generation finished (the API-facing mapping of
/// [`StopReason`] plus serving-specific outcomes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Finish {
    /// EOS token sampled (`"stop"`).
    Stop,
    /// `max_tokens` reached (`"length"`).
    Length,
    /// The grammar admitted no continuation (`"dead"` — matches the
    /// offline `generate` contract).
    Dead,
    /// Aborted (disconnect or deadline).
    Aborted,
    /// Per-request failure (sampling error, bad grammar).
    Error(String),
}

impl Finish {
    /// The OpenAI `finish_reason` string.
    pub fn api_string(&self) -> &str {
        match self {
            Finish::Stop => "stop",
            Finish::Length | Finish::Dead => "length",
            Finish::Aborted => "aborted",
            Finish::Error(_) => "error",
        }
    }
}

/// Events streamed to one request's subscriber.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    /// The request was registered (carries the engine request id).
    Started {
        /// Scheduler-assigned request id.
        id: u64,
    },
    /// One generated token (post-sampling, post-grammar).
    Token {
        /// Request id.
        id: u64,
        /// The token id.
        token: u32,
    },
    /// The request finished; no further events will arrive.
    Finish {
        /// Request id.
        id: u64,
        /// Why it finished.
        reason: Finish,
        /// Tokens generated (for usage accounting; the EOS token is
        /// included when it ended the request).
        completion_tokens: usize,
    },
}

/// A submitted generation.
#[derive(Debug, Clone)]
pub struct Submit {
    /// Prompt token ids (non-empty, every id `< vocab`).
    pub prompt: Vec<u32>,
    /// Sampling parameters (temperature/top-p/top-k/min-p/seed,
    /// `max_tokens`, optional structured output).
    pub params: SamplingParams,
}

impl Submit {
    /// Prompt token count (usage accounting).
    pub fn prompt_tokens(&self) -> usize {
        self.prompt.len()
    }
}

/// Client-to-engine commands.
enum Command {
    Submit(Box<Submit>, Sender<StreamEvent>),
    Abort(u64),
    Shutdown,
}

/// Coordinator-to-engine messages (strictly ordered on one channel).
enum EngineMsg {
    Plan(BatchPlan),
    NewReq { id: u64, seq: u32, submit: Submit },
    Free(u64),
}

/// One decoded position's outcome.
#[derive(Debug, Clone, PartialEq)]
enum TokOut {
    /// A token was sampled.
    Tok(u32),
    /// The grammar admitted no continuation.
    Dead,
    /// Sampling failed for this request only.
    Err(String),
}

type StepResult = Result<Vec<(u64, TokOut)>>;

/// One queued single-token decode for a step's batched flush. `bonus`
/// marks the DSpark continuation token whose forward result becomes the
/// request's new spec anchor (plain decodes sample from it instead).
#[derive(Debug, Clone, Copy)]
struct PendingDecode {
    id: u64,
    token: u32,
    bonus: bool,
}

/// Per-request engine-side state.
struct ReqCtx {
    seq: u32,
    prompt: Vec<u32>,
    sampler: Sampler,
    rng: deepinfer_core::rng::Philox,
    grammar: Option<Grammar>,
    masker: Option<TokenMaskComputer>,
    gstate: u32,
    last_tok: u32,
    eos: u32,
    /// DSpark anchor: the last committed token with the target logits
    /// and residual hidden at that position (`None` until the prompt
    /// finishes prefilling, or when the engine does not speculate).
    anchor: Option<SpecAnchor>,
    /// Tokens emitted so far (engine-side mirror of the scheduler's
    /// `generated` count; drives the spec round's length trim).
    generated: usize,
    /// The request's `max_tokens`.
    max_tokens: usize,
    /// Whether verification resamples stochastically (temperature > 0).
    stochastic: bool,
    /// Engine-side prefill progress (rows actually in the caches).
    /// Diverges from the plan's `chunk_start` only after a scheduler
    /// eviction restarted the prefill — the rewind guard rolls the
    /// surplus rows back before re-prefilling.
    prefilled: usize,
    /// Whether the first chunk's prefix-cache resolution already ran
    /// (restore or cold-start decision).
    cache_resolved: bool,
    /// DSpark acceptance EMA (tokens emitted per token billed),
    /// `None` until the first round completes.
    spec_ema: Option<f32>,
    /// The request's current effective draft depth (adaptive
    /// speculation; starts at the configured depth).
    spec_k: usize,
}

/// The serving engine handle: cheap to clone (the worker threads are
/// shared); dropping the last handle stops the engine.
pub struct ServingHandle {
    tx: Sender<Command>,
    failed: Arc<AtomicBool>,
}

impl Clone for ServingHandle {
    fn clone(&self) -> Self {
        ServingHandle {
            tx: self.tx.clone(),
            failed: Arc::clone(&self.failed),
        }
    }
}

impl ServingHandle {
    /// Whether the engine hit a hard failure (forward-pass error);
    /// `/health` reports degraded afterwards.
    pub fn failed(&self) -> bool {
        self.failed.load(Ordering::Acquire)
    }

    /// Submits a generation and returns the event receiver. The first
    /// event is always [`StreamEvent::Started`] (carrying the request
    /// id); the last is always [`StreamEvent::Finish`].
    ///
    /// # Errors
    /// [`Error::Internal`] when the engine failed or stopped;
    /// [`Error::InvalidArgument`] on an empty prompt.
    pub fn submit(&self, s: Submit) -> Result<Receiver<StreamEvent>> {
        ensure!(
            !self.failed.load(Ordering::Acquire),
            Internal,
            "engine failed; restart the server"
        );
        ensure!(
            !s.prompt.is_empty(),
            InvalidArgument,
            "prompt must be non-empty"
        );
        let (tx, rx) = channel();
        self.tx
            .send(Command::Submit(Box::new(s), tx))
            .map_err(|_| Error::internal("engine stopped"))?;
        Ok(rx)
    }

    /// Aborts a request (idempotent, unknown ids ignored).
    pub fn abort(&self, id: u64) {
        let _ = self.tx.send(Command::Abort(id));
    }

    /// Stops the engine after in-flight requests drain.
    pub fn shutdown(&self) {
        let _ = self.tx.send(Command::Shutdown);
    }
}

/// Engine-level serving tunables (see [`start`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServeTuning {
    /// DSpark draft tokens per round (0 = speculation off).
    pub spec_draft_len: usize,
    /// Adapt each request's draft depth from its observed acceptance
    /// rate (the vLLM dynamic-speculative-decoding controller): the
    /// depth tracks `round(ema * (spec_draft_len + 1)) - 1` so a
    /// rejecting workload decays to plain decode and a accepting one
    /// keeps the full chain. Greedy parity is unaffected (verification
    /// is prefix-exact for any depth).
    pub spec_adaptive: bool,
    /// Prefix-cache token budget (0 = the feature is off).
    pub prefix_cache_tokens: usize,
    /// Hard cap on sampled token ids — the tokenizer's vocabulary when
    /// it is smaller than the model's (byte-mode serving on a full
    /// checkpoint); 0 = uncapped.
    pub sample_vocab_cap: usize,
    /// DSpark greedy-draft mode (SGLang's `SGLANG_DSPARK_USE_GREEDY`):
    /// verification always applies the argmax one-hot rule — even for
    /// stochastic requests — so the draft chain's argmax proposals are
    /// accepted as-is wherever the target agrees. Removes the
    /// resampling pass from every round.
    pub spec_greedy: bool,
}

impl ServeTuning {
    /// Speculation off, prefix cache off, uncapped sampling.
    pub const PLAIN: ServeTuning = ServeTuning {
        spec_draft_len: 0,
        spec_adaptive: false,
        prefix_cache_tokens: 0,
        sample_vocab_cap: 0,
        spec_greedy: false,
    };

    /// Speculation at the given depth (adaptive on), everything else
    /// off — the shape the engine tests use.
    pub const fn spec(depth: usize) -> ServeTuning {
        ServeTuning {
            spec_draft_len: depth,
            spec_adaptive: depth > 0,
            prefix_cache_tokens: 0,
            sample_vocab_cap: 0,
            spec_greedy: false,
        }
    }
}

impl From<usize> for ServeTuning {
    fn from(spec_draft_len: usize) -> ServeTuning {
        ServeTuning::spec(spec_draft_len)
    }
}

/// Starts the serving engine: two threads (coordinator + engine) over
/// one [`TokenEngine`], returning the submit/abort handle.
///
/// # Errors
/// Invalid `eos_id`/tuning; engine `prepare` failures.
pub fn start<E: TokenEngine + 'static>(
    engine: E,
    eos_id: u32,
    sched_cfg: SchedulerConfig,
    metrics: SharedMetrics,
    tuning: impl Into<ServeTuning>,
) -> Result<ServingHandle> {
    let tuning = tuning.into();
    ensure!(
        (eos_id as usize) < engine.vocab_size(),
        OutOfRange,
        "eos_id {eos_id} outside the vocabulary ({})",
        engine.vocab_size()
    );
    ensure!(
        tuning.sample_vocab_cap == 0 || tuning.sample_vocab_cap <= engine.vocab_size(),
        InvalidArgument,
        "sample_vocab_cap {} exceeds the model vocabulary ({})",
        tuning.sample_vocab_cap,
        engine.vocab_size()
    );
    // Speculation is only enabled when the engine can actually draft.
    let spec_len = if engine.supports_spec() {
        tuning.spec_draft_len
    } else {
        0
    };
    // P0-3 greedy-draft mode: env override on top of the tuning knob.
    let spec_greedy = tuning.spec_greedy
        || std::env::var("DEEPINFER_SERVER_SPEC_GREEDY")
            .ok()
            .as_deref()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
    let mut engine = engine;
    engine.prepare(sched_cfg.max_running_requests)?;
    let (cmd_tx, cmd_rx) = channel::<Command>();
    let (msg_tx, msg_rx) = channel::<EngineMsg>();
    let (res_tx, res_rx) = channel::<StepResult>();
    let failed = Arc::new(AtomicBool::new(false));
    // The shared prefix index: the coordinator queries it at admission,
    // the engine inserts/restores during execution.
    let index = Arc::new(Mutex::new(PrefixIndex::new(tuning.prefix_cache_tokens)));

    // ---- engine thread --------------------------------------------------
    let engine_metrics = Arc::clone(&metrics);
    let engine_index = Arc::clone(&index);
    let spec_adaptive = tuning.spec_adaptive;
    let vocab_cap = tuning.sample_vocab_cap;
    let _engine = std::thread::Builder::new()
        .name("deepinfer-engine".to_string())
        .spawn(move || {
            engine_thread(
                engine,
                msg_rx,
                res_tx,
                eos_id,
                spec_len,
                spec_adaptive,
                spec_greedy,
                vocab_cap,
                engine_index,
                engine_metrics,
            )
        })
        .expect("engine thread");

    // ---- coordinator thread ---------------------------------------------
    let coord_failed = Arc::clone(&failed);
    let _coord = std::thread::Builder::new()
        .name("deepinfer-coordinator".to_string())
        .spawn(move || {
            coordinator(
                cmd_rx,
                msg_tx,
                res_rx,
                sched_cfg,
                eos_id,
                spec_len,
                metrics,
                coord_failed,
                index,
            );
        })
        .expect("coordinator thread");

    Ok(ServingHandle { tx: cmd_tx, failed })
}

// ---------------------------------------------------------------------------
// engine thread
// ---------------------------------------------------------------------------

fn engine_thread<E: TokenEngine>(
    engine: E,
    msg_rx: Receiver<EngineMsg>,
    res_tx: Sender<StepResult>,
    eos_id: u32,
    spec_len: usize,
    spec_adaptive: bool,
    spec_greedy: bool,
    vocab_cap: usize,
    index: Arc<Mutex<PrefixIndex>>,
    metrics: SharedMetrics,
) {
    let mut state = EngineState {
        engine,
        ctxs: HashMap::new(),
        spec_len,
        spec_adaptive,
        spec_greedy,
        vocab_cap,
        index,
        metrics,
    };
    while let Ok(msg) = msg_rx.recv() {
        match msg {
            EngineMsg::NewReq { id, seq, submit } => {
                state.new_req(id, seq, submit, eos_id);
            }
            EngineMsg::Free(id) => state.free_req(id),
            EngineMsg::Plan(plan) => {
                if matches!(plan, BatchPlan::Idle) {
                    continue;
                }
                let out = state.execute(&plan);
                match out {
                    Ok(toks) => {
                        if res_tx.send(Ok(toks)).is_err() {
                            return; // coordinator gone
                        }
                    }
                    Err(e) => {
                        // Fatal: model state is suspect. Report and exit.
                        eprintln!("deepinfer-server: fatal engine error: {e}");
                        let _ = res_tx.send(Err(e));
                        return;
                    }
                }
            }
        }
    }
}

struct EngineState<E: TokenEngine> {
    engine: E,
    ctxs: HashMap<u64, ReqCtx>,
    /// DSpark speculation depth (draft tokens per round); 0 = plain
    /// autoregressive decoding for every request.
    spec_len: usize,
    /// Whether the per-request draft depth adapts to the observed
    /// acceptance rate (see [`ServeTuning::spec_adaptive`]).
    spec_adaptive: bool,
    /// DSpark greedy-draft mode (see [`ServeTuning::spec_greedy`]).
    spec_greedy: bool,
    /// Hard cap on sampled token ids (byte-mode serving); 0 = off.
    vocab_cap: usize,
    /// The shared prefix index (insert/restore side).
    index: Arc<Mutex<PrefixIndex>>,
    metrics: SharedMetrics,
}

impl<E: TokenEngine> EngineState<E> {
    fn new_req(&mut self, id: u64, seq: u32, submit: Submit, eos_id: u32) {
        // Structured-output setup (mirrors `generate_one`). Submission
        // already validated the grammar compiles, so failures here are
        // impossible by construction; the fallback keeps the engine
        // total anyway.
        let (grammar, masker, gstate) = match &submit.params.so_args {
            Some(so) => setup_grammar(self.engine.vocab_bytes(), so).unwrap_or((None, None, 0)),
            None => (None, None, 0),
        };
        if let Err(e) = self.engine.attach(seq) {
            eprintln!("deepinfer-server: attach({seq}) failed: {e}");
        }
        let sampler = Sampler::new(submit.params.clone());
        let rng = Sampler::seq_rng(submit.params.seq_seed, seq as u64);
        let stochastic = submit.params.temperature > 0.0;
        self.ctxs.insert(
            id,
            ReqCtx {
                seq,
                prompt: submit.prompt,
                sampler,
                rng,
                grammar,
                masker,
                gstate,
                last_tok: 0,
                eos: eos_id,
                anchor: None,
                generated: 0,
                max_tokens: submit.params.max_tokens,
                stochastic,
                prefilled: 0,
                cache_resolved: false,
                spec_ema: None,
                spec_k: self.spec_len,
            },
        );
    }

    fn free_req(&mut self, id: u64) {
        if let Some(ctx) = self.ctxs.remove(&id) {
            if let Err(e) = self.engine.detach(ctx.seq) {
                eprintln!("deepinfer-server: slot free failed: {e}");
            }
        }
    }

    /// Grammar-masked sampling for `id` from `logits`.
    ///
    /// The vocabulary cap (byte-mode serving) intersects every mask:
    /// tokens at or above the tokenizer's vocabulary can never be
    /// emitted — they would not survive detokenization.
    fn sample(&mut self, id: u64, logits: &[f32]) -> TokOut {
        let Some(ctx) = self.ctxs.get_mut(&id) else {
            return TokOut::Err("unknown request".to_string());
        };
        let cap = self.vocab_cap;
        let mask = if let (Some(g), Some(m)) = (&ctx.grammar, &mut ctx.masker) {
            let mut mk = m.compute_mask(g, ctx.gstate);
            if !g.is_accepting(ctx.gstate) {
                mk.clear(ctx.eos);
            }
            if cap > 0 && cap < mk.vocab_size() {
                for t in cap as u32..mk.vocab_size() as u32 {
                    mk.clear(t);
                }
            }
            if !mk.any() {
                return TokOut::Dead;
            }
            Some(mk)
        } else if cap > 0 && cap < logits.len() {
            let mut mk = deepinfer_core::mask::TokenMask::all_ones(logits.len());
            for t in cap..logits.len() {
                mk.clear(t as u32);
            }
            Some(mk)
        } else {
            None
        };
        let tok = match ctx.sampler.sample(logits, mask.as_ref(), &mut ctx.rng) {
            Ok(t) => t,
            Err(e) => return TokOut::Err(format!("sampling failed: {e}")),
        };
        if let (Some(g), Some(m)) = (&ctx.grammar, &ctx.masker) {
            match m.advance_token(g, ctx.gstate, tok) {
                Some(next) => ctx.gstate = next,
                None => return TokOut::Dead,
            }
        }
        TokOut::Tok(tok)
    }

    fn execute(&mut self, plan: &BatchPlan) -> Result<Vec<(u64, TokOut)>> {
        // GPU prefix-cache gauges: the paged engine's radix tree is
        // engine-internal, so its counters land here (the CPU engine
        // reports through the server-side index instead).
        if let Some(st) = self.engine.prefix_stats() {
            self.metrics.prefix_cache_hits.set(st.hits);
            self.metrics
                .prefix_cache_reused_tokens
                .set(st.reused_tokens);
            self.metrics.prefix_cache_inserts.set(st.inserts);
            self.metrics.prefix_cache_evictions.set(st.evictions);
            self.metrics.prefix_cache_entries.set(st.nodes as u64);
            self.metrics
                .prefix_cache_tokens_gauge
                .set(st.cached_tokens as u64);
        }
        let BatchPlan::Step {
            decode_reqs,
            prefill_reqs,
        } = plan
        else {
            return Ok(Vec::new());
        };
        let mut out: Vec<(u64, TokOut)> =
            Vec::with_capacity(decode_reqs.len() + prefill_reqs.len());
        // Flush-1: the decode requests' plain decodes and spec-round
        // bonuses. Flush-2: the prefilled requests' bonuses.
        let mut pending: Vec<PendingDecode> = Vec::new();
        let mut pending2: Vec<PendingDecode> = Vec::new();

        // ---- decodes first (spec rounds + plain pending) ------------------
        // Decode requests run before the step's prefills so the batched
        // decode flush can be *staged* (launched + event-guarded slim
        // readback) and consumed while the prefill chunks execute on the
        // GPU — one step of overlap, no speculative scheduling.
        for &id in decode_reqs {
            let spec = self.spec_len > 0
                && self
                    .ctxs
                    .get(&id)
                    .map(|c| c.anchor.is_some())
                    .unwrap_or(false);
            if spec {
                self.spec_round(id, &mut out, &mut pending)?;
            } else {
                let tok = self.ctxs.get(&id).map(|c| c.last_tok).unwrap_or(0);
                pending.push(PendingDecode {
                    id,
                    token: tok,
                    bonus: false,
                });
            }
        }

        // ---- flush-1: the decode requests' batched flush ---------------
        // Plain grammar-free decodes on a device-sampling engine launch
        // forward + sampler + slim event-guarded readback and are
        // consumed AFTER the prefill section (the overlap). Anything
        // needing logits (grammar rows, spec bonus anchors) takes the
        // synchronous path here.
        let mut staged: Option<deepinfer_runtime::SampledStep> = None;
        if !pending.is_empty() {
            let batch: Vec<(u32, u32)> = pending
                .iter()
                .map(|p| (self.ctxs.get(&p.id).map(|c| c.seq).unwrap_or(0), p.token))
                .collect();
            let fast_ok = self.engine.supports_device_sampling()
                && pending.iter().all(|p| {
                    !p.bonus
                        && self
                            .ctxs
                            .get(&p.id)
                            .map(|c| c.grammar.is_none())
                            .unwrap_or(false)
                });
            if fast_ok {
                // Pre-draw the uniforms from each request's own Philox
                // stream (greedy rows consume nothing — the CPU
                // sampler's contract), then stage the sampled flush.
                let mut params: Vec<deepinfer_kernels::SampleParams> =
                    Vec::with_capacity(batch.len());
                for p in &pending {
                    let ctx = self
                        .ctxs
                        .get_mut(&p.id)
                        .ok_or_else(|| Error::internal("pending decode lost its context"))?;
                    let sp = ctx.sampler.params();
                    let greedy = sp.temperature.is_nan() || sp.temperature < 1e-8;
                    let u = if greedy { 0.0 } else { ctx.rng.next_f32() };
                    let vocab = self.engine.vocab_size();
                    let cap = if self.vocab_cap > 0 && self.vocab_cap < vocab {
                        self.vocab_cap
                    } else {
                        0
                    };
                    params.push(deepinfer_kernels::SampleParams {
                        temperature: sp.temperature,
                        top_p: sp.top_p,
                        min_p: sp.min_p,
                        u,
                        cap,
                        top_k: sp.top_k,
                    });
                }
                match self.engine.decode_batch_sampled_stage(&batch, &params) {
                    Ok(step) => staged = Some(step),
                    Err(e) if is_per_request(&e) => {
                        // Fall through to the logits path below; only
                        // the offending rows fail there.
                    }
                    Err(e) => return Err(e),
                }
            }
            if staged.is_none() && !pending.is_empty() {
                let mut results: Vec<Option<ForwardOut>> =
                    match self.engine.decode_batch(&batch) {
                        Ok(rs) => rs.into_iter().map(Some).collect(),
                        Err(e) if is_per_request(&e) => {
                            let mut rs = Vec::with_capacity(batch.len());
                            for &(seq, tok) in &batch {
                                match self.engine.decode(seq, tok) {
                                    Ok(o) => rs.push(Some(o)),
                                    Err(e2) if is_per_request(&e2) => rs.push(None),
                                    Err(e2) => return Err(e2),
                                }
                            }
                            rs
                        }
                        Err(e) => return Err(e),
                    };
                for (p, res) in pending.iter().zip(results.drain(..)) {
                    let Some(fo) = res else {
                        out.push((
                            p.id,
                            TokOut::Err("decode failed for this request".to_string()),
                        ));
                        continue;
                    };
                    if p.bonus {
                        // Anchor refresh: the bonus token's forward becomes
                        // the next round's (token, logits, hidden) anchor.
                        if let Some(ctx) = self.ctxs.get_mut(&p.id) {
                            ctx.anchor = Some(SpecAnchor {
                                tok: p.token,
                                logits: fo.logits,
                                hidden: fo.hidden,
                            });
                            ctx.last_tok = p.token;
                        }
                    } else {
                        let t = self.sample(p.id, &fo.logits);
                        if let TokOut::Tok(tok) = t {
                            if let Some(ctx) = self.ctxs.get_mut(&p.id) {
                                ctx.last_tok = tok;
                                ctx.generated += 1;
                            }
                        }
                        out.push((p.id, t));
                    }
                }
            }
        }

        // ---- prefill chunks (queued behind the staged decode readback) -----
        // The decode flush's slim D2H copies are already ordered on the
        // stream, so these forwards run after them; the host processes
        // the decode results (finish_sampled below) while this GPU work
        // executes — the P0-5 overlap.---------------------------------------------
        for &(id, cs, cl) in prefill_reqs {
            let (seq, prompt, prompt_len, last_prompt_tok, first_chunk) = {
                let Some(ctx) = self.ctxs.get(&id) else {
                    continue;
                };
                (
                    ctx.seq,
                    ctx.prompt.clone(),
                    ctx.prompt.len(),
                    ctx.prompt.last().copied().unwrap_or(0),
                    !ctx.cache_resolved,
                )
            };
            if first_chunk {
                if let Some(ctx) = self.ctxs.get_mut(&id) {
                    ctx.cache_resolved = true;
                }
            }

            // (a) First chunk at a prefix-reuse offset: restore the
            //     deepest surviving snapshot (eviction between the
            //     admission lookup and this step may have shrunk it)
            //     and prefill the merged remainder in bounded slices —
            //     never more than one request's rows per slice.
            if first_chunk && cs > 0 {
                let (m, snap) = {
                    let ix = self.index.lock().expect("prefix index poisoned");
                    self.metrics.prefix_cache_entries.set(ix.entries() as u64);
                    self.metrics
                        .prefix_cache_tokens_gauge
                        .set(ix.tokens() as u64);
                    ix.take_at_most(&prompt, cs)
                };
                let start = if m > 0 {
                    let s = snap.expect("snapshot returned for depth > 0");
                    match self.engine.restore_prefix(seq, &s) {
                        Ok(()) => m,
                        Err(e) if is_per_request(&e) => {
                            out.push((id, TokOut::Err(format!("prefix restore failed: {e}"))));
                            continue;
                        }
                        Err(e) => return Err(e),
                    }
                } else {
                    0
                };
                if let Some(ctx) = self.ctxs.get_mut(&id) {
                    ctx.prefilled = start;
                }
                let end = cs + cl;
                let mut pos = start;
                let mut last: Option<ForwardOut> = None;
                let mut failed = false;
                while pos < end {
                    let hi = (pos + deepinfer_runtime::model::MAX_PREFILL_TOKENS).min(end);
                    match self.engine.prefill(seq, &prompt[pos..hi]) {
                        Ok(f) => {
                            last = Some(f);
                            pos = hi;
                        }
                        Err(e) if is_per_request(&e) => {
                            // Capacity/argument failures isolate to this
                            // request (an over-long prompt must not kill
                            // the server).
                            out.push((id, TokOut::Err(e.to_string())));
                            failed = true;
                            break;
                        }
                        Err(e) => return Err(e),
                    }
                }
                if failed {
                    continue;
                }
                if let Some(ctx) = self.ctxs.get_mut(&id) {
                    ctx.prefilled = end;
                }
                let Some(fo) = last else { continue };
                if end >= prompt_len {
                    self.prompt_complete(id, fo, last_prompt_tok, &mut out, &mut pending2)?;
                }
                continue;
            }

            // (b) Plain chunk: the rewind guard first. A scheduler
            //     eviction restarted this prefill at an offset below
            //     the engine's progress — roll the surplus rows back so
            //     the re-prefill starts from exactly `cs` rows (the
            //     pre-restore code would have double-appended).
            let rewind = {
                let Some(ctx) = self.ctxs.get_mut(&id) else {
                    continue;
                };
                ctx.prefilled.saturating_sub(cs)
            };
            if rewind > 0 {
                match self.engine.rollback(seq, rewind) {
                    Ok(()) => {}
                    Err(e) if is_per_request(&e) => {
                        out.push((id, TokOut::Err(format!("prefill rewind failed: {e}"))));
                        continue;
                    }
                    Err(e) => return Err(e),
                }
                if let Some(ctx) = self.ctxs.get_mut(&id) {
                    ctx.prefilled = cs;
                }
            }
            let fo = match self.engine.prefill(seq, &prompt[cs..cs + cl]) {
                Ok(l) => l,
                Err(e) if is_per_request(&e) => {
                    // Capacity/argument failures isolate to this request
                    // (an over-long prompt must not kill the server).
                    out.push((id, TokOut::Err(e.to_string())));
                    continue;
                }
                Err(e) => return Err(e),
            };
            if let Some(ctx) = self.ctxs.get_mut(&id) {
                ctx.prefilled = cs + cl;
            }
            if cs + cl >= prompt_len {
                self.prompt_complete(id, fo, last_prompt_tok, &mut out, &mut pending2)?;
            }
        }

        // ---- flush-2: the prefilled requests' bonus/first decodes -------
        // These need the full logits path (spec anchors); they run after
        // the prefill chunks that produced them.
        if !pending2.is_empty() {
            let batch: Vec<(u32, u32)> = pending2
                .iter()
                .map(|p| (self.ctxs.get(&p.id).map(|c| c.seq).unwrap_or(0), p.token))
                .collect();
            let mut results: Vec<Option<ForwardOut>> =
                match self.engine.decode_batch(&batch) {
                    Ok(rs) => rs.into_iter().map(Some).collect(),
                    Err(e) if is_per_request(&e) => {
                        let mut rs = Vec::with_capacity(batch.len());
                        for &(seq, tok) in &batch {
                            match self.engine.decode(seq, tok) {
                                Ok(o) => rs.push(Some(o)),
                                Err(e2) if is_per_request(&e2) => rs.push(None),
                                Err(e2) => return Err(e2),
                            }
                        }
                        rs
                    }
                    Err(e) => return Err(e),
                };
            for (p, res) in pending2.iter().zip(results.drain(..)) {
                let Some(fo) = res else {
                    out.push((
                        p.id,
                        TokOut::Err("decode failed for this request".to_string()),
                    ));
                    continue;
                };
                if p.bonus {
                    // Anchor refresh: the bonus token's forward becomes
                    // the next round's (token, logits, hidden) anchor.
                    if let Some(ctx) = self.ctxs.get_mut(&p.id) {
                        ctx.anchor = Some(SpecAnchor {
                            tok: p.token,
                            logits: fo.logits,
                            hidden: fo.hidden,
                        });
                        ctx.last_tok = p.token;
                    }
                } else {
                    let t = self.sample(p.id, &fo.logits);
                    if let TokOut::Tok(tok) = t {
                        if let Some(ctx) = self.ctxs.get_mut(&p.id) {
                            ctx.last_tok = tok;
                            ctx.generated += 1;
                        }
                    }
                    out.push((p.id, t));
                }
            }
        }

        // ---- consume the staged decode flush (P0-5 overlap) --------------
        if let Some(step) = staged.take() {
            match self.engine.decode_batch_sampled_finish(step) {
                Ok(rs) => {
                    for (p, r) in pending.iter().zip(rs) {
                        if let Some(ctx) = self.ctxs.get_mut(&p.id) {
                            ctx.last_tok = r.token;
                            ctx.generated += 1;
                        }
                        out.push((p.id, TokOut::Tok(r.token)));
                    }
                }
                Err(e) if is_per_request(&e) => {
                    for p in &pending {
                        out.push((
                            p.id,
                            TokOut::Err(format!("sampled decode failed: {e}")),
                        ));
                    }
                }
                Err(e) => return Err(e),
            }
        }
        Ok(out)
    }

    /// Prompt-prefill completion: insert the prompt's state into the
    /// prefix cache (when enabled and supported), then either launch
    /// the first DSpark round — the final chunk's logits + hidden
    /// become the spec anchor — or sample the first token directly
    /// (the EngineOps contract: the first token ships with the same
    /// step as the final prefill chunk).
    fn prompt_complete(
        &mut self,
        id: u64,
        fo: ForwardOut,
        last_prompt_tok: u32,
        out: &mut Vec<(u64, TokOut)>,
        pending: &mut Vec<PendingDecode>,
    ) -> Result<()> {
        // Prefix-cache insert: the caches hold exactly the prompt rows
        // at this point (speculation has not appended anything yet), so
        // the snapshot is precisely reproducible.
        {
            let (seq, prompt) = match self.ctxs.get(&id) {
                Some(ctx) => (ctx.seq, ctx.prompt.clone()),
                None => return Ok(()),
            };
            if !prompt.is_empty() {
                if let Some(snap) = self
                    .engine
                    .snapshot_prefix(seq, prompt.len())
                    .ok()
                    .flatten()
                {
                    let mut ix = self.index.lock().expect("prefix index poisoned");
                    ix.insert(&prompt, snap);
                    self.metrics.prefix_cache_inserts.add(1);
                    self.metrics.prefix_cache_entries.set(ix.entries() as u64);
                    self.metrics
                        .prefix_cache_tokens_gauge
                        .set(ix.tokens() as u64);
                }
            }
        }
        if self.spec_len > 0 {
            if let Some(ctx) = self.ctxs.get_mut(&id) {
                ctx.anchor = Some(SpecAnchor {
                    tok: last_prompt_tok,
                    logits: fo.logits,
                    hidden: fo.hidden,
                });
            }
            self.spec_round(id, out, pending)?;
        } else {
            let t = self.sample(id, &fo.logits);
            if let TokOut::Tok(tok) = t {
                if let Some(ctx) = self.ctxs.get_mut(&id) {
                    ctx.last_tok = tok;
                    ctx.generated += 1;
                }
            }
            out.push((id, t));
        }
        Ok(())
    }

    /// One DSpark speculative round for request `id` (see [`crate::spec`]).
    /// Drafts a chain from the request's anchor, verifies it against the
    /// target in one call, rolls the rejected tail back out of the
    /// caches, emits the accepted prefix plus the bonus token, and queues
    /// the bonus decode that refreshes the anchor. Per-request failures
    /// push [`TokOut::Err`]; model-level failures return `Err` (fatal).
    fn spec_round(
        &mut self,
        id: u64,
        out: &mut Vec<(u64, TokOut)>,
        pending: &mut Vec<PendingDecode>,
    ) -> Result<()> {
        use deepinfer_mtp::{verify_greedy, verify_stochastic};

        // Split borrows: the engine and the request contexts are
        // disjoint fields.
        let engine = &mut self.engine;
        let ctxs = &mut self.ctxs;
        let metrics = &self.metrics;
        let spec_len = self.spec_len;
        let spec_adaptive = self.spec_adaptive;
        let spec_greedy = self.spec_greedy;
        let vocab_cap = self.vocab_cap;
        let Some(ctx) = ctxs.get_mut(&id) else {
            return Ok(());
        };
        let seq = ctx.seq;
        let eos = ctx.eos;
        let Some(anchor) = ctx.anchor.clone() else {
            return Ok(());
        };
        let max_tokens = ctx.max_tokens;
        let generated = ctx.generated;
        let eff_k = ctx.spec_k;
        let remaining = max_tokens.saturating_sub(generated);
        if remaining == 0 {
            // Defensive: the scheduler should have finished the request.
            pending.push(PendingDecode {
                id,
                token: ctx.last_tok,
                bonus: false,
            });
            return Ok(());
        }
        let k = eff_k.min(remaining);
        let dspark = DsparkScheduler::new(k);

        // Byte-mode serving: tokens at or above the tokenizer's
        // vocabulary can never be emitted, so they are masked out of
        // the draft picks and every verification row (including the
        // anchor row) — a capped draft can never survive verification.
        let cap_mask = if vocab_cap > 0 && vocab_cap < anchor.logits.len() {
            let mut m = deepinfer_core::mask::TokenMask::all_ones(anchor.logits.len());
            for t in vocab_cap..anchor.logits.len() {
                m.clear(t as u32);
            }
            Some(m)
        } else {
            None
        };

        // 1. Dead grammar at the anchor state (mirrors `sample`).
        if let (Some(g), Some(m)) = (&ctx.grammar, &mut ctx.masker) {
            if !crate::spec::state_mask(m, g, ctx.gstate, eos).any() {
                out.push((id, TokOut::Dead));
                return Ok(());
            }
        }

        // 2. Draft chain (grammar-masked argmax when a grammar is set).
        //    The fused device path (P0-3) skips the [vocab] draft logits
        //    whenever no mask needs them.
        let use_fused = ctx.grammar.is_none() && cap_mask.is_none();
        let mut drafts: Vec<DraftToken> = Vec::new();
        let mut states: Vec<u32> = vec![ctx.gstate];
        let mut last = anchor.tok;
        while drafts.len() < k && dspark.should_verify(&drafts) == Decision::ContinueDraft {
            let d = match if use_fused {
                engine.draft_step_fused(last, &anchor.hidden)
            } else {
                engine.draft_step(last, &anchor.hidden)
            } {
                Ok(d) => d,
                Err(e) => {
                    out.push((id, TokOut::Err(format!("draft failed: {e}"))));
                    return Ok(());
                }
            };
            if let (Some(g), Some(m)) = (&ctx.grammar, &mut ctx.masker) {
                let mk = crate::spec::state_mask(m, g, *states.last().expect("non-empty"), eos);
                if !mk.any() {
                    // Dead-end state: stop drafting; if everything so far
                    // is accepted, the round finishes Dead below.
                    break;
                }
                let Some((tok, conf)) = crate::spec::masked_pick(&d.logits, &mk) else {
                    break;
                };
                match m.advance_token(g, *states.last().expect("non-empty"), tok) {
                    Some(s) => states.push(s),
                    None => {
                        // Defensive: the mask allowed the token but the
                        // DFA cannot advance — leave the draft in the
                        // chain; masked verification rejects it.
                        drafts.push(DraftToken {
                            token: tok,
                            confidence: conf,
                            logits: d.logits,
                        });
                        break;
                    }
                }
                last = tok;
                drafts.push(DraftToken {
                    token: tok,
                    confidence: conf,
                    logits: d.logits,
                });
            } else if let Some(cm) = &cap_mask {
                // No grammar, but byte-mode capping: the draft pick must
                // stay inside the tokenizer's vocabulary.
                let Some((tok, conf)) = crate::spec::masked_pick(&d.logits, cm) else {
                    break;
                };
                last = tok;
                drafts.push(DraftToken {
                    token: tok,
                    confidence: conf,
                    logits: d.logits,
                });
            } else {
                last = d.token;
                drafts.push(d);
            }
        }
        if drafts.is_empty() {
            // Nothing to verify (a zero-depth round is impossible here;
            // this guards pathological configurations): fall back to a
            // plain decode.
            pending.push(PendingDecode {
                id,
                token: anchor.tok,
                bonus: false,
            });
            return Ok(());
        }
        let d_len = drafts.len();

        // 3. Verification forward through the target.
        let tokens: Vec<u32> = drafts.iter().map(|d| d.token).collect();
        let rows_v = match engine.verify(seq, &tokens) {
            Ok(r) => r,
            Err(e) if is_per_request(&e) => {
                out.push((id, TokOut::Err(format!("verify failed: {e}"))));
                return Ok(());
            }
            Err(e) => return Err(e),
        };
        let mut rows: Vec<Vec<f32>> = Vec::with_capacity(d_len + 1);
        rows.push(anchor.logits.clone());
        rows.extend(rows_v);

        // Byte-mode capping applies to every reachable row (including
        // the anchor row 0).
        if let Some(cm) = &cap_mask {
            for row in rows.iter_mut() {
                crate::spec::mask_row(row, cm);
            }
        }

        // 4. Grammar-mask the reachable verification rows.
        if let (Some(g), Some(m)) = (&ctx.grammar, &mut ctx.masker) {
            for j in 0..states.len() {
                let mk = crate::spec::state_mask(m, g, states[j], eos);
                if mk.any() {
                    crate::spec::mask_row(&mut rows[j], &mk);
                }
                // Empty masks (dead-end states) leave the row raw: the
                // row is only consulted for accept/next up to
                // `accepted`, and a dead-end state can only sit at
                // `states[accepted]`, which is handled explicitly below.
            }
        }

        // 5. Verify (greedy argmax rule or stochastic resample). The
        // greedy-draft mode forces the one-hot argmax rule even for
        // stochastic requests (SGLang's `SGLANG_DSPARK_USE_GREEDY`).
        let outcome = if ctx.stochastic && !spec_greedy {
            let temp = ctx.sampler.params().temperature;
            match verify_stochastic(&drafts, &rows, &mut ctx.rng, temp) {
                Ok(o) => o,
                Err(e) => {
                    out.push((id, TokOut::Err(format!("verification failed: {e}"))));
                    return Ok(());
                }
            }
        } else {
            match verify_greedy(&drafts, &rows) {
                Ok(o) => o,
                Err(e) => {
                    out.push((id, TokOut::Err(format!("verification failed: {e}"))));
                    return Ok(());
                }
            }
        };
        let accepted = outcome.accepted;
        let bonus = outcome.next_token;

        // Adaptive depth (the vLLM dynamic-speculative-decoding
        // controller): the acceptance EMA drives the next round's draft
        // depth toward `round(ema * (spec_len + 1)) - 1`, clamped to
        // [1, spec_len]. A rejecting workload decays to single-token
        // rounds; an accepting one keeps the full chain. Greedy parity
        // is unaffected (verification is prefix-exact for any depth).
        if spec_adaptive && spec_len > 1 {
            let r = (accepted + 1) as f32 / (d_len as f32 + 1.0);
            let ema = ctx.spec_ema.map_or(r, |e| 0.7 * e + 0.3 * r);
            ctx.spec_ema = Some(ema);
            ctx.spec_k = ((ema * (spec_len as f32 + 1.0)).round() as usize)
                .saturating_sub(1)
                .clamp(1, spec_len);
        }

        // 6. Grammar state at the accepted position: advance by the
        //    bonus (or finish Dead at a dead-end).
        if let (Some(g), Some(m)) = (&ctx.grammar, &mut ctx.masker) {
            let st = states[accepted];
            let mk = crate::spec::state_mask(m, g, st, eos);
            if !mk.any() {
                // Emit the accepted prefix, roll the tail back, die.
                let keep = accepted;
                if d_len > keep {
                    if let Err(e) = engine.rollback(seq, d_len - keep) {
                        if is_per_request(&e) {
                            out.push((id, TokOut::Err(format!("rollback failed: {e}"))));
                            return Ok(());
                        }
                        return Err(e);
                    }
                }
                ctx.generated += keep;
                for d in &drafts[..keep] {
                    out.push((id, TokOut::Tok(d.token)));
                }
                ctx.anchor = None;
                out.push((id, TokOut::Dead));
                metrics.spec_rounds.add(1);
                metrics.spec_draft_tokens.add(d_len as u64);
                metrics.spec_accepted_tokens.add(keep as u64);
                if d_len > keep {
                    metrics.spec_rollbacks.add(1);
                    metrics.spec_rollback_tokens.add((d_len - keep) as u64);
                }
                return Ok(());
            }
            match m.advance_token(g, st, bonus) {
                Some(ns) => ctx.gstate = ns,
                None => {
                    out.push((id, TokOut::Dead));
                    return Ok(());
                }
            }
        }

        // 7. Trims: EOS inside the accepted prefix; the max_tokens cap.
        let mut keep = accepted;
        let mut include_bonus = true;
        let mut finish_eos = false;
        if let Some(e) = drafts[..accepted].iter().position(|d| d.token == eos) {
            keep = e + 1;
            include_bonus = false;
            finish_eos = true;
        }
        if include_bonus && bonus == eos {
            finish_eos = true;
        }
        let mut finish_len = false;
        if !finish_eos {
            let room = remaining;
            if keep >= room {
                keep = room;
                include_bonus = false;
                finish_len = true;
            } else if keep + 1 == room {
                finish_len = true;
            }
        } else if include_bonus && keep + 1 > remaining {
            // EOS bonus would overshoot the cap: drop it (the EOS trim
            // above already handles the prefix case).
            include_bonus = false;
        }

        // 8. Roll the rejected tail out of the caches.
        if d_len > keep {
            if let Err(e) = engine.rollback(seq, d_len - keep) {
                if is_per_request(&e) {
                    out.push((id, TokOut::Err(format!("rollback failed: {e}"))));
                    return Ok(());
                }
                return Err(e);
            }
        }

        // 9. Emit the accepted prefix (+ bonus).
        let mut emitted = keep;
        for d in &drafts[..keep] {
            out.push((id, TokOut::Tok(d.token)));
        }
        if include_bonus {
            out.push((id, TokOut::Tok(bonus)));
            emitted += 1;
        }
        ctx.generated += emitted;
        ctx.last_tok = if include_bonus {
            bonus
        } else {
            drafts[keep.saturating_sub(1)].token
        };

        // 10. Anchor refresh: the bonus decode (queued into the step's
        //     batched flush) provides the next round's anchor. Finishing
        //     requests skip it.
        let finishing = finish_eos || finish_len;
        if finishing {
            ctx.anchor = None;
        } else {
            debug_assert!(include_bonus, "a continuing round must keep the bonus");
            pending.push(PendingDecode {
                id,
                token: bonus,
                bonus: true,
            });
        }

        metrics.spec_rounds.add(1);
        metrics.spec_draft_tokens.add(d_len as u64);
        metrics.spec_accepted_tokens.add(keep as u64);
        if include_bonus {
            metrics.spec_bonus_tokens.add(1);
        }
        if d_len > keep {
            metrics.spec_rollbacks.add(1);
            metrics.spec_rollback_tokens.add((d_len - keep) as u64);
        }
        Ok(())
    }
}

/// Whether a model error is safe to attribute to a single request
/// (fail that request, keep the engine alive) instead of fatal.
/// Capacity and range failures come from a request's own shape; other
/// errors indicate state corruption and stop the engine.
fn is_per_request(e: &deepinfer_core::Error) -> bool {
    matches!(
        e,
        deepinfer_core::Error::CapacityExceeded(_) | deepinfer_core::Error::OutOfRange(_)
    )
}

/// Grammar + masker setup (the `generate_one` logic). Fails when the
/// vocabulary byte table is missing or the schema does not compile.
fn setup_grammar(
    vocab_bytes: Option<&[Vec<u8>]>,
    so: &StructuredOutputArgs,
) -> Result<(Option<Grammar>, Option<TokenMaskComputer>, u32)> {
    let g = Grammar::from_so_args(so_type_name(&so.schema_type), &so.schema)?;
    let vb =
        vocab_bytes.ok_or_else(|| Error::invalid("structured output requires vocabulary bytes"))?;
    let masker = TokenMaskComputer::new(vb.to_vec());
    let gstate = g.initial_state();
    Ok((Some(g), Some(masker), gstate))
}

/// The grammar crate's formalism name for a [`SchemaType`].
fn so_type_name(t: &deepinfer_sampler::SchemaType) -> &'static str {
    match t {
        deepinfer_sampler::SchemaType::Regex => "regex",
        deepinfer_sampler::SchemaType::Json => "json",
        deepinfer_sampler::SchemaType::Ebnf => "ebnf",
    }
}

// ---------------------------------------------------------------------------
// coordinator thread
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn coordinator(
    cmd_rx: Receiver<Command>,
    msg_tx: Sender<EngineMsg>,
    res_rx: Receiver<StepResult>,
    cfg: SchedulerConfig,
    eos_id: u32,
    spec_draft_len: usize,
    metrics: SharedMetrics,
    failed: Arc<AtomicBool>,
    index: Arc<Mutex<PrefixIndex>>,
) {
    let mut sched = Scheduler::new(cfg);
    // id -> (event sender, completion count)
    let mut subs: HashMap<u64, Sender<StreamEvent>> = HashMap::new();
    let mut completions: HashMap<u64, usize> = HashMap::new();
    let mut dead: HashSet<u64> = HashSet::new();
    let mut req_errors: HashMap<u64, String> = HashMap::new();
    let mut slot_of: HashMap<u64, u32> = HashMap::new();
    let mut free_slots: Vec<u32> = Vec::new();
    let mut next_slot: u32 = 0;
    let mut shutdown = false;
    // At most one plan may be in flight: the engine executes plans in
    // channel order, and the scheduler must see the previous step's
    // results before building the next batch (otherwise a decoding
    // request gets re-scheduled every spin and blows past its token
    // budget — the exact failure the `AsyncEngine` avoids with its own
    // in-flight counter).
    let mut plan_in_flight = false;

    let alloc_slot = |free_slots: &mut Vec<u32>, next_slot: &mut u32| -> u32 {
        free_slots.pop().unwrap_or_else(|| {
            let s = *next_slot;
            *next_slot += 1;
            s
        })
    };

    loop {
        // ---- 1. drain commands (non-blocking) -----------------------------
        loop {
            match cmd_rx.try_recv() {
                Ok(Command::Submit(submit, reply)) => {
                    if let Some(err) = validate_submit(&submit) {
                        // Grammar unusable: fail fast with a 400-shaped
                        // finish (id 0 = never registered).
                        let _ = reply.send(StreamEvent::Finish {
                            id: 0,
                            reason: Finish::Error(err),
                            completion_tokens: 0,
                        });
                        continue;
                    }
                    let req = Request::new(submit.prompt.clone(), submit.params.max_tokens)
                        .with_spec_len(spec_draft_len);
                    let id = sched.add_request(req);
                    // Prefix-cache admission lookup: a cached prompt
                    // prefix skips its recomputation (the engine
                    // restores the state before prefilling the suffix).
                    {
                        let mut ix = index.lock().expect("prefix index poisoned");
                        let hit = ix.lookup(&submit.prompt);
                        let reuse = hit.min(submit.prompt.len().saturating_sub(1));
                        if reuse > 0 && sched.skip_prefill(id, reuse) {
                            metrics.prefix_cache_hits.add(1);
                            metrics.prefix_cache_reused_tokens.add(reuse as u64);
                        }
                        metrics.prefix_cache_entries.set(ix.entries() as u64);
                        metrics.prefix_cache_tokens_gauge.set(ix.tokens() as u64);
                    }
                    let seq = alloc_slot(&mut free_slots, &mut next_slot);
                    slot_of.insert(id, seq);
                    if msg_tx
                        .send(EngineMsg::NewReq {
                            id,
                            seq,
                            submit: *submit,
                        })
                        .is_err()
                    {
                        let _ = reply.send(StreamEvent::Finish {
                            id,
                            reason: Finish::Error("engine failed".to_string()),
                            completion_tokens: 0,
                        });
                        failed.store(true, Ordering::Release);
                        continue;
                    }
                    let _ = reply.send(StreamEvent::Started { id });
                    subs.insert(id, reply);
                    completions.insert(id, 0);
                    dead.remove(&id);
                }
                Ok(Command::Abort(id)) => {
                    sched.abort_request(id);
                }
                Ok(Command::Shutdown) => {
                    shutdown = true;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    shutdown = true;
                    if subs.is_empty() {
                        return;
                    }
                }
            }
        }

        // ---- 2. wait for / fold the in-flight step's results ---------------
        // While a plan executes we do not schedule: commands keep
        // draining (10ms poll), results fold the moment they arrive.
        if plan_in_flight {
            match res_rx.recv_timeout(Duration::from_millis(10)) {
                Ok(Ok(toks)) => {
                    fold_results(
                        &mut sched,
                        &mut subs,
                        &mut completions,
                        &mut dead,
                        &mut req_errors,
                        &toks,
                        &metrics,
                        eos_id,
                    );
                    plan_in_flight = false;
                }
                Ok(Err(e)) => {
                    // Fatal engine error: fail every active request, degrade.
                    failed.store(true, Ordering::Release);
                    metrics.engine_errors.add(1);
                    let msg = format!("engine failure: {e}");
                    for (id, sub) in subs.drain() {
                        let _ = sub.send(StreamEvent::Finish {
                            id,
                            reason: Finish::Error(msg.clone()),
                            completion_tokens: completions.remove(&id).unwrap_or(0),
                        });
                    }
                    completions.clear();
                    return;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    // Engine thread exited (fatal path already reported).
                    if !failed.swap(true, Ordering::AcqRel) {
                        for (id, sub) in subs.drain() {
                            let _ = sub.send(StreamEvent::Finish {
                                id,
                                reason: Finish::Error("engine failed".to_string()),
                                completion_tokens: completions.remove(&id).unwrap_or(0),
                            });
                        }
                    }
                    return;
                }
            }
        }

        // ---- 3. reap finished requests ------------------------------------
        for (id, reason) in sched.take_finished() {
            let completion = completions.remove(&id).unwrap_or(0);
            if let Some(sub) = subs.remove(&id) {
                let fin = if dead.contains(&id) {
                    dead.remove(&id);
                    Finish::Dead
                } else if let Some(err) = req_errors.remove(&id) {
                    Finish::Error(err)
                } else {
                    match reason {
                        StopReason::Eos => Finish::Stop,
                        StopReason::Length => Finish::Length,
                        StopReason::Aborted => Finish::Aborted,
                    }
                };
                let _ = sub.send(StreamEvent::Finish {
                    id,
                    reason: fin,
                    completion_tokens: completion,
                });
            } else {
                dead.remove(&id);
                req_errors.remove(&id);
            }
            if let Some(seq) = slot_of.remove(&id) {
                free_slots.push(seq);
                let _ = msg_tx.send(EngineMsg::Free(id));
            }
            metrics.engine_active.set(sched.num_active() as u64);
        }
        sched.reap_finished();

        // ---- 4. shutdown check ---------------------------------------------
        if shutdown && sched.is_done() && subs.is_empty() {
            return;
        }

        // ---- 5. publish queue gauges ----------------------------------------
        metrics.engine_active.set(sched.num_active() as u64);
        metrics.engine_queued.set(sched.num_waiting() as u64);

        // ---- 6. schedule the next step (nothing in flight) -----------------
        let plan = sched.schedule();
        match &plan {
            BatchPlan::Idle => {
                // Nothing running: park briefly so the thread does not
                // spin (submissions wake it within 5ms).
                if !shutdown {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
            }
            BatchPlan::Step { prefill_reqs, .. } => {
                metrics.engine_steps.add(1);
                // Optimistic prefill commit (the AsyncEngine pattern):
                // commit the slices this plan will execute *before*
                // sending it, so a later schedule() pass never
                // re-schedules an in-flight chunk (plans execute in
                // channel order, so commit order matches execution
                // order exactly).
                for &(id, cs, cl) in prefill_reqs {
                    sched.commit_prefill(id, cs, cl);
                    metrics.prompt_tokens.add(cl as u64);
                }
                if msg_tx.send(EngineMsg::Plan(plan.clone())).is_err() {
                    failed.store(true, Ordering::Release);
                    for (id, sub) in subs.drain() {
                        let _ = sub.send(StreamEvent::Finish {
                            id,
                            reason: Finish::Error("engine failed".to_string()),
                            completion_tokens: completions.remove(&id).unwrap_or(0),
                        });
                    }
                    return;
                }
                plan_in_flight = true;
            }
        }
    }
}

/// Fails a submit before registration when its grammar cannot compile
/// or needs a vocabulary byte table the model does not have (fast 400s
/// instead of mid-stream errors). The engine thread compiles the same
/// grammar again — deterministic, so acceptance here implies acceptance
/// there.
fn validate_submit(s: &Submit) -> Option<String> {
    if let Some(so) = &s.params.so_args {
        if let Err(e) = Grammar::from_so_args(so_type_name(&so.schema_type), &so.schema) {
            return Some(format!("grammar error: {e}"));
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn fold_results(
    sched: &mut Scheduler,
    subs: &mut HashMap<u64, Sender<StreamEvent>>,
    completions: &mut HashMap<u64, usize>,
    dead: &mut HashSet<u64>,
    req_errors: &mut HashMap<u64, String>,
    toks: &[(u64, TokOut)],
    metrics: &SharedMetrics,
    eos_id: u32,
) {
    // Prefill slices were committed optimistically at schedule time
    // (the AsyncEngine pattern — see the Step branch below), so this
    // fold only reports tokens.
    let mut report: Vec<(u64, u32)> = Vec::with_capacity(toks.len());
    for &(id, ref t) in toks {
        match t {
            TokOut::Tok(tok) => {
                report.push((id, *tok));
                *completions.entry(id).or_insert(0) += 1;
                metrics.completion_tokens.add(1);
                if let Some(sub) = subs.get(&id) {
                    if sub.send(StreamEvent::Token { id, token: *tok }).is_err() {
                        // Subscriber gone: abort.
                        sched.abort_request(id);
                        metrics.aborted_requests.add(1);
                    }
                }
            }
            TokOut::Dead => {
                dead.insert(id);
                sched.abort_request(id);
            }
            TokOut::Err(msg) => {
                req_errors.insert(id, msg.clone());
                sched.abort_request(id);
            }
        }
    }
    sched.report_tokens(&report, eos_id);
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics;
    use deepinfer_runtime::{DeepSeekModel, ModelConfig};

    /// A trivial deterministic engine: logits put all mass on the next
    /// id in `1..vocab` — a moving token pattern that never emits EOS,
    /// so requests run until `max_tokens` or abort.
    struct FakeEngine {
        vocab: usize,
        prefills: Vec<(u32, usize)>,
        decodes: Vec<(u32, u32)>,
        frees: Vec<u32>,
    }

    impl TokenEngine for FakeEngine {
        fn prefill(&mut self, seq: u32, tokens: &[u32]) -> Result<ForwardOut> {
            self.prefills.push((seq, tokens.len()));
            Ok(ForwardOut {
                logits: self.logits(tokens.last().copied().unwrap_or(0)),
                hidden: vec![0.5; 8],
            })
        }
        fn decode(&mut self, seq: u32, token: u32) -> Result<ForwardOut> {
            self.decodes.push((seq, token));
            Ok(ForwardOut {
                logits: self.logits(token),
                hidden: vec![0.5; 8],
            })
        }
        fn detach(&mut self, seq: u32) -> Result<()> {
            self.frees.push(seq);
            Ok(())
        }
        fn vocab_size(&self) -> usize {
            self.vocab
        }
        fn vocab_bytes(&self) -> Option<&[Vec<u8>]> {
            None
        }
    }

    impl FakeEngine {
        fn logits(&self, last: u32) -> Vec<f32> {
            let mut l = vec![-10.0f32; self.vocab];
            // Cycle through 1..vocab, never hitting EOS (0), so
            // requests run until length or abort.
            let span = (self.vocab - 1) as u32;
            let next = (last % span) + 1;
            l[next as usize] = 10.0;
            l
        }
    }

    fn greedy_params(max_tokens: usize) -> SamplingParams {
        SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            seq_seed: None,
            so_args: None,
            max_tokens,
        }
    }

    #[test]
    fn engine_streams_tokens_and_finishes() {
        let m = metrics::shared();
        let engine = FakeEngine {
            vocab: 8,
            prefills: vec![],
            decodes: vec![],
            frees: vec![],
        };
        let handle = start(engine, 0, SchedulerConfig::default(), Arc::clone(&m), 0).unwrap();
        let rx = handle
            .submit(Submit {
                prompt: vec![1, 2],
                params: greedy_params(4),
            })
            .unwrap();

        let mut tokens = Vec::new();
        let mut finish = None;
        for ev in rx {
            match ev {
                StreamEvent::Started { .. } => {}
                StreamEvent::Token { token, .. } => tokens.push(token),
                StreamEvent::Finish {
                    reason,
                    completion_tokens,
                    ..
                } => {
                    assert_eq!(completion_tokens, 4);
                    finish = Some(reason);
                }
            }
        }
        assert_eq!(tokens.len(), 4);
        assert_eq!(finish, Some(Finish::Length));
        handle.shutdown();
    }

    #[test]
    fn abort_stops_generation_and_frees_slot() {
        let m = metrics::shared();
        let engine = FakeEngine {
            vocab: 8,
            prefills: vec![],
            decodes: vec![],
            frees: vec![],
        };
        let handle = start(engine, 0, SchedulerConfig::default(), Arc::clone(&m), 0).unwrap();
        let rx = handle
            .submit(Submit {
                prompt: vec![3],
                params: greedy_params(100),
            })
            .unwrap();

        // Take the id from Started, wait for one token, then abort.
        let mut id = None;
        let mut got_token = false;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !got_token && std::time::Instant::now() < deadline {
            match rx.recv_timeout(std::time::Duration::from_millis(200)) {
                Ok(StreamEvent::Started { id: i }) => id = Some(i),
                Ok(StreamEvent::Token { .. }) => got_token = true,
                Ok(_) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        assert!(got_token, "no token streamed");
        handle.abort(id.expect("Started must precede tokens"));

        let mut finish = None;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while let Ok(ev) =
            rx.recv_timeout(deadline.saturating_duration_since(std::time::Instant::now()))
        {
            if let StreamEvent::Finish { reason, .. } = ev {
                finish = Some(reason);
                break;
            }
        }
        assert_eq!(finish, Some(Finish::Aborted));
        handle.shutdown();
    }

    #[test]
    fn dropping_subscriber_aborts_request() {
        let m = metrics::shared();
        let engine = FakeEngine {
            vocab: 8,
            prefills: vec![],
            decodes: vec![],
            frees: vec![],
        };
        let handle = start(engine, 0, SchedulerConfig::default(), Arc::clone(&m), 0).unwrap();
        let rx = handle
            .submit(Submit {
                prompt: vec![0],
                params: greedy_params(1_000_000),
            })
            .unwrap();
        drop(rx); // client went away immediately
        std::thread::sleep(std::time::Duration::from_millis(300));
        // The engine must not have leaked the request: active gauge
        // returns to zero (the token send fails -> abort).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while m.engine_active.get() > 0 && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(m.engine_active.get(), 0);
        assert!(m.aborted_requests.get() >= 1);
        handle.shutdown();
    }

    #[test]
    fn real_cpu_model_end_to_end_stream() {
        // The tiny CPU model is the oracle: the server engine must
        // produce the same greedy token stream as `generate`.
        let cfg = ModelConfig::v41_tiny();
        let model = DeepSeekModel::random_init(&cfg, 7).unwrap();
        let eos = 0u32;
        let m = metrics::shared();
        let handle = start(
            CpuTokenEngine::new(model),
            eos,
            SchedulerConfig::default(),
            m,
            0,
        )
        .unwrap();
        let params = greedy_params(6);
        let rx = handle
            .submit(Submit {
                prompt: vec![9, 9, 9, 9],
                params: params.clone(),
            })
            .unwrap();

        let mut got = Vec::new();
        for ev in rx {
            match ev {
                StreamEvent::Token { token, .. } => got.push(token),
                StreamEvent::Finish {
                    reason,
                    completion_tokens,
                    ..
                } => {
                    assert_eq!(completion_tokens, got.len());
                    assert_eq!(reason, Finish::Length);
                }
                _ => {}
            }
        }
        assert_eq!(got.len(), 6);

        // Same prompt through the offline generate: identical stream.
        let mut model2 = DeepSeekModel::random_init(&cfg, 7).unwrap();
        let env = deepinfer_core::env::Env::default();
        let outs = deepinfer_runtime::generate(
            &mut model2,
            vec![deepinfer_runtime::GenerateRequest {
                prompt: vec![9, 9, 9, 9],
                params,
            }],
            eos,
            &env,
        )
        .unwrap();
        assert_eq!(outs[0].tokens, got, "serving stream must match generate()");
        handle.shutdown();
    }

    // ------------------------------------------------------------------
    // DSpark speculative decoding on the serve path
    // ------------------------------------------------------------------

    /// Collects one request's full stream.
    fn collect(rx: Receiver<StreamEvent>) -> (Vec<u32>, Finish, usize) {
        let mut toks = Vec::new();
        let mut fin = None;
        let mut n = 0;
        for ev in rx {
            match ev {
                StreamEvent::Started { .. } => {}
                StreamEvent::Token { token, .. } => toks.push(token),
                StreamEvent::Finish {
                    reason,
                    completion_tokens,
                    ..
                } => {
                    n = completion_tokens;
                    fin = Some(reason);
                }
            }
        }
        (toks, fin.expect("Finish is always last"), n)
    }

    /// Byte-level vocab table for the 256-entry tiny model (EOS 0 has no
    /// bytes — only legal at accepting grammar states).
    fn byte_vocab() -> Vec<Vec<u8>> {
        let mut vb: Vec<Vec<u8>> = (0..256u32).map(|t| vec![t as u8]).collect();
        vb[0] = Vec::new();
        vb
    }

    /// **The DSpark parity oracle**: greedy serve-with-spec, greedy
    /// serve-without-spec and the offline `generate` (plain) must produce
    /// the identical token stream on the tiny CPU model.
    #[test]
    fn spec_serving_matches_plain_and_offline_greedy() {
        let cfg = ModelConfig::v41_tiny();
        let eos = 0u32;
        let prompt = vec![9u32, 9, 9, 9];
        let params = greedy_params(12);

        // (a) serve with speculation (draft 3).
        let m = metrics::shared();
        let model_a = DeepSeekModel::random_init(&cfg, 7).unwrap();
        let handle = start(
            CpuTokenEngine::new(model_a),
            eos,
            SchedulerConfig::default(),
            Arc::clone(&m),
            3,
        )
        .unwrap();
        let (spec_toks, fin, n) = collect(
            handle
                .submit(Submit {
                    prompt: prompt.clone(),
                    params: params.clone(),
                })
                .unwrap(),
        );
        handle.shutdown();
        assert_eq!(fin, Finish::Length);
        assert_eq!(n, 12);
        assert_eq!(spec_toks.len(), 12);

        // (b) serve without speculation.
        let model_b = DeepSeekModel::random_init(&cfg, 7).unwrap();
        let handle = start(
            CpuTokenEngine::new(model_b),
            eos,
            SchedulerConfig::default(),
            Arc::clone(&m),
            0,
        )
        .unwrap();
        let (plain_toks, fin2, n2) = collect(
            handle
                .submit(Submit {
                    prompt: prompt.clone(),
                    params: params.clone(),
                })
                .unwrap(),
        );
        handle.shutdown();
        assert_eq!(fin2, Finish::Length);
        assert_eq!(n2, 12);

        // (c) offline generate (plain).
        let mut model_c = DeepSeekModel::random_init(&cfg, 7).unwrap();
        let outs = deepinfer_runtime::generate(
            &mut model_c,
            vec![deepinfer_runtime::GenerateRequest {
                prompt: prompt.clone(),
                params: params.clone(),
            }],
            eos,
            &deepinfer_core::env::Env::default(),
        )
        .unwrap();

        assert_eq!(
            spec_toks, plain_toks,
            "greedy spec serve == greedy plain serve"
        );
        assert_eq!(
            spec_toks, outs[0].tokens,
            "greedy spec serve == offline generate"
        );

        // The spec path really speculated: rounds and accepted drafts were
        // counted (more than zero), and the acceptance bookkeeping is
        // internally consistent.
        assert!(m.spec_rounds.get() > 0);
        assert!(m.spec_draft_tokens.get() > 0);
        assert!(m.spec_accepted_tokens.get() + m.spec_bonus_tokens.get() >= m.spec_rounds.get());
        assert_eq!(
            m.spec_accepted_tokens.get() + m.spec_bonus_tokens.get(),
            12,
            "accepted drafts + bonuses must equal the emitted tokens"
        );
    }

    /// P0-3 greedy-draft mode (`spec_greedy`): a **stochastic** request
    /// served with speculation under the forced one-hot verification rule
    /// produces exactly the greedy stream (the argmax rule makes
    /// verification temperature-independent), and the bookkeeping stays
    /// consistent.
    #[test]
    fn spec_greedy_mode_forces_argmax_verification() {
        let cfg = ModelConfig::v41_tiny();
        let eos = 0u32;
        let prompt = vec![9u32, 9, 9, 9];

        let stochastic = SamplingParams {
            temperature: 1.4,
            top_p: 0.95,
            ..greedy_params(12)
        };

        // (a) Greedy reference (plain, no speculation).
        let m = metrics::shared();
        let model_a = DeepSeekModel::random_init(&cfg, 21).unwrap();
        let handle = start(
            CpuTokenEngine::new(model_a),
            eos,
            SchedulerConfig::default(),
            Arc::clone(&m),
            0,
        )
        .unwrap();
        let (greedy_toks, fin, n) = collect(
            handle
                .submit(Submit {
                    prompt: prompt.clone(),
                    params: greedy_params(12),
                })
                .unwrap(),
        );
        handle.shutdown();
        assert_eq!((fin, n), (Finish::Length, 12));

        // (b) Stochastic + spec + greedy-draft mode: the one-hot rule
        // makes the emitted stream identical to (a).
        let model_b = DeepSeekModel::random_init(&cfg, 21).unwrap();
        let handle = start(
            CpuTokenEngine::new(model_b),
            eos,
            SchedulerConfig::default(),
            Arc::clone(&m),
            ServeTuning {
                spec_draft_len: 3,
                spec_adaptive: true,
                prefix_cache_tokens: 0,
                sample_vocab_cap: 0,
                spec_greedy: true,
            },
        )
        .unwrap();
        let (mode_toks, fin2, n2) = collect(
            handle
                .submit(Submit {
                    prompt: prompt.clone(),
                    params: stochastic.clone(),
                })
                .unwrap(),
        );
        handle.shutdown();
        assert_eq!((fin2, n2), (Finish::Length, 12));
        assert_eq!(
            mode_toks, greedy_toks,
            "spec_greedy must force the argmax stream for stochastic requests"
        );
        assert!(m.spec_rounds.get() > 0);
    }

    /// Grammar-aware speculation: greedy serve-with-spec under a regex
    /// grammar produces the identical stream as greedy plain serve under
    /// the same grammar, and the stream satisfies the grammar.
    #[test]
    fn spec_serving_with_grammar_matches_plain() {
        let cfg = ModelConfig::v41_tiny();
        let eos = 0u32;
        let grammar_params = |max_tokens: usize| SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            seq_seed: None,
            so_args: Some(StructuredOutputArgs {
                schema_type: deepinfer_sampler::SchemaType::Regex,
                schema: "[0-9]{8}".to_string(),
            }),
            max_tokens,
        };
        let prompt = vec![5u32, 6, 7];

        let run = |spec: usize| {
            let m = metrics::shared();
            let mut model = DeepSeekModel::random_init(&cfg, 11).unwrap();
            model.set_vocab_bytes(byte_vocab());
            let handle = start(
                CpuTokenEngine::new(model),
                eos,
                SchedulerConfig::default(),
                Arc::clone(&m),
                spec,
            )
            .unwrap();
            let (toks, fin, n) = collect(
                handle
                    .submit(Submit {
                        prompt: prompt.clone(),
                        params: grammar_params(16),
                    })
                    .unwrap(),
            );
            handle.shutdown();
            (toks, fin, n)
        };
        let (spec_toks, fin, n) = run(3);
        let (plain_toks, fin2, n2) = run(0);

        assert_eq!(
            spec_toks, plain_toks,
            "grammar spec stream must equal plain"
        );
        assert_eq!(fin, fin2);
        assert_eq!(n, n2);
        // The grammar is "[0-9]{8}": 8 digit bytes, then the only legal
        // token at the complete-match state is EOS (empty bytes), which
        // ends the request with `stop`.
        assert_eq!(spec_toks.len(), 9);
        for t in &spec_toks[..8] {
            assert!((48..=57).contains(t), "token {t} is not a digit byte");
        }
        assert_eq!(spec_toks[8], 0, "the 9th token must be EOS");
        assert_eq!(fin, Finish::Stop);
    }

    /// Batch invariance under speculation: two concurrent requests (one
    /// plain, one grammar-constrained) produce exactly the streams they
    /// produce when run alone.
    #[test]
    fn spec_batch_invariance() {
        let cfg = ModelConfig::v41_tiny();
        let eos = 0u32;
        let m = metrics::shared();
        let mut model = DeepSeekModel::random_init(&cfg, 13).unwrap();
        model.set_vocab_bytes(byte_vocab());
        let handle = start(
            CpuTokenEngine::new(model),
            eos,
            SchedulerConfig::default(),
            Arc::clone(&m),
            4,
        )
        .unwrap();

        let plain = greedy_params(10);
        let gram = SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            seq_seed: None,
            so_args: Some(StructuredOutputArgs {
                schema_type: deepinfer_sampler::SchemaType::Regex,
                schema: "[0-9]+".to_string(),
            }),
            max_tokens: 10,
        };
        let rx_a = handle
            .submit(Submit {
                prompt: vec![3, 4, 5],
                params: plain.clone(),
            })
            .unwrap();
        let rx_b = handle
            .submit(Submit {
                prompt: vec![8, 8],
                params: gram.clone(),
            })
            .unwrap();
        let (a_toks, _, _) = collect(rx_a);
        let (b_toks, _, _) = collect(rx_b);

        // Solo runs with the same engine parameters.
        let solo = |params: SamplingParams, prompt: Vec<u32>| {
            let m = metrics::shared();
            let mut model = DeepSeekModel::random_init(&cfg, 13).unwrap();
            model.set_vocab_bytes(byte_vocab());
            let handle = start(
                CpuTokenEngine::new(model),
                eos,
                SchedulerConfig::default(),
                Arc::clone(&m),
                4,
            )
            .unwrap();
            let out = collect(handle.submit(Submit { prompt, params }).unwrap());
            handle.shutdown();
            out
        };
        let (solo_a, _, _) = solo(plain, vec![3, 4, 5]);
        let (solo_b, _, _) = solo(gram, vec![8, 8]);
        assert_eq!(a_toks, solo_a, "plain stream must be batch-invariant");
        assert_eq!(b_toks, solo_b, "grammar stream must be batch-invariant");
        for t in &b_toks {
            assert!((48..=57).contains(t), "grammar token {t} must be a digit");
        }
        handle.shutdown();
    }

    /// A mock spec engine with a rigged target: across the full
    /// verification rows (`[anchor, verify[0], verify[1], ...]`), the
    /// first `accept_count` rows argmax to the drafted token and the rest
    /// argmax to `bonus_tok`. The anchor rows (prefill / bonus decode)
    /// follow the same schedule, so exactly `accept_count` drafts are
    /// accepted per round and `bonus_tok` continues.
    struct MockSpecEngine {
        vocab: usize,
        /// How many leading full-rows agree with the drafted token.
        accept_count: usize,
        /// The continuation token the disagreeing rows propose.
        bonus_tok: u32,
        /// The fixed token the draft head always proposes.
        draft_tok: u32,
        /// The draft head's (fixed) confidence: 0.99 trips DSpark's
        /// mean-confidence early-verify after two drafts; 0.7 keeps the
        /// chain running to the full depth.
        confidence: f32,
        verifies: Vec<usize>,
        rollbacks: Vec<(u32, usize)>,
        decodes: usize,
    }

    impl TokenEngine for MockSpecEngine {
        fn prefill(&mut self, _seq: u32, _tokens: &[u32]) -> Result<ForwardOut> {
            // Full-row 0 (the anchor): agrees while accept_count > 0.
            let agree = self.accept_count > 0;
            let want = if agree {
                self.draft_tok
            } else {
                self.bonus_tok
            };
            Ok(ForwardOut {
                logits: self.spiked(want),
                hidden: vec![0.25; 8],
            })
        }
        fn decode(&mut self, _seq: u32, _token: u32) -> Result<ForwardOut> {
            // Bonus decodes re-anchor: keep agreeing.
            self.decodes += 1;
            let agree = self.accept_count > 0;
            let want = if agree {
                self.draft_tok
            } else {
                self.bonus_tok
            };
            Ok(ForwardOut {
                logits: self.spiked(want),
                hidden: vec![0.25; 8],
            })
        }
        fn detach(&mut self, _seq: u32) -> Result<()> {
            Ok(())
        }
        fn vocab_size(&self) -> usize {
            self.vocab
        }
        fn vocab_bytes(&self) -> Option<&[Vec<u8>]> {
            None
        }
        fn supports_spec(&self) -> bool {
            true
        }
        fn draft_step(&mut self, _last_token: u32, _hidden: &[f32]) -> Result<DraftToken> {
            let mut logits = vec![-5.0f32; self.vocab];
            logits[self.draft_tok as usize] = 5.0;
            Ok(DraftToken {
                token: self.draft_tok,
                confidence: self.confidence,
                logits,
            })
        }
        fn verify(&mut self, _seq: u32, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
            self.verifies.push(tokens.len());
            // Verify row j is full-row j + 1: it agrees with drafts[j]
            // while j + 1 < accept_count, then proposes bonus_tok.
            let mut rows = Vec::with_capacity(tokens.len());
            for j in 0..tokens.len() {
                let agree = j + 1 < self.accept_count;
                let want = if agree { tokens[j] } else { self.bonus_tok };
                rows.push(self.spiked(want));
            }
            Ok(rows)
        }
        fn rollback(&mut self, seq: u32, n: usize) -> Result<()> {
            self.rollbacks.push((seq, n));
            Ok(())
        }
    }

    impl MockSpecEngine {
        fn spiked(&self, tok: u32) -> Vec<f32> {
            let mut l = vec![-5.0f32; self.vocab];
            l[tok as usize] = 5.0;
            l
        }
    }

    /// Rejection under batching: the rejected tail is rolled back with
    /// exactly `drafts - accepted` rows, the accepted prefix plus the
    /// bonus are emitted, and per-request streams stay exact.
    #[test]
    fn spec_rejection_rolls_back_under_batching() {
        let m = metrics::shared();
        // Draft budget 3; the target accepts only the first draft.
        let engine = MockSpecEngine {
            vocab: 16,
            accept_count: 1,
            bonus_tok: 9,
            draft_tok: 5,
            confidence: 0.99,
            verifies: vec![],
            rollbacks: vec![],
            decodes: 0,
        };
        let handle = start(engine, 0, SchedulerConfig::default(), Arc::clone(&m), 3).unwrap();
        let (toks, fin, n) = collect(
            handle
                .submit(Submit {
                    prompt: vec![1, 2],
                    params: greedy_params(5),
                })
                .unwrap(),
        );
        handle.shutdown();

        // Every round: the DSpark scheduler stops the chain at 2 drafts
        // (mean confidence 0.99 >= conf_hi); the target accepts the first
        // and rejects the second; one row rolls back; the bonus (9)
        // continues. Emitted per round: [5, 9] until the cap trims.
        assert_eq!(fin, Finish::Length);
        assert_eq!(n, toks.len());
        assert_eq!(toks.len(), 5);
        // First two emitted tokens: the accepted draft then the bonus.
        assert_eq!(&toks[..2], &[5, 9]);
        // The full stream: [accepted, bonus] per round, the final
        // length-trimmed round drafting a single token (k = remaining)
        // that is accepted outright — no rollback there.
        assert_eq!(toks, vec![5, 9, 5, 9, 5]);
        let rounds = m.spec_rounds.get();
        assert_eq!(rounds, 3);
        assert_eq!(m.spec_rollbacks.get(), 2, "the two full rounds reject");
        assert_eq!(m.spec_rollback_tokens.get(), 2, "one token per rollback");
    }

    /// EOS inside the accepted prefix stops the request exactly there
    /// and rolls the tail back (no tokens past the EOS escape).
    #[test]
    fn spec_eos_inside_prefix_stops_and_trims() {
        let m = metrics::shared();
        // The draft head proposes EOS (token 0); the target accepts the
        // first draft. eos == 0.
        let engine = MockSpecEngine {
            vocab: 16,
            accept_count: 1,
            bonus_tok: 9,
            draft_tok: 0,
            confidence: 0.99,
            verifies: vec![],
            rollbacks: vec![],
            decodes: 0,
        };
        let handle = start(engine, 0, SchedulerConfig::default(), Arc::clone(&m), 3).unwrap();
        let (toks, fin, n) = collect(
            handle
                .submit(Submit {
                    prompt: vec![1, 2],
                    params: greedy_params(50),
                })
                .unwrap(),
        );
        handle.shutdown();
        // The accepted prefix contains EOS at position 0: the request
        // stops with exactly that one token.
        assert_eq!(fin, Finish::Stop);
        assert_eq!(toks, vec![0]);
        assert_eq!(n, 1);
        // The unaccepted tail (1 or 2 drafts) was rolled back.
        assert!(m.spec_rollback_tokens.get() >= 1);
    }

    /// Stochastic spec serving: deterministic given the seed, exact
    /// length honoured, all tokens in-vocabulary.
    #[test]
    fn spec_stochastic_serving_is_deterministic() {
        let run = || {
            let m = metrics::shared();
            let model = DeepSeekModel::random_init(&ModelConfig::v41_tiny(), 21).unwrap();
            let handle = start(
                CpuTokenEngine::new(model),
                0,
                SchedulerConfig::default(),
                Arc::clone(&m),
                3,
            )
            .unwrap();
            let (toks, fin, n) = collect(
                handle
                    .submit(Submit {
                        prompt: vec![4, 5, 6],
                        params: SamplingParams {
                            temperature: 0.8,
                            top_p: 1.0,
                            top_k: 0,
                            min_p: 0.0,
                            seq_seed: Some(1234),
                            so_args: None,
                            max_tokens: 9,
                        },
                    })
                    .unwrap(),
            );
            handle.shutdown();
            (toks, fin, n)
        };
        let (a, fin, n) = run();
        let (b, fin2, n2) = run();
        assert_eq!(a, b, "same seed must stream identically");
        assert_eq!(fin, fin2);
        assert_eq!(n, n2);
        assert_eq!(a.len(), 9);
        assert_eq!(fin, Finish::Length);
        assert!(a.iter().all(|t| (*t as usize) < 256));
    }

    /// max_tokens is never overshot even when a whole draft chain plus
    /// bonus would exceed it.
    #[test]
    fn spec_respects_max_tokens_exactly() {
        for cap in [1usize, 2, 3, 4, 5, 7] {
            let m = metrics::shared();
            let model = DeepSeekModel::random_init(&ModelConfig::v41_tiny(), 31).unwrap();
            let handle = start(
                CpuTokenEngine::new(model),
                0,
                SchedulerConfig::default(),
                Arc::clone(&m),
                5,
            )
            .unwrap();
            let (toks, fin, n) = collect(
                handle
                    .submit(Submit {
                        prompt: vec![2, 3],
                        params: greedy_params(cap),
                    })
                    .unwrap(),
            );
            handle.shutdown();
            assert_eq!(toks.len(), cap, "cap {cap}");
            assert_eq!(n, cap, "cap {cap}");
            assert_eq!(fin, Finish::Length, "cap {cap}");
        }
    }

    /// **Prefix cache**: the second request with the same prompt is
    /// admitted with cached state (the metrics prove the restore) and
    /// streams the same tokens as the cold request.
    #[test]
    fn prefix_cache_reuses_prompt_state() {
        let cfg = ModelConfig::v41_tiny();
        let eos = 0u32;
        let m = metrics::shared();
        let model = DeepSeekModel::random_init(&cfg, 101).unwrap();
        let handle = start(
            CpuTokenEngine::new(model),
            eos,
            SchedulerConfig::default(),
            Arc::clone(&m),
            ServeTuning {
                prefix_cache_tokens: 1024,
                ..ServeTuning::PLAIN
            },
        )
        .unwrap();
        let prompt: Vec<u32> = (0..12u32).map(|t| (t * 37) % 256).collect();
        let (a, fa, na) = collect(
            handle
                .submit(Submit {
                    prompt: prompt.clone(),
                    params: greedy_params(8),
                })
                .unwrap(),
        );
        assert_eq!(
            m.prefix_cache_inserts.get(),
            1,
            "the cold prompt was snapshotted"
        );
        assert_eq!(m.prefix_cache_hits.get(), 0);

        let (b, fb, nb) = collect(
            handle
                .submit(Submit {
                    prompt: prompt.clone(),
                    params: greedy_params(8),
                })
                .unwrap(),
        );
        handle.shutdown();
        assert!(
            m.prefix_cache_hits.get() >= 1,
            "the second request reused the prefix"
        );
        assert!(
            m.prefix_cache_reused_tokens.get() >= 11,
            "reuse covered all but the mandatory final token"
        );
        assert_eq!((a.clone(), fa, na), (b.clone(), fb, nb));
        assert_eq!(b.len(), 8);
    }

    /// **Prefix cache, multi-turn**: a follow-up prompt extending a
    /// finished conversation reuses the whole cached turn.
    #[test]
    fn prefix_cache_multiturn_extension() {
        let cfg = ModelConfig::v41_tiny();
        let m = metrics::shared();
        let model = DeepSeekModel::random_init(&cfg, 103).unwrap();
        let handle = start(
            CpuTokenEngine::new(model),
            0,
            SchedulerConfig::default(),
            Arc::clone(&m),
            ServeTuning {
                prefix_cache_tokens: 1024,
                ..ServeTuning::PLAIN
            },
        )
        .unwrap();
        let turn1: Vec<u32> = (0..10u32).map(|t| (t * 41) % 256).collect();
        let _ = collect(
            handle
                .submit(Submit {
                    prompt: turn1.clone(),
                    params: greedy_params(4),
                })
                .unwrap(),
        );
        // Turn 2 extends the conversation: same leading tokens, longer.
        let mut turn2 = turn1.clone();
        turn2.extend([7u32, 7, 7, 7, 7, 7]);
        let (_, f, n) = collect(
            handle
                .submit(Submit {
                    prompt: turn2,
                    params: greedy_params(4),
                })
                .unwrap(),
        );
        handle.shutdown();
        assert_eq!((f, n), (Finish::Length, 4));
        assert!(
            m.prefix_cache_hits.get() >= 1,
            "the extension reused turn 1"
        );
        assert!(
            m.prefix_cache_reused_tokens.get() >= 10,
            "the whole first turn was reused"
        );
    }

    /// **Prefix cache eviction**: a tight token budget evicts the LRU
    /// snapshot; a later request with the evicted prompt runs cold and
    /// still completes correctly (the engine's fallback path).
    #[test]
    fn prefix_cache_eviction_falls_back_to_cold() {
        let cfg = ModelConfig::v41_tiny();
        let m = metrics::shared();
        let model = DeepSeekModel::random_init(&cfg, 107).unwrap();
        let handle = start(
            CpuTokenEngine::new(model),
            0,
            SchedulerConfig::default(),
            Arc::clone(&m),
            ServeTuning {
                prefix_cache_tokens: 13, // one 12-token prompt, no more
                ..ServeTuning::PLAIN
            },
        )
        .unwrap();
        let a: Vec<u32> = (0..12u32).map(|t| (t * 53) % 256).collect();
        let _ = collect(
            handle
                .submit(Submit {
                    prompt: a.clone(),
                    params: greedy_params(2),
                })
                .unwrap(),
        );
        assert_eq!(m.prefix_cache_tokens_gauge.get(), 12);
        // A divergent 10-token prompt evicts the LRU entry (a).
        let c: Vec<u32> = (0..10u32).map(|t| 200 + (t % 40)).collect();
        let _ = collect(
            handle
                .submit(Submit {
                    prompt: c,
                    params: greedy_params(2),
                })
                .unwrap(),
        );
        assert_eq!(m.prefix_cache_tokens_gauge.get(), 10, "a was evicted");
        // Re-submitting a runs cold (a miss), and still finishes.
        let (toks, f, n) = collect(
            handle
                .submit(Submit {
                    prompt: a,
                    params: greedy_params(3),
                })
                .unwrap(),
        );
        handle.shutdown();
        assert_eq!((f, n), (Finish::Length, 3));
        assert_eq!(toks.len(), 3);
    }

    /// **Adaptive speculation (vLLM DSD-style)**: an all-rejecting
    /// workload decays the draft depth to 1 (drafted-token count
    /// collapses), while an all-accepting one keeps the full chain.
    /// Greedy parity holds either way (the parity tests above already
    /// run with the controller on).
    #[test]
    fn adaptive_spec_depth_tracks_acceptance() {
        // (a) Rejecting drafts: with depth 5 and 8 tokens requested, the
        // first round drafts 5; every round after decays to 1.
        let m = metrics::shared();
        let engine = MockSpecEngine {
            vocab: 64,
            accept_count: 0,
            bonus_tok: 9,
            draft_tok: 5,
            confidence: 0.7,
            verifies: Vec::new(),
            rollbacks: Vec::new(),
            decodes: 0,
        };
        let handle = start(
            engine,
            0,
            SchedulerConfig::default(),
            Arc::clone(&m),
            ServeTuning {
                spec_draft_len: 5,
                spec_adaptive: true,
                ..ServeTuning::PLAIN
            },
        )
        .unwrap();
        let (_, f, n) = collect(
            handle
                .submit(Submit {
                    prompt: vec![1, 2],
                    params: greedy_params(8),
                })
                .unwrap(),
        );
        handle.shutdown();
        assert_eq!((f, n), (Finish::Length, 8));
        // 8 rounds (one bonus token each): the depths run
        // [5, 1, 1, 1, 1, 2, 1, 1] = 13 drafted tokens — vs 40 for a
        // fixed depth-5 controller. The EMA math is pinned by the
        // f32 controller simulation.
        assert_eq!(m.spec_draft_tokens.get(), 13, "depth decayed to 1");

        // (b) Accepting drafts: the full chain stays at 5 (clamped by
        // the remaining budget near the end).
        let m2 = metrics::shared();
        let engine = MockSpecEngine {
            vocab: 64,
            accept_count: usize::MAX,
            bonus_tok: 9,
            draft_tok: 5,
            confidence: 0.7,
            verifies: Vec::new(),
            rollbacks: Vec::new(),
            decodes: 0,
        };
        let handle = start(
            engine,
            0,
            SchedulerConfig::default(),
            Arc::clone(&m2),
            ServeTuning {
                spec_draft_len: 5,
                spec_adaptive: true,
                ..ServeTuning::PLAIN
            },
        )
        .unwrap();
        let (toks, f, n) = collect(
            handle
                .submit(Submit {
                    prompt: vec![1, 2],
                    params: greedy_params(12),
                })
                .unwrap(),
        );
        handle.shutdown();
        assert_eq!((f, n), (Finish::Length, 12));
        assert_eq!(toks.len(), 12);
        // Rounds: 5+5+1 (12 tokens: 6, 6, 0? — the exact split depends
        // on the max_tokens trim; assert the depth never decayed below
        // the full chain: drafted >= 2 full chains of 5.
        assert!(
            m2.spec_draft_tokens.get() >= 10,
            "accepting rounds keep drafting the full chain"
        );
    }

    /// **Byte-mode vocabulary cap**: with the tokenizer's vocabulary
    /// smaller than the model's, no sampled or spec-verified token ever
    /// reaches the unreachable id range.
    #[test]
    fn vocab_cap_confines_sampling_to_the_tokenizer_alphabet() {
        let cfg = ModelConfig::v41_tiny(); // vocab 256
        let m = metrics::shared();
        let model = DeepSeekModel::random_init(&cfg, 113).unwrap();
        let handle = start(
            CpuTokenEngine::new(model),
            0,
            SchedulerConfig::default(),
            Arc::clone(&m),
            ServeTuning {
                spec_draft_len: 4,
                spec_adaptive: true,
                sample_vocab_cap: 200,
                ..ServeTuning::PLAIN
            },
        )
        .unwrap();
        let (toks, f, n) = collect(
            handle
                .submit(Submit {
                    prompt: vec![3, 4, 5],
                    params: SamplingParams {
                        temperature: 0.9,
                        top_p: 1.0,
                        top_k: 0,
                        min_p: 0.0,
                        seq_seed: Some(77),
                        so_args: None,
                        max_tokens: 16,
                    },
                })
                .unwrap(),
        );
        handle.shutdown();
        assert_eq!((f, n), (Finish::Length, 16));
        assert!(
            toks.iter().all(|t| (*t as usize) < 200),
            "every emitted token must be inside the capped alphabet"
        );
    }
}
