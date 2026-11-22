//! The HTTP API layer: OpenAI-compatible routes on top of the serving
//! engine, with SSE streaming, stop sequences, cancellation, auth and
//! request limits.
//!
//! Routes:
//!
//! | Method | Path | Purpose |
//! |---|---|---|
//! | GET | `/health` | liveness + engine status |
//! | GET | `/v1/models` | model list (the served model) |
//! | POST | `/v1/chat/completions` | chat API (template + stream) |
//! | POST | `/v1/completions` | legacy completions API |
//! | GET | `/metrics` | Prometheus text |
//! | POST | `/admin/shutdown` | graceful drain (auth required) |
//!
//! Error bodies follow the OpenAI shape
//! `{"error":{"message","type","param","code"}}`.
//!
//! Cancellation: a streaming response whose write fails aborts the
//! generation; the request deadline (server config) bounds every
//! request. Stop strings are matched against the detokenized text and
//! truncate the completion (OpenAI semantics: the stop text is not
//! part of the completion).

use std::io::Write;
use std::net::TcpStream;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use deepinfer_core::json::Json;
use deepinfer_core::{ensure, Error, Result};
use deepinfer_sampler::{SamplingParams, SchemaType, StructuredOutputArgs};

use crate::chat::{render_chat_template, ChatMessage, DeltaDecoder};
use crate::config::ServerConfig;
use crate::engine::{Finish, ServingHandle, StreamEvent, Submit};
use crate::http::{ChunkedWriter, HttpRequest, Response};
use crate::metrics::{unix_now, SharedMetrics};

/// Per-connection context shared by all workers.
pub struct ApiCtx {
    /// Server configuration.
    pub cfg: ServerConfig,
    /// The serving engine.
    pub engine: ServingHandle,
    /// Metrics.
    pub metrics: SharedMetrics,
    /// Engine kind label (`cpu` / `gpu`).
    pub engine_kind: String,
    /// Tokenizer used to encode prompts / decode tokens.
    pub tokenizer: Box<dyn deepinfer_core::tokenizer::Tokenizer>,
    /// EOS token id.
    pub eos_id: u32,
    /// Per-token byte table (fast detokenization).
    pub vocab_bytes: Vec<Vec<u8>>,
    next_id: std::sync::atomic::AtomicU64,
}

impl ApiCtx {
    /// Builds the context.
    pub fn new(
        cfg: ServerConfig,
        engine: ServingHandle,
        metrics: SharedMetrics,
        engine_kind: &str,
        tokenizer: Box<dyn deepinfer_core::tokenizer::Tokenizer>,
        eos_id: u32,
        vocab_bytes: Vec<Vec<u8>>,
    ) -> ApiCtx {
        ApiCtx {
            cfg,
            engine,
            metrics,
            engine_kind: engine_kind.to_string(),
            tokenizer,
            eos_id,
            vocab_bytes,
            next_id: std::sync::atomic::AtomicU64::new(1),
        }
    }

    fn next_completion_id(&self, prefix: &str) -> String {
        let n = self.next_id.fetch_add(1, Ordering::Relaxed);
        format!("{prefix}-{n:012}")
    }

    fn token_bytes(&self, tok: u32) -> &[u8] {
        self.vocab_bytes
            .get(tok as usize)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
}

/// An OpenAI-style error body.
fn error_body(message: &str, kind: &str, code: &str) -> String {
    Json::Obj(vec![(
        "error".to_string(),
        Json::Obj(vec![
            ("message".to_string(), Json::Str(message.to_string())),
            ("type".to_string(), Json::Str(kind.to_string())),
            ("param".to_string(), Json::Null),
            ("code".to_string(), Json::Str(code.to_string())),
        ]),
    )])
    .to_json_string()
}

fn bad_request(message: &str, code: &str) -> Response {
    Response::json(400, error_body(message, "invalid_request_error", code))
}

fn unauthorized(message: &str) -> Response {
    Response::json(
        401,
        error_body(message, "invalid_request_error", "invalid_api_key"),
    )
}

fn unavailable(message: &str) -> Response {
    Response::json(
        503,
        error_body(message, "server_error", "service_unavailable"),
    )
}

fn internal_error(message: &str) -> Response {
    Response::json(500, error_body(message, "server_error", "internal_error"))
}

/// Adds the CORS header to a response.
fn cors(ctx: &ApiCtx, r: Response) -> Response {
    r.header("Access-Control-Allow-Origin", &ctx.cfg.allow_origin)
}

/// Routes one request; returns `None` when the response was already
/// written to the socket (streaming).
pub fn route(ctx: &Arc<ApiCtx>, stream: &mut TcpStream, req: &HttpRequest) -> Option<Response> {
    let route_idx = crate::metrics::Metrics::route_index(&req.path);
    ctx.metrics.requests[route_idx].add(1);

    if req.method == "OPTIONS" {
        return Some(
            Response::empty(204)
                .header("Access-Control-Allow-Origin", &ctx.cfg.allow_origin)
                .header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                .header(
                    "Access-Control-Allow-Headers",
                    "Authorization, Content-Type",
                )
                .header("Access-Control-Max-Age", "86400"),
        );
    }

    // Auth: the /v1 and /admin surfaces; health/metrics stay open.
    if req.path.starts_with("/v1/") || req.path.starts_with("/admin/") {
        if let Some(key) = &ctx.cfg.api_key {
            match req.bearer_token() {
                Some(t) if t == key => {}
                _ => {
                    ctx.metrics.failures[route_idx].add(1);
                    return Some(
                        unauthorized("missing or invalid API key")
                            .header("WWW-Authenticate", "Bearer")
                            .header("Access-Control-Allow-Origin", &ctx.cfg.allow_origin),
                    );
                }
            }
        }
    }

    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health") => Some(health(ctx)),
        ("GET", "/v1/models") => Some(list_models(ctx)),
        ("POST", "/v1/chat/completions") => chat_completions(ctx, stream, req),
        ("POST", "/v1/completions") => completions(ctx, stream, req),
        ("GET", "/metrics") => Some(metrics_endpoint(ctx)),
        ("POST", "/admin/shutdown") => {
            ctx.engine.shutdown();
            Some(Response::json(200, r#"{"status":"draining"}"#))
        }
        (m, p) if matches!(p, "/health" | "/v1/models" | "/metrics") => {
            ctx.metrics.failures[route_idx].add(1);
            Some(bad_request(
                &format!("method {m} not allowed on {p}"),
                "method_not_allowed",
            ))
        }
        _ => {
            ctx.metrics.failures[route_idx].add(1);
            Some(bad_request(
                &format!("unknown route {} {}", req.method, req.path),
                "not_found",
            ))
        }
    }
}

fn health(ctx: &ApiCtx) -> Response {
    let status = if ctx.engine.failed() {
        "degraded"
    } else {
        "ok"
    };
    let code = if ctx.engine.failed() { 503 } else { 200 };
    let body = Json::Obj(vec![
        ("status".to_string(), Json::Str(status.to_string())),
        ("engine".to_string(), Json::Str(ctx.engine_kind.clone())),
        ("model".to_string(), Json::Str(ctx.cfg.model_name.clone())),
        (
            "active_requests".to_string(),
            Json::U64(ctx.metrics.engine_active.get()),
        ),
        (
            "queued_requests".to_string(),
            Json::U64(ctx.metrics.engine_queued.get()),
        ),
        (
            "uptime_seconds".to_string(),
            Json::F64(ctx.metrics.uptime_secs()),
        ),
    ])
    .to_json_string();
    cors(ctx, Response::json(code, body))
}

fn list_models(ctx: &ApiCtx) -> Response {
    let body = Json::Obj(vec![
        ("object".to_string(), Json::Str("list".to_string())),
        (
            "data".to_string(),
            Json::Arr(vec![Json::Obj(vec![
                ("id".to_string(), Json::Str(ctx.cfg.model_name.clone())),
                ("object".to_string(), Json::Str("model".to_string())),
                ("created".to_string(), Json::U64(unix_now())),
                ("owned_by".to_string(), Json::Str("deepinfer".to_string())),
            ])]),
        ),
    ])
    .to_json_string();
    cors(ctx, Response::json(200, body))
}

fn metrics_endpoint(ctx: &ApiCtx) -> Response {
    let text =
        crate::metrics::render_prometheus(&ctx.metrics, &ctx.cfg.model_name, &ctx.engine_kind);
    Response::text(200, text)
}

// ---------------------------------------------------------------------------
// request-body parsing
// ---------------------------------------------------------------------------

/// Sampling knobs parsed from either API's JSON body.
#[derive(Debug, Clone)]
struct ApiParams {
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    min_p: Option<f32>,
    seed: Option<u64>,
    max_tokens: usize,
    stream: bool,
    stops: Vec<String>,
    so: Option<StructuredOutputArgs>,
    include_usage: bool,
}

fn parse_api_params(ctx: &Arc<ApiCtx>, body: &Json) -> Result<ApiParams> {
    let mut p = ApiParams {
        temperature: body
            .get("temperature")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32),
        top_p: body.get("top_p").and_then(|v| v.as_f64()).map(|v| v as f32),
        top_k: body
            .get("top_k")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize),
        min_p: body.get("min_p").and_then(|v| v.as_f64()).map(|v| v as f32),
        seed: body.get("seed").and_then(|v| v.as_u64()),
        max_tokens: body
            .get("max_tokens")
            .or_else(|| body.get("max_completion_tokens"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(ctx.cfg.max_tokens_cap),
        stream: body
            .get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        stops: Vec::new(),
        so: None,
        include_usage: false,
    };
    ensure!(
        p.max_tokens >= 1,
        InvalidArgument,
        "max_tokens must be >= 1"
    );
    ensure!(
        p.max_tokens <= ctx.cfg.max_tokens_cap,
        InvalidArgument,
        "max_tokens {} exceeds the server cap {}",
        p.max_tokens,
        ctx.cfg.max_tokens_cap
    );
    if let Some(stops) = body.get("stop") {
        match stops {
            Json::Str(s) => p.stops.push(s.clone()),
            Json::Arr(items) => {
                ensure!(
                    items.len() <= 4,
                    InvalidArgument,
                    "at most 4 stop sequences"
                );
                for it in items {
                    let Some(s) = it.as_str() else {
                        return Err(Error::invalid("stop items must be strings"));
                    };
                    ensure!(
                        !s.is_empty(),
                        InvalidArgument,
                        "stop sequences must be non-empty"
                    );
                    p.stops.push(s.to_string());
                }
            }
            _ => {
                return Err(Error::invalid(
                    "stop must be a string or an array of strings",
                ))
            }
        }
    }
    if let Some(rf) = body.get("response_format") {
        let ty = rf.str_field("type").unwrap_or("");
        ensure!(
            ty == "json_schema" || ty == "json_object",
            InvalidArgument,
            "response_format.type must be json_schema or json_object"
        );
        let schema = if ty == "json_schema" {
            let js = rf
                .get("json_schema")
                .ok_or_else(|| Error::invalid("response_format.json_schema is required"))?;
            let inner = js.str_field("schema").unwrap_or("");
            if inner.trim().is_empty() {
                "{}"
            } else {
                inner
            }
        } else {
            "{}"
        };
        p.so = Some(StructuredOutputArgs {
            schema_type: SchemaType::Json,
            schema: schema.to_string(),
        });
    }
    if let Some(so) = body.get("structured_output") {
        // Native extension: {"structured_output": {"type": "ebnf"|"regex"|"json", "schema": "..."}}.
        let ty = so.str_field("type").unwrap_or("ebnf");
        let schema = so
            .str_field("schema")
            .ok_or_else(|| Error::invalid("structured_output.schema is required"))?;
        let st = match ty {
            "ebnf" => SchemaType::Ebnf,
            "regex" => SchemaType::Regex,
            "json" => SchemaType::Json,
            other => {
                return Err(Error::invalid(format!(
                    "structured_output.type must be ebnf|regex|json, got {other:?}"
                )))
            }
        };
        p.so = Some(StructuredOutputArgs {
            schema_type: st,
            schema: schema.to_string(),
        });
    }
    if let Some(opts) = body.get("stream_options") {
        if let Some(iu) = opts.get("include_usage").and_then(|v| v.as_bool()) {
            p.include_usage = iu;
        }
    }
    Ok(p)
}

fn build_sampling(api: &ApiParams) -> SamplingParams {
    SamplingParams {
        temperature: api.temperature.unwrap_or(1.0),
        top_p: api.top_p.unwrap_or(1.0),
        top_k: api.top_k.unwrap_or(0),
        min_p: api.min_p.unwrap_or(0.0),
        seq_seed: api.seed,
        so_args: api.so.clone(),
        max_tokens: api.max_tokens,
    }
}

/// Encodes text, validating ids against the vocabulary and the
/// configured prompt-token cap (token-budget admission control: an
/// over-long prompt is rejected before it can squeeze the scheduler).
fn encode_prompt(ctx: &Arc<ApiCtx>, text: &str) -> Result<Vec<u32>> {
    let ids = ctx.tokenizer.encode(text)?;
    ensure!(
        !ids.is_empty(),
        InvalidArgument,
        "prompt encoded to zero tokens"
    );
    ensure!(
        ids.len() <= ctx.cfg.max_prompt_tokens,
        InvalidArgument,
        "prompt of {} tokens exceeds max_prompt_tokens ({}); raise \
         DEEPINFER_SERVER_MAX_PROMPT_TOKENS or shorten the prompt",
        ids.len(),
        ctx.cfg.max_prompt_tokens
    );
    Ok(ids)
}

// ---------------------------------------------------------------------------
// generation drivers
// ---------------------------------------------------------------------------

/// What a generation produced.
struct GenOutcome {
    text: String,
    completion_tokens: usize,
    prompt_tokens: usize,
    finish: Finish,
}

/// One generation's event drain: converts token events into text
/// deltas, applies stop strings and the deadline, and reports the
/// outcome. `on_delta` returns `false` to signal client disconnect
/// (the generation is aborted; the drain still consumes `Finish`).
fn drain_generation(
    ctx: &Arc<ApiCtx>,
    rx: std::sync::mpsc::Receiver<StreamEvent>,
    prompt_tokens: usize,
    api: &ApiParams,
    mut on_delta: impl FnMut(&str) -> bool,
    deadline: Instant,
) -> GenOutcome {
    let mut decoder = DeltaDecoder::new();
    let mut text = String::new();
    let mut completion_tokens = 0usize;
    let mut finish: Option<Finish> = None;
    let mut request_id: Option<u64> = None;
    let mut stopped = false;

    loop {
        if Instant::now() > deadline {
            if let Some(id) = request_id {
                ctx.engine.abort(id);
            }
            ctx.metrics.aborted_requests.add(1);
            finish = Some(Finish::Error("request deadline exceeded".to_string()));
            break;
        }
        let ev = match rx.recv_timeout(Duration::from_millis(250)) {
            Ok(ev) => ev,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        };
        match ev {
            StreamEvent::Started { id } => request_id = Some(id),
            StreamEvent::Token { token, .. } => {
                completion_tokens += 1;
                if !stopped && token != ctx.eos_id {
                    let delta = decoder.push(ctx.token_bytes(token));
                    if !delta.is_empty() {
                        text.push_str(&delta);
                        if !on_delta(&delta) {
                            if let Some(id) = request_id {
                                ctx.engine.abort(id);
                            }
                            ctx.metrics.aborted_requests.add(1);
                            finish = Some(Finish::Aborted);
                            stopped = true;
                        }
                    }
                }
                if !stopped {
                    if let Some(pos) = stop_position(&api.stops, &text) {
                        text.truncate(pos);
                        if let Some(id) = request_id {
                            ctx.engine.abort(id);
                        }
                        finish = Some(Finish::Stop);
                        stopped = true;
                    }
                }
            }
            StreamEvent::Finish {
                reason,
                completion_tokens: n,
                ..
            } => {
                completion_tokens = completion_tokens.max(n);
                finish = Some(reason);
                break;
            }
        }
    }
    if !stopped {
        let tail = decoder.finish();
        if !tail.is_empty() {
            text.push_str(&tail);
            let _ = on_delta(&tail);
            // The flushed tail can itself complete a stop sequence.
            if let Some(pos) = stop_position(&api.stops, &text) {
                text.truncate(pos);
                finish = Some(Finish::Stop);
            }
        }
    }
    GenOutcome {
        text,
        completion_tokens,
        prompt_tokens,
        finish: finish.unwrap_or(Finish::Aborted),
    }
}

/// Byte position of the first occurrence of any stop sequence in
/// `text` (the OpenAI truncation point), or `None`.
fn stop_position(stops: &[String], text: &str) -> Option<usize> {
    stops.iter().filter_map(|s| text.find(s.as_str())).min()
}

fn usage_json(prompt_tokens: usize, completion_tokens: usize) -> (String, Json) {
    (
        "usage".to_string(),
        Json::Obj(vec![
            ("prompt_tokens".to_string(), Json::U64(prompt_tokens as u64)),
            (
                "completion_tokens".to_string(),
                Json::U64(completion_tokens as u64),
            ),
            (
                "total_tokens".to_string(),
                Json::U64((prompt_tokens + completion_tokens) as u64),
            ),
        ]),
    )
}

/// Parses and validates the request JSON body.
fn parse_body(req: &HttpRequest) -> std::result::Result<Json, Response> {
    let text = std::str::from_utf8(&req.body)
        .map_err(|_| bad_request("body must be UTF-8 JSON", "invalid_request"))?;
    if text.trim().is_empty() {
        return Err(bad_request(
            "request body must be a JSON object",
            "invalid_request",
        ));
    }
    Json::parse(text).map_err(|e| bad_request(&format!("invalid JSON: {e}"), "invalid_json"))
}

// ---------------------------------------------------------------------------
// streaming helper
// ---------------------------------------------------------------------------

/// Drives a streaming request end-to-end over SSE. Returns the outcome
/// (the response is already written, including the finish chunk and
/// `[DONE]`), or `Err(status)` when the request could not even start
/// (submit failure — a clean JSON error is written in that case).
struct StreamShape {
    id: String,
    object: String,
    created: u64,
    model: String,
    /// Field carrying the delta text: `"content"` (chat) or `"text"` (completions).
    delta_field: &'static str,
}

fn stream_sse(
    ctx: &Arc<ApiCtx>,
    stream: &mut TcpStream,
    submit: Submit,
    prompt_tokens: usize,
    api: &ApiParams,
    shape: &StreamShape,
    deadline: Instant,
) -> std::result::Result<GenOutcome, u16> {
    // Submit first so pre-stream failures become JSON errors.
    let rx = match ctx.engine.submit(submit) {
        Ok(rx) => rx,
        Err(e) => {
            let status = if e.to_string().contains("engine failed") {
                503u16
            } else {
                400
            };
            return Err(status);
        }
    };
    let mut writer = match ChunkedWriter::sse(stream) {
        Ok(w) => w,
        Err(_) => return Err(500),
    };

    let chunk = |delta_field: &str, delta: &str, finish: Option<&str>| -> String {
        let mut choice = vec![("index".to_string(), Json::U64(0))];
        match delta_field {
            "content" => {
                choice.push((
                    "delta".to_string(),
                    Json::Obj(vec![("content".to_string(), Json::Str(delta.to_string()))]),
                ));
            }
            _ => {
                choice.push(("text".to_string(), Json::Str(delta.to_string())));
            }
        }
        choice.push((
            "finish_reason".to_string(),
            match finish {
                Some(f) => Json::Str(f.to_string()),
                None => Json::Null,
            },
        ));
        Json::Obj(vec![
            ("id".to_string(), Json::Str(shape.id.clone())),
            ("object".to_string(), Json::Str(shape.object.clone())),
            ("created".to_string(), Json::U64(shape.created)),
            ("model".to_string(), Json::Str(shape.model.clone())),
            ("choices".to_string(), Json::Arr(vec![Json::Obj(choice)])),
        ])
        .to_json_string()
    };

    // Role chunk (chat only): announces the assistant before any text.
    if shape.delta_field == "content" {
        let role_chunk = Json::Obj(vec![
            ("id".to_string(), Json::Str(shape.id.clone())),
            ("object".to_string(), Json::Str(shape.object.clone())),
            ("created".to_string(), Json::U64(shape.created)),
            ("model".to_string(), Json::Str(shape.model.clone())),
            (
                "choices".to_string(),
                Json::Arr(vec![Json::Obj(vec![
                    ("index".to_string(), Json::U64(0)),
                    (
                        "delta".to_string(),
                        Json::Obj(vec![(
                            "role".to_string(),
                            Json::Str("assistant".to_string()),
                        )]),
                    ),
                    ("finish_reason".to_string(), Json::Null),
                ])]),
            ),
        ])
        .to_json_string();
        writer.event(&role_chunk);
    }

    let outcome = drain_generation(
        ctx,
        rx,
        prompt_tokens,
        api,
        |delta| {
            let payload = chunk(shape.delta_field, delta, None);
            writer.event(&payload);
            writer.is_ok()
        },
        deadline,
    );

    // Final chunk with the finish reason (empty delta), optional usage
    // chunk, then [DONE]. If the client disconnected mid-stream we
    // skip the terminator (the connection is dead anyway).
    if writer.is_ok() {
        let fin = chunk(shape.delta_field, "", Some(outcome.finish.api_string()));
        writer.event(&fin);
        if api.include_usage {
            let usage = Json::Obj(vec![
                ("id".to_string(), Json::Str(shape.id.clone())),
                ("object".to_string(), Json::Str(shape.object.clone())),
                ("created".to_string(), Json::U64(shape.created)),
                ("model".to_string(), Json::Str(shape.model.clone())),
                ("choices".to_string(), Json::Arr(vec![])),
                usage_json(outcome.prompt_tokens, outcome.completion_tokens),
            ])
            .to_json_string();
            writer.event(&usage);
        }
        let _ = writer.finish();
    }
    let _ = stream.flush();
    Ok(outcome)
}

// ---------------------------------------------------------------------------
// /v1/chat/completions
// ---------------------------------------------------------------------------

fn chat_completions(
    ctx: &Arc<ApiCtx>,
    stream: &mut TcpStream,
    req: &HttpRequest,
) -> Option<Response> {
    let route_idx = crate::metrics::Metrics::route_index("/v1/chat/completions");
    let started = Instant::now();
    let ttft_started = started;

    let body = match parse_body(req) {
        Ok(b) => b,
        Err(resp) => {
            ctx.metrics.failures[route_idx].add(1);
            return Some(resp);
        }
    };
    let api = match parse_api_params(ctx, &body) {
        Ok(p) => p,
        Err(e) => {
            ctx.metrics.failures[route_idx].add(1);
            return Some(bad_request(&e.to_string(), "invalid_request"));
        }
    };

    // ---- messages -> prompt -------------------------------------------------
    let mut messages: Vec<ChatMessage> = Vec::new();
    let raw_msgs = match body.get("messages").and_then(|m| m.as_arr()) {
        Some(m) if !m.is_empty() => m,
        _ => {
            ctx.metrics.failures[route_idx].add(1);
            return Some(bad_request(
                "messages must be a non-empty array",
                "invalid_request",
            ));
        }
    };
    for (i, m) in raw_msgs.iter().enumerate() {
        let role = m.str_field("role").unwrap_or("");
        let content = match m.get("content") {
            Some(Json::Str(s)) => s.clone(),
            Some(Json::Arr(parts)) => {
                // Multimodal-shaped content: concatenate the text parts.
                let mut s = String::new();
                for p in parts {
                    if let Some(t) = p.str_field("text") {
                        s.push_str(t);
                    }
                }
                s
            }
            _ => String::new(),
        };
        if !matches!(role, "system" | "user" | "assistant") || content.is_empty() {
            ctx.metrics.failures[route_idx].add(1);
            return Some(bad_request(
                &format!("messages[{i}]: role must be system|user|assistant and content non-empty"),
                "invalid_request",
            ));
        }
        messages.push(ChatMessage {
            role: role.to_string(),
            content,
        });
    }
    let prompt_text = render_chat_template(&messages, true);
    let prompt_ids = match encode_prompt(ctx, &prompt_text) {
        Ok(p) => p,
        Err(e) => {
            ctx.metrics.failures[route_idx].add(1);
            return Some(bad_request(&e.to_string(), "invalid_request"));
        }
    };

    if ctx.engine.failed() {
        ctx.metrics.failures[route_idx].add(1);
        return Some(unavailable("engine degraded; restart the server"));
    }

    let prompt_tokens = prompt_ids.len();
    let submit = Submit {
        prompt: prompt_ids,
        params: build_sampling(&api),
    };
    let deadline = started + Duration::from_secs(ctx.cfg.request_timeout_secs);
    let shape = StreamShape {
        id: ctx.next_completion_id("chatcmpl"),
        object: "chat.completion.chunk".to_string(),
        created: unix_now(),
        model: ctx.cfg.model_name.clone(),
        delta_field: "content",
    };
    let mut ttft_recorded = false;
    let record_ttft = |ttft_recorded: &mut bool| {
        if !*ttft_recorded {
            ctx.metrics
                .ttft
                .observe(ttft_started.elapsed().as_secs_f64());
            *ttft_recorded = true;
        }
    };

    if api.stream {
        ctx.metrics.active_streams.add(1);
        let outcome = match stream_sse(ctx, stream, submit, prompt_tokens, &api, &shape, deadline) {
            Ok(o) => o,
            Err(status) => {
                ctx.metrics.active_streams.sub(1);
                ctx.metrics.failures[route_idx].add(1);
                let msg = if status == 503 {
                    unavailable("engine degraded; restart the server")
                } else {
                    internal_error("generation could not start")
                };
                let body = String::from_utf8_lossy(&msg.body).into_owned();
                let _ = ChunkedWriter::sse_error(stream, status, &body);
                return None;
            }
        };
        ctx.metrics.active_streams.sub(1);
        ctx.metrics
            .request_latency
            .observe(started.elapsed().as_secs_f64());
        if matches!(outcome.finish, Finish::Error(_)) {
            ctx.metrics.failures[route_idx].add(1);
        }
        None
    } else {
        let rx = match ctx.engine.submit(submit) {
            Ok(rx) => rx,
            Err(e) => {
                ctx.metrics.failures[route_idx].add(1);
                return Some(if e.to_string().contains("engine failed") {
                    unavailable("engine degraded; restart the server")
                } else {
                    internal_error(&e.to_string())
                });
            }
        };
        let outcome = drain_generation(
            ctx,
            rx,
            prompt_tokens,
            &api,
            |_| {
                record_ttft(&mut ttft_recorded);
                true
            },
            deadline,
        );
        ctx.metrics
            .request_latency
            .observe(started.elapsed().as_secs_f64());
        let finish_reason = outcome.finish.api_string();
        let error_detail = match &outcome.finish {
            Finish::Error(msg) => Some(msg.clone()),
            _ => None,
        };
        if error_detail.is_some() {
            ctx.metrics.failures[route_idx].add(1);
        }
        let mut resp_obj = vec![
            ("id".to_string(), Json::Str(shape.id)),
            (
                "object".to_string(),
                Json::Str("chat.completion".to_string()),
            ),
            ("created".to_string(), Json::U64(shape.created)),
            ("model".to_string(), Json::Str(shape.model)),
            (
                "choices".to_string(),
                Json::Arr(vec![Json::Obj(vec![
                    ("index".to_string(), Json::U64(0)),
                    (
                        "message".to_string(),
                        Json::Obj(vec![
                            ("role".to_string(), Json::Str("assistant".to_string())),
                            ("content".to_string(), Json::Str(outcome.text.clone())),
                        ]),
                    ),
                    (
                        "finish_reason".to_string(),
                        Json::Str(finish_reason.to_string()),
                    ),
                ])]),
            ),
            usage_json(outcome.prompt_tokens, outcome.completion_tokens),
        ];
        if let Some(detail) = error_detail {
            resp_obj.push((
                "error".to_string(),
                Json::Obj(vec![
                    ("message".to_string(), Json::Str(detail)),
                    ("type".to_string(), Json::Str("server_error".to_string())),
                ]),
            ));
        }
        let resp = Json::Obj(resp_obj).to_json_string();
        Some(cors(ctx, Response::json(200, resp)))
    }
}

// ---------------------------------------------------------------------------
// /v1/completions
// ---------------------------------------------------------------------------

fn completions(ctx: &Arc<ApiCtx>, stream: &mut TcpStream, req: &HttpRequest) -> Option<Response> {
    let route_idx = crate::metrics::Metrics::route_index("/v1/completions");
    let started = Instant::now();

    let body = match parse_body(req) {
        Ok(b) => b,
        Err(resp) => {
            ctx.metrics.failures[route_idx].add(1);
            return Some(resp);
        }
    };
    let api = match parse_api_params(ctx, &body) {
        Ok(p) => p,
        Err(e) => {
            ctx.metrics.failures[route_idx].add(1);
            return Some(bad_request(&e.to_string(), "invalid_request"));
        }
    };

    // prompt: string | string[] | int[] (token ids).
    let prompt_ids: Vec<u32> = match body.get("prompt") {
        Some(Json::Str(s)) => match encode_prompt(ctx, s) {
            Ok(p) => p,
            Err(e) => {
                ctx.metrics.failures[route_idx].add(1);
                return Some(bad_request(&e.to_string(), "invalid_request"));
            }
        },
        Some(Json::Arr(items)) if !items.is_empty() => {
            if items[0].as_str().is_some() {
                let s = items[0].as_str().unwrap_or("");
                match encode_prompt(ctx, s) {
                    Ok(p) => p,
                    Err(e) => {
                        ctx.metrics.failures[route_idx].add(1);
                        return Some(bad_request(&e.to_string(), "invalid_request"));
                    }
                }
            } else {
                let mut ids = Vec::with_capacity(items.len());
                let mut ok = true;
                for it in items {
                    match it.as_u64().and_then(|v| u32::try_from(v).ok()) {
                        Some(id) => ids.push(id),
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                if !ok || ids.is_empty() {
                    ctx.metrics.failures[route_idx].add(1);
                    return Some(bad_request(
                        "prompt array must be strings or valid token ids",
                        "invalid_request",
                    ));
                }
                ids
            }
        }
        _ => {
            ctx.metrics.failures[route_idx].add(1);
            return Some(bad_request(
                "prompt must be a string, string array or token-id array",
                "invalid_request",
            ));
        }
    };

    if ctx.engine.failed() {
        ctx.metrics.failures[route_idx].add(1);
        return Some(unavailable("engine degraded; restart the server"));
    }

    let prompt_tokens = prompt_ids.len();
    let submit = Submit {
        prompt: prompt_ids,
        params: build_sampling(&api),
    };
    let deadline = started + Duration::from_secs(ctx.cfg.request_timeout_secs);
    let shape = StreamShape {
        id: ctx.next_completion_id("cmpl"),
        object: "text_completion".to_string(),
        created: unix_now(),
        model: ctx.cfg.model_name.clone(),
        delta_field: "text",
    };

    if api.stream {
        ctx.metrics.active_streams.add(1);
        let result = stream_sse(ctx, stream, submit, prompt_tokens, &api, &shape, deadline);
        ctx.metrics.active_streams.sub(1);
        match result {
            Ok(outcome) => {
                ctx.metrics
                    .request_latency
                    .observe(started.elapsed().as_secs_f64());
                if matches!(outcome.finish, Finish::Error(_)) {
                    ctx.metrics.failures[route_idx].add(1);
                }
                None
            }
            Err(status) => {
                ctx.metrics.failures[route_idx].add(1);
                let msg = if status == 503 {
                    unavailable("engine degraded; restart the server")
                } else {
                    internal_error("generation could not start")
                };
                let body = String::from_utf8_lossy(&msg.body).into_owned();
                let _ = ChunkedWriter::sse_error(stream, status, &body);
                None
            }
        }
    } else {
        let outcome = match ctx.engine.submit(submit) {
            Ok(rx) => drain_generation(ctx, rx, prompt_tokens, &api, |_| true, deadline),
            Err(e) => {
                ctx.metrics.failures[route_idx].add(1);
                return Some(internal_error(&e.to_string()));
            }
        };
        ctx.metrics
            .request_latency
            .observe(started.elapsed().as_secs_f64());
        let finish_reason = outcome.finish.api_string();
        let error_detail = match &outcome.finish {
            Finish::Error(msg) => Some(msg.clone()),
            _ => None,
        };
        if error_detail.is_some() {
            ctx.metrics.failures[route_idx].add(1);
        }
        let mut resp_obj = vec![
            ("id".to_string(), Json::Str(shape.id)),
            (
                "object".to_string(),
                Json::Str("text_completion".to_string()),
            ),
            ("created".to_string(), Json::U64(shape.created)),
            ("model".to_string(), Json::Str(shape.model)),
            (
                "choices".to_string(),
                Json::Arr(vec![Json::Obj(vec![
                    ("index".to_string(), Json::U64(0)),
                    ("text".to_string(), Json::Str(outcome.text.clone())),
                    (
                        "finish_reason".to_string(),
                        Json::Str(finish_reason.to_string()),
                    ),
                ])]),
            ),
            usage_json(outcome.prompt_tokens, outcome.completion_tokens),
        ];
        if let Some(detail) = error_detail {
            resp_obj.push((
                "error".to_string(),
                Json::Obj(vec![
                    ("message".to_string(), Json::Str(detail)),
                    ("type".to_string(), Json::Str("server_error".to_string())),
                ]),
            ));
        }
        let resp = Json::Obj(resp_obj).to_json_string();
        Some(cors(ctx, Response::json(200, resp)))
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_matching() {
        let stops = vec!["\n\n".to_string()];
        assert_eq!(stop_position(&stops, "hello\n\n"), Some(5));
        assert_eq!(stop_position(&stops, "hello"), None);
        assert_eq!(stop_position(&stops, ""), None);
        // First occurrence wins even when repeated.
        assert_eq!(stop_position(&stops, "a\n\nb\n\n"), Some(1));
        // Earliest position across multiple stops.
        let multi = vec!["bb".to_string(), "aa".to_string()];
        assert_eq!(stop_position(&multi, "xaabb"), Some(1));
    }

    #[test]
    fn error_body_shape() {
        let b = error_body("boom", "invalid_request_error", "invalid_request");
        assert!(b.contains("\"message\":\"boom\""));
        assert!(b.contains("\"code\":\"invalid_request\""));
        assert!(b.contains("\"type\":\"invalid_request_error\""));
    }
}
