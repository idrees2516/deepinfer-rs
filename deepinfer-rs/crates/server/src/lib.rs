//! # deepinfer-server
//!
//! The production serving layer of **DeepInfer-rs**: an
//! OpenAI-compatible HTTP API on top of the continuous-batching
//! scheduler and the DeepSeek-V4.1-Flash model.
//!
//! * **Zero external dependencies** — the HTTP/1.1 server, SSE
//!   streaming, JSON, metrics and the engine loop are all implemented
//!   in this crate on `std` + the workspace crates (the same
//!   supply-chain posture as the rest of the workspace).
//! * **Two engines, one API** — the CPU golden path
//!   ([`engine::CpuTokenEngine`] over [`deepinfer_runtime::DeepSeekModel`])
//!   and the GPU paged-serve path ([`engine::GpuTokenEngine`] over
//!   [`deepinfer_runtime::DeviceModel`]).
//! * **Endpoints** — `GET /health`, `GET /v1/models`,
//!   `POST /v1/chat/completions` (SSE streaming + non-streaming),
//!   `POST /v1/completions`, `GET /metrics` (Prometheus),
//!   `POST /admin/shutdown` (drain).
//! * **Operational hardening** — bounded bodies/headers/connections,
//!   per-request deadlines, bearer auth, CORS, graceful drain,
//!   cancellation on client disconnect, and degraded-mode health after
//!   an engine failure.
//!
//! ```text
//! # tiny self-check (random weights, no checkpoint needed):
//! cargo run -p deepinfer-server --bin deepinfer-server -- --tiny
//!
//! # serve a checkpoint:
//! cargo run --release -p deepinfer-server --bin deepinfer-server -- \
//!     --model /path/to/DeepSeek-V4.1-Flash-1017
//! ```
//!
//! Then, from any OpenAI client:
//!
//! ```text
//! curl http://127.0.0.1:8080/v1/chat/completions \
//!   -H 'Content-Type: application/json' \
//!   -d '{"model":"deepinfer-rs","messages":[{"role":"user","content":"Hi"}]}'
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod api;
pub mod chat;
pub mod config;
pub mod engine;
pub mod http;
pub mod metrics;
pub mod prefix;
pub mod spec;

use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use deepinfer_core::Result;

use api::ApiCtx;
use config::ServerConfig;
use engine::{ServingHandle, TokenEngine};
use http::{configure_stream, read_request, HttpError, MAX_REQUESTS_PER_CONN};
use metrics::SharedMetrics;

/// A running server. Dropping it stops accepting connections; call
/// [`Server::wait`] to block until the accept loop exits.
pub struct Server {
    listener: TcpListener,
    accept: Option<std::thread::JoinHandle<()>>,
    draining: Arc<AtomicBool>,
    metrics: SharedMetrics,
    handle: ServingHandle,
    cfg: ServerConfig,
}

impl Server {
    /// Starts the server: binds, spawns the accept loop, returns
    /// immediately.
    ///
    /// # Errors
    /// IO errors from binding; engine construction errors were raised
    /// by the caller.
    pub fn start(
        cfg: ServerConfig,
        engine: Box<dyn TokenEngine>,
        tokenizer: Box<dyn deepinfer_core::tokenizer::Tokenizer>,
        eos_id: u32,
        vocab_bytes: Vec<Vec<u8>>,
        metrics: SharedMetrics,
        engine_kind: &str,
    ) -> Result<Server> {
        cfg.validate()?;
        let listener = TcpListener::bind(cfg.bind_addr())
            .map_err(|e| deepinfer_core::Error::Io(format!("bind {}: {e}", cfg.bind_addr())))?;
        let sched_cfg = cfg.scheduler;
        let spec_draft_len = cfg.spec_draft_len;
        let spec_adaptive = cfg.spec_adaptive;
        let prefix_cache_tokens = cfg.prefix_cache_tokens;
        // Byte-mode serving: when the tokenizer's vocabulary is smaller
        // than the model's, sampling is capped to it (the ids above it
        // would not survive detokenization).
        let sample_vocab_cap = {
            let tv = tokenizer.vocab_size();
            if tv < engine.vocab_size() {
                tv
            } else {
                0
            }
        };
        let handle = engine::start(
            engine,
            eos_id,
            sched_cfg,
            Arc::clone(&metrics),
            engine::ServeTuning {
                spec_draft_len,
                spec_adaptive,
                prefix_cache_tokens,
                sample_vocab_cap,
                spec_greedy: false,
            },
        )?;
        let ctx = Arc::new(ApiCtx::new(
            cfg.clone(),
            handle.clone(),
            metrics.clone(),
            engine_kind,
            tokenizer,
            eos_id,
            vocab_bytes,
        ));
        let draining = Arc::new(AtomicBool::new(false));
        let conns = Arc::new(AtomicUsize::new(0));
        let accept_listener = listener
            .try_clone()
            .map_err(|e| deepinfer_core::Error::Io(format!("listener clone: {e}")))?;
        let accept_ctx = Arc::clone(&ctx);
        let accept_drain = Arc::clone(&draining);
        let accept_conns = Arc::clone(&conns);
        let accept = std::thread::Builder::new()
            .name("deepinfer-accept".to_string())
            .spawn(move || {
                accept_loop(accept_listener, accept_ctx, accept_drain, accept_conns);
            })
            .map_err(|e| deepinfer_core::Error::Io(format!("accept thread: {e}")))?;
        Ok(Server {
            listener,
            accept: Some(accept),
            draining,
            metrics,
            handle,
            cfg,
        })
    }

    /// The bound socket address (useful when port 0 was requested).
    pub fn local_addr(&self) -> std::net::SocketAddr {
        self.listener
            .local_addr()
            .unwrap_or_else(|_| "0.0.0.0:0".parse().unwrap())
    }

    /// The engine handle (shutdown, health probing).
    pub fn engine(&self) -> &ServingHandle {
        &self.handle
    }

    /// Shared metrics (live gauges for tests and operators).
    pub fn metrics(&self) -> &SharedMetrics {
        &self.metrics
    }

    /// The server configuration.
    pub fn config(&self) -> &ServerConfig {
        &self.cfg
    }

    /// Stops accepting new connections (existing ones drain).
    pub fn stop_accepting(&self) {
        self.draining.store(true, Ordering::Release);
    }

    /// Requests a full engine shutdown and waits for the accept loop to
    /// exit.
    pub fn shutdown(mut self) {
        self.handle.shutdown();
        self.stop_accepting();
        self.wait();
    }

    /// Blocks until the accept loop exits (after
    /// [`Server::stop_accepting`] or engine shutdown).
    pub fn wait(&mut self) {
        if let Some(t) = self.accept.take() {
            let _ = t.join();
        }
    }
}

/// Blanket forwarding so `Box<dyn TokenEngine>` is itself a
/// [`TokenEngine`] (the server's type-erased engine handle).
impl TokenEngine for Box<dyn TokenEngine> {
    fn attach(&mut self, seq: u32) -> Result<()> {
        (**self).attach(seq)
    }
    fn prefill(&mut self, seq: u32, tokens: &[u32]) -> Result<engine::ForwardOut> {
        (**self).prefill(seq, tokens)
    }
    fn decode(&mut self, seq: u32, token: u32) -> Result<engine::ForwardOut> {
        (**self).decode(seq, token)
    }
    fn decode_batch(&mut self, batch: &[(u32, u32)]) -> Result<Vec<engine::ForwardOut>> {
        (**self).decode_batch(batch)
    }
    fn detach(&mut self, seq: u32) -> Result<()> {
        (**self).detach(seq)
    }
    fn vocab_size(&self) -> usize {
        (**self).vocab_size()
    }
    fn vocab_bytes(&self) -> Option<&[Vec<u8>]> {
        (**self).vocab_bytes()
    }
    fn supports_spec(&self) -> bool {
        (**self).supports_spec()
    }
    fn draft_step(&mut self, last_token: u32, hidden: &[f32]) -> Result<deepinfer_mtp::DraftToken> {
        (**self).draft_step(last_token, hidden)
    }
    fn verify(&mut self, seq: u32, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        (**self).verify(seq, tokens)
    }
    fn rollback(&mut self, seq: u32, n: usize) -> Result<()> {
        (**self).rollback(seq, n)
    }
    fn snapshot_prefix(
        &mut self,
        seq: u32,
        n: usize,
    ) -> Result<Option<std::sync::Arc<deepinfer_runtime::PrefixState>>> {
        (**self).snapshot_prefix(seq, n)
    }
    fn restore_prefix(&mut self, seq: u32, snap: &deepinfer_runtime::PrefixState) -> Result<()> {
        (**self).restore_prefix(seq, snap)
    }
}

fn accept_loop(
    listener: TcpListener,
    ctx: Arc<ApiCtx>,
    draining: Arc<AtomicBool>,
    conns: Arc<AtomicUsize>,
) {
    // Unbounded accept backlog, bounded concurrency: the counter is the
    // admission control (each connection spawns one worker thread that
    // lives for the connection; the engine is the scarce resource and
    // the scheduler bounds it independently).
    listener.set_nonblocking(true).ok();
    loop {
        if draining.load(Ordering::Acquire) {
            return;
        }
        match listener.accept() {
            Ok((stream, _peer)) => {
                let live = conns.load(Ordering::Acquire);
                if live >= ctx.cfg.max_connections {
                    ctx.metrics.connections_rejected.add(1);
                    let mut s = stream;
                    let resp = http::Response::json(
                        503,
                        r#"{"error":{"message":"too many connections","type":"server_error","param":null,"code":"overloaded"}}"#,
                    );
                    configure_stream(&s, Duration::from_secs(5), Duration::from_secs(5));
                    let _ = resp.write(&mut s, false);
                    continue;
                }
                ctx.metrics.connections.add(1);
                conns.fetch_add(1, Ordering::AcqRel);
                let ctx2 = Arc::clone(&ctx);
                let conns2 = Arc::clone(&conns);
                let spawned = std::thread::Builder::new()
                    .name("deepinfer-conn".to_string())
                    .spawn(move || {
                        handle_connection(stream, &ctx2);
                        conns2.fetch_sub(1, Ordering::AcqRel);
                    });
                if spawned.is_err() {
                    conns.fetch_sub(1, Ordering::AcqRel);
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(_) => {
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }
}

fn handle_connection(mut stream: TcpStream, ctx: &Arc<ApiCtx>) {
    let idle = Duration::from_secs(ctx.cfg.keepalive_idle_secs);
    let write = Duration::from_secs(ctx.cfg.write_timeout_secs);
    configure_stream(&stream, idle, write);
    let mut served = 0usize;
    while served < MAX_REQUESTS_PER_CONN {
        let req = match read_request(&mut stream, ctx.cfg.max_body_bytes) {
            Ok(Some(r)) => r,
            Ok(None) => return, // orderly close
            Err(e) => {
                write_http_error(&mut stream, &e);
                return;
            }
        };
        served += 1;
        let keep_alive = wants_keep_alive(&req);
        match api::route(ctx, &mut stream, &req) {
            Some(resp) => {
                if resp.write(&mut stream, keep_alive).is_err() {
                    return;
                }
            }
            None => {
                // Streaming wrote directly; honor keep-alive.
                if !keep_alive {
                    return;
                }
            }
        }
        if !keep_alive {
            return;
        }
    }
}

fn wants_keep_alive(req: &http::HttpRequest) -> bool {
    match req.header("connection") {
        Some(v) if v.eq_ignore_ascii_case("close") => false,
        Some(v) if v.eq_ignore_ascii_case("keep-alive") => true,
        _ => true, // HTTP/1.1 default
    }
}

fn write_http_error(stream: &mut TcpStream, e: &HttpError) {
    let msg = match e {
        HttpError::BadRequest(m) => format!("malformed request: {m}"),
        other => other.phrase().to_string(),
    };
    let body = format!(
        "{{\"error\":{{\"message\":{:?},\"type\":\"invalid_request_error\",\"param\":null,\"code\":\"{}\"}}}}",
        msg,
        e.code()
    );
    let resp = http::Response::json(e.status(), body);
    let _ = resp.write(stream, false);
}
