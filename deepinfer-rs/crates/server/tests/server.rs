//! End-to-end server tests: boot the real server on an ephemeral port
//! with the tiny deterministic model and drive it over raw TCP — the
//! same wire path production clients take.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::Path;
use std::time::{Duration, Instant};

use deepinfer_core::tokenizer::{ByteTokenizer, Tokenizer};
use deepinfer_runtime::{DeepSeekModel, ModelConfig};
use deepinfer_server::config::ServerConfig;
use deepinfer_server::engine::CpuTokenEngine;
use deepinfer_server::metrics;
use deepinfer_server::Server;

struct TestServer {
    server: Option<Server>,
    port: u16,
}

impl TestServer {
    fn start() -> TestServer {
        Self::start_with(|cfg| cfg)
    }

    fn start_with(mutate: impl FnOnce(ServerConfig) -> ServerConfig) -> TestServer {
        let model = DeepSeekModel::random_init(&ModelConfig::v41_tiny(), 42).unwrap();
        let tok: Box<dyn Tokenizer> = Box::new(ByteTokenizer::new(0).unwrap());
        let eos = 0u32;
        let vocab_bytes = DeepSeekModel::vocab_bytes_from_tokenizer(tok.as_ref(), eos).unwrap();
        let mut model = model;
        model.set_vocab_bytes(vocab_bytes.clone());
        let cfg = mutate(ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 0, // ephemeral
            request_timeout_secs: 30,
            max_tokens_cap: 64,
            ..ServerConfig::default()
        });
        cfg.validate().unwrap();
        let server = Server::start(
            cfg,
            Box::new(CpuTokenEngine::new(model)),
            tok,
            eos,
            vocab_bytes,
            metrics::shared(),
            "cpu",
        )
        .expect("server start");
        let port = server.local_addr().port();
        TestServer {
            server: Some(server),
            port,
        }
    }

    fn connect(&self) -> TcpStream {
        TcpStream::connect(("127.0.0.1", self.port)).expect("connect")
    }

    fn stop(&mut self) {
        if let Some(s) = self.server.take() {
            s.shutdown();
        }
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Sends a request and reads the full (Content-Length framed) response.
fn roundtrip(stream: &mut TcpStream, raw: &str) -> String {
    stream.write_all(raw.as_bytes()).expect("write");
    stream.flush().expect("flush");
    read_response(stream)
}

/// Reads one HTTP response off the stream (headers + Content-Length body
/// or chunked until the zero chunk).
fn read_response(stream: &mut TcpStream) -> String {
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .unwrap();
    let mut buf: Vec<u8> = Vec::new();
    let mut byte = [0u8; 1];
    // Headers.
    loop {
        stream.read_exact(&mut byte).expect("header byte");
        buf.push(byte[0]);
        if buf.ends_with(b"\r\n\r\n") {
            break;
        }
        assert!(buf.len() < 64 * 1024, "runaway headers");
    }
    let head = String::from_utf8_lossy(&buf).into_owned();
    let (headers, mut body) = (head.clone(), Vec::new());
    let chunked = headers
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked");
    let content_length: Option<usize> = headers
        .lines()
        .find(|l| l.to_ascii_lowercase().starts_with("content-length:"))
        .and_then(|l| l.split(':').nth(1))
        .and_then(|v| v.trim().parse().ok());
    if let Some(n) = content_length {
        body = vec![0u8; n];
        stream.read_exact(&mut body).expect("body");
    } else if chunked {
        // Chunked: parse chunk framing until the 0 chunk.
        loop {
            let mut size_line: Vec<u8> = Vec::new();
            loop {
                stream.read_exact(&mut byte).expect("chunk size byte");
                if byte[0] == b'\n' {
                    break;
                }
                if byte[0] != b'\r' {
                    size_line.push(byte[0]);
                }
            }
            let size = usize::from_str_radix(
                std::str::from_utf8(&size_line).expect("size ascii").trim(),
                16,
            )
            .expect("chunk size");
            if size == 0 {
                // trailing CRLF after the last chunk
                let mut tail = [0u8; 2];
                let _ = stream.read_exact(&mut tail);
                break;
            }
            let mut chunk = vec![0u8; size];
            stream.read_exact(&mut chunk).expect("chunk");
            body.extend_from_slice(&chunk);
            let mut crlf = [0u8; 2];
            stream.read_exact(&mut crlf).expect("chunk crlf");
        }
    }
    format!("{}{}", head, String::from_utf8_lossy(&body))
}

fn post_json(ts: &TestServer, path: &str, body: &str) -> (u16, String) {
    let mut s = ts.connect();
    let raw = format!(
        "POST {} HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        path,
        body.len(),
        body
    );
    let resp = roundtrip(&mut s, &raw);
    parse_status_and_body(&resp)
}

fn get(ts: &TestServer, path: &str) -> (u16, String) {
    let mut s = ts.connect();
    let raw = format!(
        "GET {} HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n",
        path
    );
    let resp = roundtrip(&mut s, &raw);
    parse_status_and_body(&resp)
}

fn parse_status_and_body(resp: &str) -> (u16, String) {
    let status: u16 = resp
        .lines()
        .next()
        .and_then(|l| l.split(' ').nth(1))
        .and_then(|c| c.parse().ok())
        .expect("status line");
    let body = resp
        .split_once("\r\n\r\n")
        .map(|(_, b)| b.to_string())
        .unwrap_or_default();
    (status, body)
}

// ---------------------------------------------------------------------------

#[test]
fn health_and_models_and_metrics() {
    let mut ts = TestServer::start();
    let (code, body) = get(&ts, "/health");
    assert_eq!(code, 200);
    assert!(body.contains("\"status\":\"ok\""));
    assert!(body.contains("\"engine\":\"cpu\""));

    let (code, body) = get(&ts, "/v1/models");
    assert_eq!(code, 200);
    assert!(body.contains("\"object\":\"list\""));
    assert!(body.contains("\"owned_by\":\"deepinfer\""));

    let (code, body) = get(&ts, "/metrics");
    assert_eq!(code, 200);
    assert!(body.contains("deepinfer_build_info{engine=\"cpu\""));
    assert!(body.contains("deepinfer_uptime_seconds"));

    // Unknown route -> 400 (JSON error body).
    let (code, body) = get(&ts, "/nope");
    assert_eq!(code, 400);
    assert!(body.contains("\"error\""));
    ts.stop();
}

#[test]
fn chat_completion_nonstream_roundtrip() {
    let mut ts = TestServer::start();
    let body = r#"{
        "model": "deepinfer-rs",
        "messages": [
            {"role": "system", "content": "You are a test."},
            {"role": "user", "content": "Hello"}
        ],
        "temperature": 0.0,
        "max_tokens": 8,
        "stream": false
    }"#;
    let (code, resp) = post_json(&ts, "/v1/chat/completions", body);
    assert_eq!(code, 200, "{resp}");
    assert!(resp.contains("\"object\":\"chat.completion\""));
    assert!(resp.contains("\"role\":\"assistant\""));
    assert!(resp.contains("\"finish_reason\":\"length\""));
    assert!(resp.contains("\"prompt_tokens\":"));
    assert!(resp.contains("\"completion_tokens\":8"));

    // Determinism: the same request produces identical choices/usage
    // (ids and timestamps differ by design).
    let (code2, resp2) = post_json(&ts, "/v1/chat/completions", body);
    assert_eq!(code2, 200);
    let strip_id = |r: &str| -> String {
        r.replace(
            &format!(
                "\"id\":\"{}",
                r.split("\"id\":\"")
                    .nth(1)
                    .unwrap_or("")
                    .split('"')
                    .next()
                    .unwrap_or("")
            ),
            "\"id\":\"X\"",
        )
    };
    assert_eq!(
        strip_id(&resp),
        strip_id(&resp2),
        "greedy decode must be deterministic"
    );
    ts.stop();
}

#[test]
fn chat_completion_matches_offline_generate() {
    // The server's chat output must equal the offline `generate` on the
    // same tokenized prompt (the CPU golden oracle).
    let mut ts = TestServer::start();
    let body = r#"{
        "messages": [{"role": "user", "content": "AB"}],
        "temperature": 0.0,
        "max_tokens": 6
    }"#;
    let (code, body) = post_json(&ts, "/v1/chat/completions", body);
    assert_eq!(code, 200, "{body}");
    let json = deepinfer_core::json::Json::parse(&body).unwrap();
    let content = json
        .get("choices")
        .and_then(|c| c.as_arr())
        .and_then(|c| c.first())
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .expect("content")
        .to_string();

    // Offline: same model + same prompt through the template.
    let model = DeepSeekModel::random_init(&ModelConfig::v41_tiny(), 42).unwrap();
    let messages = vec![deepinfer_server::chat::ChatMessage {
        role: "user".to_string(),
        content: "AB".to_string(),
    }];
    let prompt_text = deepinfer_server::chat::render_chat_template(&messages, true);
    let tok = ByteTokenizer::new(0).unwrap();
    let prompt = tok.encode(&prompt_text).unwrap();
    let env = deepinfer_core::env::Env::default();
    let mut model = model;
    let out = deepinfer_runtime::generate(
        &mut model,
        vec![deepinfer_runtime::GenerateRequest {
            prompt,
            params: deepinfer_sampler::SamplingParams {
                temperature: 0.0,
                max_tokens: 6,
                ..Default::default()
            },
        }],
        0,
        &env,
    )
    .unwrap();
    // The byte tokenizer maps ids 1:1 to bytes; compare via bytes.
    let expected_bytes: Vec<u8> = out[0].tokens.iter().map(|&t| t as u8).collect();
    let expected = String::from_utf8_lossy(&expected_bytes).into_owned();
    assert_eq!(content, expected, "server output must equal generate()");
    ts.stop();
}

#[test]
fn chat_completion_with_speculation_matches_offline_generate() {
    // The full HTTP stack with DSpark speculative decoding enabled
    // (draft 4 per step): the streamed content must STILL equal the
    // offline plain `generate` on the same tokenized prompt — the
    // end-to-end parity oracle for speculative serving.
    let mut ts = TestServer::start_with(|cfg| ServerConfig {
        spec_draft_len: 4,
        ..cfg
    });
    let body = r#"{
        "messages": [{"role": "user", "content": "xy"}],
        "temperature": 0.0,
        "max_tokens": 10
    }"#;
    let (code, resp) = post_json(&ts, "/v1/chat/completions", body);
    assert_eq!(code, 200, "{resp}");
    let json = deepinfer_core::json::Json::parse(&resp).unwrap();
    let spec_content = json
        .get("choices")
        .and_then(|c| c.as_arr())
        .and_then(|c| c.first())
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .expect("content")
        .to_string();

    // Offline: same model + same prompt through the template.
    let model = DeepSeekModel::random_init(&ModelConfig::v41_tiny(), 42).unwrap();
    let messages = vec![deepinfer_server::chat::ChatMessage {
        role: "user".to_string(),
        content: "xy".to_string(),
    }];
    let prompt_text = deepinfer_server::chat::render_chat_template(&messages, true);
    let tok = ByteTokenizer::new(0).unwrap();
    let prompt = tok.encode(&prompt_text).unwrap();
    let env = deepinfer_core::env::Env::default();
    let mut model = model;
    let out = deepinfer_runtime::generate(
        &mut model,
        vec![deepinfer_runtime::GenerateRequest {
            prompt,
            params: deepinfer_sampler::SamplingParams {
                temperature: 0.0,
                max_tokens: 10,
                ..Default::default()
            },
        }],
        0,
        &env,
    )
    .unwrap();
    let expected_bytes: Vec<u8> = out[0].tokens.iter().map(|&t| t as u8).collect();
    let expected = String::from_utf8_lossy(&expected_bytes).into_owned();
    assert_eq!(
        spec_content, expected,
        "speculative serving output must equal generate()"
    );

    // The spec metrics are exposed on /metrics.
    let (code, text) = get(&ts, "/metrics");
    assert_eq!(code, 200);
    assert!(
        text.contains("deepinfer_spec_rounds_total"),
        "spec series present"
    );
    assert!(text.contains("deepinfer_spec_draft_tokens_total"));
    ts.stop();
}

#[test]
fn chat_completion_stream_sse() {
    let mut ts = TestServer::start();
    let mut s = ts.connect();
    let body = r#"{
        "messages": [{"role": "user", "content": "Stream"}],
        "temperature": 0.0,
        "max_tokens": 4,
        "stream": true,
        "stream_options": {"include_usage": true}
    }"#;
    let raw = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );
    let resp = roundtrip(&mut s, &raw);
    assert!(resp.starts_with("HTTP/1.1 200"), "{resp}");
    assert!(resp.contains("text/event-stream"), "{resp}");
    assert!(resp.contains("chat.completion.chunk"), "{resp}");
    // Role chunk first, then content chunks, finish chunk, usage, DONE.
    let role_pos = resp.find("\"role\":\"assistant\"").expect("role chunk");
    let done_pos = resp.find("data: [DONE]").expect("done marker");
    let finish_pos = resp
        .find("\"finish_reason\":\"length\"")
        .expect("finish chunk");
    assert!(role_pos < finish_pos);
    assert!(finish_pos < done_pos);
    assert!(resp.contains("\"usage\":"), "usage chunk missing: {resp}");
    ts.stop();
}

#[test]
fn completions_endpoint_and_token_prompts() {
    let mut ts = TestServer::start();
    // Text prompt.
    let (code, body) = post_json(
        &ts,
        "/v1/completions",
        r#"{"prompt": "Once", "temperature": 0.0, "max_tokens": 5}"#,
    );
    assert_eq!(code, 200, "{body}");
    assert!(body.contains("\"object\":\"text_completion\""));
    assert!(body.contains("\"text\":"));

    // Token-id prompt.
    let (code, body) = post_json(
        &ts,
        "/v1/completions",
        r#"{"prompt": [65, 66], "temperature": 0.0, "max_tokens": 3}"#,
    );
    assert_eq!(code, 200, "{body}");
    assert!(body.contains("\"completion_tokens\":3"));
    ts.stop();
}

#[test]
fn validation_errors() {
    let mut ts = TestServer::start();
    // Missing messages.
    let (code, body) = post_json(&ts, "/v1/chat/completions", r#"{"model":"x"}"#);
    assert_eq!(code, 400);
    assert!(body.contains("messages"));

    // Bad role.
    let (code, _body) = post_json(
        &ts,
        "/v1/chat/completions",
        r#"{"messages": [{"role": "tool", "content": "x"}]}"#,
    );
    assert_eq!(code, 400);

    // max_tokens over the cap.
    let (code, _) = post_json(
        &ts,
        "/v1/chat/completions",
        r#"{"messages": [{"role": "user", "content": "x"}], "max_tokens": 100000}"#,
    );
    assert_eq!(code, 400);

    // Invalid JSON.
    let (code, body) = post_json(&ts, "/v1/chat/completions", "{not json");
    assert_eq!(code, 400);
    assert!(body.contains("invalid JSON"));

    // Method not allowed.
    let (code, _) = get(&ts, "/v1/chat/completions");
    assert_eq!(code, 400);
    ts.stop();
}

#[test]
fn api_key_enforcement() {
    let mut ts = TestServer::start_with(|c| ServerConfig {
        api_key: Some("test-key-12345".to_string()),
        ..c
    });
    // No key -> 401.
    let (code, _) = post_json(
        &ts,
        "/v1/chat/completions",
        r#"{"messages": [{"role": "user", "content": "x"}]}"#,
    );
    assert_eq!(code, 401);
    // Wrong key -> 401.
    let (code, _) = post_json_keyed(
        &ts,
        "/v1/chat/completions",
        r#"{"messages": [{"role": "user", "content": "x"}], "max_tokens": 2}"#,
        "Bearer wrong",
    );
    assert_eq!(code, 401);
    // Right key -> 200. Health stays open.
    let (code, _) = post_json_keyed(
        &ts,
        "/v1/chat/completions",
        r#"{"messages": [{"role": "user", "content": "x"}], "max_tokens": 2}"#,
        "Bearer test-key-12345",
    );
    assert_eq!(code, 200);
    let (code, _) = get(&ts, "/health");
    assert_eq!(code, 200);
    ts.stop();
}

fn post_json_keyed(ts: &TestServer, path: &str, body: &str, auth: &str) -> (u16, String) {
    let mut s = ts.connect();
    let raw = format!(
        "POST {} HTTP/1.1\r\nHost: t\r\nAuthorization: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        path,
        auth,
        body.len(),
        body
    );
    let resp = roundtrip(&mut s, &raw);
    parse_status_and_body(&resp)
}

#[test]
fn concurrent_requests_batch() {
    let mut ts = TestServer::start_with(|c| ServerConfig {
        scheduler: deepinfer_scheduler::SchedulerConfig {
            max_running_requests: 8,
            max_batch_tokens: 4096,
            chunked_prefill_size: 512,
            batch_invariant: false,
        },
        ..c
    });
    let addr = ("127.0.0.1", ts.port);
    let mut handles = Vec::new();
    for i in 0..2 {
        handles.push(std::thread::spawn(move || {
            let mut s = TcpStream::connect(addr).expect("connect");
            let body = format!(
                r#"{{"messages": [{{"role": "user", "content": "req {i}"}}], "temperature": 0.0, "max_tokens": 4}}"#
            );
            let raw = format!(
                "POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            s.write_all(raw.as_bytes()).unwrap();
            s.flush().unwrap();
            let resp = read_response(&mut s);
            let (code, body) = parse_status_and_body(&resp);
            assert_eq!(code, 200, "{body}");
            body
        }));
    }
    let bodies: Vec<String> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    // Every request produced a completion with 4 tokens.
    for b in &bodies {
        assert!(b.contains("\"completion_tokens\":4"), "{b}");
        assert!(!b.contains("\"finish_reason\":\"error\""), "{b}");
    }
    let content_of = |b: &str| -> String {
        let j = deepinfer_core::json::Json::parse(b).unwrap();
        j.get("choices")
            .and_then(|c| c.as_arr())
            .and_then(|c| c.first())
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap()
            .to_string()
    };
    assert_eq!(
        content_of(&bodies[0]).len(),
        4,
        "byte tokenizer: 4 tokens = 4 bytes"
    );
    ts.stop();
}

#[test]
fn client_disconnect_aborts_generation() {
    let mut ts = TestServer::start();
    // Open a streaming request and drop the socket after the first
    // chunks arrive; the engine must abort and free the slot.
    let mut s = ts.connect();
    let body = r#"{"messages": [{"role": "user", "content": "x"}], "temperature": 0.0, "max_tokens": 60, "stream": true}"#;
    let raw = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );
    s.write_all(raw.as_bytes()).unwrap();
    s.flush().unwrap();
    // Read a little (the SSE head + some chunks).
    s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    let mut got = [0u8; 512];
    let _ = s.read(&mut got);
    drop(s); // client goes away

    // The abort must register in the metrics within a bounded time.
    let server = ts.server.as_ref().unwrap();
    let deadline = Instant::now() + Duration::from_secs(15);
    while server.metrics().aborted_requests.get() == 0 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(
        server.metrics().aborted_requests.get() >= 1,
        "abort not registered"
    );
    // And the engine must drain back to zero active requests.
    let deadline = Instant::now() + Duration::from_secs(15);
    while server.metrics().engine_active.get() > 0 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    assert_eq!(server.metrics().engine_active.get(), 0, "request leaked");

    // The server keeps serving new requests afterwards.
    let (code, _) = post_json(
        &ts,
        "/v1/chat/completions",
        r#"{"messages": [{"role": "user", "content": "after"}], "temperature": 0.0, "max_tokens": 2}"#,
    );
    assert_eq!(code, 200);
    ts.stop();
}

#[test]
fn stop_string_truncates_completion() {
    let mut ts = TestServer::start();
    // First: what does the model produce?
    let (code, body) = post_json(
        &ts,
        "/v1/chat/completions",
        r#"{"messages": [{"role": "user", "content": "S"}], "temperature": 0.0, "max_tokens": 8}"#,
    );
    assert_eq!(code, 200, "{body}");
    let j = deepinfer_core::json::Json::parse(&body).unwrap();
    let full = j
        .get("choices")
        .and_then(|c| c.as_arr())
        .and_then(|c| c.first())
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap()
        .to_string();
    assert!(full.len() >= 2, "need at least 2 chars to stop on");

    // Stop on the LAST character: the completion must be everything
    // before its first occurrence (OpenAI stop semantics).
    let stop_on = full.chars().last().unwrap().to_string();
    let expected = match full.find(&stop_on) {
        Some(pos) => full[..pos].to_string(),
        None => full.clone(),
    };
    let req = format!(
        r#"{{"messages": [{{"role": "user", "content": "S"}}], "temperature": 0.0, "max_tokens": 8, "stop": {:?}}}"#,
        stop_on
    );
    let (code, body) = post_json(&ts, "/v1/chat/completions", &req);
    assert_eq!(code, 200, "{body}");
    let j = deepinfer_core::json::Json::parse(&body).unwrap();
    let stopped = j
        .get("choices")
        .and_then(|c| c.as_arr())
        .and_then(|c| c.first())
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap()
        .to_string();
    assert_eq!(
        stopped, expected,
        "stop must truncate at the first match; full={full:?} stop_on={stop_on:?}"
    );
    assert!(body.contains("\"finish_reason\":\"stop\""));
    ts.stop();
}

#[test]
fn oversize_body_rejected() {
    let mut ts = TestServer::start_with(|c| ServerConfig {
        max_body_bytes: 2048,
        ..c
    });
    let big = "x".repeat(4096);
    let body = format!(
        r#"{{"messages": [{{"role": "user", "content": "{}"}}]}}"#,
        big
    );
    let (code, _) = post_json(&ts, "/v1/chat/completions", &body);
    assert_eq!(code, 413);
    ts.stop();
}

#[test]
fn keep_alive_serves_two_requests() {
    let mut ts = TestServer::start();
    let mut s = ts.connect();
    for _ in 0..2 {
        let body = r#"{"messages": [{"role": "user", "content": "ka"}], "temperature": 0.0, "max_tokens": 2}"#;
        let raw = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        let resp = roundtrip(&mut s, &raw);
        let (code, b) = parse_status_and_body(&resp);
        assert_eq!(code, 200, "{b}");
        assert!(b.contains("\"object\":\"chat.completion\""));
    }
    ts.stop();
}

#[test]
fn structured_output_json_schema() {
    let mut ts = TestServer::start();
    // A JSON schema accepting any object: the grammar must constrain
    // generation (here with the byte tokenizer the mask still applies).
    let (code, body) = post_json(
        &ts,
        "/v1/chat/completions",
        r#"{
            "messages": [{"role": "user", "content": "json"}],
            "temperature": 0.0,
            "max_tokens": 8,
            "response_format": {"type": "json_object"}
        }"#,
    );
    assert_eq!(code, 200, "{body}");
    // With a grammar active the completion must be valid JSON ({...}).
    let j = deepinfer_core::json::Json::parse(&body).unwrap();
    let content = j
        .get("choices")
        .and_then(|c| c.as_arr())
        .and_then(|c| c.first())
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap()
        .to_string();
    assert!(
        deepinfer_core::json::Json::parse(&content).is_ok(),
        "grammar output must be valid JSON, got {content:?}"
    );
    ts.stop();
}

#[test]
fn graceful_shutdown_drains() {
    let mut ts = TestServer::start();
    // A short request in flight, then shutdown.
    let ts_addr = ("127.0.0.1", ts.port);
    let t = std::thread::spawn(move || {
        let mut s = TcpStream::connect(ts_addr).expect("connect");
        let body = r#"{"messages": [{"role": "user", "content": "drain"}], "temperature": 0.0, "max_tokens": 3}"#;
        let raw = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        s.write_all(raw.as_bytes()).unwrap();
        s.flush().unwrap();
        let resp = read_response(&mut s);
        parse_status_and_body(&resp).0
    });
    // Give the request a moment to be admitted, then drain.
    std::thread::sleep(Duration::from_millis(200));
    let (code, _) = post_json(&ts, "/admin/shutdown", "{}");
    assert_eq!(code, 200);
    // The in-flight request must still complete.
    let final_code = t.join().unwrap();
    assert_eq!(final_code, 200);
    ts.stop();
}

#[test]
fn config_file_round_trip() {
    let dir = std::env::temp_dir().join("deepinfer-server-test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("cfg.json");
    std::fs::write(
        &path,
        r#"{"port": 9311, "model_name": "cfg-model", "max_tokens_cap": 32}"#,
    )
    .unwrap();
    let cfg = ServerConfig::from_file(Path::new(&path)).unwrap();
    assert_eq!(cfg.port, 9311);
    assert_eq!(cfg.model_name, "cfg-model");
    assert_eq!(cfg.max_tokens_cap, 32);
}
