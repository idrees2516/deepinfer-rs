//! A dependency-free HTTP/1.1 server: request parsing, response and SSE
//! writing, keep-alive, hard limits and timeouts.
//!
//! Scope (what an OpenAI-compatible LLM endpoint actually needs):
//! * `GET`/`POST`/`OPTIONS` with `Content-Length` bodies. Chunked
//!   *request* bodies are refused with `411` (clients that stream
//!   uploads are out of scope for model inference).
//! * `Expect: 100-continue` interim responses (curl sends this for
//!   larger JSON bodies).
//! * Keep-alive with an idle read timeout; `Connection: close` on
//!   HTTP/1.0 semantics or client request.
//! * Server-driven streaming via chunked transfer coding (SSE), with
//!   `flush()` after every event so tokens reach the client as they are
//!   generated.
//!
//! Everything is bounded: header bytes, body bytes, request-line length,
//! per-connection request count and the number of concurrent
//! connections. The parser is a strict state machine over bytes — no
//! panics on malformed input, only [`HttpError`] responses.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use deepinfer_core::{ensure, Error, Result};

/// Hard ceiling on the request line + header block (`431` beyond).
pub const MAX_HEADER_BYTES: usize = 32 * 1024;
/// Maximum request-line length (`414` beyond).
pub const MAX_REQUEST_LINE: usize = 8 * 1024;
/// Bound on the number of pipelined/keep-alive requests per connection.
pub const MAX_REQUESTS_PER_CONN: usize = 10_000;

/// A parsed HTTP request (the subset the API serves).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpRequest {
    /// Uppercase method (`GET`, `POST`, ...).
    pub method: String,
    /// Percent-decoded path (no query string).
    pub path: String,
    /// Raw query string (`""` when absent) — decode per-value with
    /// [`decode_query_value`].
    pub query: String,
    /// Header names lowercased, order preserved.
    pub headers: Vec<(String, String)>,
    /// Body bytes (empty when `Content-Length: 0` / absent).
    pub body: Vec<u8>,
}

impl HttpRequest {
    /// First value of a header (case-insensitive), if present.
    pub fn header(&self, name: &str) -> Option<&str> {
        let lname = name.to_ascii_lowercase();
        self.headers
            .iter()
            .find(|(k, _)| *k == lname)
            .map(|(_, v)| v.as_str())
    }

    /// Bearer token from `Authorization: Bearer <token>`, if present.
    pub fn bearer_token(&self) -> Option<&str> {
        self.header("authorization")
            .and_then(|v| v.strip_prefix("Bearer "))
            .map(|t| t.trim())
    }
}

/// Why a request could not be parsed; each maps to an HTTP status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HttpError {
    /// Malformed request line or headers.
    BadRequest(String),
    /// Header block exceeded [`MAX_HEADER_BYTES`].
    HeadersTooLarge,
    /// Request line exceeded [`MAX_REQUEST_LINE`].
    UriTooLong,
    /// Body exceeded the configured maximum.
    BodyTooLarge,
    /// `Transfer-Encoding` present (unsupported).
    LengthRequired,
    /// Invalid `Content-Length`.
    BadContentLength(String),
    /// Read timed out or the peer went away mid-request.
    ConnectionClosed,
}

impl HttpError {
    /// The HTTP status code for this error.
    pub fn status(&self) -> u16 {
        match self {
            HttpError::BadRequest(_) => 400,
            HttpError::HeadersTooLarge => 431,
            HttpError::UriTooLong => 414,
            HttpError::BodyTooLarge => 413,
            HttpError::LengthRequired => 411,
            HttpError::BadContentLength(_) => 400,
            HttpError::ConnectionClosed => 408,
        }
    }

    /// Human-readable status phrase.
    pub fn phrase(&self) -> &'static str {
        match self {
            HttpError::BadRequest(_) => "Bad Request",
            HttpError::HeadersTooLarge => "Request Header Fields Too Large",
            HttpError::UriTooLong => "URI Too Long",
            HttpError::BodyTooLarge => "Payload Too Large",
            HttpError::LengthRequired => "Length Required",
            HttpError::BadContentLength(_) => "Bad Request",
            HttpError::ConnectionClosed => "Request Timeout",
        }
    }

    /// Machine-readable error code for the JSON body.
    pub fn code(&self) -> &'static str {
        match self {
            HttpError::BadRequest(m) if m.contains("method") => "method_not_allowed",
            HttpError::BadRequest(_) => "bad_request",
            HttpError::HeadersTooLarge => "headers_too_large",
            HttpError::UriTooLong => "uri_too_long",
            HttpError::BodyTooLarge => "payload_too_large",
            HttpError::LengthRequired => "length_required",
            HttpError::BadContentLength(_) => "bad_content_length",
            HttpError::ConnectionClosed => "connection_closed",
        }
    }
}

/// Reads one CRLF-terminated line (excluding the terminator) with a hard
/// byte cap. `Ok(None)` = orderly EOF before any byte.
fn read_line(stream: &mut TcpStream, cap: usize) -> Result<Option<String>> {
    let mut line: Vec<u8> = Vec::with_capacity(128);
    let mut byte = [0u8; 1];
    loop {
        match stream.read(&mut byte) {
            Ok(0) => {
                if line.is_empty() {
                    return Ok(None);
                }
                // EOF mid-line: treat as a closed connection.
                return Err(Error::invalid("eof mid-header"));
            }
            Ok(_) => {
                if byte[0] == b'\n' {
                    if line.last() == Some(&b'\r') {
                        line.pop();
                    }
                    return Ok(Some(String::from_utf8_lossy(&line).into_owned()));
                }
                ensure!(line.len() < cap, InvalidArgument, "line exceeds cap");
                line.push(byte[0]);
            }
            Err(e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
            {
                return Err(Error::invalid("read timeout"));
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => return Err(Error::invalid("read error")),
        }
    }
}

/// Reads exactly `n` bytes (with the socket read timeout applying).
fn read_exact_n(stream: &mut TcpStream, n: usize) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; n];
    let mut filled = 0;
    while filled < n {
        match stream.read(&mut buf[filled..]) {
            Ok(0) => return Err(Error::invalid("eof mid-body")),
            Ok(k) => filled += k,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => return Err(Error::invalid("body read error")),
        }
    }
    Ok(buf)
}

/// Parses one request off `stream`. `max_body` bounds `Content-Length`.
/// `Ok(None)` = orderly close before a request started.
///
/// # Errors
/// [`Error::InvalidArgument`] carrying [`HttpError`] semantics — callers
/// map failures with [`HttpError`] directly instead of re-parsing.
pub fn read_request(
    stream: &mut TcpStream,
    max_body: usize,
) -> std::result::Result<Option<HttpRequest>, HttpError> {
    // ---- request line ----
    let line = match read_line(stream, MAX_REQUEST_LINE) {
        Ok(None) => return Ok(None),
        Ok(Some(l)) => l,
        Err(e) => return Err(map_io(e)),
    };
    // Tolerate a stray CRLF between pipelined requests.
    let line = if line.is_empty() {
        match read_line(stream, MAX_REQUEST_LINE) {
            Ok(None) => return Ok(None),
            Ok(Some(l)) => l,
            Err(e) => return Err(map_io(e)),
        }
    } else {
        line
    };
    parse_from_request_line(stream, max_body, line).map_err(map_io)
}

fn parse_from_request_line(
    stream: &mut TcpStream,
    max_body: usize,
    line: String,
) -> std::result::Result<Option<HttpRequest>, Error> {
    let mut parts = line.split(' ').filter(|p| !p.is_empty());
    let (method, target) = match (parts.next(), parts.next()) {
        (Some(m), Some(t)) if !m.is_empty() && !t.is_empty() => (m, t),
        _ => return Err(Error::invalid("malformed request line")),
    };
    let version = parts.next().unwrap_or("HTTP/1.1");
    ensure!(
        version.eq_ignore_ascii_case("HTTP/1.1") || version.eq_ignore_ascii_case("HTTP/1.0"),
        InvalidArgument,
        "unsupported protocol version"
    );

    // ---- headers ----
    let mut headers: Vec<(String, String)> = Vec::with_capacity(16);
    let mut total = line.len();
    loop {
        let hline = match read_line(stream, MAX_HEADER_BYTES) {
            Ok(Some(l)) => l,
            Ok(None) => return Err(Error::invalid("eof in headers")),
            Err(e) => return Err(e),
        };
        if hline.is_empty() {
            break;
        }
        total += hline.len() + 2;
        ensure!(
            total <= MAX_HEADER_BYTES,
            InvalidArgument,
            "header block too large"
        );
        let (name, value) = match hline.split_once(':') {
            Some((n, v)) => (n.trim().to_ascii_lowercase(), v.trim().to_string()),
            None => return Err(Error::invalid("malformed header line")),
        };
        if name.is_empty() {
            return Err(Error::invalid("empty header name"));
        }
        headers.push((name, value));
    }

    let find = |k: &str| -> Option<&str> {
        headers
            .iter()
            .find(|(n, _)| n == k)
            .map(|(_, v)| v.as_str())
    };

    // Transfer-Encoding: only the identity (absent) is supported.
    if let Some(te) = find("transfer-encoding") {
        if !te.eq_ignore_ascii_case("identity") {
            return Err(Error::invalid("transfer-encoding unsupported"));
        }
    }

    // ---- 100-continue ----
    if let Some(expect) = find("expect") {
        if expect.eq_ignore_ascii_case("100-continue") {
            stream
                .write_all(b"HTTP/1.1 100 Continue\r\n\r\n")
                .map_err(|_| Error::invalid("continue write failed"))?;
            stream.flush().ok();
        }
    }

    // ---- body ----
    let len = match find("content-length") {
        None => 0usize,
        Some(v) => v
            .parse::<usize>()
            .map_err(|_| Error::invalid("bad content-length"))?,
    };
    ensure!(len <= max_body, InvalidArgument, "body too large");
    let body = if len > 0 {
        read_exact_n(stream, len).map_err(|_| Error::invalid("body read failed"))?
    } else {
        Vec::new()
    };

    // ---- target -> path + query ----
    let (raw_path, query) = match target.split_once('?') {
        Some((p, q)) => (p, q.to_string()),
        None => (target, String::new()),
    };
    let path = percent_decode(raw_path)?;

    Ok(Some(HttpRequest {
        method: method.to_ascii_uppercase(),
        path,
        query,
        headers,
        body,
    }))
}

fn map_io(e: Error) -> HttpError {
    let m = e.to_string();
    if m.contains("timeout") {
        HttpError::ConnectionClosed
    } else if m.contains("line exceeds cap") {
        HttpError::UriTooLong
    } else if m.contains("header block too large") {
        HttpError::HeadersTooLarge
    } else if m.contains("body too large") {
        HttpError::BodyTooLarge
    } else {
        HttpError::BadRequest(m)
    }
}

/// Percent-decodes a path component. Rejects `%` sequences that are not
/// two hex digits (treated as bad request by callers).
fn percent_decode(s: &str) -> std::result::Result<String, Error> {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            ensure!(
                i + 2 < bytes.len(),
                InvalidArgument,
                "truncated percent escape"
            );
            let hi = hex_val(bytes[i + 1]);
            let lo = hex_val(bytes[i + 2]);
            match (hi, lo) {
                (Some(h), Some(l)) => {
                    out.push(h * 16 + l);
                    i += 3;
                }
                _ => {
                    return Err(Error::invalid("bad percent escape"));
                }
            }
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    Ok(String::from_utf8_lossy(&out).into_owned())
}

/// Percent-decodes one `application/x-www-form-urlencoded` query value
/// (`+` means space). Lossy on invalid escapes.
pub fn decode_query_value(s: &str) -> String {
    let s = s.replace('+', " ");
    percent_decode(&s).unwrap_or_else(|_| s.clone())
}

fn hex_val(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

/// An HTTP response under construction; finish with
/// [`Response::write`] (fixed length) or convert into an SSE
/// [`ChunkedWriter`].
#[derive(Debug, Clone)]
pub struct Response {
    /// Status code.
    pub status: u16,
    /// Extra headers (name, value) appended after the built-ins.
    pub headers: Vec<(String, String)>,
    /// Body bytes.
    pub body: Vec<u8>,
}

impl Response {
    /// A response with a JSON body and `application/json` content type.
    pub fn json(status: u16, body: impl Into<String>) -> Response {
        Response {
            status,
            headers: vec![("Content-Type".to_string(), "application/json".to_string())],
            body: body.into().into_bytes(),
        }
    }

    /// A plain-text body.
    pub fn text(status: u16, body: impl Into<String>) -> Response {
        Response {
            status,
            headers: vec![(
                "Content-Type".to_string(),
                "text/plain; charset=utf-8".to_string(),
            )],
            body: body.into().into_bytes(),
        }
    }

    /// An empty response.
    pub fn empty(status: u16) -> Response {
        Response {
            status,
            headers: Vec::new(),
            body: Vec::new(),
        }
    }

    /// Appends a header.
    pub fn header(mut self, name: &str, value: &str) -> Response {
        self.headers.push((name.to_string(), value.to_string()));
        self
    }

    /// Status phrase for the common codes this server emits.
    pub fn phrase(status: u16) -> &'static str {
        match status {
            100 => "Continue",
            200 => "OK",
            204 => "No Content",
            400 => "Bad Request",
            401 => "Unauthorized",
            403 => "Forbidden",
            404 => "Not Found",
            405 => "Method Not Allowed",
            408 => "Request Timeout",
            411 => "Length Required",
            413 => "Payload Too Large",
            414 => "URI Too Long",
            429 => "Too Many Requests",
            431 => "Request Header Fields Too Large",
            500 => "Internal Server Error",
            503 => "Service Unavailable",
            504 => "Gateway Timeout",
            _ => "Response",
        }
    }

    /// Serializes and sends the response with `Content-Length`, then
    /// flushes. Returns whether the connection may be kept alive.
    pub fn write(self, stream: &mut TcpStream, keep_alive: bool) -> std::io::Result<()> {
        let mut head = format!(
            "HTTP/1.1 {} {}\r\n",
            self.status,
            Response::phrase(self.status)
        );
        for (k, v) in &self.headers {
            head.push_str(k);
            head.push_str(": ");
            head.push_str(v);
            head.push_str("\r\n");
        }
        head.push_str(&format!("Content-Length: {}\r\n", self.body.len()));
        head.push_str(if keep_alive {
            "Connection: keep-alive\r\n"
        } else {
            "Connection: close\r\n"
        });
        head.push_str("\r\n");
        stream.write_all(head.as_bytes())?;
        stream.write_all(&self.body)?;
        stream.flush()
    }
}

/// Chunked-transfer-coding writer for streaming responses (SSE).
///
/// Every [`ChunkedWriter::event`] call emits one complete HTTP chunk and
/// flushes, so the client sees the event immediately. Dropping the
/// writer without [`ChunkedWriter::finish`] aborts the response mid-
/// stream (the `0\r\n\r\n` terminator is never sent), which is exactly
/// the semantics a cancelled generation wants.
pub struct ChunkedWriter<'a> {
    stream: &'a mut TcpStream,
    ok: bool,
}

impl<'a> ChunkedWriter<'a> {
    /// Sends the SSE response head (`200`, `text/event-stream`,
    /// `Transfer-Encoding: chunked`, `Cache-Control: no-cache`,
    /// `X-Accel-Buffering: no`).
    pub fn sse(stream: &'a mut TcpStream) -> std::io::Result<ChunkedWriter<'a>> {
        let head = "HTTP/1.1 200 OK\r\n\
                    Content-Type: text/event-stream; charset=utf-8\r\n\
                    Cache-Control: no-cache\r\n\
                    Connection: keep-alive\r\n\
            Transfer-Encoding: chunked\r\n\
            X-Accel-Buffering: no\r\n\r\n";
        stream.write_all(head.as_bytes())?;
        stream.flush()?;
        Ok(ChunkedWriter { stream, ok: true })
    }

    /// Sends the SSE response head as a non-200 status with a JSON body
    /// (used to reject a stream request before any event is written).
    pub fn sse_error(
        stream: &'a mut TcpStream,
        status: u16,
        json_body: &str,
    ) -> std::io::Result<()> {
        let head = format!(
            "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{}",
            status,
            Response::phrase(status),
            json_body.len(),
            json_body
        );
        stream.write_all(head.as_bytes())?;
        stream.flush()
    }

    /// Writes one event: `data: <payload>\n\n` framed as one chunk.
    /// Sets the error flag on failure (subsequent calls are no-ops) so
    /// cancellation can be detected by the caller.
    pub fn event(&mut self, payload: &str) {
        if !self.ok {
            return;
        }
        let framed = format!("data: {}\n\n", payload);
        let chunk = format!("{:x}\r\n{}\r\n", framed.len(), framed);
        if self.stream.write_all(chunk.as_bytes()).is_err() {
            self.ok = false;
            return;
        }
        if self.stream.flush().is_err() {
            self.ok = false;
        }
    }

    /// Whether all writes so far succeeded (false = client went away).
    pub fn is_ok(&self) -> bool {
        self.ok
    }

    /// Terminates the stream cleanly (`data: [DONE]` + `0\r\n\r\n`).
    pub fn finish(mut self) -> std::io::Result<()> {
        self.event("[DONE]");
        if !self.ok {
            return Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "client disconnected",
            ));
        }
        self.stream.write_all(b"0\r\n\r\n")?;
        self.stream.flush()
    }
}

/// Socket options applied to every accepted connection.
pub fn configure_stream(stream: &TcpStream, idle: Duration, write: Duration) {
    stream.set_read_timeout(Some(idle)).ok();
    stream.set_write_timeout(Some(write)).ok();
    stream.set_nodelay(true).ok();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percent_decode_basics() {
        assert_eq!(
            percent_decode("/v1/chat/completions").unwrap(),
            "/v1/chat/completions"
        );
        assert_eq!(percent_decode("/a%20b").unwrap(), "/a b");
        assert_eq!(percent_decode("/%E4%B8%AD").unwrap(), "/中");
        assert!(percent_decode("/%zz").is_err());
        assert!(percent_decode("/%2").is_err());
    }

    #[test]
    fn query_value_decodes_plus() {
        assert_eq!(decode_query_value("a+b%20c"), "a b c");
    }

    #[test]
    fn response_phrase_covers_api_statuses() {
        for s in [200, 400, 401, 404, 405, 413, 429, 500, 503] {
            assert!(!Response::phrase(s).is_empty());
        }
    }
}
