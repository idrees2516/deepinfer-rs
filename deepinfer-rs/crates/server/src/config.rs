//! Server configuration: JSON file, `DEEPINFER_SERVER_*` environment
//! overrides, and programmatic defaults — plus validation so a bad
//! config fails at startup, not mid-request.

use deepinfer_core::json::Json;
use deepinfer_core::{ensure, Error, Result};

/// Server-side limits and addresses.
#[derive(Debug, Clone, PartialEq)]
pub struct ServerConfig {
    /// Bind host.
    pub host: String,
    /// Bind port.
    pub port: u16,
    /// Idle keep-alive read timeout per connection.
    pub keepalive_idle_secs: u64,
    /// Write timeout per connection.
    pub write_timeout_secs: u64,
    /// Maximum concurrent TCP connections (surplus gets `503`).
    pub max_connections: usize,
    /// Maximum request body bytes (JSON payload cap).
    pub max_body_bytes: usize,
    /// Hard per-request wall-clock cap (stream + non-stream).
    pub request_timeout_secs: u64,
    /// Cap applied to the client's `max_tokens` (OpenAI semantics:
    /// absent means the cap).
    pub max_tokens_cap: usize,
    /// Optional bearer token; when set, `/v1/*` requires it.
    pub api_key: Option<String>,
    /// Model name reported by the API.
    pub model_name: String,
    /// CORS `Access-Control-Allow-Origin` (default `*`).
    pub allow_origin: String,
    /// Scheduler knobs (continuous batching).
    pub scheduler: deepinfer_scheduler::SchedulerConfig,
    /// DSpark speculation depth for the serve path: draft tokens
    /// verified per step (0 disables speculative decoding; the default
    /// matches `DEEPINFER_USE_NEXTN_DECODE`). Ignored when the engine
    /// has no MTP weights.
    pub spec_draft_len: usize,
    /// Adapt each request's draft depth from its observed acceptance
    /// rate (the vLLM dynamic-speculative-decoding controller; see
    /// [`crate::engine::ServeTuning::spec_adaptive`]).
    pub spec_adaptive: bool,
    /// Prefix-cache token budget (SGLang-RadixAttention-style prompt
    /// prefix reuse): the maximum number of prompt tokens covered by
    /// live snapshots; 0 disables the feature.
    pub prefix_cache_tokens: usize,
    /// Maximum prompt tokens per request (token-budget admission
    /// control): requests whose prompt encodes longer are rejected
    /// with a `400` before they reach the scheduler.
    pub max_prompt_tokens: usize,
}

impl Default for ServerConfig {
    fn default() -> ServerConfig {
        ServerConfig {
            host: "0.0.0.0".to_string(),
            port: 8080,
            keepalive_idle_secs: 75,
            write_timeout_secs: 600,
            max_connections: 256,
            max_body_bytes: 4 * 1024 * 1024,
            request_timeout_secs: 300,
            max_tokens_cap: 4096,
            api_key: None,
            model_name: "deepinfer-rs".to_string(),
            allow_origin: "*".to_string(),
            scheduler: deepinfer_scheduler::SchedulerConfig {
                max_running_requests: 64,
                max_batch_tokens: 8192,
                chunked_prefill_size: 2048,
                batch_invariant: false,
            },
            spec_draft_len: 5,
            spec_adaptive: true,
            prefix_cache_tokens: 4096,
            max_prompt_tokens: 32768,
        }
    }
}

impl ServerConfig {
    /// Parses a config from a `deepinfer-server.json` document. Missing
    /// fields keep the defaults; unknown fields are an error (typo
    /// protection for production configs).
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] on malformed JSON or unknown keys.
    pub fn from_json(json: &Json) -> Result<ServerConfig> {
        let mut cfg = ServerConfig::default();
        let Json::Obj(pairs) = json else {
            return Err(Error::invalid("server config must be a JSON object"));
        };
        for (key, val) in pairs {
            match key.as_str() {
                "host" => cfg.host = str_val(val, "host")?.to_string(),
                "port" => cfg.port = u16::try_from(u64_val(val, "port")?).map_err(bad)?,
                "keepalive_idle_secs" => {
                    cfg.keepalive_idle_secs = u64_val(val, "keepalive_idle_secs")?
                }
                "write_timeout_secs" => {
                    cfg.write_timeout_secs = u64_val(val, "write_timeout_secs")?
                }
                "max_connections" => {
                    cfg.max_connections =
                        usize::try_from(u64_val(val, "max_connections")?).map_err(bad)?
                }
                "max_body_bytes" => {
                    cfg.max_body_bytes =
                        usize::try_from(u64_val(val, "max_body_bytes")?).map_err(bad)?
                }
                "request_timeout_secs" => {
                    cfg.request_timeout_secs = u64_val(val, "request_timeout_secs")?
                }
                "max_tokens_cap" => {
                    cfg.max_tokens_cap =
                        usize::try_from(u64_val(val, "max_tokens_cap")?).map_err(bad)?
                }
                "api_key" => cfg.api_key = Some(str_val(val, "api_key")?.to_string()),
                "model_name" => cfg.model_name = str_val(val, "model_name")?.to_string(),
                "allow_origin" => cfg.allow_origin = str_val(val, "allow_origin")?.to_string(),
                "spec_draft_len" => {
                    cfg.spec_draft_len =
                        usize::try_from(u64_val(val, "spec_draft_len")?).map_err(bad)?
                }
                "spec_adaptive" => cfg.spec_adaptive = bool_val(val, "spec_adaptive")?,
                "prefix_cache_tokens" => {
                    cfg.prefix_cache_tokens =
                        usize::try_from(u64_val(val, "prefix_cache_tokens")?).map_err(bad)?
                }
                "max_prompt_tokens" => {
                    cfg.max_prompt_tokens =
                        usize::try_from(u64_val(val, "max_prompt_tokens")?).map_err(bad)?
                }
                "scheduler" => {
                    let Json::Obj(spairs) = val else {
                        return Err(Error::invalid("scheduler must be an object"));
                    };
                    for (sk, sv) in spairs {
                        match sk.as_str() {
                            "max_running_requests" => {
                                cfg.scheduler.max_running_requests =
                                    usize::try_from(u64_val(sv, "scheduler.max_running_requests")?)
                                        .map_err(bad)?
                            }
                            "max_batch_tokens" => {
                                cfg.scheduler.max_batch_tokens =
                                    usize::try_from(u64_val(sv, "scheduler.max_batch_tokens")?)
                                        .map_err(bad)?
                            }
                            "chunked_prefill_size" => {
                                cfg.scheduler.chunked_prefill_size =
                                    usize::try_from(u64_val(sv, "scheduler.chunked_prefill_size")?)
                                        .map_err(bad)?
                            }
                            other => {
                                return Err(Error::invalid(format!(
                                    "unknown scheduler key {other:?}"
                                )))
                            }
                        }
                    }
                }
                other => {
                    return Err(Error::invalid(format!(
                        "unknown server config key {other:?}"
                    )))
                }
            }
        }
        cfg.validate()?;
        Ok(cfg)
    }

    /// Loads `path` and parses it ([`ServerConfig::from_json`]).
    ///
    /// # Errors
    /// IO / JSON / validation errors.
    pub fn from_file(path: &std::path::Path) -> Result<ServerConfig> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| Error::Io(format!("cannot read {}: {e}", path.display())))?;
        ServerConfig::from_json(&Json::parse(&text)?)
    }

    /// Applies `DEEPINFER_SERVER_*` environment overrides on top of
    /// `self`: `HOST`, `PORT`, `API_KEY`, `MODEL_NAME`, `MAX_TOKENS_CAP`,
    /// `REQUEST_TIMEOUT_SECS`, `MAX_CONNECTIONS`, `MAX_BODY_BYTES`,
    /// `SCHED_MAX_RUNNING_REQUESTS`, `SCHED_MAX_BATCH_TOKENS`,
    /// `SCHED_CHUNKED_PREFILL_SIZE`, `SPEC_DRAFT_LEN` (0 disables
    /// speculative serving).
    pub fn apply_env(mut self) -> ServerConfig {
        let get = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        if let Some(v) = get("DEEPINFER_SERVER_HOST") {
            self.host = v;
        }
        if let Some(v) = get("DEEPINFER_SERVER_PORT") {
            if let Ok(p) = v.parse::<u16>() {
                self.port = p;
            }
        }
        if let Some(v) = get("DEEPINFER_SERVER_API_KEY") {
            self.api_key = Some(v);
        }
        if let Some(v) = get("DEEPINFER_SERVER_MODEL_NAME") {
            self.model_name = v;
        }
        if let Some(v) = get("DEEPINFER_SERVER_MAX_TOKENS_CAP") {
            if let Ok(n) = v.parse::<usize>() {
                self.max_tokens_cap = n;
            }
        }
        if let Some(v) = get("DEEPINFER_SERVER_REQUEST_TIMEOUT_SECS") {
            if let Ok(n) = v.parse::<u64>() {
                self.request_timeout_secs = n;
            }
        }
        if let Some(v) = get("DEEPINFER_SERVER_MAX_CONNECTIONS") {
            if let Ok(n) = v.parse::<usize>() {
                self.max_connections = n;
            }
        }
        if let Some(v) = get("DEEPINFER_SERVER_MAX_BODY_BYTES") {
            if let Ok(n) = v.parse::<usize>() {
                self.max_body_bytes = n;
            }
        }
        if let Some(v) = get("DEEPINFER_SERVER_SCHED_MAX_RUNNING_REQUESTS") {
            if let Ok(n) = v.parse::<usize>() {
                self.scheduler.max_running_requests = n;
            }
        }
        if let Some(v) = get("DEEPINFER_SERVER_SCHED_MAX_BATCH_TOKENS") {
            if let Ok(n) = v.parse::<usize>() {
                self.scheduler.max_batch_tokens = n;
            }
        }
        if let Some(v) = get("DEEPINFER_SERVER_SCHED_CHUNKED_PREFILL_SIZE") {
            if let Ok(n) = v.parse::<usize>() {
                self.scheduler.chunked_prefill_size = n;
            }
        }
        if let Some(v) = get("DEEPINFER_SERVER_SPEC_DRAFT_LEN") {
            if let Ok(n) = v.parse::<usize>() {
                self.spec_draft_len = n;
            }
        }
        if let Some(v) = get("DEEPINFER_SERVER_SPEC_ADAPTIVE") {
            if let Ok(b) = v.parse::<bool>() {
                self.spec_adaptive = b;
            }
        }
        if let Some(v) = get("DEEPINFER_SERVER_PREFIX_CACHE_TOKENS") {
            if let Ok(n) = v.parse::<usize>() {
                self.prefix_cache_tokens = n;
            }
        }
        if let Some(v) = get("DEEPINFER_SERVER_MAX_PROMPT_TOKENS") {
            if let Ok(n) = v.parse::<usize>() {
                self.max_prompt_tokens = n;
            }
        }
        self
    }

    /// Validates ranges (startup-time failure, not per-request).
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] listing the first violated bound.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.host.is_empty(),
            InvalidArgument,
            "host must not be empty"
        );
        ensure!(
            self.keepalive_idle_secs >= 1,
            InvalidArgument,
            "keepalive_idle_secs must be >= 1"
        );
        ensure!(
            self.write_timeout_secs >= 1,
            InvalidArgument,
            "write_timeout_secs must be >= 1"
        );
        ensure!(
            self.max_connections >= 1,
            InvalidArgument,
            "max_connections must be >= 1"
        );
        ensure!(
            self.max_body_bytes >= 1024,
            InvalidArgument,
            "max_body_bytes must be >= 1024"
        );
        ensure!(
            self.request_timeout_secs >= 1,
            InvalidArgument,
            "request_timeout_secs must be >= 1"
        );
        ensure!(
            self.max_tokens_cap >= 1,
            InvalidArgument,
            "max_tokens_cap must be >= 1"
        );
        ensure!(
            self.max_prompt_tokens >= 1,
            InvalidArgument,
            "max_prompt_tokens must be >= 1"
        );
        ensure!(
            self.prefix_cache_tokens <= 1 << 26,
            InvalidArgument,
            "prefix_cache_tokens must be <= 67108864 (64M tokens)"
        );
        ensure!(
            !self.model_name.is_empty(),
            InvalidArgument,
            "model_name must not be empty"
        );
        self.scheduler.validate()?;
        if let Some(key) = &self.api_key {
            ensure!(
                key.len() >= 8,
                InvalidArgument,
                "api_key must be at least 8 characters"
            );
        }
        Ok(())
    }

    /// The bind socket address `host:port`.
    pub fn bind_addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

fn str_val<'a>(v: &'a Json, name: &str) -> Result<&'a str> {
    v.as_str()
        .ok_or_else(|| Error::invalid(format!("{name} must be a string")))
}

fn bool_val(v: &Json, name: &str) -> Result<bool> {
    v.as_bool()
        .ok_or_else(|| Error::invalid(format!("{name} must be a boolean")))
}

fn u64_val(v: &Json, name: &str) -> Result<u64> {
    v.as_u64()
        .ok_or_else(|| Error::invalid(format!("{name} must be a non-negative integer")))
}

fn bad(e: std::num::TryFromIntError) -> Error {
    Error::invalid(format!("value out of range: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_validates() {
        ServerConfig::default().validate().unwrap();
    }

    #[test]
    fn json_round_trip_and_unknown_keys() {
        let cfg = ServerConfig::from_json(
            &Json::parse(r#"{"port": 9200, "api_key": "sekret-key", "scheduler": {"max_running_requests": 8}}"#).unwrap(),
        )
        .unwrap();
        assert_eq!(cfg.port, 9200);
        assert_eq!(cfg.api_key.as_deref(), Some("sekret-key"));
        assert_eq!(cfg.scheduler.max_running_requests, 8);
        assert!(ServerConfig::from_json(&Json::parse(r#"{"port": "x"}"#).unwrap()).is_err());
        assert!(ServerConfig::from_json(&Json::parse(r#"{"nope": 1}"#).unwrap()).is_err());
        // short api key rejected
        assert!(ServerConfig::from_json(&Json::parse(r#"{"api_key": "abc"}"#).unwrap()).is_err());
    }

    #[test]
    fn bind_addr_formats() {
        let cfg = ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 8088,
            ..ServerConfig::default()
        };
        assert_eq!(cfg.bind_addr(), "127.0.0.1:8088");
    }
}
