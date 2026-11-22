//! Server metrics: lock-free counters + latency histograms, rendered in
//! the Prometheus text exposition format.
//!
//! Everything is an `AtomicU64` behind an `Arc`, so the engine thread,
//! connection workers and the accept loop all update without locks.
//! Latencies accumulate in nanoseconds (`u64` wraps after ~584 years of
//! summed time — irrelevant).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// One labelled counter.
#[derive(Debug, Default)]
pub struct Counter(AtomicU64);

impl Counter {
    /// Increments by `n`.
    pub fn add(&self, n: u64) {
        self.0.fetch_add(n, Ordering::Relaxed);
    }
    /// Decrements by `n` (saturating at zero).
    pub fn sub(&self, n: u64) {
        self.0.fetch_sub(n.min(self.get()), Ordering::Relaxed);
    }
    /// Sets the absolute value (gauge-style counters).
    pub fn set(&self, v: u64) {
        self.0.store(v, Ordering::Relaxed);
    }
    /// Current value.
    pub fn get(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

/// Fixed-bucket histogram (Prometheus-style `_bucket`/`_sum`/`_count`).
pub struct Histogram {
    /// Upper bounds (seconds, ascending, last is `+Inf` implicitly).
    buckets: Vec<f64>,
    counts: Vec<AtomicU64>,
    sum_nanos: AtomicU64,
    count: AtomicU64,
}

impl Histogram {
    /// The standard latency layout: 5ms .. 30s, 16 buckets.
    fn latency() -> Histogram {
        Histogram::with_buckets(&[
            0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0,
        ])
    }

    /// A histogram with explicit bucket upper bounds (seconds).
    pub fn with_buckets(bounds: &[f64]) -> Histogram {
        Histogram {
            buckets: bounds.to_vec(),
            counts: (0..bounds.len()).map(|_| AtomicU64::new(0)).collect(),
            sum_nanos: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    /// Records one observation (seconds).
    pub fn observe(&self, secs: f64) {
        let nanos = (secs.max(0.0) * 1e9) as u64;
        self.sum_nanos.fetch_add(nanos, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        for (i, b) in self.buckets.iter().enumerate() {
            if secs <= *b {
                self.counts[i].fetch_add(1, Ordering::Relaxed);
                break;
            }
        }
    }

    /// `(bucket_upper_bound, cumulative_count)` pairs plus the sum/count.
    pub fn snapshot(&self) -> (Vec<(f64, u64)>, f64, u64) {
        let mut cumulative = 0u64;
        let mut out = Vec::with_capacity(self.buckets.len() + 1);
        for (i, b) in self.buckets.iter().enumerate() {
            cumulative += self.counts[i].load(Ordering::Relaxed);
            out.push((*b, cumulative));
        }
        out.push((f64::INFINITY, self.count.load(Ordering::Relaxed)));
        let sum = self.sum_nanos.load(Ordering::Relaxed) as f64 / 1e9;
        (out, sum, self.count.load(Ordering::Relaxed))
    }
}

/// The server's shared metric set.
pub struct Metrics {
    /// Wall-clock seconds since server start (nanoseconds).
    start: std::time::Instant,
    /// Total requests handled per route label.
    pub requests: [Counter; 6],
    /// Failed (4xx/5xx) requests per route label.
    pub failures: [Counter; 6],
    /// Generated completion tokens.
    pub completion_tokens: Counter,
    /// Prefilled prompt tokens.
    pub prompt_tokens: Counter,
    /// Currently streaming connections.
    pub active_streams: Counter,
    /// Requests currently inside the scheduler.
    pub engine_active: Counter,
    /// Requests queued (not yet admitted).
    pub engine_queued: Counter,
    /// Scheduler steps executed.
    pub engine_steps: Counter,
    /// Engine hard errors (step failures).
    pub engine_errors: Counter,
    /// Aborted requests (client disconnect / deadline).
    pub aborted_requests: Counter,
    /// DSpark speculative rounds executed (serve path).
    pub spec_rounds: Counter,
    /// Draft tokens proposed by the MTP head (serve path).
    pub spec_draft_tokens: Counter,
    /// Draft tokens accepted by target verification (serve path).
    pub spec_accepted_tokens: Counter,
    /// Bonus tokens emitted from the verification row (serve path).
    pub spec_bonus_tokens: Counter,
    /// Spec rounds that needed a cache rollback (serve path).
    pub spec_rollbacks: Counter,
    /// Tokens rolled back out of the caches on rejection (serve path).
    pub spec_rollback_tokens: Counter,
    /// Prefix-cache admissions that reused cached state.
    pub prefix_cache_hits: Counter,
    /// Prompt tokens served from cached prefix state instead of a
    /// fresh prefill.
    pub prefix_cache_reused_tokens: Counter,
    /// Prompt snapshots inserted into the prefix cache.
    pub prefix_cache_inserts: Counter,
    /// Live prefix-cache entries (gauge).
    pub prefix_cache_entries: Counter,
    /// Tokens covered by live prefix-cache snapshots (gauge).
    pub prefix_cache_tokens_gauge: Counter,
    /// Prefix-cache entries evicted (LRU or allocation pressure).
    pub prefix_cache_evictions: Counter,
    /// Time to first token.
    pub ttft: Histogram,
    /// Whole-request latency.
    pub request_latency: Histogram,
    /// Accepted connections.
    pub connections: Counter,
    /// Rejected connections (over `max_connections`).
    pub connections_rejected: Counter,
}

pub(crate) const ROUTE_LABELS: [&str; 6] = [
    "/health",
    "/v1/models",
    "/v1/chat/completions",
    "/v1/completions",
    "/metrics",
    "other",
];

impl Default for Metrics {
    fn default() -> Metrics {
        Metrics {
            start: std::time::Instant::now(),
            requests: Default::default(),
            failures: Default::default(),
            completion_tokens: Counter::default(),
            prompt_tokens: Counter::default(),
            active_streams: Counter::default(),
            engine_active: Counter::default(),
            engine_queued: Counter::default(),
            engine_steps: Counter::default(),
            engine_errors: Counter::default(),
            aborted_requests: Counter::default(),
            spec_rounds: Counter::default(),
            spec_draft_tokens: Counter::default(),
            spec_accepted_tokens: Counter::default(),
            spec_bonus_tokens: Counter::default(),
            spec_rollbacks: Counter::default(),
            spec_rollback_tokens: Counter::default(),
            prefix_cache_hits: Counter::default(),
            prefix_cache_reused_tokens: Counter::default(),
            prefix_cache_inserts: Counter::default(),
            prefix_cache_entries: Counter::default(),
            prefix_cache_tokens_gauge: Counter::default(),
            prefix_cache_evictions: Counter::default(),
            ttft: Histogram::latency(),
            request_latency: Histogram::latency(),
            connections: Counter::default(),
            connections_rejected: Counter::default(),
        }
    }
}

impl Metrics {
    /// Route label index for a request path.
    pub fn route_index(path: &str) -> usize {
        ROUTE_LABELS
            .iter()
            .position(|r| *r == path)
            .unwrap_or(ROUTE_LABELS.len() - 1)
    }

    /// Seconds since server start.
    pub fn uptime_secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }
}

/// Unix timestamp in seconds (`created` fields of API responses).
pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Renders the whole metric set in Prometheus text format (UTF-8).
pub fn render_prometheus(m: &Metrics, model_name: &str, engine: &str) -> String {
    let mut out = String::with_capacity(4096);

    let info = format!(
        "# HELP deepinfer_build_info Build metadata.\n# TYPE deepinfer_build_info gauge\ndeepinfer_build_info{{engine=\"{engine}\",model=\"{model_name}\"}} 1\n"
    );
    out.push_str(&info);

    let up = format!(
        "# HELP deepinfer_uptime_seconds Seconds since server start.\n# TYPE deepinfer_uptime_seconds gauge\ndeepinfer_uptime_seconds {}\n",
        fmt_f64(m.uptime_secs())
    );
    out.push_str(&up);

    out.push_str(
        "# HELP deepinfer_requests_total Total HTTP requests handled.\n# TYPE deepinfer_requests_total counter\n",
    );
    for (i, label) in ROUTE_LABELS.iter().enumerate() {
        out.push_str(&format!(
            "deepinfer_requests_total{{route=\"{label}\"}} {}\n",
            m.requests[i].get()
        ));
    }

    out.push_str(
        "# HELP deepinfer_requests_failed_total Requests that returned 4xx/5xx.\n# TYPE deepinfer_requests_failed_total counter\n",
    );
    for (i, label) in ROUTE_LABELS.iter().enumerate() {
        out.push_str(&format!(
            "deepinfer_requests_failed_total{{route=\"{label}\"}} {}\n",
            m.failures[i].get()
        ));
    }

    for (name, help, c) in [
        (
            "deepinfer_prompt_tokens_total",
            "Prompt tokens prefilled.",
            &m.prompt_tokens,
        ),
        (
            "deepinfer_completion_tokens_total",
            "Completion tokens generated.",
            &m.completion_tokens,
        ),
        (
            "deepinfer_active_streams",
            "Currently streaming responses.",
            &m.active_streams,
        ),
        (
            "deepinfer_engine_active",
            "Requests inside the scheduler.",
            &m.engine_active,
        ),
        (
            "deepinfer_engine_queued",
            "Requests waiting for admission.",
            &m.engine_queued,
        ),
        (
            "deepinfer_engine_steps_total",
            "Scheduler steps executed.",
            &m.engine_steps,
        ),
        (
            "deepinfer_engine_errors_total",
            "Engine step failures.",
            &m.engine_errors,
        ),
        (
            "deepinfer_aborted_requests_total",
            "Requests aborted (disconnect/deadline).",
            &m.aborted_requests,
        ),
        (
            "deepinfer_spec_rounds_total",
            "DSpark speculative rounds executed.",
            &m.spec_rounds,
        ),
        (
            "deepinfer_spec_draft_tokens_total",
            "Draft tokens proposed by the MTP head.",
            &m.spec_draft_tokens,
        ),
        (
            "deepinfer_spec_accepted_tokens_total",
            "Draft tokens accepted by verification.",
            &m.spec_accepted_tokens,
        ),
        (
            "deepinfer_spec_bonus_tokens_total",
            "Bonus tokens emitted from the verification row.",
            &m.spec_bonus_tokens,
        ),
        (
            "deepinfer_spec_rollbacks_total",
            "Spec rounds that rolled the caches back.",
            &m.spec_rollbacks,
        ),
        (
            "deepinfer_spec_rollback_tokens_total",
            "Tokens rolled back on draft rejection.",
            &m.spec_rollback_tokens,
        ),
        (
            "deepinfer_prefix_cache_hits_total",
            "Prefix-cache admissions that reused cached state.",
            &m.prefix_cache_hits,
        ),
        (
            "deepinfer_prefix_cache_reused_tokens_total",
            "Prompt tokens served from cached prefixes.",
            &m.prefix_cache_reused_tokens,
        ),
        (
            "deepinfer_prefix_cache_inserts_total",
            "Prompt snapshots inserted into the prefix cache.",
            &m.prefix_cache_inserts,
        ),
        (
            "deepinfer_prefix_cache_entries",
            "Live prefix-cache entries.",
            &m.prefix_cache_entries,
        ),
        (
            "deepinfer_prefix_cache_tokens",
            "Tokens covered by live prefix-cache snapshots.",
            &m.prefix_cache_tokens_gauge,
        ),
        (
            "deepinfer_prefix_cache_evictions_total",
            "Prefix-cache entries evicted (LRU or pressure).",
            &m.prefix_cache_evictions,
        ),
        (
            "deepinfer_connections_total",
            "Accepted TCP connections.",
            &m.connections,
        ),
        (
            "deepinfer_connections_rejected_total",
            "Connections rejected over the cap.",
            &m.connections_rejected,
        ),
    ] {
        out.push_str(&format!(
            "# HELP {name} {help}\n# TYPE {name} {}\n",
            if name.ends_with("_total")
                || name.ends_with("_steps_total")
                || name.ends_with("connections_total")
            {
                "counter"
            } else {
                "gauge"
            }
        ));
        out.push_str(&format!("{name} {}\n", c.get()));
    }

    for (name, help, h) in [
        (
            "deepinfer_ttft_seconds",
            "Time to first generated token.",
            &m.ttft,
        ),
        (
            "deepinfer_request_latency_seconds",
            "Whole-request latency.",
            &m.request_latency,
        ),
    ] {
        let (buckets, sum, count) = h.snapshot();
        out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} histogram\n"));
        for (b, c) in &buckets {
            let le = if b.is_infinite() {
                "+Inf".to_string()
            } else {
                fmt_f64(*b)
            };
            out.push_str(&format!("{name}_bucket{{le=\"{le}\"}} {c}\n"));
        }
        out.push_str(&format!("{name}_sum {}\n", fmt_f64(sum)));
        out.push_str(&format!("{name}_count {count}\n"));
    }

    out
}

/// Prometheus float rendering (never scientific notation for our range).
fn fmt_f64(v: f64) -> String {
    if v.is_infinite() {
        return "+Inf".to_string();
    }
    if v == v.trunc() && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{v:.6}")
    }
}

/// Shared handle.
pub type SharedMetrics = Arc<Metrics>;

/// Creates the shared metric set.
pub fn shared() -> SharedMetrics {
    Arc::new(Metrics::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_buckets_cumulate() {
        let h = Histogram::with_buckets(&[0.1, 1.0]);
        h.observe(0.05);
        h.observe(0.5);
        h.observe(5.0);
        let (b, sum, count) = h.snapshot();
        assert_eq!(b[0], (0.1, 1));
        assert_eq!(b[1], (1.0, 2));
        assert_eq!(b[2].0, f64::INFINITY);
        assert_eq!(b[2].1, 3);
        assert_eq!(count, 3);
        assert!(sum > 5.0);
    }

    #[test]
    fn prometheus_render_contains_core_series() {
        let m = shared();
        m.requests[Metrics::route_index("/v1/chat/completions")].add(3);
        m.completion_tokens.add(7);
        let text = render_prometheus(&m, "m", "cpu");
        assert!(text.contains("deepinfer_requests_total{route=\"/v1/chat/completions\"} 3"));
        assert!(text.contains("deepinfer_completion_tokens_total 7"));
        assert!(text.contains("deepinfer_ttft_seconds_bucket{le=\"+Inf\"} 0"));
        assert!(text.contains("deepinfer_build_info{engine=\"cpu\",model=\"m\"} 1"));
    }

    #[test]
    fn route_index_maps() {
        assert_eq!(Metrics::route_index("/health"), 0);
        assert_eq!(Metrics::route_index("/v1/completions"), 3);
        assert_eq!(Metrics::route_index("/nope"), 5);
    }
}
