//! `DEEPINFER_*` environment configuration, mirroring the environment
//! contract of the upstream DeepInfer release:
//!
//! | Variable | Meaning | Default |
//! |---|---|---|
//! | `DEEPINFER_ACCELERATOR` | `nvidia` / `ascend` / `cpu` | `cpu` |
//! | `DEEPINFER_HF_MODEL_DIR` | Local checkpoint directory | HF cache |
//! | `DEEPINFER_ENABLE_MTP_VARLEN` | `1` enables varlen MTP batches | `0` |
//! | `DEEPINFER_USE_NEXTN_DECODE` | Max MTP draft length | `5` |
//! | `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE` | `full` / `compact` | `full` |
//! | `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS` | Bounded-replay suffix length | `140` |
//! | `DEEPINFER_BATCH_INVARIANT` | `1` enables batch-invariant inference | `0` |
//!
//! Parsing is total and panic-free: invalid values fall back to defaults
//! and are reported through [`EnvReport::warnings`] so callers can log them.

use std::path::PathBuf;

/// Accelerator backend selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Accelerator {
    /// NVIDIA GPUs (SM100 class in upstream DeepInfer).
    Nvidia,
    /// Huawei Ascend NPUs (950 class).
    Ascend,
    /// Portable CPU reference (this implementation).
    #[default]
    Cpu,
}

impl Accelerator {
    /// Parses the `DEEPINFER_ACCELERATOR` value.
    pub fn parse(s: &str) -> Option<Accelerator> {
        match s.trim().to_ascii_lowercase().as_str() {
            "nvidia" | "cuda" => Some(Accelerator::Nvidia),
            "ascend" | "npu" => Some(Accelerator::Ascend),
            "cpu" => Some(Accelerator::Cpu),
            _ => None,
        }
    }
}

/// YOCO decoder prefill truncation mode (`DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum YocoTruncationMode {
    /// Process the full prompt through the decoder half.
    #[default]
    Full,
    /// "compact": replay only a bounded suffix (V4.1 default behaviour).
    Compact,
}

impl YocoTruncationMode {
    /// Parses the env value.
    pub fn parse(s: &str) -> Option<YocoTruncationMode> {
        match s.trim().to_ascii_lowercase().as_str() {
            "full" => Some(YocoTruncationMode::Full),
            "compact" => Some(YocoTruncationMode::Compact),
            _ => None,
        }
    }
}

/// Parsed environment configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct Env {
    /// `DEEPINFER_ACCELERATOR`
    pub accelerator: Accelerator,
    /// `DEEPINFER_HF_MODEL_DIR` — local model directory override.
    pub hf_model_dir: Option<PathBuf>,
    /// `DEEPINFER_ENABLE_MTP_VARLEN`
    pub enable_mtp_varlen: bool,
    /// `DEEPINFER_USE_NEXTN_DECODE` (max draft length; V4.1 default 5).
    pub use_nextn_decode: usize,
    /// `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE`
    pub yoco_prefill_truncation_mode: YocoTruncationMode,
    /// `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS` (V4.1 default 140).
    pub yoco_prefill_keep_tokens: usize,
    /// `DEEPINFER_BATCH_INVARIANT`
    pub batch_invariant: bool,
}

impl Default for Env {
    fn default() -> Env {
        Env {
            accelerator: Accelerator::Cpu,
            hf_model_dir: None,
            enable_mtp_varlen: false,
            use_nextn_decode: 5,
            yoco_prefill_truncation_mode: YocoTruncationMode::Full,
            yoco_prefill_keep_tokens: 140,
            batch_invariant: false,
        }
    }
}

/// Result of reading the environment: the config plus any warnings
/// produced by invalid values.
#[derive(Debug, Clone, Default)]
pub struct EnvReport {
    /// Parsed configuration.
    pub env: Env,
    /// Human-readable warnings for invalid values.
    pub warnings: Vec<String>,
}

fn parse_flag(name: &str, raw: &str, warnings: &mut Vec<String>) -> Option<bool> {
    match raw.trim() {
        "1" | "true" | "TRUE" | "yes" | "on" => Some(true),
        "0" | "false" | "FALSE" | "no" | "off" | "" => Some(false),
        other => {
            warnings.push(format!(
                "ignoring invalid boolean for {name}: {other:?} (expected 0/1)"
            ));
            None
        }
    }
}

impl Env {
    /// Reads all `DEEPINFER_*` variables from the process environment.
    pub fn from_env() -> EnvReport {
        Self::from_lookup(|k| std::env::var(k).ok())
    }

    /// Testable variant: values come from `lookup` instead of the real
    /// environment.
    pub fn from_lookup<F>(mut lookup: F) -> EnvReport
    where
        F: FnMut(&str) -> Option<String>,
    {
        let mut env = Env::default();
        let mut warnings = Vec::new();

        if let Some(v) = lookup("DEEPINFER_ACCELERATOR") {
            match Accelerator::parse(&v) {
                Some(a) => env.accelerator = a,
                None => warnings.push(format!(
                    "ignoring invalid DEEPINFER_ACCELERATOR={v:?} (expected nvidia/ascend/cpu)"
                )),
            }
        }
        if let Some(v) = lookup("DEEPINFER_HF_MODEL_DIR") {
            let t = v.trim();
            if !t.is_empty() {
                env.hf_model_dir = Some(PathBuf::from(t));
            }
        }
        if let Some(v) = lookup("DEEPINFER_ENABLE_MTP_VARLEN") {
            if let Some(b) = parse_flag("DEEPINFER_ENABLE_MTP_VARLEN", &v, &mut warnings) {
                env.enable_mtp_varlen = b;
            }
        }
        if let Some(v) = lookup("DEEPINFER_USE_NEXTN_DECODE") {
            match v.trim().parse::<usize>() {
                Ok(n) if n >= 1 && n <= 64 => env.use_nextn_decode = n,
                Ok(n) => warnings.push(format!(
                    "DEEPINFER_USE_NEXTN_DECODE={n} outside [1, 64]; keeping default {}",
                    env.use_nextn_decode
                )),
                Err(_) => warnings.push(format!(
                    "ignoring invalid DEEPINFER_USE_NEXTN_DECODE={v:?} (expected integer)"
                )),
            }
        }
        if let Some(v) = lookup("DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE") {
            match YocoTruncationMode::parse(&v) {
                Some(m) => env.yoco_prefill_truncation_mode = m,
                None => warnings.push(format!(
                    "ignoring invalid DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE={v:?} (expected full/compact)"
                )),
            }
        }
        if let Some(v) = lookup("DEEPINFER_YOCO_PREFILL_KEEP_TOKENS") {
            match v.trim().parse::<usize>() {
                Ok(n) if n >= 1 && n <= 1 << 20 => env.yoco_prefill_keep_tokens = n,
                other => warnings.push(format!(
                    "ignoring invalid DEEPINFER_YOCO_PREFILL_KEEP_TOKENS (got {other:?}); keeping default {}",
                    env.yoco_prefill_keep_tokens
                )),
            }
        }
        if let Some(v) = lookup("DEEPINFER_BATCH_INVARIANT") {
            if let Some(b) = parse_flag("DEEPINFER_BATCH_INVARIANT", &v, &mut warnings) {
                env.batch_invariant = b;
            }
        }

        EnvReport { env, warnings }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lookup<'a>(map: &'a [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |k| {
            map.iter()
                .find(|(key, _)| *key == k)
                .map(|(_, v)| v.to_string())
        }
    }

    #[test]
    fn defaults_when_unset() {
        let r = Env::from_lookup(|_| None);
        assert_eq!(r.env, Env::default());
        assert!(r.warnings.is_empty());
    }

    #[test]
    fn reads_all_vars() {
        let r = Env::from_lookup(lookup(&[
            ("DEEPINFER_ACCELERATOR", "nvidia"),
            ("DEEPINFER_HF_MODEL_DIR", "/models/v4.1"),
            ("DEEPINFER_ENABLE_MTP_VARLEN", "1"),
            ("DEEPINFER_USE_NEXTN_DECODE", "3"),
            ("DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE", "compact"),
            ("DEEPINFER_YOCO_PREFILL_KEEP_TOKENS", "140"),
            ("DEEPINFER_BATCH_INVARIANT", "1"),
        ]));
        assert_eq!(r.env.accelerator, Accelerator::Nvidia);
        assert_eq!(
            r.env.hf_model_dir.as_deref(),
            Some(std::path::Path::new("/models/v4.1"))
        );
        assert!(r.env.enable_mtp_varlen);
        assert_eq!(r.env.use_nextn_decode, 3);
        assert_eq!(r.env.yoco_prefill_truncation_mode, YocoTruncationMode::Compact);
        assert_eq!(r.env.yoco_prefill_keep_tokens, 140);
        assert!(r.env.batch_invariant);
        assert!(r.warnings.is_empty());
    }

    #[test]
    fn invalid_values_warn_not_panic() {
        let r = Env::from_lookup(lookup(&[
            ("DEEPINFER_ACCELERATOR", "gpu"),
            ("DEEPINFER_USE_NEXTN_DECODE", "many"),
            ("DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE", "aggressive"),
            ("DEEPINFER_BATCH_INVARIANT", "maybe"),
            ("DEEPINFER_YOCO_PREFILL_KEEP_TOKENS", "-5"),
        ]));
        assert_eq!(r.env, Env::default());
        assert_eq!(r.warnings.len(), 5, "{:?}", r.warnings);
    }
}
