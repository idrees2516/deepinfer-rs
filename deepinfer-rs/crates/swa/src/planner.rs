//! YOCO decoder layout and SWA bounded-replay planning.
//!
//! YOCO ("You Only Cache Once", Sun et al., arXiv:2405.05254) splits the
//! decoder into self-attention layers (each with its own sliding-window
//! KV) and cross-attention layers that read a single encoder-side KV cache
//! — so prefill of the decoder half only needs a bounded suffix of the
//! prompt. The DeepInfer env contract exposes this as
//! `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE` (`full` / `compact`) and
//! `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS` (V4.1 default 140): in compact
//! mode only the last `keep_tokens` prompt tokens are replayed, and
//! `keep_tokens` must cover the SWA window plus MTP backtracking headroom.

use deepinfer_core::env::{Env, YocoTruncationMode};
use deepinfer_core::{ensure, Error, Result};

/// YOCO decoder layer layout (self- vs cross-attention split).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct YocoLayout {
    /// Number of self-attention decoder layers (each keeps its own SWA KV).
    pub self_layers: usize,
    /// Number of cross-attention decoder layers (reading the shared
    /// encoder-side KV cache).
    pub cross_layers: usize,
}

impl YocoLayout {
    /// Total number of decoder layers (saturating — never panics).
    pub fn total(&self) -> usize {
        self.self_layers.saturating_add(self.cross_layers)
    }
}

/// Sliding-window attention configuration for the YOCO decoder half.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SwaConfig {
    /// SWA window size (number of retained key/value rows).
    pub window: usize,
    /// MTP backtracking headroom (draft tokens that may be rolled back).
    pub max_backtrack: usize,
    /// Retained prompt suffix length in compact mode (V4.1 default 140).
    pub keep_tokens: usize,
    /// Prefill truncation mode (`Full` or `Compact`).
    pub mode: YocoTruncationMode,
}

impl SwaConfig {
    /// Builds the config from the DeepInfer environment contract: the
    /// window and backtracking headroom come from the caller, while
    /// `keep_tokens` and the truncation mode are read from `env`
    /// (`DEEPINFER_YOCO_PREFILL_KEEP_TOKENS` /
    /// `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE`).
    pub fn from_env(env: &Env, window: usize, max_backtrack: usize) -> SwaConfig {
        SwaConfig {
            window,
            max_backtrack,
            keep_tokens: env.yoco_prefill_keep_tokens,
            mode: env.yoco_prefill_truncation_mode,
        }
    }

    /// Validates the configuration.
    ///
    /// * `window >= 1` and `keep_tokens >= 1` always;
    /// * in `Compact` mode, `keep_tokens >= window + max_backtrack` — the
    ///   replayed suffix must cover the SWA window plus MTP backtracking
    ///   headroom (`Err` otherwise).
    ///
    /// `Full` mode imposes no window/keep relation beyond the two lower
    /// bounds.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.window >= 1,
            InvalidArgument,
            "SWA window must be >= 1 (got {})",
            self.window
        );
        ensure!(
            self.keep_tokens >= 1,
            InvalidArgument,
            "keep_tokens must be >= 1 (got {})",
            self.keep_tokens
        );
        if self.mode == YocoTruncationMode::Compact {
            match self.window.checked_add(self.max_backtrack) {
                Some(need) => {
                    ensure!(
                        self.keep_tokens >= need,
                        InvalidArgument,
                        "compact mode requires keep_tokens ({}) >= window + max_backtrack ({})",
                        self.keep_tokens,
                        need
                    );
                }
                None => {
                    return Err(Error::InvalidArgument(format!(
                        "window ({}) + max_backtrack ({}) overflows usize",
                        self.window, self.max_backtrack
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Prefill replay plan: which suffix of the prompt must be processed by
/// the YOCO decoder half.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayPlan {
    /// Full prompt length.
    pub full_len: usize,
    /// Index of the first replayed prompt token (`0` in full mode;
    /// `full_len - keep_tokens` in compact mode).
    pub replay_start: usize,
}

impl ReplayPlan {
    /// Number of tokens the decoder half must process
    /// (`full_len - replay_start`, saturating for hand-built plans).
    pub fn replay_len(&self) -> usize {
        self.full_len.saturating_sub(self.replay_start)
    }
}

/// Bounded-replay planner for YOCO prefill.
#[derive(Debug, Clone)]
pub struct BoundedReplayPlanner {
    config: SwaConfig,
}

impl BoundedReplayPlanner {
    /// Creates a planner after validating `config`
    /// ([`SwaConfig::validate`]).
    pub fn new(config: SwaConfig) -> Result<Self> {
        config.validate()?;
        Ok(BoundedReplayPlanner { config })
    }

    /// Plans the decoder-half prefill of a prompt of `prompt_len` tokens.
    ///
    /// * `Full` mode: replay everything (`replay_start = 0`).
    /// * `Compact` mode: replay only `[len - keep_tokens, len)`
    ///   (`replay_start = len.saturating_sub(keep_tokens)`).
    pub fn plan_prefill(&self, prompt_len: usize) -> ReplayPlan {
        let replay_start = match self.config.mode {
            YocoTruncationMode::Full => 0,
            YocoTruncationMode::Compact => prompt_len.saturating_sub(self.config.keep_tokens),
        };
        ReplayPlan {
            full_len: prompt_len,
            replay_start,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepinfer_core::prop::prop;

    fn cfg(window: usize, backtrack: usize, keep: usize, mode: YocoTruncationMode) -> SwaConfig {
        SwaConfig {
            window,
            max_backtrack: backtrack,
            keep_tokens: keep,
            mode,
        }
    }

    #[test]
    fn yoco_layout_total() {
        let y = YocoLayout {
            self_layers: 12,
            cross_layers: 20,
        };
        assert_eq!(y.total(), 32);
        assert_eq!(
            YocoLayout {
                self_layers: 0,
                cross_layers: 0
            }
            .total(),
            0
        );
        // Saturating, never panics.
        assert_eq!(
            YocoLayout {
                self_layers: usize::MAX,
                cross_layers: 1
            }
            .total(),
            usize::MAX
        );
    }

    #[test]
    fn from_env_contract() {
        let d = Env::default();
        assert_eq!(d.yoco_prefill_keep_tokens, 140);
        assert_eq!(d.yoco_prefill_truncation_mode, YocoTruncationMode::Full);
        let c = SwaConfig::from_env(&d, 128, 12);
        assert_eq!(
            c,
            cfg(128, 12, 140, YocoTruncationMode::Full)
        );

        let rep = Env::from_lookup(|k| {
            if k == "DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE" {
                Some("compact".to_string())
            } else if k == "DEEPINFER_YOCO_PREFILL_KEEP_TOKENS" {
                Some("200".to_string())
            } else {
                None
            }
        });
        let c2 = SwaConfig::from_env(&rep.env, 128, 12);
        assert_eq!(
            c2,
            cfg(128, 12, 200, YocoTruncationMode::Compact)
        );
    }

    #[test]
    fn validate_accepts_and_rejects() {
        // Full mode: only the two lower bounds.
        assert!(cfg(128, 12, 1, YocoTruncationMode::Full).validate().is_ok());
        assert!(cfg(10_000, 500, 3, YocoTruncationMode::Full).validate().is_ok());
        // Compact mode: keep >= window + backtrack (boundary inclusive).
        assert!(cfg(128, 12, 140, YocoTruncationMode::Compact).validate().is_ok());
        assert!(cfg(128, 12, 139, YocoTruncationMode::Compact).validate().is_err());
        assert!(cfg(200, 10, 140, YocoTruncationMode::Compact).validate().is_err());
        // Lower bounds.
        assert!(cfg(0, 0, 10, YocoTruncationMode::Full).validate().is_err());
        assert!(cfg(128, 0, 0, YocoTruncationMode::Full).validate().is_err());
        // Overflow of window + backtrack cannot be covered by any keep_tokens.
        assert!(cfg(usize::MAX, 1, usize::MAX, YocoTruncationMode::Compact)
            .validate()
            .is_err());
    }

    #[test]
    fn planner_modes() {
        let full = BoundedReplayPlanner::new(cfg(128, 12, 140, YocoTruncationMode::Full)).unwrap();
        let p = full.plan_prefill(1000);
        assert_eq!((p.full_len, p.replay_start, p.replay_len()), (1000, 0, 1000));

        let compact =
            BoundedReplayPlanner::new(cfg(128, 12, 140, YocoTruncationMode::Compact)).unwrap();
        let p = compact.plan_prefill(1000);
        assert_eq!((p.replay_start, p.replay_len()), (860, 140));
        let short = compact.plan_prefill(50);
        assert_eq!((short.replay_start, short.replay_len()), (0, 50));
        let exact = compact.plan_prefill(140);
        assert_eq!((exact.replay_start, exact.replay_len()), (0, 140));
        let empty = compact.plan_prefill(0);
        assert_eq!((empty.replay_start, empty.replay_len()), (0, 0));

        // Invalid configs are rejected at construction.
        assert!(BoundedReplayPlanner::new(cfg(200, 10, 140, YocoTruncationMode::Compact)).is_err());
        assert!(BoundedReplayPlanner::new(cfg(0, 0, 10, YocoTruncationMode::Full)).is_err());
    }

    #[test]
    fn replay_len_saturates_for_hand_built_plans() {
        let p = ReplayPlan {
            full_len: 3,
            replay_start: 10,
        };
        assert_eq!(p.replay_len(), 0); // documented saturation
    }

    #[test]
    fn property_planner_invariants() {
        prop("swa_planner_invariants", 0x51, 120, |_i, rng| {
            let window = rng.uniform_usize(1, 200);
            let max_backtrack = rng.uniform_usize(0, 64);
            let mode = if rng.next_f32() < 0.5 {
                YocoTruncationMode::Full
            } else {
                YocoTruncationMode::Compact
            };
            let keep = rng.uniform_usize(1, 400);
            let c = cfg(window, max_backtrack, keep, mode);
            let prompt_len = rng.uniform_usize(0, 1000);
            match BoundedReplayPlanner::new(c) {
                Ok(p) => {
                    let plan = p.plan_prefill(prompt_len);
                    ensure!(plan.full_len == prompt_len, Internal, "full_len");
                    ensure!(plan.replay_start <= plan.full_len, Internal, "start <= len");
                    match mode {
                        YocoTruncationMode::Full => {
                            ensure!(plan.replay_start == 0, Internal, "full start");
                            ensure!(plan.replay_len() == prompt_len, Internal, "full len");
                        }
                        YocoTruncationMode::Compact => {
                            ensure!(
                                plan.replay_start == prompt_len.saturating_sub(keep),
                                Internal,
                                "compact start"
                            );
                            ensure!(plan.replay_len() <= keep, Internal, "compact len <= keep");
                        }
                    }
                }
                Err(_) => {
                    // Only compact mode with keep < window + backtrack fails
                    // (values here cannot overflow usize).
                    ensure!(mode == YocoTruncationMode::Compact, Internal, "mode");
                    ensure!(keep < window + max_backtrack, Internal, "relation");
                }
            }
            Ok(())
        });
    }
}
