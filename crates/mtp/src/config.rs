//! MTP configuration derived from the `DEEPINFER_*` environment contract
//! (DeepInfer env: `DEEPINFER_USE_NEXTN_DECODE`, default 5).

use deepinfer_core::env::Env;
use deepinfer_core::{ensure, Result};

/// Upper bound on `hidden` accepted by [`MtpConfig::validate`].
pub const MAX_HIDDEN: usize = 8192;
/// Upper bound on `vocab` accepted by [`MtpConfig::validate`].
pub const MAX_VOCAB: usize = 1 << 21;
/// Upper bound on `draft_len` accepted by [`MtpConfig::validate`].
pub const MAX_DRAFT_LEN: usize = 1024;

/// Configuration of the depth-1 MTP (NEXTN) draft module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MtpConfig {
    /// Model hidden size.
    pub hidden: usize,
    /// Vocabulary size.
    pub vocab: usize,
    /// Maximum draft length (`DEEPINFER_USE_NEXTN_DECODE`, V4.1 default 5).
    pub draft_len: usize,
}

impl MtpConfig {
    /// Builds a configuration from the environment contract:
    /// `draft_len = env.use_nextn_decode`.
    ///
    /// `DEEPINFER_ENABLE_MTP_VARLEN` (variable-length MTP draft batches) is
    /// a runtime *batching* concern — it changes how many draft positions
    /// different sequences of one batch carry, not the draft module itself
    /// — and is therefore deliberately not read here.
    pub fn from_env(env: &Env, hidden: usize, vocab: usize) -> Self {
        MtpConfig {
            hidden,
            vocab,
            draft_len: env.use_nextn_decode,
        }
    }

    /// Validates field ranges: `hidden` in `[1, MAX_HIDDEN]`, `vocab` in
    /// `[1, MAX_VOCAB]`, `draft_len` in `[1, MAX_DRAFT_LEN]`.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.hidden >= 1 && self.hidden <= MAX_HIDDEN,
            OutOfRange,
            "MtpConfig: hidden must be in [1, {MAX_HIDDEN}] (got {})",
            self.hidden
        );
        ensure!(
            self.vocab >= 1 && self.vocab <= MAX_VOCAB,
            OutOfRange,
            "MtpConfig: vocab must be in [1, {MAX_VOCAB}] (got {})",
            self.vocab
        );
        ensure!(
            self.draft_len >= 1 && self.draft_len <= MAX_DRAFT_LEN,
            OutOfRange,
            "MtpConfig: draft_len must be in [1, {MAX_DRAFT_LEN}] (got {})",
            self.draft_len
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_env_maps_nextn_decode() {
        let cfg = MtpConfig::from_env(&Env::default(), 64, 128);
        assert_eq!(
            cfg,
            MtpConfig {
                hidden: 64,
                vocab: 128,
                draft_len: 5
            }
        );

        let report = Env::from_lookup(|k| {
            if k == "DEEPINFER_USE_NEXTN_DECODE" {
                Some("3".to_string())
            } else {
                None
            }
        });
        assert!(report.warnings.is_empty());
        let cfg = MtpConfig::from_env(&report.env, 7168, 129280);
        assert_eq!(cfg.draft_len, 3);
        assert_eq!(cfg.hidden, 7168);
        assert_eq!(cfg.vocab, 129280);
    }

    #[test]
    fn validate_bounds() {
        assert!(MtpConfig {
            hidden: 7168,
            vocab: 129280,
            draft_len: 5
        }
        .validate()
        .is_ok());

        for bad in [
            MtpConfig {
                hidden: 0,
                vocab: 16,
                draft_len: 5
            },
            MtpConfig {
                hidden: MAX_HIDDEN + 1,
                vocab: 16,
                draft_len: 5
            },
            MtpConfig {
                hidden: 8,
                vocab: 0,
                draft_len: 5
            },
            MtpConfig {
                hidden: 8,
                vocab: MAX_VOCAB + 1,
                draft_len: 5
            },
            MtpConfig {
                hidden: 8,
                vocab: 16,
                draft_len: 0
            },
            MtpConfig {
                hidden: 8,
                vocab: 16,
                draft_len: MAX_DRAFT_LEN + 1
            },
        ] {
            assert!(bad.validate().is_err(), "{bad:?}");
        }
    }
}
