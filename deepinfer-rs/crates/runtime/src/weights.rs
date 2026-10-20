//! Runtime weight containers and the deterministic random initializer
//! used by `--tiny` self-tests.
//!
//! Layouts mirror the sibling crates' contracts exactly:
//! * [`deepinfer_mla::MlaWeights`] — self-attention projections
//!   (`MlaWeights::project_q` derives all dims from the field sizes, so
//!   the layouts here are self-describing);
//! * [`deepinfer_dsa::LightningIndexerWeights`] — DSA indexer;
//! * [`deepinfer_moe::MoeWeights`] — router + shared/routed experts;
//! * [`deepinfer_mtp::MtpWeights`] — the depth-1 MTP (NEXTN) module;
//! * cross-decoder layers carry their own `q_proj` / `o_proj` (YOCO:
//!   the cross layers read the self-decoder's KV, so they need no KV
//!   projections of their own).

use deepinfer_core::math::dot;
use deepinfer_core::rng::Philox;
use deepinfer_core::rope::{Rope, RopeStyle};
use deepinfer_core::{ensure, Error, Result};
use deepinfer_dsa::LightningIndexerWeights;
use deepinfer_mla::MlaWeights;
use deepinfer_moe::{ExpertWeights, MoeWeights};
use deepinfer_mtp::{MtpBlockWeights, MtpWeights};

use crate::config::ModelConfig;

/// Hard cap on the total number of f32 values the random initializer will
/// allocate (DoS guard — `random_init` is a self-test constructor; real
/// models come from checkpoints).
pub const MAX_RANDOM_INIT_FLOATS: u64 = 1 << 26;

/// Weights of one self-decoder layer (MLA + DSA indexer + MoE).
pub(crate) struct SelfLayerWeights {
    /// MLA attention weights (includes the layer's rope and latent norm).
    pub(crate) mla: MlaWeights,
    /// DSA lightning-indexer weights.
    pub(crate) indexer: LightningIndexerWeights,
    /// MoE FFN weights.
    pub(crate) moe: MoeWeights,
    /// Pre-attention RMSNorm gain `[hidden]`.
    pub(crate) attn_norm: Vec<f32>,
    /// Pre-FFN RMSNorm gain `[hidden]`.
    pub(crate) ffn_norm: Vec<f32>,
}

/// Weights of one cross-decoder layer (cross-attn q/o + MoE). The
/// cross-attention keys/values come from the self-decoder's KV cache
/// (YOCO "cache once"), so no KV projections exist here.
pub(crate) struct CrossLayerWeights {
    /// Cross-attention query projection `[n_heads * head_dim, hidden]`.
    pub(crate) q_proj: Vec<f32>,
    /// Cross-attention output projection `[hidden, n_heads * head_dim]`.
    pub(crate) o_proj: Vec<f32>,
    /// MoE FFN weights.
    pub(crate) moe: MoeWeights,
    /// Pre-attention RMSNorm gain `[hidden]`.
    pub(crate) attn_norm: Vec<f32>,
    /// Pre-FFN RMSNorm gain `[hidden]`.
    pub(crate) ffn_norm: Vec<f32>,
}

/// The full runtime weight set.
pub(crate) struct ModelWeights {
    /// Token embedding table `[vocab, hidden]` (row `t` embeds token `t`).
    pub(crate) embed: Vec<f32>,
    /// Final RMSNorm gain `[hidden]` (before the LM head).
    pub(crate) final_norm: Vec<f32>,
    /// LM head `[vocab, hidden]` (a clone of `embed` when tied).
    pub(crate) lm_head: Vec<f32>,
    /// Self-decoder layers, in stack order.
    pub(crate) self_layers: Vec<SelfLayerWeights>,
    /// Cross-decoder layers, in stack order.
    pub(crate) cross_layers: Vec<CrossLayerWeights>,
    /// MTP (NEXTN) draft-module weights; `None` when the checkpoint has
    /// no MTP head (speculative decoding is then disabled).
    pub(crate) mtp: Option<MtpWeights>,
    /// Shared RoPE table (built once from the config; cloned into each
    /// layer's `MlaWeights`).
    pub(crate) rope: Rope,
}

/// Allocates `len` zeroed f32 values without aborting on allocation
/// failure.
pub(crate) fn zeros(len: usize) -> Result<Vec<f32>> {
    let mut v = Vec::new();
    v.try_reserve_exact(len).map_err(|_| {
        Error::CapacityExceeded(format!("cannot allocate {len} f32 values"))
    })?;
    v.resize(len, 0.0);
    Ok(v)
}

/// Row-major matrix–vector product: `out[r] = Σ_j w[r*cols + j] · x[j]`,
/// sequential f32 accumulation (fixed order ⇒ deterministic).
pub(crate) fn matvec(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>> {
    ensure!(
        (w.len() as u64) == rows as u64 * cols as u64,
        DimensionMismatch,
        "matvec: weight buffer has {} floats, expected {rows}x{cols}",
        w.len()
    );
    ensure!(
        x.len() == cols,
        DimensionMismatch,
        "matvec: input length {} does not match column count {cols}",
        x.len()
    );
    let mut out = zeros(rows)?;
    for (r, o) in out.iter_mut().enumerate() {
        *o = dot(&w[r * cols..(r + 1) * cols], x);
    }
    Ok(out)
}

/// In-place variant of [`matvec`] writing into a caller-owned buffer.
pub(crate) fn matvec_into(
    w: &[f32],
    x: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> Result<()> {
    ensure!(
        (w.len() as u64) == rows as u64 * cols as u64,
        DimensionMismatch,
        "matvec_into: weight buffer has {} floats, expected {rows}x{cols}",
        w.len()
    );
    ensure!(
        x.len() == cols,
        DimensionMismatch,
        "matvec_into: input length {} does not match column count {cols}",
        x.len()
    );
    ensure!(
        out.len() == rows,
        DimensionMismatch,
        "matvec_into: output length {} does not match row count {rows}",
        out.len()
    );
    for (r, o) in out.iter_mut().enumerate() {
        *o = dot(&w[r * cols..(r + 1) * cols], x);
    }
    Ok(())
}

/// Uniform init helper with the `±0.02/sqrt(fan_in)` bound.
fn fill_fan(rng: &mut Philox, len: usize, fan_in: usize) -> Result<Vec<f32>> {
    let b = if fan_in >= 1 {
        0.02 / (fan_in as f32).sqrt()
    } else {
        0.02
    };
    let mut v = zeros(len)?;
    rng.fill_uniform(&mut v, -b, b);
    Ok(v)
}

/// Unit norm gains `[len]`.
fn unit_gain(len: usize) -> Result<Vec<f32>> {
    let mut v = zeros(len)?;
    v.fill(1.0);
    Ok(v)
}

/// Builds the shared rope table for a config (interleaved V4.1 style,
/// no YaRN — callers can replace the per-layer ropes if they need it).
pub(crate) fn build_rope(cfg: &ModelConfig) -> Result<Rope> {
    Rope::new(
        cfg.rope_dim,
        RopeStyle::Interleaved,
        cfg.rope_base,
        None,
        cfg.max_position_embeddings,
    )
}

/// Total number of f32 values a full random init would allocate.
fn total_floats(cfg: &ModelConfig) -> Result<u64> {
    let mut total: u64 = 0;
    let mut add = |n: u64| -> Result<()> {
        total = total
            .checked_add(n)
            .ok_or_else(|| Error::CapacityExceeded("parameter count overflows u64".into()))?;
        Ok(())
    };
    let (h, hd, vocab) = (cfg.hidden_size as u64, cfg.head_dim as u64, cfg.vocab_size as u64);
    add(vocab * h)?; // embed
    if !cfg.tie_word_embeddings {
        add(vocab * h)?; // lm_head
    }
    add(h)?; // final_norm
    let per_mlp = |mo: &deepinfer_moe::MoeConfig| -> Result<u64> {
        let experts = (mo.num_shared + mo.num_routed) as u64;
        Ok(mo.num_routed as u64 * h + mo.num_routed as u64
            + experts * (2 * mo.intermediate as u64 * h + h * mo.intermediate as u64))
    };
    for _ in 0..cfg.n_self_layers {
        add(cfg.q_lora_rank as u64 * h)?; // q_a
        add(cfg.n_heads as u64 * hd * cfg.q_lora_rank as u64)?; // q_b
        add(hd * h)?; // kv_a
        add(cfg.o_groups as u64 * cfg.o_lora_rank as u64 * (cfg.n_heads as u64
            / cfg.o_groups as u64)
            * hd)?; // wv
        add(h * cfg.o_groups as u64 * cfg.o_lora_rank as u64)?; // wo
        add(hd - cfg.rope_dim as u64)?; // kv_norm
        add(2 * h)?; // attn/ffn norms
        let idx = cfg.dsa.num_heads as u64 * cfg.dsa.head_dim as u64;
        add(2 * idx * h)?; // wq + wk
        add(cfg.dsa.num_heads as u64 * idx)?; // weights_proj
        add(4 * h)?; // gate_proj
        add(cfg.dsa.conv_kernels.iter().map(|k| *k as u64).sum())?; // conv
        add(per_mlp(&cfg.moe)?)?;
    }
    for _ in 0..cfg.n_cross_layers {
        add(2 * cfg.n_heads as u64 * hd * h)?; // q_proj + o_proj
        add(2 * h)?;
        add(per_mlp(&cfg.moe)?)?;
    }
    // MTP: eh_proj + head + block (7 * hidden^2) + norms.
    add(h * 2 * h + vocab * h + 7 * h * h + 4 * h)?;
    Ok(total)
}

/// Deterministic random initialization: every matrix is filled uniformly
/// in `±0.02/sqrt(fan_in)` (norm gains start at 1), consuming a single
/// Philox stream in a fixed, documented order: embedding, per self layer
/// (norms, MLA, indexer, MoE), per cross layer (norms, q/o, MoE),
/// final norm, LM head, MTP module.
///
/// **Self-test only** — production weights come from
/// [`crate::DeepSeekModel::load_checkpoint`]. The total parameter count
/// is checked against [`MAX_RANDOM_INIT_FLOATS`] before any allocation.
pub(crate) fn random_init(cfg: &ModelConfig, seed: u64) -> Result<ModelWeights> {
    let total = total_floats(cfg)?;
    ensure!(
        total <= MAX_RANDOM_INIT_FLOATS,
        CapacityExceeded,
        "random_init would allocate {total} f32 values (> {MAX_RANDOM_INIT_FLOATS}); \
         use load_checkpoint for models of this size"
    );
    let mut rng = Philox::new(seed, 1);
    let h = cfg.hidden_size;
    let hd = cfg.head_dim;
    let latent = hd - cfg.rope_dim;
    let rope = build_rope(cfg)?;

    let embed = fill_fan(&mut rng, cfg.vocab_size * h, h)?;
    let mut self_layers = Vec::with_capacity(cfg.n_self_layers);
    for _ in 0..cfg.n_self_layers {
        let moe = random_moe(&mut rng, &cfg.moe)?;
        let indexer = random_indexer(&mut rng, cfg)?;
        let mla = MlaWeights {
            q_a_proj: fill_fan(&mut rng, cfg.q_lora_rank * h, h)?,
            q_b_proj: fill_fan(&mut rng, cfg.n_heads * hd * cfg.q_lora_rank, cfg.q_lora_rank)?,
            kv_a_proj: fill_fan(&mut rng, hd * h, h)?,
            wv_proj: fill_fan(
                &mut rng,
                cfg.o_groups * cfg.o_lora_rank * (cfg.n_heads / cfg.o_groups) * hd,
                (cfg.n_heads / cfg.o_groups) * hd,
            )?,
            wo_proj: fill_fan(&mut rng, h * cfg.o_groups * cfg.o_lora_rank, cfg.o_groups * cfg.o_lora_rank)?,
            q_norm: None,
            kv_norm: unit_gain(latent)?,
            attn_sink: None,
            rope: rope.clone(),
        };
        self_layers.push(SelfLayerWeights {
            mla,
            indexer,
            moe,
            attn_norm: unit_gain(h)?,
            ffn_norm: unit_gain(h)?,
        });
    }
    let mut cross_layers = Vec::with_capacity(cfg.n_cross_layers);
    for _ in 0..cfg.n_cross_layers {
        cross_layers.push(CrossLayerWeights {
            q_proj: fill_fan(&mut rng, cfg.n_heads * hd * h, h)?,
            o_proj: fill_fan(&mut rng, h * cfg.n_heads * hd, cfg.n_heads * hd)?,
            moe: random_moe(&mut rng, &cfg.moe)?,
            attn_norm: unit_gain(h)?,
            ffn_norm: unit_gain(h)?,
        });
    }
    let final_norm = unit_gain(h)?;
    let lm_head = if cfg.tie_word_embeddings {
        embed.clone()
    } else {
        fill_fan(&mut rng, cfg.vocab_size * h, h)?
    };
    // MTP module (random-init models always carry one so the self-test
    // exercises the speculative path).
    let mtp = MtpWeights {
        eh_proj: fill_fan(&mut rng, h * 2 * h, 2 * h)?,
        shared_norm: unit_gain(h)?,
        depth_norm: unit_gain(h)?,
        block: MtpBlockWeights {
            attn_norm: unit_gain(h)?,
            wq: fill_fan(&mut rng, h * h, h)?,
            wk: fill_fan(&mut rng, h * h, h)?,
            wv: fill_fan(&mut rng, h * h, h)?,
            wo: fill_fan(&mut rng, h * h, h)?,
            mlp_norm: unit_gain(h)?,
            w_gate_up: fill_fan(&mut rng, 2 * h * h, h)?,
            w_down: fill_fan(&mut rng, h * h, h)?,
        },
        head: fill_fan(&mut rng, cfg.vocab_size * h, h)?,
    };
    Ok(ModelWeights {
        embed,
        final_norm,
        lm_head,
        self_layers,
        cross_layers,
        mtp: Some(mtp),
        rope,
    })
}

fn random_moe(rng: &mut Philox, mo: &deepinfer_moe::MoeConfig) -> Result<MoeWeights> {
    let expert = |rng: &mut Philox| -> Result<ExpertWeights> {
        Ok(ExpertWeights {
            w_gate_up: fill_fan(rng, 2 * mo.intermediate * mo.hidden, mo.hidden)?,
            w_down: fill_fan(rng, mo.hidden * mo.intermediate, mo.intermediate)?,
        })
    };
    let mut shared = Vec::with_capacity(mo.num_shared);
    for _ in 0..mo.num_shared {
        shared.push(expert(rng)?);
    }
    let mut routed = Vec::with_capacity(mo.num_routed);
    for _ in 0..mo.num_routed {
        routed.push(expert(rng)?);
    }
    Ok(MoeWeights {
        gate: fill_fan(rng, mo.num_routed * mo.hidden, mo.hidden)?,
        gate_bias: fill_fan(rng, mo.num_routed, 1)?,
        shared,
        routed,
    })
}

fn random_indexer(rng: &mut Philox, cfg: &ModelConfig) -> Result<LightningIndexerWeights> {
    let idx = cfg.dsa.num_heads * cfg.dsa.head_dim;
    let hidden = cfg.hidden_size;
    let mut conv_weights = Vec::with_capacity(cfg.dsa.conv_kernels.len());
    for &k in &cfg.dsa.conv_kernels {
        conv_weights.push(fill_fan(rng, k, k)?);
    }
    Ok(LightningIndexerWeights {
        wq: fill_fan(rng, idx * hidden, hidden)?,
        wk: fill_fan(rng, idx * hidden, hidden)?,
        weights_proj: fill_fan(rng, cfg.dsa.num_heads * idx, idx)?,
        gate_proj: fill_fan(rng, 4 * hidden, 2 * hidden)?,
        conv_weights,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_init_is_deterministic_and_shaped() {
        let cfg = ModelConfig::v41_tiny();
        let a = random_init(&cfg, 42).unwrap();
        let b = random_init(&cfg, 42).unwrap();
        assert_eq!(a.embed, b.embed);
        assert_eq!(a.lm_head, b.lm_head);
        assert_eq!(a.final_norm, b.final_norm);
        assert_eq!(
            a.self_layers[1].mla.kv_a_proj,
            b.self_layers[1].mla.kv_a_proj
        );
        assert_eq!(
            a.cross_layers[0].q_proj,
            b.cross_layers[0].q_proj
        );
        assert_eq!(
            a.self_layers[0].moe.gate,
            b.self_layers[0].moe.gate
        );
        assert!(a.mtp.is_some());

        let h = cfg.hidden_size;
        let hd = cfg.head_dim;
        assert_eq!(a.embed.len(), cfg.vocab_size * h);
        assert_eq!(a.lm_head.len(), cfg.vocab_size * h);
        assert_eq!(a.self_layers.len(), 2);
        assert_eq!(a.cross_layers.len(), 2);
        let sl = &a.self_layers[0];
        assert_eq!(sl.mla.q_a_proj.len(), cfg.q_lora_rank * h);
        assert_eq!(sl.mla.q_b_proj.len(), cfg.n_heads * hd * cfg.q_lora_rank);
        assert_eq!(sl.mla.kv_a_proj.len(), hd * h);
        assert_eq!(sl.mla.kv_norm.len(), hd - cfg.rope_dim);
        assert_eq!(sl.indexer.wq.len(), cfg.dsa.num_heads * cfg.dsa.head_dim * h);
        assert_eq!(sl.moe.gate.len(), cfg.moe.num_routed * h);
        assert_eq!(sl.moe.routed.len(), cfg.moe.num_routed);
        assert_eq!(sl.moe.shared.len(), cfg.moe.num_shared);
        let cl = &a.cross_layers[0];
        assert_eq!(cl.q_proj.len(), cfg.n_heads * hd * h);
        assert_eq!(cl.o_proj.len(), h * cfg.n_heads * hd);
        // Norms start at unit gain; matrices are inside the fan bound.
        assert_eq!(sl.attn_norm, vec![1.0f32; h]);
        let bound = 0.02 / (h as f32).sqrt();
        assert!(sl.mla.kv_a_proj.iter().all(|v| v.abs() <= bound));
        assert!(a.embed.iter().all(|v| v.abs() <= bound));
    }

    #[test]
    fn random_init_is_seed_sensitive() {
        let cfg = ModelConfig::v41_tiny();
        let a = random_init(&cfg, 1).unwrap();
        let b = random_init(&cfg, 2).unwrap();
        assert_ne!(a.embed, b.embed);
    }

    #[test]
    fn random_init_rejects_oversized_configs() {
        // Validate() passes but the parameter budget explodes.
        let mut cfg = ModelConfig::v41_tiny();
        cfg.hidden_size = 4096;
        cfg.n_layers = 8;
        cfg.n_self_layers = 4;
        cfg.n_cross_layers = 4;
        cfg.vocab_size = 65_536;
        cfg.q_lora_rank = 512;
        cfg.head_dim = 512;
        cfg.moe.intermediate = 512;
        cfg.moe.num_routed = 32;
        cfg.moe.hidden = 4096;
        cfg.mtp.hidden = 4096;
        cfg.dsa.hidden_size = 4096;
        cfg.mtp.vocab = 65_536;
        cfg.validate().unwrap();
        assert!(matches!(
            random_init(&cfg, 1),
            Err(Error::CapacityExceeded(_))
        ));
    }

    #[test]
    fn matvec_matches_manual_reference() {
        let w = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![0.5f32, -1.0];
        let out = matvec(&w, &x, 3, 2).unwrap();
        assert_eq!(out, vec![1.0 * 0.5 + 2.0 * -1.0, 3.0 * 0.5 - 4.0, 2.5 - 6.0]);
        assert!(matvec(&w, &x, 2, 2).is_err());
        let mut buf = vec![0.0f32; 3];
        matvec_into(&w, &x, 3, 2, &mut buf).unwrap();
        assert_eq!(buf, out);
        assert!(matvec_into(&w, &x, 3, 3, &mut buf).is_err());
    }
}
