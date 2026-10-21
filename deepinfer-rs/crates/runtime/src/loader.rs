//! Checkpoint loading: pragmatic, loosely-coupled tensor-name mapping over
//! [`deepinfer_core::safetensors::CheckpointDir`].
//!
//! Both HF-style `model.layers.{i}.<suffix>` names and flat names (the
//! same suffixes without the `model.` prefix, `layers.{i}.`, or the
//! runtime-local `self.layers.{i}.` / `cross.layers.{j}.` prefixes) are
//! accepted: for each logical tensor a candidate list is tried in order
//! and the first present name wins. Missing **required** tensors fail
//! with one [`Error::Checkpoint`] listing up to five of the missing
//! names. Optional tensors (MLA `attn_sink`, the MoE `e_score_correction`
//! bias, and the whole MTP module) default to `None` / zeros.
//!
//! Layer indexing follows the global HF convention: self-decoder layers
//! occupy `0..n_self`, cross-decoder layers `n_self..n_layers`.

use std::collections::HashSet;
#[cfg(test)]
use std::path::Path;

use deepinfer_core::safetensors::CheckpointDir;
use deepinfer_core::{ensure, Error, Result};
use deepinfer_dsa::LightningIndexerWeights;
use deepinfer_mla::MlaWeights;
use deepinfer_moe::{ExpertWeights, MoeWeights};
use deepinfer_mtp::{MtpBlockWeights, MtpWeights};

use crate::config::ModelConfig;
use crate::weights::{
    build_rope, CrossLayerWeights, ModelWeights, SelfLayerWeights,
};

/// Maximum number of missing tensor names listed in a load error.
const MAX_MISSING_LISTED: usize = 5;

/// Name resolver over a checkpoint's tensor-name set, accumulating the
/// missing required tensors.
struct Resolver {
    avail: HashSet<String>,
    missing: Vec<String>,
}

impl Resolver {
    fn new(cp: &CheckpointDir) -> Resolver {
        Resolver {
            avail: cp.names().map(str::to_string).collect(),
            missing: Vec::new(),
        }
    }

    /// First present candidate, or `None` (recorded as missing).
    fn pick(&mut self, cands: &[String], what: &str) -> Option<String> {
        let found = cands.iter().find(|c| self.avail.contains(*c)).cloned();
        if found.is_none() {
            self.missing.push(what.to_string());
        }
        found
    }

    /// First present candidate; absence is not an error.
    fn opt(&self, cands: &[String]) -> Option<String> {
        cands.iter().find(|c| self.avail.contains(*c)).cloned()
    }

    /// Fails when any required tensor was missing (up to
    /// [`MAX_MISSING_LISTED`] names in the message).
    fn finish(&self) -> Result<()> {
        if self.missing.is_empty() {
            return Ok(());
        }
        let shown: Vec<&str> = self
            .missing
            .iter()
            .take(MAX_MISSING_LISTED)
            .map(String::as_str)
            .collect();
        Err(Error::Checkpoint(format!(
            "checkpoint is missing {} required tensor(s): {}{}",
            self.missing.len(),
            shown.join(", "),
            if self.missing.len() > MAX_MISSING_LISTED {
                format!(" (+{} more)", self.missing.len() - MAX_MISSING_LISTED)
            } else {
                String::new()
            }
        )))
    }
}

/// Loads one tensor as flat f32 and checks its element count.
fn tensor_f32(cp: &CheckpointDir, name: &str, expect: usize, what: &str) -> Result<Vec<f32>> {
    let t = cp.tensor(name).map_err(|e| {
        Error::Checkpoint(format!("reading {what} tensor {name:?}: {e}"))
    })?;
    let v = t.as_f32_flat().map_err(|e| {
        Error::Checkpoint(format!("{what} tensor {name:?}: {e}"))
    })?;
    ensure!(
        v.len() == expect,
        Checkpoint,
        "{what} tensor {name:?} has {} elements, expected {expect}",
        v.len()
    );
    Ok(v)
}

/// Candidate names for a set of layer prefixes and a suffix.
fn prefixed(prefixes: &[String], suffix: &str) -> Vec<String> {
    prefixes.iter().map(|p| format!("{p}{suffix}")).collect()
}

/// Prefix candidates for self-decoder layer `i` (global index).
fn self_prefixes(i: usize) -> Vec<String> {
    vec![
        format!("model.layers.{i}."),
        format!("layers.{i}."),
        format!("self.layers.{i}."),
    ]
}

/// Prefix candidates for cross-decoder layer `j` (global index
/// `n_self + j`).
fn cross_prefixes(n_self: usize, j: usize) -> Vec<String> {
    let g = n_self + j;
    vec![
        format!("model.layers.{g}."),
        format!("layers.{g}."),
        format!("cross.layers.{j}."),
    ]
}

/// Loads the full weight set for `cfg` from an opened checkpoint
/// directory.
pub(crate) fn load_weights(cp: &CheckpointDir, cfg: &ModelConfig) -> Result<ModelWeights> {
    let mut r = Resolver::new(cp);
    let h = cfg.hidden_size;
    let hd = cfg.head_dim;
    let latent = hd - cfg.rope_dim;
    let hpg = cfg.n_heads / cfg.o_groups;
    let rope = build_rope(cfg)?;

    // ---- shared tensors ----
    let embed_name = r
        .pick(
            &[
                "model.embed_tokens.weight".to_string(),
                "embed_tokens.weight".into(),
                "embedding.weight".into(),
                "tok_embeddings.weight".into(),
            ],
            "embedding (embed_tokens)",
        )
        .ok_or_else(|| Error::internal("unreachable: embedding pick"))?;
    let final_norm_name = r
        .pick(
            &[
                "model.norm.weight".to_string(),
                "norm.weight".into(),
            ],
            "final norm (norm)",
        )
        .ok_or_else(|| Error::internal("unreachable: final norm pick"))?;
    let lm_head_cands = vec!["lm_head.weight".to_string(), "model.lm_head.weight".into()];
    let lm_head_name = if cfg.tie_word_embeddings {
        r.opt(&lm_head_cands)
    } else {
        r.pick(&lm_head_cands, "lm_head")
    };

    // ---- per-layer plans ----
    struct MoePlan {
        gate: Option<String>,
        gate_bias: Option<String>,
        shared: Vec<[Option<String>; 2]>,
        routed: Vec<[Option<String>; 2]>,
    }
    struct LayerPlan {
        attn_norm: Option<String>,
        ffn_norm: Option<String>,
        moe: MoePlan,
    }
    let moe_plan = |r: &mut Resolver, prefixes: &[String]| -> MoePlan {
        let gate = r.pick(&prefixed(prefixes, "mlp.gate.weight"), "MoE router gate");
        let gate_bias = r.opt(&prefixed(prefixes, "mlp.gate_bias.weight"));
        let mut shared = Vec::with_capacity(cfg.moe.num_shared);
        for si in 0..cfg.moe.num_shared {
            // With multiple shared experts the indexed name is required;
            // a lone shared expert may use the un-indexed name.
            let gu_cands = if cfg.moe.num_shared == 1 {
                prefixed(prefixes, "mlp.shared_experts.gate_up.weight")
            } else {
                prefixed(prefixes, &format!("mlp.shared_experts.{si}.gate_up.weight"))
            };
            let dn_cands = if cfg.moe.num_shared == 1 {
                prefixed(prefixes, "mlp.shared_experts.down.weight")
            } else {
                prefixed(prefixes, &format!("mlp.shared_experts.{si}.down.weight"))
            };
            let gu = r.pick(&gu_cands, &format!("shared expert {si} gate_up"));
            let dn = r.pick(&dn_cands, &format!("shared expert {si} down"));
            shared.push([gu, dn]);
        }
        let mut routed = Vec::with_capacity(cfg.moe.num_routed);
        for e in 0..cfg.moe.num_routed {
            let gu = r.pick(
                &prefixed(prefixes, &format!("mlp.experts.{e}.gate_up.weight")),
                &format!("routed expert {e} gate_up"),
            );
            let dn = r.pick(
                &prefixed(prefixes, &format!("mlp.experts.{e}.down.weight")),
                &format!("routed expert {e} down"),
            );
            routed.push([gu, dn]);
        }
        MoePlan {
            gate,
            gate_bias,
            shared,
            routed,
        }
    };

    struct SelfPlan {
        layer: LayerPlan,
        q_a: Option<String>,
        q_b: Option<String>,
        kv_a: Option<String>,
        wv: Option<String>,
        wo: Option<String>,
        kv_norm: Option<String>,
        attn_sink: Option<String>,
        wq: Option<String>,
        wk: Option<String>,
        weights_proj: Option<String>,
        gate_proj: Option<String>,
        conv: Vec<Option<String>>,
    }
    let mut self_plans = Vec::with_capacity(cfg.n_self_layers);
    for i in 0..cfg.n_self_layers {
        let p = self_prefixes(i);
        let layer = LayerPlan {
            attn_norm: r.pick(&prefixed(&p, "input_layernorm.weight"), &format!("layer {i} input norm")),
            ffn_norm: r.pick(&prefixed(&p, "post_attention_layernorm.weight"), &format!("layer {i} post-attention norm")),
            moe: moe_plan(&mut r, &p),
        };
        let conv = (0..cfg.dsa.conv_kernels.len())
            .map(|c| {
                r.pick(
                    &prefixed(&p, &format!("indexer.conv.weights.{c}")),
                    &format!("layer {i} indexer conv {c}"),
                )
            })
            .collect();
        self_plans.push(SelfPlan {
            layer,
            q_a: r.pick(&prefixed(&p, "mla.q_a_proj.weight"), &format!("layer {i} MLA q_a")),
            q_b: r.pick(&prefixed(&p, "mla.q_b_proj.weight"), &format!("layer {i} MLA q_b")),
            kv_a: r.pick(&prefixed(&p, "mla.kv_a_proj.weight"), &format!("layer {i} MLA kv_a")),
            wv: r.pick(&prefixed(&p, "mla.wv_proj.weight"), &format!("layer {i} MLA wv")),
            wo: r.pick(&prefixed(&p, "mla.wo_proj.weight"), &format!("layer {i} MLA wo")),
            kv_norm: r.pick(&prefixed(&p, "mla.kv_norm.weight"), &format!("layer {i} MLA kv_norm")),
            attn_sink: r.opt(&prefixed(&p, "self_attn.attn_sink.weight")),
            wq: r.pick(&prefixed(&p, "indexer.wq.weight"), &format!("layer {i} indexer wq")),
            wk: r.pick(&prefixed(&p, "indexer.wk.weight"), &format!("layer {i} indexer wk")),
            weights_proj: r.pick(&prefixed(&p, "indexer.weights_proj.weight"), &format!("layer {i} indexer weights_proj")),
            gate_proj: r.pick(&prefixed(&p, "indexer.gate_proj.weight"), &format!("layer {i} indexer gate_proj")),
            conv,
        });
    }

    struct CrossPlan {
        layer: LayerPlan,
        q_proj: Option<String>,
        o_proj: Option<String>,
    }
    let mut cross_plans = Vec::with_capacity(cfg.n_cross_layers);
    for j in 0..cfg.n_cross_layers {
        let p = cross_prefixes(cfg.n_self_layers, j);
        cross_plans.push(CrossPlan {
            layer: LayerPlan {
                attn_norm: r.pick(&prefixed(&p, "input_layernorm.weight"), &format!("cross layer {j} input norm")),
                ffn_norm: r.pick(&prefixed(&p, "post_attention_layernorm.weight"), &format!("cross layer {j} post-attention norm")),
                moe: moe_plan(&mut r, &p),
            },
            q_proj: r.pick(&prefixed(&p, "cross_attn.q_proj.weight"), &format!("cross layer {j} q_proj")),
            o_proj: r.pick(&prefixed(&p, "cross_attn.o_proj.weight"), &format!("cross layer {j} o_proj")),
        });
    }

    // ---- MTP (optional module: probed via eh_proj) ----
    let mtp_prefixes = ["mtp.".to_string(), "model.mtp.".into()];
    let mtp_eh = mtp_prefixes
        .iter()
        .find_map(|p| r.opt(&[format!("{p}eh_proj.weight")]));
    let mtp_plan = if mtp_eh.is_some() {
        let mut names = Vec::new();
        for suffix in [
            "eh_proj.weight",
            "shared_norm.weight",
            "depth_norm.weight",
            "head.weight",
            "block.attn_norm.weight",
            "block.wq.weight",
            "block.wk.weight",
            "block.wv.weight",
            "block.wo.weight",
            "block.mlp_norm.weight",
            "block.w_gate_up.weight",
            "block.w_down.weight",
        ] {
            let cands: Vec<String> = mtp_prefixes.iter().map(|p| format!("{p}{suffix}")).collect();
            let n = r.pick(&cands, &format!("MTP {suffix}"));
            names.push(n);
        }
        Some(names)
    } else {
        None
    };

    // ---- fail fast on missing required tensors ----
    r.finish()?;

    // ---- materialize ----
    let embed = tensor_f32(cp, &embed_name, cfg.vocab_size * h, "embedding")?;
    let final_norm = tensor_f32(cp, &final_norm_name, h, "final norm")?;
    let lm_head = match &lm_head_name {
        Some(n) => tensor_f32(cp, n, cfg.vocab_size * h, "lm_head")?,
        None => embed.clone(), // tied (or missing handled above)
    };

    let load_moe = |plan: &MoePlan| -> Result<MoeWeights> {
        let gate = tensor_f32(cp, plan.gate.as_ref().unwrap(), cfg.moe.num_routed * h, "MoE gate")?;
        let gate_bias = match &plan.gate_bias {
            Some(n) => tensor_f32(cp, n, cfg.moe.num_routed, "MoE gate bias")?,
            None => vec![0.0; cfg.moe.num_routed],
        };
        let expert = |gu: &Option<String>, dn: &Option<String>| -> Result<ExpertWeights> {
            Ok(ExpertWeights {
                w_gate_up: tensor_f32(cp, gu.as_ref().unwrap(), 2 * cfg.moe.intermediate * h, "expert gate_up")?,
                w_down: tensor_f32(cp, dn.as_ref().unwrap(), h * cfg.moe.intermediate, "expert down")?,
            })
        };
        let mut shared = Vec::with_capacity(plan.shared.len());
        for s in &plan.shared {
            shared.push(expert(&s[0], &s[1])?);
        }
        let mut routed = Vec::with_capacity(plan.routed.len());
        for e in &plan.routed {
            routed.push(expert(&e[0], &e[1])?);
        }
        Ok(MoeWeights {
            gate,
            gate_bias,
            shared,
            routed,
        })
    };

    let mut self_layers = Vec::with_capacity(cfg.n_self_layers);
    for plan in &self_plans {
        let idx = cfg.dsa.num_heads * cfg.dsa.head_dim;
        let mut conv_weights = Vec::with_capacity(plan.conv.len());
        for (c, n) in plan.conv.iter().enumerate() {
            conv_weights.push(tensor_f32(
                cp,
                n.as_ref().unwrap(),
                cfg.dsa.conv_kernels[c],
                "indexer conv",
            )?);
        }
        let mla = MlaWeights {
            q_a_proj: tensor_f32(cp, plan.q_a.as_ref().unwrap(), cfg.q_lora_rank * h, "MLA q_a")?,
            q_b_proj: tensor_f32(
                cp,
                plan.q_b.as_ref().unwrap(),
                cfg.n_heads * hd * cfg.q_lora_rank,
                "MLA q_b",
            )?,
            kv_a_proj: tensor_f32(cp, plan.kv_a.as_ref().unwrap(), hd * h, "MLA kv_a")?,
            wv_proj: tensor_f32(
                cp,
                plan.wv.as_ref().unwrap(),
                cfg.o_groups * cfg.o_lora_rank * hpg * hd,
                "MLA wv",
            )?,
            wo_proj: tensor_f32(
                cp,
                plan.wo.as_ref().unwrap(),
                h * cfg.o_groups * cfg.o_lora_rank,
                "MLA wo",
            )?,
            q_norm: None,
            kv_norm: tensor_f32(cp, plan.kv_norm.as_ref().unwrap(), latent, "MLA kv_norm")?,
            attn_sink: match &plan.attn_sink {
                Some(n) => Some(tensor_f32(cp, n, cfg.n_heads, "attn_sink")?),
                None => None,
            },
            rope: rope.clone(),
        };
        let indexer = LightningIndexerWeights {
            wq: tensor_f32(cp, plan.wq.as_ref().unwrap(), idx * h, "indexer wq")?,
            wk: tensor_f32(cp, plan.wk.as_ref().unwrap(), idx * h, "indexer wk")?,
            weights_proj: tensor_f32(
                cp,
                plan.weights_proj.as_ref().unwrap(),
                cfg.dsa.num_heads * idx,
                "indexer weights_proj",
            )?,
            gate_proj: tensor_f32(cp, plan.gate_proj.as_ref().unwrap(), 4 * h, "indexer gate_proj")?,
            conv_weights,
        };
        self_layers.push(SelfLayerWeights {
            mla,
            indexer,
            moe: load_moe(&plan.layer.moe)?,
            attn_norm: tensor_f32(cp, plan.layer.attn_norm.as_ref().unwrap(), h, "input norm")?,
            ffn_norm: tensor_f32(cp, plan.layer.ffn_norm.as_ref().unwrap(), h, "post-attention norm")?,
        });
    }

    let mut cross_layers = Vec::with_capacity(cfg.n_cross_layers);
    for plan in &cross_plans {
        cross_layers.push(CrossLayerWeights {
            q_proj: tensor_f32(cp, plan.q_proj.as_ref().unwrap(), cfg.n_heads * hd * h, "cross q_proj")?,
            o_proj: tensor_f32(cp, plan.o_proj.as_ref().unwrap(), h * cfg.n_heads * hd, "cross o_proj")?,
            moe: load_moe(&plan.layer.moe)?,
            attn_norm: tensor_f32(cp, plan.layer.attn_norm.as_ref().unwrap(), h, "input norm")?,
            ffn_norm: tensor_f32(cp, plan.layer.ffn_norm.as_ref().unwrap(), h, "post-attention norm")?,
        });
    }

    let mtp = match &mtp_plan {
        Some(names) => {
            let gi = h; // block intermediate == hidden (reference layout)
            Some(MtpWeights {
                eh_proj: tensor_f32(cp, names[0].as_ref().unwrap(), h * 2 * h, "MTP eh_proj")?,
                shared_norm: tensor_f32(cp, names[1].as_ref().unwrap(), h, "MTP shared_norm")?,
                depth_norm: tensor_f32(cp, names[2].as_ref().unwrap(), h, "MTP depth_norm")?,
                block: MtpBlockWeights {
                    attn_norm: tensor_f32(cp, names[4].as_ref().unwrap(), h, "MTP block attn_norm")?,
                    wq: tensor_f32(cp, names[5].as_ref().unwrap(), h * h, "MTP block wq")?,
                    wk: tensor_f32(cp, names[6].as_ref().unwrap(), h * h, "MTP block wk")?,
                    wv: tensor_f32(cp, names[7].as_ref().unwrap(), h * h, "MTP block wv")?,
                    wo: tensor_f32(cp, names[8].as_ref().unwrap(), h * h, "MTP block wo")?,
                    mlp_norm: tensor_f32(cp, names[9].as_ref().unwrap(), h, "MTP block mlp_norm")?,
                    w_gate_up: tensor_f32(cp, names[10].as_ref().unwrap(), 2 * gi * h, "MTP block w_gate_up")?,
                    w_down: tensor_f32(cp, names[11].as_ref().unwrap(), h * gi, "MTP block w_down")?,
                },
                head: tensor_f32(cp, names[3].as_ref().unwrap(), cfg.vocab_size * h, "MTP head")?,
            })
        }
        None => None,
    };

    Ok(ModelWeights {
        embed,
        final_norm,
        lm_head,
        self_layers,
        cross_layers,
        mtp,
        rope,
    })
}

/// Reads `dir/config.json` if present (test/example helper).
#[cfg(test)]
pub(crate) fn config_json_exists(dir: &Path) -> bool {
    dir.join("config.json").is_file()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use crate::model::DeepSeekModel;
    use deepinfer_core::json::Json;
    use std::fs;

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("di_runtime_{tag}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    /// Serializes `tensors` as one F32 safetensors file.
    fn write_safetensors(
        path: &std::path::Path,
        tensors: &[(String, Vec<f32>)],
    ) -> std::io::Result<()> {
        let mut entries: Vec<(String, Json)> = Vec::new();
        let mut offset = 0usize;
        for (name, data) in tensors {
            let end = offset + data.len() * 4;
            entries.push((
                name.clone(),
                Json::Obj(vec![
                    ("dtype".into(), Json::Str("F32".into())),
                    ("shape".into(), Json::Arr(vec![Json::U64(data.len() as u64)])),
                    (
                        "data_offsets".into(),
                        Json::Arr(vec![Json::U64(offset as u64), Json::U64(end as u64)]),
                    ),
                ]),
            ));
            offset = end;
        }
        let header = Json::Obj(entries).to_json_string();
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        for (_, data) in tensors {
            for v in data {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        fs::write(path, out)
    }

    /// Enumerates the model's tensors with the loader's PRIMARY
    /// (first-candidate) names; `flat` strips the leading `model.`.
    fn checkpoint_tensors(m: &DeepSeekModel, flat: bool) -> Vec<(String, Vec<f32>)> {
        let cfg = m.config();
        let mut t: Vec<(String, Vec<f32>)> = Vec::new();
        let pfx = |s: &str| -> String {
            if flat {
                s.strip_prefix("model.").unwrap_or(s).to_string()
            } else {
                s.to_string()
            }
        };
        t.push((pfx("model.embed_tokens.weight"), m.weights.embed.clone()));
        t.push((pfx("model.norm.weight"), m.weights.final_norm.clone()));
        t.push(("lm_head.weight".to_string(), m.weights.lm_head.clone()));
        let moe = |t: &mut Vec<(String, Vec<f32>)>, p: String, mo: &MoeWeights| {
            t.push((format!("{p}mlp.gate.weight"), mo.gate.clone()));
            t.push((format!("{p}mlp.gate_bias.weight"), mo.gate_bias.clone()));
            for (si, s) in mo.shared.iter().enumerate() {
                let tag = if mo.shared.len() > 1 {
                    format!(".{si}")
                } else {
                    String::new()
                };
                t.push((format!("{p}mlp.shared_experts{tag}.gate_up.weight"), s.w_gate_up.clone()));
                t.push((format!("{p}mlp.shared_experts{tag}.down.weight"), s.w_down.clone()));
            }
            for (e, w) in mo.routed.iter().enumerate() {
                t.push((format!("{p}mlp.experts.{e}.gate_up.weight"), w.w_gate_up.clone()));
                t.push((format!("{p}mlp.experts.{e}.down.weight"), w.w_down.clone()));
            }
        };
        for (i, sl) in m.weights.self_layers.iter().enumerate() {
            let p = format!("model.layers.{i}.");
            let p = pfx(&p);
            t.push((format!("{p}input_layernorm.weight"), sl.attn_norm.clone()));
            t.push((format!("{p}post_attention_layernorm.weight"), sl.ffn_norm.clone()));
            t.push((format!("{p}mla.q_a_proj.weight"), sl.mla.q_a_proj.clone()));
            t.push((format!("{p}mla.q_b_proj.weight"), sl.mla.q_b_proj.clone()));
            t.push((format!("{p}mla.kv_a_proj.weight"), sl.mla.kv_a_proj.clone()));
            t.push((format!("{p}mla.wv_proj.weight"), sl.mla.wv_proj.clone()));
            t.push((format!("{p}mla.wo_proj.weight"), sl.mla.wo_proj.clone()));
            t.push((format!("{p}mla.kv_norm.weight"), sl.mla.kv_norm.clone()));
            t.push((format!("{p}indexer.wq.weight"), sl.indexer.wq.clone()));
            t.push((format!("{p}indexer.wk.weight"), sl.indexer.wk.clone()));
            t.push((format!("{p}indexer.weights_proj.weight"), sl.indexer.weights_proj.clone()));
            t.push((format!("{p}indexer.gate_proj.weight"), sl.indexer.gate_proj.clone()));
            for (c, cw) in sl.indexer.conv_weights.iter().enumerate() {
                t.push((format!("{p}indexer.conv.weights.{c}"), cw.clone()));
            }
            moe(&mut t, p, &sl.moe);
        }
        for (j, cl) in m.weights.cross_layers.iter().enumerate() {
            let g = cfg.n_self_layers + j;
            let p = pfx(&format!("model.layers.{g}."));
            t.push((format!("{p}input_layernorm.weight"), cl.attn_norm.clone()));
            t.push((format!("{p}post_attention_layernorm.weight"), cl.ffn_norm.clone()));
            t.push((format!("{p}cross_attn.q_proj.weight"), cl.q_proj.clone()));
            t.push((format!("{p}cross_attn.o_proj.weight"), cl.o_proj.clone()));
            moe(&mut t, p, &cl.moe);
        }
        if let Some(mtp) = &m.weights.mtp {
            let p = "mtp.".to_string();
            t.push((format!("{p}eh_proj.weight"), mtp.eh_proj.clone()));
            t.push((format!("{p}shared_norm.weight"), mtp.shared_norm.clone()));
            t.push((format!("{p}depth_norm.weight"), mtp.depth_norm.clone()));
            t.push((format!("{p}head.weight"), mtp.head.clone()));
            t.push((format!("{p}block.attn_norm.weight"), mtp.block.attn_norm.clone()));
            t.push((format!("{p}block.wq.weight"), mtp.block.wq.clone()));
            t.push((format!("{p}block.wk.weight"), mtp.block.wk.clone()));
            t.push((format!("{p}block.wv.weight"), mtp.block.wv.clone()));
            t.push((format!("{p}block.wo.weight"), mtp.block.wo.clone()));
            t.push((format!("{p}block.mlp_norm.weight"), mtp.block.mlp_norm.clone()));
            t.push((format!("{p}block.w_gate_up.weight"), mtp.block.w_gate_up.clone()));
            t.push((format!("{p}block.w_down.weight"), mtp.block.w_down.clone()));
        }
        t
    }

    fn write_config_json(dir: &std::path::Path, cfg: &ModelConfig) {
        let json = Json::Obj(vec![
            ("hidden_size".into(), Json::U64(cfg.hidden_size as u64)),
            ("n_layers".into(), Json::U64(cfg.n_layers as u64)),
            ("n_self_layers".into(), Json::U64(cfg.n_self_layers as u64)),
            ("n_cross_layers".into(), Json::U64(cfg.n_cross_layers as u64)),
            ("vocab_size".into(), Json::U64(cfg.vocab_size as u64)),
        ]);
        fs::write(dir.join("config.json"), json.to_json_string()).unwrap();
    }

    #[test]
    fn load_checkpoint_missing_dir_errors_cleanly() {
        let err = DeepSeekModel::load_checkpoint(
            std::path::Path::new("/nonexistent/deepinfer/model"),
            &ModelConfig::v41_tiny(),
        )
        .err().expect("expected an error");
        assert!(matches!(err, Error::Checkpoint(_)), "{err}");
        assert!(err.to_string().contains("nonexistent"), "{err}");
    }

    #[test]
    fn load_checkpoint_empty_dir_errors() {
        let dir = temp_dir("empty");
        let err = DeepSeekModel::load_checkpoint(&dir, &ModelConfig::v41_tiny()).err().expect("expected an error");
        assert!(matches!(err, Error::Checkpoint(_)), "{err}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_checkpoint_round_trip_hf_names() {
        let cfg = ModelConfig::v41_tiny();
        let mut model = DeepSeekModel::random_init(&cfg, 1234).unwrap();
        let dir = temp_dir("rt_hf");
        write_safetensors(&dir.join("model.safetensors"), &checkpoint_tensors(&model, false))
            .unwrap();
        write_config_json(&dir, &cfg);
        let mut loaded = DeepSeekModel::load_checkpoint(&dir, &cfg).unwrap();
        assert!(loaded.has_mtp());
        // Identical weights -> identical forwards.
        let prompt: Vec<u32> = vec![3, 1, 4, 1, 5];
        let a = model.prefill(0, &prompt).unwrap();
        let b = loaded.prefill(0, &prompt).unwrap();
        assert_eq!(a, b);
        let mut ta = a.clone();
        let mut tb = b.clone();
        for tok in [7u32, 8, 9] {
            ta = model.decode_step(0, tok).unwrap();
            tb = loaded.decode_step(0, tok).unwrap();
        }
        assert_eq!(ta, tb);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_checkpoint_flat_names() {
        let cfg = ModelConfig::v41_tiny();
        let mut model = DeepSeekModel::random_init(&cfg, 99).unwrap();
        let dir = temp_dir("rt_flat");
        write_safetensors(&dir.join("m.safetensors"), &checkpoint_tensors(&model, true)).unwrap();
        let mut loaded = DeepSeekModel::load_checkpoint(&dir, &cfg).unwrap();
        let a = model.prefill(1, &[1, 2, 3]).unwrap();
        let b = loaded.prefill(1, &[1, 2, 3]).unwrap();
        assert_eq!(a, b);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_checkpoint_without_mtp_disables_speculation() {
        let cfg = ModelConfig::v41_tiny();
        let model = DeepSeekModel::random_init(&cfg, 5).unwrap();
        let dir = temp_dir("no_mtp");
        let tensors: Vec<(String, Vec<f32>)> = checkpoint_tensors(&model, false)
            .into_iter()
            .filter(|(n, _)| !n.starts_with("mtp."))
            .collect();
        write_safetensors(&dir.join("model.safetensors"), &tensors).unwrap();
        let loaded = DeepSeekModel::load_checkpoint(&dir, &cfg).unwrap();
        assert!(!loaded.has_mtp());
        // Still generates.
        let mut m = loaded;
        let out = crate::generate(
            &mut m,
            vec![crate::GenerateRequest {
                prompt: vec![1, 2],
                params: deepinfer_sampler::SamplingParams {
                    max_tokens: 4,
                    ..Default::default()
                },
            }],
            0,
            &deepinfer_core::env::Env::default(),
        )
        .unwrap();
        assert_eq!(out[0].tokens.len(), 4);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_checkpoint_missing_tensor_lists_names() {
        let cfg = ModelConfig::v41_tiny();
        let model = DeepSeekModel::random_init(&cfg, 7).unwrap();
        let dir = temp_dir("missing");
        // Drop lm_head (untied) and one indexer conv kernel.
        let tensors: Vec<(String, Vec<f32>)> = checkpoint_tensors(&model, false)
            .into_iter()
            .filter(|(n, _)| {
                n != "lm_head.weight" && n != "model.layers.1.indexer.conv.weights.0"
            })
            .collect();
        write_safetensors(&dir.join("model.safetensors"), &tensors).unwrap();
        let err = DeepSeekModel::load_checkpoint(&dir, &cfg).err().expect("expected an error");
        let msg = err.to_string();
        assert!(msg.contains("missing 2 required tensor"), "{msg}");
        assert!(msg.contains("lm_head") || msg.contains("conv"), "{msg}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_checkpoint_wrong_shapes_error() {
        let cfg = ModelConfig::v41_tiny();
        let model = DeepSeekModel::random_init(&cfg, 8).unwrap();
        let dir = temp_dir("badshape");
        let mut tensors = checkpoint_tensors(&model, false);
        for (n, v) in tensors.iter_mut() {
            if n == "model.norm.weight" {
                v.push(0.5); // one element too many
            }
        }
        write_safetensors(&dir.join("model.safetensors"), &tensors).unwrap();
        let err = DeepSeekModel::load_checkpoint(&dir, &cfg).err().expect("expected an error");
        assert!(matches!(err, Error::Checkpoint(_)), "{err}");
        assert!(err.to_string().contains("final norm"), "{err}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_checkpoint_config_json_mismatch_errors() {
        let cfg = ModelConfig::v41_tiny();
        let model = DeepSeekModel::random_init(&cfg, 9).unwrap();
        let dir = temp_dir("cfgmismatch");
        write_safetensors(&dir.join("model.safetensors"), &checkpoint_tensors(&model, false))
            .unwrap();
        let mut other = cfg.clone();
        other.vocab_size = 255;
        let err = DeepSeekModel::load_checkpoint(&dir, &other).err().expect("expected an error");
        assert!(
            err.to_string().contains("vocab_size") || err.to_string().contains("mtp shape"),
            "{err}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn config_json_exists_helper() {
        let dir = temp_dir("cfgexists");
        assert!(!config_json_exists(&dir));
        write_config_json(&dir, &ModelConfig::v41_tiny());
        assert!(config_json_exists(&dir));
        let _ = fs::remove_dir_all(&dir);
    }
}
