//! Complete Hugging Face-to-Burn weight inventory for ordinary Qwen3-VL.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{Qwen3VlConfig, Qwen3VlError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightRole {
    Embedding,
    LinearWeight,
    LinearBias,
    NormalizationScale,
    NormalizationBias,
    ConvolutionWeight,
    ConvolutionBias,
}

/// One required tensor and its source/record names and exact logical shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightSpec {
    /// Published Hugging Face checkpoint key.
    pub source: String,
    /// Burn record path. Normalization `weight`/`bias` map to Burn `gamma`/`beta`.
    pub target: String,
    pub shape: Vec<usize>,
    pub role: WeightRole,
    pub required: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightInventory {
    specs: Vec<WeightSpec>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightValidation {
    pub missing: Vec<String>,
    pub unknown: Vec<String>,
    pub duplicate: Vec<String>,
    pub shape_mismatches: Vec<WeightShapeMismatch>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightShapeMismatch {
    pub key: String,
    pub expected: Vec<usize>,
    pub actual: Vec<usize>,
}

impl WeightValidation {
    pub fn is_complete(&self) -> bool {
        self.missing.is_empty()
            && self.unknown.is_empty()
            && self.duplicate.is_empty()
            && self.shape_mismatches.is_empty()
    }

    pub fn into_result(self) -> Result<()> {
        if self.is_complete() {
            Ok(())
        } else {
            Err(Qwen3VlError::Weights(format!(
                "missing={:?}, unknown={:?}, duplicate={:?}, shape_mismatches={:?}",
                self.missing, self.unknown, self.duplicate, self.shape_mismatches
            )))
        }
    }
}

impl WeightInventory {
    /// Construct an inventory targeting a standalone [`Qwen3VlModel`](crate::Qwen3VlModel)
    /// record. Published checkpoint sources retain their leading `model.` segment while target
    /// record fields omit it.
    pub fn for_base_model(config: &Qwen3VlConfig) -> Self {
        let mut inventory = Self::for_config(config, false);
        for spec in &mut inventory.specs {
            if let Some(target) = spec.target.strip_prefix("model.") {
                spec.target = target.to_owned();
            }
        }
        inventory
    }

    /// Construct the complete 749-tensor base-model or 750-tensor causal-LM inventory for the
    /// released 36-layer/27-block Qwen3-VL shape. Targets use the conditional-generation record
    /// root (`model.language_model`, `model.visual`, and optional `lm_head`).
    pub fn for_config(config: &Qwen3VlConfig, include_lm_head: bool) -> Self {
        let text = &config.text_config;
        let vision = &config.vision_config;
        let mut specs = Vec::new();
        push(
            &mut specs,
            "model.language_model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
            [text.vocab_size, text.hidden_size],
            WeightRole::Embedding,
        );
        for layer in 0..text.num_hidden_layers {
            let source = format!("model.language_model.layers.{layer}");
            let target = source.clone();
            norm_scale(
                &mut specs,
                &source,
                &target,
                "input_layernorm",
                text.hidden_size,
            );
            norm_scale(
                &mut specs,
                &source,
                &target,
                "post_attention_layernorm",
                text.hidden_size,
            );
            for (projection, output) in [
                ("q_proj", text.num_attention_heads * text.head_dim()),
                ("k_proj", text.num_key_value_heads * text.head_dim()),
                ("v_proj", text.num_key_value_heads * text.head_dim()),
                ("o_proj", text.hidden_size),
            ] {
                push(
                    &mut specs,
                    &format!("{source}.self_attn.{projection}.weight"),
                    &format!("{target}.self_attn.{projection}.weight"),
                    [output, text.hidden_size],
                    WeightRole::LinearWeight,
                );
            }
            for normalization in ["q_norm", "k_norm"] {
                norm_scale_nested(
                    &mut specs,
                    &format!("{source}.self_attn"),
                    &format!("{target}.self_attn"),
                    normalization,
                    text.head_dim(),
                );
            }
            for projection in ["gate_proj", "up_proj"] {
                push(
                    &mut specs,
                    &format!("{source}.mlp.{projection}.weight"),
                    &format!("{target}.mlp.{projection}.weight"),
                    [text.intermediate_size, text.hidden_size],
                    WeightRole::LinearWeight,
                );
            }
            push(
                &mut specs,
                &format!("{source}.mlp.down_proj.weight"),
                &format!("{target}.mlp.down_proj.weight"),
                [text.hidden_size, text.intermediate_size],
                WeightRole::LinearWeight,
            );
        }
        norm_scale(
            &mut specs,
            "model.language_model",
            "model.language_model",
            "norm",
            text.hidden_size,
        );

        push(
            &mut specs,
            "model.visual.patch_embed.proj.weight",
            "model.visual.patch_embed.proj.weight",
            [
                vision.hidden_size,
                vision.in_channels,
                vision.temporal_patch_size,
                vision.patch_size,
                vision.patch_size,
            ],
            WeightRole::ConvolutionWeight,
        );
        push(
            &mut specs,
            "model.visual.patch_embed.proj.bias",
            "model.visual.patch_embed.proj.bias",
            [vision.hidden_size],
            WeightRole::ConvolutionBias,
        );
        push(
            &mut specs,
            "model.visual.pos_embed.weight",
            "model.visual.pos_embed.weight",
            [vision.num_position_embeddings, vision.hidden_size],
            WeightRole::Embedding,
        );
        for block in 0..vision.depth {
            let source = format!("model.visual.blocks.{block}");
            let target = source.clone();
            layer_norm(&mut specs, &source, &target, "norm1", vision.hidden_size);
            layer_norm(&mut specs, &source, &target, "norm2", vision.hidden_size);
            linear_with_bias(
                &mut specs,
                &format!("{source}.attn.qkv"),
                &format!("{target}.attn.qkv"),
                3 * vision.hidden_size,
                vision.hidden_size,
            );
            linear_with_bias(
                &mut specs,
                &format!("{source}.attn.proj"),
                &format!("{target}.attn.proj"),
                vision.hidden_size,
                vision.hidden_size,
            );
            linear_with_bias(
                &mut specs,
                &format!("{source}.mlp.linear_fc1"),
                &format!("{target}.mlp.linear_fc1"),
                vision.intermediate_size,
                vision.hidden_size,
            );
            linear_with_bias(
                &mut specs,
                &format!("{source}.mlp.linear_fc2"),
                &format!("{target}.mlp.linear_fc2"),
                vision.hidden_size,
                vision.intermediate_size,
            );
        }
        merger_specs(
            &mut specs,
            "model.visual.merger",
            vision.hidden_size,
            vision.out_hidden_size,
            vision.spatial_merge_size,
            false,
        );
        for index in 0..vision.deepstack_visual_indexes.len() {
            merger_specs(
                &mut specs,
                &format!("model.visual.deepstack_merger_list.{index}"),
                vision.hidden_size,
                vision.out_hidden_size,
                vision.spatial_merge_size,
                true,
            );
        }
        if include_lm_head && !config.tie_word_embeddings {
            push(
                &mut specs,
                "lm_head.weight",
                "lm_head.weight",
                [text.vocab_size, text.hidden_size],
                WeightRole::LinearWeight,
            );
        }
        Self { specs }
    }

    pub fn specs(&self) -> &[WeightSpec] {
        &self.specs
    }

    pub fn source_to_target(&self, source: &str) -> Option<&str> {
        self.specs
            .iter()
            .find(|spec| spec.source == source)
            .map(|spec| spec.target.as_str())
    }

    pub fn by_source(&self) -> BTreeMap<&str, &WeightSpec> {
        self.specs
            .iter()
            .map(|spec| (spec.source.as_str(), spec))
            .collect()
    }

    /// Reject missing, unknown, and duplicate checkpoint keys before allocating model tensors.
    pub fn validate_keys<'a>(&self, keys: impl IntoIterator<Item = &'a str>) -> WeightValidation {
        let expected = self
            .specs
            .iter()
            .map(|spec| spec.source.as_str())
            .collect::<BTreeSet<_>>();
        let mut observed = BTreeSet::new();
        let mut duplicate = BTreeSet::new();
        for key in keys {
            if !observed.insert(key) {
                duplicate.insert(key);
            }
        }
        WeightValidation {
            missing: expected
                .difference(&observed)
                .map(|value| (*value).to_owned())
                .collect(),
            unknown: observed
                .difference(&expected)
                .map(|value| (*value).to_owned())
                .collect(),
            duplicate: duplicate.into_iter().map(str::to_owned).collect(),
            shape_mismatches: Vec::new(),
        }
    }

    /// Validate names and logical shapes before record conversion or device allocation.
    pub fn validate_entries<'a>(
        &self,
        entries: impl IntoIterator<Item = (&'a str, &'a [usize])>,
    ) -> WeightValidation {
        let entries = entries.into_iter().collect::<Vec<_>>();
        let mut validation = self.validate_keys(entries.iter().map(|(key, _)| *key));
        let expected = self.by_source();
        validation.shape_mismatches = entries
            .into_iter()
            .filter_map(|(key, actual)| {
                let spec = expected.get(key)?;
                (spec.shape.as_slice() != actual).then(|| WeightShapeMismatch {
                    key: key.to_owned(),
                    expected: spec.shape.clone(),
                    actual: actual.to_vec(),
                })
            })
            .collect();
        validation
    }
}

fn push<const D: usize>(
    specs: &mut Vec<WeightSpec>,
    source: &str,
    target: &str,
    shape: [usize; D],
    role: WeightRole,
) {
    specs.push(WeightSpec {
        source: source.into(),
        target: target.into(),
        shape: shape.to_vec(),
        role,
        required: true,
    });
}

fn norm_scale(specs: &mut Vec<WeightSpec>, source: &str, target: &str, name: &str, size: usize) {
    norm_scale_nested(specs, source, target, name, size);
}

fn norm_scale_nested(
    specs: &mut Vec<WeightSpec>,
    source: &str,
    target: &str,
    name: &str,
    size: usize,
) {
    push(
        specs,
        &format!("{source}.{name}.weight"),
        &format!("{target}.{name}.gamma"),
        [size],
        WeightRole::NormalizationScale,
    );
}

fn layer_norm(specs: &mut Vec<WeightSpec>, source: &str, target: &str, name: &str, size: usize) {
    push(
        specs,
        &format!("{source}.{name}.weight"),
        &format!("{target}.{name}.gamma"),
        [size],
        WeightRole::NormalizationScale,
    );
    push(
        specs,
        &format!("{source}.{name}.bias"),
        &format!("{target}.{name}.beta"),
        [size],
        WeightRole::NormalizationBias,
    );
}

fn linear_with_bias(
    specs: &mut Vec<WeightSpec>,
    source: &str,
    target: &str,
    output: usize,
    input: usize,
) {
    push(
        specs,
        &format!("{source}.weight"),
        &format!("{target}.weight"),
        [output, input],
        WeightRole::LinearWeight,
    );
    push(
        specs,
        &format!("{source}.bias"),
        &format!("{target}.bias"),
        [output],
        WeightRole::LinearBias,
    );
}

fn merger_specs(
    specs: &mut Vec<WeightSpec>,
    path: &str,
    hidden: usize,
    output: usize,
    merge: usize,
    postshuffle_norm: bool,
) {
    let merged = hidden * merge * merge;
    layer_norm(
        specs,
        path,
        path,
        "norm",
        if postshuffle_norm { merged } else { hidden },
    );
    linear_with_bias(
        specs,
        &format!("{path}.linear_fc1"),
        &format!("{path}.linear_fc1"),
        merged,
        merged,
    );
    linear_with_bias(
        specs,
        &format!("{path}.linear_fc2"),
        &format!("{path}.linear_fc2"),
        output,
        merged,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::tiny_config;

    #[test]
    fn released_checkpoint_has_750_tensors_correctness() {
        let mut config = tiny_config();
        config.text_config.num_hidden_layers = 36;
        config.vision_config.depth = 27;
        config.vision_config.deepstack_visual_indexes = vec![8, 16, 24];
        let inventory = WeightInventory::for_config(&config, true);
        assert_eq!(inventory.specs().len(), 750);
        assert_eq!(
            inventory.source_to_target("model.language_model.layers.0.input_layernorm.weight"),
            Some("model.language_model.layers.0.input_layernorm.gamma")
        );
        let validation =
            inventory.validate_keys(inventory.specs().iter().map(|spec| spec.source.as_str()));
        assert!(validation.is_complete());
    }

    #[test]
    fn duplicate_and_unknown_weights_are_rejected_correctness() {
        let config = tiny_config();
        let inventory = WeightInventory::for_config(&config, false);
        let first = inventory.specs()[0].source.as_str();
        let validation = inventory.validate_keys([first, first, "unknown.weight"]);
        assert_eq!(validation.duplicate, [first]);
        assert_eq!(validation.unknown, ["unknown.weight"]);
        assert!(!validation.missing.is_empty());
    }

    #[test]
    fn incompatible_shape_is_rejected_correctness() {
        let inventory = WeightInventory::for_config(&tiny_config(), true);
        let entries = inventory
            .specs()
            .iter()
            .map(|spec| (spec.source.as_str(), spec.shape.as_slice()))
            .collect::<Vec<_>>();
        let mut altered_shapes = entries
            .iter()
            .map(|(key, shape)| ((*key).to_owned(), (*shape).to_vec()))
            .collect::<Vec<_>>();
        altered_shapes[0].1[0] += 1;
        let validation = inventory.validate_entries(
            altered_shapes
                .iter()
                .map(|(key, shape)| (key.as_str(), shape.as_slice())),
        );
        assert_eq!(validation.shape_mismatches.len(), 1);
        assert_eq!(validation.shape_mismatches[0].key, altered_shapes[0].0);
    }

    #[test]
    fn base_model_targets_drop_conditional_wrapper_correctness() {
        let inventory = WeightInventory::for_base_model(&tiny_config());
        assert_eq!(
            inventory.source_to_target("model.language_model.embed_tokens.weight"),
            Some("language_model.embed_tokens.weight")
        );
        assert_eq!(
            inventory.source_to_target("model.visual.patch_embed.proj.weight"),
            Some("visual.patch_embed.proj.weight")
        );
    }
}
