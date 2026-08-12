use serde::{Deserialize, Serialize};

use crate::{Qwen3VlError, Result};

fn one_million() -> usize {
    1_000_000
}
fn rope_theta() -> f64 {
    5_000_000.0
}
fn rms_epsilon() -> f64 {
    1e-6
}
fn layer_norm_epsilon() -> f64 {
    1e-6
}
fn hidden_act() -> String {
    "silu".into()
}
fn vision_hidden_act() -> String {
    "gelu_pytorch_tanh".into()
}
fn channels() -> usize {
    3
}
fn temporal_patch_size() -> usize {
    2
}
fn spatial_merge_size() -> usize {
    2
}
fn position_embeddings() -> usize {
    2304
}
fn mrope_section() -> [usize; 3] {
    [24, 20, 20]
}
fn true_value() -> bool {
    true
}

/// Text decoder configuration. Field names intentionally match Hugging Face `config.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen3VlTextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default = "hidden_act")]
    pub hidden_act: String,
    #[serde(default = "rms_epsilon")]
    pub rms_norm_eps: f64,
    #[serde(default = "one_million")]
    pub max_position_embeddings: usize,
    #[serde(default = "rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_scaling: Option<MropeConfig>,
    #[serde(default)]
    pub rope_parameters: Option<MropeConfig>,
}

/// Multimodal rotary layout stored under `rope_scaling` by released Qwen3-VL checkpoints.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MropeConfig {
    #[serde(default = "mrope_section")]
    pub mrope_section: [usize; 3],
    #[serde(default = "true_value")]
    pub mrope_interleaved: bool,
    #[serde(default, alias = "type")]
    pub rope_type: Option<String>,
}

impl Default for MropeConfig {
    fn default() -> Self {
        Self {
            mrope_section: mrope_section(),
            mrope_interleaved: true,
            rope_type: Some("default".into()),
        }
    }
}

impl Qwen3VlTextConfig {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn mrope(&self) -> MropeConfig {
        self.rope_parameters
            .clone()
            .or_else(|| self.rope_scaling.clone())
            .unwrap_or_default()
    }

    pub fn validate(&self) -> Result<()> {
        if self.vocab_size == 0 || self.hidden_size == 0 || self.intermediate_size == 0 {
            return Err(Qwen3VlError::InvalidConfig(
                "text dimensions must be non-zero".into(),
            ));
        }
        if self.num_attention_heads == 0 || self.num_key_value_heads == 0 {
            return Err(Qwen3VlError::InvalidConfig(
                "attention head counts must be non-zero".into(),
            ));
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(Qwen3VlError::InvalidConfig(
                "num_attention_heads must be divisible by num_key_value_heads".into(),
            ));
        }
        if self.head_dim() * self.num_attention_heads != self.hidden_size {
            return Err(Qwen3VlError::InvalidConfig(
                "head_dim * num_attention_heads must equal hidden_size".into(),
            ));
        }
        if !self.head_dim().is_multiple_of(2) {
            return Err(Qwen3VlError::InvalidConfig("head_dim must be even".into()));
        }
        let rope = self.mrope();
        if !rope.mrope_interleaved {
            return Err(Qwen3VlError::InvalidConfig(
                "only interleaved Qwen3-VL MRoPE is supported".into(),
            ));
        }
        if rope.mrope_section.iter().sum::<usize>() != self.head_dim() / 2 {
            return Err(Qwen3VlError::InvalidConfig(format!(
                "mrope_section must sum to head_dim / 2 ({})",
                self.head_dim() / 2
            )));
        }
        if self.hidden_act != "silu" {
            return Err(Qwen3VlError::InvalidConfig(format!(
                "unsupported text activation {:?}; expected silu",
                self.hidden_act
            )));
        }
        Ok(())
    }
}

/// Vision encoder configuration. Field names intentionally match Hugging Face `config.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen3VlVisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub patch_size: usize,
    #[serde(default = "temporal_patch_size")]
    pub temporal_patch_size: usize,
    #[serde(default = "spatial_merge_size")]
    pub spatial_merge_size: usize,
    pub out_hidden_size: usize,
    #[serde(default = "channels")]
    pub in_channels: usize,
    #[serde(default = "position_embeddings")]
    pub num_position_embeddings: usize,
    #[serde(default)]
    pub deepstack_visual_indexes: Vec<usize>,
    #[serde(default = "vision_hidden_act")]
    pub hidden_act: String,
    #[serde(default = "layer_norm_epsilon")]
    pub layer_norm_eps: f64,
}

impl Qwen3VlVisionConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    pub fn patch_volume(&self) -> usize {
        self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }

    pub fn validate(&self) -> Result<()> {
        if self.depth == 0
            || self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_heads == 0
            || self.patch_size == 0
            || self.temporal_patch_size == 0
            || self.spatial_merge_size == 0
        {
            return Err(Qwen3VlError::InvalidConfig(
                "vision dimensions must be non-zero".into(),
            ));
        }
        if !self.hidden_size.is_multiple_of(self.num_heads) || !self.head_dim().is_multiple_of(2) {
            return Err(Qwen3VlError::InvalidConfig(
                "vision hidden_size must divide evenly into even-sized heads".into(),
            ));
        }
        let side = (self.num_position_embeddings as f64).sqrt() as usize;
        if side * side != self.num_position_embeddings {
            return Err(Qwen3VlError::InvalidConfig(
                "num_position_embeddings must be a perfect square".into(),
            ));
        }
        if self
            .deepstack_visual_indexes
            .iter()
            .any(|&index| index >= self.depth)
        {
            return Err(Qwen3VlError::InvalidConfig(
                "deepstack_visual_indexes must address existing vision blocks".into(),
            ));
        }
        if self.hidden_act != "gelu_pytorch_tanh" {
            return Err(Qwen3VlError::InvalidConfig(format!(
                "unsupported vision activation {:?}; expected gelu_pytorch_tanh",
                self.hidden_act
            )));
        }
        Ok(())
    }
}

/// Top-level ordinary Qwen3-VL model configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen3VlConfig {
    pub text_config: Qwen3VlTextConfig,
    pub vision_config: Qwen3VlVisionConfig,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub image_token_id: usize,
    pub video_token_id: usize,
    pub vision_start_token_id: usize,
    pub vision_end_token_id: usize,
}

impl Qwen3VlConfig {
    pub fn from_json(json: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(json)?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        self.text_config.validate()?;
        self.vision_config.validate()?;
        if self.text_config.hidden_size != self.vision_config.out_hidden_size {
            return Err(Qwen3VlError::InvalidConfig(
                "vision out_hidden_size must equal text hidden_size".into(),
            ));
        }
        for (name, id) in [
            ("image_token_id", self.image_token_id),
            ("video_token_id", self.video_token_id),
            ("vision_start_token_id", self.vision_start_token_id),
            ("vision_end_token_id", self.vision_end_token_id),
        ] {
            if id >= self.text_config.vocab_size {
                return Err(Qwen3VlError::InvalidConfig(format!(
                    "{name} is outside vocab_size"
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) fn tiny_config() -> Qwen3VlConfig {
    Qwen3VlConfig {
        text_config: Qwen3VlTextConfig {
            vocab_size: 64,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: Some(4),
            hidden_act: "silu".into(),
            rms_norm_eps: 1e-6,
            max_position_embeddings: 128,
            rope_theta: 10_000.0,
            rope_scaling: Some(MropeConfig {
                mrope_section: [2, 0, 0],
                mrope_interleaved: true,
                rope_type: Some("default".into()),
            }),
            rope_parameters: None,
        },
        vision_config: Qwen3VlVisionConfig {
            depth: 1,
            hidden_size: 8,
            intermediate_size: 16,
            num_heads: 2,
            patch_size: 2,
            temporal_patch_size: 1,
            spatial_merge_size: 2,
            out_hidden_size: 8,
            in_channels: 3,
            num_position_embeddings: 16,
            deepstack_visual_indexes: vec![0],
            hidden_act: "gelu_pytorch_tanh".into(),
            layer_norm_eps: 1e-6,
        },
        tie_word_embeddings: false,
        image_token_id: 60,
        video_token_id: 61,
        vision_start_token_id: 62,
        vision_end_token_id: 63,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_shape_config_correctness() {
        let json = r#"{
          "text_config": {
            "vocab_size":151936,"hidden_size":4096,"intermediate_size":12288,
            "num_hidden_layers":36,"num_attention_heads":32,"num_key_value_heads":8,
            "head_dim":128,"hidden_act":"silu","rms_norm_eps":1e-6,
            "max_position_embeddings":262144,"rope_theta":5000000,
            "rope_scaling":{"mrope_section":[24,20,20],"mrope_interleaved":true,"rope_type":"default"}
          },
          "vision_config": {
            "depth":27,"hidden_size":1152,"intermediate_size":4304,"num_heads":16,
            "patch_size":16,"temporal_patch_size":2,"spatial_merge_size":2,
            "out_hidden_size":4096,"in_channels":3,"num_position_embeddings":2304,
            "deepstack_visual_indexes":[8,16,24],"hidden_act":"gelu_pytorch_tanh"
          },
          "tie_word_embeddings":false,"image_token_id":151655,"video_token_id":151656,
          "vision_start_token_id":151652,"vision_end_token_id":151653
        }"#;
        let config = Qwen3VlConfig::from_json(json).unwrap();
        assert_eq!(config.text_config.head_dim(), 128);
        assert_eq!(config.text_config.mrope().mrope_section, [24, 20, 20]);
        assert_eq!(config.vision_config.patch_volume(), 1536);
    }

    #[test]
    fn invalid_grouped_query_attention_is_rejected_correctness() {
        let mut config = tiny_config();
        config.text_config.num_key_value_heads = 3;
        assert!(config.validate().is_err());
    }
}
