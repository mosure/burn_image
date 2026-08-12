use burn::config::Config;
use serde::{Deserialize, Serialize};
use thiserror::Error;

const DOWN_BLOCK: &str = "DownEncoderBlock2D";
const UP_BLOCK: &str = "UpDecoderBlock2D";

/// Burn configuration for the ordinary Diffusers `AutoencoderKL` used by FLUX.1.
///
/// [`Self::flux1`] matches the public FLUX.1 VAE configuration. The architecture is deliberately
/// limited to ordinary `DownEncoderBlock2D` and `UpDecoderBlock2D` blocks with SiLU activation;
/// model-specific conditioning and orchestration do not belong in this crate.
#[derive(Config, Debug, PartialEq)]
pub struct AutoencoderKlConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub latent_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub norm_num_groups: usize,
    pub norm_epsilon: f64,
    pub scaling_factor: f64,
    pub shift_factor: f64,
    pub sample_size: usize,
    pub force_upcast: bool,
    pub use_quant_conv: bool,
    pub use_post_quant_conv: bool,
    pub mid_block_add_attention: bool,
    /// Maximum number of query positions submitted to one attention invocation.
    ///
    /// This bounds fallback attention memory to `query_chunk_size * key_length` rather than a
    /// full spatial sequence square. It does not change the learned architecture or weights.
    pub attention_query_chunk_size: usize,
}

impl AutoencoderKlConfig {
    /// Canonical FLUX.1 Dev/Schnell AutoencoderKL configuration.
    pub fn flux1() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 16,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            norm_num_groups: 32,
            norm_epsilon: 1.0e-6,
            scaling_factor: 0.3611,
            shift_factor: 0.1159,
            sample_size: 1024,
            force_upcast: true,
            use_quant_conv: false,
            use_post_quant_conv: false,
            mid_block_add_attention: true,
            attention_query_chunk_size: 512,
        }
    }

    /// Small valid configuration intended for deterministic unit and integration tests.
    pub fn tiny() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 4,
            block_out_channels: vec![8, 16],
            layers_per_block: 1,
            norm_num_groups: 4,
            norm_epsilon: 1.0e-6,
            scaling_factor: 0.5,
            shift_factor: 0.25,
            sample_size: 8,
            force_upcast: true,
            use_quant_conv: true,
            use_post_quant_conv: true,
            mid_block_add_attention: true,
            attention_query_chunk_size: 8,
        }
    }

    /// Validate all architectural and numerical invariants before allocating model parameters.
    pub fn validate(&self) -> Result<(), AutoencoderKlConfigError> {
        if self.in_channels == 0 || self.out_channels == 0 || self.latent_channels == 0 {
            return Err(AutoencoderKlConfigError::ZeroChannels);
        }
        if self.block_out_channels.is_empty() {
            return Err(AutoencoderKlConfigError::EmptyBlocks);
        }
        if self.layers_per_block == 0 {
            return Err(AutoencoderKlConfigError::ZeroLayers);
        }
        if self.norm_num_groups == 0 {
            return Err(AutoencoderKlConfigError::ZeroNormGroups);
        }
        for &channels in &self.block_out_channels {
            if channels == 0 || !channels.is_multiple_of(self.norm_num_groups) {
                return Err(AutoencoderKlConfigError::InvalidNormChannels {
                    channels,
                    groups: self.norm_num_groups,
                });
            }
        }
        if !self.norm_epsilon.is_finite() || self.norm_epsilon <= 0.0 {
            return Err(AutoencoderKlConfigError::InvalidNormEpsilon(
                self.norm_epsilon,
            ));
        }
        if !self.scaling_factor.is_finite() || self.scaling_factor == 0.0 {
            return Err(AutoencoderKlConfigError::InvalidScalingFactor(
                self.scaling_factor,
            ));
        }
        if !self.shift_factor.is_finite() {
            return Err(AutoencoderKlConfigError::InvalidShiftFactor(
                self.shift_factor,
            ));
        }
        if self.sample_size == 0 {
            return Err(AutoencoderKlConfigError::ZeroSampleSize);
        }
        if self.attention_query_chunk_size == 0 {
            return Err(AutoencoderKlConfigError::ZeroAttentionChunk);
        }
        Ok(())
    }

    /// Spatial reduction between image and latent maps.
    pub fn spatial_compression_factor(&self) -> usize {
        1usize << self.block_out_channels.len().saturating_sub(1)
    }

    /// Number of encoder output channels containing concatenated mean and log-variance.
    pub fn moment_channels(&self) -> usize {
        self.latent_channels * 2
    }

    /// Parse and strictly validate an ordinary Diffusers AutoencoderKL JSON configuration.
    pub fn from_diffusers_json(json: &str) -> Result<Self, AutoencoderKlConfigError> {
        let source: DiffusersAutoencoderKlConfig = serde_json::from_str(json)?;
        source.try_into()
    }
}

impl Default for AutoencoderKlConfig {
    fn default() -> Self {
        Self::flux1()
    }
}

/// Serialization surface for upstream Diffusers `config.json` files.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiffusersAutoencoderKlConfig {
    #[serde(default = "default_class_name", rename = "_class_name")]
    pub class_name: String,
    #[serde(default = "default_act_fn")]
    pub act_fn: String,
    #[serde(default = "default_block_out_channels")]
    pub block_out_channels: Vec<usize>,
    #[serde(default = "default_down_block_types")]
    pub down_block_types: Vec<String>,
    #[serde(default = "default_up_block_types")]
    pub up_block_types: Vec<String>,
    #[serde(default = "default_true")]
    pub force_upcast: bool,
    #[serde(default = "default_three")]
    pub in_channels: usize,
    #[serde(default = "default_latent_channels")]
    pub latent_channels: usize,
    #[serde(default = "default_layers_per_block")]
    pub layers_per_block: usize,
    #[serde(default = "default_true")]
    pub mid_block_add_attention: bool,
    #[serde(default = "default_norm_groups")]
    pub norm_num_groups: usize,
    #[serde(default = "default_three")]
    pub out_channels: usize,
    #[serde(default = "default_sample_size")]
    pub sample_size: usize,
    #[serde(default = "default_scaling_factor")]
    pub scaling_factor: f64,
    #[serde(default = "default_shift_factor")]
    pub shift_factor: Option<f64>,
    #[serde(default)]
    pub latents_mean: Option<Vec<f64>>,
    #[serde(default)]
    pub latents_std: Option<Vec<f64>>,
    #[serde(default = "default_true")]
    pub use_quant_conv: bool,
    #[serde(default = "default_true")]
    pub use_post_quant_conv: bool,
}

impl TryFrom<DiffusersAutoencoderKlConfig> for AutoencoderKlConfig {
    type Error = AutoencoderKlConfigError;

    fn try_from(source: DiffusersAutoencoderKlConfig) -> Result<Self, Self::Error> {
        if source.class_name != "AutoencoderKL" {
            return Err(AutoencoderKlConfigError::UnsupportedClass(
                source.class_name,
            ));
        }
        if !matches!(source.act_fn.as_str(), "silu" | "swish") {
            return Err(AutoencoderKlConfigError::UnsupportedActivation(
                source.act_fn,
            ));
        }
        if source.down_block_types.len() != source.block_out_channels.len()
            || source
                .down_block_types
                .iter()
                .any(|kind| kind != DOWN_BLOCK)
        {
            return Err(AutoencoderKlConfigError::UnsupportedDownBlocks(
                source.down_block_types,
            ));
        }
        if source.up_block_types.len() != source.block_out_channels.len()
            || source.up_block_types.iter().any(|kind| kind != UP_BLOCK)
        {
            return Err(AutoencoderKlConfigError::UnsupportedUpBlocks(
                source.up_block_types,
            ));
        }
        if source.latents_mean.is_some() || source.latents_std.is_some() {
            return Err(AutoencoderKlConfigError::UnsupportedLatentStatistics);
        }

        let config = Self {
            in_channels: source.in_channels,
            out_channels: source.out_channels,
            latent_channels: source.latent_channels,
            block_out_channels: source.block_out_channels,
            layers_per_block: source.layers_per_block,
            norm_num_groups: source.norm_num_groups,
            norm_epsilon: 1.0e-6,
            scaling_factor: source.scaling_factor,
            shift_factor: source.shift_factor.unwrap_or(0.0),
            sample_size: source.sample_size,
            force_upcast: source.force_upcast,
            use_quant_conv: source.use_quant_conv,
            use_post_quant_conv: source.use_post_quant_conv,
            mid_block_add_attention: source.mid_block_add_attention,
            attention_query_chunk_size: 512,
        };
        config.validate()?;
        Ok(config)
    }
}

#[derive(Debug, Error)]
pub enum AutoencoderKlConfigError {
    #[error("failed to parse Diffusers AutoencoderKL config: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported Diffusers class '{0}', expected AutoencoderKL")]
    UnsupportedClass(String),
    #[error("unsupported VAE activation '{0}', expected silu/swish")]
    UnsupportedActivation(String),
    #[error("unsupported down block sequence {0:?}")]
    UnsupportedDownBlocks(Vec<String>),
    #[error("unsupported up block sequence {0:?}")]
    UnsupportedUpBlocks(Vec<String>),
    #[error(
        "per-channel latents_mean/latents_std normalization is not a FLUX.1 scale/shift contract"
    )]
    UnsupportedLatentStatistics,
    #[error("input, output, and latent channel counts must all be non-zero")]
    ZeroChannels,
    #[error("block_out_channels must not be empty")]
    EmptyBlocks,
    #[error("layers_per_block must be non-zero")]
    ZeroLayers,
    #[error("norm_num_groups must be non-zero")]
    ZeroNormGroups,
    #[error(
        "channel count {channels} must be non-zero and divisible by {groups} normalization groups"
    )]
    InvalidNormChannels { channels: usize, groups: usize },
    #[error("norm epsilon must be finite and positive, got {0}")]
    InvalidNormEpsilon(f64),
    #[error("scaling factor must be finite and non-zero, got {0}")]
    InvalidScalingFactor(f64),
    #[error("shift factor must be finite, got {0}")]
    InvalidShiftFactor(f64),
    #[error("sample_size must be non-zero")]
    ZeroSampleSize,
    #[error("attention_query_chunk_size must be non-zero")]
    ZeroAttentionChunk,
}

fn default_class_name() -> String {
    "AutoencoderKL".to_string()
}
fn default_act_fn() -> String {
    "silu".to_string()
}
fn default_block_out_channels() -> Vec<usize> {
    vec![64]
}
fn default_down_block_types() -> Vec<String> {
    vec![DOWN_BLOCK.to_string()]
}
fn default_up_block_types() -> Vec<String> {
    vec![UP_BLOCK.to_string()]
}
fn default_true() -> bool {
    true
}
fn default_three() -> usize {
    3
}
fn default_latent_channels() -> usize {
    4
}
fn default_layers_per_block() -> usize {
    1
}
fn default_norm_groups() -> usize {
    32
}
fn default_sample_size() -> usize {
    32
}
fn default_scaling_factor() -> f64 {
    0.18215
}
fn default_shift_factor() -> Option<f64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flux1_config_correctness() {
        let config = AutoencoderKlConfig::flux1();
        config.validate().expect("FLUX.1 config");
        assert_eq!(config.block_out_channels, [128, 256, 512, 512]);
        assert_eq!(config.moment_channels(), 32);
        assert_eq!(config.spatial_compression_factor(), 8);
        assert_eq!(config.scaling_factor, 0.3611);
        assert_eq!(config.shift_factor, 0.1159);
        assert!(!config.use_quant_conv);
        assert!(!config.use_post_quant_conv);
    }

    #[test]
    fn diffusers_flux_config_parity() {
        let json = r#"{
            "_class_name":"AutoencoderKL",
            "act_fn":"silu",
            "block_out_channels":[128,256,512,512],
            "down_block_types":["DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D"],
            "up_block_types":["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
            "force_upcast":true,
            "in_channels":3,
            "latent_channels":16,
            "latents_mean":null,
            "latents_std":null,
            "layers_per_block":2,
            "mid_block_add_attention":true,
            "norm_num_groups":32,
            "out_channels":3,
            "sample_size":1024,
            "scaling_factor":0.3611,
            "shift_factor":0.1159,
            "use_post_quant_conv":false,
            "use_quant_conv":false
        }"#;
        let parsed = AutoencoderKlConfig::from_diffusers_json(json).expect("parse FLUX config");
        assert_eq!(parsed, AutoencoderKlConfig::flux1());
    }
}
