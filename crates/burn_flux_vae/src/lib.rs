//! Burn implementation of the ordinary Diffusers `AutoencoderKL` architecture used by FLUX.1.
//!
//! This crate owns only reusable VAE tensor math and weight loading. Image resizing, model
//! orchestration, schedulers, prompts, and application integration belong in higher-level crates.

#[cfg(feature = "artifacts")]
mod artifacts;
mod blocks;
mod config;
mod decoder;
mod distribution;
mod encoder;
mod inventory;
#[cfg(any(feature = "import", feature = "artifacts"))]
mod loading;
mod model;

#[cfg(feature = "artifacts")]
pub use artifacts::{
    AsyncFluxVaeStageSource, FLUX_VAE_COMPONENT_BUNDLE_ID, FLUX_VAE_COMPONENT_CONTENT_DIGEST,
    FLUX_VAE_COMPONENT_MODEL_ID, FLUX_VAE_COMPONENT_MODEL_REVISION, FLUX_VAE_COMPONENT_PROFILE,
    FLUX_VAE_COMPONENT_ROLE, FLUX_VAE_DECODER_STAGE, FLUX_VAE_ENCODER_STAGE, FluxVaeArtifactError,
    FluxVaeArtifactFloatPolicy, FluxVaeComponentContract, FluxVaeStageSource,
    RetainingAsyncFluxVaeStageSource, RetainingFluxVaeStageSource,
    VerifiedAsyncBurnpackFluxVaeStageSource, VerifiedBurnpackFluxVaeStageSource,
    flux_vae_component_dependency,
};
pub use blocks::{
    AttentionBlock, DecoderGroupNormPolicy, Downsample2d, MidBlock2d, ResnetBlock2d, Upsample2d,
};
pub use config::{AutoencoderKlConfig, AutoencoderKlConfigError, DiffusersAutoencoderKlConfig};
pub use decoder::{Decoder, UpDecoderBlock2d};
pub use distribution::DiagonalGaussian;
pub use encoder::{DownEncoderBlock2d, Encoder};
pub use inventory::{TensorInventory, TensorSpec};
#[cfg(any(feature = "import", feature = "artifacts"))]
pub use loading::{
    BurnpackShardLoader, LoadError, LoadOptions, LoadReport, apply_burnpack_part_bytes,
    diffusers_key_remap_rules, load_burnpack_file, load_burnpack_file_with_options,
    load_safetensors_file, load_safetensors_file_with_options, save_burnpack_file,
};
pub use model::AutoencoderKl;

/// Familiar Diffusers spelling for [`AutoencoderKl`].
pub type AutoencoderKL<B> = AutoencoderKl<B>;
/// Explicit FLUX-oriented spelling for [`AutoencoderKlConfig`].
pub type FluxVaeConfig = AutoencoderKlConfig;
/// Explicit FLUX-oriented spelling for [`AutoencoderKl`].
pub type FluxAutoencoderKl<B> = AutoencoderKl<B>;
/// Familiar Diffusers spelling for [`DiagonalGaussian`].
pub type DiagonalGaussianDistribution<B> = DiagonalGaussian<B>;
