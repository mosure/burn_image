//! Boogu denoiser modules.

mod attention;
mod block;
mod denoiser;
mod embedding;
mod feed_forward;
#[cfg(all(
    any(feature = "wgpu", feature = "cuda-experimental"),
    not(target_arch = "wasm32")
))]
mod native_flash;
mod norm;
mod streaming;

pub use attention::{DoubleStreamAttention, GqaAttention};
pub use block::{DoubleStreamBlock, SingleStreamBlock};
pub use denoiser::{BooguDenoiser, BooguDenoiserInput};
pub use embedding::{CombinedTimestepCaptionEmbedding, FinalProjection};
pub use feed_forward::LuminaFeedForward;
#[cfg(all(feature = "cuda-experimental", not(target_arch = "wasm32")))]
pub(crate) use native_flash::assert_supported_blackbox_configuration;
#[cfg(all(feature = "cuda-experimental", not(target_arch = "wasm32")))]
pub use native_flash::{
    NativeCudaBackend, required_chunked_cuda_flash_unit_attention,
    required_chunked_cuda_padded_blackbox_attention,
    required_chunked_cuda_padded_blackbox_attention_tiled, required_cuda_flash_unit_attention,
};
#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
pub use native_flash::{
    NativeWgpuBackend, required_chunked_flash_unit_attention,
    required_chunked_padded_blackbox_attention, required_chunked_padded_blackbox_attention_tiled,
    required_flash_unit_attention,
};
#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
pub(crate) use native_flash::{
    assert_supported_wgpu_blackbox_configuration,
    assert_supported_wgpu_blackbox_partition_configuration,
};
pub use norm::{DenoiserRmsNormPolicy, RmsNormZero};
pub use streaming::{
    AsyncBooguDenoiserStageSource, BooguDenoiserPrelude, BooguDenoiserTail, BooguStreamState,
    DenoiserStageObserver, StreamingBooguDenoiser, StreamingStageSource,
};
