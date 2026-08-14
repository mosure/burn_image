//! Boogu denoiser modules.

mod attention;
mod block;
mod denoiser;
mod embedding;
mod feed_forward;
pub(crate) mod linear;
#[cfg(all(
    any(feature = "wgpu", feature = "cuda-experimental"),
    not(target_arch = "wasm32")
))]
mod native_flash;
mod norm;
#[cfg(feature = "wgpu")]
pub mod packed_f16;
mod streaming;

pub use attention::{DoubleStreamAttention, GqaAttention};
pub use block::{DoubleStreamBlock, SingleStreamBlock};
#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
pub(crate) use denoiser::BooguRoPeGeometry;
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
#[cfg(feature = "wgpu")]
pub use packed_f16::{
    MaterializedF32Object, PACKED_F16_F32_VIEW_ALIGNMENT_BYTES,
    PACKED_F16_F32_VIEW_ALIGNMENT_ELEMENTS, PACKED_F16_MAX_BUFFER_BYTES, PackedF16Error,
    PackedF16Layout, PackedF16Object, PackedF16TensorLayout, align_packed_f16_f32_view_offset,
    materialize_packed_f16_object, materialize_packed_f16_objects,
};
#[cfg(feature = "burnpack")]
pub use streaming::RetainedDenoiserDTypeAudit;
pub use streaming::{
    AsyncBooguDenoiserStageSource, AsyncRetainingDenoiserSynchronizationPolicy,
    BooguDenoiserPrelude, BooguDenoiserTail, BooguQuantizedLinearExecutionPolicy, BooguStreamState,
    DenoiserStageObserver, RetainingAsyncBooguDenoiserStageSource, StreamingBooguDenoiser,
    StreamingStageSource,
};
