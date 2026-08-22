#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! Boogu-Image Turbo generation and edit pipelines.

/// Pinned release identities and artifact contracts.
pub mod artifacts;
/// Boogu-specific system prompts and instruction policy.
pub mod conditioning;
/// Released denoiser and task configuration.
pub mod config;
/// Canonical release selection and runtime artifact policy.
#[cfg(feature = "burnpack")]
pub mod deployment;
/// Distribution-matching-distillation schedule and update equations.
pub mod dmd;
/// Boogu pipeline errors.
pub mod error;
/// Latent patching and image-size contracts.
pub mod latent;
/// Boogu diffusion-transformer modules.
pub mod model;
#[cfg(all(feature = "burnpack", feature = "wgpu"))]
mod packed_f16_artifacts;
/// End-to-end prepared and resident pipeline orchestration.
pub mod pipeline;
/// Boogu-specific request, image-size, and reference preprocessing policies.
pub mod processing;
/// Authentication of exported upstream numerical fixtures.
#[cfg(feature = "import")]
pub mod reference;
/// Three-axis rotary position construction.
pub mod rope;
/// Model-neutral [`burn_image`] adapter for an already loaded Boogu pipeline.
#[cfg(feature = "runtime")]
pub mod runtime;
/// Frontend-independent browser execution policy.
#[cfg(all(feature = "burnpack", feature = "runtime"))]
pub mod web_policy;
/// Fail-closed selection of a native hardware WGPU adapter for standalone tools.
#[cfg(feature = "wgpu")]
pub mod wgpu_device;

pub use config::{
    BOOGU_1K_NATIVE_POLICY, BOOGU_Q4_1K_NATIVE_POLICY, BooguConfig, BooguTask, BooguVariant,
    EDIT_TURBO_1K5_NATIVE_POLICY, EDIT_TURBO_1K5_Q4_NATIVE_POLICY, NativeAutotunePolicy,
    NativeDenoiserAttentionPolicy, NativeDenoiserAttentionPrecisionPolicy,
    NativeDenoiserQkPreparationPolicy, NativeDenoiserRmsNormPolicy, NativeHighVramPolicy,
    NativeQwenSynchronizationPolicy, NativeVaeExecutionPolicy,
};
pub use dmd::{DmdSchedule, dmd_prediction, dmd_renoise};
pub use error::BooguError;
#[cfg(feature = "burnpack")]
pub use model::RetainedDenoiserDTypeAudit;
pub use model::{
    AsyncBooguDenoiserStageSource, AsyncRetainingDenoiserSynchronizationPolicy, BooguDenoiser,
    BooguDenoiserInput, BooguDenoiserPrelude, BooguDenoiserTail,
    BooguQuantizedLinearExecutionPolicy, BooguStreamState, DenoiserRmsNormPolicy,
    DenoiserStageObserver, DoubleStreamBlock, PORTABLE_ATTENTION_MINIMUM_IMAGE_QUERY_PARTITIONS,
    RetainingAsyncBooguDenoiserStageSource, SingleStreamBlock, StreamingBooguDenoiser,
    StreamingStageSource,
};
#[cfg(feature = "wgpu")]
pub use model::{
    MaterializedF32Object, PACKED_F16_F32_VIEW_ALIGNMENT_BYTES,
    PACKED_F16_F32_VIEW_ALIGNMENT_ELEMENTS, PACKED_F16_MAX_BUFFER_BYTES, PackedF16Error,
    PackedF16Layout, PackedF16Object, PackedF16TensorLayout, align_packed_f16_f32_view_offset,
    materialize_packed_f16_object, materialize_packed_f16_objects,
};
#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
pub use model::{
    NativeWgpuBackend, required_chunked_flash_unit_attention,
    required_chunked_padded_blackbox_attention, required_chunked_padded_blackbox_attention_tiled,
    required_flash_unit_attention,
};
#[cfg(all(feature = "burnpack", feature = "wgpu"))]
pub use packed_f16_artifacts::{
    PackedF16DenoiserBackend, PackedF16DenoiserCacheAudit, PackedF16DenoiserCacheState,
    TURBO_PACKED_F16_ARTIFACT_BYTES, TURBO_PACKED_F16_COMPACT_PAYLOAD_BYTES,
    TURBO_PACKED_F16_F32_WRITE_BYTES_PER_DMD, TURBO_PACKED_F16_FOUR_DMD_READ_BYTES,
    TURBO_PACKED_F16_FOUR_DMD_WRITE_BYTES, TURBO_PACKED_F16_MAX_OBJECT_BYTES,
    TURBO_PACKED_F16_MAX_OBJECT_F32_BYTES, TURBO_PACKED_F16_MAX_STAGE_F32_BYTES,
    TURBO_PACKED_F16_MAX_STAGE_PACKED_BYTES, TURBO_PACKED_F16_OBJECT_COUNT,
    TURBO_PACKED_F16_PADDED_ELEMENTS, TURBO_PACKED_F16_PADDING_ELEMENTS,
    TURBO_PACKED_F16_RETAINED_BYTES, TURBO_PACKED_F16_STAGE_COUNT, TURBO_PACKED_F16_TENSOR_COUNT,
    VerifiedAsyncPackedF16DenoiserStageSource,
};
#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
pub use pipeline::NativeFlashUnitDenoiser;
pub use pipeline::{
    AsyncBooguVaeStageSource, BooguDmdInput, BooguExecution, BooguPipelineOutput,
    BooguVaeStageSource, DmdDenoiser, ResidentBooguInput, ResidentBooguPipeline,
    RetainingAsyncBooguVaeStageSource, RetainingBooguVaeStageSource, StreamingBooguPipeline,
    VaeDecoderMemoryPolicy, encode_instruction, encode_reference, run_dmd, run_dmd_with_observer,
    trim_instruction_features,
};
#[cfg(feature = "burnpack")]
pub use pipeline::{AsyncFluxVaeStageSourceAdapter, FluxVaeStageSourceAdapter};
#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
pub use pipeline::{NativePaddedBlackboxDenoiser, NativePortableDenoiser};
pub use processing::{
    BOOGU_1K5_DEFAULT_EDGE, BOOGU_1K5_MAX_OUTPUT_PIXELS, BOOGU_1K5_MAX_OUTPUT_SIDE,
    BOOGU_1K5_OUTPUT_PRESETS, BOOGU_DEFAULT_EDGE, BOOGU_MAX_OUTPUT_PIXELS, BOOGU_MAX_OUTPUT_SIDE,
    BOOGU_MAX_REFERENCE_PIXELS, BOOGU_MAX_REFERENCE_SIDE, BOOGU_MAX_VLM_PIXELS, BOOGU_MAX_VLM_SIDE,
    PreparedInstruction, ResolvedBooguRequest, boogu_model_descriptor, boogu_processor_config,
    decode_input_image, decoder_output_data_to_host, decoder_output_to_host, prepare_instruction,
    prepare_vae_reference, resize_reference, resolve_request,
};
#[cfg(feature = "runtime")]
pub use runtime::{BooguImageModel, BooguRuntimeDTypes, BooguRuntimeMetadata};
#[cfg(feature = "wgpu")]
pub use wgpu_device::require_native_wgpu_device;
#[cfg(all(feature = "wgpu", feature = "autotune", not(target_arch = "wasm32")))]
pub use wgpu_device::{
    configure_native_autotune, configure_native_full_autotune, require_native_autotune_configured,
    require_native_full_autotune_configured,
};
