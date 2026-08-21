//! Bevy-owned browser adapter over verified bounded HTTP Range reads.
//!
//! The high-VRAM browser runtime eagerly verifies and retains every active Qwen, VAE, and denoiser
//! stage once. Production keeps eligible weights in direct packed Q4S form for every public
//! variant and executes measured packed kernels without requiring `shader-f16`. Explicit
//! diagnostics can still materialize dense F32 weights. Low-VRAM
//! execution streams Qwen and VAE stages and retains an inventory-qualified runtime-Q8 denoiser
//! for Edit releases. Explicit mixed-F16 Turbo retains an authenticated packed-F16 denoiser and
//! widens exactly one semantic stage to dense F32. No route falls back to CPU or manufactures
//! placeholder output.

use std::{
    collections::VecDeque,
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex, OnceLock},
};

use burn::{
    nn::RmsNorm,
    tensor::{DType, Tensor, TensorData, backend::Backend},
};
#[cfg(test)]
use burn_boogu::web_policy::{
    BrowserDenoiserExecutionKind,
    DEFAULT_QWEN_TEXT_LAYER_SUBMISSION_POLICY as BROWSER_DEFAULT_QWEN_TEXT_LAYER_SUBMISSION_POLICY,
    PACKED_F16_PROVENANCE_SUFFIX as BROWSER_PACKED_F16_PROVENANCE_SUFFIX,
    SURFACE_INFERENCE_PROVENANCE_SUFFIX as BROWSER_SURFACE_INFERENCE_PROVENANCE_SUFFIX,
};
use burn_boogu::{
    AsyncBooguDenoiserStageSource, AsyncBooguVaeStageSource, AsyncFluxVaeStageSourceAdapter,
    AsyncRetainingDenoiserSynchronizationPolicy, BooguConfig, BooguDenoiserInput,
    BooguDenoiserPrelude, BooguDenoiserTail, BooguError, BooguQuantizedLinearExecutionPolicy,
    BooguRuntimeDTypes, BooguTask, BooguVariant, DenoiserStageObserver, DmdSchedule,
    DoubleStreamBlock, PackedF16DenoiserCacheAudit, PackedF16DenoiserCacheState,
    RetainedDenoiserDTypeAudit, RetainingAsyncBooguDenoiserStageSource,
    RetainingAsyncBooguVaeStageSource, SingleStreamBlock, StreamingBooguDenoiser,
    VerifiedAsyncPackedF16DenoiserStageSource,
    artifacts::{
        BooguArtifactInventory, BooguDenoiserRuntimeQuantizationPolicy, BooguFloatLoadPolicy,
        BooguPackedQ4ResidentFootprint, BooguQuantizedLoadPolicy, BooguReleaseIdentity,
        BooguStorageProfile, TensorOwner, VerifiedAsyncBurnpackDenoiserStageSource,
        VerifiedAsyncBurnpackQwenStageSource, VerifiedAsyncBurnpackVaeStageSource,
        artifact_bundle_id_matches_selection, canonical_published_bundle,
        validate_canonical_release_artifact_digest,
    },
    boogu_model_descriptor, boogu_processor_config, decode_input_image,
    decoder_output_data_to_host, dmd_prediction, dmd_renoise, encode_reference,
    prepare_instruction, prepare_vae_reference, resolve_request, trim_instruction_features,
    web_policy::{
        BrowserExecutionPolicy as BrowserExecutionPolicies,
        PACKED_F16_DMD_VAE_HANDOFF_POLICY as BROWSER_PACKED_F16_DMD_VAE_HANDOFF_POLICY,
        PACKED_F16_NEXT_REQUEST_REHYDRATION_POLICY as BROWSER_PACKED_F16_NEXT_REQUEST_REHYDRATION_POLICY,
        PACKED_F16_QWEN_HANDOFF_POLICY as BROWSER_PACKED_F16_QWEN_HANDOFF_POLICY,
        PACKED_F16_QWEN_TEXT_LAYER_SUBMISSION_POLICY as BROWSER_PACKED_F16_QWEN_TEXT_LAYER_SUBMISSION_POLICY,
        denoiser_quantized_load_policy_name as denoiser_quantized_policy_name,
        float_load_policy_name as float_policy_name, quantized_linear_execution_policy_name,
        quantized_load_policy_name as quantized_policy_name,
    },
};
use burn_flux_vae::{
    AutoencoderKl, AutoencoderKlConfig, DiagonalGaussian, FluxVaeComponentContract,
    VerifiedAsyncBurnpackFluxVaeStageSource,
};
use burn_image::{
    ArtifactDependency, ArtifactFile, ArtifactManifest, ArtifactPath, CancellationToken,
    Dimensions, EditRequest, EncodedImage, GeneratedImage, GenerationOptions, HostImage,
    ImageEncoding, ImageOutput, ImageRequest, ImageTaskKind, InputImage, ModelProvenance,
    ProgressEvent, Prompt, RemoteBaseUrl, RunId, RuntimeError, Sha256Digest, StageTiming,
    StageTimings,
};
#[cfg(test)]
use burn_qwen3_vl::Qwen3VlEmbeddingExecutionPolicy;
use burn_qwen3_vl::{
    AsyncQwen3VlStageSource, AsyncRetainingSynchronizationPolicy, EmbeddingRowChunk,
    HostRoutedEmbedding, HostRoutedEmbeddingReport, Qwen3VlComponentContract, Qwen3VlConfig,
    Qwen3VlDecoderLayer, Qwen3VlImageProcessor, Qwen3VlImageProcessorConfig, Qwen3VlProcessor,
    Qwen3VlStage, Qwen3VlStageObserver, Qwen3VlStreamingPlan,
    Qwen3VlTextBlockLoadSynchronizationPolicy, Qwen3VlTextLayerAllocationPolicy,
    Qwen3VlTextLayerDiagnosticBoundary, Qwen3VlTokenizer, Qwen3VlVisionBlock,
    Qwen3VlVisionPatchMerger, Qwen3VlVisionPrelude, RetainingAsyncQwen3VlStageSource, RowChunkSpec,
    StreamingQwen3Vl, VerifiedAsyncBurnpackQwen3VlStageSource, tokenizer::HfTokenizer,
};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rand_distr::{Distribution, StandardNormal};
use wasm_bindgen_futures::spawn_local;

use crate::{
    BooguFactoryContext, BooguRuntime, BooguRuntimeFactory, BooguRuntimeJob, ImageJobId,
    ImageRunnerEvent, WgpuExecutionKind,
    artifact_stream::{
        ArtifactStreamConfig, BrowserArtifactControl, BrowserArtifactEvent,
        BrowserArtifactTrafficSnapshot, BrowserPersistentCachePlan, BrowserStageShardReader,
        MAX_BROWSER_MANIFEST_BYTES, artifact_progress_position, browser_cache_delete,
        browser_cache_match, browser_cache_put, fetch_browser_bounded_file,
        open_browser_artifact_cache, preflight_browser_persistent_cache, sibling_bundle_base_url,
    },
    browser_parity_fixture::{
        BrowserParityFixture, BrowserParityFixtureIdentity, BrowserParityVerificationSnapshot,
        FloatMetrics, RgbMetrics, compare_float, compare_rgb,
    },
};

// Burn 0.21 documents that the fused WGPU backend may need to be disabled on Wasm. Keep the
// Bevy/Burn bridge on `SharedWgpuBackend` for device initialization and attestation, while model
// execution uses the raw CubeCL backend against the same WGPU runtime/device registry.
type BrowserBackend = burn_wgpu::CubeBackend<burn_wgpu::WgpuRuntime, f32, i32, u32>;
type StandaloneBrowserVerifiedQwenSource =
    VerifiedAsyncBurnpackQwenStageSource<BrowserBackend, BrowserStageShardReader>;
type ComponentBrowserVerifiedQwenSource =
    VerifiedAsyncBurnpackQwen3VlStageSource<BrowserBackend, BrowserStageShardReader>;
type StandaloneBrowserVerifiedVaeSource =
    VerifiedAsyncBurnpackVaeStageSource<BrowserBackend, BrowserStageShardReader>;
type ComponentBrowserVerifiedVaeSource = AsyncFluxVaeStageSourceAdapter<
    VerifiedAsyncBurnpackFluxVaeStageSource<BrowserBackend, BrowserStageShardReader>,
>;
enum BrowserVerifiedQwenSource {
    Standalone(StandaloneBrowserVerifiedQwenSource),
    Component(ComponentBrowserVerifiedQwenSource),
}

impl AsyncQwen3VlStageSource<BrowserBackend> for BrowserVerifiedQwenSource {
    type Error = BooguError;

    async fn load_embedding_rows(
        &mut self,
        spec: &RowChunkSpec,
    ) -> Result<EmbeddingRowChunk<BrowserBackend>, Self::Error> {
        match self {
            Self::Standalone(source) => source.load_embedding_rows(spec).await,
            Self::Component(source) => source
                .load_embedding_rows(spec)
                .await
                .map_err(component_qwen_error),
        }
    }

    async fn load_host_routed_f16_embedding_f32(
        &mut self,
        input_ids: &[Vec<i64>],
        device: &burn_wgpu::WgpuDevice,
    ) -> Result<Option<HostRoutedEmbedding<BrowserBackend>>, Self::Error> {
        match self {
            // Standalone source bundles use the device-routed embedding contract.
            Self::Standalone(_) => Ok(None),
            Self::Component(source) => source
                .load_host_routed_f16_embedding_f32(input_ids, device)
                .await
                .map_err(component_qwen_error),
        }
    }

    async fn load_vision_prelude(
        &mut self,
    ) -> Result<Qwen3VlVisionPrelude<BrowserBackend>, Self::Error> {
        match self {
            Self::Standalone(source) => source.load_vision_prelude().await,
            Self::Component(source) => source
                .load_vision_prelude()
                .await
                .map_err(component_qwen_error),
        }
    }

    async fn load_vision_block(
        &mut self,
        index: usize,
    ) -> Result<Qwen3VlVisionBlock<BrowserBackend>, Self::Error> {
        match self {
            Self::Standalone(source) => source.load_vision_block(index).await,
            Self::Component(source) => source
                .load_vision_block(index)
                .await
                .map_err(component_qwen_error),
        }
    }

    async fn load_vision_deepstack_merger(
        &mut self,
        index: usize,
    ) -> Result<Qwen3VlVisionPatchMerger<BrowserBackend>, Self::Error> {
        match self {
            Self::Standalone(source) => source.load_vision_deepstack_merger(index).await,
            Self::Component(source) => source
                .load_vision_deepstack_merger(index)
                .await
                .map_err(component_qwen_error),
        }
    }

    async fn load_vision_final_merger(
        &mut self,
    ) -> Result<Qwen3VlVisionPatchMerger<BrowserBackend>, Self::Error> {
        match self {
            Self::Standalone(source) => source.load_vision_final_merger().await,
            Self::Component(source) => source
                .load_vision_final_merger()
                .await
                .map_err(component_qwen_error),
        }
    }

    async fn load_text_block(
        &mut self,
        index: usize,
    ) -> Result<Qwen3VlDecoderLayer<BrowserBackend>, Self::Error> {
        match self {
            Self::Standalone(source) => source.load_text_block(index).await,
            Self::Component(source) => source
                .load_text_block(index)
                .await
                .map_err(component_qwen_error),
        }
    }

    async fn load_text_final_norm(&mut self) -> Result<RmsNorm<BrowserBackend>, Self::Error> {
        match self {
            Self::Standalone(source) => source.load_text_final_norm().await,
            Self::Component(source) => source
                .load_text_final_norm()
                .await
                .map_err(component_qwen_error),
        }
    }

    async fn synchronize(&mut self) -> Result<(), Self::Error> {
        match self {
            Self::Standalone(source) => source.synchronize().await,
            Self::Component(source) => source.synchronize().await.map_err(component_qwen_error),
        }
    }
}

fn component_qwen_error(error: impl std::fmt::Display) -> BooguError {
    BooguError::Artifact(error.to_string())
}

enum BrowserVerifiedVaeSource {
    Standalone(StandaloneBrowserVerifiedVaeSource),
    Component(ComponentBrowserVerifiedVaeSource),
}

impl AsyncBooguVaeStageSource<BrowserBackend> for BrowserVerifiedVaeSource {
    async fn load_encoder(&mut self) -> Result<AutoencoderKl<BrowserBackend>, BooguError> {
        match self {
            Self::Standalone(source) => source.load_encoder().await,
            Self::Component(source) => source.load_encoder().await,
        }
    }

    async fn load_decoder(&mut self) -> Result<AutoencoderKl<BrowserBackend>, BooguError> {
        match self {
            Self::Standalone(source) => source.load_decoder().await,
            Self::Component(source) => source.load_decoder().await,
        }
    }

    async fn synchronize(&mut self) -> Result<(), BooguError> {
        match self {
            Self::Standalone(source) => source.synchronize().await,
            Self::Component(source) => source.synchronize().await,
        }
    }
}

type BrowserStandardVerifiedDenoiserSource =
    VerifiedAsyncBurnpackDenoiserStageSource<BrowserBackend, BrowserStageShardReader>;
type BrowserPackedVerifiedDenoiserSource =
    VerifiedAsyncPackedF16DenoiserStageSource<BrowserStageShardReader>;
enum BrowserVerifiedDenoiserSource {
    Standard(BrowserStandardVerifiedDenoiserSource),
    PackedF16(BrowserPackedVerifiedDenoiserSource),
}

impl AsyncBooguDenoiserStageSource<BrowserBackend> for BrowserVerifiedDenoiserSource {
    async fn load_prelude(&mut self) -> Result<BooguDenoiserPrelude<BrowserBackend>, BooguError> {
        match self {
            Self::Standard(source) => source.load_prelude().await,
            Self::PackedF16(source) => source.load_prelude().await,
        }
    }

    async fn load_context_refiner(
        &mut self,
        index: usize,
    ) -> Result<SingleStreamBlock<BrowserBackend>, BooguError> {
        match self {
            Self::Standard(source) => source.load_context_refiner(index).await,
            Self::PackedF16(source) => source.load_context_refiner(index).await,
        }
    }

    async fn load_noise_refiner(
        &mut self,
        index: usize,
    ) -> Result<SingleStreamBlock<BrowserBackend>, BooguError> {
        match self {
            Self::Standard(source) => source.load_noise_refiner(index).await,
            Self::PackedF16(source) => source.load_noise_refiner(index).await,
        }
    }

    async fn load_reference_refiner(
        &mut self,
        index: usize,
    ) -> Result<SingleStreamBlock<BrowserBackend>, BooguError> {
        match self {
            Self::Standard(source) => source.load_reference_refiner(index).await,
            Self::PackedF16(source) => source.load_reference_refiner(index).await,
        }
    }

    async fn load_double_stream(
        &mut self,
        index: usize,
    ) -> Result<DoubleStreamBlock<BrowserBackend>, BooguError> {
        match self {
            Self::Standard(source) => source.load_double_stream(index).await,
            Self::PackedF16(source) => source.load_double_stream(index).await,
        }
    }

    async fn load_single_stream(
        &mut self,
        index: usize,
    ) -> Result<SingleStreamBlock<BrowserBackend>, BooguError> {
        match self {
            Self::Standard(source) => source.load_single_stream(index).await,
            Self::PackedF16(source) => source.load_single_stream(index).await,
        }
    }

    async fn load_tail(&mut self) -> Result<BooguDenoiserTail<BrowserBackend>, BooguError> {
        match self {
            Self::Standard(source) => source.load_tail().await,
            Self::PackedF16(source) => source.load_tail().await,
        }
    }

    async fn synchronize(&mut self) -> Result<(), BooguError> {
        match self {
            Self::Standard(source) => source.synchronize().await,
            Self::PackedF16(source) => source.synchronize().await,
        }
    }
}

impl BrowserVerifiedDenoiserSource {
    fn packed_f16(&self) -> Option<&BrowserPackedVerifiedDenoiserSource> {
        match self {
            Self::PackedF16(source) => Some(source),
            Self::Standard(_) => None,
        }
    }

    fn packed_f16_mut(&mut self) -> Option<&mut BrowserPackedVerifiedDenoiserSource> {
        match self {
            Self::PackedF16(source) => Some(source),
            Self::Standard(_) => None,
        }
    }
}
type BrowserStreamingQwenSource = BrowserAsyncStageSource<BrowserVerifiedQwenSource>;
type BrowserQwenSource =
    RetainingAsyncQwen3VlStageSource<BrowserBackend, BrowserStreamingQwenSource>;
type BrowserStreamingVaeSource = BrowserAsyncStageSource<BrowserVerifiedVaeSource>;
type BrowserVaeSource =
    RetainingAsyncBooguVaeStageSource<BrowserBackend, BrowserStreamingVaeSource>;
type BrowserStreamingDenoiserSource = BrowserAsyncStageSource<BrowserVerifiedDenoiserSource>;
type BrowserDenoiserSource =
    RetainingAsyncBooguDenoiserStageSource<BrowserBackend, BrowserStreamingDenoiserSource>;

const MAX_RUNTIME_EVENTS: usize = 256;
const MAX_EVENTS_PER_POLL: usize = 64;
const BROWSER_PROGRESS_EVENT_NAME: &str = "burn-image-progress";
const BROWSER_RUNTIME_EVENT_NAME: &str = "burn-image-runtime";
// The portable attention implementation caps image-scale sequences to at least four partitions,
// so q1024 amortizes browser submission overhead without allowing a dense sequence-squared score
// tensor at smaller output sizes. This matches the already-qualified 1.5K browser request bound.
const BROWSER_PRODUCTION_DENOISER_QUERY_CHUNK_SIZE: usize = 1_024;
const BROWSER_1K5_QWEN_QUERY_CHUNK_SIZE: usize = 128;
const BROWSER_1K5_DENOISER_QUERY_CHUNK_SIZE: usize = 1_024;
const BROWSER_1K5_VAE_QUERY_CHUNK_SIZE: usize = 4_096;
const BROWSER_1K5_QWEN_ALIGNED_STAGE_COUNT: usize = 70;
const BROWSER_1K5_DENOISER_BOUNDARY_COUNT: usize = 236;
const BROWSER_1K5_DENOISER_RETAINED_STAGE_COUNT: usize = 48;
const BROWSER_1K5_AUTHENTICATED_ONLY_TENSOR_COUNT: usize = 17;
const BROWSER_1K5_NUMERICAL_TENSOR_COUNT: usize = 355;
const BROWSER_1K5_VAE_REFERENCE_REPEAT_COUNT: usize = 3;
const BROWSER_TURBO_ACTIVE_LOGICAL_OBJECTS: u32 = 186;
const BROWSER_TURBO_ACTIVE_UNIQUE_TRANSPORT_PARTS: u32 = 1_751;
const BROWSER_TURBO_ACTIVE_TRANSPORT_BYTES: u64 = 35_106_151_424;
// Browser parity is deliberately observer-heavy. Keep each asynchronous WebGPU map bounded so a
// single large semantic activation cannot require a 100+ MiB staging map. This does not alter the
// tensor values or production inference path; the complete F32 vector is assembled on the host for
// the authenticated comparison exactly as before.
const BROWSER_PARITY_MAX_READBACK_CHUNK_BYTES: usize = 16 * 1024 * 1024;
const BROWSER_PARITY_F32_ELEMENT_BYTES: usize = std::mem::size_of::<f32>();
// This identifies the schema-v1 flat closure used to calibrate the component envelope. It is
// provenance only, not a runtime-accepted canonical artifact identity. The schema-v2 modular
// composition must be requalified before its full-chain result is described as current evidence.
const BROWSER_WEBGPU_VAE_F32_ORACLE_SOURCE_CONTENT_DIGEST: &str =
    "5d7e25b1d9be1fdf4a6372bfb9db28cf62ef90253082cef22af09653047e3a7b";
const BROWSER_WEBGPU_VAE_F32_ORACLE_ENVELOPE: BrowserWebGpuVaeF32OracleEnvelope =
    BrowserWebGpuVaeF32OracleEnvelope {
        backend: "BrowserWebGpu/raw-cubecl-no-fusion",
        artifact_content_digest: BROWSER_WEBGPU_VAE_F32_ORACLE_SOURCE_CONTENT_DIGEST,
        artifact_profile: "f16-qwen-vision-f32",
        weight_storage_dtype: "f16",
        weight_load_policy: "adapt-to-f32",
        execution_dtype: "f32",
        calibrated_adapter: "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        calibrated_device: "0x2bb1",
        calibrated_driver: "610.43.02",
        portability: "no-cross-adapter-portability-claim",
        moments: BrowserParityReferenceGate {
            maximum_abs: 0.016,
            maximum_rmse: 0.000_75,
            minimum_cosine_similarity: 0.999_999,
        },
        mean: BrowserParityMaximumAbsGate { maximum_abs: 0.013 },
        logvar: BrowserParityMaximumAbsGate { maximum_abs: 0.016 },
        std: BrowserParityMaximumAbsGate {
            maximum_abs: 0.000_1,
        },
        raw_latent: BrowserParityMaximumAbsGate { maximum_abs: 0.013 },
        scaled_latent: BrowserParityReferenceGate {
            maximum_abs: 0.005,
            maximum_rmse: 0.000_2,
            minimum_cosine_similarity: 0.999_999,
        },
    };
type BrowserBuildSlot = Arc<Mutex<Option<Result<BrowserBooguEngine, RuntimeError>>>>;

pub use burn_boogu::deployment::BrowserBooguResidencyPolicy;
#[cfg(test)]
use burn_boogu::deployment::default_browser_low_vram_residency;
use burn_boogu::deployment::default_browser_residency;

#[derive(serde::Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum BrowserRuntimeEvent {
    Preparing {
        message: String,
    },
    VramPreflight {
        status: &'static str,
        model: String,
        policy: &'static str,
        required_device_bytes: u64,
        allocation_count: usize,
        largest_allocation_bytes: u64,
        allocations_committed: bool,
        shared_device_and_queue: bool,
    },
    ManifestVerified {
        bundle: String,
        weight_objects: u32,
        weight_bytes: u64,
    },
    ResidentResourcePlan {
        weight_storage_policy: &'static str,
        stored_weight_bytes: u64,
        packed_q4_value_bytes: u64,
        q4_scale_bytes: u64,
        packed_f16_weight_bytes: u64,
        f32_auxiliary_weight_bytes: u64,
        resident_weight_bytes: u64,
        activation_reserve_bytes: u64,
        conservative_planned_device_bytes: u64,
        strict_device_cap_bytes: Option<u64>,
    },
    LowVramResourcePlan {
        denoiser_quantized_load_policy: &'static str,
        denoiser_quantized_linear_execution_policy: &'static str,
        audited_retained_q8_denoiser_bytes: u64,
        expected_q8s_block32_f32_tensor_count: usize,
        expected_f32_tensor_count: usize,
        expected_q8s_block32_f32_elements: u64,
        expected_f32_elements: u64,
        expected_q8s_block32_f32_payload_bytes: u64,
        expected_f32_payload_bytes: u64,
        audited_max_streamed_qwen_stage_f32_bytes: u64,
        audited_loaded_vae_module_f32_bytes: u64,
        audited_max_dense_denoiser_stage_f32_bytes: u64,
        audited_max_phase_local_f32_stage_bytes: u64,
        runtime_quantization_workspace_bytes: u64,
        activation_reserve_bytes: u64,
        conservative_planned_device_bytes: u64,
        strict_device_cap_bytes: u64,
    },
    PackedF16ResourcePlan {
        qwen_text_layer_allocation_policy: &'static str,
        qwen_text_block_load_synchronization_policy: &'static str,
        qwen_text_layer_submission_policy: &'static str,
        qwen_text_layer_persistent_pool_requires_measured_gpu_gate: bool,
        authenticated_artifact_bytes: u64,
        canonical_compact_f16_payload_bytes: u64,
        retained_packed_f16_denoiser_bytes: u64,
        inserted_padding_elements: u64,
        padded_f16_elements: u64,
        expected_stage_count: usize,
        expected_object_count: usize,
        expected_tensor_count: usize,
        max_packed_stage_bytes: u64,
        max_materialized_stage_f32_bytes: u64,
        max_packed_object_bytes: u64,
        max_materialized_object_f32_bytes: u64,
        materialized_f32_bytes_per_dmd_step: u64,
        preload_workspace_bytes: u64,
        preload_peak_bytes: u64,
        activation_reserve_bytes: u64,
        conservative_planned_device_bytes: u64,
        strict_device_cap_bytes: u64,
        expected_stage_materializations_per_request: u64,
        expected_object_unpacks_per_request: u64,
        expected_packed_read_bytes_per_request: u64,
        expected_f32_write_bytes_per_request: u64,
        on_device_quantized_execution_claimed: bool,
    },
    PackedF16DenoiserPreload {
        traffic: BrowserArtifactTrafficReport,
        cached_stages: usize,
        cached_objects: usize,
        cached_tensors: usize,
        cached_bytes: u64,
        previous_preload_attempt_count: u64,
        preload_attempt_count: u64,
        request_scoped_rehydration: bool,
        rehydration_policy: &'static str,
    },
    PackedF16DenoiserLifecycle {
        lifecycle: BrowserPackedF16DenoiserLifecycleReport,
    },
    PackedF16DmdVaeHandoff {
        run_id: RunId,
        report: Box<BrowserPackedF16DmdVaeHandoffReport>,
    },
    PackedF16QwenHostEmbedding {
        run_id: RunId,
        report: Box<HostRoutedEmbeddingReport>,
    },
    PackedF16QwenBlock0ExecutionDiagnostics {
        run_id: RunId,
        diagnostics: Box<BrowserPackedF16QwenBlock0ExecutionDiagnostics>,
    },
    PackedF16QwenBlock0PostSyncDiagnostic {
        run_id: RunId,
        diagnostic: Box<BrowserPackedF16QwenBlock0PostSyncDiagnostic>,
    },
    PackedF16QwenPreHandoffDiagnostics {
        run_id: RunId,
        diagnostics: Box<BrowserPackedF16QwenPreHandoffDiagnostics>,
    },
    PackedF16QwenPostHandoffDiagnostics {
        run_id: RunId,
        diagnostics: Box<BrowserPackedF16QwenPostHandoffDiagnostics>,
    },
    PackedF16PreDmdInputDiagnostics {
        run_id: RunId,
        diagnostics: Box<BrowserPackedF16PreDmdInputDiagnostics>,
    },
    ArtifactTraffic {
        traffic: BrowserArtifactTrafficReport,
    },
    ResidentCacheAudit {
        run_id: RunId,
        boundary: &'static str,
        qwen_cached_stages: usize,
        qwen_expected_stages: usize,
        vae_cached_stages: usize,
        vae_expected_stages: usize,
        denoiser_cached_stages: usize,
        denoiser_expected_stages: usize,
        qwen_synchronization_pending: bool,
        denoiser_synchronization_pending: bool,
        resident_weights_preserved: bool,
    },
    Ready {
        model: String,
        request_enabled: bool,
        selected_model_cache_complete: bool,
        selected_model_device_resident: bool,
        transfer: Option<burn_image::ArtifactTransferProgress>,
        block0_execution_mode: &'static str,
        qwen_text_layer_allocation_policy: &'static str,
        qwen_text_block_load_synchronization_policy: &'static str,
        qwen_text_layer_submission_policy: &'static str,
    },
    Failed {
        message: String,
    },
    SurfaceInferenceSuspended {
        run_id: RunId,
        policy: &'static str,
        primary_window_camera_count: usize,
        saved_camera_state_count: usize,
        previously_active_camera_count: usize,
        inactive_camera_count: usize,
        active_job_count: usize,
        suspended_before_runtime_submit: bool,
        all_primary_window_cameras_inactive: bool,
    },
    SurfaceInferenceResumed {
        run_id: RunId,
        policy: &'static str,
        terminal: &'static str,
        primary_window_camera_count: usize,
        saved_camera_state_count: usize,
        restored_camera_state_count: usize,
        restored_active_camera_count: usize,
        active_job_count: usize,
        resumed_after_runtime_terminal: bool,
        resumed_before_output_ready: bool,
        exact_saved_states_restored: bool,
        all_primary_window_cameras_restored: bool,
    },
    SurfaceInferenceGateFailed {
        run_id: RunId,
        policy: &'static str,
        phase: &'static str,
        message: String,
        exact_saved_states_restored: bool,
    },
}

/// Exact per-request logical, Cache Storage, and network artifact traffic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct BrowserArtifactTrafficReport {
    pub object_reads: u64,
    pub object_read_bytes: u64,
    pub range_reads: u64,
    pub range_read_bytes: u64,
    pub verified_objects: u64,
    pub cache_lookups: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_read_bytes: u64,
    pub network_requests: u64,
    pub network_response_bytes: u64,
    pub cache_writes: u64,
    pub cache_write_bytes: u64,
    pub cache_evictions: u64,
    pub cache_evicted_entries: u64,
    pub cache_invalid_entries: u64,
    pub integrity_refetches: u64,
}

impl From<BrowserArtifactTrafficSnapshot> for BrowserArtifactTrafficReport {
    fn from(snapshot: BrowserArtifactTrafficSnapshot) -> Self {
        Self {
            object_reads: snapshot.object_reads,
            object_read_bytes: snapshot.object_read_bytes,
            range_reads: snapshot.range_fetch_requests,
            range_read_bytes: snapshot.range_response_bytes,
            verified_objects: snapshot.verified_objects,
            cache_lookups: snapshot.cache_lookup_requests,
            cache_hits: snapshot.cache_hits,
            cache_misses: snapshot.cache_misses,
            cache_read_bytes: snapshot.cache_read_bytes,
            network_requests: snapshot.network_fetch_requests,
            network_response_bytes: snapshot.network_response_bytes,
            cache_writes: snapshot.cache_write_requests,
            cache_write_bytes: snapshot.cache_write_bytes,
            cache_evictions: snapshot.cache_eviction_requests,
            cache_evicted_entries: snapshot.cache_evicted_entries,
            cache_invalid_entries: snapshot.cache_invalid_entries,
            integrity_refetches: snapshot.integrity_refetches,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct BrowserResidentResourcePlan {
    /// Exact storage policy admitted by the browser runtime.
    pub weight_storage_policy: &'static str,
    /// Exact active logical artifact weight bytes before dtype adaptation.
    pub stored_weight_bytes: u64,
    /// Exact signed four-bit value payload retained on the device.
    pub packed_q4_value_bytes: u64,
    /// Exact F32 block-scale payload paired with the Q4 values.
    pub q4_scale_bytes: u64,
    /// Exact active parameter bytes retained in packed F16 buffers.
    pub packed_f16_weight_bytes: u64,
    /// Exact active parameter bytes retained as F32 auxiliaries.
    pub f32_auxiliary_weight_bytes: u64,
    /// Sum of packed F16 weights and F32 auxiliaries resident on the device.
    pub resident_weight_bytes: u64,
    /// Conservative shape-aware activation/workspace reserve.
    pub activation_reserve_bytes: u64,
    /// Resident weights plus the conservative activation reserve.
    pub conservative_planned_device_bytes: u64,
    /// Strict external GPU-process cap for a measured qualification, when applicable.
    pub strict_device_cap_bytes: Option<u64>,
}

fn browser_low_vram_resource_plan_event(
    plan: BrowserLowVramResourcePlan,
    quantized_linear_execution_policy: BooguQuantizedLinearExecutionPolicy,
) -> BrowserRuntimeEvent {
    BrowserRuntimeEvent::LowVramResourcePlan {
        denoiser_quantized_load_policy: denoiser_quantized_policy_name(
            BooguQuantizedLoadPolicy::Preserve,
            BooguDenoiserRuntimeQuantizationPolicy::Q8sBlock32F32,
        ),
        denoiser_quantized_linear_execution_policy: quantized_linear_execution_policy_name(
            quantized_linear_execution_policy,
        ),
        audited_retained_q8_denoiser_bytes: plan.audited_retained_q8_denoiser_bytes,
        expected_q8s_block32_f32_tensor_count: plan.expected_q8s_block32_f32_tensor_count,
        expected_f32_tensor_count: plan.expected_f32_tensor_count,
        expected_q8s_block32_f32_elements: plan.expected_q8s_block32_f32_elements,
        expected_f32_elements: plan.expected_f32_elements,
        expected_q8s_block32_f32_payload_bytes: plan.expected_q8s_block32_f32_payload_bytes,
        expected_f32_payload_bytes: plan.expected_f32_payload_bytes,
        audited_max_streamed_qwen_stage_f32_bytes: plan.audited_max_streamed_qwen_stage_f32_bytes,
        audited_loaded_vae_module_f32_bytes: plan.audited_loaded_vae_module_f32_bytes,
        audited_max_dense_denoiser_stage_f32_bytes: plan.audited_max_dense_denoiser_stage_f32_bytes,
        audited_max_phase_local_f32_stage_bytes: plan.audited_max_phase_local_f32_stage_bytes,
        runtime_quantization_workspace_bytes: plan.runtime_quantization_workspace_bytes,
        activation_reserve_bytes: plan.activation_reserve_bytes,
        conservative_planned_device_bytes: plan.conservative_planned_device_bytes,
        strict_device_cap_bytes: plan.strict_device_cap_bytes,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct BrowserLowVramResourcePlan {
    /// Inventory-derived packed-Q8 plus F32 denoiser parameter bytes.
    pub audited_retained_q8_denoiser_bytes: u64,
    /// Expected inventory-qualified Q8S parameter tensor count.
    pub expected_q8s_block32_f32_tensor_count: usize,
    /// Expected non-quantized F32 parameter tensor count.
    pub expected_f32_tensor_count: usize,
    /// Expected values carried by Q8S parameter tensors.
    pub expected_q8s_block32_f32_elements: u64,
    /// Expected values carried by F32 parameter tensors.
    pub expected_f32_elements: u64,
    /// Expected packed Q8S values and block-scale bytes.
    pub expected_q8s_block32_f32_payload_bytes: u64,
    /// Expected ordinary F32 parameter bytes.
    pub expected_f32_payload_bytes: u64,
    /// Largest audited one-stage F32 Qwen payload used by the streaming policy.
    pub audited_max_streamed_qwen_stage_f32_bytes: u64,
    /// Full initialized F32 VAE module returned by the current verified source.
    ///
    /// The source applies only the selected encoder or decoder half, but currently
    /// initializes the complete autoencoder before that selection.
    pub audited_loaded_vae_module_f32_bytes: u64,
    /// Largest canonical F32 denoiser source stage present during runtime quantization/preload.
    pub audited_max_dense_denoiser_stage_f32_bytes: u64,
    /// Maximum F32 stage residency across the mutually exclusive Qwen, DMD, and VAE phases.
    pub audited_max_phase_local_f32_stage_bytes: u64,
    /// Conservative source/destination/transient quantizer and materialization workspace reserve.
    pub runtime_quantization_workspace_bytes: u64,
    /// Conservative released-shape activation and kernel-workspace reserve.
    pub activation_reserve_bytes: u64,
    /// Conservative sum used for fail-closed admission; not a measured peak.
    pub conservative_planned_device_bytes: u64,
    /// Exclusive public device-memory ceiling.
    pub strict_device_cap_bytes: u64,
}

/// Exact admission envelope for Turbo's retained packed-F16 production execution policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct BrowserPackedF16ResourcePlan {
    /// Streamed Qwen activation allocator used while the packed denoiser cache is resident.
    pub qwen_text_layer_allocation_policy: &'static str,
    /// Explicit queue barrier after each text-block load and before its forward submission.
    pub qwen_text_block_load_synchronization_policy: &'static str,
    /// Combined upload/forward submission contract with bounded task batches and scoped submits.
    pub qwen_text_layer_submission_policy: &'static str,
    /// Persistent-pool residency is not covered by a derived byte bound in this static plan.
    /// Rendered qualification must therefore enforce the aggregate measured-GPU-memory gate.
    pub qwen_text_layer_persistent_pool_requires_measured_gpu_gate: bool,
    /// Exact 106 authenticated Burnpack object bytes, including framing.
    pub authenticated_artifact_bytes: u64,
    /// Canonical compact F16 tensor payload transferred and authenticated from the artifact.
    pub canonical_compact_f16_payload_bytes: u64,
    /// Deterministically padded F16 payload retained as packed U32 WebGPU buffers.
    pub retained_packed_f16_denoiser_bytes: u64,
    /// Logical zero elements inserted so every F32 tensor view starts on 256-byte alignment.
    pub inserted_padding_elements: u64,
    /// Canonical plus padding elements retained by the packed object arenas.
    pub padded_f16_elements: u64,
    /// Semantic stages authenticated and retained before inference requests begin.
    pub expected_stage_count: usize,
    /// Immutable content-addressed objects authenticated and retained before every request.
    pub expected_object_count: usize,
    /// Canonical tensors covered by the retained objects.
    pub expected_tensor_count: usize,
    /// Largest packed-F16 semantic stage.
    pub max_packed_stage_bytes: u64,
    /// Largest one-stage dense-F32 materialization.
    pub max_materialized_stage_f32_bytes: u64,
    /// Largest one-object packed arena.
    pub max_packed_object_bytes: u64,
    /// Largest one-object F32 arena.
    pub max_materialized_object_f32_bytes: u64,
    /// Aggregate padded F32 bytes written while materializing all stages once.
    pub materialized_f32_bytes_per_dmd_step: u64,
    /// Conservative two-buffer upload workspace used only during preload.
    pub preload_workspace_bytes: u64,
    /// Peak retained packed payload plus preload workspace.
    pub preload_peak_bytes: u64,
    /// Four maximum-size activation/kernel buffers reserved during inference.
    pub activation_reserve_bytes: u64,
    /// Retained packed payload plus maximum F32 stage and activation reserve.
    pub conservative_planned_device_bytes: u64,
    /// Exclusive public device-memory ceiling.
    pub strict_device_cap_bytes: u64,
    /// Four DMD steps times all 46 semantic stages.
    pub expected_stage_materializations_per_request: u64,
    /// Four DMD steps times all 106 retained objects.
    pub expected_object_unpacks_per_request: u64,
    /// Packed-F16 bytes read by widening kernels across four DMD steps.
    pub expected_packed_read_bytes_per_request: u64,
    /// Dense-F32 bytes written by widening kernels across four DMD steps.
    pub expected_f32_write_bytes_per_request: u64,
    /// Always false: this is storage compression followed by exact dense-F32 execution.
    pub on_device_quantized_execution_claimed: bool,
}

/// Exact manifest-derived contract for Turbo's host-routed Qwen embedding objects.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct BrowserPackedF16QwenEmbeddingPlan {
    pub expected_chunk_count: usize,
    pub expected_object_count: usize,
    pub authenticated_object_bytes: u64,
    pub authenticated_f16_payload_bytes: u64,
}

/// DMD-scoped proof that Turbo had the complete raw cache through four bounded predictions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct BrowserPackedF16DenoiserLifecycleReport {
    pub cache_state: &'static str,
    pub cache_ready: bool,
    pub cached_stages: usize,
    pub cached_objects: usize,
    pub cached_tensors: usize,
    pub cached_bytes: u64,
    /// Cumulative authenticated artifact bytes across all preloads on this engine.
    pub authenticated_artifact_bytes: u64,
    /// Cumulative packed upload bytes across all preloads on this engine.
    pub packed_upload_bytes: u64,
    pub stage_materializations: u64,
    pub object_unpacks: u64,
    pub packed_read_bytes: u64,
    pub f32_write_bytes: u64,
    pub preload_attempt_count: u64,
    pub failure_count: u64,
    pub dmd_artifact_traffic: BrowserArtifactTrafficReport,
    pub synchronization_pending: bool,
    pub matches_plan: bool,
}

/// Exact host-readback statistics for one rendered-smoke DMD input tensor.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct BrowserPackedF16TensorInputDiagnostic {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub element_count: usize,
    pub finite_element_count: usize,
    pub all_finite: bool,
    pub max_abs: Option<f64>,
    pub mean: Option<f64>,
    pub rms: Option<f64>,
    pub sha256: Sha256Digest,
}

const BROWSER_QWEN_BLOCK0_SERIALIZED_DIAGNOSTIC_MODE: &str = "serialized-diagnostic";
const BROWSER_QWEN_BLOCK0_ORDINARY_MODE: &str = "ordinary";
const BROWSER_QWEN_BLOCK0_EXECUTION_MODE_QUERY: &str = "qwen-block0-execution-mode";
const BROWSER_PACKED_F16_QWEN_PRE_HANDOFF_SCOPE: &str =
    "rendered-model-smoke/ordinary-turbo-packed-f16/qwen-pre-handoff-readback";
const BROWSER_PACKED_F16_QWEN_BLOCK0_EXECUTION_SCOPE: &str =
    "rendered-model-smoke/ordinary-turbo-packed-f16/qwen-block-00-serialized-operation-readbacks";
const BROWSER_PACKED_F16_QWEN_BLOCK0_BOUNDARIES: [Qwen3VlTextLayerDiagnosticBoundary; 9] = [
    Qwen3VlTextLayerDiagnosticBoundary::LayerInput,
    Qwen3VlTextLayerDiagnosticBoundary::InputLayerNormGamma,
    Qwen3VlTextLayerDiagnosticBoundary::IdentityAddCanary,
    Qwen3VlTextLayerDiagnosticBoundary::InputNorm,
    Qwen3VlTextLayerDiagnosticBoundary::AttentionOutput,
    Qwen3VlTextLayerDiagnosticBoundary::FirstResidual,
    Qwen3VlTextLayerDiagnosticBoundary::PostAttentionNorm,
    Qwen3VlTextLayerDiagnosticBoundary::MlpOutput,
    Qwen3VlTextLayerDiagnosticBoundary::FinalResidualOutput,
];
const BROWSER_PACKED_F16_QWEN_BLOCK0_POST_SYNC_SCOPE: &str =
    "rendered-model-smoke/ordinary-turbo-packed-f16/qwen-block-00-immediate-post-sync-readback";
const BROWSER_PACKED_F16_QWEN_POST_HANDOFF_SCOPE: &str =
    "rendered-model-smoke/ordinary-turbo-packed-f16/qwen-post-handoff-readback";

/// Rendered-smoke-only localization of every text-path Qwen activation before the bounded host
/// handoff. Holding these small conditioning activations is diagnostic-only and never occurs in
/// ordinary inference.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct BrowserPackedF16QwenPreHandoffDiagnostics {
    pub scope: String,
    pub effective_instruction_length: usize,
    pub expected_stage_output_count: usize,
    pub stage_outputs: Vec<BrowserPackedF16TensorInputDiagnostic>,
    pub stage_names_exact: bool,
    pub qwen_last_hidden_state_before_trim: BrowserPackedF16TensorInputDiagnostic,
    pub instruction_after_trim_cast_before_handoff: BrowserPackedF16TensorInputDiagnostic,
    pub all_tensors_finite: bool,
    pub no_tensor_all_zero: bool,
    pub first_non_finite_tensor: Option<String>,
    pub first_all_zero_tensor: Option<String>,
    pub final_norm_matches_returned_output: bool,
    pub block_00_immediate_post_sync: BrowserPackedF16QwenBlock0PostSyncDiagnostic,
    pub block_00_immediate_matches_delayed_capture: bool,
}

/// Immediate readback after block 0's real per-stage WebGPU barrier.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct BrowserPackedF16QwenBlock0PostSyncDiagnostic {
    pub scope: String,
    pub block0_execution_mode: String,
    pub text_layer_allocation_policy: String,
    pub text_block_load_synchronization_policy: String,
    pub qwen_text_layer_submission_policy: String,
    pub tensor: BrowserPackedF16TensorInputDiagnostic,
    pub all_finite: bool,
    pub not_all_zero: bool,
}

/// One F32 parameter or activation consumed before the next block-0 operation is submitted.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct BrowserPackedF16QwenBlock0BoundaryDiagnostic {
    pub sequence_index: usize,
    pub boundary: String,
    pub tensor_kind: String,
    pub tensor: BrowserPackedF16TensorInputDiagnostic,
    pub all_finite: bool,
    pub not_all_zero: bool,
}

/// Cumulative rendered-smoke block-0 localization report. A failing run dispatches its captured
/// prefix before returning the layer error; a healthy run dispatches the complete sequence.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct BrowserPackedF16QwenBlock0ExecutionDiagnostics {
    pub scope: String,
    pub block0_execution_mode: String,
    pub text_layer_allocation_policy: String,
    pub text_block_load_synchronization_policy: String,
    pub qwen_text_layer_submission_policy: String,
    pub expected_boundary_count: usize,
    pub captured_boundary_count: usize,
    pub boundaries: Vec<BrowserPackedF16QwenBlock0BoundaryDiagnostic>,
    pub boundary_names_exact: bool,
    pub all_captured_tensors_finite: bool,
    pub no_captured_tensor_all_zero: bool,
    pub identity_add_canary_matches_input: Option<bool>,
    pub complete: bool,
    pub first_failure_boundary: Option<String>,
    pub failure_reason: Option<String>,
}

/// Rendered-smoke proof that the exact host handoff survived allocator cleanup and F32 reupload.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct BrowserPackedF16QwenPostHandoffDiagnostics {
    pub scope: String,
    pub handoff: BrowserPackedF16QwenInstructionHandoffReport,
    pub instruction_after_handoff: BrowserPackedF16TensorInputDiagnostic,
}

/// Per-request production provenance for the bounded exact-F32 Qwen-to-DMD handoff.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct BrowserPackedF16QwenInstructionHandoffReport {
    pub policy: String,
    pub qwen_release_unused_memory_after_stage: bool,
    pub qwen_text_layer_allocation_policy: String,
    pub qwen_text_block_load_synchronization_policy: String,
    pub qwen_text_layer_submission_policy: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub element_count: usize,
    pub payload_bytes: u64,
    pub device_to_host_readback_bytes: u64,
    pub host_to_device_upload_bytes: u64,
    pub total_transfer_bytes: u64,
    pub before_sha256: Sha256Digest,
    pub after_sha256: Sha256Digest,
    pub all_finite: bool,
    pub not_all_zero: bool,
    pub digest_matches: bool,
    pub cleanup_completed: bool,
    pub packed_cache: BrowserPackedF16CacheEvidence,
}

/// Per-request proof that the final exact-F32 DMD latent crossed a host boundary while every
/// packed denoiser allocation was evicted before VAE decode.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct BrowserPackedF16DmdVaeHandoffReport {
    pub policy: String,
    pub next_request_rehydration_policy: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub element_count: usize,
    pub payload_bytes: u64,
    pub device_to_host_readback_bytes: u64,
    pub host_to_device_upload_bytes: u64,
    pub total_transfer_bytes: u64,
    pub before_sha256: Sha256Digest,
    pub after_sha256: Sha256Digest,
    pub all_finite: bool,
    pub not_all_zero: bool,
    pub digest_matches: bool,
    pub wrapper_cached_stages_before_clear: usize,
    pub wrapper_cached_stages_after_clear: usize,
    pub synchronization_pending_before_cleanup: bool,
    pub synchronization_pending_after_cleanup: bool,
    pub rope_cache_cleared: bool,
    pub cleanup_completed: bool,
    pub packed_cache_before_cleanup: BrowserPackedF16CacheEvidence,
    pub packed_cache_after_cleanup: BrowserPackedF16CacheEvidence,
    pub preload_attempt_count: u64,
    pub expected_next_request_preload_attempt_count: u64,
}

struct BrowserPackedF16QwenPreHandoffContext {
    effective_instruction_length: usize,
    expected_stage_output_count: usize,
    stage_outputs: Vec<BrowserPackedF16TensorInputDiagnostic>,
    qwen_last_hidden_state_before_trim: BrowserPackedF16TensorInputDiagnostic,
    block_00_immediate_post_sync: BrowserPackedF16QwenBlock0PostSyncDiagnostic,
}

/// Allocator and retained-cache provenance immediately after the Qwen-to-DMD phase boundary.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct BrowserPackedF16CacheEvidence {
    pub state: String,
    pub cache_ready: bool,
    pub cached_stages: usize,
    pub cached_objects: usize,
    pub cached_tensors: usize,
    pub cached_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct BrowserPackedF16PreDmdPolicyEvidence {
    pub qwen_release_unused_memory_after_stage: bool,
    pub qwen_text_block_load_synchronization_policy: String,
    pub qwen_text_layer_submission_policy: String,
    pub packed_qwen_instruction_handoff_policy: String,
    pub cleanup_completed: bool,
    pub post_cleanup_packed_cache: BrowserPackedF16CacheEvidence,
}

/// Diagnostic-only proof of the exact inputs presented to the first ordinary Turbo DMD step.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct BrowserPackedF16PreDmdInputDiagnostics {
    pub scope: String,
    pub policy: BrowserPackedF16PreDmdPolicyEvidence,
    pub dmd_steps: usize,
    pub instruction: BrowserPackedF16TensorInputDiagnostic,
    pub initial_latent: BrowserPackedF16TensorInputDiagnostic,
    pub renoise: Vec<BrowserPackedF16TensorInputDiagnostic>,
    pub first_timestep: BrowserPackedF16TensorInputDiagnostic,
    pub all_inputs_finite: bool,
}

const fn packed_f16_cache_state_label(state: PackedF16DenoiserCacheState) -> &'static str {
    match state {
        PackedF16DenoiserCacheState::Empty => "empty",
        PackedF16DenoiserCacheState::Preloading => "preloading",
        PackedF16DenoiserCacheState::Ready => "ready",
        PackedF16DenoiserCacheState::Failed => "failed",
    }
}

fn packed_f16_cache_evidence(audit: PackedF16DenoiserCacheAudit) -> BrowserPackedF16CacheEvidence {
    BrowserPackedF16CacheEvidence {
        state: packed_f16_cache_state_label(audit.state).into(),
        cache_ready: audit.packed_cache_ready,
        cached_stages: audit.cached_stage_count,
        cached_objects: audit.cached_object_count,
        cached_tensors: audit.cached_tensor_count,
        cached_bytes: audit.retained_packed_bytes,
    }
}

fn packed_f16_lifecycle_report(
    variant: BooguVariant,
    before_dmd: PackedF16DenoiserCacheAudit,
    after_dmd: PackedF16DenoiserCacheAudit,
    dmd_artifact_traffic: BrowserArtifactTrafficReport,
    synchronization_pending: bool,
) -> Result<BrowserPackedF16DenoiserLifecycleReport, RuntimeError> {
    let delta = |name: &str, after: u64, before: u64| {
        after.checked_sub(before).ok_or_else(|| {
            execution_error(
                variant,
                format!("packed-F16 {name} counter moved backwards"),
            )
        })
    };
    Ok(BrowserPackedF16DenoiserLifecycleReport {
        cache_state: packed_f16_cache_state_label(after_dmd.state),
        cache_ready: after_dmd.packed_cache_ready,
        cached_stages: after_dmd.cached_stage_count,
        cached_objects: after_dmd.cached_object_count,
        cached_tensors: after_dmd.cached_tensor_count,
        cached_bytes: after_dmd.retained_packed_bytes,
        authenticated_artifact_bytes: after_dmd.packed_read_bytes,
        packed_upload_bytes: after_dmd.packed_upload_bytes,
        stage_materializations: delta(
            "stage materialization",
            after_dmd.materialized_stage_count,
            before_dmd.materialized_stage_count,
        )?,
        object_unpacks: delta(
            "object unpack",
            after_dmd.object_unpack_count,
            before_dmd.object_unpack_count,
        )?,
        packed_read_bytes: delta(
            "materialization packed-read byte",
            after_dmd.materialization_packed_read_bytes,
            before_dmd.materialization_packed_read_bytes,
        )?,
        f32_write_bytes: delta(
            "F32 write byte",
            after_dmd.f32_write_bytes,
            before_dmd.f32_write_bytes,
        )?,
        preload_attempt_count: after_dmd.preload_attempt_count,
        failure_count: after_dmd.failure_count,
        dmd_artifact_traffic,
        synchronization_pending,
        matches_plan: false,
    })
}

fn validate_packed_f16_denoiser_lifecycle(
    variant: BooguVariant,
    plan: BrowserPackedF16ResourcePlan,
    completed_dmd_steps: u64,
    mut observed: BrowserPackedF16DenoiserLifecycleReport,
) -> Result<BrowserPackedF16DenoiserLifecycleReport, RuntimeError> {
    let expected_stage_materializations = (plan.expected_stage_count as u64)
        .checked_mul(completed_dmd_steps)
        .ok_or_else(|| execution_error(variant, "packed-F16 stage count overflowed"))?;
    let expected_object_unpacks = (plan.expected_object_count as u64)
        .checked_mul(completed_dmd_steps)
        .ok_or_else(|| execution_error(variant, "packed-F16 object count overflowed"))?;
    let expected_packed_read_bytes = plan
        .retained_packed_f16_denoiser_bytes
        .checked_mul(completed_dmd_steps)
        .ok_or_else(|| execution_error(variant, "packed-F16 read-byte count overflowed"))?;
    let expected_f32_write_bytes = plan
        .materialized_f32_bytes_per_dmd_step
        .checked_mul(completed_dmd_steps)
        .ok_or_else(|| execution_error(variant, "packed-F16 write-byte count overflowed"))?;
    let expected_cumulative_artifact_bytes = plan
        .authenticated_artifact_bytes
        .checked_mul(observed.preload_attempt_count)
        .ok_or_else(|| execution_error(variant, "packed-F16 artifact-byte count overflowed"))?;
    let expected_cumulative_upload_bytes = plan
        .retained_packed_f16_denoiser_bytes
        .checked_mul(observed.preload_attempt_count)
        .ok_or_else(|| execution_error(variant, "packed-F16 upload-byte count overflowed"))?;
    observed.matches_plan = observed.cache_ready
        && observed.cache_state == "ready"
        && observed.cached_stages == plan.expected_stage_count
        && observed.cached_objects == plan.expected_object_count
        && observed.cached_tensors == plan.expected_tensor_count
        && observed.cached_bytes == plan.retained_packed_f16_denoiser_bytes
        && observed.preload_attempt_count > 0
        && observed.authenticated_artifact_bytes == expected_cumulative_artifact_bytes
        && observed.packed_upload_bytes == expected_cumulative_upload_bytes
        && observed.stage_materializations == expected_stage_materializations
        && observed.object_unpacks == expected_object_unpacks
        && observed.packed_read_bytes == expected_packed_read_bytes
        && observed.f32_write_bytes == expected_f32_write_bytes
        && observed.dmd_artifact_traffic == BrowserArtifactTrafficReport::default()
        && !observed.synchronization_pending;
    if !observed.matches_plan {
        return Err(execution_error(
            variant,
            format!(
                "browser packed-F16 denoiser lifecycle differs from its admitted plan: {observed:?}"
            ),
        ));
    }
    Ok(observed)
}

fn validate_packed_f16_dmd_vae_handoff_report(
    variant: BooguVariant,
    plan: BrowserPackedF16ResourcePlan,
    expected_shape: [usize; 4],
    report: &BrowserPackedF16DmdVaeHandoffReport,
) -> Result<(), RuntimeError> {
    let expected_elements = expected_shape
        .into_iter()
        .try_fold(1_usize, |total, dimension| total.checked_mul(dimension))
        .ok_or_else(|| execution_error(variant, "DMD-to-VAE latent element count overflowed"))?;
    let expected_payload_bytes = u64::try_from(expected_elements)
        .ok()
        .and_then(|elements| elements.checked_mul(4))
        .ok_or_else(|| execution_error(variant, "DMD-to-VAE latent byte count overflowed"))?;
    let expected_device_to_host_bytes = expected_payload_bytes
        .checked_mul(2)
        .ok_or_else(|| execution_error(variant, "DMD-to-VAE readback byte count overflowed"))?;
    let expected_total_transfer_bytes = expected_payload_bytes
        .checked_mul(3)
        .ok_or_else(|| execution_error(variant, "DMD-to-VAE transfer byte count overflowed"))?;
    let before = &report.packed_cache_before_cleanup;
    let after = &report.packed_cache_after_cleanup;
    let next_attempt_exact = report.preload_attempt_count.checked_add(1)
        == Some(report.expected_next_request_preload_attempt_count);
    let exact = report.policy == BROWSER_PACKED_F16_DMD_VAE_HANDOFF_POLICY
        && report.next_request_rehydration_policy
            == BROWSER_PACKED_F16_NEXT_REQUEST_REHYDRATION_POLICY
        && report.shape == expected_shape
        && report.dtype == "f32"
        && report.element_count == expected_elements
        && report.payload_bytes == expected_payload_bytes
        && report.device_to_host_readback_bytes == expected_device_to_host_bytes
        && report.host_to_device_upload_bytes == expected_payload_bytes
        && report.total_transfer_bytes == expected_total_transfer_bytes
        && report.before_sha256 == report.after_sha256
        && report.all_finite
        && report.not_all_zero
        && report.digest_matches
        && report.wrapper_cached_stages_before_clear == 0
        && report.wrapper_cached_stages_after_clear == 0
        && !report.synchronization_pending_before_cleanup
        && !report.synchronization_pending_after_cleanup
        && report.rope_cache_cleared
        && report.cleanup_completed
        && before.state == "ready"
        && before.cache_ready
        && before.cached_stages == plan.expected_stage_count
        && before.cached_objects == plan.expected_object_count
        && before.cached_tensors == plan.expected_tensor_count
        && before.cached_bytes == plan.retained_packed_f16_denoiser_bytes
        && after.state == "empty"
        && !after.cache_ready
        && after.cached_stages == 0
        && after.cached_objects == 0
        && after.cached_tensors == 0
        && after.cached_bytes == 0
        && report.preload_attempt_count > 0
        && next_attempt_exact;
    if !exact {
        return Err(execution_error(
            variant,
            format!(
                "browser packed-F16 DMD-to-VAE handoff differs from its exact request-scoped eviction contract: {report:?}"
            ),
        ));
    }
    Ok(())
}

fn validate_packed_f16_denoiser_preload(
    variant: BooguVariant,
    plan: BrowserPackedF16ResourcePlan,
    before: PackedF16DenoiserCacheAudit,
    after: PackedF16DenoiserCacheAudit,
) -> Result<(), RuntimeError> {
    let exact = after.state == PackedF16DenoiserCacheState::Ready
        && after.packed_cache_ready
        && after.cached_stage_count == plan.expected_stage_count
        && after.cached_object_count == plan.expected_object_count
        && after.cached_tensor_count == plan.expected_tensor_count
        && after.retained_packed_bytes == plan.retained_packed_f16_denoiser_bytes
        && after
            .packed_read_bytes
            .checked_sub(before.packed_read_bytes)
            == Some(plan.authenticated_artifact_bytes)
        && after
            .packed_upload_bytes
            .checked_sub(before.packed_upload_bytes)
            == Some(plan.retained_packed_f16_denoiser_bytes)
        && after.materialization_packed_read_bytes == before.materialization_packed_read_bytes
        && after.materialized_stage_count == before.materialized_stage_count
        && after.object_unpack_count == before.object_unpack_count
        && after.f32_write_bytes == before.f32_write_bytes
        && after
            .preload_attempt_count
            .checked_sub(before.preload_attempt_count)
            == Some(1)
        && after.failure_count == before.failure_count;
    if !exact {
        return Err(execution_error(
            variant,
            format!(
                "browser packed-F16 preload audit differs from its admitted plan: before={before:?}, after={after:?}"
            ),
        ));
    }
    Ok(())
}

fn browser_packed_f16_resource_plan_event(
    plan: BrowserPackedF16ResourcePlan,
) -> BrowserRuntimeEvent {
    BrowserRuntimeEvent::PackedF16ResourcePlan {
        qwen_text_layer_allocation_policy: plan.qwen_text_layer_allocation_policy,
        qwen_text_block_load_synchronization_policy: plan
            .qwen_text_block_load_synchronization_policy,
        qwen_text_layer_submission_policy: plan.qwen_text_layer_submission_policy,
        qwen_text_layer_persistent_pool_requires_measured_gpu_gate: plan
            .qwen_text_layer_persistent_pool_requires_measured_gpu_gate,
        authenticated_artifact_bytes: plan.authenticated_artifact_bytes,
        canonical_compact_f16_payload_bytes: plan.canonical_compact_f16_payload_bytes,
        retained_packed_f16_denoiser_bytes: plan.retained_packed_f16_denoiser_bytes,
        inserted_padding_elements: plan.inserted_padding_elements,
        padded_f16_elements: plan.padded_f16_elements,
        expected_stage_count: plan.expected_stage_count,
        expected_object_count: plan.expected_object_count,
        expected_tensor_count: plan.expected_tensor_count,
        max_packed_stage_bytes: plan.max_packed_stage_bytes,
        max_materialized_stage_f32_bytes: plan.max_materialized_stage_f32_bytes,
        max_packed_object_bytes: plan.max_packed_object_bytes,
        max_materialized_object_f32_bytes: plan.max_materialized_object_f32_bytes,
        materialized_f32_bytes_per_dmd_step: plan.materialized_f32_bytes_per_dmd_step,
        preload_workspace_bytes: plan.preload_workspace_bytes,
        preload_peak_bytes: plan.preload_peak_bytes,
        activation_reserve_bytes: plan.activation_reserve_bytes,
        conservative_planned_device_bytes: plan.conservative_planned_device_bytes,
        strict_device_cap_bytes: plan.strict_device_cap_bytes,
        expected_stage_materializations_per_request: plan
            .expected_stage_materializations_per_request,
        expected_object_unpacks_per_request: plan.expected_object_unpacks_per_request,
        expected_packed_read_bytes_per_request: plan.expected_packed_read_bytes_per_request,
        expected_f32_write_bytes_per_request: plan.expected_f32_write_bytes_per_request,
        on_device_quantized_execution_claimed: plan.on_device_quantized_execution_claimed,
    }
}

fn browser_packed_f16_denoiser_lifecycle_event(
    lifecycle: BrowserPackedF16DenoiserLifecycleReport,
) -> BrowserRuntimeEvent {
    BrowserRuntimeEvent::PackedF16DenoiserLifecycle { lifecycle }
}

fn browser_packed_f16_dmd_vae_handoff_event(
    run_id: RunId,
    report: BrowserPackedF16DmdVaeHandoffReport,
) -> BrowserRuntimeEvent {
    BrowserRuntimeEvent::PackedF16DmdVaeHandoff {
        run_id,
        report: Box::new(report),
    }
}

fn browser_packed_f16_pre_dmd_input_diagnostics_event(
    run_id: RunId,
    diagnostics: BrowserPackedF16PreDmdInputDiagnostics,
) -> BrowserRuntimeEvent {
    BrowserRuntimeEvent::PackedF16PreDmdInputDiagnostics {
        run_id,
        diagnostics: Box::new(diagnostics),
    }
}

fn browser_packed_f16_qwen_pre_handoff_diagnostics_event(
    run_id: RunId,
    diagnostics: BrowserPackedF16QwenPreHandoffDiagnostics,
) -> BrowserRuntimeEvent {
    BrowserRuntimeEvent::PackedF16QwenPreHandoffDiagnostics {
        run_id,
        diagnostics: Box::new(diagnostics),
    }
}

fn browser_packed_f16_qwen_host_embedding_event(
    run_id: RunId,
    report: HostRoutedEmbeddingReport,
) -> BrowserRuntimeEvent {
    BrowserRuntimeEvent::PackedF16QwenHostEmbedding {
        run_id,
        report: Box::new(report),
    }
}

fn browser_packed_f16_qwen_block0_post_sync_diagnostic_event(
    run_id: RunId,
    diagnostic: BrowserPackedF16QwenBlock0PostSyncDiagnostic,
) -> BrowserRuntimeEvent {
    BrowserRuntimeEvent::PackedF16QwenBlock0PostSyncDiagnostic {
        run_id,
        diagnostic: Box::new(diagnostic),
    }
}

fn browser_packed_f16_qwen_block0_execution_diagnostics_event(
    run_id: RunId,
    diagnostics: BrowserPackedF16QwenBlock0ExecutionDiagnostics,
) -> BrowserRuntimeEvent {
    BrowserRuntimeEvent::PackedF16QwenBlock0ExecutionDiagnostics {
        run_id,
        diagnostics: Box::new(diagnostics),
    }
}

fn browser_packed_f16_qwen_post_handoff_diagnostics_event(
    run_id: RunId,
    diagnostics: BrowserPackedF16QwenPostHandoffDiagnostics,
) -> BrowserRuntimeEvent {
    BrowserRuntimeEvent::PackedF16QwenPostHandoffDiagnostics {
        run_id,
        diagnostics: Box::new(diagnostics),
    }
}

/// Lazy snapshot audit proving the retained denoiser modules carry the planned runtime dtypes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct BrowserLowVramDenoiserDTypeAudit {
    /// Total retained parameter tensors inspected.
    pub tensor_count: usize,
    /// Tensors carrying exact Q8S block-32/F32 snapshots.
    pub q8s_block32_f32_tensor_count: usize,
    /// Tensors carrying ordinary F32 snapshots.
    pub f32_tensor_count: usize,
    /// Tensors carrying an unplanned dtype or quantization scheme.
    pub unexpected_dtype_tensor_count: usize,
    /// Values carried by Q8S snapshots.
    pub q8s_block32_f32_elements: u64,
    /// Values carried by F32 snapshots.
    pub f32_elements: u64,
    /// Derived packed Q8S values plus scale bytes.
    pub q8s_block32_f32_payload_bytes: u64,
    /// Derived F32 parameter bytes.
    pub f32_payload_bytes: u64,
    /// Whether counts, shapes, dtypes, and derived bytes match the release inventory.
    pub matches_inventory: bool,
}

fn validate_low_vram_denoiser_dtype_audit(
    variant: BooguVariant,
    plan: BrowserLowVramResourcePlan,
    observed: RetainedDenoiserDTypeAudit,
) -> Result<BrowserLowVramDenoiserDTypeAudit, RuntimeError> {
    let q8s_block32_f32_payload_bytes = observed
        .q8s_block32_f32_elements
        .checked_add(observed.q8s_block32_f32_elements / 32 * 4)
        .ok_or_else(|| execution_error(variant, "observed Q8S payload count overflowed"))?;
    let f32_payload_bytes = observed
        .f32_elements
        .checked_mul(4)
        .ok_or_else(|| execution_error(variant, "observed F32 payload count overflowed"))?;
    let matches_inventory = observed.unexpected_dtype_tensor_count == 0
        && observed.tensor_count
            == plan.expected_q8s_block32_f32_tensor_count + plan.expected_f32_tensor_count
        && observed.tensor_count
            == observed.q8s_block32_f32_tensor_count + observed.f32_tensor_count
        && observed.q8s_block32_f32_tensor_count == plan.expected_q8s_block32_f32_tensor_count
        && observed.f32_tensor_count == plan.expected_f32_tensor_count
        && observed.q8s_block32_f32_elements == plan.expected_q8s_block32_f32_elements
        && observed.f32_elements == plan.expected_f32_elements
        && q8s_block32_f32_payload_bytes == plan.expected_q8s_block32_f32_payload_bytes
        && f32_payload_bytes == plan.expected_f32_payload_bytes
        && q8s_block32_f32_payload_bytes.checked_add(f32_payload_bytes)
            == Some(plan.audited_retained_q8_denoiser_bytes);
    let audit = BrowserLowVramDenoiserDTypeAudit {
        tensor_count: observed.tensor_count,
        q8s_block32_f32_tensor_count: observed.q8s_block32_f32_tensor_count,
        f32_tensor_count: observed.f32_tensor_count,
        unexpected_dtype_tensor_count: observed.unexpected_dtype_tensor_count,
        q8s_block32_f32_elements: observed.q8s_block32_f32_elements,
        f32_elements: observed.f32_elements,
        q8s_block32_f32_payload_bytes,
        f32_payload_bytes,
        matches_inventory,
    };
    if !matches_inventory {
        return Err(execution_error(
            variant,
            format!(
                "browser low-vram retained denoiser dtype audit differs from inventory: {audit:?}"
            ),
        ));
    }
    Ok(audit)
}

fn validate_low_vram_denoiser_lifecycle(
    variant: BooguVariant,
    completed_steps: usize,
    expected_retained_stages: usize,
    retained_stages_before_clear: usize,
    synchronization_pending: bool,
    expected_retained_stages_after_request: usize,
    retained_stages_after_request: usize,
) -> Result<(), RuntimeError> {
    if completed_steps != 4 {
        return Err(execution_error(
            variant,
            format!("browser low-vram denoiser completed {completed_steps}/4 DMD steps"),
        ));
    }
    if synchronization_pending {
        return Err(execution_error(
            variant,
            "browser low-vram denoiser still has pending GPU work before cache clear",
        ));
    }
    if retained_stages_before_clear != expected_retained_stages {
        return Err(execution_error(
            variant,
            format!(
                "browser low-vram denoiser retained {retained_stages_before_clear}/{expected_retained_stages} verified stages"
            ),
        ));
    }
    if retained_stages_after_request != expected_retained_stages_after_request {
        return Err(execution_error(
            variant,
            format!(
                "browser low-vram denoiser retained {retained_stages_after_request}/{expected_retained_stages_after_request} stages after request finalization"
            ),
        ));
    }
    Ok(())
}

fn validate_low_vram_streamed_stage_lifecycle(
    variant: BooguVariant,
    qwen_retained_stages: usize,
    qwen_synchronization_pending: bool,
    vae_retained_stages: usize,
) -> Result<(), RuntimeError> {
    if qwen_synchronization_pending {
        return Err(execution_error(
            variant,
            "browser low-vram Qwen still has pending GPU work after its streamed forward",
        ));
    }
    if qwen_retained_stages != 0 || vae_retained_stages != 0 {
        return Err(execution_error(
            variant,
            format!(
                "browser low-vram requires streamed Qwen/VAE caches to stay empty; qwen={qwen_retained_stages}, vae={vae_retained_stages}"
            ),
        ));
    }
    Ok(())
}

// Decimal GB is intentional: accepting below 32,000,000,000 bytes is stricter than accepting
// below 32 GiB, so the public "under 32 GB" contract is unambiguous.
const BROWSER_LOW_VRAM_STRICT_DEVICE_CAP_BYTES: u64 = 32_000_000_000;
const BROWSER_LOW_VRAM_ACTIVATION_BUFFER_RESERVE_COUNT: u64 = 12;
const BROWSER_LOW_VRAM_RUNTIME_QUANTIZATION_BUFFER_RESERVE_COUNT: u64 = 2;
const BROWSER_TURBO_COMPACT_F16_PAYLOAD_BYTES: u64 = 19_869_996_096;
const BROWSER_TURBO_PACKED_F16_ARTIFACT_BYTES: u64 = 19_870_166_528;
const BROWSER_TURBO_PACKED_F16_RETAINED_BYTES: u64 = 19_870_010_624;
const BROWSER_TURBO_PACKED_F16_INSERTED_PADDING_ELEMENTS: u64 = 7_264;
const BROWSER_TURBO_PACKED_F16_PADDED_ELEMENTS: u64 = 9_935_005_312;
const BROWSER_TURBO_PACKED_F16_STAGE_COUNT: usize = 46;
const BROWSER_TURBO_PACKED_F16_OBJECT_COUNT: usize = 106;
const BROWSER_TURBO_PACKED_F16_TENSOR_COUNT: usize = 912;
const BROWSER_TURBO_PACKED_F16_MAX_STAGE_BYTES: u64 = 876_827_328;
const BROWSER_TURBO_PACKED_F16_MAX_F32_STAGE_BYTES: u64 = 1_753_654_656;
const BROWSER_TURBO_PACKED_F16_MAX_OBJECT_BYTES: u64 = 254_251_904;
const BROWSER_TURBO_PACKED_F16_MAX_F32_OBJECT_BYTES: u64 = 508_503_808;
const BROWSER_TURBO_PACKED_F16_F32_BYTES_PER_DMD_STEP: u64 = 39_740_021_248;
const BROWSER_TURBO_PACKED_F16_PRELOAD_WORKSPACE_BYTES: u64 = 2_434_252_800;
const BROWSER_TURBO_PACKED_F16_PRELOAD_PEAK_BYTES: u64 = 22_304_263_424;
const BROWSER_TURBO_PACKED_F16_ACTIVATION_RESERVE_BYTES: u64 = 4_868_505_600;
const BROWSER_TURBO_PACKED_F16_CONSERVATIVE_DEVICE_BYTES: u64 = 26_492_170_880;
const BROWSER_TURBO_PACKED_F16_STAGE_MATERIALIZATIONS_PER_REQUEST: u64 = 184;
const BROWSER_TURBO_PACKED_F16_OBJECT_UNPACKS_PER_REQUEST: u64 = 424;
const BROWSER_TURBO_PACKED_F16_READ_BYTES_PER_REQUEST: u64 = 79_480_042_496;
const BROWSER_TURBO_PACKED_F16_WRITE_BYTES_PER_REQUEST: u64 = 158_960_084_992;

fn validate_browser_packed_f16_qwen_embedding_plan(
    variant: BooguVariant,
    manifest: &ArtifactManifest,
    qwen_plan: &Qwen3VlStreamingPlan,
) -> Result<BrowserPackedF16QwenEmbeddingPlan, RuntimeError> {
    if variant != BooguVariant::Image01Turbo || qwen_plan.embedding_rows.chunks.len() != 6 {
        return Err(execution_error(
            variant,
            "browser packed-F16 host-routed Qwen embedding requires the six-chunk released Turbo plan",
        ));
    }
    let mut authenticated_object_bytes = 0_u64;
    let mut authenticated_f16_payload_bytes = 0_u64;
    let mut expected_object_count = 0_usize;
    for spec in &qwen_plan.embedding_rows.chunks {
        let component = format!("qwen-embedding-rows-{:02}", spec.chunk_index);
        let files = manifest
            .files
            .iter()
            .filter(|file| {
                file.role == burn_image::ArtifactFileRole::Weights
                    && file.component.as_ref().map(|value| value.as_str())
                        == Some(component.as_str())
            })
            .collect::<Vec<_>>();
        if files.len() != 1 {
            return Err(execution_error(
                variant,
                format!(
                    "browser host-routed Qwen component {component} has {} logical Burnpack objects, expected exactly one",
                    files.len()
                ),
            ));
        }
        authenticated_object_bytes = authenticated_object_bytes
            .checked_add(files[0].size)
            .ok_or_else(|| {
                execution_error(
                    variant,
                    "browser host-routed Qwen object byte count overflowed",
                )
            })?;
        authenticated_f16_payload_bytes = authenticated_f16_payload_bytes
            .checked_add(u64::try_from(spec.byte_len()).map_err(|_| {
                execution_error(
                    variant,
                    "browser host-routed Qwen F16 payload byte count does not fit u64",
                )
            })?)
            .ok_or_else(|| {
                execution_error(
                    variant,
                    "browser host-routed Qwen F16 payload byte count overflowed",
                )
            })?;
        expected_object_count = expected_object_count.checked_add(1).ok_or_else(|| {
            execution_error(variant, "browser host-routed Qwen object count overflowed")
        })?;
    }
    if authenticated_object_bytes <= authenticated_f16_payload_bytes {
        return Err(execution_error(
            variant,
            "browser host-routed Qwen objects do not include authenticated Burnpack framing",
        ));
    }
    Ok(BrowserPackedF16QwenEmbeddingPlan {
        expected_chunk_count: qwen_plan.embedding_rows.chunks.len(),
        expected_object_count,
        authenticated_object_bytes,
        authenticated_f16_payload_bytes,
    })
}

fn validate_browser_packed_f16_qwen_embedding_report(
    variant: BooguVariant,
    plan: BrowserPackedF16QwenEmbeddingPlan,
    report: &HostRoutedEmbeddingReport,
) -> Result<(), RuntimeError> {
    let expected_device_transfer_bytes =
        report
            .host_f32_payload_bytes
            .checked_mul(2)
            .ok_or_else(|| {
                execution_error(
                    variant,
                    "browser host-routed Qwen device transfer byte count overflowed",
                )
            })?;
    let exact = report.plan_chunk_count == plan.expected_chunk_count
        && report.authenticated_object_count == plan.expected_object_count
        && report.authenticated_object_bytes == plan.authenticated_object_bytes
        && report.authenticated_f16_payload_bytes == plan.authenticated_f16_payload_bytes
        && report.selected_row_occurrences == report.input_token_count
        && report.selected_unique_rows == report.unique_token_count
        && report.selected_f16_bytes.checked_mul(2) == Some(report.host_f32_payload_bytes)
        && report.host_to_device_upload_bytes == report.host_f32_payload_bytes
        && report.immediate_device_to_host_readback_bytes == report.host_f32_payload_bytes
        && report.total_device_transfer_bytes == expected_device_transfer_bytes
        && report.device_f32_sha256.as_deref() == Some(report.host_f32_sha256.as_str())
        && report.device_roundtrip_verified_before_text
        && report.device_roundtrip_digest_matches
        && report.all_finite
        && report.not_all_zero
        && report.coverage_complete;
    if !exact {
        return Err(execution_error(
            variant,
            format!(
                "browser packed-F16 Qwen host embedding differs from its exact manifest/transfer plan: plan={plan:?}, report={report:?}"
            ),
        ));
    }
    Ok(())
}

fn validate_browser_packed_f16_resource_plan(
    variant: BooguVariant,
    profile: BooguStorageProfile,
) -> Result<BrowserPackedF16ResourcePlan, RuntimeError> {
    validate_browser_packed_f16_resource_plan_with_cap(
        variant,
        profile,
        BROWSER_LOW_VRAM_STRICT_DEVICE_CAP_BYTES,
    )
}

fn validate_browser_packed_f16_resource_plan_with_cap(
    variant: BooguVariant,
    profile: BooguStorageProfile,
    strict_device_cap_bytes: u64,
) -> Result<BrowserPackedF16ResourcePlan, RuntimeError> {
    if variant != BooguVariant::Image01Turbo {
        return Err(execution_error(
            variant,
            "browser packed-F16 denoiser residency is restricted to ordinary Turbo",
        ));
    }
    if profile != BooguStorageProfile::F16QwenVisionF32 {
        return Err(execution_error(
            variant,
            "browser packed-F16 denoiser residency requires profile=production",
        ));
    }
    let exact_arithmetic = BROWSER_TURBO_COMPACT_F16_PAYLOAD_BYTES
        .checked_add(BROWSER_TURBO_PACKED_F16_INSERTED_PADDING_ELEMENTS * 2)
        == Some(BROWSER_TURBO_PACKED_F16_RETAINED_BYTES)
        && BROWSER_TURBO_PACKED_F16_PADDED_ELEMENTS.checked_mul(2)
            == Some(BROWSER_TURBO_PACKED_F16_RETAINED_BYTES)
        && BROWSER_TURBO_PACKED_F16_PADDED_ELEMENTS.checked_mul(4)
            == Some(BROWSER_TURBO_PACKED_F16_F32_BYTES_PER_DMD_STEP)
        && BROWSER_TURBO_PACKED_F16_RETAINED_BYTES
            .checked_add(BROWSER_TURBO_PACKED_F16_PRELOAD_WORKSPACE_BYTES)
            == Some(BROWSER_TURBO_PACKED_F16_PRELOAD_PEAK_BYTES)
        && BROWSER_TURBO_PACKED_F16_RETAINED_BYTES
            .checked_add(BROWSER_TURBO_PACKED_F16_MAX_F32_STAGE_BYTES)
            .and_then(|bytes| bytes.checked_add(BROWSER_TURBO_PACKED_F16_ACTIVATION_RESERVE_BYTES))
            == Some(BROWSER_TURBO_PACKED_F16_CONSERVATIVE_DEVICE_BYTES)
        && BROWSER_TURBO_PACKED_F16_RETAINED_BYTES.checked_mul(4)
            == Some(BROWSER_TURBO_PACKED_F16_READ_BYTES_PER_REQUEST)
        && BROWSER_TURBO_PACKED_F16_F32_BYTES_PER_DMD_STEP.checked_mul(4)
            == Some(BROWSER_TURBO_PACKED_F16_WRITE_BYTES_PER_REQUEST);
    if !exact_arithmetic {
        return Err(execution_error(
            variant,
            "browser packed-F16 aligned resource constants are internally inconsistent",
        ));
    }
    if BROWSER_TURBO_PACKED_F16_MAX_OBJECT_BYTES
        > crate::boogu::BOOGU_BROWSER_REQUESTED_BUFFER_LIMIT_BYTES
        || BROWSER_TURBO_PACKED_F16_MAX_F32_OBJECT_BYTES
            > crate::boogu::BOOGU_BROWSER_REQUESTED_BUFFER_LIMIT_BYTES
    {
        return Err(execution_error(
            variant,
            "browser packed-F16 object arena exceeds the admitted per-buffer limit",
        ));
    }
    if BROWSER_TURBO_PACKED_F16_PRELOAD_PEAK_BYTES >= strict_device_cap_bytes
        || BROWSER_TURBO_PACKED_F16_CONSERVATIVE_DEVICE_BYTES >= strict_device_cap_bytes
    {
        return Err(execution_error(
            variant,
            format!(
                "browser packed-F16 plan requires preload={} and inference={} bytes, which are not both strictly below the {strict_device_cap_bytes}-byte cap",
                BROWSER_TURBO_PACKED_F16_PRELOAD_PEAK_BYTES,
                BROWSER_TURBO_PACKED_F16_CONSERVATIVE_DEVICE_BYTES,
            ),
        ));
    }
    let plan = BrowserPackedF16ResourcePlan {
        qwen_text_layer_allocation_policy: Qwen3VlTextLayerAllocationPolicy::ExactSizePersistent
            .label(),
        qwen_text_block_load_synchronization_policy:
            Qwen3VlTextBlockLoadSynchronizationPolicy::PreForwardAndPostForward.label(),
        qwen_text_layer_submission_policy: BROWSER_PACKED_F16_QWEN_TEXT_LAYER_SUBMISSION_POLICY,
        // CubeCL does not expose a bound for this exact-size persistent pool here. Keep the
        // static estimate explicitly conservative/unmeasured and require the rendered aggregate
        // GPU-memory gate to prove the public cap.
        qwen_text_layer_persistent_pool_requires_measured_gpu_gate: true,
        authenticated_artifact_bytes: BROWSER_TURBO_PACKED_F16_ARTIFACT_BYTES,
        canonical_compact_f16_payload_bytes: BROWSER_TURBO_COMPACT_F16_PAYLOAD_BYTES,
        retained_packed_f16_denoiser_bytes: BROWSER_TURBO_PACKED_F16_RETAINED_BYTES,
        inserted_padding_elements: BROWSER_TURBO_PACKED_F16_INSERTED_PADDING_ELEMENTS,
        padded_f16_elements: BROWSER_TURBO_PACKED_F16_PADDED_ELEMENTS,
        expected_stage_count: BROWSER_TURBO_PACKED_F16_STAGE_COUNT,
        expected_object_count: BROWSER_TURBO_PACKED_F16_OBJECT_COUNT,
        expected_tensor_count: BROWSER_TURBO_PACKED_F16_TENSOR_COUNT,
        max_packed_stage_bytes: BROWSER_TURBO_PACKED_F16_MAX_STAGE_BYTES,
        max_materialized_stage_f32_bytes: BROWSER_TURBO_PACKED_F16_MAX_F32_STAGE_BYTES,
        max_packed_object_bytes: BROWSER_TURBO_PACKED_F16_MAX_OBJECT_BYTES,
        max_materialized_object_f32_bytes: BROWSER_TURBO_PACKED_F16_MAX_F32_OBJECT_BYTES,
        materialized_f32_bytes_per_dmd_step: BROWSER_TURBO_PACKED_F16_F32_BYTES_PER_DMD_STEP,
        preload_workspace_bytes: BROWSER_TURBO_PACKED_F16_PRELOAD_WORKSPACE_BYTES,
        preload_peak_bytes: BROWSER_TURBO_PACKED_F16_PRELOAD_PEAK_BYTES,
        activation_reserve_bytes: BROWSER_TURBO_PACKED_F16_ACTIVATION_RESERVE_BYTES,
        conservative_planned_device_bytes: BROWSER_TURBO_PACKED_F16_CONSERVATIVE_DEVICE_BYTES,
        strict_device_cap_bytes,
        expected_stage_materializations_per_request:
            BROWSER_TURBO_PACKED_F16_STAGE_MATERIALIZATIONS_PER_REQUEST,
        expected_object_unpacks_per_request: BROWSER_TURBO_PACKED_F16_OBJECT_UNPACKS_PER_REQUEST,
        expected_packed_read_bytes_per_request: BROWSER_TURBO_PACKED_F16_READ_BYTES_PER_REQUEST,
        expected_f32_write_bytes_per_request: BROWSER_TURBO_PACKED_F16_WRITE_BYTES_PER_REQUEST,
        on_device_quantized_execution_claimed: false,
    };
    set_browser_factory_progress(format!(
        "Model setup: packed-F16 Turbo GPU plan accepted at {:.1} GiB (conservative, not measured)",
        plan.conservative_planned_device_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    ));
    dispatch_browser_event(
        BROWSER_RUNTIME_EVENT_NAME,
        &browser_packed_f16_resource_plan_event(plan),
    );
    Ok(plan)
}

fn max_streamed_qwen_stage_f32_bytes(
    variant: BooguVariant,
    qwen_plan: &Qwen3VlStreamingPlan,
) -> Result<u64, RuntimeError> {
    let bytes = qwen_plan
        .stages
        .iter()
        .filter(|descriptor| !matches!(descriptor.stage, Qwen3VlStage::LmHeadRows { .. }))
        .map(|descriptor| {
            descriptor.byte_len(size_of::<f32>()).ok_or_else(|| {
                execution_error(
                    variant,
                    format!(
                        "browser low-vram F32 byte count overflowed for Qwen stage {:?}",
                        descriptor.stage
                    ),
                )
            })
        })
        .try_fold(0_usize, |maximum, bytes| {
            bytes.map(|bytes| maximum.max(bytes))
        })?;
    if bytes == 0 {
        return Err(execution_error(
            variant,
            "browser low-vram Qwen semantic-stage plan is empty",
        ));
    }
    u64::try_from(bytes).map_err(|_| {
        execution_error(
            variant,
            "browser low-vram Qwen semantic-stage byte count does not fit u64",
        )
    })
}

fn validate_browser_low_vram_resource_plan(
    variant: BooguVariant,
    profile: BooguStorageProfile,
    qwen_plan: &Qwen3VlStreamingPlan,
    inventory: &BooguArtifactInventory,
    quantized_linear_execution_policy: BooguQuantizedLinearExecutionPolicy,
) -> Result<BrowserLowVramResourcePlan, RuntimeError> {
    validate_browser_low_vram_resource_plan_with_cap(
        variant,
        profile,
        qwen_plan,
        inventory,
        quantized_linear_execution_policy,
        BROWSER_LOW_VRAM_STRICT_DEVICE_CAP_BYTES,
    )
}

fn validate_browser_low_vram_resource_plan_with_cap(
    variant: BooguVariant,
    profile: BooguStorageProfile,
    qwen_plan: &Qwen3VlStreamingPlan,
    inventory: &BooguArtifactInventory,
    quantized_linear_execution_policy: BooguQuantizedLinearExecutionPolicy,
    strict_device_cap_bytes: u64,
) -> Result<BrowserLowVramResourcePlan, RuntimeError> {
    if profile != BooguStorageProfile::F16QwenVisionF32 {
        return Err(execution_error(
            variant,
            "browser low-vram mode requires the unchanged canonical profile=production artifacts",
        ));
    }
    if quantized_linear_execution_policy
        == BooguQuantizedLinearExecutionPolicy::DenseF32PerSemanticStage
        && variant != BooguVariant::Image01Turbo
    {
        return Err(execution_error(
            variant,
            "bounded dense-F32 retained-Q8 execution is restricted to Turbo",
        ));
    }
    let footprint = inventory
        .denoiser_runtime_q8s_block32_f32_footprint(variant)
        .map_err(|error| execution_error(variant, error))?;
    let audited_retained_q8_denoiser_bytes = footprint.total_payload_bytes;
    // The exact verified Qwen source plan is itself derived from the validated model inventory.
    // Count every semantic module at the browser execution dtype, including the row-sliced
    // embedding chunks, instead of relying on a hand-maintained stage-size constant.
    let audited_max_streamed_qwen_stage_f32_bytes =
        max_streamed_qwen_stage_f32_bytes(variant, qwen_plan)?;
    let audited_loaded_vae_module_f32_bytes =
        inventory_stage_f32_payloads(variant, inventory, TensorOwner::FluxVae)?
            .into_values()
            .try_fold(0_u64, |total, bytes| total.checked_add(bytes))
            .ok_or_else(|| {
                execution_error(variant, "browser loaded VAE module byte count overflowed")
            })?;
    let audited_max_dense_denoiser_stage_f32_bytes = if quantized_linear_execution_policy
        == BooguQuantizedLinearExecutionPolicy::DenseF32PerSemanticStage
    {
        max_inventory_stage_f32_bytes(variant, inventory, TensorOwner::BooguDenoiser)?
    } else {
        0
    };
    let audited_max_phase_local_f32_stage_bytes = audited_max_streamed_qwen_stage_f32_bytes
        .max(audited_loaded_vae_module_f32_bytes)
        .max(audited_max_dense_denoiser_stage_f32_bytes);
    // Two maximum-size buffers cover the verified F16 source stage, packed Q8 destination, and
    // transient loader/quantizer workspace. The activation reserve covers the released Edit
    // execution shape.
    let runtime_quantization_workspace_bytes =
        crate::boogu::BOOGU_BROWSER_REQUESTED_BUFFER_LIMIT_BYTES
            .checked_mul(BROWSER_LOW_VRAM_RUNTIME_QUANTIZATION_BUFFER_RESERVE_COUNT)
            .ok_or_else(|| execution_error(variant, "low-vram quantization plan overflowed"))?;
    let activation_reserve_bytes = crate::boogu::BOOGU_BROWSER_REQUESTED_BUFFER_LIMIT_BYTES
        .checked_mul(BROWSER_LOW_VRAM_ACTIVATION_BUFFER_RESERVE_COUNT)
        .ok_or_else(|| execution_error(variant, "low-vram activation plan overflowed"))?;
    let conservative_planned_device_bytes = audited_retained_q8_denoiser_bytes
        .checked_add(audited_max_phase_local_f32_stage_bytes)
        .and_then(|bytes| bytes.checked_add(runtime_quantization_workspace_bytes))
        .and_then(|bytes| bytes.checked_add(activation_reserve_bytes))
        .ok_or_else(|| execution_error(variant, "low-vram device-byte plan overflowed"))?;
    if conservative_planned_device_bytes >= strict_device_cap_bytes {
        return Err(execution_error(
            variant,
            format!(
                "browser low-vram plan requires {conservative_planned_device_bytes} bytes, which is not strictly below the {}-byte cap",
                strict_device_cap_bytes
            ),
        ));
    }
    let plan = BrowserLowVramResourcePlan {
        audited_retained_q8_denoiser_bytes,
        expected_q8s_block32_f32_tensor_count: footprint.quantized_tensor_count,
        expected_f32_tensor_count: footprint.f32_tensor_count,
        expected_q8s_block32_f32_elements: footprint.quantized_elements,
        expected_f32_elements: footprint.f32_elements,
        expected_q8s_block32_f32_payload_bytes: footprint.quantized_payload_bytes,
        expected_f32_payload_bytes: footprint.f32_payload_bytes,
        audited_max_streamed_qwen_stage_f32_bytes,
        audited_loaded_vae_module_f32_bytes,
        audited_max_dense_denoiser_stage_f32_bytes,
        audited_max_phase_local_f32_stage_bytes,
        runtime_quantization_workspace_bytes,
        activation_reserve_bytes,
        conservative_planned_device_bytes,
        strict_device_cap_bytes,
    };
    set_browser_factory_progress(format!(
        "Model setup: low-VRAM GPU plan accepted at {:.1} GiB (conservative, not measured)",
        plan.conservative_planned_device_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    ));
    dispatch_browser_event(
        BROWSER_RUNTIME_EVENT_NAME,
        &browser_low_vram_resource_plan_event(plan, quantized_linear_execution_policy),
    );
    Ok(plan)
}

fn inventory_stage_f32_payloads(
    variant: BooguVariant,
    inventory: &BooguArtifactInventory,
    owner: TensorOwner,
) -> Result<BTreeMap<String, u64>, RuntimeError> {
    let mut stages = BTreeMap::<String, u64>::new();
    for spec in inventory
        .tensors()
        .iter()
        .filter(|spec| spec.owner == owner)
    {
        if owner == TensorOwner::BooguDenoiser
            && !variant.is_edit()
            && spec.stage.starts_with("boogu-reference-refiner-")
        {
            continue;
        }
        let elements = spec
            .target_shape
            .iter()
            .try_fold(1_u64, |total, &dimension| {
                total.checked_mul(dimension as u64)
            })
            .ok_or_else(|| {
                execution_error(
                    variant,
                    format!("F32 element count overflowed for {}", spec.target_name),
                )
            })?;
        let bytes = elements
            .checked_mul(size_of::<f32>() as u64)
            .ok_or_else(|| {
                execution_error(
                    variant,
                    format!("F32 byte count overflowed for {}", spec.target_name),
                )
            })?;
        let stage = stages.entry(spec.stage.clone()).or_default();
        *stage = stage.checked_add(bytes).ok_or_else(|| {
            execution_error(
                variant,
                format!(
                    "F32 semantic-stage byte count overflowed for {}",
                    spec.stage
                ),
            )
        })?;
    }
    if stages.is_empty() {
        return Err(execution_error(
            variant,
            format!("browser F32 inventory has no {owner:?} semantic stages"),
        ));
    }
    Ok(stages)
}

fn max_inventory_stage_f32_bytes(
    variant: BooguVariant,
    inventory: &BooguArtifactInventory,
    owner: TensorOwner,
) -> Result<u64, RuntimeError> {
    inventory_stage_f32_payloads(variant, inventory, owner)?
        .into_values()
        .max()
        .ok_or_else(|| execution_error(variant, "browser F32 stage plan is empty"))
}

// The source-bound Turbo 1024 browser run peaked at 42,345,693,184 framebuffer bytes during
// VAE/output transition while retaining 35,110,256,204 bytes of packed parameters. Twenty maximum
// applied-buffer slots cover that measured non-weight residency with two slots of guard headroom;
// eight slots did not, so the old plan was not conservative despite execution succeeding.
const BROWSER_RESIDENT_MAX_SIMULTANEOUS_ACTIVATION_BUFFERS: u64 = 20;
// Q4S removes the dominant parameter residency, including the vocabulary table, but the strict
// decimal-16-GB target still needs a bounded activation/allocator envelope. Eleven maximum
// applied-buffer slots leave a fail-closed model-side margin while avoiding request-time model
// eviction. The native GPU monitor remains the authority for measured whole-process qualification.
const BROWSER_RESIDENT_Q4_MAX_SIMULTANEOUS_ACTIVATION_BUFFERS: u64 = 11;
const BROWSER_RESIDENT_Q4_STRICT_DEVICE_CAP_BYTES: u64 = 16_000_000_000;

fn validate_browser_resident_resource_plan(
    variant: BooguVariant,
    manifest: &ArtifactManifest,
    inventory: &BooguArtifactInventory,
    profile: BooguStorageProfile,
    float_policy: BooguFloatLoadPolicy,
) -> Result<BrowserResidentResourcePlan, RuntimeError> {
    let stored_weight_bytes = manifest
        .files
        .iter()
        .filter(|file| matches!(file.role, burn_image::ArtifactFileRole::Weights))
        .filter(|file| {
            browser_resident_artifact_required(
                variant,
                file.component.as_ref().map(|component| component.as_str()),
            )
        })
        .try_fold(0_u64, |total, file| total.checked_add(file.size))
        .ok_or_else(|| execution_error(variant, "resident browser weight-byte plan overflowed"))?;
    if stored_weight_bytes == 0 {
        return Err(execution_error(
            variant,
            "resident browser resource plan contains no model weights",
        ));
    }
    let (
        weight_storage_policy,
        packed_q4_value_bytes,
        q4_scale_bytes,
        packed_f16_weight_bytes,
        f32_auxiliary_weight_bytes,
        resident_weight_bytes,
    ) = match float_policy {
        BooguFloatLoadPolicy::PackedF16WeightsF32Auxiliaries => {
            let footprint = inventory
                .packed_f16_resident_footprint(variant, profile)
                .map_err(|error| execution_error(variant, error))?;
            (
                "packed-f16-weights-f32-auxiliaries",
                0,
                0,
                footprint.packed_f16_payload_bytes,
                footprint.f32_payload_bytes,
                footprint.total_payload_bytes,
            )
        }
        BooguFloatLoadPolicy::PackedQ4sWeightsF32Auxiliaries => {
            let footprint: BooguPackedQ4ResidentFootprint = inventory
                .packed_q4_resident_footprint(variant, profile)
                .map_err(|error| execution_error(variant, error))?;
            (
                "packed-q4s-block-up-to-128/f32-scales/packed-f16-convolutions/f32-auxiliaries",
                footprint.packed_q4_value_bytes,
                footprint.scale_bytes,
                footprint.packed_f16_payload_bytes,
                footprint.f32_auxiliary_payload_bytes,
                footprint.total_payload_bytes,
            )
        }
        BooguFloatLoadPolicy::AdaptToF32 => {
            // This explicit F32 route keeps the complete logical payload in dense F32. Doubling
            // Burnpack bytes is deliberately conservative and includes framing/alignment.
            let resident_weight_bytes = stored_weight_bytes.checked_mul(2).ok_or_else(|| {
                execution_error(variant, "resident browser F32 weight-byte plan overflowed")
            })?;
            (
                "dense-f32",
                0,
                0,
                0,
                resident_weight_bytes,
                resident_weight_bytes,
            )
        }
        BooguFloatLoadPolicy::Preserve => {
            return Err(execution_error(
                variant,
                "resident browser plan cannot preserve typed F16 execution",
            ));
        }
    };
    if resident_weight_bytes == 0 {
        return Err(execution_error(
            variant,
            "resident browser parameter footprint is empty",
        ));
    }
    if float_policy == BooguFloatLoadPolicy::PackedF16WeightsF32Auxiliaries
        && packed_f16_weight_bytes > stored_weight_bytes
    {
        return Err(execution_error(
            variant,
            format!(
                "packed-F16 resident payload {} exceeds the active sealed weight bytes {stored_weight_bytes}",
                packed_f16_weight_bytes,
            ),
        ));
    }
    if float_policy == BooguFloatLoadPolicy::PackedQ4sWeightsF32Auxiliaries
        && (packed_q4_value_bytes == 0 || q4_scale_bytes == 0)
    {
        return Err(execution_error(
            variant,
            "packed-Q4S resident footprint omits values or block scales",
        ));
    }
    let activation_buffer_count =
        if float_policy == BooguFloatLoadPolicy::PackedQ4sWeightsF32Auxiliaries {
            BROWSER_RESIDENT_Q4_MAX_SIMULTANEOUS_ACTIVATION_BUFFERS
        } else {
            BROWSER_RESIDENT_MAX_SIMULTANEOUS_ACTIVATION_BUFFERS
        };
    let activation_reserve_bytes = crate::boogu::BOOGU_BROWSER_MAX_APPLIED_BUFFER_BYTES
        .checked_mul(activation_buffer_count)
        .ok_or_else(|| execution_error(variant, "resident browser activation plan overflowed"))?;
    let conservative_planned_device_bytes = resident_weight_bytes
        .checked_add(activation_reserve_bytes)
        .ok_or_else(|| execution_error(variant, "resident browser device-byte plan overflowed"))?;
    let strict_device_cap_bytes = (float_policy
        == BooguFloatLoadPolicy::PackedQ4sWeightsF32Auxiliaries)
        .then_some(BROWSER_RESIDENT_Q4_STRICT_DEVICE_CAP_BYTES);
    if strict_device_cap_bytes.is_some_and(|cap| conservative_planned_device_bytes >= cap) {
        return Err(execution_error(
            variant,
            format!(
                "packed-Q4S conservative device plan {conservative_planned_device_bytes} is not strictly below the {BROWSER_RESIDENT_Q4_STRICT_DEVICE_CAP_BYTES}-byte cap",
            ),
        ));
    }
    let plan = BrowserResidentResourcePlan {
        weight_storage_policy,
        stored_weight_bytes,
        packed_q4_value_bytes,
        q4_scale_bytes,
        packed_f16_weight_bytes,
        f32_auxiliary_weight_bytes,
        resident_weight_bytes,
        activation_reserve_bytes,
        conservative_planned_device_bytes,
        strict_device_cap_bytes,
    };
    set_browser_factory_progress(format!(
        "Model setup: GPU residency plan accepted; preloading {:.1} GiB of {} weights",
        plan.resident_weight_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        plan.weight_storage_policy,
    ));
    dispatch_browser_event(
        BROWSER_RUNTIME_EVENT_NAME,
        &BrowserRuntimeEvent::ResidentResourcePlan {
            weight_storage_policy: plan.weight_storage_policy,
            stored_weight_bytes: plan.stored_weight_bytes,
            packed_q4_value_bytes: plan.packed_q4_value_bytes,
            q4_scale_bytes: plan.q4_scale_bytes,
            packed_f16_weight_bytes: plan.packed_f16_weight_bytes,
            f32_auxiliary_weight_bytes: plan.f32_auxiliary_weight_bytes,
            resident_weight_bytes: plan.resident_weight_bytes,
            activation_reserve_bytes: plan.activation_reserve_bytes,
            conservative_planned_device_bytes: plan.conservative_planned_device_bytes,
            strict_device_cap_bytes: plan.strict_device_cap_bytes,
        },
    );
    Ok(plan)
}

fn browser_resident_artifact_required(variant: BooguVariant, component: Option<&str>) -> bool {
    if variant.is_edit() {
        return true;
    }
    let component = component.unwrap_or_default();
    !component.starts_with("qwen-vision-")
        && component != "flux-vae-encoder"
        && !component.starts_with("boogu-reference-refiner-")
}

const BROWSER_VRAM_PREFLIGHT_TIMEOUT_MS: i32 = 45_000;

async fn browser_vram_preflight_timeout() {
    use wasm_bindgen::JsValue;
    use wasm_bindgen_futures::JsFuture;

    let promise = js_sys::Promise::new(&mut |resolve, reject| {
        let result = web_sys::window()
            .ok_or_else(|| JsValue::from_str("Window is unavailable"))
            .and_then(|window| {
                window.set_timeout_with_callback_and_timeout_and_arguments_0(
                    &resolve,
                    BROWSER_VRAM_PREFLIGHT_TIMEOUT_MS,
                )
            });
        if let Err(error) = result {
            let _ = reject.call1(&JsValue::UNDEFINED, &error);
        }
    });
    let _ = JsFuture::from(promise).await;
}

async fn run_browser_vram_preflight(
    variant: BooguVariant,
    policy: &'static str,
    required_device_bytes: u64,
    applied_max_buffer_size: u64,
    allocation_device: &crate::backend::SharedWgpuAllocationDevice,
) -> Result<(), RuntimeError> {
    use futures::future::{Either, select};

    let chunks =
        crate::boogu::browser_vram_preflight_chunks(required_device_bytes, applied_max_buffer_size)
            .map_err(|error| execution_error(variant, error))?;
    let largest_allocation_bytes = chunks.iter().copied().max().unwrap_or_default();
    let model = boogu_model_descriptor(variant).id.to_string();
    report_browser_runtime_preparing(format!(
        "GPU memory preflight: committing {:.1} GiB for {model} before downloading model weights",
        required_device_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    ));
    dispatch_browser_event(
        BROWSER_RUNTIME_EVENT_NAME,
        &BrowserRuntimeEvent::VramPreflight {
            status: "started",
            model: model.clone(),
            policy,
            required_device_bytes,
            allocation_count: chunks.len(),
            largest_allocation_bytes,
            allocations_committed: false,
            shared_device_and_queue: true,
        },
    );

    let device = allocation_device.device();
    let queue = allocation_device.queue();
    let out_of_memory_scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let internal_scope = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let validation_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("burn-image-browser-vram-preflight"),
    });
    let buffers = chunks
        .iter()
        .map(|&size| {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("burn-image-browser-vram-preflight-reservation"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            // Creation alone may be lazy. Clearing every retained buffer makes the browser submit
            // real writes against the complete simultaneous residency before any weight part is
            // requested.
            encoder.clear_buffer(&buffer, 0, None);
            buffer
        })
        .collect::<Vec<_>>();
    queue.submit([encoder.finish()]);

    let (completed_tx, completed_rx) = futures::channel::oneshot::channel();
    queue.on_submitted_work_done(move || {
        let _ = completed_tx.send(());
    });
    let work = async move {
        let (completion, validation, internal, out_of_memory) = futures::join!(
            completed_rx,
            validation_scope.pop(),
            internal_scope.pop(),
            out_of_memory_scope.pop(),
        );
        completion.map_err(|_| "the WebGPU queue dropped its preflight completion".to_owned())?;
        for (kind, error) in [
            ("validation", validation),
            ("internal", internal),
            ("out-of-memory", out_of_memory),
        ] {
            if let Some(error) = error {
                return Err(format!("WebGPU {kind} error: {error}"));
            }
        }
        Ok::<(), String>(())
    };
    futures::pin_mut!(work);
    let timeout = browser_vram_preflight_timeout();
    futures::pin_mut!(timeout);
    let result = match select(work, timeout).await {
        Either::Left((result, _)) => result,
        Either::Right(((), _)) => Err(format!(
            "WebGPU did not commit the preflight allocations within {} seconds",
            BROWSER_VRAM_PREFLIGHT_TIMEOUT_MS / 1_000
        )),
    };
    for buffer in &buffers {
        buffer.destroy();
    }
    if let Err(error) = result {
        dispatch_browser_event(
            BROWSER_RUNTIME_EVENT_NAME,
            &BrowserRuntimeEvent::VramPreflight {
                status: "failed",
                model: model.clone(),
                policy,
                required_device_bytes,
                allocation_count: chunks.len(),
                largest_allocation_bytes,
                allocations_committed: false,
                shared_device_and_queue: true,
            },
        );
        return Err(execution_error(
            variant,
            format!(
                "GPU memory preflight failed before model-weight download: could not commit {required_device_bytes} bytes on the shared WebGPU device ({error})"
            ),
        ));
    }

    dispatch_browser_event(
        BROWSER_RUNTIME_EVENT_NAME,
        &BrowserRuntimeEvent::VramPreflight {
            status: "passed",
            model,
            policy,
            required_device_bytes,
            allocation_count: chunks.len(),
            largest_allocation_bytes,
            allocations_committed: true,
            shared_device_and_queue: true,
        },
    );
    report_browser_runtime_preparing(format!(
        "GPU memory preflight passed at {:.1} GiB; starting verified model download",
        required_device_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    ));
    Ok(())
}

pub(crate) fn report_browser_runtime_preparing(message: impl Into<String>) {
    let message = message.into();
    set_browser_factory_progress(message.clone());
    dispatch_browser_event(
        BROWSER_RUNTIME_EVENT_NAME,
        &BrowserRuntimeEvent::Preparing { message },
    );
}

thread_local! {
    static BROWSER_FACTORY_PROGRESS: std::cell::RefCell<Option<String>> = const {
        std::cell::RefCell::new(None)
    };
    static BROWSER_SURFACE_OVERLAY_ACTIVE: std::cell::Cell<bool> = const {
        std::cell::Cell::new(false)
    };
}

fn set_browser_factory_progress(message: String) {
    BROWSER_FACTORY_PROGRESS.with(|progress| *progress.borrow_mut() = Some(message));
}

fn take_browser_factory_progress() -> Option<String> {
    BROWSER_FACTORY_PROGRESS.with(|progress| progress.borrow_mut().take())
}

fn set_browser_surface_overlay_active(active: bool) {
    BROWSER_SURFACE_OVERLAY_ACTIVE.with(|state| state.set(active));
}

fn browser_surface_overlay_active() -> bool {
    BROWSER_SURFACE_OVERLAY_ACTIVE.with(std::cell::Cell::get)
}

fn report_browser_manifest_verified(manifest: &ArtifactManifest) {
    let weight_objects = manifest
        .files
        .iter()
        .filter(|file| matches!(file.role, burn_image::ArtifactFileRole::Weights))
        .count();
    let weight_bytes = manifest
        .files
        .iter()
        .filter(|file| matches!(file.role, burn_image::ArtifactFileRole::Weights))
        .fold(0_u64, |total, file| total.saturating_add(file.size));
    set_browser_factory_progress(format!(
        "Model setup: manifest verified; {weight_objects} weight objects declared"
    ));
    dispatch_browser_event(
        BROWSER_RUNTIME_EVENT_NAME,
        &BrowserRuntimeEvent::ManifestVerified {
            bundle: manifest.bundle.to_string(),
            weight_objects: u32::try_from(weight_objects).unwrap_or(u32::MAX),
            weight_bytes,
        },
    );
}

fn report_browser_runtime_ready(
    variant: BooguVariant,
    selected_model_device_resident: bool,
    transfer: Option<burn_image::ArtifactTransferProgress>,
    qwen_text_layer_allocation_policy: &'static str,
    qwen_text_block_load_synchronization_policy: &'static str,
    qwen_text_layer_submission_policy: &'static str,
) {
    let selected_model_cache_complete = transfer.as_ref().is_some_and(|progress| {
        progress.total_bytes > 0
            && progress.loaded_bytes == progress.total_bytes
            && progress.logical_objects_completed == progress.logical_objects_total
            && progress.physical_parts_completed == progress.physical_parts_total
            && progress.bounded_ranges_completed == progress.bounded_ranges_total
    });
    dispatch_browser_event(
        BROWSER_RUNTIME_EVENT_NAME,
        &BrowserRuntimeEvent::Ready {
            model: boogu_model_descriptor(variant).id.to_string(),
            request_enabled: true,
            selected_model_cache_complete,
            selected_model_device_resident,
            transfer,
            block0_execution_mode: browser_qwen_block0_execution_mode(),
            qwen_text_layer_allocation_policy,
            qwen_text_block_load_synchronization_policy,
            qwen_text_layer_submission_policy,
        },
    );
}

pub(crate) fn report_browser_runtime_failure(message: impl Into<String>) {
    dispatch_browser_event(
        BROWSER_RUNTIME_EVENT_NAME,
        &BrowserRuntimeEvent::Failed {
            message: message.into(),
        },
    );
}

pub(crate) fn report_browser_surface_inference_suspended(
    run_id: u64,
    primary_window_camera_count: usize,
    saved_camera_state_count: usize,
    previously_active_camera_count: usize,
    inactive_camera_count: usize,
    active_job_count: usize,
) {
    set_browser_surface_overlay_active(true);
    dispatch_browser_event(
        BROWSER_RUNTIME_EVENT_NAME,
        &BrowserRuntimeEvent::SurfaceInferenceSuspended {
            run_id: RunId(run_id),
            policy: crate::boogu::BROWSER_SURFACE_INFERENCE_POLICY,
            primary_window_camera_count,
            saved_camera_state_count,
            previously_active_camera_count,
            inactive_camera_count,
            active_job_count,
            suspended_before_runtime_submit: true,
            all_primary_window_cameras_inactive: true,
        },
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn report_browser_surface_inference_resumed(
    run_id: u64,
    terminal: &'static str,
    primary_window_camera_count: usize,
    saved_camera_state_count: usize,
    restored_camera_state_count: usize,
    restored_active_camera_count: usize,
    active_job_count: usize,
    exact_saved_states_restored: bool,
    all_primary_window_cameras_restored: bool,
) {
    dispatch_browser_event(
        BROWSER_RUNTIME_EVENT_NAME,
        &BrowserRuntimeEvent::SurfaceInferenceResumed {
            run_id: RunId(run_id),
            policy: crate::boogu::BROWSER_SURFACE_INFERENCE_POLICY,
            terminal,
            primary_window_camera_count,
            saved_camera_state_count,
            restored_camera_state_count,
            restored_active_camera_count,
            active_job_count,
            resumed_after_runtime_terminal: true,
            resumed_before_output_ready: true,
            exact_saved_states_restored,
            all_primary_window_cameras_restored,
        },
    );
    set_browser_surface_overlay_active(false);
}

pub(crate) fn report_browser_surface_inference_gate_failure(
    run_id: u64,
    phase: &'static str,
    message: &str,
    exact_saved_states_restored: bool,
) {
    dispatch_browser_event(
        BROWSER_RUNTIME_EVENT_NAME,
        &BrowserRuntimeEvent::SurfaceInferenceGateFailed {
            run_id: RunId(run_id),
            policy: crate::boogu::BROWSER_SURFACE_INFERENCE_POLICY,
            phase,
            message: message.into(),
            exact_saved_states_restored,
        },
    );
}

fn dispatch_browser_progress(event: &ProgressEvent) {
    let transfer = match event {
        ProgressEvent::ArtifactStarted { transfer, .. }
        | ProgressEvent::ArtifactProgress { transfer, .. }
        | ProgressEvent::ArtifactVerified { transfer, .. } => transfer.as_ref(),
        _ => None,
    };
    let status = transfer
        .map(browser_transfer_progress_status)
        .or_else(|| match event {
            ProgressEvent::ArtifactStarted { path, .. } => Some(format!(
                "Model setup: streaming {path}; overall totals unavailable"
            )),
            ProgressEvent::ArtifactProgress {
                loaded_bytes,
                total_bytes,
                ..
            } => Some(format!(
                "Model setup: loading current object {:.1}%",
                100.0 * *loaded_bytes as f64 / (*total_bytes).max(1) as f64
            )),
            ProgressEvent::ArtifactVerified { path, .. } => {
                Some(format!("Model setup: verified {path}"))
            }
            _ => None,
        });
    if let Some(status) = status {
        set_browser_factory_progress(status);
    }
    dispatch_browser_event(BROWSER_PROGRESS_EVENT_NAME, event);
}

fn browser_transfer_progress_status(progress: &burn_image::ArtifactTransferProgress) -> String {
    let percent = 100.0 * progress.loaded_bytes as f64 / progress.total_bytes.max(1) as f64;
    let component = progress
        .component
        .as_ref()
        .map(|component| format!(" - {component}"))
        .unwrap_or_default();
    let rate = progress
        .bytes_per_second
        .map(|bytes| format!(" - {}/s", format_transfer_bytes(bytes)))
        .unwrap_or_default();
    let eta = progress
        .eta_seconds
        .map(|seconds| format!(" - ETA {}", format_transfer_duration(seconds)))
        .unwrap_or_default();
    format!(
        "Model transfer {percent:.1}%: {}{component} - {} / {} - {}/{} logical objects - {}/{} unique parts - {}/{} bounded reads{rate}{eta}",
        progress.phase,
        format_transfer_bytes(progress.loaded_bytes),
        format_transfer_bytes(progress.total_bytes),
        progress.logical_objects_completed,
        progress.logical_objects_total,
        progress.physical_parts_completed,
        progress.physical_parts_total,
        progress.bounded_ranges_completed,
        progress.bounded_ranges_total,
    )
}

fn validate_browser_turbo_active_transfer_plan(
    progress: &burn_image::ArtifactTransferProgress,
) -> Result<(), String> {
    let actual = (
        progress.logical_objects_total,
        progress.physical_parts_total,
        progress.bounded_ranges_total,
        progress.total_bytes,
    );
    let expected = (
        BROWSER_TURBO_ACTIVE_LOGICAL_OBJECTS,
        BROWSER_TURBO_ACTIVE_UNIQUE_TRANSPORT_PARTS,
        u64::from(BROWSER_TURBO_ACTIVE_UNIQUE_TRANSPORT_PARTS),
        BROWSER_TURBO_ACTIVE_TRANSPORT_BYTES,
    );
    if actual != expected {
        return Err(format!(
            "canonical Turbo active transport plan is {actual:?}; expected {expected:?} after excluding non-executable Qwen vision, VAE encoder, and reference-refiner stages"
        ));
    }
    Ok(())
}

fn format_transfer_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = 1024.0 * KIB;
    const GIB: f64 = 1024.0 * MIB;
    let bytes = bytes as f64;
    if bytes >= GIB {
        format!("{:.2} GiB", bytes / GIB)
    } else if bytes >= MIB {
        format!("{:.1} MiB", bytes / MIB)
    } else if bytes >= KIB {
        format!("{:.1} KiB", bytes / KIB)
    } else {
        format!("{bytes:.0} B")
    }
}

fn format_transfer_duration(seconds: u64) -> String {
    if seconds < 60 {
        format!("{}s", seconds.max(1))
    } else if seconds < 3_600 {
        format!("{}m {}s", seconds / 60, seconds % 60)
    } else {
        format!("{}h {}m", seconds / 3_600, (seconds % 3_600) / 60)
    }
}

fn browser_dom_event_stream_requested() -> bool {
    if browser_surface_overlay_active() {
        return true;
    }
    static REQUESTED: OnceLock<bool> = OnceLock::new();
    *REQUESTED.get_or_init(|| {
        web_sys::window()
            .and_then(|window| window.location().search().ok())
            .and_then(|search| web_sys::UrlSearchParams::new_with_str(&search).ok())
            .is_some_and(|params| {
                params.has("headless")
                    || params.get("rendered-model-smoke").as_deref() == Some("1")
                    || params.has("rendered-surface-smoke")
            })
    })
}

fn dispatch_browser_event<T: serde::Serialize>(name: &str, value: &T) {
    // Interactive progress goes directly to ImageRunnerEvent/Bevy unless the shared canvas is
    // deliberately suspended. During that interval, forward only the request's bounded event
    // window to the web-only safety overlay. Headless and rendered qualification routes retain
    // their complete automation event contracts.
    if !browser_dom_event_stream_requested() {
        return;
    }
    let result = (|| {
        let json = serde_json::to_string(value).map_err(|error| error.to_string())?;
        let detail = js_sys::JSON::parse(&json).map_err(|error| format!("{error:?}"))?;
        let init = web_sys::CustomEventInit::new();
        init.set_detail(&detail);
        let event = web_sys::CustomEvent::new_with_event_init_dict(name, &init)
            .map_err(|error| format!("{error:?}"))?;
        let window = web_sys::window().ok_or_else(|| "Window is unavailable".to_owned())?;
        window
            .dispatch_event(event.as_ref())
            .map_err(|error| format!("{error:?}"))?;
        Ok::<(), String>(())
    })();
    if let Err(error) = result {
        web_sys::console::warn_1(
            &format!("failed to dispatch browser event {name}: {error}").into(),
        );
    }
}

struct BrowserBuildInputs {
    identity: BooguReleaseIdentity,
    base_url: burn_image::RemoteBaseUrl,
    settings: crate::BooguAdapterSettings,
    device: burn_wgpu::WgpuDevice,
    allocation_device: Option<crate::backend::SharedWgpuAllocationDevice>,
    applied_buffer_limits: BrowserAppliedBufferLimits,
}

#[derive(Clone, Copy, Debug)]
struct BrowserAppliedBufferLimits {
    max_storage_buffer_binding_size: u64,
    max_buffer_size: u64,
}

#[derive(Clone, Default)]
struct BrowserVerifiedArtifactLedger(Arc<Mutex<BTreeMap<String, usize>>>);

impl BrowserVerifiedArtifactLedger {
    fn observer(&self) -> Arc<dyn Fn(BrowserArtifactEvent) + Send + Sync> {
        let ledger = self.clone();
        Arc::new(move |event| {
            if let BrowserArtifactEvent::Verified(path) = event {
                let mut verified = ledger
                    .0
                    .lock()
                    .expect("browser verified-artifact ledger poisoned");
                *verified.entry(path.to_string()).or_default() += 1;
            }
        })
    }

    fn counts(&self) -> BTreeMap<String, usize> {
        self.0
            .lock()
            .expect("browser verified-artifact ledger poisoned")
            .clone()
    }

    fn report(
        &self,
        expected_weights: &BTreeMap<String, u64>,
        verified_runtime_metadata: &BTreeSet<String>,
    ) -> BrowserParityArtifactVerificationReport {
        let counts = self.counts();
        let actual = counts.keys().cloned().collect::<BTreeSet<_>>();
        let expected = expected_weights.keys().cloned().collect::<BTreeSet<_>>();
        let missing_weight_objects = expected.difference(&actual).cloned().collect::<Vec<_>>();
        let unexpected_verified_objects = actual.difference(&expected).cloned().collect::<Vec<_>>();
        let verified_unique_weight_bytes = actual
            .intersection(&expected)
            .map(|path| expected_weights[path])
            .sum();
        let expected_unique_weight_bytes = expected_weights.values().copied().sum();
        let expected_runtime_metadata = verified_runtime_metadata.clone();
        let passed = missing_weight_objects.is_empty()
            && unexpected_verified_objects.is_empty()
            && actual.len() == expected.len()
            && verified_unique_weight_bytes == expected_unique_weight_bytes
            && verified_runtime_metadata == &expected_runtime_metadata;
        BrowserParityArtifactVerificationReport {
            scope: "sealed canonical manifest plus every executed weight object and the four runtime-consumed metadata objects; compact manifest-bound files not consumed by inference are not redundantly fetched by this gate"
                .into(),
            sealed_manifest_validated: true,
            canonical_release_digest_validated: true,
            verified_runtime_metadata_objects: verified_runtime_metadata.len(),
            expected_runtime_metadata_objects: expected_runtime_metadata.len(),
            verified_unique_weight_objects: actual.intersection(&expected).count(),
            expected_unique_weight_objects: expected.len(),
            verified_unique_weight_bytes,
            expected_unique_weight_bytes,
            missing_weight_objects,
            unexpected_verified_objects,
            passed,
        }
    }
}

fn browser_required_runtime_metadata_paths() -> BTreeSet<String> {
    [
        "metadata/source/mllm/config.json",
        "metadata/source/vae/config.json",
        "metadata/source/mllm/tokenizer.json",
        "metadata/source/mllm/preprocessor_config.json",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

/// Real event-loop-safe barrier for the raw CubeCL WebGPU backend.
///
/// Burn's synchronous `Backend::sync` blocks this future and therefore panics on single-threaded
/// Wasm. Holding the runtime client lets the async stage traits await the same server barrier
/// without blocking the browser event loop.
#[derive(Clone)]
struct BrowserAsyncSynchronizer {
    client: cubecl::client::ComputeClient<burn_wgpu::WgpuRuntime>,
}

impl BrowserAsyncSynchronizer {
    fn new(device: &burn_wgpu::WgpuDevice) -> Self {
        Self {
            client: <burn_wgpu::WgpuRuntime as cubecl::Runtime>::client(device),
        }
    }

    async fn synchronize(&self, stage: &str) -> Result<(), BooguError> {
        self.client.sync().await.map_err(|error| {
            BooguError::Artifact(format!(
                "nonblocking WebGPU synchronization after {stage} failed: {error}"
            ))
        })
    }

    /// Submit queued upload work without waiting for GPU completion.
    ///
    /// Resident preload uses this after each semantic stage so browser-native staging buffers do
    /// not accumulate, while a single component-end asynchronous barrier replaces dozens of
    /// serial queue waits.
    fn submit(&self, stage: &str) -> Result<(), BooguError> {
        self.client.flush().map_err(|error| {
            BooguError::Artifact(format!(
                "WebGPU upload submission after {stage} failed: {error}"
            ))
        })
    }
}

async fn release_browser_phase_allocator(
    variant: BooguVariant,
    device: &burn_wgpu::WgpuDevice,
    phase: &str,
) -> Result<(), RuntimeError> {
    let synchronizer = BrowserAsyncSynchronizer::new(device);
    synchronizer
        .synchronize(&format!("{phase} before allocator cleanup"))
        .await
        .map_err(|error| map_boogu(variant, error))?;
    <BrowserBackend as Backend>::memory_cleanup(device);
    synchronizer
        .synchronize(&format!("{phase} after allocator cleanup"))
        .await
        .map_err(|error| map_boogu(variant, error))
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct BrowserParityTensorMetric {
    pub name: String,
    pub oracle: String,
    pub shape: Vec<usize>,
    pub actual_dtype: String,
    #[serde(flatten)]
    pub comparison: FloatMetrics,
}

#[derive(Clone, Default)]
struct BrowserParityOracleLedger(Arc<Mutex<BTreeSet<String>>>);

impl BrowserParityOracleLedger {
    fn record(&self, name: &str) {
        self.0
            .lock()
            .expect("browser parity oracle ledger poisoned")
            .insert(name.to_owned());
    }

    fn contains(&self, name: &str) -> bool {
        self.0
            .lock()
            .expect("browser parity oracle ledger poisoned")
            .contains(name)
    }

    fn names(&self) -> BTreeSet<String> {
        self.0
            .lock()
            .expect("browser parity oracle ledger poisoned")
            .clone()
    }
}

enum PendingBrowserTensor {
    Rank2 {
        name: String,
        oracle: String,
        tensor: Tensor<BrowserBackend, 2>,
    },
    Rank3 {
        name: String,
        oracle: String,
        tensor: Tensor<BrowserBackend, 3>,
    },
    Rank4 {
        name: String,
        oracle: String,
        tensor: Tensor<BrowserBackend, 4>,
    },
}

/// Shared observer/source rendezvous for bounded asynchronous parity readback.
///
/// Model observers only enqueue device tensors. The matching source barrier first awaits WebGPU,
/// then this control reads back, range-fetches, authenticates, compares, and drops each pair before
/// the following verified model shard is requested.
#[derive(Clone)]
struct BrowserParityControl {
    fixture: BrowserParityFixture,
    ledger: BrowserParityOracleLedger,
    pending: Arc<Mutex<Vec<PendingBrowserTensor>>>,
    metrics: Arc<Mutex<Vec<BrowserParityTensorMetric>>>,
}

impl BrowserParityControl {
    fn new(fixture: BrowserParityFixture, ledger: BrowserParityOracleLedger) -> Self {
        Self {
            fixture,
            ledger,
            pending: Arc::new(Mutex::new(Vec::new())),
            metrics: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn rank2(&self, name: String, oracle: String, tensor: Tensor<BrowserBackend, 2>) {
        self.pending
            .lock()
            .expect("browser parity pending tensor queue poisoned")
            .push(PendingBrowserTensor::Rank2 {
                name,
                oracle,
                tensor,
            });
    }

    fn rank3(&self, name: String, oracle: String, tensor: Tensor<BrowserBackend, 3>) {
        self.pending
            .lock()
            .expect("browser parity pending tensor queue poisoned")
            .push(PendingBrowserTensor::Rank3 {
                name,
                oracle,
                tensor,
            });
    }

    fn rank4(&self, name: String, oracle: String, tensor: Tensor<BrowserBackend, 4>) {
        self.pending
            .lock()
            .expect("browser parity pending tensor queue poisoned")
            .push(PendingBrowserTensor::Rank4 {
                name,
                oracle,
                tensor,
            });
    }

    async fn compare_pending(&self) -> Result<(), BooguError> {
        let pending = std::mem::take(
            &mut *self
                .pending
                .lock()
                .expect("browser parity pending tensor queue poisoned"),
        );
        for tensor in pending {
            let metric = match tensor {
                PendingBrowserTensor::Rank2 {
                    name,
                    oracle,
                    tensor,
                } => {
                    compare_browser_tensor(&self.fixture, &self.ledger, name, oracle, tensor)
                        .await?
                }
                PendingBrowserTensor::Rank3 {
                    name,
                    oracle,
                    tensor,
                } => {
                    compare_browser_tensor(&self.fixture, &self.ledger, name, oracle, tensor)
                        .await?
                }
                PendingBrowserTensor::Rank4 {
                    name,
                    oracle,
                    tensor,
                } => {
                    compare_browser_tensor(&self.fixture, &self.ledger, name, oracle, tensor)
                        .await?
                }
            };
            self.metrics
                .lock()
                .expect("browser parity metric queue poisoned")
                .push(metric);
        }
        Ok(())
    }

    fn metrics(&self) -> Vec<BrowserParityTensorMetric> {
        self.metrics
            .lock()
            .expect("browser parity metric queue poisoned")
            .clone()
    }

    fn pending_count(&self) -> usize {
        self.pending
            .lock()
            .expect("browser parity pending tensor queue poisoned")
            .len()
    }
}

async fn compare_browser_tensor<const D: usize>(
    fixture: &BrowserParityFixture,
    ledger: &BrowserParityOracleLedger,
    name: String,
    oracle: String,
    tensor: Tensor<BrowserBackend, D>,
) -> Result<BrowserParityTensorMetric, BooguError> {
    let elements = tensor.dims().iter().copied().product::<usize>();
    browser_parity_readback_milestone("start", &name, elements, 0, 0);
    let (shape, actual_dtype, actual) = read_browser_tensor_f32(tensor).await?;
    browser_parity_readback_milestone("complete", &name, elements, 0, 0);
    compare_browser_f32_values(
        fixture,
        ledger,
        name,
        oracle,
        &shape,
        &actual_dtype,
        &actual,
    )
    .await
}

async fn read_browser_tensor_f32<const D: usize>(
    tensor: Tensor<BrowserBackend, D>,
) -> Result<(Vec<usize>, String, Vec<f32>), BooguError> {
    let shape = tensor.dims().to_vec();
    let actual_dtype = tensor.dtype().name().to_owned();
    let elements = shape.iter().try_fold(1_usize, |total, dimension| {
        total.checked_mul(*dimension).ok_or_else(|| {
            BooguError::Artifact("WebGPU parity readback element count overflowed".into())
        })
    })?;
    let ranges = browser_parity_readback_ranges(elements);
    if ranges.len() == 1 {
        let data = tensor
            .into_data_async()
            .await
            .map_err(|error| {
                BooguError::Artifact(format!("WebGPU parity readback failed: {error}"))
            })?
            .convert_dtype(DType::F32);
        let values = data
            .as_slice::<f32>()
            .map_err(|error| BooguError::Artifact(error.to_string()))?
            .to_vec();
        return Ok((shape, actual_dtype, values));
    }

    let flattened: Tensor<BrowserBackend, 1> = tensor.flatten(0, D.saturating_sub(1));
    let mut values = Vec::with_capacity(elements);
    let chunk_count = ranges.len();
    for (chunk_index, (start, end)) in ranges.into_iter().enumerate() {
        browser_parity_readback_milestone(
            "chunk-start",
            "bounded-map",
            elements,
            chunk_index,
            chunk_count,
        );
        let data = flattened
            .clone()
            .slice(start..end)
            .into_data_async()
            .await
            .map_err(|error| {
                BooguError::Artifact(format!(
                    "WebGPU parity readback chunk {}/{chunk_count} failed: {error}",
                    chunk_index + 1
                ))
            })?
            .convert_dtype(DType::F32);
        values.extend_from_slice(
            data.as_slice::<f32>()
                .map_err(|error| BooguError::Artifact(error.to_string()))?,
        );
        browser_parity_readback_milestone(
            "chunk-complete",
            "bounded-map",
            elements,
            chunk_index,
            chunk_count,
        );
    }
    if values.len() != elements {
        return Err(BooguError::Artifact(format!(
            "WebGPU parity bounded readback produced {} F32 values, expected {elements}",
            values.len()
        )));
    }
    Ok((shape, actual_dtype, values))
}

fn browser_parity_readback_ranges(elements: usize) -> Vec<(usize, usize)> {
    let chunk_elements =
        (BROWSER_PARITY_MAX_READBACK_CHUNK_BYTES / BROWSER_PARITY_F32_ELEMENT_BYTES).max(1);
    let mut ranges = Vec::with_capacity(elements.div_ceil(chunk_elements));
    let mut start = 0;
    while start < elements {
        let end = start.saturating_add(chunk_elements).min(elements);
        ranges.push((start, end));
        start = end;
    }
    ranges
}

#[allow(clippy::too_many_arguments)]
async fn compare_browser_f32_values(
    fixture: &BrowserParityFixture,
    ledger: &BrowserParityOracleLedger,
    name: String,
    oracle: String,
    shape: &[usize],
    actual_dtype: &str,
    actual: &[f32],
) -> Result<BrowserParityTensorMetric, BooguError> {
    let (expected_shape, expected) = fixture
        .f32(&oracle)
        .await
        .map_err(|error| BooguError::Artifact(error.to_string()))?;
    ledger.record(&oracle);
    if shape != expected_shape {
        return Err(BooguError::InvalidShape(format!(
            "oracle {oracle} has shape {expected_shape:?}, WebGPU produced {shape:?}"
        )));
    }
    let comparison = compare_float(actual, &expected)
        .map_err(|error| BooguError::Artifact(error.to_string()))?;
    Ok(BrowserParityTensorMetric {
        name,
        oracle,
        shape: shape.to_vec(),
        actual_dtype: actual_dtype.to_owned(),
        comparison,
    })
}

/// Replaces only the verified source's blocking backend barrier; all loads still delegate to the
/// same digest-verifying bounded source.
struct BrowserAsyncStageSource<S> {
    inner: S,
    synchronizer: BrowserAsyncSynchronizer,
    pending_stage: Option<String>,
    qwen_resident_preload: bool,
    deferred_stages_since_yield: usize,
    parity: Option<BrowserParityControl>,
    denoiser_query_chunk_size: Option<usize>,
}

impl<S> BrowserAsyncStageSource<S> {
    fn new(inner: S, synchronizer: BrowserAsyncSynchronizer) -> Self {
        Self {
            inner,
            synchronizer,
            pending_stage: None,
            qwen_resident_preload: false,
            deferred_stages_since_yield: 0,
            parity: None,
            denoiser_query_chunk_size: None,
        }
    }

    fn set_parity_control(&mut self, parity: BrowserParityControl) {
        self.parity = Some(parity);
    }

    fn set_denoiser_query_chunk_size(&mut self, query_chunk_size: usize) {
        self.denoiser_query_chunk_size = Some(query_chunk_size);
    }

    fn set_qwen_resident_preload(&mut self, active: bool) {
        self.qwen_resident_preload = active;
        if !active {
            self.pending_stage = None;
        }
    }

    fn inner(&self) -> &S {
        &self.inner
    }

    fn inner_mut(&mut self) -> &mut S {
        &mut self.inner
    }

    async fn synchronize_with_parity(&self, stage: &str) -> Result<(), BooguError> {
        self.synchronizer.synchronize(stage).await?;
        if let Some(parity) = &self.parity {
            parity.compare_pending().await?;
        }
        Ok(())
    }

    async fn cooperative_browser_yield(&mut self) -> Result<(), BooguError> {
        const STAGES_PER_EVENT_LOOP_YIELD: usize = 2;

        self.deferred_stages_since_yield = self.deferred_stages_since_yield.saturating_add(1);
        if !self
            .deferred_stages_since_yield
            .is_multiple_of(STAGES_PER_EVENT_LOOP_YIELD)
        {
            return Ok(());
        }

        use wasm_bindgen::JsValue;
        use wasm_bindgen_futures::JsFuture;

        let window = web_sys::window().ok_or_else(|| {
            BooguError::Artifact("browser event-loop yield requires Window".into())
        })?;
        let promise = js_sys::Promise::new(&mut |resolve, reject| {
            if let Err(error) =
                window.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 0)
            {
                let _ = reject.call1(&JsValue::UNDEFINED, &error);
            }
        });
        JsFuture::from(promise).await.map(|_| ()).map_err(|error| {
            BooguError::Artifact(format!("browser event-loop yield failed: {error:?}"))
        })
    }
}

impl<S> AsyncQwen3VlStageSource<BrowserBackend> for BrowserAsyncStageSource<S>
where
    S: AsyncQwen3VlStageSource<BrowserBackend, Error = BooguError>,
{
    type Error = BooguError;

    async fn load_embedding_rows(
        &mut self,
        spec: &RowChunkSpec,
    ) -> Result<EmbeddingRowChunk<BrowserBackend>, Self::Error> {
        self.inner.load_embedding_rows(spec).await
    }

    async fn load_host_routed_f16_embedding_f32(
        &mut self,
        input_ids: &[Vec<i64>],
        device: &burn_wgpu::WgpuDevice,
    ) -> Result<Option<HostRoutedEmbedding<BrowserBackend>>, Self::Error> {
        self.inner
            .load_host_routed_f16_embedding_f32(input_ids, device)
            .await
    }

    async fn load_vision_prelude(
        &mut self,
    ) -> Result<Qwen3VlVisionPrelude<BrowserBackend>, Self::Error> {
        self.inner.load_vision_prelude().await
    }

    async fn load_vision_block(
        &mut self,
        index: usize,
    ) -> Result<Qwen3VlVisionBlock<BrowserBackend>, Self::Error> {
        self.inner.load_vision_block(index).await
    }

    async fn load_vision_deepstack_merger(
        &mut self,
        index: usize,
    ) -> Result<Qwen3VlVisionPatchMerger<BrowserBackend>, Self::Error> {
        self.inner.load_vision_deepstack_merger(index).await
    }

    async fn load_vision_final_merger(
        &mut self,
    ) -> Result<Qwen3VlVisionPatchMerger<BrowserBackend>, Self::Error> {
        self.inner.load_vision_final_merger().await
    }

    async fn load_text_block(
        &mut self,
        index: usize,
    ) -> Result<Qwen3VlDecoderLayer<BrowserBackend>, Self::Error> {
        let stage = if self.qwen_resident_preload {
            format!("qwen-resident-preload-text-block-{index:02}")
        } else {
            format!("qwen-text-block-{index:02}")
        };
        browser_stage_milestone(&format!("{stage}-source-load-apply-start"));
        self.pending_stage = Some(stage.clone());
        let layer = self.inner.load_text_block(index).await?;
        browser_stage_milestone(&format!("{stage}-source-load-apply-complete"));
        Ok(layer)
    }

    async fn load_text_final_norm(&mut self) -> Result<RmsNorm<BrowserBackend>, Self::Error> {
        let stage = if self.qwen_resident_preload {
            "qwen-resident-preload-text-final-norm"
        } else {
            "qwen-text-final-norm"
        };
        browser_stage_milestone(&format!("{stage}-source-load-apply-start"));
        self.pending_stage = Some(stage.into());
        let norm = self.inner.load_text_final_norm().await?;
        browser_stage_milestone(&format!("{stage}-source-load-apply-complete"));
        Ok(norm)
    }

    async fn cooperative_yield(&mut self) -> Result<(), Self::Error> {
        self.cooperative_browser_yield().await
    }

    async fn synchronize(&mut self) -> Result<(), Self::Error> {
        let stage = self.pending_stage.as_deref().unwrap_or("qwen-stage");
        let boundary = if self.qwen_resident_preload {
            "upload-sync"
        } else {
            "post-forward-sync"
        };
        browser_stage_milestone(&format!("{stage}-{boundary}-start"));
        let result = self.synchronize_with_parity(stage).await;
        if result.is_ok() {
            browser_stage_milestone(&format!("{stage}-{boundary}-complete"));
            self.pending_stage = None;
            self.deferred_stages_since_yield = 0;
        }
        result
    }
}

/// Marks the exact point after a Qwen stage forward has been queued and before its source barrier.
/// Together with the source wrapper milestones this distinguishes load/apply, forward submission,
/// and asynchronous WebGPU synchronization without forcing a readback.
struct PendingBrowserQwenInstructionDiagnostic {
    name: String,
    tensor: Tensor<BrowserBackend, 3>,
}

/// Rendered-smoke-only retention of the compact text conditioning activations. Packed production
/// inference leaves this control absent, so its ordinary path never retains or reads stage output.
#[derive(Clone, Default)]
struct BrowserQwenInstructionDiagnosticControl {
    pending: Arc<Mutex<Vec<PendingBrowserQwenInstructionDiagnostic>>>,
    immediate_post_sync_block0: Arc<Mutex<Option<BrowserPackedF16QwenBlock0PostSyncDiagnostic>>>,
    block0_boundaries: Arc<Mutex<Vec<BrowserPackedF16QwenBlock0BoundaryDiagnostic>>>,
}

struct BrowserQwenBlock0BoundaryRecordOutcome {
    report: Option<BrowserPackedF16QwenBlock0ExecutionDiagnostics>,
    failure_message: Option<String>,
}

impl BrowserQwenInstructionDiagnosticControl {
    fn capture(&self, name: String, tensor: Tensor<BrowserBackend, 3>) {
        self.pending
            .lock()
            .expect("browser Qwen diagnostic pending queue poisoned")
            .push(PendingBrowserQwenInstructionDiagnostic { name, tensor });
    }

    async fn read_all(
        &self,
        variant: BooguVariant,
    ) -> Result<Vec<BrowserPackedF16TensorInputDiagnostic>, RuntimeError> {
        let pending = std::mem::take(
            &mut *self
                .pending
                .lock()
                .expect("browser Qwen diagnostic pending queue poisoned"),
        );
        let mut diagnostics = Vec::with_capacity(pending.len());
        for pending in pending {
            diagnostics.push(
                read_packed_f16_tensor_input_diagnostic(variant, &pending.name, &pending.tensor)
                    .await?,
            );
        }
        Ok(diagnostics)
    }

    fn pending_count(&self) -> usize {
        self.pending
            .lock()
            .expect("browser Qwen diagnostic pending queue poisoned")
            .len()
    }

    fn record_immediate_post_sync_block0(
        &self,
        diagnostic: BrowserPackedF16QwenBlock0PostSyncDiagnostic,
    ) -> burn_qwen3_vl::Result<()> {
        let mut slot = self
            .immediate_post_sync_block0
            .lock()
            .expect("browser Qwen immediate block-0 diagnostic slot poisoned");
        if slot.replace(diagnostic).is_some() {
            return Err(burn_qwen3_vl::Qwen3VlError::InvalidInput(
                "browser Qwen immediate post-sync block-0 diagnostic was recorded twice".into(),
            ));
        }
        Ok(())
    }

    fn take_immediate_post_sync_block0(
        &self,
    ) -> Option<BrowserPackedF16QwenBlock0PostSyncDiagnostic> {
        self.immediate_post_sync_block0
            .lock()
            .expect("browser Qwen immediate block-0 diagnostic slot poisoned")
            .take()
    }

    fn record_block0_boundary(
        &self,
        allocation_policy: Qwen3VlTextLayerAllocationPolicy,
        synchronization_policy: Qwen3VlTextBlockLoadSynchronizationPolicy,
        submission_policy: &'static str,
        boundary: Qwen3VlTextLayerDiagnosticBoundary,
        tensor: BrowserPackedF16TensorInputDiagnostic,
    ) -> burn_qwen3_vl::Result<BrowserQwenBlock0BoundaryRecordOutcome> {
        let mut boundaries = self
            .block0_boundaries
            .lock()
            .expect("browser Qwen block-0 boundary diagnostic queue poisoned");
        let sequence_index = boundaries.len();
        let Some(expected_boundary) = BROWSER_PACKED_F16_QWEN_BLOCK0_BOUNDARIES
            .get(sequence_index)
            .copied()
        else {
            return Err(burn_qwen3_vl::Qwen3VlError::InvalidInput(
                "browser Qwen block-0 boundary diagnostic exceeded its exact sequence".into(),
            ));
        };
        if boundary != expected_boundary {
            return Err(burn_qwen3_vl::Qwen3VlError::InvalidInput(format!(
                "browser Qwen block-0 boundary {} arrived at index {sequence_index}, expected {}",
                boundary.label(),
                expected_boundary.label(),
            )));
        }
        let all_finite = tensor.all_finite;
        let not_all_zero = !packed_f16_tensor_diagnostic_is_all_zero(&tensor);
        boundaries.push(BrowserPackedF16QwenBlock0BoundaryDiagnostic {
            sequence_index,
            boundary: boundary.label().into(),
            tensor_kind: boundary.tensor_kind().into(),
            tensor,
            all_finite,
            not_all_zero,
        });

        let layer_input = boundaries
            .first()
            .filter(|diagnostic| {
                diagnostic.boundary == Qwen3VlTextLayerDiagnosticBoundary::LayerInput.label()
            })
            .map(|diagnostic| &diagnostic.tensor);
        let identity_canary = boundaries
            .iter()
            .find(|diagnostic| {
                diagnostic.boundary == Qwen3VlTextLayerDiagnosticBoundary::IdentityAddCanary.label()
            })
            .map(|diagnostic| &diagnostic.tensor);
        let identity_add_canary_matches_input = identity_canary.map(|canary| {
            layer_input.is_some_and(|input| {
                canary.shape == input.shape
                    && canary.dtype == input.dtype
                    && canary.element_count == input.element_count
                    && canary.sha256 == input.sha256
            })
        });
        let failure_reason = if !all_finite {
            Some("non-finite")
        } else if !not_all_zero {
            Some("all-zero")
        } else if boundary == Qwen3VlTextLayerDiagnosticBoundary::IdentityAddCanary
            && identity_add_canary_matches_input != Some(true)
        {
            Some("identity-add-canary-mismatch")
        } else {
            None
        };
        let is_final = boundary == Qwen3VlTextLayerDiagnosticBoundary::FinalResidualOutput;
        let should_dispatch = failure_reason.is_some() || is_final;
        let report = should_dispatch.then(|| {
            let boundary_names_exact = boundaries.iter().enumerate().all(|(index, diagnostic)| {
                BROWSER_PACKED_F16_QWEN_BLOCK0_BOUNDARIES
                    .get(index)
                    .is_some_and(|expected| diagnostic.boundary == expected.label())
            });
            let complete = failure_reason.is_none()
                && is_final
                && boundaries.len() == BROWSER_PACKED_F16_QWEN_BLOCK0_BOUNDARIES.len()
                && boundary_names_exact
                && identity_add_canary_matches_input == Some(true);
            BrowserPackedF16QwenBlock0ExecutionDiagnostics {
                scope: BROWSER_PACKED_F16_QWEN_BLOCK0_EXECUTION_SCOPE.into(),
                block0_execution_mode: BROWSER_QWEN_BLOCK0_SERIALIZED_DIAGNOSTIC_MODE.into(),
                text_layer_allocation_policy: allocation_policy.label().into(),
                text_block_load_synchronization_policy: synchronization_policy.label().into(),
                qwen_text_layer_submission_policy: submission_policy.into(),
                expected_boundary_count: BROWSER_PACKED_F16_QWEN_BLOCK0_BOUNDARIES.len(),
                captured_boundary_count: boundaries.len(),
                boundaries: boundaries.clone(),
                boundary_names_exact,
                all_captured_tensors_finite: boundaries
                    .iter()
                    .all(|diagnostic| diagnostic.all_finite),
                no_captured_tensor_all_zero: boundaries
                    .iter()
                    .all(|diagnostic| diagnostic.not_all_zero),
                identity_add_canary_matches_input,
                complete,
                first_failure_boundary: failure_reason.map(|_| boundary.label().into()),
                failure_reason: failure_reason.map(str::to_owned),
            }
        });
        Ok(BrowserQwenBlock0BoundaryRecordOutcome {
            report,
            failure_message: failure_reason.map(|reason| {
                format!(
                    "browser Qwen block 0 boundary {} failed immediate readback: {reason}",
                    boundary.label()
                )
            }),
        })
    }
}

struct BrowserQwenStageObserver {
    parity: Option<BrowserParityControl>,
    instruction_diagnostics: Option<BrowserQwenInstructionDiagnosticControl>,
    instruction_diagnostic_variant: Option<BooguVariant>,
    instruction_diagnostic_run_id: Option<RunId>,
    instruction_diagnostic_text_layer_allocation_policy: Option<Qwen3VlTextLayerAllocationPolicy>,
    instruction_diagnostic_text_block_load_synchronization_policy:
        Option<Qwen3VlTextBlockLoadSynchronizationPolicy>,
    instruction_diagnostic_text_layer_submission_policy: Option<&'static str>,
    instruction_diagnostic_block0_execution_mode: Option<&'static str>,
    multimodal: bool,
    deepstack_count: usize,
}

impl BrowserQwenStageObserver {
    fn milestones_only() -> Self {
        Self {
            parity: None,
            instruction_diagnostics: None,
            instruction_diagnostic_variant: None,
            instruction_diagnostic_run_id: None,
            instruction_diagnostic_text_layer_allocation_policy: None,
            instruction_diagnostic_text_block_load_synchronization_policy: None,
            instruction_diagnostic_text_layer_submission_policy: None,
            instruction_diagnostic_block0_execution_mode: None,
            multimodal: false,
            deepstack_count: 0,
        }
    }

    fn parity(control: BrowserParityControl, multimodal: bool, deepstack_count: usize) -> Self {
        Self {
            parity: Some(control),
            instruction_diagnostics: None,
            instruction_diagnostic_variant: None,
            instruction_diagnostic_run_id: None,
            instruction_diagnostic_text_layer_allocation_policy: None,
            instruction_diagnostic_text_block_load_synchronization_policy: None,
            instruction_diagnostic_text_layer_submission_policy: None,
            instruction_diagnostic_block0_execution_mode: None,
            multimodal,
            deepstack_count,
        }
    }

    fn with_instruction_diagnostics(
        mut self,
        control: BrowserQwenInstructionDiagnosticControl,
        variant: BooguVariant,
        run_id: RunId,
        text_layer_allocation_policy: Qwen3VlTextLayerAllocationPolicy,
        text_block_load_synchronization_policy: Qwen3VlTextBlockLoadSynchronizationPolicy,
        text_layer_submission_policy: &'static str,
    ) -> Self {
        self.instruction_diagnostics = Some(control);
        self.instruction_diagnostic_variant = Some(variant);
        self.instruction_diagnostic_run_id = Some(run_id);
        self.instruction_diagnostic_text_layer_allocation_policy =
            Some(text_layer_allocation_policy);
        self.instruction_diagnostic_text_block_load_synchronization_policy =
            Some(text_block_load_synchronization_policy);
        self.instruction_diagnostic_text_layer_submission_policy =
            Some(text_layer_submission_policy);
        self.instruction_diagnostic_block0_execution_mode =
            Some(browser_qwen_block0_execution_mode());
        self
    }

    fn oracle(&self, stage: &Qwen3VlStage) -> Option<String> {
        Some(match stage {
            Qwen3VlStage::EmbeddingRows { .. } => "qwen.text.token_embeddings".into(),
            Qwen3VlStage::VisionPrelude => "qwen.vision.prelude".into(),
            Qwen3VlStage::VisionBlock { index } => format!("qwen.vision.block.{index}"),
            Qwen3VlStage::VisionDeepstackMerger { index, .. } => {
                format!("qwen.vision.deepstack_merger.{index}")
            }
            Qwen3VlStage::VisionFinalMerger => "qwen.vision.final_merger".into(),
            Qwen3VlStage::TextBlock { index }
                if self.multimodal && *index < self.deepstack_count =>
            {
                format!("qwen.text.layer.{index}.post_deepstack")
            }
            Qwen3VlStage::TextBlock { index } => format!("qwen.text.layer.{index}"),
            Qwen3VlStage::TextFinalNorm => "qwen.text.final_norm".into(),
            Qwen3VlStage::LmHeadRows { .. } => return None,
        })
    }

    fn stage_name(stage: &Qwen3VlStage) -> String {
        match stage {
            Qwen3VlStage::EmbeddingRows { chunk } => format!("embedding_rows.{chunk}"),
            Qwen3VlStage::VisionPrelude => "vision.prelude".into(),
            Qwen3VlStage::VisionBlock { index } => format!("vision.block.{index}"),
            Qwen3VlStage::VisionDeepstackMerger { index, .. } => {
                format!("vision.deepstack_merger.{index}")
            }
            Qwen3VlStage::VisionFinalMerger => "vision.final_merger".into(),
            Qwen3VlStage::TextBlock { index } => format!("text.layer.{index}"),
            Qwen3VlStage::TextFinalNorm => "text.final_norm".into(),
            Qwen3VlStage::LmHeadRows { chunk } => format!("lm_head.rows.{chunk}"),
        }
    }

    fn instruction_diagnostic_name(stage: &Qwen3VlStage) -> Option<String> {
        match stage {
            Qwen3VlStage::EmbeddingRows { .. } => Some("qwen_embedding_output".into()),
            Qwen3VlStage::TextBlock { index } => Some(format!("qwen_text_block_{index:02}_output")),
            Qwen3VlStage::TextFinalNorm => Some("qwen_final_norm_output".into()),
            _ => None,
        }
    }

    async fn record_block0_boundary<const D: usize>(
        &mut self,
        index: usize,
        boundary: Qwen3VlTextLayerDiagnosticBoundary,
        tensor: Tensor<BrowserBackend, D>,
    ) -> burn_qwen3_vl::Result<()> {
        if index != 0 {
            return Err(burn_qwen3_vl::Qwen3VlError::InvalidInput(format!(
                "browser serialized Qwen diagnostic received text layer {index}, expected 0"
            )));
        }
        let (
            Some(control),
            Some(variant),
            Some(run_id),
            Some(allocation_policy),
            Some(synchronization_policy),
            Some(submission_policy),
        ) = (
            self.instruction_diagnostics.as_ref(),
            self.instruction_diagnostic_variant,
            self.instruction_diagnostic_run_id,
            self.instruction_diagnostic_text_layer_allocation_policy,
            self.instruction_diagnostic_text_block_load_synchronization_policy,
            self.instruction_diagnostic_text_layer_submission_policy,
        )
        else {
            return Err(burn_qwen3_vl::Qwen3VlError::InvalidInput(
                "browser serialized Qwen diagnostic is missing its rendered-smoke context".into(),
            ));
        };
        let name = format!(
            "qwen_text_block_00_{}_immediate_post_sync",
            boundary.label()
        );
        let tensor = read_packed_f16_tensor_input_diagnostic(variant, &name, &tensor)
            .await
            .map_err(|error| {
                burn_qwen3_vl::Qwen3VlError::InvalidInput(format!(
                    "browser Qwen block-0 boundary {} readback failed: {error:?}",
                    boundary.label()
                ))
            })?;
        let outcome = control.record_block0_boundary(
            allocation_policy,
            synchronization_policy,
            submission_policy,
            boundary,
            tensor,
        )?;
        if let Some(report) = outcome.report {
            dispatch_browser_event(
                BROWSER_RUNTIME_EVENT_NAME,
                &browser_packed_f16_qwen_block0_execution_diagnostics_event(run_id, report),
            );
        }
        if let Some(message) = outcome.failure_message {
            return Err(burn_qwen3_vl::Qwen3VlError::InvalidInput(message));
        }
        Ok(())
    }
}

impl Qwen3VlStageObserver<BrowserBackend> for BrowserQwenStageObserver {
    fn text_layer_boundary_diagnostics_requested(&self, index: usize) -> bool {
        index == 0
            && self.instruction_diagnostics.is_some()
            && self.instruction_diagnostic_variant == Some(BooguVariant::Image01Turbo)
            && self.instruction_diagnostic_text_layer_allocation_policy
                == Some(Qwen3VlTextLayerAllocationPolicy::ExactSizePersistent)
            && self.instruction_diagnostic_text_block_load_synchronization_policy
                == Some(Qwen3VlTextBlockLoadSynchronizationPolicy::PreForwardAndPostForward)
            && self.instruction_diagnostic_block0_execution_mode
                == Some(BROWSER_QWEN_BLOCK0_SERIALIZED_DIAGNOSTIC_MODE)
    }

    async fn text_layer_parameter_after_synchronize(
        &mut self,
        index: usize,
        boundary: Qwen3VlTextLayerDiagnosticBoundary,
        parameter: Tensor<BrowserBackend, 1>,
    ) -> burn_qwen3_vl::Result<()> {
        if boundary.tensor_kind() != "parameter-sentinel" {
            return Err(burn_qwen3_vl::Qwen3VlError::InvalidInput(format!(
                "browser Qwen parameter diagnostic received activation boundary {}",
                boundary.label()
            )));
        }
        self.record_block0_boundary(index, boundary, parameter)
            .await
    }

    async fn text_layer_activation_after_synchronize(
        &mut self,
        index: usize,
        boundary: Qwen3VlTextLayerDiagnosticBoundary,
        activation: Tensor<BrowserBackend, 3>,
    ) -> burn_qwen3_vl::Result<()> {
        if boundary.tensor_kind() != "activation" {
            return Err(burn_qwen3_vl::Qwen3VlError::InvalidInput(format!(
                "browser Qwen activation diagnostic received parameter boundary {}",
                boundary.label()
            )));
        }
        self.record_block0_boundary(index, boundary, activation)
            .await
    }

    fn rank2(
        &mut self,
        stage: &Qwen3VlStage,
        activation: Tensor<BrowserBackend, 2>,
    ) -> burn_qwen3_vl::Result<()> {
        if let (Some(control), Some(oracle)) = (&self.parity, self.oracle(stage)) {
            control.rank2(Self::stage_name(stage), oracle, activation);
        }
        Ok(())
    }

    fn rank3(
        &mut self,
        stage: &Qwen3VlStage,
        activation: Tensor<BrowserBackend, 3>,
    ) -> burn_qwen3_vl::Result<()> {
        if let Qwen3VlStage::TextBlock { index } = stage {
            browser_stage_milestone(&format!("qwen-text-block-{index:02}-forward-submitted"));
        }
        if let (Some(control), Some(name)) = (
            &self.instruction_diagnostics,
            Self::instruction_diagnostic_name(stage),
        ) {
            control.capture(name, activation.clone());
        }
        if let (Some(control), Some(oracle)) = (&self.parity, self.oracle(stage)) {
            control.rank3(Self::stage_name(stage), oracle, activation);
        }
        Ok(())
    }

    async fn rank3_after_synchronize(
        &mut self,
        stage: &Qwen3VlStage,
        activation: Tensor<BrowserBackend, 3>,
    ) -> burn_qwen3_vl::Result<()> {
        if !matches!(stage, Qwen3VlStage::TextBlock { index: 0 }) {
            return Ok(());
        }
        let (
            Some(control),
            Some(variant),
            Some(run_id),
            Some(allocation_policy),
            Some(synchronization_policy),
            Some(submission_policy),
            Some(block0_execution_mode),
        ) = (
            self.instruction_diagnostics.as_ref(),
            self.instruction_diagnostic_variant,
            self.instruction_diagnostic_run_id,
            self.instruction_diagnostic_text_layer_allocation_policy,
            self.instruction_diagnostic_text_block_load_synchronization_policy,
            self.instruction_diagnostic_text_layer_submission_policy,
            self.instruction_diagnostic_block0_execution_mode,
        )
        else {
            return Ok(());
        };
        let tensor = read_packed_f16_tensor_input_diagnostic(
            variant,
            "qwen_text_block_00_output_immediate_post_sync",
            &activation,
        )
        .await
        .map_err(|error| {
            burn_qwen3_vl::Qwen3VlError::InvalidInput(format!(
                "browser Qwen immediate post-sync block-0 readback failed: {error:?}"
            ))
        })?;
        let diagnostic = BrowserPackedF16QwenBlock0PostSyncDiagnostic {
            scope: BROWSER_PACKED_F16_QWEN_BLOCK0_POST_SYNC_SCOPE.into(),
            block0_execution_mode: block0_execution_mode.into(),
            text_layer_allocation_policy: allocation_policy.label().into(),
            text_block_load_synchronization_policy: synchronization_policy.label().into(),
            qwen_text_layer_submission_policy: submission_policy.into(),
            all_finite: tensor.all_finite,
            not_all_zero: !packed_f16_tensor_diagnostic_is_all_zero(&tensor),
            tensor,
        };
        let all_finite = diagnostic.all_finite;
        let not_all_zero = diagnostic.not_all_zero;
        control.record_immediate_post_sync_block0(diagnostic.clone())?;
        dispatch_browser_event(
            BROWSER_RUNTIME_EVENT_NAME,
            &browser_packed_f16_qwen_block0_post_sync_diagnostic_event(run_id, diagnostic),
        );
        if !all_finite {
            return Err(burn_qwen3_vl::Qwen3VlError::InvalidInput(
                "browser Qwen block 0 is non-finite immediately after its WebGPU barrier".into(),
            ));
        }
        if !not_all_zero {
            return Err(burn_qwen3_vl::Qwen3VlError::InvalidInput(
                "browser Qwen block 0 is all zero immediately after its WebGPU barrier".into(),
            ));
        }
        Ok(())
    }
}

struct BrowserDenoiserStageObserver {
    control: BrowserParityControl,
    step: usize,
}

impl BrowserDenoiserStageObserver {
    fn names(&self, boundary: &str) -> (String, String) {
        let oracle = format!("denoiser.step.{}.{}", self.step, boundary);
        (oracle.clone(), oracle)
    }
}

impl DenoiserStageObserver<BrowserBackend> for BrowserDenoiserStageObserver {
    fn rank2(&mut self, name: &str, tensor: Tensor<BrowserBackend, 2>) -> Result<(), BooguError> {
        let (name, oracle) = self.names(name);
        self.control.rank2(name, oracle, tensor);
        Ok(())
    }

    fn rank3(&mut self, name: &str, tensor: Tensor<BrowserBackend, 3>) -> Result<(), BooguError> {
        let (name, oracle) = self.names(name);
        self.control.rank3(name, oracle, tensor);
        Ok(())
    }

    fn rank4(&mut self, name: &str, tensor: Tensor<BrowserBackend, 4>) -> Result<(), BooguError> {
        let (name, oracle) = self.names(name);
        self.control.rank4(name, oracle, tensor);
        Ok(())
    }
}

impl<S> AsyncBooguVaeStageSource<BrowserBackend> for BrowserAsyncStageSource<S>
where
    S: AsyncBooguVaeStageSource<BrowserBackend>,
{
    async fn load_encoder(&mut self) -> Result<AutoencoderKl<BrowserBackend>, BooguError> {
        self.inner.load_encoder().await
    }

    async fn load_decoder(&mut self) -> Result<AutoencoderKl<BrowserBackend>, BooguError> {
        self.inner.load_decoder().await
    }

    async fn synchronize(&mut self) -> Result<(), BooguError> {
        self.synchronize_with_parity("VAE stage").await
    }
}

impl<S> AsyncBooguDenoiserStageSource<BrowserBackend> for BrowserAsyncStageSource<S>
where
    S: AsyncBooguDenoiserStageSource<BrowserBackend>,
{
    async fn load_prelude(&mut self) -> Result<BooguDenoiserPrelude<BrowserBackend>, BooguError> {
        self.inner.load_prelude().await
    }

    async fn load_context_refiner(
        &mut self,
        index: usize,
    ) -> Result<SingleStreamBlock<BrowserBackend>, BooguError> {
        let mut block = self.inner.load_context_refiner(index).await?;
        if let Some(query_chunk_size) = self.denoiser_query_chunk_size {
            block.attn.set_query_chunk_size(query_chunk_size);
        }
        Ok(block)
    }

    async fn load_noise_refiner(
        &mut self,
        index: usize,
    ) -> Result<SingleStreamBlock<BrowserBackend>, BooguError> {
        let mut block = self.inner.load_noise_refiner(index).await?;
        if let Some(query_chunk_size) = self.denoiser_query_chunk_size {
            block.attn.set_query_chunk_size(query_chunk_size);
        }
        Ok(block)
    }

    async fn load_reference_refiner(
        &mut self,
        index: usize,
    ) -> Result<SingleStreamBlock<BrowserBackend>, BooguError> {
        let mut block = self.inner.load_reference_refiner(index).await?;
        if let Some(query_chunk_size) = self.denoiser_query_chunk_size {
            block.attn.set_query_chunk_size(query_chunk_size);
        }
        Ok(block)
    }

    async fn load_double_stream(
        &mut self,
        index: usize,
    ) -> Result<DoubleStreamBlock<BrowserBackend>, BooguError> {
        let mut block = self.inner.load_double_stream(index).await?;
        if let Some(query_chunk_size) = self.denoiser_query_chunk_size {
            block.joint_attn.set_query_chunk_size(query_chunk_size);
            block.image_self_attn.set_query_chunk_size(query_chunk_size);
        }
        Ok(block)
    }

    async fn load_single_stream(
        &mut self,
        index: usize,
    ) -> Result<SingleStreamBlock<BrowserBackend>, BooguError> {
        let mut block = self.inner.load_single_stream(index).await?;
        if let Some(query_chunk_size) = self.denoiser_query_chunk_size {
            block.attn.set_query_chunk_size(query_chunk_size);
        }
        Ok(block)
    }

    async fn load_tail(&mut self) -> Result<BooguDenoiserTail<BrowserBackend>, BooguError> {
        self.inner.load_tail().await
    }

    async fn cooperative_yield(&mut self) -> Result<(), BooguError> {
        self.cooperative_browser_yield().await
    }

    async fn synchronize(&mut self) -> Result<(), BooguError> {
        let result = self.synchronize_with_parity("denoiser stage").await;
        if result.is_ok() {
            self.deferred_stages_since_yield = 0;
        }
        result
    }
}

/// Result of the opt-in no-surface diagnostic bootstrap.
///
/// This report proves WebGPU compute, the real factory validation/build path, bounded verified
/// range transport, and one real Qwen stage. It is deliberately not an image or numerical-parity
/// result.
#[derive(Clone, Debug, serde::Serialize)]
pub struct BrowserBooguBootstrapReport {
    pub mode: String,
    pub model_backend: String,
    pub adapter_name: String,
    pub adapter_backend: String,
    pub adapter_device_type: String,
    pub adapter_shader_f16: bool,
    pub device_shader_f16: bool,
    pub max_post_load_tensor_bytes: u64,
    pub requested_device_buffer_limit: u64,
    pub actual_storage_buffer_binding_size: u64,
    pub actual_max_buffer_size: u64,
    pub model: String,
    pub model_revision: String,
    pub artifact_content_digest: burn_image::Sha256Digest,
    pub numeric_format: burn_image::NumericFormat,
    pub qwen_float_load_policy: String,
    pub qwen_quantized_load_policy: String,
    pub vae_float_load_policy: String,
    pub denoiser_float_load_policy: String,
    pub denoiser_storage_policy: String,
    pub denoiser_quantized_load_policy: String,
    pub denoiser_quantized_linear_execution_policy: String,
    pub denoiser_linear_execution_policy: String,
    pub qwen_visual_execution_dtype: String,
    pub vae_execution_dtype: String,
    pub denoiser_execution_dtype: String,
    pub probe_stage: String,
    pub probe_stage_bytes: u64,
    pub probe_dtype: String,
    pub probe_elements: usize,
    pub probe_finite_elements: usize,
    pub probe_verified_objects: usize,
    pub artifacts_verified: bool,
    pub numerical_parity_claimed: bool,
}

/// One bounded packed-Q4 matrix multiplication measurement from the real browser backend.
#[derive(Clone, Debug, serde::Serialize)]
pub struct BrowserQ4MatmulProbeCase {
    pub lhs_shape: [usize; 2],
    pub rhs_shape: [usize; 2],
    pub first_dispatch_ms: f64,
    pub warm_dispatch_ms: Vec<f64>,
    pub warm_median_ms: f64,
    pub warm_gflops: f64,
    pub output_elements: usize,
    pub output_finite_elements: usize,
    pub output_nonzero_elements: usize,
}

/// Surface-free packed-Q4 WebGPU correctness and throughput probe.
///
/// This deliberately creates synthetic tensors and never opens a model manifest. It is the cheap
/// preflight for browser-kernel changes; a passing report is not model or image parity evidence.
#[derive(Clone, Debug, serde::Serialize)]
pub struct BrowserQ4MatmulProbeReport {
    pub mode: String,
    pub model_backend: String,
    pub adapter_name: String,
    pub adapter_backend: String,
    pub adapter_device_type: String,
    pub max_storage_buffer_binding_size: u64,
    pub max_buffer_size: u64,
    pub correctness_max_abs_error: f32,
    pub correctness_passed: bool,
    pub representative: BrowserQ4MatmulProbeCase,
    pub diagnostic_passed: bool,
    pub model_artifacts_loaded: bool,
    pub numerical_parity_claimed: bool,
}

/// Evidence from one complete surface-free production request.
///
/// This report records an ordinary model request and its output, not a numerical fixture replay.
#[derive(Clone, Debug, serde::Serialize)]
pub struct BrowserBooguInferenceReport {
    pub mode: String,
    pub model_backend: String,
    pub adapter_name: String,
    pub adapter_backend: String,
    pub adapter_device_type: String,
    pub adapter_shader_f16: bool,
    pub device_shader_f16: bool,
    pub max_post_load_tensor_bytes: u64,
    pub requested_device_buffer_limit: u64,
    pub actual_storage_buffer_binding_size: u64,
    pub actual_max_buffer_size: u64,
    pub model: String,
    pub model_revision: String,
    pub artifact_content_digest: Sha256Digest,
    pub numeric_format: burn_image::NumericFormat,
    pub qwen_float_load_policy: String,
    pub qwen_quantized_load_policy: String,
    pub vae_float_load_policy: String,
    pub denoiser_float_load_policy: String,
    pub denoiser_storage_policy: String,
    pub denoiser_quantized_load_policy: String,
    pub denoiser_quantized_linear_execution_policy: String,
    pub denoiser_linear_execution_policy: String,
    pub qwen_visual_execution_dtype: String,
    pub vae_execution_dtype: String,
    pub denoiser_execution_dtype: String,
    pub denoiser_requested_query_chunk_size: usize,
    pub denoiser_minimum_image_query_partitions: usize,
    pub qwen_release_unused_memory_after_stage: bool,
    pub block0_execution_mode: String,
    pub qwen_embedding_execution_policy: String,
    pub qwen_text_layer_allocation_policy: String,
    pub qwen_text_block_load_synchronization_policy: String,
    pub qwen_text_layer_submission_policy: String,
    pub packed_qwen_instruction_handoff_policy: String,
    pub packed_f16_dmd_vae_handoff_policy: String,
    pub resident_resource_plan: Option<BrowserResidentResourcePlan>,
    pub low_vram_resource_plan: Option<BrowserLowVramResourcePlan>,
    pub packed_f16_resource_plan: Option<BrowserPackedF16ResourcePlan>,
    pub packed_f16_qwen_embedding_plan: Option<BrowserPackedF16QwenEmbeddingPlan>,
    pub packed_f16_qwen_host_embedding: Option<HostRoutedEmbeddingReport>,
    pub packed_f16_qwen_instruction_handoff: Option<BrowserPackedF16QwenInstructionHandoffReport>,
    pub packed_f16_denoiser_lifecycle: Option<BrowserPackedF16DenoiserLifecycleReport>,
    pub packed_f16_dmd_vae_handoff: Option<BrowserPackedF16DmdVaeHandoffReport>,
    pub low_vram_denoiser_dtype_audit: Option<BrowserLowVramDenoiserDTypeAudit>,
    pub dense_f32_materialized_stage_clones: usize,
    pub artifact_traffic: BrowserArtifactTrafficReport,
    pub weight_traffic_contract: String,
    pub on_device_quantized_execution_claimed: bool,
    pub prompt: String,
    pub dimensions: Dimensions,
    pub seed: u64,
    pub png_file_name: String,
    pub png_bytes: u64,
    pub png_sha256: Sha256Digest,
    pub peak_wasm_linear_memory_bytes: u64,
    pub timings: StageTimings,
    pub provenance: ModelProvenance,
    pub artifacts_verified: bool,
    pub numerical_parity_claimed: bool,
}

/// Complete diagnostic output kept separate from its concise JSON report.
pub struct BrowserBooguInferenceResult {
    pub report: BrowserBooguInferenceReport,
    pub png: Vec<u8>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct BrowserParityExactMetric {
    pub name: String,
    pub shape: Vec<usize>,
    pub elements: u64,
    pub exact: bool,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct BrowserParityProcessingReport {
    pub prompt_exact: bool,
    pub dimensions_exact: bool,
    pub seed_exact: bool,
    pub effective_instruction_length: usize,
    pub integer_tensors: Vec<BrowserParityExactMetric>,
    pub pixel_values: BrowserParityTensorMetric,
    pub mrope_cos: BrowserParityTensorMetric,
    pub mrope_sin: BrowserParityTensorMetric,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct BrowserParityQwenReport {
    pub expected_aligned_stages: usize,
    pub compared_aligned_stages: usize,
    pub unique_compared_aligned_stages: usize,
    pub aligned_stage_names_exact: bool,
    pub aligned_stages: Vec<BrowserParityTensorMetric>,
    pub final_hidden_state: BrowserParityTensorMetric,
    pub expected_authenticated_only_diagnostics: usize,
    pub authenticated_only_diagnostics_exact: bool,
    pub authenticated_unaligned_diagnostics: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct BrowserParityVaeReferenceReport {
    pub input: BrowserParityTensorMetric,
    pub injected_epsilon: BrowserParityTensorMetric,
    pub f32_oracle: Vec<BrowserParityTensorMetric>,
    pub upstream_bf16_drift: Vec<BrowserParityTensorMetric>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct BrowserVaeReferenceDiagnosticRun {
    pub index: usize,
    pub elapsed_micros: u64,
    pub f32_oracle: Vec<BrowserParityTensorMetric>,
    pub upstream_bf16_drift: Vec<BrowserParityTensorMetric>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct BrowserVaeReferenceStabilityMetric {
    pub repeat_index: usize,
    pub component: String,
    pub shape: Vec<usize>,
    pub bitwise_exact: bool,
    #[serde(flatten)]
    pub comparison: FloatMetrics,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct BrowserVaeReferenceStabilityReport {
    pub baseline_repeat_index: usize,
    pub compared_repeats: usize,
    pub compared_components: usize,
    pub all_bitwise_exact: bool,
    pub maximum_abs: f32,
    pub maximum_rmse: f32,
    pub minimum_cosine_similarity: f32,
    pub metrics: Vec<BrowserVaeReferenceStabilityMetric>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct BrowserVaeEncoderArtifactObjectReport {
    pub path: String,
    pub size: u64,
    pub verification_count: usize,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct BrowserVaeEncoderArtifactVerificationReport {
    pub scope: String,
    pub expected_repeats: usize,
    pub expected_unique_weight_objects: usize,
    pub verified_unique_weight_objects: usize,
    pub expected_weight_bytes_per_repeat: u64,
    pub verified_weight_bytes_all_repeats: u64,
    pub missing_weight_objects: Vec<String>,
    pub unexpected_verified_objects: Vec<String>,
    pub objects: Vec<BrowserVaeEncoderArtifactObjectReport>,
    pub verification_count_per_object_exact: bool,
    pub passed: bool,
}

/// Diagnostic-only evidence from three fresh VAE encoder executions on one browser WebGPU device.
///
/// This deliberately cannot make a complete browser parity claim: it executes no Qwen, denoiser,
/// DMD, VAE decoder, or final-pixel surface.
#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct BrowserBooguVaeReferenceDiagnosticReport {
    pub report_schema_version: u32,
    pub mode: String,
    pub model_backend: String,
    pub adapter_name: String,
    pub adapter_backend: String,
    pub adapter_device_type: String,
    pub adapter_shader_f16: bool,
    pub device_shader_f16: bool,
    pub minimum_required_device_buffer_limit: u64,
    pub actual_storage_buffer_binding_size: u64,
    pub actual_max_buffer_size: u64,
    pub model: String,
    pub model_revision: String,
    pub artifact_content_digest: Sha256Digest,
    pub numeric_format: burn_image::NumericFormat,
    pub artifact_profile: String,
    pub vae_float_load_policy: String,
    pub vae_execution_dtype: String,
    pub expected_repeats: usize,
    pub completed_repeats: usize,
    pub fixture: BrowserParityFixtureIdentity,
    pub fixture_verification: BrowserParityVerificationSnapshot,
    pub input: BrowserParityTensorMetric,
    pub injected_epsilon: BrowserParityTensorMetric,
    pub runs: Vec<BrowserVaeReferenceDiagnosticRun>,
    pub stability: BrowserVaeReferenceStabilityReport,
    pub artifact_verification: BrowserVaeEncoderArtifactVerificationReport,
    pub browser_webgpu_vae_f32_oracle_envelope: BrowserWebGpuVaeF32OracleEnvelope,
    pub gate_failures: Vec<String>,
    pub peak_wasm_linear_memory_bytes: u64,
    pub artifacts_verified: bool,
    pub fixture_authenticated: bool,
    pub diagnostic_passed: bool,
    pub numerical_parity_claimed: bool,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct BrowserParityDmdStepReport {
    pub index: usize,
    pub schedule_sigma: f32,
    pub fixture_sigma: f32,
    pub sigma_exact: bool,
    pub input: BrowserParityTensorMetric,
    pub velocity: BrowserParityTensorMetric,
    pub prediction: BrowserParityTensorMetric,
    pub injected_noise: Option<BrowserParityTensorMetric>,
    pub renoised: Option<BrowserParityTensorMetric>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct BrowserParityDmdReport {
    pub initial_latent: BrowserParityTensorMetric,
    pub steps: Vec<BrowserParityDmdStepReport>,
    pub final_latent: BrowserParityTensorMetric,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct BrowserParityTensorGate {
    pub maximum_relative_rmse: f32,
    pub minimum_cosine_similarity: f32,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct BrowserParityReferenceGate {
    pub maximum_abs: f32,
    pub maximum_rmse: f32,
    pub minimum_cosine_similarity: f32,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct BrowserParityMaximumAbsGate {
    pub maximum_abs: f32,
}

/// Calibrated envelope for the canonical F16-stored VAE weights widened to F32 on WebGPU.
///
/// This is deliberately backend, artifact, storage, adapter, device, and driver scoped. It is not
/// a native gate and does not assert portability to another WebGPU implementation or adapter.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct BrowserWebGpuVaeF32OracleEnvelope {
    pub backend: &'static str,
    pub artifact_content_digest: &'static str,
    pub artifact_profile: &'static str,
    pub weight_storage_dtype: &'static str,
    pub weight_load_policy: &'static str,
    pub execution_dtype: &'static str,
    pub calibrated_adapter: &'static str,
    pub calibrated_device: &'static str,
    pub calibrated_driver: &'static str,
    pub portability: &'static str,
    pub moments: BrowserParityReferenceGate,
    pub mean: BrowserParityMaximumAbsGate,
    pub logvar: BrowserParityMaximumAbsGate,
    pub std: BrowserParityMaximumAbsGate,
    pub raw_latent: BrowserParityMaximumAbsGate,
    pub scaled_latent: BrowserParityReferenceGate,
}

impl BrowserWebGpuVaeF32OracleEnvelope {
    fn component_maximum(self, name: &str) -> Option<f32> {
        if name.ends_with("moments") {
            Some(self.moments.maximum_abs)
        } else if name.ends_with("mean") {
            Some(self.mean.maximum_abs)
        } else if name.ends_with("logvar") {
            Some(self.logvar.maximum_abs)
        } else if name.ends_with("std") {
            Some(self.std.maximum_abs)
        } else if name.ends_with("raw_latent") {
            Some(self.raw_latent.maximum_abs)
        } else if name.ends_with("scaled_latent") {
            Some(self.scaled_latent.maximum_abs)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct BrowserParityRgbGate {
    pub minimum_psnr_db: f32,
    pub minimum_mean_block_ssim_8x8: f32,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct BrowserParityGateReport {
    pub passed: bool,
    pub qwen_aligned_stages: BrowserParityTensorGate,
    pub qwen_final: BrowserParityTensorGate,
    pub browser_webgpu_vae_f32_oracle_envelope: BrowserWebGpuVaeF32OracleEnvelope,
    pub denoiser_boundaries: BrowserParityTensorGate,
    pub dmd_boundaries: BrowserParityTensorGate,
    pub dmd_final: BrowserParityTensorGate,
    pub propagated_decode: BrowserParityTensorGate,
    pub final_rgb: BrowserParityRgbGate,
    pub failures: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct BrowserParityTensorCoverageReport {
    pub fixture_tensors: usize,
    pub expected_numerically_compared_semantic_tensors: usize,
    pub unique_numerically_compared_semantic_tensors: usize,
    pub expected_authenticated_only_tensors: usize,
    pub numerical_name_set_exact: bool,
    pub authenticated_only_name_set_exact: bool,
    pub scope: String,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct BrowserParityArtifactVerificationReport {
    pub scope: String,
    pub sealed_manifest_validated: bool,
    pub canonical_release_digest_validated: bool,
    pub verified_runtime_metadata_objects: usize,
    pub expected_runtime_metadata_objects: usize,
    pub verified_unique_weight_objects: usize,
    pub expected_unique_weight_objects: usize,
    pub verified_unique_weight_bytes: u64,
    pub expected_unique_weight_bytes: u64,
    pub missing_weight_objects: Vec<String>,
    pub unexpected_verified_objects: Vec<String>,
    pub passed: bool,
}

/// Fail-closed evidence from the exact 1536-square exhaustive browser fixture replay.
#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct BrowserBooguParityReport {
    pub report_schema_version: u32,
    pub mode: String,
    pub model_backend: String,
    pub adapter_name: String,
    pub adapter_backend: String,
    pub adapter_device_type: String,
    pub adapter_shader_f16: bool,
    pub device_shader_f16: bool,
    pub minimum_required_device_buffer_limit: u64,
    pub actual_storage_buffer_binding_size: u64,
    pub actual_max_buffer_size: u64,
    pub model: String,
    pub model_revision: String,
    pub artifact_content_digest: Sha256Digest,
    pub numeric_format: burn_image::NumericFormat,
    pub artifact_profile: String,
    pub residency_policy: String,
    pub qwen_float_load_policy: String,
    pub vae_float_load_policy: String,
    pub denoiser_float_load_policy: String,
    pub denoiser_storage_policy: String,
    pub denoiser_quantized_load_policy: String,
    pub denoiser_quantized_linear_execution_policy: String,
    pub denoiser_linear_execution_policy: String,
    pub qwen_execution_dtype: String,
    pub vae_execution_dtype: String,
    pub denoiser_execution_dtype: String,
    pub qwen_query_chunk_size: usize,
    pub vae_attention_query_chunk_size: usize,
    pub vae_decode_policy: String,
    pub vae_decode_max_planned_buffer_bytes: u64,
    pub denoiser_query_chunk_size: usize,
    pub denoiser_residency: String,
    pub denoiser_expected_retained_stages: usize,
    pub denoiser_retained_stages_before_clear: usize,
    pub denoiser_cache_cleared_before_decode: bool,
    pub resident_resource_plan: Option<BrowserResidentResourcePlan>,
    pub low_vram_resource_plan: Option<BrowserLowVramResourcePlan>,
    pub low_vram_denoiser_dtype_audit: Option<BrowserLowVramDenoiserDTypeAudit>,
    pub weight_traffic_contract: String,
    pub on_device_quantized_execution_claimed: bool,
    pub fixture: BrowserParityFixtureIdentity,
    pub fixture_verification: BrowserParityVerificationSnapshot,
    pub tensor_coverage: BrowserParityTensorCoverageReport,
    pub artifact_verification: BrowserParityArtifactVerificationReport,
    pub processing: BrowserParityProcessingReport,
    pub qwen: BrowserParityQwenReport,
    pub vae_reference: BrowserParityVaeReferenceReport,
    pub denoiser_expected_boundaries: usize,
    pub denoiser_compared_boundaries: usize,
    pub denoiser_unique_compared_boundaries: usize,
    pub denoiser_boundary_names_exact: bool,
    pub denoiser_boundaries: Vec<BrowserParityTensorMetric>,
    pub dmd: BrowserParityDmdReport,
    pub decode_input: BrowserParityTensorMetric,
    pub decoded_tensor: BrowserParityTensorMetric,
    pub final_rgb: RgbMetrics,
    pub fixture_output_png_sha256: Sha256Digest,
    pub peak_wasm_linear_memory_bytes: u64,
    pub gates: BrowserParityGateReport,
    pub artifacts_verified: bool,
    pub fixture_authenticated: bool,
    pub numerical_parity_claimed: bool,
}

struct BrowserNoSurfaceEngine {
    engine: BrowserBooguEngine,
    // `init_setup_async` also registers a cloned setup with CubeCL. Retain this complete external
    // setup for the enclosing diagnostic future as an additional browser-lifetime guard and to
    // keep the device diagnostics below installed until the report is complete.
    _setup_guard: burn_wgpu::WgpuSetup,
    adapter: wgpu::AdapterInfo,
    backend: wgpu::Backend,
    limits: wgpu::Limits,
    adapter_shader_f16: bool,
    device_shader_f16: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BrowserNoSurfacePolicy {
    CompatibleF32,
    PreserveQwenF16,
    ResidentPackedF16,
    ResidentPackedQ4s,
    ResidentDenseF32,
    LowVramRuntimeQ8Denoiser,
    LowVramPreloadedPackedF16Denoiser,
}

fn q4_probe_lhs_values(rows: usize, inner: usize) -> Vec<f32> {
    (0..rows * inner)
        .map(|index| ((index.wrapping_mul(17) % 29) as f32 - 14.0) * (1.0 / 128.0))
        .collect()
}

fn q4_probe_weight_values(inner: usize, output: usize) -> Vec<f32> {
    (0..inner * output)
        .map(|index| {
            let input = index / output;
            let output = index % output;
            (((input + output.wrapping_mul(3)) % 15) as f32 - 7.0) * (1.0 / 64.0)
        })
        .collect()
}

fn q4_probe_reference_matmul(
    lhs: &[f32],
    weight: &[f32],
    rows: usize,
    inner: usize,
    output: usize,
) -> Vec<f32> {
    let mut result = vec![0.0_f32; rows * output];
    for row in 0..rows {
        for column in 0..output {
            let mut sum = 0.0_f32;
            for reduction in 0..inner {
                sum += lhs[row * inner + reduction] * weight[reduction * output + column];
            }
            result[row * output + column] = sum;
        }
    }
    result
}

async fn q4_probe_output(
    device: &burn_wgpu::WgpuDevice,
    lhs: &Tensor<BrowserBackend, 2>,
    weight: &Tensor<BrowserBackend, 2>,
) -> Result<(Vec<f32>, f64), BooguError> {
    let started_ms = js_sys::Date::now();
    let data = lhs
        .clone()
        .matmul(weight.clone())
        .into_data_async()
        .await
        .map_err(|error| {
            BooguError::Artifact(format!("packed-Q4 probe readback failed: {error}"))
        })?;
    // Keep a named device argument at this boundary: the async readback above is the exact queue
    // completion barrier for this dispatch, and callers reuse the same device for every sample.
    let _ = device;
    let elapsed_ms = (js_sys::Date::now() - started_ms).max(0.0);
    let values = data.to_vec::<f32>().map_err(|error| {
        BooguError::Artifact(format!("packed-Q4 probe output conversion failed: {error}"))
    })?;
    Ok((values, elapsed_ms))
}

async fn run_q4_matmul_probe_case(
    device: &burn_wgpu::WgpuDevice,
    rows: usize,
    inner: usize,
    output: usize,
) -> Result<BrowserQ4MatmulProbeCase, BooguError> {
    const WARM_ITERATIONS: usize = 3;

    let lhs = Tensor::<BrowserBackend, 2>::from_data(
        TensorData::new(q4_probe_lhs_values(rows, inner), [rows, inner]),
        device,
    );
    let weight_data = burn_boogu::artifacts::quantize_row_layout_q4s_block_up_to128_f32(
        TensorData::new(q4_probe_weight_values(inner, output), [inner, output]),
    )
    .map_err(|error| {
        BooguError::Artifact(format!("packed-Q4 probe quantization failed: {error}"))
    })?;
    let weight = Tensor::<BrowserBackend, 2>::from_data(weight_data, device);

    let (first_values, first_dispatch_ms) = q4_probe_output(device, &lhs, &weight).await?;
    let mut warm_dispatch_ms = Vec::with_capacity(WARM_ITERATIONS);
    let mut final_values = first_values;
    for _ in 0..WARM_ITERATIONS {
        let (values, elapsed_ms) = q4_probe_output(device, &lhs, &weight).await?;
        final_values = values;
        warm_dispatch_ms.push(elapsed_ms);
    }
    let mut ordered = warm_dispatch_ms.clone();
    ordered.sort_by(f64::total_cmp);
    let warm_median_ms = ordered[ordered.len() / 2];
    let operations = 2.0 * rows as f64 * inner as f64 * output as f64;
    let warm_gflops = if warm_median_ms > 0.0 {
        operations / warm_median_ms / 1_000_000.0
    } else {
        f64::INFINITY
    };
    let output_finite_elements = final_values
        .iter()
        .filter(|value| value.is_finite())
        .count();
    let output_nonzero_elements = final_values.iter().filter(|value| **value != 0.0).count();

    Ok(BrowserQ4MatmulProbeCase {
        lhs_shape: [rows, inner],
        rhs_shape: [inner, output],
        first_dispatch_ms,
        warm_dispatch_ms,
        warm_median_ms,
        warm_gflops,
        output_elements: final_values.len(),
        output_finite_elements,
        output_nonzero_elements,
    })
}

/// Asynchronously builds one pinned browser release from a remote sealed artifact directory.
pub struct BrowserBooguFactory {
    variant: BooguVariant,
    residency: BrowserBooguResidencyPolicy,
    pending: Option<BrowserBuildSlot>,
    started: bool,
}

impl BrowserBooguFactory {
    /// Select the immutable production release expected below the configured remote base URL.
    ///
    /// Every public variant defaults to resident direct Q4S. Use [`Self::with_residency`] for an
    /// explicit validation route.
    pub const fn new(variant: BooguVariant) -> Self {
        Self {
            variant,
            residency: default_browser_residency(variant),
            pending: None,
            started: false,
        }
    }

    /// Select an explicit browser residency contract.
    pub const fn with_residency(
        variant: BooguVariant,
        residency: BrowserBooguResidencyPolicy,
    ) -> Self {
        Self {
            variant,
            residency,
            pending: None,
            started: false,
        }
    }

    fn validate_context(
        variant: BooguVariant,
        context: BooguFactoryContext,
    ) -> Result<BrowserBuildInputs, RuntimeError> {
        crate::boogu::validate_execution_variant(variant, context.execution)?;
        if context.execution != WgpuExecutionKind::BrowserWebGpu {
            return Err(execution_error(
                variant,
                "browser Boogu factory requires an attested WebGPU device",
            ));
        }
        if matches!(context.device, burn_wgpu::WgpuDevice::Cpu) {
            return Err(execution_error(
                variant,
                "browser Boogu factory refuses a CPU Burn device",
            ));
        }
        if context.settings.integrity != burn_image::IntegrityPolicy::RequireSha256 {
            return Err(execution_error(
                variant,
                "browser Boogu factory requires SHA-256 verification",
            ));
        }
        context
            .settings
            .validate_concrete_cache_policy()
            .map_err(|error| execution_error(variant, error))?;
        crate::boogu::validate_browser_variant_buffer_limits(
            variant,
            context.max_storage_buffer_binding_size,
            context.max_buffer_size,
        )?;
        let base_url = match &context.settings.artifact_source {
            burn_image::ArtifactSource::Remote { base_url } => base_url.clone(),
            burn_image::ArtifactSource::LocalDirectory { .. } => {
                return Err(execution_error(
                    variant,
                    "browser Boogu factory requires a remote artifact base URL",
                ));
            }
        };
        let identity = BooguReleaseIdentity::canonical(variant);
        if !context.releases.contains(&identity) {
            return Err(execution_error(
                variant,
                "browser factory context omits the selected canonical release",
            ));
        }
        Ok(BrowserBuildInputs {
            identity,
            base_url,
            settings: context.settings,
            device: context.device,
            allocation_device: context.allocation_device,
            applied_buffer_limits: BrowserAppliedBufferLimits {
                max_storage_buffer_binding_size: context.max_storage_buffer_binding_size,
                max_buffer_size: context.max_buffer_size,
            },
        })
    }

    /// Run a bounded, artifact-free packed-Q4 WebGPU correctness and throughput probe.
    ///
    /// The small matrix validates the exact row-layout Q4 packing contract. The representative
    /// 45x4096x12288 projection matches the released Qwen MLP geometry that previously exposed an
    /// oversized workgroup stage, while remaining cheap compared with loading or executing a
    /// model.
    pub async fn q4_matmul_probe_no_surface(
        variant: BooguVariant,
    ) -> Result<BrowserQ4MatmulProbeReport, RuntimeError> {
        let (device, setup) = init_no_surface_wgpu_setup(variant).await?;
        install_no_surface_device_diagnostics(&setup);
        let adapter = setup.adapter.get_info();
        let limits = setup.device.limits();

        const CORRECTNESS_ROWS: usize = 4;
        const CORRECTNESS_INNER: usize = 128;
        const CORRECTNESS_OUTPUT: usize = 128;
        let correctness_lhs = q4_probe_lhs_values(CORRECTNESS_ROWS, CORRECTNESS_INNER);
        let correctness_weight = q4_probe_weight_values(CORRECTNESS_INNER, CORRECTNESS_OUTPUT);
        let expected = q4_probe_reference_matmul(
            &correctness_lhs,
            &correctness_weight,
            CORRECTNESS_ROWS,
            CORRECTNESS_INNER,
            CORRECTNESS_OUTPUT,
        );
        let lhs = Tensor::<BrowserBackend, 2>::from_data(
            TensorData::new(correctness_lhs, [CORRECTNESS_ROWS, CORRECTNESS_INNER]),
            &device,
        );
        let weight_data = burn_boogu::artifacts::quantize_row_layout_q4s_block_up_to128_f32(
            TensorData::new(correctness_weight, [CORRECTNESS_INNER, CORRECTNESS_OUTPUT]),
        )
        .map_err(|error| execution_error(variant, format!("Q4 probe quantization: {error}")))?;
        let weight = Tensor::<BrowserBackend, 2>::from_data(weight_data, &device);
        let (actual, _) = q4_probe_output(&device, &lhs, &weight)
            .await
            .map_err(|error| map_boogu(variant, error))?;
        if actual.len() != expected.len() {
            return Err(execution_error(
                variant,
                format!(
                    "Q4 probe output length {} did not match expected {}",
                    actual.len(),
                    expected.len()
                ),
            ));
        }
        let correctness_max_abs_error = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        let correctness_passed =
            actual.iter().all(|value| value.is_finite()) && correctness_max_abs_error <= 0.000_5;

        let representative = run_q4_matmul_probe_case(&device, 45, 4096, 12_288)
            .await
            .map_err(|error| map_boogu(variant, error))?;
        let diagnostic_passed = correctness_passed
            && representative.output_finite_elements == representative.output_elements
            && representative.output_nonzero_elements > 0;

        <BrowserBackend as Backend>::memory_cleanup(&device);
        BrowserAsyncSynchronizer::new(&device)
            .synchronize("packed-Q4 microbenchmark cleanup")
            .await
            .map_err(|error| map_boogu(variant, error))?;

        Ok(BrowserQ4MatmulProbeReport {
            mode: "diagnostic-no-surface-packed-q4-matmul".into(),
            model_backend: "burn-wgpu/raw-cubecl/webgpu".into(),
            adapter_name: adapter.name,
            adapter_backend: format!("{:?}", adapter.backend),
            adapter_device_type: format!("{:?}", adapter.device_type),
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            max_buffer_size: limits.max_buffer_size,
            correctness_max_abs_error,
            correctness_passed,
            representative,
            diagnostic_passed,
            model_artifacts_loaded: false,
            numerical_parity_claimed: false,
        })
    }

    /// Initialize a surface-free Burn WebGPU device, build this exact verified factory, and run a
    /// small real Qwen final-norm stage.
    ///
    /// This diagnostic intentionally owns a separate Burn device and never enters Bevy's render
    /// app. The normal browser app continues to require the shared Bevy/Burn device.
    pub async fn bootstrap_no_surface(
        variant: BooguVariant,
        settings: crate::BooguAdapterSettings,
    ) -> Result<BrowserBooguBootstrapReport, RuntimeError> {
        Self::bootstrap_no_surface_with_policy(
            variant,
            settings,
            BrowserNoSurfacePolicy::CompatibleF32,
        )
        .await
    }

    /// Run the same bounded final-norm diagnostic while preserving Qwen's F16 stage dtype.
    ///
    /// This fails closed before artifact construction when the adapter or requested device does
    /// not expose WebGPU `shader-f16`. It does not change the all-F32 production browser policy.
    pub async fn bootstrap_no_surface_preserve_f16(
        variant: BooguVariant,
        settings: crate::BooguAdapterSettings,
    ) -> Result<BrowserBooguBootstrapReport, RuntimeError> {
        Self::bootstrap_no_surface_with_policy(
            variant,
            settings,
            BrowserNoSurfacePolicy::PreserveQwenF16,
        )
        .await
    }

    async fn bootstrap_no_surface_with_policy(
        variant: BooguVariant,
        settings: crate::BooguAdapterSettings,
        policy: BrowserNoSurfacePolicy,
    ) -> Result<BrowserBooguBootstrapReport, RuntimeError> {
        let BrowserNoSurfaceEngine {
            mut engine,
            _setup_guard,
            adapter,
            backend,
            limits,
            adapter_shader_f16,
            device_shader_f16,
        } = build_no_surface_engine(variant, settings, policy).await?;
        let probe = engine.probe_qwen_text_final_norm().await?;
        Ok(BrowserBooguBootstrapReport {
            mode: match policy {
                BrowserNoSurfacePolicy::CompatibleF32 => "diagnostic-no-surface-not-production",
                BrowserNoSurfacePolicy::PreserveQwenF16 => {
                    "diagnostic-no-surface-preserved-qwen-f16"
                }
                BrowserNoSurfacePolicy::ResidentPackedF16 => {
                    "no-surface-browser-high-vram-resident-packed-f16"
                }
                BrowserNoSurfacePolicy::ResidentPackedQ4s => {
                    "no-surface-browser-resident-packed-q4s"
                }
                BrowserNoSurfacePolicy::ResidentDenseF32 => {
                    "no-surface-browser-high-vram-resident-dense-f32"
                }
                BrowserNoSurfacePolicy::LowVramRuntimeQ8Denoiser => {
                    "no-surface-browser-low-vram-runtime-q8-denoiser"
                }
                BrowserNoSurfacePolicy::LowVramPreloadedPackedF16Denoiser => {
                    "no-surface-browser-low-vram-preloaded-packed-f16-dense-f32-per-stage-denoiser"
                }
            }
            .into(),
            model_backend: "raw-cubecl-no-fusion".into(),
            adapter_name: adapter.name,
            adapter_backend: format!("{backend:?}"),
            adapter_device_type: format!("{:?}", adapter.device_type),
            adapter_shader_f16,
            device_shader_f16,
            max_post_load_tensor_bytes: crate::boogu::BOOGU_BROWSER_MAX_APPLIED_BUFFER_BYTES,
            requested_device_buffer_limit: crate::boogu::BOOGU_BROWSER_REQUESTED_BUFFER_LIMIT_BYTES,
            actual_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            actual_max_buffer_size: limits.max_buffer_size,
            model: boogu_model_descriptor(variant).id.to_string(),
            model_revision: engine.identity.model_revision.clone(),
            artifact_content_digest: engine.artifact_content_digest,
            numeric_format: engine.numeric_format.clone(),
            qwen_float_load_policy: float_policy_name(engine.policies.qwen_float).into(),
            qwen_quantized_load_policy: quantized_policy_name(engine.policies.qwen_quantized)
                .into(),
            vae_float_load_policy: float_policy_name(engine.policies.vae_float).into(),
            denoiser_float_load_policy: float_policy_name(engine.policies.denoiser_float).into(),
            denoiser_storage_policy: engine.policies.denoiser_storage_policy().into(),
            denoiser_quantized_load_policy: engine
                .policies
                .denoiser_quantized_load_policy_report()
                .into(),
            denoiser_quantized_linear_execution_policy: engine
                .policies
                .denoiser_quantized_linear_execution_policy_report()
                .into(),
            denoiser_linear_execution_policy: engine
                .policies
                .denoiser_linear_execution_policy()
                .into(),
            qwen_visual_execution_dtype: engine.dtypes.qwen_visual.name().into(),
            vae_execution_dtype: engine.dtypes.vae.name().into(),
            denoiser_execution_dtype: engine.dtypes.denoiser.name().into(),
            probe_stage: probe.stage,
            probe_stage_bytes: probe.bytes,
            probe_dtype: probe.dtype,
            probe_elements: probe.elements,
            probe_finite_elements: probe.finite_elements,
            probe_verified_objects: probe.verified_objects,
            artifacts_verified: true,
            numerical_parity_claimed: false,
        })
    }

    /// Execute one complete validated browser request without creating a surface.
    ///
    /// The request is resolved by the same adapter path as the production UI. The returned PNG is
    /// a real model output; the report deliberately does not claim upstream fixture parity. The
    /// variant-aware model policy derives residency from the sealed storage profile.
    pub async fn infer_no_surface(
        variant: BooguVariant,
        settings: crate::BooguAdapterSettings,
        request: ImageRequest,
    ) -> Result<BrowserBooguInferenceResult, RuntimeError> {
        let residency = burn_boogu::deployment::browser_bounded_residency_policy(
            variant,
            settings.storage_profile,
        );
        Self::infer_no_surface_with_residency(variant, settings, residency, request).await
    }

    /// Execute a surface-free request with an explicit residency policy.
    pub async fn infer_no_surface_with_residency(
        variant: BooguVariant,
        settings: crate::BooguAdapterSettings,
        residency: BrowserBooguResidencyPolicy,
        request: ImageRequest,
    ) -> Result<BrowserBooguInferenceResult, RuntimeError> {
        let job = crate::boogu::prepare_runtime_job(ImageJobId(1), variant, request, &settings)?;
        let prompt = job.resolved.prompt.clone();
        let dimensions = job.resolved.dimensions;
        let BrowserNoSurfaceEngine {
            mut engine,
            _setup_guard,
            adapter,
            backend,
            limits,
            adapter_shader_f16,
            device_shader_f16,
        } = build_no_surface_engine(
            variant,
            settings,
            match residency {
                BrowserBooguResidencyPolicy::HighVramResidentPackedF16 => {
                    BrowserNoSurfacePolicy::ResidentPackedF16
                }
                BrowserBooguResidencyPolicy::ResidentPackedQ4s => {
                    BrowserNoSurfacePolicy::ResidentPackedQ4s
                }
                BrowserBooguResidencyPolicy::HighVramResidentDenseF32 => {
                    BrowserNoSurfacePolicy::ResidentDenseF32
                }
                BrowserBooguResidencyPolicy::QualificationPerRequestF32DenoiserRetained => {
                    return Err(execution_error(
                        variant,
                        "the per-request F32 denoiser-retained policy is reserved for exact-fixture qualification",
                    ));
                }
                BrowserBooguResidencyPolicy::LowVramRuntimeQ8Denoiser => {
                    BrowserNoSurfacePolicy::LowVramRuntimeQ8Denoiser
                }
                BrowserBooguResidencyPolicy::LowVramPreloadedPackedF16Denoiser => {
                    BrowserNoSurfacePolicy::LowVramPreloadedPackedF16Denoiser
                }
            },
        )
        .await?;

        let id = job.id;
        let run_id = RunId(id.0);
        let cancellation = CancellationToken::default();
        let shared = Arc::new(Mutex::new(BrowserRuntimeShared {
            engine: None,
            active: Some((id, cancellation.clone())),
            events: VecDeque::new(),
            diagnostic_console_progress: true,
            diagnostic_peak_wasm_linear_memory_bytes: wasm_linear_memory_bytes().unwrap_or(0),
        }));
        engine
            .artifact_control
            .set_cancellation(Some(cancellation.clone()));
        install_artifact_observer(&engine.artifact_control, &shared, id, run_id);
        inference_milestone("request-start");
        let output = engine.infer(&job, &cancellation, &shared).await;
        engine.artifact_control.set_observer(None);
        engine.artifact_control.set_cancellation(None);
        let output = output?;
        inference_milestone("png-encode-start");
        let image = output
            .images
            .first()
            .ok_or_else(|| execution_error(variant, "inference returned no image"))?;
        let png = crate::encode_host_image(&image.image, ImageEncoding::Png)
            .map_err(|error| execution_error(variant, error))?;
        let png_sha256 = Sha256Digest::calculate(&png);
        let png_file_name = format!(
            "burn-image-headless-{}-{}.png",
            variant_slug(variant),
            output.seed
        );
        let peak_wasm_linear_memory_bytes = {
            let state = shared.lock().expect("browser runtime mutex poisoned");
            state
                .diagnostic_peak_wasm_linear_memory_bytes
                .max(wasm_linear_memory_bytes().unwrap_or(0))
        };
        inference_milestone("request-complete");

        Ok(BrowserBooguInferenceResult {
            report: BrowserBooguInferenceReport {
                mode: format!("no-surface-full-request/{}", residency.label()),
                model_backend: "raw-cubecl-no-fusion".into(),
                adapter_name: adapter.name,
                adapter_backend: format!("{backend:?}"),
                adapter_device_type: format!("{:?}", adapter.device_type),
                adapter_shader_f16,
                device_shader_f16,
                max_post_load_tensor_bytes: crate::boogu::BOOGU_BROWSER_MAX_APPLIED_BUFFER_BYTES,
                requested_device_buffer_limit:
                    crate::boogu::BOOGU_BROWSER_REQUESTED_BUFFER_LIMIT_BYTES,
                actual_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
                actual_max_buffer_size: limits.max_buffer_size,
                model: boogu_model_descriptor(variant).id.to_string(),
                model_revision: engine.identity.model_revision.clone(),
                artifact_content_digest: engine.artifact_content_digest,
                numeric_format: engine.numeric_format.clone(),
                qwen_float_load_policy: float_policy_name(engine.policies.qwen_float).into(),
                qwen_quantized_load_policy: quantized_policy_name(engine.policies.qwen_quantized)
                    .into(),
                vae_float_load_policy: float_policy_name(engine.policies.vae_float).into(),
                denoiser_float_load_policy: float_policy_name(engine.policies.denoiser_float)
                    .into(),
                denoiser_storage_policy: engine.policies.denoiser_storage_policy().into(),
                denoiser_quantized_load_policy: engine
                    .policies
                    .denoiser_quantized_load_policy_report()
                    .into(),
                denoiser_quantized_linear_execution_policy: engine
                    .policies
                    .denoiser_quantized_linear_execution_policy_report()
                    .into(),
                denoiser_linear_execution_policy: engine
                    .policies
                    .denoiser_linear_execution_policy()
                    .into(),
                qwen_visual_execution_dtype: engine.dtypes.qwen_visual.name().into(),
                vae_execution_dtype: engine.dtypes.vae.name().into(),
                denoiser_execution_dtype: engine.dtypes.denoiser.name().into(),
                denoiser_requested_query_chunk_size: BROWSER_PRODUCTION_DENOISER_QUERY_CHUNK_SIZE,
                denoiser_minimum_image_query_partitions:
                    burn_boogu::PORTABLE_ATTENTION_MINIMUM_IMAGE_QUERY_PARTITIONS,
                qwen_release_unused_memory_after_stage: engine
                    .qwen
                    .releases_unused_memory_after_stage(),
                block0_execution_mode: browser_qwen_block0_execution_mode().into(),
                qwen_embedding_execution_policy: engine
                    .policies
                    .qwen_embedding_execution_policy()
                    .into(),
                qwen_text_layer_allocation_policy: engine
                    .policies
                    .qwen_text_layer_allocation_policy()
                    .into(),
                qwen_text_block_load_synchronization_policy: engine
                    .policies
                    .qwen_text_block_load_synchronization_policy()
                    .into(),
                qwen_text_layer_submission_policy: engine
                    .policies
                    .qwen_text_layer_submission_policy()
                    .into(),
                packed_qwen_instruction_handoff_policy: engine
                    .policies
                    .packed_qwen_instruction_handoff_policy()
                    .into(),
                packed_f16_dmd_vae_handoff_policy: engine
                    .policies
                    .packed_f16_dmd_vae_handoff_policy()
                    .into(),
                resident_resource_plan: engine.resident_resource_plan,
                low_vram_resource_plan: engine.low_vram_resource_plan,
                packed_f16_resource_plan: engine.packed_f16_resource_plan,
                packed_f16_qwen_embedding_plan: engine.packed_f16_qwen_embedding_plan,
                packed_f16_qwen_host_embedding: engine.last_packed_f16_qwen_host_embedding.clone(),
                packed_f16_qwen_instruction_handoff: engine
                    .last_packed_f16_qwen_instruction_handoff
                    .clone(),
                packed_f16_denoiser_lifecycle: engine.last_packed_f16_denoiser_lifecycle,
                packed_f16_dmd_vae_handoff: engine.last_packed_f16_dmd_vae_handoff.clone(),
                low_vram_denoiser_dtype_audit: engine.last_low_vram_denoiser_dtype_audit,
                dense_f32_materialized_stage_clones: engine
                    .last_dense_f32_materialized_stage_clones,
                artifact_traffic: engine.last_artifact_traffic,
                weight_traffic_contract: engine.policies.weight_traffic_contract().into(),
                // Packed F16 is storage compression followed by dense F32 execution, so it can
                // never make a quantized-execution claim. Edit's distinct runtime-Q8 route also
                // withholds that claim until measured kernel evidence exists.
                on_device_quantized_execution_claimed: false,
                prompt,
                dimensions,
                seed: output.seed,
                png_file_name,
                png_bytes: png.len() as u64,
                png_sha256,
                peak_wasm_linear_memory_bytes,
                timings: output.timings,
                provenance: output.provenance.clone(),
                artifacts_verified: output.provenance.artifacts_verified,
                numerical_parity_claimed: false,
            },
            png,
        })
    }

    /// Run only the exact 1.5K fixture's VAE reference encoder surface three times.
    ///
    /// The fixture container is authenticated once. Each repeat then constructs a fresh encoder
    /// from the exact verified stage on the same real WebGPU device and reuses the exact injected
    /// input and epsilon. This diagnostic never executes Qwen, the denoiser, DMD, or the decoder,
    /// and its report always keeps `numerical_parity_claimed` false.
    pub(crate) async fn vae_reference_no_surface(
        variant: BooguVariant,
        settings: crate::BooguAdapterSettings,
        fixture_base: RemoteBaseUrl,
    ) -> Result<BrowserBooguVaeReferenceDiagnosticReport, RuntimeError> {
        if variant != BooguVariant::Image01EditTurbo1k5 {
            return Err(execution_error(
                variant,
                "browser VAE reference diagnostic requires edit-turbo-1k5",
            ));
        }
        crate::boogu::boogu_browser_1k5_parity_descriptor(variant, settings.storage_profile)?;
        let profile = settings.storage_profile;
        if settings.integrity != burn_image::IntegrityPolicy::RequireSha256 {
            return Err(execution_error(
                variant,
                "browser VAE reference diagnostic requires SHA-256 verification",
            ));
        }
        settings
            .validate_concrete_cache_policy()
            .map_err(|error| execution_error(variant, error))?;
        if !matches!(
            &settings.artifact_source,
            burn_image::ArtifactSource::Remote { .. }
        ) {
            return Err(execution_error(
                variant,
                "browser VAE reference diagnostic requires a remote artifact base URL",
            ));
        }

        let BrowserNoSurfaceEngine {
            mut engine,
            _setup_guard,
            adapter,
            backend,
            limits,
            adapter_shader_f16,
            device_shader_f16,
        } = build_no_surface_parity_engine(settings).await?;
        validate_canonical_release_artifact_digest(
            variant,
            profile,
            engine.artifact_content_digest,
        )
        .map_err(|error| execution_error(variant, error))?;

        vae_reference_milestone("fixture-open-start");
        let fixture = BrowserParityFixture::open_vae_reference(fixture_base).await?;
        let source_png = fixture.source_png().await?;
        // The diagnostic does not consume decoder output, but full fixture authentication remains
        // exact and fail closed rather than silently weakening the qualification input contract.
        drop(fixture.output_png().await?);
        vae_reference_milestone("fixture-container-authentication-start");
        fixture.authenticate_all_tensors().await?;
        vae_reference_milestone("fixture-container-authentication-complete");

        engine
            .vae_reference_1k5(
                fixture,
                source_png,
                BrowserParityAdapterEvidence {
                    adapter,
                    backend,
                    limits,
                    adapter_shader_f16,
                    device_shader_f16,
                },
            )
            .await
    }

    /// Replay the one pinned exhaustive 1536-square fixture on real browser WebGPU.
    ///
    /// This is intentionally a separate, surface-free qualification route. Release identity,
    /// profile, remote transport, SHA-256, non-CPU adapter, device limits, exact request shape,
    /// exact noise injection, and every numerical gate remain fail closed.
    pub(crate) async fn parity_no_surface(
        variant: BooguVariant,
        settings: crate::BooguAdapterSettings,
        fixture_base: RemoteBaseUrl,
    ) -> Result<BrowserBooguParityReport, RuntimeError> {
        Self::parity_no_surface_with_residency(
            variant,
            settings,
            BrowserBooguResidencyPolicy::QualificationPerRequestF32DenoiserRetained,
            fixture_base,
        )
        .await
    }

    /// Replay the exact 1.5K fixture under an explicitly selected supported residency policy.
    pub(crate) async fn parity_no_surface_with_residency(
        variant: BooguVariant,
        settings: crate::BooguAdapterSettings,
        residency: BrowserBooguResidencyPolicy,
        fixture_base: RemoteBaseUrl,
    ) -> Result<BrowserBooguParityReport, RuntimeError> {
        if variant != BooguVariant::Image01EditTurbo1k5 {
            return Err(execution_error(
                variant,
                "browser 1.5K parity requires edit-turbo-1k5",
            ));
        }
        crate::boogu::boogu_browser_1k5_parity_descriptor(variant, settings.storage_profile)?;
        let profile = settings.storage_profile;
        if settings.integrity != burn_image::IntegrityPolicy::RequireSha256 {
            return Err(execution_error(
                variant,
                "browser 1.5K parity requires SHA-256 verification",
            ));
        }
        settings
            .validate_concrete_cache_policy()
            .map_err(|error| execution_error(variant, error))?;
        if !matches!(
            &settings.artifact_source,
            burn_image::ArtifactSource::Remote { .. }
        ) {
            return Err(execution_error(
                variant,
                "browser 1.5K parity requires a remote artifact base URL",
            ));
        }

        let BrowserNoSurfaceEngine {
            mut engine,
            _setup_guard,
            adapter,
            backend,
            limits,
            adapter_shader_f16,
            device_shader_f16,
        } = build_no_surface_parity_engine_with_residency(settings, residency).await?;
        validate_canonical_release_artifact_digest(
            variant,
            profile,
            engine.artifact_content_digest,
        )
        .map_err(|error| execution_error(variant, error))?;

        parity_milestone("fixture-open-start");
        let fixture = BrowserParityFixture::open(fixture_base).await?;
        let source_png = fixture.source_png().await?;
        let output_png = fixture.output_png().await?;
        parity_milestone("fixture-container-authentication-start");
        fixture.authenticate_all_tensors().await?;
        parity_milestone("fixture-container-authentication-complete");

        engine
            .parity_1k5(
                fixture,
                source_png,
                Sha256Digest::calculate(&output_png),
                BrowserParityAdapterEvidence {
                    adapter,
                    backend,
                    limits,
                    adapter_shader_f16,
                    device_shader_f16,
                },
            )
            .await
    }
}

struct BrowserParityAdapterEvidence {
    adapter: wgpu::AdapterInfo,
    backend: wgpu::Backend,
    limits: wgpu::Limits,
    adapter_shader_f16: bool,
    device_shader_f16: bool,
}

fn install_no_surface_device_diagnostics(setup: &burn_wgpu::WgpuSetup) {
    setup.device.set_device_lost_callback(|reason, message| {
        web_sys::console::error_1(
            &format!("BURN_IMAGE_WEBGPU_DEVICE_LOST reason={reason:?} message={message}").into(),
        );
    });
    setup.device.on_uncaptured_error(Arc::new(|error| {
        web_sys::console::error_1(&format!("BURN_IMAGE_WEBGPU_UNCAPTURED_ERROR {error}").into());
    }));
}

async fn no_surface_adapter_retry_delay(variant: BooguVariant) -> Result<(), RuntimeError> {
    use wasm_bindgen::JsValue;
    use wasm_bindgen_futures::JsFuture;

    let window = web_sys::window()
        .ok_or_else(|| execution_error(variant, "WebGPU adapter retry requires Window"))?;
    let promise = js_sys::Promise::new(&mut |resolve, reject| {
        if let Err(error) =
            window.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 250)
        {
            let _ = reject.call1(&JsValue::UNDEFINED, &error);
        }
    });
    JsFuture::from(promise).await.map(|_| ()).map_err(|error| {
        execution_error(
            variant,
            format!("WebGPU adapter retry timer failed: {error:?}"),
        )
    })
}

async fn init_no_surface_wgpu_setup(
    variant: BooguVariant,
) -> Result<(burn_wgpu::WgpuDevice, burn_wgpu::WgpuSetup), RuntimeError> {
    const MAX_ADAPTER_ATTEMPTS: usize = 8;

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let options = wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    };
    let mut selected_adapter = None;
    for attempt in 1..=MAX_ADAPTER_ATTEMPTS {
        match instance.request_adapter(&options).await {
            Ok(adapter) => {
                selected_adapter = Some(adapter);
                break;
            }
            Err(error) if attempt < MAX_ADAPTER_ATTEMPTS => {
                report_browser_runtime_preparing(format!(
                    "WebGPU high-performance adapter request {attempt}/{MAX_ADAPTER_ATTEMPTS} was unavailable ({error}); retrying the same hardware-only request"
                ));
                no_surface_adapter_retry_delay(variant).await?;
            }
            Err(error) => {
                return Err(execution_error(
                    variant,
                    format!(
                        "WebGPU high-performance adapter remained unavailable after {MAX_ADAPTER_ATTEMPTS} attempts: {error}"
                    ),
                ));
            }
        }
    }
    let adapter = selected_adapter.ok_or_else(|| {
        execution_error(
            variant,
            "WebGPU high-performance adapter selection ended without an adapter",
        )
    })?;
    let adapter_info = adapter.get_info();
    if adapter_info.device_type == wgpu::DeviceType::Cpu {
        return Err(execution_error(
            variant,
            "no-surface browser diagnostic refuses a CPU WebGPU adapter",
        ));
    }
    let required_features = adapter
        .features()
        .difference(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS);
    let required_limits = adapter.limits();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("burn-image-no-surface-webgpu"),
            required_features,
            required_limits,
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        })
        .await
        .map_err(|error| {
            execution_error(
                variant,
                format!(
                    "requesting the no-surface WebGPU device on {adapter_info:?} failed: {error}"
                ),
            )
        })?;
    let setup = burn_wgpu::WgpuSetup {
        instance,
        adapter,
        device,
        queue,
        backend: adapter_info.backend,
    };
    let cube_device = burn_wgpu::init_device(setup.clone(), burn_wgpu::RuntimeOptions::default());
    Ok((cube_device, setup))
}

async fn build_no_surface_engine(
    variant: BooguVariant,
    settings: crate::BooguAdapterSettings,
    policy: BrowserNoSurfacePolicy,
) -> Result<BrowserNoSurfaceEngine, RuntimeError> {
    report_browser_runtime_preparing(
        "Requesting the WebGPU adapter, device, queue, and CubeCL no-surface runtime",
    );
    let (device, setup) = init_no_surface_wgpu_setup(variant).await?;
    report_browser_runtime_preparing(
        "WebGPU adapter, device, queue, and CubeCL runtime initialized for the no-surface diagnostic",
    );
    install_no_surface_device_diagnostics(&setup);
    let adapter = setup.adapter.get_info();
    let limits = setup.device.limits();
    let adapter_shader_f16 = setup
        .adapter
        .features()
        .contains(wgpu::Features::SHADER_F16);
    let device_shader_f16 = setup.device.features().contains(wgpu::Features::SHADER_F16);
    if adapter.device_type == wgpu::DeviceType::Cpu {
        return Err(execution_error(
            variant,
            "no-surface browser diagnostic refuses a CPU WebGPU adapter",
        ));
    }
    if policy == BrowserNoSurfacePolicy::PreserveQwenF16
        && (!adapter_shader_f16 || !device_shader_f16)
    {
        return Err(execution_error(
            variant,
            format!(
                "preserved-F16 browser probe requires WebGPU shader-f16; adapter={adapter_shader_f16}, device={device_shader_f16}"
            ),
        ));
    }
    let inputs = BrowserBooguFactory::validate_context(
        variant,
        BooguFactoryContext {
            device,
            execution: WgpuExecutionKind::BrowserWebGpu,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            max_buffer_size: limits.max_buffer_size,
            allocation_device: None,
            settings,
            releases: vec![BooguReleaseIdentity::canonical(variant)],
        },
    )?;
    report_browser_runtime_preparing(
        "No-surface browser context admitted; resolving the exact execution policy",
    );
    let policies = match policy {
        BrowserNoSurfacePolicy::CompatibleF32 => {
            BrowserExecutionPolicies::stage_probe_f32(&inputs.settings)
        }
        BrowserNoSurfacePolicy::PreserveQwenF16 => {
            BrowserExecutionPolicies::preserve_qwen_f16(&inputs.settings)
        }
        BrowserNoSurfacePolicy::ResidentPackedF16 => {
            BrowserExecutionPolicies::resident_packed_f16(&inputs.settings)
                .map_err(|error| execution_error(variant, error))?
        }
        BrowserNoSurfacePolicy::ResidentPackedQ4s => {
            BrowserExecutionPolicies::resident_packed_q4s(variant, &inputs.settings)
                .map_err(|error| execution_error(variant, error))?
        }
        BrowserNoSurfacePolicy::ResidentDenseF32 => {
            BrowserExecutionPolicies::resident_dense_f32(&inputs.settings)
                .map_err(|error| execution_error(variant, error))?
        }
        BrowserNoSurfacePolicy::LowVramRuntimeQ8Denoiser => {
            BrowserExecutionPolicies::low_vram_runtime_q8_denoiser(variant, &inputs.settings)
                .map_err(|error| execution_error(variant, error))?
        }
        BrowserNoSurfacePolicy::LowVramPreloadedPackedF16Denoiser => {
            BrowserExecutionPolicies::low_vram_preloaded_packed_f16_denoiser(
                variant,
                &inputs.settings,
            )
            .map_err(|error| execution_error(variant, error))?
        }
    };
    report_browser_runtime_preparing(
        "No-surface execution policy admitted; constructing verified model stage sources",
    );
    let engine = BrowserBooguEngine::build(
        inputs.identity,
        inputs.base_url,
        inputs.settings,
        policies,
        inputs.device,
        inputs.allocation_device,
        inputs.applied_buffer_limits,
    )
    .await?;
    report_browser_runtime_preparing(
        "Verified no-surface model engine constructed; starting the requested diagnostic",
    );
    Ok(BrowserNoSurfaceEngine {
        engine,
        _setup_guard: setup.clone(),
        adapter,
        backend: setup.backend,
        limits,
        adapter_shader_f16,
        device_shader_f16,
    })
}

async fn build_no_surface_parity_engine(
    settings: crate::BooguAdapterSettings,
) -> Result<BrowserNoSurfaceEngine, RuntimeError> {
    build_no_surface_parity_engine_with_residency(
        settings,
        BrowserBooguResidencyPolicy::QualificationPerRequestF32DenoiserRetained,
    )
    .await
}

async fn build_no_surface_parity_engine_with_residency(
    settings: crate::BooguAdapterSettings,
    residency: BrowserBooguResidencyPolicy,
) -> Result<BrowserNoSurfaceEngine, RuntimeError> {
    let variant = BooguVariant::Image01EditTurbo1k5;
    let (device, setup) = init_no_surface_wgpu_setup(variant).await?;
    install_no_surface_device_diagnostics(&setup);
    let adapter = setup.adapter.get_info();
    let limits = setup.device.limits();
    let adapter_shader_f16 = setup
        .adapter
        .features()
        .contains(wgpu::Features::SHADER_F16);
    let device_shader_f16 = setup.device.features().contains(wgpu::Features::SHADER_F16);
    if adapter.device_type == wgpu::DeviceType::Cpu {
        return Err(execution_error(
            variant,
            "browser 1.5K parity refuses a CPU WebGPU adapter/device",
        ));
    }
    crate::boogu::validate_browser_1k5_buffer_limits(
        limits.max_storage_buffer_binding_size,
        limits.max_buffer_size,
    )?;
    let base_url = match &settings.artifact_source {
        burn_image::ArtifactSource::Remote { base_url } => base_url.clone(),
        burn_image::ArtifactSource::LocalDirectory { .. } => {
            return Err(execution_error(
                variant,
                "browser 1.5K parity requires a remote artifact base URL",
            ));
        }
    };
    let identity = BooguReleaseIdentity::canonical(variant);
    let policies = match residency {
        BrowserBooguResidencyPolicy::QualificationPerRequestF32DenoiserRetained => {
            Ok(BrowserExecutionPolicies::exact_1k5_parity(&settings))
        }
        BrowserBooguResidencyPolicy::HighVramResidentPackedF16 => {
            BrowserExecutionPolicies::resident_packed_f16(&settings)
        }
        BrowserBooguResidencyPolicy::ResidentPackedQ4s => {
            BrowserExecutionPolicies::resident_packed_q4s(variant, &settings)
        }
        BrowserBooguResidencyPolicy::HighVramResidentDenseF32 => Err(
            "exact browser parity uses the qualification-per-request F32 denoiser-retained policy, not production all-stage residency",
        ),
        BrowserBooguResidencyPolicy::LowVramRuntimeQ8Denoiser => {
            BrowserExecutionPolicies::exact_1k5_low_vram_parity(&settings)
        }
        BrowserBooguResidencyPolicy::LowVramPreloadedPackedF16Denoiser => {
            Err("the Turbo-only preloaded packed-F16 dense-F32-per-stage policy is not an Edit-Turbo 1.5K parity policy")
        }
    }
    .map_err(|error| execution_error(variant, error))?;
    let engine = BrowserBooguEngine::build(
        identity,
        base_url,
        settings,
        policies,
        device,
        None,
        BrowserAppliedBufferLimits {
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            max_buffer_size: limits.max_buffer_size,
        },
    )
    .await?;
    Ok(BrowserNoSurfaceEngine {
        engine,
        _setup_guard: setup.clone(),
        adapter,
        backend: setup.backend,
        limits,
        adapter_shader_f16,
        device_shader_f16,
    })
}

impl BooguRuntimeFactory for BrowserBooguFactory {
    fn initialization_variant(&self) -> Option<BooguVariant> {
        Some(self.variant)
    }

    fn start(&mut self, context: BooguFactoryContext) -> Result<(), RuntimeError> {
        if self.started {
            return Err(execution_error(
                self.variant,
                "browser factory was already started",
            ));
        }
        let inputs = Self::validate_context(self.variant, context)?;
        if inputs.allocation_device.is_none() {
            return Err(execution_error(
                self.variant,
                "ordinary browser factory requires the exact shared Bevy WGPU device/queue for its fail-fast memory preflight",
            ));
        }
        let residency = self.residency;
        report_browser_runtime_preparing(
            "Shared WebGPU device ready; verifying the sealed model manifest",
        );

        let slot = Arc::new(Mutex::new(None));
        let result_slot = slot.clone();
        spawn_local(async move {
            let policies = match residency {
                BrowserBooguResidencyPolicy::HighVramResidentPackedF16 => {
                    BrowserExecutionPolicies::resident_packed_f16(&inputs.settings)
                        .map_err(|error| execution_error(inputs.identity.variant, error))
                }
                BrowserBooguResidencyPolicy::ResidentPackedQ4s => {
                    BrowserExecutionPolicies::resident_packed_q4s(
                        inputs.identity.variant,
                        &inputs.settings,
                    )
                    .map_err(|error| execution_error(inputs.identity.variant, error))
                }
                BrowserBooguResidencyPolicy::HighVramResidentDenseF32 => {
                    BrowserExecutionPolicies::resident_dense_f32(&inputs.settings)
                        .map_err(|error| execution_error(inputs.identity.variant, error))
                }
                BrowserBooguResidencyPolicy::QualificationPerRequestF32DenoiserRetained => {
                    Err(execution_error(
                        inputs.identity.variant,
                        "the per-request F32 denoiser-retained policy is reserved for exact-fixture qualification",
                    ))
                }
                BrowserBooguResidencyPolicy::LowVramRuntimeQ8Denoiser => {
                    BrowserExecutionPolicies::low_vram_runtime_q8_denoiser(
                        inputs.identity.variant,
                        &inputs.settings,
                    )
                    .map_err(|error| execution_error(inputs.identity.variant, error))
                }
                BrowserBooguResidencyPolicy::LowVramPreloadedPackedF16Denoiser => {
                    BrowserExecutionPolicies::low_vram_preloaded_packed_f16_denoiser(
                        inputs.identity.variant,
                        &inputs.settings,
                    )
                    .map_err(|error| execution_error(inputs.identity.variant, error))
                }
            };
            let result = match policies {
                Ok(policies) => {
                    BrowserBooguEngine::build(
                        inputs.identity,
                        inputs.base_url,
                        inputs.settings,
                        policies.for_ordinary_browser_factory(
                            crate::boogu::browser_surface_inference_gate_requested(),
                        ),
                        inputs.device,
                        inputs.allocation_device,
                        inputs.applied_buffer_limits,
                    )
                    .await
                }
                Err(error) => Err(error),
            };
            match &result {
                Ok(engine) => report_browser_runtime_ready(
                    engine.identity.variant,
                    engine.policies.eager_preload,
                    engine.artifact_control.transfer_progress(),
                    engine.policies.qwen_text_layer_allocation_policy(),
                    engine
                        .policies
                        .qwen_text_block_load_synchronization_policy(),
                    engine.policies.qwen_text_layer_submission_policy(),
                ),
                Err(error) => report_browser_runtime_failure(error.to_string()),
            }
            *result_slot
                .lock()
                .expect("browser factory result mutex poisoned") = Some(result);
        });
        self.pending = Some(slot);
        self.started = true;
        Ok(())
    }

    fn poll(&mut self) -> Result<Option<Box<dyn BooguRuntime>>, RuntimeError> {
        let Some(slot) = &self.pending else {
            return Ok(None);
        };
        let result = slot
            .lock()
            .expect("browser factory result mutex poisoned")
            .take();
        match result {
            None => Ok(None),
            Some(Ok(engine)) => {
                self.pending = None;
                Ok(Some(Box::new(BrowserBooguRuntime::new(engine))))
            }
            Some(Err(error)) => {
                self.pending = None;
                Err(error)
            }
        }
    }

    fn take_initialization_progress(&mut self) -> Option<String> {
        take_browser_factory_progress()
    }
}

struct BrowserStageProbe {
    stage: String,
    bytes: u64,
    dtype: String,
    elements: usize,
    finite_elements: usize,
    verified_objects: usize,
}

struct BrowserArtifactComposition {
    pipeline_manifest: ArtifactManifest,
    qwen_manifest: ArtifactManifest,
    vae_manifest: ArtifactManifest,
    pipeline_base_url: RemoteBaseUrl,
    qwen_base_url: RemoteBaseUrl,
    vae_base_url: RemoteBaseUrl,
    standalone: bool,
}

impl BrowserArtifactComposition {
    async fn resolve(
        variant: BooguVariant,
        pipeline_manifest: ArtifactManifest,
        pipeline_base_url: RemoteBaseUrl,
        stream_config: ArtifactStreamConfig,
    ) -> Result<Self, RuntimeError> {
        if pipeline_manifest.dependencies.is_empty() {
            if pipeline_manifest.profile.as_str() == "q4s-block-up-to128-f32" {
                return Err(execution_error(
                    variant,
                    "packed-Q4S browser releases require shared Qwen and VAE component dependencies",
                ));
            }
            if pipeline_manifest.schema_version != burn_image::ARTIFACT_MANIFEST_SCHEMA_V1 {
                return Err(execution_error(
                    variant,
                    "a schema-v2 browser pipeline manifest must seal qwen and vae dependencies",
                ));
            }
            return Ok(Self {
                qwen_manifest: pipeline_manifest.clone(),
                vae_manifest: pipeline_manifest.clone(),
                pipeline_manifest,
                qwen_base_url: pipeline_base_url.clone(),
                vae_base_url: pipeline_base_url.clone(),
                pipeline_base_url,
                standalone: true,
            });
        }
        if pipeline_manifest.schema_version != burn_image::ARTIFACT_MANIFEST_SCHEMA_V2
            || pipeline_manifest.dependencies.len() != 2
            || pipeline_manifest
                .metadata
                .get("component_dependency_count")
                .map(String::as_str)
                != Some("2")
        {
            return Err(execution_error(
                variant,
                "a composed browser manifest must be schema-v2 with exactly qwen and vae dependencies",
            ));
        }
        let qwen = browser_dependency(&pipeline_manifest, "qwen", variant)?;
        let vae = browser_dependency(&pipeline_manifest, "vae", variant)?;
        let (qwen_base_url, qwen_manifest) =
            fetch_browser_dependency_manifest(variant, &pipeline_base_url, qwen, stream_config)
                .await?;
        let (vae_base_url, vae_manifest) =
            fetch_browser_dependency_manifest(variant, &pipeline_base_url, vae, stream_config)
                .await?;
        Ok(Self {
            pipeline_manifest,
            qwen_manifest,
            vae_manifest,
            pipeline_base_url,
            qwen_base_url,
            vae_base_url,
            standalone: false,
        })
    }
}

fn browser_dependency<'a>(
    manifest: &'a ArtifactManifest,
    role: &str,
    variant: BooguVariant,
) -> Result<&'a ArtifactDependency, RuntimeError> {
    manifest
        .dependencies
        .iter()
        .find(|dependency| dependency.role.as_str() == role)
        .ok_or_else(|| {
            execution_error(
                variant,
                format!("composed browser manifest omits required {role} dependency"),
            )
        })
}

async fn fetch_browser_dependency_manifest(
    variant: BooguVariant,
    pipeline_base_url: &RemoteBaseUrl,
    dependency: &ArtifactDependency,
    stream_config: ArtifactStreamConfig,
) -> Result<(RemoteBaseUrl, ArtifactManifest), RuntimeError> {
    let base_url = sibling_bundle_base_url(pipeline_base_url, &dependency.bundle)
        .map_err(|error| execution_error(variant, error))?;
    let path =
        ArtifactPath::new("manifest.json").map_err(|error| execution_error(variant, error))?;
    let bytes =
        fetch_browser_bounded_file(&base_url, path, MAX_BROWSER_MANIFEST_BYTES, stream_config)
            .await
            .map_err(|error| execution_error(variant, error))?;
    let manifest: ArtifactManifest =
        serde_json::from_slice(&bytes).map_err(|error| execution_error(variant, error))?;
    dependency
        .validate_resolved_manifest(&manifest)
        .map_err(|error| execution_error(variant, error))?;
    if !manifest.dependencies.is_empty() {
        return Err(execution_error(
            variant,
            format!(
                "browser model component {} must be a dependency leaf",
                manifest.bundle
            ),
        ));
    }
    report_browser_manifest_verified(&manifest);
    Ok((base_url, manifest))
}

#[cfg(test)]
fn manifest_weight_artifacts(
    manifest: &ArtifactManifest,
    qualify_bundle: bool,
) -> BTreeMap<String, u64> {
    manifest
        .files
        .iter()
        .filter(|file| matches!(file.role, burn_image::ArtifactFileRole::Weights))
        .map(|file| {
            let path = if qualify_bundle {
                format!("{}/{}", manifest.bundle, file.path)
            } else {
                file.path.to_string()
            };
            (path, file.size)
        })
        .collect()
}

fn active_manifest_weight_artifacts(
    manifest: &ArtifactManifest,
    qualify_bundle: bool,
    variant: BooguVariant,
) -> BTreeMap<String, u64> {
    manifest
        .files
        .iter()
        .filter(|file| matches!(file.role, burn_image::ArtifactFileRole::Weights))
        .filter(|file| {
            browser_resident_artifact_required(
                variant,
                file.component.as_ref().map(|component| component.as_str()),
            )
        })
        .map(|file| {
            let path = if qualify_bundle {
                format!("{}/{}", manifest.bundle, file.path)
            } else {
                file.path.to_string()
            };
            (path, file.size)
        })
        .collect()
}

struct BrowserBooguEngine {
    identity: BooguReleaseIdentity,
    artifact_content_digest: burn_image::Sha256Digest,
    numeric_format: burn_image::NumericFormat,
    qwen_config: Qwen3VlConfig,
    qwen: StreamingQwen3Vl<BrowserBackend, BrowserQwenSource>,
    vae: BrowserVaeSource,
    denoiser: StreamingBooguDenoiser<BrowserBackend, BrowserDenoiserSource>,
    processor: Qwen3VlProcessor<HfTokenizer>,
    image_processor: Qwen3VlImageProcessor,
    policies: BrowserExecutionPolicies,
    dtypes: BooguRuntimeDTypes,
    device: burn_wgpu::WgpuDevice,
    applied_buffer_limits: BrowserAppliedBufferLimits,
    resident_resource_plan: Option<BrowserResidentResourcePlan>,
    low_vram_resource_plan: Option<BrowserLowVramResourcePlan>,
    packed_f16_resource_plan: Option<BrowserPackedF16ResourcePlan>,
    packed_f16_qwen_embedding_plan: Option<BrowserPackedF16QwenEmbeddingPlan>,
    last_packed_f16_qwen_host_embedding: Option<HostRoutedEmbeddingReport>,
    last_packed_f16_qwen_instruction_handoff: Option<BrowserPackedF16QwenInstructionHandoffReport>,
    last_packed_f16_denoiser_lifecycle: Option<BrowserPackedF16DenoiserLifecycleReport>,
    last_packed_f16_dmd_vae_handoff: Option<BrowserPackedF16DmdVaeHandoffReport>,
    last_low_vram_denoiser_dtype_audit: Option<BrowserLowVramDenoiserDTypeAudit>,
    last_dense_f32_materialized_stage_clones: usize,
    last_artifact_traffic: BrowserArtifactTrafficReport,
    artifact_control: BrowserArtifactControl,
    expected_weight_artifacts: BTreeMap<String, u64>,
    expected_vae_encoder_weight_artifacts: BTreeMap<String, u64>,
    verified_runtime_metadata: BTreeSet<String>,
}

impl BrowserBooguEngine {
    async fn build(
        identity: BooguReleaseIdentity,
        base_url: burn_image::RemoteBaseUrl,
        settings: crate::BooguAdapterSettings,
        policies: BrowserExecutionPolicies,
        device: burn_wgpu::WgpuDevice,
        allocation_device: Option<crate::backend::SharedWgpuAllocationDevice>,
        applied_buffer_limits: BrowserAppliedBufferLimits,
    ) -> Result<Self, RuntimeError> {
        let variant = identity.variant;
        if !policies.packed_allocator_policy_is_exact() {
            return Err(execution_error(
                variant,
                "browser execution policy has an inconsistent Qwen allocation, embedding, synchronization, or handoff contract",
            ));
        }
        let stream_config = ArtifactStreamConfig::default();
        let manifest_path =
            ArtifactPath::new("manifest.json").map_err(|error| execution_error(variant, error))?;
        let manifest_bytes = fetch_browser_bounded_file(
            &base_url,
            manifest_path,
            MAX_BROWSER_MANIFEST_BYTES,
            stream_config,
        )
        .await
        .map_err(|error| execution_error(variant, error))?;
        let manifest: ArtifactManifest = serde_json::from_slice(&manifest_bytes)
            .map_err(|error| execution_error(variant, error))?;
        manifest
            .validate_sealed()
            .map_err(|error| execution_error(variant, error))?;
        validate_browser_manifest_bundle_identity(
            variant,
            settings.storage_profile,
            manifest.bundle.as_str(),
        )?;
        let artifact_content_digest = manifest.content_digest.ok_or_else(|| {
            execution_error(variant, "sealed browser manifest omits its content digest")
        })?;
        if browser_source_requires_canonical_digest(variant, settings.storage_profile, &base_url) {
            validate_canonical_release_artifact_digest(
                variant,
                settings.storage_profile,
                artifact_content_digest,
            )
            .map_err(|error| execution_error(variant, error))?;
        }
        if policies.residency.is_low_vram() {
            // The memory audit and runtime quantization quality evidence are bound to the exact
            // canonical production manifest, even when a caller mirrors it behind artifacts=.
            validate_canonical_release_artifact_digest(
                variant,
                settings.storage_profile,
                artifact_content_digest,
            )
            .map_err(|error| execution_error(variant, error))?;
        }
        report_browser_manifest_verified(&manifest);
        let composition =
            BrowserArtifactComposition::resolve(variant, manifest, base_url, stream_config).await?;
        let expected_weight_artifacts = if composition.standalone {
            active_manifest_weight_artifacts(&composition.pipeline_manifest, false, variant)
        } else {
            [
                &composition.pipeline_manifest,
                &composition.qwen_manifest,
                &composition.vae_manifest,
            ]
            .into_iter()
            .flat_map(|manifest| active_manifest_weight_artifacts(manifest, true, variant))
            .collect()
        };
        let resident_resource_manifest = if policies.eager_preload {
            let mut resource_manifest = composition.pipeline_manifest.clone();
            if !composition.standalone {
                resource_manifest
                    .files
                    .extend(composition.qwen_manifest.files.iter().cloned());
                resource_manifest
                    .files
                    .extend(composition.vae_manifest.files.iter().cloned());
            }
            Some(resource_manifest)
        } else {
            None
        };
        let expected_vae_encoder_weight_artifacts = composition
            .vae_manifest
            .files
            .iter()
            .filter(|file| {
                matches!(file.role, burn_image::ArtifactFileRole::Weights)
                    && file.component.as_ref().map(|value| value.as_str())
                        == Some("flux-vae-encoder")
            })
            .map(|file| {
                let path = if composition.standalone {
                    file.path.to_string()
                } else {
                    format!("{}/{}", composition.vae_manifest.bundle, file.path)
                };
                (path, file.size)
            })
            .collect::<BTreeMap<_, _>>();
        if expected_vae_encoder_weight_artifacts.is_empty() {
            return Err(execution_error(
                variant,
                "sealed browser manifest omits the FLUX VAE encoder weight stage",
            ));
        }

        let bootstrap_control = BrowserArtifactControl::default();
        bootstrap_control.set_transfer_phase("Model setup");
        let bootstrap_progress_control = bootstrap_control.clone();
        bootstrap_control.set_observer(Some(Arc::new(move |event| {
            let progress = browser_artifact_progress(
                RunId(0),
                event,
                bootstrap_progress_control.transfer_progress(),
            );
            dispatch_browser_progress(&progress);
        })));
        let make_reader = |base_url, bundle| {
            let reader = if composition.standalone {
                BrowserStageShardReader::with_control(
                    base_url,
                    stream_config,
                    bootstrap_control.clone(),
                )
            } else {
                BrowserStageShardReader::for_bundle(
                    base_url,
                    bundle,
                    stream_config,
                    bootstrap_control.clone(),
                )
            };
            if policies.require_persistent_range_cache {
                reader.with_required_range_cache()
            } else {
                reader
            }
        };
        let reader = make_reader(
            composition.pipeline_base_url.clone(),
            composition.pipeline_manifest.bundle.clone(),
        )
        .with_manifest_transport_layout(&composition.pipeline_manifest)
        .await
        .map_err(|error| execution_error(variant, error))?;
        let mut qwen_reader = make_reader(
            composition.qwen_base_url.clone(),
            composition.qwen_manifest.bundle.clone(),
        )
        .with_manifest_transport_layout(&composition.qwen_manifest)
        .await
        .map_err(|error| execution_error(variant, error))?;
        let mut vae_reader = make_reader(
            composition.vae_base_url.clone(),
            composition.vae_manifest.bundle.clone(),
        )
        .with_manifest_transport_layout(&composition.vae_manifest)
        .await
        .map_err(|error| execution_error(variant, error))?;
        let active_transfer_objects = expected_weight_artifacts
            .keys()
            .map(|path| {
                ArtifactPath::new(path.clone()).map_err(|error| execution_error(variant, error))
            })
            .collect::<Result<BTreeSet<_>, _>>()?;
        bootstrap_control
            .retain_transfer_logical_objects(&active_transfer_objects)
            .map_err(|error| execution_error(variant, error))?;
        if variant == BooguVariant::Image01Turbo
            && settings.storage_profile == BooguStorageProfile::F16QwenVisionF32
            && !composition.standalone
        {
            let active_plan = bootstrap_control.transfer_progress().ok_or_else(|| {
                execution_error(variant, "canonical Turbo active transport plan is empty")
            })?;
            validate_browser_turbo_active_transfer_plan(&active_plan)
                .map_err(|error| execution_error(variant, error))?;
        }
        if policies.require_persistent_range_cache {
            let mut cache_plan = BrowserPersistentCachePlan::default();
            reader
                .extend_persistent_cache_plan(
                    &composition.pipeline_manifest,
                    &active_transfer_objects,
                    &mut cache_plan,
                )
                .map_err(|error| execution_error(variant, error))?;
            qwen_reader
                .extend_persistent_cache_plan(
                    &composition.qwen_manifest,
                    &active_transfer_objects,
                    &mut cache_plan,
                )
                .map_err(|error| execution_error(variant, error))?;
            vae_reader
                .extend_persistent_cache_plan(
                    &composition.vae_manifest,
                    &active_transfer_objects,
                    &mut cache_plan,
                )
                .map_err(|error| execution_error(variant, error))?;
            let transfer_plan = bootstrap_control
                .transfer_progress()
                .ok_or_else(|| execution_error(variant, "selected model transfer plan is empty"))?;
            let cache_shape = (cache_plan.entry_count(), cache_plan.total_bytes());
            let transfer_shape = (
                transfer_plan.bounded_ranges_total,
                transfer_plan.total_bytes,
            );
            if cache_shape != transfer_shape {
                return Err(execution_error(
                    variant,
                    format!(
                        "selected model cache plan {cache_shape:?} differs from transfer plan {transfer_shape:?}"
                    ),
                ));
            }
            let cache = preflight_browser_persistent_cache(&cache_plan)
                .await
                .map_err(|error| execution_error(variant, error))?;
            let persistence = if cache.persistent_storage_granted {
                "persistent"
            } else {
                "best effort"
            };
            let eviction = (cache.evicted_unselected_entries > 0).then(|| {
                format!(
                    "; evicted {} unselected entries and reclaimed {}",
                    cache.evicted_unselected_entries,
                    format_transfer_bytes(cache.reclaimed_storage_bytes),
                )
            });
            set_browser_factory_progress(format!(
                "Selected-model cache: {}/{} exact keys present; {} to store; {} free ({persistence}){}",
                cache.cached_entries,
                cache.total_entries,
                format_transfer_bytes(cache.missing_bytes),
                format_transfer_bytes(cache.storage_available_bytes),
                eviction.as_deref().unwrap_or_default(),
            ));
            bootstrap_control.set_transfer_phase(if cache.missing_entries == 0 {
                "Loading selected model from persistent cache"
            } else {
                "Downloading selected model into persistent cache"
            });
        }
        let qwen_config_bytes = read_manifest_file(
            &mut qwen_reader,
            &composition.qwen_manifest,
            "metadata/source/mllm/config.json",
            variant,
        )
        .await?;
        let vae_config_bytes = read_manifest_file(
            &mut vae_reader,
            &composition.vae_manifest,
            "metadata/source/vae/config.json",
            variant,
        )
        .await?;
        let tokenizer_bytes = read_manifest_file(
            &mut qwen_reader,
            &composition.qwen_manifest,
            "metadata/source/mllm/tokenizer.json",
            variant,
        )
        .await?;
        let image_processor_bytes = read_manifest_file(
            &mut qwen_reader,
            &composition.qwen_manifest,
            "metadata/source/mllm/preprocessor_config.json",
            variant,
        )
        .await?;
        let verified_runtime_metadata = if composition.standalone {
            browser_required_runtime_metadata_paths()
        } else {
            BTreeSet::from([
                format!(
                    "{}/metadata/source/mllm/config.json",
                    composition.qwen_manifest.bundle
                ),
                format!(
                    "{}/metadata/source/vae/config.json",
                    composition.vae_manifest.bundle
                ),
                format!(
                    "{}/metadata/source/mllm/tokenizer.json",
                    composition.qwen_manifest.bundle
                ),
                format!(
                    "{}/metadata/source/mllm/preprocessor_config.json",
                    composition.qwen_manifest.bundle
                ),
            ])
        };

        let qwen_config = Qwen3VlConfig::from_json(utf8(&qwen_config_bytes, variant)?)
            .map_err(|error| execution_error(variant, error))?;
        let mut vae_config =
            AutoencoderKlConfig::from_diffusers_json(utf8(&vae_config_bytes, variant)?)
                .map_err(|error| execution_error(variant, error))?;
        if variant == BooguVariant::Image01EditTurbo1k5 {
            vae_config.attention_query_chunk_size = BROWSER_1K5_VAE_QUERY_CHUNK_SIZE;
        }
        let denoiser_config = BooguConfig::default();
        let inventory = BooguArtifactInventory::new(&qwen_config, &denoiser_config, &vae_config)
            .map_err(|error| execution_error(variant, error))?;
        let profile = settings.storage_profile;
        let dtypes = policies.execution_dtypes(profile);
        let synchronizer = BrowserAsyncSynchronizer::new(&device);
        let resident_resource_plan = resident_resource_manifest
            .as_ref()
            .map(|manifest| {
                validate_browser_resident_resource_plan(
                    variant,
                    manifest,
                    &inventory,
                    profile,
                    policies.denoiser_float,
                )
            })
            .transpose()?;

        let (verified_qwen_source, qwen_plan) = if composition.standalone {
            let source = VerifiedAsyncBurnpackQwenStageSource::new_auto(
                &identity,
                composition.pipeline_manifest.clone(),
                inventory.clone(),
                qwen_config.clone(),
                profile,
                device.clone(),
                qwen_reader,
            )
            .await
            .map_err(|error| execution_error(variant, error))?
            .with_float_load_policy(policies.qwen_float)
            .with_quantized_load_policy(policies.qwen_quantized);
            let plan = source.plan().clone();
            (BrowserVerifiedQwenSource::Standalone(source), plan)
        } else {
            let contract = Qwen3VlComponentContract::released_base(
                composition.qwen_manifest.clone(),
                qwen_config.clone(),
            )
            .map_err(|error| execution_error(variant, error))?;
            let plan = contract.plan().clone();
            let source =
                VerifiedAsyncBurnpackQwen3VlStageSource::new(contract, device.clone(), qwen_reader)
                    .with_float_policy(policies.qwen_float.qwen_artifact_policy());
            (BrowserVerifiedQwenSource::Component(source), plan)
        };
        let low_vram_resource_plan =
            if policies.residency.is_low_vram() && !policies.uses_packed_f16_denoiser_source() {
                Some(validate_browser_low_vram_resource_plan(
                    variant,
                    settings.storage_profile,
                    &qwen_plan,
                    &inventory,
                    policies.denoiser_retaining_wrapper_adapter,
                )?)
            } else {
                None
            };
        let packed_f16_resource_plan = if policies.uses_packed_f16_denoiser_source() {
            Some(validate_browser_packed_f16_resource_plan(
                variant,
                settings.storage_profile,
            )?)
        } else {
            None
        };
        if packed_f16_resource_plan.is_some() != policies.uses_packed_f16_denoiser_source() {
            return Err(execution_error(
                variant,
                "browser packed-F16 plan presence differs from the selected denoiser source",
            ));
        }
        let vram_preflight = resident_resource_plan
            .map(|plan| {
                (
                    plan.weight_storage_policy,
                    plan.conservative_planned_device_bytes,
                )
            })
            .or_else(|| {
                low_vram_resource_plan.map(|plan| {
                    (
                        "low-vram-runtime-q8",
                        plan.conservative_planned_device_bytes,
                    )
                })
            })
            .or_else(|| {
                packed_f16_resource_plan.map(|plan| {
                    (
                        "preloaded-packed-f16-dense-f32-per-stage",
                        plan.conservative_planned_device_bytes,
                    )
                })
            });
        match (allocation_device.as_ref(), vram_preflight) {
            (Some(allocation_device), Some((policy, required_device_bytes))) => {
                run_browser_vram_preflight(
                    variant,
                    policy,
                    required_device_bytes,
                    applied_buffer_limits.max_buffer_size,
                    allocation_device,
                )
                .await?;
            }
            (Some(_), None) => {
                return Err(execution_error(
                    variant,
                    "ordinary browser execution policy omits a GPU memory preflight plan",
                ));
            }
            // Surface-free diagnostics own a separate device and retain their existing explicit
            // hardware/memory qualification contracts. The ordinary page always supplies the
            // exact shared Bevy device and queue above.
            (None, _) => {}
        }
        let packed_f16_qwen_embedding_plan = if policies.uses_host_routed_qwen_embedding() {
            Some(validate_browser_packed_f16_qwen_embedding_plan(
                variant,
                &composition.qwen_manifest,
                &qwen_plan,
            )?)
        } else {
            None
        };
        let qwen_source = BrowserAsyncStageSource::new(verified_qwen_source, synchronizer.clone());
        let qwen_source = if policies.retain_qwen_stages {
            RetainingAsyncQwen3VlStageSource::new(qwen_source)
        } else {
            RetainingAsyncQwen3VlStageSource::passthrough(qwen_source)
        };
        let qwen_source = if policies.defer_retained_qwen_synchronization {
            qwen_source.with_synchronization_policy(AsyncRetainingSynchronizationPolicy::Deferred)
        } else {
            qwen_source
        };
        let mut qwen = StreamingQwen3Vl::new(qwen_plan, qwen_source)
            .with_release_unused_memory_after_stage(policies.release_unused_qwen_memory_after_stage)
            .with_embedding_execution_policy(policies.qwen_embedding_execution)
            .with_text_layer_allocation_policy(policies.qwen_text_layer_allocation)
            .with_text_block_load_synchronization_policy(
                policies.qwen_text_block_load_synchronization,
            );
        if qwen.releases_unused_memory_after_stage()
            != policies.release_unused_qwen_memory_after_stage
            || qwen.embedding_execution_policy() != policies.qwen_embedding_execution
            || qwen.text_layer_allocation_policy() != policies.qwen_text_layer_allocation
            || qwen.text_block_load_synchronization_policy()
                != policies.qwen_text_block_load_synchronization
        {
            return Err(execution_error(
                variant,
                "streaming Qwen allocator/embedding/synchronization provenance differs from the admitted browser policy",
            ));
        }
        if variant == BooguVariant::Image01EditTurbo1k5 {
            qwen.set_query_chunk_size(BROWSER_1K5_QWEN_QUERY_CHUNK_SIZE);
        }
        let verified_vae_source = if composition.standalone {
            BrowserVerifiedVaeSource::Standalone(
                VerifiedAsyncBurnpackVaeStageSource::new(
                    &identity,
                    composition.pipeline_manifest.clone(),
                    inventory.clone(),
                    vae_config,
                    profile,
                    policies.vae_float,
                    device.clone(),
                    vae_reader,
                )
                .await
                .map_err(|error| execution_error(variant, error))?,
            )
        } else {
            let contract =
                FluxVaeComponentContract::new(composition.vae_manifest.clone(), vae_config)
                    .map_err(|error| execution_error(variant, error))?;
            let source =
                VerifiedAsyncBurnpackFluxVaeStageSource::new(contract, device.clone(), vae_reader)
                    .with_float_policy(policies.vae_float.vae_artifact_policy());
            BrowserVerifiedVaeSource::Component(AsyncFluxVaeStageSourceAdapter::new(source))
        };
        let vae_source = BrowserAsyncStageSource::new(verified_vae_source, synchronizer.clone());
        let vae = if policies.retain_vae_stages {
            RetainingAsyncBooguVaeStageSource::new(vae_source)
        } else {
            RetainingAsyncBooguVaeStageSource::passthrough(vae_source)
        };
        let verified_denoiser_source = if policies.uses_packed_f16_denoiser_source() {
            BrowserVerifiedDenoiserSource::PackedF16(
                VerifiedAsyncPackedF16DenoiserStageSource::new(
                    &identity,
                    composition.pipeline_manifest,
                    inventory,
                    denoiser_config.clone(),
                    profile,
                    device.clone(),
                    reader.clone(),
                )
                .await
                .map_err(|error| execution_error(variant, error))?,
            )
        } else {
            BrowserVerifiedDenoiserSource::Standard(
                VerifiedAsyncBurnpackDenoiserStageSource::new(
                    &identity,
                    composition.pipeline_manifest,
                    inventory,
                    denoiser_config.clone(),
                    profile,
                    device.clone(),
                    reader.clone(),
                )
                .await
                .map_err(|error| execution_error(variant, error))?
                .with_float_load_policy(policies.denoiser_float)
                .with_quantized_load_policy(policies.denoiser_quantized)
                .with_runtime_quantization_policy(policies.denoiser_runtime_quantization),
            )
        };
        let mut denoiser_source =
            BrowserAsyncStageSource::new(verified_denoiser_source, synchronizer);
        // Both release sizes request q1024. Portable attention independently caps image-scale
        // queries to at least four partitions, so smaller output sizes cannot accidentally
        // materialize a dense seq^2 score tensor. GPU residency never relaxes that bound.
        denoiser_source.set_denoiser_query_chunk_size(
            if policies.residency == BrowserBooguResidencyPolicy::ResidentPackedQ4s {
                512
            } else if variant == BooguVariant::Image01EditTurbo1k5 {
                BROWSER_1K5_DENOISER_QUERY_CHUNK_SIZE
            } else {
                BROWSER_PRODUCTION_DENOISER_QUERY_CHUNK_SIZE
            },
        );
        let denoiser_source = if policies.retain_denoiser_stages {
            RetainingAsyncBooguDenoiserStageSource::new(denoiser_source)
        } else {
            RetainingAsyncBooguDenoiserStageSource::passthrough(denoiser_source)
        };
        let denoiser_source = if policies.defer_retained_denoiser_synchronization {
            denoiser_source
                .with_synchronization_policy(AsyncRetainingDenoiserSynchronizationPolicy::Deferred)
        } else {
            denoiser_source
        };
        let denoiser_source = denoiser_source
            .with_quantized_linear_execution_policy(policies.denoiser_retaining_wrapper_adapter);
        let denoiser = StreamingBooguDenoiser::new(denoiser_config, denoiser_source)
            .map_err(|error| execution_error(variant, error))?;

        let tokenizer = HfTokenizer::from_bytes(&tokenizer_bytes)
            .map_err(|error| execution_error(variant, error))?;
        let pad_token_id = tokenizer
            .token_to_id("<|endoftext|>")
            .ok_or_else(|| execution_error(variant, "Qwen tokenizer is missing <|endoftext|>"))?;
        let processor = Qwen3VlProcessor::new(
            tokenizer,
            boogu_processor_config(&qwen_config, pad_token_id),
        )
        .map_err(|error| execution_error(variant, error))?;
        let image_processor = Qwen3VlImageProcessor::new(
            Qwen3VlImageProcessorConfig::from_json(utf8(&image_processor_bytes, variant)?)
                .map_err(|error| execution_error(variant, error))?,
        )
        .map_err(|error| execution_error(variant, error))?;
        let artifact_control = reader.control();
        let mut engine = Self {
            identity,
            artifact_content_digest,
            numeric_format: settings.numeric_format(),
            qwen_config,
            qwen,
            vae,
            denoiser,
            processor,
            image_processor,
            policies,
            dtypes,
            device,
            applied_buffer_limits,
            resident_resource_plan,
            low_vram_resource_plan,
            packed_f16_resource_plan,
            packed_f16_qwen_embedding_plan,
            last_packed_f16_qwen_host_embedding: None,
            last_packed_f16_qwen_instruction_handoff: None,
            last_packed_f16_denoiser_lifecycle: None,
            last_packed_f16_dmd_vae_handoff: None,
            last_low_vram_denoiser_dtype_audit: None,
            last_dense_f32_materialized_stage_clones: 0,
            last_artifact_traffic: BrowserArtifactTrafficReport::default(),
            artifact_control,
            expected_weight_artifacts,
            expected_vae_encoder_weight_artifacts,
            verified_runtime_metadata,
        };
        if policies.eager_preload {
            engine.preload_resident_weights().await.map_err(|error| {
                execution_error(
                    variant,
                    format!(
                        "browser resident preload failed without fallback under the selected high-throughput residency policy: {error}"
                    ),
                )
            })?;
        } else if policies.preload_denoiser_before_request {
            engine.preload_low_vram_denoiser().await.map_err(|error| {
                execution_error(
                    variant,
                    format!("browser low-vram denoiser preload failed without fallback: {error}"),
                )
            })?;
        }
        engine.artifact_control.set_observer(None);
        engine.artifact_control.clear_events();
        Ok(engine)
    }

    fn expected_qwen_resident_stage_count(&self) -> usize {
        let text = self.qwen.plan.embedding_rows.chunks.len()
            + self.qwen_config.text_config.num_hidden_layers
            + 1;
        let vision = if self.identity.variant.is_edit() {
            1 + self.qwen_config.vision_config.depth
                + self
                    .qwen_config
                    .vision_config
                    .deepstack_visual_indexes
                    .len()
                + 1
        } else {
            0
        };
        text + vision
    }

    fn expected_denoiser_resident_stage_count(&self) -> usize {
        let config = BooguConfig::default();
        let refiner_families = if self.identity.variant.is_edit() {
            3
        } else {
            2
        };
        1 + refiner_families * config.num_refiner_layers
            + config.num_double_stream_layers
            + config.num_single_stream_layers()
            + 1
    }

    fn packed_f16_denoiser_source(
        &self,
    ) -> Result<&BrowserPackedVerifiedDenoiserSource, RuntimeError> {
        self.denoiser
            .source()
            .source()
            .inner()
            .packed_f16()
            .ok_or_else(|| {
                execution_error(
                    self.identity.variant,
                    "browser policy requires the packed-F16 denoiser source",
                )
            })
    }

    fn packed_f16_denoiser_source_mut(
        &mut self,
    ) -> Result<&mut BrowserPackedVerifiedDenoiserSource, RuntimeError> {
        let variant = self.identity.variant;
        self.denoiser
            .source_mut()
            .source_mut()
            .inner_mut()
            .packed_f16_mut()
            .ok_or_else(|| {
                execution_error(
                    variant,
                    "browser policy requires the packed-F16 denoiser source",
                )
            })
    }

    async fn synchronize_preloaded_qwen_stage(&mut self) -> Result<(), RuntimeError> {
        let variant = self.identity.variant;
        self.qwen
            .source
            .synchronize()
            .await
            .map_err(|error| map_boogu(variant, error))?;
        BrowserAsyncSynchronizer::new(&self.device)
            .submit("resident Qwen stage upload")
            .map_err(|error| map_boogu(variant, error))
    }

    async fn finish_preloaded_qwen_uploads(&mut self) -> Result<(), RuntimeError> {
        let variant = self.identity.variant;
        self.qwen
            .source
            .synchronize_pending()
            .await
            .map_err(|error| map_boogu(variant, error))
    }

    async fn synchronize_preloaded_denoiser_stage(&mut self) -> Result<(), RuntimeError> {
        let variant = self.identity.variant;
        self.denoiser
            .source_mut()
            .synchronize()
            .await
            .map_err(|error| map_boogu(variant, error))?;
        BrowserAsyncSynchronizer::new(&self.device)
            .submit("resident denoiser stage upload")
            .map_err(|error| map_boogu(variant, error))
    }

    async fn finish_preloaded_denoiser_uploads(&mut self) -> Result<(), RuntimeError> {
        let variant = self.identity.variant;
        self.denoiser
            .source_mut()
            .synchronize_pending()
            .await
            .map_err(|error| map_boogu(variant, error))
    }

    async fn preload_resident_weights(&mut self) -> Result<(), RuntimeError> {
        let variant = self.identity.variant;
        if !self.policies.residency.is_high_vram_resident()
            || !self.qwen.source.retention_enabled()
            || !self.vae.retention_enabled()
            || !self.denoiser.source().retention_enabled()
        {
            return Err(execution_error(
                variant,
                "resident browser preload requires retaining Qwen, VAE, and denoiser sources",
            ));
        }
        report_browser_runtime_preparing(match self.policies.residency {
            BrowserBooguResidencyPolicy::HighVramResidentPackedF16 => {
                "Verifying and retaining packed-F16 Qwen, VAE, and denoiser weights on WebGPU"
            }
            BrowserBooguResidencyPolicy::ResidentPackedQ4s => {
                "Quantizing and retaining Q4S Qwen/denoiser matrices and embedding rows with packed-F16 VAE weights on WebGPU"
            }
            _ => "Verifying and materializing dense-F32 Qwen, VAE, and denoiser weights on WebGPU",
        });
        self.qwen
            .source
            .source_mut()
            .set_qwen_resident_preload(true);

        if !self.policies.uses_host_routed_qwen_embedding() {
            for spec in self.qwen.plan.embedding_rows.chunks.clone() {
                drop(
                    self.qwen
                        .source
                        .load_embedding_rows(&spec)
                        .await
                        .map_err(|error| map_boogu(variant, error))?,
                );
                self.synchronize_preloaded_qwen_stage().await?;
            }
        }
        if variant.is_edit() {
            drop(
                self.qwen
                    .source
                    .load_vision_prelude()
                    .await
                    .map_err(|error| map_boogu(variant, error))?,
            );
            self.synchronize_preloaded_qwen_stage().await?;
            for index in 0..self.qwen_config.vision_config.depth {
                drop(
                    self.qwen
                        .source
                        .load_vision_block(index)
                        .await
                        .map_err(|error| map_boogu(variant, error))?,
                );
                self.synchronize_preloaded_qwen_stage().await?;
            }
            for index in 0..self
                .qwen_config
                .vision_config
                .deepstack_visual_indexes
                .len()
            {
                drop(
                    self.qwen
                        .source
                        .load_vision_deepstack_merger(index)
                        .await
                        .map_err(|error| map_boogu(variant, error))?,
                );
                self.synchronize_preloaded_qwen_stage().await?;
            }
            drop(
                self.qwen
                    .source
                    .load_vision_final_merger()
                    .await
                    .map_err(|error| map_boogu(variant, error))?,
            );
            self.synchronize_preloaded_qwen_stage().await?;
        }
        for index in 0..self.qwen_config.text_config.num_hidden_layers {
            drop(
                self.qwen
                    .source
                    .load_text_block(index)
                    .await
                    .map_err(|error| map_boogu(variant, error))?,
            );
            self.synchronize_preloaded_qwen_stage().await?;
        }
        drop(
            self.qwen
                .source
                .load_text_final_norm()
                .await
                .map_err(|error| map_boogu(variant, error))?,
        );
        self.synchronize_preloaded_qwen_stage().await?;
        self.finish_preloaded_qwen_uploads().await?;
        self.qwen
            .source
            .source_mut()
            .set_qwen_resident_preload(false);

        if variant.is_edit() {
            drop(
                self.vae
                    .load_encoder()
                    .await
                    .map_err(|error| map_boogu(variant, error))?,
            );
            self.vae
                .synchronize()
                .await
                .map_err(|error| map_boogu(variant, error))?;
        }
        drop(
            self.vae
                .load_decoder()
                .await
                .map_err(|error| map_boogu(variant, error))?,
        );
        self.vae
            .synchronize()
            .await
            .map_err(|error| map_boogu(variant, error))?;

        drop(
            self.denoiser
                .source_mut()
                .load_prelude()
                .await
                .map_err(|error| map_boogu(variant, error))?,
        );
        self.synchronize_preloaded_denoiser_stage().await?;
        let config = BooguConfig::default();
        for index in 0..config.num_refiner_layers {
            drop(
                self.denoiser
                    .source_mut()
                    .load_context_refiner(index)
                    .await
                    .map_err(|error| map_boogu(variant, error))?,
            );
            self.synchronize_preloaded_denoiser_stage().await?;
            drop(
                self.denoiser
                    .source_mut()
                    .load_noise_refiner(index)
                    .await
                    .map_err(|error| map_boogu(variant, error))?,
            );
            self.synchronize_preloaded_denoiser_stage().await?;
            if variant.is_edit() {
                drop(
                    self.denoiser
                        .source_mut()
                        .load_reference_refiner(index)
                        .await
                        .map_err(|error| map_boogu(variant, error))?,
                );
                self.synchronize_preloaded_denoiser_stage().await?;
            }
        }
        for index in 0..config.num_double_stream_layers {
            drop(
                self.denoiser
                    .source_mut()
                    .load_double_stream(index)
                    .await
                    .map_err(|error| map_boogu(variant, error))?,
            );
            self.synchronize_preloaded_denoiser_stage().await?;
        }
        for index in 0..config.num_single_stream_layers() {
            drop(
                self.denoiser
                    .source_mut()
                    .load_single_stream(index)
                    .await
                    .map_err(|error| map_boogu(variant, error))?,
            );
            self.synchronize_preloaded_denoiser_stage().await?;
        }
        drop(
            self.denoiser
                .source_mut()
                .load_tail()
                .await
                .map_err(|error| map_boogu(variant, error))?,
        );
        self.synchronize_preloaded_denoiser_stage().await?;
        self.finish_preloaded_denoiser_uploads().await?;
        if self.policies.phase_boundary_memory_cleanup {
            release_browser_phase_allocator(variant, &self.device, "resident model preload")
                .await?;
        }
        self.validate_resident_caches()
    }

    async fn preload_low_vram_denoiser(&mut self) -> Result<(), RuntimeError> {
        let variant = self.identity.variant;
        if self.policies.residency != BrowserBooguResidencyPolicy::LowVramPreloadedPackedF16Denoiser
            || self.denoiser.source().retention_enabled()
            || !self.policies.uses_packed_f16_denoiser_source()
            || self.policies.denoiser_retaining_wrapper_adapter
                != BooguQuantizedLinearExecutionPolicy::DirectQuantizedMatmul
        {
            return Err(execution_error(
                variant,
                "low-vram denoiser preload requires the exact Turbo packed-F16 passthrough policy",
            ));
        }
        report_browser_runtime_preparing(
            "Verifying and preloading all 46 Turbo packed-F16 denoiser stages before inference",
        );
        let traffic_before = self.artifact_control.traffic_snapshot();
        let audit_before = self.packed_f16_denoiser_source()?.audit();
        let audit = self
            .packed_f16_denoiser_source_mut()?
            .preload_turbo_raw()
            .await
            .map_err(|error| map_boogu(variant, error))?;
        let plan = self.packed_f16_resource_plan.ok_or_else(|| {
            execution_error(variant, "browser packed-F16 resource plan is absent")
        })?;
        validate_packed_f16_denoiser_preload(variant, plan, audit_before, audit)?;
        let traffic = self
            .artifact_control
            .traffic_snapshot()
            .checked_delta(traffic_before)
            .ok_or_else(|| {
                execution_error(variant, "denoiser preload traffic counters moved backwards")
            })?;
        dispatch_browser_event(
            BROWSER_RUNTIME_EVENT_NAME,
            &BrowserRuntimeEvent::PackedF16DenoiserPreload {
                traffic: traffic.into(),
                cached_stages: audit.cached_stage_count,
                cached_objects: audit.cached_object_count,
                cached_tensors: audit.cached_tensor_count,
                cached_bytes: audit.retained_packed_bytes,
                previous_preload_attempt_count: audit_before.preload_attempt_count,
                preload_attempt_count: audit.preload_attempt_count,
                request_scoped_rehydration: audit_before.preload_attempt_count > 0,
                rehydration_policy: BROWSER_PACKED_F16_NEXT_REQUEST_REHYDRATION_POLICY,
            },
        );
        Ok(())
    }

    async fn ensure_preloaded_packed_f16_denoiser(&mut self) -> Result<(), RuntimeError> {
        // Resident dense-F32 also sets `preload_denoiser_before_request`: its denoiser was loaded
        // eagerly into the retaining source and has no packed-F16 plan by design. Only the exact
        // packed source owns the request-scoped cache/rehydration contract below.
        if !self.policies.requires_packed_f16_request_preload() {
            return Ok(());
        }
        let plan = self.packed_f16_resource_plan.ok_or_else(|| {
            execution_error(
                self.identity.variant,
                "browser packed-F16 resource plan is absent",
            )
        })?;
        let audit = self.packed_f16_denoiser_source()?.audit();
        if audit.state == PackedF16DenoiserCacheState::Ready
            && audit.packed_cache_ready
            && audit.cached_stage_count == plan.expected_stage_count
            && audit.cached_object_count == plan.expected_object_count
            && audit.cached_tensor_count == plan.expected_tensor_count
            && audit.retained_packed_bytes == plan.retained_packed_f16_denoiser_bytes
            && !self.denoiser.source().has_pending_synchronization()
        {
            return Ok(());
        }

        // A failed/cancelled DMD request invalidates the retained cache after an unconditional
        // queue barrier. Recover the next request from the verified persistent range cache rather
        // than silently running with a partial semantic graph.
        self.denoiser
            .source_mut()
            .synchronize()
            .await
            .map_err(|error| map_boogu(self.identity.variant, error))?;
        self.denoiser.source_mut().clear();
        self.packed_f16_denoiser_source_mut()?.clear();
        self.denoiser.clear_rope_cache();
        let synchronizer = BrowserAsyncSynchronizer::new(&self.device);
        synchronizer
            .synchronize("packed-F16 request rehydration (before allocator cleanup)")
            .await
            .map_err(|error| map_boogu(self.identity.variant, error))?;
        <BrowserBackend as Backend>::memory_cleanup(&self.device);
        synchronizer
            .synchronize("packed-F16 request rehydration (after allocator cleanup)")
            .await
            .map_err(|error| map_boogu(self.identity.variant, error))?;
        let empty_audit = self.packed_f16_denoiser_source()?.audit();
        self.require_empty_packed_f16_cache_audit(
            "next-request rehydration before preload",
            empty_audit,
        )?;
        self.preload_low_vram_denoiser().await
    }

    fn require_exact_packed_f16_cache_audit(
        &self,
        boundary: &str,
        audit: PackedF16DenoiserCacheAudit,
    ) -> Result<(), RuntimeError> {
        let variant = self.identity.variant;
        let plan = self.packed_f16_resource_plan.ok_or_else(|| {
            execution_error(variant, "browser packed-F16 resource plan is absent")
        })?;
        let exact = audit.state == PackedF16DenoiserCacheState::Ready
            && audit.packed_cache_ready
            && audit.cached_stage_count == plan.expected_stage_count
            && audit.cached_object_count == plan.expected_object_count
            && audit.cached_tensor_count == plan.expected_tensor_count
            && audit.retained_packed_bytes == plan.retained_packed_f16_denoiser_bytes
            && audit.packed_read_bytes >= plan.authenticated_artifact_bytes
            && audit.packed_upload_bytes >= plan.retained_packed_f16_denoiser_bytes;
        if !exact {
            return Err(execution_error(
                variant,
                format!(
                    "browser packed-F16 cache differs from its exact plan at {boundary}: {audit:?}"
                ),
            ));
        }
        Ok(())
    }

    fn require_empty_packed_f16_cache_audit(
        &self,
        boundary: &str,
        audit: PackedF16DenoiserCacheAudit,
    ) -> Result<(), RuntimeError> {
        let exact = audit.state == PackedF16DenoiserCacheState::Empty
            && !audit.packed_cache_ready
            && audit.cached_stage_count == 0
            && audit.cached_object_count == 0
            && audit.cached_tensor_count == 0
            && audit.retained_packed_bytes == 0;
        if !exact {
            return Err(execution_error(
                self.identity.variant,
                format!("browser packed-F16 cache is not exactly empty at {boundary}: {audit:?}"),
            ));
        }
        Ok(())
    }

    async fn packed_f16_dmd_vae_handoff(
        &mut self,
        latents: Tensor<BrowserBackend, 4>,
        expected_shape: [usize; 4],
        run_id: RunId,
    ) -> Result<Tensor<BrowserBackend, 4>, RuntimeError> {
        let variant = self.identity.variant;
        if variant != BooguVariant::Image01Turbo
            || self.policies.residency
                != BrowserBooguResidencyPolicy::LowVramPreloadedPackedF16Denoiser
            || !self.policies.uses_packed_f16_denoiser_source()
            || self.dtypes.denoiser != DType::F32
            || self.dtypes.vae != DType::F32
        {
            return Err(execution_error(
                variant,
                "packed-F16 DMD-to-VAE handoff requires the exact Turbo F32 low-VRAM policy",
            ));
        }

        let shape = latents.dims();
        if shape != expected_shape || latents.dtype() != DType::F32 {
            return Err(execution_error(
                variant,
                format!(
                    "packed-F16 final DMD latent must be exact F32 {expected_shape:?}, got {shape:?} {}",
                    latents.dtype().name()
                ),
            ));
        }
        let wrapper_cached_stages_before_clear = self.denoiser.source().cached_stage_count();
        let synchronization_pending_before_cleanup =
            self.denoiser.source().has_pending_synchronization();
        let audit_before = self.packed_f16_denoiser_source()?.audit();
        self.require_exact_packed_f16_cache_audit("DMD-to-VAE handoff entry", audit_before)?;
        if wrapper_cached_stages_before_clear != 0 || synchronization_pending_before_cleanup {
            return Err(execution_error(
                variant,
                format!(
                    "packed-F16 DMD-to-VAE handoff entered with live wrapper stages or pending work: cached_stages={wrapper_cached_stages_before_clear}, synchronization_pending={synchronization_pending_before_cleanup}"
                ),
            ));
        }

        // Consume the only final-latent device handle. This readback is the exact value that will
        // be reuploaded after the 19.87 GB request-scoped packed cache has been released.
        let latent_data = latents.into_data_async().await.map_err(|error| {
            execution_error(
                variant,
                format!("packed-F16 final DMD latent readback failed: {error}"),
            )
        })?;
        let latent_before = packed_f16_tensor_data_input_diagnostic(
            variant,
            "final_dmd_latent_before_vae_handoff",
            expected_shape.to_vec(),
            &latent_data,
        )?;
        require_finite_nonzero_packed_f16_diagnostic(variant, &latent_before)?;
        let payload_bytes = u64::try_from(latent_data.bytes.len()).map_err(|_| {
            execution_error(
                variant,
                "packed-F16 final DMD latent payload byte count does not fit u64",
            )
        })?;
        let expected_payload_bytes = u64::try_from(latent_before.element_count)
            .ok()
            .and_then(|elements| elements.checked_mul(4))
            .ok_or_else(|| {
                execution_error(
                    variant,
                    "packed-F16 final DMD latent payload byte count overflowed",
                )
            })?;
        if payload_bytes != expected_payload_bytes {
            return Err(execution_error(
                variant,
                format!(
                    "packed-F16 final DMD latent has {payload_bytes} host bytes, expected exact F32 size {expected_payload_bytes}"
                ),
            ));
        }

        // The DMD loop already crossed its real per-step barriers. One explicit boundary barrier
        // after the final-latent map makes the subsequent cache-handle destruction independently
        // auditable and protects this helper if its caller's lifecycle changes later.
        let synchronizer = BrowserAsyncSynchronizer::new(&self.device);
        synchronizer
            .synchronize("packed-F16 DMD-to-VAE handoff (before cache clear)")
            .await
            .map_err(|error| map_boogu(variant, error))?;

        // Drop every cache layer before forcing the allocator boundary so VAE cannot overlap the
        // raw packed arena.
        self.denoiser.source_mut().clear();
        self.denoiser.clear_rope_cache();
        self.packed_f16_denoiser_source_mut()?.clear();
        synchronizer
            .synchronize("packed-F16 DMD-to-VAE handoff (before allocator cleanup)")
            .await
            .map_err(|error| map_boogu(variant, error))?;
        <BrowserBackend as Backend>::memory_cleanup(&self.device);
        synchronizer
            .synchronize("packed-F16 DMD-to-VAE handoff (after allocator cleanup)")
            .await
            .map_err(|error| map_boogu(variant, error))?;

        let wrapper_cached_stages_after_clear = self.denoiser.source().cached_stage_count();
        let synchronization_pending_after_cleanup =
            self.denoiser.source().has_pending_synchronization();
        let audit_after = self.packed_f16_denoiser_source()?.audit();
        self.require_empty_packed_f16_cache_audit("DMD-to-VAE handoff exit", audit_after)?;
        if wrapper_cached_stages_after_clear != 0 || synchronization_pending_after_cleanup {
            return Err(execution_error(
                variant,
                format!(
                    "packed-F16 DMD-to-VAE cleanup retained wrapper stages or pending work: cached_stages={wrapper_cached_stages_after_clear}, synchronization_pending={synchronization_pending_after_cleanup}"
                ),
            ));
        }

        let latents =
            Tensor::<BrowserBackend, 4>::from_data(latent_data, (&self.device, DType::F32));
        let latent_after = read_packed_f16_tensor_input_diagnostic(
            variant,
            "final_dmd_latent_after_vae_handoff",
            &latents,
        )
        .await?;
        require_finite_nonzero_packed_f16_diagnostic(variant, &latent_after)?;
        let digest_matches = latent_after.shape == latent_before.shape
            && latent_after.dtype == latent_before.dtype
            && latent_after.element_count == latent_before.element_count
            && latent_after.sha256 == latent_before.sha256;
        let device_to_host_readback_bytes = payload_bytes.checked_mul(2).ok_or_else(|| {
            execution_error(
                variant,
                "packed-F16 DMD-to-VAE readback byte count overflowed",
            )
        })?;
        let total_transfer_bytes = device_to_host_readback_bytes
            .checked_add(payload_bytes)
            .ok_or_else(|| {
                execution_error(
                    variant,
                    "packed-F16 DMD-to-VAE total transfer byte count overflowed",
                )
            })?;
        let expected_next_request_preload_attempt_count = audit_after
            .preload_attempt_count
            .checked_add(1)
            .ok_or_else(|| {
                execution_error(
                    variant,
                    "packed-F16 preload-attempt counter overflowed at the VAE boundary",
                )
            })?;
        let report = BrowserPackedF16DmdVaeHandoffReport {
            policy: BROWSER_PACKED_F16_DMD_VAE_HANDOFF_POLICY.into(),
            next_request_rehydration_policy: BROWSER_PACKED_F16_NEXT_REQUEST_REHYDRATION_POLICY
                .into(),
            shape: latent_after.shape.clone(),
            dtype: latent_after.dtype.clone(),
            element_count: latent_after.element_count,
            payload_bytes,
            device_to_host_readback_bytes,
            host_to_device_upload_bytes: payload_bytes,
            total_transfer_bytes,
            before_sha256: latent_before.sha256,
            after_sha256: latent_after.sha256,
            all_finite: latent_before.all_finite && latent_after.all_finite,
            not_all_zero: !packed_f16_tensor_diagnostic_is_all_zero(&latent_before)
                && !packed_f16_tensor_diagnostic_is_all_zero(&latent_after),
            digest_matches,
            wrapper_cached_stages_before_clear,
            wrapper_cached_stages_after_clear,
            synchronization_pending_before_cleanup,
            synchronization_pending_after_cleanup,
            rope_cache_cleared: true,
            cleanup_completed: true,
            packed_cache_before_cleanup: packed_f16_cache_evidence(audit_before),
            packed_cache_after_cleanup: packed_f16_cache_evidence(audit_after),
            preload_attempt_count: audit_after.preload_attempt_count,
            expected_next_request_preload_attempt_count,
        };
        validate_packed_f16_dmd_vae_handoff_report(
            variant,
            self.packed_f16_resource_plan.ok_or_else(|| {
                execution_error(variant, "browser packed-F16 resource plan is absent")
            })?,
            expected_shape,
            &report,
        )?;
        dispatch_browser_event(
            BROWSER_RUNTIME_EVENT_NAME,
            &browser_packed_f16_dmd_vae_handoff_event(run_id, report.clone()),
        );
        self.last_packed_f16_dmd_vae_handoff = Some(report);
        Ok(latents)
    }

    async fn fail_closed_packed_f16_request_cleanup(&mut self) -> Result<(), RuntimeError> {
        let variant = self.identity.variant;
        let synchronizer = BrowserAsyncSynchronizer::new(&self.device);
        let before_clear = synchronizer
            .synchronize("failed packed-F16 request (before cache clear)")
            .await
            .map_err(|error| map_boogu(variant, error));

        self.denoiser.source_mut().clear();
        self.denoiser.clear_rope_cache();
        let source_result = self.packed_f16_denoiser_source_mut().map(|source| {
            source.fail_and_clear();
        });
        let after_clear = synchronizer
            .synchronize("failed packed-F16 request (before allocator cleanup)")
            .await
            .map_err(|error| map_boogu(variant, error));
        <BrowserBackend as Backend>::memory_cleanup(&self.device);
        let after_cleanup = synchronizer
            .synchronize("failed packed-F16 request (after allocator cleanup)")
            .await
            .map_err(|error| map_boogu(variant, error));

        let local_state_result = self.packed_f16_denoiser_source().and_then(|source| {
            let audit = source.audit();
            if audit.state != PackedF16DenoiserCacheState::Ready
                && !audit.packed_cache_ready
                && audit.cached_stage_count == 0
                && audit.cached_object_count == 0
                && audit.cached_tensor_count == 0
                && audit.retained_packed_bytes == 0
                && self.denoiser.source().cached_stage_count() == 0
                && !self.denoiser.source().has_pending_synchronization()
            {
                Ok(())
            } else {
                Err(execution_error(
                    variant,
                    format!(
                        "failed packed-F16 request cleanup left ambiguous local state: {audit:?}"
                    ),
                ))
            }
        });
        source_result
            .and(local_state_result)
            .and(before_clear)
            .and(after_clear)
            .and(after_cleanup)
    }

    async fn packed_f16_qwen_instruction_handoff(
        &mut self,
        instruction: Tensor<BrowserBackend, 3>,
        run_id: RunId,
        rendered_context: Option<BrowserPackedF16QwenPreHandoffContext>,
    ) -> Result<
        (
            Tensor<BrowserBackend, 3>,
            Option<PackedF16DenoiserCacheAudit>,
            Option<BrowserPackedF16QwenInstructionHandoffReport>,
        ),
        RuntimeError,
    > {
        if !self.policies.uses_packed_f16_denoiser_source() {
            if self.policies.release_unused_qwen_memory_after_stage
                || self.policies.packed_qwen_instruction_handoff
                || self.qwen.releases_unused_memory_after_stage()
                || rendered_context.is_some()
            {
                return Err(execution_error(
                    self.identity.variant,
                    "a non-packed browser policy enabled the packed-F16 Qwen handoff",
                ));
            }
            return Ok((instruction, None, None));
        }
        if !self.policies.packed_allocator_policy_is_exact()
            || self.qwen.releases_unused_memory_after_stage()
        {
            return Err(execution_error(
                self.identity.variant,
                "packed-F16 Qwen handoff requires per-stage allocator cleanup to remain disabled",
            ));
        }
        if self.qwen.source.cached_stage_count() != 0
            || self.qwen.source.has_pending_synchronization()
        {
            return Err(execution_error(
                self.identity.variant,
                "packed-F16 Qwen handoff requires every streamed Qwen module to be dropped and synchronized",
            ));
        }

        let variant = self.identity.variant;
        let instruction_shape = instruction.dims().to_vec();
        if instruction_shape.len() != 3
            || instruction_shape[0] != 1
            || instruction_shape[1] == 0
            || instruction_shape[2] != 4096
            || instruction.dtype() != DType::F32
        {
            return Err(execution_error(
                variant,
                format!(
                    "packed-F16 Qwen handoff requires one nonempty 4096-wide F32 instruction, got shape {instruction_shape:?} dtype {}",
                    instruction.dtype().name()
                ),
            ));
        }

        // Consume the only device instruction handle into a bounded exact-F32 host payload before
        // allocator cleanup. The released ordinary Turbo request is about 737 KiB (45*4096*4),
        // and arbitrary valid prompt lengths remain shape/count checked rather than rounded.
        let instruction_data = instruction.into_data_async().await.map_err(|error| {
            execution_error(
                variant,
                format!("packed-F16 Qwen instruction handoff readback failed: {error}"),
            )
        })?;
        let instruction_before = packed_f16_tensor_data_input_diagnostic(
            variant,
            "instruction_after_trim_cast_before_handoff",
            instruction_shape.clone(),
            &instruction_data,
        )?;
        let payload_bytes = u64::try_from(instruction_data.bytes.len()).map_err(|_| {
            execution_error(
                variant,
                "packed-F16 Qwen instruction host payload byte count does not fit u64",
            )
        })?;
        let expected_payload_bytes = u64::try_from(instruction_before.element_count)
            .ok()
            .and_then(|elements| elements.checked_mul(4))
            .ok_or_else(|| {
                execution_error(
                    variant,
                    "packed-F16 Qwen instruction handoff byte count overflowed",
                )
            })?;
        if payload_bytes != expected_payload_bytes {
            return Err(execution_error(
                variant,
                format!(
                    "packed-F16 Qwen instruction host payload has {payload_bytes} bytes, expected exact F32 size {expected_payload_bytes}"
                ),
            ));
        }

        let emit_rendered_diagnostics = rendered_context.is_some();
        if let Some(context) = rendered_context {
            let block_00_immediate_matches_delayed_capture = context
                .stage_outputs
                .iter()
                .find(|diagnostic| diagnostic.name == "qwen_text_block_00_output")
                .is_some_and(|delayed| {
                    let immediate = &context.block_00_immediate_post_sync.tensor;
                    delayed.shape == immediate.shape
                        && delayed.dtype == immediate.dtype
                        && delayed.sha256 == immediate.sha256
                });
            let final_norm_matches_returned_output =
                context.stage_outputs.last().is_some_and(|diagnostic| {
                    diagnostic.name == "qwen_final_norm_output"
                        && diagnostic.shape == context.qwen_last_hidden_state_before_trim.shape
                        && diagnostic.dtype == context.qwen_last_hidden_state_before_trim.dtype
                        && diagnostic.sha256 == context.qwen_last_hidden_state_before_trim.sha256
                });
            let first_non_finite_tensor = context
                .stage_outputs
                .iter()
                .chain(std::iter::once(&context.qwen_last_hidden_state_before_trim))
                .chain(std::iter::once(&instruction_before))
                .find(|diagnostic| !diagnostic.all_finite)
                .map(|diagnostic| diagnostic.name.clone());
            let first_all_zero_tensor = context
                .stage_outputs
                .iter()
                .chain(std::iter::once(&context.qwen_last_hidden_state_before_trim))
                .chain(std::iter::once(&instruction_before))
                .find(|diagnostic| packed_f16_tensor_diagnostic_is_all_zero(diagnostic))
                .map(|diagnostic| diagnostic.name.clone());
            let stage_count_matches =
                context.stage_outputs.len() == context.expected_stage_output_count;
            let stage_names_exact = packed_f16_qwen_stage_diagnostic_names_are_exact(
                &context.stage_outputs,
                context.expected_stage_output_count,
            );
            let diagnostics = BrowserPackedF16QwenPreHandoffDiagnostics {
                scope: BROWSER_PACKED_F16_QWEN_PRE_HANDOFF_SCOPE.into(),
                effective_instruction_length: context.effective_instruction_length,
                expected_stage_output_count: context.expected_stage_output_count,
                stage_outputs: context.stage_outputs,
                stage_names_exact,
                qwen_last_hidden_state_before_trim: context.qwen_last_hidden_state_before_trim,
                instruction_after_trim_cast_before_handoff: instruction_before.clone(),
                all_tensors_finite: first_non_finite_tensor.is_none(),
                no_tensor_all_zero: first_all_zero_tensor.is_none(),
                first_non_finite_tensor: first_non_finite_tensor.clone(),
                first_all_zero_tensor: first_all_zero_tensor.clone(),
                final_norm_matches_returned_output,
                block_00_immediate_post_sync: context.block_00_immediate_post_sync,
                block_00_immediate_matches_delayed_capture,
            };
            dispatch_browser_event(
                BROWSER_RUNTIME_EVENT_NAME,
                &browser_packed_f16_qwen_pre_handoff_diagnostics_event(run_id, diagnostics),
            );
            if !stage_count_matches {
                return Err(execution_error(
                    variant,
                    "rendered-smoke Qwen diagnostics captured the wrong stage count before handoff",
                ));
            }
            if !stage_names_exact {
                return Err(execution_error(
                    variant,
                    "rendered-smoke Qwen diagnostic stage names or order differ from the released text path",
                ));
            }
            if !final_norm_matches_returned_output {
                return Err(execution_error(
                    variant,
                    "rendered-smoke Qwen final-norm observer differs from the returned hidden state",
                ));
            }
            if !block_00_immediate_matches_delayed_capture {
                return Err(execution_error(
                    variant,
                    "rendered-smoke Qwen block-0 delayed capture differs from its immediate post-sync readback",
                ));
            }
            if let Some(name) = first_non_finite_tensor {
                return Err(execution_error(
                    variant,
                    format!("rendered-smoke Qwen tensor {name} is non-finite before handoff"),
                ));
            }
            if let Some(name) = first_all_zero_tensor {
                return Err(execution_error(
                    variant,
                    format!("rendered-smoke Qwen tensor {name} is all-zero before handoff"),
                ));
            }
        }
        require_finite_nonzero_packed_f16_diagnostic(variant, &instruction_before)?;

        // `Backend::sync` is blocking and cannot be called from the single-threaded Wasm event
        // loop. The runtime-client barriers drain the same BrowserBackend queue without blocking:
        // exact host handoff -> async sync -> backend allocator cleanup -> async sync -> exact F32
        // reupload. No live Qwen activation handle crosses the allocator boundary; packed-cache
        // handles remain rooted throughout.
        let synchronizer = BrowserAsyncSynchronizer::new(&self.device);
        synchronizer
            .synchronize("packed-F16 Qwen instruction handoff (before cleanup)")
            .await
            .map_err(|error| map_boogu(self.identity.variant, error))?;
        <BrowserBackend as Backend>::memory_cleanup(&self.device);
        synchronizer
            .synchronize("packed-F16 Qwen instruction handoff (after cleanup)")
            .await
            .map_err(|error| map_boogu(self.identity.variant, error))?;

        let audit = self.packed_f16_denoiser_source()?.audit();
        self.require_exact_packed_f16_cache_audit("post-Qwen instruction handoff", audit)?;

        let instruction =
            Tensor::<BrowserBackend, 3>::from_data(instruction_data, (&self.device, DType::F32));
        let instruction_after = read_packed_f16_tensor_input_diagnostic(
            variant,
            "instruction_after_handoff",
            &instruction,
        )
        .await?;
        let digest_matches = instruction_after.shape == instruction_before.shape
            && instruction_after.dtype == instruction_before.dtype
            && instruction_after.element_count == instruction_before.element_count
            && instruction_after.sha256 == instruction_before.sha256;
        let device_to_host_readback_bytes = payload_bytes.checked_mul(2).ok_or_else(|| {
            execution_error(
                variant,
                "packed-F16 Qwen instruction verification readback byte count overflowed",
            )
        })?;
        let total_transfer_bytes = device_to_host_readback_bytes
            .checked_add(payload_bytes)
            .ok_or_else(|| {
                execution_error(
                    variant,
                    "packed-F16 Qwen instruction total transfer byte count overflowed",
                )
            })?;
        let report = BrowserPackedF16QwenInstructionHandoffReport {
            policy: BROWSER_PACKED_F16_QWEN_HANDOFF_POLICY.into(),
            qwen_release_unused_memory_after_stage: false,
            qwen_text_layer_allocation_policy: self
                .qwen
                .text_layer_allocation_policy()
                .label()
                .into(),
            qwen_text_block_load_synchronization_policy: self
                .qwen
                .text_block_load_synchronization_policy()
                .label()
                .into(),
            qwen_text_layer_submission_policy: self
                .policies
                .qwen_text_layer_submission_policy()
                .into(),
            shape: instruction_after.shape.clone(),
            dtype: instruction_after.dtype.clone(),
            element_count: instruction_after.element_count,
            payload_bytes,
            device_to_host_readback_bytes,
            host_to_device_upload_bytes: payload_bytes,
            total_transfer_bytes,
            before_sha256: instruction_before.sha256,
            after_sha256: instruction_after.sha256,
            all_finite: instruction_before.all_finite && instruction_after.all_finite,
            not_all_zero: !packed_f16_tensor_diagnostic_is_all_zero(&instruction_before)
                && !packed_f16_tensor_diagnostic_is_all_zero(&instruction_after),
            digest_matches,
            cleanup_completed: true,
            packed_cache: packed_f16_cache_evidence(audit),
        };
        if emit_rendered_diagnostics {
            let diagnostics = BrowserPackedF16QwenPostHandoffDiagnostics {
                scope: BROWSER_PACKED_F16_QWEN_POST_HANDOFF_SCOPE.into(),
                handoff: report.clone(),
                instruction_after_handoff: instruction_after.clone(),
            };
            dispatch_browser_event(
                BROWSER_RUNTIME_EVENT_NAME,
                &browser_packed_f16_qwen_post_handoff_diagnostics_event(run_id, diagnostics),
            );
        }
        require_finite_nonzero_packed_f16_diagnostic(variant, &instruction_after)?;
        if !digest_matches {
            return Err(execution_error(
                variant,
                "packed-F16 Qwen instruction changed across exact host handoff and allocator cleanup",
            ));
        }
        Ok((instruction, Some(audit), Some(report)))
    }

    fn validate_resident_caches(&self) -> Result<(), RuntimeError> {
        if !self.policies.residency.is_high_vram_resident() {
            return Ok(());
        }
        let expected_qwen = self.expected_qwen_resident_stage_count();
        let expected_vae = if self.identity.variant.is_edit() {
            2
        } else {
            1
        };
        let expected_denoiser = self.expected_denoiser_resident_stage_count();
        let actual = (
            self.qwen.source.cached_stage_count(),
            self.vae.cached_stage_count(),
            self.denoiser.source().cached_stage_count(),
        );
        if actual != (expected_qwen, expected_vae, expected_denoiser)
            || self.qwen.source.has_pending_synchronization()
            || self.denoiser.source().has_pending_synchronization()
        {
            return Err(execution_error(
                self.identity.variant,
                format!(
                    "resident browser cache is incomplete: Qwen {}/{expected_qwen}, VAE {}/{expected_vae}, denoiser {}/{expected_denoiser}",
                    actual.0, actual.1, actual.2
                ),
            ));
        }
        Ok(())
    }

    fn report_resident_cache_audit(&self, run_id: RunId, boundary: &'static str) {
        if !self.policies.residency.is_high_vram_resident() {
            return;
        }
        let qwen_expected_stages = self.expected_qwen_resident_stage_count();
        let vae_expected_stages = if self.identity.variant.is_edit() {
            2
        } else {
            1
        };
        let denoiser_expected_stages = self.expected_denoiser_resident_stage_count();
        let qwen_cached_stages = self.qwen.source.cached_stage_count();
        let vae_cached_stages = self.vae.cached_stage_count();
        let denoiser_cached_stages = self.denoiser.source().cached_stage_count();
        let qwen_synchronization_pending = self.qwen.source.has_pending_synchronization();
        let denoiser_synchronization_pending = self.denoiser.source().has_pending_synchronization();
        let resident_weights_preserved = qwen_cached_stages == qwen_expected_stages
            && vae_cached_stages == vae_expected_stages
            && denoiser_cached_stages == denoiser_expected_stages
            && !qwen_synchronization_pending
            && !denoiser_synchronization_pending;
        dispatch_browser_event(
            BROWSER_RUNTIME_EVENT_NAME,
            &BrowserRuntimeEvent::ResidentCacheAudit {
                run_id,
                boundary,
                qwen_cached_stages,
                qwen_expected_stages,
                vae_cached_stages,
                vae_expected_stages,
                denoiser_cached_stages,
                denoiser_expected_stages,
                qwen_synchronization_pending,
                denoiser_synchronization_pending,
                resident_weights_preserved,
            },
        );
    }

    async fn probe_qwen_text_final_norm(&mut self) -> Result<BrowserStageProbe, RuntimeError> {
        let variant = self.identity.variant;
        self.artifact_control.clear_events();
        probe_milestone("final-norm-load-start");
        let norm = self
            .qwen
            .source
            .load_text_final_norm()
            .await
            .map_err(|error| map_boogu(variant, error))?;
        probe_milestone("final-norm-load-complete");
        let mut stage = None;
        let mut verified_objects = 0;
        while let Some(event) = self.artifact_control.pop_event() {
            match event {
                BrowserArtifactEvent::Started(file) => {
                    stage = Some((file.path.to_string(), file.size));
                }
                BrowserArtifactEvent::Verified(_) => verified_objects += 1,
                BrowserArtifactEvent::Progress { .. } => {}
            }
        }
        let (stage, bytes) = stage.ok_or_else(|| {
            execution_error(
                variant,
                "Qwen final-norm probe did not observe a bounded artifact object",
            )
        })?;
        if verified_objects != 1 {
            return Err(execution_error(
                variant,
                format!(
                    "Qwen final-norm probe verified {verified_objects} objects; expected exactly one"
                ),
            ));
        }
        probe_milestone("final-norm-artifact-verified");
        let dtype = norm.gamma.val().dtype();
        let elements = self.qwen_config.text_config.hidden_size;
        probe_milestone("final-norm-forward-start");
        let output = norm
            .forward(Tensor::<BrowserBackend, 3>::ones([1, 1, elements], &self.device).cast(dtype));
        probe_milestone("final-norm-forward-submitted");
        probe_milestone("final-norm-synchronize-start");
        self.qwen
            .source
            .synchronize()
            .await
            .map_err(|error| map_boogu(variant, error))?;
        probe_milestone("final-norm-synchronize-complete");
        probe_milestone("final-norm-readback-start");
        let values = output
            .into_data_async()
            .await
            .map_err(|error| execution_error(variant, format!("probe readback failed: {error}")))?
            .convert_dtype(DType::F32)
            .to_vec::<f32>()
            .map_err(|error| execution_error(variant, error))?;
        probe_milestone("final-norm-readback-complete");
        drop(norm);
        let finite_elements = values.iter().filter(|value| value.is_finite()).count();
        if finite_elements != elements {
            return Err(execution_error(
                variant,
                format!(
                    "Qwen final-norm probe returned {finite_elements}/{elements} finite values"
                ),
            ));
        }

        Ok(BrowserStageProbe {
            stage,
            bytes,
            dtype: dtype.name().into(),
            elements,
            finite_elements,
            verified_objects,
        })
    }

    async fn vae_reference_1k5(
        &mut self,
        fixture: BrowserParityFixture,
        source_png: Vec<u8>,
        adapter: BrowserParityAdapterEvidence,
    ) -> Result<BrowserBooguVaeReferenceDiagnosticReport, RuntimeError> {
        let variant = self.identity.variant;
        let metadata = fixture.metadata().clone();
        if metadata.model_revision != self.identity.model_revision
            || metadata.upstream_source_revision != self.identity.upstream_source_revision
        {
            return Err(execution_error(
                variant,
                "fixture revisions differ from the sealed model release",
            ));
        }
        if adapter.adapter.device_type == wgpu::DeviceType::Cpu
            || matches!(self.device, burn_wgpu::WgpuDevice::Cpu)
        {
            return Err(execution_error(
                variant,
                "browser VAE reference diagnostic refuses a CPU WebGPU adapter/device",
            ));
        }

        let source_dimensions =
            Dimensions::new(256, 256).map_err(|error| execution_error(variant, error))?;
        let source = decode_input_image(&InputImage::Encoded(
            EncodedImage::new(ImageEncoding::Png, Some(source_dimensions), source_png)
                .map_err(|error| execution_error(variant, error))?,
        ))
        .map_err(|error| map_boogu(variant, error))?;
        let ledger = BrowserParityOracleLedger::default();
        let normalized = prepare_vae_reference::<BrowserBackend>(&source, &self.device)
            .map_err(|error| map_boogu(variant, error))?
            .cast(self.dtypes.vae);
        let epsilon = tensor4_from_fixture(
            &fixture,
            &ledger,
            "vae.reference_epsilon",
            self.dtypes.vae,
            &self.device,
        )
        .await?;
        let input = compare_browser_tensor(
            &fixture,
            &ledger,
            "vae.reference_input".into(),
            "vae.reference_input".into(),
            normalized.clone(),
        )
        .await
        .map_err(|error| map_boogu(variant, error))?;
        let injected_epsilon = compare_browser_tensor(
            &fixture,
            &ledger,
            "vae.reference_epsilon".into(),
            "vae.reference_epsilon".into(),
            epsilon.clone(),
        )
        .await
        .map_err(|error| map_boogu(variant, error))?;

        let artifact_ledger = BrowserVerifiedArtifactLedger::default();
        self.artifact_control
            .set_observer(Some(artifact_ledger.observer()));
        let mut runs = Vec::with_capacity(BROWSER_1K5_VAE_REFERENCE_REPEAT_COUNT);
        let mut baseline = BTreeMap::<String, (Vec<usize>, Vec<f32>)>::new();
        let mut stability_metrics =
            Vec::with_capacity((BROWSER_1K5_VAE_REFERENCE_REPEAT_COUNT - 1) * 6);
        let mut gate_failures = Vec::new();

        for index in 0..BROWSER_1K5_VAE_REFERENCE_REPEAT_COUNT {
            vae_reference_milestone(&format!("repeat-{index}-start"));
            let started = now_micros();
            let encoder = self
                .vae
                .load_encoder()
                .await
                .map_err(|error| map_boogu(variant, error))?;
            require_dtype(
                variant,
                "loaded diagnostic VAE encoder",
                encoder.encoder_float_dtype().into(),
                self.dtypes.vae,
            )?;
            let moments = encoder.encode_moments(normalized.clone());
            let posterior = DiagonalGaussian::from_moments(moments.clone());
            let mean = posterior.mean();
            let logvar = posterior.logvar();
            let std = posterior.std();
            let raw_latent = posterior.sample_with_epsilon(epsilon.clone());
            let scaled_latent = encoder.scale_latents(raw_latent.clone());
            self.vae
                .synchronize()
                .await
                .map_err(|error| map_boogu(variant, error))?;
            drop(encoder);

            let actual_reference = [
                ("moments", moments),
                ("mean", mean),
                ("logvar", logvar),
                ("std", std),
                ("raw_latent", raw_latent),
                ("scaled_latent", scaled_latent),
            ];
            let mut f32_oracle = Vec::with_capacity(actual_reference.len());
            let mut upstream_bf16_drift = Vec::with_capacity(actual_reference.len());
            for (component, actual) in actual_reference {
                let (shape, actual_dtype, actual_data) = read_browser_tensor_f32(actual)
                    .await
                    .map_err(|error| map_boogu(variant, error))?;
                f32_oracle.push(
                    compare_browser_f32_values(
                        &fixture,
                        &ledger,
                        format!("vae.reference_f32_{component}"),
                        format!("vae.reference_f32_{component}"),
                        &shape,
                        &actual_dtype,
                        &actual_data,
                    )
                    .await
                    .map_err(|error| map_boogu(variant, error))?,
                );
                upstream_bf16_drift.push(
                    compare_browser_f32_values(
                        &fixture,
                        &ledger,
                        format!("vae.reference_{component}"),
                        format!("vae.reference_{component}"),
                        &shape,
                        &actual_dtype,
                        &actual_data,
                    )
                    .await
                    .map_err(|error| map_boogu(variant, error))?,
                );

                if index == 0 {
                    baseline.insert(component.into(), (shape, actual_data));
                } else {
                    let (baseline_shape, baseline_values) =
                        baseline.get(component).ok_or_else(|| {
                            execution_error(
                                variant,
                                format!("VAE stability baseline omits {component}"),
                            )
                        })?;
                    if baseline_shape != &shape {
                        return Err(execution_error(
                            variant,
                            format!(
                                "VAE repeat {index} component {component} shape {shape:?} differs from baseline {baseline_shape:?}"
                            ),
                        ));
                    }
                    let comparison = compare_float(&actual_data, baseline_values)?;
                    let bitwise_exact = actual_data
                        .iter()
                        .zip(baseline_values)
                        .all(|(actual, expected)| actual.to_bits() == expected.to_bits());
                    stability_metrics.push(BrowserVaeReferenceStabilityMetric {
                        repeat_index: index,
                        component: component.into(),
                        shape,
                        bitwise_exact,
                        comparison,
                    });
                }
            }
            check_browser_1k5_vae_reference_run(index, &f32_oracle, &mut gate_failures);
            runs.push(BrowserVaeReferenceDiagnosticRun {
                index,
                elapsed_micros: now_micros().saturating_sub(started),
                f32_oracle,
                upstream_bf16_drift,
            });
            vae_reference_milestone(&format!("repeat-{index}-complete"));
        }
        self.artifact_control.set_observer(None);

        let all_bitwise_exact = stability_metrics.iter().all(|metric| metric.bitwise_exact);
        let maximum_abs = stability_metrics
            .iter()
            .map(|metric| metric.comparison.max_abs)
            .fold(0.0_f32, f32::max);
        let maximum_rmse = stability_metrics
            .iter()
            .map(|metric| metric.comparison.rmse)
            .fold(0.0_f32, f32::max);
        let minimum_cosine_similarity = stability_metrics
            .iter()
            .map(|metric| metric.comparison.cosine_similarity)
            .fold(1.0_f32, f32::min);
        let stability = BrowserVaeReferenceStabilityReport {
            baseline_repeat_index: 0,
            compared_repeats: BROWSER_1K5_VAE_REFERENCE_REPEAT_COUNT - 1,
            compared_components: stability_metrics.len(),
            all_bitwise_exact,
            maximum_abs,
            maximum_rmse,
            minimum_cosine_similarity,
            metrics: stability_metrics,
        };
        if !stability.all_bitwise_exact {
            gate_failures.push(format!(
                "three fresh VAE encoder executions were not bitwise stable: max={} rmse={} cosine={}",
                stability.maximum_abs,
                stability.maximum_rmse,
                stability.minimum_cosine_similarity
            ));
        }
        if input.comparison.max_abs != 0.0 || injected_epsilon.comparison.max_abs != 0.0 {
            gate_failures.push("VAE diagnostic input or injected epsilon is not exact".into());
        }

        let artifact_verification = vae_encoder_artifact_verification_report(
            &artifact_ledger,
            &self.expected_vae_encoder_weight_artifacts,
        );
        let artifacts_verified = artifact_verification.passed;
        if !artifacts_verified {
            gate_failures.push(
                "executed VAE encoder artifacts were not verified exactly once per repeat".into(),
            );
        }
        let fixture_verification = fixture.snapshot()?;
        let fixture_authenticated =
            fixture_verification.qualification_inputs_verified_for(&fixture.identity());
        if !fixture_authenticated {
            gate_failures.push("the exact 1.5K fixture was not fully authenticated".into());
        }
        let diagnostic_passed = artifacts_verified
            && fixture_authenticated
            && runs.len() == BROWSER_1K5_VAE_REFERENCE_REPEAT_COUNT
            && stability.all_bitwise_exact
            && gate_failures.is_empty();
        let peak_wasm_linear_memory_bytes = wasm_linear_memory_bytes().unwrap_or(0);
        Ok(BrowserBooguVaeReferenceDiagnosticReport {
            report_schema_version: 2,
            mode: "diagnostic-no-surface-exact-1k5-vae-reference-three-repeat".into(),
            model_backend: "raw-cubecl-no-fusion".into(),
            adapter_name: adapter.adapter.name,
            adapter_backend: format!("{:?}", adapter.backend),
            adapter_device_type: format!("{:?}", adapter.adapter.device_type),
            adapter_shader_f16: adapter.adapter_shader_f16,
            device_shader_f16: adapter.device_shader_f16,
            minimum_required_device_buffer_limit:
                crate::boogu::BOOGU_BROWSER_1K5_MIN_REQUIRED_BUFFER_LIMIT_BYTES,
            actual_storage_buffer_binding_size: adapter.limits.max_storage_buffer_binding_size,
            actual_max_buffer_size: adapter.limits.max_buffer_size,
            model: boogu_model_descriptor(variant).id.to_string(),
            model_revision: self.identity.model_revision.clone(),
            artifact_content_digest: self.artifact_content_digest,
            numeric_format: self.numeric_format.clone(),
            artifact_profile: "f16-qwen-vision-f32".into(),
            vae_float_load_policy: float_policy_name(self.policies.vae_float).into(),
            vae_execution_dtype: self.dtypes.vae.name().into(),
            expected_repeats: BROWSER_1K5_VAE_REFERENCE_REPEAT_COUNT,
            completed_repeats: runs.len(),
            fixture: fixture.identity(),
            fixture_verification,
            input,
            injected_epsilon,
            runs,
            stability,
            artifact_verification,
            browser_webgpu_vae_f32_oracle_envelope: BROWSER_WEBGPU_VAE_F32_ORACLE_ENVELOPE,
            gate_failures,
            peak_wasm_linear_memory_bytes,
            artifacts_verified,
            fixture_authenticated,
            diagnostic_passed,
            numerical_parity_claimed: false,
        })
    }

    async fn parity_1k5(
        &mut self,
        fixture: BrowserParityFixture,
        source_png: Vec<u8>,
        fixture_output_png_sha256: Sha256Digest,
        adapter: BrowserParityAdapterEvidence,
    ) -> Result<BrowserBooguParityReport, RuntimeError> {
        let variant = self.identity.variant;
        if !self.policies.retain_denoiser_stages || !self.denoiser.source().retention_enabled() {
            return Err(execution_error(
                variant,
                "exact browser 1.5K parity requires a retained-denoiser source",
            ));
        }
        if self.denoiser.source().cached_stage_count() != 0 {
            return Err(execution_error(
                variant,
                "exact browser 1.5K parity requires an initially empty denoiser cache",
            ));
        }
        if self.qwen.source.synchronization_policy()
            != AsyncRetainingSynchronizationPolicy::PerStage
            || self.denoiser.source().synchronization_policy()
                != AsyncRetainingDenoiserSynchronizationPolicy::PerStage
        {
            return Err(execution_error(
                variant,
                "exact browser 1.5K parity requires per-stage Qwen and denoiser barriers so observer activations are compared and released before the next stage",
            ));
        }
        let metadata = fixture.metadata().clone();
        if metadata.model_revision != self.identity.model_revision
            || metadata.upstream_source_revision != self.identity.upstream_source_revision
        {
            return Err(execution_error(
                variant,
                "fixture revisions differ from the sealed model release",
            ));
        }
        let ledger = BrowserParityOracleLedger::default();
        let parity_control = BrowserParityControl::new(fixture.clone(), ledger.clone());
        let artifact_ledger = BrowserVerifiedArtifactLedger::default();
        self.artifact_control
            .set_observer(Some(artifact_ledger.observer()));
        self.qwen
            .source
            .source_mut()
            .set_parity_control(parity_control.clone());
        self.denoiser
            .source_mut()
            .source_mut()
            .set_parity_control(parity_control.clone());

        let source_dimensions =
            Dimensions::new(256, 256).map_err(|error| execution_error(variant, error))?;
        let output_dimensions = Dimensions::new(metadata.width as u32, metadata.height as u32)
            .map_err(|error| execution_error(variant, error))?;
        let request = ImageRequest::Edit(EditRequest {
            source: InputImage::Encoded(
                EncodedImage::new(ImageEncoding::Png, Some(source_dimensions), source_png)
                    .map_err(|error| execution_error(variant, error))?,
            ),
            instruction: Prompt::new(metadata.prompt.clone())
                .map_err(|error| execution_error(variant, error))?,
            negative_prompt: None,
            mask: None,
            strength: None,
            options: GenerationOptions {
                dimensions: Some(output_dimensions),
                steps: Some(4),
                guidance_scale: Some(1.0),
                seed: Some(metadata.seed),
                batch_size: 1,
            },
        });
        let resolved = resolve_request(variant, &request, metadata.seed)
            .map_err(|error| map_boogu(variant, error))?;
        let source = resolved
            .source
            .as_ref()
            .map(decode_input_image)
            .transpose()
            .map_err(|error| map_boogu(variant, error))?
            .ok_or_else(|| execution_error(variant, "fixture edit source is absent"))?;

        parity_milestone("processing-start");
        let mut prepared = prepare_instruction::<BrowserBackend, HfTokenizer>(
            &resolved,
            Some(&source),
            &self.processor,
            &self.image_processor,
            &self.device,
        )
        .map_err(|error| map_boogu(variant, error))?;
        cast_visual_inputs(&mut prepared.model_input, self.dtypes.qwen_visual);
        let integer_tensors = compare_prepared_integer_plan(&fixture, &ledger, &prepared).await?;
        let patches = prepared
            .model_input
            .images
            .as_ref()
            .ok_or_else(|| execution_error(variant, "prepared edit input omits image patches"))?
            .patches
            .clone();
        let pixel_values = compare_browser_tensor(
            &fixture,
            &ledger,
            "processing.pixel_values".into(),
            "processor.pixel_values".into(),
            patches,
        )
        .await
        .map_err(|error| map_boogu(variant, error))?;
        let positions =
            prepared.model_input.position_ids.as_ref().ok_or_else(|| {
                execution_error(variant, "prepared edit input omits MRoPE positions")
            })?;
        let (rope_cos, rope_sin) = positions
            .cos_sin::<BrowserBackend>(&self.qwen_config.text_config, &self.device)
            .map_err(|error| execution_error(variant, error))?;
        let mrope_cos = compare_browser_tensor(
            &fixture,
            &ledger,
            "processing.mrope_cos".into(),
            "qwen.text.rope.0".into(),
            rope_cos,
        )
        .await
        .map_err(|error| map_boogu(variant, error))?;
        let mrope_sin = compare_browser_tensor(
            &fixture,
            &ledger,
            "processing.mrope_sin".into(),
            "qwen.text.rope.1".into(),
            rope_sin,
        )
        .await
        .map_err(|error| map_boogu(variant, error))?;
        let processing = BrowserParityProcessingReport {
            prompt_exact: resolved.prompt == metadata.prompt,
            dimensions_exact: resolved.dimensions == output_dimensions,
            seed_exact: resolved.seed == metadata.seed,
            effective_instruction_length: prepared.effective_length,
            integer_tensors,
            pixel_values,
            mrope_cos,
            mrope_sin,
        };

        parity_milestone("qwen-start");
        let qwen_metric_start = parity_control.metrics().len();
        let mut qwen_observer = BrowserQwenStageObserver::parity(
            parity_control.clone(),
            true,
            self.qwen_config
                .vision_config
                .deepstack_visual_indexes
                .len(),
        );
        let qwen_output = self
            .qwen
            .forward_base_async(&self.qwen_config, prepared.model_input, &mut qwen_observer)
            .await
            .map_err(|error| execution_error(variant, format!("{error:?}")))?;
        if parity_control.pending_count() != 0 {
            return Err(execution_error(
                variant,
                "browser Qwen parity observer still retains tensors after the final stage barrier",
            ));
        }
        if self.policies.residency == BrowserBooguResidencyPolicy::LowVramRuntimeQ8Denoiser {
            validate_low_vram_streamed_stage_lifecycle(
                variant,
                self.qwen.source.cached_stage_count(),
                self.qwen.source.has_pending_synchronization(),
                self.vae.cached_stage_count(),
            )?;
        }
        let aligned_stages = parity_control.metrics()[qwen_metric_start..].to_vec();
        let final_hidden_state = compare_browser_tensor(
            &fixture,
            &ledger,
            "conditioning.qwen_final_hidden_state".into(),
            "qwen.last_hidden_state".into(),
            qwen_output.last_hidden_state.clone(),
        )
        .await
        .map_err(|error| map_boogu(variant, error))?;
        let instruction =
            trim_instruction_features(qwen_output.last_hidden_state, prepared.effective_length)
                .map_err(|error| map_boogu(variant, error))?
                .cast(self.dtypes.denoiser);
        let authenticated_unaligned_diagnostics = fixture
            .tensor_specs()
            .keys()
            .filter(|name| name.starts_with("qwen.") && !ledger.contains(name))
            .cloned()
            .collect::<Vec<_>>();
        let aligned_stage_names = aligned_stages
            .iter()
            .map(|metric| metric.oracle.clone())
            .collect::<BTreeSet<_>>();
        let expected_aligned_stage_names = expected_qwen_aligned_stage_oracles();
        let authenticated_only_names = authenticated_unaligned_diagnostics
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        let expected_authenticated_only_names = expected_qwen_authenticated_only_oracles();
        let qwen = BrowserParityQwenReport {
            expected_aligned_stages: BROWSER_1K5_QWEN_ALIGNED_STAGE_COUNT,
            compared_aligned_stages: aligned_stages.len(),
            unique_compared_aligned_stages: aligned_stage_names.len(),
            aligned_stage_names_exact: aligned_stage_names == expected_aligned_stage_names,
            aligned_stages,
            final_hidden_state,
            expected_authenticated_only_diagnostics: BROWSER_1K5_AUTHENTICATED_ONLY_TENSOR_COUNT,
            authenticated_only_diagnostics_exact: authenticated_only_names
                == expected_authenticated_only_names,
            authenticated_unaligned_diagnostics,
        };

        parity_milestone("vae-reference-start");
        let normalized = prepare_vae_reference::<BrowserBackend>(&source, &self.device)
            .map_err(|error| map_boogu(variant, error))?
            .cast(self.dtypes.vae);
        let epsilon = tensor4_from_fixture(
            &fixture,
            &ledger,
            "vae.reference_epsilon",
            self.dtypes.vae,
            &self.device,
        )
        .await?;
        let reference_input = compare_browser_tensor(
            &fixture,
            &ledger,
            "vae.reference_input".into(),
            "vae.reference_input".into(),
            normalized.clone(),
        )
        .await
        .map_err(|error| map_boogu(variant, error))?;
        let injected_epsilon = compare_browser_tensor(
            &fixture,
            &ledger,
            "vae.reference_epsilon".into(),
            "vae.reference_epsilon".into(),
            epsilon.clone(),
        )
        .await
        .map_err(|error| map_boogu(variant, error))?;
        let encoder = self
            .vae
            .load_encoder()
            .await
            .map_err(|error| map_boogu(variant, error))?;
        let moments = encoder.encode_moments(normalized);
        let posterior = DiagonalGaussian::from_moments(moments.clone());
        let mean = posterior.mean();
        let logvar = posterior.logvar();
        let std = posterior.std();
        let raw_latent = posterior.sample_with_epsilon(epsilon);
        let scaled_latent = encoder.scale_latents(raw_latent.clone());
        self.vae
            .synchronize()
            .await
            .map_err(|error| map_boogu(variant, error))?;
        drop(encoder);
        let actual_reference = [
            ("moments", moments),
            ("mean", mean),
            ("logvar", logvar),
            ("std", std),
            ("raw_latent", raw_latent),
            ("scaled_latent", scaled_latent.clone()),
        ];
        let mut f32_oracle = Vec::with_capacity(actual_reference.len());
        let mut upstream_bf16_drift = Vec::with_capacity(actual_reference.len());
        for (name, actual) in actual_reference {
            f32_oracle.push(
                compare_browser_tensor(
                    &fixture,
                    &ledger,
                    format!("vae.reference_f32_{name}"),
                    format!("vae.reference_f32_{name}"),
                    actual.clone(),
                )
                .await
                .map_err(|error| map_boogu(variant, error))?,
            );
            upstream_bf16_drift.push(
                compare_browser_tensor(
                    &fixture,
                    &ledger,
                    format!("vae.reference_{name}"),
                    format!("vae.reference_{name}"),
                    actual,
                )
                .await
                .map_err(|error| map_boogu(variant, error))?,
            );
        }
        let vae_reference = BrowserParityVaeReferenceReport {
            input: reference_input,
            injected_epsilon,
            f32_oracle,
            upstream_bf16_drift,
        };
        let reference = Some(scaled_latent.cast(self.dtypes.denoiser));

        parity_milestone("dmd-start");
        if self.policies.residency == BrowserBooguResidencyPolicy::LowVramRuntimeQ8Denoiser {
            validate_low_vram_streamed_stage_lifecycle(
                variant,
                self.qwen.source.cached_stage_count(),
                self.qwen.source.has_pending_synchronization(),
                self.vae.cached_stage_count(),
            )?;
        }
        let mut latents = tensor4_from_fixture(
            &fixture,
            &ledger,
            "dmd.initial_latents",
            self.dtypes.denoiser,
            &self.device,
        )
        .await?;
        let initial_latent = compare_browser_tensor(
            &fixture,
            &ledger,
            "trajectory.initial_latent".into(),
            "dmd.initial_latents".into(),
            latents.clone(),
        )
        .await
        .map_err(|error| map_boogu(variant, error))?;
        let schedule = DmdSchedule::upstream_for_dtype(BooguTask::Edit, self.dtypes.denoiser);
        let denoiser_metric_start = parity_control.metrics().len();
        let mut steps = Vec::with_capacity(schedule.sigmas().len());
        for (index, &sigma) in schedule.sigmas().iter().enumerate() {
            parity_milestone(&format!("dmd-step-{index}-start"));
            let fixture_sigma =
                fixture_scalar(&fixture, &ledger, &format!("dmd.step.{index}.sigma")).await?;
            let input = compare_browser_tensor(
                &fixture,
                &ledger,
                format!("trajectory.step.{index}.input"),
                format!("dmd.step.{index}.input"),
                latents.clone(),
            )
            .await
            .map_err(|error| map_boogu(variant, error))?;
            let timestep = Tensor::<BrowserBackend, 1>::from_data(
                TensorData::new(vec![sigma], [1]),
                &self.device,
            )
            .cast(self.dtypes.denoiser);
            let mut observer = BrowserDenoiserStageObserver {
                control: parity_control.clone(),
                step: index,
            };
            let velocity = self
                .denoiser
                .predict_with_observer_async(
                    BooguDenoiserInput {
                        latent: latents.clone(),
                        timestep,
                        instruction: instruction.clone(),
                        reference: reference.clone(),
                    },
                    &mut observer,
                )
                .await
                .map_err(|error| map_boogu(variant, error))?;
            self.denoiser
                .source_mut()
                .synchronize_pending()
                .await
                .map_err(|error| map_boogu(variant, error))?;
            let velocity_metric = compare_browser_tensor(
                &fixture,
                &ledger,
                format!("trajectory.step.{index}.velocity"),
                format!("dmd.step.{index}.velocity"),
                velocity.clone(),
            )
            .await
            .map_err(|error| map_boogu(variant, error))?;
            let prediction = dmd_prediction(latents, velocity, sigma);
            let prediction_oracle = if index + 1 == schedule.sigmas().len() {
                "dmd.final_latents".to_owned()
            } else {
                format!("dmd.step.{index}.prediction")
            };
            let prediction_metric = compare_browser_tensor(
                &fixture,
                &ledger,
                format!("trajectory.step.{index}.prediction"),
                prediction_oracle,
                prediction.clone(),
            )
            .await
            .map_err(|error| map_boogu(variant, error))?;
            let (injected_noise, renoised) =
                if let Some(&next_sigma) = schedule.sigmas().get(index + 1) {
                    let noise_name = format!("dmd.step.{index}.noise");
                    let noise = tensor4_from_fixture(
                        &fixture,
                        &ledger,
                        &noise_name,
                        self.dtypes.denoiser,
                        &self.device,
                    )
                    .await?;
                    let noise_metric = compare_browser_tensor(
                        &fixture,
                        &ledger,
                        format!("trajectory.step.{index}.injected_noise"),
                        noise_name,
                        noise.clone(),
                    )
                    .await
                    .map_err(|error| map_boogu(variant, error))?;
                    latents = dmd_renoise(prediction, noise, next_sigma);
                    let renoised_metric = compare_browser_tensor(
                        &fixture,
                        &ledger,
                        format!("trajectory.step.{index}.renoised"),
                        format!("dmd.step.{index}.renoised"),
                        latents.clone(),
                    )
                    .await
                    .map_err(|error| map_boogu(variant, error))?;
                    (Some(noise_metric), Some(renoised_metric))
                } else {
                    latents = prediction;
                    (None, None)
                };
            steps.push(BrowserParityDmdStepReport {
                index,
                schedule_sigma: sigma,
                fixture_sigma,
                sigma_exact: sigma.to_bits() == fixture_sigma.to_bits(),
                input,
                velocity: velocity_metric,
                prediction: prediction_metric,
                injected_noise,
                renoised,
            });
        }
        let final_latent = compare_browser_tensor(
            &fixture,
            &ledger,
            "trajectory.final_latent".into(),
            "dmd.final_latents".into(),
            latents.clone(),
        )
        .await
        .map_err(|error| map_boogu(variant, error))?;
        if parity_control.pending_count() != 0 {
            return Err(execution_error(
                variant,
                "browser denoiser parity observer still retains tensors after the final stage barrier",
            ));
        }
        let denoiser_boundaries = parity_control.metrics()[denoiser_metric_start..].to_vec();
        self.denoiser
            .source_mut()
            .synchronize_pending()
            .await
            .map_err(|error| map_boogu(variant, error))?;
        let denoiser_retained_stages_before_clear = self.denoiser.source().cached_stage_count();
        let denoiser_synchronization_pending = self.denoiser.source().has_pending_synchronization();
        let low_vram_denoiser_dtype_audit =
            if self.policies.residency == BrowserBooguResidencyPolicy::LowVramRuntimeQ8Denoiser {
                Some(validate_low_vram_denoiser_dtype_audit(
                    variant,
                    self.low_vram_resource_plan.ok_or_else(|| {
                        execution_error(variant, "browser low-vram resource plan is absent")
                    })?,
                    self.denoiser.source().retained_dtype_audit(),
                )?)
            } else {
                None
            };
        self.denoiser.source_mut().clear();
        self.denoiser.clear_rope_cache();
        let denoiser_retained_stages_after_clear = self.denoiser.source().cached_stage_count();
        let denoiser_cache_cleared_before_decode = denoiser_retained_stages_after_clear == 0;
        if self.policies.residency == BrowserBooguResidencyPolicy::LowVramRuntimeQ8Denoiser {
            validate_low_vram_denoiser_lifecycle(
                variant,
                steps.len(),
                BROWSER_1K5_DENOISER_RETAINED_STAGE_COUNT,
                denoiser_retained_stages_before_clear,
                denoiser_synchronization_pending,
                0,
                denoiser_retained_stages_after_clear,
            )?;
        } else if denoiser_retained_stages_before_clear != BROWSER_1K5_DENOISER_RETAINED_STAGE_COUNT
            || denoiser_synchronization_pending
            || !denoiser_cache_cleared_before_decode
        {
            return Err(execution_error(
                variant,
                format!(
                    "browser parity denoiser lifecycle failed: retained={denoiser_retained_stages_before_clear}/{}, synchronization_pending={denoiser_synchronization_pending}, cleared={denoiser_cache_cleared_before_decode}",
                    BROWSER_1K5_DENOISER_RETAINED_STAGE_COUNT
                ),
            ));
        }
        parity_milestone("dmd-resident-denoiser-cache-cleared");
        let dmd = BrowserParityDmdReport {
            initial_latent,
            steps,
            final_latent,
        };

        parity_milestone("vae-decode-start");
        parity_milestone("vae-decoder-load-start");
        let decoder = self
            .vae
            .load_decoder()
            .await
            .map_err(|error| map_boogu(variant, error))?;
        parity_milestone("vae-decoder-load-complete");
        let decode_input_tensor = decoder.unscale_latents(latents.cast(self.dtypes.vae));
        parity_milestone("vae-decode-input-parity-start");
        let decode_input = compare_browser_tensor(
            &fixture,
            &ledger,
            "vae.decode_input".into(),
            "vae.decode_input".into(),
            decode_input_tensor.clone(),
        )
        .await
        .map_err(|error| map_boogu(variant, error))?;
        parity_milestone("vae-decode-input-parity-complete");
        parity_milestone("vae-forward-submit-start");
        let decoded = decoder.decode_striped_tail_strict_f32(
            decode_input_tensor,
            usize::try_from(crate::boogu::BOOGU_BROWSER_1K5_PARITY_OUTPUT_EDGE / 2)
                .expect("fixed browser parity split fits usize"),
        );
        parity_milestone("vae-forward-submitted");
        parity_milestone("vae-post-forward-sync-start");
        self.vae
            .synchronize()
            .await
            .map_err(|error| map_boogu(variant, error))?;
        parity_milestone("vae-post-forward-sync-complete");
        drop(decoder);
        parity_milestone("vae-output-readback-start");
        let (decoded_shape, decoded_dtype, decoded_data) =
            read_browser_tensor_f32(decoded)
                .await
                .map_err(|error| map_boogu(variant, error))?;
        parity_milestone("vae-output-readback-complete");
        parity_milestone("vae-output-parity-start");
        let decoded_tensor = compare_browser_f32_values(
            &fixture,
            &ledger,
            "full_chain_output.decoded_tensor".into(),
            "vae.decode_output".into(),
            &decoded_shape,
            &decoded_dtype,
            &decoded_data,
        )
        .await
        .map_err(|error| map_boogu(variant, error))?;
        parity_milestone("vae-output-parity-complete");
        parity_milestone("vae-output-rgb-conversion-start");
        let HostImage::Pixels(actual_rgb) =
            decoder_output_data_to_host(TensorData::new(decoded_data, decoded_shape.clone()))
                .map_err(|error| map_boogu(variant, error))?
        else {
            return Err(execution_error(
                variant,
                "browser decoder unexpectedly returned encoded pixels",
            ));
        };
        parity_milestone("vae-output-rgb-conversion-complete");
        let (rgb_shape, expected_rgb) = fixture.u8("output.rgb_u8").await?;
        ledger.record("output.rgb_u8");
        if rgb_shape != [metadata.height, metadata.width, 3] {
            return Err(execution_error(
                variant,
                format!("output.rgb_u8 has invalid shape {rgb_shape:?}"),
            ));
        }
        let final_rgb = compare_rgb(
            actual_rgb.bytes(),
            &expected_rgb,
            metadata.width,
            metadata.height,
        )?;
        let fixture_verification = fixture.snapshot()?;
        let fixture_authenticated = fixture_verification.qualification_inputs_verified();
        let denoiser_boundary_names = denoiser_boundaries
            .iter()
            .map(|metric| metric.oracle.clone())
            .collect::<BTreeSet<_>>();
        let denoiser_boundary_names_exact =
            denoiser_boundary_names == expected_denoiser_boundary_oracles();
        let actual_numerical_names = ledger.names();
        let authenticated_only_names = expected_qwen_authenticated_only_oracles();
        let fixture_names = fixture
            .tensor_specs()
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();
        let expected_numerical_names = fixture_names
            .difference(&authenticated_only_names)
            .cloned()
            .collect::<BTreeSet<_>>();
        let numerical_name_set_exact = actual_numerical_names == expected_numerical_names;
        let actual_authenticated_only_names = fixture_names
            .difference(&actual_numerical_names)
            .cloned()
            .collect::<BTreeSet<_>>();
        let authenticated_only_name_set_exact =
            actual_authenticated_only_names == authenticated_only_names;
        let tensor_coverage = BrowserParityTensorCoverageReport {
            fixture_tensors: fixture_names.len(),
            expected_numerically_compared_semantic_tensors: BROWSER_1K5_NUMERICAL_TENSOR_COUNT,
            unique_numerically_compared_semantic_tensors: actual_numerical_names.len(),
            expected_authenticated_only_tensors: BROWSER_1K5_AUTHENTICATED_ONLY_TENSOR_COUNT,
            numerical_name_set_exact,
            authenticated_only_name_set_exact,
            scope: "355/372 fixture tensors are compared at public semantic boundaries; 17 Qwen exporter debug captures are authenticated-only because no public semantic observer exposes them"
                .into(),
        };
        let artifact_verification = artifact_ledger.report(
            &self.expected_weight_artifacts,
            &self.verified_runtime_metadata,
        );
        self.artifact_control.set_observer(None);
        let artifacts_verified = artifact_verification.passed;
        let mut gates = evaluate_browser_1k5_gates(
            self.policies.residency,
            &processing,
            &qwen,
            &vae_reference,
            &denoiser_boundaries,
            &dmd,
            &decode_input,
            &decoded_tensor,
            &final_rgb,
        );
        if !denoiser_boundary_names_exact {
            gates
                .failures
                .push("denoiser boundary names differ from the exact 236-name contract".into());
        }
        if !tensor_coverage.numerical_name_set_exact
            || !tensor_coverage.authenticated_only_name_set_exact
        {
            gates.failures.push(
                "fixture coverage differs from the exact 355 numerical / 17 authenticated-only semantic-boundary contract"
                    .into(),
            );
        }
        if !artifact_verification.passed {
            gates.failures.push(
                "executed model artifact verification did not cover the exact manifest weight inventory"
                    .into(),
            );
        }
        gates.passed = gates.failures.is_empty();
        let complete_counts = qwen.aligned_stage_names_exact
            && qwen.authenticated_only_diagnostics_exact
            && denoiser_boundary_names_exact
            && numerical_name_set_exact
            && authenticated_only_name_set_exact;
        let numerical_parity_claimed = artifacts_verified
            && fixture_authenticated
            && complete_counts
            && gates.passed
            && denoiser_retained_stages_before_clear == BROWSER_1K5_DENOISER_RETAINED_STAGE_COUNT
            && denoiser_cache_cleared_before_decode
            && adapter.adapter.device_type != wgpu::DeviceType::Cpu;
        let peak_wasm_linear_memory_bytes = wasm_linear_memory_bytes().unwrap_or(0);
        Ok(BrowserBooguParityReport {
            report_schema_version: 2,
            mode: "qualification-no-surface-exact-1k5-fixture".into(),
            model_backend: "raw-cubecl-no-fusion".into(),
            adapter_name: adapter.adapter.name,
            adapter_backend: format!("{:?}", adapter.backend),
            adapter_device_type: format!("{:?}", adapter.adapter.device_type),
            adapter_shader_f16: adapter.adapter_shader_f16,
            device_shader_f16: adapter.device_shader_f16,
            minimum_required_device_buffer_limit:
                crate::boogu::BOOGU_BROWSER_1K5_MIN_REQUIRED_BUFFER_LIMIT_BYTES,
            actual_storage_buffer_binding_size: adapter.limits.max_storage_buffer_binding_size,
            actual_max_buffer_size: adapter.limits.max_buffer_size,
            model: boogu_model_descriptor(variant).id.to_string(),
            model_revision: self.identity.model_revision.clone(),
            artifact_content_digest: self.artifact_content_digest,
            numeric_format: self.numeric_format.clone(),
            artifact_profile: "f16-qwen-vision-f32".into(),
            residency_policy: self.policies.residency.label().into(),
            qwen_float_load_policy: float_policy_name(self.policies.qwen_float).into(),
            vae_float_load_policy: float_policy_name(self.policies.vae_float).into(),
            denoiser_float_load_policy: float_policy_name(self.policies.denoiser_float).into(),
            denoiser_storage_policy: self.policies.denoiser_storage_policy().into(),
            denoiser_quantized_load_policy: self
                .policies
                .denoiser_quantized_load_policy_report()
                .into(),
            denoiser_quantized_linear_execution_policy: self
                .policies
                .denoiser_quantized_linear_execution_policy_report()
                .into(),
            denoiser_linear_execution_policy: self
                .policies
                .denoiser_linear_execution_policy()
                .into(),
            qwen_execution_dtype: self.dtypes.qwen_visual.name().into(),
            vae_execution_dtype: self.dtypes.vae.name().into(),
            denoiser_execution_dtype: self.dtypes.denoiser.name().into(),
            qwen_query_chunk_size: BROWSER_1K5_QWEN_QUERY_CHUNK_SIZE,
            vae_attention_query_chunk_size: BROWSER_1K5_VAE_QUERY_CHUNK_SIZE,
            vae_decode_policy: "exact-two-width-slabs-global-groupnorm".into(),
            vae_decode_max_planned_buffer_bytes:
                crate::boogu::BOOGU_BROWSER_1K5_VAE_STRIPED_TAIL_BUFFER_BYTES,
            denoiser_query_chunk_size: BROWSER_1K5_DENOISER_QUERY_CHUNK_SIZE,
            denoiser_residency: match self.policies.residency {
                BrowserBooguResidencyPolicy::LowVramRuntimeQ8Denoiser => {
                    "request-scoped-runtime-q8-policy-retained-through-four-dmd-steps"
                }
                BrowserBooguResidencyPolicy::LowVramPreloadedPackedF16Denoiser => {
                    "turbo-only-preloaded-packed-f16-dense-f32-policy-not-valid-for-1k5-parity"
                }
                BrowserBooguResidencyPolicy::QualificationPerRequestF32DenoiserRetained => {
                    "request-scoped-f32-policy-retained-through-four-dmd-steps"
                }
                BrowserBooguResidencyPolicy::HighVramResidentPackedF16 => {
                    "all-stages-preloaded-packed-f16-fused-f32-accumulate"
                }
                BrowserBooguResidencyPolicy::ResidentPackedQ4s => {
                    "turbo-only-resident-packed-q4s-policy-not-valid-for-1k5-parity"
                }
                BrowserBooguResidencyPolicy::HighVramResidentDenseF32 => {
                    "all-stages-preloaded-dense-f32"
                }
            }
            .into(),
            denoiser_expected_retained_stages: BROWSER_1K5_DENOISER_RETAINED_STAGE_COUNT,
            denoiser_retained_stages_before_clear,
            denoiser_cache_cleared_before_decode,
            resident_resource_plan: self.resident_resource_plan,
            low_vram_resource_plan: self.low_vram_resource_plan,
            low_vram_denoiser_dtype_audit,
            weight_traffic_contract: self.policies.weight_traffic_contract().into(),
            // Runtime Q8 policy is observable here; a measured packed-kernel execution claim is
            // deliberately withheld until the qualification harness captures that evidence.
            on_device_quantized_execution_claimed: false,
            fixture: fixture.identity(),
            fixture_verification,
            tensor_coverage,
            artifact_verification,
            processing,
            qwen,
            vae_reference,
            denoiser_expected_boundaries: BROWSER_1K5_DENOISER_BOUNDARY_COUNT,
            denoiser_compared_boundaries: denoiser_boundaries.len(),
            denoiser_unique_compared_boundaries: denoiser_boundary_names.len(),
            denoiser_boundary_names_exact,
            denoiser_boundaries,
            dmd,
            decode_input,
            decoded_tensor,
            final_rgb,
            fixture_output_png_sha256,
            peak_wasm_linear_memory_bytes,
            gates,
            artifacts_verified,
            fixture_authenticated,
            numerical_parity_claimed,
        })
    }

    async fn infer(
        &mut self,
        job: &BooguRuntimeJob,
        cancellation: &CancellationToken,
        shared: &Arc<Mutex<BrowserRuntimeShared>>,
    ) -> Result<ImageOutput, RuntimeError> {
        self.last_low_vram_denoiser_dtype_audit = None;
        self.last_packed_f16_qwen_host_embedding = None;
        self.last_packed_f16_qwen_instruction_handoff = None;
        self.last_packed_f16_denoiser_lifecycle = None;
        self.last_packed_f16_dmd_vae_handoff = None;
        self.last_dense_f32_materialized_stage_clones = 0;
        self.last_artifact_traffic = BrowserArtifactTrafficReport::default();
        let artifact_traffic_before = self.artifact_control.traffic_snapshot();
        let id = job.id;
        let run_id = RunId(id.0);
        let total_started = now_micros();
        queue_progress(
            shared,
            id,
            ProgressEvent::RunStarted {
                run_id,
                model: boogu_model_descriptor(job.variant).id,
                task: task_kind(job.task),
            },
        );
        check_cancelled(cancellation)?;
        self.ensure_preloaded_packed_f16_denoiser().await?;
        self.validate_resident_caches()?;
        self.report_resident_cache_audit(run_id, "before-request");
        let buffer_plan = crate::boogu::validate_browser_buffer_limits_for_dimensions(
            job.variant,
            job.resolved.dimensions,
            self.applied_buffer_limits.max_storage_buffer_binding_size,
            self.applied_buffer_limits.max_buffer_size,
        )?;
        check_cancelled(cancellation)?;
        let source = job
            .resolved
            .source
            .as_ref()
            .map(decode_input_image)
            .transpose()
            .map_err(|error| map_boogu(job.variant, error))?;

        let mut timings = Vec::new();
        let started = start_stage(shared, id, run_id, "processing", Some(1));
        let mut prepared = prepare_instruction::<BrowserBackend, HfTokenizer>(
            &job.resolved,
            source.as_ref(),
            &self.processor,
            &self.image_processor,
            &self.device,
        )
        .map_err(|error| map_boogu(job.variant, error))?;
        cast_visual_inputs(&mut prepared.model_input, self.dtypes.qwen_visual);
        finish_stage(shared, id, run_id, &mut timings, "processing", started);

        check_cancelled(cancellation)?;
        let started = start_stage(shared, id, run_id, "qwen", Some(1));
        let packed_rendered_diagnostics_requested =
            self.policies.uses_packed_f16_denoiser_source() && rendered_model_smoke_requested();
        let block0_execution_mode = browser_qwen_block0_execution_mode();
        if packed_rendered_diagnostics_requested
            && !matches!(
                block0_execution_mode,
                BROWSER_QWEN_BLOCK0_SERIALIZED_DIAGNOSTIC_MODE | BROWSER_QWEN_BLOCK0_ORDINARY_MODE
            )
        {
            return Err(execution_error(
                job.variant,
                format!(
                    "rendered-smoke query {BROWSER_QWEN_BLOCK0_EXECUTION_MODE_QUERY} must be {} or {}",
                    BROWSER_QWEN_BLOCK0_SERIALIZED_DIAGNOSTIC_MODE,
                    BROWSER_QWEN_BLOCK0_ORDINARY_MODE,
                ),
            ));
        }
        let qwen_instruction_diagnostic_control = packed_rendered_diagnostics_requested
            .then(BrowserQwenInstructionDiagnosticControl::default);
        let mut qwen_observer = BrowserQwenStageObserver::milestones_only();
        if let Some(control) = qwen_instruction_diagnostic_control.as_ref() {
            qwen_observer = qwen_observer.with_instruction_diagnostics(
                control.clone(),
                job.variant,
                run_id,
                self.qwen.text_layer_allocation_policy(),
                self.qwen.text_block_load_synchronization_policy(),
                self.policies.qwen_text_layer_submission_policy(),
            );
        }
        let qwen_host_input_ids = prepared.encoding.input_ids.clone();
        let qwen_output_result = if self.policies.uses_packed_f16_denoiser_source() {
            self.qwen
                .forward_base_async_with_host_input_ids(
                    &self.qwen_config,
                    prepared.model_input,
                    &qwen_host_input_ids,
                    &mut qwen_observer,
                )
                .await
        } else {
            self.qwen
                .forward_base_async(&self.qwen_config, prepared.model_input, &mut qwen_observer)
                .await
        };
        // The host-routed embedding report is produced before the streamed text layers. Take and
        // dispatch it before propagating a later layer failure so rendered-smoke failure evidence
        // still binds the healthy host/device embedding to the same run as the immediate block-0
        // diagnostic. This is provenance only; it does not turn a failed Qwen stage into success.
        let host_embedding_report = self.qwen.take_last_host_routed_embedding_report();
        match (
            self.policies.uses_packed_f16_denoiser_source(),
            self.packed_f16_qwen_embedding_plan,
            host_embedding_report,
        ) {
            (true, Some(plan), Some(report)) => {
                validate_browser_packed_f16_qwen_embedding_report(job.variant, plan, &report)?;
                dispatch_browser_event(
                    BROWSER_RUNTIME_EVENT_NAME,
                    &browser_packed_f16_qwen_host_embedding_event(run_id, report.clone()),
                );
                self.last_packed_f16_qwen_host_embedding = Some(report);
            }
            (false, None, None) => {}
            // A streamed artifact/load error can happen before the host-routed embedding has
            // been assembled and reported. In that case there is no embedding evidence to
            // validate, so preserve the underlying Qwen error below instead of replacing it
            // with a misleading policy-admission failure. Later-layer failures still take the
            // Some(report) arm above and retain the same-run embedding provenance.
            (true, Some(_), None) if qwen_output_result.is_err() => {}
            _ => {
                return Err(execution_error(
                    job.variant,
                    "browser Qwen host-embedding evidence differs from the admitted execution policy",
                ));
            }
        }
        let qwen_output = qwen_output_result.map_err(|error| {
            if cancellation.is_cancelled() {
                RuntimeError::Cancelled
            } else {
                execution_error(job.variant, format!("{error:?}"))
            }
        })?;
        self.qwen
            .source
            .synchronize_pending()
            .await
            .map_err(|error| map_boogu(job.variant, error))?;
        let burn_qwen3_vl::Qwen3VlModelOutput {
            last_hidden_state: qwen_last_hidden_state,
            hidden_states: qwen_hidden_states,
            vision_output: qwen_vision_output,
            position_deltas: qwen_position_deltas,
        } = qwen_output;
        if self.policies.uses_packed_f16_denoiser_source()
            && (qwen_hidden_states.is_some() || qwen_vision_output.is_some())
        {
            return Err(execution_error(
                job.variant,
                "ordinary Turbo Qwen unexpectedly retained hidden-state or vision-output tensors",
            ));
        }
        // Only the final hidden state crosses into conditioning. Release all optional Qwen tensor
        // roots explicitly before the later DMD-to-VAE allocator boundary.
        drop(qwen_hidden_states);
        drop(qwen_vision_output);
        drop(qwen_position_deltas);
        if self.policies.residency.is_low_vram() {
            validate_low_vram_streamed_stage_lifecycle(
                job.variant,
                self.qwen.source.cached_stage_count(),
                self.qwen.source.has_pending_synchronization(),
                self.vae.cached_stage_count(),
            )?;
        }
        let qwen_stage_diagnostics =
            if let Some(control) = qwen_instruction_diagnostic_control.as_ref() {
                let diagnostics = control.read_all(job.variant).await?;
                if control.pending_count() != 0 {
                    return Err(execution_error(
                        job.variant,
                        "rendered-smoke Qwen diagnostic tensors remain pending after readback",
                    ));
                }
                Some(diagnostics)
            } else {
                None
            };
        let qwen_block_00_immediate_post_sync =
            if let Some(control) = qwen_instruction_diagnostic_control.as_ref() {
                Some(control.take_immediate_post_sync_block0().ok_or_else(|| {
                    execution_error(
                        job.variant,
                        "rendered-smoke Qwen immediate post-sync block-0 diagnostic is absent",
                    )
                })?)
            } else {
                None
            };
        let qwen_last_hidden_state_before_trim = if packed_rendered_diagnostics_requested {
            Some(
                read_packed_f16_tensor_input_diagnostic(
                    job.variant,
                    "qwen_last_hidden_state_before_trim",
                    &qwen_last_hidden_state,
                )
                .await?,
            )
        } else {
            None
        };
        let instruction =
            trim_instruction_features(qwen_last_hidden_state, prepared.effective_length)
                .map_err(|error| map_boogu(job.variant, error))?
                .cast(self.dtypes.denoiser);
        let qwen_pre_handoff_context = match (
            qwen_stage_diagnostics,
            qwen_last_hidden_state_before_trim,
            qwen_block_00_immediate_post_sync,
        ) {
            (
                Some(stage_outputs),
                Some(qwen_last_hidden_state_before_trim),
                Some(block_00_immediate_post_sync),
            ) => Some(BrowserPackedF16QwenPreHandoffContext {
                effective_instruction_length: prepared.effective_length,
                expected_stage_output_count: self
                    .qwen_config
                    .text_config
                    .num_hidden_layers
                    .checked_add(2)
                    .ok_or_else(|| {
                        execution_error(
                            job.variant,
                            "rendered-smoke Qwen diagnostic stage count overflowed",
                        )
                    })?,
                stage_outputs,
                qwen_last_hidden_state_before_trim,
                block_00_immediate_post_sync,
            }),
            (None, None, None) => None,
            _ => {
                return Err(execution_error(
                    job.variant,
                    "rendered-smoke Qwen diagnostic capture is internally incomplete",
                ));
            }
        };
        check_cancelled(cancellation)?;
        finish_stage(shared, id, run_id, &mut timings, "qwen", started);
        let (instruction, packed_audit_after_qwen_handoff, qwen_handoff_report) = self
            .packed_f16_qwen_instruction_handoff(instruction, run_id, qwen_pre_handoff_context)
            .await?;
        self.last_packed_f16_qwen_instruction_handoff = qwen_handoff_report;
        if self.policies.phase_boundary_memory_cleanup {
            release_browser_phase_allocator(job.variant, &self.device, "Qwen-to-DMD handoff")
                .await?;
        }
        check_cancelled(cancellation)?;

        let reference = if let Some(source) = source.as_ref() {
            let started = start_stage(shared, id, run_id, "vae-encode", Some(1));
            let normalized = prepare_vae_reference::<BrowserBackend>(source, &self.device)
                .map_err(|error| map_boogu(job.variant, error))?
                .cast(self.dtypes.vae);
            let [_, _, height, width] = normalized.dims();
            let epsilon = normal_tensor::<4>(
                [1, 16, height / 8, width / 8],
                domain_seed(job.resolved.seed, 0x5641_452d_454e_434f),
                self.dtypes.vae,
                &self.device,
            );
            let encoder = self
                .vae
                .load_encoder()
                .await
                .map_err(|error| map_boogu(job.variant, error))?;
            let loaded_dtype: DType = encoder.float_dtype().into();
            require_dtype(
                job.variant,
                "loaded VAE encoder",
                loaded_dtype,
                self.dtypes.vae,
            )?;
            let reference = encode_reference(&encoder, normalized, epsilon)
                .map_err(|error| map_boogu(job.variant, error))?;
            self.vae
                .synchronize()
                .await
                .map_err(|error| map_boogu(job.variant, error))?;
            drop(encoder);
            check_cancelled(cancellation)?;
            finish_stage(shared, id, run_id, &mut timings, "vae-encode", started);
            Some(reference.cast(self.dtypes.denoiser))
        } else {
            None
        };

        let latent_shape = [
            1,
            16,
            job.resolved.dimensions.height() as usize / 8,
            job.resolved.dimensions.width() as usize / 8,
        ];
        let latents = normal_tensor::<4>(
            latent_shape,
            domain_seed(job.resolved.seed, 0x444d_442d_494e_4954),
            self.dtypes.denoiser,
            &self.device,
        );
        let renoise = (0..3)
            .map(|index| {
                normal_tensor::<4>(
                    latent_shape,
                    domain_seed(job.resolved.seed, 0x444d_442d_4e4f_4953 ^ index as u64),
                    self.dtypes.denoiser,
                    &self.device,
                )
            })
            .collect::<Vec<_>>();
        require_dtype(
            job.variant,
            "instruction conditioning",
            instruction.dtype(),
            self.dtypes.denoiser,
        )?;
        if let Some(reference) = &reference {
            require_dtype(
                job.variant,
                "reference conditioning",
                reference.dtype(),
                self.dtypes.denoiser,
            )?;
        }

        check_cancelled(cancellation)?;
        let started = start_stage(shared, id, run_id, "dmd", Some(4));
        let schedule = DmdSchedule::upstream_for_dtype(job.task, self.dtypes.denoiser);
        let packed_input_diagnostics_requested = packed_rendered_diagnostics_requested;
        let mut first_dmd_timestep = None;
        if packed_input_diagnostics_requested {
            let audit = packed_audit_after_qwen_handoff.ok_or_else(|| {
                execution_error(
                    job.variant,
                    "rendered-smoke packed-F16 input diagnostics lack the post-Qwen cache audit",
                )
            })?;
            self.require_exact_packed_f16_cache_audit(
                "rendered-smoke pre-DMD input readback",
                audit,
            )?;
            let first_sigma = *schedule.sigmas().first().ok_or_else(|| {
                execution_error(job.variant, "the DMD schedule contains no timestep")
            })?;
            let timestep = Tensor::<BrowserBackend, 1>::from_data(
                TensorData::new(vec![first_sigma], [1]),
                &self.device,
            )
            .cast(self.dtypes.denoiser);
            let instruction_diagnostic =
                read_packed_f16_tensor_input_diagnostic(job.variant, "instruction", &instruction)
                    .await?;
            let initial_latent_diagnostic =
                read_packed_f16_tensor_input_diagnostic(job.variant, "initial_latent", &latents)
                    .await?;
            let mut renoise_diagnostics = Vec::with_capacity(renoise.len());
            for (index, noise) in renoise.iter().enumerate() {
                renoise_diagnostics.push(
                    read_packed_f16_tensor_input_diagnostic(
                        job.variant,
                        &format!("renoise_{index}"),
                        noise,
                    )
                    .await?,
                );
            }
            let first_timestep_diagnostic =
                read_packed_f16_tensor_input_diagnostic(job.variant, "first_timestep", &timestep)
                    .await?;
            let all_inputs_finite = instruction_diagnostic.all_finite
                && initial_latent_diagnostic.all_finite
                && renoise_diagnostics
                    .iter()
                    .all(|diagnostic| diagnostic.all_finite)
                && first_timestep_diagnostic.all_finite;
            let diagnostics = BrowserPackedF16PreDmdInputDiagnostics {
                scope: "rendered-model-smoke/ordinary-turbo-packed-f16/pre-dmd-input-readback"
                    .into(),
                policy: BrowserPackedF16PreDmdPolicyEvidence {
                    qwen_release_unused_memory_after_stage: self
                        .qwen
                        .releases_unused_memory_after_stage(),
                    qwen_text_block_load_synchronization_policy: self
                        .qwen
                        .text_block_load_synchronization_policy()
                        .label()
                        .into(),
                    qwen_text_layer_submission_policy: self
                        .policies
                        .qwen_text_layer_submission_policy()
                        .into(),
                    packed_qwen_instruction_handoff_policy: self
                        .policies
                        .packed_qwen_instruction_handoff_policy()
                        .into(),
                    cleanup_completed: true,
                    post_cleanup_packed_cache: packed_f16_cache_evidence(audit),
                },
                dmd_steps: schedule.sigmas().len(),
                instruction: instruction_diagnostic,
                initial_latent: initial_latent_diagnostic,
                renoise: renoise_diagnostics,
                first_timestep: first_timestep_diagnostic,
                all_inputs_finite,
            };
            dispatch_browser_event(
                BROWSER_RUNTIME_EVENT_NAME,
                &browser_packed_f16_pre_dmd_input_diagnostics_event(run_id, diagnostics.clone()),
            );
            if !diagnostics.all_inputs_finite {
                let invalid = std::iter::once(&diagnostics.instruction)
                    .chain(std::iter::once(&diagnostics.initial_latent))
                    .chain(diagnostics.renoise.iter())
                    .chain(std::iter::once(&diagnostics.first_timestep))
                    .filter(|diagnostic| !diagnostic.all_finite)
                    .map(|diagnostic| diagnostic.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(execution_error(
                    job.variant,
                    format!(
                        "rendered-smoke packed-F16 pre-DMD inputs contain non-finite values: {invalid}"
                    ),
                ));
            }
            let all_zero = std::iter::once(&diagnostics.instruction)
                .chain(std::iter::once(&diagnostics.initial_latent))
                .chain(diagnostics.renoise.iter())
                .chain(std::iter::once(&diagnostics.first_timestep))
                .find(|diagnostic| packed_f16_tensor_diagnostic_is_all_zero(diagnostic));
            if let Some(diagnostic) = all_zero {
                return Err(execution_error(
                    job.variant,
                    format!(
                        "rendered-smoke packed-F16 pre-DMD input {} is all-zero",
                        diagnostic.name
                    ),
                ));
            }
            let handoff = self
                .last_packed_f16_qwen_instruction_handoff
                .as_ref()
                .ok_or_else(|| {
                    execution_error(
                        job.variant,
                        "rendered-smoke packed-F16 input diagnostics lack Qwen handoff provenance",
                    )
                })?;
            if diagnostics.instruction.sha256 != handoff.after_sha256
                || diagnostics.instruction.shape != handoff.shape
                || diagnostics.instruction.element_count != handoff.element_count
            {
                return Err(execution_error(
                    job.variant,
                    "packed-F16 instruction changed after the verified Qwen handoff and before DMD",
                ));
            }
            first_dmd_timestep = Some(timestep);
        }
        let mut noises = renoise.into_iter();
        let low_vram = self.policies.residency.is_low_vram();
        let dmd_artifact_traffic_before = self.artifact_control.traffic_snapshot();
        let dense_stage_clones_before_dmd =
            self.denoiser.source().dense_f32_materialized_stage_clones();
        let packed_audit_before_dmd = if self.policies.uses_packed_f16_denoiser_source() {
            let audit = match self
                .packed_f16_denoiser_source()
                .map(|source| source.audit())
                .and_then(|audit| {
                    self.require_exact_packed_f16_cache_audit("DMD entry", audit)?;
                    Ok(audit)
                }) {
                Ok(audit) => audit,
                Err(primary_error) => {
                    return match self.fail_closed_packed_f16_request_cleanup().await {
                        Ok(()) => Err(primary_error),
                        Err(cleanup_error) => Err(execution_error(
                            job.variant,
                            format!(
                                "{primary_error}; fail-closed packed-F16 DMD-entry cleanup also failed: {cleanup_error}"
                            ),
                        )),
                    };
                }
            };
            if packed_audit_after_qwen_handoff != Some(audit) {
                let primary_error = execution_error(
                    job.variant,
                    "packed-F16 cache audit changed after the Qwen instruction handoff and before DMD",
                );
                return match self.fail_closed_packed_f16_request_cleanup().await {
                    Ok(()) => Err(primary_error),
                    Err(cleanup_error) => Err(execution_error(
                        job.variant,
                        format!(
                            "{primary_error}; fail-closed packed-F16 DMD-entry cleanup also failed: {cleanup_error}"
                        ),
                    )),
                };
            }
            Some(audit)
        } else {
            None
        };
        if low_vram {
            let streamed_lifecycle = validate_low_vram_streamed_stage_lifecycle(
                job.variant,
                self.qwen.source.cached_stage_count(),
                self.qwen.source.has_pending_synchronization(),
                self.vae.cached_stage_count(),
            );
            let denoiser_cache = self.denoiser.source().cached_stage_count();
            let expected_entry_cache = if self.policies.preload_denoiser_before_request
                && !self.policies.uses_packed_f16_denoiser_source()
            {
                self.expected_denoiser_resident_stage_count()
            } else {
                0
            };
            let packed_cache_valid = packed_audit_before_dmd.is_none_or(|audit| {
                let Some(plan) = self.packed_f16_resource_plan else {
                    return false;
                };
                audit.state == PackedF16DenoiserCacheState::Ready
                    && audit.packed_cache_ready
                    && audit.cached_stage_count == plan.expected_stage_count
                    && audit.cached_object_count == plan.expected_object_count
                    && audit.cached_tensor_count == plan.expected_tensor_count
                    && audit.retained_packed_bytes == plan.retained_packed_f16_denoiser_bytes
            });
            let retention_valid = self.denoiser.source().retention_enabled()
                != self.policies.uses_packed_f16_denoiser_source();
            if !retention_valid
                || streamed_lifecycle.is_err()
                || denoiser_cache != expected_entry_cache
                || !packed_cache_valid
            {
                let primary_error = streamed_lifecycle.err().unwrap_or_else(|| {
                    execution_error(
                        job.variant,
                        format!(
                            "browser low-vram request entered DMD with invalid caches: retained={denoiser_cache}/{expected_entry_cache}, packed_valid={packed_cache_valid}"
                        ),
                    )
                });
                if self.policies.uses_packed_f16_denoiser_source() {
                    return match self.fail_closed_packed_f16_request_cleanup().await {
                        Ok(()) => Err(primary_error),
                        Err(cleanup_error) => Err(execution_error(
                            job.variant,
                            format!(
                                "{primary_error}; fail-closed packed-F16 DMD-entry cleanup also failed: {cleanup_error}"
                            ),
                        )),
                    };
                }
                self.denoiser.source_mut().clear();
                return Err(primary_error);
            }
        }
        let mut completed_dmd_steps = 0_usize;
        let dmd_result: Result<Tensor<BrowserBackend, 4>, RuntimeError> = async {
            let mut dmd_latents = latents;
            for (index, &sigma) in schedule.sigmas().iter().enumerate() {
                check_cancelled(cancellation)?;
                require_dtype(
                    job.variant,
                    "DMD latent",
                    dmd_latents.dtype(),
                    self.dtypes.denoiser,
                )?;
                let timestep = if index == 0 {
                    first_dmd_timestep.take().unwrap_or_else(|| {
                        Tensor::<BrowserBackend, 1>::from_data(
                            TensorData::new(vec![sigma], [1]),
                            &self.device,
                        )
                        .cast(self.dtypes.denoiser)
                    })
                } else {
                    Tensor::<BrowserBackend, 1>::from_data(
                        TensorData::new(vec![sigma], [1]),
                        &self.device,
                    )
                    .cast(self.dtypes.denoiser)
                };
                let prediction = self
                    .denoiser
                    .predict_async(BooguDenoiserInput {
                        latent: dmd_latents.clone(),
                        timestep,
                        instruction: instruction.clone(),
                        reference: reference.clone(),
                    })
                    .await
                    .map_err(|error| {
                        if cancellation.is_cancelled() {
                            RuntimeError::Cancelled
                        } else {
                            map_boogu(job.variant, error)
                        }
                    })?;
                self.denoiser
                    .source_mut()
                    .synchronize_pending()
                    .await
                    .map_err(|error| map_boogu(job.variant, error))?;
                require_finite_browser_tensor(
                    job.variant,
                    &format!("DMD step {} denoiser prediction", index + 1),
                    &prediction,
                )
                .await?;
                require_dtype(
                    job.variant,
                    "denoiser prediction",
                    prediction.dtype(),
                    self.dtypes.denoiser,
                )?;
                dmd_latents = dmd_prediction(dmd_latents, prediction, sigma);
                require_finite_browser_tensor(
                    job.variant,
                    &format!("DMD step {} prediction update", index + 1),
                    &dmd_latents,
                )
                .await?;
                if let Some(&next_sigma) = schedule.sigmas().get(index + 1) {
                    let noise = noises
                        .next()
                        .expect("the fixed four-step schedule has three renoise tensors");
                    dmd_latents = dmd_renoise(dmd_latents, noise, next_sigma);
                    require_finite_browser_tensor(
                        job.variant,
                        &format!("DMD step {} renoised latent", index + 1),
                        &dmd_latents,
                    )
                    .await?;
                }
                // The semantic-stage barrier covers the denoiser tail, but the DMD update above
                // is queued afterward. Flush that update explicitly before the next step and,
                // critically, before request-scoped weight and RoPE handles are cleared after
                // the final step.
                self.denoiser
                    .source_mut()
                    .synchronize()
                    .await
                    .map_err(|error| map_boogu(job.variant, error))?;
                self.denoiser
                    .source_mut()
                    .synchronize_pending()
                    .await
                    .map_err(|error| map_boogu(job.variant, error))?;
                completed_dmd_steps = index + 1;
                queue_progress(
                    shared,
                    id,
                    ProgressEvent::Step {
                        run_id,
                        stage: "dmd".into(),
                        step: index as u32 + 1,
                        total_steps: 4,
                        elapsed_micros: now_micros().saturating_sub(started),
                    },
                );
            }
            Ok(dmd_latents)
        }
        .await;
        let preloaded_dmd_traffic_result = if self.policies.preload_denoiser_before_request {
            self.artifact_control
                .traffic_snapshot()
                .checked_delta(dmd_artifact_traffic_before)
                .ok_or_else(|| {
                    execution_error(job.variant, "DMD artifact traffic counters moved backwards")
                })
                .and_then(|traffic| {
                    let report = BrowserArtifactTrafficReport::from(traffic);
                    if report == BrowserArtifactTrafficReport::default() {
                        Ok(report)
                    } else {
                        Err(execution_error(
                            job.variant,
                            format!(
                                "preloaded Turbo denoiser performed artifact I/O during DMD: {traffic:?}"
                            ),
                        ))
                    }
                })
        } else {
            Ok(BrowserArtifactTrafficReport::default())
        };
        let low_vram_cleanup_result = if low_vram {
            // Dense-stage materialization can submit dequantization work during
            // `load_*`, before the executor reaches its ordinary per-stage
            // barrier. Always issue a real source barrier on cleanup, including
            // error paths, before any retained Q8 or RoPE handle is cleared.
            let synchronization_result = self
                .denoiser
                .source_mut()
                .synchronize()
                .await
                .map_err(|error| map_boogu(job.variant, error));
            let pending_synchronization_result = self
                .denoiser
                .source_mut()
                .synchronize_pending()
                .await
                .map_err(|error| map_boogu(job.variant, error));
            let synchronization_result = synchronization_result.and(pending_synchronization_result);
            if self.policies.uses_packed_f16_denoiser_source() {
                // Keep every validation/downcast failure inside the cleanup result. The caller
                // combines this with the primary DMD result and always executes the async
                // fail-closed allocator cleanup before returning.
                (|| -> Result<(), RuntimeError> {
                    let synchronization_pending =
                        self.denoiser.source().has_pending_synchronization();
                    let audit_before = packed_audit_before_dmd.ok_or_else(|| {
                        execution_error(job.variant, "packed-F16 DMD entry audit is absent")
                    })?;
                    let audit_after = self.packed_f16_denoiser_source()?.audit();
                    let traffic = preloaded_dmd_traffic_result
                        .as_ref()
                        .copied()
                        .unwrap_or_default();
                    let report = packed_f16_lifecycle_report(
                        job.variant,
                        audit_before,
                        audit_after,
                        traffic,
                        synchronization_pending,
                    )
                    .and_then(|report| {
                        if dmd_result.is_ok() && completed_dmd_steps == 4 {
                            validate_packed_f16_denoiser_lifecycle(
                                job.variant,
                                self.packed_f16_resource_plan.ok_or_else(|| {
                                    execution_error(
                                        job.variant,
                                        "browser packed-F16 resource plan is absent",
                                    )
                                })?,
                                completed_dmd_steps as u64,
                                report,
                            )
                        } else {
                            Ok(report)
                        }
                    });
                    let mut preserve_packed_cache = dmd_result.is_ok()
                        && completed_dmd_steps == 4
                        && synchronization_result.is_ok()
                        && preloaded_dmd_traffic_result.is_ok()
                        && report.as_ref().is_ok_and(|report| report.matches_plan)
                        && self.denoiser.source().cached_stage_count() == 0
                        && !synchronization_pending;
                    self.denoiser.source_mut().clear();
                    self.denoiser.clear_rope_cache();
                    if preserve_packed_cache {
                        if let Ok(lifecycle) = report.as_ref().copied() {
                            self.last_packed_f16_denoiser_lifecycle = Some(lifecycle);
                            // This DMD-scoped evidence is emitted only after all four steps, queue
                            // barriers, zero-I/O checks, lifecycle counters, and cache readiness have
                            // passed. The separate DMD-to-VAE event must subsequently attest Empty/0
                            // retained bytes; partial DMD requests cannot emit either success pair.
                            dispatch_browser_event(
                                BROWSER_RUNTIME_EVENT_NAME,
                                &browser_packed_f16_denoiser_lifecycle_event(lifecycle),
                            );
                        } else {
                            preserve_packed_cache = false;
                        }
                    }
                    let packed_clear_result = if preserve_packed_cache {
                        Ok(())
                    } else {
                        self.packed_f16_denoiser_source_mut().map(|source| {
                            if source.audit().state == PackedF16DenoiserCacheState::Failed {
                                source.clear();
                            } else {
                                source.fail_and_clear();
                            }
                        })
                    };
                    synchronization_result
                    .and(preloaded_dmd_traffic_result.map(|_| ()))
                    .and(report.map(|_| ()))
                    .and(packed_clear_result)
                    .and_then(|()| {
                        if dmd_result.is_err() || preserve_packed_cache {
                            Ok(())
                        } else {
                            Err(execution_error(
                                job.variant,
                                "browser packed-F16 denoiser cache was not safe to retain after DMD",
                            ))
                        }
                    })
                })()
            } else {
                let retained_stages_before_clear = self.denoiser.source().cached_stage_count();
                let synchronization_pending = self.denoiser.source().has_pending_synchronization();
                let dense_f32_materialized_stage_clones = self
                    .denoiser
                    .source()
                    .dense_f32_materialized_stage_clones()
                    .checked_sub(dense_stage_clones_before_dmd)
                    .ok_or_else(|| {
                        execution_error(
                            job.variant,
                            "dense-F32 stage-clone counter moved backwards",
                        )
                    });
                let dtype_audit = if dmd_result.is_ok() {
                    validate_low_vram_denoiser_dtype_audit(
                        job.variant,
                        self.low_vram_resource_plan.ok_or_else(|| {
                            execution_error(job.variant, "browser low-vram resource plan is absent")
                        })?,
                        self.denoiser.source().retained_dtype_audit(),
                    )
                    .map(Some)
                } else {
                    Ok(None)
                };
                let expected_dense_stage_clones =
                    if self.policies.denoiser_retaining_wrapper_adapter
                        == BooguQuantizedLinearExecutionPolicy::DenseF32PerSemanticStage
                    {
                        self.expected_denoiser_resident_stage_count() * completed_dmd_steps
                    } else {
                        0
                    };
                let preserve_preloaded_cache = self.policies.preload_denoiser_before_request
                    && dmd_result.is_ok()
                    && synchronization_result.is_ok()
                    && dtype_audit.is_ok()
                    && dense_f32_materialized_stage_clones
                        .as_ref()
                        .is_ok_and(|observed| *observed == expected_dense_stage_clones)
                    && preloaded_dmd_traffic_result.is_ok()
                    && completed_dmd_steps == 4
                    && retained_stages_before_clear
                        == self.expected_denoiser_resident_stage_count()
                    && !synchronization_pending;
                if !preserve_preloaded_cache {
                    self.denoiser.source_mut().clear();
                }
                // RoPE geometry is request-shape-local and deliberately not part of the immutable
                // preloaded parameter cache.
                self.denoiser.clear_rope_cache();
                let retained_stages_after_request = self.denoiser.source().cached_stage_count();
                synchronization_result
                .and(preloaded_dmd_traffic_result)
                .and(dtype_audit)
                .and_then(|audit| {
                    dense_f32_materialized_stage_clones
                        .map(|dense_f32_materialized_stage_clones| {
                            (audit, dense_f32_materialized_stage_clones)
                        })
                })
                .and_then(|(audit, dense_f32_materialized_stage_clones)| {
                self.last_low_vram_denoiser_dtype_audit = audit;
                self.last_dense_f32_materialized_stage_clones =
                    dense_f32_materialized_stage_clones;
                if dmd_result.is_err() {
                    if retained_stages_after_request == 0 {
                        Ok(())
                    } else {
                        Err(execution_error(
                            job.variant,
                            "browser low-vram denoiser cache cleanup failed after DMD error",
                        ))
                    }
                } else {
                    if dense_f32_materialized_stage_clones != expected_dense_stage_clones {
                        return Err(execution_error(
                            job.variant,
                            format!(
                                "browser low-vram denoiser materialized {dense_f32_materialized_stage_clones}/{expected_dense_stage_clones} dense-F32 semantic-stage clones"
                            ),
                        ));
                    }
                    validate_low_vram_denoiser_lifecycle(
                        job.variant,
                        completed_dmd_steps,
                        self.expected_denoiser_resident_stage_count(),
                        retained_stages_before_clear,
                        synchronization_pending,
                        if preserve_preloaded_cache {
                            self.expected_denoiser_resident_stage_count()
                        } else {
                            0
                        },
                        retained_stages_after_request,
                    )
                }
            })
            }
        } else {
            Ok(())
        };
        let latents = match (dmd_result, low_vram_cleanup_result) {
            (Ok(latents), Ok(())) => latents,
            (dmd_result, cleanup_result) => {
                let primary_error = match (dmd_result, cleanup_result) {
                    (Err(dmd_error), Ok(())) => dmd_error,
                    (Ok(_), Err(cleanup_error)) => cleanup_error,
                    (Err(dmd_error), Err(cleanup_error)) => execution_error(
                        job.variant,
                        format!(
                            "{dmd_error}; packed/low-VRAM DMD cleanup also failed: {cleanup_error}"
                        ),
                    ),
                    (Ok(_), Ok(())) => unreachable!("successful DMD handled above"),
                };
                if self.policies.uses_packed_f16_denoiser_source() {
                    return match self.fail_closed_packed_f16_request_cleanup().await {
                        Ok(()) => Err(primary_error),
                        Err(fail_closed_error) => Err(execution_error(
                            job.variant,
                            format!(
                                "{primary_error}; fail-closed packed-F16 DMD cleanup also failed: {fail_closed_error}"
                            ),
                        )),
                    };
                }
                return Err(primary_error);
            }
        };
        finish_stage(shared, id, run_id, &mut timings, "dmd", started);
        let latents = if self.policies.uses_packed_f16_denoiser_source() {
            // The DMD loop no longer needs any conditioning, renoise, timestep, or iterator
            // handle. Drop those roots before crossing the allocator boundary with the sole final
            // latent value.
            drop(instruction);
            drop(reference);
            drop(noises);
            drop(first_dmd_timestep);
            let handoff = self
                .packed_f16_dmd_vae_handoff(latents, latent_shape, run_id)
                .await;
            match handoff {
                Ok(latents) => latents,
                Err(handoff_error) => {
                    // Never leave a stale Ready cache after a failed/cancelled post-DMD boundary.
                    // Preserve the primary failure while appending any best-effort barrier/cleanup
                    // failure that can affect the next request's verified rehydration.
                    match self.fail_closed_packed_f16_request_cleanup().await {
                        Ok(()) => return Err(handoff_error),
                        Err(cleanup_error) => {
                            return Err(execution_error(
                                job.variant,
                                format!(
                                    "{handoff_error}; fail-closed packed-F16 DMD-to-VAE cleanup also failed: {cleanup_error}"
                                ),
                            ));
                        }
                    }
                }
            }
        } else if self.policies.phase_boundary_memory_cleanup {
            // Preserve all resident Q4S/F16 model buffers. Only dead DMD activations and cached
            // allocator pages are eligible for cleanup before the VAE phase.
            drop(instruction);
            drop(reference);
            drop(noises);
            drop(first_dmd_timestep);
            release_browser_phase_allocator(job.variant, &self.device, "DMD-to-VAE handoff")
                .await?;
            latents
        } else {
            latents
        };

        check_cancelled(cancellation)?;
        let started = start_stage(shared, id, run_id, "vae-decode", Some(1));
        let decoder = self
            .vae
            .load_decoder()
            .await
            .map_err(|error| map_boogu(job.variant, error))?;
        // Decoder-only staging leaves the lazy encoder at the backend default dtype. Inspect the
        // stage that was actually authenticated and loaded rather than the encoder sentinel.
        let loaded_dtype: DType = decoder.decoder_float_dtype().into();
        require_dtype(
            job.variant,
            "loaded VAE decoder",
            loaded_dtype,
            self.policies.vae_parameter_dtype(),
        )?;
        let scaled_latents = latents.cast(self.dtypes.vae);
        require_finite_browser_tensor(job.variant, "VAE scaled decode input", &scaled_latents)
            .await?;
        let (decoded, decode_already_synchronized) = match buffer_plan.vae_decode_policy {
            crate::boogu::BrowserVaeDecodePolicy::FullStrictF32 => {
                (decoder.decode_scaled(scaled_latents), false)
            }
            crate::boogu::BrowserVaeDecodePolicy::StripedTailStrictF32 { split_width } => {
                let decode_input = decoder.unscale_latents(scaled_latents);
                if self.policies.phase_boundary_memory_cleanup {
                    // Submit exact decoder blocks separately so each preceding activation becomes
                    // reclaimable before the next block allocates. The resident decoder and every
                    // other model parameter remain alive; this is activation lifetime control,
                    // not the pathological request-time model-unload strategy.
                    let mut state =
                        decoder.begin_decode_striped_tail_strict_f32(decode_input, split_width);
                    release_browser_phase_allocator(
                        job.variant,
                        &self.device,
                        "VAE initial convolution and middle block",
                    )
                    .await?;
                    let stage_count = decoder.decoder.striped_tail_stage_count();
                    let mut stage_index = 0_usize;
                    while !state.is_complete() {
                        decoder.advance_decode_striped_tail_strict_f32(&mut state);
                        stage_index += 1;
                        let phase = format!("VAE decode stage {stage_index}/{stage_count}");
                        release_browser_phase_allocator(job.variant, &self.device, &phase).await?;
                    }
                    debug_assert_eq!(stage_index, stage_count);
                    (state.into_output(), true)
                } else {
                    (
                        decoder.decode_striped_tail_strict_f32(decode_input, split_width),
                        false,
                    )
                }
            }
        };
        if !decode_already_synchronized {
            self.vae
                .synchronize()
                .await
                .map_err(|error| map_boogu(job.variant, error))?;
        }
        drop(decoder);
        if self.policies.phase_boundary_memory_cleanup && !decode_already_synchronized {
            // `decoded` is still live; this boundary therefore preserves the exact output while
            // releasing only completed decoder intermediates before host readback.
            release_browser_phase_allocator(job.variant, &self.device, "VAE decode-to-output")
                .await?;
        }
        check_cancelled(cancellation)?;
        finish_stage(shared, id, run_id, &mut timings, "vae-decode", started);

        let started = start_stage(shared, id, run_id, "output", Some(1));
        // The unconditional full output readback below feeds `decoder_output_data_to_host`, which
        // validates the shape and rejects every non-finite value while converting to RGB8. Avoid a
        // redundant qualification-only device reduction and scalar-readback boundary here.
        let image = decoder_output_to_host_async(decoded)
            .await
            .map_err(|error| map_boogu(job.variant, error))?;
        if self.policies.phase_boundary_memory_cleanup {
            // Preserve the complete warm model while returning dead VAE/output pages to WebGPU.
            release_browser_phase_allocator(job.variant, &self.device, "completed output").await?;
        }
        finish_stage(shared, id, run_id, &mut timings, "output", started);

        let stage_sum = timings
            .iter()
            .map(|timing| timing.elapsed_micros)
            .sum::<u64>();
        let total_micros = now_micros().saturating_sub(total_started).max(stage_sum);
        let output = ImageOutput {
            images: vec![GeneratedImage { index: 0, image }],
            seed: job.resolved.seed,
            timings: StageTimings {
                stages: timings,
                total_micros,
            },
            provenance: ModelProvenance {
                model: boogu_model_descriptor(job.variant).id,
                model_revision: self.identity.model_revision.clone(),
                artifact_content_digest: Some(self.artifact_content_digest),
                numeric_format: self.numeric_format.clone(),
                backend: self.policies.provenance_backend(),
                artifacts_verified: true,
            },
        };
        output
            .validate()
            .map_err(|error| execution_error(job.variant, error))?;
        let artifact_traffic = self
            .artifact_control
            .traffic_snapshot()
            .checked_delta(artifact_traffic_before)
            .ok_or_else(|| {
                execution_error(job.variant, "artifact traffic counters moved backwards")
            })?;
        let artifact_traffic = BrowserArtifactTrafficReport::from(artifact_traffic);
        if self.policies.eager_preload {
            self.validate_resident_caches()?;
            self.report_resident_cache_audit(run_id, "after-request");
            if artifact_traffic != BrowserArtifactTrafficReport::default() {
                return Err(execution_error(
                    job.variant,
                    format!(
                        "resident browser request performed artifact I/O after its eager preload: {artifact_traffic:?}"
                    ),
                ));
            }
        }
        self.last_artifact_traffic = artifact_traffic;
        dispatch_browser_event(
            BROWSER_RUNTIME_EVENT_NAME,
            &BrowserRuntimeEvent::ArtifactTraffic {
                traffic: self.last_artifact_traffic,
            },
        );
        queue_progress(
            shared,
            id,
            ProgressEvent::RunCompleted {
                run_id,
                elapsed_micros: total_micros,
            },
        );
        Ok(output)
    }
}

async fn compare_prepared_integer_plan(
    fixture: &BrowserParityFixture,
    ledger: &BrowserParityOracleLedger,
    prepared: &burn_boogu::PreparedInstruction<BrowserBackend>,
) -> Result<Vec<BrowserParityExactMetric>, RuntimeError> {
    let encoding = &prepared.encoding;
    let input_ids = encoding
        .input_ids
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    let attention_mask = encoding
        .attention_mask
        .iter()
        .flatten()
        .map(|value| i64::from(*value))
        .collect::<Vec<_>>();
    let token_types = encoding
        .mm_token_type_ids
        .iter()
        .flatten()
        .map(|&value| i64::from(value))
        .collect::<Vec<_>>();
    let grids = encoding
        .image_grids
        .iter()
        .flatten()
        .flat_map(|grid| [grid.t as i64, grid.h as i64, grid.w as i64])
        .collect::<Vec<_>>();
    let sequence = encoding.sequence_length();
    let plans = [
        (
            "processor.input_ids",
            vec![encoding.batch_size(), sequence],
            input_ids.as_slice(),
        ),
        (
            "processor.attention_mask",
            vec![encoding.batch_size(), sequence],
            attention_mask.as_slice(),
        ),
        (
            "qwen.attention_mask",
            vec![encoding.batch_size(), sequence],
            attention_mask.as_slice(),
        ),
        (
            "processor.mm_token_type_ids",
            vec![encoding.batch_size(), sequence],
            token_types.as_slice(),
        ),
        (
            "processor.image_grid_thw",
            vec![grids.len() / 3, 3],
            grids.as_slice(),
        ),
    ];
    let mut metrics = Vec::with_capacity(plans.len());
    for (name, shape, actual) in plans {
        let (expected_shape, expected) = fixture.i64(name).await?;
        ledger.record(name);
        metrics.push(BrowserParityExactMetric {
            name: name.into(),
            shape: shape.clone(),
            elements: actual.len() as u64,
            exact: shape == expected_shape && actual == expected,
        });
    }
    Ok(metrics)
}

fn expected_qwen_aligned_stage_oracles() -> BTreeSet<String> {
    let mut names = BTreeSet::from([
        "qwen.text.token_embeddings".to_owned(),
        "qwen.vision.prelude".to_owned(),
        "qwen.vision.final_merger".to_owned(),
        "qwen.text.final_norm".to_owned(),
    ]);
    names.extend((0..27).map(|index| format!("qwen.vision.block.{index}")));
    names.extend((0..3).map(|index| format!("qwen.vision.deepstack_merger.{index}")));
    names.extend((0..36).map(|index| {
        if index < 3 {
            format!("qwen.text.layer.{index}.post_deepstack")
        } else {
            format!("qwen.text.layer.{index}")
        }
    }));
    debug_assert_eq!(names.len(), BROWSER_1K5_QWEN_ALIGNED_STAGE_COUNT);
    names
}

fn expected_qwen_authenticated_only_oracles() -> BTreeSet<String> {
    BTreeSet::from([
        "qwen.text.layer.0".to_owned(),
        "qwen.text.layer.0.input_layernorm".to_owned(),
        "qwen.text.layer.0.mlp".to_owned(),
        "qwen.text.layer.0.mlp.down_proj".to_owned(),
        "qwen.text.layer.0.mlp.gate_proj".to_owned(),
        "qwen.text.layer.0.mlp.up_proj".to_owned(),
        "qwen.text.layer.0.post_attention_layernorm".to_owned(),
        "qwen.text.layer.0.self_attn.0".to_owned(),
        "qwen.text.layer.0.self_attn.k_norm".to_owned(),
        "qwen.text.layer.0.self_attn.k_proj".to_owned(),
        "qwen.text.layer.0.self_attn.o_proj".to_owned(),
        "qwen.text.layer.0.self_attn.q_norm".to_owned(),
        "qwen.text.layer.0.self_attn.q_proj".to_owned(),
        "qwen.text.layer.0.self_attn.v_proj".to_owned(),
        "qwen.text.layer.1".to_owned(),
        "qwen.text.layer.2".to_owned(),
        "qwen.vision.patch_embed".to_owned(),
    ])
}

fn expected_denoiser_boundary_oracles() -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for step in 0..4 {
        for boundary in [
            "time_caption_embed.0".to_owned(),
            "time_caption_embed.1".to_owned(),
            "x_embedder".to_owned(),
            "ref_image_patch_embedder".to_owned(),
            "norm_out".to_owned(),
        ] {
            names.insert(format!("denoiser.step.{step}.{boundary}"));
        }
        names.extend((0..2).map(|index| format!("denoiser.step.{step}.context_refiner.{index}")));
        names.extend((0..2).map(|index| format!("denoiser.step.{step}.noise_refiner.{index}")));
        names.extend((0..2).map(|index| format!("denoiser.step.{step}.ref_image_refiner.{index}")));
        for index in 0..8 {
            names.insert(format!(
                "denoiser.step.{step}.double_stream_layers.{index}.0"
            ));
            names.insert(format!(
                "denoiser.step.{step}.double_stream_layers.{index}.1"
            ));
        }
        names.extend(
            (0..32).map(|index| format!("denoiser.step.{step}.single_stream_layers.{index}")),
        );
    }
    debug_assert_eq!(names.len(), BROWSER_1K5_DENOISER_BOUNDARY_COUNT);
    names
}

async fn tensor4_from_fixture(
    fixture: &BrowserParityFixture,
    ledger: &BrowserParityOracleLedger,
    name: &str,
    dtype: DType,
    device: &burn_wgpu::WgpuDevice,
) -> Result<Tensor<BrowserBackend, 4>, RuntimeError> {
    let (shape, values) = fixture.f32(name).await?;
    ledger.record(name);
    let shape: [usize; 4] = shape.try_into().map_err(|shape: Vec<usize>| {
        execution_error(
            BooguVariant::Image01EditTurbo1k5,
            format!("fixture tensor {name} must be rank four, got {shape:?}"),
        )
    })?;
    Ok(Tensor::<BrowserBackend, 4>::from_data(TensorData::new(values, shape), device).cast(dtype))
}

async fn fixture_scalar(
    fixture: &BrowserParityFixture,
    ledger: &BrowserParityOracleLedger,
    name: &str,
) -> Result<f32, RuntimeError> {
    let (shape, values) = fixture.f32(name).await?;
    ledger.record(name);
    match values.as_slice() {
        [value] if value.is_finite() && (shape.is_empty() || shape == [1]) => Ok(*value),
        _ => Err(execution_error(
            BooguVariant::Image01EditTurbo1k5,
            format!("fixture tensor {name} must contain one finite scalar"),
        )),
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "the gate evaluates each independently reported parity surface"
)]
fn evaluate_browser_1k5_gates(
    residency: BrowserBooguResidencyPolicy,
    processing: &BrowserParityProcessingReport,
    qwen: &BrowserParityQwenReport,
    vae: &BrowserParityVaeReferenceReport,
    denoiser: &[BrowserParityTensorMetric],
    dmd: &BrowserParityDmdReport,
    decode_input: &BrowserParityTensorMetric,
    decoded: &BrowserParityTensorMetric,
    rgb: &RgbMetrics,
) -> BrowserParityGateReport {
    let qwen_aligned_stages = BrowserParityTensorGate {
        maximum_relative_rmse: 0.2,
        minimum_cosine_similarity: 0.99,
    };
    let qwen_final = BrowserParityTensorGate {
        maximum_relative_rmse: 0.10,
        minimum_cosine_similarity: 0.995,
    };
    let browser_webgpu_vae_f32_oracle_envelope = BROWSER_WEBGPU_VAE_F32_ORACLE_ENVELOPE;
    let denoiser_boundaries = BrowserParityTensorGate {
        maximum_relative_rmse: 0.265_799_3,
        minimum_cosine_similarity: 0.964_623_4,
    };
    let dmd_boundaries = BrowserParityTensorGate {
        maximum_relative_rmse: 0.13,
        minimum_cosine_similarity: 0.992,
    };
    let dmd_final = BrowserParityTensorGate {
        maximum_relative_rmse: 0.085,
        minimum_cosine_similarity: 0.996,
    };
    let propagated_decode = BrowserParityTensorGate {
        maximum_relative_rmse: 0.09,
        minimum_cosine_similarity: 0.996,
    };
    let final_rgb = if residency == BrowserBooguResidencyPolicy::LowVramRuntimeQ8Denoiser {
        BrowserParityRgbGate {
            minimum_psnr_db: 24.0,
            minimum_mean_block_ssim_8x8: 0.90,
        }
    } else {
        BrowserParityRgbGate {
            minimum_psnr_db: 33.5,
            minimum_mean_block_ssim_8x8: 0.99,
        }
    };
    let mut failures = Vec::new();
    if !processing.prompt_exact || !processing.dimensions_exact || !processing.seed_exact {
        failures.push("resolved request differs from the pinned prompt/dimensions/seed".into());
    }
    for metric in &processing.integer_tensors {
        if !metric.exact {
            failures.push(format!("{} is not exact", metric.name));
        }
    }
    if processing.pixel_values.comparison.max_abs > f32::EPSILON {
        failures.push(format!(
            "processing.pixel_values max={} exceeds F32 epsilon",
            processing.pixel_values.comparison.max_abs
        ));
    }
    check_tensor_gate(&processing.mrope_cos, qwen_aligned_stages, &mut failures);
    check_tensor_gate(&processing.mrope_sin, qwen_aligned_stages, &mut failures);
    if qwen.compared_aligned_stages != qwen.expected_aligned_stages {
        failures.push(format!(
            "Qwen compared {}/{} aligned stages",
            qwen.compared_aligned_stages, qwen.expected_aligned_stages
        ));
    }
    if !qwen.aligned_stage_names_exact {
        failures.push("Qwen aligned stage names differ from the exact 70-name contract".into());
    }
    if !qwen.authenticated_only_diagnostics_exact {
        failures.push(
            "Qwen authenticated-only diagnostics differ from the exact 17-name contract".into(),
        );
    }
    for metric in &qwen.aligned_stages {
        check_tensor_gate(metric, qwen_aligned_stages, &mut failures);
    }
    check_tensor_gate(&qwen.final_hidden_state, qwen_final, &mut failures);
    if vae.f32_oracle.len() != 6 {
        failures.push(format!(
            "VAE reference compared {}/6 F32 boundaries",
            vae.f32_oracle.len()
        ));
    }
    for metric in [&vae.input, &vae.injected_epsilon] {
        if metric.comparison.max_abs != 0.0 {
            failures.push(format!("{} exact injection differs", metric.name));
        }
    }
    for metric in &vae.f32_oracle {
        let Some(component_maximum) =
            browser_webgpu_vae_f32_oracle_envelope.component_maximum(&metric.name)
        else {
            failures.push(format!("unknown VAE F32 component {}", metric.name));
            continue;
        };
        if metric.comparison.max_abs > component_maximum {
            failures.push(format!(
                "{} max={} exceeds component maximum {}",
                metric.name, metric.comparison.max_abs, component_maximum
            ));
        }
        if metric.name.ends_with("moments")
            && (metric.comparison.rmse
                > browser_webgpu_vae_f32_oracle_envelope.moments.maximum_rmse
                || metric.comparison.cosine_similarity
                    < browser_webgpu_vae_f32_oracle_envelope
                        .moments
                        .minimum_cosine_similarity)
        {
            failures.push(format!(
                "{} misses moment gate: rmse={} cosine={}",
                metric.name, metric.comparison.rmse, metric.comparison.cosine_similarity
            ));
        }
        if metric.name.ends_with("scaled_latent")
            && (metric.comparison.max_abs
                > browser_webgpu_vae_f32_oracle_envelope
                    .scaled_latent
                    .maximum_abs
                || metric.comparison.rmse
                    > browser_webgpu_vae_f32_oracle_envelope
                        .scaled_latent
                        .maximum_rmse
                || metric.comparison.cosine_similarity
                    < browser_webgpu_vae_f32_oracle_envelope
                        .scaled_latent
                        .minimum_cosine_similarity)
        {
            failures.push(format!(
                "{} misses its browser WebGPU scaled-latent gate: max={} rmse={} cosine={}",
                metric.name,
                metric.comparison.max_abs,
                metric.comparison.rmse,
                metric.comparison.cosine_similarity
            ));
        }
    }
    if denoiser.len() != BROWSER_1K5_DENOISER_BOUNDARY_COUNT {
        failures.push(format!(
            "denoiser compared {}/{} boundaries",
            denoiser.len(),
            BROWSER_1K5_DENOISER_BOUNDARY_COUNT
        ));
    }
    for metric in denoiser {
        check_tensor_gate(metric, denoiser_boundaries, &mut failures);
    }
    if dmd.initial_latent.comparison.max_abs != 0.0 {
        failures.push("trajectory.initial_latent exact injection differs".into());
    }
    if dmd.steps.len() != 4 {
        failures.push(format!("DMD executed {}/4 steps", dmd.steps.len()));
    }
    for step in &dmd.steps {
        if !step.sigma_exact {
            failures.push(format!(
                "dmd.step.{}.sigma differs: schedule={} fixture={}",
                step.index, step.schedule_sigma, step.fixture_sigma
            ));
        }
        for metric in [&step.input, &step.velocity, &step.prediction] {
            check_tensor_gate(metric, dmd_boundaries, &mut failures);
        }
        if let Some(metric) = &step.injected_noise
            && metric.comparison.max_abs != 0.0
        {
            failures.push(format!("{} exact injection differs", metric.name));
        }
        if let Some(metric) = &step.renoised {
            check_tensor_gate(metric, dmd_boundaries, &mut failures);
        }
    }
    check_tensor_gate(&dmd.final_latent, dmd_final, &mut failures);
    check_tensor_gate(decode_input, dmd_final, &mut failures);
    check_tensor_gate(decoded, propagated_decode, &mut failures);
    if rgb.psnr_db < final_rgb.minimum_psnr_db
        || rgb.mean_block_ssim_8x8 < final_rgb.minimum_mean_block_ssim_8x8
    {
        failures.push(format!(
            "final RGB misses gate: PSNR={} SSIM={}",
            rgb.psnr_db, rgb.mean_block_ssim_8x8
        ));
    }
    BrowserParityGateReport {
        passed: failures.is_empty(),
        qwen_aligned_stages,
        qwen_final,
        browser_webgpu_vae_f32_oracle_envelope,
        denoiser_boundaries,
        dmd_boundaries,
        dmd_final,
        propagated_decode,
        final_rgb,
        failures,
    }
}

fn check_browser_1k5_vae_reference_run(
    repeat_index: usize,
    metrics: &[BrowserParityTensorMetric],
    failures: &mut Vec<String>,
) {
    let expected = [
        "vae.reference_f32_moments",
        "vae.reference_f32_mean",
        "vae.reference_f32_logvar",
        "vae.reference_f32_std",
        "vae.reference_f32_raw_latent",
        "vae.reference_f32_scaled_latent",
    ];
    let actual = metrics
        .iter()
        .map(|metric| metric.name.as_str())
        .collect::<Vec<_>>();
    if actual != expected {
        failures.push(format!(
            "VAE diagnostic repeat {repeat_index} has boundary names {actual:?}, expected {expected:?}"
        ));
        return;
    }
    for metric in metrics {
        let Some(component_maximum) =
            BROWSER_WEBGPU_VAE_F32_ORACLE_ENVELOPE.component_maximum(&metric.name)
        else {
            failures.push(format!(
                "VAE diagnostic repeat {repeat_index} has unknown component {}",
                metric.name
            ));
            continue;
        };
        if metric.comparison.max_abs > component_maximum {
            failures.push(format!(
                "VAE diagnostic repeat {repeat_index} {} max={} exceeds component maximum {}",
                metric.name, metric.comparison.max_abs, component_maximum
            ));
        }
        if metric.name.ends_with("moments")
            && (metric.comparison.rmse
                > BROWSER_WEBGPU_VAE_F32_ORACLE_ENVELOPE.moments.maximum_rmse
                || metric.comparison.cosine_similarity
                    < BROWSER_WEBGPU_VAE_F32_ORACLE_ENVELOPE
                        .moments
                        .minimum_cosine_similarity)
        {
            failures.push(format!(
                "VAE diagnostic repeat {repeat_index} {} misses moment gate: rmse={} cosine={}",
                metric.name, metric.comparison.rmse, metric.comparison.cosine_similarity
            ));
        }
        if metric.name.ends_with("scaled_latent")
            && (metric.comparison.max_abs
                > BROWSER_WEBGPU_VAE_F32_ORACLE_ENVELOPE
                    .scaled_latent
                    .maximum_abs
                || metric.comparison.rmse
                    > BROWSER_WEBGPU_VAE_F32_ORACLE_ENVELOPE
                        .scaled_latent
                        .maximum_rmse
                || metric.comparison.cosine_similarity
                    < BROWSER_WEBGPU_VAE_F32_ORACLE_ENVELOPE
                        .scaled_latent
                        .minimum_cosine_similarity)
        {
            failures.push(format!(
                "VAE diagnostic repeat {repeat_index} {} misses its browser WebGPU scaled-latent gate: max={} rmse={} cosine={}",
                metric.name,
                metric.comparison.max_abs,
                metric.comparison.rmse,
                metric.comparison.cosine_similarity
            ));
        }
    }
}

fn vae_encoder_artifact_verification_report(
    ledger: &BrowserVerifiedArtifactLedger,
    expected: &BTreeMap<String, u64>,
) -> BrowserVaeEncoderArtifactVerificationReport {
    let counts = ledger.counts();
    let actual_names = counts.keys().cloned().collect::<BTreeSet<_>>();
    let expected_names = expected.keys().cloned().collect::<BTreeSet<_>>();
    let missing_weight_objects = expected_names
        .difference(&actual_names)
        .cloned()
        .collect::<Vec<_>>();
    let unexpected_verified_objects = actual_names
        .difference(&expected_names)
        .cloned()
        .collect::<Vec<_>>();
    let objects = expected
        .iter()
        .map(|(path, &size)| BrowserVaeEncoderArtifactObjectReport {
            path: path.clone(),
            size,
            verification_count: counts.get(path).copied().unwrap_or(0),
        })
        .collect::<Vec<_>>();
    let verification_count_per_object_exact = objects
        .iter()
        .all(|object| object.verification_count == BROWSER_1K5_VAE_REFERENCE_REPEAT_COUNT);
    let expected_weight_bytes_per_repeat = expected.values().copied().sum::<u64>();
    let verified_weight_bytes_all_repeats = objects.iter().fold(0_u64, |total, object| {
        total.saturating_add(object.size.saturating_mul(object.verification_count as u64))
    });
    let passed = missing_weight_objects.is_empty()
        && unexpected_verified_objects.is_empty()
        && actual_names.len() == expected_names.len()
        && verification_count_per_object_exact
        && verified_weight_bytes_all_repeats
            == expected_weight_bytes_per_repeat
                .saturating_mul(BROWSER_1K5_VAE_REFERENCE_REPEAT_COUNT as u64);
    BrowserVaeEncoderArtifactVerificationReport {
        scope: "only the manifest-declared flux-vae-encoder Burnpack stage, reconstructed and SHA-256-verified independently for each of three repeats; no Qwen, denoiser, or VAE decoder weight object is accepted"
            .into(),
        expected_repeats: BROWSER_1K5_VAE_REFERENCE_REPEAT_COUNT,
        expected_unique_weight_objects: expected_names.len(),
        verified_unique_weight_objects: actual_names.intersection(&expected_names).count(),
        expected_weight_bytes_per_repeat,
        verified_weight_bytes_all_repeats,
        missing_weight_objects,
        unexpected_verified_objects,
        objects,
        verification_count_per_object_exact,
        passed,
    }
}

fn check_tensor_gate(
    metric: &BrowserParityTensorMetric,
    gate: BrowserParityTensorGate,
    failures: &mut Vec<String>,
) {
    if metric.comparison.relative_rmse > gate.maximum_relative_rmse
        || metric.comparison.cosine_similarity < gate.minimum_cosine_similarity
    {
        failures.push(format!(
            "{} rel-RMSE={} cosine={} misses ({}, {})",
            metric.name,
            metric.comparison.relative_rmse,
            metric.comparison.cosine_similarity,
            gate.maximum_relative_rmse,
            gate.minimum_cosine_similarity
        ));
    }
}

fn browser_source_requires_canonical_digest(
    variant: BooguVariant,
    profile: BooguStorageProfile,
    base_url: &burn_image::RemoteBaseUrl,
) -> bool {
    canonical_published_bundle(variant, profile).is_some_and(|published| {
        base_url.as_str() == format!("{}/{}", crate::boogu::BOOGU_CDN_ROOT, published.bundle_id)
    })
}

fn validate_browser_manifest_bundle_identity(
    variant: BooguVariant,
    profile: BooguStorageProfile,
    actual: &str,
) -> Result<(), RuntimeError> {
    if artifact_bundle_id_matches_selection(variant, profile, actual) {
        return Ok(());
    }
    let preferred = crate::boogu::boogu_bundle_id(variant, profile);
    let source = crate::boogu::boogu_source_bundle_id(variant, profile);
    Err(execution_error(
        variant,
        format!(
            "sealed browser manifest bundle {actual} does not match the selected release/profile; expected {preferred} or explicit conversion source {source}"
        ),
    ))
}

struct BrowserBooguRuntime {
    variant: BooguVariant,
    shared: Arc<Mutex<BrowserRuntimeShared>>,
}

struct BrowserRuntimeShared {
    engine: Option<BrowserBooguEngine>,
    active: Option<(ImageJobId, CancellationToken)>,
    events: VecDeque<ImageRunnerEvent>,
    diagnostic_console_progress: bool,
    diagnostic_peak_wasm_linear_memory_bytes: u64,
}

impl BrowserBooguRuntime {
    fn new(engine: BrowserBooguEngine) -> Self {
        let variant = engine.identity.variant;
        Self {
            variant,
            shared: Arc::new(Mutex::new(BrowserRuntimeShared {
                engine: Some(engine),
                active: None,
                events: VecDeque::new(),
                diagnostic_console_progress: false,
                diagnostic_peak_wasm_linear_memory_bytes: 0,
            })),
        }
    }
}

fn browser_release_switching_enabled() -> bool {
    web_sys::window()
        .and_then(|window| window.location().search().ok())
        .and_then(|search| web_sys::UrlSearchParams::new_with_str(&search).ok())
        .is_some_and(|params| !params.has("artifacts") && !params.has("headless"))
}

const BROWSER_EDIT_CONTEXT_QUERY: &str = "edit-context-sha256";
const BROWSER_EDIT_CONTEXT_BYTES_QUERY: &str = "edit-context-bytes";
const BROWSER_EDIT_CONTEXT_CACHE: &str = "burn-image-edit-context-v1";
const BROWSER_EDIT_CONTEXT_MAX_BYTES: u64 = 64 * 1024 * 1024;

/// Pending browser navigation after a generated image has been committed to a one-shot handoff.
///
/// Success unloads the page, so only a failure can become observable to the originating Bevy app.
pub(crate) struct BrowserModelSwitchTask {
    failure: Arc<Mutex<Option<String>>>,
}

impl BrowserModelSwitchTask {
    pub(crate) fn take_failure(&self) -> Option<String> {
        self.failure.lock().ok()?.take()
    }
}

/// One-shot generated-image restoration owned by the destination Edit page.
type BrowserEditContextRestoreSlot = Arc<Mutex<Option<Result<Vec<u8>, String>>>>;
type BrowserEditContextRestoreRequest = Result<Option<BrowserEditContextRestoreTask>, String>;

pub(crate) struct BrowserEditContextRestoreTask {
    result: BrowserEditContextRestoreSlot,
}

impl BrowserEditContextRestoreTask {
    pub(crate) fn try_take(&self) -> Option<Result<Vec<u8>, String>> {
        self.result.lock().ok()?.take()
    }
}

fn browser_edit_context_cache_key(digest: Sha256Digest) -> String {
    format!(
        "https://burn-image.invalid/.well-known/edit-context/v1/{}.png",
        digest.to_hex()
    )
}

fn browser_model_switch_js_error(operation: &str, value: wasm_bindgen::JsValue) -> String {
    let detail = value.as_string().unwrap_or_else(|| format!("{value:?}"));
    format!("browser model switch {operation} failed: {detail}")
}

async fn store_browser_edit_context(bytes: &[u8], digest: Sha256Digest) -> Result<(), String> {
    let byte_count = u64::try_from(bytes.len())
        .map_err(|_| "browser Edit context byte length does not fit u64".to_owned())?;
    if byte_count == 0 || byte_count > BROWSER_EDIT_CONTEXT_MAX_BYTES {
        return Err(format!(
            "browser Edit context must contain 1..={BROWSER_EDIT_CONTEXT_MAX_BYTES} PNG bytes, found {byte_count}"
        ));
    }
    let cache = open_browser_artifact_cache(BROWSER_EDIT_CONTEXT_CACHE)
        .await
        .map_err(|error| format!("browser model switch handoff cache open failed: {error}"))?;
    browser_cache_put(
        &cache,
        BROWSER_EDIT_CONTEXT_CACHE,
        &browser_edit_context_cache_key(digest),
        bytes,
    )
    .await
    .map_err(|error| format!("browser model switch handoff cache write failed: {error}"))
}

async fn read_browser_edit_context(
    digest: Sha256Digest,
    expected_bytes: u64,
) -> Result<Vec<u8>, String> {
    if expected_bytes == 0 || expected_bytes > BROWSER_EDIT_CONTEXT_MAX_BYTES {
        return Err(format!(
            "browser Edit context length must be within 1..={BROWSER_EDIT_CONTEXT_MAX_BYTES}, found {expected_bytes}"
        ));
    }
    let cache = open_browser_artifact_cache(BROWSER_EDIT_CONTEXT_CACHE)
        .await
        .map_err(|error| format!("browser model switch handoff cache open failed: {error}"))?;
    let key = browser_edit_context_cache_key(digest);
    let read = browser_cache_match(&cache, BROWSER_EDIT_CONTEXT_CACHE, &key, expected_bytes)
        .await
        .map_err(|error| format!("browser model switch handoff cache read failed: {error}"))?
        .ok_or_else(|| {
            "The generated image handoff is missing; select the Edit model again".to_owned()
        })
        .and_then(|bytes| {
            if bytes.is_empty() {
                return Err("browser Edit context has an invalid cached representation".into());
            }
            if Sha256Digest::calculate(&bytes) != digest {
                return Err("browser Edit context failed its SHA-256 check".into());
            }
            Ok(bytes)
        });

    let deleted = browser_cache_delete(&cache, BROWSER_EDIT_CONTEXT_CACHE, &key)
        .await
        .map_err(|error| format!("browser model switch handoff cache cleanup failed: {error}"))?;
    if !deleted && read.is_ok() {
        return Err("browser Edit context could not be removed after its one-shot read".into());
    }
    read
}

fn remove_browser_edit_context_query(window: &web_sys::Window) -> Result<(), String> {
    let href = window
        .location()
        .href()
        .map_err(|error| browser_model_switch_js_error("handoff URL read", error))?;
    let url = web_sys::Url::new(&href)
        .map_err(|error| browser_model_switch_js_error("handoff URL parse", error))?;
    url.search_params().delete(BROWSER_EDIT_CONTEXT_QUERY);
    url.search_params().delete(BROWSER_EDIT_CONTEXT_BYTES_QUERY);
    window
        .history()
        .map_err(|error| browser_model_switch_js_error("handoff history access", error))?
        .replace_state_with_url(&wasm_bindgen::JsValue::NULL, "", Some(&url.href()))
        .map_err(|error| browser_model_switch_js_error("handoff URL cleanup", error))
}

/// Begins a one-shot read of the generated image carried across a browser model-release reload.
pub(crate) fn request_browser_edit_context_restore() -> BrowserEditContextRestoreRequest {
    let window = web_sys::window().ok_or_else(|| "browser Window is unavailable".to_owned())?;
    let search = window
        .location()
        .search()
        .map_err(|error| browser_model_switch_js_error("handoff query read", error))?;
    let params = web_sys::UrlSearchParams::new_with_str(&search)
        .map_err(|error| browser_model_switch_js_error("handoff query parse", error))?;
    let expected = match (
        params.get(BROWSER_EDIT_CONTEXT_QUERY),
        params.get(BROWSER_EDIT_CONTEXT_BYTES_QUERY),
    ) {
        (None, None) => return Ok(None),
        (Some(digest), Some(bytes)) => (digest, bytes),
        _ => return Err("browser Edit context query is incomplete".into()),
    };
    if !matches!(
        params.get("variant").as_deref(),
        Some("edit-turbo" | "edit-turbo-1k5")
    ) {
        return Err("browser Edit context is present on a non-Edit model release".into());
    }
    let digest = Sha256Digest::from_hex(&expected.0)
        .map_err(|error| format!("browser Edit context digest is invalid: {error}"))?;
    if digest.to_hex() != expected.0 {
        return Err("browser Edit context digest must use canonical lowercase hexadecimal".into());
    }
    let expected_bytes = expected
        .1
        .parse::<u64>()
        .map_err(|error| format!("browser Edit context byte length is invalid: {error}"))?;
    if expected_bytes == 0 || expected_bytes > BROWSER_EDIT_CONTEXT_MAX_BYTES {
        return Err(format!(
            "browser Edit context length must be within 1..={BROWSER_EDIT_CONTEXT_MAX_BYTES}, found {expected_bytes}"
        ));
    }
    let result = Arc::new(Mutex::new(None));
    let task = BrowserEditContextRestoreTask {
        result: Arc::clone(&result),
    };
    spawn_local(async move {
        let read = read_browser_edit_context(digest, expected_bytes).await;
        let query_cleanup = remove_browser_edit_context_query(&window);
        let restored = match (read, query_cleanup) {
            (Ok(bytes), Ok(())) => Ok(bytes),
            (Err(error), Ok(())) => Err(error),
            (Ok(_), Err(error)) => Err(error),
            (Err(error), Err(cleanup)) => {
                Err(format!("{error}; URL cleanup also failed: {cleanup}"))
            }
        };
        if let Ok(mut slot) = result.lock() {
            *slot = Some(restored);
        }
    });
    Ok(Some(task))
}

/// Reload the canonical browser release selected by the Bevy model dropdown.
///
/// A navigation is the browser's model-switch boundary: it drops the old runtime/device tensors
/// before the target release begins its VRAM preflight and verified artifact load. Custom artifact
/// mirrors and no-surface diagnostics remain pinned to their one exact release.
pub(crate) fn request_browser_model_release(
    model: &burn_image::ModelId,
    edit_context_png: Option<Vec<u8>>,
) -> Result<Option<BrowserModelSwitchTask>, String> {
    let Some(variant) = crate::boogu::variant_for_model(model) else {
        return Err(format!(
            "browser model switch received unknown model {model}"
        ));
    };
    if !browser_release_switching_enabled() {
        return Ok(None);
    }
    let window = web_sys::window().ok_or_else(|| "browser Window is unavailable".to_owned())?;
    let href = window
        .location()
        .href()
        .map_err(|error| format!("browser model switch could not read the page URL: {error:?}"))?;
    let url = web_sys::Url::new(&href)
        .map_err(|error| format!("browser model switch could not parse the page URL: {error:?}"))?;
    let params = url.search_params();
    if params.get("variant").as_deref() == Some(variant_slug(variant)) {
        return Ok(None);
    }
    params.set("variant", variant_slug(variant));
    if params
        .get("residency")
        .is_some_and(|value| value != "resident" && value != "low-vram")
    {
        params.set("residency", "low-vram");
    }
    let context = edit_context_png.map(|bytes| {
        let digest = Sha256Digest::calculate(&bytes);
        let byte_count = bytes.len() as u64;
        params.set(BROWSER_EDIT_CONTEXT_QUERY, &digest.to_hex());
        params.set(BROWSER_EDIT_CONTEXT_BYTES_QUERY, &byte_count.to_string());
        (bytes, digest)
    });
    if context.is_none() {
        params.delete(BROWSER_EDIT_CONTEXT_QUERY);
        params.delete(BROWSER_EDIT_CONTEXT_BYTES_QUERY);
    }
    let target = url.href();
    let failure = Arc::new(Mutex::new(None));
    let task = BrowserModelSwitchTask {
        failure: Arc::clone(&failure),
    };
    spawn_local(async move {
        let result = async {
            if let Some((bytes, digest)) = context {
                store_browser_edit_context(&bytes, digest).await?;
            }
            window
                .location()
                .assign(&target)
                .map_err(|error| browser_model_switch_js_error("navigation", error))?;
            Ok::<(), String>(())
        }
        .await;
        if let Err(error) = result
            && let Ok(mut slot) = failure.lock()
        {
            *slot = Some(error);
        }
    });
    Ok(Some(task))
}

impl BooguRuntime for BrowserBooguRuntime {
    fn variants(&self) -> Vec<BooguVariant> {
        if browser_release_switching_enabled() {
            vec![
                BooguVariant::Image01Turbo,
                BooguVariant::Image01EditTurbo,
                BooguVariant::Image01EditTurbo1k5,
            ]
        } else {
            vec![self.variant]
        }
    }

    fn readiness(&self) -> crate::ImageRunnerReadiness {
        let state = self.shared.lock().expect("browser runtime mutex poisoned");
        let Some(engine) = state.engine.as_ref() else {
            return crate::ImageRunnerReadiness::default();
        };
        crate::ImageRunnerReadiness {
            transfer: engine.artifact_control.transfer_progress(),
            selected_model_device_resident: engine.policies.eager_preload,
        }
    }

    fn submit(&mut self, job: BooguRuntimeJob) -> Result<CancellationToken, RuntimeError> {
        if job.variant != self.variant {
            return Err(execution_error(
                self.variant,
                "browser runtime received a different Boogu release",
            ));
        }
        let cancellation = CancellationToken::default();
        let mut state = self.shared.lock().expect("browser runtime mutex poisoned");
        if state.active.is_some() {
            return Err(execution_error(
                self.variant,
                "browser runtime executes one request at a time",
            ));
        }
        let mut engine = state.engine.take().ok_or_else(|| {
            execution_error(self.variant, "browser runtime engine is unavailable")
        })?;
        state.active = Some((job.id, cancellation.clone()));
        drop(state);

        let shared = self.shared.clone();
        let observer_shared = shared.clone();
        let id = job.id;
        let run_id = RunId(id.0);
        engine
            .artifact_control
            .set_cancellation(Some(cancellation.clone()));
        install_artifact_observer(&engine.artifact_control, &observer_shared, id, run_id);
        let returned_token = cancellation.clone();
        spawn_local(async move {
            let mut result = engine.infer(&job, &cancellation, &shared).await;
            // Keep the request-wide surface gate active until every packed-F16 failure or
            // cancellation has crossed a real queue barrier, dropped all retained denoiser/Qwen
            // pages, run allocator cleanup, and crossed the post-cleanup barrier. Inference has
            // returned here, so all request-local tensor handles are already out of scope. This
            // single outer boundary covers early Qwen/artifact errors as well as DMD/VAE errors;
            // the cleanup is deliberately idempotent when an inner boundary already ran it.
            if result.is_err() && engine.policies.uses_packed_f16_denoiser_source() {
                let cleanup_result = engine.fail_closed_packed_f16_request_cleanup().await;
                if let Err(cleanup_error) = cleanup_result {
                    let primary = match result {
                        Err(RuntimeError::Cancelled) => "browser request was cancelled".into(),
                        Err(ref error) => error.to_string(),
                        Ok(_) => unreachable!("cleanup is only entered after inference failure"),
                    };
                    result = Err(execution_error(
                        job.variant,
                        format!(
                            "{primary}; fail-closed packed-F16 terminal cleanup also failed: {cleanup_error}"
                        ),
                    ));
                }
            }
            engine.artifact_control.set_observer(None);
            engine.artifact_control.set_cancellation(None);
            let terminal = match result {
                Ok(output) => ImageRunnerEvent::Completed { id, output },
                Err(RuntimeError::Cancelled) => {
                    queue_progress(&shared, id, ProgressEvent::RunCancelled { run_id });
                    ImageRunnerEvent::Cancelled { id }
                }
                Err(error) => {
                    queue_progress(
                        &shared,
                        id,
                        ProgressEvent::RunFailed {
                            run_id,
                            message: error.to_string(),
                        },
                    );
                    ImageRunnerEvent::Failed { id, error }
                }
            };
            let mut state = shared.lock().expect("browser runtime mutex poisoned");
            state.engine = Some(engine);
            state.active = None;
            push_runtime_event(&mut state.events, terminal);
        });
        Ok(returned_token)
    }

    fn cancel(&mut self, id: ImageJobId) -> Result<(), RuntimeError> {
        let state = self.shared.lock().expect("browser runtime mutex poisoned");
        if let Some((active_id, cancellation)) = &state.active
            && *active_id == id
        {
            cancellation.cancel();
        }
        Ok(())
    }

    fn poll(&mut self, emit: &mut dyn FnMut(ImageRunnerEvent)) {
        let mut state = self.shared.lock().expect("browser runtime mutex poisoned");
        for _ in 0..MAX_EVENTS_PER_POLL {
            let Some(event) = state.events.pop_front() else {
                break;
            };
            emit(event);
        }
    }
}

fn install_artifact_observer(
    control: &BrowserArtifactControl,
    shared: &Arc<Mutex<BrowserRuntimeShared>>,
    id: ImageJobId,
    run_id: RunId,
) {
    control.set_transfer_phase("Inference model transfer");
    control.start_request_transfer_activity();
    let observer_shared = Arc::clone(shared);
    let progress_control = control.clone();
    control.set_observer(Some(Arc::new(move |event| {
        let progress =
            browser_artifact_progress(run_id, event, progress_control.transfer_progress());
        queue_progress(&observer_shared, id, progress);
    })));
}

fn browser_artifact_progress(
    run_id: RunId,
    event: BrowserArtifactEvent,
    transfer: Option<burn_image::ArtifactTransferProgress>,
) -> ProgressEvent {
    match event {
        BrowserArtifactEvent::Started(file) => {
            let (file_index, file_count) = artifact_progress_position(&file);
            ProgressEvent::ArtifactStarted {
                run_id,
                path: file.path,
                component: file.component,
                file_index,
                file_count,
                total_bytes: file.size,
                transfer,
            }
        }
        BrowserArtifactEvent::Progress {
            path,
            loaded_bytes,
            total_bytes,
        } => ProgressEvent::ArtifactProgress {
            run_id,
            path,
            loaded_bytes,
            total_bytes,
            transfer,
        },
        BrowserArtifactEvent::Verified(path) => ProgressEvent::ArtifactVerified {
            run_id,
            path,
            transfer,
        },
    }
}

async fn read_manifest_file(
    reader: &mut BrowserStageShardReader,
    manifest: &ArtifactManifest,
    path: &str,
    variant: BooguVariant,
) -> Result<Vec<u8>, RuntimeError> {
    let file = manifest_file(manifest, path, variant)?;
    reader
        .read_verified(&file)
        .await
        .map_err(|error| map_boogu(variant, error))
}

fn manifest_file(
    manifest: &ArtifactManifest,
    path: &str,
    variant: BooguVariant,
) -> Result<ArtifactFile, RuntimeError> {
    manifest
        .files
        .iter()
        .find(|file| file.path.as_str() == path)
        .cloned()
        .ok_or_else(|| execution_error(variant, format!("sealed manifest omits {path}")))
}

fn utf8(bytes: &[u8], variant: BooguVariant) -> Result<&str, RuntimeError> {
    std::str::from_utf8(bytes).map_err(|error| execution_error(variant, error))
}

fn cast_visual_inputs(input: &mut burn_qwen3_vl::Qwen3VlModelInput<BrowserBackend>, dtype: DType) {
    for visual in [&mut input.images, &mut input.videos].into_iter().flatten() {
        visual.patches = visual.patches.clone().cast(dtype);
    }
}

fn normal_tensor<const D: usize>(
    shape: [usize; D],
    seed: u64,
    dtype: DType,
    device: &burn_wgpu::WgpuDevice,
) -> Tensor<BrowserBackend, D> {
    let mut rng = ChaCha12Rng::seed_from_u64(seed);
    let values = StandardNormal
        .sample_iter(&mut rng)
        .take(shape.iter().product())
        .collect::<Vec<f32>>();
    Tensor::<BrowserBackend, D>::from_data(TensorData::new(values, shape), device).cast(dtype)
}

async fn require_finite_browser_tensor<const D: usize>(
    variant: BooguVariant,
    name: &str,
    tensor: &Tensor<BrowserBackend, D>,
) -> Result<(), RuntimeError> {
    // These reductions intentionally synchronize and read one scalar per boundary. Keep them
    // confined to the rendered qualification harness; ordinary production inference must not
    // acquire diagnostic device-to-host barriers in its hot path.
    if !rendered_model_smoke_requested() {
        return Ok(());
    }
    let all_finite = tensor
        .clone()
        .is_finite()
        .all()
        .into_scalar_async()
        .await
        .map_err(|error| {
            execution_error(
                variant,
                format!("WebGPU finiteness readback for {name} failed: {error}"),
            )
        })?;
    if all_finite != 0 {
        Ok(())
    } else {
        Err(execution_error(
            variant,
            format!("{name} contains non-finite values"),
        ))
    }
}

fn rendered_model_smoke_requested() -> bool {
    web_sys::window()
        .and_then(|window| window.location().search().ok())
        .and_then(|search| web_sys::UrlSearchParams::new_with_str(&search).ok())
        .is_some_and(|params| params.get("rendered-model-smoke").as_deref() == Some("1"))
}

/// Request-local distinction between the serialized localization branch and the ordinary model
/// path. The query is meaningful only for rendered-model smoke; production pages always report
/// `ordinary`. An unknown exact value reports `invalid` so the evidence contract fails closed.
fn browser_qwen_block0_execution_mode() -> &'static str {
    let Some(params) = web_sys::window()
        .and_then(|window| window.location().search().ok())
        .and_then(|search| web_sys::UrlSearchParams::new_with_str(&search).ok())
    else {
        return BROWSER_QWEN_BLOCK0_ORDINARY_MODE;
    };
    if params.get("rendered-model-smoke").as_deref() != Some("1") {
        return BROWSER_QWEN_BLOCK0_ORDINARY_MODE;
    }
    match params
        .get(BROWSER_QWEN_BLOCK0_EXECUTION_MODE_QUERY)
        .as_deref()
    {
        None | Some(BROWSER_QWEN_BLOCK0_ORDINARY_MODE) => BROWSER_QWEN_BLOCK0_ORDINARY_MODE,
        Some(BROWSER_QWEN_BLOCK0_SERIALIZED_DIAGNOSTIC_MODE) => {
            BROWSER_QWEN_BLOCK0_SERIALIZED_DIAGNOSTIC_MODE
        }
        Some(_) => "invalid",
    }
}

async fn read_packed_f16_tensor_input_diagnostic<const D: usize>(
    variant: BooguVariant,
    name: &str,
    tensor: &Tensor<BrowserBackend, D>,
) -> Result<BrowserPackedF16TensorInputDiagnostic, RuntimeError> {
    let shape = tensor.dims().to_vec();
    if tensor.dtype() != DType::F32 {
        return Err(execution_error(
            variant,
            format!(
                "rendered-smoke input {name} must be exact F32 before DMD, got {}",
                tensor.dtype().name()
            ),
        ));
    }
    let data = tensor.clone().into_data_async().await.map_err(|error| {
        execution_error(variant, format!("F32 {name} readback failed: {error}"))
    })?;
    packed_f16_tensor_data_input_diagnostic(variant, name, shape, &data)
}

fn packed_f16_tensor_data_input_diagnostic(
    variant: BooguVariant,
    name: &str,
    shape: Vec<usize>,
    data: &TensorData,
) -> Result<BrowserPackedF16TensorInputDiagnostic, RuntimeError> {
    let expected_elements = shape
        .iter()
        .try_fold(1_usize, |total, dimension| total.checked_mul(*dimension))
        .ok_or_else(|| {
            execution_error(
                variant,
                format!("F32 diagnostic {name} element count overflowed"),
            )
        })?;
    if data.dtype != DType::F32 {
        return Err(execution_error(
            variant,
            format!(
                "F32 diagnostic {name} read back as {}, expected f32",
                data.dtype.name()
            ),
        ));
    }
    let sha256 = Sha256Digest::calculate(data.bytes.as_ref());
    let values = data
        .to_vec::<f32>()
        .map_err(|error| execution_error(variant, error))?;
    if values.len() != expected_elements || values.is_empty() {
        return Err(execution_error(
            variant,
            format!(
                "F32 diagnostic {name} read back {} values for shape {shape:?}",
                values.len()
            ),
        ));
    }

    let finite_element_count = values.iter().filter(|value| value.is_finite()).count();
    let all_finite = finite_element_count == values.len();
    let (max_abs, mean, rms) = if all_finite {
        let mut maximum = 0.0_f64;
        let mut sum = 0.0_f64;
        let mut sum_squares = 0.0_f64;
        for &value in &values {
            let value = f64::from(value);
            maximum = maximum.max(value.abs());
            sum += value;
            sum_squares += value * value;
        }
        let count = values.len() as f64;
        let mean = sum / count;
        let rms = (sum_squares / count).sqrt();
        if !maximum.is_finite() || !mean.is_finite() || !rms.is_finite() {
            return Err(execution_error(
                variant,
                format!("F32 diagnostic {name} statistics overflowed"),
            ));
        }
        (Some(maximum), Some(mean), Some(rms))
    } else {
        (None, None, None)
    };

    Ok(BrowserPackedF16TensorInputDiagnostic {
        name: name.into(),
        shape,
        dtype: "f32".into(),
        element_count: values.len(),
        finite_element_count,
        all_finite,
        max_abs,
        mean,
        rms,
        sha256,
    })
}

fn packed_f16_tensor_diagnostic_is_all_zero(
    diagnostic: &BrowserPackedF16TensorInputDiagnostic,
) -> bool {
    diagnostic.all_finite && diagnostic.max_abs == Some(0.0)
}

fn packed_f16_qwen_stage_diagnostic_names_are_exact(
    diagnostics: &[BrowserPackedF16TensorInputDiagnostic],
    expected_count: usize,
) -> bool {
    if expected_count < 2 || diagnostics.len() != expected_count {
        return false;
    }
    diagnostics
        .first()
        .is_some_and(|diagnostic| diagnostic.name == "qwen_embedding_output")
        && diagnostics[1..expected_count - 1]
            .iter()
            .enumerate()
            .all(|(index, diagnostic)| {
                diagnostic.name == format!("qwen_text_block_{index:02}_output")
            })
        && diagnostics
            .last()
            .is_some_and(|diagnostic| diagnostic.name == "qwen_final_norm_output")
}

fn require_finite_nonzero_packed_f16_diagnostic(
    variant: BooguVariant,
    diagnostic: &BrowserPackedF16TensorInputDiagnostic,
) -> Result<(), RuntimeError> {
    if !diagnostic.all_finite {
        return Err(execution_error(
            variant,
            format!("packed-F16 {} is non-finite", diagnostic.name),
        ));
    }
    if packed_f16_tensor_diagnostic_is_all_zero(diagnostic) {
        return Err(execution_error(
            variant,
            format!("packed-F16 {} is all-zero", diagnostic.name),
        ));
    }
    Ok(())
}

fn domain_seed(seed: u64, domain: u64) -> u64 {
    let mut value = seed ^ domain;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

async fn decoder_output_to_host_async(
    output: Tensor<BrowserBackend, 4>,
) -> Result<HostImage, BooguError> {
    let data = output
        .into_data_async()
        .await
        .map_err(|error| BooguError::InvalidRequest(format!("WebGPU readback failed: {error}")))?
        .convert_dtype(DType::F32);
    decoder_output_data_to_host(data)
}

fn require_dtype(
    variant: BooguVariant,
    name: &str,
    actual: DType,
    expected: DType,
) -> Result<(), RuntimeError> {
    if actual == expected {
        Ok(())
    } else {
        Err(execution_error(
            variant,
            format!(
                "{name} has dtype {}, expected {}",
                actual.name(),
                expected.name()
            ),
        ))
    }
}

fn start_stage(
    shared: &Arc<Mutex<BrowserRuntimeShared>>,
    id: ImageJobId,
    run_id: RunId,
    stage: &str,
    total_steps: Option<u32>,
) -> u64 {
    queue_progress(
        shared,
        id,
        ProgressEvent::StageStarted {
            run_id,
            stage: stage.into(),
            total_steps,
        },
    );
    now_micros()
}

fn finish_stage(
    shared: &Arc<Mutex<BrowserRuntimeShared>>,
    id: ImageJobId,
    run_id: RunId,
    timings: &mut Vec<StageTiming>,
    stage: &str,
    started: u64,
) {
    let elapsed_micros = now_micros().saturating_sub(started);
    timings.push(StageTiming {
        stage: stage.into(),
        elapsed_micros,
    });
    queue_progress(
        shared,
        id,
        ProgressEvent::StageCompleted {
            run_id,
            stage: stage.into(),
            elapsed_micros,
        },
    );
}

fn queue_progress(shared: &Arc<Mutex<BrowserRuntimeShared>>, id: ImageJobId, event: ProgressEvent) {
    dispatch_browser_progress(&event);
    let mut state = shared.lock().expect("browser runtime mutex poisoned");
    let diagnostic = if state.diagnostic_console_progress {
        let current_memory = wasm_linear_memory_bytes().unwrap_or(0);
        state.diagnostic_peak_wasm_linear_memory_bytes = state
            .diagnostic_peak_wasm_linear_memory_bytes
            .max(current_memory);
        diagnostic_progress_message(&event).map(|message| {
            format!(
                "{message} wasm_linear_bytes={current_memory} peak_wasm_linear_bytes={}",
                state.diagnostic_peak_wasm_linear_memory_bytes
            )
        })
    } else {
        None
    };
    push_runtime_event(&mut state.events, ImageRunnerEvent::Progress { id, event });
    drop(state);
    if let Some(message) = diagnostic {
        report_diagnostic_progress(&message);
    }
}

fn push_runtime_event(events: &mut VecDeque<ImageRunnerEvent>, event: ImageRunnerEvent) {
    if events.len() == MAX_RUNTIME_EVENTS {
        events.pop_front();
    }
    events.push_back(event);
}

fn check_cancelled(cancellation: &CancellationToken) -> Result<(), RuntimeError> {
    cancellation.check()
}

fn task_kind(task: BooguTask) -> ImageTaskKind {
    match task {
        BooguTask::Generate => ImageTaskKind::Generate,
        BooguTask::Edit => ImageTaskKind::Edit,
    }
}

fn probe_milestone(milestone: &str) {
    web_sys::console::info_1(&format!("BURN_IMAGE_HEADLESS_MILESTONE {milestone}").into());
}

fn browser_stage_milestone(milestone: &str) {
    web_sys::console::info_1(&format!("BURN_IMAGE_BROWSER_STAGE_MILESTONE {milestone}").into());
}

fn inference_milestone(milestone: &str) {
    report_diagnostic_progress(milestone);
}

fn parity_milestone(milestone: &str) {
    web_sys::console::info_1(&format!("BURN_IMAGE_HEADLESS_PARITY_PROGRESS {milestone}").into());
    if let Some(status) = web_sys::window()
        .and_then(|window| window.document())
        .and_then(|document| document.get_element_by_id("status"))
    {
        status.set_text_content(Some(&format!(
            "BURN_IMAGE_HEADLESS_PARITY_PROGRESS {milestone}"
        )));
    }
}

fn browser_parity_readback_milestone(
    phase: &str,
    tensor: &str,
    elements: usize,
    chunk_index: usize,
    chunk_count: usize,
) {
    let chunk = if chunk_count == 0 {
        "whole".to_owned()
    } else {
        format!("{}/{}", chunk_index + 1, chunk_count)
    };
    web_sys::console::info_1(
        &format!(
            "BURN_IMAGE_HEADLESS_PARITY_READBACK phase={phase} tensor={tensor} elements={elements} chunk={chunk}"
        )
        .into(),
    );
}

fn vae_reference_milestone(milestone: &str) {
    web_sys::console::info_1(
        &format!("BURN_IMAGE_HEADLESS_VAE_REFERENCE_PROGRESS {milestone}").into(),
    );
    if let Some(status) = web_sys::window()
        .and_then(|window| window.document())
        .and_then(|document| document.get_element_by_id("status"))
    {
        status.set_text_content(Some(&format!(
            "BURN_IMAGE_HEADLESS_VAE_REFERENCE_PROGRESS {milestone}"
        )));
    }
}

fn diagnostic_progress_message(event: &ProgressEvent) -> Option<String> {
    match event {
        ProgressEvent::RunStarted { task, .. } => Some(format!("run-started task={task:?}")),
        ProgressEvent::ArtifactStarted {
            path, total_bytes, ..
        } => Some(format!("artifact-started path={path} bytes={total_bytes}")),
        ProgressEvent::StageStarted {
            stage, total_steps, ..
        } => Some(format!("stage-started stage={stage} steps={total_steps:?}")),
        ProgressEvent::Step {
            stage,
            step,
            total_steps,
            ..
        } => Some(format!("step stage={stage} value={step}/{total_steps}")),
        ProgressEvent::StageCompleted {
            stage,
            elapsed_micros,
            ..
        } => Some(format!(
            "stage-completed stage={stage} elapsed_micros={elapsed_micros}"
        )),
        ProgressEvent::Warning { message, .. } => Some(format!("warning {message}")),
        ProgressEvent::RunCompleted { elapsed_micros, .. } => {
            Some(format!("run-completed elapsed_micros={elapsed_micros}"))
        }
        ProgressEvent::RunFailed { message, .. } => Some(format!("run-failed {message}")),
        ProgressEvent::RunCancelled { .. } => Some("run-cancelled".into()),
        ProgressEvent::ArtifactProgress { .. } | ProgressEvent::ArtifactVerified { .. } => None,
    }
}

fn report_diagnostic_progress(message: &str) {
    let message = format!("BURN_IMAGE_HEADLESS_INFER_PROGRESS {message}");
    web_sys::console::info_1(&message.clone().into());
    if let Some(status) = web_sys::window()
        .and_then(|window| window.document())
        .and_then(|document| document.get_element_by_id("status"))
    {
        status.set_text_content(Some(&message));
    }
}

fn wasm_linear_memory_bytes() -> Option<u64> {
    use wasm_bindgen::JsCast;

    let memory = wasm_bindgen::memory();
    let memory = memory.dyn_ref::<js_sys::WebAssembly::Memory>()?;
    let buffer = memory.buffer();
    let buffer = buffer.dyn_ref::<js_sys::ArrayBuffer>()?;
    Some(u64::from(buffer.byte_length()))
}

/// Attach the completed diagnostic PNG to the host page and expose an explicit download link.
pub fn attach_headless_inference_png(
    png: &[u8],
    file_name: &str,
) -> Result<(), wasm_bindgen::JsValue> {
    use wasm_bindgen::JsCast;

    let bytes = js_sys::Uint8Array::from(png);
    let parts = js_sys::Array::new();
    parts.push(&bytes);
    let options = web_sys::BlobPropertyBag::new();
    options.set_type("image/png");
    let blob = web_sys::Blob::new_with_u8_array_sequence_and_options(&parts, &options)?;
    let url = web_sys::Url::create_object_url_with_blob(&blob)?;
    let document = web_sys::window()
        .and_then(|window| window.document())
        .ok_or_else(|| wasm_bindgen::JsValue::from_str("browser document is unavailable"))?;
    if let Some(canvas) = document.get_element_by_id("burn-image") {
        canvas.set_attribute("hidden", "")?;
    }
    let image = document.create_element("img")?;
    image.set_attribute("id", "burn-image-headless-result")?;
    image.set_attribute("src", &url)?;
    image.set_attribute("alt", "Boogu surface-free inference result")?;
    image.set_attribute(
        "style",
        "display:block;max-width:100%;height:auto;margin:3rem auto 1rem",
    )?;
    let download = document
        .create_element("a")?
        .dyn_into::<web_sys::HtmlAnchorElement>()?;
    download.set_id("burn-image-headless-download");
    download.set_href(&url);
    download.set_download(file_name);
    download.set_text_content(Some(&format!("Download {file_name}")));
    download.set_attribute("style", "display:block;color:white;text-align:center")?;
    let body = document
        .body()
        .ok_or_else(|| wasm_bindgen::JsValue::from_str("browser body is unavailable"))?;
    body.append_child(&image)?;
    body.append_child(&download)?;
    Ok(())
}

const fn variant_slug(variant: BooguVariant) -> &'static str {
    match variant {
        BooguVariant::Image01Turbo => "turbo",
        BooguVariant::Image01EditTurbo => "edit-turbo",
        BooguVariant::Image01EditTurbo1k5 => "edit-turbo-1k5",
    }
}

fn map_boogu(variant: BooguVariant, error: BooguError) -> RuntimeError {
    match error {
        BooguError::Cancelled => RuntimeError::Cancelled,
        error => execution_error(variant, error),
    }
}

fn execution_error(variant: BooguVariant, error: impl std::fmt::Display) -> RuntimeError {
    RuntimeError::ModelExecution {
        model: boogu_model_descriptor(variant).id,
        message: error.to_string(),
    }
}

fn now_micros() -> u64 {
    (js_sys::Date::now() * 1_000.0).max(0.0) as u64
}

#[cfg(test)]
mod browser_source_tests {
    include!("runtime/tests.rs");
}
