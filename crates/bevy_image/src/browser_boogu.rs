//! Concrete browser-local Boogu runtime over verified bounded HTTP Range reads.
//!
//! The supported browser runtime eagerly verifies and materializes every Qwen, VAE, and denoiser
//! stage once, retains only initialized WebGPU module handles, and executes inference without model
//! artifact transport in its hot path. Explicit low-memory diagnostics retain one semantic stage at
//! a time. The dedicated 1.5K parity route lazily retains denoiser handles across four DMD steps and
//! clears them before its exact decoder. No route falls back to CPU or manufactures placeholder
//! output.

use std::{
    collections::VecDeque,
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
};

use burn::{
    nn::RmsNorm,
    tensor::{DType, Tensor, TensorData},
};
use burn_boogu::{
    AsyncBooguDenoiserStageSource, AsyncBooguVaeStageSource, AsyncFluxVaeStageSourceAdapter,
    AsyncRetainingDenoiserSynchronizationPolicy, BooguConfig, BooguDenoiserInput,
    BooguDenoiserPrelude, BooguDenoiserTail, BooguError, BooguRuntimeDTypes, BooguTask,
    BooguVariant, DenoiserStageObserver, DmdSchedule, DoubleStreamBlock,
    RetainingAsyncBooguDenoiserStageSource, RetainingAsyncBooguVaeStageSource, SingleStreamBlock,
    StreamingBooguDenoiser,
    artifacts::{
        BooguArtifactInventory, BooguFloatLoadPolicy, BooguQuantizedLoadPolicy,
        BooguReleaseIdentity, BooguStorageProfile, VerifiedAsyncBurnpackDenoiserStageSource,
        VerifiedAsyncBurnpackQwenStageSource, VerifiedAsyncBurnpackVaeStageSource,
        artifact_bundle_id_is_compatible, canonical_published_bundle,
        validate_canonical_release_artifact_digest,
    },
    boogu_model_descriptor, boogu_processor_config, decode_input_image,
    decoder_output_data_to_host, dmd_prediction, dmd_renoise, encode_reference,
    prepare_instruction, prepare_vae_reference, resolve_request, trim_instruction_features,
};
use burn_flux_vae::{
    AutoencoderKl, AutoencoderKlConfig, DiagonalGaussian, FluxVaeArtifactFloatPolicy,
    FluxVaeComponentContract, VerifiedAsyncBurnpackFluxVaeStageSource,
};
use burn_image::{
    ArtifactDependency, ArtifactFile, ArtifactManifest, ArtifactPath, CancellationToken,
    Dimensions, EditRequest, EncodedImage, GeneratedImage, GenerationOptions, HostImage,
    ImageEncoding, ImageOutput, ImageRequest, ImageTaskKind, InputImage, ModelProvenance,
    ProgressEvent, Prompt, RemoteBaseUrl, RunId, RuntimeError, Sha256Digest, StageTiming,
    StageTimings,
};
use burn_qwen3_vl::{
    AsyncQwen3VlStageSource, AsyncRetainingSynchronizationPolicy, EmbeddingRowChunk,
    Qwen3VlArtifactFloatPolicy, Qwen3VlComponentContract, Qwen3VlConfig, Qwen3VlDecoderLayer,
    Qwen3VlImageProcessor, Qwen3VlImageProcessorConfig, Qwen3VlProcessor, Qwen3VlStage,
    Qwen3VlStageObserver, Qwen3VlTokenizer, Qwen3VlVisionBlock, Qwen3VlVisionPatchMerger,
    Qwen3VlVisionPrelude, RetainingAsyncQwen3VlStageSource, RowChunkSpec, StreamingQwen3Vl,
    VerifiedAsyncBurnpackQwen3VlStageSource, tokenizer::HfTokenizer,
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
        BrowserStageShardReader, MAX_BROWSER_MANIFEST_BYTES, artifact_progress_position,
        fetch_browser_bounded_file, sibling_bundle_base_url,
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
type LegacyBrowserVerifiedQwenSource =
    VerifiedAsyncBurnpackQwenStageSource<BrowserBackend, BrowserStageShardReader>;
type ComponentBrowserVerifiedQwenSource =
    VerifiedAsyncBurnpackQwen3VlStageSource<BrowserBackend, BrowserStageShardReader>;
type LegacyBrowserVerifiedVaeSource =
    VerifiedAsyncBurnpackVaeStageSource<BrowserBackend, BrowserStageShardReader>;
type ComponentBrowserVerifiedVaeSource = AsyncFluxVaeStageSourceAdapter<
    VerifiedAsyncBurnpackFluxVaeStageSource<BrowserBackend, BrowserStageShardReader>,
>;
enum BrowserVerifiedQwenSource {
    Legacy(LegacyBrowserVerifiedQwenSource),
    Component(ComponentBrowserVerifiedQwenSource),
}

impl AsyncQwen3VlStageSource<BrowserBackend> for BrowserVerifiedQwenSource {
    type Error = BooguError;

    async fn load_embedding_rows(
        &mut self,
        spec: &RowChunkSpec,
    ) -> Result<EmbeddingRowChunk<BrowserBackend>, Self::Error> {
        match self {
            Self::Legacy(source) => source.load_embedding_rows(spec).await,
            Self::Component(source) => source
                .load_embedding_rows(spec)
                .await
                .map_err(component_qwen_error),
        }
    }

    async fn load_vision_prelude(
        &mut self,
    ) -> Result<Qwen3VlVisionPrelude<BrowserBackend>, Self::Error> {
        match self {
            Self::Legacy(source) => source.load_vision_prelude().await,
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
            Self::Legacy(source) => source.load_vision_block(index).await,
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
            Self::Legacy(source) => source.load_vision_deepstack_merger(index).await,
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
            Self::Legacy(source) => source.load_vision_final_merger().await,
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
            Self::Legacy(source) => source.load_text_block(index).await,
            Self::Component(source) => source
                .load_text_block(index)
                .await
                .map_err(component_qwen_error),
        }
    }

    async fn load_text_final_norm(&mut self) -> Result<RmsNorm<BrowserBackend>, Self::Error> {
        match self {
            Self::Legacy(source) => source.load_text_final_norm().await,
            Self::Component(source) => source
                .load_text_final_norm()
                .await
                .map_err(component_qwen_error),
        }
    }

    async fn synchronize(&mut self) -> Result<(), Self::Error> {
        match self {
            Self::Legacy(source) => source.synchronize().await,
            Self::Component(source) => source.synchronize().await.map_err(component_qwen_error),
        }
    }
}

fn component_qwen_error(error: impl std::fmt::Display) -> BooguError {
    BooguError::Artifact(error.to_string())
}

enum BrowserVerifiedVaeSource {
    Legacy(LegacyBrowserVerifiedVaeSource),
    Component(ComponentBrowserVerifiedVaeSource),
}

impl AsyncBooguVaeStageSource<BrowserBackend> for BrowserVerifiedVaeSource {
    async fn load_encoder(&mut self) -> Result<AutoencoderKl<BrowserBackend>, BooguError> {
        match self {
            Self::Legacy(source) => source.load_encoder().await,
            Self::Component(source) => source.load_encoder().await,
        }
    }

    async fn load_decoder(&mut self) -> Result<AutoencoderKl<BrowserBackend>, BooguError> {
        match self {
            Self::Legacy(source) => source.load_decoder().await,
            Self::Component(source) => source.load_decoder().await,
        }
    }

    async fn synchronize(&mut self) -> Result<(), BooguError> {
        match self {
            Self::Legacy(source) => source.synchronize().await,
            Self::Component(source) => source.synchronize().await,
        }
    }
}

type BrowserVerifiedDenoiserSource =
    VerifiedAsyncBurnpackDenoiserStageSource<BrowserBackend, BrowserStageShardReader>;
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
const BROWSER_PRODUCTION_DENOISER_QUERY_CHUNK_SIZE: usize = 128;
const BROWSER_1K5_QWEN_QUERY_CHUNK_SIZE: usize = 128;
const BROWSER_1K5_DENOISER_QUERY_CHUNK_SIZE: usize = 1_024;
const BROWSER_1K5_VAE_QUERY_CHUNK_SIZE: usize = 4_096;
const BROWSER_1K5_QWEN_ALIGNED_STAGE_COUNT: usize = 70;
const BROWSER_1K5_DENOISER_BOUNDARY_COUNT: usize = 236;
const BROWSER_1K5_DENOISER_RETAINED_STAGE_COUNT: usize = 48;
const BROWSER_1K5_AUTHENTICATED_ONLY_TENSOR_COUNT: usize = 17;
const BROWSER_1K5_NUMERICAL_TENSOR_COUNT: usize = 355;
const BROWSER_1K5_VAE_REFERENCE_REPEAT_COUNT: usize = 3;
// This identifies the schema-v1 flat closure used to calibrate the component envelope. It is
// provenance only, not a runtime-accepted canonical artifact identity. The schema-v2 modular
// composition must be requalified before its full-chain result is described as current evidence.
const BROWSER_WEBGPU_VAE_F32_ORACLE_LEGACY_FLAT_CONTENT_DIGEST: &str =
    "5d7e25b1d9be1fdf4a6372bfb9db28cf62ef90253082cef22af09653047e3a7b";
const BROWSER_WEBGPU_VAE_F32_ORACLE_ENVELOPE: BrowserWebGpuVaeF32OracleEnvelope =
    BrowserWebGpuVaeF32OracleEnvelope {
        backend: "BrowserWebGpu/raw-cubecl-no-fusion",
        artifact_content_digest: BROWSER_WEBGPU_VAE_F32_ORACLE_LEGACY_FLAT_CONTENT_DIGEST,
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

/// Browser model-weight residency contract.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BrowserBooguResidencyPolicy {
    /// Eagerly verify and materialize every model stage, then keep dense-F32 WebGPU handles resident.
    #[default]
    HighVramResidentDenseF32,
    /// Explicit low-memory diagnostic that reloads one verified semantic stage at a time.
    LayerStreamedDiagnostic,
}

impl BrowserBooguResidencyPolicy {
    /// Stable provenance suffix for reports and output metadata.
    pub const fn label(self) -> &'static str {
        match self {
            Self::HighVramResidentDenseF32 => "browser-high-vram-resident-dense-f32",
            Self::LayerStreamedDiagnostic => "browser-layer-streamed-diagnostic",
        }
    }

    /// Parse the public browser query selector.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "resident" | "high-vram-resident-dense-f32" => Some(Self::HighVramResidentDenseF32),
            "layer-streamed-diagnostic" => Some(Self::LayerStreamedDiagnostic),
            _ => None,
        }
    }
}

#[derive(serde::Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum BrowserRuntimeEvent {
    Preparing {
        message: String,
    },
    ManifestVerified {
        bundle: String,
        weight_objects: u32,
        weight_bytes: u64,
    },
    ResidentResourcePlan {
        stored_weight_bytes: u64,
        conservative_f32_weight_bytes: u64,
        activation_reserve_bytes: u64,
        conservative_planned_device_bytes: u64,
    },
    Ready {
        model: String,
    },
    Failed {
        message: String,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BrowserResidentResourcePlan {
    stored_weight_bytes: u64,
    conservative_f32_weight_bytes: u64,
    activation_reserve_bytes: u64,
    conservative_planned_device_bytes: u64,
}

const BROWSER_RESIDENT_MAX_SIMULTANEOUS_ACTIVATION_BUFFERS: u64 = 8;

fn validate_browser_resident_resource_plan(
    variant: BooguVariant,
    manifest: &ArtifactManifest,
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
    // The production bundle mixes F16 and F32 storage. Doubling the complete physical Burnpack
    // weight payload is a conservative bound for dense-F32 materialization, including headers and
    // alignment. The ordinary browser surface supports 256 square, whose largest qualified applied
    // buffer is bounded below; eight simultaneous buffers conservatively cover the fused workset.
    let conservative_f32_weight_bytes = stored_weight_bytes.checked_mul(2).ok_or_else(|| {
        execution_error(variant, "resident browser F32 weight-byte plan overflowed")
    })?;
    let activation_reserve_bytes = crate::boogu::BOOGU_BROWSER_MAX_APPLIED_BUFFER_BYTES
        .checked_mul(BROWSER_RESIDENT_MAX_SIMULTANEOUS_ACTIVATION_BUFFERS)
        .ok_or_else(|| execution_error(variant, "resident browser activation plan overflowed"))?;
    let conservative_planned_device_bytes = conservative_f32_weight_bytes
        .checked_add(activation_reserve_bytes)
        .ok_or_else(|| execution_error(variant, "resident browser device-byte plan overflowed"))?;
    let plan = BrowserResidentResourcePlan {
        stored_weight_bytes,
        conservative_f32_weight_bytes,
        activation_reserve_bytes,
        conservative_planned_device_bytes,
    };
    dispatch_browser_event(
        BROWSER_RUNTIME_EVENT_NAME,
        &BrowserRuntimeEvent::ResidentResourcePlan {
            stored_weight_bytes: plan.stored_weight_bytes,
            conservative_f32_weight_bytes: plan.conservative_f32_weight_bytes,
            activation_reserve_bytes: plan.activation_reserve_bytes,
            conservative_planned_device_bytes: plan.conservative_planned_device_bytes,
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

pub(crate) fn report_browser_runtime_preparing(message: impl Into<String>) {
    dispatch_browser_event(
        BROWSER_RUNTIME_EVENT_NAME,
        &BrowserRuntimeEvent::Preparing {
            message: message.into(),
        },
    );
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
    dispatch_browser_event(
        BROWSER_RUNTIME_EVENT_NAME,
        &BrowserRuntimeEvent::ManifestVerified {
            bundle: manifest.bundle.to_string(),
            weight_objects: u32::try_from(weight_objects).unwrap_or(u32::MAX),
            weight_bytes,
        },
    );
}

fn report_browser_runtime_ready(variant: BooguVariant) {
    dispatch_browser_event(
        BROWSER_RUNTIME_EVENT_NAME,
        &BrowserRuntimeEvent::Ready {
            model: boogu_model_descriptor(variant).id.to_string(),
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

fn dispatch_browser_progress(event: &ProgressEvent) {
    dispatch_browser_event(BROWSER_PROGRESS_EVENT_NAME, event);
}

fn dispatch_browser_event<T: serde::Serialize>(name: &str, value: &T) {
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
}

async fn compare_browser_tensor<const D: usize>(
    fixture: &BrowserParityFixture,
    ledger: &BrowserParityOracleLedger,
    name: String,
    oracle: String,
    tensor: Tensor<BrowserBackend, D>,
) -> Result<BrowserParityTensorMetric, BooguError> {
    let (shape, actual_dtype, actual) = read_browser_tensor_f32(tensor).await?;
    let actual = actual
        .as_slice::<f32>()
        .map_err(|error| BooguError::Artifact(error.to_string()))?;
    compare_browser_f32_values(fixture, ledger, name, oracle, &shape, &actual_dtype, actual).await
}

async fn read_browser_tensor_f32<const D: usize>(
    tensor: Tensor<BrowserBackend, D>,
) -> Result<(Vec<usize>, String, TensorData), BooguError> {
    let shape = tensor.dims().to_vec();
    let actual_dtype = tensor.dtype().name().to_owned();
    let data = tensor
        .into_data_async()
        .await
        .map_err(|error| BooguError::Artifact(format!("WebGPU parity readback failed: {error}")))?
        .convert_dtype(DType::F32);
    Ok((shape, actual_dtype, data))
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
    parity: Option<BrowserParityControl>,
    denoiser_query_chunk_size: Option<usize>,
}

impl<S> BrowserAsyncStageSource<S> {
    fn new(inner: S, synchronizer: BrowserAsyncSynchronizer) -> Self {
        Self {
            inner,
            synchronizer,
            pending_stage: None,
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

    async fn synchronize_with_parity(&self, stage: &str) -> Result<(), BooguError> {
        self.synchronizer.synchronize(stage).await?;
        if let Some(parity) = &self.parity {
            parity.compare_pending().await?;
        }
        Ok(())
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
        let stage = format!("qwen-text-block-{index:02}");
        browser_stage_milestone(&format!("{stage}-source-load-apply-start"));
        self.pending_stage = Some(stage.clone());
        let layer = self.inner.load_text_block(index).await?;
        browser_stage_milestone(&format!("{stage}-source-load-apply-complete"));
        Ok(layer)
    }

    async fn load_text_final_norm(&mut self) -> Result<RmsNorm<BrowserBackend>, Self::Error> {
        self.inner.load_text_final_norm().await
    }

    async fn synchronize(&mut self) -> Result<(), Self::Error> {
        let stage = self.pending_stage.as_deref().unwrap_or("qwen-stage");
        browser_stage_milestone(&format!("{stage}-post-forward-sync-start"));
        let result = self.synchronize_with_parity(stage).await;
        if result.is_ok() {
            browser_stage_milestone(&format!("{stage}-post-forward-sync-complete"));
            self.pending_stage = None;
        }
        result
    }
}

/// Marks the exact point after a Qwen stage forward has been queued and before its source barrier.
/// Together with the source wrapper milestones this distinguishes load/apply, forward submission,
/// and asynchronous WebGPU synchronization without forcing a readback.
struct BrowserQwenStageObserver {
    parity: Option<BrowserParityControl>,
    multimodal: bool,
    deepstack_count: usize,
}

impl BrowserQwenStageObserver {
    fn milestones_only() -> Self {
        Self {
            parity: None,
            multimodal: false,
            deepstack_count: 0,
        }
    }

    fn parity(control: BrowserParityControl, multimodal: bool, deepstack_count: usize) -> Self {
        Self {
            parity: Some(control),
            multimodal,
            deepstack_count,
        }
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
}

impl Qwen3VlStageObserver<BrowserBackend> for BrowserQwenStageObserver {
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
        if let (Some(control), Some(oracle)) = (&self.parity, self.oracle(stage)) {
            control.rank3(Self::stage_name(stage), oracle, activation);
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

    async fn synchronize(&mut self) -> Result<(), BooguError> {
        self.synchronize_with_parity("denoiser stage").await
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
    pub denoiser_quantized_load_policy: String,
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
    pub denoiser_quantized_load_policy: String,
    pub qwen_visual_execution_dtype: String,
    pub vae_execution_dtype: String,
    pub denoiser_execution_dtype: String,
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
    pub qwen_float_load_policy: String,
    pub vae_float_load_policy: String,
    pub denoiser_float_load_policy: String,
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
    ResidentDenseF32,
}

/// Asynchronously builds one pinned browser release from a remote sealed artifact directory.
pub struct BrowserBooguFactory {
    variant: BooguVariant,
    residency: BrowserBooguResidencyPolicy,
    pending: Option<BrowserBuildSlot>,
    started: bool,
}

impl BrowserBooguFactory {
    /// Select the immutable release expected below the configured remote base URL.
    pub const fn new(variant: BooguVariant) -> Self {
        Self {
            variant,
            residency: BrowserBooguResidencyPolicy::HighVramResidentDenseF32,
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
        for (name, actual) in [
            (
                "max_storage_buffer_binding_size",
                context.max_storage_buffer_binding_size,
            ),
            ("max_buffer_size", context.max_buffer_size),
        ] {
            if actual < crate::boogu::BOOGU_BROWSER_MAX_APPLIED_BUFFER_BYTES {
                return Err(execution_error(
                    variant,
                    format!(
                        "browser {name} is {actual} bytes; post-load model tensor requires {} bytes",
                        crate::boogu::BOOGU_BROWSER_MAX_APPLIED_BUFFER_BYTES
                    ),
                ));
            }
        }
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
                BrowserNoSurfacePolicy::ResidentDenseF32 => {
                    "no-surface-browser-high-vram-resident-dense-f32"
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
            denoiser_quantized_load_policy: quantized_policy_name(
                engine.policies.denoiser_quantized,
            )
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

    /// Execute one complete, validated 256 by 256 Turbo request without creating a surface.
    ///
    /// The request is resolved by the same adapter path as the production UI. The returned PNG is
    /// a real model output; the report deliberately does not claim upstream fixture parity.
    pub async fn infer_no_surface(
        variant: BooguVariant,
        settings: crate::BooguAdapterSettings,
        request: ImageRequest,
    ) -> Result<BrowserBooguInferenceResult, RuntimeError> {
        Self::infer_no_surface_with_residency(
            variant,
            settings,
            BrowserBooguResidencyPolicy::HighVramResidentDenseF32,
            request,
        )
        .await
    }

    /// Execute a surface-free request with an explicit residency policy.
    pub async fn infer_no_surface_with_residency(
        variant: BooguVariant,
        settings: crate::BooguAdapterSettings,
        residency: BrowserBooguResidencyPolicy,
        request: ImageRequest,
    ) -> Result<BrowserBooguInferenceResult, RuntimeError> {
        let job = crate::boogu::prepare_runtime_job(ImageJobId(1), variant, request, &settings)?;
        if job.resolved.dimensions
            != Dimensions::new(256, 256).expect("fixed diagnostic dimensions are valid")
        {
            return Err(execution_error(
                variant,
                "surface-free full inference is restricted to 256 by 256",
            ));
        }
        let prompt = job.resolved.prompt.clone();
        let dimensions = job.resolved.dimensions;
        let BrowserNoSurfaceEngine {
            mut engine,
            adapter,
            backend,
            limits,
            adapter_shader_f16,
            device_shader_f16,
        } = build_no_surface_engine(
            variant,
            settings,
            match residency {
                BrowserBooguResidencyPolicy::HighVramResidentDenseF32 => {
                    BrowserNoSurfacePolicy::ResidentDenseF32
                }
                BrowserBooguResidencyPolicy::LayerStreamedDiagnostic => {
                    BrowserNoSurfacePolicy::CompatibleF32
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
                denoiser_quantized_load_policy: quantized_policy_name(
                    engine.policies.denoiser_quantized,
                )
                .into(),
                qwen_visual_execution_dtype: engine.dtypes.qwen_visual.name().into(),
                vae_execution_dtype: engine.dtypes.vae.name().into(),
                denoiser_execution_dtype: engine.dtypes.denoiser.name().into(),
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
    /// This is intentionally a separate, surface-free qualification route. It bypasses only the
    /// ordinary browser support advertisement that restricts interactive inference to 256 square;
    /// release identity, profile, remote transport, SHA-256, non-CPU adapter, device limits, exact
    /// request shape, exact noise injection, and every numerical gate remain fail closed.
    pub(crate) async fn parity_no_surface(
        variant: BooguVariant,
        settings: crate::BooguAdapterSettings,
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

async fn build_no_surface_engine(
    variant: BooguVariant,
    settings: crate::BooguAdapterSettings,
    policy: BrowserNoSurfacePolicy,
) -> Result<BrowserNoSurfaceEngine, RuntimeError> {
    let device = burn_wgpu::WgpuDevice::DefaultDevice;
    let setup = burn_wgpu::init_setup_async::<burn_wgpu::graphics::WebGpu>(
        &device,
        burn_wgpu::RuntimeOptions::default(),
    )
    .await;
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
            settings,
            releases: vec![BooguReleaseIdentity::canonical(variant)],
        },
    )?;
    let policies = match policy {
        BrowserNoSurfacePolicy::CompatibleF32 => {
            BrowserExecutionPolicies::layer_streamed_diagnostic(&inputs.settings)
        }
        BrowserNoSurfacePolicy::PreserveQwenF16 => {
            BrowserExecutionPolicies::preserve_qwen_f16(&inputs.settings)
        }
        BrowserNoSurfacePolicy::ResidentDenseF32 => {
            BrowserExecutionPolicies::resident_dense_f32(&inputs.settings)
                .map_err(|error| execution_error(variant, error))?
        }
    };
    let engine = BrowserBooguEngine::build(
        inputs.identity,
        inputs.base_url,
        inputs.settings,
        policies,
        inputs.device,
    )
    .await?;
    Ok(BrowserNoSurfaceEngine {
        engine,
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
    let variant = BooguVariant::Image01EditTurbo1k5;
    let device = burn_wgpu::WgpuDevice::DefaultDevice;
    let setup = burn_wgpu::init_setup_async::<burn_wgpu::graphics::WebGpu>(
        &device,
        burn_wgpu::RuntimeOptions::default(),
    )
    .await;
    let adapter = setup.adapter.get_info();
    let limits = setup.device.limits();
    let adapter_shader_f16 = setup
        .adapter
        .features()
        .contains(wgpu::Features::SHADER_F16);
    let device_shader_f16 = setup.device.features().contains(wgpu::Features::SHADER_F16);
    if adapter.device_type == wgpu::DeviceType::Cpu || matches!(device, burn_wgpu::WgpuDevice::Cpu)
    {
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
    let policies = BrowserExecutionPolicies::exact_1k5_parity(&settings);
    let engine = BrowserBooguEngine::build(identity, base_url, settings, policies, device).await?;
    Ok(BrowserNoSurfaceEngine {
        engine,
        adapter,
        backend: setup.backend,
        limits,
        adapter_shader_f16,
        device_shader_f16,
    })
}

impl BooguRuntimeFactory for BrowserBooguFactory {
    fn start(&mut self, context: BooguFactoryContext) -> Result<(), RuntimeError> {
        if self.started {
            return Err(execution_error(
                self.variant,
                "browser factory was already started",
            ));
        }
        let inputs = Self::validate_context(self.variant, context)?;
        let residency = self.residency;
        report_browser_runtime_preparing(
            "Shared WebGPU device ready; verifying the sealed model manifest",
        );

        let slot = Arc::new(Mutex::new(None));
        let result_slot = slot.clone();
        spawn_local(async move {
            let policies = match residency {
                BrowserBooguResidencyPolicy::HighVramResidentDenseF32 => {
                    BrowserExecutionPolicies::resident_dense_f32(&inputs.settings)
                        .map_err(|error| execution_error(inputs.identity.variant, error))
                }
                BrowserBooguResidencyPolicy::LayerStreamedDiagnostic => Ok(
                    BrowserExecutionPolicies::layer_streamed_diagnostic(&inputs.settings),
                ),
            };
            let result = match policies {
                Ok(policies) => {
                    BrowserBooguEngine::build(
                        inputs.identity,
                        inputs.base_url,
                        inputs.settings,
                        policies,
                        inputs.device,
                    )
                    .await
                }
                Err(error) => Err(error),
            };
            match &result {
                Ok(engine) => report_browser_runtime_ready(engine.identity.variant),
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
}

struct BrowserStageProbe {
    stage: String,
    bytes: u64,
    dtype: String,
    elements: usize,
    finite_elements: usize,
    verified_objects: usize,
}

#[derive(Clone, Copy)]
struct BrowserExecutionPolicies {
    qwen_float: BooguFloatLoadPolicy,
    qwen_quantized: BooguQuantizedLoadPolicy,
    vae_float: BooguFloatLoadPolicy,
    denoiser_float: BooguFloatLoadPolicy,
    denoiser_quantized: BooguQuantizedLoadPolicy,
    residency: BrowserBooguResidencyPolicy,
    retain_qwen_stages: bool,
    retain_vae_stages: bool,
    retain_denoiser_stages: bool,
    eager_preload: bool,
    defer_retained_synchronization: bool,
}

impl BrowserExecutionPolicies {
    fn layer_streamed_diagnostic(settings: &crate::BooguAdapterSettings) -> Self {
        Self {
            // Chrome WebGPU currently rejects Burn/CubeCL F16 kernels. The async verified sources
            // adapt each bounded floating stage before upload, so storage remains unchanged.
            qwen_float: BooguFloatLoadPolicy::AdaptToF32,
            qwen_quantized: settings.qwen_quantized_load_policy(),
            vae_float: BooguFloatLoadPolicy::AdaptToF32,
            denoiser_float: BooguFloatLoadPolicy::AdaptToF32,
            denoiser_quantized: settings.denoiser_quantized_load_policy(),
            residency: BrowserBooguResidencyPolicy::LayerStreamedDiagnostic,
            retain_qwen_stages: false,
            retain_vae_stages: false,
            retain_denoiser_stages: false,
            eager_preload: false,
            defer_retained_synchronization: false,
        }
    }

    fn resident_dense_f32(settings: &crate::BooguAdapterSettings) -> Result<Self, &'static str> {
        if settings.storage_profile != BooguStorageProfile::F16QwenVisionF32 {
            return Err(
                "browser resident-dense production requires profile=production; F16 and Q8 profiles are explicit layer-streamed diagnostics",
            );
        }
        let mut policies = Self::layer_streamed_diagnostic(settings);
        policies.residency = BrowserBooguResidencyPolicy::HighVramResidentDenseF32;
        policies.retain_qwen_stages = true;
        policies.retain_vae_stages = true;
        policies.retain_denoiser_stages = true;
        policies.eager_preload = true;
        policies.defer_retained_synchronization = true;
        Ok(policies)
    }

    fn exact_1k5_parity(settings: &crate::BooguAdapterSettings) -> Self {
        let mut policies = Self::layer_streamed_diagnostic(settings);
        policies.retain_denoiser_stages = true;
        policies
    }

    fn preserve_qwen_f16(settings: &crate::BooguAdapterSettings) -> Self {
        Self {
            // This policy is used only by the explicit SHADER_F16 final-norm probe. VAE and
            // denoiser remain on the production-compatible F32 policy and are not executed.
            qwen_float: BooguFloatLoadPolicy::Preserve,
            qwen_quantized: settings.qwen_quantized_load_policy(),
            vae_float: BooguFloatLoadPolicy::AdaptToF32,
            denoiser_float: BooguFloatLoadPolicy::AdaptToF32,
            denoiser_quantized: settings.denoiser_quantized_load_policy(),
            residency: BrowserBooguResidencyPolicy::LayerStreamedDiagnostic,
            retain_qwen_stages: false,
            retain_vae_stages: false,
            retain_denoiser_stages: false,
            eager_preload: false,
            defer_retained_synchronization: false,
        }
    }

    fn execution_dtypes(self, profile: BooguStorageProfile) -> BooguRuntimeDTypes {
        let mut dtypes = BooguRuntimeDTypes::from_artifact_policies(
            profile,
            self.vae_float,
            self.denoiser_float,
        );
        if self.qwen_float == BooguFloatLoadPolicy::AdaptToF32 {
            dtypes.qwen_visual = DType::F32;
        }
        dtypes
    }
}

struct BrowserArtifactComposition {
    pipeline_manifest: ArtifactManifest,
    qwen_manifest: ArtifactManifest,
    vae_manifest: ArtifactManifest,
    pipeline_base_url: RemoteBaseUrl,
    qwen_base_url: RemoteBaseUrl,
    vae_base_url: RemoteBaseUrl,
    legacy_monolith: bool,
}

impl BrowserArtifactComposition {
    async fn resolve(
        variant: BooguVariant,
        pipeline_manifest: ArtifactManifest,
        pipeline_base_url: RemoteBaseUrl,
        stream_config: ArtifactStreamConfig,
    ) -> Result<Self, RuntimeError> {
        if pipeline_manifest.dependencies.is_empty() {
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
                legacy_monolith: true,
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
            legacy_monolith: false,
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
    ) -> Result<Self, RuntimeError> {
        let variant = identity.variant;
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
        report_browser_manifest_verified(&manifest);
        let composition =
            BrowserArtifactComposition::resolve(variant, manifest, base_url, stream_config).await?;
        let expected_weight_artifacts = if composition.legacy_monolith {
            manifest_weight_artifacts(&composition.pipeline_manifest, false)
        } else {
            [
                &composition.pipeline_manifest,
                &composition.qwen_manifest,
                &composition.vae_manifest,
            ]
            .into_iter()
            .flat_map(|manifest| manifest_weight_artifacts(manifest, true))
            .collect()
        };
        if policies.eager_preload {
            let mut resource_manifest = composition.pipeline_manifest.clone();
            if !composition.legacy_monolith {
                resource_manifest
                    .files
                    .extend(composition.qwen_manifest.files.iter().cloned());
                resource_manifest
                    .files
                    .extend(composition.vae_manifest.files.iter().cloned());
            }
            validate_browser_resident_resource_plan(variant, &resource_manifest)?;
        }
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
                let path = if composition.legacy_monolith {
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
        bootstrap_control.set_observer(Some(Arc::new(|event| {
            let progress = browser_artifact_progress(RunId(0), event);
            dispatch_browser_progress(&progress);
        })));
        let make_reader = |base_url, bundle| {
            if composition.legacy_monolith {
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
            }
        };
        let reader = make_reader(
            composition.pipeline_base_url.clone(),
            composition.pipeline_manifest.bundle.clone(),
        );
        let mut qwen_reader = make_reader(
            composition.qwen_base_url.clone(),
            composition.qwen_manifest.bundle.clone(),
        );
        let mut vae_reader = make_reader(
            composition.vae_base_url.clone(),
            composition.vae_manifest.bundle.clone(),
        );
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
        let verified_runtime_metadata = if composition.legacy_monolith {
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

        let (verified_qwen_source, qwen_plan) = if composition.legacy_monolith {
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
            (BrowserVerifiedQwenSource::Legacy(source), plan)
        } else {
            let contract = Qwen3VlComponentContract::released_base(
                composition.qwen_manifest.clone(),
                qwen_config.clone(),
            )
            .map_err(|error| execution_error(variant, error))?;
            let plan = contract.plan().clone();
            let float_policy = match policies.qwen_float {
                BooguFloatLoadPolicy::Preserve => Qwen3VlArtifactFloatPolicy::Preserve,
                BooguFloatLoadPolicy::AdaptToF32 => Qwen3VlArtifactFloatPolicy::AdaptToF32,
            };
            let source =
                VerifiedAsyncBurnpackQwen3VlStageSource::new(contract, device.clone(), qwen_reader)
                    .with_float_policy(float_policy);
            (BrowserVerifiedQwenSource::Component(source), plan)
        };
        let qwen_source = BrowserAsyncStageSource::new(verified_qwen_source, synchronizer.clone());
        let qwen_source = if policies.retain_qwen_stages {
            RetainingAsyncQwen3VlStageSource::new(qwen_source)
        } else {
            RetainingAsyncQwen3VlStageSource::passthrough(qwen_source)
        };
        let qwen_source = if policies.defer_retained_synchronization {
            qwen_source.with_synchronization_policy(AsyncRetainingSynchronizationPolicy::Deferred)
        } else {
            qwen_source
        };
        let mut qwen = StreamingQwen3Vl::new(qwen_plan, qwen_source);
        if variant == BooguVariant::Image01EditTurbo1k5 {
            qwen.set_query_chunk_size(BROWSER_1K5_QWEN_QUERY_CHUNK_SIZE);
        }
        let verified_vae_source = if composition.legacy_monolith {
            BrowserVerifiedVaeSource::Legacy(
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
            let float_policy = match policies.vae_float {
                BooguFloatLoadPolicy::Preserve => FluxVaeArtifactFloatPolicy::Preserve,
                BooguFloatLoadPolicy::AdaptToF32 => FluxVaeArtifactFloatPolicy::AdaptToF32,
            };
            let source =
                VerifiedAsyncBurnpackFluxVaeStageSource::new(contract, device.clone(), vae_reader)
                    .with_float_policy(float_policy);
            BrowserVerifiedVaeSource::Component(AsyncFluxVaeStageSourceAdapter::new(source))
        };
        let vae_source = BrowserAsyncStageSource::new(verified_vae_source, synchronizer.clone());
        let vae = if policies.retain_vae_stages {
            RetainingAsyncBooguVaeStageSource::new(vae_source)
        } else {
            RetainingAsyncBooguVaeStageSource::passthrough(vae_source)
        };
        let verified_denoiser_source = VerifiedAsyncBurnpackDenoiserStageSource::new(
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
        .with_quantized_load_policy(policies.denoiser_quantized);
        let mut denoiser_source =
            BrowserAsyncStageSource::new(verified_denoiser_source, synchronizer);
        // The 1.5K fixture-qualified path uses q1024 while remaining far below its full joint
        // sequence. Ordinary 256px production keeps the model's q128 default: using q1024 there
        // can cover the entire short sequence and accidentally materialize a dense seq^2 score
        // tensor. GPU residency never relaxes the bounded-attention contract.
        denoiser_source.set_denoiser_query_chunk_size(
            if variant == BooguVariant::Image01EditTurbo1k5 {
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
        let denoiser_source = if policies.defer_retained_synchronization {
            denoiser_source
                .with_synchronization_policy(AsyncRetainingDenoiserSynchronizationPolicy::Deferred)
        } else {
            denoiser_source
        };
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
                        "browser resident-dense preload failed without fallback: {error}; use residency=layer-streamed-diagnostic only for explicit low-memory diagnosis"
                    ),
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

    async fn synchronize_preloaded_qwen_stage(&mut self) -> Result<(), RuntimeError> {
        let variant = self.identity.variant;
        self.qwen
            .source
            .synchronize()
            .await
            .map_err(|error| map_boogu(variant, error))?;
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
        self.denoiser
            .source_mut()
            .synchronize_pending()
            .await
            .map_err(|error| map_boogu(variant, error))
    }

    async fn preload_resident_weights(&mut self) -> Result<(), RuntimeError> {
        let variant = self.identity.variant;
        if self.policies.residency != BrowserBooguResidencyPolicy::HighVramResidentDenseF32
            || !self.qwen.source.retention_enabled()
            || !self.vae.retention_enabled()
            || !self.denoiser.source().retention_enabled()
        {
            return Err(execution_error(
                variant,
                "resident browser preload requires retaining Qwen, VAE, and denoiser sources",
            ));
        }
        report_browser_runtime_preparing(
            "Verifying and materializing dense-F32 Qwen, VAE, and denoiser weights on WebGPU",
        );

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
        self.validate_resident_caches()
    }

    fn validate_resident_caches(&self) -> Result<(), RuntimeError> {
        if self.policies.residency != BrowserBooguResidencyPolicy::HighVramResidentDenseF32 {
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
                let actual_values = actual_data
                    .as_slice::<f32>()
                    .map_err(|error| execution_error(variant, error))?;
                f32_oracle.push(
                    compare_browser_f32_values(
                        &fixture,
                        &ledger,
                        format!("vae.reference_f32_{component}"),
                        format!("vae.reference_f32_{component}"),
                        &shape,
                        &actual_dtype,
                        actual_values,
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
                        actual_values,
                    )
                    .await
                    .map_err(|error| map_boogu(variant, error))?,
                );

                if index == 0 {
                    baseline.insert(component.into(), (shape, actual_values.to_vec()));
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
                    let comparison = compare_float(actual_values, baseline_values)?;
                    let bitwise_exact = actual_values
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
                "exact browser 1.5K parity requires the high-VRAM retained-denoiser source",
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
        let denoiser_boundaries = parity_control.metrics()[denoiser_metric_start..].to_vec();
        let denoiser_retained_stages_before_clear = self.denoiser.source().cached_stage_count();
        if denoiser_retained_stages_before_clear != BROWSER_1K5_DENOISER_RETAINED_STAGE_COUNT {
            return Err(execution_error(
                variant,
                format!(
                    "high-VRAM browser parity retained {denoiser_retained_stages_before_clear}/{} exact denoiser stages",
                    BROWSER_1K5_DENOISER_RETAINED_STAGE_COUNT
                ),
            ));
        }
        self.denoiser.source_mut().clear();
        let denoiser_cache_cleared_before_decode = self.denoiser.source().cached_stage_count() == 0;
        if !denoiser_cache_cleared_before_decode {
            return Err(execution_error(
                variant,
                "high-VRAM browser parity could not release denoiser handles before VAE decode",
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
        let decoded_values = decoded_data
            .as_slice::<f32>()
            .map_err(|error| execution_error(variant, error))?;
        parity_milestone("vae-output-parity-start");
        let decoded_tensor = compare_browser_f32_values(
            &fixture,
            &ledger,
            "full_chain_output.decoded_tensor".into(),
            "vae.decode_output".into(),
            &decoded_shape,
            &decoded_dtype,
            decoded_values,
        )
        .await
        .map_err(|error| map_boogu(variant, error))?;
        parity_milestone("vae-output-parity-complete");
        parity_milestone("vae-output-rgb-conversion-start");
        let HostImage::Pixels(actual_rgb) =
            decoder_output_data_to_host(decoded_data).map_err(|error| map_boogu(variant, error))?
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
            qwen_float_load_policy: float_policy_name(self.policies.qwen_float).into(),
            vae_float_load_policy: float_policy_name(self.policies.vae_float).into(),
            denoiser_float_load_policy: float_policy_name(self.policies.denoiser_float).into(),
            qwen_execution_dtype: self.dtypes.qwen_visual.name().into(),
            vae_execution_dtype: self.dtypes.vae.name().into(),
            denoiser_execution_dtype: self.dtypes.denoiser.name().into(),
            qwen_query_chunk_size: BROWSER_1K5_QWEN_QUERY_CHUNK_SIZE,
            vae_attention_query_chunk_size: BROWSER_1K5_VAE_QUERY_CHUNK_SIZE,
            vae_decode_policy: "exact-two-width-slabs-global-groupnorm".into(),
            vae_decode_max_planned_buffer_bytes:
                crate::boogu::BOOGU_BROWSER_1K5_VAE_STRIPED_TAIL_BUFFER_BYTES,
            denoiser_query_chunk_size: BROWSER_1K5_DENOISER_QUERY_CHUNK_SIZE,
            denoiser_residency: "lazy-resident-first-pass-through-four-dmd-steps".into(),
            denoiser_expected_retained_stages: BROWSER_1K5_DENOISER_RETAINED_STAGE_COUNT,
            denoiser_retained_stages_before_clear,
            denoiser_cache_cleared_before_decode,
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
        self.validate_resident_caches()?;
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
        let mut qwen_observer = BrowserQwenStageObserver::milestones_only();
        let qwen_output = self
            .qwen
            .forward_base_async(&self.qwen_config, prepared.model_input, &mut qwen_observer)
            .await
            .map_err(|error| {
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
        let instruction =
            trim_instruction_features(qwen_output.last_hidden_state, prepared.effective_length)
                .map_err(|error| map_boogu(job.variant, error))?
                .cast(self.dtypes.denoiser);
        check_cancelled(cancellation)?;
        finish_stage(shared, id, run_id, &mut timings, "qwen", started);

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
        let mut latents = normal_tensor::<4>(
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
        let mut noises = renoise.into_iter();
        for (index, &sigma) in schedule.sigmas().iter().enumerate() {
            check_cancelled(cancellation)?;
            require_dtype(
                job.variant,
                "DMD latent",
                latents.dtype(),
                self.dtypes.denoiser,
            )?;
            let timestep = Tensor::<BrowserBackend, 1>::from_data(
                TensorData::new(vec![sigma], [1]),
                &self.device,
            )
            .cast(self.dtypes.denoiser);
            let prediction = self
                .denoiser
                .predict_async(BooguDenoiserInput {
                    latent: latents.clone(),
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
            require_dtype(
                job.variant,
                "denoiser prediction",
                prediction.dtype(),
                self.dtypes.denoiser,
            )?;
            latents = dmd_prediction(latents, prediction, sigma);
            if let Some(&next_sigma) = schedule.sigmas().get(index + 1) {
                let noise = noises
                    .next()
                    .expect("the fixed four-step schedule has three renoise tensors");
                latents = dmd_renoise(latents, noise, next_sigma);
            }
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
        finish_stage(shared, id, run_id, &mut timings, "dmd", started);

        check_cancelled(cancellation)?;
        let started = start_stage(shared, id, run_id, "vae-decode", Some(1));
        let decoder = self
            .vae
            .load_decoder()
            .await
            .map_err(|error| map_boogu(job.variant, error))?;
        let loaded_dtype: DType = decoder.float_dtype().into();
        require_dtype(
            job.variant,
            "loaded VAE decoder",
            loaded_dtype,
            self.dtypes.vae,
        )?;
        let decoded = decoder.decode_scaled(latents.cast(self.dtypes.vae));
        self.vae
            .synchronize()
            .await
            .map_err(|error| map_boogu(job.variant, error))?;
        drop(decoder);
        check_cancelled(cancellation)?;
        finish_stage(shared, id, run_id, &mut timings, "vae-decode", started);

        let started = start_stage(shared, id, run_id, "output", Some(1));
        let image = decoder_output_to_host_async(decoded)
            .await
            .map_err(|error| map_boogu(job.variant, error))?;
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
                backend: format!("burn-webgpu/{}", self.policies.residency.label()),
                artifacts_verified: true,
            },
        };
        output
            .validate()
            .map_err(|error| execution_error(job.variant, error))?;
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
    let final_rgb = BrowserParityRgbGate {
        minimum_psnr_db: 33.5,
        minimum_mean_block_ssim_8x8: 0.99,
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
    if artifact_bundle_id_is_compatible(variant, profile, actual) {
        return Ok(());
    }
    let preferred = crate::boogu::boogu_bundle_id(variant, profile);
    let legacy = crate::boogu::boogu_legacy_bundle_id(variant, profile);
    Err(execution_error(
        variant,
        format!(
            "sealed browser manifest bundle {actual} is incompatible with the selected release/profile; expected {preferred} or legacy {legacy}"
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

impl BooguRuntime for BrowserBooguRuntime {
    fn variants(&self) -> Vec<BooguVariant> {
        vec![self.variant]
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
            let result = engine.infer(&job, &cancellation, &shared).await;
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
    let observer_shared = Arc::clone(shared);
    control.set_observer(Some(Arc::new(move |event| {
        let progress = browser_artifact_progress(run_id, event);
        queue_progress(&observer_shared, id, progress);
    })));
}

fn browser_artifact_progress(run_id: RunId, event: BrowserArtifactEvent) -> ProgressEvent {
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
        },
        BrowserArtifactEvent::Verified(path) => ProgressEvent::ArtifactVerified { run_id, path },
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
        // Kept exhaustive for provenance/file naming, although factory validation rejects this
        // native-only release before browser artifact loading or inference can begin.
        BooguVariant::Image01EditTurbo1k5 => "edit-turbo-1k5",
    }
}

const fn float_policy_name(policy: BooguFloatLoadPolicy) -> &'static str {
    match policy {
        BooguFloatLoadPolicy::Preserve => "preserve",
        BooguFloatLoadPolicy::AdaptToF32 => "adapt-to-f32",
    }
}

const fn quantized_policy_name(policy: BooguQuantizedLoadPolicy) -> &'static str {
    match policy {
        BooguQuantizedLoadPolicy::Preserve => "preserve",
        BooguQuantizedLoadPolicy::DequantizeF16 => "dequantize-f16",
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
    use std::collections::BTreeMap;

    use burn_image::{
        ARTIFACT_MANIFEST_SCHEMA_V1, ARTIFACT_MANIFEST_SCHEMA_V2, ArtifactBundleId,
        ArtifactComponentId, ArtifactDependency, ArtifactFileRole, ArtifactProfileId, ModelId,
        NumericFormat,
    };

    use super::*;

    fn tiny_manifest(bundle: &str, schema_version: u32) -> ArtifactManifest {
        let bytes = b"tiny";
        let mut manifest = ArtifactManifest {
            schema_version,
            bundle: ArtifactBundleId::new(bundle).unwrap(),
            profile: ArtifactProfileId::new("test-profile").unwrap(),
            model: ModelId::new(format!("test/{bundle}")).unwrap(),
            model_revision: "revision".into(),
            numeric_format: NumericFormat::F16,
            components: Vec::new(),
            files: vec![ArtifactFile {
                path: ArtifactPath::new("objects/tiny.bpk").unwrap(),
                size: bytes.len() as u64,
                sha256: Sha256Digest::calculate(bytes),
                role: ArtifactFileRole::Metadata,
                component: None,
                shard: None,
            }],
            dependencies: Vec::new(),
            metadata: BTreeMap::new(),
            content_digest: None,
        };
        manifest.seal().unwrap();
        manifest
    }

    fn dependency(role: &str, manifest: &ArtifactManifest) -> ArtifactDependency {
        ArtifactDependency {
            role: ArtifactComponentId::new(role).unwrap(),
            bundle: manifest.bundle.clone(),
            profile: manifest.profile.clone(),
            model: manifest.model.clone(),
            model_revision: manifest.model_revision.clone(),
            content_digest: manifest.content_digest.unwrap(),
        }
    }

    #[test]
    fn browser_residency_selector_is_fail_closed_and_stably_labeled_correctness() {
        assert_eq!(
            BrowserBooguResidencyPolicy::default(),
            BrowserBooguResidencyPolicy::HighVramResidentDenseF32
        );
        assert_eq!(
            BrowserBooguResidencyPolicy::parse("resident"),
            Some(BrowserBooguResidencyPolicy::HighVramResidentDenseF32)
        );
        assert_eq!(
            BrowserBooguResidencyPolicy::parse("layer-streamed-diagnostic"),
            Some(BrowserBooguResidencyPolicy::LayerStreamedDiagnostic)
        );
        assert_eq!(BrowserBooguResidencyPolicy::parse("streamed"), None);
        assert_eq!(
            BrowserBooguResidencyPolicy::HighVramResidentDenseF32.label(),
            "browser-high-vram-resident-dense-f32"
        );
        assert_eq!(
            BrowserBooguResidencyPolicy::LayerStreamedDiagnostic.label(),
            "browser-layer-streamed-diagnostic"
        );
    }

    #[test]
    fn browser_webgpu_vae_f32_oracle_envelope_is_complete_and_scoped_correctness() {
        let envelope = BROWSER_WEBGPU_VAE_F32_ORACLE_ENVELOPE;
        assert_eq!(
            envelope.artifact_content_digest,
            BROWSER_WEBGPU_VAE_F32_ORACLE_LEGACY_FLAT_CONTENT_DIGEST
        );
        assert_eq!(envelope.weight_storage_dtype, "f16");
        assert_eq!(envelope.weight_load_policy, "adapt-to-f32");
        assert_eq!(envelope.execution_dtype, "f32");
        assert_eq!(envelope.portability, "no-cross-adapter-portability-claim");
        assert_eq!(envelope.moments.maximum_abs, 0.016);
        assert_eq!(envelope.moments.maximum_rmse, 0.000_75);
        assert_eq!(envelope.mean.maximum_abs, 0.013);
        assert_eq!(envelope.logvar.maximum_abs, 0.016);
        assert_eq!(envelope.std.maximum_abs, 0.000_1);
        assert_eq!(envelope.raw_latent.maximum_abs, 0.013);
        assert_eq!(envelope.scaled_latent.maximum_abs, 0.005);
        assert_eq!(envelope.scaled_latent.maximum_rmse, 0.000_2);
        assert_eq!(
            envelope.component_maximum("vae.reference_f32_scaled_latent"),
            Some(envelope.scaled_latent.maximum_abs)
        );
        assert_eq!(
            envelope.component_maximum("vae.reference_f32_unknown"),
            None
        );

        let serialized = serde_json::to_value(envelope).unwrap();
        assert_eq!(
            serialized["calibrated_device"],
            serde_json::Value::String("0x2bb1".into())
        );
        assert_eq!(serialized["moments"]["maximum_abs"], 0.016);
        assert_eq!(serialized["scaled_latent"]["maximum_abs"], 0.005);
    }

    #[test]
    fn canonical_digest_requirement_tracks_exact_origin_correctness() {
        let variant = BooguVariant::Image01Turbo;
        let profile = BooguStorageProfile::F16QwenVisionF32;
        let canonical = burn_image::RemoteBaseUrl::new(format!(
            "{}/boogu-image-0.1-turbo",
            crate::boogu::BOOGU_CDN_ROOT
        ))
        .unwrap();
        let legacy = burn_image::RemoteBaseUrl::new(format!(
            "{}/boogu-image-0.1-turbo-f16-qwen-vision-f32",
            crate::boogu::BOOGU_CDN_ROOT
        ))
        .unwrap();
        let custom = burn_image::RemoteBaseUrl::new(
            "https://models.example/boogu-image-0.1-turbo-f16-qwen-vision-f32",
        )
        .unwrap();
        assert!(browser_source_requires_canonical_digest(
            variant, profile, &canonical
        ));
        assert!(!browser_source_requires_canonical_digest(
            variant, profile, &legacy
        ));
        assert!(!browser_source_requires_canonical_digest(
            variant, profile, &custom
        ));
        assert!(!browser_source_requires_canonical_digest(
            variant,
            BooguStorageProfile::F16,
            &canonical
        ));
    }

    #[test]
    fn browser_composition_requires_exact_component_roles_correctness() {
        let qwen = tiny_manifest("shared-qwen", ARTIFACT_MANIFEST_SCHEMA_V1);
        let vae = tiny_manifest("shared-vae", ARTIFACT_MANIFEST_SCHEMA_V1);
        let mut parent = tiny_manifest("pipeline", ARTIFACT_MANIFEST_SCHEMA_V2);
        parent.dependencies = vec![dependency("qwen", &qwen), dependency("vae", &vae)];
        parent
            .metadata
            .insert("component_dependency_count".into(), "2".into());
        parent.content_digest = None;
        parent.seal().unwrap();
        let variant = BooguVariant::Image01Turbo;
        assert_eq!(
            browser_dependency(&parent, "qwen", variant).unwrap().bundle,
            qwen.bundle
        );
        assert_eq!(
            browser_dependency(&parent, "vae", variant).unwrap().bundle,
            vae.bundle
        );

        parent.dependencies.pop();
        let error = browser_dependency(&parent, "vae", variant)
            .unwrap_err()
            .to_string();
        assert!(error.contains("omits required vae dependency"), "{error}");
    }

    #[test]
    fn browser_weight_ledger_qualifies_component_bundle_paths_correctness() {
        let manifest = tiny_manifest("shared-qwen", ARTIFACT_MANIFEST_SCHEMA_V1);
        let unqualified = manifest_weight_artifacts(&manifest, false);
        let qualified = manifest_weight_artifacts(&manifest, true);
        // Tiny fixture uses metadata; make sure both routes remain empty rather
        // than accidentally counting compact bootstrap files as model weights.
        assert!(unqualified.is_empty());
        assert!(qualified.is_empty());

        let mut weights = manifest;
        weights.files[0].role = ArtifactFileRole::Weights;
        assert!(manifest_weight_artifacts(&weights, false).contains_key("objects/tiny.bpk"));
        assert!(
            manifest_weight_artifacts(&weights, true).contains_key("shared-qwen/objects/tiny.bpk")
        );
    }

    #[test]
    fn custom_browser_source_rejects_arbitrary_bundle_identity_correctness() {
        let variant = BooguVariant::Image01Turbo;
        let profile = BooguStorageProfile::F16QwenVisionF32;

        validate_browser_manifest_bundle_identity(variant, profile, "boogu-image-0.1-turbo")
            .unwrap();
        validate_browser_manifest_bundle_identity(
            variant,
            profile,
            "boogu-image-0.1-turbo-f16-qwen-vision-f32",
        )
        .unwrap();
        let error = validate_browser_manifest_bundle_identity(
            variant,
            profile,
            "boogu-image-0.1-turbo-arbitrary",
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("incompatible with the selected release/profile"),
            "{error}"
        );
        assert!(
            error.contains(
                "expected boogu-image-0.1-turbo or legacy boogu-image-0.1-turbo-f16-qwen-vision-f32"
            ),
            "{error}"
        );
    }
}
