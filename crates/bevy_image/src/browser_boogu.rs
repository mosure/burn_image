//! Concrete browser-local Boogu runtime over verified bounded HTTP Range reads.
//!
//! The runtime retains activations only. Qwen, VAE, and denoiser modules are fetched, verified,
//! executed, synchronized, and dropped at their semantic stage boundaries. It never falls back to
//! CPU and never manufactures placeholder output.

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use burn::{
    nn::RmsNorm,
    tensor::{DType, Tensor, TensorData},
};
use burn_boogu::{
    AsyncBooguDenoiserStageSource, AsyncBooguVaeStageSource, BooguConfig, BooguDenoiserInput,
    BooguDenoiserPrelude, BooguDenoiserTail, BooguError, BooguRuntimeDTypes, BooguTask,
    BooguVariant, DmdSchedule, DoubleStreamBlock, SingleStreamBlock, StreamingBooguDenoiser,
    artifacts::{
        BooguArtifactInventory, BooguFloatLoadPolicy, BooguQuantizedLoadPolicy,
        BooguReleaseIdentity, BooguStorageProfile, VerifiedAsyncBurnpackDenoiserStageSource,
        VerifiedAsyncBurnpackQwenStageSource, VerifiedAsyncBurnpackVaeStageSource,
        canonical_published_bundle, validate_canonical_release_artifact_digest,
    },
    boogu_model_descriptor, boogu_processor_config, decode_input_image, dmd_prediction,
    dmd_renoise, encode_reference, prepare_instruction, prepare_vae_reference,
    trim_instruction_features,
};
use burn_flux_vae::{AutoencoderKl, AutoencoderKlConfig};
use burn_image::{
    ArtifactFile, ArtifactManifest, ArtifactPath, CancellationToken, ColorSpace, Dimensions,
    GeneratedImage, HostImage, ImageEncoding, ImageOutput, ImageRequest, ImageTaskKind,
    ModelProvenance, PixelBuffer, PixelFormat, ProgressEvent, RunId, RuntimeError, Sha256Digest,
    StageTiming, StageTimings,
};
use burn_qwen3_vl::{
    AsyncQwen3VlStageSource, EmbeddingRowChunk, Qwen3VlConfig, Qwen3VlDecoderLayer,
    Qwen3VlImageProcessor, Qwen3VlImageProcessorConfig, Qwen3VlProcessor, Qwen3VlStage,
    Qwen3VlStageObserver, Qwen3VlTokenizer, Qwen3VlVisionBlock, Qwen3VlVisionPatchMerger,
    Qwen3VlVisionPrelude, RowChunkSpec, StreamingQwen3Vl, tokenizer::HfTokenizer,
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
        fetch_browser_bounded_file,
    },
};

// Burn 0.21 documents that the fused WGPU backend may need to be disabled on Wasm. Keep the
// Bevy/Burn bridge on `SharedWgpuBackend` for device initialization and attestation, while model
// execution uses the raw CubeCL backend against the same WGPU runtime/device registry.
type BrowserBackend = burn_wgpu::CubeBackend<burn_wgpu::WgpuRuntime, f32, i32, u32>;
type BrowserVerifiedQwenSource =
    VerifiedAsyncBurnpackQwenStageSource<BrowserBackend, BrowserStageShardReader>;
type BrowserVerifiedVaeSource =
    VerifiedAsyncBurnpackVaeStageSource<BrowserBackend, BrowserStageShardReader>;
type BrowserVerifiedDenoiserSource =
    VerifiedAsyncBurnpackDenoiserStageSource<BrowserBackend, BrowserStageShardReader>;
type BrowserQwenSource = BrowserAsyncStageSource<BrowserVerifiedQwenSource>;
type BrowserVaeSource = BrowserAsyncStageSource<BrowserVerifiedVaeSource>;
type BrowserDenoiserSource = BrowserAsyncStageSource<BrowserVerifiedDenoiserSource>;

const MAX_RUNTIME_EVENTS: usize = 256;
const MAX_EVENTS_PER_POLL: usize = 64;
const BROWSER_PROGRESS_EVENT_NAME: &str = "burn-image-progress";
const BROWSER_RUNTIME_EVENT_NAME: &str = "burn-image-runtime";
type BrowserBuildSlot = Arc<Mutex<Option<Result<BrowserBooguEngine, RuntimeError>>>>;

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
    Ready {
        model: String,
    },
    Failed {
        message: String,
    },
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

/// Replaces only the verified source's blocking backend barrier; all loads still delegate to the
/// same digest-verifying bounded source.
struct BrowserAsyncStageSource<S> {
    inner: S,
    synchronizer: BrowserAsyncSynchronizer,
    pending_stage: Option<String>,
}

impl<S> BrowserAsyncStageSource<S> {
    fn new(inner: S, synchronizer: BrowserAsyncSynchronizer) -> Self {
        Self {
            inner,
            synchronizer,
            pending_stage: None,
        }
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
        let result = self.synchronizer.synchronize(stage).await;
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
struct BrowserQwenStageObserver;

impl Qwen3VlStageObserver<BrowserBackend> for BrowserQwenStageObserver {
    fn rank3(
        &mut self,
        stage: &Qwen3VlStage,
        _activation: Tensor<BrowserBackend, 3>,
    ) -> burn_qwen3_vl::Result<()> {
        if let Qwen3VlStage::TextBlock { index } = stage {
            browser_stage_milestone(&format!("qwen-text-block-{index:02}-forward-submitted"));
        }
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
        self.synchronizer.synchronize("VAE stage").await
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
        self.inner.load_context_refiner(index).await
    }

    async fn load_noise_refiner(
        &mut self,
        index: usize,
    ) -> Result<SingleStreamBlock<BrowserBackend>, BooguError> {
        self.inner.load_noise_refiner(index).await
    }

    async fn load_reference_refiner(
        &mut self,
        index: usize,
    ) -> Result<SingleStreamBlock<BrowserBackend>, BooguError> {
        self.inner.load_reference_refiner(index).await
    }

    async fn load_double_stream(
        &mut self,
        index: usize,
    ) -> Result<DoubleStreamBlock<BrowserBackend>, BooguError> {
        self.inner.load_double_stream(index).await
    }

    async fn load_single_stream(
        &mut self,
        index: usize,
    ) -> Result<SingleStreamBlock<BrowserBackend>, BooguError> {
        self.inner.load_single_stream(index).await
    }

    async fn load_tail(&mut self) -> Result<BooguDenoiserTail<BrowserBackend>, BooguError> {
        self.inner.load_tail().await
    }

    async fn synchronize(&mut self) -> Result<(), BooguError> {
        self.synchronizer.synchronize("denoiser stage").await
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
}

/// Asynchronously builds one pinned browser release from a remote sealed artifact directory.
pub struct BrowserBooguFactory {
    variant: BooguVariant,
    pending: Option<BrowserBuildSlot>,
    started: bool,
}

impl BrowserBooguFactory {
    /// Select the immutable release expected below the configured remote base URL.
    pub const fn new(variant: BooguVariant) -> Self {
        Self {
            variant,
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
        } = build_no_surface_engine(variant, settings, BrowserNoSurfacePolicy::CompatibleF32)
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
                mode: "diagnostic-no-surface-full-request".into(),
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
            BrowserExecutionPolicies::compatible_f32(&inputs.settings)
        }
        BrowserNoSurfacePolicy::PreserveQwenF16 => {
            BrowserExecutionPolicies::preserve_qwen_f16(&inputs.settings)
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

impl BooguRuntimeFactory for BrowserBooguFactory {
    fn start(&mut self, context: BooguFactoryContext) -> Result<(), RuntimeError> {
        if self.started {
            return Err(execution_error(
                self.variant,
                "browser factory was already started",
            ));
        }
        let inputs = Self::validate_context(self.variant, context)?;
        report_browser_runtime_preparing(
            "Shared WebGPU device ready; verifying the sealed model manifest",
        );

        let slot = Arc::new(Mutex::new(None));
        let result_slot = slot.clone();
        spawn_local(async move {
            let policies = BrowserExecutionPolicies::compatible_f32(&inputs.settings);
            let result = BrowserBooguEngine::build(
                inputs.identity,
                inputs.base_url,
                inputs.settings,
                policies,
                inputs.device,
            )
            .await;
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
}

impl BrowserExecutionPolicies {
    fn compatible_f32(settings: &crate::BooguAdapterSettings) -> Self {
        Self {
            // Chrome WebGPU currently rejects Burn/CubeCL F16 kernels. The async verified sources
            // adapt each bounded floating stage before upload, so storage remains unchanged.
            qwen_float: BooguFloatLoadPolicy::AdaptToF32,
            qwen_quantized: settings.qwen_quantized_load_policy(),
            vae_float: BooguFloatLoadPolicy::AdaptToF32,
            denoiser_float: BooguFloatLoadPolicy::AdaptToF32,
            denoiser_quantized: settings.denoiser_quantized_load_policy(),
        }
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

        let mut reader = BrowserStageShardReader::new(base_url, stream_config);
        let bootstrap_control = reader.control();
        bootstrap_control.set_observer(Some(Arc::new(|event| {
            let progress = browser_artifact_progress(RunId(0), event);
            dispatch_browser_progress(&progress);
        })));
        let qwen_config_bytes = read_manifest_file(
            &mut reader,
            &manifest,
            "metadata/source/mllm/config.json",
            variant,
        )
        .await?;
        let vae_config_bytes = read_manifest_file(
            &mut reader,
            &manifest,
            "metadata/source/vae/config.json",
            variant,
        )
        .await?;
        let tokenizer_bytes = read_manifest_file(
            &mut reader,
            &manifest,
            "metadata/source/mllm/tokenizer.json",
            variant,
        )
        .await?;
        let image_processor_bytes = read_manifest_file(
            &mut reader,
            &manifest,
            "metadata/source/mllm/preprocessor_config.json",
            variant,
        )
        .await?;

        let qwen_config = Qwen3VlConfig::from_json(utf8(&qwen_config_bytes, variant)?)
            .map_err(|error| execution_error(variant, error))?;
        let vae_config =
            AutoencoderKlConfig::from_diffusers_json(utf8(&vae_config_bytes, variant)?)
                .map_err(|error| execution_error(variant, error))?;
        let denoiser_config = BooguConfig::default();
        let inventory = BooguArtifactInventory::new(&qwen_config, &denoiser_config, &vae_config)
            .map_err(|error| execution_error(variant, error))?;
        let profile = settings.storage_profile;
        let dtypes = policies.execution_dtypes(profile);
        let synchronizer = BrowserAsyncSynchronizer::new(&device);

        let verified_qwen_source = VerifiedAsyncBurnpackQwenStageSource::new_auto(
            &identity,
            manifest.clone(),
            inventory.clone(),
            qwen_config.clone(),
            profile,
            device.clone(),
            reader.clone(),
        )
        .await
        .map_err(|error| execution_error(variant, error))?
        .with_float_load_policy(policies.qwen_float)
        .with_quantized_load_policy(policies.qwen_quantized);
        let qwen_plan = verified_qwen_source.plan().clone();
        let qwen_source = BrowserAsyncStageSource::new(verified_qwen_source, synchronizer.clone());
        let qwen = StreamingQwen3Vl::new(qwen_plan, qwen_source);
        let verified_vae_source = VerifiedAsyncBurnpackVaeStageSource::new(
            &identity,
            manifest.clone(),
            inventory.clone(),
            vae_config,
            profile,
            policies.vae_float,
            device.clone(),
            reader.clone(),
        )
        .await
        .map_err(|error| execution_error(variant, error))?;
        let vae = BrowserAsyncStageSource::new(verified_vae_source, synchronizer.clone());
        let verified_denoiser_source = VerifiedAsyncBurnpackDenoiserStageSource::new(
            &identity,
            manifest,
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
        let denoiser_source = BrowserAsyncStageSource::new(verified_denoiser_source, synchronizer);
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
        artifact_control.set_observer(None);
        artifact_control.clear_events();

        Ok(Self {
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
        })
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

    async fn infer(
        &mut self,
        job: &BooguRuntimeJob,
        cancellation: &CancellationToken,
        shared: &Arc<Mutex<BrowserRuntimeShared>>,
    ) -> Result<ImageOutput, RuntimeError> {
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
        let mut qwen_observer = BrowserQwenStageObserver;
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
                backend: "burn-webgpu/browser-async-stage-streamed".into(),
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

fn browser_source_requires_canonical_digest(
    variant: BooguVariant,
    profile: BooguStorageProfile,
    base_url: &burn_image::RemoteBaseUrl,
) -> bool {
    canonical_published_bundle(variant, profile).is_some_and(|published| {
        base_url.as_str() == format!("{}/{}", crate::boogu::BOOGU_CDN_ROOT, published.bundle_id)
    })
}

#[cfg(test)]
mod browser_source_tests {
    use super::*;

    #[test]
    fn canonical_digest_requirement_tracks_exact_origin_correctness() {
        let variant = BooguVariant::Image01Turbo;
        let profile = BooguStorageProfile::F16QwenVisionF32;
        let canonical = burn_image::RemoteBaseUrl::new(format!(
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
            variant, profile, &custom
        ));
        assert!(!browser_source_requires_canonical_digest(
            variant,
            BooguStorageProfile::F16,
            &canonical
        ));
    }
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
    let [batch, channels, height, width] = output.dims();
    if batch != 1 || channels != 3 {
        return Err(BooguError::InvalidShape(format!(
            "decoder output must be [1,3,H,W], got [{batch},{channels},{height},{width}]"
        )));
    }
    let values = output
        .into_data_async()
        .await
        .map_err(|error| BooguError::InvalidRequest(format!("WebGPU readback failed: {error}")))?
        .convert_dtype(DType::F32)
        .to_vec::<f32>()
        .map_err(|error| BooguError::InvalidShape(error.to_string()))?;
    let plane = height * width;
    let mut bytes = vec![0_u8; plane * 3];
    for pixel in 0..plane {
        for channel in 0..3 {
            let normalized = (values[channel * plane + pixel] / 2.0 + 0.5).clamp(0.0, 1.0);
            bytes[pixel * 3 + channel] = (normalized * 255.0).round() as u8;
        }
    }
    let dimensions = Dimensions::new(width as u32, height as u32)
        .map_err(|error| BooguError::InvalidShape(error.to_string()))?;
    let pixels = PixelBuffer::new(dimensions, PixelFormat::Rgb8, ColorSpace::Srgb, bytes)
        .map_err(|error| BooguError::InvalidShape(error.to_string()))?;
    Ok(HostImage::Pixels(pixels))
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
