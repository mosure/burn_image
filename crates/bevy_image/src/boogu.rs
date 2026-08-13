//! Feature-gated adapter between the Bevy frontend and `burn_boogu`.
//!
//! `burn_boogu` exposes verified resident, synchronous staged, and asynchronous staged Burnpack
//! loaders. [`crate::NativeBooguFactory`] supplies the local-directory implementation and
//! `BrowserBooguFactory` supplies bounded HTTP Range/WebGPU execution when `boogu-web` is enabled.
//! Model identities and request policy stay authoritative in `burn_boogu`; this adapter owns
//! shared WGPU startup, request dispatch, cancellation, and output provenance checks.

use std::{
    collections::{BTreeMap, HashSet},
    sync::Mutex,
};

use bevy::prelude::*;
use burn_boogu::{
    BooguTask, BooguVariant, ResolvedBooguRequest,
    artifacts::{
        BooguFloatLoadPolicy, BooguQuantizedLoadPolicy, BooguReleaseIdentity, BooguStorageProfile,
        canonical_published_bundle, legacy_artifact_bundle_id, preferred_artifact_bundle_id,
    },
    boogu_model_descriptor,
    conditioning::InstructionPolicy,
    resolve_request,
};
use burn_image::{
    ArtifactCachePolicy, ArtifactProfileId, ArtifactSource, CancellationToken, Dimensions,
    ImageRequest, IntegrityPolicy, ModelDescriptor, ModelId, NumericFormat, RuntimeConfig,
    RuntimeError,
};
#[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))]
use burn_image::{GenerateRequest, GenerationOptions, Prompt};
use serde::{Deserialize, Serialize};

use crate::{
    BackendState, BackendStatus, CompleteImageJob, FailImageJob, FrontendError, ImageFrontendSet,
    ImageJobCancellationRequested, ImageJobDispatched, ImageJobId, ImageRunnerCapabilities,
    ImageRunnerEvent, ImageRunnerState, ImageRunnerStatus, ReportImageProgress, WgpuExecutionKind,
};

/// Largest current browser tensor after verified F16 Qwen row storage is adapted to F32.
pub const BOOGU_BROWSER_MAX_APPLIED_BUFFER_BYTES: u64 = 414_892_032;
/// F32 `[1, 256, 1536, 1536]` feature buffer the ordinary untiled VAE would require.
pub const BOOGU_BROWSER_1K5_VAE_FINAL_FEATURE_BUFFER_BYTES: u64 = 2_415_919_104;
/// Conservative largest buffer in the exact two-slab VAE tail.
///
/// This covers one F32 `[1, 256, 1538, 772]` explicitly padded convolution input. The actual
/// 1536-square core slab is `[1, 256, 1536, 768]`; the extra rows/columns cover its 3x3 halo and
/// Burn's current explicit padding materialization.
pub const BOOGU_BROWSER_1K5_VAE_STRIPED_TAIL_BUFFER_BYTES: u64 = 1_215_832_064;
/// F32 denoiser feed-forward buffer required by the exact 1.5K browser parity fixture.
pub const BOOGU_BROWSER_1K5_DENOISER_FFN_BUFFER_BYTES: u64 = 522_042_368;
/// Largest single runtime buffer required by the exact 1.5K browser parity fixture.
pub const BOOGU_BROWSER_1K5_MIN_RUNTIME_BUFFER_BYTES: u64 =
    BOOGU_BROWSER_1K5_VAE_STRIPED_TAIL_BUFFER_BYTES;
/// Device-buffer limit requested by the ordinary 256-square Boogu browser runtime.
///
/// Exact 1.5K parity uses a separate limit contract so an unsupported qualification route cannot
/// raise the device request for the released interactive browser surface.
pub const BOOGU_BROWSER_REQUESTED_BUFFER_LIMIT_BYTES: u64 = 512 * 1024 * 1024;
/// Minimum applied device-buffer limit required by the exact 1.5K parity route.
///
/// The exact two-slab decoder tail keeps this below Chrome's observed 2,147,483,644-byte WebGPU
/// ceiling without changing the global middle attention or GroupNorm semantics.
pub const BOOGU_BROWSER_1K5_MIN_REQUIRED_BUFFER_LIMIT_BYTES: u64 =
    BOOGU_BROWSER_1K5_MIN_RUNTIME_BUFFER_BYTES;
/// Released browser execution is intentionally limited to the validated 256-square request.
pub const BOOGU_BROWSER_OUTPUT_EDGE: u32 = 256;
/// Pixel ceiling corresponding to [`BOOGU_BROWSER_OUTPUT_EDGE`] squared.
pub const BOOGU_BROWSER_MAX_OUTPUT_PIXELS: u64 =
    BOOGU_BROWSER_OUTPUT_EDGE as u64 * BOOGU_BROWSER_OUTPUT_EDGE as u64;
/// Exact output edge replayed by the dedicated 1.5K browser parity route.
pub const BOOGU_BROWSER_1K5_PARITY_OUTPUT_EDGE: u32 = 1536;
/// Exact output pixel count replayed by the dedicated 1.5K browser parity route.
pub const BOOGU_BROWSER_1K5_PARITY_OUTPUT_PIXELS: u64 =
    BOOGU_BROWSER_1K5_PARITY_OUTPUT_EDGE as u64 * BOOGU_BROWSER_1K5_PARITY_OUTPUT_EDGE as u64;
/// Canonical public origin for immutable Boogu artifact bundles.
pub const BOOGU_CDN_ROOT: &str = "https://aberration.technology/model";

/// Artifact and cache policy supplied to a concrete native or browser runtime.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BooguAdapterSettings {
    pub artifact_source: ArtifactSource,
    pub storage_profile: BooguStorageProfile,
    pub integrity: IntegrityPolicy,
    pub cache: ArtifactCachePolicy,
}

impl BooguAdapterSettings {
    /// Construct the parity-oriented production policy for a caller-selected source.
    pub fn verified_default(artifact_source: ArtifactSource) -> Self {
        Self {
            artifact_source,
            storage_profile: BooguStorageProfile::F16QwenVisionF32,
            integrity: IntegrityPolicy::RequireSha256,
            cache: ArtifactCachePolicy::UseCached,
        }
    }

    /// Construct the diagnostic all-F16 policy for a caller-selected source.
    pub fn verified_f16(artifact_source: ArtifactSource) -> Self {
        Self {
            artifact_source,
            storage_profile: BooguStorageProfile::F16,
            integrity: IntegrityPolicy::RequireSha256,
            cache: ArtifactCachePolicy::UseCached,
        }
    }

    /// Canonical model-neutral runtime configuration for one release.
    pub fn runtime_config(&self, variant: BooguVariant) -> RuntimeConfig {
        RuntimeConfig {
            model: boogu_model_id(variant),
            artifact_profile: artifact_profile_id(self.storage_profile),
            artifact_source: self.artifact_source.clone(),
            integrity: self.integrity,
            cache: self.cache,
        }
    }

    pub fn numeric_format(&self) -> NumericFormat {
        numeric_format(self.storage_profile)
    }

    /// Baseline VAE execution follows Diffusers `force_upcast=true`.
    ///
    /// The separately parity-qualified native high-VRAM mixed-F16 factory overrides this with its
    /// typed preserved-F16 policy. Browser, layer-streamed, Q8, and all-F16 routes retain this
    /// baseline.
    pub const fn vae_float_load_policy(&self) -> BooguFloatLoadPolicy {
        BooguFloatLoadPolicy::AdaptToF32
    }

    /// Q8 denoiser weights require F32 activations on Burn 0.21 WGPU; F16 profiles preserve F16.
    pub const fn denoiser_float_load_policy(&self) -> BooguFloatLoadPolicy {
        match self.storage_profile {
            BooguStorageProfile::F16 | BooguStorageProfile::F16QwenVisionF32 => {
                BooguFloatLoadPolicy::Preserve
            }
            BooguStorageProfile::Q8sBlock32F32
            | BooguStorageProfile::Q8sBlock32F32QwenVisionF32 => BooguFloatLoadPolicy::AdaptToF32,
        }
    }

    /// Qwen's Col-layout matrices must be dequantized stage-locally before transpose on Burn 0.21.
    pub const fn qwen_quantized_load_policy(&self) -> BooguQuantizedLoadPolicy {
        match self.storage_profile {
            BooguStorageProfile::F16 | BooguStorageProfile::F16QwenVisionF32 => {
                BooguQuantizedLoadPolicy::Preserve
            }
            BooguStorageProfile::Q8sBlock32F32
            | BooguStorageProfile::Q8sBlock32F32QwenVisionF32 => {
                BooguQuantizedLoadPolicy::DequantizeF16
            }
        }
    }

    /// Boogu denoiser matrices use the accurate row-layout Q8 kernel and remain quantized.
    pub const fn denoiser_quantized_load_policy(&self) -> BooguQuantizedLoadPolicy {
        BooguQuantizedLoadPolicy::Preserve
    }

    /// Concrete directory/range sources currently rely on their platform's natural cache and do
    /// not implement refresh or bypass semantics.
    #[cfg(any(test, feature = "boogu-native", feature = "boogu-web"))]
    pub(crate) fn validate_concrete_cache_policy(&self) -> Result<(), &'static str> {
        match self.cache {
            ArtifactCachePolicy::UseCached => Ok(()),
            ArtifactCachePolicy::Refresh | ArtifactCachePolicy::Bypass => {
                Err("concrete Boogu factories implement only ArtifactCachePolicy::UseCached")
            }
        }
    }
}

/// Inputs available when the shared Bevy/Burn WGPU device is ready.
///
/// Factory implementations should retain the device and start bounded native
/// worker or Wasm-local artifact/runtime initialization. The canonical
/// release identities are supplied so implementations never discover an
/// unpinned revision at runtime.
#[derive(Clone, Debug)]
pub struct BooguFactoryContext {
    pub device: burn_wgpu::WgpuDevice,
    pub execution: WgpuExecutionKind,
    /// Limits actually applied to the shared WGPU device, not merely advertised by its adapter.
    pub max_storage_buffer_binding_size: u64,
    pub max_buffer_size: u64,
    pub settings: BooguAdapterSettings,
    pub releases: Vec<BooguReleaseIdentity>,
}

/// Fully resolved Boogu work passed to an injected runtime.
#[derive(Clone, Debug)]
pub struct BooguRuntimeJob {
    pub id: ImageJobId,
    pub variant: BooguVariant,
    pub task: BooguTask,
    pub release: BooguReleaseIdentity,
    pub resolved: ResolvedBooguRequest,
    pub instruction_policy: InstructionPolicy,
    pub runtime_config: RuntimeConfig,
    pub request: ImageRequest,
}

/// Native worker or browser-local WebGPU implementation for the prepared
/// `burn_boogu` pipeline.
///
/// Browser implementations use the asynchronous Qwen/VAE/denoiser source traits with one bounded
/// verified object resident at a time. `submit` returns the exact cancellation token checked
/// between range, semantic-stage, and DMD boundaries; `poll` must be non-blocking and bounded.
pub trait BooguRuntime: Send + Sync + 'static {
    /// Releases this initialized runtime can execute.
    fn variants(&self) -> Vec<BooguVariant>;

    fn submit(&mut self, job: BooguRuntimeJob) -> Result<CancellationToken, RuntimeError>;

    fn cancel(&mut self, id: ImageJobId) -> Result<(), RuntimeError>;

    fn poll(&mut self, emit: &mut dyn FnMut(ImageRunnerEvent));
}

/// Device-aware asynchronous construction seam for real Boogu runtimes.
///
/// `start` and `poll` run on Bevy's main thread and therefore must remain
/// bounded. Native factories normally spawn a worker from `start`; Wasm
/// factories normally spawn local futures and advance state through `poll`.
/// Returning `Ok(None)` is the only pending state. Returning a runtime makes
/// the exact supported descriptors visible and enables request dispatch.
pub trait BooguRuntimeFactory: Send + Sync + 'static {
    fn start(&mut self, context: BooguFactoryContext) -> Result<(), RuntimeError>;

    fn poll(&mut self) -> Result<Option<Box<dyn BooguRuntime>>, RuntimeError>;
}

/// Observable state of Boogu-specific runtime construction.
#[derive(Resource, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum BooguAdapterStatus {
    #[default]
    WaitingForSharedGpu,
    BuildingRuntime,
    Ready {
        variants: Vec<BooguVariant>,
    },
    Failed {
        error: FrontendError,
    },
}

#[derive(Clone)]
struct ActiveBooguJob {
    token: CancellationToken,
    model: ModelId,
    revision: String,
    numeric_format: NumericFormat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FactoryPhase {
    WaitingForSharedGpu,
    Building,
    Ready,
    Failed,
}

#[derive(Resource)]
struct BooguAdapterHost {
    settings: BooguAdapterSettings,
    factory: Option<Box<dyn BooguRuntimeFactory>>,
    runtime: Option<Box<dyn BooguRuntime>>,
    phase: FactoryPhase,
    active: BTreeMap<ImageJobId, ActiveBooguJob>,
}

/// Installs exact Boogu model routing around an injected real runtime factory.
///
/// Add [`crate::BurnImageFrontendPlugin`] first so the job messages and shared
/// WGPU bridge exist. This plugin never installs a mock runtime and never
/// fabricates an output when construction fails.
pub struct BooguAdapterPlugin<F: BooguRuntimeFactory> {
    settings: BooguAdapterSettings,
    factory: Mutex<Option<F>>,
}

impl<F: BooguRuntimeFactory> BooguAdapterPlugin<F> {
    pub fn new(settings: BooguAdapterSettings, factory: F) -> Self {
        Self {
            settings,
            factory: Mutex::new(Some(factory)),
        }
    }
}

impl<F: BooguRuntimeFactory> Plugin for BooguAdapterPlugin<F> {
    fn build(&self, app: &mut App) {
        assert!(
            app.is_plugin_added::<crate::ImageJobPlugin>(),
            "BooguAdapterPlugin requires BurnImageFrontendPlugin/ImageJobPlugin first"
        );
        let factory = self
            .factory
            .lock()
            .expect("Boogu adapter factory mutex poisoned")
            .take()
            .expect("a BooguAdapterPlugin can only be installed once");

        app.insert_resource(ImageRunnerStatus::initializing(
            "Boogu runtime is waiting for the shared WGPU device",
        ))
        .init_resource::<BooguAdapterStatus>()
        .insert_resource(BooguAdapterHost {
            settings: self.settings.clone(),
            factory: Some(Box::new(factory)),
            runtime: None,
            phase: FactoryPhase::WaitingForSharedGpu,
            active: BTreeMap::new(),
        })
        .add_systems(PreUpdate, initialize_boogu_runtime)
        .add_systems(
            Update,
            (
                stop_on_backend_loss,
                submit_boogu_jobs,
                cancel_boogu_jobs,
                poll_boogu_runtime,
            )
                .chain()
                .in_set(ImageFrontendSet::Dispatch),
        );
    }
}

fn initialize_boogu_runtime(
    backend: Res<BackendStatus>,
    burn_device: Option<Res<bevy_burn::BurnDevice>>,
    mut host: ResMut<BooguAdapterHost>,
    mut adapter_status: ResMut<BooguAdapterStatus>,
    mut runner_status: ResMut<ImageRunnerStatus>,
) {
    if matches!(host.phase, FactoryPhase::Ready | FactoryPhase::Failed) {
        return;
    }
    if let BackendState::Failed { reason } = &backend.state {
        fail_adapter(
            &mut host,
            &mut adapter_status,
            &mut runner_status,
            FrontendError::backend(reason.to_string()),
        );
        return;
    }
    if !backend.is_ready() {
        return;
    }

    if host.phase == FactoryPhase::WaitingForSharedGpu {
        let Some(device) = burn_device
            .as_deref()
            .and_then(bevy_burn::BurnDevice::device)
        else {
            fail_adapter(
                &mut host,
                &mut adapter_status,
                &mut runner_status,
                FrontendError::backend("the attested shared Burn WGPU device is missing"),
            );
            return;
        };
        if matches!(device, burn_wgpu::WgpuDevice::Cpu) {
            fail_adapter(
                &mut host,
                &mut adapter_status,
                &mut runner_status,
                FrontendError::backend("Boogu refuses a CPU Burn device"),
            );
            return;
        }
        let device_info = match &backend.state {
            BackendState::Ready { device } => device,
            _ => return,
        };
        let context = BooguFactoryContext {
            device: device.clone(),
            execution: current_execution_kind(),
            max_storage_buffer_binding_size: device_info.max_storage_buffer_binding_size,
            max_buffer_size: device_info.max_buffer_size,
            settings: host.settings.clone(),
            releases: canonical_releases(),
        };
        let start_result = host
            .factory
            .as_mut()
            .expect("waiting adapter retains its factory")
            .start(context);
        if let Err(error) = start_result {
            fail_adapter(
                &mut host,
                &mut adapter_status,
                &mut runner_status,
                FrontendError::from(error),
            );
            return;
        }
        host.phase = FactoryPhase::Building;
        *adapter_status = BooguAdapterStatus::BuildingRuntime;
        *runner_status = ImageRunnerStatus::initializing(
            "Boogu runtime is loading verified artifacts on the shared WGPU device",
        );
    }

    let poll_result = host
        .factory
        .as_mut()
        .expect("building adapter retains its factory")
        .poll();
    match poll_result {
        Ok(None) => {}
        Ok(Some(runtime)) => {
            let variants = match validate_runtime_variants(runtime.variants()) {
                Ok(variants) => variants,
                Err(error) => {
                    fail_adapter(&mut host, &mut adapter_status, &mut runner_status, error);
                    return;
                }
            };
            let capabilities = ImageRunnerCapabilities {
                execution: current_execution_kind(),
                models: variants
                    .iter()
                    .copied()
                    .map(|variant| boogu_descriptor(variant, host.settings.storage_profile))
                    .collect(),
                streams_progress: true,
                cooperative_cancellation: true,
                returns_host_images: true,
            };
            let ready = match ImageRunnerStatus::ready(capabilities) {
                Ok(ready) => ready,
                Err(error) => {
                    fail_adapter(&mut host, &mut adapter_status, &mut runner_status, error);
                    return;
                }
            };
            host.runtime = Some(runtime);
            host.factory = None;
            host.phase = FactoryPhase::Ready;
            *adapter_status = BooguAdapterStatus::Ready { variants };
            *runner_status = ready;
        }
        Err(error) => fail_adapter(
            &mut host,
            &mut adapter_status,
            &mut runner_status,
            FrontendError::from(error),
        ),
    }
}

fn fail_adapter(
    host: &mut BooguAdapterHost,
    adapter_status: &mut BooguAdapterStatus,
    runner_status: &mut ImageRunnerStatus,
    error: FrontendError,
) {
    #[cfg(all(feature = "boogu-web", target_arch = "wasm32"))]
    crate::browser_boogu::report_browser_runtime_failure(error.to_string());
    host.runtime = None;
    host.factory = None;
    host.phase = FactoryPhase::Failed;
    *adapter_status = BooguAdapterStatus::Failed {
        error: error.clone(),
    };
    runner_status.state = ImageRunnerState::Failed { error };
}

fn stop_on_backend_loss(
    backend: Res<BackendStatus>,
    mut host: ResMut<BooguAdapterHost>,
    mut adapter_status: ResMut<BooguAdapterStatus>,
    mut runner_status: ResMut<ImageRunnerStatus>,
    mut failed: MessageWriter<FailImageJob>,
) {
    if backend.is_ready() || host.runtime.is_none() {
        return;
    }
    let message = backend
        .unavailable_message()
        .unwrap_or_else(|| "shared WGPU backend was lost".into());
    let error = FrontendError::backend(message);
    let active = std::mem::take(&mut host.active);
    if let Some(runtime) = host.runtime.as_mut() {
        for (id, job) in &active {
            job.token.cancel();
            let _ = runtime.cancel(*id);
            failed.write(FailImageJob {
                id: *id,
                error: error.clone(),
            });
        }
    }
    fail_adapter(&mut host, &mut adapter_status, &mut runner_status, error);
}

fn submit_boogu_jobs(
    mut host: ResMut<BooguAdapterHost>,
    mut dispatched: MessageReader<ImageJobDispatched>,
    mut failed: MessageWriter<FailImageJob>,
) {
    for dispatch in dispatched.read() {
        let Some(variant) = variant_for_model(&dispatch.model) else {
            continue;
        };
        let job = match prepare_runtime_job(
            dispatch.id,
            variant,
            dispatch.request.clone(),
            &host.settings,
        ) {
            Ok(job) => job,
            Err(error) => {
                failed.write(FailImageJob {
                    id: dispatch.id,
                    error: FrontendError::from(error),
                });
                continue;
            }
        };
        let revision = job.release.model_revision.clone();
        let model = job.runtime_config.model.clone();
        let numeric_format = host.settings.numeric_format();
        let Some(runtime) = host.runtime.as_mut() else {
            failed.write(FailImageJob {
                id: dispatch.id,
                error: FrontendError::model_runtime(
                    "Boogu request reached dispatch before a real runtime was ready",
                ),
            });
            continue;
        };
        match runtime.submit(job) {
            Ok(token) => {
                host.active.insert(
                    dispatch.id,
                    ActiveBooguJob {
                        token,
                        model,
                        revision,
                        numeric_format,
                    },
                );
            }
            Err(error) => {
                failed.write(FailImageJob {
                    id: dispatch.id,
                    error: FrontendError::from(error),
                });
            }
        }
    }
}

fn cancel_boogu_jobs(
    mut host: ResMut<BooguAdapterHost>,
    mut cancellations: MessageReader<ImageJobCancellationRequested>,
) {
    for cancellation in cancellations.read() {
        if let Some(active) = host.active.remove(&cancellation.id) {
            active.token.cancel();
        }
        if let Some(runtime) = host.runtime.as_mut() {
            let _ = runtime.cancel(cancellation.id);
        }
    }
}

fn poll_boogu_runtime(
    mut host: ResMut<BooguAdapterHost>,
    mut progress: MessageWriter<ReportImageProgress>,
    mut completed: MessageWriter<CompleteImageJob>,
    mut failed: MessageWriter<FailImageJob>,
) {
    let Some(runtime) = host.runtime.as_mut() else {
        return;
    };
    let mut events = Vec::new();
    runtime.poll(&mut |event| events.push(event));

    for event in events {
        match event {
            ImageRunnerEvent::Progress { id, event } if host.active.contains_key(&id) => {
                progress.write(ReportImageProgress { id, event });
            }
            ImageRunnerEvent::Completed { id, output } => {
                let Some(active) = host.active.remove(&id) else {
                    continue;
                };
                if let Err(error) = validate_boogu_output(&output, &active, host.settings.integrity)
                {
                    failed.write(FailImageJob {
                        id,
                        error: FrontendError::from(error),
                    });
                } else {
                    completed.write(CompleteImageJob { id, output });
                }
            }
            ImageRunnerEvent::Failed { id, error } if host.active.remove(&id).is_some() => {
                failed.write(FailImageJob {
                    id,
                    error: FrontendError::from(error),
                });
            }
            ImageRunnerEvent::Cancelled { id } => {
                host.active.remove(&id);
            }
            ImageRunnerEvent::Progress { .. } | ImageRunnerEvent::Failed { .. } => {}
        }
    }
}

pub(crate) fn prepare_runtime_job(
    id: ImageJobId,
    variant: BooguVariant,
    request: ImageRequest,
    settings: &BooguAdapterSettings,
) -> Result<BooguRuntimeJob, RuntimeError> {
    prepare_runtime_job_for_execution(id, variant, request, settings, current_execution_kind())
}

fn prepare_runtime_job_for_execution(
    id: ImageJobId,
    variant: BooguVariant,
    mut request: ImageRequest,
    settings: &BooguAdapterSettings,
    execution: WgpuExecutionKind,
) -> Result<BooguRuntimeJob, RuntimeError> {
    let model = boogu_model_id(variant);
    validate_execution_variant(variant, execution)?;
    validate_variant_profile(variant, settings.storage_profile)?;
    apply_execution_defaults(&mut request, execution);
    boogu_descriptor_for_execution(variant, settings.storage_profile, execution)
        .capabilities
        .validate_request(&model, &request)?;
    let resolved = resolve_request(variant, &request, 0)
        .map_err(|error| model_execution(&model, error.to_string()))?;
    let task = resolved.task;
    let instruction_policy =
        InstructionPolicy::upstream(task, usize::from(resolved.source.is_some()))
            .map_err(|error| model_execution(&model, error.to_string()))?;
    let release = BooguReleaseIdentity::canonical(variant);
    release
        .validate()
        .map_err(|error| model_execution(&model, error.to_string()))?;
    Ok(BooguRuntimeJob {
        id,
        variant,
        task,
        release,
        resolved,
        instruction_policy,
        runtime_config: settings.runtime_config(variant),
        request,
    })
}

fn apply_execution_defaults(request: &mut ImageRequest, execution: WgpuExecutionKind) {
    if execution != WgpuExecutionKind::BrowserWebGpu {
        return;
    }
    let options = match request {
        ImageRequest::Generate(request) => &mut request.options,
        ImageRequest::Edit(request) => &mut request.options,
    };
    options.dimensions.get_or_insert_with(|| {
        Dimensions::new(BOOGU_BROWSER_OUTPUT_EDGE, BOOGU_BROWSER_OUTPUT_EDGE)
            .expect("fixed browser dimensions are valid")
    });
}

/// Build the deliberately narrow request accepted by the surface-free full-inference diagnostic.
///
/// Keeping this parser beside ordinary request resolution makes the diagnostic reuse the same
/// descriptor, dimensions, task, and release validation as the production UI path.
#[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))]
pub(crate) fn prepare_headless_generate_request(
    variant: BooguVariant,
    prompt: Option<String>,
    seed: Option<String>,
    width: Option<String>,
    height: Option<String>,
) -> Result<ImageRequest, RuntimeError> {
    let model = boogu_model_id(variant);
    if variant != BooguVariant::Image01Turbo {
        return Err(model_execution(
            &model,
            "headless=infer currently supports only the Turbo generate release",
        ));
    }
    let prompt = prompt.ok_or_else(|| {
        model_execution(
            &model,
            "headless=infer requires a non-empty prompt query parameter",
        )
    })?;
    let prompt = Prompt::new(prompt).map_err(|error| model_execution(&model, error.to_string()))?;
    let seed = parse_required_query::<u64>(&model, "seed", seed)?;
    let width = parse_required_query::<u32>(&model, "width", width)?;
    let height = parse_required_query::<u32>(&model, "height", height)?;
    let dimensions = Dimensions::new(width, height)
        .map_err(|error| model_execution(&model, error.to_string()))?;
    if dimensions != Dimensions::new(256, 256).expect("fixed diagnostic dimensions are valid") {
        return Err(model_execution(
            &model,
            "headless=infer is intentionally restricted to width=256&height=256",
        ));
    }
    Ok(ImageRequest::Generate(GenerateRequest {
        prompt,
        negative_prompt: None,
        options: GenerationOptions {
            dimensions: Some(dimensions),
            steps: Some(4),
            guidance_scale: Some(1.0),
            seed: Some(seed),
            batch_size: 1,
        },
    }))
}

#[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))]
fn parse_required_query<T>(
    model: &ModelId,
    name: &'static str,
    value: Option<String>,
) -> Result<T, RuntimeError>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let value = value.ok_or_else(|| {
        model_execution(
            model,
            format!("headless=infer requires the {name} query parameter"),
        )
    })?;
    value.parse::<T>().map_err(|error| {
        model_execution(
            model,
            format!("invalid headless=infer {name} value {value:?}: {error}"),
        )
    })
}

fn validate_boogu_output(
    output: &burn_image::ImageOutput,
    active: &ActiveBooguJob,
    integrity: IntegrityPolicy,
) -> Result<(), RuntimeError> {
    output
        .validate()
        .map_err(|error| model_execution(&active.model, error.to_string()))?;
    let provenance = &output.provenance;
    if provenance.model != active.model
        || provenance.model_revision != active.revision
        || provenance.numeric_format != active.numeric_format
    {
        return Err(model_execution(
            &active.model,
            "output provenance does not match the dispatched Boogu release/profile",
        ));
    }
    let must_be_verified = integrity == IntegrityPolicy::RequireSha256;
    if provenance.artifacts_verified != must_be_verified
        || (must_be_verified && provenance.artifact_content_digest.is_none())
    {
        return Err(model_execution(
            &active.model,
            "output artifact verification provenance does not match the configured integrity policy",
        ));
    }
    let backend = provenance.backend.to_ascii_lowercase();
    if !backend.contains("wgpu") && !backend.contains("webgpu") {
        return Err(model_execution(
            &active.model,
            "Boogu output did not attest a WGPU/WebGPU backend",
        ));
    }
    Ok(())
}

fn model_execution(model: &ModelId, message: impl Into<String>) -> RuntimeError {
    RuntimeError::ModelExecution {
        model: model.clone(),
        message: message.into(),
    }
}

fn validate_runtime_variants(
    variants: Vec<BooguVariant>,
) -> Result<Vec<BooguVariant>, FrontendError> {
    if variants.is_empty() {
        return Err(FrontendError::model_runtime(
            "Boogu runtime advertises no initialized release",
        ));
    }
    let mut seen = HashSet::new();
    for variant in &variants {
        if !seen.insert(*variant) {
            return Err(FrontendError::model_runtime(
                "Boogu runtime advertises a release more than once",
            ));
        }
        BooguReleaseIdentity::canonical(*variant)
            .validate()
            .map_err(|error| FrontendError::model_runtime(error.to_string()))?;
    }
    Ok(variants)
}

fn canonical_releases() -> Vec<BooguReleaseIdentity> {
    vec![
        BooguReleaseIdentity::canonical(BooguVariant::Image01Turbo),
        BooguReleaseIdentity::canonical(BooguVariant::Image01EditTurbo),
        BooguReleaseIdentity::canonical(BooguVariant::Image01EditTurbo1k5),
    ]
}

pub(crate) fn validate_execution_variant(
    variant: BooguVariant,
    execution: WgpuExecutionKind,
) -> Result<(), RuntimeError> {
    if execution == WgpuExecutionKind::BrowserWebGpu && variant == BooguVariant::Image01EditTurbo1k5
    {
        return Err(model_execution(
            &boogu_model_id(variant),
            "Edit-Turbo 1.5K is native-WGPU only: browser WebGPU numerical and performance parity have not been validated",
        ));
    }
    Ok(())
}

pub(crate) fn validate_variant_profile(
    variant: BooguVariant,
    profile: BooguStorageProfile,
) -> Result<(), RuntimeError> {
    let descriptor = boogu_model_descriptor(variant);
    let format = numeric_format(profile);
    if !descriptor.capabilities.numeric_formats.contains(&format) {
        return Err(model_execution(
            &descriptor.id,
            format!(
                "artifact profile {} is not validated for this immutable release",
                artifact_profile_id(profile).as_str()
            ),
        ));
    }
    Ok(())
}

pub fn boogu_model_id(variant: BooguVariant) -> ModelId {
    boogu_model_descriptor(variant).id
}

pub fn variant_for_model(model: &ModelId) -> Option<BooguVariant> {
    [
        BooguVariant::Image01Turbo,
        BooguVariant::Image01EditTurbo,
        BooguVariant::Image01EditTurbo1k5,
    ]
    .into_iter()
    .find(|variant| boogu_model_descriptor(*variant).id == *model)
}

/// Stable directory stem for one logical Boogu release.
pub const fn boogu_bundle_slug(variant: BooguVariant) -> &'static str {
    match variant {
        BooguVariant::Image01Turbo => "boogu-image-0.1-turbo",
        BooguVariant::Image01EditTurbo => "boogu-image-0.1-edit-turbo",
        BooguVariant::Image01EditTurbo1k5 => "boogu-image-0.1-edit-turbo-1k5",
    }
}

/// Stable internal storage-profile identity used by manifests and runtime provenance.
///
/// The production CDN uses concise model-name bundle ids. This precise profile name remains
/// unchanged so existing local bundles, CLI arguments, and sealed Burnpack metadata stay
/// compatible.
pub const fn boogu_profile_slug(profile: BooguStorageProfile) -> &'static str {
    match profile {
        BooguStorageProfile::F16 => "f16",
        BooguStorageProfile::F16QwenVisionF32 => "f16-qwen-vision-f32",
        BooguStorageProfile::Q8sBlock32F32 => "q8s-block32-f32",
        BooguStorageProfile::Q8sBlock32F32QwenVisionF32 => "q8s-block32-f32-qwen-vision-f32",
    }
}

/// Legacy descriptive manifest identity used by explicit-source bundles and diagnostic conversions.
pub fn boogu_legacy_bundle_id(variant: BooguVariant, profile: BooguStorageProfile) -> String {
    legacy_artifact_bundle_id(variant, profile)
}

/// Preferred manifest bundle id used as the single `{model_name}` CDN path segment.
///
/// Published tuples take their immutable id from `burn_boogu`'s release registry. Unpublished
/// diagnostic tuples retain the legacy variant/profile identity and require an explicit source.
pub fn boogu_bundle_id(variant: BooguVariant, profile: BooguStorageProfile) -> String {
    preferred_artifact_bundle_id(variant, profile)
}

/// Canonical immutable CDN base URL for one published variant/profile bundle.
///
/// Diagnostic tuples deliberately return `None`; callers must provide an explicit source rather
/// than accidentally synthesizing an apparently canonical Aberration URL.
pub fn boogu_cdn_base_url(variant: BooguVariant, profile: BooguStorageProfile) -> Option<String> {
    canonical_published_bundle(variant, profile)
        .map(|bundle| format!("{BOOGU_CDN_ROOT}/{}", bundle.bundle_id))
}

pub fn boogu_descriptor(variant: BooguVariant, profile: BooguStorageProfile) -> ModelDescriptor {
    boogu_descriptor_for_execution(variant, profile, current_execution_kind())
}

/// Exact descriptor used only by the surface-free 1.5K browser parity replay.
///
/// Ordinary browser capabilities intentionally remain fixed at 256 square and reject the 1.5K
/// release. This separate seam prevents an unfinished qualification path from being advertised by
/// the interactive UI or accepted by normal request dispatch.
#[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))]
pub(crate) fn boogu_browser_1k5_parity_descriptor(
    variant: BooguVariant,
    profile: BooguStorageProfile,
) -> Result<ModelDescriptor, RuntimeError> {
    if variant != BooguVariant::Image01EditTurbo1k5 {
        return Err(model_execution(
            &boogu_model_id(variant),
            "headless=parity requires variant=edit-turbo-1k5",
        ));
    }
    validate_variant_profile(variant, profile)?;
    if profile != BooguStorageProfile::F16QwenVisionF32 {
        return Err(model_execution(
            &boogu_model_id(variant),
            "1.5K browser parity requires the production mixed-F16 artifact profile",
        ));
    }

    let mut descriptor =
        boogu_descriptor_for_execution(variant, profile, WgpuExecutionKind::NativeWgpu);
    let exact = Dimensions::new(
        BOOGU_BROWSER_1K5_PARITY_OUTPUT_EDGE,
        BOOGU_BROWSER_1K5_PARITY_OUTPUT_EDGE,
    )
    .expect("fixed 1.5K parity dimensions are valid");
    let dimensions = &mut descriptor.capabilities.dimensions;
    dimensions.min_width = exact.width();
    dimensions.max_width = exact.width();
    dimensions.min_height = exact.height();
    dimensions.max_height = exact.height();
    dimensions.max_pixels = Some(BOOGU_BROWSER_1K5_PARITY_OUTPUT_PIXELS);
    dimensions.allowed_dimensions = Some([exact].into_iter().collect());
    descriptor
        .validate()
        .map_err(|error| model_execution(&descriptor.id, error.to_string()))?;
    Ok(descriptor)
}

/// Validate the applied device limits for the exact 1.5K F32 striped-tail browser execution plan.
///
/// Both limits are checked because WebGPU implementations can independently cap a buffer's total
/// allocation size and the portion visible through one storage binding.
#[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))]
pub(crate) fn validate_browser_1k5_buffer_limits(
    max_storage_buffer_binding_size: u64,
    max_buffer_size: u64,
) -> Result<(), RuntimeError> {
    let model = boogu_model_id(BooguVariant::Image01EditTurbo1k5);
    for (name, actual) in [
        (
            "max_storage_buffer_binding_size",
            max_storage_buffer_binding_size,
        ),
        ("max_buffer_size", max_buffer_size),
    ] {
        if actual < BOOGU_BROWSER_1K5_MIN_RUNTIME_BUFFER_BYTES {
            return Err(model_execution(
                &model,
                format!(
                    "browser {name} is {actual} bytes; exact 1536-square parity requires at least {} bytes for the F32 two-slab VAE tail (an untiled [1,256,1536,1536] feature would require {} bytes; the F32 denoiser FFN requires {} bytes)",
                    BOOGU_BROWSER_1K5_VAE_STRIPED_TAIL_BUFFER_BYTES,
                    BOOGU_BROWSER_1K5_VAE_FINAL_FEATURE_BUFFER_BYTES,
                    BOOGU_BROWSER_1K5_DENOISER_FFN_BUFFER_BYTES,
                ),
            ));
        }
    }
    Ok(())
}

fn boogu_descriptor_for_execution(
    variant: BooguVariant,
    profile: BooguStorageProfile,
    execution: WgpuExecutionKind,
) -> ModelDescriptor {
    let numeric_format = numeric_format(profile);
    let mut descriptor = boogu_model_descriptor(variant);
    // The model descriptor lists every supported artifact format; one adapter
    // instance advertises only the profile its factory was asked to load. An unsupported
    // selection must never overwrite the release's validated format set with a false claim.
    if descriptor
        .capabilities
        .numeric_formats
        .contains(&numeric_format)
    {
        descriptor.capabilities.numeric_formats = [numeric_format].into_iter().collect();
    }
    if execution == WgpuExecutionKind::BrowserWebGpu {
        let dimensions = &mut descriptor.capabilities.dimensions;
        dimensions.min_width = BOOGU_BROWSER_OUTPUT_EDGE;
        dimensions.max_width = BOOGU_BROWSER_OUTPUT_EDGE;
        dimensions.min_height = BOOGU_BROWSER_OUTPUT_EDGE;
        dimensions.max_height = BOOGU_BROWSER_OUTPUT_EDGE;
        dimensions.max_pixels = Some(BOOGU_BROWSER_MAX_OUTPUT_PIXELS);
    }
    descriptor
}

fn artifact_profile_id(profile: BooguStorageProfile) -> ArtifactProfileId {
    ArtifactProfileId::new(boogu_profile_slug(profile))
        .expect("canonical Boogu artifact profile ids are valid")
}

fn numeric_format(profile: BooguStorageProfile) -> NumericFormat {
    match profile {
        BooguStorageProfile::F16 => NumericFormat::F16,
        BooguStorageProfile::F16QwenVisionF32 => NumericFormat::Other("f16-qwen-vision-f32".into()),
        BooguStorageProfile::Q8sBlock32F32 => NumericFormat::Other("q8s-block32-f32".into()),
        BooguStorageProfile::Q8sBlock32F32QwenVisionF32 => {
            NumericFormat::Other("q8s-block32-f32-qwen-vision-f32".into())
        }
    }
}

const fn current_execution_kind() -> WgpuExecutionKind {
    if cfg!(target_arch = "wasm32") {
        WgpuExecutionKind::BrowserWebGpu
    } else {
        WgpuExecutionKind::NativeWgpu
    }
}

/// Native executable entry point for applications that supply a real factory.
///
/// Keeping the factory argument mandatory prevents a CLI from appearing to run
/// Boogu while silently showing a placeholder or model-neutral shell.
#[cfg(all(feature = "app", not(target_arch = "wasm32")))]
pub fn run_boogu_cli<F: BooguRuntimeFactory>(settings: BooguAdapterSettings, factory: F) {
    crate::app::build_boogu_app(settings, factory).run();
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use bevy::prelude::*;
    use burn_boogu::artifacts::{
        BooguStorageProfile, EDIT_TURBO_1K5_REVISION, EDIT_TURBO_REVISION, TURBO_REVISION,
        artifact_bundle_id_is_compatible,
    };
    use burn_image::{
        ArtifactSource, ColorSpace, EditRequest, GenerateRequest, GenerationOptions, ImageRequest,
        InputImage, PixelBuffer, PixelFormat, Prompt, RemoteBaseUrl,
    };

    use crate::{
        BackendDeviceInfo, BackendStatus, ImageJobId, ImageJobPhase, ImageJobPlugin, ImageJobs,
        SubmitImageJob,
    };

    use super::*;

    fn settings() -> BooguAdapterSettings {
        BooguAdapterSettings::verified_f16(ArtifactSource::Remote {
            base_url: RemoteBaseUrl::new("https://cdn.example/boogu").unwrap(),
        })
    }

    fn request() -> ImageRequest {
        ImageRequest::Generate(GenerateRequest {
            prompt: Prompt::new("a lighthouse at dusk").unwrap(),
            negative_prompt: None,
            options: GenerationOptions {
                steps: Some(4),
                guidance_scale: Some(1.0),
                seed: Some(7),
                ..GenerationOptions::default()
            },
        })
    }

    fn edit_request() -> ImageRequest {
        let dimensions = Dimensions::new(16, 16).unwrap();
        ImageRequest::Edit(EditRequest {
            source: InputImage::Pixels(
                PixelBuffer::new(
                    dimensions,
                    PixelFormat::Rgb8,
                    ColorSpace::Srgb,
                    vec![127; dimensions.checked_byte_len(3).unwrap()],
                )
                .unwrap(),
            ),
            instruction: Prompt::new("make the light warmer").unwrap(),
            negative_prompt: None,
            mask: None,
            strength: None,
            options: GenerationOptions {
                steps: Some(4),
                guidance_scale: Some(1.0),
                seed: Some(7),
                ..GenerationOptions::default()
            },
        })
    }

    #[test]
    fn descriptors_are_pinned_to_burn_boogu_releases_correctness() {
        let generate = boogu_descriptor(BooguVariant::Image01Turbo, BooguStorageProfile::F16);
        let edit = boogu_descriptor(
            BooguVariant::Image01EditTurbo,
            BooguStorageProfile::Q8sBlock32F32,
        );
        let edit_1k5 = boogu_descriptor(
            BooguVariant::Image01EditTurbo1k5,
            BooguStorageProfile::F16QwenVisionF32,
        );
        generate.validate().unwrap();
        edit.validate().unwrap();
        edit_1k5.validate().unwrap();
        assert_eq!(generate.id.as_str(), "Boogu/Boogu-Image-0.1-Turbo");
        assert_eq!(generate.revision, TURBO_REVISION);
        assert_eq!(edit.id.as_str(), "Boogu/Boogu-Image-0.1-Edit-Turbo");
        assert_eq!(edit.revision, EDIT_TURBO_REVISION);
        assert_eq!(edit_1k5.id.as_str(), "Boogu/Boogu-Image-0.1-Edit-Turbo-1K5");
        assert_eq!(edit_1k5.revision, EDIT_TURBO_1K5_REVISION);
        assert_eq!(
            boogu_bundle_slug(BooguVariant::Image01EditTurbo1k5),
            "boogu-image-0.1-edit-turbo-1k5"
        );
        assert_eq!(
            boogu_bundle_id(
                BooguVariant::Image01EditTurbo1k5,
                BooguStorageProfile::F16QwenVisionF32,
            ),
            "boogu-image-0.1-edit-turbo-1k5"
        );
        assert_eq!(
            boogu_legacy_bundle_id(
                BooguVariant::Image01EditTurbo1k5,
                BooguStorageProfile::F16QwenVisionF32,
            ),
            "boogu-image-0.1-edit-turbo-1k5-f16-qwen-vision-f32"
        );
        assert_eq!(
            boogu_cdn_base_url(
                BooguVariant::Image01Turbo,
                BooguStorageProfile::F16QwenVisionF32,
            ),
            Some("https://aberration.technology/model/boogu-image-0.1-turbo".into())
        );
        assert_eq!(
            boogu_cdn_base_url(BooguVariant::Image01Turbo, BooguStorageProfile::F16),
            None
        );
        assert!(artifact_bundle_id_is_compatible(
            BooguVariant::Image01Turbo,
            BooguStorageProfile::F16QwenVisionF32,
            "boogu-image-0.1-turbo-f16-qwen-vision-f32",
        ));
        assert!(artifact_bundle_id_is_compatible(
            BooguVariant::Image01Turbo,
            BooguStorageProfile::F16QwenVisionF32,
            "boogu-image-0.1-turbo",
        ));
        assert_eq!(
            variant_for_model(&edit_1k5.id),
            Some(BooguVariant::Image01EditTurbo1k5)
        );
        assert_eq!(generate.capabilities.min_steps, 4);
        assert_eq!(generate.capabilities.max_steps, 4);
        assert!(!edit.capabilities.supports_masks);
    }

    #[test]
    fn browser_descriptor_and_default_are_exact_256_correctness() {
        let native = boogu_descriptor_for_execution(
            BooguVariant::Image01Turbo,
            BooguStorageProfile::F16QwenVisionF32,
            WgpuExecutionKind::NativeWgpu,
        );
        assert_eq!(
            native.capabilities.dimensions,
            boogu_model_descriptor(BooguVariant::Image01Turbo)
                .capabilities
                .dimensions
        );

        let browser = boogu_descriptor_for_execution(
            BooguVariant::Image01Turbo,
            BooguStorageProfile::F16QwenVisionF32,
            WgpuExecutionKind::BrowserWebGpu,
        );
        let dimensions = &browser.capabilities.dimensions;
        assert_eq!(dimensions.min_width, BOOGU_BROWSER_OUTPUT_EDGE);
        assert_eq!(dimensions.max_width, BOOGU_BROWSER_OUTPUT_EDGE);
        assert_eq!(dimensions.min_height, BOOGU_BROWSER_OUTPUT_EDGE);
        assert_eq!(dimensions.max_height, BOOGU_BROWSER_OUTPUT_EDGE);
        assert_eq!(dimensions.max_pixels, Some(BOOGU_BROWSER_MAX_OUTPUT_PIXELS));
        assert!(
            dimensions
                .supports(Dimensions::new(256, 256).unwrap())
                .is_ok()
        );
        assert!(
            dimensions
                .supports(Dimensions::new(512, 512).unwrap())
                .is_err()
        );

        let browser_job = prepare_runtime_job_for_execution(
            ImageJobId(90),
            BooguVariant::Image01Turbo,
            request(),
            &settings(),
            WgpuExecutionKind::BrowserWebGpu,
        )
        .unwrap();
        assert_eq!(
            browser_job.resolved.dimensions,
            Dimensions::new(256, 256).unwrap()
        );
        let ImageRequest::Generate(browser_request) = browser_job.request else {
            panic!("Turbo request must remain generation")
        };
        assert_eq!(
            browser_request.options.dimensions,
            Some(Dimensions::new(256, 256).unwrap())
        );

        let native_job = prepare_runtime_job_for_execution(
            ImageJobId(91),
            BooguVariant::Image01Turbo,
            request(),
            &settings(),
            WgpuExecutionKind::NativeWgpu,
        )
        .unwrap();
        assert_eq!(
            native_job.resolved.dimensions,
            Dimensions::new(1024, 1024).unwrap()
        );
        let ImageRequest::Generate(native_request) = native_job.request else {
            panic!("Turbo request must remain generation")
        };
        assert_eq!(native_request.options.dimensions, None);
    }

    #[test]
    fn edit_turbo_1k5_native_defaults_and_browser_rejection_are_explicit_correctness() {
        let supported = BooguAdapterSettings::verified_default(ArtifactSource::Remote {
            base_url: RemoteBaseUrl::new("https://cdn.example/boogu").unwrap(),
        });
        let native_job = prepare_runtime_job_for_execution(
            ImageJobId(92),
            BooguVariant::Image01EditTurbo1k5,
            edit_request(),
            &supported,
            WgpuExecutionKind::NativeWgpu,
        )
        .unwrap();
        assert_eq!(native_job.task, BooguTask::Edit);
        assert_eq!(native_job.release.model_revision, EDIT_TURBO_1K5_REVISION);
        assert_eq!(
            native_job.resolved.dimensions,
            Dimensions::new(1536, 1536).unwrap()
        );
        assert_eq!(native_job.request.options().dimensions, None);

        let error = prepare_runtime_job_for_execution(
            ImageJobId(93),
            BooguVariant::Image01EditTurbo1k5,
            edit_request(),
            &supported,
            WgpuExecutionKind::BrowserWebGpu,
        )
        .unwrap_err();
        let message = error.to_string();
        assert!(message.contains("native-WGPU only"), "{message}");
        assert!(
            message.contains("parity have not been validated"),
            "{message}"
        );

        let mut unsupported = settings();
        unsupported.storage_profile = BooguStorageProfile::F16;
        let error = prepare_runtime_job_for_execution(
            ImageJobId(94),
            BooguVariant::Image01EditTurbo1k5,
            edit_request(),
            &unsupported,
            WgpuExecutionKind::NativeWgpu,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("profile f16 is not validated for this immutable release")
        );

        let descriptor = boogu_descriptor_for_execution(
            BooguVariant::Image01EditTurbo1k5,
            BooguStorageProfile::F16,
            WgpuExecutionKind::NativeWgpu,
        );
        assert_eq!(
            descriptor.capabilities.numeric_formats,
            [NumericFormat::Other("f16-qwen-vision-f32".into())]
                .into_iter()
                .collect()
        );
    }

    #[test]
    fn production_settings_select_mixed_qwen_vision_profile_correctness() {
        let settings = BooguAdapterSettings::verified_default(ArtifactSource::Remote {
            base_url: RemoteBaseUrl::new("https://cdn.example/boogu").unwrap(),
        });
        assert_eq!(
            settings.storage_profile,
            BooguStorageProfile::F16QwenVisionF32
        );
        assert_eq!(
            settings.numeric_format(),
            NumericFormat::Other("f16-qwen-vision-f32".into())
        );
        assert_eq!(
            settings
                .runtime_config(BooguVariant::Image01Turbo)
                .artifact_profile
                .as_str(),
            "f16-qwen-vision-f32"
        );
        assert_eq!(
            settings.vae_float_load_policy(),
            BooguFloatLoadPolicy::AdaptToF32
        );
        assert_eq!(
            settings.denoiser_float_load_policy(),
            BooguFloatLoadPolicy::Preserve
        );
        assert_eq!(
            settings.qwen_quantized_load_policy(),
            BooguQuantizedLoadPolicy::Preserve
        );
    }

    #[test]
    fn q8_runtime_policy_uses_f32_denoiser_activations_correctness() {
        let mut settings = settings();
        settings.storage_profile = BooguStorageProfile::Q8sBlock32F32QwenVisionF32;
        assert_eq!(
            settings.denoiser_float_load_policy(),
            BooguFloatLoadPolicy::AdaptToF32
        );
        assert_eq!(
            settings.vae_float_load_policy(),
            BooguFloatLoadPolicy::AdaptToF32
        );
        assert_eq!(
            settings.qwen_quantized_load_policy(),
            BooguQuantizedLoadPolicy::DequantizeF16
        );
        assert_eq!(
            settings.denoiser_quantized_load_policy(),
            BooguQuantizedLoadPolicy::Preserve
        );
    }

    #[test]
    fn concrete_factories_reject_unimplemented_cache_semantics_correctness() {
        let mut settings = settings();
        assert_eq!(settings.validate_concrete_cache_policy(), Ok(()));
        settings.cache = ArtifactCachePolicy::Refresh;
        assert!(settings.validate_concrete_cache_policy().is_err());
        settings.cache = ArtifactCachePolicy::Bypass;
        assert!(settings.validate_concrete_cache_policy().is_err());
    }

    #[test]
    fn browser_device_limit_covers_largest_post_load_tensor_correctness() {
        // The released row plan stores 25,323 x 4,096 F16 values per embedding object. Browser
        // execution adapts that object to F32 before upload, independently of the transport cap.
        assert_eq!(
            BOOGU_BROWSER_MAX_APPLIED_BUFFER_BYTES,
            25_323_u64 * 4_096 * std::mem::size_of::<f32>() as u64
        );
        assert_eq!(
            BOOGU_BROWSER_REQUESTED_BUFFER_LIMIT_BYTES,
            512 * 1024 * 1024
        );
        assert_eq!(
            BOOGU_BROWSER_REQUESTED_BUFFER_LIMIT_BYTES
                .saturating_sub(BOOGU_BROWSER_MAX_APPLIED_BUFFER_BYTES),
            121_978_880
        );
        assert_eq!(
            BOOGU_BROWSER_1K5_VAE_FINAL_FEATURE_BUFFER_BYTES,
            256_u64
                * BOOGU_BROWSER_1K5_PARITY_OUTPUT_EDGE as u64
                * BOOGU_BROWSER_1K5_PARITY_OUTPUT_EDGE as u64
                * std::mem::size_of::<f32>() as u64
        );
        assert_eq!(
            BOOGU_BROWSER_1K5_MIN_RUNTIME_BUFFER_BYTES,
            BOOGU_BROWSER_1K5_VAE_STRIPED_TAIL_BUFFER_BYTES
        );
        assert_eq!(
            BOOGU_BROWSER_1K5_VAE_STRIPED_TAIL_BUFFER_BYTES,
            256_u64 * 1_538 * 772 * std::mem::size_of::<f32>() as u64
        );
        assert_eq!(
            BOOGU_BROWSER_1K5_MIN_REQUIRED_BUFFER_LIMIT_BYTES,
            BOOGU_BROWSER_1K5_MIN_RUNTIME_BUFFER_BYTES
        );
        assert_eq!(BOOGU_BROWSER_1K5_MIN_RUNTIME_BUFFER_BYTES % 256, 0);
        assert_eq!(BOOGU_BROWSER_1K5_DENOISER_FFN_BUFFER_BYTES, 522_042_368);
    }

    #[test]
    fn browser_1k5_parity_surface_is_exact_and_fail_closed_correctness() {
        let descriptor = boogu_browser_1k5_parity_descriptor(
            BooguVariant::Image01EditTurbo1k5,
            BooguStorageProfile::F16QwenVisionF32,
        )
        .unwrap();
        let dimensions = &descriptor.capabilities.dimensions;
        let exact = Dimensions::new(1536, 1536).unwrap();
        assert_eq!(dimensions.min_width, 1536);
        assert_eq!(dimensions.max_width, 1536);
        assert_eq!(dimensions.min_height, 1536);
        assert_eq!(dimensions.max_height, 1536);
        assert_eq!(dimensions.max_pixels, Some(1536_u64 * 1536));
        assert_eq!(
            dimensions.allowed_dimensions,
            Some([exact].into_iter().collect())
        );
        assert!(dimensions.supports(exact).is_ok());
        assert!(
            dimensions
                .supports(Dimensions::new(1536, 1520).unwrap())
                .is_err()
        );

        for variant in [BooguVariant::Image01Turbo, BooguVariant::Image01EditTurbo] {
            let error =
                boogu_browser_1k5_parity_descriptor(variant, BooguStorageProfile::F16QwenVisionF32)
                    .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("headless=parity requires variant=edit-turbo-1k5")
            );
        }
        assert!(
            boogu_browser_1k5_parity_descriptor(
                BooguVariant::Image01EditTurbo1k5,
                BooguStorageProfile::F16,
            )
            .is_err()
        );
    }

    #[test]
    fn browser_1k5_parity_limits_cover_each_measured_buffer_correctness() {
        validate_browser_1k5_buffer_limits(
            BOOGU_BROWSER_1K5_MIN_RUNTIME_BUFFER_BYTES,
            BOOGU_BROWSER_1K5_MIN_RUNTIME_BUFFER_BYTES,
        )
        .unwrap();

        for (storage, buffer, missing) in [
            (
                BOOGU_BROWSER_1K5_MIN_RUNTIME_BUFFER_BYTES - 1,
                BOOGU_BROWSER_1K5_MIN_RUNTIME_BUFFER_BYTES,
                "max_storage_buffer_binding_size",
            ),
            (
                BOOGU_BROWSER_1K5_MIN_RUNTIME_BUFFER_BYTES,
                BOOGU_BROWSER_1K5_MIN_RUNTIME_BUFFER_BYTES - 1,
                "max_buffer_size",
            ),
        ] {
            let error = validate_browser_1k5_buffer_limits(storage, buffer).unwrap_err();
            let message = error.to_string();
            assert!(message.contains(missing), "{message}");
            assert!(message.contains("1215832064"), "{message}");
            assert!(message.contains("2415919104"), "{message}");
            assert!(message.contains("two-slab"), "{message}");
            assert!(message.contains("522042368"), "{message}");
        }

        // Chrome's qualification limit cannot cover the untiled feature but does cover the exact
        // striped-tail plan.
        let chrome_max_buffer_size = 2_147_483_644;
        validate_browser_1k5_buffer_limits(
            BOOGU_BROWSER_1K5_MIN_RUNTIME_BUFFER_BYTES,
            chrome_max_buffer_size,
        )
        .unwrap();
        assert!(BOOGU_BROWSER_1K5_VAE_FINAL_FEATURE_BUFFER_BYTES > chrome_max_buffer_size);
    }

    #[test]
    fn headless_infer_query_is_exact_and_fail_closed_correctness() {
        let request = prepare_headless_generate_request(
            BooguVariant::Image01Turbo,
            Some("a matte red cube on a white background".into()),
            Some("1234".into()),
            Some("256".into()),
            Some("256".into()),
        )
        .unwrap();
        let ImageRequest::Generate(generate) = request else {
            panic!("Turbo diagnostic must create a generation request")
        };
        assert_eq!(
            generate.prompt.as_str(),
            "a matte red cube on a white background"
        );
        assert_eq!(generate.options.seed, Some(1234));
        assert_eq!(generate.options.steps, Some(4));
        assert_eq!(generate.options.guidance_scale, Some(1.0));
        assert_eq!(
            generate.options.dimensions,
            Some(Dimensions::new(256, 256).unwrap())
        );

        for rejected in [
            prepare_headless_generate_request(
                BooguVariant::Image01EditTurbo,
                Some("edit".into()),
                Some("1".into()),
                Some("256".into()),
                Some("256".into()),
            ),
            prepare_headless_generate_request(
                BooguVariant::Image01Turbo,
                None,
                Some("1".into()),
                Some("256".into()),
                Some("256".into()),
            ),
            prepare_headless_generate_request(
                BooguVariant::Image01Turbo,
                Some("cube".into()),
                Some("not-a-seed".into()),
                Some("256".into()),
                Some("256".into()),
            ),
            prepare_headless_generate_request(
                BooguVariant::Image01Turbo,
                Some("cube".into()),
                Some("1".into()),
                Some("512".into()),
                Some("256".into()),
            ),
        ] {
            assert!(rejected.is_err());
        }
    }

    #[test]
    fn request_mapping_preserves_upstream_instruction_policy_correctness() {
        let job = prepare_runtime_job(
            ImageJobId(3),
            BooguVariant::Image01Turbo,
            request(),
            &settings(),
        )
        .unwrap();
        assert_eq!(job.task, BooguTask::Generate);
        assert_eq!(job.release.model_revision, TURBO_REVISION);
        assert_eq!(job.instruction_policy.image_count, 0);
        assert_eq!(job.instruction_policy.max_sequence_length, 1280);
        assert!(!job.instruction_policy.truncate);
    }

    #[test]
    fn unsupported_generic_controls_are_rejected_correctness() {
        let mut request = request();
        let ImageRequest::Generate(generate) = &mut request else {
            unreachable!()
        };
        generate.negative_prompt = Some(Prompt::new("fog").unwrap());
        assert!(
            prepare_runtime_job(
                ImageJobId(4),
                BooguVariant::Image01Turbo,
                request,
                &settings(),
            )
            .unwrap_err()
            .to_string()
            .contains("negative prompt")
        );
    }

    struct CapturingRuntime {
        jobs: Arc<Mutex<Vec<BooguRuntimeJob>>>,
    }

    impl BooguRuntime for CapturingRuntime {
        fn variants(&self) -> Vec<BooguVariant> {
            vec![BooguVariant::Image01Turbo]
        }

        fn submit(&mut self, job: BooguRuntimeJob) -> Result<CancellationToken, RuntimeError> {
            self.jobs.lock().unwrap().push(job);
            Ok(CancellationToken::default())
        }

        fn cancel(&mut self, _id: ImageJobId) -> Result<(), RuntimeError> {
            Ok(())
        }

        fn poll(&mut self, _emit: &mut dyn FnMut(ImageRunnerEvent)) {}
    }

    struct ImmediateFactory {
        runtime: Option<CapturingRuntime>,
        started: bool,
    }

    impl BooguRuntimeFactory for ImmediateFactory {
        fn start(&mut self, context: BooguFactoryContext) -> Result<(), RuntimeError> {
            assert!(matches!(
                context.device,
                burn_wgpu::WgpuDevice::Existing(17)
            ));
            assert_eq!(context.releases.len(), 3);
            self.started = true;
            Ok(())
        }

        fn poll(&mut self) -> Result<Option<Box<dyn BooguRuntime>>, RuntimeError> {
            assert!(self.started);
            Ok(self
                .runtime
                .take()
                .map(|runtime| Box::new(runtime) as Box<dyn BooguRuntime>))
        }
    }

    #[test]
    fn plugin_initializes_on_shared_device_and_dispatches_real_job_correctness() {
        let jobs = Arc::new(Mutex::new(Vec::new()));
        let factory = ImmediateFactory {
            runtime: Some(CapturingRuntime {
                jobs: Arc::clone(&jobs),
            }),
            started: false,
        };
        let mut app = App::new();
        app.insert_resource(BackendStatus::ready(BackendDeviceInfo {
            adapter_name: "test gpu".into(),
            backend: "BrowserWebGpu".into(),
            device_type: "Other".into(),
            driver: "test".into(),
            max_storage_buffer_binding_size: 512 * 1024 * 1024,
            max_buffer_size: 512 * 1024 * 1024,
            shared_adapter_device_queue: true,
        }))
        .insert_resource(bevy_burn::BurnDevice::ready(
            burn_wgpu::WgpuDevice::Existing(17),
        ))
        .add_plugins(ImageJobPlugin)
        .add_plugins(BooguAdapterPlugin::new(settings(), factory));

        app.update();
        assert!(matches!(
            app.world().resource::<ImageRunnerStatus>().state,
            ImageRunnerState::Ready { .. }
        ));

        app.world_mut()
            .resource_mut::<Messages<SubmitImageJob>>()
            .write(SubmitImageJob {
                id: ImageJobId(19),
                model: boogu_model_id(BooguVariant::Image01Turbo),
                request: request(),
            });
        app.update();

        let captured = jobs.lock().unwrap();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].variant, BooguVariant::Image01Turbo);
        assert_eq!(captured[0].release.model_revision, TURBO_REVISION);
        assert!(matches!(
            app.world()
                .resource::<ImageJobs>()
                .get(ImageJobId(19))
                .unwrap()
                .phase,
            ImageJobPhase::Queued
        ));
    }
}
