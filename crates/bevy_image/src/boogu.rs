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
#[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))]
use bevy::{
    camera::RenderTarget,
    window::{PrimaryWindow, WindowRef},
};
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
    ArtifactCachePolicy, ArtifactProfileId, ArtifactSource, CancellationToken, ImageRequest,
    IntegrityPolicy, ModelDescriptor, ModelId, NumericFormat, RuntimeConfig, RuntimeError,
};
#[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))]
use burn_image::{Dimensions, GenerateRequest, GenerationOptions, Prompt};
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
/// Measured F32 denoiser feed-forward buffer from the pinned exact 1.5K browser parity fixture.
/// Shape preflight below computes the conservative released-shape bound including the maximum
/// allowed instruction length.
pub const BOOGU_BROWSER_1K5_DENOISER_FFN_BUFFER_BYTES: u64 = 522_042_368;
/// Largest single runtime buffer required by the exact 1.5K browser parity fixture.
pub const BOOGU_BROWSER_1K5_MIN_RUNTIME_BUFFER_BYTES: u64 =
    BOOGU_BROWSER_1K5_VAE_STRIPED_TAIL_BUFFER_BYTES;
/// Device-buffer limit requested by the ordinary Boogu browser runtime.
///
/// This is the maximum shape-aware buffer plan across the released Turbo/Edit 1K surface and all
/// ten official Edit-Turbo 1.5K aspect-ratio presets. The maximum is the strict-F32 striped VAE
/// tail for the 1392x1696 preset.
pub const BOOGU_BROWSER_REQUESTED_BUFFER_LIMIT_BYTES: u64 = 1_217_126_400;
/// Minimum applied device-buffer limit required by the exact 1.5K parity route.
///
/// The exact two-slab decoder tail keeps this below Chrome's observed 2,147,483,644-byte WebGPU
/// ceiling without changing the global middle attention or GroupNorm semantics.
pub const BOOGU_BROWSER_1K5_MIN_REQUIRED_BUFFER_LIMIT_BYTES: u64 =
    BOOGU_BROWSER_1K5_MIN_RUNTIME_BUFFER_BYTES;
/// Legacy 256-square browser edge retained for source compatibility.
///
/// This is no longer the browser output ceiling. Released browser descriptors now expose their
/// native shape presets, with Turbo and Edit defaulting to 1024 square.
#[deprecated(
    since = "0.1.1",
    note = "the browser now uses descriptor-specific full-resolution presets"
)]
pub const BOOGU_BROWSER_OUTPUT_EDGE: u32 = 256;
/// Legacy pixel count corresponding to [`BOOGU_BROWSER_OUTPUT_EDGE`].
///
/// This value is retained only for source compatibility and is not a current browser limit.
#[allow(deprecated)]
#[deprecated(
    since = "0.1.1",
    note = "the browser now validates descriptor-specific full-resolution shapes"
)]
pub const BOOGU_BROWSER_MAX_OUTPUT_PIXELS: u64 =
    BOOGU_BROWSER_OUTPUT_EDGE as u64 * BOOGU_BROWSER_OUTPUT_EDGE as u64;
/// Output side at which browser VAE decode switches to the bounded strict-F32 striped tail.
pub const BOOGU_BROWSER_STRIPED_VAE_MIN_OUTPUT_SIDE: u32 = 1024;
/// Exact output edge replayed by the dedicated 1.5K browser parity route.
pub const BOOGU_BROWSER_1K5_PARITY_OUTPUT_EDGE: u32 = 1536;
/// Exact output pixel count replayed by the dedicated 1.5K browser parity route.
pub const BOOGU_BROWSER_1K5_PARITY_OUTPUT_PIXELS: u64 =
    BOOGU_BROWSER_1K5_PARITY_OUTPUT_EDGE as u64 * BOOGU_BROWSER_1K5_PARITY_OUTPUT_EDGE as u64;
/// Canonical public origin for immutable Boogu artifact bundles.
pub const BOOGU_CDN_ROOT: &str = "https://aberration.technology/model";

/// Exact ordinary-browser policy attested by runtime events and output provenance.
///
/// Inactive primary-window cameras are removed during render extraction, which prevents Bevy from
/// acquiring or presenting the WebGPU canvas while model inference owns the shared device/queue.
#[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))]
pub(crate) const BROWSER_SURFACE_INFERENCE_POLICY: &str = "request-scoped-surface-acquire-suspended/primary-window-cameras-inactive-before-runtime-submit/exact-state-restored-after-terminal-before-output-ready";

/// VAE decode strategy selected by the browser's shape-aware buffer plan.
#[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BrowserVaeDecodePolicy {
    /// Ordinary strict-F32 decode for shapes whose full final feature fits the device contract.
    FullStrictF32,
    /// Exact strict-F32 two-slab final tail, split at this output-width coordinate.
    StripedTailStrictF32 { split_width: usize },
}

/// Largest individual buffers planned for one browser inference shape.
#[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BrowserBufferPlan {
    pub vae_decode_policy: BrowserVaeDecodePolicy,
    pub vae_decode_max_buffer_bytes: u64,
    pub denoiser_ffn_max_buffer_bytes: u64,
    pub required_buffer_limit_bytes: u64,
}

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

    /// Return the latest human-readable construction milestone, if one was
    /// produced since the previous call. Implementations should coalesce
    /// high-frequency byte events and keep this method non-blocking.
    fn take_initialization_progress(&mut self) -> Option<String> {
        None
    }
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

/// Main-world ownership for the ordinary browser's request-scoped render-surface gate.
///
/// This deliberately does not derive its lifetime from `BooguAdapterHost::active`: a frontend
/// cancellation request removes that entry immediately, while the browser task may still own the
/// shared GPU queue until it emits its actual `Cancelled` terminal event.
#[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))]
#[derive(Resource, Debug, Default)]
struct BrowserSurfaceInferenceGate {
    active_jobs: HashSet<ImageJobId>,
    saved_primary_window_camera_states: BTreeMap<Entity, bool>,
    primary_window: Option<Entity>,
    violation: Option<String>,
}

#[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BrowserSurfaceSuspendReport {
    primary_window_camera_count: usize,
    saved_camera_state_count: usize,
    previously_active_camera_count: usize,
    inactive_camera_count: usize,
    active_job_count: usize,
}

#[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BrowserSurfaceResumeReport {
    primary_window_camera_count: usize,
    saved_camera_state_count: usize,
    restored_camera_state_count: usize,
    restored_active_camera_count: usize,
    active_job_count: usize,
    exact_saved_states_restored: bool,
    all_primary_window_cameras_restored: bool,
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
        });
        #[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))]
        app.init_resource::<BrowserSurfaceInferenceGate>()
            // `Last` still runs before render-world extraction. It catches any primary-window
            // camera created after dispatch and forces it inactive before that frame can acquire.
            .add_systems(Last, enforce_browser_surface_inference_gate);
        app.add_systems(PreUpdate, initialize_boogu_runtime)
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

#[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))]
fn render_target_is_primary_window(target: &RenderTarget, primary_window: Entity) -> bool {
    match target {
        RenderTarget::Window(WindowRef::Primary) => true,
        RenderTarget::Window(WindowRef::Entity(entity)) => *entity == primary_window,
        RenderTarget::Image(_) | RenderTarget::TextureView(_) | RenderTarget::None { .. } => false,
    }
}

#[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))]
fn suspend_browser_surface_inference(
    gate: &mut BrowserSurfaceInferenceGate,
    primary_windows: &Query<Entity, With<PrimaryWindow>>,
    cameras: &mut Query<(Entity, &mut Camera, &RenderTarget)>,
) -> Result<BrowserSurfaceSuspendReport, String> {
    if !gate.active_jobs.is_empty()
        || !gate.saved_primary_window_camera_states.is_empty()
        || gate.primary_window.is_some()
    {
        return Err(
            "browser surface gate refuses a second request while a GPU request is active".into(),
        );
    }
    let primary_window = primary_windows.single().map_err(|error| {
        format!("browser surface gate requires exactly one primary window: {error}")
    })?;
    gate.primary_window = Some(primary_window);
    gate.violation = None;

    let mut primary_window_camera_count = 0;
    let mut previously_active_camera_count = 0;
    let mut inactive_camera_count = 0;
    for (entity, mut camera, target) in cameras.iter_mut() {
        if !render_target_is_primary_window(target, primary_window) {
            continue;
        }
        primary_window_camera_count += 1;
        previously_active_camera_count += usize::from(camera.is_active);
        gate.saved_primary_window_camera_states
            .insert(entity, camera.is_active);
        camera.is_active = false;
        inactive_camera_count += usize::from(!camera.is_active);
    }

    if primary_window_camera_count < 2 || inactive_camera_count != primary_window_camera_count {
        let report = restore_pending_browser_surface_inference(gate, primary_windows, cameras);
        return Err(format!(
            "browser surface gate requires both primary-window cameras inactive before runtime submit; found {primary_window_camera_count}, inactive {inactive_camera_count}, rollback_exact={}",
            report.exact_saved_states_restored
        ));
    }

    Ok(BrowserSurfaceSuspendReport {
        primary_window_camera_count,
        saved_camera_state_count: gate.saved_primary_window_camera_states.len(),
        previously_active_camera_count,
        inactive_camera_count,
        active_job_count: 0,
    })
}

#[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))]
fn capture_and_deactivate_new_primary_window_cameras(
    gate: &mut BrowserSurfaceInferenceGate,
    primary_window: Entity,
    cameras: &mut Query<(Entity, &mut Camera, &RenderTarget)>,
) -> usize {
    let mut primary_window_camera_count = 0;
    for (entity, mut camera, target) in cameras.iter_mut() {
        if !render_target_is_primary_window(target, primary_window) {
            continue;
        }
        primary_window_camera_count += 1;
        gate.saved_primary_window_camera_states
            .entry(entity)
            .or_insert(camera.is_active);
        camera.is_active = false;
    }
    primary_window_camera_count
}

#[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))]
fn restore_pending_browser_surface_inference(
    gate: &mut BrowserSurfaceInferenceGate,
    primary_windows: &Query<Entity, With<PrimaryWindow>>,
    cameras: &mut Query<(Entity, &mut Camera, &RenderTarget)>,
) -> BrowserSurfaceResumeReport {
    restore_browser_surface_camera_states(gate, primary_windows, cameras)
}

#[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))]
fn resume_browser_surface_inference(
    gate: &mut BrowserSurfaceInferenceGate,
    id: ImageJobId,
    primary_windows: &Query<Entity, With<PrimaryWindow>>,
    cameras: &mut Query<(Entity, &mut Camera, &RenderTarget)>,
) -> BrowserSurfaceResumeReport {
    if !gate.active_jobs.remove(&id) {
        return BrowserSurfaceResumeReport {
            primary_window_camera_count: 0,
            saved_camera_state_count: gate.saved_primary_window_camera_states.len(),
            restored_camera_state_count: 0,
            restored_active_camera_count: 0,
            active_job_count: gate.active_jobs.len(),
            exact_saved_states_restored: false,
            all_primary_window_cameras_restored: false,
        };
    }
    if !gate.active_jobs.is_empty() {
        gate.violation.get_or_insert_with(|| {
            "browser surface gate observed unsupported concurrent active jobs".into()
        });
        return BrowserSurfaceResumeReport {
            primary_window_camera_count: 0,
            saved_camera_state_count: gate.saved_primary_window_camera_states.len(),
            restored_camera_state_count: 0,
            restored_active_camera_count: 0,
            active_job_count: gate.active_jobs.len(),
            exact_saved_states_restored: false,
            all_primary_window_cameras_restored: false,
        };
    }

    if let Ok(primary_window) = primary_windows.single() {
        capture_and_deactivate_new_primary_window_cameras(gate, primary_window, cameras);
    } else {
        gate.violation.get_or_insert_with(|| {
            "browser surface gate lost the unique primary window before terminal restoration".into()
        });
    }
    restore_browser_surface_camera_states(gate, primary_windows, cameras)
}

#[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))]
fn restore_browser_surface_camera_states(
    gate: &mut BrowserSurfaceInferenceGate,
    primary_windows: &Query<Entity, With<PrimaryWindow>>,
    cameras: &mut Query<(Entity, &mut Camera, &RenderTarget)>,
) -> BrowserSurfaceResumeReport {
    let current_primary_window = primary_windows.single().ok();
    let primary_window_matches = current_primary_window == gate.primary_window;
    let saved_camera_state_count = gate.saved_primary_window_camera_states.len();
    let mut primary_window_camera_count = 0;
    let mut restored_camera_state_count = 0;
    let mut restored_active_camera_count = 0;
    let mut every_current_primary_camera_was_saved = true;
    let mut every_saved_camera_still_targets_primary = true;

    for (entity, mut camera, target) in cameras.iter_mut() {
        let targets_primary = current_primary_window
            .is_some_and(|primary| render_target_is_primary_window(target, primary));
        if targets_primary {
            primary_window_camera_count += 1;
            every_current_primary_camera_was_saved &= gate
                .saved_primary_window_camera_states
                .contains_key(&entity);
        }
        let Some(previously_active) = gate
            .saved_primary_window_camera_states
            .get(&entity)
            .copied()
        else {
            continue;
        };
        camera.is_active = previously_active;
        restored_camera_state_count += 1;
        restored_active_camera_count += usize::from(previously_active);
        every_saved_camera_still_targets_primary &= targets_primary;
    }

    let exact_saved_states_restored = gate.violation.is_none()
        && primary_window_matches
        && primary_window_camera_count >= 2
        && restored_camera_state_count == saved_camera_state_count
        && every_current_primary_camera_was_saved
        && every_saved_camera_still_targets_primary;
    let report = BrowserSurfaceResumeReport {
        primary_window_camera_count,
        saved_camera_state_count,
        restored_camera_state_count,
        restored_active_camera_count,
        active_job_count: gate.active_jobs.len(),
        exact_saved_states_restored,
        all_primary_window_cameras_restored: exact_saved_states_restored,
    };
    gate.saved_primary_window_camera_states.clear();
    gate.primary_window = None;
    gate.violation = None;
    report
}

#[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))]
fn enforce_browser_surface_inference_gate(
    mut gate: ResMut<BrowserSurfaceInferenceGate>,
    primary_windows: Query<Entity, With<PrimaryWindow>>,
    mut cameras: Query<(Entity, &mut Camera, &RenderTarget)>,
) {
    if gate.active_jobs.is_empty() {
        return;
    }
    let id = *gate
        .active_jobs
        .iter()
        .next()
        .expect("non-empty browser surface gate has an active job");
    let Ok(primary_window) = primary_windows.single() else {
        if gate.violation.is_none() {
            let message =
                "browser surface gate lost the unique primary window while inference was active";
            gate.violation = Some(message.into());
            report_browser_surface_gate_failure(id, "enforce", message, false);
        }
        return;
    };
    if gate.primary_window != Some(primary_window) && gate.violation.is_none() {
        let message = "browser surface gate primary-window identity changed during inference";
        gate.violation = Some(message.into());
        report_browser_surface_gate_failure(id, "enforce", message, false);
    }
    let camera_count =
        capture_and_deactivate_new_primary_window_cameras(&mut gate, primary_window, &mut cameras);
    if camera_count < 2 && gate.violation.is_none() {
        let message = format!(
            "browser surface gate found only {camera_count} primary-window cameras while inference was active"
        );
        gate.violation = Some(message.clone());
        report_browser_surface_gate_failure(id, "enforce", &message, false);
    }
}

#[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))]
fn report_browser_surface_suspended(id: ImageJobId, report: BrowserSurfaceSuspendReport) {
    #[cfg(all(feature = "boogu-web", target_arch = "wasm32"))]
    crate::browser_boogu::report_browser_surface_inference_suspended(
        id.0,
        report.primary_window_camera_count,
        report.saved_camera_state_count,
        report.previously_active_camera_count,
        report.inactive_camera_count,
        report.active_job_count,
    );
    #[cfg(test)]
    let _ = (id, report);
}

#[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))]
fn report_browser_surface_resumed(
    id: ImageJobId,
    terminal: &'static str,
    report: BrowserSurfaceResumeReport,
) {
    #[cfg(all(feature = "boogu-web", target_arch = "wasm32"))]
    crate::browser_boogu::report_browser_surface_inference_resumed(
        id.0,
        terminal,
        report.primary_window_camera_count,
        report.saved_camera_state_count,
        report.restored_camera_state_count,
        report.restored_active_camera_count,
        report.active_job_count,
        report.exact_saved_states_restored,
        report.all_primary_window_cameras_restored,
    );
    #[cfg(test)]
    let _ = (id, terminal, report);
}

#[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))]
fn report_browser_surface_gate_failure(
    id: ImageJobId,
    phase: &'static str,
    message: &str,
    exact_saved_states_restored: bool,
) {
    #[cfg(all(feature = "boogu-web", target_arch = "wasm32"))]
    crate::browser_boogu::report_browser_surface_inference_gate_failure(
        id.0,
        phase,
        message,
        exact_saved_states_restored,
    );
    #[cfg(test)]
    let _ = (id, phase, message, exact_saved_states_restored);
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

    let factory = host
        .factory
        .as_mut()
        .expect("building adapter retains its factory");
    if let Some(message) = factory.take_initialization_progress() {
        *runner_status = ImageRunnerStatus::initializing(message);
    }
    let poll_result = factory.poll();
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

// Bevy system parameters are independently scheduled resources/queries; grouping the browser-only
// camera gate into a custom SystemParam would obscure the exact mutable access contract here.
#[allow(clippy::too_many_arguments)]
fn stop_on_backend_loss(
    backend: Res<BackendStatus>,
    mut host: ResMut<BooguAdapterHost>,
    mut adapter_status: ResMut<BooguAdapterStatus>,
    mut runner_status: ResMut<ImageRunnerStatus>,
    mut failed: MessageWriter<FailImageJob>,
    #[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))] primary_windows: Query<
        Entity,
        With<PrimaryWindow>,
    >,
    #[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))] mut cameras: Query<(
        Entity,
        &mut Camera,
        &RenderTarget,
    )>,
    #[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))] mut surface_gate: ResMut<
        BrowserSurfaceInferenceGate,
    >,
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
    #[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))]
    if let Some(id) = surface_gate.active_jobs.iter().copied().next() {
        let report =
            resume_browser_surface_inference(&mut surface_gate, id, &primary_windows, &mut cameras);
        report_browser_surface_resumed(id, "backend_loss", report);
        if !report.exact_saved_states_restored {
            report_browser_surface_gate_failure(
                id,
                "restore",
                "browser surface gate could not restore exact camera state after backend loss",
                false,
            );
        }
    }
    fail_adapter(&mut host, &mut adapter_status, &mut runner_status, error);
}

fn submit_boogu_jobs(
    mut host: ResMut<BooguAdapterHost>,
    mut dispatched: MessageReader<ImageJobDispatched>,
    mut failed: MessageWriter<FailImageJob>,
    #[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))] primary_windows: Query<
        Entity,
        With<PrimaryWindow>,
    >,
    #[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))] mut cameras: Query<(
        Entity,
        &mut Camera,
        &RenderTarget,
    )>,
    #[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))] mut surface_gate: ResMut<
        BrowserSurfaceInferenceGate,
    >,
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

        // On Wasm `spawn_local` first polls on a later microtask. Mutating the main-world cameras
        // here is therefore synchronous-before-submit and reaches render extraction before model
        // inference can begin on the shared queue.
        #[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))]
        let surface_suspend = match suspend_browser_surface_inference(
            &mut surface_gate,
            &primary_windows,
            &mut cameras,
        ) {
            Ok(report) => report,
            Err(message) => {
                report_browser_surface_gate_failure(dispatch.id, "suspend", &message, false);
                failed.write(FailImageJob {
                    id: dispatch.id,
                    error: FrontendError::model_runtime(message),
                });
                continue;
            }
        };
        match runtime.submit(job) {
            Ok(token) => {
                #[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))]
                {
                    surface_gate.active_jobs.insert(dispatch.id);
                    let mut report = surface_suspend;
                    report.active_job_count = surface_gate.active_jobs.len();
                    report_browser_surface_suspended(dispatch.id, report);
                }
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
                #[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))]
                {
                    let report = restore_pending_browser_surface_inference(
                        &mut surface_gate,
                        &primary_windows,
                        &mut cameras,
                    );
                    report_browser_surface_gate_failure(
                        dispatch.id,
                        "runtime_submit",
                        &error.to_string(),
                        report.exact_saved_states_restored,
                    );
                }
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
    #[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))] primary_windows: Query<
        Entity,
        With<PrimaryWindow>,
    >,
    #[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))] mut cameras: Query<(
        Entity,
        &mut Camera,
        &RenderTarget,
    )>,
    #[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))] mut surface_gate: ResMut<
        BrowserSurfaceInferenceGate,
    >,
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
                #[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))]
                let surface_resume = resume_browser_surface_inference(
                    &mut surface_gate,
                    id,
                    &primary_windows,
                    &mut cameras,
                );
                #[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))]
                report_browser_surface_resumed(id, "completed", surface_resume);
                let Some(active) = host.active.remove(&id) else {
                    continue;
                };
                #[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))]
                if !surface_resume.exact_saved_states_restored
                    || !surface_resume.all_primary_window_cameras_restored
                {
                    report_browser_surface_gate_failure(
                        id,
                        "restore",
                        "browser surface gate could not restore exact camera state after completion",
                        false,
                    );
                    failed.write(FailImageJob {
                        id,
                        error: FrontendError::model_runtime(
                            "browser surface gate could not restore every primary-window camera to its exact pre-request active state",
                        ),
                    });
                    continue;
                }
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
            ImageRunnerEvent::Failed { id, error } => {
                #[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))]
                {
                    let report = resume_browser_surface_inference(
                        &mut surface_gate,
                        id,
                        &primary_windows,
                        &mut cameras,
                    );
                    report_browser_surface_resumed(id, "failed", report);
                    if !report.exact_saved_states_restored {
                        report_browser_surface_gate_failure(
                            id,
                            "restore",
                            "browser surface gate could not restore exact camera state after failure",
                            false,
                        );
                    }
                }
                if host.active.remove(&id).is_some() {
                    failed.write(FailImageJob {
                        id,
                        error: FrontendError::from(error),
                    });
                }
            }
            ImageRunnerEvent::Cancelled { id } => {
                #[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))]
                {
                    let report = resume_browser_surface_inference(
                        &mut surface_gate,
                        id,
                        &primary_windows,
                        &mut cameras,
                    );
                    report_browser_surface_resumed(id, "cancelled", report);
                    if !report.exact_saved_states_restored {
                        report_browser_surface_gate_failure(
                            id,
                            "restore",
                            "browser surface gate could not restore exact camera state after cancellation",
                            false,
                        );
                    }
                }
                host.active.remove(&id);
            }
            ImageRunnerEvent::Progress { .. } => {}
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
    request: ImageRequest,
    settings: &BooguAdapterSettings,
    execution: WgpuExecutionKind,
) -> Result<BooguRuntimeJob, RuntimeError> {
    let model = boogu_model_id(variant);
    validate_execution_variant(variant, execution)?;
    validate_variant_profile(variant, settings.storage_profile)?;
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

/// Build the Turbo generation request accepted by the surface-free full-inference diagnostic.
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
    let request = ImageRequest::Generate(GenerateRequest {
        prompt,
        negative_prompt: None,
        options: GenerationOptions {
            dimensions: Some(dimensions),
            steps: Some(4),
            guidance_scale: Some(1.0),
            seed: Some(seed),
            batch_size: 1,
        },
    });
    boogu_model_descriptor(variant)
        .capabilities
        .validate_request(&model, &request)?;
    Ok(request)
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
    _variant: BooguVariant,
    _execution: WgpuExecutionKind,
) -> Result<(), RuntimeError> {
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

/// Exact descriptor used by the surface-free 1.5K browser parity replay.
///
/// The ordinary browser descriptor exposes every released shape. This narrower descriptor keeps
/// the exhaustive fixture replay pinned to its authenticated 1536-square tensor inventory.
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
    let dimensions = Dimensions::new(
        BOOGU_BROWSER_1K5_PARITY_OUTPUT_EDGE,
        BOOGU_BROWSER_1K5_PARITY_OUTPUT_EDGE,
    )
    .expect("fixed 1.5K parity dimensions are valid");
    validate_browser_buffer_limits_for_dimensions(
        BooguVariant::Image01EditTurbo1k5,
        dimensions,
        max_storage_buffer_binding_size,
        max_buffer_size,
    )
    .map(|_| ())
}

/// Build the exact maximum-single-buffer plan for one released browser output shape.
#[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))]
pub(crate) fn browser_buffer_plan(
    variant: BooguVariant,
    dimensions: Dimensions,
) -> Result<BrowserBufferPlan, RuntimeError> {
    let model = boogu_model_id(variant);
    boogu_model_descriptor(variant)
        .capabilities
        .dimensions
        .supports(dimensions)
        .map_err(|error| model_execution(&model, error.to_string()))?;

    let width = u64::from(dimensions.width());
    let height = u64::from(dimensions.height());
    let uses_striped_tail = dimensions.width() >= BOOGU_BROWSER_STRIPED_VAE_MIN_OUTPUT_SIDE
        || dimensions.height() >= BOOGU_BROWSER_STRIPED_VAE_MIN_OUTPUT_SIDE;
    let (vae_decode_policy, vae_decode_max_buffer_bytes) = if uses_striped_tail {
        let split_width = dimensions.width() / 2;
        debug_assert!(split_width >= 2 && split_width.is_multiple_of(2));
        let largest_slab_width = u64::from(split_width.max(dimensions.width() - split_width));
        // The first strict-F32 residual after the final upsample operates on 256 channels. Add
        // two rows for explicit vertical padding and four columns for the split halo plus explicit
        // horizontal padding. This reproduces the measured 1536-square [1,256,1538,772] bound.
        let bytes = 256_u64
            .checked_mul(
                height
                    .checked_add(2)
                    .ok_or_else(|| model_execution(&model, "browser VAE height plan overflowed"))?,
            )
            .and_then(|value| value.checked_mul(largest_slab_width.checked_add(4)?))
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>() as u64))
            .ok_or_else(|| model_execution(&model, "browser striped VAE buffer plan overflowed"))?;
        (
            BrowserVaeDecodePolicy::StripedTailStrictF32 {
                split_width: usize::try_from(split_width)
                    .map_err(|error| model_execution(&model, error.to_string()))?,
            },
            bytes,
        )
    } else {
        let bytes = 256_u64
            .checked_mul(height)
            .and_then(|value| value.checked_mul(width))
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>() as u64))
            .ok_or_else(|| model_execution(&model, "browser full VAE buffer plan overflowed"))?;
        (BrowserVaeDecodePolicy::FullStrictF32, bytes)
    };
    // The denoiser patches the /8 latent at 2x2, hence one image token per 16x16 output region.
    // InstructionPolicy::upstream rejects, rather than truncates, sequences above 1280 tokens.
    let image_tokens = width
        .checked_div(16)
        .and_then(|width| width.checked_mul(height / 16))
        .ok_or_else(|| model_execution(&model, "browser denoiser token plan overflowed"))?;
    let joint_tokens = image_tokens
        .checked_add(1_280)
        .ok_or_else(|| model_execution(&model, "browser denoiser sequence plan overflowed"))?;
    let denoiser_ffn_max_buffer_bytes = joint_tokens
        .checked_mul(burn_boogu::BooguConfig::default().ffn_inner_dim() as u64)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>() as u64))
        .ok_or_else(|| model_execution(&model, "browser denoiser FFN plan overflowed"))?;
    let required_buffer_limit_bytes = BOOGU_BROWSER_MAX_APPLIED_BUFFER_BYTES
        .max(denoiser_ffn_max_buffer_bytes)
        .max(vae_decode_max_buffer_bytes);
    Ok(BrowserBufferPlan {
        vae_decode_policy,
        vae_decode_max_buffer_bytes,
        denoiser_ffn_max_buffer_bytes,
        required_buffer_limit_bytes,
    })
}

/// Fail before model execution when either applied WebGPU limit cannot cover the selected shape.
#[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))]
pub(crate) fn validate_browser_buffer_limits_for_dimensions(
    variant: BooguVariant,
    dimensions: Dimensions,
    max_storage_buffer_binding_size: u64,
    max_buffer_size: u64,
) -> Result<BrowserBufferPlan, RuntimeError> {
    let plan = browser_buffer_plan(variant, dimensions)?;
    let model = boogu_model_id(variant);
    for (name, actual) in [
        (
            "max_storage_buffer_binding_size",
            max_storage_buffer_binding_size,
        ),
        ("max_buffer_size", max_buffer_size),
    ] {
        if actual < plan.required_buffer_limit_bytes {
            return Err(model_execution(
                &model,
                format!(
                    "browser {name} is {actual} bytes; {}x{} requires at least {} bytes (VAE decode plan {}, untiled 1536-square feature {} bytes, shape-aware maximum denoiser FFN {} bytes)",
                    dimensions.width(),
                    dimensions.height(),
                    plan.required_buffer_limit_bytes,
                    plan.vae_decode_max_buffer_bytes,
                    BOOGU_BROWSER_1K5_VAE_FINAL_FEATURE_BUFFER_BYTES,
                    plan.denoiser_ffn_max_buffer_bytes,
                ),
            ));
        }
    }
    Ok(plan)
}

/// Validate that the applied limits cover every output shape advertised by one browser release.
#[cfg(any(test, all(feature = "boogu-web", target_arch = "wasm32")))]
pub(crate) fn validate_browser_variant_buffer_limits(
    variant: BooguVariant,
    max_storage_buffer_binding_size: u64,
    max_buffer_size: u64,
) -> Result<(), RuntimeError> {
    let descriptor = boogu_model_descriptor(variant);
    if let Some(allowed) = descriptor.capabilities.dimensions.allowed_dimensions {
        for dimensions in allowed {
            validate_browser_buffer_limits_for_dimensions(
                variant,
                dimensions,
                max_storage_buffer_binding_size,
                max_buffer_size,
            )?;
        }
    } else {
        let dimensions = Dimensions::new(
            descriptor.capabilities.dimensions.max_width,
            descriptor.capabilities.dimensions.max_height,
        )
        .expect("released Boogu maximum dimensions are valid");
        validate_browser_buffer_limits_for_dimensions(
            variant,
            dimensions,
            max_storage_buffer_binding_size,
            max_buffer_size,
        )?;
    }
    Ok(())
}

fn boogu_descriptor_for_execution(
    variant: BooguVariant,
    profile: BooguStorageProfile,
    _execution: WgpuExecutionKind,
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

    use bevy::ecs::system::SystemState;
    use bevy::prelude::*;
    use bevy::window::PrimaryWindow;
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

    type BrowserSurfaceGateTestState<'w, 's> = SystemState<(
        ResMut<'w, BrowserSurfaceInferenceGate>,
        Query<'w, 's, Entity, With<PrimaryWindow>>,
        Query<'w, 's, (Entity, &'static mut Camera, &'static RenderTarget)>,
    )>;

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

    fn spawn_primary_surface(world: &mut World) -> (Entity, Entity, Entity) {
        let window = world.spawn((Window::default(), PrimaryWindow)).id();
        let active_camera = world
            .spawn((Camera::default(), RenderTarget::Window(WindowRef::Primary)))
            .id();
        let inactive_camera = world
            .spawn((
                Camera {
                    is_active: false,
                    ..default()
                },
                RenderTarget::Window(WindowRef::Entity(window)),
            ))
            .id();
        (window, active_camera, inactive_camera)
    }

    fn request_with_dimensions(dimensions: Dimensions) -> ImageRequest {
        let mut request = request();
        match &mut request {
            ImageRequest::Generate(request) => request.options.dimensions = Some(dimensions),
            ImageRequest::Edit(_) => unreachable!("generation helper returned an edit request"),
        }
        request
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

    fn edit_request_with_dimensions(dimensions: Dimensions) -> ImageRequest {
        let mut request = edit_request();
        match &mut request {
            ImageRequest::Edit(request) => request.options.dimensions = Some(dimensions),
            ImageRequest::Generate(_) => unreachable!("edit helper returned a generation request"),
        }
        request
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
    fn browser_descriptor_and_defaults_match_native_full_resolution_correctness() {
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
        assert_eq!(dimensions, &native.capabilities.dimensions);
        assert!(
            dimensions
                .supports(Dimensions::new(256, 256).unwrap())
                .is_ok()
        );
        assert!(
            dimensions
                .supports(Dimensions::new(1024, 1024).unwrap())
                .is_ok()
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
            Dimensions::new(1024, 1024).unwrap()
        );
        let ImageRequest::Generate(browser_request) = browser_job.request else {
            panic!("Turbo request must remain generation")
        };
        assert_eq!(browser_request.options.dimensions, None);

        for edge in [256, 1024] {
            let requested = Dimensions::new(edge, edge).unwrap();
            let job = prepare_runtime_job_for_execution(
                ImageJobId(100 + u64::from(edge)),
                BooguVariant::Image01Turbo,
                request_with_dimensions(requested),
                &settings(),
                WgpuExecutionKind::BrowserWebGpu,
            )
            .unwrap();
            assert_eq!(job.resolved.dimensions, requested);
        }
        let edit_dimensions = Dimensions::new(1024, 1024).unwrap();
        let edit_job = prepare_runtime_job_for_execution(
            ImageJobId(2_024),
            BooguVariant::Image01EditTurbo,
            edit_request_with_dimensions(edit_dimensions),
            &settings(),
            WgpuExecutionKind::BrowserWebGpu,
        )
        .unwrap();
        assert_eq!(edit_job.resolved.dimensions, edit_dimensions);

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
    fn edit_turbo_1k5_native_and_browser_expose_the_same_released_shapes_correctness() {
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

        let browser_job = prepare_runtime_job_for_execution(
            ImageJobId(93),
            BooguVariant::Image01EditTurbo1k5,
            edit_request(),
            &supported,
            WgpuExecutionKind::BrowserWebGpu,
        )
        .unwrap();
        assert_eq!(browser_job.task, BooguTask::Edit);
        assert_eq!(
            browser_job.resolved.dimensions,
            native_job.resolved.dimensions
        );
        let native_descriptor = boogu_descriptor_for_execution(
            BooguVariant::Image01EditTurbo1k5,
            BooguStorageProfile::F16QwenVisionF32,
            WgpuExecutionKind::NativeWgpu,
        );
        let browser_descriptor = boogu_descriptor_for_execution(
            BooguVariant::Image01EditTurbo1k5,
            BooguStorageProfile::F16QwenVisionF32,
            WgpuExecutionKind::BrowserWebGpu,
        );
        assert_eq!(browser_descriptor, native_descriptor);
        for (width, height) in burn_boogu::BOOGU_1K5_OUTPUT_PRESETS {
            assert!(
                browser_descriptor
                    .capabilities
                    .dimensions
                    .supports(Dimensions::new(width, height).unwrap())
                    .is_ok(),
                "browser descriptor rejected released {width}x{height} preset"
            );
        }

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
    fn browser_device_limit_covers_full_resolution_shape_plans_correctness() {
        // The released row plan stores 25,323 x 4,096 F16 values per embedding object. Browser
        // execution adapts that object to F32 before upload, independently of the transport cap.
        assert_eq!(
            BOOGU_BROWSER_MAX_APPLIED_BUFFER_BYTES,
            25_323_u64 * 4_096 * std::mem::size_of::<f32>() as u64
        );
        assert_eq!(BOOGU_BROWSER_REQUESTED_BUFFER_LIMIT_BYTES, 1_217_126_400);
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

        let one_k = Dimensions::new(1024, 1024).unwrap();
        let one_k_plan = browser_buffer_plan(BooguVariant::Image01Turbo, one_k).unwrap();
        assert_eq!(
            one_k_plan.vae_decode_policy,
            BrowserVaeDecodePolicy::StripedTailStrictF32 { split_width: 512 }
        );
        assert_eq!(one_k_plan.vae_decode_max_buffer_bytes, 542_121_984);
        assert_eq!(one_k_plan.denoiser_ffn_max_buffer_bytes, 291_766_272);
        validate_browser_variant_buffer_limits(
            BooguVariant::Image01Turbo,
            one_k_plan.required_buffer_limit_bytes,
            one_k_plan.required_buffer_limit_bytes,
        )
        .unwrap();

        let mut maximum = 0;
        let mut maximum_shape = None;
        for (width, height) in burn_boogu::BOOGU_1K5_OUTPUT_PRESETS {
            let dimensions = Dimensions::new(width, height).unwrap();
            let plan = browser_buffer_plan(BooguVariant::Image01EditTurbo1k5, dimensions).unwrap();
            assert_eq!(
                plan.vae_decode_policy,
                BrowserVaeDecodePolicy::StripedTailStrictF32 {
                    split_width: width as usize / 2,
                }
            );
            validate_browser_buffer_limits_for_dimensions(
                BooguVariant::Image01EditTurbo1k5,
                dimensions,
                BOOGU_BROWSER_REQUESTED_BUFFER_LIMIT_BYTES,
                BOOGU_BROWSER_REQUESTED_BUFFER_LIMIT_BYTES,
            )
            .unwrap();
            if plan.required_buffer_limit_bytes > maximum {
                maximum = plan.required_buffer_limit_bytes;
                maximum_shape = Some((width, height));
            }
        }
        assert_eq!(maximum_shape, Some((1392, 1696)));
        assert_eq!(maximum, BOOGU_BROWSER_REQUESTED_BUFFER_LIMIT_BYTES);
        validate_browser_variant_buffer_limits(BooguVariant::Image01EditTurbo1k5, maximum, maximum)
            .unwrap();
        let error = validate_browser_variant_buffer_limits(
            BooguVariant::Image01EditTurbo1k5,
            maximum - 1,
            maximum,
        )
        .unwrap_err();
        let message = error.to_string();
        assert!(message.contains("1392x1696"), "{message}");
        assert!(message.contains("1217126400"), "{message}");
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
            assert!(message.contains("VAE decode plan"), "{message}");
            assert!(message.contains("569638912"), "{message}");
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

        let one_k = prepare_headless_generate_request(
            BooguVariant::Image01Turbo,
            Some("a lighthouse at dusk".into()),
            Some("7".into()),
            Some("1024".into()),
            Some("1024".into()),
        )
        .unwrap();
        assert_eq!(
            one_k.options().dimensions,
            Some(Dimensions::new(1024, 1024).unwrap())
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
                Some("1040".into()),
                Some("1024".into()),
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
    fn browser_surface_gate_preserves_states_covers_new_cameras_and_waits_for_terminal_correctness()
    {
        let mut app = App::new();
        app.init_resource::<BrowserSurfaceInferenceGate>();
        let (primary_window, active_camera, inactive_camera) =
            spawn_primary_surface(app.world_mut());
        let id = ImageJobId(41);
        let mut gate_state: BrowserSurfaceGateTestState<'_, '_> = SystemState::new(app.world_mut());

        {
            let (mut gate, primary_windows, mut cameras) =
                gate_state.get_mut(app.world_mut()).unwrap();
            let report =
                suspend_browser_surface_inference(&mut gate, &primary_windows, &mut cameras)
                    .unwrap();
            assert_eq!(report.primary_window_camera_count, 2);
            assert_eq!(report.saved_camera_state_count, 2);
            assert_eq!(report.previously_active_camera_count, 1);
            assert_eq!(report.inactive_camera_count, 2);
            assert!(gate.active_jobs.insert(id));
        }
        gate_state.apply(app.world_mut());
        assert!(!app.world().get::<Camera>(active_camera).unwrap().is_active);
        assert!(
            !app.world()
                .get::<Camera>(inactive_camera)
                .unwrap()
                .is_active
        );

        // A camera created while the request is active is captured at the next Last enforcement
        // boundary and cannot reacquire the primary surface.
        let new_camera = app
            .world_mut()
            .spawn((
                Camera::default(),
                RenderTarget::Window(WindowRef::Entity(primary_window)),
            ))
            .id();
        {
            let (mut gate, primary_windows, mut cameras) =
                gate_state.get_mut(app.world_mut()).unwrap();
            let primary_window = primary_windows.single().unwrap();
            assert_eq!(
                capture_and_deactivate_new_primary_window_cameras(
                    &mut gate,
                    primary_window,
                    &mut cameras,
                ),
                3
            );
            // This is the cancellation-request boundary: cancellation may change frontend/host
            // state, but only an actual runtime terminal is allowed to remove this gate ID.
            assert!(gate.active_jobs.contains(&id));
        }
        gate_state.apply(app.world_mut());
        assert!(!app.world().get::<Camera>(new_camera).unwrap().is_active);

        let report = {
            let (mut gate, primary_windows, mut cameras) =
                gate_state.get_mut(app.world_mut()).unwrap();
            resume_browser_surface_inference(&mut gate, id, &primary_windows, &mut cameras)
        };
        gate_state.apply(app.world_mut());
        assert_eq!(report.primary_window_camera_count, 3);
        assert_eq!(report.saved_camera_state_count, 3);
        assert_eq!(report.restored_camera_state_count, 3);
        assert_eq!(report.restored_active_camera_count, 2);
        assert_eq!(report.active_job_count, 0);
        assert!(report.exact_saved_states_restored);
        assert!(report.all_primary_window_cameras_restored);
        assert!(app.world().get::<Camera>(active_camera).unwrap().is_active);
        assert!(
            !app.world()
                .get::<Camera>(inactive_camera)
                .unwrap()
                .is_active
        );
        assert!(app.world().get::<Camera>(new_camera).unwrap().is_active);
    }

    #[test]
    fn browser_surface_gate_fails_closed_when_saved_camera_disappears_correctness() {
        let mut app = App::new();
        app.init_resource::<BrowserSurfaceInferenceGate>();
        let (_, active_camera, _) = spawn_primary_surface(app.world_mut());
        let id = ImageJobId(42);
        let mut gate_state: BrowserSurfaceGateTestState<'_, '_> = SystemState::new(app.world_mut());
        {
            let (mut gate, primary_windows, mut cameras) =
                gate_state.get_mut(app.world_mut()).unwrap();
            suspend_browser_surface_inference(&mut gate, &primary_windows, &mut cameras).unwrap();
            gate.active_jobs.insert(id);
        }
        gate_state.apply(app.world_mut());
        assert!(app.world_mut().despawn(active_camera));
        let report = {
            let (mut gate, primary_windows, mut cameras) =
                gate_state.get_mut(app.world_mut()).unwrap();
            resume_browser_surface_inference(&mut gate, id, &primary_windows, &mut cameras)
        };
        gate_state.apply(app.world_mut());
        assert_eq!(report.saved_camera_state_count, 2);
        assert_eq!(report.restored_camera_state_count, 1);
        assert!(!report.exact_saved_states_restored);
        assert!(!report.all_primary_window_cameras_restored);
    }

    #[test]
    fn browser_surface_gate_schedule_orders_suspend_and_resume_around_runtime_correctness() {
        let source = include_str!("boogu.rs");
        let submit_start = source.find("fn submit_boogu_jobs(").unwrap();
        let submit_end = source[submit_start..]
            .find("fn cancel_boogu_jobs(")
            .map(|offset| submit_start + offset)
            .unwrap();
        let submit = &source[submit_start..submit_end];
        assert!(
            submit.find("suspend_browser_surface_inference(").unwrap()
                < submit.find("runtime.submit(job)").unwrap()
        );
        assert!(
            submit.find("runtime.submit(job)").unwrap()
                < submit.find("report_browser_surface_suspended(").unwrap()
        );

        let poll_start = source.find("fn poll_boogu_runtime(").unwrap();
        let poll_end = source[poll_start..]
            .find("pub(crate) fn prepare_runtime_job(")
            .map(|offset| poll_start + offset)
            .unwrap();
        let poll = &source[poll_start..poll_end];
        assert!(
            poll.find("resume_browser_surface_inference(").unwrap()
                < poll.find("completed.write(CompleteImageJob").unwrap()
        );
        assert!(source.contains(".add_systems(Last, enforce_browser_surface_inference_gate)"));
        assert_eq!(
            BROWSER_SURFACE_INFERENCE_POLICY,
            "request-scoped-surface-acquire-suspended/primary-window-cameras-inactive-before-runtime-submit/exact-state-restored-after-terminal-before-output-ready"
        );
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
        spawn_primary_surface(app.world_mut());
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
