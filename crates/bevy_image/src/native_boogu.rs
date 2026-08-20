//! Concrete native worker for a sealed Boogu Burnpack bundle.
//!
//! The default high-VRAM policy verifies, decodes, and uploads every required Qwen/VAE stage
//! before the runtime becomes ready. It then retains their shared WGPU handles with one verified
//! denoiser for every request and all four DMD steps. No model-weight filesystem read, host decode,
//! or host-to-device upload occurs in that runtime's forward hot path.
//!
//! The supported low-VRAM policy preserves the same qualified native kernels and mixed-F16
//! execution policy, but bounds model-weight residency by streaming Qwen and one VAE half at a time
//! while retaining only the variant-required denoiser weights across DMD steps and requests. Its
//! static resource plan fails closed unless the inventory-audited live-weight bound plus a
//! conservative non-weight reserve stays below 32 GB; runtime telemetry is still required before
//! claiming a measured device peak.
//!
//! Directory sources are synchronous filesystem readers, while the browser adapter performs
//! asynchronous verified CDN orchestration behind the same public [`crate::BooguRuntimeFactory`]
//! seam.

use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, TryRecvError},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use burn::{nn::RmsNorm, prelude::Backend};
use burn_boogu::{
    BooguConfig, BooguError, BooguExecution, BooguImageModel, BooguRuntimeDTypes,
    BooguRuntimeMetadata, BooguVaeStageSource, BooguVariant, DenoiserRmsNormPolicy,
    FluxVaeStageSourceAdapter, NativeAutotunePolicy, NativeDenoiserAttentionPolicy,
    NativeDenoiserQkPreparationPolicy, NativeDenoiserRmsNormPolicy, NativePaddedBlackboxDenoiser,
    NativePortableDenoiser, NativeQwenSynchronizationPolicy, NativeVaeExecutionPolicy,
    RetainingBooguVaeStageSource, StreamingBooguPipeline, VaeDecoderMemoryPolicy,
    artifacts::{
        BooguArtifactInventory, BooguReleaseIdentity, BooguResidentLoadMemoryPolicy,
        BooguStorageProfile, DirectoryStageShardReader, VerifiedArtifactDirectory,
        VerifiedBurnpackQwenStageSource, VerifiedDirectoryVaeStageSource,
        artifact_bundle_id_matches_selection, canonical_published_bundle,
        load_resident_denoiser_from_directory_with_memory_policy,
        load_resident_denoiser_from_directory_with_policies,
        validate_canonical_release_artifact_digest,
    },
    boogu_model_descriptor, boogu_processor_config,
};
use burn_flux_vae::{
    AutoencoderKl, AutoencoderKlConfig, DecoderGroupNormPolicy, FluxVaeArtifactFloatPolicy,
    VerifiedBurnpackFluxVaeStageSource,
};
use burn_image::{
    ArtifactSource, CancellationToken, ColorSpace, Dimensions, DirectoryArtifactShardReader,
    EditRequest, GenerationOptions, ImageModel, ImageOutput, ImageRequest, ImageRuntime,
    InputImage, IntegrityPolicy, PixelBuffer, PixelFormat, ProgressEvent, Prompt, RunId,
    RuntimeConfig, RuntimeError,
};
use burn_qwen3_vl::{
    EmbeddingRowChunk, Qwen3VlArtifactFloatPolicy, Qwen3VlConfig, Qwen3VlDecoderLayer,
    Qwen3VlImageProcessor, Qwen3VlImageProcessorConfig, Qwen3VlProcessor, Qwen3VlStage,
    Qwen3VlStageSource, Qwen3VlStreamingPlan, Qwen3VlTokenizer, Qwen3VlVisionBlock,
    Qwen3VlVisionPatchMerger, Qwen3VlVisionPrelude, RetainingQwen3VlStageSource,
    RetainingSynchronizationPolicy, RowChunkSpec, StreamingQwen3Vl,
    VerifiedBurnpackQwen3VlStageSource, tokenizer::HfTokenizer,
};

use crate::{
    BooguFactoryContext, BooguRuntime, BooguRuntimeFactory, BooguRuntimeJob, ImageJobId,
    ImageRunnerEvent, WgpuExecutionKind, boogu_bundle_id, boogu_profile_slug,
    boogu_source_bundle_id, default_boogu_storage_profile, default_native_boogu_model_base_url,
    native_boogu_source_requires_canonical_digest, prepare_runtime_job,
    resolve_native_boogu_artifact_directory,
};

pub use burn_boogu::deployment::NativeBooguResidencyPolicy;
use burn_boogu::deployment::{
    native_kernel_policy_label, native_qwen_query_chunk_size, native_resident_allocation_policy,
    native_runtime_policy_label, qualified_native_execution_policy,
};

type NativeBackend = burn_wgpu::Wgpu<f32, i32, u32>;
type StandaloneNativeQwenSource =
    VerifiedBurnpackQwenStageSource<NativeBackend, DirectoryStageShardReader>;
type ComponentNativeQwenSource =
    VerifiedBurnpackQwen3VlStageSource<NativeBackend, DirectoryArtifactShardReader>;
type StandaloneNativeVaeSource = VerifiedDirectoryVaeStageSource<NativeBackend>;
type ComponentNativeVaeSource = FluxVaeStageSourceAdapter<
    VerifiedBurnpackFluxVaeStageSource<NativeBackend, DirectoryArtifactShardReader>,
>;
type NativeRuntimeLoad = Result<(NativeBooguRuntime, BooguFactoryContext), RuntimeError>;

enum NativeQwenSource {
    Standalone(Box<StandaloneNativeQwenSource>),
    Component(Box<ComponentNativeQwenSource>),
}

impl Qwen3VlStageSource<NativeBackend> for NativeQwenSource {
    type Error = BooguError;

    fn load_embedding_rows(
        &mut self,
        spec: &RowChunkSpec,
    ) -> Result<EmbeddingRowChunk<NativeBackend>, Self::Error> {
        match self {
            Self::Standalone(source) => source.load_embedding_rows(spec),
            Self::Component(source) => source
                .load_embedding_rows(spec)
                .map_err(component_qwen_error),
        }
    }

    fn load_vision_prelude(&mut self) -> Result<Qwen3VlVisionPrelude<NativeBackend>, Self::Error> {
        match self {
            Self::Standalone(source) => source.load_vision_prelude(),
            Self::Component(source) => source.load_vision_prelude().map_err(component_qwen_error),
        }
    }

    fn load_vision_block(
        &mut self,
        index: usize,
    ) -> Result<Qwen3VlVisionBlock<NativeBackend>, Self::Error> {
        match self {
            Self::Standalone(source) => source.load_vision_block(index),
            Self::Component(source) => source
                .load_vision_block(index)
                .map_err(component_qwen_error),
        }
    }

    fn load_vision_deepstack_merger(
        &mut self,
        index: usize,
    ) -> Result<Qwen3VlVisionPatchMerger<NativeBackend>, Self::Error> {
        match self {
            Self::Standalone(source) => source.load_vision_deepstack_merger(index),
            Self::Component(source) => source
                .load_vision_deepstack_merger(index)
                .map_err(component_qwen_error),
        }
    }

    fn load_vision_final_merger(
        &mut self,
    ) -> Result<Qwen3VlVisionPatchMerger<NativeBackend>, Self::Error> {
        match self {
            Self::Standalone(source) => source.load_vision_final_merger(),
            Self::Component(source) => source
                .load_vision_final_merger()
                .map_err(component_qwen_error),
        }
    }

    fn load_text_block(
        &mut self,
        index: usize,
    ) -> Result<Qwen3VlDecoderLayer<NativeBackend>, Self::Error> {
        match self {
            Self::Standalone(source) => source.load_text_block(index),
            Self::Component(source) => source.load_text_block(index).map_err(component_qwen_error),
        }
    }

    fn load_text_final_norm(&mut self) -> Result<RmsNorm<NativeBackend>, Self::Error> {
        match self {
            Self::Standalone(source) => source.load_text_final_norm(),
            Self::Component(source) => source.load_text_final_norm().map_err(component_qwen_error),
        }
    }

    fn synchronize(&mut self) -> Result<(), Self::Error> {
        match self {
            Self::Standalone(source) => source.synchronize(),
            Self::Component(source) => source.synchronize().map_err(component_qwen_error),
        }
    }
}

fn component_qwen_error(error: impl std::fmt::Display) -> BooguError {
    BooguError::Artifact(error.to_string())
}

enum NativeVaeSource {
    Standalone(Box<StandaloneNativeVaeSource>),
    Component(Box<ComponentNativeVaeSource>),
}

impl BooguVaeStageSource<NativeBackend> for NativeVaeSource {
    fn load_encoder(&mut self) -> Result<AutoencoderKl<NativeBackend>, BooguError> {
        match self {
            Self::Standalone(source) => source.load_encoder(),
            Self::Component(source) => source.load_encoder(),
        }
    }

    fn load_decoder(&mut self) -> Result<AutoencoderKl<NativeBackend>, BooguError> {
        match self {
            Self::Standalone(source) => source.load_decoder(),
            Self::Component(source) => source.load_decoder(),
        }
    }
}

const MAX_EVENTS_PER_POLL: usize = 64;

const GIB: u64 = 1024 * 1024 * 1024;
// Decimal GB is intentional and matches the browser contract. This is stricter than 32 GiB.
const NATIVE_LOW_VRAM_DEVICE_BUDGET_BYTES: u64 = 32_000_000_000;
// This is a conservative static allowance, not a measured peak. It covers activations, kernel
// workspaces, the shared Bevy device, and allocator slack; release qualification must still
// continuously measure total process VRAM.
const NATIVE_LOW_VRAM_NON_WEIGHT_RESERVE_BYTES: u64 = 10_000_000_000;
// Turbo's steady state is smaller after reference-refiner pruning, but construction currently
// materializes the complete denoiser before releasing those dormant modules. Admission therefore
// covers that larger authenticated initialization phase rather than only steady-state residency.
const NATIVE_LOW_VRAM_TURBO_LIVE_WEIGHT_BYTES: u64 = 20_585_112_576;
const NATIVE_LOW_VRAM_EDIT_LIVE_WEIGHT_BYTES: u64 = 20_971_005_440;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NativeLowVramResourcePlan {
    inventory_audited_peak_weight_bytes: u64,
    non_weight_reserve_bytes: u64,
    planned_device_bytes: u64,
    device_budget_bytes: u64,
}

fn native_low_vram_resource_plan(
    variant: BooguVariant,
    profile: BooguStorageProfile,
) -> Result<NativeLowVramResourcePlan, RuntimeError> {
    native_low_vram_resource_plan_with_budget(variant, profile, NATIVE_LOW_VRAM_DEVICE_BUDGET_BYTES)
}

fn native_low_vram_resource_plan_with_budget(
    variant: BooguVariant,
    profile: BooguStorageProfile,
    device_budget_bytes: u64,
) -> Result<NativeLowVramResourcePlan, RuntimeError> {
    if profile != BooguStorageProfile::F16QwenVisionF32 {
        return Err(execution_error(
            variant,
            format!(
                "native low-vram currently requires profile=production ({:?}); profile {} is not qualified",
                BooguStorageProfile::F16QwenVisionF32,
                boogu_profile_slug(profile)
            ),
        ));
    }
    let inventory_audited_peak_weight_bytes = match variant {
        BooguVariant::Image01Turbo => NATIVE_LOW_VRAM_TURBO_LIVE_WEIGHT_BYTES,
        BooguVariant::Image01EditTurbo | BooguVariant::Image01EditTurbo1k5 => {
            NATIVE_LOW_VRAM_EDIT_LIVE_WEIGHT_BYTES
        }
    };
    let planned_device_bytes = inventory_audited_peak_weight_bytes
        .checked_add(NATIVE_LOW_VRAM_NON_WEIGHT_RESERVE_BYTES)
        .ok_or_else(|| execution_error(variant, "native low-vram resource plan overflowed"))?;
    if planned_device_bytes >= device_budget_bytes {
        return Err(execution_error(
            variant,
            format!(
                "native low-vram resource plan requires {planned_device_bytes} device bytes (peak initialization/forward weights {inventory_audited_peak_weight_bytes} + non-weight reserve {}), which does not stay below the {device_budget_bytes}-byte decimal-GB budget",
                NATIVE_LOW_VRAM_NON_WEIGHT_RESERVE_BYTES
            ),
        ));
    }
    Ok(NativeLowVramResourcePlan {
        inventory_audited_peak_weight_bytes,
        non_weight_reserve_bytes: NATIVE_LOW_VRAM_NON_WEIGHT_RESERVE_BYTES,
        planned_device_bytes,
        device_budget_bytes,
    })
}

/// Loads one pinned Boogu release from a local sealed directory, then runs requests sequentially on
/// a dedicated native worker thread.
///
/// Builds with `native-autotune` must configure the selected CubeCL policy before Bevy creates or
/// imports its WGPU device. Builds without that opt-in feature use static kernels and skip tuning,
/// which keeps interactive first-use latency predictable. The packaged `burn-image-viewer`
/// performs the feature-enabled setup automatically.
pub struct NativeBooguFactory {
    variant: BooguVariant,
    variants: Vec<BooguVariant>,
    residency: NativeBooguResidencyPolicy,
    autotune: NativeAutotunePolicy,
    loading: Mutex<Option<Receiver<NativeRuntimeLoad>>>,
    progress: Mutex<Option<Receiver<String>>>,
}

impl NativeBooguFactory {
    /// Select one immutable release with the default high-VRAM policy.
    pub fn new(variant: BooguVariant) -> Self {
        Self::with_residency(variant, NativeBooguResidencyPolicy::HighVram)
    }

    /// Select one immutable release and explicit weight-residency policy.
    pub fn with_residency(variant: BooguVariant, residency: NativeBooguResidencyPolicy) -> Self {
        Self::with_residency_and_autotune(variant, residency, NativeAutotunePolicy::Full)
    }

    /// Select one immutable release, weight-residency policy, and native autotune policy.
    pub fn with_residency_and_autotune(
        variant: BooguVariant,
        residency: NativeBooguResidencyPolicy,
        autotune: NativeAutotunePolicy,
    ) -> Self {
        Self {
            variant,
            variants: vec![variant],
            residency,
            autotune,
            loading: Mutex::new(None),
            progress: Mutex::new(None),
        }
    }

    /// Load one canonical release initially while exposing all three canonical releases to the
    /// UI. The returned runtime keeps only one release resident and switches lazily on the next
    /// request, so model selection does not triple startup traffic or GPU residency.
    pub fn with_canonical_model_switching(
        variant: BooguVariant,
        residency: NativeBooguResidencyPolicy,
        autotune: NativeAutotunePolicy,
    ) -> Self {
        let mut factory = Self::with_residency_and_autotune(variant, residency, autotune);
        factory.variants = vec![
            BooguVariant::Image01Turbo,
            BooguVariant::Image01EditTurbo,
            BooguVariant::Image01EditTurbo1k5,
        ];
        factory
    }

    /// Whether this exact variant/residency/profile tuple selects the full-autotune native policy.
    ///
    /// Callers use this before creating the shared Bevy/Burn WGPU device. Construction checks the
    /// same selector and fails closed if full autotune was not configured.
    pub fn requires_full_autotune(
        variant: BooguVariant,
        residency: NativeBooguResidencyPolicy,
        profile: BooguStorageProfile,
    ) -> bool {
        cfg!(feature = "native-autotune")
            && qualified_native_execution_policy(variant, residency, profile)
                .is_some_and(|policy| matches!(policy.autotune, NativeAutotunePolicy::Full))
    }
}

fn isolates_interactive_compute(variants: &[BooguVariant]) -> bool {
    variants.len() > 1
}

impl BooguRuntimeFactory for NativeBooguFactory {
    fn initialization_variant(&self) -> Option<BooguVariant> {
        Some(self.variant)
    }

    fn start(&mut self, context: BooguFactoryContext) -> Result<(), RuntimeError> {
        if context.execution != WgpuExecutionKind::NativeWgpu {
            return Err(execution_error(
                self.variant,
                "the local-directory Boogu factory is native-only",
            ));
        }
        let variant_aware_profiles = self.variants.len() > 1;
        for variant in self.variants.iter().copied() {
            let profile = if variant_aware_profiles {
                default_boogu_storage_profile(variant)
            } else {
                context.settings.storage_profile
            };
            crate::boogu::validate_variant_profile(variant, profile)?;
        }
        if self.residency == NativeBooguResidencyPolicy::LowVram {
            // Validate the exact supported profile and the static <32-GB plan before artifact
            // resolution, thread creation, or any model/device allocation.
            native_low_vram_resource_plan(self.variant, context.settings.storage_profile)?;
        }
        if matches!(context.device, burn_wgpu::WgpuDevice::Cpu) {
            return Err(execution_error(
                self.variant,
                "the native Boogu factory refuses a CPU Burn device",
            ));
        }
        #[cfg(feature = "native-autotune")]
        if qualified_native_execution_policy(
            self.variant,
            self.residency,
            context.settings.storage_profile,
        )
        .is_some()
        {
            burn_boogu::require_native_autotune_configured(self.autotune)
                .map_err(|error| execution_error(self.variant, error))?;
        }
        if context.settings.integrity != IntegrityPolicy::RequireSha256 {
            return Err(execution_error(
                self.variant,
                "the native Boogu factory requires SHA-256 artifact verification",
            ));
        }
        context
            .settings
            .validate_concrete_cache_policy()
            .map_err(|error| execution_error(self.variant, error))?;
        if self.residency == NativeBooguResidencyPolicy::LowVram {
            let plan =
                native_low_vram_resource_plan(self.variant, context.settings.storage_profile)?;
            bevy::log::info!(
                "starting supported native low-VRAM execution: Qwen and VAE stream per request; the variant-required denoiser remains resident; planned {:.3} GiB is below the {:.1} GiB device target",
                plan.planned_device_bytes as f64 / GIB as f64,
                plan.device_budget_bytes as f64 / GIB as f64,
            );
        }
        let mut loading = self
            .loading
            .lock()
            .map_err(|_| execution_error(self.variant, "native factory mutex was poisoned"))?;
        if loading.is_some() {
            return Err(execution_error(
                self.variant,
                "the native factory was started more than once",
            ));
        }

        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let (progress_tx, progress_rx) = mpsc::channel();
        let variant = self.variant;
        let residency = self.residency;
        let autotune = self.autotune;
        let isolate_interactive_compute = isolates_interactive_compute(&self.variants);
        thread::Builder::new()
            .name("burn-image-boogu-loader".into())
            .spawn(move || {
                let mut context = context;
                if isolate_interactive_compute {
                    let _ = progress_tx.send(
                        "Model setup: creating an isolated GPU compute queue so window rendering remains responsive"
                            .into(),
                    );
                    context.device = match burn_boogu::require_native_wgpu_device() {
                        Ok(device) => device,
                        Err(error) => {
                            let _ = ready_tx.send(Err(execution_error(variant, error)));
                            return;
                        }
                    };
                }
                let cleanup_device = context.device.clone();
                let retained_context = context.clone();
                let mut result = load_native_runtime_for_interactive(
                    context,
                    variant,
                    residency,
                    autotune,
                    |message| {
                        let _ = progress_tx.send(message);
                    },
                )
                .map(|runtime| (runtime, retained_context));
                if result.is_err()
                    && residency == NativeBooguResidencyPolicy::LowVram
                    && let Err(cleanup_error) =
                        cleanup_native_device_allocations(&cleanup_device, variant)
                    && let Err(load_error) = result
                {
                    result = Err(execution_error(
                        variant,
                        format!(
                            "{load_error}; backend cleanup after failed low-VRAM model loading also failed: {cleanup_error}"
                        ),
                    ));
                }
                let _ = ready_tx.send(result);
            })
            .map_err(|error| {
                execution_error(
                    self.variant,
                    format!("could not spawn native Boogu loader: {error}"),
                )
            })?;
        *loading = Some(ready_rx);
        *self
            .progress
            .lock()
            .map_err(|_| execution_error(self.variant, "native progress mutex was poisoned"))? =
            Some(progress_rx);
        Ok(())
    }

    fn poll(&mut self) -> Result<Option<Box<dyn BooguRuntime>>, RuntimeError> {
        let mut loading = self
            .loading
            .lock()
            .map_err(|_| execution_error(self.variant, "native factory mutex was poisoned"))?;
        let Some(receiver) = loading.as_ref() else {
            return Err(execution_error(
                self.variant,
                "native factory poll called before start",
            ));
        };
        match receiver.try_recv() {
            Ok(Ok((runtime, context))) => {
                *loading = None;
                if self.variants.len() == 1 {
                    return Ok(Some(Box::new(runtime)));
                }
                Ok(Some(Box::new(NativeSwitchableBooguRuntime::spawn(
                    runtime,
                    context,
                    self.variants.clone(),
                    self.residency,
                    self.autotune,
                    true,
                )?)))
            }
            Ok(Err(error)) => {
                *loading = None;
                Err(error)
            }
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => {
                *loading = None;
                Err(execution_error(
                    self.variant,
                    "native Boogu loader exited without reporting a result",
                ))
            }
        }
    }

    fn take_initialization_progress(&mut self) -> Option<String> {
        let mut progress = self.progress.lock().ok()?;
        let receiver = progress.as_ref()?;
        let mut latest = None;
        loop {
            match receiver.try_recv() {
                Ok(message) => latest = Some(message),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    *progress = None;
                    break;
                }
            }
        }
        latest
    }
}

enum WorkerCommand {
    Infer { job: Box<BooguRuntimeJob> },
    Shutdown,
}

/// Initialized native worker returned by [`NativeBooguFactory`].
pub struct NativeBooguRuntime {
    variant: BooguVariant,
    commands: mpsc::Sender<WorkerCommand>,
    events: Mutex<Receiver<ImageRunnerEvent>>,
    busy: Arc<AtomicBool>,
    cancellation: CancellationToken,
    active: Option<(ImageJobId, CancellationToken)>,
    worker: Option<JoinHandle<()>>,
}

impl NativeBooguRuntime {
    fn spawn<M>(variant: BooguVariant, runtime: ImageRuntime<M>) -> Result<Self, RuntimeError>
    where
        M: ImageModel<Output = ImageOutput> + Send + 'static,
    {
        let cancellation = runtime.cancellation_token();
        let (command_tx, command_rx) = mpsc::channel();
        let (event_tx, event_rx) = mpsc::channel();
        let busy = Arc::new(AtomicBool::new(false));
        let worker_busy = Arc::clone(&busy);
        let worker = thread::Builder::new()
            .name("burn-image-boogu-inference".into())
            .spawn(move || run_worker(variant, runtime, command_rx, event_tx, worker_busy))
            .map_err(|error| {
                execution_error(
                    variant,
                    format!("could not spawn native Boogu inference worker: {error}"),
                )
            })?;
        Ok(Self {
            variant,
            commands: command_tx,
            events: Mutex::new(event_rx),
            busy,
            cancellation,
            active: None,
            worker: Some(worker),
        })
    }

    /// Retire an idle resident runtime before another release is allocated on the same device.
    ///
    /// Model switching already runs on its own worker, so waiting here cannot block Bevy's event
    /// loop. Joining is essential: the model and all retained GPU module handles are owned by the
    /// inference worker and are not guaranteed to drop merely because this outer handle is gone.
    fn retire_before_model_switch(mut self) -> Result<(), RuntimeError> {
        if let Some((_, token)) = self.active.take() {
            token.cancel();
        }
        let _ = self.commands.send(WorkerCommand::Shutdown);
        if let Some(worker) = self.worker.take()
            && worker.join().is_err()
        {
            return Err(execution_error(
                self.variant,
                "the previous native inference worker panicked while retiring its resident model",
            ));
        }
        self.busy.store(false, Ordering::Release);
        Ok(())
    }
}

impl BooguRuntime for NativeBooguRuntime {
    fn variants(&self) -> Vec<BooguVariant> {
        vec![self.variant]
    }

    fn submit(&mut self, job: BooguRuntimeJob) -> Result<CancellationToken, RuntimeError> {
        if self.active.is_some()
            || self
                .busy
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
        {
            return Err(execution_error(
                self.variant,
                "the native Boogu worker executes one request at a time",
            ));
        }
        let id = job.id;
        self.cancellation.reset();
        let token = self.cancellation.clone();
        if self
            .commands
            .send(WorkerCommand::Infer { job: Box::new(job) })
            .is_err()
        {
            self.busy.store(false, Ordering::Release);
            return Err(execution_error(
                self.variant,
                "native Boogu inference worker is unavailable",
            ));
        }
        self.active = Some((id, token.clone()));
        Ok(token)
    }

    fn cancel(&mut self, id: ImageJobId) -> Result<(), RuntimeError> {
        if let Some((active_id, token)) = &self.active
            && *active_id == id
        {
            token.cancel();
        }
        Ok(())
    }

    fn poll(&mut self, emit: &mut dyn FnMut(ImageRunnerEvent)) {
        for _ in 0..MAX_EVENTS_PER_POLL {
            let event = match self.events.lock() {
                Ok(receiver) => receiver.try_recv(),
                Err(_) => return,
            };
            let event = match event {
                Ok(event) => event,
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
            };
            let terminal_id = match &event {
                ImageRunnerEvent::Completed { id, .. }
                | ImageRunnerEvent::Failed { id, .. }
                | ImageRunnerEvent::Cancelled { id } => Some(*id),
                ImageRunnerEvent::Progress { .. } => None,
            };
            if terminal_id.is_some_and(|id| {
                self.active
                    .as_ref()
                    .is_some_and(|(active_id, _)| *active_id == id)
            }) {
                self.active = None;
            }
            emit(event);
        }
    }
}

impl Drop for NativeBooguRuntime {
    fn drop(&mut self) {
        if let Some((_, token)) = self.active.take() {
            token.cancel();
        }
        let _ = self.commands.send(WorkerCommand::Shutdown);
        // Joining during a long GPU dispatch would freeze Bevy shutdown. An idle
        // worker is joined; an active one observes cancellation at the next
        // model boundary and its handle is safely detached.
        if !self.busy.load(Ordering::Acquire)
            && let Some(worker) = self.worker.take()
        {
            let _ = worker.join();
        }
    }
}

enum SwitchableWorkerCommand {
    Infer {
        job: Box<BooguRuntimeJob>,
        cancellation: CancellationToken,
    },
    Shutdown,
}

/// Long-lived native UI runtime that keeps one dense release resident and swaps it only when the
/// next submitted request selects a different canonical model.
pub struct NativeSwitchableBooguRuntime {
    variants: Vec<BooguVariant>,
    commands: mpsc::Sender<SwitchableWorkerCommand>,
    events: Mutex<Receiver<ImageRunnerEvent>>,
    active: Option<(ImageJobId, CancellationToken)>,
    worker: Option<JoinHandle<()>>,
}

impl NativeSwitchableBooguRuntime {
    fn spawn(
        runtime: NativeBooguRuntime,
        context: BooguFactoryContext,
        variants: Vec<BooguVariant>,
        residency: NativeBooguResidencyPolicy,
        autotune: NativeAutotunePolicy,
        variant_aware_profiles: bool,
    ) -> Result<Self, RuntimeError> {
        let (command_tx, command_rx) = mpsc::channel();
        let (event_tx, event_rx) = mpsc::channel();
        let worker = thread::Builder::new()
            .name("burn-image-boogu-model-switcher".into())
            .spawn(move || {
                run_switchable_worker(
                    runtime,
                    context,
                    residency,
                    autotune,
                    variant_aware_profiles,
                    command_rx,
                    event_tx,
                )
            })
            .map_err(|error| {
                execution_error(
                    variants
                        .first()
                        .copied()
                        .unwrap_or(BooguVariant::Image01Turbo),
                    format!("could not spawn native Boogu model-switch worker: {error}"),
                )
            })?;
        Ok(Self {
            variants,
            commands: command_tx,
            events: Mutex::new(event_rx),
            active: None,
            worker: Some(worker),
        })
    }
}

impl BooguRuntime for NativeSwitchableBooguRuntime {
    fn variants(&self) -> Vec<BooguVariant> {
        self.variants.clone()
    }

    fn submit(&mut self, job: BooguRuntimeJob) -> Result<CancellationToken, RuntimeError> {
        if self.active.is_some() {
            return Err(execution_error(
                job.variant,
                "the native model-switch runtime executes one request at a time",
            ));
        }
        if !self.variants.contains(&job.variant) {
            return Err(execution_error(
                job.variant,
                "the selected release is not available to this native runtime",
            ));
        }
        let id = job.id;
        let cancellation = CancellationToken::default();
        self.commands
            .send(SwitchableWorkerCommand::Infer {
                job: Box::new(job),
                cancellation: cancellation.clone(),
            })
            .map_err(|_| {
                execution_error(
                    self.variants[0],
                    "native Boogu model-switch worker is unavailable",
                )
            })?;
        self.active = Some((id, cancellation.clone()));
        Ok(cancellation)
    }

    fn cancel(&mut self, id: ImageJobId) -> Result<(), RuntimeError> {
        if let Some((active_id, cancellation)) = &self.active
            && *active_id == id
        {
            cancellation.cancel();
        }
        Ok(())
    }

    fn poll(&mut self, emit: &mut dyn FnMut(ImageRunnerEvent)) {
        for _ in 0..MAX_EVENTS_PER_POLL {
            let event = match self.events.lock() {
                Ok(receiver) => receiver.try_recv(),
                Err(_) => return,
            };
            let event = match event {
                Ok(event) => event,
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
            };
            let terminal_id = match &event {
                ImageRunnerEvent::Completed { id, .. }
                | ImageRunnerEvent::Failed { id, .. }
                | ImageRunnerEvent::Cancelled { id } => Some(*id),
                ImageRunnerEvent::Progress { .. } => None,
            };
            if terminal_id.is_some_and(|id| {
                self.active
                    .as_ref()
                    .is_some_and(|(active_id, _)| *active_id == id)
            }) {
                self.active = None;
            }
            emit(event);
        }
    }
}

impl Drop for NativeSwitchableBooguRuntime {
    fn drop(&mut self) {
        let was_active = self.active.is_some();
        if let Some((_, cancellation)) = self.active.take() {
            cancellation.cancel();
        }
        let _ = self.commands.send(SwitchableWorkerCommand::Shutdown);
        if !was_active && let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn run_switchable_worker(
    initial_runtime: NativeBooguRuntime,
    base_context: BooguFactoryContext,
    residency: NativeBooguResidencyPolicy,
    autotune: NativeAutotunePolicy,
    variant_aware_profiles: bool,
    commands: Receiver<SwitchableWorkerCommand>,
    events: mpsc::Sender<ImageRunnerEvent>,
) {
    let mut runtime = Some(initial_runtime);
    while let Ok(command) = commands.recv() {
        let SwitchableWorkerCommand::Infer {
            mut job,
            cancellation,
        } = command
        else {
            break;
        };
        let id = job.id;
        if cancellation.is_cancelled() {
            let _ = events.send(ImageRunnerEvent::Cancelled { id });
            continue;
        }

        let target = job.variant;
        let target_context =
            match native_switch_context(&base_context, target, variant_aware_profiles) {
                Ok(context) => context,
                Err(error) => {
                    let _ = events.send(ImageRunnerEvent::Failed { id, error });
                    continue;
                }
            };
        job.runtime_config = target_context.settings.runtime_config(target);
        let current = runtime.as_ref().map(|runtime| runtime.variant);
        if current != Some(target) {
            let run_id = RunId(id.0);
            let setup_steps = native_setup_step_count(residency);
            let _ = events.send(ImageRunnerEvent::Progress {
                id,
                event: ProgressEvent::StageStarted {
                    run_id,
                    stage: "model-switch".into(),
                    total_steps: None,
                },
            });
            let switch_started = Instant::now();
            let _ = events.send(model_switch_progress_event(
                id,
                run_id,
                setup_steps,
                "Unloading the previous model from GPU memory".into(),
            ));
            let retirement = runtime
                .take()
                .map_or(Ok(()), NativeBooguRuntime::retire_before_model_switch);
            // The old worker owns every retained module. Only clean the allocator after it has
            // joined and dropped those handles, so old and new release residency cannot overlap.
            let cleanup = cleanup_native_device_allocations(&target_context.device, target);
            if let Err(error) = retirement {
                let _ = events.send(ImageRunnerEvent::Failed { id, error });
                continue;
            }
            if let Err(error) = cleanup {
                let _ = events.send(ImageRunnerEvent::Failed { id, error });
                continue;
            }
            bevy::log::info!(
                "switching resident native Boogu model to {:?}; only this release will remain on the GPU",
                target
            );
            let progress_events = events.clone();
            let loaded = load_native_runtime_for_interactive(
                target_context,
                target,
                residency,
                autotune,
                move |message| {
                    bevy::log::info!("model switch: {message}");
                    let _ = progress_events.send(model_switch_progress_event(
                        id,
                        run_id,
                        setup_steps,
                        message,
                    ));
                },
            );
            let loaded = match loaded {
                Ok(runtime) => runtime,
                Err(error) => {
                    let _ = events.send(ImageRunnerEvent::Failed { id, error });
                    continue;
                }
            };
            runtime = Some(loaded);
            let _ = events.send(ImageRunnerEvent::Progress {
                id,
                event: ProgressEvent::StageCompleted {
                    run_id,
                    stage: "model-switch".into(),
                    elapsed_micros: u64::try_from(switch_started.elapsed().as_micros())
                        .unwrap_or(u64::MAX),
                },
            });
        }

        let Some(active_runtime) = runtime.as_mut() else {
            let _ = events.send(ImageRunnerEvent::Failed {
                id,
                error: execution_error(target, "native model switch did not produce a runtime"),
            });
            continue;
        };
        let inner_token = match active_runtime.submit(*job) {
            Ok(token) => token,
            Err(error) => {
                let _ = events.send(ImageRunnerEvent::Failed { id, error });
                continue;
            }
        };
        let mut terminal = false;
        while !terminal {
            if cancellation.is_cancelled() {
                inner_token.cancel();
                let _ = active_runtime.cancel(id);
            }
            active_runtime.poll(&mut |event| {
                terminal = matches!(
                    event,
                    ImageRunnerEvent::Completed { .. }
                        | ImageRunnerEvent::Failed { .. }
                        | ImageRunnerEvent::Cancelled { .. }
                );
                let _ = events.send(event);
            });
            if !terminal {
                thread::sleep(Duration::from_millis(2));
            }
        }
    }
}

fn model_switch_progress_event(
    id: ImageJobId,
    run_id: RunId,
    setup_steps: u32,
    message: String,
) -> ImageRunnerEvent {
    ImageRunnerEvent::Progress {
        id,
        event: ProgressEvent::StageStarted {
            run_id,
            stage: format!(
                "{}{setup_steps}:{message}",
                crate::MODEL_SWITCH_PROGRESS_STAGE_PREFIX
            ),
            total_steps: None,
        },
    }
}

fn native_switch_context(
    base: &BooguFactoryContext,
    variant: BooguVariant,
    variant_aware_profile: bool,
) -> Result<BooguFactoryContext, RuntimeError> {
    let mut context = base.clone();
    if variant_aware_profile {
        context.settings.storage_profile = default_boogu_storage_profile(variant);
    }
    if matches!(
        context.settings.artifact_source,
        ArtifactSource::Remote { .. }
    ) {
        context.settings.artifact_source = ArtifactSource::Remote {
            base_url: default_native_boogu_model_base_url(
                variant,
                context.settings.storage_profile,
            )
            .map_err(|error| execution_error(variant, error))?,
        };
    }
    Ok(context)
}

fn run_worker<M>(
    variant: BooguVariant,
    mut runtime: ImageRuntime<M>,
    commands: Receiver<WorkerCommand>,
    events: mpsc::Sender<ImageRunnerEvent>,
    busy: Arc<AtomicBool>,
) where
    M: ImageModel<Output = ImageOutput>,
{
    while let Ok(command) = commands.recv() {
        let WorkerCommand::Infer { job } = command else {
            break;
        };
        if let Err(error) = validate_job(variant, runtime.config(), &job) {
            let _ = events.send(ImageRunnerEvent::Failed { id: job.id, error });
            busy.store(false, Ordering::Release);
            continue;
        }

        let id = job.id;
        let progress_events = events.clone();
        runtime.set_observer(Arc::new(move |event: &ProgressEvent| {
            let _ = progress_events.send(ImageRunnerEvent::Progress {
                id,
                event: event.clone(),
            });
        }));
        let event = match runtime.infer(&job.request) {
            Ok(output) => ImageRunnerEvent::Completed { id, output },
            Err(RuntimeError::Cancelled) => ImageRunnerEvent::Cancelled { id },
            Err(error) => ImageRunnerEvent::Failed { id, error },
        };
        let _ = events.send(event);
        busy.store(false, Ordering::Release);
    }
    busy.store(false, Ordering::Release);
}

fn validate_job(
    variant: BooguVariant,
    runtime_config: &RuntimeConfig,
    job: &BooguRuntimeJob,
) -> Result<(), RuntimeError> {
    let canonical = BooguReleaseIdentity::canonical(variant);
    if job.variant != variant
        || job.resolved.variant != variant
        || job.release.model_revision != canonical.model_revision
        || job.release.upstream_source_revision != canonical.upstream_source_revision
        || &job.runtime_config != runtime_config
    {
        return Err(execution_error(
            variant,
            "dispatched job does not match the loaded immutable release/runtime configuration",
        ));
    }
    Ok(())
}

fn cleanup_native_device_allocations(
    device: &burn_wgpu::WgpuDevice,
    variant: BooguVariant,
) -> Result<(), RuntimeError> {
    <NativeBackend as Backend>::sync(device).map_err(|error| {
        execution_error(
            variant,
            format!("backend synchronization before native model cleanup failed: {error}"),
        )
    })?;
    <NativeBackend as Backend>::memory_cleanup(device);
    <NativeBackend as Backend>::sync(device).map_err(|error| {
        execution_error(
            variant,
            format!("backend synchronization after native model cleanup failed: {error}"),
        )
    })?;
    Ok(())
}

fn load_native_runtime_for_interactive(
    context: BooguFactoryContext,
    variant: BooguVariant,
    residency: NativeBooguResidencyPolicy,
    autotune: NativeAutotunePolicy,
    report_progress: impl Fn(String),
) -> Result<NativeBooguRuntime, RuntimeError> {
    let settings = context.settings.clone();
    let mut runtime = load_native_runtime(context, variant, residency, autotune, &report_progress)?;
    if cfg!(feature = "native-autotune")
        && variant == BooguVariant::Image01EditTurbo
        && residency == NativeBooguResidencyPolicy::HighVram
        && autotune == NativeAutotunePolicy::Balanced
    {
        report_progress(
            "Optimizing resident Edit 1K GPU kernels before the first interactive request".into(),
        );
        match warm_native_edit_runtime(&mut runtime, &settings) {
            Ok(()) => report_progress("Edit 1K GPU warmup complete; runtime ready".into()),
            Err(error) => {
                // Prewarming is a latency optimization, never a new availability gate. The
                // worker is still usable after a model-level warmup error and the real request
                // remains fully validated on its own input.
                bevy::log::warn!("Edit 1K GPU warmup did not complete: {error}");
                report_progress("Edit 1K runtime ready; optional GPU warmup was skipped".into());
            }
        }
    }
    Ok(runtime)
}

fn warm_native_edit_runtime(
    runtime: &mut NativeBooguRuntime,
    settings: &crate::BooguAdapterSettings,
) -> Result<(), RuntimeError> {
    const EDGE: u32 = 1_024;
    const WARMUP_TIMEOUT: Duration = Duration::from_secs(180);
    let dimensions = Dimensions::new(EDGE, EDGE)
        .map_err(|error| execution_error(BooguVariant::Image01EditTurbo, error))?;
    // A spatially and chromatically varying input avoids the degenerate all-constant image path
    // while still being deterministic and entirely in-memory.
    let mut pixels = Vec::with_capacity((EDGE as usize) * (EDGE as usize) * 4);
    for y in 0..EDGE {
        for x in 0..EDGE {
            let checker = if ((x / 128) + (y / 128)) % 2 == 0 {
                32
            } else {
                0
            };
            pixels.extend_from_slice(&[
                ((x * 191 / (EDGE - 1)) + checker).min(255) as u8,
                ((y * 191 / (EDGE - 1)) + checker).min(255) as u8,
                (((x + y) * 95 / (EDGE - 1)) + 32).min(255) as u8,
                255,
            ]);
        }
    }
    let source = InputImage::Pixels(
        PixelBuffer::new(dimensions, PixelFormat::Rgba8, ColorSpace::Srgb, pixels)
            .map_err(|error| execution_error(BooguVariant::Image01EditTurbo, error))?,
    );
    let request = ImageRequest::Edit(EditRequest {
        source,
        instruction: Prompt::new("Preserve the reference image while applying a subtle warm tone")
            .map_err(|error| execution_error(BooguVariant::Image01EditTurbo, error))?,
        negative_prompt: None,
        mask: None,
        strength: None,
        options: GenerationOptions {
            dimensions: Some(dimensions),
            steps: Some(4),
            guidance_scale: Some(1.0),
            seed: Some(0),
            batch_size: 1,
        },
    });
    let id = ImageJobId(0);
    let job = prepare_runtime_job(id, BooguVariant::Image01EditTurbo, request, settings)?;
    runtime.submit(job)?;
    let started = Instant::now();
    let mut outcome = None;
    while outcome.is_none() && started.elapsed() < WARMUP_TIMEOUT {
        runtime.poll(&mut |event| match event {
            ImageRunnerEvent::Completed { output, .. } => {
                outcome = Some(output.validate().map_err(RuntimeError::from));
            }
            ImageRunnerEvent::Failed { error, .. } => outcome = Some(Err(error)),
            ImageRunnerEvent::Cancelled { .. } => outcome = Some(Err(RuntimeError::Cancelled)),
            ImageRunnerEvent::Progress { .. } => {}
        });
        if outcome.is_none() {
            thread::sleep(Duration::from_millis(2));
        }
    }
    let result = outcome.unwrap_or_else(|| {
        Err(execution_error(
            BooguVariant::Image01EditTurbo,
            "interactive Edit 1K GPU warmup exceeded 180 seconds",
        ))
    });
    while runtime.busy.load(Ordering::Acquire) {
        thread::yield_now();
    }
    result
}

const fn native_setup_step_count(residency: NativeBooguResidencyPolicy) -> u32 {
    match residency {
        NativeBooguResidencyPolicy::HighVram => 5,
        NativeBooguResidencyPolicy::LowVram => 4,
    }
}

fn load_native_runtime(
    context: BooguFactoryContext,
    variant: BooguVariant,
    residency: NativeBooguResidencyPolicy,
    autotune: NativeAutotunePolicy,
    report_progress: impl Fn(String),
) -> Result<NativeBooguRuntime, RuntimeError> {
    let setup_steps = native_setup_step_count(residency);
    report_progress(format!(
        "Model setup 0/{setup_steps}: resolving and verifying sealed artifacts"
    ));
    bevy::log::info!(
        "initializing native Boogu runtime with residency policy {}",
        residency.label()
    );
    let native_policy =
        qualified_native_execution_policy(variant, residency, context.settings.storage_profile);
    let qwen_query_chunk_size = native_policy
        .map(|policy| native_qwen_query_chunk_size(variant, residency, autotune, policy));
    let artifact_directories = resolve_native_boogu_artifact_directory(
        variant,
        context.settings.storage_profile,
        &context.settings.artifact_source,
        |message| {
            bevy::log::info!("{message}");
            report_progress(format!("Model setup: {message}"));
        },
    )
    .map_err(|error| execution_error(variant, error))?;
    let root = artifact_directories.pipeline_root().to_owned();
    if !root.is_dir() {
        return Err(execution_error(
            variant,
            format!("artifact directory does not exist: {}", root.display()),
        ));
    }

    let directory =
        VerifiedArtifactDirectory::open(&root).map_err(|error| execution_error(variant, error))?;
    let qwen_directory = VerifiedArtifactDirectory::open(artifact_directories.qwen_root())
        .map_err(|error| execution_error(variant, error))?;
    let vae_directory = VerifiedArtifactDirectory::open(artifact_directories.vae_root())
        .map_err(|error| execution_error(variant, error))?;
    let manifest = directory.manifest();
    let descriptor = boogu_model_descriptor(variant);
    let expected_bundle = boogu_bundle_id(variant, context.settings.storage_profile);
    let source_bundle = boogu_source_bundle_id(variant, context.settings.storage_profile);
    let expected_profile = boogu_profile_slug(context.settings.storage_profile);
    let require_canonical_digest = native_boogu_source_requires_canonical_digest(
        variant,
        context.settings.storage_profile,
        &context.settings.artifact_source,
    )
    .map_err(|error| execution_error(variant, error))?;
    let bundle_matches = if require_canonical_digest {
        manifest.bundle.as_str() == expected_bundle
    } else {
        artifact_bundle_id_matches_selection(
            variant,
            context.settings.storage_profile,
            manifest.bundle.as_str(),
        )
    };
    if !bundle_matches
        || manifest.profile.as_str() != expected_profile
        || manifest.model != descriptor.id
        || manifest.model_revision != descriptor.revision
    {
        return Err(execution_error(
            variant,
            format!(
                "sealed manifest identity does not match the selected Boogu release: expected bundle={expected_bundle} (an explicit local conversion source may use {source_bundle}), profile={expected_profile}, model={}, revision={}; found bundle={}, profile={}, model={}, revision={}",
                descriptor.id,
                descriptor.revision,
                manifest.bundle,
                manifest.profile,
                manifest.model,
                manifest.model_revision
            ),
        ));
    }
    let content_digest = manifest
        .content_digest
        .ok_or_else(|| execution_error(variant, "sealed manifest is missing its content digest"))?;
    if require_canonical_digest {
        validate_canonical_release_artifact_digest(
            variant,
            context.settings.storage_profile,
            content_digest,
        )
        .map_err(|error| execution_error(variant, error))?;
    }
    let identity = BooguReleaseIdentity::canonical(variant);
    identity
        .validate()
        .map_err(|error| execution_error(variant, error))?;

    report_progress(format!(
        "Model setup 1/{setup_steps}: loading configuration and tokenizer"
    ));

    let qwen_config = Qwen3VlConfig::from_json(
        &qwen_directory
            .read_text("metadata/source/mllm/config.json")
            .map_err(|error| execution_error(variant, error))?,
    )
    .map_err(|error| execution_error(variant, error))?;
    let mut vae_config = AutoencoderKlConfig::from_diffusers_json(
        &vae_directory
            .read_text("metadata/source/vae/config.json")
            .map_err(|error| execution_error(variant, error))?,
    )
    .map_err(|error| execution_error(variant, error))?;
    if let Some(policy) = native_policy {
        vae_config.attention_query_chunk_size = policy.vae_attention_query_chunk_size;
    }
    let denoiser_config = BooguConfig::default();
    let inventory = BooguArtifactInventory::new(&qwen_config, &denoiser_config, &vae_config)
        .map_err(|error| execution_error(variant, error))?;
    let device_scope = if matches!(context.device, burn_wgpu::WgpuDevice::DefaultDevice) {
        "isolated-interactive-device"
    } else {
        "shared-bevy-device"
    };
    let device = context.device;
    let profile = context.settings.storage_profile;
    let low_vram_plan = if residency == NativeBooguResidencyPolicy::LowVram {
        let plan = native_low_vram_resource_plan(variant, profile)?;
        report_progress(format!(
            "Model setup 2/{setup_steps}: low-VRAM plan accepted ({:.3} GiB planned; {:.3} GiB inventory-audited peak weights; <32 decimal GB target)",
            plan.planned_device_bytes as f64 / GIB as f64,
            plan.inventory_audited_peak_weight_bytes as f64 / GIB as f64,
        ));
        Some(plan)
    } else {
        None
    };
    let vae_policy = if let Some(policy) = native_policy {
        match policy.vae_execution {
            NativeVaeExecutionPolicy::PreserveF16StorageF32GroupNorm => {
                // The qualified native numerical and synchronized performance gates preserve the
                // authenticated mixed-F16 VAE. Adapting it to F32 selects a different runtime.
                burn_boogu::artifacts::BooguFloatLoadPolicy::Preserve
            }
        }
    } else {
        context.settings.vae_float_load_policy()
    };
    let denoiser_policy = context.settings.denoiser_float_load_policy();
    let qwen_quantized_policy = context.settings.qwen_quantized_load_policy();
    let denoiser_quantized_policy = context.settings.denoiser_quantized_load_policy();
    let execution_dtypes =
        BooguRuntimeDTypes::from_artifact_policies(profile, vae_policy, denoiser_policy);

    let (qwen_source, qwen_plan, vae) = if artifact_directories.is_standalone() {
        let qwen_source = VerifiedBurnpackQwenStageSource::<
            NativeBackend,
            DirectoryStageShardReader,
        >::from_directory_auto(
            &identity,
            &root,
            inventory.clone(),
            qwen_config.clone(),
            profile,
            device.clone(),
        )
        .map_err(|error| execution_error(variant, error))?
        .with_float_load_policy(context.settings.qwen_float_load_policy())
        .with_quantized_load_policy(qwen_quantized_policy);
        let qwen_plan = qwen_source.plan().clone();
        let vae = VerifiedDirectoryVaeStageSource::<NativeBackend>::new(
            &identity,
            &root,
            inventory.clone(),
            vae_config,
            profile,
            vae_policy,
            device.clone(),
        )
        .map_err(|error| execution_error(variant, error))?;
        (
            NativeQwenSource::Standalone(Box::new(qwen_source)),
            qwen_plan,
            NativeVaeSource::Standalone(Box::new(vae)),
        )
    } else {
        let qwen_source = VerifiedBurnpackQwen3VlStageSource::<
            NativeBackend,
            DirectoryArtifactShardReader,
        >::from_directory(
            artifact_directories.qwen_root(),
            qwen_config.clone(),
            device.clone(),
        )
        .map_err(|error| execution_error(variant, error))?
        .with_float_policy(Qwen3VlArtifactFloatPolicy::Preserve);
        let qwen_plan = qwen_source.contract().plan().clone();
        let component_vae_policy = match vae_policy {
            burn_boogu::artifacts::BooguFloatLoadPolicy::Preserve => {
                FluxVaeArtifactFloatPolicy::Preserve
            }
            burn_boogu::artifacts::BooguFloatLoadPolicy::AdaptToF32 => {
                FluxVaeArtifactFloatPolicy::AdaptToF32
            }
            burn_boogu::artifacts::BooguFloatLoadPolicy::PackedF16WeightsF32Auxiliaries => {
                FluxVaeArtifactFloatPolicy::PackedF16WeightsF32Auxiliaries
            }
            burn_boogu::artifacts::BooguFloatLoadPolicy::PackedQ4sWeightsF32Auxiliaries => {
                FluxVaeArtifactFloatPolicy::PackedF16WeightsF32Auxiliaries
            }
        };
        let vae = VerifiedBurnpackFluxVaeStageSource::<
            NativeBackend,
            DirectoryArtifactShardReader,
        >::from_directory(
            artifact_directories.vae_root(), vae_config, device.clone()
        )
        .map_err(|error| execution_error(variant, error))?
        .with_float_policy(component_vae_policy);
        (
            NativeQwenSource::Component(Box::new(qwen_source)),
            qwen_plan,
            NativeVaeSource::Component(Box::new(FluxVaeStageSourceAdapter::new(vae))),
        )
    };

    let tokenizer_bytes = qwen_directory
        .read_file("metadata/source/mllm/tokenizer.json")
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
        Qwen3VlImageProcessorConfig::from_json(
            &qwen_directory
                .read_text("metadata/source/mllm/preprocessor_config.json")
                .map_err(|error| execution_error(variant, error))?,
        )
        .map_err(|error| execution_error(variant, error))?,
    )
    .map_err(|error| execution_error(variant, error))?;
    let runtime_config = context.settings.runtime_config(variant);
    let resident_allocation_policy = native_resident_allocation_policy(profile);
    let backend_policy = match (residency, native_policy) {
        (NativeBooguResidencyPolicy::HighVram, Some(policy)) => native_runtime_policy_label(
            policy,
            autotune,
            qwen_query_chunk_size.expect("qualified native policy supplies a Qwen chunk"),
        ),
        (NativeBooguResidencyPolicy::LowVram, Some(policy)) => format!(
            "{}/denoiser-per-physical-shard-upload-flush-allocator-cleanup/qwen-direct-release-dtype-embedding-upload/qwen-streamed-per-stage-allocator-cleanup/vae-exact-transient-allocation-pre-tail-cleanup/phase-boundary-allocator-cleanup/qualified-native-kernels={}",
            residency.label(),
            native_kernel_policy_label(variant, policy, autotune)
        ),
        (NativeBooguResidencyPolicy::HighVram, None)
            if profile == BooguStorageProfile::Q4sBlockUpTo128F32 =>
        {
            format!(
                "{}/q4-packed-resident/vae-exact-striped-tail-stage-cleanup/phase-boundary-allocator-cleanup",
                residency.label()
            )
        }
        _ => residency.label().to_owned(),
    };
    let resource_policy = low_vram_plan.map_or_else(String::new, |plan| {
        format!(
            "/resource-plan-peak-weights={}-reserve={}-planned={}-budget={}",
            plan.inventory_audited_peak_weight_bytes,
            plan.non_weight_reserve_bytes,
            plan.planned_device_bytes,
            plan.device_budget_bytes,
        )
    });
    let metadata = BooguRuntimeMetadata {
        numeric_format: numeric_format(profile),
        backend: format!(
            "burn-wgpu-native/{device_scope}/{}/{backend_policy}{resource_policy}",
            residency
                .weight_traffic_contract(canonical_published_bundle(variant, profile).is_some())
        ),
        artifact_content_digest: Some(content_digest),
        artifacts_verified: true,
        execution_dtypes,
        default_seed: 0,
    };

    let runtime = match residency {
        NativeBooguResidencyPolicy::HighVram => {
            report_progress("Model setup 2/5: loading denoiser weights to GPU".into());
            bevy::log::info!(
                "eagerly loading required Qwen/VAE weights and one resident Boogu denoiser before native runtime readiness"
            );
            let expected_reference_refiners = denoiser_config.num_refiner_layers;
            let (mut denoiser, report) =
                load_resident_denoiser_from_directory_with_policies::<NativeBackend>(
                    &identity,
                    &root,
                    inventory,
                    denoiser_config,
                    profile,
                    denoiser_policy,
                    denoiser_quantized_policy,
                    &device,
                )
                .map_err(|error| execution_error(variant, error))?;
            bevy::log::info!(
                "loaded resident Boogu denoiser: {} tensors in {} shards",
                report.tensors,
                report.shards
            );
            if variant == BooguVariant::Image01Turbo {
                if denoiser.ref_image_refiner.len() != expected_reference_refiners {
                    return Err(execution_error(
                        variant,
                        format!(
                            "loaded denoiser has {} reference refiners; the sealed configuration requires {expected_reference_refiners}",
                            denoiser.ref_image_refiner.len()
                        ),
                    ));
                }
                // Generate can never consume an edit reference. Keep every executable Turbo
                // module resident, but drop these authenticated edit-only handles before Qwen/VAE
                // preload so their packed buffers cannot inflate steady-state or VAE peak VRAM.
                denoiser.ref_image_refiner.clear();
            }
            // Retained stages never need a device barrier before a module handle is dropped.
            // The eager preload below performs one explicit final barrier after all authenticated
            // host uploads. Forward then submits the dense GPU graph without a CPU wait per layer;
            // the normal component/output barriers remain the synchronization points.
            let synchronization = native_policy
                .map(|policy| match policy.qwen_synchronization {
                    NativeQwenSynchronizationPolicy::PerStage => {
                        RetainingSynchronizationPolicy::PerStage
                    }
                    NativeQwenSynchronizationPolicy::DeferredToStageBoundary => {
                        RetainingSynchronizationPolicy::Deferred
                    }
                })
                .unwrap_or(RetainingSynchronizationPolicy::Deferred);
            let mut qwen_source = RetainingQwen3VlStageSource::new(qwen_source)
                .with_synchronization_policy(synchronization);
            report_progress("Model setup 3/5: loading Qwen stages to GPU".into());
            let retained_qwen_stages = preload_retained_qwen(
                &mut qwen_source,
                &qwen_plan,
                variant != BooguVariant::Image01Turbo,
            )
            .map_err(|error| execution_error(variant, error))?;
            bevy::log::info!(
                "native Qwen preload complete: {retained_qwen_stages} semantic stages resident on the inference WGPU device"
            );
            let mut qwen = StreamingQwen3Vl::new(qwen_plan, qwen_source);
            let mut vae = RetainingBooguVaeStageSource::new(vae);
            report_progress("Model setup 4/5: loading VAE stages and finalizing".into());
            if variant != BooguVariant::Image01Turbo {
                drop(
                    vae.load_encoder()
                        .map_err(|error| execution_error(variant, error))?,
                );
            }
            drop(
                vae.load_decoder()
                    .map_err(|error| execution_error(variant, error))?,
            );
            <NativeBackend as Backend>::sync(&device).map_err(|error| {
                execution_error(
                    variant,
                    format!("device sync after eager VAE residency load failed: {error}"),
                )
            })?;
            bevy::log::info!(
                "native VAE preload complete: {} required halves resident on the inference WGPU device",
                vae.cached_stage_count()
            );
            if resident_allocation_policy.phase_boundary_cleanup {
                cleanup_native_device_allocations(&device, variant)?;
            }
            if let Some(policy) = native_policy {
                qwen.set_query_chunk_size(
                    qwen_query_chunk_size.expect("qualified native policy supplies a Qwen chunk"),
                );
                denoiser.set_attention_query_chunk_size(policy.denoiser_query_chunk_size);
                let denoiser_rms_norm_policy = match policy.denoiser_rms_norm {
                    NativeDenoiserRmsNormPolicy::StrictF32 => DenoiserRmsNormPolicy::StrictF32,
                };
                let denoiser = match policy.denoiser_attention {
                    NativeDenoiserAttentionPolicy::PaddedBlackbox => {
                        NativePaddedBlackboxDenoiser::new(denoiser)
                            .with_partition_configuration(
                                policy.blackbox_num_planes,
                                policy.blackbox_seq_kv_tiles,
                                policy.blackbox_seq_q_tiles,
                            )
                            .with_rms_norm_policy(denoiser_rms_norm_policy)
                    }
                };
                let denoiser = match policy.denoiser_qk_preparation {
                    NativeDenoiserQkPreparationPolicy::Composed => denoiser,
                    NativeDenoiserQkPreparationPolicy::BalancedStrictQkNormRope => {
                        denoiser.with_balanced_strict_qk_norm_rope(true)
                    }
                };
                let decoder_group_norm_policy = match policy.vae_execution {
                    NativeVaeExecutionPolicy::PreserveF16StorageF32GroupNorm => {
                        DecoderGroupNormPolicy::F16StorageF32Accum
                    }
                };
                let pipeline =
                    StreamingBooguPipeline::new(variant, qwen_config, qwen, vae, denoiser)
                        .with_decoder_group_norm_policy(decoder_group_norm_policy)
                        .with_decoder_memory_policy(resident_allocation_policy.vae_decoder);
                spawn_native_runtime(
                    variant,
                    runtime_config,
                    pipeline,
                    processor,
                    image_processor,
                    device,
                    metadata,
                    resident_allocation_policy.phase_boundary_cleanup,
                )
            } else {
                let denoiser = NativePortableDenoiser::new(denoiser);
                let pipeline =
                    StreamingBooguPipeline::new(variant, qwen_config, qwen, vae, denoiser)
                        .with_decoder_memory_policy(resident_allocation_policy.vae_decoder);
                spawn_native_runtime(
                    variant,
                    runtime_config,
                    pipeline,
                    processor,
                    image_processor,
                    device,
                    metadata,
                    resident_allocation_policy.phase_boundary_cleanup,
                )
            }
        }
        NativeBooguResidencyPolicy::LowVram => {
            report_progress(
                "Model setup 3/4: loading variant-scoped resident denoiser weights to GPU".into(),
            );
            bevy::log::info!(
                "loading one qualified mixed-F16 denoiser for low-VRAM residency; Qwen and VAE remain verified one-stage-per-request sources"
            );
            let expected_reference_refiners = denoiser_config.num_refiner_layers;
            let (mut denoiser, report) =
                load_resident_denoiser_from_directory_with_memory_policy::<NativeBackend>(
                    &identity,
                    &root,
                    inventory,
                    denoiser_config,
                    profile,
                    denoiser_policy,
                    denoiser_quantized_policy,
                    BooguResidentLoadMemoryPolicy::ReleaseTransientBuffersPerShard,
                    &device,
                )
                .map_err(|error| execution_error(variant, error))?;
            if variant == BooguVariant::Image01Turbo {
                if denoiser.ref_image_refiner.len() != expected_reference_refiners {
                    return Err(execution_error(
                        variant,
                        format!(
                            "loaded denoiser has {} reference refiners; the sealed configuration requires {expected_reference_refiners}",
                            denoiser.ref_image_refiner.len()
                        ),
                    ));
                }
                // Turbo never accepts a reference image. Dropping these authenticated but unused
                // module handles makes the retained set match the inventory-audited Turbo bound.
                denoiser.ref_image_refiner.clear();
            }
            <NativeBackend as Backend>::sync(&device).map_err(|error| {
                execution_error(
                    variant,
                    format!("device sync after low-VRAM denoiser load failed: {error}"),
                )
            })?;
            <NativeBackend as Backend>::memory_cleanup(&device);
            <NativeBackend as Backend>::sync(&device).map_err(|error| {
                execution_error(
                    variant,
                    format!(
                        "device sync after low-VRAM denoiser allocator cleanup failed: {error}"
                    ),
                )
            })?;
            bevy::log::info!(
                "low-VRAM denoiser ready: {} authenticated tensors from {} shards; {} reference-refiner modules retained",
                report.tensors,
                report.shards,
                denoiser.ref_image_refiner.len(),
            );

            let policy = native_policy.ok_or_else(|| {
                execution_error(
                    variant,
                    "native low-vram lost its qualified production execution policy",
                )
            })?;
            let mut qwen = StreamingQwen3Vl::new(qwen_plan, qwen_source)
                .with_release_unused_memory_after_stage(true);
            qwen.set_query_chunk_size(policy.qwen_query_chunk_size);
            denoiser.set_attention_query_chunk_size(policy.denoiser_query_chunk_size);
            let denoiser_rms_norm_policy = match policy.denoiser_rms_norm {
                NativeDenoiserRmsNormPolicy::StrictF32 => DenoiserRmsNormPolicy::StrictF32,
            };
            let denoiser = match policy.denoiser_attention {
                NativeDenoiserAttentionPolicy::PaddedBlackbox => {
                    NativePaddedBlackboxDenoiser::new(denoiser)
                        .with_partition_configuration(
                            policy.blackbox_num_planes,
                            policy.blackbox_seq_kv_tiles,
                            policy.blackbox_seq_q_tiles,
                        )
                        .with_rms_norm_policy(denoiser_rms_norm_policy)
                }
            };
            let denoiser = match policy.denoiser_qk_preparation {
                NativeDenoiserQkPreparationPolicy::Composed => denoiser,
                NativeDenoiserQkPreparationPolicy::BalancedStrictQkNormRope => {
                    denoiser.with_balanced_strict_qk_norm_rope(true)
                }
            };
            let decoder_group_norm_policy = match policy.vae_execution {
                NativeVaeExecutionPolicy::PreserveF16StorageF32GroupNorm => {
                    DecoderGroupNormPolicy::F16StorageF32Accum
                }
            };
            let pipeline = StreamingBooguPipeline::new(variant, qwen_config, qwen, vae, denoiser)
                .with_decoder_group_norm_policy(decoder_group_norm_policy)
                .with_decoder_memory_policy(VaeDecoderMemoryPolicy::ExactTransientWithTailCleanup);
            spawn_native_runtime(
                variant,
                runtime_config,
                pipeline,
                processor,
                image_processor,
                device,
                metadata,
                true,
            )
        }
    }?;
    report_progress(format!(
        "Model setup {setup_steps}/{setup_steps}: runtime ready"
    ));
    Ok(runtime)
}

/// Populate every base-model stage cache before the runtime accepts its first request.
///
/// The retained wrapper keeps the underlying WGPU allocations alive. This function deliberately
/// excludes the untied LM head because Boogu consumes Qwen's base hidden state and never computes
/// vocabulary logits.
fn preload_retained_qwen<S>(
    source: &mut RetainingQwen3VlStageSource<NativeBackend, S>,
    plan: &Qwen3VlStreamingPlan,
    include_vision: bool,
) -> Result<usize, burn_boogu::BooguError>
where
    S: Qwen3VlStageSource<NativeBackend, Error = burn_boogu::BooguError>,
{
    let expected = plan
        .stages
        .iter()
        .filter(|descriptor| qwen_stage_is_required(&descriptor.stage, include_vision))
        .count();
    for descriptor in &plan.stages {
        if !qwen_stage_is_required(&descriptor.stage, include_vision) {
            continue;
        }
        let loaded = match &descriptor.stage {
            Qwen3VlStage::EmbeddingRows { chunk } => {
                let spec = plan.embedding_rows.chunks.get(*chunk).ok_or_else(|| {
                    burn_boogu::BooguError::Artifact(format!(
                        "Qwen preload plan references missing embedding chunk {chunk}"
                    ))
                })?;
                drop(source.load_embedding_rows(spec)?);
                true
            }
            Qwen3VlStage::VisionPrelude => {
                drop(source.load_vision_prelude()?);
                true
            }
            Qwen3VlStage::VisionBlock { index } => {
                drop(source.load_vision_block(*index)?);
                true
            }
            Qwen3VlStage::VisionDeepstackMerger { index, .. } => {
                drop(source.load_vision_deepstack_merger(*index)?);
                true
            }
            Qwen3VlStage::VisionFinalMerger => {
                drop(source.load_vision_final_merger()?);
                true
            }
            Qwen3VlStage::TextBlock { index } => {
                drop(source.load_text_block(*index)?);
                true
            }
            Qwen3VlStage::TextFinalNorm => {
                drop(source.load_text_final_norm()?);
                true
            }
            Qwen3VlStage::LmHeadRows { .. } => false,
        };
        if loaded {
            // Bound host staging memory to one semantic stage. This initialization-only barrier is
            // deliberately outside model forward; cache-hit forward calls remain deferred.
            source.synchronize()?;
            source.synchronize_pending()?;
        }
    }
    let loaded = source.cached_stage_count();
    if loaded != expected {
        return Err(burn_boogu::BooguError::Artifact(format!(
            "Qwen eager-residency contract loaded {loaded} stages, expected {expected}"
        )));
    }
    Ok(loaded)
}

fn qwen_stage_is_required(stage: &Qwen3VlStage, include_vision: bool) -> bool {
    match stage {
        Qwen3VlStage::VisionPrelude
        | Qwen3VlStage::VisionBlock { .. }
        | Qwen3VlStage::VisionDeepstackMerger { .. }
        | Qwen3VlStage::VisionFinalMerger => include_vision,
        Qwen3VlStage::LmHeadRows { .. } => false,
        Qwen3VlStage::EmbeddingRows { .. }
        | Qwen3VlStage::TextBlock { .. }
        | Qwen3VlStage::TextFinalNorm => true,
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_native_runtime<E>(
    variant: BooguVariant,
    runtime_config: RuntimeConfig,
    pipeline: E,
    processor: Qwen3VlProcessor<HfTokenizer>,
    image_processor: Qwen3VlImageProcessor,
    device: burn_wgpu::WgpuDevice,
    metadata: BooguRuntimeMetadata,
    phase_boundary_memory_cleanup: bool,
) -> Result<NativeBooguRuntime, RuntimeError>
where
    E: BooguExecution<NativeBackend> + Send + 'static,
{
    let image_model = BooguImageModel::new(pipeline, processor, image_processor, device, metadata)
        .map_err(|error| execution_error(variant, error))?
        .with_phase_boundary_memory_cleanup(phase_boundary_memory_cleanup);
    let runtime = ImageRuntime::new(runtime_config, image_model)
        .map_err(|error| execution_error(variant, error))?;
    NativeBooguRuntime::spawn(variant, runtime)
}

fn numeric_format(profile: BooguStorageProfile) -> burn_image::NumericFormat {
    match profile {
        BooguStorageProfile::F16 => burn_image::NumericFormat::F16,
        BooguStorageProfile::F16QwenVisionF32 => {
            burn_image::NumericFormat::Other("f16-qwen-vision-f32".into())
        }
        BooguStorageProfile::Q8sBlock32F32 => {
            burn_image::NumericFormat::Other("q8s-block32-f32".into())
        }
        BooguStorageProfile::Q8sBlock32F32QwenVisionF32 => {
            burn_image::NumericFormat::Other("q8s-block32-f32-qwen-vision-f32".into())
        }
        BooguStorageProfile::Q4sBlockUpTo128F32 => {
            burn_image::NumericFormat::Other("q4s-block-up-to128-f32".into())
        }
    }
}

fn execution_error(variant: BooguVariant, message: impl std::fmt::Display) -> RuntimeError {
    RuntimeError::ModelExecution {
        model: boogu_model_descriptor(variant).id,
        message: message.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, path::PathBuf};

    use burn_boogu::{
        BOOGU_1K_NATIVE_POLICY, EDIT_TURBO_1K5_NATIVE_POLICY, artifacts::TensorOwner,
    };
    use burn_flux_vae::AutoencoderKlConfig;
    use burn_image::{ArtifactCachePolicy, ArtifactSource, IntegrityPolicy};

    use super::*;
    use crate::BooguAdapterSettings;

    fn context(execution: WgpuExecutionKind) -> BooguFactoryContext {
        BooguFactoryContext {
            device: burn_wgpu::WgpuDevice::Existing(7),
            execution,
            max_storage_buffer_binding_size: u64::MAX,
            max_buffer_size: u64::MAX,
            settings: BooguAdapterSettings {
                artifact_source: ArtifactSource::LocalDirectory {
                    root: PathBuf::from("missing-test-artifacts"),
                },
                storage_profile: BooguStorageProfile::F16,
                integrity: IntegrityPolicy::RequireSha256,
                cache: ArtifactCachePolicy::UseCached,
            },
            releases: vec![],
        }
    }

    fn released_qwen_config() -> Qwen3VlConfig {
        Qwen3VlConfig::from_json(
            r#"{
              "text_config": {
                "vocab_size":151936,"hidden_size":4096,"intermediate_size":12288,
                "num_hidden_layers":36,"num_attention_heads":32,"num_key_value_heads":8,
                "head_dim":128,"hidden_act":"silu","rms_norm_eps":1e-6,
                "max_position_embeddings":262144,"rope_theta":5000000,
                "rope_scaling":{"mrope_section":[24,20,20],"mrope_interleaved":true,"rope_type":"default"}
              },
              "vision_config": {
                "depth":27,"hidden_size":1152,"intermediate_size":4304,"num_heads":16,
                "patch_size":16,"temporal_patch_size":2,"spatial_merge_size":2,
                "out_hidden_size":4096,"in_channels":3,"num_position_embeddings":2304,
                "deepstack_visual_indexes":[8,16,24],"hidden_act":"gelu_pytorch_tanh",
                "layer_norm_eps":1e-6
              },
              "tie_word_embeddings":false,"image_token_id":151655,"video_token_id":151656,
              "vision_start_token_id":151652,"vision_end_token_id":151653
            }"#,
        )
        .unwrap()
    }

    fn inventory_tensor_bytes(shape: &[usize], element_bytes: u64) -> u64 {
        shape
            .iter()
            .try_fold(element_bytes, |bytes, &dimension| {
                bytes.checked_mul(dimension as u64)
            })
            .expect("released inventory byte size fits u64")
    }

    fn inventory_low_vram_live_weight_bounds() -> (u64, u64) {
        let qwen = released_qwen_config();
        let vae = AutoencoderKlConfig::flux1();
        let inventory = BooguArtifactInventory::new(&qwen, &BooguConfig::default(), &vae).unwrap();

        let mut qwen_stages = BTreeMap::<&str, u64>::new();
        let mut vae_stages = BTreeMap::<&str, u64>::new();
        let mut turbo_denoiser_bytes = 0_u64;
        let mut edit_denoiser_bytes = 0_u64;
        for spec in inventory.tensors() {
            match spec.owner {
                TensorOwner::Qwen3Vl if spec.stage != "qwen-lm-head" => {
                    // The released mixed profile preserves text/embedding F16 but its explicitly
                    // named vision stages execute in F32.
                    if spec.stage != "qwen-embedding" {
                        let element_bytes = if spec.stage.starts_with("qwen-vision-") {
                            4
                        } else {
                            2
                        };
                        *qwen_stages.entry(&spec.stage).or_default() +=
                            inventory_tensor_bytes(&spec.target_shape, element_bytes);
                    }
                }
                TensorOwner::FluxVae => {
                    *vae_stages.entry(&spec.stage).or_default() +=
                        inventory_tensor_bytes(&spec.target_shape, 2);
                }
                TensorOwner::BooguDenoiser => {
                    let bytes = inventory_tensor_bytes(&spec.target_shape, 2);
                    edit_denoiser_bytes += bytes;
                    if !spec.stage.starts_with("boogu-reference-refiner-") {
                        turbo_denoiser_bytes += bytes;
                    }
                }
                TensorOwner::Qwen3Vl => {}
            }
        }

        // Qwen's large embedding is row-routed rather than loaded as the inventory's logical
        // whole table, so account for the largest released row chunk explicitly.
        let qwen_plan = Qwen3VlStreamingPlan::released_f16(&qwen, false).unwrap();
        let max_embedding_chunk_bytes = qwen_plan
            .embedding_rows
            .chunks
            .iter()
            .map(|chunk| {
                let rows = chunk.row_range.end - chunk.row_range.start;
                inventory_tensor_bytes(&[rows, qwen.text_config.hidden_size], 2)
            })
            .max()
            .unwrap();
        let max_inventory_qwen_stage_bytes = qwen_stages.values().copied().max().unwrap();
        let max_qwen_stage_bytes = max_inventory_qwen_stage_bytes.max(max_embedding_chunk_bytes);
        let max_vae_half_bytes = vae_stages.values().copied().max().unwrap();
        let max_streamed_phase_bytes = max_qwen_stage_bytes.max(max_vae_half_bytes);

        assert_eq!(max_embedding_chunk_bytes, 207_446_016);
        assert_eq!(max_inventory_qwen_stage_bytes, 385_892_864);
        assert_eq!(max_qwen_stage_bytes, max_inventory_qwen_stage_bytes);
        assert!(max_vae_half_bytes < max_qwen_stage_bytes);
        assert_eq!(
            edit_denoiser_bytes - turbo_denoiser_bytes,
            715_116_480,
            "Turbo must exclude all reference-refiner weights from its resident set"
        );
        (
            edit_denoiser_bytes.max(turbo_denoiser_bytes + max_streamed_phase_bytes),
            edit_denoiser_bytes + max_streamed_phase_bytes,
        )
    }

    #[test]
    fn native_factory_rejects_browser_execution_correctness() {
        let mut factory = NativeBooguFactory::new(BooguVariant::Image01Turbo);
        assert!(
            factory
                .start(context(WgpuExecutionKind::BrowserWebGpu))
                .unwrap_err()
                .to_string()
                .contains("native-only")
        );
    }

    #[test]
    fn model_switch_setup_messages_are_forwarded_to_the_active_job_correctness() {
        let event = model_switch_progress_event(
            ImageJobId(9),
            RunId(9),
            5,
            "Model setup 3/5: loading Qwen stages to GPU".into(),
        );
        let ImageRunnerEvent::Progress {
            id,
            event:
                ProgressEvent::StageStarted {
                    run_id,
                    stage,
                    total_steps,
                },
        } = event
        else {
            panic!("expected model-switch progress event");
        };
        assert_eq!(id, ImageJobId(9));
        assert_eq!(run_id, RunId(9));
        assert_eq!(total_steps, None);
        assert_eq!(
            stage,
            format!(
                "{}5:Model setup 3/5: loading Qwen stages to GPU",
                crate::MODEL_SWITCH_PROGRESS_STAGE_PREFIX
            )
        );
    }

    #[test]
    fn model_switch_joins_and_cleans_the_old_runtime_before_new_residency_correctness() {
        let source = include_str!("native_boogu.rs");
        let switch_start = source.find("fn run_switchable_worker(").unwrap();
        let switch_end = source[switch_start..]
            .find("fn model_switch_progress_event(")
            .map(|offset| switch_start + offset)
            .unwrap();
        let switch = &source[switch_start..switch_end];
        assert!(
            switch.find("retire_before_model_switch").unwrap()
                < switch.find("cleanup_native_device_allocations").unwrap()
        );
        assert!(
            switch.find("cleanup_native_device_allocations").unwrap()
                < switch.find("load_native_runtime_for_interactive").unwrap()
        );
        let retire_start = source.find("fn retire_before_model_switch").unwrap();
        let retire_end = source[retire_start..]
            .find("impl BooguRuntime for NativeBooguRuntime")
            .map(|offset| retire_start + offset)
            .unwrap();
        assert!(source[retire_start..retire_end].contains("worker.join()"));
    }

    #[test]
    fn native_submit_only_enqueues_and_never_waits_on_the_inference_worker_correctness() {
        let source = include_str!("native_boogu.rs");
        let implementation = source
            .split_once("impl BooguRuntime for NativeBooguRuntime")
            .unwrap()
            .1
            .split_once("impl Drop for NativeBooguRuntime")
            .unwrap()
            .0;
        let submit = implementation
            .split_once("fn submit(")
            .unwrap()
            .1
            .split_once("fn cancel(")
            .unwrap()
            .0;
        assert!(submit.contains("WorkerCommand::Infer"));
        assert!(submit.contains(".send("));
        assert!(submit.contains("self.cancellation.clone()"));
        assert!(!submit.contains("recv("));
        assert!(!submit.contains("sync_channel"));
    }

    #[test]
    fn native_factory_defaults_to_retained_qwen_and_resident_denoiser_correctness() {
        let factory = NativeBooguFactory::new(BooguVariant::Image01Turbo);
        assert_eq!(factory.residency, NativeBooguResidencyPolicy::HighVram);
        assert_eq!(factory.autotune, NativeAutotunePolicy::Full);
        assert_eq!(factory.variants, vec![BooguVariant::Image01Turbo]);
        assert!(factory.residency.is_gpu_resident());
        assert_eq!(factory.residency.label(), "native-high-vram-gpu-resident");
        assert_eq!(
            NativeBooguResidencyPolicy::LowVram.label(),
            "native-low-vram-phase-resident-mixed-f16"
        );
        assert_eq!(
            factory.residency.weight_traffic_contract(true),
            "gpu-resident/zero-forward-host-weight-transfers"
        );
        assert_eq!(
            factory.residency.weight_traffic_contract(false),
            "diagnostic-gpu-resident-unqualified/zero-forward-host-weight-transfers"
        );
        assert_eq!(
            NativeBooguResidencyPolicy::LowVram.weight_traffic_contract(true),
            "phase-resident/qwen+vae-per-request/denoiser-resident-zero-dmd-weight-reloads"
        );
        assert_eq!(
            NativeBooguResidencyPolicy::LowVram.weight_traffic_contract(false),
            "unsupported-low-vram-profile/fail-closed-before-model-load"
        );
        assert!(!NativeBooguResidencyPolicy::LowVram.is_gpu_resident());

        let interactive = NativeBooguFactory::with_residency_and_autotune(
            BooguVariant::Image01EditTurbo,
            NativeBooguResidencyPolicy::HighVram,
            NativeAutotunePolicy::Balanced,
        );
        assert_eq!(interactive.autotune, NativeAutotunePolicy::Balanced);
        let interactive_qwen_chunk = native_qwen_query_chunk_size(
            BooguVariant::Image01EditTurbo,
            interactive.residency,
            interactive.autotune,
            BOOGU_1K_NATIVE_POLICY,
        );
        assert_eq!(interactive_qwen_chunk, 1_024);
        assert_eq!(
            native_runtime_policy_label(
                BOOGU_1K_NATIVE_POLICY,
                interactive.autotune,
                interactive_qwen_chunk,
            ),
            format!(
                "native-high-vram-retained-qwen-deferred-sync/{}/1k-mixed-f16/qwen-q1024/denoiser-padded-blackbox-p4-kv1-q1-q8192-rms-strict-f32-qk-balanced-strict-norm-rope/vae-q4096-f16-storage-f32-accum",
                if cfg!(feature = "native-autotune") {
                    "balanced-autotune"
                } else {
                    "no-autotune-static-kernels"
                }
            )
        );
        assert_eq!(
            native_qwen_query_chunk_size(
                BooguVariant::Image01EditTurbo,
                NativeBooguResidencyPolicy::HighVram,
                NativeAutotunePolicy::Full,
                BOOGU_1K_NATIVE_POLICY,
            ),
            if cfg!(feature = "native-autotune") {
                128
            } else {
                1_024
            }
        );

        let switchable = NativeBooguFactory::with_canonical_model_switching(
            BooguVariant::Image01EditTurbo,
            NativeBooguResidencyPolicy::HighVram,
            NativeAutotunePolicy::Balanced,
        );
        assert!(!isolates_interactive_compute(&interactive.variants));
        assert!(isolates_interactive_compute(&switchable.variants));
        assert_eq!(
            switchable.variants,
            vec![
                BooguVariant::Image01Turbo,
                BooguVariant::Image01EditTurbo,
                BooguVariant::Image01EditTurbo1k5,
            ]
        );
        let mut base = context(WgpuExecutionKind::NativeWgpu);
        base.settings.artifact_source = ArtifactSource::Remote {
            base_url: burn_image::RemoteBaseUrl::new("https://models.example/old").unwrap(),
        };
        let turbo = native_switch_context(&base, BooguVariant::Image01Turbo, true).unwrap();
        assert_eq!(
            turbo.settings.storage_profile,
            BooguStorageProfile::Q4sBlockUpTo128F32
        );
        let ArtifactSource::Remote { base_url } = turbo.settings.artifact_source else {
            panic!("canonical switch must keep a remote source")
        };
        assert_eq!(
            base_url.as_str(),
            "https://aberration.technology/model/boogu-image-0.1-turbo-q4s-block-up-to128-f32"
        );
        let edit = native_switch_context(&base, BooguVariant::Image01EditTurbo, true).unwrap();
        assert_eq!(
            edit.settings.storage_profile,
            BooguStorageProfile::Q4sBlockUpTo128F32
        );
        let ArtifactSource::Remote { base_url } = edit.settings.artifact_source else {
            panic!("canonical switch must keep a remote source")
        };
        assert_eq!(
            base_url.as_str(),
            "https://aberration.technology/model/boogu-image-0.1-edit-turbo-q4s-block-up-to128-f32"
        );
        assert_eq!(
            native_qwen_query_chunk_size(
                BooguVariant::Image01EditTurbo1k5,
                NativeBooguResidencyPolicy::HighVram,
                NativeAutotunePolicy::Balanced,
                EDIT_TURBO_1K5_NATIVE_POLICY,
            ),
            128
        );
    }

    #[test]
    fn native_q4_residency_bounds_transients_without_unloading_weights_correctness() {
        let q4 = native_resident_allocation_policy(BooguStorageProfile::Q4sBlockUpTo128F32);
        assert_eq!(
            q4.vae_decoder,
            VaeDecoderMemoryPolicy::ExactStripedTailWithStageCleanup
        );
        assert!(q4.phase_boundary_cleanup);

        for profile in [
            BooguStorageProfile::F16,
            BooguStorageProfile::F16QwenVisionF32,
            BooguStorageProfile::Q8sBlock32F32,
            BooguStorageProfile::Q8sBlock32F32QwenVisionF32,
        ] {
            let policy = native_resident_allocation_policy(profile);
            assert_eq!(policy.vae_decoder, VaeDecoderMemoryPolicy::BackendDefault);
            assert!(!policy.phase_boundary_cleanup);
        }

        let source = include_str!("native_boogu.rs");
        assert!(source.contains("q4-packed-resident/vae-exact-striped-tail-stage-cleanup"));
        assert!(source.contains("Live tensors and cached module parameters remain referenced"));
        let high_vram = source
            .split_once("NativeBooguResidencyPolicy::HighVram => {")
            .unwrap()
            .1
            .split_once("NativeBooguResidencyPolicy::LowVram => {")
            .unwrap()
            .0;
        assert!(high_vram.contains("denoiser.ref_image_refiner.clear();"));
        assert!(high_vram.contains("cleanup_native_device_allocations(&device, variant)?;"));
    }

    #[test]
    fn native_low_vram_resource_plan_is_inventory_bound_and_below_32_gb_correctness() {
        let (derived_turbo, derived_edit) = inventory_low_vram_live_weight_bounds();
        assert_eq!(derived_turbo, NATIVE_LOW_VRAM_TURBO_LIVE_WEIGHT_BYTES);
        assert_eq!(derived_edit, NATIVE_LOW_VRAM_EDIT_LIVE_WEIGHT_BYTES);

        let turbo = native_low_vram_resource_plan(
            BooguVariant::Image01Turbo,
            BooguStorageProfile::F16QwenVisionF32,
        )
        .unwrap();
        assert_eq!(
            turbo.inventory_audited_peak_weight_bytes,
            NATIVE_LOW_VRAM_TURBO_LIVE_WEIGHT_BYTES
        );
        assert_eq!(turbo.planned_device_bytes, 30_585_112_576);
        assert!(turbo.planned_device_bytes < turbo.device_budget_bytes);

        for variant in [
            BooguVariant::Image01EditTurbo,
            BooguVariant::Image01EditTurbo1k5,
        ] {
            let edit =
                native_low_vram_resource_plan(variant, BooguStorageProfile::F16QwenVisionF32)
                    .unwrap();
            assert_eq!(
                edit.inventory_audited_peak_weight_bytes,
                NATIVE_LOW_VRAM_EDIT_LIVE_WEIGHT_BYTES
            );
            assert_eq!(edit.planned_device_bytes, 30_971_005_440);
            assert!(edit.planned_device_bytes < edit.device_budget_bytes);
        }
    }

    #[test]
    fn native_low_vram_resource_plan_fails_closed_at_budget_correctness() {
        let accepted = native_low_vram_resource_plan(
            BooguVariant::Image01EditTurbo,
            BooguStorageProfile::F16QwenVisionF32,
        )
        .unwrap();
        let error = native_low_vram_resource_plan_with_budget(
            BooguVariant::Image01EditTurbo,
            BooguStorageProfile::F16QwenVisionF32,
            accepted.planned_device_bytes,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("does not stay below"), "{error}");
    }

    #[test]
    fn native_low_vram_rejects_non_production_profile_before_loading_correctness() {
        let mut factory = NativeBooguFactory::with_residency(
            BooguVariant::Image01Turbo,
            NativeBooguResidencyPolicy::LowVram,
        );
        let error = factory
            .start(context(WgpuExecutionKind::NativeWgpu))
            .unwrap_err()
            .to_string();
        assert!(error.contains("requires profile=production"), "{error}");
        assert!(error.contains("profile f16 is not qualified"), "{error}");
    }

    #[test]
    fn native_qwen_eager_residency_selects_only_request_graph_stages_correctness() {
        assert!(qwen_stage_is_required(
            &Qwen3VlStage::EmbeddingRows { chunk: 0 },
            false
        ));
        assert!(qwen_stage_is_required(
            &Qwen3VlStage::TextBlock { index: 0 },
            false
        ));
        assert!(!qwen_stage_is_required(
            &Qwen3VlStage::VisionBlock { index: 0 },
            false
        ));
        assert!(qwen_stage_is_required(
            &Qwen3VlStage::VisionBlock { index: 0 },
            true
        ));
        assert!(!qwen_stage_is_required(
            &Qwen3VlStage::LmHeadRows { chunk: 0 },
            true
        ));
    }

    #[test]
    fn qualified_policy_covers_native_high_and_low_vram_mixed_f16_correctness() {
        for residency in [
            NativeBooguResidencyPolicy::HighVram,
            NativeBooguResidencyPolicy::LowVram,
        ] {
            for variant in [BooguVariant::Image01Turbo, BooguVariant::Image01EditTurbo] {
                assert_eq!(
                    NativeBooguFactory::requires_full_autotune(
                        variant,
                        residency,
                        BooguStorageProfile::F16QwenVisionF32,
                    ),
                    cfg!(feature = "native-autotune")
                );
                assert_eq!(
                    qualified_native_execution_policy(
                        variant,
                        residency,
                        BooguStorageProfile::F16QwenVisionF32,
                    ),
                    Some(BOOGU_1K_NATIVE_POLICY)
                );
            }
            assert_eq!(
                qualified_native_execution_policy(
                    BooguVariant::Image01EditTurbo1k5,
                    residency,
                    BooguStorageProfile::F16QwenVisionF32,
                ),
                Some(EDIT_TURBO_1K5_NATIVE_POLICY)
            );
            assert_eq!(
                NativeBooguFactory::requires_full_autotune(
                    BooguVariant::Image01EditTurbo1k5,
                    residency,
                    BooguStorageProfile::F16QwenVisionF32,
                ),
                cfg!(feature = "native-autotune")
            );
        }

        for variant in [
            BooguVariant::Image01Turbo,
            BooguVariant::Image01EditTurbo,
            BooguVariant::Image01EditTurbo1k5,
        ] {
            for profile in [
                BooguStorageProfile::F16,
                BooguStorageProfile::Q8sBlock32F32,
                BooguStorageProfile::Q8sBlock32F32QwenVisionF32,
            ] {
                for residency in [
                    NativeBooguResidencyPolicy::HighVram,
                    NativeBooguResidencyPolicy::LowVram,
                ] {
                    assert!(!NativeBooguFactory::requires_full_autotune(
                        variant, residency, profile,
                    ));
                    assert_eq!(
                        qualified_native_execution_policy(variant, residency, profile),
                        None
                    );
                }
            }
        }
    }

    #[test]
    fn native_factory_rejects_unverified_policy_correctness() {
        let mut context = context(WgpuExecutionKind::NativeWgpu);
        context.settings.integrity = IntegrityPolicy::SizeOnlyForDevelopment;
        let mut factory = NativeBooguFactory::new(BooguVariant::Image01Turbo);
        assert!(
            factory
                .start(context)
                .unwrap_err()
                .to_string()
                .contains("SHA-256")
        );
    }

    #[test]
    fn native_factory_rejects_unimplemented_cache_policy_correctness() {
        let mut context = context(WgpuExecutionKind::NativeWgpu);
        context.settings.cache = ArtifactCachePolicy::Bypass;
        let mut factory = NativeBooguFactory::new(BooguVariant::Image01Turbo);
        assert!(
            factory
                .start(context)
                .unwrap_err()
                .to_string()
                .contains("UseCached")
        );
    }

    #[test]
    fn native_factory_rejects_unvalidated_1k5_profile_before_loading_correctness() {
        let context = context(WgpuExecutionKind::NativeWgpu);
        let mut factory = NativeBooguFactory::new(BooguVariant::Image01EditTurbo1k5);
        let error = factory.start(context).unwrap_err().to_string();
        assert!(
            error.contains("profile f16 is not validated for this immutable release"),
            "{error}"
        );
    }
}
