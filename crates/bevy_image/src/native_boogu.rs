//! Concrete native worker for a sealed Boogu Burnpack bundle.
//!
//! The default high-VRAM policy verifies, decodes, and uploads every required Qwen/VAE stage
//! before the runtime becomes ready. It then retains their shared WGPU handles with one verified
//! denoiser for every request and all four DMD steps. No model-weight filesystem read, host decode,
//! or host-to-device upload occurs in that runtime's forward hot path.
//!
//! The explicit layer-streamed diagnostic policy is intentionally different: it rereads Qwen and
//! VAE stages per request and denoiser stages per DMD step. It requires an explicit local artifact
//! override and reports that traffic policy in provenance. This is deliberately separate from the
//! browser adapter: the directory sources are synchronous filesystem readers, while a browser
//! needs asynchronous range/CDN orchestration behind the same public
//! [`crate::BooguRuntimeFactory`] seam.

use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, SyncSender, TryRecvError},
    },
    thread::{self, JoinHandle},
};

use burn::{nn::RmsNorm, prelude::Backend};
use burn_boogu::{
    BOOGU_1K_NATIVE_POLICY, BooguConfig, BooguError, BooguExecution, BooguImageModel,
    BooguRuntimeDTypes, BooguRuntimeMetadata, BooguVaeStageSource, BooguVariant,
    DenoiserRmsNormPolicy, EDIT_TURBO_1K5_NATIVE_POLICY, FluxVaeStageSourceAdapter,
    NativeAutotunePolicy, NativeDenoiserAttentionPolicy, NativeDenoiserQkPreparationPolicy,
    NativeDenoiserRmsNormPolicy, NativeHighVramPolicy, NativePaddedBlackboxDenoiser,
    NativePortableDenoiser, NativeQwenSynchronizationPolicy, NativeVaeExecutionPolicy,
    RetainingBooguVaeStageSource, StreamingBooguDenoiser, StreamingBooguPipeline,
    artifacts::{
        BooguArtifactInventory, BooguReleaseIdentity, BooguStorageProfile,
        DirectoryStageShardReader, VerifiedArtifactDirectory, VerifiedBurnpackQwenStageSource,
        VerifiedBurnpackStageSource, VerifiedDirectoryVaeStageSource,
        artifact_bundle_id_is_compatible, load_resident_denoiser_from_directory_with_policies,
        validate_canonical_release_artifact_digest,
    },
    boogu_model_descriptor, boogu_processor_config,
};
use burn_flux_vae::{
    AutoencoderKl, AutoencoderKlConfig, DecoderGroupNormPolicy, FluxVaeArtifactFloatPolicy,
    VerifiedBurnpackFluxVaeStageSource,
};
use burn_image::{
    ArtifactSource, CancellationToken, DirectoryArtifactShardReader, ImageModel, ImageOutput,
    ImageRuntime, IntegrityPolicy, ProgressEvent, RuntimeConfig, RuntimeError,
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
    ImageRunnerEvent, WgpuExecutionKind, boogu_bundle_id, boogu_legacy_bundle_id,
    boogu_profile_slug, native_boogu_source_requires_canonical_digest,
    resolve_native_boogu_artifact_directory,
};

type NativeBackend = burn_wgpu::Wgpu<f32, i32, u32>;
type LegacyNativeQwenSource =
    VerifiedBurnpackQwenStageSource<NativeBackend, DirectoryStageShardReader>;
type ComponentNativeQwenSource =
    VerifiedBurnpackQwen3VlStageSource<NativeBackend, DirectoryArtifactShardReader>;
type LegacyNativeVaeSource = VerifiedDirectoryVaeStageSource<NativeBackend>;
type ComponentNativeVaeSource = FluxVaeStageSourceAdapter<
    VerifiedBurnpackFluxVaeStageSource<NativeBackend, DirectoryArtifactShardReader>,
>;

enum NativeQwenSource {
    Legacy(Box<LegacyNativeQwenSource>),
    Component(Box<ComponentNativeQwenSource>),
}

impl Qwen3VlStageSource<NativeBackend> for NativeQwenSource {
    type Error = BooguError;

    fn load_embedding_rows(
        &mut self,
        spec: &RowChunkSpec,
    ) -> Result<EmbeddingRowChunk<NativeBackend>, Self::Error> {
        match self {
            Self::Legacy(source) => source.load_embedding_rows(spec),
            Self::Component(source) => source
                .load_embedding_rows(spec)
                .map_err(component_qwen_error),
        }
    }

    fn load_vision_prelude(&mut self) -> Result<Qwen3VlVisionPrelude<NativeBackend>, Self::Error> {
        match self {
            Self::Legacy(source) => source.load_vision_prelude(),
            Self::Component(source) => source.load_vision_prelude().map_err(component_qwen_error),
        }
    }

    fn load_vision_block(
        &mut self,
        index: usize,
    ) -> Result<Qwen3VlVisionBlock<NativeBackend>, Self::Error> {
        match self {
            Self::Legacy(source) => source.load_vision_block(index),
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
            Self::Legacy(source) => source.load_vision_deepstack_merger(index),
            Self::Component(source) => source
                .load_vision_deepstack_merger(index)
                .map_err(component_qwen_error),
        }
    }

    fn load_vision_final_merger(
        &mut self,
    ) -> Result<Qwen3VlVisionPatchMerger<NativeBackend>, Self::Error> {
        match self {
            Self::Legacy(source) => source.load_vision_final_merger(),
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
            Self::Legacy(source) => source.load_text_block(index),
            Self::Component(source) => source.load_text_block(index).map_err(component_qwen_error),
        }
    }

    fn load_text_final_norm(&mut self) -> Result<RmsNorm<NativeBackend>, Self::Error> {
        match self {
            Self::Legacy(source) => source.load_text_final_norm(),
            Self::Component(source) => source.load_text_final_norm().map_err(component_qwen_error),
        }
    }

    fn synchronize(&mut self) -> Result<(), Self::Error> {
        match self {
            Self::Legacy(source) => source.synchronize(),
            Self::Component(source) => source.synchronize().map_err(component_qwen_error),
        }
    }
}

fn component_qwen_error(error: impl std::fmt::Display) -> BooguError {
    BooguError::Artifact(error.to_string())
}

enum NativeVaeSource {
    Legacy(Box<LegacyNativeVaeSource>),
    Component(Box<ComponentNativeVaeSource>),
}

impl BooguVaeStageSource<NativeBackend> for NativeVaeSource {
    fn load_encoder(&mut self) -> Result<AutoencoderKl<NativeBackend>, BooguError> {
        match self {
            Self::Legacy(source) => source.load_encoder(),
            Self::Component(source) => source.load_encoder(),
        }
    }

    fn load_decoder(&mut self) -> Result<AutoencoderKl<NativeBackend>, BooguError> {
        match self {
            Self::Legacy(source) => source.load_decoder(),
            Self::Component(source) => source.load_decoder(),
        }
    }
}

const MAX_EVENTS_PER_POLL: usize = 64;

/// Native weight-residency policy selected before model construction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NativeBooguResidencyPolicy {
    /// Eagerly load every required Qwen/VAE stage and retain it with the denoiser on the GPU.
    #[default]
    HighVram,
    /// Diagnostic-only host streaming that reloads weights inside model execution.
    ///
    /// This is not a supported production residency policy. The native factory accepts it only
    /// with an explicit local artifact override, preventing accidental CDN/cache selection from
    /// enabling repeated host weight traffic.
    LayerStreamed,
}

impl NativeBooguResidencyPolicy {
    /// Stable label reported in logs and backend provenance.
    pub const fn label(self) -> &'static str {
        match self {
            Self::HighVram => "native-high-vram-gpu-resident-dense",
            Self::LayerStreamed => "native-diagnostic-layer-streamed",
        }
    }

    /// Whether this policy is a supported no-hot-path-weight-transfer execution mode.
    pub const fn is_gpu_resident(self) -> bool {
        matches!(self, Self::HighVram)
    }

    /// Exact steady-state model-weight traffic contract attached to runtime provenance.
    pub const fn weight_traffic_contract(self, production_profile: bool) -> &'static str {
        match self {
            Self::HighVram if production_profile => {
                "gpu-resident-dense/zero-forward-host-weight-transfers"
            }
            Self::HighVram => {
                "diagnostic-gpu-resident-unqualified/zero-forward-host-weight-transfers"
            }
            Self::LayerStreamed => {
                "diagnostic-host-streamed/qwen+vae-per-request/denoiser-per-dmd-step"
            }
        }
    }
}

/// Loads one pinned Boogu release from a local sealed directory, then runs requests sequentially on
/// a dedicated native worker thread.
///
/// Embedders selecting a qualified native high-VRAM mixed-F16 policy must call
/// [`burn_boogu::configure_native_full_autotune`] before Bevy creates or imports its WGPU device.
/// Construction fails closed when that process-global policy was not configured in time. The
/// packaged `burn-image-viewer` binary performs this setup automatically.
pub struct NativeBooguFactory {
    variant: BooguVariant,
    residency: NativeBooguResidencyPolicy,
    loading: Mutex<Option<Receiver<Result<NativeBooguRuntime, RuntimeError>>>>,
    progress: Mutex<Option<Receiver<String>>>,
}

impl NativeBooguFactory {
    /// Select one immutable release with the default high-VRAM policy.
    pub fn new(variant: BooguVariant) -> Self {
        Self::with_residency(variant, NativeBooguResidencyPolicy::HighVram)
    }

    /// Select one immutable release and explicit weight-residency policy.
    pub fn with_residency(variant: BooguVariant, residency: NativeBooguResidencyPolicy) -> Self {
        Self {
            variant,
            residency,
            loading: Mutex::new(None),
            progress: Mutex::new(None),
        }
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
        qualified_native_high_vram_policy(variant, residency, profile)
            .is_some_and(|policy| matches!(policy.autotune, NativeAutotunePolicy::Full))
    }
}

impl BooguRuntimeFactory for NativeBooguFactory {
    fn start(&mut self, context: BooguFactoryContext) -> Result<(), RuntimeError> {
        if context.execution != WgpuExecutionKind::NativeWgpu {
            return Err(execution_error(
                self.variant,
                "the local-directory Boogu factory is native-only",
            ));
        }
        crate::boogu::validate_variant_profile(self.variant, context.settings.storage_profile)?;
        if self.residency == NativeBooguResidencyPolicy::LayerStreamed
            && !matches!(
                &context.settings.artifact_source,
                ArtifactSource::LocalDirectory { .. }
            )
        {
            return Err(execution_error(
                self.variant,
                "native diagnostic layer streaming requires an explicit local artifact directory; the verified CDN/cache path is GPU-resident-only",
            ));
        }
        if self.variant == BooguVariant::Image01EditTurbo1k5
            && self.residency != NativeBooguResidencyPolicy::HighVram
        {
            return Err(execution_error(
                self.variant,
                "Edit-Turbo 1.5K is released only with the parity-gated native high-VRAM policy",
            ));
        }
        if matches!(context.device, burn_wgpu::WgpuDevice::Cpu) {
            return Err(execution_error(
                self.variant,
                "the native Boogu factory refuses a CPU Burn device",
            ));
        }
        if Self::requires_full_autotune(
            self.variant,
            self.residency,
            context.settings.storage_profile,
        ) {
            burn_boogu::require_native_full_autotune_configured()
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
        if self.residency == NativeBooguResidencyPolicy::LayerStreamed {
            bevy::log::warn!(
                "starting opt-in diagnostic native host streaming: Qwen and VAE weights reload per request; denoiser weights reload for every DMD step; this mode is not production-supported"
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
        thread::Builder::new()
            .name("burn-image-boogu-loader".into())
            .spawn(move || {
                let result = load_native_runtime(context, variant, residency, |message| {
                    let _ = progress_tx.send(message);
                });
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
            Ok(Ok(runtime)) => {
                *loading = None;
                Ok(Some(Box::new(runtime)))
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
    Infer {
        job: Box<BooguRuntimeJob>,
        started: SyncSender<Result<CancellationToken, RuntimeError>>,
    },
    Shutdown,
}

/// Initialized native worker returned by [`NativeBooguFactory`].
pub struct NativeBooguRuntime {
    variant: BooguVariant,
    commands: mpsc::Sender<WorkerCommand>,
    events: Mutex<Receiver<ImageRunnerEvent>>,
    busy: Arc<AtomicBool>,
    active: Option<(ImageJobId, CancellationToken)>,
    worker: Option<JoinHandle<()>>,
}

impl NativeBooguRuntime {
    fn spawn<M>(variant: BooguVariant, runtime: ImageRuntime<M>) -> Result<Self, RuntimeError>
    where
        M: ImageModel<Output = ImageOutput> + Send + 'static,
    {
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
            active: None,
            worker: Some(worker),
        })
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
        let (started_tx, started_rx) = mpsc::sync_channel(1);
        if self
            .commands
            .send(WorkerCommand::Infer {
                job: Box::new(job),
                started: started_tx,
            })
            .is_err()
        {
            self.busy.store(false, Ordering::Release);
            return Err(execution_error(
                self.variant,
                "native Boogu inference worker is unavailable",
            ));
        }
        let token = started_rx.recv().map_err(|_| {
            self.busy.store(false, Ordering::Release);
            execution_error(
                self.variant,
                "native Boogu worker exited before accepting the request",
            )
        })??;
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
        let WorkerCommand::Infer { job, started } = command else {
            break;
        };
        if let Err(error) = validate_job(variant, runtime.config(), &job) {
            let _ = started.send(Err(error));
            busy.store(false, Ordering::Release);
            continue;
        }

        let token = runtime.cancellation_token();
        token.reset();
        let id = job.id;
        let progress_events = events.clone();
        runtime.set_observer(Arc::new(move |event: &ProgressEvent| {
            let _ = progress_events.send(ImageRunnerEvent::Progress {
                id,
                event: event.clone(),
            });
        }));
        if started.send(Ok(token)).is_err() {
            busy.store(false, Ordering::Release);
            continue;
        }

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

fn load_native_runtime(
    context: BooguFactoryContext,
    variant: BooguVariant,
    residency: NativeBooguResidencyPolicy,
    report_progress: impl Fn(String),
) -> Result<NativeBooguRuntime, RuntimeError> {
    let setup_steps = match residency {
        NativeBooguResidencyPolicy::HighVram => 5,
        NativeBooguResidencyPolicy::LayerStreamed => 3,
    };
    report_progress(format!(
        "Model setup 0/{setup_steps}: resolving and verifying sealed artifacts"
    ));
    bevy::log::info!(
        "initializing native Boogu runtime with residency policy {}",
        residency.label()
    );
    let native_policy =
        qualified_native_high_vram_policy(variant, residency, context.settings.storage_profile);
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
    let legacy_bundle = boogu_legacy_bundle_id(variant, context.settings.storage_profile);
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
        artifact_bundle_id_is_compatible(
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
                "sealed manifest identity does not match the selected Boogu release: expected bundle={expected_bundle} (explicit sources may also use legacy {legacy_bundle}), profile={expected_profile}, model={}, revision={}; found bundle={}, profile={}, model={}, revision={}",
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
    let device = context.device;
    let profile = context.settings.storage_profile;
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

    let (qwen_source, qwen_plan, vae) = if artifact_directories.is_legacy_monolith() {
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
            NativeQwenSource::Legacy(Box::new(qwen_source)),
            qwen_plan,
            NativeVaeSource::Legacy(Box::new(vae)),
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
    let backend_policy = native_policy
        .map(|policy| policy.provenance_label)
        .unwrap_or_else(|| residency.label());
    let metadata = BooguRuntimeMetadata {
        numeric_format: numeric_format(profile),
        backend: format!(
            "burn-wgpu-native/shared-bevy-device/{}/{backend_policy}",
            residency.weight_traffic_contract(profile == BooguStorageProfile::F16QwenVisionF32)
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
                "native Qwen preload complete: {retained_qwen_stages} semantic stages resident on the shared WGPU device"
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
                "native VAE preload complete: {} required halves resident on the shared WGPU device",
                vae.cached_stage_count()
            );
            if let Some(policy) = native_policy {
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
                let pipeline =
                    StreamingBooguPipeline::new(variant, qwen_config, qwen, vae, denoiser)
                        .with_decoder_group_norm_policy(decoder_group_norm_policy);
                spawn_native_runtime(
                    variant,
                    runtime_config,
                    pipeline,
                    processor,
                    image_processor,
                    device,
                    metadata,
                )
            } else {
                let denoiser = NativePortableDenoiser::new(denoiser);
                let pipeline =
                    StreamingBooguPipeline::new(variant, qwen_config, qwen, vae, denoiser);
                spawn_native_runtime(
                    variant,
                    runtime_config,
                    pipeline,
                    processor,
                    image_processor,
                    device,
                    metadata,
                )
            }
        }
        NativeBooguResidencyPolicy::LayerStreamed => {
            report_progress("Model setup 2/3: preparing diagnostic streamed stage loaders".into());
            let qwen = StreamingQwen3Vl::new(qwen_plan, qwen_source);
            let source = VerifiedBurnpackStageSource::<
                NativeBackend,
                DirectoryStageShardReader,
            >::from_directory(
                &identity,
                &root,
                inventory,
                denoiser_config.clone(),
                profile,
                device.clone(),
            )
            .map_err(|error| execution_error(variant, error))?
            .with_float_load_policy(denoiser_policy)
            .with_quantized_load_policy(denoiser_quantized_policy);
            let denoiser = StreamingBooguDenoiser::new(denoiser_config, source)
                .map_err(|error| execution_error(variant, error))?;
            let pipeline = StreamingBooguPipeline::new(variant, qwen_config, qwen, vae, denoiser);
            spawn_native_runtime(
                variant,
                runtime_config,
                pipeline,
                processor,
                image_processor,
                device,
                metadata,
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

fn qualified_native_high_vram_policy(
    variant: BooguVariant,
    residency: NativeBooguResidencyPolicy,
    profile: BooguStorageProfile,
) -> Option<NativeHighVramPolicy> {
    if residency != NativeBooguResidencyPolicy::HighVram
        || profile != BooguStorageProfile::F16QwenVisionF32
    {
        return None;
    }
    match variant {
        BooguVariant::Image01Turbo | BooguVariant::Image01EditTurbo => Some(BOOGU_1K_NATIVE_POLICY),
        BooguVariant::Image01EditTurbo1k5 => Some(EDIT_TURBO_1K5_NATIVE_POLICY),
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
) -> Result<NativeBooguRuntime, RuntimeError>
where
    E: BooguExecution<NativeBackend> + Send + 'static,
{
    let image_model = BooguImageModel::new(pipeline, processor, image_processor, device, metadata)
        .map_err(|error| execution_error(variant, error))?;
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
    use std::path::PathBuf;

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
    fn native_factory_defaults_to_retained_qwen_and_resident_denoiser_correctness() {
        let factory = NativeBooguFactory::new(BooguVariant::Image01Turbo);
        assert_eq!(factory.residency, NativeBooguResidencyPolicy::HighVram);
        assert!(factory.residency.is_gpu_resident());
        assert_eq!(
            factory.residency.label(),
            "native-high-vram-gpu-resident-dense"
        );
        assert_eq!(
            NativeBooguResidencyPolicy::LayerStreamed.label(),
            "native-diagnostic-layer-streamed"
        );
        assert_eq!(
            factory.residency.weight_traffic_contract(true),
            "gpu-resident-dense/zero-forward-host-weight-transfers"
        );
        assert_eq!(
            factory.residency.weight_traffic_contract(false),
            "diagnostic-gpu-resident-unqualified/zero-forward-host-weight-transfers"
        );
        assert_eq!(
            NativeBooguResidencyPolicy::LayerStreamed.weight_traffic_contract(false),
            "diagnostic-host-streamed/qwen+vae-per-request/denoiser-per-dmd-step"
        );
        assert!(!NativeBooguResidencyPolicy::LayerStreamed.is_gpu_resident());
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
    fn qualified_policy_is_scoped_to_native_high_vram_mixed_f16_correctness() {
        for variant in [BooguVariant::Image01Turbo, BooguVariant::Image01EditTurbo] {
            assert!(NativeBooguFactory::requires_full_autotune(
                variant,
                NativeBooguResidencyPolicy::HighVram,
                BooguStorageProfile::F16QwenVisionF32,
            ));
            assert_eq!(
                qualified_native_high_vram_policy(
                    variant,
                    NativeBooguResidencyPolicy::HighVram,
                    BooguStorageProfile::F16QwenVisionF32,
                ),
                Some(BOOGU_1K_NATIVE_POLICY)
            );
        }
        assert_eq!(
            qualified_native_high_vram_policy(
                BooguVariant::Image01EditTurbo1k5,
                NativeBooguResidencyPolicy::HighVram,
                BooguStorageProfile::F16QwenVisionF32,
            ),
            Some(EDIT_TURBO_1K5_NATIVE_POLICY)
        );
        assert!(NativeBooguFactory::requires_full_autotune(
            BooguVariant::Image01EditTurbo1k5,
            NativeBooguResidencyPolicy::HighVram,
            BooguStorageProfile::F16QwenVisionF32,
        ));

        for variant in [
            BooguVariant::Image01Turbo,
            BooguVariant::Image01EditTurbo,
            BooguVariant::Image01EditTurbo1k5,
        ] {
            assert!(!NativeBooguFactory::requires_full_autotune(
                variant,
                NativeBooguResidencyPolicy::LayerStreamed,
                BooguStorageProfile::F16QwenVisionF32,
            ));
            assert_eq!(
                qualified_native_high_vram_policy(
                    variant,
                    NativeBooguResidencyPolicy::LayerStreamed,
                    BooguStorageProfile::F16QwenVisionF32,
                ),
                None
            );
            for profile in [
                BooguStorageProfile::F16,
                BooguStorageProfile::Q8sBlock32F32,
                BooguStorageProfile::Q8sBlock32F32QwenVisionF32,
            ] {
                assert!(!NativeBooguFactory::requires_full_autotune(
                    variant,
                    NativeBooguResidencyPolicy::HighVram,
                    profile,
                ));
                assert_eq!(
                    qualified_native_high_vram_policy(
                        variant,
                        NativeBooguResidencyPolicy::HighVram,
                        profile,
                    ),
                    None
                );
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
    fn native_diagnostic_streaming_requires_explicit_local_artifacts_correctness() {
        let mut context = context(WgpuExecutionKind::NativeWgpu);
        context.settings.artifact_source = ArtifactSource::Remote {
            base_url: burn_image::RemoteBaseUrl::new(
                "https://aberration.technology/model/diagnostic-test",
            )
            .unwrap(),
        };
        let mut factory = NativeBooguFactory::with_residency(
            BooguVariant::Image01Turbo,
            NativeBooguResidencyPolicy::LayerStreamed,
        );
        let error = factory.start(context).unwrap_err().to_string();
        assert!(
            error.contains("requires an explicit local artifact directory"),
            "{error}"
        );
        assert!(error.contains("GPU-resident-only"), "{error}");
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

    #[test]
    fn native_factory_rejects_unvalidated_1k5_layer_streaming_before_loading_correctness() {
        let mut context = context(WgpuExecutionKind::NativeWgpu);
        context.settings.storage_profile = BooguStorageProfile::F16QwenVisionF32;
        let mut factory = NativeBooguFactory::with_residency(
            BooguVariant::Image01EditTurbo1k5,
            NativeBooguResidencyPolicy::LayerStreamed,
        );
        let error = factory.start(context).unwrap_err().to_string();
        assert!(
            error.contains("parity-gated native high-VRAM policy"),
            "{error}"
        );
    }
}
