//! Concrete native worker for a sealed Boogu Burnpack bundle.
//!
//! The default high-VRAM policy loads every verified Qwen and VAE stage once, retains their shared
//! WGPU handles for later requests, and keeps one verified denoiser resident for all four DMD
//! steps. An explicit layer-streamed
//! diagnostic policy rereads Qwen and denoiser stages. This is deliberately separate from the
//! browser adapter: the directory sources are synchronous filesystem readers,
//! while a browser needs asynchronous range/CDN orchestration behind the same
//! public [`crate::BooguRuntimeFactory`] seam.

use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, SyncSender, TryRecvError},
    },
    thread::{self, JoinHandle},
};

use burn_boogu::{
    BOOGU_1K_NATIVE_POLICY, BooguConfig, BooguExecution, BooguImageModel, BooguRuntimeDTypes,
    BooguRuntimeMetadata, BooguVariant, DenoiserRmsNormPolicy, EDIT_TURBO_1K5_NATIVE_POLICY,
    NativeAutotunePolicy, NativeDenoiserAttentionPolicy, NativeDenoiserQkPreparationPolicy,
    NativeDenoiserRmsNormPolicy, NativeHighVramPolicy, NativePaddedBlackboxDenoiser,
    NativeQwenSynchronizationPolicy, NativeVaeExecutionPolicy, RetainingBooguVaeStageSource,
    StreamingBooguDenoiser, StreamingBooguPipeline,
    artifacts::{
        BooguArtifactInventory, BooguReleaseIdentity, BooguStorageProfile,
        DirectoryStageShardReader, VerifiedArtifactDirectory, VerifiedBurnpackQwenStageSource,
        VerifiedBurnpackStageSource, VerifiedDirectoryVaeStageSource,
        load_resident_denoiser_from_directory_with_policies,
        validate_canonical_release_artifact_digest,
    },
    boogu_model_descriptor, boogu_processor_config,
};
use burn_flux_vae::{AutoencoderKlConfig, DecoderGroupNormPolicy};
use burn_image::{
    CancellationToken, ImageModel, ImageOutput, ImageRuntime, IntegrityPolicy, ProgressEvent,
    RuntimeConfig, RuntimeError,
};
use burn_qwen3_vl::{
    Qwen3VlConfig, Qwen3VlImageProcessor, Qwen3VlImageProcessorConfig, Qwen3VlProcessor,
    Qwen3VlTokenizer, RetainingQwen3VlStageSource, RetainingSynchronizationPolicy,
    StreamingQwen3Vl, tokenizer::HfTokenizer,
};

use crate::{
    BooguFactoryContext, BooguRuntime, BooguRuntimeFactory, BooguRuntimeJob, ImageJobId,
    ImageRunnerEvent, WgpuExecutionKind, boogu_bundle_id, boogu_profile_slug,
    native_boogu_source_requires_canonical_digest, resolve_native_boogu_artifact_directory,
};

type NativeBackend = burn_wgpu::Wgpu<f32, i32, u32>;

const MAX_EVENTS_PER_POLL: usize = 64;

/// Native weight-residency policy selected before model construction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NativeBooguResidencyPolicy {
    /// Load each Qwen/VAE stage once and retain it with the denoiser for later requests.
    #[default]
    HighVram,
    /// Reload Qwen, VAE, and denoiser stages to minimize GPU residency.
    LayerStreamed,
}

impl NativeBooguResidencyPolicy {
    /// Stable label reported in logs and backend provenance.
    pub const fn label(self) -> &'static str {
        match self {
            Self::HighVram => "native-high-vram-retained-qwen",
            Self::LayerStreamed => "native-layer-streamed",
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
        let variant = self.variant;
        let residency = self.residency;
        thread::Builder::new()
            .name("burn-image-boogu-loader".into())
            .spawn(move || {
                let result = load_native_runtime(context, variant, residency);
                let _ = ready_tx.send(result);
            })
            .map_err(|error| {
                execution_error(
                    self.variant,
                    format!("could not spawn native Boogu loader: {error}"),
                )
            })?;
        *loading = Some(ready_rx);
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
) -> Result<NativeBooguRuntime, RuntimeError> {
    bevy::log::info!(
        "initializing native Boogu runtime with residency policy {}",
        residency.label()
    );
    let native_policy =
        qualified_native_high_vram_policy(variant, residency, context.settings.storage_profile);
    let root = resolve_native_boogu_artifact_directory(
        variant,
        context.settings.storage_profile,
        &context.settings.artifact_source,
        |message| bevy::log::info!("{message}"),
    )
    .map_err(|error| execution_error(variant, error))?;
    if !root.is_dir() {
        return Err(execution_error(
            variant,
            format!("artifact directory does not exist: {}", root.display()),
        ));
    }

    let directory =
        VerifiedArtifactDirectory::open(&root).map_err(|error| execution_error(variant, error))?;
    let manifest = directory.manifest();
    let descriptor = boogu_model_descriptor(variant);
    let expected_bundle = boogu_bundle_id(variant, context.settings.storage_profile);
    let expected_profile = boogu_profile_slug(context.settings.storage_profile);
    if manifest.bundle.as_str() != expected_bundle
        || manifest.profile.as_str() != expected_profile
        || manifest.model != descriptor.id
        || manifest.model_revision != descriptor.revision
    {
        return Err(execution_error(
            variant,
            format!(
                "sealed manifest identity does not match the selected Boogu release: expected bundle={expected_bundle}, profile={expected_profile}, model={}, revision={}; found bundle={}, profile={}, model={}, revision={}",
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
    if native_boogu_source_requires_canonical_digest(
        variant,
        context.settings.storage_profile,
        &context.settings.artifact_source,
    )
    .map_err(|error| execution_error(variant, error))?
    {
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

    let qwen_config = Qwen3VlConfig::from_json(
        &directory
            .read_text("metadata/source/mllm/config.json")
            .map_err(|error| execution_error(variant, error))?,
    )
    .map_err(|error| execution_error(variant, error))?;
    let mut vae_config = AutoencoderKlConfig::from_diffusers_json(
        &directory
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

    let tokenizer_bytes = directory
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
            &directory
                .read_text("metadata/source/mllm/preprocessor_config.json")
                .map_err(|error| execution_error(variant, error))?,
        )
        .map_err(|error| execution_error(variant, error))?,
    )
    .map_err(|error| execution_error(variant, error))?;
    let runtime_config = context.settings.runtime_config(variant);
    let backend_policy = if let Some(policy) = native_policy {
        policy.provenance_label
    } else {
        residency.label()
    };
    let metadata = BooguRuntimeMetadata {
        numeric_format: numeric_format(profile),
        backend: format!("burn-wgpu-native/shared-bevy-device/{backend_policy}"),
        artifact_content_digest: Some(content_digest),
        artifacts_verified: true,
        execution_dtypes,
        default_seed: 0,
    };

    match residency {
        NativeBooguResidencyPolicy::HighVram => {
            bevy::log::info!(
                "retaining verified Qwen stages after their first load and loading one resident Boogu denoiser"
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
            let qwen_source = if let Some(policy) = native_policy {
                let synchronization = match policy.qwen_synchronization {
                    NativeQwenSynchronizationPolicy::PerStage => {
                        RetainingSynchronizationPolicy::PerStage
                    }
                    NativeQwenSynchronizationPolicy::DeferredToStageBoundary => {
                        RetainingSynchronizationPolicy::Deferred
                    }
                };
                RetainingQwen3VlStageSource::new(qwen_source)
                    .with_synchronization_policy(synchronization)
            } else {
                RetainingQwen3VlStageSource::new(qwen_source)
            };
            let mut qwen = StreamingQwen3Vl::new(qwen_plan, qwen_source);
            let vae = RetainingBooguVaeStageSource::new(vae);
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
        assert_eq!(factory.residency.label(), "native-high-vram-retained-qwen");
        assert_eq!(
            NativeBooguResidencyPolicy::LayerStreamed.label(),
            "native-layer-streamed"
        );
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
