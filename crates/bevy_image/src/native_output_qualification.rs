//! Opt-in native output qualification through the ordinary Bevy/Boogu runtime path.

use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU8, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use bevy::{app::AppExit, prelude::*};
use burn_boogu::{
    BooguVariant,
    artifacts::{
        BooguStorageProfile, VerifiedArtifactDirectory, canonical_published_bundle,
        validate_canonical_release_artifact_digest, verify_modular_release_artifact_directories,
    },
    boogu_model_descriptor,
};
use burn_image::{
    ColorSpace, Dimensions, EditRequest, GenerateRequest, GenerationOptions, HostImage,
    ImageEncoding, ImageRequest, NumericFormat, PixelFormat, ProgressEvent, Prompt, Sha256Digest,
};
use serde::{Deserialize, Serialize};

use crate::{
    CompleteImageJob, FailImageJob, ImageFrontendSet, ImageJobId, ImageJobPhase, ImageJobRejected,
    ImageJobs, ImageRunnerState, ImageRunnerStatus, ReportImageProgress, SubmitImageJob,
    boogu_model_id, decode_input_image, encode_host_image,
};

/// Decimal byte ceiling used by every supported native low-VRAM release.
pub const NATIVE_OUTPUT_QUALIFICATION_DEVICE_CEILING_BYTES: u64 = 32_000_000_000;

/// Match the interactive reference-image payload ceiling before decoding.
const NATIVE_OUTPUT_QUALIFICATION_MAX_SOURCE_BYTES: u64 = 64 * 1024 * 1024;

const NVIDIA_SMI_SAMPLE_INTERVAL: Duration = Duration::from_millis(100);
const MIN_MATCHED_FRAMEBUFFER_SAMPLES: u64 = 4;
const MIN_NONZERO_FRAMEBUFFER_SAMPLES: u64 = 4;
const FRAMEBUFFER_MIB_BYTES: u64 = 1024 * 1024;

/// Original upload identity retained for an Edit request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOutputSourceIdentity {
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
    pub width: u32,
    pub height: u32,
}

/// Model-relevant request identity shared with the browser output-comparison report.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NativeOutputQualificationRequestIdentity {
    pub variant: String,
    pub task: String,
    pub model: String,
    pub model_revision: String,
    pub prompt: String,
    pub seed: u64,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub guidance_scale: f32,
    pub batch_size: u32,
    pub source: Option<NativeOutputSourceIdentity>,
}

/// Full bounded authentication evidence for one canonical schema-v2 composition.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NativeOutputArtifactVerification {
    pub pipeline_root: PathBuf,
    pub parent_bundle: String,
    pub parent_content_digest: String,
    pub qwen_bundle: String,
    pub qwen_content_digest: String,
    pub vae_bundle: String,
    pub vae_content_digest: String,
    pub verified_files: usize,
    pub verified_bytes: u64,
    pub verified_weight_objects: usize,
    pub verified_tensors: usize,
    pub largest_object_bytes: u64,
    pub dependency_closure_verified: bool,
    pub component_contracts_verified: bool,
    pub reconstructed_inventory_verified: bool,
    pub canonical_parent_digest_verified: bool,
    pub semantic_verification_milliseconds: f64,
}

/// Authenticate one released production composition and both component leaves before model GPU
/// allocation. Every declared compact file and Burnpack object is SHA-256 checked.
pub fn verify_native_output_artifacts(
    pipeline_root: impl AsRef<Path>,
    variant: BooguVariant,
) -> Result<NativeOutputArtifactVerification, String> {
    let started = Instant::now();
    let profile = BooguStorageProfile::F16QwenVisionF32;
    let pipeline_root = fs::canonicalize(pipeline_root.as_ref()).map_err(|error| {
        format!(
            "canonicalize native artifact root {}: {error}",
            pipeline_root.as_ref().display()
        )
    })?;
    let parent = VerifiedArtifactDirectory::open(&pipeline_root)
        .map_err(|error| format!("open native parent manifest: {error}"))?;
    let manifest = parent.manifest();
    let published = canonical_published_bundle(variant, profile)
        .ok_or_else(|| format!("{variant:?}/production has no canonical published bundle"))?;
    if manifest.bundle.as_str() != published.bundle_id {
        return Err(format!(
            "native output qualification requires canonical parent bundle {}, found {}",
            published.bundle_id, manifest.bundle
        ));
    }
    let parent_digest = manifest
        .content_digest
        .ok_or_else(|| "canonical parent manifest has no sealed digest".to_owned())?;
    validate_canonical_release_artifact_digest(variant, profile, parent_digest)
        .map_err(|error| format!("validate canonical parent digest: {error}"))?;

    let sibling_root = pipeline_root.parent().ok_or_else(|| {
        format!(
            "canonical composition has no sibling-bundle parent: {}",
            pipeline_root.display()
        )
    })?;
    let dependency_root = |role: &str| -> Result<PathBuf, String> {
        let dependency = manifest
            .dependencies
            .iter()
            .find(|dependency| dependency.role.as_str() == role)
            .ok_or_else(|| format!("canonical composition omits {role} dependency"))?;
        Ok(sibling_root.join(dependency.bundle.as_str()))
    };
    let semantic = verify_modular_release_artifact_directories(
        &pipeline_root,
        dependency_root("qwen")?,
        dependency_root("vae")?,
    )
    .map_err(|error| format!("authenticate canonical modular closure: {error}"))?;
    if semantic.variant != variant
        || semantic.profile != profile
        || !semantic.dependency_closure_verified
        || !semantic.component_contracts_verified
        || !semantic.reconstructed_inventory_verified
    {
        return Err("modular verifier returned an incomplete or mismatched closure".into());
    }

    Ok(NativeOutputArtifactVerification {
        pipeline_root,
        parent_bundle: published.bundle_id.to_owned(),
        parent_content_digest: parent_digest.to_hex(),
        qwen_bundle: semantic.qwen.bundle,
        qwen_content_digest: semantic.qwen.content_digest.to_hex(),
        vae_bundle: semantic.vae.bundle,
        vae_content_digest: semantic.vae.content_digest.to_hex(),
        verified_files: semantic.verified_files,
        verified_bytes: semantic.verified_bytes,
        verified_weight_objects: semantic.verified_weight_objects,
        verified_tensors: semantic.verified_tensors,
        largest_object_bytes: semantic.largest_object_bytes,
        dependency_closure_verified: semantic.dependency_closure_verified,
        component_contracts_verified: semantic.component_contracts_verified,
        reconstructed_inventory_verified: semantic.reconstructed_inventory_verified,
        canonical_parent_digest_verified: true,
        semantic_verification_milliseconds: started.elapsed().as_secs_f64() * 1_000.0,
    })
}

/// Construct one released exact request. Edit inputs are decoded through the same bounded native
/// upload path as the UI while the report retains the SHA-256 of the original selected file.
pub fn prepare_native_output_qualification_request(
    variant: BooguVariant,
    prompt: String,
    seed: u64,
    width: u32,
    height: u32,
    source_path: Option<PathBuf>,
) -> Result<(ImageRequest, NativeOutputQualificationRequestIdentity), String> {
    let model = boogu_model_id(variant);
    let descriptor = boogu_model_descriptor(variant);
    let dimensions = Dimensions::new(width, height).map_err(|error| error.to_string())?;
    let prompt = Prompt::new(prompt).map_err(|error| error.to_string())?;
    let options = GenerationOptions {
        dimensions: Some(dimensions),
        steps: Some(4),
        guidance_scale: Some(1.0),
        seed: Some(seed),
        batch_size: 1,
    };
    let (request, source) = match (variant, source_path) {
        (BooguVariant::Image01Turbo, None) => (
            ImageRequest::Generate(GenerateRequest {
                prompt: prompt.clone(),
                negative_prompt: None,
                options,
            }),
            None,
        ),
        (BooguVariant::Image01Turbo, Some(_)) => {
            return Err("Turbo output qualification forbids a source image".into());
        }
        (BooguVariant::Image01EditTurbo | BooguVariant::Image01EditTurbo1k5, Some(source_path)) => {
            let source_path = fs::canonicalize(&source_path).map_err(|error| {
                format!(
                    "canonicalize qualification source {}: {error}",
                    source_path.display()
                )
            })?;
            let bytes = read_bounded_qualification_source(&source_path)?;
            let source = decode_input_image(&bytes, None)
                .map_err(|error| format!("decode bounded qualification source image: {error}"))?;
            let source_dimensions = source
                .dimensions()
                .ok_or_else(|| "decoded qualification source omits dimensions".to_owned())?;
            let identity = NativeOutputSourceIdentity {
                path: source_path,
                bytes: bytes.len() as u64,
                sha256: Sha256Digest::calculate(&bytes).to_hex(),
                width: source_dimensions.width(),
                height: source_dimensions.height(),
            };
            (
                ImageRequest::Edit(EditRequest {
                    source,
                    instruction: prompt.clone(),
                    negative_prompt: None,
                    mask: None,
                    strength: None,
                    options,
                }),
                Some(identity),
            )
        }
        (BooguVariant::Image01EditTurbo | BooguVariant::Image01EditTurbo1k5, None) => {
            return Err("Edit output qualification requires an explicit source image path".into());
        }
    };
    descriptor
        .capabilities
        .validate_request(&model, &request)
        .map_err(|error| format!("released request validation failed: {error}"))?;
    Ok((
        request,
        NativeOutputQualificationRequestIdentity {
            variant: variant_slug(variant).into(),
            task: if variant == BooguVariant::Image01Turbo {
                "generate"
            } else {
                "edit"
            }
            .into(),
            model: model.to_string(),
            model_revision: descriptor.revision,
            prompt: prompt.into_inner(),
            seed,
            width,
            height,
            steps: 4,
            guidance_scale: 1.0,
            batch_size: 1,
            source,
        },
    ))
}

fn read_bounded_qualification_source(path: &Path) -> Result<Vec<u8>, String> {
    let file = fs::File::open(path).map_err(|error| {
        format!(
            "open qualification source image {}: {error}",
            path.display()
        )
    })?;
    let metadata = file.metadata().map_err(|error| {
        format!(
            "inspect qualification source image {}: {error}",
            path.display()
        )
    })?;
    if !metadata.is_file() {
        return Err(format!(
            "qualification source image {} is not a regular file",
            path.display()
        ));
    }
    read_bounded_qualification_source_handle(
        file,
        metadata.len(),
        NATIVE_OUTPUT_QUALIFICATION_MAX_SOURCE_BYTES,
        path,
    )
}

fn read_bounded_qualification_source_handle(
    file: fs::File,
    preflight_bytes: u64,
    max_bytes: u64,
    path: &Path,
) -> Result<Vec<u8>, String> {
    if preflight_bytes == 0 {
        return Err(format!(
            "qualification source image {} is empty",
            path.display()
        ));
    }
    if preflight_bytes > max_bytes {
        return Err(format!(
            "qualification source image {} is {preflight_bytes} bytes; limit is {max_bytes} bytes",
            path.display()
        ));
    }

    // Continue reading from the already-inspected handle. The handle survives a path replacement,
    // while MAX+1 detects growth of that same file after the metadata preflight.
    let capacity = usize::try_from(preflight_bytes)
        .map_err(|_| "qualification source size does not fit host memory".to_owned())?;
    let mut bytes = Vec::with_capacity(capacity);
    file.take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|error| {
            format!(
                "read qualification source image {}: {error}",
                path.display()
            )
        })?;
    if bytes.is_empty() {
        return Err(format!(
            "qualification source image {} became empty while reading",
            path.display()
        ));
    }
    if bytes.len() as u64 > max_bytes {
        return Err(format!(
            "qualification source image {} grew beyond the {max_bytes}-byte limit while reading",
            path.display()
        ));
    }
    Ok(bytes)
}

/// Configuration for one exact released native output run.
#[derive(Clone, Debug)]
pub struct NativeOutputQualification {
    pub variant: BooguVariant,
    pub request: ImageRequest,
    pub request_identity: NativeOutputQualificationRequestIdentity,
    pub output_directory: PathBuf,
    pub artifacts: NativeOutputArtifactVerification,
}

/// Adds request automation, production PNG saving, report emission, and PID-scoped VRAM telemetry
/// to an otherwise ordinary native Bevy/Boogu app.
pub struct NativeOutputQualificationPlugin {
    configuration: NativeOutputQualification,
}

impl NativeOutputQualificationPlugin {
    pub fn new(configuration: NativeOutputQualification) -> Self {
        Self { configuration }
    }
}

impl Plugin for NativeOutputQualificationPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(QualificationHost {
            configuration: self.configuration.clone(),
            state: QualificationState::WaitingForRuntime,
            monitor: None,
            progress: Vec::new(),
            started: Instant::now(),
        })
        .add_systems(Startup, start_native_output_telemetry)
        .add_systems(
            Update,
            drive_native_output_qualification.in_set(ImageFrontendSet::Display),
        );
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QualificationState {
    WaitingForRuntime,
    Submitted(ImageJobId),
    Finished,
}

#[derive(Resource)]
struct QualificationHost {
    configuration: NativeOutputQualification,
    state: QualificationState,
    monitor: Option<NvidiaSmiPidMonitor>,
    progress: Vec<ProgressEvent>,
    started: Instant,
}

fn start_native_output_telemetry(
    mut host: ResMut<QualificationHost>,
    mut exits: MessageWriter<AppExit>,
) {
    if let Err(error) = fs::create_dir_all(&host.configuration.output_directory) {
        finish_failure(
            &mut host,
            format!("create qualification output directory: {error}"),
            &mut exits,
        );
        return;
    }
    match NvidiaSmiPidMonitor::start(NVIDIA_SMI_SAMPLE_INTERVAL) {
        Ok(monitor) => host.monitor = Some(monitor),
        Err(error) => finish_failure(&mut host, error, &mut exits),
    }
}

#[allow(clippy::too_many_arguments)]
fn drive_native_output_qualification(
    mut host: ResMut<QualificationHost>,
    runner: Res<ImageRunnerStatus>,
    mut jobs: ResMut<ImageJobs>,
    mut submit: MessageWriter<SubmitImageJob>,
    mut completions: MessageReader<CompleteImageJob>,
    mut failures: MessageReader<FailImageJob>,
    mut rejected: MessageReader<ImageJobRejected>,
    mut progress: MessageReader<ReportImageProgress>,
    mut exits: MessageWriter<AppExit>,
) {
    if host.state == QualificationState::Finished {
        return;
    }
    let active_id = match host.state {
        QualificationState::Submitted(id) => Some(id),
        QualificationState::WaitingForRuntime | QualificationState::Finished => None,
    };
    for update in progress.read() {
        if active_id == Some(update.id) {
            if let Some(phase) = memory_phase_for_progress_event(&update.event)
                && let Some(monitor) = host.monitor.as_ref()
            {
                monitor.set_phase(phase);
            }
            host.progress.push(update.event.clone());
        }
    }
    for failure in failures.read() {
        if active_id == Some(failure.id) {
            finish_failure(
                &mut host,
                format!("native inference failed: {}", failure.error),
                &mut exits,
            );
            return;
        }
    }
    for rejection in rejected.read() {
        if active_id == Some(rejection.id) {
            finish_failure(
                &mut host,
                format!("native request was rejected: {}", rejection.error),
                &mut exits,
            );
            return;
        }
    }
    for completion in completions.read() {
        if active_id != Some(completion.id) {
            continue;
        }
        if !jobs
            .get(completion.id)
            .is_some_and(|job| job.phase == ImageJobPhase::Completed)
        {
            finish_failure(
                &mut host,
                "output did not pass ordinary frontend completion checks".into(),
                &mut exits,
            );
            return;
        }
        match finish_output(&mut host, completion.id, &completion.output) {
            Ok(true) => {
                host.state = QualificationState::Finished;
                exits.write(AppExit::Success);
            }
            Ok(false) => {
                host.state = QualificationState::Finished;
                exits.write(AppExit::error());
            }
            Err(error) => finish_failure(&mut host, error, &mut exits),
        }
        return;
    }

    match (&runner.state, host.state) {
        (ImageRunnerState::Failed { error }, QualificationState::WaitingForRuntime) => {
            finish_failure(
                &mut host,
                format!("native runtime initialization failed: {error}"),
                &mut exits,
            );
        }
        (ImageRunnerState::Ready { capabilities }, QualificationState::WaitingForRuntime) => {
            let model = boogu_model_id(host.configuration.variant);
            if capabilities.descriptor(&model).is_none() {
                finish_failure(
                    &mut host,
                    format!("native runtime became ready without selected model {model}"),
                    &mut exits,
                );
                return;
            }
            let id = jobs.reserve_id();
            if let Some(monitor) = host.monitor.as_ref() {
                monitor.set_phase(NativeOutputMemoryPhase::Processing);
            }
            submit.write(SubmitImageJob {
                id,
                model,
                request: host.configuration.request.clone(),
            });
            host.state = QualificationState::Submitted(id);
        }
        _ => {}
    }
}

fn finish_output(
    host: &mut QualificationHost,
    job_id: ImageJobId,
    output: &burn_image::ImageOutput,
) -> Result<bool, String> {
    if let Some(monitor) = host.monitor.as_ref() {
        monitor.set_phase(NativeOutputMemoryPhase::Finalization);
    }
    validate_output(&host.configuration, output)?;
    let png = encode_host_image(&output.images[0].image, ImageEncoding::Png)
        .map_err(|error| format!("encode output through production PNG path: {error}"))?;
    let stem = output_stem(&host.configuration.request_identity);
    let png_path = host
        .configuration
        .output_directory
        .join(format!("{stem}.png"));
    fs::write(&png_path, &png).map_err(|error| format!("write qualification PNG: {error}"))?;
    let png_path = fs::canonicalize(&png_path)
        .map_err(|error| format!("canonicalize qualification PNG: {error}"))?;
    let device_memory = finish_monitor(host)?;
    let failures = device_memory.failures.clone();
    let passed = device_memory.passed;
    let report = serde_json::json!({
        "schema_version": 1,
        "test": "burn_image_native_low_vram_output",
        "claim": if passed {
            "production native low-VRAM output candidate for a separate cross-runtime numerical gate"
        } else {
            "native output completed but failed the measured low-VRAM qualification"
        },
        "ok": passed,
        "failures": failures,
        "request_identity": &host.configuration.request_identity,
        "selection": {
            "variant": variant_slug(host.configuration.variant),
            "model": boogu_model_id(host.configuration.variant),
            "profile_selector": "production",
            "manifest_profile": "f16-qwen-vision-f32",
            "residency": "native-low-vram-phase-resident-mixed-f16",
            "job_id": job_id.0,
        },
        "artifacts": &host.configuration.artifacts,
        "output": {
            "path": png_path,
            "file_name": format!("{stem}.png"),
            "bytes": png.len(),
            "sha256": Sha256Digest::calculate(&png).to_hex(),
            "width": host.configuration.request_identity.width,
            "height": host.configuration.request_identity.height,
            "encoding": "png",
            "save_path": "bevy_burn_image::encode_host_image(ImageEncoding::Png)",
            "opaque_srgb_rgb8_source": true,
            "seed": output.seed,
            "timings": &output.timings,
            "provenance": &output.provenance,
        },
        "progress_events": &host.progress,
        "device_memory": device_memory,
        "elapsed_milliseconds": host.started.elapsed().as_secs_f64() * 1_000.0,
        "completed_unix_milliseconds": unix_milliseconds(),
    });
    write_report(&host.configuration.output_directory, &stem, &report)?;
    Ok(passed)
}

fn validate_output(
    configuration: &NativeOutputQualification,
    output: &burn_image::ImageOutput,
) -> Result<(), String> {
    output
        .validate()
        .map_err(|error| format!("qualification output validation failed: {error}"))?;
    if output.seed != configuration.request_identity.seed
        || output.images.len() != 1
        || output.images[0].index != 0
    {
        return Err("qualification output does not contain the exact requested image index".into());
    }
    let descriptor = boogu_model_descriptor(configuration.variant);
    let published =
        canonical_published_bundle(configuration.variant, BooguStorageProfile::F16QwenVisionF32)
            .ok_or_else(|| "selected production release has no canonical bundle".to_owned())?;
    let expected_digest = Sha256Digest::from_hex(published.content_digest)
        .map_err(|error| format!("invalid compiled release digest: {error}"))?;
    let provenance = &output.provenance;
    if provenance.model != descriptor.id
        || provenance.model_revision != descriptor.revision
        || provenance.artifact_content_digest != Some(expected_digest)
        || provenance.numeric_format != NumericFormat::Other("f16-qwen-vision-f32".into())
        || !provenance.artifacts_verified
    {
        return Err(format!(
            "qualification output provenance is not canonical production: {provenance:?}"
        ));
    }
    let expected_peak_weights = match configuration.variant {
        BooguVariant::Image01Turbo => 20_585_112_576_u64,
        BooguVariant::Image01EditTurbo | BooguVariant::Image01EditTurbo1k5 => 20_971_005_440,
    };
    let expected_planned = expected_peak_weights + 10_000_000_000;
    let resource_plan = format!(
        "resource-plan-peak-weights={expected_peak_weights}-reserve=10000000000-planned={expected_planned}-budget=32000000000"
    );
    for required in [
        "burn-wgpu-native/shared-bevy-device/",
        "phase-resident/qwen+vae-per-request/denoiser-resident-zero-dmd-weight-reloads",
        "native-low-vram-phase-resident-mixed-f16",
        "denoiser-per-physical-shard-upload-flush-allocator-cleanup/",
        "qwen-direct-release-dtype-embedding-upload/qwen-streamed-per-stage-allocator-cleanup/vae-exact-transient-allocation-pre-tail-cleanup/phase-boundary-allocator-cleanup/qualified-native-kernels=",
        resource_plan.as_str(),
    ] {
        if !provenance.backend.contains(required) {
            return Err(format!(
                "native backend omits low-VRAM provenance {required:?}: {}",
                provenance.backend
            ));
        }
    }
    let HostImage::Pixels(pixels) = &output.images[0].image else {
        return Err("qualification output must be canonical host pixels".into());
    };
    let dimensions = Dimensions::new(
        configuration.request_identity.width,
        configuration.request_identity.height,
    )
    .map_err(|error| error.to_string())?;
    if pixels.dimensions() != dimensions
        || pixels.format() != PixelFormat::Rgb8
        || pixels.color_space() != ColorSpace::Srgb
    {
        return Err(format!(
            "output must be exact {} RGB8/sRGB pixels, found {} {:?}/{:?}",
            dimensions,
            pixels.dimensions(),
            pixels.format(),
            pixels.color_space()
        ));
    }
    Ok(())
}

fn finish_failure(host: &mut QualificationHost, error: String, exits: &mut MessageWriter<AppExit>) {
    if host.state == QualificationState::Finished {
        return;
    }
    if let Some(monitor) = host.monitor.as_ref() {
        monitor.set_phase(NativeOutputMemoryPhase::Finalization);
    }
    let device_memory = finish_monitor(host).ok();
    let stem = output_stem(&host.configuration.request_identity);
    let report = serde_json::json!({
        "schema_version": 1,
        "test": "burn_image_native_low_vram_output",
        "claim": "failed native output qualification; no cross-runtime numerical evidence",
        "ok": false,
        "failures": [&error],
        "request_identity": &host.configuration.request_identity,
        "artifacts": &host.configuration.artifacts,
        "progress_events": &host.progress,
        "device_memory": device_memory,
        "elapsed_milliseconds": host.started.elapsed().as_secs_f64() * 1_000.0,
        "completed_unix_milliseconds": unix_milliseconds(),
    });
    if let Err(write_error) = write_report(&host.configuration.output_directory, &stem, &report) {
        eprintln!("native output qualification report write failed: {write_error}");
    }
    eprintln!("native output qualification failed: {error}");
    host.state = QualificationState::Finished;
    exits.write(AppExit::error());
}

fn output_stem(identity: &NativeOutputQualificationRequestIdentity) -> String {
    format!(
        "burn-image-native-low-vram-{}-{}x{}",
        identity.variant, identity.width, identity.height
    )
}

fn write_report(
    output_directory: &Path,
    stem: &str,
    report: &serde_json::Value,
) -> Result<(), String> {
    fs::create_dir_all(output_directory)
        .map_err(|error| format!("create qualification report directory: {error}"))?;
    let report_path = output_directory.join(format!("{stem}-report.json"));
    let temporary_path = output_directory.join(format!(".{stem}-report.json.tmp"));
    let mut bytes = serde_json::to_vec_pretty(report)
        .map_err(|error| format!("serialize qualification report: {error}"))?;
    bytes.push(b'\n');
    fs::write(&temporary_path, bytes)
        .map_err(|error| format!("write qualification report temporary file: {error}"))?;
    fs::rename(&temporary_path, &report_path)
        .map_err(|error| format!("commit qualification report: {error}"))?;
    println!("{}", report_path.display());
    Ok(())
}

const fn variant_slug(variant: BooguVariant) -> &'static str {
    match variant {
        BooguVariant::Image01Turbo => "turbo",
        BooguVariant::Image01EditTurbo => "edit-turbo",
        BooguVariant::Image01EditTurbo1k5 => "edit-turbo-1k5",
    }
}

fn unix_milliseconds() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum NativeOutputMemoryPhase {
    ModelInitialization,
    Processing,
    Qwen,
    VaeEncode,
    Dmd,
    VaeDecode,
    Output,
    Finalization,
    OtherInference,
}

impl NativeOutputMemoryPhase {
    const ALL: [Self; 9] = [
        Self::ModelInitialization,
        Self::Processing,
        Self::Qwen,
        Self::VaeEncode,
        Self::Dmd,
        Self::VaeDecode,
        Self::Output,
        Self::Finalization,
        Self::OtherInference,
    ];

    const fn as_str(self) -> &'static str {
        match self {
            Self::ModelInitialization => "model_initialization",
            Self::Processing => "processing",
            Self::Qwen => "qwen",
            Self::VaeEncode => "vae_encode",
            Self::Dmd => "dmd",
            Self::VaeDecode => "vae_decode",
            Self::Output => "output",
            Self::Finalization => "finalization",
            Self::OtherInference => "other_inference",
        }
    }

    const fn from_u8(value: u8) -> Self {
        match value {
            value if value == Self::ModelInitialization as u8 => Self::ModelInitialization,
            value if value == Self::Processing as u8 => Self::Processing,
            value if value == Self::Qwen as u8 => Self::Qwen,
            value if value == Self::VaeEncode as u8 => Self::VaeEncode,
            value if value == Self::Dmd as u8 => Self::Dmd,
            value if value == Self::VaeDecode as u8 => Self::VaeDecode,
            value if value == Self::Output as u8 => Self::Output,
            value if value == Self::Finalization as u8 => Self::Finalization,
            _ => Self::OtherInference,
        }
    }

    const fn index(self) -> usize {
        self as usize
    }
}

fn memory_phase_for_progress_event(event: &ProgressEvent) -> Option<NativeOutputMemoryPhase> {
    match event {
        ProgressEvent::RunStarted { .. } => Some(NativeOutputMemoryPhase::Processing),
        ProgressEvent::StageStarted { stage, .. } => Some(match stage.as_str() {
            "processing" => NativeOutputMemoryPhase::Processing,
            "qwen" => NativeOutputMemoryPhase::Qwen,
            "vae-encode" => NativeOutputMemoryPhase::VaeEncode,
            "dmd" => NativeOutputMemoryPhase::Dmd,
            "vae-decode" => NativeOutputMemoryPhase::VaeDecode,
            "output" => NativeOutputMemoryPhase::Output,
            _ => NativeOutputMemoryPhase::OtherInference,
        }),
        ProgressEvent::RunCompleted { .. }
        | ProgressEvent::RunFailed { .. }
        | ProgressEvent::RunCancelled { .. } => Some(NativeOutputMemoryPhase::Finalization),
        ProgressEvent::ArtifactStarted { .. }
        | ProgressEvent::ArtifactProgress { .. }
        | ProgressEvent::ArtifactVerified { .. }
        | ProgressEvent::Step { .. }
        | ProgressEvent::StageCompleted { .. }
        | ProgressEvent::Warning { .. } => None,
    }
}

#[derive(Clone, Debug, Default)]
struct PidFramebufferPhaseSamples {
    matched_samples: u64,
    nonzero_samples: u64,
    peak_total_framebuffer_mib: u64,
    peak_observed_elapsed_milliseconds: Option<u64>,
    peak_observed_unix_milliseconds: Option<u64>,
}

#[derive(Clone, Debug)]
struct PidFramebufferSamples {
    attempted_samples: u64,
    matched_samples: u64,
    nonzero_samples: u64,
    peak_total_framebuffer_mib: u64,
    peak_phase_before: Option<NativeOutputMemoryPhase>,
    peak_phase_after: Option<NativeOutputMemoryPhase>,
    peak_observed_elapsed_milliseconds: Option<u64>,
    peak_observed_unix_milliseconds: Option<u64>,
    phase_transition_samples: u64,
    phase_samples: Vec<PidFramebufferPhaseSamples>,
    sample_error_count: u64,
    sample_errors: Vec<String>,
}

impl Default for PidFramebufferSamples {
    fn default() -> Self {
        Self {
            attempted_samples: 0,
            matched_samples: 0,
            nonzero_samples: 0,
            peak_total_framebuffer_mib: 0,
            peak_phase_before: None,
            peak_phase_after: None,
            peak_observed_elapsed_milliseconds: None,
            peak_observed_unix_milliseconds: None,
            phase_transition_samples: 0,
            phase_samples: vec![
                PidFramebufferPhaseSamples::default();
                NativeOutputMemoryPhase::ALL.len()
            ],
            sample_error_count: 0,
            sample_errors: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NativeOutputDeviceMemoryPhasePeak {
    pub phase: String,
    pub matched_samples: u64,
    pub nonzero_samples: u64,
    pub peak_total_framebuffer_mib: u64,
    pub peak_total_framebuffer_bytes: u64,
    pub peak_observed_elapsed_milliseconds: Option<u64>,
    pub peak_observed_unix_milliseconds: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NativeOutputDeviceMemoryQualification {
    pub provider: String,
    pub process_id: u32,
    pub sample_interval_milliseconds: u64,
    pub attempted_samples: u64,
    pub matched_samples: u64,
    pub nonzero_samples: u64,
    pub peak_total_framebuffer_mib: u64,
    pub peak_total_framebuffer_bytes: u64,
    pub peak_phase: Option<String>,
    pub peak_spanned_phase_transition: bool,
    pub peak_observed_elapsed_milliseconds: Option<u64>,
    pub peak_observed_unix_milliseconds: Option<u64>,
    pub phase_transition_samples: u64,
    pub phase_attribution: String,
    pub phase_peaks: Vec<NativeOutputDeviceMemoryPhasePeak>,
    pub strict_ceiling_bytes: u64,
    pub strictly_below_ceiling: bool,
    pub sample_error_count: u64,
    pub sample_errors: Vec<String>,
    pub passed: bool,
    pub failures: Vec<String>,
}

struct NvidiaSmiPidMonitor {
    process_id: u32,
    sample_interval: Duration,
    stop: Arc<AtomicBool>,
    phase: Arc<AtomicU8>,
    samples: Arc<Mutex<PidFramebufferSamples>>,
    worker: Option<JoinHandle<()>>,
}

impl NvidiaSmiPidMonitor {
    fn start(sample_interval: Duration) -> Result<Self, String> {
        let inventory = Command::new("nvidia-smi")
            .args(["--query-gpu=uuid", "--format=csv,noheader,nounits"])
            .output()
            .map_err(|error| format!("qualification requires nvidia-smi telemetry: {error}"))?;
        if !inventory.status.success() {
            return Err(format!(
                "could not inventory NVIDIA GPUs: {}",
                String::from_utf8_lossy(&inventory.stderr).trim()
            ));
        }
        if String::from_utf8_lossy(&inventory.stdout)
            .lines()
            .all(|line| line.trim().is_empty())
        {
            return Err("qualification requires at least one NVIDIA GPU".into());
        }
        let process_id = std::process::id();
        let stop = Arc::new(AtomicBool::new(false));
        let phase = Arc::new(AtomicU8::new(
            NativeOutputMemoryPhase::ModelInitialization as u8,
        ));
        let samples = Arc::new(Mutex::new(PidFramebufferSamples::default()));
        let worker_stop = Arc::clone(&stop);
        let worker_phase = Arc::clone(&phase);
        let worker_samples = Arc::clone(&samples);
        let worker_started = Instant::now();
        let worker = thread::Builder::new()
            .name("burn-image-output-nvidia-smi".into())
            .spawn(move || {
                loop {
                    let phase_before =
                        NativeOutputMemoryPhase::from_u8(worker_phase.load(Ordering::Acquire));
                    let sample = sample_pid_total_framebuffer_mib(process_id);
                    let observed_elapsed_milliseconds =
                        u64::try_from(worker_started.elapsed().as_millis()).unwrap_or(u64::MAX);
                    let observed_unix_milliseconds =
                        u64::try_from(unix_milliseconds()).unwrap_or(u64::MAX);
                    let phase_after =
                        NativeOutputMemoryPhase::from_u8(worker_phase.load(Ordering::Acquire));
                    if let Ok(mut samples) = worker_samples.lock() {
                        record_pid_framebuffer_sample(
                            &mut samples,
                            phase_before,
                            phase_after,
                            sample,
                            observed_elapsed_milliseconds,
                            observed_unix_milliseconds,
                        );
                    }
                    if worker_stop.load(Ordering::Acquire) {
                        break;
                    }
                    thread::sleep(sample_interval);
                }
            })
            .map_err(|error| format!("spawn telemetry worker: {error}"))?;
        Ok(Self {
            process_id,
            sample_interval,
            stop,
            phase,
            samples,
            worker: Some(worker),
        })
    }

    fn set_phase(&self, phase: NativeOutputMemoryPhase) {
        self.phase.store(phase as u8, Ordering::Release);
    }

    fn finish(mut self) -> Result<NativeOutputDeviceMemoryQualification, String> {
        self.stop_and_join()?;
        let samples = self
            .samples
            .lock()
            .map_err(|_| "telemetry state was poisoned".to_owned())?
            .clone();
        Ok(evaluate_device_memory_qualification(
            self.process_id,
            self.sample_interval,
            samples,
        ))
    }

    fn stop_and_join(&mut self) -> Result<(), String> {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            worker
                .join()
                .map_err(|_| "telemetry worker panicked".to_owned())?;
        }
        Ok(())
    }
}

impl Drop for NvidiaSmiPidMonitor {
    fn drop(&mut self) {
        let _ = self.stop_and_join();
    }
}

fn finish_monitor(
    host: &mut QualificationHost,
) -> Result<NativeOutputDeviceMemoryQualification, String> {
    host.monitor
        .take()
        .ok_or_else(|| "qualification telemetry was not started".to_owned())?
        .finish()
}

fn record_pid_framebuffer_sample(
    samples: &mut PidFramebufferSamples,
    phase_before: NativeOutputMemoryPhase,
    phase_after: NativeOutputMemoryPhase,
    sample: Result<Option<u64>, String>,
    observed_elapsed_milliseconds: u64,
    observed_unix_milliseconds: u64,
) {
    samples.attempted_samples += 1;
    match sample {
        Ok(Some(total_mib)) => {
            samples.matched_samples += 1;
            samples.nonzero_samples += u64::from(total_mib > 0);
            if samples.peak_observed_elapsed_milliseconds.is_none()
                || total_mib > samples.peak_total_framebuffer_mib
            {
                samples.peak_total_framebuffer_mib = total_mib;
                samples.peak_phase_before = Some(phase_before);
                samples.peak_phase_after = Some(phase_after);
                samples.peak_observed_elapsed_milliseconds = Some(observed_elapsed_milliseconds);
                samples.peak_observed_unix_milliseconds = Some(observed_unix_milliseconds);
            }

            if phase_before != phase_after {
                samples.phase_transition_samples += 1;
                return;
            }
            let phase_samples = &mut samples.phase_samples[phase_before.index()];
            phase_samples.matched_samples += 1;
            phase_samples.nonzero_samples += u64::from(total_mib > 0);
            if phase_samples.peak_observed_elapsed_milliseconds.is_none()
                || total_mib > phase_samples.peak_total_framebuffer_mib
            {
                phase_samples.peak_total_framebuffer_mib = total_mib;
                phase_samples.peak_observed_elapsed_milliseconds =
                    Some(observed_elapsed_milliseconds);
                phase_samples.peak_observed_unix_milliseconds = Some(observed_unix_milliseconds);
            }
        }
        Ok(None) => {}
        Err(error) => {
            samples.sample_error_count += 1;
            if samples.sample_errors.len() < 8 {
                samples.sample_errors.push(error);
            }
        }
    }
}

fn sample_pid_total_framebuffer_mib(process_id: u32) -> Result<Option<u64>, String> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .map_err(|error| format!("sample framebuffer telemetry: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "nvidia-smi process sample failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    parse_pid_total_framebuffer_mib(&String::from_utf8_lossy(&output.stdout), process_id)
}

fn parse_pid_total_framebuffer_mib(output: &str, process_id: u32) -> Result<Option<u64>, String> {
    let mut matched = false;
    let mut total_mib = 0_u64;
    for line in output
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
    {
        let (pid, memory) = line
            .split_once(',')
            .ok_or_else(|| format!("unparseable nvidia-smi process row {line:?}"))?;
        let row_pid = pid
            .trim()
            .parse::<u32>()
            .map_err(|_| format!("unparseable nvidia-smi process PID {pid:?}"))?;
        if row_pid != process_id {
            continue;
        }
        matched = true;
        let memory = memory.trim().trim_end_matches("MiB").trim();
        let memory_mib = memory.parse::<u64>().map_err(|_| {
            format!("unparseable framebuffer value {memory:?} for PID {process_id}")
        })?;
        total_mib = total_mib
            .checked_add(memory_mib)
            .ok_or_else(|| "PID-scoped framebuffer sum overflowed".to_owned())?;
    }
    Ok(matched.then_some(total_mib))
}

fn evaluate_device_memory_qualification(
    process_id: u32,
    sample_interval: Duration,
    samples: PidFramebufferSamples,
) -> NativeOutputDeviceMemoryQualification {
    let peak_total_framebuffer_bytes = samples
        .peak_total_framebuffer_mib
        .saturating_mul(FRAMEBUFFER_MIB_BYTES);
    let peak_spanned_phase_transition = matches!(
        (samples.peak_phase_before, samples.peak_phase_after),
        (Some(before), Some(after)) if before != after
    );
    let peak_phase = match (samples.peak_phase_before, samples.peak_phase_after) {
        (Some(before), Some(after)) if before == after => Some(before.as_str().to_owned()),
        _ => None,
    };
    let phase_peaks = NativeOutputMemoryPhase::ALL
        .into_iter()
        .map(|phase| {
            let samples = &samples.phase_samples[phase.index()];
            NativeOutputDeviceMemoryPhasePeak {
                phase: phase.as_str().to_owned(),
                matched_samples: samples.matched_samples,
                nonzero_samples: samples.nonzero_samples,
                peak_total_framebuffer_mib: samples.peak_total_framebuffer_mib,
                peak_total_framebuffer_bytes: samples
                    .peak_total_framebuffer_mib
                    .saturating_mul(FRAMEBUFFER_MIB_BYTES),
                peak_observed_elapsed_milliseconds: samples.peak_observed_elapsed_milliseconds,
                peak_observed_unix_milliseconds: samples.peak_observed_unix_milliseconds,
            }
        })
        .collect();
    let strictly_below_ceiling =
        peak_total_framebuffer_bytes < NATIVE_OUTPUT_QUALIFICATION_DEVICE_CEILING_BYTES;
    let mut failures = Vec::new();
    if samples.matched_samples < MIN_MATCHED_FRAMEBUFFER_SAMPLES {
        failures.push(format!(
            "nvidia-smi matched PID {process_id} in only {} intervals; at least {MIN_MATCHED_FRAMEBUFFER_SAMPLES} are required",
            samples.matched_samples
        ));
    }
    if samples.nonzero_samples < MIN_NONZERO_FRAMEBUFFER_SAMPLES {
        failures.push(format!(
            "nvidia-smi observed nonzero PID framebuffer in only {} intervals; at least {MIN_NONZERO_FRAMEBUFFER_SAMPLES} are required",
            samples.nonzero_samples
        ));
    }
    if samples.sample_error_count != 0 {
        failures.push(format!(
            "nvidia-smi encountered {} sampling errors",
            samples.sample_error_count
        ));
    }
    if !strictly_below_ceiling {
        failures.push(format!(
            "PID-scoped framebuffer peak {peak_total_framebuffer_bytes} bytes is not strictly below {} bytes",
            NATIVE_OUTPUT_QUALIFICATION_DEVICE_CEILING_BYTES
        ));
    }
    NativeOutputDeviceMemoryQualification {
        provider: "nvidia-smi".into(),
        process_id,
        sample_interval_milliseconds: u64::try_from(sample_interval.as_millis())
            .unwrap_or(u64::MAX),
        attempted_samples: samples.attempted_samples,
        matched_samples: samples.matched_samples,
        nonzero_samples: samples.nonzero_samples,
        peak_total_framebuffer_mib: samples.peak_total_framebuffer_mib,
        peak_total_framebuffer_bytes,
        peak_phase,
        peak_spanned_phase_transition,
        peak_observed_elapsed_milliseconds: samples.peak_observed_elapsed_milliseconds,
        peak_observed_unix_milliseconds: samples.peak_observed_unix_milliseconds,
        phase_transition_samples: samples.phase_transition_samples,
        phase_attribution: "phase observed immediately before and after each nvidia-smi query; samples spanning a ProgressEvent phase transition are excluded from per-phase peaks".into(),
        phase_peaks,
        strict_ceiling_bytes: NATIVE_OUTPUT_QUALIFICATION_DEVICE_CEILING_BYTES,
        strictly_below_ceiling,
        sample_error_count: samples.sample_error_count,
        sample_errors: samples.sample_errors,
        passed: failures.is_empty(),
        failures,
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    #[test]
    fn turbo_request_identity_is_exact_and_ascii_correctness() {
        let prompt = "A studio photograph of a blue ceramic bird on a plain white table.";
        let (request, identity) = prepare_native_output_qualification_request(
            BooguVariant::Image01Turbo,
            prompt.into(),
            0,
            1_024,
            1_024,
            None,
        )
        .unwrap();
        assert!(identity.prompt.is_ascii());
        assert_eq!(identity.task, "generate");
        assert_eq!(identity.prompt, prompt);
        assert_eq!(identity.seed, 0);
        assert_eq!((identity.width, identity.height), (1_024, 1_024));
        assert!(identity.source.is_none());
        assert!(matches!(request, ImageRequest::Generate(_)));
    }

    #[test]
    fn released_edit_requests_require_an_explicit_source_correctness() {
        for variant in [
            BooguVariant::Image01EditTurbo,
            BooguVariant::Image01EditTurbo1k5,
        ] {
            let error = prepare_native_output_qualification_request(
                variant,
                "edit instruction".into(),
                7,
                if variant == BooguVariant::Image01EditTurbo {
                    1_024
                } else {
                    1_536
                },
                if variant == BooguVariant::Image01EditTurbo {
                    1_024
                } else {
                    1_536
                },
                None,
            )
            .unwrap_err();
            assert!(error.contains("requires an explicit source"));
        }
        assert!(
            prepare_native_output_qualification_request(
                BooguVariant::Image01Turbo,
                "prompt".into(),
                0,
                1_024,
                1_024,
                Some("source.png".into()),
            )
            .unwrap_err()
            .contains("forbids")
        );
    }

    #[test]
    fn qualification_source_reader_rejects_preflight_oversize_correctness() {
        let mut source = tempfile::NamedTempFile::new().unwrap();
        source.write_all(b"123456").unwrap();
        source.flush().unwrap();
        let file = fs::File::open(source.path()).unwrap();
        let preflight_bytes = file.metadata().unwrap().len();

        let error =
            read_bounded_qualification_source_handle(file, preflight_bytes, 5, source.path())
                .unwrap_err();
        assert!(error.contains("6 bytes; limit is 5 bytes"));
    }

    #[test]
    fn qualification_source_reader_rejects_growth_after_preflight_correctness() {
        let mut source = tempfile::NamedTempFile::new().unwrap();
        source.write_all(b"12345").unwrap();
        source.flush().unwrap();
        let file = fs::File::open(source.path()).unwrap();
        let preflight_bytes = file.metadata().unwrap().len();

        // Mutate the same inode after metadata was sampled. The production reader retains the
        // already-open handle and reads at most MAX+1, so this cannot become an unbounded read.
        source.write_all(b"6").unwrap();
        source.flush().unwrap();
        let error =
            read_bounded_qualification_source_handle(file, preflight_bytes, 5, source.path())
                .unwrap_err();
        assert!(error.contains("grew beyond the 5-byte limit"));
    }

    #[test]
    fn pid_framebuffer_parser_sums_all_matching_gpu_rows_correctness() {
        let rows = "81, 100 MiB\n12, 9 MiB\n81, 250 MiB\n";
        assert_eq!(parse_pid_total_framebuffer_mib(rows, 81), Ok(Some(350)));
        assert_eq!(parse_pid_total_framebuffer_mib(rows, 99), Ok(None));
        assert!(parse_pid_total_framebuffer_mib("invalid", 81).is_err());
    }

    #[test]
    fn progress_event_selects_memory_phase_correctness() {
        let started = |stage: &str| ProgressEvent::StageStarted {
            run_id: burn_image::RunId(1),
            stage: stage.into(),
            total_steps: Some(1),
        };
        for (stage, expected) in [
            ("processing", NativeOutputMemoryPhase::Processing),
            ("qwen", NativeOutputMemoryPhase::Qwen),
            ("vae-encode", NativeOutputMemoryPhase::VaeEncode),
            ("dmd", NativeOutputMemoryPhase::Dmd),
            ("vae-decode", NativeOutputMemoryPhase::VaeDecode),
            ("output", NativeOutputMemoryPhase::Output),
            ("future-stage", NativeOutputMemoryPhase::OtherInference),
        ] {
            assert_eq!(
                memory_phase_for_progress_event(&started(stage)),
                Some(expected)
            );
        }
        assert_eq!(
            memory_phase_for_progress_event(&ProgressEvent::RunStarted {
                run_id: burn_image::RunId(1),
                model: burn_image::ModelId::new("test/model").unwrap(),
                task: burn_image::ImageTaskKind::Generate,
            }),
            Some(NativeOutputMemoryPhase::Processing)
        );
        assert_eq!(
            memory_phase_for_progress_event(&ProgressEvent::RunCompleted {
                run_id: burn_image::RunId(1),
                elapsed_micros: 1,
            }),
            Some(NativeOutputMemoryPhase::Finalization)
        );
        assert_eq!(
            memory_phase_for_progress_event(&ProgressEvent::StageCompleted {
                run_id: burn_image::RunId(1),
                stage: "qwen".into(),
                elapsed_micros: 1,
            }),
            None
        );
    }

    #[test]
    fn phase_peak_attribution_and_timestamps_correctness() {
        let mut samples = PidFramebufferSamples::default();
        for (phase, total_mib, elapsed, unix) in [
            (NativeOutputMemoryPhase::ModelInitialization, 100, 10, 1_010),
            (NativeOutputMemoryPhase::Qwen, 200, 20, 1_020),
            (NativeOutputMemoryPhase::Dmd, 180, 30, 1_030),
            (NativeOutputMemoryPhase::VaeDecode, 300, 40, 1_040),
        ] {
            record_pid_framebuffer_sample(
                &mut samples,
                phase,
                phase,
                Ok(Some(total_mib)),
                elapsed,
                unix,
            );
        }

        let qualification =
            evaluate_device_memory_qualification(7, Duration::from_millis(100), samples);
        assert!(qualification.passed);
        assert_eq!(qualification.peak_phase.as_deref(), Some("vae_decode"));
        assert!(!qualification.peak_spanned_phase_transition);
        assert_eq!(qualification.peak_total_framebuffer_mib, 300);
        assert_eq!(qualification.peak_observed_elapsed_milliseconds, Some(40));
        assert_eq!(qualification.peak_observed_unix_milliseconds, Some(1_040));
        let qwen = qualification
            .phase_peaks
            .iter()
            .find(|peak| peak.phase == "qwen")
            .unwrap();
        assert_eq!(qwen.matched_samples, 1);
        assert_eq!(qwen.peak_total_framebuffer_mib, 200);
        assert_eq!(qwen.peak_observed_elapsed_milliseconds, Some(20));
        assert_eq!(qwen.peak_observed_unix_milliseconds, Some(1_020));
    }

    #[test]
    fn cross_phase_sample_is_not_misattributed_correctness() {
        let mut samples = PidFramebufferSamples::default();
        record_pid_framebuffer_sample(
            &mut samples,
            NativeOutputMemoryPhase::Qwen,
            NativeOutputMemoryPhase::Dmd,
            Ok(Some(500)),
            50,
            1_050,
        );

        let qualification =
            evaluate_device_memory_qualification(7, Duration::from_millis(100), samples);
        assert_eq!(qualification.matched_samples, 1);
        assert_eq!(qualification.phase_transition_samples, 1);
        assert_eq!(qualification.peak_phase, None);
        assert!(qualification.peak_spanned_phase_transition);
        assert!(
            qualification
                .phase_peaks
                .iter()
                .all(|peak| peak.matched_samples == 0)
        );
    }

    #[test]
    fn device_memory_gate_is_strictly_below_decimal_32_gb_correctness() {
        let sample = |peak_total_framebuffer_mib| PidFramebufferSamples {
            attempted_samples: 5,
            matched_samples: 5,
            nonzero_samples: 5,
            peak_total_framebuffer_mib,
            ..Default::default()
        };
        let accepted =
            evaluate_device_memory_qualification(7, Duration::from_millis(100), sample(30_517));
        assert!(accepted.passed);
        let rejected =
            evaluate_device_memory_qualification(7, Duration::from_millis(100), sample(30_518));
        assert!(!rejected.passed);
        assert!(!rejected.strictly_below_ceiling);
    }
}
