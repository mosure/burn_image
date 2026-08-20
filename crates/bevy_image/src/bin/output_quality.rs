//! Fail-closed same-seed final-output quality comparison for one native/browser request.

use std::{
    collections::BTreeMap,
    error::Error,
    fs,
    io::Read,
    path::{Path, PathBuf},
};

use bevy_burn_image::{BROWSER_SURFACE_INFERENCE_POLICY, NativeOutputQualificationRequestIdentity};
use burn_boogu::{
    BOOGU_1K5_OUTPUT_PRESETS, BooguVariant,
    artifacts::{
        BooguStorageProfile, VerifiedArtifactDirectory, canonical_published_bundle,
        validate_canonical_release_artifact_digest,
    },
    boogu_model_descriptor,
    web_policy::{
        PACKED_F16_DMD_VAE_HANDOFF_POLICY as BROWSER_PACKED_F16_DMD_VAE_HANDOFF_POLICY,
        PACKED_F16_NEXT_REQUEST_REHYDRATION_POLICY as BROWSER_PACKED_F16_NEXT_REQUEST_REHYDRATION_POLICY,
    },
};
use burn_image::{Dimensions, Sha256Digest, ValidationError};
use image::{ColorType, ImageReader};
use serde::Serialize;
use serde_json::{Value, json};

const MINIMUM_PSNR_DB: f32 = 24.0;
const MINIMUM_MEAN_BLOCK_SSIM_8X8: f32 = 0.90;
const DEVICE_CEILING_BYTES: u64 = 32_000_000_000;
const MAX_EDIT_SOURCE_BYTES: u64 = 64 * 1024 * 1024;
const JAVASCRIPT_MAX_SAFE_INTEGER_U64: u64 = 9_007_199_254_740_991;
const NATIVE_TEST: &str = "burn_image_native_low_vram_output";
const BROWSER_TURBO_TEST: &str = "burn_image_browser_rendered_turbo_1024_smoke";
const BROWSER_GENERIC_TEST: &str = "burn_image_browser_rendered_output_smoke";
const BROWSER_LOW_VRAM_BACKEND: &str = "burn-webgpu/browser-low-vram-preloaded-packed-f16-dense-f32-per-stage-denoiser/request-scoped-packed-cache-evicted-before-vae/request-scoped-surface-acquire-suspended";
const BROWSER_QWEN_BLOCK0_ORDINARY_MODE: &str = "ordinary";
const TURBO_PACKED_F16_STAGES: u64 = 46;
const TURBO_PACKED_F16_OBJECTS: u64 = 106;
const TURBO_PACKED_F16_TENSORS: u64 = 912;
const TURBO_PACKED_F16_ARTIFACT_BYTES: u64 = 19_870_166_528;
const TURBO_PACKED_F16_COMPACT_BYTES: u64 = 19_869_996_096;
const TURBO_PACKED_F16_RETAINED_BYTES: u64 = 19_870_010_624;
const TURBO_PACKED_F16_STAGE_MATERIALIZATIONS: u64 = 184;
const TURBO_PACKED_F16_OBJECT_UNPACKS: u64 = 424;
const TURBO_PACKED_F16_REQUEST_READ_BYTES: u64 = 79_480_042_496;
const TURBO_PACKED_F16_REQUEST_WRITE_BYTES: u64 = 158_960_084_992;
const ARTIFACT_TRAFFIC_FIELDS: [&str; 17] = [
    "object_reads",
    "object_read_bytes",
    "range_reads",
    "range_read_bytes",
    "verified_objects",
    "cache_lookups",
    "cache_hits",
    "cache_misses",
    "cache_read_bytes",
    "network_requests",
    "network_response_bytes",
    "cache_writes",
    "cache_write_bytes",
    "cache_evictions",
    "cache_evicted_entries",
    "cache_invalid_entries",
    "integrity_refetches",
];

#[derive(Debug)]
struct Args {
    native_report: PathBuf,
    browser_report: PathBuf,
    output: PathBuf,
}

#[derive(Debug, Serialize)]
struct RgbMetrics {
    width: usize,
    height: usize,
    max_abs_u8: u8,
    mean_abs_u8: f32,
    rmse_u8: f32,
    psnr_db: f32,
    mean_block_ssim_8x8: f32,
    exact_fraction: f32,
}

#[derive(Debug)]
struct ValidatedOutput {
    request: NativeOutputQualificationRequestIdentity,
    report_sha256: String,
    png_path: PathBuf,
    png_sha256: String,
    artifact_bundles: BTreeMap<String, String>,
    peak_framebuffer_bytes: u64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args(std::env::args().skip(1))?;
    let outcome = compare_reports(&args);
    let (report, passed) = match outcome {
        Ok(report) => {
            let passed = report
                .get("passed")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            (report, passed)
        }
        Err(error) => (
            json!({
                "schema_version": 1,
                "test": "burn_image_native_browser_same_seed_output_quality",
                "claim": "failed required same-seed final-output quality comparison; no numerical parity evidence",
                "numerical_parity_claimed": false,
                "exact_noise_tensors_injected": false,
                "gate_kind": "same-seed-final-output-quality",
                "passed": false,
                "failures": [error],
            }),
            false,
        ),
    };
    write_json_atomically(&args.output, &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if !passed {
        return Err("native/browser same-seed output quality gate failed".into());
    }
    Ok(())
}

fn parse_args(arguments: impl IntoIterator<Item = String>) -> Result<Args, String> {
    let mut native_report = None;
    let mut browser_report = None;
    let mut output = None;
    let mut arguments = arguments.into_iter();
    while let Some(argument) = arguments.next() {
        let target = match argument.as_str() {
            "--native-report" => &mut native_report,
            "--browser-report" => &mut browser_report,
            "--output" => &mut output,
            _ => return Err(format!("unknown argument {argument:?}")),
        };
        if target.is_some() {
            return Err(format!("argument {argument} was supplied more than once"));
        }
        *target =
            Some(PathBuf::from(arguments.next().ok_or_else(|| {
                format!("argument {argument} requires a path")
            })?));
    }
    Ok(Args {
        native_report: native_report.ok_or("--native-report is required")?,
        browser_report: browser_report.ok_or("--browser-report is required")?,
        output: output.ok_or("--output is required")?,
    })
}

fn compare_reports(args: &Args) -> Result<Value, String> {
    let native_bytes = read(&args.native_report, "native report")?;
    let browser_bytes = read(&args.browser_report, "browser report")?;
    let native_json: Value = serde_json::from_slice(&native_bytes)
        .map_err(|error| format!("parse native report: {error}"))?;
    let browser_json: Value = serde_json::from_slice(&browser_bytes)
        .map_err(|error| format!("parse browser report: {error}"))?;
    let native = validate_native_report(&native_json, &native_bytes)?;
    let browser = validate_browser_report(&browser_json, &browser_bytes)?;
    if !request_identities_match(&native.request, &browser.request) {
        return Err(format!(
            "native/browser request identities differ: native={:?} browser={:?}",
            native.request, browser.request
        ));
    }
    if native.artifact_bundles != browser.artifact_bundles {
        return Err(format!(
            "native/browser modular closures differ: native={:?} browser={:?}",
            native.artifact_bundles, browser.artifact_bundles
        ));
    }

    let native_rgb = decode_normalized_rgb8(&native.png_path, &native.request)?;
    let browser_rgb = decode_normalized_rgb8(&browser.png_path, &browser.request)?;
    let metrics = compare_rgb(
        &native_rgb,
        &browser_rgb,
        native.request.width as usize,
        native.request.height as usize,
    )?;
    let mut failures = Vec::new();
    if metrics.psnr_db < MINIMUM_PSNR_DB {
        failures.push(format!(
            "RGB PSNR {} dB is below {MINIMUM_PSNR_DB} dB",
            metrics.psnr_db
        ));
    }
    if metrics.mean_block_ssim_8x8 < MINIMUM_MEAN_BLOCK_SSIM_8X8 {
        failures.push(format!(
            "RGB mean block SSIM {} is below {MINIMUM_MEAN_BLOCK_SSIM_8X8}",
            metrics.mean_block_ssim_8x8
        ));
    }
    let passed = failures.is_empty();
    Ok(json!({
        "schema_version": 1,
        "test": "burn_image_native_browser_same_seed_output_quality",
        "claim": "required same-request, same-seed final RGB8 quality/similarity comparison between production native and rendered-browser low-VRAM execution; each runtime independently generates noise, so this is not numerical parity evidence",
        "numerical_parity_claimed": false,
        "exact_noise_tensors_injected": false,
        "gate_kind": "same-seed-final-output-quality",
        "passed": passed,
        "request_identity": native.request,
        "normalization": "decode PNG without resize or color transform, require opaque RGB8/RGBA8, drop alpha, compare packed RGB8",
        "thresholds": {
            "minimum_psnr_db": MINIMUM_PSNR_DB,
            "minimum_mean_block_ssim_8x8": MINIMUM_MEAN_BLOCK_SSIM_8X8,
        },
        "metrics": metrics,
        "native": {
            "report_path": canonical_or_original(&args.native_report),
            "report_sha256": native.report_sha256,
            "png_path": native.png_path,
            "png_sha256": native.png_sha256,
            "peak_framebuffer_bytes": native.peak_framebuffer_bytes,
            "execution": "native-low-vram-phase-resident-mixed-f16",
        },
        "browser": {
            "report_path": canonical_or_original(&args.browser_report),
            "report_sha256": browser.report_sha256,
            "png_path": browser.png_path,
            "png_sha256": browser.png_sha256,
            "peak_framebuffer_bytes": browser.peak_framebuffer_bytes,
            "execution": BROWSER_LOW_VRAM_BACKEND,
            "qwen_block0_execution_mode": BROWSER_QWEN_BLOCK0_ORDINARY_MODE,
        },
        "artifact_bundles": native.artifact_bundles,
        "failures": failures,
    }))
}

fn validate_native_report(report: &Value, bytes: &[u8]) -> Result<ValidatedOutput, String> {
    exact(report, "/schema_version", json!(1), "native schema")?;
    exact(report, "/test", json!(NATIVE_TEST), "native test")?;
    exact(report, "/ok", json!(true), "native result")?;
    let request: NativeOutputQualificationRequestIdentity =
        serde_json::from_value(required(report, "/request_identity")?.clone())
            .map_err(|error| format!("decode native request identity: {error}"))?;
    validate_request_identity(&request)?;
    let variant = request_variant(&request)?;
    let published = canonical_published_bundle(variant, BooguStorageProfile::F16QwenVisionF32)
        .ok_or_else(|| "native request has no canonical production bundle".to_owned())?;
    exact(
        report,
        "/artifacts/parent_bundle",
        json!(published.bundle_id),
        "native canonical parent bundle",
    )?;
    exact(
        report,
        "/artifacts/parent_content_digest",
        json!(published.content_digest),
        "native canonical parent digest",
    )?;
    for pointer in [
        "/artifacts/dependency_closure_verified",
        "/artifacts/component_contracts_verified",
        "/artifacts/reconstructed_inventory_verified",
        "/artifacts/canonical_parent_digest_verified",
    ] {
        exact(report, pointer, json!(true), "native modular verification")?;
    }
    exact(
        report,
        "/selection/model",
        json!(request.model),
        "native selected model",
    )?;
    exact(
        report,
        "/selection/residency",
        json!("native-low-vram-phase-resident-mixed-f16"),
        "native residency",
    )?;
    exact(
        report,
        "/output/provenance/model",
        json!(request.model),
        "native output model",
    )?;
    exact(
        report,
        "/output/provenance/model_revision",
        json!(request.model_revision),
        "native revision",
    )?;
    exact(
        report,
        "/output/provenance/artifact_content_digest",
        json!(published.content_digest),
        "native output digest",
    )?;
    exact(
        report,
        "/output/provenance/artifacts_verified",
        json!(true),
        "native artifact verification",
    )?;
    exact(
        report,
        "/output/provenance/numeric_format/other",
        json!("f16-qwen-vision-f32"),
        "native numeric format",
    )?;
    exact(
        report,
        "/output/save_path",
        json!("bevy_burn_image::encode_host_image(ImageEncoding::Png)"),
        "native production PNG path",
    )?;
    exact(
        report,
        "/output/opaque_srgb_rgb8_source",
        json!(true),
        "native RGB source",
    )?;
    exact(
        report,
        "/device_memory/passed",
        json!(true),
        "native memory gate",
    )?;
    exact(
        report,
        "/device_memory/strict_ceiling_bytes",
        json!(DEVICE_CEILING_BYTES),
        "native memory ceiling",
    )?;
    let peak = positive_sub_ceiling(
        required_u64(report, "/device_memory/peak_total_framebuffer_bytes")?,
        "native",
    )?;
    let backend = required_str(report, "/output/provenance/backend")?;
    for marker in [
        "burn-wgpu-native/shared-bevy-device/",
        "native-low-vram-phase-resident-mixed-f16",
        "phase-resident/qwen+vae-per-request/denoiser-resident-zero-dmd-weight-reloads",
    ] {
        if !backend.contains(marker) {
            return Err(format!("native backend omits {marker:?}: {backend}"));
        }
    }
    let png_path = PathBuf::from(required_str(report, "/output/path")?);
    let png_sha256 = validate_file(
        &png_path,
        required_str(report, "/output/sha256")?,
        required_u64(report, "/output/bytes")?,
        "native PNG",
    )?;
    let artifact_bundles = native_bundles(report, variant)?;
    Ok(ValidatedOutput {
        request,
        report_sha256: Sha256Digest::calculate(bytes).to_hex(),
        png_path,
        png_sha256,
        artifact_bundles,
        peak_framebuffer_bytes: peak,
    })
}

fn validate_browser_report(report: &Value, bytes: &[u8]) -> Result<ValidatedOutput, String> {
    exact(report, "/schema_version", json!(1), "browser schema")?;
    exact(report, "/ok", json!(true), "browser result")?;
    let test = required_str(report, "/test")?;
    let (evidence, expected_test) = match required(report, "/evidence/turbo_1024_model") {
        Ok(evidence) => (evidence, BROWSER_TURBO_TEST),
        Err(_) => (
            required(report, "/evidence/model_output")?,
            BROWSER_GENERIC_TEST,
        ),
    };
    if test != expected_test {
        return Err(format!(
            "browser test identity {test:?} does not match {expected_test:?}"
        ));
    }
    validate_browser_ordinary_execution_mode(evidence)?;
    validate_browser_surface_gate_evidence(evidence)?;
    if expected_test == BROWSER_TURBO_TEST {
        validate_turbo_packed_f16_evidence(evidence)?;
    }
    let request = serde_json::from_value(required(evidence, "/request_identity")?.clone())
        .map_err(|error| format!("decode browser request identity: {error}"))?;
    validate_request_identity(&request)?;
    let published = canonical_published_bundle(
        request_variant(&request)?,
        BooguStorageProfile::F16QwenVisionF32,
    )
    .ok_or_else(|| "browser request has no canonical production bundle".to_owned())?;
    exact(
        evidence,
        "/output_ready/model",
        json!(request.model),
        "browser output model",
    )?;
    exact(
        evidence,
        "/output_ready/model_revision",
        json!(request.model_revision),
        "browser revision",
    )?;
    exact(
        evidence,
        "/output_ready/backend",
        json!(BROWSER_LOW_VRAM_BACKEND),
        "browser backend",
    )?;
    exact(
        evidence,
        "/output_ready/artifacts_verified",
        json!(true),
        "browser artifact verification",
    )?;
    exact(
        evidence,
        "/output_ready/numeric_format",
        json!("f16-qwen-vision-f32"),
        "browser numeric format",
    )?;
    exact(
        evidence,
        "/output_ready/artifact_content_digest",
        json!(published.content_digest),
        "browser canonical output digest",
    )?;
    exact(
        evidence,
        "/output_ready/width",
        json!(request.width),
        "browser output width",
    )?;
    exact(
        evidence,
        "/output_ready/height",
        json!(request.height),
        "browser output height",
    )?;
    exact(
        evidence,
        "/interaction/prompt_value",
        json!(request.prompt),
        "browser typed prompt",
    )?;
    exact(
        evidence,
        "/interaction/seed_value",
        json!(request.seed.to_string()),
        "browser typed seed",
    )?;
    for pointer in [
        "/interaction/prompt_typed_via_cdp",
        "/interaction/seed_typed_via_cdp",
        "/interaction/run_clicked_via_cdp",
        "/interaction/save_clicked_via_cdp",
    ] {
        exact(evidence, pointer, json!(true), "browser CDP input")?;
    }
    if let Some(source) = &request.source {
        exact(
            evidence,
            "/interaction/source_uploaded_via_file_dialog",
            json!(true),
            "browser edit source upload",
        )?;
        exact(
            evidence,
            "/interaction/source_sha256",
            json!(source.sha256),
            "browser edit source digest",
        )?;
        exact(
            evidence,
            "/interaction/source_bytes",
            json!(source.bytes),
            "browser edit source bytes",
        )?;
    }
    exact(
        evidence,
        "/native_gpu_attestation/validated",
        json!(true),
        "browser memory gate",
    )?;
    exact(
        evidence,
        "/native_gpu_attestation/maximum_framebuffer_bytes_exclusive",
        json!(DEVICE_CEILING_BYTES),
        "browser memory ceiling",
    )?;
    let peak = positive_sub_ceiling(
        required_u64(
            evidence,
            "/native_gpu_attestation/observed_peak_aggregate_framebuffer_bytes",
        )?,
        "browser",
    )?;
    let png_path = PathBuf::from(required_str(evidence, "/downloaded_png/path")?);
    let png_sha256 = validate_file(
        &png_path,
        required_str(evidence, "/downloaded_png/sha256")?,
        required_u64(evidence, "/downloaded_png/bytes")?,
        "browser PNG",
    )?;
    let artifact_bundles = browser_bundles(evidence, &request)?;
    Ok(ValidatedOutput {
        request,
        report_sha256: Sha256Digest::calculate(bytes).to_hex(),
        png_path,
        png_sha256,
        artifact_bundles,
        peak_framebuffer_bytes: peak,
    })
}

fn validate_turbo_packed_f16_evidence(evidence: &Value) -> Result<(), String> {
    let events = required(evidence, "/runtime_events")?
        .as_array()
        .ok_or_else(|| "browser Turbo runtime_events is not an array".to_owned())?;
    let ready = exact_runtime_event(events, "ready")?;
    exact(
        ready,
        "/block0_execution_mode",
        json!(BROWSER_QWEN_BLOCK0_ORDINARY_MODE),
        "browser output-quality ready-event Qwen block-0 execution mode",
    )?;
    let plan = exact_runtime_event(events, "packed_f16_resource_plan")?;
    for (field, expected) in [
        (
            "authenticated_artifact_bytes",
            TURBO_PACKED_F16_ARTIFACT_BYTES,
        ),
        (
            "canonical_compact_f16_payload_bytes",
            TURBO_PACKED_F16_COMPACT_BYTES,
        ),
        (
            "retained_packed_f16_denoiser_bytes",
            TURBO_PACKED_F16_RETAINED_BYTES,
        ),
        ("inserted_padding_elements", 7_264),
        ("padded_f16_elements", 9_935_005_312),
        ("expected_stage_count", TURBO_PACKED_F16_STAGES),
        ("expected_object_count", TURBO_PACKED_F16_OBJECTS),
        ("expected_tensor_count", TURBO_PACKED_F16_TENSORS),
        ("max_packed_stage_bytes", 876_827_328),
        ("max_materialized_stage_f32_bytes", 1_753_654_656),
        ("max_packed_object_bytes", 254_251_904),
        ("max_materialized_object_f32_bytes", 508_503_808),
        ("materialized_f32_bytes_per_dmd_step", 39_740_021_248),
        ("preload_workspace_bytes", 2_434_252_800),
        ("preload_peak_bytes", 22_304_263_424),
        ("activation_reserve_bytes", 4_868_505_600),
        ("conservative_planned_device_bytes", 26_492_170_880),
        ("strict_device_cap_bytes", DEVICE_CEILING_BYTES),
        (
            "expected_stage_materializations_per_request",
            TURBO_PACKED_F16_STAGE_MATERIALIZATIONS,
        ),
        (
            "expected_object_unpacks_per_request",
            TURBO_PACKED_F16_OBJECT_UNPACKS,
        ),
        (
            "expected_packed_read_bytes_per_request",
            TURBO_PACKED_F16_REQUEST_READ_BYTES,
        ),
        (
            "expected_f32_write_bytes_per_request",
            TURBO_PACKED_F16_REQUEST_WRITE_BYTES,
        ),
    ] {
        exact(
            plan,
            &format!("/{field}"),
            json!(expected),
            "browser packed-F16 resource plan",
        )?;
    }
    exact(
        plan,
        "/on_device_quantized_execution_claimed",
        json!(false),
        "browser packed-F16 quantized-execution claim",
    )?;

    let preload = exact_runtime_event(events, "packed_f16_denoiser_preload")?;
    for (field, expected) in [
        ("cached_stages", TURBO_PACKED_F16_STAGES),
        ("cached_objects", TURBO_PACKED_F16_OBJECTS),
        ("cached_tensors", TURBO_PACKED_F16_TENSORS),
        ("cached_bytes", TURBO_PACKED_F16_RETAINED_BYTES),
        ("previous_preload_attempt_count", 0),
        ("preload_attempt_count", 1),
    ] {
        exact(
            preload,
            &format!("/{field}"),
            json!(expected),
            "browser packed-F16 preload",
        )?;
    }
    exact(
        preload,
        "/request_scoped_rehydration",
        json!(false),
        "browser initial packed-F16 preload",
    )?;
    exact(
        preload,
        "/rehydration_policy",
        json!(BROWSER_PACKED_F16_NEXT_REQUEST_REHYDRATION_POLICY),
        "browser initial packed-F16 preload rehydration policy",
    )?;

    let lifecycle_event = exact_runtime_event(events, "packed_f16_denoiser_lifecycle")?;
    let lifecycle = required(evidence, "/packed_f16_denoiser_lifecycle")?;
    if lifecycle_event.pointer("/lifecycle") != Some(lifecycle) {
        return Err(
            "browser top-level packed-F16 lifecycle differs from its exact runtime event".into(),
        );
    }
    exact(
        lifecycle,
        "/cache_state",
        json!("ready"),
        "browser packed-F16 lifecycle",
    )?;
    for (field, expected) in [
        ("cached_stages", TURBO_PACKED_F16_STAGES),
        ("cached_objects", TURBO_PACKED_F16_OBJECTS),
        ("cached_tensors", TURBO_PACKED_F16_TENSORS),
        ("cached_bytes", TURBO_PACKED_F16_RETAINED_BYTES),
        (
            "authenticated_artifact_bytes",
            TURBO_PACKED_F16_ARTIFACT_BYTES,
        ),
        ("packed_upload_bytes", TURBO_PACKED_F16_RETAINED_BYTES),
        (
            "stage_materializations",
            TURBO_PACKED_F16_STAGE_MATERIALIZATIONS,
        ),
        ("object_unpacks", TURBO_PACKED_F16_OBJECT_UNPACKS),
        ("packed_read_bytes", TURBO_PACKED_F16_REQUEST_READ_BYTES),
        ("f32_write_bytes", TURBO_PACKED_F16_REQUEST_WRITE_BYTES),
        ("preload_attempt_count", 1),
        ("failure_count", 0),
    ] {
        exact(
            lifecycle,
            &format!("/{field}"),
            json!(expected),
            "browser packed-F16 lifecycle",
        )?;
    }
    for (pointer, expected) in [
        ("/cache_ready", true),
        ("/synchronization_pending", false),
        ("/matches_plan", true),
    ] {
        exact(
            lifecycle,
            pointer,
            json!(expected),
            "browser packed-F16 lifecycle",
        )?;
    }
    for field in ARTIFACT_TRAFFIC_FIELDS {
        exact(
            lifecycle,
            &format!("/dmd_artifact_traffic/{field}"),
            json!(0),
            "browser packed-F16 zero-DMD-I/O lifecycle",
        )?;
    }

    let handoff_event = exact_runtime_event(events, "packed_f16_dmd_vae_handoff")?;
    let top_level_handoff = required(evidence, "/packed_f16_dmd_vae_handoff")?;
    if handoff_event != top_level_handoff {
        return Err(
            "browser top-level packed-F16 DMD-to-VAE handoff differs from its exact runtime event"
                .into(),
        );
    }
    let handoff = required(handoff_event, "/report")?;
    exact(
        handoff,
        "/policy",
        json!(BROWSER_PACKED_F16_DMD_VAE_HANDOFF_POLICY),
        "browser packed-F16 DMD-to-VAE handoff policy",
    )?;
    exact(
        handoff,
        "/next_request_rehydration_policy",
        json!(BROWSER_PACKED_F16_NEXT_REQUEST_REHYDRATION_POLICY),
        "browser packed-F16 next-request rehydration policy",
    )?;
    exact(
        handoff,
        "/shape",
        json!([1, 16, 128, 128]),
        "browser packed-F16 DMD-to-VAE latent shape",
    )?;
    exact(
        handoff,
        "/dtype",
        json!("f32"),
        "browser packed-F16 DMD-to-VAE latent dtype",
    )?;
    for (field, expected) in [
        ("element_count", 262_144),
        ("payload_bytes", 1_048_576),
        ("device_to_host_readback_bytes", 2_097_152),
        ("host_to_device_upload_bytes", 1_048_576),
        ("total_transfer_bytes", 3_145_728),
        ("wrapper_cached_stages_before_clear", 0),
        ("wrapper_cached_stages_after_clear", 0),
        ("preload_attempt_count", 1),
        ("expected_next_request_preload_attempt_count", 2),
    ] {
        exact(
            handoff,
            &format!("/{field}"),
            json!(expected),
            "browser packed-F16 DMD-to-VAE handoff",
        )?;
    }
    for (pointer, expected) in [
        ("/all_finite", true),
        ("/not_all_zero", true),
        ("/digest_matches", true),
        ("/synchronization_pending_before_cleanup", false),
        ("/synchronization_pending_after_cleanup", false),
        ("/rope_cache_cleared", true),
        ("/cleanup_completed", true),
    ] {
        exact(
            handoff,
            pointer,
            json!(expected),
            "browser packed-F16 DMD-to-VAE handoff",
        )?;
    }
    for (field, expected) in [
        ("state", json!("ready")),
        ("cache_ready", json!(true)),
        ("cached_stages", json!(TURBO_PACKED_F16_STAGES)),
        ("cached_objects", json!(TURBO_PACKED_F16_OBJECTS)),
        ("cached_tensors", json!(TURBO_PACKED_F16_TENSORS)),
        ("cached_bytes", json!(TURBO_PACKED_F16_RETAINED_BYTES)),
    ] {
        exact(
            handoff,
            &format!("/packed_cache_before_cleanup/{field}"),
            expected,
            "browser packed-F16 pre-cleanup cache",
        )?;
    }
    for (field, expected) in [
        ("state", json!("empty")),
        ("cache_ready", json!(false)),
        ("cached_stages", json!(0)),
        ("cached_objects", json!(0)),
        ("cached_tensors", json!(0)),
        ("cached_bytes", json!(0)),
    ] {
        exact(
            handoff,
            &format!("/packed_cache_after_cleanup/{field}"),
            expected,
            "browser packed-F16 post-cleanup cache",
        )?;
    }
    let before_digest = required_str(handoff, "/before_sha256")?;
    let after_digest = required_str(handoff, "/after_sha256")?;
    Sha256Digest::from_hex(before_digest)
        .map_err(|error| format!("browser DMD-to-VAE before digest is invalid: {error}"))?;
    Sha256Digest::from_hex(after_digest)
        .map_err(|error| format!("browser DMD-to-VAE after digest is invalid: {error}"))?;
    if before_digest != after_digest {
        return Err("browser DMD-to-VAE latent digest changed after exact F32 reupload".into());
    }

    let progress = required(evidence, "/progress_events")?
        .as_array()
        .ok_or_else(|| "browser Turbo progress_events is not an array".to_owned())?;
    let dmd_completed = progress
        .iter()
        .filter(|event| {
            event.pointer("/event").and_then(Value::as_str) == Some("stage_completed")
                && event.pointer("/stage").and_then(Value::as_str) == Some("dmd")
        })
        .collect::<Vec<_>>();
    let vae_started = progress
        .iter()
        .filter(|event| {
            event.pointer("/event").and_then(Value::as_str) == Some("stage_started")
                && event.pointer("/stage").and_then(Value::as_str) == Some("vae-decode")
        })
        .collect::<Vec<_>>();
    let ([dmd_completed], [vae_started]) = (dmd_completed.as_slice(), vae_started.as_slice())
    else {
        return Err(
            "browser DMD-to-VAE handoff is not bounded by exactly one DMD completion and VAE start"
                .into(),
        );
    };
    let handoff_at = required(handoff_event, "/at_ms")?
        .as_f64()
        .filter(|value| value.is_finite() && *value > 0.0)
        .ok_or_else(|| "browser DMD-to-VAE handoff timestamp is invalid".to_owned())?;
    let dmd_completed_at = required(dmd_completed, "/at_ms")?
        .as_f64()
        .filter(|value| value.is_finite() && *value > 0.0)
        .ok_or_else(|| "browser DMD completion timestamp is invalid".to_owned())?;
    let vae_started_at = required(vae_started, "/at_ms")?
        .as_f64()
        .filter(|value| value.is_finite() && *value > 0.0)
        .ok_or_else(|| "browser VAE start timestamp is invalid".to_owned())?;
    if !(dmd_completed_at < handoff_at && handoff_at < vae_started_at)
        || handoff_event.pointer("/run_id") != dmd_completed.pointer("/run_id")
        || handoff_event.pointer("/run_id") != vae_started.pointer("/run_id")
    {
        return Err(
            "browser packed-F16 DMD-to-VAE handoff is not ordered and run-bound between DMD completion and VAE start"
                .into(),
        );
    }
    Ok(())
}

fn validate_browser_ordinary_execution_mode(evidence: &Value) -> Result<(), String> {
    exact(
        evidence,
        "/qwen_block0_execution_mode",
        json!(BROWSER_QWEN_BLOCK0_ORDINARY_MODE),
        "browser output-quality Qwen block-0 execution mode",
    )
}

fn validate_browser_surface_gate_evidence(evidence: &Value) -> Result<(), String> {
    let runtime_events = required(evidence, "/runtime_events")?
        .as_array()
        .ok_or_else(|| "browser surface-gate runtime_events is not an array".to_owned())?;
    let progress_events = required(evidence, "/progress_events")?
        .as_array()
        .ok_or_else(|| "browser surface-gate progress_events is not an array".to_owned())?;
    let suspended = exact_runtime_event(runtime_events, "surface_inference_suspended")?;
    let resumed = exact_runtime_event(runtime_events, "surface_inference_resumed")?;
    let run_started = exact_runtime_event(progress_events, "run_started")?;
    let run_completed = exact_runtime_event(progress_events, "run_completed")?;
    let output_ready = required(evidence, "/output_ready")?;
    validate_browser_output_ready_job_id(output_ready, run_started)?;

    for (event, label) in [(suspended, "suspend"), (resumed, "resume")] {
        exact(
            event,
            "/policy",
            json!(BROWSER_SURFACE_INFERENCE_POLICY),
            &format!("browser surface {label} policy"),
        )?;
        if event.pointer("/run_id") != run_started.pointer("/run_id") {
            return Err(format!(
                "browser surface {label} run ID differs from run_started"
            ));
        }
    }
    for (pointer, expected) in [
        ("/primary_window_camera_count", json!(2)),
        ("/saved_camera_state_count", json!(2)),
        ("/previously_active_camera_count", json!(2)),
        ("/inactive_camera_count", json!(2)),
        ("/active_job_count", json!(1)),
        ("/suspended_before_runtime_submit", json!(true)),
        ("/all_primary_window_cameras_inactive", json!(true)),
    ] {
        exact(suspended, pointer, expected, "browser surface suspension")?;
    }
    for (pointer, expected) in [
        ("/terminal", json!("completed")),
        ("/primary_window_camera_count", json!(2)),
        ("/saved_camera_state_count", json!(2)),
        ("/restored_camera_state_count", json!(2)),
        ("/restored_active_camera_count", json!(2)),
        ("/active_job_count", json!(0)),
        ("/resumed_after_runtime_terminal", json!(true)),
        ("/resumed_before_output_ready", json!(true)),
        ("/exact_saved_states_restored", json!(true)),
        ("/all_primary_window_cameras_restored", json!(true)),
    ] {
        exact(resumed, pointer, expected, "browser surface restoration")?;
    }
    if run_completed.pointer("/run_id") != run_started.pointer("/run_id") {
        return Err("browser surface-gate runtime terminal run ID differs from run_started".into());
    }
    let suspended_at = required_positive_timestamp(suspended, "/at_ms", "surface suspend")?;
    let run_started_at =
        required_positive_timestamp(run_started, "/at_ms", "surface-gated run_started")?;
    let run_completed_at =
        required_positive_timestamp(run_completed, "/at_ms", "surface-gated run_completed")?;
    let resumed_at = required_positive_timestamp(resumed, "/at_ms", "surface resume")?;
    let output_ready_at =
        required_positive_timestamp(output_ready, "/at_ms", "surface-gated output_ready")?;
    if !(suspended_at < run_started_at
        && run_completed_at < resumed_at
        && resumed_at < output_ready_at)
    {
        return Err(
            "browser surface gate is not ordered suspend < run_started and run_completed < resume < output_ready"
                .into(),
        );
    }

    let windows = required(evidence, "/surface_texture_gate_windows")?
        .as_array()
        .ok_or_else(|| "browser compact surface gate windows are not an array".to_owned())?;
    let [window] = windows.as_slice() else {
        return Err(format!(
            "browser evidence contains {} compact surface gate windows; expected exactly one",
            windows.len()
        ));
    };
    if window.pointer("/run_id") != run_started.pointer("/run_id") {
        return Err("browser compact surface window run ID differs from run_started".into());
    }
    exact(
        window,
        "/policy",
        json!(BROWSER_SURFACE_INFERENCE_POLICY),
        "browser compact surface-window policy",
    )?;
    exact(
        window,
        "/resume_policy",
        json!(BROWSER_SURFACE_INFERENCE_POLICY),
        "browser compact surface-window resume policy",
    )?;
    exact(
        window,
        "/terminal",
        json!("completed"),
        "browser compact surface-window terminal",
    )?;
    if required_positive_timestamp(window, "/suspended_at_ms", "compact surface suspend")?
        != suspended_at
        || required_positive_timestamp(window, "/resumed_at_ms", "compact surface resume")?
            != resumed_at
    {
        return Err("browser compact surface window is not bound to its runtime events".into());
    }

    let acquisition_count_start =
        required_u64(evidence, "/surface_texture_acquisition_count_start")?;
    let acquisition_count_end = required_u64(evidence, "/surface_texture_acquisition_count_end")?;
    let acquisition_count_at_suspend = required_u64(window, "/acquisition_count_at_suspend")?;
    let acquisition_count_at_resume = required_u64(window, "/acquisition_count_at_resume")?;
    if acquisition_count_end < acquisition_count_start
        || acquisition_count_at_suspend < acquisition_count_start
        || acquisition_count_at_resume != acquisition_count_at_suspend
        || acquisition_count_at_resume > acquisition_count_end
        || required_u64(window, "/gated_call_count")? != 0
    {
        return Err(
            "browser called GPUCanvasContext.getCurrentTexture while the request surface gate was active"
                .into(),
        );
    }
    let violations = required(evidence, "/surface_texture_gate_violation_calls")?
        .as_array()
        .ok_or_else(|| "browser surface gate violation calls are not an array".to_owned())?;
    if !violations.is_empty()
        || required_u64(evidence, "/surface_texture_gate_windows_overflow")? != 0
        || required_u64(evidence, "/surface_texture_gate_overlap_count")? != 0
        || required_u64(
            evidence,
            "/surface_texture_gate_violation_calls_overflow_start",
        )? != 0
        || required_u64(
            evidence,
            "/surface_texture_gate_violation_calls_overflow_end",
        )? != 0
    {
        return Err(
            "browser compact surface gate evidence overflowed or recorded a gated call".into(),
        );
    }

    let pre_request = required(window, "/pre_request_acquisition")?;
    exact(
        pre_request,
        "/succeeded",
        json!(true),
        "browser pre-request surface acquisition",
    )?;
    exact(
        pre_request,
        "/canvas_id",
        json!("burn-image"),
        "browser pre-request surface acquisition",
    )?;
    if required_positive_timestamp(pre_request, "/at_ms", "pre-request surface acquisition")?
        >= suspended_at
        || required_u64(pre_request, "/call_index")? == 0
        || required_u64(pre_request, "/call_index")? > acquisition_count_at_suspend
    {
        return Err("browser lacks a real successful pre-request surface acquisition".into());
    }
    let post_resume = required(window, "/first_successful_post_resume_acquisition")?;
    exact(
        post_resume,
        "/succeeded",
        json!(true),
        "browser post-resume surface acquisition",
    )?;
    exact(
        post_resume,
        "/canvas_id",
        json!("burn-image"),
        "browser post-resume surface acquisition",
    )?;
    if required_positive_timestamp(post_resume, "/at_ms", "post-resume surface acquisition")?
        <= resumed_at
        || required_u64(post_resume, "/call_index")? <= acquisition_count_at_resume
        || required_u64(post_resume, "/call_index")? > acquisition_count_end
    {
        return Err("browser lacks a successful post-resume surface acquisition".into());
    }
    if evidence
        .pointer("/surface_inference_state_after_request")
        .is_some_and(|value| !value.is_null())
        || evidence
            .pointer("/active_surface_gate_after_request")
            .is_some_and(|value| !value.is_null())
    {
        return Err("browser left the surface gate or page pause attribute active".into());
    }
    Ok(())
}

fn validate_browser_output_ready_job_id(
    output_ready: &Value,
    run_started: &Value,
) -> Result<(), String> {
    let raw_job_id = required(output_ready, "/job_id")?.as_str().ok_or_else(|| {
        "browser output_ready job_id is not a canonical u64 decimal string".to_owned()
    })?;
    let job_id = raw_job_id
        .parse::<u64>()
        .ok()
        .filter(|parsed| !raw_job_id.is_empty() && parsed.to_string().as_str() == raw_job_id)
        .ok_or_else(|| {
            "browser output_ready job_id is not a canonical u64 decimal string".to_owned()
        })?;
    let run_id = required(run_started, "/run_id")?
        .as_u64()
        .filter(|run_id| *run_id <= JAVASCRIPT_MAX_SAFE_INTEGER_U64)
        .ok_or_else(|| {
            "browser run_started run_id is not a nonnegative safe JSON integer".to_owned()
        })?;
    if job_id != run_id || raw_job_id != run_id.to_string() {
        return Err(
            "browser output_ready job_id differs from the exact decimal representation of run_started run_id"
                .into(),
        );
    }
    Ok(())
}

fn required_positive_timestamp(value: &Value, pointer: &str, label: &str) -> Result<f64, String> {
    required(value, pointer)?
        .as_f64()
        .filter(|timestamp| timestamp.is_finite() && *timestamp > 0.0)
        .ok_or_else(|| format!("{label} timestamp is missing or invalid"))
}

fn exact_runtime_event<'a>(events: &'a [Value], event_name: &str) -> Result<&'a Value, String> {
    let matches = events
        .iter()
        .filter(|event| event.get("event").and_then(Value::as_str) == Some(event_name))
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [event] => Ok(*event),
        _ => Err(format!(
            "browser evidence contains {} {event_name:?} runtime events; expected exactly one",
            matches.len()
        )),
    }
}

fn validate_request_identity(
    request: &NativeOutputQualificationRequestIdentity,
) -> Result<(), String> {
    let variant = request_variant(request)?;
    let descriptor = boogu_model_descriptor(variant);
    if request.model != descriptor.id.as_str()
        || request.model_revision != descriptor.revision
        || request.prompt.trim().is_empty()
        || request.steps != 4
        || request.guidance_scale != 1.0
        || request.batch_size != 1
    {
        return Err(format!("invalid released request identity: {request:?}"));
    }
    let dimensions = Dimensions::new(request.width, request.height)
        .map_err(|error| format!("invalid request dimensions: {error}"))?;
    descriptor
        .capabilities
        .dimensions
        .supports(dimensions)
        .map_err(|error| format!("unsupported released output dimensions: {error}"))?;
    if variant == BooguVariant::Image01EditTurbo1k5
        && !BOOGU_1K5_OUTPUT_PRESETS.contains(&(request.width, request.height))
    {
        return Err(format!(
            "Edit-Turbo 1.5K output {}x{} is not an official preset",
            request.width, request.height
        ));
    }
    match (variant, &request.source, request.task.as_str()) {
        (BooguVariant::Image01Turbo, None, "generate") => {}
        (
            BooguVariant::Image01EditTurbo | BooguVariant::Image01EditTurbo1k5,
            Some(source),
            "edit",
        ) => {
            let digest = Sha256Digest::from_hex(&source.sha256)
                .map_err(|error| format!("invalid source SHA-256: {error}"))?;
            if source.bytes == 0 || source.bytes > MAX_EDIT_SOURCE_BYTES {
                return Err(format!(
                    "request source byte identity {} is outside 1..={MAX_EDIT_SOURCE_BYTES}",
                    source.bytes
                ));
            }
            let bytes = read_bounded_edit_source(&source.path)?;
            if bytes.len() as u64 != source.bytes || Sha256Digest::calculate(&bytes) != digest {
                return Err("request source file no longer matches its exact identity".into());
            }
        }
        _ => return Err("request task/source does not match released variant".into()),
    }
    let published = canonical_published_bundle(variant, BooguStorageProfile::F16QwenVisionF32)
        .ok_or_else(|| "request variant has no canonical production artifact".to_owned())?;
    validate_canonical_release_artifact_digest(
        variant,
        BooguStorageProfile::F16QwenVisionF32,
        Sha256Digest::from_hex(published.content_digest).map_err(validation_string)?,
    )
    .map_err(|error| error.to_string())
}

fn request_variant(
    request: &NativeOutputQualificationRequestIdentity,
) -> Result<BooguVariant, String> {
    match request.variant.as_str() {
        "turbo" => Ok(BooguVariant::Image01Turbo),
        "edit-turbo" => Ok(BooguVariant::Image01EditTurbo),
        "edit-turbo-1k5" => Ok(BooguVariant::Image01EditTurbo1k5),
        value => Err(format!("unsupported request variant {value:?}")),
    }
}

fn request_identities_match(
    native: &NativeOutputQualificationRequestIdentity,
    browser: &NativeOutputQualificationRequestIdentity,
) -> bool {
    native.variant == browser.variant
        && native.task == browser.task
        && native.model == browser.model
        && native.model_revision == browser.model_revision
        && native.prompt == browser.prompt
        && native.seed == browser.seed
        && native.width == browser.width
        && native.height == browser.height
        && native.steps == browser.steps
        && native.guidance_scale == browser.guidance_scale
        && native.batch_size == browser.batch_size
        && match (&native.source, &browser.source) {
            (None, None) => true,
            (Some(native), Some(browser)) => {
                native.bytes == browser.bytes
                    && native.sha256 == browser.sha256
                    && native.width == browser.width
                    && native.height == browser.height
            }
            _ => false,
        }
}

fn native_bundles(
    report: &Value,
    variant: BooguVariant,
) -> Result<BTreeMap<String, String>, String> {
    let mut bundles = BTreeMap::new();
    for role in ["parent", "qwen", "vae"] {
        bundles.insert(
            required_str(report, &format!("/artifacts/{role}_bundle"))?.into(),
            required_str(report, &format!("/artifacts/{role}_content_digest"))?.into(),
        );
    }
    if bundles.len() != 3 {
        return Err("native report does not authenticate three unique modular bundles".into());
    }
    let pipeline_root = PathBuf::from(required_str(report, "/artifacts/pipeline_root")?);
    let parent = VerifiedArtifactDirectory::open(&pipeline_root)
        .map_err(|error| format!("reopen sealed native parent manifest: {error}"))?;
    let manifest = parent.manifest();
    let published = canonical_published_bundle(variant, BooguStorageProfile::F16QwenVisionF32)
        .ok_or_else(|| "native variant has no canonical production bundle".to_owned())?;
    if manifest.bundle.as_str() != published.bundle_id
        || manifest
            .content_digest
            .map(|digest| digest.to_hex())
            .as_deref()
            != Some(published.content_digest)
        || manifest.dependencies.len() != 2
    {
        return Err("local parent manifest is not the compiled canonical composition".into());
    }
    let sibling_root = pipeline_root
        .parent()
        .ok_or_else(|| "native parent has no sibling-bundle directory".to_owned())?;
    for dependency in &manifest.dependencies {
        let bundle = dependency.bundle.as_str();
        let dependency_digest = dependency.content_digest.to_hex();
        if !matches!(dependency.role.as_str(), "qwen" | "vae")
            || bundles.get(bundle).map(String::as_str) != Some(dependency_digest.as_str())
        {
            return Err(format!(
                "reported component does not match sealed parent dependency {}",
                dependency.role
            ));
        }
        let resolved = VerifiedArtifactDirectory::open(sibling_root.join(bundle))
            .map_err(|error| format!("reopen sealed dependency {bundle}: {error}"))?;
        dependency
            .validate_resolved_manifest(resolved.manifest())
            .map_err(|error| format!("validate sealed dependency {bundle}: {error}"))?;
    }
    Ok(bundles)
}

fn browser_bundles(
    evidence: &Value,
    request: &NativeOutputQualificationRequestIdentity,
) -> Result<BTreeMap<String, String>, String> {
    exact(
        evidence,
        "/modular_artifact_transport/validated",
        json!(true),
        "browser modular transport",
    )?;
    let list = required(evidence, "/modular_artifact_transport/bundles")?
        .as_array()
        .ok_or_else(|| "browser bundle list is not an array".to_owned())?;
    let mut bundles = BTreeMap::new();
    for bundle in list {
        if bundles
            .insert(
                required_str(bundle, "/bundle")?.into(),
                required_str(bundle, "/content_digest")?.into(),
            )
            .is_some()
        {
            return Err("browser modular transport repeats a bundle".into());
        }
    }
    if bundles.len() != 3 {
        return Err("browser report does not authenticate three modular bundles".into());
    }
    let published = canonical_published_bundle(
        request_variant(request)?,
        BooguStorageProfile::F16QwenVisionF32,
    )
    .ok_or_else(|| "browser request has no canonical production bundle".to_owned())?;
    if bundles.get(published.bundle_id).map(String::as_str) != Some(published.content_digest) {
        return Err("browser modular transport does not include the canonical parent".into());
    }
    Ok(bundles)
}

fn decode_normalized_rgb8(
    path: &Path,
    request: &NativeOutputQualificationRequestIdentity,
) -> Result<Vec<u8>, String> {
    let reader = ImageReader::open(path)
        .map_err(|error| format!("open PNG {}: {error}", path.display()))?
        .with_guessed_format()
        .map_err(|error| format!("identify PNG {}: {error}", path.display()))?;
    if reader.format() != Some(image::ImageFormat::Png) {
        return Err(format!("output {} is not a PNG", path.display()));
    }
    let decoded = reader
        .decode()
        .map_err(|error| format!("decode PNG {}: {error}", path.display()))?;
    if (decoded.width(), decoded.height()) != (request.width, request.height) {
        return Err(format!(
            "PNG {} dimensions do not match request",
            path.display()
        ));
    }
    if !matches!(decoded.color(), ColorType::Rgb8 | ColorType::Rgba8) {
        return Err(format!("PNG {} is not RGB8/RGBA8", path.display()));
    }
    let rgba = decoded.to_rgba8();
    let mut rgb =
        Vec::with_capacity((u64::from(request.width) * u64::from(request.height) * 3) as usize);
    for pixel in rgba.pixels() {
        if pixel[3] != u8::MAX {
            return Err(format!("PNG {} contains non-opaque alpha", path.display()));
        }
        rgb.extend_from_slice(&pixel.0[..3]);
    }
    Ok(rgb)
}

fn compare_rgb(
    actual: &[u8],
    expected: &[u8],
    width: usize,
    height: usize,
) -> Result<RgbMetrics, String> {
    let length = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| "RGB dimensions overflow".to_owned())?;
    if length == 0 || actual.len() != length || expected.len() != length {
        return Err("RGB byte lengths do not match nonzero dimensions".into());
    }
    let mut max_abs = 0;
    let mut exact_count = 0_u64;
    let mut sum_abs = 0.0_f64;
    let mut sum_squared = 0.0_f64;
    for (&actual, &expected) in actual.iter().zip(expected) {
        let difference = actual.abs_diff(expected);
        max_abs = max_abs.max(difference);
        exact_count += u64::from(difference == 0);
        sum_abs += f64::from(difference);
        sum_squared += f64::from(difference).powi(2);
    }
    let count = length as f64;
    let rmse = (sum_squared / count).sqrt();
    Ok(RgbMetrics {
        width,
        height,
        max_abs_u8: max_abs,
        mean_abs_u8: finite_f32(sum_abs / count, "mean abs")?,
        rmse_u8: finite_f32(rmse, "RMSE")?,
        psnr_db: if rmse == 0.0 {
            100.0
        } else {
            finite_f32(20.0 * (255.0 / rmse).log10(), "PSNR")?
        },
        mean_block_ssim_8x8: mean_block_ssim_8x8(actual, expected, width, height)?,
        exact_fraction: finite_f32(exact_count as f64 / count, "exact fraction")?,
    })
}

fn mean_block_ssim_8x8(
    actual: &[u8],
    expected: &[u8],
    width: usize,
    height: usize,
) -> Result<f32, String> {
    let c1 = (0.01_f64 * 255.0).powi(2);
    let c2 = (0.03_f64 * 255.0).powi(2);
    let mut total = 0.0;
    let mut blocks = 0_u64;
    for top in (0..height).step_by(8) {
        for left in (0..width).step_by(8) {
            let bottom = (top + 8).min(height);
            let right = (left + 8).min(width);
            for channel in 0..3 {
                let count = ((bottom - top) * (right - left)) as f64;
                let mut mean_a = 0.0;
                let mut mean_b = 0.0;
                for y in top..bottom {
                    for x in left..right {
                        let index = (y * width + x) * 3 + channel;
                        mean_a += f64::from(actual[index]);
                        mean_b += f64::from(expected[index]);
                    }
                }
                mean_a /= count;
                mean_b /= count;
                let (mut variance_a, mut variance_b, mut covariance) = (0.0, 0.0, 0.0);
                for y in top..bottom {
                    for x in left..right {
                        let index = (y * width + x) * 3 + channel;
                        let delta_a = f64::from(actual[index]) - mean_a;
                        let delta_b = f64::from(expected[index]) - mean_b;
                        variance_a += delta_a * delta_a;
                        variance_b += delta_b * delta_b;
                        covariance += delta_a * delta_b;
                    }
                }
                variance_a /= count;
                variance_b /= count;
                covariance /= count;
                total += ((2.0 * mean_a * mean_b + c1) * (2.0 * covariance + c2))
                    / ((mean_a.powi(2) + mean_b.powi(2) + c1) * (variance_a + variance_b + c2));
                blocks += 1;
            }
        }
    }
    if blocks == 0 {
        return Err("SSIM has no blocks".into());
    }
    finite_f32(total / blocks as f64, "SSIM")
}

fn validate_file(path: &Path, digest: &str, size: u64, label: &str) -> Result<String, String> {
    let bytes = read(path, label)?;
    let actual = Sha256Digest::calculate(&bytes).to_hex();
    if bytes.len() as u64 != size || actual != digest || Sha256Digest::from_hex(digest).is_err() {
        return Err(format!("{label} no longer matches reported size/SHA-256"));
    }
    Ok(actual)
}

fn positive_sub_ceiling(value: u64, runtime: &str) -> Result<u64, String> {
    if value == 0 || value >= DEVICE_CEILING_BYTES {
        Err(format!(
            "{runtime} framebuffer peak {value} is not positive and below {DEVICE_CEILING_BYTES}"
        ))
    } else {
        Ok(value)
    }
}

fn read(path: &Path, label: &str) -> Result<Vec<u8>, String> {
    fs::read(path).map_err(|error| format!("read {label} {}: {error}", path.display()))
}

fn read_bounded_edit_source(path: &Path) -> Result<Vec<u8>, String> {
    let file = fs::File::open(path)
        .map_err(|error| format!("open request source image {}: {error}", path.display()))?;
    let metadata = file
        .metadata()
        .map_err(|error| format!("inspect request source image {}: {error}", path.display()))?;
    if !metadata.is_file() {
        return Err(format!(
            "request source image {} is not a regular file",
            path.display()
        ));
    }
    read_bounded_edit_source_handle(file, metadata.len(), MAX_EDIT_SOURCE_BYTES, path)
}

fn read_bounded_edit_source_handle(
    file: fs::File,
    preflight_bytes: u64,
    max_bytes: u64,
    path: &Path,
) -> Result<Vec<u8>, String> {
    if preflight_bytes == 0 {
        return Err(format!("request source image {} is empty", path.display()));
    }
    if preflight_bytes > max_bytes {
        return Err(format!(
            "request source image {} is {preflight_bytes} bytes; limit is {max_bytes} bytes",
            path.display()
        ));
    }
    let capacity = usize::try_from(preflight_bytes)
        .map_err(|_| "request source size does not fit host memory".to_owned())?;
    let mut bytes = Vec::with_capacity(capacity);
    file.take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|error| format!("read request source image {}: {error}", path.display()))?;
    if bytes.is_empty() {
        return Err(format!(
            "request source image {} became empty while reading",
            path.display()
        ));
    }
    if bytes.len() as u64 > max_bytes {
        return Err(format!(
            "request source image {} grew beyond the {max_bytes}-byte limit while reading",
            path.display()
        ));
    }
    Ok(bytes)
}

fn required<'a>(value: &'a Value, pointer: &str) -> Result<&'a Value, String> {
    value
        .pointer(pointer)
        .ok_or_else(|| format!("required value is missing at {pointer}"))
}

fn required_str<'a>(value: &'a Value, pointer: &str) -> Result<&'a str, String> {
    required(value, pointer)?
        .as_str()
        .ok_or_else(|| format!("required string is missing at {pointer}"))
}

fn required_u64(value: &Value, pointer: &str) -> Result<u64, String> {
    required(value, pointer)?
        .as_u64()
        .ok_or_else(|| format!("required unsigned integer is missing at {pointer}"))
}

fn exact(value: &Value, pointer: &str, expected: Value, label: &str) -> Result<(), String> {
    let actual = required(value, pointer)?;
    if actual == &expected {
        Ok(())
    } else {
        Err(format!("{label} is {actual}, expected {expected}"))
    }
}

fn finite_f32(value: f64, label: &str) -> Result<f32, String> {
    let value = value as f32;
    value
        .is_finite()
        .then_some(value)
        .ok_or_else(|| format!("{label} is non-finite"))
}

fn validation_string(error: ValidationError) -> String {
    error.to_string()
}

fn canonical_or_original(path: &Path) -> PathBuf {
    fs::canonicalize(path).unwrap_or_else(|_| path.to_owned())
}

fn write_json_atomically(path: &Path, value: &Value) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("json.tmp");
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    fs::write(&temporary, bytes)?;
    fs::rename(temporary, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    #[test]
    fn exact_cli_paths_are_required_correctness() {
        let args = parse_args([
            "--native-report".into(),
            "native.json".into(),
            "--browser-report".into(),
            "browser.json".into(),
            "--output".into(),
            "output.json".into(),
        ])
        .unwrap();
        assert_eq!(args.native_report, PathBuf::from("native.json"));
        assert!(parse_args(["--native-report".into(), "native.json".into()]).is_err());
    }

    #[test]
    fn identical_rgb_is_exact_with_capped_psnr_correctness() {
        let rgb = vec![127; 8 * 8 * 3];
        let metrics = compare_rgb(&rgb, &rgb, 8, 8).unwrap();
        assert_eq!(metrics.max_abs_u8, 0);
        assert_eq!(metrics.psnr_db, 100.0);
        assert_eq!(metrics.mean_block_ssim_8x8, 1.0);
        assert_eq!(metrics.exact_fraction, 1.0);
    }

    #[test]
    fn image_gate_detects_material_drift_correctness() {
        let black = vec![0; 16 * 16 * 3];
        let white = vec![255; 16 * 16 * 3];
        let metrics = compare_rgb(&black, &white, 16, 16).unwrap();
        assert!(metrics.psnr_db < MINIMUM_PSNR_DB);
        assert!(metrics.mean_block_ssim_8x8 < MINIMUM_MEAN_BLOCK_SSIM_8X8);
        assert!(compare_rgb(&[0; 3], &[0; 6], 1, 1).is_err());
    }

    #[test]
    fn browser_quality_gate_requires_request_scoped_surface_evidence_correctness() {
        let evidence = json!({
            "runtime_events": [
                {
                    "at_ms": 100.0,
                    "event": "surface_inference_suspended",
                    "run_id": 7,
                    "policy": BROWSER_SURFACE_INFERENCE_POLICY,
                    "primary_window_camera_count": 2,
                    "saved_camera_state_count": 2,
                    "previously_active_camera_count": 2,
                    "inactive_camera_count": 2,
                    "active_job_count": 1,
                    "suspended_before_runtime_submit": true,
                    "all_primary_window_cameras_inactive": true,
                },
                {
                    "at_ms": 400.0,
                    "event": "surface_inference_resumed",
                    "run_id": 7,
                    "policy": BROWSER_SURFACE_INFERENCE_POLICY,
                    "terminal": "completed",
                    "primary_window_camera_count": 2,
                    "saved_camera_state_count": 2,
                    "restored_camera_state_count": 2,
                    "restored_active_camera_count": 2,
                    "active_job_count": 0,
                    "resumed_after_runtime_terminal": true,
                    "resumed_before_output_ready": true,
                    "exact_saved_states_restored": true,
                    "all_primary_window_cameras_restored": true,
                },
            ],
            "progress_events": [
                { "at_ms": 200.0, "event": "run_started", "run_id": 7 },
                { "at_ms": 300.0, "event": "run_completed", "run_id": 7 },
            ],
            "output_ready": { "at_ms": 500.0, "job_id": "7" },
            "surface_texture_acquisition_count_start": 10,
            "surface_texture_acquisition_count_end": 11,
            "surface_texture_gate_windows": [{
                "run_id": 7,
                "policy": BROWSER_SURFACE_INFERENCE_POLICY,
                "resume_policy": BROWSER_SURFACE_INFERENCE_POLICY,
                "terminal": "completed",
                "suspended_at_ms": 100.0,
                "resumed_at_ms": 400.0,
                "acquisition_count_at_suspend": 10,
                "acquisition_count_at_resume": 10,
                "gated_call_count": 0,
                "pre_request_acquisition": {
                    "at_ms": 90.0,
                    "call_index": 10,
                    "canvas_id": "burn-image",
                    "succeeded": true,
                },
                "first_successful_post_resume_acquisition": {
                    "at_ms": 450.0,
                    "call_index": 11,
                    "canvas_id": "burn-image",
                    "succeeded": true,
                },
            }],
            "surface_texture_gate_violation_calls": [],
            "surface_texture_gate_windows_overflow": 0,
            "surface_texture_gate_overlap_count": 0,
            "surface_texture_gate_violation_calls_overflow_start": 0,
            "surface_texture_gate_violation_calls_overflow_end": 0,
            "surface_inference_state_after_request": null,
            "active_surface_gate_after_request": null,
        });
        validate_browser_surface_gate_evidence(&evidence).unwrap();

        let mut ungated = evidence.clone();
        ungated["runtime_events"] = json!([]);
        assert!(
            validate_browser_surface_gate_evidence(&ungated)
                .unwrap_err()
                .contains("expected exactly one")
        );
        let mut acquired_while_gated = evidence.clone();
        acquired_while_gated["surface_texture_gate_windows"][0]["acquisition_count_at_resume"] =
            json!(11);
        assert!(
            validate_browser_surface_gate_evidence(&acquired_while_gated)
                .unwrap_err()
                .contains("getCurrentTexture")
        );
        let mut early_resume = evidence;
        early_resume["runtime_events"][1]["at_ms"] = json!(250.0);
        assert!(
            validate_browser_surface_gate_evidence(&early_resume)
                .unwrap_err()
                .contains("not ordered")
        );
    }

    #[test]
    fn browser_quality_gate_requires_canonical_run_bound_output_job_id_correctness() {
        let run_started = json!({ "run_id": 7 });
        validate_browser_output_ready_job_id(&json!({ "job_id": "7" }), &run_started).unwrap();
        validate_browser_output_ready_job_id(
            &json!({ "job_id": "9007199254740991" }),
            &json!({ "run_id": JAVASCRIPT_MAX_SAFE_INTEGER_U64 }),
        )
        .unwrap();
        assert!(
            validate_browser_output_ready_job_id(
                &json!({ "job_id": "18446744073709551615" }),
                &json!({ "run_id": 7 }),
            )
            .unwrap_err()
            .contains("exact decimal representation")
        );

        for invalid in [
            json!({ "job_id": 7 }),
            json!({ "job_id": "07" }),
            json!({ "job_id": "18446744073709551616" }),
        ] {
            assert!(
                validate_browser_output_ready_job_id(&invalid, &run_started)
                    .unwrap_err()
                    .contains("canonical u64 decimal string")
            );
        }

        assert!(
            validate_browser_output_ready_job_id(&json!({ "job_id": "8" }), &run_started)
                .unwrap_err()
                .contains("exact decimal representation")
        );
        assert!(
            validate_browser_output_ready_job_id(&json!({}), &run_started)
                .unwrap_err()
                .contains("required value is missing at /job_id")
        );
        assert!(
            validate_browser_output_ready_job_id(
                &json!({ "job_id": "9007199254740992" }),
                &json!({ "run_id": 9_007_199_254_740_992_u64 }),
            )
            .unwrap_err()
            .contains("nonnegative safe JSON integer")
        );
    }

    #[test]
    fn browser_quality_gate_requires_exact_packed_f16_lifecycle_correctness() {
        let zero_traffic = json!({
            "object_reads": 0,
            "object_read_bytes": 0,
            "range_reads": 0,
            "range_read_bytes": 0,
            "verified_objects": 0,
            "cache_lookups": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_read_bytes": 0,
            "network_requests": 0,
            "network_response_bytes": 0,
            "cache_writes": 0,
            "cache_write_bytes": 0,
            "cache_evictions": 0,
            "cache_evicted_entries": 0,
            "cache_invalid_entries": 0,
            "integrity_refetches": 0,
        });
        let lifecycle = json!({
            "cache_state": "ready",
            "cache_ready": true,
            "cached_stages": TURBO_PACKED_F16_STAGES,
            "cached_objects": TURBO_PACKED_F16_OBJECTS,
            "cached_tensors": TURBO_PACKED_F16_TENSORS,
            "cached_bytes": TURBO_PACKED_F16_RETAINED_BYTES,
            "authenticated_artifact_bytes": TURBO_PACKED_F16_ARTIFACT_BYTES,
            "packed_upload_bytes": TURBO_PACKED_F16_RETAINED_BYTES,
            "stage_materializations": TURBO_PACKED_F16_STAGE_MATERIALIZATIONS,
            "object_unpacks": TURBO_PACKED_F16_OBJECT_UNPACKS,
            "packed_read_bytes": TURBO_PACKED_F16_REQUEST_READ_BYTES,
            "f32_write_bytes": TURBO_PACKED_F16_REQUEST_WRITE_BYTES,
            "preload_attempt_count": 1,
            "failure_count": 0,
            "dmd_artifact_traffic": zero_traffic,
            "synchronization_pending": false,
            "matches_plan": true,
        });
        let handoff = json!({
            "at_ms": 8_830,
            "event": "packed_f16_dmd_vae_handoff",
            "run_id": 7,
            "report": {
                "policy": BROWSER_PACKED_F16_DMD_VAE_HANDOFF_POLICY,
                "next_request_rehydration_policy":
                    BROWSER_PACKED_F16_NEXT_REQUEST_REHYDRATION_POLICY,
                "shape": [1, 16, 128, 128],
                "dtype": "f32",
                "element_count": 262_144,
                "payload_bytes": 1_048_576,
                "device_to_host_readback_bytes": 2_097_152,
                "host_to_device_upload_bytes": 1_048_576,
                "total_transfer_bytes": 3_145_728,
                "before_sha256": "7".repeat(64),
                "after_sha256": "7".repeat(64),
                "all_finite": true,
                "not_all_zero": true,
                "digest_matches": true,
                "wrapper_cached_stages_before_clear": 0,
                "wrapper_cached_stages_after_clear": 0,
                "synchronization_pending_before_cleanup": false,
                "synchronization_pending_after_cleanup": false,
                "rope_cache_cleared": true,
                "cleanup_completed": true,
                "packed_cache_before_cleanup": {
                    "state": "ready",
                    "cache_ready": true,
                    "cached_stages": TURBO_PACKED_F16_STAGES,
                    "cached_objects": TURBO_PACKED_F16_OBJECTS,
                    "cached_tensors": TURBO_PACKED_F16_TENSORS,
                    "cached_bytes": TURBO_PACKED_F16_RETAINED_BYTES,
                },
                "packed_cache_after_cleanup": {
                    "state": "empty",
                    "cache_ready": false,
                    "cached_stages": 0,
                    "cached_objects": 0,
                    "cached_tensors": 0,
                    "cached_bytes": 0,
                },
                "preload_attempt_count": 1,
                "expected_next_request_preload_attempt_count": 2,
            },
        });
        let evidence = json!({
            "qwen_block0_execution_mode": BROWSER_QWEN_BLOCK0_ORDINARY_MODE,
            "runtime_events": [
                {
                    "event": "packed_f16_resource_plan",
                    "authenticated_artifact_bytes": TURBO_PACKED_F16_ARTIFACT_BYTES,
                    "canonical_compact_f16_payload_bytes": TURBO_PACKED_F16_COMPACT_BYTES,
                    "retained_packed_f16_denoiser_bytes": TURBO_PACKED_F16_RETAINED_BYTES,
                    "inserted_padding_elements": 7_264_u64,
                    "padded_f16_elements": 9_935_005_312_u64,
                    "expected_stage_count": TURBO_PACKED_F16_STAGES,
                    "expected_object_count": TURBO_PACKED_F16_OBJECTS,
                    "expected_tensor_count": TURBO_PACKED_F16_TENSORS,
                    "max_packed_stage_bytes": 876_827_328_u64,
                    "max_materialized_stage_f32_bytes": 1_753_654_656_u64,
                    "max_packed_object_bytes": 254_251_904_u64,
                    "max_materialized_object_f32_bytes": 508_503_808_u64,
                    "materialized_f32_bytes_per_dmd_step": 39_740_021_248_u64,
                    "preload_workspace_bytes": 2_434_252_800_u64,
                    "preload_peak_bytes": 22_304_263_424_u64,
                    "activation_reserve_bytes": 4_868_505_600_u64,
                    "conservative_planned_device_bytes": 26_492_170_880_u64,
                    "strict_device_cap_bytes": DEVICE_CEILING_BYTES,
                    "expected_stage_materializations_per_request":
                        TURBO_PACKED_F16_STAGE_MATERIALIZATIONS,
                    "expected_object_unpacks_per_request": TURBO_PACKED_F16_OBJECT_UNPACKS,
                    "expected_packed_read_bytes_per_request":
                        TURBO_PACKED_F16_REQUEST_READ_BYTES,
                    "expected_f32_write_bytes_per_request":
                        TURBO_PACKED_F16_REQUEST_WRITE_BYTES,
                    "on_device_quantized_execution_claimed": false,
                },
                {
                    "event": "packed_f16_denoiser_preload",
                    "cached_stages": TURBO_PACKED_F16_STAGES,
                    "cached_objects": TURBO_PACKED_F16_OBJECTS,
                    "cached_tensors": TURBO_PACKED_F16_TENSORS,
                    "cached_bytes": TURBO_PACKED_F16_RETAINED_BYTES,
                    "previous_preload_attempt_count": 0,
                    "preload_attempt_count": 1,
                    "request_scoped_rehydration": false,
                    "rehydration_policy":
                        BROWSER_PACKED_F16_NEXT_REQUEST_REHYDRATION_POLICY,
                },
                {
                    "event": "packed_f16_denoiser_lifecycle",
                    "lifecycle": lifecycle.clone(),
                },
                handoff.clone(),
                {
                    "event": "ready",
                    "block0_execution_mode": BROWSER_QWEN_BLOCK0_ORDINARY_MODE,
                },
            ],
            "progress_events": [
                { "at_ms": 8_000, "event": "stage_completed", "run_id": 7, "stage": "dmd" },
                { "at_ms": 8_840, "event": "stage_started", "run_id": 7, "stage": "vae-decode" },
            ],
            "packed_f16_denoiser_lifecycle": lifecycle,
            "packed_f16_dmd_vae_handoff": handoff,
        });
        validate_turbo_packed_f16_evidence(&evidence).unwrap();

        let mut lifecycle_corrupt = evidence.clone();
        lifecycle_corrupt["runtime_events"][2]["lifecycle"]["object_unpacks"] = json!(423);
        assert!(
            validate_turbo_packed_f16_evidence(&lifecycle_corrupt)
                .unwrap_err()
                .contains("differs from its exact runtime event")
        );

        let mut stale_cache = evidence.clone();
        for pointer in [
            "/runtime_events/3/report/packed_cache_after_cleanup/cached_bytes",
            "/packed_f16_dmd_vae_handoff/report/packed_cache_after_cleanup/cached_bytes",
        ] {
            *stale_cache.pointer_mut(pointer).unwrap() = json!(TURBO_PACKED_F16_RETAINED_BYTES);
        }
        assert!(
            validate_turbo_packed_f16_evidence(&stale_cache)
                .unwrap_err()
                .contains("post-cleanup cache")
        );

        let mut digest_drift = evidence.clone();
        for pointer in [
            "/runtime_events/3/report/after_sha256",
            "/packed_f16_dmd_vae_handoff/report/after_sha256",
        ] {
            *digest_drift.pointer_mut(pointer).unwrap() = json!("8".repeat(64));
        }
        assert!(
            validate_turbo_packed_f16_evidence(&digest_drift)
                .unwrap_err()
                .contains("digest changed")
        );
    }

    #[test]
    fn browser_quality_gate_rejects_nonordinary_block0_execution_correctness() {
        let ordinary = json!({
            "qwen_block0_execution_mode": BROWSER_QWEN_BLOCK0_ORDINARY_MODE,
        });
        validate_browser_ordinary_execution_mode(&ordinary).unwrap();

        let diagnostic = json!({"qwen_block0_execution_mode": "serialized-diagnostic"});
        assert!(
            validate_browser_ordinary_execution_mode(&diagnostic)
                .unwrap_err()
                .contains("browser output-quality Qwen block-0 execution mode")
        );
        assert!(
            validate_browser_ordinary_execution_mode(&json!({}))
                .unwrap_err()
                .contains("/qwen_block0_execution_mode")
        );

        let mut turbo = json!({
            "qwen_block0_execution_mode": BROWSER_QWEN_BLOCK0_ORDINARY_MODE,
            "runtime_events": [{
                "event": "ready",
                "block0_execution_mode": BROWSER_QWEN_BLOCK0_ORDINARY_MODE,
            }],
        });
        let diagnostic_ready = json!({
            "event": "ready",
            "block0_execution_mode": "serialized-diagnostic",
        });
        turbo["runtime_events"][0] = diagnostic_ready;
        assert!(
            validate_turbo_packed_f16_evidence(&turbo)
                .unwrap_err()
                .contains("ready-event Qwen block-0 execution mode")
        );
    }

    #[test]
    fn bounded_edit_source_rejects_malformed_and_oversize_correctness() {
        let directory = tempfile::tempdir().unwrap();
        assert!(
            read_bounded_edit_source(directory.path())
                .unwrap_err()
                .contains("not a regular file")
        );

        let empty = tempfile::NamedTempFile::new().unwrap();
        assert!(
            read_bounded_edit_source(empty.path())
                .unwrap_err()
                .contains("is empty")
        );

        let mut oversized = tempfile::NamedTempFile::new().unwrap();
        oversized.write_all(b"123456").unwrap();
        oversized.flush().unwrap();
        let file = fs::File::open(oversized.path()).unwrap();
        let preflight_bytes = file.metadata().unwrap().len();
        assert!(
            read_bounded_edit_source_handle(file, preflight_bytes, 5, oversized.path())
                .unwrap_err()
                .contains("6 bytes; limit is 5 bytes")
        );
    }
}
