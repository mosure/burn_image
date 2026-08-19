// Opt-in, real-hardware rendered-window smoke for the app-only Bevy WebGPU shell.
//
// Build externally before running the hardware path:
//   cargo build -p bevy_burn_image --target wasm32-unknown-unknown --profile wasm-release \
//     --no-default-features --features web --locked --lib
//   wasm-bindgen --target web --out-dir crates/bevy_image/www/out \
//     --out-name bevy_burn_image target/wasm32-unknown-unknown/wasm-release/bevy_burn_image.wasm
//   install -m 0644 crates/bevy_image/www/burn-image-icon.png \
//     crates/bevy_image/www/out/burn-image-icon.png
//   BURN_IMAGE_RENDERED_SURFACE_SMOKE=1 \
//     node crates/bevy_image/tests/wasm_rendered_surface_smoke.mjs
//
// The rendered real-model route additionally requires the current modular CDN staging tree:
//   cargo build -p bevy_burn_image --target wasm32-unknown-unknown --profile wasm-release \
//     --no-default-features --features boogu-web --locked --lib
//   wasm-bindgen --target web --out-dir crates/bevy_image/www/out \
//     --out-name bevy_burn_image target/wasm32-unknown-unknown/wasm-release/bevy_burn_image.wasm
//   BURN_IMAGE_RENDERED_TURBO_1024_SMOKE=1 \
//   BURN_IMAGE_RENDERED_TURBO_ARTIFACT_ROOT=.artifacts/cdn-upload-modular/aberration.technology/model \
//     node crates/bevy_image/tests/wasm_rendered_surface_smoke.mjs
// Set BURN_IMAGE_RENDERED_TURBO_1024_MULTI_REQUEST_QUALIFICATION=1 instead of the single-request
// variable to run two sequential ordinary Generate/Save requests in the same page and engine.
// It uses CDP keyboard/mouse input against the rendered Prompt, Seed, Run, and Save PNG controls,
// validates the production 1024x1024 PNG download, and explicitly verifies the Turbo preloaded
// packed-F16 storage / dense-F32-per-semantic-stage low-VRAM plan, its
// separate denoiser-preload and request
// traffic windows, and zero artifact access in the four-step DMD hot path. It also enforces an aggregate,
// measured, decimal-32-GB-exclusive Chrome GPU-process framebuffer ceiling across all PIDs per
// interval.
//
// The smoke is intentionally not numerical model parity. It qualifies a headful Bevy surface,
// the shared Bevy/Burn device-ready state, and a hardware NVIDIA BrowserWebGpu adapter. CI uses
// BURN_IMAGE_RENDERED_SURFACE_VALIDATE_ONLY=1 for cheap committed-source contract validation.

import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { constants as fsConstants, createReadStream, existsSync } from "node:fs";
import {
  access,
  mkdir,
  mkdtemp,
  open,
  readFile,
  readdir,
  realpath,
  rm,
  stat,
  statfs,
  writeFile,
} from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { delimiter, dirname, isAbsolute, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import {
  aggregateChromeGpuInterval,
  BACKEND_EVENT_NAME,
  BOOGU_WEB_REQUIRED_DEVICE_FEATURES,
  CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES,
  GENERIC_WEB_REQUIRED_DEVICE_FEATURES,
  GPU_INTERVAL_AGGREGATION_POLICY,
  gpuTerminalDiagnostic,
  captureCdpNetworkRequestContext,
  cdpNetworkLoadingFailedDiagnostic,
  LOW_VRAM_DEVICE_CAP_BYTES,
  LOW_VRAM_BACKEND,
  LOW_VRAM_PUBLIC_SELECTOR,
  OUTPUT_READY_EVENT_NAME,
  PROGRESS_EVENT_NAME,
  RENDERED_LAUNCH_READINESS_PROBE_POLICY,
  RUNTIME_EVENT_NAME,
  renderedChromeLaunchEvidence,
  renderedSurfaceReportIdentity,
  resolveTurboSecondRequestRunReadyUiContract,
  selectChromeSharedMemoryPolicy,
  summarizeTurboDmdCdpNetwork,
  summarizeTurboPreloadCdpNetwork,
  summarizeTurboRequestCdpNetwork,
  TURBO_DMD_RUNTIME_ZERO_IO_POLICY,
  TURBO_DENOISER_LINEAR_EXECUTION_POLICY,
  TURBO_DENOISER_QUANTIZED_EXECUTION_POLICY,
  TURBO_DENOISER_QUANTIZED_LOAD_POLICY,
  TURBO_DENOISER_STORAGE_POLICY,
  TURBO_LOW_VRAM_WEIGHT_TRAFFIC_CONTRACT,
  TURBO_PACKED_F16_CACHED_OBJECTS,
  TURBO_PACKED_F16_CACHED_STAGES,
  TURBO_PACKED_F16_CACHED_TENSORS,
  TURBO_PACKED_F16_RESOURCE_PLAN,
  TURBO_MODEL_ID,
  TURBO_MULTI_REQUEST_POLICY,
  TURBO_PRODUCTION_CONTENT_DIGEST,
  TURBO_QWEN_BLOCK0_ORDINARY_MODE,
  TURBO_QWEN_BLOCK0_SERIALIZED_DIAGNOSTIC_MODE,
  UI_CONTRACT_EVENT_NAME,
  attestRenderedSurfaceRuntimeAdapter,
  isCanonicalU64DecimalString,
  outputJobIdMatchesNumericRunId,
  validateHardwareNvidiaAdapter,
  validateRenderedSurfaceEvidence,
  validateRenderedSurfaceSnapshot,
  validateTurbo1024ModelEvidence,
  validateTurbo1024MultiRequestEvidence,
} from "./wasm_rendered_surface_contract.mjs";
import {
  ARTIFACT_TRANSPORT_LAYOUT_PATH,
  validateArtifactBundleTransport,
  transportTelemetryFiles,
} from "./artifact_transport_contract.mjs";

const ENABLE_ENV = "BURN_IMAGE_RENDERED_SURFACE_SMOKE";
const VALIDATE_ONLY_ENV = "BURN_IMAGE_RENDERED_SURFACE_VALIDATE_ONLY";
const CHROME_ENV = "BURN_IMAGE_RENDERED_SURFACE_CHROME";
const WWW_OUT_ENV = "BURN_IMAGE_RENDERED_SURFACE_WWW_OUT_DIR";
const OUTPUT_ENV = "BURN_IMAGE_RENDERED_SURFACE_OUTPUT_DIR";
const TIMEOUT_ENV = "BURN_IMAGE_RENDERED_SURFACE_TIMEOUT_MS";
const MODEL_ENV = "BURN_IMAGE_RENDERED_TURBO_1024_SMOKE";
const MULTI_REQUEST_MODEL_ENV =
  "BURN_IMAGE_RENDERED_TURBO_1024_MULTI_REQUEST_QUALIFICATION";
const MODEL_ARTIFACT_ROOT_ENV = "BURN_IMAGE_RENDERED_TURBO_ARTIFACT_ROOT";
const MODEL_TIMEOUT_ENV = "BURN_IMAGE_RENDERED_TURBO_TIMEOUT_MS";
const QWEN_BLOCK0_EXECUTION_MODE_ENV =
  "BURN_IMAGE_RENDERED_TURBO_QWEN_BLOCK0_EXECUTION_MODE";
// Release qualification exercises the ordinary path. The serialized branch is an explicit,
// diagnostic-only opt-in for localizing block-0 WebGPU failures.
const DEFAULT_TIMEOUT_MS = 90_000;
const DEFAULT_MODEL_TIMEOUT_MS = 4 * 60 * 60 * 1000;
const DEVTOOLS_TIMEOUT_MS = 45_000;
const STABILITY_MS = 5_000;
const MAX_CAPTURED_CHROME_BYTES = 2 * 1024 * 1024;
const MAX_CAPTURED_MONITOR_BYTES = 256 * 1024;
const MAX_CAPTURED_PROCESS_COMMAND_CHARS = 4_096;
const MAX_TRACKED_CHROME_GPU_PROCESSES = 64;
const MAX_RECORDED_GPU_SAMPLES = 240;
const MAX_RECORDED_MONITOR_ERRORS = 32;
const MIN_GPU_MATCHED_SAMPLE_INTERVALS = 3;
const MIN_GPU_ACTIVE_SAMPLE_INTERVALS = 1;
const GPU_SAMPLE_INTERVAL_MS = 1_000;
const CDP_CALL_TIMEOUT_MS = 20_000;
const INPUT_FOCUS_TIMEOUT_MS = 10_000;
const POINTER_SETTLE_MS = 100;
const CHROME_SHARED_MEMORY_PROBE_CHUNK_BYTES = 8 * 1024 * 1024;

const testsDir = dirname(fileURLToPath(import.meta.url));
const crateDir = resolve(testsDir, "..");
const repoRoot = resolve(crateDir, "../..");
const committedIndexPath = join(crateDir, "www/index.html");
const appSourcePath = join(crateDir, "src/app.rs");
const controlsSourcePath = join(crateDir, "src/controls.rs");
const displaySourcePath = join(crateDir, "src/display.rs");
const booguSourcePath = join(crateDir, "src/boogu.rs");
const browserBooguSourcePath = join(crateDir, "src/browser_boogu.rs");
const artifactStreamSourcePath = join(crateDir, "src/artifact_stream.rs");
const vaeDecoderSourcePath = join(repoRoot, "crates/burn_flux_vae/src/decoder.rs");
const modelSelectorSourcePath = join(crateDir, "www/model_selector.mjs");
const appIconPath = join(crateDir, "www/burn-image-icon.png");
const renderedHarnessSourcePath = fileURLToPath(import.meta.url);
const renderedContractSourcePath = join(testsDir, "wasm_rendered_surface_contract.mjs");
const artifactTransportContractSourcePath = join(testsDir, "artifact_transport_contract.mjs");
const TURBO_BUNDLE = "boogu-image-0.1-turbo";
const QWEN_BUNDLE = "qwen3-vl-8b-base-boogu-image-0.1";
const VAE_BUNDLE = "flux1-vae-boogu-image-0.1";
const MODEL_BUNDLES = [TURBO_BUNDLE, QWEN_BUNDLE, VAE_BUNDLE];
const MODEL_PROMPT = "A studio photograph of a blue ceramic bird on a plain white table.";

let interruptedSignal;
for (const signal of ["SIGINT", "SIGTERM"]) {
  process.once(signal, () => {
    interruptedSignal = signal;
  });
}

function throwIfInterrupted() {
  if (interruptedSignal) throw new Error(`interrupted by ${interruptedSignal}`);
}

function delay(milliseconds) {
  return new Promise((resolveDelay) => setTimeout(resolveDelay, milliseconds));
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

async function exactFileIdentity(path, root, expectedRelativePath) {
  const [canonicalRoot, canonicalPath] = await Promise.all([realpath(root), realpath(path)]);
  if (!canonicalPath.startsWith(`${canonicalRoot}/`)) {
    throw new Error(`identity path escapes ${canonicalRoot}: ${canonicalPath}`);
  }
  const relativePath = relative(canonicalRoot, canonicalPath).replaceAll("\\", "/");
  if (relativePath !== expectedRelativePath) {
    throw new Error(
      `identity relative path=${relativePath}, expected exact ${expectedRelativePath}`,
    );
  }
  const bytes = await readFile(canonicalPath);
  if (bytes.length <= 0) throw new Error(`identity file is empty: ${canonicalPath}`);
  return {
    absolute_path: canonicalPath,
    relative_path: relativePath,
    bytes: bytes.length,
    sha256: sha256(bytes),
  };
}

async function collectTestedPackageIdentity(wwwOutDir) {
  const [
    javascript,
    webassembly,
    modelSelector,
    appIcon,
    browserRuntimeSource,
    vaeDecoderSource,
    renderedHarnessSource,
    renderedContractSource,
    artifactTransportContractSource,
  ] = await Promise.all([
    exactFileIdentity(
      join(wwwOutDir, "bevy_burn_image.js"),
      wwwOutDir,
      "bevy_burn_image.js",
    ),
    exactFileIdentity(
      join(wwwOutDir, "bevy_burn_image_bg.wasm"),
      wwwOutDir,
      "bevy_burn_image_bg.wasm",
    ),
    exactFileIdentity(
      modelSelectorSourcePath,
      repoRoot,
      "crates/bevy_image/www/model_selector.mjs",
    ),
    exactFileIdentity(
      join(wwwOutDir, "burn-image-icon.png"),
      wwwOutDir,
      "burn-image-icon.png",
    ),
    exactFileIdentity(
      browserBooguSourcePath,
      repoRoot,
      "crates/bevy_image/src/browser_boogu.rs",
    ),
    exactFileIdentity(
      vaeDecoderSourcePath,
      repoRoot,
      "crates/burn_flux_vae/src/decoder.rs",
    ),
    exactFileIdentity(
      renderedHarnessSourcePath,
      repoRoot,
      "crates/bevy_image/tests/wasm_rendered_surface_smoke.mjs",
    ),
    exactFileIdentity(
      renderedContractSourcePath,
      repoRoot,
      "crates/bevy_image/tests/wasm_rendered_surface_contract.mjs",
    ),
    exactFileIdentity(
      artifactTransportContractSourcePath,
      repoRoot,
      "crates/bevy_image/tests/artifact_transport_contract.mjs",
    ),
  ]);
  return {
    policy: "exact-local-package-and-runtime-source-bytes-served-to-browser",
    generated_package: { javascript, webassembly, app_icon: appIcon },
    page_modules: { model_selector: modelSelector },
    sources: {
      browser_runtime: browserRuntimeSource,
      vae_decoder: vaeDecoderSource,
      rendered_harness: renderedHarnessSource,
      rendered_contract: renderedContractSource,
      artifact_transport_contract: artifactTransportContractSource,
    },
    validated: true,
  };
}

function parseTimeout() {
  const raw = process.env[TIMEOUT_ENV];
  if (raw === undefined) return DEFAULT_TIMEOUT_MS;
  const parsed = Number(raw);
  if (!Number.isSafeInteger(parsed) || parsed < 10_000 || parsed > 10 * 60_000) {
    throw new Error(`${TIMEOUT_ENV} must be an integer from 10000 through 600000`);
  }
  return parsed;
}

function parseModelTimeout() {
  const raw = process.env[MODEL_TIMEOUT_ENV];
  if (raw === undefined) return DEFAULT_MODEL_TIMEOUT_MS;
  const parsed = Number(raw);
  if (!Number.isSafeInteger(parsed) || parsed < 60_000 || parsed > 12 * 60 * 60 * 1000) {
    throw new Error(`${MODEL_TIMEOUT_ENV} must be an integer from 60000 through 43200000`);
  }
  return parsed;
}

function appendBounded(current, chunk) {
  const combined = current + String(chunk);
  return combined.length <= MAX_CAPTURED_CHROME_BYTES
    ? combined
    : combined.slice(combined.length - MAX_CAPTURED_CHROME_BYTES);
}

async function readPrefix(path, length) {
  const handle = await open(path, "r");
  try {
    const bytes = Buffer.alloc(length);
    const { bytesRead } = await handle.read(bytes, 0, length, 0);
    if (bytesRead !== length) throw new Error(`short local prefix read for ${path}`);
    return bytes;
  } finally {
    await handle.close();
  }
}

async function validateCommittedSources() {
  const [
    indexBytes,
    appSourceBytes,
    controlsSourceBytes,
    displaySourceBytes,
    booguSourceBytes,
    browserBooguSourceBytes,
    artifactStreamSourceBytes,
    vaeDecoderSourceBytes,
    renderedHarnessSourceBytes,
    renderedContractSourceBytes,
    modelSelectorSourceBytes,
    appIconBytes,
  ] = await Promise.all([
    readFile(committedIndexPath),
    readFile(appSourcePath),
    readFile(controlsSourcePath),
    readFile(displaySourcePath),
    readFile(booguSourcePath),
    readFile(browserBooguSourcePath),
    readFile(artifactStreamSourcePath),
    readFile(vaeDecoderSourcePath),
    readFile(renderedHarnessSourcePath),
    readFile(renderedContractSourcePath),
    readFile(modelSelectorSourcePath),
    readFile(appIconPath),
  ]);
  const indexSource = indexBytes.toString("utf8");
  const appSource = appSourceBytes.toString("utf8");
  const controlsSource = controlsSourceBytes.toString("utf8");
  const displaySource = displaySourceBytes.toString("utf8");
  const booguSource = booguSourceBytes.toString("utf8");
  const browserBooguSource = browserBooguSourceBytes.toString("utf8");
  const artifactStreamSource = artifactStreamSourceBytes.toString("utf8");
  const vaeDecoderSource = vaeDecoderSourceBytes.toString("utf8");
  const renderedHarnessSource = renderedHarnessSourceBytes.toString("utf8");
  const renderedContractSource = renderedContractSourceBytes.toString("utf8");
  const modelSelectorSource = modelSelectorSourceBytes.toString("utf8");
  const failures = [];
  for (const required of [
    'id="burn-image"',
    'id="status" aria-hidden="true"',
    'id="burn-image-reference-input"',
    'provide_reference_image_error',
    'reportReferenceError',
    'rel="icon" type="image/png" sizes="512x512" href="./out/burn-image-icon.png"',
    "The visible interface is exclusively Bevy on every platform",
    "await init()",
  ]) {
    if (!indexSource.includes(required)) failures.push(`www/index.html omits ${required}`);
  }
  for (const forbidden of [
    'id="model-loader"',
    'id="artifact-progress"',
    "loader-panel",
    "burn-image-runtime",
    "burn-image-progress",
    "configureModelReleaseSelector",
    "surface_inference_suspended",
  ]) {
    if (indexSource.includes(forbidden)) {
      failures.push(`www/index.html retains fragmented browser UI behavior ${forbidden}`);
    }
  }
  const pngSignature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  if (
    appIconBytes.length <= 25 ||
    !appIconBytes.subarray(0, pngSignature.length).equals(pngSignature) ||
    appIconBytes.toString("ascii", 12, 16) !== "IHDR" ||
    appIconBytes.readUInt32BE(16) !== 512 ||
    appIconBytes.readUInt32BE(20) !== 512 ||
    appIconBytes[24] !== 8 ||
    appIconBytes[25] !== 2
  ) {
    failures.push("www/burn-image-icon.png is not an exact 512x512 8-bit RGB PNG");
  }
  for (const required of [
    "MODEL_RELEASES",
    "modelReleaseSelectionState",
    "modelReleaseUrl",
    "configureModelReleaseSelector",
  ]) {
    if (!modelSelectorSource.includes(required)) {
      failures.push(`model selector omits ${required}`);
    }
  }
  for (const required of [
    `BROWSER_BACKEND_EVENT_NAME: &str = "${BACKEND_EVENT_NAME}"`,
    '"shared_adapter_device_queue"',
    "fn browser_safe_wgpu_settings_for(browser_webgpu: bool)",
    "settings.features = WgpuFeatures::empty();",
    'feature = "boogu-web"',
    "fn boogu_wgpu_settings_for(browser_webgpu: bool)",
    "WgpuFeatures::TIMESTAMP_QUERY",
    'BackendState::Ready { device }',
    'BackendState::Failed { reason }',
  ]) {
    if (!appSource.includes(required)) failures.push(`src/app.rs omits ${required}`);
  }
  for (const required of [
    `BROWSER_UI_CONTRACT_EVENT_NAME: &str = "${UI_CONTRACT_EVENT_NAME}"`,
    "BOOGU_DEFAULT_EDGE",
    'params.get("rendered-model-smoke")',
    'set("prompt_x", prompt_center.x.into())',
    'set("prompt_focused", prompt_focused.into())',
    'set("seed_x", seed_center.x.into())',
    'set("seed_focused", seed_focused.into())',
    'set("run_x", run_center.x.into())',
    'set("save_x", save_center.x.into())',
  ]) {
    if (!controlsSource.includes(required)) failures.push(`src/controls.rs omits ${required}`);
  }
  for (const required of [
    `BROWSER_OUTPUT_READY_EVENT_NAME: &str = "${OUTPUT_READY_EVENT_NAME}"`,
    "fn browser_output_ready_job_id(job: ImageJobId) -> String",
    "job.0.to_string()",
    'set("job_id", browser_output_ready_job_id(key.job).into())',
    "artifact_content_digest",
    "artifacts_verified",
  ]) {
    if (!displaySource.includes(required)) failures.push(`src/display.rs omits ${required}`);
  }
  if (displaySource.includes('set("job_id", key.job.0.into())')) {
    failures.push("src/display.rs converts a u64 output job ID directly to JavaScript BigInt");
  }
  for (const required of [
    "BROWSER_SURFACE_INFERENCE_POLICY",
    "BrowserSurfaceInferenceGate",
    "active_jobs: HashSet<ImageJobId>",
    "saved_primary_window_camera_states",
    "suspend_browser_surface_inference",
    "resume_browser_surface_inference",
    "enforce_browser_surface_inference_gate",
    ".add_systems(Last, enforce_browser_surface_inference_gate)",
    "report_browser_surface_inference_suspended",
    "report_browser_surface_inference_resumed",
  ]) {
    if (!booguSource.includes(required)) failures.push(`src/boogu.rs omits ${required}`);
  }
  for (const required of [
    LOW_VRAM_BACKEND.replace("burn-webgpu/", ""),
    LOW_VRAM_PUBLIC_SELECTOR,
    TURBO_DENOISER_STORAGE_POLICY,
    TURBO_DENOISER_QUANTIZED_LOAD_POLICY,
    TURBO_DENOISER_QUANTIZED_EXECUTION_POLICY,
    TURBO_DENOISER_LINEAR_EXECUTION_POLICY,
    TURBO_LOW_VRAM_WEIGHT_TRAFFIC_CONTRACT,
    "with_required_range_cache()",
    "BrowserRuntimeEvent::PackedF16DenoiserPreload",
    "BrowserRuntimeEvent::VramPreflight",
    "run_browser_vram_preflight(",
    "queue.on_submitted_work_done",
    "failed before model-weight download",
    "BrowserRuntimeEvent::PackedF16DenoiserLifecycle",
    "BrowserRuntimeEvent::PackedF16DmdVaeHandoff",
    "BrowserRuntimeEvent::PackedF16QwenHostEmbedding",
    "BrowserRuntimeEvent::PackedF16QwenBlock0ExecutionDiagnostics",
    "BrowserRuntimeEvent::PackedF16QwenBlock0PostSyncDiagnostic",
    "forward_base_async_with_host_input_ids",
    "validate_browser_packed_f16_qwen_embedding_report",
    "BrowserRuntimeEvent::PackedF16QwenPreHandoffDiagnostics",
    "BrowserRuntimeEvent::PackedF16QwenPostHandoffDiagnostics",
    "BrowserRuntimeEvent::PackedF16PreDmdInputDiagnostics",
    "browser_packed_f16_qwen_host_embedding_event(run_id, report.clone())",
    "browser_packed_f16_qwen_block0_execution_diagnostics_event(run_id, report)",
    "browser_packed_f16_qwen_block0_post_sync_diagnostic_event(run_id, diagnostic)",
    "browser_packed_f16_qwen_pre_handoff_diagnostics_event(run_id, diagnostics)",
    "browser_packed_f16_qwen_post_handoff_diagnostics_event(run_id, diagnostics)",
    "browser_packed_f16_pre_dmd_input_diagnostics_event(run_id, diagnostics.clone())",
    "browser_packed_f16_dmd_vae_handoff_event(run_id, report.clone())",
    "if !diagnostics.all_inputs_finite",
    "rendered-smoke packed-F16 pre-DMD inputs contain non-finite values",
    "packed_f16_qwen_instruction_handoff",
    "with_text_block_load_synchronization_policy",
    "BROWSER_PACKED_F16_QWEN_TEXT_LAYER_SUBMISSION_POLICY",
    "BROWSER_QWEN_BLOCK0_SERIALIZED_DIAGNOSTIC_MODE",
    "BROWSER_QWEN_BLOCK0_ORDINARY_MODE",
    "BROWSER_QWEN_BLOCK0_EXECUTION_MODE_QUERY",
    "None | Some(BROWSER_QWEN_BLOCK0_ORDINARY_MODE)",
    "Some(BROWSER_QWEN_BLOCK0_SERIALIZED_DIAGNOSTIC_MODE) =>",
    "validate_packed_f16_denoiser_lifecycle(",
    "checked_delta(dmd_artifact_traffic_before)",
    "preloaded Turbo denoiser performed artifact I/O during DMD",
    "observed.dmd_artifact_traffic == BrowserArtifactTrafficReport::default()",
    "report == BrowserArtifactTrafficReport::default()",
    "completed_dmd_steps == 4",
    "preserve_packed_cache",
    "fail_closed_packed_f16_request_cleanup",
    "retained-Q8 dense-F32-per-stage Turbo policy is retired",
    "const BROWSER_PRODUCTION_DENOISER_QUERY_CHUNK_SIZE: usize = 1_024;",
    "denoiser_minimum_image_query_partitions:",
    "burn_boogu::PORTABLE_ATTENTION_MINIMUM_IMAGE_QUERY_PARTITIONS",
  ]) {
    if (!browserBooguSource.includes(required)) {
      failures.push(`src/browser_boogu.rs omits ${required}`);
    }
  }
  const transportBootstrapCount = browserBooguSource.split(
    ".with_manifest_transport_layout(",
  ).length - 1;
  if (transportBootstrapCount < MODEL_BUNDLES.length) {
    failures.push(
      `src/browser_boogu.rs authenticates ${transportBootstrapCount} transport layouts; expected pipeline, Qwen, and VAE`,
    );
  }
  for (const required of [
    "fetch_browser_transport_layout",
    "fetch_browser_range_with_total(request, Some(object_size))",
    "fetch_browser_complete_file",
    "BROWSER_ARTIFACT_PART_CACHE_NAME",
    "browser_part_cache_key",
    "fetch_transport_part_complete_attempt",
    "fetch_and_cache_transport_part",
    "browser_complete_file_transport_required",
    "fetch_direct_complete_file_attempt",
    "fetch_and_cache_direct_complete_file",
    "read_browser_response_body_bounded",
    "read_browser_complete_response_body_bounded",
    "validate_browser_content_length",
    "validate_browser_content_length_if_exposed",
    "validate_browser_content_encoding",
    "RequestCache::NoStore",
    "actual.is_some_and",
    '!actual.eq_ignore_ascii_case("identity")',
    "response.blob()",
    "let blob_size = blob.size();",
    "blob.array_buffer()",
    "Uint8Array::new(&buffer)",
    "validate_browser_response_size",
    "transport_object_for_file",
    "fetch_verified_transport_part_bytes",
    "verify_browser_transport_part_bytes",
    "verify_browser_transport_part_bytes_async",
    'digest_with_str_and_u8_array("SHA-256", bytes)',
    "wasm_bindgen_futures::JsFuture::from(promise)",
    "VerifiedArtifactBytesBuilder::new(file)",
    ".extend_from_slice(&part_bytes)",
    "AsyncStageShardRead::from_verified_artifact_bytes",
    "protect_verified_transport_part(part)",
    "ARTIFACT_TRANSPORT_TARGET_PART_BYTES.min(ARTIFACT_TRANSPORT_MAX_PART_BYTES)",
    "BrowserTransportObjectMissing",
  ]) {
    if (!artifactStreamSource.includes(required)) {
      failures.push(`src/artifact_stream.rs omits ${required}`);
    }
  }
  if (artifactStreamSource.includes("response.array_buffer()")) {
    failures.push(
      "src/artifact_stream.rs copies a network Response through unbounded array_buffer()",
    );
  }
  if (artifactStreamSource.includes("probe_browser_file_size")) {
    failures.push(
      "src/artifact_stream.rs still performs a serial one-byte range probe before bounded compact-file fetches",
    );
  }
  const boundedBodyStart = artifactStreamSource.indexOf(
    "async fn read_browser_response_body_bounded(",
  );
  const boundedBodyEnd = artifactStreamSource.indexOf("\n#[cfg(", boundedBodyStart);
  const boundedBodySource = artifactStreamSource.slice(boundedBodyStart, boundedBodyEnd);
  const contentLengthGate = boundedBodySource.indexOf("validate_browser_content_length(");
  const encodingGate = boundedBodySource.indexOf("validate_browser_content_encoding(");
  const responseBlob = boundedBodySource.indexOf("response.blob()");
  const blobSizeGate = boundedBodySource.indexOf("blob.size()");
  const blobCopy = boundedBodySource.indexOf("blob.array_buffer()");
  const typedArrayLengthGate = boundedBodySource.indexOf(
    "validate_browser_response_size(expected_bytes, u64::from(bytes.length()))",
  );
  if (
    boundedBodyStart < 0 ||
    boundedBodyEnd < 0 ||
    !(
      contentLengthGate >= 0 &&
      contentLengthGate < encodingGate &&
      encodingGate < responseBlob &&
      responseBlob < blobSizeGate &&
      blobSizeGate < blobCopy &&
      blobCopy < typedArrayLengthGate
    )
  ) {
    failures.push(
      "src/artifact_stream.rs must validate exact identity framing, gate browser Blob size before its bounded Wasm copy, and recheck the typed-array length",
    );
  }
  if (
    boundedBodySource.includes("ReadableStreamDefaultReader") ||
    boundedBodySource.includes("response.body()") ||
    boundedBodySource.includes("reader.read()")
  ) {
    failures.push(
      "src/artifact_stream.rs uses the abort-prone manual ReadableStream path for bounded 206 responses",
    );
  }
  const contentLengthValidatorStart = artifactStreamSource.indexOf(
    "fn validate_browser_content_length(",
  );
  const contentLengthValidatorEnd = artifactStreamSource.indexOf(
    "\n#[cfg(",
    contentLengthValidatorStart,
  );
  const contentLengthValidatorSource = artifactStreamSource.slice(
    contentLengthValidatorStart,
    contentLengthValidatorEnd,
  );
  for (const required of [
    "actual.is_some_and",
    "parsed == expected",
    "parsed.to_string() == actual",
  ]) {
    if (!contentLengthValidatorSource.includes(required)) {
      failures.push(
        `src/artifact_stream.rs Content-Length validator does not require exact canonical framing: missing ${required}`,
      );
    }
  }
  const boundedRangeResponseReadCount = artifactStreamSource.split(
    "read_browser_response_body_bounded(&response,",
  ).length - 1;
  const boundedCompleteResponseReadCount = artifactStreamSource.split(
    "read_browser_complete_response_body_bounded(&response,",
  ).length - 1;
  if (boundedRangeResponseReadCount < 1 || boundedCompleteResponseReadCount < 1) {
    failures.push(
      `src/artifact_stream.rs has ${boundedRangeResponseReadCount} bounded range reads and ${boundedCompleteResponseReadCount} bounded complete-object reads; expected both transport paths`,
    );
  }
  const packedPreDmdCleanupPrimitives = [
    "instruction.into_data_async()",
    'synchronize("packed-F16 Qwen instruction handoff (before cleanup)")',
    "<BrowserBackend as Backend>::memory_cleanup(&self.device)",
    'synchronize("packed-F16 Qwen instruction handoff (after cleanup)")',
    'require_exact_packed_f16_cache_audit("post-Qwen instruction handoff", audit)',
    "Tensor::<BrowserBackend, 3>::from_data(instruction_data, (&self.device, DType::F32))",
    "instruction_after.sha256 == instruction_before.sha256",
  ];
  let packedPreDmdCleanupCursor = browserBooguSource.indexOf(
    "async fn packed_f16_qwen_instruction_handoff",
  );
  if (packedPreDmdCleanupCursor < 0) {
    failures.push("src/browser_boogu.rs omits the packed-F16 Qwen handoff implementation");
  } else {
    for (const primitive of packedPreDmdCleanupPrimitives) {
      const next = browserBooguSource.indexOf(primitive, packedPreDmdCleanupCursor + 1);
      if (next < 0) {
        failures.push(
          `src/browser_boogu.rs omits or reorders packed-F16 pre-DMD primitive ${primitive}`,
        );
        break;
      }
      packedPreDmdCleanupCursor = next;
    }
  }
  const packedDmdVaeCleanupPrimitives = [
    "latents.into_data_async()",
    'synchronize("packed-F16 DMD-to-VAE handoff (before cache clear)")',
    "self.denoiser.source_mut().clear();",
    "self.denoiser.clear_rope_cache();",
    "self.packed_f16_denoiser_source_mut()?.clear();",
    'synchronize("packed-F16 DMD-to-VAE handoff (before allocator cleanup)")',
    "<BrowserBackend as Backend>::memory_cleanup(&self.device)",
    'synchronize("packed-F16 DMD-to-VAE handoff (after allocator cleanup)")',
    'require_empty_packed_f16_cache_audit("DMD-to-VAE handoff exit", audit_after)',
    "Tensor::<BrowserBackend, 4>::from_data",
    "latent_after.sha256 == latent_before.sha256",
  ];
  let packedDmdVaeCleanupCursor = browserBooguSource.indexOf(
    "async fn packed_f16_dmd_vae_handoff",
  );
  if (packedDmdVaeCleanupCursor < 0) {
    failures.push("src/browser_boogu.rs omits the packed-F16 DMD-to-VAE handoff implementation");
  } else {
    for (const primitive of packedDmdVaeCleanupPrimitives) {
      const next = browserBooguSource.indexOf(primitive, packedDmdVaeCleanupCursor + 1);
      if (next < 0) {
        failures.push(
          `src/browser_boogu.rs omits or reorders packed-F16 DMD-to-VAE primitive ${primitive}`,
        );
        break;
      }
      packedDmdVaeCleanupCursor = next;
    }
  }
  const packedDmdVaeCallerPrimitives = [
    "drop(instruction);",
    "drop(reference);",
    "drop(noises);",
    "drop(first_dmd_timestep);",
    ".packed_f16_dmd_vae_handoff(latents, latent_shape, run_id)",
    ".fail_closed_packed_f16_request_cleanup()",
  ];
  let packedDmdVaeCallerCursor = browserBooguSource.indexOf(
    "let latents = if self.policies.uses_packed_f16_denoiser_source()",
  );
  if (packedDmdVaeCallerCursor < 0) {
    failures.push("src/browser_boogu.rs omits the packed-F16 DMD-to-VAE caller boundary");
  } else {
    for (const primitive of packedDmdVaeCallerPrimitives) {
      const next = browserBooguSource.indexOf(primitive, packedDmdVaeCallerCursor + 1);
      if (next < 0) {
        failures.push(
          `src/browser_boogu.rs omits or reorders packed-F16 DMD-to-VAE caller primitive ${primitive}`,
        );
        break;
      }
      packedDmdVaeCallerCursor = next;
    }
  }
  const vaeStripedTailPrimitives = [
    "let first_resnet = final_block",
    "let (left, right) = resnet_two_width_slabs_strict_f32(first_resnet, left, right);",
    "hidden = Tensor::cat(vec![left, right], 3);",
    "for resnet in final_block.resnets.iter().skip(1)",
    "resnet.forward_with_group_norm_policy(hidden, DecoderGroupNormPolicy::StrictF32);",
    "self.conv_out.forward(silu(group_norm_with_policy(",
    "&self.conv_norm_out,",
  ];
  let vaeStripedTailCursor = vaeDecoderSource.indexOf(
    "pub fn forward_striped_tail_strict_f32",
  );
  if (vaeStripedTailCursor < 0) {
    failures.push("burn_flux_vae decoder omits the bounded striped-tail entry point");
  } else {
    for (const primitive of vaeStripedTailPrimitives) {
      const next = vaeDecoderSource.indexOf(primitive, vaeStripedTailCursor + 1);
      if (next < 0) {
        failures.push(
          `burn_flux_vae decoder omits or reorders qualified striped-tail primitive ${primitive}`,
        );
        break;
      }
      vaeStripedTailCursor = next;
    }
  }
  const qwenFailureEmbeddingPrimitives = [
    "let qwen_output_result = if",
    "let host_embedding_report = self.qwen.take_last_host_routed_embedding_report();",
    "browser_packed_f16_qwen_host_embedding_event(run_id, report.clone())",
    "(true, Some(_), None) if qwen_output_result.is_err() => {}",
    "let qwen_output = qwen_output_result.map_err(|error|",
  ];
  let qwenFailureEmbeddingCursor = browserBooguSource.indexOf(
    "let qwen_output_result = if",
  );
  if (qwenFailureEmbeddingCursor < 0) {
    failures.push("src/browser_boogu.rs omits the streamed Qwen output result boundary");
  } else {
    for (const primitive of qwenFailureEmbeddingPrimitives.slice(1)) {
      const next = browserBooguSource.indexOf(primitive, qwenFailureEmbeddingCursor + 1);
      if (next < 0) {
        failures.push(
          `src/browser_boogu.rs does not preserve host-embedding failure provenance before ${primitive}`,
        );
        break;
      }
      qwenFailureEmbeddingCursor = next;
    }
  }
  const packedFailureCleanupPrimitives = [
    "let mut result = engine.infer(&job, &cancellation, &shared).await;",
    "if result.is_err() && engine.policies.uses_packed_f16_denoiser_source()",
    "engine.fail_closed_packed_f16_request_cleanup().await",
    "let terminal = match result",
  ];
  let packedFailureCleanupCursor = browserBooguSource.indexOf(
    packedFailureCleanupPrimitives[0],
  );
  if (packedFailureCleanupCursor < 0) {
    failures.push("src/browser_boogu.rs omits the outer packed-F16 terminal cleanup boundary");
  } else {
    for (const primitive of packedFailureCleanupPrimitives.slice(1)) {
      const next = browserBooguSource.indexOf(primitive, packedFailureCleanupCursor + 1);
      if (next < 0) {
        failures.push(
          `src/browser_boogu.rs omits or reorders packed-F16 terminal cleanup before ${primitive}`,
        );
        break;
      }
      packedFailureCleanupCursor = next;
    }
  }
  for (const required of [
    "Page.addScriptToEvaluateOnNewDocument",
    'recordWebGpu("request-adapter-start"',
    'recordWebGpu("request-device-resolved"',
    "GPUCanvasContext",
    "prototype.getCurrentTexture",
    "surface_texture_acquisition_count",
    "surface_texture_gate_windows",
    "surface_texture_gate_violation_calls",
    "surface_texture_gate_violation_calls_overflow",
    "enabledFeatures",
    "RENDERED_LAUNCH_READINESS_PROBE_POLICY",
    "runtime_webgpu_adapter_attestation",
    "packed_f16_pre_dmd_input_diagnostics",
    "packed_f16_qwen_host_embedding",
    "packed_f16_qwen_block0_execution_diagnostics",
    "packed_f16_qwen_block0_post_sync_diagnostic",
    "packed_f16_qwen_pre_handoff_diagnostics",
    "packed_f16_qwen_post_handoff_diagnostics",
    "BURN_IMAGE_RENDERED_TURBO_QWEN_BLOCK0_EXECUTION_MODE",
    "renderedSurfaceReportIdentity",
    "inspectChromeSharedMemory",
    "quotaAwareAllocationProbe",
    "selectChromeSharedMemoryPolicy",
    "renderedChromeLaunchEvidence",
    "sharedMemoryPolicy.disable_dev_shm_usage",
    'query.set("surface-gate", "1")',
    'query.set("qwen-block0-execution-mode", requestedQwenBlock0ExecutionMode)',
    'pathname.endsWith(".mjs")',
    '"/model_selector.mjs"',
    "page_modules",
    "outputJobIdMatchesNumericRunId",
    "resolveTurboSecondRequestRunReadyUiContract",
    'message.method === "Network.requestWillBeSent"',
    "captureCdpNetworkRequestContext(message.params, previous)",
    'message.method === "Network.loadingFailed"',
    "cdpNetworkLoadingFailedDiagnostic(",
    "diagnostic.proven_benign_favicon",
    "networkFailures.push(diagnostic)",
    "ignoredBenignNetworkFailures.push(diagnostic)",
  ]) {
    if (!renderedHarnessSource.includes(required)) {
      failures.push(`rendered harness omits ${required}`);
    }
  }
  const secondRequestRunReadyPrimitives = [
    "const secondSeedReady = await waitForTextValue(",
    "const secondRunReady = resolveTurboSecondRequestRunReadyUiContract({",
    "uiEvents: secondSeedReady.snapshot.ui_events ?? [],",
    "uiStartIndex: secondStarts.ui,",
    "seedChangedEvent: secondSeedReady.changed,",
    "postRequestUiContract: saveReady.uiContract,",
    "run_readiness: secondRunReady.evidence,",
  ];
  let secondRequestRunReadyCursor = renderedHarnessSource.indexOf(
    secondRequestRunReadyPrimitives[0],
  );
  if (secondRequestRunReadyCursor < 0) {
    failures.push("rendered harness omits the exact second-request seed-change boundary");
  } else {
    for (const primitive of secondRequestRunReadyPrimitives.slice(1)) {
      const next = renderedHarnessSource.indexOf(primitive, secondRequestRunReadyCursor + 1);
      if (next < 0) {
        failures.push(
          `rendered harness does not preserve second-request Run readiness before ${primitive}`,
        );
        break;
      }
      secondRequestRunReadyCursor = next;
    }
  }
  if (
    !/const requestedQwenBlock0ExecutionMode\s*=\s*process\.env\[QWEN_BLOCK0_EXECUTION_MODE_ENV\]\s*\?\?\s*TURBO_QWEN_BLOCK0_ORDINARY_MODE;/.test(
      renderedHarnessSource,
    )
  ) {
    failures.push("rendered harness does not default Qwen block-0 execution to ordinary");
  }
  if (
    !renderedHarnessSource.includes("const MAX_CAPTURED_CHROME_BYTES = 2 * 1024 * 1024;") ||
    !renderedHarnessSource.includes("function appendBounded(current, chunk)") ||
    !renderedHarnessSource.includes('outcome.chrome_stderr = finalChromeStderr;')
  ) {
    failures.push("rendered harness does not persist bounded Chrome stderr in terminal evidence");
  }
  for (const required of [
    "summarizeRenderedSurfaceRuntimeWebGpuCalls",
    "attestRenderedSurfaceRuntimeAdapter",
    "validateRenderedRuntimeWebGpuEvidence",
    "renderedSurfaceReportIdentity",
    "GENERIC_WEB_REQUIRED_DEVICE_FEATURES",
    "BOOGU_WEB_REQUIRED_DEVICE_FEATURES",
    "BROWSER_WEBGPU_ENABLED_FEATURE_BASELINE",
    '"core-features-and-limits"',
    "enabled_features",
    "expected_enabled_features",
    "validatePackedF16PreDmdInputDiagnostics",
    "validatePackedF16QwenHostEmbedding",
    "validatePackedF16QwenBlock0ExecutionDiagnostics",
    "validatePackedF16QwenBlock0PostSyncDiagnostic",
    "validatePackedF16QwenPreHandoffDiagnostics",
    "validatePackedF16QwenPostHandoffDiagnostics",
    'event.backend !== "BrowserWebGpu"',
    "selected_device_success_at_ms",
    "SURFACE_INFERENCE_POLICY",
    "validateRequestScopedSurfaceGate",
    "GPUCanvasContext.getCurrentTexture",
    "isCanonicalU64DecimalString",
    "outputJobIdMatchesNumericRunId",
    "TURBO_SECOND_REQUEST_RUN_READY_POLICY",
    "resolveTurboSecondRequestRunReadyUiContract",
    "fallback_exact_last_pre_boundary_ready",
    "captureCdpNetworkRequestContext",
    "cdpNetworkLoadingFailedDiagnostic",
    "isProvenBenignFaviconFailure",
    "validateCdpNetworkFailureEvidence",
  ]) {
    if (!renderedContractSource.includes(required)) {
      failures.push(`rendered contract omits ${required}`);
    }
  }
  if (failures.length > 0) {
    throw new Error(`rendered-surface source contract failed:\n${failures.join("\n")}`);
  }
  return {
    committed_index_path: committedIndexPath,
    committed_index_bytes: indexBytes.length,
    committed_index_sha256: sha256(indexBytes),
    app_source_path: appSourcePath,
    app_source_sha256: sha256(appSourceBytes),
    controls_source_sha256: sha256(controlsSourceBytes),
    display_source_sha256: sha256(displaySourceBytes),
    boogu_frontend_source_sha256: sha256(booguSourceBytes),
    browser_runtime_source_sha256: sha256(browserBooguSourceBytes),
    artifact_stream_source_sha256: sha256(artifactStreamSourceBytes),
    vae_decoder_source_sha256: sha256(vaeDecoderSourceBytes),
    rendered_harness_source_sha256: sha256(renderedHarnessSourceBytes),
    rendered_contract_source_sha256: sha256(renderedContractSourceBytes),
    model_selector_source_sha256: sha256(modelSelectorSourceBytes),
    app_icon_bytes: appIconBytes.length,
    app_icon_sha256: sha256(appIconBytes),
    validated: true,
  };
}

function contentType(pathname) {
  if (pathname.endsWith(".html")) return "text/html; charset=utf-8";
  if (pathname.endsWith(".js") || pathname.endsWith(".mjs")) {
    return "text/javascript; charset=utf-8";
  }
  if (pathname.endsWith(".json")) return "application/json";
  if (pathname.endsWith(".wasm")) return "application/wasm";
  if (pathname.endsWith(".png")) return "image/png";
  return "application/octet-stream";
}

function commonHeaders(type, length) {
  return {
    "Cache-Control": "no-store",
    "Content-Length": String(length),
    "Content-Type": type,
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "X-Content-Type-Options": "nosniff",
  };
}

async function createAppServer(indexBytes, wwwOutDir, artifactRoot = undefined) {
  const generated = new Map([
    ["/out/bevy_burn_image.js", join(wwwOutDir, "bevy_burn_image.js")],
    ["/out/bevy_burn_image_bg.wasm", join(wwwOutDir, "bevy_burn_image_bg.wasm")],
    ["/out/burn-image-icon.png", join(wwwOutDir, "burn-image-icon.png")],
  ]);
  const pageModules = new Map([["/model_selector.mjs", modelSelectorSourcePath]]);
  const outRoot = await realpath(wwwOutDir);
  const generatedMetadata = new Map();
  for (const [route, path] of generated) {
    if (!existsSync(path)) throw new Error(`required browser output is missing: ${path}`);
    const canonical = await realpath(path);
    if (!canonical.startsWith(`${outRoot}/`)) {
      throw new Error(`generated browser output escapes ${outRoot}: ${canonical}`);
    }
    const metadata = await stat(canonical);
    if (!metadata.isFile() || metadata.size <= 0) {
      throw new Error(`generated browser output is not a non-empty file: ${canonical}`);
    }
    generatedMetadata.set(route, { path: canonical, bytes: metadata.size });
  }
  const pageModuleMetadata = new Map();
  for (const [route, path] of pageModules) {
    const canonical = await realpath(path);
    const canonicalCrateRoot = await realpath(crateDir);
    if (!canonical.startsWith(`${canonicalCrateRoot}/`)) {
      throw new Error(`page module escapes ${canonicalCrateRoot}: ${canonical}`);
    }
    const metadata = await stat(canonical);
    if (!metadata.isFile() || metadata.size <= 0) {
      throw new Error(`page module is not a non-empty file: ${canonical}`);
    }
    pageModuleMetadata.set(route, { path: canonical, bytes: metadata.size });
  }
  let canonicalArtifactRoot;
  if (artifactRoot) {
    canonicalArtifactRoot = await realpath(artifactRoot);
    for (const bundle of MODEL_BUNDLES) {
      const manifestPath = join(canonicalArtifactRoot, bundle, "manifest.json");
      if (!existsSync(manifestPath)) {
        throw new Error(`required modular browser manifest is missing: ${manifestPath}`);
      }
    }
  }

  const server = createServer((request, response) => {
    const url = new URL(request.url ?? "/", "http://127.0.0.1");
    if (!["GET", "HEAD", "OPTIONS"].includes(request.method ?? "")) {
      response.writeHead(405, { Allow: "GET, HEAD, OPTIONS" });
      response.end();
      return;
    }
    if (url.pathname === "/favicon.ico") {
      response.writeHead(204, { "Cache-Control": "no-store" });
      response.end();
      return;
    }
    if (url.pathname === "/probe") {
      const body = Buffer.from("burn_image rendered surface probe\n");
      response.writeHead(200, commonHeaders("text/plain; charset=utf-8", body.length));
      if (request.method === "HEAD") response.end();
      else response.end(body);
      return;
    }
    if (url.pathname === "/index.html") {
      response.writeHead(200, commonHeaders(contentType(url.pathname), indexBytes.length));
      if (request.method === "HEAD") response.end();
      else response.end(indexBytes);
      return;
    }
    const artifactMatch = /^\/model\/([^/]+)\/(.+)$/.exec(url.pathname);
    if (artifactMatch && canonicalArtifactRoot) {
      const [, bundle, encodedPath] = artifactMatch;
      if (!MODEL_BUNDLES.includes(bundle)) {
        response.writeHead(404, { "Cache-Control": "no-store" });
        response.end();
        return;
      }
      let relativePath;
      try {
        relativePath = encodedPath
          .split("/")
          .map((segment) => decodeURIComponent(segment))
          .join("/");
      } catch {
        response.writeHead(400, { "Cache-Control": "no-store" });
        response.end();
        return;
      }
      if (
        !relativePath ||
        relativePath.startsWith("/") ||
        relativePath.split("/").some((segment) => !segment || segment === "." || segment === "..")
      ) {
        response.writeHead(400, { "Cache-Control": "no-store" });
        response.end();
        return;
      }
      const requestedPath = join(canonicalArtifactRoot, bundle, relativePath);
      void (async () => {
        try {
          const canonicalPath = await realpath(requestedPath);
          const canonicalBundleRoot = await realpath(join(canonicalArtifactRoot, bundle));
          if (!canonicalPath.startsWith(`${canonicalBundleRoot}/`)) {
            throw new Error("artifact path escapes its bundle root");
          }
          const metadata = await stat(canonicalPath);
          if (!metadata.isFile() || metadata.size <= 0) {
            throw new Error("artifact is not a non-empty regular file");
          }
          const baseHeaders = {
            ...commonHeaders(contentType(canonicalPath), metadata.size),
            "Accept-Ranges": "bytes",
            "Access-Control-Allow-Headers": "Range",
            "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Expose-Headers": "Accept-Ranges, Content-Length, Content-Range",
          };
          if (request.method === "OPTIONS") {
            response.writeHead(204, { ...baseHeaders, "Content-Length": "0" });
            response.end();
            return;
          }
          const range = request.headers.range;
          if (range) {
            const match = /^bytes=(\d+)-(\d*)$/.exec(range);
            if (!match) {
              response.writeHead(416, {
                ...baseHeaders,
                "Content-Range": `bytes */${metadata.size}`,
              });
              response.end();
              return;
            }
            const start = Number(match[1]);
            const end = match[2] ? Number(match[2]) : metadata.size - 1;
            if (
              !Number.isSafeInteger(start) ||
              !Number.isSafeInteger(end) ||
              start < 0 ||
              end < start ||
              start >= metadata.size ||
              end >= metadata.size
            ) {
              response.writeHead(416, {
                ...baseHeaders,
                "Content-Range": `bytes */${metadata.size}`,
              });
              response.end();
              return;
            }
            const length = end - start + 1;
            response.writeHead(206, {
              ...baseHeaders,
              "Content-Length": String(length),
              "Content-Range": `bytes ${start}-${end}/${metadata.size}`,
            });
            if (request.method === "HEAD") response.end();
            else createReadStream(canonicalPath, { start, end }).pipe(response);
            return;
          }
          response.writeHead(200, baseHeaders);
          if (request.method === "HEAD") response.end();
          else createReadStream(canonicalPath).pipe(response);
        } catch (error) {
          if (!response.headersSent) {
            response.writeHead(404, { "Cache-Control": "no-store" });
          }
          response.end();
        }
      })();
      return;
    }
    const file =
      generatedMetadata.get(url.pathname) ?? pageModuleMetadata.get(url.pathname);
    if (request.method === "OPTIONS") {
      response.writeHead(405, { Allow: "GET, HEAD" });
      response.end();
      return;
    }
    if (!file) {
      response.writeHead(404, { "Cache-Control": "no-store" });
      response.end();
      return;
    }
    response.writeHead(200, commonHeaders(contentType(url.pathname), file.bytes));
    if (request.method === "HEAD") response.end();
    else createReadStream(file.path).pipe(response);
  });
  await new Promise((resolveListen, rejectListen) => {
    server.once("error", rejectListen);
    server.listen(0, "127.0.0.1", resolveListen);
  });
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("app server has no TCP address");
  return {
    server,
    port: address.port,
    generated: Object.fromEntries(generatedMetadata),
    page_modules: Object.fromEntries(pageModuleMetadata),
  };
}

async function closeServer(server) {
  if (!server) return;
  await new Promise((resolveClose) => server.close(() => resolveClose()));
}

async function validateServedApp(baseUrl, sourceEvidence, testedPackageIdentity) {
  const indexUrl = `${baseUrl}/index.html`;
  const indexResponse = await fetch(indexUrl, { cache: "no-store" });
  if (!indexResponse.ok) throw new Error(`served committed app returned HTTP ${indexResponse.status}`);
  const servedIndex = Buffer.from(await indexResponse.arrayBuffer());
  if (sha256(servedIndex) !== sourceEvidence.committed_index_sha256) {
    throw new Error("served index bytes differ from committed www/index.html");
  }
  const generated = {};
  for (const name of ["bevy_burn_image.js", "bevy_burn_image_bg.wasm"]) {
    const response = await fetch(`${baseUrl}/out/${name}`, { cache: "no-store" });
    const body = Buffer.from(await response.arrayBuffer());
    const declaredBytes = Number(response.headers.get("content-length"));
    if (
      !response.ok ||
      !Number.isSafeInteger(declaredBytes) ||
      declaredBytes <= 0 ||
      body.length !== declaredBytes
    ) {
      throw new Error(`served ${name} is unavailable or empty`);
    }
    const identity =
      name === "bevy_burn_image.js"
        ? testedPackageIdentity.generated_package.javascript
        : testedPackageIdentity.generated_package.webassembly;
    const bodySha256 = sha256(body);
    if (body.length !== identity.bytes || bodySha256 !== identity.sha256) {
      throw new Error(`served ${name} bytes differ from the exact tested package identity`);
    }
    generated[name] = {
      bytes: body.length,
      sha256: bodySha256,
      content_type: response.headers.get("content-type"),
    };
  }
  const modelSelectorResponse = await fetch(`${baseUrl}/model_selector.mjs`, {
    cache: "no-store",
  });
  const modelSelectorBytes = Buffer.from(await modelSelectorResponse.arrayBuffer());
  const modelSelectorIdentity = testedPackageIdentity.page_modules.model_selector;
  const modelSelectorDeclaredBytes = Number(
    modelSelectorResponse.headers.get("content-length"),
  );
  const modelSelectorContentType = modelSelectorResponse.headers.get("content-type");
  if (
    !modelSelectorResponse.ok ||
    modelSelectorContentType !== "text/javascript; charset=utf-8" ||
    modelSelectorDeclaredBytes !== modelSelectorIdentity.bytes ||
    modelSelectorBytes.length !== modelSelectorIdentity.bytes ||
    sha256(modelSelectorBytes) !== modelSelectorIdentity.sha256
  ) {
    throw new Error(
      "served model_selector.mjs MIME, bytes, or SHA-256 differ from the exact tested package identity",
    );
  }
  const appIconResponse = await fetch(`${baseUrl}/out/burn-image-icon.png`, {
    cache: "no-store",
  });
  const appIconBytes = Buffer.from(await appIconResponse.arrayBuffer());
  const appIconIdentity = testedPackageIdentity.generated_package.app_icon;
  const appIconDeclaredBytes = Number(appIconResponse.headers.get("content-length"));
  const appIconContentType = appIconResponse.headers.get("content-type");
  if (
    !appIconResponse.ok ||
    appIconContentType !== "image/png" ||
    appIconDeclaredBytes !== appIconIdentity.bytes ||
    appIconBytes.length !== appIconIdentity.bytes ||
    sha256(appIconBytes) !== appIconIdentity.sha256
  ) {
    throw new Error(
      "served burn-image-icon.png MIME, bytes, or SHA-256 differ from the exact tested package identity",
    );
  }
  generated["burn-image-icon.png"] = {
    bytes: appIconBytes.length,
    sha256: sha256(appIconBytes),
    content_type: appIconContentType,
  };
  return {
    exact_committed_index_sha256: sourceEvidence.committed_index_sha256,
    exact_committed_index_bytes: servedIndex.length,
    generated,
    page_modules: {
      "model_selector.mjs": {
        bytes: modelSelectorBytes.length,
        sha256: sha256(modelSelectorBytes),
        content_type: modelSelectorContentType,
      },
    },
    validated: true,
  };
}

async function validateModelArtifactTransport(baseUrl, artifactRoot) {
  const bundles = [];
  const transportTelemetryByPath = new Map();
  let totalLogicalFiles = 0;
  let totalLogicalBytes = 0;
  let totalPhysicalParts = 0;
  let totalPhysicalPartBytes = 0;
  let maximumPhysicalPartBytes = 0;
  for (const bundle of MODEL_BUNDLES) {
    const localRoot = join(artifactRoot, bundle);
    const localManifestBytes = await readFile(join(localRoot, "manifest.json"));
    const manifest = JSON.parse(localManifestBytes.toString("utf8"));
    if (manifest.bundle !== bundle) {
      throw new Error(`modular manifest bundle=${manifest.bundle}, expected ${bundle}`);
    }
    if (!Array.isArray(manifest.files) || manifest.files.length === 0) {
      throw new Error(`modular manifest ${bundle} has no files`);
    }
    if (bundle === TURBO_BUNDLE) {
      if (manifest.content_digest !== TURBO_PRODUCTION_CONTENT_DIGEST) {
        throw new Error("local Turbo parent is not the canonical production composition");
      }
      const dependencies = new Map(
        (manifest.dependencies ?? []).map((dependency) => [dependency.bundle, dependency]),
      );
      for (const dependency of [QWEN_BUNDLE, VAE_BUNDLE]) {
        if (!dependencies.has(dependency)) {
          throw new Error(`Turbo parent omits modular dependency ${dependency}`);
        }
      }
    } else if ((manifest.dependencies ?? []).length !== 0) {
      throw new Error(`modular leaf ${bundle} must not declare dependencies`);
    }
    const localTransport = await validateArtifactBundleTransport({
      bundleRoot: localRoot,
      manifest,
    });

    const manifestUrl = `${baseUrl}/model/${bundle}/manifest.json`;
    const response = await fetch(manifestUrl, { cache: "no-store" });
    const remoteManifestBytes = Buffer.from(await response.arrayBuffer());
    if (!response.ok || sha256(remoteManifestBytes) !== sha256(localManifestBytes)) {
      throw new Error(`served ${bundle}/manifest.json differs from exact local bytes`);
    }
    if (response.headers.get("access-control-allow-origin") !== "*") {
      throw new Error(`served ${bundle} manifest omits wildcard CORS`);
    }
    const options = await fetch(manifestUrl, {
      method: "OPTIONS",
      headers: { Origin: "https://browser-smoke.invalid", Range: "bytes=0-7" },
    });
    if (
      options.status !== 204 ||
      options.headers.get("access-control-allow-headers") !== "Range" ||
      !String(options.headers.get("access-control-expose-headers")).includes("Content-Range")
    ) {
      throw new Error(`served ${bundle} does not provide the required Range/CORS preflight`);
    }

    for (const file of manifest.files.filter((entry) => entry.role !== "weights")) {
      const localPath = join(localRoot, file.path);
      const metadata = await stat(localPath);
      if (!metadata.isFile() || metadata.size !== file.size) {
        throw new Error(
          `local modular file ${bundle}/${file.path} is ${metadata.size} bytes, expected ${file.size}`,
        );
      }
      const head = await fetch(`${baseUrl}/model/${bundle}/${file.path}`, {
        method: "HEAD",
        headers: { Origin: "https://browser-smoke.invalid" },
      });
      if (
        !head.ok ||
        Number(head.headers.get("content-length")) !== file.size ||
        head.headers.get("accept-ranges") !== "bytes" ||
        head.headers.get("access-control-allow-origin") !== "*"
      ) {
        throw new Error(`served modular HEAD contract failed for ${bundle}/${file.path}`);
      }
    }

    const representativePart = transportTelemetryFiles(localTransport)[0];
    if (!representativePart) {
      throw new Error(`${bundle} has no physical transport part to probe`);
    }
    const servedProbes = [
      manifest.files.find((file) => file.path === ARTIFACT_TRANSPORT_LAYOUT_PATH),
      representativePart,
    ];
    for (const file of servedProbes) {
      const end = Math.min(7, file.size - 1);
      const range = await fetch(`${baseUrl}/model/${bundle}/${file.path}`, {
        headers: {
          Origin: "https://browser-smoke.invalid",
          Range: `bytes=0-${end}`,
        },
      });
      const bytes = Buffer.from(await range.arrayBuffer());
      const expectedRange = `bytes 0-${end}/${file.size}`;
      if (
        range.status !== 206 ||
        range.headers.get("content-range") !== expectedRange ||
        bytes.length !== end + 1
      ) {
        throw new Error(`served modular Range contract failed for ${bundle}/${file.path}`);
      }
      const localBytes = await readPrefix(join(localRoot, file.path), end + 1);
      if (!bytes.equals(localBytes)) {
        throw new Error(`served modular Range bytes differ for ${bundle}/${file.path}`);
      }
    }

    bundles.push({
      bundle,
      content_digest: manifest.content_digest,
      logical_artifacts: localTransport.logical,
      direct_artifacts: localTransport.direct,
      physical_transport: localTransport.transport,
      transport_sidecar: localTransport.sidecar,
      manifest_sha256: sha256(localManifestBytes),
    });
    for (const entry of transportTelemetryFiles(localTransport)) {
      transportTelemetryByPath.set(`/model/${bundle}/${entry.path}`, {
        bundle,
        component: entry.component,
        components: entry.components,
        logical_paths: entry.logical_paths,
        shared_physical_part: entry.shared_physical_part,
      });
    }
    totalLogicalFiles += localTransport.logical.file_count;
    totalLogicalBytes += localTransport.logical.bytes;
    totalPhysicalParts += localTransport.transport.unique_part_count;
    totalPhysicalPartBytes += localTransport.transport.unique_part_bytes;
    maximumPhysicalPartBytes = Math.max(
      maximumPhysicalPartBytes,
      localTransport.transport.max_part_bytes,
    );
  }
  const evidence = {
    policy: "exact-local-modular-part-only-parent-plus-qwen-vae-siblings-range-cors",
    bundles,
    total_logical_artifact_files: totalLogicalFiles,
    total_logical_artifact_bytes: totalLogicalBytes,
    total_physical_transport_parts: totalPhysicalParts,
    total_physical_transport_part_bytes: totalPhysicalPartBytes,
    maximum_physical_transport_part_bytes: maximumPhysicalPartBytes,
    validated: true,
  };
  Object.defineProperty(evidence, "transportTelemetryByPath", {
    value: transportTelemetryByPath,
    enumerable: false,
  });
  return evidence;
}

async function executable(path) {
  try {
    await access(path, fsConstants.X_OK);
    return true;
  } catch {
    return false;
  }
}

async function findChrome() {
  const configured = process.env[CHROME_ENV];
  if (configured) {
    const candidate = resolve(configured);
    if (await executable(candidate)) return candidate;
    throw new Error(`${CHROME_ENV} is not executable: ${candidate}`);
  }
  const candidates = [
    "/opt/google/chrome/chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
  ];
  for (const directory of String(process.env.PATH ?? "").split(delimiter)) {
    if (!directory) continue;
    for (const name of ["google-chrome-stable", "google-chrome", "chromium", "chromium-browser"]) {
      candidates.push(join(directory, name));
    }
  }
  for (const candidate of candidates) {
    if (isAbsolute(candidate) && (await executable(candidate))) return candidate;
  }
  throw new Error(`Chrome/Chromium not found; set ${CHROME_ENV}`);
}

async function commandOutput(executable, arguments_, timeoutMs = 10_000) {
  const child = spawn(executable, arguments_, { stdio: ["ignore", "pipe", "pipe"] });
  let stdout = "";
  let stderr = "";
  child.stdout.on("data", (chunk) => {
    stdout = appendBounded(stdout, chunk);
  });
  child.stderr.on("data", (chunk) => {
    stderr = appendBounded(stderr, chunk);
  });
  const { code, signal } = await new Promise((resolveCommand, rejectCommand) => {
    let timer;
    let settled = false;
    const settle = (callback, value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      child.off("error", onError);
      child.off("exit", onExit);
      callback(value);
    };
    const onError = (error) => settle(rejectCommand, error);
    const onExit = (exitCode, exitSignal) =>
      settle(resolveCommand, { code: exitCode, signal: exitSignal });
    child.once("error", onError);
    child.once("exit", onExit);
    timer = setTimeout(() => {
      child.kill("SIGKILL");
      settle(
        rejectCommand,
        new Error(`${executable} ${arguments_.join(" ")} timed out after ${timeoutMs} ms`),
      );
    }, timeoutMs);
  });
  if (code !== 0) {
    throw new Error(
      `${executable} ${arguments_.join(" ")} failed (code=${code}, signal=${signal}): ${stderr.trim()}`,
    );
  }
  return stdout;
}

async function readProcProcess(pid, depth = 0) {
  const statLine = await readFile(`/proc/${pid}/stat`, "utf8");
  const commandEnd = statLine.lastIndexOf(")");
  if (commandEnd < 0) throw new Error(`/proc/${pid}/stat has no command terminator`);
  const fields = statLine.slice(commandEnd + 2).trim().split(/\s+/);
  const ppid = Number(fields[1]);
  if (!Number.isSafeInteger(ppid) || ppid < 0) {
    throw new Error(`/proc/${pid}/stat has invalid parent PID ${fields[1]}`);
  }
  let command;
  try {
    command = (await readFile(`/proc/${pid}/cmdline`))
      .toString("utf8")
      .replaceAll("\0", " ")
      .trim();
  } catch {
    command = statLine.slice(statLine.indexOf("(") + 1, commandEnd);
  }
  return {
    pid,
    ppid,
    command: command.slice(0, MAX_CAPTURED_PROCESS_COMMAND_CHARS),
    depth,
  };
}

async function readProcDescendants(rootPid) {
  const descendants = [];
  const pending = [{ pid: rootPid, depth: 0 }];
  const seen = new Set([rootPid]);
  while (pending.length > 0) {
    const parent = pending.pop();
    let childPids = [];
    try {
      const tasks = await readdir(`/proc/${parent.pid}/task`, { withFileTypes: true });
      const childLists = await Promise.all(
        tasks
          .filter((entry) => entry.isDirectory() && /^\d+$/.test(entry.name))
          .map(async (entry) => {
            try {
              return await readFile(
                `/proc/${parent.pid}/task/${entry.name}/children`,
                "utf8",
              );
            } catch {
              return "";
            }
          }),
      );
      childPids = Array.from(
        new Set(
          childLists
            .flatMap((children) => children.trim().split(/\s+/))
            .filter(Boolean)
            .map(Number)
            .filter((pid) => Number.isSafeInteger(pid) && pid > 1),
        ),
      );
    } catch {
      // Chrome children can exit while procfs is enumerated.
    }
    for (const pid of childPids) {
      if (seen.has(pid)) continue;
      seen.add(pid);
      try {
        const child = await readProcProcess(pid, parent.depth + 1);
        descendants.push(child);
        pending.push(child);
      } catch {
        // Chrome helpers can exit between the child list and their process record.
      }
    }
  }
  return descendants;
}

function parseCounter(value) {
  return /^\d+$/.test(value ?? "") ? Number(value) : null;
}

function parseNvidiaPmonHeader(header) {
  const columns = header
    .replace(/^\s*#\s*/, "")
    .trim()
    .toLowerCase()
    .split(/\s+/);
  const column = (...names) =>
    names.map((name) => columns.indexOf(name)).find((index) => index >= 0);
  const indexes = {
    date: column("date"),
    time: column("time"),
    gpu: column("gpu"),
    pid: column("pid"),
    type: column("type"),
    sm: column("sm"),
    memory: column("mem"),
    framebuffer: column("fb", "fbmem"),
    command: column("command", "name"),
  };
  for (const [name, index] of Object.entries(indexes)) {
    if (index === undefined) {
      throw new Error(
        `nvidia-smi pmon header omits required ${name} column: ${JSON.stringify(columns)}`,
      );
    }
  }
  return { indexes, minimumFields: Math.max(...Object.values(indexes)) + 1 };
}

function parseNvidiaPmonLine(line, layout) {
  const fields = line.trim().split(/\s+/);
  if (fields.length < layout.minimumFields) return null;
  const { indexes } = layout;
  return {
    sampleKey: `${fields[indexes.date]} ${fields[indexes.time]}`,
    row: {
      gpu_index: parseCounter(fields[indexes.gpu]),
      pid: parseCounter(fields[indexes.pid]),
      process_type: fields[indexes.type],
      sm_percent: parseCounter(fields[indexes.sm]),
      memory_percent: parseCounter(fields[indexes.memory]),
      framebuffer_mib: parseCounter(fields[indexes.framebuffer]),
      command: fields.slice(indexes.command).join(" "),
    },
  };
}

async function recordNativeNvidiaGpuInterval(rootPid, evidence, sampleKey, pmonRows) {
  const sampledAt = Date.now();
  evidence.sample_attempts += 1;
  const processes = [];
  try {
    processes.push(await readProcProcess(rootPid));
  } catch {
    // The stop path can race the root Chrome process after the final pmon interval.
  }
  processes.push(...(await readProcDescendants(rootPid)));
  const gpuProcesses = processes
    .filter((process) => process.command.includes("--type=gpu-process"))
    .map((process) => ({ pid: process.pid, ppid: process.ppid, command: process.command }));
  for (const process of gpuProcesses) {
    const observed = evidence.observed_gpu_processes.get(process.pid);
    if (!observed && evidence.observed_gpu_processes.size >= MAX_TRACKED_CHROME_GPU_PROCESSES) {
      evidence.dropped_observed_gpu_processes += 1;
      continue;
    }
    evidence.observed_gpu_processes.set(process.pid, {
      ...process,
      first_seen_at_ms: observed?.first_seen_at_ms ?? sampledAt,
      last_seen_at_ms: sampledAt,
      observed_intervals: (observed?.observed_intervals ?? 0) + 1,
    });
  }
  const gpuProcessPids = new Set(gpuProcesses.map((process) => process.pid));
  const aggregate = aggregateChromeGpuInterval(pmonRows, gpuProcessPids);
  evidence.samples += 1;
  evidence.matched_rows += aggregate.matched_rows.length;
  if (aggregate.matched_rows.length > 0) evidence.matched_sample_intervals += 1;
  if (aggregate.aggregate_framebuffer_mib > 0 && aggregate.aggregate_sm_percent > 0) {
    evidence.active_sample_intervals += 1;
  }
  evidence.peak_aggregate_framebuffer_mib = Math.max(
    evidence.peak_aggregate_framebuffer_mib,
    aggregate.aggregate_framebuffer_mib,
  );
  evidence.peak_aggregate_sm_percent = Math.max(
    evidence.peak_aggregate_sm_percent,
    aggregate.aggregate_sm_percent,
  );
  evidence.peak_process_sm_percent = Math.max(
    evidence.peak_process_sm_percent,
    aggregate.max_process_sm_percent,
  );
  for (const row of aggregate.matched_rows) {
    evidence.gpu_indexes.add(row.gpu_index);
    evidence.pids.add(row.pid);
  }
  if (evidence.sample_records.length === MAX_RECORDED_GPU_SAMPLES) {
    evidence.sample_records.shift();
    evidence.dropped_sample_records += 1;
  }
  evidence.sample_records.push({
    at_ms: sampledAt,
    native_sample_key: sampleKey,
    chrome_gpu_process_pids: [...gpuProcessPids].sort((left, right) => left - right),
    ...aggregate,
  });
}

async function startNativeGpuMonitor(rootPid) {
  const inventoryOutput = await commandOutput("nvidia-smi", [
    "--query-gpu=index,uuid,name,driver_version,memory.total",
    "--format=csv,noheader,nounits",
  ]);
  const gpuInventory = inventoryOutput
    .trim()
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => {
      const [index, uuid, name, driverVersion, memoryTotalMib] = line
        .split(",")
        .map((value) => value.trim());
      return {
        index: Number(index),
        uuid,
        name,
        driver_version: driverVersion,
        memory_total_mib: Number(memoryTotalMib),
      };
    });
  if (
    gpuInventory.length === 0 ||
    gpuInventory.some(
      (gpu) =>
        !Number.isInteger(gpu.index) ||
        gpu.index < 0 ||
        !gpu.uuid.startsWith("GPU-") ||
        !gpu.name ||
        !gpu.driver_version ||
        !Number.isFinite(gpu.memory_total_mib) ||
        gpu.memory_total_mib <= 0,
    )
  ) {
    throw new Error(`nvidia-smi returned an invalid NVIDIA GPU inventory: ${inventoryOutput.trim()}`);
  }

  const evidence = {
    provider: "nvidia-smi",
    root_chrome_pid: rootPid,
    workload_window_started_at_ms: Date.now(),
    gpu_inventory: gpuInventory,
    interval_aggregation_policy: GPU_INTERVAL_AGGREGATION_POLICY,
    maximum_framebuffer_bytes_exclusive: LOW_VRAM_DEVICE_CAP_BYTES,
    sample_attempts: 0,
    samples: 0,
    matched_rows: 0,
    matched_sample_intervals: 0,
    active_sample_intervals: 0,
    peak_aggregate_framebuffer_mib: 0,
    peak_aggregate_sm_percent: 0,
    peak_process_sm_percent: 0,
    observed_gpu_processes: new Map(),
    dropped_observed_gpu_processes: 0,
    gpu_indexes: new Set(),
    pids: new Set(),
    sample_records: [],
    dropped_sample_records: 0,
    sample_error_count: 0,
    sample_errors: [],
  };
  const monitorArguments = [
    "pmon",
    "-s",
    "um",
    "-d",
    String(GPU_SAMPLE_INTERVAL_MS / 1_000),
    "-o",
    "DT",
  ];
  const monitor = spawn("nvidia-smi", monitorArguments, {
    stdio: ["ignore", "pipe", "pipe"],
  });
  let monitorStdout = "";
  let monitorStderr = "";
  let stdoutRemainder = "";
  let layout;
  let pendingKey;
  let pendingRows = [];
  let intervalChain = Promise.resolve();
  const recordMonitorError = (error) => {
    evidence.sample_error_count += 1;
    if (evidence.sample_errors.length < MAX_RECORDED_MONITOR_ERRORS) {
      evidence.sample_errors.push(error instanceof Error ? error.message : String(error));
    }
  };
  const enqueueInterval = (sampleKey, rows) => {
    intervalChain = intervalChain.then(async () => {
      try {
        await recordNativeNvidiaGpuInterval(rootPid, evidence, sampleKey, rows);
      } catch (error) {
        recordMonitorError(error);
      }
    });
  };
  const processMonitorLine = (line) => {
    if (/^\s*#/.test(line)) {
      if (/\bgpu\s+pid\b/i.test(line) && /\bdate\s+time\b/i.test(line)) {
        try {
          layout = parseNvidiaPmonHeader(line);
        } catch (error) {
          recordMonitorError(error);
        }
      }
      return;
    }
    if (!layout || line.trim() === "") return;
    const parsed = parseNvidiaPmonLine(line, layout);
    if (!parsed) return;
    if (pendingKey !== undefined && parsed.sampleKey !== pendingKey) {
      enqueueInterval(pendingKey, pendingRows);
      pendingRows = [];
    }
    pendingKey = parsed.sampleKey;
    pendingRows.push(parsed.row);
  };
  monitor.stdout.on("data", (chunk) => {
    monitorStdout = appendBounded(monitorStdout, chunk).slice(-MAX_CAPTURED_MONITOR_BYTES);
    stdoutRemainder += chunk.toString();
    const lines = stdoutRemainder.split(/\r?\n/);
    stdoutRemainder = lines.pop() ?? "";
    for (const line of lines) processMonitorLine(line);
  });
  monitor.stderr.on("data", (chunk) => {
    monitorStderr = appendBounded(monitorStderr, chunk).slice(-MAX_CAPTURED_MONITOR_BYTES);
  });
  const monitorClosed = new Promise((resolveClose) => {
    monitor.once("error", (error) =>
      resolveClose({ code: null, signal: null, error: error.message }),
    );
    monitor.once("close", (code, signal) => resolveClose({ code, signal }));
  });
  await new Promise((resolveSpawn, rejectSpawn) => {
    monitor.once("spawn", resolveSpawn);
    monitor.once("error", rejectSpawn);
  });

  let stoppedResult;
  return {
    async requestWindow(startEpochMs, endEpochMs) {
      if (stoppedResult) {
        return nativeGpuRequestWindow(stoppedResult, startEpochMs, endEpochMs);
      }
      await intervalChain;
      return nativeGpuRequestWindow(
        {
          ...evidence,
          observed_gpu_processes: [...evidence.observed_gpu_processes.values()].sort(
            (left, right) => left.pid - right.pid,
          ),
        },
        startEpochMs,
        endEpochMs,
      );
    },
    async stop() {
      if (stoppedResult) return stoppedResult;
      const exitedBeforeStop = monitor.exitCode !== null || monitor.signalCode !== null;
      if (!exitedBeforeStop) monitor.kill("SIGTERM");
      let monitorExit = await Promise.race([
        monitorClosed,
        delay(5_000).then(() => null),
      ]);
      if (!monitorExit) {
        monitor.kill("SIGKILL");
        monitorExit = await monitorClosed;
      }
      if (stdoutRemainder.trim() !== "") processMonitorLine(stdoutRemainder);
      if (pendingKey !== undefined) enqueueInterval(pendingKey, pendingRows);
      await intervalChain;

      const result = {
        ...evidence,
        monitor_arguments: monitorArguments,
        monitor_exit: monitorExit,
        monitor_exited_before_stop: exitedBeforeStop,
        monitor_stdout_tail: monitorStdout,
        monitor_stderr_tail: monitorStderr,
        workload_window_finished_at_ms: Date.now(),
        workload_window_elapsed_ms: Date.now() - evidence.workload_window_started_at_ms,
        observed_gpu_processes: [...evidence.observed_gpu_processes.values()].sort(
          (left, right) => left.pid - right.pid,
        ),
        gpu_indexes: [...evidence.gpu_indexes].sort((left, right) => left - right),
        pids: [...evidence.pids].sort((left, right) => left - right),
        observed_peak_aggregate_framebuffer_bytes:
          evidence.peak_aggregate_framebuffer_mib * 1024 * 1024,
      };
      const validationFailures = [];
      if (result.observed_gpu_processes.length === 0) {
        validationFailures.push("no Chrome GPU-process descendant was observed");
      }
      if (result.matched_sample_intervals < MIN_GPU_MATCHED_SAMPLE_INTERVALS) {
        validationFailures.push(
          `nvidia-smi matched Chrome GPU PIDs in ${result.matched_sample_intervals} intervals; at least ${MIN_GPU_MATCHED_SAMPLE_INTERVALS} are required`,
        );
      }
      if (result.active_sample_intervals < MIN_GPU_ACTIVE_SAMPLE_INTERVALS) {
        validationFailures.push("nvidia-smi did not observe Chrome GPU compute activity");
      }
      if (result.observed_peak_aggregate_framebuffer_bytes <= 0) {
        validationFailures.push("aggregate Chrome GPU framebuffer use was not positive");
      }
      if (
        result.observed_peak_aggregate_framebuffer_bytes >=
        result.maximum_framebuffer_bytes_exclusive
      ) {
        validationFailures.push(
          `aggregate Chrome GPU framebuffer peaked at ${result.observed_peak_aggregate_framebuffer_bytes} bytes; low-VRAM mode requires strictly below ${result.maximum_framebuffer_bytes_exclusive}`,
        );
      }
      if (!layout) {
        validationFailures.push("persistent nvidia-smi pmon emitted no parseable dated header");
      }
      if (exitedBeforeStop) {
        validationFailures.push(
          `persistent nvidia-smi pmon exited before workload completion: ${JSON.stringify(monitorExit)}`,
        );
      }
      if (result.sample_error_count !== 0) {
        validationFailures.push(
          `persistent nvidia-smi monitor encountered ${result.sample_error_count} sampling errors`,
        );
      }
      if (result.dropped_observed_gpu_processes !== 0) {
        validationFailures.push(
          `${result.dropped_observed_gpu_processes} Chrome GPU-process observations exceeded the bounded inventory`,
        );
      }
      stoppedResult = {
        ...result,
        validation_failures: validationFailures,
        validated: validationFailures.length === 0,
      };
      return stoppedResult;
    },
  };
}

function nativeGpuRequestWindow(attestation, startEpochMs, endEpochMs) {
  if (
    !Number.isFinite(startEpochMs) ||
    !Number.isFinite(endEpochMs) ||
    endEpochMs <= startEpochMs
  ) {
    throw new Error(`invalid native GPU request window ${startEpochMs}..${endEpochMs}`);
  }
  const sampleRecords = (attestation?.sample_records ?? []).filter(
    (sample) => sample?.at_ms >= startEpochMs && sample?.at_ms <= endEpochMs,
  );
  const matchedSampleIntervals = sampleRecords.filter(
    (sample) => Array.isArray(sample?.matched_rows) && sample.matched_rows.length > 0,
  ).length;
  const activeSampleIntervals = sampleRecords.filter(
    (sample) => sample?.aggregate_framebuffer_mib > 0 && sample?.aggregate_sm_percent > 0,
  ).length;
  const observedPeakAggregateFramebufferBytes =
    sampleRecords.reduce(
      (maximum, sample) => Math.max(maximum, Number(sample?.aggregate_framebuffer_mib ?? 0)),
      0,
    ) *
    1024 *
    1024;
  const validationFailures = [];
  if (sampleRecords.length === 0) validationFailures.push("no nvidia-smi sample fell inside the request window");
  if (matchedSampleIntervals === 0) validationFailures.push("no request sample matched a Chrome GPU-process PID");
  if (activeSampleIntervals === 0) validationFailures.push("no request sample observed Chrome GPU compute activity");
  if (observedPeakAggregateFramebufferBytes <= 0) {
    validationFailures.push("request-window aggregate Chrome GPU framebuffer use was not positive");
  }
  if (observedPeakAggregateFramebufferBytes >= LOW_VRAM_DEVICE_CAP_BYTES) {
    validationFailures.push(
      `request-window Chrome GPU framebuffer peaked at ${observedPeakAggregateFramebufferBytes} bytes`,
    );
  }
  if (!Array.isArray(attestation?.observed_gpu_processes) || attestation.observed_gpu_processes.length === 0) {
    validationFailures.push("no Chrome GPU-process descendant was observed during the workload");
  }
  return {
    provider: attestation?.provider,
    interval_aggregation_policy: attestation?.interval_aggregation_policy,
    maximum_framebuffer_bytes_exclusive: attestation?.maximum_framebuffer_bytes_exclusive,
    window_start_epoch_ms: startEpochMs,
    window_end_epoch_ms: endEpochMs,
    sample_records: sampleRecords,
    matched_sample_intervals: matchedSampleIntervals,
    active_sample_intervals: activeSampleIntervals,
    observed_peak_aggregate_framebuffer_bytes: observedPeakAggregateFramebufferBytes,
    observed_gpu_processes: attestation?.observed_gpu_processes ?? [],
    validation_failures: validationFailures,
    validated: validationFailures.length === 0,
  };
}

function exactIntegerForJson(value) {
  if (typeof value !== "bigint") {
    throw new Error(`expected bigint for exact JSON integer, got ${typeof value}`);
  }
  if (
    value >= BigInt(Number.MIN_SAFE_INTEGER) &&
    value <= BigInt(Number.MAX_SAFE_INTEGER)
  ) {
    return Number(value);
  }
  return value.toString();
}

function nonNegativeBigInt(value) {
  if (typeof value === "bigint") return value >= 0n ? value : null;
  if (Number.isSafeInteger(value) && value >= 0) return BigInt(value);
  return null;
}

function systemErrorEvidence(phase, error) {
  const errno = Number.isSafeInteger(error?.errno) ? error.errno : null;
  const absoluteErrno = errno === null ? null : Math.abs(errno);
  const storage_exhaustion =
    error?.code === "EDQUOT" || absoluteErrno === 122
      ? "per-user-or-group-quota-exhausted"
      : error?.code === "ENOSPC" || absoluteErrno === 28
        ? "filesystem-capacity-exhausted"
        : null;
  return {
    phase,
    code: typeof error?.code === "string" ? error.code : null,
    errno,
    syscall: typeof error?.syscall === "string" ? error.syscall : null,
    storage_exhaustion,
    message: error instanceof Error ? error.message : String(error),
  };
}

async function quotaAwareAllocationProbe(path, requestedBytes) {
  if (!Number.isSafeInteger(requestedBytes) || requestedBytes <= 0) {
    throw new Error("quota-aware allocation probe size must be a positive safe integer");
  }
  const evidence = {
    method: "bounded-real-write-plus-fsync-and-delete",
    attempted: true,
    requested_bytes: requestedBytes,
    written_bytes: 0,
    succeeded: false,
    cleanup_succeeded: false,
    errors: [],
  };
  let probeDirectory;
  let probeFile;
  let allocationSucceeded = false;
  try {
    probeDirectory = await mkdtemp(join(path, ".burn-image-chrome-shmem-probe-"));
    probeFile = await open(join(probeDirectory, "allocation.bin"), "wx", 0o600);
    const chunk = Buffer.alloc(
      Math.min(CHROME_SHARED_MEMORY_PROBE_CHUNK_BYTES, requestedBytes),
      0xa5,
    );
    while (evidence.written_bytes < requestedBytes) {
      const remaining = requestedBytes - evidence.written_bytes;
      const requestedWrite = Math.min(chunk.length, remaining);
      const { bytesWritten } = await probeFile.write(chunk, 0, requestedWrite, null);
      if (!Number.isSafeInteger(bytesWritten) || bytesWritten <= 0) {
        throw new Error(`allocation probe made no progress after ${evidence.written_bytes} bytes`);
      }
      evidence.written_bytes += bytesWritten;
    }
    await probeFile.sync();
    allocationSucceeded = true;
  } catch (error) {
    evidence.errors.push(systemErrorEvidence("allocate-and-fsync", error));
  } finally {
    if (probeFile) {
      try {
        await probeFile.close();
      } catch (error) {
        evidence.errors.push(systemErrorEvidence("close", error));
      }
    }
    if (probeDirectory) {
      try {
        await rm(probeDirectory, { recursive: true, force: true });
        evidence.cleanup_succeeded = true;
      } catch (error) {
        evidence.errors.push(systemErrorEvidence("cleanup", error));
      }
    } else {
      evidence.cleanup_succeeded = true;
    }
  }
  evidence.succeeded =
    allocationSucceeded && evidence.cleanup_succeeded && evidence.errors.length === 0;
  return evidence;
}

async function measureChromeSharedMemoryPath(path) {
  const measurement = {
    path,
    exists: false,
    directory: false,
    writable: false,
    statfs: null,
    quota_aware_allocation_probe: {
      method: "bounded-real-write-plus-fsync-and-delete",
      attempted: false,
      requested_bytes: CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES,
      written_bytes: 0,
      succeeded: false,
      cleanup_succeeded: true,
      skipped_reason: null,
      errors: [],
    },
    errors: [],
  };
  try {
    const pathStat = await stat(path);
    measurement.exists = true;
    measurement.directory = pathStat.isDirectory();
    if (!measurement.directory) {
      measurement.errors.push({
        phase: "stat",
        code: null,
        errno: null,
        syscall: null,
        storage_exhaustion: null,
        message: `${path} is not a directory`,
      });
    }
  } catch (error) {
    measurement.errors.push(systemErrorEvidence("stat", error));
  }
  if (measurement.directory) {
    try {
      await access(path, fsConstants.W_OK | fsConstants.X_OK);
      measurement.writable = true;
    } catch (error) {
      measurement.errors.push(systemErrorEvidence("access", error));
    }
  }

  let availableBytes;
  if (measurement.exists) {
    try {
      const fileSystem = await statfs(path, { bigint: true });
      const blockSize = nonNegativeBigInt(fileSystem.bsize);
      const blockCount = nonNegativeBigInt(fileSystem.blocks);
      const freeBlocks = nonNegativeBigInt(fileSystem.bfree);
      const availableBlocks = nonNegativeBigInt(fileSystem.bavail);
      if (
        blockSize === null ||
        blockSize === 0n ||
        blockCount === null ||
        freeBlocks === null ||
        availableBlocks === null
      ) {
        throw new Error("statfs returned invalid or unsafe block counters");
      }
      availableBytes = availableBlocks * blockSize;
      measurement.statfs = {
        arithmetic: "bigint-products; JSON numbers when safe, decimal strings otherwise",
        block_size_bytes: exactIntegerForJson(blockSize),
        blocks: exactIntegerForJson(blockCount),
        free_blocks: exactIntegerForJson(freeBlocks),
        available_blocks: exactIntegerForJson(availableBlocks),
        total_bytes: exactIntegerForJson(blockCount * blockSize),
        free_bytes: exactIntegerForJson(freeBlocks * blockSize),
        available_bytes: exactIntegerForJson(availableBytes),
      };
    } catch (error) {
      measurement.errors.push(systemErrorEvidence("statfs", error));
    }
  }

  if (!measurement.directory) {
    measurement.quota_aware_allocation_probe.skipped_reason = "not-a-directory";
  } else if (!measurement.writable) {
    measurement.quota_aware_allocation_probe.skipped_reason = "not-writable";
  } else if (availableBytes === undefined) {
    measurement.quota_aware_allocation_probe.skipped_reason = "statfs-unavailable";
  } else if (availableBytes < BigInt(CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES)) {
    measurement.quota_aware_allocation_probe.skipped_reason =
      "statfs-available-below-minimum";
  } else {
    measurement.quota_aware_allocation_probe = await quotaAwareAllocationProbe(
      path,
      CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES,
    );
  }
  return measurement;
}

async function inspectChromeSharedMemory() {
  const tempPath = tmpdir();
  const baseEvidence = {
    schema_version: 1,
    platform: process.platform,
    measurement_method:
      "bigint-statfs-global-capacity-plus-bounded-real-write-fsync-probe-for-effective-user-quota",
    byte_serialization:
      "exact JSON number when within JavaScript safe-integer range, otherwise decimal string",
    capacities: {
      dev_shm: null,
      temp_directory: null,
    },
  };
  if (process.platform !== "linux") {
    return {
      ...baseEvidence,
      ...selectChromeSharedMemoryPolicy({
        platform: process.platform,
        devShm: null,
        tempPath,
      }),
      capacities: {
        dev_shm: { path: "/dev/shm", measured: false, reason: "non-linux-platform" },
        temp_directory: { path: tempPath, measured: false, reason: "non-linux-platform" },
      },
    };
  }

  const devShm = await measureChromeSharedMemoryPath("/dev/shm");
  const tempDirectory = await measureChromeSharedMemoryPath(tempPath);
  return {
    ...baseEvidence,
    ...selectChromeSharedMemoryPolicy({
      platform: process.platform,
      devShm,
      tempPath,
    }),
    capacities: {
      dev_shm: devShm,
      temp_directory: tempDirectory,
    },
  };
}

function chromeLaunchArguments(profile, probeUrl, sharedMemoryPolicy) {
  const arguments_ = [
    `--user-data-dir=${profile}`,
    "--no-sandbox",
    "--window-size=1200,900",
    "--ozone-platform=x11",
  ];
  if (sharedMemoryPolicy.disable_dev_shm_usage) {
    arguments_.push("--disable-dev-shm-usage");
  }
  arguments_.push(
    "--enable-gpu",
    "--disable-software-rasterizer",
    "--ignore-gpu-blocklist",
    "--enable-unsafe-webgpu",
    "--enable-dawn-features=allow_unsafe_apis",
    "--enable-webgpu-developer-features",
    "--use-gpu-in-tests",
    "--enable-accelerated-2d-canvas",
    "--use-gl=angle",
    "--use-angle=vulkan",
    "--enable-features=Vulkan,ForceEnableWebGpuInterop,WebGPUService",
    "--no-first-run",
    "--no-default-browser-check",
    "--remote-debugging-port=0",
    "--enable-logging=stderr",
    probeUrl,
  );
  return arguments_;
}

async function startChrome(executablePath, arguments_) {
  if (!process.env.DISPLAY) {
    throw new Error("headful X11 rendered-surface smoke requires DISPLAY");
  }
  const child = spawn(executablePath, arguments_, {
    detached: true,
    stdio: ["ignore", "pipe", "pipe"],
  });
  child.capturedStdout = "";
  child.capturedStderr = "";
  child.stdout.on("data", (chunk) => {
    child.capturedStdout = appendBounded(child.capturedStdout, chunk);
  });
  child.stderr.on("data", (chunk) => {
    child.capturedStderr = appendBounded(child.capturedStderr, chunk);
  });
  await new Promise((resolveSpawn, rejectSpawn) => {
    child.once("spawn", resolveSpawn);
    child.once("error", rejectSpawn);
  });
  return { child, arguments_, process_group_id: child.pid };
}

function processGroupExists(processGroupId) {
  try {
    process.kill(-processGroupId, 0);
    return true;
  } catch (error) {
    return error?.code === "EPERM";
  }
}

async function waitForProcessGroupExit(processGroupId, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (!processGroupExists(processGroupId)) return true;
    await delay(100);
  }
  return !processGroupExists(processGroupId);
}

function signalProcessGroup(processGroupId, signal, errors) {
  if (!Number.isSafeInteger(processGroupId) || processGroupId <= 1) {
    errors.push(`refused to signal invalid Chrome process group ${processGroupId}`);
    return;
  }
  try {
    process.kill(-processGroupId, signal);
  } catch (error) {
    if (error?.code !== "ESRCH") errors.push(`${signal} process group: ${error}`);
  }
}

async function stopChrome(browser) {
  if (!browser?.child) return null;
  const errors = [];
  const processGroupId = browser.process_group_id;
  signalProcessGroup(processGroupId, "SIGTERM", errors);
  let exited = await waitForProcessGroupExit(processGroupId, 5_000);
  if (!exited) {
    signalProcessGroup(processGroupId, "SIGKILL", errors);
    exited = await waitForProcessGroupExit(processGroupId, 5_000);
  }
  if (!exited) errors.push(`Chrome process group ${processGroupId} survived SIGKILL`);
  return {
    root_pid: browser.child.pid,
    process_group_id: processGroupId,
    process_group_exited: exited,
    errors,
  };
}

async function readDevToolsPort(profile, browser, deadline) {
  const path = join(profile, "DevToolsActivePort");
  while (Date.now() < deadline) {
    throwIfInterrupted();
    if (browser.child.exitCode !== null || browser.child.signalCode !== null) {
      throw new Error(
        `Chrome exited before DevTools started (code=${browser.child.exitCode}, signal=${browser.child.signalCode})`,
      );
    }
    try {
      const [line] = (await readFile(path, "utf8")).trim().split(/\r?\n/);
      const port = Number(line);
      if (Number.isSafeInteger(port) && port > 0 && port <= 65535) return port;
    } catch {
      // Chrome publishes DevToolsActivePort atomically when the endpoint is ready.
    }
    await delay(100);
  }
  throw new Error("timed out waiting for Chrome DevToolsActivePort");
}

async function findExactPageTarget(port, exactUrl, browser, deadline) {
  while (Date.now() < deadline) {
    throwIfInterrupted();
    if (browser.child.exitCode !== null || browser.child.signalCode !== null) {
      throw new Error("Chrome exited before opening the exact probe URL");
    }
    try {
      const response = await fetch(`http://127.0.0.1:${port}/json/list`, {
        signal: AbortSignal.timeout(1_000),
      });
      const targets = await response.json();
      const exact = targets.find((target) => target.type === "page" && target.url === exactUrl);
      if (exact?.webSocketDebuggerUrl) return exact;
    } catch {
      // The JSON endpoint can lag DevToolsActivePort briefly.
    }
    await delay(100);
  }
  throw new Error(`timed out waiting for exact Chrome target ${exactUrl}`);
}

function renderRemoteArgument(argument) {
  if (Object.hasOwn(argument ?? {}, "value")) return String(argument.value);
  return argument?.description ?? argument?.type ?? "<unavailable>";
}

async function openCdp(url) {
  if (typeof WebSocket !== "function") {
    throw new Error("this opt-in harness requires a Node runtime with global WebSocket support");
  }
  const socket = new WebSocket(url);
  await new Promise((resolveOpen, rejectOpen) => {
    const timer = setTimeout(() => rejectOpen(new Error("timed out opening CDP WebSocket")), 10_000);
    socket.addEventListener(
      "open",
      () => {
        clearTimeout(timer);
        resolveOpen();
      },
      { once: true },
    );
    socket.addEventListener(
      "error",
      () => {
        clearTimeout(timer);
        rejectOpen(new Error("CDP WebSocket failed to open"));
      },
      { once: true },
    );
  });

  let nextId = 1;
  let socketError;
  const pending = new Map();
  const pageErrors = [];
  const gpuErrors = [];
  const networkFailures = [];
  const ignoredBenignNetworkFailures = [];
  const networkRequestContexts = new Map();
  const events = [];
  const record = (kind, message) => {
    const rendered = String(message ?? "unknown browser error");
    const target = kind === "gpu" ? gpuErrors : pageErrors;
    if (!target.includes(rendered)) target.push(rendered);
  };

  socket.addEventListener("message", (event) => {
    let message;
    try {
      message = JSON.parse(String(event.data));
    } catch (error) {
      socketError = new Error(`invalid CDP message: ${error}`);
      return;
    }
    if (message.id && pending.has(message.id)) {
      const entry = pending.get(message.id);
      pending.delete(message.id);
      clearTimeout(entry.timer);
      if (message.error) entry.reject(new Error(`CDP error: ${JSON.stringify(message.error)}`));
      else entry.resolve(message.result);
      return;
    }
    events.push({ at_ms: Date.now(), method: message.method, params: message.params });
    if (message.method === "Network.requestWillBeSent") {
      const requestId = message.params?.requestId;
      if (typeof requestId !== "string" || requestId.length === 0) {
        record("page", "CDP Network.requestWillBeSent omitted requestId");
      } else {
        const previous = networkRequestContexts.get(requestId) ?? null;
        networkRequestContexts.set(
          requestId,
          captureCdpNetworkRequestContext(message.params, previous),
        );
      }
    } else if (message.method === "Network.loadingFinished") {
      networkRequestContexts.delete(message.params?.requestId);
    } else if (message.method === "Runtime.exceptionThrown") {
      const details = message.params?.exceptionDetails;
      record("page", details?.exception?.description ?? details?.text ?? "uncaught exception");
    } else if (message.method === "Runtime.consoleAPICalled") {
      const rendered = (message.params?.args ?? []).map(renderRemoteArgument).join(" ");
      if (["error", "assert"].includes(message.params?.type)) record("page", rendered);
      if (gpuTerminalDiagnostic(rendered)) record("gpu", rendered);
    } else if (message.method === "Log.entryAdded") {
      const entry = message.params?.entry;
      if (entry?.url?.endsWith("/favicon.ico")) return;
      if (entry?.level === "error") record("page", entry.text);
      if (gpuTerminalDiagnostic(entry?.text)) record("gpu", entry.text);
    } else if (message.method === "Network.loadingFailed") {
      const requestId = message.params?.requestId;
      const requestContext = networkRequestContexts.get(requestId) ?? null;
      const diagnostic = cdpNetworkLoadingFailedDiagnostic(
        message.params,
        requestContext,
      );
      networkRequestContexts.delete(requestId);
      if (diagnostic.proven_benign_favicon) {
        ignoredBenignNetworkFailures.push(diagnostic);
      } else {
        networkFailures.push(diagnostic);
        record("page", `network request failed: ${JSON.stringify(diagnostic)}`);
      }
    } else if (message.method === "Network.responseReceived") {
      const response = message.params?.response;
      if (response?.status >= 400 && !response?.url?.endsWith("/favicon.ico")) {
        record("page", `HTTP ${response.status}: ${response.url}`);
      }
    } else if (message.method === "Inspector.targetCrashed") {
      record("page", "Chrome page target crashed");
    } else if (message.method === "Inspector.detached") {
      record("page", `Chrome inspector detached: ${message.params?.reason ?? "unknown"}`);
    }
  });
  socket.addEventListener("error", () => {
    socketError = new Error("CDP WebSocket error");
  });

  const call = (method, params = {}, timeoutMs = CDP_CALL_TIMEOUT_MS) => {
    if (socketError) return Promise.reject(socketError);
    const id = nextId++;
    socket.send(JSON.stringify({ id, method, params }));
    return new Promise((resolveCall, rejectCall) => {
      const timer = setTimeout(() => {
        pending.delete(id);
        rejectCall(new Error(`CDP ${method} timed out`));
      }, timeoutMs);
      pending.set(id, { resolve: resolveCall, reject: rejectCall, timer });
    });
  };
  await Promise.all([
    call("Runtime.enable"),
    call("Log.enable"),
    call("Network.enable"),
    call("Page.enable"),
    call("Inspector.enable"),
  ]);
  return {
    call,
    pageErrors,
    gpuErrors,
    networkFailures,
    ignoredBenignNetworkFailures,
    events,
    get socketError() {
      return socketError;
    },
    close() {
      if (socket.readyState === WebSocket.OPEN) socket.close();
    },
  };
}

async function evaluateValue(cdp, expression, awaitPromise = false) {
  const evaluation = await cdp.call("Runtime.evaluate", {
    expression,
    awaitPromise,
    returnByValue: true,
  });
  if (evaluation.exceptionDetails) {
    throw new Error(
      `CDP evaluation failed: ${evaluation.exceptionDetails.exception?.description ?? evaluation.exceptionDetails.text}`,
    );
  }
  return evaluation.result?.value;
}

// This pre-navigation request only confirms that launching the real page is worth attempting.
// Runtime hardware proof comes exclusively from PRELOAD_INSTRUMENTATION intercepting the exact
// requestAdapter/requestDevice calls made by the rendered Bevy/WGPU page after navigation.
async function browserLaunchReadinessAdapterInfoWithRetry(cdp, deadline) {
  let lastFailure;
  let attempts = 0;
  while (Date.now() < deadline) {
    attempts += 1;
    throwIfInterrupted();
    try {
      const adapter = await evaluateValue(
        cdp,
        `(async () => {
          if (!navigator.gpu) throw new Error("navigator.gpu is unavailable");
          const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
          if (!adapter) throw new Error("requestAdapter returned null");
          const info = adapter.info ?? {};
          return {
            vendor: info.vendor ?? "",
            architecture: info.architecture ?? "",
            device: info.device ?? "",
            description: info.description ?? "",
            is_fallback_adapter: adapter.isFallbackAdapter ?? info.isFallbackAdapter ?? false,
            features: Array.from(adapter.features ?? []).sort(),
            limits: {
              max_buffer_size: Number(adapter.limits?.maxBufferSize ?? 0),
              max_storage_buffer_binding_size: Number(
                adapter.limits?.maxStorageBufferBindingSize ?? 0
              ),
            },
          };
        })()`,
        true,
      );
      const failures = validateHardwareNvidiaAdapter(adapter);
      if (failures.length > 0) throw new Error(failures.join("; "));
      return { ...adapter, request_attempts: attempts };
    } catch (error) {
      lastFailure = error;
      await delay(500);
    }
  }
  throw new Error(`WebGPU launch-readiness adapter retry window expired: ${lastFailure}`);
}

const PRELOAD_INSTRUMENTATION = `(() => {
  const evidence = {
    engine_session_id: crypto.randomUUID(),
    backend_events: [],
    runtime_events: [],
    progress_events: [],
    ui_events: [],
    output_events: [],
    webgpu_calls: [],
    webgpu_dropped_calls: 0,
    surface_texture_acquisition_count: 0,
    surface_texture_acquisition_failure_count: 0,
    latest_successful_surface_texture_acquisition: null,
    surface_texture_gate_windows: [],
    surface_texture_gate_windows_overflow: 0,
    surface_texture_gate_violation_calls: [],
    surface_texture_gate_violation_calls_overflow: 0,
    surface_texture_gate_overlap_count: 0,
    active_surface_gate: null,
    artifact_progress_events: 0,
    animation_frames: 0,
    installed_at_ms: performance.now(),
  };
  Object.defineProperty(globalThis, "__burnImageRenderedSurface", {
    configurable: false,
    enumerable: false,
    writable: false,
    value: evidence,
  });
  const copyDetail = (event) => {
    let detail;
    try {
      detail = JSON.parse(JSON.stringify(event.detail ?? null));
    } catch {
      detail = { event: "invalid", message: "browser event detail was not serializable" };
    }
    return { at_ms: performance.now(), ...detail };
  };
  const boundedPush = (target, value) => {
    if (target.length < 20000) target.push(value);
  };
  let nextWebGpuRequestId = 1;
  const recordWebGpu = (event, detail = null) => {
    if (evidence.webgpu_calls.length >= 20000) {
      evidence.webgpu_dropped_calls += 1;
      return;
    }
    evidence.webgpu_calls.push({ event, at_ms: performance.now(), detail });
  };
  let pendingPostResumeWindow = null;
  const beginSurfaceTextureAcquisition = (detail) => {
    evidence.surface_texture_acquisition_count += 1;
    const acquisition = {
      call_index: evidence.surface_texture_acquisition_count,
      ...detail,
      succeeded: null,
    };
    if (evidence.active_surface_gate) {
      evidence.active_surface_gate.gated_call_count += 1;
      const violation = {
        run_id: evidence.active_surface_gate.run_id,
        policy: evidence.active_surface_gate.policy,
        ...acquisition,
      };
      if (evidence.surface_texture_gate_violation_calls.length < 128) {
        evidence.surface_texture_gate_violation_calls.push(violation);
        acquisition.violation_record = violation;
      } else {
        evidence.surface_texture_gate_violation_calls_overflow += 1;
      }
    }
    return acquisition;
  };
  const finishSurfaceTextureAcquisition = (acquisition, succeeded, error = null) => {
    acquisition.succeeded = succeeded;
    if (error != null) acquisition.error = String(error);
    if (acquisition.violation_record) {
      acquisition.violation_record.succeeded = succeeded;
      if (error != null) acquisition.violation_record.error = String(error);
      delete acquisition.violation_record;
    }
    if (!succeeded) {
      evidence.surface_texture_acquisition_failure_count += 1;
      return;
    }
    const compact = { ...acquisition };
    evidence.latest_successful_surface_texture_acquisition = compact;
    if (!evidence.active_surface_gate && pendingPostResumeWindow) {
      pendingPostResumeWindow.first_successful_post_resume_acquisition = compact;
      pendingPostResumeWindow = null;
    }
  };
  const adapterOptions = (options) => {
    try {
      return {
        powerPreference: options?.powerPreference ?? null,
        forceFallbackAdapter: options?.forceFallbackAdapter ?? null,
      };
    } catch (error) {
      return { telemetry_error: String(error) };
    }
  };
  const deviceDescriptor = (descriptor) => {
    try {
      return {
        label: descriptor?.label ?? null,
        requiredFeatures: Array.from(
          descriptor?.requiredFeatures ?? [],
          (feature) => String(feature),
        ),
        requiredLimits: Object.fromEntries(
          Object.entries(descriptor?.requiredLimits ?? {}).map(([name, value]) => [
            name,
            typeof value === "bigint" ? Number(value) : value,
          ]),
        ),
      };
    } catch (error) {
      return { telemetry_error: String(error) };
    }
  };
  const deviceInfo = (device) => {
    try {
      if (device?.features == null) {
        throw new Error("resolved GPUDevice.features is unavailable");
      }
      return {
        enabledFeatures: Array.from(device.features, (feature) => String(feature)),
      };
    } catch (error) {
      return { telemetry_error: String(error) };
    }
  };
  const adapterInfo = (adapter) => {
    if (!adapter) return null;
    try {
      const info = adapter.info ?? null;
      return {
        is_fallback_adapter:
          (typeof adapter.isFallbackAdapter === "boolean"
            ? adapter.isFallbackAdapter
            : null) ??
          (typeof info?.isFallbackAdapter === "boolean" ? info.isFallbackAdapter : null),
        vendor: info?.vendor ?? null,
        architecture: info?.architecture ?? null,
        device: info?.device ?? null,
        description: info?.description ?? null,
      };
    } catch (error) {
      return {
        is_fallback_adapter: null,
        vendor: null,
        architecture: null,
        device: null,
        description: null,
        telemetry_error: String(error),
      };
    }
  };
  try {
    const gpu = navigator.gpu;
    if (!gpu) {
      recordWebGpu("navigator-gpu-unavailable");
    } else {
      const requestAdapter = gpu.requestAdapter.bind(gpu);
      Object.defineProperty(gpu, "requestAdapter", {
        configurable: true,
        writable: true,
        value: async (options) => {
          const requestId = nextWebGpuRequestId++;
          const summary = { request_id: requestId, ...adapterOptions(options) };
          recordWebGpu("request-adapter-start", summary);
          try {
            const adapter = await requestAdapter(options);
            recordWebGpu("request-adapter-resolved", {
              request_id: requestId,
              available: adapter != null,
              info: adapterInfo(adapter),
            });
            if (adapter) {
              try {
                const prototype = Object.getPrototypeOf(adapter);
                if (!prototype.__burnImageRenderedRequestDeviceInstrumented) {
                  const requestDevice = prototype.requestDevice;
                  Object.defineProperty(prototype, "requestDevice", {
                    configurable: true,
                    writable: true,
                    value: async function (descriptor) {
                      const deviceRequestId = nextWebGpuRequestId++;
                      const deviceSummary = {
                        request_id: deviceRequestId,
                        adapter_request_id: requestId,
                        ...deviceDescriptor(descriptor),
                      };
                      recordWebGpu("request-device-start", deviceSummary);
                      try {
                        const device = await requestDevice.call(this, descriptor);
                        recordWebGpu("request-device-resolved", {
                          ...deviceSummary,
                          ...deviceInfo(device),
                        });
                        return device;
                      } catch (error) {
                        recordWebGpu("request-device-rejected", {
                          ...deviceSummary,
                          error: String(error),
                        });
                        throw error;
                      }
                    },
                  });
                  Object.defineProperty(
                    prototype,
                    "__burnImageRenderedRequestDeviceInstrumented",
                    { value: true },
                  );
                }
              } catch (error) {
                recordWebGpu("request-device-instrumentation-unavailable", {
                  error: String(error),
                });
              }
            }
            return adapter;
          } catch (error) {
            recordWebGpu("request-adapter-rejected", {
              request_id: requestId,
              error: String(error),
            });
            throw error;
          }
        },
      });
    }
  } catch (error) {
    recordWebGpu("request-adapter-instrumentation-unavailable", {
      error: String(error),
    });
  }
  try {
    const prototype = globalThis.GPUCanvasContext?.prototype;
    if (!prototype || typeof prototype.getCurrentTexture !== "function") {
      recordWebGpu("surface-texture-instrumentation-unavailable", {
        error: "GPUCanvasContext.prototype.getCurrentTexture is unavailable",
      });
    } else {
      const getCurrentTexture = prototype.getCurrentTexture;
      Object.defineProperty(prototype, "getCurrentTexture", {
        configurable: true,
        writable: true,
        value: function (...args) {
          const atMs = performance.now();
          let canvas = null;
          try {
            canvas = this.canvas ?? null;
          } catch {
            // A missing canvas identity is retained as fail-closed evidence below.
          }
          const acquisition = beginSurfaceTextureAcquisition({
            at_ms: atMs,
            canvas_id: canvas?.id ?? null,
            canvas_width: Number(canvas?.width ?? 0),
            canvas_height: Number(canvas?.height ?? 0),
          });
          try {
            const texture = Reflect.apply(getCurrentTexture, this, args);
            finishSurfaceTextureAcquisition(acquisition, true);
            return texture;
          } catch (error) {
            finishSurfaceTextureAcquisition(acquisition, false, error);
            throw error;
          }
        },
      });
      Object.defineProperty(prototype, "__burnImageRenderedSurfaceTextureInstrumented", {
        value: true,
      });
    }
  } catch (error) {
    recordWebGpu("surface-texture-instrumentation-unavailable", {
      error: String(error),
    });
  }
  const recordSurfaceGateRuntimeEvent = (detail) => {
    if (detail.event === "surface_inference_suspended") {
      pendingPostResumeWindow = null;
      if (evidence.active_surface_gate) {
        evidence.surface_texture_gate_overlap_count += 1;
        return;
      }
      evidence.active_surface_gate = {
        run_id: detail.run_id,
        policy: detail.policy,
        suspended_at_ms: detail.at_ms,
        acquisition_count_at_suspend: evidence.surface_texture_acquisition_count,
        pre_request_acquisition:
          evidence.latest_successful_surface_texture_acquisition == null
            ? null
            : { ...evidence.latest_successful_surface_texture_acquisition },
        gated_call_count: 0,
        resumed_at_ms: null,
        terminal: null,
        acquisition_count_at_resume: null,
        first_successful_post_resume_acquisition: null,
      };
      return;
    }
    if (detail.event !== "surface_inference_resumed") return;
    let window = evidence.active_surface_gate;
    if (!window || JSON.stringify(window.run_id) !== JSON.stringify(detail.run_id)) {
      evidence.surface_texture_gate_overlap_count += 1;
      window = {
        run_id: detail.run_id,
        policy: detail.policy,
        suspended_at_ms: null,
        acquisition_count_at_suspend: null,
        pre_request_acquisition: null,
        gated_call_count: 0,
        first_successful_post_resume_acquisition: null,
      };
    }
    window.resume_policy = detail.policy;
    window.resumed_at_ms = detail.at_ms;
    window.terminal = detail.terminal;
    window.acquisition_count_at_resume = evidence.surface_texture_acquisition_count;
    evidence.active_surface_gate = null;
    if (evidence.surface_texture_gate_windows.length < 64) {
      evidence.surface_texture_gate_windows.push(window);
      pendingPostResumeWindow = window;
    } else {
      evidence.surface_texture_gate_windows_overflow += 1;
      pendingPostResumeWindow = null;
    }
  };
  window.addEventListener(${JSON.stringify(BACKEND_EVENT_NAME)}, (event) => {
    boundedPush(evidence.backend_events, copyDetail(event));
  });
  window.addEventListener(${JSON.stringify(RUNTIME_EVENT_NAME)}, (event) => {
    const detail = copyDetail(event);
    boundedPush(evidence.runtime_events, detail);
    recordSurfaceGateRuntimeEvent(detail);
  });
  window.addEventListener(${JSON.stringify(PROGRESS_EVENT_NAME)}, (event) => {
    const detail = copyDetail(event);
    if (detail.event === "artifact_progress") {
      evidence.artifact_progress_events += 1;
    } else {
      boundedPush(evidence.progress_events, detail);
    }
  });
  window.addEventListener(${JSON.stringify(UI_CONTRACT_EVENT_NAME)}, (event) => {
    boundedPush(evidence.ui_events, copyDetail(event));
  });
  window.addEventListener(${JSON.stringify(OUTPUT_READY_EVENT_NAME)}, (event) => {
    boundedPush(evidence.output_events, copyDetail(event));
  });
  const frame = () => {
    evidence.animation_frames += 1;
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
})();`;

async function pageSnapshot(cdp) {
  return evaluateValue(
    cdp,
    `(() => {
      const canvas = document.querySelector("#burn-image");
      const rect = canvas?.getBoundingClientRect();
      return {
        url: location.href,
        time_origin_epoch_ms: Number(performance.timeOrigin),
        secure_context: globalThis.isSecureContext === true,
        ready_state: document.readyState,
        title: document.title,
        host_status: document.querySelector("#status")?.textContent ?? "",
        device_pixel_ratio: Number(devicePixelRatio),
        animation_frames: globalThis.__burnImageRenderedSurface?.animation_frames ?? 0,
        engine_session_id: globalThis.__burnImageRenderedSurface?.engine_session_id ?? null,
        webgpu_calls: globalThis.__burnImageRenderedSurface?.webgpu_calls ?? [],
        webgpu_dropped_calls:
          globalThis.__burnImageRenderedSurface?.webgpu_dropped_calls ?? 0,
        surface_texture_acquisition_count:
          globalThis.__burnImageRenderedSurface?.surface_texture_acquisition_count ?? 0,
        surface_texture_acquisition_failure_count:
          globalThis.__burnImageRenderedSurface?.surface_texture_acquisition_failure_count ?? 0,
        latest_successful_surface_texture_acquisition:
          globalThis.__burnImageRenderedSurface
            ?.latest_successful_surface_texture_acquisition ?? null,
        surface_texture_gate_windows:
          globalThis.__burnImageRenderedSurface?.surface_texture_gate_windows ?? [],
        surface_texture_gate_windows_overflow:
          globalThis.__burnImageRenderedSurface?.surface_texture_gate_windows_overflow ?? 0,
        surface_texture_gate_violation_calls:
          globalThis.__burnImageRenderedSurface?.surface_texture_gate_violation_calls ?? [],
        surface_texture_gate_violation_calls_overflow:
          globalThis.__burnImageRenderedSurface
            ?.surface_texture_gate_violation_calls_overflow ?? 0,
        surface_texture_gate_overlap_count:
          globalThis.__burnImageRenderedSurface?.surface_texture_gate_overlap_count ?? 0,
        active_surface_gate:
          globalThis.__burnImageRenderedSurface?.active_surface_gate ?? null,
        surface_inference_state:
          document.documentElement.dataset.surfaceInference ?? null,
        backend_events: globalThis.__burnImageRenderedSurface?.backend_events ?? [],
        runtime_events: globalThis.__burnImageRenderedSurface?.runtime_events ?? [],
        progress_events: globalThis.__burnImageRenderedSurface?.progress_events ?? [],
        ui_events: globalThis.__burnImageRenderedSurface?.ui_events ?? [],
        output_events: globalThis.__burnImageRenderedSurface?.output_events ?? [],
        artifact_progress_events:
          globalThis.__burnImageRenderedSurface?.artifact_progress_events ?? 0,
        canvas: canvas ? {
          width: Number(canvas.width),
          height: Number(canvas.height),
          client_width: Number(canvas.clientWidth),
          client_height: Number(canvas.clientHeight),
          rect_width: Number(rect?.width ?? 0),
          rect_height: Number(rect?.height ?? 0),
        } : null,
      };
    })()`,
  );
}

async function waitForReadySurface(cdp, browser, expectedUrl, deadline) {
  let firstReadyAt;
  let firstReadyFrames;
  let lastSnapshot;
  while (Date.now() < deadline) {
    throwIfInterrupted();
    if (browser.child.exitCode !== null || browser.child.signalCode !== null) {
      throw new Error("Chrome exited while waiting for the rendered Bevy surface");
    }
    if (cdp.socketError) throw cdp.socketError;
    lastSnapshot = await pageSnapshot(cdp);
    const backendEvents = lastSnapshot.backend_events ?? [];
    const failed = backendEvents.find((event) => event?.event === "failed");
    if (failed) throw new Error(`Bevy backend failed: ${JSON.stringify(failed)}`);
    if (cdp.pageErrors.length > 0 || cdp.gpuErrors.length > 0) {
      throw new Error(
        `browser emitted page/GPU errors: ${JSON.stringify([...cdp.pageErrors, ...cdp.gpuErrors])}`,
      );
    }
    const ready = [...backendEvents].reverse().find((event) => event?.event === "ready");
    const snapshotFailures = validateRenderedSurfaceSnapshot(lastSnapshot, expectedUrl);
    if (ready && snapshotFailures.length === 0) {
      if (firstReadyAt === undefined) {
        firstReadyAt = Date.now();
        firstReadyFrames = lastSnapshot.animation_frames;
      }
      if (Date.now() - firstReadyAt >= STABILITY_MS) {
        if (lastSnapshot.animation_frames - firstReadyFrames < 2) {
          throw new Error("Bevy surface did not continue presenting animation frames");
        }
        return { snapshot: lastSnapshot, ready, stable_ms: Date.now() - firstReadyAt };
      }
    } else {
      firstReadyAt = undefined;
      firstReadyFrames = undefined;
    }
    await delay(250);
  }
  throw new Error(`timed out waiting for stable Bevy GPU-ready surface: ${JSON.stringify(lastSnapshot)}`);
}

function throwBrowserFailure(cdp, browser, snapshot, phase) {
  if (browser.child.exitCode !== null || browser.child.signalCode !== null) {
    throw new Error(`Chrome exited during ${phase}`);
  }
  if (cdp.socketError) throw cdp.socketError;
  const backendFailure = snapshot?.backend_events?.find((event) => event?.event === "failed");
  const runtimeFailure = snapshot?.runtime_events?.find((event) => event?.event === "failed");
  const runFailure = snapshot?.progress_events?.find((event) =>
    ["run_failed", "run_cancelled"].includes(event?.event),
  );
  if (backendFailure || runtimeFailure || runFailure) {
    throw new Error(
      `${phase} emitted a terminal app event: ${JSON.stringify(backendFailure ?? runtimeFailure ?? runFailure)}`,
    );
  }
  if (cdp.pageErrors.length > 0 || cdp.gpuErrors.length > 0) {
    throw new Error(
      `${phase} emitted page/GPU errors: ${JSON.stringify([...cdp.pageErrors, ...cdp.gpuErrors])}`,
    );
  }
}

async function waitForTurboUiReady(cdp, browser, expectedUrl, deadline) {
  let lastSnapshot;
  while (Date.now() < deadline) {
    throwIfInterrupted();
    lastSnapshot = await pageSnapshot(cdp);
    throwBrowserFailure(cdp, browser, lastSnapshot, "Turbo model initialization");
    const surfaceFailures = validateRenderedSurfaceSnapshot(lastSnapshot, expectedUrl);
    const backendReady = lastSnapshot.backend_events?.find((event) => event?.event === "ready");
    const runtimeReady = lastSnapshot.runtime_events?.find(
      (event) => event?.event === "ready" && event?.model === TURBO_MODEL_ID,
    );
    const packedF16Plan = lastSnapshot.runtime_events?.find(
      (event) => event?.event === "packed_f16_resource_plan",
    );
    const uiContract = [...(lastSnapshot.ui_events ?? [])]
      .reverse()
      .find((event) => event?.event === "ready");
    if (
      surfaceFailures.length === 0 &&
      backendReady &&
      runtimeReady &&
      packedF16Plan &&
      uiContract?.model === TURBO_MODEL_ID &&
      uiContract.width === 1024 &&
      uiContract.height === 1024 &&
      uiContract.prompt_enabled === true &&
      uiContract.seed_enabled === true
    ) {
      return { snapshot: lastSnapshot, backendReady, runtimeReady, packedF16Plan, uiContract };
    }
    await delay(500);
  }
  throw new Error(`timed out waiting for ordinary Turbo 1024 UI readiness: ${JSON.stringify(lastSnapshot)}`);
}

function uiControlPoint(uiContract, control) {
  const x = uiContract?.[`${control}_x`];
  const y = uiContract?.[`${control}_y`];
  if (!Number.isFinite(x) || !Number.isFinite(y) || x <= 0 || y <= 0) {
    throw new Error(`ordinary UI ${control} center is invalid: ${JSON.stringify({ x, y })}`);
  }
  if (uiContract?.[`${control}_enabled`] !== true) {
    throw new Error(`ordinary UI ${control} control is disabled`);
  }
  return { x, y };
}

async function dispatchCdpMouseClick(cdp, uiContract, control, clickCount = 1) {
  const point = uiControlPoint(uiContract, control);
  const atMs = Date.now();
  await cdp.call("Input.dispatchMouseEvent", {
    type: "mouseMoved",
    x: point.x,
    y: point.y,
    button: "none",
    buttons: 0,
    pointerType: "mouse",
  });
  await delay(POINTER_SETTLE_MS);
  await cdp.call("Input.dispatchMouseEvent", {
    type: "mousePressed",
    x: point.x,
    y: point.y,
    button: "left",
    buttons: 1,
    clickCount,
    pointerType: "mouse",
  });
  await delay(POINTER_SETTLE_MS);
  await cdp.call("Input.dispatchMouseEvent", {
    type: "mouseReleased",
    x: point.x,
    y: point.y,
    button: "left",
    buttons: 0,
    clickCount,
    pointerType: "mouse",
  });
  await delay(POINTER_SETTLE_MS);
  return {
    at_ms: atMs,
    control,
    click_count: clickCount,
    center: point,
    methods: [
      "Input.dispatchMouseEvent(mouseMoved)",
      "Input.dispatchMouseEvent(mousePressed)",
      "Input.dispatchMouseEvent(mouseReleased)",
    ],
  };
}

async function waitForControlFocus(cdp, browser, control) {
  const deadline = Date.now() + INPUT_FOCUS_TIMEOUT_MS;
  let lastSnapshot;
  while (Date.now() < deadline) {
    throwIfInterrupted();
    lastSnapshot = await pageSnapshot(cdp);
    throwBrowserFailure(cdp, browser, lastSnapshot, `ordinary ${control} focus`);
    const uiContract = [...(lastSnapshot.ui_events ?? [])]
      .reverse()
      .find((event) => event?.event === "ready");
    if (uiContract?.[`${control}_focused`] === true) {
      return { snapshot: lastSnapshot, uiContract };
    }
    await delay(50);
  }
  throw new Error(
    `ordinary ${control} did not acquire Bevy InputFocus after its real CDP pointer click: ${JSON.stringify(lastSnapshot)}`,
  );
}

function cdpPrintableKey(character) {
  if (character === " ") {
    return { key: " ", code: "Space", virtualKeyCode: 32 };
  }
  if (/^[A-Za-z]$/.test(character)) {
    const upper = character.toUpperCase();
    return { key: character, code: `Key${upper}`, virtualKeyCode: upper.charCodeAt(0) };
  }
  if (/^[0-9]$/.test(character)) {
    return { key: character, code: `Digit${character}`, virtualKeyCode: character.charCodeAt(0) };
  }
  if (character === ".") {
    return { key: character, code: "Period", virtualKeyCode: 190 };
  }
  throw new Error(`the fixed smoke input contains an unsupported CDP key: ${JSON.stringify(character)}`);
}

async function typeTextViaCdpKeyboard(cdp, value) {
  for (const character of value) {
    const key = cdpPrintableKey(character);
    await cdp.call("Input.dispatchKeyEvent", {
      type: "keyDown",
      key: key.key,
      code: key.code,
      text: character,
      unmodifiedText: character,
      windowsVirtualKeyCode: key.virtualKeyCode,
      nativeVirtualKeyCode: key.virtualKeyCode,
    });
    await cdp.call("Input.dispatchKeyEvent", {
      type: "keyUp",
      key: key.key,
      code: key.code,
      windowsVirtualKeyCode: key.virtualKeyCode,
      nativeVirtualKeyCode: key.virtualKeyCode,
    });
  }
}

async function replaceEditableTextViaCdp(
  cdp,
  browser,
  uiContract,
  control,
  value,
  { selectExisting = false } = {},
) {
  const click = await dispatchCdpMouseClick(cdp, uiContract, control, 1);
  const focused = await waitForControlFocus(cdp, browser, control);
  const selectionClicks = [];
  if (selectExisting) {
    // Bevy's EditableText maps a second Pointer<Press> to SelectWordAtPoint and every
    // third-or-later consecutive press to SelectAll. These are two more real browser clicks
    // after the focus click above, all within Bevy's 500 ms multi-click interval.
    selectionClicks.push(await dispatchCdpMouseClick(cdp, uiContract, control, 2));
    selectionClicks.push(await dispatchCdpMouseClick(cdp, uiContract, control, 3));
  }
  await delay(50);
  await typeTextViaCdpKeyboard(cdp, value);
  return {
    at_ms: Date.now(),
    control,
    value,
    focus_click: click,
    focus_event: focused.uiContract,
    replacement_mode: selectExisting
      ? "bevy-editable-text-triple-click-select-all"
      : "known-empty-direct-keyboard-entry",
    selection_clicks: selectionClicks,
    methods: ["Input.dispatchMouseEvent", "Input.dispatchKeyEvent(per-character text)"],
  };
}

async function waitForTextValue(cdp, browser, eventName, expected, startIndex, deadline) {
  let lastSnapshot;
  while (Date.now() < deadline) {
    throwIfInterrupted();
    lastSnapshot = await pageSnapshot(cdp);
    throwBrowserFailure(cdp, browser, lastSnapshot, `ordinary ${eventName} keyboard input`);
    const changed = (lastSnapshot.ui_events ?? [])
      .slice(startIndex)
      .find(
        (event) => event?.event === eventName && event?.value === expected,
      );
    if (changed) return { snapshot: lastSnapshot, changed };
    await delay(100);
  }
  throw new Error(
    `Bevy ${eventName} did not receive exact CDP-entered value ${JSON.stringify(expected)}: ${JSON.stringify(lastSnapshot)}`,
  );
}

async function waitForRunStarted(cdp, browser, startIndex, deadline) {
  let lastSnapshot;
  while (Date.now() < deadline) {
    throwIfInterrupted();
    lastSnapshot = await pageSnapshot(cdp);
    throwBrowserFailure(cdp, browser, lastSnapshot, "ordinary Run mouse input");
    const started = (lastSnapshot.progress_events ?? []).slice(startIndex).find(
      (event) => event?.event === "run_started" && event?.model === TURBO_MODEL_ID,
    );
    if (started) return { snapshot: lastSnapshot, started };
    await delay(100);
  }
  throw new Error(`Bevy Run control did not start Turbo inference: ${JSON.stringify(lastSnapshot)}`);
}

async function waitForRunReady(cdp, browser, startIndex, deadline) {
  let lastSnapshot;
  while (Date.now() < deadline) {
    throwIfInterrupted();
    lastSnapshot = await pageSnapshot(cdp);
    throwBrowserFailure(cdp, browser, lastSnapshot, "post-prompt Run readiness");
    const uiContract = (lastSnapshot.ui_events ?? []).slice(startIndex).find(
      (event) =>
        event?.event === "ready" &&
        event?.model === TURBO_MODEL_ID &&
        event?.width === 1024 &&
        event?.height === 1024 &&
        event?.run_enabled === true,
    );
    if (uiContract) return { snapshot: lastSnapshot, uiContract };
    await delay(100);
  }
  throw new Error(`ordinary Run control did not become enabled after prompt entry: ${JSON.stringify(lastSnapshot)}`);
}

async function waitForTurboOutput(
  cdp,
  browser,
  expectedUrl,
  deadline,
  { progressStartIndex = 0, outputStartIndex = 0 } = {},
) {
  let lastSnapshot;
  while (Date.now() < deadline) {
    throwIfInterrupted();
    lastSnapshot = await pageSnapshot(cdp);
    throwBrowserFailure(cdp, browser, lastSnapshot, "ordinary Turbo inference");
    const started = lastSnapshot.progress_events?.slice(progressStartIndex).find(
      (event) => event?.event === "run_started" && event?.model === TURBO_MODEL_ID,
    );
    const completed = lastSnapshot.progress_events?.slice(progressStartIndex).find(
      (event) =>
        event?.event === "run_completed" &&
        JSON.stringify(event?.run_id) === JSON.stringify(started?.run_id),
    );
    const output = lastSnapshot.output_events?.slice(outputStartIndex).find(
      (event) =>
        event?.event === "ready" &&
        event?.model === TURBO_MODEL_ID &&
        event?.width === 1024 &&
        event?.height === 1024 &&
        outputJobIdMatchesNumericRunId(event?.job_id, started?.run_id),
    );
    if (
      started &&
      completed &&
      output &&
      validateRenderedSurfaceSnapshot(lastSnapshot, expectedUrl).length === 0
    ) {
      await delay(2_000);
      const finalSnapshot = await pageSnapshot(cdp);
      throwBrowserFailure(cdp, browser, finalSnapshot, "post-output presentation");
      return { snapshot: finalSnapshot, started, completed, output };
    }
    await delay(1_000);
  }
  throw new Error(`timed out waiting for ordinary Turbo 1024 output: ${JSON.stringify(lastSnapshot)}`);
}

async function waitForSaveReady(cdp, browser, deadline) {
  let lastSnapshot;
  while (Date.now() < deadline) {
    throwIfInterrupted();
    lastSnapshot = await pageSnapshot(cdp);
    throwBrowserFailure(cdp, browser, lastSnapshot, "ordinary Save PNG readiness");
    const uiContract = [...(lastSnapshot.ui_events ?? [])]
      .reverse()
      .find(
        (event) =>
          event?.event === "ready" &&
          event?.model === TURBO_MODEL_ID &&
          event?.width === 1024 &&
          event?.height === 1024 &&
          event?.save_enabled === true,
      );
    if (uiContract) return { snapshot: lastSnapshot, uiContract };
    await delay(100);
  }
  throw new Error(`ordinary Save PNG control did not become enabled: ${JSON.stringify(lastSnapshot)}`);
}

async function waitForCompletedPngDownload(cdp, browser, downloadDir, eventStartIndex, deadline) {
  let willBegin;
  let completed;
  while (Date.now() < deadline) {
    throwIfInterrupted();
    const downloadEvents = cdp.events.slice(eventStartIndex).filter(
      (event) =>
        event.method === "Browser.downloadWillBegin" ||
        event.method === "Browser.downloadProgress",
    );
    willBegin ??= downloadEvents.find(
      (event) =>
        event.method === "Browser.downloadWillBegin" &&
        /^burn-image-[0-9]+\.png$/.test(String(event.params?.suggestedFilename ?? "")),
    );
    if (willBegin) {
      completed = downloadEvents.find(
        (event) =>
          event.method === "Browser.downloadProgress" &&
          event.params?.guid === willBegin.params.guid &&
          event.params?.state === "completed",
      );
    }
    if (completed) break;
    const snapshot = await pageSnapshot(cdp);
    throwBrowserFailure(cdp, browser, snapshot, "production Save PNG download");
    await delay(100);
  }
  if (!willBegin || !completed) {
    throw new Error("timed out waiting for matching completed Browser download events");
  }

  const fileName = willBegin.params.suggestedFilename;
  const path = join(downloadDir, fileName);
  let metadata;
  while (Date.now() < deadline) {
    try {
      metadata = await stat(path);
      if (metadata.isFile() && metadata.size > 0) break;
    } catch {
      // Browser.downloadProgress can arrive just before the final directory entry is observable.
    }
    await delay(50);
  }
  if (!metadata?.isFile() || metadata.size <= 0) {
    throw new Error(`completed browser PNG download is missing: ${path}`);
  }
  const canonicalRoot = await realpath(downloadDir);
  const canonicalPath = await realpath(path);
  if (!canonicalPath.startsWith(`${canonicalRoot}/`)) {
    throw new Error(`downloaded PNG escaped its evidence directory: ${canonicalPath}`);
  }
  const bytes = await readFile(canonicalPath);
  const signatureHex = bytes.subarray(0, 8).toString("hex");
  const ihdr =
    bytes.length >= 24
      ? {
          length: bytes.readUInt32BE(8),
          type: bytes.subarray(12, 16).toString("ascii"),
          width: bytes.readUInt32BE(16),
          height: bytes.readUInt32BE(20),
        }
      : null;
  if (
    signatureHex !== "89504e470d0a1a0a" ||
    ihdr?.length !== 13 ||
    ihdr?.type !== "IHDR" ||
    ihdr?.width !== 1024 ||
    ihdr?.height !== 1024
  ) {
    throw new Error(
      `production Save PNG returned an invalid 1024x1024 file: ${JSON.stringify({ signatureHex, ihdr })}`,
    );
  }
  return {
    path: canonicalPath,
    file_name: fileName,
    bytes: bytes.length,
    sha256: sha256(bytes),
    signature_hex: signatureHex,
    ihdr,
    browser_events: { will_begin: willBegin, completed },
  };
}

function assertValidatedTurboOutputBeforeSave(output, runId) {
  const failures = [];
  if (output?.event !== "ready") failures.push("output-ready event is absent");
  if (!isCanonicalU64DecimalString(output?.job_id)) {
    failures.push("output-ready job ID is not a canonical u64 decimal string");
  } else if (!outputJobIdMatchesNumericRunId(output.job_id, runId)) {
    failures.push("output-ready job ID differs from the exact decimal representation of run_id");
  }
  if (output?.model !== TURBO_MODEL_ID) failures.push("output-ready model is not Turbo");
  if (output?.width !== 1024 || output?.height !== 1024) {
    failures.push("output-ready dimensions are not 1024x1024");
  }
  if (output?.backend !== LOW_VRAM_BACKEND) {
    failures.push(
      "output-ready backend is not the Turbo preloaded packed-F16 dense-F32-per-stage policy",
    );
  }
  if (output?.artifacts_verified !== true) failures.push("output-ready artifacts are unverified");
  if (output?.artifact_content_digest !== TURBO_PRODUCTION_CONTENT_DIGEST) {
    failures.push("output-ready artifact digest is not the canonical Turbo digest");
  }
  if (output?.numeric_format !== "f16-qwen-vision-f32") {
    failures.push("output-ready numeric format does not identify the production artifact");
  }
  if (failures.length > 0) {
    throw new Error(`refusing to click Save for an unvalidated output: ${failures.join("; ")}`);
  }
}

async function captureCanvasScreenshot(cdp, path) {
  const canvas = await evaluateValue(
    cdp,
    `(() => {
      const canvas = document.querySelector("#burn-image");
      const rect = canvas?.getBoundingClientRect();
      return rect ? {
        x: Number(rect.left),
        y: Number(rect.top),
        width: Number(rect.width),
        height: Number(rect.height),
      } : null;
    })()`,
  );
  if (
    !canvas ||
    ![canvas.x, canvas.y, canvas.width, canvas.height].every(Number.isFinite) ||
    canvas.width <= 0 ||
    canvas.height <= 0
  ) {
    throw new Error(`cannot capture invalid canvas rectangle: ${JSON.stringify(canvas)}`);
  }
  const capture = await cdp.call("Page.captureScreenshot", {
    format: "png",
    fromSurface: true,
    clip: { ...canvas, scale: 1 },
  });
  const bytes = Buffer.from(capture?.data ?? "", "base64");
  if (bytes.length < 1024) throw new Error("Chrome returned an empty canvas screenshot");
  await writeFile(path, bytes);
  return { path, bytes: bytes.length, sha256: sha256(bytes), data: bytes };
}

function changedByteCount(before, after) {
  const common = Math.min(before.length, after.length);
  let changed = Math.abs(before.length - after.length);
  for (let index = 0; index < common; index += 1) {
    if (before[index] !== after[index]) changed += 1;
  }
  return changed;
}

async function captureScreenshot(cdp, path) {
  const capture = await cdp.call("Page.captureScreenshot", {
    format: "png",
    captureBeyondViewport: true,
    fromSurface: true,
  });
  const bytes = Buffer.from(capture?.data ?? "", "base64");
  const pngSignature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  if (bytes.length < 1024 || !bytes.subarray(0, 8).equals(pngSignature)) {
    throw new Error("Chrome returned an empty or invalid rendered-window PNG screenshot");
  }
  await writeFile(path, bytes);
  return { path, bytes: bytes.length, sha256: sha256(bytes) };
}

async function main() {
  const enabled = process.env[ENABLE_ENV] === "1";
  const singleRequestModelMode = process.env[MODEL_ENV] === "1";
  const multiRequestModelMode = process.env[MULTI_REQUEST_MODEL_ENV] === "1";
  const modelMode = singleRequestModelMode || multiRequestModelMode;
  const requiredDeviceFeatures = modelMode
    ? BOOGU_WEB_REQUIRED_DEVICE_FEATURES
    : GENERIC_WEB_REQUIRED_DEVICE_FEATURES;
  const validateOnly = process.env[VALIDATE_ONLY_ENV] === "1";
  if (!enabled && !modelMode && !validateOnly) {
    console.log(
      `burn_image rendered-surface smoke: skipped (set ${ENABLE_ENV}=1, ${MODEL_ENV}=1, or ${MULTI_REQUEST_MODEL_ENV}=1; CI may set ${VALIDATE_ONLY_ENV}=1)`,
    );
    return;
  }
  if ([enabled, singleRequestModelMode, multiRequestModelMode, validateOnly].filter(Boolean).length !== 1) {
    throw new Error(
      `${ENABLE_ENV}, ${MODEL_ENV}, ${MULTI_REQUEST_MODEL_ENV}, and ${VALIDATE_ONLY_ENV} are mutually exclusive`,
    );
  }

  const startedAt = Date.now();
  const outputDir = resolve(
    process.env[OUTPUT_ENV] ?? join(tmpdir(), `burn-image-rendered-surface-${process.pid}`),
  );
  await mkdir(outputDir, { recursive: true });
  const reportPath = join(
    outputDir,
    modelMode
      ? multiRequestModelMode
        ? "burn-image-rendered-turbo-1024-multi-request-report.json"
        : "burn-image-rendered-turbo-1024-report.json"
      : "burn-image-rendered-surface-report.json",
  );
  const sourceEvidence = await validateCommittedSources();
  if (validateOnly) {
    const report = {
      schema_version: 1,
      test: "burn_image_browser_rendered_surface_source_validation",
      claim: "source-contract-only; no browser, GPU, rendered-surface, or numerical parity claim",
      ok: true,
      source_evidence: sourceEvidence,
      elapsed_ms: Date.now() - startedAt,
    };
    await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`);
    console.log(JSON.stringify(report, null, 2));
    return;
  }

  const requestedQwenBlock0ExecutionMode =
    process.env[QWEN_BLOCK0_EXECUTION_MODE_ENV] ??
    TURBO_QWEN_BLOCK0_ORDINARY_MODE;
  if (
    modelMode &&
    ![
      TURBO_QWEN_BLOCK0_SERIALIZED_DIAGNOSTIC_MODE,
      TURBO_QWEN_BLOCK0_ORDINARY_MODE,
    ].includes(requestedQwenBlock0ExecutionMode)
  ) {
    throw new Error(
      `${QWEN_BLOCK0_EXECUTION_MODE_ENV} must be ${TURBO_QWEN_BLOCK0_SERIALIZED_DIAGNOSTIC_MODE} or ${TURBO_QWEN_BLOCK0_ORDINARY_MODE}`,
    );
  }

  const timeoutMs = modelMode ? parseModelTimeout() : parseTimeout();
  const deadline = startedAt + timeoutMs;
  const wwwOutDir = resolve(process.env[WWW_OUT_ENV] ?? join(crateDir, "www/out"));
  const artifactRoot = modelMode
    ? resolve(
        process.env[MODEL_ARTIFACT_ROOT_ENV] ??
          join(
            repoRoot,
            ".artifacts/cdn-upload-modular/aberration.technology/model",
          ),
      )
    : undefined;
  const indexBytes = await readFile(committedIndexPath);
  let server;
  let browser;
  let cdp;
  let profile;
  let screenshot;
  let canvasBefore;
  let canvasAfter;
  let downloadedPng;
  let interactionEvidence;
  let gpuMonitor;
  let gpuAttestation;
  let modelEvidence;
  let multiRequestEvidence;
  let firstRequestDraft;
  let secondRequestDraft;
  let evidence;
  let testedPackageIdentity;
  let modelBaseUrl;
  let modelBaseUrls;
  let launchReadinessWebGpuProbe;
  let bevyBackendReady;
  let runtimeWebGpuCalls;
  let runtimeWebGpuDroppedCalls;
  let runtimeWebGpuAdapterAttestation;
  let chromeExecutable;
  let chromeArguments_;
  let chromeSharedMemory;
  let outcome;
  let failure;
  try {
    chromeSharedMemory = await inspectChromeSharedMemory();
    testedPackageIdentity = await collectTestedPackageIdentity(wwwOutDir);
    const hosted = await createAppServer(indexBytes, wwwOutDir, artifactRoot);
    server = hosted.server;
    const baseUrl = `http://127.0.0.1:${hosted.port}`;
    const transport = await validateServedApp(
      baseUrl,
      sourceEvidence,
      testedPackageIdentity,
    );
    const modelTransport = modelMode
      ? await validateModelArtifactTransport(baseUrl, artifactRoot)
      : null;
    const query = new URLSearchParams({
      "rendered-surface-smoke": sourceEvidence.committed_index_sha256,
    });
    if (modelMode) {
      query.set("rendered-model-smoke", "1");
      query.set("surface-gate", "1");
      query.set("qwen-block0-execution-mode", requestedQwenBlock0ExecutionMode);
      query.set("variant", "turbo");
      query.set("profile", "production");
      // This harness is the strict low-VRAM release gate. Keep it independent from the app's
      // evolving default residency policy and from the resident packed-F16 qualification.
      query.set("residency", "low-vram");
      modelBaseUrl = `${baseUrl}/model/${TURBO_BUNDLE}`;
      modelBaseUrls = MODEL_BUNDLES.map((bundle) => `${baseUrl}/model/${bundle}`);
      query.set("artifacts", modelBaseUrl);
    }
    const exactUrl = `${baseUrl}/index.html?${query}`;
    const probeUrl = `${baseUrl}/probe`;

    chromeExecutable = await findChrome();
    // Model-mode Cache Storage can hold tens of gigabytes of authenticated ranges.
    // Keep the isolated Chrome profile beside the caller-selected evidence directory
    // instead of the often-small /tmp tmpfs, then remove it during normal cleanup.
    const profileParent = modelMode ? outputDir : tmpdir();
    profile = await mkdtemp(join(profileParent, "burn-image-rendered-surface-profile-"));
    chromeArguments_ = chromeLaunchArguments(profile, probeUrl, chromeSharedMemory);
    browser = await startChrome(chromeExecutable, chromeArguments_);
    const devToolsDeadline = Math.min(deadline, Date.now() + DEVTOOLS_TIMEOUT_MS);
    const port = await readDevToolsPort(profile, browser, devToolsDeadline);
    const target = await findExactPageTarget(port, probeUrl, browser, devToolsDeadline);
    cdp = await openCdp(target.webSocketDebuggerUrl);
    const browserVersion = await cdp.call("Browser.getVersion");
    const launchReadinessAdapter =
      await browserLaunchReadinessAdapterInfoWithRetry(cdp, devToolsDeadline);
    launchReadinessWebGpuProbe = {
      policy: RENDERED_LAUNCH_READINESS_PROBE_POLICY,
      runtime_hardware_proof: false,
      adapter: launchReadinessAdapter,
    };
    if (modelMode) gpuMonitor = await startNativeGpuMonitor(browser.child.pid);
    await cdp.call("Page.addScriptToEvaluateOnNewDocument", { source: PRELOAD_INSTRUMENTATION });
    const navigation = await cdp.call("Page.navigate", { url: exactUrl });
    if (navigation?.errorText) throw new Error(`Chrome navigation failed: ${navigation.errorText}`);
    const readySurface = await waitForReadySurface(cdp, browser, exactUrl, deadline);
    bevyBackendReady = readySurface.ready;
    let finalSnapshot = readySurface.snapshot;
    if (modelMode) {
      const uiReady = await waitForTurboUiReady(cdp, browser, exactUrl, deadline);
      canvasBefore = await captureCanvasScreenshot(
        cdp,
        join(outputDir, "burn-image-rendered-turbo-1024-before.png"),
      );
      const firstCdpEventStart = cdp.events.length;
      const promptEventStart = uiReady.snapshot.ui_events?.length ?? 0;
      const promptInput = await replaceEditableTextViaCdp(
        cdp,
        browser,
        uiReady.uiContract,
        "prompt",
        MODEL_PROMPT,
      );
      const promptReady = await waitForTextValue(
        cdp,
        browser,
        "prompt_changed",
        MODEL_PROMPT,
        promptEventStart,
        deadline,
      );
      const runReady = await waitForRunReady(cdp, browser, promptEventStart, deadline);
      const intermediateSeedEventStart = runReady.snapshot.ui_events?.length ?? 0;
      const intermediateSeedInput = await replaceEditableTextViaCdp(
        cdp,
        browser,
        runReady.uiContract,
        "seed",
        "1",
        { selectExisting: true },
      );
      const intermediateSeedReady = await waitForTextValue(
        cdp,
        browser,
        "seed_changed",
        "1",
        intermediateSeedEventStart,
        deadline,
      );
      const seedEventStart = intermediateSeedReady.snapshot.ui_events?.length ?? 0;
      const seedInput = await replaceEditableTextViaCdp(
        cdp,
        browser,
        runReady.uiContract,
        "seed",
        "0",
        { selectExisting: true },
      );
      const seedReady = await waitForTextValue(
        cdp,
        browser,
        "seed_changed",
        "0",
        seedEventStart,
        deadline,
      );
      const firstStarts = {
        runtime: seedReady.snapshot.runtime_events?.length ?? 0,
        progress: seedReady.snapshot.progress_events?.length ?? 0,
        output: seedReady.snapshot.output_events?.length ?? 0,
        ui: promptEventStart,
        surface: seedReady.snapshot.surface_texture_gate_windows?.length ?? 0,
        surface_acquisitions: seedReady.snapshot.surface_texture_acquisition_count ?? 0,
        surface_acquisition_failures:
          seedReady.snapshot.surface_texture_acquisition_failure_count ?? 0,
        surface_violations:
          seedReady.snapshot.surface_texture_gate_violation_calls?.length ?? 0,
        surface_violation_overflow:
          seedReady.snapshot.surface_texture_gate_violation_calls_overflow ?? 0,
        cdp: firstCdpEventStart,
      };
      const firstRequestStartEpochMs = Date.now();
      const runClick = await dispatchCdpMouseClick(cdp, runReady.uiContract, "run");
      const runStarted = await waitForRunStarted(
        cdp,
        browser,
        firstStarts.progress,
        deadline,
      );
      const completed = await waitForTurboOutput(cdp, browser, exactUrl, deadline, {
        progressStartIndex: firstStarts.progress,
        outputStartIndex: firstStarts.output,
      });
      assertValidatedTurboOutputBeforeSave(completed.output, completed.started?.run_id);
      const saveReady = await waitForSaveReady(cdp, browser, deadline);
      const downloadDir = join(outputDir, "downloads");
      await mkdir(downloadDir, { recursive: true });
      await cdp.call("Browser.setDownloadBehavior", {
        behavior: "allow",
        downloadPath: downloadDir,
        eventsEnabled: true,
      });
      const downloadEventStart = cdp.events.length;
      const saveClick = await dispatchCdpMouseClick(cdp, saveReady.uiContract, "save");
      downloadedPng = await waitForCompletedPngDownload(
        cdp,
        browser,
        downloadDir,
        downloadEventStart,
        deadline,
      );
      interactionEvidence = {
        mechanism: "cdp-keyboard-and-mouse",
        prompt_typed_via_cdp: true,
        prompt_value: MODEL_PROMPT,
        prompt_event: promptReady.changed,
        prompt_input: promptInput,
        seed_typed_via_cdp: true,
        seed_value: "0",
        seed_event: seedReady.changed,
        seed_input: seedInput,
        seed_intermediate_value: "1",
        seed_intermediate_event: intermediateSeedReady.changed,
        seed_intermediate_input: intermediateSeedInput,
        run_clicked_via_cdp: true,
        run_input: runClick,
        run_started_event: runStarted.started,
        save_clicked_via_cdp: true,
        save_input: saveClick,
      };
      const firstCanvasAfter = await captureCanvasScreenshot(
        cdp,
        join(
          outputDir,
          multiRequestModelMode
            ? "burn-image-rendered-turbo-1024-request-1-after.png"
            : "burn-image-rendered-turbo-1024-after.png",
        ),
      );
      const firstScreenshot = await captureScreenshot(
        cdp,
        join(
          outputDir,
          multiRequestModelMode
            ? "burn-image-rendered-turbo-1024-request-1.png"
            : "burn-image-rendered-turbo-1024.png",
        ),
      );
      const firstRequestEndEpochMs = Date.now();
      const firstEndSnapshot = await pageSnapshot(cdp);
      throwBrowserFailure(cdp, browser, firstEndSnapshot, "first post-capture model qualification");
      const firstEnds = {
        runtime: firstEndSnapshot.runtime_events?.length ?? 0,
        progress: firstEndSnapshot.progress_events?.length ?? 0,
        output: firstEndSnapshot.output_events?.length ?? 0,
        ui: firstEndSnapshot.ui_events?.length ?? 0,
        surface: firstEndSnapshot.surface_texture_gate_windows?.length ?? 0,
        surface_acquisitions: firstEndSnapshot.surface_texture_acquisition_count ?? 0,
        surface_acquisition_failures:
          firstEndSnapshot.surface_texture_acquisition_failure_count ?? 0,
        surface_violations:
          firstEndSnapshot.surface_texture_gate_violation_calls?.length ?? 0,
        surface_violation_overflow:
          firstEndSnapshot.surface_texture_gate_violation_calls_overflow ?? 0,
        cdp: cdp.events.length,
      };
      const initialRuntimeEvents = firstEndSnapshot.runtime_events.slice(0, firstStarts.runtime);
      const firstRuntimeEvents = firstEndSnapshot.runtime_events.slice(
        firstStarts.runtime,
        firstEnds.runtime,
      );
      const firstProgressEvents = firstEndSnapshot.progress_events.slice(
        firstStarts.progress,
        firstEnds.progress,
      );
      const firstOutputEvents = firstEndSnapshot.output_events.slice(
        firstStarts.output,
        firstEnds.output,
      );
      const firstUiEvents = firstEndSnapshot.ui_events.slice(firstStarts.ui, firstEnds.ui);
      const firstSurfaceTextureGateWindows = firstEndSnapshot.surface_texture_gate_windows.slice(
        firstStarts.surface,
        firstEnds.surface,
      );
      const firstSurfaceTextureGateViolationCalls =
        firstEndSnapshot.surface_texture_gate_violation_calls.slice(
          firstStarts.surface_violations,
          firstEnds.surface_violations,
        );
      const firstArtifactTraffic = firstRuntimeEvents.find(
        (event) => event?.event === "artifact_traffic",
      )?.traffic ?? null;
      const packedF16PreloadEvents = initialRuntimeEvents.filter(
        (event) => event?.event === "packed_f16_denoiser_preload",
      );
      const packedF16PreloadTraffic = packedF16PreloadEvents?.at(-1)?.traffic ?? null;
      const firstPackedF16LifecycleEvent = firstRuntimeEvents.find(
        (event) => event?.event === "packed_f16_denoiser_lifecycle",
      );
      const firstPackedF16Lifecycle = firstPackedF16LifecycleEvent?.lifecycle ?? null;
      const firstPackedF16DmdVaeHandoff = firstRuntimeEvents.find(
        (event) => event?.event === "packed_f16_dmd_vae_handoff",
      ) ?? null;
      const firstPackedF16PreDmdInputDiagnostics = firstRuntimeEvents.find(
        (event) => event?.event === "packed_f16_pre_dmd_input_diagnostics",
      ) ?? null;
      const firstPackedF16QwenHostEmbedding = firstRuntimeEvents.find(
        (event) => event?.event === "packed_f16_qwen_host_embedding",
      ) ?? null;
      const firstPackedF16QwenPreHandoffDiagnostics = firstRuntimeEvents.find(
        (event) => event?.event === "packed_f16_qwen_pre_handoff_diagnostics",
      ) ?? null;
      const firstPackedF16QwenBlock0ExecutionDiagnostics = firstRuntimeEvents.find(
        (event) => event?.event === "packed_f16_qwen_block0_execution_diagnostics",
      ) ?? null;
      const firstPackedF16QwenBlock0PostSyncDiagnostic = firstRuntimeEvents.find(
        (event) => event?.event === "packed_f16_qwen_block0_post_sync_diagnostic",
      ) ?? null;
      const firstPackedF16QwenPostHandoffDiagnostics = firstRuntimeEvents.find(
        (event) => event?.event === "packed_f16_qwen_post_handoff_diagnostics",
      ) ?? null;
      const cdpPreloadNetworkTraffic = summarizeTurboPreloadCdpNetwork(
        cdp.events,
        firstEndSnapshot,
        modelBaseUrls,
        modelTransport.transportTelemetryByPath,
      );
      const firstRequestSnapshot = {
        ...firstEndSnapshot,
        runtime_events: firstRuntimeEvents,
        progress_events: firstProgressEvents,
        output_events: firstOutputEvents,
        ui_events: firstUiEvents,
      };
      const firstCdpNetworkTraffic = summarizeTurboRequestCdpNetwork(
        cdp.events,
        firstRequestSnapshot,
        modelBaseUrls,
        modelTransport.transportTelemetryByPath,
      );
      const firstCdpDmdNetworkTraffic = summarizeTurboDmdCdpNetwork(
        cdp.events,
        firstRequestSnapshot,
        modelBaseUrls,
        modelTransport.transportTelemetryByPath,
      );
      firstRequestDraft = {
        request_ordinal: 1,
        qwen_block0_execution_mode: requestedQwenBlock0ExecutionMode,
        page_identity: {
          engine_session_id: firstEndSnapshot.engine_session_id,
          url: firstEndSnapshot.url,
          time_origin_epoch_ms: firstEndSnapshot.time_origin_epoch_ms,
        },
        request_epoch_window: {
          start_epoch_ms: firstRequestStartEpochMs,
          end_epoch_ms: firstRequestEndEpochMs,
        },
        event_boundaries: {
          runtime_start_index: firstStarts.runtime,
          runtime_end_index: firstEnds.runtime,
          progress_start_index: firstStarts.progress,
          progress_end_index: firstEnds.progress,
          output_start_index: firstStarts.output,
          output_end_index: firstEnds.output,
          ui_start_index: firstStarts.ui,
          ui_end_index: firstEnds.ui,
          surface_start_index: firstStarts.surface,
          surface_end_index: firstEnds.surface,
          cdp_start_index: firstStarts.cdp,
          cdp_end_index: firstEnds.cdp,
        },
        cdp_event_count: firstEnds.cdp - firstStarts.cdp,
        runtime_events: firstRuntimeEvents,
        progress_events: firstProgressEvents,
        output_events: firstOutputEvents,
        ui_events: firstUiEvents,
        surface_texture_gate_windows: firstSurfaceTextureGateWindows,
        surface_texture_gate_windows_overflow:
          firstEndSnapshot.surface_texture_gate_windows_overflow,
        surface_texture_gate_violation_calls: firstSurfaceTextureGateViolationCalls,
        surface_texture_gate_violation_calls_overflow_start:
          firstStarts.surface_violation_overflow,
        surface_texture_gate_violation_calls_overflow_end:
          firstEnds.surface_violation_overflow,
        surface_texture_gate_overlap_count:
          firstEndSnapshot.surface_texture_gate_overlap_count,
        surface_texture_acquisition_count_start: firstStarts.surface_acquisitions,
        surface_texture_acquisition_count_end: firstEnds.surface_acquisitions,
        surface_texture_acquisition_failure_count_start:
          firstStarts.surface_acquisition_failures,
        surface_texture_acquisition_failure_count_end:
          firstEnds.surface_acquisition_failures,
        active_surface_gate_after_request: firstEndSnapshot.active_surface_gate,
        surface_inference_state_after_request: firstEndSnapshot.surface_inference_state,
        artifact_traffic: firstArtifactTraffic,
        packed_f16_denoiser_lifecycle: firstPackedF16Lifecycle,
        packed_f16_dmd_vae_handoff: firstPackedF16DmdVaeHandoff,
        packed_f16_qwen_host_embedding: firstPackedF16QwenHostEmbedding,
        packed_f16_qwen_block0_execution_diagnostics:
          firstPackedF16QwenBlock0ExecutionDiagnostics,
        packed_f16_qwen_block0_post_sync_diagnostic:
          firstPackedF16QwenBlock0PostSyncDiagnostic,
        packed_f16_qwen_pre_handoff_diagnostics:
          firstPackedF16QwenPreHandoffDiagnostics,
        packed_f16_qwen_post_handoff_diagnostics:
          firstPackedF16QwenPostHandoffDiagnostics,
        packed_f16_pre_dmd_input_diagnostics: firstPackedF16PreDmdInputDiagnostics,
        cdp_network_traffic: firstCdpNetworkTraffic,
        cdp_dmd_network_traffic: firstCdpDmdNetworkTraffic,
        dmd_runtime_io_attestation: {
          policy: TURBO_DMD_RUNTIME_ZERO_IO_POLICY,
          run_id: runStarted.started.run_id,
          completed_dmd_steps: firstProgressEvents.filter(
            (event) => event?.event === "step" && event?.stage === "dmd",
          ).length,
          traffic: firstPackedF16Lifecycle?.dmd_artifact_traffic ?? null,
          lifecycle_event_at_ms: firstPackedF16LifecycleEvent?.at_ms ?? null,
          runtime_source_sha256: testedPackageIdentity.sources.browser_runtime.sha256,
        },
        modular_artifact_transport: modelTransport,
        served_transport: transport,
        tested_package_identity: testedPackageIdentity,
        ui_contract: saveReady.uiContract,
        output_ready: completed.output,
        interaction: interactionEvidence,
        downloaded_png: downloadedPng,
        canvas_before_png: {
          path: canvasBefore.path,
          bytes: canvasBefore.bytes,
          sha256: canvasBefore.sha256,
        },
        canvas_after_png: {
          path: firstCanvasAfter.path,
          bytes: firstCanvasAfter.bytes,
          sha256: firstCanvasAfter.sha256,
        },
        canvas_png_changed_bytes: changedByteCount(canvasBefore.data, firstCanvasAfter.data),
        model_screenshot_path: firstScreenshot.path,
        model_screenshot_bytes: firstScreenshot.bytes,
        model_screenshot_sha256: firstScreenshot.sha256,
      };

      if (multiRequestModelMode) {
        // Snapshot request 1 before the bounded long-running monitor record rotates while request
        // 2 executes. The same persistent nvidia-smi process continues across both requests.
        firstRequestDraft.native_gpu_attestation = await gpuMonitor.requestWindow(
          firstRequestStartEpochMs,
          firstRequestEndEpochMs,
        );
      }

      if (multiRequestModelMode) {
        const secondStarts = {
          runtime: firstEnds.runtime,
          progress: firstEnds.progress,
          output: firstEnds.output,
          ui: firstEnds.ui,
          surface: firstEnds.surface,
          surface_acquisitions: firstEnds.surface_acquisitions,
          surface_acquisition_failures: firstEnds.surface_acquisition_failures,
          surface_violations: firstEnds.surface_violations,
          surface_violation_overflow: firstEnds.surface_violation_overflow,
          cdp: cdp.events.length,
        };
        const secondSeedInput = await replaceEditableTextViaCdp(
          cdp,
          browser,
          saveReady.uiContract,
          "seed",
          "1",
          { selectExisting: true },
        );
        const secondSeedReady = await waitForTextValue(
          cdp,
          browser,
          "seed_changed",
          "1",
          secondStarts.ui,
          deadline,
        );
        const secondRunReady = resolveTurboSecondRequestRunReadyUiContract({
          uiEvents: secondSeedReady.snapshot.ui_events ?? [],
          uiStartIndex: secondStarts.ui,
          seedChangedEvent: secondSeedReady.changed,
          postRequestUiContract: saveReady.uiContract,
        });
        const secondRequestStartEpochMs = Date.now();
        const secondRunClick = await dispatchCdpMouseClick(
          cdp,
          secondRunReady.uiContract,
          "run",
        );
        const secondRunStarted = await waitForRunStarted(
          cdp,
          browser,
          secondStarts.progress,
          deadline,
        );
        const secondCompleted = await waitForTurboOutput(cdp, browser, exactUrl, deadline, {
          progressStartIndex: secondStarts.progress,
          outputStartIndex: secondStarts.output,
        });
        assertValidatedTurboOutputBeforeSave(
          secondCompleted.output,
          secondCompleted.started?.run_id,
        );
        const secondSaveReady = await waitForSaveReady(cdp, browser, deadline);
        const secondDownloadEventStart = cdp.events.length;
        const secondSaveClick = await dispatchCdpMouseClick(
          cdp,
          secondSaveReady.uiContract,
          "save",
        );
        const secondDownloadedPng = await waitForCompletedPngDownload(
          cdp,
          browser,
          downloadDir,
          secondDownloadEventStart,
          deadline,
        );
        const secondCanvasAfter = await captureCanvasScreenshot(
          cdp,
          join(outputDir, "burn-image-rendered-turbo-1024-request-2-after.png"),
        );
        const secondScreenshot = await captureScreenshot(
          cdp,
          join(outputDir, "burn-image-rendered-turbo-1024-request-2.png"),
        );
        const secondRequestEndEpochMs = Date.now();
        finalSnapshot = await pageSnapshot(cdp);
        throwBrowserFailure(cdp, browser, finalSnapshot, "second post-capture model qualification");
        const secondEnds = {
          runtime: finalSnapshot.runtime_events?.length ?? 0,
          progress: finalSnapshot.progress_events?.length ?? 0,
          output: finalSnapshot.output_events?.length ?? 0,
          ui: finalSnapshot.ui_events?.length ?? 0,
          surface: finalSnapshot.surface_texture_gate_windows?.length ?? 0,
          surface_acquisitions: finalSnapshot.surface_texture_acquisition_count ?? 0,
          surface_acquisition_failures:
            finalSnapshot.surface_texture_acquisition_failure_count ?? 0,
          surface_violations:
            finalSnapshot.surface_texture_gate_violation_calls?.length ?? 0,
          surface_violation_overflow:
            finalSnapshot.surface_texture_gate_violation_calls_overflow ?? 0,
          cdp: cdp.events.length,
        };
        const secondRuntimeEvents = finalSnapshot.runtime_events.slice(
          secondStarts.runtime,
          secondEnds.runtime,
        );
        const secondProgressEvents = finalSnapshot.progress_events.slice(
          secondStarts.progress,
          secondEnds.progress,
        );
        const secondOutputEvents = finalSnapshot.output_events.slice(
          secondStarts.output,
          secondEnds.output,
        );
        const secondUiEvents = finalSnapshot.ui_events.slice(secondStarts.ui, secondEnds.ui);
        const secondSurfaceTextureGateWindows =
          finalSnapshot.surface_texture_gate_windows.slice(
            secondStarts.surface,
            secondEnds.surface,
          );
        const secondSurfaceTextureGateViolationCalls =
          finalSnapshot.surface_texture_gate_violation_calls.slice(
            secondStarts.surface_violations,
            secondEnds.surface_violations,
          );
        const secondPackedF16LifecycleEvent = secondRuntimeEvents.find(
          (event) => event?.event === "packed_f16_denoiser_lifecycle",
        );
        const secondPackedF16Lifecycle = secondPackedF16LifecycleEvent?.lifecycle ?? null;
        const secondPackedF16DmdVaeHandoff = secondRuntimeEvents.find(
          (event) => event?.event === "packed_f16_dmd_vae_handoff",
        ) ?? null;
        const secondPackedF16PreDmdInputDiagnostics = secondRuntimeEvents.find(
          (event) => event?.event === "packed_f16_pre_dmd_input_diagnostics",
        ) ?? null;
        const secondPackedF16QwenHostEmbedding = secondRuntimeEvents.find(
          (event) => event?.event === "packed_f16_qwen_host_embedding",
        ) ?? null;
        const secondPackedF16QwenPreHandoffDiagnostics = secondRuntimeEvents.find(
          (event) => event?.event === "packed_f16_qwen_pre_handoff_diagnostics",
        ) ?? null;
        const secondPackedF16QwenBlock0ExecutionDiagnostics = secondRuntimeEvents.find(
          (event) => event?.event === "packed_f16_qwen_block0_execution_diagnostics",
        ) ?? null;
        const secondPackedF16QwenBlock0PostSyncDiagnostic = secondRuntimeEvents.find(
          (event) => event?.event === "packed_f16_qwen_block0_post_sync_diagnostic",
        ) ?? null;
        const secondPackedF16QwenPostHandoffDiagnostics = secondRuntimeEvents.find(
          (event) => event?.event === "packed_f16_qwen_post_handoff_diagnostics",
        ) ?? null;
        const secondRequestSnapshot = {
          ...finalSnapshot,
          runtime_events: secondRuntimeEvents,
          progress_events: secondProgressEvents,
          output_events: secondOutputEvents,
          ui_events: secondUiEvents,
        };
        secondRequestDraft = {
          request_ordinal: 2,
          qwen_block0_execution_mode: requestedQwenBlock0ExecutionMode,
          page_identity: {
            engine_session_id: finalSnapshot.engine_session_id,
            url: finalSnapshot.url,
            time_origin_epoch_ms: finalSnapshot.time_origin_epoch_ms,
          },
          request_epoch_window: {
            start_epoch_ms: secondRequestStartEpochMs,
            end_epoch_ms: secondRequestEndEpochMs,
          },
          event_boundaries: {
            runtime_start_index: secondStarts.runtime,
            runtime_end_index: secondEnds.runtime,
            progress_start_index: secondStarts.progress,
            progress_end_index: secondEnds.progress,
            output_start_index: secondStarts.output,
            output_end_index: secondEnds.output,
            ui_start_index: secondStarts.ui,
            ui_end_index: secondEnds.ui,
            surface_start_index: secondStarts.surface,
            surface_end_index: secondEnds.surface,
            cdp_start_index: secondStarts.cdp,
            cdp_end_index: secondEnds.cdp,
          },
          cdp_event_count: secondEnds.cdp - secondStarts.cdp,
          runtime_events: secondRuntimeEvents,
          progress_events: secondProgressEvents,
          output_events: secondOutputEvents,
          ui_events: secondUiEvents,
          surface_texture_gate_windows: secondSurfaceTextureGateWindows,
          surface_texture_gate_windows_overflow:
            finalSnapshot.surface_texture_gate_windows_overflow,
          surface_texture_gate_violation_calls: secondSurfaceTextureGateViolationCalls,
          surface_texture_gate_violation_calls_overflow_start:
            secondStarts.surface_violation_overflow,
          surface_texture_gate_violation_calls_overflow_end:
            secondEnds.surface_violation_overflow,
          surface_texture_gate_overlap_count:
            finalSnapshot.surface_texture_gate_overlap_count,
          surface_texture_acquisition_count_start: secondStarts.surface_acquisitions,
          surface_texture_acquisition_count_end: secondEnds.surface_acquisitions,
          surface_texture_acquisition_failure_count_start:
            secondStarts.surface_acquisition_failures,
          surface_texture_acquisition_failure_count_end:
            secondEnds.surface_acquisition_failures,
          active_surface_gate_after_request: finalSnapshot.active_surface_gate,
          surface_inference_state_after_request: finalSnapshot.surface_inference_state,
          artifact_traffic: secondRuntimeEvents.find(
            (event) => event?.event === "artifact_traffic",
          )?.traffic ?? null,
          packed_f16_denoiser_lifecycle: secondPackedF16Lifecycle,
          packed_f16_dmd_vae_handoff: secondPackedF16DmdVaeHandoff,
          packed_f16_qwen_host_embedding: secondPackedF16QwenHostEmbedding,
          packed_f16_qwen_block0_execution_diagnostics:
            secondPackedF16QwenBlock0ExecutionDiagnostics,
          packed_f16_qwen_block0_post_sync_diagnostic:
            secondPackedF16QwenBlock0PostSyncDiagnostic,
          packed_f16_qwen_pre_handoff_diagnostics:
            secondPackedF16QwenPreHandoffDiagnostics,
          packed_f16_qwen_post_handoff_diagnostics:
            secondPackedF16QwenPostHandoffDiagnostics,
          packed_f16_pre_dmd_input_diagnostics: secondPackedF16PreDmdInputDiagnostics,
          cdp_network_traffic: summarizeTurboRequestCdpNetwork(
            cdp.events,
            secondRequestSnapshot,
            modelBaseUrls,
            modelTransport.transportTelemetryByPath,
          ),
          cdp_dmd_network_traffic: summarizeTurboDmdCdpNetwork(
            cdp.events,
            secondRequestSnapshot,
            modelBaseUrls,
            modelTransport.transportTelemetryByPath,
          ),
          dmd_runtime_io_attestation: {
            policy: TURBO_DMD_RUNTIME_ZERO_IO_POLICY,
            run_id: secondRunStarted.started.run_id,
          completed_dmd_steps: secondProgressEvents.filter(
              (event) => event?.event === "step" && event?.stage === "dmd",
            ).length,
            traffic: secondPackedF16Lifecycle?.dmd_artifact_traffic ?? null,
            lifecycle_event_at_ms: secondPackedF16LifecycleEvent?.at_ms ?? null,
            runtime_source_sha256: testedPackageIdentity.sources.browser_runtime.sha256,
          },
          modular_artifact_transport: modelTransport,
          served_transport: transport,
          tested_package_identity: testedPackageIdentity,
          ui_contract: secondSaveReady.uiContract,
          output_ready: secondCompleted.output,
          interaction: {
            mechanism: "cdp-keyboard-and-mouse",
            prompt_reused_from_same_engine: true,
            prompt_value: MODEL_PROMPT,
            seed_typed_via_cdp: true,
            seed_value: "1",
            seed_event: secondSeedReady.changed,
            seed_input: secondSeedInput,
            run_readiness: secondRunReady.evidence,
            run_clicked_via_cdp: true,
            run_input: secondRunClick,
            run_started_event: secondRunStarted.started,
            save_clicked_via_cdp: true,
            save_input: secondSaveClick,
          },
          downloaded_png: secondDownloadedPng,
          canvas_before_png: {
            path: firstCanvasAfter.path,
            bytes: firstCanvasAfter.bytes,
            sha256: firstCanvasAfter.sha256,
          },
          canvas_after_png: {
            path: secondCanvasAfter.path,
            bytes: secondCanvasAfter.bytes,
            sha256: secondCanvasAfter.sha256,
          },
          canvas_png_changed_bytes: changedByteCount(
            firstCanvasAfter.data,
            secondCanvasAfter.data,
          ),
          model_screenshot_path: secondScreenshot.path,
          model_screenshot_bytes: secondScreenshot.bytes,
          model_screenshot_sha256: secondScreenshot.sha256,
        };
        screenshot = secondScreenshot;
        canvasAfter = secondCanvasAfter;
      } else {
        finalSnapshot = firstEndSnapshot;
        screenshot = firstScreenshot;
        canvasAfter = firstCanvasAfter;
      }

      gpuAttestation = await gpuMonitor.stop();
      firstRequestDraft.native_gpu_attestation ??= nativeGpuRequestWindow(
        gpuAttestation,
        firstRequestDraft.request_epoch_window.start_epoch_ms,
        firstRequestDraft.request_epoch_window.end_epoch_ms,
      );
      if (secondRequestDraft) {
        secondRequestDraft.native_gpu_attestation = nativeGpuRequestWindow(
          gpuAttestation,
          secondRequestDraft.request_epoch_window.start_epoch_ms,
          secondRequestDraft.request_epoch_window.end_epoch_ms,
        );
      }

      runtimeWebGpuCalls = finalSnapshot.webgpu_calls ?? [];
      runtimeWebGpuDroppedCalls = finalSnapshot.webgpu_dropped_calls ?? 0;
      runtimeWebGpuAdapterAttestation = attestRenderedSurfaceRuntimeAdapter(
        bevyBackendReady,
        runtimeWebGpuCalls,
        finalSnapshot.engine_session_id,
        runtimeWebGpuDroppedCalls,
        requiredDeviceFeatures,
      );

      modelEvidence = {
        ...firstRequestDraft,
        qwen_block0_execution_mode: requestedQwenBlock0ExecutionMode,
        fixed_ascii_prompt: MODEL_PROMPT,
        engine_session_id: finalSnapshot.engine_session_id,
        bevy_backend_ready: bevyBackendReady,
        runtime_webgpu_calls: runtimeWebGpuCalls,
        runtime_webgpu_dropped_calls: runtimeWebGpuDroppedCalls,
        runtime_webgpu_adapter_attestation: runtimeWebGpuAdapterAttestation,
        runtime_events: [...initialRuntimeEvents, ...firstRequestDraft.runtime_events],
        artifact_progress_events: firstEndSnapshot.artifact_progress_events,
        packed_f16_denoiser_preload_traffic: packedF16PreloadTraffic,
        cdp_preload_network_traffic: cdpPreloadNetworkTraffic,
      };
      if (multiRequestModelMode) {
        multiRequestEvidence = {
          policy: TURBO_MULTI_REQUEST_POLICY,
          request_count: 2,
          engine_session_id: finalSnapshot.engine_session_id,
          page_url: finalSnapshot.url,
          time_origin_epoch_ms: finalSnapshot.time_origin_epoch_ms,
          fixed_ascii_prompt: MODEL_PROMPT,
          qwen_block0_execution_mode: requestedQwenBlock0ExecutionMode,
          bevy_backend_ready: bevyBackendReady,
          runtime_webgpu_calls: runtimeWebGpuCalls,
          runtime_webgpu_dropped_calls: runtimeWebGpuDroppedCalls,
          runtime_webgpu_adapter_attestation: runtimeWebGpuAdapterAttestation,
          initial_runtime_events: initialRuntimeEvents,
          initial_packed_f16_denoiser_preload_traffic: packedF16PreloadTraffic,
          cdp_preload_network_traffic: cdpPreloadNetworkTraffic,
          request_scoped_denoiser_policy: {
            policy: TURBO_DENOISER_STORAGE_POLICY,
            expected_stages: TURBO_PACKED_F16_CACHED_STAGES,
            expected_objects: TURBO_PACKED_F16_CACHED_OBJECTS,
            expected_tensors: TURBO_PACKED_F16_CACHED_TENSORS,
            expected_bytes: TURBO_PACKED_F16_RESOURCE_PLAN.retained_packed_f16_denoiser_bytes,
            initial_preload_stages: packedF16PreloadEvents.at(-1)?.cached_stages ?? null,
            initial_preload_objects: packedF16PreloadEvents.at(-1)?.cached_objects ?? null,
            initial_preload_tensors: packedF16PreloadEvents.at(-1)?.cached_tensors ?? null,
            initial_preload_bytes: packedF16PreloadEvents.at(-1)?.cached_bytes ?? null,
            request_local_preload_events: [firstRequestDraft, secondRequestDraft]
              .flatMap((request) => request.runtime_events)
              .filter((event) => event?.event === "packed_f16_denoiser_preload").length,
            successful_requests_with_post_dmd_eviction: 2,
            preload_attempt_counts: [firstRequestDraft, secondRequestDraft].map(
              (request) => request?.packed_f16_denoiser_lifecycle?.preload_attempt_count ?? null,
            ),
            raw_packed_cache_empty_before_vae:
              [firstRequestDraft, secondRequestDraft].every(
                (request) =>
                  request?.packed_f16_denoiser_lifecycle?.cache_ready === true &&
                  request?.packed_f16_denoiser_lifecycle?.matches_plan === true &&
                  request?.packed_f16_dmd_vae_handoff?.report?.packed_cache_after_cleanup
                    ?.state === "empty" &&
                  request?.packed_f16_dmd_vae_handoff?.report?.packed_cache_after_cleanup
                    ?.cached_bytes === 0,
              ),
            repeat_rehydration_cache_only:
              secondRequestDraft.runtime_events.find(
                (event) => event?.event === "packed_f16_denoiser_preload",
              )?.request_scoped_rehydration === true,
          },
          tested_package_identity: testedPackageIdentity,
          native_gpu_attestation: gpuAttestation,
          requests: [firstRequestDraft, secondRequestDraft],
        };
        const multiFailures = validateTurbo1024MultiRequestEvidence(multiRequestEvidence);
        if (multiFailures.length > 0) {
          throw new Error(
            `ordinary Turbo 1024 multi-request evidence failed:\n${multiFailures.join("\n")}`,
          );
        }
      } else {
        // Preserve the original one-request qualification and evidence contract unchanged.
        modelEvidence.native_gpu_attestation = gpuAttestation;
        const modelFailures = validateTurbo1024ModelEvidence(modelEvidence);
        if (modelFailures.length > 0) {
          throw new Error(`ordinary Turbo 1024 evidence failed:\n${modelFailures.join("\n")}`);
        }
      }
    } else {
      screenshot = await captureScreenshot(
        cdp,
        join(outputDir, "burn-image-rendered-surface.png"),
      );
      runtimeWebGpuCalls = finalSnapshot.webgpu_calls ?? [];
      runtimeWebGpuDroppedCalls = finalSnapshot.webgpu_dropped_calls ?? 0;
      runtimeWebGpuAdapterAttestation = attestRenderedSurfaceRuntimeAdapter(
        bevyBackendReady,
        runtimeWebGpuCalls,
        finalSnapshot.engine_session_id,
        runtimeWebGpuDroppedCalls,
        requiredDeviceFeatures,
      );
    }
    evidence = {
      expected_url: exactUrl,
      secure_origin_policy: "loopback HTTP is a potentially trustworthy secure context",
      committed_source: sourceEvidence,
      served_transport: transport,
      tested_package_identity: testedPackageIdentity,
      browser: browserVersion,
      ...renderedChromeLaunchEvidence({
        executable: chromeExecutable,
        arguments: chromeArguments_,
        profile,
        sharedMemory: chromeSharedMemory,
      }),
      launch_readiness_webgpu_probe: launchReadinessWebGpuProbe,
      runtime_webgpu_calls: runtimeWebGpuCalls,
      runtime_webgpu_adapter_attestation: runtimeWebGpuAdapterAttestation,
      required_device_features: requiredDeviceFeatures,
      bevy_backend_ready: bevyBackendReady,
      bevy_backend_events: finalSnapshot.backend_events,
      page_snapshot: finalSnapshot,
      stable_surface_ms: readySurface.stable_ms,
      page_errors: [...cdp.pageErrors],
      gpu_errors: [...cdp.gpuErrors],
      network_failures: [...cdp.networkFailures],
      ignored_benign_network_failures: [...cdp.ignoredBenignNetworkFailures],
      chrome_stderr: browser.child.capturedStderr,
      screenshot_path: screenshot.path,
      screenshot_bytes: screenshot.bytes,
      screenshot_sha256: screenshot.sha256,
      turbo_1024_model: modelEvidence ?? null,
      turbo_1024_multi_request: multiRequestEvidence ?? null,
    };
    const failures = validateRenderedSurfaceEvidence(evidence);
    if (failures.length > 0) {
      throw new Error(`rendered-surface evidence failed:\n${failures.join("\n")}`);
    }
    outcome = {
      schema_version: 1,
      ...renderedSurfaceReportIdentity({
        modelMode,
        multiRequestModelMode,
        qwenBlock0ExecutionMode: requestedQwenBlock0ExecutionMode,
        ok: true,
      }),
      ok: true,
      evidence,
      elapsed_ms: Date.now() - startedAt,
    };
  } catch (error) {
    failure = error instanceof Error ? error : new Error(String(error));
    let failureSnapshot = null;
    if (cdp) {
      try {
        failureSnapshot = await pageSnapshot(cdp);
      } catch {
        // Preserve the primary failure when a crashed page cannot be inspected.
      }
    }
    if (failureSnapshot) {
      runtimeWebGpuCalls = failureSnapshot.webgpu_calls ?? [];
      runtimeWebGpuDroppedCalls = failureSnapshot.webgpu_dropped_calls ?? 0;
      bevyBackendReady ??= [...(failureSnapshot.backend_events ?? [])]
        .reverse()
        .find((event) => event?.event === "ready");
      runtimeWebGpuAdapterAttestation = attestRenderedSurfaceRuntimeAdapter(
        bevyBackendReady,
        runtimeWebGpuCalls,
        failureSnapshot.engine_session_id,
        runtimeWebGpuDroppedCalls,
        requiredDeviceFeatures,
      );
    }
    if (cdp && !screenshot) {
      try {
        screenshot = await captureScreenshot(
          cdp,
          join(
            outputDir,
            modelMode
              ? "burn-image-rendered-turbo-1024-failure.png"
              : "burn-image-rendered-surface-failure.png",
          ),
        );
      } catch {
        // The report retains the primary failure when a crashed page cannot be captured.
      }
    }
    outcome = {
      schema_version: 1,
      ...renderedSurfaceReportIdentity({
        modelMode,
        multiRequestModelMode,
        qwenBlock0ExecutionMode: requestedQwenBlock0ExecutionMode,
        ok: false,
      }),
      ok: false,
      source_evidence: sourceEvidence,
      tested_package_identity: testedPackageIdentity ?? null,
      partial_evidence:
        evidence ??
        (failureSnapshot
          ? {
              launch_readiness_webgpu_probe: launchReadinessWebGpuProbe ?? null,
              runtime_webgpu_calls: runtimeWebGpuCalls ?? [],
              runtime_webgpu_adapter_attestation:
                runtimeWebGpuAdapterAttestation ?? null,
              bevy_backend_ready: bevyBackendReady ?? null,
              page_snapshot: failureSnapshot,
            }
          : null),
      partial_model_evidence:
        modelEvidence ??
        (modelMode && failureSnapshot
          ? {
              qwen_block0_execution_mode: requestedQwenBlock0ExecutionMode,
              runtime_events: failureSnapshot.runtime_events,
              engine_session_id: failureSnapshot.engine_session_id,
              bevy_backend_ready: bevyBackendReady ?? null,
              runtime_webgpu_calls: runtimeWebGpuCalls ?? [],
              runtime_webgpu_dropped_calls: runtimeWebGpuDroppedCalls ?? null,
              runtime_webgpu_adapter_attestation:
                runtimeWebGpuAdapterAttestation ?? null,
              progress_events: failureSnapshot.progress_events,
              artifact_progress_events: failureSnapshot.artifact_progress_events,
              surface_texture_acquisition_count:
                failureSnapshot.surface_texture_acquisition_count ?? null,
              surface_texture_acquisition_failure_count:
                failureSnapshot.surface_texture_acquisition_failure_count ?? null,
              latest_successful_surface_texture_acquisition:
                failureSnapshot.latest_successful_surface_texture_acquisition ?? null,
              surface_texture_gate_windows:
                failureSnapshot.surface_texture_gate_windows ?? [],
              surface_texture_gate_windows_overflow:
                failureSnapshot.surface_texture_gate_windows_overflow ?? null,
              surface_texture_gate_violation_calls:
                failureSnapshot.surface_texture_gate_violation_calls ?? [],
              surface_texture_gate_violation_calls_overflow:
                failureSnapshot.surface_texture_gate_violation_calls_overflow ?? null,
              active_surface_gate: failureSnapshot.active_surface_gate ?? null,
              surface_inference_state_after_request:
                failureSnapshot.surface_inference_state ?? null,
              partial_surface_gate_events: (failureSnapshot.runtime_events ?? []).filter(
                (event) =>
                  event?.event === "surface_inference_suspended" ||
                  event?.event === "surface_inference_resumed",
              ),
              packed_f16_denoiser_preload_traffic:
                [...(failureSnapshot.runtime_events ?? [])]
                  .reverse()
                  .find((event) => event?.event === "packed_f16_denoiser_preload")?.traffic ?? null,
              packed_f16_denoiser_lifecycle:
                [...(failureSnapshot.runtime_events ?? [])]
                  .reverse()
                  .find((event) => event?.event === "packed_f16_denoiser_lifecycle")?.lifecycle ??
                null,
              packed_f16_dmd_vae_handoff:
                [...(failureSnapshot.runtime_events ?? [])]
                  .reverse()
                  .find((event) => event?.event === "packed_f16_dmd_vae_handoff") ?? null,
              packed_f16_qwen_host_embedding:
                [...(failureSnapshot.runtime_events ?? [])]
                  .reverse()
                  .find((event) => event?.event === "packed_f16_qwen_host_embedding") ?? null,
              packed_f16_qwen_block0_execution_diagnostics:
                [...(failureSnapshot.runtime_events ?? [])]
                  .reverse()
                  .find(
                    (event) =>
                      event?.event === "packed_f16_qwen_block0_execution_diagnostics",
                  ) ?? null,
              packed_f16_qwen_block0_post_sync_diagnostic:
                [...(failureSnapshot.runtime_events ?? [])]
                  .reverse()
                  .find(
                    (event) =>
                      event?.event === "packed_f16_qwen_block0_post_sync_diagnostic",
                  ) ?? null,
              packed_f16_qwen_pre_handoff_diagnostics:
                [...(failureSnapshot.runtime_events ?? [])]
                  .reverse()
                  .find(
                    (event) => event?.event === "packed_f16_qwen_pre_handoff_diagnostics",
                  ) ?? null,
              packed_f16_qwen_post_handoff_diagnostics:
                [...(failureSnapshot.runtime_events ?? [])]
                  .reverse()
                  .find(
                    (event) => event?.event === "packed_f16_qwen_post_handoff_diagnostics",
                  ) ?? null,
              packed_f16_pre_dmd_input_diagnostics:
                [...(failureSnapshot.runtime_events ?? [])]
                  .reverse()
                  .find(
                    (event) => event?.event === "packed_f16_pre_dmd_input_diagnostics",
                  ) ?? null,
              cdp_preload_network_traffic: (() => {
                try {
                  return summarizeTurboPreloadCdpNetwork(
                    cdp?.events ?? [],
                    failureSnapshot,
                    modelBaseUrls ?? [],
                    modelTransport?.transportTelemetryByPath,
                  );
                } catch {
                  return null;
                }
              })(),
              artifact_traffic:
                [...(failureSnapshot.runtime_events ?? [])]
                  .reverse()
                  .find((event) => event?.event === "artifact_traffic")?.traffic ?? null,
              cdp_network_traffic: (() => {
                try {
                  return summarizeTurboRequestCdpNetwork(
                    cdp?.events ?? [],
                    failureSnapshot,
                    modelBaseUrls ?? [],
                    modelTransport?.transportTelemetryByPath,
                  );
                } catch {
                  return null;
                }
              })(),
              ui_events: failureSnapshot.ui_events,
              output_events: failureSnapshot.output_events,
              interaction: interactionEvidence ?? null,
              downloaded_png: downloadedPng ?? null,
            }
          : null),
      partial_multi_request_evidence:
        multiRequestEvidence ??
        (multiRequestModelMode
          ? {
              policy: TURBO_MULTI_REQUEST_POLICY,
              first_request: firstRequestDraft ?? null,
              second_request: secondRequestDraft ?? null,
              engine_session_id: failureSnapshot?.engine_session_id ?? null,
              bevy_backend_ready: bevyBackendReady ?? null,
              runtime_webgpu_calls: runtimeWebGpuCalls ?? [],
              runtime_webgpu_dropped_calls: runtimeWebGpuDroppedCalls ?? null,
              runtime_webgpu_adapter_attestation:
                runtimeWebGpuAdapterAttestation ?? null,
              page_snapshot: failureSnapshot,
            }
          : null),
      native_gpu_attestation: gpuAttestation ?? null,
      page_errors: cdp?.pageErrors ?? [],
      gpu_errors: cdp?.gpuErrors ?? [],
      network_failures: cdp?.networkFailures ?? [],
      ignored_benign_network_failures: cdp?.ignoredBenignNetworkFailures ?? [],
      screenshot: screenshot ?? null,
      error: failure.stack ?? failure.message,
      elapsed_ms: Date.now() - startedAt,
    };
  } finally {
    if (gpuMonitor && !gpuAttestation) {
      try {
        gpuAttestation = await gpuMonitor.stop();
      } catch (error) {
        if (!failure) failure = error instanceof Error ? error : new Error(String(error));
      }
    }
    cdp?.close();
    const chromeCleanup = await stopChrome(browser);
    await closeServer(server);
    const profileCanBeRemoved = Boolean(
      profile && (!browser || chromeCleanup?.process_group_exited),
    );
    if (profileCanBeRemoved) {
      await rm(profile, { recursive: true, force: true });
    }
    outcome.chrome_cleanup = chromeCleanup;
    outcome.chrome_profile_removed = profileCanBeRemoved;
    const chromeLaunchEvidence = renderedChromeLaunchEvidence({
      executable: chromeExecutable,
      arguments: chromeArguments_,
      profile,
      sharedMemory: chromeSharedMemory,
    });
    Object.assign(outcome, chromeLaunchEvidence);
    if (outcome.evidence) Object.assign(outcome.evidence, chromeLaunchEvidence);
    if (outcome.partial_evidence) {
      Object.assign(outcome.partial_evidence, chromeLaunchEvidence);
    }
    if (modelMode) outcome.native_gpu_monitor_cleanup = gpuAttestation ?? null;
    outcome.tested_package_identity = testedPackageIdentity ?? null;
    runtimeWebGpuCalls ??= [];
    runtimeWebGpuDroppedCalls ??= 0;
    runtimeWebGpuAdapterAttestation ??= attestRenderedSurfaceRuntimeAdapter(
      bevyBackendReady,
      runtimeWebGpuCalls,
      null,
      runtimeWebGpuDroppedCalls,
      requiredDeviceFeatures,
    );
    outcome.launch_readiness_webgpu_probe = launchReadinessWebGpuProbe ?? null;
    outcome.runtime_webgpu_calls = runtimeWebGpuCalls;
    outcome.runtime_webgpu_adapter_attestation = runtimeWebGpuAdapterAttestation;

    const finalChromeStderr = browser?.child?.capturedStderr ?? "";
    // Keep the bounded Chrome diagnostics on failures as well as successes. GPU-process resets
    // can surface to WebGPU only as a later DeviceLost/map error; stderr is the only remaining
    // place Chromium may preserve the initiating Dawn/ANGLE/Vulkan diagnostic after teardown.
    outcome.chrome_stderr = finalChromeStderr;
    if (outcome.evidence) outcome.evidence.chrome_stderr = finalChromeStderr;
    if (outcome.ok && gpuTerminalDiagnostic(finalChromeStderr)) {
      failure = new Error("Chrome emitted a terminal WebGPU/device-loss diagnostic during shutdown");
    }
    if (
      outcome.ok &&
      chromeCleanup &&
      (!chromeCleanup.process_group_exited || chromeCleanup.errors.length > 0)
    ) {
      failure = new Error(`Chrome cleanup failed: ${JSON.stringify(chromeCleanup)}`);
    }
    if (failure && outcome.ok) {
      outcome.ok = false;
      Object.assign(
        outcome,
        renderedSurfaceReportIdentity({
          modelMode,
          multiRequestModelMode,
          qwenBlock0ExecutionMode: requestedQwenBlock0ExecutionMode,
          ok: false,
        }),
      );
      outcome.error = failure.stack ?? failure.message;
    }
    outcome.elapsed_ms = Date.now() - startedAt;
    await writeFile(reportPath, `${JSON.stringify(outcome, null, 2)}\n`);
    console.log(JSON.stringify(outcome, null, 2));
    console.log(`burn_image rendered-surface report: ${reportPath}`);
  }
  if (failure) throw failure;
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
});
