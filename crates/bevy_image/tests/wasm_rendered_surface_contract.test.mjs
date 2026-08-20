import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  ARTIFACT_TRAFFIC_FIELDS,
  aggregateChromeGpuInterval,
  BROWSER_WEBGPU_ENABLED_FEATURE_BASELINE,
  BOOGU_WEB_REQUIRED_DEVICE_FEATURES,
  CHROME_SHARED_MEMORY_DEV_SHM_POLICY,
  CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES,
  CHROME_SHARED_MEMORY_PLATFORM_DEFAULT_POLICY,
  CHROME_SHARED_MEMORY_TMP_FALLBACK_POLICY,
  GENERIC_WEB_REQUIRED_DEVICE_FEATURES,
  GPU_INTERVAL_AGGREGATION_POLICY,
  gpuTerminalDiagnostic,
  captureCdpNetworkRequestContext,
  cdpNetworkLoadingFailedDiagnostic,
  isProvenBenignFaviconFailure,
  LOW_VRAM_DEVICE_CAP_BYTES,
  LOW_VRAM_BACKEND,
  RENDERED_LAUNCH_READINESS_PROBE_POLICY,
  SURFACE_INFERENCE_POLICY,
  renderedChromeLaunchEvidence,
  renderedSurfaceReportIdentity,
  selectChromeSharedMemoryPolicy,
  attestRenderedSurfaceRuntimeAdapter,
  summarizeRenderedSurfaceRuntimeWebGpuCalls,
  summarizeTurboDmdCdpNetwork,
  summarizeTurboPreloadCdpNetwork,
  summarizeTurboRequestCdpNetwork,
  TURBO_CDP_PRELOAD_NETWORK_POLICY,
  TURBO_CDP_DMD_NETWORK_POLICY,
  TURBO_CDP_REQUEST_NETWORK_POLICY,
  TURBO_DENOISER_STORAGE_POLICY,
  TURBO_DENOISER_REHYDRATION_TRAFFIC,
  TURBO_PACKED_F16_PRELOAD_MESSAGE,
  TURBO_DENOISER_PRELOAD_TRAFFIC,
  TURBO_DMD_RUNTIME_ZERO_IO_POLICY,
  TURBO_DMD_ZERO_IO,
  TURBO_GENERATE_REQUEST_TRAFFIC,
  TURBO_PACKED_F16_CACHED_OBJECTS,
  TURBO_PACKED_F16_CACHED_STAGES,
  TURBO_PACKED_F16_CACHED_TENSORS,
  TURBO_PACKED_F16_PRE_DMD_INPUT_SCOPE,
  TURBO_PACKED_F16_DMD_VAE_HANDOFF_POLICY,
  TURBO_PACKED_F16_NEXT_REQUEST_REHYDRATION_POLICY,
  TURBO_PACKED_F16_QWEN_HANDOFF_POLICY,
  TURBO_PACKED_F16_QWEN_BLOCK0_BOUNDARIES,
  TURBO_PACKED_F16_QWEN_BLOCK0_EXECUTION_SCOPE,
  TURBO_PACKED_F16_QWEN_BLOCK0_POST_SYNC_SCOPE,
  TURBO_PACKED_F16_QWEN_HOST_EMBEDDING_F16_BYTES,
  TURBO_PACKED_F16_QWEN_HOST_EMBEDDING_F32_BYTES,
  TURBO_PACKED_F16_QWEN_HOST_EMBEDDING_PLAN,
  TURBO_PACKED_F16_QWEN_HOST_EMBEDDING_POLICY,
  TURBO_PACKED_F16_QWEN_HOST_EMBEDDING_SHA256,
  TURBO_PACKED_F16_QWEN_POST_HANDOFF_SCOPE,
  TURBO_PACKED_F16_QWEN_PRE_HANDOFF_SCOPE,
  TURBO_PACKED_F16_QWEN_STAGE_OUTPUT_COUNT,
  TURBO_PACKED_F16_REQUEST_LIFECYCLE,
  TURBO_PACKED_F16_RESOURCE_PLAN,
  TURBO_MODEL_ID,
  TURBO_MULTI_REQUEST_POLICY,
  TURBO_PRODUCTION_CONTENT_DIGEST,
  TURBO_RENDERED_Q4_RESIDENT_MULTI_REQUEST_TEST,
  TURBO_REPEAT_GENERATE_REQUEST_TRAFFIC,
  TURBO_QWEN_TEXT_LAYER_ALLOCATION_POLICY,
  TURBO_QWEN_TEXT_BLOCK_LOAD_SYNCHRONIZATION_POLICY,
  TURBO_QWEN_TEXT_LAYER_SUBMISSION_POLICY,
  TURBO_QWEN_BLOCK0_ORDINARY_MODE,
  TURBO_QWEN_BLOCK0_SERIALIZED_DIAGNOSTIC_MODE,
  TURBO_RENDERED_ORDINARY_MULTI_REQUEST_TEST,
  TURBO_RENDERED_ORDINARY_SMOKE_TEST,
  TURBO_RENDERED_SERIALIZED_DIAGNOSTIC_TEST,
  TURBO_RENDERED_SERIALIZED_MULTI_REQUEST_DIAGNOSTIC_TEST,
  TURBO_SECOND_REQUEST_RUN_READY_POLICY,
  isCanonicalU64DecimalString,
  outputJobIdMatchesNumericRunId,
  resolveTurboSecondRequestRunReadyUiContract,
  validateBackendReadyEvent,
  validateHardwareNvidiaAdapter,
  validateCdpNetworkFailureEvidence,
  validatePackedF16PreDmdInputDiagnostics,
  validatePackedF16DmdVaeHandoff,
  validatePackedF16QwenHostEmbedding,
  validatePackedF16QwenBlock0ExecutionDiagnostics,
  validatePackedF16QwenBlock0PostSyncDiagnostic,
  validatePackedF16QwenPostHandoffDiagnostics,
  validatePackedF16QwenPreHandoffDiagnostics,
  validateRenderedSurfaceEvidence,
  validateRenderedModelTransportEvidence,
  validateRenderedSurfaceSnapshot,
  validateRequestScopedSurfaceGate,
  validateTestedPackageIdentity,
  validateTurbo1024ModelEvidence,
  validateTurbo1024MultiRequestEvidence,
} from "./wasm_rendered_surface_contract.mjs";
import {
  ARTIFACT_TRANSPORT_HARD_MAX_PART_BYTES,
  ARTIFACT_TRANSPORT_LAYOUT_PATH,
  ARTIFACT_TRANSPORT_TARGET_PART_BYTES,
} from "./artifact_transport_contract.mjs";

const TEST_ENGINE_SESSION_ID = "3c6eeea1-a432-4d50-92cc-f39f867d1941";
const TEST_ADAPTER_NAME = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition";

function expectedBrowserEnabledFeatures(requiredFeatures) {
  return [...new Set([...BROWSER_WEBGPU_ENABLED_FEATURE_BASELINE, ...requiredFeatures])].sort();
}

function admittedSharedMemoryMeasurement({
  availableBytes = CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES * 2,
  writtenBytes = CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES,
} = {}) {
  return {
    path: "/dev/shm",
    exists: true,
    directory: true,
    writable: true,
    statfs: { available_bytes: availableBytes },
    quota_aware_allocation_probe: {
      attempted: true,
      requested_bytes: CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES,
      written_bytes: writtenBytes,
      succeeded: true,
    },
  };
}

test("selects /dev/shm only after global and quota-aware headroom admission", () => {
  assert.deepEqual(
    selectChromeSharedMemoryPolicy({
      platform: "linux",
      devShm: admittedSharedMemoryMeasurement(),
      tempPath: "/tmp",
    }),
    {
      policy: CHROME_SHARED_MEMORY_DEV_SHM_POLICY,
      selected_backing: "dev-shm",
      selected_path: "/dev/shm",
      disable_dev_shm_usage: false,
      minimum_admitted_headroom_bytes: CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES,
      dev_shm_admitted: true,
      dev_shm_rejections: [],
    },
  );

  const overflowSafe = selectChromeSharedMemoryPolicy({
    platform: "linux",
    devShm: admittedSharedMemoryMeasurement({
      availableBytes: "18446744073709551615",
      writtenBytes: String(CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES),
    }),
    tempPath: "/tmp",
  });
  assert.equal(overflowSafe.disable_dev_shm_usage, false);
  assert.equal(overflowSafe.selected_path, "/dev/shm");

  const unsafeNumber = selectChromeSharedMemoryPolicy({
    platform: "linux",
    devShm: admittedSharedMemoryMeasurement({
      availableBytes: Number.MAX_SAFE_INTEGER + 1,
    }),
    tempPath: "/tmp",
  });
  assert.equal(unsafeNumber.disable_dev_shm_usage, true);
  assert.ok(unsafeNumber.dev_shm_rejections.includes("statfs-available-unknown"));
  assert.throws(
    () =>
      selectChromeSharedMemoryPolicy({
        platform: "linux",
        devShm: admittedSharedMemoryMeasurement(),
        minimumHeadroomBytes: Number.MAX_SAFE_INTEGER + 1,
      }),
    /positive safe integer/,
  );
});

test("retains temp-backed Chrome shared memory for missing, tiny, or quota-limited /dev/shm", () => {
  const missing = selectChromeSharedMemoryPolicy({
    platform: "linux",
    devShm: null,
    tempPath: "/tmp",
  });
  assert.equal(missing.policy, CHROME_SHARED_MEMORY_TMP_FALLBACK_POLICY);
  assert.equal(missing.disable_dev_shm_usage, true);
  assert.equal(missing.selected_path, "/tmp");
  assert.ok(missing.dev_shm_rejections.includes("missing"));

  const tiny = selectChromeSharedMemoryPolicy({
    platform: "linux",
    devShm: {
      ...admittedSharedMemoryMeasurement(),
      statfs: { available_bytes: CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES - 1 },
      quota_aware_allocation_probe: {
        attempted: false,
        requested_bytes: CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES,
        written_bytes: 0,
        succeeded: false,
      },
    },
    tempPath: "/tmp",
  });
  assert.equal(tiny.disable_dev_shm_usage, true);
  assert.ok(tiny.dev_shm_rejections.includes("statfs-available-below-minimum"));

  const quotaLimited = selectChromeSharedMemoryPolicy({
    platform: "linux",
    devShm: {
      ...admittedSharedMemoryMeasurement(),
      quota_aware_allocation_probe: {
        attempted: true,
        requested_bytes: CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES,
        written_bytes: 32 * 1024 * 1024,
        succeeded: false,
        errors: [{ code: "EDQUOT" }],
      },
    },
    tempPath: "/tmp",
  });
  assert.equal(quotaLimited.disable_dev_shm_usage, true);
  assert.ok(quotaLimited.dev_shm_rejections.includes("quota-aware-probe-failed"));
});

test("non-Linux launch policy omits the Linux-only temp-backing flag", () => {
  const selected = selectChromeSharedMemoryPolicy({
    platform: "darwin",
    devShm: null,
    tempPath: "/var/folders/example",
  });
  assert.equal(selected.policy, CHROME_SHARED_MEMORY_PLATFORM_DEFAULT_POLICY);
  assert.equal(selected.disable_dev_shm_usage, false);
  assert.equal(selected.dev_shm_admitted, null);
});

test("Chrome launch evidence preserves exact launch state and explicit pre-launch nulls", () => {
  const sharedMemory = {
    policy: CHROME_SHARED_MEMORY_DEV_SHM_POLICY,
    selected_path: "/dev/shm",
  };
  const success = renderedChromeLaunchEvidence({
    executable: "/opt/google/chrome/google-chrome",
    arguments: ["--user-data-dir=/evidence/profile", "http://127.0.0.1:1234/probe"],
    profile: "/evidence/profile",
    sharedMemory,
  });
  assert.deepEqual(success, {
    chrome_executable: "/opt/google/chrome/google-chrome",
    chrome_arguments: [
      "--user-data-dir=/evidence/profile",
      "http://127.0.0.1:1234/probe",
    ],
    chrome_profile: "/evidence/profile",
    chrome_shared_memory: sharedMemory,
  });
  assert.deepEqual(
    renderedChromeLaunchEvidence({}),
    {
      chrome_executable: null,
      chrome_arguments: null,
      chrome_profile: null,
      chrome_shared_memory: null,
    },
  );
});

test("rendered harness conditionally applies shmem fallback and persists launch evidence", async () => {
  const source = await readFile(
    new URL("./wasm_rendered_surface_smoke.mjs", import.meta.url),
    "utf8",
  );
  assert.match(source, /statfs\(path, \{ bigint: true \}\)/);
  assert.match(source, /await probeFile\.sync\(\)/);
  assert.match(source, /per-user-or-group-quota-exhausted/);
  assert.match(
    source,
    /if \(sharedMemoryPolicy\.disable_dev_shm_usage\) \{\s*arguments_\.push\("--disable-dev-shm-usage"\);\s*\}/,
  );
  assert.equal(source.match(/"--disable-dev-shm-usage"/g)?.length, 1);
  assert.match(source, /Object\.assign\(outcome, chromeLaunchEvidence\)/);
  assert.match(source, /if \(outcome\.partial_evidence\)/);
  assert.ok(
    source.indexOf("chromeSharedMemory = await inspectChromeSharedMemory()") <
      source.indexOf("browser = await startChrome"),
  );
});

test("rendered Q4 readiness requires its resident plan instead of the packed-F16 plan", async () => {
  const source = await readFile(
    new URL("./wasm_rendered_surface_smoke.mjs", import.meta.url),
    "utf8",
  );
  assert.match(source, /q4ResidentMode \? "resident_resource_plan" : "packed_f16_resource_plan"/);
  assert.match(
    source,
    /waitForTurboUiReady\(\s*cdp,\s*browser,\s*exactUrl,\s*deadline,\s*q4ResidentMode,/,
  );
});

test("keeps serialized block-0 success diagnostic-only by outer report identity", () => {
  const ordinary = renderedSurfaceReportIdentity({
    modelMode: true,
    multiRequestModelMode: false,
    qwenBlock0ExecutionMode: TURBO_QWEN_BLOCK0_ORDINARY_MODE,
    ok: true,
  });
  assert.deepEqual(ordinary, {
    test: TURBO_RENDERED_ORDINARY_SMOKE_TEST,
    claim:
      "ordinary rendered Bevy UI Turbo 1024 preloaded packed-F16 storage / dense-F32-per-semantic-stage low-VRAM real-model smoke; not numerical parity",
  });

  const diagnostic = renderedSurfaceReportIdentity({
    modelMode: true,
    multiRequestModelMode: false,
    qwenBlock0ExecutionMode: TURBO_QWEN_BLOCK0_SERIALIZED_DIAGNOSTIC_MODE,
    ok: true,
  });
  assert.equal(diagnostic.test, TURBO_RENDERED_SERIALIZED_DIAGNOSTIC_TEST);
  assert.notEqual(diagnostic.test, TURBO_RENDERED_ORDINARY_SMOKE_TEST);
  assert.match(diagnostic.claim, /^diagnostic-only /);
  assert.match(diagnostic.claim, /not model-smoke, release-qualification, output-quality/);

  const ordinaryMulti = renderedSurfaceReportIdentity({
    modelMode: true,
    multiRequestModelMode: true,
    qwenBlock0ExecutionMode: TURBO_QWEN_BLOCK0_ORDINARY_MODE,
    ok: true,
  });
  assert.deepEqual(ordinaryMulti, {
    test: TURBO_RENDERED_ORDINARY_MULTI_REQUEST_TEST,
    claim:
      "same-page same-engine two-request ordinary rendered Bevy UI Turbo 1024 low-VRAM qualification; not numerical parity",
  });
  const residentQ4Multi = renderedSurfaceReportIdentity({
    modelMode: true,
    multiRequestModelMode: true,
    q4ResidentMode: true,
    qwenBlock0ExecutionMode: TURBO_QWEN_BLOCK0_ORDINARY_MODE,
    ok: true,
  });
  assert.equal(residentQ4Multi.test, TURBO_RENDERED_Q4_RESIDENT_MULTI_REQUEST_TEST);
  assert.match(residentQ4Multi.claim, /resident-Q4 warm-session qualification/);
  const diagnosticMulti = renderedSurfaceReportIdentity({
    modelMode: true,
    multiRequestModelMode: true,
    qwenBlock0ExecutionMode: TURBO_QWEN_BLOCK0_SERIALIZED_DIAGNOSTIC_MODE,
    ok: true,
  });
  assert.equal(
    diagnosticMulti.test,
    TURBO_RENDERED_SERIALIZED_MULTI_REQUEST_DIAGNOSTIC_TEST,
  );
  assert.notEqual(diagnosticMulti.test, TURBO_RENDERED_ORDINARY_MULTI_REQUEST_TEST);

  const failedDiagnostic = renderedSurfaceReportIdentity({
    modelMode: true,
    multiRequestModelMode: false,
    qwenBlock0ExecutionMode: TURBO_QWEN_BLOCK0_SERIALIZED_DIAGNOSTIC_MODE,
    ok: false,
  });
  assert.equal(failedDiagnostic.test, TURBO_RENDERED_SERIALIZED_DIAGNOSTIC_TEST);
  assert.match(failedDiagnostic.claim, /^failed serialized-diagnostic /);
  assert.deepEqual(
    renderedSurfaceReportIdentity({
      modelMode: true,
      multiRequestModelMode: false,
      qwenBlock0ExecutionMode: TURBO_QWEN_BLOCK0_ORDINARY_MODE,
      ok: false,
    }),
    {
      test: TURBO_RENDERED_ORDINARY_SMOKE_TEST,
      claim: "failed ordinary-UI model attempt; no model-smoke or numerical parity claim",
    },
  );
  assert.throws(
    () =>
      renderedSurfaceReportIdentity({
        modelMode: true,
        multiRequestModelMode: false,
        qwenBlock0ExecutionMode: "invalid",
        ok: true,
      }),
    /unknown rendered Turbo Qwen block-0 execution mode/,
  );
});

function validBackendReady(adapterName = TEST_ADAPTER_NAME, deviceType = "DiscreteGpu") {
  return {
    at_ms: 40,
    event: "ready",
    adapter_name: adapterName,
    backend: "BrowserWebGpu",
    device_type: deviceType,
    shared_adapter_device_queue: true,
    message: `GPU: ${adapterName} (BrowserWebGpu, ${deviceType}) - shared device ready`,
  };
}

function validRuntimeWebGpuCalls({
  adapterName = TEST_ADAPTER_NAME,
  vendor = "nvidia",
  fallback = false,
  adapterRequestId = 1,
  deviceRequestId = 2,
  requiredFeatures = BOOGU_WEB_REQUIRED_DEVICE_FEATURES,
  enabledFeatures = expectedBrowserEnabledFeatures(requiredFeatures),
} = {}) {
  const device = {
    request_id: deviceRequestId,
    adapter_request_id: adapterRequestId,
    label: null,
    requiredFeatures: [...requiredFeatures],
    requiredLimits: {},
  };
  return [
    {
      event: "request-adapter-start",
      at_ms: 10,
      detail: {
        request_id: adapterRequestId,
        powerPreference: "high-performance",
        forceFallbackAdapter: false,
      },
    },
    {
      event: "request-adapter-resolved",
      at_ms: 20,
      detail: {
        request_id: adapterRequestId,
        available: true,
        info: {
          is_fallback_adapter: fallback,
          vendor,
          architecture: "blackwell",
          device: "0x2bb1",
          description: adapterName,
        },
      },
    },
    { event: "request-device-start", at_ms: 25, detail: { ...device } },
    {
      event: "request-device-resolved",
      at_ms: 30,
      detail: {
        ...device,
        enabledFeatures: [...enabledFeatures],
      },
    },
  ];
}

function validRuntimeEvidence(
  backendReady = validBackendReady(),
  engineSessionId = TEST_ENGINE_SESSION_ID,
) {
  const calls = validRuntimeWebGpuCalls({ adapterName: backendReady.adapter_name });
  return {
    engine_session_id: engineSessionId,
    bevy_backend_ready: backendReady,
    runtime_webgpu_calls: calls,
    runtime_webgpu_dropped_calls: 0,
    runtime_webgpu_adapter_attestation: attestRenderedSurfaceRuntimeAdapter(
      backendReady,
      calls,
      engineSessionId,
      0,
    ),
  };
}

function validTestedPackageIdentity() {
  const entry = (absolutePath, relativePath, marker) => ({
    absolute_path: absolutePath,
    relative_path: relativePath,
    bytes: 4096,
    sha256: marker.repeat(64),
  });
  return {
    policy: "exact-local-package-and-runtime-source-bytes-served-to-browser",
    generated_package: {
      javascript: entry(
        "/tmp/browser-package/bevy_burn_image.js",
        "bevy_burn_image.js",
        "1",
      ),
      webassembly: entry(
        "/tmp/browser-package/bevy_burn_image_bg.wasm",
        "bevy_burn_image_bg.wasm",
        "2",
      ),
      app_icon: entry(
        "/tmp/browser-package/burn-image-icon.png",
        "burn-image-icon.png",
        "8",
      ),
    },
    page_modules: {
      model_selector: entry(
        "/workspace/crates/bevy_image/www/model_selector.mjs",
        "crates/bevy_image/www/model_selector.mjs",
        "6",
      ),
    },
    sources: {
      browser_runtime: entry(
        "/workspace/crates/bevy_image/src/browser_boogu/runtime.rs",
        "crates/bevy_image/src/browser_boogu/runtime.rs",
        "3",
      ),
      rendered_harness: entry(
        "/workspace/crates/bevy_image/tests/wasm_rendered_surface_smoke.mjs",
        "crates/bevy_image/tests/wasm_rendered_surface_smoke.mjs",
        "4",
      ),
      rendered_contract: entry(
        "/workspace/crates/bevy_image/tests/wasm_rendered_surface_contract.mjs",
        "crates/bevy_image/tests/wasm_rendered_surface_contract.mjs",
        "5",
      ),
      artifact_transport_contract: entry(
        "/workspace/crates/bevy_image/tests/artifact_transport_contract.mjs",
        "crates/bevy_image/tests/artifact_transport_contract.mjs",
        "7",
      ),
    },
    validated: true,
  };
}

function validEvidence() {
  const testedPackageIdentity = validTestedPackageIdentity();
  const runtime = validRuntimeEvidence();
  const launchAdapter = {
    vendor: "nvidia",
    architecture: "blackwell",
    device: "0x2bb1",
    description: TEST_ADAPTER_NAME,
    is_fallback_adapter: false,
  };
  return {
    expected_url: "http://127.0.0.1:39001/index.html?rendered-surface-smoke=abc",
    launch_readiness_webgpu_probe: {
      policy: RENDERED_LAUNCH_READINESS_PROBE_POLICY,
      runtime_hardware_proof: false,
      adapter: launchAdapter,
    },
    runtime_webgpu_calls: runtime.runtime_webgpu_calls,
    runtime_webgpu_adapter_attestation: runtime.runtime_webgpu_adapter_attestation,
    bevy_backend_ready: runtime.bevy_backend_ready,
    bevy_backend_events: [],
    page_snapshot: {
      url: "http://127.0.0.1:39001/index.html?rendered-surface-smoke=abc",
      engine_session_id: runtime.engine_session_id,
      webgpu_dropped_calls: runtime.runtime_webgpu_dropped_calls,
      secure_context: true,
      ready_state: "complete",
      device_pixel_ratio: 2,
      canvas: {
        width: 1800,
        height: 1200,
        client_width: 900,
        client_height: 600,
        rect_width: 900,
        rect_height: 600,
      },
    },
    page_errors: [],
    gpu_errors: [],
    network_failures: [],
    ignored_benign_network_failures: [],
    chrome_stderr: "normal Chrome diagnostics",
    screenshot_bytes: 4096,
    screenshot_sha256: "a".repeat(64),
    tested_package_identity: testedPackageIdentity,
    served_transport: {
      generated: {
        "bevy_burn_image.js": {
          bytes: testedPackageIdentity.generated_package.javascript.bytes,
          sha256: testedPackageIdentity.generated_package.javascript.sha256,
        },
        "bevy_burn_image_bg.wasm": {
          bytes: testedPackageIdentity.generated_package.webassembly.bytes,
          sha256: testedPackageIdentity.generated_package.webassembly.sha256,
        },
        "burn-image-icon.png": {
          bytes: testedPackageIdentity.generated_package.app_icon.bytes,
          sha256: testedPackageIdentity.generated_package.app_icon.sha256,
          content_type: "image/png",
        },
      },
      page_modules: {
        "model_selector.mjs": {
          bytes: testedPackageIdentity.page_modules.model_selector.bytes,
          sha256: testedPackageIdentity.page_modules.model_selector.sha256,
          content_type: "text/javascript; charset=utf-8",
        },
      },
    },
  };
}

test("accepts hardware NVIDIA shared-device rendered-surface evidence", () => {
  assert.deepEqual(validateRenderedSurfaceEvidence(validEvidence()), []);
});

test("separates bounded physical transport evidence from logical model artifacts", () => {
  const transport = validModelTransport();
  assert.deepEqual(validateRenderedModelTransportEvidence(transport), []);
  transport.bundles[0].physical_transport.max_part_bytes = 25_000_001;
  assert.ok(
    validateRenderedModelTransportEvidence(transport).some((failure) =>
      failure.includes("physical transport inventory"),
    ),
  );
});

test("keeps the generic rendered shell portable without Boogu timing features", () => {
  const calls = validRuntimeWebGpuCalls({
    requiredFeatures: GENERIC_WEB_REQUIRED_DEVICE_FEATURES,
  });
  const attestation = attestRenderedSurfaceRuntimeAdapter(
    validBackendReady(),
    calls,
    TEST_ENGINE_SESSION_ID,
    0,
    GENERIC_WEB_REQUIRED_DEVICE_FEATURES,
  );
  assert.equal(attestation.validated, true, attestation.validation_failures.join("\n"));
  assert.deepEqual(attestation.requested_features, []);
  assert.deepEqual(
    attestation.enabled_features,
    BROWSER_WEBGPU_ENABLED_FEATURE_BASELINE,
  );
  assert.deepEqual(
    attestation.expected_enabled_features,
    BROWSER_WEBGPU_ENABLED_FEATURE_BASELINE,
  );
});

test("rejects an unrecognized caller-selected device feature contract", () => {
  const calls = validRuntimeWebGpuCalls({
    requiredFeatures: ["shader-f16"],
  });
  const attestation = attestRenderedSurfaceRuntimeAdapter(
    validBackendReady(),
    calls,
    TEST_ENGINE_SESSION_ID,
    0,
    ["shader-f16"],
  );
  assert.equal(attestation.validated, false);
  assert.ok(
    attestation.validation_failures.some((failure) => /feature contract/.test(failure)),
    attestation.validation_failures.join("\n"),
  );
});

test("normalizes the exact rendered-page adapter and device requests", () => {
  const calls = validRuntimeWebGpuCalls();
  const summary = summarizeRenderedSurfaceRuntimeWebGpuCalls(
    calls,
    TEST_ENGINE_SESSION_ID,
    0,
  );
  assert.equal(summary.adapter_request_attempts, 1);
  assert.equal(summary.adapter_successful_attempts, 1);
  assert.equal(summary.device_request_attempts, 1);
  assert.equal(summary.device_successful_attempts, 1);
  assert.equal(summary.is_fallback_adapter, false);
  assert.equal(summary.power_preference, "high-performance");
  assert.deepEqual(summary.requested_features, BOOGU_WEB_REQUIRED_DEVICE_FEATURES);
  assert.deepEqual(
    summary.enabled_features,
    expectedBrowserEnabledFeatures(BOOGU_WEB_REQUIRED_DEVICE_FEATURES),
  );
  assert.deepEqual(summary.requested_limits, {});
  assert.equal(summary.selected_device_adapter_request_id, summary.selected_adapter_request_id);
});

test("accepts privacy-redacted runtime identity when exact NVIDIA evidence survives", () => {
  const ready = validBackendReady("", "Other");
  const calls = validRuntimeWebGpuCalls({ adapterName: "" });
  const attestation = attestRenderedSurfaceRuntimeAdapter(
    ready,
    calls,
    TEST_ENGINE_SESSION_ID,
    0,
  );
  assert.equal(attestation.validated, true, attestation.validation_failures.join("\n"));
});

test("accepts one exactly paired rejected adapter attempt before runtime success", () => {
  const retry = [
    {
      event: "request-adapter-start",
      at_ms: 1,
      detail: {
        request_id: 10,
        powerPreference: "high-performance",
        forceFallbackAdapter: false,
      },
    },
    {
      event: "request-adapter-rejected",
      at_ms: 2,
      detail: { request_id: 10, error: "temporarily unavailable" },
    },
    ...validRuntimeWebGpuCalls(),
  ];
  const attestation = attestRenderedSurfaceRuntimeAdapter(
    validBackendReady(),
    retry,
    TEST_ENGINE_SESSION_ID,
    0,
  );
  assert.equal(attestation.adapter_request_attempts, 2);
  assert.equal(attestation.adapter_rejected_attempts, 1);
  assert.equal(attestation.validated, true, attestation.validation_failures.join("\n"));
});

test("fails runtime proof on missing, fallback, software, mismatched, or incomplete evidence", () => {
  const cases = [
    [[], validBackendReady(), 0, /did not record any runtime WebGPU calls/],
    [
      validRuntimeWebGpuCalls({ fallback: true }),
      validBackendReady(),
      0,
      /fallback status is not explicitly false/,
    ],
    [
      validRuntimeWebGpuCalls({ vendor: "swiftshader", adapterName: "SwiftShader" }),
      validBackendReady("SwiftShader"),
      0,
      /not non-software NVIDIA hardware/,
    ],
    [
      validRuntimeWebGpuCalls({ adapterName: "NVIDIA mismatch" }),
      validBackendReady(),
      0,
      /does not exactly match Bevy adapter/,
    ],
    [validRuntimeWebGpuCalls().slice(0, -1), validBackendReady(), 0, /expected exactly one of each/],
    [validRuntimeWebGpuCalls(), validBackendReady(), 1, /dropped 1 calls/],
  ];
  for (const [calls, ready, dropped, expected] of cases) {
    const attestation = attestRenderedSurfaceRuntimeAdapter(
      ready,
      calls,
      TEST_ENGINE_SESSION_ID,
      dropped,
    );
    assert.equal(attestation.validated, false);
    assert.ok(
      attestation.validation_failures.some((failure) => expected.test(failure)),
      `${expected}: ${attestation.validation_failures.join("\n")}`,
    );
  }
});

test("fails runtime proof on unavailable instrumentation or extra/rejected devices", () => {
  const unavailable = [
    {
      event: "request-adapter-instrumentation-unavailable",
      at_ms: 1,
      detail: { error: "read only" },
    },
    ...validRuntimeWebGpuCalls(),
  ];
  const extraDevice = [
    ...validRuntimeWebGpuCalls(),
    {
      event: "request-device-start",
      at_ms: 31,
      detail: {
        request_id: 3,
        adapter_request_id: 1,
        requiredFeatures: [...BOOGU_WEB_REQUIRED_DEVICE_FEATURES],
        requiredLimits: {},
      },
    },
    {
      event: "request-device-rejected",
      at_ms: 32,
      detail: { request_id: 3, adapter_request_id: 1, error: "second device failed" },
    },
  ];
  for (const [calls, expected] of [
    [unavailable, /instrumentation or telemetry was unavailable/],
    [extraDevice, /expected exactly one of each|rejected device requests/],
  ]) {
    const attestation = attestRenderedSurfaceRuntimeAdapter(
      validBackendReady(),
      calls,
      TEST_ENGINE_SESSION_ID,
      0,
    );
    assert.equal(attestation.validated, false);
    assert.ok(
      attestation.validation_failures.some((failure) => expected.test(failure)),
      `${expected}: ${attestation.validation_failures.join("\n")}`,
    );
  }
});

test("fails closed on inexact requested or enabled browser feature sets", () => {
  const cases = [
    ["request", "requiredFeatures", [], /requested_features/],
    ["request duplicate", "requiredFeatures", ["timestamp-query", "timestamp-query"], /requested_features/],
    ["device missing all", "enabledFeatures", [], /enabled_features/],
    ["device missing required", "enabledFeatures", ["core-features-and-limits"], /enabled_features/],
    ["device missing baseline", "enabledFeatures", ["timestamp-query"], /enabled_features/],
    ["device extra", "enabledFeatures", ["core-features-and-limits", "shader-f16", "timestamp-query"], /enabled_features/],
    ["device duplicate", "enabledFeatures", ["core-features-and-limits", "timestamp-query", "timestamp-query"], /enabled_features/],
    ["device unsorted", "enabledFeatures", ["timestamp-query", "core-features-and-limits"], /enabled_features/],
  ];
  for (const [label, field, value, expected] of cases) {
    const calls = validRuntimeWebGpuCalls();
    const target = field === "requiredFeatures" ? calls[2].detail : calls[3].detail;
    target[field] = value;
    if (field === "requiredFeatures") calls[3].detail[field] = value;
    const attestation = attestRenderedSurfaceRuntimeAdapter(
      validBackendReady(),
      calls,
      TEST_ENGINE_SESSION_ID,
      0,
    );
    assert.equal(attestation.validated, false, label);
    assert.ok(
      attestation.validation_failures.some((failure) => expected.test(failure)),
      `${label}: ${attestation.validation_failures.join("\n")}`,
    );
  }
});

for (const adapter of [
  { vendor: "google", description: "SwiftShader", is_fallback_adapter: true },
  { vendor: "mesa", description: "llvmpipe" },
  { vendor: "amd", description: "AMD Radeon" },
]) {
  test(`rejects non-qualifying adapter ${adapter.description}`, () => {
    assert.notDeepEqual(validateHardwareNvidiaAdapter(adapter), []);
  });
}

test("rejects a CPU or non-shared Bevy backend event", () => {
  const evidence = validEvidence();
  evidence.bevy_backend_ready.device_type = "Cpu";
  evidence.bevy_backend_ready.shared_adapter_device_queue = false;
  const failures = validateBackendReadyEvent(
    evidence.bevy_backend_ready,
    evidence.runtime_webgpu_adapter_attestation,
  );
  assert.ok(failures.some((failure) => /GPU class|Cpu/.test(failure)));
  assert.ok(failures.some((failure) => /shared/.test(failure)));
});

test("rejects an insecure or inexact page and zero-sized canvas", () => {
  const evidence = validEvidence();
  evidence.page_snapshot.url = "http://127.0.0.1:39001/other";
  evidence.page_snapshot.secure_context = false;
  evidence.page_snapshot.canvas.width = 0;
  const failures = validateRenderedSurfaceSnapshot(
    evidence.page_snapshot,
    evidence.expected_url,
  );
  assert.ok(failures.some((failure) => /page URL/.test(failure)));
  assert.ok(failures.some((failure) => /secure context/.test(failure)));
  assert.ok(failures.some((failure) => /canvas.width/.test(failure)));
});

for (const diagnostic of [
  "WebgpuSwapChainTexture: DeviceLost",
  "wgpu device lost while presenting",
  "GPU process crashed",
  "Dawn validation failed",
]) {
  test(`classifies terminal GPU diagnostic: ${diagnostic}`, () => {
    assert.equal(gpuTerminalDiagnostic(diagnostic), true);
  });
}

test("rejects backend failure and browser errors", () => {
  const evidence = validEvidence();
  evidence.bevy_backend_events.push({ event: "failed", message: "device lost" });
  evidence.page_errors.push("uncaught page exception");
  evidence.gpu_errors.push("WebGPU device lost");
  evidence.chrome_stderr = "WebgpuSwapChainTexture DeviceLost";
  const failures = validateRenderedSurfaceEvidence(evidence);
  assert.ok(failures.some((failure) => /failed backend state/.test(failure)));
  assert.ok(failures.some((failure) => /page_errors/.test(failure)));
  assert.ok(failures.some((failure) => /gpu_errors/.test(failure)));
  assert.ok(failures.some((failure) => /Chrome stderr/.test(failure)));
});

test("binds Network.loadingFailed to exact request URL, method, Range, type, and cancellation", () => {
  const context = captureCdpNetworkRequestContext({
    requestId: "request-17",
    type: "Fetch",
    documentURL: "http://127.0.0.1:39001/index.html",
    initiator: { type: "script" },
    request: {
      url: "http://127.0.0.1:39001/model/boogu-image-0.1-turbo/transport/a.part",
      method: "GET",
      headers: { range: "bytes=0-4194303" },
    },
  });
  assert.deepEqual(context, {
    request_id: "request-17",
    url: "http://127.0.0.1:39001/model/boogu-image-0.1-turbo/transport/a.part",
    method: "GET",
    range_header: "bytes=0-4194303",
    request_type: "Fetch",
    document_url: "http://127.0.0.1:39001/index.html",
    initiator_type: "script",
    redirect_count: 0,
    model_artifact_request: true,
  });
  const diagnostic = cdpNetworkLoadingFailedDiagnostic(
    {
      requestId: "request-17",
      type: "Fetch",
      errorText: "net::ERR_ABORTED",
      canceled: true,
    },
    context,
  );
  assert.equal(diagnostic.context_bound, true);
  assert.equal(diagnostic.model_artifact_request, true);
  assert.equal(diagnostic.type, "Fetch");
  assert.equal(diagnostic.errorText, "net::ERR_ABORTED");
  assert.equal(diagnostic.error_text, "net::ERR_ABORTED");
  assert.equal(diagnostic.canceled, true);
  assert.equal(diagnostic.proven_benign_favicon, false);
  assert.ok(
    validateCdpNetworkFailureEvidence([diagnostic], []).some((failure) =>
      failure.includes("/model/boogu-image-0.1-turbo/transport/a.part"),
    ),
  );
});

test("ignores only an exact proven canceled root favicon request", () => {
  const context = captureCdpNetworkRequestContext({
    requestId: "favicon-1",
    type: "Other",
    request: {
      url: "http://127.0.0.1:39001/favicon.ico",
      method: "GET",
      headers: {},
    },
  });
  const favicon = cdpNetworkLoadingFailedDiagnostic(
    {
      requestId: "favicon-1",
      type: "Other",
      errorText: "net::ERR_ABORTED",
      canceled: true,
    },
    context,
  );
  assert.equal(isProvenBenignFaviconFailure(favicon), true);
  assert.deepEqual(validateCdpNetworkFailureEvidence([], [favicon]), []);
  for (const mutation of [
    { ...favicon, url: "http://127.0.0.1:39001/model/favicon.ico" },
    { ...favicon, range_header: "bytes=0-7" },
    { ...favicon, canceled: false },
    { ...favicon, context_bound: false },
    { ...favicon, error_text: "net::ERR_FAILED" },
  ]) {
    assert.equal(isProvenBenignFaviconFailure(mutation), false);
    assert.ok(validateCdpNetworkFailureEvidence([], [mutation]).length > 0);
  }
});

test("rendered source records structured loadingFailed context before fail-closed reporting", async () => {
  const source = await readFile(
    new URL("./wasm_rendered_surface_smoke.mjs", import.meta.url),
    "utf8",
  );
  for (const marker of [
    'message.method === "Network.requestWillBeSent"',
    "captureCdpNetworkRequestContext(message.params, previous)",
    'message.method === "Network.loadingFailed"',
    "cdpNetworkLoadingFailedDiagnostic(",
    "diagnostic.proven_benign_favicon",
    "networkFailures.push(diagnostic)",
    "ignoredBenignNetworkFailures.push(diagnostic)",
    'record("page", `network request failed: ${JSON.stringify(diagnostic)}`)',
  ]) {
    assert.ok(source.includes(marker), `rendered harness omits ${marker}`);
  }
});

test("requires exact tested JS, Wasm, page assets, runtime, and harness identities", () => {
  const evidence = validEvidence();
  assert.deepEqual(
    validateTestedPackageIdentity(
      evidence.tested_package_identity,
      evidence.served_transport,
    ),
    [],
  );

  evidence.tested_package_identity.generated_package.javascript.relative_path = "other.js";
  evidence.tested_package_identity.generated_package.webassembly.sha256 = "invalid";
  evidence.tested_package_identity.page_modules.model_selector.relative_path = "other.mjs";
  evidence.tested_package_identity.generated_package.app_icon.relative_path = "other.png";
  evidence.tested_package_identity.sources.browser_runtime.absolute_path = "relative.rs";
  evidence.tested_package_identity.sources.rendered_harness.bytes = 0;
  evidence.served_transport.generated["bevy_burn_image.js"].sha256 = "5".repeat(64);
  evidence.served_transport.page_modules["model_selector.mjs"].content_type =
    "application/octet-stream";
  evidence.served_transport.generated["burn-image-icon.png"].sha256 = "9".repeat(64);
  const failures = validateRenderedSurfaceEvidence(evidence);
  for (const pattern of [
    /javascript.relative_path/,
    /webassembly.sha256/,
    /model_selector.relative_path/,
    /app_icon.relative_path/,
    /browser_runtime.absolute_path/,
    /rendered_harness.bytes/,
    /served bevy_burn_image.js/,
    /served model_selector.mjs MIME/,
    /served burn-image-icon.png MIME/,
  ]) {
    assert.ok(failures.some((failure) => pattern.test(failure)), String(pattern));
  }
});

function validPackedF16TensorInputDiagnostic(name, shape, digestMarker) {
  const elementCount = shape.reduce((product, dimension) => product * dimension, 1);
  return {
    name,
    shape: [...shape],
    dtype: "f32",
    element_count: elementCount,
    finite_element_count: elementCount,
    all_finite: true,
    max_abs: name === "first_timestep" ? 0.0010000000474974513 : 3.25,
    mean: name === "first_timestep" ? 0.0010000000474974513 : 0.125,
    rms: name === "first_timestep" ? 0.0010000000474974513 : 1.25,
    sha256: digestMarker.length === 64 ? digestMarker : digestMarker.repeat(64),
  };
}

function validPackedF16PreDmdInputDiagnostics(runId = 7) {
  return {
    at_ms: 2_000,
    event: "packed_f16_pre_dmd_input_diagnostics",
    run_id: runId,
    diagnostics: {
      scope: TURBO_PACKED_F16_PRE_DMD_INPUT_SCOPE,
      policy: {
        qwen_release_unused_memory_after_stage: false,
        qwen_text_block_load_synchronization_policy:
          TURBO_QWEN_TEXT_BLOCK_LOAD_SYNCHRONIZATION_POLICY,
        qwen_text_layer_submission_policy: TURBO_QWEN_TEXT_LAYER_SUBMISSION_POLICY,
        packed_qwen_instruction_handoff_policy: TURBO_PACKED_F16_QWEN_HANDOFF_POLICY,
        cleanup_completed: true,
        post_cleanup_packed_cache: {
          state: "ready",
          cache_ready: true,
          cached_stages: TURBO_PACKED_F16_CACHED_STAGES,
          cached_objects: TURBO_PACKED_F16_CACHED_OBJECTS,
          cached_tensors: TURBO_PACKED_F16_CACHED_TENSORS,
          cached_bytes: TURBO_PACKED_F16_RESOURCE_PLAN.retained_packed_f16_denoiser_bytes,
        },
      },
      dmd_steps: 4,
      instruction: validPackedF16TensorInputDiagnostic(
        "instruction",
        [1, 45, 4096],
        "1",
      ),
      initial_latent: validPackedF16TensorInputDiagnostic(
        "initial_latent",
        [1, 16, 128, 128],
        "2",
      ),
      renoise: [
        validPackedF16TensorInputDiagnostic("renoise_0", [1, 16, 128, 128], "3"),
        validPackedF16TensorInputDiagnostic("renoise_1", [1, 16, 128, 128], "4"),
        validPackedF16TensorInputDiagnostic("renoise_2", [1, 16, 128, 128], "5"),
      ],
      first_timestep: validPackedF16TensorInputDiagnostic("first_timestep", [1], "6"),
      all_inputs_finite: true,
    },
  };
}

function validPackedF16QwenPreHandoffDiagnostics(runId = 7) {
  const stageOutputs = [
    validPackedF16TensorInputDiagnostic("qwen_embedding_output", [1, 45, 4096], "a"),
    ...Array.from({ length: 36 }, (_, index) =>
      validPackedF16TensorInputDiagnostic(
        `qwen_text_block_${String(index).padStart(2, "0")}_output`,
        [1, 45, 4096],
        (index % 16).toString(16),
      ),
    ),
    validPackedF16TensorInputDiagnostic("qwen_final_norm_output", [1, 45, 4096], "f"),
  ];
  stageOutputs[0].sha256 = TURBO_PACKED_F16_QWEN_HOST_EMBEDDING_SHA256;
  return {
    at_ms: 1_070,
    event: "packed_f16_qwen_pre_handoff_diagnostics",
    run_id: runId,
    diagnostics: {
      scope: TURBO_PACKED_F16_QWEN_PRE_HANDOFF_SCOPE,
      effective_instruction_length: 45,
      expected_stage_output_count: TURBO_PACKED_F16_QWEN_STAGE_OUTPUT_COUNT,
      stage_outputs: stageOutputs,
      stage_names_exact: true,
      qwen_last_hidden_state_before_trim: validPackedF16TensorInputDiagnostic(
        "qwen_last_hidden_state_before_trim",
        [1, 45, 4096],
        "f",
      ),
      instruction_after_trim_cast_before_handoff: validPackedF16TensorInputDiagnostic(
        "instruction_after_trim_cast_before_handoff",
        [1, 45, 4096],
        "1",
      ),
      all_tensors_finite: true,
      no_tensor_all_zero: true,
      first_non_finite_tensor: null,
      first_all_zero_tensor: null,
      final_norm_matches_returned_output: true,
      block_00_immediate_post_sync: validPackedF16QwenBlock0PostSyncDiagnostic(runId).diagnostic,
      block_00_immediate_matches_delayed_capture: true,
    },
  };
}

function validPackedF16QwenBlock0PostSyncDiagnostic(runId = 7) {
  return {
    at_ms: 1_045,
    event: "packed_f16_qwen_block0_post_sync_diagnostic",
    run_id: runId,
    diagnostic: {
      scope: TURBO_PACKED_F16_QWEN_BLOCK0_POST_SYNC_SCOPE,
      block0_execution_mode: TURBO_QWEN_BLOCK0_SERIALIZED_DIAGNOSTIC_MODE,
      text_layer_allocation_policy: TURBO_QWEN_TEXT_LAYER_ALLOCATION_POLICY,
      text_block_load_synchronization_policy:
        TURBO_QWEN_TEXT_BLOCK_LOAD_SYNCHRONIZATION_POLICY,
      qwen_text_layer_submission_policy: TURBO_QWEN_TEXT_LAYER_SUBMISSION_POLICY,
      tensor: validPackedF16TensorInputDiagnostic(
        "qwen_text_block_00_output_immediate_post_sync",
        [1, 45, 4096],
        "0",
      ),
      all_finite: true,
      not_all_zero: true,
    },
  };
}

function validPackedF16QwenBlock0ExecutionDiagnostics(runId = 7) {
  const digestMarkers = [
    TURBO_PACKED_F16_QWEN_HOST_EMBEDDING_SHA256,
    "1",
    TURBO_PACKED_F16_QWEN_HOST_EMBEDDING_SHA256,
    "3",
    "4",
    "5",
    "6",
    "7",
    "0",
  ];
  const boundaries = TURBO_PACKED_F16_QWEN_BLOCK0_BOUNDARIES.map((expected, index) => ({
    sequence_index: index,
    boundary: expected.boundary,
    tensor_kind: expected.tensor_kind,
    tensor: validPackedF16TensorInputDiagnostic(
      `qwen_text_block_00_${expected.boundary}_immediate_post_sync`,
      expected.tensor_kind === "parameter-sentinel" ? [4096] : [1, 45, 4096],
      digestMarkers[index],
    ),
    all_finite: true,
    not_all_zero: true,
  }));
  return {
    at_ms: 1_040,
    event: "packed_f16_qwen_block0_execution_diagnostics",
    run_id: runId,
    diagnostics: {
      scope: TURBO_PACKED_F16_QWEN_BLOCK0_EXECUTION_SCOPE,
      block0_execution_mode: TURBO_QWEN_BLOCK0_SERIALIZED_DIAGNOSTIC_MODE,
      text_layer_allocation_policy: TURBO_QWEN_TEXT_LAYER_ALLOCATION_POLICY,
      text_block_load_synchronization_policy:
        TURBO_QWEN_TEXT_BLOCK_LOAD_SYNCHRONIZATION_POLICY,
      qwen_text_layer_submission_policy: TURBO_QWEN_TEXT_LAYER_SUBMISSION_POLICY,
      expected_boundary_count: TURBO_PACKED_F16_QWEN_BLOCK0_BOUNDARIES.length,
      captured_boundary_count: TURBO_PACKED_F16_QWEN_BLOCK0_BOUNDARIES.length,
      boundaries,
      boundary_names_exact: true,
      all_captured_tensors_finite: true,
      no_captured_tensor_all_zero: true,
      identity_add_canary_matches_input: true,
      complete: true,
      first_failure_boundary: null,
      failure_reason: null,
    },
  };
}

function validPackedF16QwenHostEmbedding(runId = 7) {
  return {
    at_ms: 1_050,
    event: "packed_f16_qwen_host_embedding",
    run_id: runId,
    report: {
      policy: TURBO_PACKED_F16_QWEN_HOST_EMBEDDING_POLICY,
      shape: [1, 45, 4096],
      dtype: "f32",
      input_token_count: 45,
      unique_token_count: 33,
      plan_chunk_count: TURBO_PACKED_F16_QWEN_HOST_EMBEDDING_PLAN.expected_chunk_count,
      authenticated_object_count:
        TURBO_PACKED_F16_QWEN_HOST_EMBEDDING_PLAN.expected_object_count,
      authenticated_object_bytes:
        TURBO_PACKED_F16_QWEN_HOST_EMBEDDING_PLAN.authenticated_object_bytes,
      authenticated_f16_payload_bytes:
        TURBO_PACKED_F16_QWEN_HOST_EMBEDDING_PLAN.authenticated_f16_payload_bytes,
      selected_row_occurrences: 45,
      selected_unique_rows: 33,
      selected_f16_bytes: TURBO_PACKED_F16_QWEN_HOST_EMBEDDING_F16_BYTES,
      host_f32_payload_bytes: TURBO_PACKED_F16_QWEN_HOST_EMBEDDING_F32_BYTES,
      host_to_device_upload_bytes: TURBO_PACKED_F16_QWEN_HOST_EMBEDDING_F32_BYTES,
      immediate_device_to_host_readback_bytes:
        TURBO_PACKED_F16_QWEN_HOST_EMBEDDING_F32_BYTES,
      total_device_transfer_bytes: TURBO_PACKED_F16_QWEN_HOST_EMBEDDING_F32_BYTES * 2,
      host_f32_sha256: TURBO_PACKED_F16_QWEN_HOST_EMBEDDING_SHA256,
      device_f32_sha256: TURBO_PACKED_F16_QWEN_HOST_EMBEDDING_SHA256,
      device_roundtrip_verified_before_text: true,
      device_roundtrip_digest_matches: true,
      all_finite: true,
      not_all_zero: true,
      coverage_complete: true,
    },
  };
}

function validPackedF16QwenPostHandoffDiagnostics(runId = 7) {
  const instruction = validPackedF16TensorInputDiagnostic(
    "instruction_after_handoff",
    [1, 45, 4096],
    "1",
  );
  return {
    at_ms: 1_080,
    event: "packed_f16_qwen_post_handoff_diagnostics",
    run_id: runId,
    diagnostics: {
      scope: TURBO_PACKED_F16_QWEN_POST_HANDOFF_SCOPE,
      handoff: {
        policy: TURBO_PACKED_F16_QWEN_HANDOFF_POLICY,
        qwen_release_unused_memory_after_stage: false,
        qwen_text_layer_allocation_policy: TURBO_QWEN_TEXT_LAYER_ALLOCATION_POLICY,
        qwen_text_block_load_synchronization_policy:
          TURBO_QWEN_TEXT_BLOCK_LOAD_SYNCHRONIZATION_POLICY,
        qwen_text_layer_submission_policy: TURBO_QWEN_TEXT_LAYER_SUBMISSION_POLICY,
        shape: [1, 45, 4096],
        dtype: "f32",
        element_count: 184_320,
        payload_bytes: 737_280,
        device_to_host_readback_bytes: 1_474_560,
        host_to_device_upload_bytes: 737_280,
        total_transfer_bytes: 2_211_840,
        before_sha256: "1".repeat(64),
        after_sha256: "1".repeat(64),
        all_finite: true,
        not_all_zero: true,
        digest_matches: true,
        cleanup_completed: true,
        packed_cache: {
          state: "ready",
          cache_ready: true,
          cached_stages: TURBO_PACKED_F16_CACHED_STAGES,
          cached_objects: TURBO_PACKED_F16_CACHED_OBJECTS,
          cached_tensors: TURBO_PACKED_F16_CACHED_TENSORS,
          cached_bytes: TURBO_PACKED_F16_RESOURCE_PLAN.retained_packed_f16_denoiser_bytes,
        },
      },
      instruction_after_handoff: instruction,
    },
  };
}

function validPackedF16Lifecycle(preloadAttemptCount = 1) {
  return {
    ...structuredClone(TURBO_PACKED_F16_REQUEST_LIFECYCLE),
    authenticated_artifact_bytes:
      TURBO_PACKED_F16_RESOURCE_PLAN.authenticated_artifact_bytes * preloadAttemptCount,
    packed_upload_bytes:
      TURBO_PACKED_F16_RESOURCE_PLAN.retained_packed_f16_denoiser_bytes *
      preloadAttemptCount,
    preload_attempt_count: preloadAttemptCount,
  };
}

function validPackedF16DmdVaeHandoff(
  runId = 7,
  preloadAttemptCount = 1,
  atMs = 8_830,
) {
  const readyCache = {
    state: "ready",
    cache_ready: true,
    cached_stages: TURBO_PACKED_F16_CACHED_STAGES,
    cached_objects: TURBO_PACKED_F16_CACHED_OBJECTS,
    cached_tensors: TURBO_PACKED_F16_CACHED_TENSORS,
    cached_bytes: TURBO_PACKED_F16_RESOURCE_PLAN.retained_packed_f16_denoiser_bytes,
  };
  const emptyCache = {
    state: "empty",
    cache_ready: false,
    cached_stages: 0,
    cached_objects: 0,
    cached_tensors: 0,
    cached_bytes: 0,
  };
  return {
    at_ms: atMs,
    event: "packed_f16_dmd_vae_handoff",
    run_id: runId,
    report: {
      policy: TURBO_PACKED_F16_DMD_VAE_HANDOFF_POLICY,
      next_request_rehydration_policy:
        TURBO_PACKED_F16_NEXT_REQUEST_REHYDRATION_POLICY,
      shape: [1, 16, 128, 128],
      dtype: "f32",
      element_count: 262_144,
      payload_bytes: 1_048_576,
      device_to_host_readback_bytes: 2_097_152,
      host_to_device_upload_bytes: 1_048_576,
      total_transfer_bytes: 3_145_728,
      before_sha256: "7".repeat(64),
      after_sha256: "7".repeat(64),
      all_finite: true,
      not_all_zero: true,
      digest_matches: true,
      wrapper_cached_stages_before_clear: 0,
      wrapper_cached_stages_after_clear: 0,
      synchronization_pending_before_cleanup: false,
      synchronization_pending_after_cleanup: false,
      rope_cache_cleared: true,
      cleanup_completed: true,
      packed_cache_before_cleanup: readyCache,
      packed_cache_after_cleanup: emptyCache,
      preload_attempt_count: preloadAttemptCount,
      expected_next_request_preload_attempt_count: preloadAttemptCount + 1,
    },
  };
}

function validSurfaceTextureGateWindow(
  runId,
  {
    suspendedAt = 950,
    resumedAt = 9_010,
    preRequestAt = 800,
    postResumeAt = 9_030,
    acquisitionCountAtGate = 40,
  } = {},
) {
  return {
    run_id: runId,
    policy: SURFACE_INFERENCE_POLICY,
    resume_policy: SURFACE_INFERENCE_POLICY,
    suspended_at_ms: suspendedAt,
    resumed_at_ms: resumedAt,
    terminal: "completed",
    acquisition_count_at_suspend: acquisitionCountAtGate,
    acquisition_count_at_resume: acquisitionCountAtGate,
    gated_call_count: 0,
    pre_request_acquisition: {
      call_index: acquisitionCountAtGate,
      at_ms: preRequestAt,
      canvas_id: "burn-image",
      canvas_width: 1800,
      canvas_height: 1200,
      succeeded: true,
    },
    first_successful_post_resume_acquisition: {
      call_index: acquisitionCountAtGate + 1,
      at_ms: postResumeAt,
      canvas_id: "burn-image",
      canvas_width: 1800,
      canvas_height: 1200,
      succeeded: true,
    },
  };
}

function validModelTransport() {
  const bundles = [
    "boogu-image-0.1-turbo",
    "qwen3-vl-8b-base-boogu-image-0.1",
    "flux1-vae-boogu-image-0.1",
  ].map((bundle, index) => ({
    bundle,
    content_digest: String(index + 1).repeat(64),
    logical_artifacts: {
      file_count: 10 + index,
      bytes: 1_000 + index,
      max_file_bytes: 900 + index,
      weight_file_count: 2,
      weight_bytes: 900 + index,
      max_weight_file_bytes: 500 + index,
    },
    direct_artifacts: { file_count: 8 + index, bytes: 100, max_file_bytes: 50 },
    physical_transport: {
      part_reference_count: 2,
      unique_part_count: 2,
      reconstructed_bytes: 900 + index,
      unique_part_bytes: 900 + index,
      max_part_bytes: 500 + index,
      target_part_bytes: ARTIFACT_TRANSPORT_TARGET_PART_BYTES,
      hard_max_part_bytes: ARTIFACT_TRANSPORT_HARD_MAX_PART_BYTES,
      every_part_statted: true,
      part_sha256_policy: "verified-by-browser-runtime-before-use",
    },
    transport_sidecar: {
      path: ARTIFACT_TRANSPORT_LAYOUT_PATH,
      size: 100 + index,
      sha256: String(index + 4).repeat(64),
      authenticated: true,
    },
    manifest_sha256: String(index + 7).repeat(64),
  }));
  return {
    policy: "exact-local-modular-part-only-parent-plus-qwen-vae-siblings-range-cors",
    bundles,
    total_logical_artifact_files: bundles.reduce(
      (total, bundle) => total + bundle.logical_artifacts.file_count,
      0,
    ),
    total_logical_artifact_bytes: bundles.reduce(
      (total, bundle) => total + bundle.logical_artifacts.bytes,
      0,
    ),
    total_physical_transport_parts: bundles.reduce(
      (total, bundle) => total + bundle.physical_transport.unique_part_count,
      0,
    ),
    total_physical_transport_part_bytes: bundles.reduce(
      (total, bundle) => total + bundle.physical_transport.unique_part_bytes,
      0,
    ),
    maximum_physical_transport_part_bytes: Math.max(
      ...bundles.map((bundle) => bundle.physical_transport.max_part_bytes),
    ),
    validated: true,
  };
}

function validTurboEvidence() {
  const runId = 7;
  const preDmdInputDiagnostics = validPackedF16PreDmdInputDiagnostics(runId);
  const qwenHostEmbedding = validPackedF16QwenHostEmbedding(runId);
  const qwenBlock0ExecutionDiagnostics = validPackedF16QwenBlock0ExecutionDiagnostics(runId);
  const qwenBlock0PostSyncDiagnostic = validPackedF16QwenBlock0PostSyncDiagnostic(runId);
  const qwenPreHandoffDiagnostics = validPackedF16QwenPreHandoffDiagnostics(runId);
  const qwenPostHandoffDiagnostics = validPackedF16QwenPostHandoffDiagnostics(runId);
  const dmdVaeHandoff = validPackedF16DmdVaeHandoff(runId);
  const lifecycle = validPackedF16Lifecycle();
  const denoiserPreloadTraffic = { ...TURBO_DENOISER_PRELOAD_TRAFFIC };
  const artifactTraffic = { ...TURBO_GENERATE_REQUEST_TRAFFIC };
  const modelBaseUrls = [
    "http://127.0.0.1:39001/model/boogu-image-0.1-turbo",
    "http://127.0.0.1:39001/model/flux1-vae-boogu-image-0.1",
    "http://127.0.0.1:39001/model/qwen3-vl-8b-base-boogu-image-0.1",
  ];
  const cdpTraffic = (policy, terminalEvent, traffic, start, end) => ({
    policy,
    model_base_urls: [
      ...modelBaseUrls,
    ],
    window_start_epoch_ms: start,
    window_end_epoch_ms: end,
    terminal_event: terminalEvent,
    model_response_count: traffic.network_requests,
    http_200_complete_part_response_count: traffic.network_requests,
    http_206_response_count: 0,
    complete_object_validated_response_count: traffic.network_requests,
    content_range_validated_response_count: 0,
    response_body_bytes: traffic.network_response_bytes,
    unexpected_status_response_count: 0,
    missing_content_length_count: 0,
    invalid_content_range_response_count: 0,
  });
  const cdpPreloadNetworkTraffic = cdpTraffic(
    TURBO_CDP_PRELOAD_NETWORK_POLICY,
    "packed_f16_denoiser_preload",
    denoiserPreloadTraffic,
    1_000_000_100,
    1_000_000_900,
  );
  const cdpNetworkTraffic = cdpTraffic(
    TURBO_CDP_REQUEST_NETWORK_POLICY,
    "ready",
    artifactTraffic,
    1_000_001_000,
    1_000_009_000,
  );
  return {
    ...validRuntimeEvidence(),
    qwen_block0_execution_mode: TURBO_QWEN_BLOCK0_SERIALIZED_DIAGNOSTIC_MODE,
    runtime_events: [
      {
        event: "packed_f16_resource_plan",
        ...TURBO_PACKED_F16_RESOURCE_PLAN,
      },
      { at_ms: 100, event: "preparing", message: TURBO_PACKED_F16_PRELOAD_MESSAGE },
      {
        at_ms: 900,
        event: "packed_f16_denoiser_preload",
        traffic: denoiserPreloadTraffic,
        cached_stages: TURBO_PACKED_F16_CACHED_STAGES,
        cached_objects: TURBO_PACKED_F16_CACHED_OBJECTS,
        cached_tensors: TURBO_PACKED_F16_CACHED_TENSORS,
        cached_bytes: TURBO_PACKED_F16_RESOURCE_PLAN.retained_packed_f16_denoiser_bytes,
        previous_preload_attempt_count: 0,
        preload_attempt_count: 1,
        request_scoped_rehydration: false,
        rehydration_policy: TURBO_PACKED_F16_NEXT_REQUEST_REHYDRATION_POLICY,
      },
      {
        event: "ready",
        model: TURBO_MODEL_ID,
        block0_execution_mode: TURBO_QWEN_BLOCK0_SERIALIZED_DIAGNOSTIC_MODE,
        qwen_text_layer_allocation_policy: TURBO_QWEN_TEXT_LAYER_ALLOCATION_POLICY,
        qwen_text_block_load_synchronization_policy:
          TURBO_QWEN_TEXT_BLOCK_LOAD_SYNCHRONIZATION_POLICY,
        qwen_text_layer_submission_policy: TURBO_QWEN_TEXT_LAYER_SUBMISSION_POLICY,
      },
      {
        at_ms: 950,
        event: "surface_inference_suspended",
        run_id: runId,
        policy: SURFACE_INFERENCE_POLICY,
        primary_window_camera_count: 2,
        saved_camera_state_count: 2,
        previously_active_camera_count: 2,
        inactive_camera_count: 2,
        active_job_count: 1,
        suspended_before_runtime_submit: true,
        all_primary_window_cameras_inactive: true,
      },
      structuredClone(qwenBlock0ExecutionDiagnostics),
      structuredClone(qwenBlock0PostSyncDiagnostic),
      structuredClone(qwenHostEmbedding),
      structuredClone(qwenPreHandoffDiagnostics),
      structuredClone(qwenPostHandoffDiagnostics),
      structuredClone(preDmdInputDiagnostics),
      {
        at_ms: 8_800,
        event: "packed_f16_denoiser_lifecycle",
        lifecycle: structuredClone(lifecycle),
      },
      structuredClone(dmdVaeHandoff),
      { at_ms: 8_900, event: "artifact_traffic", traffic: artifactTraffic },
      {
        at_ms: 9_010,
        event: "surface_inference_resumed",
        run_id: runId,
        policy: SURFACE_INFERENCE_POLICY,
        terminal: "completed",
        primary_window_camera_count: 2,
        saved_camera_state_count: 2,
        restored_camera_state_count: 2,
        restored_active_camera_count: 2,
        active_job_count: 0,
        resumed_after_runtime_terminal: true,
        resumed_before_output_ready: true,
        exact_saved_states_restored: true,
        all_primary_window_cameras_restored: true,
      },
    ],
    progress_events: [
      { at_ms: 1_000, event: "run_started", run_id: runId, model: TURBO_MODEL_ID, task: "generate" },
      { at_ms: 1_010, event: "stage_started", run_id: runId, stage: "qwen", total_steps: 1 },
      { at_ms: 1_060, event: "stage_completed", run_id: runId, stage: "qwen", elapsed_micros: 1 },
      { at_ms: 1_100, event: "stage_started", run_id: runId, stage: "dmd", total_steps: 4 },
      ...Array.from({ length: 4 }, (_, index) => ({
        at_ms: 3_000 + index * 1_000,
        event: "step",
        run_id: runId,
        stage: "dmd",
        step: index + 1,
        total_steps: 4,
        elapsed_micros: index + 1,
      })),
      { at_ms: 8_000, event: "stage_completed", run_id: runId, stage: "dmd", elapsed_micros: 5 },
      { at_ms: 8_840, event: "stage_started", run_id: runId, stage: "vae-decode", total_steps: 1 },
      { at_ms: 8_880, event: "stage_completed", run_id: runId, stage: "vae-decode", elapsed_micros: 1 },
      { at_ms: 9_000, event: "run_completed", run_id: runId, elapsed_micros: 1 },
    ],
    packed_f16_denoiser_preload_traffic: denoiserPreloadTraffic,
    packed_f16_denoiser_lifecycle: structuredClone(lifecycle),
    packed_f16_dmd_vae_handoff: structuredClone(dmdVaeHandoff),
    packed_f16_qwen_host_embedding: structuredClone(qwenHostEmbedding),
    packed_f16_qwen_block0_execution_diagnostics: structuredClone(
      qwenBlock0ExecutionDiagnostics,
    ),
    packed_f16_qwen_block0_post_sync_diagnostic: structuredClone(
      qwenBlock0PostSyncDiagnostic,
    ),
    packed_f16_qwen_pre_handoff_diagnostics: structuredClone(qwenPreHandoffDiagnostics),
    packed_f16_qwen_post_handoff_diagnostics: structuredClone(qwenPostHandoffDiagnostics),
    packed_f16_pre_dmd_input_diagnostics: structuredClone(preDmdInputDiagnostics),
    cdp_preload_network_traffic: cdpPreloadNetworkTraffic,
    artifact_traffic: artifactTraffic,
    cdp_network_traffic: cdpNetworkTraffic,
    modular_artifact_transport: validModelTransport(),
    ui_contract: {
      event: "ready",
      model: TURBO_MODEL_ID,
      width: 1024,
      height: 1024,
      prompt_x: 100,
      prompt_y: 200,
      prompt_enabled: true,
      prompt_focused: false,
      seed_x: 100,
      seed_y: 300,
      seed_enabled: true,
      seed_focused: true,
      run_x: 100,
      run_y: 400,
      run_enabled: true,
      save_x: 300,
      save_y: 400,
      save_enabled: true,
    },
    output_ready: {
      at_ms: 9_020,
      event: "ready",
      job_id: "7",
      output_index: 0,
      width: 1024,
      height: 1024,
      model: TURBO_MODEL_ID,
      model_revision: "pinned",
      numeric_format: "f16-qwen-vision-f32",
      backend: LOW_VRAM_BACKEND,
      artifacts_verified: true,
      artifact_content_digest: TURBO_PRODUCTION_CONTENT_DIGEST,
    },
    surface_texture_gate_windows: [validSurfaceTextureGateWindow(runId)],
    surface_texture_gate_windows_overflow: 0,
    surface_texture_gate_violation_calls: [],
    surface_texture_gate_violation_calls_overflow_start: 0,
    surface_texture_gate_violation_calls_overflow_end: 0,
    surface_texture_gate_overlap_count: 0,
    surface_texture_acquisition_count_start: 30,
    surface_texture_acquisition_count_end: 45,
    surface_texture_acquisition_failure_count_start: 0,
    surface_texture_acquisition_failure_count_end: 0,
    active_surface_gate_after_request: null,
    surface_inference_state_after_request: null,
    fixed_ascii_prompt: "A studio photograph of a blue ceramic bird on a plain white table.",
    interaction: {
      mechanism: "cdp-keyboard-and-mouse",
      prompt_typed_via_cdp: true,
      prompt_value: "A studio photograph of a blue ceramic bird on a plain white table.",
      prompt_input: {
        focus_event: { prompt_focused: true },
        focus_click: { click_count: 1 },
        replacement_mode: "known-empty-direct-keyboard-entry",
        selection_clicks: [],
      },
      seed_typed_via_cdp: true,
      seed_value: "0",
      seed_intermediate_input: {
        focus_event: { seed_focused: true },
        focus_click: { click_count: 1 },
        replacement_mode: "bevy-editable-text-triple-click-select-all",
        selection_clicks: [{ click_count: 2 }, { click_count: 3 }],
      },
      seed_input: {
        focus_event: { seed_focused: true },
        focus_click: { click_count: 1 },
        replacement_mode: "bevy-editable-text-triple-click-select-all",
        selection_clicks: [{ click_count: 2 }, { click_count: 3 }],
      },
      run_clicked_via_cdp: true,
      save_clicked_via_cdp: true,
    },
    downloaded_png: {
      path: "/tmp/burn-image/downloads/burn-image-7.png",
      file_name: "burn-image-7.png",
      bytes: 4_096,
      sha256: "b".repeat(64),
      signature_hex: "89504e470d0a1a0a",
      ihdr: { length: 13, type: "IHDR", width: 1024, height: 1024 },
      browser_events: {
        will_begin: {
          method: "Browser.downloadWillBegin",
          params: { guid: "download-guid", suggestedFilename: "burn-image-7.png" },
        },
        completed: {
          method: "Browser.downloadProgress",
          params: { guid: "download-guid", state: "completed", receivedBytes: 4_096 },
        },
      },
    },
    canvas_png_changed_bytes: 1,
    model_screenshot_bytes: 4096,
    native_gpu_attestation: {
      provider: "nvidia-smi",
      interval_aggregation_policy: GPU_INTERVAL_AGGREGATION_POLICY,
      maximum_framebuffer_bytes_exclusive: LOW_VRAM_DEVICE_CAP_BYTES,
      observed_peak_aggregate_framebuffer_bytes: 24_000_000_000,
      matched_sample_intervals: 10,
      active_sample_intervals: 5,
      observed_gpu_processes: [
        { pid: 1001, command: "chrome --type=gpu-process" },
        { pid: 1002, command: "chrome --type=gpu-process" },
      ],
      validation_failures: [],
      validated: true,
    },
  };
}

test("aggregates all Chrome GPU PIDs in an interval without duplicate double-counting", () => {
  const aggregate = aggregateChromeGpuInterval(
    [
      { gpu_index: 0, pid: 1001, framebuffer_mib: 10_000, sm_percent: 20 },
      { gpu_index: 0, pid: 1002, framebuffer_mib: 12_000, sm_percent: 30 },
      { gpu_index: 0, pid: 1002, framebuffer_mib: 11_500, sm_percent: 35 },
      { gpu_index: 0, pid: 9999, framebuffer_mib: 40_000, sm_percent: 99 },
    ],
    new Set([1001, 1002]),
  );
  assert.equal(aggregate.matched_rows.length, 2);
  assert.equal(aggregate.aggregate_framebuffer_mib, 22_000);
  assert.equal(aggregate.aggregate_sm_percent, 55);
});

test("accepts ordinary-UI low-VRAM Turbo 1024 evidence", () => {
  assert.deepEqual(validateTurbo1024ModelEvidence(validTurboEvidence()), []);
  assert.ok(LOW_VRAM_BACKEND.endsWith("/request-scoped-surface-acquire-suspended"));
});

test("requires an exact canonical u64 output job ID string bound to numeric run_id", () => {
  assert.equal(isCanonicalU64DecimalString("0"), true);
  assert.equal(isCanonicalU64DecimalString("18446744073709551615"), true);
  assert.equal(outputJobIdMatchesNumericRunId("9007199254740991", Number.MAX_SAFE_INTEGER), true);
  for (const invalid of [
    7,
    7n,
    "07",
    "+7",
    "-7",
    "7.0",
    " 7",
    "7 ",
    "18446744073709551616",
    null,
  ]) {
    assert.equal(isCanonicalU64DecimalString(invalid), false, String(invalid));
    const evidence = validTurboEvidence();
    evidence.output_ready.job_id = invalid;
    assert.ok(
      validateTurbo1024ModelEvidence(evidence).some((failure) =>
        /canonical u64 decimal string/.test(failure),
      ),
      String(invalid),
    );
  }

  const mismatched = validTurboEvidence();
  mismatched.output_ready.job_id = "8";
  assert.ok(
    validateTurbo1024ModelEvidence(mismatched).some((failure) =>
      /exact decimal representation of run_started/.test(failure),
    ),
  );
  assert.equal(outputJobIdMatchesNumericRunId("9007199254740992", Number.MAX_SAFE_INTEGER + 1), false);
});

test("reuses exact post-request Run readiness without requiring a duplicate ready event", () => {
  const postRequestReady = {
    at_ms: 10,
    event: "ready",
    model: TURBO_MODEL_ID,
    width: 1024,
    height: 1024,
    run_enabled: true,
    save_enabled: true,
  };
  const seedChanged = { at_ms: 20, event: "seed_changed", value: "1" };
  const resolved = resolveTurboSecondRequestRunReadyUiContract({
    uiEvents: [postRequestReady, seedChanged],
    uiStartIndex: 1,
    seedChangedEvent: seedChanged,
    postRequestUiContract: postRequestReady,
  });
  assert.deepEqual(resolved.uiContract, postRequestReady);
  assert.deepEqual(resolved.evidence, {
    policy: TURBO_SECOND_REQUEST_RUN_READY_POLICY,
    source: "exact-last-pre-boundary-post-request-ready",
    ui_start_index: 1,
    ui_event_count_at_resolution: 2,
    seed_changed_event_index: 1,
    run_ready_event_index: 0,
    duplicate_ready_after_seed_change_required: false,
    fallback_exact_last_pre_boundary_ready: true,
    selected_ui_contract: postRequestReady,
  });

  const localReady = { ...postRequestReady, at_ms: 15 };
  const local = resolveTurboSecondRequestRunReadyUiContract({
    uiEvents: [postRequestReady, localReady, seedChanged],
    uiStartIndex: 1,
    seedChangedEvent: seedChanged,
    postRequestUiContract: postRequestReady,
  });
  assert.deepEqual(local.uiContract, localReady);
  assert.equal(local.evidence.source, "second-request-ui-partition");
  assert.equal(local.evidence.run_ready_event_index, 1);
});

test("rejects stale, disabled, or wrong second-request Run readiness contracts", () => {
  const valid = {
    at_ms: 10,
    event: "ready",
    model: TURBO_MODEL_ID,
    width: 1024,
    height: 1024,
    run_enabled: true,
    save_enabled: true,
  };
  const seedChanged = { at_ms: 20, event: "seed_changed", value: "1" };
  const resolve = (postRequestUiContract, uiEvents = [postRequestUiContract, seedChanged]) =>
    resolveTurboSecondRequestRunReadyUiContract({
      uiEvents,
      uiStartIndex: 1,
      seedChangedEvent: seedChanged,
      postRequestUiContract,
    });

  assert.throws(
    () => resolve({ ...valid, at_ms: 9 }, [valid, seedChanged]),
    /fallback is stale or differs from the last pre-boundary ready event/,
  );
  assert.throws(() => resolve({ ...valid, run_enabled: false }), /Run control is disabled/);
  assert.throws(() => resolve({ ...valid, model: "wrong/model" }), /expected exact Turbo/);
  assert.throws(() => resolve({ ...valid, width: 512 }), /expected 1024x1024/);
  assert.throws(
    () => resolve(valid, [valid, { ...valid, at_ms: 15, model: "wrong/model" }, seedChanged]),
    /partition-local Run readiness model=.*expected exact Turbo/,
  );
  assert.throws(
    () => resolveTurboSecondRequestRunReadyUiContract({
      uiEvents: [valid],
      uiStartIndex: 1,
      seedChangedEvent: seedChanged,
      postRequestUiContract: valid,
    }),
    /outside the second-request UI partition/,
  );
});

test("rejects missing, duplicated, misordered, or unbound surface-gate evidence", () => {
  const cases = [
    {
      mutate(evidence) {
        evidence.runtime_events = evidence.runtime_events.filter(
          (event) => event.event !== "surface_inference_suspended",
        );
      },
      expected: /exactly one surface suspend\/resume pair/,
    },
    {
      mutate(evidence) {
        evidence.runtime_events.push(
          structuredClone(
            evidence.runtime_events.find(
              (event) => event.event === "surface_inference_resumed",
            ),
          ),
        );
      },
      expected: /exactly one surface suspend\/resume pair/,
    },
    {
      mutate(evidence) {
        evidence.runtime_events.find(
          (event) => event.event === "surface_inference_suspended",
        ).at_ms = 1_001;
      },
      expected: /did not precede run_started/,
    },
    {
      mutate(evidence) {
        evidence.runtime_events.find(
          (event) => event.event === "surface_inference_suspended",
        ).policy = "weaker-surface-policy";
        evidence.runtime_events.find(
          (event) => event.event === "surface_inference_resumed",
        ).run_id = 99;
      },
      expected: /policy is missing or inexact|run ID differs/,
    },
    {
      mutate(evidence) {
        const window = evidence.surface_texture_gate_windows[0];
        window.acquisition_count_at_resume += 1;
        window.gated_call_count = 1;
        evidence.surface_texture_gate_violation_calls.push({
          run_id: 7,
          policy: SURFACE_INFERENCE_POLICY,
          call_index: window.acquisition_count_at_resume,
          at_ms: 5_000,
          canvas_id: "burn-image",
          succeeded: true,
        });
      },
      expected: /getCurrentTexture.*while.*surface gate/,
    },
    {
      mutate(evidence) {
        evidence.runtime_events.find(
          (event) => event.event === "surface_inference_resumed",
        ).at_ms = 8_999;
      },
      expected: /resumed before the actual runtime terminal/,
    },
    {
      mutate(evidence) {
        evidence.surface_texture_gate_windows[0].pre_request_acquisition = null;
      },
      expected: /real pre-request surface acquisition/,
    },
    {
      mutate(evidence) {
        evidence.surface_texture_gate_violation_calls_overflow_end = 1;
      },
      expected: /compact surface gate evidence overflowed/,
    },
  ];
  for (const { mutate, expected } of cases) {
    const evidence = validTurboEvidence();
    mutate(evidence);
    const failures = validateRequestScopedSurfaceGate(evidence);
    assert.ok(failures.some((failure) => expected.test(failure)), failures.join("\n"));
  }
});

test("accepts an ordinary block-0 run without serialized boundary evidence", () => {
  const evidence = validTurboEvidence();
  evidence.qwen_block0_execution_mode = TURBO_QWEN_BLOCK0_ORDINARY_MODE;
  evidence.runtime_events.find((event) => event.event === "ready").block0_execution_mode =
    TURBO_QWEN_BLOCK0_ORDINARY_MODE;
  evidence.runtime_events = evidence.runtime_events.filter(
    (event) => event.event !== "packed_f16_qwen_block0_execution_diagnostics",
  );
  evidence.packed_f16_qwen_block0_execution_diagnostics = null;
  evidence.packed_f16_qwen_block0_post_sync_diagnostic.diagnostic.block0_execution_mode =
    TURBO_QWEN_BLOCK0_ORDINARY_MODE;
  evidence.runtime_events.find(
    (event) => event.event === "packed_f16_qwen_block0_post_sync_diagnostic",
  ).diagnostic.block0_execution_mode = TURBO_QWEN_BLOCK0_ORDINARY_MODE;
  evidence.packed_f16_qwen_pre_handoff_diagnostics.diagnostics.block_00_immediate_post_sync
    .block0_execution_mode = TURBO_QWEN_BLOCK0_ORDINARY_MODE;
  evidence.runtime_events.find(
    (event) => event.event === "packed_f16_qwen_pre_handoff_diagnostics",
  ).diagnostics.block_00_immediate_post_sync.block0_execution_mode =
    TURBO_QWEN_BLOCK0_ORDINARY_MODE;
  assert.deepEqual(validateTurbo1024ModelEvidence(evidence), []);

  const conflated = validTurboEvidence();
  conflated.qwen_block0_execution_mode = TURBO_QWEN_BLOCK0_ORDINARY_MODE;
  assert.ok(
    validateTurbo1024ModelEvidence(conflated).some((failure) =>
      /requested Qwen block-0 execution mode differs/.test(failure),
    ),
  );
});

test("accepts the exact packed-F16 pre-DMD input diagnostic contract", () => {
  const evidence = validTurboEvidence();
  assert.deepEqual(
    validatePackedF16PreDmdInputDiagnostics(
      evidence.packed_f16_pre_dmd_input_diagnostics,
      evidence.progress_events,
    ),
    [],
  );
});

test("accepts the exact request-scoped packed-F16 DMD-to-VAE handoff contract", () => {
  const evidence = validTurboEvidence();
  assert.deepEqual(
    validatePackedF16DmdVaeHandoff(
      evidence.packed_f16_dmd_vae_handoff,
      evidence.progress_events,
    ),
    [],
  );
});

test("rejects packed-F16 DMD-to-VAE handoff corruption and stale cache mutations", () => {
  const mutations = [
    ["digest", (event) => (event.report.after_sha256 = "8".repeat(64)), /digest/],
    [
      "stale cache",
      (event) => {
        event.report.packed_cache_after_cleanup.state = "ready";
        event.report.packed_cache_after_cleanup.cache_ready = true;
        event.report.packed_cache_after_cleanup.cached_bytes =
          TURBO_PACKED_F16_RESOURCE_PLAN.retained_packed_f16_denoiser_bytes;
      },
      /post-cleanup packed cache/,
    ],
    ["non-finite latent", (event) => (event.report.all_finite = false), /all_finite/],
    ["all-zero latent", (event) => (event.report.not_all_zero = false), /not_all_zero/],
    ["attempt drift", (event) => (event.report.preload_attempt_count = 2), /attempt/],
  ];
  for (const [label, mutate, expected] of mutations) {
    const evidence = validTurboEvidence();
    mutate(evidence.packed_f16_dmd_vae_handoff);
    const failures = validatePackedF16DmdVaeHandoff(
      evidence.packed_f16_dmd_vae_handoff,
      evidence.progress_events,
    );
    assert.ok(failures.some((failure) => expected.test(failure)), `${label}: ${failures}`);
  }

  const missing = validTurboEvidence();
  missing.runtime_events = missing.runtime_events.filter(
    (event) => event.event !== "packed_f16_dmd_vae_handoff",
  );
  assert.ok(
    validateTurbo1024ModelEvidence(missing).some((failure) =>
      /DMD-to-VAE handoff/.test(failure),
    ),
  );

  const outOfOrder = validTurboEvidence();
  outOfOrder.packed_f16_dmd_vae_handoff.at_ms = 8_900;
  outOfOrder.runtime_events.find(
    (event) => event.event === "packed_f16_dmd_vae_handoff",
  ).at_ms = 8_900;
  assert.ok(
    validateTurbo1024ModelEvidence(outOfOrder).some((failure) =>
      /between DMD completion and VAE start/.test(failure),
    ),
  );
});

test("accepts exact Qwen stage and instruction handoff diagnostics", () => {
  const evidence = validTurboEvidence();
  assert.deepEqual(
    validatePackedF16QwenHostEmbedding(
      evidence.packed_f16_qwen_host_embedding,
      evidence.progress_events,
    ),
    [],
  );
  assert.deepEqual(
    validatePackedF16QwenBlock0ExecutionDiagnostics(
      evidence.packed_f16_qwen_block0_execution_diagnostics,
      evidence.progress_events,
    ),
    [],
  );
  assert.deepEqual(
    validatePackedF16QwenBlock0PostSyncDiagnostic(
      evidence.packed_f16_qwen_block0_post_sync_diagnostic,
      evidence.progress_events,
    ),
    [],
  );
  assert.deepEqual(
    validatePackedF16QwenPreHandoffDiagnostics(
      evidence.packed_f16_qwen_pre_handoff_diagnostics,
      evidence.progress_events,
    ),
    [],
  );
  assert.deepEqual(
    validatePackedF16QwenPostHandoffDiagnostics(
      evidence.packed_f16_qwen_post_handoff_diagnostics,
      evidence.packed_f16_qwen_pre_handoff_diagnostics,
      evidence.progress_events,
    ),
    [],
  );
});

test("rejects incomplete, all-zero, or alias-drifted block-0 execution boundaries", () => {
  const missing = validTurboEvidence();
  missing.runtime_events = missing.runtime_events.filter(
    (event) => event.event !== "packed_f16_qwen_block0_execution_diagnostics",
  );
  missing.packed_f16_qwen_block0_execution_diagnostics = null;
  assert.ok(
    validateTurbo1024ModelEvidence(missing).some((failure) => /block-0 execution/.test(failure)),
  );

  const zero = validPackedF16QwenBlock0ExecutionDiagnostics();
  zero.diagnostics.boundaries[3].tensor.max_abs = 0;
  zero.diagnostics.boundaries[3].tensor.rms = 0;
  zero.diagnostics.boundaries[3].not_all_zero = false;
  zero.diagnostics.no_captured_tensor_all_zero = false;
  zero.diagnostics.complete = false;
  zero.diagnostics.first_failure_boundary = "input_norm";
  zero.diagnostics.failure_reason = "all-zero";
  assert.ok(
    validatePackedF16QwenBlock0ExecutionDiagnostics(zero, validTurboEvidence().progress_events)
      .some((failure) => /all-zero|incomplete or failed/.test(failure)),
  );

  const aliasDrift = validPackedF16QwenBlock0ExecutionDiagnostics();
  aliasDrift.diagnostics.boundaries[2].tensor.sha256 = "d".repeat(64);
  assert.ok(
    validatePackedF16QwenBlock0ExecutionDiagnostics(
      aliasDrift,
      validTurboEvidence().progress_events,
    ).some((failure) => /add-zero canary/.test(failure)),
  );
});

test("rejects missing, all-zero, or transfer-drifted Qwen handoff evidence", () => {
  const missingHostEmbedding = validTurboEvidence();
  missingHostEmbedding.runtime_events = missingHostEmbedding.runtime_events.filter(
    (event) => event.event !== "packed_f16_qwen_host_embedding",
  );
  missingHostEmbedding.packed_f16_qwen_host_embedding = null;
  assert.ok(
    validateTurbo1024ModelEvidence(missingHostEmbedding).some((failure) =>
      /host-embedding/.test(failure),
    ),
  );

  const hostDrift = validTurboEvidence();
  hostDrift.packed_f16_qwen_host_embedding.report.authenticated_object_bytes -= 1;
  hostDrift.packed_f16_qwen_host_embedding.report.device_f32_sha256 = "0".repeat(64);
  hostDrift.packed_f16_qwen_host_embedding.report.total_device_transfer_bytes -= 4;
  const hostDriftFailures = validatePackedF16QwenHostEmbedding(
    hostDrift.packed_f16_qwen_host_embedding,
    hostDrift.progress_events,
  );
  for (const pattern of [/authenticated_object_bytes/, /device_f32_sha256/, /total_device_transfer_bytes/]) {
    assert.ok(hostDriftFailures.some((failure) => pattern.test(failure)), String(pattern));
  }

  const missing = validTurboEvidence();
  missing.runtime_events = missing.runtime_events.filter(
    (event) => event.event !== "packed_f16_qwen_pre_handoff_diagnostics",
  );
  missing.packed_f16_qwen_pre_handoff_diagnostics = null;
  assert.ok(
    validateTurbo1024ModelEvidence(missing).some((failure) => /Qwen pre\/post handoff/.test(failure)),
  );

  const zero = validTurboEvidence();
  const embedding = zero.packed_f16_qwen_pre_handoff_diagnostics.diagnostics.stage_outputs[0];
  embedding.max_abs = 0;
  embedding.rms = 0;
  zero.packed_f16_qwen_pre_handoff_diagnostics.diagnostics.no_tensor_all_zero = false;
  zero.packed_f16_qwen_pre_handoff_diagnostics.diagnostics.first_all_zero_tensor =
    "qwen_embedding_output";
  const zeroFailures = validatePackedF16QwenPreHandoffDiagnostics(
    zero.packed_f16_qwen_pre_handoff_diagnostics,
    zero.progress_events,
  );
  assert.ok(zeroFailures.some((failure) => /all-zero/.test(failure)));

  const drift = validTurboEvidence();
  const handoff = drift.packed_f16_qwen_post_handoff_diagnostics.diagnostics.handoff;
  handoff.qwen_release_unused_memory_after_stage = true;
  handoff.qwen_text_block_load_synchronization_policy = "post-forward-sync-only";
  handoff.qwen_text_layer_submission_policy = "backend-default";
  handoff.device_to_host_readback_bytes -= 4;
  handoff.after_sha256 = "e".repeat(64);
  const driftFailures = validatePackedF16QwenPostHandoffDiagnostics(
    drift.packed_f16_qwen_post_handoff_diagnostics,
    drift.packed_f16_qwen_pre_handoff_diagnostics,
    drift.progress_events,
  );
  for (const pattern of [/policy or success provenance/, /transfer accounting/, /digest/]) {
    assert.ok(driftFailures.some((failure) => pattern.test(failure)), String(pattern));
  }
});

test("rejects late or pre-handoff-unbound Qwen block-0 post-sync evidence", () => {
  const late = validTurboEvidence();
  late.packed_f16_qwen_block0_post_sync_diagnostic.at_ms = 1_060;
  assert.ok(
    validatePackedF16QwenBlock0PostSyncDiagnostic(
      late.packed_f16_qwen_block0_post_sync_diagnostic,
      late.progress_events,
    ).some((failure) => /not bound inside/.test(failure)),
  );

  const unbound = validTurboEvidence();
  const runtime = unbound.runtime_events.find(
    (event) => event.event === "packed_f16_qwen_block0_post_sync_diagnostic",
  );
  runtime.diagnostic.tensor.sha256 = "e".repeat(64);
  unbound.packed_f16_qwen_block0_post_sync_diagnostic = structuredClone(runtime);
  assert.ok(
    validateTurbo1024ModelEvidence(unbound).some((failure) =>
      /JSON-bound to pre-handoff/.test(failure),
    ),
  );
});

test("rejects missing packed-F16 pre-DMD input diagnostics", () => {
  const evidence = validTurboEvidence();
  evidence.runtime_events = evidence.runtime_events.filter(
    (event) => event.event !== "packed_f16_pre_dmd_input_diagnostics",
  );
  evidence.packed_f16_pre_dmd_input_diagnostics = null;
  const failures = validateTurbo1024ModelEvidence(evidence);
  assert.ok(
    failures.some((failure) => /exactly one packed-F16 pre-DMD input diagnostic/.test(failure)),
  );
});

test("rejects non-finite, malformed, unbound, and policy-drifted pre-DMD diagnostics", () => {
  const evidence = validTurboEvidence();
  const event = evidence.packed_f16_pre_dmd_input_diagnostics;
  event.run_id = 99;
  event.at_ms = 3_500;
  event.diagnostics.scope = "unbound-diagnostic";
  event.diagnostics.policy.qwen_text_block_load_synchronization_policy =
    "post-forward-sync-only";
  event.diagnostics.policy.qwen_text_layer_submission_policy = "backend-default";
  event.diagnostics.policy.packed_qwen_instruction_handoff_policy = "none";
  event.diagnostics.policy.cleanup_completed = false;
  event.diagnostics.policy.post_cleanup_packed_cache.cached_stages -= 1;
  event.diagnostics.instruction.finite_element_count -= 1;
  event.diagnostics.instruction.all_finite = false;
  event.diagnostics.instruction.max_abs = null;
  event.diagnostics.instruction.sha256 = "invalid";
  event.diagnostics.initial_latent.unexpected = true;
  event.diagnostics.renoise.pop();
  event.diagnostics.all_inputs_finite = false;
  const failures = validatePackedF16PreDmdInputDiagnostics(
    event,
    evidence.progress_events,
  );
  for (const pattern of [
    /scope/,
    /instruction handoff policy/,
    /cleanup did not complete/,
    /cached_stages/,
    /every element finite/,
    /max_abs is null/,
    /SHA-256/,
    /fields differ from the frozen schema/,
    /exactly three renoise/,
    /aggregate input finiteness/,
    /run ID differs/,
    /before step 1/,
  ]) {
    assert.ok(failures.some((failure) => pattern.test(failure)), String(pattern));
  }
});

test("rejects wrong shape, profile, residency, and incomplete inference", () => {
  const evidence = validTurboEvidence();
  evidence.ui_contract.width = 512;
  evidence.output_ready.width = 512;
  evidence.output_ready.backend = "burn-webgpu/browser-high-vram-resident-dense-f32";
  evidence.output_ready.artifact_content_digest = "0".repeat(64);
  evidence.progress_events = evidence.progress_events.filter(
    (event) => event.event !== "run_completed",
  );
  evidence.canvas_png_changed_bytes = 0;
  evidence.native_gpu_attestation.observed_peak_aggregate_framebuffer_bytes =
    LOW_VRAM_DEVICE_CAP_BYTES;
  const failures = validateTurbo1024ModelEvidence(evidence);
  for (const pattern of [
    /default dimensions/,
    /output dimensions/,
    /output backend/,
    /digest/,
    /run_completed/,
    /canvas/,
    /framebuffer peak/,
  ]) {
    assert.ok(failures.some((failure) => pattern.test(failure)), String(pattern));
  }
});

test("rejects source-side interaction claims and an invalid Save PNG download", () => {
  const evidence = validTurboEvidence();
  evidence.interaction.mechanism = "ecs-direct-mutation";
  evidence.interaction.save_clicked_via_cdp = false;
  evidence.interaction.prompt_input.focus_event.prompt_focused = false;
  evidence.interaction.prompt_input.selection_clicks.push({ click_count: 2 });
  evidence.interaction.seed_input.focus_event.seed_focused = false;
  evidence.interaction.seed_input.selection_clicks[1].click_count = 2;
  evidence.downloaded_png.ihdr.width = 512;
  evidence.downloaded_png.browser_events.completed.params.guid = "other-guid";
  const failures = validateTurbo1024ModelEvidence(evidence);
  for (const pattern of [
    /CDP keyboard and mouse/,
    /Prompt click/,
    /attested-empty/,
    /Seed clicks/,
    /triple-click/,
    /Save PNG button/,
    /1024x1024/,
    /downloadProgress/i,
  ]) {
    assert.ok(failures.some((failure) => pattern.test(failure)), String(pattern));
  }
});

test("rejects drift from the exact Turbo packed-F16 resource, lifecycle, and traffic identity", () => {
  const evidence = validTurboEvidence();
  const plan = evidence.runtime_events.find(
    (event) => event.event === "packed_f16_resource_plan",
  );
  plan.denoiser_runtime_q8_scope = "turbo-main-core-ffn-gate-up-q8";
  plan.qwen_text_block_load_synchronization_policy = "post-forward-sync-only";
  plan.qwen_text_layer_submission_policy = "backend-default";
  plan.inserted_padding_elements -= 1;
  plan.max_materialized_stage_f32_bytes -= 1;
  plan.conservative_planned_device_bytes -= 1;
  plan.on_device_quantized_execution_claimed = true;
  const lifecycle = evidence.runtime_events.find(
    (event) => event.event === "packed_f16_denoiser_lifecycle",
  ).lifecycle;
  lifecycle.stage_materializations -= 1;
  lifecycle.dmd_artifact_traffic.object_reads = 1;
  const preload = evidence.packed_f16_denoiser_preload_traffic;
  preload.object_reads -= 1;
  preload.range_reads -= 1;
  const traffic = evidence.artifact_traffic;
  traffic.range_reads -= 1;
  traffic.cache_invalid_entries = 1;
  const failures = validateTurbo1024ModelEvidence(evidence);
  for (const pattern of [
    /qwen_text_block_load_synchronization_policy/,
    /inserted_padding_elements/,
    /max_materialized_stage_f32_bytes/,
    /conservative_planned_device_bytes/,
    /quantized execution/,
    /stage_materializations/,
    /DMD artifact traffic object_reads/,
    /packed-F16 denoiser preload artifact traffic object_reads/,
    /packed-F16 denoiser preload artifact traffic range_reads/,
    /Generate request artifact traffic range_reads/,
    /cache_invalid_entries/,
  ]) {
    assert.ok(failures.some((failure) => pattern.test(failure)), String(pattern));
  }
});

test("requires complete clean preload and per-request cache/network accounting", () => {
  const evidence = validTurboEvidence();
  assert.deepEqual(Object.keys(evidence.artifact_traffic).sort(), [...ARTIFACT_TRAFFIC_FIELDS].sort());
  assert.deepEqual(
    Object.keys(evidence.packed_f16_denoiser_preload_traffic).sort(),
    [...ARTIFACT_TRAFFIC_FIELDS].sort(),
  );
  evidence.packed_f16_denoiser_preload_traffic.network_requests -= 1;
  evidence.cdp_preload_network_traffic.response_body_bytes -= 1;
  evidence.artifact_traffic.network_requests -= 1;
  evidence.artifact_traffic.network_response_bytes -= 1;
  evidence.cdp_network_traffic.invalid_content_range_response_count = 1;
  const failures = validateTurbo1024ModelEvidence(evidence);
  assert.ok(failures.some((failure) => /packed-F16 denoiser preload artifact traffic network_requests/.test(failure)));
  assert.ok(failures.some((failure) => /packed-F16 denoiser preload CDP exact Content-Length/.test(failure)));
  assert.ok(failures.some((failure) => /Generate request artifact traffic network_requests/.test(failure)));
  assert.ok(failures.some((failure) => /Generate request artifact traffic network_response_bytes/.test(failure)));
  assert.ok(failures.some((failure) => /invalid_content_range_response_count/.test(failure)));
});

test("keeps packed-F16 preload outside the Generate request traffic window", () => {
  const evidence = validTurboEvidence();
  evidence.runtime_events.find(
    (event) => event.event === "packed_f16_denoiser_preload",
  ).at_ms = 1_100;
  evidence.runtime_events.find((event) => event.event === "artifact_traffic").at_ms = 900;
  const failures = validateTurbo1024ModelEvidence(evidence);
  assert.ok(failures.some((failure) => /preload did not complete before/.test(failure)));
  assert.ok(failures.some((failure) => /artifact traffic is not contained/.test(failure)));
});

test("attributes only exact modular complete-part responses inside the Bevy request window", () => {
  const timeOrigin = 1_000_000;
  const bases = [
    "http://127.0.0.1:39001/model/boogu-image-0.1-turbo",
    "http://127.0.0.1:39001/model/qwen3-vl-8b-base-boogu-image-0.1",
    "http://127.0.0.1:39001/model/flux1-vae-boogu-image-0.1",
  ];
  const snapshot = {
    time_origin_epoch_ms: timeOrigin,
    progress_events: [
      { event: "run_started", model: TURBO_MODEL_ID, at_ms: 1_000 },
      { event: "run_completed", at_ms: 8_000 },
    ],
    output_events: [{ event: "ready", model: TURBO_MODEL_ID, at_ms: 9_000 }],
  };
  const response = (atMs, url, start, end, total, status) => {
    const physicalPart = url.includes("/transport/") && url.endsWith(".part");
    const actualStatus = status ?? (physicalPart ? 200 : 206);
    return {
      at_ms: atMs,
      method: "Network.responseReceived",
      params: {
        response: {
          url,
          status: actualStatus,
          headers: {
            "Content-Length": String(end - start + 1),
            ...(actualStatus === 206
              ? { "Content-Range": `bytes ${start}-${end}/${total}` }
              : {}),
          },
        },
      },
    };
  };
  const summary = summarizeTurboRequestCdpNetwork(
    [
      response(timeOrigin + 500, `${bases[0]}/before.bpk`, 0, 3, 8),
      response(timeOrigin + 2_000, `${bases[0]}/transport/a.part`, 0, 3, 8),
      response(timeOrigin + 3_000, `${bases[1]}/transport/b.part`, 4, 7, 8),
      response(timeOrigin + 4_000, `${bases[2]}-lookalike/ignored.bpk`, 0, 3, 8),
      response(timeOrigin + 9_500, `${bases[2]}/after.bpk`, 0, 3, 8),
    ],
    snapshot,
    bases,
    new Map([
      [
        "/model/boogu-image-0.1-turbo/transport/a.part",
        {
          bundle: "boogu-image-0.1-turbo",
          component: "shared:boogu-block-0+boogu-block-1",
          components: ["boogu-block-0", "boogu-block-1"],
          logical_paths: ["objects/a.bpk", "objects/b.bpk"],
          shared_physical_part: true,
        },
      ],
      [
        "/model/qwen3-vl-8b-base-boogu-image-0.1/transport/b.part",
        {
          bundle: "qwen3-vl-8b-base-boogu-image-0.1",
          component: "qwen-text",
          components: ["qwen-text"],
          logical_paths: ["objects/c.bpk"],
          shared_physical_part: false,
        },
      ],
    ]),
  );
  assert.equal(summary.model_response_count, 2);
  assert.equal(summary.http_200_complete_part_response_count, 2);
  assert.equal(summary.http_206_response_count, 0);
  assert.equal(summary.complete_object_validated_response_count, 2);
  assert.equal(summary.content_range_validated_response_count, 0);
  assert.equal(summary.response_body_bytes, 8);
  assert.equal(summary.terminal_event, "ready");
  assert.deepEqual(summary.model_base_urls, [...bases].sort());
  assert.equal(summary.physical_transport_response_count, 2);
  assert.equal(summary.physical_transport_response_bytes, 8);
  assert.equal(summary.unmapped_physical_transport_response_count, 0);
  assert.deepEqual(
    summary.logical_component_traffic[
      "boogu-image-0.1-turbo/shared:boogu-block-0+boogu-block-1"
    ].components,
    ["boogu-block-0", "boogu-block-1"],
  );
});

test("attributes only exact modular 206 responses inside the denoiser preload window", () => {
  const timeOrigin = 2_000_000;
  const base = "http://127.0.0.1:39001/model/boogu-image-0.1-turbo";
  const snapshot = {
    time_origin_epoch_ms: timeOrigin,
    runtime_events: [
      { event: "preparing", message: TURBO_PACKED_F16_PRELOAD_MESSAGE, at_ms: 1_000 },
      { event: "packed_f16_denoiser_preload", at_ms: 5_000 },
      { event: "ready", model: TURBO_MODEL_ID, at_ms: 6_000 },
    ],
  };
  const response = (atMs, url, start, end, total) => ({
    at_ms: atMs,
    method: "Network.responseReceived",
    params: {
      response: {
        url,
        status: 206,
        headers: {
          "Content-Length": String(end - start + 1),
          "Content-Range": `bytes ${start}-${end}/${total}`,
        },
      },
    },
  });
  const summary = summarizeTurboPreloadCdpNetwork(
    [
      response(timeOrigin + 500, `${base}/before.bpk`, 0, 3, 8),
      response(timeOrigin + 2_000, `${base}/preload-0.bpk`, 0, 3, 8),
      response(timeOrigin + 4_000, `${base}/preload-1.bpk`, 4, 7, 8),
      response(timeOrigin + 5_500, `${base}/after.bpk`, 0, 3, 8),
    ],
    snapshot,
    [base],
  );
  assert.equal(summary.policy, TURBO_CDP_PRELOAD_NETWORK_POLICY);
  assert.equal(summary.model_response_count, 2);
  assert.equal(summary.content_range_validated_response_count, 2);
  assert.equal(summary.response_body_bytes, 8);
  assert.equal(summary.terminal_event, "packed_f16_denoiser_preload");
});

test("CDP request-window evidence exposes malformed ranges and failure terminals", () => {
  const base = "http://127.0.0.1:39001/model/boogu-image-0.1-turbo";
  const summary = summarizeTurboRequestCdpNetwork(
    [
      {
        at_ms: 1_002_000,
        method: "Network.responseReceived",
        params: {
          response: {
            url: `${base}/bad.bpk`,
            status: 206,
            headers: { "Content-Length": "4", "Content-Range": "bytes 0-2/8" },
          },
        },
      },
    ],
    {
      time_origin_epoch_ms: 1_000_000,
      progress_events: [
        { event: "run_started", model: TURBO_MODEL_ID, at_ms: 1_000 },
        { event: "run_failed", at_ms: 3_000 },
      ],
      output_events: [],
    },
    [base],
  );
  assert.equal(summary.terminal_event, "run_failed");
  assert.equal(summary.invalid_content_range_response_count, 1);
  assert.equal(summary.content_range_validated_response_count, 0);
});

function validMultiRequestEvidence() {
  const base = validTurboEvidence();
  const packageIdentity = validTestedPackageIdentity();
  const servedTransport = validEvidence().served_transport;
  const initialRuntimeEvents = base.runtime_events.filter(
    (event) =>
      ![
        "packed_f16_qwen_host_embedding",
        "packed_f16_qwen_block0_execution_diagnostics",
        "packed_f16_qwen_block0_post_sync_diagnostic",
        "packed_f16_qwen_pre_handoff_diagnostics",
        "packed_f16_qwen_post_handoff_diagnostics",
        "packed_f16_pre_dmd_input_diagnostics",
        "packed_f16_denoiser_lifecycle",
        "packed_f16_dmd_vae_handoff",
        "artifact_traffic",
        "surface_inference_suspended",
        "surface_inference_resumed",
      ].includes(event.event),
  );
  const firstRuntimeEvents = base.runtime_events.filter((event) =>
    [
      "packed_f16_qwen_host_embedding",
      "packed_f16_qwen_block0_execution_diagnostics",
      "packed_f16_qwen_block0_post_sync_diagnostic",
      "packed_f16_qwen_pre_handoff_diagnostics",
      "packed_f16_qwen_post_handoff_diagnostics",
      "packed_f16_pre_dmd_input_diagnostics",
      "packed_f16_denoiser_lifecycle",
      "packed_f16_dmd_vae_handoff",
      "artifact_traffic",
      "surface_inference_suspended",
      "surface_inference_resumed",
    ].includes(event.event),
  );
  const withTimesAndRun = (events, runId, offset) =>
    events.map((event) => ({
      ...event,
      run_id: runId,
      at_ms: event.at_ms + offset,
    }));
  const firstProgressEvents = withTimesAndRun(base.progress_events, 7, 0);
  const secondProgressEvents = withTimesAndRun(base.progress_events, 8, 10_000);
  const modelBaseUrls = base.cdp_network_traffic.model_base_urls;
  const cdpNetwork = (traffic, start, end) => ({
    policy: TURBO_CDP_REQUEST_NETWORK_POLICY,
    model_base_urls: [...modelBaseUrls],
    window_start_epoch_ms: start,
    window_end_epoch_ms: end,
    terminal_event: "ready",
    model_response_count: traffic.network_requests,
    http_200_complete_part_response_count: traffic.network_requests,
    http_206_response_count: 0,
    complete_object_validated_response_count: traffic.network_requests,
    content_range_validated_response_count: 0,
    response_body_bytes: traffic.network_response_bytes,
    unexpected_status_response_count: 0,
    missing_content_length_count: 0,
    invalid_content_range_response_count: 0,
  });
  const dmdNetwork = (start, end) => ({
    policy: TURBO_CDP_DMD_NETWORK_POLICY,
    model_base_urls: [...modelBaseUrls],
    window_start_epoch_ms: start,
    window_end_epoch_ms: end,
    terminal_event: "stage_completed:dmd",
    model_response_count: 0,
    http_200_complete_part_response_count: 0,
    http_206_response_count: 0,
    complete_object_validated_response_count: 0,
    content_range_validated_response_count: 0,
    response_body_bytes: 0,
    unexpected_status_response_count: 0,
    missing_content_length_count: 0,
    invalid_content_range_response_count: 0,
  });
  const gpuWindow = (start, end) => ({
    provider: "nvidia-smi",
    interval_aggregation_policy: GPU_INTERVAL_AGGREGATION_POLICY,
    maximum_framebuffer_bytes_exclusive: LOW_VRAM_DEVICE_CAP_BYTES,
    window_start_epoch_ms: start,
    window_end_epoch_ms: end,
    sample_records: [
      {
        at_ms: start + 500,
        matched_rows: [{ gpu_index: 0, pid: 1001, framebuffer_mib: 20_000 }],
        aggregate_framebuffer_mib: 20_000,
        aggregate_sm_percent: 50,
      },
    ],
    matched_sample_intervals: 1,
    active_sample_intervals: 1,
    observed_peak_aggregate_framebuffer_bytes: 20_000 * 1024 * 1024,
    observed_gpu_processes: [{ pid: 1001, command: "chrome --type=gpu-process" }],
    validation_failures: [],
    validated: true,
  });
  const dmdAttestation = (runId, lifecycleEventAtMs) => ({
    policy: TURBO_DMD_RUNTIME_ZERO_IO_POLICY,
    run_id: runId,
    completed_dmd_steps: 4,
    traffic: { ...TURBO_DMD_ZERO_IO },
    lifecycle_event_at_ms: lifecycleEventAtMs,
    runtime_source_sha256: packageIdentity.sources.browser_runtime.sha256,
  });
  const firstWindow = { start_epoch_ms: 1_000_001_000, end_epoch_ms: 1_000_009_500 };
  const secondWindow = { start_epoch_ms: 1_000_011_000, end_epoch_ms: 1_000_019_500 };
  const commonPageIdentity = {
    engine_session_id: "3c6eeea1-a432-4d50-92cc-f39f867d1941",
    url: "http://127.0.0.1:39001/index.html?rendered-model-smoke=1",
    time_origin_epoch_ms: 1_000_000_000,
  };
  const firstOutput = { ...base.output_ready, at_ms: 9_020 };
  const first = {
    request_ordinal: 1,
    qwen_block0_execution_mode: TURBO_QWEN_BLOCK0_SERIALIZED_DIAGNOSTIC_MODE,
    page_identity: { ...commonPageIdentity },
    request_epoch_window: firstWindow,
    event_boundaries: {
      runtime_start_index: 4,
      runtime_end_index: 15,
      progress_start_index: 0,
      progress_end_index: 12,
      output_start_index: 0,
      output_end_index: 1,
      ui_start_index: 0,
      ui_end_index: 1,
      surface_start_index: 0,
      surface_end_index: 1,
      cdp_start_index: 100,
      cdp_end_index: 4_000,
    },
    cdp_event_count: 3_900,
    runtime_events: firstRuntimeEvents,
    progress_events: firstProgressEvents,
    output_events: [firstOutput],
    ui_events: [base.ui_contract],
    surface_texture_gate_windows: [validSurfaceTextureGateWindow(7)],
    surface_texture_gate_windows_overflow: 0,
    surface_texture_gate_violation_calls: [],
    surface_texture_gate_violation_calls_overflow_start: 0,
    surface_texture_gate_violation_calls_overflow_end: 0,
    surface_texture_gate_overlap_count: 0,
    surface_texture_acquisition_count_start: 30,
    surface_texture_acquisition_count_end: 45,
    surface_texture_acquisition_failure_count_start: 0,
    surface_texture_acquisition_failure_count_end: 0,
    active_surface_gate_after_request: null,
    surface_inference_state_after_request: null,
    artifact_traffic: { ...TURBO_GENERATE_REQUEST_TRAFFIC },
    packed_f16_denoiser_lifecycle: validPackedF16Lifecycle(1),
    packed_f16_dmd_vae_handoff: structuredClone(base.packed_f16_dmd_vae_handoff),
    packed_f16_qwen_host_embedding: structuredClone(base.packed_f16_qwen_host_embedding),
    packed_f16_qwen_block0_execution_diagnostics: structuredClone(
      base.packed_f16_qwen_block0_execution_diagnostics,
    ),
    packed_f16_qwen_block0_post_sync_diagnostic: structuredClone(
      base.runtime_events.find(
        (event) => event.event === "packed_f16_qwen_block0_post_sync_diagnostic",
      ),
    ),
    packed_f16_qwen_pre_handoff_diagnostics: structuredClone(
      base.packed_f16_qwen_pre_handoff_diagnostics,
    ),
    packed_f16_qwen_post_handoff_diagnostics: structuredClone(
      base.packed_f16_qwen_post_handoff_diagnostics,
    ),
    packed_f16_pre_dmd_input_diagnostics: structuredClone(
      base.packed_f16_pre_dmd_input_diagnostics,
    ),
    cdp_network_traffic: cdpNetwork(
      TURBO_GENERATE_REQUEST_TRAFFIC,
      1_000_001_100,
      1_000_009_000,
    ),
    cdp_dmd_network_traffic: dmdNetwork(1_000_001_200, 1_000_001_700),
    dmd_runtime_io_attestation: dmdAttestation(7, 8_800),
    modular_artifact_transport: structuredClone(base.modular_artifact_transport),
    served_transport: structuredClone(servedTransport),
    tested_package_identity: structuredClone(packageIdentity),
    ui_contract: structuredClone(base.ui_contract),
    output_ready: structuredClone(firstOutput),
    interaction: structuredClone(base.interaction),
    downloaded_png: structuredClone(base.downloaded_png),
    canvas_png_changed_bytes: 100,
    model_screenshot_bytes: 4_096,
    native_gpu_attestation: gpuWindow(firstWindow.start_epoch_ms, firstWindow.end_epoch_ms),
  };
  const warmTraffic = { ...TURBO_REPEAT_GENERATE_REQUEST_TRAFFIC };
  const secondPng = structuredClone(base.downloaded_png);
  secondPng.path = "/tmp/burn-image/downloads/burn-image-8.png";
  secondPng.file_name = "burn-image-8.png";
  secondPng.sha256 = "c".repeat(64);
  secondPng.browser_events.will_begin.params.guid = "download-guid-2";
  secondPng.browser_events.will_begin.params.suggestedFilename = secondPng.file_name;
  secondPng.browser_events.completed.params.guid = "download-guid-2";
  const secondOutput = { ...base.output_ready, job_id: "8", at_ms: 19_020 };
  const secondSeedChanged = { at_ms: 10_000, event: "seed_changed", value: "1" };
  const second = {
    request_ordinal: 2,
    qwen_block0_execution_mode: TURBO_QWEN_BLOCK0_SERIALIZED_DIAGNOSTIC_MODE,
    page_identity: { ...commonPageIdentity },
    request_epoch_window: secondWindow,
    event_boundaries: {
      runtime_start_index: 15,
      runtime_end_index: 28,
      progress_start_index: 12,
      progress_end_index: 24,
      output_start_index: 1,
      output_end_index: 2,
      ui_start_index: 1,
      ui_end_index: 2,
      surface_start_index: 1,
      surface_end_index: 2,
      cdp_start_index: 4_000,
      cdp_end_index: 4_500,
    },
    cdp_event_count: 500,
    runtime_events: [
      {
        at_ms: 10_010,
        event: "surface_inference_suspended",
        run_id: 8,
        policy: SURFACE_INFERENCE_POLICY,
        primary_window_camera_count: 2,
        saved_camera_state_count: 2,
        previously_active_camera_count: 2,
        inactive_camera_count: 2,
        active_job_count: 1,
        suspended_before_runtime_submit: true,
        all_primary_window_cameras_inactive: true,
      },
      {
        at_ms: 10_015,
        event: "preparing",
        message: TURBO_PACKED_F16_PRELOAD_MESSAGE,
      },
      {
        at_ms: 10_020,
        event: "packed_f16_denoiser_preload",
        traffic: { ...TURBO_DENOISER_REHYDRATION_TRAFFIC },
        cached_stages: TURBO_PACKED_F16_CACHED_STAGES,
        cached_objects: TURBO_PACKED_F16_CACHED_OBJECTS,
        cached_tensors: TURBO_PACKED_F16_CACHED_TENSORS,
        cached_bytes: TURBO_PACKED_F16_RESOURCE_PLAN.retained_packed_f16_denoiser_bytes,
        previous_preload_attempt_count: 1,
        preload_attempt_count: 2,
        request_scoped_rehydration: true,
        rehydration_policy: TURBO_PACKED_F16_NEXT_REQUEST_REHYDRATION_POLICY,
      },
      {
        ...validPackedF16QwenBlock0ExecutionDiagnostics(8),
        at_ms: 11_040,
      },
      {
        ...validPackedF16QwenBlock0PostSyncDiagnostic(8),
        at_ms: 11_045,
      },
      {
        ...validPackedF16QwenHostEmbedding(8),
        at_ms: 11_050,
      },
      {
        ...validPackedF16QwenPreHandoffDiagnostics(8),
        at_ms: 11_070,
      },
      {
        ...validPackedF16QwenPostHandoffDiagnostics(8),
        at_ms: 11_080,
      },
      {
        ...validPackedF16PreDmdInputDiagnostics(8),
        at_ms: 12_000,
      },
      {
        at_ms: 18_800,
        event: "packed_f16_denoiser_lifecycle",
        lifecycle: validPackedF16Lifecycle(2),
      },
      validPackedF16DmdVaeHandoff(8, 2, 18_830),
      { at_ms: 18_900, event: "artifact_traffic", traffic: warmTraffic },
      {
        at_ms: 19_010,
        event: "surface_inference_resumed",
        run_id: 8,
        policy: SURFACE_INFERENCE_POLICY,
        terminal: "completed",
        primary_window_camera_count: 2,
        saved_camera_state_count: 2,
        restored_camera_state_count: 2,
        restored_active_camera_count: 2,
        active_job_count: 0,
        resumed_after_runtime_terminal: true,
        resumed_before_output_ready: true,
        exact_saved_states_restored: true,
        all_primary_window_cameras_restored: true,
      },
    ],
    progress_events: secondProgressEvents,
    output_events: [secondOutput],
    ui_events: [secondSeedChanged],
    surface_texture_gate_windows: [
      validSurfaceTextureGateWindow(8, {
        suspendedAt: 10_010,
        resumedAt: 19_010,
        preRequestAt: 10_000,
        postResumeAt: 19_030,
        acquisitionCountAtGate: 50,
      }),
    ],
    surface_texture_gate_windows_overflow: 0,
    surface_texture_gate_violation_calls: [],
    surface_texture_gate_violation_calls_overflow_start: 0,
    surface_texture_gate_violation_calls_overflow_end: 0,
    surface_texture_gate_overlap_count: 0,
    surface_texture_acquisition_count_start: 45,
    surface_texture_acquisition_count_end: 55,
    surface_texture_acquisition_failure_count_start: 0,
    surface_texture_acquisition_failure_count_end: 0,
    active_surface_gate_after_request: null,
    surface_inference_state_after_request: null,
    artifact_traffic: warmTraffic,
    packed_f16_denoiser_lifecycle: validPackedF16Lifecycle(2),
    packed_f16_dmd_vae_handoff: validPackedF16DmdVaeHandoff(8, 2, 18_830),
    packed_f16_qwen_host_embedding: {
      ...validPackedF16QwenHostEmbedding(8),
      at_ms: 11_050,
    },
    packed_f16_qwen_block0_execution_diagnostics: {
      ...validPackedF16QwenBlock0ExecutionDiagnostics(8),
      at_ms: 11_040,
    },
    packed_f16_qwen_block0_post_sync_diagnostic: {
      ...validPackedF16QwenBlock0PostSyncDiagnostic(8),
      at_ms: 11_045,
    },
    packed_f16_qwen_pre_handoff_diagnostics: {
      ...validPackedF16QwenPreHandoffDiagnostics(8),
      at_ms: 11_070,
    },
    packed_f16_qwen_post_handoff_diagnostics: {
      ...validPackedF16QwenPostHandoffDiagnostics(8),
      at_ms: 11_080,
    },
    packed_f16_pre_dmd_input_diagnostics: {
      ...validPackedF16PreDmdInputDiagnostics(8),
      at_ms: 12_000,
    },
    cdp_network_traffic: cdpNetwork(warmTraffic, 1_000_011_100, 1_000_019_000),
    cdp_dmd_network_traffic: dmdNetwork(1_000_011_200, 1_000_011_700),
    dmd_runtime_io_attestation: dmdAttestation(8, 18_800),
    modular_artifact_transport: structuredClone(base.modular_artifact_transport),
    served_transport: structuredClone(servedTransport),
    tested_package_identity: structuredClone(packageIdentity),
    ui_contract: structuredClone(base.ui_contract),
    output_ready: secondOutput,
    interaction: {
      mechanism: "cdp-keyboard-and-mouse",
      prompt_reused_from_same_engine: true,
      prompt_value: base.fixed_ascii_prompt,
      seed_typed_via_cdp: true,
      seed_value: "1",
      seed_event: secondSeedChanged,
      seed_input: structuredClone(base.interaction.seed_input),
      run_readiness: {
        policy: TURBO_SECOND_REQUEST_RUN_READY_POLICY,
        source: "exact-last-pre-boundary-post-request-ready",
        ui_start_index: 1,
        ui_event_count_at_resolution: 2,
        seed_changed_event_index: 1,
        run_ready_event_index: 0,
        duplicate_ready_after_seed_change_required: false,
        fallback_exact_last_pre_boundary_ready: true,
        selected_ui_contract: structuredClone(base.ui_contract),
      },
      run_clicked_via_cdp: true,
      save_clicked_via_cdp: true,
    },
    downloaded_png: secondPng,
    canvas_png_changed_bytes: 200,
    model_screenshot_bytes: 4_096,
    native_gpu_attestation: gpuWindow(secondWindow.start_epoch_ms, secondWindow.end_epoch_ms),
  };
  return {
    policy: TURBO_MULTI_REQUEST_POLICY,
    request_count: 2,
    engine_session_id: commonPageIdentity.engine_session_id,
    page_url: commonPageIdentity.url,
    time_origin_epoch_ms: commonPageIdentity.time_origin_epoch_ms,
    fixed_ascii_prompt: base.fixed_ascii_prompt,
    qwen_block0_execution_mode: TURBO_QWEN_BLOCK0_SERIALIZED_DIAGNOSTIC_MODE,
    bevy_backend_ready: structuredClone(base.bevy_backend_ready),
    runtime_webgpu_calls: structuredClone(base.runtime_webgpu_calls),
    runtime_webgpu_dropped_calls: base.runtime_webgpu_dropped_calls,
    runtime_webgpu_adapter_attestation: structuredClone(
      base.runtime_webgpu_adapter_attestation,
    ),
    initial_runtime_events: initialRuntimeEvents,
    initial_packed_f16_denoiser_preload_traffic: { ...TURBO_DENOISER_PRELOAD_TRAFFIC },
    cdp_preload_network_traffic: structuredClone(base.cdp_preload_network_traffic),
    request_scoped_denoiser_policy: {
      policy: TURBO_DENOISER_STORAGE_POLICY,
      expected_stages: TURBO_PACKED_F16_CACHED_STAGES,
      expected_objects: TURBO_PACKED_F16_CACHED_OBJECTS,
      expected_tensors: TURBO_PACKED_F16_CACHED_TENSORS,
      expected_bytes: TURBO_PACKED_F16_RESOURCE_PLAN.retained_packed_f16_denoiser_bytes,
      initial_preload_stages: TURBO_PACKED_F16_CACHED_STAGES,
      initial_preload_objects: TURBO_PACKED_F16_CACHED_OBJECTS,
      initial_preload_tensors: TURBO_PACKED_F16_CACHED_TENSORS,
      initial_preload_bytes: TURBO_PACKED_F16_RESOURCE_PLAN.retained_packed_f16_denoiser_bytes,
      request_local_preload_events: 1,
      successful_requests_with_post_dmd_eviction: 2,
      preload_attempt_counts: [1, 2],
      raw_packed_cache_empty_before_vae: true,
      repeat_rehydration_cache_only: true,
    },
    tested_package_identity: structuredClone(packageIdentity),
    native_gpu_attestation: structuredClone(base.native_gpu_attestation),
    requests: [first, second],
  };
}

test("accepts same-engine two-request evidence without a duplicate pre-Run ready event", () => {
  const evidence = validMultiRequestEvidence();
  assert.equal(
    evidence.requests[0].runtime_events.filter((event) => event.event === "preparing").length,
    0,
  );
  assert.equal(
    evidence.requests[1].runtime_events.filter((event) => event.event === "preparing").length,
    1,
  );
  assert.equal(
    evidence.requests[1].ui_events.some((event) => event.event === "ready"),
    false,
  );
  assert.deepEqual(validateTurbo1024MultiRequestEvidence(evidence), []);
});

test("requires exact request-local packed-F16 preparation cardinality, message, and ordering", () => {
  const missing = validMultiRequestEvidence();
  missing.requests[1].runtime_events = missing.requests[1].runtime_events.filter(
    (event) => event.event !== "preparing",
  );
  missing.requests[1].event_boundaries.runtime_end_index -= 1;
  let failures = validateTurbo1024MultiRequestEvidence(missing);
  assert.ok(
    failures.some(
      (failure) =>
        /request 2 emitted 0 request-local packed-F16 preparing event\(s\), expected 1/.test(
          failure,
        ),
    ),
    failures.join("\n"),
  );

  const duplicate = validMultiRequestEvidence();
  const duplicateSecond = duplicate.requests[1];
  const duplicatePreparingIndex = duplicateSecond.runtime_events.findIndex(
    (event) => event.event === "preparing",
  );
  duplicateSecond.runtime_events.splice(
    duplicatePreparingIndex + 1,
    0,
    structuredClone(duplicateSecond.runtime_events[duplicatePreparingIndex]),
  );
  duplicateSecond.event_boundaries.runtime_end_index += 1;
  failures = validateTurbo1024MultiRequestEvidence(duplicate);
  assert.ok(
    failures.some(
      (failure) =>
        /request 2 emitted 2 request-local packed-F16 preparing event\(s\), expected 1/.test(
          failure,
        ),
    ),
    failures.join("\n"),
  );

  const wrongMessage = validMultiRequestEvidence();
  wrongMessage.requests[1].runtime_events.find(
    (event) => event.event === "preparing",
  ).message = "Preparing a different denoiser";
  failures = validateTurbo1024MultiRequestEvidence(wrongMessage);
  assert.ok(
    failures.some(
      (failure) => /request 2 packed-F16 preparing message=.*expected/.test(failure),
    ),
    failures.join("\n"),
  );

  const misordered = validMultiRequestEvidence();
  const misorderedEvents = misordered.requests[1].runtime_events;
  const preparingIndex = misorderedEvents.findIndex((event) => event.event === "preparing");
  const preloadIndex = misorderedEvents.findIndex(
    (event) => event.event === "packed_f16_denoiser_preload",
  );
  [misorderedEvents[preparingIndex], misorderedEvents[preloadIndex]] = [
    misorderedEvents[preloadIndex],
    misorderedEvents[preparingIndex],
  ];
  failures = validateTurbo1024MultiRequestEvidence(misordered);
  assert.ok(
    failures.some(
      (failure) =>
        /request 2 packed-F16 preparing event is not ordered before its packed-F16 preload event/.test(
          failure,
        ),
    ),
    failures.join("\n"),
  );

  const unexpectedFirst = validMultiRequestEvidence();
  unexpectedFirst.requests[0].runtime_events.unshift({
    at_ms: 949,
    event: "preparing",
    message: TURBO_PACKED_F16_PRELOAD_MESSAGE,
  });
  unexpectedFirst.requests[0].event_boundaries.runtime_end_index += 1;
  unexpectedFirst.requests[1].event_boundaries.runtime_start_index += 1;
  unexpectedFirst.requests[1].event_boundaries.runtime_end_index += 1;
  failures = validateTurbo1024MultiRequestEvidence(unexpectedFirst);
  assert.ok(
    failures.some(
      (failure) =>
        /request 1 emitted 1 request-local packed-F16 preparing event\(s\), expected 0/.test(
          failure,
        ),
    ),
    failures.join("\n"),
  );
});

test("rejects stale, disabled, or wrong multi-request Run readiness evidence", () => {
  const stale = validMultiRequestEvidence();
  stale.requests[1].interaction.run_readiness.run_ready_event_index = 1;
  let failures = validateTurbo1024MultiRequestEvidence(stale);
  assert.ok(
    failures.some((failure) => /exact last pre-boundary post-request ready/.test(failure)),
    failures.join("\n"),
  );

  const disabled = validMultiRequestEvidence();
  disabled.requests[1].interaction.run_readiness.selected_ui_contract.run_enabled = false;
  failures = validateTurbo1024MultiRequestEvidence(disabled);
  assert.ok(
    failures.some((failure) => /selected Run readiness Run control is disabled/.test(failure)),
    failures.join("\n"),
  );

  const wrong = validMultiRequestEvidence();
  wrong.requests[1].interaction.run_readiness.selected_ui_contract.model = "wrong/model";
  failures = validateTurbo1024MultiRequestEvidence(wrong);
  assert.ok(
    failures.some((failure) => /selected Run readiness model=.*expected exact Turbo/.test(failure)),
    failures.join("\n"),
  );
});

test("binds each canonical multi-request output job ID to its numeric run ID", () => {
  const evidence = validMultiRequestEvidence();
  evidence.requests[1].output_ready.job_id = "9";
  evidence.requests[1].output_events[0].job_id = "9";
  const failures = validateTurbo1024MultiRequestEvidence(evidence);
  assert.ok(
    failures.some(
      (failure) =>
        /request 2/.test(failure) && /exact decimal representation of its run ID/.test(failure),
    ),
    failures.join("\n"),
  );
  assert.ok(
    !failures.some((failure) => /output-ready evidence differs from its recorded output event/.test(failure)),
    failures.join("\n"),
  );
});

test("requires one independent surface-gate pair for every same-engine request", () => {
  const missingSecondResume = validMultiRequestEvidence();
  missingSecondResume.requests[1].runtime_events =
    missingSecondResume.requests[1].runtime_events.filter(
      (event) => event.event !== "surface_inference_resumed",
    );
  missingSecondResume.requests[1].event_boundaries.runtime_end_index -= 1;
  let failures = validateTurbo1024MultiRequestEvidence(missingSecondResume);
  assert.ok(
    failures.some(
      (failure) =>
        /request 2/.test(failure) && /exactly one surface suspend\/resume pair/.test(failure),
    ),
    failures.join("\n"),
  );

  const acquiredDuringSecondRun = validMultiRequestEvidence();
  const secondWindow = acquiredDuringSecondRun.requests[1].surface_texture_gate_windows[0];
  secondWindow.acquisition_count_at_resume += 1;
  secondWindow.gated_call_count = 1;
  acquiredDuringSecondRun.requests[1].surface_texture_gate_violation_calls.push({
    run_id: 8,
    policy: SURFACE_INFERENCE_POLICY,
    call_index: secondWindow.acquisition_count_at_resume,
    at_ms: 15_000,
    canvas_id: "burn-image",
    succeeded: true,
  });
  failures = validateTurbo1024MultiRequestEvidence(acquiredDuringSecondRun);
  assert.ok(
    failures.some(
      (failure) => /request 2/.test(failure) && /getCurrentTexture/.test(failure),
    ),
    failures.join("\n"),
  );
});

test("rejects missing, network-backed, or attempt-drifted second-request rehydration", () => {
  const missing = validMultiRequestEvidence();
  missing.requests[1].runtime_events = missing.requests[1].runtime_events.filter(
    (event) => event.event !== "packed_f16_denoiser_preload",
  );
  missing.requests[1].event_boundaries.runtime_end_index -= 1;
  let failures = validateTurbo1024MultiRequestEvidence(missing);
  assert.ok(failures.some((failure) => /request-local packed-F16 preload/.test(failure)));

  const networkBacked = validMultiRequestEvidence();
  const networkPreload = networkBacked.requests[1].runtime_events.find(
    (event) => event.event === "packed_f16_denoiser_preload",
  );
  networkPreload.traffic.network_requests = 1;
  networkPreload.traffic.network_response_bytes = 1;
  failures = validateTurbo1024MultiRequestEvidence(networkBacked);
  assert.ok(
    failures.some((failure) => /rehydration .*network|rehydration .*traffic/.test(failure)),
    failures.join("\n"),
  );

  const attemptDrift = validMultiRequestEvidence();
  const attemptPreload = attemptDrift.requests[1].runtime_events.find(
    (event) => event.event === "packed_f16_denoiser_preload",
  );
  attemptPreload.preload_attempt_count = 1;
  failures = validateTurbo1024MultiRequestEvidence(attemptDrift);
  assert.ok(failures.some((failure) => /rehydration preload_attempt_count/.test(failure)));
});

test("rejects cross-request Qwen digest drift for the same prompt", () => {
  const evidence = validMultiRequestEvidence();
  const second = evidence.requests[1];
  const driftedDigest = "d".repeat(64);
  second.packed_f16_qwen_pre_handoff_diagnostics.diagnostics.stage_outputs[10].sha256 =
    driftedDigest;
  second.runtime_events.find(
    (event) => event.event === "packed_f16_qwen_pre_handoff_diagnostics",
  ).diagnostics.stage_outputs[10].sha256 = driftedDigest;
  const failures = validateTurbo1024MultiRequestEvidence(evidence);
  assert.ok(
    failures.some((failure) => /exact cross-request Qwen/.test(failure)),
    failures.join("\n"),
  );
});

test("fails multi-request proof on session drift or a second runtime device", () => {
  const sessionDrift = validMultiRequestEvidence();
  sessionDrift.engine_session_id = "ffffffff-ffff-4fff-8fff-ffffffffffff";
  let failures = validateTurbo1024MultiRequestEvidence(sessionDrift);
  assert.ok(
    failures.some((failure) => /attestation differs|same page\/runtime identity/.test(failure)),
    failures.join("\n"),
  );

  const secondDevice = validMultiRequestEvidence();
  secondDevice.runtime_webgpu_calls.push(
    {
      event: "request-device-start",
      at_ms: 31,
      detail: {
        request_id: 3,
        adapter_request_id: 1,
        requiredFeatures: [...BOOGU_WEB_REQUIRED_DEVICE_FEATURES],
        requiredLimits: {},
      },
    },
    {
      event: "request-device-resolved",
      at_ms: 32,
      detail: {
        request_id: 3,
        adapter_request_id: 1,
        requiredFeatures: [...BOOGU_WEB_REQUIRED_DEVICE_FEATURES],
        requiredLimits: {},
        enabledFeatures: expectedBrowserEnabledFeatures(
          BOOGU_WEB_REQUIRED_DEVICE_FEATURES,
        ),
      },
    },
  );
  failures = validateTurbo1024MultiRequestEvidence(secondDevice);
  assert.ok(
    failures.some((failure) => /device requests=2|attestation differs/.test(failure)),
    failures.join("\n"),
  );
});

test("fails closed when repeat rehydration is duplicated or misses persistent cache", () => {
  const evidence = validMultiRequestEvidence();
  const second = evidence.requests[1];
  second.runtime_events.unshift({
    event: "packed_f16_denoiser_preload",
    cached_stages: TURBO_PACKED_F16_CACHED_STAGES,
  });
  second.event_boundaries.runtime_end_index += 1;
  second.runtime_events.find((event) => event.event === "artifact_traffic").traffic.cache_hits -= 1;
  second.runtime_events.find((event) => event.event === "artifact_traffic").traffic.cache_misses = 1;
  second.runtime_events.find((event) => event.event === "artifact_traffic").traffic.network_requests = 1;
  second.cdp_network_traffic.model_response_count = 1;
  const failures = validateTurbo1024MultiRequestEvidence(evidence);
  assert.ok(failures.some((failure) => /request-local packed-F16 preload/.test(failure)));
  assert.ok(failures.some((failure) => /cache_hits/.test(failure)));
  assert.ok(failures.some((failure) => /network_requests/.test(failure)));
});

test("fails closed on DMD I/O, package drift, page replacement, or duplicate PNG", () => {
  const evidence = validMultiRequestEvidence();
  const second = evidence.requests[1];
  second.dmd_runtime_io_attestation.traffic.cache_lookups = 1;
  second.runtime_events.find(
    (event) => event.event === "packed_f16_denoiser_lifecycle",
  ).lifecycle.object_unpacks -= 1;
  second.cdp_dmd_network_traffic.model_response_count = 1;
  second.tested_package_identity.sources.browser_runtime.sha256 = "d".repeat(64);
  second.page_identity.engine_session_id = "replacement-engine-session";
  second.downloaded_png.file_name = evidence.requests[0].downloaded_png.file_name;
  const failures = validateTurbo1024MultiRequestEvidence(evidence);
  for (const pattern of [
    /DMD runtime artifact\/cache\/network traffic is not exact zero/,
    /object_unpacks/,
    /DMD CDP model_response_count/,
    /package identities differ/,
    /same page\/runtime identity/,
    /same filename/,
  ]) {
    assert.ok(failures.some((failure) => pattern.test(failure)), String(pattern));
  }
});

test("summarizes a request-scoped DMD CDP window independently", () => {
  const base = "http://127.0.0.1:39001/model/boogu-image-0.1-turbo";
  const snapshot = {
    time_origin_epoch_ms: 1_000_000,
    progress_events: [
      { at_ms: 1_000, event: "stage_started", stage: "dmd", run_id: 9 },
      { at_ms: 2_000, event: "stage_completed", stage: "dmd", run_id: 9 },
    ],
  };
  const summary = summarizeTurboDmdCdpNetwork(
    [
      {
        at_ms: 1_001_500,
        method: "Network.responseReceived",
        params: {
          response: {
            url: `${base}/unexpected.bpk`,
            status: 206,
            headers: { "Content-Length": "4", "Content-Range": "bytes 0-3/8" },
          },
        },
      },
    ],
    snapshot,
    [base],
  );
  assert.equal(summary.policy, TURBO_CDP_DMD_NETWORK_POLICY);
  assert.equal(summary.model_response_count, 1);
  assert.equal(summary.response_body_bytes, 4);
  assert.equal(summary.terminal_event, "stage_completed:dmd");
});
