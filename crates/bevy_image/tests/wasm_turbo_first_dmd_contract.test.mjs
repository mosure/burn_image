import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  ARTIFACT_TRAFFIC_FIELDS,
  REQUIRED_1K_FIXTURE_TENSOR_BYTES,
  REQUIRED_FIXTURE_TENSOR_BYTES,
  REQUIRED_FIXTURE_TENSOR_NAMES,
  TURBO_ARTIFACT_CONTENT_DIGEST,
  TURBO_1K_FIXTURE,
  TURBO_EXPECTED_CACHED_STAGES,
  TURBO_FIRST_DMD_CHROME_SHARED_MEMORY_DEV_SHM_POLICY,
  TURBO_FIRST_DMD_CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES,
  TURBO_FIRST_DMD_CHROME_SHARED_MEMORY_TMP_FALLBACK_POLICY,
  TURBO_FIRST_DMD_CHROME_SHARED_MEMORY_UNAVAILABLE_POLICY,
  TURBO_FIRST_DMD_LIFECYCLE,
  TURBO_FIRST_DMD_MODE,
  TURBO_FIXTURE,
  TURBO_LINEAR_EXECUTION_POLICY,
  TURBO_MODEL,
  TURBO_MODEL_REVISION,
  TURBO_QUANTIZED_EXECUTION_POLICY,
  TURBO_QUANTIZED_LOAD_POLICY,
  TURBO_RESIDENCY,
  TURBO_RESOURCE_PLAN,
  TURBO_STORAGE_POLICY,
  inspectTurboFirstDmdStorageAdmission,
  selectTurboFirstDmdChromeSharedMemoryPolicy,
  summarizeTurboFirstDmdWebGpuCalls,
  turboFirstDmdChromeLaunchEvidence,
  validateTurboFirstDmdReport,
  validateTurboFirstDmdSourceContract,
} from "./wasm_turbo_first_dmd_contract.mjs";

const testsDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(testsDir, "../../..");

function zeroTraffic() {
  return Object.fromEntries(ARTIFACT_TRAFFIC_FIELDS.map((field) => [field, 0]));
}

function validMetric(oracle, shape, elements) {
  return {
    name: `trajectory.${oracle}`,
    oracle,
    shape,
    actual_dtype: "f32",
    element_count: elements,
    max_abs: 0.25,
    mean_abs: 0.01,
    rmse: 0.02,
    relative_rmse: 0.03,
    cosine_similarity: 0.999,
  };
}

function validReport(fixture = TURBO_FIXTURE) {
  const fullResolution = fixture === TURBO_1K_FIXTURE;
  const qwenShape = fullResolution ? [1, 45, 4096] : [1, 49, 4096];
  const dmdShape = fullResolution ? [1, 16, 128, 128] : [1, 16, 32, 32];
  const dmdElements = fullResolution ? 262_144 : 16_384;
  const requiredBytes = fullResolution
    ? REQUIRED_1K_FIXTURE_TENSOR_BYTES
    : REQUIRED_FIXTURE_TENSOR_BYTES;
  const fixtureSigma = Math.fround(0.000_999_450_683_593_75);
  const productionSigma = Math.fround(0.001);
  return {
    report_schema_version: 2,
    mode: TURBO_FIRST_DMD_MODE,
    model_backend: "raw-cubecl-no-fusion",
    adapter_name: "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
    adapter_backend: "BrowserWebGpu",
    adapter_device_type: "DiscreteGpu",
    adapter_shader_f16: true,
    device_shader_f16: true,
    actual_storage_buffer_binding_size: 2_147_483_644,
    actual_max_buffer_size: 4_294_967_292,
    model: TURBO_MODEL,
    model_revision: TURBO_MODEL_REVISION,
    artifact_content_digest: TURBO_ARTIFACT_CONTENT_DIGEST,
    artifact_profile: "f16-qwen-vision-f32",
    residency_policy: TURBO_RESIDENCY,
    denoiser_storage_policy: TURBO_STORAGE_POLICY,
    denoiser_float_load_policy: "adapt-to-f32",
    denoiser_quantized_load_policy: TURBO_QUANTIZED_LOAD_POLICY,
    denoiser_quantized_linear_execution_policy: TURBO_QUANTIZED_EXECUTION_POLICY,
    denoiser_linear_execution_policy: TURBO_LINEAR_EXECUTION_POLICY,
    denoiser_execution_dtype: "f32",
    packed_f16_resource_plan: { ...TURBO_RESOURCE_PLAN },
    packed_f16_denoiser_lifecycle: {
      ...TURBO_FIRST_DMD_LIFECYCLE,
      dmd_artifact_traffic: zeroTraffic(),
    },
    expected_cached_stages: TURBO_EXPECTED_CACHED_STAGES,
    cached_stages_before_predict: TURBO_EXPECTED_CACHED_STAGES,
    cached_stages_after_predict: TURBO_EXPECTED_CACHED_STAGES,
    synchronization_pending_before_predict: false,
    synchronization_pending_after_predict: false,
    dmd_artifact_traffic: zeroTraffic(),
    fixture: structuredClone(fixture),
    fixture_verification: {
      verified_metadata_files: 1,
      verified_metadata_bytes: fixture.metadata.size,
      verified_metadata_sha256: fixture.metadata.sha256,
      verified_safetensors_headers: 1,
      verified_safetensors_header_bytes: 35_192,
      whole_safetensors_identity_pinned: true,
      whole_safetensors_file_verified: false,
      verified_required_tensors: REQUIRED_FIXTURE_TENSOR_NAMES.length,
      verified_required_tensor_bytes: requiredBytes,
      expected_required_tensors: REQUIRED_FIXTURE_TENSOR_NAMES.length,
      expected_required_tensor_bytes: requiredBytes,
      verified_required_tensor_names: [...REQUIRED_FIXTURE_TENSOR_NAMES],
    },
    fixture_dtype: "bf16",
    injected_qwen_shape: qwenShape,
    injected_dmd_input_shape: dmdShape,
    execution_sigma_source: "authenticated-fixture-bf16-widened-to-f32",
    fixture_bf16_sigma_widened_f32: fixtureSigma,
    production_f32_sigma: productionSigma,
    production_minus_fixture_sigma: Math.fround(productionSigma - fixtureSigma),
    velocity: validMetric("dmd.step.0.velocity", dmdShape, dmdElements),
    prediction: validMetric("dmd.step.0.prediction", dmdShape, dmdElements),
    fixture_required_inputs_authenticated: true,
    artifacts_verified: true,
    diagnostic_passed: true,
    on_device_quantized_execution_claimed: false,
    numerical_parity_claimed: false,
  };
}

function validBrowserWebGpuAttestation() {
  return summarizeTurboFirstDmdWebGpuCalls([
    {
      event: "request-adapter-start",
      detail: { powerPreference: "high-performance" },
    },
    {
      event: "request-adapter-resolved",
      detail: {
        available: true,
        info: {
          is_fallback_adapter: false,
          vendor: "nvidia",
          architecture: "blackwell",
          device: "0x2bb1",
          description: "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        },
      },
    },
  ]);
}

function admittedStorage(path) {
  return {
    path,
    exists: true,
    directory: true,
    writable: true,
    statfs: {
      available_bytes: TURBO_FIRST_DMD_CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES * 2,
    },
    quota_aware_allocation_probe: {
      attempted: true,
      requested_bytes: TURBO_FIRST_DMD_CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES,
      written_bytes: TURBO_FIRST_DMD_CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES,
      succeeded: true,
      cleanup_succeeded: true,
      errors: [],
    },
  };
}

function admittedDevShm() {
  return admittedStorage("/dev/shm");
}

test("admits /dev/shm only after global and quota-aware 256 MiB capacity proof", () => {
  const selected = selectTurboFirstDmdChromeSharedMemoryPolicy({
    platform: "linux",
    devShm: admittedDevShm(),
    tempDirectory: admittedStorage("/tmp"),
    tempPath: "/tmp",
  });
  assert.deepEqual(selected, {
    policy: TURBO_FIRST_DMD_CHROME_SHARED_MEMORY_DEV_SHM_POLICY,
    selected_backing: "dev-shm",
    selected_path: "/dev/shm",
    disable_dev_shm_usage: false,
    launch_admitted: true,
    minimum_admitted_headroom_bytes:
      TURBO_FIRST_DMD_CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES,
    dev_shm_admitted: true,
    dev_shm_rejections: [],
    temp_directory_admitted: null,
    temp_directory_rejections: [],
  });
});

test("falls back to temp-backed Chrome shmem when /dev/shm is quota-limited", () => {
  const devShm = admittedDevShm();
  devShm.quota_aware_allocation_probe.written_bytes = 107_565_056;
  devShm.quota_aware_allocation_probe.succeeded = false;
  const selected = selectTurboFirstDmdChromeSharedMemoryPolicy({
    platform: "linux",
    devShm,
    tempDirectory: admittedStorage("/tmp"),
    tempPath: "/tmp",
  });
  assert.equal(selected.policy, TURBO_FIRST_DMD_CHROME_SHARED_MEMORY_TMP_FALLBACK_POLICY);
  assert.equal(selected.selected_backing, "temp-directory");
  assert.equal(selected.selected_path, "/tmp");
  assert.equal(selected.disable_dev_shm_usage, true);
  assert.equal(selected.launch_admitted, true);
  assert.equal(selected.dev_shm_admitted, false);
  assert.equal(selected.temp_directory_admitted, true);
  assert.ok(selected.dev_shm_rejections.includes("quota-aware-probe-failed"));
});

test("fails closed when neither Chrome shared-memory backing passes quota admission", () => {
  const devShm = admittedDevShm();
  devShm.quota_aware_allocation_probe.succeeded = false;
  const tempDirectory = admittedStorage("/tmp");
  tempDirectory.quota_aware_allocation_probe.written_bytes = 93_106_176;
  tempDirectory.quota_aware_allocation_probe.succeeded = false;
  const selected = selectTurboFirstDmdChromeSharedMemoryPolicy({
    platform: "linux",
    devShm,
    tempDirectory,
    tempPath: "/tmp",
  });
  assert.equal(selected.policy, TURBO_FIRST_DMD_CHROME_SHARED_MEMORY_UNAVAILABLE_POLICY);
  assert.equal(selected.selected_backing, null);
  assert.equal(selected.disable_dev_shm_usage, null);
  assert.equal(selected.launch_admitted, false);
  assert.equal(selected.dev_shm_admitted, false);
  assert.equal(selected.temp_directory_admitted, false);
  assert.ok(selected.dev_shm_rejections.includes("quota-aware-probe-failed"));
  assert.ok(selected.temp_directory_rejections.includes("quota-aware-probe-failed"));
});

test("requires the Chrome profile filesystem to pass capacity and quota admission", () => {
  const profile = admittedStorage("/evidence");
  assert.deepEqual(inspectTurboFirstDmdStorageAdmission(profile), {
    admitted: true,
    minimum_admitted_headroom_bytes:
      TURBO_FIRST_DMD_CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES,
    rejections: [],
  });
  profile.quota_aware_allocation_probe.succeeded = false;
  const rejected = inspectTurboFirstDmdStorageAdmission(profile);
  assert.equal(rejected.admitted, false);
  assert.ok(rejected.rejections.includes("quota-aware-probe-failed"));
});

test("serializes exact Chrome launch and cleanup evidence", () => {
  const sharedMemory = selectTurboFirstDmdChromeSharedMemoryPolicy({
    platform: "linux",
    devShm: admittedDevShm(),
    tempDirectory: admittedStorage("/tmp"),
    tempPath: "/tmp",
  });
  const profileStorage = {
    path: "/evidence",
    measurement: admittedStorage("/evidence"),
    ...inspectTurboFirstDmdStorageAdmission(admittedStorage("/evidence")),
  };
  assert.deepEqual(
    turboFirstDmdChromeLaunchEvidence({
      executable: "/usr/bin/google-chrome",
      arguments: ["--user-data-dir=/evidence/profile", "about:blank"],
      profile: "/evidence/profile",
      sharedMemory,
      profileStorage,
      cleanup: {
        root_pid: 123,
        process_group_id: 123,
        process_group_exited: true,
        errors: [],
      },
      profileRemoved: true,
    }),
    {
      chrome_executable: "/usr/bin/google-chrome",
      chrome_arguments: ["--user-data-dir=/evidence/profile", "about:blank"],
      chrome_profile: "/evidence/profile",
      chrome_shared_memory: sharedMemory,
      chrome_profile_storage: profileStorage,
      chrome_cleanup: {
        root_pid: 123,
        process_group_id: 123,
        process_group_exited: true,
        errors: [],
      },
      chrome_profile_removed: true,
    },
  );
});

test("accepts exact current-policy first-DMD diagnostic without inventing a parity gate", () => {
  assert.deepEqual(validateTurboFirstDmdReport(validReport()), []);
});

test("accepts the strict full-resolution qualification fixture identity", () => {
  assert.deepEqual(validateTurboFirstDmdReport(validReport(TURBO_1K_FIXTURE)), []);
});

test("accepts BrowserWebGpu Other only with matching non-fallback NVIDIA evidence", () => {
  const report = validReport(TURBO_1K_FIXTURE);
  report.adapter_device_type = "Other";
  assert.deepEqual(
    validateTurboFirstDmdReport(report, validBrowserWebGpuAttestation()),
    [],
  );
});

test("accepts the exact privacy-redacted live BrowserWebGpu hardware evidence", () => {
  const report = validReport(TURBO_1K_FIXTURE);
  report.adapter_device_type = "Other";
  report.adapter_name = "";
  const attestation = summarizeTurboFirstDmdWebGpuCalls([
    {
      event: "request-adapter-start",
      detail: { powerPreference: "high-performance" },
    },
    {
      event: "request-adapter-resolved",
      detail: {
        available: true,
        info: {
          isFallbackAdapter: false,
          vendor: "nvidia",
          architecture: "blackwell",
          device: "",
          description: "",
        },
      },
    },
  ]);
  assert.deepEqual(validateTurboFirstDmdReport(report, attestation), []);
});

test("fails closed when BrowserWebGpu Other evidence is absent, fallback, or mismatched", () => {
  const report = validReport();
  report.adapter_device_type = "Other";
  assert.ok(
    validateTurboFirstDmdReport(report).some((failure) => /lacks instrumented/.test(failure)),
  );

  const fallback = validBrowserWebGpuAttestation();
  fallback.is_fallback_adapter = true;
  assert.ok(
    validateTurboFirstDmdReport(report, fallback).some((failure) => /explicitly false/.test(failure)),
  );

  const mismatched = validBrowserWebGpuAttestation();
  mismatched.vendor = "unknown";
  mismatched.description = "Unknown Adapter";
  assert.ok(
    validateTurboFirstDmdReport(report, mismatched).some((failure) => /not non-software NVIDIA/.test(failure)),
  );
  assert.ok(
    validateTurboFirstDmdReport(report, mismatched).some((failure) => /does not match/.test(failure)),
  );
});

test("fails closed on residency, packed cache lifecycle, DMD traffic, and claims", () => {
  const report = validReport();
  report.residency_policy = "browser-low-vram-runtime-q8-denoiser";
  report.denoiser_runtime_q8_scope = "turbo-main-core-ffn-gate-up-q8";
  report.low_vram_resource_plan = {};
  report.packed_f16_denoiser_lifecycle.cached_objects = 105;
  report.packed_f16_denoiser_lifecycle.stage_materializations = 45;
  report.packed_f16_denoiser_lifecycle.dmd_artifact_traffic.cache_lookups = 1;
  report.cached_stages_after_predict = 0;
  report.synchronization_pending_after_predict = true;
  report.dmd_artifact_traffic.network_requests = 1;
  report.on_device_quantized_execution_claimed = true;
  report.numerical_parity_claimed = true;
  const failures = validateTurboFirstDmdReport(report);
  for (const pattern of [
    /residency_policy/,
    /retired Q8-policy field denoiser_runtime_q8_scope/,
    /retired Q8-policy field low_vram_resource_plan/,
    /cached_objects/,
    /stage_materializations/,
    /packed_f16_denoiser_lifecycle\.dmd_artifact_traffic/,
    /cached_stages_after_predict/,
    /synchronization_pending_after_predict/,
    /network_requests/,
    /on_device_quantized_execution_claimed/,
    /numerical_parity_claimed/,
  ]) {
    assert.ok(failures.some((failure) => pattern.test(failure)), String(pattern));
  }
});

test("fails closed on fixture identity/authentication and non-finite output metrics", () => {
  const report = validReport();
  report.fixture.tensors.sha256 = "0".repeat(64);
  report.fixture_verification.whole_safetensors_file_verified = true;
  report.fixture_verification.verified_required_tensor_names.pop();
  report.fixture_required_inputs_authenticated = false;
  report.velocity.rmse = Number.NaN;
  const failures = validateTurboFirstDmdReport(report);
  for (const pattern of [
    /fixture.tensors/,
    /whole_safetensors_file_verified/,
    /verified required tensor names/,
    /fixture_required_inputs_authenticated/,
    /rmse is not finite/,
  ]) {
    assert.ok(failures.some((failure) => pattern.test(failure)), String(pattern));
  }
});

test("source contract pins the release fixture and exactly-one-predict scope", async () => {
  const inputs = Object.fromEntries(
    await Promise.all(
      Object.entries({
        libSource: "crates/bevy_image/src/lib.rs",
        browserSource: "crates/bevy_image/src/browser_boogu.rs",
        fixtureSource: "crates/bevy_image/src/browser_turbo_first_dmd_fixture.rs",
        parityFixtureSource: "crates/bevy_image/src/browser_parity_fixture.rs",
        parityWorkflowSource: ".github/workflows/parity.yml",
        harnessSource: "crates/bevy_image/tests/wasm_turbo_first_dmd_probe.mjs",
      }).map(async ([name, path]) => [name, await readFile(join(repoRoot, path), "utf8")]),
    ),
  );
  assert.deepEqual(validateTurboFirstDmdSourceContract(inputs), []);
});

test("source contract rejects unadmitted storage, temp defaults, weak cleanup, and lost identities", async () => {
  const inputs = Object.fromEntries(
    await Promise.all(
      Object.entries({
        libSource: "crates/bevy_image/src/lib.rs",
        browserSource: "crates/bevy_image/src/browser_boogu.rs",
        fixtureSource: "crates/bevy_image/src/browser_turbo_first_dmd_fixture.rs",
        parityFixtureSource: "crates/bevy_image/src/browser_parity_fixture.rs",
        parityWorkflowSource: ".github/workflows/parity.yml",
        harnessSource: "crates/bevy_image/tests/wasm_turbo_first_dmd_probe.mjs",
      }).map(async ([name, path]) => [name, await readFile(join(repoRoot, path), "utf8")]),
    ),
  );
  inputs.harnessSource = inputs.harnessSource
    .replace(
      `  if (sharedMemoryPolicy.disable_dev_shm_usage) {
    arguments_.push("--disable-dev-shm-usage");
  }`,
      `  arguments_.push("--disable-dev-shm-usage");`,
    )
    .replace(
      'mkdtemp(join(outputDir, "burn-image-turbo-first-dmd-chrome-"))',
      'mkdtemp(join(tmpdir(), "burn-image-turbo-first-dmd-chrome-"))',
    )
    .replace("detached: true", "detached: false")
    .replace("maxRetries: 8", "maxRetries: 0")
    .replace("...(inputEvidence ?? {}),", "// validated identities omitted")
    .replace("process_group_exited: exited", "process_group_exited: false")
    .replace("if (!validateOnly && !explicitOutputDir)", "if (false)")
    .replace("chromeSharedMemory.launch_admitted !== true", "false")
    .replace("chromeProfileStorage.admitted !== true", "false")
    .replaceAll("tempDirectory,", "")
    .replaceAll("SERVER_CLOSE_TIMEOUT_MS", "UNBOUNDED_SERVER_CLOSE")
    .replace("server.closeAllConnections?.();", "// active connections retained")
    .replace("socket.destroy();", "// socket retained")
    .replace("terminalOutput.server_cleanup = serverCleanup;", "// cleanup evidence omitted")
    .replaceAll("CDP_CALL_TIMEOUT_MS", "UNBOUNDED_CDP_CALL")
    .replace("failPending(error)", "ignorePending(error)")
    .replace('this.socket.addEventListener("error"', 'this.socket.ignoreEvent("error"')
    .replace('this.socket.addEventListener("close"', 'this.socket.ignoreEvent("close"')
    .replaceAll("clearTimeout(pending.timeout)", "// pending timeout retained")
    .replace("this.pending.clear()", "// pending calls retained");
  const failures = validateTurboFirstDmdSourceContract(inputs);
  for (const pattern of [
    /quota-aware fallback/,
    /profile under outputDir/,
    /detached: true/,
    /process_group_exited/,
    /maxRetries: 8/,
    /already-validated input identities/,
    /explicitOutputDir/,
    /chromeSharedMemory\.launch_admitted/,
    /chromeProfileStorage\.admitted/,
    /tempDirectory/,
    /SERVER_CLOSE_TIMEOUT_MS/,
    /server\.closeAllConnections/,
    /socket\.destroy/,
    /terminalOutput\.server_cleanup/,
    /CDP_CALL_TIMEOUT_MS/,
    /failPending\(error\)/,
    /this\.socket\.addEventListener\("error"/,
    /this\.socket\.addEventListener\("close"/,
    /clearTimeout\(pending\.timeout\)/,
    /this\.pending\.clear/,
    /bounded timeout for each CDP call/,
  ]) {
    assert.ok(failures.some((failure) => pattern.test(failure)), String(pattern));
  }
});

test("source contract rejects an extra predict and coupling to the 1.5K reader", async () => {
  const inputs = Object.fromEntries(
    await Promise.all(
      Object.entries({
        libSource: "crates/bevy_image/src/lib.rs",
        browserSource: "crates/bevy_image/src/browser_boogu.rs",
        fixtureSource: "crates/bevy_image/src/browser_turbo_first_dmd_fixture.rs",
        parityFixtureSource: "crates/bevy_image/src/browser_parity_fixture.rs",
        parityWorkflowSource: ".github/workflows/parity.yml",
        harnessSource: "crates/bevy_image/tests/wasm_turbo_first_dmd_probe.mjs",
      }).map(async ([name, path]) => [name, await readFile(join(repoRoot, path), "utf8")]),
    ),
  );
  inputs.browserSource = inputs.browserSource.replace(
    "async fn vae_reference_1k5(",
    ".predict_async(fake);\nasync fn vae_reference_1k5(",
  );
  inputs.browserSource = inputs.browserSource.replace(
    TURBO_FIRST_DMD_MODE,
    "diagnostic-no-surface-turbo-first-dmd-wrong-packed-policy",
  );
  inputs.parityFixtureSource += "\nconst TURBO_FIRST_DMD_COUPLING: bool = true;\n";
  const failures = validateTurboFirstDmdSourceContract(inputs);
  assert.ok(failures.some((failure) => /exactly one predict_async/.test(failure)));
  assert.ok(
    failures.some((failure) => /packed-f16-dense-f32-per-stage-policy/.test(failure)),
  );
  assert.ok(failures.some((failure) => /1.5K fixture reader/.test(failure)));
});

test("source contract rejects rendered Turbo workflow mode drift", async () => {
  const inputs = Object.fromEntries(
    await Promise.all(
      Object.entries({
        libSource: "crates/bevy_image/src/lib.rs",
        browserSource: "crates/bevy_image/src/browser_boogu.rs",
        fixtureSource: "crates/bevy_image/src/browser_turbo_first_dmd_fixture.rs",
        parityFixtureSource: "crates/bevy_image/src/browser_parity_fixture.rs",
        parityWorkflowSource: ".github/workflows/parity.yml",
        harnessSource: "crates/bevy_image/tests/wasm_turbo_first_dmd_probe.mjs",
      }).map(async ([name, path]) => [name, await readFile(join(repoRoot, path), "utf8")]),
    ),
  );
  inputs.parityWorkflowSource = inputs.parityWorkflowSource.replace(
    "BURN_IMAGE_RENDERED_TURBO_QWEN_BLOCK0_EXECUTION_MODE=ordinary",
    "BURN_IMAGE_RENDERED_TURBO_QWEN_BLOCK0_EXECUTION_MODE=serialized-diagnostic",
  );
  const failures = validateTurboFirstDmdSourceContract(inputs);
  assert.ok(failures.some((failure) => /both rendered Turbo gates/.test(failure)));
  assert.ok(failures.some((failure) => /must not feed serialized diagnostics/.test(failure)));
});
