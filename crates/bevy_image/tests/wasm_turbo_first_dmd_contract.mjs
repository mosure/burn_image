export const TERMINAL_OK = "BURN_IMAGE_HEADLESS_TURBO_FIRST_DMD_OK ";
export const TERMINAL_FAILED = "BURN_IMAGE_HEADLESS_TURBO_FIRST_DMD_FAILED";
export const TURBO_MODEL = "Boogu/Boogu-Image-0.1-Turbo";
export const TURBO_FIRST_DMD_MODE =
  "diagnostic-no-surface-turbo-first-dmd-packed-f16-dense-f32-per-stage-policy";
export const TURBO_MODEL_REVISION = "53ad54522023f64d049f7f38e4d679359ef3fb92";
export const TURBO_UPSTREAM_SOURCE_REVISION =
  "25f8f888298224a94e5ec2abafb98abea9031a0d";
export const TURBO_ARTIFACT_CONTENT_DIGEST =
  "32b2f0a972d7c00e4bc914f949dcf15195c10c428be456330a168a556576138a";
export const TURBO_RESIDENCY =
  "browser-low-vram-preloaded-packed-f16-dense-f32-per-stage-denoiser";
export const TURBO_STORAGE_POLICY =
  "authenticated-compact-f16/padded-u32-retained/dense-f32-per-semantic-stage";
export const TURBO_QUANTIZED_LOAD_POLICY =
  "not-applicable-packed-f16-storage";
export const TURBO_QUANTIZED_EXECUTION_POLICY =
  "not-applicable-packed-f16-storage";
export const TURBO_LINEAR_EXECUTION_POLICY =
  "packed-f16-storage/device-widen-f32-per-semantic-stage/dense-f32-matmul";
export const TURBO_EXPECTED_CACHED_STAGES = 46;
export const TURBO_EXPECTED_CACHED_OBJECTS = 106;
export const TURBO_EXPECTED_TENSORS = 912;
export const STRICT_DEVICE_CAP_BYTES = 32_000_000_000;
export const TURBO_FIRST_DMD_CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES = 256 * 1024 * 1024;
export const TURBO_FIRST_DMD_CHROME_SHARED_MEMORY_DEV_SHM_POLICY =
  "linux-dev-shm-statfs-and-quota-aware-probe-admitted";
export const TURBO_FIRST_DMD_CHROME_SHARED_MEMORY_TMP_FALLBACK_POLICY =
  "linux-temp-fallback-dev-shm-not-admitted-and-temp-admitted";
export const TURBO_FIRST_DMD_CHROME_SHARED_MEMORY_UNAVAILABLE_POLICY =
  "linux-no-quota-and-capacity-admitted-shared-memory-backing";
export const TURBO_FIRST_DMD_CHROME_SHARED_MEMORY_PLATFORM_DEFAULT_POLICY =
  "non-linux-platform-default";

function exactNonNegativeInteger(value) {
  if (Number.isSafeInteger(value) && value >= 0) return BigInt(value);
  if (typeof value === "string" && /^(0|[1-9][0-9]*)$/.test(value)) {
    return BigInt(value);
  }
  return null;
}

export function inspectTurboFirstDmdStorageAdmission(
  measurement,
  minimumHeadroomBytes = TURBO_FIRST_DMD_CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES,
) {
  if (!Number.isSafeInteger(minimumHeadroomBytes) || minimumHeadroomBytes <= 0) {
    throw new Error("Chrome storage minimum headroom must be a positive safe integer");
  }
  const rejections = [];
  if (measurement?.exists !== true) rejections.push("missing");
  if (measurement?.directory !== true) rejections.push("not-directory");
  if (measurement?.writable !== true) rejections.push("not-writable");
  const availableBytes = exactNonNegativeInteger(measurement?.statfs?.available_bytes);
  if (availableBytes === null) {
    rejections.push("statfs-available-unknown");
  } else if (availableBytes < BigInt(minimumHeadroomBytes)) {
    rejections.push("statfs-available-below-minimum");
  }
  const allocationProbe = measurement?.quota_aware_allocation_probe;
  if (allocationProbe?.attempted !== true) {
    rejections.push("quota-aware-probe-not-attempted");
  } else if (allocationProbe?.succeeded !== true) {
    rejections.push("quota-aware-probe-failed");
  } else if (
    exactNonNegativeInteger(allocationProbe?.written_bytes) === null ||
    exactNonNegativeInteger(allocationProbe.written_bytes) < BigInt(minimumHeadroomBytes)
  ) {
    rejections.push("quota-aware-probe-below-minimum");
  }
  return {
    admitted: rejections.length === 0,
    minimum_admitted_headroom_bytes: minimumHeadroomBytes,
    rejections,
  };
}

export function selectTurboFirstDmdChromeSharedMemoryPolicy({
  platform,
  devShm,
  tempDirectory,
  tempPath = null,
  minimumHeadroomBytes = TURBO_FIRST_DMD_CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES,
}) {
  if (!Number.isSafeInteger(minimumHeadroomBytes) || minimumHeadroomBytes <= 0) {
    throw new Error("Chrome shared-memory minimum headroom must be a positive safe integer");
  }
  if (platform !== "linux") {
    return {
      policy: TURBO_FIRST_DMD_CHROME_SHARED_MEMORY_PLATFORM_DEFAULT_POLICY,
      selected_backing: "platform-default",
      selected_path: null,
      disable_dev_shm_usage: false,
      launch_admitted: true,
      minimum_admitted_headroom_bytes: minimumHeadroomBytes,
      dev_shm_admitted: null,
      dev_shm_rejections: [],
      temp_directory_admitted: null,
      temp_directory_rejections: [],
    };
  }

  const devShmAdmission = inspectTurboFirstDmdStorageAdmission(
    devShm,
    minimumHeadroomBytes,
  );

  if (devShmAdmission.admitted) {
    return {
      policy: TURBO_FIRST_DMD_CHROME_SHARED_MEMORY_DEV_SHM_POLICY,
      selected_backing: "dev-shm",
      selected_path: devShm.path,
      disable_dev_shm_usage: false,
      launch_admitted: true,
      minimum_admitted_headroom_bytes: minimumHeadroomBytes,
      dev_shm_admitted: true,
      dev_shm_rejections: [],
      temp_directory_admitted: null,
      temp_directory_rejections: [],
    };
  }
  const tempDirectoryAdmission = inspectTurboFirstDmdStorageAdmission(
    tempDirectory,
    minimumHeadroomBytes,
  );
  if (!tempDirectoryAdmission.admitted) {
    return {
      policy: TURBO_FIRST_DMD_CHROME_SHARED_MEMORY_UNAVAILABLE_POLICY,
      selected_backing: null,
      selected_path: null,
      disable_dev_shm_usage: null,
      launch_admitted: false,
      minimum_admitted_headroom_bytes: minimumHeadroomBytes,
      dev_shm_admitted: false,
      dev_shm_rejections: devShmAdmission.rejections,
      temp_directory_admitted: false,
      temp_directory_rejections: tempDirectoryAdmission.rejections,
    };
  }
  return {
    policy: TURBO_FIRST_DMD_CHROME_SHARED_MEMORY_TMP_FALLBACK_POLICY,
    selected_backing: "temp-directory",
    selected_path:
      typeof tempDirectory?.path === "string"
        ? tempDirectory.path
        : typeof tempPath === "string"
          ? tempPath
          : null,
    disable_dev_shm_usage: true,
    launch_admitted: true,
    minimum_admitted_headroom_bytes: minimumHeadroomBytes,
    dev_shm_admitted: false,
    dev_shm_rejections: devShmAdmission.rejections,
    temp_directory_admitted: true,
    temp_directory_rejections: [],
  };
}

export function turboFirstDmdChromeLaunchEvidence({
  executable,
  arguments: chromeArguments,
  profile,
  sharedMemory,
  profileStorage,
  cleanup,
  profileRemoved,
}) {
  return {
    chrome_executable: typeof executable === "string" ? executable : null,
    chrome_arguments: Array.isArray(chromeArguments) ? [...chromeArguments] : null,
    chrome_profile: typeof profile === "string" ? profile : null,
    chrome_shared_memory: sharedMemory ?? null,
    chrome_profile_storage: profileStorage ?? null,
    chrome_cleanup: cleanup ?? null,
    chrome_profile_removed: profileRemoved === true,
  };
}

export const TURBO_RESOURCE_PLAN = Object.freeze({
  authenticated_artifact_bytes: 19_870_166_528,
  canonical_compact_f16_payload_bytes: 19_869_996_096,
  retained_packed_f16_denoiser_bytes: 19_870_010_624,
  inserted_padding_elements: 7_264,
  padded_f16_elements: 9_935_005_312,
  expected_stage_count: TURBO_EXPECTED_CACHED_STAGES,
  expected_object_count: TURBO_EXPECTED_CACHED_OBJECTS,
  expected_tensor_count: TURBO_EXPECTED_TENSORS,
  max_packed_stage_bytes: 876_827_328,
  max_materialized_stage_f32_bytes: 1_753_654_656,
  max_packed_object_bytes: 254_251_904,
  max_materialized_object_f32_bytes: 508_503_808,
  materialized_f32_bytes_per_dmd_step: 39_740_021_248,
  preload_workspace_bytes: 2_434_252_800,
  preload_peak_bytes: 22_304_263_424,
  activation_reserve_bytes: 4_868_505_600,
  conservative_planned_device_bytes: 26_492_170_880,
  strict_device_cap_bytes: STRICT_DEVICE_CAP_BYTES,
  expected_stage_materializations_per_request: 184,
  expected_object_unpacks_per_request: 424,
  expected_packed_read_bytes_per_request: 79_480_042_496,
  expected_f32_write_bytes_per_request: 158_960_084_992,
  on_device_quantized_execution_claimed: false,
});

export const TURBO_FIRST_DMD_LIFECYCLE = Object.freeze({
  cache_state: "ready",
  cache_ready: true,
  cached_stages: TURBO_EXPECTED_CACHED_STAGES,
  cached_objects: TURBO_EXPECTED_CACHED_OBJECTS,
  cached_tensors: TURBO_EXPECTED_TENSORS,
  cached_bytes: TURBO_RESOURCE_PLAN.retained_packed_f16_denoiser_bytes,
  authenticated_artifact_bytes: TURBO_RESOURCE_PLAN.authenticated_artifact_bytes,
  packed_upload_bytes: TURBO_RESOURCE_PLAN.retained_packed_f16_denoiser_bytes,
  stage_materializations: TURBO_EXPECTED_CACHED_STAGES,
  object_unpacks: TURBO_EXPECTED_CACHED_OBJECTS,
  packed_read_bytes: TURBO_RESOURCE_PLAN.retained_packed_f16_denoiser_bytes,
  f32_write_bytes: TURBO_RESOURCE_PLAN.materialized_f32_bytes_per_dmd_step,
  preload_attempt_count: 1,
  failure_count: 0,
  synchronization_pending: false,
  matches_plan: true,
});

export const TURBO_FIXTURE = Object.freeze({
  profile: "release-256-bf16",
  schema_version: 2,
  variant: "turbo",
  model_revision: TURBO_MODEL_REVISION,
  upstream_source_revision: TURBO_UPSTREAM_SOURCE_REVISION,
  width: 256,
  height: 256,
  seed: 42,
  metadata: Object.freeze({
    path: "metadata.json",
    size: 73_936,
    sha256: "b85f757d18c5dfdb94b3366692d0b3ceb6f73d699e98820ba5177d90d36d838d",
  }),
  tensors: Object.freeze({
    path: "tensors.safetensors",
    size: 375_578_272,
    sha256: "d26178de365bea3a7e4fef6c9a292e8220af5481dfe398e74491817adc1fcd75",
  }),
  output: Object.freeze({
    path: "output.png",
    size: 61_874,
    sha256: "80beb2ebd1f15ead37b48bab2741beb0e06befe1530730eba304596b5a1d5554",
  }),
});

export const TURBO_1K_FIXTURE = Object.freeze({
  profile: "qualification-1024-bf16",
  schema_version: 2,
  variant: "turbo",
  model_revision: TURBO_MODEL_REVISION,
  upstream_source_revision: TURBO_UPSTREAM_SOURCE_REVISION,
  width: 1024,
  height: 1024,
  seed: 0,
  metadata: Object.freeze({
    path: "metadata.json",
    size: 82_947,
    sha256: "a7cf73b0ea0183d58b25f5c41eb732c28ed7a0aef52465365387d84cf2af0758",
  }),
  tensors: Object.freeze({
    path: "tensors.safetensors",
    size: 4_829_366_000,
    sha256: "eb3a81e7285f25df69a4e20a9f7d71d318bf9ccc5f84f2819c38fc2c1311f40e",
  }),
  output: Object.freeze({
    path: "output.png",
    size: 1_153_523,
    sha256: "4abd717984140ace64143617f1981025917c1f35ceb2271501880b350961d703",
  }),
});

export const REQUIRED_FIXTURE_TENSOR_NAMES = Object.freeze([
  "dmd.step.0.input",
  "dmd.step.0.prediction",
  "dmd.step.0.sigma",
  "dmd.step.0.velocity",
  "qwen.last_hidden_state",
]);
export const REQUIRED_FIXTURE_TENSOR_BYTES = 499_714;
export const REQUIRED_1K_FIXTURE_TENSOR_BYTES = 1_941_506;

export const ARTIFACT_TRAFFIC_FIELDS = Object.freeze([
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
]);

const SOFTWARE_ADAPTER = /(swiftshader|llvmpipe|lavapipe|software adapter|warp)/i;
const NVIDIA_ADAPTER = /nvidia/i;

/**
 * Summarize the JavaScript interception of the exact navigator.gpu adapter request made by wgpu.
 * wgpu 29 deliberately reports BrowserWebGpu adapters as DeviceType::Other, so this independent
 * browser-native evidence is required before that redacted Rust value can represent hardware.
 */
export function summarizeTurboFirstDmdWebGpuCalls(webGpuCalls) {
  const calls = Array.isArray(webGpuCalls) ? webGpuCalls : [];
  const starts = calls.filter((entry) => entry?.event === "request-adapter-start");
  const rejections = calls.filter((entry) => entry?.event === "request-adapter-rejected");
  const successes = calls.filter(
    (entry) => entry?.event === "request-adapter-resolved" && entry?.detail?.available === true,
  );
  const selected = successes.at(-1)?.detail?.info ?? null;
  const selectedStart = starts.at(-1)?.detail ?? null;
  return {
    source: "instrumented-navigator-gpu-request-adapter",
    request_attempts: starts.length,
    rejected_attempts: rejections.length,
    successful_attempts: successes.length,
    power_preference: selectedStart?.powerPreference ?? null,
    force_fallback_adapter: selectedStart?.forceFallbackAdapter ?? null,
    // Accept the original probe field while preserving the normalized snake-case form emitted by
    // the hardened HTML. This keeps already-captured, identity-pinned evidence re-validatable.
    is_fallback_adapter:
      selected?.is_fallback_adapter ?? selected?.isFallbackAdapter ?? null,
    vendor: selected?.vendor ?? null,
    architecture: selected?.architecture ?? null,
    device: selected?.device ?? null,
    description: selected?.description ?? null,
  };
}

function exactObject(actual, expected, scope, failures) {
  if (!actual || typeof actual !== "object" || Array.isArray(actual)) {
    failures.push(`${scope} is not an object`);
    return;
  }
  for (const [field, value] of Object.entries(expected)) {
    const equal =
      value && typeof value === "object"
        ? JSON.stringify(actual[field]) === JSON.stringify(value)
        : Object.is(actual[field], value);
    if (!equal) {
      failures.push(
        `${scope}.${field} is ${JSON.stringify(actual[field])}; expected ${JSON.stringify(value)}`,
      );
    }
  }
}

function validateMetric(metric, oracle, shape, elements, failures) {
  if (metric?.oracle !== oracle) {
    failures.push(`${oracle} metric has oracle ${JSON.stringify(metric?.oracle)}`);
  }
  if (JSON.stringify(metric?.shape) !== JSON.stringify(shape)) {
    failures.push(`${oracle} metric shape is ${JSON.stringify(metric?.shape)}`);
  }
  if (metric?.actual_dtype !== "f32") {
    failures.push(`${oracle} actual_dtype is ${JSON.stringify(metric?.actual_dtype)}`);
  }
  if (metric?.element_count !== elements) {
    failures.push(`${oracle} element_count is ${JSON.stringify(metric?.element_count)}`);
  }
  for (const field of [
    "max_abs",
    "mean_abs",
    "rmse",
    "relative_rmse",
    "cosine_similarity",
  ]) {
    if (!Number.isFinite(metric?.[field])) {
      failures.push(`${oracle}.${field} is not finite`);
    }
  }
}

/** Validate one returned report without inventing an uncalibrated numerical threshold. */
export function validateTurboFirstDmdReport(report, hardwareAdapterAttestation = null) {
  const failures = [];
  const fixture = report?.fixture?.profile === TURBO_1K_FIXTURE.profile
    ? TURBO_1K_FIXTURE
    : TURBO_FIXTURE;
  const fullResolution = fixture === TURBO_1K_FIXTURE;
  const qwenShape = fullResolution ? [1, 45, 4096] : [1, 49, 4096];
  const dmdShape = fullResolution ? [1, 16, 128, 128] : [1, 16, 32, 32];
  const dmdElements = fullResolution ? 262_144 : 16_384;
  const requiredTensorBytes = fullResolution
    ? REQUIRED_1K_FIXTURE_TENSOR_BYTES
    : REQUIRED_FIXTURE_TENSOR_BYTES;
  const exact = {
    report_schema_version: 2,
    mode: TURBO_FIRST_DMD_MODE,
    model_backend: "raw-cubecl-no-fusion",
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
    expected_cached_stages: TURBO_EXPECTED_CACHED_STAGES,
    cached_stages_before_predict: TURBO_EXPECTED_CACHED_STAGES,
    cached_stages_after_predict: TURBO_EXPECTED_CACHED_STAGES,
    synchronization_pending_before_predict: false,
    synchronization_pending_after_predict: false,
    fixture_dtype: "bf16",
    execution_sigma_source: "authenticated-fixture-bf16-widened-to-f32",
    fixture_required_inputs_authenticated: true,
    artifacts_verified: true,
    diagnostic_passed: true,
    on_device_quantized_execution_claimed: false,
    numerical_parity_claimed: false,
  };
  exactObject(report, exact, "report", failures);
  for (const retiredField of [
    "denoiser_runtime_q8_scope",
    "low_vram_resource_plan",
    "low_vram_denoiser_dtype_audit",
    "dense_f32_materialized_stage_clones",
    "expected_retained_stages",
    "retained_stages_before_predict",
    "retained_stages_after_predict",
  ]) {
    if (Object.hasOwn(report ?? {}, retiredField)) {
      failures.push(`report exposes retired Q8-policy field ${retiredField}`);
    }
  }

  if (!/webgpu/i.test(report?.adapter_backend ?? "")) {
    failures.push(`adapter_backend is not WebGPU: ${JSON.stringify(report?.adapter_backend)}`);
  }
  const adapterDeviceType = report?.adapter_device_type ?? "";
  if (/^(discretegpu|integratedgpu)$/i.test(adapterDeviceType)) {
    // Native wgpu backends expose the device type directly.
  } else if (/^other$/i.test(adapterDeviceType)) {
    const attestation = hardwareAdapterAttestation ?? {};
    if (attestation.source !== "instrumented-navigator-gpu-request-adapter") {
      failures.push("BrowserWebGpu DeviceType::Other lacks instrumented adapter evidence");
    }
    if (!Number.isSafeInteger(attestation.request_attempts) || attestation.request_attempts < 1) {
      failures.push("BrowserWebGpu adapter attestation has no request attempt");
    }
    if (attestation.successful_attempts !== 1) {
      failures.push(
        `BrowserWebGpu adapter attestation has ${JSON.stringify(attestation.successful_attempts)} successful requests; expected exactly one`,
      );
    }
    if (attestation.power_preference !== "high-performance") {
      failures.push(
        `BrowserWebGpu adapter power preference is ${JSON.stringify(attestation.power_preference)}`,
      );
    }
    if (attestation.force_fallback_adapter === true) {
      failures.push("BrowserWebGpu adapter request explicitly forced fallback");
    }
    if (attestation.is_fallback_adapter !== false) {
      failures.push(
        `BrowserWebGpu adapter fallback status is not explicitly false: ${JSON.stringify(attestation.is_fallback_adapter)}`,
      );
    }
    const nativeIdentity = [
      attestation.vendor,
      attestation.architecture,
      attestation.device,
      attestation.description,
    ].join(" ");
    if (!NVIDIA_ADAPTER.test(nativeIdentity) || SOFTWARE_ADAPTER.test(nativeIdentity)) {
      failures.push(
        `BrowserWebGpu adapter evidence is not non-software NVIDIA hardware: ${JSON.stringify(nativeIdentity)}`,
      );
    }
    if (
      typeof attestation.description !== "string" ||
      attestation.description !== report?.adapter_name
    ) {
      failures.push(
        `BrowserWebGpu adapter description ${JSON.stringify(attestation.description)} does not match Rust adapter_name ${JSON.stringify(report?.adapter_name)}`,
      );
    }
  } else {
    failures.push(`adapter_device_type is not a hardware GPU: ${JSON.stringify(adapterDeviceType)}`);
  }
  if (SOFTWARE_ADAPTER.test(report?.adapter_name ?? "")) {
    failures.push(`software adapter is not accepted: ${JSON.stringify(report?.adapter_name)}`);
  }
  for (const field of ["actual_storage_buffer_binding_size", "actual_max_buffer_size"]) {
    if (!Number.isSafeInteger(report?.[field]) || report[field] <= 0) {
      failures.push(`${field} is not a positive safe integer`);
    }
  }

  exactObject(report?.packed_f16_resource_plan, TURBO_RESOURCE_PLAN, "resource_plan", failures);
  if (
    report?.packed_f16_resource_plan?.preload_peak_bytes >= STRICT_DEVICE_CAP_BYTES ||
    report?.packed_f16_resource_plan?.conservative_planned_device_bytes >=
    STRICT_DEVICE_CAP_BYTES
  ) {
    failures.push("resource preload or inference plan is not strictly below 32,000,000,000 bytes");
  }
  exactObject(
    report?.packed_f16_denoiser_lifecycle,
    {
      ...TURBO_FIRST_DMD_LIFECYCLE,
      dmd_artifact_traffic: Object.fromEntries(
        ARTIFACT_TRAFFIC_FIELDS.map((field) => [field, 0]),
      ),
    },
    "packed_f16_denoiser_lifecycle",
    failures,
  );

  for (const field of ARTIFACT_TRAFFIC_FIELDS) {
    if (report?.dmd_artifact_traffic?.[field] !== 0) {
      failures.push(`dmd_artifact_traffic.${field} is not zero`);
    }
  }
  exactObject(report?.fixture, fixture, "fixture", failures);
  const verification = report?.fixture_verification;
  exactObject(
    verification,
    {
      verified_metadata_files: 1,
      verified_metadata_bytes: fixture.metadata.size,
      verified_metadata_sha256: fixture.metadata.sha256,
      verified_safetensors_headers: 1,
      whole_safetensors_identity_pinned: true,
      whole_safetensors_file_verified: false,
      verified_required_tensors: REQUIRED_FIXTURE_TENSOR_NAMES.length,
      verified_required_tensor_bytes: requiredTensorBytes,
      expected_required_tensors: REQUIRED_FIXTURE_TENSOR_NAMES.length,
      expected_required_tensor_bytes: requiredTensorBytes,
    },
    "fixture_verification",
    failures,
  );
  if (!Number.isSafeInteger(verification?.verified_safetensors_header_bytes) ||
      verification.verified_safetensors_header_bytes <= 8) {
    failures.push("fixture SafeTensors header byte count is invalid");
  }
  const names = [...(verification?.verified_required_tensor_names ?? [])].sort();
  if (JSON.stringify(names) !== JSON.stringify(REQUIRED_FIXTURE_TENSOR_NAMES)) {
    failures.push(`verified required tensor names are ${JSON.stringify(names)}`);
  }
  if (JSON.stringify(report?.injected_qwen_shape) !== JSON.stringify(qwenShape)) {
    failures.push(`injected_qwen_shape is ${JSON.stringify(report?.injected_qwen_shape)}`);
  }
  if (JSON.stringify(report?.injected_dmd_input_shape) !== JSON.stringify(dmdShape)) {
    failures.push(
      `injected_dmd_input_shape is ${JSON.stringify(report?.injected_dmd_input_shape)}`,
    );
  }

  const fixtureSigma = Math.fround(0.000_999_450_683_593_75);
  const productionSigma = Math.fround(0.001);
  if (Math.fround(report?.fixture_bf16_sigma_widened_f32) !== fixtureSigma) {
    failures.push("fixture BF16 sigma is not the exact widened captured value");
  }
  if (Math.fround(report?.production_f32_sigma) !== productionSigma) {
    failures.push("production F32 sigma is not 0.001");
  }
  if (
    Math.fround(report?.production_minus_fixture_sigma) !==
    Math.fround(productionSigma - fixtureSigma)
  ) {
    failures.push("reported production/captured sigma delta is inconsistent");
  }
  if (fixtureSigma === productionSigma) {
    failures.push("contract incorrectly collapses captured BF16 and production F32 sigma");
  }

  validateMetric(report?.velocity, "dmd.step.0.velocity", dmdShape, dmdElements, failures);
  validateMetric(report?.prediction, "dmd.step.0.prediction", dmdShape, dmdElements, failures);
  return failures;
}

function functionSlice(source, start, end) {
  const suffix = source.split(start)[1];
  if (!suffix) return null;
  return suffix.split(end)[0] ?? null;
}

function occurrenceCount(source, needle) {
  return source?.split(needle).length - 1;
}

/** Source-level guard for route ownership and the exactly-one-predict diagnostic scope. */
export function validateTurboFirstDmdSourceContract({
  libSource,
  browserSource,
  fixtureSource,
  parityFixtureSource,
  parityWorkflowSource,
  harnessSource,
  transportContractSource,
}) {
  const failures = [];
  for (const required of [
    "mod browser_turbo_first_dmd_fixture;",
    'Some("turbo-first-dmd") => Ok(Some(BrowserHeadlessMode::TurboFirstDmd))',
    "BrowserBooguFactory::turbo_first_dmd_no_surface",
    TERMINAL_OK.trimEnd(),
    TERMINAL_FAILED,
  ]) {
    if (!libSource.includes(required)) failures.push(`lib.rs omits ${required}`);
  }
  const diagnostic = functionSlice(
    browserSource,
    "async fn turbo_first_dmd(",
    "async fn vae_reference_1k5(",
  );
  if (!diagnostic) {
    failures.push("browser runtime omits the private first-DMD diagnostic");
  } else {
    if (occurrenceCount(diagnostic, ".predict_async(") !== 1) {
      failures.push("first-DMD diagnostic does not execute exactly one predict_async call");
    }
    if (occurrenceCount(diagnostic, "dmd_prediction(") !== 1) {
      failures.push("first-DMD diagnostic does not execute exactly one dmd_prediction call");
    }
    for (const required of [
      "LowVramPreloadedPackedF16Denoiser",
      "uses_packed_f16_denoiser_source()",
      "DirectQuantizedMatmul",
      TURBO_FIRST_DMD_MODE,
      "expected_cached_stages != 46",
      "audit_before_predict.cached_object_count",
      "audit_before_predict.cached_tensor_count",
      "audit_before_predict.retained_packed_bytes",
      "validate_packed_f16_denoiser_lifecycle(",
      "dmd_artifact_traffic != BrowserArtifactTrafficReport::default()",
      'execution_sigma_source: "authenticated-fixture-bf16-widened-to-f32"',
      "production_f32_sigma",
      "numerical_parity_claimed: false",
    ]) {
      if (!diagnostic.includes(required)) failures.push(`diagnostic omits ${required}`);
    }
  }
  for (const required of [
    TURBO_FIXTURE.metadata.sha256,
    TURBO_FIXTURE.tensors.sha256,
    TURBO_FIXTURE.output.sha256,
    "TURBO_FIRST_DMD_REQUIRED_TENSOR_COUNT: usize = 5",
    "TURBO_FIRST_DMD_REQUIRED_TENSOR_BYTES: u64 = 499_714",
    "TURBO_FIRST_DMD_1K_REQUIRED_TENSOR_BYTES: u64 = 1_941_506",
    TURBO_1K_FIXTURE.metadata.sha256,
    TURBO_1K_FIXTURE.tensors.sha256,
    TURBO_1K_FIXTURE.output.sha256,
    "whole_safetensors_file_verified: false",
    "first-DMD fixture does not expose",
  ]) {
    if (!fixtureSource.includes(required)) failures.push(`fixture reader omits ${required}`);
  }
  for (const digest of [
    TURBO_FIXTURE.metadata.sha256,
    TURBO_FIXTURE.tensors.sha256,
    TURBO_FIXTURE.output.sha256,
  ]) {
    if (!parityWorkflowSource.includes(digest)) {
      failures.push(`release parity workflow omits ${digest}`);
    }
  }
  if (
    !parityWorkflowSource.includes(
      "Browser Turbo 1024 first-DMD packed-F16 dense-F32-per-stage diagnostic",
    )
  ) {
    failures.push("release parity workflow labels the packed-F16 first-DMD policy incorrectly");
  }
  const ordinaryRenderedGate =
    "BURN_IMAGE_RENDERED_TURBO_QWEN_BLOCK0_EXECUTION_MODE=ordinary";
  if (occurrenceCount(parityWorkflowSource, ordinaryRenderedGate) !== 2) {
    failures.push(
      "release parity workflow must pin both rendered Turbo gates to ordinary block-0 execution",
    );
  }
  if (parityWorkflowSource.includes("BURN_IMAGE_RENDERED_TURBO_QWEN_BLOCK0_EXECUTION_MODE=serialized-diagnostic")) {
    failures.push("release parity workflow must not feed serialized diagnostics into release gates");
  }
  if (parityFixtureSource.includes("TURBO_FIRST_DMD")) {
    failures.push("strict 1.5K fixture reader was coupled to the Turbo diagnostic");
  }
  for (const required of [
    "headless=turbo-first-dmd",
    "summarizeTurboFirstDmdWebGpuCalls",
    "validateTurboFirstDmdReport",
    "hardware_adapter_attestation",
    "BURN_IMAGE_TURBO_FIRST_DMD_VALIDATE_ONLY",
    "quotaAwareAllocationProbe",
    "inspectChromeSharedMemory",
    "inspectChromeProfileStorage",
    "inspectTurboFirstDmdStorageAdmission",
    "selectTurboFirstDmdChromeSharedMemoryPolicy",
    "turboFirstDmdChromeLaunchEvidence",
    "validateArtifactBundleTransport",
    "transportTelemetryFiles",
    "physical_transport_part",
    'join(wwwOut, "burn-image-icon.png")',
    'content_type: "image/png"',
    "browser package server failed the exact icon MIME/size/SHA-256 self-test",
  ]) {
    if (!harnessSource.includes(required)) failures.push(`Node harness omits ${required}`);
  }
  for (const required of [
    "ARTIFACT_TRANSPORT_TARGET_PART_BYTES = 20 * 1024 * 1024",
    "ARTIFACT_TRANSPORT_HARD_MAX_PART_BYTES = 25_000_000",
    "metadata/transport-layout.json",
    "verified-by-browser-runtime-before-use",
    "explicit-legacy-direct-layout-no-browser-cache-shard-claim",
  ]) {
    if (!transportContractSource?.includes(required)) {
      failures.push(`transport contract omits ${required}`);
    }
  }
  if (
    !/if \(sharedMemoryPolicy\.disable_dev_shm_usage\) \{\s*arguments_\.push\("--disable-dev-shm-usage"\);\s*\}/.test(
      harnessSource,
    ) ||
    occurrenceCount(harnessSource, '"--disable-dev-shm-usage"') !== 1
  ) {
    failures.push("Node harness must apply --disable-dev-shm-usage only after quota-aware fallback");
  }
  if (
    !/mkdtemp\(join\(outputDir, "burn-image-turbo-first-dmd-chrome-"\)\)/.test(
      harnessSource,
    )
  ) {
    failures.push("Node harness must place the model Chrome profile under outputDir");
  }
  for (const required of [
    "if (!validateOnly && !explicitOutputDir)",
    "chromeSharedMemory.launch_admitted !== true",
    "chromeProfileStorage.admitted !== true",
    "tempDirectory,",
  ]) {
    if (!harnessSource.includes(required)) {
      failures.push(`Node harness omits fail-closed Chrome storage admission ${required}`);
    }
  }
  for (const required of [
    "detached: true",
    "process.kill(-processGroupId",
    "process_group_exited: exited",
    "maxRetries: 8",
    "retryDelay: 100",
  ]) {
    if (!harnessSource.includes(required)) {
      failures.push(`Node harness omits hardened Chrome cleanup marker ${required}`);
    }
  }
  for (const required of [
    "SERVER_CLOSE_TIMEOUT_MS",
    "server.closeAllConnections?.()",
    "socket.destroy()",
    "terminalOutput.server_cleanup",
  ]) {
    if (!harnessSource.includes(required)) {
      failures.push(`Node harness omits bounded HTTP cleanup marker ${required}`);
    }
  }
  for (const required of [
    "CDP_CALL_TIMEOUT_MS",
    "\n  failPending(error) {",
    "clearTimeout(pending.timeout)",
    "this.pending.clear()",
  ]) {
    if (!harnessSource.includes(required)) {
      failures.push(`Node harness omits bounded CDP lifecycle marker ${required}`);
    }
  }
  const cdpClassStart = harnessSource.indexOf("class Cdp {");
  const cdpClassEnd = harnessSource.indexOf("\nasync function openPage(", cdpClassStart);
  const cdpClass =
    cdpClassStart >= 0 && cdpClassEnd > cdpClassStart
      ? harnessSource.slice(cdpClassStart, cdpClassEnd)
      : "";
  for (const eventName of ["error", "close"]) {
    const marker = `this.socket.addEventListener("${eventName}"`;
    if (occurrenceCount(cdpClass, marker) < 2) {
      failures.push(`Node harness omits post-open CDP lifecycle handler ${marker}`);
    }
  }
  if (
    !/call\(method, params = \{\}\) \{[\s\S]*?const timeout = setTimeout\([\s\S]*?CDP_CALL_TIMEOUT_MS/.test(
      cdpClass,
    )
  ) {
    failures.push("Node harness omits a bounded timeout for each CDP call");
  }
  const failureStart = harnessSource.indexOf(
    '} catch (error) {\n  const detail = error instanceof Error',
  );
  const failureEnd = harnessSource.indexOf("} finally {", failureStart);
  const failurePath =
    failureStart >= 0 && failureEnd > failureStart
      ? harnessSource.slice(failureStart, failureEnd)
      : null;
  if (!failurePath?.includes("...(inputEvidence ?? {})")) {
    failures.push("Node harness failure evidence omits already-validated input identities");
  }
  return failures;
}
