import {
  ARTIFACT_TRANSPORT_HARD_MAX_PART_BYTES,
  ARTIFACT_TRANSPORT_LAYOUT_PATH,
  ARTIFACT_TRANSPORT_TARGET_PART_BYTES,
} from "./artifact_transport_contract.mjs";

export const BACKEND_EVENT_NAME = "burn-image-backend";
export const RUNTIME_EVENT_NAME = "burn-image-runtime";
export const PROGRESS_EVENT_NAME = "burn-image-progress";
export const UI_CONTRACT_EVENT_NAME = "burn-image-ui-contract";
export const OUTPUT_READY_EVENT_NAME = "burn-image-output-ready";
export const GENERIC_WEB_REQUIRED_DEVICE_FEATURES = Object.freeze([]);
export const BOOGU_WEB_REQUIRED_DEVICE_FEATURES = Object.freeze(["timestamp-query"]);
export const BROWSER_WEBGPU_ENABLED_FEATURE_BASELINE = Object.freeze([
  "core-features-and-limits",
]);
export const TURBO_MODEL_ID = "Boogu/Boogu-Image-0.1-Turbo";
export const TURBO_PRODUCTION_CONTENT_DIGEST =
  "32b2f0a972d7c00e4bc914f949dcf15195c10c428be456330a168a556576138a";
export const TURBO_Q4_PRODUCTION_CONTENT_DIGEST =
  "012237fb5e14c52188632ea220c043e5a9d59eaa243970100e7db35048942081";
export const TURBO_Q4_QWEN_COMPONENT_CONTENT_DIGEST =
  "d3e332ebd710d87fa6a2ae97eef3302f5c9f5e7d3f4e27675f0c4c4f5a31c5de";
export const TURBO_Q4_VAE_COMPONENT_CONTENT_DIGEST =
  "fcd840d188556b3f8aa3f5ffd240a240ac94420285a4c47676104dead5183a52";
export const TURBO_Q4_RESIDENT_BACKEND =
  "burn-webgpu/browser-resident-packed-q4s-block-up-to-128/request-scoped-surface-acquire-suspended";
export const TURBO_Q4_RESIDENT_WEIGHT_TRAFFIC_CONTRACT =
  "eager-preload/qwen+vae+denoiser/resident-q4s-matrices+embedding+packed-f16-convolutions+f32-auxiliaries/zero-inference-artifact-transfers/no-model-unload";
export const TURBO_Q4_RESIDENT_MULTI_REQUEST_POLICY =
  "same-page/same-engine/two-request/resident-q4s/exact-cache-audit/zero-request-artifact-io/rendered-bevy-surface-gated";
export const TURBO_Q4_STRICT_DEVICE_BYTES_EXCLUSIVE = 16_000_000_000;
export const LOW_VRAM_PUBLIC_SELECTOR =
  "low-vram-preloaded-packed-f16-dense-f32-per-stage-denoiser";
export const LOW_VRAM_BACKEND =
  "burn-webgpu/browser-low-vram-preloaded-packed-f16-dense-f32-per-stage-denoiser/request-scoped-packed-cache-evicted-before-vae/request-scoped-surface-acquire-suspended";
export const SURFACE_INFERENCE_POLICY =
  "request-scoped-surface-acquire-suspended/primary-window-cameras-inactive-before-runtime-submit/exact-state-restored-after-terminal-before-output-ready";
export const SURFACE_INFERENCE_SUSPENDED_EVENT = "surface_inference_suspended";
export const SURFACE_INFERENCE_RESUMED_EVENT = "surface_inference_resumed";
export const LOW_VRAM_DEVICE_CAP_BYTES = 32_000_000_000;
export const TURBO_DENOISER_STORAGE_POLICY =
  "authenticated-compact-f16/padded-u32-retained/dense-f32-per-semantic-stage";
export const TURBO_DENOISER_QUANTIZED_LOAD_POLICY =
  "not-applicable-packed-f16-storage";
export const TURBO_DENOISER_QUANTIZED_EXECUTION_POLICY =
  "not-applicable-packed-f16-storage";
export const TURBO_DENOISER_LINEAR_EXECUTION_POLICY =
  "packed-f16-storage/device-widen-f32-per-semantic-stage/dense-f32-matmul";
export const TURBO_QWEN_TEXT_LAYER_ALLOCATION_POLICY =
  "backend-exact-size-persistent-pool-per-text-layer";
export const TURBO_QWEN_TEXT_BLOCK_LOAD_SYNCHRONIZATION_POLICY =
  "async-source-sync-after-every-text-block-load-before-forward/plus-post-forward-sync";
export const TURBO_QWEN_TEXT_LAYER_SUBMISSION_POLICY =
  "explicit-pre-forward-upload-barrier/explicit-post-forward-barrier/bounded-task-batches/per-submit-error-scopes";
export const TURBO_QWEN_BLOCK0_SERIALIZED_DIAGNOSTIC_MODE = "serialized-diagnostic";
export const TURBO_QWEN_BLOCK0_ORDINARY_MODE = "ordinary";
export const RENDERED_SURFACE_SMOKE_TEST = "burn_image_browser_rendered_surface_smoke";
export const TURBO_RENDERED_ORDINARY_SMOKE_TEST =
  "burn_image_browser_rendered_turbo_1024_smoke";
export const TURBO_RENDERED_ORDINARY_MULTI_REQUEST_TEST =
  "burn_image_browser_rendered_turbo_1024_multi_request_qualification";
export const TURBO_RENDERED_Q4_RESIDENT_MULTI_REQUEST_TEST =
  "burn_image_browser_rendered_turbo_q4_1024_resident_multi_request_qualification";
export const TURBO_RENDERED_SERIALIZED_DIAGNOSTIC_TEST =
  "burn_image_browser_rendered_turbo_1024_qwen_block0_serialized_diagnostic";
export const TURBO_RENDERED_SERIALIZED_MULTI_REQUEST_DIAGNOSTIC_TEST =
  "burn_image_browser_rendered_turbo_1024_multi_request_qwen_block0_serialized_diagnostic";
export const TURBO_LOW_VRAM_WEIGHT_TRAFFIC_CONTRACT =
  "persistent-transport-part-cache/qwen+vae+packed-f16-denoiser-rehydrated-before-each-request/zero-dmd-artifact-transfers/zero-repeat-network-required/request-scoped-packed-cache-evicted-before-vae/dense-f32-materialized-per-semantic-stage";
export const TURBO_PACKED_F16_CACHED_STAGES = 46;
export const TURBO_PACKED_F16_CACHED_OBJECTS = 106;
export const TURBO_PACKED_F16_CACHED_TENSORS = 912;
export const TURBO_PACKED_F16_RESOURCE_PLAN = Object.freeze({
  qwen_text_layer_allocation_policy: TURBO_QWEN_TEXT_LAYER_ALLOCATION_POLICY,
  qwen_text_block_load_synchronization_policy:
    TURBO_QWEN_TEXT_BLOCK_LOAD_SYNCHRONIZATION_POLICY,
  qwen_text_layer_submission_policy: TURBO_QWEN_TEXT_LAYER_SUBMISSION_POLICY,
  qwen_text_layer_persistent_pool_requires_measured_gpu_gate: true,
  authenticated_artifact_bytes: 19_870_166_528,
  canonical_compact_f16_payload_bytes: 19_869_996_096,
  retained_packed_f16_denoiser_bytes: 19_870_010_624,
  inserted_padding_elements: 7_264,
  padded_f16_elements: 9_935_005_312,
  expected_stage_count: TURBO_PACKED_F16_CACHED_STAGES,
  expected_object_count: TURBO_PACKED_F16_CACHED_OBJECTS,
  expected_tensor_count: TURBO_PACKED_F16_CACHED_TENSORS,
  max_packed_stage_bytes: 876_827_328,
  max_materialized_stage_f32_bytes: 1_753_654_656,
  max_packed_object_bytes: 254_251_904,
  max_materialized_object_f32_bytes: 508_503_808,
  materialized_f32_bytes_per_dmd_step: 39_740_021_248,
  preload_workspace_bytes: 2_434_252_800,
  preload_peak_bytes: 22_304_263_424,
  activation_reserve_bytes: 4_868_505_600,
  conservative_planned_device_bytes: 26_492_170_880,
  strict_device_cap_bytes: LOW_VRAM_DEVICE_CAP_BYTES,
  expected_stage_materializations_per_request: 184,
  expected_object_unpacks_per_request: 424,
  expected_packed_read_bytes_per_request: 79_480_042_496,
  expected_f32_write_bytes_per_request: 158_960_084_992,
  on_device_quantized_execution_claimed: false,
});
export const TURBO_PACKED_F16_PRELOAD_MESSAGE =
  "Verifying and preloading all 46 Turbo packed-F16 denoiser stages before inference";
export const TURBO_DENOISER_PRELOAD_TRAFFIC = Object.freeze({
  object_reads: 106,
  object_read_bytes: 19_870_166_528,
  range_reads: 1_001,
  range_read_bytes: 19_870_166_528,
  verified_objects: 106,
  cache_lookups: 1_001,
  cache_hits: 0,
  cache_misses: 1_001,
  cache_read_bytes: 0,
  network_requests: 1_001,
  network_response_bytes: 19_870_166_528,
  cache_writes: 1_001,
  cache_write_bytes: 19_870_166_528,
  cache_evictions: 0,
  cache_evicted_entries: 0,
  cache_invalid_entries: 0,
  integrity_refetches: 0,
});
export const TURBO_GENERATE_REQUEST_TRAFFIC = Object.freeze({
  object_reads: 80,
  object_read_bytes: 15_235_984_896,
  range_reads: 750,
  range_read_bytes: 15_235_984_896,
  verified_objects: 80,
  cache_lookups: 750,
  cache_hits: 0,
  cache_misses: 750,
  cache_read_bytes: 0,
  network_requests: 750,
  network_response_bytes: 15_235_984_896,
  cache_writes: 750,
  cache_write_bytes: 15_235_984_896,
  cache_evictions: 0,
  cache_evicted_entries: 0,
  cache_invalid_entries: 0,
  integrity_refetches: 0,
});
export const TURBO_REPEAT_GENERATE_REQUEST_TRAFFIC = Object.freeze({
  object_reads: 186,
  object_read_bytes: 35_106_151_424,
  range_reads: 1_751,
  range_read_bytes: 35_106_151_424,
  verified_objects: 186,
  cache_lookups: 1_751,
  cache_hits: 1_751,
  cache_misses: 0,
  cache_read_bytes: 35_106_151_424,
  network_requests: 0,
  network_response_bytes: 0,
  cache_writes: 0,
  cache_write_bytes: 0,
  cache_evictions: 0,
  cache_evicted_entries: 0,
  cache_invalid_entries: 0,
  integrity_refetches: 0,
});
export const TURBO_DENOISER_REHYDRATION_TRAFFIC = Object.freeze({
  object_reads: 106,
  object_read_bytes: 19_870_166_528,
  range_reads: 1_001,
  range_read_bytes: 19_870_166_528,
  verified_objects: 106,
  cache_lookups: 1_001,
  cache_hits: 1_001,
  cache_misses: 0,
  cache_read_bytes: 19_870_166_528,
  network_requests: 0,
  network_response_bytes: 0,
  cache_writes: 0,
  cache_write_bytes: 0,
  cache_evictions: 0,
  cache_evicted_entries: 0,
  cache_invalid_entries: 0,
  integrity_refetches: 0,
});
export const TURBO_DMD_ZERO_IO = Object.freeze({
  object_reads: 0,
  object_read_bytes: 0,
  range_reads: 0,
  range_read_bytes: 0,
  verified_objects: 0,
  cache_lookups: 0,
  cache_hits: 0,
  cache_misses: 0,
  cache_read_bytes: 0,
  network_requests: 0,
  network_response_bytes: 0,
  cache_writes: 0,
  cache_write_bytes: 0,
  cache_evictions: 0,
  cache_evicted_entries: 0,
  cache_invalid_entries: 0,
  integrity_refetches: 0,
});
export const TURBO_PACKED_F16_REQUEST_LIFECYCLE = Object.freeze({
  cache_state: "ready",
  cache_ready: true,
  cached_stages: TURBO_PACKED_F16_CACHED_STAGES,
  cached_objects: TURBO_PACKED_F16_CACHED_OBJECTS,
  cached_tensors: TURBO_PACKED_F16_CACHED_TENSORS,
  cached_bytes: TURBO_PACKED_F16_RESOURCE_PLAN.retained_packed_f16_denoiser_bytes,
  authenticated_artifact_bytes: TURBO_PACKED_F16_RESOURCE_PLAN.authenticated_artifact_bytes,
  packed_upload_bytes: TURBO_PACKED_F16_RESOURCE_PLAN.retained_packed_f16_denoiser_bytes,
  stage_materializations:
    TURBO_PACKED_F16_RESOURCE_PLAN.expected_stage_materializations_per_request,
  object_unpacks: TURBO_PACKED_F16_RESOURCE_PLAN.expected_object_unpacks_per_request,
  packed_read_bytes: TURBO_PACKED_F16_RESOURCE_PLAN.expected_packed_read_bytes_per_request,
  f32_write_bytes: TURBO_PACKED_F16_RESOURCE_PLAN.expected_f32_write_bytes_per_request,
  preload_attempt_count: 1,
  failure_count: 0,
  dmd_artifact_traffic: TURBO_DMD_ZERO_IO,
  synchronization_pending: false,
  matches_plan: true,
});
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
export const GPU_INTERVAL_AGGREGATION_POLICY =
  "sum-per-interval-all-chrome-gpu-pids-deduplicated-by-gpu-and-pid";
export const TURBO_CDP_REQUEST_NETWORK_POLICY =
  "exact-modular-bundle-206-responses-in-run-started-to-output-or-failure-window";
export const TURBO_CDP_PRELOAD_NETWORK_POLICY =
  "exact-modular-bundle-206-responses-in-denoiser-preparing-to-preload-complete-window";
export const TURBO_CDP_DMD_NETWORK_POLICY =
  "zero-exact-modular-bundle-responses-in-four-step-dmd-window";
export const TURBO_MULTI_REQUEST_POLICY =
  "same-page-same-engine-two-sequential-ordinary-generate-requests";
export const TURBO_SECOND_REQUEST_RUN_READY_POLICY =
  "exact-seed-change-in-second-ui-partition/partition-local-ready-preferred/exact-last-pre-boundary-post-request-ready-reused";
export const TURBO_DMD_RUNTIME_ZERO_IO_POLICY =
  "successful-output-bound-to-fail-closed-runtime-dmd-traffic-delta";
export const TURBO_PACKED_F16_PRE_DMD_INPUT_SCOPE =
  "rendered-model-smoke/ordinary-turbo-packed-f16/pre-dmd-input-readback";
export const TURBO_PACKED_F16_QWEN_HANDOFF_POLICY =
  "qwen-per-stage-cleanup-disabled/exact-f32-instruction-host-handoff/async-webgpu-sync/backend-memory-cleanup/async-webgpu-sync/exact-f32-reupload/post-upload-digest-verify/packed-cache-reaudit";
export const TURBO_PACKED_F16_DMD_VAE_HANDOFF_POLICY =
  "exact-f32-final-latent-host-handoff/drop-dmd-input-handles/pre-clear-async-webgpu-sync/clear-packed-source-wrapper-rope/async-webgpu-sync/backend-memory-cleanup/async-webgpu-sync/require-empty-packed-cache/exact-f32-reupload/post-upload-digest-verify";
export const TURBO_PACKED_F16_NEXT_REQUEST_REHYDRATION_POLICY =
  "ensure-preloaded-low-vram-denoiser/verified-persistent-cache-storage/bounded-object-replay";
export const TURBO_PACKED_F16_QWEN_PRE_HANDOFF_SCOPE =
  "rendered-model-smoke/ordinary-turbo-packed-f16/qwen-pre-handoff-readback";
export const TURBO_PACKED_F16_QWEN_BLOCK0_EXECUTION_SCOPE =
  "rendered-model-smoke/ordinary-turbo-packed-f16/qwen-block-00-serialized-operation-readbacks";
export const TURBO_PACKED_F16_QWEN_BLOCK0_BOUNDARIES = Object.freeze([
  Object.freeze({ boundary: "layer_input", tensor_kind: "activation" }),
  Object.freeze({ boundary: "input_layernorm_gamma", tensor_kind: "parameter-sentinel" }),
  Object.freeze({ boundary: "identity_add_canary", tensor_kind: "activation" }),
  Object.freeze({ boundary: "input_norm", tensor_kind: "activation" }),
  Object.freeze({ boundary: "attention_output", tensor_kind: "activation" }),
  Object.freeze({ boundary: "first_residual", tensor_kind: "activation" }),
  Object.freeze({ boundary: "post_attention_norm", tensor_kind: "activation" }),
  Object.freeze({ boundary: "mlp_output", tensor_kind: "activation" }),
  Object.freeze({ boundary: "final_residual_output", tensor_kind: "activation" }),
]);
export const TURBO_PACKED_F16_QWEN_BLOCK0_POST_SYNC_SCOPE =
  "rendered-model-smoke/ordinary-turbo-packed-f16/qwen-block-00-immediate-post-sync-readback";
export const TURBO_PACKED_F16_QWEN_POST_HANDOFF_SCOPE =
  "rendered-model-smoke/ordinary-turbo-packed-f16/qwen-post-handoff-readback";
export const TURBO_PACKED_F16_QWEN_STAGE_OUTPUT_COUNT = 38;
export const TURBO_PACKED_F16_QWEN_HOST_EMBEDDING_POLICY =
  "authenticated-full-f16-row-objects/host-token-row-select/f16-to-f32/one-compact-upload";
export const TURBO_PACKED_F16_QWEN_EMBEDDING_EXECUTION_POLICY =
  "authenticated-full-f16-row-objects/host-token-row-select/f16-to-f32/one-compact-upload/immediate-device-readback-before-text";
export const TURBO_PACKED_F16_QWEN_HOST_EMBEDDING_SHA256 =
  "a6aa0501f1d6f5a622934ee10a64b526843f723937f6d5abd96058b29ea8b6fe";
export const TURBO_PACKED_F16_QWEN_HOST_EMBEDDING_PLAN = Object.freeze({
  expected_chunk_count: 6,
  expected_object_count: 6,
  authenticated_object_bytes: 1_244_662_784,
  authenticated_f16_payload_bytes: 1_244_659_712,
});

/**
 * Return the exact outer report identity for a rendered-surface attempt.
 *
 * Serialized block-0 execution is a localization diagnostic. Even when its model execution
 * succeeds, it must never reuse the ordinary smoke or multi-request qualification identity.
 */
export function renderedSurfaceReportIdentity({
  modelMode,
  multiRequestModelMode,
  q4ResidentMode = false,
  qwenBlock0ExecutionMode,
  ok,
}) {
  if (typeof ok !== "boolean") throw new Error("rendered report success state is not boolean");
  if (!modelMode) {
    return {
      test: RENDERED_SURFACE_SMOKE_TEST,
      claim: ok
        ? "headful Bevy WebGPU rendered-surface smoke; not numerical model parity"
        : "failed headful surface attempt; no rendered-surface or numerical parity claim",
    };
  }
  if (
    ![
      TURBO_QWEN_BLOCK0_ORDINARY_MODE,
      TURBO_QWEN_BLOCK0_SERIALIZED_DIAGNOSTIC_MODE,
    ].includes(qwenBlock0ExecutionMode)
  ) {
    throw new Error(`unknown rendered Turbo Qwen block-0 execution mode ${qwenBlock0ExecutionMode}`);
  }
  if (q4ResidentMode) {
    if (!multiRequestModelMode) {
      throw new Error("rendered resident-Q4 qualification requires two requests");
    }
    return {
      test: TURBO_RENDERED_Q4_RESIDENT_MULTI_REQUEST_TEST,
      claim: ok
        ? "same-page same-engine two-request rendered Bevy UI Turbo 1024 resident-Q4 warm-session qualification; not numerical parity"
        : "failed rendered resident-Q4 warm-session attempt; no model-smoke or numerical-parity claim",
    };
  }
  if (qwenBlock0ExecutionMode === TURBO_QWEN_BLOCK0_SERIALIZED_DIAGNOSTIC_MODE) {
    return {
      test: multiRequestModelMode
        ? TURBO_RENDERED_SERIALIZED_MULTI_REQUEST_DIAGNOSTIC_TEST
        : TURBO_RENDERED_SERIALIZED_DIAGNOSTIC_TEST,
      claim: ok
        ? multiRequestModelMode
          ? "diagnostic-only same-page same-engine two-request serialized Qwen block-0 boundary localization in rendered Bevy UI Turbo 1024; not release-qualification, output-quality, model-parity, or numerical-parity evidence"
          : "diagnostic-only serialized Qwen block-0 boundary localization in rendered Bevy UI Turbo 1024; not model-smoke, release-qualification, output-quality, model-parity, or numerical-parity evidence"
        : "failed serialized-diagnostic Qwen block-0 localization attempt; no model-smoke, release-qualification, output-quality, model-parity, or numerical-parity claim",
    };
  }
  return {
    test: multiRequestModelMode
      ? TURBO_RENDERED_ORDINARY_MULTI_REQUEST_TEST
      : TURBO_RENDERED_ORDINARY_SMOKE_TEST,
    claim: ok
      ? multiRequestModelMode
        ? "same-page same-engine two-request ordinary rendered Bevy UI Turbo 1024 low-VRAM qualification; not numerical parity"
        : "ordinary rendered Bevy UI Turbo 1024 preloaded packed-F16 storage / dense-F32-per-semantic-stage low-VRAM real-model smoke; not numerical parity"
      : "failed ordinary-UI model attempt; no model-smoke or numerical parity claim",
  };
}
export const TURBO_PACKED_F16_QWEN_HOST_EMBEDDING_F16_BYTES = 368_640;
export const TURBO_PACKED_F16_QWEN_HOST_EMBEDDING_F32_BYTES = 737_280;

const SOFTWARE_ADAPTER = /(swiftshader|llvmpipe|lavapipe|software adapter|warp)/i;
const NVIDIA_ADAPTER = /nvidia/i;
const GPU_TERMINAL_DIAGNOSTIC =
  /(WebgpuSwapChainTexture|device[ _-]*lost|lost[ _-]*device|GPU process[^\n]*(crash|exit|lost)|context[^\n]*lost|(?:webgpu|wgpu|dawn|vulkan)[^\n]*(?:error|failed|lost|invalid))/i;
const RUNTIME_ADAPTER_SOURCE = "instrumented-rendered-page-navigator-gpu-request-adapter";
export const RENDERED_LAUNCH_READINESS_PROBE_POLICY =
  "pre-navigation-launch-readiness-only-not-runtime-adapter-or-device-proof";
export const CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES = 256 * 1024 * 1024;
export const CHROME_SHARED_MEMORY_DEV_SHM_POLICY =
  "linux-dev-shm-statfs-and-quota-aware-probe-admitted";
export const CHROME_SHARED_MEMORY_TMP_FALLBACK_POLICY =
  "linux-temp-fallback-dev-shm-not-admitted";
export const CHROME_SHARED_MEMORY_PLATFORM_DEFAULT_POLICY =
  "non-linux-platform-default";

function exactNonNegativeInteger(value) {
  if (Number.isSafeInteger(value) && value >= 0) return BigInt(value);
  if (typeof value === "string" && /^(0|[1-9][0-9]*)$/.test(value)) {
    return BigInt(value);
  }
  return null;
}

export function selectChromeSharedMemoryPolicy({
  platform,
  devShm,
  tempPath = null,
  minimumHeadroomBytes = CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES,
}) {
  if (!Number.isSafeInteger(minimumHeadroomBytes) || minimumHeadroomBytes <= 0) {
    throw new Error("Chrome shared-memory minimum headroom must be a positive safe integer");
  }
  if (platform !== "linux") {
    return {
      policy: CHROME_SHARED_MEMORY_PLATFORM_DEFAULT_POLICY,
      selected_backing: "platform-default",
      selected_path: null,
      disable_dev_shm_usage: false,
      minimum_admitted_headroom_bytes: minimumHeadroomBytes,
      dev_shm_admitted: null,
      dev_shm_rejections: [],
    };
  }

  const rejections = [];
  if (devShm?.exists !== true) rejections.push("missing");
  if (devShm?.writable !== true) rejections.push("not-writable");
  const availableBytes = exactNonNegativeInteger(devShm?.statfs?.available_bytes);
  if (availableBytes === null) {
    rejections.push("statfs-available-unknown");
  } else if (availableBytes < BigInt(minimumHeadroomBytes)) {
    rejections.push("statfs-available-below-minimum");
  }
  const allocationProbe = devShm?.quota_aware_allocation_probe;
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

  if (rejections.length === 0) {
    return {
      policy: CHROME_SHARED_MEMORY_DEV_SHM_POLICY,
      selected_backing: "dev-shm",
      selected_path: devShm.path,
      disable_dev_shm_usage: false,
      minimum_admitted_headroom_bytes: minimumHeadroomBytes,
      dev_shm_admitted: true,
      dev_shm_rejections: [],
    };
  }
  return {
    policy: CHROME_SHARED_MEMORY_TMP_FALLBACK_POLICY,
    selected_backing: "temp-directory",
    selected_path: typeof tempPath === "string" ? tempPath : null,
    disable_dev_shm_usage: true,
    minimum_admitted_headroom_bytes: minimumHeadroomBytes,
    dev_shm_admitted: false,
    dev_shm_rejections: rejections,
  };
}

export function renderedChromeLaunchEvidence({
  executable,
  arguments: chromeArguments,
  profile,
  sharedMemory,
}) {
  return {
    chrome_executable: typeof executable === "string" ? executable : null,
    chrome_arguments: Array.isArray(chromeArguments) ? [...chromeArguments] : null,
    chrome_profile: typeof profile === "string" ? profile : null,
    chrome_shared_memory: sharedMemory ?? null,
  };
}

function report(value) {
  const rendered = JSON.stringify(value);
  return rendered === undefined ? String(value) : rendered;
}

function positiveInteger(value) {
  return Number.isSafeInteger(value) && value > 0;
}

function positiveFinite(value) {
  return Number.isFinite(value) && value > 0;
}

function nonNegativeInteger(value) {
  return Number.isSafeInteger(value) && value >= 0;
}

const U64_MAX_DECIMAL = "18446744073709551615";

export function isCanonicalU64DecimalString(value) {
  if (typeof value !== "string" || !/^(?:0|[1-9][0-9]*)$/.test(value)) return false;
  return (
    value.length < U64_MAX_DECIMAL.length ||
    (value.length === U64_MAX_DECIMAL.length && value <= U64_MAX_DECIMAL)
  );
}

export function outputJobIdMatchesNumericRunId(jobId, runId) {
  return (
    isCanonicalU64DecimalString(jobId) &&
    nonNegativeInteger(runId) &&
    jobId === String(runId)
  );
}

function sameRunId(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function sameJsonValue(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

export function validateTurboRunReadyUiContract(uiContract, label = "Turbo Run readiness") {
  const failures = [];
  if (!uiContract || typeof uiContract !== "object" || Array.isArray(uiContract)) {
    return [`${label} UI contract is missing`];
  }
  if (uiContract.event !== "ready") failures.push(`${label} UI event is not ready`);
  if (uiContract.model !== TURBO_MODEL_ID) {
    failures.push(`${label} model=${report(uiContract.model)}, expected exact Turbo`);
  }
  if (uiContract.width !== 1024 || uiContract.height !== 1024) {
    failures.push(
      `${label} dimensions=${report([uiContract.width, uiContract.height])}, expected 1024x1024`,
    );
  }
  if (uiContract.run_enabled !== true) failures.push(`${label} Run control is disabled`);
  return failures;
}

/**
 * Resolve the exact Run control contract after the second request's seed edit.
 *
 * The Bevy UI event intentionally suppresses duplicate `ready` details. Changing only the seed
 * therefore need not emit another `ready` event. The fallback is safe only when it is exactly the
 * last ready event before the preserved second-request UI boundary and still attests the expected
 * model, dimensions, and enabled Run control.
 */
export function resolveTurboSecondRequestRunReadyUiContract({
  uiEvents,
  uiStartIndex,
  seedChangedEvent,
  postRequestUiContract,
}) {
  if (!Array.isArray(uiEvents)) {
    throw new Error("second-request UI events are missing");
  }
  if (
    !nonNegativeInteger(uiStartIndex) ||
    uiStartIndex === 0 ||
    uiStartIndex > uiEvents.length
  ) {
    throw new Error(`second-request UI start index is invalid: ${report(uiStartIndex)}`);
  }
  if (seedChangedEvent?.event !== "seed_changed" || seedChangedEvent?.value !== "1") {
    throw new Error(
      `second-request seed_changed evidence is not exact seed 1: ${report(seedChangedEvent)}`,
    );
  }

  const partition = uiEvents.slice(uiStartIndex);
  const seedChangedOffset = partition.findIndex((event) =>
    sameJsonValue(event, seedChangedEvent),
  );
  if (seedChangedOffset < 0) {
    throw new Error("exact seed_changed evidence is outside the second-request UI partition");
  }

  const lastPreBoundaryReadyIndex = uiEvents
    .slice(0, uiStartIndex)
    .findLastIndex((event) => event?.event === "ready");
  if (lastPreBoundaryReadyIndex < 0) {
    throw new Error("post-request Run readiness has no pre-boundary ready event");
  }
  const lastPreBoundaryReady = uiEvents[lastPreBoundaryReadyIndex];
  if (!sameJsonValue(lastPreBoundaryReady, postRequestUiContract)) {
    throw new Error(
      "post-request Run readiness fallback is stale or differs from the last pre-boundary ready event",
    );
  }
  const fallbackFailures = validateTurboRunReadyUiContract(
    postRequestUiContract,
    "post-request Run readiness fallback",
  );
  if (postRequestUiContract?.save_enabled !== true) {
    fallbackFailures.push("post-request Run readiness fallback Save PNG control is disabled");
  }
  if (fallbackFailures.length > 0) {
    throw new Error(fallbackFailures.join("; "));
  }

  const localReadyOffset = partition.findLastIndex((event) => event?.event === "ready");
  const usedPostRequestFallback = localReadyOffset < 0;
  const uiContract = usedPostRequestFallback
    ? postRequestUiContract
    : partition[localReadyOffset];
  const selectedFailures = validateTurboRunReadyUiContract(
    uiContract,
    usedPostRequestFallback
      ? "post-request Run readiness fallback"
      : "second-request partition-local Run readiness",
  );
  if (selectedFailures.length > 0) {
    throw new Error(selectedFailures.join("; "));
  }

  return {
    uiContract,
    evidence: {
      policy: TURBO_SECOND_REQUEST_RUN_READY_POLICY,
      source: usedPostRequestFallback
        ? "exact-last-pre-boundary-post-request-ready"
        : "second-request-ui-partition",
      ui_start_index: uiStartIndex,
      ui_event_count_at_resolution: uiEvents.length,
      seed_changed_event_index: uiStartIndex + seedChangedOffset,
      run_ready_event_index: usedPostRequestFallback
        ? lastPreBoundaryReadyIndex
        : uiStartIndex + localReadyOffset,
      duplicate_ready_after_seed_change_required: false,
      fallback_exact_last_pre_boundary_ready: true,
      selected_ui_contract: uiContract,
    },
  };
}

export function validateRequestScopedSurfaceGate(evidence, label = "Generate request") {
  const failures = [];
  const runtimeEvents = evidence?.runtime_events;
  const progressEvents = evidence?.progress_events;
  const windows = evidence?.surface_texture_gate_windows;
  const violations = evidence?.surface_texture_gate_violation_calls;
  if (!Array.isArray(runtimeEvents)) {
    return [`${label} surface gate runtime events are missing`];
  }
  if (!Array.isArray(progressEvents)) {
    return [`${label} surface gate progress events are missing`];
  }
  if (!Array.isArray(windows)) {
    failures.push(`${label} compact surface gate windows are missing`);
  }
  if (!Array.isArray(violations)) {
    failures.push(`${label} surface gate violation-call evidence is missing`);
  } else if (violations.length !== 0) {
    failures.push(
      `${label} recorded ${violations.length} GPUCanvasContext.getCurrentTexture violation call(s) while gated`,
    );
  }
  if (
    evidence?.surface_texture_gate_windows_overflow !== 0 ||
    evidence?.surface_texture_gate_violation_calls_overflow_start !==
      evidence?.surface_texture_gate_violation_calls_overflow_end ||
    evidence?.surface_texture_gate_violation_calls_overflow_end !== 0
  ) {
    failures.push(`${label} compact surface gate evidence overflowed`);
  }
  if (evidence?.surface_texture_gate_overlap_count !== 0) {
    failures.push(`${label} surface gate instrumentation observed overlapping or unpaired windows`);
  }
  const acquisitionCountStart = evidence?.surface_texture_acquisition_count_start;
  const acquisitionCountEnd = evidence?.surface_texture_acquisition_count_end;
  if (
    !nonNegativeInteger(acquisitionCountStart) ||
    !nonNegativeInteger(acquisitionCountEnd) ||
    acquisitionCountEnd < acquisitionCountStart
  ) {
    failures.push(`${label} surface acquisition counter boundary is invalid`);
  }
  if (
    !nonNegativeInteger(evidence?.surface_texture_acquisition_failure_count_start) ||
    evidence?.surface_texture_acquisition_failure_count_end !==
      evidence?.surface_texture_acquisition_failure_count_start
  ) {
    failures.push(`${label} surface acquisition failure counter changed during the request`);
  }

  const suspendedEvents = runtimeEvents.filter(
    (event) => event?.event === SURFACE_INFERENCE_SUSPENDED_EVENT,
  );
  const resumedEvents = runtimeEvents.filter(
    (event) => event?.event === SURFACE_INFERENCE_RESUMED_EVENT,
  );
  if (suspendedEvents.length !== 1 || resumedEvents.length !== 1) {
    failures.push(
      `${label} expected exactly one surface suspend/resume pair, found ${suspendedEvents.length}/${resumedEvents.length}`,
    );
    return failures;
  }
  const suspended = suspendedEvents[0];
  const resumed = resumedEvents[0];
  const runStarted = progressEvents.find(
    (event) => event?.event === "run_started" && event?.model === TURBO_MODEL_ID,
  );
  const terminalEvents = progressEvents.filter((event) =>
    ["run_completed", "run_failed", "run_cancelled"].includes(event?.event),
  );
  const terminal = terminalEvents[0];
  const output = evidence?.output_ready;
  const window = Array.isArray(windows) && windows.length === 1 ? windows[0] : null;
  if (Array.isArray(windows) && windows.length !== 1) {
    failures.push(`${label} expected exactly one compact surface gate window, found ${windows.length}`);
  }

  if (
    suspended.policy !== SURFACE_INFERENCE_POLICY ||
    resumed.policy !== SURFACE_INFERENCE_POLICY
  ) {
    failures.push(`${label} surface gate policy is missing or inexact`);
  }
  if (
    !sameRunId(suspended.run_id, runStarted?.run_id) ||
    !sameRunId(resumed.run_id, runStarted?.run_id)
  ) {
    failures.push(`${label} surface gate run ID differs from run_started`);
  }
  if (
    suspended.primary_window_camera_count !== 2 ||
    suspended.saved_camera_state_count !== 2 ||
    suspended.previously_active_camera_count !== 2 ||
    suspended.inactive_camera_count !== 2 ||
    suspended.active_job_count !== 1 ||
    suspended.suspended_before_runtime_submit !== true ||
    suspended.all_primary_window_cameras_inactive !== true
  ) {
    failures.push(`${label} surface suspension does not attest both primary-window cameras`);
  }
  if (
    resumed.primary_window_camera_count !== 2 ||
    resumed.saved_camera_state_count !== 2 ||
    resumed.restored_camera_state_count !== 2 ||
    resumed.restored_active_camera_count !== 2 ||
    resumed.active_job_count !== 0 ||
    resumed.resumed_after_runtime_terminal !== true ||
    resumed.resumed_before_output_ready !== true ||
    resumed.exact_saved_states_restored !== true ||
    resumed.all_primary_window_cameras_restored !== true
  ) {
    failures.push(`${label} surface resume does not attest exact two-camera state restoration`);
  }
  if (terminalEvents.length !== 1 || terminal?.event !== "run_completed") {
    failures.push(`${label} surface gate is not bound to exactly one completed runtime terminal`);
  } else if (!sameRunId(terminal.run_id, runStarted?.run_id)) {
    failures.push(`${label} surface gate terminal run ID differs from run_started`);
  }
  if (resumed.terminal !== "completed") {
    failures.push(`${label} surface resume terminal=${report(resumed.terminal)}, expected completed`);
  }
  if (
    !positiveFinite(suspended.at_ms) ||
    !positiveFinite(runStarted?.at_ms) ||
    suspended.at_ms >= runStarted.at_ms
  ) {
    failures.push(`${label} surface suspension did not precede run_started/runtime compute`);
  }
  const firstCompute = progressEvents
    .filter((event) => ["stage_started", "step", "stage_completed"].includes(event?.event))
    .sort((left, right) => left.at_ms - right.at_ms)[0];
  if (
    firstCompute &&
    (!positiveFinite(firstCompute.at_ms) || suspended.at_ms >= firstCompute.at_ms)
  ) {
    failures.push(`${label} surface suspension did not precede the first compute event`);
  }
  if (
    !positiveFinite(terminal?.at_ms) ||
    !positiveFinite(resumed.at_ms) ||
    terminal.at_ms >= resumed.at_ms
  ) {
    failures.push(`${label} surface resumed before the actual runtime terminal completed`);
  }
  if (!positiveFinite(output?.at_ms) || resumed.at_ms >= output.at_ms) {
    failures.push(`${label} surface did not resume before output_ready`);
  }

  if (
    !window ||
    !sameRunId(window.run_id, runStarted?.run_id) ||
    window.policy !== SURFACE_INFERENCE_POLICY ||
    window.resume_policy !== SURFACE_INFERENCE_POLICY ||
    window.terminal !== "completed" ||
    window.suspended_at_ms !== suspended.at_ms ||
    window.resumed_at_ms !== resumed.at_ms
  ) {
    failures.push(`${label} compact surface gate window is missing or not event-bound`);
  }
  const preRequest = window?.pre_request_acquisition;
  if (
    preRequest?.succeeded !== true ||
    preRequest?.canvas_id !== "burn-image" ||
    !positiveFinite(preRequest?.at_ms) ||
    preRequest.at_ms >= suspended.at_ms ||
    !positiveInteger(preRequest?.call_index) ||
    preRequest.call_index > window?.acquisition_count_at_suspend
  ) {
    failures.push(`${label} lacks a successful real pre-request surface acquisition/present`);
  }
  if (
    !nonNegativeInteger(window?.acquisition_count_at_suspend) ||
    window?.acquisition_count_at_resume !== window?.acquisition_count_at_suspend ||
    window?.gated_call_count !== 0 ||
    window?.acquisition_count_at_suspend < acquisitionCountStart ||
    window?.acquisition_count_at_resume > acquisitionCountEnd
  ) {
    failures.push(`${label} called GPUCanvasContext.getCurrentTexture while inference held the surface gate`);
  }
  const postResume = window?.first_successful_post_resume_acquisition;
  if (
    postResume?.succeeded !== true ||
    postResume?.canvas_id !== "burn-image" ||
    !positiveFinite(postResume?.at_ms) ||
    postResume.at_ms <= resumed.at_ms ||
    !positiveInteger(postResume?.call_index) ||
    postResume.call_index <= window?.acquisition_count_at_resume ||
    postResume.call_index > acquisitionCountEnd
  ) {
    failures.push(`${label} lacks a successful post-resume surface acquisition`);
  }
  if (evidence?.surface_inference_state_after_request != null) {
    failures.push(`${label} left the page surface-inference pause attribute active`);
  }
  if (evidence?.active_surface_gate_after_request != null) {
    failures.push(`${label} left compact surface gate instrumentation active after the request`);
  }
  return failures;
}

function responseHeader(headers, name) {
  const expected = name.toLowerCase();
  for (const [key, value] of Object.entries(headers ?? {})) {
    if (key.toLowerCase() === expected) return String(value);
  }
  return undefined;
}

function networkRequestPathIdentity(url) {
  try {
    const parsed = new URL(url);
    return {
      protocol: parsed.protocol,
      pathname: parsed.pathname,
      search: parsed.search,
      hash: parsed.hash,
    };
  } catch {
    return null;
  }
}

/** Bind a CDP requestId to the exact request context needed to diagnose a later loadingFailed. */
export function captureCdpNetworkRequestContext(params, previous = null) {
  const request = params?.request;
  const requestId = typeof params?.requestId === "string" ? params.requestId : null;
  const url = typeof request?.url === "string" ? request.url : null;
  const method = typeof request?.method === "string" ? request.method : null;
  const rangeHeader = responseHeader(request?.headers, "Range") ?? null;
  const path = url === null ? null : networkRequestPathIdentity(url);
  return {
    request_id: requestId,
    url,
    method,
    range_header: rangeHeader,
    request_type: typeof params?.type === "string" ? params.type : null,
    document_url: typeof params?.documentURL === "string" ? params.documentURL : null,
    initiator_type:
      typeof params?.initiator?.type === "string" ? params.initiator.type : null,
    redirect_count:
      previous?.request_id === requestId && Number.isSafeInteger(previous?.redirect_count)
        ? previous.redirect_count + 1
        : 0,
    model_artifact_request: path?.pathname.startsWith("/model/") === true,
  };
}

/** Produce a stable, structured failure record from loadingFailed plus its request context. */
export function cdpNetworkLoadingFailedDiagnostic(params, requestContext = null) {
  const requestId = typeof params?.requestId === "string" ? params.requestId : null;
  const contextBound =
    requestId !== null && requestContext?.request_id === requestId;
  const diagnostic = {
    request_id: requestId,
    context_bound: contextBound,
    url: contextBound ? requestContext.url : null,
    method: contextBound ? requestContext.method : null,
    range_header: contextBound ? requestContext.range_header : null,
    request_type: contextBound ? requestContext.request_type : null,
    type: typeof params?.type === "string" ? params.type : null,
    errorText: typeof params?.errorText === "string" ? params.errorText : null,
    failure_type: typeof params?.type === "string" ? params.type : null,
    error_text: typeof params?.errorText === "string" ? params.errorText : null,
    canceled: params?.canceled === true,
    blocked_reason:
      typeof params?.blockedReason === "string" ? params.blockedReason : null,
    cors_error_status: params?.corsErrorStatus ?? null,
    model_artifact_request:
      contextBound && requestContext.model_artifact_request === true,
    redirect_count:
      contextBound && Number.isSafeInteger(requestContext.redirect_count)
        ? requestContext.redirect_count
        : null,
  };
  return {
    ...diagnostic,
    proven_benign_favicon: isProvenBenignFaviconFailure(diagnostic),
  };
}

/** The only ignored loading failure: an exact canceled GET for the implicit root favicon. */
export function isProvenBenignFaviconFailure(diagnostic) {
  const path = networkRequestPathIdentity(diagnostic?.url);
  return (
    diagnostic?.context_bound === true &&
    diagnostic?.model_artifact_request === false &&
    diagnostic?.method === "GET" &&
    diagnostic?.range_header === null &&
    diagnostic?.failure_type === "Other" &&
    diagnostic?.error_text === "net::ERR_ABORTED" &&
    diagnostic?.canceled === true &&
    /^https?:$/.test(path?.protocol ?? "") &&
    path?.pathname === "/favicon.ico" &&
    path?.search === "" &&
    path?.hash === ""
  );
}

export function validateCdpNetworkFailureEvidence(failures, ignoredFailures) {
  const validationFailures = [];
  if (!Array.isArray(failures)) {
    validationFailures.push("CDP network_failures is not an array");
  } else {
    for (const failure of failures) {
      if (failure?.proven_benign_favicon === true) {
        validationFailures.push("a proven benign favicon failure was not separated");
      } else {
        validationFailures.push(
          `CDP network failure: ${JSON.stringify(failure)}`,
        );
      }
    }
  }
  if (!Array.isArray(ignoredFailures)) {
    validationFailures.push("CDP ignored_benign_network_failures is not an array");
  } else {
    for (const failure of ignoredFailures) {
      if (!isProvenBenignFaviconFailure(failure)) {
        validationFailures.push(
          `CDP ignored a network failure without exact favicon proof: ${JSON.stringify(failure)}`,
        );
      }
    }
  }
  return validationFailures;
}

function exactModelResponseUrl(url, modelBaseUrls) {
  return modelBaseUrls.some((baseUrl) => url.startsWith(`${baseUrl}/`));
}

function exactModelBaseUrls(modelBaseUrls) {
  if (!Array.isArray(modelBaseUrls) || modelBaseUrls.length === 0) {
    throw new Error("exact modular model base URLs are missing");
  }
  const exactBaseUrls = [...new Set(modelBaseUrls.map((value) => {
    const url = new URL(String(value));
    if (!/^https?:$/.test(url.protocol) || url.search || url.hash) {
      throw new Error(`invalid modular model base URL ${value}`);
    }
    return url.href.replace(/\/$/, "");
  }))].sort();
  if (exactBaseUrls.length !== modelBaseUrls.length) {
    throw new Error("modular model base URLs are not unique");
  }
  return exactBaseUrls;
}

function summarizeCdpNetworkWindow(
  events,
  modelBaseUrls,
  windowStartEpochMs,
  windowEndEpochMs,
  policy,
  terminalEvent,
  transportTelemetryByPath,
) {
  if (!Array.isArray(events)) throw new Error("CDP events are not an array");
  const exactBaseUrls = exactModelBaseUrls(modelBaseUrls);
  if (
    !positiveFinite(windowStartEpochMs) ||
    !positiveFinite(windowEndEpochMs) ||
    windowEndEpochMs <= windowStartEpochMs
  ) {
    throw new Error(`Turbo CDP ${policy} window is invalid`);
  }
  const summary = {
    policy,
    model_base_urls: exactBaseUrls,
    window_start_epoch_ms: windowStartEpochMs,
    window_end_epoch_ms: windowEndEpochMs,
    terminal_event: terminalEvent,
    model_response_count: 0,
    http_200_complete_part_response_count: 0,
    http_206_response_count: 0,
    complete_object_validated_response_count: 0,
    content_range_validated_response_count: 0,
    response_body_bytes: 0,
    unexpected_status_response_count: 0,
    missing_content_length_count: 0,
    invalid_content_range_response_count: 0,
  };
  const componentTraffic = new Map();
  if (transportTelemetryByPath instanceof Map) {
    Object.assign(summary, {
      physical_transport_response_count: 0,
      physical_transport_response_bytes: 0,
      unmapped_physical_transport_response_count: 0,
      logical_component_traffic: {},
    });
  }
  for (const event of events) {
    if (
      event?.method !== "Network.responseReceived" ||
      !positiveFinite(event?.at_ms) ||
      event.at_ms < windowStartEpochMs ||
      event.at_ms > windowEndEpochMs
    ) {
      continue;
    }
    const response = event.params?.response;
    const url = String(response?.url ?? "");
    if (!exactModelResponseUrl(url, exactBaseUrls)) continue;
    summary.model_response_count += 1;
    let pathname = "";
    try {
      pathname = new URL(url).pathname;
    } catch {
      // exactModelResponseUrl already constrained the response to a model base.
    }
    const physicalTransportPart =
      pathname.includes("/transport/") && pathname.endsWith(".part");
    const expectedStatus = physicalTransportPart ? 200 : 206;
    if (response?.status !== expectedStatus) {
      summary.unexpected_status_response_count += 1;
      continue;
    }
    if (physicalTransportPart) summary.http_200_complete_part_response_count += 1;
    else summary.http_206_response_count += 1;
    const contentLengthText = responseHeader(response.headers, "Content-Length");
    const contentLength = Number(contentLengthText);
    if (!/^[1-9][0-9]*$/.test(contentLengthText ?? "") || !positiveInteger(contentLength)) {
      summary.missing_content_length_count += 1;
      continue;
    }
    summary.response_body_bytes += contentLength;
    if (transportTelemetryByPath instanceof Map) {
      if (physicalTransportPart) {
        summary.physical_transport_response_count += 1;
        summary.physical_transport_response_bytes += contentLength;
        const identity = transportTelemetryByPath.get(pathname);
        if (!identity) {
          summary.unmapped_physical_transport_response_count += 1;
        } else {
          const components = [...(identity.components ?? [identity.component])].sort();
          const logicalPaths = [...(identity.logical_paths ?? [])].sort();
          const key = `${identity.bundle}\u0000${components.join("\u0000")}`;
          const counters = componentTraffic.get(key) ?? {
            bundle: identity.bundle,
            component: identity.component,
            components,
            logical_paths: logicalPaths,
            shared_physical_part: identity.shared_physical_part === true,
            response_count: 0,
            response_bytes: 0,
          };
          counters.response_count += 1;
          counters.response_bytes += contentLength;
          componentTraffic.set(key, counters);
        }
      }
    }
    if (physicalTransportPart) {
      summary.complete_object_validated_response_count += 1;
      continue;
    }
    const contentRange = responseHeader(response.headers, "Content-Range");
    const match = /^bytes ([0-9]+)-([0-9]+)\/([1-9][0-9]*)$/.exec(contentRange ?? "");
    if (!match) {
      summary.invalid_content_range_response_count += 1;
      continue;
    }
    const start = Number(match[1]);
    const end = Number(match[2]);
    const total = Number(match[3]);
    if (
      ![start, end, total].every(Number.isSafeInteger) ||
      start < 0 ||
      end < start ||
      end >= total ||
      end - start + 1 !== contentLength
    ) {
      summary.invalid_content_range_response_count += 1;
      continue;
    }
    summary.content_range_validated_response_count += 1;
  }
  if (transportTelemetryByPath instanceof Map) {
    summary.logical_component_traffic = Object.fromEntries(
      [...componentTraffic.values()]
        .sort(
          (left, right) =>
            left.bundle.localeCompare(right.bundle) ||
            left.component.localeCompare(right.component),
        )
        .map((entry) => [`${entry.bundle}/${entry.component}`, entry]),
    );
  }
  return summary;
}

export function summarizeTurboPreloadCdpNetwork(
  events,
  snapshot,
  modelBaseUrls,
  transportTelemetryByPath,
) {
  const timeOrigin = snapshot?.time_origin_epoch_ms;
  if (!positiveFinite(timeOrigin)) throw new Error("browser performance time origin is missing");
  const started = snapshot?.runtime_events?.find(
    (event) =>
      event?.event === "preparing" &&
      event?.message === TURBO_PACKED_F16_PRELOAD_MESSAGE &&
      positiveFinite(event?.at_ms),
  );
  if (!started) throw new Error("Turbo denoiser preload start time anchor is missing");
  const terminal = snapshot?.runtime_events?.find(
    (event) =>
      event?.event === "packed_f16_denoiser_preload" &&
      positiveFinite(event?.at_ms) &&
      event.at_ms >= started.at_ms,
  );
  if (!terminal) throw new Error("Turbo denoiser preload completion time anchor is missing");
  return summarizeCdpNetworkWindow(
    events,
    modelBaseUrls,
    timeOrigin + started.at_ms,
    timeOrigin + terminal.at_ms,
    TURBO_CDP_PRELOAD_NETWORK_POLICY,
    terminal.event,
    transportTelemetryByPath,
  );
}

export function summarizeTurboRequestCdpNetwork(
  events,
  snapshot,
  modelBaseUrls,
  transportTelemetryByPath,
) {
  const timeOrigin = snapshot?.time_origin_epoch_ms;
  if (!positiveFinite(timeOrigin)) throw new Error("browser performance time origin is missing");
  const started = snapshot?.progress_events?.find(
    (event) => event?.event === "run_started" && event?.model === TURBO_MODEL_ID,
  );
  if (!positiveFinite(started?.at_ms)) throw new Error("Turbo run_started time anchor is missing");
  const output = snapshot?.output_events?.find(
    (event) =>
      event?.event === "ready" &&
      event?.model === TURBO_MODEL_ID &&
      positiveFinite(event?.at_ms) &&
      event.at_ms >= started.at_ms,
  );
  const failure = snapshot?.progress_events?.find(
    (event) =>
      ["run_failed", "run_cancelled"].includes(event?.event) &&
      positiveFinite(event?.at_ms) &&
      event.at_ms >= started.at_ms,
  );
  const terminal = output ?? failure;
  if (!terminal) throw new Error("Turbo output/failure time anchor is missing");
  return summarizeCdpNetworkWindow(
    events,
    modelBaseUrls,
    timeOrigin + started.at_ms,
    timeOrigin + terminal.at_ms,
    TURBO_CDP_REQUEST_NETWORK_POLICY,
    terminal.event,
    transportTelemetryByPath,
  );
}

export function summarizeTurboDmdCdpNetwork(
  events,
  snapshot,
  modelBaseUrls,
  transportTelemetryByPath,
) {
  const timeOrigin = snapshot?.time_origin_epoch_ms;
  if (!positiveFinite(timeOrigin)) throw new Error("browser performance time origin is missing");
  const started = snapshot?.progress_events?.find(
    (event) =>
      event?.event === "stage_started" &&
      event?.stage === "dmd" &&
      positiveFinite(event?.at_ms),
  );
  if (!started) throw new Error("Turbo DMD stage_started time anchor is missing");
  const terminal = snapshot?.progress_events?.find(
    (event) =>
      event?.event === "stage_completed" &&
      event?.stage === "dmd" &&
      JSON.stringify(event?.run_id) === JSON.stringify(started.run_id) &&
      positiveFinite(event?.at_ms) &&
      event.at_ms > started.at_ms,
  );
  if (!terminal) throw new Error("Turbo DMD stage_completed time anchor is missing");
  return summarizeCdpNetworkWindow(
    events,
    modelBaseUrls,
    timeOrigin + started.at_ms,
    timeOrigin + terminal.at_ms,
    TURBO_CDP_DMD_NETWORK_POLICY,
    "stage_completed:dmd",
    transportTelemetryByPath,
  );
}

export function gpuTerminalDiagnostic(message) {
  return GPU_TERMINAL_DIAGNOSTIC.test(String(message ?? ""));
}

export function aggregateChromeGpuInterval(rows, chromeGpuPids) {
  const pidSet =
    chromeGpuPids instanceof Set
      ? chromeGpuPids
      : new Set(Array.isArray(chromeGpuPids) ? chromeGpuPids : []);
  const unique = new Map();
  for (const row of Array.isArray(rows) ? rows : []) {
    if (
      !Number.isSafeInteger(row?.gpu_index) ||
      row.gpu_index < 0 ||
      !Number.isSafeInteger(row?.pid) ||
      !pidSet.has(row.pid)
    ) {
      continue;
    }
    const key = `${row.gpu_index}:${row.pid}`;
    const previous = unique.get(key);
    unique.set(key, {
      ...row,
      framebuffer_mib: Math.max(
        Number.isSafeInteger(previous?.framebuffer_mib) ? previous.framebuffer_mib : 0,
        Number.isSafeInteger(row.framebuffer_mib) ? row.framebuffer_mib : 0,
      ),
      sm_percent: Math.max(
        Number.isSafeInteger(previous?.sm_percent) ? previous.sm_percent : 0,
        Number.isSafeInteger(row.sm_percent) ? row.sm_percent : 0,
      ),
      memory_percent: Math.max(
        Number.isSafeInteger(previous?.memory_percent) ? previous.memory_percent : 0,
        Number.isSafeInteger(row.memory_percent) ? row.memory_percent : 0,
      ),
    });
  }
  const matchedRows = [...unique.values()].sort(
    (left, right) => left.gpu_index - right.gpu_index || left.pid - right.pid,
  );
  return {
    matched_rows: matchedRows,
    aggregate_framebuffer_mib: matchedRows.reduce(
      (sum, row) => sum + row.framebuffer_mib,
      0,
    ),
    aggregate_sm_percent: matchedRows.reduce((sum, row) => sum + row.sm_percent, 0),
    max_process_sm_percent: matchedRows.reduce(
      (maximum, row) => Math.max(maximum, row.sm_percent),
      0,
    ),
  };
}

export function validateHardwareNvidiaAdapter(adapter) {
  const failures = [];
  if (!adapter || typeof adapter !== "object") {
    return ["navigator.gpu did not return WebGPU adapter information"];
  }
  const identity = [
    adapter.vendor,
    adapter.architecture,
    adapter.device,
    adapter.description,
  ]
    .map((value) => String(value ?? ""))
    .join(" ");
  if (adapter.is_fallback_adapter === true || SOFTWARE_ADAPTER.test(identity)) {
    failures.push(`browser selected a software or fallback WebGPU adapter: ${report(adapter)}`);
  }
  if (!NVIDIA_ADAPTER.test(identity)) {
    failures.push(`browser WebGPU adapter is not attributable to NVIDIA: ${report(adapter)}`);
  }
  return failures;
}

/** Normalize the exact navigator.gpu call sequence made by the rendered Bevy page. */
export function summarizeRenderedSurfaceRuntimeWebGpuCalls(
  webGpuCalls,
  engineSessionId = null,
  droppedCalls = 0,
) {
  const calls = Array.isArray(webGpuCalls) ? webGpuCalls : [];
  const adapterStarts = calls.filter((entry) => entry?.event === "request-adapter-start");
  const adapterRejections = calls.filter(
    (entry) => entry?.event === "request-adapter-rejected",
  );
  const adapterSuccesses = calls.filter(
    (entry) => entry?.event === "request-adapter-resolved" && entry?.detail?.available === true,
  );
  const deviceStarts = calls.filter((entry) => entry?.event === "request-device-start");
  const deviceRejections = calls.filter(
    (entry) => entry?.event === "request-device-rejected",
  );
  const deviceSuccesses = calls.filter(
    (entry) => entry?.event === "request-device-resolved",
  );
  const selectedAdapterSuccess = adapterSuccesses.at(-1);
  const selectedAdapterRequestId = selectedAdapterSuccess?.detail?.request_id;
  const selectedAdapter = selectedAdapterSuccess?.detail?.info ?? null;
  const selectedAdapterStart = adapterStarts.find(
    (entry) => entry?.detail?.request_id === selectedAdapterRequestId,
  )?.detail ?? null;
  const selectedDeviceSuccess = deviceSuccesses.at(-1);
  const selectedDeviceRequestId = selectedDeviceSuccess?.detail?.request_id;
  const selectedDevice = selectedDeviceSuccess?.detail ?? deviceStarts.find(
    (entry) => entry?.detail?.request_id === selectedDeviceRequestId,
  )?.detail ?? null;
  const allowedEvents = new Set([
    "navigator-gpu-unavailable",
    "request-adapter-instrumentation-unavailable",
    "request-adapter-start",
    "request-adapter-resolved",
    "request-adapter-rejected",
    "request-device-instrumentation-unavailable",
    "request-device-start",
    "request-device-resolved",
    "request-device-rejected",
  ]);
  let previousAtMs = -Infinity;
  let invalidRawCallCount = 0;
  let nonMonotonicRawCallCount = 0;
  let instrumentationFailureCount = 0;
  for (const entry of calls) {
    if (
      !entry ||
      typeof entry !== "object" ||
      !allowedEvents.has(entry.event) ||
      !Number.isFinite(entry.at_ms) ||
      entry.at_ms < 0
    ) {
      invalidRawCallCount += 1;
      continue;
    }
    if (
      entry.event.endsWith("instrumentation-unavailable") ||
      entry.event === "navigator-gpu-unavailable" ||
      entry.detail?.telemetry_error ||
      entry.detail?.info?.telemetry_error
    ) {
      instrumentationFailureCount += 1;
    }
    if (entry.at_ms < previousAtMs) nonMonotonicRawCallCount += 1;
    previousAtMs = entry.at_ms;
  }
  const adapterStartIds = adapterStarts.map((entry) => entry?.detail?.request_id);
  const adapterTerminals = calls.filter((entry) =>
    ["request-adapter-resolved", "request-adapter-rejected"].includes(entry?.event),
  );
  const deviceStartIds = deviceStarts.map((entry) => entry?.detail?.request_id);
  const deviceTerminals = calls.filter((entry) =>
    ["request-device-resolved", "request-device-rejected"].includes(entry?.event),
  );
  const hasUniqueSafeIds = (ids) =>
    ids.every((id) => Number.isSafeInteger(id) && id > 0) && new Set(ids).size === ids.length;
  const terminalCount = (terminals, id) =>
    terminals.filter((entry) => entry?.detail?.request_id === id).length;
  const incompleteAdapterAttempts = adapterStartIds.filter(
    (id) => terminalCount(adapterTerminals, id) !== 1,
  ).length;
  const unknownAdapterTerminals = adapterTerminals.filter(
    (entry) => !adapterStartIds.includes(entry?.detail?.request_id),
  ).length;
  const incompleteDeviceAttempts = deviceStartIds.filter(
    (id) => terminalCount(deviceTerminals, id) !== 1,
  ).length;
  const unknownDeviceTerminals = deviceTerminals.filter(
    (entry) => !deviceStartIds.includes(entry?.detail?.request_id),
  ).length;
  const adapterSuccessIndex = calls.indexOf(selectedAdapterSuccess);
  const deviceStartIndex = calls.findIndex(
    (entry) =>
      entry?.event === "request-device-start" &&
      entry?.detail?.request_id === selectedDeviceRequestId,
  );
  const deviceSuccessIndex = calls.indexOf(selectedDeviceSuccess);
  return {
    schema_version: 1,
    source: RUNTIME_ADAPTER_SOURCE,
    engine_session_id: engineSessionId,
    raw_call_count: calls.length,
    dropped_call_count: droppedCalls,
    invalid_raw_call_count: invalidRawCallCount,
    non_monotonic_raw_call_count: nonMonotonicRawCallCount,
    instrumentation_failure_count: instrumentationFailureCount,
    adapter_request_attempts: adapterStarts.length,
    adapter_rejected_attempts: adapterRejections.length,
    adapter_successful_attempts: adapterSuccesses.length,
    adapter_request_ids_valid: hasUniqueSafeIds(adapterStartIds),
    incomplete_adapter_attempts: incompleteAdapterAttempts,
    unknown_adapter_terminals: unknownAdapterTerminals,
    device_request_attempts: deviceStarts.length,
    device_rejected_attempts: deviceRejections.length,
    device_successful_attempts: deviceSuccesses.length,
    device_request_ids_valid: hasUniqueSafeIds(deviceStartIds),
    incomplete_device_attempts: incompleteDeviceAttempts,
    unknown_device_terminals: unknownDeviceTerminals,
    selected_adapter_request_id: selectedAdapterRequestId ?? null,
    selected_device_request_id: selectedDeviceRequestId ?? null,
    selected_device_adapter_request_id:
      selectedDeviceSuccess?.detail?.adapter_request_id ?? null,
    selected_adapter_call_index: adapterSuccessIndex,
    selected_device_start_call_index: deviceStartIndex,
    selected_device_success_call_index: deviceSuccessIndex,
    selected_device_success_at_ms: selectedDeviceSuccess?.at_ms ?? null,
    power_preference: selectedAdapterStart?.powerPreference ?? null,
    force_fallback_adapter: selectedAdapterStart?.forceFallbackAdapter ?? null,
    is_fallback_adapter:
      selectedAdapter?.is_fallback_adapter ?? selectedAdapter?.isFallbackAdapter ?? null,
    vendor: selectedAdapter?.vendor ?? null,
    architecture: selectedAdapter?.architecture ?? null,
    device: selectedAdapter?.device ?? null,
    description: selectedAdapter?.description ?? null,
    requested_features: Array.isArray(selectedDevice?.requiredFeatures)
      ? selectedDevice.requiredFeatures
      : null,
    enabled_features: Array.isArray(selectedDeviceSuccess?.detail?.enabledFeatures)
      ? selectedDeviceSuccess.detail.enabledFeatures
      : null,
    requested_limits:
      selectedDevice?.requiredLimits && typeof selectedDevice.requiredLimits === "object"
        ? selectedDevice.requiredLimits
        : null,
  };
}

/**
 * Bind Bevy's shared-device-ready event to the exact adapter and device requests made by wgpu.
 * The separate pre-navigation probe is deliberately excluded: it is launch readiness only.
 */
export function attestRenderedSurfaceRuntimeAdapter(
  event,
  webGpuCalls,
  engineSessionId = null,
  droppedCalls = 0,
  requiredDeviceFeatures = BOOGU_WEB_REQUIRED_DEVICE_FEATURES,
) {
  const attestation = summarizeRenderedSurfaceRuntimeWebGpuCalls(
    webGpuCalls,
    engineSessionId,
    droppedCalls,
  );
  const failures = [];
  const expectedEnabledFeatures = [
    ...new Set([
      ...BROWSER_WEBGPU_ENABLED_FEATURE_BASELINE,
      ...(Array.isArray(requiredDeviceFeatures) ? requiredDeviceFeatures : []),
    ]),
  ].sort();
  if (
    ![
      JSON.stringify(GENERIC_WEB_REQUIRED_DEVICE_FEATURES),
      JSON.stringify(BOOGU_WEB_REQUIRED_DEVICE_FEATURES),
    ].includes(JSON.stringify(requiredDeviceFeatures))
  ) {
    failures.push(
      `rendered device feature contract=${report(requiredDeviceFeatures)}, expected exact generic or Boogu-Web policy`,
    );
  }
  if (attestation.source !== RUNTIME_ADAPTER_SOURCE) {
    failures.push("runtime adapter attestation does not come from the rendered page interception");
  }
  if (!/^[0-9a-f-]{16,}$/i.test(String(attestation.engine_session_id ?? ""))) {
    failures.push("rendered runtime WebGPU attestation has no engine-session identity");
  }
  if (!Array.isArray(webGpuCalls)) {
    failures.push("runtime WebGPU raw calls are not an array");
  }
  if (attestation.raw_call_count === 0) {
    failures.push("rendered page did not record any runtime WebGPU calls");
  }
  if (
    !Number.isSafeInteger(attestation.dropped_call_count) ||
    attestation.dropped_call_count !== 0
  ) {
    failures.push(
      `rendered runtime WebGPU evidence dropped ${report(attestation.dropped_call_count)} calls`,
    );
  }
  if (attestation.invalid_raw_call_count !== 0) {
    failures.push(
      `runtime WebGPU evidence has ${attestation.invalid_raw_call_count} invalid raw calls`,
    );
  }
  if (attestation.non_monotonic_raw_call_count !== 0) {
    failures.push("runtime WebGPU raw calls are not in monotonic page-time order");
  }
  if (attestation.instrumentation_failure_count !== 0) {
    failures.push("rendered runtime WebGPU instrumentation or telemetry was unavailable");
  }
  if (
    !Number.isSafeInteger(attestation.adapter_request_attempts) ||
    attestation.adapter_request_attempts < 1
  ) {
    failures.push("rendered runtime did not make an adapter request");
  }
  if (attestation.adapter_successful_attempts !== 1) {
    failures.push(
      `rendered runtime has ${report(attestation.adapter_successful_attempts)} successful adapter requests; expected exactly one`,
    );
  }
  if (
    attestation.adapter_request_ids_valid !== true ||
    attestation.incomplete_adapter_attempts !== 0 ||
    attestation.unknown_adapter_terminals !== 0
  ) {
    failures.push("rendered runtime adapter attempts are not exactly paired by request ID");
  }
  if (attestation.device_request_attempts !== 1 || attestation.device_successful_attempts !== 1) {
    failures.push(
      `rendered runtime has device requests=${report(attestation.device_request_attempts)}, successes=${report(attestation.device_successful_attempts)}; expected exactly one of each`,
    );
  }
  if (attestation.device_rejected_attempts !== 0) {
    failures.push(
      `rendered runtime has ${attestation.device_rejected_attempts} rejected device requests`,
    );
  }
  if (
    attestation.device_request_ids_valid !== true ||
    attestation.incomplete_device_attempts !== 0 ||
    attestation.unknown_device_terminals !== 0
  ) {
    failures.push("rendered runtime device attempts are not exactly paired by request ID");
  }
  if (attestation.power_preference !== "high-performance") {
    failures.push(
      `rendered runtime adapter power preference=${report(attestation.power_preference)}, expected "high-performance"`,
    );
  }
  if (attestation.force_fallback_adapter === true) {
    failures.push("rendered runtime adapter request explicitly forced fallback");
  }
  if (attestation.is_fallback_adapter !== false) {
    failures.push(
      `rendered runtime adapter fallback status is not explicitly false: ${report(attestation.is_fallback_adapter)}`,
    );
  }
  const nativeIdentity = [
    attestation.vendor,
    attestation.architecture,
    attestation.device,
    attestation.description,
  ]
    .map((value) => String(value ?? ""))
    .join(" ");
  if (!NVIDIA_ADAPTER.test(nativeIdentity) || SOFTWARE_ADAPTER.test(nativeIdentity)) {
    failures.push(
      `rendered runtime adapter evidence is not non-software NVIDIA hardware: ${report(nativeIdentity)}`,
    );
  }
  if (
    JSON.stringify(attestation.requested_features) !== JSON.stringify(requiredDeviceFeatures)
  ) {
    failures.push(
      `rendered device requested_features=${report(attestation.requested_features)}, expected exact ${report(requiredDeviceFeatures)}`,
    );
  }
  if (JSON.stringify(attestation.enabled_features) !== JSON.stringify(expectedEnabledFeatures)) {
    failures.push(
      `rendered device enabled_features=${report(attestation.enabled_features)}, expected exact ordered browser-baseline union ${report(expectedEnabledFeatures)}`,
    );
  }
  if (!attestation.requested_limits || Array.isArray(attestation.requested_limits)) {
    failures.push("rendered runtime device requested_limits were not captured");
  }
  if (
    !(
      attestation.selected_adapter_call_index >= 0 &&
      attestation.selected_device_start_call_index > attestation.selected_adapter_call_index &&
      attestation.selected_device_success_call_index > attestation.selected_device_start_call_index
    )
  ) {
    failures.push("rendered runtime adapter/device call order is not exact");
  }
  if (
    attestation.selected_device_adapter_request_id !==
    attestation.selected_adapter_request_id
  ) {
    failures.push("rendered runtime device is not bound to the selected adapter request");
  }
  if (
    !positiveFinite(event?.at_ms) ||
    !Number.isFinite(attestation.selected_device_success_at_ms) ||
    attestation.selected_device_success_at_ms >= event.at_ms
  ) {
    failures.push("rendered runtime device did not resolve before Bevy emitted ready");
  }
  failures.push(...validateBackendReadyEvent(event, attestation));
  return {
    ...attestation,
    browser_enabled_feature_baseline: [...BROWSER_WEBGPU_ENABLED_FEATURE_BASELINE],
    required_device_features: Array.isArray(requiredDeviceFeatures)
      ? [...requiredDeviceFeatures]
      : null,
    expected_enabled_features: expectedEnabledFeatures,
    bevy_adapter_name: event?.adapter_name ?? null,
    bevy_adapter_backend: event?.backend ?? null,
    bevy_adapter_device_type: event?.device_type ?? null,
    validation_failures: failures,
    validated: failures.length === 0,
  };
}

export function validateBackendReadyEvent(event, runtimeAttestation) {
  const failures = [];
  if (!event || typeof event !== "object") {
    return ["Bevy did not emit a burn-image-backend ready event"];
  }
  if (event.event !== "ready") {
    failures.push(`Bevy backend event=${report(event.event)}, expected "ready"`);
  }
  if (event.backend !== "BrowserWebGpu") {
    failures.push(`Bevy ready backend=${report(event.backend)}, expected exact BrowserWebGpu`);
  }
  if (!["Other", "DiscreteGpu", "IntegratedGpu", "VirtualGpu"].includes(event.device_type)) {
    failures.push(`Bevy ready device type is not a GPU class: ${report(event.device_type)}`);
  }
  if (event.shared_adapter_device_queue !== true) {
    failures.push("Bevy/Burn did not attest one shared adapter, device, and queue");
  }
  if (!/shared device ready/i.test(String(event.message ?? ""))) {
    failures.push(`Bevy visible GPU-ready status is missing: ${report(event.message)}`);
  }

  const browserDescription = String(runtimeAttestation?.description ?? "").trim();
  const bevyAdapter = String(event.adapter_name ?? "").trim();
  if (bevyAdapter && !NVIDIA_ADAPTER.test(bevyAdapter)) {
    failures.push(`Bevy ready adapter is not NVIDIA: ${report(event.adapter_name)}`);
  }
  if (browserDescription !== bevyAdapter) {
    failures.push(
      `rendered runtime adapter ${report(browserDescription)} does not exactly match Bevy adapter ${report(bevyAdapter)}`,
    );
  }
  return failures;
}

export function validateRenderedSurfaceSnapshot(snapshot, expectedUrl) {
  const failures = [];
  if (!snapshot || typeof snapshot !== "object") {
    return ["rendered page snapshot is missing"];
  }
  if (snapshot.url !== expectedUrl) {
    failures.push(`page URL=${report(snapshot.url)}, expected exact ${report(expectedUrl)}`);
  }
  if (snapshot.secure_context !== true) {
    failures.push("rendered page is not a secure context");
  }
  if (snapshot.ready_state !== "complete") {
    failures.push(`document.readyState=${report(snapshot.ready_state)}, expected "complete"`);
  }
  const canvas = snapshot.canvas;
  if (!canvas || typeof canvas !== "object") {
    failures.push("#burn-image canvas is missing");
  } else {
    for (const field of ["width", "height", "client_width", "client_height"]) {
      if (!positiveInteger(canvas[field])) {
        failures.push(`canvas.${field}=${report(canvas[field])}, expected a positive integer`);
      }
    }
    for (const field of ["rect_width", "rect_height"]) {
      if (!positiveFinite(canvas[field])) {
        failures.push(`canvas.${field}=${report(canvas[field])}, expected a positive finite value`);
      }
    }
  }
  if (!positiveFinite(snapshot.device_pixel_ratio)) {
    failures.push(
      `window.devicePixelRatio=${report(snapshot.device_pixel_ratio)}, expected a positive finite value`,
    );
  }
  return failures;
}

export function validateTestedPackageIdentity(identity, servedTransport) {
  const failures = [];
  if (!identity || typeof identity !== "object") {
    return ["exact tested browser package identity is missing"];
  }
  if (
    identity.policy !== "exact-local-package-and-runtime-source-bytes-served-to-browser" ||
    identity.validated !== true
  ) {
    failures.push("tested browser package identity policy/validation is incorrect");
  }
  const expected = [
    ["generated_package.javascript", identity.generated_package?.javascript, "bevy_burn_image.js"],
    [
      "generated_package.webassembly",
      identity.generated_package?.webassembly,
      "bevy_burn_image_bg.wasm",
    ],
    [
      "page_modules.model_selector",
      identity.page_modules?.model_selector,
      "crates/bevy_image/www/model_selector.mjs",
    ],
    [
      "generated_package.app_icon",
      identity.generated_package?.app_icon,
      "burn-image-icon.png",
    ],
    [
      "sources.browser_runtime",
      identity.sources?.browser_runtime,
      "crates/bevy_image/src/browser_boogu/runtime.rs",
    ],
    [
      "sources.rendered_harness",
      identity.sources?.rendered_harness,
      "crates/bevy_image/tests/wasm_rendered_surface_smoke.mjs",
    ],
    [
      "sources.rendered_contract",
      identity.sources?.rendered_contract,
      "crates/bevy_image/tests/wasm_rendered_surface_contract.mjs",
    ],
    [
      "sources.artifact_transport_contract",
      identity.sources?.artifact_transport_contract,
      "crates/bevy_image/tests/artifact_transport_contract.mjs",
    ],
  ];
  for (const [label, entry, expectedRelativePath] of expected) {
    if (!entry || typeof entry !== "object") {
      failures.push(`tested package ${label} identity is missing`);
      continue;
    }
    if (typeof entry.absolute_path !== "string" || !entry.absolute_path.startsWith("/")) {
      failures.push(`tested package ${label}.absolute_path is not canonical absolute identity`);
    }
    if (entry.relative_path !== expectedRelativePath) {
      failures.push(
        `tested package ${label}.relative_path=${report(entry.relative_path)}, expected ${expectedRelativePath}`,
      );
    }
    if (!positiveInteger(entry.bytes)) {
      failures.push(`tested package ${label}.bytes is not a positive integer`);
    }
    if (!/^[a-f0-9]{64}$/.test(String(entry.sha256 ?? ""))) {
      failures.push(`tested package ${label}.sha256 is missing or malformed`);
    }
  }

  for (const [packageKey, servedKey] of [
    ["javascript", "bevy_burn_image.js"],
    ["webassembly", "bevy_burn_image_bg.wasm"],
  ]) {
    const local = identity.generated_package?.[packageKey];
    const served = servedTransport?.generated?.[servedKey];
    if (!served || served.bytes !== local?.bytes || served.sha256 !== local?.sha256) {
      failures.push(`served ${servedKey} does not match its exact tested local identity`);
    }
  }
  const localSelector = identity.page_modules?.model_selector;
  const servedSelector = servedTransport?.page_modules?.["model_selector.mjs"];
  if (
    !servedSelector ||
    servedSelector.bytes !== localSelector?.bytes ||
    servedSelector.sha256 !== localSelector?.sha256 ||
    servedSelector.content_type !== "text/javascript; charset=utf-8"
  ) {
    failures.push(
      "served model_selector.mjs MIME, bytes, or SHA-256 do not match its exact tested identity",
    );
  }
  const localIcon = identity.generated_package?.app_icon;
  const servedIcon = servedTransport?.generated?.["burn-image-icon.png"];
  if (
    !servedIcon ||
    servedIcon.bytes !== localIcon?.bytes ||
    servedIcon.sha256 !== localIcon?.sha256 ||
    servedIcon.content_type !== "image/png"
  ) {
    failures.push(
      "served burn-image-icon.png MIME, bytes, or SHA-256 do not match its exact tested identity",
    );
  }
  return failures;
}

export function validateRenderedLaunchReadinessProbe(probe) {
  const failures = [];
  if (!probe || typeof probe !== "object") {
    return ["pre-navigation WebGPU launch-readiness probe is missing"];
  }
  if (probe.policy !== RENDERED_LAUNCH_READINESS_PROBE_POLICY) {
    failures.push(
      `launch-readiness probe policy=${report(probe.policy)}, expected an explicit non-proof label`,
    );
  }
  if (probe.runtime_hardware_proof !== false) {
    failures.push("pre-navigation launch-readiness probe is incorrectly labeled runtime proof");
  }
  failures.push(
    ...validateHardwareNvidiaAdapter(probe.adapter).map(
      (failure) => `launch-readiness only: ${failure}`,
    ),
  );
  return failures;
}

export function validateRenderedRuntimeWebGpuEvidence(
  event,
  webGpuCalls,
  engineSessionId,
  droppedCalls,
  providedAttestation,
  requiredDeviceFeatures = BOOGU_WEB_REQUIRED_DEVICE_FEATURES,
) {
  const calculated = attestRenderedSurfaceRuntimeAdapter(
    event,
    webGpuCalls,
    engineSessionId,
    droppedCalls,
    requiredDeviceFeatures,
  );
  const failures = [...calculated.validation_failures];
  if (!providedAttestation || typeof providedAttestation !== "object") {
    failures.push("normalized rendered-runtime WebGPU attestation is missing");
  } else if (JSON.stringify(providedAttestation) !== JSON.stringify(calculated)) {
    failures.push("normalized rendered-runtime WebGPU attestation differs from its raw calls");
  }
  return failures;
}

export function validateRenderedSurfaceEvidence(evidence) {
  const failures = [
    ...validateRenderedLaunchReadinessProbe(evidence?.launch_readiness_webgpu_probe),
    ...validateRenderedRuntimeWebGpuEvidence(
      evidence?.bevy_backend_ready,
      evidence?.runtime_webgpu_calls,
      evidence?.page_snapshot?.engine_session_id,
      evidence?.page_snapshot?.webgpu_dropped_calls,
      evidence?.runtime_webgpu_adapter_attestation,
      evidence?.required_device_features,
    ),
    ...validateRenderedSurfaceSnapshot(evidence?.page_snapshot, evidence?.expected_url),
    ...validateTestedPackageIdentity(
      evidence?.tested_package_identity,
      evidence?.served_transport,
    ),
    ...validateCdpNetworkFailureEvidence(
      evidence?.network_failures,
      evidence?.ignored_benign_network_failures,
    ),
  ];

  const backendFailures = Array.isArray(evidence?.bevy_backend_events)
    ? evidence.bevy_backend_events.filter((event) => event?.event === "failed")
    : [];
  if (backendFailures.length > 0) {
    failures.push(`Bevy emitted failed backend state: ${report(backendFailures)}`);
  }
  for (const [field, values] of [
    ["page_errors", evidence?.page_errors],
    ["gpu_errors", evidence?.gpu_errors],
  ]) {
    if (!Array.isArray(values)) {
      failures.push(`${field} is not an array`);
    } else if (values.length > 0) {
      failures.push(`${field} is not empty: ${report(values)}`);
    }
  }
  if (gpuTerminalDiagnostic(evidence?.chrome_stderr)) {
    failures.push("Chrome stderr contains a terminal WebGPU/device-loss diagnostic");
  }
  if (!Number.isSafeInteger(evidence?.screenshot_bytes) || evidence.screenshot_bytes < 1024) {
    failures.push("rendered-window screenshot is empty or missing");
  }
  if (!/^[a-f0-9]{64}$/.test(String(evidence?.screenshot_sha256 ?? ""))) {
    failures.push("rendered-window screenshot SHA-256 is missing or malformed");
  }
  return failures;
}

function validateExactArtifactTraffic(traffic, expected, label) {
  const failures = [];
  if (!traffic || typeof traffic !== "object") {
    return [`${label} artifact traffic payload is missing`];
  }
  for (const field of ARTIFACT_TRAFFIC_FIELDS) {
    if (!nonNegativeInteger(traffic[field])) {
      failures.push(`${label} artifact traffic ${field}=${report(traffic[field])}, expected a non-negative integer`);
    }
    if (traffic[field] !== expected[field]) {
      failures.push(
        `${label} artifact traffic ${field}=${report(traffic[field])}, expected exact ${report(expected[field])}`,
      );
    }
  }
  if (traffic.verified_objects !== traffic.object_reads) {
    failures.push(`${label} verified_objects does not equal completed object_reads`);
  }
  if (traffic.range_reads !== traffic.cache_lookups) {
    failures.push(`${label} required persistent cache was not consulted for every range read`);
  }
  if (traffic.cache_hits + traffic.cache_misses !== traffic.cache_lookups) {
    failures.push(`${label} cache hits and misses do not sum to cache lookups`);
  }
  if (traffic.range_read_bytes !== traffic.cache_read_bytes + traffic.network_response_bytes) {
    failures.push(`${label} cache/network bytes do not sum to logical range bytes`);
  }
  if (traffic.range_read_bytes !== traffic.object_read_bytes) {
    failures.push(`${label} logical range bytes do not equal completed object bytes`);
  }
  if (traffic.network_requests !== traffic.cache_misses) {
    failures.push(`${label} network requests do not equal clean persistent-cache misses`);
  }
  if (traffic.cache_writes !== traffic.network_requests) {
    failures.push(`${label} cache writes do not equal successful network requests`);
  }
  if (traffic.cache_write_bytes !== traffic.network_response_bytes) {
    failures.push(`${label} cache-write bytes do not equal network response bytes`);
  }
  return failures;
}

function validatePackedF16Lifecycle(lifecycle, label, expectedPreloadAttemptCount = 1) {
  const failures = [];
  if (!lifecycle || typeof lifecycle !== "object" || Array.isArray(lifecycle)) {
    return [`${label} packed-F16 lifecycle payload is missing`];
  }
  const expectedLifecycle = {
    ...TURBO_PACKED_F16_REQUEST_LIFECYCLE,
    authenticated_artifact_bytes:
      TURBO_PACKED_F16_RESOURCE_PLAN.authenticated_artifact_bytes *
      expectedPreloadAttemptCount,
    packed_upload_bytes:
      TURBO_PACKED_F16_RESOURCE_PLAN.retained_packed_f16_denoiser_bytes *
      expectedPreloadAttemptCount,
    preload_attempt_count: expectedPreloadAttemptCount,
  };
  const expectedKeys = Object.keys(expectedLifecycle).sort();
  const observedKeys = Object.keys(lifecycle).sort();
  if (JSON.stringify(observedKeys) !== JSON.stringify(expectedKeys)) {
    failures.push(`${label} packed-F16 lifecycle fields differ from the frozen schema`);
  }
  for (const [field, expected] of Object.entries(expectedLifecycle)) {
    if (field === "dmd_artifact_traffic") continue;
    if (!Object.is(lifecycle[field], expected)) {
      failures.push(
        `${label} packed-F16 lifecycle ${field}=${report(lifecycle[field])}, expected ${report(expected)}`,
      );
    }
  }
  failures.push(
    ...validateExactArtifactTraffic(
      lifecycle.dmd_artifact_traffic,
      TURBO_DMD_ZERO_IO,
      `${label} DMD`,
    ),
  );
  return failures;
}

function exactObjectKeys(value, expected, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return [`${label} is missing or not an object`];
  }
  const observed = Object.keys(value).sort();
  const frozen = [...expected].sort();
  return JSON.stringify(observed) === JSON.stringify(frozen)
    ? []
    : [`${label} fields differ from the frozen schema`];
}

function validatePackedF16F32Diagnostic(tensor, expectedName, shapePredicate, label) {
  const failures = exactObjectKeys(
    tensor,
    [
      "name",
      "shape",
      "dtype",
      "element_count",
      "finite_element_count",
      "all_finite",
      "max_abs",
      "mean",
      "rms",
      "sha256",
    ],
    `${label} tensor`,
  );
  if (failures.length > 0) return failures;
  if (tensor.name !== expectedName) {
    failures.push(`${label} name=${report(tensor.name)}, expected ${expectedName}`);
  }
  if (!Array.isArray(tensor.shape) || !shapePredicate(tensor.shape)) {
    failures.push(`${label} shape=${report(tensor.shape)} is incorrect`);
  }
  const shapeElements = Array.isArray(tensor.shape)
    ? tensor.shape.reduce(
        (product, dimension) =>
          positiveInteger(dimension) && Number.isSafeInteger(product * dimension)
            ? product * dimension
            : Number.NaN,
        1,
      )
    : Number.NaN;
  if (!positiveInteger(tensor.element_count) || tensor.element_count !== shapeElements) {
    failures.push(`${label} element_count does not equal its shape product`);
  }
  if (tensor.dtype !== "f32") failures.push(`${label} dtype=${report(tensor.dtype)}, expected f32`);
  if (tensor.all_finite !== true || tensor.finite_element_count !== tensor.element_count) {
    failures.push(`${label} does not attest every element finite`);
  }
  for (const field of ["max_abs", "mean", "rms"]) {
    if (!Number.isFinite(tensor[field])) failures.push(`${label} ${field} is null or non-finite`);
  }
  if (!(tensor.max_abs > 0)) failures.push(`${label} is all-zero or has invalid max_abs`);
  if (!(tensor.rms > 0)) failures.push(`${label} is all-zero or has invalid RMS`);
  if (!/^[a-f0-9]{64}$/.test(String(tensor.sha256 ?? ""))) {
    failures.push(`${label} SHA-256 is missing or malformed`);
  }
  return failures;
}

function validatePackedF16CacheEvidence(cache, label) {
  const failures = exactObjectKeys(
    cache,
    ["state", "cache_ready", "cached_stages", "cached_objects", "cached_tensors", "cached_bytes"],
    label,
  );
  if (failures.length > 0) return failures;
  for (const [field, expected] of [
    ["state", "ready"],
    ["cache_ready", true],
    ["cached_stages", TURBO_PACKED_F16_CACHED_STAGES],
    ["cached_objects", TURBO_PACKED_F16_CACHED_OBJECTS],
    ["cached_tensors", TURBO_PACKED_F16_CACHED_TENSORS],
    ["cached_bytes", TURBO_PACKED_F16_RESOURCE_PLAN.retained_packed_f16_denoiser_bytes],
  ]) {
    if (!Object.is(cache[field], expected)) {
      failures.push(`${label} ${field}=${report(cache[field])}, expected ${report(expected)}`);
    }
  }
  return failures;
}

function validateEmptyPackedF16CacheEvidence(cache, label) {
  const failures = exactObjectKeys(
    cache,
    ["state", "cache_ready", "cached_stages", "cached_objects", "cached_tensors", "cached_bytes"],
    label,
  );
  if (failures.length > 0) return failures;
  for (const [field, expected] of [
    ["state", "empty"],
    ["cache_ready", false],
    ["cached_stages", 0],
    ["cached_objects", 0],
    ["cached_tensors", 0],
    ["cached_bytes", 0],
  ]) {
    if (!Object.is(cache[field], expected)) {
      failures.push(`${label} ${field}=${report(cache[field])}, expected ${report(expected)}`);
    }
  }
  return failures;
}

export function validatePackedF16DmdVaeHandoff(
  event,
  progressEvents,
  expectedPreloadAttemptCount = 1,
  label = "Generate request",
) {
  const failures = exactObjectKeys(
    event,
    ["at_ms", "event", "run_id", "report"],
    `${label} DMD-to-VAE handoff event`,
  );
  if (failures.length > 0) return failures;
  if (event.event !== "packed_f16_dmd_vae_handoff") {
    failures.push(`${label} DMD-to-VAE handoff event name is incorrect`);
  }
  const handoff = event.report;
  failures.push(
    ...exactObjectKeys(
      handoff,
      [
        "policy",
        "next_request_rehydration_policy",
        "shape",
        "dtype",
        "element_count",
        "payload_bytes",
        "device_to_host_readback_bytes",
        "host_to_device_upload_bytes",
        "total_transfer_bytes",
        "before_sha256",
        "after_sha256",
        "all_finite",
        "not_all_zero",
        "digest_matches",
        "wrapper_cached_stages_before_clear",
        "wrapper_cached_stages_after_clear",
        "synchronization_pending_before_cleanup",
        "synchronization_pending_after_cleanup",
        "rope_cache_cleared",
        "cleanup_completed",
        "packed_cache_before_cleanup",
        "packed_cache_after_cleanup",
        "preload_attempt_count",
        "expected_next_request_preload_attempt_count",
      ],
      `${label} DMD-to-VAE handoff report`,
    ),
  );
  if (!handoff || typeof handoff !== "object") return failures;
  for (const [field, expected] of [
    ["policy", TURBO_PACKED_F16_DMD_VAE_HANDOFF_POLICY],
    ["next_request_rehydration_policy", TURBO_PACKED_F16_NEXT_REQUEST_REHYDRATION_POLICY],
    ["dtype", "f32"],
    ["element_count", 262_144],
    ["payload_bytes", 1_048_576],
    ["device_to_host_readback_bytes", 2_097_152],
    ["host_to_device_upload_bytes", 1_048_576],
    ["total_transfer_bytes", 3_145_728],
    ["all_finite", true],
    ["not_all_zero", true],
    ["digest_matches", true],
    ["wrapper_cached_stages_before_clear", 0],
    ["wrapper_cached_stages_after_clear", 0],
    ["synchronization_pending_before_cleanup", false],
    ["synchronization_pending_after_cleanup", false],
    ["rope_cache_cleared", true],
    ["cleanup_completed", true],
    ["preload_attempt_count", expectedPreloadAttemptCount],
    ["expected_next_request_preload_attempt_count", expectedPreloadAttemptCount + 1],
  ]) {
    if (!Object.is(handoff[field], expected)) {
      failures.push(
        `${label} DMD-to-VAE handoff ${field}=${report(handoff[field])}, expected ${report(expected)}`,
      );
    }
  }
  if (JSON.stringify(handoff.shape) !== JSON.stringify([1, 16, 128, 128])) {
    failures.push(`${label} DMD-to-VAE handoff latent shape is not exact 1024-square geometry`);
  }
  if (
    !/^[a-f0-9]{64}$/.test(String(handoff.before_sha256 ?? "")) ||
    handoff.before_sha256 !== handoff.after_sha256
  ) {
    failures.push(`${label} DMD-to-VAE handoff latent digest is missing or changed after reupload`);
  }
  failures.push(
    ...validatePackedF16CacheEvidence(
      handoff.packed_cache_before_cleanup,
      `${label} pre-cleanup packed cache`,
    ),
    ...validateEmptyPackedF16CacheEvidence(
      handoff.packed_cache_after_cleanup,
      `${label} post-cleanup packed cache`,
    ),
  );
  const dmdCompleted = progressEvents?.find(
    (progress) => progress?.event === "stage_completed" && progress?.stage === "dmd",
  );
  const vaeStarted = progressEvents?.find(
    (progress) => progress?.event === "stage_started" && progress?.stage === "vae-decode",
  );
  if (
    !positiveFinite(event.at_ms) ||
    JSON.stringify(event.run_id) !== JSON.stringify(dmdCompleted?.run_id) ||
    JSON.stringify(event.run_id) !== JSON.stringify(vaeStarted?.run_id) ||
    !positiveFinite(dmdCompleted?.at_ms) ||
    !positiveFinite(vaeStarted?.at_ms) ||
    event.at_ms <= dmdCompleted.at_ms ||
    event.at_ms >= vaeStarted.at_ms
  ) {
    failures.push(`${label} DMD-to-VAE handoff is not bound between DMD completion and VAE start`);
  }
  return failures;
}

export function validatePackedF16QwenHostEmbedding(
  event,
  progressEvents,
  label = "Generate request",
) {
  const failures = exactObjectKeys(
    event,
    ["at_ms", "event", "run_id", "report"],
    `${label} Qwen host-embedding event`,
  );
  if (failures.length > 0) return failures;
  if (event.event !== "packed_f16_qwen_host_embedding") {
    failures.push(`${label} Qwen host-embedding event name is incorrect`);
  }
  if (!positiveFinite(event.at_ms)) failures.push(`${label} Qwen host-embedding timestamp is missing`);
  const embedding = event.report;
  failures.push(
    ...exactObjectKeys(
      embedding,
      [
        "policy",
        "shape",
        "dtype",
        "input_token_count",
        "unique_token_count",
        "plan_chunk_count",
        "authenticated_object_count",
        "authenticated_object_bytes",
        "authenticated_f16_payload_bytes",
        "selected_row_occurrences",
        "selected_unique_rows",
        "selected_f16_bytes",
        "host_f32_payload_bytes",
        "host_to_device_upload_bytes",
        "immediate_device_to_host_readback_bytes",
        "total_device_transfer_bytes",
        "host_f32_sha256",
        "device_f32_sha256",
        "device_roundtrip_verified_before_text",
        "device_roundtrip_digest_matches",
        "all_finite",
        "not_all_zero",
        "coverage_complete",
      ],
      `${label} Qwen host-embedding report`,
    ),
  );
  if (failures.length > 0) return failures;
  if (embedding.policy !== TURBO_PACKED_F16_QWEN_HOST_EMBEDDING_POLICY) {
    failures.push(`${label} Qwen host-embedding policy is incorrect`);
  }
  if (JSON.stringify(embedding.shape) !== JSON.stringify([1, 45, 4096])) {
    failures.push(`${label} Qwen host-embedding shape is not exact`);
  }
  for (const [field, expected] of [
    ["dtype", "f32"],
    ["input_token_count", 45],
    ["unique_token_count", 33],
    ["plan_chunk_count", TURBO_PACKED_F16_QWEN_HOST_EMBEDDING_PLAN.expected_chunk_count],
    ["authenticated_object_count", TURBO_PACKED_F16_QWEN_HOST_EMBEDDING_PLAN.expected_object_count],
    ["authenticated_object_bytes", TURBO_PACKED_F16_QWEN_HOST_EMBEDDING_PLAN.authenticated_object_bytes],
    ["authenticated_f16_payload_bytes", TURBO_PACKED_F16_QWEN_HOST_EMBEDDING_PLAN.authenticated_f16_payload_bytes],
    ["selected_row_occurrences", 45],
    ["selected_unique_rows", 33],
    ["selected_f16_bytes", TURBO_PACKED_F16_QWEN_HOST_EMBEDDING_F16_BYTES],
    ["host_f32_payload_bytes", TURBO_PACKED_F16_QWEN_HOST_EMBEDDING_F32_BYTES],
    ["host_to_device_upload_bytes", TURBO_PACKED_F16_QWEN_HOST_EMBEDDING_F32_BYTES],
    ["immediate_device_to_host_readback_bytes", TURBO_PACKED_F16_QWEN_HOST_EMBEDDING_F32_BYTES],
    ["total_device_transfer_bytes", TURBO_PACKED_F16_QWEN_HOST_EMBEDDING_F32_BYTES * 2],
    ["host_f32_sha256", TURBO_PACKED_F16_QWEN_HOST_EMBEDDING_SHA256],
    ["device_f32_sha256", TURBO_PACKED_F16_QWEN_HOST_EMBEDDING_SHA256],
    ["device_roundtrip_verified_before_text", true],
    ["device_roundtrip_digest_matches", true],
    ["all_finite", true],
    ["not_all_zero", true],
    ["coverage_complete", true],
  ]) {
    if (!Object.is(embedding[field], expected)) {
      failures.push(`${label} Qwen host-embedding ${field}=${report(embedding[field])}, expected ${report(expected)}`);
    }
  }
  const qwenStarted = progressEvents?.find(
    (progress) => progress?.event === "stage_started" && progress?.stage === "qwen",
  );
  const qwenCompleted = progressEvents?.find(
    (progress) => progress?.event === "stage_completed" && progress?.stage === "qwen",
  );
  if (
    JSON.stringify(event.run_id) !== JSON.stringify(qwenStarted?.run_id) ||
    JSON.stringify(event.run_id) !== JSON.stringify(qwenCompleted?.run_id) ||
    !positiveFinite(qwenStarted?.at_ms) ||
    !positiveFinite(qwenCompleted?.at_ms) ||
    event.at_ms <= qwenStarted.at_ms ||
    event.at_ms >= qwenCompleted.at_ms
  ) {
    failures.push(`${label} Qwen host-embedding event is not inside the Qwen progress window`);
  }
  return failures;
}

export function validatePackedF16QwenBlock0PostSyncDiagnostic(
  event,
  progressEvents,
  label = "Generate request",
) {
  const failures = exactObjectKeys(
    event,
    ["at_ms", "event", "run_id", "diagnostic"],
    `${label} Qwen block-0 post-sync event`,
  );
  if (failures.length > 0) return failures;
  if (event.event !== "packed_f16_qwen_block0_post_sync_diagnostic") {
    failures.push(`${label} Qwen block-0 post-sync event name is incorrect`);
  }
  const diagnostic = event.diagnostic;
  failures.push(
    ...exactObjectKeys(
      diagnostic,
      [
        "scope",
        "block0_execution_mode",
        "text_layer_allocation_policy",
        "text_block_load_synchronization_policy",
        "qwen_text_layer_submission_policy",
        "tensor",
        "all_finite",
        "not_all_zero",
      ],
      `${label} Qwen block-0 post-sync diagnostic`,
    ),
  );
  if (failures.length > 0) return failures;
  if (diagnostic.scope !== TURBO_PACKED_F16_QWEN_BLOCK0_POST_SYNC_SCOPE) {
    failures.push(`${label} Qwen block-0 post-sync scope is incorrect`);
  }
  if (
    ![
      TURBO_QWEN_BLOCK0_SERIALIZED_DIAGNOSTIC_MODE,
      TURBO_QWEN_BLOCK0_ORDINARY_MODE,
    ].includes(diagnostic.block0_execution_mode)
  ) {
    failures.push(`${label} Qwen block-0 post-sync execution mode is invalid`);
  }
  if (diagnostic.text_layer_allocation_policy !== TURBO_QWEN_TEXT_LAYER_ALLOCATION_POLICY) {
    failures.push(`${label} Qwen block-0 post-sync allocator policy is incorrect`);
  }
  if (
    diagnostic.text_block_load_synchronization_policy !==
    TURBO_QWEN_TEXT_BLOCK_LOAD_SYNCHRONIZATION_POLICY
  ) {
    failures.push(`${label} Qwen block-0 post-sync load barrier policy is incorrect`);
  }
  if (diagnostic.qwen_text_layer_submission_policy !== TURBO_QWEN_TEXT_LAYER_SUBMISSION_POLICY) {
    failures.push(`${label} Qwen block-0 post-sync submission policy is incorrect`);
  }
  failures.push(
    ...validatePackedF16F32Diagnostic(
      diagnostic.tensor,
      "qwen_text_block_00_output_immediate_post_sync",
      (shape) => shape.length === 3 && shape[0] === 1 && positiveInteger(shape[1]) && shape[2] === 4096,
      `${label} Qwen block-0 immediate tensor`,
    ),
  );
  if (diagnostic.all_finite !== true || diagnostic.not_all_zero !== true) {
    failures.push(`${label} Qwen block 0 is non-finite or all-zero immediately after sync`);
  }
  const qwenStarted = progressEvents?.find(
    (progress) => progress?.event === "stage_started" && progress?.stage === "qwen",
  );
  const qwenCompleted = progressEvents?.find(
    (progress) => progress?.event === "stage_completed" && progress?.stage === "qwen",
  );
  if (
    !positiveFinite(event.at_ms) ||
    JSON.stringify(event.run_id) !== JSON.stringify(qwenStarted?.run_id) ||
    JSON.stringify(event.run_id) !== JSON.stringify(qwenCompleted?.run_id) ||
    !positiveFinite(qwenStarted?.at_ms) ||
    !positiveFinite(qwenCompleted?.at_ms) ||
    event.at_ms <= qwenStarted.at_ms ||
    event.at_ms >= qwenCompleted.at_ms
  ) {
    failures.push(`${label} Qwen block-0 post-sync event is not bound inside its Qwen run`);
  }
  return failures;
}

export function validatePackedF16QwenBlock0ExecutionDiagnostics(
  event,
  progressEvents,
  label = "Generate request",
) {
  const failures = exactObjectKeys(
    event,
    ["at_ms", "event", "run_id", "diagnostics"],
    `${label} Qwen block-0 execution event`,
  );
  if (failures.length > 0) return failures;
  if (event.event !== "packed_f16_qwen_block0_execution_diagnostics") {
    failures.push(`${label} Qwen block-0 execution event name is incorrect`);
  }
  const diagnostics = event.diagnostics;
  failures.push(
    ...exactObjectKeys(
      diagnostics,
      [
        "scope",
        "block0_execution_mode",
        "text_layer_allocation_policy",
        "text_block_load_synchronization_policy",
        "qwen_text_layer_submission_policy",
        "expected_boundary_count",
        "captured_boundary_count",
        "boundaries",
        "boundary_names_exact",
        "all_captured_tensors_finite",
        "no_captured_tensor_all_zero",
        "identity_add_canary_matches_input",
        "complete",
        "first_failure_boundary",
        "failure_reason",
      ],
      `${label} Qwen block-0 execution diagnostics`,
    ),
  );
  if (failures.length > 0) return failures;
  if (diagnostics.scope !== TURBO_PACKED_F16_QWEN_BLOCK0_EXECUTION_SCOPE) {
    failures.push(`${label} Qwen block-0 execution scope is incorrect`);
  }
  if (diagnostics.block0_execution_mode !== TURBO_QWEN_BLOCK0_SERIALIZED_DIAGNOSTIC_MODE) {
    failures.push(`${label} Qwen block-0 execution report is not serialized-diagnostic`);
  }
  if (diagnostics.text_layer_allocation_policy !== TURBO_QWEN_TEXT_LAYER_ALLOCATION_POLICY) {
    failures.push(`${label} Qwen block-0 execution allocator policy is incorrect`);
  }
  if (
    diagnostics.text_block_load_synchronization_policy !==
    TURBO_QWEN_TEXT_BLOCK_LOAD_SYNCHRONIZATION_POLICY
  ) {
    failures.push(`${label} Qwen block-0 execution load barrier policy is incorrect`);
  }
  if (diagnostics.qwen_text_layer_submission_policy !== TURBO_QWEN_TEXT_LAYER_SUBMISSION_POLICY) {
    failures.push(`${label} Qwen block-0 execution submission policy is incorrect`);
  }
  if (
    diagnostics.expected_boundary_count !== TURBO_PACKED_F16_QWEN_BLOCK0_BOUNDARIES.length ||
    diagnostics.captured_boundary_count !== TURBO_PACKED_F16_QWEN_BLOCK0_BOUNDARIES.length ||
    !Array.isArray(diagnostics.boundaries) ||
    diagnostics.boundaries.length !== TURBO_PACKED_F16_QWEN_BLOCK0_BOUNDARIES.length
  ) {
    failures.push(`${label} Qwen block-0 execution boundary coverage is incomplete`);
  }
  if (Array.isArray(diagnostics.boundaries)) {
    diagnostics.boundaries.forEach((boundary, index) => {
      failures.push(
        ...exactObjectKeys(
          boundary,
          ["sequence_index", "boundary", "tensor_kind", "tensor", "all_finite", "not_all_zero"],
          `${label} Qwen block-0 boundary ${index}`,
        ),
      );
      const expected = TURBO_PACKED_F16_QWEN_BLOCK0_BOUNDARIES[index];
      if (
        boundary?.sequence_index !== index ||
        boundary?.boundary !== expected?.boundary ||
        boundary?.tensor_kind !== expected?.tensor_kind
      ) {
        failures.push(`${label} Qwen block-0 boundary ${index} identity/order is incorrect`);
      }
      const expectedShape = expected?.tensor_kind === "parameter-sentinel"
        ? (shape) => JSON.stringify(shape) === JSON.stringify([4096])
        : (shape) =>
            shape.length === 3 && shape[0] === 1 && positiveInteger(shape[1]) && shape[2] === 4096;
      failures.push(
        ...validatePackedF16F32Diagnostic(
          boundary?.tensor,
          `qwen_text_block_00_${expected?.boundary}_immediate_post_sync`,
          expectedShape,
          `${label} Qwen block-0 boundary ${index} tensor`,
        ),
      );
      if (boundary?.all_finite !== true || boundary?.not_all_zero !== true) {
        failures.push(`${label} Qwen block-0 boundary ${index} is non-finite or all-zero`);
      }
    });
  }
  const layerInput = diagnostics.boundaries?.[0]?.tensor;
  const identityCanary = diagnostics.boundaries?.[2]?.tensor;
  if (
    diagnostics.identity_add_canary_matches_input !== true ||
    identityCanary?.sha256 !== layerInput?.sha256 ||
    JSON.stringify(identityCanary?.shape) !== JSON.stringify(layerInput?.shape)
  ) {
    failures.push(`${label} Qwen block-0 add-zero canary differs from its layer input`);
  }
  if (
    diagnostics.boundary_names_exact !== true ||
    diagnostics.all_captured_tensors_finite !== true ||
    diagnostics.no_captured_tensor_all_zero !== true ||
    diagnostics.complete !== true ||
    diagnostics.first_failure_boundary !== null ||
    diagnostics.failure_reason !== null
  ) {
    failures.push(`${label} Qwen block-0 execution summary is incomplete or failed`);
  }
  const qwenStarted = progressEvents?.find(
    (progress) => progress?.event === "stage_started" && progress?.stage === "qwen",
  );
  const qwenCompleted = progressEvents?.find(
    (progress) => progress?.event === "stage_completed" && progress?.stage === "qwen",
  );
  if (
    !positiveFinite(event.at_ms) ||
    JSON.stringify(event.run_id) !== JSON.stringify(qwenStarted?.run_id) ||
    JSON.stringify(event.run_id) !== JSON.stringify(qwenCompleted?.run_id) ||
    !positiveFinite(qwenStarted?.at_ms) ||
    !positiveFinite(qwenCompleted?.at_ms) ||
    event.at_ms <= qwenStarted.at_ms ||
    event.at_ms >= qwenCompleted.at_ms
  ) {
    failures.push(`${label} Qwen block-0 execution event is not bound inside its Qwen run`);
  }
  return failures;
}

export function validatePackedF16QwenPreHandoffDiagnostics(
  event,
  progressEvents,
  label = "Generate request",
) {
  const failures = exactObjectKeys(
    event,
    ["at_ms", "event", "run_id", "diagnostics"],
    `${label} Qwen pre-handoff event`,
  );
  if (failures.length > 0) return failures;
  if (event.event !== "packed_f16_qwen_pre_handoff_diagnostics") {
    failures.push(`${label} Qwen pre-handoff event name is incorrect`);
  }
  if (!positiveFinite(event.at_ms)) failures.push(`${label} Qwen pre-handoff timestamp is missing`);
  const diagnostics = event.diagnostics;
  failures.push(
    ...exactObjectKeys(
      diagnostics,
      [
        "scope",
        "effective_instruction_length",
        "expected_stage_output_count",
        "stage_outputs",
        "stage_names_exact",
        "qwen_last_hidden_state_before_trim",
        "instruction_after_trim_cast_before_handoff",
        "all_tensors_finite",
        "no_tensor_all_zero",
        "first_non_finite_tensor",
        "first_all_zero_tensor",
        "final_norm_matches_returned_output",
        "block_00_immediate_post_sync",
        "block_00_immediate_matches_delayed_capture",
      ],
      `${label} Qwen pre-handoff diagnostics`,
    ),
  );
  if (failures.length > 0) return failures;
  if (diagnostics.scope !== TURBO_PACKED_F16_QWEN_PRE_HANDOFF_SCOPE) {
    failures.push(`${label} Qwen pre-handoff scope is incorrect`);
  }
  if (diagnostics.effective_instruction_length !== 45) {
    failures.push(`${label} Qwen pre-handoff effective instruction length is not exact`);
  }
  if (
    diagnostics.expected_stage_output_count !== TURBO_PACKED_F16_QWEN_STAGE_OUTPUT_COUNT ||
    !Array.isArray(diagnostics.stage_outputs) ||
    diagnostics.stage_outputs.length !== TURBO_PACKED_F16_QWEN_STAGE_OUTPUT_COUNT ||
    diagnostics.stage_names_exact !== true
  ) {
    failures.push(`${label} Qwen pre-handoff stage coverage is not exact`);
  }
  const expectedStageNames = [
    "qwen_embedding_output",
    ...Array.from({ length: 36 }, (_, index) => `qwen_text_block_${String(index).padStart(2, "0")}_output`),
    "qwen_final_norm_output",
  ];
  const stageShape = (shape) =>
    shape.length === 3 && shape[0] === 1 && positiveInteger(shape[1]) && shape[2] === 4096;
  if (Array.isArray(diagnostics.stage_outputs)) {
    diagnostics.stage_outputs.forEach((tensor, index) => {
      failures.push(
        ...validatePackedF16F32Diagnostic(
          tensor,
          expectedStageNames[index],
          stageShape,
          `${label} Qwen stage ${index}`,
        ),
      );
    });
  }
  failures.push(
    ...validatePackedF16F32Diagnostic(
      diagnostics.qwen_last_hidden_state_before_trim,
      "qwen_last_hidden_state_before_trim",
      stageShape,
      `${label} returned Qwen hidden state`,
    ),
  );
  const effectiveLength = diagnostics.effective_instruction_length;
  failures.push(
    ...validatePackedF16F32Diagnostic(
      diagnostics.instruction_after_trim_cast_before_handoff,
      "instruction_after_trim_cast_before_handoff",
      (shape) =>
        shape.length === 3 && shape[0] === 1 && shape[1] === effectiveLength && shape[2] === 4096,
      `${label} instruction before handoff`,
    ),
  );
  const finalStage = diagnostics.stage_outputs?.at(-1);
  if (
    diagnostics.final_norm_matches_returned_output !== true ||
    finalStage?.sha256 !== diagnostics.qwen_last_hidden_state_before_trim?.sha256 ||
    JSON.stringify(finalStage?.shape) !==
      JSON.stringify(diagnostics.qwen_last_hidden_state_before_trim?.shape)
  ) {
    failures.push(`${label} final-norm observer does not match returned Qwen output`);
  }
  const immediateBlock0 = diagnostics.block_00_immediate_post_sync;
  failures.push(
    ...exactObjectKeys(
      immediateBlock0,
      [
        "scope",
        "block0_execution_mode",
        "text_layer_allocation_policy",
        "text_block_load_synchronization_policy",
        "qwen_text_layer_submission_policy",
        "tensor",
        "all_finite",
        "not_all_zero",
      ],
      `${label} embedded Qwen block-0 post-sync diagnostic`,
    ),
    ...validatePackedF16F32Diagnostic(
      immediateBlock0?.tensor,
      "qwen_text_block_00_output_immediate_post_sync",
      stageShape,
      `${label} embedded Qwen block-0 immediate tensor`,
    ),
  );
  if (
    immediateBlock0?.scope !== TURBO_PACKED_F16_QWEN_BLOCK0_POST_SYNC_SCOPE ||
    ![
      TURBO_QWEN_BLOCK0_SERIALIZED_DIAGNOSTIC_MODE,
      TURBO_QWEN_BLOCK0_ORDINARY_MODE,
    ].includes(immediateBlock0?.block0_execution_mode) ||
    immediateBlock0?.text_layer_allocation_policy !== TURBO_QWEN_TEXT_LAYER_ALLOCATION_POLICY ||
    immediateBlock0?.text_block_load_synchronization_policy !==
      TURBO_QWEN_TEXT_BLOCK_LOAD_SYNCHRONIZATION_POLICY ||
    immediateBlock0?.qwen_text_layer_submission_policy !==
      TURBO_QWEN_TEXT_LAYER_SUBMISSION_POLICY ||
    immediateBlock0?.all_finite !== true ||
    immediateBlock0?.not_all_zero !== true
  ) {
    failures.push(`${label} embedded Qwen block-0 post-sync diagnostic is invalid`);
  }
  const delayedBlock0 = diagnostics.stage_outputs?.[1];
  if (
    diagnostics.block_00_immediate_matches_delayed_capture !== true ||
    delayedBlock0?.sha256 !== diagnostics.block_00_immediate_post_sync?.tensor?.sha256
  ) {
    failures.push(`${label} immediate and delayed Qwen block-0 digests differ`);
  }
  if (
    diagnostics.all_tensors_finite !== true ||
    diagnostics.no_tensor_all_zero !== true ||
    diagnostics.first_non_finite_tensor !== null ||
    diagnostics.first_all_zero_tensor !== null
  ) {
    failures.push(`${label} Qwen pre-handoff summary reports a non-finite or all-zero tensor`);
  }
  const runStarted = progressEvents?.find((progress) => progress?.event === "run_started");
  const qwenCompleted = progressEvents?.find(
    (progress) => progress?.event === "stage_completed" && progress?.stage === "qwen",
  );
  const dmdStarted = progressEvents?.find(
    (progress) => progress?.event === "stage_started" && progress?.stage === "dmd",
  );
  if (
    JSON.stringify(event.run_id) !== JSON.stringify(runStarted?.run_id) ||
    JSON.stringify(event.run_id) !== JSON.stringify(qwenCompleted?.run_id) ||
    JSON.stringify(event.run_id) !== JSON.stringify(dmdStarted?.run_id)
  ) {
    failures.push(`${label} Qwen pre-handoff run ID differs from its progress window`);
  }
  if (
    !positiveFinite(qwenCompleted?.at_ms) ||
    !positiveFinite(dmdStarted?.at_ms) ||
    event.at_ms <= qwenCompleted.at_ms ||
    event.at_ms >= dmdStarted.at_ms
  ) {
    failures.push(`${label} Qwen pre-handoff event is not after Qwen completion and before DMD`);
  }
  return failures;
}

export function validatePackedF16QwenPostHandoffDiagnostics(
  event,
  preEvent,
  progressEvents,
  label = "Generate request",
) {
  const failures = exactObjectKeys(
    event,
    ["at_ms", "event", "run_id", "diagnostics"],
    `${label} Qwen post-handoff event`,
  );
  if (failures.length > 0) return failures;
  if (event.event !== "packed_f16_qwen_post_handoff_diagnostics") {
    failures.push(`${label} Qwen post-handoff event name is incorrect`);
  }
  const diagnostics = event.diagnostics;
  failures.push(
    ...exactObjectKeys(
      diagnostics,
      ["scope", "handoff", "instruction_after_handoff"],
      `${label} Qwen post-handoff diagnostics`,
    ),
  );
  if (failures.length > 0) return failures;
  if (diagnostics.scope !== TURBO_PACKED_F16_QWEN_POST_HANDOFF_SCOPE) {
    failures.push(`${label} Qwen post-handoff scope is incorrect`);
  }
  const handoff = diagnostics.handoff;
  failures.push(
    ...exactObjectKeys(
      handoff,
      [
        "policy",
        "qwen_release_unused_memory_after_stage",
        "qwen_text_layer_allocation_policy",
        "qwen_text_block_load_synchronization_policy",
        "qwen_text_layer_submission_policy",
        "shape",
        "dtype",
        "element_count",
        "payload_bytes",
        "device_to_host_readback_bytes",
        "host_to_device_upload_bytes",
        "total_transfer_bytes",
        "before_sha256",
        "after_sha256",
        "all_finite",
        "not_all_zero",
        "digest_matches",
        "cleanup_completed",
        "packed_cache",
      ],
      `${label} Qwen handoff report`,
    ),
  );
  if (failures.length > 0) return failures;
  if (
    handoff.policy !== TURBO_PACKED_F16_QWEN_HANDOFF_POLICY ||
    handoff.qwen_release_unused_memory_after_stage !== false ||
    handoff.qwen_text_layer_allocation_policy !== TURBO_QWEN_TEXT_LAYER_ALLOCATION_POLICY ||
    handoff.qwen_text_block_load_synchronization_policy !==
      TURBO_QWEN_TEXT_BLOCK_LOAD_SYNCHRONIZATION_POLICY ||
    handoff.qwen_text_layer_submission_policy !== TURBO_QWEN_TEXT_LAYER_SUBMISSION_POLICY ||
    handoff.dtype !== "f32" ||
    handoff.all_finite !== true ||
    handoff.not_all_zero !== true ||
    handoff.digest_matches !== true ||
    handoff.cleanup_completed !== true
  ) {
    failures.push(`${label} Qwen handoff policy or success provenance is incorrect`);
  }
  const shapeElements = Array.isArray(handoff.shape)
    ? handoff.shape.reduce((product, dimension) => product * dimension, 1)
    : Number.NaN;
  if (
    !positiveInteger(handoff.element_count) ||
    handoff.element_count !== shapeElements ||
    JSON.stringify(handoff.shape) !== JSON.stringify([1, 45, 4096]) ||
    handoff.payload_bytes !== 737_280 ||
    handoff.payload_bytes !== handoff.element_count * 4 ||
    handoff.device_to_host_readback_bytes !== handoff.payload_bytes * 2 ||
    handoff.host_to_device_upload_bytes !== handoff.payload_bytes ||
    handoff.total_transfer_bytes !== handoff.payload_bytes * 3
  ) {
    failures.push(`${label} Qwen handoff transfer accounting is inconsistent`);
  }
  if (
    !/^[a-f0-9]{64}$/.test(String(handoff.before_sha256 ?? "")) ||
    handoff.before_sha256 !== handoff.after_sha256
  ) {
    failures.push(`${label} Qwen handoff digest did not survive cleanup and reupload exactly`);
  }
  failures.push(...validatePackedF16CacheEvidence(handoff.packed_cache, `${label} handoff cache`));
  failures.push(
    ...validatePackedF16F32Diagnostic(
      diagnostics.instruction_after_handoff,
      "instruction_after_handoff",
      (shape) => JSON.stringify(shape) === JSON.stringify(handoff.shape),
      `${label} instruction after handoff`,
    ),
  );
  if (
    diagnostics.instruction_after_handoff?.sha256 !== handoff.after_sha256 ||
    preEvent?.diagnostics?.instruction_after_trim_cast_before_handoff?.sha256 !==
      handoff.before_sha256
  ) {
    failures.push(`${label} Qwen pre/post instruction digests are not bound to the handoff report`);
  }
  const dmdStarted = progressEvents?.find(
    (progress) => progress?.event === "stage_started" && progress?.stage === "dmd",
  );
  if (
    JSON.stringify(event.run_id) !== JSON.stringify(preEvent?.run_id) ||
    JSON.stringify(event.run_id) !== JSON.stringify(dmdStarted?.run_id) ||
    !positiveFinite(event.at_ms) ||
    !positiveFinite(preEvent?.at_ms) ||
    event.at_ms <= preEvent.at_ms ||
    event.at_ms >= dmdStarted.at_ms
  ) {
    failures.push(`${label} Qwen post-handoff event is not between pre-handoff and DMD start`);
  }
  return failures;
}

export function validatePackedF16PreDmdInputDiagnostics(
  event,
  progressEvents,
  label = "Generate request",
) {
  const failures = [];
  const exactKeys = (value, expected, fieldLabel) => {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      failures.push(`${label} ${fieldLabel} is missing or not an object`);
      return false;
    }
    const observed = Object.keys(value).sort();
    const frozen = [...expected].sort();
    if (JSON.stringify(observed) !== JSON.stringify(frozen)) {
      failures.push(`${label} ${fieldLabel} fields differ from the frozen schema`);
      return false;
    }
    return true;
  };

  if (!exactKeys(event, ["at_ms", "event", "run_id", "diagnostics"], "pre-DMD diagnostic event")) {
    return failures;
  }
  if (event.event !== "packed_f16_pre_dmd_input_diagnostics") {
    failures.push(`${label} pre-DMD diagnostic event name is incorrect`);
  }
  if (!positiveFinite(event.at_ms)) {
    failures.push(`${label} pre-DMD diagnostic timestamp is missing`);
  }

  const diagnostics = event.diagnostics;
  const diagnosticsKeys = [
    "scope",
    "policy",
    "dmd_steps",
    "instruction",
    "initial_latent",
    "renoise",
    "first_timestep",
    "all_inputs_finite",
  ];
  if (!exactKeys(diagnostics, diagnosticsKeys, "pre-DMD diagnostics")) return failures;
  if (diagnostics.scope !== TURBO_PACKED_F16_PRE_DMD_INPUT_SCOPE) {
    failures.push(`${label} pre-DMD diagnostic scope is incorrect`);
  }
  if (diagnostics.dmd_steps !== 4) {
    failures.push(`${label} pre-DMD diagnostic is not bound to the four-step DMD schedule`);
  }

  const policy = diagnostics.policy;
  if (
    exactKeys(
      policy,
      [
        "qwen_release_unused_memory_after_stage",
        "qwen_text_block_load_synchronization_policy",
        "qwen_text_layer_submission_policy",
        "packed_qwen_instruction_handoff_policy",
        "cleanup_completed",
        "post_cleanup_packed_cache",
      ],
      "pre-DMD policy",
    )
  ) {
    if (policy.qwen_release_unused_memory_after_stage !== false) {
      failures.push(`${label} pre-DMD policy enabled unsafe Qwen per-stage allocator cleanup`);
    }
    if (
      policy.qwen_text_block_load_synchronization_policy !==
      TURBO_QWEN_TEXT_BLOCK_LOAD_SYNCHRONIZATION_POLICY
    ) {
      failures.push(`${label} pre-DMD Qwen text-block load barrier policy is incorrect`);
    }
    if (policy.qwen_text_layer_submission_policy !== TURBO_QWEN_TEXT_LAYER_SUBMISSION_POLICY) {
      failures.push(`${label} pre-DMD Qwen text-layer submission policy is incorrect`);
    }
    if (
      policy.packed_qwen_instruction_handoff_policy !==
      TURBO_PACKED_F16_QWEN_HANDOFF_POLICY
    ) {
      failures.push(`${label} packed-F16 Qwen instruction handoff policy is incorrect`);
    }
    if (policy.cleanup_completed !== true) {
      failures.push(`${label} pre-DMD allocator cleanup did not complete`);
    }
    const cache = policy.post_cleanup_packed_cache;
    if (
      exactKeys(
        cache,
        [
          "state",
          "cache_ready",
          "cached_stages",
          "cached_objects",
          "cached_tensors",
          "cached_bytes",
        ],
        "post-cleanup packed cache",
      )
    ) {
      for (const [field, expected] of [
        ["state", "ready"],
        ["cache_ready", true],
        ["cached_stages", TURBO_PACKED_F16_CACHED_STAGES],
        ["cached_objects", TURBO_PACKED_F16_CACHED_OBJECTS],
        ["cached_tensors", TURBO_PACKED_F16_CACHED_TENSORS],
        ["cached_bytes", TURBO_PACKED_F16_RESOURCE_PLAN.retained_packed_f16_denoiser_bytes],
      ]) {
        if (!Object.is(cache[field], expected)) {
          failures.push(
            `${label} post-cleanup packed cache ${field}=${report(cache[field])}, expected ${report(expected)}`,
          );
        }
      }
    }
  }

  const expectedTensorKeys = [
    "name",
    "shape",
    "dtype",
    "element_count",
    "finite_element_count",
    "all_finite",
    "max_abs",
    "mean",
    "rms",
    "sha256",
  ];
  const validateTensor = (tensor, expectedName, expectedShape) => {
    const tensorLabel = `pre-DMD ${expectedName}`;
    if (!exactKeys(tensor, expectedTensorKeys, tensorLabel)) return;
    if (tensor.name !== expectedName) {
      failures.push(`${label} ${tensorLabel} name=${report(tensor.name)}, expected ${expectedName}`);
    }
    const shapeIsExact =
      Array.isArray(tensor.shape) &&
      (typeof expectedShape === "function"
        ? expectedShape(tensor.shape)
        : JSON.stringify(tensor.shape) === JSON.stringify(expectedShape));
    if (!shapeIsExact) {
      failures.push(`${label} ${tensorLabel} shape=${report(tensor.shape)} is incorrect`);
    }
    const shapeElementCount = Array.isArray(tensor.shape)
      ? tensor.shape.reduce(
          (product, dimension) =>
            positiveInteger(dimension) && Number.isSafeInteger(product * dimension)
              ? product * dimension
              : Number.NaN,
          1,
        )
      : Number.NaN;
    if (!positiveInteger(tensor.element_count) || tensor.element_count !== shapeElementCount) {
      failures.push(`${label} ${tensorLabel} element_count does not equal its exact shape product`);
    }
    if (tensor.dtype !== "f32") {
      failures.push(`${label} ${tensorLabel} dtype=${report(tensor.dtype)}, expected f32`);
    }
    if (
      tensor.all_finite !== true ||
      tensor.finite_element_count !== tensor.element_count
    ) {
      failures.push(`${label} ${tensorLabel} does not attest every element finite`);
    }
    for (const field of ["max_abs", "mean", "rms"]) {
      if (!Number.isFinite(tensor[field])) {
        failures.push(`${label} ${tensorLabel} ${field} is null or non-finite`);
      }
    }
    if (Number.isFinite(tensor.max_abs) && tensor.max_abs <= 0) {
      failures.push(`${label} ${tensorLabel} is all-zero or has invalid max_abs`);
    }
    if (Number.isFinite(tensor.rms) && tensor.rms < 0) {
      failures.push(`${label} ${tensorLabel} rms is negative`);
    }
    if (!/^[a-f0-9]{64}$/.test(String(tensor.sha256 ?? ""))) {
      failures.push(`${label} ${tensorLabel} SHA-256 is missing or malformed`);
    }
  };

  validateTensor(
    diagnostics.instruction,
    "instruction",
    (shape) => shape.length === 3 && shape[0] === 1 && positiveInteger(shape[1]) && shape[2] === 4096,
  );
  validateTensor(diagnostics.initial_latent, "initial_latent", [1, 16, 128, 128]);
  if (!Array.isArray(diagnostics.renoise) || diagnostics.renoise.length !== 3) {
    failures.push(`${label} pre-DMD diagnostics do not contain exactly three renoise tensors`);
  } else {
    diagnostics.renoise.forEach((tensor, index) => {
      validateTensor(tensor, `renoise_${index}`, [1, 16, 128, 128]);
    });
  }
  validateTensor(diagnostics.first_timestep, "first_timestep", [1]);
  if (diagnostics.all_inputs_finite !== true) {
    failures.push(`${label} pre-DMD aggregate input finiteness attestation is false`);
  }

  const runStarted = Array.isArray(progressEvents)
    ? progressEvents.find((progress) => progress?.event === "run_started")
    : null;
  const dmdStarted = Array.isArray(progressEvents)
    ? progressEvents.find(
        (progress) => progress?.event === "stage_started" && progress?.stage === "dmd",
      )
    : null;
  const firstDmdStep = Array.isArray(progressEvents)
    ? progressEvents.find(
        (progress) =>
          progress?.event === "step" && progress?.stage === "dmd" && progress?.step === 1,
      )
    : null;
  if (JSON.stringify(event.run_id) !== JSON.stringify(runStarted?.run_id)) {
    failures.push(`${label} pre-DMD diagnostic run ID differs from run_started`);
  }
  if (
    JSON.stringify(event.run_id) !== JSON.stringify(dmdStarted?.run_id) ||
    JSON.stringify(event.run_id) !== JSON.stringify(firstDmdStep?.run_id)
  ) {
    failures.push(`${label} pre-DMD diagnostic run ID differs from the DMD progress window`);
  }
  if (
    !positiveFinite(runStarted?.at_ms) ||
    !positiveFinite(dmdStarted?.at_ms) ||
    !positiveFinite(firstDmdStep?.at_ms) ||
    event.at_ms <= dmdStarted.at_ms ||
    event.at_ms >= firstDmdStep.at_ms
  ) {
    failures.push(`${label} pre-DMD diagnostic was not emitted after DMD start and before step 1`);
  }
  return failures;
}

function validateCdpNetworkAgainstRust(cdp, traffic, policy, terminalEvent, label) {
  const failures = [];
  if (!cdp || typeof cdp !== "object") {
    return [`independent ${label} CDP network evidence is missing`];
  }
  if (cdp.policy !== policy) {
    failures.push(`${label} CDP network policy=${report(cdp.policy)}, expected ${policy}`);
  }
  if (!Array.isArray(cdp.model_base_urls) || cdp.model_base_urls.length !== 3) {
    failures.push(`${label} CDP evidence does not bind three exact modular bases`);
  }
  if (
    !positiveFinite(cdp.window_start_epoch_ms) ||
    !positiveFinite(cdp.window_end_epoch_ms) ||
    cdp.window_end_epoch_ms <= cdp.window_start_epoch_ms ||
    cdp.terminal_event !== terminalEvent
  ) {
    failures.push(`${label} CDP evidence has an invalid event-anchored window`);
  }
  for (const field of [
    "model_response_count",
    "http_200_complete_part_response_count",
    "http_206_response_count",
    "complete_object_validated_response_count",
    "content_range_validated_response_count",
    "response_body_bytes",
    "unexpected_status_response_count",
    "missing_content_length_count",
    "invalid_content_range_response_count",
  ]) {
    if (!nonNegativeInteger(cdp[field])) {
      failures.push(`${label} CDP network ${field}=${report(cdp[field])}, expected non-negative integer`);
    }
  }
  if (
    cdp.model_response_count !== traffic?.network_requests ||
    cdp.http_200_complete_part_response_count + cdp.http_206_response_count !==
      traffic?.network_requests ||
    cdp.complete_object_validated_response_count +
        cdp.content_range_validated_response_count !==
      traffic?.network_requests
  ) {
    failures.push(`${label} CDP exact bounded response count does not match Rust network_requests`);
  }
  if (cdp.response_body_bytes !== traffic?.network_response_bytes) {
    failures.push(`${label} CDP exact Content-Length bytes do not match Rust network_response_bytes`);
  }
  for (const field of [
    "unexpected_status_response_count",
    "missing_content_length_count",
    "invalid_content_range_response_count",
  ]) {
    if (cdp[field] !== 0) {
      failures.push(`${label} CDP network ${field}=${report(cdp[field])}, expected 0`);
    }
  }
  return failures;
}

export function validateRenderedModelTransportEvidence(transport) {
  const failures = [];
  if (
    transport?.policy !==
    "exact-local-modular-part-only-parent-plus-qwen-vae-siblings-range-cors"
  ) {
    failures.push("model transport does not identify the canonical part-only modular policy");
  }
  if (!Array.isArray(transport?.bundles) || transport.bundles.length !== 3) {
    return [...failures, "model transport does not contain exactly three modular bundles"];
  }
  const expectedBundles = new Set([
    "boogu-image-0.1-turbo",
    "qwen3-vl-8b-base-boogu-image-0.1",
    "flux1-vae-boogu-image-0.1",
  ]);
  let logicalFiles = 0;
  let logicalBytes = 0;
  let physicalParts = 0;
  let physicalBytes = 0;
  let maximumPhysicalPartBytes = 0;
  for (const bundle of transport.bundles) {
    if (!expectedBundles.delete(bundle?.bundle)) {
      failures.push(`model transport contains unexpected bundle ${report(bundle?.bundle)}`);
    }
    if (
      bundle?.transport_sidecar?.path !== ARTIFACT_TRANSPORT_LAYOUT_PATH ||
      bundle?.transport_sidecar?.authenticated !== true ||
      !/^[0-9a-f]{64}$/.test(bundle?.transport_sidecar?.sha256 ?? "")
    ) {
      failures.push(`${report(bundle?.bundle)} transport sidecar is not SHA-256 authenticated`);
    }
    const logical = bundle?.logical_artifacts;
    const physical = bundle?.physical_transport;
    if (
      !positiveInteger(logical?.file_count) ||
      !positiveInteger(logical?.weight_file_count) ||
      !positiveInteger(logical?.bytes) ||
      !positiveInteger(logical?.weight_bytes)
    ) {
      failures.push(`${report(bundle?.bundle)} logical artifact inventory is invalid`);
    }
    if (
      physical?.target_part_bytes !== ARTIFACT_TRANSPORT_TARGET_PART_BYTES ||
      physical?.hard_max_part_bytes !== ARTIFACT_TRANSPORT_HARD_MAX_PART_BYTES ||
      !positiveInteger(physical?.unique_part_count) ||
      !positiveInteger(physical?.unique_part_bytes) ||
      !positiveInteger(physical?.max_part_bytes) ||
      physical.max_part_bytes > ARTIFACT_TRANSPORT_HARD_MAX_PART_BYTES ||
      physical?.reconstructed_bytes !== logical?.weight_bytes ||
      physical?.every_part_statted !== true ||
      physical?.part_sha256_policy !== "verified-by-browser-runtime-before-use"
    ) {
      failures.push(`${report(bundle?.bundle)} physical transport inventory is invalid`);
    }
    logicalFiles += logical?.file_count ?? 0;
    logicalBytes += logical?.bytes ?? 0;
    physicalParts += physical?.unique_part_count ?? 0;
    physicalBytes += physical?.unique_part_bytes ?? 0;
    maximumPhysicalPartBytes = Math.max(
      maximumPhysicalPartBytes,
      physical?.max_part_bytes ?? 0,
    );
  }
  if (expectedBundles.size !== 0) failures.push("model transport omits a required modular bundle");
  for (const [field, expected] of [
    ["total_logical_artifact_files", logicalFiles],
    ["total_logical_artifact_bytes", logicalBytes],
    ["total_physical_transport_parts", physicalParts],
    ["total_physical_transport_part_bytes", physicalBytes],
    ["maximum_physical_transport_part_bytes", maximumPhysicalPartBytes],
  ]) {
    if (transport?.[field] !== expected) {
      failures.push(`model transport ${field}=${report(transport?.[field])}, expected ${expected}`);
    }
  }
  if (transport?.validated !== true) failures.push("model transport is not validated");
  return failures;
}

export function validateTurbo1024ModelEvidence(evidence) {
  const failures = [];
  failures.push(...validateRenderedModelTransportEvidence(evidence?.modular_artifact_transport));
  failures.push(
    ...validateRenderedRuntimeWebGpuEvidence(
      evidence?.bevy_backend_ready,
      evidence?.runtime_webgpu_calls,
      evidence?.engine_session_id,
      evidence?.runtime_webgpu_dropped_calls,
      evidence?.runtime_webgpu_adapter_attestation,
      BOOGU_WEB_REQUIRED_DEVICE_FEATURES,
    ).map((failure) => `engine WebGPU attestation: ${failure}`),
  );
  const runtimeEvents = evidence?.runtime_events;
  const progressEvents = evidence?.progress_events;
  const uiContract = evidence?.ui_contract;
  const output = evidence?.output_ready;
  const runStarted = Array.isArray(progressEvents)
    ? progressEvents.find(
        (event) => event?.event === "run_started" && event?.model === TURBO_MODEL_ID,
      )
    : null;
  failures.push(...validateRequestScopedSurfaceGate(evidence));

  if (!Array.isArray(runtimeEvents)) {
    failures.push("runtime_events is not an array");
  } else {
    const failed = runtimeEvents.filter((event) => event?.event === "failed");
    if (failed.length > 0) failures.push(`browser model runtime failed: ${report(failed)}`);
    const ready = runtimeEvents.find(
      (event) => event?.event === "ready" && event?.model === TURBO_MODEL_ID,
    );
    const block0ExecutionMode = ready?.block0_execution_mode;
    if (!ready) failures.push("browser model runtime did not emit Turbo ready");
    else if (
      ![
        TURBO_QWEN_BLOCK0_SERIALIZED_DIAGNOSTIC_MODE,
        TURBO_QWEN_BLOCK0_ORDINARY_MODE,
      ].includes(block0ExecutionMode) ||
      ready.qwen_text_layer_allocation_policy !== TURBO_QWEN_TEXT_LAYER_ALLOCATION_POLICY ||
      ready.qwen_text_block_load_synchronization_policy !==
        TURBO_QWEN_TEXT_BLOCK_LOAD_SYNCHRONIZATION_POLICY ||
      ready.qwen_text_layer_submission_policy !== TURBO_QWEN_TEXT_LAYER_SUBMISSION_POLICY
    ) {
      failures.push("browser model runtime ready event has the wrong Qwen text execution policy");
    }
    if (evidence?.qwen_block0_execution_mode !== block0ExecutionMode) {
      failures.push("requested Qwen block-0 execution mode differs from runtime ready evidence");
    }
    const plans = runtimeEvents.filter((event) => event?.event === "packed_f16_resource_plan");
    if (plans.length !== 1) {
      failures.push(
        `expected exactly one packed-F16 Turbo resource plan, found ${plans.length}`,
      );
    } else {
      const plan = plans[0];
      for (const [field, expected] of Object.entries(TURBO_PACKED_F16_RESOURCE_PLAN)) {
        if (plan[field] !== expected) {
          failures.push(
            `packed-F16 plan ${field}=${report(plan[field])}, expected exact Turbo value ${report(expected)}`,
          );
        }
      }
      for (const field of Object.keys(TURBO_PACKED_F16_RESOURCE_PLAN).filter(
        (field) =>
          ![
            "on_device_quantized_execution_claimed",
            "qwen_text_layer_allocation_policy",
            "qwen_text_block_load_synchronization_policy",
            "qwen_text_layer_submission_policy",
            "qwen_text_layer_persistent_pool_requires_measured_gpu_gate",
          ].includes(field),
      )) {
        if (!nonNegativeInteger(plan[field])) {
          failures.push(
            `packed-F16 plan ${field}=${report(plan[field])}, expected a non-negative integer`,
          );
        }
      }
      if (
        plan.canonical_compact_f16_payload_bytes + plan.inserted_padding_elements * 2 !==
          plan.retained_packed_f16_denoiser_bytes ||
        plan.padded_f16_elements * 2 !== plan.retained_packed_f16_denoiser_bytes ||
        plan.padded_f16_elements * 4 !== plan.materialized_f32_bytes_per_dmd_step
      ) {
        failures.push("packed-F16 compact, padded, retained, and dense-F32 bytes are inconsistent");
      }
      if (
        plan.retained_packed_f16_denoiser_bytes + plan.preload_workspace_bytes !==
        plan.preload_peak_bytes
      ) {
        failures.push("packed-F16 preload peak does not equal retained bytes plus workspace");
      }
      if (
        plan.retained_packed_f16_denoiser_bytes +
          plan.max_materialized_stage_f32_bytes +
          plan.activation_reserve_bytes !==
        plan.conservative_planned_device_bytes
      ) {
        failures.push("packed-F16 conservative inference plan arithmetic is inconsistent");
      }
      if (
        plan.expected_stage_count * 4 !== plan.expected_stage_materializations_per_request ||
        plan.expected_object_count * 4 !== plan.expected_object_unpacks_per_request ||
        plan.retained_packed_f16_denoiser_bytes * 4 !==
          plan.expected_packed_read_bytes_per_request ||
        plan.materialized_f32_bytes_per_dmd_step * 4 !==
          plan.expected_f32_write_bytes_per_request
      ) {
        failures.push("packed-F16 four-step per-request materialization plan is inconsistent");
      }
      if (
        plan.preload_peak_bytes >= plan.strict_device_cap_bytes ||
        plan.conservative_planned_device_bytes >= plan.strict_device_cap_bytes
      ) {
        failures.push("packed-F16 preload or inference plan is not strictly below its device cap");
      }
      if (plan.strict_device_cap_bytes !== LOW_VRAM_DEVICE_CAP_BYTES) {
        failures.push(
          `packed-F16 strict cap=${report(plan.strict_device_cap_bytes)}, expected decimal 32 GB boundary`,
        );
      }
      if (plan.on_device_quantized_execution_claimed !== false) {
        failures.push("packed-F16 storage plan incorrectly claims on-device quantized execution");
      }
    }

    const preloadEvents = runtimeEvents.filter(
      (event) => event?.event === "packed_f16_denoiser_preload",
    );
    if (preloadEvents.length !== 1) {
      failures.push(`expected exactly one packed-F16 preload event, found ${preloadEvents.length}`);
    } else {
      const preload = preloadEvents[0];
      failures.push(
        ...exactObjectKeys(
          preload,
          [
            "at_ms",
            "event",
            "traffic",
            "cached_stages",
            "cached_objects",
            "cached_tensors",
            "cached_bytes",
            "previous_preload_attempt_count",
            "preload_attempt_count",
            "request_scoped_rehydration",
            "rehydration_policy",
          ],
          "packed-F16 denoiser preload event",
        ),
      );
      const runStarted = Array.isArray(progressEvents)
        ? progressEvents.find(
            (event) => event?.event === "run_started" && event?.model === TURBO_MODEL_ID,
          )
        : null;
      if (
        !positiveFinite(preload.at_ms) ||
        !positiveFinite(runStarted?.at_ms) ||
        preload.at_ms >= runStarted.at_ms
      ) {
        failures.push("packed-F16 preload did not complete before the Generate request window");
      }
      for (const [field, expected] of [
        ["cached_stages", TURBO_PACKED_F16_CACHED_STAGES],
        ["cached_objects", TURBO_PACKED_F16_CACHED_OBJECTS],
        ["cached_tensors", TURBO_PACKED_F16_CACHED_TENSORS],
        ["cached_bytes", TURBO_PACKED_F16_RESOURCE_PLAN.retained_packed_f16_denoiser_bytes],
        ["previous_preload_attempt_count", 0],
        ["preload_attempt_count", 1],
        ["request_scoped_rehydration", false],
        ["rehydration_policy", TURBO_PACKED_F16_NEXT_REQUEST_REHYDRATION_POLICY],
      ]) {
        if (preload[field] !== expected) {
          failures.push(
            `packed-F16 preload ${field}=${report(preload[field])}, expected ${report(expected)}`,
          );
        }
      }
      failures.push(
        ...validateExactArtifactTraffic(
          preload.traffic,
          TURBO_DENOISER_PRELOAD_TRAFFIC,
          "packed-F16 denoiser preload",
        ),
      );
      if (
        JSON.stringify(evidence?.packed_f16_denoiser_preload_traffic) !==
        JSON.stringify(preload.traffic)
      ) {
        failures.push("top-level packed-F16 preload traffic does not match the runtime event exactly");
      }
      failures.push(
        ...validateCdpNetworkAgainstRust(
          evidence?.cdp_preload_network_traffic,
          preload.traffic,
          TURBO_CDP_PRELOAD_NETWORK_POLICY,
          "packed_f16_denoiser_preload",
          "packed-F16 denoiser preload",
        ),
      );
    }

    const qwenHostEmbeddingEvents = runtimeEvents.filter(
      (event) => event?.event === "packed_f16_qwen_host_embedding",
    );
    const qwenHostEmbeddingEvent = qwenHostEmbeddingEvents[0] ?? null;
    if (qwenHostEmbeddingEvents.length !== 1) {
      failures.push(
        `expected exactly one Qwen host-embedding event, found ${qwenHostEmbeddingEvents.length}`,
      );
    } else {
      failures.push(
        ...validatePackedF16QwenHostEmbedding(
          qwenHostEmbeddingEvent,
          progressEvents,
          "Generate request",
        ),
      );
      if (
        JSON.stringify(evidence?.packed_f16_qwen_host_embedding) !==
        JSON.stringify(qwenHostEmbeddingEvent)
      ) {
        failures.push("top-level Qwen host-embedding evidence differs from its runtime event");
      }
    }

    const qwenPreHandoffEvents = runtimeEvents.filter(
      (event) => event?.event === "packed_f16_qwen_pre_handoff_diagnostics",
    );
    const qwenBlock0ExecutionEvents = runtimeEvents.filter(
      (event) => event?.event === "packed_f16_qwen_block0_execution_diagnostics",
    );
    const expectedBlock0ExecutionEventCount =
      block0ExecutionMode === TURBO_QWEN_BLOCK0_SERIALIZED_DIAGNOSTIC_MODE ? 1 : 0;
    if (qwenBlock0ExecutionEvents.length !== expectedBlock0ExecutionEventCount) {
      failures.push(
        `expected ${expectedBlock0ExecutionEventCount} Qwen block-0 execution event(s) for mode ${report(block0ExecutionMode)}, found ${qwenBlock0ExecutionEvents.length}`,
      );
    } else if (expectedBlock0ExecutionEventCount === 1) {
      failures.push(
        ...validatePackedF16QwenBlock0ExecutionDiagnostics(
          qwenBlock0ExecutionEvents[0],
          progressEvents,
          "Generate request",
        ),
      );
      if (
        JSON.stringify(evidence?.packed_f16_qwen_block0_execution_diagnostics) !==
        JSON.stringify(qwenBlock0ExecutionEvents[0])
      ) {
        failures.push("top-level Qwen block-0 execution evidence differs from its runtime event");
      }
    } else if (evidence?.packed_f16_qwen_block0_execution_diagnostics != null) {
      failures.push("ordinary block-0 execution unexpectedly exposes serialized boundary evidence");
    }
    const qwenBlock0PostSyncEvents = runtimeEvents.filter(
      (event) => event?.event === "packed_f16_qwen_block0_post_sync_diagnostic",
    );
    if (qwenBlock0PostSyncEvents.length !== 1) {
      failures.push(
        `expected exactly one Qwen block-0 post-sync event, found ${qwenBlock0PostSyncEvents.length}`,
      );
    } else {
      failures.push(
        ...validatePackedF16QwenBlock0PostSyncDiagnostic(
          qwenBlock0PostSyncEvents[0],
          progressEvents,
          "Generate request",
        ),
      );
      if (
        JSON.stringify(evidence?.packed_f16_qwen_block0_post_sync_diagnostic) !==
        JSON.stringify(qwenBlock0PostSyncEvents[0])
      ) {
        failures.push("top-level Qwen block-0 post-sync evidence differs from its runtime event");
      }
    }
    const qwenPostHandoffEvents = runtimeEvents.filter(
      (event) => event?.event === "packed_f16_qwen_post_handoff_diagnostics",
    );
    const qwenPreHandoffEvent = qwenPreHandoffEvents[0] ?? null;
    const qwenPostHandoffEvent = qwenPostHandoffEvents[0] ?? null;
    const qwenBlock0ExecutionEvent = qwenBlock0ExecutionEvents[0] ?? null;
    const qwenBlock0PostSyncEvent = qwenBlock0PostSyncEvents[0] ?? null;
    if (
      qwenBlock0PostSyncEvent &&
      qwenBlock0PostSyncEvent.diagnostic?.block0_execution_mode !== block0ExecutionMode
    ) {
      failures.push("Qwen block-0 post-sync execution mode differs from runtime ready policy");
    }
    if (
      qwenBlock0ExecutionEvent &&
      qwenBlock0PostSyncEvent &&
      (qwenBlock0ExecutionEvent.at_ms >= qwenBlock0PostSyncEvent.at_ms ||
        qwenBlock0ExecutionEvent.diagnostics?.boundaries?.at(-1)?.tensor?.sha256 !==
          qwenBlock0PostSyncEvent.diagnostic?.tensor?.sha256 ||
        qwenBlock0ExecutionEvent.diagnostics?.boundaries?.[0]?.tensor?.sha256 !==
          qwenHostEmbeddingEvent?.report?.device_f32_sha256)
    ) {
      failures.push(
        "Qwen block-0 serialized boundaries are not ordered/bound to host input and final post-sync output",
      );
    }
    if (
      qwenBlock0PostSyncEvent &&
      qwenPreHandoffEvent &&
      (qwenBlock0PostSyncEvent.at_ms >= qwenPreHandoffEvent.at_ms ||
        JSON.stringify(qwenBlock0PostSyncEvent.diagnostic) !==
          JSON.stringify(qwenPreHandoffEvent.diagnostics?.block_00_immediate_post_sync))
    ) {
      failures.push(
        "Qwen block-0 post-sync event is not ordered before and JSON-bound to pre-handoff evidence",
      );
    }
    if (qwenPreHandoffEvents.length !== 1 || qwenPostHandoffEvents.length !== 1) {
      failures.push(
        `expected exactly one Qwen pre/post handoff event, found ${qwenPreHandoffEvents.length}/${qwenPostHandoffEvents.length}`,
      );
    } else {
      failures.push(
        ...validatePackedF16QwenPreHandoffDiagnostics(
          qwenPreHandoffEvent,
          progressEvents,
          "Generate request",
        ),
        ...validatePackedF16QwenPostHandoffDiagnostics(
          qwenPostHandoffEvent,
          qwenPreHandoffEvent,
          progressEvents,
          "Generate request",
        ),
      );
      if (
        JSON.stringify(evidence?.packed_f16_qwen_pre_handoff_diagnostics) !==
          JSON.stringify(qwenPreHandoffEvent) ||
        JSON.stringify(evidence?.packed_f16_qwen_post_handoff_diagnostics) !==
          JSON.stringify(qwenPostHandoffEvent)
      ) {
        failures.push("top-level Qwen handoff diagnostics differ from their runtime events");
      }
      if (
        qwenHostEmbeddingEvent?.report?.device_f32_sha256 !==
        qwenPreHandoffEvent?.diagnostics?.stage_outputs?.[0]?.sha256
      ) {
        failures.push("Qwen host-embedding digest differs from the first streamed Qwen activation");
      }
    }

    const preDmdInputDiagnosticEvents = runtimeEvents.filter(
      (event) => event?.event === "packed_f16_pre_dmd_input_diagnostics",
    );
    const preDmdInputDiagnosticEvent = preDmdInputDiagnosticEvents[0] ?? null;
    if (preDmdInputDiagnosticEvents.length !== 1) {
      failures.push(
        `expected exactly one packed-F16 pre-DMD input diagnostic event, found ${preDmdInputDiagnosticEvents.length}`,
      );
    } else {
      failures.push(
        ...validatePackedF16PreDmdInputDiagnostics(
          preDmdInputDiagnosticEvent,
          progressEvents,
          "Generate request",
        ),
      );
      if (
        JSON.stringify(evidence?.packed_f16_pre_dmd_input_diagnostics) !==
        JSON.stringify(preDmdInputDiagnosticEvent)
      ) {
        failures.push(
          "top-level packed-F16 pre-DMD input diagnostics differ from the runtime event",
        );
      }
      if (
        preDmdInputDiagnosticEvent?.diagnostics?.instruction?.sha256 !==
        qwenPostHandoffEvent?.diagnostics?.handoff?.after_sha256
      ) {
        failures.push("pre-DMD instruction digest differs from the verified Qwen handoff");
      }
    }

    const lifecycleEvents = runtimeEvents.filter(
      (event) => event?.event === "packed_f16_denoiser_lifecycle",
    );
    const dmdVaeHandoffEvents = runtimeEvents.filter(
      (event) => event?.event === "packed_f16_dmd_vae_handoff",
    );
    if (lifecycleEvents.length !== 1) {
      failures.push(
        `expected exactly one post-DMD packed-F16 lifecycle event, found ${lifecycleEvents.length}`,
      );
    } else {
      const lifecycleEvent = lifecycleEvents[0];
      const runStarted = Array.isArray(progressEvents)
        ? progressEvents.find(
            (event) => event?.event === "run_started" && event?.model === TURBO_MODEL_ID,
          )
        : null;
      if (
        !positiveFinite(lifecycleEvent.at_ms) ||
        !positiveFinite(runStarted?.at_ms) ||
        lifecycleEvent.at_ms <= runStarted.at_ms
      ) {
        failures.push("packed-F16 lifecycle event is not inside the successful request window");
      }
      if (
        positiveFinite(preDmdInputDiagnosticEvent?.at_ms) &&
        preDmdInputDiagnosticEvent.at_ms >= lifecycleEvent.at_ms
      ) {
        failures.push("packed-F16 pre-DMD input diagnostics were not emitted before the lifecycle event");
      }
      failures.push(
        ...validatePackedF16Lifecycle(lifecycleEvent.lifecycle, "Generate request"),
      );
      if (
        JSON.stringify(evidence?.packed_f16_denoiser_lifecycle) !==
        JSON.stringify(lifecycleEvent.lifecycle)
      ) {
        failures.push("top-level packed-F16 lifecycle differs from its runtime event");
      }
    }
    if (dmdVaeHandoffEvents.length !== 1) {
      failures.push(
        `expected exactly one packed-F16 DMD-to-VAE handoff event, found ${dmdVaeHandoffEvents.length}`,
      );
    } else {
      const handoffEvent = dmdVaeHandoffEvents[0];
      failures.push(
        ...validatePackedF16DmdVaeHandoff(handoffEvent, progressEvents, 1, "Generate request"),
      );
      if (
        JSON.stringify(evidence?.packed_f16_dmd_vae_handoff) !== JSON.stringify(handoffEvent)
      ) {
        failures.push("top-level packed-F16 DMD-to-VAE handoff differs from its runtime event");
      }
      if (
        lifecycleEvents.length === 1 &&
        (!positiveFinite(lifecycleEvents[0]?.at_ms) ||
          lifecycleEvents[0].at_ms >= handoffEvent.at_ms)
      ) {
        failures.push("packed-F16 DMD lifecycle was not emitted before cache eviction");
      }
    }

    const trafficEvents = runtimeEvents.filter((event) => event?.event === "artifact_traffic");
    if (trafficEvents.length !== 1) {
      failures.push(`expected exactly one per-request artifact traffic event, found ${trafficEvents.length}`);
    } else {
      const trafficEvent = trafficEvents[0];
      const traffic = trafficEvent?.traffic;
      const runStarted = Array.isArray(progressEvents)
        ? progressEvents.find(
            (event) => event?.event === "run_started" && event?.model === TURBO_MODEL_ID,
          )
        : null;
      if (
        !positiveFinite(trafficEvent?.at_ms) ||
        !positiveFinite(runStarted?.at_ms) ||
        trafficEvent.at_ms <= runStarted.at_ms
      ) {
        failures.push("per-request artifact traffic is not contained in the Generate request window");
      }
      failures.push(
        ...validateExactArtifactTraffic(
          traffic,
          TURBO_GENERATE_REQUEST_TRAFFIC,
          "Generate request",
        ),
      );
      if (JSON.stringify(evidence?.artifact_traffic) !== JSON.stringify(traffic)) {
        failures.push("top-level request artifact traffic does not match the runtime event exactly");
      }
      failures.push(
        ...validateCdpNetworkAgainstRust(
          evidence?.cdp_network_traffic,
          traffic,
          TURBO_CDP_REQUEST_NETWORK_POLICY,
          "ready",
          "Generate request",
        ),
      );
      if (
        traffic?.object_reads !== TURBO_GENERATE_REQUEST_TRAFFIC.object_reads ||
        traffic?.range_reads !== TURBO_GENERATE_REQUEST_TRAFFIC.range_reads
      ) {
        failures.push("DMD hot path performed artifact or persistent-cache reads after denoiser preload");
      }
    }
  }

  if (!uiContract || typeof uiContract !== "object") {
    failures.push("ordinary Bevy UI contract is missing");
  } else {
    if (uiContract.event !== "ready") failures.push("ordinary Bevy UI is not ready");
    if (uiContract.model !== TURBO_MODEL_ID) {
      failures.push(`ordinary UI selected ${report(uiContract.model)}, expected Turbo`);
    }
    if (uiContract.width !== 1024 || uiContract.height !== 1024) {
      failures.push(
        `ordinary UI default dimensions=${report([uiContract.width, uiContract.height])}, expected 1024x1024`,
      );
    }
    for (const control of ["prompt", "seed", "run", "save"]) {
      if (uiContract[`${control}_enabled`] !== true) {
        failures.push(`ordinary UI ${control} control is disabled`);
      }
    }
    for (const control of ["prompt", "seed"]) {
      if (typeof uiContract[`${control}_focused`] !== "boolean") {
        failures.push(`ordinary UI ${control}_focused is not a boolean InputFocus attestation`);
      }
    }
    for (const field of [
      "prompt_x",
      "prompt_y",
      "seed_x",
      "seed_y",
      "run_x",
      "run_y",
      "save_x",
      "save_y",
    ]) {
      if (!positiveFinite(uiContract[field])) {
        failures.push(`ordinary UI ${field}=${report(uiContract[field])}, expected positive finite`);
      }
    }
  }

  if (!Array.isArray(progressEvents)) {
    failures.push("progress_events is not an array");
  } else {
    for (const terminal of ["run_failed", "run_cancelled"]) {
      const matches = progressEvents.filter((event) => event?.event === terminal);
      if (matches.length > 0) failures.push(`browser inference emitted ${terminal}: ${report(matches)}`);
    }
    const started = progressEvents.find(
      (event) => event?.event === "run_started" && event?.model === TURBO_MODEL_ID,
    );
    const completed = progressEvents.find((event) => event?.event === "run_completed");
    if (!started) failures.push("ordinary UI did not emit Turbo run_started");
    if (!completed) failures.push("ordinary UI did not emit run_completed");
    if (
      started &&
      completed &&
      JSON.stringify(started.run_id) !== JSON.stringify(completed.run_id)
    ) {
      failures.push("run_started and run_completed IDs differ");
    }
    const dmdStarted = progressEvents.find(
      (event) => event?.event === "stage_started" && event?.stage === "dmd",
    );
    const dmdSteps = progressEvents
      .filter((event) => event?.event === "step" && event?.stage === "dmd")
      .sort((left, right) => left.step - right.step);
    const dmdCompleted = progressEvents.find(
      (event) => event?.event === "stage_completed" && event?.stage === "dmd",
    );
    if (dmdStarted?.total_steps !== 4 || dmdSteps.length !== 4 || !dmdCompleted) {
      failures.push("ordinary UI did not attest one complete four-step DMD schedule");
    } else {
      for (let index = 0; index < dmdSteps.length; index += 1) {
        const step = dmdSteps[index];
        if (
          step.step !== index + 1 ||
          step.total_steps !== 4 ||
          JSON.stringify(step.run_id) !== JSON.stringify(started?.run_id)
        ) {
          failures.push(`DMD progress step ${index + 1} has an invalid index, total, or run ID`);
        }
      }
    }
  }

  if (!output || typeof output !== "object") {
    failures.push("validated/materialized output-ready evidence is missing");
  } else {
    if (!isCanonicalU64DecimalString(output.job_id)) {
      failures.push("output job ID is not a canonical u64 decimal string");
    } else if (!outputJobIdMatchesNumericRunId(output.job_id, runStarted?.run_id)) {
      failures.push("output job ID differs from the exact decimal representation of run_started");
    }
    if (output.width !== 1024 || output.height !== 1024) {
      failures.push(`output dimensions=${report([output.width, output.height])}, expected 1024x1024`);
    }
    if (output.model !== TURBO_MODEL_ID) {
      failures.push(`output model=${report(output.model)}, expected Turbo`);
    }
    if (output.backend !== LOW_VRAM_BACKEND) {
      failures.push(`output backend=${report(output.backend)}, expected ${LOW_VRAM_BACKEND}`);
    }
    if (output.artifacts_verified !== true) {
      failures.push("output provenance does not attest verified artifacts");
    }
    if (output.artifact_content_digest !== TURBO_PRODUCTION_CONTENT_DIGEST) {
      failures.push(
        `output digest=${report(output.artifact_content_digest)}, expected canonical Turbo production digest`,
      );
    }
    if (output.numeric_format !== "f16-qwen-vision-f32") {
      failures.push(
        `output numeric format=${report(output.numeric_format)}, expected production artifact identity`,
      );
    }
  }

  const interaction = evidence?.interaction;
  if (interaction?.mechanism !== "cdp-keyboard-and-mouse") {
    failures.push("the rendered-model interaction is not attributed to CDP keyboard and mouse input");
  }
  if (interaction?.prompt_typed_via_cdp !== true || interaction?.prompt_value !== evidence?.fixed_ascii_prompt) {
    failures.push("CDP did not attest typing the exact fixed ASCII prompt into EditableText");
  }
  if (interaction?.prompt_input?.focus_event?.prompt_focused !== true) {
    failures.push("Bevy InputFocus did not attest the CDP Prompt click before keyboard input");
  }
  if (
    interaction?.prompt_input?.replacement_mode !== "known-empty-direct-keyboard-entry" ||
    interaction?.prompt_input?.focus_click?.click_count !== 1 ||
    interaction?.prompt_input?.selection_clicks?.length !== 0
  ) {
    failures.push("Prompt was not entered directly into the attested-empty focused field");
  }
  if (interaction?.seed_typed_via_cdp !== true || interaction?.seed_value !== "0") {
    failures.push("CDP did not attest typing the exact seed 0 into EditableText");
  }
  if (
    interaction?.seed_intermediate_input?.focus_event?.seed_focused !== true ||
    interaction?.seed_input?.focus_event?.seed_focused !== true
  ) {
    failures.push("Bevy InputFocus did not attest the CDP Seed clicks before keyboard input");
  }
  for (const [label, input] of [
    ["intermediate", interaction?.seed_intermediate_input],
    ["final", interaction?.seed_input],
  ]) {
    if (
      input?.replacement_mode !== "bevy-editable-text-triple-click-select-all" ||
      input?.focus_click?.click_count !== 1 ||
      input?.selection_clicks?.length !== 2 ||
      input?.selection_clicks?.[0]?.click_count !== 2 ||
      input?.selection_clicks?.[1]?.click_count !== 3
    ) {
      failures.push(`Seed ${label} value did not use the real CDP triple-click selection contract`);
    }
  }
  if (interaction?.run_clicked_via_cdp !== true) {
    failures.push("CDP did not attest clicking the ordinary Bevy Run button");
  }
  if (interaction?.save_clicked_via_cdp !== true) {
    failures.push("CDP did not attest clicking the ordinary Bevy Save PNG button");
  }

  const png = evidence?.downloaded_png;
  if (!png || typeof png !== "object") {
    failures.push("production Save PNG download evidence is missing");
  } else {
    if (typeof png.path !== "string" || png.path.length === 0) {
      failures.push("downloaded PNG path is missing");
    }
    if (!/^burn-image-[0-9]+\.png$/.test(String(png.file_name ?? ""))) {
      failures.push(`downloaded PNG filename=${report(png.file_name)}, expected production name`);
    }
    if (!positiveInteger(png.bytes)) failures.push("downloaded PNG byte count is missing");
    if (!/^[a-f0-9]{64}$/.test(String(png.sha256 ?? ""))) {
      failures.push("downloaded PNG SHA-256 is missing or malformed");
    }
    if (png.signature_hex !== "89504e470d0a1a0a") {
      failures.push(`downloaded PNG signature=${report(png.signature_hex)}, expected PNG signature`);
    }
    if (
      png.ihdr?.length !== 13 ||
      png.ihdr?.type !== "IHDR" ||
      png.ihdr?.width !== 1024 ||
      png.ihdr?.height !== 1024
    ) {
      failures.push(`downloaded PNG IHDR is not an exact 1024x1024 image: ${report(png.ihdr)}`);
    }
    const willBegin = png.browser_events?.will_begin;
    const completed = png.browser_events?.completed;
    if (
      willBegin?.method !== "Browser.downloadWillBegin" ||
      typeof willBegin?.params?.guid !== "string" ||
      willBegin.params.guid.length === 0
    ) {
      failures.push("Browser.downloadWillBegin evidence is missing");
    }
    if (
      completed?.method !== "Browser.downloadProgress" ||
      completed?.params?.state !== "completed" ||
      completed?.params?.guid !== willBegin?.params?.guid
    ) {
      failures.push("matching completed Browser.downloadProgress evidence is missing");
    }
  }
  if (!Number.isSafeInteger(evidence?.canvas_png_changed_bytes) || evidence.canvas_png_changed_bytes <= 0) {
    failures.push("rendered Bevy canvas did not change after the generation request");
  }
  if (!Number.isSafeInteger(evidence?.model_screenshot_bytes) || evidence.model_screenshot_bytes < 1024) {
    failures.push("post-generation rendered-window screenshot is missing or empty");
  }

  const gpu = evidence?.native_gpu_attestation;
  if (!gpu || typeof gpu !== "object") {
    failures.push("native nvidia-smi Chrome GPU-process attestation is missing");
  } else {
    if (gpu.provider !== "nvidia-smi") {
      failures.push(`GPU attestation provider=${report(gpu.provider)}, expected nvidia-smi`);
    }
    if (gpu.interval_aggregation_policy !== GPU_INTERVAL_AGGREGATION_POLICY) {
      failures.push("GPU attestation does not aggregate every Chrome GPU PID per interval");
    }
    if (gpu.maximum_framebuffer_bytes_exclusive !== LOW_VRAM_DEVICE_CAP_BYTES) {
      failures.push(
        `GPU attestation cap=${report(gpu.maximum_framebuffer_bytes_exclusive)}, expected decimal 32 GB boundary`,
      );
    }
    if (
      !positiveInteger(gpu.observed_peak_aggregate_framebuffer_bytes) ||
      gpu.observed_peak_aggregate_framebuffer_bytes >= LOW_VRAM_DEVICE_CAP_BYTES
    ) {
      failures.push(
        `aggregate Chrome GPU framebuffer peak=${report(gpu.observed_peak_aggregate_framebuffer_bytes)}, expected positive and strictly below decimal 32 GB`,
      );
    }
    if (!positiveInteger(gpu.matched_sample_intervals)) {
      failures.push("GPU attestation did not match a Chrome GPU PID interval");
    }
    if (!positiveInteger(gpu.active_sample_intervals)) {
      failures.push("GPU attestation did not observe Chrome GPU compute activity");
    }
    if (!Array.isArray(gpu.observed_gpu_processes) || gpu.observed_gpu_processes.length === 0) {
      failures.push("GPU attestation did not inventory a Chrome GPU-process descendant");
    }
    if (gpu.validated !== true || !Array.isArray(gpu.validation_failures)) {
      failures.push("GPU attestation was not independently validated");
    } else if (gpu.validation_failures.length > 0) {
      failures.push(`GPU attestation failed: ${report(gpu.validation_failures)}`);
    }
  }
  return failures;
}

function validateProductionPngDownload(png, label) {
  const failures = [];
  if (!png || typeof png !== "object") {
    return [`${label} production Save PNG download evidence is missing`];
  }
  if (typeof png.path !== "string" || png.path.length === 0) {
    failures.push(`${label} downloaded PNG path is missing`);
  }
  if (!/^burn-image-[0-9]+\.png$/.test(String(png.file_name ?? ""))) {
    failures.push(`${label} downloaded PNG filename=${report(png.file_name)}, expected production name`);
  }
  if (!positiveInteger(png.bytes)) failures.push(`${label} downloaded PNG byte count is missing`);
  if (!/^[a-f0-9]{64}$/.test(String(png.sha256 ?? ""))) {
    failures.push(`${label} downloaded PNG SHA-256 is missing or malformed`);
  }
  if (png.signature_hex !== "89504e470d0a1a0a") {
    failures.push(`${label} downloaded PNG signature=${report(png.signature_hex)}, expected PNG signature`);
  }
  if (
    png.ihdr?.length !== 13 ||
    png.ihdr?.type !== "IHDR" ||
    png.ihdr?.width !== 1024 ||
    png.ihdr?.height !== 1024
  ) {
    failures.push(`${label} downloaded PNG IHDR is not an exact 1024x1024 image: ${report(png.ihdr)}`);
  }
  const willBegin = png.browser_events?.will_begin;
  const completed = png.browser_events?.completed;
  if (
    willBegin?.method !== "Browser.downloadWillBegin" ||
    typeof willBegin?.params?.guid !== "string" ||
    willBegin.params.guid.length === 0
  ) {
    failures.push(`${label} Browser.downloadWillBegin evidence is missing`);
  }
  if (
    completed?.method !== "Browser.downloadProgress" ||
    completed?.params?.state !== "completed" ||
    completed?.params?.guid !== willBegin?.params?.guid
  ) {
    failures.push(`${label} matching completed Browser.downloadProgress evidence is missing`);
  }
  return failures;
}

function validateTurboOutputCore(output, label) {
  const failures = [];
  if (!output || typeof output !== "object") return [`${label} output-ready evidence is missing`];
  if (output.event !== "ready") failures.push(`${label} output-ready event is absent`);
  if (!isCanonicalU64DecimalString(output.job_id)) {
    failures.push(`${label} output-ready job ID is not a canonical u64 decimal string`);
  }
  if (output.model !== TURBO_MODEL_ID) failures.push(`${label} output-ready model is not Turbo`);
  if (output.width !== 1024 || output.height !== 1024) {
    failures.push(`${label} output-ready dimensions are not 1024x1024`);
  }
  if (output.backend !== LOW_VRAM_BACKEND) {
    failures.push(
      `${label} output-ready backend is not the exact preloaded packed-F16 dense-F32-per-stage policy`,
    );
  }
  if (output.artifacts_verified !== true) failures.push(`${label} output-ready artifacts are unverified`);
  if (output.artifact_content_digest !== TURBO_PRODUCTION_CONTENT_DIGEST) {
    failures.push(`${label} output-ready artifact digest is not canonical Turbo`);
  }
  if (output.numeric_format !== "f16-qwen-vision-f32") {
    failures.push(`${label} output-ready numeric format is not the production artifact identity`);
  }
  return failures;
}

function validateRequestDmdProgress(progressEvents, label) {
  const failures = [];
  if (!Array.isArray(progressEvents)) return [`${label} progress_events is not an array`];
  const terminalFailures = progressEvents.filter((event) =>
    ["run_failed", "run_cancelled"].includes(event?.event),
  );
  if (terminalFailures.length !== 0) {
    failures.push(`${label} emitted terminal failures: ${report(terminalFailures)}`);
  }
  const starts = progressEvents.filter(
    (event) => event?.event === "run_started" && event?.model === TURBO_MODEL_ID,
  );
  const completions = progressEvents.filter((event) => event?.event === "run_completed");
  const dmdStarts = progressEvents.filter(
    (event) => event?.event === "stage_started" && event?.stage === "dmd",
  );
  const dmdCompletions = progressEvents.filter(
    (event) => event?.event === "stage_completed" && event?.stage === "dmd",
  );
  const dmdSteps = progressEvents
    .filter((event) => event?.event === "step" && event?.stage === "dmd")
    .sort((left, right) => left.step - right.step);
  if (starts.length !== 1 || completions.length !== 1) {
    failures.push(`${label} does not contain exactly one started and completed run`);
  }
  if (dmdStarts.length !== 1 || dmdStarts[0]?.total_steps !== 4 || dmdCompletions.length !== 1) {
    failures.push(`${label} does not contain exactly one complete four-step DMD stage`);
  }
  if (dmdSteps.length !== 4) {
    failures.push(`${label} DMD step count=${dmdSteps.length}, expected 4`);
  }
  const runId = starts[0]?.run_id;
  for (let index = 0; index < dmdSteps.length; index += 1) {
    const step = dmdSteps[index];
    if (
      step.step !== index + 1 ||
      step.total_steps !== 4 ||
      JSON.stringify(step.run_id) !== JSON.stringify(runId)
    ) {
      failures.push(`${label} DMD step ${index + 1} has an invalid index, total, or run ID`);
    }
  }
  for (const event of [...completions, ...dmdStarts, ...dmdCompletions]) {
    if (event && JSON.stringify(event.run_id) !== JSON.stringify(runId)) {
      failures.push(`${label} progress event run ID differs from run_started`);
    }
  }
  return failures;
}

function validateZeroDmdCdpNetwork(cdp, label) {
  const failures = [];
  if (!cdp || typeof cdp !== "object") return [`${label} DMD CDP network evidence is missing`];
  if (cdp.policy !== TURBO_CDP_DMD_NETWORK_POLICY) {
    failures.push(`${label} DMD CDP policy=${report(cdp.policy)}, expected ${TURBO_CDP_DMD_NETWORK_POLICY}`);
  }
  if (!Array.isArray(cdp.model_base_urls) || cdp.model_base_urls.length !== 3) {
    failures.push(`${label} DMD CDP evidence does not bind three exact modular bases`);
  }
  if (
    !positiveFinite(cdp.window_start_epoch_ms) ||
    !positiveFinite(cdp.window_end_epoch_ms) ||
    cdp.window_end_epoch_ms <= cdp.window_start_epoch_ms ||
    cdp.terminal_event !== "stage_completed:dmd"
  ) {
    failures.push(`${label} DMD CDP evidence has an invalid event-anchored window`);
  }
  for (const field of [
    "model_response_count",
    "http_200_complete_part_response_count",
    "http_206_response_count",
    "complete_object_validated_response_count",
    "content_range_validated_response_count",
    "response_body_bytes",
    "unexpected_status_response_count",
    "missing_content_length_count",
    "invalid_content_range_response_count",
  ]) {
    if (cdp[field] !== 0) failures.push(`${label} DMD CDP ${field}=${report(cdp[field])}, expected 0`);
  }
  return failures;
}

function validateDmdRuntimeZeroIoAttestation(attestation, request, label) {
  const failures = [];
  if (!attestation || typeof attestation !== "object") {
    return [`${label} DMD runtime zero-I/O attestation is missing`];
  }
  if (attestation.policy !== TURBO_DMD_RUNTIME_ZERO_IO_POLICY) {
    failures.push(`${label} DMD runtime zero-I/O policy is incorrect`);
  }
  if (attestation.completed_dmd_steps !== 4) {
    failures.push(`${label} DMD runtime zero-I/O attestation is not bound to four completed steps`);
  }
  const runStarted = request?.progress_events?.find((event) => event?.event === "run_started");
  if (JSON.stringify(attestation.run_id) !== JSON.stringify(runStarted?.run_id)) {
    failures.push(`${label} DMD runtime zero-I/O attestation run ID differs from run_started`);
  }
  if (
    attestation.runtime_source_sha256 !==
    request?.tested_package_identity?.sources?.browser_runtime?.sha256
  ) {
    failures.push(`${label} DMD runtime zero-I/O attestation is not bound to the exact tested runtime source`);
  }
  const lifecycleEvents = (request?.runtime_events ?? []).filter(
    (event) => event?.event === "packed_f16_denoiser_lifecycle",
  );
  if (lifecycleEvents.length !== 1) {
    failures.push(`${label} DMD runtime attestation is not bound to one packed-F16 lifecycle event`);
  } else {
    const lifecycleEvent = lifecycleEvents[0];
    if (attestation.lifecycle_event_at_ms !== lifecycleEvent.at_ms) {
      failures.push(`${label} DMD runtime attestation timestamp differs from its lifecycle event`);
    }
    if (
      JSON.stringify(attestation.traffic) !==
      JSON.stringify(lifecycleEvent.lifecycle?.dmd_artifact_traffic)
    ) {
      failures.push(`${label} DMD runtime attestation traffic differs from its lifecycle event`);
    }
  }
  if (JSON.stringify(attestation.traffic) !== JSON.stringify(TURBO_DMD_ZERO_IO)) {
    failures.push(`${label} DMD runtime artifact/cache/network traffic is not exact zero`);
  }
  return failures;
}

function validateRequestGpuWindow(gpu, requestEpochWindow, label) {
  const failures = [];
  if (!gpu || typeof gpu !== "object") return [`${label} GPU request-window evidence is missing`];
  if (gpu.provider !== "nvidia-smi") failures.push(`${label} GPU request-window provider is not nvidia-smi`);
  if (gpu.interval_aggregation_policy !== GPU_INTERVAL_AGGREGATION_POLICY) {
    failures.push(`${label} GPU request-window aggregation policy is incorrect`);
  }
  if (gpu.maximum_framebuffer_bytes_exclusive !== LOW_VRAM_DEVICE_CAP_BYTES) {
    failures.push(`${label} GPU request-window cap is not decimal 32 GB`);
  }
  if (
    gpu.window_start_epoch_ms !== requestEpochWindow?.start_epoch_ms ||
    gpu.window_end_epoch_ms !== requestEpochWindow?.end_epoch_ms ||
    !positiveFinite(gpu.window_start_epoch_ms) ||
    !positiveFinite(gpu.window_end_epoch_ms) ||
    gpu.window_end_epoch_ms <= gpu.window_start_epoch_ms
  ) {
    failures.push(`${label} GPU request window is invalid or not exactly bound to the request`);
  }
  if (!Array.isArray(gpu.sample_records) || gpu.sample_records.length === 0) {
    failures.push(`${label} GPU request window contains no native samples`);
  } else if (
    gpu.sample_records.some(
      (sample) =>
        !positiveFinite(sample?.at_ms) ||
        sample.at_ms < gpu.window_start_epoch_ms ||
        sample.at_ms > gpu.window_end_epoch_ms,
    )
  ) {
    failures.push(`${label} GPU sample escaped its exact request window`);
  }
  if (!positiveInteger(gpu.matched_sample_intervals)) {
    failures.push(`${label} GPU request window matched no Chrome GPU-process interval`);
  }
  if (!positiveInteger(gpu.active_sample_intervals)) {
    failures.push(`${label} GPU request window observed no Chrome compute activity`);
  }
  if (
    !positiveInteger(gpu.observed_peak_aggregate_framebuffer_bytes) ||
    gpu.observed_peak_aggregate_framebuffer_bytes >= LOW_VRAM_DEVICE_CAP_BYTES
  ) {
    failures.push(`${label} GPU request-window framebuffer peak is absent or not below 32 GB`);
  }
  if (!Array.isArray(gpu.observed_gpu_processes) || gpu.observed_gpu_processes.length === 0) {
    failures.push(`${label} GPU request window has no Chrome GPU-process inventory`);
  }
  if (gpu.validated !== true || !Array.isArray(gpu.validation_failures)) {
    failures.push(`${label} GPU request window was not independently validated`);
  } else if (gpu.validation_failures.length !== 0) {
    failures.push(`${label} GPU request-window validation failed: ${report(gpu.validation_failures)}`);
  }
  return failures;
}

function validateSecondRequestInteraction(interaction, fixedPrompt, request, previousRequest) {
  const failures = [];
  if (interaction?.mechanism !== "cdp-keyboard-and-mouse") {
    failures.push("second request interaction is not attributed to CDP keyboard and mouse input");
  }
  if (interaction?.prompt_reused_from_same_engine !== true || interaction?.prompt_value !== fixedPrompt) {
    failures.push("second request did not attest preserving the exact prompt in the same engine");
  }
  if (interaction?.seed_typed_via_cdp !== true || interaction?.seed_value !== "1") {
    failures.push("second request did not type exact seed 1 through CDP keyboard input");
  }
  if (interaction?.seed_event?.event !== "seed_changed" || interaction?.seed_event?.value !== "1") {
    failures.push("second request does not contain exact seed_changed evidence for seed 1");
  }
  const readiness = interaction?.run_readiness;
  const boundary = request?.event_boundaries;
  const uiStartIndex = boundary?.ui_start_index;
  const uiEndIndex = boundary?.ui_end_index;
  if (
    readiness?.policy !== TURBO_SECOND_REQUEST_RUN_READY_POLICY ||
    ![
      "exact-last-pre-boundary-post-request-ready",
      "second-request-ui-partition",
    ].includes(readiness?.source) ||
    readiness?.duplicate_ready_after_seed_change_required !== false ||
    readiness?.fallback_exact_last_pre_boundary_ready !== true
  ) {
    failures.push("second request Run readiness provenance is missing or inexact");
  }
  if (
    !nonNegativeInteger(uiStartIndex) ||
    !nonNegativeInteger(uiEndIndex) ||
    uiEndIndex < uiStartIndex ||
    readiness?.ui_start_index !== uiStartIndex ||
    !nonNegativeInteger(readiness?.ui_event_count_at_resolution) ||
    readiness.ui_event_count_at_resolution <= uiStartIndex ||
    readiness.ui_event_count_at_resolution > uiEndIndex
  ) {
    failures.push("second request Run readiness is not bound to its exact UI event partition");
  }
  const seedChangedIndex = readiness?.seed_changed_event_index;
  const runReadyIndex = readiness?.run_ready_event_index;
  if (
    !nonNegativeInteger(seedChangedIndex) ||
    seedChangedIndex < uiStartIndex ||
    seedChangedIndex >= readiness?.ui_event_count_at_resolution ||
    !sameJsonValue(
      request?.ui_events?.[seedChangedIndex - uiStartIndex],
      interaction?.seed_event,
    )
  ) {
    failures.push("second request exact seed_changed evidence escaped its UI event partition");
  }
  const selectedUiContract = readiness?.selected_ui_contract;
  failures.push(
    ...validateTurboRunReadyUiContract(
      selectedUiContract,
      "second request selected Run readiness",
    ),
  );
  if (readiness?.source === "second-request-ui-partition") {
    const localReadyEvents = (request?.ui_events ?? [])
      .slice(0, readiness.ui_event_count_at_resolution - uiStartIndex)
      .map((event, offset) => ({ event, index: uiStartIndex + offset }))
      .filter(({ event }) => event?.event === "ready");
    const latestLocalReady = localReadyEvents.at(-1);
    if (
      !latestLocalReady ||
      runReadyIndex !== latestLocalReady.index ||
      !sameJsonValue(selectedUiContract, latestLocalReady.event)
    ) {
      failures.push("second request did not select its latest partition-local ready contract");
    }
  } else if (readiness?.source === "exact-last-pre-boundary-post-request-ready") {
    const localReadyBeforeResolution = (request?.ui_events ?? [])
      .slice(0, readiness.ui_event_count_at_resolution - uiStartIndex)
      .some((event) => event?.event === "ready");
    const previousBoundary = previousRequest?.event_boundaries;
    const previousUiStart = previousBoundary?.ui_start_index;
    const previousUiEnd = previousBoundary?.ui_end_index;
    const previousReadyEvents = (previousRequest?.ui_events ?? [])
      .map((event, offset) => ({ event, index: previousUiStart + offset }))
      .filter(({ event }) => event?.event === "ready");
    const lastPreBoundaryReady = previousReadyEvents.at(-1);
    if (
      localReadyBeforeResolution ||
      previousUiEnd !== uiStartIndex ||
      !lastPreBoundaryReady ||
      runReadyIndex !== lastPreBoundaryReady.index ||
      runReadyIndex >= uiStartIndex ||
      !sameJsonValue(selectedUiContract, lastPreBoundaryReady.event)
    ) {
      failures.push(
        "second request did not reuse the exact last pre-boundary post-request ready contract",
      );
    }
  }
  const input = interaction?.seed_input;
  if (
    input?.focus_event?.seed_focused !== true ||
    input?.replacement_mode !== "bevy-editable-text-triple-click-select-all" ||
    input?.focus_click?.click_count !== 1 ||
    input?.selection_clicks?.length !== 2 ||
    input?.selection_clicks?.[0]?.click_count !== 2 ||
    input?.selection_clicks?.[1]?.click_count !== 3
  ) {
    failures.push("second request Seed value did not use the real CDP triple-click selection contract");
  }
  if (interaction?.run_clicked_via_cdp !== true) {
    failures.push("second request did not click ordinary Run through CDP");
  }
  if (interaction?.save_clicked_via_cdp !== true) {
    failures.push("second request did not click ordinary Save PNG through CDP");
  }
  return failures;
}

function validateTurboQ4OutputCore(output, label) {
  const failures = [];
  if (!output || typeof output !== "object") return [`${label} output-ready evidence is missing`];
  if (output.event !== "ready") failures.push(`${label} output-ready event is absent`);
  if (!isCanonicalU64DecimalString(output.job_id)) {
    failures.push(`${label} output-ready job ID is not a canonical u64 decimal string`);
  }
  if (output.model !== TURBO_MODEL_ID) failures.push(`${label} output-ready model is not Turbo`);
  if (output.width !== 1024 || output.height !== 1024) {
    failures.push(`${label} output-ready dimensions are not 1024x1024`);
  }
  if (output.backend !== TURBO_Q4_RESIDENT_BACKEND) {
    failures.push(`${label} output-ready backend is not the exact resident-Q4 surface-gated policy`);
  }
  if (output.artifacts_verified !== true) failures.push(`${label} output-ready artifacts are unverified`);
  if (output.artifact_content_digest !== TURBO_Q4_PRODUCTION_CONTENT_DIGEST) {
    failures.push(`${label} output-ready artifact digest is not canonical resident Q4`);
  }
  if (output.numeric_format !== "q4s-block-up-to128-f32") {
    failures.push(`${label} output-ready numeric format is not the resident-Q4 identity`);
  }
  return failures;
}

function validateResidentQ4ZeroCdpNetwork(cdp, label) {
  const failures = [];
  if (!cdp || typeof cdp !== "object") return [`${label} CDP network evidence is missing`];
  if (!Array.isArray(cdp.model_base_urls) || cdp.model_base_urls.length !== 1) {
    failures.push(`${label} CDP evidence does not bind one exact Q4 bundle base`);
  }
  for (const field of [
    "model_response_count",
    "http_200_complete_part_response_count",
    "http_206_response_count",
    "complete_object_validated_response_count",
    "content_range_validated_response_count",
    "response_body_bytes",
    "unexpected_status_response_count",
    "missing_content_length_count",
    "invalid_content_range_response_count",
  ]) {
    if (cdp[field] !== 0) {
      failures.push(`${label} CDP ${field}=${report(cdp[field])}, expected 0`);
    }
  }
  return failures;
}

function validateResidentQ4CacheAudits(request, label) {
  const failures = [];
  const audits = (request?.runtime_events ?? []).filter(
    (event) => event?.event === "resident_cache_audit",
  );
  if (audits.length !== 2 || audits[0]?.boundary !== "before-request" || audits[1]?.boundary !== "after-request") {
    return [`${label} does not contain the exact before/after resident-cache audit pair`];
  }
  const runStarted = request?.progress_events?.find((event) => event?.event === "run_started");
  for (const [index, audit] of audits.entries()) {
    if (JSON.stringify(audit.run_id) !== JSON.stringify(runStarted?.run_id)) {
      failures.push(`${label} cache audit ${index + 1} run ID differs from run_started`);
    }
    for (const [cached, expected, exactExpected] of [
      ["qwen_cached_stages", "qwen_expected_stages", 43],
      ["vae_cached_stages", "vae_expected_stages", 1],
      ["denoiser_cached_stages", "denoiser_expected_stages", 46],
    ]) {
      if (audit?.[cached] !== exactExpected || audit?.[expected] !== exactExpected) {
        failures.push(
          `${label} cache audit ${index + 1} ${cached}/${expected}=${report(audit?.[cached])}/${report(audit?.[expected])}, expected ${exactExpected}/${exactExpected}`,
        );
      }
    }
    if (
      audit?.qwen_synchronization_pending !== false ||
      audit?.denoiser_synchronization_pending !== false ||
      audit?.resident_weights_preserved !== true
    ) {
      failures.push(`${label} cache audit ${index + 1} does not attest synchronized resident weights`);
    }
  }
  return failures;
}

/** Validate two real rendered Bevy requests with one warm resident-Q4 WebGPU engine. */
export function validateTurboQ4ResidentMultiRequestEvidence(evidence) {
  const failures = [];
  if (!evidence || typeof evidence !== "object") return ["resident-Q4 multi-request evidence is missing"];
  if (evidence.policy !== TURBO_Q4_RESIDENT_MULTI_REQUEST_POLICY) {
    failures.push("resident-Q4 multi-request policy is incorrect");
  }
  if (evidence.request_count !== 2 || !Array.isArray(evidence.requests) || evidence.requests.length !== 2) {
    return [...failures, "resident-Q4 qualification does not contain exactly two requests"];
  }
  if (!/^[0-9a-f-]{16,}$/i.test(String(evidence.engine_session_id ?? ""))) {
    failures.push("resident-Q4 same-engine page session identity is missing");
  }
  failures.push(
    ...validateRenderedRuntimeWebGpuEvidence(
      evidence.bevy_backend_ready,
      evidence.runtime_webgpu_calls,
      evidence.engine_session_id,
      evidence.runtime_webgpu_dropped_calls,
      evidence.runtime_webgpu_adapter_attestation,
      BOOGU_WEB_REQUIRED_DEVICE_FEATURES,
    ).map((failure) => `same-engine WebGPU attestation: ${failure}`),
  );

  const initial = Array.isArray(evidence.initial_runtime_events)
    ? evidence.initial_runtime_events
    : [];
  const plans = initial.filter((event) => event?.event === "resident_resource_plan");
  const ready = initial.filter(
    (event) => event?.event === "ready" && event?.model === TURBO_MODEL_ID,
  );
  const preflight = initial.filter((event) => event?.event === "vram_preflight");
  if (
    plans.length !== 1 ||
    plans[0]?.weight_storage_policy !==
      "packed-q4s-block-up-to-128/f32-scales/packed-f16-convolutions/f32-auxiliaries" ||
    !positiveInteger(plans[0]?.resident_weight_bytes) ||
    plans[0]?.strict_device_cap_bytes !== TURBO_Q4_STRICT_DEVICE_BYTES_EXCLUSIVE ||
    !positiveInteger(plans[0]?.conservative_planned_device_bytes) ||
    plans[0].conservative_planned_device_bytes >= TURBO_Q4_STRICT_DEVICE_BYTES_EXCLUSIVE
  ) {
    failures.push("initial runtime partition does not contain one valid sub-16GB resident-Q4 resource plan");
  }
  if (
    ready.length !== 1 ||
    ready[0]?.request_enabled !== true ||
    ready[0]?.selected_model_cache_complete !== true ||
    ready[0]?.selected_model_device_resident !== true
  ) {
    failures.push("initial runtime partition does not attest a complete device-resident Q4 model");
  }
  if (
    preflight.length !== 2 ||
    preflight[0]?.status !== "started" ||
    preflight[1]?.status !== "passed" ||
    preflight[1]?.allocations_committed !== true ||
    preflight[1]?.shared_device_and_queue !== true ||
    preflight[1]?.required_device_bytes !== plans[0]?.conservative_planned_device_bytes
  ) {
    failures.push("resident-Q4 VRAM preflight did not commit the exact conservative plan before download");
  }

  const aggregateGpu = evidence.native_gpu_attestation;
  if (
    aggregateGpu?.provider !== "nvidia-smi" ||
    aggregateGpu?.validated !== true ||
    !positiveInteger(aggregateGpu?.observed_peak_aggregate_framebuffer_bytes) ||
    aggregateGpu.observed_peak_aggregate_framebuffer_bytes >= TURBO_Q4_STRICT_DEVICE_BYTES_EXCLUSIVE
  ) {
    failures.push("aggregate resident-Q4 GPU evidence is absent, invalid, or not below 16GB");
  }

  const allAudits = [];
  for (const [index, request] of evidence.requests.entries()) {
    const label = `request ${index + 1}`;
    if (request?.request_ordinal !== index + 1) failures.push(`${label} ordinal is incorrect`);
    if (
      request?.page_identity?.engine_session_id !== evidence.engine_session_id ||
      request?.page_identity?.url !== evidence.page_url ||
      request?.page_identity?.time_origin_epoch_ms !== evidence.time_origin_epoch_ms
    ) {
      failures.push(`${label} is not bound to the same page/runtime identity`);
    }
    failures.push(...validateRequestScopedSurfaceGate(request, label));
    failures.push(...validateRequestDmdProgress(request?.progress_events, label));
    failures.push(...validateTurboQ4OutputCore(request?.output_ready, label));
    failures.push(...validateProductionPngDownload(request?.downloaded_png, label));
    failures.push(...validateRequestGpuWindow(request?.native_gpu_attestation, request?.request_epoch_window, label));
    if (
      request?.native_gpu_attestation?.observed_peak_aggregate_framebuffer_bytes >=
      TURBO_Q4_STRICT_DEVICE_BYTES_EXCLUSIVE
    ) {
      failures.push(`${label} GPU request-window peak is not below 16GB`);
    }
    failures.push(...validateExactArtifactTraffic(request?.artifact_traffic, TURBO_DMD_ZERO_IO, `${label} resident request`));
    failures.push(...validateResidentQ4ZeroCdpNetwork(request?.cdp_network_traffic, `${label} request`));
    failures.push(...validateResidentQ4ZeroCdpNetwork(request?.cdp_dmd_network_traffic, `${label} DMD`));
    failures.push(...validateResidentQ4CacheAudits(request, label));
    allAudits.push(
      ...(request?.runtime_events ?? []).filter(
        (event) => event?.event === "resident_cache_audit",
      ),
    );
    if ((request?.runtime_events ?? []).some((event) => event?.event === "packed_f16_denoiser_preload")) {
      failures.push(`${label} unexpectedly used request-scoped packed-F16 rehydration`);
    }
    if (!positiveInteger(request?.canvas_png_changed_bytes)) {
      failures.push(`${label} did not materially change the rendered canvas`);
    }
  }
  if (
    allAudits.length !== 4 ||
    allAudits.some(
      (audit) =>
        audit.qwen_cached_stages !== 43 ||
        audit.vae_cached_stages !== 1 ||
        audit.denoiser_cached_stages !== 46,
    )
  ) {
    failures.push("resident-Q4 cache cardinality changed across the two requests");
  }
  const [first, second] = evidence.requests;
  failures.push(...validateSecondRequestInteraction(second?.interaction, evidence.fixed_ascii_prompt, second, first));
  if (
    first?.output_ready?.job_id === second?.output_ready?.job_id ||
    first?.downloaded_png?.sha256 === second?.downloaded_png?.sha256
  ) {
    failures.push("resident-Q4 requests did not produce distinct jobs and PNG outputs");
  }
  const transport = first?.modular_artifact_transport;
  const transportDigests = new Set(
    (transport?.bundles ?? []).map((bundle) => bundle?.content_digest),
  );
  if (
    transport?.validated !== true ||
    transport?.bundles?.length !== 3 ||
    !transportDigests.has(TURBO_Q4_PRODUCTION_CONTENT_DIGEST) ||
    !transportDigests.has(TURBO_Q4_QWEN_COMPONENT_CONTENT_DIGEST) ||
    !transportDigests.has(TURBO_Q4_VAE_COMPONENT_CONTENT_DIGEST) ||
    transport.maximum_physical_transport_part_bytes > ARTIFACT_TRANSPORT_HARD_MAX_PART_BYTES
  ) {
    failures.push("resident-Q4 request is not bound to the exact three-bundle modular transport closure");
  }
  return failures;
}

export function validateTurbo1024MultiRequestEvidence(evidence) {
  const failures = [];
  const block0ExecutionMode = evidence?.initial_runtime_events?.find(
    (event) => event?.event === "ready" && event?.model === TURBO_MODEL_ID,
  )?.block0_execution_mode;
  if (evidence?.qwen_block0_execution_mode !== block0ExecutionMode) {
    failures.push("multi-request Qwen block-0 execution mode differs from runtime ready evidence");
  }
  if (!evidence || typeof evidence !== "object") {
    return ["Turbo multi-request evidence is missing"];
  }
  if (evidence.policy !== TURBO_MULTI_REQUEST_POLICY) {
    failures.push(`multi-request policy=${report(evidence.policy)}, expected exact policy`);
  }
  if (evidence.request_count !== 2 || !Array.isArray(evidence.requests) || evidence.requests.length !== 2) {
    failures.push("multi-request qualification does not contain exactly two requests");
    return failures;
  }
  if (!/^[0-9a-f-]{16,}$/i.test(String(evidence.engine_session_id ?? ""))) {
    failures.push("same-engine page session identity is missing");
  }
  if (!positiveFinite(evidence.time_origin_epoch_ms) || typeof evidence.page_url !== "string") {
    failures.push("same-page URL/time-origin identity is missing");
  }
  failures.push(
    ...validateRenderedRuntimeWebGpuEvidence(
      evidence?.bevy_backend_ready,
      evidence?.runtime_webgpu_calls,
      evidence?.engine_session_id,
      evidence?.runtime_webgpu_dropped_calls,
      evidence?.runtime_webgpu_adapter_attestation,
      BOOGU_WEB_REQUIRED_DEVICE_FEATURES,
    ).map((failure) => `same-engine WebGPU attestation: ${failure}`),
  );
  const aggregateGpu = evidence.native_gpu_attestation;
  if (
    aggregateGpu?.provider !== "nvidia-smi" ||
    aggregateGpu?.interval_aggregation_policy !== GPU_INTERVAL_AGGREGATION_POLICY ||
    aggregateGpu?.maximum_framebuffer_bytes_exclusive !== LOW_VRAM_DEVICE_CAP_BYTES ||
    !positiveInteger(aggregateGpu?.observed_peak_aggregate_framebuffer_bytes) ||
    aggregateGpu.observed_peak_aggregate_framebuffer_bytes >= LOW_VRAM_DEVICE_CAP_BYTES ||
    aggregateGpu?.validated !== true ||
    !Array.isArray(aggregateGpu?.validation_failures) ||
    aggregateGpu.validation_failures.length !== 0
  ) {
    failures.push("aggregate two-request native GPU attestation is absent, invalid, or not below 32 GB");
  }
  if (!Array.isArray(evidence.initial_runtime_events)) {
    failures.push("initial runtime event partition is missing");
  } else {
    const preload = evidence.initial_runtime_events.filter(
      (event) => event?.event === "packed_f16_denoiser_preload",
    );
    const plans = evidence.initial_runtime_events.filter(
      (event) => event?.event === "packed_f16_resource_plan",
    );
    if (
      preload.length !== 1 ||
      preload[0]?.cached_stages !== TURBO_PACKED_F16_CACHED_STAGES ||
      preload[0]?.cached_objects !== TURBO_PACKED_F16_CACHED_OBJECTS ||
      preload[0]?.cached_tensors !== TURBO_PACKED_F16_CACHED_TENSORS ||
      preload[0]?.cached_bytes !==
        TURBO_PACKED_F16_RESOURCE_PLAN.retained_packed_f16_denoiser_bytes ||
      preload[0]?.previous_preload_attempt_count !== 0 ||
      preload[0]?.preload_attempt_count !== 1 ||
      preload[0]?.request_scoped_rehydration !== false ||
      preload[0]?.rehydration_policy !== TURBO_PACKED_F16_NEXT_REQUEST_REHYDRATION_POLICY
    ) {
      failures.push("initial runtime partition does not attest one complete packed-F16 preload");
    }
    if (plans.length !== 1) {
      failures.push("initial runtime partition does not contain exactly one packed-F16 resource plan");
    }
  }

  const [first, second] = evidence.requests;
  const firstRequestFailures = validateTurbo1024ModelEvidence({
    ...first,
    fixed_ascii_prompt: evidence.fixed_ascii_prompt,
    runtime_events: [...(evidence.initial_runtime_events ?? []), ...(first?.runtime_events ?? [])],
    packed_f16_denoiser_preload_traffic:
      evidence.initial_packed_f16_denoiser_preload_traffic,
    cdp_preload_network_traffic: evidence.cdp_preload_network_traffic,
    bevy_backend_ready: evidence.bevy_backend_ready,
    runtime_webgpu_calls: evidence.runtime_webgpu_calls,
    engine_session_id: evidence.engine_session_id,
    runtime_webgpu_dropped_calls: evidence.runtime_webgpu_dropped_calls,
    runtime_webgpu_adapter_attestation: evidence.runtime_webgpu_adapter_attestation,
  });
  failures.push(...firstRequestFailures.map((failure) => `request 1: ${failure}`));
  for (const [index, request] of evidence.requests.entries()) {
    const label = `request ${index + 1}`;
    if (request?.request_ordinal !== index + 1) failures.push(`${label} ordinal is incorrect`);
    if (request?.qwen_block0_execution_mode !== block0ExecutionMode) {
      failures.push(`${label} Qwen block-0 execution mode differs from the session policy`);
    }
    if (
      request?.page_identity?.engine_session_id !== evidence.engine_session_id ||
      request?.page_identity?.url !== evidence.page_url ||
      request?.page_identity?.time_origin_epoch_ms !== evidence.time_origin_epoch_ms
    ) {
      failures.push(`${label} is not bound to the same page/runtime identity`);
    }
    const boundary = request?.event_boundaries;
    for (const kind of ["runtime", "progress", "output", "ui", "cdp", "surface"]) {
      const start = boundary?.[`${kind}_start_index`];
      const end = boundary?.[`${kind}_end_index`];
      if (!nonNegativeInteger(start) || !nonNegativeInteger(end) || end < start) {
        failures.push(`${label} ${kind} event boundary is invalid`);
      }
    }
    if (
      nonNegativeInteger(boundary?.cdp_start_index) &&
      nonNegativeInteger(boundary?.cdp_end_index) &&
      request?.cdp_event_count !== boundary.cdp_end_index - boundary.cdp_start_index
    ) {
      failures.push(`${label} CDP event count does not match its exact event partition`);
    }
    if (!Array.isArray(request?.runtime_events)) failures.push(`${label} runtime events are missing`);
    if (!Array.isArray(request?.progress_events)) failures.push(`${label} progress events are missing`);
    if (!Array.isArray(request?.output_events)) failures.push(`${label} output events are missing`);
    if (!Array.isArray(request?.ui_events)) failures.push(`${label} UI events are missing`);
    for (const kind of ["runtime", "progress", "output", "ui"]) {
      const events = request?.[`${kind}_events`];
      if (
        Array.isArray(events) &&
        nonNegativeInteger(boundary?.[`${kind}_start_index`]) &&
        nonNegativeInteger(boundary?.[`${kind}_end_index`]) &&
        events.length !==
          boundary[`${kind}_end_index`] - boundary[`${kind}_start_index`]
      ) {
        failures.push(`${label} ${kind} events do not exactly cover their indexed partition`);
      }
    }
    if (!positiveInteger(request?.cdp_event_count)) {
      failures.push(`${label} CDP event partition is empty`);
    }
    if (
      Array.isArray(request?.surface_texture_gate_windows) &&
      nonNegativeInteger(boundary?.surface_start_index) &&
      nonNegativeInteger(boundary?.surface_end_index) &&
      request.surface_texture_gate_windows.length !==
        boundary.surface_end_index - boundary.surface_start_index
    ) {
      failures.push(`${label} surface gate windows do not exactly cover their indexed partition`);
    }
    failures.push(...validateRequestScopedSurfaceGate(request, label));
    failures.push(...validateRequestDmdProgress(request?.progress_events, label));
    failures.push(...validateTurboOutputCore(request?.output_ready, label));
    const requestRunStarted = request?.progress_events?.find(
      (event) => event?.event === "run_started" && event?.model === TURBO_MODEL_ID,
    );
    if (
      !outputJobIdMatchesNumericRunId(
        request?.output_ready?.job_id,
        requestRunStarted?.run_id,
      )
    ) {
      failures.push(`${label} output job ID differs from the exact decimal representation of its run ID`);
    }
    failures.push(...validateProductionPngDownload(request?.downloaded_png, label));
    failures.push(...validateZeroDmdCdpNetwork(request?.cdp_dmd_network_traffic, label));
    failures.push(...validateDmdRuntimeZeroIoAttestation(request?.dmd_runtime_io_attestation, request, label));
    failures.push(...validateRequestGpuWindow(request?.native_gpu_attestation, request?.request_epoch_window, label));
    failures.push(
      ...validateTestedPackageIdentity(
        request?.tested_package_identity,
        request?.served_transport,
      ).map((failure) => `${label}: ${failure}`),
    );
    if (!positiveInteger(request?.canvas_png_changed_bytes)) {
      failures.push(`${label} did not materially change the rendered canvas`);
    }
    if (!positiveInteger(request?.model_screenshot_bytes) || request.model_screenshot_bytes < 1024) {
      failures.push(`${label} rendered screenshot is missing`);
    }
    const requestPreparingEvents = (request?.runtime_events ?? []).filter(
      (event) => event?.event === "preparing",
    );
    const expectedRequestPreparingEvents = index === 0 ? 0 : 1;
    if (requestPreparingEvents.length !== expectedRequestPreparingEvents) {
      failures.push(
        `${label} emitted ${requestPreparingEvents.length} request-local packed-F16 preparing event(s), expected ${expectedRequestPreparingEvents}`,
      );
    } else if (
      index === 1 &&
      requestPreparingEvents[0]?.message !== TURBO_PACKED_F16_PRELOAD_MESSAGE
    ) {
      failures.push(
        `${label} packed-F16 preparing message=${report(requestPreparingEvents[0]?.message)}, expected ${report(TURBO_PACKED_F16_PRELOAD_MESSAGE)}`,
      );
    }
    const requestPreloads = (request?.runtime_events ?? []).filter(
      (event) => event?.event === "packed_f16_denoiser_preload",
    );
    const expectedRequestPreloads = index === 0 ? 0 : 1;
    if (requestPreloads.length !== expectedRequestPreloads) {
      failures.push(
        `${label} emitted ${requestPreloads.length} request-local packed-F16 preload event(s), expected ${expectedRequestPreloads}`,
      );
    } else if (index === 1) {
      const preload = requestPreloads[0];
      for (const [field, expected] of [
        ["cached_stages", TURBO_PACKED_F16_CACHED_STAGES],
        ["cached_objects", TURBO_PACKED_F16_CACHED_OBJECTS],
        ["cached_tensors", TURBO_PACKED_F16_CACHED_TENSORS],
        ["cached_bytes", TURBO_PACKED_F16_RESOURCE_PLAN.retained_packed_f16_denoiser_bytes],
        ["previous_preload_attempt_count", 1],
        ["preload_attempt_count", 2],
        ["request_scoped_rehydration", true],
        ["rehydration_policy", TURBO_PACKED_F16_NEXT_REQUEST_REHYDRATION_POLICY],
      ]) {
        if (!Object.is(preload?.[field], expected)) {
          failures.push(`${label} rehydration ${field}=${report(preload?.[field])}, expected ${report(expected)}`);
        }
      }
      failures.push(
        ...validateExactArtifactTraffic(
          preload?.traffic,
          TURBO_DENOISER_REHYDRATION_TRAFFIC,
          `${label} packed-F16 rehydration`,
        ),
      );
      const preparingIndex = request.runtime_events.indexOf(requestPreparingEvents[0]);
      const preloadIndex = request.runtime_events.indexOf(preload);
      if (preparingIndex < 0 || preparingIndex >= preloadIndex) {
        failures.push(
          `${label} packed-F16 preparing event is not ordered before its packed-F16 preload event`,
        );
      }
    }
    const unexpectedRuntimeEvents = (request?.runtime_events ?? []).filter(
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
          "packed_f16_denoiser_preload",
          "preparing",
          "artifact_traffic",
          SURFACE_INFERENCE_SUSPENDED_EVENT,
          SURFACE_INFERENCE_RESUMED_EVENT,
        ].includes(event?.event),
    );
    if (unexpectedRuntimeEvents.length !== 0) {
      failures.push(`${label} emitted request-local initialization/runtime events: ${report(unexpectedRuntimeEvents)}`);
    }
    const trafficEvents = (request?.runtime_events ?? []).filter(
      (event) => event?.event === "artifact_traffic",
    );
    if (trafficEvents.length !== 1) {
      failures.push(`${label} does not contain exactly one artifact_traffic event`);
    } else if (JSON.stringify(request?.artifact_traffic) !== JSON.stringify(trafficEvents[0].traffic)) {
      failures.push(`${label} top-level artifact traffic differs from its recorded runtime event`);
    }
    const lifecycleEvents = (request?.runtime_events ?? []).filter(
      (event) => event?.event === "packed_f16_denoiser_lifecycle",
    );
    const dmdVaeHandoffEvents = (request?.runtime_events ?? []).filter(
      (event) => event?.event === "packed_f16_dmd_vae_handoff",
    );
    const qwenHostEmbeddingEvents = (request?.runtime_events ?? []).filter(
      (event) => event?.event === "packed_f16_qwen_host_embedding",
    );
    if (qwenHostEmbeddingEvents.length !== 1) {
      failures.push(`${label} does not contain exactly one Qwen host-embedding event`);
    } else {
      failures.push(
        ...validatePackedF16QwenHostEmbedding(
          qwenHostEmbeddingEvents[0],
          request?.progress_events,
          label,
        ),
      );
      if (
        JSON.stringify(request?.packed_f16_qwen_host_embedding) !==
        JSON.stringify(qwenHostEmbeddingEvents[0])
      ) {
        failures.push(`${label} top-level Qwen host-embedding evidence differs from runtime event`);
      }
    }
    const qwenPreHandoffEvents = (request?.runtime_events ?? []).filter(
      (event) => event?.event === "packed_f16_qwen_pre_handoff_diagnostics",
    );
    const qwenBlock0ExecutionEvents = (request?.runtime_events ?? []).filter(
      (event) => event?.event === "packed_f16_qwen_block0_execution_diagnostics",
    );
    const expectedBlock0ExecutionEventCount =
      block0ExecutionMode === TURBO_QWEN_BLOCK0_SERIALIZED_DIAGNOSTIC_MODE ? 1 : 0;
    if (qwenBlock0ExecutionEvents.length !== expectedBlock0ExecutionEventCount) {
      failures.push(
        `${label} contains ${qwenBlock0ExecutionEvents.length} Qwen block-0 execution event(s), expected ${expectedBlock0ExecutionEventCount} for mode ${report(block0ExecutionMode)}`,
      );
    } else if (expectedBlock0ExecutionEventCount === 1) {
      failures.push(
        ...validatePackedF16QwenBlock0ExecutionDiagnostics(
          qwenBlock0ExecutionEvents[0],
          request?.progress_events,
          label,
        ),
      );
      if (
        JSON.stringify(request?.packed_f16_qwen_block0_execution_diagnostics) !==
        JSON.stringify(qwenBlock0ExecutionEvents[0])
      ) {
        failures.push(`${label} top-level Qwen block-0 execution evidence differs from runtime event`);
      }
    } else if (request?.packed_f16_qwen_block0_execution_diagnostics != null) {
      failures.push(`${label} ordinary block-0 execution exposes serialized boundary evidence`);
    }
    const qwenBlock0PostSyncEvents = (request?.runtime_events ?? []).filter(
      (event) => event?.event === "packed_f16_qwen_block0_post_sync_diagnostic",
    );
    if (qwenBlock0PostSyncEvents.length !== 1) {
      failures.push(`${label} does not contain exactly one Qwen block-0 post-sync event`);
    } else {
      failures.push(
        ...validatePackedF16QwenBlock0PostSyncDiagnostic(
          qwenBlock0PostSyncEvents[0],
          request?.progress_events,
          label,
        ),
      );
      if (
        JSON.stringify(request?.packed_f16_qwen_block0_post_sync_diagnostic) !==
        JSON.stringify(qwenBlock0PostSyncEvents[0])
      ) {
        failures.push(`${label} top-level Qwen block-0 post-sync evidence differs from runtime event`);
      }
    }
    const qwenPostHandoffEvents = (request?.runtime_events ?? []).filter(
      (event) => event?.event === "packed_f16_qwen_post_handoff_diagnostics",
    );
    if (
      qwenBlock0PostSyncEvents.length === 1 &&
      qwenBlock0PostSyncEvents[0].diagnostic?.block0_execution_mode !== block0ExecutionMode
    ) {
      failures.push(`${label} block-0 post-sync mode differs from runtime ready policy`);
    }
    if (
      qwenBlock0ExecutionEvents.length === 1 &&
      qwenBlock0PostSyncEvents.length === 1 &&
      (qwenBlock0ExecutionEvents[0].at_ms >= qwenBlock0PostSyncEvents[0].at_ms ||
        qwenBlock0ExecutionEvents[0].diagnostics?.boundaries?.at(-1)?.tensor?.sha256 !==
          qwenBlock0PostSyncEvents[0].diagnostic?.tensor?.sha256 ||
        qwenBlock0ExecutionEvents[0].diagnostics?.boundaries?.[0]?.tensor?.sha256 !==
          qwenHostEmbeddingEvents[0]?.report?.device_f32_sha256)
    ) {
      failures.push(
        `${label} Qwen block-0 serialized boundaries are not ordered/bound to input and output`,
      );
    }
    if (
      qwenBlock0PostSyncEvents.length === 1 &&
      qwenPreHandoffEvents.length === 1 &&
      (qwenBlock0PostSyncEvents[0].at_ms >= qwenPreHandoffEvents[0].at_ms ||
        JSON.stringify(qwenBlock0PostSyncEvents[0].diagnostic) !==
          JSON.stringify(qwenPreHandoffEvents[0].diagnostics?.block_00_immediate_post_sync))
    ) {
      failures.push(
        `${label} Qwen block-0 post-sync event is not ordered before and JSON-bound to pre-handoff evidence`,
      );
    }
    if (qwenPreHandoffEvents.length !== 1 || qwenPostHandoffEvents.length !== 1) {
      failures.push(`${label} does not contain exactly one Qwen pre/post handoff event`);
    } else {
      failures.push(
        ...validatePackedF16QwenPreHandoffDiagnostics(
          qwenPreHandoffEvents[0],
          request?.progress_events,
          label,
        ),
        ...validatePackedF16QwenPostHandoffDiagnostics(
          qwenPostHandoffEvents[0],
          qwenPreHandoffEvents[0],
          request?.progress_events,
          label,
        ),
      );
      if (
        JSON.stringify(request?.packed_f16_qwen_pre_handoff_diagnostics) !==
          JSON.stringify(qwenPreHandoffEvents[0]) ||
        JSON.stringify(request?.packed_f16_qwen_post_handoff_diagnostics) !==
          JSON.stringify(qwenPostHandoffEvents[0])
      ) {
        failures.push(`${label} top-level Qwen handoff diagnostics differ from runtime events`);
      }
      if (
        qwenHostEmbeddingEvents[0]?.report?.device_f32_sha256 !==
        qwenPreHandoffEvents[0]?.diagnostics?.stage_outputs?.[0]?.sha256
      ) {
        failures.push(`${label} Qwen host-embedding digest differs from first Qwen activation`);
      }
    }
    const preDmdInputDiagnosticEvents = (request?.runtime_events ?? []).filter(
      (event) => event?.event === "packed_f16_pre_dmd_input_diagnostics",
    );
    if (preDmdInputDiagnosticEvents.length !== 1) {
      failures.push(`${label} does not contain exactly one packed-F16 pre-DMD input diagnostic event`);
    } else {
      failures.push(
        ...validatePackedF16PreDmdInputDiagnostics(
          preDmdInputDiagnosticEvents[0],
          request?.progress_events,
          label,
        ),
      );
      if (
        JSON.stringify(request?.packed_f16_pre_dmd_input_diagnostics) !==
        JSON.stringify(preDmdInputDiagnosticEvents[0])
      ) {
        failures.push(
          `${label} top-level packed-F16 pre-DMD input diagnostics differ from its runtime event`,
        );
      }
      if (
        preDmdInputDiagnosticEvents[0]?.diagnostics?.instruction?.sha256 !==
        qwenPostHandoffEvents[0]?.diagnostics?.handoff?.after_sha256
      ) {
        failures.push(`${label} pre-DMD instruction digest differs from its Qwen handoff`);
      }
    }
    if (lifecycleEvents.length !== 1) {
      failures.push(`${label} does not contain exactly one packed-F16 lifecycle event`);
    } else {
      failures.push(
        ...validatePackedF16Lifecycle(lifecycleEvents[0].lifecycle, label, index + 1),
      );
      if (
        JSON.stringify(request?.packed_f16_denoiser_lifecycle) !==
        JSON.stringify(lifecycleEvents[0].lifecycle)
      ) {
        failures.push(`${label} top-level packed-F16 lifecycle differs from its runtime event`);
      }
      if (
        positiveFinite(preDmdInputDiagnosticEvents[0]?.at_ms) &&
        positiveFinite(lifecycleEvents[0]?.at_ms) &&
        preDmdInputDiagnosticEvents[0].at_ms >= lifecycleEvents[0].at_ms
      ) {
        failures.push(`${label} pre-DMD input diagnostics were not emitted before the lifecycle event`);
      }
      if (
        positiveFinite(trafficEvents[0]?.at_ms) &&
        positiveFinite(lifecycleEvents[0]?.at_ms) &&
        lifecycleEvents[0].at_ms >= trafficEvents[0].at_ms
      ) {
        failures.push(`${label} packed-F16 lifecycle was not emitted before final request traffic`);
      }
    }
    if (dmdVaeHandoffEvents.length !== 1) {
      failures.push(`${label} does not contain exactly one packed-F16 DMD-to-VAE handoff event`);
    } else {
      failures.push(
        ...validatePackedF16DmdVaeHandoff(
          dmdVaeHandoffEvents[0],
          request?.progress_events,
          index + 1,
          label,
        ),
      );
      if (
        JSON.stringify(request?.packed_f16_dmd_vae_handoff) !==
        JSON.stringify(dmdVaeHandoffEvents[0])
      ) {
        failures.push(`${label} top-level DMD-to-VAE handoff differs from its runtime event`);
      }
      if (
        lifecycleEvents.length === 1 &&
        (!positiveFinite(lifecycleEvents[0]?.at_ms) ||
          lifecycleEvents[0].at_ms >= dmdVaeHandoffEvents[0].at_ms)
      ) {
        failures.push(`${label} DMD lifecycle was not emitted before packed-cache eviction`);
      }
      if (
        positiveFinite(trafficEvents[0]?.at_ms) &&
        dmdVaeHandoffEvents[0].at_ms >= trafficEvents[0].at_ms
      ) {
        failures.push(`${label} DMD-to-VAE handoff was not emitted before final request traffic`);
      }
    }
    const outputs = (request?.output_events ?? []).filter(
      (event) => event?.event === "ready" && event?.model === TURBO_MODEL_ID,
    );
    if (outputs.length !== 1 || JSON.stringify(outputs[0]) !== JSON.stringify(request?.output_ready)) {
      failures.push(`${label} output-ready evidence differs from its recorded output event`);
    }
  }

  if (first?.event_boundaries?.runtime_end_index !== second?.event_boundaries?.runtime_start_index) {
    failures.push("runtime event partitions do not prove a contiguous same-engine handoff");
  }
  if (first?.event_boundaries?.progress_end_index !== second?.event_boundaries?.progress_start_index) {
    failures.push("progress event partitions do not prove a contiguous same-engine handoff");
  }
  if (first?.event_boundaries?.output_end_index !== second?.event_boundaries?.output_start_index) {
    failures.push("output event partitions do not prove a contiguous same-engine handoff");
  }
  if (first?.output_ready?.job_id === second?.output_ready?.job_id) {
    failures.push("the two sequential outputs have the same job ID");
  }
  const firstRun = first?.progress_events?.find((event) => event?.event === "run_started");
  const secondRun = second?.progress_events?.find((event) => event?.event === "run_started");
  if (JSON.stringify(firstRun?.run_id) === JSON.stringify(secondRun?.run_id)) {
    failures.push("the two sequential requests have the same run ID");
  }
  if (first?.downloaded_png?.file_name === second?.downloaded_png?.file_name) {
    failures.push("the two production PNG downloads have the same filename");
  }
  if (first?.downloaded_png?.sha256 === second?.downloaded_png?.sha256) {
    failures.push("the seed-0 and seed-1 production PNG downloads have the same digest");
  }
  if (
    JSON.stringify(first?.tested_package_identity) !== JSON.stringify(second?.tested_package_identity) ||
    JSON.stringify(first?.tested_package_identity) !== JSON.stringify(evidence.tested_package_identity)
  ) {
    failures.push("per-request package identities differ from the same-engine tested package");
  }
  if (JSON.stringify(first?.served_transport) !== JSON.stringify(second?.served_transport)) {
    failures.push("per-request served package transport identities differ");
  }
  if (
    JSON.stringify(first?.modular_artifact_transport) !==
    JSON.stringify(second?.modular_artifact_transport)
  ) {
    failures.push("per-request physical model transport identities differ");
  }
  const deterministicQwenDigests = (request) => ({
    host_embedding_host:
      request?.packed_f16_qwen_host_embedding?.report?.host_f32_sha256 ?? null,
    host_embedding_device:
      request?.packed_f16_qwen_host_embedding?.report?.device_f32_sha256 ?? null,
    stage_outputs:
      request?.packed_f16_qwen_pre_handoff_diagnostics?.diagnostics?.stage_outputs?.map(
        (tensor) => ({ name: tensor?.name ?? null, shape: tensor?.shape ?? null, sha256: tensor?.sha256 ?? null }),
      ) ?? null,
    returned_hidden:
      request?.packed_f16_qwen_pre_handoff_diagnostics?.diagnostics
        ?.qwen_last_hidden_state_before_trim?.sha256 ?? null,
    instruction_before_handoff:
      request?.packed_f16_qwen_pre_handoff_diagnostics?.diagnostics
        ?.instruction_after_trim_cast_before_handoff?.sha256 ?? null,
    handoff_before:
      request?.packed_f16_qwen_post_handoff_diagnostics?.diagnostics?.handoff?.before_sha256 ?? null,
    handoff_after:
      request?.packed_f16_qwen_post_handoff_diagnostics?.diagnostics?.handoff?.after_sha256 ?? null,
    instruction_after_handoff:
      request?.packed_f16_qwen_post_handoff_diagnostics?.diagnostics?.instruction_after_handoff
        ?.sha256 ?? null,
    pre_dmd_instruction:
      request?.packed_f16_pre_dmd_input_diagnostics?.diagnostics?.instruction?.sha256 ?? null,
  });
  const firstQwenDigests = deterministicQwenDigests(first);
  const secondQwenDigests = deterministicQwenDigests(second);
  if (
    !Array.isArray(firstQwenDigests.stage_outputs) ||
    firstQwenDigests.stage_outputs.length !== TURBO_PACKED_F16_QWEN_STAGE_OUTPUT_COUNT ||
    !Array.isArray(secondQwenDigests.stage_outputs) ||
    secondQwenDigests.stage_outputs.length !== TURBO_PACKED_F16_QWEN_STAGE_OUTPUT_COUNT ||
    JSON.stringify(firstQwenDigests) !== JSON.stringify(secondQwenDigests)
  ) {
    failures.push(
      "same-prompt requests do not have exact cross-request Qwen host, 38-stage, handoff, and pre-DMD instruction digests",
    );
  }

  const firstTraffic = first?.runtime_events?.find((event) => event?.event === "artifact_traffic")?.traffic;
  failures.push(...validateExactArtifactTraffic(firstTraffic, TURBO_GENERATE_REQUEST_TRAFFIC, "first request"));
  failures.push(
    ...validateCdpNetworkAgainstRust(
      first?.cdp_network_traffic,
      firstTraffic,
      TURBO_CDP_REQUEST_NETWORK_POLICY,
      "ready",
      "first request",
    ),
  );
  const secondTraffic = second?.runtime_events?.find((event) => event?.event === "artifact_traffic")?.traffic;
  failures.push(
    ...validateExactArtifactTraffic(
      secondTraffic,
      TURBO_REPEAT_GENERATE_REQUEST_TRAFFIC,
      "second request",
    ),
  );
  failures.push(
    ...validateCdpNetworkAgainstRust(
      second?.cdp_network_traffic,
      secondTraffic,
      TURBO_CDP_REQUEST_NETWORK_POLICY,
      "ready",
      "second request",
    ),
  );
  failures.push(
    ...validateSecondRequestInteraction(
      second?.interaction,
      evidence.fixed_ascii_prompt,
      second,
      first,
    ),
  );

  const retained = evidence.request_scoped_denoiser_policy;
  if (
    retained?.expected_stages !== TURBO_PACKED_F16_CACHED_STAGES ||
    retained?.expected_objects !== TURBO_PACKED_F16_CACHED_OBJECTS ||
    retained?.expected_tensors !== TURBO_PACKED_F16_CACHED_TENSORS ||
    retained?.expected_bytes !==
      TURBO_PACKED_F16_RESOURCE_PLAN.retained_packed_f16_denoiser_bytes ||
    retained?.initial_preload_stages !== TURBO_PACKED_F16_CACHED_STAGES ||
    retained?.initial_preload_objects !== TURBO_PACKED_F16_CACHED_OBJECTS ||
    retained?.initial_preload_tensors !== TURBO_PACKED_F16_CACHED_TENSORS ||
    retained?.initial_preload_bytes !==
      TURBO_PACKED_F16_RESOURCE_PLAN.retained_packed_f16_denoiser_bytes ||
    retained?.request_local_preload_events !== 1 ||
    retained?.successful_requests_with_post_dmd_eviction !== 2 ||
    retained?.policy !== TURBO_DENOISER_STORAGE_POLICY ||
    retained?.raw_packed_cache_empty_before_vae !== true ||
    retained?.repeat_rehydration_cache_only !== true ||
    JSON.stringify(retained?.preload_attempt_counts) !== JSON.stringify([1, 2])
  ) {
    failures.push(
      "multi-request evidence does not attest request-scoped eviction and exact warm rehydration",
    );
  }
  return failures;
}
