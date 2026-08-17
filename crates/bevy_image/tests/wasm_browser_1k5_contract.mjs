import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { join } from "node:path";

import {
  ARTIFACT_TRANSPORT_HARD_MAX_PART_BYTES,
  ARTIFACT_TRANSPORT_LAYOUT_PATH,
  ARTIFACT_TRANSPORT_TARGET_PART_BYTES,
} from "./artifact_transport_contract.mjs";

export const BROWSER_1K5_F32_QUALIFICATION_DENOISER_RESIDENCY =
  "request-scoped-f32-policy-retained-through-four-dmd-steps";
export const BROWSER_1K5_LOW_VRAM_DENOISER_RESIDENCY =
  "request-scoped-runtime-q8-policy-retained-through-four-dmd-steps";
export const BROWSER_1K5_LOW_VRAM_STRICT_DEVICE_CAP_BYTES = 32_000_000_000;
export const BROWSER_1K5_LOW_VRAM_RESOURCE_PLAN = Object.freeze({
  audited_retained_q8_denoiser_bytes: 12_590_785_792,
  expected_q8s_block32_f32_tensor_count: 377,
  expected_f32_tensor_count: 565,
  expected_q8s_block32_f32_elements: 9_940_674_560,
  expected_f32_elements: 351_881_728,
  expected_q8s_block32_f32_payload_bytes: 11_183_258_880,
  expected_f32_payload_bytes: 1_407_526_912,
  audited_max_streamed_qwen_stage_f32_bytes: 771_785_728,
  audited_loaded_vae_module_f32_bytes: 335_278_732,
  audited_max_dense_denoiser_stage_f32_bytes: 0,
  audited_max_phase_local_f32_stage_bytes: 771_785_728,
  runtime_quantization_workspace_bytes: 2_434_252_800,
  activation_reserve_bytes: 14_605_516_800,
  conservative_planned_device_bytes: 30_402_341_120,
  strict_device_cap_bytes: BROWSER_1K5_LOW_VRAM_STRICT_DEVICE_CAP_BYTES,
});
export const BROWSER_1K5_CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES = 256 * 1024 * 1024;
export const BROWSER_1K5_CHROME_SHARED_MEMORY_DEV_SHM_POLICY =
  "linux-dev-shm-statfs-and-quota-aware-probe-admitted";
export const BROWSER_1K5_CHROME_SHARED_MEMORY_TMP_FALLBACK_POLICY =
  "linux-temp-fallback-dev-shm-not-admitted";
export const BROWSER_1K5_CHROME_SHARED_MEMORY_PLATFORM_DEFAULT_POLICY =
  "non-linux-platform-default";

const SOFTWARE_ADAPTER = /(swiftshader|llvmpipe|lavapipe|software adapter|warp)/i;
const NVIDIA_ADAPTER = /nvidia/i;
const RUNTIME_ADAPTER_SOURCE = "instrumented-navigator-gpu-request-adapter";

function exactNonNegativeInteger(value) {
  if (Number.isSafeInteger(value) && value >= 0) return BigInt(value);
  if (typeof value === "string" && /^(0|[1-9][0-9]*)$/.test(value)) {
    return BigInt(value);
  }
  return null;
}

export function selectBrowser1k5ChromeSharedMemoryPolicy({
  platform,
  devShm,
  tempPath = null,
  minimumHeadroomBytes = BROWSER_1K5_CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES,
}) {
  if (!Number.isSafeInteger(minimumHeadroomBytes) || minimumHeadroomBytes <= 0) {
    throw new Error("Chrome shared-memory minimum headroom must be a positive safe integer");
  }
  if (platform !== "linux") {
    return {
      policy: BROWSER_1K5_CHROME_SHARED_MEMORY_PLATFORM_DEFAULT_POLICY,
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
      policy: BROWSER_1K5_CHROME_SHARED_MEMORY_DEV_SHM_POLICY,
      selected_backing: "dev-shm",
      selected_path: devShm.path,
      disable_dev_shm_usage: false,
      minimum_admitted_headroom_bytes: minimumHeadroomBytes,
      dev_shm_admitted: true,
      dev_shm_rejections: [],
    };
  }
  return {
    policy: BROWSER_1K5_CHROME_SHARED_MEMORY_TMP_FALLBACK_POLICY,
    selected_backing: "temp-directory",
    selected_path: typeof tempPath === "string" ? tempPath : null,
    disable_dev_shm_usage: true,
    minimum_admitted_headroom_bytes: minimumHeadroomBytes,
    dev_shm_admitted: false,
    dev_shm_rejections: rejections,
  };
}

export function browser1k5ChromeLaunchEvidence({
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

const BROWSER_PACKAGE_IDENTITY_PATHS = Object.freeze({
  browser_javascript: "crates/bevy_image/www/out/bevy_burn_image.js",
  browser_wasm: "crates/bevy_image/www/out/bevy_burn_image_bg.wasm",
  browser_icon: "crates/bevy_image/www/out/burn-image-icon.png",
  browser_runtime_source: "crates/bevy_image/src/browser_boogu.rs",
  qualification_harness_script: "crates/bevy_image/tests/wasm_browser_1k5_parity.mjs",
  qualification_harness_page: "crates/bevy_image/tests/wasm_browser_1k5_parity.html",
  qualification_contract_source: "crates/bevy_image/tests/wasm_browser_1k5_contract.mjs",
  qualification_scope_source: "crates/bevy_image/tests/wasm_browser_1k5_scope.mjs",
  artifact_transport_contract_source:
    "crates/bevy_image/tests/artifact_transport_contract.mjs",
});

export function denoiserResidencyPolicyForMode(residencyMode) {
  if (residencyMode === "low-vram") {
    return BROWSER_1K5_LOW_VRAM_DENOISER_RESIDENCY;
  }
  if (residencyMode === "qualification-f32") {
    return BROWSER_1K5_F32_QUALIFICATION_DENOISER_RESIDENCY;
  }
  throw new Error(`unsupported 1.5K browser residency mode ${JSON.stringify(residencyMode)}`);
}

export function validateDenoiserResidencyPolicy(attestation, residencyMode) {
  const expectedPolicy = denoiserResidencyPolicyForMode(residencyMode);
  return attestation?.policy === expectedPolicy
    ? []
    : [
        `denoiser residency policy is ${JSON.stringify(attestation?.policy ?? null)}; expected ${JSON.stringify(expectedPolicy)} for ${residencyMode}`,
      ];
}

export function validateBrowser1k5TransportValidation(validation) {
  const failures = [];
  for (const field of [
    "artifact_file_count",
    "artifact_weight_file_count",
    "artifact_bytes",
    "artifact_weight_bytes",
    "physical_transport_unique_part_count",
    "physical_transport_unique_part_bytes",
    "direct_artifact_file_count",
    "direct_artifact_bytes",
  ]) {
    if (!Number.isSafeInteger(validation?.[field]) || validation[field] <= 0) {
      failures.push(`${field} is not a positive safe integer`);
    }
  }
  if (
    validation?.transport_layout_path !== ARTIFACT_TRANSPORT_LAYOUT_PATH ||
    validation?.physical_transport_target_part_bytes !==
      ARTIFACT_TRANSPORT_TARGET_PART_BYTES ||
    validation?.physical_transport_hard_max_part_bytes !==
      ARTIFACT_TRANSPORT_HARD_MAX_PART_BYTES ||
    !Number.isSafeInteger(validation?.physical_transport_max_part_bytes) ||
    validation.physical_transport_max_part_bytes <= 0 ||
    validation.physical_transport_max_part_bytes > ARTIFACT_TRANSPORT_HARD_MAX_PART_BYTES
  ) {
    failures.push("physical transport does not attest the exact 20 MiB target / 25000000-byte cap");
  }
  if (
    validation?.physical_transport_reconstructed_bytes !== validation?.artifact_weight_bytes
  ) {
    failures.push("physical transport reconstruction bytes differ from logical weight bytes");
  }
  if (!Array.isArray(validation?.transport_sidecars) || validation.transport_sidecars.length !== 3) {
    failures.push("transport validation does not contain exactly three modular sidecars");
  } else {
    for (const sidecar of validation.transport_sidecars) {
      if (
        sidecar?.path !== ARTIFACT_TRANSPORT_LAYOUT_PATH ||
        sidecar?.authenticated !== true ||
        !/^[0-9a-f]{64}$/.test(sidecar?.sha256 ?? "")
      ) {
        failures.push(`${JSON.stringify(sidecar?.bundle)} sidecar is not SHA-256 authenticated`);
      }
    }
  }
  return failures;
}

function resourcePlanValue(value) {
  return value === undefined ? "undefined" : JSON.stringify(value);
}

/** Validate the exact inventory-derived resource plan emitted by the current 1.5K runtime. */
export function validateBrowser1k5LowVramResourcePlan(plan) {
  if (!plan || typeof plan !== "object" || Array.isArray(plan)) {
    return ["low_vram_resource_plan is not an object"];
  }

  const failures = [];
  const expectedFields = new Set(Object.keys(BROWSER_1K5_LOW_VRAM_RESOURCE_PLAN));
  for (const [field, expected] of Object.entries(BROWSER_1K5_LOW_VRAM_RESOURCE_PLAN)) {
    if (plan[field] !== expected) {
      failures.push(
        `low_vram_resource_plan.${field}=${resourcePlanValue(plan[field])}, expected ${resourcePlanValue(expected)}`,
      );
    }
  }
  for (const field of Object.keys(plan)) {
    if (!expectedFields.has(field)) {
      failures.push(`low_vram_resource_plan.${field} is an unexpected field`);
    }
  }

  if (
    plan.expected_q8s_block32_f32_payload_bytes + plan.expected_f32_payload_bytes !==
    plan.audited_retained_q8_denoiser_bytes
  ) {
    failures.push("low_vram_resource_plan payload bytes do not sum to retained denoiser bytes");
  }
  if (
    Math.max(
      plan.audited_max_streamed_qwen_stage_f32_bytes,
      plan.audited_loaded_vae_module_f32_bytes,
      plan.audited_max_dense_denoiser_stage_f32_bytes,
    ) !== plan.audited_max_phase_local_f32_stage_bytes
  ) {
    failures.push("low_vram_resource_plan phase-local maximum is inconsistent");
  }
  if (
    plan.audited_retained_q8_denoiser_bytes +
      plan.audited_max_phase_local_f32_stage_bytes +
      plan.runtime_quantization_workspace_bytes +
      plan.activation_reserve_bytes !==
    plan.conservative_planned_device_bytes
  ) {
    failures.push("low_vram_resource_plan conservative device-byte sum is inconsistent");
  }
  if (!(plan.conservative_planned_device_bytes < plan.strict_device_cap_bytes)) {
    failures.push("low_vram_resource_plan is not strictly below its device cap");
  }
  return failures;
}

/** Normalize the exact navigator.gpu call sequence made after the 1.5K harness navigation. */
export function summarizeBrowser1k5RuntimeWebGpuCalls(webGpuCalls) {
  const calls = Array.isArray(webGpuCalls) ? webGpuCalls : [];
  const adapterStarts = calls.filter((entry) => entry?.event === "request-adapter-start");
  const adapterRejections = calls.filter((entry) => entry?.event === "request-adapter-rejected");
  const adapterSuccesses = calls.filter(
    (entry) => entry?.event === "request-adapter-resolved" && entry?.detail?.available === true,
  );
  const deviceStarts = calls.filter((entry) => entry?.event === "request-device-start");
  const deviceRejections = calls.filter((entry) => entry?.event === "request-device-rejected");
  const deviceSuccesses = calls.filter((entry) => entry?.event === "request-device-resolved");
  const selected = adapterSuccesses.at(-1)?.detail?.info ?? null;
  const selectedStart = adapterStarts.at(-1)?.detail ?? null;
  return {
    source: RUNTIME_ADAPTER_SOURCE,
    adapter_request_attempts: adapterStarts.length,
    adapter_rejected_attempts: adapterRejections.length,
    adapter_successful_attempts: adapterSuccesses.length,
    device_request_attempts: deviceStarts.length,
    device_rejected_attempts: deviceRejections.length,
    device_successful_attempts: deviceSuccesses.length,
    power_preference: selectedStart?.powerPreference ?? null,
    force_fallback_adapter: selectedStart?.forceFallbackAdapter ?? null,
    is_fallback_adapter:
      selected?.is_fallback_adapter ?? selected?.isFallbackAdapter ?? null,
    vendor: selected?.vendor ?? null,
    architecture: selected?.architecture ?? null,
    device: selected?.device ?? null,
    description: selected?.description ?? null,
  };
}

/**
 * Bind a redacted wgpu BrowserWebGpu report to the browser-native adapter that created its device.
 *
 * wgpu 29 maps every BrowserWebGpu adapter to `DeviceType::Other`, so the Rust enum is not hardware
 * evidence. This attestation fails closed unless the exact instrumented request used by the Wasm
 * runtime returned an explicitly non-fallback NVIDIA adapter and exactly one device.
 */
export function attestBrowser1k5RuntimeAdapter(report, webGpuCalls) {
  const attestation = summarizeBrowser1k5RuntimeWebGpuCalls(webGpuCalls);
  const failures = [];
  if (!/browserwebgpu/i.test(report?.adapter_backend ?? "")) {
    failures.push(
      `runtime report adapter_backend is ${JSON.stringify(report?.adapter_backend ?? null)}; expected BrowserWebGpu`,
    );
  }
  const deviceType = report?.adapter_device_type ?? null;
  if (deviceType === "Other") {
    if (attestation.source !== RUNTIME_ADAPTER_SOURCE) {
      failures.push("runtime adapter attestation does not come from the exact instrumented request");
    }
    if (
      !Number.isSafeInteger(attestation.adapter_request_attempts) ||
      attestation.adapter_request_attempts < 1
    ) {
      failures.push("runtime adapter attestation has no adapter request attempt");
    }
    if (attestation.adapter_successful_attempts !== 1) {
      failures.push(
        `runtime adapter attestation has ${JSON.stringify(attestation.adapter_successful_attempts)} successful adapter requests; expected exactly one`,
      );
    }
    if (attestation.device_request_attempts !== 1 || attestation.device_successful_attempts !== 1) {
      failures.push(
        `runtime adapter attestation has device requests=${JSON.stringify(attestation.device_request_attempts)}, successes=${JSON.stringify(attestation.device_successful_attempts)}; expected exactly one of each`,
      );
    }
    if (attestation.device_rejected_attempts !== 0) {
      failures.push(
        `runtime adapter attestation has ${attestation.device_rejected_attempts} rejected device requests`,
      );
    }
    if (attestation.power_preference !== "high-performance") {
      failures.push(
        `runtime adapter power preference is ${JSON.stringify(attestation.power_preference)}`,
      );
    }
    if (attestation.force_fallback_adapter === true) {
      failures.push("runtime adapter request explicitly forced fallback");
    }
    if (attestation.is_fallback_adapter !== false) {
      failures.push(
        `runtime adapter fallback status is not explicitly false: ${JSON.stringify(attestation.is_fallback_adapter)}`,
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
        `runtime adapter evidence is not non-software NVIDIA hardware: ${JSON.stringify(nativeIdentity)}`,
      );
    }
    if (
      typeof attestation.description !== "string" ||
      attestation.description !== report?.adapter_name
    ) {
      failures.push(
        `runtime adapter description ${JSON.stringify(attestation.description)} does not match Rust adapter_name ${JSON.stringify(report?.adapter_name ?? null)}`,
      );
    }
  } else if (!["DiscreteGpu", "IntegratedGpu", "VirtualGpu"].includes(deviceType)) {
    failures.push(
      `runtime report adapter_device_type is ${JSON.stringify(deviceType)}; expected a GPU`,
    );
  }
  return {
    ...attestation,
    rust_adapter_name: report?.adapter_name ?? null,
    rust_adapter_backend: report?.adapter_backend ?? null,
    rust_adapter_device_type: deviceType,
    validation_failures: failures,
    validated: failures.length === 0,
  };
}

async function sha256File(path) {
  const metadata = await stat(path);
  if (!metadata.isFile()) throw new Error(`package identity input is not a file: ${path}`);
  const digest = createHash("sha256");
  await new Promise((resolveHash, rejectHash) => {
    const stream = createReadStream(path);
    stream.on("data", (chunk) => digest.update(chunk));
    stream.once("error", rejectHash);
    stream.once("end", resolveHash);
  });
  return { size_bytes: metadata.size, sha256: digest.digest("hex") };
}

async function identifyPackageFile(path, logicalPath) {
  try {
    return { logical_path: logicalPath, ...(await sha256File(path)) };
  } catch (error) {
    return {
      logical_path: logicalPath,
      size_bytes: null,
      sha256: null,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

export function validateBrowserPackageIdentity(identity) {
  const failures = [];
  if (identity?.schema_version !== 1) {
    failures.push(
      `browser package identity schema_version is ${identity?.schema_version}; expected 1`,
    );
  }
  for (const [name, expectedPath] of Object.entries(BROWSER_PACKAGE_IDENTITY_PATHS)) {
    const file = identity?.files?.[name];
    if (file?.logical_path !== expectedPath) {
      failures.push(
        `${name}.logical_path is ${JSON.stringify(file?.logical_path ?? null)}; expected ${JSON.stringify(expectedPath)}`,
      );
    }
    if (!Number.isSafeInteger(file?.size_bytes) || file.size_bytes <= 0) {
      failures.push(`${name}.size_bytes is not a positive safe integer`);
    }
    if (!/^[0-9a-f]{64}$/.test(file?.sha256 ?? "")) {
      failures.push(`${name}.sha256 is not an exact lowercase SHA-256 digest`);
    }
    if (file?.error) failures.push(`${name} could not be identified: ${file.error}`);
  }
  return failures;
}

export async function collectBrowserPackageIdentity({
  wwwOutDir,
  repoRoot,
  testsDir,
  harnessScriptPath,
}) {
  const fileInputs = {
    browser_javascript: join(wwwOutDir, "bevy_burn_image.js"),
    browser_wasm: join(wwwOutDir, "bevy_burn_image_bg.wasm"),
    browser_icon: join(wwwOutDir, "burn-image-icon.png"),
    browser_runtime_source: join(repoRoot, "crates/bevy_image/src/browser_boogu.rs"),
    qualification_harness_script: harnessScriptPath,
    qualification_harness_page: join(testsDir, "wasm_browser_1k5_parity.html"),
    qualification_contract_source: join(testsDir, "wasm_browser_1k5_contract.mjs"),
    qualification_scope_source: join(testsDir, "wasm_browser_1k5_scope.mjs"),
    artifact_transport_contract_source: join(testsDir, "artifact_transport_contract.mjs"),
  };
  const files = Object.fromEntries(
    await Promise.all(
      Object.entries(fileInputs).map(async ([name, path]) => [
        name,
        await identifyPackageFile(path, BROWSER_PACKAGE_IDENTITY_PATHS[name]),
      ]),
    ),
  );
  const identity = { schema_version: 1, files };
  const validationFailures = validateBrowserPackageIdentity(identity);
  return {
    ...identity,
    validation_failures: validationFailures,
    validated: validationFailures.length === 0,
  };
}
