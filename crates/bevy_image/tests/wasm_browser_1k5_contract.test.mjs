import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  BROWSER_1K5_CHROME_SHARED_MEMORY_DEV_SHM_POLICY,
  BROWSER_1K5_CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES,
  BROWSER_1K5_CHROME_SHARED_MEMORY_PLATFORM_DEFAULT_POLICY,
  BROWSER_1K5_CHROME_SHARED_MEMORY_TMP_FALLBACK_POLICY,
  BROWSER_1K5_LOW_VRAM_RESOURCE_PLAN,
  attestBrowser1k5RuntimeAdapter,
  browser1k5ChromeLaunchEvidence,
  collectBrowserPackageIdentity,
  denoiserResidencyPolicyForMode,
  selectBrowser1k5ChromeSharedMemoryPolicy,
  summarizeBrowser1k5RuntimeWebGpuCalls,
  validateBrowser1k5LowVramResourcePlan,
  validateBrowserPackageIdentity,
  validateDenoiserResidencyPolicy,
} from "./wasm_browser_1k5_contract.mjs";

const testsDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(testsDir, "../../..");
const harnessScriptPath = join(testsDir, "wasm_browser_1k5_parity.mjs");
const LOW_VRAM_POLICY = "request-scoped-runtime-q8-policy-retained-through-four-dmd-steps";
const QUALIFICATION_F32_POLICY =
  "request-scoped-f32-policy-retained-through-four-dmd-steps";

function admittedSharedMemoryMeasurement({
  availableBytes = BROWSER_1K5_CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES,
  writtenBytes = BROWSER_1K5_CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES,
} = {}) {
  return {
    path: "/dev/shm",
    exists: true,
    directory: true,
    writable: true,
    statfs: { available_bytes: availableBytes },
    quota_aware_allocation_probe: {
      attempted: true,
      requested_bytes: BROWSER_1K5_CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES,
      written_bytes: writtenBytes,
      succeeded: true,
    },
  };
}

function runtimeWebGpuCalls({ fallback = false, description = "", vendor = "nvidia" } = {}) {
  return [
    {
      event: "request-adapter-start",
      detail: { powerPreference: "high-performance" },
    },
    {
      event: "request-adapter-resolved",
      detail: {
        available: true,
        info: {
          is_fallback_adapter: fallback,
          vendor,
          architecture: "blackwell",
          device: "",
          description,
        },
      },
    },
    { event: "request-device-start", detail: { requiredFeatures: [], requiredLimits: {} } },
    { event: "request-device-resolved", detail: { requiredFeatures: [], requiredLimits: {} } },
  ];
}

function redactedRuntimeReport() {
  return {
    adapter_name: "",
    adapter_backend: "BrowserWebGpu",
    adapter_device_type: "Other",
  };
}

test("accepts the exact inventory-derived 1.5K low-VRAM resource plan", () => {
  assert.deepEqual(
    validateBrowser1k5LowVramResourcePlan({ ...BROWSER_1K5_LOW_VRAM_RESOURCE_PLAN }),
    [],
  );
});

test("rejects stale, missing, mutated, and internally inconsistent resource-plan fields", () => {
  const stale = { ...BROWSER_1K5_LOW_VRAM_RESOURCE_PLAN };
  delete stale.audited_max_streamed_qwen_stage_f32_bytes;
  stale.audited_max_streamed_stage_bytes = 771_785_728;
  const staleFailures = validateBrowser1k5LowVramResourcePlan(stale);
  assert.ok(
    staleFailures.some((failure) =>
      failure.includes("audited_max_streamed_qwen_stage_f32_bytes=undefined"),
    ),
    staleFailures.join("\n"),
  );
  assert.ok(
    staleFailures.some((failure) =>
      failure.includes("audited_max_streamed_stage_bytes is an unexpected field"),
    ),
    staleFailures.join("\n"),
  );

  for (const field of [
    "audited_max_streamed_qwen_stage_f32_bytes",
    "audited_loaded_vae_module_f32_bytes",
    "audited_max_dense_denoiser_stage_f32_bytes",
    "audited_max_phase_local_f32_stage_bytes",
  ]) {
    const mutated = { ...BROWSER_1K5_LOW_VRAM_RESOURCE_PLAN, [field]: 1 };
    const failures = validateBrowser1k5LowVramResourcePlan(mutated);
    assert.ok(
      failures.some((failure) => failure.includes(`low_vram_resource_plan.${field}=1`)),
      `${field}: ${failures.join("\n")}`,
    );
  }

  const inconsistent = {
    ...BROWSER_1K5_LOW_VRAM_RESOURCE_PLAN,
    conservative_planned_device_bytes:
      BROWSER_1K5_LOW_VRAM_RESOURCE_PLAN.strict_device_cap_bytes,
  };
  const inconsistentFailures = validateBrowser1k5LowVramResourcePlan(inconsistent);
  assert.ok(
    inconsistentFailures.includes(
      "low_vram_resource_plan conservative device-byte sum is inconsistent",
    ),
    inconsistentFailures.join("\n"),
  );
  assert.ok(
    inconsistentFailures.includes("low_vram_resource_plan is not strictly below its device cap"),
    inconsistentFailures.join("\n"),
  );
});

test("selects and exactly validates the denoiser residency policy for each mode", () => {
  assert.equal(denoiserResidencyPolicyForMode("low-vram"), LOW_VRAM_POLICY);
  assert.equal(
    denoiserResidencyPolicyForMode("qualification-f32"),
    QUALIFICATION_F32_POLICY,
  );
  assert.deepEqual(
    validateDenoiserResidencyPolicy({ policy: LOW_VRAM_POLICY }, "low-vram"),
    [],
  );
  assert.deepEqual(
    validateDenoiserResidencyPolicy({ policy: QUALIFICATION_F32_POLICY }, "qualification-f32"),
    [],
  );
  assert.notDeepEqual(
    validateDenoiserResidencyPolicy({ policy: QUALIFICATION_F32_POLICY }, "low-vram"),
    [],
  );
  assert.notDeepEqual(
    validateDenoiserResidencyPolicy({ policy: LOW_VRAM_POLICY }, "qualification-f32"),
    [],
  );
  assert.throws(() => denoiserResidencyPolicyForMode("high-vram"), /unsupported/);
});

test("admits /dev/shm only after exact global and quota-aware capacity proof", () => {
  assert.deepEqual(
    selectBrowser1k5ChromeSharedMemoryPolicy({
      platform: "linux",
      devShm: admittedSharedMemoryMeasurement(),
      tempPath: "/tmp",
    }),
    {
      policy: BROWSER_1K5_CHROME_SHARED_MEMORY_DEV_SHM_POLICY,
      selected_backing: "dev-shm",
      selected_path: "/dev/shm",
      disable_dev_shm_usage: false,
      minimum_admitted_headroom_bytes:
        BROWSER_1K5_CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES,
      dev_shm_admitted: true,
      dev_shm_rejections: [],
    },
  );

  const overflowSafe = selectBrowser1k5ChromeSharedMemoryPolicy({
    platform: "linux",
    devShm: admittedSharedMemoryMeasurement({
      availableBytes: "18446744073709551615",
      writtenBytes: String(BROWSER_1K5_CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES),
    }),
    tempPath: "/tmp",
  });
  assert.equal(overflowSafe.disable_dev_shm_usage, false);
  assert.equal(overflowSafe.selected_path, "/dev/shm");

  const unsafeNumber = selectBrowser1k5ChromeSharedMemoryPolicy({
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
      selectBrowser1k5ChromeSharedMemoryPolicy({
        platform: "linux",
        devShm: admittedSharedMemoryMeasurement(),
        minimumHeadroomBytes: Number.MAX_SAFE_INTEGER + 1,
      }),
    /positive safe integer/,
  );
});

test("uses temp-backed Chrome shared memory when /dev/shm is missing, tiny, or quota-limited", () => {
  const missing = selectBrowser1k5ChromeSharedMemoryPolicy({
    platform: "linux",
    devShm: null,
    tempPath: "/tmp",
  });
  assert.equal(missing.policy, BROWSER_1K5_CHROME_SHARED_MEMORY_TMP_FALLBACK_POLICY);
  assert.equal(missing.disable_dev_shm_usage, true);
  assert.equal(missing.selected_path, "/tmp");
  assert.ok(missing.dev_shm_rejections.includes("missing"));

  const tiny = selectBrowser1k5ChromeSharedMemoryPolicy({
    platform: "linux",
    devShm: {
      ...admittedSharedMemoryMeasurement(),
      statfs: {
        available_bytes: BROWSER_1K5_CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES - 1,
      },
      quota_aware_allocation_probe: {
        attempted: false,
        requested_bytes: BROWSER_1K5_CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES,
        written_bytes: 0,
        succeeded: false,
      },
    },
    tempPath: "/tmp",
  });
  assert.equal(tiny.disable_dev_shm_usage, true);
  assert.ok(tiny.dev_shm_rejections.includes("statfs-available-below-minimum"));

  const quotaLimited = selectBrowser1k5ChromeSharedMemoryPolicy({
    platform: "linux",
    devShm: {
      ...admittedSharedMemoryMeasurement(),
      quota_aware_allocation_probe: {
        attempted: true,
        requested_bytes: BROWSER_1K5_CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES,
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

test("uses the platform default shared-memory policy outside Linux", () => {
  const selected = selectBrowser1k5ChromeSharedMemoryPolicy({
    platform: "darwin",
    devShm: null,
    tempPath: "/var/folders/example",
  });
  assert.equal(selected.policy, BROWSER_1K5_CHROME_SHARED_MEMORY_PLATFORM_DEFAULT_POLICY);
  assert.equal(selected.disable_dev_shm_usage, false);
  assert.equal(selected.dev_shm_admitted, null);
});

test("preserves exact 1.5K Chrome launch state and explicit pre-launch nulls", () => {
  const sharedMemory = {
    policy: BROWSER_1K5_CHROME_SHARED_MEMORY_DEV_SHM_POLICY,
    selected_path: "/dev/shm",
  };
  const success = browser1k5ChromeLaunchEvidence({
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
  assert.deepEqual(browser1k5ChromeLaunchEvidence({}), {
    chrome_executable: null,
    chrome_arguments: null,
    chrome_profile: null,
    chrome_shared_memory: null,
  });
});

test("1.5K harness conditionally applies shmem fallback and persists launch evidence", async () => {
  const source = await readFile(harnessScriptPath, "utf8");
  assert.match(source, /statfs\(path, \{ bigint: true \}\)/);
  assert.match(source, /await probeFile\.sync\(\)/);
  assert.match(source, /per-user-or-group-quota-exhausted/);
  assert.match(
    source,
    /if \(sharedMemoryPolicy\.disable_dev_shm_usage\) \{\s*arguments_\.push\("--disable-dev-shm-usage"\);\s*\}/,
  );
  assert.equal(source.match(/"--disable-dev-shm-usage"/g)?.length, 1);
  assert.equal(source.match(/browser1k5ChromeLaunchEvidence\(\{/g)?.length, 3);
  assert.match(source, /if \(outcome\) Object\.assign\(outcome, chromeLaunchEvidence\)/);
  assert.ok(
    source.indexOf("chromeSharedMemory = await inspectChromeSharedMemory()") <
      source.indexOf("browser = await startChrome"),
  );
  assert.ok(
    source.indexOf("chromeArguments_ = chromeLaunchArguments") <
      source.indexOf("browser = await startChrome"),
  );
});

test("attests the exact privacy-redacted non-fallback NVIDIA runtime adapter and device", () => {
  const calls = runtimeWebGpuCalls();
  const summary = summarizeBrowser1k5RuntimeWebGpuCalls(calls);
  assert.equal(summary.adapter_request_attempts, 1);
  assert.equal(summary.adapter_successful_attempts, 1);
  assert.equal(summary.device_successful_attempts, 1);
  assert.equal(summary.is_fallback_adapter, false);
  const attestation = attestBrowser1k5RuntimeAdapter(redactedRuntimeReport(), calls);
  assert.equal(attestation.validated, true, attestation.validation_failures.join("\n"));
});

test("accepts an adapter retry but requires one exact successful adapter and device", () => {
  const calls = [
    { event: "request-adapter-start", detail: { powerPreference: "high-performance" } },
    { event: "request-adapter-rejected", detail: { error: "temporarily unavailable" } },
    ...runtimeWebGpuCalls({ description: "NVIDIA RTX PRO 6000 Blackwell Workstation Edition" }),
  ];
  const report = {
    ...redactedRuntimeReport(),
    adapter_name: "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
  };
  const attestation = attestBrowser1k5RuntimeAdapter(report, calls);
  assert.equal(attestation.adapter_request_attempts, 2);
  assert.equal(attestation.adapter_rejected_attempts, 1);
  assert.equal(attestation.validated, true, attestation.validation_failures.join("\n"));
});

test("fails runtime Other closed on missing, fallback, software, or mismatched exact evidence", () => {
  const report = redactedRuntimeReport();
  const cases = [
    [[], /successful adapter requests/],
    [runtimeWebGpuCalls({ fallback: true }), /not explicitly false/],
    [runtimeWebGpuCalls({ vendor: "swiftshader" }), /not non-software NVIDIA/],
    [runtimeWebGpuCalls({ description: "NVIDIA mismatch" }), /does not match/],
    [runtimeWebGpuCalls().slice(0, -1), /expected exactly one of each/],
  ];
  for (const [calls, expected] of cases) {
    const attestation = attestBrowser1k5RuntimeAdapter(report, calls);
    assert.equal(attestation.validated, false);
    assert.ok(
      attestation.validation_failures.some((failure) => expected.test(failure)),
      `${expected}: ${attestation.validation_failures.join("\n")}`,
    );
  }
});

test("binds browser JavaScript, Wasm, runtime, and harness sources by size and SHA-256", async () => {
  const wwwOutDir = await mkdtemp(join(tmpdir(), "burn-image-1k5-package-identity-"));
  const javascript = Buffer.from("export const browserPackage = true;\n");
  const wasm = Buffer.from([0, 97, 115, 109, 1, 0, 0, 0]);
  try {
    await writeFile(join(wwwOutDir, "bevy_burn_image.js"), javascript);
    await writeFile(join(wwwOutDir, "bevy_burn_image_bg.wasm"), wasm);
    const identity = await collectBrowserPackageIdentity({
      wwwOutDir,
      repoRoot,
      testsDir,
      harnessScriptPath,
    });
    assert.equal(identity.validated, true, identity.validation_failures.join("\n"));
    assert.deepEqual(validateBrowserPackageIdentity(identity), []);
    assert.equal(identity.files.browser_javascript.size_bytes, javascript.length);
    assert.equal(
      identity.files.browser_javascript.sha256,
      createHash("sha256").update(javascript).digest("hex"),
    );
    assert.equal(identity.files.browser_wasm.size_bytes, wasm.length);
    assert.equal(
      identity.files.browser_wasm.sha256,
      createHash("sha256").update(wasm).digest("hex"),
    );
    for (const source of [
      "browser_runtime_source",
      "qualification_harness_script",
      "qualification_harness_page",
      "qualification_contract_source",
      "qualification_scope_source",
    ]) {
      assert.ok(identity.files[source].size_bytes > 0, `${source} has no size`);
      assert.match(identity.files[source].sha256, /^[0-9a-f]{64}$/);
    }
  } finally {
    await rm(wwwOutDir, { recursive: true, force: true });
  }
});

test("fails package identity closed when a browser payload is absent", async () => {
  const wwwOutDir = await mkdtemp(join(tmpdir(), "burn-image-1k5-package-identity-missing-"));
  try {
    await writeFile(join(wwwOutDir, "bevy_burn_image.js"), "export {};\n");
    const identity = await collectBrowserPackageIdentity({
      wwwOutDir,
      repoRoot,
      testsDir,
      harnessScriptPath,
    });
    assert.equal(identity.validated, false);
    assert.ok(
      identity.validation_failures.some((failure) => failure.includes("browser_wasm")),
      identity.validation_failures.join("\n"),
    );
  } finally {
    await rm(wwwOutDir, { recursive: true, force: true });
  }
});
