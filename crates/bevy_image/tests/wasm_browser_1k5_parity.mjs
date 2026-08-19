// Opt-in real-hardware qualification harness. Build the boogu-web Wasm output into
// crates/bevy_image/www/out first, then run with BURN_IMAGE_BROWSER_1K5_PARITY=1.
// The release gate requires BURN_IMAGE_BROWSER_1K5_RESIDENCY=low-vram. The optional non-blocking
// F32 control diagnostic uses BURN_IMAGE_BROWSER_1K5_RESIDENCY=qualification-f32; it does not
// replace the required low-VRAM numerical or memory gate. The resident-packed-f16 selector
// independently qualifies all-stage two-byte weight residency and fused F32 accumulation.
// Set BURN_IMAGE_BROWSER_1K5_VAE_REFERENCE=1 instead for the diagnostic-only compact-fixture
// three-repeat VAE encoder probe; the two workload selectors are mutually exclusive.
// Set BURN_IMAGE_BROWSER_1K5_PARITY_VALIDATE_ONLY=1 to test only the mounted files,
// exact byte-range responses, and CORS contract without launching Chrome or a GPU workload.

import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import {
  constants as fsConstants,
  createReadStream,
  existsSync,
} from "node:fs";
import {
  access,
  mkdir,
  mkdtemp,
  open,
  readdir,
  readFile,
  realpath,
  rm,
  stat,
  statfs,
  writeFile,
} from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import {
  delimiter,
  dirname,
  extname,
  isAbsolute,
  join,
  relative,
  resolve,
  sep,
} from "node:path";
import { fileURLToPath } from "node:url";

import {
  BROWSER_1K5_CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES,
  BROWSER_1K5_F32_QUALIFICATION_DENOISER_RESIDENCY,
  BROWSER_1K5_LOW_VRAM_DENOISER_RESIDENCY,
  BROWSER_1K5_PACKED_F16_DENOISER_RESIDENCY,
  BROWSER_1K5_LOW_VRAM_STRICT_DEVICE_CAP_BYTES,
  attestBrowser1k5RuntimeAdapter,
  browser1k5ChromeLaunchEvidence,
  collectBrowserPackageIdentity,
  denoiserResidencyPolicyForMode,
  selectBrowser1k5ChromeSharedMemoryPolicy,
  validateBrowser1k5LowVramResourcePlan,
  validateBrowser1k5TransportValidation,
  validateDenoiserResidencyPolicy,
} from "./wasm_browser_1k5_contract.mjs";
import { attestCalibratedBrowserWebGpuScope } from "./wasm_browser_1k5_scope.mjs";
import {
  ARTIFACT_TRANSPORT_HARD_MAX_PART_BYTES,
  ARTIFACT_TRANSPORT_LAYOUT_PATH,
  ARTIFACT_TRANSPORT_TARGET_PART_BYTES,
  transportTelemetryFiles,
  validateArtifactBundleTransport,
} from "./artifact_transport_contract.mjs";

const ENABLE_ENV = "BURN_IMAGE_BROWSER_1K5_PARITY";
const VAE_REFERENCE_ENV = "BURN_IMAGE_BROWSER_1K5_VAE_REFERENCE";
const RESIDENCY_ENV = "BURN_IMAGE_BROWSER_1K5_RESIDENCY";
const CHROME_ENV = "BURN_IMAGE_BROWSER_1K5_PARITY_CHROME";
const TIMEOUT_ENV = "BURN_IMAGE_BROWSER_1K5_PARITY_TIMEOUT_MS";
const ARTIFACT_DIR_ENV = "BURN_IMAGE_BROWSER_1K5_PARITY_ARTIFACT_DIR";
const FIXTURE_DIR_ENV = "BURN_IMAGE_BROWSER_1K5_PARITY_FIXTURE_DIR";
const WWW_OUT_DIR_ENV = "BURN_IMAGE_BROWSER_1K5_PARITY_WWW_OUT_DIR";
const OUTPUT_DIR_ENV = "BURN_IMAGE_BROWSER_1K5_PARITY_OUTPUT_DIR";
const HEADFUL_ENV = "BURN_IMAGE_BROWSER_1K5_PARITY_HEADFUL";
const VALIDATE_ONLY_ENV = "BURN_IMAGE_BROWSER_1K5_PARITY_VALIDATE_ONLY";
const DEFAULT_TIMEOUT_MS = 4 * 60 * 60 * 1000;
const DEVTOOLS_START_TIMEOUT_MS = 45_000;
const CDP_CALL_TIMEOUT_MS = 30_000;
const MAX_CAPTURED_CHROME_BYTES = 2 * 1024 * 1024;
const MAX_EVENT_COUNT = 20_000;
const F32_EPSILON = 2 ** -23;
const vaeReferenceMode = process.env[VAE_REFERENCE_ENV] === "1";
const fullParityMode = process.env[ENABLE_ENV] === "1";
if (vaeReferenceMode === fullParityMode) {
  if (!vaeReferenceMode) {
    console.log(
      `burn_image 1.5K browser qualification: skipped (set exactly one of ${ENABLE_ENV}=1 or ${VAE_REFERENCE_ENV}=1)`,
    );
    process.exit(0);
  }
  throw new Error(`${ENABLE_ENV} and ${VAE_REFERENCE_ENV} are mutually exclusive`);
}
const requestedResidencySelector = process.env[RESIDENCY_ENV];
const residencySelector = requestedResidencySelector;
if (vaeReferenceMode && requestedResidencySelector !== undefined) {
  throw new Error(`${RESIDENCY_ENV} is valid only with ${ENABLE_ENV}=1`);
}
if (
  fullParityMode &&
  !["qualification-f32", "low-vram", "resident-packed-f16"].includes(residencySelector)
) {
  throw new Error(
    `${ENABLE_ENV}=1 requires ${RESIDENCY_ENV}=qualification-f32, ${RESIDENCY_ENV}=low-vram, or ${RESIDENCY_ENV}=resident-packed-f16`,
  );
}
const lowVramMode = fullParityMode && residencySelector === "low-vram";
const packedResidentMode = fullParityMode && residencySelector === "resident-packed-f16";
const WORKLOAD_NAME = vaeReferenceMode ? "vae-reference" : "parity";
const WORKLOAD_TEST = vaeReferenceMode
  ? "burn_image_browser_1k5_vae_reference"
  : "burn_image_browser_1k5_parity";
const TERMINAL_OK = vaeReferenceMode
  ? "BURN_IMAGE_HEADLESS_VAE_REFERENCE_OK "
  : "BURN_IMAGE_HEADLESS_PARITY_OK ";
const TERMINAL_FAILED = vaeReferenceMode
  ? "BURN_IMAGE_HEADLESS_VAE_REFERENCE_FAILED"
  : "BURN_IMAGE_HEADLESS_PARITY_FAILED";
const CANONICAL_BUNDLE = "boogu-image-0.1-edit-turbo-1k5";
const CANONICAL_QWEN_BUNDLE = "qwen3-vl-8b-base-boogu-image-0.1";
const CANONICAL_VAE_BUNDLE = "flux1-vae-boogu-image-0.1";
const CANONICAL_MODEL = "Boogu/Boogu-Image-0.1-Edit-Turbo-1K5";
const CANONICAL_MODEL_REVISION = "60981c49e48cffadf2c169532a4ba3f6108afd5e";
const CANONICAL_UPSTREAM_SOURCE_REVISION = "25f8f888298224a94e5ec2abafb98abea9031a0d";
const CANONICAL_PROFILE = "f16-qwen-vision-f32";
const CANONICAL_ARTIFACT_CONTENT_DIGEST =
  "7d81dacfedc71c50639d303c52f035813a6f4cc0125166bd7c8879c8314dd620";
// Provenance of the schema-v1 flat closure used for the calibrated VAE envelope. This is not the
// canonical transport identity; a successful run of this harness requalifies the modular closure.
const LEGACY_FLAT_QUALIFIED_ARTIFACT_CONTENT_DIGEST =
  "5d7e25b1d9be1fdf4a6372bfb9db28cf62ef90253082cef22af09653047e3a7b";
const CANONICAL_ARTIFACT_FILE_COUNT = 253;
const CANONICAL_ARTIFACT_WEIGHT_FILE_COUNT = 223;
const CANONICAL_ARTIFACT_BYTES = 38_224_723_735;
const CANONICAL_MAX_ARTIFACT_FILE_BYTES = 256 * 1024 * 1024;
const BROWSER_1K5_MIN_REQUIRED_BUFFER_LIMIT_BYTES = 1_215_832_064;
const BROWSER_1K5_MIN_RUNTIME_BUFFER_BYTES = 1_215_832_064;
const BROWSER_1K5_QWEN_QUERY_CHUNK_SIZE = 128;
const BROWSER_1K5_VAE_QUERY_CHUNK_SIZE = 4_096;
const BROWSER_1K5_VAE_DECODE_POLICY = "exact-two-width-slabs-global-groupnorm";
const BROWSER_1K5_DENOISER_QUERY_CHUNK_SIZE = 1_024;
const BROWSER_F32_QUALIFICATION_RESIDENCY_POLICY =
  "browser-qualification-per-request-f32-denoiser-retained";
const BROWSER_PACKED_F16_RESIDENCY_POLICY = "browser-high-vram-resident-packed-f16";
const BROWSER_LOW_VRAM_RESIDENCY_POLICY = "browser-low-vram-runtime-q8-denoiser";
const BROWSER_LOW_VRAM_WEIGHT_TRAFFIC_CONTRACT =
  "per-request/qwen+vae+denoiser-first-dmd-step/denoiser-cache-hits-steps-2-through-4";
const BROWSER_F32_QUALIFICATION_WEIGHT_TRAFFIC_CONTRACT =
  "qualification-per-request/qwen+vae+denoiser-first-dmd-step/denoiser-cache-hits-steps-2-through-4";
const BROWSER_PACKED_F16_WEIGHT_TRAFFIC_CONTRACT =
  "eager-preload/qwen+vae+denoiser/resident-f16-weights/fused-f32-accumulate/zero-inference-artifact-transfers/zero-full-stage-widening";
const BROWSER_PACKED_F16_LOAD_POLICY = "packed-f16-weights-f32-auxiliaries";
const BROWSER_PACKED_F16_STORAGE_POLICY =
  "verified-f16-matrix-convolution-buffers/f32-auxiliaries/no-full-stage-widening";
const BROWSER_PACKED_F16_LINEAR_POLICY =
  "resident-f16-storage/integer-unpack/fused-f32-accumulate-matmul";
const BROWSER_1K5_PACKED_F16_RESOURCE_PLAN = Object.freeze({
  weight_storage_policy: "packed-f16-weights-f32-auxiliaries",
  stored_weight_bytes: 38_195_437_952,
  packed_f16_weight_bytes: 34_640_380_224,
  f32_auxiliary_weight_bytes: 2_314_599_372,
  resident_weight_bytes: 36_954_979_596,
  activation_reserve_bytes: 8_297_840_640,
  conservative_planned_device_bytes: 45_252_820_236,
});
const BROWSER_1K5_DENOISER_RETAINED_STAGE_COUNT = 48;
const BROWSER_1K5_QWEN_ALIGNED_STAGE_COUNT = 70;
const BROWSER_1K5_DENOISER_BOUNDARY_COUNT = 236;
const BROWSER_1K5_DMD_STEP_COUNT = 4;
const BROWSER_1K5_VAE_REFERENCE_REPEAT_COUNT = 3;
const BROWSER_WEBGPU_VAE_F32_ORACLE_ENVELOPE = Object.freeze({
  backend: "BrowserWebGpu/raw-cubecl-no-fusion",
  artifact_content_digest: LEGACY_FLAT_QUALIFIED_ARTIFACT_CONTENT_DIGEST,
  artifact_profile: CANONICAL_PROFILE,
  weight_storage_dtype: "f16",
  weight_load_policy: "adapt-to-f32",
  execution_dtype: "f32",
  calibrated_adapter: "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
  calibrated_device: "0x2bb1",
  calibrated_driver: "610.43.02",
  portability: "no-cross-adapter-portability-claim",
  moments: Object.freeze({
    maximum_abs: 0.016,
    maximum_rmse: 0.000_75,
    minimum_cosine_similarity: 0.999_999,
  }),
  mean: Object.freeze({ maximum_abs: 0.013 }),
  logvar: Object.freeze({ maximum_abs: 0.016 }),
  std: Object.freeze({ maximum_abs: 0.000_1 }),
  raw_latent: Object.freeze({ maximum_abs: 0.013 }),
  scaled_latent: Object.freeze({
    maximum_abs: 0.005,
    maximum_rmse: 0.000_2,
    minimum_cosine_similarity: 0.999_999,
  }),
});
const CALIBRATED_BROWSER_WEBGPU_RUNTIME_SCOPE = Object.freeze({
  chrome_product: "Chrome/151.0.7922.108",
  chrome_revision: "@4744b886309d987d292e43232776d2206cccb13d",
  adapter_vendor: "nvidia",
  adapter_architecture: "blackwell",
  adapter_device: BROWSER_WEBGPU_VAE_F32_ORACLE_ENVELOPE.calibrated_device,
  adapter_description: BROWSER_WEBGPU_VAE_F32_ORACLE_ENVELOPE.calibrated_adapter,
  nvidia_driver_version: BROWSER_WEBGPU_VAE_F32_ORACLE_ENVELOPE.calibrated_driver,
});
const FIXTURE_PROMPT =
  "Turn the white circle into a bright orange sun and add two small clouds.";
const FIXTURE_TENSOR_COUNT = 372;
const FIXTURE_FILES = Object.freeze({
  metadata: Object.freeze({
    path: "metadata.json",
    size: 93_942,
    sha256: "1e78233c703ed32ee351c25d54ca4b05e3efeb898ee2836d1cc96c522e2abcae",
  }),
  tensors: Object.freeze({
    path: "tensors.safetensors",
    size: 11_258_528_368,
    sha256: "2585ddf2337e41f884218a4abeceb8a10baa7553e43d37f33016be68edc3eeb9",
  }),
  source: Object.freeze({
    path: "source.png",
    size: 6_367,
    sha256: "96534b93904478caf92c1d0e1b431396f81e7b62f09bb5505443378f245d9647",
  }),
  output: Object.freeze({
    path: "output.png",
    size: 1_803_055,
    sha256: "8e88d6c3580593da723049ef4027a60c5d730b6006ef766d49971a23c6446a70",
  }),
});
const VAE_REFERENCE_FIXTURE_FILES = Object.freeze({
  metadata: Object.freeze({
    path: "metadata.json",
    size: 19_604,
    sha256: "4a3847347adefd38f5978844f311b934606f9b8a6be0013235dd1fcaf5393ebb",
  }),
  tensors: Object.freeze({
    path: "tensors.safetensors",
    size: 49_120_176,
    sha256: "bdd429af5b8f146fea3ac05238cd1d711d3be7f974dc54544ae85c149874a2df",
  }),
  source: FIXTURE_FILES.source,
  output: Object.freeze({
    path: "output.png",
    size: 1_799_947,
    sha256: "f6d8e1b45351bfe203136da075b43afaf6f80c9eda481f529bc6707eb91787bc",
  }),
});
const ACTIVE_FIXTURE_FILES = vaeReferenceMode
  ? VAE_REFERENCE_FIXTURE_FILES
  : FIXTURE_FILES;
const ACTIVE_FIXTURE_TENSOR_COUNT = vaeReferenceMode ? 47 : FIXTURE_TENSOR_COUNT;
const GPU_SAMPLE_INTERVAL_MS = vaeReferenceMode ? 1_000 : 2_000;
const HOST_RESOURCE_SAMPLE_INTERVAL_MS = vaeReferenceMode ? 1_000 : 5_000;
const MIN_GPU_MATCHED_SAMPLE_INTERVALS = vaeReferenceMode ? 2 : 3;
const MIN_GPU_ACTIVE_SAMPLE_INTERVALS = vaeReferenceMode ? 2 : 3;
const MIN_GPU_CONSECUTIVE_ACTIVE_SAMPLE_INTERVALS = vaeReferenceMode ? 1 : 2;
const MIN_F32_RETAINED_DENOISER_FRAMEBUFFER_MIB = 30 * 1024;
const MAX_RECORDED_GPU_SAMPLES = 4_096;
const MAX_RECORDED_HOST_RESOURCE_SAMPLES = 512;
const MAX_TRACKED_MILESTONES = 1_024;
const MAX_TRACKED_HTTP_AGGREGATES = 4_096;
const MAX_TRACKED_CHROME_PROCESSES = 256;
const MAX_RECORDED_MONITOR_ERRORS = 20;
const MAX_CAPTURED_MONITOR_BYTES = 64 * 1024;
const MAX_RECORDED_BROWSER_ERRORS = 512;
const MAX_CAPTURED_PROCESS_COMMAND_CHARS = 4_096;
const CHROME_SHARED_MEMORY_PROBE_CHUNK_BYTES = 8 * 1024 * 1024;

if (typeof WebSocket === "undefined") {
  throw new Error("this harness requires Node 22 or newer (global WebSocket is unavailable)");
}

let interruptedSignal;
const signalHandlers = new Map();
for (const signal of ["SIGINT", "SIGTERM"]) {
  const handler = () => {
    interruptedSignal ??= signal;
  };
  signalHandlers.set(signal, handler);
  process.once(signal, handler);
}

function throwIfInterrupted() {
  if (interruptedSignal) throw new Error(`interrupted by ${interruptedSignal}`);
}

const scriptPath = fileURLToPath(import.meta.url);
const testsDir = dirname(scriptPath);
const repoRoot = resolve(testsDir, "../../..");
const harnessFileName = "wasm_browser_1k5_parity.html";

function parseTimeout() {
  const source = process.env[TIMEOUT_ENV];
  if (source === undefined) return DEFAULT_TIMEOUT_MS;
  if (!/^\d+$/.test(source)) {
    throw new Error(`${TIMEOUT_ENV} must be a positive integer, got ${JSON.stringify(source)}`);
  }
  const value = Number(source);
  if (!Number.isSafeInteger(value) || value < 1_000) {
    throw new Error(`${TIMEOUT_ENV} must be a safe integer of at least 1000 milliseconds`);
  }
  return value;
}

const delay = (milliseconds) =>
  new Promise((resolveDelay) => setTimeout(resolveDelay, milliseconds));

function isWithinRoot(root, candidate) {
  const pathFromRoot = relative(root, candidate);
  return pathFromRoot === "" || (!pathFromRoot.startsWith(`..${sep}`) && pathFromRoot !== "..");
}

function contentType(path) {
  switch (extname(path).toLowerCase()) {
    case ".html":
      return "text/html; charset=utf-8";
    case ".js":
    case ".mjs":
      return "text/javascript; charset=utf-8";
    case ".json":
      return "application/json; charset=utf-8";
    case ".wasm":
      return "application/wasm";
    case ".png":
      return "image/png";
    case ".safetensors":
      return "application/octet-stream";
    default:
      return "application/octet-stream";
  }
}

function parseRange(header, size) {
  if (!header?.startsWith("bytes=") || header.includes(",")) return undefined;
  const match = /^bytes=(\d*)-(\d*)$/.exec(header);
  if (!match || (match[1] === "" && match[2] === "")) return undefined;

  let start;
  let end;
  if (match[1] === "") {
    const suffixLength = Number(match[2]);
    if (!Number.isSafeInteger(suffixLength) || suffixLength <= 0 || size === 0) return undefined;
    start = Math.max(0, size - suffixLength);
    end = size - 1;
  } else {
    start = Number(match[1]);
    end = match[2] === "" ? size - 1 : Number(match[2]);
    if (
      !Number.isSafeInteger(start) ||
      !Number.isSafeInteger(end) ||
      start < 0 ||
      end < start ||
      start >= size
    ) {
      return undefined;
    }
    end = Math.min(end, size - 1);
  }
  return { start, end };
}

function corsHeaders() {
  return {
    "Access-Control-Allow-Headers": "Content-Type, Range",
    "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Expose-Headers": "Accept-Ranges, Content-Length, Content-Range",
    "Cross-Origin-Resource-Policy": "cross-origin",
  };
}

function emptyHttpCounters() {
  return {
    requests: 0,
    successful_get_requests: 0,
    ranged_requests: 0,
    response_content_length_bytes: 0,
  };
}

function incrementHttpCounters(counters, event) {
  counters.requests += 1;
  if (event.method === "GET" && (event.status === 200 || event.status === 206)) {
    counters.successful_get_requests += 1;
    counters.response_content_length_bytes += event.response_content_length_bytes;
  }
  if (event.range !== null) counters.ranged_requests += 1;
}

function requestRoute(url) {
  let pathname;
  try {
    pathname = new URL(url ?? "/", "http://localhost").pathname;
  } catch {
    return "invalid";
  }
  for (const route of [
    "artifacts",
    CANONICAL_QWEN_BUNDLE,
    CANONICAL_VAE_BUNDLE,
    "fixture",
    "app",
    "harness",
  ]) {
    if (pathname === `/${route}` || pathname.startsWith(`/${route}/`)) return route;
  }
  if (pathname === "/probe") return "probe";
  return "other";
}

function isArtifactRoute(route) {
  return ["artifacts", CANONICAL_QWEN_BUNDLE, CANONICAL_VAE_BUNDLE].includes(route);
}

function isIgnoredBrowserResource(url) {
  if (typeof url !== "string" || url.length === 0) return false;
  try {
    return new URL(url, "http://localhost").pathname === "/favicon.ico";
  } catch {
    return false;
  }
}

function isFatalConsoleMessage(type, message) {
  if (!["error", "assert"].includes(type)) return false;
  return (
    /\bpanicked at\b|\bRuntimeError\b|\bwasm trap\b|\bunreachable\b/i.test(message) ||
    /\b(?:webgpu|wgpu|dawn|vulkan|gpu(?:device)?)\b.*\b(?:error|failed|lost|invalid)\b/i.test(
      message,
    )
  );
}

function createWorkloadTelemetry(denoiserResidencyPolicy) {
  let currentParityMilestone = "pre-parity";
  const milestoneCounts = new Map();
  const byRoute = new Map();
  const byParityMilestone = new Map();
  const denoiserByParityMilestone = new Map();
  const artifactComponentsByPath = new Map();
  const totals = emptyHttpCounters();
  let droppedMilestoneNames = 0;
  let droppedHttpAggregateKeys = 0;

  const countersFor = (map, key) => {
    let counters = map.get(key);
    if (!counters) {
      if (map.size >= MAX_TRACKED_HTTP_AGGREGATES) {
        droppedHttpAggregateKeys += 1;
        return null;
      }
      counters = emptyHttpCounters();
      map.set(key, counters);
    }
    return counters;
  };
  const recordMilestone = (kind, name, atMs) => {
    const key = `${kind}:${name}`;
    const observed = milestoneCounts.get(key);
    if (observed) {
      observed.count += 1;
      observed.last_at_ms = atMs;
    } else if (milestoneCounts.size < MAX_TRACKED_MILESTONES) {
      milestoneCounts.set(key, {
        kind,
        name,
        count: 1,
        first_at_ms: atMs,
        last_at_ms: atMs,
      });
    } else {
      droppedMilestoneNames += 1;
    }
  };

  return {
    resetForWorkload() {
      currentParityMilestone = "pre-parity";
      milestoneCounts.clear();
      byRoute.clear();
      byParityMilestone.clear();
      denoiserByParityMilestone.clear();
      for (const field of Object.keys(totals)) totals[field] = 0;
      droppedMilestoneNames = 0;
      droppedHttpAggregateKeys = 0;
    },
    configureArtifactInventory(bundles) {
      artifactComponentsByPath.clear();
      for (const { prefix, bundle, files } of bundles) {
        for (const file of files) {
          if (typeof file?.path !== "string" || typeof file?.component !== "string") continue;
          artifactComponentsByPath.set(`/${prefix}/${file.path}`, {
            bundle,
            component: file.component,
            components: [...(file.components ?? [file.component])].sort(),
            logical_paths: file.logical_paths ?? [file.logical_path ?? file.path],
            shared_physical_part: file.shared_physical_part === true,
            physical_transport_part: file.physical_transport_part === true,
          });
        }
      }
    },
    captureParityMilestone() {
      return currentParityMilestone;
    },
    noteConsoleMessage(message, atMs = Date.now()) {
      for (const [prefix, kind] of [
        ["BURN_IMAGE_HEADLESS_PARITY_PROGRESS ", "parity"],
        ["BURN_IMAGE_HEADLESS_VAE_REFERENCE_PROGRESS ", "parity"],
        ["BURN_IMAGE_BROWSER_STAGE_MILESTONE ", "browser-stage"],
        ["BURN_IMAGE_HEADLESS_INFER_PROGRESS ", "inference"],
      ]) {
        if (!message.startsWith(prefix)) continue;
        const name = message.slice(prefix.length).trim();
        if (name.length === 0) return;
        recordMilestone(kind, name, atMs);
        if (kind === "parity") currentParityMilestone = name;
        return;
      }
    },
    recordHttp(event) {
      const route = requestRoute(event.url);
      let pathname = "";
      try {
        pathname = new URL(event.url ?? "/", "http://localhost").pathname;
      } catch {
        // requestRoute already records this as invalid.
      }
      const artifactIdentity = artifactComponentsByPath.get(pathname) ?? null;
      event.artifact_bundle = artifactIdentity?.bundle ?? null;
      event.artifact_component = artifactIdentity?.component ?? null;
      event.artifact_components = artifactIdentity?.components ?? [];
      event.artifact_logical_paths = artifactIdentity?.logical_paths ?? [];
      event.shared_physical_part = artifactIdentity?.shared_physical_part ?? false;
      event.physical_transport_part = artifactIdentity?.physical_transport_part ?? false;
      incrementHttpCounters(totals, event);
      const routeCounters = countersFor(byRoute, route);
      if (routeCounters) incrementHttpCounters(routeCounters, event);
      const milestoneCounters = countersFor(
        byParityMilestone,
        `${event.parity_milestone}|${route}`,
      );
      if (milestoneCounters) incrementHttpCounters(milestoneCounters, event);
      if (event.artifact_components.some((component) => component.startsWith("boogu-"))) {
        const denoiserCounters = countersFor(
          denoiserByParityMilestone,
          event.parity_milestone,
        );
        if (denoiserCounters) incrementHttpCounters(denoiserCounters, event);
      }
    },
    snapshot() {
      const phases = {};
      for (const [key, counters] of byParityMilestone) {
        const separator = key.lastIndexOf("|");
        const phase = key.slice(0, separator);
        const route = key.slice(separator + 1);
        phases[phase] ??= {};
        phases[phase][route] = { ...counters };
      }
      return {
        current_parity_milestone: currentParityMilestone,
        totals: { ...totals },
        by_route: Object.fromEntries(
          Array.from(byRoute, ([route, counters]) => [route, { ...counters }]).sort(),
        ),
        by_parity_milestone: phases,
        boogu_denoiser_by_parity_milestone: Object.fromEntries(
          Array.from(denoiserByParityMilestone, ([phase, counters]) => [
            phase,
            { ...counters },
          ]).sort(),
        ),
        milestones: Array.from(milestoneCounts.values()).sort(
          (left, right) => left.first_at_ms - right.first_at_ms || left.name.localeCompare(right.name),
        ),
        tracked_milestone_names: milestoneCounts.size,
        dropped_milestone_names: droppedMilestoneNames,
        dropped_http_aggregate_keys: droppedHttpAggregateKeys,
      };
    },
    denoiserResidencyAttestation() {
      const snapshot = this.snapshot();
      const failures = [];
      const milestone = (name) =>
        snapshot.milestones.find((entry) => entry.kind === "parity" && entry.name === name);
      const artifactCounters = (phase) => {
        const counters = emptyHttpCounters();
        for (const [route, observed] of Object.entries(
          snapshot.by_parity_milestone[phase] ?? {},
        )) {
          if (!isArtifactRoute(route)) continue;
          for (const field of Object.keys(counters)) counters[field] += observed[field];
        }
        return counters;
      };
      const denoiserCounters = (phase) =>
        snapshot.boogu_denoiser_by_parity_milestone[phase] ?? emptyHttpCounters();
      const required = [
        "dmd-step-0-start",
        "dmd-step-1-start",
        "dmd-step-2-start",
        "dmd-step-3-start",
        "dmd-resident-denoiser-cache-cleared",
      ];
      for (const name of required) {
        if (milestone(name)?.count !== 1) {
          failures.push(`parity milestone ${name} was not observed exactly once`);
        }
      }
      const packedResident =
        denoiserResidencyPolicy === BROWSER_1K5_PACKED_F16_DENOISER_RESIDENCY;
      const preload = denoiserCounters("pre-parity");
      const step0 = denoiserCounters("dmd-step-0-start");
      if (packedResident) {
        if (preload.successful_get_requests === 0 || preload.response_content_length_bytes === 0) {
          failures.push("packed-F16 resident preload did not visibly fetch denoiser artifacts");
        }
      } else if (
        step0.successful_get_requests === 0 ||
        step0.response_content_length_bytes === 0
      ) {
        failures.push("DMD step 0 did not visibly fetch the resident denoiser artifacts");
      }
      for (const index of packedResident ? [0, 1, 2, 3] : [1, 2, 3]) {
        const counters = denoiserCounters(`dmd-step-${index}-start`);
        if (counters.requests !== 0 || counters.response_content_length_bytes !== 0) {
          failures.push(
            `DMD step ${index} issued ${counters.requests} Boogu denoiser requests totaling ${counters.response_content_length_bytes} response bytes after the ${packedResident ? "eager-preload" : "first-pass"} residency boundary`,
          );
        }
        const allArtifacts = artifactCounters(`dmd-step-${index}-start`);
        if (allArtifacts.requests !== 0 || allArtifacts.response_content_length_bytes !== 0) {
          failures.push(
            `DMD step ${index} unexpectedly issued ${allArtifacts.requests} total model-artifact requests totaling ${allArtifacts.response_content_length_bytes} response bytes`,
          );
        }
      }
      if (snapshot.dropped_milestone_names !== 0) {
        failures.push(
          `${snapshot.dropped_milestone_names} runtime milestone names exceeded the bounded telemetry inventory`,
        );
      }
      if (snapshot.dropped_http_aggregate_keys !== 0) {
        failures.push(
          `${snapshot.dropped_http_aggregate_keys} HTTP aggregate keys exceeded the bounded telemetry inventory`,
        );
      }
      return {
        policy: denoiserResidencyPolicy,
        first_pass: { ...(packedResident ? preload : step0) },
        reused_steps: (packedResident ? [0, 1, 2, 3] : [1, 2, 3]).map((index) => ({
          step: index,
          ...denoiserCounters(`dmd-step-${index}-start`),
        })),
        validation_failures: failures,
        validated: failures.length === 0,
        telemetry: snapshot,
      };
    },
  };
}

async function canonicalMount(prefix, root) {
  const canonicalRoot = await realpath(root);
  const metadata = await stat(canonicalRoot);
  if (!metadata.isDirectory()) throw new Error(`server mount is not a directory: ${canonicalRoot}`);
  return { prefix, root: canonicalRoot };
}

async function createStaticServer(mountInputs, requestEvents, workloadTelemetry) {
  const mounts = await Promise.all(
    mountInputs.map(({ prefix, root }) => canonicalMount(prefix, root)),
  );
  mounts.sort((left, right) => right.prefix.length - left.prefix.length);

  const server = createServer(async (request, response) => {
    const startedAt = Date.now();
    const parityMilestone = workloadTelemetry.captureParityMilestone();
    let statusCode = 500;
    let responseContentLength = 0;
    try {
      if (request.method === "OPTIONS") {
        statusCode = 204;
        response.writeHead(statusCode, corsHeaders());
        response.end();
        return;
      }
      if (request.method !== "GET" && request.method !== "HEAD") {
        statusCode = 405;
        response.writeHead(statusCode, { ...corsHeaders(), Allow: "GET, HEAD, OPTIONS" });
        response.end("method not allowed\n");
        return;
      }

      let pathname;
      try {
        pathname = decodeURIComponent(new URL(request.url ?? "/", "http://localhost").pathname);
      } catch {
        statusCode = 400;
        response.writeHead(statusCode, corsHeaders());
        response.end("invalid URL\n");
        return;
      }
      if (pathname.includes("\0")) {
        statusCode = 400;
        response.writeHead(statusCode, corsHeaders());
        response.end("invalid path\n");
        return;
      }

      if (pathname === "/probe") {
        const body = Buffer.from(
          "<!doctype html><meta charset=utf-8><title>burn-image-webgpu-probe</title>",
        );
        statusCode = 200;
        response.writeHead(statusCode, {
          ...corsHeaders(),
          "Cache-Control": "no-store",
          "Content-Length": body.length,
          "Content-Type": "text/html; charset=utf-8",
          "Cross-Origin-Embedder-Policy": "require-corp",
          "Cross-Origin-Opener-Policy": "same-origin",
        });
        if (request.method === "HEAD") response.end();
        else response.end(body);
        return;
      }

      const mount = mounts.find(
        ({ prefix }) => pathname === prefix.slice(0, -1) || pathname.startsWith(prefix),
      );
      if (!mount) {
        statusCode = 404;
        response.writeHead(statusCode, corsHeaders());
        response.end("not found\n");
        return;
      }
      const mountedPath = pathname === mount.prefix.slice(0, -1)
        ? ""
        : pathname.slice(mount.prefix.length);
      const candidate = resolve(mount.root, mountedPath);
      if (!isWithinRoot(mount.root, candidate)) {
        statusCode = 403;
        response.writeHead(statusCode, corsHeaders());
        response.end("forbidden\n");
        return;
      }

      let canonicalPath;
      let metadata;
      try {
        canonicalPath = await realpath(candidate);
        if (!isWithinRoot(mount.root, canonicalPath)) throw new Error("path escapes mount root");
        metadata = await stat(canonicalPath);
      } catch {
        statusCode = 404;
        response.writeHead(statusCode, corsHeaders());
        response.end("not found\n");
        return;
      }
      if (!metadata.isFile()) {
        statusCode = 404;
        response.writeHead(statusCode, corsHeaders());
        response.end("not found\n");
        return;
      }

      const baseHeaders = {
        ...corsHeaders(),
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-store",
        "Content-Type": contentType(canonicalPath),
        "Cross-Origin-Embedder-Policy": "require-corp",
        "Cross-Origin-Opener-Policy": "same-origin",
      };
      const rangeHeader = request.headers.range;
      const range = rangeHeader === undefined ? undefined : parseRange(rangeHeader, metadata.size);
      if (rangeHeader !== undefined && !range) {
        statusCode = 416;
        response.writeHead(statusCode, {
          ...baseHeaders,
          "Content-Range": `bytes */${metadata.size}`,
        });
        response.end();
        return;
      }

      const start = range?.start ?? 0;
      const end = range?.end ?? metadata.size - 1;
      const contentLength = metadata.size === 0 ? 0 : end - start + 1;
      responseContentLength = request.method === "GET" ? contentLength : 0;
      statusCode = range ? 206 : 200;
      response.writeHead(statusCode, {
        ...baseHeaders,
        "Content-Length": contentLength,
        ...(range ? { "Content-Range": `bytes ${start}-${end}/${metadata.size}` } : {}),
      });
      if (request.method === "HEAD" || contentLength === 0) {
        response.end();
        return;
      }
      const stream = createReadStream(canonicalPath, { start, end });
      stream.on("error", (error) => response.destroy(error));
      stream.pipe(response);
    } catch (error) {
      if (!response.headersSent) response.writeHead(500, corsHeaders());
      response.end(`server error: ${error instanceof Error ? error.message : String(error)}\n`);
    } finally {
      const event = {
        at_ms: startedAt,
        observed_at_ms: Date.now(),
        duration_ms: Date.now() - startedAt,
        method: request.method,
        url: request.url,
        range: request.headers.range ?? null,
        status: statusCode,
        response_content_length_bytes: responseContentLength,
        parity_milestone: parityMilestone,
      };
      workloadTelemetry.recordHttp(event);
      if (requestEvents.length < MAX_EVENT_COUNT) {
        requestEvents.push(event);
      }
    }
  });

  await new Promise((resolveListen, rejectListen) => {
    server.once("error", rejectListen);
    server.listen(0, "127.0.0.1", () => {
      server.off("error", rejectListen);
      resolveListen();
    });
  });
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("static server has no TCP address");
  return { server, port: address.port };
}

async function closeServer(server) {
  if (!server) return;
  const closed = new Promise((resolveClose) => server.close(resolveClose));
  server.closeAllConnections?.();
  await closed;
}

async function assertResponse(response, expectedStatus, expectedUrl) {
  if (response.status !== expectedStatus) {
    throw new Error(`${expectedUrl} returned HTTP ${response.status}, expected ${expectedStatus}`);
  }
  if (response.headers.get("access-control-allow-origin") !== "*") {
    throw new Error(`${expectedUrl} is missing Access-Control-Allow-Origin: *`);
  }
  return response;
}

async function validateRangeResponse(url, size, localPath) {
  if (size < 8) throw new Error(`range validation target is too small: ${url} (${size} bytes)`);
  const response = await assertResponse(
    await fetch(url, {
      headers: { Origin: "https://browser-parity.invalid", Range: "bytes=1-7" },
    }),
    206,
    url,
  );
  const expectedContentRange = `bytes 1-7/${size}`;
  if (response.headers.get("content-range") !== expectedContentRange) {
    throw new Error(
      `${url} returned Content-Range ${response.headers.get("content-range")}, expected ${expectedContentRange}`,
    );
  }
  if (response.headers.get("accept-ranges") !== "bytes") {
    throw new Error(`${url} is missing Accept-Ranges: bytes`);
  }
  if (response.headers.get("content-length") !== "7") {
    throw new Error(`${url} returned an incorrect byte-range Content-Length`);
  }
  const responseBytes = new Uint8Array(await response.arrayBuffer());
  if (responseBytes.byteLength !== 7) {
    throw new Error(`${url} returned an incorrect byte-range body length`);
  }
  const expectedBytes = Buffer.alloc(7);
  const file = await open(localPath, "r");
  try {
    const read = await file.read(expectedBytes, 0, expectedBytes.length, 1);
    if (read.bytesRead !== expectedBytes.length) {
      throw new Error(`${localPath} could not provide the expected local byte range`);
    }
  } finally {
    await file.close();
  }
  if (!Buffer.from(responseBytes).equals(expectedBytes)) {
    throw new Error(`${url} returned bytes that differ from the mounted local file`);
  }

  const unsatisfiable = await fetch(url, {
    headers: { Origin: "https://browser-parity.invalid", Range: `bytes=${size}-` },
  });
  await assertResponse(unsatisfiable, 416, url);
  if (unsatisfiable.headers.get("content-range") !== `bytes */${size}`) {
    throw new Error(`${url} returned an incorrect unsatisfiable Content-Range`);
  }
}

async function validateLocalFileIdentity(root, identity) {
  const path = join(root, identity.path);
  const metadata = await stat(path);
  if (!metadata.isFile() || metadata.size !== identity.size) {
    throw new Error(
      `${path} has size ${metadata.size}, expected the pinned ${identity.size}-byte file`,
    );
  }
  const digest = createHash("sha256").update(await readFile(path)).digest("hex");
  if (digest !== identity.sha256) {
    throw new Error(`${path} has SHA-256 ${digest}, expected ${identity.sha256}`);
  }
}

async function readSafeTensorsLayout(path, expectedSize) {
  const metadata = await stat(path);
  if (!metadata.isFile() || metadata.size !== expectedSize) {
    throw new Error(`${path} has size ${metadata.size}, expected ${expectedSize}`);
  }
  const prefix = Buffer.alloc(8);
  const file = await open(path, "r");
  try {
    const read = await file.read(prefix, 0, prefix.length, 0);
    if (read.bytesRead !== prefix.length) throw new Error(`${path} has no SafeTensors header`);
  } finally {
    await file.close();
  }
  const headerLength = Number(prefix.readBigUInt64LE());
  if (!Number.isSafeInteger(headerLength) || headerLength < 2 || headerLength > 4 * 1024 * 1024) {
    throw new Error(`${path} has invalid SafeTensors header length ${headerLength}`);
  }
  const dataStart = 8 + headerLength;
  if (dataStart >= expectedSize) throw new Error(`${path} has an empty SafeTensors data section`);
  return {
    header_bytes: dataStart,
    tensor_bytes: expectedSize - dataStart,
  };
}

async function validateArtifactInventory(manifest, artifactDir) {
  const validation = await validateArtifactBundleTransport({
    bundleRoot: artifactDir,
    manifest,
  });
  if (validation.logical.max_file_bytes > CANONICAL_MAX_ARTIFACT_FILE_BYTES) {
    throw new Error(
      `canonical logical artifact inventory has ${validation.logical.max_file_bytes}-byte object above ${CANONICAL_MAX_ARTIFACT_FILE_BYTES}`,
    );
  }
  if (
    validation.transport.target_part_bytes !== ARTIFACT_TRANSPORT_TARGET_PART_BYTES ||
    validation.transport.hard_max_part_bytes !== ARTIFACT_TRANSPORT_HARD_MAX_PART_BYTES ||
    validation.transport.max_part_bytes > ARTIFACT_TRANSPORT_HARD_MAX_PART_BYTES
  ) {
    throw new Error("canonical physical transport does not enforce exact 20 MiB / 25000000-byte bounds");
  }
  return validation;
}

async function validateServer(baseUrl, artifactDir, fixtureDir, workloadTelemetry) {
  const fixtureFiles = ACTIVE_FIXTURE_FILES;
  for (const [route, expectedType] of [
    ["/harness/wasm_browser_1k5_parity.html", "text/html"],
    ["/app/out/bevy_burn_image.js", "text/javascript"],
    ["/app/out/bevy_burn_image_bg.wasm", "application/wasm"],
    ["/app/out/burn-image-icon.png", "image/png"],
  ]) {
    const url = `${baseUrl}${route}`;
    const response = await assertResponse(
      await fetch(url, { method: "HEAD", headers: { Origin: "https://browser-parity.invalid" } }),
      200,
      url,
    );
    if (!(response.headers.get("content-type") ?? "").startsWith(expectedType)) {
      throw new Error(
        `${url} returned unexpected Content-Type ${response.headers.get("content-type")}`,
      );
    }
    if (response.headers.get("accept-ranges") !== "bytes") {
      throw new Error(`${url} is missing Accept-Ranges: bytes`);
    }
  }

  const manifestUrl = `${baseUrl}/artifacts/manifest.json`;
  const manifestResponse = await assertResponse(
    await fetch(manifestUrl, { headers: { Origin: "https://browser-parity.invalid" } }),
    200,
    manifestUrl,
  );
  const manifest = await manifestResponse.json();
  if (!Array.isArray(manifest.files) || manifest.files.length === 0) {
    throw new Error("canonical artifact manifest has no files inventory");
  }
  if (
    manifest.schema_version !== 2 ||
    manifest.bundle !== CANONICAL_BUNDLE ||
    manifest.model !== CANONICAL_MODEL ||
    manifest.model_revision !== CANONICAL_MODEL_REVISION ||
    manifest.profile !== CANONICAL_PROFILE ||
    manifest.content_digest !== CANONICAL_ARTIFACT_CONTENT_DIGEST ||
    manifest.metadata?.source_revision !== CANONICAL_UPSTREAM_SOURCE_REVISION ||
    manifest.numeric_format?.other !== CANONICAL_PROFILE ||
    manifest.metadata?.artifact_layout !== "semantic-burnpack-composition-v2" ||
    manifest.metadata?.component_dependency_count !== "2" ||
    !Array.isArray(manifest.dependencies) ||
    manifest.dependencies.length !== 2
  ) {
    throw new Error(
      `artifact directory is not the canonical production 1.5K bundle: ${JSON.stringify({
        bundle: manifest.bundle,
        model: manifest.model,
        profile: manifest.profile,
        content_digest: manifest.content_digest,
      })}`,
    );
  }

  const dependencyByRole = new Map(manifest.dependencies.map((entry) => [entry.role, entry]));
  const qwenDependency = dependencyByRole.get("qwen");
  const vaeDependency = dependencyByRole.get("vae");
  if (
    dependencyByRole.size !== 2 ||
    qwenDependency?.bundle !== CANONICAL_QWEN_BUNDLE ||
    vaeDependency?.bundle !== CANONICAL_VAE_BUNDLE
  ) {
    throw new Error(
      `canonical composition has invalid dependency roles: ${JSON.stringify(manifest.dependencies)}`,
    );
  }

  const modelRoot = dirname(artifactDir);
  const dependencies = [];
  for (const dependency of [qwenDependency, vaeDependency]) {
    const dependencyDir = join(modelRoot, dependency.bundle);
    const dependencyUrl = `${baseUrl}/${dependency.bundle}/manifest.json`;
    const response = await assertResponse(
      await fetch(dependencyUrl, { headers: { Origin: "https://browser-parity.invalid" } }),
      200,
      dependencyUrl,
    );
    const child = await response.json();
    if (
      child.schema_version !== 1 ||
      child.bundle !== dependency.bundle ||
      child.profile !== dependency.profile ||
      child.model !== dependency.model ||
      child.model_revision !== dependency.model_revision ||
      child.content_digest !== dependency.content_digest ||
      !Array.isArray(child.files) ||
      child.files.length === 0 ||
      (child.dependencies?.length ?? 0) !== 0
    ) {
      throw new Error(
        `resolved ${dependency.role} manifest differs from its sealed parent reference: ${JSON.stringify({
          expected: dependency,
          actual: {
            bundle: child.bundle,
            profile: child.profile,
            model: child.model,
            model_revision: child.model_revision,
            content_digest: child.content_digest,
          },
        })}`,
      );
    }
    dependencies.push({ dependency, manifest: child, dir: dependencyDir });
  }

  const parentInventory = await validateArtifactInventory(manifest, artifactDir);
  const dependencyInventories = await Promise.all(
    dependencies.map(({ manifest: child, dir }) => validateArtifactInventory(child, dir)),
  );
  const validatedBundles = [
    {
      prefix: "artifacts",
      bundle: manifest.bundle,
      manifest,
      dir: artifactDir,
      validation: parentInventory,
    },
    ...dependencies.map(({ dependency, manifest: child, dir }, index) => ({
      prefix: dependency.bundle,
      bundle: dependency.bundle,
      manifest: child,
      dir,
      validation: dependencyInventories[index],
    })),
  ];
  workloadTelemetry.configureArtifactInventory(
    validatedBundles.map(({ prefix, bundle, validation }) => ({
      prefix,
      bundle,
      files: transportTelemetryFiles(validation),
    })),
  );
  const artifactInventory = [parentInventory, ...dependencyInventories].reduce(
    (total, inventory) => ({
      files: total.files + inventory.logical.file_count,
      totalBytes: total.totalBytes + inventory.logical.bytes,
      maximumBytes: Math.max(total.maximumBytes, inventory.logical.max_file_bytes),
      weightFiles: total.weightFiles + inventory.logical.weight_file_count,
    }),
    { files: 0, totalBytes: 0, maximumBytes: 0, weightFiles: 0 },
  );
  const physicalTransportInventory = [parentInventory, ...dependencyInventories].reduce(
    (total, inventory) => ({
      partReferences: total.partReferences + inventory.transport.part_reference_count,
      uniqueParts: total.uniqueParts + inventory.transport.unique_part_count,
      reconstructedBytes: total.reconstructedBytes + inventory.transport.reconstructed_bytes,
      uniquePartBytes: total.uniquePartBytes + inventory.transport.unique_part_bytes,
      maximumPartBytes: Math.max(
        total.maximumPartBytes,
        inventory.transport.max_part_bytes,
      ),
      directFiles: total.directFiles + inventory.direct.file_count,
      directBytes: total.directBytes + inventory.direct.bytes,
    }),
    {
      partReferences: 0,
      uniqueParts: 0,
      reconstructedBytes: 0,
      uniquePartBytes: 0,
      maximumPartBytes: 0,
      directFiles: 0,
      directBytes: 0,
    },
  );
  if (
    artifactInventory.files !== CANONICAL_ARTIFACT_FILE_COUNT ||
    artifactInventory.totalBytes !== CANONICAL_ARTIFACT_BYTES ||
    artifactInventory.weightFiles !== CANONICAL_ARTIFACT_WEIGHT_FILE_COUNT
  ) {
    throw new Error(
      `canonical modular artifact closure differs from the pinned release: ${JSON.stringify(artifactInventory)}`,
    );
  }
  const logicalWeightBytes = [parentInventory, ...dependencyInventories].reduce(
    (total, inventory) => total + inventory.logical.weight_bytes,
    0,
  );
  if (
    physicalTransportInventory.reconstructedBytes !== logicalWeightBytes ||
    physicalTransportInventory.maximumPartBytes > ARTIFACT_TRANSPORT_HARD_MAX_PART_BYTES
  ) {
    throw new Error(
      `physical transport does not exactly cover the logical weight closure: ${JSON.stringify({ logicalWeightBytes, physicalTransportInventory })}`,
    );
  }
  const vaeManifest = dependencies.find(({ dependency }) => dependency.role === "vae");
  const vaeEncoderWeightFiles = vaeManifest.manifest.files.filter(
    (entry) => entry.role === "weights" && entry.component === "flux-vae-encoder",
  );
  const vaeEncoderWeightBytes = vaeEncoderWeightFiles.reduce(
    (total, entry) => total + entry.size,
    0,
  );
  if (vaeEncoderWeightFiles.length === 0 || !Number.isSafeInteger(vaeEncoderWeightBytes)) {
    throw new Error("canonical artifact manifest has no bounded FLUX VAE encoder stage");
  }
  for (const { prefix, bundle, dir, validation } of validatedBundles) {
    const rangedArtifact = transportTelemetryFiles(validation).find(
      (entry) => Number.isSafeInteger(entry?.size) && entry.size >= 8,
    );
    if (!rangedArtifact) {
      throw new Error(`${bundle} has no range-testable physical transport part`);
    }
    await validateRangeResponse(
      `${baseUrl}/${prefix}/${rangedArtifact.path.split("/").map(encodeURIComponent).join("/")}`,
      rangedArtifact.size,
      join(dir, rangedArtifact.path),
    );
  }

  const fixtureMetadataUrl = `${baseUrl}/fixture/metadata.json`;
  const fixtureMetadataResponse = await assertResponse(
    await fetch(fixtureMetadataUrl, { headers: { Origin: "https://browser-parity.invalid" } }),
    200,
    fixtureMetadataUrl,
  );
  const fixtureMetadata = await fixtureMetadataResponse.json();
  if (
    fixtureMetadata.variant !== "edit-turbo-1k5" ||
    fixtureMetadata.resolution_profile !== "1k5" ||
    fixtureMetadata.width !== 1536 ||
    fixtureMetadata.height !== 1536 ||
    fixtureMetadata.schema_version !== 2 ||
    fixtureMetadata.dtype !== "bf16" ||
    fixtureMetadata.prompt !== FIXTURE_PROMPT ||
    fixtureMetadata.seed !== 42 ||
    fixtureMetadata.capture_blocks !== !vaeReferenceMode ||
    fixtureMetadata.capture_qwen !== !vaeReferenceMode ||
    fixtureMetadata.model_revision !== CANONICAL_MODEL_REVISION ||
    fixtureMetadata.upstream_source_revision !== CANONICAL_UPSTREAM_SOURCE_REVISION ||
    fixtureMetadata.output?.align_res !== false ||
    fixtureMetadata.output?.requested?.width !== 1536 ||
    fixtureMetadata.output?.requested?.height !== 1536 ||
    fixtureMetadata.output?.actual?.width !== 1536 ||
    fixtureMetadata.output?.actual?.height !== 1536 ||
    fixtureMetadata.output?.actual?.mode !== "RGB" ||
    fixtureMetadata.output?.actual?.image_count !== 1 ||
    fixtureMetadata.output?.validated !== true ||
    fixtureMetadata.provenance?.source_image?.copy_verified !== true ||
    fixtureMetadata.provenance?.source_image?.size !== fixtureFiles.source.size ||
    fixtureMetadata.provenance?.source_image?.sha256 !== fixtureFiles.source.sha256 ||
    typeof fixtureMetadata.tensors !== "object" ||
    fixtureMetadata.tensors === null ||
    Object.keys(fixtureMetadata.tensors).length !== ACTIVE_FIXTURE_TENSOR_COUNT
  ) {
    throw new Error(
      `fixture is not the exhaustive 1536x1536 Edit-Turbo 1.5K oracle: ${JSON.stringify({
        variant: fixtureMetadata.variant,
        width: fixtureMetadata.width,
        height: fixtureMetadata.height,
        schema_version: fixtureMetadata.schema_version,
        dtype: fixtureMetadata.dtype,
        capture_blocks: fixtureMetadata.capture_blocks,
        capture_qwen: fixtureMetadata.capture_qwen,
        tensor_count: Object.keys(fixtureMetadata.tensors ?? {}).length,
      })}`,
    );
  }
  if (fixtureMetadata.model_revision !== manifest.model_revision) {
    throw new Error(
      `fixture model revision ${fixtureMetadata.model_revision} does not match bundle revision ${manifest.model_revision}`,
    );
  }
  await Promise.all([
    validateLocalFileIdentity(fixtureDir, fixtureFiles.metadata),
    validateLocalFileIdentity(fixtureDir, fixtureFiles.source),
    validateLocalFileIdentity(fixtureDir, fixtureFiles.output),
    ...(vaeReferenceMode
      ? [validateLocalFileIdentity(fixtureDir, fixtureFiles.tensors)]
      : []),
  ]);
  const tensorPath = join(fixtureDir, "tensors.safetensors");
  const tensorLayout = await readSafeTensorsLayout(tensorPath, fixtureFiles.tensors.size);
  const tensorSize = fixtureFiles.tensors.size;
  await validateRangeResponse(`${baseUrl}/fixture/tensors.safetensors`, tensorSize, tensorPath);

  const optionsResponse = await fetch(fixtureMetadataUrl, {
    method: "OPTIONS",
    headers: {
      Origin: "https://browser-parity.invalid",
      "Access-Control-Request-Headers": "range",
      "Access-Control-Request-Method": "GET",
    },
  });
  await assertResponse(optionsResponse, 204, fixtureMetadataUrl);
  if (!(optionsResponse.headers.get("access-control-allow-headers") ?? "").toLowerCase().includes("range")) {
    throw new Error(`${fixtureMetadataUrl} OPTIONS response does not allow Range`);
  }
  return {
    artifact_content_digest: manifest.content_digest ?? null,
    artifact_file_count: artifactInventory.files,
    artifact_weight_file_count: artifactInventory.weightFiles,
    artifact_weight_bytes: logicalWeightBytes,
    artifact_bytes: artifactInventory.totalBytes,
    artifact_max_file_bytes: artifactInventory.maximumBytes,
    transport_layout_path: ARTIFACT_TRANSPORT_LAYOUT_PATH,
    physical_transport_part_reference_count: physicalTransportInventory.partReferences,
    physical_transport_unique_part_count: physicalTransportInventory.uniqueParts,
    physical_transport_reconstructed_bytes: physicalTransportInventory.reconstructedBytes,
    physical_transport_unique_part_bytes: physicalTransportInventory.uniquePartBytes,
    physical_transport_max_part_bytes: physicalTransportInventory.maximumPartBytes,
    physical_transport_target_part_bytes: ARTIFACT_TRANSPORT_TARGET_PART_BYTES,
    physical_transport_hard_max_part_bytes: ARTIFACT_TRANSPORT_HARD_MAX_PART_BYTES,
    direct_artifact_file_count: physicalTransportInventory.directFiles,
    direct_artifact_bytes: physicalTransportInventory.directBytes,
    transport_sidecars: validatedBundles.map(({ bundle, validation }) => ({
      bundle,
      ...validation.sidecar,
    })),
    vae_encoder_weight_file_count: vaeEncoderWeightFiles.length,
    vae_encoder_weight_bytes: vaeEncoderWeightBytes,
    vae_encoder_weight_objects: vaeEncoderWeightFiles
      .map((entry) => ({ path: entry.path, size: entry.size }))
      .sort((left, right) => left.path.localeCompare(right.path)),
    model_revision: manifest.model_revision,
    fixture_schema_version: fixtureMetadata.schema_version ?? null,
    fixture_tensor_count: Object.keys(fixtureMetadata.tensors).length,
    fixture_tensor_bytes: tensorSize,
    fixture_safetensors_header_bytes: tensorLayout.header_bytes,
    fixture_expected_tensor_bytes: tensorLayout.tensor_bytes,
    fixture_files: fixtureFiles,
  };
}

async function canExecute(path) {
  try {
    await access(path, fsConstants.X_OK);
    return true;
  } catch {
    return false;
  }
}

async function findOnPath(name) {
  if (isAbsolute(name) || name.includes(sep)) return (await canExecute(name)) ? name : undefined;
  for (const directory of (process.env.PATH ?? "").split(delimiter)) {
    if (!directory) continue;
    const candidate = join(directory, name);
    if (await canExecute(candidate)) return candidate;
  }
  return undefined;
}

async function findChrome() {
  const override = process.env[CHROME_ENV] ?? process.env.CHROME_BIN;
  if (override) {
    const resolved = await findOnPath(override);
    if (!resolved) throw new Error(`${CHROME_ENV} executable was not found: ${override}`);
    return resolved;
  }
  for (const candidate of [
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
  ]) {
    const resolved = await findOnPath(candidate);
    if (resolved) return resolved;
  }
  throw new Error(`Chrome/Chromium not found; set ${CHROME_ENV} to its executable`);
}

function appendBounded(current, chunk) {
  const combined = current + chunk.toString();
  return combined.length > MAX_CAPTURED_CHROME_BYTES
    ? combined.slice(combined.length - MAX_CAPTURED_CHROME_BYTES)
    : combined;
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
      requested_bytes: BROWSER_1K5_CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES,
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
  } else if (
    availableBytes < BigInt(BROWSER_1K5_CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES)
  ) {
    measurement.quota_aware_allocation_probe.skipped_reason =
      "statfs-available-below-minimum";
  } else {
    measurement.quota_aware_allocation_probe = await quotaAwareAllocationProbe(
      path,
      BROWSER_1K5_CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES,
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
      ...selectBrowser1k5ChromeSharedMemoryPolicy({
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
    ...selectBrowser1k5ChromeSharedMemoryPolicy({
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
  const statFields = statLine.slice(commandEnd + 2).trim().split(/\s+/);
  const ppid = Number(statFields[1]);
  if (!Number.isSafeInteger(ppid) || ppid < 0) {
    throw new Error(`/proc/${pid}/stat has invalid parent PID ${statFields[1]}`);
  }
  let command = "";
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
    let childPids;
    try {
      const taskEntries = await readdir(`/proc/${parent.pid}/task`, {
        withFileTypes: true,
      });
      const taskChildLists = await Promise.all(
        taskEntries
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
          taskChildLists
            .flatMap((children) => children.trim().split(/\s+/))
            .filter(Boolean)
            .map(Number)
            .filter((pid) => Number.isSafeInteger(pid) && pid > 1),
        ),
      );
    } catch {
      childPids = [];
    }
    for (const pid of childPids) {
      if (seen.has(pid)) continue;
      seen.add(pid);
      try {
        const child = await readProcProcess(pid, parent.depth + 1);
        descendants.push(child);
        pending.push(child);
      } catch {
        // Chrome helper processes can exit between reading the children list and their stat file.
      }
    }
  }
  return descendants;
}

function parseProcKiB(text, source) {
  const values = {};
  for (const line of text.split(/\r?\n/)) {
    const match = /^([A-Za-z_()]+):\s+(\d+)\s+kB$/.exec(line);
    if (match) values[match[1]] = Number(match[2]);
  }
  if (Object.keys(values).length === 0) throw new Error(`${source} has no kB counters`);
  return values;
}

async function readProcSmapsRollup(pid) {
  const values = parseProcKiB(
    await readFile(`/proc/${pid}/smaps_rollup`, "utf8"),
    `/proc/${pid}/smaps_rollup`,
  );
  const privateKiB =
    (values.Private_Clean ?? 0) +
    (values.Private_Dirty ?? 0) +
    (values.Private_Hugetlb ?? 0);
  const sharedKiB =
    (values.Shared_Clean ?? 0) +
    (values.Shared_Dirty ?? 0) +
    (values.Shared_Hugetlb ?? 0);
  return {
    rss_kib: values.Rss ?? 0,
    pss_kib: values.Pss ?? 0,
    private_kib: privateKiB,
    shared_kib: sharedKiB,
    swap_kib: values.Swap ?? 0,
  };
}

async function readProcMeminfo() {
  const values = parseProcKiB(await readFile("/proc/meminfo", "utf8"), "/proc/meminfo");
  for (const field of ["MemAvailable", "Cached", "Shmem"]) {
    if (!Number.isFinite(values[field])) throw new Error(`/proc/meminfo omits ${field}`);
  }
  return {
    mem_available_kib: values.MemAvailable,
    cached_kib: values.Cached,
    shmem_kib: values.Shmem,
    swap_total_kib: values.SwapTotal ?? 0,
    swap_free_kib: values.SwapFree ?? 0,
  };
}

function sumProcessMemory(records) {
  return records.reduce(
    (total, record) => {
      for (const field of ["rss_kib", "pss_kib", "private_kib", "shared_kib", "swap_kib"]) {
        total[field] += record[field];
      }
      return total;
    },
    {
      process_count: records.length,
      rss_kib: 0,
      pss_kib: 0,
      private_kib: 0,
      shared_kib: 0,
      swap_kib: 0,
    },
  );
}

async function sampleChromeHostResources(rootPid, parityMilestone) {
  const root = await readProcProcess(rootPid);
  const processes = [root, ...(await readProcDescendants(rootPid))];
  const memory = [];
  for (const process of processes) {
    try {
      memory.push({ ...process, ...(await readProcSmapsRollup(process.pid)) });
    } catch {
      // Helpers can disappear during a sample. The root and stable GPU process are validated below.
    }
  }
  const gpuMemory = memory.filter((process) => process.command.includes("--type=gpu-process"));
  return {
    at_ms: Date.now(),
    parity_milestone: parityMilestone,
    chrome_tree: sumProcessMemory(memory),
    chrome_gpu_processes: sumProcessMemory(gpuMemory),
    system: await readProcMeminfo(),
    processes: memory,
  };
}

function peakProcessMemory(target, value) {
  target.process_count = Math.max(target.process_count, value.process_count);
  for (const field of ["rss_kib", "pss_kib", "private_kib", "shared_kib", "swap_kib"]) {
    target[field] = Math.max(target[field], value[field]);
  }
}

async function startHostResourceMonitor(rootPid, workloadTelemetry) {
  const evidence = {
    provider: "linux-procfs-smaps-rollup",
    root_chrome_pid: rootPid,
    workload_window_started_at_ms: Date.now(),
    sample_attempts: 0,
    samples: 0,
    skipped_overlapping_samples: 0,
    sample_error_count: 0,
    sample_errors: [],
    sample_records: [],
    dropped_sample_records: 0,
    baseline: null,
    last: null,
    peak_chrome_tree: sumProcessMemory([]),
    peak_chrome_gpu_processes: sumProcessMemory([]),
    minimum_system_mem_available_kib: null,
    maximum_system_cached_kib: 0,
    maximum_system_shmem_kib: 0,
    observed_processes: new Map(),
    dropped_observed_processes: 0,
  };
  let sampling = false;
  let activeSample = Promise.resolve();
  const sample = () => {
    if (sampling) {
      evidence.skipped_overlapping_samples += 1;
      return activeSample;
    }
    sampling = true;
    evidence.sample_attempts += 1;
    activeSample = (async () => {
      try {
        const full = await sampleChromeHostResources(
          rootPid,
          workloadTelemetry.captureParityMilestone(),
        );
        const record = {
          at_ms: full.at_ms,
          parity_milestone: full.parity_milestone,
          chrome_tree: full.chrome_tree,
          chrome_gpu_processes: full.chrome_gpu_processes,
          system: full.system,
        };
        evidence.samples += 1;
        evidence.baseline ??= record;
        evidence.last = record;
        peakProcessMemory(evidence.peak_chrome_tree, record.chrome_tree);
        peakProcessMemory(evidence.peak_chrome_gpu_processes, record.chrome_gpu_processes);
        evidence.minimum_system_mem_available_kib = Math.min(
          evidence.minimum_system_mem_available_kib ?? record.system.mem_available_kib,
          record.system.mem_available_kib,
        );
        evidence.maximum_system_cached_kib = Math.max(
          evidence.maximum_system_cached_kib,
          record.system.cached_kib,
        );
        evidence.maximum_system_shmem_kib = Math.max(
          evidence.maximum_system_shmem_kib,
          record.system.shmem_kib,
        );
        for (const process of full.processes) {
          const observed = evidence.observed_processes.get(process.pid);
          if (!observed && evidence.observed_processes.size >= MAX_TRACKED_CHROME_PROCESSES) {
            evidence.dropped_observed_processes += 1;
            continue;
          }
          evidence.observed_processes.set(process.pid, {
            pid: process.pid,
            ppid: process.ppid,
            command: process.command,
            first_seen_at_ms: observed?.first_seen_at_ms ?? full.at_ms,
            last_seen_at_ms: full.at_ms,
            observed_intervals: (observed?.observed_intervals ?? 0) + 1,
            max_pss_kib: Math.max(observed?.max_pss_kib ?? 0, process.pss_kib),
            max_private_kib: Math.max(observed?.max_private_kib ?? 0, process.private_kib),
            max_shared_kib: Math.max(observed?.max_shared_kib ?? 0, process.shared_kib),
          });
        }
        if (evidence.sample_records.length === MAX_RECORDED_HOST_RESOURCE_SAMPLES) {
          evidence.sample_records.shift();
          evidence.dropped_sample_records += 1;
        }
        evidence.sample_records.push(record);
      } catch (error) {
        evidence.sample_error_count += 1;
        if (evidence.sample_errors.length < MAX_RECORDED_MONITOR_ERRORS) {
          evidence.sample_errors.push(error instanceof Error ? error.message : String(error));
        }
      } finally {
        sampling = false;
      }
    })();
    return activeSample;
  };
  await sample();
  const timer = setInterval(() => void sample(), HOST_RESOURCE_SAMPLE_INTERVAL_MS);
  timer.unref?.();
  let stoppedResult;
  return {
    async stop() {
      if (stoppedResult) return stoppedResult;
      clearInterval(timer);
      await activeSample;
      await sample();
      const validationFailures = [];
      const result = {
        ...evidence,
        workload_window_finished_at_ms: Date.now(),
        workload_window_elapsed_ms: Date.now() - evidence.workload_window_started_at_ms,
        observed_processes: Array.from(evidence.observed_processes.values()).sort(
          (left, right) => left.pid - right.pid,
        ),
      };
      result.delta_from_baseline = result.baseline && result.last
        ? {
            chrome_tree_pss_kib:
              result.last.chrome_tree.pss_kib - result.baseline.chrome_tree.pss_kib,
            chrome_tree_private_kib:
              result.last.chrome_tree.private_kib - result.baseline.chrome_tree.private_kib,
            chrome_tree_shared_kib:
              result.last.chrome_tree.shared_kib - result.baseline.chrome_tree.shared_kib,
            system_mem_available_kib:
              result.last.system.mem_available_kib - result.baseline.system.mem_available_kib,
            system_cached_kib: result.last.system.cached_kib - result.baseline.system.cached_kib,
            system_shmem_kib: result.last.system.shmem_kib - result.baseline.system.shmem_kib,
            system_swap_free_kib:
              result.last.system.swap_free_kib - result.baseline.system.swap_free_kib,
          }
        : null;
      if (result.samples < 3) {
        validationFailures.push(
          `procfs captured ${result.samples} Chrome resource samples; at least 3 are required`,
        );
      }
      if (result.peak_chrome_tree.pss_kib <= 0) {
        validationFailures.push("procfs did not capture positive Chrome process-tree PSS");
      }
      if (result.peak_chrome_gpu_processes.pss_kib <= 0) {
        validationFailures.push("procfs did not capture the Chrome GPU process smaps_rollup");
      }
      if (!(result.minimum_system_mem_available_kib > 0)) {
        validationFailures.push("procfs did not capture positive MemAvailable");
      }
      if (result.sample_error_count !== 0) {
        validationFailures.push(
          `procfs resource monitor encountered ${result.sample_error_count} sampling errors`,
        );
      }
      if (result.dropped_observed_processes !== 0) {
        validationFailures.push(
          `${result.dropped_observed_processes} Chrome process observations exceeded the bounded process inventory`,
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

function parseCounter(value) {
  return /^\d+$/.test(value ?? "") ? Number(value) : null;
}

function parseNvidiaPmonHeader(header) {
  const columns = header
    .replace(/^\s*#\s*/, "")
    .trim()
    .toLowerCase()
    .split(/\s+/);
  const column = (...names) => names.map((name) => columns.indexOf(name)).find((index) => index >= 0);
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
  const sampleKey = `${fields[indexes.date]} ${fields[indexes.time]}`;
  return {
    sampleKey,
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
  const descendants = await readProcDescendants(rootPid);
  const gpuProcesses = descendants
    .filter((process) => process.command.includes("--type=gpu-process"))
    .map((process) => ({ pid: process.pid, ppid: process.ppid, command: process.command }));
  for (const process of gpuProcesses) {
    const observed = evidence.observed_gpu_processes.get(process.pid);
    if (!observed && evidence.observed_gpu_processes.size >= MAX_TRACKED_CHROME_PROCESSES) {
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
  const rows = pmonRows.filter(
    (row) => row.gpu_index !== null && row.pid !== null && gpuProcessPids.has(row.pid),
  );
  const activeRows = rows.filter(
    (row) => (row.framebuffer_mib ?? 0) > 0 && (row.sm_percent ?? 0) > 0,
  );
  // One pmon interval may contain multiple Chrome GPU-process rows and/or one row per GPU. The
  // low-VRAM contract applies to their total framebuffer footprint, not the largest individual
  // row. Sum the complete scoped interval before updating the observed peak.
  const totalFramebufferMib = rows.reduce(
    (total, row) => total + (row.framebuffer_mib ?? 0),
    0,
  );
  evidence.samples += 1;
  if (rows.length > 0) evidence.matched_sample_intervals += 1;
  if (activeRows.length > 0) {
    evidence.active_sample_intervals += 1;
    evidence.current_consecutive_active_intervals += 1;
    evidence.max_consecutive_active_intervals = Math.max(
      evidence.max_consecutive_active_intervals,
      evidence.current_consecutive_active_intervals,
    );
  } else {
    evidence.current_consecutive_active_intervals = 0;
  }
  evidence.max_framebuffer_mib = Math.max(evidence.max_framebuffer_mib, totalFramebufferMib);
  for (const row of rows) {
    evidence.matched_samples += 1;
    evidence.max_sm_percent = Math.max(evidence.max_sm_percent, row.sm_percent ?? 0);
    evidence.max_memory_percent = Math.max(evidence.max_memory_percent, row.memory_percent ?? 0);
    evidence.gpu_indexes.add(row.gpu_index);
    evidence.pids.add(row.pid);
  }
  evidence.active_rows += activeRows.length;
  const sampleRecord = {
    at_ms: sampledAt,
    native_sample_key: sampleKey,
    chrome_gpu_process_pids: Array.from(gpuProcessPids).sort((left, right) => left - right),
    total_framebuffer_mib: totalFramebufferMib,
    matched_rows: rows,
    active_rows: activeRows.length,
  };
  if (evidence.sample_records.length === MAX_RECORDED_GPU_SAMPLES) {
    evidence.sample_records.shift();
    evidence.dropped_sample_records += 1;
  }
  evidence.sample_records.push(sampleRecord);
}

async function startNativeGpuMonitor(
  rootPid,
  minimumFramebufferMib = MIN_F32_RETAINED_DENOISER_FRAMEBUFFER_MIB,
  framebufferPolicy = "F32 qualification retained-denoiser policy",
  maximumFramebufferBytesExclusive = null,
) {
  if (
    maximumFramebufferBytesExclusive !== null &&
    (!Number.isSafeInteger(maximumFramebufferBytesExclusive) ||
      maximumFramebufferBytesExclusive <= 0)
  ) {
    throw new Error(
      `invalid exclusive framebuffer byte ceiling ${maximumFramebufferBytesExclusive}`,
    );
  }
  const inventoryOutput = await commandOutput("nvidia-smi", [
    "--query-gpu=index,uuid,name,driver_version,memory.total",
    "--format=csv,noheader,nounits",
  ]);
  const gpuInventory = inventoryOutput
    .trim()
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => {
      const [index, uuid, name, driver_version, memory_total_mib] = line
        .split(",")
        .map((value) => value.trim());
      return {
        index: Number(index),
        uuid,
        name,
        driver_version,
        memory_total_mib: Number(memory_total_mib),
      };
    });
  if (
    gpuInventory.length === 0 ||
    gpuInventory.some(
      (gpu) =>
        !Number.isInteger(gpu.index) ||
        gpu.index < 0 ||
        !gpu.uuid.startsWith("GPU-") ||
        gpu.name.length === 0 ||
        gpu.driver_version.length === 0 ||
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
    sample_attempts: 0,
    samples: 0,
    matched_samples: 0,
    matched_sample_intervals: 0,
    active_rows: 0,
    active_sample_intervals: 0,
    current_consecutive_active_intervals: 0,
    max_consecutive_active_intervals: 0,
    max_framebuffer_mib: 0,
    max_sm_percent: 0,
    max_memory_percent: 0,
    minimum_framebuffer_mib: minimumFramebufferMib,
    maximum_framebuffer_bytes_exclusive: maximumFramebufferBytesExclusive,
    framebuffer_policy: framebufferPolicy,
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
    async stop() {
      if (stoppedResult) return stoppedResult;
      const exitedBeforeStop = monitor.exitCode !== null || monitor.signalCode !== null;
      if (!exitedBeforeStop) monitor.kill("SIGTERM");
      let monitorExit;
      try {
        monitorExit = await Promise.race([
          monitorClosed,
          delay(5_000).then(() => null),
        ]);
        if (!monitorExit) {
          monitor.kill("SIGKILL");
          monitorExit = await monitorClosed;
        }
      } catch (error) {
        recordMonitorError(error);
        monitorExit = { code: monitor.exitCode, signal: monitor.signalCode };
      }
      if (stdoutRemainder.trim() !== "") processMonitorLine(stdoutRemainder);
      if (pendingKey !== undefined) enqueueInterval(pendingKey, pendingRows);
      await intervalChain;
      const validationFailures = [];
      const result = {
        ...evidence,
        monitor_arguments: monitorArguments,
        monitor_exit: monitorExit,
        monitor_exited_before_stop: exitedBeforeStop,
        monitor_stdout_tail: monitorStdout,
        monitor_stderr_tail: monitorStderr,
        workload_window_finished_at_ms: Date.now(),
        workload_window_elapsed_ms: Date.now() - evidence.workload_window_started_at_ms,
        observed_gpu_processes: Array.from(evidence.observed_gpu_processes.values()).sort(
          (left, right) => left.pid - right.pid,
        ),
        gpu_indexes: Array.from(evidence.gpu_indexes).sort((left, right) => left - right),
        pids: Array.from(evidence.pids).sort((left, right) => left - right),
        observed_max_framebuffer_bytes: evidence.max_framebuffer_mib * 1024 * 1024,
      };
      delete result.current_consecutive_active_intervals;
      if (result.observed_gpu_processes.length === 0) {
        validationFailures.push("no Chrome GPU-process descendant was observed in the workload window");
      }
      if (result.matched_sample_intervals < MIN_GPU_MATCHED_SAMPLE_INTERVALS) {
        validationFailures.push(
          `nvidia-smi matched the Chrome GPU PID in ${result.matched_sample_intervals} intervals; at least ${MIN_GPU_MATCHED_SAMPLE_INTERVALS} are required`,
        );
      }
      if (result.active_sample_intervals < MIN_GPU_ACTIVE_SAMPLE_INTERVALS) {
        validationFailures.push(
          `nvidia-smi observed nonzero framebuffer and SM use together in ${result.active_sample_intervals} Chrome GPU intervals; at least ${MIN_GPU_ACTIVE_SAMPLE_INTERVALS} are required`,
        );
      }
      if (
        result.max_consecutive_active_intervals < MIN_GPU_CONSECUTIVE_ACTIVE_SAMPLE_INTERVALS
      ) {
        validationFailures.push(
          `nvidia-smi observed only ${result.max_consecutive_active_intervals} consecutive active Chrome GPU intervals; at least ${MIN_GPU_CONSECUTIVE_ACTIVE_SAMPLE_INTERVALS} are required`,
        );
      }
      if (result.max_framebuffer_mib <= 0 || result.max_sm_percent <= 0) {
        validationFailures.push(
          "nvidia-smi did not attribute both framebuffer allocation and SM utilization to an observed Chrome GPU PID",
        );
      }
      if (result.max_framebuffer_mib < minimumFramebufferMib) {
        validationFailures.push(
          `Chrome GPU-process framebuffer peaked at ${result.max_framebuffer_mib} MiB; the ${framebufferPolicy} requires at least ${minimumFramebufferMib} MiB`,
        );
      }
      if (
        maximumFramebufferBytesExclusive !== null &&
        result.observed_max_framebuffer_bytes >= maximumFramebufferBytesExclusive
      ) {
        validationFailures.push(
          `Chrome GPU-process framebuffer peaked at ${result.observed_max_framebuffer_bytes} bytes; the ${framebufferPolicy} requires a peak strictly below ${maximumFramebufferBytesExclusive} bytes`,
        );
      }
      if (!layout) {
        validationFailures.push("persistent nvidia-smi pmon emitted no parseable dated header");
      }
      if (exitedBeforeStop) {
        validationFailures.push(
          `persistent nvidia-smi pmon exited before workload completion (code=${monitorExit?.code}, signal=${monitorExit?.signal})`,
        );
      }
      if (result.sample_error_count !== 0) {
        validationFailures.push(
          `persistent nvidia-smi monitor encountered ${result.sample_error_count} sampling errors`,
        );
      }
      if (result.dropped_observed_gpu_processes !== 0) {
        validationFailures.push(
          `${result.dropped_observed_gpu_processes} Chrome GPU-process observations exceeded the bounded process inventory`,
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

function chromeLaunchArguments(profile, url, headful, sharedMemoryPolicy) {
  const arguments_ = ["--no-sandbox"];
  if (headful) {
    arguments_.push("--window-size=800,600", "--ozone-platform=x11");
  } else {
    arguments_.push("--headless=new", "--disable-vulkan-surface");
  }
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
    `--user-data-dir=${profile}`,
    "--remote-debugging-port=0",
    url,
  );
  return arguments_;
}

async function startChrome(executable, arguments_) {
  const child = spawn(executable, arguments_, {
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
  return { child, arguments_: [...arguments_], processGroupId: child.pid };
}

function signalPid(pid, signal, errors) {
  try {
    process.kill(pid, signal);
    return true;
  } catch (error) {
    if (error?.code !== "ESRCH") errors.push(`${signal} PID ${pid}: ${error}`);
    return false;
  }
}

function signalProcessGroup(processGroupId, signal, errors) {
  if (!Number.isInteger(processGroupId) || processGroupId <= 1) {
    errors.push(`refused to signal invalid Chrome process group ${processGroupId}`);
    return false;
  }
  try {
    process.kill(-processGroupId, signal);
    return true;
  } catch (error) {
    if (error?.code !== "ESRCH") errors.push(`${signal} process group ${processGroupId}: ${error}`);
    return false;
  }
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

async function stopChrome(browser) {
  const child = browser?.child;
  if (!child) return null;
  const rootPid = child.pid;
  const processGroupId = browser.processGroupId;
  const errors = [];
  let descendants = [];
  try {
    descendants = (await readProcDescendants(rootPid)).sort(
      (left, right) => right.depth - left.depth || right.pid - left.pid,
    );
  } catch (error) {
    errors.push(`could not enumerate Chrome descendants: ${error}`);
  }
  for (const process of descendants) signalPid(process.pid, "SIGTERM", errors);
  signalPid(rootPid, "SIGTERM", errors);
  // The process group is dedicated by startChrome. Signaling it after the leaf-first pass catches
  // descendants that reparented or appeared between process-table enumeration and shutdown.
  signalProcessGroup(processGroupId, "SIGTERM", errors);
  let exited = await waitForProcessGroupExit(processGroupId, 5_000);
  if (!exited) {
    for (const process of descendants) signalPid(process.pid, "SIGKILL", errors);
    signalPid(rootPid, "SIGKILL", errors);
    signalProcessGroup(processGroupId, "SIGKILL", errors);
    exited = await waitForProcessGroupExit(processGroupId, 5_000);
  }
  if (!exited) errors.push(`Chrome process group ${processGroupId} survived SIGKILL`);
  const cleanup = {
    root_pid: rootPid,
    process_group_id: processGroupId,
    enumerated_descendant_pids: descendants.map((process) => process.pid),
    process_group_exited: exited,
    errors,
  };
  browser.cleanup = cleanup;
  return cleanup;
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
      const lines = (await readFile(path, "utf8")).trim().split(/\r?\n/);
      const port = Number(lines[0]);
      if (Number.isInteger(port) && port > 0 && port <= 65535) return port;
    } catch {
      // Chrome writes this file when the DevTools endpoint is ready.
    }
    await delay(100);
  }
  throw new Error("timed out waiting for Chrome's DevToolsActivePort");
}

async function findPageTarget(port, harnessUrl, browser, deadline) {
  while (Date.now() < deadline) {
    throwIfInterrupted();
    if (browser.child.exitCode !== null || browser.child.signalCode !== null) {
      throw new Error(
        `Chrome exited before opening the harness (code=${browser.child.exitCode}, signal=${browser.child.signalCode})`,
      );
    }
    try {
      const response = await fetch(`http://127.0.0.1:${port}/json/list`, {
        signal: AbortSignal.timeout(1_000),
      });
      const targets = await response.json();
      const exact = targets.find((target) => target.type === "page" && target.url === harnessUrl);
      if (exact?.webSocketDebuggerUrl) return exact;
    } catch {
      // The endpoint and target can appear shortly after DevToolsActivePort.
    }
    await delay(100);
  }
  throw new Error("timed out waiting for the browser page target");
}

function renderRemoteArgument(argument) {
  if (Object.hasOwn(argument ?? {}, "value")) return argument.value;
  return argument?.description ?? argument?.type ?? "<unavailable>";
}

async function openCdp(url, events, workloadTelemetry) {
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
  let fatalError;
  let terminalError;
  let terminalFailure;
  const pending = new Map();
  const terminalWaiters = new Set();
  const pageErrors = [];
  const gpuErrors = [];
  let droppedPageErrors = 0;
  let droppedGpuErrors = 0;

  const recordEvent = (event) => {
    if (events.length < MAX_EVENT_COUNT) events.push({ at_ms: Date.now(), ...event });
  };
  const recordPageError = (message) => {
    if (pageErrors.length < MAX_RECORDED_BROWSER_ERRORS) pageErrors.push(message);
    else droppedPageErrors += 1;
    recordEvent({ type: "page_error", message });
  };
  const recordGpuError = (message) => {
    if (gpuErrors.length < MAX_RECORDED_BROWSER_ERRORS) gpuErrors.push(message);
    else droppedGpuErrors += 1;
  };
  const recordTerminalError = (kind, message) => {
    if (terminalError) return;
    terminalFailure = { at_ms: Date.now(), kind, message };
    terminalError = new Error(`browser ${kind}: ${message}`);
    terminalError.browserTerminalFailure = terminalFailure;
    recordEvent({ type: "browser_terminal_failure", ...terminalFailure });
    for (const entry of pending.values()) {
      clearTimeout(entry.timer);
      entry.reject(terminalError);
    }
    pending.clear();
    for (const resolveWaiter of terminalWaiters) resolveWaiter(terminalError);
    terminalWaiters.clear();
  };

  socket.addEventListener("message", (event) => {
    let message;
    try {
      message = JSON.parse(String(event.data));
    } catch (error) {
      fatalError = new Error(`invalid CDP message: ${error instanceof Error ? error.message : String(error)}`);
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
    if (message.method === "Runtime.exceptionThrown") {
      const details = message.params?.exceptionDetails;
      const rendered =
        details?.exception?.description ?? details?.text ?? "uncaught page exception";
      recordPageError(rendered);
      recordTerminalError("uncaught page exception", rendered);
    } else if (message.method === "Runtime.consoleAPICalled") {
      const type = message.params?.type ?? "unknown";
      const rendered = (message.params?.args ?? []).map(renderRemoteArgument).join(" ");
      workloadTelemetry.noteConsoleMessage(rendered);
      recordEvent({ type: `console.${type}`, message: rendered });
      if (["error", "assert"].includes(type)) {
        recordPageError(`console.${type}: ${rendered}`);
      }
      if (/\b(webgpu|wgpu|dawn|vulkan|gpu(device)?)\b.*\b(error|failed|lost|invalid)\b/i.test(rendered)) {
        recordGpuError(rendered);
      }
      if (isFatalConsoleMessage(type, rendered)) {
        recordTerminalError("fatal console error", rendered);
      }
    } else if (message.method === "Log.entryAdded") {
      const entry = message.params?.entry;
      const ignoredResource = isIgnoredBrowserResource(entry?.url);
      recordEvent({
        type: `log.${entry?.level ?? "unknown"}`,
        source: entry?.source,
        url: entry?.url,
        ignored: ignoredResource,
        message: entry?.text,
      });
      if (ignoredResource) return;
      if (entry?.level === "error") recordPageError(`page log: ${entry.text}`);
      if (
        (entry?.source === "rendering" || entry?.source === "gpu") &&
        (entry?.level === "error" || /\b(error|failed|lost|invalid)\b/i.test(entry?.text ?? ""))
      ) {
        recordGpuError(entry.text);
        recordTerminalError(`${entry.source} failure`, entry.text ?? "unknown browser failure");
      }
    } else if (message.method === "Network.loadingFailed") {
      const failure = message.params;
      const detail = `network request ${failure?.requestId} failed: ${failure?.errorText}`;
      recordPageError(detail);
    } else if (message.method === "Network.responseReceived") {
      const response = message.params?.response;
      if (response?.status >= 400 && !isIgnoredBrowserResource(response.url)) {
        recordPageError(`network response HTTP ${response.status}: ${response.url}`);
      }
    } else if (message.method === "Inspector.targetCrashed") {
      recordPageError("Chrome page target crashed");
      recordTerminalError("renderer crash", "Chrome page target crashed");
    } else if (message.method === "Inspector.detached") {
      const reason = message.params?.reason ?? "unknown reason";
      recordPageError(`Chrome inspector detached: ${reason}`);
      recordTerminalError("renderer detached", reason);
    }
  });
  socket.addEventListener("error", () => {
    fatalError = new Error("CDP WebSocket error");
  });
  socket.addEventListener("close", () => {
    if (!fatalError) fatalError = new Error("CDP WebSocket closed unexpectedly");
    for (const entry of pending.values()) {
      clearTimeout(entry.timer);
      entry.reject(fatalError);
    }
    pending.clear();
  });

  const call = (method, params = {}, timeoutMs = CDP_CALL_TIMEOUT_MS) => {
    if (fatalError) return Promise.reject(fatalError);
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
    call("Network.enable", { maxTotalBufferSize: 1_000_000, maxResourceBufferSize: 100_000 }),
    call("Page.enable"),
    call("Inspector.enable"),
  ]);
  return {
    call,
    close() {
      if (socket.readyState === WebSocket.OPEN) socket.close();
    },
    get fatalError() {
      return fatalError;
    },
    get terminalError() {
      return terminalError;
    },
    get terminalFailure() {
      return terminalFailure;
    },
    waitForTerminalError(timeoutMs) {
      if (terminalError) return Promise.resolve(terminalError);
      return new Promise((resolveWait) => {
        let timer;
        const finish = (error) => {
          if (timer) clearTimeout(timer);
          terminalWaiters.delete(finish);
          resolveWait(error);
        };
        terminalWaiters.add(finish);
        timer = setTimeout(() => finish(null), timeoutMs);
      });
    },
    gpuErrors,
    pageErrors,
    get droppedGpuErrors() {
      return droppedGpuErrors;
    },
    get droppedPageErrors() {
      return droppedPageErrors;
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

async function browserAdapterInfo(cdp) {
  const info = await evaluateValue(
    cdp,
    `(async () => {
      if (!navigator.gpu) throw new Error("navigator.gpu is unavailable");
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (!adapter) throw new Error("native WebGPU adapter request returned null");
      const source = adapter.info ?? {};
      return {
        vendor: source.vendor ?? "",
        architecture: source.architecture ?? "",
        device: source.device ?? "",
        description: source.description ?? "",
        is_fallback_adapter: adapter.isFallbackAdapter ?? source.isFallbackAdapter ?? false,
        features: Array.from(adapter.features ?? []).sort(),
        limits: {
          max_buffer_size: Number(adapter.limits?.maxBufferSize ?? 0),
          max_storage_buffer_binding_size: Number(
            adapter.limits?.maxStorageBufferBindingSize ?? 0,
          ),
        },
      };
    })()`,
    true,
  );
  const identity = JSON.stringify(info).toLowerCase();
  if (info?.is_fallback_adapter || /(swiftshader|llvmpipe|lavapipe|software adapter|warp)/.test(identity)) {
    throw new Error(`browser selected a software WebGPU adapter: ${JSON.stringify(info)}`);
  }
  if (!info) throw new Error("browser did not return WebGPU adapter information");
  return {
    ...info,
    identity_redacted: ![info.vendor, info.architecture, info.device, info.description].some(Boolean),
  };
}

function validateBrowser1k5AdapterLimits(adapter) {
  const failures = [];
  for (const [name, actual] of Object.entries({
    maxBufferSize: adapter?.limits?.max_buffer_size,
    maxStorageBufferBindingSize: adapter?.limits?.max_storage_buffer_binding_size,
  })) {
    if (!Number.isSafeInteger(actual) || actual < BROWSER_1K5_MIN_RUNTIME_BUFFER_BYTES) {
      failures.push(
        `${name}=${reportValue(actual)}, exact 1.5K F32 VAE decode requires at least ${BROWSER_1K5_MIN_RUNTIME_BUFFER_BYTES}`,
      );
    }
  }
  if (failures.length > 0) {
    const error = new Error(
      `browser adapter cannot execute exact 1.5K parity without an oversized-buffer panic:\n${failures.join("\n")}`,
    );
    error.adapterLimitFailures = failures;
    throw error;
  }
}

function reportValue(value) {
  const rendered = JSON.stringify(value);
  return rendered === undefined ? String(value) : rendered;
}

function expectReportEqual(failures, path, actual, expected) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    failures.push(`${path}=${reportValue(actual)}, expected ${reportValue(expected)}`);
  }
}

function expectReportNumber(failures, path, actual, predicate, requirement) {
  if (!Number.isFinite(actual) || !predicate(actual)) {
    failures.push(`${path}=${reportValue(actual)}, expected ${requirement}`);
  }
}

function expectReportClose(failures, path, actual, expected) {
  const tolerance = Math.max(1e-9, Math.abs(expected) * 1e-6);
  expectReportNumber(
    failures,
    path,
    actual,
    (value) => Math.abs(value - expected) <= tolerance,
    `${expected} (+/- ${tolerance})`,
  );
}

function validateBrowserWebGpuVaeF32OracleEnvelope(actual, path, failures) {
  for (const field of [
    "backend",
    "artifact_content_digest",
    "artifact_profile",
    "weight_storage_dtype",
    "weight_load_policy",
    "execution_dtype",
    "calibrated_adapter",
    "calibrated_device",
    "calibrated_driver",
    "portability",
  ]) {
    expectReportEqual(
      failures,
      `${path}.${field}`,
      actual?.[field],
      BROWSER_WEBGPU_VAE_F32_ORACLE_ENVELOPE[field],
    );
  }
  for (const component of ["moments", "mean", "logvar", "std", "raw_latent", "scaled_latent"]) {
    for (const [field, expected] of Object.entries(
      BROWSER_WEBGPU_VAE_F32_ORACLE_ENVELOPE[component],
    )) {
      expectReportClose(failures, `${path}.${component}.${field}`, actual?.[component]?.[field], expected);
    }
  }
}

function validateTensorMetric(metric, path, failures, expected = {}) {
  if (!metric || typeof metric !== "object" || Array.isArray(metric)) {
    failures.push(`${path} is not a tensor metric object`);
    return;
  }
  if (typeof metric.name !== "string" || metric.name.length === 0) {
    failures.push(`${path}.name is empty`);
  }
  if (typeof metric.oracle !== "string" || metric.oracle.length === 0) {
    failures.push(`${path}.oracle is empty`);
  }
  if (
    !Array.isArray(metric.shape) ||
    metric.shape.some((dimension) => !Number.isSafeInteger(dimension) || dimension < 0)
  ) {
    failures.push(`${path}.shape is invalid: ${reportValue(metric.shape)}`);
  }
  if (metric.actual_dtype !== "f32") {
    failures.push(`${path}.actual_dtype=${reportValue(metric.actual_dtype)}, expected "f32"`);
  }
  if (expected.name !== undefined) expectReportEqual(failures, `${path}.name`, metric.name, expected.name);
  if (expected.oracle !== undefined) {
    expectReportEqual(failures, `${path}.oracle`, metric.oracle, expected.oracle);
  }
  if (expected.shape !== undefined) {
    expectReportEqual(failures, `${path}.shape`, metric.shape, expected.shape);
  }
  const numericFields = [
    "element_count",
    "max_abs",
    "mean_abs",
    "rmse",
    "relative_rmse",
    "cosine_similarity",
  ];
  for (const field of numericFields) {
    if (!Number.isFinite(metric[field])) failures.push(`${path}.${field} is not finite`);
  }
  if (Array.isArray(metric.shape)) {
    const elements = metric.shape.reduce((total, dimension) => total * dimension, 1);
    if (Number.isSafeInteger(elements)) {
      expectReportEqual(failures, `${path}.element_count`, metric.element_count, elements);
    }
  }
  for (const field of ["max_abs", "mean_abs", "rmse", "relative_rmse"]) {
    if (Number.isFinite(metric[field]) && metric[field] < 0) {
      failures.push(`${path}.${field} is negative`);
    }
  }
  if (
    Number.isFinite(metric.cosine_similarity) &&
    (metric.cosine_similarity < -1 || metric.cosine_similarity > 1)
  ) {
    failures.push(`${path}.cosine_similarity is outside [-1, 1]`);
  }
}

function validateTensorGate(metric, gate, path, failures) {
  if (
    !metric ||
    metric.relative_rmse > gate.maximum_relative_rmse ||
    metric.cosine_similarity < gate.minimum_cosine_similarity
  ) {
    failures.push(
      `${path} misses rel-RMSE/cosine gate ${reportValue(gate)}: ${reportValue({
        relative_rmse: metric?.relative_rmse,
        cosine_similarity: metric?.cosine_similarity,
      })}`,
    );
  }
}

function validateFixtureFileIdentity(actual, expected, path, failures) {
  expectReportEqual(failures, path, actual, expected);
}

function expectedQwenAlignedStages() {
  const stages = [
    ["embedding_rows.5", "qwen.text.token_embeddings"],
    ["vision.prelude", "qwen.vision.prelude"],
  ];
  const mergers = new Map([
    [8, 0],
    [16, 1],
    [24, 2],
  ]);
  for (let index = 0; index < 27; index += 1) {
    stages.push([`vision.block.${index}`, `qwen.vision.block.${index}`]);
    const merger = mergers.get(index);
    if (merger !== undefined) {
      stages.push([
        `vision.deepstack_merger.${merger}`,
        `qwen.vision.deepstack_merger.${merger}`,
      ]);
    }
  }
  stages.push(["vision.final_merger", "qwen.vision.final_merger"]);
  for (let index = 0; index < 36; index += 1) {
    stages.push([
      `text.layer.${index}`,
      index < 3 ? `qwen.text.layer.${index}.post_deepstack` : `qwen.text.layer.${index}`,
    ]);
  }
  stages.push(["text.final_norm", "qwen.text.final_norm"]);
  if (stages.length !== BROWSER_1K5_QWEN_ALIGNED_STAGE_COUNT) {
    throw new Error(`internal Qwen qualification plan has ${stages.length} stages`);
  }
  return stages;
}

const QWEN_UNALIGNED_AUTHENTICATED_DIAGNOSTICS = Object.freeze([
  "qwen.text.layer.0",
  "qwen.text.layer.0.input_layernorm",
  "qwen.text.layer.0.mlp",
  "qwen.text.layer.0.mlp.down_proj",
  "qwen.text.layer.0.mlp.gate_proj",
  "qwen.text.layer.0.mlp.up_proj",
  "qwen.text.layer.0.post_attention_layernorm",
  "qwen.text.layer.0.self_attn.0",
  "qwen.text.layer.0.self_attn.k_norm",
  "qwen.text.layer.0.self_attn.k_proj",
  "qwen.text.layer.0.self_attn.o_proj",
  "qwen.text.layer.0.self_attn.q_norm",
  "qwen.text.layer.0.self_attn.q_proj",
  "qwen.text.layer.0.self_attn.v_proj",
  "qwen.text.layer.1",
  "qwen.text.layer.2",
  "qwen.vision.patch_embed",
]);

function validateVaeReferenceTraffic(requestEvents, workloadTelemetry, transportValidation) {
  const failures = [];
  const snapshot = workloadTelemetry.snapshot();
  const repeats = [];
  const successfulArtifactGets = requestEvents.filter(
    (event) =>
      event.method === "GET" &&
      (event.status === 200 || event.status === 206) &&
      isArtifactRoute(requestRoute(event.url)),
  );
  for (let index = 0; index < BROWSER_1K5_VAE_REFERENCE_REPEAT_COUNT; index += 1) {
    const phase = `repeat-${index}-start`;
    const startCount = snapshot.milestones.find(
      (entry) => entry.kind === "parity" && entry.name === phase,
    )?.count;
    const completeCount = snapshot.milestones.find(
      (entry) => entry.kind === "parity" && entry.name === `repeat-${index}-complete`,
    )?.count;
    if (startCount !== 1 || completeCount !== 1) {
      failures.push(
        `VAE repeat ${index} milestones are incomplete: start=${startCount}, complete=${completeCount}`,
      );
    }
    const events = successfulArtifactGets.filter(
      (event) =>
        event.parity_milestone === phase &&
        (event.artifact_components?.length ?? 0) > 0,
    );
    const bytes = events.reduce(
      (total, event) => total + event.response_content_length_bytes,
      0,
    );
    const unexpected = events.filter(
      (event) => !event.artifact_components.includes("flux-vae-encoder"),
    );
    if (bytes !== transportValidation.vae_encoder_weight_bytes) {
      failures.push(
        `VAE repeat ${index} fetched ${bytes} component bytes, expected exactly ${transportValidation.vae_encoder_weight_bytes}`,
      );
    }
    if (unexpected.length > 0) {
      failures.push(
        `VAE repeat ${index} fetched physical parts not attributed to the encoder: ${JSON.stringify(unexpected.map((event) => event.artifact_components))}`,
      );
    }
    repeats.push({
      index,
      successful_component_get_requests: events.length,
      response_content_length_bytes: bytes,
      components: Array.from(new Set(events.flatMap((event) => event.artifact_components))).sort(),
    });
  }
  const executedWeightEvents = successfulArtifactGets.filter(
    (event) => (event.artifact_components?.length ?? 0) > 0,
  );
  const outsideRepeats = executedWeightEvents.filter(
    (event) => !/^repeat-[0-2]-start$/.test(event.parity_milestone),
  );
  if (outsideRepeats.length > 0) {
    failures.push(
      `${outsideRepeats.length} model-component requests occurred outside an exact VAE repeat`,
    );
  }
  if (snapshot.dropped_milestone_names !== 0 || snapshot.dropped_http_aggregate_keys !== 0) {
    failures.push("bounded workload telemetry dropped VAE diagnostic evidence");
  }
  return {
    policy: "three-fresh-verified-flux-vae-encoder-loads-only",
    expected_repeats: BROWSER_1K5_VAE_REFERENCE_REPEAT_COUNT,
    expected_weight_bytes_per_repeat: transportValidation.vae_encoder_weight_bytes,
    repeats,
    validation_failures: failures,
    validated: failures.length === 0,
    telemetry: snapshot,
  };
}

function validateVaeReferenceReport(report, transportValidation) {
  const failures = [];
  if (!report || typeof report !== "object" || Array.isArray(report)) {
    throw new Error("browser VAE reference report is not an object");
  }
  expectReportEqual(failures, "report_schema_version", report.report_schema_version, 2);
  expectReportEqual(
    failures,
    "mode",
    report.mode,
    "diagnostic-no-surface-exact-1k5-vae-reference-three-repeat",
  );
  expectReportEqual(failures, "model_backend", report.model_backend, "raw-cubecl-no-fusion");
  expectReportEqual(failures, "adapter_backend", report.adapter_backend, "BrowserWebGpu");
  if (!["Other", "IntegratedGpu", "DiscreteGpu", "VirtualGpu"].includes(report.adapter_device_type)) {
    failures.push(`adapter_device_type=${reportValue(report.adapter_device_type)} is not a GPU`);
  }
  if (typeof report.adapter_name !== "string") failures.push("adapter_name is not a string");
  for (const field of ["adapter_shader_f16", "device_shader_f16"]) {
    if (typeof report[field] !== "boolean") failures.push(`${field} is not boolean`);
  }
  expectReportEqual(
    failures,
    "minimum_required_device_buffer_limit",
    report.minimum_required_device_buffer_limit,
    BROWSER_1K5_MIN_REQUIRED_BUFFER_LIMIT_BYTES,
  );
  for (const field of ["actual_storage_buffer_binding_size", "actual_max_buffer_size"]) {
    expectReportNumber(
      failures,
      field,
      report[field],
      (value) => value >= BROWSER_1K5_MIN_RUNTIME_BUFFER_BYTES,
      `at least ${BROWSER_1K5_MIN_RUNTIME_BUFFER_BYTES}`,
    );
  }
  expectReportEqual(failures, "model", report.model, CANONICAL_MODEL);
  expectReportEqual(failures, "model_revision", report.model_revision, CANONICAL_MODEL_REVISION);
  expectReportEqual(
    failures,
    "artifact_content_digest",
    report.artifact_content_digest,
    CANONICAL_ARTIFACT_CONTENT_DIGEST,
  );
  expectReportEqual(
    failures,
    "artifact_content_digest vs transport",
    report.artifact_content_digest,
    transportValidation.artifact_content_digest,
  );
  expectReportEqual(failures, "numeric_format", report.numeric_format, { other: CANONICAL_PROFILE });
  expectReportEqual(failures, "artifact_profile", report.artifact_profile, CANONICAL_PROFILE);
  expectReportEqual(failures, "vae_float_load_policy", report.vae_float_load_policy, "adapt-to-f32");
  expectReportEqual(failures, "vae_execution_dtype", report.vae_execution_dtype, "f32");
  expectReportEqual(
    failures,
    "expected_repeats",
    report.expected_repeats,
    BROWSER_1K5_VAE_REFERENCE_REPEAT_COUNT,
  );
  expectReportEqual(
    failures,
    "completed_repeats",
    report.completed_repeats,
    BROWSER_1K5_VAE_REFERENCE_REPEAT_COUNT,
  );

  const fixture = report.fixture ?? {};
  expectReportEqual(failures, "fixture.schema_version", fixture.schema_version, 2);
  expectReportEqual(failures, "fixture.variant", fixture.variant, "edit-turbo-1k5");
  expectReportEqual(failures, "fixture.model_revision", fixture.model_revision, CANONICAL_MODEL_REVISION);
  expectReportEqual(
    failures,
    "fixture.upstream_source_revision",
    fixture.upstream_source_revision,
    CANONICAL_UPSTREAM_SOURCE_REVISION,
  );
  expectReportEqual(failures, "fixture.width", fixture.width, 1536);
  expectReportEqual(failures, "fixture.height", fixture.height, 1536);
  expectReportEqual(failures, "fixture.seed", fixture.seed, 42);
  for (const key of Object.keys(VAE_REFERENCE_FIXTURE_FILES)) {
    validateFixtureFileIdentity(
      fixture[key],
      VAE_REFERENCE_FIXTURE_FILES[key],
      `fixture.${key}`,
      failures,
    );
  }
  const verification = report.fixture_verification ?? {};
  const expectedVerification = {
    verified_metadata_files: 1,
    verified_metadata_bytes: VAE_REFERENCE_FIXTURE_FILES.metadata.size,
    verified_metadata_sha256: VAE_REFERENCE_FIXTURE_FILES.metadata.sha256,
    verified_safetensors_headers: 1,
    verified_safetensors_header_bytes: transportValidation.fixture_safetensors_header_bytes,
    verified_safetensors_files: 1,
    verified_safetensors_file_bytes: VAE_REFERENCE_FIXTURE_FILES.tensors.size,
    verified_safetensors_sha256: VAE_REFERENCE_FIXTURE_FILES.tensors.sha256,
    verified_source_files: 1,
    verified_source_bytes: VAE_REFERENCE_FIXTURE_FILES.source.size,
    verified_source_sha256: VAE_REFERENCE_FIXTURE_FILES.source.sha256,
    verified_output_files: 1,
    verified_output_bytes: VAE_REFERENCE_FIXTURE_FILES.output.size,
    verified_output_sha256: VAE_REFERENCE_FIXTURE_FILES.output.sha256,
    verified_tensors: ACTIVE_FIXTURE_TENSOR_COUNT,
    verified_tensor_bytes: transportValidation.fixture_expected_tensor_bytes,
    expected_tensors: ACTIVE_FIXTURE_TENSOR_COUNT,
    expected_tensor_bytes: transportValidation.fixture_expected_tensor_bytes,
  };
  for (const [field, expected] of Object.entries(expectedVerification)) {
    expectReportEqual(failures, `fixture_verification.${field}`, verification[field], expected);
  }

  validateTensorMetric(report.input, "input", failures, {
    name: "vae.reference_input",
    oracle: "vae.reference_input",
    shape: [1, 3, 256, 256],
  });
  validateTensorMetric(report.injected_epsilon, "injected_epsilon", failures, {
    name: "vae.reference_epsilon",
    oracle: "vae.reference_epsilon",
    shape: [1, 16, 32, 32],
  });
  if (report.input?.max_abs !== 0 || report.injected_epsilon?.max_abs !== 0) {
    failures.push("VAE input or epsilon is not exact");
  }
  const components = [
    ["moments", [1, 32, 32, 32]],
    ["mean", [1, 16, 32, 32]],
    ["logvar", [1, 16, 32, 32]],
    ["std", [1, 16, 32, 32]],
    ["raw_latent", [1, 16, 32, 32]],
    ["scaled_latent", [1, 16, 32, 32]],
  ];
  if (!Array.isArray(report.runs) || report.runs.length !== BROWSER_1K5_VAE_REFERENCE_REPEAT_COUNT) {
    failures.push(`runs has invalid count ${report.runs?.length}`);
  } else {
    for (const [runIndex, run] of report.runs.entries()) {
      expectReportEqual(failures, `runs[${runIndex}].index`, run.index, runIndex);
      expectReportNumber(
        failures,
        `runs[${runIndex}].elapsed_micros`,
        run.elapsed_micros,
        (value) => Number.isSafeInteger(value) && value > 0,
        "a positive safe integer",
      );
      for (const [field, prefix] of [
        ["f32_oracle", "vae.reference_f32_"],
        ["upstream_bf16_drift", "vae.reference_"],
      ]) {
        if (!Array.isArray(run[field]) || run[field].length !== components.length) {
          failures.push(`runs[${runIndex}].${field} has invalid count ${run[field]?.length}`);
          continue;
        }
        for (const [componentIndex, [component, shape]] of components.entries()) {
          const identity = `${prefix}${component}`;
          const metric = run[field][componentIndex];
          validateTensorMetric(metric, `runs[${runIndex}].${field}[${componentIndex}]`, failures, {
            name: identity,
            oracle: identity,
            shape,
          });
          if (field === "f32_oracle") {
            const componentEnvelope = BROWSER_WEBGPU_VAE_F32_ORACLE_ENVELOPE[component];
            if (metric?.max_abs > componentEnvelope.maximum_abs) {
              failures.push(
                `runs[${runIndex}].${field}[${componentIndex}] exceeds ${componentEnvelope.maximum_abs}`,
              );
            }
            if (
              componentEnvelope.maximum_rmse !== undefined &&
              (metric?.rmse > componentEnvelope.maximum_rmse ||
                metric?.cosine_similarity < componentEnvelope.minimum_cosine_similarity)
            ) {
              failures.push(
                `runs[${runIndex}].${field}[${componentIndex}] misses its browser WebGPU ${component === "moments" ? "moments gate" : "scaled-latent gate"}`,
              );
            }
          }
        }
      }
    }
  }

  const stability = report.stability ?? {};
  expectReportEqual(failures, "stability.baseline_repeat_index", stability.baseline_repeat_index, 0);
  expectReportEqual(failures, "stability.compared_repeats", stability.compared_repeats, 2);
  expectReportEqual(failures, "stability.compared_components", stability.compared_components, 12);
  expectReportEqual(failures, "stability.all_bitwise_exact", stability.all_bitwise_exact, true);
  for (const field of ["maximum_abs", "maximum_rmse"]) {
    expectReportEqual(failures, `stability.${field}`, stability[field], 0);
  }
  expectReportEqual(failures, "stability.minimum_cosine_similarity", stability.minimum_cosine_similarity, 1);
  if (!Array.isArray(stability.metrics) || stability.metrics.length !== 12) {
    failures.push(`stability.metrics has invalid count ${stability.metrics?.length}`);
  } else {
    for (const [index, metric] of stability.metrics.entries()) {
      const repeatIndex = 1 + Math.floor(index / components.length);
      const [component, shape] = components[index % components.length];
      expectReportEqual(failures, `stability.metrics[${index}].repeat_index`, metric.repeat_index, repeatIndex);
      expectReportEqual(failures, `stability.metrics[${index}].component`, metric.component, component);
      expectReportEqual(failures, `stability.metrics[${index}].shape`, metric.shape, shape);
      expectReportEqual(failures, `stability.metrics[${index}].bitwise_exact`, metric.bitwise_exact, true);
      expectReportEqual(failures, `stability.metrics[${index}].element_count`, metric.element_count, shape.reduce((a, b) => a * b, 1));
      for (const field of ["max_abs", "mean_abs", "rmse", "relative_rmse"]) {
        expectReportEqual(failures, `stability.metrics[${index}].${field}`, metric[field], 0);
      }
      expectReportEqual(failures, `stability.metrics[${index}].cosine_similarity`, metric.cosine_similarity, 1);
    }
  }

  const artifact = report.artifact_verification ?? {};
  expectReportEqual(failures, "artifact_verification.expected_repeats", artifact.expected_repeats, 3);
  expectReportEqual(
    failures,
    "artifact_verification.expected_unique_weight_objects",
    artifact.expected_unique_weight_objects,
    transportValidation.vae_encoder_weight_file_count,
  );
  expectReportEqual(
    failures,
    "artifact_verification.verified_unique_weight_objects",
    artifact.verified_unique_weight_objects,
    transportValidation.vae_encoder_weight_file_count,
  );
  expectReportEqual(
    failures,
    "artifact_verification.expected_weight_bytes_per_repeat",
    artifact.expected_weight_bytes_per_repeat,
    transportValidation.vae_encoder_weight_bytes,
  );
  expectReportEqual(
    failures,
    "artifact_verification.verified_weight_bytes_all_repeats",
    artifact.verified_weight_bytes_all_repeats,
    transportValidation.vae_encoder_weight_bytes * 3,
  );
  expectReportEqual(failures, "artifact_verification.missing_weight_objects", artifact.missing_weight_objects, []);
  expectReportEqual(failures, "artifact_verification.unexpected_verified_objects", artifact.unexpected_verified_objects, []);
  expectReportEqual(failures, "artifact_verification.verification_count_per_object_exact", artifact.verification_count_per_object_exact, true);
  expectReportEqual(failures, "artifact_verification.passed", artifact.passed, true);
  const expectedObjects = transportValidation.vae_encoder_weight_objects.map((entry) => ({
    ...entry,
    verification_count: 3,
  }));
  expectReportEqual(failures, "artifact_verification.objects", artifact.objects, expectedObjects);

  validateBrowserWebGpuVaeF32OracleEnvelope(
    report.browser_webgpu_vae_f32_oracle_envelope,
    "browser_webgpu_vae_f32_oracle_envelope",
    failures,
  );
  expectReportEqual(failures, "gate_failures", report.gate_failures, []);
  expectReportNumber(
    failures,
    "peak_wasm_linear_memory_bytes",
    report.peak_wasm_linear_memory_bytes,
    (value) => Number.isSafeInteger(value) && value > 0,
    "a positive safe integer",
  );
  for (const field of ["artifacts_verified", "fixture_authenticated", "diagnostic_passed"]) {
    expectReportEqual(failures, field, report[field], true);
  }
  expectReportEqual(failures, "numerical_parity_claimed", report.numerical_parity_claimed, false);
  if (failures.length > 0) {
    const error = new Error(`browser VAE reference report contract failed:\n${failures.join("\n")}`);
    error.parityReport = report;
    error.contractFailures = failures;
    throw error;
  }
  return report;
}

function validateCompleteParityReport(report, transportValidation) {
  const failures = [];
  if (!report || typeof report !== "object" || Array.isArray(report)) {
    throw new Error("browser parity report is not an object");
  }
  expectReportEqual(failures, "report_schema_version", report.report_schema_version, 2);
  expectReportEqual(
    failures,
    "mode",
    report.mode,
    "qualification-no-surface-exact-1k5-fixture",
  );
  expectReportEqual(failures, "model_backend", report.model_backend, "raw-cubecl-no-fusion");
  expectReportEqual(failures, "adapter_backend", report.adapter_backend, "BrowserWebGpu");
  if (!["Other", "IntegratedGpu", "DiscreteGpu", "VirtualGpu"].includes(report.adapter_device_type)) {
    failures.push(`adapter_device_type=${reportValue(report.adapter_device_type)} is not a GPU`);
  }
  if (typeof report.adapter_name !== "string") failures.push("adapter_name is not a string");
  if (typeof report.adapter_shader_f16 !== "boolean") {
    failures.push("adapter_shader_f16 is not boolean");
  }
  if (typeof report.device_shader_f16 !== "boolean") {
    failures.push("device_shader_f16 is not boolean");
  }
  expectReportEqual(
    failures,
    "minimum_required_device_buffer_limit",
    report.minimum_required_device_buffer_limit,
    BROWSER_1K5_MIN_REQUIRED_BUFFER_LIMIT_BYTES,
  );
  for (const field of ["actual_storage_buffer_binding_size", "actual_max_buffer_size"]) {
    expectReportNumber(
      failures,
      field,
      report[field],
      (value) => value >= BROWSER_1K5_MIN_RUNTIME_BUFFER_BYTES,
      `at least ${BROWSER_1K5_MIN_RUNTIME_BUFFER_BYTES}`,
    );
  }
  expectReportEqual(failures, "model", report.model, CANONICAL_MODEL);
  expectReportEqual(failures, "model_revision", report.model_revision, CANONICAL_MODEL_REVISION);
  expectReportEqual(
    failures,
    "artifact_content_digest",
    report.artifact_content_digest,
    CANONICAL_ARTIFACT_CONTENT_DIGEST,
  );
  expectReportEqual(
    failures,
    "artifact_content_digest vs transport",
    report.artifact_content_digest,
    transportValidation.artifact_content_digest,
  );
  expectReportEqual(failures, "numeric_format", report.numeric_format, { other: CANONICAL_PROFILE });
  expectReportEqual(failures, "artifact_profile", report.artifact_profile, CANONICAL_PROFILE);
  expectReportEqual(
    failures,
    "residency_policy",
    report.residency_policy,
    lowVramMode
      ? BROWSER_LOW_VRAM_RESIDENCY_POLICY
      : packedResidentMode
        ? BROWSER_PACKED_F16_RESIDENCY_POLICY
        : BROWSER_F32_QUALIFICATION_RESIDENCY_POLICY,
  );
  for (const field of [
    "qwen_float_load_policy",
    "vae_float_load_policy",
    "denoiser_float_load_policy",
  ]) {
    expectReportEqual(
      failures,
      field,
      report[field],
      packedResidentMode ? BROWSER_PACKED_F16_LOAD_POLICY : "adapt-to-f32",
    );
  }
  for (const field of ["qwen_execution_dtype", "vae_execution_dtype", "denoiser_execution_dtype"]) {
    expectReportEqual(failures, field, report[field], "f32");
  }
  expectReportEqual(
    failures,
    "denoiser_quantized_load_policy",
    report.denoiser_quantized_load_policy,
    lowVramMode ? "runtime-quantize-q8s-block32-f32" : "preserve",
  );
  expectReportEqual(
    failures,
    "weight_traffic_contract",
    report.weight_traffic_contract,
    lowVramMode
      ? BROWSER_LOW_VRAM_WEIGHT_TRAFFIC_CONTRACT
      : packedResidentMode
        ? BROWSER_PACKED_F16_WEIGHT_TRAFFIC_CONTRACT
        : BROWSER_F32_QUALIFICATION_WEIGHT_TRAFFIC_CONTRACT,
  );
  if (packedResidentMode) {
    expectReportEqual(
      failures,
      "denoiser_storage_policy",
      report.denoiser_storage_policy,
      BROWSER_PACKED_F16_STORAGE_POLICY,
    );
    expectReportEqual(
      failures,
      "denoiser_linear_execution_policy",
      report.denoiser_linear_execution_policy,
      BROWSER_PACKED_F16_LINEAR_POLICY,
    );
    const plan = report.resident_resource_plan ?? {};
    for (const [field, expected] of Object.entries(BROWSER_1K5_PACKED_F16_RESOURCE_PLAN)) {
      expectReportEqual(failures, `resident_resource_plan.${field}`, plan[field], expected);
    }
    for (const field of Object.keys(plan)) {
      if (!(field in BROWSER_1K5_PACKED_F16_RESOURCE_PLAN)) {
        failures.push(`resident_resource_plan.${field} is unexpected`);
      }
    }
    if (
      plan.packed_f16_weight_bytes + plan.f32_auxiliary_weight_bytes !==
      plan.resident_weight_bytes
    ) {
      failures.push("resident_resource_plan weight bytes do not sum exactly");
    }
    if (
      plan.resident_weight_bytes + plan.activation_reserve_bytes !==
      plan.conservative_planned_device_bytes
    ) {
      failures.push("resident_resource_plan conservative bytes do not sum exactly");
    }
  } else {
    expectReportEqual(
      failures,
      "resident_resource_plan",
      report.resident_resource_plan ?? null,
      null,
    );
  }
  expectReportEqual(
    failures,
    "on_device_quantized_execution_claimed",
    report.on_device_quantized_execution_claimed,
    false,
  );
  if (lowVramMode) {
    const plan = report.low_vram_resource_plan ?? {};
    failures.push(...validateBrowser1k5LowVramResourcePlan(plan));

    const audit = report.low_vram_denoiser_dtype_audit ?? {};
    expectReportEqual(
      failures,
      "low_vram_denoiser_dtype_audit.matches_inventory",
      audit.matches_inventory,
      true,
    );
    expectReportEqual(
      failures,
      "low_vram_denoiser_dtype_audit.unexpected_dtype_tensor_count",
      audit.unexpected_dtype_tensor_count,
      0,
    );
    for (const [auditField, planField] of [
      ["q8s_block32_f32_tensor_count", "expected_q8s_block32_f32_tensor_count"],
      ["f32_tensor_count", "expected_f32_tensor_count"],
      ["q8s_block32_f32_elements", "expected_q8s_block32_f32_elements"],
      ["f32_elements", "expected_f32_elements"],
      ["q8s_block32_f32_payload_bytes", "expected_q8s_block32_f32_payload_bytes"],
      ["f32_payload_bytes", "expected_f32_payload_bytes"],
    ]) {
      expectReportEqual(
        failures,
        `low_vram_denoiser_dtype_audit.${auditField}`,
        audit[auditField],
        plan[planField],
      );
    }
    expectReportEqual(
      failures,
      "low_vram_denoiser_dtype_audit.tensor_count",
      audit.tensor_count,
      audit.q8s_block32_f32_tensor_count + audit.f32_tensor_count,
    );
  } else {
    expectReportEqual(failures, "low_vram_resource_plan", report.low_vram_resource_plan, null);
    expectReportEqual(
      failures,
      "low_vram_denoiser_dtype_audit",
      report.low_vram_denoiser_dtype_audit,
      null,
    );
  }
  expectReportEqual(
    failures,
    "qwen_query_chunk_size",
    report.qwen_query_chunk_size,
    BROWSER_1K5_QWEN_QUERY_CHUNK_SIZE,
  );
  expectReportEqual(
    failures,
    "vae_attention_query_chunk_size",
    report.vae_attention_query_chunk_size,
    BROWSER_1K5_VAE_QUERY_CHUNK_SIZE,
  );
  expectReportEqual(
    failures,
    "vae_decode_policy",
    report.vae_decode_policy,
    BROWSER_1K5_VAE_DECODE_POLICY,
  );
  expectReportEqual(
    failures,
    "vae_decode_max_planned_buffer_bytes",
    report.vae_decode_max_planned_buffer_bytes,
    BROWSER_1K5_MIN_RUNTIME_BUFFER_BYTES,
  );
  expectReportEqual(
    failures,
    "denoiser_query_chunk_size",
    report.denoiser_query_chunk_size,
    BROWSER_1K5_DENOISER_QUERY_CHUNK_SIZE,
  );
  expectReportEqual(
    failures,
    "denoiser_residency",
    report.denoiser_residency,
    lowVramMode
      ? BROWSER_1K5_LOW_VRAM_DENOISER_RESIDENCY
      : packedResidentMode
        ? BROWSER_1K5_PACKED_F16_DENOISER_RESIDENCY
      : BROWSER_1K5_F32_QUALIFICATION_DENOISER_RESIDENCY,
  );
  expectReportEqual(
    failures,
    "denoiser_expected_retained_stages",
    report.denoiser_expected_retained_stages,
    BROWSER_1K5_DENOISER_RETAINED_STAGE_COUNT,
  );
  expectReportEqual(
    failures,
    "denoiser_retained_stages_before_clear",
    report.denoiser_retained_stages_before_clear,
    BROWSER_1K5_DENOISER_RETAINED_STAGE_COUNT,
  );
  expectReportEqual(
    failures,
    "denoiser_cache_cleared_before_decode",
    report.denoiser_cache_cleared_before_decode,
    true,
  );

  const fixture = report.fixture ?? {};
  expectReportEqual(failures, "fixture.schema_version", fixture.schema_version, 2);
  expectReportEqual(failures, "fixture.variant", fixture.variant, "edit-turbo-1k5");
  expectReportEqual(
    failures,
    "fixture.model_revision",
    fixture.model_revision,
    CANONICAL_MODEL_REVISION,
  );
  expectReportEqual(
    failures,
    "fixture.upstream_source_revision",
    fixture.upstream_source_revision,
    CANONICAL_UPSTREAM_SOURCE_REVISION,
  );
  expectReportEqual(failures, "fixture.width", fixture.width, 1536);
  expectReportEqual(failures, "fixture.height", fixture.height, 1536);
  expectReportEqual(failures, "fixture.seed", fixture.seed, 42);
  for (const key of Object.keys(FIXTURE_FILES)) {
    validateFixtureFileIdentity(fixture[key], FIXTURE_FILES[key], `fixture.${key}`, failures);
  }

  const verification = report.fixture_verification ?? {};
  const expectedVerification = {
    verified_metadata_files: 1,
    verified_metadata_bytes: FIXTURE_FILES.metadata.size,
    verified_metadata_sha256: FIXTURE_FILES.metadata.sha256,
    verified_safetensors_headers: 1,
    verified_safetensors_header_bytes: transportValidation.fixture_safetensors_header_bytes,
    verified_safetensors_files: 1,
    verified_safetensors_file_bytes: FIXTURE_FILES.tensors.size,
    verified_safetensors_sha256: FIXTURE_FILES.tensors.sha256,
    verified_source_files: 1,
    verified_source_bytes: FIXTURE_FILES.source.size,
    verified_source_sha256: FIXTURE_FILES.source.sha256,
    verified_output_files: 1,
    verified_output_bytes: FIXTURE_FILES.output.size,
    verified_output_sha256: FIXTURE_FILES.output.sha256,
    verified_tensors: FIXTURE_TENSOR_COUNT,
    verified_tensor_bytes: transportValidation.fixture_expected_tensor_bytes,
    expected_tensors: FIXTURE_TENSOR_COUNT,
    expected_tensor_bytes: transportValidation.fixture_expected_tensor_bytes,
  };
  for (const [field, expected] of Object.entries(expectedVerification)) {
    expectReportEqual(failures, `fixture_verification.${field}`, verification[field], expected);
  }

  const processing = report.processing ?? {};
  for (const field of ["prompt_exact", "dimensions_exact", "seed_exact"]) {
    expectReportEqual(failures, `processing.${field}`, processing[field], true);
  }
  expectReportEqual(failures, "processing.effective_instruction_length", processing.effective_instruction_length, 147);
  const integerPlan = new Map([
    ["processor.input_ids", [1, 147]],
    ["processor.attention_mask", [1, 147]],
    ["qwen.attention_mask", [1, 147]],
    ["processor.mm_token_type_ids", [1, 147]],
    ["processor.image_grid_thw", [1, 3]],
  ]);
  if (!Array.isArray(processing.integer_tensors) || processing.integer_tensors.length !== integerPlan.size) {
    failures.push(`processing.integer_tensors has invalid count ${processing.integer_tensors?.length}`);
  } else {
    const observed = new Set();
    for (const [index, metric] of processing.integer_tensors.entries()) {
      const path = `processing.integer_tensors[${index}]`;
      const shape = integerPlan.get(metric?.name);
      if (!shape || observed.has(metric.name)) failures.push(`${path}.name is unknown or duplicated`);
      observed.add(metric?.name);
      expectReportEqual(failures, `${path}.shape`, metric?.shape, shape);
      expectReportEqual(failures, `${path}.elements`, metric?.elements, shape?.reduce((a, b) => a * b, 1));
      expectReportEqual(failures, `${path}.exact`, metric?.exact, true);
    }
  }
  validateTensorMetric(processing.pixel_values, "processing.pixel_values", failures, {
    name: "processing.pixel_values",
    oracle: "processor.pixel_values",
    shape: [256, 1536],
  });
  validateTensorMetric(processing.mrope_cos, "processing.mrope_cos", failures, {
    name: "processing.mrope_cos",
    oracle: "qwen.text.rope.0",
    shape: [1, 147, 128],
  });
  validateTensorMetric(processing.mrope_sin, "processing.mrope_sin", failures, {
    name: "processing.mrope_sin",
    oracle: "qwen.text.rope.1",
    shape: [1, 147, 128],
  });

  const qwen = report.qwen ?? {};
  expectReportEqual(
    failures,
    "qwen.expected_aligned_stages",
    qwen.expected_aligned_stages,
    BROWSER_1K5_QWEN_ALIGNED_STAGE_COUNT,
  );
  expectReportEqual(
    failures,
    "qwen.compared_aligned_stages",
    qwen.compared_aligned_stages,
    BROWSER_1K5_QWEN_ALIGNED_STAGE_COUNT,
  );
  const expectedQwenStages = expectedQwenAlignedStages();
  if (!Array.isArray(qwen.aligned_stages) || qwen.aligned_stages.length !== BROWSER_1K5_QWEN_ALIGNED_STAGE_COUNT) {
    failures.push(`qwen.aligned_stages has invalid count ${qwen.aligned_stages?.length}`);
  } else {
    for (const [index, metric] of qwen.aligned_stages.entries()) {
      const [name, oracle] = expectedQwenStages[index];
      validateTensorMetric(metric, `qwen.aligned_stages[${index}]`, failures, { name, oracle });
    }
  }
  validateTensorMetric(qwen.final_hidden_state, "qwen.final_hidden_state", failures, {
    name: "conditioning.qwen_final_hidden_state",
    oracle: "qwen.last_hidden_state",
    shape: [1, 147, 4096],
  });
  expectReportEqual(
    failures,
    "qwen.authenticated_unaligned_diagnostics",
    qwen.authenticated_unaligned_diagnostics,
    QWEN_UNALIGNED_AUTHENTICATED_DIAGNOSTICS,
  );

  const vae = report.vae_reference ?? {};
  validateTensorMetric(vae.input, "vae_reference.input", failures, {
    name: "vae.reference_input",
    oracle: "vae.reference_input",
    shape: [1, 3, 256, 256],
  });
  validateTensorMetric(vae.injected_epsilon, "vae_reference.injected_epsilon", failures, {
    name: "vae.reference_epsilon",
    oracle: "vae.reference_epsilon",
    shape: [1, 16, 32, 32],
  });
  for (const [field, prefix] of [
    ["f32_oracle", "vae.reference_f32_"],
    ["upstream_bf16_drift", "vae.reference_"],
  ]) {
    if (!Array.isArray(vae[field]) || vae[field].length !== 6) {
      failures.push(`vae_reference.${field} has invalid count ${vae[field]?.length}`);
    } else {
      for (const [index, metric] of vae[field].entries()) {
        validateTensorMetric(metric, `vae_reference.${field}[${index}]`, failures);
        if (!metric.name.startsWith(prefix)) {
          failures.push(`vae_reference.${field}[${index}].name has the wrong policy prefix`);
        }
      }
    }
  }

  expectReportEqual(
    failures,
    "denoiser_expected_boundaries",
    report.denoiser_expected_boundaries,
    BROWSER_1K5_DENOISER_BOUNDARY_COUNT,
  );
  expectReportEqual(
    failures,
    "denoiser_compared_boundaries",
    report.denoiser_compared_boundaries,
    BROWSER_1K5_DENOISER_BOUNDARY_COUNT,
  );
  if (
    !Array.isArray(report.denoiser_boundaries) ||
    report.denoiser_boundaries.length !== BROWSER_1K5_DENOISER_BOUNDARY_COUNT
  ) {
    failures.push(`denoiser_boundaries has invalid count ${report.denoiser_boundaries?.length}`);
  } else {
    const oracles = new Set();
    const perStep = [0, 0, 0, 0];
    for (const [index, metric] of report.denoiser_boundaries.entries()) {
      validateTensorMetric(metric, `denoiser_boundaries[${index}]`, failures);
      const match = /^denoiser\.step\.([0-3])\./.exec(metric.oracle);
      if (!match || metric.name !== metric.oracle || oracles.has(metric.oracle)) {
        failures.push(`denoiser_boundaries[${index}] has invalid or duplicate exact oracle identity`);
      } else {
        perStep[Number(match[1])] += 1;
      }
      oracles.add(metric.oracle);
    }
    expectReportEqual(failures, "denoiser_boundaries per-step counts", perStep, [59, 59, 59, 59]);
  }

  const dmd = report.dmd ?? {};
  validateTensorMetric(dmd.initial_latent, "dmd.initial_latent", failures, {
    name: "trajectory.initial_latent",
    oracle: "dmd.initial_latents",
    shape: [1, 16, 192, 192],
  });
  if (!Array.isArray(dmd.steps) || dmd.steps.length !== BROWSER_1K5_DMD_STEP_COUNT) {
    failures.push(`dmd.steps has invalid count ${dmd.steps?.length}`);
  } else {
    for (const [index, step] of dmd.steps.entries()) {
      expectReportEqual(failures, `dmd.steps[${index}].index`, step.index, index);
      expectReportEqual(failures, `dmd.steps[${index}].sigma_exact`, step.sigma_exact, true);
      expectReportEqual(
        failures,
        `dmd.steps[${index}].schedule_sigma`,
        step.schedule_sigma,
        step.fixture_sigma,
      );
      for (const field of ["input", "velocity", "prediction"]) {
        validateTensorMetric(step[field], `dmd.steps[${index}].${field}`, failures, {
          shape: [1, 16, 192, 192],
        });
      }
      for (const field of ["injected_noise", "renoised"]) {
        if (index < BROWSER_1K5_DMD_STEP_COUNT - 1) {
          validateTensorMetric(step[field], `dmd.steps[${index}].${field}`, failures, {
            shape: [1, 16, 192, 192],
          });
        } else {
          expectReportEqual(failures, `dmd.steps[${index}].${field}`, step[field], null);
        }
      }
    }
  }
  validateTensorMetric(dmd.final_latent, "dmd.final_latent", failures, {
    name: "trajectory.final_latent",
    oracle: "dmd.final_latents",
    shape: [1, 16, 192, 192],
  });
  validateTensorMetric(report.decode_input, "decode_input", failures, {
    name: "vae.decode_input",
    oracle: "vae.decode_input",
    shape: [1, 16, 192, 192],
  });
  validateTensorMetric(report.decoded_tensor, "decoded_tensor", failures, {
    name: "full_chain_output.decoded_tensor",
    oracle: "vae.decode_output",
    shape: [1, 3, 1536, 1536],
  });
  const rgb = report.final_rgb ?? {};
  expectReportEqual(failures, "final_rgb.width", rgb.width, 1536);
  expectReportEqual(failures, "final_rgb.height", rgb.height, 1536);
  for (const field of [
    "max_abs_u8",
    "mean_abs_u8",
    "rmse_u8",
    "psnr_db",
    "mean_block_ssim_8x8",
    "exact_fraction",
  ]) {
    if (!Number.isFinite(rgb[field])) failures.push(`final_rgb.${field} is not finite`);
  }
  expectReportEqual(
    failures,
    "fixture_output_png_sha256",
    report.fixture_output_png_sha256,
    FIXTURE_FILES.output.sha256,
  );
  expectReportNumber(
    failures,
    "peak_wasm_linear_memory_bytes",
    report.peak_wasm_linear_memory_bytes,
    (value) => Number.isSafeInteger(value) && value > 0,
    "a positive safe integer",
  );

  const gates = report.gates ?? {};
  expectReportEqual(failures, "gates.passed", gates.passed, true);
  expectReportEqual(failures, "gates.failures", gates.failures, []);
  const expectedGates = {
    qwen_aligned_stages: [0.2, 0.99],
    qwen_final: [0.1, 0.995],
    denoiser_boundaries: [0.265_799_3, 0.964_623_4],
    dmd_boundaries: [0.13, 0.992],
    dmd_final: [0.085, 0.996],
    propagated_decode: [0.09, 0.996],
  };
  for (const [field, [maximumRelativeRmse, minimumCosine]] of Object.entries(expectedGates)) {
    expectReportClose(
      failures,
      `gates.${field}.maximum_relative_rmse`,
      gates[field]?.maximum_relative_rmse,
      maximumRelativeRmse,
    );
    expectReportClose(
      failures,
      `gates.${field}.minimum_cosine_similarity`,
      gates[field]?.minimum_cosine_similarity,
      minimumCosine,
    );
  }
  validateBrowserWebGpuVaeF32OracleEnvelope(
    gates.browser_webgpu_vae_f32_oracle_envelope,
    "gates.browser_webgpu_vae_f32_oracle_envelope",
    failures,
  );
  expectReportClose(
    failures,
    "gates.final_rgb.minimum_psnr_db",
    gates.final_rgb?.minimum_psnr_db,
    lowVramMode ? 24.0 : 33.5,
  );
  expectReportClose(
    failures,
    "gates.final_rgb.minimum_mean_block_ssim_8x8",
    gates.final_rgb?.minimum_mean_block_ssim_8x8,
    lowVramMode ? 0.90 : 0.99,
  );

  // Rust emits this metric as f32, while serde's shortest JSON representation of
  // f32::EPSILON (`1.1920929e-7`) parses to a slightly larger f64 in JavaScript.
  // Round the transported value back to its declared dtype before applying the
  // exact f32 gate; this does not widen the numerical tolerance.
  if (Math.fround(processing.pixel_values?.max_abs) > F32_EPSILON) {
    failures.push("processing.pixel_values exceeds F32 epsilon");
  }
  for (const [metric, path] of [
    [processing.mrope_cos, "processing.mrope_cos"],
    [processing.mrope_sin, "processing.mrope_sin"],
  ]) {
    validateTensorGate(metric, gates.qwen_aligned_stages ?? {}, path, failures);
  }
  for (const [index, metric] of (qwen.aligned_stages ?? []).entries()) {
    validateTensorGate(metric, gates.qwen_aligned_stages ?? {}, `qwen.aligned_stages[${index}]`, failures);
  }
  validateTensorGate(qwen.final_hidden_state, gates.qwen_final ?? {}, "qwen.final_hidden_state", failures);
  for (const [metric, path] of [
    [vae.input, "vae_reference.input"],
    [vae.injected_epsilon, "vae_reference.injected_epsilon"],
    [dmd.initial_latent, "dmd.initial_latent"],
  ]) {
    if (metric?.max_abs !== 0) failures.push(`${path} is not an exact injected tensor`);
  }
  for (const [index, metric] of (vae.f32_oracle ?? []).entries()) {
    const suffix = metric?.name?.replace("vae.reference_f32_", "");
    const componentEnvelope = BROWSER_WEBGPU_VAE_F32_ORACLE_ENVELOPE[suffix];
    if (componentEnvelope === undefined || metric.max_abs > componentEnvelope.maximum_abs) {
      failures.push(`vae_reference.f32_oracle[${index}] misses its component maximum`);
    }
    if (
      suffix === "moments" &&
      (metric.rmse > componentEnvelope.maximum_rmse ||
        metric.cosine_similarity < componentEnvelope.minimum_cosine_similarity)
    ) {
      failures.push("VAE moments miss the F32 oracle gate");
    }
    if (
      suffix === "scaled_latent" &&
      (metric.max_abs > componentEnvelope.maximum_abs ||
        metric.rmse > componentEnvelope.maximum_rmse ||
        metric.cosine_similarity < componentEnvelope.minimum_cosine_similarity)
    ) {
      failures.push("VAE scaled latent misses its browser WebGPU scaled-latent gate");
    }
  }
  for (const [index, metric] of (report.denoiser_boundaries ?? []).entries()) {
    validateTensorGate(metric, gates.denoiser_boundaries ?? {}, `denoiser_boundaries[${index}]`, failures);
  }
  for (const [index, step] of (dmd.steps ?? []).entries()) {
    for (const field of ["input", "velocity", "prediction", "renoised"]) {
      if (step?.[field]) {
        validateTensorGate(step[field], gates.dmd_boundaries ?? {}, `dmd.steps[${index}].${field}`, failures);
      }
    }
    if (step?.injected_noise && step.injected_noise.max_abs !== 0) {
      failures.push(`dmd.steps[${index}].injected_noise is not exact`);
    }
  }
  validateTensorGate(dmd.final_latent, gates.dmd_final ?? {}, "dmd.final_latent", failures);
  validateTensorGate(report.decode_input, gates.dmd_final ?? {}, "decode_input", failures);
  validateTensorGate(report.decoded_tensor, gates.propagated_decode ?? {}, "decoded_tensor", failures);
  if (
    rgb.psnr_db < gates.final_rgb?.minimum_psnr_db ||
    rgb.mean_block_ssim_8x8 < gates.final_rgb?.minimum_mean_block_ssim_8x8
  ) {
    failures.push("final_rgb misses the published PSNR/SSIM gate");
  }
  for (const field of ["artifacts_verified", "fixture_authenticated", "numerical_parity_claimed"]) {
    expectReportEqual(failures, field, report[field], true);
  }
  if (failures.length > 0) {
    const error = new Error(`browser parity report contract failed:\n${failures.join("\n")}`);
    error.parityReport = report;
    error.contractFailures = failures;
    throw error;
  }
  return report;
}

function terminalFailure(status, fallback) {
  let parityReport;
  const payload = status.startsWith(TERMINAL_FAILED)
    ? status.slice(TERMINAL_FAILED.length).trim()
    : "";
  if (payload.startsWith("{")) {
    try {
      parityReport = JSON.parse(payload);
    } catch (error) {
      const failure = new Error(`browser parity emitted invalid failure report JSON: ${error}`);
      failure.failurePayload = payload;
      return failure;
    }
  }
  const error = new Error(`browser parity failed: ${status || fallback || "unknown failure"}`);
  if (parityReport) error.parityReport = parityReport;
  return error;
}

function attachPartialBrowserState(error, pageSnapshot, terminalFailure = undefined) {
  const failure = error instanceof Error ? error : new Error(String(error));
  failure.pageSnapshot ??= pageSnapshot ?? null;
  failure.browserTerminalFailure ??= terminalFailure;
  const partialReport = pageSnapshot?.exported?.report;
  if (partialReport && typeof partialReport === "object") {
    failure.parityReport ??= partialReport;
  }
  return failure;
}

async function waitForResult(cdp, browser, deadline, transportValidation) {
  let lastSnapshot;
  const throwTerminalFailure = () => {
    if (cdp.terminalError) {
      throw attachPartialBrowserState(cdp.terminalError, lastSnapshot, cdp.terminalFailure);
    }
    if (cdp.fatalError) {
      throw attachPartialBrowserState(cdp.fatalError, lastSnapshot);
    }
  };
  while (Date.now() < deadline) {
    throwIfInterrupted();
    if (browser.child.exitCode !== null || browser.child.signalCode !== null) {
      throw attachPartialBrowserState(
        new Error(
          `Chrome exited during parity (code=${browser.child.exitCode}, signal=${browser.child.signalCode})`,
        ),
        lastSnapshot,
      );
    }
    throwTerminalFailure();

    try {
      lastSnapshot = await evaluateValue(
        cdp,
        `({
          title: document.title,
          status: document.querySelector("#status")?.textContent ?? "",
          exported: globalThis.__burnImageHeadlessParity ?? null,
        })`,
      );
    } catch (error) {
      throwTerminalFailure();
      throw attachPartialBrowserState(error, lastSnapshot);
    }
    const status = lastSnapshot?.status ?? "";
    if (status.startsWith(TERMINAL_FAILED)) {
      throw attachPartialBrowserState(
        terminalFailure(status, lastSnapshot?.exported?.error),
        lastSnapshot,
      );
    }
    if (lastSnapshot?.exported?.stage === "failed") {
      throw attachPartialBrowserState(
        terminalFailure(status, lastSnapshot.exported?.error),
        lastSnapshot,
      );
    }
    if (status.startsWith(TERMINAL_OK)) {
      let report;
      try {
        report = JSON.parse(status.slice(TERMINAL_OK.length));
      } catch (error) {
        throw new Error(`browser parity emitted invalid report JSON: ${error}`);
      }
      if (vaeReferenceMode) validateVaeReferenceReport(report, transportValidation);
      else validateCompleteParityReport(report, transportValidation);
      if (
        cdp.pageErrors.length > 0 ||
        cdp.gpuErrors.length > 0 ||
        cdp.droppedPageErrors > 0 ||
        cdp.droppedGpuErrors > 0
      ) {
        throw new Error(
          `browser parity emitted errors (dropped page=${cdp.droppedPageErrors}, GPU=${cdp.droppedGpuErrors}):\n${[...cdp.pageErrors, ...cdp.gpuErrors].join("\n")}`,
        );
      }
      return report;
    }
    const terminalError = await cdp.waitForTerminalError(1_000);
    if (terminalError) {
      throw attachPartialBrowserState(terminalError, lastSnapshot, cdp.terminalFailure);
    }
  }
  throw attachPartialBrowserState(
    new Error(`browser parity timed out; last page state: ${JSON.stringify(lastSnapshot)}`),
    lastSnapshot,
  );
}

async function capturePagePng(cdp, path) {
  const capture = await cdp.call("Page.captureScreenshot", {
    format: "png",
    captureBeyondViewport: true,
    fromSurface: true,
  });
  if (!capture?.data) throw new Error("Chrome returned an empty page screenshot");
  await writeFile(path, Buffer.from(capture.data, "base64"));
}

async function captureResultPng(cdp, path) {
  const dataUrl = await evaluateValue(
    cdp,
    `(async () => {
      const image = document.querySelector("#burn-image-headless-result");
      if (!image?.src) return null;
      const blob = await fetch(image.src).then((response) => {
        if (!response.ok) throw new Error("result image fetch failed: HTTP " + response.status);
        return response.blob();
      });
      return await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(reader.error ?? new Error("result image FileReader failed"));
        reader.readAsDataURL(blob);
      });
    })()`,
    true,
  );
  if (typeof dataUrl !== "string") return false;
  const match = /^data:image\/png;base64,(.+)$/.exec(dataUrl);
  if (!match) throw new Error("browser result image is not a base64 PNG data URL");
  await writeFile(path, Buffer.from(match[1], "base64"));
  return true;
}

function chromeDiagnostics(browser) {
  if (!browser) return "";
  const output = [];
  if (browser.child.capturedStdout?.trim()) {
    output.push(`Chrome stdout:\n${browser.child.capturedStdout.trim()}`);
  }
  if (browser.child.capturedStderr?.trim()) {
    output.push(`Chrome stderr:\n${browser.child.capturedStderr.trim()}`);
  }
  return output.join("\n");
}

async function main() {
  const timeoutMs = parseTimeout();
  const validateOnly = process.env[VALIDATE_ONLY_ENV] === "1";
  const headful = process.env[HEADFUL_ENV] === "1";
  const artifactDir = resolve(
    process.env[ARTIFACT_DIR_ENV] ??
      join(
        repoRoot,
        ".artifacts/cdn-upload-modular/aberration.technology/model/boogu-image-0.1-edit-turbo-1k5",
      ),
  );
  const fixtureDir = resolve(
    process.env[FIXTURE_DIR_ENV] ??
      (vaeReferenceMode
        ? "/tmp/boogu-1k5-bf16-1536"
        : "/tmp/boogu-1k5-bf16-1536-exhaustive"),
  );
  const wwwOutDir = resolve(
    process.env[WWW_OUT_DIR_ENV] ?? join(repoRoot, "crates/bevy_image/www/out"),
  );
  const outputDir = resolve(
    process.env[OUTPUT_DIR_ENV] ??
      join(tmpdir(), `burn-image-browser-1k5-${WORKLOAD_NAME}-${process.pid}`),
  );
  await mkdir(outputDir, { recursive: true });
  console.log(`burn_image 1.5K browser ${WORKLOAD_NAME} output: ${outputDir}`);

  const requestEvents = [];
  const browserEvents = [];
  const denoiserResidencyPolicy = fullParityMode
    ? denoiserResidencyPolicyForMode(residencySelector)
    : null;
  const workloadTelemetry = createWorkloadTelemetry(denoiserResidencyPolicy);
  const browserPackageIdentity = await collectBrowserPackageIdentity({
    wwwOutDir,
    repoRoot,
    testsDir,
    harnessScriptPath: scriptPath,
  });
  const startedAt = Date.now();
  const deadline = startedAt + timeoutMs;
  let browser;
  let gpuMonitor;
  let gpuAttestation;
  let browserWebGpuScopeAttestation;
  let runtimeWebGpuCalls;
  let runtimeAdapterAttestation;
  let hostResourceMonitor;
  let hostResourceAttestation;
  let httpResidencyAttestation;
  let vaeReferenceTrafficAttestation;
  let chrome;
  let chromeArguments_;
  let chromeSharedMemory;
  let cdp;
  let adapter;
  let browserVersion;
  let profile;
  let server;
  let transportValidation;
  let parityReport;
  let pageSnapshot;
  let pagePngPath;
  let resultPngPath;
  let reportPath;
  let logPath;
  let outcome;
  let failure;
  let finalizationFailure;
  try {
    if (!browserPackageIdentity.validated) {
      throw new Error(
        `browser package identity validation failed:\n${browserPackageIdentity.validation_failures.join("\n")}`,
      );
    }
    for (const required of [
      join(testsDir, harnessFileName),
      join(wwwOutDir, "bevy_burn_image.js"),
      join(wwwOutDir, "bevy_burn_image_bg.wasm"),
      join(wwwOutDir, "burn-image-icon.png"),
      join(artifactDir, "manifest.json"),
      join(dirname(artifactDir), CANONICAL_QWEN_BUNDLE, "manifest.json"),
      join(dirname(artifactDir), CANONICAL_VAE_BUNDLE, "manifest.json"),
      join(fixtureDir, "metadata.json"),
      join(fixtureDir, "source.png"),
      join(fixtureDir, "output.png"),
      join(fixtureDir, "tensors.safetensors"),
    ]) {
      if (!existsSync(required)) {
        throw new Error(`required browser parity input is missing: ${required}`);
      }
    }

    const hosted = await createStaticServer(
      [
        { prefix: "/harness/", root: testsDir },
        { prefix: "/app/out/", root: wwwOutDir },
        { prefix: "/artifacts/", root: artifactDir },
        {
          prefix: `/${CANONICAL_QWEN_BUNDLE}/`,
          root: join(dirname(artifactDir), CANONICAL_QWEN_BUNDLE),
        },
        {
          prefix: `/${CANONICAL_VAE_BUNDLE}/`,
          root: join(dirname(artifactDir), CANONICAL_VAE_BUNDLE),
        },
        { prefix: "/fixture/", root: fixtureDir },
      ],
      requestEvents,
      workloadTelemetry,
    );
    server = hosted.server;
    // Loopback is a potentially trustworthy development origin and matches Chrome's native
    // Vulkan qualification path. Keep the server bound to loopback for both pages.
    const baseUrl = `http://127.0.0.1:${hosted.port}`;
    transportValidation = await validateServer(
      baseUrl,
      artifactDir,
      fixtureDir,
      workloadTelemetry,
    );
    const transportValidationFailures =
      validateBrowser1k5TransportValidation(transportValidation);
    if (transportValidationFailures.length > 0) {
      throw new Error(
        `browser 1.5K physical transport contract failed:\n${transportValidationFailures.join("\n")}`,
      );
    }
    if (validateOnly) {
      outcome = {
        schema_version: 1,
        test: `${WORKLOAD_TEST}_transport_validation`,
        ok: true,
        variant: "edit-turbo-1k5",
        residency: fullParityMode ? residencySelector : null,
        browser_package_identity: browserPackageIdentity,
        transport_validation: transportValidation,
        workload_traffic: workloadTelemetry.snapshot(),
        elapsed_ms: Date.now() - startedAt,
      };
      reportPath = join(outputDir, `burn-image-browser-1k5-${WORKLOAD_NAME}-report.json`);
      await writeFile(reportPath, `${JSON.stringify(outcome, null, 2)}\n`);
      console.log(JSON.stringify(outcome, null, 2));
      return;
    }
    // Static transport probes intentionally touch one weight in every bundle. Keep those probes
    // out of the execution-traffic ledger so the VAE and denoiser residency gates measure only
    // browser-runtime requests and still fail on any genuine pre-forward model load.
    requestEvents.length = 0;
    workloadTelemetry.resetForWorkload();

    const query = new URLSearchParams({
      headless: vaeReferenceMode ? "vae-reference" : "parity",
      variant: "edit-turbo-1k5",
      profile: "production",
      ...(fullParityMode
        ? {
            residency: lowVramMode
              ? "low-vram"
              : packedResidentMode
                ? "resident"
                : "qualification-f32",
          }
        : {}),
      artifacts: `${baseUrl}/artifacts`,
      fixture: `${baseUrl}/fixture`,
    });
    const harnessUrl = `${baseUrl}/harness/${harnessFileName}?${query}`;
    const probeUrl = `${baseUrl}/probe`;

    chromeSharedMemory = await inspectChromeSharedMemory();
    chrome = await findChrome();
    profile = await mkdtemp(join(tmpdir(), `burn-image-browser-1k5-${WORKLOAD_NAME}-profile-`));
    chromeArguments_ = chromeLaunchArguments(profile, probeUrl, headful, chromeSharedMemory);
    browser = await startChrome(chrome, chromeArguments_);
    const devToolsDeadline = Math.min(deadline, Date.now() + DEVTOOLS_START_TIMEOUT_MS);
    const devToolsPort = await readDevToolsPort(profile, browser, devToolsDeadline);
    const target = await findPageTarget(devToolsPort, probeUrl, browser, devToolsDeadline);
    cdp = await openCdp(target.webSocketDebuggerUrl, browserEvents, workloadTelemetry);
    browserVersion = await cdp.call("Browser.getVersion");
    // Chrome can publish the page target before the Vulkan WebGPU service has finished starting.
    // Avoid sampling navigator.gpu from that transient initialization window.
    await delay(3_000);
    adapter = await browserAdapterInfo(cdp);
    validateBrowser1k5AdapterLimits(adapter);
    // Start the PID-scoped monitor only after Chrome/CDP/adapter initialization. Navigation below
    // begins the parity workload, so startup/compositor samples cannot satisfy the GPU gate.
    hostResourceMonitor = await startHostResourceMonitor(browser.child.pid, workloadTelemetry);
    gpuMonitor = await startNativeGpuMonitor(
      browser.child.pid,
      vaeReferenceMode || lowVramMode ? 1 : MIN_F32_RETAINED_DENOISER_FRAMEBUFFER_MIB,
      vaeReferenceMode
        ? "real VAE encoder diagnostic"
        : lowVramMode
          ? "low-VRAM runtime-Q8 denoiser policy"
          : packedResidentMode
            ? "all-stage packed-F16 resident policy"
            : "F32 qualification retained-denoiser policy",
      lowVramMode ? BROWSER_1K5_LOW_VRAM_STRICT_DEVICE_CAP_BYTES : null,
    );
    const navigation = await cdp.call("Page.navigate", { url: harnessUrl });
    if (navigation?.errorText) {
      throw new Error(`Chrome could not navigate to the parity harness: ${navigation.errorText}`);
    }
    parityReport = await waitForResult(cdp, browser, deadline, transportValidation);
    runtimeWebGpuCalls = await evaluateValue(
      cdp,
      "globalThis.__burnImageHeadlessParity?.webgpu_calls ?? []",
    );
    runtimeAdapterAttestation = attestBrowser1k5RuntimeAdapter(
      parityReport,
      runtimeWebGpuCalls,
    );
    if (vaeReferenceMode) {
      vaeReferenceTrafficAttestation = validateVaeReferenceTraffic(
        requestEvents,
        workloadTelemetry,
        transportValidation,
      );
    } else {
      httpResidencyAttestation = workloadTelemetry.denoiserResidencyAttestation();
    }
    gpuAttestation = await gpuMonitor.stop();
    gpuMonitor = undefined;
    hostResourceAttestation = await hostResourceMonitor.stop();
    hostResourceMonitor = undefined;
    browserWebGpuScopeAttestation = attestCalibratedBrowserWebGpuScope(
      browserVersion,
      adapter,
      gpuAttestation,
      CALIBRATED_BROWSER_WEBGPU_RUNTIME_SCOPE,
    );
    const resourceFailures = [
      ...runtimeAdapterAttestation.validation_failures.map(
        (message) => `exact runtime WebGPU adapter: ${message}`,
      ),
      ...browserWebGpuScopeAttestation.validation_failures.map(
        (message) => `calibrated browser WebGPU scope: ${message}`,
      ),
      ...(vaeReferenceMode
        ? vaeReferenceTrafficAttestation.validation_failures.map(
            (message) => `VAE encoder traffic: ${message}`,
          )
        : [
            ...httpResidencyAttestation.validation_failures,
            ...validateDenoiserResidencyPolicy(httpResidencyAttestation, residencySelector),
          ].map((message) => `HTTP residency: ${message}`)),
      ...gpuAttestation.validation_failures.map((message) => `GPU residency: ${message}`),
      ...hostResourceAttestation.validation_failures.map((message) => `host resources: ${message}`),
    ];
    if (lowVramMode) {
      if (
        gpuAttestation.maximum_framebuffer_bytes_exclusive !==
        BROWSER_1K5_LOW_VRAM_STRICT_DEVICE_CAP_BYTES
      ) {
        resourceFailures.push(
          `GPU residency: low-VRAM exclusive ceiling is ${gpuAttestation.maximum_framebuffer_bytes_exclusive}, expected ${BROWSER_1K5_LOW_VRAM_STRICT_DEVICE_CAP_BYTES}`,
        );
      }
      if (
        !Number.isSafeInteger(gpuAttestation.observed_max_framebuffer_bytes) ||
        gpuAttestation.observed_max_framebuffer_bytes <= 0 ||
        gpuAttestation.observed_max_framebuffer_bytes >=
          BROWSER_1K5_LOW_VRAM_STRICT_DEVICE_CAP_BYTES
      ) {
        resourceFailures.push(
          `GPU residency: low-VRAM observed Chrome GPU peak ${gpuAttestation.observed_max_framebuffer_bytes} is not positive and strictly below ${BROWSER_1K5_LOW_VRAM_STRICT_DEVICE_CAP_BYTES}`,
        );
      }
    }
    if (resourceFailures.length > 0) {
      const error = new Error(
        `browser resource/residency attestation failed:\n${resourceFailures.join("\n")}`,
      );
      error.parityReport = parityReport;
      error.gpuAttestation = gpuAttestation;
      error.browserWebGpuScopeAttestation = browserWebGpuScopeAttestation;
      error.runtimeAdapterAttestation = runtimeAdapterAttestation;
      error.hostResourceAttestation = hostResourceAttestation;
      error.httpResidencyAttestation = httpResidencyAttestation;
      error.vaeReferenceTrafficAttestation = vaeReferenceTrafficAttestation;
      throw error;
    }

    if (!vaeReferenceMode) {
      resultPngPath = join(outputDir, "burn-image-browser-1k5-parity-output.png");
      const capturedResult = await captureResultPng(cdp, resultPngPath);
      if (!capturedResult) resultPngPath = null;
    }
    pagePngPath = join(outputDir, `burn-image-browser-1k5-${WORKLOAD_NAME}-page.png`);
    await capturePagePng(cdp, pagePngPath);
    outcome = {
      schema_version: 1,
      test: WORKLOAD_TEST,
      ok: true,
      variant: "edit-turbo-1k5",
      residency: fullParityMode ? residencySelector : null,
      browser_package_identity: browserPackageIdentity,
      ...browser1k5ChromeLaunchEvidence({
        executable: chrome,
        arguments: chromeArguments_,
        profile,
        sharedMemory: chromeSharedMemory,
      }),
      browser: browserVersion,
      native_webgpu_adapter: adapter,
      runtime_webgpu_calls: runtimeWebGpuCalls,
      runtime_webgpu_adapter_attestation: runtimeAdapterAttestation,
      browser_webgpu_scope_attestation: browserWebGpuScopeAttestation,
      native_gpu_attestation: gpuAttestation,
      host_resource_attestation: hostResourceAttestation,
      model_residency_http_attestation: httpResidencyAttestation,
      vae_encoder_http_attestation: vaeReferenceTrafficAttestation,
      transport_validation: transportValidation,
      elapsed_ms: Date.now() - startedAt,
      parity_report: parityReport,
      output_png: resultPngPath,
      page_png: pagePngPath,
    };
    reportPath = join(outputDir, `burn-image-browser-1k5-${WORKLOAD_NAME}-report.json`);
    await writeFile(reportPath, `${JSON.stringify(outcome, null, 2)}\n`);
    console.log(JSON.stringify(outcome, null, 2));
  } catch (error) {
    failure = error;
    parityReport ??= error?.parityReport;
    pageSnapshot ??= error?.pageSnapshot;
    gpuAttestation ??= error?.gpuAttestation;
    browserWebGpuScopeAttestation ??= error?.browserWebGpuScopeAttestation;
    runtimeWebGpuCalls ??= pageSnapshot?.exported?.webgpu_calls ?? [];
    runtimeAdapterAttestation ??= error?.runtimeAdapterAttestation;
    if (!runtimeAdapterAttestation && parityReport) {
      runtimeAdapterAttestation = attestBrowser1k5RuntimeAdapter(
        parityReport,
        runtimeWebGpuCalls,
      );
    }
    hostResourceAttestation ??= error?.hostResourceAttestation;
    httpResidencyAttestation ??= error?.httpResidencyAttestation;
    vaeReferenceTrafficAttestation ??= error?.vaeReferenceTrafficAttestation;
    if (gpuMonitor) {
      try {
        gpuAttestation = await gpuMonitor.stop();
        if (!gpuAttestation.validated) {
          browserEvents.push({
            at_ms: Date.now(),
            type: "gpu_attestation_error",
            message: gpuAttestation.validation_failures.join("; "),
          });
        }
      } catch (attestationError) {
        browserEvents.push({
          at_ms: Date.now(),
          type: "gpu_attestation_error",
          message: attestationError instanceof Error ? attestationError.message : String(attestationError),
        });
      }
      gpuMonitor = undefined;
    }
    if (hostResourceMonitor) {
      try {
        hostResourceAttestation = await hostResourceMonitor.stop();
        if (!hostResourceAttestation.validated) {
          browserEvents.push({
            at_ms: Date.now(),
            type: "host_resource_attestation_error",
            message: hostResourceAttestation.validation_failures.join("; "),
          });
        }
      } catch (attestationError) {
        browserEvents.push({
          at_ms: Date.now(),
          type: "host_resource_attestation_error",
          message: attestationError instanceof Error ? attestationError.message : String(attestationError),
        });
      }
      hostResourceMonitor = undefined;
    }
    if (vaeReferenceMode) {
      vaeReferenceTrafficAttestation ??= transportValidation
        ? validateVaeReferenceTraffic(requestEvents, workloadTelemetry, transportValidation)
        : null;
    } else {
      httpResidencyAttestation ??= workloadTelemetry.denoiserResidencyAttestation();
    }
    outcome = {
      schema_version: 1,
      test: WORKLOAD_TEST,
      ok: false,
      variant: "edit-turbo-1k5",
      residency: fullParityMode ? residencySelector : null,
      browser_package_identity: browserPackageIdentity,
      ...browser1k5ChromeLaunchEvidence({
        executable: chrome,
        arguments: chromeArguments_,
        profile,
        sharedMemory: chromeSharedMemory,
      }),
      browser: browserVersion ?? null,
      native_webgpu_adapter: adapter ?? null,
      runtime_webgpu_calls: runtimeWebGpuCalls,
      runtime_webgpu_adapter_attestation: runtimeAdapterAttestation ?? null,
      browser_webgpu_scope_attestation: browserWebGpuScopeAttestation ?? null,
      native_gpu_attestation: gpuAttestation ?? null,
      host_resource_attestation: hostResourceAttestation ?? null,
      model_residency_http_attestation: httpResidencyAttestation,
      vae_encoder_http_attestation: vaeReferenceTrafficAttestation,
      transport_validation: transportValidation ?? null,
      parity_report: parityReport ?? null,
      partial_page_state: pageSnapshot ?? null,
      browser_terminal_failure:
        error?.browserTerminalFailure ?? cdp?.terminalFailure ?? null,
      adapter_limit_failures: error?.adapterLimitFailures ?? null,
      page_errors: cdp?.pageErrors ?? [],
      gpu_errors: cdp?.gpuErrors ?? [],
      dropped_page_errors: cdp?.droppedPageErrors ?? 0,
      dropped_gpu_errors: cdp?.droppedGpuErrors ?? 0,
      elapsed_ms: Date.now() - startedAt,
      error: error instanceof Error ? error.stack ?? error.message : String(error),
    };
    reportPath = join(outputDir, `burn-image-browser-1k5-${WORKLOAD_NAME}-report.json`);
    await writeFile(reportPath, `${JSON.stringify(outcome, null, 2)}\n`);
    if (cdp) {
      pagePngPath = join(outputDir, `burn-image-browser-1k5-${WORKLOAD_NAME}-failure.png`);
      try {
        await capturePagePng(cdp, pagePngPath);
      } catch (captureError) {
        browserEvents.push({
          at_ms: Date.now(),
          type: "harness_error",
          message: `failure screenshot could not be captured: ${captureError}`,
        });
      }
    }
    throw error;
  } finally {
    if (gpuMonitor) {
      try {
        gpuAttestation ??= await gpuMonitor.stop();
      } catch (attestationError) {
        browserEvents.push({
          at_ms: Date.now(),
          type: "gpu_attestation_error",
          message: attestationError instanceof Error ? attestationError.message : String(attestationError),
        });
      }
    }
    if (hostResourceMonitor) {
      try {
        hostResourceAttestation ??= await hostResourceMonitor.stop();
      } catch (attestationError) {
        browserEvents.push({
          at_ms: Date.now(),
          type: "host_resource_attestation_error",
          message: attestationError instanceof Error ? attestationError.message : String(attestationError),
        });
      }
    }
    cdp?.close();
    const chromeCleanup = await stopChrome(browser);
    if (chromeCleanup?.errors.length > 0) {
      browserEvents.push({
        at_ms: Date.now(),
        type: "chrome_cleanup_error",
        message: chromeCleanup.errors.join("; "),
      });
    }
    await closeServer(server);
    const chromeLaunchEvidence = browser1k5ChromeLaunchEvidence({
      executable: chrome,
      arguments: chromeArguments_,
      profile,
      sharedMemory: chromeSharedMemory,
    });
    if (outcome) Object.assign(outcome, chromeLaunchEvidence);
    if (outcome?.test === WORKLOAD_TEST) {
      if (vaeReferenceMode) {
        vaeReferenceTrafficAttestation = transportValidation
          ? validateVaeReferenceTraffic(requestEvents, workloadTelemetry, transportValidation)
          : null;
      } else {
        httpResidencyAttestation = workloadTelemetry.denoiserResidencyAttestation();
      }
      outcome.native_gpu_attestation = gpuAttestation ?? null;
      outcome.runtime_webgpu_calls = runtimeWebGpuCalls ?? [];
      outcome.runtime_webgpu_adapter_attestation = runtimeAdapterAttestation ?? null;
      outcome.browser_webgpu_scope_attestation = browserWebGpuScopeAttestation ?? null;
      outcome.host_resource_attestation = hostResourceAttestation ?? null;
      outcome.model_residency_http_attestation = httpResidencyAttestation;
      outcome.vae_encoder_http_attestation = vaeReferenceTrafficAttestation;
      outcome.elapsed_ms = Date.now() - startedAt;
      const lateTrafficAttestation = vaeReferenceMode
        ? vaeReferenceTrafficAttestation
        : httpResidencyAttestation;
      const lateTrafficFailures = [
        ...(lateTrafficAttestation?.validation_failures ?? ["traffic attestation is missing"]),
        ...(!vaeReferenceMode && lateTrafficAttestation
          ? validateDenoiserResidencyPolicy(lateTrafficAttestation, residencySelector)
          : []),
      ];
      if (outcome.ok && lateTrafficFailures.length > 0) {
        finalizationFailure = new Error(
          `late model HTTP traffic invalidated ${WORKLOAD_NAME}:\n${lateTrafficFailures.join("\n")}`,
        );
        failure ??= finalizationFailure;
        outcome.ok = false;
        outcome.error = finalizationFailure.stack ?? finalizationFailure.message;
      }
      if (reportPath) await writeFile(reportPath, `${JSON.stringify(outcome, null, 2)}\n`);
    }
    logPath = join(outputDir, `burn-image-browser-1k5-${WORKLOAD_NAME}.log`);
    const diagnosticLog = [
      `test=${WORKLOAD_TEST}`,
      `ok=${outcome?.ok === true}`,
      `started_at_ms=${startedAt}`,
      `elapsed_ms=${Date.now() - startedAt}`,
      `chrome=${chrome ?? "not launched"}`,
      `chrome_arguments=${JSON.stringify(chromeArguments_ ?? [])}`,
      `chrome_shared_memory=${JSON.stringify(chromeSharedMemory ?? null)}`,
      `artifact_dir=${artifactDir}`,
      `fixture_dir=${fixtureDir}`,
      `www_out_dir=${wwwOutDir}`,
      `output_dir=${outputDir}`,
      `chrome_cleanup=${JSON.stringify(browser?.cleanup ?? null)}`,
      `failure=${failure instanceof Error ? failure.stack ?? failure.message : failure ?? ""}`,
      "browser_events=",
      ...browserEvents.map((event) => JSON.stringify(event)),
      "http_events=",
      ...requestEvents.map((event) => JSON.stringify(event)),
      chromeDiagnostics(browser),
      "",
    ].join("\n");
    await writeFile(logPath, diagnosticLog);
    if (profile) {
      await rm(profile, { recursive: true, force: true, maxRetries: 8, retryDelay: 100 });
    }
    for (const [signal, handler] of signalHandlers) process.off(signal, handler);
    console.log(`report: ${reportPath}`);
    console.log(`PNG: ${resultPngPath ?? pagePngPath ?? "not captured"}`);
    console.log(`log: ${logPath}`);
    if (finalizationFailure) throw finalizationFailure;
  }
}

main().catch((error) => {
  console.error(
    `burn_image 1.5K browser ${WORKLOAD_NAME} failed: ${error instanceof Error ? error.stack : String(error)}`,
  );
  process.exitCode = 1;
});
