// Bounded real-browser diagnostic for the current Turbo preloaded packed-F16 storage /
// dense-F32-per-semantic-stage execution policy
// using headless=turbo-first-dmd.
// Set BURN_IMAGE_TURBO_FIRST_DMD=1 for a real WebGPU run, or set
// BURN_IMAGE_TURBO_FIRST_DMD_VALIDATE_ONLY=1 as well to validate inputs and HTTP Range transport
// without launching Chrome or touching the GPU.

import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { constants as fsConstants, createReadStream, existsSync } from "node:fs";
import {
  access,
  mkdir,
  mkdtemp,
  open,
  readFile,
  realpath,
  rm,
  stat,
  statfs,
  writeFile,
} from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, extname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

import {
  TERMINAL_FAILED,
  TERMINAL_OK,
  TURBO_FIRST_DMD_CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES,
  TURBO_1K_FIXTURE,
  TURBO_ARTIFACT_CONTENT_DIGEST,
  TURBO_FIXTURE,
  TURBO_MODEL,
  TURBO_MODEL_REVISION,
  inspectTurboFirstDmdStorageAdmission,
  selectTurboFirstDmdChromeSharedMemoryPolicy,
  summarizeTurboFirstDmdWebGpuCalls,
  turboFirstDmdChromeLaunchEvidence,
  validateTurboFirstDmdReport,
  validateTurboFirstDmdSourceContract,
} from "./wasm_turbo_first_dmd_contract.mjs";

const ENABLE_ENV = "BURN_IMAGE_TURBO_FIRST_DMD";
const VALIDATE_ONLY_ENV = "BURN_IMAGE_TURBO_FIRST_DMD_VALIDATE_ONLY";
const ARTIFACT_ROOT_ENV = "BURN_IMAGE_TURBO_FIRST_DMD_ARTIFACT_ROOT";
const FIXTURE_DIR_ENV = "BURN_IMAGE_TURBO_FIRST_DMD_FIXTURE_DIR";
const FIXTURE_PROFILE_ENV = "BURN_IMAGE_TURBO_FIRST_DMD_FIXTURE_PROFILE";
const WWW_OUT_DIR_ENV = "BURN_IMAGE_TURBO_FIRST_DMD_WWW_OUT_DIR";
const OUTPUT_DIR_ENV = "BURN_IMAGE_TURBO_FIRST_DMD_OUTPUT_DIR";
const CHROME_ENV = "BURN_IMAGE_TURBO_FIRST_DMD_CHROME";
const TIMEOUT_ENV = "BURN_IMAGE_TURBO_FIRST_DMD_TIMEOUT_MS";
const DEFAULT_TIMEOUT_MS = 2 * 60 * 60 * 1000;
const DEVTOOLS_TIMEOUT_MS = 45_000;
const FIRST_MODEL_REQUEST_TIMEOUT_MS = 60_000;
const MAX_CHROME_LOG_BYTES = 2 * 1024 * 1024;
const MAX_SERVER_REQUEST_EVENTS = 20_000;
const CHROME_SHARED_MEMORY_PROBE_CHUNK_BYTES = 8 * 1024 * 1024;
const SERVER_CLOSE_TIMEOUT_MS = 5_000;
const CDP_CALL_TIMEOUT_MS = 30_000;

const testsDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(testsDir, "../../..");
const htmlPath = join(testsDir, "wasm_turbo_first_dmd_probe.html");
const harnessPath = fileURLToPath(import.meta.url);
const contractPath = join(testsDir, "wasm_turbo_first_dmd_contract.mjs");
const defaultWwwOut = join(repoRoot, "crates/bevy_image/www/out");

if (process.env[ENABLE_ENV] !== "1") {
  console.log(`burn_image Turbo first-DMD diagnostic: skipped (set ${ENABLE_ENV}=1)`);
  process.exit(0);
}

const validateOnly = process.env[VALIDATE_ONLY_ENV] === "1";
const timeoutMs = Number(process.env[TIMEOUT_ENV] ?? DEFAULT_TIMEOUT_MS);
if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 60_000) {
  throw new Error(`${TIMEOUT_ENV} must be an integer of at least 60000 milliseconds`);
}

function requiredAbsoluteDirectory(name) {
  const value = process.env[name];
  if (!value || !isAbsolute(value)) throw new Error(`${name} must be an absolute directory`);
  return value;
}

const artifactRootInput = requiredAbsoluteDirectory(ARTIFACT_ROOT_ENV);
const fixtureDirInput = requiredAbsoluteDirectory(FIXTURE_DIR_ENV);
const wwwOutInput = process.env[WWW_OUT_DIR_ENV] ?? defaultWwwOut;
if (!isAbsolute(wwwOutInput)) throw new Error(`${WWW_OUT_DIR_ENV} must be absolute when set`);
const explicitOutputDir = process.env[OUTPUT_DIR_ENV];
if (!validateOnly && !explicitOutputDir) {
  throw new Error(`${OUTPUT_DIR_ENV} must be an explicit absolute directory for a real run`);
}
const outputDirInput = explicitOutputDir ?? join(tmpdir(), "burn-image-turbo-first-dmd");
if (!isAbsolute(outputDirInput)) throw new Error(`${OUTPUT_DIR_ENV} must be absolute when set`);
const fixtureProfile = process.env[FIXTURE_PROFILE_ENV] ?? TURBO_1K_FIXTURE.profile;
const expectedFixture = {
  [TURBO_FIXTURE.profile]: TURBO_FIXTURE,
  [TURBO_1K_FIXTURE.profile]: TURBO_1K_FIXTURE,
}[fixtureProfile];
if (!expectedFixture) {
  throw new Error(
    `${FIXTURE_PROFILE_ENV} must be ${TURBO_FIXTURE.profile} or ${TURBO_1K_FIXTURE.profile}`,
  );
}

const BUNDLES = Object.freeze({
  "boogu-image-0.1-turbo": Object.freeze({
    content_digest: TURBO_ARTIFACT_CONTENT_DIGEST,
    model: TURBO_MODEL,
    model_revision: TURBO_MODEL_REVISION,
  }),
  "qwen3-vl-8b-base-boogu-image-0.1": Object.freeze({
    schema_version: 1,
    content_digest: "2f7ed91e09f208853b189ee8c3d6db74a02d2512e07f4818f6688131359d98fc",
  }),
  "flux1-vae-boogu-image-0.1": Object.freeze({
    schema_version: 1,
    content_digest: "8ff1043ac3d47e6addbb5e07f437c04585f678819ffd0e505ac46effdf1c31d6",
  }),
});

async function requireDirectory(path, name) {
  const metadata = await stat(path).catch(() => null);
  if (!metadata?.isDirectory()) throw new Error(`${name} is not a directory: ${path}`);
  return realpath(path);
}

async function identifyFile(path) {
  const metadata = await stat(path);
  if (!metadata.isFile()) throw new Error(`not a file: ${path}`);
  const digest = createHash("sha256");
  await new Promise((resolveHash, rejectHash) => {
    const input = createReadStream(path);
    input.on("data", (chunk) => digest.update(chunk));
    input.once("error", rejectHash);
    input.once("end", resolveHash);
  });
  return { bytes: metadata.size, sha256: digest.digest("hex") };
}

async function validateFixture(fixtureDir) {
  const files = {};
  for (const [name, expected] of Object.entries({
    metadata: expectedFixture.metadata,
    tensors: expectedFixture.tensors,
    output: expectedFixture.output,
  })) {
    const path = join(fixtureDir, expected.path);
    const actual = await identifyFile(path);
    if (actual.bytes !== expected.size || actual.sha256 !== expected.sha256) {
      throw new Error(
        `Turbo fixture ${name} identity ${JSON.stringify(actual)} differs from ${JSON.stringify(expected)}`,
      );
    }
    files[name] = { path, ...actual };
  }
  const metadata = JSON.parse(await readFile(files.metadata.path, "utf8"));
  for (const [field, expected] of Object.entries({
    schema_version: expectedFixture.schema_version,
    variant: expectedFixture.variant,
    model_revision: expectedFixture.model_revision,
    upstream_source_revision: expectedFixture.upstream_source_revision,
    width: expectedFixture.width,
    height: expectedFixture.height,
    seed: expectedFixture.seed,
    dtype: "bf16",
    capture_qwen: true,
    capture_blocks: true,
  })) {
    if (metadata[field] !== expected) {
      throw new Error(`Turbo fixture metadata.${field} differs from ${JSON.stringify(expected)}`);
    }
  }
  const expectedTensorCount = expectedFixture === TURBO_1K_FIXTURE ? 323 : 320;
  if (Object.keys(metadata.tensors ?? {}).length !== expectedTensorCount) {
    throw new Error(`Turbo fixture must contain exactly ${expectedTensorCount} tensors`);
  }
  return { files, metadata };
}

async function validateArtifacts(artifactRoot) {
  const manifests = {};
  for (const [bundle, expected] of Object.entries(BUNDLES)) {
    const bundleRoot = await requireDirectory(join(artifactRoot, bundle), bundle);
    const path = join(bundleRoot, "manifest.json");
    const bytes = await readFile(path);
    const manifest = JSON.parse(bytes);
    const expectedSchemaVersion = expected.schema_version ?? 2;
    if (manifest.schema_version !== expectedSchemaVersion || manifest.bundle !== bundle) {
      throw new Error(
        `${bundle} is not the exact schema-${expectedSchemaVersion} modular bundle/leaf`,
      );
    }
    if (manifest.content_digest !== expected.content_digest) {
      throw new Error(
        `${bundle} content digest ${manifest.content_digest} differs from ${expected.content_digest}`,
      );
    }
    for (const field of ["model", "model_revision"]) {
      if (expected[field] && manifest[field] !== expected[field]) {
        throw new Error(`${bundle}.${field} differs from ${expected[field]}`);
      }
    }
    manifests[bundle] = {
      root: bundleRoot,
      path,
      bytes: bytes.length,
      sha256: createHash("sha256").update(bytes).digest("hex"),
      content_digest: manifest.content_digest,
    };
  }
  return manifests;
}

async function validatePackage(wwwOut) {
  const javascriptPath = join(wwwOut, "bevy_burn_image.js");
  const wasmPath = join(wwwOut, "bevy_burn_image_bg.wasm");
  const [javascript, webassembly] = await Promise.all([
    identifyFile(javascriptPath),
    identifyFile(wasmPath),
  ]);
  const javascriptSource = await readFile(javascriptPath, "utf8");
  if (!javascriptSource.includes("export function start_boogu_web()")) {
    throw new Error("browser package omits start_boogu_web export");
  }
  return {
    javascript: { path: javascriptPath, ...javascript },
    webassembly: { path: wasmPath, ...webassembly },
  };
}

async function validateSources() {
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
  const failures = validateTurboFirstDmdSourceContract(inputs);
  if (failures.length > 0) throw new Error(`source contract failed:\n${failures.join("\n")}`);
  return Object.fromEntries(
    await Promise.all(
      [
        join(repoRoot, "crates/bevy_image/src/lib.rs"),
        join(repoRoot, "crates/bevy_image/src/browser_boogu.rs"),
        join(repoRoot, "crates/bevy_image/src/browser_turbo_first_dmd_fixture.rs"),
        harnessPath,
        contractPath,
        htmlPath,
      ].map(async (path) => [relative(repoRoot, path), await identifyFile(path)]),
    ),
  );
}

function inside(root, path) {
  const suffix = relative(root, path);
  return suffix === "" || (!suffix.startsWith(`..${sep}`) && suffix !== ".." && !isAbsolute(suffix));
}

function mime(path) {
  return {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".wasm": "application/wasm",
    ".json": "application/json",
    ".bpk": "application/octet-stream",
    ".safetensors": "application/octet-stream",
    ".png": "image/png",
  }[extname(path)] ?? "application/octet-stream";
}

function cors() {
  return {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Range, Content-Type",
    "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
    "Access-Control-Expose-Headers": "Accept-Ranges, Content-Length, Content-Range",
    "Cross-Origin-Resource-Policy": "cross-origin",
  };
}

function parseRange(value, size) {
  if (!value) return null;
  const match = /^bytes=([0-9]+)-([0-9]+)$/.exec(value);
  if (!match) throw new Error("malformed Range header");
  const start = Number(match[1]);
  const end = Number(match[2]);
  if (![start, end].every(Number.isSafeInteger) || start < 0 || end < start || end >= size) {
    throw new Error("unsatisfiable Range header");
  }
  return { start, end, length: end - start + 1 };
}

async function startServer({ artifactRoot, fixtureDir, wwwOut }) {
  const html = await readFile(htmlPath);
  const requests = [];
  const recordRequest = (event) => {
    requests.push({ at_ms: Date.now(), ...event });
    if (requests.length > MAX_SERVER_REQUEST_EVENTS) requests.shift();
  };
  const sockets = new Set();
  const server = createServer(async (request, response) => {
    try {
      if (request.method === "OPTIONS") {
        response.writeHead(204, cors());
        response.end();
        return;
      }
      if (!["GET", "HEAD"].includes(request.method)) {
        response.writeHead(405, { ...cors(), Allow: "GET, HEAD, OPTIONS" });
        response.end();
        return;
      }
      const url = new URL(request.url, "http://127.0.0.1");
      let path;
      if (url.pathname === "/probe" || url.pathname === "/") {
        recordRequest({
          method: request.method,
          path: url.pathname,
          range: request.headers.range ?? null,
          status: 200,
          response_bytes: html.length,
        });
        response.writeHead(200, {
          ...cors(),
          "Content-Type": "text/html; charset=utf-8",
          "Content-Length": html.length,
        });
        if (request.method === "HEAD") response.end();
        else response.end(html);
        return;
      }
      if (url.pathname.startsWith("/app/out/")) {
        const name = url.pathname.slice("/app/out/".length);
        if (!new Set(["bevy_burn_image.js", "bevy_burn_image_bg.wasm"]).has(name)) {
          response.writeHead(404, cors());
          response.end();
          return;
        }
        path = resolve(wwwOut, name);
        if (!inside(wwwOut, path)) throw new Error("package path escaped its root");
      } else if (url.pathname.startsWith("/fixture/")) {
        const name = url.pathname.slice("/fixture/".length);
        if (!new Set(["metadata.json", "tensors.safetensors", "output.png"]).has(name)) {
          response.writeHead(404, cors());
          response.end();
          return;
        }
        path = resolve(fixtureDir, name);
        if (!inside(fixtureDir, path)) throw new Error("fixture path escaped its root");
      } else if (url.pathname.startsWith("/model/")) {
        const suffix = url.pathname.slice("/model/".length);
        const slash = suffix.indexOf("/");
        const bundle = slash < 0 ? suffix : suffix.slice(0, slash);
        const object = slash < 0 ? "" : suffix.slice(slash + 1);
        if (!Object.hasOwn(BUNDLES, bundle) || !object) {
          response.writeHead(404, cors());
          response.end();
          return;
        }
        const bundleRoot = resolve(artifactRoot, bundle);
        path = resolve(bundleRoot, object);
        if (!inside(bundleRoot, path)) throw new Error("model path escaped its bundle root");
      } else {
        response.writeHead(404, cors());
        response.end();
        return;
      }

      const metadata = await stat(path).catch(() => null);
      if (!metadata?.isFile()) {
        response.writeHead(404, cors());
        response.end();
        return;
      }
      let range;
      try {
        range = parseRange(request.headers.range, metadata.size);
      } catch {
        response.writeHead(416, {
          ...cors(),
          "Accept-Ranges": "bytes",
          "Content-Range": `bytes */${metadata.size}`,
        });
        response.end();
        return;
      }
      const start = range?.start ?? 0;
      const end = range?.end ?? metadata.size - 1;
      const length = range?.length ?? metadata.size;
      recordRequest({
        method: request.method,
        path: url.pathname,
        range: request.headers.range ?? null,
        status: range ? 206 : 200,
        response_bytes: length,
      });
      response.writeHead(range ? 206 : 200, {
        ...cors(),
        "Accept-Ranges": "bytes",
        "Content-Type": mime(path),
        "Content-Length": length,
        ...(range ? { "Content-Range": `bytes ${start}-${end}/${metadata.size}` } : {}),
      });
      if (request.method === "HEAD") response.end();
      else createReadStream(path, { start, end }).pipe(response);
    } catch (error) {
      if (!response.headersSent) response.writeHead(500, cors());
      response.end(String(error));
    }
  });
  server.on("connection", (socket) => {
    sockets.add(socket);
    socket.once("close", () => sockets.delete(socket));
  });
  await new Promise((resolveListen, rejectListen) => {
    server.once("error", rejectListen);
    server.listen(0, "127.0.0.1", resolveListen);
  });
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("HTTP server has no TCP address");
  const close = async () => {
    const activeConnectionsBefore = sockets.size;
    let closeError = null;
    let closeSettled = false;
    const closePromise = new Promise((resolveClose) => {
      server.close((error) => {
        closeError = error ?? null;
        closeSettled = true;
        resolveClose();
      });
      server.closeIdleConnections?.();
    });
    const graceful = await Promise.race([
      closePromise.then(() => true),
      new Promise((resolveTimeout) =>
        setTimeout(() => resolveTimeout(false), SERVER_CLOSE_TIMEOUT_MS),
      ),
    ]);
    let forced = false;
    if (!graceful) {
      forced = true;
      server.closeAllConnections?.();
      for (const socket of sockets) socket.destroy();
      await Promise.race([
        closePromise,
        new Promise((resolveTimeout) => setTimeout(resolveTimeout, SERVER_CLOSE_TIMEOUT_MS)),
      ]);
    }
    if (!closeSettled) {
      throw new Error(
        `HTTP server did not close within ${SERVER_CLOSE_TIMEOUT_MS * 2} milliseconds`,
      );
    }
    if (closeError) throw closeError;
    return {
      graceful,
      forced,
      timeout_ms: SERVER_CLOSE_TIMEOUT_MS,
      active_connections_before: activeConnectionsBefore,
      active_connections_after: sockets.size,
    };
  };
  return {
    origin: `http://127.0.0.1:${address.port}`,
    requests: () => requests.slice(),
    close,
  };
}

async function validateTransport(origin) {
  const range = await fetch(`${origin}/fixture/tensors.safetensors`, {
    headers: { Range: "bytes=0-7" },
  });
  if (
    range.status !== 206 ||
    range.headers.get("content-range") !== `bytes 0-7/${expectedFixture.tensors.size}` ||
    Number(range.headers.get("content-length")) !== 8 ||
    (await range.arrayBuffer()).byteLength !== 8
  ) {
    throw new Error("fixture server failed its exact 8-byte Range self-test");
  }
  for (const bundle of Object.keys(BUNDLES)) {
    const manifest = await fetch(`${origin}/model/${bundle}/manifest.json`);
    if (!manifest.ok || !(await manifest.text()).includes(BUNDLES[bundle].content_digest)) {
      throw new Error(`model server failed the ${bundle} manifest self-test`);
    }
  }
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
      requested_bytes: TURBO_FIRST_DMD_CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES,
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
    availableBytes < BigInt(TURBO_FIRST_DMD_CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES)
  ) {
    measurement.quota_aware_allocation_probe.skipped_reason =
      "statfs-available-below-minimum";
  } else {
    measurement.quota_aware_allocation_probe = await quotaAwareAllocationProbe(
      path,
      TURBO_FIRST_DMD_CHROME_SHARED_MEMORY_MIN_HEADROOM_BYTES,
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
      ...selectTurboFirstDmdChromeSharedMemoryPolicy({
        platform: process.platform,
        devShm: null,
        tempDirectory: null,
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
    ...selectTurboFirstDmdChromeSharedMemoryPolicy({
      platform: process.platform,
      devShm,
      tempDirectory,
      tempPath,
    }),
    capacities: {
      dev_shm: devShm,
      temp_directory: tempDirectory,
    },
  };
}

async function inspectChromeProfileStorage(path) {
  const measurement = await measureChromeSharedMemoryPath(path);
  return {
    schema_version: 1,
    purpose: "isolated-model-browser-profile-and-cache-storage",
    path,
    measurement_method:
      "bigint-statfs-global-capacity-plus-bounded-real-write-fsync-probe-for-effective-user-quota",
    measurement,
    ...inspectTurboFirstDmdStorageAdmission(measurement),
  };
}

function appendBounded(current, chunk) {
  const value = current + chunk;
  return value.length <= MAX_CHROME_LOG_BYTES
    ? value
    : value.slice(value.length - MAX_CHROME_LOG_BYTES);
}

async function findChrome() {
  const explicit = process.env[CHROME_ENV];
  if (explicit) {
    if (!isAbsolute(explicit) || !existsSync(explicit)) {
      throw new Error(`${CHROME_ENV} must identify an absolute Chrome executable`);
    }
    return explicit;
  }
  for (const candidate of [
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
  ]) {
    if (existsSync(candidate)) return candidate;
  }
  throw new Error(`Chrome not found; set ${CHROME_ENV}`);
}

async function launchChrome(profile, sharedMemoryPolicy) {
  const executable = await findChrome();
  const arguments_ = [
    "--headless=new",
    "--no-sandbox",
  ];
  if (sharedMemoryPolicy.disable_dev_shm_usage) {
    arguments_.push("--disable-dev-shm-usage");
  }
  arguments_.push(
    "--enable-gpu",
    "--enable-unsafe-webgpu",
    "--enable-features=Vulkan,VulkanFromANGLE,WebGPUDeveloperFeatures",
    "--use-angle=vulkan",
    "--disable-vulkan-surface",
    "--ignore-gpu-blocklist",
    "--remote-debugging-port=0",
    `--user-data-dir=${profile}`,
    "about:blank",
  );
  const child = spawn(executable, arguments_, {
    detached: true,
    stdio: ["ignore", "pipe", "pipe"],
  });
  let stdout = "";
  let stderr = "";
  let devtoolsUrl;
  const endpoint = new Promise((resolveEndpoint, rejectEndpoint) => {
    const timeout = setTimeout(
      () => rejectEndpoint(new Error("Chrome DevTools endpoint did not start")),
      DEVTOOLS_TIMEOUT_MS,
    );
    const inspect = (chunk) => {
      const match = /DevTools listening on (ws:\/\/[^\s]+)/.exec(String(chunk));
      if (match && !devtoolsUrl) {
        devtoolsUrl = match[1];
        clearTimeout(timeout);
        resolveEndpoint(devtoolsUrl);
      }
    };
    child.stderr.on("data", inspect);
    child.once("exit", (code, signal) => {
      if (!devtoolsUrl) {
        clearTimeout(timeout);
        rejectEndpoint(new Error(`Chrome exited before DevTools: code=${code} signal=${signal}`));
      }
    });
  });
  child.stdout.on("data", (chunk) => {
    stdout = appendBounded(stdout, String(chunk));
  });
  child.stderr.on("data", (chunk) => {
    stderr = appendBounded(stderr, String(chunk));
  });
  return {
    executable,
    arguments_,
    child,
    processGroupId: child.pid,
    endpoint,
    devtoolsUrl: () => devtoolsUrl,
    logs: () => ({ stdout, stderr }),
  };
}

class Cdp {
  constructor(webSocketUrl) {
    if (typeof globalThis.WebSocket !== "function") {
      throw new Error("the real diagnostic requires Node with global WebSocket support");
    }
    this.socket = new WebSocket(webSocketUrl);
    this.nextId = 1;
    this.pending = new Map();
    this.events = [];
    this.closedError = null;
    this.socket.addEventListener("message", (event) => {
      let message;
      try {
        message = JSON.parse(String(event.data));
      } catch (error) {
        this.failPending(
          new Error(`Chrome CDP WebSocket returned invalid JSON: ${String(error)}`),
        );
        return;
      }
      if (message.id) {
        const pending = this.pending.get(message.id);
        if (!pending) return;
        this.pending.delete(message.id);
        clearTimeout(pending.timeout);
        if (message.error) pending.reject(new Error(JSON.stringify(message.error)));
        else pending.resolve(message.result);
      } else {
        this.events.push(message);
      }
    });
    this.socket.addEventListener("error", () => {
      this.failPending(new Error("Chrome CDP WebSocket error"));
    });
    this.socket.addEventListener("close", (event) => {
      this.failPending(
        new Error(
          `Chrome CDP WebSocket closed: code=${event.code} reason=${JSON.stringify(event.reason)}`,
        ),
      );
    });
  }

  failPending(error) {
    const detail = error instanceof Error ? error : new Error(String(error));
    this.closedError ??= detail;
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timeout);
      pending.reject(this.closedError);
    }
    this.pending.clear();
  }

  async open() {
    if (this.closedError) throw this.closedError;
    if (this.socket.readyState === WebSocket.OPEN) return;
    if (this.socket.readyState !== WebSocket.CONNECTING) {
      throw new Error(`Chrome CDP WebSocket cannot open from state ${this.socket.readyState}`);
    }
    await new Promise((resolveOpen, rejectOpen) => {
      let settled = false;
      const settle = (error) => {
        if (settled) return;
        settled = true;
        clearTimeout(timeout);
        this.socket.removeEventListener("open", onOpen);
        this.socket.removeEventListener("error", onError);
        this.socket.removeEventListener("close", onClose);
        if (error) {
          this.failPending(error);
          rejectOpen(this.closedError);
        } else {
          resolveOpen();
        }
      };
      const onOpen = () => settle(null);
      const onError = () =>
        settle(this.closedError ?? new Error("Chrome CDP WebSocket error"));
      const onClose = () =>
        settle(this.closedError ?? new Error("Chrome CDP WebSocket closed before opening"));
      const timeout = setTimeout(
        () => settle(new Error(`Chrome CDP WebSocket open exceeded ${CDP_CALL_TIMEOUT_MS} ms`)),
        CDP_CALL_TIMEOUT_MS,
      );
      this.socket.addEventListener("open", onOpen, { once: true });
      this.socket.addEventListener("error", onError, { once: true });
      this.socket.addEventListener("close", onClose, { once: true });
    });
  }

  call(method, params = {}) {
    if (this.closedError) return Promise.reject(this.closedError);
    if (this.socket.readyState !== WebSocket.OPEN) {
      return Promise.reject(
        new Error(`Chrome CDP WebSocket is not open for ${method}: ${this.socket.readyState}`),
      );
    }
    const id = this.nextId++;
    return new Promise((resolveCall, rejectCall) => {
      const timeout = setTimeout(() => {
        if (!this.pending.has(id)) return;
        this.failPending(
          new Error(`Chrome CDP call ${method} exceeded ${CDP_CALL_TIMEOUT_MS} ms`),
        );
      }, CDP_CALL_TIMEOUT_MS);
      this.pending.set(id, { resolve: resolveCall, reject: rejectCall, timeout, method });
      try {
        this.socket.send(JSON.stringify({ id, method, params }));
      } catch (error) {
        clearTimeout(timeout);
        this.pending.delete(id);
        rejectCall(
          new Error(`Chrome CDP call ${method} could not be sent: ${String(error)}`),
        );
      }
    });
  }

  close() {
    this.failPending(new Error("Chrome CDP WebSocket closed by harness"));
    if ([WebSocket.CONNECTING, WebSocket.OPEN].includes(this.socket.readyState)) {
      this.socket.close();
    }
  }
}

async function openPage(browser, url) {
  const devtools = new URL(browser.devtoolsUrl());
  const createUrl = `http://${devtools.host}/json/new?${encodeURIComponent(url)}`;
  const response = await fetch(createUrl, { method: "PUT" });
  if (!response.ok) throw new Error(`Chrome page creation failed: HTTP ${response.status}`);
  const target = await response.json();
  const cdp = new Cdp(target.webSocketDebuggerUrl);
  await cdp.open();
  await Promise.all([cdp.call("Runtime.enable"), cdp.call("Page.enable"), cdp.call("Log.enable")]);
  return cdp;
}

async function readProbeState(cdp) {
  const evaluation = await cdp.call("Runtime.evaluate", {
    expression: "globalThis.__burnImageTurboFirstDmdProbe ?? null",
    returnByValue: true,
  });
  return evaluation.result?.value ?? null;
}

function fatalCdpDiagnostic(events) {
  for (const event of events) {
    if (event.method === "Runtime.exceptionThrown") return event;
    if (event.method === "Runtime.consoleAPICalled" && event.params?.type === "error") return event;
    if (event.method === "Log.entryAdded" && event.params?.entry?.level === "error") return event;
  }
  return null;
}

async function waitForTerminal(cdp, server, browserRequestStart) {
  const deadline = Date.now() + timeoutMs;
  const firstModelRequestDeadline = Date.now() + FIRST_MODEL_REQUEST_TIMEOUT_MS;
  while (Date.now() < deadline) {
    const state = await readProbeState(cdp);
    if (state?.stage === "complete") return state;
    if (state?.stage === "failed") {
      throw new Error(state.error ?? "browser first-DMD diagnostic failed");
    }
    const fatal = fatalCdpDiagnostic(cdp.events);
    if (fatal) {
      throw new Error(`browser first-DMD diagnostic emitted a fatal CDP event: ${JSON.stringify(fatal)}`);
    }
    if (
      Date.now() >= firstModelRequestDeadline &&
      !server
        .requests()
        .slice(browserRequestStart)
        .some((event) => event.path.startsWith("/model/"))
    ) {
      throw new Error(
        `browser first-DMD diagnostic made no model request within ${FIRST_MODEL_REQUEST_TIMEOUT_MS} ms; state=${JSON.stringify(state)}`,
      );
    }
    await new Promise((resolveWait) => setTimeout(resolveWait, 1_000));
  }
  throw new Error(`browser first-DMD diagnostic exceeded ${timeoutMs} ms`);
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
    await new Promise((resolveWait) => setTimeout(resolveWait, 100));
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
  const processGroupId = browser.processGroupId;
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

let server;
let browser;
let cdp;
let profile;
let browserRequestStart = 0;
let hardwareAdapterAttestation;
let chromeSharedMemory;
let chromeProfileStorage;
let inputEvidence;
let terminalOutput;
let terminalReport;
let primaryFailureDetail;
const startedAt = new Date().toISOString();
const unresolvedOutputDir = resolve(outputDirInput);
await mkdir(unresolvedOutputDir, { recursive: true });
const outputDir = await realpath(unresolvedOutputDir);
try {
  const [artifactRoot, fixtureDir, wwwOut] = await Promise.all([
    requireDirectory(artifactRootInput, ARTIFACT_ROOT_ENV),
    requireDirectory(fixtureDirInput, FIXTURE_DIR_ENV),
    requireDirectory(wwwOutInput, WWW_OUT_DIR_ENV),
  ]);
  const [manifests, fixture, browserPackage, sources] = await Promise.all([
    validateArtifacts(artifactRoot),
    validateFixture(fixtureDir),
    validatePackage(wwwOut),
    validateSources(),
  ]);
  inputEvidence = {
    schema_version: 1,
    started_at: startedAt,
    validate_only: validateOnly,
    probe_url: null,
    manifests,
    fixture_files: fixture.files,
    browser_package: browserPackage,
    sources,
    output_dir: outputDir,
    transport_validated: false,
  };
  if (!validateOnly) {
    chromeSharedMemory = await inspectChromeSharedMemory();
    if (chromeSharedMemory.launch_admitted !== true) {
      throw new Error(
        `no quota-and-capacity-admitted Chrome shared-memory backing: ${JSON.stringify(chromeSharedMemory)}`,
      );
    }
    chromeProfileStorage = await inspectChromeProfileStorage(outputDir);
    if (chromeProfileStorage.admitted !== true) {
      throw new Error(
        `Chrome profile filesystem lacks admitted capacity/quota: ${JSON.stringify(chromeProfileStorage)}`,
      );
    }
  }
  server = await startServer({ artifactRoot, fixtureDir, wwwOut });
  await validateTransport(server.origin);
  const modelBase = `${server.origin}/model/boogu-image-0.1-turbo`;
  const fixtureBase = `${server.origin}/fixture`;
  const query = new URLSearchParams({
    headless: "turbo-first-dmd",
    variant: "turbo",
    profile: "production",
    residency: "low-vram",
    artifacts: modelBase,
    fixture: fixtureBase,
    "fixture-profile": fixtureProfile,
  });
  const probeUrl = `${server.origin}/probe?${query}`;
  inputEvidence.probe_url = probeUrl;
  inputEvidence.transport_validated = true;
  if (validateOnly) {
    terminalOutput = { ...inputEvidence, outcome: "validated-without-gpu-launch" };
  } else {
    profile = await mkdtemp(join(outputDir, "burn-image-turbo-first-dmd-chrome-"));
    browser = await launchChrome(profile, chromeSharedMemory);
    await browser.endpoint;
    browserRequestStart = server.requests().length;
    cdp = await openPage(browser, probeUrl);
    const state = await waitForTerminal(cdp, server, browserRequestStart);
    const report = state.report;
    terminalReport = report;
    hardwareAdapterAttestation = summarizeTurboFirstDmdWebGpuCalls(state.webgpu_calls);
    const failures = validateTurboFirstDmdReport(report, hardwareAdapterAttestation);
    if (failures.length > 0) {
      throw new Error(`Turbo first-DMD report contract failed:\n${failures.join("\n")}`);
    }
    const gpuDiagnostics = cdp.events
      .filter((event) =>
        ["Runtime.exceptionThrown", "Log.entryAdded"].includes(event.method),
      )
      .map((event) => event.params);
    terminalOutput = {
      ...inputEvidence,
      outcome: "diagnostic-passed-no-full-parity-claim",
      completed_at: new Date().toISOString(),
      browser: {
        executable: browser.executable,
        arguments: browser.arguments_,
        logs: browser.logs(),
        gpu_diagnostics: gpuDiagnostics,
        webgpu_calls: state.webgpu_calls,
      },
      hardware_adapter_attestation: hardwareAdapterAttestation,
      server_requests: server.requests().slice(browserRequestStart),
      report,
    };
  }
} catch (error) {
  const detail = error instanceof Error ? error.stack ?? error.message : String(error);
  primaryFailureDetail = detail;
  const probeState = cdp ? await readProbeState(cdp).catch(() => null) : null;
  hardwareAdapterAttestation ??= summarizeTurboFirstDmdWebGpuCalls(probeState?.webgpu_calls);
  terminalOutput = {
    ...(inputEvidence ?? {}),
    schema_version: 1,
    started_at: startedAt,
    completed_at: new Date().toISOString(),
    outcome: "failed",
    error: detail,
    probe_state: probeState,
    server_requests: server?.requests?.().slice(browserRequestStart) ?? [],
    cdp_events: cdp?.events ?? [],
    chrome_logs: browser?.logs() ?? null,
    hardware_adapter_attestation: hardwareAdapterAttestation,
  };
  process.exitCode = 1;
} finally {
  const cleanupErrors = [];
  try {
    cdp?.close();
  } catch (error) {
    cleanupErrors.push(`CDP close failed: ${error}`);
  }
  let chromeCleanup = null;
  try {
    chromeCleanup = await stopChrome(browser);
    cleanupErrors.push(...(chromeCleanup?.errors ?? []));
  } catch (error) {
    cleanupErrors.push(`Chrome process-group cleanup failed: ${error}`);
  }
  let serverCleanup = null;
  try {
    serverCleanup = await server?.close();
  } catch (error) {
    cleanupErrors.push(`HTTP server cleanup failed: ${error}`);
  }
  let profileRemoved = false;
  if (profile) {
    if (!browser || chromeCleanup?.process_group_exited === true) {
      try {
        await rm(profile, {
          recursive: true,
          force: true,
          maxRetries: 8,
          retryDelay: 100,
        });
        profileRemoved = true;
      } catch (error) {
        cleanupErrors.push(`Chrome profile cleanup failed: ${error}`);
      }
    } else {
      cleanupErrors.push("Chrome profile retained because the process group did not exit");
    }
  }

  terminalOutput ??= {
    ...(inputEvidence ?? {}),
    schema_version: 1,
    started_at: startedAt,
    outcome: "failed",
    error: "diagnostic ended without terminal output",
  };
  Object.assign(
    terminalOutput,
    turboFirstDmdChromeLaunchEvidence({
      executable: browser?.executable,
      arguments: browser?.arguments_,
      profile,
      sharedMemory: chromeSharedMemory,
      profileStorage: chromeProfileStorage,
      cleanup: chromeCleanup,
      profileRemoved,
    }),
  );
  terminalOutput.server_cleanup = serverCleanup;
  if (terminalOutput.browser) terminalOutput.browser.logs = browser?.logs() ?? null;
  else terminalOutput.chrome_logs = browser?.logs() ?? terminalOutput.chrome_logs ?? null;
  terminalOutput.cleanup_errors = cleanupErrors;
  terminalOutput.completed_at = new Date().toISOString();

  if (cleanupErrors.length > 0 && terminalOutput.outcome !== "failed") {
    terminalOutput.pre_cleanup_outcome = terminalOutput.outcome;
    terminalOutput.outcome = "failed";
    terminalOutput.error = `diagnostic cleanup failed: ${cleanupErrors.join("; ")}`;
    primaryFailureDetail = terminalOutput.error;
    process.exitCode = 1;
  }

  const outputName =
    terminalOutput.outcome === "diagnostic-passed-no-full-parity-claim"
      ? "burn-image-turbo-first-dmd-report.json"
      : terminalOutput.outcome === "validated-without-gpu-launch"
        ? "burn-image-turbo-first-dmd-validate-only.json"
        : "burn-image-turbo-first-dmd-failure.json";
  await writeFile(join(outputDir, outputName), `${JSON.stringify(terminalOutput, null, 2)}\n`);
  if (terminalOutput.outcome === "diagnostic-passed-no-full-parity-claim") {
    process.exitCode = 0;
    console.log(`${TERMINAL_OK}${JSON.stringify(terminalReport)}`);
  } else if (terminalOutput.outcome === "validated-without-gpu-launch") {
    process.exitCode = 0;
    console.log(JSON.stringify(terminalOutput, null, 2));
  } else {
    process.exitCode = 1;
    console.error(`${TERMINAL_FAILED} ${primaryFailureDetail ?? terminalOutput.error}`);
  }
}
