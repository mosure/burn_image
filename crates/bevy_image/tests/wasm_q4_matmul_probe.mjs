#!/usr/bin/env node

import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { mkdtemp, mkdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import { extname, join, normalize, resolve, sep } from "node:path";

const TERMINAL_OK = "BURN_IMAGE_HEADLESS_Q4_MATMUL_OK ";
const TERMINAL_FAILED = "BURN_IMAGE_HEADLESS_Q4_MATMUL_FAILED ";
const packageRoot = resolve(
  process.env.BURN_IMAGE_Q4_MATMUL_WWW_OUT_DIR ?? "crates/bevy_image/www",
);
const outputRoot = resolve(
  process.env.BURN_IMAGE_Q4_MATMUL_OUTPUT_DIR ??
    (await mkdtemp("/tmp/burn-image-q4-matmul-output-")),
);
const timeoutMs = Number(process.env.BURN_IMAGE_Q4_MATMUL_TIMEOUT_MS ?? 120_000);
const chromeBinary = process.env.CHROME_BIN ?? "/usr/bin/google-chrome";
await mkdir(outputRoot, { recursive: true });

const mime = new Map([
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".wasm", "application/wasm"],
  [".png", "image/png"],
  [".json", "application/json"],
]);

function safePath(url) {
  const pathname = decodeURIComponent(new URL(url, "http://localhost").pathname);
  const candidate = resolve(packageRoot, `.${normalize(pathname === "/" ? "/index.html" : pathname)}`);
  if (candidate !== packageRoot && !candidate.startsWith(`${packageRoot}${sep}`)) {
    throw new Error(`request escaped package root: ${pathname}`);
  }
  return candidate;
}

const server = createServer(async (request, response) => {
  try {
    const path = safePath(request.url ?? "/");
    const info = await stat(path);
    if (!info.isFile()) throw new Error("not a file");
    const body = await readFile(path);
    response.writeHead(200, {
      "Content-Type": mime.get(extname(path)) ?? "application/octet-stream",
      "Content-Length": body.length,
      "Cache-Control": "no-store",
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    });
    response.end(body);
  } catch {
    response.writeHead(404, { "Content-Type": "text/plain" });
    response.end("not found");
  }
});
await new Promise((resolvePromise, reject) => {
  server.once("error", reject);
  server.listen(0, "127.0.0.1", resolvePromise);
});
const address = server.address();
if (!address || typeof address === "string") throw new Error("test server has no TCP address");

const profile = await mkdtemp(join(outputRoot, "chrome-profile-"));
const chrome = spawn(
  chromeBinary,
  [
    "--headless=new",
    "--no-sandbox",
    "--disable-gpu-sandbox",
    "--enable-unsafe-webgpu",
    "--use-angle=vulkan",
    "--enable-features=Vulkan,UseSkiaRenderer,WebGPUService",
    "--disable-features=UseChromeOSDirectVideoDecoder",
    "--remote-debugging-port=0",
    `--user-data-dir=${profile}`,
    "about:blank",
  ],
  { detached: true, stdio: ["ignore", "ignore", "pipe"] },
);
let stderr = "";
chrome.stderr.setEncoding("utf8");
chrome.stderr.on("data", (chunk) => {
  stderr += chunk;
  if (stderr.length > 64_000) stderr = stderr.slice(-64_000);
});

async function waitForDevtools() {
  const activePort = join(profile, "DevToolsActivePort");
  const deadline = Date.now() + 15_000;
  while (Date.now() < deadline) {
    try {
      const [port] = (await readFile(activePort, "utf8")).trim().split(/\s+/);
      const response = await fetch(`http://127.0.0.1:${port}/json/new?http://127.0.0.1:${address.port}/`, {
        method: "PUT",
      });
      if (response.ok) return response.json();
    } catch {}
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 100));
  }
  throw new Error(`Chrome DevTools did not become ready; stderr=${stderr}`);
}

class Cdp {
  constructor(url) {
    this.next = 1;
    this.pending = new Map();
    this.socket = new WebSocket(url);
  }
  async open() {
    await new Promise((resolvePromise, reject) => {
      this.socket.addEventListener("open", resolvePromise, { once: true });
      this.socket.addEventListener("error", reject, { once: true });
    });
    this.socket.addEventListener("message", ({ data }) => {
      const message = JSON.parse(data);
      if (!message.id) return;
      const pending = this.pending.get(message.id);
      if (!pending) return;
      this.pending.delete(message.id);
      if (message.error) pending.reject(new Error(JSON.stringify(message.error)));
      else pending.resolve(message.result);
    });
    this.socket.addEventListener("close", () => {
      for (const { reject } of this.pending.values()) reject(new Error("CDP socket closed"));
      this.pending.clear();
    });
  }
  call(method, params = {}) {
    const id = this.next++;
    return new Promise((resolvePromise, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`CDP ${method} timed out`));
      }, 30_000);
      this.pending.set(id, {
        resolve: (value) => {
          clearTimeout(timer);
          resolvePromise(value);
        },
        reject: (error) => {
          clearTimeout(timer);
          reject(error);
        },
      });
      this.socket.send(JSON.stringify({ id, method, params }));
    });
  }
  close() {
    this.socket.close();
  }
}

let cdp;
let report;
try {
  const target = await waitForDevtools();
  cdp = new Cdp(target.webSocketDebuggerUrl);
  await cdp.open();
  await cdp.call("Runtime.enable");
  await cdp.call("Page.enable");
  await cdp.call("Page.navigate", {
    url: `http://127.0.0.1:${address.port}/?headless=q4-matmul-probe&variant=turbo`,
  });
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const result = await cdp.call("Runtime.evaluate", {
      expression: "document.getElementById('status')?.textContent ?? ''",
      returnByValue: true,
    });
    const status = result.result?.value ?? "";
    if (status.startsWith(TERMINAL_OK)) {
      report = JSON.parse(status.slice(TERMINAL_OK.length));
      break;
    }
    if (status.startsWith(TERMINAL_FAILED)) throw new Error(status);
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 250));
  }
  if (!report) throw new Error(`packed-Q4 probe timed out after ${timeoutMs} ms`);
  await writeFile(join(outputRoot, "burn-image-q4-matmul-report.json"), `${JSON.stringify(report, null, 2)}\n`);
  process.stdout.write(`${TERMINAL_OK}${JSON.stringify(report)}\nOUTPUT_DIR=${outputRoot}\n`);
} finally {
  try {
    await cdp?.call("Browser.close");
  } catch {}
  cdp?.close();
  if (chrome.exitCode === null) {
    try { process.kill(-chrome.pid, "SIGTERM"); } catch {}
  }
  await Promise.race([
    new Promise((resolvePromise) => chrome.once("exit", resolvePromise)),
    new Promise((resolvePromise) => setTimeout(resolvePromise, 5_000)),
  ]);
  if (chrome.exitCode === null) {
    try { process.kill(-chrome.pid, "SIGKILL"); } catch {}
  }
  await new Promise((resolvePromise) => server.close(resolvePromise));
  await rm(profile, { recursive: true, force: true, maxRetries: 8, retryDelay: 100 });
}
