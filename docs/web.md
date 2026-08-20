# Browser and GitHub Pages

The browser build runs the same Bevy application and Boogu runtime as native, adapted to WebGPU,
HTTP Range transport, Cache Storage, and Wasm memory limits.

## Build

```sh
cargo build -p bevy_burn_image \
  --target wasm32-unknown-unknown \
  --profile wasm-release \
  --no-default-features \
  --features boogu-web \
  --lib
```

Run `wasm-bindgen` with `--target web`, then copy the web shell and
`crates/bevy_image/www/burn-image-icon.png` into the generated package. The deployment workflow
contains the complete packaging recipe.

## Startup

1. Bevy creates the browser WGPU adapter, device, and queue.
2. Burn attaches to those exact handles.
3. The selected model manifest and dependency manifests are authenticated.
4. Transport layouts are authenticated and the executable model closure is planned.
5. The runtime checks adapter limits and its conservative VRAM plan.
6. Required parts are read from Cache Storage or fetched from the CDN.
7. Logical objects are reconstructed, verified, and uploaded to the GPU.
8. The Bevy UI reports the runtime ready state.

For Q4 models, all three public parents resolve the same Q4 Qwen and F16 VAE manifests. Switching
between Generate, Edit, and Edit 1.5K fetches only the selected denoiser payload that is not already
shared by the active dependency cache.

There is no separate HTML model-loading modal. Bootstrap failures and progress are surfaced through
the Bevy UI so native and browser state transitions remain aligned.

## Progress

Progress is aggregate and monotonic for the active model closure. The UI reports:

- verified bytes and total bytes;
- logical objects completed and total;
- unique physical parts completed and total;
- bounded reads completed and total;
- smoothed throughput and ETA after enough samples exist;
- the current semantic phase.

Stage-local counts are not presented as total model progress. When all bytes are cached but stages
are still being applied, the bar becomes indeterminate and names that work explicitly.

## Persistent cache

The browser stores one authenticated physical transport part per Cache Storage entry. On a cache
hit it still checks the expected size and SHA-256 digest. A missing or corrupt entry is evicted and
replaced from the CDN; a completed active-session cache must not silently fall back to repeated
network transfer.

Only one reconstructed logical object needs to be live in Wasm memory at a time. GPU weights may
remain resident after upload, so this bound does not force per-request model unloading.

Persistent-storage quota values returned by browser APIs are converted without narrowing to 32-bit
integers. Before failing a selected-model quota preflight, the runtime removes only entries that
are not part of the selected closure from burn_image's dedicated cache. Shared Qwen/VAE entries are
retained. Quota failure is then reported with actionable context; ample host disk space alone does
not guarantee that a browser profile grants enough origin quota.

## CDN requirements

Every manifest, compact payload, layout, and physical part must support exact complete-object and
Range validation. For Range requests the CDN must return:

- HTTP `206`;
- exact server-side `Content-Range` including the full object length;
- exact `Content-Length`;
- absent or `identity` `Content-Encoding`;
- CORS permission for the requesting Pages origin.

The canonical loader fetches each bounded physical part as a complete immutable object and verifies
its exact size and SHA-256 digest, so it does not depend on JavaScript-visible `Content-Range`.
Exposing `Content-Range` remains required for consumers of the lower-level browser Range API and is
therefore reported as a readiness warning when absent.

Physical parts target 20,971,520 bytes and must not exceed 25,000,000 bytes. Payloads should use an
immutable cache policy. Manifest cache policy is advisory because manifests are sealed and refer to
immutable payload identities; readiness still authenticates every fetched manifest.

## UI and model selection

The model selector determines task capability:

- Turbo shows prompt-only generation controls.
- Edit Turbo shows a 1024 px reference-image workflow.
- Edit Turbo 1.5K shows a 1536 px reference-image workflow.

Run is enabled only when the active model has a valid prompt and, for Edit, a decoded reference.
Model switching removes unused modules before loading the next model. Completed images remain under
user control and are not automatically substituted as edit references.

## Automation

Native automation uses the `burn-image-viewer` CLI. Browser qualification uses explicit query/env
routes in the checked-in harnesses; ordinary users interact through the Bevy UI. Headless routes
emit structured reports and terminal markers and are not part of the interactive control surface.

## Rendering safety

Inference work runs outside the Bevy update loop. The WebAssembly frontend temporarily suspends
surface acquisition while heavy compute owns the shared WebGPU queue, then restores it before
presenting output. A web-only DOM safety overlay covers the intentionally frozen canvas and shows
the current stage, step, or verified-cache activity from the same structured runtime progress
events; native windows are never suspended by this policy. Gate violations, device loss,
uncaptured errors, and cleanup failures fail qualification.

## Deployment readiness

Pages deployment verifies:

- the generated JS/Wasm/icon package;
- manifest seals and component pins;
- layout structure and the complete immutable URL/size/SHA-256 inventory;
- exact size and SHA-256 for compact direct files;
- exact Range/CORS/cache headers for bounded first/last physical-part samples in every bundle;
- source/workflow contracts.

The page is deployed only after readiness succeeds. CDN artifacts are published independently and
must already exist at their immutable URLs. A manual dispatch with `full_cdn_audit=true` retains the
expensive publication-time audit that downloads and hashes every unique payload; normal source
deploys intentionally do not repeat that roughly 103 GB transfer.
