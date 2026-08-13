# bevy_burn_image

Bevy 0.19 native/WebGPU frontend contracts for the model-neutral
[`burn_image`](../burn_image) API. The package is named `bevy_burn_image`
because `bevy_image` is already an engine crate.

The crate owns:

- generation/edit editor state and validated ECS request routing;
- structured progress, cancellation, completion, and failure state;
- one shared Bevy/Burn WGPU adapter, device, instance, and queue through
  `bevy_burn`;
- RGBA8 Bevy asset materialization plus exact/latest UI image-view components;
- byte-based image load/encode operations on native and Wasm, with optional
  native filesystem helpers and download-ready filename/MIME/byte messages;
- exact HTTP range descriptions and transactional, SHA-256-verified artifact
  streaming with a 16 MiB hard per-chunk cap.

Model-neutral inference remains outside the core frontend. `ImageRunnerPlugin`
connects a native worker or browser-local WebGPU task through canonical
`burn_image` contracts: `ModelDescriptor`, `ImageRequest`, `CancellationToken`,
`ProgressEvent`, `RuntimeError`, and `ImageOutput`. The runner must advertise a
native-WGPU or browser-WebGPU execution kind, streamed progress, cooperative
cancellation, and host image results. There is no mock runner or placeholder
image path. Advanced model plugins may consume `ImageJobDispatched` directly,
but must provide an equivalent ready `ImageRunnerStatus` and schedule dispatch
inside `ImageFrontendSet::Dispatch`.

## Viewer controls

`BurnImageShellPlugin` installs a compact, responsive control panel and a
camera-backed image viewport. The panel provides a multiline prompt or edit
instruction, generate/edit and initialized-model selection, capability-aware
size presets, a numeric `u64` seed, reference-image input, Run, Cancel, Save
PNG, Fit image, and 100% controls. Loading a reference selects edit mode and an
edit-capable model when available. The BrowserWebGpu Boogu descriptor exposes
only its validated 256x256 preset; native follows the selected model descriptor.
The native Edit-Turbo 1.5K release starts at 1536x1536 and exposes its official
aspect-ratio presets through the same filtering.

On native builds, **Reference** opens the operating-system file dialog; a PNG,
JPEG, or WebP can also be dropped on the window. In the browser, **Reference**
opens the browser file picker, and the supplied web host accepts the same files
by drag and drop. Both paths enforce the same byte bound and use the same image
decode messages. **Save PNG** writes `burn-image-<job>.png` in the native
current directory or starts a browser Blob download.

The latest reference or generated image is fitted automatically. Drag with the
left or middle mouse button to pan, use the wheel or a touchpad pinch to zoom
toward the pointer, select **Fit image** to recenter and contain the image, or
select **100%** for one image pixel per logical display pixel. Camera input is
limited to the image viewport and is disabled while a text field or the panel
owns input. The panel moves below the viewport on narrow windows and remains
scrollable when vertical space is limited.

Progress remains visible throughout shared-GPU setup, manifest and shard
transfer, model-stage initialization, inference steps, output preparation,
cancellation, and failures. The headline, detail text, and progress bar report
the active phase instead of treating device creation or a downloaded manifest
as model readiness. A missing model runtime is shown explicitly; pressing
**Run** cannot produce a placeholder result.

## Boogu adapter

The optional `boogu` feature adds `BooguAdapterPlugin`. It obtains each model id,
revision, task, dimensions, control bounds, and supported numeric formats from
`burn_boogu::boogu_model_descriptor`, then narrows the advertised numeric
format to the artifact profile that this adapter instance actually loads. It
also delegates request resolution to `burn_boogu::resolve_request`, constructs
the upstream `InstructionPolicy`, passes the selected Burnpack
profile, and validates completed output provenance against the dispatched
revision/profile and WGPU backend.

`boogu-native` adds `NativeBooguFactory`, the real verified-filesystem-artifact implementation
used by `burn-image-viewer`. With no `--artifacts` override, the viewer resolves
`https://aberration.technology/model/<exact-production-bundle-id>`, incrementally verifies every
declared file, and caches the completed immutable bundle at
`~/.burn_image/models/<exact-manifest-bundle-id>/`. `BURN_IMAGE_CACHE_DIR` overrides the broad
cache root, `BURN_IMAGE_MODEL_CACHE_DIR` overrides the exact bundle directory, and
`BURN_IMAGE_MODEL_BASE_URL` is an explicit custom-CDN diagnostic override. Canonical downloads use
the production bundle id as their cache key; custom remotes default to a separate legacy tuple key so the
two sources cannot silently satisfy each other. Set `BURN_IMAGE_MODEL_CACHE_DIR` when multiple
custom sources need independent exact cache directories.
`BURN_IMAGE_MODEL_MANIFEST_URL` overrides only the exact manifest request; payload URLs still
resolve from the selected source/base URL. The factory receives the exact shared
`burn_wgpu::WgpuDevice` only after Bevy/Burn GPU attestation. Its loader
validates the sealed manifest and production row-sliced inventory, then the
worker executes one request at a time. The default production `native-high-vram` policy verifies,
materializes, and synchronizes every required Qwen, VAE, and denoiser stage before reporting
**Ready**. Inference then clones retained WGPU module handles: no forward pass rereads, rehashes,
decodes, or reuploads model weights, and one resident denoiser serves all four DMD steps. The
explicit `diagnostic-layer-streamed` policy is local-bundle-only and rereads Qwen and VAE per
request plus denoiser stages on each DMD step. It is a memory diagnostic, not a supported
production path. Neither policy substitutes a mock result or CPU tensor backend.

An embedder selecting the parity-qualified native high-VRAM production policy (sealed internally
as `f16-qwen-vision-f32`) must call
`burn_boogu::configure_native_full_autotune()` on the main thread before Bevy creates or imports its
WGPU device. For Turbo and 1K Edit this selects retained Qwen q128 with synchronization deferred
to the mandatory Qwen-stage boundary, padded-blackbox `p4/kv1/q1` denoiser
q8192 with strict-F32 RMSNorm, preserved-F16 VAE q1024, and safe F32 GroupNorm accumulation.
Edit-Turbo 1.5K retains its
separately qualified q16384/VAE-q4096 bounds. `NativeBooguFactory` fails closed if full autotune was
omitted or selected too late. The packaged `burn-image-viewer` binary performs this setup
automatically. Browser and diagnostic storage/residency paths retain their independent policies.

Run a converted Turbo bundle with:

```sh
cargo run -p bevy_burn_image --release --features boogu-native \
  --bin burn-image-viewer -- \
  --variant turbo \
  --profile production \
  --residency native-high-vram
```

Add `--artifacts .artifacts/boogu-image-0.1-turbo` to bypass downloading and
use an already verified local conversion.

Use `--variant edit-turbo` with the matching Edit-Turbo bundle. The viewer also accepts the
distinct ordinary-UI-native `--variant edit-turbo-1k5` release with a
`boogu-image-0.1-edit-turbo-1k5` bundle. Its omitted-size default is 1536x1536; the
official native presets are 1536x1536, 1264x1856, 1856x1264, 1344x1744, 1744x1344, 1392x1696,
1696x1392, 1152x2032, 2032x1152, and 2368x992, bounded to 2,360,832 pixels. The viewer
accepts only the authenticated `f16-qwen-vision-f32` profile for 1.5K; other profiles fail before
artifact loading. Turbo and 1K Edit retain their four-profile selection surface. Loading and
inference failures remain visible in the UI. The 1.5K release is also restricted to the
high-VRAM retained policy whose exact attention/VAE configuration passed its native gate; the
diagnostic layer-streamed policy rejects 1.5K. The 1536x1536 default is the
checkpoint-gated and benchmarked preset; the other official presets are exposed as bounded model
configurations but do not inherit that shape-specific evidence.

The `BooguRuntimeFactory`/`BooguRuntime` injection boundary remains public for embedders.
`BrowserBooguFactory` is the concrete Wasm implementation: it uses bounded HTTP ranges plus
digest-verified async Qwen/VAE/denoiser sources. Production defaults to
`browser-high-vram-resident-dense-f32`: before **Ready**, it sequentially verifies and uploads one
bounded semantic object at a time, releases each host payload, and retains every initialized
dense-F32 WebGPU stage. Forward execution therefore performs no repeated model-weight download,
hash/decode, or host-to-device transfer. This needs workstation-class WebGPU memory and fails
before **Ready** if its resource plan, allocation, or device synchronization fails; it never falls
back to CPU. `browser-layer-streamed-diagnostic` is an explicit low-memory diagnostic. The
synchronous native directory factory is not compiled for Wasm.

The opt-in, no-surface 1536-square qualification route retains exactly the 48 verified denoiser
stages across four DMD steps, then clears them before its exact VAE decoder. Its historical
schema-v1 flat-bundle replay passed on the pinned RTX PRO 6000 Blackwell/Chrome 151 stack. The new
modular closure still needs a fresh replay, and that result does not qualify the ordinary resident
UI path, browser performance, other dimensions, or another adapter/browser. The ordinary UI still
rejects Edit-Turbo 1.5K; browser 1K Edit and diagnostic storage profiles remain experimental.

The executable entry is deliberately factory-requiring, so there is no mode
that substitutes a mock image:

```rust,ignore
fn main() {
    let settings = bevy_burn_image::BooguAdapterSettings::verified_default(source);
    let factory = MyNativeBooguRuntimeFactory::new();
    bevy_burn_image::run_boogu_cli(settings, factory);
}
```

For Wasm, call `app::build_boogu_app(settings, factory).run()` from the host's
`#[wasm_bindgen(start)]` function. Enabling `boogu` disables this crate's
model-neutral automatic Wasm start specifically to make factory injection
mandatory.

## Device policy

`BurnImageFrontendPlugin` installs `BevyBurnBridgePlugin` and records a ready
`BackendStatus` only after Burn has been initialized from Bevy's WGPU objects.
Submitting a job while the bridge is initializing, absent, or lost produces a
failed job and `ImageJobRejected`. An explicitly reported CPU WGPU device and the CPU tensor
backend are rejected, and there is no CPU tensor-backend fallback state. Because WebGPU can report
both privacy-redacted hardware and software implementations as `Other`, hardware claims require
external scoped GPU-process attestation rather than relying on that adapter label.

## Artifact streaming

`StreamingArtifactLoader` requests one exact range at a time, hashes borrowed
bytes without retaining them, and commits a `TransactionalArtifactSink` only
after the complete file digest passes. Heavy profiles should make each
manifest file a bounded shard, release its loader after commit, then proceed to
the next shard. On Wasm, `fetch_browser_range` performs the actual HTTP request,
requires 206 plus the matching `Content-Range`, and rejects a response above 16
MiB. This prevents a complete model bundle from entering Wasm linear memory.

## Builds

```sh
# Native viewer and library
cargo run -p bevy_burn_image --bin burn-image-viewer

# Compile the injected Boogu adapter surface
cargo check -p bevy_burn_image --features boogu

# Compile the real native local-artifact worker and CLI
cargo check -p bevy_burn_image --features boogu-native --all-targets

# Browser library/viewer surface
cargo check -p bevy_burn_image \
  --target wasm32-unknown-unknown \
  --no-default-features --features web

# Concrete Boogu browser factory and optimized package
cargo build -p bevy_burn_image --target wasm32-unknown-unknown \
  --profile wasm-release --no-default-features --features boogu-web --locked --lib
mkdir -p crates/bevy_image/www/out
wasm-bindgen --target web --out-dir crates/bevy_image/www/out \
  --out-name bevy_burn_image \
  target/wasm32-unknown-unknown/wasm-release/bevy_burn_image.wasm
rg '^export function (start_boogu_web|provide_reference_image)' \
  crates/bevy_image/www/out/bevy_burn_image.js
npx --yes serve crates/bevy_image/www --listen 8080

# Contract-only library without the window or Burn bridge
cargo test -p bevy_burn_image --no-default-features
```

The browser host must provide an element with id `burn-image`. `BrowserBooguFactory` uses async
Qwen/VAE/denoiser sources and passes exact ranges through `fetch_browser_range`. Preload holds at
most one digest-verified semantic payload in Wasm at a time, releases it after upload, and keeps
the resulting production modules resident on WebGPU. Select the bundle and policy with `variant`,
`profile`, `residency`, and `artifacts` query parameters as documented in the repository
[web deployment notes](https://github.com/mosure/burn_image/blob/main/docs/web.md). Native file
helpers remain excluded from Wasm builds.

The repository's GitHub Pages workflow packages the tracked `www/` shell and ignored generated
`www/out/` bindgen output only; model objects stay on the Aberration CDN. During runtime build and
inference, Wasm dispatches structured `burn-image-runtime` and `burn-image-progress` DOM events.
The shell turns them into exact current-object bytes/shard progress, transfer rate and ETA,
verified-object totals, stage/step state, terminal errors, and a manual full-runtime reload action.

The Wasm feature sets compile and package. Before resident-by-default production was introduced,
an externally attested headful X11/Vulkan `headless=infer` run completed one real 256x256 Turbo
request with the `f16-qwen-vision-f32` bundle through the then-layer-streamed Qwen, four DMD steps,
and VAE decode; it encoded and attached a 60,926-byte PNG Blob. That evidence proves real WebGPU
execution for the historical policy, not numerical or performance qualification of the new
ordinary resident mode. Chrome 151's production *headless* Bevy window still loses its WebGPU
device while creating the swap-chain shared image, so that separate UI/surface smoke remains
failed.

Browser execution uses raw CubeCL without Burn fusion and adapts floating model stages to F32.
The hardware adapter and requested device both reported `shader-f16=false`; the explicit
`headless=f16-probe` therefore rejects before artifact construction rather than pretending the
mixed F16 execution policy is available. The Boogu app requests 512 MiB storage-buffer and buffer
limits for the current 414,892,032-byte largest post-load tensor; the model-neutral app keeps
portable baseline limits. Concrete factories accept only `ArtifactCachePolicy::UseCached` until
refresh/bypass semantics are implemented. See the repository [web deployment
notes](https://github.com/mosure/burn_image/blob/main/docs/web.md) for the exact commands, hashes,
resource bounds, and distinction between no-surface inference, the Bevy window smoke, and
numerical parity.
