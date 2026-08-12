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

`BurnImageShellPlugin` installs a usable model-neutral control panel rather
than a demo-only status screen. It provides an editable multiline prompt or
instruction, generate/edit switching, initialized-model selection, common
square/landscape/portrait sizes, a numeric `u64` seed, run/cancel actions,
structured artifact/stage/step progress, surfaced runtime and I/O errors, and
the latest generated image. Loading a reference selects edit mode and an
edit-capable initialized model when one is available. Size initialization and
cycling are capability-aware: the BrowserWebGpu Boogu descriptor exposes only
the validated 256×256 preset, while native follows the core model descriptor. The native
Edit-Turbo 1.5K release starts at its 1536×1536 model default and exposes its official
aspect-ratio presets through the same capability-filtered control.

On native builds, drop a PNG, JPEG, or WebP file onto the window and use
**Save PNG** to write `burn-image-<job>.png` in the current directory. In the
browser, **Reference** opens the host file picker and **Save PNG** creates a
Blob download. Both paths decode/encode through the same bounded host-image
messages. A missing model runtime is shown explicitly; pressing **Run** cannot
produce a placeholder result.

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
`https://aberration.technology/model/<exact-manifest-bundle-id>`, incrementally verifies every
declared file, and caches the completed immutable bundle at
`~/.burn_image/models/<exact-manifest-bundle-id>/`. `BURN_IMAGE_CACHE_DIR` overrides the broad
cache root, `BURN_IMAGE_MODEL_CACHE_DIR` overrides the exact bundle directory, and
`BURN_IMAGE_MODEL_BASE_URL` is an explicit custom-CDN diagnostic override.
`BURN_IMAGE_MODEL_MANIFEST_URL` overrides only the exact manifest request; payload URLs still
resolve from the selected source/base URL. The factory receives the exact shared
`burn_wgpu::WgpuDevice` only after Bevy/Burn GPU attestation. Its loader
validates the sealed manifest and production row-sliced inventory, then the
worker executes one request at a time. The default `native-high-vram` policy wraps the verified
Qwen source in `RetainingQwen3VlStageSource`, so each stage is loaded once and later requests clone
only shared WGPU handles. It independently loads and retains VAE encoder/decoder stages and reuses
one resident denoiser across all four DMD steps. The explicit `native-layer-streamed` policy
minimizes GPU weight residency but rereads Qwen and VAE per request and denoiser stages on each DMD
step. Neither path substitutes a mock result.

An embedder selecting the parity-qualified native high-VRAM `f16-qwen-vision-f32` policy must call
`burn_boogu::configure_native_full_autotune()` on the main thread before Bevy creates or imports its
WGPU device. For Turbo and 1K Edit this selects retained Qwen q128 with synchronization deferred
to the mandatory Qwen-stage boundary, padded-blackbox `p4/kv1/q1` denoiser
q8192 with strict-F32 RMSNorm, preserved-F16 VAE q1024, and safe F32 GroupNorm accumulation.
Edit-Turbo 1.5K retains its
separately qualified q16384/VAE-q4096 bounds. `NativeBooguFactory` fails closed if full autotune was
omitted or selected too late. The packaged `burn-image-viewer` binary performs this setup
automatically. Browser, layer-streamed, Q8, and all-F16 paths retain their independent policies.

Run a converted Turbo bundle with:

```sh
cargo run -p bevy_burn_image --release --features boogu-native \
  --bin burn-image-viewer -- \
  --variant turbo \
  --profile f16-qwen-vision-f32 \
  --residency native-high-vram
```

Add `--artifacts .artifacts/boogu-image-0.1-turbo-f16-qwen-vision-f32` to bypass downloading and
use an already verified local conversion.

Use `--variant edit-turbo` with the matching Edit-Turbo bundle. The viewer also accepts the
distinct native-only `--variant edit-turbo-1k5` release with a
`boogu-image-0.1-edit-turbo-1k5-<profile>` bundle. Its omitted-size default is 1536×1536; the
official native presets are 1536×1536, 1264×1856, 1856×1264, 1344×1744, 1744×1344, 1392×1696,
1696×1392, 1152×2032, 2032×1152, and 2368×992, bounded to 2,360,832 pixels. The viewer
accepts only the authenticated `f16-qwen-vision-f32` profile for 1.5K; other profiles fail before
artifact loading. Turbo and 1K Edit retain their four-profile selection surface. Loading and
inference failures remain visible in the UI. The 1.5K release is also restricted to the
high-VRAM retained policy whose exact attention/VAE configuration passed its native gate;
layer-streamed 1.5K is rejected until separately validated. The 1536×1536 default is the
checkpoint-gated and benchmarked preset; the other official presets are exposed as bounded model
configurations but do not inherit that shape-specific evidence.

The `BooguRuntimeFactory`/`BooguRuntime` injection boundary remains public for embedders.
`BrowserBooguFactory` is the concrete Wasm implementation: it uses bounded HTTP ranges plus
digest-verified async Qwen/VAE/denoiser sources and keeps browser model stages non-retaining. The
synchronous native directory factory is not compiled for Wasm. Its query/configuration surface can
select 1K Edit and all four importer profiles, but explicitly rejects Edit-Turbo 1.5K before any
browser artifact loading because that release has no browser numerical/performance validation.
Only Turbo with `f16-qwen-vision-f32` has completed
the hardware-browser run documented below. Browser Edit and the remaining storage profiles are
experimental and unvalidated rather than supported parity configurations.

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
Qwen/VAE/denoiser sources and passes exact ranges through `fetch_browser_range`; it retains only
one digest-verified semantic object at a time. Select the bundle with `variant`, `profile`, and
`artifacts` query parameters as documented in the repository [web deployment
notes](https://github.com/mosure/burn_image/blob/main/docs/web.md). Native file helpers remain
excluded from Wasm builds.

The repository's GitHub Pages workflow packages the tracked `www/` shell and ignored generated
`www/out/` bindgen output only; model objects stay on the Aberration CDN. During runtime build and
inference, Wasm dispatches structured `burn-image-runtime` and `burn-image-progress` DOM events.
The shell turns them into exact current-object bytes/shard progress, transfer rate and ETA,
verified-object totals, stage/step state, terminal errors, and a manual full-runtime reload action.

The Wasm feature sets compile and package. On the externally attested headful X11/Vulkan hardware
path, `headless=infer` completed one real 256×256 Turbo request with the
`f16-qwen-vision-f32` bundle through streamed Qwen, all four DMD steps, and VAE decode; it encoded
and attached a 60,926-byte PNG Blob. The scoped Chrome GPU process, framebuffer allocation, and SM
utilization establish hardware use; the adapter label alone does not. The report explicitly sets
`numerical_parity_claimed=false`, and the Blob was not independently downloaded for pixel-level
inspection. Chrome 151's production *headless* Bevy window still loses its WebGPU device while
creating the swap-chain shared image, so that separate UI/surface smoke remains failed.

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
