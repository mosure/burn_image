# Web deployment

## build the concrete Boogu frontend

Install the Cargo.lock-matched CLI used by CI with
`cargo install wasm-bindgen-cli --version 0.2.127 --locked`.

```bash
cargo build -p bevy_burn_image --target wasm32-unknown-unknown \
  --profile wasm-release --no-default-features --features boogu-web --locked --lib
mkdir -p crates/bevy_image/www/out
wasm-bindgen --target web --out-dir crates/bevy_image/www/out \
  --out-name bevy_burn_image \
  target/wasm32-unknown-unknown/wasm-release/bevy_burn_image.wasm
rg '^export function (start_boogu_web|provide_reference_image)' \
  crates/bevy_image/www/out/bevy_burn_image.js
npx --yes serve crates/bevy_image/www --listen 8080
```

Open a URL with the remote sealed bundle and release selection:

```text
http://localhost:8080/?variant=turbo&profile=f16-qwen-vision-f32
```

With `artifacts` omitted, that selection resolves exactly to
`https://aberration.technology/model/boogu-image-0.1-turbo-f16-qwen-vision-f32`.
An explicit URL override remains available for local or pre-publication testing, for example
`artifacts=http%3A%2F%2Flocalhost%3A8080%2Fartifacts%2Fboogu-image-0.1-turbo-f16-qwen-vision-f32`.

`variant` accepts `turbo` or the 1K `edit-turbo`; `profile` accepts all four importer profile
slugs. `edit-turbo-1k5`/`1k5` is deliberately rejected before artifact loading: the 1.5K release
is native-WGPU only until browser numerical and performance parity are validated. The
complete hardware-browser run below validates only Turbo with `f16-qwen-vision-f32`. Browser Edit
and the other three storage profiles are implemented but remain experimental and unvalidated; do
not treat their presence in the query surface as a support or parity claim. The exact manifest
bundle id is always the one `{model_name}` path segment beneath the canonical CDN root. WebGPU
requires a secure context (`https://` or localhost). The artifact server must allow the `Range`
request header through CORS, return `206`, and expose `Content-Range` to browser code.

## implemented contracts

- Bevy and Burn share the same WGPU adapter/device/queue through `bevy_burn`.
- An explicitly reported CPU adapter/device and the CPU tensor backend are rejected. WebGPU may
  privacy-redact both hardware adapters and software implementations as `Other`, so hardware
  execution claims require an external scoped GPU-process attestation like the one below.
- The UI exposes prompt/instruction, mode/model, size, seed, reference selection, run/cancel,
  structured progress/errors, result display, and PNG download.
- The Boogu BrowserWebGpu descriptor accepts exactly 256×256 (65,536 pixels), supplies that size
  when request dimensions are omitted, and exposes only capability-compatible UI presets. Native
  WGPU continues to use the core model descriptor: 1K releases default to 1024×1024 and the
  native-only Edit-Turbo 1.5K release defaults to 1536×1536.
- `BrowserBooguFactory` constructs real async Qwen, VAE, and denoiser stage sources on the shared
  WebGPU device; no placeholder or CPU tensor-backend fallback exists. Adapter metadata alone is
  not used to distinguish privacy-redacted hardware from a software WebGPU implementation.
- `fetch_browser_range` performs the actual browser `fetch`, requires HTTP 206 and a matching
  exposed `Content-Range`, enforces the 16 MiB response cap, and returns one requested chunk.
- `BrowserStageShardReader` aggregates at most one sealed semantic object (hard cap 256 MiB),
  verifies its exact size and SHA-256, and releases its host bytes after upload/apply.
- Async Qwen and four-step DMD execution suspend at network boundaries, report artifact/stage/step
  progress, and check cancellation before every range and semantic boundary.
- VAE encode and decode are independently loaded with the released F32 force-upcast policy.
- Q8 profiles dequantize Qwen Col-layout stages through F16 before transpose and widen them to the
  browser's F32 execution policy; denoiser Row-layout Q8 stays device-quantized with F32
  activations.

The implemented per-stage sequence is:

```text
fetch bounded ranges -> verify object -> upload stage -> execute/sync -> release -> next stage
```

The synchronous native directory factory is not compiled for Wasm. Browser Qwen and denoiser
sources deliberately do not use the native retaining wrappers.

Output currently crosses the model boundary as a validated host image and is then uploaded into a
Bevy texture. A zero-readback, device-resident result handoff is not implemented.

## current browser validation result

Chrome 151 completed one real 256×256 Turbo request with the `f16-qwen-vision-f32` bundle on the
workstation's hardware Vulkan WebGPU path. The scoped Chrome GPU process, framebuffer allocation,
and SM utilization provide the hardware attestation; the privacy-redacted adapter label alone does
not. This is positive browser inference evidence, but the emitted report deliberately says
`numerical_parity_claimed=false`: it was an ordinary prompt/seed request, not an upstream tensor
fixture replay. The production *headless* Bevy swap-chain smoke remains a separate failure.

### surface-free full request

`headless=infer` runs the ordinary request resolver and the concrete `BrowserBooguFactory` without
creating a render surface. It accepts only Turbo generation and fails closed unless `prompt`,
`seed`, `width=256`, and `height=256` are supplied. The browser model backend is raw `CubeBackend`
rather than Burn's fused alias. Floating stages are adapted to F32, and every asynchronous semantic
source awaits the CubeCL `ComputeClient::sync()` future instead of blocking Wasm's event loop.

The validated launch was headful so Chrome used the real GPU rather than SwiftShader:

```bash
export BURN_IMAGE_CHROME_PROFILE=/tmp/burn-image-chrome-profile
export DISPLAY=:0
google-chrome \
  --user-data-dir="$BURN_IMAGE_CHROME_PROFILE" \
  --no-sandbox --disable-dev-shm-usage \
  --ozone-platform=x11 --use-angle=vulkan \
  --enable-features=Vulkan,ForceEnableWebGpuInterop \
  --enable-unsafe-webgpu --enable-logging=stderr \
  'http://127.0.0.1:8080/?headless=infer&variant=turbo&profile=f16-qwen-vision-f32&prompt=A%20matte%20red%20cube%20centered%20on%20a%20plain%20white%20studio%20background%2C%20soft%20shadow%2C%20front%20view.&seed=42&width=256&height=256'
```

The scoped Chrome GPU process appeared in `nvidia-smi` as C+G, peaked at 3,008 MiB framebuffer
memory and 49% SM utilization, and exited after capture. The request made 22,842 exact HTTP 206
responses totaling 94,728,981,689 bytes (88.223239 GiB). Peak Wasm linear memory was
2,683,895,808 bytes and peak scoped aggregate RSS was 8,167,008 KiB. The unusually large transfer
is expected from the current non-retaining browser policy: Qwen streams once, the denoiser streams
for each of four DMD steps, and the VAE decoder streams once.

The report recorded these stage times:

| Stage | Seconds |
| --- | ---: |
| processing | 0.011 |
| Qwen | 201.399 |
| four-step DMD | 1,001.937 |
| VAE decode | 3.935 |
| host output | 0.014 |
| total request | 1,207.301 |

The output encoder produced a 60,926-byte PNG with SHA-256
`cb392986a8aef566fb36e57c5ab6a0af366cf7607d46482bae2d36f90d28df09` and attached it as a visible
Blob image plus download link. The Blob could not be independently downloaded from that non-CDP
run, so the exact pixel dimensions/content were not separately inspected outside the model report.
The report attested the pinned model revision, sealed bundle digest
`4f94cf68c00af12d5de486db4d316ce889d6d21e78913a1c74edab4bd0119ce3`, verified artifacts, and
`burn-webgpu/browser-async-stage-streamed` provenance.

Full-run evidence:

- Chrome log: `/tmp/burn-image-hw-full256-chrome.A9gL4G.log`, SHA-256
  `2cc0156b3f54af9ab6b8585c81a509dd21076feb506344887d3866597f74db30`;
- exact-Range server log: `/tmp/burn-image-hw-full256-server.lmhpSM.log`, SHA-256
  `57fca17712fa684cb59beb0b90df9cade83e959d2d35ec66281819f03dc491b5`;
- resource samples: `/tmp/burn-image-hw-full256-resources.8FNQ4R.tsv`, SHA-256
  `15c5f9b0439ccd51e9dd4a8bdbbae4e54b1475844e97aeb64d604de380faeb9e`.

The package used by that run contained 119,012 bytes of JavaScript (SHA-256
`8b9e13fbe14e458072c2c605ce0f534ba447a97229ba2093c3f97916442b670c`) and a 34,486,862-byte Wasm
module (SHA-256 `6d408c6de5198a0f59504a34b0b49b3d924007e6caf2d96bd1731f2923ab4416`).

### stage probe and F16 capability

`headless=bootstrap` is the shorter F32 diagnostic. It verifies one sealed 8,448-byte Qwen
final-norm object, submits the module, awaits the nonblocking device barrier, and reads back
4,096/4,096 finite F32 values. It remains useful for deployment checks before a 20-minute full
request.

`headless=f16-probe` uses the same stage but requires WebGPU `shader-f16` and preserves the Qwen
stage dtype. On the attested hardware path both the adapter and requested device reported
`shader-f16=false`, so it failed closed before artifact construction; no F16 kernel or readback was
run. The scoped GPU PID was present, but the server saw zero artifact 206 responses. Evidence:

- Chrome log: `/tmp/burn-image-f16-probe-chrome.8nB5DM.log`, SHA-256
  `555d93cf3a13186928af3fa18364cab55a22a0a385f47d3aeb8153317f3615ec`;
- server log: `/tmp/burn-image-f16-probe-server.6x3eqD.log`, SHA-256
  `60a204abf9b6e7c06881428fa595f92b2e1f9340acd187abc587f047e10dfff7`;
- GPU samples: `/tmp/burn-image-f16-probe-gpu.u1X0uT.log`, SHA-256
  `f6ff6336e9906a3c9ef85ffcd9c5377ff54684747a05f891aac91e5e870353b6`.

The browser production policy therefore remains all-F32 execution even when the sealed transport
profile stores F16 weights. Earlier fused-shader, blocking-mapper, and blocking-barrier failures
have been fixed; they are historical implementation evidence, not the current runtime result.

### production shared-device window

The production headless smoke creates the Bevy window and selects `BrowserWebGpu`, but this
workstation then fails Chrome's swap-chain shared-image allocation:

```text
Could not find SharedImageBackingFactory ... WebgpuSwapChainTexture
Caught DeviceLost error: Destroyed Device was destroyed.
Quitting the application due to DeviceLost RenderError
```

The failure reproduces with the app-only build (no `bevy_burn`, Burn bridge, or Boogu runtime), and
with Bevy's `WgpuSettingsPriority::WebGPU`. The captured Chrome log is 21,091 bytes with SHA-256
`711a81abd2d8a13f04fece38842b1e50174a7b4150b414023c58c30b679f7be5`. This is a failed
window/surface smoke, not a failure of the successful surface-free model request above. A Chrome
headless no-surface attempt also selected SwiftShader and is excluded from hardware evidence.

### limits and remaining parity gate

Transport remains bounded by a 4 MiB default chunk, 16 MiB hard response cap, 4 MiB manifest cap,
and 256 MiB hard semantic-object cap. That object cap is not a WGPU binding limit. The largest
current post-load tensor is a 25,323 by 4,096 embedding row adapted to F32, or 414,892,032 bytes.
Only the Boogu app requests 512 MiB minimum storage-buffer/buffer limits; the model-neutral app
keeps WebGPU's portable baseline. The full run reported applied limits of 2,147,483,644 bytes for
`max_storage_buffer_binding_size` and 4,294,967,292 bytes for `max_buffer_size`. Factory startup
fails closed below the exact model requirement.

Concrete native/browser factories implement only `ArtifactCachePolicy::UseCached`; `Refresh` and
`Bypass` fail closed. The current checked package, which adds the exact-256 UI/default contract and
the fail-closed F16 capability route, contains 119,078 bytes of JavaScript (SHA-256
`5cd51de59b9ab34195227e21f1ee3348995d696bed993885ee4fef9d91c14ffd`) and a 34,490,961-byte Wasm
module (SHA-256 `af664d6cc17a09c625eaf7f5c52ce5f2c3fdf86f8c01c6af0c5446fe6e624644`). Browser numerical
parity still requires the upstream fixture gates; neither an ordinary full request nor a finite
stage readback is presented as that result.
