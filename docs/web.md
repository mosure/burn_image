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
http://localhost:8080/?variant=turbo&profile=production
```

The tracked static host lives in `crates/bevy_image/www/`; generated bindgen JavaScript and Wasm
remain under the ignored `www/out/` directory. `.github/workflows/deploy-pages.yml` mirrors the
repository's local package command and never copies model objects into the Pages artifact. Manual
dispatch runs the complete deployment; `main` pushes deploy automatically only after the
repository variable `BURN_IMAGE_PAGES_READY=true` is set. Before setting it, publish the canonical
CDN bundle and enable Pages with **GitHub Actions** as its source. The deploy job probes both the
manifest and one weight object for the exact `206`, CORS, exposed `Content-Range`, and immutable
cache headers required by the bounded Range reader. A missing/private bundle or incomplete CDN
header policy therefore blocks deployment instead of publishing a page that cannot load its
model. The expected project URL is `https://mosure.github.io/burn_image/`.

The HTML shell listens for the Wasm-emitted `burn-image-runtime` and `burn-image-progress` custom
events. Its accessible loading panel distinguishes runtime/manifest preparation, current semantic
object transfer and SHA-256 verification, and GPU inference stages. It displays exact current
bytes, actual zero-based manifest shard position rendered as one-based UI text, rolling transfer
rate/current-object ETA, cumulative request bytes, verified-object count, and DMD steps. The panel
does not invent a whole-run percentage: heterogeneous Qwen, VAE, and denoiser objects have very
different transfer, verification, upload, and synchronization costs. Production performs that work
once during resident preload; all four denoiser passes then reuse the cached WebGPU modules.
`aria-live` announcements occur at lifecycle boundaries, while high-frequency 4 MiB range updates
are coalesced to animation frames.

With `artifacts` omitted, that selection resolves exactly to
`https://aberration.technology/model/boogu-image-0.1-turbo`.
An explicit URL override remains available for local or pre-publication testing, for example
`artifacts=http%3A%2F%2Flocalhost%3A8080%2Fartifacts%2Fboogu-image-0.1-turbo`.

`variant` accepts `turbo` or the 1K `edit-turbo`. The default `profile=production` selector maps to
the precise sealed `f16-qwen-vision-f32` contract; the old precise selector remains an alias for
existing links. Residency defaults to `resident`; `residency=layer-streamed-diagnostic` is the
explicit low-memory diagnostic. Diagnostic storage profiles and layer streaming require an
explicit `artifacts=` URL and are not canonical CDN production paths. The ordinary UI still rejects
`edit-turbo-1k5`/`1k5` before artifact loading. A separate no-surface gate historically qualified
exact 1536×1536 numerical parity for the former schema-v1 flat closure on the pinned RTX PRO 6000
Blackwell/Chrome 151 stack. The schema-v2 modular closure requires a fresh real-artifact run and is
not yet qualified. Neither result enables ordinary resident UI support or qualifies browser
performance, other shapes, or other browser/GPU stacks. Browser 1K Edit remains experimental and
unvalidated.
WebGPU requires a secure context (`https://` or localhost).
The artifact server must allow the `Range` request header through CORS, return `206`, and expose
`Content-Range` to browser code.

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
  native UI's Edit-Turbo 1.5K release defaults to 1536×1536.
- `BrowserBooguFactory` constructs real async Qwen, VAE, and denoiser stage sources on the shared
  WebGPU device; no placeholder or CPU tensor-backend fallback exists. Adapter metadata alone is
  not used to distinguish privacy-redacted hardware from a software WebGPU implementation.
- The production factory reports **Ready** only after it has materialized and synchronized every
  required Qwen, VAE, and denoiser stage as dense F32 WebGPU modules. A forward pass performs zero
  model-weight HTTP reads, verification/decoding, or host-to-device uploads.
- `fetch_browser_range` performs the actual browser `fetch`, requires HTTP 206 and a matching
  exposed `Content-Range`, enforces the 16 MiB response cap, and returns one requested chunk.
- `BrowserStageShardReader` aggregates at most one sealed semantic object (hard cap 256 MiB),
  verifies its exact size and SHA-256, uploads/materializes it, and releases its host bytes. Wasm
  linear memory never holds a complete bundle even though initialized modules accumulate on GPU.
- Async preload suspends at network boundaries, reports artifact/stage progress to both Bevy and
  the DOM loading panel, and checks cancellation before every range and semantic boundary. Resident
  Qwen, VAE, and four-step DMD forward execution then reports compute stages without refetching.
- VAE encode and decode are independently materialized with the released F32 force-upcast policy.
- Production is the dense-F32 execution contract. Q8 and all-F16 storage selections are explicit
  layer-streamed diagnostics and do not inherit production residency or qualification claims.

The implemented per-stage sequence is:

```text
fetch bounded ranges -> verify one object -> upload/materialize retained stage
-> release host payload -> next object -> synchronize resident graph -> Ready
```

The synchronous native directory factory is not compiled for Wasm. Browser sources enable their
async retention caches for production and disable retention only for the explicit layer-streamed
diagnostic. Resource planning, allocation, or synchronization failure prevents **Ready**; there is
no CPU fallback. Expect workstation-class/high-VRAM hardware.

Output currently crosses the model boundary as a validated host image and is then uploaded into a
Bevy texture. A zero-readback, device-resident result handoff is not implemented.

## browser validation evidence

Chrome 151 completed one real 256×256 Turbo request with the `f16-qwen-vision-f32` bundle on the
workstation's hardware Vulkan WebGPU path. The scoped Chrome GPU process, framebuffer allocation,
and SM utilization provide the hardware attestation; the privacy-redacted adapter label alone does
not. This is positive browser inference evidence for the former non-retaining policy, but the
emitted report deliberately says `numerical_parity_claimed=false`: it was an ordinary prompt/seed
request, not an upstream tensor fixture replay. It does not qualify the new full resident preload
contract. The production *headless* Bevy swap-chain smoke remains a separate failure.

### historical streamed 256 full request

`headless=infer` runs the ordinary request resolver and the concrete `BrowserBooguFactory` without
creating a render surface. It accepts only Turbo generation and fails closed unless `prompt`,
`seed`, `width=256`, and `height=256` are supplied. The browser model backend is raw `CubeBackend`
rather than Burn's fused alias. Floating stages are adapted to F32, and every asynchronous semantic
source awaits the CubeCL `ComputeClient::sync()` future instead of blocking Wasm's event loop. It
now defaults to resident preload; the measurements below predate that default.

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
came from the former non-retaining policy: Qwen streamed once, the denoiser streamed separately for
all four DMD steps, and the VAE decoder streamed once. Production now rejects that hot-path traffic
contract by completing full GPU residency before **Ready**. To select comparable low-memory
behavior today, use the explicit `residency=layer-streamed-diagnostic` policy with a custom
`artifacts=` URL.

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
The historical report attested the pinned model revision, sealed bundle digest
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

### 1.5K VAE encoder calibration

The opt-in `headless=vae-reference` route isolates the exact F32 VAE encoder reference surface
without loading Qwen, the denoiser, DMD, or the decoder. It authenticates the pinned compact
47-tensor fixture (49,120,176-byte SafeTensors container), then performs three fresh verified
encoder loads on one real no-surface WebGPU device with the same exact input and epsilon. The
report includes all six F32-oracle and six upstream-BF16-drift metrics per repeat, run-to-run
bitwise stability deltas, exact encoder-object traffic, host/GPU resource telemetry, and always
sets `numerical_parity_claimed=false`.

The 2026-08-13 calibration used Chrome `151.0.7922.108` (revision `4744b886...`),
`BrowserWebGpu` with raw CubeCL and no fusion, and an NVIDIA RTX PRO 6000 Blackwell Workstation
Edition at device `0x2bb1` on driver `610.43.02`. This was a schema-v1 flat-bundle calibration with
artifact digest `5d7e25b1d9be1fdf4a6372bfb9db28cf62ef90253082cef22af09653047e3a7b`,
not a run of the current modular composition. Its VAE weights were stored as F16, loaded with
`adapt-to-f32`, and executed as F32. Three separate Chrome processes each made three fresh
authenticated encoder loads. All nine executions produced identical six-component metric tuples,
and the three executions within every process were bitwise identical. The three calibration-report
SHA-256 values were
`9728667927ee5cf5aab78f1f9e495b35afdad46a52f2418e27fd887401822c55`,
`6bba6a415bc5c360f73f0b917344edadb0b66ec17e986b7a4def12229fe7570f`, and
`e09a589d6dd43dd751310bfcf26a273263f47858a77cbdaba3d38b9ab6f806e6`.

The resulting `browser_webgpu_vae_f32_oracle_envelope` is deliberately scoped to that browser,
backend, release, storage/load policy, adapter, device, and driver:

| F32-oracle surface | Observed | Limit | Remaining headroom |
| --- | ---: | ---: | ---: |
| moments maximum | `0.014656067` | `0.016` | `9.17%` |
| moments RMSE | `0.0006903031` | `0.00075` | `8.65%` |
| moments cosine | `1.0` | `>= 0.999999` | `0.000001` absolute |
| mean maximum | `0.011677384` | `0.013` | `11.33%` |
| log-variance maximum | `0.014656067` | `0.016` | `9.17%` |
| standard-deviation maximum | `0.00006417744` | `0.0001` | `55.82%` |
| raw-latent maximum | `0.011677384` | `0.013` | `11.33%` |
| scaled-latent maximum | `0.0042167306` | `0.005` | `18.58%` |
| scaled-latent RMSE | `0.0001381072` | `0.0002` | `44.82%` |
| scaled-latent cosine | `1.0` | `>= 0.999999` | `0.000001` absolute |

The fixture identities were authenticated independently. The compact metadata/tensors/source/output
SHA-256 values are `4a3847347adefd38f5978844f311b934606f9b8a6be0013235dd1fcaf5393ebb`,
`bdd429af5b8f146fea3ac05238cd1d711d3be7f974dc54544ae85c149874a2df`,
`96534b93904478caf92c1d0e1b431396f81e7b62f09bb5505443378f245d9647`, and
`f6d8e1b45351bfe203136da075b43afaf6f80c9eda481f529bc6707eb91787bc`. The exhaustive equivalents
are `1e78233c703ed32ee351c25d54ca4b05e3efeb898ee2836d1cc96c522e2abcae`,
`2585ddf2337e41f884218a4abeceb8a10baa7553e43d37f33016be68edc3eeb9`,
`96534b93904478caf92c1d0e1b431396f81e7b62f09bb5505443378f245d9647`, and
`8e88d6c3580593da723049ef4027a60c5d730b6006ef766d49971a23c6446a70`.

A native WGPU current-bundle control kept the stricter native component limits unchanged. Its
moments maximum/RMSE, mean maximum, log-variance maximum, standard-deviation maximum, raw-latent
maximum, and scaled-latent maximum/RMSE were respectively `0.001693964/0.000046921377`,
`0.001693964`, `0.0011615753`, `0.0000025853515`, `0.001693964`, and
`0.00061166286/0.000012481793`, with cosine `1.0` for moments and scaled latent. The required-run
report SHA-256 was `d5adca3c9f4197805e2885430177b1509389f409e0baea18afc588919defb8a0`.
This preserves the native gates; the wider component maxima and moments RMSE apply only to the
serialized browser envelope. These measurements establish deterministic behavior on the attested
stack, not a cross-adapter WebGPU portability claim. The encoder diagnostic itself always retains
`numerical_parity_claimed=false`; only the subsequent exhaustive full-chain gate below may make the
scoped numerical claim.

After building `crates/bevy_image/www/out`, run it with:

```sh
BURN_IMAGE_BROWSER_1K5_VAE_REFERENCE=1 \
  node crates/bevy_image/tests/wasm_browser_1k5_parity.mjs
```

The default fixture directory is `/tmp/boogu-1k5-bf16-1536`; override it with
`BURN_IMAGE_BROWSER_1K5_PARITY_FIXTURE_DIR`. Full parity continues to require
`BURN_IMAGE_BROWSER_1K5_PARITY=1` and the exhaustive fixture, and the two selectors are mutually
exclusive.

### Historical flat-bundle 1.5K exhaustive browser parity

The current canonical schema-v2 modular parent has digest
`4eb95001708becebeab5bb7417b02003e9dbe704775bb49557b681a5b617fd5a` and sealed `qwen` and `vae`
dependencies. The harness now resolves and validates all three sibling bundles independently, but
that modular closure has not yet had a fresh real-GPU replay. Its browser requalification is
pending; the result below must not be presented as evidence for the modular closure.

The 2026-08-13 schema-v1 flat full-chain replay passed with top-level `ok=true`, report schema 2,
`gates.passed=true`, `artifacts_verified=true`, `fixture_authenticated=true`, and
`numerical_parity_claimed=true`. It authenticated the historical flat
`boogu-image-0.1-edit-turbo-1k5` artifact digest
`5d7e25b1d9be1fdf4a6372bfb9db28cf62ef90253082cef22af09653047e3a7b` and all 372 tensors in the
11,258,528,368-byte exhaustive fixture. The exact stack was Chrome `151.0.7922.108` revision
`4744b886309d987d292e43232776d2206cccb13d`, raw CubeCL WebGPU on the NVIDIA RTX PRO 6000
Blackwell device `0x2bb1`, and driver `610.43.02`.

The 681.477-second qualification run retained all 48 verified denoiser stages after DMD step 0.
The first step fetched 20,585,288,320 model bytes; steps 1, 2, and 3 each fetched exactly zero model
bytes. The cache was proven empty before the exact two-width-slab VAE decoder. Peak scoped Chrome
GPU framebuffer usage was 56,723 MiB with 99% peak SM utilization. Peak Chrome-tree PSS was
6,297,416 KiB, peak Chrome GPU-process PSS was 2,110,982 KiB, both reported zero swap, and peak
Wasm linear memory was 2,501,640,192 bytes.

The propagated results passed without relaxed full-chain thresholds:

| Surface | Result | Gate |
| --- | ---: | ---: |
| final latent relative RMSE / cosine | `0.07558763` / `0.9971675` | `<=0.085` / `>=0.996` |
| decoded tensor relative RMSE / cosine | `0.07995286` / `0.99693114` | `<=0.09` / `>=0.996` |
| final RGB PSNR / mean block SSIM | `33.980225 dB` / `0.99232423` | `>=33.5 dB` / `>=0.99` |

Evidence is retained at
`/tmp/burn-image-browser-1k5-qualified-final-20260813/`. The report SHA-256 is
`93ba7d3861def7ce194ee9884e65fb1e9730b60541aa195cbba6b2003e5679e3`; the log and page-capture
SHA-256 values are `9f5d823de583d1abf8f33b1ca1c00294cbba6a0ee805374327b752be08da68a8` and
`f69063bc072a11d25ce4d5c90157f676002b47f1c6730d2d722c4b200d1af153`. A separate exact-stack
attestation binds that report to the observed Chrome, adapter, device, and driver; its SHA-256 is
`71be8d80f0281785b2a9774b09506bf64bbb667d3aa1f835a8bd24cc75f31282`.

This is historical exact 1536×1536 no-surface numerical evidence for that schema-v1 flat closure
and stack. It does not qualify the current schema-v2 modular composition, establish a browser
performance target, rendered-window support, another official 1.5K aspect ratio, or
cross-browser/adapter/driver portability. The ordinary web UI therefore continues to reject the
1.5K variant until those distinct contracts are separately validated.

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

### limits and remaining runtime gates

Transport remains bounded by a 4 MiB default chunk, 16 MiB hard response cap, 4 MiB manifest cap,
and 256 MiB hard semantic-object cap. That object cap is not a WGPU binding limit. The largest
current post-load tensor is a 25,323 by 4,096 embedding row adapted to F32, or 414,892,032 bytes.
Only the Boogu app requests 512 MiB minimum storage-buffer/buffer limits; the model-neutral app
keeps WebGPU's portable baseline. The full run reported applied limits of 2,147,483,644 bytes for
`max_storage_buffer_binding_size` and 4,294,967,292 bytes for `max_buffer_size`. Factory startup
fails closed below the exact model requirement. Production also reports a conservative full
resident weight-plus-activation plan before preload; API buffer limits alone are not treated as
available VRAM, so an allocation or device-loss failure still prevents **Ready**.

The opt-in exact 1536-square qualification route has a separate, stricter limit contract. An
ordinary F32 VAE decode would create a `[1, 256, 1536, 1536]` feature tensor requiring one
2,415,919,104-byte buffer, which exceeds the qualification Chrome device's 2,147,483,644-byte
storage-binding ceiling. The qualification route therefore uses an exact two-width-slab tail: it
preserves the global middle attention and lower-resolution decoder, supplies one-pixel convolution
halos, reduces GroupNorm statistics across both slabs, and stitches only after the first final
residual block reduces 256 channels to 128. Its conservative largest planned buffer is
1,215,832,064 bytes (`[1,256,1538,772]`, including the halo and explicit convolution padding), so
the same Chrome limit passes preflight. This policy is not enabled by the ordinary browser UI and
is qualified only through the exhaustive numerical fixture gate above.

Concrete native/browser factories implement only `ArtifactCachePolicy::UseCached`; `Refresh` and
`Bypass` fail closed. The historical flat-closure 1.5K no-surface fixture replay is the prior
browser numerical result; the current modular closure still needs a fresh replay. The
ordinary Turbo request and finite stage probes remain separate functional evidence. Rendered-window
support and synchronized browser performance still require their own positive gates.
