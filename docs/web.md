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
install -m 0644 crates/bevy_image/www/burn-image-icon.png \
  crates/bevy_image/www/out/burn-image-icon.png
rg '^export function (start_boogu_web|provide_reference_image)' \
  crates/bevy_image/www/out/bevy_burn_image.js
npx --yes serve crates/bevy_image/www --listen 8080
```

### bounded WGPU writes, truthful queue completion, and downstream builds

The workspace root pins and patches both `wgpu = 29.0.4` and `cubecl-wgpu = 0.10.0`. Patched `wgpu`
splits Wasm `Queue::write_buffer` and splittable `Queue::write_texture` payloads into JavaScript calls
of at most 2 MiB, below the large-write transport path that failed on the qualification Chrome/Dawn
stack. It also exposes a result-bearing submitted-work callback so a rejected
`GPUQueue.onSubmittedWorkDone()` promise is not converted into apparent success. Native/custom
backends retain their existing callback behavior.

Patched `cubecl-wgpu` submits pending upload-only work even when no compute task has yet been queued,
keeps queue completion and scoped validation/out-of-memory/internal errors asynchronous on Wasm,
and propagates either failure through `ComputeClient::sync()`. Semantic stage completion therefore
cannot be reported before its uploads/computation have passed a real queue boundary.

Cargo applies both patches when building the checkout and Pages package, but patches declared by a
dependency are not inherited by crates.io consumers. A downstream application must vendor
equivalent patched `wgpu` and `cubecl-wgpu` sources and select both from its own root manifest, for
example:

```toml
[patch.crates-io]
wgpu = { path = "vendor/wgpu-29.0.4" }
cubecl-wgpu = { path = "vendor/cubecl-wgpu-0.10.0" }
```

The paths must contain equivalent bounded-write, queue-result, upload-submit, and error-propagation
changes. This requirement remains until both fixes are available from upstream/resolvable releases;
WebGPU builds from the published project crates with either consumer-side patch absent are outside
the browser support claim.

Open a URL with the remote sealed bundle and release selection:

```text
http://localhost:8080/?variant=turbo&profile=production
```

The tracked static host lives in `crates/bevy_image/www/`; generated bindgen JavaScript and Wasm,
plus an exact copy of the tracked app icon, remain under the ignored `www/out/` directory.
`.github/workflows/deploy-pages.yml` mirrors the
repository's local package command and never copies model objects into the Pages artifact. Manual
dispatch runs the complete deployment; `main` pushes deploy automatically only after the
repository variable `BURN_IMAGE_PAGES_READY=true` is set. Before setting it, publish all five
canonical CDN entries and enable Pages with **GitHub Actions** as its source. The deploy job probes
all five manifests, every directly stored non-weight payload, and every sealed-layout physical part
as exact HTTP 200 complete objects with canonical `Content-Length`, absent-or-identity
`Content-Encoding`, cross-origin readability, and payload `immutable` caching. A manifest that is
not served with the recommended `no-cache` policy emits a deployment warning rather than bypassing
the mandatory sealed-digest and payload checks.
It authenticates the sidecar and its complete logical reconstruction contract
instead of requesting absent logical Burnpack URLs. A missing/private entry or incomplete CDN
header policy therefore blocks deployment instead of publishing a page that cannot load its model.
The expected project URL is
`https://mosure.github.io/burn_image/`.

As of 2026-08-16 the canonical model URLs are public. A real-browser transport probe authenticated
the Turbo/Qwen/VAE manifests and layouts, verified cold whole-part downloads, and resumed from warm
CacheStorage. Pages warns because the public CDN marks reusable manifest URLs `immutable` instead
of the recommended `no-cache`; sealed-digest and physical-payload failures remain blocking. This
transport probe is not a full generation or numerical-parity claim.

On Linux, the browser harnesses do not choose shared-memory backing from nominal free blocks alone.
They use BigInt `statfs` arithmetic and a real bounded 256 MiB write/`fsync`/delete probe to catch
effective user/group quotas. The current ordinary, same-engine, and 1.5K evidence selected
`/dev/shm` under policy `linux-dev-shm-statfs-and-quota-aware-probe-admitted` and omitted
`--disable-dev-shm-usage`: `/dev/shm` wrote all 268,435,456 probe bytes, while `/tmp` stopped with
quota exhaustion despite reporting free space. The exact `/tmp` writes before failure were
158,838,784 bytes for the ordinary run, 107,565,056 bytes for the canonical same-engine run,
159,440,896 bytes for browser 1.5K, and 93,999,104 bytes for the final-source first-DMD diagnostic.
Probe files are deleted after admission testing.

The HTML shell has no visible loading overlay: native and browser sessions use the same Bevy
controls, notices, and monotonic selected-model progress surface. That progress reports aggregate
authenticated bytes, logical objects, unique physical parts, bounded reads, smoothed throughput,
and ETA; stage-local Qwen/VAE/denoiser file counts are never presented as the complete denominator.
The runtime-ready boundary means that generation controls can accept a request; it does not imply
that every request-lazy stage is already cached. If a cold first Run must finish the selected
closure, the Bevy bar preserves and advances the same aggregate counters. It is one setup
lifecycle, not a second Qwen download phase. Ordinary browser inference keeps the Bevy cameras
active so status and controls remain responsive; `surface-gate=1` is reserved for rendered
qualification that measures suspended acquisition and exact camera restoration. Explicit
resident mode performs all model work during preload. Explicit low-VRAM mode is variant-aware: Edit retains one
runtime-Q8/F32 denoiser for four DMD passes, while Turbo initially preloads 46 packed-F16 stages,
widens one dense-F32 semantic stage at a time during DMD, and evicts the packed cache before VAE
decode. The ordinary persistent cache stores complete, independently authenticated physical parts
no larger than 20,971,520 bytes; it never stores a complete 256 MiB semantic object or bundle.

With `artifacts` omitted, that selection resolves exactly to
`https://aberration.technology/model/boogu-image-0.1-turbo`.
An explicit URL override remains available for local or pre-publication testing, for example
`artifacts=http%3A%2F%2Flocalhost%3A8080%2Fartifacts%2Fboogu-image-0.1-turbo`.

`variant` accepts `turbo`, `edit-turbo`, or `edit-turbo-1k5` (with `1k5` retained as an alias). On an
ordinary page without `artifacts=`, the accessible **Model release** selector changes this query
parameter and reloads the page. It preserves `profile` and other ordinary query state. An exact,
variant-specific residency selector is normalized to `low-vram`, allowing the target release to
choose its own bounded policy. The selector is disabled with an explanation when `artifacts=` pins
one exact custom bundle or when a no-surface diagnostic is active. This is startup selection, not a
live engine swap: only one WebGPU model runtime is constructed at a time. The
default `profile=production` selector maps to the precise sealed `f16-qwen-vision-f32` contract; the
old precise selector remains an alias for existing links and low-level provenance. The name means
that Qwen vision is stored F32 while the denoiser, Qwen text tower, and VAE are stored F16; it does
not describe an all-F32 model.

The interactive page defaults to `residency=resident`. After its mandatory shared-device VRAM
preflight, it loads and retains the selected dense-F32 request graph before enabling **Run**, so a
page session remains warm across requests. `residency=low-vram` is the explicit bounded-memory
alternative: it keeps the canonical artifact unchanged and streams Qwen and selected VAE objects.
Edit retains a request-scoped runtime-Q8/F32 denoiser across four DMD steps; ordinary Turbo uses the
packed-F16/dense-F32-per-semantic-stage policy described below. The new warm source default requires
refreshed browser hardware qualification; current evidence below remains scoped to low-VRAM.
`residency=layer-streamed-diagnostic` is the intentionally
host-heavy diagnostic that reloads denoiser stages on every step. Diagnostic storage profiles and
layer streaming require an explicit `artifacts=` URL and are not canonical CDN production paths.

Each ordinary UI startup can select one of the three releases and then exposes that release's same
shapes as native; the in-canvas **Loaded model** control is disabled for this truthful singleton
runtime. A separate no-surface gate now qualifies the current schema-v2 modular Edit-Turbo 1.5K
low-VRAM closure for exact 1536x1536 numerical parity and sub-32-decimal-GB memory on the pinned RTX
PRO 6000 Blackwell/Chrome 151 stack.
`qualification-f32` is an optional non-blocking control diagnostic, disabled by default in the
release workflow; its last run ended in device loss. The exact F32 fixture route is per-request
denoiser retention, not a substitute for the mandatory low-VRAM numerical and measured-memory gate.
The ordinary UI's all-stage `resident` mode is a separate pending run.
Descriptor support for the other released shapes does not establish rendered operation,
performance, or cross-stack portability.
WebGPU requires a secure context (`https://` or localhost). Canonical part-only artifacts require
cross-origin HTTP 200 reads with exact physical bytes; `Content-Encoding` must be absent or
`identity`. Exposed `Content-Length` is checked early when available, while exact `Blob.size` and
SHA-256 checks remain the authority. Legacy direct Burnpacks additionally require exact HTTP 206
range framing and exposed `Content-Range`.

## implemented contracts

- Bevy and Burn share the same WGPU adapter/device/queue through `bevy_burn`.
- An explicitly reported CPU adapter/device and the CPU tensor backend are rejected. WebGPU may
  privacy-redact both hardware adapters and software implementations as `Other`, so hardware
  execution claims require an external scoped GPU-process attestation like the one below.
- The UI exposes prompt/instruction, capability-safe mode, loaded-model identity, size, seed,
  reference selection, run/cancel,
  structured progress/errors, result display, and PNG download.
- BrowserWebGpu uses the same core descriptors as native WGPU. Turbo and 1K Edit accept dimensions
  divisible by 16 from 256 through 1024, bounded to 1,048,576 pixels, and default to 1024x1024.
  Edit-Turbo 1.5K defaults to 1536x1536 and exposes its ten official aspect-ratio presets.
- `BrowserBooguFactory` constructs real async Qwen, VAE, and denoiser stage sources on the shared
  WebGPU device; no placeholder or CPU tensor-backend fallback exists. Adapter metadata alone is
  not used to distinguish privacy-redacted hardware from a software WebGPU implementation.
- The ordinary `resident` production factory reports **Ready** only after it has materialized and
  synchronized every required Qwen, VAE, and denoiser stage as dense F32 WebGPU modules. A forward
  pass performs zero model-weight HTTP reads, verification/decoding, or host-to-device uploads.
- For Edit, the `low-vram` factory streams Qwen and selected VAE objects and runtime-quantizes
  selected inventory-qualified row-layout denoiser matrices to Q8S block-32/F32. The verified denoiser
  remains resident for four DMD steps and is cleared before VAE decode. Source dispatch preserves
  QFloat weights into `q_matmul`. The passed Edit real-browser gate
  reports the retained dtype/cache audit, but captured no backend kernel trace and therefore does
  not claim on-device quantized execution.
- For Turbo, `low-vram` reports backend
  `burn-webgpu/browser-low-vram-preloaded-packed-f16-dense-f32-per-stage-denoiser/request-scoped-packed-cache-evicted-before-vae/request-scoped-surface-acquire-suspended`
  and public selector `low-vram-preloaded-packed-f16-dense-f32-per-stage-denoiser`. Initial setup
  authenticates 46 stages / 106 objects / 912 F16 tensors and retains 19,870,010,624 padded
  packed-F16 bytes. Every DMD step widens one semantic stage at a time on device and executes
  dense-F32 matmul; runtime quantization and quantized execution are not applicable. Four steps
  require 184 stage materializations, 424 object unpacks, 79,480,042,496 packed bytes read, and
  158,960,084,992 F32 bytes written, with zero artifact/cache/network traffic.
- Turbo's packed cache is request-scoped. After DMD, the exact final F32 latent crosses a
  synchronized host handoff, all packed arenas and allocator state are cleared, an empty cache is
  proven, and the identical latent is reuploaded before VAE decode. A second request rehydrates all
  106 denoiser objects from the persistent range cache. Its complete request traffic is 186 objects
  / 35,106,151,424 bytes / 8,489 cache-hit ranges with zero network responses. The initial preload
  remains 106 objects / 19,870,166,528 bytes / 4,780 ranges; the first Generate request reads 80
  Qwen-text/VAE-decoder objects / 15,235,984,896 bytes / 3,709 ranges.
- Ordinary rendered inference disables both primary-window cameras before runtime submission,
  records zero `getCurrentTexture()` calls during the request-scoped gate, and restores the exact
  camera-active state after the terminal model event and before output-ready publication. On any
  packed-F16 error or cancellation, the runtime keeps that gate active while a centralized cleanup
  crosses checked queue barriers, clears denoiser, packed-cache, and RoPE roots, runs allocator
  cleanup, and verifies zero packed residency before queuing the terminal. The ordinary rendered
  Turbo 1024 run below completed this contract; failure/cancellation stability remains separately
  fail-closed.
- The retired `low-vram-retained-q8-dense-f32-per-stage-denoiser` selector fails closed. Its
  real-browser first-DMD output was non-finite; it is neither a compatibility fallback nor an
  alternate supported Turbo mode.
- Canonical physical parts and compact declared files use one HTTP 200 complete-object fetch and
  one CacheStorage entry. The reader checks any exposed exact `Content-Length`, verifies
  `Blob.size` before `Blob.arrayBuffer()` enters Wasm, and authenticates SHA-256. The legacy
  `fetch_browser_range` path retains exact HTTP 206/`Content-Range` framing and 4 MiB cache entries.
- `BrowserStageShardReader` authenticates the manifest-sealed transport layout, reconstructs one
  logical Burnpack from content-addressed physical parts targeting 20,971,520 bytes and hard-capped
  at 25,000,000 bytes, then verifies the logical object's exact size and SHA-256. It aggregates at
  most one sealed semantic object (hard cap 256 MiB), uploads/materializes it, and releases its host
  bytes. Wasm linear memory never holds a complete bundle even though initialized modules
  accumulate on GPU.
- Async loading suspends at network boundaries, reports artifact/stage progress to Bevy, and
  checks cancellation before every range and semantic boundary. Cache-hit part SHA-256 runs in
  browser-native Web Crypto so the Wasm main thread does not perform both physical and logical
  digest loops; the reconstructed logical object still receives its independent Rust SHA-256 gate.
  Ordinary resident forward execution uses the preloaded graph; low-VRAM execution reports exact
  variant-specific phase boundaries and weight traffic.
- VAE encode and decode apply independently selected objects with the released F32 force-upcast
  policy. The current source nevertheless initializes the complete 335,278,732-byte F32
  `AutoencoderKL` before applying either selection, so the browser resource plan charges the full
  module in both phases; bounded object transport is not a half-module residency claim.
- Canonical production storage remains `f16-qwen-vision-f32`. Explicit Q8/all-F16 storage bundles
  are diagnostics. Browser low-VRAM runtime quantization is current only for Edit; ordinary Turbo
  uses packed-F16 storage widened to dense F32 per semantic stage and inherits no qualification
  claim from historical Q8 evidence.

The implemented ordinary `resident` sequence is:

```text
fetch bounded ranges -> verify one object -> upload/materialize retained stage
-> release host payload -> next object -> synchronize resident graph -> Ready
```

The implemented low-VRAM Edit request sequence is:

```text
stream Qwen stages -> release Qwen
-> initialize full 335,278,732-byte F32 VAE -> stream/apply selected encoder objects -> release VAE
-> load and runtime-pack one verified denoiser stage at a time -> retain packed denoiser
-> reuse for all four DMD steps -> clear denoiser
-> initialize full F32 VAE -> stream/apply selected decoder objects -> output
```

The implemented low-VRAM Turbo sequence is:

```text
before Ready: fetch/cache bounded ranges -> authenticate and retain 46 packed-F16 denoiser stages
request: stream Qwen text -> widen/execute/release one dense-F32 semantic stage at a time
-> run four DMD steps with zero artifact I/O -> exact synchronized final-latent host handoff
-> clear packed cache/RoPE -> allocator cleanup -> prove empty cache -> reupload exact F32 latent
-> initialize full F32 VAE -> stream/apply decoder objects -> output
next request: rehydrate all 106 packed-F16 objects from the persistent cache before execution
```

The synchronous native directory factory is not compiled for Wasm. Browser sources enable
all-stage retention for ordinary `resident` production, request-scoped denoiser retention for
low-VRAM Edit, request-scoped packed-F16 retention for low-VRAM Turbo, and zero denoiser retention
only for the explicit diagnostic.
Resource planning, allocation, or synchronization failure is terminal; there is no CPU fallback.

Low-VRAM means strictly below decimal 32,000,000,000 bytes, not 32 GiB. Turbo's packed-F16 preload
peak is 22,304,263,424 bytes: 19,870,010,624 retained bytes plus 2,434,252,800 bytes of upload
workspace. Its conservative inference plan is 26,492,170,880 bytes: the retained cache plus the
1,753,654,656-byte maximum dense-F32 stage and a 4,868,505,600-byte activation reserve. The plan
explicitly does not derive a bound for the exact-size persistent Qwen text-layer pool, so the
aggregate measured GPU-memory gate remains mandatory. Edit's runtime-Q8/F32 plan is
30,402,341,120 bytes. These remain planning bounds. The current modular 1536x1536 Edit gate
independently measured 29,828 MiB / 31,276,924,928 bytes and passed, leaving 723,075,072 bytes below
the decimal cap; no other browser release/shape inherits that peak.

A separate native WGPU Turbo 1024x1024 low-VRAM output-qualification candidate now passes its
artifact, output, and strict memory checks. It measured 27,055 MiB / 28,369,223,680 bytes at the
VAE-decode/overall peak across 2,246 matched, nonzero PID-scoped samples; its canonical modular
closure verified 223 weight objects, 253 files, 1,940 tensors, and 38,224,723,494 bytes under
parent digest `555019af867a80bb4d7cec5dc2f0ba60ae799071994a5fd24d7e71918cb9ce36`. The 1,448,891-byte
PNG retained SHA-256 `b2cfbc50f7c8f9d486799abd8c5be90c8770059a1dbc020ad02ac41a91abfab1` across its low-VRAM-only,
math-neutral allocator change: direct released-F16 Qwen embedding upload plus exact-size VAE
transient allocation and pre-tail synchronization/cleanup. The browser policies are unchanged.
Report SHA-256 is
`4f67f468110addef18a4d6f27d4ed01ab57f1c3c03de7174e6450fe793d38376`. This native result does
not itself provide browser evidence or numerical cross-runtime parity; the distinct ordinary
rendered browser Turbo result is recorded below.

Output currently crosses the model boundary as a validated host image and is then uploaded into a
Bevy texture. A zero-readback, device-resident result handoff is not implemented.

## browser validation evidence

### Current packed-F16 Turbo 1024 evidence status

The current ordinary Browser Turbo route is the packed-F16/dense-F32-per-semantic-stage policy. Two
single-request 1024x1024 runs completed the full model lifecycle, zero-DMD-I/O audit, request-scoped
surface gate, PNG download, and measured aggregate GPU-memory gate on the pinned Chrome/Blackwell
stack, but with deliberately different claim scopes.

Serialized Run C forced only the Qwen block-0 synchronization boundary for localization. It passed
with report SHA-256 `b0dfcc8e53fd7ad1c4731d3169e2f43c50063aa2b54e5ca6347789e18630c6e6`, 750/750 matched GPU
intervals, 476 active, 99% peak SM activity, and peak 22,824 MiB / 23,932,698,624 bytes. It produced
the 1,452,562-byte PNG SHA-256
`5d7a947301c9cefef8bca5cd42db52b98fe3fca440e98d302312eff6afe2eb38`, but its serialized identity
is diagnostic-only: it is not release output-quality, numerical-parity, or performance evidence.

The subsequent ordinary, non-serialized UI request passed with report SHA-256
`36525be1d5ff482c409c3b7484027fcb335340e474e4a95f182720ea3f032a28`. It produced the same PNG,
recorded zero page/GPU errors and zero gated texture acquisitions, restored exact camera state, and
successfully acquired the surface after resume. All 748 GPU intervals matched, 461 were active,
peak SM activity was 99%, and peak use was again 23,932,698,624 bytes. Its 780.261-second duration
includes preload, cache population, compilation, inference, UI interaction, and save, so it is cold
qualification timing rather than a warm performance row. The same-seed comparison against native
passed at `37.517250061 dB` PSNR / `0.985732973` mean 8x8 block SSIM; quality-report SHA-256 is
`31da8e541013c38dd215257431159a99c7112ad79714a079e2a4b25f9c855103`. Because the runtimes
generated noise independently, that is final-output quality rather than exact-noise parity.

The canonical same-page/same-engine rerun passed both sequential ordinary model requests on one
engine with exactly one adapter request, one device request, and one Chrome GPU process.
Packed-cache preload attempts were `[1, 2]`. Request 2 read all 186 objects / 35,106,151,424 bytes /
8,489 ranges as cache hits, with zero misses and zero network requests or bytes. Both requests
completed four zero-I/O DMD steps, digest-preserving Qwen and DMD-to-VAE handoffs, cache cleanup
from ready to empty, and one surface-suspension window with zero gated acquisitions, failures,
violations, or overlap and a successful first acquisition after resume. Peak Chrome GPU-process
memory was 23,255 MiB / 24,384,634,880 bytes; page/GPU error lists were empty, the process group
exited, and the Chrome profile was removed. The distinct 1024x1024 downloads have SHA-256
`5d7a947301c9cefef8bca5cd42db52b98fe3fca440e98d302312eff6afe2eb38` and
`815c553a70a4322aa8e49a51aeb0d46b75ccf2178b435c9b0ba0fedec3da5e0c`; canonical report SHA-256
is `90da22207398ae907e6b0d0bc93881c689a2a7362a1e52aac5435deac525b5d5`. This is an ordinary
same-engine rendered smoke, not numerical parity. Exact-noise full-chain Turbo parity and
synchronized warm performance remain pending.

The final-source packed-F16 Turbo first-DMD diagnostic passed with outcome
`diagnostic-passed-no-full-parity-claim`. It binds JavaScript SHA-256
`64197f892ae850d901a9b76ff70dba7f543fa70af02028f605bb9eb126dc1b37`, WebAssembly SHA-256
`001f0bcc93fbaeea9a9b32d2adcb8b46f1897b80d36386919c03d69869dca86b`, probe/harness SHA-256
`d6485fa204233b25c1d12128410ae162a8d1ce59053179d3f31a3db63155dd88`, contract SHA-256
`dadd84a4ef9c5162c4aea7f3251cb40461e08d969bfaada3ae99f94dc6fb4b86`, report SHA-256
`0a600471ec9e3119eeaebd616e9dd29a84c62067881b6f9834949019f92d5eab`, and console SHA-256
`20af8aa43d3a53608c0658fabbef0fb8d7e85f2ff4b9655736fc74b959d489fe`. The cache remained exactly
46 stages / 106 objects / 912 tensors; the single prediction performed 46 stage materializations
and 106 object unpacks, read 19,870,010,624 packed bytes, wrote 39,740,021,248 dense-F32 bytes, and
made zero DMD artifact/range/cache/network traffic. Velocity relative RMSE / cosine were
`0.03869645` / `0.9992708`, and prediction relative RMSE / cosine were `0.042713966` /
`0.99910367`. `/dev/shm` passed quota-aware admission, and process-group exit, Chrome-profile
removal, and artifact-server teardown were clean. This is one-prediction diagnostic stability
evidence only: it makes no calibrated numerical-correctness, full-chain/full-resolution parity, or
fully on-device-quantized execution claim.

The predecessor core-Q8 first-DMD diagnostic remains useful historical evidence. On a non-fallback
NVIDIA Blackwell adapter it authenticated five selected tensors / 1,941,506 bytes, retained the old
46-stage 284-Q8 / 628-F32 inventory, made zero DMD artifact/range/cache/network reads, and produced
finite velocity/prediction metrics of `0.039951164` / `0.044186924` relative RMSE and `0.99920744` /
`0.9990256` cosine. The historical JavaScript/Wasm SHA-256 values were
`7a7004085a273ac2ab2ce7a3150e86cf5d9875d167bc9fcfca140a42d0bce69f` and
`6f6d8e1143bfd3f1c4461450e992193f52bfae438bf4505b62b19068864dc8e6`; report SHA-256 was
`62e14a0712811e088b33d74d047fa6370c5ae8191920cbbc87313b93fb3e68d0`. Its outcome was explicitly
`diagnostic-passed-no-full-parity-claim`, and it does not qualify the current packed-F16 route.

Current-source exact numerical qualification uses the modular three-entry closure and exposes the
required runtime-Q8 low-VRAM case alongside an optional F32 control diagnostic:

```sh
export BURN_IMAGE_BROWSER_1K5_PARITY=1
export BURN_IMAGE_BROWSER_1K5_PARITY_ARTIFACT_DIR=/absolute/path/to/model/boogu-image-0.1-edit-turbo-1k5
export BURN_IMAGE_BROWSER_1K5_PARITY_FIXTURE_DIR=/absolute/path/to/boogu-reference-edit-1k5

BURN_IMAGE_BROWSER_1K5_RESIDENCY=qualification-f32 \
  node crates/bevy_image/tests/wasm_browser_1k5_parity.mjs
BURN_IMAGE_BROWSER_1K5_RESIDENCY=low-vram \
  node crates/bevy_image/tests/wasm_browser_1k5_parity.mjs
```

`qualification-f32` serializes as
`browser-qualification-per-request-f32-denoiser-retained`; its denoiser residency is
`request-scoped-f32-policy-retained-through-four-dmd-steps`. It streams Qwen and selected VAE
objects for the fixture request and retains only the F32 denoiser through DMD. It is not ordinary
`residency=resident`, which eagerly preloads all required stages before UI readiness. The Node
harness accepts `high-vram` only as a legacy input alias and canonicalizes its report to
`qualification-f32`. In `.github/workflows/parity.yml`, this control runs only when the boolean
workflow-dispatch input `run_browser_f32_control_diagnostic` is true. It is `continue-on-error`, is
excluded from the final required-outcome aggregation, and keeps its evidence in the common upload.
The low-VRAM step remains unconditional and its outcome remains mandatory.

The low-VRAM harness additionally scopes `nvidia-smi` samples to Chrome GPU-process descendants,
requires positive GPU activity, and fails at or above 32,000,000,000 framebuffer bytes. It verifies
the runtime-Q8/F32 denoiser inventory, request lifecycle, and the published 24 dB PSNR / 0.90 SSIM
floor. The optional `qualification-f32` control retains the stricter 33.5 dB / 0.99 final-pixel
threshold when explicitly run, but its failure cannot waive or replace the low-VRAM gates. The
current modular low-VRAM outcome passes and is recorded below. The F32 control's last run ended in
device loss and no passing result is claimed; the separate ordinary resident-UI run is also pending.
Static memory plans are not substituted for the measured result.

### Current modular low-VRAM 1.5K qualification

The 2026-08-14 source-bound no-surface exact-fixture rerun completed with top-level `ok=true`,
`gates.passed=true`, `artifacts_verified=true`, `fixture_authenticated=true`, and
`numerical_parity_claimed=true`. It authenticated current schema-v2 parent digest
`4eb95001708becebeab5bb7417b02003e9dbe704775bb49557b681a5b617fd5a`, resolved the sealed Qwen
and VAE siblings, verified all 223 executed weight objects, authenticated all 372 fixture tensors,
and compared the exact 355 tensors exposed at public semantic boundaries.

This browser result is distinct from the current-source native 1.5K pass. Native stage readers use
the separately qualified schema-v1 flat parity root and passed at 30,310,137,856 bytes with report
SHA-256 `a013adfcd30b7e6b2323ecc3723b22396f9858d14dd3cdd4a0da2699e199abe3`; the browser result below
uses the canonical schema-v2 modular root. Neither root is an alias for the other.

The request retained all 48 verified denoiser stages through four DMD steps, then cleared them
before the exact striped VAE decoder. Its dtype audit matched the 942-tensor inventory exactly: 377
Q8S block-32/F32 tensors, 565 F32 tensors, and no unexpected dtype. Quantized linear source
dispatch preserves QFloat weights into `q_matmul`; the report deliberately keeps
`on_device_quantized_execution_claimed=false` because the run did not capture a backend kernel
trace.

Final latent passed at `0.07212386` relative RMSE / `0.997419` cosine, propagated decode passed at
`0.07496766` / `0.9972718`, and final RGB passed at `34.531677 dB` PSNR / `0.99253726` mean 8x8
block SSIM against the low-VRAM `>=24 dB` / `>=0.90` floor. Scoped Chrome GPU-process telemetry
matched all 443 sample intervals, 224 with positive activity, and peaked at 29,828 MiB /
31,276,924,928 bytes with 99% peak SM activity. That leaves 723,075,072 bytes below the strict
decimal cap. Peak Wasm linear memory was 2,009,137,152 bytes and end-to-end qualification time was
904.960 seconds. The exact stack was Chrome
`151.0.7922.108` revision `@4744b886309d987d292e43232776d2206cccb13d`, raw CubeCL BrowserWebGPU
on NVIDIA RTX PRO 6000 Blackwell device `0x2bb1`, driver `610.43.02`.

The report SHA-256 is
`c895ae2c1cba3823afe756035b6e564d5ef27caf3722f5f350c07e23086e3b54`; its 7,373,531-byte page
capture has SHA-256 `0e273bf6b0660cdf6f96bbf163f56f0712b446c15e31ba6a95daac9a348c97b7`, and the browser log has
SHA-256 `c01579e2b2b01398f25828892a9e537e8e1b6fb8b664dfda8b4e49acee6d0f61`. The report binds
JavaScript SHA-256 `64197f892ae850d901a9b76ff70dba7f543fa70af02028f605bb9eb126dc1b37`, Wasm SHA-256
`c7edc0c0aaa6cfedfeddcd8b73a3a8ba2276d54741908dae871c8eab635bb99a`, browser-runtime source
SHA-256 `598b39e3fbcead0ef8aa875ec1aeede8accde5a863e2f6606124ec38daf0710a`, harness SHA-256
`cee29e844c33325a2dac1e29b3a03f731f61be2b926ade93a1a50f5443b8efd8`, and contract SHA-256
`d6a0ff5b8ebe8890be831efd1909cd36e2ede9709dc119fe1d965d4b8aa414ea`.

The first current-source run had already passed the same inner artifact, fixture, numerical, dtype,
and sub-cap memory gates, but its immutable outer report was `ok=false` solely because the stale host-key
contract expected retired field `audited_max_streamed_stage_bytes` while the runtime emitted
canonical field `audited_max_streamed_qwen_stage_f32_bytes`. Corrected offline replay has zero failures but remains
noncanonical; its report SHA-256 is
`3dec48ec032c7abd1ffcb9aab1546d81765e97503340f8ea895724dfe1aacd5b`. Only the subsequent
source-bound report above is promoted. It qualifies current modular low-VRAM numerical correctness
and measured memory only for this exact 1536x1536 calibrated stack. It does not qualify rendered-
window behavior, another released shape, synchronized browser performance, the explicit dense-F32
resident mode, the optional `qualification-f32` control, cross-stack portability, or canonical CDN
availability.

Chrome 151 completed one real 256×256 Turbo request with the `f16-qwen-vision-f32` bundle on the
workstation's hardware Vulkan WebGPU path. The scoped Chrome GPU process, framebuffer allocation,
and SM utilization provide the hardware attestation; the privacy-redacted adapter label alone does
not. This is positive historical browser inference evidence for the former non-retaining policy,
but the emitted report deliberately says `numerical_parity_claimed=false`: it was an ordinary
prompt/seed request, not an upstream tensor fixture replay. It does not qualify the current
ordinary resident, optional `qualification-f32` control, low-VRAM, full-resolution, or
rendered-window contracts.
The current low-VRAM 1536x1536 qualification above comes from the later exact modular replay;
ordinary resident and other Edit shapes remain pending, while rendered Turbo 1024 has the distinct
ordinary output/surface/memory evidence above and the F32 control lacks a passing result but is not
a release gate.

### historical streamed 256 full request

`headless=infer` runs the ordinary request resolver and the concrete `BrowserBooguFactory` without
creating a render surface. It accepts Turbo generation across the released 256-1024 range and
fails closed unless `prompt`, `seed`, `width`, and `height` are supplied. Edit inference remains in
the ordinary UI because it requires a reference image. The browser model backend is raw
`CubeBackend` rather than Burn's fused alias. Floating stages are adapted to F32, and every
asynchronous semantic source awaits the CubeCL `ComputeClient::sync()` future instead of blocking
Wasm's event loop. Headless diagnostics still default to low-VRAM unless their route fixes another
policy explicitly; `residency=resident` selects dense-F32 preload. The measurements below are
specifically the historical 256-square streamed run.

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
all four DMD steps, and the VAE decoder streamed once. Current low-VRAM production rejects that DMD
hot-path traffic: Edit retains one runtime-Q8/F32 denoiser across all four steps, while ordinary
Turbo preloads a request-scoped packed-F16 cache and widens stages without artifact reads. To
reproduce the former per-step reload behavior, use explicit `residency=layer-streamed-diagnostic`
with a custom `artifacts=` URL.

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

Browser Qwen/VAE stages and floating activations therefore remain F32 even when the sealed
transport profile stores F16 weights. The interactive default `resident` materializes and retains
dense-F32 denoiser weights. Explicit low-VRAM Edit runtime-packs inventory-qualified denoiser
matrices to Q8S block-32/F32; low-VRAM Turbo retains packed F16 and widens one dense-F32 semantic
stage at a time. Earlier
fused-shader, blocking-mapper, blocking-barrier, and Turbo Q8 failures are historical implementation
evidence, not current packed-F16 runtime success.

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
dependencies. Its current low-VRAM result is recorded above. The result below predates that modular
closure and is retained only as historical schema-v1 evidence; it must not be presented as the
current modular qualification.

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
cross-browser/adapter/driver portability. A web UI started with the 1.5K release exposes all its
official shapes. The current modular low-VRAM 1536x1536 evidence above supersedes the historical
run for that exact policy/scope; other shapes and rendered behavior remain unqualified until their
distinct current gates pass.

### Historical shared-device window failure

An earlier production smoke created the Bevy window and selected `BrowserWebGpu`, but that build
then failed Chrome's swap-chain shared-image allocation:

```text
Could not find SharedImageBackingFactory ... WebgpuSwapChainTexture
Caught DeviceLost error: Destroyed Device was destroyed.
Quitting the application due to DeviceLost RenderError
```

That historical failure reproduced with the then-current app-only build (no `bevy_burn`, Burn
bridge, or Boogu runtime), and with Bevy's `WgpuSettingsPriority::WebGPU`. The captured Chrome log is
21,091 bytes with SHA-256
`711a81abd2d8a13f04fece38842b1e50174a7b4150b414023c58c30b679f7be5`. It is preserved as a failed
window/surface smoke, not a failure of the successful surface-free model request above and not a
claim about the current source. The later ordinary rendered 1024 Turbo run passed its scoped
model/output/surface/memory gate as recorded above; that does not retroactively change this
historical failure or qualify rendered 1.5K behavior. A Chrome headless no-surface attempt that
selected SwiftShader remains excluded from hardware evidence.

### limits and remaining runtime gates

The artifact layers have separate bounds: logical Burnpacks have a 256 MiB semantic cap; physical
CDN parts target 20,971,520 bytes and have an exact decimal 25,000,000-byte hard cap; each part is
one complete authenticated CacheStorage entry, while legacy network ranges remain at most 4 MiB;
and the manifest plus sealed transport sidecar each have a 4 MiB bootstrap cap. The
semantic-object cap is not a WGPU binding limit. The largest
post-load Qwen tensor is a 25,323 by 4,096 embedding row adapted to F32, or 414,892,032 bytes. The
Boogu app requests a shape-aware 1,217,126,400-byte minimum storage-buffer/buffer limit, covering
the largest released 1.5K striped-tail plan; the model-neutral app keeps WebGPU's portable
baseline. The historical full run reported applied limits of 2,147,483,644 bytes for
`max_storage_buffer_binding_size` and 4,294,967,292 bytes for `max_buffer_size`. Factory startup
fails closed below the selected shape's exact model requirement. Resource plans do not infer
available VRAM from API buffer limits. The ordinary rendered page therefore performs a second,
capacity-oriented admission before requesting any model-weight transport part: after authenticating
only the bounded manifests, transport layouts, configs, and tokenizer, it retains 256 MiB-or-smaller
storage reservations totaling the selected policy's conservative device-byte plan on the exact
shared Bevy/Burn device, clears every reservation to defeat lazy allocation, and waits for queue plus
validation/internal/out-of-memory scopes. Success releases every reservation and starts the verified
weight download; failure or a 45-second timeout leaves the model unloaded. This is a fail-fast
allocation test, not a free-VRAM query or durable hardware guarantee: WebGPU exposes no portable
free-memory counter, and memory pressure can change after admission.

An ordinary strict-F32 1536-square decode would create a `[1, 256, 1536, 1536]` feature tensor
requiring one 2,415,919,104-byte buffer, which exceeds the qualification Chrome device's
2,147,483,644-byte storage-binding ceiling. The browser therefore uses an exact two-width-slab tail
for every released shape with either side at least 1024. It preserves the global middle attention
and lower-resolution decoder, supplies one-pixel convolution halos, reduces GroupNorm statistics
across both slabs, and stitches only after the first final residual block reduces 256 channels to
128. The conservative 1536-square buffer is
1,215,832,064 bytes (`[1,256,1538,772]`, including the halo and explicit convolution padding); the
largest plan across all official shapes is 1,217,126,400 bytes at 1392x1696. Shapes below 1024 use
the ordinary full strict-F32 decode.

The exact 1536-square qualification route additionally narrows its descriptor to the authenticated
fixture shape and applies the same 1,215,832,064-byte striped-tail preflight. The math is shared with
the ordinary shape-aware browser path, but only the current modular low-VRAM exact fixture run above
is qualified.

Concrete native/browser factories implement only `ArtifactCachePolicy::UseCached`; `Refresh` and
`Bypass` fail closed. The current modular low-VRAM 1.5K no-surface fixture replay is the qualified
1536x1536 browser numerical and sub-32-decimal-GB result. The optional, non-blocking
`qualification-f32` control has no passing result after its last device-loss run and does not weaken
that required low-VRAM gate; the ordinary all-stage-resident UI run is separate. The streamed 256
Turbo request and core-Q8 finite stage probe remain historical functional evidence. Serialized Run
C is a passing diagnostic-only packed-F16 Turbo 1024 result, while the subsequent ordinary run is
the passing output/surface/memory result and the canonical same-engine two-request ordinary smoke
also passes. Exact-noise Turbo parity, other released shapes, synchronized browser performance, and
cross-stack claims remain pending; each requires its own positive real-GPU gate.
