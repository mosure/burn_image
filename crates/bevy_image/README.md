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
instruction, a model selector that derives Generate or Edit behavior from the selected model,
capability-aware
size presets, a numeric `u64` seed, reference-image input, Run, Cancel, Save
PNG, and an edit-only **Use output as reference** action. Loading a reference selects edit mode and an
edit-capable model when available; it does not enter Edit mode when the initialized runtime cannot
execute edits. Reusing an output is always explicit and never replaces the reference after a Generate
request. The canonical native viewer exposes Generate, Edit 1K, and Edit 1.5K in a real model
drop-down. It initializes only the selected release, then unloads that release and lazily loads the
new selection on the next **Run**, keeping one model resident instead of tripling startup traffic or
VRAM. Singleton embedders and the browser runtime keep the Model control disabled rather than
implying a hot swap they do not implement.
NativeWgpu and BrowserWebGpu use the same
Boogu descriptors: Turbo and 1K Edit accept 256-1024 output dimensions bounded
to 1,048,576 pixels and default to 1024x1024; Edit-Turbo 1.5K defaults to
1536x1536 and exposes all ten official aspect-ratio presets. Ordinary rendered browser Turbo
1024 now passes its real-model/output/surface/memory gate. The first current-source modular 1.5K
browser computation passed its inner numerical and memory gates but failed only a stale host-key
contract; the corrected source-bound canonical rerun now passes the exact no-surface 1536x1536
numerical/memory gate.

On native builds, **Reference** opens the operating-system file dialog; a PNG,
JPEG, or WebP can also be dropped on the window. In the browser, **Reference**
opens the browser file picker, and the supplied web host accepts the same files
by drag and drop. Both paths enforce the same byte bound and use the same image
decode messages. **Save PNG** writes `burn-image-<job>.png` in the native
current directory or starts a browser Blob download.

The latest reference or generated image is fitted automatically. Drag with the
left or middle mouse button to pan, use the wheel or a touchpad pinch to zoom
toward the pointer, and use those manual controls to choose the desired framing. Camera input is
limited to the image viewport and is disabled while a text field or the panel
owns input. The panel moves below the viewport on narrow windows and remains
scrollable when vertical space is limited.

Progress remains visible throughout shared-GPU setup, manifest and shard
transfer, model-stage initialization, inference steps, output preparation,
cancellation, and failures. The headline, detail text, and progress bar report
the active phase instead of treating device creation or a downloaded manifest
as model readiness. Lazy native model switches forward their setup milestones and bundle-local
download file counts into the active job instead of leaving a static model-switch label. A missing
model runtime is shown explicitly; pressing
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
implemented `low-vram` policy instead streams one verified Qwen stage at a time and only the VAE
half required by the current request phase while retaining the unchanged mixed-F16 denoiser across
all four DMD steps and later requests. Its static plans are 30,585,112,576 bytes for Turbo and
30,971,005,440 bytes for either Edit release, including a 10,000,000,000-byte non-weight reserve.
The cap is decimal 32,000,000,000 bytes, not 32 GiB, and these plans are conservative planning
bounds rather than peak evidence. A cold native Turbo 1024x1024 run with a fresh XDG/autotune
cache passed its separate output-qualification gate: all 2,246 PID-scoped framebuffer samples
matched and were nonzero, and VAE decode set the 27,055 MiB / 28,369,223,680-byte overall peak.
Initialization, Qwen, and DMD peaked at 24,814,551,040, 27,087,863,808, and 27,096,252,416 bytes.
The canonical modular closure verified 223 weight objects, 253 files, 1,940 tensors, and
38,224,723,494 bytes under parent digest
`555019af867a80bb4d7cec5dc2f0ba60ae799071994a5fd24d7e71918cb9ce36`; the 1,448,891-byte PNG
retained SHA-256 `b2cfbc50f7c8f9d486799abd8c5be90c8770059a1dbc020ad02ac41a91abfab1` across the allocator
change. Total cold-process/inference time was `281.404/212.459 s`, and all six autotune-load
messages reported zero cached entries. Report SHA-256 is
`4f67f468110addef18a4d6f27d4ed01ab57f1c3c03de7174e6450fe793d38376`.

That native Turbo report is an output-qualification candidate, not fixture-backed numerical or
cross-runtime parity, a browser result, or a warm benchmark. Its low-VRAM-only allocation policy
uploads the Qwen embedding directly at its released F16 dtype, then uses exact-size VAE transient
allocation and synchronization/cleanup before the decoder tail. These choices are math-neutral
and do not change the high-VRAM or browser policies. The current-source 1536x1536 Edit-Turbo 1.5K
replay separately passed the strict real-GPU telemetry/numerical gate at 28,906 MiB /
30,310,137,856 sampled bytes, with 608 attempts and 607 matched/nonzero samples. Report SHA-256 is
`a013adfcd30b7e6b2323ecc3723b22396f9858d14dd3cdd4a0da2699e199abe3`. That result uses qualified
legacy schema-v1 flat artifact digest `4e8b12ac5ca95272f9009080a23baf1bc52d1b0e7aebf2e9e5f394a492369213`, not the canonical
schema-v2 modular closure;
1K Edit remains unmeasured in low-VRAM mode. The explicit `diagnostic-layer-streamed` policy is
local-bundle-only and rereads Qwen and VAE per request plus denoiser stages on each DMD step. It is
a memory diagnostic, not a supported production path. Neither policy substitutes a mock result or
CPU tensor backend.

The canonical five-entry CDN deployment is live. A 2026-08-16 real-browser probe authenticated the
Turbo/Qwen/VAE manifests and layouts, then verified both cold whole-part downloads and a warm
CacheStorage resume. The deployment gate still requires reusable `manifest.json` URLs to use
`no-cache`; the current public manifest response is incorrectly marked `immutable`.

Linux evidence harnesses combine BigInt `statfs` arithmetic with a real bounded 256 MiB
write/`fsync`/delete quota probe. Current runs admitted `/dev/shm`, rejected quota-limited `/tmp`,
and omitted `--disable-dev-shm-usage`; nominal free blocks alone are not treated as usable capacity.

Public viewer invocations should use `--profile production`. That selector maps to the precise
sealed manifest/provenance identity `f16-qwen-vision-f32`: Qwen vision is stored F32 while the
denoiser, Qwen text tower, and VAE are stored F16. The precise name remains accepted for backward
compatibility and low-level tooling; it does not mean the complete model executes F32.

An embedder built with the optional `native-autotune` feature and selecting the qualified native
mixed-F16 execution kernels in either high- or low-VRAM mode must call
`burn_boogu::configure_native_full_autotune()` on the main thread before Bevy creates or imports its
WGPU device. For Turbo and 1K Edit this selects padded-blackbox `p4/kv1/q1` denoiser q8192 with
strict-F32 RMSNorm, preserved-F16 VAE q4096, and safe F32 GroupNorm accumulation; Edit-Turbo 1.5K
uses its separately qualified q16384/VAE-q4096 bounds. Full-autotune qualification and 1.5K retain
q128 Qwen. Interactive balanced high-VRAM 1K uses q1024 with deferred terminal-stage
synchronization; low-VRAM streams q128 Qwen with a per-stage boundary.
`NativeBooguFactory` fails closed if a feature-enabled selected autotune policy was omitted or
selected too late. Ordinary `boogu-native` builds instead use static kernels and perform no tuning
or pre-Ready warmup. Add `native-autotune` and pass `--autotune balanced` to opt into shape-bucketed
tuning; balanced resident Edit 1K then performs one best-effort kernel warmup before reporting
**Ready**. On that measured feature-enabled warm path, an ordinary
1024x1024 Edit request completed in 10.848 model seconds / 10.883 host seconds (Qwen 0.654 s, VAE
encode 0.164 s, four-step DMD 9.743 s, VAE decode 0.227 s). Pass `--autotune full` in the same
feature-enabled build for the qualified fixed-shape policy. Browser and diagnostic
storage/residency paths retain their independent policies.

Run a converted Turbo bundle with:

```sh
cargo run -p bevy_burn_image --release --features boogu-native \
  --bin burn-image-viewer -- \
  --variant turbo \
  --profile production \
  --residency native-high-vram
```

Add `--artifacts .artifacts/boogu-image-0.1-turbo` to bypass downloading and
use an already verified local conversion. Replace `native-high-vram` with `low-vram` to select the
phase-resident policy. Its static sub-32-decimal-GB plan remains distinct from the measured native
Turbo output-qualification candidate above.

For unattended native execution, pass the request directly instead of automating the UI:

```sh
target/release/burn-image-viewer \
  --variant turbo \
  --prompt "a glossy blue ceramic bird" \
  --width 1024 --height 1024 --seed 0 \
  --output result.png \
  --report result.json \
  --timeout-seconds 1800
```

`--output` selects one-shot automation, hides the window by default, submits through the ordinary
`SubmitImageJob` path after runtime readiness, saves through the production PNG encoder, writes
load/request/stage timings and provenance, then returns success or failure as the process exit code.
Edit mode additionally requires `--source <IMAGE>`; Turbo rejects one. `--show-window` is an
optional visual-debug aid and never changes request routing.

Interactive canonical native runs default to `--variant turbo`; `--variant edit-turbo` and
`--variant edit-turbo-1k5` choose the initial selection. The Model drop-down can change among all
three without restarting: the selected release is loaded on the next **Run**. The switch worker
first joins the previous release's inference worker so every retained module is dropped, then
synchronizes and cleans the inference-device allocator, and only then constructs the new release.
This prevents old and new model residency from overlapping. Model loading, switching, and inference
run on dedicated native worker threads; submitting and polling from Bevy stay nonblocking. The
interactive multi-model viewer also uses a separate native compute device/queue from Bevy's render
queue so long inference dispatches do not serialize swap-chain presentation. Automated,
qualification, custom-artifact, and browser runs remain deliberately single-release. The native and
browser viewers accept the distinct `edit-turbo-1k5` release with a
`boogu-image-0.1-edit-turbo-1k5` bundle. Its omitted-size default is 1536x1536; the
official presets are 1536x1536, 1264x1856, 1856x1264, 1344x1744, 1744x1344, 1392x1696,
1696x1392, 1152x2032, 2032x1152, and 2368x992, bounded to 2,360,832 pixels. The viewer
accepts only public `production` / sealed `f16-qwen-vision-f32` for 1.5K; other profiles fail before
artifact loading. Turbo and 1K Edit retain their four-profile selection surface. Loading and
inference failures remain visible in the UI. The 1.5K release accepts high-VRAM or low-VRAM
production residency and rejects diagnostic per-DMD-step layer streaming. The 1536x1536 default is
the checkpoint-gated and benchmarked preset; the other official presets are exposed as bounded
model configurations but do not inherit that shape-specific evidence.

The `BooguRuntimeFactory`/`BooguRuntime` injection boundary remains public for embedders.
`BrowserBooguFactory` is the concrete Wasm implementation: it uses bounded HTTP ranges plus
digest-verified async Qwen/VAE/denoiser sources. Browser production defaults to the low-VRAM policy
described below. Explicit `residency=resident` selects the advanced
`browser-high-vram-resident-dense-f32` policy: before **Ready**, it sequentially verifies and
uploads one bounded semantic object at a time, releases each host payload, and retains every
initialized dense-F32 WebGPU stage. Forward execution therefore performs no repeated model-weight
download, hash/decode, or host-to-device transfer. This needs workstation-class WebGPU memory and
fails before **Ready** if its resource plan, allocation, or device synchronization fails; it never
falls back to CPU. This explicit dense-F32 mode is not currently qualified.

Repository and Pages builds inherit both browser patches from the workspace root. Patched `wgpu`
29.0.4 bounds `writeBuffer`/`writeTexture` calls to 2 MiB and exposes rejected queue-completion
promises; patched `cubecl-wgpu` 0.10.0 submits pending upload-only work and propagates queue and
error-scope failures through its asynchronous synchronization future. Cargo does not propagate root
patches through crates.io, so an external application enabling `boogu-web` must vendor/apply
equivalent `wgpu` **and** `cubecl-wgpu` patches in its own workspace root until both fixes are
upstream/resolvable. The browser qualification statements do not cover a graph missing either
patch.

The default `residency=low-vram` path is variant-aware while keeping the same canonical production
artifact. Edit streams Qwen and selected VAE objects, runtime-quantizes inventory-qualified
denoiser matrices to Q8S block-32/F32, retains that request-scoped denoiser through four DMD steps,
and clears it before decode. Runtime Q8 is current only for Edit. Ordinary Turbo instead uses
`low-vram-preloaded-packed-f16-dense-f32-per-stage-denoiser`: its initial setup authenticates 46
stages / 106 objects / 912 F16 tensors and retains 19,870,010,624 padded packed-F16 bytes. Each of
the four DMD steps widens one semantic stage at a time on the device and executes dense-F32 matmul;
the exact request counters are 184 stage materializations, 424 object unpacks, 79,480,042,496 packed
bytes read, and 158,960,084,992 F32 bytes written. DMD permits zero artifact/cache/network traffic.

Turbo's packed cache is request-scoped. Initial preload completes before **Ready**, but after the
fourth DMD step the runtime transfers the exact final F32 latent, synchronizes, removes all packed
arenas and allocator state, proves an empty cache, and only then starts VAE decode. The next request
rehydrates all 106 objects from the integrity-checked persistent range cache. The first Generate
request reads Qwen text and VAE decoder objects only, exactly 80 / 15,235,984,896 bytes / 3,709
ranges; a second request reads 186 objects / 35,106,151,424 bytes / 8,489 cache-hit ranges and
requires zero network responses. The packed-F16 plan records a 22,304,263,424-byte preload peak and
a 26,492,170,880-byte conservative inference bound, but the exact-size persistent Qwen text-layer
pool still requires the measured aggregate GPU-memory gate. Browser VAE transport applies only
selected objects while the current source initializes the full 335,278,732-byte F32 autoencoder.
This is storage compression followed by dense-F32 execution, not quantized execution.

Ordinary rendered requests also use the request-scoped surface gate: both primary-window cameras
are inactive before runtime submission, texture acquisition remains suspended through the terminal
model event, and their exact active states are restored before output-ready publication. Current
serialized Run C and the subsequent ordinary run both completed this gate, produced the same
1,452,562-byte 1024x1024 PNG (SHA-256
`5d7a947301c9cefef8bca5cd42db52b98fe3fca440e98d302312eff6afe2eb38`), and measured a
22,824 MiB / 23,932,698,624-byte Chrome GPU-process peak with no page/GPU errors. Run C is explicitly
diagnostic-only; its report SHA-256 is
`b0dfcc8e53fd7ad1c4731d3169e2f43c50063aa2b54e5ca6347789e18630c6e6`. The ordinary release/output
smoke report SHA-256 is `36525be1d5ff482c409c3b7484027fcb335340e474e4a95f182720ea3f032a28`.
Its same-seed comparison against the native PNG passed at `37.517250061 dB` PSNR / `0.985732973` mean
8x8 block SSIM; quality-report SHA-256 is
`31da8e541013c38dd215257431159a99c7112ad79714a079e2a4b25f9c855103`, and it is output-quality,
not exact-noise numerical parity.

The canonical same-page/same-engine qualification passed job 1 and job 2 on one engine with exactly
one adapter request, one device request, and one Chrome GPU process. Packed-cache preload attempts
were `[1, 2]`. Request 2 rehydrated the denoiser and read all 186 objects / 35,106,151,424 bytes /
8,489 ranges as cache hits, with zero misses and zero network requests or bytes. Both requests
completed four zero-I/O DMD steps, digest-preserving Qwen and DMD-to-VAE handoffs, cache cleanup
from ready to empty, and one surface-suspension window with zero gated acquisitions, failures,
violations, or overlap and a successful first acquisition after resume. Peak Chrome GPU-process
memory was 24,384,634,880 bytes; page/GPU error lists were empty, the process group exited, and the
Chrome profile was removed. The distinct 1024x1024 downloads have SHA-256
`5d7a947301c9cefef8bca5cd42db52b98fe3fca440e98d302312eff6afe2eb38` and
`815c553a70a4322aa8e49a51aeb0d46b75ccf2178b435c9b0ba0fedec3da5e0c`; canonical report SHA-256
is `90da22207398ae907e6b0d0bc93881c689a2a7362a1e52aac5435deac525b5d5`. This is an ordinary
same-engine rendered smoke, not numerical parity; exact-noise full-chain parity and synchronized
browser performance remain pending.

The final-source packed-F16 Turbo first-DMD diagnostic passed with outcome
`diagnostic-passed-no-full-parity-claim`. It binds JavaScript SHA-256
`64197f892ae850d901a9b76ff70dba7f543fa70af02028f605bb9eb126dc1b37`, WebAssembly SHA-256
`001f0bcc93fbaeea9a9b32d2adcb8b46f1897b80d36386919c03d69869dca86b`, probe/harness SHA-256
`d6485fa204233b25c1d12128410ae162a8d1ce59053179d3f31a3db63155dd88`, contract SHA-256
`dadd84a4ef9c5162c4aea7f3251cb40461e08d969bfaada3ae99f94dc6fb4b86`, report SHA-256
`0a600471ec9e3119eeaebd616e9dd29a84c62067881b6f9834949019f92d5eab`, and console SHA-256
`20af8aa43d3a53608c0658fabbef0fb8d7e85f2ff4b9655736fc74b959d489fe`. Its exact cache inventory
was 46 stages / 106 objects / 912 tensors; the single prediction performed 46 stage
materializations and 106 object unpacks, read 19,870,010,624 packed bytes, wrote 39,740,021,248
dense-F32 bytes, and made zero DMD artifact/range/cache/network traffic. Velocity relative RMSE /
cosine were `0.03869645` / `0.9992708`, and prediction relative RMSE / cosine were `0.042713966` /
`0.99910367`. `/dev/shm` passed quota-aware admission, and process-group exit, Chrome-profile
removal, and artifact-server teardown were clean. This is one-prediction diagnostic stability
evidence only: it makes no calibrated numerical-correctness, full-chain/full-resolution parity, or
fully on-device-quantized execution claim.

A predecessor core-Q8 Turbo first-DMD diagnostic is retained as historical evidence. It produced
finite velocity/prediction metrics of `0.039951164` / `0.044186924` relative RMSE and
`0.99920744` / `0.9990256` cosine, with report SHA-256
`62e14a0712811e088b33d74d047fa6370c5ae8191920cbbc87313b93fb3e68d0`. Its outcome was
`diagnostic-passed-no-full-parity-claim`; it does not qualify the current packed-F16 policy.

The first current-source modular 1536x1536 Edit-Turbo 1.5K attempt completed all model work and its
inner report passed every numerical gate, but its saved outer report was `ok=false`: the host still
expected retired field `audited_max_streamed_stage_bytes` instead of the runtime's canonical
`audited_max_streamed_qwen_stage_f32_bytes`. Offline replay against the corrected validator was
noncanonical even though it had zero failures. The subsequent 2026-08-14 source-bound rerun is the
promoted pass: top-level `ok=true`, all artifact/fixture/numerical gates true, 443/443 matched GPU
intervals, 224 active, peak 29,828 MiB / 31,276,924,928 bytes, and peak Wasm linear memory
2,009,137,152 bytes. Its exact dtype audit matched 377 Q8S and 565 F32 tensors; final RGB was
`34.531677 dB` PSNR / `0.99253726` SSIM. Report SHA-256 is
`c895ae2c1cba3823afe756035b6e564d5ef27caf3722f5f350c07e23086e3b54`; the report binds harness
SHA-256 `cee29e844c33325a2dac1e29b3a03f731f61be2b926ade93a1a50f5443b8efd8` and contract SHA-256
`d6a0ff5b8ebe8890be831efd1909cd36e2ede9709dc119fe1d965d4b8aa414ea`.
`browser-layer-streamed-diagnostic` is the separate host-heavy diagnostic. The synchronous native
directory factory is not compiled for Wasm.

The opt-in, no-surface 1536-square qualification route retains exactly the 48 verified denoiser
stages across four DMD steps, then clears them before its exact VAE decoder. Its current modular
canonical rerun passes on the pinned RTX PRO 6000 Blackwell/Chrome 151 stack, with 723,075,072 bytes
of headroom below the strict decimal cap. When the browser starts with Edit-Turbo 1.5K, the ordinary
UI exposes all official shapes, but its rendered-window, other-shape, performance, and cross-stack
gates remain separate.
The modular `qualification-f32` route is an optional non-blocking control
diagnostic, disabled by default in the release workflow; its last run ended in device loss and it
cannot replace the mandatory low-VRAM numerical and measured-memory gate. Diagnostic storage
profiles remain explicitly unqualified.

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

`StreamingArtifactLoader` retains its generic range contract for legacy objects. Canonical
manifests keep logical Burnpacks below the 256 MiB semantic ceiling and seal a sidecar that
reconstructs them from content-addressed physical CDN parts. Parts target 20,971,520 bytes and may
never exceed 25,000,000 bytes. Canonical Wasm loading uses one HTTP 200/CacheStorage entry per part,
checks any exposed identity framing, verifies exact `Blob.size` and the part SHA-256, then copies
only that bounded part into Wasm. Compact declared files use the same whole-object path. Legacy
direct Burnpacks retain exact 206 ranges and 4 MiB cache entries. A complete model bundle never
enters Wasm linear memory.

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
install -m 0644 crates/bevy_image/www/burn-image-icon.png \
  crates/bevy_image/www/out/burn-image-icon.png
rg '^export function (start_boogu_web|provide_reference_image)' \
  crates/bevy_image/www/out/bevy_burn_image.js
npx --yes serve crates/bevy_image/www --listen 8080

# Contract-only library without the window or Burn bridge
cargo test -p bevy_burn_image --no-default-features
```

The browser host must provide an element with id `burn-image`. `BrowserBooguFactory` uses async
Qwen/VAE/denoiser sources and passes exact ranges through `fetch_browser_range`. Loading holds at
most one digest-verified semantic payload in Wasm at a time and releases it after upload. High-VRAM
mode retains all resulting modules; low-VRAM Edit retains a request-scoped runtime-Q8 denoiser,
while ordinary low-VRAM Turbo retains packed F16 only for the DMD phase, widens one dense-F32 stage
at a time, and evicts the packed cache before VAE decode. Select the bundle and policy with
`variant`, `profile`, `residency`, and `artifacts`
query parameters as documented in the repository
[web deployment notes](https://github.com/mosure/burn_image/blob/main/docs/web.md). On canonical
pages without an explicit `artifacts=` override, the HTML release selector changes `variant` and
reloads the page so only one model is resident. An explicit custom artifact URL locks that selector
because it names one exact bundle. Native file
helpers remain excluded from Wasm builds.

The repository's GitHub Pages workflow packages the tracked `www/` shell, the exact app-icon copy,
and ignored generated `www/out/` bindgen output only; model objects are never copied into Pages and must be published on
the Aberration CDN before the deployment gate can pass. During runtime build and
inference, Wasm dispatches structured `burn-image-runtime` and `burn-image-progress` DOM events.
The shell turns them into exact current-object bytes/shard progress, transfer rate and ETA,
verified-object totals, stage/step state, terminal errors, and a manual full-runtime reload action.

The Wasm feature sets compile and package. Before low-VRAM became the browser production default,
an externally attested headful X11/Vulkan `headless=infer` run completed one real 256x256 Turbo
request with the `f16-qwen-vision-f32` bundle through the then-layer-streamed Qwen, four DMD steps,
and VAE decode; it encoded and attached a 60,926-byte PNG Blob. That evidence proves real WebGPU
execution for the historical policy. The first current-source modular low-VRAM 1536x1536
computation produced positive inner evidence but failed the stale host-key contract; the corrected
source-bound rerun now provides the canonical no-surface numerical/memory pass described above. It
does not qualify the explicit resident policy, the optional `qualification-f32` control
diagnostic, rendered-window behavior, another shape, or performance. The
historical core-Q8 first-DMD result likewise does not transfer to ordinary Turbo's packed-F16
policy. An earlier Chrome 151 Bevy swap-chain failure is retained as historical evidence. Current
serialized Run C and ordinary packed-F16 Turbo 1024 now pass their narrower diagnostic and ordinary
output scopes as described above. The canonical same-engine two-request ordinary smoke also passes;
exact-noise parity and performance gates remain pending.

Browser execution uses raw CubeCL without Burn fusion and adapts floating model stages to F32.
The hardware adapter and requested device both reported `shader-f16=false`; the explicit
`headless=f16-probe` therefore rejects before artifact construction rather than pretending the
mixed F16 execution policy is available. The Boogu app requests a shape-aware
1,217,126,400-byte storage-buffer/buffer limit covering every released browser shape; the
model-neutral app keeps portable baseline limits. Concrete factories accept only
`ArtifactCachePolicy::UseCached` until
refresh/bypass semantics are implemented. See the repository [web deployment
notes](https://github.com/mosure/burn_image/blob/main/docs/web.md) for the exact commands, hashes,
resource bounds, and distinction between no-surface inference, the Bevy window smoke, and
numerical parity.
