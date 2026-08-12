# burn_image 🕊️🔥🖼️

[![CI](https://github.com/mosure/burn_image/actions/workflows/ci.yml/badge.svg)](https://github.com/mosure/burn_image/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![Burn](https://img.shields.io/badge/Burn-0.21-orange.svg)](https://burn.dev)
[![Bevy](https://img.shields.io/badge/Bevy-0.19-232326.svg)](https://bevy.org)

Burn-native image generation and editing for native WGPU and browser WebGPU. The first model family is
[Boogu-Image 0.1](https://github.com/boogu-project/Boogu-Image): ordinary Qwen3-VL and FLUX VAE
crates plus the Boogu denoiser and four-step DMD pipeline.

## status

- ✅ model-neutral generate/edit, artifact, integrity, progress, cancellation, and provenance API;
- ✅ reusable Qwen3-VL, FLUX `AutoencoderKL`, and Boogu Turbo model code;
- ✅ pinned SafeTensors import to bounded, content-addressed Burnpack shards;
- ✅ native hybrid WGPU factory and a Bevy frontend with real artifact dispatch;
- ✅ native Turbo and Edit-Turbo execution at the released 1024×1024 output ceiling, with
  synchronized upstream/Burn performance reports: the release-validated uncached Burn WGPU
  policy is `1.9643x/2.0059x` upstream at p50/p95 for Turbo and `1.8619x/1.8707x` for Edit,
  without conditioning reuse. Both p50 values and Edit p95 meet `<= 2x`; Turbo p95 is at the
  noise boundary, missing by `10.360 ms` in the canonical replay while a second exact-policy
  repeat passed at `1.9804x`;
- ✅ a distinct native `edit-turbo-1k5` runtime/configuration for the official 1.5K hotfix,
  including its 1536×1536 default, released aspect-ratio presets, immutable revision, and separately
  sealed artifacts; exhaustive q128 Qwen and portable-denoiser component gates plus the exact
  accelerated production full-chain native WGPU gate pass. Its synchronized uncached performance
  is `2.1032x/2.1371x` upstream at p50/p95—much closer than the historical `5.55x`, but still not
  within the strict `<= 2x` target;
- ✅ concrete browser factory with digest-verified, bounded async Qwen/VAE/denoiser streaming;
- ✅ one complete 256×256 Turbo request on headful Chrome hardware WebGPU through the surface-free
  factory, including a SHA-256-recorded PNG Blob result;
- 🧪 opt-in, real-fixture native parity tools (large snapshots/converted bundles are required);
- ⚠️ the production Bevy window still loses Chrome 151's WebGPU device during headless swap-chain
  setup. The complete no-surface request is real browser inference, but it is not a rendered Bevy
  UI pass or a numerical-parity claim.

## workspace

| Crate | Responsibility |
| --- | --- |
| [`burn_image`](crates/burn_image) | Model-neutral requests, outputs, jobs, manifests, integrity, and transport contracts |
| [`burn_qwen3_vl`](crates/burn_qwen3_vl) | Ordinary Qwen3-VL text/vision model, processor, MRoPE, and weight inventory |
| [`burn_flux_vae`](crates/burn_flux_vae) | Diffusers-compatible FLUX `AutoencoderKL` and deterministic posterior sampling |
| [`burn_boogu`](crates/burn_boogu) | Boogu conditioning, denoiser, latent layout, DMD, artifacts, and native runner |
| [`bevy_burn_image`](crates/bevy_image) | Native/web Bevy controls, progress, image I/O/display, and shared WGPU integration |

The frontend lives in `crates/bevy_image`, but the package is `bevy_burn_image` because Bevy
already publishes a crate named `bevy_image`.

## checkpoints and profiles

| Runtime identity | Source repository | Task/configuration | Pinned Hugging Face revision |
| --- | --- | --- | --- |
| `Boogu/Boogu-Image-0.1-Turbo` | `Boogu/Boogu-Image-0.1-Turbo` | text-to-image, 1K | `53ad54522023f64d049f7f38e4d679359ef3fb92` |
| `Boogu/Boogu-Image-0.1-Edit-Turbo` | `Boogu/Boogu-Image-0.1-Edit-Turbo` | one-image edit, 1K | `132a0ab9051b42c1d9be4919a68873d1f132c0c8` |
| `Boogu/Boogu-Image-0.1-Edit-Turbo-1K5` | `Boogu/Boogu-Image-0.1-Edit-Turbo` | one-image edit, 1.5K hotfix | `60981c49e48cffadf2c169532a4ba3f6108afd5e` |

The upstream implementation is pinned at `25f8f888298224a94e5ec2abafb98abea9031a0d`.
The upstream `hotfix-1k5-20260708` revision resolves to the immutable 1.5K commit shown above.
The logical 1.5K identity deliberately does not alias the 1K release: its importer, manifest,
runtime selection, fixture metadata, and reports must all name `edit-turbo-1k5` and the 1.5K
revision, even where Hugging Face can reuse identical content-addressed objects. Omitted dimensions
default to 1536×1536. The released native presets are 1536×1536, 1264×1856, 1856×1264,
1344×1744, 1744×1344, 1392×1696, 1696×1392, 1152×2032, 2032×1152, and 2368×992, with a
2,360,832-pixel ceiling; arbitrary in-envelope dimensions now fail closed. The 1536×1536 default
has the numerical and synchronized performance evidence below. The other official presets are
routed and bounded but have not each received a separate checkpoint benchmark.

The converted mixed-F16 1.5K bundle is sealed as
`4e8b12ac5ca95272f9009080a23baf1bc52d1b0e7aebf2e9e5f394a492369213`; a strict verifier covered
249 files, 38,224,723,721 bytes, 223 objects, and 1,940 tensors. Its verifier-report SHA-256 is
`b702ef29e4a9e26afc67d2fdf8ad46582f561995d6041763439a6ed7ef3ee64f`. This proves the distinct
artifact identity/inventory, not native numerical or performance parity.

The fixture-injected 1536×1536 native WGPU full chain passes its exact released policy: full
autotune, retained Qwen q128 with synchronization deferred to the mandatory Qwen-stage boundary,
forced padded-blackbox denoiser `p4/kv1/q1` q16384 with strict-F32 RMSNorm and composed Q/K
preparation, and preserved-F16
VAE q4096 with safe mixed GroupNorm. Qwen relative RMSE/cosine is
`0.09276403/0.9956884`, worst DMD velocity `0.12193716/0.99257636`, final latent
`0.07782394/0.99699116`, decode `0.08242943/0.99667007`, and RGB
`33.72779 dB/0.9920097` SSIM. The report SHA-256 is
`5834c99b9139ce054891b30e29ecad4bf2d8a4ed1fb9d6482628d1d7350e87ef`; stderr is empty and the
report records `edit_1k5_release_policy_validated=true`.

`f16-qwen-vision-f32` is the parity-oriented default: Qwen vision weights remain F32 and other
weights use F16. `q8s-block32-f32-qwen-vision-f32` additionally stores eligible non-vision linear
matrices as symmetric block-32 Q8 with F32 scales. On Burn 0.21, the row-layout Boogu denoiser can
keep those weights quantized with F32 activations; Qwen's column-layout mapper cannot, so its
verified Q8 objects are dequantized through F16 one bounded stage at a time. The profile is a
smaller transport/storage format, not a claim that every layer stays quantized on device. The
all-F16 and all-eligible-Q8 profiles remain diagnostic alternatives. Official FP8 weights depend
on CUDA/Triton semantics and are not presented as a portable WebGPU format.

For the measured native high-VRAM path, the sealed F16 VAE can remain F16 instead of being adapted
to F32. Its direct decoder oracle gate and the optimized Turbo and Edit fixture-injected full-chain
gates pass at 256×256. The exact release-validated parity report SHA-256 digests are
`055548fcbe5bd4e2ff612e57f0038cf194b82f0114e47b3095f925ee0d0fa948` and
`746f44d18d66b2e348ce00bc0425777477137e6ba917d0b91d4a46f1a6dcf4f5` respectively. Both reports
have empty stderr, pass every gate, record `native_release_policy_validated: true`, and name
balanced strict Q/K preparation plus `vae_attention_query_chunk_size: 4096`.

`burn-image-viewer` applies that exact q128/Q8192/VAE-q4096 `p4/kv1/q1` policy to Turbo and 1K Edit
when `native-high-vram` and `f16-qwen-vision-f32` are selected. It uses strict-F32 denoiser
RMSNorm with balanced Q/K preparation, defers synchronization to the retained Qwen stage boundary,
configures full autotune before device creation, and preserves the VAE in F16 with safe F32
GroupNorm accumulation. The complete policy is recorded in backend provenance. The factory fails
closed if an embedder omits full autotune. The 1.5K, browser, layer-streamed, Q8, and all-F16 paths
keep their separately qualified policies.

Native inference prepares the four deterministic DMD host-noise buffers concurrently with
uncached Qwen/Edit-VAE GPU work, while cached text-only and Wasm requests retain serial
generation. Thread-spawn failure falls back to that exact serial path. Decoder readback converts
F16 values directly to validated RGB8 bytes instead of allocating a full intermediate F32 image;
the final Turbo PNG remains byte-identical.

## setup

### build and test

```bash
cargo test --workspace --all-features --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo check -p bevy_burn_image --target wasm32-unknown-unknown \
  --no-default-features --features boogu-web --lib --locked
cargo build -p bevy_burn_image --target wasm32-unknown-unknown \
  --profile wasm-release --no-default-features --features boogu-web --locked --lib
mkdir -p crates/bevy_image/www/out
wasm-bindgen --target web --out-dir crates/bevy_image/www/out \
  --out-name bevy_burn_image \
  target/wasm32-unknown-unknown/wasm-release/bevy_burn_image.wasm
```

### download and convert

`hf download` prints the immutable snapshot directory used below. Invoke it once for each pinned
release identity; the examples cover Turbo and the distinct Edit-Turbo 1.5K revision.

```bash
export BURN_IMAGE_TURBO_SNAPSHOT="$(hf download \
  Boogu/Boogu-Image-0.1-Turbo \
  --revision 53ad54522023f64d049f7f38e4d679359ef3fb92)"

cargo run -p burn_boogu --release --features import --bin boogu-import -- \
  --source "$BURN_IMAGE_TURBO_SNAPSHOT" \
  --output .artifacts/boogu-image-0.1-turbo-f16-qwen-vision-f32 \
  --variant image01-turbo \
  --profile f16-qwen-vision-f32

export BURN_IMAGE_EDIT_1K5_SNAPSHOT="$(hf download \
  Boogu/Boogu-Image-0.1-Edit-Turbo \
  --revision 60981c49e48cffadf2c169532a4ba3f6108afd5e)"

cargo run -p burn_boogu --release --features import --bin boogu-import -- \
  --source "$BURN_IMAGE_EDIT_1K5_SNAPSHOT" \
  --output .artifacts/boogu-image-0.1-edit-turbo-1k5-f16-qwen-vision-f32 \
  --variant image01-edit-turbo-1k5 \
  --profile f16-qwen-vision-f32
```

The importer requires the selected canonical revision to be asserted by the snapshot path or a
revision marker, rejects partial inventories, and hashes the exact supplied content. It writes
validated Burnpack objects, the complete tensor inventory, per-file SHA-256 digests, ordered
shard-chain states, and the sealed bundle digest without assembling the full checkpoint in memory.
The content seal proves what was converted; it does not independently prove that a user-authored
revision marker came from Hugging Face.

### run

Run the real native hybrid pipeline directly:

```bash
cargo run -p burn_boogu --release --features "runtime,import,wgpu" \
  --bin boogu-run -- \
  --artifacts .artifacts/boogu-image-0.1-turbo-f16-qwen-vision-f32 \
  --variant turbo \
  --profile f16-qwen-vision-f32 \
  --vae-float-policy preserve-f16 \
  --denoiser-query-chunk-size 512 \
  --prompt "a lighthouse at dusk" \
  --output boogu-output.png
```

Or run the same production composition through the Bevy frontend:

```bash
cargo run -p bevy_burn_image --release --features boogu-native \
  --bin burn-image-viewer -- \
  --variant turbo \
  --profile f16-qwen-vision-f32 \
  --residency native-high-vram
```

With no `--artifacts` override, native resolves the exact bundle beneath
`https://aberration.technology/model/`, streams and verifies its sealed files, and commits it to
`~/.burn_image/models/<exact-manifest-bundle-id>/` only after all payloads pass. Add
`--artifacts .artifacts/boogu-image-0.1-turbo-f16-qwen-vision-f32` to use a local conversion.
`BURN_IMAGE_MODEL_BASE_URL` selects a custom bundle base;
`BURN_IMAGE_MODEL_MANIFEST_URL` changes only the manifest request while payload paths continue to
resolve from that base.

Enter the prompt in the viewer, then run, cancel, inspect progress, and save the
result from the same UI. For editing, use the matching Edit-Turbo bundle and
`--variant edit-turbo`, then choose or drop a reference image. Select `--variant edit-turbo-1k5`
with its separately sealed bundle for the native 1.5K configuration; omitted output dimensions use
1536×1536. Browser construction rejects that variant before artifact loading because real browser
WebGPU numerical and performance parity have not been established.

The default native policy verifies each Qwen and VAE stage on first use, retains their shared WGPU
handles for later requests, and keeps one denoiser resident across all four DMD steps. This is
intended for a high-VRAM GPU. `--residency native-layer-streamed` rereads Qwen and VAE stages per
request and denoiser stages per DMD step to reduce weight residency.

The model-neutral shell remains available for embedders that install another runtime:

```bash
cargo run -p bevy_burn_image --release --bin burn-image-viewer
```

Without `boogu-native`, the viewer deliberately reports that no model runtime is installed until
an embedding supplies a real `BooguRuntimeFactory`; it never substitutes a placeholder or Burn
CPU tensor-backend inference path. WebGPU adapter labels may be privacy-redacted, so browser
hardware claims additionally use scoped external GPU-process evidence. See
[`crates/bevy_image`](crates/bevy_image) for that integration boundary and
[web deployment](docs/web.md) for the concrete asynchronous browser factory and current Chrome
validation details.

To serve the concrete browser app, package into the ignored web output directory and pass an
HTTP-Range-capable sealed artifact URL:

```bash
cargo build -p bevy_burn_image --target wasm32-unknown-unknown \
  --profile wasm-release --no-default-features --features boogu-web --locked --lib
mkdir -p crates/bevy_image/www/out
wasm-bindgen --target web --out-dir crates/bevy_image/www/out \
  --out-name bevy_burn_image \
  target/wasm32-unknown-unknown/wasm-release/bevy_burn_image.wasm
npx --yes serve crates/bevy_image/www --listen 8080
```

Then open `http://localhost:8080/?variant=turbo&profile=f16-qwen-vision-f32`. With no explicit
`artifacts` override, the browser uses
`https://aberration.technology/model/boogu-image-0.1-turbo-f16-qwen-vision-f32`; local testing may
still pass an encoded bundle base URL. BrowserWebGpu advertises and defaults to the currently
validated exact 256×256 output size. The artifact origin must return `206` and expose
`Content-Range` through CORS. `headless=bootstrap` selects the short surface-free F32 stage
diagnostic; `headless=infer` runs the explicit real 256×256 Turbo request described in
[web deployment](docs/web.md). Neither replaces the production shared-device Bevy app.

## correctness

Reference fixtures are external inputs consisting of one `tensors.safetensors` file plus
`metadata.json`; tensor dtypes are preserved rather than flattened into `.npy` F32 files. Before
comparing values, every parity binary requires the metadata and SafeTensors keysets to match and
authenticates each tensor's exact shape, dtype, and raw-byte SHA-256. Release fixtures are accepted
only when their whole-file digests match the release evidence. The Rust parity binaries enforce
processor data, streamed Qwen stages/final hidden state, FLUX VAE encode/decode, final RGB metrics,
DMD arithmetic, and streamed denoiser boundary gates when their checkpoint/artifact inputs are
supplied. `boogu-qwen-parity`, `boogu-parity`, `boogu-denoiser-parity`, and
`boogu-full-parity` are the real executable parity entry points. The last one propagates the
production execution-dtype schedule through Qwen, edit VAE encode, all four DMD steps, VAE decode,
and final RGB comparison.

Ordinary unit CI does not silently count a missing checkpoint as real parity. Release parity is
dispatched with explicit snapshot, sealed-artifact, and external fixture-cache paths, then invokes
the Cargo parity binaries named above. Reference SafeTensors, images, model weights, generated
fixtures, and parity outputs are intentionally not stored in Git; the external fixture package and
release evidence carry their exact hashes and provenance contract. Browser compilation and smoke
results remain separate from native numerical parity. The release workflow unconditionally
requires Turbo, 1K Edit, and the complete 1.5K tuple
`BURN_IMAGE_EDIT_1K5_SNAPSHOT`, `BURN_IMAGE_EDIT_1K5_ARTIFACTS`, and
`BURN_IMAGE_EDIT_1K5_FIXTURE`; a legacy two-release run is diagnostic and is not a release gate.
The external 372-tensor fixture and exact sealed-bundle
digest are authenticated before use. Qwen q128 compares all 70 semantic stages, the portable
streamed denoiser compares 240 internal boundaries, and the accelerated production configuration
passes its propagated full-chain gate. Browser support remains explicitly unavailable.

The finalized uncached native-WGPU comparison uses forced padded blackbox attention
(`p4/kv1/q1`), balanced strict Q/K preparation, an 8192-row denoiser query bound, retained q128
Qwen with deferred synchronization, and the F16 VAE with safe mixed GroupNorm and a 4096-row VAE
query bound. With one cold request followed by five uncached warm requests, the post-promotion
release replay measured Turbo at `3.400695447/3.494256796 s` p50/nearest-rank p95
(`1.964348x/2.005947x` the matching upstream percentiles); Edit measured
`3.908315376/3.933613488 s` (`1.861895x/1.870652x`). Both p50 values and Edit p95 meet the target.
Turbo p95 misses by `10.360 ms`; an independent exact-policy repeat measured
`3.344787549/3.449805839 s` (`1.932053x/1.980429x`), so the tail is near the measurement-noise
boundary rather than robustly below it. The original `5.82x` Turbo and `5.55x` Edit Burn baselines
remain the historical starting point.

During the release replay, hot board samples averaged `99.82%` GPU utilization for both Turbo and
Edit, with `44,129/46,045 MiB` PID framebuffer peaks. A separate 1.5K resident-path
profile averaged `99.89%` across 37 hot board samples and reached a `46,041 MiB` PID peak. These
board aggregates demonstrate GPU-resident, compute-busy execution but are not process-exclusive
per-kernel occupancy measurements; the benchmark report records the sampling filters and hashes.

The separately gated 1536×1536 Edit-Turbo 1.5K run used disabled conditioning cache and the exact
policy named above. Against the matching upstream BF16 p50 and nearest-rank p95, Burn measured
`12.553919/12.830840 s` versus `5.968882/6.003932 s`, or `2.1032x/2.1371x`. Performance parity is
not achieved: it misses the two-times budgets by `0.616/0.823 s`. The output is a verified
1536×1536 RGB PNG; the benchmark JSON SHA-256 is
`19cd7392f51e5e1b2a777c56a7e95b7f1220c6c30bddc16f2271649b159a3644`.

An earlier q1024 production-policy measurement of the exact-request single-entry cache reused
instruction conditioning only when prompt, source, and effective token length all matched. It
measured Turbo at `3.396570999 / 3.475443177 s` p50/p95 (`1.96196x`/`1.9976x`) and Edit at
`3.904326915 / 3.906317438 s` (`1.859995x`/`1.858183x`). These cached ratios describe
historical repeated-identical-request behavior; they neither qualify the current q4096 policy nor
form a strict comparison with upstream, which recomputed conditioning in its benchmark. A native
CUDA timing probe produced invalid output and remains excluded. See the benchmark report for
evidence hashes, GPU telemetry, and rejected optimization diagnostics.

- [Architecture](docs/architecture.md)
- [Artifact conversion and CDN layout](docs/artifacts.md)
- [Numerical parity](docs/parity.md)
- [Performance methodology](docs/performance.md)
- [RTX PRO 6000 benchmark](docs/reports/2026-08-11-rtx-pro-6000.md)
- [Web deployment](docs/web.md)

## license

Dual-licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
Checkpoint and upstream source licenses remain those published by their respective authors;
converted artifacts are not committed to this repository.
