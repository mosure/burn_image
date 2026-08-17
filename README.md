# burn_image 🕊️🔥🖼️

[![CI](https://github.com/mosure/burn_image/actions/workflows/ci.yml/badge.svg)](https://github.com/mosure/burn_image/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![Burn](https://img.shields.io/badge/Burn-0.21-orange.svg)](https://burn.dev)
[![Bevy](https://img.shields.io/badge/Bevy-0.19-232326.svg)](https://bevy.org)

Burn-native image generation and editing with
[Boogu-Image 0.1](https://github.com/boogu-project/Boogu-Image). Use it through the Bevy desktop
viewer or embed the model-neutral Rust API. Native WGPU and browser WebGPU share the same model
descriptors and GPU-only execution contract. Current-source native low-VRAM Edit-Turbo 1.5K,
current modular browser low-VRAM Edit-Turbo 1.5K at 1536x1536, and ordinary rendered browser Turbo
1024 all have positive real-GPU results. The browser 1.5K claim is limited to its exact no-surface
numerical and measured-memory gate; other shapes and rendered/resident/performance paths remain
separate gates.

## Support

| Model path | Current status |
| --- | --- |
| Turbo 1K generation | Native high-VRAM production path is qualified; a cold native low-VRAM 1024x1024 output-qualification candidate passes the artifact, output, and strict memory gates; ordinary rendered browser low-VRAM 1024 also passes its model/output/surface/memory gate and the same-seed native/browser quality floor, while exact-noise full-chain parity and synchronized browser performance remain pending |
| Edit-Turbo 1K | Native high-VRAM production path is qualified; browser full-resolution UI/runtime support is implemented, with current rendered real-GPU gates pending |
| Edit-Turbo 1.5K | Native high-VRAM and current-source low-VRAM numerical gates pass; the first current modular browser low-VRAM 1536x1536 computation passed its inner gates but failed only the stale host-key contract, and the corrected source-bound canonical rerun now passes; other shapes, resident UI, rendered-window behavior, performance, and cross-stack portability remain separate; `qualification-f32` is an optional non-blocking control diagnostic, not a release gate |
| Low-VRAM execution | Native modular Turbo 1024 passed its output-qualification memory gate at 28,369,223,680 sampled bytes; current-source native historical-flat 1.5K passed at 30,310,137,856 bytes; ordinary rendered browser Turbo 1024 passed at 23,932,698,624 bytes; current modular browser 1.5K passed its exact no-surface gate at 31,276,924,928 bytes; other release/runtime tuples remain pending |

> [!IMPORTANT]
> These are large models. Expect tens of gigabytes of artifacts and a workstation-class GPU for
> high-VRAM resident execution. The low-VRAM policies instead bound verified phase-local loading
> while keeping model-layer execution on GPU. The browser never loads the complete bundle into
> Wasm memory. Here, 32 GB means decimal 32,000,000,000 bytes, not 32 GiB. Static plans are not
> measurements; the native 1024x1024 Turbo and native/browser 1536x1536 results named below are
> measured gates with different correctness scopes.

## Quick start

The full viewer requires Rust 1.95 or newer:

```bash
cargo run --release --locked -p bevy_burn_image \
  --features boogu-native --bin burn-image-viewer -- \
  --variant turbo \
  --profile production \
  --residency native-high-vram
```

`--profile production` is the stable user-facing selector. It maps to the sealed manifest and
provenance profile `f16-qwen-vision-f32`: the denoiser, Qwen text tower, and VAE are stored as F16,
while the Qwen vision tower remains F32. The longer name is intentionally retained only where
low-level artifact/parity tools need the exact immutable identity; it does not mean an F32 model.
The five-entry canonical CDN release is public. The viewer resolves and verifies it, and native
clients cache it under `~/.burn_image/models/`; `--artifacts /path/to/production-bundle` selects a
local mirror. A 2026-08-16 browser probe authenticated the composed manifest, both dependencies,
all three transport layouts, and cold/warm physical-part loading from the public CDN. Pages remains
strict on sealed manifest digests and every physical payload; it warns, but does not block, when a
reusable `manifest.json` response is `immutable` instead of the recommended `no-cache` policy.

The upload-ready artifact has distinct semantic, physical, and browser-cache layers. Logical
Burnpacks remain bounded by 256 MiB; the sealed transport layout reconstructs them from physical
parts targeting 20,971,520 bytes with an exact 25,000,000-byte hard maximum; and the browser stores
each authenticated physical part as one complete CacheStorage object. Legacy direct Burnpacks keep
authenticated range entries no larger than 4 MiB. Each network object completes into a
browser-owned `Blob` whose exact size is checked before one bounded copy enters Wasm. Preparing this
local tree does not replace remote payload authentication or a full browser model qualification.
Enter a prompt, select **Run**, then save the generated PNG from the viewer.

Interactive native builds use static CubeCL kernels by default, so opening the app or trying a new
shape never starts an opaque tuning pass. Long-running workloads can opt in with
`--features boogu-native,native-autotune` and then choose `--autotune balanced`; qualification uses
`--autotune full`. The first uncached feature-enabled request deliberately benchmarks candidates
and must not be treated as ordinary interactive latency.

Native automation uses ordinary CLI arguments and the same validated image-job path as the UI; it
does not synthesize pointer or keyboard input. Supplying `--output` hides the window, waits for the
selected runtime, executes exactly one request, writes the PNG plus a JSON timing/provenance report,
and exits with a meaningful status code:

```bash
target/release/burn-image-viewer \
  --variant edit-turbo \
  --prompt "make the bird red" \
  --source input.jpg \
  --width 1024 --height 1024 --seed 0 \
  --output result.png \
  --report result.json \
  --timeout-seconds 1800
```

For generation, use `--variant turbo` and omit `--source`. Use `--show-window` only when visually
debugging an automated request; no window interaction is required. Run `burn-image-viewer --help`
for the complete contract.

For image editing, use `--variant edit-turbo` and select or drop a reference image. The distinct
1.5K checkpoint is selected with `--variant edit-turbo-1k5` and its matching artifact bundle on
native or browser builds. Canonical native interactive runs expose all three in the in-canvas Model
drop-down: the next **Run** releases the idle model and lazily loads only the new selection. This
keeps one GPU-resident model and avoids downloading every release at startup. Automated, explicit
custom-artifact, and qualification runs stay single-release. On canonical browser pages, the outer
**Model release** selector changes `variant` and reloads the page; an explicit custom `artifacts=`
URL locks that exact release.

To select the implemented native low-VRAM policy, replace the final quick-start argument with
`--residency low-vram`. It streams Qwen and the required VAE half by phase while retaining one
mixed-F16 denoiser across all four DMD steps. Its static plan is 30,585,112,576 bytes for Turbo and
30,971,005,440 bytes for either Edit release, including a 10,000,000,000-byte non-weight reserve.
These are fail-closed planning bounds, not measured VRAM evidence; see the parity guide for the
strict telemetry gates. A fresh-cache native Turbo 1024x1024 run independently passed its
output-qualification gate with `ok=true`: all 2,246 PID-scoped samples matched the process and were
nonzero, and peak framebuffer use was 27,055 MiB / 28,369,223,680 bytes during VAE decode. Its
phase peaks were 24,814,551,040 bytes for initialization, 27,087,863,808 for Qwen,
27,096,252,416 for DMD, and 28,369,223,680 for VAE decode. The canonical modular closure verified
223 weight objects, 253 files, 1,940 tensors, and 38,224,723,494 bytes under parent digest
`555019af867a80bb4d7cec5dc2f0ba60ae799071994a5fd24d7e71918cb9ce36`. Its 1024x1024 PNG
remained exactly 1,448,891 bytes with SHA-256
`b2cfbc50f7c8f9d486799abd8c5be90c8770059a1dbc020ad02ac41a91abfab1` across the allocator-policy
change. The cold process took `281.404 s`, including `212.459 s` of inference; six
`Loaded 0 autotune cached entries` records establish that its isolated XDG cache contained no
autotune entries. Report SHA-256 is
`4f67f468110addef18a4d6f27d4ed01ab57f1c3c03de7174e6450fe793d38376`.

That Turbo result is an output-qualification candidate, not exact-noise/reference numerical or
cross-runtime parity. It does not qualify browser Turbo, a warm performance row, or another shape.
The current-source native 1536x1536 Edit-Turbo 1.5K replay separately passed all numerical gates at
28,906 MiB / 30,310,137,856 sampled bytes, with 608 attempts, 607 matched/nonzero samples, and no
sampler errors. Its report SHA-256 is
`a013adfcd30b7e6b2323ecc3723b22396f9858d14dd3cdd4a0da2699e199abe3`; it uses the deliberately
separate qualified schema-v1 flat parity artifact digest
`4e8b12ac5ca95272f9009080a23baf1bc52d1b0e7aebf2e9e5f394a492369213`. 1K Edit remains unmeasured
in low-VRAM mode. Native Turbo low-VRAM obtains the lower allocator peak by
uploading the Qwen embedding directly in its released F16 dtype and using exact-size VAE transient
allocation with synchronization/cleanup before the decoder tail. These are low-VRAM-only,
math-neutral allocator policies; they do not change released tensors or extend to high-VRAM or
browser execution.

The release workflow intentionally takes two non-interchangeable artifact locations. Native
`boogu-full-parity` consumes the exact legacy schema-v1 flat bundle because its stage readers require
all three model owners in one directory. Canonical publication and browser gates consume the
five-entry schema-v2 modular root. Both are independently digest- and semantics-verified, and the
workflow rejects using the canonical 1.5K parent as the flat parity root. Its browser
`qualification-f32` control is disabled by default and non-blocking when explicitly dispatched; the
browser low-VRAM numerical and strict measured-memory outcome remains mandatory.

Interactive browser builds now default to `residency=resident`: after the mandatory shared-device
VRAM preflight, the selected Qwen/VAE/denoiser request graph is verified, materialized, and retained
on WebGPU before **Ready**, so repeat Runs reuse a warm pipeline. `residency=low-vram` remains the
explicit bounded-memory path. That policy is variant-aware: Edit retains a
request-scoped runtime-Q8/F32 denoiser, while ordinary Turbo uses
`low-vram-preloaded-packed-f16-dense-f32-per-stage-denoiser`. Turbo authenticates 46 stages / 106
objects / 912 F16 tensors, retaining 19,870,010,624 padded packed-F16 bytes. Each DMD step widens one
semantic stage at a time on device and executes dense-F32 matmul; runtime Q8 is not part of this
Turbo policy. DMD is fail-closed on artifact, cache, or network traffic. After the fourth step, the
runtime transfers the exact final F32 latent across a synchronized boundary, evicts the entire
request-scoped packed cache before VAE decode, and rehydrates it from the verified persistent range
cache before the next request. The published low-VRAM evidence below remains scoped to that
explicit policy; the new warm interactive default requires refreshed browser hardware evidence.

The packed-F16 Turbo plan records a 22,304,263,424-byte preload peak and a 26,492,170,880-byte
conservative inference bound. Its exact-size persistent Qwen text-layer pool still requires the
measured aggregate GPU-memory gate; those static figures are not peak evidence. Initial preload is
106 verified objects / 19,870,166,528 bytes / 4,780 ranges. The first Generate request then reads
only Qwen text and VAE decoder objects, exactly 80 / 15,235,984,896 bytes / 3,709 ranges; a second
same-engine request must additionally rehydrate the 106 denoiser objects from cache and require zero
network responses. Although VAE transport applies only selected objects, the browser source
initializes the full 335,278,732-byte F32 autoencoder before selection.

Two current packed-F16 single-request browser runs are positive, with deliberately different claim
scopes. Serialized Run C completed the model, surface gate, measured-memory gate, and PNG download
at 23,932,698,624 bytes; report SHA-256 is
`b0dfcc8e53fd7ad1c4731d3169e2f43c50063aa2b54e5ca6347789e18630c6e6`. It is explicitly a
Qwen block-0 localization diagnostic, not release/output-quality or numerical-parity evidence. The
ordinary, non-serialized UI run then passed with report SHA-256
`36525be1d5ff482c409c3b7484027fcb335340e474e4a95f182720ea3f032a28`; it downloaded a
1,452,562-byte 1024x1024 PNG with SHA-256
`5d7a947301c9cefef8bca5cd42db52b98fe3fca440e98d302312eff6afe2eb38`, recorded zero page/GPU
errors and zero gated texture acquisitions, and stayed below the same measured peak. Against the
native same-prompt/seed output it passed the required final-RGB quality gate at `37.517250061 dB`
PSNR and `0.985732973` mean 8x8 block SSIM; quality-report SHA-256 is
`31da8e541013c38dd215257431159a99c7112ad79714a079e2a4b25f9c855103`. Independent RNG means
that quality comparison is not exact-noise numerical parity.

The canonical same-page/same-engine rerun passed two sequential ordinary requests with exactly one
adapter request, one device request, and one Chrome GPU process. Packed-cache preload attempts were
`[1, 2]`; request 2 read all 186 objects / 35,106,151,424 bytes / 8,489 ranges from persistent
cache, with 8,489 hits, zero misses, and zero network requests or bytes. Both requests completed
four zero-I/O DMD steps, exact Qwen and DMD-to-VAE handoffs, one violation-free surface-suspension
window, cache-ready-to-empty cleanup before VAE, and successful post-resume acquisition. The
process group exited, the Chrome profile was removed, and page/GPU error lists were empty. Peak
Chrome GPU-process memory was 24,384,634,880 bytes. The two distinct 1024x1024 downloads have
SHA-256 `5d7a947301c9cefef8bca5cd42db52b98fe3fca440e98d302312eff6afe2eb38` and
`815c553a70a4322aa8e49a51aeb0d46b75ccf2178b435c9b0ba0fedec3da5e0c`; canonical report SHA-256
is `90da22207398ae907e6b0d0bc93881c689a2a7362a1e52aac5435deac525b5d5`. This is an ordinary
same-engine rendered smoke, not numerical parity. Exact-noise full-chain Turbo parity and
synchronized browser performance remain pending.

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

The explicit `residency=resident` path is an advanced dense-F32 high-VRAM mode and remains
unqualified. The first current-source modular browser 1.5K run produced positive inner numerical,
artifact, fixture, and sub-cap memory evidence, but its immutable outer report failed only because
the host expected retired field `audited_max_streamed_stage_bytes`. With the host contract corrected
to the runtime's canonical `audited_max_streamed_qwen_stage_f32_bytes`, the 2026-08-14 source-bound
rerun passed with top-level `ok=true`: 443/443 GPU samples matched, 224 were active, peak use was
29,828 MiB / 31,276,924,928 bytes (723,075,072 bytes below the decimal cap), and final RGB was
`34.531677 dB` PSNR / `0.99253726` SSIM. Report SHA-256 is
`c895ae2c1cba3823afe756035b6e564d5ef27caf3722f5f350c07e23086e3b54`. This qualifies only the
current modular low-VRAM exact 1536x1536 no-surface numerical/memory scope; other 1.5K shapes,
rendered UI, the resident policy, synchronized performance, and cross-stack portability remain
separate gates.

Browser evidence harnesses choose Chrome shared-memory backing with BigInt `statfs` accounting plus
a real bounded 256 MiB write/`fsync`/delete quota probe. The current runs selected `/dev/shm` and did
not pass `--disable-dev-shm-usage`: `/dev/shm` completed all 268,435,456 probe bytes, while `/tmp`
hit its effective quota despite reporting free blocks. This is quota-aware admission, not a
`statfs`-only capacity claim.

For local browser builds and GitHub Pages deployment, see the [web guide](docs/web.md). The same
Bevy UI used natively reports the current component, aggregate part transfer, verification, and
inference stage so a large model load does not look frozen; there is no separate browser overlay.
Every ordinary browser model policy stores the selected executable closure under exact
URL/digest/size Cache Storage keys. Startup counts existing keys, checks origin quota for only the
missing bytes plus reserve, and requests eviction-resistant storage when available. Browser caches
are origin-scoped, so localhost and the GitHub Pages app do not share downloaded models. Before the
ordinary page requests weight parts, it also commits and
releases the selected policy's conservative GPU-memory plan on the exact shared Bevy/Burn WebGPU
device; failure stops before the large CDN transfer rather than treating API buffer limits as VRAM.

This workspace carries two required root `[patch.crates-io]` entries: patched `wgpu` 29.0.4 bounds
browser uploads and surfaces rejected `GPUQueue.onSubmittedWorkDone()` promises, while patched
`cubecl-wgpu` 0.10.0 submits pending upload-only work and propagates queue/error-scope failure through
asynchronous synchronization. Those patches protect checkout and Pages builds, but Cargo does not
propagate a dependency's root patches. A downstream application consuming the published crates must
vendor/apply equivalent `wgpu` **and** `cubecl-wgpu` patches in its own workspace root until both
changes are available from upstream/resolvable releases. A graph missing either patch is outside
the browser support claim.

## Workspace

| Crate | Responsibility |
| --- | --- |
| [`burn_image`](crates/burn_image) | Model-neutral requests, jobs, manifests, integrity, transport, and provenance |
| [`burn_qwen3_vl`](crates/burn_qwen3_vl) | Qwen3-VL model/processor plus its sealed component, semantic shard loader, and device cache |
| [`burn_flux_vae`](crates/burn_flux_vae) | FLUX `AutoencoderKL` plus its sealed component, encoder/decoder shard loader, and device cache |
| [`burn_boogu`](crates/burn_boogu) | Boogu conditioning, variant-specific denoiser, DMD composition, and native runners |
| [`bevy_burn_image`](crates/bevy_image) | Native/web Bevy UI and shared WGPU integration |

The frontend package is named `bevy_burn_image` because Bevy already publishes a crate named
`bevy_image`.

## Development

```bash
cargo fmt --all -- --check
cargo test -p burn_image --all-targets --locked
cargo clippy -p burn_image --all-targets --locked -- -D warnings
```

The [CI workflow](.github/workflows/ci.yml) contains the authoritative package-isolated test,
feature-contract, and WebAssembly build matrix. Use it instead of combining every model and
frontend feature into one linker-heavy workspace test.

## Documentation

- [Architecture](docs/architecture.md) — runtime structure, component ownership, and GPU lifecycle
- [Artifacts](docs/artifacts.md) — profiles, conversion, verification, caching, and CDN layout
- [Numerical parity](docs/parity.md) — fixtures, thresholds, and current model evidence
- [Performance](docs/performance.md) — methodology, accepted results, and remaining gaps
- [RTX PRO 6000 report](docs/reports/2026-08-11-rtx-pro-6000.md) — detailed machine evidence
- [Web deployment](docs/web.md) — Wasm build, Pages deployment, loading UI, and browser limits

Exact checkpoint revisions, artifact digests, numerical metrics, benchmark samples, and GPU
telemetry live in those focused documents rather than on this landing page.

## License

Dual-licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
Checkpoint and upstream source licenses remain those published by their respective authors;
converted artifacts are not committed to this repository.
