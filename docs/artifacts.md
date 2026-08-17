# Artifacts

Burn Image artifacts are deployment objects, not opaque checkpoint archives. Conversion preserves
enough information to prove exactly what was loaded and to stream only the component needed next.

## Production storage policy

The CDN release contract defines one production policy per model. Viewer and browser entry points select it with
`--profile production` or `profile=production`; its precise sealed manifest profile remains
`f16-qwen-vision-f32` for provenance and compatibility. Low-level conversion and parity binaries
continue to accept the precise name because their reports must bind the immutable storage identity.
It is an internal description, not a request to execute the entire model in F32.

This is an F16-first bundle, not an F32 model:

| Component | Stored dtype | Reason |
| --- | --- | --- |
| Boogu denoiser, Qwen text, VAE | F16 | compact, accelerated, and parity-qualified |
| Qwen vision tower | F32 | the F16 vision path accumulated unacceptable numerical drift |

The mixed release closure contains 1,589 F16 tensor entries and 351 F32 Qwen-vision entries. The pinned
upstream checkpoints are predominantly BF16 for Qwen and the denoiser, with an F32 VAE; they are
not upstream F16 checkpoints. Storing the vision exception as F32 adds about 1.15 GB, roughly 3%,
over the rejected all-F16 candidate.

The confusing legacy name `q8s-block32-f32-qwen-vision-f32` describes two unrelated details:
Q8S values use 32-value quantization blocks with F32 scales, while Qwen vision is separately kept
F32. It is not an all-F32 bundle. Those Q8 bundles are smaller, but recorded lower fidelity and
slower warm inference, Qwen is dequantized stage-locally, and no 1.5K Q8 release is qualified.
They, plain `f16`, and plain `q8s-block32-f32` remain explicit diagnostics and are not part of the
production CDN upload set.

Burn 0.21 runtime application policy is component-specific. Q8 Qwen Col-layout matrices are
digest-verified, dequantized stage-locally through F16, and then transposed; directly transposing
their block scales is invalid. Boogu denoiser Row-layout Q8 matrices remain quantized, while its
non-quantized parameters and all DMD activations run F32. Native mixed-F16 routes preserve F16
denoiser execution where separately gated; this is not the ordinary browser Turbo policy. The
parity-qualified native high-VRAM `f16-qwen-vision-f32` route for Turbo and 1K Edit
preserves F16 VAE weights/activations and uses F16 storage with F32 GroupNorm reduction scalars.
Browser, layer-streamed, Q8, and all-F16 routes retain the strict `force_upcast=true` F32 VAE
policy. The separately gated native 1.5K release uses the same typed VAE precision contract with
its own larger attention bounds; neither policy is a profile-wide default.

The browser `low-vram` runtime does not introduce another CDN artifact profile. It verifies the
same `production` files and adapts bounded Qwen and selected VAE-stage objects. For VAE, the current
source initializes the complete 335,278,732-byte F32 `AutoencoderKL` before applying only selected
encoder or decoder objects; bounded transport therefore does not imply half-module device
allocation. Edit runtime-quantizes eligible denoiser matrices to Q8S block-32/F32 and retains them
for one request. The first current-source modular 1536x1536 Edit computation passed its retained
dtype audit, full-chain numerical checks, and strict memory cap, but a stale host resource-plan key
left the outer report failed. The corrected source-bound canonical rerun now passes that exact
no-surface numerical/memory gate. Source dispatch preserved QFloat weights into `q_matmul`, but the
run captured no backend kernel trace and therefore made no on-device quantized-execution claim.

Ordinary Turbo uses the same canonical F16 denoiser payload without runtime quantization. The
packed-F16 source authenticates exactly 46 stages / 106 objects / 912 tensors and
19,870,166,528 artifact bytes, then places 19,869,996,096 compact F16 payload bytes plus 7,264
alignment elements into 19,870,010,624 bytes of retained packed U32 arenas. During DMD, one
semantic stage at a time is widened on device and executed as dense F32. Four steps perform 184
stage materializations and 424 object unpacks with zero artifact/cache/network traffic.

The ordinary UI requires the integrity-bound persistent range cache. Initial preload reads the 106
objects over 4,780 ranges before **Ready**. The first Generate request reads only the 80 Qwen-text
and VAE-decoder objects / 15,235,984,896 bytes / 3,709 ranges. After DMD, an exact synchronized F32
latent handoff clears every packed arena before VAE decode; a second request therefore rehydrates
the denoiser's 106 objects from cache rather than preserving GPU residency. Its combined 186-object
/ 35,106,151,424-byte / 8,489-range read is required to be all cache hits with zero network
responses. The 22,304,263,424-byte preload peak and 26,492,170,880-byte inference plan are static
bounds, and the exact-size persistent Qwen text-layer pool still requires a measured aggregate
GPU-memory gate.

Current serialized Run C and the subsequent ordinary rendered Turbo 1024 run both authenticated
and executed this packed-F16 storage/application policy through output at a measured
23,932,698,624-byte Chrome GPU peak. Run C remains diagnostic-only (report SHA-256
`b0dfcc8e53fd7ad1c4731d3169e2f43c50063aa2b54e5ca6347789e18630c6e6`); the ordinary run is the
positive release/output smoke (report SHA-256
`36525be1d5ff482c409c3b7484027fcb335340e474e4a95f182720ea3f032a28`) and downloaded a
1,452,562-byte PNG with SHA-256
`5d7a947301c9cefef8bca5cd42db52b98fe3fca440e98d302312eff6afe2eb38`. The same-seed native/browser
quality comparison passed at `37.517250061 dB` PSNR / `0.985732973` SSIM; quality-report SHA-256 is
`31da8e541013c38dd215257431159a99c7112ad79714a079e2a4b25f9c855103`, but it is not exact-noise
parity. The canonical same-engine rerun passed two sequential ordinary requests on one adapter,
one device, and one Chrome GPU process, with packed-cache preload attempts `[1, 2]`. Request 2 read
all 186 objects / 35,106,151,424 bytes / 8,489 ranges from persistent cache, with 8,489 hits, zero
misses, and zero network requests or bytes. Each request completed four zero-I/O DMD steps,
digest-preserving Qwen and DMD-to-VAE handoffs, cache-ready-to-empty cleanup, and one violation-free
surface-suspension window followed by successful acquisition. Peak Chrome GPU-process memory was
24,384,634,880 bytes; page/GPU error lists were empty, the process group exited, and the Chrome
profile was removed. The distinct 1024x1024 PNGs have SHA-256
`5d7a947301c9cefef8bca5cd42db52b98fe3fca440e98d302312eff6afe2eb38` and
`815c553a70a4322aa8e49a51aeb0d46b75ccf2178b435c9b0ba0fedec3da5e0c`; canonical report SHA-256
is `90da22207398ae907e6b0d0bc93881c689a2a7362a1e52aac5435deac525b5d5`. This is an ordinary
same-engine rendered smoke, not numerical parity. The final-source packed-F16 Turbo first-DMD
diagnostic passed with outcome `diagnostic-passed-no-full-parity-claim`. It binds JavaScript
SHA-256 `64197f892ae850d901a9b76ff70dba7f543fa70af02028f605bb9eb126dc1b37`, WebAssembly SHA-256
`001f0bcc93fbaeea9a9b32d2adcb8b46f1897b80d36386919c03d69869dca86b`, probe/harness SHA-256
`d6485fa204233b25c1d12128410ae162a8d1ce59053179d3f31a3db63155dd88`, contract SHA-256
`dadd84a4ef9c5162c4aea7f3251cb40461e08d969bfaada3ae99f94dc6fb4b86`, report SHA-256
`0a600471ec9e3119eeaebd616e9dd29a84c62067881b6f9834949019f92d5eab`, and console SHA-256
`20af8aa43d3a53608c0658fabbef0fb8d7e85f2ff4b9655736fc74b959d489fe`. Its exact cache inventory
was 46 stages / 106 objects / 912 tensors; one prediction performed 46 stage materializations and
106 object unpacks, read 19,870,010,624 packed bytes, wrote 39,740,021,248 dense-F32 bytes, and made
zero DMD artifact/range/cache/network traffic. Velocity relative RMSE / cosine were `0.03869645` /
`0.9992708`, and prediction relative RMSE / cosine were `0.042713966` / `0.99910367`. `/dev/shm`
passed quota-aware admission, and process-group exit, Chrome-profile removal, and artifact-server
teardown were clean. This is one-prediction diagnostic stability evidence only: it makes no
calibrated numerical-correctness, full-chain/full-resolution parity, or fully on-device-quantized
execution claim. The earlier core-Q8 first-DMD report is retained only as historical evidence and
does not qualify this storage/application policy. Browser correctness also depends on both root
patches: downstream WebGPU applications must carry equivalent patched `wgpu` and `cubecl-wgpu`
sources because Cargo does not propagate them through published dependencies.

- `fp8`: deliberately unsupported for WebGPU. The official Boogu files require CUDA/Triton
  semantics and are not relabeled as a portable Burn artifact.

The exact internal profile remains part of the manifest and parity report even though it no longer
appears in the public URL.

Release configuration is also part of the identity. The `edit-turbo-1k5` runtime uses
logical model id `Boogu/Boogu-Image-0.1-Edit-Turbo-1K5` and immutable source revision
`60981c49e48cffadf2c169532a4ba3f6108afd5e`. It must be converted and sealed separately from 1K
Edit even where the Hugging Face content store reuses identical payload objects. A 1K manifest is
rejected for 1.5K. Native and browser descriptors now expose the same ten official 1.5K shapes.
Browser shapes at or above 1024 use the exact bounded two-slab VAE tail. The first current modular
Edit runtime-Q8 low-VRAM 1536x1536 computation passed internally but not its stale host contract;
the corrected source-bound canonical rerun now passes. `qualification-f32` is an optional non-blocking control
diagnostic, disabled by default in the release workflow; its last run ended in device loss. It
retains a request-scoped F32 denoiser rather than the ordinary UI's eager all-stage `resident` graph
and cannot substitute for the mandatory low-VRAM numerical and measured-memory gate. Rendered UI,
other shapes, and performance qualification remain separate.

## Components and stages

The converter inventories every tensor before writing anything, then assigns it to one of:

- processor/tokenizer/config;
- Qwen vision embedding, vision block, merger, text embedding, text block, and final norm stages;
- VAE encoder, quant convolution, post-quant convolution, and decoder stages;
- Boogu embedding/refiner, each dual-stream block, each single-stream block, and final projection.

Stage boundaries are semantic. Lexical filename order is never used to decide runtime residency.
The default logical Burnpack object ceiling is 256 MiB (268,435,456 bytes). Production conversion
rejects any tensor that cannot fit that semantic bound; the released Qwen vocabulary tables use
row-sliced stages for this reason. The diagnostic `--allow-oversized-tensors` escape hatch permits
one tensor to occupy an explicitly oversized native-only semantic object, which the browser rejects.

CDN storage is a separate physical layer. Each logical weight object is reconstructed from
content-addressed `transport/<sha256>.part` files declared by the manifest-sealed
`metadata/transport-layout.json` sidecar. Non-final parts are exactly 20,971,520 bytes (20 MiB), the
final part is at most that target, and every physical part has an exact decimal 25,000,000-byte hard
maximum. The browser reads each part as ranges no larger than 4 MiB, so a CacheStorage entry is never
the full 20 MiB part and never exceeds 4 MiB.

## Manifest invariants

A deployment manifest includes:

- schema, bundle/profile/model identity, pinned source and model revisions;
- conversion tool version and complete source-file digests;
- component/stage dependency order and residency hints;
- every tensor name, shape, source dtype, stored dtype, quantization parameters, and target shard;
- every file path, exact byte length, SHA-256 digest, and role;
- sequential per-component shard-chain states and one canonical content digest.

The manifest is canonicalized independently of declaration order. Loading is fail-closed: duplicate,
unknown, missing, corrupt, wrong-sized, wrong-shaped, or incompatible tensors abort before a model
is advertised as ready.

## CDN contract

The canonical public root is `https://aberration.technology/model`. Each child path is an exact
sealed manifest `bundle` id. The production release has five entries:

| Entry | Owns | Schema |
| --- | --- | ---: |
| `qwen3-vl-8b-base-boogu-image-0.1` | shared Qwen text/vision weights, configs, processor, tokenizer, and owner-filtered inventories | 1 |
| `flux1-vae-boogu-image-0.1` | shared FLUX VAE weights, normalized config, and owner-filtered inventories | 1 |
| `boogu-image-0.1-turbo` | Turbo denoiser/composition files plus exact Qwen and VAE dependency pins | 2 |
| `boogu-image-0.1-edit-turbo` | 1K Edit denoiser/composition files plus the same dependency pins | 2 |
| `boogu-image-0.1-edit-turbo-1k5` | 1.5K Edit denoiser/composition files plus the same dependency pins | 2 |

The five-entry upload is public as of 2026-08-16. A real-browser probe authenticated the composed
Turbo manifest, both dependency manifests, all three transport layouts, cold whole-part downloads,
and a warm CacheStorage resume. The remaining publication-policy defect is that reusable
`manifest.json` URLs are marked `immutable`; Pages stays fail-closed until they use `no-cache`.

The Linux browser evidence harnesses separately protect host transport from quota surprises: BigInt
`statfs` accounting is combined with a real bounded 256 MiB write/`fsync`/delete probe. Current runs
admitted `/dev/shm`, rejected quota-limited `/tmp`, and omitted `--disable-dev-shm-usage`; this does
not change artifact identity or relax any digest check.

The three source models share all 111 Qwen Burnpacks (17,442,483,200 bytes) and both VAE
Burnpacks (167,666,432 bytes), including identical upstream source declarations. The VAE configs
differed only by a workstation-local `_name_or_path`, which is removed from the canonical component
config. Denoiser object digests are pairwise disjoint. Publishing Qwen and VAE once removes two
duplicate copies: 35,220,299,264 weight bytes (about 32.80 GiB).

Schema-v2 parent manifests name dependencies by role (`qwen`, `vae`) and seal each dependency's
bundle, profile, model id, model revision, and content digest. A mirror or local cache resolves them
as sibling directories beneath the same model root. Legacy schema-v1 monoliths remain compatible
only when selected explicitly; canonical URLs always use the five-entry graph.

The derived component model revisions are
`020ea5b58bd3fc9abf5f23e92e4039864a6d6ff4993db777a713b489bbd6c5a1` for Qwen and
`5f9271cca82f45ef89910f1a5a4a775745dca788f518d25d93afe5bae9e6b8b8` for VAE. Preparation
recomputes each as SHA-256 over the compact JSON array of owner-filtered source-file declarations,
sorted by path with fields `path`, `size`, `sha256`, followed by one LF byte.

Publish the sealed manifest, every directly stored non-weight payload, the transport-layout
sidecar, and every layout-declared physical part beneath that immutable, single-use bundle prefix.
Logical Burnpack entries retain `objects/<sha256>.bpk` identities in the manifest but are
deliberately absent from the part-only upload tree. The sidecar binds their exact reconstruction to
content-addressed `transport/<sha256>.part` files. Canonical clients fetch physical objects whole;
servers may retain byte-range support for legacy direct Burnpacks. Direct payloads and transport
parts are immutable; manifests are revalidated so publication can commit them last:

```text
Access-Control-Allow-Origin: *
Content-Length: <exact physical-object byte count>
Content-Encoding: identity
Payload Cache-Control: public, max-age=31536000,immutable
Manifest Cache-Control: no-cache
Content-Type: application/octet-stream
Cross-Origin-Resource-Policy: cross-origin
```

`Content-Encoding` may be omitted for an identity response, but an intermediary must never change
the sealed physical bytes. Exposing `Content-Length` and `Content-Encoding` lets the browser reject
bad framing before buffering; exact `Blob.size` and SHA-256 checks remain mandatory when CORS hides
those headers.

Keep converted bundles outside Git, one directory per exact manifest bundle id. The verifier
detects a schema-v2 parent, resolves its sibling Qwen/VAE directories, validates both model-owned
component contracts, authenticates every physical part, reconstructs every logical Burnpack
digest, and checks the complete 223-object / 1,940-tensor semantic closure:

```bash
cargo run --release --locked -p burn_boogu --features import \
  --bin boogu-verify-artifacts -- \
  --artifacts .artifacts/cdn-upload-modular/aberration.technology/model/boogu-image-0.1-turbo \
  --require-published-release
```

For one of the three exact legacy schema-v1 monoliths consumed by native parity binaries, select the
separate `--require-legacy-flat-parity-release` mode. It performs the full single-directory semantic
verification and pins the promotable legacy digest; it rejects dependency-composed artifacts and
cannot be combined with `--require-published-release`. The publication flag instead requires the
canonical schema-v2 parent digest and complete sibling dependency closure. One artifact directory
cannot satisfy both contracts.

Create the five-entry release from the three exact legacy mixed-F16 monoliths with:

```bash
cargo run --release --locked -p burn_boogu --features import \
  --bin boogu-prepare-cdn-release -- \
  --artifact-root .artifacts \
  --output-root .artifacts/cdn-upload-modular
```

The tool requires all three pinned legacy digests and authenticates every legacy Burnpack before
promotion. It proves Qwen/VAE declaration and upstream-source equality, denoiser disjointness, and
that each modular closure reconstructs the old flat tensor/source contract. New owner-filtered
inventories and the normalized VAE config are written as distinct files; no hardlinked source file
is ever mutated. It then emits the sealed 20 MiB-target transport layout, removes direct logical
Burnpacks from the upload tree, and runs the strict semantic-plus-physical verifier for every
parent.

Output lives at
`.artifacts/cdn-upload-modular/aberration.technology/model/<bundle-id>/`. The upload plan is
`.artifacts/cdn-upload-modular/upload-plan.json`; strict reports are under
`.artifacts/cdn-upload-modular/verification-reports/`. The tool refuses an existing output root and
never modifies the legacy sources or `.artifacts/cdn-upload-production`.

This staging tree, converted weights, manifests, reports, and upload plans are generated release
outputs and stay outside Git. A hardlinked staging tree must remain on the artifact filesystem;
create an independent copy only when required.

Upload the physical files in the four phases recorded by the plan:

1. Qwen and VAE direct metadata and transport parts, excluding their manifests.
2. Qwen and VAE manifests after every dependency payload is readable.
3. All three parent direct metadata and transport-part sets, excluding their manifests.
4. Parent manifests only after their dependency manifests and own payloads are readable.

For S3-compatible storage, require an empty immutable destination for every prefix. Payloads use
one-year immutable caching. Manifests are committed last with `no-cache` so an incomplete graph is
never advertised. Do not use a recursive directory upload: a publisher must consume the generated
`upload-plan.json` allowlist, upload only manifest-declared non-weight payloads and
sidecar-declared physical parts, and refuse a declared layout if any logical weight is
materialized. The storage-specific publisher is intentionally separate from the repository; its
recorded operations must preserve the plan's dependency-first, payload-first, manifest-last order.

The destination prefix is part of the release identity and must never be reused, even to resume a
partial failed publication; retry with a fresh prefix. Publish updates under a new release id.
Configure CORS/Range behavior on the bucket or CDN separately using the headers above. The
publishing principal needs `s3:ListBucket` for the destination-prefix preflight and `s3:PutObject`
for uploads; use one publisher per prefix.

`bevy_burn_image` exposes a generic bounded range loader for legacy artifacts. Its canonical browser
reader authenticates the sealed transport sidecar and fetches one physical part at a time as an
HTTP 200 complete object. It validates any exposed canonical `Content-Length`, rejects visible
non-identity `Content-Encoding`, and always checks exact `Blob.size` and SHA-256 before the bounded
copy into Wasm through `Blob.arrayBuffer()`.
The reader then reconstructs at most one bounded semantic object before verifying its complete
SHA-256. Physical parts target 20,971,520 bytes and are hard-capped at 25,000,000 bytes.
CacheStorage uses one independently authenticated entry per physical part; legacy ranges remain no
larger than 4 MiB. Every logical object remains below the 256 MiB semantic
ceiling, while `manifest.json` and the transport sidecar each remain below their 4 MiB bootstrap
caps. The contract forbids an unbounded response or complete bundle from becoming one
`ArrayBuffer`.
Asynchronous Qwen, VAE, and denoiser stage sources then apply that verified object, synchronize
execution, and release it before fetching the next stage. The native directory sources remain
synchronous and are not compiled into the browser factory.

The native viewer uses the same canonical sibling CDN bases when `--artifacts` is omitted. It caches
the Qwen, VAE, and selected parent independently, so another pipeline or downstream component-only
application reuses already verified shared weights. Each manifest is the cache commit point only
after all of its declared payloads pass exact-size and SHA-256 verification. Completed entries live
at `~/.burn_image/models/<parent-bundle-id>/` for the selected parent and
`~/.burn_image/models/<component-bundle-id>/<content-digest>/` for shared children, following the
sibling `burn_jepa` per-user model-cache convention. Noncanonical/custom remote sources use a
distinct legacy tuple cache key and cannot
silently replace canonical entries. `BURN_IMAGE_CACHE_DIR` changes the broad cache root;
`BURN_IMAGE_MODEL_CACHE_DIR` chooses an exact custom-source directory, and
`BURN_IMAGE_MODEL_BASE_URL` selects a custom CDN.
`BURN_IMAGE_MODEL_MANIFEST_URL` overrides only the exact manifest request; declared payload paths
still resolve from `BURN_IMAGE_MODEL_BASE_URL` or the selected source base.

## Conversion guarantees

Semantic Burnpack creation and physical-part staging are fail-closed, but a complete conversion is
intentionally not restartable in place. Remote discovery is committed only when the manifest is
uploaded last; the preceding physical uploads are not described as an atomic publication.
The importer refuses an existing destination, so an interrupted output directory must be inspected
and moved or removed before retrying with a fresh destination. During one conversion, each temporary
shard is hashed and validated before it is renamed into the output tree, and the sealed manifest is
written last. A content-addressed object already produced earlier in that same run is reused only
when its size and digest match; conflicting paths are rejected rather than overwritten.

Source downloads are pinned with `huggingface-cli`/`hf download --revision <commit>`. Mutable
branches are accepted only as a discovery input; their resolved commit is what enters the manifest.

## Pre-transport semantic baseline

The following 2026-08-13 results are the logical semantic baseline that preceded the sealed
part-only transport layout. They authenticated all five entries and all three composed closures,
but their direct 256 MiB-class Burnpack upload shape is not the current browser-cache-friendly CDN
contract and does not establish public deployment. The regenerated transport release must publish
new manifest/report digests and physical-part counts before promotion.

| Canonical entry | Content digest | Files / Burnpacks | Declared payload bytes |
| --- | --- | ---: | ---: |
| `qwen3-vl-8b-base-boogu-image-0.1` | `2f7ed91e09f208853b189ee8c3d6db74a02d2512e07f4818f6688131359d98fc` | 132 / 111 | 17,470,596,755 |
| `flux1-vae-boogu-image-0.1` | `8ff1043ac3d47e6addbb5e07f437c04585f678819ffd0e505ac46effdf1c31d6` | 5 / 2 | 167,875,849 |
| `boogu-image-0.1-turbo` | `555019af867a80bb4d7cec5dc2f0ba60ae799071994a5fd24d7e71918cb9ce36` | 116 / 110 | 20,586,250,890 |
| `boogu-image-0.1-edit-turbo` | `28b1b51f2fb152557b11a9f0ef8e872ae7d163bcab7abd42f9eaf4bfef10e7aa` | 116 / 110 | 20,586,251,131 |
| `boogu-image-0.1-edit-turbo-1k5` | `4eb95001708becebeab5bb7417b02003e9dbe704775bb49557b681a5b617fd5a` | 116 / 110 | 20,586,251,131 |

The five prefixes declare 79,397,225,756 payload bytes, including compact metadata. The three old
flat monoliths declared 114,674,170,993 bytes. The conservative equality proof separately records
35,220,299,264 eliminated duplicate *weight* bytes; it does not label metadata savings as weight
savings.

| Composed verification | Verified bytes | Largest object | Report SHA-256 |
| --- | ---: | ---: | --- |
| Turbo | 38,224,723,494 | 254,254,592 | `d3782a920fdca9f93da812bf2734d768a2a8cb09158ac9493efad293db30888c` |
| Edit Turbo | 38,224,723,735 | 254,254,592 | `566a39320e65bb7071e5c995c25c082409f9c437d68a6d698037802fe984be45` |
| Edit Turbo 1.5K | 38,224,723,735 | 254,254,592 | `4fbe36c124a05faf8f824ccec170b95258610f94198d29470ce0e6739cffbd4d` |

The modular-equivalence report SHA-256 is
`1592b5ddb5acf60bdec261d7953b3dc967aaaefb83d6358b9d263896002d9d28`.
The upload plan SHA-256 is
`c8cce1fcdc46b500da03dad1570b1b269a4318039abda6e48ee76bf4a2e8439f`.
The current-source native 1536x1536 full-chain pass deliberately uses the separate qualified
schema-v1 flat parity root required by the native stage readers; it measured 30,310,137,856 bytes
and produced report SHA-256
`a013adfcd30b7e6b2323ecc3723b22396f9858d14dd3cdd4a0da2699e199abe3`. Canonical publication and
browser qualification instead use the schema-v2 modular root and its sealed sibling closure. These
roots are independently verified and are not aliases.
The ordinary browser UI now accepts the 1.5K release and its ten official shapes. Historical
no-surface browser parity authenticated the legacy schema-v1 monolith. The first current-source
modular low-VRAM 1536x1536 attempt authenticated the schema-v2 1.5K parent digest above plus its
sealed Qwen and VAE dependencies, verified all 223 executed weight objects, and passed the inner
full-chain numerical and memory gates. Its immutable outer report SHA-256 is
`3dec48ec032c7abd1ffcb9aab1546d81765e97503340f8ea895724dfe1aacd5b`; `ok=false` reflects only the
stale host expectation for retired resource-plan field `audited_max_streamed_stage_bytes`, and its
clean corrected-validator replay is noncanonical. The subsequent source-bound rerun passed with
top-level `ok=true`, 443/443 matched GPU intervals, peak 29,828 MiB / 31,276,924,928 bytes, and
report SHA-256 `c895ae2c1cba3823afe756035b6e564d5ef27caf3722f5f350c07e23086e3b54`. The explicit
`qualification-f32` control remains optional and lacks a passing result;
ordinary all-stage-resident UI, other 1.5K shapes, rendered 1.5K, and browser performance still
require their own gates. Ordinary rendered Turbo 1024 now has its separate positive output/memory
gate.

Generated bundles stay in the ignored `.artifacts/` cache; only compact identities and evidence
belong in Git. Re-run the same modular gate before publication:

```bash
cargo run --release -p burn_boogu --features import \
  --bin boogu-verify-artifacts -- \
  --artifacts /path/to/model/boogu-image-0.1-turbo \
  --require-published-release
```
