# Artifacts

Burn Image artifacts are deployment objects, not opaque checkpoint archives. Conversion preserves
enough information to prove exactly what was loaded and to stream only the component needed next.

## Production storage policy

The CDN publishes one production policy per model. User-facing tools call it `production`; its
precise sealed manifest profile remains `f16-qwen-vision-f32` for provenance and compatibility.

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
non-quantized parameters and all DMD activations run F32. Both F16 profiles preserve F16 denoiser
execution. The parity-qualified native high-VRAM `f16-qwen-vision-f32` route for Turbo and 1K Edit
preserves F16 VAE weights/activations and uses F16 storage with F32 GroupNorm reduction scalars.
Browser, layer-streamed, Q8, and all-F16 routes retain the strict `force_upcast=true` F32 VAE
policy. The separately gated 1.5K release uses the same typed VAE precision contract with its own
larger attention bounds; neither policy is a profile-wide default.
- `fp8`: deliberately unsupported for WebGPU. The official Boogu files require CUDA/Triton
  semantics and are not relabeled as a portable Burn artifact.

The exact internal profile remains part of the manifest and parity report even though it no longer
appears in the public URL.

Release configuration is also part of the identity. The native `edit-turbo-1k5` runtime uses
logical model id `Boogu/Boogu-Image-0.1-Edit-Turbo-1K5` and immutable source revision
`60981c49e48cffadf2c169532a4ba3f6108afd5e`. It must be converted and sealed separately from 1K
Edit even where the Hugging Face content store reuses identical payload objects. A 1K manifest is
rejected for 1.5K. Ordinary browser construction still rejects 1.5K; the separate surface-free
qualification route uses an exact bounded two-slab VAE tail and remains unsupported until its
WebGPU numerical/performance gates pass.

## Components and stages

The converter inventories every tensor before writing anything, then assigns it to one of:

- processor/tokenizer/config;
- Qwen vision embedding, vision block, merger, text embedding, text block, and final norm stages;
- VAE encoder, quant convolution, post-quant convolution, and decoder stages;
- Boogu embedding/refiner, each dual-stream block, each single-stream block, and final projection.

Stage boundaries are semantic. Lexical filename order is never used to decide runtime residency.
The default physical shard target is 256 MiB. Production conversion rejects any tensor that cannot
fit that bound; the released Qwen vocabulary tables use row-sliced stages for this reason. The
diagnostic `--allow-oversized-tensors` escape hatch permits one tensor to occupy an explicitly
oversized native-only shard, which the browser's 256 MiB semantic-object limit will reject.

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

Publish the sealed manifest and every declared payload beneath that immutable, single-use bundle
prefix. Burnpack weight objects use their SHA-256 digest as the filename; the sealed manifest binds
metadata paths to their exact sizes and SHA-256 digests. Servers should support byte ranges.
Payloads are immutable; manifests are revalidated so publication can commit them last:

```text
Access-Control-Allow-Origin: *
Access-Control-Expose-Headers: Content-Range
Accept-Ranges: bytes
Payload Cache-Control: public, max-age=31536000,immutable
Manifest Cache-Control: no-cache
Content-Type: application/octet-stream
Cross-Origin-Resource-Policy: cross-origin
```

Keep converted bundles outside Git, one directory per exact manifest bundle id. The verifier
detects a schema-v2 parent, resolves its sibling Qwen/VAE directories, validates both model-owned
component contracts, and then checks all 223 Burnpacks and 1,940 stored tensor entries:

```bash
cargo run --release --locked -p burn_boogu --features import \
  --bin boogu-verify-artifacts -- \
  --artifacts .artifacts/cdn-upload-modular/aberration.technology/model/boogu-image-0.1-turbo \
  --require-published-release
```

For a legacy schema-v1 monolith, the same CLI retains its single-directory verification behavior.
It refuses diagnostics, dependency drift, self-sealed but unpinned digests, converter-version
drift, unknown/missing tensors, and semantic verification failures.

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
is ever mutated. It then runs the strict modular semantic verifier for every parent.

Output lives at
`.artifacts/cdn-upload-modular/aberration.technology/model/<bundle-id>/`. The upload plan is
`.artifacts/cdn-upload-modular/upload-plan.json`; strict reports are under
`.artifacts/cdn-upload-modular/verification-reports/`. The tool refuses an existing output root and
never modifies the legacy sources or `.artifacts/cdn-upload-production`.

This staging tree, converted weights, manifests, reports, and upload plans are generated release
outputs and stay outside Git. A hardlinked staging tree must remain on the artifact filesystem;
create an independent copy only when required.

Upload in the four phases recorded by the plan:

1. Qwen and VAE payloads, excluding their manifests.
2. Qwen and VAE manifests after every dependency payload is readable.
3. All three parent payload sets, excluding their manifests.
4. Parent manifests only after their dependency manifests and own payloads are readable.

For S3-compatible storage, require an empty immutable destination for every prefix. Payloads use
one-year immutable caching. Manifests are committed last with `no-cache` so an incomplete graph is
never advertised. For example, after publishing both dependencies, a parent upload is:

```bash
export BUNDLE=boogu-image-0.1-turbo
export ARTIFACT_DIR=".artifacts/cdn-upload-modular/aberration.technology/model/$BUNDLE"
export DESTINATION="s3://example-models/model/$BUNDLE"
cargo run --release --locked -p burn_boogu --features import \
  --bin boogu-verify-artifacts -- \
  --artifacts "$ARTIFACT_DIR" --require-published-release
test "$(aws s3api list-objects-v2 \
  --bucket example-models --prefix "model/$BUNDLE/" --max-keys 1 \
  --query KeyCount --output text)" = 0
aws s3 sync "$ARTIFACT_DIR/" "$DESTINATION/" \
  --exclude manifest.json --cache-control public,max-age=31536000,immutable
aws s3 cp "$ARTIFACT_DIR/manifest.json" "$DESTINATION/manifest.json" \
  --content-type application/json \
  --cache-control no-cache
```

The destination prefix is part of the release identity and must never be reused, even to resume a
partial failed publication; retry with a fresh prefix. Publish updates under a new release id.
Configure CORS/Range behavior on the bucket or CDN separately using the headers above. The
publishing principal needs `s3:ListBucket` for the destination-prefix preflight and `s3:PutObject`
for uploads; use one publisher per prefix.

`bevy_burn_image` exposes a generic bounded range loader that incrementally verifies SHA-256 and
commits through a transactional sink. Its concrete browser model reader performs HTTP 206 range
fetches, validates each `Content-Range`, and aggregates at most one bounded semantic object before
verifying its complete SHA-256. The contract forbids retaining a multi-gigabyte response as one
`ArrayBuffer`. Every released object is below the 256 MiB semantic-object ceiling; transport uses
4 MiB ranges with a 16 MiB hard response cap, and `manifest.json` is below its 4 MiB bootstrap cap.
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

Shard publication is atomic, but a complete conversion is intentionally not restartable in place.
The importer refuses an existing destination, so an interrupted output directory must be inspected
and moved or removed before retrying with a fresh destination. During one conversion, each temporary
shard is hashed and validated before it is renamed into the output tree, and the sealed manifest is
written last. A content-addressed object already produced earlier in that same run is reused only
when its size and digest match; conflicting paths are rejected rather than overwritten.

Source downloads are pinned with `huggingface-cli`/`hf download --revision <commit>`. Mutable
branches are accepted only as a discovery input; their resolved commit is what enters the manifest.

## Verified production bundles

The 2026-08-13 preparation gate authenticated all five sealed entries and all three composed
closures. Each closure contains 253 files, 223 Burnpacks, and 1,940 stored tensor entries. Qwen and
VAE manifests are schema v1 and dependency-free; parents are schema v2 and pin both complete child
identities.

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
The ordinary browser UI still rejects 1.5K. Historical no-surface browser parity authenticated the
legacy schema-v1 monolith; the new modular 1.5K closure must be rerun before that evidence can be
claimed for the five-entry release.

Generated bundles stay in the ignored `.artifacts/` cache; only compact identities and evidence
belong in Git. Re-run the same modular gate before publication:

```bash
cargo run --release -p burn_boogu --features import \
  --bin boogu-verify-artifacts -- \
  --artifacts /path/to/model/boogu-image-0.1-turbo \
  --require-published-release
```
