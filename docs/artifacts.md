# Artifacts

Burn Image artifacts are deployment objects, not opaque checkpoint archives. Conversion preserves
enough information to prove exactly what was loaded and to stream only the component needed next.

## Profiles

- `f16-qwen-vision-f32`: parity-oriented default. Qwen vision tensors stay F32; other BF16 source
  tensors are deterministically rounded to IEEE F16.
- `q8s-block32-f32-qwen-vision-f32`: Qwen vision tensors stay F32; eligible non-vision aligned
  rank-two linear weights use symmetric Q8 with 32-value blocks and F32 scales; embeddings, norms,
  biases, VAE convolutions, and activations remain F16.
- `f16` and `q8s-block32-f32`: diagnostic alternatives without the F32 Qwen vision exception.

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

Each profile has a distinct bundle identity and parity report.

Release configuration is also part of the identity. The native `edit-turbo-1k5` runtime uses
logical model id `Boogu/Boogu-Image-0.1-Edit-Turbo-1K5` and immutable source revision
`60981c49e48cffadf2c169532a4ba3f6108afd5e`. It must be converted and sealed separately from 1K
Edit even where the Hugging Face content store reuses identical payload objects. A 1K manifest is
rejected for 1.5K, and browser construction rejects 1.5K before artifact loading until its WebGPU
numerical/performance gates exist.

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

The canonical public root is `https://aberration.technology/model`. Its single `{model_name}` path
segment is the exact sealed manifest `bundle` id, including variant and storage profile. For
example, Turbo mixed F16 resolves from:

```text
https://aberration.technology/model/boogu-image-0.1-turbo-f16-qwen-vision-f32/manifest.json
```

Publish the sealed manifest and every declared payload beneath that immutable, single-use bundle
prefix. Burnpack weight objects use their SHA-256 digest as the filename; the sealed manifest binds
metadata paths to their exact sizes and SHA-256 digests. Servers should support byte ranges and
immutable caching:

```text
Access-Control-Allow-Origin: *
Access-Control-Expose-Headers: Content-Range
Accept-Ranges: bytes
Cache-Control: public, max-age=31536000, immutable
Content-Type: application/octet-stream
Cross-Origin-Resource-Policy: cross-origin
```

Keep converted bundles outside Git, one directory per exact manifest bundle id. Before upload,
verify each directory against the immutable published-release contract with the Rust verifier:

```bash
cargo run --release --locked -p burn_boogu --features import \
  --bin boogu-verify-artifacts -- \
  --artifacts .artifacts/boogu-image-0.1-turbo-f16-qwen-vision-f32 \
  --require-published-release
```

The verifier refuses diagnostics, basename/manifest identity mismatches, self-sealed but unpinned
digests, converter-version drift, and semantic verification failures. Release staging writes
bundle directories to
`.artifacts/cdn-upload/aberration.technology/model/<bundle-id>/` and the compact, non-payload
inventory to `.artifacts/cdn-upload/upload-plan.json`. Full strict-verifier JSON reports are written
outside the upload tree under `.artifacts/cdn-upload/verification-reports/<bundle-id>.json` and are
also embedded in the plan. This staging tree, converted weights, shards, manifests produced during
conversion, verification reports, and upload plans are generated release outputs and stay outside
Git. A hardlinked staging tree must remain on the artifact filesystem; create an independent copy
only when required.

For S3-compatible storage, verify again, require an empty immutable destination, upload payloads
with immutable caching, and commit `manifest.json` last. The destination must end in
`/model/<exact-bundle-id>`, matching the public Aberration URL:

```bash
export BUNDLE=boogu-image-0.1-turbo-f16-qwen-vision-f32
export ARTIFACT_DIR=".artifacts/cdn-upload/aberration.technology/model/$BUNDLE"
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
  --cache-control public,max-age=31536000,immutable
```

The destination prefix is part of the release identity and must never be reused, even to resume a
partial failed publication; retry with a fresh prefix. Both payloads and the manifest use immutable
one-year caching, so publish updates at a new release prefix. Configure CORS/Range behavior on the
bucket or CDN separately using the headers above. The publishing principal needs `s3:ListBucket`
for the destination-prefix preflight and `s3:PutObject` for uploads; use one publisher per prefix.

`bevy_burn_image` exposes a generic bounded range loader that incrementally verifies SHA-256 and
commits through a transactional sink. Its concrete browser model reader performs HTTP 206 range
fetches, validates each `Content-Range`, and aggregates at most one bounded semantic object before
verifying its complete SHA-256. The contract forbids retaining a multi-gigabyte response as one
`ArrayBuffer`. Every released object is below the 256 MiB semantic-object ceiling; transport uses
4 MiB ranges with a 16 MiB hard response cap, and `manifest.json` is below its 4 MiB bootstrap cap.
Asynchronous Qwen, VAE, and denoiser stage sources then apply that verified object, synchronize
execution, and release it before fetching the next stage. The native directory sources remain
synchronous and are not compiled into the browser factory.

The native viewer uses the same canonical per-bundle CDN bases when `--artifacts` is omitted. It
downloads into a temporary verified path and treats `manifest.json` as the cache commit point only
after every declared payload passes exact-size and SHA-256 verification. Completed bundles live at
`~/.burn_image/models/<exact-manifest-bundle-id>/`, following the sibling `burn_jepa` per-user
model-cache convention. `BURN_IMAGE_CACHE_DIR` changes the broad cache root;
`BURN_IMAGE_MODEL_CACHE_DIR` and `BURN_IMAGE_MODEL_BASE_URL` are explicit exact-directory/custom-CDN
overrides. `BURN_IMAGE_MODEL_MANIFEST_URL` overrides only the exact manifest request; declared
payload paths still resolve from `BURN_IMAGE_MODEL_BASE_URL` or the selected source base.

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

`boogu-verify-artifacts` was run over every byte of the four original production bundles and the
native Edit-Turbo 1.5K mixed-F16 release after the strict semantic verifier landed. All five runs
completed with empty stderr and proved the
canonical release identity, source/config graph, exact 1,940-tensor compiled inventory, object
bounds, Burnpack tensor contracts, and stored payload digests.

| Variant/profile | Content digest | Files / Burnpacks | Verified bytes | Largest object | Report SHA-256 |
| --- | --- | ---: | ---: | ---: | --- |
| Turbo mixed F16 | `4f94cf68c00af12d5de486db4d316ce889d6d21e78913a1c74edab4bd0119ce3` | 249 / 223 | 38,224,723,551 | 254,254,592 | `0c63e240f6d590be3ae8c9cd6ecb6ef0825a204b3b97de1ad6293a22dec424f3` |
| Edit 1K mixed F16 | `14acbafd13dc9b79757e7d554b504396bee30ea7ed231f533919c6c82a6e6a32` | 249 / 223 | 38,224,723,721 | 254,254,592 | `0ab13d00dad78cb7f45c5910fcc75272e11388b5de754f917e5612971b3beb43` |
| Edit 1.5K mixed F16 | `4e8b12ac5ca95272f9009080a23baf1bc52d1b0e7aebf2e9e5f394a492369213` | 249 / 223 | 38,224,723,721 | 254,254,592 | `b702ef29e4a9e26afc67d2fdf8ad46582f561995d6041763439a6ed7ef3ee64f` |
| Turbo hybrid Q8 | `8685559e73cf836e98e1ebdf80815e3d66765f7d620624408148d5f98c87c0dd` | 159 / 133 | 23,449,152,358 | 255,513,344 | `2662b49b2f1e7a8710b95a613b90ce13c0ef3ca47fe16ed533dbec110b93d9d5` |
| Edit 1K hybrid Q8 | `ffde989bb66df3a541d44957422f996790633dab46ca3547a59dfdfb871f0b7a` | 159 / 133 | 23,449,152,528 | 255,513,344 | `61a8b14a7b398cf0f87e8f6b394fd28ce970175a1589cb130a92888de1083d55` |

Their canonical model-name segments are, respectively,
`boogu-image-0.1-turbo-f16-qwen-vision-f32`,
`boogu-image-0.1-edit-turbo-f16-qwen-vision-f32`,
`boogu-image-0.1-edit-turbo-1k5-f16-qwen-vision-f32`,
`boogu-image-0.1-turbo-q8s-block32-f32-qwen-vision-f32`, and
`boogu-image-0.1-edit-turbo-q8s-block32-f32-qwen-vision-f32`. The 1.5K bundle is staged and cached
for native WGPU; browser construction still rejects that variant until browser parity is qualified.

The converted bundles stay in the ignored local `.artifacts/` cache; only compact identities and
evidence belong in Git. Re-run the same gate before publication:

```bash
cargo run --release -p burn_boogu --features import \
  --bin boogu-verify-artifacts -- \
  --artifacts /path/to/sealed-bundle --require-published-release
```
