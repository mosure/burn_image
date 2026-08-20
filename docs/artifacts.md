# Artifacts and CDN layout

Model artifacts are immutable, sealed, and split into semantic and physical layers.

## Logical bundles

A bundle manifest binds:

- model, revision, variant, storage profile, and numeric format;
- every config, tokenizer, template, inventory, and logical weight object;
- byte size, role, and SHA-256 digest for every file;
- tensor name, shape, dtype, quantization metadata, and semantic stage;
- component dependencies and their exact content digests;
- conversion identity and the manifest's own sealed content digest.

Component manifests describe Qwen and VAE releases. A composed Boogu manifest pins those components
and adds the denoiser/configuration payload. The runtime verifies the complete dependency closure.
The Q4 release is exactly five bundles: one shared packed-Q4 Qwen component, one shared F16 VAE
component, and three denoiser-only parents. Qwen and VAE payloads are therefore stored and uploaded
once, not copied into every Generate/Edit variant.

## Physical transport

Logical weight objects are Burnpacks, but a browser/CDN release does not publish those `.bpk` files
directly. Each bundle contains a sealed `metadata/transport-layout.json` mapping every logical weight
to ordered immutable `transport/<sha256>.part` files.

The physical contract is:

| property | value |
|---|---:|
| deterministic target part size | 20,971,520 bytes |
| hard maximum published object size | 25,000,000 bytes |
| maximum layout sidecar size | 4 MiB |
| logical weight files in published tree | none |

Parts never have gaps or overlaps. Offsets must begin at zero and cover the exact logical length.
Part paths are content-addressed; reuse is allowed only when size and digest are identical. After
part verification, the reader reconstructs and verifies the logical Burnpack digest before parsing
tensors.

Compact manifest-declared files are published directly. They remain subject to the same
25,000,000-byte physical ceiling and exact size/digest verification.

## Canonical local tree

Release generation writes:

```text
.artifacts/cdn-upload-modular/
└── aberration.technology/model/
    ├── boogu-image-0.1-turbo-.../
    ├── boogu-image-0.1-edit-turbo-.../
    ├── boogu-image-0.1-edit-turbo-1k5-.../
    ├── qwen-.../
    └── flux-vae-.../
```

Each bundle directory contains `manifest.json`, compact metadata, a transport layout, and physical
parts. Generated release trees and large payloads are not committed.

## Build a release

Source checkpoints and revisions must already be pinned and imported into `.artifacts`.

```sh
cargo run --release --locked -p burn_boogu --features import \
  --bin boogu-prepare-cdn-release -- \
  --artifact-root .artifacts \
  --output-root .artifacts/cdn-upload-q4s-complete \
  --q4-only
```

Generation is fail-closed: it verifies source manifests, creates deterministic parts, seals the
layout, reconstructs logical objects, verifies component dependencies, and writes the final tree
only after all checks pass.

## Verify a bundle

Full verification hashes physical parts and reconstructed logical objects:

```sh
cargo run --release --locked -p burn_boogu --features import \
  --bin boogu-verify-artifacts -- \
  --artifacts .artifacts/cdn-upload-modular/aberration.technology/model/BUNDLE
```

Manifest/layout inspection is available without reading every payload:

```sh
cargo run --release --locked -p burn_boogu --features import \
  --bin boogu-verify-artifacts -- \
  --manifest-only BUNDLE/manifest.json \
  --transport-layout BUNDLE/metadata/transport-layout.json
```

That narrower command proves structure and sealed metadata, not payload bytes.

## Publication order

Upload immutable payloads first, component manifests next, and composed manifests last. A manifest
must never become visible before every object it names is readable with the required HTTP contract.
Deployment rechecks size, SHA-256, CORS, Range behavior, dependency pins, and reconstructed logical
digests.

Payload URLs use long-lived immutable caching. A cached manifest remains safe because it is sealed
and points only to immutable content-addressed payloads; deployment emits a warning, rather than a
release failure, when a host does not use `no-cache` for manifests.

## Runtime caches

Native execution stores verified bundles under `~/.burn_image`. Browser execution uses Cache
Storage for authenticated physical parts. Neither cache trusts presence alone: size, digest, layout,
and manifest identity remain mandatory.
