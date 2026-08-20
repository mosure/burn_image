# burn_image

Model-neutral contracts for image generation and editing runtimes.

This crate owns the stable boundary between applications and model implementations. It deliberately
does not depend on Burn tensor backends or on a concrete image model.

## Responsibilities

- requests, model capabilities, jobs, progress, and cancellation;
- device-resident output abstractions and encoded image results;
- model, artifact-profile, and numeric-format identities;
- sealed artifact manifests and SHA-256 verification;
- local directories, remote sources, cache policy, and provenance;
- part-only CDN transport layouts and logical Burnpack reconstruction.

Model crates implement these contracts. UI and platform crates consume them without adding
model-specific policy to this crate.

## Artifact transport

A manifest describes logical files and their digests. CDN weight payloads are physical
`transport/<sha256>.part` files declared by a sealed `metadata/transport-layout.json` sidecar.

- target part size: 20,971,520 bytes;
- hard physical-object limit: 25,000,000 bytes;
- logical weights are absent from the published part-only tree;
- every part and reconstructed logical file is verified before use;
- direct config, tokenizer, and metadata files are also size- and digest-bound.

The verified reader type is the trust boundary: callers cannot obtain artifact bytes through the
strict API until manifest, layout, path, size, and digest checks succeed.

## Feature policy

`burn_image` has no backend features. Backend selection belongs to the model runtime or application.

## Validation

```sh
cargo test -p burn_image --locked
cargo clippy -p burn_image --all-targets --all-features --locked -- -D warnings
```

See the workspace [artifact contract](../../docs/artifacts.md) and
[architecture](../../docs/architecture.md).
