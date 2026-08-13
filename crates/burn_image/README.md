# burn_image

Model-neutral request, artifact, progress, and runtime contracts for image generation and image
editing.

`burn_image` deliberately does not implement a model or select a tensor backend. Model crates
implement the `ImageModel` trait and may return any output type, including a device-resident tensor.
This keeps request validation, artifact integrity, runtime routing, and provenance consistent across
native and web runtimes without coupling the crate to a particular model family.

## API shape

```rust
use burn_image::{
    Dimensions, GenerateRequest, GenerationOptions, ImageRequest, Prompt,
};

let request = ImageRequest::Generate(GenerateRequest {
    prompt: Prompt::new("a glass greenhouse in a redwood forest")?,
    negative_prompt: None,
    options: GenerationOptions {
        dimensions: Some(Dimensions::new(1024, 1024)?),
        steps: Some(8),
        guidance_scale: Some(3.5),
        seed: Some(42),
        batch_size: 1,
    },
});

request.validate()?;
# Ok::<(), burn_image::ImageError>(())
```

Model implementations expose a `ModelDescriptor` and implement `ImageModel`. `ImageRuntime` applies
the descriptor's capability constraints before dispatching the request and provides structured
progress and cancellation.

## Artifact integrity

Artifact manifests use validated relative paths, explicit byte lengths, SHA-256 digests,
component-aware shard metadata, ordered shard hash chains, and a deterministic bundle content
digest. `ArtifactVerifier` supports bounded sequential range verification so a web loader does not
need to retain a complete shard in Wasm linear memory. `BundleVerifier` accepts completed files in
any arrival order but yields a bundle only after every sealed-manifest file has been verified.

Schema-v2 manifests can pin immutable sibling component bundles by role, identity, revision, and
sealed digest. `ArtifactShardReader` and `AsyncArtifactShardReader` provide model-neutral bounded
object reads, while `FilesystemArtifactCache` streams native downloads into a digest-keyed cache
and writes `manifest.json` only after every payload verifies. Model crates own their tensor/stage
semantics on top of these APIs; applications provide HTTP or browser transport adapters.

Integrity verification is strict by default. A loader must not make unverified artifacts equivalent
to verified artifacts in logs or provenance.

## Scope

This crate owns:

- portable generation and edit request types;
- host image, mask, output, timing, and provenance types;
- model descriptors and capability validation;
- artifact manifests, sources, shard validation, and streaming integrity checks;
- model registry, progress, cancellation, and generic runtime dispatch.

It does not own:

- model architecture or weights;
- tokenization or prompt templates for a specific model;
- Burn tensor/backend selection;
- protocol-specific HTTP or browser transport implementations;
- Bevy integration.

See the repository root documentation for model implementations, setup, parity reports, and web
deployment.
