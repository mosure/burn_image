# Engineering contract

## Ownership

- `burn_image` owns model-neutral requests, outputs, jobs, artifacts, and provenance.
- `burn_qwen3_vl` owns ordinary Qwen3-VL architecture, processing, semantic artifact stages,
  verified shard loading, and device-resident stage retention.
- `burn_flux_vae` owns ordinary FLUX-compatible `AutoencoderKL` math, semantic artifact stages,
  verified shard loading, and device-resident stage retention.
- `burn_boogu` owns Boogu conditioning, denoiser, DMD sampling, and composition.
- `bevy_burn_image` owns ECS, UI, device sharing, display, and platform I/O only.

Dependencies point from model/application crates toward reusable crates. Reusable crates must not
contain Boogu prompts, schedules, size policy, CDN URLs, or Bevy types.

## Correctness

- Tests use `_smoke`, `_correctness`, `_parity`, and `_reference` suffixes.
- A missing opt-in fixture may skip a local test but must never be reported as parity success.
- Release parity jobs require pinned real checkpoints and fail when artifacts are absent.
- Compare prompt processing, image processing, Qwen features, VAE moments/samples, RoPE,
  denoiser hooks, every DMD step, decoded tensors, and final pixels.
- Seeds alone are not cross-runtime evidence. Reference tests inject exact noise tensors.

## GPU and Web

- WGPU/WebGPU modes fail clearly rather than silently falling back to CPU.
- The Bevy frontend and Burn share one adapter, device, and queue.
- Production attention must not allocate a dense sequence-by-sequence score tensor.
- An on-device quantized execution claim must remain quantized and use a measured kernel.
  Storage-compressed profiles may dequantize one verified bounded stage when a backend mapper or
  kernel is incorrect, but the manifest/runtime report and documentation must name that policy;
  they must not be described as fully on-device quantized execution.
- Browser loaders verify and apply bounded shards sequentially; no full bundle may enter Wasm
  linear memory.

## Artifacts

- Pin immutable upstream source and Hugging Face revisions.
- Manifests include every tensor, config, tokenizer/template, size, dtype, quantization scheme,
  conversion version, and SHA-256 digest.
- Reject duplicate, unknown, missing, or shape-incompatible tensors.
- Represent Burnpack weights as immutable content-addressed logical objects, but publish canonical
  browser/CDN payloads only as sealed immutable content-addressed transport parts; direct logical
  Burnpack files must be absent from a part-only tree. Compact configs, tokenizers, templates, and
  inventories may retain semantic `metadata/...` paths only inside a single-use immutable bundle
  prefix and must be exact-size/SHA-256-bound by the sealed manifest. Commit only compact
  manifests/evidence.

## Validation

Before a supported claim, run formatting, Clippy with warnings denied, unit/integration tests,
Wasm compilation, native WGPU real-artifact parity, browser WebGPU real-artifact parity, artifact
integrity tests, and synchronized performance measurements.
