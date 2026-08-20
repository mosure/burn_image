# Correctness and parity

Compilation and a rendered window are not model qualification. A supported result requires exact
artifact identity, real GPU execution, complete numerical checkpoints, and final output evidence.

## Test layers

### Unit and contract tests

These cover shapes, schedules, quantization layouts, manifest sealing, transport reconstruction,
cache behavior, request validation, and UI/runtime state transitions. Test names use `_smoke`,
`_correctness`, `_parity`, or `_reference` suffixes.

### Reference tests

CPU/reference tests compare individual modules against pinned tensors. They cover:

- prompt and image preprocessing;
- Qwen stages and RoPE;
- VAE moments, injected samples, and decode;
- denoiser hooks and each DMD step;
- packed linear, convolution, and embedding kernels.

### Native WGPU parity

Native qualification uses the ordinary released runtime and sealed artifacts. It must prove that
the selected GPU backend executed, that no CPU fallback occurred, and that all requested module and
pixel comparisons passed.

### Browser WebGPU parity

Browser qualification runs the deployed Bevy surface and Burn runtime on one adapter/device/queue.
It additionally validates HTTP transport, persistent cache behavior, Wasm memory bounds, surface
suspension during vulnerable inference windows, GPU errors, and cleanup.

## Evidence requirements

Every qualification report binds:

- source revision and package identity;
- model, revision, profile, manifest digest, and dependency closure;
- backend, adapter, device, queue, limits, and enabled features;
- exact fixture identity and injected noise tensors;
- every required intermediate comparison;
- output dimensions, encoding, and digest;
- timing, memory, transport, and cleanup counters;
- an explicit claim scope.

A missing fixture skips a local opt-in test; it never becomes a pass. A seed is useful for repeatable
execution but is not cross-runtime evidence without exact injected noise.

## Numerical comparisons

Thresholds are defined per tensor or module according to the selected numeric profile. Reports use
metrics such as maximum absolute error, relative RMSE, cosine similarity, PSNR, and SSIM. Packed-Q4
acceptance is based on module-level error plus perceptual output checks; it is not inferred from a
single end-to-end image.

## Core validation

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --locked
cargo doc --workspace --all-features --no-deps --locked

cargo check -p bevy_burn_image --target wasm32-unknown-unknown \
  --no-default-features --features boogu-web --lib --locked
node --test crates/bevy_image/tests/*.test.mjs
```

Hardware jobs are opt-in and require pinned fixture/artifact roots. The CI and parity workflows are
the source of truth for their environment variables and exact commands.

## Claim boundaries

- Artifact validation does not prove model math.
- A module probe does not prove end-to-end output parity.
- A smoke run does not prove numerical parity.
- Storage compression does not prove on-device quantized execution.
- A successful first request does not prove warm-cache behavior.

Public documentation should state only the strongest layer that has passed on the exact source,
package, artifacts, and hardware under discussion.
