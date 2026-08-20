# burn_boogu

Boogu Image generation and editing pipelines implemented in Burn.

## Models

| variant | task | ordinary resolution | ordinary profile |
|---|---|---:|---|
| `Image01Turbo` | text-to-image | 1024 × 1024 | `q4s-block-up-to128-f32` |
| `Image01EditTurbo` | image editing | 1024 × 1024 | `f16-qwen-vision-f32` |
| `Image01EditTurbo1k5` | image editing | 1536 × 1536 | `f16-qwen-vision-f32` |

The selected profile is immutable release identity. Packed-Q4 execution keeps signed Q4 matrix
weights packed for measured GPU kernels and retains explicitly declared auxiliaries in F32.
The Q4 CDN release stores Qwen and VAE once as shared dependency bundles; each public variant parent
contains only its own denoiser weights and exact component pins.

## Responsibilities

- prompt and reference-image conditioning;
- Boogu denoiser architecture and attention;
- DMD sampling and latent updates;
- Qwen and VAE composition;
- runtime resource plans and residency policy;
- Boogu artifact import, release preparation, and verification.

Qwen and VAE implementation details remain in their reusable crates. Application and Bevy code do
not belong here.

## Features

| feature | purpose |
|---|---|
| `runtime` | composed inference runtime |
| `burnpack` | Burnpack artifact support |
| `import` | conversion and release tools |
| `wgpu` / `webgpu` | GPU execution |
| `ndarray` | CPU reference validation |
| `autotune` | explicit kernel tuning for qualification or long-running workloads |

The `wgpu` and `webgpu` features compile against the public Burn API, but the resident packed-Q4
and packed-F16 execution guarantees require the root workspace's pinned `burn-cubecl`,
`cubecl-wgpu`, and `wgpu` patches. Cargo does not propagate a library crate's `[patch.crates-io]`
section, so downstream GPU applications must apply those exact patches until equivalent upstream
releases are available. CPU/reference features do not require them.

## Artifact tools

Import pinned checkpoints into sealed source bundles:

```sh
cargo run -p burn_boogu --features import --bin boogu-import -- --help
```

Build the canonical part-only CDN release:

```sh
cargo run --release --locked -p burn_boogu --features import \
  --bin boogu-prepare-cdn-release -- \
  --artifact-root .artifacts \
  --output-root .artifacts/cdn-upload-q4s-complete \
  --q4-only
```

Verify a release tree, including transport parts and reconstructed logical digests:

```sh
cargo run --release --locked -p burn_boogu --features import \
  --bin boogu-verify-artifacts -- \
  --artifacts .artifacts/cdn-upload-modular/aberration.technology/model/BOOGU_BUNDLE
```

## Validation

```sh
cargo test -p burn_boogu --all-features --lib --tests --locked
cargo clippy -p burn_boogu --all-targets --all-features --locked -- -D warnings
```

Real-checkpoint parity injects exact noise and compares conditioning, denoiser hooks, every DMD
step, VAE handoff, decoded tensors, and final pixels.
