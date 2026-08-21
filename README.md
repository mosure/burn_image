# burn_image 🔥🖼️

[![CI](https://github.com/mosure/burn_image/actions/workflows/ci.yml/badge.svg)](https://github.com/mosure/burn_image/actions/workflows/ci.yml)
[![deploy github pages](https://github.com/mosure/burn_image/actions/workflows/deploy-pages.yml/badge.svg)](https://github.com/mosure/burn_image/actions/workflows/deploy-pages.yml)

Burn-native image generation and editing with a shared native/WebGPU runtime and a Bevy UI.

The workspace implements Boogu Image Turbo, Edit Turbo, and Edit Turbo 1.5K. Model artifacts are
sealed, content-addressed, range-loadable, and usable from native WGPU and browser WebGPU without a
CPU inference fallback.

## Highlights

- Generate 1024 px images or edit references at 1024 px and 1536 px.
- Share one WGPU adapter, device, and queue between Bevy rendering and Burn inference.
- Keep the selected model's packed-Q4 weights resident for warm-session inference.
- Share one Q4 Qwen component and one VAE component across all three Q4 model variants.
- Load only the selected model and cache verified native artifacts under `~/.burn_image`.
- Stream browser artifacts as immutable transport parts no larger than 25,000,000 bytes.
- Verify manifests, transport parts, reconstructed Burnpacks, tensor inventories, and provenance.
- Drive the Bevy app interactively or through its unattended CLI.

## Workspace

| crate | responsibility |
|---|---|
| [`burn_image`](./crates/burn_image) | model-neutral requests, jobs, outputs, artifacts, and provenance |
| [`burn_qwen3_vl`](./crates/burn_qwen3_vl) | Qwen3-VL text/vision processing and artifact stages |
| [`burn_flux_vae`](./crates/burn_flux_vae) | FLUX-compatible `AutoencoderKL` |
| [`burn_boogu`](./crates/burn_boogu) | conditioning, denoising, DMD sampling, and model composition |
| [`bevy_burn_image`](./crates/bevy_image) | ECS, UI, device sharing, display, and platform I/O |

## Run the app

Install the native WGPU application from the workspace and launch it:

```sh
cargo install --path crates/bevy_image --locked --force
bevy_image
```

The default feature set includes the concrete native Boogu WGPU runtime. For development, run the
same binary directly from Cargo:

```sh
cargo run -p bevy_burn_image --bin bevy_image --release
```

The default source is the verified CDN cache. Use a local sealed bundle when developing artifacts:

```sh
cargo run -p bevy_burn_image --bin bevy_image --release -- \
  --variant turbo \
  --artifacts .artifacts/cdn-upload-q4s-complete/aberration.technology/model/boogu-image-0.1-turbo-q4s-block-up-to128-f32
```

Run one unattended request:

```sh
cargo run -p bevy_burn_image --bin bevy_image --release -- \
  --variant turbo \
  --prompt "a blue ceramic bird" \
  --output result.png
```

Edit requests use `--variant edit-turbo` or `--variant edit-turbo-1k5` and require `--source`.

## Model profiles

All three public variants use the sealed `q4s-block-up-to128-f32` profile by default. It preserves
signed Q4 matrix weights for measured GPU kernels and keeps F32 auxiliaries explicit. The larger
`f16-qwen-vision-f32` releases remain explicit validation profiles; a runtime never silently
substitutes a different profile because profile identity is part of the manifest and provenance.

## Browser build

```sh
cargo build -p bevy_burn_image \
  --target wasm32-unknown-unknown \
  --profile wasm-release \
  --no-default-features \
  --features boogu-web \
  --lib
```

The browser uses the same Bevy controls as native. Cache Storage retains authenticated transport
parts, while Wasm reconstructs and verifies one logical object at a time before uploading it to the
shared GPU device.

## Documentation

- [architecture](./docs/architecture.md)
- [artifact and CDN contract](./docs/artifacts.md)
- [correctness and parity](./docs/parity.md)
- [performance methodology](./docs/performance.md)
- [browser deployment](./docs/web.md)

## Validation

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --locked
cargo check -p bevy_burn_image --target wasm32-unknown-unknown \
  --no-default-features --features boogu-web --lib --locked
```

Real-artifact native and browser qualification is opt-in and is run only against pinned manifests
and checkpoints. See [correctness and parity](./docs/parity.md) for the required evidence layers.

## GPU backend compatibility

The model-neutral `burn_image` crate does not depend on a GPU backend and can be consumed directly
from crates.io. The resident packed-Q4 and packed-F16 WGPU/WebGPU profiles currently require the
workspace's pinned `burn-cubecl`, `cubecl-wgpu`, and `wgpu` patches. Cargo patches are selected by
the final application workspace and are not inherited from a library dependency; downstream GPU
applications must apply the same revisions until those changes are available in upstream releases.
