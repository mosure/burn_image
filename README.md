# burn_image 🔥🕊️🖼️

[![CI](https://github.com/mosure/burn_image/actions/workflows/ci.yml/badge.svg)](https://github.com/mosure/burn_image/actions/workflows/ci.yml)
[![deploy github pages](https://github.com/mosure/burn_image/actions/workflows/deploy-pages.yml/badge.svg)](https://github.com/mosure/burn_image/actions/workflows/deploy-pages.yml)
[![crates.io](https://img.shields.io/crates/v/burn_image.svg)](https://crates.io/crates/burn_image)
[![docs.rs](https://docs.rs/burn_image/badge.svg)](https://docs.rs/burn_image)
[![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

Burn-native image generation and editing with
[Boogu-Image 0.1](https://github.com/boogu-project/Boogu-Image), a native/WebGPU Bevy viewer,
verified model artifacts, and a model-neutral Rust API.

## features

### high-level

- Generates 256-1024 px images with Boogu-Image Turbo.
- Edits reference images with the 1K and 1.5K Edit checkpoints.
- Runs model layers on WGPU/WebGPU without a silent CPU fallback.
- Shares one adapter, device, and queue between Bevy and Burn in the browser.
- Keeps selected high-VRAM pipelines resident for fast repeat inference.
- Offers bounded low-VRAM policies with explicit storage and execution provenance.
- Loads sealed, content-addressed model releases from the canonical CDN or a local mirror.
- Caches native models under `~/.burn_image/models/` and browser parts in origin-scoped
  Cache Storage.
- Ships a responsive Bevy UI with model/size selectors, prompt and reference validation,
  cancellation, progress, image pan/zoom, and PNG export.
- Supports unattended native generation and editing through ordinary CLI arguments.

### model paths

| selector | task | default | notes |
|---|---|---:|---|
| `turbo` | generation | 1024x1024 | official 1K aspect-ratio presets |
| `edit-turbo` | image editing | 1024x1024 | requires a reference image |
| `edit-turbo-1k5` | image editing | 1536x1536 | distinct 1.5K checkpoint and presets |

These are large models. Resident execution expects a workstation-class GPU and tens of gigabytes
of artifact storage. Low-VRAM mode reduces simultaneous residency; it is not a CPU fallback.
Validation claims are backend, model, resolution, and policy specific. See
[numerical parity](./docs/parity.md) and [performance](./docs/performance.md) for the evidence matrix.

## quick start

The Bevy viewer requires Rust 1.95 or newer:

```sh
cargo run --release --locked -p bevy_burn_image \
  --features boogu-native --bin burn-image-viewer -- \
  --variant turbo \
  --profile production \
  --residency native-high-vram
```

The viewer downloads and verifies only the selected release. Use `--variant edit-turbo` or
`--variant edit-turbo-1k5` to start in an edit model, or switch models from the in-app selector.
The previous runtime is unloaded before the next model is constructed.

Use an existing local artifact mirror with `--artifacts`:

```sh
cargo run --release --locked -p bevy_burn_image \
  --features boogu-native --bin burn-image-viewer -- \
  --variant turbo --profile production \
  --artifacts .artifacts/boogu-image-0.1-turbo-f16-qwen-vision-f32
```

Select the bounded policy with `--residency low-vram`. Static kernels are the interactive default;
build with `--features boogu-native,native-autotune` and pass `--autotune balanced` only when
first-use tuning is acceptable.

## automation

One-shot generation uses the same validated request path as the UI:

```sh
target/release/burn-image-viewer \
  --variant turbo \
  --prompt "a glossy blue ceramic bird" \
  --width 1024 --height 1024 --seed 0 \
  --output result.png \
  --report result.json \
  --timeout-seconds 1800
```

For editing, use `--variant edit-turbo` or `edit-turbo-1k5` and add `--source input.jpg`.
`--output` hides the window unless `--show-window` is supplied.

## browser viewer

Open the hosted app at [mosure.github.io/burn_image](https://mosure.github.io/burn_image/).

The browser downloads only the selected model's executable closure. CDN weights are sealed
transport parts targeting 20 MiB with an exact 25,000,000-byte maximum; a complete bundle never
enters Wasm linear memory. Startup checks exact existing cache keys, available origin quota, and
the selected model's conservative VRAM plan before transferring large weights. Repeat requests
reuse the warm WebGPU pipeline, while model releases sharing Qwen/VAE parts reuse those browser
cache entries.

Build and serve the WebAssembly frontend locally with the commands in the
[web deployment guide](./docs/web.md). The checkout carries required `wgpu` and `cubecl-wgpu`
patches for bounded browser uploads and truthful asynchronous queue completion; downstream browser
applications must apply equivalent patches until those fixes are available upstream.

## workspace

| crate | responsibility |
|---|---|
| [`burn_image`](./crates/burn_image) | model-neutral requests, jobs, artifacts, integrity, and provenance |
| [`burn_qwen3_vl`](./crates/burn_qwen3_vl) | Qwen3-VL processing, architecture, stages, and device retention |
| [`burn_flux_vae`](./crates/burn_flux_vae) | FLUX-compatible `AutoencoderKL` and verified stage loading |
| [`burn_boogu`](./crates/burn_boogu) | Boogu conditioning, denoiser, DMD sampling, and composition |
| [`bevy_burn_image`](./crates/bevy_image) | Bevy ECS/UI, shared GPU integration, display, and platform I/O |

The frontend package is named `bevy_burn_image` because Bevy already publishes a crate named
`bevy_image`.

## development

```sh
cargo +stable fmt --all -- --check
cargo +stable clippy -p burn_image --all-targets --locked -- -D warnings
cargo +stable test -p burn_image --all-targets --locked
cargo +stable test -p bevy_burn_image --lib --all-features --locked
cargo +stable check -p bevy_burn_image --target wasm32-unknown-unknown \
  --no-default-features --features boogu-web --locked --lib
```

The [CI workflow](./.github/workflows/ci.yml) is authoritative for package-isolated feature,
MSRV, archive, and WebAssembly checks. Hardware and real-artifact jobs are intentionally separate.

## docs

- [architecture](./docs/architecture.md) - crate ownership and runtime lifecycle
- [artifacts](./docs/artifacts.md) - conversion, manifests, CDN layout, and caches
- [numerical parity](./docs/parity.md) - fixtures, thresholds, and supported claims
- [performance](./docs/performance.md) - measurement method and current results
- [web deployment](./docs/web.md) - Wasm build, Pages, browser limits, and troubleshooting
- [RTX PRO 6000 report](./docs/reports/2026-08-11-rtx-pro-6000.md) - detailed machine evidence

## license

Dual-licensed under either [MIT](./LICENSE-MIT) or [Apache-2.0](./LICENSE-APACHE), at your option.
Checkpoint and upstream source licenses remain those published by their respective authors;
converted model artifacts are not committed to this repository.
