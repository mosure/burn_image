# burn_image 🕊️🔥🖼️

[![CI](https://github.com/mosure/burn_image/actions/workflows/ci.yml/badge.svg)](https://github.com/mosure/burn_image/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![Burn](https://img.shields.io/badge/Burn-0.21-orange.svg)](https://burn.dev)
[![Bevy](https://img.shields.io/badge/Bevy-0.19-232326.svg)](https://bevy.org)

Burn-native image generation and editing with
[Boogu-Image 0.1](https://github.com/boogu-project/Boogu-Image). Use it through the Bevy desktop
viewer or embed the model-neutral Rust API. Native WGPU is the primary runtime; browser WebGPU is
available as an experimental, GPU-resident path.

## Support

| Model path | Current status |
| --- | --- |
| Turbo 1K generation | Native production path; numerical gates pass and performance is at the 2× target boundary |
| Edit-Turbo 1K | Native production path; numerical and 2× performance gates pass |
| Edit-Turbo 1.5K | Native numerical gates pass; historical flat-bundle 1536 browser parity passed on the attested RTX/Chrome stack, while the modular bundle, browser UI, and browser performance remain unqualified |
| Turbo in the browser | Resident mechanism is experimental; a historical streamed request ran on real WebGPU, but resident UI/numerical/performance qualification is still open |

> [!IMPORTANT]
> These are large models. Expect tens of gigabytes of artifacts and a workstation-class GPU for
> resident execution. Native and browser production modes finish verified GPU preloading before
> becoming ready; the browser never loads the complete bundle into Wasm memory.

## Quick start

The full viewer requires Rust 1.95 or newer:

```bash
cargo run --release --locked -p bevy_burn_image \
  --features boogu-native --bin burn-image-viewer -- \
  --variant turbo \
  --profile production \
  --residency native-high-vram
```

`production` selects the parity-qualified, F16-first storage policy. Once the canonical CDN bundle
is published, the viewer resolves it, verifies it, and caches it under
`~/.burn_image/models/`. Until then, add `--artifacts /path/to/production-bundle` to use a local
bundle.
Enter a prompt, select **Run**, then save the generated PNG from the viewer.

For image editing, use `--variant edit-turbo` and select or drop a reference image. The native
1.5K checkpoint is selected with `--variant edit-turbo-1k5` and its matching artifact bundle.

For local browser builds and GitHub Pages deployment, see the [web guide](docs/web.md). The web
shell reports the current component, shard transfer, verification, and inference stage so a large
model load does not look frozen.

## Workspace

| Crate | Responsibility |
| --- | --- |
| [`burn_image`](crates/burn_image) | Model-neutral requests, jobs, manifests, integrity, transport, and provenance |
| [`burn_qwen3_vl`](crates/burn_qwen3_vl) | Qwen3-VL model/processor plus its sealed component, semantic shard loader, and device cache |
| [`burn_flux_vae`](crates/burn_flux_vae) | FLUX `AutoencoderKL` plus its sealed component, encoder/decoder shard loader, and device cache |
| [`burn_boogu`](crates/burn_boogu) | Boogu conditioning, variant-specific denoiser, DMD composition, and native runners |
| [`bevy_burn_image`](crates/bevy_image) | Native/web Bevy UI and shared WGPU integration |

The frontend package is named `bevy_burn_image` because Bevy already publishes a crate named
`bevy_image`.

## Development

```bash
cargo fmt --all -- --check
cargo test -p burn_image --all-targets --locked
cargo clippy -p burn_image --all-targets --locked -- -D warnings
```

The [CI workflow](.github/workflows/ci.yml) contains the authoritative package-isolated test,
feature-contract, and WebAssembly build matrix. Use it instead of combining every model and
frontend feature into one linker-heavy workspace test.

## Documentation

- [Architecture](docs/architecture.md) — runtime structure, component ownership, and GPU lifecycle
- [Artifacts](docs/artifacts.md) — profiles, conversion, verification, caching, and CDN layout
- [Numerical parity](docs/parity.md) — fixtures, thresholds, and current model evidence
- [Performance](docs/performance.md) — methodology, accepted results, and remaining gaps
- [RTX PRO 6000 report](docs/reports/2026-08-11-rtx-pro-6000.md) — detailed machine evidence
- [Web deployment](docs/web.md) — Wasm build, Pages deployment, loading UI, and browser limits

Exact checkpoint revisions, artifact digests, numerical metrics, benchmark samples, and GPU
telemetry live in those focused documents rather than on this landing page.

## License

Dual-licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
Checkpoint and upstream source licenses remain those published by their respective authors;
converted artifacts are not committed to this repository.
