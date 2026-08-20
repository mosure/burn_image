# burn_flux_vae

Burn-native, Diffusers-compatible FLUX `AutoencoderKL`.

## Responsibilities

- encoder and decoder architecture;
- diagonal Gaussian moments and sampling;
- FLUX latent scaling and shifting;
- semantic artifact-stage descriptions and verified stage loading;
- packed weight execution selected by the surrounding runtime.

The crate is reusable and contains no Boogu prompts, schedules, CDN URLs, or Bevy types.

## Features

| feature | purpose |
|---|---|
| `ndarray` | CPU reference backend; enabled by default |
| `wgpu` | native WGPU backend |
| `webgpu` | browser WebGPU backend |
| `flex` | Burn flex backend |
| `import` | record/checkpoint import |
| `artifacts` | sealed artifact stages through `burn_image` |

Packed-F16 GPU execution currently requires the final application's patched Burn/CubeCL WGPU
backend. The ordinary model architecture and CPU/reference features do not require those patches.

## Validation

```sh
cargo test -p burn_flux_vae --all-features --locked
cargo clippy -p burn_flux_vae --all-targets --all-features --locked -- -D warnings
cargo check -p burn_flux_vae --target wasm32-unknown-unknown --all-features --locked
```

Correctness covers moments, injected samples, decoded tensors, and final pixels. Random seeds alone
are not used as cross-runtime evidence.
