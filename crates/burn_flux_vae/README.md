# burn_flux_vae

`burn_flux_vae` is a Burn 0.21 implementation of the ordinary Diffusers
`AutoencoderKL` architecture used by FLUX.1 Dev and Schnell. It provides reusable VAE tensor math;
it does not own image resizing, prompts, schedulers, Boogu conditioning, model download policy, or
application integration.

## Implemented contract

- NCHW encoder and decoder with Diffusers-compatible module names.
- `DownEncoderBlock2D` and `UpDecoderBlock2D` residual stacks.
- asymmetric bottom/right-padded strided downsampling.
- nearest-neighbor upsampling followed by a padded 3x3 convolution.
- residual-attention-residual middle blocks.
- F32 GroupNorm reductions for F16/BF16 inputs.
- query-chunked spatial attention with F32 score/softmax evaluation.
- optional `quant_conv` and `post_quant_conv` layers.
- diagonal Gaussian mean/log-variance split with Diffusers' `[-30, 20]` clamp.
- exact caller-provided epsilon sampling for cross-runtime parity.
- explicit FLUX latent scale and shift helpers.
- deterministic Burn/Diffusers tensor inventories and optional strict Safetensors/Burnpack loading.

`AutoencoderKlConfig::flux1()` matches the public FLUX.1 configuration:

| Field | Value |
| --- | --- |
| channels | `3 -> [128, 256, 512, 512] -> 3` |
| latent channels | `16` |
| residual layers per encoder block | `2` |
| residual layers per decoder block | `3` |
| GroupNorm | `32` groups, epsilon `1e-6` |
| spatial compression | `8x` |
| scaling factor | `0.3611` |
| shift factor | `0.1159` |
| quant/post-quant convolution | disabled |
| middle attention | enabled |

## Usage

```rust
use burn::backend::NdArray;
use burn::tensor::Tensor;
use burn_flux_vae::AutoencoderKlConfig;

type Backend = NdArray<f32>;

let device = Default::default();
let vae = AutoencoderKlConfig::flux1().init::<Backend>(&device);
let images = Tensor::<Backend, 4>::zeros([1, 3, 1024, 1024], &device);

// Diffusers AutoencoderKL returns an unscaled posterior.
let posterior = vae.encode(images);
let raw_latents = posterior.mode();

// FLUX pipelines apply scale/shift outside AutoencoderKL.
let pipeline_latents = vae.scale_latents(raw_latents);
let decoded = vae.decode_scaled(pipeline_latents);
assert_eq!(decoded.dims(), [1, 3, 1024, 1024]);
```

For numerical parity, inject the exact reference epsilon tensor:

```rust,ignore
let posterior = vae.encode(images);
let raw_latents = posterior.sample_with_epsilon(reference_epsilon);
```

Backend RNG seeds are not a portable parity contract.

## Weight loading

Loading is isolated behind the `import` feature:

```rust,ignore
use burn_flux_vae::{AutoencoderKlConfig, load_safetensors_file};

let config = AutoencoderKlConfig::from_diffusers_json(&config_json)?;
let (vae, report) = load_safetensors_file::<Backend>(
    &device,
    "vae/diffusion_pytorch_model.safetensors",
    &config,
)?;
assert!(report.is_complete());
```

The Safetensors loader remaps Diffusers GroupNorm and `to_out.0` keys, applies the PyTorch-to-Burn
linear layout adapter, converts BF16/F16/F64 source weights to F32 by default, validates shapes, and
rejects incomplete complete-file loads. Set `LoadOptions::force_f32` to `false` to disable that
conversion on a backend that supports the stored dtype; it is `true` by default. Burnpack files can
be loaded or saved directly.
`BurnpackShardLoader` additionally tracks duplicate, unexpected, and missing tensors across
sequential partial Burnpack payloads.

`TensorInventory::from_config` returns both Burn and Diffusers names and shapes. This is intended for
conversion manifests and preflight checks; artifact provenance, checksums, shard bounds, and CDN
transport belong to the repository's model-neutral artifact layer.

## Features

| Feature | Purpose |
| --- | --- |
| `ndarray` | default CPU backend for tests and tools |
| `flex` | Burn Flex backend |
| `wgpu` | native WGPU backend |
| `webgpu` | browser WebGPU backend |
| `import` | Safetensors and Burnpack loading/conversion surfaces |

The upstream FLUX configuration sets `force_upcast = true`. The flag is preserved and exposed, but
mixed-dtype parameter mutation is intentionally not hidden inside `forward`; production callers
should load the VAE in F32 when strict Diffusers force-upcast behavior is required.

## Validation

The crate includes tiny real tensor tests for configuration parsing, asymmetric downsampling,
upsampling, bounded attention, posterior clamping and explicit epsilon sampling, scale/shift
round-trips, encode/decode shapes, inventory completeness, key remapping, and Burnpack round trips.
These are structural and mathematical checks, not a claim of real-checkpoint parity. Pinned
Diffusers tensor-by-tensor parity belongs in the workspace integration suite.
