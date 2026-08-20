# Architecture

`burn_image` is split so reusable model code does not depend on the application shell.

```text
burn_image
├── burn_qwen3_vl
├── burn_flux_vae
└── burn_boogu
    ├── burn_qwen3_vl
    └── burn_flux_vae

bevy_burn_image
├── burn_image
└── burn_boogu
```

Dependencies point from model/application crates toward reusable contracts. A reusable crate must
not contain Boogu UI policy, CDN locations, or Bevy types.

## Ownership

### `burn_image`

Owns model-neutral requests, outputs, jobs, capabilities, artifact identity, transport, integrity,
cache policy, progress, and provenance.

### `burn_qwen3_vl`

Owns Qwen3-VL architecture, text and image processing, semantic artifact stages, verified stage
loading, and device-resident stage retention.

### `burn_flux_vae`

Owns the FLUX-compatible `AutoencoderKL`, latent conventions, semantic artifact stages, verified
stage loading, and device-resident stage retention.

### `burn_boogu`

Owns Boogu conditioning, denoising, DMD sampling, variant/profile policy, model composition,
resource plans, import, and release verification.

### `bevy_burn_image`

Owns ECS state, the Bevy UI, shared WGPU integration, display, native/browser I/O, and task
orchestration. Model execution is delegated through runtime factories and model-neutral jobs.

## Request flow

1. The UI selects a model variant and validates prompt/reference requirements.
2. The platform runtime resolves one immutable bundle/profile.
3. A device resource plan is validated before artifact transfer.
4. Manifests and transport layouts are authenticated.
5. Required semantic stages are streamed, verified, and uploaded to the GPU.
6. Qwen produces conditioning; the denoiser executes every DMD step; the VAE decodes the latent.
7. The runtime emits a device-resident result and complete provenance.
8. The frontend converts the result for display or PNG output without changing model semantics.

Only the selected variant is loaded. Switching variants tears down modules that the next pipeline
does not use before allocating replacement weights.

## GPU ownership

Bevy creates the WGPU adapter, device, and queue. Burn receives those exact handles through
`bevy_burn`; it does not create a second device. The contract is validated at runtime and GPU modes
fail if it cannot be established.

Interactive inference runs outside the Bevy update loop. ECS receives progress and completion
events, so window input and rendering remain responsive while GPU work is active. Canvas pan/zoom
is disabled while pointer focus belongs to UI controls.

## Numeric profiles

Storage profile, load policy, and execution policy are separate and are all recorded:

- Turbo ordinary execution uses packed signed-Q4 matrix weights with F32 auxiliaries.
- Edit ordinary execution uses its sealed mixed-F16 profile.
- Packed kernels accumulate in F32 and do not require WebGPU `shader-f16`.
- Any stage conversion is bounded and named; storage compression alone is not called quantized
  execution.

## Trust boundaries

Raw paths and JSON are not trusted runtime inputs. Manifests, transport layouts, local directories,
and remote responses become usable only through verified types. The loader rejects missing,
duplicate, unknown, oversized, shape-incompatible, or digest-mismatched data.

See [artifacts](./artifacts.md), [browser deployment](./web.md), and
[correctness](./parity.md).
