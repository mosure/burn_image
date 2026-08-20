# burn_qwen3_vl

Reusable Qwen3-VL text and vision components implemented in Burn.

## Responsibilities

- ordinary Qwen3-VL architecture and processing;
- chat-template and tokenizer integration;
- image preprocessing and multimodal position construction;
- semantic artifact-stage descriptions and verified stage loading;
- bounded device-resident stage retention;
- packed quantized matrix execution where the selected artifact profile requires it.

Boogu prompts, DMD schedules, CDN locations, and Bevy types do not belong in this crate.

## Features

| feature | purpose |
|---|---|
| `std` | core model support; enabled by default |
| `tokenizers` | tokenizer and chat-template processing |
| `import` | checkpoint/Burnpack import support |
| `artifacts` | sealed artifact stages through `burn_image` |

## Artifact loading

Qwen weights are grouped by semantic stages. A source authenticates the sealed inventory and loads
only the requested stage into the selected device. Packed profiles preserve their declared storage
format for the measured kernels; the runtime report names any conversion policy explicitly.
Resident packed-Q4 execution currently requires the final application's patched Burn/CubeCL WGPU
backend; the architecture and CPU/reference surfaces do not.

## Validation

```sh
cargo test -p burn_qwen3_vl --all-features --locked
cargo clippy -p burn_qwen3_vl --all-targets --all-features --locked -- -D warnings
cargo check -p burn_qwen3_vl --target wasm32-unknown-unknown --all-features --locked
```

Cross-runtime correctness compares text processing, image processing, stage outputs, RoPE, and the
final conditioning tensors against pinned fixtures.
