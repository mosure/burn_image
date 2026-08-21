# bevy_burn_image

Bevy frontend for Burn image generation and editing on native WGPU and browser WebGPU.

This crate is intentionally an application shell. It owns ECS, UI, shared-device integration,
display, user input, file I/O, and browser transport adaptation. Model architecture, prompts,
schedules, and artifact semantics live in the model/runtime crates.

## Run

```sh
cargo install --path crates/bevy_image --locked --force
bevy_image
```

The default feature set installs the concrete native Boogu WGPU runtime. Run the workspace binary
without installing it with:

```sh
cargo run -p bevy_burn_image --bin bevy_image --release
```

The interactive UI can switch between Generate, Edit 1K, and Edit 1.5K. It loads only the selected
model, unloads modules that are no longer needed during a switch, validates prompt/reference input,
and keeps the active resident pipeline warm for subsequent requests.

The image canvas supports pointer-safe pan and zoom. Save controls appear only after an output is
ready. Edit mode accepts supported image formats through the platform file picker and can promote
the current output to the next request's reference.

## Unattended CLI

```sh
# Generate
cargo run -p bevy_burn_image --bin bevy_image --release -- \
  --variant turbo --prompt "a blue ceramic bird" --output result.png

# Edit
cargo run -p bevy_burn_image --bin bevy_image --release -- \
  --variant edit-turbo --source input.jpg --prompt "make it red" --output result.png
```

Supplying `--output` submits one job, writes the PNG and a timing/provenance report, then exits.
Use `--artifacts` for a local sealed bundle; otherwise native execution uses the verified CDN cache
under `~/.burn_image`.

## Browser

```sh
cargo build -p bevy_burn_image \
  --target wasm32-unknown-unknown \
  --profile wasm-release \
  --no-default-features \
  --features boogu-web \
  --lib
```

The browser and native app use the same Bevy UI and progress model. Browser startup performs a
device/VRAM plan check before transferring weights. Aggregate progress reports verified bytes,
logical objects, physical parts, throughput, and ETA. Cache Storage keeps authenticated transport
parts; warm requests reuse the active GPU pipeline and cached parts.

The generated package must include `www/burn-image-icon.png`. It is used for the browser favicon
and the native window icon.

## Device sharing

Bevy and Burn must use one adapter, device, and queue. Browser and native GPU modes fail clearly if
that contract cannot be established; they do not fall back to CPU inference.

## Features

| feature | purpose |
|---|---|
| `app` | Bevy UI and display |
| `gpu-interop` | shared Bevy/Burn WGPU device |
| `boogu-native` | native model runtime and verified CDN cache |
| `output-quality` | release-only native/browser output comparison helper |
| `boogu-web` | browser runtime and bounded Range loading |
| `native-autotune` | explicit native kernel tuning |

## Validation

```sh
cargo test -p bevy_burn_image --all-features --lib --locked
cargo clippy -p bevy_burn_image --all-targets --all-features --locked -- -D warnings
cargo check -p bevy_burn_image --target wasm32-unknown-unknown \
  --no-default-features --features boogu-web --lib --locked
node --test crates/bevy_image/tests/*.test.mjs
```

Browser model runs are opt-in because they require the exact released artifact tree and a supported
WebGPU device.
