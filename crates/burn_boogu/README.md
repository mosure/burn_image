# burn_boogu

Burn implementation of the Boogu-Image 0.1 Turbo, Edit-Turbo 1K, and Edit-Turbo 1.5K
composition layer.

`burn_boogu` owns Boogu-specific instruction preparation, conditioning, the 10B denoiser,
four-step DMD sampling, and composition of the reusable Qwen3-VL and FLUX VAE crates. Ordinary
Qwen3-VL architecture and processing live in `burn_qwen3_vl`; ordinary FLUX-compatible
`AutoencoderKL` math lives in `burn_flux_vae`; model-neutral image requests and results live in
`burn_image`.

Support claims are evidence-based. Tiny random-weight tests establish module contracts, while the
native parity binaries consume pinned real checkpoints and schema-2 upstream fixtures. A missing
fixture is a skipped local diagnostic, never a parity pass.

The two Edit configurations are distinct immutable runtime releases. `Image01EditTurbo1k5`
selects logical model id `Boogu/Boogu-Image-0.1-Edit-Turbo-1K5`, CLI slug
`edit-turbo-1k5`, and Hugging Face source revision
`60981c49e48cffadf2c169532a4ba3f6108afd5e` from the shared
`Boogu/Boogu-Image-0.1-Edit-Turbo` repository. It defaults to 1536×1536 and accepts only the
official presets listed below, bounded by side 2368 and 2,360,832 pixels. A 1K manifest is never
accepted as a 1.5K artifact identity, even if the upstream content store can reuse payload blobs.
Only the 1536×1536 default is currently numerical-parity and performance qualified; the other nine
presets are supported request shapes without independent parity or performance qualification.

## Features

| Feature | Surface |
| --- | --- |
| `std` (default) | model math, DMD schedule, and composition traits |
| `burnpack` | sealed manifest validation and sync/async staged Burnpack sources |
| `import` | converter plus SafeTensors-backed parity binaries; implies `burnpack` |
| `runtime` | `burn_image::ImageModel` host adapter and Hugging Face tokenizer integration |
| `ndarray` | CPU correctness/reference backend |
| `wgpu` | native Burn WGPU execution |
| `webgpu` | Wasm/WebGPU compilation through Burn WGPU |
| `quantized` | Q8 profile marker; artifact storage and load policy remain explicit |
| `reference-hooks` | named denoiser observations for exhaustive fixture comparison |

## Public API map

- `BooguConfig`, `BooguDenoiser`, and `BooguDenoiserInput` describe the released Boogu-specific
  transformer.
- `DmdSchedule`, `dmd_prediction`, and `dmd_renoise` expose the exact four-step update math.
- `ResidentBooguPipeline` composes resident components. `StreamingBooguPipeline` composes a
  streamed Qwen source, independently staged VAE, and either a resident or streamed denoiser.
- `StreamingBooguDenoiser` loads a prelude, each refiner/block, and the tail one at a time.
- `BooguImageModel` adapts a loaded pipeline to model-neutral `burn_image` requests, outputs,
  timings, and provenance.

With `burnpack`, the artifact boundary is deliberately explicit:

- `VerifiedBurnpackQwenStageSource` and `VerifiedAsyncBurnpackQwenStageSource` implement the
  reusable sync and Wasm-local async Qwen stage traits.
- `VerifiedBurnpackStageSource` and `VerifiedAsyncBurnpackDenoiserStageSource` implement sync and
  async one-stage-at-a-time denoiser loading.
- `VerifiedDirectoryVaeStageSource` and `VerifiedAsyncBurnpackVaeStageSource` load only the VAE
  half needed at the encode or decode boundary.
- `AsyncStageShardReader` receives the exact sealed `ArtifactFile` and a pre-read byte cap. The
  verified sources enforce file identity, declared size, SHA-256, stage ownership, key, shape,
  stored dtype, duplicate/missing tensors, and module-apply completeness.

No loader calls `collect()` on an unloaded lazy module. Qwen rank-two checkpoint tensors retain
their saved `[out, in]` shape until Burn's ordinary Col load mapper applies them.

## Bounded attention

`DoubleStreamAttention` and `GqaAttention` retain the complete key/value sequence but evaluate
queries in fixed, configurable tiles. Every tile uses Burn's ordinary unmasked attention operation
with the same default scaling and activation dtype as the former dense call. A tile size `q` bounds
the largest fallback score tensor to `batch * heads * q * key_length` elements instead of
`batch * heads * query_length * key_length`; only attended output tiles remain live until
concatenation. Grouped key/value heads are still expanded before attention, which is linear in
sequence length and does not create a sequence-squared tensor. The accepted native high-VRAM 1K
policy uses an 8192-row bound; the native 1.5K policy uses 16384 rows. Browser and constrained
runtimes may select smaller bounds. Each release has its own numerical and synchronized timing
evidence rather than inheriting a claim from another shape.

## Storage and execution profiles

| Profile | Storage | Execution contract | Status |
| --- | --- | --- | --- |
| `f16` | all floating tensors F16 | F16 Qwen/denoiser | diagnostic; Qwen vision accumulates excessive F16 drift |
| `f16-qwen-vision-f32` | Qwen vision F32, other model weights F16 | F32 Qwen vision, F16 text/denoiser; VAE is F32 compatibility or verified F16 native policy | canonical production policy; frontend alias `production` |
| `q8s-block32-f32` | eligible matrices Q8S block-32, other tensors F16 | policy depends on owner; F16 vision remains diagnostic | diagnostic |
| `q8s-block32-f32-qwen-vision-f32` | Qwen vision F32, eligible non-vision matrices Q8S | split policy below | legacy diagnostic; not in the production CDN set |

The Q8 split is required by measured Burn 0.21 behavior:

- Qwen uses `BooguQuantizedLoadPolicy::DequantizeF16`. Each already verified Q8S stage is
  dequantized on the host through F16 and then applied as an ordinary float snapshot, so the Col
  mapper transposes values without corrupting block scales. Qwen embeddings/text execute in F16;
  vision executes in F32 for the hybrid profile. Qwen is therefore **Q8 storage, not on-device Q8
  execution**.
- The Boogu denoiser uses `BooguQuantizedLoadPolicy::Preserve` with
  `BooguFloatLoadPolicy::AdaptToF32` and F32 activations. Its checkpoint matrices are already in
  Burn row layout, and the measured WGPU Q8/F32 kernel remains accurate.
- The VAE can use `force-f32` to match `force_upcast=true`, or `preserve-f16` on the measured native
  path. The latter has a separate retained-decoder numerical gate and must not be inferred for an
  untested backend.

`Preserve` is rejected by `boogu-qwen-parity` for a Q8 Qwen profile. The same quantized policy can
be selected independently on sync and async Qwen/denoiser sources with
`with_quantized_load_policy`.

## Residency policies

The Bevy native high-VRAM factory eagerly verifies and materializes the exact request graph before
reporting ready: required Qwen and VAE modules plus one denoiser remain resident on WGPU. Forward
then clones shared handles and performs no model-weight filesystem read, decode, or upload. Core
runners can use the same retaining wrappers lazily when startup latency is preferable. Native
construction can use `load_resident_denoiser_from_directory_with_policies` when sufficient VRAM is
available.

The Bevy browser production factory applies the equivalent contract to non-`Send` async sources:
one bounded, verified payload is held in Wasm during preload, then discarded after its initialized
dense-F32 WebGPU module is retained. The complete bundle is never held in Wasm linear memory, and
forward performs no model-weight HTTP read, decode, or upload. Explicit layer-streamed diagnostics
keep stages short-lived and intentionally repeat that traffic. The exact 1.5K parity route has its
own narrower lifetime: it retains all 48 denoiser stages across four DMD steps and clears them
before VAE decode.

## Binaries

| Binary | Purpose |
| --- | --- |
| `boogu-import` | convert a pinned Hugging Face snapshot into a descriptive sealed candidate bundle |
| `boogu-prepare-cdn-release` | split three exact legacy monoliths into two shared component entries and three dependency-pinned parents, with strict closure reports |
| `boogu-run` | native verified-artifact WGPU generation/editing; Qwen/VAE retained after first use, resident denoiser |
| `boogu-run-wgpu-blackbox` | native WGPU runner exposing required padded blackbox attention and fail-closed kernel controls |
| `boogu-qwen-parity` | aligned real-fixture Qwen vision/text/final-state parity and dtype observations |
| `boogu-denoiser-parity` | block-local isolated-step or cumulative captured-sigma DMD oracle replay |
| `boogu-full-parity` | production execution-dtype Qwen, VAE, DMD schedule, decode, and RGB chain |
| `boogu-parity` | processor, VAE, helper-DMD, decode, and final RGB comparisons |

Example native run:

```bash
cargo run --release -p burn_boogu --features runtime,import,wgpu --bin boogu-run -- \
  --artifacts .artifacts/boogu-image-0.1-turbo \
  --variant turbo --profile f16-qwen-vision-f32 \
  --vae-float-policy preserve-f16 --denoiser-query-chunk-size 512 \
  --prompt "A small red fox beneath a pine tree" \
  --width 256 --height 256 --seed 42 --repeat 2
```

`--repeat 2` reports iteration 0 as `cold-load` and iteration 1 as `warm-retained`, including wall
and per-stage timings. It is a high-VRAM benchmark, not the browser residency policy.

The 1.5K Edit runtime uses a separately converted bundle and requires a source image:

```bash
cargo run --release -p burn_boogu --features runtime,import,wgpu \
  --bin boogu-run-wgpu-blackbox -- \
  --artifacts .artifacts/boogu-image-0.1-edit-turbo-1k5 \
  --variant edit-turbo-1k5 --profile f16-qwen-vision-f32 \
  --vae-float-policy preserve-f16 \
  --vae-group-norm-policy f16-storage-f32-accum \
  --qwen-synchronization-policy deferred \
  --qwen-query-chunk-size 128 \
  --denoiser-query-chunk-size 16384 \
  --vae-attention-query-chunk-size 4096 \
  --denoiser-rms-norm-policy strict-f32 \
  --denoiser-qk-preparation-policy composed \
  --blackbox-num-planes 4 --blackbox-seq-kv-tiles 1 --blackbox-seq-q-tiles 1 \
  --conditioning-cache-policy disabled \
  --source /absolute/path/to/source.png \
  --prompt "Turn the circle into a bright orange sun" \
  --width 1536 --height 1536 --seed 42 --repeat 6
```

Omitting both dimensions selects the 1536×1536 model default. Other accepted native presets are
1264×1856, 1856×1264, 1344×1744, 1744×1344, 1392×1696, 1696×1392, 1152×2032,
2032×1152, and 2368×992; arbitrary in-envelope dimensions fail closed. Only the 1536×1536 default
has the native numerical and synchronized performance evidence below. Ordinary Browser WebGPU
still rejects `edit-turbo-1k5` before artifact loading. A separate no-surface, exact-fixture route
passed 1536×1536 numerical parity on the pinned RTX PRO 6000 Blackwell/Chrome 151 stack using
the historical schema-v1 flat bundle. It does not qualify the new schema-v2 modular closure,
browser performance, the ordinary UI route, another shape, or another stack; the modular browser
replay remains pending.

The parity-gated 1024 native-WGPU performance policy is selected explicitly:

```bash
cargo run --release -p burn_boogu --features runtime,import,wgpu \
  --bin boogu-run-wgpu-blackbox -- \
  --artifacts .artifacts/boogu-image-0.1-turbo \
  --variant turbo --profile f16-qwen-vision-f32 \
  --vae-float-policy preserve-f16 \
  --vae-group-norm-policy f16-storage-f32-accum \
  --vae-attention-query-chunk-size 4096 \
  --qwen-synchronization-policy deferred \
  --qwen-query-chunk-size 128 \
  --denoiser-query-chunk-size 8192 \
  --denoiser-rms-norm-policy strict-f32 \
  --denoiser-qk-preparation-policy balanced-strict-qk-norm-rope \
  --blackbox-num-planes 4 --blackbox-seq-kv-tiles 1 --blackbox-seq-q-tiles 1 \
  --conditioning-cache-policy disabled \
  --prompt "A matte red cube centered on a plain white studio background, soft shadow, front view." \
  --width 1024 --height 1024 --seed 42 --repeat 6
```

The generic native runner intentionally retains conservative defaults; benchmark claims apply to
the fully named policy above.

## Native performance evidence

The historical tables below predate the distinct 1.5K runtime and apply only to Turbo and 1K Edit.
The historical schema-v1 flat 1.5K mixed-F16 artifact is sealed as
`4e8b12ac5ca95272f9009080a23baf1bc52d1b0e7aebf2e9e5f394a492369213`. Its external exhaustive
1536×1536 fixture contains 372 tensors: q128 Qwen passes 70/70 semantic stages and the portable
streamed denoiser passes 240/240 internal boundaries. The exact accelerated production full chain
also passes: Qwen relative RMSE/cosine `0.09276403/0.9956884`, worst DMD velocity
`0.12193716/0.99257636`, final latent `0.07782394/0.99699116`, decode
`0.08242943/0.99667007`, and RGB `33.72779 dB/0.9920097` SSIM. Its JSON SHA-256 is
`5834c99b9139ce054891b30e29ecad4bf2d8a4ed1fb9d6482628d1d7350e87ef`, with empty stderr. This
historical evidence is not attributed to the current schema-v2 modular release.

The exact-policy uncached one-cold/five-warm result is Burn p50/nearest-rank p95
`12.553919/12.830840 s` versus upstream `5.968882/6.003932 s`, or `2.1032x/2.1371x`. This is a
large improvement over the original `5.55x` gap, but it does not meet the strict `<= 2x` target.
Its JSON SHA-256 is `19cd7392f51e5e1b2a777c56a7e95b7f1220c6c30bddc16f2271649b159a3644`.
That 1.5K policy remains q128/q16384/VAE-q4096 with strict-F32 denoiser RMSNorm, composed Q/K
preparation, deferred retained-Qwen synchronization, `p4/kv1/q1`, and full autotune; the 1K-only
balanced Q/K result is not inherited by this variant.

The original native WGPU baselines used one cold request followed by five warm requests with
retained verified Qwen/VAE stages and a resident denoiser.

| Task/profile | Shape | Cold | Warm p50 | Peak VRAM | JSON SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| Turbo mixed F16 | 256x256 | `75.2976 s` | `0.845361 s` | `42,748 MiB` | `c0102823f92683e323493ead7c3a6981fa7a3043045a40541478ddecf6a09202` |
| Turbo Q8 storage profile | 256x256 | `83.7512 s` | `2.206708 s` | `35,452 MiB` | `6b8b723c5300f1d198f2ef3bb6f30e244e17f8f1bf54d6944eea245cd778e86d` |
| Turbo mixed F16, original | 1024x1024 | `177.8581 s` | `10.073344 s` | `44,533 MiB` | `29701136910cdba8a90caa90e7c7a4eb1452b784b4359a19eb7cac38e0b4962e` |
| Edit mixed F16, original | 1024x1024 | `124.7204 s` | `11.655912 s` | `46,449 MiB` | `a66cfe57dd1e67e73dff98682cdf065c05ea9daf661c7bca56be95071fd75bb2` |

The 1024x1024 runs use artifact digests
`4f94cf68c00af12d5de486db4d316ce889d6d21e78913a1c74edab4bd0119ce3` for Turbo and
`14acbafd13dc9b79757e7d554b504396bee30ea7ed231f533919c6c82a6e6a32` for Edit. Their resulting
RGB8 PNG hashes are
`d6a2de570fe1bd4ff6b1d0dc27d998c435aa59928b27e365434061cbc526d005` and
`aa408d3c699978b5c17bda335d5b67c9645a7ad3672ae65896366f631b7faf6b`.

Against matching upstream BF16 warm medians of `1.731209 s` and `2.099106 s`, the original native
Burn baselines were `5.82x` slower for Turbo and `5.55x` slower for Edit. Those historical gaps are
preserved; the accepted 1K stack is forced padded blackbox attention `p4/kv1/q1`, balanced strict
Q/K preparation, an 8192-row denoiser query bound, F16 VAE safe mixed GroupNorm, a 4096-row VAE
query bound, retained q128 Qwen with deferred synchronization, and full autotune.

The uncached, apples-to-apples comparison recomputes conditioning on every measured request and
uses nearest-rank p50/p95 over five warm samples on both sides:

| Task | Burn p50 / p95 | Matching upstream p50 / p95 | Burn/upstream p50 / p95 |
| --- | ---: | ---: | ---: |
| Turbo | `3.400695447 / 3.494256796 s` | `1.731209 / 1.741948 s` | `1.964348x / 2.005947x` |
| Edit | `3.908315376 / 3.933613488 s` | `2.099106 / 2.102804 s` | `1.861895x / 1.870652x` |

Both tasks reach `<= 2x` at p50 and Edit also does at p95 without conditioning reuse. Turbo p95
misses by `10.360 ms`; a second exact-policy repeat passed p95 at `1.980429x`, so that tail is at
the noise boundary rather than robustly below it. Edit has `289.896/271.994 ms` of p50/p95
headroom. The release-validated performance JSON SHA-256 values are
`537344f26f342fe40142d631a0668bae2820b71eeb2dfefdee5a55e4f9bf275d` and
`9a3659dd6356d105b4c4a7477f2c0dad8e06d8d6185df7541fcc26ce37d43433`. Hot telemetry rows
averaged `99.82%` board GPU utilization for both tasks, with `44,129`/`46,045 MiB` PID framebuffer
peaks; board utilization is not process-exclusive kernel occupancy.

For historical context, the superseded q1024 policy's exact-request conditioning cache
reused Qwen instruction features only when prompt, source, and effective token length matched:

| Task | Cached p50 / p95 | Ratio to the historical upstream percentile |
| --- | ---: | ---: |
| Turbo | `3.396570999 / 3.475443177 s` | `1.96196x / 1.9976x` |
| Edit | `3.904326915 / 3.906317438 s` | `1.859995x / 1.858183x` |

The cached table is historical repeated-identical-request evidence, not a strict comparison with
upstream recomputation and not a measurement of the current q4096 policy. Turbo cached evidence
has JSON/PNG SHA-256
`9fa21321eeccfcfbb96775dd191e07bc0fd120926dca664d65e5b6d521cacb5e`/
`ba06424d204e4dfca006eda82d04c7a9ec6b450faff4e34b8de774bd49151e38`; Edit has
`f87c3519b5be3e63a513a3427ace9a2abde7077a69cf4ba28d547662eab09e27`/
`ea84161c0551c1754f7b4350a8fae55ed8345d6cb9c12bbdf5fe666208cd8d17`.

These standard production-RNG runs are released-shape operational/performance evidence, not
fixture-injected tensor parity. The optimized Turbo and Edit full chains independently pass their
256x256 exact fixture gates; their release-validated JSON SHA-256 digests are
`055548fcbe5bd4e2ff612e57f0038cf194b82f0114e47b3095f925ee0d0fa948` and
`746f44d18d66b2e348ce00bc0425777477137e6ba917d0b91d4a46f1a6dcf4f5`. Both reports have empty
stderr, pass every full-chain gate, and explicitly record `native_release_policy_validated: true`,
`vae_attention_query_chunk_size: 4096`, and balanced strict Q/K preparation.
See the repository
[performance report](https://github.com/mosure/burn_image/blob/main/docs/reports/2026-08-11-rtx-pro-6000.md).

The native runtime overlaps exact deterministic DMD host-noise preparation with uncached GPU
conditioning when useful and falls back to the serial path on Wasm, cached text-only requests, or
thread-spawn failure. F16 decoder readback maps directly into validated RGB8 bytes without a full
F32 host-image allocation; the accepted Turbo PNG SHA-256 remains unchanged.

## Historical native mixed/Q8 component evidence

The following 256x256 Edit/Turbo measurements used the matching sealed mixed-F16 or
`q8s-block32-f32-qwen-vision-f32` bundles on native WGPU. They predate the final native-autotune
build, and the denoiser measurements also predate bounded Boogu query attention. They remain
component diagnostics; the repository parity guide contains the current source-bound full-chain
results.

- Turbo Qwen: 38/38 aligned stages, minimum cosine `0.9958083`, worst relative RMSE
  `0.12069529`; final hidden cosine `0.9995802`, relative RMSE `0.028973013`; `67.353 s`.
- Edit Qwen: 70/70 aligned stages, F32 vision and F16 text observed across the insertion boundary;
  minimum cosine `0.9940735`, worst relative RMSE `0.10910508`; final hidden cosine
  `0.99536407`, relative RMSE `0.09617868`; `77.602 s`.
- Edit denoiser isolated-step replay: 240/240 boundaries across four calls, minimum cosine
  `0.9988636`, worst relative RMSE `0.047695775`; `189.837 s`. This is block-local evidence and
  does not substitute for a cumulative captured-sigma replay or the production full-chain gate.
- Edit denoiser cumulative mixed-F16 captured-sigma replay: worst velocity relative RMSE/cosine
  `0.15863511`/`0.9874189`; final-latent relative RMSE/cosine `0.08544214`/`0.99634737`;
  `272.579 s`.
- Edit denoiser cumulative Q8/F32 captured-sigma replay: worst velocity relative RMSE/cosine
  `0.22839242`/`0.974051`; final-latent relative RMSE/cosine `0.17953193`/`0.9839146`;
  `177.631 s`. This is inside the independently measured upstream Edit F16-versus-BF16 execution
  envelope (`0.2657993`/`0.9646234` at the worst velocity and `0.2001689`/`0.9799298` at the
  final latent).
- Turbo denoiser cumulative mixed-F16 captured-sigma replay: worst velocity relative RMSE/cosine
  `0.19143246`/`0.98166704`; final-latent relative RMSE/cosine `0.082120545`/`0.9966279`;
  `262.259 s`. It passes the same Edit-calibrated cross-dtype envelope; this is not an independent
  Turbo envelope calibration.
- Turbo end-to-end smoke: coherent 256x256 RGB output, `73.500 s` cold inference
  (`66.231 s` Qwen, `4.616 s` four-step DMD, `2.648 s` VAE decode), with about `35.0 GiB` sampled
  peak VRAM under the high-VRAM policy. The highest manual PID-scoped `nvidia-smi` sample was
  `35,033 MiB` during VAE decode; this was not continuous sampling and is not a true peak bound.

Artifact digests and complete boundary JSON remain part of the run output. See the repository
[parity guide](https://github.com/mosure/burn_image/blob/main/docs/parity.md) for gates and the
distinction between isolated and captured-sigma trajectory evidence. The latter is an oracle
replay; `boogu-full-parity` owns production execution-dtype propagation.

## Current limits

- Native WGPU has real-checkpoint numerical evidence at 256x256 and operational/performance
  acceptance at 1024x1024 for Turbo and 1K Edit. The 1.5K runtime/configuration, strict artifact
  identity, 1536x1536 production full-chain numerical gate, and synchronized performance row are
  validated; its exact-policy uncached p50/p95 are `2.1032x/2.1371x`, so performance parity is not
  achieved. Exhaustive q128 Qwen and portable-denoiser components are covered, while accelerated
  attention is gated through propagated full-chain boundaries. The concrete browser
  runtime has completed one real hardware-WebGPU
  Turbo request at its deliberately narrower 256x256 limit, but that ordinary prompt/seed run is
  not upstream fixture parity. The production headless Bevy swap-chain smoke remains failed on
  this Chrome/host combination; see the repository web report for the exact distinction.
- Q8 reduces stored/denoiser device weight cost, but Qwen currently executes from stage-local F16
  dequantization because the Burn 0.21 Col Q8 mapper is not numerically valid.
- Performance comparisons are meaningful only after the matching numerical gate passes and must
  name artifact digest, backend, adapter, dimensions, and residency policy.
- The forced native-WGPU padded blackbox `p4/kv1/q1` policy has direct numerical and end-to-end
  acceptance on the reported RTX PRO 6000. Wider query partitions fail closed. Browser WebGPU does
  not inherit this native cooperative-matrix claim. The native CUDA probe produced invalid output
  and is excluded from supported correctness and performance claims.
- `edit-turbo-1k5` is not exposed by the ordinary browser UI and cannot reuse the 1K Edit artifact,
  fixture, or evidence row. A historical surface-free browser route authenticated the legacy
  schema-v1 monolith, but the modular five-entry closure still needs its own rerun; browser UI and
  performance qualification remain separate gates.
