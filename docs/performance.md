# Performance methodology

This document defines the acceptance methodology for future reproducible benchmark reports.
Performance should be compared only after the matching numerical gate passes, with upstream
PyTorch and Burn using the same checkpoint, prompt, dimensions, warmup count, sample count, and GPU
power/performance state.

Release identity is part of that match. `edit-turbo-1k5` means the immutable
`60981c49e48cffadf2c169532a4ba3f6108afd5e` hotfix, a distinct sealed bundle, an authenticated
source image, and one of the official 1.5K presets. Neither 1K Edit timing nor a run that silently
aligns to the source image may be used as its denominator.
Only the 1536x1536 default currently has matching numerical and synchronized performance evidence.
The other nine official presets are accepted request shapes, not parity- or performance-qualified
benchmark configurations.

Reports record:

- adapter, driver, operating system, browser, Burn/WGPU and upstream revisions;
- artifact profile and on-disk/downloaded byte counts;
- native residency policy (`native-high-vram` or `diagnostic-layer-streamed`) and bytes reread per step;
- cold manifest-to-first-pixel time and warm inference time;
- tokenizer, Qwen, VAE encode, each denoiser/DMD step, VAE decode, readback, and encoding times;
- peak host RSS, Wasm memory high-water mark, and peak/delta GPU memory;
- shader compilation/pipeline-cache state and number of network requests;
- effective linear-layer throughput and attention throughput for representative shapes.

Native timings use backend synchronization around measured GPU regions. Browser timings use
timestamp queries when exposed, otherwise queue-completion wall time and clearly label it as such.
CPU submission time is not presented as GPU execution time.

Native `burn-wgpu` builds enable CubeCL autotuning; Wasm builds deliberately do not. A native
report must include the first-call tuning/compilation cost in the cold sample and measure warm
samples only after the selected kernels have been cached in that process.

Two stable, CPU-only microbench entry points catch local regressions in bounded artifact hashing
and the DMD update algebra without claiming model throughput:

```bash
cargo bench -p burn_image --bench artifact_verification
cargo bench -p burn_boogu --features ndarray --bench dmd_update
```

Their stdout is informational in ordinary CI. Release performance claims continue to come from the
synchronized real-checkpoint runners and reviewed machine report below.

## Current native evidence

The original source-bound native WGPU comparison on the RTX PRO 6000 used sealed artifacts,
`native-high-vram` residency, one cold request, and five warm requests in the same process. The
reported GPU peak is the maximum PID-scoped `nvidia-smi` sample. Upstream used one warmup followed
by five synchronized BF16 samples; its peak is PyTorch allocator accounting and is therefore not
directly comparable to the PID-scoped Burn peak.

| Runtime/task/profile | Shape | Cold inference or load | Warm wall p50 / p95 | Peak |
| --- | ---: | ---: | ---: | ---: |
| Burn Turbo mixed F16 | 256x256 | `75.2976 s` cold | `0.845361 / 0.872075 s` | `42,748 MiB` |
| Burn Turbo Q8 storage profile | 256x256 | `83.7512 s` cold | `2.206708 / 2.211212 s` | `35,452 MiB` |
| Burn Turbo mixed F16, original baseline | 1024x1024 | `177.8581 s` cold | `10.073344 / 10.705904 s` | `44,533 MiB` |
| Upstream Turbo BF16 | 1024x1024 | `22.9234 s` load | `1.731209 / 1.739817 s` | `39,687,443,968 / 42,832,232,448 B` |
| Burn Edit mixed F16, original baseline | 1024x1024 | `124.7204 s` cold | `11.655912 / 11.693669 s` | `46,449 MiB` |
| Upstream Edit BF16 | 1024x1024 | `15.8174 s` load | `2.099106 / 2.102224 s` | `39,688,911,872 / 43,115,347,968 B` |

The original released-shape Burn warm p50 was `5.82x` upstream for Turbo and `5.55x` upstream for
Edit. These historical gaps remain the starting baseline. An intermediate portable-WGPU
optimization preserved the verified F16 VAE tensors and used an explicit 512-row denoiser query
tile:

| Task | Warm wall | Warm stage medians (Qwen / VAE encode / DMD / VAE decode) | Upstream p50 | Ratio | Peak |
| --- | ---: | --- | ---: | ---: | ---: |
| Turbo | `4.148876 s` (one warmed candidate sample) | `75.955 / n/a / 3,625.568 / 425.293 ms` | `1.731209 s` | `2.3965x` | `44,149 MiB` across the tile sweep |
| Edit | `4.781508 s` p50 (`4.835903 s` p95) | `181.599 / 19.864 / 4,109.803 / 460.115 ms` | `2.099106 s` | `2.2779x` | `46,041 MiB` |

This historical step reduced the original Burn wall median by about `58.8%` for Turbo and `59.0%`
for Edit. Its report SHA-256 values are
`1f1d3ce070421849f626a5db8f2ba4a28636382f4066412000d543967243e3d9` and
`ec3080d57c08050f5b500c57469d166f1df2d14d3ddd1cdb52a6659e6e2a8508`.

### Accepted 1K uncached comparison

The finalized native-WGPU stack forces padded blackbox attention with four planes, one 16-row K/V
tile, and one 16-row query tile per plane (`p4/kv1/q1`). For Turbo and 1K Edit it combines balanced
strict Q/K RMSNorm/RoPE preparation, an 8192-row denoiser query bound, the F16 VAE safe
mixed-GroupNorm policy, a 4096-row VAE attention query bound, retained q128 Qwen with
synchronization deferred to its mandatory terminal stage boundary, a resident denoiser, and full
autotune. Native inference additionally overlaps exact host-noise generation with uncached GPU
conditioning and converts F16 decoder output directly to RGB8 without a full F32 host-image
allocation. Wider blackbox query partitions failed the real nonzero gate and fail closed. The
1.5K release remains on its separately qualified composed Q/K path.

The production Bevy factory selects that complete policy, including strict-F32 denoiser RMSNorm,
for Turbo and 1K Edit only when native high-VRAM residency and `f16-qwen-vision-f32` are both
selected. It requires full autotune before
device creation and reports every bound in backend provenance. Browser, layer-streamed, Q8, and
all-F16 paths do not inherit this native qualification.

For the strict comparison below, Burn recomputes conditioning for every measured request, matching
the upstream recomputation policy. Each ratio pairs the same percentile rather than dividing both
Burn percentiles by one upstream median.

| Task | Burn p50 / p95 | Upstream p50 / p95 | Burn/upstream p50 / p95 | Two-times result |
| --- | ---: | ---: | ---: | --- |
| Turbo | `3.400695447 / 3.494256796 s` | `1.731208596 / 1.741948438 s` | `1.964348x / 2.005947x` | p50 passes by `61.722 ms`; p95 misses by `10.360 ms` |
| Edit | `3.908315376 / 3.933613488 s` | `2.099105709 / 2.102803962 s` | `1.861895x / 1.870652x` | passes by `289.896 / 271.994 ms` |

Percentiles use nearest rank over five synchronized warm requests on both sides; Burn follows one
cold request in the same process. The warm
Turbo samples in ascending order were `3.388378764`, `3.394506076`, `3.400695447`,
`3.466112184`, and `3.494256796 s`; Edit was `3.858876513`, `3.875007640`, `3.908315376`,
`3.933095656`, and `3.933613488 s`. Relative to the original Burn medians, the release-replay p50
is `66.2%` lower for Turbo and `66.5%` lower for Edit. Both tasks meet `<= 2x` at p50; Edit also
passes p95. Turbo's nearest-rank p95 is at the noise boundary: this canonical replay misses by
`10.360 ms`, while an independent exact-policy repeat measured `3.449805839 s` or `1.980429x`.
The upstream JSON summary fields use linear-interpolated p95; this table recomputes nearest rank
from their five raw runs, yielding `1.741948438 s` for Turbo and `2.102803962 s` for Edit.

The warm stage p50/nearest-rank p95 values were:

| Task | Processing | Qwen | VAE encode | Four-step DMD | VAE decode | Output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Turbo | `0.416 / 0.689 ms` | `51.939 / 107.358 ms` | n/a | `3,107.201 / 3,178.810 ms` | `227.682 / 239.785 ms` | `15.582 / 19.347 ms` |
| Edit | `2.897 / 3.079 ms` | `116.770 / 149.810 ms` | `9.610 / 15.374 ms` | `3,521.836 / 3,573.403 ms` | `211.520 / 235.095 ms` | `15.919 / 18.350 ms` |

The release-validated performance JSON SHA-256 values are
`537344f26f342fe40142d631a0668bae2820b71eeb2dfefdee5a55e4f9bf275d` for Turbo and
`9a3659dd6356d105b4c4a7477f2c0dad8e06d8d6185df7541fcc26ce37d43433` for Edit. Their RGB PNG
hashes are `7bec793273064d9737925f7fc9889aa37e2c3b47e7d7bcbb132de20a0802121b` and
`b7a0f471832ef9cd1919c28c0f86f16782bbd9f71955f7a6eeed50c66e2fd719` respectively.

### GPU utilization evidence

The accepted repeat-six runs sampled PID framebuffer use together with board-level NVIDIA
telemetry at about 4 Hz. Restricting the summary to samples where the Burn PID held framebuffer
memory and board GPU utilization was at least 90% gives:

| Task | Hot samples | Mean GPU / memory-controller utilization | Mean / peak power | PID framebuffer peak | Board framebuffer peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| Turbo | `103` | `99.82% / 10.19%` | `338.03 / 453.29 W` | `44,129 MiB` | `45,724 MiB` |
| Edit | `109` | `99.81% / 9.88%` | `349.98 / 450.89 W` | `46,045 MiB` | `47,580 MiB` |

These samples demonstrate compute-busy GPU intervals and substantial device residency; they do
not prove exact per-kernel occupancy. Utilization and power are board aggregates, so desktop or
remote-rendering work on the same adapter can contribute. The low memory-controller percentage
relative to GPU utilization is consistent with the remaining DMD path being compute/dispatch
bound, not evidence of a CPU tensor-backend fallback. The telemetry SHA-256 values are
`5cd1a775e76727e2fcf56781836ebe3370df3bb79a80536a5d1f5de59d7e402b` for Turbo and
`cbbc3295e5b85b0d336ecb766edbb5c7273d06c498498c79c3d1231a5d957ffe` for Edit.

A post-tail-optimization Turbo replay retained the exact PNG SHA-256
`7bec793273064d9737925f7fc9889aa37e2c3b47e7d7bcbb132de20a0802121b`. During that session the
adapter already carried `10-25%` unrelated desktop utilization while no model PID existed. The
sampled run still averaged `99.67%` board GPU utilization over 110 hot model samples, but lower
mean/peak power (`322.21/409.68 W`) and globally slower DMD timings produced a contended
`4.019487101/4.045008616 s` p50/p95. A no-sampler confirmation was similarly
`4.034059261/4.055311088 s`. These are retained as contention evidence, not substituted for the
clean comparison table. A same-session composed-Q/K control measured
`4.098237399/4.106558598 s`, confirming that balanced Q/K still saved about `64 ms` at p50.

A separate 1.5K q128/q16384/VAE-q4096 `p4/kv1/q1` repeat-three profile recorded 37 board samples
at or above 90% GPU utilization: mean GPU/memory-controller utilization was `99.89%/11.27%`, mean
power `359.51 W`, peak power `424 W`, and the Boogu PID framebuffer peak was `46,041 MiB`. Its
output hash matches the accepted 1.5K baseline PNG. This trace predates the report schema fields
that explicitly name deferred synchronization and composed Q/K preparation, so it supports GPU
residency/utilization for the 1.5K runtime but is not substituted for its exact-policy parity or
five-warm timing rows. The board/PID telemetry hashes are
`4a54266de00a66a3ebf4c4f48f3d071d6803ec3c93c0a9d37f93aa659daa5c` and
`729e40140a37afda19b54cc3acd10aa5380b4d70dad991ff2f059d6465264890`.

### Historical exact repeated-request conditioning cache

The superseded q1024 policy also measured a single-entry instruction-conditioning cache. Reuse
requires an exact match on prompt, source, and effective token length; a mismatch recomputes Qwen
conditioning. This does not cache Edit VAE reference encoding or DMD/decode work.

| Task | Cached Burn p50 / p95 | Ratio to historical upstream p50 / p95 | PID-scoped peak |
| --- | ---: | ---: | ---: |
| Turbo | `3.396570999 / 3.475443177 s` | `1.96196x / 1.9976x` | `44,129 MiB` |
| Edit | `3.904326915 / 3.906317438 s` | `1.859995x / 1.858183x` | `46,101 MiB` |

Both cached rows are below two-times the corresponding historical upstream percentile, but they
are historical repeated-identical-request evidence, not an apples-to-apples comparison with the
upstream benchmark, which recomputed conditioning, and do not benchmark the accepted q4096 policy.
The cached Edit p50/p95 stage timings were
processing `2.914/3.025 ms`, Qwen cache lookup `0.095/0.312 ms`, VAE encode `12.452/19.400 ms`, DMD
`3,620.143/3,638.008 ms`, decode `233.694/236.433 ms`, and output `17.028/21.010 ms`.

Cached Turbo JSON/PNG SHA-256 values are
`9fa21321eeccfcfbb96775dd191e07bc0fd120926dca664d65e5b6d521cacb5e` and
`ba06424d204e4dfca006eda82d04c7a9ec6b450faff4e34b8de774bd49151e38`. Cached Edit JSON/stderr/PNG/
VRAM-sample SHA-256 values are
`f87c3519b5be3e63a513a3427ace9a2abde7077a69cf4ba28d547662eab09e27`,
`c4cf18101afc01b457f4b9ed7fdcd8b458d3eaafa96e3ecf83c823c11e7b1488`,
`ea84161c0551c1754f7b4350a8fae55ed8345d6cb9c12bbdf5fe666208cd8d17`, and
`9fca1622c793064d39498e106f639d1197ccfa8b26eff65624070fc6d636afb6`.

These are measured performance gaps, not 1024x1024 parity claims. Ordinary `boogu-run` uses its
production RNG rather than fixture-injected upstream noise. The matching optimized 256x256
fixture-backed full-chain gates pass for both tasks. Their release-validated JSON SHA-256 values
are `055548fcbe5bd4e2ff612e57f0038cf194b82f0114e47b3095f925ee0d0fa948` for Turbo and
`746f44d18d66b2e348ce00bc0425777477137e6ba917d0b91d4a46f1a6dcf4f5` for Edit. Both q4096 reports
have empty stderr, pass every gate, and explicitly record `native_release_policy_validated: true`
and balanced strict Q/K preparation. See the
[parity guide](parity.md).

The accepted forced padded-blackbox result is native WGPU-specific. It does not imply browser
WebGPU cooperative-matrix support. A native CUDA timing probe remains excluded because its output
was numerically invalid.

### Edit-Turbo 1.5K validation row

The historical schema-v1 flat mixed-F16 1.5K conversion is sealed under bundle digest
`4e8b12ac5ca95272f9009080a23baf1bc52d1b0e7aebf2e9e5f394a492369213`. The strict verifier covered
249 files, 38,224,723,721 bytes, 223 semantic objects, and 1,940 tensors; its report SHA-256 is
`b702ef29e4a9e26afc67d2fdf8ad46582f561995d6041763439a6ed7ef3ee64f`. This establishes artifact
identity and integrity, not numerical or performance parity for the current schema-v2 modular
closure; its browser replay remains pending.

The 1536x1536 fixture-injected native WGPU full chain subsequently passed its production
execution-dtype gate. Its JSON SHA-256 is
`5834c99b9139ce054891b30e29ecad4bf2d8a4ed1fb9d6482628d1d7350e87ef`; stderr is empty. The
`131.883 s` fresh-process total includes artifact loading and cold compilation/autotune, so it is
not a benchmark sample and does not fill the table below.

| Runtime/configuration | Shape | Burn p50 / p95 | Upstream p50 / p95 | Ratio | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Edit-Turbo 1.5K mixed F16 | 1536x1536 | `12.553919 / 12.830840 s` | `5.968882 / 6.003932 s` | `2.1032x / 2.1371x` | numerical gate passes; `<= 2x` performance target fails |

Both sides used the same prompt/source/seed, `align_res=false`, one unreported warmup/cold request,
five measured warm requests, synchronization, and uncached conditioning. The p95 comparison above
uses nearest rank on both five-sample sets; the upstream JSON additionally records its linearly
interpolated p95 as `6.002870 s`. The target is `<= 2x` at p50 and p95. Cached repeated-request
timing cannot replace this comparison. The ordinary browser UI rejects this variant and there is
no browser 1.5K performance row. A separate no-surface schema-v1 flat-bundle parity replay is
numerical evidence only; the new modular composition still requires its own replay.

The exact released policy used full autotune, retained Qwen q128 with synchronization deferred to
the terminal Qwen-stage barrier, `p4/kv1/q1` denoiser q16384 with strict-F32 RMSNorm and composed
Q/K preparation, VAE q4096
with preserved F16 and safe mixed GroupNorm, and disabled conditioning cache. Its five-warm stage
p50 values were processing `2.136 ms`, Qwen `181.861 ms`, Edit VAE encode `11.052 ms`, four-step
DMD `11,702.996 ms`, VAE decode `580.527 ms`, and output `42.264 ms`. The output is 1536x1536 RGB.
Evidence SHA-256 values are JSON
`19cd7392f51e5e1b2a777c56a7e95b7f1220c6c30bddc16f2271649b159a3644`, progress stderr
`0a344ef6f7bdb773f2212052de905070d9d0fd5334ffcce6e6ced46e79534d23`, and PNG
`b3cb1902e084764cac200787aeea607f828e1cf17d8cffeedf36e1389e92f4cb`.

Deferred Qwen synchronization was independently gated through the complete 1.5K chain: report
SHA-256 `8e78d1dee46898d3f46fad34171e58233d20fe9736817b28b8d1ee5e05a3d9ed`, and the fixture gates
passed. The separate standard-RNG output was not byte-identical to per-stage synchronization:
pixel RMSE was `1.2151/255`, PSNR `46.4385 dB`, SSIM approximately `0.997171`, and maximum channel
difference `65`. Its uncached repeat-six diagnostic measured `12.535297 / 13.006941 s` p50/p95.
The p50 improves modestly, but a first-warm compilation outlier makes this particular nearest-rank
p95 unsuitable as a regression claim. The policy is accepted within the numerical gates and for
lower steady Qwen synchronization cost, not as byte identity or achievement of the still-unmet
1.5K `<=2x` target.

This measured result is retained as the supported runtime's performance evidence, but not as
performance parity: it misses the two-times p50/p95 budgets by `0.616/0.823 s`.
A final safe `p2/kv2/q1` probe regressed to p50/p95 `15.768605/15.905656 s`
(`2.6418x/2.6492x`) and is likewise rejected; JSON SHA-256
`38eb72c120d4a9a18a6dc7e626e70eecaf1f41a9c0eae79e920a7a2bd2780dcb`.
The stock 8-plane attention diagnostic first failed with maximum error `1.3017578`, RMSE
`0.824376`, and non-finite cosine (`NaN`). An isolated CubeK shared-workspace repair later passed
the propagated gate, but its warm p50/p95 regressed to `12.940407/13.052121 s` and the local
dependency override was removed rather than shipped. Its diagnostic benchmark JSON SHA-256 is
`af76039d86c98ff3a4e5d80f96bac20180317f105520ad6b489b66debd67c789`.

Additional parity-safe candidates were kept only when they improved the clean production result.
Host RoPE lookup precomputation is retained: at the 1.5K geometry it replaces about `13.38 million`
trigonometric/power evaluations across four DMD steps with about `71.1 thousand` unique lookups
(`~188x` fewer), is bit-exact against the scalar construction, and passed the full 1.5K chain
(report SHA-256 `9484d2a87182c3357fe4ef988730892a643435757078c2e2091619501d30e2ff`).
A safe single-query-chunk path also avoids an
unnecessary concatenate without changing attention math.

The following diagnostics remain default-off or were reverted: a persistent device RoPE cache
passed parity but regressed 1.5K to `13.8453/14.2948 s`; full fused Q/K RMSNorm+RoPE+GQA passed the
fixture but produced a `13.6063 s` 1.5K median; narrow fused RoPE/GQA padding did not beat the clean
production baseline; balanced Q/K preparation at 1.5K measured `13.6188 s` p50 and therefore is
qualified only for 1K; splitting the double-stream shared projection was effectively neutral
(`13.6226 s` p50 with balanced Q/K). None is silently enabled for 1.5K.

## Performance gates

- No CPU tensor backend may be selected by a WGPU/WebGPU test.
- No whole-checkpoint host copy or whole-response browser `ArrayBuffer` is allowed.
- Warm p50 and p95 are reported separately; at least one warmup and five measured samples are used.
- Every resolution-specific variant must pass its own numerical gate; 1K evidence cannot qualify
  the 1.5K runtime.
- Once a stable baseline and comparison job are checked in, a regression above 10% in a stable
  stage should fail that job unless accompanied by a reviewed baseline update.
- Q8 is retained only where measured end-to-end latency or peak residency improves and its parity
  gate passes. A smaller file alone is not enough.

The native viewer defaults to `native-high-vram`: every required Qwen/VAE stage and the denoiser
are verified and resident before **Ready**, so measured forward execution contains no model-weight
filesystem/decode/upload traffic. `diagnostic-layer-streamed` is an explicit local-only memory
tradeoff whose benchmark must include repeated Qwen, VAE, and denoiser artifact reads and
verification. Browser production now follows the same ready-state boundary with dense-F32 WebGPU
modules; its explicit `layer-streamed-diagnostic` policy is the only ordinary browser mode that
repeats model-weight traffic. Historical streamed-browser numbers do not qualify the resident
default.

The first reviewed native baseline is the
[2026-08-11 RTX PRO 6000 report](reports/2026-08-11-rtx-pro-6000.md). It records a real performance
gap rather than asserting parity. Raw machine-readable reports live outside Git by default under
`docs/reports/raw/`; compact reviewed summaries may be checked in under `docs/reports/`.
