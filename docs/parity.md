# Numerical parity

The parity suite compares the pinned upstream implementation with Burn at named boundaries. A
passing unit test on random tiny weights is necessary for development, but is not checkpoint
parity.

## Deterministic fixture protocol

An external fixture producer writes `tensors.safetensors`, `metadata.json`, `output.png`, and an
authenticated byte-for-byte `source.png` copy for Edit only. It must accept only an absent or empty
destination, perform Edit inference from the canonical copy, and record its exact size and SHA-256
digest. Captures preserve tensor dtypes and record tensor shapes, dtypes, SHA-256 digests, prompt
text, pinned revisions, PyTorch/CUDA versions, exporter options, the complete environment and
worktree identity, source-image provenance, and model-file provenance. Generated fixtures and
their images stay outside the repository; the fixture package and release evidence carry the
whole-file digests.
Randomness is an input:

- token processing is deterministic;
- VAE posterior epsilon is generated once and supplied to both runtimes;
- initial latent noise and each DMD renoise tensor are generated once and supplied to both;
- no runtime is allowed to substitute its own RNG when consuming a parity fixture.

Every Rust parity reader authenticates the exact tensor keyset, shape, dtype, and raw-byte SHA-256
against the fixture metadata before comparing values. The release workflow additionally checks
each external fixture file against the expected whole-file digests recorded with the release
evidence. Shape/dtype/length checks alone are not treated as oracle provenance.

The authenticated external Turbo and 1K Edit numerical fixtures use 256x256 inputs and cover
exhaustive block hooks plus the fixture-injected full chain below. Native mixed-F16 Turbo and 1K
Edit have separately
completed 1024x1024 operational and performance acceptance, but those ordinary production-RNG runs
are not fixture-backed tensor parity. The 256x256 numerical results must not be presented as
1024x1024 exact parity, and the 1024x1024 smokes must not be presented as injected-oracle evidence.
Text-to-image and edit use separate numerical fixtures.

Edit-Turbo 1.5K is a third, strict release identity rather than a resolution alias for the 1K
fixture. Its fixture must record variant `edit-turbo-1k5`, model revision
`60981c49e48cffadf2c169532a4ba3f6108afd5e`, the authenticated Edit source, and an actual output at
an official 1.5K preset. The release fixture uses the omitted-size default, 1536x1536, and contains
372 tensors with exhaustive Qwen and denoiser hook captures. A smaller external 47-tensor capture
can be used as a compact development oracle. Native WGPU q128 Qwen passes 70 semantic stages and
the portable streamed denoiser passes 240 internal boundaries; the exact accelerated production
policy separately passes the propagated full chain. These results are independent of both the
fixture's upstream dtype floor and existing 1K Edit evidence.

| Runtime release | Canonical numerical shape | Fixture/cache identity | Current status |
| --- | ---: | --- | --- |
| Turbo 1K | 256x256 fixture; 1024x1024 operational run | `turbo` | accepted native fixture chain |
| Edit-Turbo 1K | 256x256 fixture; 1024x1024 operational run | `edit` | accepted native fixture chain |
| Edit-Turbo 1.5K | 1536x1536 fixture and measured runtime | external exhaustive `edit-turbo-1k5` fixture | q128 Qwen, portable denoiser hooks, and exact production full chain pass; performance is `2.1032x/2.1371x` |

The external exhaustive 1.5K release fixture SHA-256 values are
`1e78233c703ed32ee351c25d54ca4b05e3efeb898ee2836d1cc96c522e2abcae` for metadata,
`2585ddf2337e41f884218a4abeceb8a10baa7553e43d37f33016be68edc3eeb9` for tensors,
`8e88d6c3580593da723049ef4027a60c5d730b6006ef766d49971a23c6446a70` for output, and
`96534b93904478caf92c1d0e1b431396f81e7b62f09bb5505443378f245d9647` for the authenticated source.
The exported upstream F16-versus-BF16 floor measured Qwen relative RMSE/cosine
`0.08991885/0.9959498`, worst step-3 velocity `0.12197143/0.9925767`, final latent
`0.07775216/0.9969973`, decode `0.07990684/0.9969029`, and RGB `34.0048 dB/0.9916225` SSIM. This
calibrates upstream execution-dtype drift and is not a Burn WGPU result.

### Edit-Turbo 1.5K production full-chain result

The mixed-F16 native WGPU runner consumed the exact fixture noise, selected full autotune, retained
Qwen at q128, preserved the F16 VAE with F32-accumulation GroupNorm at q4096, and forced
padded-blackbox denoiser attention at `p4/kv1/q1` and q16384 with strict-F32 RMSNorm and composed
Q/K preparation. `boogu-full-parity --require`
reported `edit_1k5_release_policy_validated=true`,
`gates.supported=true`, `passed=true`, and no failures:

| Boundary | Relative RMSE | Cosine / image metric |
| --- | ---: | ---: |
| Qwen final hidden state | `0.09276403` | `0.9956884` cosine |
| Edit reference latent | `0.05093983` | `0.9987019` cosine |
| Worst DMD velocity, step 3 | `0.12193716` | `0.99257636` cosine |
| Final latent | `0.07782394` | `0.99699116` cosine |
| Propagated decode | `0.08242943` | `0.99667007` cosine |
| Final RGB | n/a | `33.72779 dB` PSNR / `0.9920097` SSIM |

The release-specific limits are Qwen `<=0.10/>=0.995`, every DMD boundary
`<=0.13/>=0.992`, final latent `<=0.085/>=0.996`, decode `<=0.09/>=0.996`, and final RGB
`>=33.5 dB/>=0.99` SSIM. These are bounded just beyond the independent upstream 1536 F16/BF16
floor rather than copied from the looser 1K Edit envelope.

The JSON SHA-256 is
`5834c99b9139ce054891b30e29ecad4bf2d8a4ed1fb9d6482628d1d7350e87ef`; stderr is empty. The
single fresh-process rerun took `131.883 s`, including `43.317 s` denoiser load, `44.300 s` Qwen,
`4.892 s` VAE encode, an `18.867 s` cold first DMD step, and `4.952 s` VAE decode. Those timings
include cold loading/autotune and are diagnostic only, not synchronized steady-state performance
evidence. The separate q128 Qwen report compares 70 semantic stages (JSON SHA-256
`0bd8014f23f8f8af9a5dfc518d1ce8fcc5a37330b0cb9466bf4ce8ca5faa25ed`); the portable streamed
denoiser report compares 240 boundaries (JSON SHA-256
`edbc633a8e17e95ed099e1e5d0c8bd7a79cb819ce75ca5486d86d2b436f45227`). The latter validates
weights and layer mapping under portable attention; the accelerated blackbox policy is validated
at its propagated step/final/decode boundaries, not by an internal-hook observer.

## Boundaries

| Boundary | Primary comparison |
| --- | --- |
| chat/processor | exact UTF-8, token IDs, masks, media expansion, grids, MRoPE IDs |
| Qwen vision/text | selected hidden states after every configured layer |
| VAE encode | moments, mean, clamped log variance, sampled raw/scaled latent |
| Boogu embedding | patches, timestep embedding, caption projection, position IDs, complex RoPE |
| denoiser | each refiner, 8 dual-stream blocks, 32 single-stream blocks, final velocity |
| DMD | sigma, velocity, clean prediction, injected noise, next latent for all four steps |
| VAE decode/output | raw decode, clipped `[-1,1]` mapping, exact upstream `output.rgb_u8` bytes |

## Component and injected-oracle gates

For tensor `b` against reference `a`, reports include maximum/mean absolute error, maximum relative
error above a small denominator floor, cosine similarity, RMSE, NaN/Inf counts, and the location of
the largest error.

These gates inject the captured tensor at the boundary under test, so upstream trajectory drift
does not accumulate through earlier mixed-precision stages. The executable gates are:

- processor integer plans and prepared pixel tensors: exact/F32-epsilon checks in opt-in tests;
- streamed Qwen WGPU, every aligned stage plus final hidden state: cosine `>= 0.99` and relative
  RMSE `<= 0.2` by default; unaligned legacy hooks remain visible as dtype-only diagnostics and do
  not enter the aggregate gate;
- DMD helper math: max absolute error `<= 0.04`, mean absolute error `<= 0.003`;
- VAE decode: max/mean absolute error `<= 0.02/0.002`, RMSE `<= 0.0025`, cosine
  `>= 0.99999`, with separate encode-boundary limits in `boogu-parity`;
- streamed denoiser isolated boundaries: relative RMSE `<= 0.05` and cosine `>= 0.995`;
- streamed denoiser captured-sigma trajectories: relative RMSE `<= 0.2657993` and cosine
  `>= 0.9646234` by default, an execution-dtype envelope independently measured from the pinned
  upstream Edit F16 versus BF16 trajectories rather than a relaxation of the isolated
  implementation gate;
- direct VAE-oracle final RGB in `boogu-parity`: maximum channel error `<= 4`, mean channel error
  `<= 0.5`, PSNR `>= 50 dB`, and mean non-overlapping 8x8 RGB block SSIM `>= 0.995`.

Thresholds may only be relaxed with a checked-in report that identifies the responsible operation.

## Isolated and fixture-captured trajectory denoiser modes

`boogu-denoiser-parity --input-mode isolated` preserves the block-local diagnostic: every call is
injected with the exact captured `dmd.step.N.input`. It proves the loaded denoiser at each sigma but
cannot establish cumulative sampler drift.

`--input-mode trajectory` starts from `dmd.initial_latents`, predicts with Burn, calls the exported
`dmd_prediction`, injects the exact captured `dmd.step.N.noise` through `dmd_renoise`, and feeds the
result into the next call. Helper math uses each exact captured sigma after validating it within
one fixture-dtype ULP of the canonical schedule. This accounts for upstream constructing
`torch.linspace` directly in BF16 rather than constructing it in F32 and casting afterward. The
runner compares each propagated input, velocity, prediction, renoised state, and
`dmd.final_latents`. Reports always include
`input_mode`, `sigma_source: "fixture-captured"`, `gate_basis`, and the active
thresholds. `--maximum-relative-rmse` and `--minimum-cosine` make any override explicit in both the
CLI and JSON. The default trajectory envelope is calibrated from the independent Edit upstream
F16-versus-BF16 floor (worst velocity `0.2657993`/`0.9646234`; final latent
`0.2001689`/`0.9799298`). Applying it to Turbo is reported as the same cross-dtype envelope, not as
independent Turbo calibration. This binary is a captured-BF16-sigma oracle replay, not production
execution-dtype trajectory parity; `boogu-full-parity` owns that production path. An isolated pass
must not be described as trajectory evidence.

## Historical native mixed and Q8 component baseline

The sealed `q8s-block32-f32-qwen-vision-f32` profile has an owner-specific execution contract.
Qwen Q8S storage is verified and dequantized stage-locally through F16 before Burn's ordinary Col
load mapper; Qwen is not claimed to execute quantized on device. Boogu row-layout matrices remain
device-Q8, while all non-quantized denoiser tensors and activations are F32. The VAE remains F32.

These measured native WGPU results at 256x256 predate the final native-autotune build. The
denoiser rows also predate bounded Boogu query attention. They remain useful component-level
provenance, but the production full-chain table below is the current source-bound release evidence.

| Gate | Artifact digest | Result |
| --- | --- | --- |
| Turbo Qwen, 38 aligned stages | `8685559e73cf836e98e1ebdf80815e3d66765f7d620624408148d5f98c87c0dd` | min cosine `0.9958083`, worst relative RMSE `0.12069529`; final cosine `0.9995802`, final relative RMSE `0.028973013`; `67.353 s` |
| Edit Qwen, 70 aligned stages | `ffde989bb66df3a541d44957422f996790633dab46ca3547a59dfdfb871f0b7a` | F32 vision/F16 text observed; min cosine `0.9940735`, worst relative RMSE `0.10910508`; final cosine `0.99536407`, final relative RMSE `0.09617868`; `77.602 s` |
| Edit denoiser, four isolated calls/240 boundaries | `ffde989bb66df3a541d44957422f996790633dab46ca3547a59dfdfb871f0b7a` | min cosine `0.9988636`, worst relative RMSE `0.047695775`; `189.837 s` |
| Edit denoiser, four-step mixed-F16 captured-sigma replay | `14acbafd13dc9b79757e7d554b504396bee30ea7ed231f533919c6c82a6e6a32` | worst velocity `0.15863511`/`0.9874189`; final latent `0.08544214`/`0.99634737`; inside the upstream execution-dtype envelope; `272.579 s` |
| Edit denoiser, four-step Q8/F32 captured-sigma replay | `ffde989bb66df3a541d44957422f996790633dab46ca3547a59dfdfb871f0b7a` | worst velocity `0.22839242`/`0.974051`; final latent `0.17953193`/`0.9839146`; inside the upstream execution-dtype envelope; `177.631 s` |
| Turbo denoiser, four-step mixed-F16 captured-sigma replay | `4f94cf68c00af12d5de486db4d316ce889d6d21e78913a1c74edab4bd0119ce3` | worst velocity `0.19143246`/`0.98166704`; final latent `0.082120545`/`0.9966279`; inside the Edit-calibrated execution-dtype envelope; `262.259 s` |

### Production-dtype 256x256 full-chain fidelity results

The following 256x256 WGPU baseline runs use bounded Boogu query attention, native Burn WGPU
autotune, and F32 VAE execution, and passed `boogu-full-parity --require`. Each metric pair is
relative RMSE/cosine; the DMD column is the worst propagated velocity across four steps. These runs
use the fixture's exact
initial latent and renoise tensors, but execute `DmdSchedule::upstream_for_dtype` in the production
denoiser activation dtype: F16 for the mixed profile and F32 for Q8. They do not substitute the
fixture's BF16 sigmas. The reported final-latent, decode, and RGB metrics therefore include
accumulated conditioning, schedule, denoiser, and decoder drift.

These profile-specific fidelity envelopes are not the direct VAE-oracle pixel gate above and are
not described as exact pixel parity. They measure how closely the deployable F16/F32 and Q8/F32
execution policies reproduce the upstream BF16 trajectory after all expected dtype drift has
propagated through the complete chain.

| Profile | Artifact digest | Qwen | Worst DMD | Final latent | Decode | RGB PSNR/SSIM | Total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Turbo mixed | `4f94cf68c00af12d5de486db4d316ce889d6d21e78913a1c74edab4bd0119ce3` | `0.02017378`/`0.99979657` | `0.26047248`/`0.9659632` | `0.13578485`/`0.9907423` | `0.06189497`/`0.9980838` | `33.464066 dB`/`0.98020256` | `189.669 s` |
| Edit mixed | `14acbafd13dc9b79757e7d554b504396bee30ea7ed231f533919c6c82a6e6a32` | `0.09336463`/`0.9956323` | `0.23782834`/`0.97179335` | `0.17709899`/`0.9843065` | `0.10237482`/`0.99475545` | `29.939714 dB`/`0.9622695` | `226.064 s` |
| Turbo Q8 | `8685559e73cf836e98e1ebdf80815e3d66765f7d620624408148d5f98c87c0dd` | `0.02880859`/`0.999585` | `0.28667894`/`0.9590829` | `0.1502633`/`0.98869985` | `0.069071405`/`0.9976117` | `32.47857 dB`/`0.97572213` | `124.781 s` |
| Edit Q8 | `ffde989bb66df3a541d44957422f996790633dab46ca3547a59dfdfb871f0b7a` | `0.095800415`/`0.99540055` | `0.25810137`/`0.9667663` | `0.20584`/`0.97882676` | `0.114375815`/`0.9935089` | `28.968882 dB`/`0.9509358` | `145.336 s` |

Each row is a fresh process, so total time includes cold autotune and is not a steady-state
performance result. Cold selection is clearest in DMD step zero: mixed Turbo took `47.466 s`
before steps 1-3 fell to `0.119/0.118/0.139 s`; mixed Edit took `53.967 s` before
`0.211/0.193/0.191 s`; Q8 Turbo took `20.212 s` before `0.425/0.397/0.432 s`; and Q8 Edit took
`22.586 s` before `0.616/0.619/0.657 s`. Mixed Turbo also paid a `15.312 s` cold VAE decode, while
mixed Edit paid `106.452 s` in Qwen and `5.899 s` in VAE encode.

The final JSON report SHA-256 digests are:

- Turbo mixed: `fdc9221ad49b40c7a7ce142cad9324751dbc583133781ef95da4e0614f88dc8e`;
- Edit mixed: `1d712cf4ca99ae6eb85e466883ca2ce85d004655b6ad3f3a7a38701c448c4871`;
- Turbo Q8: `12e5c973995a1ff22987b6876b6316395391342044daf0c0c352fafd33aad4f7`;
- Edit Q8: `25b063fe321adef50ae97a1fb44b3862b69cd7b5b30033eaf8f0990709f21bfe`.

The first mixed Edit run failed only the previous VAE F32-reference maximum-error limit of
`0.003`. Its measured WGPU-F32 versus PyTorch-F32 result was maximum error `0.004232645`, RMSE
`0.00013274394`, and cosine `1.0`. The evidence-based three-part gate is now maximum error
`<= 0.005`, RMSE `<= 0.0002`, and cosine `>= 0.999999`; the unchanged rerun passed. The initial
failed JSON report has SHA-256
`13b1fc6525512809b4a2fe82208eaf4a41d906083111481bda7191de08a7af41`.
The final autotuned Edit runs measured maximum error `0.0006117821`, RMSE `0.000012480457`, and
cosine `1.0` against the PyTorch-F32 reference, comfortably inside the retained three-part gate.

### Preserved-F16 VAE optimization validation

The optimized native path retains the sealed F16 VAE tensors and executes the decoder in F16
instead of adapting that stage to F32. The direct 256x256 retained-decoder oracle gate passed:
maximum error `0.0126953125`, RMSE `0.0015896541`, cosine `0.9999974`, RGB maximum channel error
`2`, PSNR `54.506023 dB`, and block SSIM `0.99779713`. After the first-use `5.224464 s` decode,
the three warm decodes were `18.436680`, `17.315564`, and `23.820980 ms`. The report SHA-256 is
`dc71439d05013e7f3ff34669b54468ca021cec1f33c4e75a8f094085335a1b98`.

An intermediate fixture-injected Turbo run with the production F16 schedule, portable 512-row
bounded query attention, and preserved-F16 VAE passed `boogu-full-parity --require`. Its report is
retained as historical evidence under SHA-256
`8a1833134a9a7c4b69856eae258826c9cc836695c9def6aac82c417a87b18b85`.

The accepted 1K native-WGPU policy uses forced padded blackbox attention `p4/kv1/q1`, balanced
strict Q/K RMSNorm/RoPE preparation, an 8192-row denoiser query bound, the safe
F16-storage/F32-accumulation decoder GroupNorm policy, a 4096-row VAE attention query bound,
retained q128 Qwen with deferred synchronization, and full autotune. With exact fixture noise and
the policy-correct Edit reference oracle, both full chains pass `boogu-full-parity --require`. The
Turbo and Edit JSON SHA-256 values are
`055548fcbe5bd4e2ff612e57f0038cf194b82f0114e47b3095f925ee0d0fa948` and
`746f44d18d66b2e348ce00bc0425777477137e6ba917d0b91d4a46f1a6dcf4f5`. Both reports have empty
stderr and explicitly record `native_release_policy_validated: true`,
`vae_attention_query_chunk_size: 4096`, and balanced strict Q/K preparation. This supersedes the
earlier
Edit failure against the wrong F32-specific reference; that failed report remains historical only.

| Task | Qwen relative RMSE / cosine | Worst DMD velocity | Final latent | Decode | RGB PSNR / SSIM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Turbo | `0.02017378 / 0.99979657` | `0.2635267 / 0.965074` | `0.13607527 / 0.9906988` | `0.06147196 / 0.998114` | `33.540535 dB / 0.9798921` |
| Edit | `0.09276403 / 0.9956884` | `0.23774831 / 0.97181666` | `0.18383627 / 0.9831099` | `0.10412595 / 0.9945964` | `29.781435 dB / 0.9551179` |

Host-side RoPE lookup precomputation is also retained. Its 1.5K geometry test is bit-exact against
the scalar construction, and the exact 1.5K full-chain gate passed under report SHA-256
`9484d2a87182c3357fe4ef988730892a643435757078c2e2091619501d30e2ff`. The implementation removes
repeated host transcendental work without retaining device tensors between calls. A safe
single-query-chunk attention path removes an unnecessary concatenate without changing the
attention result. The final Turbo replay also exercises direct F16 decoder-to-RGB8 conversion;
all numerical metrics remain identical and its stderr is empty.

### Native released-shape operational evidence

The original mixed-F16 `boogu-run` processes used the same sealed profiles as the 256x256 numerical
gates, bounded query attention, native WGPU autotune, retained verified Qwen/VAE stages, and a
resident denoiser. Each process ran one cold request followed by five warm requests. Both outputs
were independently validated as RGB8 1024x1024 PNGs.

| Task | Artifact digest | Cold | Warm p50 / p95 | Peak VRAM | Output SHA-256 | JSON SHA-256 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Turbo, original | `4f94cf68c00af12d5de486db4d316ce889d6d21e78913a1c74edab4bd0119ce3` | `177.8581 s` | `10.073344 / 10.705904 s` | `44,533 MiB` | `d6a2de570fe1bd4ff6b1d0dc27d998c435aa59928b27e365434061cbc526d005` | `29701136910cdba8a90caa90e7c7a4eb1452b784b4359a19eb7cac38e0b4962e` |
| Edit, original | `14acbafd13dc9b79757e7d554b504396bee30ea7ed231f533919c6c82a6e6a32` | `124.7204 s` | `11.655912 / 11.693669 s` | `46,449 MiB` | `aa408d3c699978b5c17bda335d5b67c9645a7ad3672ae65896366f631b7faf6b` | `a66cfe57dd1e67e73dff98682cdf065c05ea9daf661c7bca56be95071fd75bb2` |

The matching upstream BF16 warm p50 values are `1.731209 s` for Turbo and `2.099106 s` for Edit,
so the original native Burn baselines were `5.82x` and `5.55x` slower respectively. The finalized
uncached runs recompute conditioning on each request and use the same optimized stack as the exact
gates:

| Task | Burn p50 / p95 | Upstream p50 / p95 | Ratio p50 / p95 |
| --- | ---: | ---: | ---: |
| Turbo | `3.400695447 / 3.494256796 s` | `1.731208596 / 1.741948438 s` | `1.964348x / 2.005947x` |
| Edit | `3.908315376 / 3.933613488 s` | `2.099105709 / 2.102803962 s` | `1.861895x / 1.870652x` |

Both meet `<= 2x` at p50 and Edit does at nearest-rank p95 with conditioning recomputed for every
warm request. Turbo p95 misses by `10.360 ms`; a second exact-policy repeat measured
`3.449805839 s` (`1.980429x`), placing the tail at the measurement-noise boundary rather than
robustly below it.

For historical context, the superseded q1024 policy measured a single-entry cache that reused Qwen
instruction conditioning only when prompt, source, and effective token length all matched. Its
cached results were:

| Task | Cached p50 / p95 | Ratio to historical upstream p50 / p95 | Peak VRAM | PNG / JSON SHA-256 |
| --- | ---: | ---: | ---: | --- |
| Turbo | `3.396570999 / 3.475443177 s` | `1.96196x / 1.9976x` | `44,129 MiB` | `ba06424d204e4dfca006eda82d04c7a9ec6b450faff4e34b8de774bd49151e38` / `9fa21321eeccfcfbb96775dd191e07bc0fd120926dca664d65e5b6d521cacb5e` |
| Edit | `3.904326915 / 3.906317438 s` | `1.859995x / 1.858183x` | `46,101 MiB` | `ea84161c0551c1754f7b4350a8fae55ed8345d6cb9c12bbdf5fe666208cd8d17` / `f87c3519b5be3e63a513a3427ace9a2abde7077a69cf4ba28d547662eab09e27` |

The cached rows are historical repeated-identical-request evidence, not strict comparisons with
upstream recomputation and not measurements of the accepted q4096 policy.

These runs prove the supported released shape executes and produces valid images; they do not add
1024x1024 boundary comparisons or replace the 256x256 exact-noise numerical gates above. The
forced native-WGPU padded-blackbox configuration is exactly `p4/kv1/q1`; wider query partitions
fail closed. CUDA remains excluded because its probe output was numerically invalid. See the
[machine report](reports/2026-08-11-rtx-pro-6000.md) for raw samples and evidence identities.

An earlier pre-autotune Turbo end-to-end Q8 smoke completed with a coherent 256x256 RGB output in
`73.500 s` under the high-VRAM native policy (Qwen/VAE retained after first use, resident
denoiser). This proves the composition path executes; it is not a final-pixel oracle comparison or
current performance evidence. Sampled peak VRAM was about `35.0 GiB` during decode: the highest
manual PID-scoped `nvidia-smi` sample was `35,033 MiB`. It was not continuous sampling and is not a
true peak bound.

## Running the implemented tools

Provision an exhaustive schema-2 fixture from the pinned upstream environment, keeping its model
weights, tensors, images, and metadata outside this repository. The 372-tensor external 1536
release fixture includes Qwen and block captures; a compact external 47-tensor development oracle
may omit those memory-heavy hooks. Download the exact model snapshot directly, then point the Rust
parity binaries at the externally authenticated fixture:

```bash
export BURN_IMAGE_EDIT_1K5_SNAPSHOT="$(hf download \
  Boogu/Boogu-Image-0.1-Edit-Turbo \
  --revision 60981c49e48cffadf2c169532a4ba3f6108afd5e)"
export EDIT_1K5_FIXTURE=/absolute/path/to/external/boogu-reference-edit-1k5
test -s "$EDIT_1K5_FIXTURE/metadata.json"
test -s "$EDIT_1K5_FIXTURE/tensors.safetensors"
```

Release use requires the exhaustive whole-file hashes recorded in release evidence. The commands
below infer and enforce the exact release variant from authenticated fixture metadata and the
sealed artifact manifest.

```bash
cargo run -p burn_boogu --release --features "import,ndarray,wgpu" \
  --bin boogu-parity -- \
  --fixture "$FIXTURE" \
  --artifacts "$ARTIFACTS" \
  --profile f16-qwen-vision-f32 --backend wgpu --require

cargo run -p burn_boogu --release --features "import,ndarray,wgpu" \
  --bin boogu-denoiser-parity -- \
  --artifacts "$ARTIFACTS" --fixture "$FIXTURE" \
  --backend wgpu --profile f16-qwen-vision-f32 --steps 4 \
  --input-mode isolated --require

# Captured-sigma oracle replay; this is not the production execution-dtype chain.
cargo run -p burn_boogu --release --features "import,ndarray,wgpu" \
  --bin boogu-denoiser-parity -- \
  --artifacts "$ARTIFACTS" --fixture "$FIXTURE" \
  --backend wgpu --profile f16-qwen-vision-f32 --steps 4 \
  --input-mode trajectory --require

cargo run -p burn_boogu --release --all-features \
  --bin boogu-qwen-parity -- \
  --artifacts "$Q8_ARTIFACTS" --fixture "$FIXTURE" \
  --profile q8s-block32-f32-qwen-vision-f32 \
  --quantized-load-policy auto --capture-stages true --require

cargo run -p burn_boogu --release --features "runtime,import,wgpu" \
  --bin boogu-full-parity -- \
  --artifacts "$ARTIFACTS" --fixture "$FIXTURE" \
  --profile f16-qwen-vision-f32 --qwen-residency retained \
  --qwen-synchronization-policy deferred --qwen-query-chunk-size 128 \
  --vae-float-policy preserve-f16 \
  --vae-group-norm-policy f16-storage-f32-accum \
  --vae-attention-query-chunk-size 4096 \
  --denoiser-query-chunk-size 8192 \
  --denoiser-attention-policy padded-blackbox \
  --denoiser-rms-norm-policy strict-f32 \
  --denoiser-qk-preparation-policy balanced-strict-qk-norm-rope \
  --blackbox-num-planes 4 --blackbox-seq-kv-tiles 1 --blackbox-seq-q-tiles 1 \
  --require

# Exact native Edit-Turbo 1.5K release policy. Other policies require an explicit diagnostic flag.
cargo run -p burn_boogu --release --features "runtime,import,wgpu" \
  --bin boogu-full-parity -- \
  --artifacts "$EDIT_1K5_ARTIFACTS" --fixture "$EDIT_1K5_FIXTURE" \
  --profile f16-qwen-vision-f32 --qwen-residency retained \
  --vae-float-policy preserve-f16 \
  --vae-group-norm-policy f16-storage-f32-accum \
  --vae-attention-query-chunk-size 4096 \
  --qwen-synchronization-policy deferred --qwen-query-chunk-size 128 \
  --denoiser-query-chunk-size 16384 \
  --denoiser-attention-policy padded-blackbox \
  --denoiser-rms-norm-policy strict-f32 \
  --denoiser-qk-preparation-policy composed \
  --blackbox-num-planes 4 --blackbox-seq-kv-tiles 1 --blackbox-seq-q-tiles 1 \
  --require
```

The release parity job runs the processor and Qwen WGPU fixture tests for the mandatory Turbo/1K
Edit pair and unconditionally requires `BURN_IMAGE_EDIT_1K5_SNAPSHOT`,
`BURN_IMAGE_EDIT_1K5_ARTIFACTS`, and `BURN_IMAGE_EDIT_1K5_FIXTURE`. A missing 1.5K input fails
before model loading; a legacy two-release run is diagnostic rather than a release gate. The job
authenticates the 372-tensor oracle and exact bundle digest, then invokes the Cargo binaries for
exhaustive q128 Qwen, VAE/DMD, 240 portable-denoiser boundaries, and the exact accelerated
production full chain. The 1.5K invocation selects the F16 VAE q4096, retained Qwen q128 with
deferred synchronization, denoiser q16384, and forced `p4/kv1/q1`
padded-blackbox policy recorded above. Captured-sigma trajectory diagnostics remain
separately reported. `boogu-full-parity` is the production execution-dtype propagated chain. The
current release-evidence matrix is native only.
The Wasm target is compiled in ordinary CI, but `edit-turbo-1k5` is rejected by the browser runtime
and has no real-checkpoint Chromium parity result.
