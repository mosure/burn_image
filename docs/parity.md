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
| Turbo 1K | 256x256 fixture; 1024x1024 operational run | `turbo` | accepted native high-VRAM fixture chain; native low-VRAM 1024 output-qualification candidate passes artifact/output/memory checks but is not fixture parity; ordinary rendered browser low-VRAM 1024 passes its model/output/surface/memory gate and same-seed quality floor, while the serialized Run C pass is diagnostic-only and exact-noise full-chain parity remains pending |
| Edit-Turbo 1K | 256x256 fixture; 1024x1024 operational run | `edit` | accepted native high-VRAM fixture chain; browser full-resolution and low-VRAM reruns pending |
| Edit-Turbo 1.5K | 1536x1536 fixture and measured runtime | external exhaustive `edit-turbo-1k5` fixture | native high-VRAM and current-source low-VRAM exact chains pass on the historical flat artifact; the first current modular browser low-VRAM computation passed internally but failed the stale host-key contract, and its corrected source-bound canonical rerun now passes the exact no-surface numerical/memory gate; `qualification-f32` is optional and non-blocking |

The current ordinary browser Turbo runtime preloads 46 stages / 106 objects / 912 canonical F16
tensors before **Ready**, retaining 19,870,010,624 padded packed-F16 bytes. It does not runtime-
quantize those weights. Every DMD step widens one semantic stage at a time on device and uses
dense-F32 matmul; the four-step contract requires exactly 184 stage materializations, 424 object
unpacks, 79,480,042,496 packed bytes read, 158,960,084,992 F32 bytes written, and zero DMD
artifact/cache/network traffic.

The packed cache is request-scoped rather than warm GPU residency. After DMD, an exact synchronized
F32 latent handoff clears every packed arena and proves an empty cache before VAE decode. The next
request must rehydrate all 106 denoiser objects from the integrity-checked persistent range cache.
Initial preload is 19,870,166,528 authenticated bytes / 4,780 ranges. The first Generate request
then reads 80 Qwen-text/VAE-decoder objects / 15,235,984,896 bytes / 3,709 ranges; the second reads
186 objects / 35,106,151,424 bytes / 8,489 cache-hit ranges and permits zero network responses. The
static plan records 22,304,263,424 preload bytes and 26,492,170,880 conservative inference bytes,
but the exact-size persistent Qwen text-layer pool makes the measured aggregate GPU-memory gate
mandatory.

The manual hardware workflow additionally requires a distinct exhaustive 1024x1024 Turbo BF16
fixture. Its metadata, tensors, and output identities are respectively
`a7cf73b0ea0183d58b25f5c41eb732c28ed7a0aef52465365387d84cf2af0758`,
`eb3a81e7285f25df69a4e20a9f7d71d318bf9ccc5f84f2819c38fc2c1311f40e`, and
`4abd717984140ace64143617f1981025917c1f35ceb2271501880b350961d703`. The current browser first-DMD
diagnostic authenticates that fixture, injects its exact Qwen/DMD inputs, executes exactly one
packed-F16/dense-F32 prediction, requires finite reported metrics and zero DMD
artifact/cache/network traffic, and uploads its report or failure record. It deliberately has no
calibrated numerical threshold and does not turn a diagnostic pass into full-resolution parity.
The final-source hardware run passed with outcome `diagnostic-passed-no-full-parity-claim`. It binds
JavaScript SHA-256 `64197f892ae850d901a9b76ff70dba7f543fa70af02028f605bb9eb126dc1b37`, WebAssembly SHA-256
`001f0bcc93fbaeea9a9b32d2adcb8b46f1897b80d36386919c03d69869dca86b`, probe/harness SHA-256
`d6485fa204233b25c1d12128410ae162a8d1ce59053179d3f31a3db63155dd88`, contract SHA-256
`dadd84a4ef9c5162c4aea7f3251cb40461e08d969bfaada3ae99f94dc6fb4b86`, report SHA-256
`0a600471ec9e3119eeaebd616e9dd29a84c62067881b6f9834949019f92d5eab`, and console SHA-256
`20af8aa43d3a53608c0658fabbef0fb8d7e85f2ff4b9655736fc74b959d489fe`. Its exact cache inventory
was 46 stages / 106 objects / 912 tensors; one prediction performed 46 stage materializations and
106 object unpacks, read 19,870,010,624 packed bytes, wrote 39,740,021,248 dense-F32 bytes, and made
zero DMD artifact/range/cache/network traffic. Velocity relative RMSE / cosine were `0.03869645` /
`0.9992708`, and prediction relative RMSE / cosine were `0.042713966` / `0.99910367`. `/dev/shm`
passed quota-aware admission, and process-group exit, Chrome-profile removal, and artifact-server
teardown were clean. This is one-prediction diagnostic stability evidence only: it makes no
calibrated numerical-correctness, full-chain/full-resolution parity, or fully on-device-quantized
execution claim.

A predecessor core-Q8 run is preserved as historical evidence. It bound JavaScript SHA-256
`7a7004085a273ac2ab2ce7a3150e86cf5d9875d167bc9fcfca140a42d0bce69f` and Wasm SHA-256
`6f6d8e1143bfd3f1c4461450e992193f52bfae438bf4505b62b19068864dc8e6`, authenticated five selected
tensors / 1,941,506 bytes, and produced finite velocity/prediction metrics of `0.039951164` /
`0.044186924` relative RMSE and `0.99920744` / `0.9990256` cosine. Report SHA-256 is
`62e14a0712811e088b33d74d047fa6370c5ae8191920cbbc87313b93fb3e68d0`; its outcome was
`diagnostic-passed-no-full-parity-claim`, with both stronger claim flags false. It does not qualify
the current packed-F16 route.

The distinct headful same-page/same-engine gate requires two valid, distinct 1024 PNG downloads,
the initial preload plus second-request 106-object packed-cache rehydration, exact first-request and
all-cache-hit repeat traffic, packed-cache eviction before each VAE decode, and zero DMD I/O for
both requests. Each request must also suspend primary-window surface acquisition before runtime
submission and restore exact camera state only after its terminal model event and before output
publication. The canonical rerun passed two sequential ordinary requests on one engine, with
exactly one adapter request, one device request, one Chrome GPU process, and packed-cache preload
attempts `[1, 2]`. Request 2 read all 186 objects / 35,106,151,424 bytes / 8,489 ranges as cache
hits, with zero misses and zero network requests or bytes. Both requests completed four zero-I/O
DMD steps, digest-preserving Qwen and DMD-to-VAE handoffs, cache-ready-to-empty cleanup, and one
surface-suspension window with zero gated acquisitions, failures, violations, or overlap and a
successful first acquisition after resume. Peak Chrome GPU-process memory was 24,384,634,880
bytes; page/GPU error lists were empty, the process group exited, and the Chrome profile was
removed. The distinct 1024x1024 PNGs have SHA-256
`5d7a947301c9cefef8bca5cd42db52b98fe3fca440e98d302312eff6afe2eb38` and
`815c553a70a4322aa8e49a51aeb0d46b75ccf2178b435c9b0ba0fedec3da5e0c`; canonical report SHA-256
is `90da22207398ae907e6b0d0bc93881c689a2a7362a1e52aac5435deac525b5d5`. This is an ordinary
same-engine rendered smoke, not numerical parity.

### Browser low-VRAM Turbo 1K rendered output evidence

Two current 1024x1024 runs completed on one non-fallback NVIDIA Blackwell BrowserWebGPU device.
Serialized Run C intentionally forced the Qwen block-0 synchronization boundary and passed the
complete model, surface, PNG, and measured-memory checks. Its report SHA-256 is
`b0dfcc8e53fd7ad1c4731d3169e2f43c50063aa2b54e5ca6347789e18630c6e6`; the report identity is
diagnostic-only and cannot establish release output quality or numerical parity.

The subsequent ordinary, non-serialized UI run passed with report SHA-256
`36525be1d5ff482c409c3b7484027fcb335340e474e4a95f182720ea3f032a28`. It exercised ordinary
Qwen block 0, all four DMD steps, exact final-latent handoff, packed-cache eviction before VAE,
zero request-window DMD I/O, zero gated texture acquisitions, exact camera restore, and a stable
post-resume surface. The Chrome GPU process peaked at 22,824 MiB / 23,932,698,624 bytes across
748/748 matched intervals, 461 active, with 99% peak SM activity. Page/GPU error lists and packed
lifecycle failures were empty. Its 1,452,562-byte PNG SHA-256 is
`5d7a947301c9cefef8bca5cd42db52b98fe3fca440e98d302312eff6afe2eb38`.

The required same-prompt/seed comparison against the native low-VRAM PNG passed the published
`>=24 dB` PSNR / `>=0.90` mean block SSIM floor at `37.517250061 dB` / `0.985732973`. Quality-report
SHA-256 is `31da8e541013c38dd215257431159a99c7112ad79714a079e2a4b25f9c855103`. Each runtime generated
its own noise, so the report truthfully sets `numerical_parity_claimed=false`; exact-noise
full-chain browser/native parity remains pending.

### Native low-VRAM Turbo 1K output qualification candidate

A cold native WGPU Turbo 1024x1024 generation from the current canonical modular closure completed
with report `ok=true`. The closure bound parent digest
`555019af867a80bb4d7cec5dc2f0ba60ae799071994a5fd24d7e71918cb9ce36` and verified 223 weight
objects, 253 files, 1,940 tensors, and 38,224,723,494 bytes. Its isolated fresh XDG cache produced
six `Loaded 0 autotune cached entries` messages. All 2,246 PID-scoped framebuffer samples matched
the Burn process and were nonzero. Initialization peaked at 24,814,551,040 bytes, Qwen at
27,087,863,808, DMD at 27,096,252,416, and VAE decode at the 27,055 MiB /
28,369,223,680-byte overall peak, strictly below the decimal 32,000,000,000-byte ceiling.

The generated 1024x1024 PNG was 1,448,891 bytes and retained SHA-256
`b2cfbc50f7c8f9d486799abd8c5be90c8770059a1dbc020ad02ac41a91abfab1` exactly across the
low-VRAM allocator change. Total cold-process time was `281.404 s`, of which `212.459 s` was
inference. Report SHA-256 is
`4f67f468110addef18a4d6f27d4ed01ab57f1c3c03de7174e6450fe793d38376`.

The memory change is confined to native low-VRAM: Qwen uploads the sealed embedding directly at
its released F16 dtype, and VAE decode uses exact-size transient allocation plus synchronization
and cleanup before the decoder tail. These are allocator-only, math-neutral policies. The report
does not inject or compare an external reference tensor chain, so it is an output-qualification
candidate rather than numerical or cross-runtime parity. It does not qualify browser Turbo,
another shape, or synchronized warm performance; exact-noise full-chain cross-runtime parity
remains pending.

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

### Native low-VRAM Edit-Turbo 1.5K result

The native phase-resident replay passed the same exact 1536x1536 exhaustive fixture chain while
staying strictly below the decimal 32,000,000,000-byte ceiling. It used streamed q128 Qwen with a
per-stage synchronization boundary, an unretained phase-loaded preserved-F16 VAE at q4096 with
F32-accumulation GroupNorm, and one mixed-F16 q16384 `p4/kv1/q1` denoiser retained across all four
DMD steps. The run selected full autotune, strict-F32 denoiser RMSNorm, and composed Q/K
preparation; no denoiser weight was reloaded inside DMD.

The in-process NVIDIA sampler used a configured 250 ms delay plus each `nvidia-smi` subprocess's
runtime. The current-source run attempted 608 samples, matched the Burn PID with nonzero framebuffer
use in 607, and measured a 28,906 MiB / 30,310,137,856-byte peak. The strict memory gate and every numerical gate
passed. Final RGB was `33.72779 dB` PSNR / `0.9920097` mean 8x8 block SSIM; propagated decode was
`0.08242943` relative RMSE / `0.99667007` cosine. The fresh-process total was `173.546 s`, including
verified loading and cold compilation/autotune, so it is qualification timing rather than a warm
performance result.

The report SHA-256 is
`a013adfcd30b7e6b2323ecc3723b22396f9858d14dd3cdd4a0da2699e199abe3`; stderr is empty. It is
scoped to native Burn WGPU on the reported NVIDIA RTX PRO 6000 Blackwell Workstation Edition,
driver 610.43.02, the external schema-2 BF16 fixture, and historical schema-v1 flat artifact digest
`4e8b12ac5ca95272f9009080a23baf1bc52d1b0e7aebf2e9e5f394a492369213`. It does not qualify the
schema-v2 modular closure, supply Turbo or 1K Edit fixture parity, qualify another shape, or cover
browser WebGPU.

### Browser low-VRAM Edit-Turbo 1.5K result

The 2026-08-14 source-bound schema-v2 modular low-VRAM rerun passed the exact 1536x1536 exhaustive
fixture computation with top-level `ok=true`, `gates.passed=true`, `artifacts_verified=true`,
`fixture_authenticated=true`, and `numerical_parity_claimed=true`. It bound canonical parent digest
`4eb95001708becebeab5bb7417b02003e9dbe704775bb49557b681a5b617fd5a`, resolved its sealed Qwen
and VAE dependencies, verified all 223 executed weight objects, authenticated all 372 fixture
tensors, and compared the exact 355 public semantic tensors.

The request streamed Qwen and selected VAE objects, retained 48 runtime-Q8/F32 denoiser stages
through all four DMD steps, then cleared the cache before decode. Its 942-tensor dtype audit matched
the inventory exactly: 377 Q8S block-32/F32 tensors and 565 F32 tensors, with no unexpected dtype.
The resource plan uses the canonical exact fields: 771,785,728 streamed-Qwen bytes, 335,278,732
loaded-VAE bytes, zero dense-denoiser-stage bytes, a 771,785,728-byte phase-local maximum,
2,434,252,800 runtime-quantization workspace bytes, and 30,402,341,120 conservative planned bytes.
Source dispatch preserved QFloat weights into `q_matmul`; no backend kernel trace was captured, so
`on_device_quantized_execution_claimed=false` remains required.

Every numerical gate passed. Final latent relative RMSE/cosine was `0.07212386/0.997419`,
propagated decode was `0.07496766/0.9972718`, and final RGB was `34.531677 dB` PSNR /
`0.99253726` mean 8x8 block SSIM. Scoped Chrome GPU-process telemetry matched all 443 sample
intervals, 224 with positive activity, and peaked at 29,828 MiB / 31,276,924,928 bytes with 99%
peak SM activity. That leaves 723,075,072 bytes below the strict decimal cap. Peak Wasm linear
memory was 2,009,137,152 bytes, and end-to-end qualification time was 904.960 seconds.

The canonical report SHA-256 is
`c895ae2c1cba3823afe756035b6e564d5ef27caf3722f5f350c07e23086e3b54`; its page-capture PNG is
7,373,531 bytes with SHA-256
`0e273bf6b0660cdf6f96bbf163f56f0712b446c15e31ba6a95daac9a348c97b7`. The report binds harness
SHA-256 `cee29e844c33325a2dac1e29b3a03f731f61be2b926ade93a1a50f5443b8efd8` and exact contract
SHA-256 `d6a0ff5b8ebe8890be831efd1909cd36e2ede9709dc119fe1d965d4b8aa414ea`.

The first current-source attempt had already passed the same inner artifact, fixture, numerical,
and memory gates, but its immutable outer report was `ok=false` because the host expected retired
field `audited_max_streamed_stage_bytes` instead of the runtime's canonical
`audited_max_streamed_qwen_stage_f32_bytes`. Corrected offline replay of that report has zero
failures, but remains noncanonical; its SHA-256 is
`3dec48ec032c7abd1ffcb9aab1546d81765e97503340f8ea895724dfe1aacd5b`. Only the subsequent
source-bound report above is the promoted pass. Its scope is current modular low-VRAM, exact
1536x1536, no-surface numerical correctness and measured memory on the recorded Chrome/Blackwell
stack—not rendered-window behavior, another 1.5K shape, the explicit resident policy, performance,
the optional F32 control, CDN availability, or cross-stack portability.

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

Viewer and browser users select the canonical storage contract with `--profile production` or
`profile=production`. The parity binaries below deliberately use `f16-qwen-vision-f32`, the exact
sealed manifest identity written into their reports. It means F16 storage for the denoiser, Qwen
text tower, and VAE with an F32 Qwen vision exception; it is not an all-F32 execution profile.

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

# Exact native Edit-Turbo 1.5K release policy over the qualified schema-v1 flat artifact.
# Other runtime policies require an explicit diagnostic flag.
cargo run -p burn_boogu --release --features "runtime,import,wgpu" \
  --bin boogu-full-parity -- \
  --artifacts "$EDIT_1K5_FLAT_PARITY_ARTIFACTS" --fixture "$EDIT_1K5_FIXTURE" \
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

# Qualified native low-VRAM replay: phase-streamed Qwen/VAE, resident mixed-F16 denoiser,
# plus strict sampled process VRAM below decimal 32 GB.
cargo run -p burn_boogu --release --locked --features "runtime,import,wgpu" \
  --bin boogu-full-parity -- \
  --artifacts "$EDIT_1K5_FLAT_PARITY_ARTIFACTS" --fixture "$EDIT_1K5_FIXTURE" \
  --profile f16-qwen-vision-f32 --native-runtime-policy low-vram \
  --qwen-residency streamed --qwen-synchronization-policy per-stage \
  --qwen-query-chunk-size 128 \
  --vae-float-policy preserve-f16 \
  --vae-group-norm-policy f16-storage-f32-accum \
  --vae-attention-query-chunk-size 4096 \
  --denoiser-query-chunk-size 16384 \
  --denoiser-attention-policy padded-blackbox \
  --denoiser-rms-norm-policy strict-f32 \
  --denoiser-qk-preparation-policy composed \
  --blackbox-num-planes 4 --blackbox-seq-kv-tiles 1 --blackbox-seq-q-tiles 1 \
  --require
```

The low-VRAM gate above fails unless `nvidia-smi` captures at least four matched and four nonzero
PID-scoped framebuffer samples and the sampled total remains strictly below 32,000,000,000 bytes.
The 30,971,005,440-byte Edit static plan remains a conservative planning bound rather than a peak;
the current-source qualifying replay measured 30,310,137,856 bytes as reported above.

The release parity job runs the processor and Qwen WGPU fixture tests for the mandatory Turbo/1K
Edit pair and unconditionally requires `BURN_IMAGE_EDIT_1K5_SNAPSHOT`,
`BURN_IMAGE_EDIT_1K5_FLAT_PARITY_ARTIFACTS`, `BURN_IMAGE_EDIT_1K5_FIXTURE`, and
`BURN_IMAGE_CANONICAL_MODULAR_ARTIFACT_ROOT`. The workflow-dispatch inputs remain
`edit_1k5_artifacts` and `modular_artifact_root`, but they map to those deliberately distinct
environment contracts. The first must be the exact qualified schema-v1 flat bundle because
`boogu-full-parity` reads Qwen, VAE, and denoiser stages from one directory. The second must be the
five-entry schema-v2 publication root used by browser and canonical release verification. The job
rejects an aliased 1.5K root, verifies all three modular parents with `--require-published-release`,
and separately pins each flat runtime bundle with `--require-legacy-flat-parity-release`; the 1.5K
flat bundle also retains `--require-edit-turbo-1k5-release`.

A missing 1.5K input fails before model loading; a legacy two-release run is diagnostic rather than
a release gate. The job authenticates the 372-tensor oracle and both exact artifact contracts, then
invokes the Cargo binaries for exhaustive q128 Qwen, VAE/DMD, 240 portable-denoiser boundaries, and
the exact accelerated production full chain. The 1.5K invocation selects the F16 VAE q4096,
retained Qwen q128 with deferred synchronization, denoiser q16384, and forced `p4/kv1/q1`
padded-blackbox policy recorded above. Captured-sigma trajectory diagnostics remain separately
reported. `boogu-full-parity` is the production execution-dtype propagated chain.

The job also requires a native/browser Turbo 1024 final-output quality gate. That gate
authenticates the same prompt, seed, request dimensions, model revision, and modular artifact
closure, then applies the published 24 dB PSNR / 0.90 mean block SSIM floor to the two final PNGs.
Each runtime independently generates its noise, so the report always sets
`numerical_parity_claimed=false` and must not be presented as exact or exhaustive cross-runtime
parity. The separate pinned first-DMD diagnostic keeps its exact injected-noise fixture and boundary
checks. Its final-source packed-F16 browser run now has the scoped
`diagnostic-passed-no-full-parity-claim` result recorded above; the prior core-Q8 pass is historical.
Exact-noise full-chain Turbo 1024 cross-runtime parity also remains pending.

Every browser evidence job must resolve the repository's paired root patches. Patched `wgpu`
provides bounded writes and result-bearing queue completion; patched `cubecl-wgpu` submits pending
upload-only work and propagates queue/error-scope failures. Downstream runners must apply equivalent
versions of **both** because Cargo does not inherit root patches from published dependencies.
On Linux, those jobs combine BigInt `statfs` with a real bounded 256 MiB write/`fsync`/delete quota
probe; current evidence admitted `/dev/shm`, rejected quota-limited `/tmp`, and omitted
`--disable-dev-shm-usage`.

The ordinary Wasm UI and browser descriptor expose `edit-turbo-1k5` and all ten official shapes.
The first current schema-v2 modular low-VRAM no-surface computation passed its inner 1536x1536
authentication, numerical, and memory gates but failed the stale host-key contract; the corrected
source-bound rerun now passes as recorded above. The former schema-v1 flat closure also has a
historical positive replay, recorded separately in
[the web guide](web.md#historical-flat-bundle-15k-exhaustive-browser-parity); it does not expand the
current modular result's scope.

The release workflow's modular `qualification-f32` replay is an opt-in workflow-dispatch,
non-blocking control diagnostic; its last run ended in device loss, so it has no passing result. It
streams Qwen and selected VAE objects per request and retains one F32 denoiser across four DMD
steps. It is neither the ordinary UI's eager all-stage `resident` policy nor a substitute for the
required low-VRAM numerical and strict measured-memory outcome. Its evidence is still uploaded when
enabled. The ordinary resident rendered run, other released shapes, synchronized performance, and
cross-stack portability remain separate gates. Ordinary
rendered Turbo 1024 now has its narrower output/surface/memory evidence above.
Canonical CDN/Pages deployment remains blocked as of 2026-08-14 because the prepared entries return
HTTP 403 without the required Range/`Content-Range` CORS policy; local modular evidence does not
waive that publication gate.
