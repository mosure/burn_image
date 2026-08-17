# Architecture

The workspace draws a hard line between reusable architectures and Boogu policy.

```text
                              application dependencies

                         +---------------------+
                         |   bevy_burn_image   |
                         | ECS/UI/device/I/O   |
                         +----------+----------+
                                    |
             +----------------------+----------------------+
             |                      |                      |
             v                      v                      v
       +------------+        +-------------+       +---------------+
       | burn_image |<-------| burn_boogu  |------>| burn_qwen3_vl |
       | neutral API|        | Boogu policy|       | ordinary Qwen |
       +------------+        +------+------+       +---------------+
                                    |
                                    v
                             +---------------+
                             | burn_flux_vae |
                             | ordinary FLUX |
                             +---------------+
```

Arrows point from a consumer to a dependency. `bevy_burn_image` also depends directly on the two
reusable model crates for its native/browser stage-source adapters; those edges are omitted above
only to keep the ownership graph readable. With their optional `artifacts` features,
`burn_qwen3_vl` and `burn_flux_vae` depend on the model-neutral manifest, reader, and cache
contracts in `burn_image`. Neither model crate depends on Boogu or on the other model crate.

`burn_qwen3_vl` and `burn_flux_vae` must remain usable without Boogu. If a checkpoint diverges from
the ordinary architecture, the divergence belongs in an adapter in `burn_boogu`, not in the
reusable crate.

## Model graph

### Text-to-image

1. Render the pinned Boogu system/user chat with the ordinary Qwen3-VL processor.
2. Tokenize without truncation (upstream limit 1280), construct the attention mask and MRoPE IDs.
3. Run Qwen3-VL and trim the unpadded final hidden state used by the released checkpoint.
4. Generate an injected noise tensor, patchify the latent, and build three-axis Boogu RoPE.
5. Run two instruction refiners and two noise refiners.
6. Run eight dual-stream blocks, then concatenate the streams and run 32 single-stream blocks.
7. Apply the adaptive final projection and update the DMD state with
   `linspace(conditioning_sigma, 1, steps + 1)[:-1]`. The conditioning sigma is `0.001` for
   generation; exact values follow the execution dtype. The externally authenticated BF16 fixture records
   `0.0009994507, 0.251953125, 0.5, 0.75`.
8. Reverse FLUX VAE scale/shift, decode, apply the upstream direct `[-1,1]`-to-RGB8 mapping, and
   encode or display the bytes labeled sRGB; no transfer function is added.

### Image edit

The edit graph adds ordinary Qwen3-VL image preprocessing/vision tokens and a FLUX VAE encode.
Posterior sampling consumes a caller-provided epsilon tensor so PyTorch, NdArray, WGPU, and WebGPU
execute the same stochastic path. Reference latent patches precede generated patches in the image
stream. The edit DMD sigmas are `0.0, 0.25, 0.5, 0.75`.

## GPU execution

The default native high-VRAM production policy eagerly verifies, materializes, and synchronizes
every Qwen, VAE, and denoiser stage required by its request graph before the runtime reports
**Ready**. The forward path clones resident WGPU module handles and performs no model-weight
filesystem read, hash, decode, or host-to-device upload. Encoder and decoder remain separate
verified modules, and one denoiser serves all four DMD steps.

The implemented native `low-vram` policy keeps the unchanged production mixed-F16 denoiser
resident while Qwen streams one verified semantic stage at a time and only the VAE half needed by
the current request phase is materialized. Qwen and VAE therefore have per-request host-to-device
traffic, but no denoiser weight is reloaded inside the four-step DMD hot path. Its inventory-derived
static plans are 30,585,112,576 bytes for Turbo and 30,971,005,440 bytes for 1K or 1.5K Edit,
including a conservative 10,000,000,000-byte non-weight reserve. The cap is decimal
32,000,000,000 bytes, not 32 GiB. These plans are conservative rather than measured peaks. The
native Turbo 1024x1024 modular run independently passed its output-qualification memory gate at
27,055 MiB / 28,369,223,680 bytes across 2,246 matched, nonzero PID-scoped samples. Its phase peaks
were 24,814,551,040 bytes during initialization, 27,087,863,808 during Qwen, 27,096,252,416 during
DMD, and 28,369,223,680 during VAE decode. The current-source 1536x1536 Edit-Turbo 1.5K replay over
the separately qualified legacy schema-v1 flat artifact root passed the strict numerical and PID-scoped memory gate
at 28,906 MiB / 30,310,137,856 bytes, with 608 attempts and 607 matched/nonzero samples; its report
SHA-256 is `a013adfcd30b7e6b2323ecc3723b22396f9858d14dd3cdd4a0da2699e199abe3`. 1K Edit still needs its
own measured low-VRAM gate.

Two native Turbo memory details are deliberately confined to `low-vram`. The sealed Qwen
embedding is uploaded directly at its released F16 dtype instead of first materializing an F32
device tensor. VAE decode requests exact-size transient allocations and synchronizes and cleans up
before the decoder tail, avoiding retention of oversized allocator pages. Both are allocator-only,
math-neutral policies: tensor values and ordinary Qwen/VAE operations are unchanged, and native
high-VRAM and browser paths do not inherit them. The output remained byte-identical across this
allocator change. The separate native layer-streamed diagnostic accepts only an explicit local
bundle and additionally rereads denoiser stages per step.

The explicit browser high-VRAM policy follows the same ready-state boundary with dense-F32 WebGPU modules.
Async preload fetches, verifies, and uploads one bounded semantic object at a time, releases its
Wasm payload, and retains the initialized module. After every required Qwen, VAE, and denoiser stage
is present and synchronized, inference has zero repeated model-weight transport or host-to-device
upload. It is an advanced unqualified mode rather than the browser default.

The default browser `low-vram` policy leaves the canonical mixed-F16 artifact unchanged and is
variant-aware. Edit streams Qwen and selected VAE objects and retains an inventory-qualified
runtime-Q8/F32 denoiser through four DMD steps. The first current-source modular 1536x1536 attempt
passed every inner numerical/artifact/fixture gate, but its saved outer report failed because the
host still expected retired resource-plan field `audited_max_streamed_stage_bytes`. The runtime
correctly emitted `audited_max_streamed_qwen_stage_f32_bytes`; offline replay against the corrected
contract was noncanonical. The subsequent source-bound rerun passed top-level `ok=true`, with
443/443 matched GPU intervals, 224 active, and peak 29,828 MiB / 31,276,924,928 bytes. Report
SHA-256 is `c895ae2c1cba3823afe756035b6e564d5ef27caf3722f5f350c07e23086e3b54`. The Edit audit matched
377 Q8S and 565 F32 tensors. No backend kernel trace was recorded, so
`on_device_quantized_execution_claimed=false` remains correct.

Ordinary Turbo does not use runtime Q8. Its initial low-VRAM setup authenticates 46 stages / 106
objects / 912 canonical F16 tensors, inserts 7,264 alignment elements, and retains
19,870,010,624 bytes as packed U32 arenas. For every DMD step, one semantic stage is widened on
device to dense F32, executed with dense-F32 matmul, and released before the next stage. Across four
steps that is exactly 184 stage materializations, 424 object unpacks, 79,480,042,496 packed bytes
read, and 158,960,084,992 F32 bytes written, with zero artifact/cache/network traffic inside DMD.

The Turbo packed cache is deliberately request-scoped. Initial preload happens before **Ready**.
After the fourth DMD step, the only final latent is copied through an exact F32 host handoff; queue
work is synchronized, every packed arena and request-local RoPE/cache handle is cleared, allocator
cleanup is synchronized, an empty cache is proven, and the identical latent is reuploaded before
VAE decode. A later request rehydrates all 106 objects from the integrity-checked persistent range
cache. Thus the next request requires no network response, but it does not reuse a live denoiser on
the GPU.

The exact static plan records a 22,304,263,424-byte preload peak and a 26,492,170,880-byte
conservative inference bound: 19,870,010,624 retained packed-F16 bytes plus the
1,753,654,656-byte maximum materialized F32 stage and a 4,868,505,600-byte activation reserve. The
exact-size persistent Qwen text-layer pool is not represented by a derived byte bound, so the
aggregate measured-GPU-memory gate remains mandatory. Browser VAE transport applies only selected
encoder or decoder objects, while the current source initializes the full 335,278,732-byte F32
autoencoder before selection. These plans are not measured peaks.

The separate `qualification-f32` exact-fixture route retains a request-scoped dense-F32 denoiser as
an optional control diagnostic. It is disabled by default, non-blocking when workflow-dispatched,
and is not a production residency architecture or a substitute for the mandatory browser low-VRAM
numerical and measured-memory gate.

Current serialized Run C and the ordinary rendered Turbo 1024 request both completed the packed-F16
model lifecycle, output download, surface gate, and sub-cap memory check at a measured
23,932,698,624-byte Chrome GPU-process peak. Run C is explicitly a diagnostic-only Qwen block-0
localization result (report SHA-256
`b0dfcc8e53fd7ad1c4731d3169e2f43c50063aa2b54e5ca6347789e18630c6e6`). The ordinary result
is the release/output smoke (report SHA-256
`36525be1d5ff482c409c3b7484027fcb335340e474e4a95f182720ea3f032a28`) and produced the
1,452,562-byte PNG SHA-256
`5d7a947301c9cefef8bca5cd42db52b98fe3fca440e98d302312eff6afe2eb38`. Its same-seed comparison
against native passed `37.517250061 dB` PSNR / `0.985732973` SSIM, which is output-quality evidence rather
than exact-noise parity; quality-report SHA-256 is
`31da8e541013c38dd215257431159a99c7112ad79714a079e2a4b25f9c855103`. The architecture also
holds the gate across centralized packed-F16
failure/cancellation cleanup. The final-source packed-F16 Turbo first-DMD diagnostic passed with
outcome `diagnostic-passed-no-full-parity-claim`. It binds JavaScript SHA-256
`64197f892ae850d901a9b76ff70dba7f543fa70af02028f605bb9eb126dc1b37`, WebAssembly SHA-256
`001f0bcc93fbaeea9a9b32d2adcb8b46f1897b80d36386919c03d69869dca86b`, probe/harness SHA-256
`d6485fa204233b25c1d12128410ae162a8d1ce59053179d3f31a3db63155dd88`, contract SHA-256
`dadd84a4ef9c5162c4aea7f3251cb40461e08d969bfaada3ae99f94dc6fb4b86`, report SHA-256
`0a600471ec9e3119eeaebd616e9dd29a84c62067881b6f9834949019f92d5eab`, and console SHA-256
`20af8aa43d3a53608c0658fabbef0fb8d7e85f2ff4b9655736fc74b959d489fe`. The cache remained exactly
46 stages / 106 objects / 912 tensors; the single prediction performed 46 stage materializations
and 106 object unpacks, read 19,870,010,624 packed bytes, wrote 39,740,021,248 dense-F32 bytes, and
made zero DMD artifact/range/cache/network traffic. Velocity relative RMSE / cosine were
`0.03869645` / `0.9992708`, and prediction relative RMSE / cosine were `0.042713966` /
`0.99910367`. `/dev/shm` passed quota-aware admission, and process-group exit, Chrome-profile
removal, and artifact-server teardown were clean. This is one-prediction diagnostic stability
evidence only: it makes no calibrated numerical-correctness, full-chain/full-resolution parity, or
fully on-device-quantized execution claim. A predecessor core-Q8 first-DMD diagnostic produced
finite output and report SHA-256
`62e14a0712811e088b33d74d047fa6370c5ae8191920cbbc87313b93fb3e68d0`, but its explicit
`diagnostic-passed-no-full-parity-claim` outcome remains historical and does not qualify this
architecture. The canonical same-engine rerun passed two sequential ordinary requests through one
adapter, one device, and one Chrome GPU process, with preload attempts `[1, 2]`. Request 2 read all
186 objects / 35,106,151,424 bytes / 8,489 ranges as persistent-cache hits and made zero network
requests. Both requests completed four zero-I/O DMD steps, digest-preserving Qwen and DMD-to-VAE
handoffs, cache-ready-to-empty cleanup, and one violation-free surface-suspension window followed
by successful acquisition. Peak Chrome GPU-process memory was 24,384,634,880 bytes, and final
process/profile cleanup succeeded with empty page/GPU error lists. The distinct output PNGs have
SHA-256 `5d7a947301c9cefef8bca5cd42db52b98fe3fca440e98d302312eff6afe2eb38` and
`815c553a70a4322aa8e49a51aeb0d46b75ccf2178b435c9b0ba0fedec3da5e0c`; canonical report SHA-256
is `90da22207398ae907e6b0d0bc93881c689a2a7362a1e52aac5435deac525b5d5`. This is an ordinary
same-engine rendered smoke, not numerical parity. The explicit browser
layer-streamed policy also remains diagnostic and reloads the denoiser on every DMD step. CPU work
is limited to input decoding, tokenization, small position plans, bounded artifact
verification/transport, host handoffs named by the policy, and output
encoding; it does not execute model layers.

The historical Q8 work retained the useful packed-weight/direct-quantized-matmul design constraint
demonstrated by the
[Voxtral Mini realtime GGUF runtime](https://github.com/TrevorS/voxtral-mini-realtime-rs). Ordinary
Turbo now deliberately uses a different contract: compact F16 storage is widened only one semantic
stage at a time for dense-F32 execution. Q4 remains disabled without image-quality and kernel-trace
qualification of its own.

Native WGPU and browser WebGPU use the same Burn modules. Qwen attention currently expands grouped
key/value heads with Burn tensor operations, then evaluates explicit F32 score matmuls and softmax
in bounded query chunks. This keeps the largest score activation at
`query_chunk_size * sequence_length` rather than materializing a full sequence-squared tensor, but
it is not a fused attention or native GQA kernel. A future backend attention path must preserve the
same causal-mask, packed-frame, F32-softmax, and observer semantics.

The Boogu denoiser likewise retains complete key/value context while evaluating a bounded,
policy-selected number of query rows per attention call. The portable baseline uses 128 rows;
qualified resident policies may select a larger bound to improve GPU density without allocating a
full sequence-by-sequence score tensor. Its largest fallback score activation is therefore
`batch * heads * query_chunk_size * key_length`. Here, “resident dense” describes initialized,
unquantized stage weights on the GPU, not dense sequence-squared attention.

The Bevy renderer and Burn runtime must be created from the same WGPU adapter/device/queue. A
viewer that silently dispatches inference to NdArray is considered a configuration error, even if
it successfully renders a window.

Browser queue correctness is also part of the architecture. Checkout and Pages builds use both
root patches: patched `wgpu` bounds browser uploads and reports rejected queue-completion promises,
while patched `cubecl-wgpu` submits pending upload-only work and propagates queue/error-scope
failure through asynchronous synchronization. Cargo does not propagate either root patch through a
published dependency, so downstream WebGPU applications must apply equivalent versions of both.

## Browser memory lifecycle

Artifacts are grouped by executable component and layer. Production browser initialization uses
this lifecycle:

```text
fetch bounded ranges -> assemble one bounded semantic object -> verify SHA-256 and inventory
-> upload/materialize retained WebGPU stage -> release host payload -> fetch next object
-> synchronize complete resident graph -> Ready
```

The Bevy app wires both production lifecycles through `BrowserBooguFactory`. Asynchronous Qwen, VAE,
and denoiser sources keep only one verified transport payload in Wasm at a time. Explicit high-VRAM
mode accumulates all initialized modules on WebGPU before **Ready**. Default low-VRAM Edit
accumulates only the runtime-Q8 denoiser during a request, reuses it for all DMD steps, clears it,
and then streams the decoder. Default low-VRAM Turbo initially retains the exact packed-F16 cache,
widens and releases one dense-F32 stage at a time during DMD, then empties the packed cache before
VAE decode and rehydrates it from persistent storage before a later request. Any failed or cancelled
packed-F16 request takes the same fail-closed root-release and allocator-cleanup boundary before its
terminal can release the surface gate.
Each mode checks its conservative device-resource plan before allocation; allocation or
synchronization failure is terminal, and there is no CPU fallback. The explicit diagnostic policy
instead uploads, executes, and drops every stage, including denoiser stages, one at a time.

The browser and native descriptors expose the same 256-1024 contract for Turbo and 1K Edit and the
same ten official shapes for Edit-Turbo 1.5K. Browser shapes at or above 1024 use the exact
strict-F32 striped VAE tail selected by the shape-aware buffer plan; smaller shapes retain the full
decoder. The first current modular low-VRAM 1536x1536 computation passed internally but not its
stale host contract; the corrected source-bound canonical rerun now passes that exact no-surface
numerical/memory gate. Descriptor equality and buffer-plan tests establish the other implemented
shapes, not rendered real-hardware qualification.

Immutable objects are content-addressed and can be cached by a browser host or native filesystem
cache. These mechanisms and their static/cache-contract tests do not by themselves qualify a
browser/adapter for numerical correctness, memory, or performance. On Linux, the evidence harnesses
select Chrome shared-memory backing with BigInt `statfs` plus a real bounded 256 MiB
write/`fsync`/delete quota probe; current runs admitted `/dev/shm`, rejected quota-limited `/tmp`,
and omitted `--disable-dev-shm-usage`.

Ordinary rendered Turbo 1024
now has its scoped output/surface/memory result; exact-noise parity remains separate. Current
modular low-VRAM 1536x1536 has its scoped no-surface numerical/memory pass. Other shapes, the
explicit resident mode, rendered 1.5K behavior, synchronized performance, and cross-stack
portability still require their own attested gates.

Canonical browser/publication evidence uses the separate schema-v2 modular root. As of 2026-08-16,
its public CDN is readable and a real-browser probe verified authenticated cold whole-part loading
and warm CacheStorage resume. Pages warns when reusable manifests are `immutable` rather than the
recommended `no-cache`, while sealed manifest and payload failures remain blocking; verified
transport does not replace generation/parity evidence.
