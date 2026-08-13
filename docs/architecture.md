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

The default native production policy eagerly verifies, materializes, and synchronizes every Qwen,
VAE, and denoiser stage required by its request graph before the runtime reports **Ready**. The
forward path clones resident WGPU module handles and performs no model-weight filesystem read,
hash, decode, or host-to-device upload. Encoder and decoder remain separate verified modules, and
one denoiser serves all four DMD steps. The explicit native layer-streamed diagnostic accepts only
an explicit local bundle and rereads Qwen and VAE stages per request plus denoiser stages per step.

The ordinary browser production policy follows the same ready-state boundary with dense-F32
WebGPU modules. Async preload fetches, verifies, and uploads one bounded semantic object at a time,
releases its Wasm payload, and retains the initialized module. After every required Qwen, VAE, and
denoiser stage is present and synchronized, inference has zero repeated model-weight transport or
host-to-device upload. The explicit browser layer-streamed policy is diagnostic only. CPU work in
either resident path is limited to input decoding, tokenization, small position plans, preload
integrity/transport orchestration, and output encoding; it does not execute model layers.

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

## Browser memory lifecycle

Artifacts are grouped by executable component and layer. Production browser initialization uses
this lifecycle:

```text
fetch bounded ranges -> assemble one bounded semantic object -> verify SHA-256 and inventory
-> upload/materialize retained WebGPU stage -> release host payload -> fetch next object
-> synchronize complete resident graph -> Ready
```

The Bevy app now wires this lifecycle through `BrowserBooguFactory`: asynchronous Qwen, VAE, and
denoiser sources keep only one verified payload in Wasm at a time while the uploaded modules
accumulate on WebGPU. A conservative device-resource plan is checked before preload; allocation or
synchronization failure prevents **Ready**, and there is no CPU fallback. The explicit diagnostic
policy instead uploads, executes, and drops one stage at a time.

Immutable objects are content-addressed and can be cached by a browser host or native filesystem
cache. This resident mechanism and its static/cache-contract tests do not by themselves qualify a
browser/adapter for numerical correctness or performance; those claims still require a completed,
attested real-checkpoint WebGPU gate.
