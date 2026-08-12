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
only to keep the ownership graph readable. `burn_image`, `burn_qwen3_vl`, and `burn_flux_vae` do
not depend on one another or on Boogu.

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

The default native Bevy policy retains verified Qwen and VAE stages after their first request and
keeps the Boogu denoiser on the GPU across all four DMD steps. Encoder and decoder remain separate
verified objects even when both handles are retained. An explicit lower-residency native policy
rereads Qwen and VAE stages per request and denoiser layers per step.
Browser execution uses async non-retaining semantic sources throughout. CPU work covers input
decoding, tokenization, small position plans, integrity hashing, transport/orchestration, and
output encoding.

Native WGPU and browser WebGPU use the same Burn modules. Qwen attention currently expands grouped
key/value heads with Burn tensor operations, then evaluates explicit F32 score matmuls and softmax
in bounded query chunks. This keeps the largest score activation at
`query_chunk_size * sequence_length` rather than materializing a full sequence-squared tensor, but
it is not a fused attention or native GQA kernel. A future backend attention path must preserve the
same causal-mask, packed-frame, F32-softmax, and observer semantics.

The Boogu denoiser likewise retains complete key/value context while evaluating at most 128 query
rows per attention call. It preserves the production activation dtype and default Burn scaling,
so mixed artifacts execute this attention in F16 and the Q8/F32 denoiser policy executes it in
F32. Its largest fallback score activation is therefore
`batch * heads * 128 * key_length`, never the full query-by-key matrix.

The Bevy renderer and Burn runtime must be created from the same WGPU adapter/device/queue. A
viewer that silently dispatches inference to NdArray is considered a configuration error, even if
it successfully renders a window.

## Browser memory lifecycle

Artifacts are grouped by executable component and layer. The implemented manifest, verified range
loader, and staged model interfaces are designed for this lifecycle:

```text
fetch bounded ranges -> assemble one bounded semantic object -> verify SHA-256 and inventory
-> upload the requested stage -> execute -> release stage weights -> fetch next shard
```

The Bevy app now wires this lifecycle through `BrowserBooguFactory`: asynchronous Qwen, VAE, and
denoiser sources fetch and verify one semantic object, upload and execute it, synchronize, and drop
its module before requesting the next. The runtime does not require those weights to coexist in
Wasm linear memory. Immutable objects are content-addressed and can be cached by a browser host or
native filesystem cache. Browser numerical support still requires a completed real-checkpoint
WebGPU parity run; successful compilation and packaging alone are not inference evidence.
