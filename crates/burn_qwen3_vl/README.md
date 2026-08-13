# burn_qwen3_vl

Reusable [Burn](https://burn.dev) 0.21 implementation of the ordinary Qwen3-VL text and vision
architecture. The crate owns model math, multimodal position planning, generic Qwen chat rendering,
tokenizer integration, and a strict checkpoint inventory. It deliberately does not own a product's
prompt, image-generation schedule, artifact URL, or UI policy.

## What is implemented

- Hugging Face-compatible nested configuration with validation for grouped-query attention,
  head dimensions, MRoPE sections, vision merge geometry, deep-stack indices, and special ids.
- Causal Qwen language layers: bias-free Q/K/V/O projections, per-head Q/K RMSNorm, interleaved
  temporal/height/width RoPE, grouped-query KV expansion, SwiGLU MLPs, and pre-norm residuals.
- Qwen vision layers: strided Conv3d patch embedding, learned 2D position-table interpolation,
  merge-block patch ordering, 2D rotary embeddings, packed frame-isolated attention, tanh GELU
  blocks, final patch merger, and deep-stack patch mergers.
- Correct visual embedding replacement followed by deep-stack additions in the first language
  layers.
- Untied and tied vocabulary projections.
- Generic ChatML messages, `<|image_pad|>` / `<|video_pad|>` expansion, padding, attention masks,
  modality ids, visual token destinations, tool calls/responses, timestamp-separated video frames,
  and exact multimodal position deltas.
- Published Qwen smart resize constraints, decoded-RGB bicubic resize, `1/255` rescaling,
  per-channel normalization, temporal duplication, merge-block ordering, patch flattening, and
  `grid_thw` production.
- Optional `tokenizers` adapter for a standard Hugging Face `tokenizer.json`.
- Complete source-to-record tensor inventory. The released 36-text-layer, 27-vision-block,
  three-deep-stack configuration has 749 base tensors or 750 tensors with an untied LM head.
- Optional strict SafeTensors importer: validates the entire index and every shard header, key,
  shape, dtype, file length, and shard assignment before allocation, then applies one shard at a
  time.
- Base-model-only loading that still validates an untied `lm_head`, but does not allocate or upload
  it when the consumer only needs multimodal conditioning.
- Semantic stage decomposition for constrained devices: row-routed embedding chunks, a vision
  prelude, individual vision blocks/mergers, individual decoder layers, final norm, optional
  row-chunked output projection, activation observers, and a runtime-supplied stage source.

Attention is evaluated in bounded query chunks (128 tokens by default), so neither the causal text
path nor packed vision path materializes a full sequence-by-sequence score tensor. The operators
are backend-generic and avoid activation readbacks. Row-routed streaming reads only the small token
id tensor; its browser path uses Burn's asynchronous readback API.

## Model construction

```rust,no_run
use burn_qwen3_vl::Qwen3VlBuilder;

# fn build<B: burn::tensor::backend::Backend>(
#     config_json: &str,
#     device: &B::Device,
# ) -> burn_qwen3_vl::Result<()> {
let builder = Qwen3VlBuilder::from_json(config_json)?;

// Validate checkpoint keys and shapes before constructing or applying a record.
let inventory = builder.causal_lm_inventory();
for spec in inventory.specs() {
    println!("{} -> {} {:?}", spec.source, spec.target, spec.shape);
}

let mut model = builder.build_causal_lm::<B>(device)?;
// model.set_query_chunk_size(64); // smaller attention tiles for tighter memory limits
# let _ = model;
# Ok(())
# }
```

Linear layers use the crate's checkpoint-compatible column layout. Published `[out, in]` weights
therefore keep their checkpoint shape while the load mapper prepares Burn's `[in, out]` runtime
matmul representation. Native builds alias Burn's `Linear` and `LinearLayout::Col` exactly; wasm32
uses the same transpose save/load/init mappers without Burn 0.21's blocking `Backend::sync`, which
is unavailable in a browser without wasm threads. Both paths forward through
`burn::tensor::module::linear` and preserve the same `weight` / `bias` record paths.
LayerNorm and RMSNorm checkpoint names map from `weight` / `bias` to Burn record fields `gamma` /
`beta`; `WeightInventory::source_to_target` is the canonical mapping.

## Processing and forward contract

`Qwen3VlProcessor<T>` accepts any `Qwen3VlTokenizer`. Rendered chat contains one media placeholder
per item. Images expand to `t * h * w / spatial_merge_size²` language tokens. Videos expand to one
timestamped vision span per temporal grid frame, matching Qwen3-VL's released template and MRoPE
grouping. The processor returns:

- padded `input_ids` and boolean `attention_mask`;
- `mm_token_type_ids` (`0=text`, `1=image`, `2=video`);
- flattened visual-token destinations;
- a deterministic `MropePositionIds` plan.

`Qwen3VlImageProcessor` accepts decoded `image::RgbImage` values. `preprocess_dynamic` additionally
handles decoded 8/16-bit grayscale, grayscale-alpha, RGB, and RGBA images with Pillow-compatible
16-to-8-bit truncation and alpha removal. Its config directly deserializes the released
`preprocessor_config.json`; `ProcessedVisionPixels::to_tensor` produces the tensor and `Grid`
required by `Qwen3VlVisualInput`.

Preprocessed vision input to `Qwen3VlVisualInput` is a rank-2 tensor with shape
`[sum(t*h*w), in_channels*temporal_patch_size*patch_size*patch_size]`. Patch rows must already use
Qwen's spatial merge-block order. The corresponding `Grid` values remain in pre-merge `(t,h,w)`
units. `Qwen3VlModelInput` carries images and videos separately so their placeholder destinations
remain explicit.

The model returns the unmerged hidden states after the final vision block, merged visual features,
each configured deep-stack feature, language hidden states on request, MRoPE position deltas, and
causal-LM logits.

With the `import` feature, `HfCheckpoint::from_index` resolves a standard
`model.safetensors.index.json`; `load_causal_lm_from_safetensors` and
`load_base_from_safetensors` perform strict native import. The explicit-dtype variants convert
source BF16 lazily to F32 or F16. Source BF16 is rejected on Burn 0.21 WGPU because a real decoder
layer overflows on that kernel path; use F16 for native WGPU/WebGPU or F32 for diagnosis. This is a
conversion/development path, not the browser artifact path: browser runtimes should consume bounded
Burnpack shards from their artifact layer.

## Bounded stage streaming

`Qwen3VlStreamingPlan::released_f16` partitions the released 151,936 by 4,096 embedding table into
six contiguous chunks of about 197.8 MiB each, below the 256 MiB semantic-object cap. For adapters
with a lower limit, `RowChunkPlan::for_max_bytes` derives a gap-free plan. Token-id routing selects only
the rows needed from each short-lived chunk into a retained activation; the full 1.159 GiB F16
table is never resident. The untied LM head uses the same optional row-slice mechanism, and is
absent from the ordinary base-model conditioning stream.

`Qwen3VlStageSource` deliberately does not prescribe URLs or cache locations. With the optional
`artifacts` feature, `Qwen3VlComponentContract` validates a sealed standalone component manifest,
and `VerifiedBurnpackQwen3VlStageSource` / `VerifiedAsyncBurnpackQwen3VlStageSource` verify and
apply bounded Burnpack objects through the transport-neutral readers in `burn_image`. The Qwen
crate owns shard semantics, stage loading, and device-resident retaining wrappers; `burn_image`
owns the generic filesystem/browser transport and cache seam. The loaders strip the canonical
stage prefix and apply tensors to a fresh lazy module, advance `Qwen3VlVisionState` or
`Qwen3VlTextState`, synchronize the
backend, and then drops the module. Column-layout linear shards stay in checkpoint `[out, in]`
shape and use an identity adapter; the Qwen parameter load mapper performs the runtime transpose.
On wasm32 that mapper submits the transpose without a blocking sync, leaving the async source's
stage barrier responsible for completion. Never collect or forward a stage module before applying
its shard, because doing so initializes the parameter and changes its validation shape.

The released component is `qwen3-vl-8b-base-boogu-image-0.1` with profile
`f16-text-f32-vision-base`. Its model revision is derived from the four sorted upstream source-file
declarations, and its exact manifest digest is pinned by the crate. `qwen_component_dependency()`
returns the complete role/bundle/profile/model/revision/digest edge for a composed manifest. A
Qwen-only consumer can use the same model-neutral verified cache and readers without depending on
`burn_boogu`.

`StreamingQwen3Vl::forward_base` performs that complete orchestration for the ordinary base model,
including concurrent image/video state progression, visual replacement, DeepStack additions,
optional hidden states, and a synchronization after each droppable module. The released
storage/native-WGPU policy is `Qwen3VlStageDTypePolicy::released_hybrid()`: embeddings/text stay
F16, while the vision prelude, blocks, and mergers use F32. The largest F32 vision stage is
160,503,808 bytes (153.07 MiB), so it remains below the semantic-object cap. Vision outputs are
converted to text dtype only at the placeholder/DeepStack insertion boundary, matching
Transformers. Browser runtimes may adapt every stored floating stage to F32 when their WebGPU F16
kernels are not accurate; that runtime policy is separate and requires browser parity evidence.

`AsyncQwen3VlStageSource` and `StreamingQwen3Vl::forward_base_async` expose the same semantic
sequence for browser runtimes. Their futures intentionally do not require `Send`, allowing fetch,
Cache Storage, and WebGPU handles to stay on one wasm event loop. The orchestrator awaits each
bounded row/module fetch, computes the stage, awaits source synchronization, drops the stage, and
only then requests the next one. The asynchronous and synchronous paths share exact observer
boundaries and are tested against the same resident multimodal model.

Runtimes with sufficient VRAM can opt into `RetainingQwen3VlStageSource` or
`RetainingAsyncQwen3VlStageSource`. Each wraps a verified source, keeps the first successfully
loaded copy of each embedding/LM-head row and semantic module, and serves shared device-handle
clones on later forwards while preserving the selected synchronization policy. `clear` releases
all retained weights. The asynchronous path still fetches and verifies only one bounded artifact
object at a time; GPU residency does not imply retaining Burnpack bytes in Wasm memory.

## Correctness surface

The unit suite uses tiny deterministic NdArray models and pure CPU plans to check:

- the published configuration dimensions and exact 750-tensor inventory;
- image/text MRoPE positions, delta semantics, and axis-frequency interleaving;
- spatial merge ordering and bilinear weights;
- RGB normalization, temporal duplication, patch order, placeholder expansion, timestamped video
  grouping, tool-template rendering, and modality token ids;
- causal prefix independence and packed-frame attention isolation;
- finite text, vision, and complete multimodal forward execution.
- a complete synthetic SafeTensors checkpoint inspection and load through every model parameter;
- exact token-routed chunked embedding versus full-table lookup, and staged decoder execution
  versus the resident decoder.

Opt-in real tests additionally validate the pinned four-shard, 750-tensor BF16 checkpoint, compare
the decoded edit source against Transformers preprocessing, and compare WGPU boundaries/final
hidden states against externally supplied `--capture-qwen` SafeTensors fixtures. The externally
authenticated 16-bit RGBA edit source matches the Transformers processor to one F32 ULP after
Pillow-compatible decode. An on-demand deterministic 100-by-200 stress input forces smart resize
to 192 by 384 and validates the portable
scale-dependent antialiased bicubic kernel against torchvision: more than 99.3% of raw channels
are exact, maximum error is two RGB levels, normalized RMSE is `0.000624`, and cosine similarity is
`0.9999991`.

The text-only Turbo fixture is stable on WGPU F16 and its remaining final-hidden error versus BF16
is almost entirely explained by the upstream BF16-to-F16 baseline. The vision tower is not stable
in Burn 0.21 WGPU F16: the error begins in patch Conv3d and compounds through the blocks/merger.
F32 vision avoids that collapse and is therefore required by the hybrid policy; a text-only pass
must not be interpreted as vision/deep-stack parity.

## Features

- `std` (default): ordinary Rust standard-library build.
- `tokenizers`: Hugging Face tokenizer adapter (`HfTokenizer`).
- `import`: strict native Hugging Face SafeTensors index/shard loading through `burn-store`.
- `artifacts`: sealed component-manifest validation plus synchronous and asynchronous bounded
  Burnpack stage sources; transport and persistent cache policy remain in `burn_image` adapters.

The core model does not select a Burn backend. Consumers can use NdArray, WGPU/WebGPU, or another
Burn 0.21 backend without changing this crate's model API.

The smart-resize dimensions, normalization, duplication, and patchification match the published
processor algorithm. Resizing uses a bounded-support Pillow-compatible fixed-point bicubic kernel;
its support expands when downsampling to provide antialias filtering. The external resize oracle
records the remaining torchvision-versus-Pillow rounding envelope explicitly rather than claiming
bit identity between those implementations; no generated image or tensor fixture is committed to
this crate.

## License

Licensed under either Apache-2.0 or MIT, at your option.
