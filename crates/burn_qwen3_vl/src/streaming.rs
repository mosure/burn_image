//! One-semantic-stage-at-a-time Qwen3-VL execution for constrained devices.
//!
//! This module contains model decomposition only. Fetching, integrity verification, caching, and
//! application policy remain responsibilities of the artifact/runtime crate implementing
//! [`Qwen3VlStageSource`] or [`AsyncQwen3VlStageSource`].

use core::{marker::PhantomData, ops::Range};

use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, RmsNorm},
    tensor::{Bool, IndexingUpdateOp, Int, Tensor, TensorData, backend::Backend},
};
use serde::{Deserialize, Serialize};

use crate::{
    DeepstackEmbeddings, Grid, MropePositionIds, Qwen3VlConfig, Qwen3VlError, Qwen3VlTextConfig,
    Qwen3VlVisionConfig, Qwen3VlVisionOutput, Result, WeightInventory, WeightSpec,
    model::{
        Qwen3VlModelInput, append_deepstack, assign_visual_features, combine_vision_outputs,
        validate_visual_destinations,
    },
    outputs::Qwen3VlModelOutput,
    text::Qwen3VlDecoderLayer,
    vision::{
        Qwen3VlVisionBlock, Qwen3VlVisionModel, Qwen3VlVisionPatchEmbed, Qwen3VlVisionPatchMerger,
        VisionPositionPlan,
    },
};

/// A practical default that splits the released 1.2-GiB F16 vocabulary table into six
/// approximately 193-MiB bindings. Call [`RowChunkPlan::for_max_bytes`] with the adapter's
/// reported limit when it is lower.
pub const DEFAULT_VOCABULARY_CHUNKS: usize = 6;

/// A contiguous row slice of an embedding or vocabulary projection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RowChunkSpec {
    pub chunk_index: usize,
    pub row_range: Range<usize>,
    pub total_rows: usize,
    pub hidden_size: usize,
    pub element_bytes: usize,
}

impl RowChunkSpec {
    pub fn rows(&self) -> usize {
        self.row_range.end - self.row_range.start
    }

    pub fn byte_len(&self) -> usize {
        self.rows() * self.hidden_size * self.element_bytes
    }
}

/// Complete, gap-free row partition with an explicit per-binding byte bound.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RowChunkPlan {
    pub chunks: Vec<RowChunkSpec>,
}

impl RowChunkPlan {
    pub fn even(
        total_rows: usize,
        hidden_size: usize,
        chunk_count: usize,
        element_bytes: usize,
    ) -> Result<Self> {
        if total_rows == 0 || hidden_size == 0 || chunk_count == 0 || element_bytes == 0 {
            return Err(Qwen3VlError::InvalidConfig(
                "row chunk dimensions and count must be non-zero".into(),
            ));
        }
        let chunk_count = chunk_count.min(total_rows);
        let quotient = total_rows / chunk_count;
        let remainder = total_rows % chunk_count;
        let mut chunks = Vec::with_capacity(chunk_count);
        let mut start = 0;
        for chunk_index in 0..chunk_count {
            let rows = quotient + usize::from(chunk_index < remainder);
            let end = start + rows;
            chunks.push(RowChunkSpec {
                chunk_index,
                row_range: start..end,
                total_rows,
                hidden_size,
                element_bytes,
            });
            start = end;
        }
        let plan = Self { chunks };
        plan.validate()?;
        Ok(plan)
    }

    pub fn for_max_bytes(
        total_rows: usize,
        hidden_size: usize,
        element_bytes: usize,
        max_bytes: usize,
    ) -> Result<Self> {
        let row_bytes = hidden_size
            .checked_mul(element_bytes)
            .ok_or_else(|| Qwen3VlError::InvalidConfig("row chunk byte-size overflow".into()))?;
        if max_bytes < row_bytes {
            return Err(Qwen3VlError::InvalidConfig(format!(
                "binding limit {max_bytes} cannot hold one {row_bytes}-byte vocabulary row"
            )));
        }
        let rows_per_chunk = max_bytes / row_bytes;
        let chunk_count = total_rows.div_ceil(rows_per_chunk);
        let plan = Self::even(total_rows, hidden_size, chunk_count, element_bytes)?;
        if plan.chunks.iter().any(|chunk| chunk.byte_len() > max_bytes) {
            return Err(Qwen3VlError::InvalidConfig(
                "balanced row partition exceeds requested byte bound".into(),
            ));
        }
        Ok(plan)
    }

    pub fn validate(&self) -> Result<()> {
        if self.chunks.is_empty() {
            return Err(Qwen3VlError::InvalidConfig(
                "row chunk plan must not be empty".into(),
            ));
        }
        let first = &self.chunks[0];
        let mut cursor = 0;
        for (index, chunk) in self.chunks.iter().enumerate() {
            if chunk.chunk_index != index
                || chunk.row_range.start != cursor
                || chunk.row_range.start >= chunk.row_range.end
                || chunk.total_rows != first.total_rows
                || chunk.hidden_size != first.hidden_size
                || chunk.element_bytes != first.element_bytes
            {
                return Err(Qwen3VlError::InvalidConfig(
                    "row chunks must form one ordered, contiguous partition".into(),
                ));
            }
            cursor = chunk.row_range.end;
        }
        if cursor != first.total_rows {
            return Err(Qwen3VlError::InvalidConfig(
                "row chunks do not cover the complete table".into(),
            ));
        }
        Ok(())
    }
}

/// Row-slice transform consumed by artifact converters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RowSliceWeightSpec {
    pub source: String,
    pub target: String,
    pub full_shape: [usize; 2],
    pub chunk_shape: [usize; 2],
    pub row_range: Range<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Qwen3VlStage {
    EmbeddingRows { chunk: usize },
    VisionPrelude,
    VisionBlock { index: usize },
    VisionDeepstackMerger { index: usize, after_block: usize },
    VisionFinalMerger,
    TextBlock { index: usize },
    TextFinalNorm,
    LmHeadRows { chunk: usize },
}

/// Artifact precision for an independently loadable semantic stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Qwen3VlStageDType {
    F16,
    F32,
}

/// Stage-specific precision policy. Burn 0.21 WGPU needs F32 vision math for the released Qwen
/// checkpoint, while the much larger embedding/text weights remain practical in F16.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Qwen3VlStageDTypePolicy {
    pub embedding: Qwen3VlStageDType,
    pub vision: Qwen3VlStageDType,
    pub text: Qwen3VlStageDType,
    pub lm_head: Qwen3VlStageDType,
}

impl Qwen3VlStageDTypePolicy {
    pub const fn uniform(dtype: Qwen3VlStageDType) -> Self {
        Self {
            embedding: dtype,
            vision: dtype,
            text: dtype,
            lm_head: dtype,
        }
    }

    /// Released hybrid storage/native-WGPU policy.
    ///
    /// Native WGPU parity validates F16 embeddings/text with F32 vision. A browser runtime may
    /// adapt these stored floating stages to F32 when its WebGPU implementation does not execute
    /// the F16 kernels accurately; that execution policy belongs to the runtime, not this
    /// checkpoint inventory type.
    pub const fn released_hybrid() -> Self {
        Self {
            embedding: Qwen3VlStageDType::F16,
            vision: Qwen3VlStageDType::F32,
            text: Qwen3VlStageDType::F16,
            lm_head: Qwen3VlStageDType::F16,
        }
    }

    pub const fn for_stage(self, stage: &Qwen3VlStage) -> Qwen3VlStageDType {
        match stage {
            Qwen3VlStage::EmbeddingRows { .. } => self.embedding,
            Qwen3VlStage::VisionPrelude
            | Qwen3VlStage::VisionBlock { .. }
            | Qwen3VlStage::VisionDeepstackMerger { .. }
            | Qwen3VlStage::VisionFinalMerger => self.vision,
            Qwen3VlStage::TextBlock { .. } | Qwen3VlStage::TextFinalNorm => self.text,
            Qwen3VlStage::LmHeadRows { .. } => self.lm_head,
        }
    }
}

/// Exact full tensors or row slice assigned to one independently loadable stage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Qwen3VlStageDescriptor {
    pub stage: Qwen3VlStage,
    pub tensors: Vec<WeightSpec>,
    pub row_slice: Option<RowSliceWeightSpec>,
}

impl Qwen3VlStageDescriptor {
    pub fn byte_len(&self, element_bytes: usize) -> Option<usize> {
        if element_bytes == 0 {
            return None;
        }
        let tensors = self.tensors.iter().try_fold(0_usize, |total, tensor| {
            tensor
                .shape
                .iter()
                .try_fold(element_bytes, |bytes, &dimension| {
                    bytes.checked_mul(dimension)
                })
                .and_then(|bytes| total.checked_add(bytes))
        })?;
        self.row_slice.as_ref().map_or(Some(tensors), |slice| {
            slice
                .chunk_shape
                .iter()
                .try_fold(element_bytes, |bytes, &dimension| {
                    bytes.checked_mul(dimension)
                })
                .and_then(|bytes| tensors.checked_add(bytes))
        })
    }
}

/// Deterministic semantic stage and weight partition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Qwen3VlStreamingPlan {
    pub stages: Vec<Qwen3VlStageDescriptor>,
    pub embedding_rows: RowChunkPlan,
    pub lm_head_rows: Option<RowChunkPlan>,
}

impl Qwen3VlStreamingPlan {
    /// Six balanced F16 vocabulary chunks by default; LM-head stages are optional because base
    /// multimodal conditioning never computes logits.
    pub fn released_f16(config: &Qwen3VlConfig, include_lm_head: bool) -> Result<Self> {
        let embedding_rows = RowChunkPlan::even(
            config.text_config.vocab_size,
            config.text_config.hidden_size,
            DEFAULT_VOCABULARY_CHUNKS,
            2,
        )?;
        let lm_head_rows = include_lm_head
            .then(|| {
                RowChunkPlan::even(
                    config.text_config.vocab_size,
                    config.text_config.hidden_size,
                    DEFAULT_VOCABULARY_CHUNKS,
                    2,
                )
            })
            .transpose()?;
        Self::new(config, embedding_rows, lm_head_rows)
    }

    pub fn new(
        config: &Qwen3VlConfig,
        embedding_rows: RowChunkPlan,
        lm_head_rows: Option<RowChunkPlan>,
    ) -> Result<Self> {
        config.validate()?;
        embedding_rows.validate()?;
        if embedding_rows.chunks[0].total_rows != config.text_config.vocab_size
            || embedding_rows.chunks[0].hidden_size != config.text_config.hidden_size
        {
            return Err(Qwen3VlError::InvalidConfig(
                "embedding row plan does not match text configuration".into(),
            ));
        }
        if let Some(plan) = &lm_head_rows {
            plan.validate()?;
            if config.tie_word_embeddings {
                return Err(Qwen3VlError::InvalidConfig(
                    "a tied vocabulary projection reuses embedding chunks and has no LM-head tensor"
                        .into(),
                ));
            }
            if plan.chunks[0].total_rows != config.text_config.vocab_size
                || plan.chunks[0].hidden_size != config.text_config.hidden_size
            {
                return Err(Qwen3VlError::InvalidConfig(
                    "LM-head row plan does not match text configuration".into(),
                ));
            }
        }

        let inventory = WeightInventory::for_config(config, true);
        let specs = inventory.specs();
        let mut stages = embedding_rows
            .chunks
            .iter()
            .map(|chunk| Qwen3VlStageDescriptor {
                stage: Qwen3VlStage::EmbeddingRows {
                    chunk: chunk.chunk_index,
                },
                tensors: Vec::new(),
                row_slice: Some(row_slice("model.language_model.embed_tokens.weight", chunk)),
            })
            .collect::<Vec<_>>();
        stages.push(descriptor(Qwen3VlStage::VisionPrelude, specs, |name| {
            name.starts_with("model.visual.patch_embed.") || name == "model.visual.pos_embed.weight"
        }));
        for index in 0..config.vision_config.depth {
            let prefix = format!("model.visual.blocks.{index}.");
            stages.push(descriptor(
                Qwen3VlStage::VisionBlock { index },
                specs,
                |name| name.starts_with(&prefix),
            ));
        }
        for (index, &after_block) in config
            .vision_config
            .deepstack_visual_indexes
            .iter()
            .enumerate()
        {
            let prefix = format!("model.visual.deepstack_merger_list.{index}.");
            stages.push(descriptor(
                Qwen3VlStage::VisionDeepstackMerger { index, after_block },
                specs,
                |name| name.starts_with(&prefix),
            ));
        }
        stages.push(descriptor(Qwen3VlStage::VisionFinalMerger, specs, |name| {
            name.starts_with("model.visual.merger.")
        }));
        for index in 0..config.text_config.num_hidden_layers {
            let prefix = format!("model.language_model.layers.{index}.");
            stages.push(descriptor(
                Qwen3VlStage::TextBlock { index },
                specs,
                |name| name.starts_with(&prefix),
            ));
        }
        stages.push(descriptor(Qwen3VlStage::TextFinalNorm, specs, |name| {
            name.starts_with("model.language_model.norm.")
        }));
        if let Some(plan) = &lm_head_rows {
            stages.extend(plan.chunks.iter().map(|chunk| Qwen3VlStageDescriptor {
                stage: Qwen3VlStage::LmHeadRows {
                    chunk: chunk.chunk_index,
                },
                tensors: Vec::new(),
                row_slice: Some(row_slice("lm_head.weight", chunk)),
            }));
        }
        Ok(Self {
            stages,
            embedding_rows,
            lm_head_rows,
        })
    }
}

fn descriptor(
    stage: Qwen3VlStage,
    specs: &[WeightSpec],
    predicate: impl Fn(&str) -> bool,
) -> Qwen3VlStageDescriptor {
    Qwen3VlStageDescriptor {
        stage,
        tensors: specs
            .iter()
            .filter(|spec| predicate(&spec.source))
            .cloned()
            .collect(),
        row_slice: None,
    }
}

fn row_slice(source: &str, chunk: &RowChunkSpec) -> RowSliceWeightSpec {
    RowSliceWeightSpec {
        source: source.into(),
        target: source.into(),
        full_shape: [chunk.total_rows, chunk.hidden_size],
        chunk_shape: [chunk.rows(), chunk.hidden_size],
        row_range: chunk.row_range.clone(),
    }
}

/// One short-lived embedding table slice in source `[row, hidden]` layout.
#[derive(Debug, Clone)]
pub struct EmbeddingRowChunk<B: Backend> {
    pub spec: RowChunkSpec,
    pub weight: Tensor<B, 2>,
}

impl<B: Backend> EmbeddingRowChunk<B> {
    pub fn new(spec: RowChunkSpec, weight: Tensor<B, 2>) -> Result<Self> {
        if weight.dims() != [spec.rows(), spec.hidden_size] {
            return Err(Qwen3VlError::InvalidInput(format!(
                "embedding chunk tensor has shape {:?}, expected [{}, {}]",
                weight.dims(),
                spec.rows(),
                spec.hidden_size
            )));
        }
        Ok(Self { spec, weight })
    }
}

/// Activation accumulator for token-ID-routed row chunks. A chunk can be dropped immediately
/// after [`Self::apply_chunk`] returns.
pub struct ChunkedEmbeddingState<B: Backend> {
    input_ids: Vec<i64>,
    covered: Vec<bool>,
    batch: usize,
    sequence: usize,
    total_rows: usize,
    hidden_size: usize,
    output: Option<Tensor<B, 3>>,
    device: B::Device,
}

impl<B: Backend> ChunkedEmbeddingState<B> {
    pub fn new(
        input_ids: &[Vec<i64>],
        total_rows: usize,
        hidden_size: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let batch = input_ids.len();
        let sequence = input_ids.first().map_or(0, Vec::len);
        if batch == 0
            || sequence == 0
            || input_ids.iter().any(|row| row.len() != sequence)
            || input_ids
                .iter()
                .flatten()
                .any(|&id| id < 0 || id as usize >= total_rows)
        {
            return Err(Qwen3VlError::InvalidInput(
                "chunked embedding ids must be a non-empty rectangular in-vocabulary batch".into(),
            ));
        }
        Ok(Self {
            input_ids: input_ids.iter().flatten().copied().collect(),
            covered: vec![false; batch * sequence],
            batch,
            sequence,
            total_rows,
            hidden_size,
            output: None,
            device: device.clone(),
        })
    }

    pub fn apply_chunk(&mut self, chunk: &EmbeddingRowChunk<B>) -> Result<()> {
        if chunk.spec.total_rows != self.total_rows || chunk.spec.hidden_size != self.hidden_size {
            return Err(Qwen3VlError::InvalidInput(
                "embedding chunk metadata differs from accumulator".into(),
            ));
        }
        let routed = self
            .input_ids
            .iter()
            .enumerate()
            .filter_map(|(position, &id)| {
                let id = id as usize;
                chunk
                    .spec
                    .row_range
                    .contains(&id)
                    .then(|| (position, (id - chunk.spec.row_range.start) as i64))
            })
            .collect::<Vec<_>>();
        if routed.is_empty() {
            return Ok(());
        }
        if routed.iter().any(|(position, _)| self.covered[*position]) {
            return Err(Qwen3VlError::InvalidInput(
                "overlapping embedding row chunks route the same token twice".into(),
            ));
        }
        let positions = Tensor::<B, 1, Int>::from_data(
            TensorData::new(
                routed
                    .iter()
                    .map(|(position, _)| *position as i64)
                    .collect::<Vec<_>>(),
                [routed.len()],
            ),
            &self.device,
        );
        let local_ids = Tensor::<B, 1, Int>::from_data(
            TensorData::new(
                routed.iter().map(|(_, id)| *id).collect::<Vec<_>>(),
                [routed.len()],
            ),
            &self.device,
        );
        let selected = chunk.weight.clone().select(0, local_ids);
        let output = self.output.take().unwrap_or_else(|| {
            Tensor::<B, 2>::zeros([self.batch * self.sequence, self.hidden_size], &self.device)
                .cast(chunk.weight.dtype())
                .reshape([self.batch, self.sequence, self.hidden_size])
        });
        self.output = Some(
            output
                .reshape([self.batch * self.sequence, self.hidden_size])
                .select_assign(0, positions, selected, IndexingUpdateOp::Add)
                .reshape([self.batch, self.sequence, self.hidden_size]),
        );
        for (position, _) in routed {
            self.covered[position] = true;
        }
        Ok(())
    }

    pub fn finish(self) -> Result<Tensor<B, 3>> {
        if self.covered.iter().any(|covered| !covered) {
            return Err(Qwen3VlError::InvalidInput(
                "embedding row chunks did not cover every input token".into(),
            ));
        }
        self.output
            .ok_or_else(|| Qwen3VlError::InvalidInput("no embedding chunk was applied".into()))
    }
}

/// An optional streamed vocabulary projection slice. Consumers can inspect/sample each row range
/// without ever allocating `[batch, sequence, full_vocabulary]` logits.
#[derive(Debug, Clone)]
pub struct OutputProjectionRowChunk<B: Backend> {
    pub spec: RowChunkSpec,
    pub weight: Tensor<B, 2>,
}

impl<B: Backend> OutputProjectionRowChunk<B> {
    pub fn project(&self, hidden_states: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        let [batch, sequence, hidden] = hidden_states.dims();
        if self.weight.dims() != [self.spec.rows(), hidden] || hidden != self.spec.hidden_size {
            return Err(Qwen3VlError::InvalidInput(
                "LM-head row chunk shape does not match hidden states".into(),
            ));
        }
        Ok(hidden_states
            .reshape([batch * sequence, hidden])
            .matmul(self.weight.clone().transpose())
            .reshape([batch, sequence, self.spec.rows()]))
    }
}

/// Patch embedding and learned position table, independently loadable from all vision blocks.
#[derive(Module, Debug)]
pub struct Qwen3VlVisionPrelude<B: Backend> {
    pub patch_embed: Qwen3VlVisionPatchEmbed<B>,
    pub pos_embed: Embedding<B>,
    #[module(skip)]
    config: Qwen3VlVisionConfig,
}

impl<B: Backend> Qwen3VlVisionPrelude<B> {
    pub fn new(config: Qwen3VlVisionConfig, device: &B::Device) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            patch_embed: Qwen3VlVisionPatchEmbed::new(&config, device),
            pos_embed: EmbeddingConfig::new(config.num_position_embeddings, config.hidden_size)
                .init(device),
            config,
        })
    }

    /// Clone just the patch/position prelude from a resident model. Primarily useful for parity
    /// tests and migration tools; streamed production sources normally load this stage directly.
    pub fn from_resident(model: &Qwen3VlVisionModel<B>) -> Self {
        Self {
            patch_embed: model.patch_embed.clone(),
            pos_embed: model.pos_embed.clone(),
            config: model.config().clone(),
        }
    }

    pub fn begin(&self, patches: Tensor<B, 2>, grids: &[Grid]) -> Result<Qwen3VlVisionState<B>> {
        let [patch_count, patch_width] = patches.dims();
        let expected = grids.iter().map(|grid| grid.patch_count()).sum::<usize>();
        if patch_width != self.config.patch_volume() || patch_count != expected {
            return Err(Qwen3VlError::InvalidInput(format!(
                "vision prelude received [{patch_count}, {patch_width}], grids/config require [{expected}, {}]",
                self.config.patch_volume()
            )));
        }
        let plan = VisionPositionPlan::new(
            grids,
            self.config.spatial_merge_size,
            self.config.num_position_embeddings,
        )?;
        let mut hidden_states = self.patch_embed.forward(patches);
        let device = hidden_states.device();
        let dtype = self.pos_embed.weight.val().dtype();
        let mut positions =
            Tensor::<B, 2>::zeros([plan.patch_count(), self.config.hidden_size], &device)
                .cast(dtype);
        for corner in 0..4 {
            let indices = Tensor::<B, 2, Int>::from_data(
                TensorData::new(
                    plan.interpolation_indices[corner].clone(),
                    [1, plan.patch_count()],
                ),
                &device,
            );
            let embedding = self
                .pos_embed
                .forward(indices)
                .reshape([plan.patch_count(), self.config.hidden_size]);
            let weights = Tensor::<B, 2>::from_data(
                TensorData::new(
                    plan.interpolation_weights[corner].clone(),
                    [plan.patch_count(), 1],
                ),
                &device,
            )
            .cast(dtype);
            positions = positions + embedding * weights;
        }
        hidden_states = hidden_states + positions;
        let (cos, sin) = plan.vision_cos_sin::<B>(self.config.head_dim(), &device)?;
        Ok(Qwen3VlVisionState {
            hidden_states,
            plan,
            cos,
            sin,
            deepstack_features: Vec::new(),
            next_block: 0,
            config: self.config.clone(),
        })
    }
}

pub struct Qwen3VlVisionState<B: Backend> {
    pub hidden_states: Tensor<B, 2>,
    pub plan: VisionPositionPlan,
    cos: Tensor<B, 2>,
    sin: Tensor<B, 2>,
    deepstack_features: Vec<Tensor<B, 2>>,
    next_block: usize,
    config: Qwen3VlVisionConfig,
}

impl<B: Backend> Qwen3VlVisionState<B> {
    pub fn apply_block<O: Qwen3VlStageObserver<B>>(
        &mut self,
        index: usize,
        block: &Qwen3VlVisionBlock<B>,
        observer: &mut O,
    ) -> Result<()> {
        if index != self.next_block || index >= self.config.depth {
            return Err(Qwen3VlError::InvalidInput(format!(
                "vision block {index} is out of streamed order; expected {}",
                self.next_block
            )));
        }
        self.hidden_states = block.forward(
            self.hidden_states.clone(),
            &self.plan.frame_ranges,
            self.cos.clone(),
            self.sin.clone(),
        );
        self.next_block += 1;
        observer.rank2(
            &Qwen3VlStage::VisionBlock { index },
            self.hidden_states.clone(),
        )
    }

    pub fn capture_deepstack<O: Qwen3VlStageObserver<B>>(
        &mut self,
        merger_index: usize,
        merger: &Qwen3VlVisionPatchMerger<B>,
        observer: &mut O,
    ) -> Result<()> {
        let &after_block = self
            .config
            .deepstack_visual_indexes
            .get(merger_index)
            .ok_or_else(|| Qwen3VlError::InvalidInput("unknown deep-stack merger".into()))?;
        if self.next_block != after_block + 1 || self.deepstack_features.len() != merger_index {
            return Err(Qwen3VlError::InvalidInput(
                "deep-stack merger is out of streamed order".into(),
            ));
        }
        let feature = merger.forward(self.hidden_states.clone());
        observer.rank2(
            &Qwen3VlStage::VisionDeepstackMerger {
                index: merger_index,
                after_block,
            },
            feature.clone(),
        )?;
        self.deepstack_features.push(feature);
        Ok(())
    }

    pub fn finish<O: Qwen3VlStageObserver<B>>(
        self,
        merger: &Qwen3VlVisionPatchMerger<B>,
        observer: &mut O,
    ) -> Result<Qwen3VlVisionOutput<B>> {
        if self.next_block != self.config.depth
            || self.deepstack_features.len() != self.config.deepstack_visual_indexes.len()
        {
            return Err(Qwen3VlError::InvalidInput(
                "vision stream is incomplete".into(),
            ));
        }
        let pooler_output = merger.forward(self.hidden_states.clone());
        observer.rank2(&Qwen3VlStage::VisionFinalMerger, pooler_output.clone())?;
        Ok(Qwen3VlVisionOutput {
            last_hidden_state: self.hidden_states,
            pooler_output,
            deepstack_features: self.deepstack_features,
        })
    }
}

pub struct Qwen3VlTextState<B: Backend> {
    pub hidden_states: Tensor<B, 3>,
    attention_mask: Option<Tensor<B, 2, Bool>>,
    cos: Tensor<B, 3>,
    sin: Tensor<B, 3>,
    deepstack: Option<DeepstackEmbeddings<B>>,
    next_layer: usize,
    config: Qwen3VlTextConfig,
}

impl<B: Backend> Qwen3VlTextState<B> {
    pub fn new(
        hidden_states: Tensor<B, 3>,
        attention_mask: Option<Tensor<B, 2, Bool>>,
        positions: &MropePositionIds,
        deepstack: Option<DeepstackEmbeddings<B>>,
        config: Qwen3VlTextConfig,
    ) -> Result<Self> {
        let [batch, sequence, hidden] = hidden_states.dims();
        if hidden != config.hidden_size
            || positions.batch_size() != batch
            || positions.sequence_length() != sequence
            || attention_mask
                .as_ref()
                .is_some_and(|mask| mask.dims() != [batch, sequence])
        {
            return Err(Qwen3VlError::InvalidInput(
                "streamed text prelude shapes are inconsistent".into(),
            ));
        }
        if let Some(deepstack) = &deepstack
            && (deepstack.features.len() > config.num_hidden_layers
                || deepstack.features.iter().any(|feature| {
                    feature.dims() != [deepstack.token_indices.len(), config.hidden_size]
                })
                || deepstack
                    .token_indices
                    .iter()
                    .any(|&index| index >= batch * sequence))
        {
            return Err(Qwen3VlError::InvalidInput(
                "streamed deep-stack feature shapes are inconsistent".into(),
            ));
        }
        let (cos, sin) = positions.cos_sin::<B>(&config, &hidden_states.device())?;
        Ok(Self {
            hidden_states,
            attention_mask,
            cos,
            sin,
            deepstack,
            next_layer: 0,
            config,
        })
    }

    pub fn apply_layer<O: Qwen3VlStageObserver<B>>(
        &mut self,
        index: usize,
        layer: &Qwen3VlDecoderLayer<B>,
        observer: &mut O,
    ) -> Result<()> {
        if index != self.next_layer || index >= self.config.num_hidden_layers {
            return Err(Qwen3VlError::InvalidInput(format!(
                "text layer {index} is out of streamed order; expected {}",
                self.next_layer
            )));
        }
        let [batch, sequence, hidden] = self.hidden_states.dims();
        self.hidden_states = layer.forward(
            self.hidden_states.clone(),
            self.cos.clone(),
            self.sin.clone(),
            self.attention_mask.as_ref(),
        );
        if let Some(deepstack) = &self.deepstack
            && let Some(feature) = deepstack.features.get(index)
        {
            let indices = Tensor::<B, 1, Int>::from_data(
                TensorData::new(
                    deepstack
                        .token_indices
                        .iter()
                        .map(|&value| value as i64)
                        .collect::<Vec<_>>(),
                    [deepstack.token_indices.len()],
                ),
                &self.hidden_states.device(),
            );
            self.hidden_states = self
                .hidden_states
                .clone()
                .reshape([batch * sequence, hidden])
                .select_assign(0, indices, feature.clone(), IndexingUpdateOp::Add)
                .reshape([batch, sequence, hidden]);
        }
        self.next_layer += 1;
        observer.rank3(
            &Qwen3VlStage::TextBlock { index },
            self.hidden_states.clone(),
        )
    }

    pub fn finish<O: Qwen3VlStageObserver<B>>(
        mut self,
        norm: &RmsNorm<B>,
        observer: &mut O,
    ) -> Result<Tensor<B, 3>> {
        if self.next_layer != self.config.num_hidden_layers {
            return Err(Qwen3VlError::InvalidInput(
                "text stream is incomplete".into(),
            ));
        }
        self.hidden_states = norm.forward(self.hidden_states);
        observer.rank3(&Qwen3VlStage::TextFinalNorm, self.hidden_states.clone())?;
        Ok(self.hidden_states)
    }
}

/// Observer boundaries are exact semantic points suitable for parity fixtures and timing.
pub trait Qwen3VlStageObserver<B: Backend> {
    fn rank2(&mut self, _stage: &Qwen3VlStage, _activation: Tensor<B, 2>) -> Result<()> {
        Ok(())
    }

    fn rank3(&mut self, _stage: &Qwen3VlStage, _activation: Tensor<B, 3>) -> Result<()> {
        Ok(())
    }
}

impl<B: Backend> Qwen3VlStageObserver<B> for () {}

/// Source of verified short-lived base-model stages. Implementations may synchronously consume
/// bytes prefetched by an async CDN layer; every returned module can be dropped after its matching
/// state transition and [`Self::synchronize`].
///
/// Stage descriptors use canonical full checkpoint paths. A loader strips the descriptor's exact
/// module prefix and applies tensors to a *fresh, still-lazy* stage module. In particular, Burn
/// column-layout linear weights remain in checkpoint/source `[out, in]` shape and use the identity
/// store adapter; the parameter load mapper performs the one required transpose. Calling
/// `ModuleSnapshot::collect`, `Param::val`, or a forward method before applying a stage initializes
/// it, changes its validation shape to runtime `[in, out]`, and violates this contract.
pub trait Qwen3VlStageSource<B: Backend> {
    type Error;

    fn load_embedding_rows(
        &mut self,
        spec: &RowChunkSpec,
    ) -> core::result::Result<EmbeddingRowChunk<B>, Self::Error>;
    fn load_vision_prelude(&mut self)
    -> core::result::Result<Qwen3VlVisionPrelude<B>, Self::Error>;
    fn load_vision_block(
        &mut self,
        index: usize,
    ) -> core::result::Result<Qwen3VlVisionBlock<B>, Self::Error>;
    fn load_vision_deepstack_merger(
        &mut self,
        index: usize,
    ) -> core::result::Result<Qwen3VlVisionPatchMerger<B>, Self::Error>;
    fn load_vision_final_merger(
        &mut self,
    ) -> core::result::Result<Qwen3VlVisionPatchMerger<B>, Self::Error>;
    fn load_text_block(
        &mut self,
        index: usize,
    ) -> core::result::Result<Qwen3VlDecoderLayer<B>, Self::Error>;
    fn load_text_final_norm(&mut self) -> core::result::Result<RmsNorm<B>, Self::Error>;
    fn synchronize(&mut self) -> core::result::Result<(), Self::Error>;
}

/// Optional extension for consumers that require vocabulary logits. Base-model conditioning does
/// not implement or call this trait.
pub trait Qwen3VlCausalLmStageSource<B: Backend>: Qwen3VlStageSource<B> {
    fn load_lm_head_rows(
        &mut self,
        spec: &RowChunkSpec,
    ) -> core::result::Result<OutputProjectionRowChunk<B>, Self::Error>;
}

/// Synchronization behavior for a retained synchronous Qwen3-VL source.
///
/// [`Self::PerStage`] preserves the streaming executor's bounded-residency ordering exactly.
/// [`Self::Deferred`] records its synchronization requests without forwarding them, allowing a
/// caller to submit consecutive retained stages before explicitly calling
/// [`RetainingQwen3VlStageSource::synchronize_pending`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RetainingSynchronizationPolicy {
    /// Forward every synchronization request to the wrapped source.
    #[default]
    PerStage,
    /// Coalesce synchronization requests until the caller explicitly synchronizes.
    Deferred,
}

/// Opt-in retaining wrapper for repeated native inference on devices with enough memory.
///
/// The wrapped source performs each verified load at most once for a given row specification or
/// indexed semantic stage. Later requests return Burn module/tensor clones, which are shared
/// backend handles on Burn's WGPU backend rather than device-buffer copies. By default,
/// synchronization is delegated on every request, including cache hits, so execution ordering
/// remains identical to a non-retaining source. Callers may explicitly opt into deferred
/// synchronization when every requested stage is retained and can safely remain submitted until
/// the final output barrier.
///
/// This wrapper intentionally implements only the synchronous native source traits. Browser
/// runtimes should keep using an [`AsyncQwen3VlStageSource`] directly to preserve bounded stage
/// residency.
pub struct RetainingQwen3VlStageSource<B: Backend, S> {
    source: S,
    synchronization_policy: RetainingSynchronizationPolicy,
    synchronization_pending: bool,
    embedding_rows: Vec<EmbeddingRowChunk<B>>,
    vision_prelude: Option<Qwen3VlVisionPrelude<B>>,
    vision_blocks: Vec<(usize, Qwen3VlVisionBlock<B>)>,
    vision_deepstack_mergers: Vec<(usize, Qwen3VlVisionPatchMerger<B>)>,
    vision_final_merger: Option<Qwen3VlVisionPatchMerger<B>>,
    text_blocks: Vec<(usize, Qwen3VlDecoderLayer<B>)>,
    text_final_norm: Option<RmsNorm<B>>,
    lm_head_rows: Vec<OutputProjectionRowChunk<B>>,
}

impl<B: Backend, S> RetainingQwen3VlStageSource<B, S> {
    /// Wrap a verified source with an initially empty, retaining cache.
    pub const fn new(source: S) -> Self {
        Self {
            source,
            synchronization_policy: RetainingSynchronizationPolicy::PerStage,
            synchronization_pending: false,
            embedding_rows: Vec::new(),
            vision_prelude: None,
            vision_blocks: Vec::new(),
            vision_deepstack_mergers: Vec::new(),
            vision_final_merger: None,
            text_blocks: Vec::new(),
            text_final_norm: None,
            lm_head_rows: Vec::new(),
        }
    }

    /// Select synchronization behavior for this wrapper.
    ///
    /// The default is [`RetainingSynchronizationPolicy::PerStage`]. Deferred mode is intended for
    /// fully retained native execution; the caller remains responsible for calling
    /// [`Self::synchronize_pending`] before timing, readback, or other work that requires completed
    /// output.
    pub const fn with_synchronization_policy(
        mut self,
        synchronization_policy: RetainingSynchronizationPolicy,
    ) -> Self {
        self.synchronization_policy = synchronization_policy;
        self
    }

    /// Return the configured synchronization behavior.
    pub const fn synchronization_policy(&self) -> RetainingSynchronizationPolicy {
        self.synchronization_policy
    }

    /// Whether deferred execution has observed at least one synchronization request that the
    /// caller has not yet forwarded to the wrapped source.
    pub const fn has_pending_synchronization(&self) -> bool {
        self.synchronization_pending
    }

    /// Borrow the underlying verified source, for load statistics or lifecycle state.
    pub const fn source(&self) -> &S {
        &self.source
    }

    /// Mutably borrow the underlying verified source.
    ///
    /// This does not implicitly flush deferred work.
    pub fn source_mut(&mut self) -> &mut S {
        &mut self.source
    }

    /// Consume the wrapper and return its underlying source, dropping all retained weights.
    ///
    /// This does not implicitly flush deferred work.
    pub fn into_source(self) -> S {
        self.source
    }

    /// Number of independently loadable weight stages currently retained.
    pub fn cached_stage_count(&self) -> usize {
        self.embedding_rows.len()
            + usize::from(self.vision_prelude.is_some())
            + self.vision_blocks.len()
            + self.vision_deepstack_mergers.len()
            + usize::from(self.vision_final_merger.is_some())
            + self.text_blocks.len()
            + usize::from(self.text_final_norm.is_some())
            + self.lm_head_rows.len()
    }

    /// Drop every retained module and row tensor while preserving the wrapped source.
    ///
    /// This does not implicitly flush deferred work.
    pub fn clear(&mut self) {
        self.embedding_rows.clear();
        self.vision_prelude = None;
        self.vision_blocks.clear();
        self.vision_deepstack_mergers.clear();
        self.vision_final_merger = None;
        self.text_blocks.clear();
        self.text_final_norm = None;
        self.lm_head_rows.clear();
    }
}

impl<B, S> RetainingQwen3VlStageSource<B, S>
where
    B: Backend,
    S: Qwen3VlStageSource<B>,
{
    /// Forward one pending deferred synchronization request to the wrapped source.
    ///
    /// Repeated calls without newly submitted work are no-ops. If synchronization fails, the
    /// pending flag remains set so the caller can retry or surface the source error.
    pub fn synchronize_pending(&mut self) -> core::result::Result<(), S::Error> {
        if !self.synchronization_pending {
            return Ok(());
        }
        self.source.synchronize()?;
        self.synchronization_pending = false;
        Ok(())
    }
}

fn load_retained_indexed<T: Clone, E>(
    cache: &mut Vec<(usize, T)>,
    index: usize,
    load: impl FnOnce() -> core::result::Result<T, E>,
) -> core::result::Result<T, E> {
    if let Some((_, value)) = cache.iter().find(|(cached, _)| *cached == index) {
        return Ok(value.clone());
    }
    let value = load()?;
    cache.push((index, value.clone()));
    Ok(value)
}

impl<B, S> Qwen3VlStageSource<B> for RetainingQwen3VlStageSource<B, S>
where
    B: Backend,
    S: Qwen3VlStageSource<B>,
{
    type Error = S::Error;

    fn load_embedding_rows(
        &mut self,
        spec: &RowChunkSpec,
    ) -> core::result::Result<EmbeddingRowChunk<B>, Self::Error> {
        if let Some(chunk) = self.embedding_rows.iter().find(|chunk| chunk.spec == *spec) {
            return Ok(chunk.clone());
        }
        let chunk = self.source.load_embedding_rows(spec)?;
        self.embedding_rows.push(chunk.clone());
        Ok(chunk)
    }

    fn load_vision_prelude(
        &mut self,
    ) -> core::result::Result<Qwen3VlVisionPrelude<B>, Self::Error> {
        if let Some(prelude) = &self.vision_prelude {
            return Ok(prelude.clone());
        }
        let prelude = self.source.load_vision_prelude()?;
        self.vision_prelude = Some(prelude.clone());
        Ok(prelude)
    }

    fn load_vision_block(
        &mut self,
        index: usize,
    ) -> core::result::Result<Qwen3VlVisionBlock<B>, Self::Error> {
        let source = &mut self.source;
        load_retained_indexed(&mut self.vision_blocks, index, || {
            source.load_vision_block(index)
        })
    }

    fn load_vision_deepstack_merger(
        &mut self,
        index: usize,
    ) -> core::result::Result<Qwen3VlVisionPatchMerger<B>, Self::Error> {
        let source = &mut self.source;
        load_retained_indexed(&mut self.vision_deepstack_mergers, index, || {
            source.load_vision_deepstack_merger(index)
        })
    }

    fn load_vision_final_merger(
        &mut self,
    ) -> core::result::Result<Qwen3VlVisionPatchMerger<B>, Self::Error> {
        if let Some(merger) = &self.vision_final_merger {
            return Ok(merger.clone());
        }
        let merger = self.source.load_vision_final_merger()?;
        self.vision_final_merger = Some(merger.clone());
        Ok(merger)
    }

    fn load_text_block(
        &mut self,
        index: usize,
    ) -> core::result::Result<Qwen3VlDecoderLayer<B>, Self::Error> {
        let source = &mut self.source;
        load_retained_indexed(&mut self.text_blocks, index, || {
            source.load_text_block(index)
        })
    }

    fn load_text_final_norm(&mut self) -> core::result::Result<RmsNorm<B>, Self::Error> {
        if let Some(norm) = &self.text_final_norm {
            return Ok(norm.clone());
        }
        let norm = self.source.load_text_final_norm()?;
        self.text_final_norm = Some(norm.clone());
        Ok(norm)
    }

    fn synchronize(&mut self) -> core::result::Result<(), Self::Error> {
        match self.synchronization_policy {
            RetainingSynchronizationPolicy::PerStage => self.source.synchronize(),
            RetainingSynchronizationPolicy::Deferred => {
                self.synchronization_pending = true;
                Ok(())
            }
        }
    }
}

impl<B, S> Qwen3VlCausalLmStageSource<B> for RetainingQwen3VlStageSource<B, S>
where
    B: Backend,
    S: Qwen3VlCausalLmStageSource<B>,
{
    fn load_lm_head_rows(
        &mut self,
        spec: &RowChunkSpec,
    ) -> core::result::Result<OutputProjectionRowChunk<B>, Self::Error> {
        if let Some(chunk) = self.lm_head_rows.iter().find(|chunk| chunk.spec == *spec) {
            return Ok(chunk.clone());
        }
        let chunk = self.source.load_lm_head_rows(spec)?;
        self.lm_head_rows.push(chunk.clone());
        Ok(chunk)
    }
}

/// Asynchronous source of verified short-lived base-model stages.
///
/// This is the browser-facing counterpart to [`Qwen3VlStageSource`]. Futures deliberately have
/// no `Send` requirement: browser fetch, cache, and WebGPU handles commonly live on one wasm event
/// loop. Each returned module remains resident only until the orchestrator awaits
/// [`Self::synchronize`], drops it, and requests the following stage.
#[allow(async_fn_in_trait)]
pub trait AsyncQwen3VlStageSource<B: Backend> {
    type Error;

    async fn load_embedding_rows(
        &mut self,
        spec: &RowChunkSpec,
    ) -> core::result::Result<EmbeddingRowChunk<B>, Self::Error>;
    async fn load_vision_prelude(
        &mut self,
    ) -> core::result::Result<Qwen3VlVisionPrelude<B>, Self::Error>;
    async fn load_vision_block(
        &mut self,
        index: usize,
    ) -> core::result::Result<Qwen3VlVisionBlock<B>, Self::Error>;
    async fn load_vision_deepstack_merger(
        &mut self,
        index: usize,
    ) -> core::result::Result<Qwen3VlVisionPatchMerger<B>, Self::Error>;
    async fn load_vision_final_merger(
        &mut self,
    ) -> core::result::Result<Qwen3VlVisionPatchMerger<B>, Self::Error>;
    async fn load_text_block(
        &mut self,
        index: usize,
    ) -> core::result::Result<Qwen3VlDecoderLayer<B>, Self::Error>;
    async fn load_text_final_norm(&mut self) -> core::result::Result<RmsNorm<B>, Self::Error>;
    async fn synchronize(&mut self) -> core::result::Result<(), Self::Error>;
}

/// Optional asynchronous extension for consumers that require vocabulary logits.
#[allow(async_fn_in_trait)]
pub trait AsyncQwen3VlCausalLmStageSource<B: Backend>: AsyncQwen3VlStageSource<B> {
    async fn load_lm_head_rows(
        &mut self,
        spec: &RowChunkSpec,
    ) -> core::result::Result<OutputProjectionRowChunk<B>, Self::Error>;
}

/// Typed holder for a source and its validated plan; orchestration can borrow the source while
/// retaining all activation states outside it.
pub struct StreamingQwen3Vl<B: Backend, S> {
    pub plan: Qwen3VlStreamingPlan,
    pub source: S,
    query_chunk_size: usize,
    _backend: PhantomData<B>,
}

/// Separates model/input failures from artifact, transport, or backend synchronization failures.
#[derive(Debug)]
pub enum StreamingForwardError<E> {
    Model(Qwen3VlError),
    Source(E),
}

impl<E> From<Qwen3VlError> for StreamingForwardError<E> {
    fn from(error: Qwen3VlError) -> Self {
        Self::Model(error)
    }
}

impl<B: Backend, S> StreamingQwen3Vl<B, S> {
    pub fn new(plan: Qwen3VlStreamingPlan, source: S) -> Self {
        Self {
            plan,
            source,
            query_chunk_size: 128,
            _backend: PhantomData,
        }
    }

    pub fn set_query_chunk_size(&mut self, query_chunk_size: usize) {
        self.query_chunk_size = query_chunk_size.max(1);
    }
}

impl<B, S> StreamingQwen3Vl<B, S>
where
    B: Backend,
    S: Qwen3VlStageSource<B>,
{
    /// Execute the complete ordinary Qwen3-VL base model while retaining only activations and one
    /// semantic weight stage. The untied LM head is intentionally outside this conditioning path.
    pub fn forward_base<O: Qwen3VlStageObserver<B>>(
        &mut self,
        config: &Qwen3VlConfig,
        input: Qwen3VlModelInput<B>,
        observer: &mut O,
    ) -> core::result::Result<Qwen3VlModelOutput<B>, StreamingForwardError<S::Error>> {
        config.validate().map_err(StreamingForwardError::Model)?;
        self.plan
            .embedding_rows
            .validate()
            .map_err(StreamingForwardError::Model)?;
        let first_embedding = &self.plan.embedding_rows.chunks[0];
        if first_embedding.total_rows != config.text_config.vocab_size
            || first_embedding.hidden_size != config.text_config.hidden_size
        {
            return Err(StreamingForwardError::Model(Qwen3VlError::InvalidConfig(
                "streaming plan embedding rows do not match model config".into(),
            )));
        }

        let Qwen3VlModelInput {
            input_ids,
            attention_mask,
            position_ids,
            images,
            videos,
            output_hidden_states,
        } = input;
        let [batch, sequence] = input_ids.dims();
        let device = input_ids.device();
        let has_visual = images.is_some() || videos.is_some();
        if has_visual && position_ids.is_none() {
            return Err(StreamingForwardError::Model(Qwen3VlError::InvalidInput(
                "multimodal streamed forward requires processor-planned MRoPE position ids".into(),
            )));
        }

        // Token ids originate on the host processor. Reading this small tensor keeps the public
        // resident input contract reusable; callers that already have ids can construct it on CPU.
        let flat_ids = input_ids
            .into_data()
            .convert::<i64>()
            .to_vec::<i64>()
            .map_err(|error| {
                StreamingForwardError::Model(Qwen3VlError::InvalidInput(format!(
                    "failed to read token ids for row-routed embedding: {error}"
                )))
            })?;
        let host_ids = flat_ids
            .chunks_exact(sequence)
            .map(<[i64]>::to_vec)
            .collect::<Vec<_>>();
        if host_ids.len() != batch {
            return Err(StreamingForwardError::Model(Qwen3VlError::InvalidInput(
                "token-id tensor shape is inconsistent with its data".into(),
            )));
        }
        let mut embedding_state = ChunkedEmbeddingState::new(
            &host_ids,
            config.text_config.vocab_size,
            config.text_config.hidden_size,
            &device,
        )
        .map_err(StreamingForwardError::Model)?;
        for spec in &self.plan.embedding_rows.chunks {
            let chunk = self
                .source
                .load_embedding_rows(spec)
                .map_err(StreamingForwardError::Source)?;
            embedding_state
                .apply_chunk(&chunk)
                .map_err(StreamingForwardError::Model)?;
            self.source
                .synchronize()
                .map_err(StreamingForwardError::Source)?;
            drop(chunk);
        }
        let mut embeddings = embedding_state
            .finish()
            .map_err(StreamingForwardError::Model)?;
        let text_dtype = embeddings.dtype();
        let last_embedding_chunk = self.plan.embedding_rows.chunks.len() - 1;
        observer
            .rank3(
                &Qwen3VlStage::EmbeddingRows {
                    chunk: last_embedding_chunk,
                },
                embeddings.clone(),
            )
            .map_err(StreamingForwardError::Model)?;

        let mut pending_vision = Vec::new();
        let visual_inputs = [images, videos].into_iter().flatten().collect::<Vec<_>>();
        if !visual_inputs.is_empty() {
            let prelude = self
                .source
                .load_vision_prelude()
                .map_err(StreamingForwardError::Source)?;
            for visual in visual_inputs {
                let state = prelude
                    .begin(visual.patches, &visual.grids)
                    .map_err(StreamingForwardError::Model)?;
                observer
                    .rank2(&Qwen3VlStage::VisionPrelude, state.hidden_states.clone())
                    .map_err(StreamingForwardError::Model)?;
                pending_vision.push((state, visual.token_indices));
            }
            self.source
                .synchronize()
                .map_err(StreamingForwardError::Source)?;
            drop(prelude);

            for index in 0..config.vision_config.depth {
                let mut block = self
                    .source
                    .load_vision_block(index)
                    .map_err(StreamingForwardError::Source)?;
                block.attn.set_query_chunk_size(self.query_chunk_size);
                for (state, _) in &mut pending_vision {
                    state
                        .apply_block(index, &block, observer)
                        .map_err(StreamingForwardError::Model)?;
                }
                self.source
                    .synchronize()
                    .map_err(StreamingForwardError::Source)?;
                drop(block);

                if let Some(merger_index) = config
                    .vision_config
                    .deepstack_visual_indexes
                    .iter()
                    .position(|&after| after == index)
                {
                    let merger = self
                        .source
                        .load_vision_deepstack_merger(merger_index)
                        .map_err(StreamingForwardError::Source)?;
                    for (state, _) in &mut pending_vision {
                        state
                            .capture_deepstack(merger_index, &merger, observer)
                            .map_err(StreamingForwardError::Model)?;
                    }
                    self.source
                        .synchronize()
                        .map_err(StreamingForwardError::Source)?;
                    drop(merger);
                }
            }
        }

        let mut indexed_visual_outputs = Vec::new();
        if !pending_vision.is_empty() {
            let merger = self
                .source
                .load_vision_final_merger()
                .map_err(StreamingForwardError::Source)?;
            for (state, token_indices) in pending_vision {
                let output = state
                    .finish(&merger, observer)
                    .map_err(StreamingForwardError::Model)?;
                indexed_visual_outputs.push((output, token_indices));
            }
            self.source
                .synchronize()
                .map_err(StreamingForwardError::Source)?;
            drop(merger);
        }

        let mut visual_outputs = Vec::with_capacity(indexed_visual_outputs.len());
        let mut visual_indices = Vec::new();
        let mut deepstack_by_layer =
            vec![Vec::new(); config.vision_config.deepstack_visual_indexes.len()];
        for (output, token_indices) in indexed_visual_outputs {
            validate_visual_destinations(
                &output,
                &token_indices,
                batch * sequence,
                config.text_config.hidden_size,
            )
            .map_err(StreamingForwardError::Model)?;
            embeddings = assign_visual_features(
                embeddings,
                &token_indices,
                output.pooler_output.clone().cast(text_dtype),
            );
            append_deepstack(&mut deepstack_by_layer, &output, text_dtype)
                .map_err(StreamingForwardError::Model)?;
            visual_indices.extend(token_indices);
            visual_outputs.push(output);
        }
        let deepstack = has_visual.then(|| DeepstackEmbeddings {
            token_indices: visual_indices,
            features: deepstack_by_layer
                .into_iter()
                .map(|parts| {
                    if parts.is_empty() {
                        Tensor::<B, 2>::zeros([0, config.text_config.hidden_size], &device)
                    } else {
                        Tensor::cat(parts, 0)
                    }
                })
                .collect(),
        });

        let positions =
            position_ids.unwrap_or_else(|| MropePositionIds::text_only(batch, sequence));
        let position_deltas = positions.deltas().to_vec();
        let mut text_state = Qwen3VlTextState::new(
            embeddings,
            attention_mask,
            &positions,
            deepstack,
            config.text_config.clone(),
        )
        .map_err(StreamingForwardError::Model)?;
        let mut hidden_states = output_hidden_states.then(Vec::new);
        for index in 0..config.text_config.num_hidden_layers {
            if let Some(hidden_states) = &mut hidden_states {
                hidden_states.push(text_state.hidden_states.clone());
            }
            let mut layer = self
                .source
                .load_text_block(index)
                .map_err(StreamingForwardError::Source)?;
            layer.self_attn.set_query_chunk_size(self.query_chunk_size);
            text_state
                .apply_layer(index, &layer, observer)
                .map_err(StreamingForwardError::Model)?;
            self.source
                .synchronize()
                .map_err(StreamingForwardError::Source)?;
            drop(layer);
        }
        let norm = self
            .source
            .load_text_final_norm()
            .map_err(StreamingForwardError::Source)?;
        let last_hidden_state = text_state
            .finish(&norm, observer)
            .map_err(StreamingForwardError::Model)?;
        self.source
            .synchronize()
            .map_err(StreamingForwardError::Source)?;
        drop(norm);
        if let Some(hidden_states) = &mut hidden_states {
            hidden_states.push(last_hidden_state.clone());
        }

        Ok(Qwen3VlModelOutput {
            last_hidden_state,
            hidden_states,
            vision_output: combine_vision_outputs(visual_outputs),
            position_deltas,
        })
    }
}

impl<B, S> StreamingQwen3Vl<B, S>
where
    B: Backend,
    S: AsyncQwen3VlStageSource<B>,
{
    /// Asynchronously execute the complete ordinary Qwen3-VL base model while retaining only
    /// activations and one semantic weight stage.
    ///
    /// Every fetch and synchronization is awaited. The just-used row chunk or module is dropped
    /// before the following fetch begins, which preserves the bounded-residency contract for
    /// browser caches and WebGPU. Token-id readback also uses Burn's asynchronous tensor API so
    /// this path never relies on a native-only blocking reader.
    pub async fn forward_base_async<O: Qwen3VlStageObserver<B>>(
        &mut self,
        config: &Qwen3VlConfig,
        input: Qwen3VlModelInput<B>,
        observer: &mut O,
    ) -> core::result::Result<Qwen3VlModelOutput<B>, StreamingForwardError<S::Error>> {
        config.validate().map_err(StreamingForwardError::Model)?;
        self.plan
            .embedding_rows
            .validate()
            .map_err(StreamingForwardError::Model)?;
        let first_embedding = &self.plan.embedding_rows.chunks[0];
        if first_embedding.total_rows != config.text_config.vocab_size
            || first_embedding.hidden_size != config.text_config.hidden_size
        {
            return Err(StreamingForwardError::Model(Qwen3VlError::InvalidConfig(
                "streaming plan embedding rows do not match model config".into(),
            )));
        }

        let Qwen3VlModelInput {
            input_ids,
            attention_mask,
            position_ids,
            images,
            videos,
            output_hidden_states,
        } = input;
        let [batch, sequence] = input_ids.dims();
        let device = input_ids.device();
        let has_visual = images.is_some() || videos.is_some();
        if has_visual && position_ids.is_none() {
            return Err(StreamingForwardError::Model(Qwen3VlError::InvalidInput(
                "multimodal streamed forward requires processor-planned MRoPE position ids".into(),
            )));
        }

        let input_data = input_ids.into_data_async().await.map_err(|error| {
            StreamingForwardError::Model(Qwen3VlError::InvalidInput(format!(
                "failed to asynchronously read token ids for row-routed embedding: {error}"
            )))
        })?;
        let flat_ids = input_data
            .convert::<i64>()
            .to_vec::<i64>()
            .map_err(|error| {
                StreamingForwardError::Model(Qwen3VlError::InvalidInput(format!(
                    "failed to decode token ids for row-routed embedding: {error}"
                )))
            })?;
        let host_ids = flat_ids
            .chunks_exact(sequence)
            .map(<[i64]>::to_vec)
            .collect::<Vec<_>>();
        if host_ids.len() != batch {
            return Err(StreamingForwardError::Model(Qwen3VlError::InvalidInput(
                "token-id tensor shape is inconsistent with its data".into(),
            )));
        }
        let mut embedding_state = ChunkedEmbeddingState::new(
            &host_ids,
            config.text_config.vocab_size,
            config.text_config.hidden_size,
            &device,
        )
        .map_err(StreamingForwardError::Model)?;
        for spec in &self.plan.embedding_rows.chunks {
            let chunk = self
                .source
                .load_embedding_rows(spec)
                .await
                .map_err(StreamingForwardError::Source)?;
            embedding_state
                .apply_chunk(&chunk)
                .map_err(StreamingForwardError::Model)?;
            self.source
                .synchronize()
                .await
                .map_err(StreamingForwardError::Source)?;
            drop(chunk);
        }
        let mut embeddings = embedding_state
            .finish()
            .map_err(StreamingForwardError::Model)?;
        let text_dtype = embeddings.dtype();
        let last_embedding_chunk = self.plan.embedding_rows.chunks.len() - 1;
        observer
            .rank3(
                &Qwen3VlStage::EmbeddingRows {
                    chunk: last_embedding_chunk,
                },
                embeddings.clone(),
            )
            .map_err(StreamingForwardError::Model)?;

        let mut pending_vision = Vec::new();
        let visual_inputs = [images, videos].into_iter().flatten().collect::<Vec<_>>();
        if !visual_inputs.is_empty() {
            let prelude = self
                .source
                .load_vision_prelude()
                .await
                .map_err(StreamingForwardError::Source)?;
            for visual in visual_inputs {
                let state = prelude
                    .begin(visual.patches, &visual.grids)
                    .map_err(StreamingForwardError::Model)?;
                observer
                    .rank2(&Qwen3VlStage::VisionPrelude, state.hidden_states.clone())
                    .map_err(StreamingForwardError::Model)?;
                pending_vision.push((state, visual.token_indices));
            }
            self.source
                .synchronize()
                .await
                .map_err(StreamingForwardError::Source)?;
            drop(prelude);

            for index in 0..config.vision_config.depth {
                let mut block = self
                    .source
                    .load_vision_block(index)
                    .await
                    .map_err(StreamingForwardError::Source)?;
                block.attn.set_query_chunk_size(self.query_chunk_size);
                for (state, _) in &mut pending_vision {
                    state
                        .apply_block(index, &block, observer)
                        .map_err(StreamingForwardError::Model)?;
                }
                self.source
                    .synchronize()
                    .await
                    .map_err(StreamingForwardError::Source)?;
                drop(block);

                if let Some(merger_index) = config
                    .vision_config
                    .deepstack_visual_indexes
                    .iter()
                    .position(|&after| after == index)
                {
                    let merger = self
                        .source
                        .load_vision_deepstack_merger(merger_index)
                        .await
                        .map_err(StreamingForwardError::Source)?;
                    for (state, _) in &mut pending_vision {
                        state
                            .capture_deepstack(merger_index, &merger, observer)
                            .map_err(StreamingForwardError::Model)?;
                    }
                    self.source
                        .synchronize()
                        .await
                        .map_err(StreamingForwardError::Source)?;
                    drop(merger);
                }
            }
        }

        let mut indexed_visual_outputs = Vec::new();
        if !pending_vision.is_empty() {
            let merger = self
                .source
                .load_vision_final_merger()
                .await
                .map_err(StreamingForwardError::Source)?;
            for (state, token_indices) in pending_vision {
                let output = state
                    .finish(&merger, observer)
                    .map_err(StreamingForwardError::Model)?;
                indexed_visual_outputs.push((output, token_indices));
            }
            self.source
                .synchronize()
                .await
                .map_err(StreamingForwardError::Source)?;
            drop(merger);
        }

        let mut visual_outputs = Vec::with_capacity(indexed_visual_outputs.len());
        let mut visual_indices = Vec::new();
        let mut deepstack_by_layer =
            vec![Vec::new(); config.vision_config.deepstack_visual_indexes.len()];
        for (output, token_indices) in indexed_visual_outputs {
            validate_visual_destinations(
                &output,
                &token_indices,
                batch * sequence,
                config.text_config.hidden_size,
            )
            .map_err(StreamingForwardError::Model)?;
            embeddings = assign_visual_features(
                embeddings,
                &token_indices,
                output.pooler_output.clone().cast(text_dtype),
            );
            append_deepstack(&mut deepstack_by_layer, &output, text_dtype)
                .map_err(StreamingForwardError::Model)?;
            visual_indices.extend(token_indices);
            visual_outputs.push(output);
        }
        let deepstack = has_visual.then(|| DeepstackEmbeddings {
            token_indices: visual_indices,
            features: deepstack_by_layer
                .into_iter()
                .map(|parts| {
                    if parts.is_empty() {
                        Tensor::<B, 2>::zeros([0, config.text_config.hidden_size], &device)
                    } else {
                        Tensor::cat(parts, 0)
                    }
                })
                .collect(),
        });

        let positions =
            position_ids.unwrap_or_else(|| MropePositionIds::text_only(batch, sequence));
        let position_deltas = positions.deltas().to_vec();
        let mut text_state = Qwen3VlTextState::new(
            embeddings,
            attention_mask,
            &positions,
            deepstack,
            config.text_config.clone(),
        )
        .map_err(StreamingForwardError::Model)?;
        let mut hidden_states = output_hidden_states.then(Vec::new);
        for index in 0..config.text_config.num_hidden_layers {
            if let Some(hidden_states) = &mut hidden_states {
                hidden_states.push(text_state.hidden_states.clone());
            }
            let mut layer = self
                .source
                .load_text_block(index)
                .await
                .map_err(StreamingForwardError::Source)?;
            layer.self_attn.set_query_chunk_size(self.query_chunk_size);
            text_state
                .apply_layer(index, &layer, observer)
                .map_err(StreamingForwardError::Model)?;
            self.source
                .synchronize()
                .await
                .map_err(StreamingForwardError::Source)?;
            drop(layer);
        }
        let norm = self
            .source
            .load_text_final_norm()
            .await
            .map_err(StreamingForwardError::Source)?;
        let last_hidden_state = text_state
            .finish(&norm, observer)
            .map_err(StreamingForwardError::Model)?;
        self.source
            .synchronize()
            .await
            .map_err(StreamingForwardError::Source)?;
        drop(norm);
        if let Some(hidden_states) = &mut hidden_states {
            hidden_states.push(last_hidden_state.clone());
        }

        Ok(Qwen3VlModelOutput {
            last_hidden_state,
            hidden_states,
            vision_output: combine_vision_outputs(visual_outputs),
            position_deltas,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Qwen3VlModel, Qwen3VlModelInput, Qwen3VlVisualInput, config::tiny_config};
    use burn_ndarray::NdArray;
    use core::convert::Infallible;

    struct ResidentStageSource<B: Backend> {
        model: Qwen3VlModel<B>,
        synchronizations: usize,
        loads: Vec<Qwen3VlStage>,
    }

    impl<B: Backend> Qwen3VlStageSource<B> for ResidentStageSource<B> {
        type Error = Infallible;

        fn load_embedding_rows(
            &mut self,
            spec: &RowChunkSpec,
        ) -> core::result::Result<EmbeddingRowChunk<B>, Self::Error> {
            self.loads.push(Qwen3VlStage::EmbeddingRows {
                chunk: spec.chunk_index,
            });
            Ok(EmbeddingRowChunk::new(
                spec.clone(),
                self.model
                    .language_model
                    .embed_tokens
                    .weight
                    .val()
                    .slice([spec.row_range.clone(), 0..spec.hidden_size]),
            )
            .unwrap())
        }

        fn load_vision_prelude(
            &mut self,
        ) -> core::result::Result<Qwen3VlVisionPrelude<B>, Self::Error> {
            self.loads.push(Qwen3VlStage::VisionPrelude);
            Ok(Qwen3VlVisionPrelude::from_resident(&self.model.visual))
        }

        fn load_vision_block(
            &mut self,
            index: usize,
        ) -> core::result::Result<Qwen3VlVisionBlock<B>, Self::Error> {
            self.loads.push(Qwen3VlStage::VisionBlock { index });
            Ok(self.model.visual.blocks[index].clone())
        }

        fn load_vision_deepstack_merger(
            &mut self,
            index: usize,
        ) -> core::result::Result<Qwen3VlVisionPatchMerger<B>, Self::Error> {
            self.loads.push(Qwen3VlStage::VisionDeepstackMerger {
                index,
                after_block: self.model.visual.config().deepstack_visual_indexes[index],
            });
            Ok(self.model.visual.deepstack_merger_list[index].clone())
        }

        fn load_vision_final_merger(
            &mut self,
        ) -> core::result::Result<Qwen3VlVisionPatchMerger<B>, Self::Error> {
            self.loads.push(Qwen3VlStage::VisionFinalMerger);
            Ok(self.model.visual.merger.clone())
        }

        fn load_text_block(
            &mut self,
            index: usize,
        ) -> core::result::Result<Qwen3VlDecoderLayer<B>, Self::Error> {
            self.loads.push(Qwen3VlStage::TextBlock { index });
            Ok(self.model.language_model.layers[index].clone())
        }

        fn load_text_final_norm(&mut self) -> core::result::Result<RmsNorm<B>, Self::Error> {
            self.loads.push(Qwen3VlStage::TextFinalNorm);
            Ok(self.model.language_model.norm.clone())
        }

        fn synchronize(&mut self) -> core::result::Result<(), Self::Error> {
            self.synchronizations += 1;
            Ok(())
        }
    }

    impl<B: Backend> AsyncQwen3VlStageSource<B> for ResidentStageSource<B> {
        type Error = Infallible;

        async fn load_embedding_rows(
            &mut self,
            spec: &RowChunkSpec,
        ) -> core::result::Result<EmbeddingRowChunk<B>, Self::Error> {
            self.loads.push(Qwen3VlStage::EmbeddingRows {
                chunk: spec.chunk_index,
            });
            Ok(EmbeddingRowChunk::new(
                spec.clone(),
                self.model
                    .language_model
                    .embed_tokens
                    .weight
                    .val()
                    .slice([spec.row_range.clone(), 0..spec.hidden_size]),
            )
            .unwrap())
        }

        async fn load_vision_prelude(
            &mut self,
        ) -> core::result::Result<Qwen3VlVisionPrelude<B>, Self::Error> {
            self.loads.push(Qwen3VlStage::VisionPrelude);
            Ok(Qwen3VlVisionPrelude::from_resident(&self.model.visual))
        }

        async fn load_vision_block(
            &mut self,
            index: usize,
        ) -> core::result::Result<Qwen3VlVisionBlock<B>, Self::Error> {
            self.loads.push(Qwen3VlStage::VisionBlock { index });
            Ok(self.model.visual.blocks[index].clone())
        }

        async fn load_vision_deepstack_merger(
            &mut self,
            index: usize,
        ) -> core::result::Result<Qwen3VlVisionPatchMerger<B>, Self::Error> {
            self.loads.push(Qwen3VlStage::VisionDeepstackMerger {
                index,
                after_block: self.model.visual.config().deepstack_visual_indexes[index],
            });
            Ok(self.model.visual.deepstack_merger_list[index].clone())
        }

        async fn load_vision_final_merger(
            &mut self,
        ) -> core::result::Result<Qwen3VlVisionPatchMerger<B>, Self::Error> {
            self.loads.push(Qwen3VlStage::VisionFinalMerger);
            Ok(self.model.visual.merger.clone())
        }

        async fn load_text_block(
            &mut self,
            index: usize,
        ) -> core::result::Result<Qwen3VlDecoderLayer<B>, Self::Error> {
            self.loads.push(Qwen3VlStage::TextBlock { index });
            Ok(self.model.language_model.layers[index].clone())
        }

        async fn load_text_final_norm(&mut self) -> core::result::Result<RmsNorm<B>, Self::Error> {
            self.loads.push(Qwen3VlStage::TextFinalNorm);
            Ok(self.model.language_model.norm.clone())
        }

        async fn synchronize(&mut self) -> core::result::Result<(), Self::Error> {
            self.synchronizations += 1;
            Ok(())
        }
    }

    fn block_on_immediate<F: core::future::Future>(future: F) -> F::Output {
        let mut future = std::pin::pin!(future);
        let waker = std::task::Waker::noop();
        let mut context = std::task::Context::from_waker(waker);
        loop {
            match future.as_mut().poll(&mut context) {
                std::task::Poll::Ready(output) => return output,
                std::task::Poll::Pending => std::thread::yield_now(),
            }
        }
    }

    #[derive(Default)]
    struct RecordingObserver {
        stages: Vec<Qwen3VlStage>,
    }

    impl<B: Backend> Qwen3VlStageObserver<B> for RecordingObserver {
        fn rank2(&mut self, stage: &Qwen3VlStage, _activation: Tensor<B, 2>) -> Result<()> {
            self.stages.push(stage.clone());
            Ok(())
        }

        fn rank3(&mut self, stage: &Qwen3VlStage, _activation: Tensor<B, 3>) -> Result<()> {
            self.stages.push(stage.clone());
            Ok(())
        }
    }

    #[test]
    fn released_embedding_chunks_are_bounded_and_complete_correctness() {
        let mut config = tiny_config();
        config.text_config.vocab_size = 151_936;
        config.text_config.hidden_size = 4096;
        config.text_config.num_attention_heads = 32;
        config.text_config.num_key_value_heads = 8;
        config.text_config.head_dim = Some(128);
        config
            .text_config
            .rope_scaling
            .as_mut()
            .unwrap()
            .mrope_section = [24, 20, 20];
        config.vision_config.out_hidden_size = 4096;
        let plan = RowChunkPlan::even(151_936, 4096, 6, 2).unwrap();
        assert_eq!(plan.chunks.len(), 6);
        assert_eq!(plan.chunks.first().unwrap().row_range.start, 0);
        assert_eq!(plan.chunks.last().unwrap().row_range.end, 151_936);
        assert!(
            plan.chunks
                .iter()
                .all(|chunk| chunk.byte_len() < 256 * 1024 * 1024)
        );
    }

    #[test]
    fn hybrid_webgpu_policy_keeps_every_stage_below_binding_limit_correctness() {
        let mut config = tiny_config();
        config.text_config.vocab_size = 151_936;
        config.text_config.hidden_size = 4096;
        config.text_config.intermediate_size = 12_288;
        config.text_config.num_hidden_layers = 36;
        config.text_config.num_attention_heads = 32;
        config.text_config.num_key_value_heads = 8;
        config.text_config.head_dim = Some(128);
        config
            .text_config
            .rope_scaling
            .as_mut()
            .unwrap()
            .mrope_section = [24, 20, 20];
        config.vision_config.depth = 27;
        config.vision_config.hidden_size = 1152;
        config.vision_config.intermediate_size = 4304;
        config.vision_config.num_heads = 16;
        config.vision_config.patch_size = 16;
        config.vision_config.temporal_patch_size = 2;
        config.vision_config.spatial_merge_size = 2;
        config.vision_config.out_hidden_size = 4096;
        config.vision_config.in_channels = 3;
        config.vision_config.num_position_embeddings = 2304;
        config.vision_config.deepstack_visual_indexes = vec![8, 16, 24];
        let plan = Qwen3VlStreamingPlan::released_f16(&config, false).unwrap();
        let policy = Qwen3VlStageDTypePolicy::released_hybrid();
        let largest_vision = plan
            .stages
            .iter()
            .filter(|stage| policy.for_stage(&stage.stage) == Qwen3VlStageDType::F32)
            .map(|stage| stage.byte_len(4).unwrap())
            .max()
            .unwrap();
        assert_eq!(largest_vision, 160_503_808);
        assert!(largest_vision < 256 * 1024 * 1024);
        assert!(
            plan.stages
                .iter()
                .filter(|stage| matches!(stage.stage, Qwen3VlStage::EmbeddingRows { .. }))
                .all(|stage| stage.byte_len(2).unwrap() < 256 * 1024 * 1024)
        );
    }

    #[test]
    fn token_routed_embedding_chunks_match_full_lookup_correctness() {
        type B = NdArray<f32>;
        let device = Default::default();
        let plan = RowChunkPlan::even(8, 3, 3, 4).unwrap();
        let full = (0..24).map(|value| value as f32).collect::<Vec<_>>();
        let ids = vec![vec![7, 0, 3], vec![2, 5, 1]];
        let mut state = ChunkedEmbeddingState::<B>::new(&ids, 8, 3, &device).unwrap();
        for spec in &plan.chunks {
            let start = spec.row_range.start * 3;
            let end = spec.row_range.end * 3;
            let chunk = EmbeddingRowChunk::new(
                spec.clone(),
                Tensor::from_data(
                    TensorData::new(full[start..end].to_vec(), [spec.rows(), 3]),
                    &device,
                ),
            )
            .unwrap();
            state.apply_chunk(&chunk).unwrap();
        }
        let actual = state.finish().unwrap().into_data().to_vec::<f32>().unwrap();
        let expected = ids
            .iter()
            .flatten()
            .flat_map(|&id| full[id as usize * 3..id as usize * 3 + 3].iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn streamed_text_layers_match_resident_text_model_correctness() {
        type B = NdArray<f32>;
        let config = tiny_config().text_config;
        let device = Default::default();
        B::seed(&device, 31);
        let resident = crate::text::Qwen3VlTextModel::<B>::new(config.clone(), &device).unwrap();
        let ids = Tensor::<B, 2, Int>::from_data([[1, 2, 3]], &device);
        let embeddings = resident.embed(ids.clone());
        let positions = MropePositionIds::text_only(1, 3);
        let mut state =
            Qwen3VlTextState::new(embeddings, None, &positions, None, config.clone()).unwrap();
        for (index, layer) in resident.layers.iter().enumerate() {
            state.apply_layer(index, layer, &mut ()).unwrap();
        }
        let streamed = state
            .finish(&resident.norm, &mut ())
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let expected = resident
            .forward(ids, None, Some(&positions), None, false)
            .unwrap()
            .last_hidden_state
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for (left, right) in streamed.iter().zip(expected) {
            assert!((left - right).abs() < 1e-5);
        }
    }

    #[test]
    fn complete_sync_and_async_streamed_multimodal_forward_match_resident_correctness() {
        type B = NdArray<f32>;
        let config = tiny_config();
        let device = Default::default();
        B::seed(&device, 41);
        let resident = Qwen3VlModel::<B>::new(config.clone(), &device).unwrap();
        let embedding_rows = RowChunkPlan::even(
            config.text_config.vocab_size,
            config.text_config.hidden_size,
            3,
            size_of::<f32>(),
        )
        .unwrap();
        let plan = Qwen3VlStreamingPlan::new(&config, embedding_rows, None).unwrap();

        let positions = MropePositionIds::from_batch(
            &[vec![0, 1, 0]],
            &[vec![true; 3]],
            &[vec![Grid::new(1, 2, 2)]],
            &[vec![]],
            config.vision_config.spatial_merge_size,
        )
        .unwrap();
        let patches = Tensor::<B, 2>::from_data(
            TensorData::new(
                (0..4 * config.vision_config.patch_volume())
                    .map(|index| index as f32 / 1000.0)
                    .collect(),
                [4, config.vision_config.patch_volume()],
            ),
            &device,
        );
        let input = Qwen3VlModelInput {
            input_ids: Tensor::<B, 2, Int>::from_data([[1_i64, 60, 2]], &device),
            attention_mask: None,
            position_ids: Some(positions),
            images: Some(Qwen3VlVisualInput {
                patches,
                grids: vec![Grid::new(1, 2, 2)],
                token_indices: vec![1],
            }),
            videos: None,
            output_hidden_states: true,
        };
        let expected = resident.forward(input.clone()).unwrap();
        // Clone only after the resident pass has initialized every lazy parameter. Independent
        // clones of an uninitialized module intentionally draw independent random values.
        let source = ResidentStageSource {
            model: resident.clone(),
            synchronizations: 0,
            loads: Vec::new(),
        };
        let mut streamed = StreamingQwen3Vl::<B, _>::new(plan.clone(), source);
        let mut synchronous_observer = RecordingObserver::default();
        let actual = streamed
            .forward_base(&config, input.clone(), &mut synchronous_observer)
            .unwrap();
        let async_source = ResidentStageSource {
            model: resident.clone(),
            synchronizations: 0,
            loads: Vec::new(),
        };
        let mut async_streamed = StreamingQwen3Vl::<B, _>::new(plan, async_source);
        let mut asynchronous_observer = RecordingObserver::default();
        let async_actual = block_on_immediate(async_streamed.forward_base_async(
            &config,
            input,
            &mut asynchronous_observer,
        ))
        .unwrap();

        let expected_values = expected
            .last_hidden_state
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let actual_values = actual
            .last_hidden_state
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let async_values = async_actual
            .last_hidden_state
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let max_hidden_delta = actual_values
            .iter()
            .zip(expected_values)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        let max_async_delta = async_values
            .iter()
            .zip(&actual_values)
            .map(|(asynchronous, synchronous)| (asynchronous - synchronous).abs())
            .fold(0.0_f32, f32::max);
        let expected_vision = expected.vision_output.unwrap();
        let actual_vision = actual.vision_output.unwrap();
        let async_vision = async_actual.vision_output.unwrap();
        let expected_pool = expected_vision
            .pooler_output
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let actual_pool = actual_vision
            .pooler_output
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let async_pool = async_vision
            .pooler_output
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let max_pool_delta = actual_pool
            .iter()
            .zip(expected_pool)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        let max_async_pool_delta = async_pool
            .iter()
            .zip(&actual_pool)
            .map(|(asynchronous, synchronous)| (asynchronous - synchronous).abs())
            .fold(0.0_f32, f32::max);
        let hidden_stage_deltas = actual
            .hidden_states
            .as_ref()
            .unwrap()
            .iter()
            .zip(expected.hidden_states.as_ref().unwrap())
            .map(|(actual, expected)| {
                actual
                    .clone()
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap()
                    .into_iter()
                    .zip(expected.clone().into_data().to_vec::<f32>().unwrap())
                    .map(|(actual, expected)| (actual - expected).abs())
                    .fold(0.0_f32, f32::max)
            })
            .collect::<Vec<_>>();
        assert!(
            max_hidden_delta < 1e-5,
            "streamed hidden-state max delta {max_hidden_delta:e}; vision pool {max_pool_delta:e}; stages {hidden_stage_deltas:?}"
        );
        assert!(
            max_pool_delta < 1e-5,
            "streamed vision-pool max delta {max_pool_delta:e}"
        );
        assert!(
            max_async_delta < 1e-5 && max_async_pool_delta < 1e-5,
            "async versus sync max hidden delta {max_async_delta:e}; vision pool {max_async_pool_delta:e}"
        );
        assert_eq!(
            actual.hidden_states.as_ref().unwrap().len(),
            expected.hidden_states.as_ref().unwrap().len()
        );
        assert_eq!(
            async_actual.hidden_states.as_ref().unwrap().len(),
            actual.hidden_states.as_ref().unwrap().len()
        );
        assert_eq!(asynchronous_observer.stages, synchronous_observer.stages);
        let expected_synchronizations = 3
            + 1
            + config.vision_config.depth
            + config.vision_config.deepstack_visual_indexes.len()
            + 1
            + config.text_config.num_hidden_layers
            + 1;
        assert_eq!(streamed.source.synchronizations, expected_synchronizations);
        assert_eq!(
            async_streamed.source.synchronizations,
            expected_synchronizations
        );
    }

    #[test]
    fn retaining_source_loads_each_stage_once_across_two_forwards_correctness() {
        type B = NdArray<f32>;
        let config = tiny_config();
        let device = Default::default();
        B::seed(&device, 53);
        let resident = Qwen3VlModel::<B>::new(config.clone(), &device).unwrap();
        let embedding_rows = RowChunkPlan::even(
            config.text_config.vocab_size,
            config.text_config.hidden_size,
            3,
            size_of::<f32>(),
        )
        .unwrap();
        let plan = Qwen3VlStreamingPlan::new(&config, embedding_rows, None).unwrap();
        let expected_loads = plan
            .stages
            .iter()
            .map(|descriptor| descriptor.stage.clone())
            .collect::<Vec<_>>();

        let positions = MropePositionIds::from_batch(
            &[vec![0, 1, 0]],
            &[vec![true; 3]],
            &[vec![Grid::new(1, 2, 2)]],
            &[vec![]],
            config.vision_config.spatial_merge_size,
        )
        .unwrap();
        let patches = Tensor::<B, 2>::from_data(
            TensorData::new(
                (0..4 * config.vision_config.patch_volume())
                    .map(|index| index as f32 / 997.0)
                    .collect(),
                [4, config.vision_config.patch_volume()],
            ),
            &device,
        );
        let input = Qwen3VlModelInput {
            input_ids: Tensor::<B, 2, Int>::from_data([[1_i64, 60, 2]], &device),
            attention_mask: None,
            position_ids: Some(positions),
            images: Some(Qwen3VlVisualInput {
                patches,
                grids: vec![Grid::new(1, 2, 2)],
                token_indices: vec![1],
            }),
            videos: None,
            output_hidden_states: false,
        };

        // Initialize the lazy resident parameters before cloning them into the verified test
        // source; otherwise independently cloned lazy initializers intentionally draw new values.
        resident.forward(input.clone()).unwrap();
        let source = ResidentStageSource {
            model: resident,
            synchronizations: 0,
            loads: Vec::new(),
        };
        let retaining = RetainingQwen3VlStageSource::new(source);
        let mut streamed = StreamingQwen3Vl::<B, _>::new(plan, retaining);
        assert_eq!(
            streamed.source.synchronization_policy(),
            RetainingSynchronizationPolicy::PerStage,
            "retaining sources must preserve per-stage barriers by default"
        );
        let first = streamed
            .forward_base(&config, input.clone(), &mut ())
            .unwrap();
        let second = streamed.forward_base(&config, input, &mut ()).unwrap();

        let first_hidden = first.last_hidden_state.into_data().to_vec::<f32>().unwrap();
        let second_hidden = second
            .last_hidden_state
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let hidden_delta = first_hidden
            .iter()
            .zip(second_hidden)
            .map(|(first, second)| (first - second).abs())
            .fold(0.0_f32, f32::max);
        let first_pool = first
            .vision_output
            .unwrap()
            .pooler_output
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let second_pool = second
            .vision_output
            .unwrap()
            .pooler_output
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let pool_delta = first_pool
            .iter()
            .zip(second_pool)
            .map(|(first, second)| (first - second).abs())
            .fold(0.0_f32, f32::max);

        assert!(hidden_delta < 1.0e-6, "hidden max delta {hidden_delta:e}");
        assert!(pool_delta < 1.0e-6, "vision max delta {pool_delta:e}");
        assert_eq!(streamed.source.source().loads, expected_loads);
        assert_eq!(streamed.source.cached_stage_count(), expected_loads.len());
        assert_eq!(
            streamed.source.source().synchronizations,
            2 * expected_loads.len(),
            "cache hits must preserve per-stage synchronization"
        );
    }

    #[test]
    fn deferred_retaining_source_leaves_final_barrier_to_caller_correctness() {
        type B = NdArray<f32>;
        let config = tiny_config();
        let device = Default::default();
        B::seed(&device, 59);
        let resident = Qwen3VlModel::<B>::new(config.clone(), &device).unwrap();
        let embedding_rows = RowChunkPlan::even(
            config.text_config.vocab_size,
            config.text_config.hidden_size,
            3,
            size_of::<f32>(),
        )
        .unwrap();
        let expected_internal_barriers =
            embedding_rows.chunks.len() + config.text_config.num_hidden_layers + 1;
        assert!(expected_internal_barriers > 1);
        let plan = Qwen3VlStreamingPlan::new(&config, embedding_rows, None).unwrap();
        let input = Qwen3VlModelInput {
            input_ids: Tensor::<B, 2, Int>::from_data([[1_i64, 2, 3]], &device),
            attention_mask: None,
            position_ids: Some(MropePositionIds::text_only(1, 3)),
            images: None,
            videos: None,
            output_hidden_states: false,
        };

        // Initialize lazy parameters before the retained source clones them, and keep this
        // resident result as the numerical reference for the deferred path.
        let expected = resident.forward(input.clone()).unwrap();
        let source = ResidentStageSource {
            model: resident,
            synchronizations: 0,
            loads: Vec::new(),
        };
        let retaining = RetainingQwen3VlStageSource::new(source)
            .with_synchronization_policy(RetainingSynchronizationPolicy::Deferred);
        let mut streamed = StreamingQwen3Vl::<B, _>::new(plan, retaining);
        let actual = streamed.forward_base(&config, input, &mut ()).unwrap();

        assert_eq!(
            streamed.source.source().loads.len(),
            expected_internal_barriers,
            "the complete text path must submit every semantic stage"
        );
        assert_eq!(
            streamed.source.source().synchronizations,
            0,
            "deferred mode must not forward intermediate or final executor barriers"
        );
        assert!(
            streamed.source.has_pending_synchronization(),
            "the returned output must retain a caller-visible pending barrier"
        );

        streamed.source.synchronize_pending().unwrap();
        assert_eq!(streamed.source.source().synchronizations, 1);
        assert!(!streamed.source.has_pending_synchronization());
        streamed.source.synchronize_pending().unwrap();
        assert_eq!(
            streamed.source.source().synchronizations,
            1,
            "a caller flush without newly submitted work must be a no-op"
        );

        let expected_hidden = expected
            .last_hidden_state
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let actual_hidden = actual
            .last_hidden_state
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let max_delta = actual_hidden
            .iter()
            .zip(expected_hidden)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_delta < 1.0e-5,
            "deferred hidden max delta {max_delta:e}"
        );
    }
}
