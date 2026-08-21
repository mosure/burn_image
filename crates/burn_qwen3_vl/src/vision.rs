//! Qwen3-VL packed vision transformer and spatial patch merger.

use burn::{
    module::Module,
    nn::{
        Embedding, EmbeddingConfig, Gelu, LayerNorm, LayerNormConfig,
        conv::{Conv3d, Conv3dConfig},
    },
    tensor::{DType, Int, Tensor, TensorData, activation, backend::Backend},
};

use crate::{
    Qwen3VlError, QwenLinear, QwenLinearConfig, Result, config::Qwen3VlVisionConfig,
    linear::qwen_linear_forward, outputs::Qwen3VlVisionOutput, processor::Grid,
};

const DEFAULT_QUERY_CHUNK_SIZE: usize = 128;
const VISION_ROPE_THETA: f64 = 10_000.0;

/// CPU-side positional plan in the exact spatial-merge-block ordering used by Qwen3-VL.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionPositionPlan {
    /// `(row, column)` coordinates for vision rotary embeddings.
    pub rotary_coordinates: Vec<[usize; 2]>,
    /// Four learned-table indices per patch for bilinear interpolation.
    pub interpolation_indices: [Vec<i64>; 4],
    /// Corresponding bilinear weights; each patch's four values sum to one.
    pub interpolation_weights: [Vec<f32>; 4],
    /// Exclusive frame boundaries used to keep packed images/frames attention-independent.
    pub frame_ranges: Vec<(usize, usize)>,
}

impl VisionPositionPlan {
    pub fn new(
        grids: &[Grid],
        spatial_merge_size: usize,
        num_position_embeddings: usize,
    ) -> Result<Self> {
        if grids.is_empty() {
            return Err(Qwen3VlError::InvalidInput(
                "vision position plan requires at least one grid".into(),
            ));
        }
        let table_side = (num_position_embeddings as f64).sqrt() as usize;
        if table_side * table_side != num_position_embeddings {
            return Err(Qwen3VlError::InvalidConfig(
                "vision position table must be square".into(),
            ));
        }
        let mut rotary_coordinates = Vec::new();
        let mut interpolation_indices = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut interpolation_weights = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut frame_ranges = Vec::new();
        let mut offset = 0;
        for &grid in grids {
            grid.validate(spatial_merge_size)?;
            let spatial_coordinates = merge_order_coordinates(grid.h, grid.w, spatial_merge_size);
            let spatial_interpolation = spatial_coordinates
                .iter()
                .map(|&[row, column]| bilinear_entry(row, column, grid.h, grid.w, table_side))
                .collect::<Vec<_>>();
            for _ in 0..grid.t {
                rotary_coordinates.extend_from_slice(&spatial_coordinates);
                for (indices, weights) in &spatial_interpolation {
                    for corner in 0..4 {
                        interpolation_indices[corner].push(indices[corner]);
                        interpolation_weights[corner].push(weights[corner]);
                    }
                }
                frame_ranges.push((offset, offset + grid.h * grid.w));
                offset += grid.h * grid.w;
            }
        }
        Ok(Self {
            rotary_coordinates,
            interpolation_indices,
            interpolation_weights,
            frame_ranges,
        })
    }

    pub fn patch_count(&self) -> usize {
        self.rotary_coordinates.len()
    }

    pub fn vision_cos_sin<B: Backend>(
        &self,
        head_dim: usize,
        device: &B::Device,
    ) -> Result<(Tensor<B, 2>, Tensor<B, 2>)> {
        if !head_dim.is_multiple_of(4) {
            return Err(Qwen3VlError::InvalidConfig(
                "vision attention head_dim must be divisible by four".into(),
            ));
        }
        // The upstream rotary module is initialized with head_dim / 2. Its even-index inverse
        // frequencies are applied to row and column independently, then duplicated for rotate-half.
        let coordinate_dim = head_dim / 2;
        let frequency_count = coordinate_dim / 2;
        let inverse = (0..frequency_count)
            .map(|index| VISION_ROPE_THETA.powf(-((2 * index) as f64) / coordinate_dim as f64))
            .collect::<Vec<_>>();
        let mut cos = Vec::with_capacity(self.patch_count() * head_dim);
        let mut sin = Vec::with_capacity(self.patch_count() * head_dim);
        for [row, column] in &self.rotary_coordinates {
            let mut angles = Vec::with_capacity(coordinate_dim);
            angles.extend(inverse.iter().map(|frequency| *row as f64 * frequency));
            angles.extend(inverse.iter().map(|frequency| *column as f64 * frequency));
            for _ in 0..2 {
                cos.extend(angles.iter().map(|angle| angle.cos() as f32));
                sin.extend(angles.iter().map(|angle| angle.sin() as f32));
            }
        }
        let shape = [self.patch_count(), head_dim];
        Ok((
            Tensor::from_data(TensorData::new(cos, shape), device),
            Tensor::from_data(TensorData::new(sin, shape), device),
        ))
    }
}

/// Interpolate the learned vision-position table without widening the full table.
///
/// A released Q4 profile stores this table as a packed rank-two tensor. Burn's ordinary
/// [`Embedding::forward`] leaves a selected result quantized, while `Tensor::cast` accepts only
/// ordinary floating source tensors. Select the small set of required rows while the table is
/// still packed, then dequantize only those rows before interpolation. Floating tables follow the
/// same path because `dequantize` is an identity for them.
pub(crate) fn interpolate_learned_positions<B: Backend>(
    pos_embed: &Embedding<B>,
    plan: &VisionPositionPlan,
    hidden_size: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let patches = plan.patch_count();
    let table = pos_embed.weight.val();
    let mut output = None;
    for corner in 0..4 {
        let indices = Tensor::<B, 1, Int>::from_data(
            TensorData::new(plan.interpolation_indices[corner].clone(), [patches]),
            device,
        );
        let embedding = table.clone().select(0, indices).dequantize();
        let weights = Tensor::<B, 2>::from_data(
            TensorData::new(plan.interpolation_weights[corner].clone(), [patches, 1]),
            device,
        )
        .cast(embedding.dtype());
        let contribution = embedding * weights;
        output = Some(match output {
            Some(previous) => previous + contribution,
            None => contribution,
        });
    }
    output
        .expect("vision position interpolation has exactly four corners")
        .reshape([patches, hidden_size])
}

fn merge_order_coordinates(height: usize, width: usize, merge: usize) -> Vec<[usize; 2]> {
    let mut coordinates = Vec::with_capacity(height * width);
    for block_row in 0..height / merge {
        for block_column in 0..width / merge {
            for within_row in 0..merge {
                for within_column in 0..merge {
                    coordinates.push([
                        block_row * merge + within_row,
                        block_column * merge + within_column,
                    ]);
                }
            }
        }
    }
    coordinates
}

fn bilinear_entry(
    row: usize,
    column: usize,
    height: usize,
    width: usize,
    table_side: usize,
) -> ([i64; 4], [f32; 4]) {
    let scaled_row = if height > 1 {
        row as f64 * (table_side - 1) as f64 / (height - 1) as f64
    } else {
        0.0
    };
    let scaled_column = if width > 1 {
        column as f64 * (table_side - 1) as f64 / (width - 1) as f64
    } else {
        0.0
    };
    let row_low = scaled_row.floor() as usize;
    let row_high = scaled_row.ceil() as usize;
    let column_low = scaled_column.floor() as usize;
    let column_high = scaled_column.ceil() as usize;
    let row_fraction = (scaled_row - row_low as f64) as f32;
    let column_fraction = (scaled_column - column_low as f64) as f32;
    (
        [
            (row_low * table_side + column_low) as i64,
            (row_low * table_side + column_high) as i64,
            (row_high * table_side + column_low) as i64,
            (row_high * table_side + column_high) as i64,
        ],
        [
            (1.0 - row_fraction) * (1.0 - column_fraction),
            (1.0 - row_fraction) * column_fraction,
            row_fraction * (1.0 - column_fraction),
            row_fraction * column_fraction,
        ],
    )
}

#[derive(Module, Debug)]
pub struct Qwen3VlVisionPatchEmbed<B: Backend> {
    pub proj: Conv3d<B>,
    #[module(skip)]
    in_channels: usize,
    #[module(skip)]
    temporal_patch_size: usize,
    #[module(skip)]
    patch_size: usize,
    #[module(skip)]
    hidden_size: usize,
}

impl<B: Backend> Qwen3VlVisionPatchEmbed<B> {
    pub fn new(config: &Qwen3VlVisionConfig, device: &B::Device) -> Self {
        let kernel = [
            config.temporal_patch_size,
            config.patch_size,
            config.patch_size,
        ];
        Self {
            proj: Conv3dConfig::new([config.in_channels, config.hidden_size], kernel)
                .with_stride(kernel)
                .with_bias(true)
                .init(device),
            in_channels: config.in_channels,
            temporal_patch_size: config.temporal_patch_size,
            patch_size: config.patch_size,
            hidden_size: config.hidden_size,
        }
    }

    pub fn forward(&self, patches: Tensor<B, 2>) -> Tensor<B, 2> {
        let [patch_count, _] = patches.dims();
        // Processors intentionally produce F32 pixels. Match the loaded patch projection
        // precision here so F16 WebGPU checkpoints do not require callers to know model dtype.
        let patches = patches.cast(self.proj.weight.val().dtype());
        self.proj
            .forward(patches.reshape([
                patch_count,
                self.in_channels,
                self.temporal_patch_size,
                self.patch_size,
                self.patch_size,
            ]))
            .reshape([patch_count, self.hidden_size])
    }
}

#[derive(Module, Debug)]
pub struct Qwen3VlVisionMlp<B: Backend> {
    pub linear_fc1: QwenLinear<B>,
    pub linear_fc2: QwenLinear<B>,
    activation: Gelu,
}

impl<B: Backend> Qwen3VlVisionMlp<B> {
    pub fn new(config: &Qwen3VlVisionConfig, device: &B::Device) -> Self {
        Self {
            linear_fc1: QwenLinearConfig::new(config.hidden_size, config.intermediate_size)
                .init(device),
            linear_fc2: QwenLinearConfig::new(config.intermediate_size, config.hidden_size)
                .init(device),
            activation: Gelu::new_approximate(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        qwen_linear_forward(
            &self.linear_fc2,
            self.activation
                .forward(qwen_linear_forward(&self.linear_fc1, input)),
        )
    }
}

#[derive(Module, Debug)]
pub struct Qwen3VlVisionAttention<B: Backend> {
    pub qkv: QwenLinear<B>,
    pub proj: QwenLinear<B>,
    #[module(skip)]
    num_heads: usize,
    #[module(skip)]
    head_dim: usize,
    #[module(skip)]
    query_chunk_size: usize,
}

impl<B: Backend> Qwen3VlVisionAttention<B> {
    pub fn new(config: &Qwen3VlVisionConfig, device: &B::Device) -> Self {
        Self {
            qkv: QwenLinearConfig::new(config.hidden_size, 3 * config.hidden_size).init(device),
            proj: QwenLinearConfig::new(config.hidden_size, config.hidden_size).init(device),
            num_heads: config.num_heads,
            head_dim: config.head_dim(),
            query_chunk_size: DEFAULT_QUERY_CHUNK_SIZE,
        }
    }

    pub fn set_query_chunk_size(&mut self, query_chunk_size: usize) {
        self.query_chunk_size = query_chunk_size.max(1);
    }

    pub fn forward(
        &self,
        input: Tensor<B, 2>,
        frame_ranges: &[(usize, usize)],
        cos: Tensor<B, 2>,
        sin: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let [sequence, hidden] = input.dims();
        let qkv = qwen_linear_forward(&self.qkv, input);
        let query = qkv
            .clone()
            .slice([0..sequence, 0..hidden])
            .reshape([1, sequence, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let key = qkv
            .clone()
            .slice([0..sequence, hidden..2 * hidden])
            .reshape([1, sequence, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let value = qkv
            .slice([0..sequence, 2 * hidden..3 * hidden])
            .reshape([1, sequence, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let (query, key) = apply_vision_rope(query, key, cos, sin);
        let dtype = query.dtype();
        let mut frame_outputs = Vec::with_capacity(frame_ranges.len());
        for &(frame_start, frame_end) in frame_ranges {
            let frame_length = frame_end - frame_start;
            let frame_key = key
                .clone()
                .slice([
                    0..1,
                    0..self.num_heads,
                    frame_start..frame_end,
                    0..self.head_dim,
                ])
                .swap_dims(2, 3);
            let frame_value = value.clone().slice([
                0..1,
                0..self.num_heads,
                frame_start..frame_end,
                0..self.head_dim,
            ]);
            let mut query_outputs = Vec::new();
            let mut start = frame_start;
            while start < frame_end {
                let end = (start + self.query_chunk_size).min(frame_end);
                let query_chunk =
                    query
                        .clone()
                        .slice([0..1, 0..self.num_heads, start..end, 0..self.head_dim]);
                let scores = query_chunk
                    .cast(DType::F32)
                    .matmul(frame_key.clone().cast(DType::F32))
                    .mul_scalar(1.0 / (self.head_dim as f64).sqrt());
                query_outputs.push(
                    activation::softmax(scores, 3)
                        .cast(dtype)
                        .matmul(frame_value.clone()),
                );
                start = end;
            }
            frame_outputs.push(
                Tensor::cat(query_outputs, 2)
                    .swap_dims(1, 2)
                    .reshape([frame_length, hidden]),
            );
        }
        qwen_linear_forward(&self.proj, Tensor::cat(frame_outputs, 0))
    }
}

#[derive(Module, Debug)]
pub struct Qwen3VlVisionBlock<B: Backend> {
    pub norm1: LayerNorm<B>,
    pub norm2: LayerNorm<B>,
    pub attn: Qwen3VlVisionAttention<B>,
    pub mlp: Qwen3VlVisionMlp<B>,
}

impl<B: Backend> Qwen3VlVisionBlock<B> {
    pub fn new(config: &Qwen3VlVisionConfig, device: &B::Device) -> Self {
        Self {
            norm1: LayerNormConfig::new(config.hidden_size)
                .with_epsilon(config.layer_norm_eps)
                .init(device),
            norm2: LayerNormConfig::new(config.hidden_size)
                .with_epsilon(config.layer_norm_eps)
                .init(device),
            attn: Qwen3VlVisionAttention::new(config, device),
            mlp: Qwen3VlVisionMlp::new(config, device),
        }
    }

    pub fn forward(
        &self,
        hidden_states: Tensor<B, 2>,
        frame_ranges: &[(usize, usize)],
        cos: Tensor<B, 2>,
        sin: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let hidden_states = hidden_states.clone()
            + self
                .attn
                .forward(self.norm1.forward(hidden_states), frame_ranges, cos, sin);
        hidden_states.clone() + self.mlp.forward(self.norm2.forward(hidden_states))
    }
}

#[derive(Module, Debug)]
pub struct Qwen3VlVisionPatchMerger<B: Backend> {
    pub norm: LayerNorm<B>,
    pub linear_fc1: QwenLinear<B>,
    pub linear_fc2: QwenLinear<B>,
    activation: Gelu,
    #[module(skip)]
    merged_hidden_size: usize,
    #[module(skip)]
    postshuffle_norm: bool,
}

impl<B: Backend> Qwen3VlVisionPatchMerger<B> {
    pub fn new(config: &Qwen3VlVisionConfig, postshuffle_norm: bool, device: &B::Device) -> Self {
        let merged_hidden_size =
            config.hidden_size * config.spatial_merge_size * config.spatial_merge_size;
        Self {
            norm: LayerNormConfig::new(if postshuffle_norm {
                merged_hidden_size
            } else {
                config.hidden_size
            })
            .with_epsilon(config.layer_norm_eps)
            .init(device),
            linear_fc1: QwenLinearConfig::new(merged_hidden_size, merged_hidden_size).init(device),
            linear_fc2: QwenLinearConfig::new(merged_hidden_size, config.out_hidden_size)
                .init(device),
            activation: Gelu::new(),
            merged_hidden_size,
            postshuffle_norm,
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let [patches, hidden] = input.dims();
        let normalized = if self.postshuffle_norm {
            self.norm.forward(input.reshape([
                patches * hidden / self.merged_hidden_size,
                self.merged_hidden_size,
            ]))
        } else {
            self.norm.forward(input).reshape([
                patches * hidden / self.merged_hidden_size,
                self.merged_hidden_size,
            ])
        };
        qwen_linear_forward(
            &self.linear_fc2,
            self.activation
                .forward(qwen_linear_forward(&self.linear_fc1, normalized)),
        )
    }
}

/// Ordinary Qwen3-VL vision transformer.
#[derive(Module, Debug)]
pub struct Qwen3VlVisionModel<B: Backend> {
    pub patch_embed: Qwen3VlVisionPatchEmbed<B>,
    pub pos_embed: Embedding<B>,
    pub blocks: Vec<Qwen3VlVisionBlock<B>>,
    pub merger: Qwen3VlVisionPatchMerger<B>,
    pub deepstack_merger_list: Vec<Qwen3VlVisionPatchMerger<B>>,
    #[module(skip)]
    config: Qwen3VlVisionConfig,
}

impl<B: Backend> Qwen3VlVisionModel<B> {
    pub fn new(config: Qwen3VlVisionConfig, device: &B::Device) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            patch_embed: Qwen3VlVisionPatchEmbed::new(&config, device),
            pos_embed: EmbeddingConfig::new(config.num_position_embeddings, config.hidden_size)
                .init(device),
            blocks: (0..config.depth)
                .map(|_| Qwen3VlVisionBlock::new(&config, device))
                .collect(),
            merger: Qwen3VlVisionPatchMerger::new(&config, false, device),
            deepstack_merger_list: config
                .deepstack_visual_indexes
                .iter()
                .map(|_| Qwen3VlVisionPatchMerger::new(&config, true, device))
                .collect(),
            config,
        })
    }

    pub fn config(&self) -> &Qwen3VlVisionConfig {
        &self.config
    }

    /// Bound the number of query rows in each packed-frame attention score tile.
    pub fn set_query_chunk_size(&mut self, query_chunk_size: usize) {
        for block in &mut self.blocks {
            block.attn.set_query_chunk_size(query_chunk_size);
        }
    }

    pub fn forward(&self, patches: Tensor<B, 2>, grids: &[Grid]) -> Result<Qwen3VlVisionOutput<B>> {
        let [patch_count, patch_volume] = patches.dims();
        if patch_volume != self.config.patch_volume() {
            return Err(Qwen3VlError::InvalidInput(format!(
                "flattened patch width is {patch_volume}, expected {}",
                self.config.patch_volume()
            )));
        }
        let expected_patches = grids.iter().map(|grid| grid.patch_count()).sum::<usize>();
        if patch_count != expected_patches {
            return Err(Qwen3VlError::InvalidInput(format!(
                "received {patch_count} patches, but grids require {expected_patches}"
            )));
        }
        let plan = VisionPositionPlan::new(
            grids,
            self.config.spatial_merge_size,
            self.config.num_position_embeddings,
        )?;
        let mut hidden_states = self.patch_embed.forward(patches);
        let device = hidden_states.device();
        hidden_states = hidden_states + self.interpolate_position_embeddings(&plan, &device);
        let (cos, sin) =
            plan.vision_cos_sin::<B>(self.config.head_dim(), &hidden_states.device())?;
        let mut deepstack_features = Vec::with_capacity(self.deepstack_merger_list.len());
        for (block_index, block) in self.blocks.iter().enumerate() {
            hidden_states =
                block.forward(hidden_states, &plan.frame_ranges, cos.clone(), sin.clone());
            if let Some(merger_index) = self
                .config
                .deepstack_visual_indexes
                .iter()
                .position(|&index| index == block_index)
            {
                deepstack_features
                    .push(self.deepstack_merger_list[merger_index].forward(hidden_states.clone()));
            }
        }
        let pooler_output = self.merger.forward(hidden_states.clone());
        Ok(Qwen3VlVisionOutput {
            last_hidden_state: hidden_states,
            pooler_output,
            deepstack_features,
        })
    }

    pub fn interpolate_position_embeddings(
        &self,
        plan: &VisionPositionPlan,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        interpolate_learned_positions(&self.pos_embed, plan, self.config.hidden_size, device)
    }
}

fn rotate_half<B: Backend>(tensor: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, heads, sequence, head_dim] = tensor.dims();
    let half = head_dim / 2;
    Tensor::cat(
        vec![
            tensor
                .clone()
                .slice([0..batch, 0..heads, 0..sequence, half..head_dim])
                .neg(),
            tensor.slice([0..batch, 0..heads, 0..sequence, 0..half]),
        ],
        3,
    )
}

fn apply_vision_rope<B: Backend>(
    query: Tensor<B, 4>,
    key: Tensor<B, 4>,
    cos: Tensor<B, 2>,
    sin: Tensor<B, 2>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let query_dtype = query.dtype();
    let key_dtype = key.dtype();
    let [cos_sequence, cos_dim] = cos.dims();
    let [sin_sequence, sin_dim] = sin.dims();
    let cos = cos.reshape([1, 1, cos_sequence, cos_dim]).cast(DType::F32);
    let sin = sin.reshape([1, 1, sin_sequence, sin_dim]).cast(DType::F32);
    let query = query.cast(DType::F32);
    let key = key.cast(DType::F32);
    let rotated_query = rotate_half(query.clone());
    let rotated_key = rotate_half(key.clone());
    (
        (query * cos.clone() + rotated_query * sin.clone()).cast(query_dtype),
        (key * cos + rotated_key * sin).cast(key_dtype),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::tiny_config;
    use burn_ndarray::NdArray;

    #[test]
    fn merge_order_and_interpolation_are_exact_correctness() {
        let plan = VisionPositionPlan::new(&[Grid::new(1, 4, 4)], 2, 16).unwrap();
        assert_eq!(
            plan.rotary_coordinates,
            vec![
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
                [0, 2],
                [0, 3],
                [1, 2],
                [1, 3],
                [2, 0],
                [2, 1],
                [3, 0],
                [3, 1],
                [2, 2],
                [2, 3],
                [3, 2],
                [3, 3],
            ]
        );
        for patch in 0..plan.patch_count() {
            let sum = (0..4)
                .map(|corner| plan.interpolation_weights[corner][patch])
                .sum::<f32>();
            assert!((sum - 1.0).abs() < 1e-6);
        }
        assert_eq!(plan.frame_ranges, [(0, 16)]);
    }

    #[test]
    fn tiny_vision_forward_is_finite_smoke() {
        type B = NdArray<f32>;
        let config = tiny_config().vision_config;
        let device = Default::default();
        B::seed(&device, 11);
        let model = Qwen3VlVisionModel::<B>::new(config.clone(), &device).unwrap();
        let patches = Tensor::<B, 2>::from_data(
            TensorData::new(
                vec![0.25_f32; 4 * config.patch_volume()],
                [4, config.patch_volume()],
            ),
            &device,
        );
        let output = model.forward(patches, &[Grid::new(1, 2, 2)]).unwrap();
        assert_eq!(output.last_hidden_state.dims(), [4, 8]);
        assert_eq!(output.pooler_output.dims(), [1, 8]);
        assert_eq!(output.deepstack_features.len(), 1);
        assert!(
            output
                .pooler_output
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn packed_frames_do_not_cross_attend_correctness() {
        type B = NdArray<f32>;
        let config = tiny_config().vision_config;
        let device = Default::default();
        B::seed(&device, 23);
        let model = Qwen3VlVisionModel::<B>::new(config.clone(), &device).unwrap();
        let width = config.patch_volume();
        let mut first_patches = vec![0.2_f32; 8 * width];
        let mut second_patches = first_patches.clone();
        for value in &mut first_patches[4 * width..] {
            *value = -0.8;
        }
        for value in &mut second_patches[4 * width..] {
            *value = 1.7;
        }
        let run = |values| {
            model
                .forward(
                    Tensor::<B, 2>::from_data(TensorData::new(values, [8, width]), &device),
                    &[Grid::new(2, 2, 2)],
                )
                .unwrap()
                .last_hidden_state
                .slice([0..4, 0..8])
                .into_data()
                .to_vec::<f32>()
                .unwrap()
        };
        let first = run(first_patches);
        let second = run(second_patches);
        for (left, right) in first.iter().zip(second) {
            assert!((left - right).abs() < 1e-5, "{left} != {right}");
        }
    }
}
