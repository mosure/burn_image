//! Multimodal rotary position planning and tensor construction.

use burn::tensor::{Tensor, TensorData, backend::Backend};
use serde::{Deserialize, Serialize};

use crate::{Qwen3VlError, Result, config::Qwen3VlTextConfig, processor::Grid};

/// The difference between the largest assigned position plus one and the valid token count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PositionDelta(pub i64);

/// CPU representation of Qwen3-VL temporal, height, and width position ids.
///
/// Values are flattened as `[batch, sequence]` independently for each axis. Keeping the plan on
/// CPU makes it cheap to inspect and compare with a reference implementation before a single
/// compact upload to the backend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MropePositionIds {
    axes: [Vec<i64>; 3],
    batch_size: usize,
    sequence_length: usize,
    deltas: Vec<PositionDelta>,
}

impl MropePositionIds {
    pub fn new(
        axes: [Vec<i64>; 3],
        batch_size: usize,
        sequence_length: usize,
        deltas: Vec<PositionDelta>,
    ) -> Result<Self> {
        let expected = batch_size * sequence_length;
        if axes.iter().any(|axis| axis.len() != expected) {
            return Err(Qwen3VlError::InvalidInput(format!(
                "MRoPE axes must each contain {expected} entries"
            )));
        }
        if deltas.len() != batch_size {
            return Err(Qwen3VlError::InvalidInput(
                "MRoPE delta count must equal batch size".into(),
            ));
        }
        Ok(Self {
            axes,
            batch_size,
            sequence_length,
            deltas,
        })
    }

    /// Ordinary text-only positions, repeated on all three MRoPE axes.
    pub fn text_only(batch_size: usize, sequence_length: usize) -> Self {
        let mut axis = Vec::with_capacity(batch_size * sequence_length);
        for _ in 0..batch_size {
            axis.extend((0..sequence_length).map(|value| value as i64));
        }
        Self {
            axes: [axis.clone(), axis.clone(), axis],
            batch_size,
            sequence_length,
            deltas: vec![PositionDelta(0); batch_size],
        }
    }

    /// Plan positions from per-token modality ids (`0=text`, `1=image`, `2=video`).
    ///
    /// This follows Qwen3-VL's grouped-token algorithm. Videos are split into one temporal grid
    /// per frame because ordinary Qwen3-VL places timestamp text between frames.
    pub fn from_batch(
        token_types: &[Vec<u8>],
        attention_mask: &[Vec<bool>],
        image_grids: &[Vec<Grid>],
        video_grids: &[Vec<Grid>],
        spatial_merge_size: usize,
    ) -> Result<Self> {
        if token_types.is_empty()
            || token_types.len() != attention_mask.len()
            || token_types.len() != image_grids.len()
            || token_types.len() != video_grids.len()
        {
            return Err(Qwen3VlError::InvalidInput(
                "MRoPE batch fields must have the same non-zero batch size".into(),
            ));
        }
        let sequence_length = token_types[0].len();
        if sequence_length == 0
            || token_types.iter().zip(attention_mask).any(|(types, mask)| {
                types.len() != sequence_length || mask.len() != sequence_length
            })
        {
            return Err(Qwen3VlError::InvalidInput(
                "MRoPE samples must have a common non-zero padded sequence length".into(),
            ));
        }
        if spatial_merge_size == 0 {
            return Err(Qwen3VlError::InvalidInput(
                "spatial_merge_size must be non-zero".into(),
            ));
        }

        let batch_size = token_types.len();
        let mut axes = [
            vec![0_i64; batch_size * sequence_length],
            vec![0_i64; batch_size * sequence_length],
            vec![0_i64; batch_size * sequence_length],
        ];
        let mut deltas = Vec::with_capacity(batch_size);

        for batch in 0..batch_size {
            let valid_indices = attention_mask[batch]
                .iter()
                .enumerate()
                .filter_map(|(index, valid)| valid.then_some(index))
                .collect::<Vec<_>>();
            let valid_types = valid_indices
                .iter()
                .map(|&index| token_types[batch][index])
                .collect::<Vec<_>>();
            if valid_types.iter().any(|&kind| kind > 2) {
                return Err(Qwen3VlError::InvalidInput(
                    "token modality ids must be 0, 1, or 2".into(),
                ));
            }

            let images = &image_grids[batch];
            let expanded_videos = video_grids[batch]
                .iter()
                .flat_map(|grid| (0..grid.t).map(move |_| Grid::new(1, grid.h, grid.w)))
                .collect::<Vec<_>>();
            let mut image_index = 0;
            let mut video_index = 0;
            let mut local_axes = [Vec::new(), Vec::new(), Vec::new()];
            let mut current_position = 0_i64;
            let mut group_start = 0;

            while group_start < valid_types.len() {
                let modality = valid_types[group_start];
                let mut group_end = group_start + 1;
                while group_end < valid_types.len() && valid_types[group_end] == modality {
                    group_end += 1;
                }
                let group_len = group_end - group_start;
                if modality == 0 {
                    for offset in 0..group_len {
                        let position = current_position + offset as i64;
                        for axis in &mut local_axes {
                            axis.push(position);
                        }
                    }
                    current_position += group_len as i64;
                } else {
                    let grid = if modality == 1 {
                        let value = images.get(image_index).copied().ok_or_else(|| {
                            Qwen3VlError::InvalidInput(
                                "an image token group has no corresponding image grid".into(),
                            )
                        })?;
                        image_index += 1;
                        value
                    } else {
                        let value = expanded_videos.get(video_index).copied().ok_or_else(|| {
                            Qwen3VlError::InvalidInput(
                                "a video token group has no corresponding frame grid".into(),
                            )
                        })?;
                        video_index += 1;
                        value
                    };
                    grid.validate(spatial_merge_size)?;
                    let positions = vision_positions(current_position, grid, spatial_merge_size);
                    if positions[0].len() != group_len {
                        return Err(Qwen3VlError::InvalidInput(format!(
                            "modality group contains {group_len} tokens but grid {:?} requires {}",
                            grid,
                            positions[0].len()
                        )));
                    }
                    for axis in 0..3 {
                        local_axes[axis].extend_from_slice(&positions[axis]);
                    }
                    current_position += (grid.h.max(grid.w) / spatial_merge_size) as i64;
                }
                group_start = group_end;
            }

            if image_index != images.len() || video_index != expanded_videos.len() {
                return Err(Qwen3VlError::InvalidInput(
                    "unused image or video grids remain after MRoPE planning".into(),
                ));
            }
            for (valid_offset, &padded_index) in valid_indices.iter().enumerate() {
                for axis in 0..3 {
                    axes[axis][batch * sequence_length + padded_index] =
                        local_axes[axis][valid_offset];
                }
            }
            let max_position = local_axes
                .iter()
                .flat_map(|axis| axis.iter())
                .copied()
                .max()
                .unwrap_or(-1);
            deltas.push(PositionDelta(max_position + 1 - valid_indices.len() as i64));
        }

        Self::new(axes, batch_size, sequence_length, deltas)
    }

    pub fn axes(&self) -> &[Vec<i64>; 3] {
        &self.axes
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn sequence_length(&self) -> usize {
        self.sequence_length
    }

    pub fn deltas(&self) -> &[PositionDelta] {
        &self.deltas
    }

    /// Compute interleaved MRoPE cosine and sine tensors with shape `[batch, sequence, head_dim]`.
    pub fn cos_sin<B: Backend>(
        &self,
        config: &Qwen3VlTextConfig,
        device: &B::Device,
    ) -> Result<(Tensor<B, 3>, Tensor<B, 3>)> {
        if self.batch_size == 0 || self.sequence_length == 0 {
            return Err(Qwen3VlError::InvalidInput(
                "cannot build rotary tensors for an empty position plan".into(),
            ));
        }
        let head_dim = config.head_dim();
        let half_dim = head_dim / 2;
        let rope = config.mrope();
        if rope.mrope_section.iter().sum::<usize>() != half_dim {
            return Err(Qwen3VlError::InvalidConfig(
                "MRoPE sections do not cover half the attention head".into(),
            ));
        }
        let mut inverse_frequencies = Vec::with_capacity(half_dim);
        for index in 0..half_dim {
            inverse_frequencies.push(
                config
                    .rope_theta
                    .powf(-((2 * index) as f64) / head_dim as f64),
            );
        }

        let total = self.batch_size * self.sequence_length;
        let mut cos = vec![0_f32; total * head_dim];
        let mut sin = vec![0_f32; total * head_dim];
        for token in 0..total {
            for frequency in 0..half_dim {
                // Qwen3-VL starts with temporal frequencies then overwrites 1,4,7... with
                // height and 2,5,8... with width for the configured section lengths.
                let axis = if frequency % 3 == 1 && frequency < rope.mrope_section[1] * 3 {
                    1
                } else if frequency % 3 == 2 && frequency < rope.mrope_section[2] * 3 {
                    2
                } else {
                    0
                };
                let angle = self.axes[axis][token] as f64 * inverse_frequencies[frequency];
                let cosine = angle.cos() as f32;
                let sine = angle.sin() as f32;
                cos[token * head_dim + frequency] = cosine;
                cos[token * head_dim + half_dim + frequency] = cosine;
                sin[token * head_dim + frequency] = sine;
                sin[token * head_dim + half_dim + frequency] = sine;
            }
        }
        let shape = [self.batch_size, self.sequence_length, head_dim];
        Ok((
            Tensor::from_data(TensorData::new(cos, shape), device),
            Tensor::from_data(TensorData::new(sin, shape), device),
        ))
    }
}

fn vision_positions(start: i64, grid: Grid, merge: usize) -> [Vec<i64>; 3] {
    let grid_t = grid.t;
    let grid_h = grid.h / merge;
    let grid_w = grid.w / merge;
    let mut temporal = Vec::with_capacity(grid_t * grid_h * grid_w);
    let mut height = Vec::with_capacity(temporal.capacity());
    let mut width = Vec::with_capacity(temporal.capacity());
    for t in 0..grid_t {
        for h in 0..grid_h {
            for w in 0..grid_w {
                temporal.push(start + t as i64);
                height.push(start + h as i64);
                width.push(start + w as i64);
            }
        }
    }
    [temporal, height, width]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::tiny_config;
    use burn_ndarray::NdArray;

    #[test]
    fn image_positions_match_reference_correctness() {
        let positions = MropePositionIds::from_batch(
            &[vec![0, 0, 1, 1, 1, 1, 0]],
            &[vec![true; 7]],
            &[vec![Grid::new(1, 4, 4)]],
            &[vec![]],
            2,
        )
        .unwrap();
        assert_eq!(positions.axes()[0], [0, 1, 2, 2, 2, 2, 4]);
        assert_eq!(positions.axes()[1], [0, 1, 2, 2, 3, 3, 4]);
        assert_eq!(positions.axes()[2], [0, 1, 2, 3, 2, 3, 4]);
        assert_eq!(positions.deltas(), [PositionDelta(-2)]);
    }

    #[test]
    fn mrope_axis_interleaving_correctness() {
        type B = NdArray<f32>;
        let mut config = tiny_config().text_config;
        config.head_dim = Some(6);
        config.hidden_size = 12;
        config.num_attention_heads = 2;
        config.rope_scaling.as_mut().unwrap().mrope_section = [1, 1, 1];
        let positions =
            MropePositionIds::new([vec![1], vec![2], vec![3]], 1, 1, vec![PositionDelta(0)])
                .unwrap();
        let (cos, _) = positions
            .cos_sin::<B>(&config, &Default::default())
            .unwrap();
        let values = cos.into_data().to_vec::<f32>().unwrap();
        let inv1 = config.rope_theta.powf(-2.0 / 6.0);
        let inv2 = config.rope_theta.powf(-4.0 / 6.0);
        assert!((values[0] - 1_f32.cos()).abs() < 1e-6);
        assert!((values[1] - (2.0 * inv1).cos() as f32).abs() < 1e-6);
        assert!((values[2] - (3.0 * inv2).cos() as f32).abs() < 1e-6);
        assert_eq!(&values[..3], &values[3..]);
    }
}
