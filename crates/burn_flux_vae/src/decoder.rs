use burn::{
    nn,
    prelude::{Backend, Module},
    tensor::{FloatDType, Tensor},
};

use crate::{
    blocks::{
        DecoderGroupNormPolicy, MidBlock2d, ResnetBlock2d, Upsample2d, conv3, group_norm,
        group_norm_with_policy, silu,
    },
    config::AutoencoderKlConfig,
};

/// Ordinary Diffusers `UpDecoderBlock2D`.
#[derive(Module, Debug)]
pub struct UpDecoderBlock2d<B: Backend> {
    pub resnets: Vec<ResnetBlock2d<B>>,
    pub upsamplers: Vec<Upsample2d<B>>,
}

impl<B: Backend> UpDecoderBlock2d<B> {
    pub fn new(
        device: &B::Device,
        in_channels: usize,
        out_channels: usize,
        layers: usize,
        groups: usize,
        epsilon: f64,
        add_upsample: bool,
    ) -> Self {
        let mut resnets = Vec::with_capacity(layers);
        for index in 0..layers {
            resnets.push(ResnetBlock2d::new(
                device,
                if index == 0 {
                    in_channels
                } else {
                    out_channels
                },
                out_channels,
                groups,
                epsilon,
            ));
        }
        Self {
            resnets,
            upsamplers: add_upsample
                .then(|| Upsample2d::new(device, out_channels))
                .into_iter()
                .collect(),
        }
    }

    pub fn forward(&self, hidden: Tensor<B, 4>) -> Tensor<B, 4> {
        self.forward_with_group_norm_policy(hidden, DecoderGroupNormPolicy::StrictF32)
    }

    fn forward_with_group_norm_policy(
        &self,
        mut hidden: Tensor<B, 4>,
        policy: DecoderGroupNormPolicy,
    ) -> Tensor<B, 4> {
        for resnet in &self.resnets {
            hidden = resnet.forward_with_group_norm_policy(hidden, policy);
        }
        if let Some(upsample) = self.upsamplers.first() {
            hidden = upsample.forward(hidden);
        }
        hidden
    }
}

/// Diffusers-compatible AutoencoderKL decoder.
#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    pub conv_in: nn::conv::Conv2d<B>,
    pub mid_block: MidBlock2d<B>,
    pub up_blocks: Vec<UpDecoderBlock2d<B>>,
    pub conv_norm_out: nn::GroupNorm<B>,
    pub conv_out: nn::conv::Conv2d<B>,
}

/// Live state for an exact strict-F32 striped decode split at synchronization-safe stages.
///
/// The state owns only the current activation tensor. Decoder parameters remain borrowed from the
/// resident [`Decoder`], so synchronizing and cleaning the backend allocator between advances can
/// never evict model weights.
pub struct StripedTailDecodeState<B: Backend> {
    hidden: Option<Tensor<B, 4>>,
    left: Option<Tensor<B, 4>>,
    right: Option<Tensor<B, 4>>,
    residual_left: Option<Tensor<B, 4>>,
    residual_right: Option<Tensor<B, 4>>,
    right_halo: Option<Tensor<B, 4>>,
    norm_stats: Option<TwoWidthGroupNormStats<B>>,
    split_width: usize,
    next_stage: usize,
    complete: bool,
}

struct TwoWidthGroupNormStats<B: Backend> {
    mean: Tensor<B, 3>,
    inverse_std: Tensor<B, 3>,
    accumulation_dtype: FloatDType,
    output_dtype: FloatDType,
}

const STRIPED_RESNET_STAGE_COUNT: usize = 15;
const STRIPED_OUTPUT_STAGE_COUNT: usize = 7;

impl<B: Backend> StripedTailDecodeState<B> {
    /// Whether the exact output tensor is ready to consume.
    pub const fn is_complete(&self) -> bool {
        self.complete
    }

    /// Consume a completed staged decode and return its exact output tensor.
    pub fn into_output(mut self) -> Tensor<B, 4> {
        assert!(self.complete, "striped decoder output is not complete");
        self.hidden
            .take()
            .expect("striped decoder complete state retains its output")
    }
}

impl<B: Backend> Decoder<B> {
    pub fn new(device: &B::Device, config: &AutoencoderKlConfig) -> Self {
        let first_channels = config.block_out_channels[0];
        let last_channels = *config
            .block_out_channels
            .last()
            .expect("validated VAE block channels");
        let reversed = config
            .block_out_channels
            .iter()
            .copied()
            .rev()
            .collect::<Vec<_>>();
        let mut previous_channels = reversed[0];
        let mut up_blocks = Vec::with_capacity(reversed.len());
        for (index, output_channels) in reversed.iter().copied().enumerate() {
            up_blocks.push(UpDecoderBlock2d::new(
                device,
                previous_channels,
                output_channels,
                config.layers_per_block + 1,
                config.norm_num_groups,
                config.norm_epsilon,
                index + 1 != reversed.len(),
            ));
            previous_channels = output_channels;
        }
        Self {
            conv_in: conv3(device, config.latent_channels, last_channels),
            mid_block: MidBlock2d::new(
                device,
                last_channels,
                config.norm_num_groups,
                config.norm_epsilon,
                config.mid_block_add_attention,
                config.attention_query_chunk_size,
            ),
            up_blocks,
            conv_norm_out: group_norm(
                device,
                config.norm_num_groups,
                first_channels,
                config.norm_epsilon,
            ),
            conv_out: conv3(device, first_channels, config.out_channels),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        self.forward_with_group_norm_policy(input, DecoderGroupNormPolicy::StrictF32)
    }

    /// Update exact attention partitioning without reallocating model parameters.
    pub fn set_attention_query_chunk_size(&mut self, query_chunk_size: usize) {
        self.mid_block
            .set_attention_query_chunk_size(query_chunk_size);
    }

    /// Decode with an explicit mixed-precision GroupNorm execution policy.
    ///
    /// The ordinary [`Self::forward`] API remains strict F32 for F16/BF16 activations.
    pub fn forward_with_group_norm_policy(
        &self,
        input: Tensor<B, 4>,
        policy: DecoderGroupNormPolicy,
    ) -> Tensor<B, 4> {
        let mut hidden = self.conv_in.forward(input);
        hidden = self
            .mid_block
            .forward_with_group_norm_policy(hidden, policy);
        for block in &self.up_blocks {
            hidden = block.forward_with_group_norm_policy(hidden, policy);
        }
        self.conv_out.forward(silu(group_norm_with_policy(
            &self.conv_norm_out,
            hidden,
            policy,
        )))
    }

    /// Decode with one fallible barrier before the final full-resolution residual block.
    ///
    /// This preserves every decoder operation and tensor value. The callback runs after the
    /// penultimate up block has produced the final-resolution feature tensor, when all earlier
    /// decoder intermediates are dead but before the largest final residual block allocates its
    /// workspaces. Memory-bounded runtimes can use this boundary to synchronize deferred drops and
    /// release unused backend allocator pages. Ordinary execution remains
    /// [`Self::forward_with_group_norm_policy`] and pays no barrier cost.
    pub fn forward_with_group_norm_policy_and_tail_barrier<E>(
        &self,
        input: Tensor<B, 4>,
        policy: DecoderGroupNormPolicy,
        tail_barrier: impl FnOnce(&B::Device) -> Result<(), E>,
    ) -> Result<Tensor<B, 4>, E> {
        let device = input.device();
        let barrier_after_block = self
            .up_blocks
            .len()
            .checked_sub(2)
            .expect("VAE decoder tail barrier requires at least two up blocks");
        let mut barrier = Some(tail_barrier);
        let mut hidden = self.conv_in.forward(input);
        hidden = self
            .mid_block
            .forward_with_group_norm_policy(hidden, policy);
        for (index, block) in self.up_blocks.iter().enumerate() {
            hidden = block.forward_with_group_norm_policy(hidden, policy);
            if index == barrier_after_block {
                barrier.take().expect("tail barrier runs exactly once")(&device)?;
            }
        }
        Ok(self.conv_out.forward(silu(group_norm_with_policy(
            &self.conv_norm_out,
            hidden,
            policy,
        ))))
    }

    /// Decode with the final full-resolution feature map split into two exact spatial slabs.
    ///
    /// This preserves the ordinary decoder through its global middle attention and every
    /// lower-resolution block. Only the last 2x upsample and the first residual block after it are
    /// split. Convolutions receive the neighboring one-pixel halo, while GroupNorm statistics are
    /// reduced across both slabs before either slab is normalized. The slabs are concatenated only
    /// after the channel-reducing residual block, bounding the largest individual feature buffer.
    ///
    /// `split_width` is expressed in output pixels and must be even so the split is aligned with
    /// nearest-neighbor upsampling. This path deliberately uses strict-F32 GroupNorm semantics and
    /// is intended for exact high-resolution browser qualification, not ordinary decoder calls.
    pub fn forward_striped_tail_strict_f32(
        &self,
        input: Tensor<B, 4>,
        split_width: usize,
    ) -> Tensor<B, 4> {
        let mut state = self.begin_striped_tail_strict_f32(input, split_width);
        while !state.is_complete() {
            self.advance_striped_tail_strict_f32(&mut state);
        }
        state.into_output()
    }

    /// Begin an exact striped decode with only the initial convolution and middle block applied.
    pub fn begin_striped_tail_strict_f32(
        &self,
        input: Tensor<B, 4>,
        split_width: usize,
    ) -> StripedTailDecodeState<B> {
        self.validate_striped_tail_structure();
        let hidden = self.mid_block.forward_with_group_norm_policy(
            self.conv_in.forward(input),
            DecoderGroupNormPolicy::StrictF32,
        );
        StripedTailDecodeState {
            hidden: Some(hidden),
            left: None,
            right: None,
            residual_left: None,
            residual_right: None,
            right_halo: None,
            norm_stats: None,
            split_width,
            next_stage: 0,
            complete: false,
        }
    }

    /// Number of synchronization-safe advances after [`Self::begin_striped_tail_strict_f32`].
    pub fn striped_tail_stage_count(&self) -> usize {
        let final_block_index = self
            .up_blocks
            .len()
            .checked_sub(1)
            .expect("striped decoder requires an up block");
        let upsample_block_index = final_block_index
            .checked_sub(1)
            .expect("striped decoder requires at least two up blocks");
        let prefix_stages = self.up_blocks[..upsample_block_index]
            .iter()
            .try_fold(0_usize, |total, block| {
                total.checked_add(block.resnets.len() + block.upsamplers.len())
            })
            .expect("striped decoder prefix stage count overflowed");
        let transition_stages = self.up_blocks[upsample_block_index]
            .resnets
            .len()
            .checked_add(1)
            .expect("striped decoder transition stage count overflowed");
        let resnet_stages = self.up_blocks[final_block_index]
            .resnets
            .len()
            .checked_mul(STRIPED_RESNET_STAGE_COUNT)
            .expect("striped decoder residual stage count overflowed");
        prefix_stages
            .checked_add(transition_stages)
            .and_then(|value| value.checked_add(resnet_stages))
            .and_then(|value| value.checked_add(STRIPED_OUTPUT_STAGE_COUNT))
            .expect("striped decoder stage count overflowed")
    }

    /// Advance one exact decoder stage.
    ///
    /// The previous activation is dropped before this method returns. A browser caller may then
    /// await its queue and clean unused allocator pages before advancing again.
    pub fn advance_striped_tail_strict_f32(&self, state: &mut StripedTailDecodeState<B>) {
        assert!(!state.complete, "striped decoder is already complete");
        self.validate_striped_tail_structure();
        let final_block_index = self.up_blocks.len() - 1;
        let upsample_block_index = final_block_index - 1;
        let prefix_stage_count = self.up_blocks[..upsample_block_index]
            .iter()
            .map(|block| block.resnets.len() + block.upsamplers.len())
            .sum::<usize>();
        let transition_resnets_start = prefix_stage_count;
        let split_upsample_stage =
            transition_resnets_start + self.up_blocks[upsample_block_index].resnets.len();
        let resnet_start_stage = split_upsample_stage + 1;
        let final_resnets = &self.up_blocks[final_block_index].resnets;
        let output_start_stage =
            resnet_start_stage + final_resnets.len() * STRIPED_RESNET_STAGE_COUNT;

        if state.next_stage < prefix_stage_count {
            let mut stage_start = 0_usize;
            let mut advanced = false;
            for block in &self.up_blocks[..upsample_block_index] {
                let resnet_end = stage_start + block.resnets.len();
                if state.next_stage < resnet_end {
                    let hidden = state
                        .hidden
                        .take()
                        .expect("striped decoder prefix retains its activation");
                    state.hidden = Some(
                        block.resnets[state.next_stage - stage_start]
                            .forward_with_group_norm_policy(
                                hidden,
                                DecoderGroupNormPolicy::StrictF32,
                            ),
                    );
                    advanced = true;
                    break;
                }
                if state.next_stage == resnet_end {
                    let hidden = state
                        .hidden
                        .take()
                        .expect("striped decoder prefix upsample retains its activation");
                    state.hidden = Some(
                        block
                            .upsamplers
                            .first()
                            .expect("validated prefix upsampler")
                            .forward(hidden),
                    );
                    advanced = true;
                    break;
                }
                stage_start = resnet_end + block.upsamplers.len();
            }
            assert!(advanced, "striped decoder prefix stage was not resolved");
        } else if state.next_stage < split_upsample_stage {
            let upsample_block = &self.up_blocks[upsample_block_index];
            let hidden = state
                .hidden
                .take()
                .expect("striped decoder transition retains its activation");
            state.hidden = Some(
                upsample_block.resnets[state.next_stage - transition_resnets_start]
                    .forward_with_group_norm_policy(hidden, DecoderGroupNormPolicy::StrictF32),
            );
        } else if state.next_stage == split_upsample_stage {
            let upsample_block = &self.up_blocks[upsample_block_index];
            let hidden = state
                .hidden
                .take()
                .expect("striped decoder split upsample retains its activation");
            let (left, right) = upsample_two_width_slabs(
                upsample_block
                    .upsamplers
                    .first()
                    .expect("validated final upsampler"),
                hidden,
                state.split_width,
            );
            state.left = Some(left);
            state.right = Some(right);
        } else if state.next_stage < output_start_stage {
            let relative_stage = state.next_stage - resnet_start_stage;
            let resnet_index = relative_stage / STRIPED_RESNET_STAGE_COUNT;
            let resnet_stage = relative_stage % STRIPED_RESNET_STAGE_COUNT;
            let resnet = &final_resnets[resnet_index];
            match resnet_stage {
                0 => {
                    let left = state.left.as_ref().expect("striped residual left input");
                    let right = state.right.as_ref().expect("striped residual right input");
                    assert!(
                        state.residual_left.is_none() && state.residual_right.is_none(),
                        "striped residual shortcut state leaked across blocks"
                    );
                    state.residual_left = Some(
                        resnet
                            .conv_shortcut
                            .as_ref()
                            .map(|shortcut| shortcut.forward(left.clone()))
                            .unwrap_or_else(|| left.clone()),
                    );
                    state.residual_right = Some(
                        resnet
                            .conv_shortcut
                            .as_ref()
                            .map(|shortcut| shortcut.forward(right.clone()))
                            .unwrap_or_else(|| right.clone()),
                    );
                }
                1 => {
                    state.norm_stats = Some(group_norm_two_width_slabs_stats_strict_f32(
                        &resnet.norm1,
                        state.left.as_ref().expect("striped norm1 left input"),
                        state.right.as_ref().expect("striped norm1 right input"),
                    ));
                }
                2 => {
                    state.left = Some(apply_group_norm_width_slab_strict_f32(
                        &resnet.norm1,
                        state.left.take().expect("striped norm1 left input"),
                        state.norm_stats.as_ref().expect("striped norm1 statistics"),
                    ));
                }
                3 => {
                    state.right = Some(apply_group_norm_width_slab_strict_f32(
                        &resnet.norm1,
                        state.right.take().expect("striped norm1 right input"),
                        state.norm_stats.as_ref().expect("striped norm1 statistics"),
                    ));
                    state.norm_stats.take();
                }
                4 => {
                    state.left = Some(silu(
                        state.left.take().expect("striped conv1 left activation"),
                    ));
                }
                5 => {
                    state.right = Some(silu(
                        state.right.take().expect("striped conv1 right activation"),
                    ));
                }
                6 => {
                    let (left, right_halo) = conv3_left_width_slab(
                        &resnet.conv1,
                        state.left.take().expect("striped conv1 left input"),
                        state.right.as_ref().expect("striped conv1 right input"),
                    );
                    state.left = Some(left);
                    state.right_halo = Some(right_halo);
                }
                7 => {
                    state.right = Some(conv3_right_width_slab(
                        &resnet.conv1,
                        state.right.take().expect("striped conv1 right input"),
                        state.right_halo.take().expect("striped conv1 right halo"),
                    ));
                }
                8 => {
                    state.norm_stats = Some(group_norm_two_width_slabs_stats_strict_f32(
                        &resnet.norm2,
                        state.left.as_ref().expect("striped norm2 left input"),
                        state.right.as_ref().expect("striped norm2 right input"),
                    ));
                }
                9 => {
                    state.left = Some(apply_group_norm_width_slab_strict_f32(
                        &resnet.norm2,
                        state.left.take().expect("striped norm2 left input"),
                        state.norm_stats.as_ref().expect("striped norm2 statistics"),
                    ));
                }
                10 => {
                    state.right = Some(apply_group_norm_width_slab_strict_f32(
                        &resnet.norm2,
                        state.right.take().expect("striped norm2 right input"),
                        state.norm_stats.as_ref().expect("striped norm2 statistics"),
                    ));
                    state.norm_stats.take();
                }
                11 => {
                    state.left = Some(silu(
                        state.left.take().expect("striped conv2 left activation"),
                    ));
                }
                12 => {
                    state.right = Some(silu(
                        state.right.take().expect("striped conv2 right activation"),
                    ));
                }
                13 => {
                    let (left, right_halo) = conv3_left_width_slab(
                        &resnet.conv2,
                        state.left.take().expect("striped conv2 left input"),
                        state.right.as_ref().expect("striped conv2 right input"),
                    );
                    state.left = Some(
                        state
                            .residual_left
                            .take()
                            .expect("striped residual left shortcut")
                            + left,
                    );
                    state.right_halo = Some(right_halo);
                }
                14 => {
                    let right = conv3_right_width_slab(
                        &resnet.conv2,
                        state.right.take().expect("striped conv2 right input"),
                        state.right_halo.take().expect("striped conv2 right halo"),
                    );
                    state.right = Some(
                        state
                            .residual_right
                            .take()
                            .expect("striped residual right shortcut")
                            + right,
                    );
                }
                _ => unreachable!("striped residual stage is bounded"),
            }
        } else {
            match state.next_stage - output_start_stage {
                0 => {
                    state.norm_stats = Some(group_norm_two_width_slabs_stats_strict_f32(
                        &self.conv_norm_out,
                        state.left.as_ref().expect("striped output norm left input"),
                        state
                            .right
                            .as_ref()
                            .expect("striped output norm right input"),
                    ));
                }
                1 => {
                    state.left = Some(apply_group_norm_width_slab_strict_f32(
                        &self.conv_norm_out,
                        state.left.take().expect("striped output norm left input"),
                        state
                            .norm_stats
                            .as_ref()
                            .expect("striped output norm statistics"),
                    ));
                }
                2 => {
                    state.right = Some(apply_group_norm_width_slab_strict_f32(
                        &self.conv_norm_out,
                        state.right.take().expect("striped output norm right input"),
                        state
                            .norm_stats
                            .as_ref()
                            .expect("striped output norm statistics"),
                    ));
                    state.norm_stats.take();
                }
                3 => {
                    state.left = Some(silu(
                        state.left.take().expect("striped output left activation"),
                    ));
                }
                4 => {
                    state.right = Some(silu(
                        state.right.take().expect("striped output right activation"),
                    ));
                }
                5 => {
                    let (left, right_halo) = conv3_left_width_slab(
                        &self.conv_out,
                        state.left.take().expect("striped output left input"),
                        state.right.as_ref().expect("striped output right input"),
                    );
                    state.left = Some(left);
                    state.right_halo = Some(right_halo);
                }
                6 => {
                    let right = conv3_right_width_slab(
                        &self.conv_out,
                        state.right.take().expect("striped output right input"),
                        state.right_halo.take().expect("striped output right halo"),
                    );
                    let left = state.left.take().expect("striped output left result");
                    state.hidden = Some(Tensor::cat(vec![left, right], 3));
                    state.complete = true;
                }
                _ => panic!("striped decoder stage index exceeds its exact plan"),
            }
        }
        state.next_stage += 1;
        debug_assert_eq!(
            state.complete,
            state.next_stage == self.striped_tail_stage_count()
        );
    }

    fn validate_striped_tail_structure(&self) {
        let final_block_index = self
            .up_blocks
            .len()
            .checked_sub(1)
            .expect("striped decoder requires an up block");
        let upsample_block_index = final_block_index
            .checked_sub(1)
            .expect("striped decoder requires at least two up blocks");
        assert!(
            self.up_blocks[..upsample_block_index]
                .iter()
                .all(|block| block.upsamplers.len() == 1),
            "striped decoder expects one upsampler in every preceding up block"
        );
        assert_eq!(
            self.up_blocks[upsample_block_index].upsamplers.len(),
            1,
            "striped decoder expects one final upsampler"
        );
        assert!(
            self.up_blocks[final_block_index].upsamplers.is_empty(),
            "striped decoder expects no upsampler in the final up block"
        );
        assert!(
            !self.up_blocks[final_block_index].resnets.is_empty(),
            "striped decoder final block requires a residual layer"
        );
    }
}

fn upsample_two_width_slabs<B: Backend>(
    upsample: &Upsample2d<B>,
    input: Tensor<B, 4>,
    split_width: usize,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [batch, channels, height, width] = input.dims();
    let output_width = width * 2;
    assert!(
        split_width >= 2 && split_width + 2 <= output_width && split_width.is_multiple_of(2),
        "striped decoder split must be an interior even output coordinate"
    );
    let low_split = split_width / 2;
    let output_height = height * 2;

    // The left slice contains the first low-resolution sample needed by the right-hand side of
    // its final 3x3 output. The right slice starts one low-resolution sample before the split; its
    // first two nearest-neighbor outputs are halo and are cropped after convolution.
    let left =
        upsample.forward(
            input
                .clone()
                .slice([0..batch, 0..channels, 0..height, 0..low_split + 1]),
        );
    let right =
        upsample.forward(input.slice([0..batch, 0..channels, 0..height, low_split - 1..width]));
    let right_width = output_width - split_width;
    (
        left.slice([0..batch, 0..channels, 0..output_height, 0..split_width]),
        right.slice([0..batch, 0..channels, 0..output_height, 2..right_width + 2]),
    )
}

#[cfg(test)]
fn resnet_two_width_slabs_strict_f32<B: Backend>(
    resnet: &ResnetBlock2d<B>,
    left: Tensor<B, 4>,
    right: Tensor<B, 4>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let residual_left = resnet
        .conv_shortcut
        .as_ref()
        .map(|shortcut| shortcut.forward(left.clone()))
        .unwrap_or_else(|| left.clone());
    let residual_right = resnet
        .conv_shortcut
        .as_ref()
        .map(|shortcut| shortcut.forward(right.clone()))
        .unwrap_or_else(|| right.clone());

    let (left, right) = group_norm_two_width_slabs_strict_f32(&resnet.norm1, left, right);
    let (left, right) = conv3_two_width_slabs(&resnet.conv1, silu(left), silu(right));
    let (left, right) = group_norm_two_width_slabs_strict_f32(&resnet.norm2, left, right);
    let (left, right) = conv3_two_width_slabs(&resnet.conv2, silu(left), silu(right));
    (residual_left + left, residual_right + right)
}

#[cfg(test)]
fn conv3_two_width_slabs<B: Backend>(
    conv: &nn::conv::Conv2d<B>,
    left: Tensor<B, 4>,
    right: Tensor<B, 4>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let (left, right_halo) = conv3_left_width_slab(conv, left, &right);
    let right = conv3_right_width_slab(conv, right, right_halo);
    (left, right)
}

fn conv3_left_width_slab<B: Backend>(
    conv: &nn::conv::Conv2d<B>,
    left: Tensor<B, 4>,
    right: &Tensor<B, 4>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [batch, channels, height, left_width] = left.dims();
    let [right_batch, right_channels, right_height, right_width] = right.dims();
    assert_eq!(
        [batch, channels, height],
        [right_batch, right_channels, right_height]
    );
    assert!(left_width > 0 && right_width > 0, "empty convolution slab");
    validate_striped_conv3(conv);

    let left_halo = right
        .clone()
        .slice([0..batch, 0..channels, 0..height, 0..1]);
    let right_halo =
        left.clone()
            .slice([0..batch, 0..channels, 0..height, left_width - 1..left_width]);
    let left = conv.forward(Tensor::cat(vec![left, left_halo], 3)).slice([
        0..batch,
        0..conv.weight.dims()[0],
        0..height,
        0..left_width,
    ]);
    (left, right_halo)
}

fn conv3_right_width_slab<B: Backend>(
    conv: &nn::conv::Conv2d<B>,
    right: Tensor<B, 4>,
    right_halo: Tensor<B, 4>,
) -> Tensor<B, 4> {
    let [batch, channels, height, right_width] = right.dims();
    assert_eq!(
        right_halo.dims(),
        [batch, channels, height, 1],
        "striped convolution right halo mismatch"
    );
    assert!(right_width > 0, "empty convolution slab");
    validate_striped_conv3(conv);
    conv.forward(Tensor::cat(vec![right_halo, right], 3))
        .slice([
            0..batch,
            0..conv.weight.dims()[0],
            0..height,
            1..right_width + 1,
        ])
}

fn validate_striped_conv3<B: Backend>(conv: &nn::conv::Conv2d<B>) {
    assert_eq!(
        conv.kernel_size,
        [3, 3],
        "striped decoder requires 3x3 convolution"
    );
    assert_eq!(
        conv.stride,
        [1, 1],
        "striped decoder requires unit convolution stride"
    );
    assert_eq!(
        conv.dilation,
        [1, 1],
        "striped decoder requires unit convolution dilation"
    );
    assert_eq!(conv.padding, nn::PaddingConfig2d::Explicit(1, 1, 1, 1));
}

#[cfg(test)]
fn group_norm_two_width_slabs_strict_f32<B: Backend>(
    norm: &nn::GroupNorm<B>,
    left: Tensor<B, 4>,
    right: Tensor<B, 4>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let stats = group_norm_two_width_slabs_stats_strict_f32(norm, &left, &right);
    (
        apply_group_norm_width_slab_strict_f32(norm, left, &stats),
        apply_group_norm_width_slab_strict_f32(norm, right, &stats),
    )
}

fn group_norm_two_width_slabs_stats_strict_f32<B: Backend>(
    norm: &nn::GroupNorm<B>,
    left: &Tensor<B, 4>,
    right: &Tensor<B, 4>,
) -> TwoWidthGroupNormStats<B> {
    let [batch, channels, height, left_width] = left.dims();
    let [right_batch, right_channels, right_height, right_width] = right.dims();
    assert_eq!(
        [batch, channels, height],
        [right_batch, right_channels, right_height]
    );
    assert_eq!(channels, norm.num_channels, "GroupNorm channel mismatch");
    assert!(left_width > 0 && right_width > 0, "empty GroupNorm slab");

    let output_dtype: FloatDType = left.dtype().into();
    assert_eq!(
        output_dtype,
        right.dtype().into(),
        "GroupNorm slab dtype mismatch"
    );
    let accumulation_dtype = if matches!(output_dtype, FloatDType::F16 | FloatDType::BF16) {
        FloatDType::F32
    } else {
        output_dtype
    };
    let group_channels = channels / norm.num_groups;
    let left_group_width = group_channels * height * left_width;
    let right_group_width = group_channels * height * right_width;
    let left =
        left.clone()
            .cast(accumulation_dtype)
            .reshape([batch, norm.num_groups, left_group_width]);
    let right =
        right
            .clone()
            .cast(accumulation_dtype)
            .reshape([batch, norm.num_groups, right_group_width]);
    let mean = (left.clone().sum_dim(2) + right.clone().sum_dim(2))
        / (left_group_width + right_group_width) as f64;
    let left_variance = (left - mean.clone()).square().sum_dim(2);
    let right_variance = (right - mean.clone()).square().sum_dim(2);
    let variance = (left_variance + right_variance) / (left_group_width + right_group_width) as f64;
    let inverse_std = variance.add_scalar(norm.epsilon).sqrt().recip();
    TwoWidthGroupNormStats {
        mean,
        inverse_std,
        accumulation_dtype,
        output_dtype,
    }
}

fn apply_group_norm_width_slab_strict_f32<B: Backend>(
    norm: &nn::GroupNorm<B>,
    input: Tensor<B, 4>,
    stats: &TwoWidthGroupNormStats<B>,
) -> Tensor<B, 4> {
    let [batch, channels, height, width] = input.dims();
    assert_eq!(channels, norm.num_channels, "GroupNorm channel mismatch");
    assert!(width > 0, "empty GroupNorm slab");
    let group_width = channels / norm.num_groups * height * width;
    let input =
        ((input
            .cast(stats.accumulation_dtype)
            .reshape([batch, norm.num_groups, group_width])
            - stats.mean.clone())
            * stats.inverse_std.clone())
        .reshape([batch, channels, height, width]);
    if !norm.affine {
        return input.cast(stats.output_dtype);
    }
    let gamma = norm
        .gamma
        .as_ref()
        .expect("affine GroupNorm gamma")
        .val()
        .cast(stats.accumulation_dtype)
        .reshape([1, channels, 1, 1]);
    let beta = norm
        .beta
        .as_ref()
        .expect("affine GroupNorm beta")
        .val()
        .cast(stats.accumulation_dtype)
        .reshape([1, channels, 1, 1]);
    (input * gamma + beta).cast(stats.output_dtype)
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Distribution, TensorData};

    use super::*;

    type TestBackend = burn::backend::NdArray<f32>;

    #[test]
    fn striped_group_norm_uses_global_statistics_correctness() {
        let device = Default::default();
        let norm = group_norm::<TestBackend>(&device, 2, 4, 1.0e-6);
        let left = Tensor::ones([1, 4, 2, 3], &device).mul_scalar(-8.0);
        let right = Tensor::ones([1, 4, 2, 2], &device).mul_scalar(5.0);
        let full = norm.forward(Tensor::cat(vec![left.clone(), right.clone()], 3));
        let (left, right) = group_norm_two_width_slabs_strict_f32(&norm, left, right);
        let striped = Tensor::cat(vec![left, right], 3);
        let max_abs = (full - striped).abs().max().into_scalar();
        assert!(max_abs <= 1.0e-6, "striped GroupNorm max_abs={max_abs}");
    }

    #[test]
    fn striped_upsample_conv_preserves_seam_impulse_correctness() {
        let device = Default::default();
        let upsample = Upsample2d::<TestBackend>::new(&device, 4);
        let mut values = vec![0.0_f32; 4 * 3 * 5];
        // Exercise both low-resolution samples whose nearest-neighbor expansion crosses the
        // requested output seam at x=4.
        values[2] = 1.0;
        values[3] = -2.0;
        let input = Tensor::from_data(TensorData::new(values, [1, 4, 3, 5]), &device);
        let full = upsample.forward(input.clone());
        let (left, right) = upsample_two_width_slabs(&upsample, input, 4);
        let striped = Tensor::cat(vec![left, right], 3);
        let max_abs = (full - striped).abs().max().into_scalar();
        assert!(
            max_abs <= 1.0e-6,
            "striped upsample convolution max_abs={max_abs}"
        );
    }

    #[test]
    fn striped_resnet_matches_full_ragged_width_parity() {
        let device = Default::default();
        let resnet = ResnetBlock2d::<TestBackend>::new(&device, 8, 4, 4, 1.0e-6);
        let input = Tensor::random([1, 8, 3, 7], Distribution::Default, &device);
        let full = resnet.forward(input.clone());
        let left = input.clone().slice([0..1, 0..8, 0..3, 0..3]);
        let right = input.slice([0..1, 0..8, 0..3, 3..7]);
        let (left, right) = resnet_two_width_slabs_strict_f32(&resnet, left, right);
        let striped = Tensor::cat(vec![left, right], 3);
        let max_abs = (full - striped).abs().max().into_scalar();
        assert!(max_abs <= 1.0e-5, "striped residual max_abs={max_abs}");
    }
}
