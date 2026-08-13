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

        let mut hidden = self.conv_in.forward(input);
        hidden = self
            .mid_block
            .forward_with_group_norm_policy(hidden, DecoderGroupNormPolicy::StrictF32);
        for block in &self.up_blocks[..upsample_block_index] {
            hidden =
                block.forward_with_group_norm_policy(hidden, DecoderGroupNormPolicy::StrictF32);
        }

        let upsample_block = &self.up_blocks[upsample_block_index];
        for resnet in &upsample_block.resnets {
            hidden =
                resnet.forward_with_group_norm_policy(hidden, DecoderGroupNormPolicy::StrictF32);
        }
        let (left, right) = upsample_two_width_slabs(
            upsample_block
                .upsamplers
                .first()
                .expect("validated final upsampler"),
            hidden,
            split_width,
        );

        let final_block = &self.up_blocks[final_block_index];
        let first_resnet = final_block
            .resnets
            .first()
            .expect("decoder up block contains a residual layer");
        let (left, right) = resnet_two_width_slabs_strict_f32(first_resnet, left, right);
        hidden = Tensor::cat(vec![left, right], 3);
        for resnet in final_block.resnets.iter().skip(1) {
            hidden =
                resnet.forward_with_group_norm_policy(hidden, DecoderGroupNormPolicy::StrictF32);
        }

        self.conv_out.forward(silu(group_norm_with_policy(
            &self.conv_norm_out,
            hidden,
            DecoderGroupNormPolicy::StrictF32,
        )))
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

fn conv3_two_width_slabs<B: Backend>(
    conv: &nn::conv::Conv2d<B>,
    left: Tensor<B, 4>,
    right: Tensor<B, 4>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [batch, channels, height, left_width] = left.dims();
    let [right_batch, right_channels, right_height, right_width] = right.dims();
    assert_eq!(
        [batch, channels, height],
        [right_batch, right_channels, right_height]
    );
    assert!(left_width > 0 && right_width > 0, "empty convolution slab");
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
    let right = conv
        .forward(Tensor::cat(vec![right_halo, right], 3))
        .slice([
            0..batch,
            0..conv.weight.dims()[0],
            0..height,
            1..right_width + 1,
        ]);
    (left, right)
}

fn group_norm_two_width_slabs_strict_f32<B: Backend>(
    norm: &nn::GroupNorm<B>,
    left: Tensor<B, 4>,
    right: Tensor<B, 4>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [batch, channels, height, left_width] = left.dims();
    let [right_batch, right_channels, right_height, right_width] = right.dims();
    assert_eq!(
        [batch, channels, height],
        [right_batch, right_channels, right_height]
    );
    assert_eq!(channels, norm.num_channels, "GroupNorm channel mismatch");
    assert!(left_width > 0 && right_width > 0, "empty GroupNorm slab");

    let dtype: FloatDType = left.dtype().into();
    assert_eq!(dtype, right.dtype().into(), "GroupNorm slab dtype mismatch");
    let accumulation_dtype = if matches!(dtype, FloatDType::F16 | FloatDType::BF16) {
        FloatDType::F32
    } else {
        dtype
    };
    let group_channels = channels / norm.num_groups;
    let left_group_width = group_channels * height * left_width;
    let right_group_width = group_channels * height * right_width;
    let left = left
        .cast(accumulation_dtype)
        .reshape([batch, norm.num_groups, left_group_width]);
    let right = right
        .cast(accumulation_dtype)
        .reshape([batch, norm.num_groups, right_group_width]);
    let mean = (left.clone().sum_dim(2) + right.clone().sum_dim(2))
        / (left_group_width + right_group_width) as f64;
    let left = left - mean.clone();
    let right = right - mean;
    let variance = (left.clone().square().sum_dim(2) + right.clone().square().sum_dim(2))
        / (left_group_width + right_group_width) as f64;
    let inverse_std = variance.add_scalar(norm.epsilon).sqrt().recip();
    let left = left * inverse_std.clone();
    let right = right * inverse_std;

    let affine = |input: Tensor<B, 3>, width: usize| {
        let input = input.reshape([batch, channels, height, width]);
        if !norm.affine {
            return input.cast(dtype);
        }
        let gamma = norm
            .gamma
            .as_ref()
            .expect("affine GroupNorm gamma")
            .val()
            .cast(accumulation_dtype)
            .reshape([1, channels, 1, 1]);
        let beta = norm
            .beta
            .as_ref()
            .expect("affine GroupNorm beta")
            .val()
            .cast(accumulation_dtype)
            .reshape([1, channels, 1, 1]);
        (input * gamma + beta).cast(dtype)
    };
    (affine(left, left_width), affine(right, right_width))
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
