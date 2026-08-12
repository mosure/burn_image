use burn::{
    nn,
    prelude::{Backend, Module},
    tensor::Tensor,
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
}
