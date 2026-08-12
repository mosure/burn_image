use burn::{
    nn,
    prelude::{Backend, Module},
    tensor::Tensor,
};

use crate::{
    blocks::{Downsample2d, MidBlock2d, ResnetBlock2d, conv3, group_norm, group_norm_f32, silu},
    config::AutoencoderKlConfig,
};

/// Ordinary Diffusers `DownEncoderBlock2D`.
#[derive(Module, Debug)]
pub struct DownEncoderBlock2d<B: Backend> {
    pub resnets: Vec<ResnetBlock2d<B>>,
    pub downsamplers: Vec<Downsample2d<B>>,
}

impl<B: Backend> DownEncoderBlock2d<B> {
    pub fn new(
        device: &B::Device,
        in_channels: usize,
        out_channels: usize,
        layers: usize,
        groups: usize,
        epsilon: f64,
        add_downsample: bool,
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
            downsamplers: add_downsample
                .then(|| Downsample2d::new(device, out_channels))
                .into_iter()
                .collect(),
        }
    }

    pub fn forward(&self, mut hidden: Tensor<B, 4>) -> Tensor<B, 4> {
        for resnet in &self.resnets {
            hidden = resnet.forward(hidden);
        }
        if let Some(downsample) = self.downsamplers.first() {
            hidden = downsample.forward(hidden);
        }
        hidden
    }
}

/// Diffusers-compatible AutoencoderKL encoder producing concatenated mean/log-variance moments.
#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    pub conv_in: nn::conv::Conv2d<B>,
    pub down_blocks: Vec<DownEncoderBlock2d<B>>,
    pub mid_block: MidBlock2d<B>,
    pub conv_norm_out: nn::GroupNorm<B>,
    pub conv_out: nn::conv::Conv2d<B>,
}

impl<B: Backend> Encoder<B> {
    pub fn new(device: &B::Device, config: &AutoencoderKlConfig) -> Self {
        let first_channels = config.block_out_channels[0];
        let last_channels = *config
            .block_out_channels
            .last()
            .expect("validated VAE block channels");
        let mut input_channels = first_channels;
        let mut down_blocks = Vec::with_capacity(config.block_out_channels.len());
        for (index, &output_channels) in config.block_out_channels.iter().enumerate() {
            down_blocks.push(DownEncoderBlock2d::new(
                device,
                input_channels,
                output_channels,
                config.layers_per_block,
                config.norm_num_groups,
                config.norm_epsilon,
                index + 1 != config.block_out_channels.len(),
            ));
            input_channels = output_channels;
        }
        Self {
            conv_in: conv3(device, config.in_channels, first_channels),
            down_blocks,
            mid_block: MidBlock2d::new(
                device,
                last_channels,
                config.norm_num_groups,
                config.norm_epsilon,
                config.mid_block_add_attention,
                config.attention_query_chunk_size,
            ),
            conv_norm_out: group_norm(
                device,
                config.norm_num_groups,
                last_channels,
                config.norm_epsilon,
            ),
            conv_out: conv3(device, last_channels, config.moment_channels()),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut hidden = self.conv_in.forward(input);
        for block in &self.down_blocks {
            hidden = block.forward(hidden);
        }
        hidden = self.mid_block.forward(hidden);
        self.conv_out
            .forward(silu(group_norm_f32(&self.conv_norm_out, hidden)))
    }
}
