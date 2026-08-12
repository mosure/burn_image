use burn::{nn, prelude::*, tensor::activation::silu};

use super::norm::{DenoiserRmsNormPolicy, layer_norm_no_affine, rms_norm_with_policy};

/// Combined timestep and Qwen caption embedding.
#[derive(Module, Debug)]
pub struct CombinedTimestepCaptionEmbedding<B: Backend> {
    /// First timestep MLP projection.
    pub time_linear_1: nn::Linear<B>,
    /// Second timestep MLP projection.
    pub time_linear_2: nn::Linear<B>,
    /// Caption RMSNorm.
    pub caption_norm: nn::RmsNorm<B>,
    /// Caption width projection.
    pub caption_linear: nn::Linear<B>,
    frequency_width: usize,
    conditioning_width: usize,
    timestep_scale: f64,
}

impl<B: Backend> CombinedTimestepCaptionEmbedding<B> {
    /// Create the released embedding topology.
    pub fn new(
        model_width: usize,
        instruction_width: usize,
        frequency_width: usize,
        conditioning_width: usize,
        epsilon: f64,
        timestep_scale: f64,
        device: &B::Device,
    ) -> Self {
        Self {
            time_linear_1: nn::LinearConfig::new(frequency_width, conditioning_width).init(device),
            time_linear_2: nn::LinearConfig::new(conditioning_width, conditioning_width)
                .init(device),
            caption_norm: nn::RmsNormConfig::new(instruction_width)
                .with_epsilon(epsilon)
                .init(device),
            caption_linear: nn::LinearConfig::new(instruction_width, model_width).init(device),
            frequency_width,
            conditioning_width,
            timestep_scale,
        }
    }

    /// Embed scalar timesteps and instruction tokens.
    pub fn forward(
        &self,
        timestep: Tensor<B, 1>,
        instruction: Tensor<B, 3>,
    ) -> (Tensor<B, 2>, Tensor<B, 3>) {
        self.forward_with_rms_norm_policy(timestep, instruction, DenoiserRmsNormPolicy::StrictF32)
    }

    pub(crate) fn forward_with_rms_norm_policy(
        &self,
        timestep: Tensor<B, 1>,
        instruction: Tensor<B, 3>,
        rms_norm_policy: DenoiserRmsNormPolicy,
    ) -> (Tensor<B, 2>, Tensor<B, 3>) {
        let time = timestep_embedding(timestep, self.frequency_width, self.timestep_scale);
        let time = self
            .time_linear_2
            .forward(silu(self.time_linear_1.forward(time)));
        debug_assert_eq!(time.dims()[1], self.conditioning_width);
        let caption = self.caption_linear.forward(rms_norm_with_policy(
            &self.caption_norm,
            instruction,
            rms_norm_policy,
        ));
        (time, caption)
    }
}

fn timestep_embedding<B: Backend>(
    timestep: Tensor<B, 1>,
    width: usize,
    scale: f64,
) -> Tensor<B, 2> {
    let device = timestep.device();
    let dtype = timestep.dtype();
    let half = width / 2;
    let exponents = Tensor::<B, 1, Int>::arange(0..half as i64, &device).float()
        * (-10000.0_f64.ln() / half as f64);
    let exponents = exponents.cast(dtype);
    let frequencies = exponents.exp();
    let phase = timestep.unsqueeze_dim(1) * frequencies.unsqueeze_dim(0) * scale;
    // Diffusers Timesteps with flip_sin_to_cos=true.
    Tensor::cat(vec![phase.clone().cos(), phase.sin()], 1)
}

/// Final adaptive normalization and latent-patch projection.
#[derive(Module, Debug)]
pub struct FinalProjection<B: Backend> {
    /// Conditioning-to-scale projection.
    pub linear_1: nn::Linear<B>,
    /// Patch output projection.
    pub linear_2: nn::Linear<B>,
    epsilon: f64,
}

impl<B: Backend> FinalProjection<B> {
    /// Create the upstream final projection.
    pub fn new(
        width: usize,
        conditioning_width: usize,
        output_width: usize,
        epsilon: f64,
        device: &B::Device,
    ) -> Self {
        Self {
            linear_1: nn::LinearConfig::new(conditioning_width, width).init(device),
            linear_2: nn::LinearConfig::new(width, output_width).init(device),
            epsilon,
        }
    }

    /// Project joint tokens to latent patches.
    pub fn forward(&self, tokens: Tensor<B, 3>, conditioning: Tensor<B, 2>) -> Tensor<B, 3> {
        let scale = self.linear_1.forward(silu(conditioning)).unsqueeze_dim(1) + 1.0;
        self.linear_2
            .forward(layer_norm_no_affine(tokens, self.epsilon) * scale)
    }
}
