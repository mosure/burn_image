use burn::{
    nn::{self, PaddingConfig2d},
    prelude::{Backend, Module},
    tensor::{FloatDType, Tensor, activation, module::interpolate, ops::InterpolateOptions},
};

/// Numerical policy for GroupNorm operations inside the VAE decoder.
///
/// [`Self::StrictF32`] preserves the established path by widening F16/BF16 activations before
/// normalization. [`Self::F16StorageF32Accum`] keeps full-size F16 intermediates in F16, but uses
/// overflow-safe power-of-two scaling and F32 accumulation for reductions. The latter is an opt-in
/// native performance policy and must pass the caller's numerical gate on the selected backend.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DecoderGroupNormPolicy {
    /// Widen F16/BF16 activations to F32 for the complete normalization and affine operation.
    #[default]
    StrictF32,
    /// Keep F16 activation storage while GroupNorm reductions accumulate in F32.
    F16StorageF32Accum,
}

/// Diffusers-compatible VAE residual block without timestep conditioning.
#[derive(Module, Debug)]
pub struct ResnetBlock2d<B: Backend> {
    pub norm1: nn::GroupNorm<B>,
    pub conv1: nn::conv::Conv2d<B>,
    pub norm2: nn::GroupNorm<B>,
    pub conv2: nn::conv::Conv2d<B>,
    pub conv_shortcut: Option<nn::conv::Conv2d<B>>,
}

impl<B: Backend> ResnetBlock2d<B> {
    pub fn new(
        device: &B::Device,
        in_channels: usize,
        out_channels: usize,
        groups: usize,
        epsilon: f64,
    ) -> Self {
        Self {
            norm1: group_norm(device, groups, in_channels, epsilon),
            conv1: conv3(device, in_channels, out_channels),
            norm2: group_norm(device, groups, out_channels, epsilon),
            conv2: conv3(device, out_channels, out_channels),
            conv_shortcut: (in_channels != out_channels)
                .then(|| conv1(device, in_channels, out_channels)),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        self.forward_with_group_norm_policy(input, DecoderGroupNormPolicy::StrictF32)
    }

    pub(crate) fn forward_with_group_norm_policy(
        &self,
        input: Tensor<B, 4>,
        policy: DecoderGroupNormPolicy,
    ) -> Tensor<B, 4> {
        let hidden = self.conv1.forward(silu(group_norm_with_policy(
            &self.norm1,
            input.clone(),
            policy,
        )));
        let hidden = self
            .conv2
            .forward(silu(group_norm_with_policy(&self.norm2, hidden, policy)));
        let residual = self
            .conv_shortcut
            .as_ref()
            .map(|shortcut| shortcut.forward(input.clone()))
            .unwrap_or(input);
        residual + hidden
    }
}

/// Diffusers `Downsample2D` with asymmetric bottom/right zero padding.
#[derive(Module, Debug)]
pub struct Downsample2d<B: Backend> {
    pub conv: nn::conv::Conv2d<B>,
}

impl<B: Backend> Downsample2d<B> {
    pub fn new(device: &B::Device, channels: usize) -> Self {
        Self {
            conv: nn::conv::Conv2dConfig::new([channels, channels], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Valid)
                .init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        self.conv.forward(pad_bottom_right(input))
    }
}

/// Diffusers `Upsample2D`: nearest-neighbor 2x resize followed by a padded 3x3 convolution.
#[derive(Module, Debug)]
pub struct Upsample2d<B: Backend> {
    pub conv: nn::conv::Conv2d<B>,
}

impl<B: Backend> Upsample2d<B> {
    pub fn new(device: &B::Device, channels: usize) -> Self {
        Self {
            conv: conv3(device, channels, channels),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_batch, _channels, height, width] = input.dims();
        let input = interpolate(
            input,
            [height * 2, width * 2],
            InterpolateOptions::new(burn::tensor::ops::InterpolateMode::Nearest),
        );
        self.conv.forward(input)
    }
}

/// Single-head spatial self-attention used in the AutoencoderKL middle block.
///
/// Query chunking preserves exact attention semantics while bounding fallback score storage.
/// Softmax is evaluated in F32, matching Diffusers' `upcast_softmax=true` behavior.
#[derive(Module, Debug)]
pub struct AttentionBlock<B: Backend> {
    pub group_norm: nn::GroupNorm<B>,
    pub to_q: nn::Linear<B>,
    pub to_k: nn::Linear<B>,
    pub to_v: nn::Linear<B>,
    pub to_out: nn::Linear<B>,
    channels: usize,
    query_chunk_size: usize,
}

impl<B: Backend> AttentionBlock<B> {
    pub fn new(
        device: &B::Device,
        channels: usize,
        groups: usize,
        epsilon: f64,
        query_chunk_size: usize,
    ) -> Self {
        assert!(
            query_chunk_size > 0,
            "attention query chunk must be non-zero"
        );
        let linear = || {
            nn::LinearConfig::new(channels, channels)
                .with_bias(true)
                .init(device)
        };
        Self {
            group_norm: group_norm(device, groups, channels, epsilon),
            to_q: linear(),
            to_k: linear(),
            to_v: linear(),
            to_out: linear(),
            channels,
            query_chunk_size,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        self.forward_with_group_norm_policy(input, DecoderGroupNormPolicy::StrictF32)
    }

    /// Update the exact-query attention partition without changing model parameters.
    pub fn set_query_chunk_size(&mut self, query_chunk_size: usize) {
        assert!(
            query_chunk_size > 0,
            "attention query chunk must be non-zero"
        );
        self.query_chunk_size = query_chunk_size;
    }

    pub(crate) fn forward_with_group_norm_policy(
        &self,
        input: Tensor<B, 4>,
        policy: DecoderGroupNormPolicy,
    ) -> Tensor<B, 4> {
        let [batch, channels, height, width] = input.dims();
        assert_eq!(channels, self.channels, "attention channel mismatch");
        let tokens = height * width;
        let hidden = group_norm_with_policy(&self.group_norm, input.clone(), policy)
            .reshape([batch, channels, tokens])
            .swap_dims(1, 2);
        let query = self
            .to_q
            .forward(hidden.clone())
            .reshape([batch, 1, tokens, channels]);
        let key = self
            .to_k
            .forward(hidden.clone())
            .reshape([batch, 1, tokens, channels]);
        let value = self
            .to_v
            .forward(hidden)
            .reshape([batch, 1, tokens, channels]);
        let dtype: FloatDType = query.dtype().into();
        let key_transposed = key.cast(FloatDType::F32).swap_dims(2, 3);
        let scale = (channels as f64).powf(-0.5);
        let mut chunks = Vec::with_capacity(tokens.div_ceil(self.query_chunk_size));
        for start in (0..tokens).step_by(self.query_chunk_size) {
            let end = (start + self.query_chunk_size).min(tokens);
            let query = query
                .clone()
                .slice([0..batch, 0..1, start..end, 0..channels])
                .cast(FloatDType::F32);
            let scores = query.matmul(key_transposed.clone()).mul_scalar(scale);
            let probabilities = activation::softmax(scores, 3).cast(dtype);
            chunks.push(probabilities.matmul(value.clone()));
        }
        let attended = Tensor::cat(chunks, 2).reshape([batch, tokens, channels]);
        let output = self
            .to_out
            .forward(attended)
            .swap_dims(1, 2)
            .reshape([batch, channels, height, width]);
        input + output
    }
}

/// Diffusers `UNetMidBlock2D`: residual, optional attention, residual.
#[derive(Module, Debug)]
pub struct MidBlock2d<B: Backend> {
    pub resnets: Vec<ResnetBlock2d<B>>,
    pub attentions: Vec<AttentionBlock<B>>,
}

impl<B: Backend> MidBlock2d<B> {
    pub fn new(
        device: &B::Device,
        channels: usize,
        groups: usize,
        epsilon: f64,
        add_attention: bool,
        query_chunk_size: usize,
    ) -> Self {
        Self {
            resnets: vec![
                ResnetBlock2d::new(device, channels, channels, groups, epsilon),
                ResnetBlock2d::new(device, channels, channels, groups, epsilon),
            ],
            attentions: add_attention
                .then(|| AttentionBlock::new(device, channels, groups, epsilon, query_chunk_size))
                .into_iter()
                .collect(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        self.forward_with_group_norm_policy(input, DecoderGroupNormPolicy::StrictF32)
    }

    /// Update attention partitioning for every middle-block attention module.
    pub fn set_attention_query_chunk_size(&mut self, query_chunk_size: usize) {
        for attention in &mut self.attentions {
            attention.set_query_chunk_size(query_chunk_size);
        }
    }

    pub(crate) fn forward_with_group_norm_policy(
        &self,
        input: Tensor<B, 4>,
        policy: DecoderGroupNormPolicy,
    ) -> Tensor<B, 4> {
        let mut hidden = self.resnets[0].forward_with_group_norm_policy(input, policy);
        if let Some(attention) = self.attentions.first() {
            hidden = attention.forward_with_group_norm_policy(hidden, policy);
        }
        self.resnets[1].forward_with_group_norm_policy(hidden, policy)
    }
}

pub(crate) fn conv1<B: Backend>(
    device: &B::Device,
    in_channels: usize,
    out_channels: usize,
) -> nn::conv::Conv2d<B> {
    nn::conv::Conv2dConfig::new([in_channels, out_channels], [1, 1]).init(device)
}

pub(crate) fn conv3<B: Backend>(
    device: &B::Device,
    in_channels: usize,
    out_channels: usize,
) -> nn::conv::Conv2d<B> {
    nn::conv::Conv2dConfig::new([in_channels, out_channels], [3, 3])
        .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
        .init(device)
}

pub(crate) fn group_norm<B: Backend>(
    device: &B::Device,
    groups: usize,
    channels: usize,
    epsilon: f64,
) -> nn::GroupNorm<B> {
    nn::GroupNormConfig::new(groups, channels)
        .with_epsilon(epsilon)
        .init(device)
}

pub(crate) fn silu<B: Backend, const D: usize>(input: Tensor<B, D>) -> Tensor<B, D> {
    input.clone() * activation::sigmoid(input)
}

/// GroupNorm with F32 reduction for F16/BF16 inputs, then cast back to the input dtype.
pub(crate) fn group_norm_f32<B: Backend, const D: usize>(
    norm: &nn::GroupNorm<B>,
    input: Tensor<B, D>,
) -> Tensor<B, D> {
    let dtype: FloatDType = input.dtype().into();
    if !matches!(dtype, FloatDType::F16 | FloatDType::BF16) {
        return norm.forward(input);
    }
    let shape = input.shape();
    assert_eq!(shape[1], norm.num_channels, "GroupNorm channel mismatch");
    let batch = shape[0];
    let channels = shape[1];
    let group_width = shape[2..].iter().product::<usize>() * channels / norm.num_groups;
    let grouped = input
        .cast(FloatDType::F32)
        .reshape([batch, norm.num_groups, group_width]);
    let centered = grouped.clone() - grouped.mean_dim(2);
    let variance = centered.clone().square().mean_dim(2);
    let normalized = centered / variance.add_scalar(norm.epsilon).sqrt();
    let mut affine_shape = [1; D];
    affine_shape[1] = channels;
    let output = normalized.reshape(shape)
        * norm
            .gamma
            .as_ref()
            .expect("affine GroupNorm gamma")
            .val()
            .cast(FloatDType::F32)
            .reshape(affine_shape)
        + norm
            .beta
            .as_ref()
            .expect("affine GroupNorm beta")
            .val()
            .cast(FloatDType::F32)
            .reshape(affine_shape);
    output.cast(dtype)
}

/// GroupNorm retaining F16 storage without allowing an F16 sum or square to overflow.
///
/// Burn's ordinary GroupNorm computes `sum_dim / N`. CubeCL accumulates that sum in F32, but the
/// reduction result is written in the input dtype before the division. For a large F16 group this
/// can therefore overflow even when its mean is small. The centered square is also evaluated in
/// F16 before its reduction and can overflow for a finite input whose magnitude exceeds 256.
///
/// Scaling by 2^-9 bounds any centered finite F16 value to 255.875, so its square remains finite.
/// `mean_dim` performs the division in the reduction accumulator on CubeCL, avoiding a large F16
/// sum result. Only the reduced variance and inverse standard deviation use F32 storage. Splitting
/// the inverse-standard-deviation product by 2^5 keeps both F16 factors finite even for zero
/// variance and epsilon=1e-6.
fn group_norm_f16_storage_f32_accum<B: Backend, const D: usize>(
    norm: &nn::GroupNorm<B>,
    input: Tensor<B, D>,
) -> Tensor<B, D> {
    const STORAGE_SCALE: f64 = 1.0 / 512.0;
    const NORMALIZATION_BOOST: f64 = 32.0;

    let dtype: FloatDType = input.dtype().into();
    if dtype == FloatDType::F32 {
        return norm.forward(input);
    }
    assert_eq!(
        dtype,
        FloatDType::F16,
        "F16-storage GroupNorm requires F16 or F32 input"
    );

    let shape = input.shape();
    assert_eq!(shape[1], norm.num_channels, "GroupNorm channel mismatch");
    let batch = shape[0];
    let channels = shape[1];
    let group_width = shape[2..].iter().product::<usize>() * channels / norm.num_groups;
    let grouped_scaled =
        input
            .mul_scalar(STORAGE_SCALE)
            .reshape([batch, norm.num_groups, group_width]);
    let centered_scaled = grouped_scaled.clone() - grouped_scaled.mean_dim(2);
    let variance_scaled = centered_scaled.clone().square().mean_dim(2);
    let inverse_std_split = variance_scaled
        .cast(FloatDType::F32)
        .add_scalar(norm.epsilon * STORAGE_SCALE * STORAGE_SCALE)
        .sqrt()
        .recip()
        .div_scalar(NORMALIZATION_BOOST)
        .cast(dtype);
    let normalized = centered_scaled.mul_scalar(NORMALIZATION_BOOST) * inverse_std_split;

    let mut affine_shape = [1; D];
    affine_shape[1] = channels;
    normalized.reshape(shape)
        * norm
            .gamma
            .as_ref()
            .expect("affine GroupNorm gamma")
            .val()
            .cast(dtype)
            .reshape(affine_shape)
        + norm
            .beta
            .as_ref()
            .expect("affine GroupNorm beta")
            .val()
            .cast(dtype)
            .reshape(affine_shape)
}

pub(crate) fn group_norm_with_policy<B: Backend, const D: usize>(
    norm: &nn::GroupNorm<B>,
    input: Tensor<B, D>,
    policy: DecoderGroupNormPolicy,
) -> Tensor<B, D> {
    match policy {
        DecoderGroupNormPolicy::StrictF32 => group_norm_f32(norm, input),
        DecoderGroupNormPolicy::F16StorageF32Accum => group_norm_f16_storage_f32_accum(norm, input),
    }
}

pub(crate) fn pad_bottom_right<B: Backend>(input: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, channels, height, width] = input.dims();
    let device = input.device();
    let dtype: FloatDType = input.dtype().into();
    let right = Tensor::<B, 4>::zeros([batch, channels, height, 1], &device).cast(dtype);
    let input = Tensor::cat(vec![input, right], 3);
    let bottom = Tensor::<B, 4>::zeros([batch, channels, 1, width + 1], &device).cast(dtype);
    Tensor::cat(vec![input, bottom], 2)
}

#[cfg(test)]
mod tests {
    use burn::tensor::Distribution;
    #[cfg(feature = "flex")]
    use burn::tensor::TensorData;

    use super::*;

    type TestBackend = burn::backend::NdArray<f32>;
    #[cfg(feature = "flex")]
    type MixedPrecisionTestBackend = burn::backend::Flex;

    #[test]
    fn downsample_shape_and_padding_correctness() {
        let device = Default::default();
        let downsample = Downsample2d::<TestBackend>::new(&device, 4);
        let input = Tensor::random([1, 4, 8, 10], Distribution::Default, &device);
        let output = downsample.forward(input);
        assert_eq!(output.dims(), [1, 4, 4, 5]);
    }

    #[test]
    fn upsample_shape_smoke() {
        let device = Default::default();
        let upsample = Upsample2d::<TestBackend>::new(&device, 4);
        let input = Tensor::zeros([1, 4, 3, 5], &device);
        assert_eq!(upsample.forward(input).dims(), [1, 4, 6, 10]);
    }

    #[test]
    fn chunked_attention_shape_and_finite_correctness() {
        let device = Default::default();
        let attention = AttentionBlock::<TestBackend>::new(&device, 8, 4, 1.0e-6, 3);
        let input = Tensor::random([1, 8, 3, 3], Distribution::Default, &device);
        let output = attention.forward(input);
        assert_eq!(output.dims(), [1, 8, 3, 3]);
        assert!(output.is_finite().all().into_scalar());
    }

    #[test]
    fn attention_query_chunking_parity() {
        let device = Default::default();
        let mut attention = AttentionBlock::<TestBackend>::new(&device, 8, 4, 1.0e-6, 3);
        let input = Tensor::random([1, 8, 3, 3], Distribution::Default, &device);
        let chunked = attention
            .forward(input.clone())
            .to_data()
            .to_vec::<f32>()
            .expect("chunked values");
        attention.set_query_chunk_size(64);
        let unchunked = attention
            .forward(input)
            .to_data()
            .to_vec::<f32>()
            .expect("unchunked values");
        let max_abs = chunked
            .iter()
            .zip(unchunked)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_abs <= 1.0e-6, "attention chunking max_abs={max_abs}");
    }

    #[test]
    #[cfg(feature = "flex")]
    fn mixed_group_norm_avoids_f16_sum_overflow_correctness() {
        let device = Default::default();
        let norm = group_norm::<MixedPrecisionTestBackend>(&device, 1, 2, 1.0e-6);
        // Burn's ordinary `sum_dim / N` GroupNorm writes 1024 * 100 to F16 before division.
        let input = Tensor::<MixedPrecisionTestBackend, 4>::ones([1, 2, 16, 32], &device)
            .mul_scalar(100.0)
            .cast(FloatDType::F16);
        let strict = group_norm_f32(&norm, input.clone());
        let mixed = group_norm_f16_storage_f32_accum(&norm, input);

        assert!(mixed.clone().is_finite().all().into_scalar());
        let max_abs = (strict.cast(FloatDType::F32) - mixed.cast(FloatDType::F32))
            .abs()
            .max()
            .into_scalar();
        assert!(max_abs <= 1.0e-3, "mixed GroupNorm max_abs={max_abs}");
    }

    #[test]
    #[cfg(feature = "flex")]
    fn mixed_group_norm_avoids_f16_square_overflow_correctness() {
        let device = Default::default();
        let norm = group_norm::<MixedPrecisionTestBackend>(&device, 1, 2, 1.0e-6);
        let input = Tensor::<MixedPrecisionTestBackend, 4>::from_data(
            TensorData::new(vec![65504.0_f32, -65504.0, 65504.0, -65504.0], [1, 2, 1, 2]),
            &device,
        )
        .cast(FloatDType::F16);
        let strict = group_norm_f32(&norm, input.clone());
        let mixed = group_norm_f16_storage_f32_accum(&norm, input);

        assert!(mixed.clone().is_finite().all().into_scalar());
        let max_abs = (strict.cast(FloatDType::F32) - mixed.cast(FloatDType::F32))
            .abs()
            .max()
            .into_scalar();
        assert!(max_abs <= 2.0e-3, "mixed GroupNorm max_abs={max_abs}");
    }

    #[test]
    #[cfg(feature = "flex")]
    #[should_panic(expected = "F16-storage GroupNorm requires F16 or F32 input")]
    fn mixed_group_norm_rejects_bf16_storage_correctness() {
        let device = Default::default();
        let norm = group_norm::<MixedPrecisionTestBackend>(&device, 1, 2, 1.0e-6);
        let input = Tensor::<MixedPrecisionTestBackend, 4>::ones([1, 2, 2, 2], &device)
            .cast(FloatDType::BF16);
        let _ = group_norm_f16_storage_f32_accum(&norm, input);
    }
}
