use burn::{
    nn,
    prelude::{Backend, Module},
    tensor::{FloatDType, Tensor},
};

use crate::{
    blocks::{DecoderGroupNormPolicy, conv1},
    config::{AutoencoderKlConfig, AutoencoderKlConfigError},
    decoder::Decoder,
    distribution::DiagonalGaussian,
    encoder::Encoder,
};

/// Ordinary Diffusers-compatible FLUX `AutoencoderKL`.
#[derive(Module, Debug)]
pub struct AutoencoderKl<B: Backend> {
    pub encoder: Encoder<B>,
    pub decoder: Decoder<B>,
    pub quant_conv: Option<nn::conv::Conv2d<B>>,
    pub post_quant_conv: Option<nn::conv::Conv2d<B>>,
    #[module(skip)]
    latent_channels: usize,
    #[module(skip)]
    scaling_factor: f64,
    #[module(skip)]
    shift_factor: f64,
    #[module(skip)]
    force_upcast: bool,
}

impl<B: Backend> AutoencoderKl<B> {
    pub fn try_new(
        device: &B::Device,
        config: &AutoencoderKlConfig,
    ) -> Result<Self, AutoencoderKlConfigError> {
        config.validate()?;
        Ok(Self {
            encoder: Encoder::new(device, config),
            decoder: Decoder::new(device, config),
            quant_conv: config
                .use_quant_conv
                .then(|| conv1(device, config.moment_channels(), config.moment_channels())),
            post_quant_conv: config
                .use_post_quant_conv
                .then(|| conv1(device, config.latent_channels, config.latent_channels)),
            latent_channels: config.latent_channels,
            scaling_factor: config.scaling_factor,
            shift_factor: config.shift_factor,
            force_upcast: config.force_upcast,
        })
    }

    /// Allocate a validated model, panicking only for an invalid programmer-supplied config.
    pub fn new(device: &B::Device, config: &AutoencoderKlConfig) -> Self {
        Self::try_new(device, config).expect("valid AutoencoderKL configuration")
    }

    /// Encode images into raw concatenated mean/log-variance moments.
    pub fn encode_moments(&self, images: Tensor<B, 4>) -> Tensor<B, 4> {
        assert_eq!(
            images.dims()[1],
            self.encoder.conv_in.weight.dims()[1],
            "AutoencoderKL image channel mismatch"
        );
        let moments = self.encoder.forward(images);
        self.quant_conv
            .as_ref()
            .map(|conv| conv.forward(moments.clone()))
            .unwrap_or(moments)
    }

    /// Encode images into a raw posterior. Scaling and shifting are intentionally separate.
    pub fn encode(&self, images: Tensor<B, 4>) -> DiagonalGaussian<B> {
        DiagonalGaussian::from_moments(self.encode_moments(images))
    }

    /// Encode using the posterior mode without applying the FLUX pipeline scale/shift.
    pub fn encode_mode(&self, images: Tensor<B, 4>) -> Tensor<B, 4> {
        self.encode(images).mode()
    }

    /// Encode and reparameterize using an exact caller-supplied epsilon tensor.
    pub fn encode_with_epsilon(&self, images: Tensor<B, 4>, epsilon: Tensor<B, 4>) -> Tensor<B, 4> {
        self.encode(images).sample_with_epsilon(epsilon)
    }

    /// Decode raw, unscaled latent maps.
    pub fn decode(&self, latents: Tensor<B, 4>) -> Tensor<B, 4> {
        self.decode_with_group_norm_policy(latents, DecoderGroupNormPolicy::StrictF32)
    }

    /// Decode raw, unscaled latent maps with an explicit decoder GroupNorm execution policy.
    ///
    /// This policy affects only the decoder. Encoder APIs retain the established strict-F32
    /// normalization path for F16/BF16 activations.
    pub fn decode_with_group_norm_policy(
        &self,
        latents: Tensor<B, 4>,
        policy: DecoderGroupNormPolicy,
    ) -> Tensor<B, 4> {
        assert_eq!(
            latents.dims()[1],
            self.latent_channels,
            "AutoencoderKL latent channel mismatch"
        );
        let latents = self
            .post_quant_conv
            .as_ref()
            .map(|conv| conv.forward(latents.clone()))
            .unwrap_or(latents);
        self.decoder.forward_with_group_norm_policy(latents, policy)
    }

    /// Apply FLUX pipeline latent normalization: `(latents - shift_factor) * scaling_factor`.
    pub fn scale_latents(&self, latents: Tensor<B, 4>) -> Tensor<B, 4> {
        (latents - self.shift_factor) * self.scaling_factor
    }

    /// Undo FLUX pipeline latent normalization: `latents / scaling_factor + shift_factor`.
    pub fn unscale_latents(&self, latents: Tensor<B, 4>) -> Tensor<B, 4> {
        latents / self.scaling_factor + self.shift_factor
    }

    pub fn encode_scaled_mode(&self, images: Tensor<B, 4>) -> Tensor<B, 4> {
        self.scale_latents(self.encode_mode(images))
    }

    pub fn encode_scaled_with_epsilon(
        &self,
        images: Tensor<B, 4>,
        epsilon: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        self.scale_latents(self.encode_with_epsilon(images, epsilon))
    }

    pub fn decode_scaled(&self, scaled_latents: Tensor<B, 4>) -> Tensor<B, 4> {
        self.decode_scaled_with_group_norm_policy(scaled_latents, DecoderGroupNormPolicy::StrictF32)
    }

    /// Undo FLUX scaling and decode with an explicit decoder GroupNorm execution policy.
    pub fn decode_scaled_with_group_norm_policy(
        &self,
        scaled_latents: Tensor<B, 4>,
        policy: DecoderGroupNormPolicy,
    ) -> Tensor<B, 4> {
        self.decode_with_group_norm_policy(self.unscale_latents(scaled_latents), policy)
    }

    /// Diffusers-style deterministic VAE forward; it does not apply pipeline latent scaling.
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 4> {
        self.decode(self.encode_mode(images))
    }

    /// Diffusers-style sampled VAE forward with explicit reparameterization epsilon.
    pub fn forward_with_epsilon(
        &self,
        images: Tensor<B, 4>,
        epsilon: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        self.decode(self.encode_with_epsilon(images, epsilon))
    }

    pub fn scaling_factor(&self) -> f64 {
        self.scaling_factor
    }

    pub fn shift_factor(&self) -> f64 {
        self.shift_factor
    }

    pub fn force_upcast(&self) -> bool {
        self.force_upcast
    }

    pub fn float_dtype(&self) -> FloatDType {
        self.encoder.conv_in.weight.val().dtype().into()
    }

    /// Parameter dtype of the encoder stage.
    ///
    /// This differs from [`Self::decoder_float_dtype`] for independently staged loaders, where
    /// the opposite half remains lazy in the backend's default dtype.
    pub fn encoder_float_dtype(&self) -> FloatDType {
        self.encoder.conv_in.weight.val().dtype().into()
    }

    /// Parameter dtype of the decoder stage.
    ///
    /// This must be used by decoder-only runtimes because the lazy encoder does not describe the
    /// loaded decoder's execution policy.
    pub fn decoder_float_dtype(&self) -> FloatDType {
        self.decoder.conv_in.weight.val().dtype().into()
    }
}

impl AutoencoderKlConfig {
    pub fn try_init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<AutoencoderKl<B>, AutoencoderKlConfigError> {
        AutoencoderKl::try_new(device, self)
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> AutoencoderKl<B> {
        AutoencoderKl::new(device, self)
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Distribution, TensorData};

    use super::*;

    type TestBackend = burn::backend::NdArray<f32>;

    fn values<const D: usize>(tensor: Tensor<TestBackend, D>) -> Vec<f32> {
        tensor.to_data().to_vec::<f32>().expect("tensor values")
    }

    #[test]
    fn autoencoder_shapes_smoke() {
        let device = Default::default();
        let model = AutoencoderKlConfig::tiny().init::<TestBackend>(&device);
        let images = Tensor::random([1, 3, 8, 8], Distribution::Default, &device);
        let moments = model.encode_moments(images);
        assert_eq!(moments.dims(), [1, 8, 4, 4]);
        let decoded = model.decode(DiagonalGaussian::from_moments(moments).mode());
        assert_eq!(decoded.dims(), [1, 3, 8, 8]);
        assert!(decoded.is_finite().all().into_scalar());
    }

    #[test]
    fn explicit_zero_epsilon_matches_mode_correctness() {
        let device = Default::default();
        let model = AutoencoderKlConfig::tiny().init::<TestBackend>(&device);
        let images = Tensor::random([1, 3, 8, 8], Distribution::Default, &device);
        let mode = model.encode_mode(images.clone());
        let sampled = model.encode_with_epsilon(images, Tensor::zeros([1, 4, 4, 4], &device));
        assert_eq!(values(mode), values(sampled));
    }

    #[test]
    fn scale_shift_round_trip_correctness() {
        let device = Default::default();
        let model = AutoencoderKlConfig::tiny().init::<TestBackend>(&device);
        let latents = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(vec![-1.0, 0.0, 1.0, 2.0], [1, 4, 1, 1]),
            &device,
        );
        let round_trip = model.unscale_latents(model.scale_latents(latents.clone()));
        assert_eq!(
            values(model.scale_latents(latents.clone())),
            vec![-0.625, -0.125, 0.375, 0.875]
        );
        let actual = values(round_trip);
        let expected = values(latents);
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() <= 1.0e-6);
        }
    }

    #[test]
    fn decoder_group_norm_policy_f32_cpu_parity() {
        assert_eq!(
            DecoderGroupNormPolicy::default(),
            DecoderGroupNormPolicy::StrictF32
        );

        let device = Default::default();
        let model = AutoencoderKlConfig::tiny().init::<TestBackend>(&device);
        let latents = Tensor::random([1, 4, 4, 4], Distribution::Default, &device);
        let established = model.decode(latents.clone());
        let strict =
            model.decode_with_group_norm_policy(latents.clone(), DecoderGroupNormPolicy::StrictF32);
        let mixed = model
            .decode_with_group_norm_policy(latents, DecoderGroupNormPolicy::F16StorageF32Accum);

        assert_eq!(established.dims(), [1, 3, 8, 8]);
        assert!(established.clone().is_finite().all().into_scalar());
        let established = values(established);
        assert_eq!(established, values(strict));
        assert_eq!(established, values(mixed));
    }
}
