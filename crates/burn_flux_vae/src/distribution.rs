use burn::{
    prelude::Backend,
    tensor::{Distribution, FloatDType, Tensor},
};

/// Diagonal Gaussian posterior produced by a Diffusers `AutoencoderKL` encoder.
///
/// Moments are split along NCHW channel dimension and log-variance is clamped to `[-30, 20]`,
/// matching Diffusers. Numerical parity should use [`Self::sample_with_epsilon`] with the exact
/// upstream epsilon tensor rather than relying on backend-specific random-number generators.
#[derive(Debug)]
pub struct DiagonalGaussian<B: Backend> {
    mean: Tensor<B, 4>,
    logvar: Tensor<B, 4>,
    deterministic: bool,
}

impl<B: Backend> Clone for DiagonalGaussian<B> {
    fn clone(&self) -> Self {
        Self {
            mean: self.mean.clone(),
            logvar: self.logvar.clone(),
            deterministic: self.deterministic,
        }
    }
}

impl<B: Backend> DiagonalGaussian<B> {
    /// Split `[batch, 2 * latent_channels, height, width]` moments into mean and log-variance.
    pub fn from_moments(moments: Tensor<B, 4>) -> Self {
        Self::from_moments_deterministic(moments, false)
    }

    /// Construct a posterior, optionally forcing the distribution to its mean.
    pub fn from_moments_deterministic(moments: Tensor<B, 4>, deterministic: bool) -> Self {
        let [batch, channels, height, width] = moments.dims();
        assert!(
            channels > 0 && channels.is_multiple_of(2),
            "AutoencoderKL moments require a non-zero even channel count, got {channels}"
        );
        let latent_channels = channels / 2;
        let mean = moments
            .clone()
            .slice([0..batch, 0..latent_channels, 0..height, 0..width]);
        let logvar = moments
            .slice([0..batch, latent_channels..channels, 0..height, 0..width])
            .clamp(-30.0, 20.0);
        Self {
            mean,
            logvar,
            deterministic,
        }
    }

    pub fn mean(&self) -> Tensor<B, 4> {
        self.mean.clone()
    }

    pub fn logvar(&self) -> Tensor<B, 4> {
        self.logvar.clone()
    }

    pub fn variance(&self) -> Tensor<B, 4> {
        if self.deterministic {
            Tensor::<B, 4>::zeros(self.mean.dims(), &self.mean.device()).cast(self.float_dtype())
        } else {
            self.logvar.clone().exp()
        }
    }

    pub fn std(&self) -> Tensor<B, 4> {
        if self.deterministic {
            Tensor::<B, 4>::zeros(self.mean.dims(), &self.mean.device()).cast(self.float_dtype())
        } else {
            self.logvar.clone().mul_scalar(0.5).exp()
        }
    }

    pub fn is_deterministic(&self) -> bool {
        self.deterministic
    }

    /// Return the posterior mode.
    pub fn mode(&self) -> Tensor<B, 4> {
        self.mean()
    }

    /// Reparameterize with an explicit standard-normal epsilon tensor.
    ///
    /// Supplying epsilon explicitly is the portable numerical-correctness contract. Its shape
    /// must exactly match the latent mean.
    pub fn sample_with_epsilon(&self, epsilon: Tensor<B, 4>) -> Tensor<B, 4> {
        assert_eq!(
            epsilon.dims(),
            self.mean.dims(),
            "AutoencoderKL epsilon shape must match posterior mean"
        );
        if self.deterministic {
            self.mean()
        } else {
            self.mean() + self.std() * epsilon
        }
    }

    /// Draw backend-local standard-normal epsilon and sample the posterior.
    ///
    /// This is convenient but is not a cross-runtime parity surface; use
    /// [`Self::sample_with_epsilon`] for reference comparisons.
    pub fn sample_random(&self) -> Tensor<B, 4> {
        let epsilon = Tensor::<B, 4>::random(
            self.mean.dims(),
            Distribution::Normal(0.0, 1.0),
            &self.mean.device(),
        )
        .cast(self.float_dtype());
        self.sample_with_epsilon(epsilon)
    }

    /// KL divergence against a unit normal, reduced over channel and spatial dimensions.
    pub fn kl_standard_normal(&self) -> Tensor<B, 1> {
        let [batch, _channels, _height, _width] = self.mean.dims();
        if self.deterministic {
            return Tensor::<B, 1>::zeros([batch], &self.mean.device()).cast(self.float_dtype());
        }
        (self.mean.clone().square() + self.variance() - self.logvar() - 1.0)
            .mul_scalar(0.5)
            .sum_dim(3)
            .sum_dim(2)
            .sum_dim(1)
            .reshape([batch])
    }

    fn float_dtype(&self) -> FloatDType {
        self.mean.dtype().into()
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;

    use super::*;

    type TestBackend = burn::backend::NdArray<f32>;

    fn values<const D: usize>(tensor: Tensor<TestBackend, D>) -> Vec<f32> {
        tensor.to_data().to_vec::<f32>().expect("tensor values")
    }

    #[test]
    fn explicit_epsilon_correctness() {
        let device = Default::default();
        let moments = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(vec![1.0, -2.0, 0.0, 0.0], [1, 4, 1, 1]),
            &device,
        );
        let epsilon = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(vec![0.25, -0.5], [1, 2, 1, 1]),
            &device,
        );
        let posterior = DiagonalGaussian::from_moments(moments);
        let actual = values(posterior.sample_with_epsilon(epsilon));
        assert_eq!(actual, vec![1.25, -2.5]);
    }

    #[test]
    fn logvar_clamp_correctness() {
        let device = Default::default();
        let moments = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(vec![0.0, 0.0, -100.0, 100.0], [1, 4, 1, 1]),
            &device,
        );
        let posterior = DiagonalGaussian::from_moments(moments);
        assert_eq!(values(posterior.logvar()), vec![-30.0, 20.0]);
    }

    #[test]
    fn zero_epsilon_matches_mode_correctness() {
        let device = Default::default();
        let moments =
            Tensor::<TestBackend, 4>::random([1, 8, 2, 2], Distribution::Default, &device);
        let posterior = DiagonalGaussian::from_moments(moments);
        let epsilon = Tensor::zeros([1, 4, 2, 2], &device);
        assert_eq!(
            values(posterior.sample_with_epsilon(epsilon)),
            values(posterior.mode())
        );
    }

    #[test]
    fn standard_normal_kl_is_zero_correctness() {
        let device = Default::default();
        let moments = Tensor::<TestBackend, 4>::zeros([2, 8, 2, 2], &device);
        let kl = values(DiagonalGaussian::from_moments(moments).kl_standard_normal());
        assert_eq!(kl, vec![0.0, 0.0]);
    }
}
