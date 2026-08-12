use burn::{
    nn,
    prelude::*,
    tensor::{DType, activation::silu},
};

/// Numerical policy for RMS normalization inside the Boogu denoiser.
///
/// The default retains Burn's released whole-input F32 normalization. The mixed-storage policy is
/// an opt-in diagnostic policy for native F16 execution and must pass the real-artifact parity gate
/// before it can become part of a released runtime policy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DenoiserRmsNormPolicy {
    /// Cast the complete activation to F32 for the RMS reduction, matching Burn's `RmsNorm`.
    #[default]
    StrictF32,
    /// Keep full-size activations in F16 and cast only the reduced mean square to F32.
    F16StorageF32Accum,
}

/// Compute the affine-free RMS-normalized value shared by norms over the same tensor.
///
/// This is exactly the input-dependent prefix of Burn's [`nn::RmsNorm::forward`]. Keeping it
/// separate lets adjacent adaptive norms reuse one F32 reduction while retaining their distinct
/// learned gamma and modulation projections.
pub(crate) fn rms_normalized<B: Backend, const D: usize>(
    input: Tensor<B, D>,
    epsilon: f64,
) -> Tensor<B, D> {
    let dtype = input.dtype();
    let rms = (input.clone().cast(DType::F32).square().mean_dim(D - 1) + epsilon).sqrt();
    input / rms.cast(dtype)
}

/// RMS normalization that retains F16 storage without allowing a finite F16 square to overflow.
///
/// Scaling by 2^-9 bounds every finite F16 value to 127.9375, so its square stays finite. CubeCL's
/// `mean_dim` reduces in F32; only its reduced result is explicitly stored in F32 for epsilon,
/// square root, and reciprocal. Splitting the reciprocal by 2^5 keeps the factor cast back to F16
/// finite for zero variance at the denoiser's epsilon. The power-of-two scaling itself is exact for
/// normal F16 values, apart from unavoidable underflow of values too small to survive F16 storage.
fn rms_normalized_f16_storage_f32_accum<B: Backend, const D: usize>(
    input: Tensor<B, D>,
    epsilon: f64,
) -> Tensor<B, D> {
    const STORAGE_SCALE: f64 = 1.0 / 512.0;
    const NORMALIZATION_BOOST: f64 = 32.0;

    let dtype = input.dtype();
    if dtype == DType::F32 {
        return rms_normalized(input, epsilon);
    }
    assert_eq!(
        dtype,
        DType::F16,
        "F16-storage RMSNorm requires F16 or F32 input"
    );

    let scaled = input.mul_scalar(STORAGE_SCALE);
    let mean_square_scaled = scaled.clone().square().mean_dim(D - 1);
    let inverse_rms_split = mean_square_scaled
        .cast(DType::F32)
        .add_scalar(epsilon * STORAGE_SCALE * STORAGE_SCALE)
        .sqrt()
        .recip()
        .div_scalar(NORMALIZATION_BOOST)
        .cast(dtype);
    scaled.mul_scalar(NORMALIZATION_BOOST) * inverse_rms_split
}

pub(crate) fn rms_normalized_with_policy<B: Backend, const D: usize>(
    input: Tensor<B, D>,
    epsilon: f64,
    policy: DenoiserRmsNormPolicy,
) -> Tensor<B, D> {
    match policy {
        DenoiserRmsNormPolicy::StrictF32 => rms_normalized(input, epsilon),
        DenoiserRmsNormPolicy::F16StorageF32Accum => {
            rms_normalized_f16_storage_f32_accum(input, epsilon)
        }
    }
}

pub(crate) fn rms_norm_with_policy<B: Backend, const D: usize>(
    norm: &nn::RmsNorm<B>,
    input: Tensor<B, D>,
    policy: DenoiserRmsNormPolicy,
) -> Tensor<B, D> {
    match policy {
        DenoiserRmsNormPolicy::StrictF32 => norm.forward(input),
        DenoiserRmsNormPolicy::F16StorageF32Accum => {
            rms_normalized_f16_storage_f32_accum(input, norm.epsilon) * norm.gamma.val().unsqueeze()
        }
    }
}

/// Adaptive RMS normalization used by modulated Boogu blocks.
#[derive(Module, Debug)]
pub struct RmsNormZero<B: Backend> {
    /// Projection producing attention/MLP scale and gates.
    pub linear: nn::Linear<B>,
    /// RMS normalization over model width.
    pub norm: nn::RmsNorm<B>,
}

impl<B: Backend> RmsNormZero<B> {
    /// Create the modulation module.
    pub fn new(width: usize, conditioning_width: usize, epsilon: f64, device: &B::Device) -> Self {
        Self {
            linear: nn::LinearConfig::new(conditioning_width, 4 * width).init(device),
            norm: nn::RmsNormConfig::new(width)
                .with_epsilon(epsilon)
                .init(device),
        }
    }

    /// Return normalized tokens and attention/MLP modulation vectors.
    pub fn forward(
        &self,
        tokens: Tensor<B, 3>,
        conditioning: Tensor<B, 2>,
    ) -> (Tensor<B, 3>, Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        self.forward_with_policy(tokens, conditioning, DenoiserRmsNormPolicy::StrictF32)
    }

    pub(crate) fn forward_with_policy(
        &self,
        tokens: Tensor<B, 3>,
        conditioning: Tensor<B, 2>,
        policy: DenoiserRmsNormPolicy,
    ) -> (Tensor<B, 3>, Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let normalized = rms_normalized_with_policy(tokens, self.norm.epsilon, policy);
        self.forward_from_rms_normalized(normalized, conditioning)
    }

    pub(crate) fn forward_from_rms_normalized(
        &self,
        normalized: Tensor<B, 3>,
        conditioning: Tensor<B, 2>,
    ) -> (Tensor<B, 3>, Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let width = normalized.dims()[2];
        let modulation = self.linear.forward(silu(conditioning));
        let chunks = modulation.split_with_sizes(vec![width, width, width, width], 1);
        let scale_msa = chunks[0].clone();
        let gate_msa = chunks[1].clone();
        let scale_mlp = chunks[2].clone();
        let gate_mlp = chunks[3].clone();
        let normalized =
            (normalized * self.norm.gamma.val().unsqueeze()) * (scale_msa.unsqueeze_dim(1) + 1.0);
        (normalized, gate_msa, scale_mlp, gate_mlp)
    }
}

/// Affine-free layer normalization over the last tensor dimension.
pub(crate) fn layer_norm_no_affine<B: Backend>(x: Tensor<B, 3>, epsilon: f64) -> Tensor<B, 3> {
    let dtype = x.dtype();
    let x_f32 = x.cast(burn::tensor::DType::F32);
    let centered = x_f32.clone() - x_f32.mean_dim(2);
    let inverse_std = (centered.clone().square().mean_dim(2) + epsilon)
        .sqrt()
        .recip();
    (centered * inverse_std).cast(dtype)
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn shared_rms_prefix_matches_independent_adaptive_norm_correctness() {
        let device = Default::default();
        let module = RmsNormZero::<TestBackend>::new(8, 6, 1.0e-5, &device);
        let tokens = Tensor::from_data(
            TensorData::new(
                (0..80)
                    .map(|index| ((index * 17 + 5) % 41) as f32 / 13.0 - 1.5)
                    .collect(),
                [2, 5, 8],
            ),
            &device,
        );
        let conditioning = Tensor::from_data(
            TensorData::new(
                (0..12)
                    .map(|index| ((index * 11 + 3) % 19) as f32 / 7.0 - 1.0)
                    .collect(),
                [2, 6],
            ),
            &device,
        );

        let modulation = module.linear.forward(silu(conditioning.clone()));
        let chunks = modulation.split_with_sizes(vec![8, 8, 8, 8], 1);
        let expected =
            module.norm.forward(tokens.clone()) * (chunks[0].clone().unsqueeze_dim(1) + 1.0);
        let shared = rms_normalized(tokens, module.norm.epsilon);
        let (actual, gate, scale_mlp, gate_mlp) =
            module.forward_from_rms_normalized(shared, conditioning);

        assert_eq!(actual.into_data(), expected.into_data());
        assert_eq!(gate.into_data(), chunks[1].clone().into_data());
        assert_eq!(scale_mlp.into_data(), chunks[2].clone().into_data());
        assert_eq!(gate_mlp.into_data(), chunks[3].clone().into_data());
    }

    #[test]
    fn f32_mixed_storage_policy_is_strict_parity() {
        let device = Default::default();
        let input = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                (0..80)
                    .map(|index| ((index * 17 + 5) % 41) as f32 / 13.0 - 1.5)
                    .collect(),
                [2, 5, 8],
            ),
            &device,
        );
        let strict =
            rms_normalized_with_policy(input.clone(), 1.0e-5, DenoiserRmsNormPolicy::StrictF32);
        let mixed =
            rms_normalized_with_policy(input, 1.0e-5, DenoiserRmsNormPolicy::F16StorageF32Accum);

        assert_eq!(strict.into_data(), mixed.into_data());
    }

    #[test]
    #[cfg(feature = "flex")]
    fn mixed_storage_rms_avoids_f16_square_overflow_correctness() {
        type MixedPrecisionTestBackend = burn::backend::Flex;

        let device = Default::default();
        let input = Tensor::<MixedPrecisionTestBackend, 3>::from_data(
            TensorData::new(
                vec![
                    65504.0_f32,
                    -65504.0,
                    32752.0,
                    -32752.0,
                    16.0,
                    -16.0,
                    0.0,
                    0.0,
                ],
                [1, 1, 8],
            ),
            &device,
        )
        .cast(DType::F16);
        let strict = rms_normalized(input.clone(), 1.0e-5);
        let mixed = rms_normalized_f16_storage_f32_accum(input, 1.0e-5);

        assert!(mixed.clone().is_finite().all().into_scalar());
        let max_abs = (strict.cast(DType::F32) - mixed.cast(DType::F32))
            .abs()
            .max()
            .into_scalar();
        assert!(max_abs <= 2.0e-3, "mixed RMSNorm max_abs={max_abs}");
    }

    #[test]
    #[cfg(feature = "flex")]
    #[should_panic(expected = "F16-storage RMSNorm requires F16 or F32 input")]
    fn mixed_storage_rms_rejects_bf16_correctness() {
        type MixedPrecisionTestBackend = burn::backend::Flex;

        let device = Default::default();
        let input =
            Tensor::<MixedPrecisionTestBackend, 3>::ones([1, 2, 8], &device).cast(DType::BF16);
        let _ = rms_normalized_f16_storage_f32_accum(input, 1.0e-5);
    }
}
