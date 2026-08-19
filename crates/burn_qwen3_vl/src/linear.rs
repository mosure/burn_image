//! Checkpoint-compatible linear projection used by Qwen3-VL.
//!
//! Qwen checkpoints store projection weights as `[d_output, d_input]`, while Burn's linear
//! primitive consumes `[d_input, d_output]`. On native targets this module is an exact alias for
//! Burn's [`burn::nn::Linear`] configured with [`burn::nn::LinearLayout::Col`]. On wasm32 the
//! equivalent parameter mappers omit `Backend::sync`: browser synchronization is owned by the
//! asynchronous stage executor, and a blocking sync cannot be implemented without wasm threads.

use burn::{
    config::Config,
    module::Initializer,
    tensor::{DType, Shape, Tensor, backend::Backend},
};

#[cfg(target_arch = "wasm32")]
use burn::module::{Module, Param};

/// Qwen linear projection with a raw checkpoint weight shape of `[d_output, d_input]`.
///
/// Native builds use Burn's `Linear` type verbatim. The wasm32 implementation has the same
/// public `weight` and `bias` fields, record paths, lazy shape, initialized shape, and forward
/// primitive, but its transpose mappers do not perform a blocking backend synchronization.
#[cfg(not(target_arch = "wasm32"))]
pub type QwenLinear<B> = burn::nn::Linear<B>;

/// Qwen linear projection with non-blocking checkpoint transpose mappers for wasm32.
#[cfg(target_arch = "wasm32")]
#[derive(Module, Debug)]
pub struct QwenLinear<B: Backend> {
    /// Internal weight has shape `[d_input, d_output]`; snapshots use `[d_output, d_input]`.
    pub weight: Param<Tensor<B, 2>>,
    /// Optional bias of shape `[d_output]`.
    pub bias: Option<Param<Tensor<B, 1>>>,
}

/// Configuration for a checkpoint-compatible [`QwenLinear`] projection.
#[derive(Config, Debug)]
pub struct QwenLinearConfig {
    /// Number of input features.
    pub d_input: usize,
    /// Number of output features.
    pub d_output: usize,
    /// Whether to add a bias.
    #[config(default = true)]
    pub bias: bool,
    /// Parameter initializer. This matches Burn's linear default exactly.
    #[config(
        default = "Initializer::KaimingUniform { gain: 1.0 / 3.0_f64.sqrt(), fan_out_only: false }"
    )]
    pub initializer: Initializer,
}

impl QwenLinearConfig {
    /// Initialize a Qwen projection in checkpoint-compatible column layout.
    pub fn init<B: Backend>(&self, device: &B::Device) -> QwenLinear<B> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            burn::nn::LinearConfig::new(self.d_input, self.d_output)
                .with_bias(self.bias)
                .with_initializer(self.initializer.clone())
                .with_layout(burn::nn::LinearLayout::Col)
                .init(device)
        }

        #[cfg(target_arch = "wasm32")]
        {
            // Keep the uninitialized shape in saved Qwen/PyTorch layout. The three mappers are
            // intentionally the same transposes as Burn's Col layout without its blocking
            // B::sync calls, which panic in a browser without wasm threads.
            let weight = self
                .initializer
                .init_with(
                    [self.d_output, self.d_input],
                    Some(self.d_output),
                    Some(self.d_input),
                    device,
                )
                .save_mapper(|tensor| tensor.transpose())
                .load_mapper(|tensor| tensor.transpose())
                .init_mapper(|tensor| tensor.transpose());
            let bias = self.bias.then(|| {
                self.initializer.init_with(
                    [self.d_output],
                    Some(self.d_input),
                    Some(self.d_output),
                    device,
                )
            });
            QwenLinear { weight, bias }
        }
    }
}

#[cfg(target_arch = "wasm32")]
impl<B: Backend> QwenLinear<B> {
    /// Apply `output = input * weight + bias` without dequantizing a packed QFloat weight.
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        qwen_linear_forward(self, input)
    }
}

/// Apply a Qwen projection while preserving packed quantized weight storage through matmul.
///
/// Burn's released linear helper requests an ordinary floating primitive and therefore widens a
/// QFloat parameter before dispatch. Ordinary F16/F32 modules retain the released path exactly;
/// only a quantized weight takes the mixed `Float x QFloat` backend operation.
pub fn qwen_linear_forward<B: Backend, const D: usize>(
    linear: &QwenLinear<B>,
    input: Tensor<B, D>,
) -> Tensor<B, D> {
    if !matches!(linear.weight.val().dtype(), DType::QFloat(_)) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            return linear.forward(input);
        }
        #[cfg(target_arch = "wasm32")]
        {
            return burn::tensor::module::linear(
                input,
                linear.weight.val(),
                linear.bias.as_ref().map(|bias| bias.val()),
            );
        }
    }

    assert!(D >= 2, "Qwen linear projections require rank >= 2");
    let [input_width, output_width] = linear.weight.dims();
    let mut weight_shape = vec![1; D];
    weight_shape[D - 2] = input_width;
    weight_shape[D - 1] = output_width;
    let output = input.matmul(linear.weight.val().reshape(Shape::from(weight_shape)));
    match &linear.bias {
        Some(bias) => {
            let mut bias_shape = vec![1; D];
            bias_shape[D - 1] = output_width;
            output + bias.val().reshape(Shape::from(bias_shape))
        }
        None => output,
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    #[cfg(feature = "import")]
    use burn::{
        module::ParamId,
        tensor::{Tensor, TensorData},
    };
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn native_type_and_lazy_checkpoint_shape_correctness() {
        fn accepts_burn_linear<B: Backend>(_: &burn::nn::Linear<B>) {}

        let device = Default::default();
        let projection = QwenLinearConfig::new(2, 3)
            .with_bias(false)
            .init::<TestBackend>(&device);

        accepts_burn_linear(&projection);
        assert_eq!(projection.weight.lazy_shape().dims::<2>(), [3, 2]);
        assert_eq!(projection.weight.val().dims(), [2, 3]);
    }

    #[cfg(feature = "import")]
    #[test]
    fn raw_checkpoint_layout_and_forward_value_correctness() {
        use burn_store::{ModuleSnapshot, TensorSnapshot};

        let device = Default::default();
        let mut projection = QwenLinearConfig::new(2, 3)
            .with_bias(false)
            .init::<TestBackend>(&device);
        let raw = TensorSnapshot::from_data(
            TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]),
            vec!["weight".into()],
            Vec::new(),
            ParamId::new(),
        );

        let result = projection.apply(vec![raw], None, None, false);
        assert_eq!(result.applied, ["weight"]);
        assert!(result.missing.is_empty());
        assert!(result.unused.is_empty());
        assert!(result.errors.is_empty());
        assert_eq!(projection.weight.val().dims(), [2, 3]);

        let output = projection
            .forward(Tensor::<TestBackend, 2>::from_data(
                TensorData::new(vec![10.0_f32, 20.0], [1, 2]),
                &device,
            ))
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(output, vec![50.0, 110.0, 170.0]);

        let saved = projection.collect(None, None, false);
        assert_eq!(saved.len(), 1);
        assert_eq!(saved[0].full_path(), "weight");
        assert_eq!(saved[0].shape.dims::<2>(), [3, 2]);
        assert_eq!(
            (saved[0].clone_data_fn())()
                .unwrap()
                .to_vec::<f32>()
                .unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }
}
