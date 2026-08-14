use burn::{
    nn,
    prelude::*,
    tensor::{DType, Shape},
};

/// Apply a Boogu linear projection without losing a quantized weight primitive.
///
/// Burn's [`nn::Linear::forward`] lowers through `tensor::module::linear`, which requests a
/// floating primitive for every parameter and therefore dequantizes a `QFloat` weight before the
/// backend sees the operation. Calling [`Tensor::matmul`] directly keeps the mixed
/// `Float x QFloat` operands intact so quantization-aware backends dispatch `q_matmul` instead.
/// Ordinary floating-point modules retain the released `nn::Linear` path exactly.
pub(crate) fn linear_forward<B: Backend, const D: usize>(
    linear: &nn::Linear<B>,
    input: Tensor<B, D>,
) -> Tensor<B, D> {
    if !matches!(linear.weight.val().dtype(), DType::QFloat(_)) {
        return linear.forward(input);
    }

    assert!(D >= 2, "Boogu linear projections require rank >= 2");
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

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use super::*;
    use burn::{
        backend::NdArray,
        module::Param,
        tensor::{TensorData, quantization::*},
    };

    type TestBackend = NdArray<f32>;

    #[test]
    fn quantized_linear_direct_matmul_matches_released_linear_semantics_correctness() {
        let device = Default::default();
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_level(QuantLevel::block([32]))
            .with_param(QuantParam::F32)
            .with_store(QuantStore::PackedU32(0));
        let weight = Tensor::<TestBackend, 2>::from_data(
            TensorData::quantized(
                (-16_i8..16).collect::<Vec<_>>(),
                [4, 8],
                scheme,
                &[0.03125_f32],
            ),
            &device,
        );
        let bias = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(
                vec![0.25_f32, -0.5, 0.75, -1.0, 1.25, -1.5, 1.75, -2.0],
                [8],
            ),
            &device,
        );
        let linear = nn::Linear {
            weight: Param::from_tensor(weight),
            bias: Some(Param::from_tensor(bias)),
        };
        let input = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                vec![
                    0.5_f32, -0.25, 0.75, 1.0, -1.0, 0.125, 0.25, -0.5, 0.375, 0.625, -0.75, 0.875,
                    -0.125, -0.375, 0.5, 0.25,
                ],
                [2, 2, 4],
            ),
            &device,
        );

        assert!(matches!(linear.weight.val().dtype(), DType::QFloat(_)));
        let expected = linear.forward(input.clone()).into_data();
        let actual = linear_forward(&linear, input).into_data();

        actual.assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
        assert!(matches!(linear.weight.val().dtype(), DType::QFloat(_)));
    }

    #[test]
    fn quantized_linear_source_contract_rejects_dequantizing_helper_correctness() {
        let source = include_str!("linear.rs");
        let helper = source
            .split("pub(crate) fn linear_forward")
            .nth(1)
            .expect("linear helper must remain present")
            .split("#[cfg(all(test")
            .next()
            .expect("linear helper must end before its tests");

        assert!(helper.contains("input.matmul("));
        assert!(!helper.contains(".dequantize("));
        assert!(!helper.contains("tensor::module::linear"));
    }
}
