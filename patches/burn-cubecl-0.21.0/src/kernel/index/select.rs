use crate::{CubeRuntime, kernel::utils::address_type, tensor::CubeTensor};
use crate::{kernel::utils::shape_divmod, ops::numeric::empty_device_dtype};
use burn_backend::TensorMetadata;
use cubecl::{CubeDim, calculate_cube_count_elemwise, std::tensor::layout::linear::LinearView};
use cubecl::{prelude::*, std::FastDivmod};

#[cube(launch_unchecked, address_type = "dynamic")]
fn select_kernel<T: Numeric, I: Numeric>(
    input: &Tensor<T>,
    indices: &LinearView<I>,
    output: &mut LinearView<T, ReadWrite>,
    out_shape: Sequence<FastDivmod<usize>>,
    dim: usize,
    #[define(T, I)] _dtypes: [StorageType; 2],
) {
    if ABSOLUTE_POS >= output.shape() {
        terminate!();
    }

    let rank = out_shape.len().comptime();

    let mut offset = ABSOLUTE_POS;
    let mut offset_input = 0;

    #[unroll]
    for i in 0..rank {
        let i = rank - i - 1;
        let (rem, offset_local) = out_shape[i].div_mod(offset);
        offset = rem;

        let offset_local = cubecl::prelude::select(
            i == dim,
            usize::cast_from(indices[offset_local]),
            offset_local,
        );

        offset_input += offset_local * input.stride(i);
    }

    output[ABSOLUTE_POS] = input[offset_input];
}

pub(crate) fn select<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    dim: usize,
    indices: CubeTensor<R>,
) -> CubeTensor<R> {
    let mut shape_output = tensor.shape();
    shape_output[dim] = indices.meta.shape()[0];
    let total_elem = shape_output.num_elements();

    let output = empty_device_dtype(
        tensor.client.clone(),
        tensor.device.clone(),
        shape_output,
        tensor.dtype,
    );

    let working_units = total_elem;
    let cube_dim = CubeDim::new(&indices.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&indices.client, working_units, cube_dim);

    let (tensor_dtype, indices_dtype) = (tensor.dtype, indices.dtype);

    unsafe {
        select_kernel::launch_unchecked(
            &output.client,
            cube_count,
            cube_dim,
            address_type!(tensor, indices, output),
            tensor.into_tensor_arg(),
            indices.into_linear_view(),
            output.clone().into_linear_view(),
            shape_divmod(&output),
            dim,
            [tensor_dtype.into(), indices_dtype.into()],
        )
    };
    output
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn quantized_select_rows_kernel<I: Numeric>(
    values: &Tensor<u32>,
    scales: &Tensor<f32>,
    indices: &Tensor<I>,
    output_values: &mut Tensor<u32>,
    output_scales: &mut Tensor<f32>,
    #[define(I)] _indices_dtype: StorageType,
) {
    if ABSOLUTE_POS < output_values.len() {
        let columns = output_values.shape(1);
        let output_row = ABSOLUTE_POS / columns;
        let column = ABSOLUTE_POS % columns;
        let source_row = usize::cast_from(indices[output_row]);
        output_values[ABSOLUTE_POS] =
            values[source_row * values.stride(0) + column * values.stride(1)];
    }
    if ABSOLUTE_POS < output_scales.len() {
        let columns = output_scales.shape(1);
        let output_row = ABSOLUTE_POS / columns;
        let column = ABSOLUTE_POS % columns;
        let source_row = usize::cast_from(indices[output_row]);
        output_scales[ABSOLUTE_POS] =
            scales[source_row * scales.stride(0) + column * scales.stride(1)];
    }
}

/// Select rows from a packed block-quantized matrix without widening the source table.
pub(crate) fn select_quantized_rows<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    indices: CubeTensor<R>,
    output: CubeTensor<R>,
) -> CubeTensor<R> {
    let (values, scales) = tensor
        .quantized_handles()
        .expect("quantized row selection requires values and scales");
    let (output_values, output_scales) = output
        .quantized_handles()
        .expect("quantized row-selection output requires values and scales");
    assert_eq!(values.rank(), 2, "quantized row values must be a matrix");
    assert_eq!(scales.rank(), 2, "quantized row scales must be a matrix");
    assert_eq!(indices.rank(), 1, "quantized row indices must be a vector");
    assert!(
        values.is_contiguous(),
        "quantized row values must be contiguous"
    );
    assert!(
        scales.is_contiguous(),
        "quantized row scales must be contiguous"
    );
    assert!(
        output_values.is_contiguous() && output_scales.is_contiguous(),
        "quantized row-selection output must be contiguous"
    );
    assert_eq!(
        values.shape()[0],
        scales.shape()[0],
        "quantized value and scale row counts differ"
    );
    assert_eq!(
        output_values.shape()[0],
        indices.shape()[0],
        "quantized selected row count differs from the index count"
    );

    let work_items = output_values
        .meta
        .num_elements()
        .max(output_scales.meta.num_elements());
    let cube_dim = CubeDim::new(&output.client, work_items);
    let cube_count = calculate_cube_count_elemwise(&output.client, work_items, cube_dim);
    let indices_dtype = indices.dtype;
    unsafe {
        quantized_select_rows_kernel::launch_unchecked(
            &output.client,
            cube_count,
            cube_dim,
            crate::kernel::utils::address_type!(
                values,
                scales,
                indices,
                output_values,
                output_scales
            ),
            values.into_tensor_arg(),
            scales.into_tensor_arg(),
            indices.into_tensor_arg(),
            output_values.into_tensor_arg(),
            output_scales.into_tensor_arg(),
            indices_dtype.into(),
        )
    };
    output
}
