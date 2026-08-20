//! Browser-safe execution directly from IEEE binary16 weight storage.
//!
//! WebGPU exposes `shader-f16` as an optional feature.  A backend without that feature can still
//! retain immutable F16 weight bytes in storage buffers and bind those bytes as `array<u32>`.
//! These kernels unpack one half value at the point of use and accumulate in F32, avoiding a
//! second, model-sized F32 weight allocation.

use crate::{
    CubeRuntime,
    kernel::{into_contiguous_aligned, utils::address_type},
    ops::numeric::empty_device_contiguous_dtype,
    tensor::CubeTensor,
};
use burn_backend::{
    DType, Shape, TensorMetadata,
    ops::{ConvOptions, conv::calculate_conv_output_sizes},
};
use cubecl::{
    CubeCount, CubeDim, calculate_cube_count_elemwise,
    ir::{ElemType, FloatKind},
    prelude::*,
};

const WORKGROUP_AXIS: usize = 16;
const WORKGROUP_AXIS_U32: u32 = WORKGROUP_AXIS as u32;
const OUTPUT_ROWS_PER_UNIT: usize = 2;
const OUTPUT_ROW_TILE: usize = WORKGROUP_AXIS * OUTPUT_ROWS_PER_UNIT;
const OUTPUT_COLUMNS_PER_UNIT: usize = 4;
const OUTPUT_COLUMN_TILE: usize = WORKGROUP_AXIS * OUTPUT_COLUMNS_PER_UNIT;
const INPUT_SHARED_ELEMENTS: usize = OUTPUT_ROW_TILE * WORKGROUP_AXIS;
const WEIGHT_SHARED_ELEMENTS: usize = WORKGROUP_AXIS * OUTPUT_COLUMN_TILE;

#[derive(CubeLaunch, CubeType, Clone)]
struct PackedMatmulArgs {
    rhs_inner_stride: u32,
    rhs_column_stride: u32,
}

/// Whether this tensor needs integer-unpack F16 execution on its current device.
pub fn requires_packed_f16_unpack<R: CubeRuntime>(tensor: &CubeTensor<R>) -> bool {
    tensor.dtype == DType::F16
        && !tensor
            .client
            .properties()
            .supports_type(ElemType::Float(FloatKind::F16))
}

#[cube]
fn widen_f16_bits_to_f32(bits: u32) -> f32 {
    let sign = (bits & 0x8000u32) << 16u32;
    let exponent = (bits >> 10u32) & 0x1fu32;
    let mantissa = bits & 0x03ffu32;

    let widened_bits = if exponent == 0u32 {
        if mantissa == 0u32 {
            sign
        } else {
            let shift = mantissa.leading_zeros() - 21u32;
            let widened_exponent = 113u32 - shift;
            let widened_mantissa = ((mantissa << shift) & 0x03ffu32) << 13u32;
            sign | (widened_exponent << 23u32) | widened_mantissa
        }
    } else if exponent == 0x1fu32 {
        if mantissa == 0u32 {
            sign | 0x7f80_0000u32
        } else {
            sign | 0x7fc0_0000u32 | (mantissa << 13u32)
        }
    } else {
        sign | ((exponent + 112u32) << 23u32) | (mantissa << 13u32)
    };

    f32::reinterpret(widened_bits)
}

#[cube]
fn load_packed_f16(packed: &Tensor<u32>, logical_index: usize) -> f32 {
    let word = packed[logical_index / 2];
    let bits = if logical_index % 2 == 0 {
        word & 0xffffu32
    } else {
        word >> 16u32
    };
    widen_f16_bits_to_f32(bits)
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn packed_f16_rhs_matmul_kernel(
    lhs: &Tensor<f32>,
    packed_rhs: &Tensor<u32>,
    output: &mut Tensor<f32>,
    args: PackedMatmulArgs,
) {
    let unit_x = UNIT_POS_X as usize;
    let unit_y = UNIT_POS_Y as usize;
    let row_start = CUBE_POS_Y as usize * OUTPUT_ROW_TILE + unit_y;
    let column_start = CUBE_POS_X as usize * OUTPUT_COLUMN_TILE + unit_x;
    let inner = lhs.shape(lhs.rank() - 1);
    let cols = output.shape(output.rank() - 1);
    let rows = output.len() / cols;
    let tile_count = (inner + WORKGROUP_AXIS - 1) / WORKGROUP_AXIS;
    let mut lhs_tile = SharedMemory::<f32>::new(INPUT_SHARED_ELEMENTS);
    let mut rhs_tile = SharedMemory::<f32>::new(WEIGHT_SHARED_ELEMENTS);
    let mut sums = Array::<f32>::new(OUTPUT_ROWS_PER_UNIT * OUTPUT_COLUMNS_PER_UNIT);
    for lane in 0..OUTPUT_ROWS_PER_UNIT * OUTPUT_COLUMNS_PER_UNIT {
        sums[lane] = 0.0f32;
    }

    for tile in 0..tile_count {
        let lhs_inner = tile * WORKGROUP_AXIS + unit_x;
        for row_lane in 0..OUTPUT_ROWS_PER_UNIT {
            let row = row_start + row_lane * WORKGROUP_AXIS;
            let mut lhs_value = 0.0f32;
            if row < rows && lhs_inner < inner {
                lhs_value = lhs[row * inner + lhs_inner];
            }
            lhs_tile[(unit_y + row_lane * WORKGROUP_AXIS) * WORKGROUP_AXIS + unit_x] = lhs_value;
        }

        // Transposed linear weights are physically row-major [out, in]. Let X walk the
        // contiguous inner dimension while Y selects the output column; the shared tile then
        // presents the conventional [inner, column] view to every output thread. This avoids
        // strided global reads across a subgroup without materializing a transposed weight.
        let rhs_inner = tile * WORKGROUP_AXIS + unit_x;
        for column_lane in 0..OUTPUT_COLUMNS_PER_UNIT {
            let column =
                CUBE_POS_X as usize * OUTPUT_COLUMN_TILE + unit_y + column_lane * WORKGROUP_AXIS;
            let mut rhs_value = 0.0f32;
            if rhs_inner < inner && column < cols {
                let physical_index = rhs_inner * args.rhs_inner_stride as usize
                    + column * args.rhs_column_stride as usize;
                rhs_value = load_packed_f16(packed_rhs, physical_index);
            }
            rhs_tile[unit_x * OUTPUT_COLUMN_TILE + unit_y + column_lane * WORKGROUP_AXIS] =
                rhs_value;
        }
        sync_cube();

        for offset in 0..WORKGROUP_AXIS {
            for row_lane in 0..OUTPUT_ROWS_PER_UNIT {
                let lhs_value =
                    lhs_tile[(unit_y + row_lane * WORKGROUP_AXIS) * WORKGROUP_AXIS + offset];
                for column_lane in 0..OUTPUT_COLUMNS_PER_UNIT {
                    sums[row_lane * OUTPUT_COLUMNS_PER_UNIT + column_lane] += lhs_value
                        * rhs_tile
                            [offset * OUTPUT_COLUMN_TILE + unit_x + column_lane * WORKGROUP_AXIS];
                }
            }
        }
        sync_cube();
    }

    for row_lane in 0..OUTPUT_ROWS_PER_UNIT {
        let row = row_start + row_lane * WORKGROUP_AXIS;
        if row < rows {
            for column_lane in 0..OUTPUT_COLUMNS_PER_UNIT {
                let column = column_start + column_lane * WORKGROUP_AXIS;
                if column < cols {
                    output[row * cols + column] =
                        sums[row_lane * OUTPUT_COLUMNS_PER_UNIT + column_lane];
                }
            }
        }
    }
}

/// Multiply a contiguous F32 activation by a broadcast F16 right-hand weight.
///
/// The two matrix strides are explicit so a checkpoint-compatible transpose can remain a
/// zero-copy view. No F16 permutation kernel or dense F32 materialization is needed during load.
pub fn packed_f16_rhs_matmul<R: CubeRuntime>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
) -> CubeTensor<R> {
    assert_eq!(
        lhs.dtype,
        DType::F32,
        "packed-F16 matmul requires F32 activations"
    );
    assert_eq!(
        rhs.dtype,
        DType::F16,
        "packed-F16 matmul requires F16 weights"
    );
    assert_eq!(lhs.rank(), rhs.rank(), "packed-F16 matmul ranks differ");
    assert!(lhs.rank() >= 2, "packed-F16 matmul requires rank >= 2");
    assert!(
        rhs.shape()[..rhs.rank() - 2].iter().all(|dim| *dim == 1),
        "packed-F16 matmul supports one broadcast weight matrix"
    );

    let lhs = into_contiguous_aligned(lhs);
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    let rhs_strides = rhs.meta.strides();
    let inner = lhs_shape[lhs.rank() - 1];
    let rhs_inner = rhs_shape[rhs.rank() - 2];
    let cols = rhs_shape[rhs.rank() - 1];
    assert_eq!(
        inner, rhs_inner,
        "packed-F16 matmul inner dimensions differ"
    );
    let rhs_elements = inner
        .checked_mul(cols)
        .expect("packed-F16 matmul weight size overflowed");
    assert!(
        rhs_elements.is_multiple_of(2),
        "packed-F16 matmul requires a whole number of packed u32 words"
    );

    let mut output_shape = lhs_shape;
    let output_rank = output_shape.num_dims();
    output_shape[output_rank - 1] = cols;
    let output = empty_device_contiguous_dtype(
        lhs.client.clone(),
        lhs.device.clone(),
        output_shape,
        DType::F32,
    );
    let rows = output.meta.num_elements() / cols;
    let cubes_x = u32::try_from(cols.div_ceil(OUTPUT_COLUMN_TILE))
        .expect("packed-F16 matmul column dispatch exceeds u32");
    let cubes_y = u32::try_from(rows.div_ceil(OUTPUT_ROW_TILE))
        .expect("packed-F16 matmul row dispatch exceeds u32");
    let rhs_inner_stride = u32::try_from(rhs_strides[rhs.rank() - 2])
        .expect("packed-F16 matmul inner stride exceeds u32");
    let rhs_column_stride = u32::try_from(rhs_strides[rhs.rank() - 1])
        .expect("packed-F16 matmul column stride exceeds u32");
    assert!(
        cubes_x > 0 && cubes_y > 0,
        "packed-F16 matmul cannot be empty"
    );

    unsafe {
        packed_f16_rhs_matmul_kernel::launch_unchecked(
            &output.client,
            CubeCount::Static(cubes_x, cubes_y, 1),
            CubeDim::new_2d(WORKGROUP_AXIS_U32, WORKGROUP_AXIS_U32),
            address_type!(lhs, rhs, output),
            lhs.into_tensor_arg(),
            rhs.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            PackedMatmulArgsLaunch::new(rhs_inner_stride, rhs_column_stride),
        )
    };
    output
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn packed_f16_select_rows_kernel(
    packed: &Tensor<u32>,
    indices: &Tensor<i32>,
    output: &mut Tensor<f32>,
) {
    if ABSOLUTE_POS < output.len() {
        let cols = output.shape(output.rank() - 1);
        let output_row = ABSOLUTE_POS / cols;
        let col = ABSOLUTE_POS % cols;
        let source_row = indices[output_row] as usize;
        output[ABSOLUTE_POS] = load_packed_f16(packed, source_row * cols + col);
    }
}

/// Select F32 rows directly from one contiguous F16 table.
pub fn packed_f16_select_rows<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    dim: usize,
    indices: CubeTensor<R>,
) -> CubeTensor<R> {
    assert_eq!(
        tensor.dtype,
        DType::F16,
        "packed-F16 select requires F16 storage"
    );
    assert_eq!(tensor.rank(), 2, "packed-F16 select requires a matrix");
    assert_eq!(dim, 0, "packed-F16 select only supports row selection");
    assert_eq!(
        indices.dtype,
        DType::I32,
        "packed-F16 select requires I32 indices"
    );
    assert_eq!(
        indices.rank(),
        1,
        "packed-F16 select requires rank-one indices"
    );
    assert!(
        tensor.is_contiguous(),
        "packed-F16 select table must be contiguous"
    );
    let elements = tensor.meta.num_elements();
    assert!(
        elements.is_multiple_of(2),
        "packed-F16 select requires a whole number of packed u32 words"
    );
    let indices = into_contiguous_aligned(indices);
    let cols = tensor.shape()[1];
    let output = empty_device_contiguous_dtype(
        tensor.client.clone(),
        tensor.device.clone(),
        Shape::new([indices.shape()[0], cols]),
        DType::F32,
    );
    let work_items = output.meta.num_elements();
    let cube_dim = CubeDim::new(&output.client, work_items);
    let cube_count = calculate_cube_count_elemwise(&output.client, work_items, cube_dim);
    unsafe {
        packed_f16_select_rows_kernel::launch_unchecked(
            &output.client,
            cube_count,
            cube_dim,
            address_type!(tensor, indices, output),
            tensor.into_tensor_arg(),
            indices.into_tensor_arg(),
            output.clone().into_tensor_arg(),
        )
    };
    output
}

#[derive(CubeLaunch, CubeType, Clone)]
struct PackedConv2dArgs {
    stride_h: u32,
    stride_w: u32,
    dilation_h: u32,
    dilation_w: u32,
    padding_h: i32,
    padding_w: i32,
    row_tile_stride: u32,
}

#[cube]
fn packed_conv2d_input_value(
    input: &Tensor<f32>,
    row: usize,
    inner_index: usize,
    weight: &Tensor<u32>,
    output: &Tensor<f32>,
    args: &PackedConv2dArgs,
) -> f32 {
    let out_h = output.shape(2);
    let out_w = output.shape(3);
    let in_channels = input.shape(1);
    let in_h = input.shape(2);
    let in_w = input.shape(3);
    let kernel_h = weight.shape(2);
    let kernel_w = weight.shape(3);
    let kernel_area = kernel_h * kernel_w;
    let batch = row / (out_h * out_w);
    let spatial = row % (out_h * out_w);
    let out_y = spatial / out_w;
    let out_x = spatial % out_w;
    let in_channel = inner_index / kernel_area;
    let kernel_index = inner_index % kernel_area;
    let kernel_y = kernel_index / kernel_w;
    let kernel_x = kernel_index % kernel_w;
    let in_y = (out_y * args.stride_h as usize + kernel_y * args.dilation_h as usize) as i32
        - args.padding_h;
    let in_x = (out_x * args.stride_w as usize + kernel_x * args.dilation_w as usize) as i32
        - args.padding_w;
    let mut value = 0.0f32;
    if in_channel < in_channels
        && in_y >= 0
        && in_y < in_h as i32
        && in_x >= 0
        && in_x < in_w as i32
    {
        value = input
            [((batch * in_channels + in_channel) * in_h + in_y as usize) * in_w + in_x as usize];
    }
    value
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn packed_f16_conv2d_kernel(
    input: &Tensor<f32>,
    packed_weight: &Tensor<u32>,
    bias: ComptimeOption<Tensor<f32>>,
    output: &mut Tensor<f32>,
    args: PackedConv2dArgs,
) {
    let unit_x = UNIT_POS_X as usize;
    let unit_y = UNIT_POS_Y as usize;
    let row_tile = CUBE_POS_Y as usize + CUBE_POS_Z as usize * args.row_tile_stride as usize;
    let row = row_tile * WORKGROUP_AXIS + unit_y;
    let out_channel_start = CUBE_POS_X as usize * OUTPUT_COLUMN_TILE + unit_x;
    let out_channels = output.shape(1);
    let out_h = output.shape(2);
    let out_w = output.shape(3);
    let rows = output.shape(0) * out_h * out_w;
    let in_channels = input.shape(1);
    let kernel_h = packed_weight.shape(2);
    let kernel_w = packed_weight.shape(3);
    let inner = in_channels * kernel_h * kernel_w;
    let tile_count = (inner + WORKGROUP_AXIS - 1) / WORKGROUP_AXIS;
    let shared_index = unit_y * WORKGROUP_AXIS + unit_x;
    let mut input_tile = SharedMemory::<f32>::new(INPUT_SHARED_ELEMENTS);
    let mut weight_tile = SharedMemory::<f32>::new(WEIGHT_SHARED_ELEMENTS);
    let mut sums = Array::<f32>::new(OUTPUT_COLUMNS_PER_UNIT);
    for lane in 0..OUTPUT_COLUMNS_PER_UNIT {
        sums[lane] = 0.0f32;
    }

    for tile in 0..tile_count {
        let input_inner = tile * WORKGROUP_AXIS + unit_x;
        let mut input_value = 0.0f32;
        if row < rows && input_inner < inner {
            input_value =
                packed_conv2d_input_value(input, row, input_inner, packed_weight, output, &args);
        }
        input_tile[shared_index] = input_value;
        // OIHW is contiguous in the flattened input/kernel dimension. Cooperatively load one
        // contiguous inner run per output channel, then consume the transposed shared-memory
        // view. Adjacent subgroup lanes no longer jump by a complete convolution kernel.
        let weight_inner = tile * WORKGROUP_AXIS + unit_x;
        for column_lane in 0..OUTPUT_COLUMNS_PER_UNIT {
            let out_channel =
                CUBE_POS_X as usize * OUTPUT_COLUMN_TILE + unit_y + column_lane * WORKGROUP_AXIS;
            let mut weight_value = 0.0f32;
            if out_channel < out_channels && weight_inner < inner {
                weight_value = load_packed_f16(packed_weight, out_channel * inner + weight_inner);
            }
            weight_tile[unit_x * OUTPUT_COLUMN_TILE + unit_y + column_lane * WORKGROUP_AXIS] =
                weight_value;
        }
        sync_cube();

        for offset in 0..WORKGROUP_AXIS {
            let input_value = input_tile[unit_y * WORKGROUP_AXIS + offset];
            for column_lane in 0..OUTPUT_COLUMNS_PER_UNIT {
                sums[column_lane] += input_value
                    * weight_tile
                        [offset * OUTPUT_COLUMN_TILE + unit_x + column_lane * WORKGROUP_AXIS];
            }
        }
        sync_cube();
    }

    if row < rows {
        let batch = row / (out_h * out_w);
        let spatial = row % (out_h * out_w);
        let out_y = spatial / out_w;
        let out_x = spatial % out_w;
        for column_lane in 0..OUTPUT_COLUMNS_PER_UNIT {
            let out_channel = out_channel_start + column_lane * WORKGROUP_AXIS;
            if out_channel < out_channels {
                let output_index =
                    ((batch * out_channels + out_channel) * out_h + out_y) * out_w + out_x;
                let bias: ComptimeOption<f32> = bias.map(|values| values[out_channel]);
                output[output_index] =
                    sums[column_lane] + bias.unwrap_or_else(|| f32::new(0.0_f32));
            }
        }
    }
}

/// Apply one ordinary groups=1 NCHW convolution from F16 weight storage with F32 accumulation.
pub fn packed_f16_conv2d<R: CubeRuntime>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<2>,
) -> CubeTensor<R> {
    assert_eq!(
        input.dtype,
        DType::F32,
        "packed-F16 convolution requires F32 activations"
    );
    assert_eq!(
        weight.dtype,
        DType::F16,
        "packed-F16 convolution requires F16 weights"
    );
    assert_eq!(
        input.rank(),
        4,
        "packed-F16 convolution requires NCHW input"
    );
    assert_eq!(
        weight.rank(),
        4,
        "packed-F16 convolution requires OIHW weights"
    );
    assert_eq!(
        options.groups, 1,
        "packed-F16 convolution currently requires groups=1"
    );
    assert!(
        weight.is_contiguous(),
        "packed-F16 convolution weight must be contiguous"
    );
    if let Some(bias) = &bias {
        assert_eq!(
            bias.dtype,
            DType::F32,
            "packed-F16 convolution bias must be F32"
        );
    }
    let weight_elements = weight.meta.num_elements();
    assert!(
        weight_elements.is_multiple_of(2),
        "packed-F16 convolution requires a whole number of packed u32 words"
    );

    let input = into_contiguous_aligned(input);
    let [batch, _in_channels, in_h, in_w] = input.shape().dims();
    let [out_channels, _weight_in_channels, kernel_h, kernel_w] = weight.shape().dims();
    let out_size = calculate_conv_output_sizes(
        &[kernel_h, kernel_w],
        &options.stride,
        &options.padding,
        &options.dilation,
        &[in_h, in_w],
    );
    let output = empty_device_contiguous_dtype(
        input.client.clone(),
        input.device.clone(),
        Shape::new([batch, out_channels, out_size[0], out_size[1]]),
        DType::F32,
    );
    let rows = batch
        .checked_mul(out_size[0])
        .and_then(|rows| rows.checked_mul(out_size[1]))
        .expect("packed-F16 convolution row count overflowed");
    let row_tiles = rows.div_ceil(WORKGROUP_AXIS);
    let row_tile_stride = row_tiles.min(u16::MAX as usize).max(1);
    let cubes_x = u32::try_from(out_channels.div_ceil(OUTPUT_COLUMN_TILE))
        .expect("packed-F16 convolution channel dispatch exceeds u32");
    let cubes_y =
        u32::try_from(row_tile_stride).expect("packed-F16 convolution row dispatch exceeds u32");
    let cubes_z = u32::try_from(row_tiles.div_ceil(row_tile_stride))
        .expect("packed-F16 convolution depth dispatch exceeds u32");

    unsafe {
        packed_f16_conv2d_kernel::launch_unchecked(
            &output.client,
            CubeCount::Static(cubes_x, cubes_y, cubes_z),
            CubeDim::new_2d(WORKGROUP_AXIS_U32, WORKGROUP_AXIS_U32),
            address_type!(input, weight, bias, output),
            input.into_tensor_arg(),
            weight.into_tensor_arg(),
            bias.map(|value| value.into_tensor_arg()).into(),
            output.clone().into_tensor_arg(),
            PackedConv2dArgsLaunch::new(
                options.stride[0] as u32,
                options.stride[1] as u32,
                options.dilation[0] as u32,
                options.dilation[1] as u32,
                options.padding[0] as i32,
                options.padding[1] as i32,
                cubes_y,
            ),
        )
    };
    output
}
