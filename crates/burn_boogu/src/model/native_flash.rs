//! Native CubeCL attention operations that must execute Cubek FlashAttention.

use core::{fmt, marker::PhantomData};

use burn::{
    prelude::*,
    tensor::{DType, Shape, TensorPrimitive, ops::AttentionModuleOptions},
};
use burn_cubecl::{
    BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement,
    kernel::attention::{AttentionStrategy, attention},
    ops::numeric::empty_device_dtype,
    tensor::CubeTensor,
};
use burn_fusion::{
    Fusion, FusionBackend, FusionRuntime, FusionTensor,
    stream::{Operation, OperationStreams},
};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};

#[cfg(feature = "wgpu")]
use burn_wgpu::WgpuRuntime;
use cubek::attention::{
    definition::{
        AccumulatorPrecision, AttentionBlueprint, AttentionDims, AttentionGlobalTypes,
        AttentionOptions, AttentionPartitionSize, AttentionStageSize, AttentionTileSize,
        AttentionTilingScheme, HypercubeBlueprint,
    },
    launch::{BlueprintStrategy, Strategy, launch_ref},
    routines::blackbox_accelerated::BlackboxAcceleratedRoutine,
};

const BOOGU_ATTENTION_HEAD_DIM: usize = 120;
const PADDED_BLACKBOX_HEAD_DIM: usize = 128;

#[cfg(feature = "wgpu")]
type FusionQkv<B> = (
    Tensor<Fusion<B>, 4>,
    Tensor<Fusion<B>, 4>,
    Tensor<Fusion<B>, 4>,
);

/// One-dispatch preparation for the already-validated padded Cubek attention kernel.
///
/// The operation deliberately stops at materializing the exact Q/K/V tensors consumed by the
/// existing blackbox blueprint. Keeping this as a separate preparation dispatch lets every query
/// chunk reuse one expanded key/value pair while removing Burn's repeat, scale, zero, cast, and
/// concatenation graph.
#[cfg(feature = "wgpu")]
mod gqa_padding {
    use burn::tensor::{DType, Shape};
    use burn_cubecl::{CubeRuntime, cubecl, ops::numeric::empty_device_dtype, tensor::CubeTensor};
    use cubecl::{calculate_cube_count_elemwise, prelude::*};

    use super::{BOOGU_ATTENTION_HEAD_DIM, PADDED_BLACKBOX_HEAD_DIM};

    /// Sum one 128-lane shared-memory row in deterministic F32 tree order.
    #[cube]
    fn reduce_shared_row_128(shared: &mut SharedMemory<f32>) -> f32 {
        sync_cube();
        if UNIT_POS < 64 {
            shared[UNIT_POS as usize] += shared[(UNIT_POS + 64) as usize];
        }
        sync_cube();
        if UNIT_POS < 32 {
            shared[UNIT_POS as usize] += shared[(UNIT_POS + 32) as usize];
        }
        sync_cube();
        if UNIT_POS < 16 {
            shared[UNIT_POS as usize] += shared[(UNIT_POS + 16) as usize];
        }
        sync_cube();
        if UNIT_POS < 8 {
            shared[UNIT_POS as usize] += shared[(UNIT_POS + 8) as usize];
        }
        sync_cube();
        if UNIT_POS < 4 {
            shared[UNIT_POS as usize] += shared[(UNIT_POS + 4) as usize];
        }
        sync_cube();
        if UNIT_POS < 2 {
            shared[UNIT_POS as usize] += shared[(UNIT_POS + 2) as usize];
        }
        sync_cube();
        if UNIT_POS < 1 {
            shared[0] += shared[1];
        }
        sync_cube();
        shared[0]
    }

    /// Normalize one 120-wide row per workgroup without materializing a full F32 activation.
    ///
    /// The reduction, divide, and gamma multiply are all explicit F32 operations. Only the final
    /// normalized value is converted to the activation storage type. Q and K invoke this kernel
    /// in separate dispatches so the much smaller grouped-K domain cannot serialize work inside
    /// a subset of the Q workgroups.
    #[cube(launch)]
    fn balanced_strict_rms_norm_kernel<F: Float>(
        input: &Tensor<F>,
        gamma: &Tensor<F>,
        output: &mut Tensor<F>,
        epsilon: InputScalar,
        #[define(F)] _dtype: StorageType,
    ) {
        let component = UNIT_POS as usize;
        let row = CUBE_POS;
        let mut shared = SharedMemory::<f32>::new(PADDED_BLACKBOX_HEAD_DIM);

        let input_rows = input.shape(0) * input.shape(1) * input.shape(2);
        if row < input_rows {
            let sequence = row % input.shape(2);
            let batch_head = row / input.shape(2);
            let head = batch_head % input.shape(1);
            let batch = batch_head / input.shape(1);
            let source = batch * input.stride(0)
                + head * input.stride(1)
                + sequence * input.stride(2)
                + component * input.stride(3);
            let target = batch * output.stride(0)
                + head * output.stride(1)
                + sequence * output.stride(2)
                + component * output.stride(3);
            let value = if component < input.shape(3) {
                f32::cast_from(input[source])
            } else {
                f32::cast_from(0)
            };
            shared[component] = value * value;
            let sum_square = reduce_shared_row_128(&mut shared);

            if component < input.shape(3) {
                let rms = f32::sqrt(sum_square / input.shape(3) as f32 + epsilon.get::<f32>());
                let gamma = f32::cast_from(gamma[component * gamma.stride(0)]);
                output[target] = F::cast_from(value / rms * gamma);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[cube(launch)]
    fn prepare_gqa_padded_qkv_kernel<F: Float>(
        query: &Tensor<F>,
        key: &Tensor<F>,
        value: &Tensor<F>,
        padded_query: &mut Tensor<F>,
        padded_key: &mut Tensor<F>,
        padded_value: &mut Tensor<F>,
        query_scale: InputScalar,
        #[define(F)] _dtype: StorageType,
    ) {
        let position = ABSOLUTE_POS;

        let query_elements = padded_query.shape(0)
            * padded_query.shape(1)
            * padded_query.shape(2)
            * padded_query.shape(3);
        if position < query_elements {
            let padded_dim = padded_query.shape(3);
            let component = position % padded_dim;
            let row = position / padded_dim;
            let sequence = row % padded_query.shape(2);
            let batch_head = row / padded_query.shape(2);
            let head = batch_head % padded_query.shape(1);
            let batch = batch_head / padded_query.shape(1);
            let target = batch * padded_query.stride(0)
                + head * padded_query.stride(1)
                + sequence * padded_query.stride(2)
                + component * padded_query.stride(3);

            padded_query[target] = if component < query.shape(3) {
                let source = batch * query.stride(0)
                    + head * query.stride(1)
                    + sequence * query.stride(2)
                    + component * query.stride(3);
                query[source] * query_scale.get::<F>()
            } else {
                F::new(0.0_f32)
            };
        }

        let key_elements =
            padded_key.shape(0) * padded_key.shape(1) * padded_key.shape(2) * padded_key.shape(3);
        if position < key_elements {
            let padded_dim = padded_key.shape(3);
            let component = position % padded_dim;
            let row = position / padded_dim;
            let sequence = row % padded_key.shape(2);
            let batch_head = row / padded_key.shape(2);
            let query_head = batch_head % padded_key.shape(1);
            let batch = batch_head / padded_key.shape(1);
            let groups = padded_key.shape(1) / key.shape(1);
            let key_value_head = query_head / groups;
            let target = batch * padded_key.stride(0)
                + query_head * padded_key.stride(1)
                + sequence * padded_key.stride(2)
                + component * padded_key.stride(3);

            padded_key[target] = if component < key.shape(3) {
                let source = batch * key.stride(0)
                    + key_value_head * key.stride(1)
                    + sequence * key.stride(2)
                    + component * key.stride(3);
                key[source]
            } else {
                F::new(0.0_f32)
            };
        }

        let value_elements = padded_value.shape(0)
            * padded_value.shape(1)
            * padded_value.shape(2)
            * padded_value.shape(3);
        if position < value_elements {
            let padded_dim = padded_value.shape(3);
            let component = position % padded_dim;
            let row = position / padded_dim;
            let sequence = row % padded_value.shape(2);
            let batch_head = row / padded_value.shape(2);
            let query_head = batch_head % padded_value.shape(1);
            let batch = batch_head / padded_value.shape(1);
            let groups = padded_value.shape(1) / value.shape(1);
            let key_value_head = query_head / groups;
            let target = batch * padded_value.stride(0)
                + query_head * padded_value.stride(1)
                + sequence * padded_value.stride(2)
                + component * padded_value.stride(3);

            padded_value[target] = if component < value.shape(3) {
                let source = batch * value.stride(0)
                    + key_value_head * value.stride(1)
                    + sequence * value.stride(2)
                    + component * value.stride(3);
                value[source]
            } else {
                F::new(0.0_f32)
            };
        }
    }

    /// Fuse only repeated-pair RoPE with the established GQA expansion and padding operation.
    ///
    /// Q/K arrive after the stock Burn strict-F32 RMSNorm graph has completed. This kernel keeps
    /// all remaining activation arithmetic in the input F16 dtype, matching the prior RoPE and
    /// query-scale boundaries while avoiding their intermediate tensors and dispatches.
    #[allow(clippy::too_many_arguments)]
    #[cube(launch)]
    fn prepare_gqa_rope_padded_qkv_kernel<F: Float>(
        query: &Tensor<F>,
        key: &Tensor<F>,
        value: &Tensor<F>,
        cos: &Tensor<F>,
        sin: &Tensor<F>,
        padded_query: &mut Tensor<F>,
        padded_key: &mut Tensor<F>,
        padded_value: &mut Tensor<F>,
        query_scale: InputScalar,
        #[define(F)] _dtype: StorageType,
    ) {
        let position = ABSOLUTE_POS;
        let padded_dim = PADDED_BLACKBOX_HEAD_DIM;

        let query_elements =
            padded_query.shape(0) * padded_query.shape(1) * padded_query.shape(2) * padded_dim;
        if position < query_elements {
            let component = position % padded_dim;
            let row = position / padded_dim;
            let sequence = row % query.shape(2);
            let batch_head = row / query.shape(2);
            let head = batch_head % query.shape(1);
            let batch = batch_head / query.shape(1);
            let target = batch * padded_query.stride(0)
                + head * padded_query.stride(1)
                + sequence * padded_query.stride(2)
                + component * padded_query.stride(3);
            if component < query.shape(3) {
                let source = batch * query.stride(0)
                    + head * query.stride(1)
                    + sequence * query.stride(2)
                    + component * query.stride(3);
                let paired_component = if component.is_multiple_of(2) {
                    component + 1
                } else {
                    component - 1
                };
                let paired_source = batch * query.stride(0)
                    + head * query.stride(1)
                    + sequence * query.stride(2)
                    + paired_component * query.stride(3);
                let rotated = if component.is_multiple_of(2) {
                    -query[paired_source]
                } else {
                    query[paired_source]
                };
                let rope_batch = if cos.shape(0) == 1 {
                    usize::cast_from(0)
                } else {
                    batch
                };
                let cos_index = rope_batch * cos.stride(0)
                    + sequence * cos.stride(1)
                    + component * cos.stride(2);
                let sin_index = rope_batch * sin.stride(0)
                    + sequence * sin.stride(1)
                    + component * sin.stride(2);
                let roped = query[source] * cos[cos_index] + rotated * sin[sin_index];
                padded_query[target] = roped * query_scale.get::<F>();
            } else {
                padded_query[target] = F::new(0.0_f32);
            }
        }

        let key_source_elements = key.shape(0) * key.shape(1) * key.shape(2) * padded_dim;
        if position < key_source_elements {
            let component = position % padded_dim;
            let row = position / padded_dim;
            let sequence = row % key.shape(2);
            let batch_head = row / key.shape(2);
            let key_value_head = batch_head % key.shape(1);
            let batch = batch_head / key.shape(1);
            let groups = padded_key.shape(1) / key.shape(1);
            let prepared = if component < key.shape(3) {
                let source = batch * key.stride(0)
                    + key_value_head * key.stride(1)
                    + sequence * key.stride(2)
                    + component * key.stride(3);
                let paired_component = if component.is_multiple_of(2) {
                    component + 1
                } else {
                    component - 1
                };
                let paired_source = batch * key.stride(0)
                    + key_value_head * key.stride(1)
                    + sequence * key.stride(2)
                    + paired_component * key.stride(3);
                let rotated = if component.is_multiple_of(2) {
                    -key[paired_source]
                } else {
                    key[paired_source]
                };
                let rope_batch = if cos.shape(0) == 1 {
                    usize::cast_from(0)
                } else {
                    batch
                };
                let cos_index = rope_batch * cos.stride(0)
                    + sequence * cos.stride(1)
                    + component * cos.stride(2);
                let sin_index = rope_batch * sin.stride(0)
                    + sequence * sin.stride(1)
                    + component * sin.stride(2);
                key[source] * cos[cos_index] + rotated * sin[sin_index]
            } else {
                F::new(0.0_f32)
            };
            for group in 0..groups {
                let query_head = key_value_head * groups + group;
                let target = batch * padded_key.stride(0)
                    + query_head * padded_key.stride(1)
                    + sequence * padded_key.stride(2)
                    + component * padded_key.stride(3);
                padded_key[target] = prepared;
            }
        }

        let value_source_elements = value.shape(0) * value.shape(1) * value.shape(2) * padded_dim;
        if position < value_source_elements {
            let component = position % padded_dim;
            let row = position / padded_dim;
            let sequence = row % value.shape(2);
            let batch_head = row / value.shape(2);
            let key_value_head = batch_head % value.shape(1);
            let batch = batch_head / value.shape(1);
            let groups = padded_value.shape(1) / value.shape(1);
            let prepared = if component < value.shape(3) {
                let source = batch * value.stride(0)
                    + key_value_head * value.stride(1)
                    + sequence * value.stride(2)
                    + component * value.stride(3);
                value[source]
            } else {
                F::new(0.0_f32)
            };
            for group in 0..groups {
                let query_head = key_value_head * groups + group;
                let target = batch * padded_value.stride(0)
                    + query_head * padded_value.stride(1)
                    + sequence * padded_value.stride(2)
                    + component * padded_value.stride(3);
                padded_value[target] = prepared;
            }
        }
    }

    /// One workgroup owns one raw Q, K, and V row at the same flattened row index.
    ///
    /// Q/K reduction and all arithmetic before the F16 output store are explicit. K/V head
    /// expansion happens only after each source row has been read, avoiding duplicate reductions
    /// and duplicate source loads for grouped heads.
    #[allow(clippy::too_many_arguments)]
    #[cube(launch)]
    fn prepare_gqa_strict_norm_rope_padded_qkv_kernel<F: Float>(
        query: &Tensor<F>,
        key: &Tensor<F>,
        value: &Tensor<F>,
        query_gamma: &Tensor<F>,
        key_gamma: &Tensor<F>,
        cos: &Tensor<F>,
        sin: &Tensor<F>,
        padded_query: &mut Tensor<F>,
        padded_key: &mut Tensor<F>,
        padded_value: &mut Tensor<F>,
        query_epsilon: InputScalar,
        key_epsilon: InputScalar,
        query_scale: InputScalar,
        #[define(F)] _dtype: StorageType,
    ) {
        let component = UNIT_POS as usize;
        let row = CUBE_POS;
        let mut shared = SharedMemory::<f32>::new(PADDED_BLACKBOX_HEAD_DIM);

        let query_rows = query.shape(0) * query.shape(1) * query.shape(2);
        if row < query_rows {
            let sequence = row % query.shape(2);
            let batch_head = row / query.shape(2);
            let head = batch_head % query.shape(1);
            let batch = batch_head / query.shape(1);
            let source = batch * query.stride(0)
                + head * query.stride(1)
                + sequence * query.stride(2)
                + component * query.stride(3);
            let square = if component < query.shape(3) {
                let value = f32::cast_from(query[source]);
                value * value
            } else {
                f32::cast_from(0)
            };
            shared[component] = square;
            let sum_square = reduce_shared_row_128(&mut shared);
            let rms = f32::sqrt(sum_square / query.shape(3) as f32 + query_epsilon.get::<f32>());
            let target = batch * padded_query.stride(0)
                + head * padded_query.stride(1)
                + sequence * padded_query.stride(2)
                + component * padded_query.stride(3);
            if component < query.shape(3) {
                // Preserve the exact public graph's F16 boundary after the strict F32 RMS.
                let rms = F::cast_from(rms);
                let normalized =
                    query[source] / rms * query_gamma[component * query_gamma.stride(0)];
                let paired_component = if component.is_multiple_of(2) {
                    component + 1
                } else {
                    component - 1
                };
                let paired_source = batch * query.stride(0)
                    + head * query.stride(1)
                    + sequence * query.stride(2)
                    + paired_component * query.stride(3);
                let paired = query[paired_source] / rms
                    * query_gamma[paired_component * query_gamma.stride(0)];
                let rotated = if component.is_multiple_of(2) {
                    -paired
                } else {
                    paired
                };
                let rope_batch = if cos.shape(0) == 1 {
                    usize::cast_from(0)
                } else {
                    batch
                };
                let rope_index = rope_batch * cos.stride(0)
                    + sequence * cos.stride(1)
                    + component * cos.stride(2);
                let sin_index = rope_batch * sin.stride(0)
                    + sequence * sin.stride(1)
                    + component * sin.stride(2);
                let roped = normalized * cos[rope_index] + rotated * sin[sin_index];
                padded_query[target] = roped * query_scale.get::<F>();
            } else {
                padded_query[target] = F::new(0.0_f32);
            }
        }

        // Every query lane consumed the reduction before shared memory is reused for K.
        sync_cube();

        let key_rows = key.shape(0) * key.shape(1) * key.shape(2);
        if row < key_rows {
            let sequence = row % key.shape(2);
            let batch_head = row / key.shape(2);
            let key_value_head = batch_head % key.shape(1);
            let batch = batch_head / key.shape(1);
            let source = batch * key.stride(0)
                + key_value_head * key.stride(1)
                + sequence * key.stride(2)
                + component * key.stride(3);
            let square = if component < key.shape(3) {
                let value = f32::cast_from(key[source]);
                value * value
            } else {
                f32::cast_from(0)
            };
            shared[component] = square;
            let sum_square = reduce_shared_row_128(&mut shared);
            let rms = f32::sqrt(sum_square / key.shape(3) as f32 + key_epsilon.get::<f32>());
            let groups = padded_key.shape(1) / key.shape(1);
            if component < key.shape(3) {
                let rms = F::cast_from(rms);
                let normalized = key[source] / rms * key_gamma[component * key_gamma.stride(0)];
                let paired_component = if component.is_multiple_of(2) {
                    component + 1
                } else {
                    component - 1
                };
                let paired_source = batch * key.stride(0)
                    + key_value_head * key.stride(1)
                    + sequence * key.stride(2)
                    + paired_component * key.stride(3);
                let paired =
                    key[paired_source] / rms * key_gamma[paired_component * key_gamma.stride(0)];
                let rotated = if component.is_multiple_of(2) {
                    -paired
                } else {
                    paired
                };
                let rope_batch = if cos.shape(0) == 1 {
                    usize::cast_from(0)
                } else {
                    batch
                };
                let rope_index = rope_batch * cos.stride(0)
                    + sequence * cos.stride(1)
                    + component * cos.stride(2);
                let sin_index = rope_batch * sin.stride(0)
                    + sequence * sin.stride(1)
                    + component * sin.stride(2);
                let roped = normalized * cos[rope_index] + rotated * sin[sin_index];
                for group in 0..groups {
                    let query_head = key_value_head * groups + group;
                    let target = batch * padded_key.stride(0)
                        + query_head * padded_key.stride(1)
                        + sequence * padded_key.stride(2)
                        + component * padded_key.stride(3);
                    padded_key[target] = roped;
                }
            } else {
                for group in 0..groups {
                    let query_head = key_value_head * groups + group;
                    let target = batch * padded_key.stride(0)
                        + query_head * padded_key.stride(1)
                        + sequence * padded_key.stride(2)
                        + component * padded_key.stride(3);
                    padded_key[target] = F::new(0.0_f32);
                }
            }
        }

        let value_rows = value.shape(0) * value.shape(1) * value.shape(2);
        if row < value_rows {
            let sequence = row % value.shape(2);
            let batch_head = row / value.shape(2);
            let key_value_head = batch_head % value.shape(1);
            let batch = batch_head / value.shape(1);
            let groups = padded_value.shape(1) / value.shape(1);
            let prepared = if component < value.shape(3) {
                let source = batch * value.stride(0)
                    + key_value_head * value.stride(1)
                    + sequence * value.stride(2)
                    + component * value.stride(3);
                value[source]
            } else {
                F::new(0.0_f32)
            };
            for group in 0..groups {
                let query_head = key_value_head * groups + group;
                let target = batch * padded_value.stride(0)
                    + query_head * padded_value.stride(1)
                    + sequence * padded_value.stride(2)
                    + component * padded_value.stride(3);
                padded_value[target] = prepared;
            }
        }
    }

    pub(super) fn launch<R: CubeRuntime>(
        query: CubeTensor<R>,
        key: CubeTensor<R>,
        value: CubeTensor<R>,
    ) -> (CubeTensor<R>, CubeTensor<R>, CubeTensor<R>) {
        let batch = query.meta.shape[0];
        let query_heads = query.meta.shape[1];
        let query_len = query.meta.shape[2];
        let key_len = key.meta.shape[2];
        let value_len = value.meta.shape[2];
        let client = query.client.clone();
        let device = query.device.clone();
        let dtype = query.dtype;
        let padded_query = empty_device_dtype::<R>(
            client.clone(),
            device.clone(),
            Shape::new([batch, query_heads, query_len, PADDED_BLACKBOX_HEAD_DIM]),
            dtype,
        );
        let padded_key = empty_device_dtype::<R>(
            client.clone(),
            device.clone(),
            Shape::new([batch, query_heads, key_len, PADDED_BLACKBOX_HEAD_DIM]),
            dtype,
        );
        let padded_value = empty_device_dtype::<R>(
            client.clone(),
            device,
            Shape::new([batch, query_heads, value_len, PADDED_BLACKBOX_HEAD_DIM]),
            dtype,
        );
        let working_units = padded_query
            .meta
            .num_elements()
            .max(padded_key.meta.num_elements())
            .max(padded_value.meta.num_elements());
        let cube_dim = CubeDim::new(&client, working_units);
        let cube_count = calculate_cube_count_elemwise(&client, working_units, cube_dim);
        let query_scale =
            (PADDED_BLACKBOX_HEAD_DIM as f64 / BOOGU_ATTENTION_HEAD_DIM as f64).sqrt();

        prepare_gqa_padded_qkv_kernel::launch(
            &client,
            cube_count,
            cube_dim,
            query.into_tensor_arg(),
            key.into_tensor_arg(),
            value.into_tensor_arg(),
            padded_query.clone().into_tensor_arg(),
            padded_key.clone().into_tensor_arg(),
            padded_value.clone().into_tensor_arg(),
            InputScalar::new(query_scale, dtype),
            dtype.into(),
        );

        (padded_query, padded_key, padded_value)
    }

    pub(super) fn launch_rope<R: CubeRuntime>(
        query: CubeTensor<R>,
        key: CubeTensor<R>,
        value: CubeTensor<R>,
        cos: CubeTensor<R>,
        sin: CubeTensor<R>,
    ) -> (CubeTensor<R>, CubeTensor<R>, CubeTensor<R>) {
        let batch = query.meta.shape[0];
        let query_heads = query.meta.shape[1];
        let query_len = query.meta.shape[2];
        let key_len = key.meta.shape[2];
        let value_len = value.meta.shape[2];
        let client = query.client.clone();
        let device = query.device.clone();
        let dtype = query.dtype;
        let padded_query = empty_device_dtype::<R>(
            client.clone(),
            device.clone(),
            Shape::new([batch, query_heads, query_len, PADDED_BLACKBOX_HEAD_DIM]),
            dtype,
        );
        let padded_key = empty_device_dtype::<R>(
            client.clone(),
            device.clone(),
            Shape::new([batch, query_heads, key_len, PADDED_BLACKBOX_HEAD_DIM]),
            dtype,
        );
        let padded_value = empty_device_dtype::<R>(
            client.clone(),
            device,
            Shape::new([batch, query_heads, value_len, PADDED_BLACKBOX_HEAD_DIM]),
            dtype,
        );
        let query_elements = padded_query.meta.num_elements();
        let key_source_elements =
            key.meta.shape[0] * key.meta.shape[1] * key.meta.shape[2] * PADDED_BLACKBOX_HEAD_DIM;
        let value_source_elements = value.meta.shape[0]
            * value.meta.shape[1]
            * value.meta.shape[2]
            * PADDED_BLACKBOX_HEAD_DIM;
        let working_units = query_elements
            .max(key_source_elements)
            .max(value_source_elements);
        let cube_dim = CubeDim::new(&client, working_units);
        let cube_count = calculate_cube_count_elemwise(&client, working_units, cube_dim);
        let query_scale =
            (PADDED_BLACKBOX_HEAD_DIM as f64 / BOOGU_ATTENTION_HEAD_DIM as f64).sqrt();

        prepare_gqa_rope_padded_qkv_kernel::launch(
            &client,
            cube_count,
            cube_dim,
            query.into_tensor_arg(),
            key.into_tensor_arg(),
            value.into_tensor_arg(),
            cos.into_tensor_arg(),
            sin.into_tensor_arg(),
            padded_query.clone().into_tensor_arg(),
            padded_key.clone().into_tensor_arg(),
            padded_value.clone().into_tensor_arg(),
            InputScalar::new(query_scale, dtype),
            dtype.into(),
        );

        (padded_query, padded_key, padded_value)
    }

    pub(super) fn launch_balanced_strict_rms_norm<R: CubeRuntime>(
        input: CubeTensor<R>,
        gamma: CubeTensor<R>,
        epsilon: f64,
    ) -> CubeTensor<R> {
        let row_count = input.meta.shape[0] * input.meta.shape[1] * input.meta.shape[2];
        let client = input.client.clone();
        let dtype = input.dtype;
        let output = empty_device_dtype::<R>(
            client.clone(),
            input.device.clone(),
            Shape::new([
                input.meta.shape[0],
                input.meta.shape[1],
                input.meta.shape[2],
                input.meta.shape[3],
            ]),
            dtype,
        );
        let cube_dim = CubeDim::new_1d(PADDED_BLACKBOX_HEAD_DIM as u32);
        let cube_count =
            calculate_cube_count_elemwise(&client, row_count * PADDED_BLACKBOX_HEAD_DIM, cube_dim);

        balanced_strict_rms_norm_kernel::launch(
            &client,
            cube_count,
            cube_dim,
            input.into_tensor_arg(),
            gamma.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            InputScalar::new(epsilon, DType::F32),
            dtype.into(),
        );

        output
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch_strict_norm_rope<R: CubeRuntime>(
        query: CubeTensor<R>,
        key: CubeTensor<R>,
        value: CubeTensor<R>,
        query_gamma: CubeTensor<R>,
        key_gamma: CubeTensor<R>,
        cos: CubeTensor<R>,
        sin: CubeTensor<R>,
        query_epsilon: f64,
        key_epsilon: f64,
    ) -> (CubeTensor<R>, CubeTensor<R>, CubeTensor<R>) {
        let batch = query.meta.shape[0];
        let query_heads = query.meta.shape[1];
        let query_len = query.meta.shape[2];
        let key_len = key.meta.shape[2];
        let value_len = value.meta.shape[2];
        let client = query.client.clone();
        let device = query.device.clone();
        let dtype = query.dtype;
        let padded_query = empty_device_dtype::<R>(
            client.clone(),
            device.clone(),
            Shape::new([batch, query_heads, query_len, PADDED_BLACKBOX_HEAD_DIM]),
            dtype,
        );
        let padded_key = empty_device_dtype::<R>(
            client.clone(),
            device.clone(),
            Shape::new([batch, query_heads, key_len, PADDED_BLACKBOX_HEAD_DIM]),
            dtype,
        );
        let padded_value = empty_device_dtype::<R>(
            client.clone(),
            device,
            Shape::new([batch, query_heads, value_len, PADDED_BLACKBOX_HEAD_DIM]),
            dtype,
        );
        let query_rows = query.meta.shape[0] * query.meta.shape[1] * query.meta.shape[2];
        let key_rows = key.meta.shape[0] * key.meta.shape[1] * key.meta.shape[2];
        let value_rows = value.meta.shape[0] * value.meta.shape[1] * value.meta.shape[2];
        let row_count = query_rows.max(key_rows).max(value_rows);
        let cube_dim = CubeDim::new_1d(PADDED_BLACKBOX_HEAD_DIM as u32);
        let cube_count =
            calculate_cube_count_elemwise(&client, row_count * PADDED_BLACKBOX_HEAD_DIM, cube_dim);
        let query_scale =
            (PADDED_BLACKBOX_HEAD_DIM as f64 / BOOGU_ATTENTION_HEAD_DIM as f64).sqrt();

        prepare_gqa_strict_norm_rope_padded_qkv_kernel::launch(
            &client,
            cube_count,
            cube_dim,
            query.into_tensor_arg(),
            key.into_tensor_arg(),
            value.into_tensor_arg(),
            query_gamma.into_tensor_arg(),
            key_gamma.into_tensor_arg(),
            cos.into_tensor_arg(),
            sin.into_tensor_arg(),
            padded_query.clone().into_tensor_arg(),
            padded_key.clone().into_tensor_arg(),
            padded_value.clone().into_tensor_arg(),
            InputScalar::new(query_epsilon, DType::F32),
            InputScalar::new(key_epsilon, DType::F32),
            InputScalar::new(query_scale, dtype),
            dtype.into(),
        );

        (padded_query, padded_key, padded_value)
    }
}

/// Exact backend accepted by the native WGPU required-FlashUnit execution path.
///
/// This is the same fused backend as `burn_wgpu::Wgpu<f32, i32, u32>`. Naming it here makes the
/// backend restriction on the optimized denoiser API explicit.
#[cfg(feature = "wgpu")]
pub type NativeWgpuBackend = Fusion<CubeBackend<WgpuRuntime, f32, i32, u32>>;

trait RequiredFlashUnitBackend: FusionBackend {
    fn launch_required_flash_unit(
        query: Self::FloatTensorPrimitive,
        key: Self::FloatTensorPrimitive,
        value: Self::FloatTensorPrimitive,
    ) -> Self::FloatTensorPrimitive;

    fn launch_required_blackbox_accelerated(
        query: Self::FloatTensorPrimitive,
        key: Self::FloatTensorPrimitive,
        value: Self::FloatTensorPrimitive,
        num_planes: u8,
        seq_kv_tiles: u8,
        seq_q_tiles: u8,
    ) -> Self::FloatTensorPrimitive;

    #[cfg(feature = "wgpu")]
    fn launch_prepare_gqa_padded_qkv(
        query: Self::FloatTensorPrimitive,
        key: Self::FloatTensorPrimitive,
        value: Self::FloatTensorPrimitive,
    ) -> (
        Self::FloatTensorPrimitive,
        Self::FloatTensorPrimitive,
        Self::FloatTensorPrimitive,
    );

    #[cfg(feature = "wgpu")]
    fn launch_prepare_gqa_rope_padded_qkv(
        query: Self::FloatTensorPrimitive,
        key: Self::FloatTensorPrimitive,
        value: Self::FloatTensorPrimitive,
        cos: Self::FloatTensorPrimitive,
        sin: Self::FloatTensorPrimitive,
    ) -> (
        Self::FloatTensorPrimitive,
        Self::FloatTensorPrimitive,
        Self::FloatTensorPrimitive,
    );

    #[cfg(feature = "wgpu")]
    fn launch_balanced_strict_rms_norm(
        input: Self::FloatTensorPrimitive,
        gamma: Self::FloatTensorPrimitive,
        epsilon: f64,
    ) -> Self::FloatTensorPrimitive;

    #[cfg(feature = "wgpu")]
    #[allow(clippy::too_many_arguments)]
    fn launch_prepare_gqa_strict_norm_rope_padded_qkv(
        query: Self::FloatTensorPrimitive,
        key: Self::FloatTensorPrimitive,
        value: Self::FloatTensorPrimitive,
        query_gamma: Self::FloatTensorPrimitive,
        key_gamma: Self::FloatTensorPrimitive,
        cos: Self::FloatTensorPrimitive,
        sin: Self::FloatTensorPrimitive,
        query_epsilon: f64,
        key_epsilon: f64,
    ) -> (
        Self::FloatTensorPrimitive,
        Self::FloatTensorPrimitive,
        Self::FloatTensorPrimitive,
    );
}

impl<R, F, I, BT> RequiredFlashUnitBackend for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn launch_required_flash_unit(
        query: Self::FloatTensorPrimitive,
        key: Self::FloatTensorPrimitive,
        value: Self::FloatTensorPrimitive,
    ) -> Self::FloatTensorPrimitive {
        attention::<R>(
            query,
            key,
            value,
            None,
            None,
            AttentionModuleOptions::default(),
            AttentionStrategy::FlashUnit,
        )
        .unwrap_or_else(|error| {
            panic!(
                "required native Cubek FlashUnit attention is unavailable; refusing dense \
                 fallback: {error:?}"
            )
        })
    }

    fn launch_required_blackbox_accelerated(
        query: Self::FloatTensorPrimitive,
        key: Self::FloatTensorPrimitive,
        value: Self::FloatTensorPrimitive,
        num_planes: u8,
        seq_kv_tiles: u8,
        seq_q_tiles: u8,
    ) -> Self::FloatTensorPrimitive {
        forced_blackbox_attention::<R>(query, key, value, num_planes, seq_kv_tiles, seq_q_tiles)
            .unwrap_or_else(|error| {
                panic!(
                    "required native Cubek padded FlashBlackboxAccelerated attention is \
                     unavailable; refusing fallback: {error:?}"
                )
            })
    }

    #[cfg(feature = "wgpu")]
    fn launch_prepare_gqa_padded_qkv(
        query: Self::FloatTensorPrimitive,
        key: Self::FloatTensorPrimitive,
        value: Self::FloatTensorPrimitive,
    ) -> (
        Self::FloatTensorPrimitive,
        Self::FloatTensorPrimitive,
        Self::FloatTensorPrimitive,
    ) {
        gqa_padding::launch::<R>(query, key, value)
    }

    #[cfg(feature = "wgpu")]
    fn launch_prepare_gqa_rope_padded_qkv(
        query: Self::FloatTensorPrimitive,
        key: Self::FloatTensorPrimitive,
        value: Self::FloatTensorPrimitive,
        cos: Self::FloatTensorPrimitive,
        sin: Self::FloatTensorPrimitive,
    ) -> (
        Self::FloatTensorPrimitive,
        Self::FloatTensorPrimitive,
        Self::FloatTensorPrimitive,
    ) {
        gqa_padding::launch_rope::<R>(query, key, value, cos, sin)
    }

    #[cfg(feature = "wgpu")]
    fn launch_balanced_strict_rms_norm(
        input: Self::FloatTensorPrimitive,
        gamma: Self::FloatTensorPrimitive,
        epsilon: f64,
    ) -> Self::FloatTensorPrimitive {
        gqa_padding::launch_balanced_strict_rms_norm::<R>(input, gamma, epsilon)
    }

    #[cfg(feature = "wgpu")]
    fn launch_prepare_gqa_strict_norm_rope_padded_qkv(
        query: Self::FloatTensorPrimitive,
        key: Self::FloatTensorPrimitive,
        value: Self::FloatTensorPrimitive,
        query_gamma: Self::FloatTensorPrimitive,
        key_gamma: Self::FloatTensorPrimitive,
        cos: Self::FloatTensorPrimitive,
        sin: Self::FloatTensorPrimitive,
        query_epsilon: f64,
        key_epsilon: f64,
    ) -> (
        Self::FloatTensorPrimitive,
        Self::FloatTensorPrimitive,
        Self::FloatTensorPrimitive,
    ) {
        gqa_padding::launch_strict_norm_rope::<R>(
            query,
            key,
            value,
            query_gamma,
            key_gamma,
            cos,
            sin,
            query_epsilon,
            key_epsilon,
        )
    }
}

fn forced_blackbox_attention<R: CubeRuntime>(
    query: CubeTensor<R>,
    key: CubeTensor<R>,
    value: CubeTensor<R>,
    num_planes: u8,
    seq_kv_tiles: u8,
    seq_q_tiles: u8,
) -> Result<CubeTensor<R>, cubek::attention::definition::AttentionSetupError> {
    assert_supported_forced_blackbox_partition_configuration(num_planes, seq_kv_tiles, seq_q_tiles);
    let client = query.client.clone();
    let dims = AttentionDims {
        batch: query.meta.shape[0],
        num_heads: query.meta.shape[1],
        seq_q: query.meta.shape[2],
        head_dim: query.meta.shape[3],
        seq_kv: key.meta.shape[2],
        val_dim: value.meta.shape[3],
    };
    let out = empty_device_dtype::<R>(
        client.clone(),
        query.device.clone(),
        Shape::new([dims.batch, dims.num_heads, dims.seq_q, dims.val_dim]),
        query.dtype,
    );
    let global_types = AttentionGlobalTypes {
        query: query.dtype.into(),
        key: key.dtype.into(),
        value: value.dtype.into(),
        mask: AttentionGlobalTypes::mask_dtype(&client),
        out: out.dtype.into(),
    };
    let vector_sizes =
        cubek::attention::definition::AttentionVectorSizes::new_max(&client, &global_types);
    let tiling_scheme = AttentionTilingScheme {
        tile_size: AttentionTileSize {
            seq_q: 16,
            head_dim: 16,
            seq_kv: 16,
            val_dim: 16,
        },
        partition_size: AttentionPartitionSize {
            seq_q: u32::from(seq_q_tiles),
            head_dim: 8,
            seq_kv: u32::from(seq_kv_tiles),
            val_dim: 8,
        },
        stage_size: AttentionStageSize {
            seq_q: u32::from(num_planes),
        },
    };
    let blueprint = AttentionBlueprint {
        hypercube_blueprint: HypercubeBlueprint::builder().build(),
        tiling_scheme,
        plane_dim: client.properties().hardware.plane_size_max,
        two_rows_in_array_tile: false,
        vector_sizes,
        masked: false,
        causal: false,
        check_bounds: tiling_scheme.check_bounds(&dims),
    };

    launch_ref::<R>(
        Strategy::BlackboxAccelerated(BlueprintStrategy::<BlackboxAcceleratedRoutine>::Forced(
            blueprint,
        )),
        &client,
        query.binding(),
        key.binding(),
        value.binding(),
        None,
        out.clone().binding(),
        &global_types,
        AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::Strict(
                burn_cubecl::cubecl::ir::StorageType::Scalar(
                    burn_cubecl::cubecl::ir::ElemType::Float(
                        burn_cubecl::cubecl::ir::FloatKind::F32,
                    ),
                ),
            ),
        },
    )?;
    Ok(out)
}

struct FlashUnitOperation<B: RequiredFlashUnitBackend> {
    desc: CustomOpIr,
    _backend: PhantomData<fn() -> B>,
}

impl<B: RequiredFlashUnitBackend> fmt::Debug for FlashUnitOperation<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FlashUnitOperation")
            .field("desc", &self.desc)
            .finish()
    }
}

impl<B> Operation<B::FusionRuntime> for FlashUnitOperation<B>
where
    B: RequiredFlashUnitBackend + 'static,
{
    fn execute(
        &self,
        handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>,
    ) {
        let ([query, key, value], [out]) = self.desc.as_fixed();
        let query = handles.get_float_tensor::<B>(query);
        let key = handles.get_float_tensor::<B>(key);
        let value = handles.get_float_tensor::<B>(value);
        let output = B::launch_required_flash_unit(query, key, value);

        handles.register_float_tensor::<B>(&out.id, output);
    }
}

#[cfg(feature = "wgpu")]
struct GqaPaddedQkvOperation<B: RequiredFlashUnitBackend> {
    desc: CustomOpIr,
    _backend: PhantomData<fn() -> B>,
}

#[cfg(feature = "wgpu")]
struct GqaRopePaddedQkvOperation<B: RequiredFlashUnitBackend> {
    desc: CustomOpIr,
    _backend: PhantomData<fn() -> B>,
}

#[cfg(feature = "wgpu")]
struct BalancedStrictRmsNormOperation<B: RequiredFlashUnitBackend> {
    desc: CustomOpIr,
    epsilon: f64,
    _backend: PhantomData<fn() -> B>,
}

#[cfg(feature = "wgpu")]
struct GqaStrictNormRopePaddedQkvOperation<B: RequiredFlashUnitBackend> {
    desc: CustomOpIr,
    query_epsilon: f64,
    key_epsilon: f64,
    _backend: PhantomData<fn() -> B>,
}

#[cfg(feature = "wgpu")]
impl<B: RequiredFlashUnitBackend> fmt::Debug for BalancedStrictRmsNormOperation<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BalancedStrictRmsNormOperation")
            .field("desc", &self.desc)
            .field("epsilon", &self.epsilon)
            .finish()
    }
}

#[cfg(feature = "wgpu")]
impl<B> Operation<B::FusionRuntime> for BalancedStrictRmsNormOperation<B>
where
    B: RequiredFlashUnitBackend + 'static,
{
    fn execute(
        &self,
        handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>,
    ) {
        let ([input, gamma], [output]) = self.desc.as_fixed();
        let input = handles.get_float_tensor::<B>(input);
        let gamma = handles.get_float_tensor::<B>(gamma);
        let normalized = B::launch_balanced_strict_rms_norm(input, gamma, self.epsilon);

        handles.register_float_tensor::<B>(&output.id, normalized);
    }
}

#[cfg(feature = "wgpu")]
impl<B: RequiredFlashUnitBackend> fmt::Debug for GqaStrictNormRopePaddedQkvOperation<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GqaStrictNormRopePaddedQkvOperation")
            .field("desc", &self.desc)
            .field("query_epsilon", &self.query_epsilon)
            .field("key_epsilon", &self.key_epsilon)
            .finish()
    }
}

#[cfg(feature = "wgpu")]
impl<B> Operation<B::FusionRuntime> for GqaStrictNormRopePaddedQkvOperation<B>
where
    B: RequiredFlashUnitBackend + 'static,
{
    fn execute(
        &self,
        handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>,
    ) {
        let (
            [query, key, value, query_gamma, key_gamma, cos, sin],
            [padded_query, padded_key, padded_value],
        ) = self.desc.as_fixed();
        let query = handles.get_float_tensor::<B>(query);
        let key = handles.get_float_tensor::<B>(key);
        let value = handles.get_float_tensor::<B>(value);
        let query_gamma = handles.get_float_tensor::<B>(query_gamma);
        let key_gamma = handles.get_float_tensor::<B>(key_gamma);
        let cos = handles.get_float_tensor::<B>(cos);
        let sin = handles.get_float_tensor::<B>(sin);
        let (query, key, value) = B::launch_prepare_gqa_strict_norm_rope_padded_qkv(
            query,
            key,
            value,
            query_gamma,
            key_gamma,
            cos,
            sin,
            self.query_epsilon,
            self.key_epsilon,
        );

        handles.register_float_tensor::<B>(&padded_query.id, query);
        handles.register_float_tensor::<B>(&padded_key.id, key);
        handles.register_float_tensor::<B>(&padded_value.id, value);
    }
}

#[cfg(feature = "wgpu")]
impl<B: RequiredFlashUnitBackend> fmt::Debug for GqaPaddedQkvOperation<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GqaPaddedQkvOperation")
            .field("desc", &self.desc)
            .finish()
    }
}

#[cfg(feature = "wgpu")]
impl<B> Operation<B::FusionRuntime> for GqaPaddedQkvOperation<B>
where
    B: RequiredFlashUnitBackend + 'static,
{
    fn execute(
        &self,
        handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>,
    ) {
        let ([query, key, value], [padded_query, padded_key, padded_value]) = self.desc.as_fixed();
        let query = handles.get_float_tensor::<B>(query);
        let key = handles.get_float_tensor::<B>(key);
        let value = handles.get_float_tensor::<B>(value);
        let (query, key, value) = B::launch_prepare_gqa_padded_qkv(query, key, value);

        handles.register_float_tensor::<B>(&padded_query.id, query);
        handles.register_float_tensor::<B>(&padded_key.id, key);
        handles.register_float_tensor::<B>(&padded_value.id, value);
    }
}

#[cfg(feature = "wgpu")]
impl<B: RequiredFlashUnitBackend> fmt::Debug for GqaRopePaddedQkvOperation<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GqaRopePaddedQkvOperation")
            .field("desc", &self.desc)
            .finish()
    }
}

#[cfg(feature = "wgpu")]
impl<B> Operation<B::FusionRuntime> for GqaRopePaddedQkvOperation<B>
where
    B: RequiredFlashUnitBackend + 'static,
{
    fn execute(
        &self,
        handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>,
    ) {
        let ([query, key, value, cos, sin], [padded_query, padded_key, padded_value]) =
            self.desc.as_fixed();
        let query = handles.get_float_tensor::<B>(query);
        let key = handles.get_float_tensor::<B>(key);
        let value = handles.get_float_tensor::<B>(value);
        let cos = handles.get_float_tensor::<B>(cos);
        let sin = handles.get_float_tensor::<B>(sin);
        let (query, key, value) =
            B::launch_prepare_gqa_rope_padded_qkv(query, key, value, cos, sin);

        handles.register_float_tensor::<B>(&padded_query.id, query);
        handles.register_float_tensor::<B>(&padded_key.id, key);
        handles.register_float_tensor::<B>(&padded_value.id, value);
    }
}

struct PaddedBlackboxOperation<B: RequiredFlashUnitBackend> {
    desc: CustomOpIr,
    num_planes: u8,
    seq_kv_tiles: u8,
    seq_q_tiles: u8,
    _backend: PhantomData<fn() -> B>,
}

impl<B: RequiredFlashUnitBackend> fmt::Debug for PaddedBlackboxOperation<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PaddedBlackboxOperation")
            .field("desc", &self.desc)
            .field("num_planes", &self.num_planes)
            .field("seq_kv_tiles", &self.seq_kv_tiles)
            .field("seq_q_tiles", &self.seq_q_tiles)
            .finish()
    }
}

impl<B> Operation<B::FusionRuntime> for PaddedBlackboxOperation<B>
where
    B: RequiredFlashUnitBackend + 'static,
{
    fn execute(
        &self,
        handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>,
    ) {
        let ([query, key, value], [out]) = self.desc.as_fixed();
        let query = handles.get_float_tensor::<B>(query);
        let key = handles.get_float_tensor::<B>(key);
        let value = handles.get_float_tensor::<B>(value);
        let output = B::launch_required_blackbox_accelerated(
            query,
            key,
            value,
            self.num_planes,
            self.seq_kv_tiles,
            self.seq_q_tiles,
        );

        handles.register_float_tensor::<B>(&out.id, output);
    }
}

/// Schedule one full-sequence FlashUnit operation on native WGPU inside Burn Fusion.
///
/// The operation deliberately bypasses Burn's default attention strategy. It either launches
/// Cubek `FlashUnit` or fails during fused-stream execution; it never invokes dense attention or
/// the attention autotuner. Cubek's Burn 0.21 flash routines use F16 tiles, so this production
/// seam accepts only preserved-F16 activations.
#[cfg(feature = "wgpu")]
pub fn required_flash_unit_attention(
    query: Tensor<NativeWgpuBackend, 4>,
    key: Tensor<NativeWgpuBackend, 4>,
    value: Tensor<NativeWgpuBackend, 4>,
) -> Tensor<NativeWgpuBackend, 4> {
    required_flash_unit_attention_impl(query, key, value)
}

/// Schedule bounded-query required FlashUnit operations on native WGPU.
///
/// Every tile sees the complete key/value sequence and is forced through Cubek `FlashUnit`; the
/// query bound only avoids pathological full-query kernel shapes and controls dispatch count.
#[cfg(feature = "wgpu")]
pub fn required_chunked_flash_unit_attention(
    query: Tensor<NativeWgpuBackend, 4>,
    key: Tensor<NativeWgpuBackend, 4>,
    value: Tensor<NativeWgpuBackend, 4>,
    query_chunk_size: usize,
) -> Tensor<NativeWgpuBackend, 4> {
    required_chunked_flash_unit_attention_impl(query, key, value, query_chunk_size)
}

/// Schedule bounded-query, head-dimension-padded blackbox FlashAttention on native WGPU.
///
/// Boogu's 120-wide heads are zero-padded to 128 so WGPU CMMA instructions can divide the head
/// dimension. Query values are multiplied by `sqrt(128 / 120)` before padding, exactly preserving
/// the required `1 / sqrt(120)` attention scale when the kernel applies `1 / sqrt(128)`. Every
/// query tile retains the complete key/value sequence. The operation accepts only F16 activations
/// and fails closed when the requested accelerated kernel cannot be launched.
///
/// `num_planes` must be one of the two configurations validated numerically on native WGPU: 2 or
/// 4. The upstream 8-plane configuration is deliberately rejected because it produced incorrect
/// nonzero outputs on the release adapter.
#[cfg(feature = "wgpu")]
pub fn required_chunked_padded_blackbox_attention(
    query: Tensor<NativeWgpuBackend, 4>,
    key: Tensor<NativeWgpuBackend, 4>,
    value: Tensor<NativeWgpuBackend, 4>,
    query_chunk_size: usize,
    num_planes: u8,
) -> Tensor<NativeWgpuBackend, 4> {
    required_chunked_padded_blackbox_attention_tiled(
        query,
        key,
        value,
        query_chunk_size,
        num_planes,
        1,
    )
}

/// Schedule bounded-query, head-dimension-padded blackbox FlashAttention with an explicit native
/// key/value partition width.
///
/// This has the same numerical and fail-closed contract as
/// [`required_chunked_padded_blackbox_attention`]. `seq_kv_tiles` controls how many 16-row
/// key/value tiles are processed per online-softmax partition and must be 1 or 2. Two tiles require
/// the two-plane configuration; wider four-plane partitions failed the real nonzero WGPU parity
/// gate and are rejected.
#[cfg(feature = "wgpu")]
pub fn required_chunked_padded_blackbox_attention_tiled(
    query: Tensor<NativeWgpuBackend, 4>,
    key: Tensor<NativeWgpuBackend, 4>,
    value: Tensor<NativeWgpuBackend, 4>,
    query_chunk_size: usize,
    num_planes: u8,
    seq_kv_tiles: u8,
) -> Tensor<NativeWgpuBackend, 4> {
    required_chunked_padded_blackbox_attention_partitioned(
        query,
        key,
        value,
        query_chunk_size,
        num_planes,
        seq_kv_tiles,
        1,
    )
}

/// Schedule padded native WGPU blackbox FlashAttention with explicit query and key/value
/// partition widths.
///
/// `seq_q_tiles` is the number of 16-row query tiles retained by each plane. Only 1 is accepted;
/// the two-tile configuration failed the real nonzero native-WGPU numerical gate.
#[cfg(feature = "wgpu")]
pub(crate) fn required_chunked_padded_blackbox_attention_partitioned(
    query: Tensor<NativeWgpuBackend, 4>,
    key: Tensor<NativeWgpuBackend, 4>,
    value: Tensor<NativeWgpuBackend, 4>,
    query_chunk_size: usize,
    num_planes: u8,
    seq_kv_tiles: u8,
    seq_q_tiles: u8,
) -> Tensor<NativeWgpuBackend, 4> {
    required_chunked_padded_blackbox_attention_impl(
        query,
        key,
        value,
        query_chunk_size,
        num_planes,
        seq_kv_tiles,
        seq_q_tiles,
    )
}

/// Internal GQA route used by the native denoiser before key/value heads are materialized.
///
/// Its result and failure contract are identical to
/// [`required_chunked_padded_blackbox_attention_tiled`]. Only the input preparation differs: key
/// and value retain their grouped heads until one custom CubeCL dispatch expands and pads Q/K/V.
#[cfg(feature = "wgpu")]
#[allow(dead_code)]
pub(crate) fn required_chunked_gqa_padded_blackbox_attention_tiled(
    query: Tensor<NativeWgpuBackend, 4>,
    key: Tensor<NativeWgpuBackend, 4>,
    value: Tensor<NativeWgpuBackend, 4>,
    query_chunk_size: usize,
    num_planes: u8,
    seq_kv_tiles: u8,
) -> Tensor<NativeWgpuBackend, 4> {
    required_chunked_gqa_padded_blackbox_attention_partitioned(
        query,
        key,
        value,
        query_chunk_size,
        num_planes,
        seq_kv_tiles,
        1,
    )
}

#[cfg(feature = "wgpu")]
pub(crate) fn required_chunked_gqa_padded_blackbox_attention_partitioned(
    query: Tensor<NativeWgpuBackend, 4>,
    key: Tensor<NativeWgpuBackend, 4>,
    value: Tensor<NativeWgpuBackend, 4>,
    query_chunk_size: usize,
    num_planes: u8,
    seq_kv_tiles: u8,
    seq_q_tiles: u8,
) -> Tensor<NativeWgpuBackend, 4> {
    required_chunked_gqa_padded_blackbox_attention_impl(
        query,
        key,
        value,
        query_chunk_size,
        num_planes,
        seq_kv_tiles,
        seq_q_tiles,
    )
}

/// Native-WGPU route for the parity-gated partition configurations.
#[cfg(feature = "wgpu")]
pub(crate) fn required_chunked_wgpu_padded_blackbox_attention_partitioned(
    query: Tensor<NativeWgpuBackend, 4>,
    key: Tensor<NativeWgpuBackend, 4>,
    value: Tensor<NativeWgpuBackend, 4>,
    query_chunk_size: usize,
    num_planes: u8,
    seq_kv_tiles: u8,
    seq_q_tiles: u8,
) -> Tensor<NativeWgpuBackend, 4> {
    assert_supported_wgpu_blackbox_partition_configuration(num_planes, seq_kv_tiles, seq_q_tiles);
    required_chunked_padded_blackbox_attention_impl_unchecked(
        query,
        key,
        value,
        query_chunk_size,
        num_planes,
        seq_kv_tiles,
        seq_q_tiles,
    )
}

/// Native-WGPU GQA route for the parity-gated partition configurations.
#[cfg(feature = "wgpu")]
pub(crate) fn required_chunked_gqa_wgpu_padded_blackbox_attention_partitioned(
    query: Tensor<NativeWgpuBackend, 4>,
    key: Tensor<NativeWgpuBackend, 4>,
    value: Tensor<NativeWgpuBackend, 4>,
    query_chunk_size: usize,
    num_planes: u8,
    seq_kv_tiles: u8,
    seq_q_tiles: u8,
) -> Tensor<NativeWgpuBackend, 4> {
    assert_supported_wgpu_blackbox_partition_configuration(num_planes, seq_kv_tiles, seq_q_tiles);
    required_chunked_gqa_padded_blackbox_attention_impl_unchecked(
        query,
        key,
        value,
        query_chunk_size,
        num_planes,
        seq_kv_tiles,
        seq_q_tiles,
    )
}

/// Opt-in p4/kv1/q1 candidate that preserves stock strict RMSNorm and fuses the remaining prep.
///
/// Q/K must already be normalized by the established Burn strict-F32 RMSNorm graph. This folds
/// repeated-pair RoPE, query scaling, grouped-head expansion, and 120-to-128 padding into the one
/// preparation dispatch consumed by the existing forced blackbox attention implementation.
#[cfg(feature = "wgpu")]
pub(crate) fn required_chunked_gqa_wgpu_fused_rope_padded_blackbox_attention(
    query: Tensor<NativeWgpuBackend, 4>,
    key: Tensor<NativeWgpuBackend, 4>,
    value: Tensor<NativeWgpuBackend, 4>,
    cos: Tensor<NativeWgpuBackend, 3>,
    sin: Tensor<NativeWgpuBackend, 3>,
    query_chunk_size: usize,
) -> Tensor<NativeWgpuBackend, 4> {
    let (query, key, value) = prepare_gqa_rope_padded_blackbox_inputs(query, key, value, cos, sin);
    required_chunked_prepared_padded_blackbox_attention_impl_unchecked(
        query,
        key,
        value,
        query_chunk_size,
        4,
        1,
        1,
    )
}

/// Opt-in p4/kv1/q1 candidate with balanced Q/K RMSNorm followed by narrow preparation fusion.
///
/// Q and K normalization run as separate row-parallel dispatches. Their reduction, divide, and
/// gamma arithmetic remain in F32 until the final F16 output store; the established narrow kernel
/// then handles RoPE, query scaling, grouped-head expansion, and padding without coupling that
/// work to the reduction workgroups.
#[cfg(feature = "wgpu")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn required_chunked_gqa_wgpu_balanced_strict_qk_norm_rope_padded_blackbox_attention(
    query: Tensor<NativeWgpuBackend, 4>,
    key: Tensor<NativeWgpuBackend, 4>,
    value: Tensor<NativeWgpuBackend, 4>,
    query_gamma: Tensor<NativeWgpuBackend, 1>,
    key_gamma: Tensor<NativeWgpuBackend, 1>,
    cos: Tensor<NativeWgpuBackend, 3>,
    sin: Tensor<NativeWgpuBackend, 3>,
    query_epsilon: f64,
    key_epsilon: f64,
    query_chunk_size: usize,
) -> Tensor<NativeWgpuBackend, 4> {
    let query = balanced_strict_rms_norm(query, query_gamma, query_epsilon);
    let key = balanced_strict_rms_norm(key, key_gamma, key_epsilon);
    required_chunked_gqa_wgpu_fused_rope_padded_blackbox_attention(
        query,
        key,
        value,
        cos,
        sin,
        query_chunk_size,
    )
}

/// Opt-in p4/kv1/q1 candidate that folds strict-F32 Q/K RMSNorm and RoPE into GQA preparation.
///
/// This route is intentionally not used by the released denoiser aliases. It exists behind the
/// attention-kernel type marker so real-checkpoint numerical and performance gates can validate
/// it before policy promotion. Unlike the ordinary preparation route, its inputs are raw projected
/// F16 Q/K plus their RMSNorm weights and the repeated-pair RoPE tables.
#[cfg(feature = "wgpu")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn required_chunked_gqa_wgpu_fused_strict_qk_norm_rope_padded_blackbox_attention(
    query: Tensor<NativeWgpuBackend, 4>,
    key: Tensor<NativeWgpuBackend, 4>,
    value: Tensor<NativeWgpuBackend, 4>,
    query_gamma: Tensor<NativeWgpuBackend, 1>,
    key_gamma: Tensor<NativeWgpuBackend, 1>,
    cos: Tensor<NativeWgpuBackend, 3>,
    sin: Tensor<NativeWgpuBackend, 3>,
    query_epsilon: f64,
    key_epsilon: f64,
    query_chunk_size: usize,
) -> Tensor<NativeWgpuBackend, 4> {
    let (query, key, value) = prepare_gqa_strict_norm_rope_padded_blackbox_inputs(
        query,
        key,
        value,
        query_gamma,
        key_gamma,
        cos,
        sin,
        query_epsilon,
        key_epsilon,
    );
    required_chunked_prepared_padded_blackbox_attention_impl_unchecked(
        query,
        key,
        value,
        query_chunk_size,
        4,
        1,
        1,
    )
}

fn required_chunked_flash_unit_attention_impl<B>(
    query: Tensor<Fusion<B>, 4>,
    key: Tensor<Fusion<B>, 4>,
    value: Tensor<Fusion<B>, 4>,
    query_chunk_size: usize,
) -> Tensor<Fusion<B>, 4>
where
    B: RequiredFlashUnitBackend + 'static,
{
    assert!(
        query_chunk_size > 0,
        "FlashUnit query chunk must be non-zero"
    );
    let [batch, heads, query_len, head_dim] = query.dims();
    assert!(query_len > 0, "FlashUnit query sequence must be non-empty");

    let mut outputs = Vec::with_capacity(query_len.div_ceil(query_chunk_size));
    for start in (0..query_len).step_by(query_chunk_size) {
        let end = start.saturating_add(query_chunk_size).min(query_len);
        let query = query
            .clone()
            .slice([0..batch, 0..heads, start..end, 0..head_dim]);
        outputs.push(required_flash_unit_attention_impl(
            query,
            key.clone(),
            value.clone(),
        ));
    }
    merge_query_chunks(outputs)
}

fn required_chunked_padded_blackbox_attention_impl<B>(
    query: Tensor<Fusion<B>, 4>,
    key: Tensor<Fusion<B>, 4>,
    value: Tensor<Fusion<B>, 4>,
    query_chunk_size: usize,
    num_planes: u8,
    seq_kv_tiles: u8,
    seq_q_tiles: u8,
) -> Tensor<Fusion<B>, 4>
where
    B: RequiredFlashUnitBackend + 'static,
{
    assert_supported_blackbox_partition_configuration(num_planes, seq_kv_tiles, seq_q_tiles);
    required_chunked_padded_blackbox_attention_impl_unchecked(
        query,
        key,
        value,
        query_chunk_size,
        num_planes,
        seq_kv_tiles,
        seq_q_tiles,
    )
}

fn required_chunked_padded_blackbox_attention_impl_unchecked<B>(
    query: Tensor<Fusion<B>, 4>,
    key: Tensor<Fusion<B>, 4>,
    value: Tensor<Fusion<B>, 4>,
    query_chunk_size: usize,
    num_planes: u8,
    seq_kv_tiles: u8,
    seq_q_tiles: u8,
) -> Tensor<Fusion<B>, 4>
where
    B: RequiredFlashUnitBackend + 'static,
{
    assert!(
        query_chunk_size > 0,
        "padded blackbox query chunk must be non-zero"
    );
    assert_eq!(
        query.dtype(),
        DType::F16,
        "padded blackbox query must be F16"
    );
    assert_eq!(key.dtype(), DType::F16, "padded blackbox key must be F16");
    assert_eq!(
        value.dtype(),
        DType::F16,
        "padded blackbox value must be F16"
    );

    let (query, key, value) = pad_blackbox_attention_inputs(query, key, value);
    required_chunked_prepared_padded_blackbox_attention_impl_unchecked(
        query,
        key,
        value,
        query_chunk_size,
        num_planes,
        seq_kv_tiles,
        seq_q_tiles,
    )
}

#[cfg(feature = "wgpu")]
fn required_chunked_gqa_padded_blackbox_attention_impl<B>(
    query: Tensor<Fusion<B>, 4>,
    key: Tensor<Fusion<B>, 4>,
    value: Tensor<Fusion<B>, 4>,
    query_chunk_size: usize,
    num_planes: u8,
    seq_kv_tiles: u8,
    seq_q_tiles: u8,
) -> Tensor<Fusion<B>, 4>
where
    B: RequiredFlashUnitBackend + 'static,
{
    assert_supported_blackbox_partition_configuration(num_planes, seq_kv_tiles, seq_q_tiles);
    required_chunked_gqa_padded_blackbox_attention_impl_unchecked(
        query,
        key,
        value,
        query_chunk_size,
        num_planes,
        seq_kv_tiles,
        seq_q_tiles,
    )
}

#[cfg(feature = "wgpu")]
fn required_chunked_gqa_padded_blackbox_attention_impl_unchecked<B>(
    query: Tensor<Fusion<B>, 4>,
    key: Tensor<Fusion<B>, 4>,
    value: Tensor<Fusion<B>, 4>,
    query_chunk_size: usize,
    num_planes: u8,
    seq_kv_tiles: u8,
    seq_q_tiles: u8,
) -> Tensor<Fusion<B>, 4>
where
    B: RequiredFlashUnitBackend + 'static,
{
    assert!(
        query_chunk_size > 0,
        "padded blackbox query chunk must be non-zero"
    );
    assert_eq!(
        query.dtype(),
        DType::F16,
        "padded blackbox query must be F16"
    );
    assert_eq!(key.dtype(), DType::F16, "padded blackbox key must be F16");
    assert_eq!(
        value.dtype(),
        DType::F16,
        "padded blackbox value must be F16"
    );

    let (query, key, value) = prepare_gqa_padded_blackbox_inputs(query, key, value);
    required_chunked_prepared_padded_blackbox_attention_impl_unchecked(
        query,
        key,
        value,
        query_chunk_size,
        num_planes,
        seq_kv_tiles,
        seq_q_tiles,
    )
}

fn required_chunked_prepared_padded_blackbox_attention_impl_unchecked<B>(
    query: Tensor<Fusion<B>, 4>,
    key: Tensor<Fusion<B>, 4>,
    value: Tensor<Fusion<B>, 4>,
    query_chunk_size: usize,
    num_planes: u8,
    seq_kv_tiles: u8,
    seq_q_tiles: u8,
) -> Tensor<Fusion<B>, 4>
where
    B: RequiredFlashUnitBackend + 'static,
{
    let [batch, heads, query_len, padded_head_dim] = query.dims();
    let padded_value_dim = value.dims()[3];
    let mut outputs = Vec::with_capacity(query_len.div_ceil(query_chunk_size));
    for start in (0..query_len).step_by(query_chunk_size) {
        let end = start.saturating_add(query_chunk_size).min(query_len);
        let query = query
            .clone()
            .slice([0..batch, 0..heads, start..end, 0..padded_head_dim]);
        let true_query_len = end - start;
        let query =
            pad_blackbox_query_sequence_partitioned_unchecked(query, num_planes, seq_q_tiles);
        let output = required_padded_blackbox_attention_impl_unchecked(
            query,
            key.clone(),
            value.clone(),
            num_planes,
            seq_kv_tiles,
            seq_q_tiles,
        );
        outputs.push(output.slice([0..batch, 0..heads, 0..true_query_len, 0..padded_value_dim]));
    }
    merge_query_chunks(outputs).slice([
        0..batch,
        0..heads,
        0..query_len,
        0..BOOGU_ATTENTION_HEAD_DIM.min(padded_value_dim),
    ])
}

fn merge_query_chunks<B: Backend>(outputs: Vec<Tensor<B, 4>>) -> Tensor<B, 4> {
    merge_query_chunks_with(outputs, |outputs| Tensor::cat(outputs, 2))
}

fn merge_query_chunks_with<T>(mut outputs: Vec<T>, merge: impl FnOnce(Vec<T>) -> T) -> T {
    assert!(
        !outputs.is_empty(),
        "native attention must produce at least one query chunk"
    );
    if outputs.len() == 1 {
        outputs
            .pop()
            .expect("one native attention query chunk was checked")
    } else {
        merge(outputs)
    }
}

#[cfg(test)]
fn pad_blackbox_query_sequence_partitioned<B: Backend>(
    query: Tensor<B, 4>,
    num_planes: u8,
    seq_q_tiles: u8,
) -> Tensor<B, 4> {
    assert_supported_blackbox_num_planes(num_planes);
    assert_supported_blackbox_seq_q_tiles(seq_q_tiles);
    pad_blackbox_query_sequence_partitioned_unchecked(query, num_planes, seq_q_tiles)
}

fn pad_blackbox_query_sequence_partitioned_unchecked<B: Backend>(
    query: Tensor<B, 4>,
    num_planes: u8,
    seq_q_tiles: u8,
) -> Tensor<B, 4> {
    let [batch, heads, query_len, head_dim] = query.dims();
    assert!(query_len > 0, "padded blackbox query must be non-empty");

    // The forced blueprint uses 16-row score tiles. Each plane owns `seq_q_tiles` query tiles, so
    // the complete stage width must divide every submitted query sequence exactly.
    let stage_multiple = 16 * usize::from(num_planes) * usize::from(seq_q_tiles);
    let padded_query_len = query_len.next_multiple_of(stage_multiple);
    if padded_query_len == query_len {
        return query;
    }

    let dtype = query.dtype();
    let device = query.device();
    let padding = Tensor::<B, 4>::zeros(
        [batch, heads, padded_query_len - query_len, head_dim],
        (&device, dtype),
    );
    Tensor::cat(vec![query, padding], 2)
}

fn required_padded_blackbox_attention_impl_unchecked<B>(
    query: Tensor<Fusion<B>, 4>,
    key: Tensor<Fusion<B>, 4>,
    value: Tensor<Fusion<B>, 4>,
    num_planes: u8,
    seq_kv_tiles: u8,
    seq_q_tiles: u8,
) -> Tensor<Fusion<B>, 4>
where
    B: RequiredFlashUnitBackend + 'static,
{
    let [batch, heads, query_len, head_dim] = query.dims();
    let [key_batch, key_heads, key_len, key_head_dim] = key.dims();
    let [value_batch, value_heads, value_len, value_dim] = value.dims();

    assert!(query_len > 0, "padded blackbox query must be non-empty");
    assert!(key_len > 0, "padded blackbox key must be non-empty");
    assert_eq!(key_batch, batch, "padded blackbox query/key batch mismatch");
    assert_eq!(
        value_batch, batch,
        "padded blackbox query/value batch mismatch"
    );
    assert_eq!(key_heads, heads, "padded blackbox query/key head mismatch");
    assert_eq!(
        value_heads, heads,
        "padded blackbox query/value head mismatch"
    );
    assert_eq!(
        head_dim, PADDED_BLACKBOX_HEAD_DIM,
        "padded blackbox query width must be 128"
    );
    assert_eq!(
        key_head_dim, PADDED_BLACKBOX_HEAD_DIM,
        "padded blackbox key width must be 128"
    );
    assert_eq!(
        value_dim, PADDED_BLACKBOX_HEAD_DIM,
        "padded blackbox value width must be 128"
    );
    assert_eq!(
        value_len, key_len,
        "padded blackbox key/value length mismatch"
    );
    assert_eq!(
        query.dtype(),
        DType::F16,
        "padded blackbox query must be F16"
    );
    assert_eq!(key.dtype(), DType::F16, "padded blackbox key must be F16");
    assert_eq!(
        value.dtype(),
        DType::F16,
        "padded blackbox value must be F16"
    );

    let query = require_float_primitive(query, "padded blackbox query");
    let key = require_float_primitive(key, "padded blackbox key");
    let value = require_float_primitive(value, "padded blackbox value");
    let client = query.client.clone();
    let streams = OperationStreams::with_inputs([&query, &key, &value]);
    let out = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([batch, heads, query_len, value_dim]),
        DType::F16,
    );
    let desc = CustomOpIr::new(
        "boogu_required_padded_blackbox_attention",
        &[query.into_ir(), key.into_ir(), value.into_ir()],
        &[out],
    );
    let output: FusionTensor<_> = client
        .register(
            streams,
            OperationIr::Custom(desc.clone()),
            PaddedBlackboxOperation::<B> {
                desc,
                num_planes,
                seq_kv_tiles,
                seq_q_tiles,
                _backend: PhantomData,
            },
        )
        .output();

    Tensor::from_primitive(TensorPrimitive::Float(output))
}

#[cfg(feature = "wgpu")]
fn prepare_gqa_padded_blackbox_inputs<B>(
    query: Tensor<Fusion<B>, 4>,
    key: Tensor<Fusion<B>, 4>,
    value: Tensor<Fusion<B>, 4>,
) -> FusionQkv<B>
where
    B: RequiredFlashUnitBackend + 'static,
{
    assert_gqa_blackbox_input_shapes(&query, &key, &value);
    let [batch, query_heads, query_len, _] = query.dims();
    let key_len = key.dims()[2];
    let value_len = value.dims()[2];
    let query = require_float_primitive(query, "GQA padded blackbox query");
    let key = require_float_primitive(key, "GQA padded blackbox key");
    let value = require_float_primitive(value, "GQA padded blackbox value");
    let client = query.client.clone();
    let streams = OperationStreams::with_inputs([&query, &key, &value]);
    let padded_query = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([batch, query_heads, query_len, PADDED_BLACKBOX_HEAD_DIM]),
        DType::F16,
    );
    let padded_key = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([batch, query_heads, key_len, PADDED_BLACKBOX_HEAD_DIM]),
        DType::F16,
    );
    let padded_value = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([batch, query_heads, value_len, PADDED_BLACKBOX_HEAD_DIM]),
        DType::F16,
    );
    let desc = CustomOpIr::new(
        "boogu_prepare_gqa_padded_blackbox_qkv",
        &[query.into_ir(), key.into_ir(), value.into_ir()],
        &[padded_query, padded_key, padded_value],
    );
    let [query, key, value]: [FusionTensor<_>; 3] = client
        .register(
            streams,
            OperationIr::Custom(desc.clone()),
            GqaPaddedQkvOperation::<B> {
                desc,
                _backend: PhantomData,
            },
        )
        .outputs();

    (
        Tensor::from_primitive(TensorPrimitive::Float(query)),
        Tensor::from_primitive(TensorPrimitive::Float(key)),
        Tensor::from_primitive(TensorPrimitive::Float(value)),
    )
}

#[cfg(feature = "wgpu")]
fn prepare_gqa_rope_padded_blackbox_inputs<B>(
    query: Tensor<Fusion<B>, 4>,
    key: Tensor<Fusion<B>, 4>,
    value: Tensor<Fusion<B>, 4>,
    cos: Tensor<Fusion<B>, 3>,
    sin: Tensor<Fusion<B>, 3>,
) -> FusionQkv<B>
where
    B: RequiredFlashUnitBackend + 'static,
{
    assert_gqa_blackbox_input_shapes(&query, &key, &value);
    let [batch, query_heads, query_len, _] = query.dims();
    let key_len = key.dims()[2];
    let value_len = value.dims()[2];
    assert_eq!(
        key_len, query_len,
        "fused RoPE+GQA padding preparation requires equal Q/K sequence lengths"
    );
    let [rope_batch, rope_len, rope_dim] = cos.dims();
    assert!(
        rope_batch == 1 || rope_batch == batch,
        "fused RoPE batch must be one or match Q/K batch"
    );
    assert_eq!(rope_len, query_len, "fused RoPE sequence length mismatch");
    assert_eq!(
        rope_dim, BOOGU_ATTENTION_HEAD_DIM,
        "fused RoPE width mismatch"
    );
    assert_eq!(sin.dims(), cos.dims(), "fused RoPE cos/sin shape mismatch");
    for (name, dtype) in [
        ("query", query.dtype()),
        ("key", key.dtype()),
        ("value", value.dtype()),
        ("RoPE cosine", cos.dtype()),
        ("RoPE sine", sin.dtype()),
    ] {
        assert_eq!(
            dtype,
            DType::F16,
            "fused RoPE+GQA padding {name} must be F16"
        );
    }

    let query = require_float_primitive(query, "normalized fused-RoPE GQA query");
    let key = require_float_primitive(key, "normalized fused-RoPE GQA key");
    let value = require_float_primitive(value, "fused-RoPE GQA value");
    let cos = require_float_primitive(cos, "fused RoPE cosine");
    let sin = require_float_primitive(sin, "fused RoPE sine");
    let client = query.client.clone();
    let streams = OperationStreams::with_inputs([&query, &key, &value, &cos, &sin]);
    let padded_query = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([batch, query_heads, query_len, PADDED_BLACKBOX_HEAD_DIM]),
        DType::F16,
    );
    let padded_key = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([batch, query_heads, key_len, PADDED_BLACKBOX_HEAD_DIM]),
        DType::F16,
    );
    let padded_value = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([batch, query_heads, value_len, PADDED_BLACKBOX_HEAD_DIM]),
        DType::F16,
    );
    let desc = CustomOpIr::new(
        "boogu_prepare_gqa_rope_padded_blackbox_qkv",
        &[
            query.into_ir(),
            key.into_ir(),
            value.into_ir(),
            cos.into_ir(),
            sin.into_ir(),
        ],
        &[padded_query, padded_key, padded_value],
    );
    let [query, key, value]: [FusionTensor<_>; 3] = client
        .register(
            streams,
            OperationIr::Custom(desc.clone()),
            GqaRopePaddedQkvOperation::<B> {
                desc,
                _backend: PhantomData,
            },
        )
        .outputs();

    (
        Tensor::from_primitive(TensorPrimitive::Float(query)),
        Tensor::from_primitive(TensorPrimitive::Float(key)),
        Tensor::from_primitive(TensorPrimitive::Float(value)),
    )
}

#[cfg(feature = "wgpu")]
fn balanced_strict_rms_norm<B>(
    input: Tensor<Fusion<B>, 4>,
    gamma: Tensor<Fusion<B>, 1>,
    epsilon: f64,
) -> Tensor<Fusion<B>, 4>
where
    B: RequiredFlashUnitBackend + 'static,
{
    let shape = input.dims();
    assert!(
        shape[0] > 0 && shape[1] > 0 && shape[2] > 0,
        "balanced strict RMSNorm input must have non-empty rows"
    );
    assert_eq!(
        shape[3], BOOGU_ATTENTION_HEAD_DIM,
        "balanced strict RMSNorm input width must be 120"
    );
    assert_eq!(
        gamma.dims(),
        [BOOGU_ATTENTION_HEAD_DIM],
        "balanced strict RMSNorm gamma width mismatch"
    );
    assert!(
        epsilon.is_finite() && epsilon > 0.0,
        "balanced strict RMSNorm epsilon must be finite and positive"
    );
    assert_eq!(
        input.dtype(),
        DType::F16,
        "balanced strict RMSNorm input must be F16"
    );
    assert_eq!(
        gamma.dtype(),
        DType::F16,
        "balanced strict RMSNorm gamma must be F16"
    );

    let input = require_float_primitive(input, "balanced strict RMSNorm input");
    let gamma = require_float_primitive(gamma, "balanced strict RMSNorm gamma");
    let client = input.client.clone();
    let streams = OperationStreams::with_inputs([&input, &gamma]);
    let output = TensorIr::uninit(client.create_empty_handle(), Shape::new(shape), DType::F16);
    let desc = CustomOpIr::new(
        "boogu_balanced_strict_rms_norm",
        &[input.into_ir(), gamma.into_ir()],
        &[output],
    );
    let output: FusionTensor<_> = client
        .register(
            streams,
            OperationIr::Custom(desc.clone()),
            BalancedStrictRmsNormOperation::<B> {
                desc,
                epsilon,
                _backend: PhantomData,
            },
        )
        .output();

    Tensor::from_primitive(TensorPrimitive::Float(output))
}

#[cfg(feature = "wgpu")]
#[allow(clippy::too_many_arguments)]
fn prepare_gqa_strict_norm_rope_padded_blackbox_inputs<B>(
    query: Tensor<Fusion<B>, 4>,
    key: Tensor<Fusion<B>, 4>,
    value: Tensor<Fusion<B>, 4>,
    query_gamma: Tensor<Fusion<B>, 1>,
    key_gamma: Tensor<Fusion<B>, 1>,
    cos: Tensor<Fusion<B>, 3>,
    sin: Tensor<Fusion<B>, 3>,
    query_epsilon: f64,
    key_epsilon: f64,
) -> FusionQkv<B>
where
    B: RequiredFlashUnitBackend + 'static,
{
    assert_gqa_blackbox_input_shapes(&query, &key, &value);
    let [batch, query_heads, query_len, _] = query.dims();
    let key_len = key.dims()[2];
    let value_len = value.dims()[2];
    assert_eq!(
        key_len, query_len,
        "fused strict Q/K norm+RoPE preparation requires equal Q/K sequence lengths"
    );
    assert_eq!(
        query_gamma.dims(),
        [BOOGU_ATTENTION_HEAD_DIM],
        "fused query RMSNorm gamma width mismatch"
    );
    assert_eq!(
        key_gamma.dims(),
        [BOOGU_ATTENTION_HEAD_DIM],
        "fused key RMSNorm gamma width mismatch"
    );
    let [rope_batch, rope_len, rope_dim] = cos.dims();
    assert!(
        rope_batch == 1 || rope_batch == batch,
        "fused RoPE batch must be one or match Q/K batch"
    );
    assert_eq!(rope_len, query_len, "fused RoPE sequence length mismatch");
    assert_eq!(
        rope_dim, BOOGU_ATTENTION_HEAD_DIM,
        "fused RoPE width mismatch"
    );
    assert_eq!(sin.dims(), cos.dims(), "fused RoPE cos/sin shape mismatch");
    assert!(
        query_epsilon.is_finite() && query_epsilon > 0.0,
        "fused query RMSNorm epsilon must be finite and positive"
    );
    assert!(
        key_epsilon.is_finite() && key_epsilon > 0.0,
        "fused key RMSNorm epsilon must be finite and positive"
    );
    for (name, dtype) in [
        ("query", query.dtype()),
        ("key", key.dtype()),
        ("value", value.dtype()),
        ("query gamma", query_gamma.dtype()),
        ("key gamma", key_gamma.dtype()),
        ("RoPE cosine", cos.dtype()),
        ("RoPE sine", sin.dtype()),
    ] {
        assert_eq!(
            dtype,
            DType::F16,
            "fused strict Q/K norm+RoPE {name} must be F16"
        );
    }

    let query = require_float_primitive(query, "fused raw GQA query");
    let key = require_float_primitive(key, "fused raw GQA key");
    let value = require_float_primitive(value, "fused raw GQA value");
    let query_gamma = require_float_primitive(query_gamma, "fused query RMSNorm gamma");
    let key_gamma = require_float_primitive(key_gamma, "fused key RMSNorm gamma");
    let cos = require_float_primitive(cos, "fused RoPE cosine");
    let sin = require_float_primitive(sin, "fused RoPE sine");
    let client = query.client.clone();
    let streams =
        OperationStreams::with_inputs([&query, &key, &value, &query_gamma, &key_gamma, &cos, &sin]);
    let padded_query = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([batch, query_heads, query_len, PADDED_BLACKBOX_HEAD_DIM]),
        DType::F16,
    );
    let padded_key = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([batch, query_heads, key_len, PADDED_BLACKBOX_HEAD_DIM]),
        DType::F16,
    );
    let padded_value = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([batch, query_heads, value_len, PADDED_BLACKBOX_HEAD_DIM]),
        DType::F16,
    );
    let desc = CustomOpIr::new(
        "boogu_prepare_gqa_strict_norm_rope_padded_blackbox_qkv",
        &[
            query.into_ir(),
            key.into_ir(),
            value.into_ir(),
            query_gamma.into_ir(),
            key_gamma.into_ir(),
            cos.into_ir(),
            sin.into_ir(),
        ],
        &[padded_query, padded_key, padded_value],
    );
    let [query, key, value]: [FusionTensor<_>; 3] = client
        .register(
            streams,
            OperationIr::Custom(desc.clone()),
            GqaStrictNormRopePaddedQkvOperation::<B> {
                desc,
                query_epsilon,
                key_epsilon,
                _backend: PhantomData,
            },
        )
        .outputs();

    (
        Tensor::from_primitive(TensorPrimitive::Float(query)),
        Tensor::from_primitive(TensorPrimitive::Float(key)),
        Tensor::from_primitive(TensorPrimitive::Float(value)),
    )
}

#[cfg(any(feature = "wgpu", test))]
fn assert_gqa_blackbox_input_shapes<B: Backend>(
    query: &Tensor<B, 4>,
    key: &Tensor<B, 4>,
    value: &Tensor<B, 4>,
) {
    let [batch, query_heads, query_len, head_dim] = query.dims();
    let [key_batch, key_value_heads, key_len, key_head_dim] = key.dims();
    let [value_batch, value_heads, value_len, value_dim] = value.dims();
    assert!(query_len > 0, "GQA padded blackbox query must be non-empty");
    assert!(key_len > 0, "GQA padded blackbox key must be non-empty");
    assert!(
        query_heads > 0,
        "GQA padded blackbox query heads must be non-zero"
    );
    assert!(
        key_value_heads > 0,
        "GQA padded blackbox key/value heads must be non-zero"
    );
    assert_eq!(
        query_heads % key_value_heads,
        0,
        "GQA padded blackbox query heads must be divisible by key/value heads"
    );
    assert_eq!(
        key_batch, batch,
        "GQA padded blackbox query/key batch mismatch"
    );
    assert_eq!(
        value_batch, batch,
        "GQA padded blackbox query/value batch mismatch"
    );
    assert_eq!(
        value_heads, key_value_heads,
        "GQA padded blackbox key/value head mismatch"
    );
    assert_eq!(
        value_len, key_len,
        "GQA padded blackbox key/value length mismatch"
    );
    assert_eq!(
        head_dim, BOOGU_ATTENTION_HEAD_DIM,
        "Boogu GQA blackbox query width must be 120"
    );
    assert_eq!(
        key_head_dim, BOOGU_ATTENTION_HEAD_DIM,
        "Boogu GQA blackbox key width must be 120"
    );
    assert_eq!(
        value_dim, BOOGU_ATTENTION_HEAD_DIM,
        "Boogu GQA blackbox value width must be 120"
    );
}

#[cfg(test)]
fn reference_prepare_gqa_padded_blackbox_inputs<B: Backend>(
    query: Tensor<B, 4>,
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
    assert_gqa_blackbox_input_shapes(&query, &key, &value);
    let [batch, query_heads, _, head_dim] = query.dims();
    let [_, key_value_heads, key_len, _] = key.dims();
    let groups = query_heads / key_value_heads;
    let key = key
        .reshape([batch, key_value_heads, 1, key_len, head_dim])
        .repeat_dim(2, groups)
        .reshape([batch, query_heads, key_len, head_dim]);
    let value = value
        .reshape([batch, key_value_heads, 1, key_len, head_dim])
        .repeat_dim(2, groups)
        .reshape([batch, query_heads, key_len, head_dim]);
    pad_blackbox_attention_inputs(query, key, value)
}

#[cfg(test)]
fn reference_prepare_gqa_rope_padded_blackbox_inputs<B: Backend>(
    query: Tensor<B, 4>,
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
    cos: Tensor<B, 3>,
    sin: Tensor<B, 3>,
) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
    assert_gqa_blackbox_input_shapes(&query, &key, &value);
    let [batch, _, query_len, _] = query.dims();
    assert_eq!(
        key.dims()[2],
        query_len,
        "fused RoPE+GQA padding preparation requires equal Q/K sequence lengths"
    );
    let [rope_batch, rope_len, rope_dim] = cos.dims();
    assert!(rope_batch == 1 || rope_batch == batch);
    assert_eq!(rope_len, query_len);
    assert_eq!(rope_dim, BOOGU_ATTENTION_HEAD_DIM);
    assert_eq!(sin.dims(), cos.dims());
    let apply_repeated_pair_rope = |input: Tensor<B, 4>| {
        let [batch, tokens, heads, head_dim] = input.dims();
        let pairs = head_dim / 2;
        let paired = input.clone().reshape([batch, tokens, heads, pairs, 2]);
        let real = paired
            .clone()
            .slice([0..batch, 0..tokens, 0..heads, 0..pairs, 0..1]);
        let imag = paired.slice([0..batch, 0..tokens, 0..heads, 0..pairs, 1..2]);
        let rotated =
            Tensor::cat(vec![imag.neg(), real], 4).reshape([batch, tokens, heads, head_dim]);
        input * cos.clone().unsqueeze_dim(2) + rotated * sin.clone().unsqueeze_dim(2)
    };
    let query = apply_repeated_pair_rope(query.permute([0, 2, 1, 3])).permute([0, 2, 1, 3]);
    let key = apply_repeated_pair_rope(key.permute([0, 2, 1, 3])).permute([0, 2, 1, 3]);
    reference_prepare_gqa_padded_blackbox_inputs(query, key, value)
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn reference_prepare_gqa_strict_norm_rope_padded_blackbox_inputs<B: Backend>(
    query: Tensor<B, 4>,
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
    query_gamma: Tensor<B, 1>,
    key_gamma: Tensor<B, 1>,
    cos: Tensor<B, 3>,
    sin: Tensor<B, 3>,
    query_epsilon: f64,
    key_epsilon: f64,
) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
    assert_gqa_blackbox_input_shapes(&query, &key, &value);
    let strict_rms_norm = |input: Tensor<B, 4>, gamma: Tensor<B, 1>, epsilon: f64| {
        let dtype = input.dtype();
        let rms = (input.clone().cast(DType::F32).square().mean_dim(3) + epsilon).sqrt();
        input / rms.cast(dtype) * gamma.unsqueeze()
    };
    let query = strict_rms_norm(query, query_gamma, query_epsilon);
    let key = strict_rms_norm(key, key_gamma, key_epsilon);
    reference_prepare_gqa_rope_padded_blackbox_inputs(query, key, value, cos, sin)
}

#[cfg(test)]
fn reference_balanced_strict_rms_norm<B: Backend>(
    input: Tensor<B, 4>,
    gamma: Tensor<B, 1>,
    epsilon: f64,
) -> Tensor<B, 4> {
    let dtype = input.dtype();
    let input = input.cast(DType::F32);
    let rms = (input.clone().square().mean_dim(3) + epsilon).sqrt();
    (input / rms * gamma.cast(DType::F32).unsqueeze()).cast(dtype)
}

fn pad_blackbox_attention_inputs<B: Backend>(
    query: Tensor<B, 4>,
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
    let [batch, heads, query_len, head_dim] = query.dims();
    let [key_batch, key_heads, key_len, key_head_dim] = key.dims();
    let [value_batch, value_heads, value_len, value_dim] = value.dims();
    assert!(query_len > 0, "padded blackbox query must be non-empty");
    assert!(key_len > 0, "padded blackbox key must be non-empty");
    assert_eq!(key_batch, batch, "padded blackbox query/key batch mismatch");
    assert_eq!(
        value_batch, batch,
        "padded blackbox query/value batch mismatch"
    );
    assert_eq!(key_heads, heads, "padded blackbox query/key head mismatch");
    assert_eq!(
        value_heads, heads,
        "padded blackbox query/value head mismatch"
    );
    assert_eq!(
        head_dim, BOOGU_ATTENTION_HEAD_DIM,
        "Boogu blackbox query width must be 120"
    );
    assert_eq!(
        key_head_dim, BOOGU_ATTENTION_HEAD_DIM,
        "Boogu blackbox key width must be 120"
    );
    assert_eq!(
        value_dim, BOOGU_ATTENTION_HEAD_DIM,
        "Boogu blackbox value width must be 120"
    );
    assert_eq!(
        value_len, key_len,
        "padded blackbox key/value length mismatch"
    );

    let padding = PADDED_BLACKBOX_HEAD_DIM - BOOGU_ATTENTION_HEAD_DIM;
    let query_dtype = query.dtype();
    let key_dtype = key.dtype();
    let value_dtype = value.dtype();
    let query_device = query.device();
    let key_device = key.device();
    let value_device = value.device();
    let query_padding = Tensor::<B, 4>::zeros(
        [batch, heads, query_len, padding],
        (&query_device, query_dtype),
    );
    let key_padding =
        Tensor::<B, 4>::zeros([batch, heads, key_len, padding], (&key_device, key_dtype));
    let value_padding = Tensor::<B, 4>::zeros(
        [batch, heads, value_len, padding],
        (&value_device, value_dtype),
    );
    let query_scale = (PADDED_BLACKBOX_HEAD_DIM as f64 / BOOGU_ATTENTION_HEAD_DIM as f64).sqrt();

    (
        Tensor::cat(vec![query.mul_scalar(query_scale), query_padding], 3),
        Tensor::cat(vec![key, key_padding], 3),
        Tensor::cat(vec![value, value_padding], 3),
    )
}

pub(crate) fn assert_supported_blackbox_num_planes(num_planes: u8) {
    assert!(
        matches!(num_planes, 2 | 4),
        "padded blackbox num_planes must be one of the supported values 2 or 4"
    );
}

fn assert_supported_forced_blackbox_partition_configuration(
    num_planes: u8,
    seq_kv_tiles: u8,
    seq_q_tiles: u8,
) {
    assert!(
        matches!(num_planes, 2 | 4),
        "internal forced padded blackbox num_planes must be one of 2 or 4"
    );
    assert_supported_blackbox_seq_kv_tiles(seq_kv_tiles);
    assert_supported_blackbox_seq_q_tiles(seq_q_tiles);
    assert!(
        num_planes == 2 || seq_kv_tiles == 1,
        "internal forced padded blackbox multi-KV-tile configurations require two planes"
    );
}

#[cfg(feature = "wgpu")]
pub(crate) fn assert_supported_wgpu_blackbox_num_planes(num_planes: u8) {
    assert!(
        matches!(num_planes, 2 | 4),
        "native WGPU padded blackbox num_planes must be one of 2 or 4"
    );
}

pub(crate) fn assert_supported_blackbox_seq_kv_tiles(seq_kv_tiles: u8) {
    assert!(
        matches!(seq_kv_tiles, 1 | 2),
        "padded blackbox seq_kv_tiles must be one of the supported values 1 or 2"
    );
}

pub(crate) fn assert_supported_blackbox_seq_q_tiles(seq_q_tiles: u8) {
    assert!(
        seq_q_tiles == 1,
        "padded blackbox seq_q_tiles must be 1; two query tiles failed the native WGPU nonzero parity gate"
    );
}

pub(crate) fn assert_supported_blackbox_partition_configuration(
    num_planes: u8,
    seq_kv_tiles: u8,
    seq_q_tiles: u8,
) {
    assert_supported_blackbox_num_planes(num_planes);
    assert_supported_blackbox_seq_kv_tiles(seq_kv_tiles);
    assert_supported_blackbox_seq_q_tiles(seq_q_tiles);
    assert!(
        num_planes == 2 || seq_kv_tiles == 1,
        "padded blackbox four-plane/multi-KV-tile configurations failed the native WGPU nonzero parity gate"
    );
}

#[cfg(feature = "wgpu")]
pub(crate) fn assert_supported_wgpu_blackbox_partition_configuration(
    num_planes: u8,
    seq_kv_tiles: u8,
    seq_q_tiles: u8,
) {
    assert_supported_wgpu_blackbox_num_planes(num_planes);
    assert_supported_forced_blackbox_partition_configuration(num_planes, seq_kv_tiles, seq_q_tiles);
}

#[cfg(feature = "wgpu")]
pub(crate) fn assert_supported_wgpu_blackbox_configuration(num_planes: u8, seq_kv_tiles: u8) {
    assert_supported_wgpu_blackbox_partition_configuration(num_planes, seq_kv_tiles, 1);
}

fn required_flash_unit_attention_impl<B>(
    query: Tensor<Fusion<B>, 4>,
    key: Tensor<Fusion<B>, 4>,
    value: Tensor<Fusion<B>, 4>,
) -> Tensor<Fusion<B>, 4>
where
    B: RequiredFlashUnitBackend + 'static,
{
    let [batch, heads, query_len, head_dim] = query.dims();
    let [key_batch, key_heads, key_len, key_head_dim] = key.dims();
    let [value_batch, value_heads, value_len, value_dim] = value.dims();

    assert!(query_len > 0, "FlashUnit query sequence must be non-empty");
    assert!(key_len > 0, "FlashUnit key sequence must be non-empty");
    assert_eq!(key_batch, batch, "FlashUnit query/key batch mismatch");
    assert_eq!(value_batch, batch, "FlashUnit query/value batch mismatch");
    assert_eq!(key_heads, heads, "FlashUnit query/key head mismatch");
    assert_eq!(value_heads, heads, "FlashUnit query/value head mismatch");
    assert_eq!(key_head_dim, head_dim, "FlashUnit query/key width mismatch");
    assert_eq!(value_len, key_len, "FlashUnit key/value length mismatch");
    assert_eq!(query.dtype(), DType::F16, "FlashUnit query must be F16");
    assert_eq!(key.dtype(), DType::F16, "FlashUnit key must be F16");
    assert_eq!(value.dtype(), DType::F16, "FlashUnit value must be F16");

    let query = require_float_primitive(query, "query");
    let key = require_float_primitive(key, "key");
    let value = require_float_primitive(value, "value");
    let client = query.client.clone();
    let streams = OperationStreams::with_inputs([&query, &key, &value]);
    let out = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([batch, heads, query_len, value_dim]),
        DType::F16,
    );
    let desc = CustomOpIr::new(
        "boogu_required_flash_unit_attention",
        &[query.into_ir(), key.into_ir(), value.into_ir()],
        &[out],
    );
    let output: FusionTensor<_> = client
        .register(
            streams,
            OperationIr::Custom(desc.clone()),
            FlashUnitOperation::<B> {
                desc,
                _backend: PhantomData,
            },
        )
        .output();

    Tensor::from_primitive(TensorPrimitive::Float(output))
}

fn require_float_primitive<B, const D: usize>(
    tensor: Tensor<Fusion<B>, D>,
    name: &str,
) -> FusionTensor<B::FusionRuntime>
where
    B: RequiredFlashUnitBackend,
{
    match tensor.into_primitive() {
        TensorPrimitive::Float(tensor) => tensor,
        TensorPrimitive::QFloat(_) => {
            panic!("required native attention {name} activation must not be quantized")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BooguDenoiser, BooguDenoiserInput, BooguError, DmdDenoiser};
    use burn::tensor::{TensorData, module::attention};
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn deterministic_tensor<const D: usize>(
        shape: [usize; D],
        offset: usize,
        device: &burn_ndarray::NdArrayDevice,
    ) -> Tensor<TestBackend, D> {
        let elements = shape.iter().product();
        let values = (0..elements)
            .map(|index| {
                let integer = ((index + offset) * 19 + 7) % 53;
                (integer as f32 - 26.0) / 17.0
            })
            .collect::<Vec<_>>();
        Tensor::from_data(TensorData::new(values, shape), device)
    }

    #[test]
    fn single_query_chunk_bypasses_cat_and_matches_cat_correctness() {
        let device = Default::default();
        let chunk = deterministic_tensor([1, 2, 3, 4], 23, &device);
        let expected = Tensor::cat(vec![chunk.clone()], 2);
        let actual = merge_query_chunks_with(vec![chunk], |_| {
            panic!("one query chunk must bypass Tensor::cat")
        });

        assert_eq!(actual.dims(), expected.dims());
        assert_eq!(actual.into_data(), expected.into_data());
    }

    #[test]
    fn multiple_query_chunks_preserve_cat_order_correctness() {
        let device = Default::default();
        let first = deterministic_tensor([1, 2, 3, 4], 29, &device);
        let second = deterministic_tensor([1, 2, 2, 4], 31, &device);
        let expected = Tensor::cat(vec![first.clone(), second.clone()], 2);
        let actual = merge_query_chunks(vec![first, second]);

        assert_eq!(actual.dims(), [1, 2, 5, 4]);
        assert_eq!(actual.into_data(), expected.into_data());
    }

    #[test]
    fn padded_blackbox_transform_preserves_attention_math_correctness() {
        let device = Default::default();
        let query = deterministic_tensor([1, 2, 3, BOOGU_ATTENTION_HEAD_DIM], 1, &device);
        let key = deterministic_tensor([1, 2, 5, BOOGU_ATTENTION_HEAD_DIM], 5, &device);
        let value = deterministic_tensor([1, 2, 5, BOOGU_ATTENTION_HEAD_DIM], 9, &device);
        let expected = attention(
            query.clone(),
            key.clone(),
            value.clone(),
            None,
            None,
            AttentionModuleOptions::default(),
        );

        let (query, key, value) = pad_blackbox_attention_inputs(query, key, value);
        assert_eq!(query.dims(), [1, 2, 3, PADDED_BLACKBOX_HEAD_DIM]);
        assert_eq!(key.dims(), [1, 2, 5, PADDED_BLACKBOX_HEAD_DIM]);
        assert_eq!(value.dims(), [1, 2, 5, PADDED_BLACKBOX_HEAD_DIM]);
        let actual = attention(
            query,
            key,
            value,
            None,
            None,
            AttentionModuleOptions::default(),
        )
        .slice([0..1, 0..2, 0..3, 0..BOOGU_ATTENTION_HEAD_DIM]);
        let expected = expected
            .into_data()
            .to_vec::<f32>()
            .expect("original attention values");
        let actual = actual
            .into_data()
            .to_vec::<f32>()
            .expect("padded attention values");
        let max_abs = expected
            .iter()
            .zip(actual)
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs <= 2.0e-6,
            "padded blackbox transform max_abs={max_abs}"
        );
    }

    #[test]
    fn gqa_padded_blackbox_transform_preserves_attention_math_correctness() {
        let device = Default::default();
        let query = deterministic_tensor([1, 4, 3, BOOGU_ATTENTION_HEAD_DIM], 2, &device);
        let key = deterministic_tensor([1, 2, 5, BOOGU_ATTENTION_HEAD_DIM], 7, &device);
        let value = deterministic_tensor([1, 2, 5, BOOGU_ATTENTION_HEAD_DIM], 11, &device);
        let key_expanded = key
            .clone()
            .reshape([1, 2, 1, 5, BOOGU_ATTENTION_HEAD_DIM])
            .repeat_dim(2, 2)
            .reshape([1, 4, 5, BOOGU_ATTENTION_HEAD_DIM]);
        let value_expanded = value
            .clone()
            .reshape([1, 2, 1, 5, BOOGU_ATTENTION_HEAD_DIM])
            .repeat_dim(2, 2)
            .reshape([1, 4, 5, BOOGU_ATTENTION_HEAD_DIM]);
        let expected = attention(
            query.clone(),
            key_expanded,
            value_expanded,
            None,
            None,
            AttentionModuleOptions::default(),
        );

        let (query, key, value) = reference_prepare_gqa_padded_blackbox_inputs(query, key, value);
        assert_eq!(query.dims(), [1, 4, 3, PADDED_BLACKBOX_HEAD_DIM]);
        assert_eq!(key.dims(), [1, 4, 5, PADDED_BLACKBOX_HEAD_DIM]);
        assert_eq!(value.dims(), [1, 4, 5, PADDED_BLACKBOX_HEAD_DIM]);
        let actual = attention(
            query,
            key,
            value,
            None,
            None,
            AttentionModuleOptions::default(),
        )
        .slice([0..1, 0..4, 0..3, 0..BOOGU_ATTENTION_HEAD_DIM]);
        let max_abs = expected.sub(actual).abs().max().into_scalar();
        assert!(
            max_abs <= 2.0e-6,
            "GQA padded blackbox transform max_abs={max_abs}"
        );

        let invalid_key = deterministic_tensor([1, 3, 5, BOOGU_ATTENTION_HEAD_DIM], 7, &device);
        let invalid_value = deterministic_tensor([1, 3, 5, BOOGU_ATTENTION_HEAD_DIM], 11, &device);
        let invalid_query = deterministic_tensor([1, 4, 3, BOOGU_ATTENTION_HEAD_DIM], 2, &device);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                reference_prepare_gqa_padded_blackbox_inputs(
                    invalid_query,
                    invalid_key,
                    invalid_value,
                )
            }))
            .is_err(),
            "non-divisible GQA heads must fail closed"
        );
    }

    #[test]
    fn fused_rope_gqa_padding_prep_matches_composed_reference_correctness() {
        let device = Default::default();
        let query = deterministic_tensor([2, 4, 3, BOOGU_ATTENTION_HEAD_DIM], 23, &device);
        let key = deterministic_tensor([2, 1, 3, BOOGU_ATTENTION_HEAD_DIM], 29, &device);
        let value = deterministic_tensor([2, 1, 3, BOOGU_ATTENTION_HEAD_DIM], 31, &device);
        let phases = (0..3 * BOOGU_ATTENTION_HEAD_DIM)
            .map(|index| ((index / 2) % 17) as f32 / 19.0)
            .collect::<Vec<_>>();
        let phase = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(phases, [1, 3, BOOGU_ATTENTION_HEAD_DIM]),
            &device,
        );
        let cos = phase.clone().cos();
        let sin = phase.sin();

        let actual = reference_prepare_gqa_rope_padded_blackbox_inputs(
            query.clone(),
            key.clone(),
            value.clone(),
            cos.clone(),
            sin.clone(),
        );
        let rope = |input: Tensor<TestBackend, 4>| {
            let token_major = input.permute([0, 2, 1, 3]);
            let [batch, sequence, heads, width] = token_major.dims();
            let pairs = width / 2;
            let paired = token_major
                .clone()
                .reshape([batch, sequence, heads, pairs, 2]);
            let real = paired
                .clone()
                .slice([0..batch, 0..sequence, 0..heads, 0..pairs, 0..1]);
            let imag = paired.slice([0..batch, 0..sequence, 0..heads, 0..pairs, 1..2]);
            let rotated =
                Tensor::cat(vec![imag.neg(), real], 4).reshape([batch, sequence, heads, width]);
            (token_major * cos.clone().unsqueeze_dim(2) + rotated * sin.clone().unsqueeze_dim(2))
                .permute([0, 2, 1, 3])
        };
        let expected = reference_prepare_gqa_padded_blackbox_inputs(rope(query), rope(key), value);

        for (name, actual, expected) in [
            ("query", actual.0, expected.0),
            ("key", actual.1, expected.1),
            ("value", actual.2, expected.2),
        ] {
            assert_eq!(actual.dims(), expected.dims(), "{name} shape");
            let max_abs = actual.sub(expected).abs().max().into_scalar();
            assert!(max_abs <= 1.0e-6, "{name} fused prep max_abs={max_abs}");
        }
    }

    #[test]
    fn fused_strict_qk_norm_rope_prep_matches_composed_reference_correctness() {
        let device = Default::default();
        let query = deterministic_tensor([1, 4, 3, BOOGU_ATTENTION_HEAD_DIM], 31, &device);
        let key = deterministic_tensor([1, 2, 3, BOOGU_ATTENTION_HEAD_DIM], 37, &device);
        let value = deterministic_tensor([1, 2, 3, BOOGU_ATTENTION_HEAD_DIM], 41, &device);
        let query_gamma = deterministic_tensor([BOOGU_ATTENTION_HEAD_DIM], 43, &device)
            .mul_scalar(0.25)
            .add_scalar(1.0);
        let key_gamma = deterministic_tensor([BOOGU_ATTENTION_HEAD_DIM], 47, &device)
            .mul_scalar(0.25)
            .add_scalar(1.0);
        let phases = (0..3 * BOOGU_ATTENTION_HEAD_DIM)
            .map(|index| ((index / 2) % 17) as f32 / 19.0)
            .collect::<Vec<_>>();
        let phase = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(phases, [1, 3, BOOGU_ATTENTION_HEAD_DIM]),
            &device,
        );
        let cos = phase.clone().cos();
        let sin = phase.sin();
        let epsilon = 1.0e-5;

        let actual = reference_prepare_gqa_strict_norm_rope_padded_blackbox_inputs(
            query.clone(),
            key.clone(),
            value.clone(),
            query_gamma.clone(),
            key_gamma.clone(),
            cos.clone(),
            sin.clone(),
            epsilon,
            epsilon,
        );
        let normalize = |input: Tensor<TestBackend, 4>, gamma: Tensor<TestBackend, 1>| {
            let rms = (input.clone().square().mean_dim(3) + epsilon).sqrt();
            input / rms * gamma.unsqueeze()
        };
        let rope = |input: Tensor<TestBackend, 4>| {
            let [batch, sequence, heads, width] = input.dims();
            let pairs = width / 2;
            let paired = input.clone().reshape([batch, sequence, heads, pairs, 2]);
            let real = paired
                .clone()
                .slice([0..batch, 0..sequence, 0..heads, 0..pairs, 0..1]);
            let imag = paired.slice([0..batch, 0..sequence, 0..heads, 0..pairs, 1..2]);
            let rotated =
                Tensor::cat(vec![imag.neg(), real], 4).reshape([batch, sequence, heads, width]);
            input * cos.clone().unsqueeze_dim(2) + rotated * sin.clone().unsqueeze_dim(2)
        };
        let query = rope(normalize(query.permute([0, 2, 1, 3]), query_gamma)).permute([0, 2, 1, 3]);
        let key = rope(normalize(key.permute([0, 2, 1, 3]), key_gamma)).permute([0, 2, 1, 3]);
        let expected = reference_prepare_gqa_padded_blackbox_inputs(query, key, value);

        for (name, actual, expected) in [
            ("query", actual.0, expected.0),
            ("key", actual.1, expected.1),
            ("value", actual.2, expected.2),
        ] {
            assert_eq!(actual.dims(), expected.dims(), "{name} shape");
            let max_abs = actual.sub(expected).abs().max().into_scalar();
            assert!(max_abs <= 1.0e-6, "{name} fused prep max_abs={max_abs}");
        }
    }

    #[test]
    fn balanced_strict_rms_norm_f32_affine_math_matches_scalar_reference_correctness() {
        let device = Default::default();
        let shape = [2, 3, 4, BOOGU_ATTENTION_HEAD_DIM];
        let input_values = (0..shape.iter().product())
            .map(|index| {
                let integer = (index * 29 + 11) % 101;
                (integer as f32 - 50.0) / 23.0
            })
            .collect::<Vec<_>>();
        let gamma_values = (0..BOOGU_ATTENTION_HEAD_DIM)
            .map(|index| 0.75 + (index % 17) as f32 / 31.0)
            .collect::<Vec<_>>();
        let input = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(input_values.clone(), shape),
            &device,
        );
        let gamma = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(gamma_values.clone(), [BOOGU_ATTENTION_HEAD_DIM]),
            &device,
        );
        let epsilon = 1.0e-5;

        let actual = reference_balanced_strict_rms_norm(input, gamma, epsilon)
            .into_data()
            .to_vec::<f32>()
            .expect("balanced RMSNorm reference values");
        let expected = input_values
            .as_chunks::<BOOGU_ATTENTION_HEAD_DIM>()
            .0
            .iter()
            .flat_map(|row| {
                let rms = (row.iter().map(|value| value * value).sum::<f32>()
                    / BOOGU_ATTENTION_HEAD_DIM as f32
                    + epsilon as f32)
                    .sqrt();
                row.iter()
                    .zip(&gamma_values)
                    .map(move |(&value, &gamma)| value / rms * gamma)
            })
            .collect::<Vec<_>>();

        assert_eq!(actual.len(), expected.len());
        let max_abs = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_abs <= 2.0e-6, "balanced RMSNorm max_abs={max_abs}");
    }

    #[test]
    fn padded_blackbox_query_rows_preserve_real_outputs_correctness() {
        let device = Default::default();
        let query = deterministic_tensor([1, 1, 49, PADDED_BLACKBOX_HEAD_DIM], 3, &device);
        let key = deterministic_tensor([1, 1, 67, PADDED_BLACKBOX_HEAD_DIM], 8, &device);
        let value = deterministic_tensor([1, 1, 67, PADDED_BLACKBOX_HEAD_DIM], 13, &device);
        let expected = attention(
            query.clone(),
            key.clone(),
            value.clone(),
            None,
            None,
            AttentionModuleOptions::default(),
        );

        for (num_planes, seq_q_tiles) in [(2, 1), (4, 1)] {
            let padded_query =
                pad_blackbox_query_sequence_partitioned(query.clone(), num_planes, seq_q_tiles);
            assert_eq!(
                padded_query.dims()[2] % (16 * usize::from(num_planes) * usize::from(seq_q_tiles)),
                0
            );
            let actual = attention(
                padded_query,
                key.clone(),
                value.clone(),
                None,
                None,
                AttentionModuleOptions::default(),
            )
            .slice([0..1, 0..1, 0..49, 0..PADDED_BLACKBOX_HEAD_DIM]);
            let max_abs = expected.clone().sub(actual).abs().max().into_scalar();
            assert!(
                max_abs <= 1.0e-6,
                "padded query rows planes={num_planes} seq_q_tiles={seq_q_tiles} max_abs={max_abs}"
            );
        }
        assert!(
            std::panic::catch_unwind(|| assert_supported_blackbox_num_planes(8)).is_err(),
            "the generic public eight-plane policy must fail closed"
        );
        for seq_kv_tiles in [1, 2] {
            assert_supported_blackbox_seq_kv_tiles(seq_kv_tiles);
        }
        assert_supported_blackbox_seq_q_tiles(1);
        assert!(
            std::panic::catch_unwind(|| assert_supported_blackbox_seq_q_tiles(2)).is_err(),
            "the numerically invalid two-query-tile policy must fail closed"
        );
        assert!(
            std::panic::catch_unwind(|| assert_supported_blackbox_seq_kv_tiles(4)).is_err(),
            "the numerically invalid four-tile key/value partition must fail closed"
        );
        assert!(
            std::panic::catch_unwind(|| {
                assert_supported_blackbox_partition_configuration(4, 2, 1)
            })
            .is_err(),
            "the numerically invalid four-plane/two-KV-tile policy must fail closed"
        );
        assert!(
            std::panic::catch_unwind(|| {
                assert_supported_blackbox_partition_configuration(2, 1, 2)
            })
            .is_err(),
            "two query tiles with two planes must fail closed"
        );
    }

    #[cfg(feature = "wgpu")]
    #[test]
    #[allow(clippy::type_complexity)]
    fn native_wgpu_flash_unit_api_compiles_smoke() {
        fn assert_dmd_denoiser<D: DmdDenoiser<NativeWgpuBackend>>() {}
        assert_dmd_denoiser::<crate::NativeFlashUnitDenoiser>();

        let _forward: fn(
            &BooguDenoiser<NativeWgpuBackend>,
            BooguDenoiserInput<NativeWgpuBackend>,
        ) -> Result<Tensor<NativeWgpuBackend, 4>, BooguError> =
            BooguDenoiser::forward_native_flash_unit;
        let _chunked: fn(
            Tensor<NativeWgpuBackend, 4>,
            Tensor<NativeWgpuBackend, 4>,
            Tensor<NativeWgpuBackend, 4>,
            usize,
        ) -> Tensor<NativeWgpuBackend, 4> = required_chunked_flash_unit_attention;

        fn assert_blackbox_dmd_denoiser<D: DmdDenoiser<NativeWgpuBackend>>() {}
        assert_blackbox_dmd_denoiser::<crate::NativePaddedBlackboxDenoiser>();
        let _fused_prep_builder: fn(
            crate::NativePaddedBlackboxDenoiser,
            bool,
        ) -> crate::NativePaddedBlackboxDenoiser =
            crate::NativePaddedBlackboxDenoiser::with_fused_strict_qk_norm_rope;
        let _fused_prep_setter: fn(&mut crate::NativePaddedBlackboxDenoiser, bool) =
            crate::NativePaddedBlackboxDenoiser::set_fused_strict_qk_norm_rope;
        let _fused_prep_getter: fn(&crate::NativePaddedBlackboxDenoiser) -> bool =
            crate::NativePaddedBlackboxDenoiser::fused_strict_qk_norm_rope;
        let _fused_rope_gqa_padding_builder: fn(
            crate::NativePaddedBlackboxDenoiser,
            bool,
        ) -> crate::NativePaddedBlackboxDenoiser =
            crate::NativePaddedBlackboxDenoiser::with_fused_rope_gqa_padding;
        let _fused_rope_gqa_padding_setter: fn(&mut crate::NativePaddedBlackboxDenoiser, bool) =
            crate::NativePaddedBlackboxDenoiser::set_fused_rope_gqa_padding;
        let _fused_rope_gqa_padding_getter: fn(&crate::NativePaddedBlackboxDenoiser) -> bool =
            crate::NativePaddedBlackboxDenoiser::fused_rope_gqa_padding;
        let _balanced_strict_qk_norm_rope_builder: fn(
            crate::NativePaddedBlackboxDenoiser,
            bool,
        )
            -> crate::NativePaddedBlackboxDenoiser =
            crate::NativePaddedBlackboxDenoiser::with_balanced_strict_qk_norm_rope;
        let _balanced_strict_qk_norm_rope_setter: fn(
            &mut crate::NativePaddedBlackboxDenoiser,
            bool,
        ) = crate::NativePaddedBlackboxDenoiser::set_balanced_strict_qk_norm_rope;
        let _balanced_strict_qk_norm_rope_getter: fn(&crate::NativePaddedBlackboxDenoiser) -> bool =
            crate::NativePaddedBlackboxDenoiser::balanced_strict_qk_norm_rope;
        let _split_shared_projection_builder: fn(
            crate::NativePaddedBlackboxDenoiser,
            bool,
        ) -> crate::NativePaddedBlackboxDenoiser =
            crate::NativePaddedBlackboxDenoiser::with_split_double_stream_shared_projection;
        let _split_shared_projection_setter: fn(&mut crate::NativePaddedBlackboxDenoiser, bool) =
            crate::NativePaddedBlackboxDenoiser::set_split_double_stream_shared_projection;
        let _split_shared_projection_getter: fn(&crate::NativePaddedBlackboxDenoiser) -> bool =
            crate::NativePaddedBlackboxDenoiser::split_double_stream_shared_projection;
        let _blackbox_forward: fn(
            &BooguDenoiser<NativeWgpuBackend>,
            BooguDenoiserInput<NativeWgpuBackend>,
            u8,
        ) -> Result<Tensor<NativeWgpuBackend, 4>, BooguError> =
            BooguDenoiser::forward_native_padded_blackbox;
        let _blackbox_tiled_forward: fn(
            &BooguDenoiser<NativeWgpuBackend>,
            BooguDenoiserInput<NativeWgpuBackend>,
            u8,
            u8,
        )
            -> Result<Tensor<NativeWgpuBackend, 4>, BooguError> =
            BooguDenoiser::forward_native_padded_blackbox_tiled;
        let _blackbox_chunked: fn(
            Tensor<NativeWgpuBackend, 4>,
            Tensor<NativeWgpuBackend, 4>,
            Tensor<NativeWgpuBackend, 4>,
            usize,
            u8,
        ) -> Tensor<NativeWgpuBackend, 4> = required_chunked_padded_blackbox_attention;
        let _blackbox_tiled_chunked: fn(
            Tensor<NativeWgpuBackend, 4>,
            Tensor<NativeWgpuBackend, 4>,
            Tensor<NativeWgpuBackend, 4>,
            usize,
            u8,
            u8,
        ) -> Tensor<NativeWgpuBackend, 4> = required_chunked_padded_blackbox_attention_tiled;
        let _blackbox_partitioned_chunked: fn(
            Tensor<NativeWgpuBackend, 4>,
            Tensor<NativeWgpuBackend, 4>,
            Tensor<NativeWgpuBackend, 4>,
            usize,
            u8,
            u8,
            u8,
        ) -> Tensor<NativeWgpuBackend, 4> = required_chunked_padded_blackbox_attention_partitioned;
        let _gqa_blackbox_tiled_chunked: fn(
            Tensor<NativeWgpuBackend, 4>,
            Tensor<NativeWgpuBackend, 4>,
            Tensor<NativeWgpuBackend, 4>,
            usize,
            u8,
            u8,
        ) -> Tensor<NativeWgpuBackend, 4> = required_chunked_gqa_padded_blackbox_attention_tiled;
        let _fused_strict_qk_norm_rope: fn(
            Tensor<NativeWgpuBackend, 4>,
            Tensor<NativeWgpuBackend, 4>,
            Tensor<NativeWgpuBackend, 4>,
            Tensor<NativeWgpuBackend, 1>,
            Tensor<NativeWgpuBackend, 1>,
            Tensor<NativeWgpuBackend, 3>,
            Tensor<NativeWgpuBackend, 3>,
            f64,
            f64,
            usize,
        ) -> Tensor<NativeWgpuBackend, 4> =
            required_chunked_gqa_wgpu_fused_strict_qk_norm_rope_padded_blackbox_attention;
        let _fused_rope_gqa_padding: fn(
            Tensor<NativeWgpuBackend, 4>,
            Tensor<NativeWgpuBackend, 4>,
            Tensor<NativeWgpuBackend, 4>,
            Tensor<NativeWgpuBackend, 3>,
            Tensor<NativeWgpuBackend, 3>,
            usize,
        ) -> Tensor<NativeWgpuBackend, 4> =
            required_chunked_gqa_wgpu_fused_rope_padded_blackbox_attention;
        let _balanced_strict_qk_norm_rope: fn(
            Tensor<NativeWgpuBackend, 4>,
            Tensor<NativeWgpuBackend, 4>,
            Tensor<NativeWgpuBackend, 4>,
            Tensor<NativeWgpuBackend, 1>,
            Tensor<NativeWgpuBackend, 1>,
            Tensor<NativeWgpuBackend, 3>,
            Tensor<NativeWgpuBackend, 3>,
            f64,
            f64,
            usize,
        ) -> Tensor<NativeWgpuBackend, 4> =
            required_chunked_gqa_wgpu_balanced_strict_qk_norm_rope_padded_blackbox_attention;
    }

    /// Run this before loading the full checkpoint to confirm that the selected native adapter
    /// exposes a compatible accelerated CMMA instruction for every supported plane count.
    #[cfg(feature = "wgpu")]
    #[test]
    #[ignore = "requires an explicitly selected native hardware WGPU adapter"]
    fn native_wgpu_padded_blackbox_kernel_smoke() {
        let device = crate::require_native_wgpu_device().expect("native hardware WGPU adapter");
        let tensor = |shape: [usize; 4], offset: usize| {
            let elements = shape.iter().product();
            let values = (0..elements)
                .map(|index| {
                    let integer = ((index + offset) * 19 + 7) % 53;
                    (integer as f32 - 26.0) / 17.0
                })
                .collect::<Vec<_>>();
            Tensor::<NativeWgpuBackend, 4>::from_data(TensorData::new(values, shape), &device)
                .cast(DType::F16)
        };
        // The 49-row query exercises the real prompt-tail padding contract.
        let query = tensor([1, 1, 49, 120], 1);
        let key = tensor([1, 1, 512, 120], 5);
        let value = tensor([1, 1, 512, 120], 9);
        let expected = attention(
            query.clone(),
            key.clone(),
            value.clone(),
            None,
            None,
            AttentionModuleOptions::default(),
        )
        .into_data()
        .to_vec::<half::f16>()
        .expect("portable WGPU attention values")
        .into_iter()
        .map(f32::from)
        .collect::<Vec<_>>();

        for (num_planes, seq_kv_tiles, seq_q_tiles) in [(2, 1, 1), (2, 2, 1), (4, 1, 1)] {
            let output = required_chunked_padded_blackbox_attention_partitioned(
                query.clone(),
                key.clone(),
                value.clone(),
                128,
                num_planes,
                seq_kv_tiles,
                seq_q_tiles,
            );
            assert_eq!(output.dims(), [1, 1, 49, 120]);
            assert_eq!(output.dtype(), DType::F16);
            let actual = output
                .into_data()
                .to_vec::<half::f16>()
                .expect("padded blackbox WGPU attention values")
                .into_iter()
                .map(f32::from)
                .collect::<Vec<_>>();
            assert!(actual.iter().all(|value| value.is_finite()));
            let mut max_abs = 0.0_f32;
            let mut squared_error = 0.0_f64;
            let mut dot = 0.0_f64;
            let mut expected_square = 0.0_f64;
            let mut actual_square = 0.0_f64;
            for (&expected, &actual) in expected.iter().zip(&actual) {
                let delta = expected - actual;
                max_abs = max_abs.max(delta.abs());
                squared_error += f64::from(delta).powi(2);
                dot += f64::from(expected) * f64::from(actual);
                expected_square += f64::from(expected).powi(2);
                actual_square += f64::from(actual).powi(2);
            }
            let rmse = (squared_error / expected.len() as f64).sqrt();
            let cosine = dot / (expected_square.sqrt() * actual_square.sqrt());
            assert!(
                max_abs <= 0.02 && rmse <= 0.003 && cosine >= 0.9999,
                "padded blackbox planes={num_planes} seq_kv_tiles={seq_kv_tiles} seq_q_tiles={seq_q_tiles}: \
                     max={max_abs}, rmse={rmse}, cosine={cosine}"
            );
        }

        // Exercise the production GQA preparation seam itself: the accelerated path receives
        // unexpanded K/V heads, while the portable oracle materializes the exact head mapping.
        let gqa_query = tensor([1, 4, 49, 120], 13);
        let gqa_key = tensor([1, 1, 512, 120], 17);
        let gqa_value = tensor([1, 1, 512, 120], 21);
        let expected = attention(
            gqa_query.clone(),
            gqa_key.clone().repeat_dim(1, 4),
            gqa_value.clone().repeat_dim(1, 4),
            None,
            None,
            AttentionModuleOptions::default(),
        )
        .into_data()
        .to_vec::<half::f16>()
        .expect("portable GQA WGPU attention values")
        .into_iter()
        .map(f32::from)
        .collect::<Vec<_>>();
        for num_planes in [4] {
            let actual = required_chunked_gqa_wgpu_padded_blackbox_attention_partitioned(
                gqa_query.clone(),
                gqa_key.clone(),
                gqa_value.clone(),
                128,
                num_planes,
                1,
                1,
            )
            .into_data()
            .to_vec::<half::f16>()
            .expect("fused GQA padded blackbox WGPU attention values")
            .into_iter()
            .map(f32::from)
            .collect::<Vec<_>>();
            assert!(actual.iter().all(|value| value.is_finite()));
            let mut max_abs = 0.0_f32;
            let mut squared_error = 0.0_f64;
            let mut dot = 0.0_f64;
            let mut expected_square = 0.0_f64;
            let mut actual_square = 0.0_f64;
            for (&expected, &actual) in expected.iter().zip(&actual) {
                let delta = expected - actual;
                max_abs = max_abs.max(delta.abs());
                squared_error += f64::from(delta).powi(2);
                dot += f64::from(expected) * f64::from(actual);
                expected_square += f64::from(expected).powi(2);
                actual_square += f64::from(actual).powi(2);
            }
            let rmse = (squared_error / expected.len() as f64).sqrt();
            let cosine = dot / (expected_square.sqrt() * actual_square.sqrt());
            assert!(
                max_abs <= 0.02 && rmse <= 0.003 && cosine >= 0.9999,
                "fused GQA padded blackbox planes={num_planes}: max={max_abs}, rmse={rmse}, cosine={cosine}"
            );
        }
    }

    /// Hardware oracle for the opt-in one-dispatch RoPE+GQA padding preparation.
    #[cfg(feature = "wgpu")]
    #[test]
    #[ignore = "requires an explicitly selected native hardware WGPU adapter"]
    fn native_wgpu_fused_rope_gqa_padding_preparation_reference() {
        let device = crate::require_native_wgpu_device().expect("native hardware WGPU adapter");
        let tensor = |shape: [usize; 4], offset: usize| {
            let elements = shape.iter().product();
            let values = (0..elements)
                .map(|index| {
                    let integer = ((index + offset) * 19 + 7) % 53;
                    (integer as f32 - 26.0) / 17.0
                })
                .collect::<Vec<_>>();
            Tensor::<NativeWgpuBackend, 4>::from_data(TensorData::new(values, shape), &device)
                .cast(DType::F16)
        };
        let query = tensor([1, 4, 49, BOOGU_ATTENTION_HEAD_DIM], 37);
        let key = tensor([1, 1, 49, BOOGU_ATTENTION_HEAD_DIM], 41);
        let value = tensor([1, 1, 49, BOOGU_ATTENTION_HEAD_DIM], 43);
        let phases = (0..49 * BOOGU_ATTENTION_HEAD_DIM)
            .map(|index| ((index / 2) % 23) as f32 / 29.0)
            .collect::<Vec<_>>();
        let phase = Tensor::<NativeWgpuBackend, 3>::from_data(
            TensorData::new(phases, [1, 49, BOOGU_ATTENTION_HEAD_DIM]),
            &device,
        )
        .cast(DType::F16);
        let cos = phase.clone().cos();
        let sin = phase.sin();
        let rope = |input: Tensor<NativeWgpuBackend, 4>| {
            let token_major = input.permute([0, 2, 1, 3]);
            let [batch, sequence, heads, width] = token_major.dims();
            let pairs = width / 2;
            let paired = token_major
                .clone()
                .reshape([batch, sequence, heads, pairs, 2]);
            let real = paired
                .clone()
                .slice([0..batch, 0..sequence, 0..heads, 0..pairs, 0..1]);
            let imag = paired.slice([0..batch, 0..sequence, 0..heads, 0..pairs, 1..2]);
            let rotated =
                Tensor::cat(vec![imag.neg(), real], 4).reshape([batch, sequence, heads, width]);
            (token_major * cos.clone().unsqueeze_dim(2) + rotated * sin.clone().unsqueeze_dim(2))
                .permute([0, 2, 1, 3])
        };
        let expected = reference_prepare_gqa_padded_blackbox_inputs(
            rope(query.clone()),
            rope(key.clone()),
            value.clone(),
        );
        let actual = prepare_gqa_rope_padded_blackbox_inputs(query, key, value, cos, sin);

        for (name, actual, expected) in [
            ("query", actual.0, expected.0),
            ("key", actual.1, expected.1),
            ("value", actual.2, expected.2),
        ] {
            assert_eq!(actual.dims(), expected.dims(), "{name} shape");
            let expected = expected
                .into_data()
                .to_vec::<half::f16>()
                .expect("composed F16 preparation values")
                .into_iter()
                .map(f32::from)
                .collect::<Vec<_>>();
            let actual = actual
                .into_data()
                .to_vec::<half::f16>()
                .expect("fused F16 preparation values")
                .into_iter()
                .map(f32::from)
                .collect::<Vec<_>>();
            let max_abs = expected
                .iter()
                .zip(actual)
                .map(|(expected, actual)| (expected - actual).abs())
                .fold(0.0_f32, f32::max);
            assert!(max_abs <= 0.002, "{name} fused prep max_abs={max_abs}");
        }
    }

    /// Hardware oracle for the opt-in balanced strict Q/K RMSNorm dispatches.
    #[cfg(feature = "wgpu")]
    #[test]
    #[ignore = "requires an explicitly selected native hardware WGPU adapter"]
    fn native_wgpu_balanced_strict_qk_norm_reference() {
        let device = crate::require_native_wgpu_device().expect("native hardware WGPU adapter");
        let tensor = |shape: [usize; 4], offset: usize| {
            let elements = shape.iter().product();
            let values = (0..elements)
                .map(|index| {
                    let integer = ((index + offset) * 19 + 7) % 53;
                    (integer as f32 - 26.0) / 17.0
                })
                .collect::<Vec<_>>();
            Tensor::<NativeWgpuBackend, 4>::from_data(TensorData::new(values, shape), &device)
                .cast(DType::F16)
        };
        let gamma_data = TensorData::new(
            (0..BOOGU_ATTENTION_HEAD_DIM)
                .map(|index| 0.75 + (index % 13) as f32 / 26.0)
                .collect::<Vec<_>>(),
            [BOOGU_ATTENTION_HEAD_DIM],
        );
        let gamma = Tensor::<NativeWgpuBackend, 1>::from_data(gamma_data, &device).cast(DType::F16);
        let epsilon = 1.0e-5;

        for (name, input) in [
            ("query", tensor([1, 4, 49, BOOGU_ATTENTION_HEAD_DIM], 61)),
            ("key", tensor([1, 1, 49, BOOGU_ATTENTION_HEAD_DIM], 67)),
        ] {
            let expected =
                reference_balanced_strict_rms_norm(input.clone(), gamma.clone(), epsilon);
            let actual = balanced_strict_rms_norm(input, gamma.clone(), epsilon);
            assert_eq!(actual.dims(), expected.dims(), "{name} shape");
            assert_eq!(actual.dtype(), DType::F16, "{name} dtype");
            let expected = expected
                .into_data()
                .to_vec::<half::f16>()
                .expect("reference balanced RMSNorm values")
                .into_iter()
                .map(f32::from)
                .collect::<Vec<_>>();
            let actual = actual
                .into_data()
                .to_vec::<half::f16>()
                .expect("native balanced RMSNorm values")
                .into_iter()
                .map(f32::from)
                .collect::<Vec<_>>();
            let max_abs = expected
                .iter()
                .zip(actual)
                .map(|(expected, actual)| (expected - actual).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                max_abs <= 0.002,
                "{name} balanced RMSNorm max_abs={max_abs}"
            );
        }
    }

    /// Hardware oracle for the opt-in one-dispatch strict Q/K RMSNorm+RoPE preparation.
    #[cfg(feature = "wgpu")]
    #[test]
    #[ignore = "requires an explicitly selected native hardware WGPU adapter"]
    fn native_wgpu_fused_strict_qk_norm_rope_preparation_reference() {
        let device = crate::require_native_wgpu_device().expect("native hardware WGPU adapter");
        let tensor = |shape: [usize; 4], offset: usize| {
            let elements = shape.iter().product();
            let values = (0..elements)
                .map(|index| {
                    let integer = ((index + offset) * 19 + 7) % 53;
                    (integer as f32 - 26.0) / 17.0
                })
                .collect::<Vec<_>>();
            Tensor::<NativeWgpuBackend, 4>::from_data(TensorData::new(values, shape), &device)
                .cast(DType::F16)
        };
        let query = tensor([1, 4, 49, BOOGU_ATTENTION_HEAD_DIM], 51);
        let key = tensor([1, 1, 49, BOOGU_ATTENTION_HEAD_DIM], 53);
        let value = tensor([1, 1, 49, BOOGU_ATTENTION_HEAD_DIM], 59);
        let gamma_data = TensorData::new(
            (0..BOOGU_ATTENTION_HEAD_DIM)
                .map(|index| 0.75 + (index % 13) as f32 / 26.0)
                .collect::<Vec<_>>(),
            [BOOGU_ATTENTION_HEAD_DIM],
        );
        let query_gamma =
            Tensor::<NativeWgpuBackend, 1>::from_data(gamma_data.clone(), &device).cast(DType::F16);
        let key_gamma =
            Tensor::<NativeWgpuBackend, 1>::from_data(gamma_data, &device).cast(DType::F16);
        let phases = (0..49 * BOOGU_ATTENTION_HEAD_DIM)
            .map(|index| ((index / 2) % 23) as f32 / 29.0)
            .collect::<Vec<_>>();
        let phase = Tensor::<NativeWgpuBackend, 3>::from_data(
            TensorData::new(phases, [1, 49, BOOGU_ATTENTION_HEAD_DIM]),
            &device,
        )
        .cast(DType::F16);
        let cos = phase.clone().cos();
        let sin = phase.sin();
        let epsilon = 1.0e-5;

        let expected_query = {
            let token_major = query.clone().permute([0, 2, 1, 3]);
            let rms = (token_major.clone().cast(DType::F32).square().mean_dim(3) + epsilon)
                .sqrt()
                .cast(DType::F16);
            let normalized = token_major / rms * query_gamma.clone().unsqueeze();
            let [batch, sequence, heads, width] = normalized.dims();
            let pairs = width / 2;
            let paired = normalized
                .clone()
                .reshape([batch, sequence, heads, pairs, 2]);
            let real = paired
                .clone()
                .slice([0..batch, 0..sequence, 0..heads, 0..pairs, 0..1]);
            let imag = paired.slice([0..batch, 0..sequence, 0..heads, 0..pairs, 1..2]);
            let rotated =
                Tensor::cat(vec![imag.neg(), real], 4).reshape([batch, sequence, heads, width]);
            (normalized * cos.clone().unsqueeze_dim(2) + rotated * sin.clone().unsqueeze_dim(2))
                .permute([0, 2, 1, 3])
        };
        let expected_key = {
            let token_major = key.clone().permute([0, 2, 1, 3]);
            let rms = (token_major.clone().cast(DType::F32).square().mean_dim(3) + epsilon)
                .sqrt()
                .cast(DType::F16);
            let normalized = token_major / rms * key_gamma.clone().unsqueeze();
            let [batch, sequence, heads, width] = normalized.dims();
            let pairs = width / 2;
            let paired = normalized
                .clone()
                .reshape([batch, sequence, heads, pairs, 2]);
            let real = paired
                .clone()
                .slice([0..batch, 0..sequence, 0..heads, 0..pairs, 0..1]);
            let imag = paired.slice([0..batch, 0..sequence, 0..heads, 0..pairs, 1..2]);
            let rotated =
                Tensor::cat(vec![imag.neg(), real], 4).reshape([batch, sequence, heads, width]);
            (normalized * cos.clone().unsqueeze_dim(2) + rotated * sin.clone().unsqueeze_dim(2))
                .permute([0, 2, 1, 3])
        };
        let expected = reference_prepare_gqa_padded_blackbox_inputs(
            expected_query,
            expected_key,
            value.clone(),
        );
        let actual = prepare_gqa_strict_norm_rope_padded_blackbox_inputs(
            query,
            key,
            value,
            query_gamma,
            key_gamma,
            cos,
            sin,
            epsilon,
            epsilon,
        );

        for (name, actual, expected) in [
            ("query", actual.0, expected.0),
            ("key", actual.1, expected.1),
            ("value", actual.2, expected.2),
        ] {
            assert_eq!(actual.dims(), expected.dims(), "{name} shape");
            let expected = expected
                .into_data()
                .to_vec::<half::f16>()
                .expect("composed F16 preparation values")
                .into_iter()
                .map(f32::from)
                .collect::<Vec<_>>();
            let actual = actual
                .into_data()
                .to_vec::<half::f16>()
                .expect("fused F16 preparation values")
                .into_iter()
                .map(f32::from)
                .collect::<Vec<_>>();
            let max_abs = expected
                .iter()
                .zip(actual)
                .map(|(expected, actual)| (expected - actual).abs())
                .fold(0.0_f32, f32::max);
            assert!(max_abs <= 0.002, "{name} fused prep max_abs={max_abs}");
        }
    }
}
