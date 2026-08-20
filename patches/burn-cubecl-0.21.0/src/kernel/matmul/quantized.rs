use std::{fmt::Display, marker::PhantomData};

use cubecl::{
    Runtime,
    client::ComputeClient,
    ir::{ElemType, FloatKind, StorageType},
};
use cubek::matmul::{
    components::{
        batch::{BatchMatmulFamily, PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul},
        global::{
            PlaneWriterFamily, UnitWriterFamily,
            read::{FullLoadingStrategy, sync_full_cyclic::SyncFullCyclicLoading},
            single_stage::simple::SimpleMatmulFamily,
        },
        stage::{
            ColMajorTilingOrder, PlaneMatmulFamily, RowMajorTilingOrder, StridedStageFamily,
            UnitMatmulFamily,
        },
        tile::TileMatmulKind,
    },
    definition::{
        MatmulElems, MatmulProblem, MatmulSetupError, MatmulVectorSizes, TilingBlueprint,
        TilingScheme,
    },
    launch::RuntimeConfig,
    routines::{
        BlueprintStrategy, DeviceSettings, ExpandInfo, LaunchInfo, PartitionScaling,
        PlaneTilingBlueprintOptions, Routine, StageScaling, TileSizeSelection,
        UnitTilingBlueprintOptions, infer_blueprint_plane, infer_blueprint_unit,
    },
};
use cubek::std::tile::Strided;

// Two cooperative rows compile to 22,528 bytes of workgroup storage for the Qwen packed-Q4S
// projections. WebGPU guarantees only 16 KiB. Selecting one row there preserves the cooperative
// matrix path while reducing the compiled shared stage to the portable limit; adapters with the
// larger effective limit keep the faster two-row launch.
const QUANTIZED_CMMA_TWO_ROW_SHARED_MEMORY_BYTES: usize = 22_528;
const QUANTIZED_CMMA_TINY_SELECTION_ENABLED: bool = false;
// WebGPU subgroup metadata is advisory unless the subgroup feature is enabled. Chrome can report
// a minimum subgroup size of four while the ordinary portable device exposes no subgroup
// operations at all. Feeding that value into the unit selector creates only eight workers and
// starves a 32-lane NVIDIA warp. Register matmul uses no subgroup intrinsic, so use the portable
// 32-worker scheduling quantum explicitly; the selected resource-bounded topology contains 32
// workers and remains valid on WebGPU implementations whose physical subgroup differs.
const QUANTIZED_PORTABLE_UNIT_PLANE_DIM: u32 = 32;
// CubeCL's stage allocations can be swizzled up to 128-byte boundaries. Round every independent
// allocation up to this value when rejecting a blueprint before shader compilation.
const QUANTIZED_PORTABLE_SHARED_ALIGNMENT_BYTES: usize = 128;

const fn quantized_cmma_row_count(max_shared_memory_size: usize) -> u32 {
    if max_shared_memory_size < QUANTIZED_CMMA_TWO_ROW_SHARED_MEMORY_BYTES {
        1
    } else {
        2
    }
}

fn aligned_allocation_bytes(
    elements_per_stage: u64,
    element_size: usize,
    stages: u32,
) -> Option<usize> {
    let bytes = usize::try_from(elements_per_stage)
        .ok()?
        .checked_mul(element_size)?;
    let aligned = bytes.checked_add(QUANTIZED_PORTABLE_SHARED_ALIGNMENT_BYTES - 1)?
        / QUANTIZED_PORTABLE_SHARED_ALIGNMENT_BYTES
        * QUANTIZED_PORTABLE_SHARED_ALIGNMENT_BYTES;
    aligned.checked_mul(stages as usize)
}

fn quantized_unit_tiling_shared_memory_bytes(
    tiling: &TilingScheme,
    lhs_element_size: usize,
    rhs_element_size: usize,
    out_element_size: usize,
    lhs_stages: u32,
    rhs_stages: u32,
) -> Option<usize> {
    let lhs_elements = u64::from(tiling.elements_per_stage_along_m())
        .checked_mul(u64::from(tiling.elements_per_stage_along_k()))?;
    let rhs_elements = u64::from(tiling.elements_per_stage_along_k())
        .checked_mul(u64::from(tiling.elements_per_stage_along_n()))?;
    // UnitWriter replaces the per-partition tile counts with one tile per unit. Its shared output
    // allocation therefore covers one tile for each M/N stage partition, not the full input-stage
    // partition footprint.
    let out_elements = u64::from(tiling.tile_size.m)
        .checked_mul(u64::from(tiling.tile_size.n))?
        .checked_mul(u64::from(tiling.stage_size.m))?
        .checked_mul(u64::from(tiling.stage_size.n))?;

    aligned_allocation_bytes(lhs_elements, lhs_element_size, lhs_stages)?
        .checked_add(aligned_allocation_bytes(
            rhs_elements,
            rhs_element_size,
            rhs_stages,
        )?)?
        .checked_add(aligned_allocation_bytes(out_elements, out_element_size, 1)?)
}

fn quantized_unit_shared_memory_bytes(
    blueprint: &TilingBlueprint,
    dtypes: &MatmulElems,
) -> Option<usize> {
    quantized_unit_tiling_shared_memory_bytes(
        &blueprint.tiling_scheme,
        dtypes.lhs_stage.size(),
        dtypes.rhs_stage.size(),
        dtypes.acc_stage.size(),
        1,
        1,
    )
}

/// Cooperative-matrix matmul with a bounded stage for packed quantized operands.
pub(crate) struct QuantizedCmmaAlgorithm<
    LL = SyncFullCyclicLoading<ColMajorTilingOrder>,
    RL = SyncFullCyclicLoading<RowMajorTilingOrder>,
    AL = SyncFullCyclicLoading<RowMajorTilingOrder>,
> {
    _ll: PhantomData<LL>,
    _rl: PhantomData<RL>,
    _al: PhantomData<AL>,
}

#[derive(Default, Clone, Debug)]
pub(crate) struct QuantizedCmmaSelectionArgs;

impl Display for QuantizedCmmaSelectionArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("_quantized_portable")
    }
}

impl<RC, LL, RL, AL> Routine<RC> for QuantizedCmmaAlgorithm<LL, RL, AL>
where
    RC: RuntimeConfig,
    LL: FullLoadingStrategy<RC, TileKind = Strided>,
    RL: FullLoadingStrategy<
            RC,
            Stage = LL::Stage,
            TileKind = Strided,
            SyncStrategy = LL::SyncStrategy,
        >,
    AL: FullLoadingStrategy<RC, TileKind = Strided, SyncStrategy = LL::SyncStrategy>,
{
    type Strategy = QuantizedCmmaSelectionArgs;
    type BatchMatmul = PartitionedBatchMatmulFamily<
        RC,
        SimpleMatmulFamily<
            PlaneMatmulFamily<LL::Stage, RL::Stage, Option<AL::Stage>>,
            RC,
            LL,
            RL,
            AL,
            PlaneWriterFamily,
        >,
        RowMajorGlobalPartitionMatmul,
    >;
    type Blueprint = TilingBlueprint;
    type Config = <Self::BatchMatmul as BatchMatmulFamily<RC>>::Config;

    fn expand_blueprint<R: Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        strategy: &BlueprintStrategy<RC, Self>,
    ) -> Result<ExpandInfo<Self::Blueprint>, MatmulSetupError> {
        let mut dtypes = MatmulElems::from_globals(&problem.global_dtypes);
        let tile_matmul = TileMatmulKind::Cmma;
        let f16 = StorageType::Scalar(ElemType::Float(FloatKind::F16));
        dtypes.lhs_stage = f16;
        dtypes.rhs_stage = f16;
        dtypes.lhs_register = f16;
        dtypes.rhs_register = f16;

        let (blueprint, dtypes) = match strategy {
            BlueprintStrategy::Forced(blueprint) => (blueprint.clone(), dtypes),
            BlueprintStrategy::Inferred(_) => infer_blueprint_plane::<R>(
                tile_matmul,
                &device_settings.client,
                problem,
                device_settings.plane_dim,
                dtypes,
                &device_settings.vector_sizes,
                PlaneTilingBlueprintOptions {
                    partition_k: Some(1),
                    row_count: Some(quantized_cmma_row_count(
                        device_settings
                            .client
                            .properties()
                            .hardware
                            .max_shared_memory_size,
                    )),
                    partition_buffering: Some(
                        cubek::matmul::components::stage::PartitionBuffering::Single,
                    ),
                    // The generic tiny selector returns before applying `row_count` and selected
                    // a 22,528-byte stage for Qwen prompt matmuls. Keep this portable path on the
                    // explicit adapter-bounded blueprint above.
                    tiny_selection_enabled: QUANTIZED_CMMA_TINY_SELECTION_ENABLED,
                    swizzled: tile_matmul.should_swizzle(&device_settings.client),
                    ..Default::default()
                },
            )?,
        };
        Ok(ExpandInfo { blueprint, dtypes })
    }

    fn prepare<R: Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        expand_info: ExpandInfo<Self::Blueprint>,
    ) -> Result<LaunchInfo<Self::Blueprint>, MatmulSetupError> {
        let ExpandInfo { blueprint, dtypes } = expand_info;
        <Self as Routine<RC>>::validate_blueprint(
            &device_settings.client,
            &blueprint,
            problem,
            &dtypes,
            &device_settings.vector_sizes,
        )?;
        let cubedim_resource = Self::BatchMatmul::cubedim_resource(
            &blueprint,
            &dtypes,
            &device_settings.vector_sizes,
        )?;
        LaunchInfo::new(
            blueprint,
            dtypes,
            problem,
            cubedim_resource,
            device_settings,
        )
    }

    fn device_settings<R: Runtime>(
        client: &ComputeClient<R>,
        vector_sizes: MatmulVectorSizes,
    ) -> DeviceSettings<R> {
        let plane_dim = match client.properties().hardware.plane_size_min {
            0 => 32,
            plane_dim => plane_dim,
        };
        DeviceSettings {
            client: client.clone(),
            plane_dim,
            vector_sizes,
            max_cube_count: client.properties().hardware.max_cube_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        QUANTIZED_CMMA_TINY_SELECTION_ENABLED, QUANTIZED_CMMA_TWO_ROW_SHARED_MEMORY_BYTES,
        quantized_cmma_row_count, quantized_unit_tiling_shared_memory_bytes,
    };
    use cubek::matmul::definition::TilingScheme;

    #[test]
    fn quantized_portable_respects_webgpu_shared_memory_limit_correctness() {
        assert_eq!(quantized_cmma_row_count(16_384), 1);
        assert_eq!(
            quantized_cmma_row_count(QUANTIZED_CMMA_TWO_ROW_SHARED_MEMORY_BYTES - 1),
            1
        );
        assert_eq!(
            quantized_cmma_row_count(QUANTIZED_CMMA_TWO_ROW_SHARED_MEMORY_BYTES),
            2
        );
        assert!(!QUANTIZED_CMMA_TINY_SELECTION_ENABLED);
    }

    #[test]
    fn quantized_unit_fallback_fits_portable_shared_memory_correctness() {
        let former = TilingScheme::builder()
            .with_tile_size((1, 8, 8).into())
            .with_partition_size((8, 2, 2).into())
            .with_stage_size((16, 8, 1).into())
            .build()
            .expect("former Q4 tiling");
        let bounded = TilingScheme::builder()
            .with_tile_size((1, 8, 8).into())
            .with_partition_size((8, 2, 2).into())
            .with_stage_size((8, 4, 1).into())
            .build()
            .expect("bounded Q4 tiling");

        assert_eq!(
            quantized_unit_tiling_shared_memory_bytes(&former, 4, 4, 4, 2, 2),
            Some(36_864)
        );
        assert_eq!(
            quantized_unit_tiling_shared_memory_bytes(&bounded, 4, 4, 4, 1, 1),
            Some(9_216)
        );
        assert!(
            quantized_unit_tiling_shared_memory_bytes(&bounded, 4, 4, 4, 1, 1)
                .expect("bounded Q4 shared memory")
                <= 16_384
        );
    }
}

/// Resource-bounded register-tiled fallback for WebGPU and adapters without cooperative matrices.
pub(crate) struct QuantizedUnitAlgorithm;

#[derive(Default, Clone, Debug)]
pub(crate) struct QuantizedUnitSelectionArgs;

impl Display for QuantizedUnitSelectionArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("_quantized_portable_unit")
    }
}

impl<RC: RuntimeConfig> Routine<RC> for QuantizedUnitAlgorithm {
    type Strategy = QuantizedUnitSelectionArgs;
    type BatchMatmul = PartitionedBatchMatmulFamily<
        RC,
        SimpleMatmulFamily<
            UnitMatmulFamily<StridedStageFamily, Option<StridedStageFamily>>,
            RC,
            SyncFullCyclicLoading<ColMajorTilingOrder>,
            SyncFullCyclicLoading<RowMajorTilingOrder>,
            SyncFullCyclicLoading<RowMajorTilingOrder>,
            UnitWriterFamily,
        >,
        RowMajorGlobalPartitionMatmul,
    >;
    type Blueprint = TilingBlueprint;
    type Config = <Self::BatchMatmul as BatchMatmulFamily<RC>>::Config;

    fn expand_blueprint<R: Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        strategy: &BlueprintStrategy<RC, Self>,
    ) -> Result<ExpandInfo<Self::Blueprint>, MatmulSetupError> {
        let mut dtypes = MatmulElems::from_globals(&problem.global_dtypes);
        let tile_matmul = TileMatmulKind::Register;
        if tile_matmul.can_cast_stage_element() {
            dtypes.adjust_stage_dtypes();
        }
        let (blueprint, dtypes) = match strategy {
            BlueprintStrategy::Forced(blueprint) => (blueprint.clone(), dtypes),
            BlueprintStrategy::Inferred(_) => {
                let (blueprint, dtypes) = infer_blueprint_unit(
                    &device_settings.client,
                    problem,
                    device_settings.plane_dim,
                    // Use the double-buffer selector's smaller stage partitions, but execute the
                    // result through the single-buffer family below. Together with halving both
                    // stage axes this retains a full 32-worker group while fitting F32 stages in
                    // WebGPU's portable 16-KiB workgroup-storage limit.
                    true,
                    &device_settings.vector_sizes,
                    UnitTilingBlueprintOptions {
                        tile: TileSizeSelection::MinTileSize,
                        stage: StageScaling::Enabled(2),
                        partition: PartitionScaling::Disabled,
                        swizzle: tile_matmul.should_swizzle(&device_settings.client),
                    },
                    &problem.global_dtypes,
                );
                (blueprint, dtypes)
            }
        };
        Ok(ExpandInfo { blueprint, dtypes })
    }

    fn prepare<R: Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        expand_info: ExpandInfo<Self::Blueprint>,
    ) -> Result<LaunchInfo<Self::Blueprint>, MatmulSetupError> {
        let ExpandInfo { blueprint, dtypes } = expand_info;
        <Self as Routine<RC>>::validate_blueprint(
            &device_settings.client,
            &blueprint,
            problem,
            &dtypes,
            &device_settings.vector_sizes,
        )?;
        let required_shared_memory = quantized_unit_shared_memory_bytes(&blueprint, &dtypes)
            .ok_or_else(|| {
                MatmulSetupError::InvalidConfig(Box::new(
                    "packed-Q4 portable shared-memory calculation overflowed",
                ))
            })?;
        let maximum_shared_memory = device_settings
            .client
            .properties()
            .hardware
            .max_shared_memory_size;
        if required_shared_memory > maximum_shared_memory {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "packed-Q4 portable matmul requires {required_shared_memory} shared-memory bytes, but the adapter exposes {maximum_shared_memory}"
            ))));
        }
        let cubedim_resource = Self::BatchMatmul::cubedim_resource(
            &blueprint,
            &dtypes,
            &device_settings.vector_sizes,
        )?;
        LaunchInfo::new(
            blueprint,
            dtypes,
            problem,
            cubedim_resource,
            device_settings,
        )
    }

    fn device_settings<R: Runtime>(
        client: &ComputeClient<R>,
        vector_sizes: MatmulVectorSizes,
    ) -> DeviceSettings<R> {
        DeviceSettings {
            client: client.clone(),
            plane_dim: QUANTIZED_PORTABLE_UNIT_PLANE_DIM,
            vector_sizes,
            max_cube_count: client.properties().hardware.max_cube_count,
        }
    }
}
