use super::init_matmul_output;
use crate::{CubeRuntime, kernel::quantization::dequantize, tensor::CubeTensor};
use burn_backend::{DType, QTensorPrimitive};
use burn_std::QuantLevel;
use cubek::{
    matmul::{
        definition::{MatmulElems, MatmulGlobalElems, MatmulSetupError},
        launch::Strategy,
        routines::BlueprintStrategy,
    },
    std::InputBinding,
};

use super::quantized::{
    QuantizedCmmaAlgorithm, QuantizedCmmaSelectionArgs, QuantizedUnitAlgorithm,
    QuantizedUnitSelectionArgs,
};

#[cfg(feature = "autotune")]
use super::matmul_autotune;

/// The strategy to be used when launching a matmul kernel.
pub enum MatmulStrategy {
    #[cfg(feature = "autotune")]
    /// Using autotune to choose the best kernel based on runtime information.
    Autotune,
    /// Cube implementation of matmul.
    Cube,
    /// Bounded Cube implementation for packed quantized operands.
    ///
    /// The default cooperative-matrix selector can infer a 64-KiB shared stage for F32 x Q4S,
    /// exceeding WebGPU adapters that expose only the portable 16-KiB workgroup-storage minimum.
    /// The custom selector keeps Q4 values packed at rest and uses a bounded
    /// F16-stage/F32-accumulation cooperative kernel when the adapter supports it. A
    /// resource-checked register-tiled kernel is the portable fallback for WebGPU and adapters
    /// without cooperative matrices.
    QuantizedPortable,
}

impl Default for MatmulStrategy {
    fn default() -> Self {
        // if autotune is enabled, default to autotune
        #[cfg(feature = "autotune")]
        return MatmulStrategy::Autotune;

        #[cfg(not(feature = "autotune"))]
        MatmulStrategy::Cube
    }
}

/// Launch a matmul kernel using the given strategy.
pub fn matmul<R: CubeRuntime>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: Option<CubeTensor<R>>,
    strategy: MatmulStrategy,
    out_dtype: DType,
) -> Result<CubeTensor<R>, MatmulSetupError> {
    match strategy {
        MatmulStrategy::Cube => {
            let out = out.unwrap_or_else(|| init_matmul_output(&lhs, &rhs, out_dtype));
            launch_matmul(&Default::default(), lhs, rhs, out.clone())?;
            Ok(out)
        }
        MatmulStrategy::QuantizedPortable => {
            let out = out.unwrap_or_else(|| init_matmul_output(&lhs, &rhs, out_dtype));
            launch_matmul_quantized(lhs, rhs, out.clone())?;
            Ok(out)
        }
        #[cfg(feature = "autotune")]
        MatmulStrategy::Autotune => Ok(matmul_autotune(lhs, rhs, out, out_dtype)),
    }
}

#[cfg(feature = "autotune")]
pub(crate) fn launch_matmul_naive<R: CubeRuntime>(
    strategy: &Strategy,
    mut lhs: CubeTensor<R>,
    mut rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), MatmulSetupError> {
    // Naive has very specific layout requirements for block scaled tensors, so we need to manually
    // dequantize if it fails to launch normally. This is because naive is assumed to always work.
    if lhs.qparams.is_some() || rhs.qparams.is_some() {
        match launch_matmul(strategy, lhs.clone(), rhs.clone(), out.clone()) {
            Err(_) => {
                if lhs.qparams.is_some() {
                    lhs = dequantize(lhs, out.dtype);
                }
                if rhs.qparams.is_some() {
                    rhs = dequantize(rhs, out.dtype);
                }
                launch_matmul(strategy, lhs, rhs, out)
            }
            Ok(_) => Ok(()),
        }
    } else {
        launch_matmul(strategy, lhs, rhs, out)
    }
}

pub(crate) fn launch_matmul<R: CubeRuntime>(
    strategy: &Strategy,
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), MatmulSetupError> {
    launch_matmul_with(MatmulLaunch::Standard(strategy), lhs, rhs, out)
}

fn launch_matmul_quantized<R: CubeRuntime>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), MatmulSetupError> {
    launch_matmul_with(MatmulLaunch::QuantizedPortable, lhs, rhs, out)
}

#[derive(Clone, Copy)]
enum MatmulLaunch<'a> {
    Standard(&'a Strategy),
    QuantizedPortable,
}

fn launch_matmul_with<R: CubeRuntime>(
    strategy: MatmulLaunch<'_>,
    lhs: CubeTensor<R>,
    mut rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), MatmulSetupError> {
    let client = &out.client;

    let lhs_quant_handles = lhs.quantized_handles();
    let out_dtype: DType = out.dtype;

    let (lhs_dtype, lhs_handle) = match lhs_quant_handles {
        None => {
            let lhs_dtype = lhs.dtype;
            (
                lhs_dtype,
                InputBinding::new(lhs.binding(), lhs_dtype.into()),
            )
        }
        Some((data, scale)) => {
            let scheme = *lhs.scheme();
            let data_dtype = data.dtype;
            let scale_dtype = scale.dtype;
            (
                out_dtype,
                InputBinding::quantized(
                    data.binding(),
                    scale.binding(),
                    lhs.meta.shape().clone(),
                    scheme,
                    data_dtype.into(),
                    scale_dtype.into(),
                ),
            )
        }
    };

    let rhs_quant_handles = rhs.quantized_handles();

    let (rhs_dtype, rhs_handle) = match rhs_quant_handles {
        None => (
            lhs_dtype,
            InputBinding::new(rhs.binding(), lhs_dtype.into()),
        ),
        Some((data, scale)) => {
            // Extremely hacky fix to ensure naive can run in every case
            if matches!(strategy, MatmulLaunch::Standard(Strategy::Naive))
                && matches!(rhs.scheme().level, QuantLevel::Block(_))
            {
                rhs = dequantize(rhs.clone(), lhs_dtype);
                let rhs_dtype = rhs.dtype;
                (
                    lhs_dtype,
                    InputBinding::new(rhs.binding(), rhs_dtype.into()),
                )
            } else {
                let scheme = *rhs.scheme();
                let data_dtype = data.dtype;
                let scale_dtype = scale.dtype;
                (
                    out_dtype,
                    InputBinding::quantized(
                        data.binding(),
                        scale.binding(),
                        rhs.meta.shape().clone(),
                        scheme,
                        data_dtype.into(),
                        scale_dtype.into(),
                    ),
                )
            }
        }
    };

    let mut dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: lhs_dtype.into(),
        rhs: rhs_dtype.into(),
        out: out_dtype.into(),
    });

    match strategy {
        MatmulLaunch::Standard(strategy) => cubek::matmul::launch::launch_ref(
            strategy,
            client,
            lhs_handle,
            rhs_handle,
            out.clone().binding(),
            &mut dtypes,
        )?,
        MatmulLaunch::QuantizedPortable => {
            let cooperative =
                cubek::matmul::launch::launch_tiling::launch_ref::<R, QuantizedCmmaAlgorithm>(
                    client,
                    lhs_handle.clone(),
                    rhs_handle.clone(),
                    out.clone().binding(),
                    &BlueprintStrategy::Inferred(QuantizedCmmaSelectionArgs),
                    &mut dtypes,
                );
            if let Err(MatmulSetupError::Unavailable(_)) = cooperative {
                cubek::matmul::launch::launch_tiling::launch_ref::<R, QuantizedUnitAlgorithm>(
                    client,
                    lhs_handle,
                    rhs_handle,
                    out.clone().binding(),
                    &BlueprintStrategy::Inferred(QuantizedUnitSelectionArgs),
                    &mut dtypes,
                )?;
            } else {
                cooperative?;
            }
        }
    }

    Ok(())
}
