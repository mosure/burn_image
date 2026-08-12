use burn::prelude::*;
use burn::tensor::DType;
use serde::{Deserialize, Serialize};

use crate::{BooguError, BooguTask};

/// Exact few-step DMD schedule used by the Turbo checkpoints.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DmdSchedule {
    sigmas: Vec<f32>,
}

impl DmdSchedule {
    /// Build the upstream linear schedule.
    pub fn new(steps: usize, conditioning_sigma: f32) -> Result<Self, BooguError> {
        if steps == 0 {
            return Err(BooguError::InvalidRequest(
                "DMD inference requires at least one step".into(),
            ));
        }
        if !(0.0..=1.0).contains(&conditioning_sigma) {
            return Err(BooguError::InvalidRequest(
                "conditioning sigma must be in [0, 1]".into(),
            ));
        }
        let span = 1.0_f32 - conditioning_sigma;
        let denominator = steps as f32;
        let sigmas = (0..steps)
            .map(|index| conditioning_sigma + span * index as f32 / denominator)
            .collect();
        Ok(Self { sigmas })
    }

    /// Upstream default for the selected task.
    ///
    /// The returned values are F32. Runtime tensor operations cast them to the selected execution
    /// dtype, matching direct F16 linspace for the production F16 profile and F32 linspace for the
    /// Q8/F32 profile. A BF16 reference fixture must preserve its directly constructed BF16 sigma
    /// tensors; PyTorch BF16 linspace is not equivalent to rounding an F32 linspace afterward.
    pub fn upstream(task: BooguTask) -> Self {
        let start = match task {
            BooguTask::Generate => 0.001,
            BooguTask::Edit => 0.0,
        };
        Self::new(4, start).expect("the fixed upstream DMD schedule is valid")
    }

    /// Exact four-step upstream schedule constructed in an execution dtype.
    ///
    /// PyTorch evaluates `linspace` directly in the latent dtype. In particular, a direct F16 or
    /// BF16 linspace is not always equal to casting an F32 linspace after construction. Production
    /// runtimes should use this constructor with the denoiser activation dtype; [`Self::upstream`]
    /// remains the mathematical F32 schedule used to validate custom/captured schedules.
    pub fn upstream_for_dtype(task: BooguTask, dtype: DType) -> Self {
        if task == BooguTask::Edit {
            return Self {
                sigmas: vec![0.0, 0.25, 0.5, 0.75],
            };
        }
        let sigmas = match dtype {
            DType::F16 => vec![0.001_000_404_4, 0.250_732_42, 0.500_488_3, 0.75],
            DType::BF16 => vec![0.000_999_450_7, 0.251_953_13, 0.5, 0.75],
            _ => return Self::upstream(task),
        };
        Self { sigmas }
    }

    /// Build from custom upstream timesteps. Values above one are divided by 1000.
    pub fn from_timesteps(timesteps: &[f32]) -> Result<Self, BooguError> {
        if timesteps.is_empty() || timesteps.iter().any(|value| !value.is_finite()) {
            return Err(BooguError::InvalidRequest(
                "custom DMD timesteps must be finite and non-empty".into(),
            ));
        }
        let divide = timesteps.iter().copied().fold(f32::NEG_INFINITY, f32::max) > 1.0;
        let sigmas = timesteps
            .iter()
            .map(|value| if divide { value / 1000.0 } else { *value })
            .collect();
        Ok(Self { sigmas })
    }

    /// Sigma values in execution order.
    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }
}

/// Convert a velocity prediction into the DMD student prediction.
pub fn dmd_prediction<B: Backend>(
    latents: Tensor<B, 4>,
    model_prediction: Tensor<B, 4>,
    sigma: f32,
) -> Tensor<B, 4> {
    latents + model_prediction * (1.0_f32 - sigma)
}

/// Mix a prediction with fresh noise for the next DMD sigma.
pub fn dmd_renoise<B: Backend>(
    prediction: Tensor<B, 4>,
    noise: Tensor<B, 4>,
    next_sigma: f32,
) -> Tensor<B, 4> {
    noise * (1.0_f32 - next_sigma) + prediction * next_sigma
}

#[cfg(test)]
mod tests {
    use half::f16;

    use super::*;

    #[test]
    fn schedules_match_upstream_reference() {
        assert_eq!(
            DmdSchedule::upstream(BooguTask::Generate).sigmas(),
            &[0.001, 0.25075, 0.5005, 0.75025]
        );
        assert_eq!(
            DmdSchedule::upstream(BooguTask::Edit).sigmas(),
            &[0.0, 0.25, 0.5, 0.75]
        );
        assert_eq!(
            DmdSchedule::from_timesteps(&[0.0, 250.0, 500.0, 750.0])
                .unwrap()
                .sigmas(),
            &[0.0, 0.25, 0.5, 0.75]
        );
    }

    #[test]
    fn production_f16_schedule_matches_direct_torch_linspace_reference() {
        let actual = DmdSchedule::upstream_for_dtype(BooguTask::Generate, DType::F16);
        assert_eq!(
            actual.sigmas(),
            &[0.001_000_404_4, 0.250_732_42, 0.500_488_3, 0.75,]
        );
        assert_eq!(
            actual
                .sigmas()
                .iter()
                .map(|&sigma| f16::from_f32(sigma).to_f32())
                .collect::<Vec<_>>(),
            actual.sigmas()
        );
    }

    #[test]
    fn reference_bf16_schedule_matches_direct_torch_linspace_reference() {
        assert_eq!(
            DmdSchedule::upstream_for_dtype(BooguTask::Generate, DType::BF16).sigmas(),
            &[0.000_999_450_7, 0.251_953_13, 0.5, 0.75]
        );
    }
}
