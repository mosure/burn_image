use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::{EncodedImage, ModelId, NumericFormat, PixelBuffer, Sha256Digest, ValidationError};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HostImage {
    Pixels(PixelBuffer),
    Encoded(EncodedImage),
}

impl HostImage {
    pub fn dimensions(&self) -> Option<crate::Dimensions> {
        match self {
            Self::Pixels(pixels) => Some(pixels.dimensions()),
            Self::Encoded(encoded) => encoded.dimensions(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeneratedImage {
    pub index: u32,
    pub image: HostImage,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageTiming {
    pub stage: String,
    pub elapsed_micros: u64,
}

impl StageTiming {
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.stage.trim().is_empty() {
            return Err(ValidationError::Empty {
                field: "timing.stage",
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageTimings {
    pub stages: Vec<StageTiming>,
    pub total_micros: u64,
}

impl StageTimings {
    pub fn validate(&self) -> Result<(), ValidationError> {
        let mut names = BTreeSet::new();
        let mut sum = 0u64;
        for timing in &self.stages {
            timing.validate()?;
            if !names.insert(timing.stage.as_str()) {
                return Err(ValidationError::InvalidTimingInterval {
                    stage: timing.stage.clone(),
                });
            }
            sum = sum.saturating_add(timing.elapsed_micros);
        }
        if sum > self.total_micros {
            return Err(ValidationError::InvalidTimingInterval {
                stage: "total".to_string(),
            });
        }
        Ok(())
    }
}

/// Reproducibility metadata attached to a completed result.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelProvenance {
    pub model: ModelId,
    pub model_revision: String,
    pub artifact_content_digest: Option<Sha256Digest>,
    pub numeric_format: NumericFormat,
    pub backend: String,
    pub artifacts_verified: bool,
}

impl ModelProvenance {
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.model_revision.trim().is_empty() {
            return Err(ValidationError::Empty {
                field: "provenance.model_revision",
            });
        }
        if self.backend.trim().is_empty() {
            return Err(ValidationError::Empty {
                field: "provenance.backend",
            });
        }
        self.numeric_format.validate()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageOutput {
    pub images: Vec<GeneratedImage>,
    pub seed: u64,
    pub timings: StageTimings,
    pub provenance: ModelProvenance,
}

impl ImageOutput {
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.images.is_empty() {
            return Err(ValidationError::EmptyOutput);
        }
        let mut indices = BTreeSet::new();
        for output in &self.images {
            if !indices.insert(output.index) {
                return Err(ValidationError::DuplicateOutputIndex {
                    index: output.index,
                });
            }
        }
        self.timings.validate()?;
        self.provenance.validate()
    }
}

#[cfg(test)]
mod tests {
    use super::{StageTiming, StageTimings};

    #[test]
    fn stage_timings_must_fit_inside_total_correctness() {
        let timings = StageTimings {
            stages: vec![StageTiming {
                stage: "decode".to_string(),
                elapsed_micros: 11,
            }],
            total_micros: 10,
        };
        assert!(timings.validate().is_err());
    }
}
