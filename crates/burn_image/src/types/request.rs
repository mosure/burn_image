use serde::{Deserialize, Serialize};

use crate::{Dimensions, ImageTaskKind, InputImage, InputMask, Prompt, ValidationError};

/// Model-neutral inference controls. `None` delegates to the selected model's
/// documented default; supplied values are never silently rewritten.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenerationOptions {
    pub dimensions: Option<Dimensions>,
    pub steps: Option<u32>,
    pub guidance_scale: Option<f32>,
    pub seed: Option<u64>,
    pub batch_size: u32,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            dimensions: None,
            steps: None,
            guidance_scale: None,
            seed: None,
            batch_size: 1,
        }
    }
}

impl GenerationOptions {
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.steps == Some(0) {
            return Err(ValidationError::MustBePositive { field: "steps" });
        }
        if let Some(guidance) = self.guidance_scale {
            if !guidance.is_finite() {
                return Err(ValidationError::NonFinite {
                    field: "guidance_scale",
                    value: guidance.to_string(),
                });
            }
            if guidance < 0.0 {
                return Err(ValidationError::OutOfRange {
                    field: "guidance_scale",
                    range: "0..",
                    value: guidance.to_string(),
                });
            }
        }
        if self.batch_size == 0 {
            return Err(ValidationError::MustBePositive {
                field: "batch_size",
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub prompt: Prompt,
    pub negative_prompt: Option<Prompt>,
    pub options: GenerationOptions,
}

impl GenerateRequest {
    pub fn validate(&self) -> Result<(), ValidationError> {
        self.options.validate()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EditRequest {
    pub source: InputImage,
    pub instruction: Prompt,
    pub negative_prompt: Option<Prompt>,
    pub mask: Option<InputMask>,
    pub strength: Option<f32>,
    pub options: GenerationOptions,
}

impl EditRequest {
    pub fn validate(&self) -> Result<(), ValidationError> {
        self.options.validate()?;
        if let Some(strength) = self.strength {
            if !strength.is_finite() {
                return Err(ValidationError::NonFinite {
                    field: "edit.strength",
                    value: strength.to_string(),
                });
            }
            if !(0.0..=1.0).contains(&strength) {
                return Err(ValidationError::OutOfRange {
                    field: "edit.strength",
                    range: "0..=1",
                    value: strength.to_string(),
                });
            }
        }
        if let (Some(source), Some(mask)) = (self.source.dimensions(), self.mask.as_ref())
            && source != mask.dimensions()
        {
            return Err(ValidationError::MaskDimensionMismatch {
                mask: mask.dimensions(),
                source_dimensions: source,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "task", rename_all = "snake_case")]
pub enum ImageRequest {
    Generate(GenerateRequest),
    Edit(EditRequest),
}

impl ImageRequest {
    pub fn task_kind(&self) -> ImageTaskKind {
        match self {
            Self::Generate(_) => ImageTaskKind::Generate,
            Self::Edit(_) => ImageTaskKind::Edit,
        }
    }

    pub fn options(&self) -> &GenerationOptions {
        match self {
            Self::Generate(request) => &request.options,
            Self::Edit(request) => &request.options,
        }
    }

    pub fn has_mask(&self) -> bool {
        matches!(self, Self::Edit(request) if request.mask.is_some())
    }

    pub fn validate(&self) -> Result<(), ValidationError> {
        match self {
            Self::Generate(request) => request.validate(),
            Self::Edit(request) => request.validate(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{EditRequest, GenerationOptions};
    use crate::{
        ColorSpace, Dimensions, InputImage, InputMask, MaskSemantics, PixelBuffer, PixelFormat,
        Prompt,
    };

    #[test]
    fn generation_options_reject_nonfinite_guidance_correctness() {
        let options = GenerationOptions {
            guidance_scale: Some(f32::NAN),
            ..GenerationOptions::default()
        };
        assert!(options.validate().is_err());
    }

    #[test]
    fn edit_mask_must_match_known_source_dimensions_correctness() {
        let source_dimensions = Dimensions::new(2, 2).unwrap();
        let mask_dimensions = Dimensions::new(1, 2).unwrap();
        let request = EditRequest {
            source: InputImage::Pixels(
                PixelBuffer::new(
                    source_dimensions,
                    PixelFormat::Rgba8,
                    ColorSpace::Srgb,
                    vec![0; 16],
                )
                .unwrap(),
            ),
            instruction: Prompt::new("replace the sky").unwrap(),
            negative_prompt: None,
            mask: Some(
                InputMask::new(mask_dimensions, MaskSemantics::WhiteEdits, vec![0; 2]).unwrap(),
            ),
            strength: Some(0.75),
            options: GenerationOptions::default(),
        };
        assert!(request.validate().is_err());
    }
}
