use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::{
    Dimensions, ImageRequest, ImageTaskKind, ModelId, NumericFormat, RuntimeError, ValidationError,
};

/// Inclusive constraints for model input/output dimensions.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DimensionConstraints {
    pub min_width: u32,
    pub max_width: u32,
    pub min_height: u32,
    pub max_height: u32,
    pub width_multiple: u32,
    pub height_multiple: u32,
    pub max_pixels: Option<u64>,
    /// Optional exact output sizes, applied in addition to the range and alignment constraints.
    /// Missing values deserialize as `None` and `None` is omitted from serialized descriptors.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_dimensions: Option<BTreeSet<Dimensions>>,
}

impl DimensionConstraints {
    pub fn validate(&self) -> Result<(), ValidationError> {
        for (field, value) in [
            ("dimensions.min_width", self.min_width),
            ("dimensions.max_width", self.max_width),
            ("dimensions.min_height", self.min_height),
            ("dimensions.max_height", self.max_height),
            ("dimensions.width_multiple", self.width_multiple),
            ("dimensions.height_multiple", self.height_multiple),
        ] {
            if value == 0 {
                return Err(ValidationError::MustBePositive { field });
            }
        }
        if self.min_width > self.max_width {
            return Err(ValidationError::OutOfRange {
                field: "dimensions.min_width",
                range: "0..=max_width",
                value: self.min_width.to_string(),
            });
        }
        if self.min_height > self.max_height {
            return Err(ValidationError::OutOfRange {
                field: "dimensions.min_height",
                range: "0..=max_height",
                value: self.min_height.to_string(),
            });
        }
        if self.max_pixels == Some(0) {
            return Err(ValidationError::MustBePositive {
                field: "dimensions.max_pixels",
            });
        }
        if let Some(allowed_dimensions) = &self.allowed_dimensions {
            if allowed_dimensions.is_empty() {
                return Err(ValidationError::Empty {
                    field: "dimensions.allowed_dimensions",
                });
            }
            for &dimensions in allowed_dimensions {
                if self.supports_envelope(dimensions).is_err() {
                    return Err(ValidationError::OutOfRange {
                        field: "dimensions.allowed_dimensions",
                        range: "the configured range, alignment, and pixel limits",
                        value: dimensions.to_string(),
                    });
                }
            }
        }
        Ok(())
    }

    pub fn supports(&self, dimensions: Dimensions) -> Result<(), String> {
        self.supports_envelope(dimensions)?;
        if let Some(allowed_dimensions) = &self.allowed_dimensions
            && !allowed_dimensions.contains(&dimensions)
        {
            let allowed = allowed_dimensions
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "dimensions must be one of [{allowed}] (got {dimensions})"
            ));
        }
        Ok(())
    }

    fn supports_envelope(&self, dimensions: Dimensions) -> Result<(), String> {
        let width = dimensions.width();
        let height = dimensions.height();
        if width < self.min_width || width > self.max_width {
            return Err(format!(
                "width must be in {}..={} (got {width})",
                self.min_width, self.max_width
            ));
        }
        if height < self.min_height || height > self.max_height {
            return Err(format!(
                "height must be in {}..={} (got {height})",
                self.min_height, self.max_height
            ));
        }
        if !width.is_multiple_of(self.width_multiple) {
            return Err(format!(
                "width must be a multiple of {} (got {width})",
                self.width_multiple
            ));
        }
        if !height.is_multiple_of(self.height_multiple) {
            return Err(format!(
                "height must be a multiple of {} (got {height})",
                self.height_multiple
            ));
        }
        if let Some(max_pixels) = self.max_pixels
            && dimensions.area() > max_pixels
        {
            return Err(format!(
                "pixel count must not exceed {max_pixels} (got {})",
                dimensions.area()
            ));
        }
        Ok(())
    }
}

/// Portable capabilities advertised by a concrete model adapter.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCapabilities {
    pub tasks: BTreeSet<ImageTaskKind>,
    pub supports_masks: bool,
    pub dimensions: DimensionConstraints,
    pub min_steps: u32,
    pub max_steps: u32,
    pub max_batch_size: u32,
    pub numeric_formats: BTreeSet<NumericFormat>,
}

impl ModelCapabilities {
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.tasks.is_empty() {
            return Err(ValidationError::Empty {
                field: "capabilities.tasks",
            });
        }
        self.dimensions.validate()?;
        if self.min_steps == 0 {
            return Err(ValidationError::MustBePositive {
                field: "capabilities.min_steps",
            });
        }
        if self.min_steps > self.max_steps {
            return Err(ValidationError::OutOfRange {
                field: "capabilities.min_steps",
                range: "1..=max_steps",
                value: self.min_steps.to_string(),
            });
        }
        if self.max_batch_size == 0 {
            return Err(ValidationError::MustBePositive {
                field: "capabilities.max_batch_size",
            });
        }
        if self.supports_masks && !self.tasks.contains(&ImageTaskKind::Edit) {
            return Err(ValidationError::OutOfRange {
                field: "capabilities.supports_masks",
                range: "false unless edit is supported",
                value: "true".to_string(),
            });
        }
        if self.numeric_formats.is_empty() {
            return Err(ValidationError::Empty {
                field: "capabilities.numeric_formats",
            });
        }
        Ok(())
    }

    pub fn validate_request(
        &self,
        model: &ModelId,
        request: &ImageRequest,
    ) -> Result<(), RuntimeError> {
        request.validate()?;
        let task = request.task_kind();
        if !self.tasks.contains(&task) {
            return Err(RuntimeError::UnsupportedTask {
                model: model.clone(),
                task,
            });
        }
        if request.has_mask() && !self.supports_masks {
            return Err(RuntimeError::MasksUnsupported {
                model: model.clone(),
            });
        }
        let options = request.options();
        if let Some(dimensions) = options.dimensions
            && let Err(reason) = self.dimensions.supports(dimensions)
        {
            return Err(RuntimeError::UnsupportedDimensions {
                model: model.clone(),
                requested: dimensions,
                reason,
            });
        }
        if let Some(steps) = options.steps
            && !(self.min_steps..=self.max_steps).contains(&steps)
        {
            return Err(RuntimeError::UnsupportedSteps {
                model: model.clone(),
                steps,
                min: self.min_steps,
                max: self.max_steps,
            });
        }
        if options.batch_size > self.max_batch_size {
            return Err(RuntimeError::UnsupportedBatchSize {
                model: model.clone(),
                requested: options.batch_size,
                max: self.max_batch_size,
            });
        }
        Ok(())
    }
}

/// Stable metadata and capabilities for a model adapter.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelDescriptor {
    pub id: ModelId,
    pub display_name: String,
    pub revision: String,
    pub capabilities: ModelCapabilities,
}

impl ModelDescriptor {
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.display_name.trim().is_empty() {
            return Err(ValidationError::Empty {
                field: "model.display_name",
            });
        }
        if self.revision.trim().is_empty() {
            return Err(ValidationError::Empty {
                field: "model.revision",
            });
        }
        self.capabilities.validate()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{DimensionConstraints, ModelCapabilities};
    use crate::{
        Dimensions, GenerateRequest, GenerationOptions, ImageRequest, ImageTaskKind, ModelId,
        NumericFormat, Prompt, RuntimeError,
    };

    fn capabilities() -> ModelCapabilities {
        ModelCapabilities {
            tasks: BTreeSet::from([ImageTaskKind::Generate]),
            supports_masks: false,
            dimensions: DimensionConstraints {
                min_width: 256,
                max_width: 1024,
                min_height: 256,
                max_height: 1024,
                width_multiple: 64,
                height_multiple: 64,
                max_pixels: Some(1024 * 1024),
                allowed_dimensions: None,
            },
            min_steps: 1,
            max_steps: 50,
            max_batch_size: 4,
            numeric_formats: BTreeSet::from([NumericFormat::F16]),
        }
    }

    #[test]
    fn dimensions_enforce_range_multiple_and_area_correctness() {
        let constraints = capabilities().dimensions;
        assert!(
            constraints
                .supports(Dimensions::new(512, 768).unwrap())
                .is_ok()
        );
        assert!(
            constraints
                .supports(Dimensions::new(513, 768).unwrap())
                .unwrap_err()
                .contains("multiple")
        );
        assert!(
            constraints
                .supports(Dimensions::new(128, 512).unwrap())
                .unwrap_err()
                .contains("width")
        );
    }

    #[test]
    fn model_capabilities_enforce_exact_dimension_allowlist_correctness() {
        let mut caps = capabilities();
        let allowed = Dimensions::new(512, 768).unwrap();
        caps.dimensions.allowed_dimensions = Some(BTreeSet::from([allowed]));
        let model = ModelId::new("test/model").unwrap();
        let request = |dimensions| {
            ImageRequest::Generate(GenerateRequest {
                prompt: Prompt::new("a red cube").unwrap(),
                negative_prompt: None,
                options: GenerationOptions {
                    dimensions: Some(dimensions),
                    ..GenerationOptions::default()
                },
            })
        };

        assert!(caps.validate_request(&model, &request(allowed)).is_ok());
        let rejected = Dimensions::new(576, 768).unwrap();
        assert!(matches!(
            caps.validate_request(&model, &request(rejected)),
            Err(RuntimeError::UnsupportedDimensions {
                requested,
                ..
            }) if requested == rejected
        ));
    }

    #[test]
    fn optional_dimension_allowlist_preserves_legacy_serde_correctness() {
        let dimensions = capabilities().dimensions;
        let serialized = serde_json::to_value(&dimensions).unwrap();
        assert!(serialized.get("allowed_dimensions").is_none());

        let legacy = serde_json::json!({
            "min_width": 256,
            "max_width": 1024,
            "min_height": 256,
            "max_height": 1024,
            "width_multiple": 64,
            "height_multiple": 64,
            "max_pixels": 1048576
        });
        assert_eq!(
            serde_json::from_value::<DimensionConstraints>(legacy).unwrap(),
            dimensions
        );
    }

    #[test]
    fn masks_require_edit_capability_correctness() {
        let mut caps = capabilities();
        caps.supports_masks = true;
        assert!(caps.validate().is_err());
        caps.tasks.insert(ImageTaskKind::Edit);
        assert!(caps.validate().is_ok());
    }
}
