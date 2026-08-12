use std::{fmt::Display, str::FromStr};

use serde::{Deserialize, Serialize};

use crate::ValidationError;

const MAX_MODEL_ID_BYTES: usize = 256;

/// Stable, transport-safe identifier for a model adapter or published model.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ModelId(String);

impl ModelId {
    pub fn new(value: impl Into<String>) -> Result<Self, ValidationError> {
        let value = value.into();
        validate_identifier("model_id", &value, MAX_MODEL_ID_BYTES, true)?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl FromStr for ModelId {
    type Err = ValidationError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::new(value)
    }
}

impl TryFrom<String> for ModelId {
    type Error = ValidationError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<ModelId> for String {
    fn from(value: ModelId) -> Self {
        value.0
    }
}

/// Generation or image-editing operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageTaskKind {
    Generate,
    Edit,
}

impl Display for ImageTaskKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Generate => f.write_str("generate"),
            Self::Edit => f.write_str("edit"),
        }
    }
}

/// Numeric storage/compute format advertised by a model artifact profile.
///
/// `Other` is intentionally explicit: support for a new format must be
/// advertised by name rather than silently aliasing it to a wider format.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NumericFormat {
    F32,
    F16,
    Bf16,
    I8,
    U8,
    Other(String),
}

impl NumericFormat {
    pub fn validate(&self) -> Result<(), ValidationError> {
        if let Self::Other(name) = self {
            validate_identifier("numeric_format", name, 64, false)?;
        }
        Ok(())
    }
}

pub(crate) fn validate_identifier(
    field: &'static str,
    value: &str,
    max: usize,
    allow_slash: bool,
) -> Result<(), ValidationError> {
    if value.is_empty() {
        return Err(ValidationError::Empty { field });
    }
    if value.len() > max {
        return Err(ValidationError::TooLong {
            field,
            max,
            actual: value.len(),
        });
    }
    if value.trim() != value {
        return Err(ValidationError::InvalidCharacter { field, index: 0 });
    }
    for (index, byte) in value.bytes().enumerate() {
        let valid = byte.is_ascii_alphanumeric()
            || matches!(byte, b'-' | b'_' | b'.' | b':')
            || (allow_slash && byte == b'/');
        if !valid {
            return Err(ValidationError::InvalidCharacter { field, index });
        }
    }
    if value.contains("//") || value.split('/').any(|segment| segment == "..") {
        return Err(ValidationError::InvalidCharacter { field, index: 0 });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::ModelId;

    #[test]
    fn model_id_accepts_repository_style_names_correctness() {
        let id = ModelId::new("boogu-project/Boogu-Image-0.1-Turbo").unwrap();
        assert_eq!(id.as_str(), "boogu-project/Boogu-Image-0.1-Turbo");
    }

    #[test]
    fn model_id_rejects_whitespace_and_parent_segments_correctness() {
        assert!(ModelId::new(" model").is_err());
        assert!(ModelId::new("owner/../model").is_err());
    }
}
