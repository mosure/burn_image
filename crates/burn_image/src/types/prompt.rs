use serde::{Deserialize, Serialize};

use crate::ValidationError;

const MAX_PROMPT_BYTES: usize = 64 * 1024;

/// Prompt or edit instruction preserved byte-for-byte for model-specific processing.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Prompt(String);

impl Prompt {
    /// Validate prompt text without allocating an owned [`Prompt`].
    pub fn validate_text(value: &str) -> Result<(), ValidationError> {
        if value.trim().is_empty() {
            return Err(ValidationError::Empty { field: "prompt" });
        }
        if value.len() > MAX_PROMPT_BYTES {
            return Err(ValidationError::TooLong {
                field: "prompt",
                max: MAX_PROMPT_BYTES,
                actual: value.len(),
            });
        }
        Ok(())
    }

    pub fn new(value: impl Into<String>) -> Result<Self, ValidationError> {
        let value = value.into();
        Self::validate_text(&value)?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_inner(self) -> String {
        self.0
    }
}

impl TryFrom<String> for Prompt {
    type Error = ValidationError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<Prompt> for String {
    fn from(value: Prompt) -> Self {
        value.0
    }
}

impl TryFrom<&str> for Prompt {
    type Error = ValidationError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

#[cfg(test)]
mod tests {
    use super::Prompt;

    #[test]
    fn prompt_preserves_text_exactly_correctness() {
        let prompt = Prompt::new("  preserve interior and edge spacing  ").unwrap();
        assert_eq!(prompt.as_str(), "  preserve interior and edge spacing  ");
    }

    #[test]
    fn prompt_rejects_blank_text_correctness() {
        assert!(Prompt::new(" \n\t").is_err());
    }
}
