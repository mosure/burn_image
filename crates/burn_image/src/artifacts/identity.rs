use std::{fmt::Display, str::FromStr};

use serde::{Deserialize, Serialize};

use crate::{ValidationError, types::identity::validate_identifier};

const MAX_ARTIFACT_PATH_BYTES: usize = 1024;

/// Validated relative artifact path suitable for joining to a directory or URL.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ArtifactPath(String);

impl ArtifactPath {
    pub fn new(value: impl Into<String>) -> Result<Self, ValidationError> {
        let value = value.into();
        if value.is_empty() {
            return Err(ValidationError::Empty {
                field: "artifact_path",
            });
        }
        if value.len() > MAX_ARTIFACT_PATH_BYTES {
            return Err(ValidationError::TooLong {
                field: "artifact_path",
                max: MAX_ARTIFACT_PATH_BYTES,
                actual: value.len(),
            });
        }
        if value.starts_with('/') || value.ends_with('/') || value.contains('\\') {
            return Err(ValidationError::InvalidCharacter {
                field: "artifact_path",
                index: 0,
            });
        }
        for (index, byte) in value.bytes().enumerate() {
            let valid =
                byte.is_ascii_alphanumeric() || matches!(byte, b'/' | b'-' | b'_' | b'.' | b'~');
            if !valid {
                return Err(ValidationError::InvalidCharacter {
                    field: "artifact_path",
                    index,
                });
            }
        }
        if value
            .split('/')
            .any(|segment| segment.is_empty() || segment == "." || segment == "..")
        {
            return Err(ValidationError::InvalidCharacter {
                field: "artifact_path",
                index: 0,
            });
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Display for ArtifactPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl FromStr for ArtifactPath {
    type Err = ValidationError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::new(value)
    }
}

impl TryFrom<String> for ArtifactPath {
    type Error = ValidationError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<ArtifactPath> for String {
    fn from(value: ArtifactPath) -> Self {
        value.0
    }
}

macro_rules! identifier_type {
    ($name:ident, $field:literal, $max:expr) => {
        #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(try_from = "String", into = "String")]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self, ValidationError> {
                let value = value.into();
                validate_identifier($field, &value, $max, false)?;
                Ok(Self(value))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl TryFrom<String> for $name {
            type Error = ValidationError;

            fn try_from(value: String) -> Result<Self, Self::Error> {
                Self::new(value)
            }
        }

        impl From<$name> for String {
            fn from(value: $name) -> Self {
                value.0
            }
        }
    };
}

identifier_type!(ArtifactBundleId, "artifact_bundle_id", 128);
identifier_type!(ArtifactComponentId, "artifact_component_id", 128);
identifier_type!(ArtifactProfileId, "artifact_profile_id", 128);

#[cfg(test)]
mod tests {
    use super::ArtifactPath;

    #[test]
    fn artifact_path_rejects_traversal_and_urls_correctness() {
        assert!(ArtifactPath::new("weights/part-000.bpk").is_ok());
        assert!(ArtifactPath::new("../secret").is_err());
        assert!(ArtifactPath::new("https://example.com/model.bpk").is_err());
        assert!(ArtifactPath::new("weights\\part.bpk").is_err());
    }
}
