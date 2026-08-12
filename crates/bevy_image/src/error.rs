use std::{error::Error, fmt::Display};

use serde::{Deserialize, Serialize};

/// Stable error category suitable for UI state and cross-thread messages.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrontendErrorKind {
    BackendUnavailable,
    ModelRuntime,
    InvalidRequest,
    ImageDecode,
    ImageEncode,
    UnsupportedImage,
    NativeIo,
    ArtifactProtocol,
    ArtifactIntegrity,
    ArtifactSink,
}

/// Cloneable frontend error. Detailed third-party errors are normalized to a
/// stable category plus an owned diagnostic message before entering ECS state.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrontendError {
    pub kind: FrontendErrorKind,
    pub message: String,
}

impl FrontendError {
    pub fn new(kind: FrontendErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub fn backend(message: impl Into<String>) -> Self {
        Self::new(FrontendErrorKind::BackendUnavailable, message)
    }

    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::new(FrontendErrorKind::InvalidRequest, message)
    }

    pub fn model_runtime(message: impl Into<String>) -> Self {
        Self::new(FrontendErrorKind::ModelRuntime, message)
    }
}

impl Display for FrontendError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for FrontendError {}

impl From<burn_image::ValidationError> for FrontendError {
    fn from(error: burn_image::ValidationError) -> Self {
        Self::invalid_request(error.to_string())
    }
}

impl From<burn_image::IntegrityError> for FrontendError {
    fn from(error: burn_image::IntegrityError) -> Self {
        Self::new(FrontendErrorKind::ArtifactIntegrity, error.to_string())
    }
}

impl From<burn_image::RuntimeError> for FrontendError {
    fn from(error: burn_image::RuntimeError) -> Self {
        Self::model_runtime(error.to_string())
    }
}

impl From<image::ImageError> for FrontendError {
    fn from(error: image::ImageError) -> Self {
        Self::new(FrontendErrorKind::ImageDecode, error.to_string())
    }
}

#[cfg(all(feature = "native-io", not(target_arch = "wasm32")))]
impl From<std::io::Error> for FrontendError {
    fn from(error: std::io::Error) -> Self {
        Self::new(FrontendErrorKind::NativeIo, error.to_string())
    }
}
