use thiserror::Error;

/// Errors raised by Boogu configuration, artifacts, or execution.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum BooguError {
    /// Cooperative cancellation observed at a model-defined safe boundary.
    #[error("inference was cancelled")]
    Cancelled,
    /// A request violates the fixed Turbo contract.
    #[error("invalid Turbo request: {0}")]
    InvalidRequest(String),
    /// A model configuration is internally inconsistent.
    #[error("invalid Boogu configuration: {0}")]
    InvalidConfig(String),
    /// A tensor or image shape is unsupported.
    #[error("invalid shape: {0}")]
    InvalidShape(String),
    /// Artifact metadata did not match the pinned model contract.
    #[error("artifact validation failed: {0}")]
    Artifact(String),
}
