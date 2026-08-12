use thiserror::Error;

/// Errors produced while configuring, preprocessing, or executing Qwen3-VL.
#[derive(Debug, Error)]
pub enum Qwen3VlError {
    #[error("invalid Qwen3-VL configuration: {0}")]
    InvalidConfig(String),
    #[error("invalid multimodal input: {0}")]
    InvalidInput(String),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("weight inventory error: {0}")]
    Weights(String),
    #[error("checkpoint error: {0}")]
    Checkpoint(String),
    #[error("failed to parse configuration JSON: {0}")]
    ConfigJson(#[from] serde_json::Error),
}

pub type Result<T> = core::result::Result<T, Qwen3VlError>;
