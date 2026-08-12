//! Reusable model output types.

use burn::tensor::{Tensor, backend::Backend};

use crate::PositionDelta;

#[derive(Debug, Clone)]
pub struct Qwen3VlTextOutput<B: Backend> {
    pub last_hidden_state: Tensor<B, 3>,
    pub hidden_states: Option<Vec<Tensor<B, 3>>>,
    pub position_deltas: Vec<PositionDelta>,
}

#[derive(Debug, Clone)]
pub struct Qwen3VlVisionOutput<B: Backend> {
    /// Unmerged patch sequence after the final vision block, shape `[patches, vision_hidden]`.
    pub last_hidden_state: Tensor<B, 2>,
    /// Spatially merged language features, shape `[visual_tokens, text_hidden]`.
    pub pooler_output: Tensor<B, 2>,
    /// Merged features captured at the configured deep-stack vision blocks.
    pub deepstack_features: Vec<Tensor<B, 2>>,
}

#[derive(Debug, Clone)]
pub struct Qwen3VlModelOutput<B: Backend> {
    pub last_hidden_state: Tensor<B, 3>,
    pub hidden_states: Option<Vec<Tensor<B, 3>>>,
    pub vision_output: Option<Qwen3VlVisionOutput<B>>,
    pub position_deltas: Vec<PositionDelta>,
}

#[derive(Debug, Clone)]
pub struct Qwen3VlCausalLmOutput<B: Backend> {
    pub logits: Tensor<B, 3>,
    pub hidden_states: Option<Vec<Tensor<B, 3>>>,
    pub vision_output: Option<Qwen3VlVisionOutput<B>>,
    pub position_deltas: Vec<PositionDelta>,
}
