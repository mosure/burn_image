use burn::{nn, prelude::*, tensor::activation::silu};

use super::linear::linear_forward;

/// Lumina SwiGLU feed-forward network.
#[derive(Module, Debug)]
pub struct LuminaFeedForward<B: Backend> {
    /// First gated projection.
    pub linear_1: nn::Linear<B>,
    /// Output projection.
    pub linear_2: nn::Linear<B>,
    /// Second gated projection.
    pub linear_3: nn::Linear<B>,
}

impl<B: Backend> LuminaFeedForward<B> {
    /// Create an FFN with an already rounded inner dimension.
    pub fn new(width: usize, inner: usize, device: &B::Device) -> Self {
        let no_bias = |input, output| {
            nn::LinearConfig::new(input, output)
                .with_bias(false)
                .init(device)
        };
        Self {
            linear_1: no_bias(width, inner),
            linear_2: no_bias(inner, width),
            linear_3: no_bias(width, inner),
        }
    }

    /// Apply SwiGLU and the output projection.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gated =
            silu(linear_forward(&self.linear_1, x.clone())) * linear_forward(&self.linear_3, x);
        linear_forward(&self.linear_2, gated)
    }
}
