//! Qwen3-VL grouped-query causal language decoder.

use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, RmsNorm, RmsNormConfig},
    tensor::{
        Bool, DType, IndexingUpdateOp, Int, Tensor, TensorData, activation, backend::Backend,
    },
};

use crate::{
    MropePositionIds, Qwen3VlError, QwenLinear, QwenLinearConfig, Result,
    config::Qwen3VlTextConfig, outputs::Qwen3VlTextOutput,
};

const DEFAULT_QUERY_CHUNK_SIZE: usize = 128;

/// Per-layer visual features added at the flattened multimodal token locations.
#[derive(Debug)]
pub struct DeepstackEmbeddings<B: Backend> {
    pub token_indices: Vec<usize>,
    pub features: Vec<Tensor<B, 2>>,
}

#[derive(Module, Debug)]
pub struct Qwen3VlTextMlp<B: Backend> {
    pub gate_proj: QwenLinear<B>,
    pub up_proj: QwenLinear<B>,
    pub down_proj: QwenLinear<B>,
}

impl<B: Backend> Qwen3VlTextMlp<B> {
    pub fn new(config: &Qwen3VlTextConfig, device: &B::Device) -> Self {
        let linear = |input, output| {
            QwenLinearConfig::new(input, output)
                .with_bias(false)
                .init(device)
        };
        Self {
            gate_proj: linear(config.hidden_size, config.intermediate_size),
            up_proj: linear(config.hidden_size, config.intermediate_size),
            down_proj: linear(config.intermediate_size, config.hidden_size),
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        self.down_proj.forward(
            activation::silu(self.gate_proj.forward(input.clone())) * self.up_proj.forward(input),
        )
    }
}

#[derive(Module, Debug)]
pub struct Qwen3VlTextAttention<B: Backend> {
    pub q_proj: QwenLinear<B>,
    pub k_proj: QwenLinear<B>,
    pub v_proj: QwenLinear<B>,
    pub o_proj: QwenLinear<B>,
    pub q_norm: RmsNorm<B>,
    pub k_norm: RmsNorm<B>,
    #[module(skip)]
    num_heads: usize,
    #[module(skip)]
    num_key_value_heads: usize,
    #[module(skip)]
    head_dim: usize,
    #[module(skip)]
    query_chunk_size: usize,
}

impl<B: Backend> Qwen3VlTextAttention<B> {
    pub fn new(config: &Qwen3VlTextConfig, device: &B::Device) -> Self {
        let linear = |input, output| {
            QwenLinearConfig::new(input, output)
                .with_bias(false)
                .init(device)
        };
        let head_dim = config.head_dim();
        Self {
            q_proj: linear(config.hidden_size, config.num_attention_heads * head_dim),
            k_proj: linear(config.hidden_size, config.num_key_value_heads * head_dim),
            v_proj: linear(config.hidden_size, config.num_key_value_heads * head_dim),
            o_proj: linear(config.hidden_size, config.hidden_size),
            q_norm: RmsNormConfig::new(head_dim)
                .with_epsilon(config.rms_norm_eps)
                .init(device),
            k_norm: RmsNormConfig::new(head_dim)
                .with_epsilon(config.rms_norm_eps)
                .init(device),
            num_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim,
            query_chunk_size: DEFAULT_QUERY_CHUNK_SIZE,
        }
    }

    pub fn set_query_chunk_size(&mut self, query_chunk_size: usize) {
        self.query_chunk_size = query_chunk_size.max(1);
    }

    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        cos: Tensor<B, 3>,
        sin: Tensor<B, 3>,
        attention_mask: Option<&Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let [batch, sequence, _] = hidden_states.dims();
        let query = self
            .q_norm
            .forward(self.q_proj.forward(hidden_states.clone()).reshape([
                batch,
                sequence,
                self.num_heads,
                self.head_dim,
            ]));
        let key = self
            .k_norm
            .forward(self.k_proj.forward(hidden_states.clone()).reshape([
                batch,
                sequence,
                self.num_key_value_heads,
                self.head_dim,
            ]));
        let value = self.v_proj.forward(hidden_states).reshape([
            batch,
            sequence,
            self.num_key_value_heads,
            self.head_dim,
        ]);
        let (query, key) = apply_text_rope(query, key, cos, sin);
        let query = query.swap_dims(1, 2);
        let key = repeat_key_value(
            key.swap_dims(1, 2),
            self.num_heads / self.num_key_value_heads,
        );
        let value = repeat_key_value(
            value.swap_dims(1, 2),
            self.num_heads / self.num_key_value_heads,
        );
        let key_transposed = key.swap_dims(2, 3);

        let dtype = query.dtype();
        let mut chunks = Vec::new();
        let mut start = 0;
        while start < sequence {
            let end = (start + self.query_chunk_size).min(sequence);
            let chunk =
                query
                    .clone()
                    .slice([0..batch, 0..self.num_heads, start..end, 0..self.head_dim]);
            // For local query row `r`, offset `start + 1` selects global key columns greater
            // than `start + r`. Building this per chunk bounds mask memory to O(chunk*sequence).
            // Burn's triangular mask is false inside the selected triangle, hence the inversion.
            let mut mask = Tensor::<B, 3, Bool>::triu_mask(
                [batch, end - start, sequence],
                (start + 1) as i64,
                &query.device(),
            )
            .bool_not();
            if let Some(valid) = attention_mask {
                let invalid_keys = valid
                    .clone()
                    .bool_not()
                    .reshape([batch, 1, sequence])
                    .expand([batch as i64, (end - start) as i64, sequence as i64]);
                mask = mask.bool_or(invalid_keys);
            }
            let mask = mask.unsqueeze_dim::<4>(1).expand([
                batch as i64,
                self.num_heads as i64,
                (end - start) as i64,
                sequence as i64,
            ]);
            let scores = chunk
                .cast(DType::F32)
                .matmul(key_transposed.clone().cast(DType::F32))
                .mul_scalar(1.0 / (self.head_dim as f64).sqrt())
                .mask_fill(mask, -1.0e30_f32);
            let probabilities = activation::softmax(scores, 3).cast(dtype);
            chunks.push(probabilities.matmul(value.clone()));
            start = end;
        }
        let output = Tensor::cat(chunks, 2).swap_dims(1, 2).reshape([
            batch,
            sequence,
            self.num_heads * self.head_dim,
        ]);
        self.o_proj.forward(output)
    }
}

#[derive(Module, Debug)]
pub struct Qwen3VlDecoderLayer<B: Backend> {
    pub self_attn: Qwen3VlTextAttention<B>,
    pub mlp: Qwen3VlTextMlp<B>,
    pub input_layernorm: RmsNorm<B>,
    pub post_attention_layernorm: RmsNorm<B>,
}

impl<B: Backend> Qwen3VlDecoderLayer<B> {
    pub fn new(config: &Qwen3VlTextConfig, device: &B::Device) -> Self {
        Self {
            self_attn: Qwen3VlTextAttention::new(config, device),
            mlp: Qwen3VlTextMlp::new(config, device),
            input_layernorm: RmsNormConfig::new(config.hidden_size)
                .with_epsilon(config.rms_norm_eps)
                .init(device),
            post_attention_layernorm: RmsNormConfig::new(config.hidden_size)
                .with_epsilon(config.rms_norm_eps)
                .init(device),
        }
    }

    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        cos: Tensor<B, 3>,
        sin: Tensor<B, 3>,
        attention_mask: Option<&Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let hidden_states = hidden_states.clone()
            + self.self_attn.forward(
                self.input_layernorm.forward(hidden_states),
                cos,
                sin,
                attention_mask,
            );
        hidden_states.clone()
            + self
                .mlp
                .forward(self.post_attention_layernorm.forward(hidden_states))
    }
}

/// Ordinary Qwen3-VL language model without a task-specific head.
#[derive(Module, Debug)]
pub struct Qwen3VlTextModel<B: Backend> {
    pub embed_tokens: Embedding<B>,
    pub layers: Vec<Qwen3VlDecoderLayer<B>>,
    pub norm: RmsNorm<B>,
    #[module(skip)]
    config: Qwen3VlTextConfig,
}

impl<B: Backend> Qwen3VlTextModel<B> {
    pub fn new(config: Qwen3VlTextConfig, device: &B::Device) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            embed_tokens: EmbeddingConfig::new(config.vocab_size, config.hidden_size).init(device),
            layers: (0..config.num_hidden_layers)
                .map(|_| Qwen3VlDecoderLayer::new(&config, device))
                .collect(),
            norm: RmsNormConfig::new(config.hidden_size)
                .with_epsilon(config.rms_norm_eps)
                .init(device),
            config,
        })
    }

    pub fn config(&self) -> &Qwen3VlTextConfig {
        &self.config
    }

    /// Bound the number of query rows in each attention score tile.
    pub fn set_query_chunk_size(&mut self, query_chunk_size: usize) {
        for layer in &mut self.layers {
            layer.self_attn.set_query_chunk_size(query_chunk_size);
        }
    }

    pub fn embed(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.embed_tokens.forward(input_ids)
    }

    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Option<&Tensor<B, 2, Bool>>,
        position_ids: Option<&MropePositionIds>,
        deepstack: Option<&DeepstackEmbeddings<B>>,
        output_hidden_states: bool,
    ) -> Result<Qwen3VlTextOutput<B>> {
        self.forward_embeddings(
            self.embed(input_ids),
            attention_mask,
            position_ids,
            deepstack,
            output_hidden_states,
        )
    }

    pub fn forward_embeddings(
        &self,
        mut hidden_states: Tensor<B, 3>,
        attention_mask: Option<&Tensor<B, 2, Bool>>,
        position_ids: Option<&MropePositionIds>,
        deepstack: Option<&DeepstackEmbeddings<B>>,
        output_hidden_states: bool,
    ) -> Result<Qwen3VlTextOutput<B>> {
        let [batch, sequence, hidden] = hidden_states.dims();
        if hidden != self.config.hidden_size {
            return Err(Qwen3VlError::InvalidInput(format!(
                "text embeddings have hidden size {hidden}, expected {}",
                self.config.hidden_size
            )));
        }
        if let Some(mask) = attention_mask
            && mask.dims() != [batch, sequence]
        {
            return Err(Qwen3VlError::InvalidInput(
                "attention mask shape must match input ids".into(),
            ));
        }
        let default_positions;
        let positions = if let Some(positions) = position_ids {
            if positions.batch_size() != batch || positions.sequence_length() != sequence {
                return Err(Qwen3VlError::InvalidInput(
                    "MRoPE plan shape must match text embeddings".into(),
                ));
            }
            positions
        } else {
            default_positions = MropePositionIds::text_only(batch, sequence);
            &default_positions
        };
        if let Some(deepstack) = deepstack {
            if deepstack.features.len() > self.layers.len() {
                return Err(Qwen3VlError::InvalidInput(
                    "more deep-stack features than text decoder layers".into(),
                ));
            }
            for feature in &deepstack.features {
                if feature.dims() != [deepstack.token_indices.len(), hidden] {
                    return Err(Qwen3VlError::InvalidInput(
                        "deep-stack feature shape does not match token indices and hidden size"
                            .into(),
                    ));
                }
            }
            if deepstack
                .token_indices
                .iter()
                .any(|&index| index >= batch * sequence)
            {
                return Err(Qwen3VlError::InvalidInput(
                    "deep-stack token index is outside the flattened text batch".into(),
                ));
            }
        }
        let (cos, sin) = positions.cos_sin::<B>(&self.config, &hidden_states.device())?;
        let mut all_hidden = output_hidden_states.then(Vec::new);
        for (index, layer) in self.layers.iter().enumerate() {
            if let Some(values) = &mut all_hidden {
                values.push(hidden_states.clone());
            }
            hidden_states = layer.forward(hidden_states, cos.clone(), sin.clone(), attention_mask);
            if let Some(deepstack) = deepstack
                && let Some(feature) = deepstack.features.get(index)
            {
                let indices = deepstack
                    .token_indices
                    .iter()
                    .map(|&value| value as i64)
                    .collect::<Vec<_>>();
                let indices = Tensor::<B, 1, Int>::from_data(
                    TensorData::new(indices, [deepstack.token_indices.len()]),
                    &hidden_states.device(),
                );
                hidden_states = hidden_states
                    .reshape([batch * sequence, hidden])
                    .select_assign(0, indices, feature.clone(), IndexingUpdateOp::Add)
                    .reshape([batch, sequence, hidden]);
            }
        }
        hidden_states = self.norm.forward(hidden_states);
        if let Some(values) = &mut all_hidden {
            values.push(hidden_states.clone());
        }
        Ok(Qwen3VlTextOutput {
            last_hidden_state: hidden_states,
            hidden_states: all_hidden,
            position_deltas: positions.deltas().to_vec(),
        })
    }
}

fn repeat_key_value<B: Backend>(tensor: Tensor<B, 4>, groups: usize) -> Tensor<B, 4> {
    if groups == 1 {
        return tensor;
    }
    let [batch, key_value_heads, sequence, head_dim] = tensor.dims();
    tensor
        .unsqueeze_dim::<5>(2)
        .expand([
            batch as i64,
            key_value_heads as i64,
            groups as i64,
            sequence as i64,
            head_dim as i64,
        ])
        .reshape([batch, key_value_heads * groups, sequence, head_dim])
}

fn rotate_half<B: Backend>(tensor: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, heads, sequence, head_dim] = tensor.dims();
    let half = head_dim / 2;
    Tensor::cat(
        vec![
            tensor
                .clone()
                .slice([0..batch, 0..heads, 0..sequence, half..head_dim])
                .neg(),
            tensor.slice([0..batch, 0..heads, 0..sequence, 0..half]),
        ],
        3,
    )
}

fn apply_text_rope<B: Backend>(
    query: Tensor<B, 4>,
    key: Tensor<B, 4>,
    cos: Tensor<B, 3>,
    sin: Tensor<B, 3>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let query_dtype = query.dtype();
    let key_dtype = key.dtype();
    let cos = cos.unsqueeze_dim::<4>(2).cast(DType::F32);
    let sin = sin.unsqueeze_dim::<4>(2).cast(DType::F32);
    let query_float = query.cast(DType::F32);
    let key_float = key.cast(DType::F32);
    let rotated_query = rotate_half(query_float.clone());
    let rotated_key = rotate_half(key_float.clone());
    (
        (query_float * cos.clone() + rotated_query * sin.clone()).cast(query_dtype),
        (key_float * cos + rotated_key * sin).cast(key_dtype),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::tiny_config;
    use burn_ndarray::NdArray;

    #[test]
    fn tiny_text_forward_is_finite_smoke() {
        type B = NdArray<f32>;
        let config = tiny_config().text_config;
        let device = Default::default();
        B::seed(&device, 7);
        let mut model = Qwen3VlTextModel::<B>::new(config, &device).unwrap();
        for layer in &mut model.layers {
            layer.self_attn.set_query_chunk_size(2);
        }
        let ids = Tensor::<B, 2, Int>::from_data([[1, 2, 3, 4]], &device);
        let mask = Tensor::<B, 2, Bool>::from_data([[true, true, true, true]], &device);
        let output = model.forward(ids, Some(&mask), None, None, true).unwrap();
        assert_eq!(output.last_hidden_state.dims(), [1, 4, 8]);
        assert_eq!(output.hidden_states.unwrap().len(), 3);
        assert!(
            output
                .last_hidden_state
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn causal_prefix_does_not_depend_on_future_tokens_correctness() {
        type B = NdArray<f32>;
        let config = tiny_config().text_config;
        let device = Default::default();
        B::seed(&device, 13);
        let mut model = Qwen3VlTextModel::<B>::new(config, &device).unwrap();
        for layer in &mut model.layers {
            layer.self_attn.set_query_chunk_size(2);
        }
        let first = model
            .forward(
                Tensor::<B, 2, Int>::from_data([[1, 2, 3, 4]], &device),
                None,
                None,
                None,
                false,
            )
            .unwrap()
            .last_hidden_state
            .slice([0..1, 0..3, 0..8])
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let second = model
            .forward(
                Tensor::<B, 2, Int>::from_data([[1, 2, 3, 47]], &device),
                None,
                None,
                None,
                false,
            )
            .unwrap()
            .last_hidden_state
            .slice([0..1, 0..3, 0..8])
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for (left, right) in first.iter().zip(second) {
            assert!((left - right).abs() < 1e-5, "{left} != {right}");
        }
    }
}
