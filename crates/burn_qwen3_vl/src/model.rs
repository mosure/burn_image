//! Composition of the ordinary Qwen3-VL vision and language towers.

use burn::{
    module::Module,
    tensor::{Bool, DType, IndexingUpdateOp, Int, Tensor, TensorData, backend::Backend},
};

use crate::{
    MropePositionIds, Qwen3VlConfig, Qwen3VlError, QwenLinear, QwenLinearConfig, Result,
    outputs::{Qwen3VlCausalLmOutput, Qwen3VlModelOutput, Qwen3VlVisionOutput},
    processor::Grid,
    text::{DeepstackEmbeddings, Qwen3VlTextModel},
    vision::Qwen3VlVisionModel,
};

/// One modality's flattened preprocessed patches and their language-token destinations.
#[derive(Debug, Clone)]
pub struct Qwen3VlVisualInput<B: Backend> {
    /// Shape `[sum(t*h*w), channels*temporal_patch*patch*patch]`.
    pub patches: Tensor<B, 2>,
    pub grids: Vec<Grid>,
    /// Flattened `[batch, sequence]` positions of the post-merge placeholder tokens.
    pub token_indices: Vec<usize>,
}

/// Complete ordinary Qwen3-VL model input.
#[derive(Debug, Clone)]
pub struct Qwen3VlModelInput<B: Backend> {
    pub input_ids: Tensor<B, 2, Int>,
    /// `true` for valid tokens and `false` for padding.
    pub attention_mask: Option<Tensor<B, 2, Bool>>,
    pub position_ids: Option<MropePositionIds>,
    pub images: Option<Qwen3VlVisualInput<B>>,
    pub videos: Option<Qwen3VlVisualInput<B>>,
    pub output_hidden_states: bool,
}

impl<B: Backend> Qwen3VlModelInput<B> {
    pub fn text(input_ids: Tensor<B, 2, Int>) -> Self {
        Self {
            input_ids,
            attention_mask: None,
            position_ids: None,
            images: None,
            videos: None,
            output_hidden_states: false,
        }
    }
}

/// Base ordinary Qwen3-VL model. Checkpoint field names intentionally mirror Hugging Face.
#[derive(Module, Debug)]
pub struct Qwen3VlModel<B: Backend> {
    pub language_model: Qwen3VlTextModel<B>,
    pub visual: Qwen3VlVisionModel<B>,
    #[module(skip)]
    config: Qwen3VlConfig,
}

impl<B: Backend> Qwen3VlModel<B> {
    pub fn new(config: Qwen3VlConfig, device: &B::Device) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            language_model: Qwen3VlTextModel::new(config.text_config.clone(), device)?,
            visual: Qwen3VlVisionModel::new(config.vision_config.clone(), device)?,
            config,
        })
    }

    pub fn config(&self) -> &Qwen3VlConfig {
        &self.config
    }

    /// Set the shared query-tile bound for both language and vision attention.
    pub fn set_query_chunk_size(&mut self, query_chunk_size: usize) {
        self.language_model.set_query_chunk_size(query_chunk_size);
        self.visual.set_query_chunk_size(query_chunk_size);
    }

    pub fn forward(&self, input: Qwen3VlModelInput<B>) -> Result<Qwen3VlModelOutput<B>> {
        let [batch, sequence] = input.input_ids.dims();
        let mut embeddings = self.language_model.embed(input.input_ids);
        let text_dtype = embeddings.dtype();
        let mut visual_outputs = Vec::new();
        let mut visual_indices = Vec::new();
        let mut deepstack_by_layer =
            vec![Vec::new(); self.config.vision_config.deepstack_visual_indexes.len()];

        if let Some(images) = input.images {
            let output = self.visual.forward(images.patches, &images.grids)?;
            validate_visual_destinations(
                &output,
                &images.token_indices,
                batch * sequence,
                self.config.text_config.hidden_size,
            )?;
            embeddings = assign_visual_features(
                embeddings,
                &images.token_indices,
                output.pooler_output.clone().cast(text_dtype),
            );
            append_deepstack(&mut deepstack_by_layer, &output, text_dtype)?;
            visual_indices.extend(images.token_indices);
            visual_outputs.push(output);
        }
        if let Some(videos) = input.videos {
            let output = self.visual.forward(videos.patches, &videos.grids)?;
            validate_visual_destinations(
                &output,
                &videos.token_indices,
                batch * sequence,
                self.config.text_config.hidden_size,
            )?;
            embeddings = assign_visual_features(
                embeddings,
                &videos.token_indices,
                output.pooler_output.clone().cast(text_dtype),
            );
            append_deepstack(&mut deepstack_by_layer, &output, text_dtype)?;
            visual_indices.extend(videos.token_indices);
            visual_outputs.push(output);
        }
        if !visual_outputs.is_empty() && input.position_ids.is_none() {
            return Err(Qwen3VlError::InvalidInput(
                "multimodal forward requires processor-planned MRoPE position ids".into(),
            ));
        }

        let deepstack_features = deepstack_by_layer
            .into_iter()
            .map(|parts| {
                if parts.is_empty() {
                    Tensor::<B, 2>::zeros(
                        [0, self.config.text_config.hidden_size],
                        &embeddings.device(),
                    )
                } else {
                    Tensor::cat(parts, 0)
                }
            })
            .collect::<Vec<_>>();
        let deepstack = (!visual_outputs.is_empty()).then_some(DeepstackEmbeddings {
            token_indices: visual_indices,
            features: deepstack_features,
        });
        let text_output = self.language_model.forward_embeddings(
            embeddings,
            input.attention_mask.as_ref(),
            input.position_ids.as_ref(),
            deepstack.as_ref(),
            input.output_hidden_states,
        )?;
        Ok(Qwen3VlModelOutput {
            last_hidden_state: text_output.last_hidden_state,
            hidden_states: text_output.hidden_states,
            vision_output: combine_vision_outputs(visual_outputs),
            position_deltas: text_output.position_deltas,
        })
    }
}

/// Qwen3-VL base model plus the ordinary vocabulary projection.
#[derive(Module, Debug)]
pub struct Qwen3VlForConditionalGeneration<B: Backend> {
    pub model: Qwen3VlModel<B>,
    pub lm_head: Option<QwenLinear<B>>,
    #[module(skip)]
    tie_word_embeddings: bool,
}

impl<B: Backend> Qwen3VlForConditionalGeneration<B> {
    pub fn new(config: Qwen3VlConfig, device: &B::Device) -> Result<Self> {
        let tie_word_embeddings = config.tie_word_embeddings;
        let lm_head = (!tie_word_embeddings).then(|| {
            QwenLinearConfig::new(
                config.text_config.hidden_size,
                config.text_config.vocab_size,
            )
            .with_bias(false)
            .init(device)
        });
        Ok(Self {
            model: Qwen3VlModel::new(config, device)?,
            lm_head,
            tie_word_embeddings,
        })
    }

    pub fn set_query_chunk_size(&mut self, query_chunk_size: usize) {
        self.model.set_query_chunk_size(query_chunk_size);
    }

    pub fn forward(&self, input: Qwen3VlModelInput<B>) -> Result<Qwen3VlCausalLmOutput<B>> {
        let output = self.model.forward(input)?;
        let logits = if self.tie_word_embeddings {
            let [batch, sequence, hidden] = output.last_hidden_state.dims();
            let vocabulary = self.model.config.text_config.vocab_size;
            output
                .last_hidden_state
                .clone()
                .reshape([batch * sequence, hidden])
                .matmul(
                    self.model
                        .language_model
                        .embed_tokens
                        .weight
                        .val()
                        .transpose(),
                )
                .reshape([batch, sequence, vocabulary])
        } else {
            self.lm_head
                .as_ref()
                .expect("untied model always initializes lm_head")
                .forward(output.last_hidden_state)
        };
        Ok(Qwen3VlCausalLmOutput {
            logits,
            hidden_states: output.hidden_states,
            vision_output: output.vision_output,
            position_deltas: output.position_deltas,
        })
    }
}

pub(crate) fn assign_visual_features<B: Backend>(
    embeddings: Tensor<B, 3>,
    token_indices: &[usize],
    features: Tensor<B, 2>,
) -> Tensor<B, 3> {
    let [batch, sequence, hidden] = embeddings.dims();
    let indices = token_indices
        .iter()
        .map(|&index| index as i64)
        .collect::<Vec<_>>();
    let indices = Tensor::<B, 1, Int>::from_data(
        TensorData::new(indices, [token_indices.len()]),
        &embeddings.device(),
    );
    let flattened = embeddings.reshape([batch * sequence, hidden]);
    // Burn's portable float tensor contract implements additive indexed updates on every
    // backend. Subtracting the selected values first gives exact replacement without relying on
    // a backend-specific assign kernel.
    let delta = features - flattened.clone().select(0, indices.clone());
    flattened
        .select_assign(0, indices, delta, IndexingUpdateOp::Add)
        .reshape([batch, sequence, hidden])
}

pub(crate) fn validate_visual_destinations<B: Backend>(
    output: &Qwen3VlVisionOutput<B>,
    token_indices: &[usize],
    flattened_sequence: usize,
    hidden_size: usize,
) -> Result<()> {
    if token_indices
        .iter()
        .any(|&index| index >= flattened_sequence)
    {
        return Err(Qwen3VlError::InvalidInput(
            "visual token destination is outside flattened text batch".into(),
        ));
    }
    if output.pooler_output.dims() != [token_indices.len(), hidden_size] {
        return Err(Qwen3VlError::InvalidInput(format!(
            "vision tower emitted {:?}, destinations require [{}, {hidden_size}]",
            output.pooler_output.dims(),
            token_indices.len()
        )));
    }
    Ok(())
}

pub(crate) fn append_deepstack<B: Backend>(
    destination: &mut [Vec<Tensor<B, 2>>],
    output: &Qwen3VlVisionOutput<B>,
    text_dtype: DType,
) -> Result<()> {
    if destination.len() != output.deepstack_features.len() {
        return Err(Qwen3VlError::InvalidInput(
            "vision deep-stack output count does not match configuration".into(),
        ));
    }
    for (destination, feature) in destination.iter_mut().zip(&output.deepstack_features) {
        destination.push(feature.clone().cast(text_dtype));
    }
    Ok(())
}

pub(crate) fn combine_vision_outputs<B: Backend>(
    mut outputs: Vec<Qwen3VlVisionOutput<B>>,
) -> Option<Qwen3VlVisionOutput<B>> {
    if outputs.is_empty() {
        return None;
    }
    if outputs.len() == 1 {
        return outputs.pop();
    }
    let deepstack_count = outputs[0].deepstack_features.len();
    Some(Qwen3VlVisionOutput {
        last_hidden_state: Tensor::cat(
            outputs
                .iter()
                .map(|output| output.last_hidden_state.clone())
                .collect(),
            0,
        ),
        pooler_output: Tensor::cat(
            outputs
                .iter()
                .map(|output| output.pooler_output.clone())
                .collect(),
            0,
        ),
        deepstack_features: (0..deepstack_count)
            .map(|index| {
                Tensor::cat(
                    outputs
                        .iter()
                        .map(|output| output.deepstack_features[index].clone())
                        .collect(),
                    0,
                )
            })
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::tiny_config, processor::Grid};
    use burn_ndarray::NdArray;

    #[test]
    fn tiny_multimodal_forward_is_finite_smoke() {
        type B = NdArray<f32>;
        let config = tiny_config();
        let device = Default::default();
        B::seed(&device, 19);
        let model = Qwen3VlForConditionalGeneration::<B>::new(config.clone(), &device).unwrap();
        let input_ids = Tensor::<B, 2, Int>::from_data([[1, 60, 2]], &device);
        let positions = MropePositionIds::from_batch(
            &[vec![0, 1, 0]],
            &[vec![true; 3]],
            &[vec![Grid::new(1, 2, 2)]],
            &[vec![]],
            2,
        )
        .unwrap();
        let patches = Tensor::<B, 2>::from_data(
            TensorData::new(
                vec![0.1; 4 * config.vision_config.patch_volume()],
                [4, config.vision_config.patch_volume()],
            ),
            &device,
        );
        let output = model
            .forward(Qwen3VlModelInput {
                input_ids,
                attention_mask: None,
                position_ids: Some(positions),
                images: Some(Qwen3VlVisualInput {
                    patches,
                    grids: vec![Grid::new(1, 2, 2)],
                    token_indices: vec![1],
                }),
                videos: None,
                output_hidden_states: false,
            })
            .unwrap();
        assert_eq!(output.logits.dims(), [1, 3, 64]);
        assert!(output.vision_output.is_some());
    }
}
