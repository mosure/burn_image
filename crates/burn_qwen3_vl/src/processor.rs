//! Tokenizer-independent multimodal chat processing.

use burn::tensor::{Bool, Int, Tensor, TensorData, backend::Backend};
use serde::{Deserialize, Serialize};

use crate::{
    Qwen3VlConfig, Qwen3VlError, Result,
    chat::{ChatMessage, ChatTemplate, ChatTemplateConfig},
    rope::MropePositionIds,
};

/// Vision patch grid in temporal, height, width order, before spatial merging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Grid {
    pub t: usize,
    pub h: usize,
    pub w: usize,
}

impl Grid {
    pub const fn new(t: usize, h: usize, w: usize) -> Self {
        Self { t, h, w }
    }

    pub fn patch_count(self) -> usize {
        self.t * self.h * self.w
    }

    pub fn merged_token_count(self, spatial_merge_size: usize) -> usize {
        self.patch_count() / (spatial_merge_size * spatial_merge_size)
    }

    pub fn validate(self, spatial_merge_size: usize) -> Result<()> {
        if self.t == 0 || self.h == 0 || self.w == 0 {
            return Err(Qwen3VlError::InvalidInput(
                "vision grid dimensions must be non-zero".into(),
            ));
        }
        if spatial_merge_size == 0
            || !self.h.is_multiple_of(spatial_merge_size)
            || !self.w.is_multiple_of(spatial_merge_size)
        {
            return Err(Qwen3VlError::InvalidInput(format!(
                "grid {self:?} must divide evenly by spatial merge size {spatial_merge_size}"
            )));
        }
        Ok(())
    }
}

/// Minimal tokenizer contract. Implementations must preserve registered special tokens as one id.
pub trait Qwen3VlTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i64>>;
    fn decode(&self, ids: &[i64], skip_special_tokens: bool) -> Result<String>;
    fn token_to_id(&self, token: &str) -> Option<i64>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PaddingSide {
    Left,
    Right,
}

/// Processor configuration, including the checkpoint's special-token ids.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Qwen3VlProcessorConfig {
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub image_token_id: i64,
    pub video_token_id: i64,
    pub pad_token_id: i64,
    pub padding_side: PaddingSide,
    pub chat_template: ChatTemplateConfig,
}

impl Qwen3VlProcessorConfig {
    pub fn from_model(config: &Qwen3VlConfig, pad_token_id: i64) -> Self {
        Self {
            spatial_merge_size: config.vision_config.spatial_merge_size,
            temporal_patch_size: config.vision_config.temporal_patch_size,
            image_token_id: config.image_token_id as i64,
            video_token_id: config.video_token_id as i64,
            pad_token_id,
            padding_side: PaddingSide::Left,
            chat_template: ChatTemplateConfig::default(),
        }
    }
}

/// Sampled-video timing metadata used to construct Qwen3-VL's timestamp-separated frame prompt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoMetadata {
    /// Indices of sampled source frames before temporal patch merging.
    pub frame_indices: Vec<usize>,
    /// Source frames per second. Qwen's reference processor falls back to 24 when absent.
    pub fps: Option<f64>,
}

impl VideoMetadata {
    pub fn new(frame_indices: Vec<usize>, fps: Option<f64>) -> Self {
        Self { frame_indices, fps }
    }
}

/// One chat plus the grids referenced, in encounter order, by its image/video content.
#[derive(Debug, Clone)]
pub struct ProcessorSample<'a> {
    pub messages: &'a [ChatMessage],
    pub image_grids: &'a [Grid],
    pub video_grids: &'a [Grid],
    /// One entry for every video grid. When omitted, sequential sampled frames and 24 fps are used.
    pub video_metadata: &'a [VideoMetadata],
}

/// Padded CPU encoding and the original per-sample grids needed by MRoPE and the vision tower.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchEncoding {
    pub input_ids: Vec<Vec<i64>>,
    pub attention_mask: Vec<Vec<bool>>,
    pub mm_token_type_ids: Vec<Vec<u8>>,
    pub visual_token_indices: Vec<Vec<usize>>,
    pub image_grids: Vec<Vec<Grid>>,
    pub video_grids: Vec<Vec<Grid>>,
}

impl BatchEncoding {
    pub fn batch_size(&self) -> usize {
        self.input_ids.len()
    }

    pub fn sequence_length(&self) -> usize {
        self.input_ids.first().map_or(0, Vec::len)
    }

    pub fn position_ids(&self, spatial_merge_size: usize) -> Result<MropePositionIds> {
        MropePositionIds::from_batch(
            &self.mm_token_type_ids,
            &self.attention_mask,
            &self.image_grids,
            &self.video_grids,
            spatial_merge_size,
        )
    }

    /// Flatten image placeholder locations across the padded `[batch, sequence]` tensor.
    pub fn flattened_image_token_indices(&self) -> Vec<usize> {
        self.flattened_modality_indices(1)
    }

    /// Flatten video placeholder locations across the padded `[batch, sequence]` tensor.
    pub fn flattened_video_token_indices(&self) -> Vec<usize> {
        self.flattened_modality_indices(2)
    }

    fn flattened_modality_indices(&self, modality: u8) -> Vec<usize> {
        let sequence = self.sequence_length();
        self.mm_token_type_ids
            .iter()
            .enumerate()
            .flat_map(|(batch, types)| {
                types.iter().enumerate().filter_map(move |(token, &kind)| {
                    (kind == modality).then_some(batch * sequence + token)
                })
            })
            .collect()
    }

    pub fn to_tensors<B: Backend>(&self, device: &B::Device) -> Result<BatchTensors<B>> {
        let batch = self.batch_size();
        let sequence = self.sequence_length();
        if batch == 0
            || sequence == 0
            || self.input_ids.iter().any(|row| row.len() != sequence)
            || self.attention_mask.iter().any(|row| row.len() != sequence)
        {
            return Err(Qwen3VlError::InvalidInput(
                "batch encoding is empty or ragged".into(),
            ));
        }
        let ids = self.input_ids.iter().flatten().copied().collect::<Vec<_>>();
        let mask = self
            .attention_mask
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        Ok(BatchTensors {
            input_ids: Tensor::<B, 2, Int>::from_data(
                TensorData::new(ids, [batch, sequence]),
                device,
            ),
            attention_mask: Tensor::<B, 2, Bool>::from_data(
                TensorData::new(mask, [batch, sequence]),
                device,
            ),
        })
    }
}

pub struct BatchTensors<B: Backend> {
    pub input_ids: Tensor<B, 2, Int>,
    pub attention_mask: Tensor<B, 2, Bool>,
}

/// Generic Qwen3-VL chat/token processor.
#[derive(Debug, Clone)]
pub struct Qwen3VlProcessor<T> {
    tokenizer: T,
    template: ChatTemplate,
    config: Qwen3VlProcessorConfig,
}

impl<T: Qwen3VlTokenizer> Qwen3VlProcessor<T> {
    pub fn new(tokenizer: T, config: Qwen3VlProcessorConfig) -> Result<Self> {
        if config.spatial_merge_size == 0 {
            return Err(Qwen3VlError::InvalidConfig(
                "processor spatial_merge_size must be non-zero".into(),
            ));
        }
        if config.temporal_patch_size == 0 {
            return Err(Qwen3VlError::InvalidConfig(
                "processor temporal_patch_size must be non-zero".into(),
            ));
        }
        for (token, expected) in [
            (&config.chat_template.image_pad, config.image_token_id),
            (&config.chat_template.video_pad, config.video_token_id),
        ] {
            if tokenizer.token_to_id(token) != Some(expected) {
                return Err(Qwen3VlError::InvalidConfig(format!(
                    "tokenizer id for {token:?} does not match processor config"
                )));
            }
        }
        let template = ChatTemplate::new(config.chat_template.clone());
        Ok(Self {
            tokenizer,
            template,
            config,
        })
    }

    pub fn tokenizer(&self) -> &T {
        &self.tokenizer
    }

    pub fn config(&self) -> &Qwen3VlProcessorConfig {
        &self.config
    }

    /// Expand one placeholder per media item to the exact number of post-merge language tokens.
    pub fn expand_placeholders(
        &self,
        rendered_chat: &str,
        image_grids: &[Grid],
        video_grids: &[Grid],
    ) -> Result<String> {
        self.expand_placeholders_with_video_metadata(rendered_chat, image_grids, video_grids, &[])
    }

    /// Expand media placeholders, including Qwen3-VL timestamp text between video frames.
    pub fn expand_placeholders_with_video_metadata(
        &self,
        rendered_chat: &str,
        image_grids: &[Grid],
        video_grids: &[Grid],
        video_metadata: &[VideoMetadata],
    ) -> Result<String> {
        if !video_metadata.is_empty() && video_metadata.len() != video_grids.len() {
            return Err(Qwen3VlError::InvalidInput(
                "video metadata count must equal video grid count".into(),
            ));
        }
        let mut expanded = rendered_chat.to_owned();
        let mut image_replacements = Vec::with_capacity(image_grids.len());
        for grid in image_grids {
            grid.validate(self.config.spatial_merge_size)?;
            let count = grid.merged_token_count(self.config.spatial_merge_size);
            image_replacements.push(self.config.chat_template.image_pad.repeat(count));
        }
        expanded = replace_ordered(
            &expanded,
            &self.config.chat_template.image_pad,
            &image_replacements,
        )?;
        let video_placeholder = format!(
            "{}{}{}",
            self.config.chat_template.vision_start,
            self.config.chat_template.video_pad,
            self.config.chat_template.vision_end
        );
        let mut video_replacements = Vec::with_capacity(video_grids.len());
        for (video_index, grid) in video_grids.iter().enumerate() {
            grid.validate(self.config.spatial_merge_size)?;
            let frame_tokens =
                grid.h * grid.w / (self.config.spatial_merge_size * self.config.spatial_merge_size);
            let default_metadata;
            let metadata = if let Some(metadata) = video_metadata.get(video_index) {
                metadata
            } else {
                default_metadata = VideoMetadata {
                    frame_indices: (0..grid.t * self.config.temporal_patch_size).collect(),
                    fps: None,
                };
                &default_metadata
            };
            let timestamps = video_timestamps(metadata, self.config.temporal_patch_size, grid.t)?;
            let mut replacement = String::new();
            for timestamp in timestamps {
                replacement.push_str(&format!("<{timestamp:.1} seconds>"));
                replacement.push_str(&self.config.chat_template.vision_start);
                replacement.push_str(&self.config.chat_template.video_pad.repeat(frame_tokens));
                replacement.push_str(&self.config.chat_template.vision_end);
            }
            video_replacements.push(replacement);
        }
        expanded = replace_ordered(&expanded, &video_placeholder, &video_replacements)?;
        if expanded.contains(&self.config.chat_template.image_pad)
            && count_occurrences(&expanded, &self.config.chat_template.image_pad)
                != image_grids
                    .iter()
                    .map(|grid| grid.merged_token_count(self.config.spatial_merge_size))
                    .sum::<usize>()
        {
            return Err(Qwen3VlError::InvalidInput(
                "chat/image-grid placeholder count mismatch".into(),
            ));
        }
        if expanded.contains(&self.config.chat_template.video_pad)
            && count_occurrences(&expanded, &self.config.chat_template.video_pad)
                != video_grids
                    .iter()
                    .map(|grid| grid.merged_token_count(self.config.spatial_merge_size))
                    .sum::<usize>()
        {
            return Err(Qwen3VlError::InvalidInput(
                "chat/video-grid placeholder count mismatch".into(),
            ));
        }
        Ok(expanded)
    }

    pub fn encode_batch(
        &self,
        samples: &[ProcessorSample<'_>],
        add_generation_prompt: bool,
    ) -> Result<BatchEncoding> {
        if samples.is_empty() {
            return Err(Qwen3VlError::InvalidInput(
                "processor batch must not be empty".into(),
            ));
        }
        let mut unpadded = Vec::with_capacity(samples.len());
        for sample in samples {
            let rendered = self
                .template
                .render(sample.messages, add_generation_prompt)?;
            let expanded = self.expand_placeholders_with_video_metadata(
                &rendered,
                sample.image_grids,
                sample.video_grids,
                sample.video_metadata,
            )?;
            unpadded.push(self.tokenizer.encode(&expanded, false)?);
        }
        let sequence_length = unpadded.iter().map(Vec::len).max().unwrap_or(0);
        if sequence_length == 0 {
            return Err(Qwen3VlError::Tokenizer(
                "tokenizer produced an empty batch".into(),
            ));
        }
        let mut input_ids = Vec::with_capacity(samples.len());
        let mut attention_mask = Vec::with_capacity(samples.len());
        let mut mm_token_type_ids = Vec::with_capacity(samples.len());
        let mut visual_token_indices = Vec::with_capacity(samples.len());
        for ids in unpadded {
            let padding = sequence_length - ids.len();
            let (row, mask) = match self.config.padding_side {
                PaddingSide::Left => {
                    let mut row = vec![self.config.pad_token_id; padding];
                    row.extend_from_slice(&ids);
                    let mut mask = vec![false; padding];
                    mask.extend(core::iter::repeat_n(true, ids.len()));
                    (row, mask)
                }
                PaddingSide::Right => {
                    let mut row = ids;
                    row.resize(sequence_length, self.config.pad_token_id);
                    let mut mask = vec![true; sequence_length - padding];
                    mask.resize(sequence_length, false);
                    (row, mask)
                }
            };
            let types = row
                .iter()
                .map(|&id| {
                    if id == self.config.image_token_id {
                        1
                    } else if id == self.config.video_token_id {
                        2
                    } else {
                        0
                    }
                })
                .collect::<Vec<_>>();
            visual_token_indices.push(
                types
                    .iter()
                    .enumerate()
                    .filter_map(|(index, &kind)| (kind != 0).then_some(index))
                    .collect(),
            );
            input_ids.push(row);
            attention_mask.push(mask);
            mm_token_type_ids.push(types);
        }
        Ok(BatchEncoding {
            input_ids,
            attention_mask,
            mm_token_type_ids,
            visual_token_indices,
            image_grids: samples
                .iter()
                .map(|sample| sample.image_grids.to_vec())
                .collect(),
            video_grids: samples
                .iter()
                .map(|sample| sample.video_grids.to_vec())
                .collect(),
        })
    }
}

fn replace_ordered(input: &str, needle: &str, replacements: &[String]) -> Result<String> {
    let occurrences = input.match_indices(needle).count();
    if occurrences != replacements.len() {
        return Err(Qwen3VlError::InvalidInput(format!(
            "found {occurrences} {needle:?} placeholders but received {} media entries",
            replacements.len()
        )));
    }
    let mut output = String::with_capacity(input.len());
    let mut cursor = 0;
    for replacement in replacements {
        let relative = input[cursor..]
            .find(needle)
            .expect("placeholder count was checked");
        let position = cursor + relative;
        output.push_str(&input[cursor..position]);
        output.push_str(replacement);
        cursor = position + needle.len();
    }
    output.push_str(&input[cursor..]);
    Ok(output)
}

fn count_occurrences(input: &str, needle: &str) -> usize {
    input.match_indices(needle).count()
}

fn video_timestamps(
    metadata: &VideoMetadata,
    temporal_patch_size: usize,
    expected_frames: usize,
) -> Result<Vec<f64>> {
    if metadata.frame_indices.is_empty() {
        return Err(Qwen3VlError::InvalidInput(
            "video frame_indices must not be empty".into(),
        ));
    }
    let fps = metadata.fps.unwrap_or(24.0);
    if !fps.is_finite() || fps <= 0.0 {
        return Err(Qwen3VlError::InvalidInput(
            "video fps must be finite and positive".into(),
        ));
    }
    let mut indices = metadata.frame_indices.clone();
    let remainder = indices.len() % temporal_patch_size;
    if remainder != 0 {
        let last = *indices.last().expect("checked non-empty frame indices");
        indices.extend(core::iter::repeat_n(last, temporal_patch_size - remainder));
    }
    let timestamps = indices
        .chunks_exact(temporal_patch_size)
        .map(|chunk| (chunk[0] as f64 + chunk[temporal_patch_size - 1] as f64) / (2.0 * fps))
        .collect::<Vec<_>>();
    if timestamps.len() != expected_frames {
        return Err(Qwen3VlError::InvalidInput(format!(
            "video metadata produces {} temporal patches, grid requires {expected_frames}",
            timestamps.len()
        )));
    }
    Ok(timestamps)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::chat::{ChatContent, ChatRole};

    #[derive(Debug, Clone)]
    struct SplitTokenizer {
        specials: HashMap<String, i64>,
    }

    impl Qwen3VlTokenizer for SplitTokenizer {
        fn encode(&self, text: &str, _add_special_tokens: bool) -> Result<Vec<i64>> {
            let mut ids = Vec::new();
            let mut offset = 0;
            while offset < text.len() {
                if let Some((token, id)) = self
                    .specials
                    .iter()
                    .find(|(token, _)| text[offset..].starts_with(token.as_str()))
                {
                    ids.push(*id);
                    offset += token.len();
                } else {
                    ids.push(text.as_bytes()[offset] as i64);
                    offset += 1;
                }
            }
            Ok(ids)
        }

        fn decode(&self, _ids: &[i64], _skip_special_tokens: bool) -> Result<String> {
            unimplemented!()
        }

        fn token_to_id(&self, token: &str) -> Option<i64> {
            self.specials.get(token).copied()
        }
    }

    fn processor() -> Qwen3VlProcessor<SplitTokenizer> {
        let config = Qwen3VlProcessorConfig {
            spatial_merge_size: 2,
            temporal_patch_size: 2,
            image_token_id: 1000,
            video_token_id: 1001,
            pad_token_id: 0,
            padding_side: PaddingSide::Left,
            chat_template: ChatTemplateConfig::default(),
        };
        let tokenizer = SplitTokenizer {
            specials: HashMap::from([
                (config.chat_template.image_pad.clone(), 1000),
                (config.chat_template.video_pad.clone(), 1001),
                (config.chat_template.im_start.clone(), 1002),
                (config.chat_template.im_end.clone(), 1003),
                (config.chat_template.vision_start.clone(), 1004),
                (config.chat_template.vision_end.clone(), 1005),
            ]),
        };
        Qwen3VlProcessor::new(tokenizer, config).unwrap()
    }

    #[test]
    fn placeholder_expansion_and_types_correctness() {
        let processor = processor();
        let messages = [ChatMessage::new(
            ChatRole::User,
            vec![ChatContent::Image, ChatContent::text("ok")],
        )];
        let grids = [Grid::new(1, 4, 4)];
        let encoding = processor
            .encode_batch(
                &[ProcessorSample {
                    messages: &messages,
                    image_grids: &grids,
                    video_grids: &[],
                    video_metadata: &[],
                }],
                true,
            )
            .unwrap();
        assert_eq!(
            encoding.mm_token_type_ids[0]
                .iter()
                .filter(|&&kind| kind == 1)
                .count(),
            4
        );
        assert_eq!(encoding.visual_token_indices[0].len(), 4);
        let positions = encoding.position_ids(2).unwrap();
        assert_eq!(positions.batch_size(), 1);
    }

    #[test]
    fn video_frames_are_timestamp_separated_correctness() {
        let processor = processor();
        let messages = [ChatMessage::new(
            ChatRole::User,
            vec![ChatContent::Video, ChatContent::text("Describe.")],
        )];
        let grids = [Grid::new(2, 2, 2)];
        let metadata = [VideoMetadata::new(vec![0, 1, 24, 25], Some(24.0))];
        let encoding = processor
            .encode_batch(
                &[ProcessorSample {
                    messages: &messages,
                    image_grids: &[],
                    video_grids: &grids,
                    video_metadata: &metadata,
                }],
                false,
            )
            .unwrap();
        let video_groups = encoding.mm_token_type_ids[0]
            .split(|kind| *kind != 2)
            .filter(|group| !group.is_empty())
            .count();
        assert_eq!(video_groups, 2);
        assert_eq!(
            encoding.mm_token_type_ids[0]
                .iter()
                .filter(|&&kind| kind == 2)
                .count(),
            2
        );
        encoding.position_ids(2).unwrap();
    }

    #[test]
    fn multiple_media_expansions_preserve_encounter_order_correctness() {
        let processor = processor();
        let messages = [ChatMessage::new(
            ChatRole::User,
            vec![ChatContent::Image, ChatContent::Image],
        )];
        let grids = [Grid::new(1, 4, 4), Grid::new(1, 6, 6)];
        let encoding = processor
            .encode_batch(
                &[ProcessorSample {
                    messages: &messages,
                    image_grids: &grids,
                    video_grids: &[],
                    video_metadata: &[],
                }],
                false,
            )
            .unwrap();
        let group_lengths = encoding.mm_token_type_ids[0]
            .split(|kind| *kind != 1)
            .filter(|group| !group.is_empty())
            .map(<[u8]>::len)
            .collect::<Vec<_>>();
        assert_eq!(group_lengths, [4, 9]);
        encoding.position_ids(2).unwrap();
    }
}
