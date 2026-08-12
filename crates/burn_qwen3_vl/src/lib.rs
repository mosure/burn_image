//! Reusable Qwen3-VL architecture, multimodal position planning, and processing.
//!
//! This crate intentionally contains no application-specific prompt, sampling, CDN, or
//! generation policy. It models the ordinary Qwen3-VL language and vision towers and exposes
//! deterministic preprocessing contracts suitable for native and WebGPU Burn backends.

pub mod builder;
pub mod chat;
#[cfg(feature = "import")]
pub mod checkpoint;
pub mod config;
pub mod error;
pub mod image_processor;
pub mod linear;
pub mod model;
pub mod outputs;
pub mod processor;
pub mod rope;
pub mod streaming;
pub mod text;
#[cfg(feature = "tokenizers")]
pub mod tokenizer;
pub mod vision;
pub mod weights;

pub use builder::Qwen3VlBuilder;
pub use chat::{ChatContent, ChatMessage, ChatRole, ChatTemplate, ChatTemplateConfig, ToolCall};
#[cfg(feature = "import")]
pub use checkpoint::{
    CheckpointDType, CheckpointInspection, CheckpointLoadReport, HfCheckpoint,
    load_base_from_safetensors, load_base_from_safetensors_with_dtype,
    load_causal_lm_from_safetensors, load_causal_lm_from_safetensors_with_dtype,
};
pub use config::{Qwen3VlConfig, Qwen3VlTextConfig, Qwen3VlVisionConfig};
pub use error::{Qwen3VlError, Result};
pub use image_processor::{
    ProcessedVisionPixels, Qwen3VlImageProcessor, Qwen3VlImageProcessorConfig,
};
pub use linear::{QwenLinear, QwenLinearConfig};
pub use model::{
    Qwen3VlForConditionalGeneration, Qwen3VlModel, Qwen3VlModelInput, Qwen3VlVisualInput,
};
pub use outputs::{
    Qwen3VlCausalLmOutput, Qwen3VlModelOutput, Qwen3VlTextOutput, Qwen3VlVisionOutput,
};
pub use processor::{
    BatchEncoding, BatchTensors, Grid, PaddingSide, ProcessorSample, Qwen3VlProcessor,
    Qwen3VlProcessorConfig, Qwen3VlTokenizer, VideoMetadata,
};
pub use rope::{MropePositionIds, PositionDelta};
pub use streaming::{
    AsyncQwen3VlCausalLmStageSource, AsyncQwen3VlStageSource, ChunkedEmbeddingState,
    DEFAULT_VOCABULARY_CHUNKS, EmbeddingRowChunk, OutputProjectionRowChunk,
    Qwen3VlCausalLmStageSource, Qwen3VlStage, Qwen3VlStageDType, Qwen3VlStageDTypePolicy,
    Qwen3VlStageDescriptor, Qwen3VlStageObserver, Qwen3VlStageSource, Qwen3VlStreamingPlan,
    Qwen3VlTextState, Qwen3VlVisionPrelude, Qwen3VlVisionState, RetainingQwen3VlStageSource,
    RetainingSynchronizationPolicy, RowChunkPlan, RowChunkSpec, RowSliceWeightSpec,
    StreamingForwardError, StreamingQwen3Vl,
};
pub use text::{DeepstackEmbeddings, Qwen3VlDecoderLayer};
pub use vision::{Qwen3VlVisionBlock, Qwen3VlVisionPatchMerger};
pub use weights::{WeightInventory, WeightRole, WeightShapeMismatch, WeightSpec, WeightValidation};
