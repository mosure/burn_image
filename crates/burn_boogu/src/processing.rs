//! Boogu-only host processing around the reusable Qwen3-VL and FLUX VAE crates.

use std::collections::BTreeSet;

use burn::{
    prelude::Backend,
    tensor::{DType, Tensor, TensorData},
};
use burn_image::{
    ColorSpace, DimensionConstraints, Dimensions, HostImage, ImageRequest, ImageTaskKind,
    InputImage, ModelCapabilities, ModelDescriptor, ModelId, NumericFormat, PixelBuffer,
    PixelFormat,
};
use burn_qwen3_vl::{
    BatchEncoding, ChatContent, ChatMessage, ChatRole, PaddingSide, ProcessorSample, Qwen3VlConfig,
    Qwen3VlImageProcessor, Qwen3VlModelInput, Qwen3VlProcessor, Qwen3VlProcessorConfig,
    Qwen3VlTokenizer, Qwen3VlVisualInput,
};
use image::{DynamicImage, ImageReader, Rgb, RgbImage, imageops::FilterType};

use crate::{
    BooguError, BooguTask, BooguVariant,
    artifacts::{EDIT_TURBO_1K5_REVISION, EDIT_TURBO_REVISION, TURBO_REVISION},
    conditioning::InstructionPolicy,
    latent::validate_image_size,
};

/// Default output edge used by the released pipelines.
pub const BOOGU_DEFAULT_EDGE: u32 = 1024;
/// Largest output edge with released-shape runtime acceptance evidence.
pub const BOOGU_MAX_OUTPUT_SIDE: u32 = 1024;
/// Largest output pixel count with released-shape runtime acceptance evidence.
pub const BOOGU_MAX_OUTPUT_PIXELS: u64 = 1_048_576;
/// Default output edge encoded by the Edit-Turbo 1.5K release configuration.
pub const BOOGU_1K5_DEFAULT_EDGE: u32 = 1536;
/// Largest output edge among the Edit-Turbo 1.5K aspect-ratio presets.
pub const BOOGU_1K5_MAX_OUTPUT_SIDE: u32 = 2368;
/// Largest output pixel count supported by the Edit-Turbo 1.5K release.
pub const BOOGU_1K5_MAX_OUTPUT_PIXELS: u64 = 2_360_832;
/// Official Edit-Turbo 1.5K output presets exposed by the upstream release.
pub const BOOGU_1K5_OUTPUT_PRESETS: [(u32, u32); 10] = [
    (1536, 1536),
    (1264, 1856),
    (1856, 1264),
    (1344, 1744),
    (1744, 1344),
    (1392, 1696),
    (1696, 1392),
    (1152, 2032),
    (2032, 1152),
    (2368, 992),
];
/// Upstream edit-reference VAE preprocessing limit.
pub const BOOGU_MAX_REFERENCE_PIXELS: u64 = 4_194_304;
/// Upstream edit-reference VAE side limit.
pub const BOOGU_MAX_REFERENCE_SIDE: u32 = 4096;
/// Upstream Qwen image preprocessing limit used by the released pipeline.
pub const BOOGU_MAX_VLM_PIXELS: u64 = 147_456;
/// Upstream Qwen image preprocessing side limit.
pub const BOOGU_MAX_VLM_SIDE: u32 = 768;

/// Fully validated model-neutral request after applying documented Boogu defaults.
#[derive(Debug, Clone)]
pub struct ResolvedBooguRequest {
    /// Checkpoint selected by the runtime.
    pub variant: BooguVariant,
    /// Generate or edit task implied by the checkpoint.
    pub task: BooguTask,
    /// User prompt or edit instruction.
    pub prompt: String,
    /// Edit source, present only for Edit-Turbo.
    pub source: Option<InputImage>,
    /// Requested output dimensions.
    pub dimensions: Dimensions,
    /// Exact four-step Turbo count.
    pub steps: u32,
    /// Deterministic host seed chosen by the caller/runtime.
    pub seed: u64,
}

/// Descriptor for one immutable Boogu release.
pub fn boogu_model_descriptor(variant: BooguVariant) -> ModelDescriptor {
    let (id, display_name, revision, task, max_output_side, max_output_pixels) = match variant {
        BooguVariant::Image01Turbo => (
            "Boogu/Boogu-Image-0.1-Turbo",
            "Boogu-Image 0.1 Turbo",
            TURBO_REVISION,
            ImageTaskKind::Generate,
            BOOGU_MAX_OUTPUT_SIDE,
            BOOGU_MAX_OUTPUT_PIXELS,
        ),
        BooguVariant::Image01EditTurbo => (
            "Boogu/Boogu-Image-0.1-Edit-Turbo",
            "Boogu-Image 0.1 Edit Turbo",
            EDIT_TURBO_REVISION,
            ImageTaskKind::Edit,
            BOOGU_MAX_OUTPUT_SIDE,
            BOOGU_MAX_OUTPUT_PIXELS,
        ),
        BooguVariant::Image01EditTurbo1k5 => (
            "Boogu/Boogu-Image-0.1-Edit-Turbo-1K5",
            "Boogu-Image 0.1 Edit Turbo 1.5K",
            EDIT_TURBO_1K5_REVISION,
            ImageTaskKind::Edit,
            BOOGU_1K5_MAX_OUTPUT_SIDE,
            BOOGU_1K5_MAX_OUTPUT_PIXELS,
        ),
    };
    let numeric_formats = if variant == BooguVariant::Image01EditTurbo1k5 {
        BTreeSet::from([
            NumericFormat::Other("f16-qwen-vision-f32".into()),
            NumericFormat::Other("q4s-block-up-to128-f32".into()),
        ])
    } else {
        BTreeSet::from([
            NumericFormat::F16,
            NumericFormat::Other("f16-qwen-vision-f32".into()),
            NumericFormat::Other("q8s-block32-f32".into()),
            NumericFormat::Other("q8s-block32-f32-qwen-vision-f32".into()),
            NumericFormat::Other("q4s-block-up-to128-f32".into()),
        ])
    };
    let allowed_dimensions = (variant == BooguVariant::Image01EditTurbo1k5).then(|| {
        BOOGU_1K5_OUTPUT_PRESETS
            .into_iter()
            .map(|(width, height)| {
                Dimensions::new(width, height).expect("released 1.5K preset is valid")
            })
            .collect()
    });
    ModelDescriptor {
        id: ModelId::new(id).expect("canonical Boogu model id is valid"),
        display_name: display_name.into(),
        revision: revision.into(),
        capabilities: ModelCapabilities {
            tasks: BTreeSet::from([task]),
            supports_masks: false,
            dimensions: DimensionConstraints {
                min_width: 256,
                max_width: max_output_side,
                min_height: 256,
                max_height: max_output_side,
                width_multiple: 16,
                height_multiple: 16,
                max_pixels: Some(max_output_pixels),
                allowed_dimensions,
            },
            // The released Turbo weights encode the four-step DMD trajectory. This is not an
            // ordinary scheduler where an arbitrary step count remains meaningful.
            min_steps: 4,
            max_steps: 4,
            max_batch_size: 1,
            numeric_formats,
        },
    }
}

/// Validate a portable request against the selected immutable release.
///
/// `default_seed` is supplied by the host so browser entropy policy stays outside model code.
pub fn resolve_request(
    variant: BooguVariant,
    request: &ImageRequest,
    default_seed: u64,
) -> Result<ResolvedBooguRequest, BooguError> {
    let descriptor = boogu_model_descriptor(variant);
    descriptor
        .capabilities
        .validate_request(&descriptor.id, request)
        .map_err(|error| BooguError::InvalidRequest(error.to_string()))?;
    let options = request.options();
    if let Some(guidance) = options.guidance_scale
        && guidance != 1.0
    {
        return Err(BooguError::InvalidRequest(format!(
            "Turbo requires guidance_scale=1, got {guidance}"
        )));
    }
    let (task, prompt, source) = match (variant, request) {
        (BooguVariant::Image01Turbo, ImageRequest::Generate(value)) => {
            if value.negative_prompt.is_some() {
                return Err(BooguError::InvalidRequest(
                    "Turbo does not support negative prompts or CFG".into(),
                ));
            }
            (BooguTask::Generate, value.prompt.as_str().to_owned(), None)
        }
        (
            BooguVariant::Image01EditTurbo | BooguVariant::Image01EditTurbo1k5,
            ImageRequest::Edit(value),
        ) => {
            if value.negative_prompt.is_some() {
                return Err(BooguError::InvalidRequest(
                    "Edit-Turbo does not support negative prompts or CFG".into(),
                ));
            }
            if value.mask.is_some() {
                return Err(BooguError::InvalidRequest(
                    "Edit-Turbo does not support masks".into(),
                ));
            }
            if value.strength.is_some() {
                return Err(BooguError::InvalidRequest(
                    "Edit-Turbo uses its fixed DMD path and does not expose edit strength".into(),
                ));
            }
            (
                BooguTask::Edit,
                value.instruction.as_str().to_owned(),
                Some(value.source.clone()),
            )
        }
        (BooguVariant::Image01Turbo, ImageRequest::Edit(_)) => {
            return Err(BooguError::InvalidRequest(
                "Boogu-Image-0.1-Turbo only supports generation".into(),
            ));
        }
        (
            BooguVariant::Image01EditTurbo | BooguVariant::Image01EditTurbo1k5,
            ImageRequest::Generate(_),
        ) => {
            return Err(BooguError::InvalidRequest(
                "Boogu-Image-0.1-Edit-Turbo only supports editing".into(),
            ));
        }
    };
    let dimensions = match options.dimensions {
        Some(dimensions) => dimensions,
        None if variant == BooguVariant::Image01EditTurbo1k5 => {
            Dimensions::new(BOOGU_1K5_DEFAULT_EDGE, BOOGU_1K5_DEFAULT_EDGE)
                .expect("released 1.5K default dimensions are valid")
        }
        None if task == BooguTask::Edit => {
            let source_dimensions = source
                .as_ref()
                .and_then(InputImage::dimensions)
                .ok_or_else(|| {
                    BooguError::InvalidRequest(
                        "Edit-Turbo requires source dimensions when output dimensions are omitted"
                            .into(),
                    )
                })?;
            let (height, width) = limited_dimensions(
                source_dimensions.height(),
                source_dimensions.width(),
                BOOGU_MAX_REFERENCE_PIXELS,
                BOOGU_MAX_REFERENCE_SIDE,
                16,
            )?;
            Dimensions::new(width, height)
                .map_err(|error| BooguError::InvalidRequest(error.to_string()))?
        }
        None => Dimensions::new(BOOGU_DEFAULT_EDGE, BOOGU_DEFAULT_EDGE)
            .expect("released default dimensions are valid"),
    };
    descriptor
        .capabilities
        .dimensions
        .supports(dimensions)
        .map_err(BooguError::InvalidRequest)?;
    validate_image_size(dimensions.height() as usize, dimensions.width() as usize)?;
    InstructionPolicy::upstream(task, usize::from(source.is_some()))?;
    Ok(ResolvedBooguRequest {
        variant,
        task,
        prompt,
        source,
        dimensions,
        steps: options.steps.unwrap_or(4),
        seed: options.seed.unwrap_or(default_seed),
    })
}

/// Decode a model-neutral input into the RGB representation consumed by both edit paths.
pub fn decode_input_image(input: &InputImage) -> Result<DynamicImage, BooguError> {
    match input {
        InputImage::Encoded(encoded) => ImageReader::new(std::io::Cursor::new(encoded.bytes()))
            .with_guessed_format()
            .map_err(|error| BooguError::InvalidRequest(format!("cannot identify image: {error}")))?
            .decode()
            .map_err(|error| BooguError::InvalidRequest(format!("cannot decode image: {error}"))),
        InputImage::Pixels(pixels) => decode_pixels(pixels),
    }
}

fn decode_pixels(pixels: &PixelBuffer) -> Result<DynamicImage, BooguError> {
    let dimensions = pixels.dimensions();
    let (width, height) = (dimensions.width(), dimensions.height());
    let bytes = pixels.bytes();
    let image = match pixels.format() {
        PixelFormat::Rgb8 => RgbImage::from_raw(width, height, bytes.to_vec()),
        PixelFormat::Rgba8 => {
            let rgba =
                image::RgbaImage::from_raw(width, height, bytes.to_vec()).ok_or_else(|| {
                    BooguError::InvalidShape(
                        "validated RGBA pixel buffer could not be decoded".into(),
                    )
                })?;
            return Ok(DynamicImage::ImageRgba8(rgba));
        }
        PixelFormat::L8 => {
            let luma =
                image::GrayImage::from_raw(width, height, bytes.to_vec()).ok_or_else(|| {
                    BooguError::InvalidShape(
                        "validated luma pixel buffer could not be decoded".into(),
                    )
                })?;
            return Ok(DynamicImage::ImageLuma8(luma));
        }
        PixelFormat::Rgba16Float | PixelFormat::Rgba32Float => {
            return Err(BooguError::InvalidRequest(
                "host float pixel inputs are not accepted; provide encoded or 8-bit pixels".into(),
            ));
        }
    }
    .ok_or_else(|| {
        BooguError::InvalidShape("validated RGB pixel buffer could not be decoded".into())
    })?;
    Ok(DynamicImage::ImageRgb8(image))
}

/// Resize one edit source using the upstream floor-to-16, no-upscale policy.
pub fn resize_reference(
    source: &DynamicImage,
    max_pixels: u64,
    max_side: u32,
) -> Result<RgbImage, BooguError> {
    let rgb = pillow_compatible_rgb8(source);
    let (height, width) = limited_dimensions(rgb.height(), rgb.width(), max_pixels, max_side, 16)?;
    if width == rgb.width() && height == rgb.height() {
        Ok(rgb)
    } else {
        Ok(image::imageops::resize(
            &rgb,
            width,
            height,
            FilterType::Lanczos3,
        ))
    }
}

/// Create normalized `[1,3,H,W]` FLUX VAE input for Edit-Turbo.
pub fn prepare_vae_reference<B: Backend>(
    source: &DynamicImage,
    device: &B::Device,
) -> Result<Tensor<B, 4>, BooguError> {
    let image = resize_reference(source, BOOGU_MAX_REFERENCE_PIXELS, BOOGU_MAX_REFERENCE_SIDE)?;
    let (height, width) = (image.height() as usize, image.width() as usize);
    let mut values = vec![0.0_f32; 3 * height * width];
    for (x, y, pixel) in image.enumerate_pixels() {
        let offset = y as usize * width + x as usize;
        for channel in 0..3 {
            values[channel * height * width + offset] = pixel[channel] as f32 / 127.5 - 1.0;
        }
    }
    Ok(Tensor::from_data(
        TensorData::new(values, [1, 3, height, width]),
        device,
    ))
}

/// Construct the ordinary Qwen processor configuration required by Boogu.
pub fn boogu_processor_config(config: &Qwen3VlConfig, pad_token_id: i64) -> Qwen3VlProcessorConfig {
    let mut processor = Qwen3VlProcessorConfig::from_model(config, pad_token_id);
    processor.padding_side = PaddingSide::Right;
    processor
}

/// Qwen tensors plus the inspectable CPU encoding used to produce them.
pub struct PreparedInstruction<B: Backend> {
    /// Complete model input, including vision patches for Edit-Turbo.
    pub model_input: Qwen3VlModelInput<B>,
    /// Number of valid right-padded tokens retained for Boogu conditioning.
    pub effective_length: usize,
    /// CPU token/mask/grid plan for provenance and exact parity checks.
    pub encoding: BatchEncoding,
    /// VLM-resized image, when editing.
    pub vision_image: Option<RgbImage>,
}

/// Render, tokenize, position, and patchify one Boogu instruction with ordinary Qwen APIs.
pub fn prepare_instruction<B: Backend, T: Qwen3VlTokenizer>(
    request: &ResolvedBooguRequest,
    source: Option<&DynamicImage>,
    processor: &Qwen3VlProcessor<T>,
    image_processor: &Qwen3VlImageProcessor,
    device: &B::Device,
) -> Result<PreparedInstruction<B>, BooguError> {
    let expected_source = matches!(request.task, BooguTask::Edit);
    if expected_source != source.is_some() {
        return Err(BooguError::InvalidRequest(
            "decoded reference presence does not match the selected Boogu task".into(),
        ));
    }
    let policy = InstructionPolicy::upstream(request.task, usize::from(source.is_some()))?;
    let vision_image = source
        .map(|source| resize_reference(source, BOOGU_MAX_VLM_PIXELS, BOOGU_MAX_VLM_SIDE))
        .transpose()?;
    let vision = vision_image
        .as_ref()
        .map(|image| image_processor.preprocess(image))
        .transpose()
        .map_err(|error| BooguError::InvalidRequest(error.to_string()))?;
    let image_grids = vision
        .as_ref()
        .map(|value| vec![value.grid])
        .unwrap_or_default();
    let messages = vec![
        ChatMessage::text(ChatRole::System, policy.system_prompt()),
        ChatMessage::new(
            ChatRole::User,
            if expected_source {
                vec![ChatContent::Image, ChatContent::text(&request.prompt)]
            } else {
                vec![ChatContent::text(&request.prompt)]
            },
        ),
    ];
    let encoding = processor
        .encode_batch(
            &[ProcessorSample {
                messages: &messages,
                image_grids: &image_grids,
                video_grids: &[],
                video_metadata: &[],
            }],
            false,
        )
        .map_err(|error| BooguError::InvalidRequest(error.to_string()))?;
    let effective_length = encoding
        .attention_mask
        .first()
        .map(|mask| mask.iter().filter(|&&valid| valid).count())
        .unwrap_or(0);
    if effective_length == 0 || effective_length > policy.max_sequence_length {
        return Err(BooguError::InvalidRequest(format!(
            "instruction expands to {effective_length} tokens; released maximum is {} and truncation is disabled",
            policy.max_sequence_length
        )));
    }
    let positions = encoding
        .position_ids(processor.config().spatial_merge_size)
        .map_err(|error| BooguError::InvalidRequest(error.to_string()))?;
    let tensors = encoding
        .to_tensors::<B>(device)
        .map_err(|error| BooguError::InvalidRequest(error.to_string()))?;
    let images = vision.map(|value| Qwen3VlVisualInput {
        patches: value.to_tensor(device),
        grids: vec![value.grid],
        token_indices: encoding.flattened_image_token_indices(),
    });
    Ok(PreparedInstruction {
        model_input: Qwen3VlModelInput {
            input_ids: tensors.input_ids,
            attention_mask: Some(tensors.attention_mask),
            position_ids: Some(positions),
            images,
            videos: None,
            output_hidden_states: false,
        },
        effective_length,
        encoding,
        vision_image,
    })
}

/// Apply the upstream Diffusers postprocess mapping to validated RGB8 host pixels.
///
/// Decoder values are mapped directly from `[-1, 1]` to rounded bytes and labeled sRGB to match
/// upstream `output.rgb_u8`; no linear-to-sRGB transfer function is applied.
pub fn decoder_output_to_host<B: Backend>(output: Tensor<B, 4>) -> Result<HostImage, BooguError> {
    decoder_output_data_to_host(output.into_data())
}

/// Convert already-materialized decoder output into validated interleaved RGB8 host pixels.
///
/// This is the asynchronous-runtime counterpart to [`decoder_output_to_host`]: browser callers
/// can await one device readback, use the same [`TensorData`] for numerical comparison, and then
/// pass it here without reading the full decoded tensor from the device a second time.
pub fn decoder_output_data_to_host(data: TensorData) -> Result<HostImage, BooguError> {
    let shape = data.shape.clone();
    let [batch, channels, height, width] = shape.as_slice() else {
        return Err(BooguError::InvalidShape(format!(
            "decoder output must be rank 4 [1,3,H,W], got {shape:?}"
        )));
    };
    if *batch != 1 || *channels != 3 {
        return Err(BooguError::InvalidShape(format!(
            "decoder output must be [1,3,H,W], got [{batch},{channels},{height},{width}]"
        )));
    }
    let plane = height.checked_mul(*width).ok_or_else(|| {
        BooguError::InvalidShape("decoder output dimensions overflow host indexing".into())
    })?;
    let bytes = decoder_data_to_rgb8(data, plane)?;
    let width = u32::try_from(*width)
        .map_err(|_| BooguError::InvalidShape("decoder output width exceeds u32".into()))?;
    let height = u32::try_from(*height)
        .map_err(|_| BooguError::InvalidShape("decoder output height exceeds u32".into()))?;
    let dimensions = Dimensions::new(width, height)
        .map_err(|error| BooguError::InvalidShape(error.to_string()))?;
    let pixels = PixelBuffer::new(dimensions, PixelFormat::Rgb8, ColorSpace::Srgb, bytes)
        .map_err(|error| BooguError::InvalidShape(error.to_string()))?;
    Ok(HostImage::Pixels(pixels))
}

fn decoder_data_to_rgb8(data: TensorData, plane: usize) -> Result<Vec<u8>, BooguError> {
    match data.dtype {
        DType::F16 => decoder_values_to_rgb8(
            data.as_slice::<half::f16>()
                .map_err(|error| BooguError::InvalidShape(error.to_string()))?,
            plane,
            f32::from,
        ),
        DType::BF16 => decoder_values_to_rgb8(
            data.as_slice::<half::bf16>()
                .map_err(|error| BooguError::InvalidShape(error.to_string()))?,
            plane,
            f32::from,
        ),
        DType::F32 => decoder_values_to_rgb8(
            data.as_slice::<f32>()
                .map_err(|error| BooguError::InvalidShape(error.to_string()))?,
            plane,
            |value| value,
        ),
        DType::F64 => decoder_values_to_rgb8(
            data.as_slice::<f64>()
                .map_err(|error| BooguError::InvalidShape(error.to_string()))?,
            plane,
            |value| value as f32,
        ),
        _ => {
            let values = data
                .convert_dtype(DType::F32)
                .to_vec::<f32>()
                .map_err(|error| BooguError::InvalidShape(error.to_string()))?;
            decoder_values_to_rgb8(&values, plane, |value| value)
        }
    }
}

fn decoder_values_to_rgb8<E: Copy>(
    values: &[E],
    plane: usize,
    to_f32: impl Fn(E) -> f32,
) -> Result<Vec<u8>, BooguError> {
    let mut bytes = vec![0_u8; plane * 3];
    for pixel in 0..plane {
        for channel in 0..3 {
            let value = to_f32(values[channel * plane + pixel]);
            if !value.is_finite() {
                return Err(BooguError::InvalidShape(
                    "decoder output contains non-finite values".into(),
                ));
            }
            // Diffusers VaeImageProcessor.postprocess: clamp after mapping [-1,1] to [0,1],
            // then convert to uint8 with NumPy/Pillow's nearest integer behavior.
            let normalized = (value / 2.0 + 0.5).clamp(0.0, 1.0);
            bytes[pixel * 3 + channel] = (normalized * 255.0).round() as u8;
        }
    }
    Ok(bytes)
}

fn limited_dimensions(
    height: u32,
    width: u32,
    max_pixels: u64,
    max_side: u32,
    multiple: u32,
) -> Result<(u32, u32), BooguError> {
    if height == 0 || width == 0 || max_pixels == 0 || max_side == 0 || multiple == 0 {
        return Err(BooguError::InvalidShape(
            "image and resize limits must be non-zero".into(),
        ));
    }
    let current_pixels = u64::from(height) * u64::from(width);
    let pixel_ratio = (max_pixels as f64 / current_pixels as f64).sqrt();
    let side_ratio = f64::from(max_side) / f64::from(height.max(width));
    let ratio = pixel_ratio.min(side_ratio).min(1.0);
    let new_height = ((f64::from(height) * ratio) as u32 / multiple) * multiple;
    let new_width = ((f64::from(width) * ratio) as u32 / multiple) * multiple;
    if new_height == 0 || new_width == 0 {
        return Err(BooguError::InvalidShape(format!(
            "image {width}x{height} becomes empty after floor-to-{multiple} resize"
        )));
    }
    Ok((new_height, new_width))
}

fn pillow_compatible_rgb8(image: &DynamicImage) -> RgbImage {
    match image {
        DynamicImage::ImageLuma16(buffer) => {
            RgbImage::from_fn(buffer.width(), buffer.height(), |x, y| {
                let value = (buffer.get_pixel(x, y)[0] >> 8) as u8;
                Rgb([value; 3])
            })
        }
        DynamicImage::ImageLumaA16(buffer) => {
            RgbImage::from_fn(buffer.width(), buffer.height(), |x, y| {
                let value = (buffer.get_pixel(x, y)[0] >> 8) as u8;
                Rgb([value; 3])
            })
        }
        DynamicImage::ImageRgb16(buffer) => {
            RgbImage::from_fn(buffer.width(), buffer.height(), |x, y| {
                let pixel = buffer.get_pixel(x, y);
                Rgb([
                    (pixel[0] >> 8) as u8,
                    (pixel[1] >> 8) as u8,
                    (pixel[2] >> 8) as u8,
                ])
            })
        }
        DynamicImage::ImageRgba16(buffer) => {
            RgbImage::from_fn(buffer.width(), buffer.height(), |x, y| {
                let pixel = buffer.get_pixel(x, y);
                Rgb([
                    (pixel[0] >> 8) as u8,
                    (pixel[1] >> 8) as u8,
                    (pixel[2] >> 8) as u8,
                ])
            })
        }
        _ => image.to_rgb8(),
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;
    use burn_image::{EditRequest, GenerateRequest, GenerationOptions, Prompt};
    use image::{Rgb, RgbImage};

    use super::*;

    type B = NdArray<f32>;

    #[test]
    fn descriptors_lock_task_revision_steps_and_formats_correctness() {
        let generate = boogu_model_descriptor(BooguVariant::Image01Turbo);
        let edit = boogu_model_descriptor(BooguVariant::Image01EditTurbo);
        let edit_one_k_five = boogu_model_descriptor(BooguVariant::Image01EditTurbo1k5);
        assert_eq!(generate.revision, TURBO_REVISION);
        assert_eq!(edit.revision, EDIT_TURBO_REVISION);
        assert_eq!(edit_one_k_five.revision, EDIT_TURBO_1K5_REVISION);
        assert_eq!(
            edit_one_k_five.id.as_str(),
            "Boogu/Boogu-Image-0.1-Edit-Turbo-1K5"
        );
        assert_eq!(
            generate.capabilities.tasks,
            BTreeSet::from([ImageTaskKind::Generate])
        );
        assert_eq!(
            (
                generate.capabilities.min_steps,
                generate.capabilities.max_steps
            ),
            (4, 4)
        );
        assert!(
            generate
                .capabilities
                .numeric_formats
                .contains(&NumericFormat::F16)
        );
        for descriptor in [&generate, &edit] {
            let dimensions = &descriptor.capabilities.dimensions;
            assert_eq!(dimensions.min_width, 256);
            assert_eq!(dimensions.min_height, 256);
            assert_eq!(dimensions.max_width, BOOGU_MAX_OUTPUT_SIDE);
            assert_eq!(dimensions.max_height, BOOGU_MAX_OUTPUT_SIDE);
            assert_eq!(dimensions.width_multiple, 16);
            assert_eq!(dimensions.height_multiple, 16);
            assert_eq!(dimensions.max_pixels, Some(BOOGU_MAX_OUTPUT_PIXELS));
            assert_eq!(dimensions.allowed_dimensions, None);
        }
        let dimensions = &edit_one_k_five.capabilities.dimensions;
        assert_eq!(dimensions.max_width, BOOGU_1K5_MAX_OUTPUT_SIDE);
        assert_eq!(dimensions.max_height, BOOGU_1K5_MAX_OUTPUT_SIDE);
        assert_eq!(dimensions.max_pixels, Some(BOOGU_1K5_MAX_OUTPUT_PIXELS));
        assert_eq!(
            dimensions.allowed_dimensions,
            Some(
                BOOGU_1K5_OUTPUT_PRESETS
                    .into_iter()
                    .map(|(width, height)| Dimensions::new(width, height).unwrap())
                    .collect()
            )
        );
        assert_eq!(
            edit_one_k_five.capabilities.numeric_formats,
            BTreeSet::from([
                NumericFormat::Other("f16-qwen-vision-f32".into()),
                NumericFormat::Other("q4s-block-up-to128-f32".into()),
            ])
        );
        assert!(generate.validate().is_ok());
        assert!(edit.validate().is_ok());
        assert!(edit_one_k_five.validate().is_ok());
    }

    #[test]
    fn request_resolution_accepts_ceiling_and_rejects_larger_outputs_correctness() {
        let request = |width, height| {
            ImageRequest::Generate(GenerateRequest {
                prompt: Prompt::new("a red cube").unwrap(),
                negative_prompt: None,
                options: GenerationOptions {
                    dimensions: Some(Dimensions::new(width, height).unwrap()),
                    steps: Some(4),
                    guidance_scale: Some(1.0),
                    seed: Some(7),
                    batch_size: 1,
                },
            })
        };
        assert!(
            resolve_request(
                BooguVariant::Image01Turbo,
                &request(BOOGU_MAX_OUTPUT_SIDE, BOOGU_MAX_OUTPUT_SIDE),
                7,
            )
            .is_ok()
        );
        assert!(resolve_request(BooguVariant::Image01Turbo, &request(1040, 1024), 7).is_err());
        assert!(resolve_request(BooguVariant::Image01Turbo, &request(1024, 1040), 7).is_err());
    }

    #[test]
    fn request_resolution_rejects_cfg_and_cross_task_dispatch_correctness() {
        let request = ImageRequest::Generate(GenerateRequest {
            prompt: Prompt::new("a red cube").unwrap(),
            negative_prompt: None,
            options: GenerationOptions {
                guidance_scale: Some(1.5),
                ..GenerationOptions::default()
            },
        });
        assert!(resolve_request(BooguVariant::Image01Turbo, &request, 7).is_err());
        assert!(resolve_request(BooguVariant::Image01EditTurbo, &request, 7).is_err());
    }

    #[test]
    fn edit_request_preserves_source_and_default_seed_correctness() {
        let dimensions = Dimensions::new(256, 256).unwrap();
        let source = InputImage::Pixels(
            PixelBuffer::new(
                dimensions,
                PixelFormat::Rgb8,
                ColorSpace::Srgb,
                vec![128; 256 * 256 * 3],
            )
            .unwrap(),
        );
        let request = ImageRequest::Edit(EditRequest {
            source: source.clone(),
            instruction: Prompt::new("make it orange").unwrap(),
            negative_prompt: None,
            mask: None,
            strength: None,
            options: GenerationOptions::default(),
        });
        let resolved = resolve_request(BooguVariant::Image01EditTurbo, &request, 99).unwrap();
        assert_eq!(resolved.seed, 99);
        assert_eq!(resolved.source, Some(source));
        assert_eq!(resolved.dimensions, dimensions);
    }

    #[test]
    fn explicit_edit_dimensions_disable_reference_alignment_correctness() {
        let source_dimensions = Dimensions::new(256, 256).unwrap();
        let source = InputImage::Pixels(
            PixelBuffer::new(
                source_dimensions,
                PixelFormat::Rgb8,
                ColorSpace::Srgb,
                vec![128; 256 * 256 * 3],
            )
            .unwrap(),
        );
        let request = ImageRequest::Edit(EditRequest {
            source,
            instruction: Prompt::new("make it orange").unwrap(),
            negative_prompt: None,
            mask: None,
            strength: None,
            options: GenerationOptions {
                dimensions: Some(Dimensions::new(1024, 1024).unwrap()),
                steps: Some(4),
                guidance_scale: Some(1.0),
                seed: Some(99),
                batch_size: 1,
            },
        });
        let resolved = resolve_request(BooguVariant::Image01EditTurbo, &request, 7).unwrap();
        assert_eq!(resolved.dimensions, Dimensions::new(1024, 1024).unwrap());
    }

    #[test]
    fn one_k_five_defaults_and_aspect_ratio_ceiling_correctness() {
        let source_dimensions = Dimensions::new(1024, 1024).unwrap();
        let source = InputImage::Pixels(
            PixelBuffer::new(
                source_dimensions,
                PixelFormat::Rgb8,
                ColorSpace::Srgb,
                vec![128; 1024 * 1024 * 3],
            )
            .unwrap(),
        );
        let request = |dimensions| {
            ImageRequest::Edit(EditRequest {
                source: source.clone(),
                instruction: Prompt::new("make it orange").unwrap(),
                negative_prompt: None,
                mask: None,
                strength: None,
                options: GenerationOptions {
                    dimensions,
                    ..GenerationOptions::default()
                },
            })
        };
        let resolved =
            resolve_request(BooguVariant::Image01EditTurbo1k5, &request(None), 7).unwrap();
        assert_eq!(
            resolved.dimensions,
            Dimensions::new(BOOGU_1K5_DEFAULT_EDGE, BOOGU_1K5_DEFAULT_EDGE).unwrap()
        );
        assert!(
            resolve_request(
                BooguVariant::Image01EditTurbo1k5,
                &request(Some(Dimensions::new(2368, 992).unwrap())),
                7,
            )
            .is_ok()
        );
        assert!(
            resolve_request(
                BooguVariant::Image01EditTurbo,
                &request(Some(Dimensions::new(2368, 992).unwrap())),
                7,
            )
            .is_err()
        );
        assert!(
            resolve_request(
                BooguVariant::Image01EditTurbo1k5,
                &request(Some(Dimensions::new(1520, 1536).unwrap())),
                7,
            )
            .is_err()
        );
    }

    #[test]
    fn resize_policy_never_upscales_and_floors_to_sixteen_correctness() {
        assert_eq!(
            limited_dimensions(257, 513, 4_194_304, 4096, 16).unwrap(),
            (256, 512)
        );
        let (height, width) = limited_dimensions(3000, 5000, 4_194_304, 4096, 16).unwrap();
        assert!(height <= 3000 && width <= 4096);
        assert!(height.is_multiple_of(16) && width.is_multiple_of(16));
        assert!(u64::from(height) * u64::from(width) <= 4_194_304);
    }

    #[test]
    fn vae_reference_is_bchw_and_normalized_correctness() {
        let source = DynamicImage::ImageRgb8(RgbImage::from_pixel(16, 16, Rgb([0, 128, 255])));
        let tensor = prepare_vae_reference::<B>(&source, &Default::default()).unwrap();
        assert_eq!(tensor.dims(), [1, 3, 16, 16]);
        let values = tensor.to_data().to_vec::<f32>().unwrap();
        assert!((values[0] + 1.0).abs() < 1.0e-6);
        assert!((values[256] - (128.0 / 127.5 - 1.0)).abs() < 1.0e-6);
        assert!((values[512] - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn decoder_output_conversion_matches_diffusers_rounding_correctness() {
        let tensor = Tensor::<B, 4>::from_data(
            TensorData::new(vec![-1.0, 0.0, 1.0], [1, 3, 1, 1]),
            &Default::default(),
        );
        let HostImage::Pixels(output) = decoder_output_to_host(tensor).unwrap() else {
            panic!("expected pixels")
        };
        assert_eq!(output.bytes(), &[0, 128, 255]);
    }

    #[test]
    fn materialized_decoder_output_conversion_matches_tensor_path_correctness() {
        let values = vec![-1.0_f32, -0.5, 0.0, 0.5, 1.0, 2.0];
        let data = TensorData::new(values.clone(), [1, 3, 1, 2]);
        let HostImage::Pixels(materialized) = decoder_output_data_to_host(data).unwrap() else {
            panic!("expected pixels")
        };
        let tensor =
            Tensor::<B, 4>::from_data(TensorData::new(values, [1, 3, 1, 2]), &Default::default());
        let HostImage::Pixels(from_tensor) = decoder_output_to_host(tensor).unwrap() else {
            panic!("expected pixels")
        };

        assert_eq!(materialized, from_tensor);
        assert_eq!(materialized.bytes(), [0, 128, 255, 64, 191, 255]);
    }

    #[test]
    fn decoder_f16_host_conversion_matches_f32_bytes_correctness() {
        let values = [-1.0_f32, -0.5, 0.0, 0.5, 1.0, 2.0];
        let f32_bytes = decoder_values_to_rgb8(&values, 2, |value| value).unwrap();
        let f16_values = values.map(half::f16::from_f32);
        let f16_bytes = decoder_values_to_rgb8(&f16_values, 2, f32::from).unwrap();
        assert_eq!(f16_bytes, f32_bytes);
        assert_eq!(f16_bytes, [0, 128, 255, 64, 191, 255]);

        let data = TensorData::new(f16_values.to_vec(), [1, 3, 1, 2]);
        assert_eq!(decoder_data_to_rgb8(data, 2).unwrap(), f16_bytes);
    }

    #[test]
    fn decoder_f16_host_conversion_rejects_non_finite_values_correctness() {
        let values = [half::f16::NAN, half::f16::ZERO, half::f16::ONE];
        let error = decoder_values_to_rgb8(&values, 1, f32::from).unwrap_err();
        assert!(error.to_string().contains("non-finite"));
    }

    #[test]
    fn decoder_output_rejects_non_finite_values_correctness() {
        let tensor = Tensor::<B, 4>::from_data(
            TensorData::new(vec![f32::NAN, 0.0, 1.0], [1, 3, 1, 1]),
            &Default::default(),
        );
        let error = decoder_output_to_host(tensor).unwrap_err();
        assert!(error.to_string().contains("non-finite"));
    }

    /// Opt-in exact comparison against a fixture exported from the pinned Boogu processor.
    #[cfg(all(feature = "runtime", feature = "import"))]
    #[test]
    fn real_boogu_processor_reference() {
        use burn_qwen3_vl::{
            Qwen3VlImageProcessorConfig, Qwen3VlTokenizer, tokenizer::HfTokenizer,
        };
        use half::{bf16, f16};
        use safetensors::{Dtype, SafeTensors, tensor::TensorView};

        let Ok(model_directory) = std::env::var("BOOGU_PROCESSOR_MODEL_DIR") else {
            return;
        };
        let Ok(fixture_directory) = std::env::var("BOOGU_PROCESSOR_FIXTURE_DIR") else {
            return;
        };
        let model_directory = std::path::Path::new(&model_directory);
        let fixture_directory = std::path::Path::new(&fixture_directory);
        let metadata_bytes = std::fs::read(fixture_directory.join("metadata.json")).unwrap();
        let metadata: serde_json::Value = serde_json::from_slice(&metadata_bytes).unwrap();
        let prompt = metadata["prompt"].as_str().unwrap();
        let variant = match metadata["variant"].as_str().unwrap() {
            "turbo" => BooguVariant::Image01Turbo,
            "edit-turbo" => BooguVariant::Image01EditTurbo,
            "edit-turbo-1k5" => BooguVariant::Image01EditTurbo1k5,
            value => panic!("unsupported fixture variant {value}"),
        };
        let config = Qwen3VlConfig::from_json(
            &std::fs::read_to_string(model_directory.join("mllm/config.json")).unwrap(),
        )
        .unwrap();
        let tokenizer =
            HfTokenizer::from_file(model_directory.join("mllm/tokenizer.json")).unwrap();
        let pad_token_id = tokenizer.token_to_id("<|endoftext|>").unwrap();
        let processor =
            Qwen3VlProcessor::new(tokenizer, boogu_processor_config(&config, pad_token_id))
                .unwrap();
        let image_processor = Qwen3VlImageProcessor::new(
            Qwen3VlImageProcessorConfig::from_json(
                &std::fs::read_to_string(model_directory.join("mllm/preprocessor_config.json"))
                    .unwrap(),
            )
            .unwrap(),
        )
        .unwrap();

        let source = variant
            .is_edit()
            .then(|| image::open(fixture_directory.join("source.png")).unwrap());
        let request = match &source {
            None => ImageRequest::Generate(GenerateRequest {
                prompt: Prompt::new(prompt).unwrap(),
                negative_prompt: None,
                options: GenerationOptions::default(),
            }),
            Some(image) => {
                let rgb = image.to_rgb8();
                let dimensions = Dimensions::new(rgb.width(), rgb.height()).unwrap();
                ImageRequest::Edit(EditRequest {
                    source: InputImage::Pixels(
                        PixelBuffer::new(
                            dimensions,
                            PixelFormat::Rgb8,
                            ColorSpace::Srgb,
                            rgb.into_raw(),
                        )
                        .unwrap(),
                    ),
                    instruction: Prompt::new(prompt).unwrap(),
                    negative_prompt: None,
                    mask: None,
                    strength: None,
                    options: GenerationOptions::default(),
                })
            }
        };
        let resolved = resolve_request(variant, &request, 42).unwrap();
        let actual = prepare_instruction::<B, _>(
            &resolved,
            source.as_ref(),
            &processor,
            &image_processor,
            &Default::default(),
        )
        .unwrap();
        let fixture_bytes = std::fs::read(fixture_directory.join("tensors.safetensors")).unwrap();
        crate::reference::verify_reference_fixture(&metadata_bytes, &fixture_bytes).unwrap();
        let fixture = SafeTensors::deserialize(&fixture_bytes).unwrap();
        let integers = |name: &str| {
            let view = fixture.tensor(name).unwrap();
            assert_eq!(view.dtype(), Dtype::I64);
            view.data()
                .chunks_exact(8)
                .map(|bytes| i64::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            actual.encoding.input_ids[0],
            integers("processor.input_ids")
        );
        assert_eq!(
            actual.encoding.attention_mask[0],
            integers("processor.attention_mask")
                .into_iter()
                .map(|value| value != 0)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            actual.encoding.mm_token_type_ids[0],
            integers("processor.mm_token_type_ids")
                .into_iter()
                .map(|value| value as u8)
                .collect::<Vec<_>>()
        );
        if let Some(vision) = &actual.model_input.images {
            let values = |view: TensorView<'_>| match view.dtype() {
                Dtype::F32 => view
                    .data()
                    .chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                    .collect::<Vec<_>>(),
                Dtype::F16 => view
                    .data()
                    .chunks_exact(2)
                    .map(|bytes| {
                        f16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap())).to_f32()
                    })
                    .collect::<Vec<_>>(),
                Dtype::BF16 => view
                    .data()
                    .chunks_exact(2)
                    .map(|bytes| {
                        bf16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap())).to_f32()
                    })
                    .collect::<Vec<_>>(),
                dtype => panic!("unsupported fixture dtype {dtype:?}"),
            };
            let expected = values(fixture.tensor("processor.pixel_values").unwrap());
            let observed = vision.patches.to_data().to_vec::<f32>().unwrap();
            assert_eq!(observed.len(), expected.len());
            let max_abs = observed
                .iter()
                .zip(expected)
                .map(|(&left, right)| (left - right).abs())
                .fold(0.0_f32, f32::max);
            assert!(max_abs <= f32::EPSILON, "processor max_abs={max_abs:e}");
            assert_eq!(
                [
                    vision.grids[0].t as i64,
                    vision.grids[0].h as i64,
                    vision.grids[0].w as i64,
                ],
                integers("processor.image_grid_thw").as_slice()
            );
        }
    }
}
