//! Decoded-RGB preprocessing compatible with Qwen3-VL image checkpoints.

use burn::tensor::{Tensor, TensorData, backend::Backend};
use image::{DynamicImage, Rgb, RgbImage};
use serde::{Deserialize, Serialize};

use crate::{Grid, Qwen3VlError, Result};

fn default_shortest_edge() -> usize {
    65_536
}
fn default_longest_edge() -> usize {
    16_777_216
}
fn default_patch_size() -> usize {
    16
}
fn default_temporal_patch_size() -> usize {
    2
}
fn default_merge_size() -> usize {
    2
}
fn default_mean() -> [f32; 3] {
    [0.5; 3]
}
fn default_std() -> [f32; 3] {
    [0.5; 3]
}
fn default_rescale_factor() -> f32 {
    1.0 / 255.0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PixelLimits {
    #[serde(default = "default_shortest_edge")]
    pub shortest_edge: usize,
    #[serde(default = "default_longest_edge")]
    pub longest_edge: usize,
}

impl Default for PixelLimits {
    fn default() -> Self {
        Self {
            shortest_edge: default_shortest_edge(),
            longest_edge: default_longest_edge(),
        }
    }
}

/// Fields from the published Qwen `preprocessor_config.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen3VlImageProcessorConfig {
    #[serde(default)]
    pub size: PixelLimits,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: usize,
    #[serde(default = "default_merge_size", alias = "spatial_merge_size")]
    pub merge_size: usize,
    #[serde(default = "default_mean")]
    pub image_mean: [f32; 3],
    #[serde(default = "default_std")]
    pub image_std: [f32; 3],
    #[serde(default = "default_rescale_factor")]
    pub rescale_factor: f32,
}

impl Default for Qwen3VlImageProcessorConfig {
    fn default() -> Self {
        Self {
            size: PixelLimits::default(),
            patch_size: default_patch_size(),
            temporal_patch_size: default_temporal_patch_size(),
            merge_size: default_merge_size(),
            image_mean: default_mean(),
            image_std: default_std(),
            rescale_factor: default_rescale_factor(),
        }
    }
}

impl Qwen3VlImageProcessorConfig {
    pub fn from_json(json: &str) -> Result<Self> {
        let config = serde_json::from_str(json)?;
        let processor = Qwen3VlImageProcessor::new(config)?;
        Ok(processor.config)
    }
}

/// Flattened, normalized vision patches and the grid consumed by the vision transformer.
#[derive(Debug, Clone, PartialEq)]
pub struct ProcessedVisionPixels {
    pub patches: Vec<f32>,
    pub grid: Grid,
    pub resized_height: usize,
    pub resized_width: usize,
    pub patch_width: usize,
}

impl ProcessedVisionPixels {
    pub fn patch_count(&self) -> usize {
        self.grid.patch_count()
    }

    pub fn to_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 2> {
        Tensor::from_data(
            TensorData::new(self.patches.clone(), [self.patch_count(), self.patch_width]),
            device,
        )
    }
}

/// Qwen smart-resize, bicubic RGB normalization, temporal duplication, and patch flattening.
#[derive(Debug, Clone)]
pub struct Qwen3VlImageProcessor {
    config: Qwen3VlImageProcessorConfig,
}

impl Qwen3VlImageProcessor {
    pub fn new(config: Qwen3VlImageProcessorConfig) -> Result<Self> {
        if config.patch_size == 0 || config.temporal_patch_size == 0 || config.merge_size == 0 {
            return Err(Qwen3VlError::InvalidConfig(
                "image patch and merge sizes must be non-zero".into(),
            ));
        }
        if config.size.shortest_edge == 0 || config.size.longest_edge < config.size.shortest_edge {
            return Err(Qwen3VlError::InvalidConfig(
                "image pixel limits must be positive and ordered".into(),
            ));
        }
        if config
            .image_std
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
            || config.image_mean.iter().any(|value| !value.is_finite())
            || !config.rescale_factor.is_finite()
        {
            return Err(Qwen3VlError::InvalidConfig(
                "image normalization values must be finite and std must be positive".into(),
            ));
        }
        Ok(Self { config })
    }

    pub fn config(&self) -> &Qwen3VlImageProcessorConfig {
        &self.config
    }

    /// Qwen's aspect-preserving resize with dimensions divisible by `patch_size * merge_size`.
    pub fn smart_resize(&self, height: usize, width: usize) -> Result<(usize, usize)> {
        if height == 0 || width == 0 {
            return Err(Qwen3VlError::InvalidInput(
                "decoded image dimensions must be non-zero".into(),
            ));
        }
        let aspect_ratio = height.max(width) as f64 / height.min(width) as f64;
        if aspect_ratio > 200.0 {
            return Err(Qwen3VlError::InvalidInput(format!(
                "absolute image aspect ratio must not exceed 200, got {aspect_ratio}"
            )));
        }
        let factor = self.config.patch_size * self.config.merge_size;
        // Python's `round`, used by the reference processor, is ties-to-even.
        let mut resized_height =
            ((height as f64 / factor as f64).round_ties_even() as usize).saturating_mul(factor);
        let mut resized_width =
            ((width as f64 / factor as f64).round_ties_even() as usize).saturating_mul(factor);
        let rounded_pixels = resized_height.saturating_mul(resized_width);
        if rounded_pixels > self.config.size.longest_edge {
            let beta = ((height * width) as f64 / self.config.size.longest_edge as f64).sqrt();
            resized_height =
                factor.max(((height as f64 / beta / factor as f64).floor() as usize) * factor);
            resized_width =
                factor.max(((width as f64 / beta / factor as f64).floor() as usize) * factor);
        } else if rounded_pixels < self.config.size.shortest_edge {
            let beta = (self.config.size.shortest_edge as f64 / (height * width) as f64).sqrt();
            resized_height = ((height as f64 * beta / factor as f64).ceil() as usize) * factor;
            resized_width = ((width as f64 * beta / factor as f64).ceil() as usize) * factor;
        }
        Ok((resized_height, resized_width))
    }

    /// Process one decoded RGB image. Its single frame is repeated to fill the temporal patch.
    pub fn preprocess(&self, image: &RgbImage) -> Result<ProcessedVisionPixels> {
        let (resized_height, resized_width) =
            self.smart_resize(image.height() as usize, image.width() as usize)?;
        let resized = resize_rgb8_bicubic_antialias(image, resized_width, resized_height);
        let grid_h = resized_height / self.config.patch_size;
        let grid_w = resized_width / self.config.patch_size;
        let grid = Grid::new(1, grid_h, grid_w);
        grid.validate(self.config.merge_size)?;
        let patch_width =
            3 * self.config.temporal_patch_size * self.config.patch_size * self.config.patch_size;
        let mut patches = Vec::with_capacity(grid.patch_count() * patch_width);

        // Equivalent to the reference view+permute:
        // [grid_t, block_h, block_w, merge_h, merge_w, channel, temporal, patch_h, patch_w].
        for block_h in 0..grid_h / self.config.merge_size {
            for block_w in 0..grid_w / self.config.merge_size {
                for merge_h in 0..self.config.merge_size {
                    for merge_w in 0..self.config.merge_size {
                        let patch_h = block_h * self.config.merge_size + merge_h;
                        let patch_w = block_w * self.config.merge_size + merge_w;
                        for channel in 0..3 {
                            for _temporal in 0..self.config.temporal_patch_size {
                                for y in 0..self.config.patch_size {
                                    for x in 0..self.config.patch_size {
                                        let pixel = resized.get_pixel(
                                            (patch_w * self.config.patch_size + x) as u32,
                                            (patch_h * self.config.patch_size + y) as u32,
                                        );
                                        let value =
                                            pixel[channel] as f32 * self.config.rescale_factor;
                                        patches.push(
                                            (value - self.config.image_mean[channel])
                                                / self.config.image_std[channel],
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(ProcessedVisionPixels {
            patches,
            grid,
            resized_height,
            resized_width,
            patch_width,
        })
    }

    /// Process a dynamically typed decoded image using the same RGB conversion as the
    /// Transformers PIL input path.
    ///
    /// In particular, Pillow converts 16-bit integer channels to RGB8 by retaining the high
    /// byte. [`DynamicImage::to_rgb8`] instead rescales with rounding, which changes most
    /// non-8-bit-aligned samples by one and is observable after Qwen normalization.
    pub fn preprocess_dynamic(&self, image: &DynamicImage) -> Result<ProcessedVisionPixels> {
        self.preprocess(&pillow_compatible_rgb8(image))
    }

    /// Process and concatenate a batch while retaining one grid per source image.
    pub fn preprocess_batch(&self, images: &[RgbImage]) -> Result<(Vec<f32>, Vec<Grid>)> {
        if images.is_empty() {
            return Err(Qwen3VlError::InvalidInput(
                "image batch must not be empty".into(),
            ));
        }
        let mut patches = Vec::new();
        let mut grids = Vec::with_capacity(images.len());
        for image in images {
            let output = self.preprocess(image)?;
            patches.extend(output.patches);
            grids.push(output.grid);
        }
        Ok((patches, grids))
    }
}

const RESAMPLE_PRECISION_BITS: u32 = 22;

#[derive(Debug)]
struct ResampleCoefficients {
    start: usize,
    weights: Vec<i32>,
}

/// Pillow-compatible separable bicubic resampling for decoded RGB8 images.
///
/// Transformers' fast Qwen processor evaluates torchvision's antialiased bicubic operator on a
/// uint8 tensor. Pillow's fixed-point bicubic kernel is the closest portable host implementation:
/// on the pinned resized stress fixture it agrees on more than 99.3% of channels, never differs
/// by more than two RGB levels, and is materially closer than `image`'s Catmull-Rom kernel. The
/// scale-dependent support also provides the required antialias low-pass when downsampling.
fn resize_rgb8_bicubic_antialias(
    image: &RgbImage,
    output_width: usize,
    output_height: usize,
) -> RgbImage {
    let input_width = image.width() as usize;
    let input_height = image.height() as usize;
    if (input_width, input_height) == (output_width, output_height) {
        return image.clone();
    }

    let horizontal = resample_coefficients(input_width, output_width);
    let vertical = resample_coefficients(input_height, output_height);
    let source = image.as_raw();
    let mut intermediate = vec![0_u8; input_height * output_width * 3];
    for y in 0..input_height {
        for (x, coefficients) in horizontal.iter().enumerate() {
            let output = (y * output_width + x) * 3;
            for channel in 0..3 {
                let mut accumulator = 1_i64 << (RESAMPLE_PRECISION_BITS - 1);
                for (offset, &weight) in coefficients.weights.iter().enumerate() {
                    let input = (y * input_width + coefficients.start + offset) * 3 + channel;
                    accumulator += i64::from(source[input]) * i64::from(weight);
                }
                intermediate[output + channel] = resample_clip(accumulator);
            }
        }
    }

    let mut output = vec![0_u8; output_height * output_width * 3];
    for (y, coefficients) in vertical.iter().enumerate() {
        for x in 0..output_width {
            let destination = (y * output_width + x) * 3;
            for channel in 0..3 {
                let mut accumulator = 1_i64 << (RESAMPLE_PRECISION_BITS - 1);
                for (offset, &weight) in coefficients.weights.iter().enumerate() {
                    let source = ((coefficients.start + offset) * output_width + x) * 3 + channel;
                    accumulator += i64::from(intermediate[source]) * i64::from(weight);
                }
                output[destination + channel] = resample_clip(accumulator);
            }
        }
    }
    RgbImage::from_raw(output_width as u32, output_height as u32, output)
        .expect("resampler constructs an exact RGB buffer")
}

fn resample_coefficients(input: usize, output: usize) -> Vec<ResampleCoefficients> {
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.0);
    let support = 2.0 * filter_scale;
    let fixed_scale = (1_i64 << RESAMPLE_PRECISION_BITS) as f64;
    (0..output)
        .map(|position| {
            let center = (position as f64 + 0.5) * scale;
            // These casts intentionally truncate toward zero, matching Pillow's C implementation.
            let start = ((center - support + 0.5) as isize).max(0) as usize;
            let end = ((center + support + 0.5) as usize).min(input);
            let mut floating = (start..end)
                .map(|sample| bicubic_kernel((sample as f64 - center + 0.5) / filter_scale))
                .collect::<Vec<_>>();
            let sum = floating.iter().sum::<f64>();
            if sum != 0.0 {
                for weight in &mut floating {
                    *weight /= sum;
                }
            }
            let weights = floating
                .into_iter()
                .map(|weight| {
                    let rounded = weight * fixed_scale + if weight < 0.0 { -0.5 } else { 0.5 };
                    rounded as i32
                })
                .collect();
            ResampleCoefficients { start, weights }
        })
        .collect()
}

fn bicubic_kernel(value: f64) -> f64 {
    let value = value.abs();
    const A: f64 = -0.5;
    if value < 1.0 {
        ((A + 2.0) * value - (A + 3.0)) * value * value + 1.0
    } else if value < 2.0 {
        ((A * value - 5.0 * A) * value + 8.0 * A) * value - 4.0 * A
    } else {
        0.0
    }
}

fn resample_clip(accumulator: i64) -> u8 {
    (accumulator >> RESAMPLE_PRECISION_BITS).clamp(0, 255) as u8
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
    use super::*;
    use image::Rgb;

    #[cfg(feature = "import")]
    fn safetensors_f32(view: safetensors::tensor::TensorView<'_>) -> Vec<f32> {
        assert_eq!(view.dtype(), safetensors::Dtype::F32);
        view.data()
            .chunks_exact(size_of::<f32>())
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect()
    }

    #[cfg(feature = "import")]
    fn safetensors_i64(view: safetensors::tensor::TensorView<'_>) -> Vec<i64> {
        assert_eq!(view.dtype(), safetensors::Dtype::I64);
        view.data()
            .chunks_exact(size_of::<i64>())
            .map(|bytes| i64::from_le_bytes(bytes.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn published_preprocessor_config_parses_reference() {
        let json = r#"{
          "size":{"longest_edge":16777216,"shortest_edge":65536},
          "patch_size":16,"temporal_patch_size":2,"merge_size":2,
          "image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],
          "processor_class":"Qwen3VLProcessor","image_processor_type":"Qwen2VLImageProcessorFast"
        }"#;
        let config = Qwen3VlImageProcessorConfig::from_json(json).unwrap();
        assert_eq!(config.size.shortest_edge, 65_536);
        assert_eq!(config.size.longest_edge, 16_777_216);
    }

    #[test]
    fn smart_resize_matches_reference_correctness() {
        let processor = Qwen3VlImageProcessor::new(Qwen3VlImageProcessorConfig::default()).unwrap();
        assert_eq!(processor.smart_resize(100, 200).unwrap(), (192, 384));
        assert_eq!(processor.smart_resize(1024, 1024).unwrap(), (1024, 1024));
        assert!(processor.smart_resize(1, 201).is_err());
    }

    #[test]
    fn normalization_temporal_duplication_and_patch_order_correctness() {
        let config = Qwen3VlImageProcessorConfig {
            size: PixelLimits {
                shortest_edge: 16,
                longest_edge: 16,
            },
            patch_size: 2,
            temporal_patch_size: 2,
            merge_size: 2,
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
            rescale_factor: 1.0 / 255.0,
        };
        let processor = Qwen3VlImageProcessor::new(config).unwrap();
        let image = RgbImage::from_pixel(4, 4, Rgb([0, 127, 255]));
        let output = processor.preprocess(&image).unwrap();
        assert_eq!(output.grid, Grid::new(1, 2, 2));
        assert_eq!(output.patch_count(), 4);
        assert_eq!(output.patch_width, 24);
        let first = &output.patches[..24];
        assert!(first[..8].iter().all(|value| (*value + 1.0).abs() < 1e-6));
        assert!(
            first[8..16]
                .iter()
                .all(|value| (*value - (127.0 / 127.5 - 1.0)).abs() < 1e-6)
        );
        assert!(first[16..].iter().all(|value| (*value - 1.0).abs() < 1e-6));
    }

    /// Opt-in comparison with a Transformers reference bundle. The fixture directory must
    /// contain `source.png` and `tensors.safetensors`; the processor config is supplied
    /// separately so this remains independent of any application-specific artifact layout.
    #[cfg(feature = "import")]
    #[test]
    fn real_transformers_image_processor_reference() {
        let Ok(fixture_directory) = std::env::var("QWEN3_VL_PROCESSOR_FIXTURE_DIR") else {
            return;
        };
        let config_path = std::env::var("QWEN3_VL_PREPROCESSOR_CONFIG")
            .expect("QWEN3_VL_PREPROCESSOR_CONFIG is required with a processor fixture");
        let config =
            Qwen3VlImageProcessorConfig::from_json(&std::fs::read_to_string(config_path).unwrap())
                .unwrap();
        let processor = Qwen3VlImageProcessor::new(config).unwrap();
        let image =
            image::open(std::path::Path::new(&fixture_directory).join("source.png")).unwrap();
        let actual = processor.preprocess_dynamic(&image).unwrap();
        let rounded_conversion = processor.preprocess(&image.to_rgb8()).unwrap();
        assert_eq!(
            (actual.resized_height, actual.resized_width),
            (image.height() as usize, image.width() as usize),
            "the pinned fixture isolates decoded RGB conversion because resize is a no-op"
        );

        let bytes =
            std::fs::read(std::path::Path::new(&fixture_directory).join("tensors.safetensors"))
                .unwrap();
        let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let expected_grid = safetensors_i64(tensors.tensor("processor.image_grid_thw").unwrap());
        assert_eq!(
            expected_grid,
            vec![
                actual.grid.t as i64,
                actual.grid.h as i64,
                actual.grid.w as i64
            ]
        );
        let expected = safetensors_f32(tensors.tensor("processor.pixel_values").unwrap());
        assert_eq!(expected.len(), actual.patches.len());

        let mut max_abs = 0.0_f32;
        let mut sum_square = 0.0_f64;
        let mut differing = 0_usize;
        let mut rounded_max_abs = 0.0_f32;
        let mut rounded_sum_square = 0.0_f64;
        for (&reference, &observed) in expected.iter().zip(&actual.patches) {
            let error = (reference - observed).abs();
            max_abs = max_abs.max(error);
            sum_square += f64::from(error).powi(2);
            differing += usize::from(error != 0.0);
        }
        for (&reference, &observed) in expected.iter().zip(&rounded_conversion.patches) {
            let error = (reference - observed).abs();
            rounded_max_abs = rounded_max_abs.max(error);
            rounded_sum_square += f64::from(error).powi(2);
        }
        let rmse = (sum_square / expected.len() as f64).sqrt();
        let rounded_rmse = (rounded_sum_square / expected.len() as f64).sqrt();
        eprintln!(
            "processor reference: grid={:?}, shape=[{}, {}], differing={}/{}, max_abs={max_abs:e}, rmse={rmse:e}; image::to_rgb8 rounded path max_abs={rounded_max_abs:e}, rmse={rounded_rmse:e}",
            actual.grid,
            actual.patch_count(),
            actual.patch_width,
            differing,
            expected.len()
        );
        assert!(max_abs <= f32::EPSILON);
        assert!(rmse <= f64::from(f32::EPSILON));
        assert!((rounded_max_abs - 1.0 / 127.5).abs() <= f32::EPSILON);
    }

    /// Opt-in resized-image diagnostic against Transformers' torchvision antialiased bicubic
    /// path. This is separate from the pinned Edit fixture, whose 256 by 256 source is not
    /// resized and therefore cannot validate interpolation semantics.
    #[cfg(feature = "import")]
    #[test]
    fn real_transformers_resized_image_processor_reference() {
        let Some(fixture_directory) =
            std::env::var_os("QWEN3_VL_RESIZE_FIXTURE_DIR").map(std::path::PathBuf::from)
        else {
            return;
        };
        let config_path = std::env::var_os("QWEN3_VL_PREPROCESSOR_CONFIG")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| fixture_directory.join("preprocessor_config.json"));
        let config =
            Qwen3VlImageProcessorConfig::from_json(&std::fs::read_to_string(config_path).unwrap())
                .unwrap();
        let processor = Qwen3VlImageProcessor::new(config).unwrap();
        let image = image::open(fixture_directory.join("source.png")).unwrap();
        let actual = processor.preprocess_dynamic(&image).unwrap();
        assert_ne!(
            (actual.resized_height, actual.resized_width),
            (image.height() as usize, image.width() as usize),
            "resize fixture must exercise interpolation"
        );

        let bytes = std::fs::read(fixture_directory.join("tensors.safetensors")).unwrap();
        let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let expected_grid = safetensors_i64(tensors.tensor("processor.image_grid_thw").unwrap());
        assert_eq!(
            expected_grid,
            vec![
                actual.grid.t as i64,
                actual.grid.h as i64,
                actual.grid.w as i64
            ]
        );
        let expected = safetensors_f32(tensors.tensor("processor.pixel_values").unwrap());
        assert_eq!(expected.len(), actual.patches.len());

        let mut max_abs = 0.0_f32;
        let mut sum_abs = 0.0_f64;
        let mut sum_square = 0.0_f64;
        let mut dot = 0.0_f64;
        let mut actual_square = 0.0_f64;
        let mut expected_square = 0.0_f64;
        for (&reference, &observed) in expected.iter().zip(&actual.patches) {
            let error = reference - observed;
            max_abs = max_abs.max(error.abs());
            sum_abs += f64::from(error.abs());
            sum_square += f64::from(error).powi(2);
            dot += f64::from(reference) * f64::from(observed);
            actual_square += f64::from(observed).powi(2);
            expected_square += f64::from(reference).powi(2);
        }
        let count = expected.len() as f64;
        let rmse = (sum_square / count).sqrt();
        let cosine = dot / (actual_square.sqrt() * expected_square.sqrt());
        eprintln!(
            "resized processor reference: source={}x{}, resized={}x{}, grid={:?}, max_abs={max_abs:e}, mean_abs={:e}, rmse={rmse:e}, cosine={cosine:.9}",
            image.height(),
            image.width(),
            actual.resized_height,
            actual.resized_width,
            actual.grid,
            sum_abs / count,
        );
        // Torchvision and Pillow deliberately use slightly different fixed-point rounding. The
        // portable kernel agrees on more than 99.3% of raw channels; every mismatch is at most
        // two RGB levels. These bounds are the normalized form of that measured contract.
        assert!(max_abs <= 2.0 / 127.5 + f32::EPSILON);
        assert!(sum_abs / count <= 5.0e-5);
        assert!(rmse <= 6.3e-4);
        assert!(cosine >= 0.999999);
    }
}
