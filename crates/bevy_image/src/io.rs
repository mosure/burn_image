use std::io::Cursor;

use bevy::prelude::*;
use burn_image::{
    ColorSpace, Dimensions, HostImage, ImageEncoding, InputImage, PixelBuffer, PixelFormat,
};
use half::f16;
use image::{DynamicImage, ImageFormat, RgbaImage};

use crate::{FrontendError, FrontendErrorKind};

/// Converts a canonical host image into tightly packed RGBA8 pixels for Bevy
/// display or conventional image encoders.
pub fn host_image_rgba8(image: &HostImage) -> Result<(Dimensions, Vec<u8>), FrontendError> {
    match image {
        HostImage::Pixels(pixels) => pixel_buffer_rgba8(pixels),
        HostImage::Encoded(encoded) => {
            let format = encoding_to_format(encoded.encoding());
            let decoded = if let Some(format) = format {
                image::load_from_memory_with_format(encoded.bytes(), format)?
            } else {
                image::load_from_memory(encoded.bytes())?
            };
            dynamic_image_rgba8(decoded)
        }
    }
}

pub fn pixel_buffer_rgba8(pixels: &PixelBuffer) -> Result<(Dimensions, Vec<u8>), FrontendError> {
    if pixels.color_space() == ColorSpace::DisplayP3 {
        return Err(FrontendError::new(
            FrontendErrorKind::UnsupportedImage,
            "Display-P3 conversion is not available; refusing to mislabel colors as sRGB",
        ));
    }
    let dimensions = pixels.dimensions();
    let pixel_count = usize::try_from(dimensions.area()).map_err(|_| {
        FrontendError::new(
            FrontendErrorKind::UnsupportedImage,
            "image area does not fit host memory",
        )
    })?;
    let mut rgba = Vec::with_capacity(pixel_count.saturating_mul(4));
    match pixels.format() {
        PixelFormat::L8 => {
            for &value in pixels.bytes() {
                let value = encode_u8_channel(value, pixels.color_space());
                rgba.extend_from_slice(&[value, value, value, 255]);
            }
        }
        PixelFormat::Rgb8 => {
            for pixel in pixels.bytes().chunks_exact(3) {
                rgba.extend_from_slice(&[
                    encode_u8_channel(pixel[0], pixels.color_space()),
                    encode_u8_channel(pixel[1], pixels.color_space()),
                    encode_u8_channel(pixel[2], pixels.color_space()),
                    255,
                ]);
            }
        }
        PixelFormat::Rgba8 => {
            for pixel in pixels.bytes().chunks_exact(4) {
                rgba.extend_from_slice(&[
                    encode_u8_channel(pixel[0], pixels.color_space()),
                    encode_u8_channel(pixel[1], pixels.color_space()),
                    encode_u8_channel(pixel[2], pixels.color_space()),
                    pixel[3],
                ]);
            }
        }
        PixelFormat::Rgba16Float => {
            for pixel in pixels.bytes().chunks_exact(8) {
                let channel =
                    |index: usize| f16::from_le_bytes([pixel[index], pixel[index + 1]]).to_f32();
                rgba.extend_from_slice(&[
                    encode_float_channel(channel(0), pixels.color_space()),
                    encode_float_channel(channel(2), pixels.color_space()),
                    encode_float_channel(channel(4), pixels.color_space()),
                    encode_alpha(channel(6)),
                ]);
            }
        }
        PixelFormat::Rgba32Float => {
            for pixel in pixels.bytes().chunks_exact(16) {
                let channel = |index: usize| {
                    f32::from_le_bytes(pixel[index..index + 4].try_into().expect("four bytes"))
                };
                rgba.extend_from_slice(&[
                    encode_float_channel(channel(0), pixels.color_space()),
                    encode_float_channel(channel(4), pixels.color_space()),
                    encode_float_channel(channel(8), pixels.color_space()),
                    encode_alpha(channel(12)),
                ]);
            }
        }
    }
    Ok((dimensions, rgba))
}

fn encode_u8_channel(value: u8, color_space: ColorSpace) -> u8 {
    if color_space == ColorSpace::LinearSrgb {
        encode_srgb(f32::from(value) / 255.0)
    } else {
        value
    }
}

fn encode_float_channel(value: f32, color_space: ColorSpace) -> u8 {
    if color_space == ColorSpace::LinearSrgb {
        encode_srgb(value)
    } else {
        encode_unorm(value)
    }
}

fn encode_srgb(value: f32) -> u8 {
    let linear = value.clamp(0.0, 1.0);
    let srgb = if linear <= 0.003_130_8 {
        12.92 * linear
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    };
    encode_unorm(srgb)
}

fn encode_alpha(value: f32) -> u8 {
    encode_unorm(value)
}

fn encode_unorm(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn dynamic_image_rgba8(decoded: DynamicImage) -> Result<(Dimensions, Vec<u8>), FrontendError> {
    let rgba = decoded.to_rgba8();
    let dimensions = Dimensions::new(rgba.width(), rgba.height())?;
    Ok((dimensions, rgba.into_raw()))
}

/// Decodes browser/native bytes into a canonical edit input.
pub fn decode_input_image(
    bytes: &[u8],
    encoding: Option<ImageEncoding>,
) -> Result<InputImage, FrontendError> {
    if bytes.is_empty() {
        return Err(FrontendError::new(
            FrontendErrorKind::ImageDecode,
            "input image bytes are empty",
        ));
    }
    let decoded = match encoding.and_then(encoding_to_format) {
        Some(format) => image::load_from_memory_with_format(bytes, format)?,
        None => image::load_from_memory(bytes)?,
    };
    let (dimensions, rgba) = dynamic_image_rgba8(decoded)?;
    Ok(InputImage::Pixels(PixelBuffer::new(
        dimensions,
        PixelFormat::Rgba8,
        ColorSpace::Srgb,
        rgba,
    )?))
}

/// Encodes a host output without filesystem access, making the same operation
/// usable by native save dialogs and browser download APIs.
pub fn encode_host_image(
    image: &HostImage,
    encoding: ImageEncoding,
) -> Result<Vec<u8>, FrontendError> {
    let format = encoding_to_format(encoding).ok_or_else(|| {
        FrontendError::new(
            FrontendErrorKind::UnsupportedImage,
            format!("encoding {encoding:?} is not enabled"),
        )
    })?;
    let (dimensions, rgba) = host_image_rgba8(image)?;
    let rgba =
        RgbaImage::from_raw(dimensions.width(), dimensions.height(), rgba).ok_or_else(|| {
            FrontendError::new(
                FrontendErrorKind::ImageEncode,
                "RGBA byte length does not match image dimensions",
            )
        })?;
    let dynamic = if encoding == ImageEncoding::Jpeg {
        DynamicImage::ImageRgb8(DynamicImage::ImageRgba8(rgba).to_rgb8())
    } else {
        DynamicImage::ImageRgba8(rgba)
    };
    let mut output = Cursor::new(Vec::new());
    dynamic
        .write_to(&mut output, format)
        .map_err(|error| FrontendError::new(FrontendErrorKind::ImageEncode, error.to_string()))?;
    Ok(output.into_inner())
}

fn encoding_to_format(encoding: ImageEncoding) -> Option<ImageFormat> {
    match encoding {
        ImageEncoding::Png => Some(ImageFormat::Png),
        ImageEncoding::Jpeg => Some(ImageFormat::Jpeg),
        ImageEncoding::Webp => Some(ImageFormat::WebP),
        ImageEncoding::Avif | ImageEncoding::Other => None,
    }
}

#[cfg(all(feature = "native-io", not(target_arch = "wasm32")))]
pub fn load_image_file(path: impl AsRef<std::path::Path>) -> Result<InputImage, FrontendError> {
    decode_input_image(&std::fs::read(path)?, None)
}

#[cfg(all(feature = "native-io", not(target_arch = "wasm32")))]
pub fn save_image_file(
    path: impl AsRef<std::path::Path>,
    image: &HostImage,
    encoding: ImageEncoding,
) -> Result<(), FrontendError> {
    std::fs::write(path, encode_host_image(image, encoding)?)?;
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageIoId(pub u64);

#[derive(Message, Clone, Debug)]
pub struct LoadImageBytes {
    pub id: ImageIoId,
    pub bytes: Vec<u8>,
    pub encoding: Option<ImageEncoding>,
}

#[derive(Message, Clone, Debug)]
pub struct ImageBytesLoaded {
    pub id: ImageIoId,
    pub image: InputImage,
}

#[derive(Message, Clone, Debug)]
pub struct EncodeImageBytes {
    pub id: ImageIoId,
    pub image: HostImage,
    pub encoding: ImageEncoding,
}

#[derive(Message, Clone, Debug)]
pub struct ImageBytesEncoded {
    pub id: ImageIoId,
    pub bytes: Vec<u8>,
    pub encoding: ImageEncoding,
}

/// Requests download-ready encoded bytes without invoking browser globals.
/// A web host can create a `Blob`/object URL from [`ImageDownloadReady`].
#[derive(Message, Clone, Debug)]
pub struct PrepareImageDownload {
    pub id: ImageIoId,
    pub image: HostImage,
    pub encoding: ImageEncoding,
    pub file_stem: String,
}

#[derive(Message, Clone, Debug)]
pub struct ImageDownloadReady {
    pub id: ImageIoId,
    pub file_name: String,
    pub mime_type: &'static str,
    pub bytes: Vec<u8>,
}

#[derive(Message, Clone, Debug)]
pub struct ImageIoFailed {
    pub id: ImageIoId,
    pub error: FrontendError,
}

pub struct ImageIoPlugin;

impl Plugin for ImageIoPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<LoadImageBytes>()
            .add_message::<ImageBytesLoaded>()
            .add_message::<EncodeImageBytes>()
            .add_message::<ImageBytesEncoded>()
            .add_message::<PrepareImageDownload>()
            .add_message::<ImageDownloadReady>()
            .add_message::<ImageIoFailed>()
            .add_systems(
                Update,
                (
                    decode_load_messages,
                    encode_save_messages,
                    prepare_download_messages,
                ),
            );
    }
}

fn prepare_download_messages(
    mut requests: MessageReader<PrepareImageDownload>,
    mut ready: MessageWriter<ImageDownloadReady>,
    mut failed: MessageWriter<ImageIoFailed>,
) {
    for request in requests.read() {
        let result = download_metadata(&request.file_stem, request.encoding).and_then(
            |(file_name, mime_type)| {
                encode_host_image(&request.image, request.encoding)
                    .map(|bytes| (file_name, mime_type, bytes))
            },
        );
        match result {
            Ok((file_name, mime_type, bytes)) => {
                ready.write(ImageDownloadReady {
                    id: request.id,
                    file_name,
                    mime_type,
                    bytes,
                });
            }
            Err(error) => {
                failed.write(ImageIoFailed {
                    id: request.id,
                    error,
                });
            }
        }
    }
}

fn download_metadata(
    file_stem: &str,
    encoding: ImageEncoding,
) -> Result<(String, &'static str), FrontendError> {
    if file_stem.is_empty()
        || file_stem.len() > 128
        || !file_stem
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(FrontendError::invalid_request(
            "download file stem must contain 1..=128 ASCII letters, digits, '-' or '_'",
        ));
    }
    let (extension, mime_type) = match encoding {
        ImageEncoding::Png => ("png", "image/png"),
        ImageEncoding::Jpeg => ("jpg", "image/jpeg"),
        ImageEncoding::Webp => ("webp", "image/webp"),
        ImageEncoding::Avif | ImageEncoding::Other => {
            return Err(FrontendError::new(
                FrontendErrorKind::UnsupportedImage,
                format!("encoding {encoding:?} is not enabled"),
            ));
        }
    };
    Ok((format!("{file_stem}.{extension}"), mime_type))
}

fn decode_load_messages(
    mut requests: MessageReader<LoadImageBytes>,
    mut loaded: MessageWriter<ImageBytesLoaded>,
    mut failed: MessageWriter<ImageIoFailed>,
) {
    for request in requests.read() {
        match decode_input_image(&request.bytes, request.encoding) {
            Ok(image) => {
                loaded.write(ImageBytesLoaded {
                    id: request.id,
                    image,
                });
            }
            Err(error) => {
                failed.write(ImageIoFailed {
                    id: request.id,
                    error,
                });
            }
        }
    }
}

fn encode_save_messages(
    mut requests: MessageReader<EncodeImageBytes>,
    mut encoded: MessageWriter<ImageBytesEncoded>,
    mut failed: MessageWriter<ImageIoFailed>,
) {
    for request in requests.read() {
        match encode_host_image(&request.image, request.encoding) {
            Ok(bytes) => {
                encoded.write(ImageBytesEncoded {
                    id: request.id,
                    bytes,
                    encoding: request.encoding,
                });
            }
            Err(error) => {
                failed.write(ImageIoFailed {
                    id: request.id,
                    error,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use burn_image::{ColorSpace, HostImage, PixelBuffer, PixelFormat};

    use super::*;

    #[test]
    fn png_roundtrip_preserves_rgba_pixels_correctness() {
        let dimensions = Dimensions::new(2, 1).unwrap();
        let host = HostImage::Pixels(
            PixelBuffer::new(
                dimensions,
                PixelFormat::Rgba8,
                ColorSpace::Srgb,
                vec![1, 2, 3, 4, 250, 251, 252, 253],
            )
            .unwrap(),
        );
        let encoded = encode_host_image(&host, ImageEncoding::Png).unwrap();
        let decoded = decode_input_image(&encoded, Some(ImageEncoding::Png)).unwrap();
        let InputImage::Pixels(decoded) = decoded else {
            panic!("decoder should return pixels");
        };
        assert_eq!(decoded.dimensions(), dimensions);
        assert_eq!(decoded.bytes(), &[1, 2, 3, 4, 250, 251, 252, 253]);
    }

    #[test]
    fn linear_float_pixels_are_encoded_to_srgb_correctness() {
        let dimensions = Dimensions::new(1, 1).unwrap();
        let values = [0.5f32, 0.5, 0.5, 1.0];
        let bytes = values.into_iter().flat_map(f32::to_le_bytes).collect();
        let pixels = PixelBuffer::new(
            dimensions,
            PixelFormat::Rgba32Float,
            ColorSpace::LinearSrgb,
            bytes,
        )
        .unwrap();
        let (_, rgba) = pixel_buffer_rgba8(&pixels).unwrap();
        assert_eq!(rgba, [188, 188, 188, 255]);
    }

    #[test]
    fn browser_download_metadata_is_safe_correctness() {
        assert_eq!(
            download_metadata("boogu_result_01", ImageEncoding::Png).unwrap(),
            ("boogu_result_01.png".to_string(), "image/png")
        );
        assert!(download_metadata("../result", ImageEncoding::Png).is_err());
    }
}
