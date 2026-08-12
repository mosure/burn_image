use std::fmt::Display;

use serde::{Deserialize, Serialize};

use crate::ValidationError;

/// Validated two-dimensional image extent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "DimensionsWire", into = "DimensionsWire")]
pub struct Dimensions {
    width: u32,
    height: u32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct DimensionsWire {
    width: u32,
    height: u32,
}

impl Dimensions {
    pub fn new(width: u32, height: u32) -> Result<Self, ValidationError> {
        if width == 0 || height == 0 {
            return Err(ValidationError::ZeroDimensions { width, height });
        }
        width
            .checked_mul(height)
            .ok_or(ValidationError::DimensionOverflow { width, height })?;
        Ok(Self { width, height })
    }

    pub fn width(self) -> u32 {
        self.width
    }

    pub fn height(self) -> u32 {
        self.height
    }

    pub fn area(self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    pub fn checked_byte_len(self, bytes_per_pixel: usize) -> Result<usize, ValidationError> {
        let area =
            usize::try_from(self.area()).map_err(|_| ValidationError::DimensionOverflow {
                width: self.width,
                height: self.height,
            })?;
        area.checked_mul(bytes_per_pixel)
            .ok_or(ValidationError::DimensionOverflow {
                width: self.width,
                height: self.height,
            })
    }
}

impl TryFrom<DimensionsWire> for Dimensions {
    type Error = ValidationError;

    fn try_from(value: DimensionsWire) -> Result<Self, Self::Error> {
        Self::new(value.width, value.height)
    }
}

impl From<Dimensions> for DimensionsWire {
    fn from(value: Dimensions) -> Self {
        Self {
            width: value.width,
            height: value.height,
        }
    }
}

impl Display for Dimensions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ColorSpace {
    Srgb,
    LinearSrgb,
    DisplayP3,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PixelFormat {
    L8,
    Rgb8,
    Rgba8,
    Rgba16Float,
    Rgba32Float,
}

impl PixelFormat {
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::L8 => 1,
            Self::Rgb8 => 3,
            Self::Rgba8 => 4,
            Self::Rgba16Float => 8,
            Self::Rgba32Float => 16,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageEncoding {
    Png,
    Jpeg,
    Webp,
    Avif,
    Other,
}

/// Decoded, tightly packed host pixels.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "PixelBufferWire", into = "PixelBufferWire")]
pub struct PixelBuffer {
    dimensions: Dimensions,
    format: PixelFormat,
    color_space: ColorSpace,
    bytes: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PixelBufferWire {
    dimensions: Dimensions,
    format: PixelFormat,
    color_space: ColorSpace,
    bytes: Vec<u8>,
}

impl PixelBuffer {
    pub fn new(
        dimensions: Dimensions,
        format: PixelFormat,
        color_space: ColorSpace,
        bytes: Vec<u8>,
    ) -> Result<Self, ValidationError> {
        let expected = dimensions.checked_byte_len(format.bytes_per_pixel())?;
        if bytes.len() != expected {
            return Err(ValidationError::PixelLengthMismatch {
                expected,
                actual: bytes.len(),
            });
        }
        Ok(Self {
            dimensions,
            format,
            color_space,
            bytes,
        })
    }

    pub fn dimensions(&self) -> Dimensions {
        self.dimensions
    }

    pub fn format(&self) -> PixelFormat {
        self.format
    }

    pub fn color_space(&self) -> ColorSpace {
        self.color_space
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

impl TryFrom<PixelBufferWire> for PixelBuffer {
    type Error = ValidationError;

    fn try_from(value: PixelBufferWire) -> Result<Self, Self::Error> {
        Self::new(
            value.dimensions,
            value.format,
            value.color_space,
            value.bytes,
        )
    }
}

impl From<PixelBuffer> for PixelBufferWire {
    fn from(value: PixelBuffer) -> Self {
        Self {
            dimensions: value.dimensions,
            format: value.format,
            color_space: value.color_space,
            bytes: value.bytes,
        }
    }
}

/// Encoded input or output bytes plus optional decoded dimensions.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "EncodedImageWire", into = "EncodedImageWire")]
pub struct EncodedImage {
    encoding: ImageEncoding,
    dimensions: Option<Dimensions>,
    bytes: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct EncodedImageWire {
    encoding: ImageEncoding,
    dimensions: Option<Dimensions>,
    bytes: Vec<u8>,
}

impl EncodedImage {
    pub fn new(
        encoding: ImageEncoding,
        dimensions: Option<Dimensions>,
        bytes: Vec<u8>,
    ) -> Result<Self, ValidationError> {
        if bytes.is_empty() {
            return Err(ValidationError::Empty {
                field: "encoded_image.bytes",
            });
        }
        Ok(Self {
            encoding,
            dimensions,
            bytes,
        })
    }

    pub fn encoding(&self) -> ImageEncoding {
        self.encoding
    }

    pub fn dimensions(&self) -> Option<Dimensions> {
        self.dimensions
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

impl TryFrom<EncodedImageWire> for EncodedImage {
    type Error = ValidationError;

    fn try_from(value: EncodedImageWire) -> Result<Self, Self::Error> {
        Self::new(value.encoding, value.dimensions, value.bytes)
    }
}

impl From<EncodedImage> for EncodedImageWire {
    fn from(value: EncodedImage) -> Self {
        Self {
            encoding: value.encoding,
            dimensions: value.dimensions,
            bytes: value.bytes,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InputImage {
    Pixels(PixelBuffer),
    Encoded(EncodedImage),
}

impl InputImage {
    pub fn dimensions(&self) -> Option<Dimensions> {
        match self {
            Self::Pixels(pixels) => Some(pixels.dimensions()),
            Self::Encoded(encoded) => encoded.dimensions(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaskSemantics {
    /// White pixels select content to replace.
    WhiteEdits,
    /// Black pixels select content to replace.
    BlackEdits,
}

/// Validated one-byte-per-pixel edit mask.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "InputMaskWire", into = "InputMaskWire")]
pub struct InputMask {
    dimensions: Dimensions,
    semantics: MaskSemantics,
    bytes: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct InputMaskWire {
    dimensions: Dimensions,
    semantics: MaskSemantics,
    bytes: Vec<u8>,
}

impl InputMask {
    pub fn new(
        dimensions: Dimensions,
        semantics: MaskSemantics,
        bytes: Vec<u8>,
    ) -> Result<Self, ValidationError> {
        let expected = dimensions.checked_byte_len(1)?;
        if bytes.len() != expected {
            return Err(ValidationError::PixelLengthMismatch {
                expected,
                actual: bytes.len(),
            });
        }
        Ok(Self {
            dimensions,
            semantics,
            bytes,
        })
    }

    pub fn dimensions(&self) -> Dimensions {
        self.dimensions
    }

    pub fn semantics(&self) -> MaskSemantics {
        self.semantics
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

impl TryFrom<InputMaskWire> for InputMask {
    type Error = ValidationError;

    fn try_from(value: InputMaskWire) -> Result<Self, Self::Error> {
        Self::new(value.dimensions, value.semantics, value.bytes)
    }
}

impl From<InputMask> for InputMaskWire {
    fn from(value: InputMask) -> Self {
        Self {
            dimensions: value.dimensions,
            semantics: value.semantics,
            bytes: value.bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ColorSpace, Dimensions, PixelBuffer, PixelFormat};

    #[test]
    fn pixel_buffer_checks_exact_byte_length_correctness() {
        let dimensions = Dimensions::new(2, 3).unwrap();
        assert!(
            PixelBuffer::new(
                dimensions,
                PixelFormat::Rgba8,
                ColorSpace::Srgb,
                vec![0; 24]
            )
            .is_ok()
        );
        assert!(
            PixelBuffer::new(
                dimensions,
                PixelFormat::Rgba8,
                ColorSpace::Srgb,
                vec![0; 23]
            )
            .is_err()
        );
    }

    #[test]
    fn invalid_dimensions_are_rejected_during_deserialization_correctness() {
        let json = r#"{"width":0,"height":512}"#;
        assert!(serde_json::from_str::<Dimensions>(json).is_err());
    }
}
