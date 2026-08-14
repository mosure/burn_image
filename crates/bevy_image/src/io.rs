use std::{io::Cursor, sync::Arc};

use bevy::prelude::*;
use burn_image::{
    ColorSpace, Dimensions, HostImage, ImageEncoding, InputImage, PixelBuffer, PixelFormat,
};
use half::f16;
use image::{DynamicImage, ImageFormat, ImageReader, Limits, RgbaImage};

/// Maximum decoded input edge accepted by the interactive frontend.
///
/// The released models use substantially smaller bounded inputs. This extra
/// headroom keeps the generic frontend useful without allowing a compressed
/// image to request an unbounded decoder allocation.
pub const MAX_INPUT_IMAGE_EDGE: u32 = 4_096;
/// Maximum tightly packed RGBA8 image retained by the frontend.
pub const MAX_DECODED_RGBA_BYTES: u64 = 64 * 1024 * 1024;
const MAX_IMAGE_DECODER_ALLOCATION_BYTES: u64 = 96 * 1024 * 1024;

use crate::{FrontendError, FrontendErrorKind};

/// Converts a canonical host image into tightly packed RGBA8 pixels for Bevy
/// display or conventional image encoders.
pub fn host_image_rgba8(image: &HostImage) -> Result<(Dimensions, Vec<u8>), FrontendError> {
    match image {
        HostImage::Pixels(pixels) => pixel_buffer_rgba8(pixels),
        HostImage::Encoded(encoded) => {
            let decoded =
                decode_dynamic_image(encoded.bytes(), encoding_to_format(encoded.encoding()))?;
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
    validate_decoded_dimensions(decoded.width(), decoded.height())?;
    let rgba = decoded.to_rgba8();
    if rgba.as_raw().len() as u64 > MAX_DECODED_RGBA_BYTES {
        return Err(FrontendError::new(
            FrontendErrorKind::UnsupportedImage,
            format!(
                "decoded image exceeds the {} MiB RGBA limit",
                MAX_DECODED_RGBA_BYTES / (1024 * 1024)
            ),
        ));
    }
    let dimensions = Dimensions::new(rgba.width(), rgba.height())?;
    Ok((dimensions, rgba.into_raw()))
}

fn validate_decoded_dimensions(width: u32, height: u32) -> Result<(), FrontendError> {
    let rgba_bytes = u64::from(width)
        .saturating_mul(u64::from(height))
        .saturating_mul(4);
    if width > MAX_INPUT_IMAGE_EDGE
        || height > MAX_INPUT_IMAGE_EDGE
        || rgba_bytes > MAX_DECODED_RGBA_BYTES
    {
        return Err(FrontendError::new(
            FrontendErrorKind::UnsupportedImage,
            format!(
                "image dimensions {width}x{height} exceed the interactive limit (maximum edge {}, maximum RGBA {} MiB)",
                MAX_INPUT_IMAGE_EDGE,
                MAX_DECODED_RGBA_BYTES / (1024 * 1024)
            ),
        ));
    }
    Ok(())
}

fn decode_dynamic_image(
    bytes: &[u8],
    encoding: Option<ImageFormat>,
) -> Result<DynamicImage, FrontendError> {
    let cursor = Cursor::new(bytes);
    let mut reader = match encoding {
        Some(format) => ImageReader::with_format(cursor, format),
        None => ImageReader::new(cursor)
            .with_guessed_format()
            .map_err(|error| {
                FrontendError::new(FrontendErrorKind::ImageDecode, error.to_string())
            })?,
    };
    let mut limits = Limits::default();
    limits.max_image_width = Some(MAX_INPUT_IMAGE_EDGE);
    limits.max_image_height = Some(MAX_INPUT_IMAGE_EDGE);
    limits.max_alloc = Some(MAX_IMAGE_DECODER_ALLOCATION_BYTES);
    reader.limits(limits);
    let decoded = reader.decode()?;
    validate_decoded_dimensions(decoded.width(), decoded.height())?;
    Ok(decoded)
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
    let decoded = decode_dynamic_image(bytes, encoding.and_then(encoding_to_format))?;
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
    use std::{fs::File, io::Read};

    let path = path.as_ref();
    let file = File::open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(FrontendError::invalid_request(format!(
            "input image is not a regular file: {}",
            path.display()
        )));
    }
    if metadata.len() == 0 || metadata.len() > MAX_DECODED_RGBA_BYTES {
        return Err(FrontendError::invalid_request(format!(
            "input image must contain 1..={MAX_DECODED_RGBA_BYTES} bytes, found {}",
            metadata.len()
        )));
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(MAX_DECODED_RGBA_BYTES.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_DECODED_RGBA_BYTES {
        return Err(FrontendError::invalid_request(format!(
            "input image exceeds the {MAX_DECODED_RGBA_BYTES}-byte limit"
        )));
    }
    decode_input_image(&bytes, None)
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageIoId(pub u64);

#[derive(Message, Clone, Debug)]
pub struct LoadImageBytes {
    pub id: ImageIoId,
    pub bytes: Arc<[u8]>,
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
            .add_systems(Update, (encode_save_messages, prepare_download_messages));

        #[cfg(all(feature = "app", feature = "native-io", not(target_arch = "wasm32")))]
        app.init_resource::<native_decode::NativeImageDecodeChannel>()
            .init_resource::<native_decode::NativeImageDecodeState>()
            .add_systems(Update, native_decode::drive_native_image_decode);

        #[cfg(not(all(feature = "app", feature = "native-io", not(target_arch = "wasm32"))))]
        app.add_systems(Update, decode_load_messages);
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

#[cfg(not(all(feature = "app", feature = "native-io", not(target_arch = "wasm32"))))]
fn decode_load_messages(
    mut requests: MessageReader<LoadImageBytes>,
    mut loaded: MessageWriter<ImageBytesLoaded>,
    mut failed: MessageWriter<ImageIoFailed>,
) {
    // Browser and source-only builds cannot rely on a native worker pool. Still
    // collapse a burst of picker/drop events so only the newest payload for an
    // I/O id is decoded on the main thread.
    let mut newest = Vec::<LoadImageBytes>::new();
    for request in requests.read() {
        if let Some(pending) = newest.iter_mut().find(|pending| pending.id == request.id) {
            *pending = request.clone();
        } else {
            newest.push(request.clone());
        }
    }

    for request in newest {
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

#[cfg(all(feature = "app", feature = "native-io", not(target_arch = "wasm32")))]
mod native_decode {
    use std::{
        collections::{HashMap, VecDeque},
        sync::{Mutex, mpsc},
    };

    use bevy::{
        tasks::AsyncComputeTaskPool,
        winit::{EventLoopProxy, EventLoopProxyWrapper, WinitUserEvent},
    };

    use super::*;

    /// The frontend has one interactive image target today, but a small fixed
    /// queue keeps the generic message API useful to embedders without allowing
    /// a burst of unique ids to retain unbounded compressed payloads.
    const MAX_PENDING_IMAGE_DECODES: usize = 8;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct DecodeTicket {
        id: ImageIoId,
        revision: u64,
    }

    #[derive(Debug)]
    struct PendingDecode {
        ticket: DecodeTicket,
        request: LoadImageBytes,
    }

    struct CompletedDecode {
        ticket: DecodeTicket,
        result: Result<InputImage, FrontendError>,
    }

    #[derive(Resource)]
    pub(super) struct NativeImageDecodeChannel {
        sender: mpsc::SyncSender<CompletedDecode>,
        receiver: Mutex<mpsc::Receiver<CompletedDecode>>,
    }

    impl Default for NativeImageDecodeChannel {
        fn default() -> Self {
            // There can only be one active task, and the next task is not
            // launched until its predecessor has been received.
            let (sender, receiver) = mpsc::sync_channel(1);
            Self {
                sender,
                receiver: Mutex::new(receiver),
            }
        }
    }

    #[derive(Resource, Default)]
    pub(super) struct NativeImageDecodeState {
        next_revision: u64,
        active: Option<DecodeTicket>,
        pending: VecDeque<PendingDecode>,
        latest_revision: HashMap<ImageIoId, u64>,
    }

    impl NativeImageDecodeState {
        fn enqueue(&mut self, request: LoadImageBytes) -> Option<PendingDecode> {
            self.next_revision = self.next_revision.wrapping_add(1);
            let ticket = DecodeTicket {
                id: request.id,
                revision: self.next_revision,
            };
            self.latest_revision.insert(ticket.id, ticket.revision);

            if let Some(index) = self
                .pending
                .iter()
                .position(|pending| pending.ticket.id == ticket.id)
            {
                self.pending.remove(index);
            }

            let displaced = if self.pending.len() == MAX_PENDING_IMAGE_DECODES {
                self.pending.pop_front()
            } else {
                None
            };
            if let Some(displaced) = displaced.as_ref()
                && self.latest_revision.get(&displaced.ticket.id)
                    == Some(&displaced.ticket.revision)
            {
                self.latest_revision.remove(&displaced.ticket.id);
            }
            self.pending.push_back(PendingDecode { ticket, request });
            displaced
        }

        fn begin_next(&mut self) -> Option<PendingDecode> {
            if self.active.is_some() {
                return None;
            }
            let pending = self.pending.pop_front()?;
            self.active = Some(pending.ticket);
            Some(pending)
        }

        fn finish(&mut self, ticket: DecodeTicket) -> bool {
            if self.active != Some(ticket) {
                return false;
            }
            self.active = None;
            let is_latest = self.latest_revision.get(&ticket.id) == Some(&ticket.revision);
            if is_latest {
                self.latest_revision.remove(&ticket.id);
            }
            is_latest
        }
    }

    pub(super) fn drive_native_image_decode(
        mut requests: MessageReader<LoadImageBytes>,
        mut loaded: MessageWriter<ImageBytesLoaded>,
        mut failed: MessageWriter<ImageIoFailed>,
        mut state: ResMut<NativeImageDecodeState>,
        channel: Res<NativeImageDecodeChannel>,
        event_loop_proxy: Option<Res<EventLoopProxyWrapper>>,
    ) {
        // Record new revisions before accepting a worker result. If a newer
        // selection and an older completion arrive in the same frame, the old
        // completion is therefore discarded rather than briefly displayed.
        for request in requests.read() {
            if let Some(displaced) = state.enqueue(request.clone()) {
                failed.write(ImageIoFailed {
                    id: displaced.ticket.id,
                    error: FrontendError::invalid_request(
                        "image decode queue is full; the pending request was superseded",
                    ),
                });
            }
        }

        let completed = channel
            .receiver
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .try_recv()
            .ok();
        if let Some(completed) = completed
            && state.finish(completed.ticket)
        {
            match completed.result {
                Ok(image) => {
                    loaded.write(ImageBytesLoaded {
                        id: completed.ticket.id,
                        image,
                    });
                }
                Err(error) => {
                    failed.write(ImageIoFailed {
                        id: completed.ticket.id,
                        error,
                    });
                }
            }
        }

        let Some(pending) = state.begin_next() else {
            return;
        };
        let sender = channel.sender.clone();
        let event_loop_proxy = event_loop_proxy
            .as_ref()
            .map(|proxy| EventLoopProxy::clone(&**proxy));
        AsyncComputeTaskPool::get()
            .spawn(async move {
                let result = decode_input_image(&pending.request.bytes, pending.request.encoding);
                let _ = sender.send(CompletedDecode {
                    ticket: pending.ticket,
                    result,
                });
                if let Some(event_loop_proxy) = event_loop_proxy {
                    let _ = event_loop_proxy.send_event(WinitUserEvent::WakeUp);
                }
            })
            .detach();
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn request(id: u64, marker: u8) -> LoadImageBytes {
            LoadImageBytes {
                id: ImageIoId(id),
                bytes: Arc::from([marker]),
                encoding: None,
            }
        }

        #[test]
        fn same_id_pending_requests_coalesce_to_newest_correctness() {
            let mut state = NativeImageDecodeState::default();
            state.enqueue(request(7, 1));
            state.enqueue(request(7, 2));

            assert_eq!(state.pending.len(), 1);
            assert_eq!(state.pending.front().unwrap().request.bytes.as_ref(), &[2]);
        }

        #[test]
        fn stale_active_completion_is_not_published_correctness() {
            let mut state = NativeImageDecodeState::default();
            state.enqueue(request(7, 1));
            let first = state.begin_next().unwrap().ticket;
            state.enqueue(request(7, 2));

            assert!(!state.finish(first));
            let second = state.begin_next().unwrap().ticket;
            assert!(state.finish(second));
        }

        #[test]
        fn native_decode_queue_and_worker_are_bounded_correctness() {
            let mut state = NativeImageDecodeState::default();
            for id in 0..MAX_PENDING_IMAGE_DECODES as u64 {
                assert!(state.enqueue(request(id, id as u8)).is_none());
            }
            let displaced = state
                .enqueue(request(MAX_PENDING_IMAGE_DECODES as u64, 9))
                .expect("oldest pending request should be displaced");

            assert_eq!(state.pending.len(), MAX_PENDING_IMAGE_DECODES);
            assert_eq!(displaced.ticket.id, ImageIoId(0));
            assert!(state.begin_next().is_some());
            assert!(state.begin_next().is_none(), "only one task may be active");
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

    #[cfg(all(feature = "native-io", not(target_arch = "wasm32")))]
    #[test]
    fn native_file_loader_rejects_non_regular_and_oversized_inputs_correctness() {
        use std::io::{Seek, SeekFrom, Write};

        let directory = tempfile::tempdir().unwrap();
        let non_regular = load_image_file(directory.path())
            .expect_err("a directory must not be treated as an input image");
        assert!(non_regular.message.contains("regular file"));

        let oversized_path = directory.path().join("oversized.png");
        let mut oversized = std::fs::File::create(&oversized_path).unwrap();
        oversized
            .seek(SeekFrom::Start(MAX_DECODED_RGBA_BYTES))
            .unwrap();
        oversized.write_all(&[0]).unwrap();
        let error = load_image_file(&oversized_path)
            .expect_err("an input above the byte ceiling must fail before decoding");
        assert!(error.message.contains("1..="));
    }

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
    fn decoder_rejects_oversized_dimensions_before_rgba_materialization_correctness() {
        let oversized = DynamicImage::ImageRgba8(RgbaImage::new(MAX_INPUT_IMAGE_EDGE + 1, 1));
        let mut encoded = Cursor::new(Vec::new());
        oversized.write_to(&mut encoded, ImageFormat::Png).unwrap();

        let error = decode_input_image(&encoded.into_inner(), Some(ImageEncoding::Png))
            .expect_err("oversized input must fail closed");
        assert!(matches!(
            error.kind,
            FrontendErrorKind::ImageDecode | FrontendErrorKind::UnsupportedImage
        ));
        assert!(error.message.contains("image dimensions") || error.message.contains("limit"));
    }

    #[test]
    fn decoded_rgba_limit_matches_maximum_edge_contract_correctness() {
        assert_eq!(
            u64::from(MAX_INPUT_IMAGE_EDGE) * u64::from(MAX_INPUT_IMAGE_EDGE) * 4,
            MAX_DECODED_RGBA_BYTES
        );
        assert!(validate_decoded_dimensions(MAX_INPUT_IMAGE_EDGE, MAX_INPUT_IMAGE_EDGE).is_ok());
        assert!(validate_decoded_dimensions(MAX_INPUT_IMAGE_EDGE + 1, 1).is_err());
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
