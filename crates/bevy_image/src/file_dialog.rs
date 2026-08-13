//! Cross-platform ECS contract for opening an input image.
//!
//! Native shells implement the contract with an asynchronous operating-system
//! file dialog. Browser shells retain their host-provided file input and can
//! publish the same selected/cancelled/failed messages when appropriate.

use std::sync::Arc;

use bevy::prelude::*;

use crate::{FrontendError, ImageIoId};

/// Requests an image file picker for a bounded byte payload.
#[derive(Message, Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpenImageFileDialog {
    pub id: ImageIoId,
    pub max_bytes: usize,
}

/// Requests a bounded asynchronous read for a file dropped on a native window.
#[cfg(all(feature = "native-io", not(target_arch = "wasm32")))]
#[derive(Message, Clone, Debug, PartialEq, Eq)]
pub struct ReadDroppedImageFile {
    pub id: ImageIoId,
    pub path: std::path::PathBuf,
    pub max_bytes: usize,
}

/// Reports a selected file and its bounded byte contents.
#[derive(Message, Clone, Debug, PartialEq, Eq)]
pub struct ImageFileDialogSelected {
    pub id: ImageIoId,
    /// A display-safe ASCII file name, never a native filesystem path.
    pub file_name: String,
    pub bytes: Arc<[u8]>,
}

/// Reports that the picker was closed without selecting a file.
#[derive(Message, Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageFileDialogCancelled {
    pub id: ImageIoId,
}

/// Reports a picker or native filesystem failure.
#[derive(Message, Clone, Debug, PartialEq, Eq)]
pub struct ImageFileDialogFailed {
    pub id: ImageIoId,
    pub error: FrontendError,
}

/// Observable picker state, used to disable duplicate open requests in UI.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ImageFileDialogState {
    pub is_open: bool,
}

/// Registers the image-dialog messages and the native picker implementation.
pub struct ImageFileDialogPlugin;

impl Plugin for ImageFileDialogPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ImageFileDialogState>()
            .add_message::<OpenImageFileDialog>()
            .add_message::<ImageFileDialogSelected>()
            .add_message::<ImageFileDialogCancelled>()
            .add_message::<ImageFileDialogFailed>();

        #[cfg(all(feature = "app", feature = "native-io", not(target_arch = "wasm32")))]
        app.add_message::<ReadDroppedImageFile>()
            .init_resource::<native::ImageFileDialogChannel>()
            .init_resource::<native::NativeDroppedImageReadState>()
            .add_systems(
                Update,
                (
                    native::open_image_dialog,
                    native::read_dropped_image,
                    native::poll_image_dialog,
                )
                    .chain(),
            );
    }
}

/// Convert a user-controlled file name to conservative ASCII suitable for
/// Bevy's default font and status text. The native path is intentionally not
/// exposed as a display label.
#[cfg(any(
    test,
    all(feature = "app", feature = "native-io", not(target_arch = "wasm32"))
))]
pub(crate) fn sanitize_image_file_name(value: &str) -> String {
    const MAX_DISPLAY_BYTES: usize = 96;

    let mut output = String::with_capacity(value.len().min(MAX_DISPLAY_BYTES));
    let mut pending_separator = false;
    for character in value.chars() {
        if output.len() >= MAX_DISPLAY_BYTES {
            break;
        }
        if character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '_') {
            if pending_separator && !output.is_empty() && output.len() < MAX_DISPLAY_BYTES {
                output.push('_');
            }
            pending_separator = false;
            output.push(character);
        } else {
            pending_separator = true;
        }
    }

    let output = output.trim_matches('_');
    if output.is_empty() {
        "image".to_owned()
    } else {
        output.to_owned()
    }
}

#[cfg(all(feature = "app", feature = "native-io", not(target_arch = "wasm32")))]
mod native {
    use std::{
        path::Path,
        sync::{Mutex, mpsc},
    };

    use bevy::{
        tasks::AsyncComputeTaskPool,
        winit::{EventLoopProxy, EventLoopProxyWrapper, WinitUserEvent},
    };

    use super::*;
    use crate::FrontendErrorKind;

    enum ImageFileDialogResult {
        Selected(ImageFileDialogSelected),
        Cancelled(ImageFileDialogCancelled),
        Failed(ImageFileDialogFailed),
    }

    impl ImageFileDialogResult {
        fn id(&self) -> ImageIoId {
            match self {
                Self::Selected(message) => message.id,
                Self::Cancelled(message) => message.id,
                Self::Failed(message) => message.id,
            }
        }
    }

    struct CompletedImageFileRead {
        result: ImageFileDialogResult,
        closes_dialog: bool,
    }

    #[derive(Resource)]
    pub(super) struct ImageFileDialogChannel {
        sender: mpsc::Sender<CompletedImageFileRead>,
        receiver: Mutex<mpsc::Receiver<CompletedImageFileRead>>,
    }

    /// A single native file read plus one latest pending selection keeps drag
    /// bursts from spawning unbounded filesystem tasks.
    #[derive(Resource, Default)]
    pub(super) struct NativeDroppedImageReadState {
        in_flight: bool,
        pending: Option<ReadDroppedImageFile>,
    }

    impl Default for ImageFileDialogChannel {
        fn default() -> Self {
            let (sender, receiver) = mpsc::channel();
            Self {
                sender,
                receiver: Mutex::new(receiver),
            }
        }
    }

    pub(super) fn open_image_dialog(
        mut requests: MessageReader<OpenImageFileDialog>,
        mut failed: MessageWriter<ImageFileDialogFailed>,
        mut state: ResMut<ImageFileDialogState>,
        channel: Res<ImageFileDialogChannel>,
        event_loop_proxy: Option<Res<EventLoopProxyWrapper>>,
    ) {
        for request in requests.read() {
            if state.is_open {
                failed.write(ImageFileDialogFailed {
                    id: request.id,
                    error: FrontendError::invalid_request("an image file dialog is already open"),
                });
                continue;
            }
            if request.max_bytes == 0 {
                failed.write(ImageFileDialogFailed {
                    id: request.id,
                    error: FrontendError::invalid_request(
                        "image file dialog byte limit must be greater than zero",
                    ),
                });
                continue;
            }

            state.is_open = true;
            let id = request.id;
            let max_bytes = request.max_bytes;
            let sender = channel.sender.clone();
            let event_loop_proxy = event_loop_proxy
                .as_ref()
                .map(|proxy| EventLoopProxy::clone(&**proxy));
            AsyncComputeTaskPool::get()
                .spawn(async move {
                    let selected = rfd::AsyncFileDialog::new()
                        .set_title("Open reference image")
                        .add_filter("Image", &["png", "jpg", "jpeg", "webp"])
                        .pick_file()
                        .await;
                    let result = match selected {
                        Some(file) => {
                            let file_name = sanitize_image_file_name(&file.file_name());
                            match read_bounded_file(file.path(), max_bytes, &file_name) {
                                Ok(bytes) => {
                                    ImageFileDialogResult::Selected(ImageFileDialogSelected {
                                        id,
                                        file_name,
                                        bytes: bytes.into(),
                                    })
                                }
                                Err(error) => {
                                    ImageFileDialogResult::Failed(ImageFileDialogFailed {
                                        id,
                                        error,
                                    })
                                }
                            }
                        }
                        None => ImageFileDialogResult::Cancelled(ImageFileDialogCancelled { id }),
                    };
                    let _ = sender.send(CompletedImageFileRead {
                        result,
                        closes_dialog: true,
                    });
                    if let Some(event_loop_proxy) = event_loop_proxy {
                        let _ = event_loop_proxy.send_event(WinitUserEvent::WakeUp);
                    }
                })
                .detach();
        }
    }

    pub(super) fn read_dropped_image(
        mut requests: MessageReader<ReadDroppedImageFile>,
        mut failed: MessageWriter<ImageFileDialogFailed>,
        mut state: ResMut<NativeDroppedImageReadState>,
        channel: Res<ImageFileDialogChannel>,
        event_loop_proxy: Option<Res<EventLoopProxyWrapper>>,
    ) {
        for request in requests.read() {
            if request.max_bytes == 0 {
                failed.write(ImageFileDialogFailed {
                    id: request.id,
                    error: FrontendError::invalid_request(
                        "dropped image byte limit must be greater than zero",
                    ),
                });
                continue;
            }
            state.pending = Some(request.clone());
        }

        if state.in_flight {
            return;
        }
        let Some(request) = state.pending.take() else {
            return;
        };
        state.in_flight = true;
        let id = request.id;
        let path = request.path;
        let max_bytes = request.max_bytes;
        let sender = channel.sender.clone();
        let event_loop_proxy = event_loop_proxy
            .as_ref()
            .map(|proxy| EventLoopProxy::clone(&**proxy));
        AsyncComputeTaskPool::get()
            .spawn(async move {
                let file_name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .map(sanitize_image_file_name)
                    .unwrap_or_else(|| "image".to_owned());
                let result = match read_bounded_file(&path, max_bytes, &file_name) {
                    Ok(bytes) => ImageFileDialogResult::Selected(ImageFileDialogSelected {
                        id,
                        file_name,
                        bytes: bytes.into(),
                    }),
                    Err(error) => {
                        ImageFileDialogResult::Failed(ImageFileDialogFailed { id, error })
                    }
                };
                let _ = sender.send(CompletedImageFileRead {
                    result,
                    closes_dialog: false,
                });
                if let Some(event_loop_proxy) = event_loop_proxy {
                    let _ = event_loop_proxy.send_event(WinitUserEvent::WakeUp);
                }
            })
            .detach();
    }

    pub(super) fn poll_image_dialog(
        channel: Res<ImageFileDialogChannel>,
        mut state: ResMut<ImageFileDialogState>,
        mut dropped_reads: ResMut<NativeDroppedImageReadState>,
        mut selected: MessageWriter<ImageFileDialogSelected>,
        mut cancelled: MessageWriter<ImageFileDialogCancelled>,
        mut failed: MessageWriter<ImageFileDialogFailed>,
    ) {
        let results = channel
            .receiver
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .try_iter()
            .collect::<Vec<_>>();
        for completed in results {
            if completed.closes_dialog {
                state.is_open = false;
            } else {
                dropped_reads.in_flight = false;
            }
            let superseded = !completed.closes_dialog
                && dropped_reads
                    .pending
                    .as_ref()
                    .is_some_and(|pending| pending.id == completed.result.id());
            if superseded {
                continue;
            }
            match completed.result {
                ImageFileDialogResult::Selected(message) => {
                    selected.write(message);
                }
                ImageFileDialogResult::Cancelled(message) => {
                    cancelled.write(message);
                }
                ImageFileDialogResult::Failed(message) => {
                    failed.write(message);
                }
            }
        }
    }

    fn read_bounded_file(
        path: &Path,
        max_bytes: usize,
        file_name: &str,
    ) -> Result<Vec<u8>, FrontendError> {
        let metadata = std::fs::metadata(path).map_err(|error| {
            native_io_error(format!(
                "could not inspect selected image {file_name}: {error}"
            ))
        })?;
        if !metadata.is_file() {
            return Err(FrontendError::invalid_request(format!(
                "selected image {file_name} is not a regular file"
            )));
        }
        if metadata.len() > max_bytes as u64 {
            return Err(too_large_error(file_name, metadata.len(), max_bytes));
        }

        let bytes = std::fs::read(path).map_err(|error| {
            native_io_error(format!(
                "could not read selected image {file_name}: {error}"
            ))
        })?;
        if bytes.is_empty() {
            return Err(FrontendError::invalid_request(format!(
                "selected image {file_name} is empty"
            )));
        }
        if bytes.len() > max_bytes {
            return Err(too_large_error(file_name, bytes.len() as u64, max_bytes));
        }
        Ok(bytes)
    }

    fn native_io_error(message: String) -> FrontendError {
        FrontendError::new(FrontendErrorKind::NativeIo, message)
    }

    fn too_large_error(file_name: &str, actual_bytes: u64, max_bytes: usize) -> FrontendError {
        FrontendError::invalid_request(format!(
            "selected image {file_name} is {actual_bytes} bytes; limit is {max_bytes} bytes"
        ))
    }

    #[cfg(test)]
    mod tests {
        use std::io::Write;

        use super::*;

        #[test]
        fn bounded_file_reader_accepts_payload_at_limit_correctness() {
            let mut file = tempfile::NamedTempFile::new().unwrap();
            file.write_all(b"image").unwrap();
            assert_eq!(
                read_bounded_file(file.path(), 5, "image.png").unwrap(),
                b"image"
            );
        }

        #[test]
        fn bounded_file_reader_rejects_empty_and_oversize_payloads_correctness() {
            let mut file = tempfile::NamedTempFile::new().unwrap();
            assert!(read_bounded_file(file.path(), 5, "empty.png").is_err());
            file.write_all(b"123456").unwrap();
            assert!(read_bounded_file(file.path(), 5, "large.png").is_err());
        }

        #[test]
        fn dropped_read_completion_does_not_close_active_dialog_correctness() {
            let channel = ImageFileDialogChannel::default();
            let sender = channel.sender.clone();
            let mut app = App::new();
            app.insert_resource(channel)
                .insert_resource(ImageFileDialogState { is_open: true })
                .init_resource::<NativeDroppedImageReadState>()
                .add_message::<ImageFileDialogSelected>()
                .add_message::<ImageFileDialogCancelled>()
                .add_message::<ImageFileDialogFailed>()
                .add_systems(Update, poll_image_dialog);

            sender
                .send(CompletedImageFileRead {
                    result: ImageFileDialogResult::Selected(ImageFileDialogSelected {
                        id: ImageIoId(7),
                        file_name: "reference.png".to_owned(),
                        bytes: Arc::from([1]),
                    }),
                    closes_dialog: false,
                })
                .unwrap();
            app.update();
            assert!(app.world().resource::<ImageFileDialogState>().is_open);

            sender
                .send(CompletedImageFileRead {
                    result: ImageFileDialogResult::Cancelled(ImageFileDialogCancelled {
                        id: ImageIoId(8),
                    }),
                    closes_dialog: true,
                })
                .unwrap();
            app.update();
            assert!(!app.world().resource::<ImageFileDialogState>().is_open);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::sanitize_image_file_name;

    #[test]
    fn display_file_name_is_ascii_and_bounded_correctness() {
        let sanitized = sanitize_image_file_name("caf\u{e9} / sketch \u{1f5bc}.png");
        assert_eq!(sanitized, "caf_sketch_.png");
        assert!(sanitized.is_ascii());

        let long = sanitize_image_file_name(&"a".repeat(256));
        assert_eq!(long.len(), 96);
    }

    #[test]
    fn display_file_name_has_safe_fallback_correctness() {
        assert_eq!(sanitize_image_file_name("\u{1f5bc}\u{fe0f}"), "image");
        assert_eq!(
            sanitize_image_file_name("reference-image.webp"),
            "reference-image.webp"
        );
    }
}
