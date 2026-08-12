use burn_image::{
    ArtifactFile, ArtifactPath, ArtifactReadRequest, ArtifactSource, ArtifactVerifier, ByteRange,
    IntegrityPolicy, RemoteBaseUrl, VerifiedArtifact,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
use burn_boogu::{BooguError, artifacts::AsyncStageShardReader};
#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
use burn_image::CancellationToken;

/// Hard cap on a single browser-delivered chunk. A model may choose a smaller
/// limit according to Wasm memory and WebGPU upload measurements.
pub const MAX_BROWSER_CHUNK_BYTES: u64 = 16 * 1024 * 1024;
pub const DEFAULT_BROWSER_CHUNK_BYTES: u64 = 4 * 1024 * 1024;
/// Hard ceiling for one semantic Burnpack object retained in Wasm linear memory.
pub const MAX_BROWSER_STAGE_BYTES: u64 = 256 * 1024 * 1024;
/// Bootstrap metadata must remain small enough to fetch before the sealed manifest is known.
pub const MAX_BROWSER_MANIFEST_BYTES: u64 = 4 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactStreamConfig {
    max_chunk_bytes: u64,
}

impl Default for ArtifactStreamConfig {
    fn default() -> Self {
        Self {
            max_chunk_bytes: DEFAULT_BROWSER_CHUNK_BYTES,
        }
    }
}

impl ArtifactStreamConfig {
    pub fn new(max_chunk_bytes: u64) -> Result<Self, ArtifactStreamError> {
        if max_chunk_bytes == 0 || max_chunk_bytes > MAX_BROWSER_CHUNK_BYTES {
            return Err(ArtifactStreamError::InvalidChunkLimit {
                requested: max_chunk_bytes,
                maximum: MAX_BROWSER_CHUNK_BYTES,
            });
        }
        Ok(Self { max_chunk_bytes })
    }

    pub fn max_chunk_bytes(self) -> u64 {
        self.max_chunk_bytes
    }
}

/// One owned transport response. Its bytes are borrowed by the loader and are
/// never retained there after [`StreamingArtifactLoader::push_chunk`] returns.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactChunk {
    pub path: ArtifactPath,
    pub range: ByteRange,
    pub bytes: Vec<u8>,
}

/// Browser fetch description with an exact HTTP Range header.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BrowserRangeRequest {
    pub path: ArtifactPath,
    pub url: String,
    pub range: ByteRange,
    pub range_header: String,
}

/// Return the zero-based shard position used by `ProgressEvent::ArtifactStarted`.
///
/// Unsharded semantic objects are represented as the sole object in a one-object group. Keeping
/// this conversion beside the transport contract prevents browser adapters from accidentally
/// reporting one-based indices that UI formatters increment a second time.
#[cfg(any(test, all(target_arch = "wasm32", feature = "boogu-web")))]
pub(crate) fn artifact_progress_position(file: &ArtifactFile) -> (u32, u32) {
    file.shard
        .as_ref()
        .map(|shard| (shard.index, shard.count))
        .unwrap_or((0, 1))
}

impl BrowserRangeRequest {
    pub fn new(
        base_url: &RemoteBaseUrl,
        request: &ArtifactReadRequest,
    ) -> Result<Self, ArtifactStreamError> {
        let range = request
            .range
            .ok_or(ArtifactStreamError::UnboundedBrowserRequest)?;
        Ok(Self {
            path: request.path.clone(),
            url: base_url.resolve(&request.path),
            range,
            range_header: range.http_range_header(),
        })
    }

    pub fn from_source(
        source: &ArtifactSource,
        request: &ArtifactReadRequest,
    ) -> Result<Self, ArtifactStreamError> {
        match source {
            ArtifactSource::Remote { base_url } => Self::new(base_url, request),
            ArtifactSource::LocalDirectory { .. } => Err(ArtifactStreamError::LocalBrowserSource),
        }
    }
}

/// A sink receives tentative bytes and must expose them only after `commit`.
/// `abort` must discard all tentative state, including device allocations.
pub trait TransactionalArtifactSink {
    fn begin(&mut self, file: &ArtifactFile) -> Result<(), String>;
    fn write(&mut self, range: ByteRange, bytes: &[u8]) -> Result<(), String>;
    fn commit(&mut self, verified: &VerifiedArtifact) -> Result<(), String>;
    fn abort(&mut self);
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArtifactStreamProgress {
    NeedMore {
        verified_bytes: u64,
        total_bytes: u64,
    },
    Verified(VerifiedArtifact),
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ArtifactStreamError {
    #[error("artifact chunk limit {requested} is invalid; maximum is {maximum}")]
    InvalidChunkLimit { requested: u64, maximum: u64 },
    #[error("browser artifact requests must always contain a bounded byte range")]
    UnboundedBrowserRequest,
    #[error("a browser range request requires a remote artifact source")]
    LocalBrowserSource,
    #[error("browser fetch is unavailable because Window is missing")]
    BrowserWindowUnavailable,
    #[error("browser fetch request failed: {0}")]
    BrowserRequest(String),
    #[error("browser range fetch returned HTTP {status} for {url}; expected 206")]
    BrowserHttpStatus { status: u16, url: String },
    #[error("browser range response has Content-Range {actual:?}; expected {expected}")]
    BrowserContentRange {
        expected: String,
        actual: Option<String>,
    },
    #[error("browser range response contains {actual} bytes; expected {expected}")]
    BrowserResponseSize { expected: u64, actual: u64 },
    #[error("browser Content-Range header is malformed: {0:?}")]
    BrowserMalformedContentRange(Option<String>),
    #[error("browser file contains {actual} bytes, above the bounded maximum {maximum}")]
    BrowserFileTooLarge { actual: u64, maximum: u64 },
    #[error("artifact stream has already completed or failed")]
    StreamClosed,
    #[error("expected artifact '{expected}', got '{actual}'")]
    UnexpectedPath {
        expected: ArtifactPath,
        actual: ArtifactPath,
    },
    #[error("artifact chunk contains {actual} bytes, above the configured maximum {maximum}")]
    ChunkTooLarge { actual: u64, maximum: u64 },
    #[error("artifact integrity check failed: {0}")]
    Integrity(#[from] burn_image::IntegrityError),
    #[error("artifact sink failed during {operation}: {message}")]
    Sink {
        operation: &'static str,
        message: String,
    },
}

/// Execute one exact HTTP range request in a browser.
///
/// This is the real transport half of browser artifact streaming: it requires
/// HTTP 206, validates the exposed `Content-Range`, rejects responses above the
/// hard chunk cap, and returns only the requested bytes. The caller should feed
/// the chunk directly into [`StreamingArtifactLoader::push_chunk`] and release
/// it before fetching the next range.
#[cfg(target_arch = "wasm32")]
pub async fn fetch_browser_range(
    request: &BrowserRangeRequest,
) -> Result<ArtifactChunk, ArtifactStreamError> {
    use js_sys::Uint8Array;
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{Headers, Request, RequestInit, Response};

    if request.range.length() > MAX_BROWSER_CHUNK_BYTES {
        return Err(ArtifactStreamError::ChunkTooLarge {
            actual: request.range.length(),
            maximum: MAX_BROWSER_CHUNK_BYTES,
        });
    }

    let headers = Headers::new().map_err(browser_request_error)?;
    headers
        .set("Range", &request.range_header)
        .map_err(browser_request_error)?;
    let init = RequestInit::new();
    init.set_method("GET");
    init.set_headers_headers(&headers);
    let fetch_request =
        Request::new_with_str_and_init(&request.url, &init).map_err(browser_request_error)?;
    let window = web_sys::window().ok_or(ArtifactStreamError::BrowserWindowUnavailable)?;
    let response = JsFuture::from(window.fetch_with_request(&fetch_request))
        .await
        .map_err(browser_request_error)?
        .dyn_into::<Response>()
        .map_err(browser_request_error)?;
    if response.status() != 206 {
        return Err(ArtifactStreamError::BrowserHttpStatus {
            status: response.status(),
            url: request.url.clone(),
        });
    }

    let content_range = response
        .headers()
        .get("Content-Range")
        .map_err(browser_request_error)?;
    validate_content_range(request.range, content_range.as_deref())?;
    let buffer = JsFuture::from(response.array_buffer().map_err(browser_request_error)?)
        .await
        .map_err(browser_request_error)?;
    let bytes = Uint8Array::new(&buffer).to_vec();
    let actual =
        u64::try_from(bytes.len()).map_err(|_| burn_image::IntegrityError::ByteCountOverflow)?;
    if actual != request.range.length() {
        return Err(ArtifactStreamError::BrowserResponseSize {
            expected: request.range.length(),
            actual,
        });
    }
    Ok(ArtifactChunk {
        path: request.path.clone(),
        range: request.range,
        bytes,
    })
}

/// Fetch an initially unknown-size browser file using only bounded HTTP range requests.
///
/// A one-byte probe obtains the authoritative total from `Content-Range`; the file is then read
/// in exact configured chunks. This is used only to bootstrap `manifest.json`. Every file named
/// by that manifest is subsequently read through its sealed size and SHA-256 contract.
#[cfg(target_arch = "wasm32")]
pub async fn fetch_browser_bounded_file(
    base_url: &RemoteBaseUrl,
    path: ArtifactPath,
    maximum_bytes: u64,
    config: ArtifactStreamConfig,
) -> Result<Vec<u8>, ArtifactStreamError> {
    let total = probe_browser_file_size(base_url, &path).await?;
    if total == 0 || total > maximum_bytes {
        return Err(ArtifactStreamError::BrowserFileTooLarge {
            actual: total,
            maximum: maximum_bytes,
        });
    }
    let capacity =
        usize::try_from(total).map_err(|_| ArtifactStreamError::BrowserFileTooLarge {
            actual: total,
            maximum: usize::MAX as u64,
        })?;
    let source = ArtifactSource::Remote {
        base_url: base_url.clone(),
    };
    let mut bytes = Vec::with_capacity(capacity);
    let mut offset = 0_u64;
    while offset < total {
        let length = (total - offset).min(config.max_chunk_bytes());
        let range = ByteRange::new(offset, length)
            .expect("bounded non-zero browser range cannot overflow the file size");
        let request = ArtifactReadRequest::ranged(path.clone(), range);
        let browser = BrowserRangeRequest::from_source(&source, &request)?;
        let chunk = fetch_browser_range(&browser).await?;
        bytes.extend_from_slice(&chunk.bytes);
        offset = range.end_exclusive();
    }
    Ok(bytes)
}

#[cfg(target_arch = "wasm32")]
async fn probe_browser_file_size(
    base_url: &RemoteBaseUrl,
    path: &ArtifactPath,
) -> Result<u64, ArtifactStreamError> {
    use js_sys::Uint8Array;
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{Headers, Request, RequestInit, Response};

    let url = base_url.resolve(path);
    let headers = Headers::new().map_err(browser_request_error)?;
    headers
        .set("Range", "bytes=0-0")
        .map_err(browser_request_error)?;
    let init = RequestInit::new();
    init.set_method("GET");
    init.set_headers_headers(&headers);
    let request = Request::new_with_str_and_init(&url, &init).map_err(browser_request_error)?;
    let window = web_sys::window().ok_or(ArtifactStreamError::BrowserWindowUnavailable)?;
    let response = JsFuture::from(window.fetch_with_request(&request))
        .await
        .map_err(browser_request_error)?
        .dyn_into::<Response>()
        .map_err(browser_request_error)?;
    if response.status() != 206 {
        return Err(ArtifactStreamError::BrowserHttpStatus {
            status: response.status(),
            url,
        });
    }
    let content_range = response
        .headers()
        .get("Content-Range")
        .map_err(browser_request_error)?;
    let probe_range = ByteRange::new(0, 1)
        .expect("the fixed one-byte browser size probe is always a valid range");
    validate_content_range(probe_range, content_range.as_deref())?;
    let (start, end, total) = parse_content_range(content_range.as_deref())?;
    debug_assert_eq!((start, end), (0, 0));
    let buffer = JsFuture::from(response.array_buffer().map_err(browser_request_error)?)
        .await
        .map_err(browser_request_error)?;
    if Uint8Array::new(&buffer).length() != 1 {
        return Err(ArtifactStreamError::BrowserResponseSize {
            expected: 1,
            actual: u64::from(Uint8Array::new(&buffer).length()),
        });
    }
    Ok(total)
}

#[cfg(target_arch = "wasm32")]
fn browser_request_error(value: wasm_bindgen::JsValue) -> ArtifactStreamError {
    ArtifactStreamError::BrowserRequest(value.as_string().unwrap_or_else(|| format!("{value:?}")))
}

#[cfg(any(target_arch = "wasm32", test))]
fn validate_content_range(
    range: ByteRange,
    actual: Option<&str>,
) -> Result<(), ArtifactStreamError> {
    let expected = format!(
        "bytes {}-{}/<total>",
        range.offset(),
        range.end_exclusive() - 1
    );
    let prefix = format!("bytes {}-{}/", range.offset(), range.end_exclusive() - 1);
    let valid = actual.is_some_and(|value| {
        value.strip_prefix(&prefix).is_some_and(|total| {
            total == "*"
                || total
                    .parse::<u64>()
                    .is_ok_and(|size| size >= range.end_exclusive())
        })
    });
    if !valid {
        return Err(ArtifactStreamError::BrowserContentRange {
            expected,
            actual: actual.map(str::to_owned),
        });
    }
    Ok(())
}

#[cfg(any(target_arch = "wasm32", test))]
fn parse_content_range(actual: Option<&str>) -> Result<(u64, u64, u64), ArtifactStreamError> {
    let raw = actual.ok_or(ArtifactStreamError::BrowserMalformedContentRange(None))?;
    let value = raw
        .strip_prefix("bytes ")
        .ok_or_else(|| ArtifactStreamError::BrowserMalformedContentRange(Some(raw.into())))?;
    let (interval, total) = value
        .split_once('/')
        .ok_or_else(|| ArtifactStreamError::BrowserMalformedContentRange(Some(raw.into())))?;
    let (start, end) = interval
        .split_once('-')
        .ok_or_else(|| ArtifactStreamError::BrowserMalformedContentRange(Some(raw.into())))?;
    let parsed = (
        start.parse::<u64>(),
        end.parse::<u64>(),
        total.parse::<u64>(),
    );
    let (Ok(start), Ok(end), Ok(total)) = parsed else {
        return Err(ArtifactStreamError::BrowserMalformedContentRange(Some(
            raw.into(),
        )));
    };
    if start > end || end >= total {
        return Err(ArtifactStreamError::BrowserMalformedContentRange(Some(
            raw.into(),
        )));
    }
    Ok((start, end, total))
}

/// Observable lifecycle of one digest-verified browser artifact object.
#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
#[derive(Clone, Debug)]
pub enum BrowserArtifactEvent {
    Started(ArtifactFile),
    Progress {
        path: ArtifactPath,
        loaded_bytes: u64,
        total_bytes: u64,
    },
    Verified(ArtifactPath),
}

/// Shared cancellation/progress control used by every clone of the browser shard reader.
#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
#[derive(Clone, Default)]
pub struct BrowserArtifactControl {
    inner: Arc<Mutex<BrowserArtifactControlState>>,
}

#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
#[derive(Default)]
struct BrowserArtifactControlState {
    cancellation: Option<CancellationToken>,
    events: VecDeque<BrowserArtifactEvent>,
    observer: Option<Arc<dyn Fn(BrowserArtifactEvent) + Send + Sync>>,
}

#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
impl BrowserArtifactControl {
    const MAX_PENDING_EVENTS: usize = 128;

    pub fn set_cancellation(&self, cancellation: Option<CancellationToken>) {
        self.inner
            .lock()
            .expect("browser artifact control mutex poisoned")
            .cancellation = cancellation;
    }

    pub fn pop_event(&self) -> Option<BrowserArtifactEvent> {
        self.inner
            .lock()
            .expect("browser artifact control mutex poisoned")
            .events
            .pop_front()
    }

    pub fn clear_events(&self) {
        self.inner
            .lock()
            .expect("browser artifact control mutex poisoned")
            .events
            .clear();
    }

    pub fn set_observer(&self, observer: Option<Arc<dyn Fn(BrowserArtifactEvent) + Send + Sync>>) {
        self.inner
            .lock()
            .expect("browser artifact control mutex poisoned")
            .observer = observer;
    }

    fn check_cancelled(&self) -> Result<(), BooguError> {
        let cancelled = self
            .inner
            .lock()
            .expect("browser artifact control mutex poisoned")
            .cancellation
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled);
        if cancelled {
            Err(BooguError::Cancelled)
        } else {
            Ok(())
        }
    }

    fn push(&self, event: BrowserArtifactEvent) {
        let mut state = self
            .inner
            .lock()
            .expect("browser artifact control mutex poisoned");
        if let Some(observer) = state.observer.clone() {
            drop(state);
            observer(event);
            return;
        }
        if state.events.len() == Self::MAX_PENDING_EVENTS {
            state.events.pop_front();
        }
        state.events.push_back(event);
    }
}

/// HTTP Range reader for one sealed semantic Burnpack at a time.
///
/// Each response is capped by [`ArtifactStreamConfig`]; the aggregate is capped by
/// [`MAX_BROWSER_STAGE_BYTES`], verified against the manifest file SHA-256, and returned to the
/// model source for an independent verification before parsing.
#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
#[derive(Clone)]
pub struct BrowserStageShardReader {
    source: ArtifactSource,
    config: ArtifactStreamConfig,
    control: BrowserArtifactControl,
}

#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
impl BrowserStageShardReader {
    pub fn new(base_url: RemoteBaseUrl, config: ArtifactStreamConfig) -> Self {
        Self {
            source: ArtifactSource::Remote { base_url },
            config,
            control: BrowserArtifactControl::default(),
        }
    }

    pub fn control(&self) -> BrowserArtifactControl {
        self.control.clone()
    }

    pub async fn read_verified(&mut self, file: &ArtifactFile) -> Result<Vec<u8>, BooguError> {
        self.read_shard(file, MAX_BROWSER_STAGE_BYTES).await
    }
}

#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
impl AsyncStageShardReader for BrowserStageShardReader {
    async fn read_shard(
        &mut self,
        file: &ArtifactFile,
        max_bytes: u64,
    ) -> Result<Vec<u8>, BooguError> {
        let maximum = max_bytes.min(MAX_BROWSER_STAGE_BYTES);
        if file.size > maximum {
            return Err(BooguError::Artifact(format!(
                "browser stage {} is {} bytes, exceeding the bounded maximum {maximum}",
                file.path, file.size
            )));
        }
        let capacity = usize::try_from(file.size).map_err(|_| {
            BooguError::Artifact(format!(
                "browser stage {} does not fit Wasm address space",
                file.path
            ))
        })?;
        self.control.check_cancelled()?;
        self.control
            .push(BrowserArtifactEvent::Started(file.clone()));
        let mut bytes = Vec::with_capacity(capacity);
        let mut offset = 0_u64;
        while offset < file.size {
            self.control.check_cancelled()?;
            let length = (file.size - offset).min(self.config.max_chunk_bytes());
            let range = ByteRange::new(offset, length).map_err(|error| {
                BooguError::Artifact(format!("invalid browser range for {}: {error}", file.path))
            })?;
            let request = ArtifactReadRequest::ranged(file.path.clone(), range);
            let browser = BrowserRangeRequest::from_source(&self.source, &request)
                .map_err(|error| BooguError::Artifact(error.to_string()))?;
            let chunk = fetch_browser_range(&browser)
                .await
                .map_err(|error| BooguError::Artifact(error.to_string()))?;
            bytes.extend_from_slice(&chunk.bytes);
            offset = range.end_exclusive();
            self.control.push(BrowserArtifactEvent::Progress {
                path: file.path.clone(),
                loaded_bytes: offset,
                total_bytes: file.size,
            });
        }
        ArtifactVerifier::verify_bytes(file, &bytes, IntegrityPolicy::RequireSha256)
            .map_err(|error| BooguError::Artifact(error.to_string()))?;
        self.control
            .push(BrowserArtifactEvent::Verified(file.path.clone()));
        Ok(bytes)
    }
}

/// Sequential one-file loader. Production manifests shard heavy weights, so
/// callers construct one loader per shard and release it before requesting the
/// next shard. The loader itself retains zero payload bytes.
pub struct StreamingArtifactLoader<S: TransactionalArtifactSink> {
    file: ArtifactFile,
    verifier: Option<ArtifactVerifier>,
    sink: Option<S>,
    config: ArtifactStreamConfig,
    observed_size: u64,
    closed: bool,
}

impl<S: TransactionalArtifactSink> StreamingArtifactLoader<S> {
    pub fn new(
        file: ArtifactFile,
        policy: IntegrityPolicy,
        config: ArtifactStreamConfig,
        mut sink: S,
    ) -> Result<Self, ArtifactStreamError> {
        sink.begin(&file)
            .map_err(|message| ArtifactStreamError::Sink {
                operation: "begin",
                message,
            })?;
        Ok(Self {
            verifier: Some(ArtifactVerifier::new(&file, policy)),
            file,
            sink: Some(sink),
            config,
            observed_size: 0,
            closed: false,
        })
    }

    pub fn next_request(&self) -> Result<Option<ArtifactReadRequest>, ArtifactStreamError> {
        if self.closed {
            return Err(ArtifactStreamError::StreamClosed);
        }
        if self.observed_size == self.file.size {
            return Ok(None);
        }
        let remaining = self.file.size - self.observed_size;
        let length = remaining.min(self.config.max_chunk_bytes);
        let range = ByteRange::new(self.observed_size, length)
            .expect("non-zero bounded range cannot overflow file size");
        Ok(Some(ArtifactReadRequest::ranged(
            self.file.path.clone(),
            range,
        )))
    }

    pub fn push_chunk(
        &mut self,
        chunk: &ArtifactChunk,
    ) -> Result<ArtifactStreamProgress, ArtifactStreamError> {
        if self.closed {
            return Err(ArtifactStreamError::StreamClosed);
        }
        let result = self.push_chunk_inner(chunk);
        if result.is_err() {
            self.closed = true;
            self.sink
                .as_mut()
                .expect("sink remains present until extraction")
                .abort();
        }
        result
    }

    fn push_chunk_inner(
        &mut self,
        chunk: &ArtifactChunk,
    ) -> Result<ArtifactStreamProgress, ArtifactStreamError> {
        if chunk.path != self.file.path {
            return Err(ArtifactStreamError::UnexpectedPath {
                expected: self.file.path.clone(),
                actual: chunk.path.clone(),
            });
        }
        let actual = u64::try_from(chunk.bytes.len())
            .map_err(|_| burn_image::IntegrityError::ByteCountOverflow)?;
        if actual > self.config.max_chunk_bytes {
            return Err(ArtifactStreamError::ChunkTooLarge {
                actual,
                maximum: self.config.max_chunk_bytes,
            });
        }
        let verifier = self
            .verifier
            .as_mut()
            .ok_or(ArtifactStreamError::StreamClosed)?;
        verifier.update_range(chunk.range, &chunk.bytes)?;
        self.sink
            .as_mut()
            .expect("sink remains present until extraction")
            .write(chunk.range, &chunk.bytes)
            .map_err(|message| ArtifactStreamError::Sink {
                operation: "write",
                message,
            })?;
        self.observed_size = verifier.observed_size();

        if self.observed_size < self.file.size {
            return Ok(ArtifactStreamProgress::NeedMore {
                verified_bytes: self.observed_size,
                total_bytes: self.file.size,
            });
        }

        let verified = self
            .verifier
            .take()
            .ok_or(ArtifactStreamError::StreamClosed)?
            .finish()?;
        self.sink
            .as_mut()
            .expect("sink remains present until extraction")
            .commit(&verified)
            .map_err(|message| ArtifactStreamError::Sink {
                operation: "commit",
                message,
            })?;
        self.closed = true;
        Ok(ArtifactStreamProgress::Verified(verified))
    }

    pub fn observed_size(&self) -> u64 {
        self.observed_size
    }

    /// The loader hashes borrowed chunks and never owns payload storage.
    pub const fn retained_payload_bytes(&self) -> usize {
        0
    }

    pub fn sink(&self) -> &S {
        self.sink
            .as_ref()
            .expect("sink remains present until extraction")
    }

    pub fn into_sink(mut self) -> S {
        self.closed = true;
        self.sink
            .take()
            .expect("sink remains present until extraction")
    }
}

impl<S: TransactionalArtifactSink> Drop for StreamingArtifactLoader<S> {
    fn drop(&mut self) {
        if !self.closed {
            if let Some(sink) = self.sink.as_mut() {
                sink.abort();
            }
            self.closed = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use burn_image::{ArtifactFileRole, ArtifactShard, Sha256Digest};

    use super::*;

    #[derive(Default)]
    struct RecordingSink {
        begun: bool,
        committed: bool,
        aborted: bool,
        max_write: usize,
        writes: usize,
    }

    impl TransactionalArtifactSink for RecordingSink {
        fn begin(&mut self, _file: &ArtifactFile) -> Result<(), String> {
            self.begun = true;
            Ok(())
        }

        fn write(&mut self, _range: ByteRange, bytes: &[u8]) -> Result<(), String> {
            self.max_write = self.max_write.max(bytes.len());
            self.writes += 1;
            Ok(())
        }

        fn commit(&mut self, _verified: &VerifiedArtifact) -> Result<(), String> {
            self.committed = true;
            Ok(())
        }

        fn abort(&mut self) {
            self.aborted = true;
        }
    }

    fn file(bytes: &[u8]) -> ArtifactFile {
        ArtifactFile {
            path: ArtifactPath::new("weights/part-000.bpk").unwrap(),
            size: bytes.len() as u64,
            sha256: Sha256Digest::calculate(bytes),
            role: ArtifactFileRole::Weights,
            component: None,
            shard: None,
        }
    }

    #[test]
    fn bounded_ranges_stream_and_commit_correctness() {
        let bytes = b"abcdefghij";
        let mut loader = StreamingArtifactLoader::new(
            file(bytes),
            IntegrityPolicy::RequireSha256,
            ArtifactStreamConfig::new(4).unwrap(),
            RecordingSink::default(),
        )
        .unwrap();
        while let Some(request) = loader.next_request().unwrap() {
            let range = request.range.unwrap();
            let start = range.offset() as usize;
            let end = range.end_exclusive() as usize;
            let progress = loader
                .push_chunk(&ArtifactChunk {
                    path: request.path,
                    range,
                    bytes: bytes[start..end].to_vec(),
                })
                .unwrap();
            if matches!(progress, ArtifactStreamProgress::Verified(_)) {
                break;
            }
        }
        assert_eq!(loader.retained_payload_bytes(), 0);
        assert_eq!(loader.sink().max_write, 4);
        assert_eq!(loader.sink().writes, 3);
        assert!(loader.sink().committed);
        assert!(!loader.sink().aborted);
    }

    #[test]
    fn out_of_order_range_aborts_transaction_correctness() {
        let bytes = b"abcdefgh";
        let mut loader = StreamingArtifactLoader::new(
            file(bytes),
            IntegrityPolicy::RequireSha256,
            ArtifactStreamConfig::new(4).unwrap(),
            RecordingSink::default(),
        )
        .unwrap();
        let error = loader
            .push_chunk(&ArtifactChunk {
                path: ArtifactPath::new("weights/part-000.bpk").unwrap(),
                range: ByteRange::new(4, 4).unwrap(),
                bytes: bytes[4..].to_vec(),
            })
            .unwrap_err();
        assert!(matches!(error, ArtifactStreamError::Integrity(_)));
        assert!(loader.sink().aborted);
        assert!(!loader.sink().committed);
    }

    #[test]
    fn browser_request_is_exact_and_bounded_correctness() {
        let source = ArtifactSource::Remote {
            base_url: RemoteBaseUrl::new("https://cdn.example/models").unwrap(),
        };
        let request = ArtifactReadRequest::ranged(
            ArtifactPath::new("weights/a.bpk").unwrap(),
            ByteRange::new(8, 4).unwrap(),
        );
        let browser = BrowserRangeRequest::from_source(&source, &request).unwrap();
        assert_eq!(browser.url, "https://cdn.example/models/weights/a.bpk");
        assert_eq!(browser.range_header, "bytes=8-11");
    }

    #[test]
    fn content_range_must_match_requested_interval_correctness() {
        let range = ByteRange::new(8, 4).unwrap();
        validate_content_range(range, Some("bytes 8-11/32")).unwrap();
        validate_content_range(range, Some("bytes 8-11/*")).unwrap();
        assert_eq!(
            parse_content_range(Some("bytes 8-11/32")).unwrap(),
            (8, 11, 32)
        );
        assert!(parse_content_range(Some("bytes 8-11/*")).is_err());
        for actual in [None, Some("bytes 7-10/32"), Some("bytes 8-11/10")] {
            assert!(matches!(
                validate_content_range(range, actual),
                Err(ArtifactStreamError::BrowserContentRange { .. })
            ));
        }
    }

    #[test]
    fn artifact_progress_uses_zero_based_manifest_shards_correctness() {
        let mut artifact = file(b"shard");
        assert_eq!(artifact_progress_position(&artifact), (0, 1));

        artifact.shard = Some(ArtifactShard {
            index: 2,
            count: 4,
            chain_sha256: None,
        });
        assert_eq!(artifact_progress_position(&artifact), (2, 4));
    }
}
