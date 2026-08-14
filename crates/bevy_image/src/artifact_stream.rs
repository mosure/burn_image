use burn_image::{
    ArtifactBundleId, ArtifactFile, ArtifactPath, ArtifactReadRequest, ArtifactSource,
    ArtifactVerifier, ByteRange, IntegrityPolicy, RemoteBaseUrl, Sha256Digest, VerifiedArtifact,
};
#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
use burn_image::{
    ArtifactComponentId, ArtifactReadError, AsyncArtifactShardReader, VerifiedArtifactBytes,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[cfg(any(test, all(target_arch = "wasm32", feature = "boogu-web")))]
use std::collections::BTreeSet;
#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
use std::{
    collections::{BTreeMap, VecDeque},
    sync::{Arc, Mutex},
};

#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
use burn_boogu::{
    BooguError,
    artifacts::{AsyncStageShardRead, AsyncStageShardReader},
};
#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
use burn_image::CancellationToken;

/// Hard cap on a single browser-delivered chunk. A model may choose a smaller
/// limit according to Wasm memory and WebGPU upload measurements.
pub const MAX_BROWSER_CHUNK_BYTES: u64 = 16 * 1024 * 1024;
pub const DEFAULT_BROWSER_CHUNK_BYTES: u64 = 4 * 1024 * 1024;
/// Cache Storage entries are deliberately no larger than the default transport
/// chunk. Keeping the cache object granularity fixed makes admission and
/// cold/warm traffic accounting independent of caller tuning.
pub const MAX_BROWSER_CACHE_CHUNK_BYTES: u64 = 4 * 1024 * 1024;
/// Versioned, origin-scoped Cache Storage namespace for authenticated weight
/// ranges. Changing the key or response representation requires a new name.
pub const BROWSER_ARTIFACT_RANGE_CACHE_NAME: &str = "burn-image-artifact-ranges-v1";
/// Hard ceiling for one semantic Burnpack object retained in Wasm linear memory.
pub const MAX_BROWSER_STAGE_BYTES: u64 = 256 * 1024 * 1024;
/// Bootstrap metadata must remain small enough to fetch before the sealed manifest is known.
pub const MAX_BROWSER_MANIFEST_BYTES: u64 = 4 * 1024 * 1024;

/// Browser range-cache contract. Disabled preserves single-pass readers such
/// as exact 1.5K parity; required mode is for policies that deliberately read
/// the same immutable object more than once and must not fall back to repeated
/// network transfer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BrowserRangeCachePolicy {
    #[default]
    Disabled,
    Required,
}

/// Resolve an immutable dependency bundle beside a composed bundle prefix.
///
/// A composed URL such as `https://cdn.example/model/pipeline` resolves the
/// dependency `qwen` as `https://cdn.example/model/qwen`. The dependency id is
/// already a validated artifact identifier, so it cannot inject URL path
/// traversal or a second origin.
pub fn sibling_bundle_base_url(
    composed_base: &RemoteBaseUrl,
    dependency_bundle: &ArtifactBundleId,
) -> Result<RemoteBaseUrl, ArtifactStreamError> {
    let value = composed_base.as_str();
    let scheme_end = value
        .find("://")
        .expect("RemoteBaseUrl always contains a validated HTTP(S) scheme")
        + 3;
    let slash = value[scheme_end..]
        .rfind('/')
        .map(|index| scheme_end + index)
        .ok_or_else(|| ArtifactStreamError::DependencySiblingBase {
            base_url: value.to_owned(),
        })?;
    RemoteBaseUrl::new(format!(
        "{}/{}",
        &value[..slash],
        dependency_bundle.as_str()
    ))
    .map_err(|_| ArtifactStreamError::DependencySiblingBase {
        base_url: value.to_owned(),
    })
}

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
    #[error("artifact base URL {base_url} has no sibling bundle prefix")]
    DependencySiblingBase { base_url: String },
    #[error("browser fetch is unavailable because Window is missing")]
    BrowserWindowUnavailable,
    #[error("browser fetch request failed: {0}")]
    BrowserRequest(String),
    #[error("browser Cache Storage is required for bounded stage loading but is unavailable: {0}")]
    BrowserCacheUnavailable(String),
    #[error(
        "browser artifact range cache '{cache}' failed during {operation}: {message}; this cache is required and repeated-network fallback is disabled"
    )]
    BrowserCacheOperation {
        cache: &'static str,
        operation: &'static str,
        message: String,
    },
    #[error(
        "browser artifact range cache '{cache}' lost {path} bytes {offset}..{end_exclusive} after this active reader session populated it; refusing a repeated network transfer (entries from an earlier browser session are opportunistic until rewritten)"
    )]
    BrowserCacheSessionEntryLost {
        cache: &'static str,
        path: ArtifactPath,
        offset: u64,
        end_exclusive: u64,
    },
    #[error(
        "browser artifact {path} still has SHA-256 {actual} after one cache eviction and network refetch; expected {expected}"
    )]
    BrowserCacheIntegrityRetryFailed {
        path: ArtifactPath,
        expected: Sha256Digest,
        actual: Sha256Digest,
    },
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

/// A synthetic Cache Storage key never reaches the network. It binds the
/// cache-format version, exact source URL, sealed object digest, and byte
/// range. Hashing the complete URL avoids ambiguous escaping while retaining
/// collision resistance equivalent to the object's SHA-256 identity.
#[cfg(any(all(target_arch = "wasm32", feature = "boogu-web"), test))]
fn browser_range_cache_key(request: &BrowserRangeRequest, object_digest: Sha256Digest) -> String {
    let url_digest = Sha256Digest::calculate(request.url.as_bytes());
    format!(
        "https://burn-image.invalid/.well-known/range-cache/v1/{url_digest}/{object_digest}/{}-{}",
        request.range.offset(),
        request.range.end_exclusive()
    )
}

#[cfg(any(all(target_arch = "wasm32", feature = "boogu-web"), test))]
const fn browser_cache_chunk_length(remaining: u64, configured: u64) -> u64 {
    let configured = if configured < MAX_BROWSER_CACHE_CHUNK_BYTES {
        configured
    } else {
        MAX_BROWSER_CACHE_CHUNK_BYTES
    };
    if remaining < configured {
        remaining
    } else {
        configured
    }
}

#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
async fn open_browser_artifact_range_cache() -> Result<web_sys::Cache, ArtifactStreamError> {
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;

    let window = web_sys::window().ok_or(ArtifactStreamError::BrowserWindowUnavailable)?;
    let storage = window
        .caches()
        .map_err(|value| ArtifactStreamError::BrowserCacheUnavailable(browser_js_message(value)))?;
    JsFuture::from(storage.open(BROWSER_ARTIFACT_RANGE_CACHE_NAME))
        .await
        .map_err(|value| browser_cache_operation_error("open", value))?
        .dyn_into::<web_sys::Cache>()
        .map_err(|value| browser_cache_operation_error("open result conversion", value))
}

#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
async fn browser_cache_match(
    cache: &web_sys::Cache,
    key: &str,
    expected_bytes: u64,
) -> Result<Option<Vec<u8>>, ArtifactStreamError> {
    use js_sys::Uint8Array;
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;

    let value = JsFuture::from(cache.match_with_str(key))
        .await
        .map_err(|value| browser_cache_operation_error("match", value))?;
    if value.is_null() || value.is_undefined() {
        return Ok(None);
    }
    let response = value
        .dyn_into::<web_sys::Response>()
        .map_err(|value| browser_cache_operation_error("match result conversion", value))?;
    if response.status() != 200 {
        return Ok(Some(Vec::new()));
    }
    // Inspect the browser-owned Blob length before copying the body into Wasm
    // linear memory. A malicious or stale Cache Storage entry therefore cannot
    // turn a <=4 MiB range read into an unbounded Wasm allocation.
    let blob = JsFuture::from(
        response
            .blob()
            .map_err(|value| browser_cache_operation_error("read cached response", value))?,
    )
    .await
    .map_err(|value| browser_cache_operation_error("read cached response", value))?
    .dyn_into::<web_sys::Blob>()
    .map_err(|value| browser_cache_operation_error("cached Blob conversion", value))?;
    if blob.size() != expected_bytes as f64 {
        return Ok(Some(Vec::new()));
    }
    let buffer = JsFuture::from(blob.array_buffer())
        .await
        .map_err(|value| browser_cache_operation_error("copy cached response", value))?;
    Ok(Some(Uint8Array::new(&buffer).to_vec()))
}

#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
async fn browser_cache_put(
    cache: &web_sys::Cache,
    key: &str,
    bytes: &[u8],
) -> Result<(), ArtifactStreamError> {
    use js_sys::Uint8Array;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{Response, ResponseInit};

    // Cache.put rejects partial (206) responses. Copy the authenticated-range
    // payload into a synthetic status-200 response instead of storing the
    // transport response or a view into Wasm linear memory.
    let copied = Uint8Array::from(bytes);
    let init = ResponseInit::new();
    init.set_status(200);
    let response = Response::new_with_opt_js_u8_array_and_init(Some(&copied), &init)
        .map_err(|value| browser_cache_operation_error("construct status-200 response", value))?;
    JsFuture::from(cache.put_with_str(key, &response))
        .await
        .map_err(|value| browser_cache_operation_error("put required range", value))?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
async fn browser_cache_delete(
    cache: &web_sys::Cache,
    key: &str,
) -> Result<bool, ArtifactStreamError> {
    use wasm_bindgen_futures::JsFuture;

    let value = JsFuture::from(cache.delete_with_str(key))
        .await
        .map_err(|value| browser_cache_operation_error("delete", value))?;
    value
        .as_bool()
        .ok_or_else(|| browser_cache_operation_error("delete result conversion", value))
}

#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
fn browser_cache_operation_error(
    operation: &'static str,
    value: wasm_bindgen::JsValue,
) -> ArtifactStreamError {
    ArtifactStreamError::BrowserCacheOperation {
        cache: BROWSER_ARTIFACT_RANGE_CACHE_NAME,
        operation,
        message: browser_js_message(value),
    }
}

#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
fn browser_js_message(value: wasm_bindgen::JsValue) -> String {
    value.as_string().unwrap_or_else(|| format!("{value:?}"))
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

/// Monotonic browser artifact-reader traffic counters.
///
/// `range_fetch_requests` and `range_response_bytes` preserve the original
/// logical-reader contract. The explicit cache and network counters identify
/// how those logical reads were served without relying on opaque Fetch API
/// cache attribution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BrowserArtifactTrafficSnapshot {
    pub object_reads: u64,
    pub object_read_bytes: u64,
    pub range_fetch_requests: u64,
    pub range_response_bytes: u64,
    pub verified_objects: u64,
    pub cache_lookup_requests: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_read_bytes: u64,
    pub network_fetch_requests: u64,
    pub network_response_bytes: u64,
    pub cache_write_requests: u64,
    pub cache_write_bytes: u64,
    pub cache_eviction_requests: u64,
    pub cache_evicted_entries: u64,
    pub cache_invalid_entries: u64,
    pub integrity_refetches: u64,
}

impl BrowserArtifactTrafficSnapshot {
    pub fn checked_delta(self, earlier: Self) -> Option<Self> {
        Some(Self {
            object_reads: self.object_reads.checked_sub(earlier.object_reads)?,
            object_read_bytes: self
                .object_read_bytes
                .checked_sub(earlier.object_read_bytes)?,
            range_fetch_requests: self
                .range_fetch_requests
                .checked_sub(earlier.range_fetch_requests)?,
            range_response_bytes: self
                .range_response_bytes
                .checked_sub(earlier.range_response_bytes)?,
            verified_objects: self
                .verified_objects
                .checked_sub(earlier.verified_objects)?,
            cache_lookup_requests: self
                .cache_lookup_requests
                .checked_sub(earlier.cache_lookup_requests)?,
            cache_hits: self.cache_hits.checked_sub(earlier.cache_hits)?,
            cache_misses: self.cache_misses.checked_sub(earlier.cache_misses)?,
            cache_read_bytes: self
                .cache_read_bytes
                .checked_sub(earlier.cache_read_bytes)?,
            network_fetch_requests: self
                .network_fetch_requests
                .checked_sub(earlier.network_fetch_requests)?,
            network_response_bytes: self
                .network_response_bytes
                .checked_sub(earlier.network_response_bytes)?,
            cache_write_requests: self
                .cache_write_requests
                .checked_sub(earlier.cache_write_requests)?,
            cache_write_bytes: self
                .cache_write_bytes
                .checked_sub(earlier.cache_write_bytes)?,
            cache_eviction_requests: self
                .cache_eviction_requests
                .checked_sub(earlier.cache_eviction_requests)?,
            cache_evicted_entries: self
                .cache_evicted_entries
                .checked_sub(earlier.cache_evicted_entries)?,
            cache_invalid_entries: self
                .cache_invalid_entries
                .checked_sub(earlier.cache_invalid_entries)?,
            integrity_refetches: self
                .integrity_refetches
                .checked_sub(earlier.integrity_refetches)?,
        })
    }
}

/// Keys successfully committed or fully SHA-256-verified by this active reader-control session.
/// The set lives inside [`BrowserArtifactControl`], so every cloned reader observes the same
/// continuity contract. A prior-session hit becomes protected only after the complete sealed
/// object has passed its digest gate; a later miss then fails instead of repeating network I/O.
#[cfg(any(test, all(target_arch = "wasm32", feature = "boogu-web")))]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct BrowserRangeCacheSession {
    populated_keys: BTreeSet<String>,
}

#[cfg(any(test, all(target_arch = "wasm32", feature = "boogu-web")))]
impl BrowserRangeCacheSession {
    fn record_populated(&mut self, key: &str) {
        self.populated_keys.insert(key.to_owned());
    }

    fn was_populated(&self, key: &str) -> bool {
        self.populated_keys.contains(key)
    }
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
    traffic: BrowserArtifactTrafficSnapshot,
    active_loaded_bytes: BTreeMap<ArtifactPath, u64>,
    cache_session: BrowserRangeCacheSession,
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

    pub fn traffic_snapshot(&self) -> BrowserArtifactTrafficSnapshot {
        self.inner
            .lock()
            .expect("browser artifact control mutex poisoned")
            .traffic
    }

    pub fn set_observer(&self, observer: Option<Arc<dyn Fn(BrowserArtifactEvent) + Send + Sync>>) {
        self.inner
            .lock()
            .expect("browser artifact control mutex poisoned")
            .observer = observer;
    }

    fn record_cache_lookup(&self, hit_bytes: Option<u64>, invalid: bool) {
        let mut state = self
            .inner
            .lock()
            .expect("browser artifact control mutex poisoned");
        state.traffic.cache_lookup_requests = state.traffic.cache_lookup_requests.saturating_add(1);
        match hit_bytes {
            Some(bytes) => {
                state.traffic.cache_hits = state.traffic.cache_hits.saturating_add(1);
                state.traffic.cache_read_bytes =
                    state.traffic.cache_read_bytes.saturating_add(bytes);
            }
            None => {
                state.traffic.cache_misses = state.traffic.cache_misses.saturating_add(1);
            }
        }
        if invalid {
            state.traffic.cache_invalid_entries =
                state.traffic.cache_invalid_entries.saturating_add(1);
        }
    }

    fn record_logical_range(&self, bytes: u64) {
        let mut state = self
            .inner
            .lock()
            .expect("browser artifact control mutex poisoned");
        state.traffic.range_fetch_requests = state.traffic.range_fetch_requests.saturating_add(1);
        state.traffic.range_response_bytes =
            state.traffic.range_response_bytes.saturating_add(bytes);
    }

    fn record_network_fetch(&self, bytes: u64) {
        let mut state = self
            .inner
            .lock()
            .expect("browser artifact control mutex poisoned");
        state.traffic.network_fetch_requests =
            state.traffic.network_fetch_requests.saturating_add(1);
        state.traffic.network_response_bytes =
            state.traffic.network_response_bytes.saturating_add(bytes);
    }

    fn record_cache_write(&self, key: &str, bytes: u64) {
        let mut state = self
            .inner
            .lock()
            .expect("browser artifact control mutex poisoned");
        state.traffic.cache_write_requests = state.traffic.cache_write_requests.saturating_add(1);
        state.traffic.cache_write_bytes = state.traffic.cache_write_bytes.saturating_add(bytes);
        state.cache_session.record_populated(key);
    }

    fn cache_key_was_populated(&self, key: &str) -> bool {
        self.inner
            .lock()
            .expect("browser artifact control mutex poisoned")
            .cache_session
            .was_populated(key)
    }

    fn protect_verified_cache_key(&self, key: &str) {
        self.inner
            .lock()
            .expect("browser artifact control mutex poisoned")
            .cache_session
            .record_populated(key);
    }

    fn record_cache_eviction(&self, removed: bool) {
        let mut state = self
            .inner
            .lock()
            .expect("browser artifact control mutex poisoned");
        state.traffic.cache_eviction_requests =
            state.traffic.cache_eviction_requests.saturating_add(1);
        if removed {
            state.traffic.cache_evicted_entries =
                state.traffic.cache_evicted_entries.saturating_add(1);
        }
    }

    fn record_integrity_refetch(&self) {
        let mut state = self
            .inner
            .lock()
            .expect("browser artifact control mutex poisoned");
        state.traffic.integrity_refetches = state.traffic.integrity_refetches.saturating_add(1);
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
        match &event {
            BrowserArtifactEvent::Started(file) => {
                state.traffic.object_reads = state.traffic.object_reads.saturating_add(1);
                state.traffic.object_read_bytes =
                    state.traffic.object_read_bytes.saturating_add(file.size);
                state.active_loaded_bytes.insert(file.path.clone(), 0);
            }
            BrowserArtifactEvent::Progress {
                path, loaded_bytes, ..
            } => {
                state
                    .active_loaded_bytes
                    .insert(path.clone(), *loaded_bytes);
            }
            BrowserArtifactEvent::Verified(path) => {
                state.traffic.verified_objects = state.traffic.verified_objects.saturating_add(1);
                state.active_loaded_bytes.remove(path);
            }
        }
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
/// [`MAX_BROWSER_STAGE_BYTES`] and verified exactly once against the manifest file SHA-256. Typed
/// evidence is returned with the bytes so the model source can validate the proof without hashing
/// the complete object again.
#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
#[derive(Clone)]
pub struct BrowserStageShardReader {
    source: ArtifactSource,
    config: ArtifactStreamConfig,
    control: BrowserArtifactControl,
    progress_bundle: Option<ArtifactBundleId>,
    cache_policy: BrowserRangeCachePolicy,
    cache: Option<web_sys::Cache>,
}

#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
impl BrowserStageShardReader {
    pub fn new(base_url: RemoteBaseUrl, config: ArtifactStreamConfig) -> Self {
        Self {
            source: ArtifactSource::Remote { base_url },
            config,
            control: BrowserArtifactControl::default(),
            progress_bundle: None,
            cache_policy: BrowserRangeCachePolicy::Disabled,
            cache: None,
        }
    }

    /// Construct a reader whose progress events identify the independently
    /// sealed bundle and share cancellation/event routing with sibling readers.
    pub fn for_bundle(
        base_url: RemoteBaseUrl,
        bundle: ArtifactBundleId,
        config: ArtifactStreamConfig,
        control: BrowserArtifactControl,
    ) -> Self {
        Self {
            source: ArtifactSource::Remote { base_url },
            config,
            control,
            progress_bundle: Some(bundle),
            cache_policy: BrowserRangeCachePolicy::Disabled,
            cache: None,
        }
    }

    /// Construct an unqualified legacy reader that shares progress and
    /// cancellation routing with sibling reader wrappers.
    pub fn with_control(
        base_url: RemoteBaseUrl,
        config: ArtifactStreamConfig,
        control: BrowserArtifactControl,
    ) -> Self {
        Self {
            source: ArtifactSource::Remote { base_url },
            config,
            control,
            progress_bundle: None,
            cache_policy: BrowserRangeCachePolicy::Disabled,
            cache: None,
        }
    }

    pub fn control(&self) -> BrowserArtifactControl {
        self.control.clone()
    }

    /// Require verified <=4 MiB range entries in Cache Storage. Any cache or
    /// quota failure aborts instead of silently repeating network traffic.
    pub fn with_required_range_cache(mut self) -> Self {
        self.cache_policy = BrowserRangeCachePolicy::Required;
        self
    }

    pub const fn range_cache_policy(&self) -> BrowserRangeCachePolicy {
        self.cache_policy
    }

    pub async fn read_verified(&mut self, file: &ArtifactFile) -> Result<Vec<u8>, BooguError> {
        <Self as AsyncStageShardReader>::read_shard(self, file, MAX_BROWSER_STAGE_BYTES).await
    }

    fn progress_file(&self, file: &ArtifactFile) -> ArtifactFile {
        let Some(bundle) = &self.progress_bundle else {
            return file.clone();
        };
        let mut progress = file.clone();
        progress.path = ArtifactPath::new(format!("{bundle}/{}", file.path))
            .expect("validated bundle and artifact path compose into a valid path");
        progress.component = Some(
            ArtifactComponentId::new(bundle.as_str())
                .expect("an artifact bundle id is also a valid component id"),
        );
        progress
    }

    fn progress_path(&self, path: &ArtifactPath) -> ArtifactPath {
        self.progress_bundle.as_ref().map_or_else(
            || path.clone(),
            |bundle| {
                ArtifactPath::new(format!("{bundle}/{path}"))
                    .expect("validated bundle and artifact path compose into a valid path")
            },
        )
    }

    async fn cache(&mut self) -> Result<web_sys::Cache, BooguError> {
        if let Some(cache) = &self.cache {
            return Ok(cache.clone());
        }
        let cache = open_browser_artifact_range_cache()
            .await
            .map_err(|error| BooguError::Artifact(error.to_string()))?;
        self.cache = Some(cache.clone());
        Ok(cache)
    }

    async fn fetch_range_cached(
        &mut self,
        request: &BrowserRangeRequest,
        object_digest: Sha256Digest,
        force_network: bool,
    ) -> Result<ArtifactChunk, BooguError> {
        if request.range.length() > MAX_BROWSER_CACHE_CHUNK_BYTES {
            return Err(BooguError::Artifact(format!(
                "browser cache range for {} is {} bytes, exceeding the fixed {}-byte cache-entry cap",
                request.path,
                request.range.length(),
                MAX_BROWSER_CACHE_CHUNK_BYTES
            )));
        }
        let cache = self.cache().await?;
        let key = browser_range_cache_key(request, object_digest);
        if !force_network {
            let cached = browser_cache_match(&cache, &key, request.range.length())
                .await
                .map_err(|error| BooguError::Artifact(error.to_string()))?;
            match cached {
                Some(bytes) if u64::try_from(bytes.len()).ok() == Some(request.range.length()) => {
                    self.control
                        .record_cache_lookup(Some(request.range.length()), false);
                    self.control.record_logical_range(request.range.length());
                    return Ok(ArtifactChunk {
                        path: request.path.clone(),
                        range: request.range,
                        bytes,
                    });
                }
                Some(_) => {
                    // Cache entries are untrusted. A malformed response is a
                    // miss after successful eviction, never accepted or used
                    // to satisfy a logical range read.
                    self.control.record_cache_lookup(None, true);
                    let removed = browser_cache_delete(&cache, &key)
                        .await
                        .map_err(|error| BooguError::Artifact(error.to_string()))?;
                    self.control.record_cache_eviction(removed);
                }
                None => {
                    self.control.record_cache_lookup(None, false);
                    if self.control.cache_key_was_populated(&key) {
                        return Err(BooguError::Artifact(
                            ArtifactStreamError::BrowserCacheSessionEntryLost {
                                cache: BROWSER_ARTIFACT_RANGE_CACHE_NAME,
                                path: request.path.clone(),
                                offset: request.range.offset(),
                                end_exclusive: request.range.end_exclusive(),
                            }
                            .to_string(),
                        ));
                    }
                }
            }
        }

        let chunk = fetch_browser_range(request)
            .await
            .map_err(|error| BooguError::Artifact(error.to_string()))?;
        let bytes = u64::try_from(chunk.bytes.len()).map_err(|_| {
            BooguError::Artifact("browser range response byte count overflowed u64".into())
        })?;
        self.control.record_network_fetch(bytes);
        browser_cache_put(&cache, &key, &chunk.bytes)
            .await
            .map_err(|error| BooguError::Artifact(error.to_string()))?;
        self.control.record_cache_write(&key, bytes);
        self.control.record_logical_range(bytes);
        Ok(chunk)
    }

    async fn fetch_range(
        &mut self,
        request: &BrowserRangeRequest,
        object_digest: Sha256Digest,
        force_network: bool,
    ) -> Result<ArtifactChunk, BooguError> {
        if self.cache_policy == BrowserRangeCachePolicy::Required {
            return self
                .fetch_range_cached(request, object_digest, force_network)
                .await;
        }
        let chunk = fetch_browser_range(request)
            .await
            .map_err(|error| BooguError::Artifact(error.to_string()))?;
        let bytes = u64::try_from(chunk.bytes.len()).map_err(|_| {
            BooguError::Artifact("browser range response byte count overflowed u64".into())
        })?;
        self.control.record_network_fetch(bytes);
        self.control.record_logical_range(bytes);
        Ok(chunk)
    }

    async fn evict_object_ranges(&mut self, file: &ArtifactFile) -> Result<(), BooguError> {
        let cache = self.cache().await?;
        let mut offset = 0_u64;
        while offset < file.size {
            let length =
                browser_cache_chunk_length(file.size - offset, self.config.max_chunk_bytes());
            let range = ByteRange::new(offset, length).map_err(|error| {
                BooguError::Artifact(format!(
                    "invalid browser cache eviction range for {}: {error}",
                    file.path
                ))
            })?;
            let request = ArtifactReadRequest::ranged(file.path.clone(), range);
            let browser = BrowserRangeRequest::from_source(&self.source, &request)
                .map_err(|error| BooguError::Artifact(error.to_string()))?;
            let key = browser_range_cache_key(&browser, file.sha256);
            let removed = browser_cache_delete(&cache, &key)
                .await
                .map_err(|error| BooguError::Artifact(error.to_string()))?;
            self.control.record_cache_eviction(removed);
            offset = range.end_exclusive();
        }
        Ok(())
    }

    async fn fetch_shard_bytes_attempt(
        &mut self,
        file: &ArtifactFile,
        max_bytes: u64,
        force_network: bool,
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
        let mut bytes = Vec::with_capacity(capacity);
        let mut offset = 0_u64;
        while offset < file.size {
            self.control.check_cancelled()?;
            let length =
                browser_cache_chunk_length(file.size - offset, self.config.max_chunk_bytes());
            let range = ByteRange::new(offset, length).map_err(|error| {
                BooguError::Artifact(format!("invalid browser range for {}: {error}", file.path))
            })?;
            let request = ArtifactReadRequest::ranged(file.path.clone(), range);
            let browser = BrowserRangeRequest::from_source(&self.source, &request)
                .map_err(|error| BooguError::Artifact(error.to_string()))?;
            let chunk = self
                .fetch_range(&browser, file.sha256, force_network)
                .await?;
            bytes.extend_from_slice(&chunk.bytes);
            offset = range.end_exclusive();
            self.control.push(BrowserArtifactEvent::Progress {
                path: self.progress_path(&file.path),
                loaded_bytes: offset,
                total_bytes: file.size,
            });
        }
        Ok(bytes)
    }

    async fn fetch_verified_shard_bytes(
        &mut self,
        file: &ArtifactFile,
        max_bytes: u64,
    ) -> Result<Vec<u8>, BooguError> {
        self.control.check_cancelled()?;
        self.control
            .push(BrowserArtifactEvent::Started(self.progress_file(file)));
        let bytes = self
            .fetch_shard_bytes_attempt(file, max_bytes, false)
            .await?;
        let actual = Sha256Digest::calculate(&bytes);
        if actual == file.sha256 {
            self.protect_verified_object_ranges(file)?;
            return Ok(bytes);
        }

        // A complete-object digest is the final trust gate. Purge every range
        // for this URL/object identity and permit exactly one cache-bypassing
        // network refetch. The replacement ranges are still required to enter
        // Cache Storage successfully; quota failure never degrades to repeated
        // network reads.
        if self.cache_policy != BrowserRangeCachePolicy::Required {
            return Err(BooguError::Artifact(format!(
                "artifact integrity verification failed for {}: expected SHA-256 {}, found {}",
                file.path, file.sha256, actual
            )));
        }
        // The retry may reconstruct another full semantic object. Release the
        // failed allocation before any eviction awaits or replacement range
        // fetch so peak Wasm memory remains one bounded object plus one chunk.
        drop(bytes);
        self.control.record_integrity_refetch();
        self.evict_object_ranges(file).await?;
        let bytes = self
            .fetch_shard_bytes_attempt(file, max_bytes, true)
            .await?;
        let actual = Sha256Digest::calculate(&bytes);
        if actual != file.sha256 {
            drop(bytes);
            self.evict_object_ranges(file).await?;
            return Err(BooguError::Artifact(
                ArtifactStreamError::BrowserCacheIntegrityRetryFailed {
                    path: file.path.clone(),
                    expected: file.sha256,
                    actual,
                }
                .to_string(),
            ));
        }
        self.protect_verified_object_ranges(file)?;
        Ok(bytes)
    }

    fn protect_verified_object_ranges(&self, file: &ArtifactFile) -> Result<(), BooguError> {
        if self.cache_policy != BrowserRangeCachePolicy::Required {
            return Ok(());
        }
        let mut offset = 0_u64;
        while offset < file.size {
            let length =
                browser_cache_chunk_length(file.size - offset, self.config.max_chunk_bytes());
            let range = ByteRange::new(offset, length).map_err(|error| {
                BooguError::Artifact(format!(
                    "invalid verified browser cache range for {}: {error}",
                    file.path
                ))
            })?;
            let request = ArtifactReadRequest::ranged(file.path.clone(), range);
            let browser = BrowserRangeRequest::from_source(&self.source, &request)
                .map_err(|error| BooguError::Artifact(error.to_string()))?;
            self.control
                .protect_verified_cache_key(&browser_range_cache_key(&browser, file.sha256));
            offset = range.end_exclusive();
        }
        Ok(())
    }
}

#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
impl AsyncStageShardReader for BrowserStageShardReader {
    async fn read_shard(
        &mut self,
        file: &ArtifactFile,
        max_bytes: u64,
    ) -> Result<Vec<u8>, BooguError> {
        Ok(self.read_stage_shard(file, max_bytes).await?.into_bytes())
    }

    async fn read_stage_shard(
        &mut self,
        file: &ArtifactFile,
        max_bytes: u64,
    ) -> Result<AsyncStageShardRead, BooguError> {
        let bytes = self.fetch_verified_shard_bytes(file, max_bytes).await?;
        let read = AsyncStageShardRead::verify_sha256(file, bytes)?;
        self.control.push(BrowserArtifactEvent::Verified(
            self.progress_path(&file.path),
        ));
        Ok(read)
    }
}

/// Model crates consume the model-neutral reader contract. Keep the legacy Boogu reader
/// implementation above only for the variant-specific denoiser during the compatibility window.
#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
impl AsyncArtifactShardReader for BrowserStageShardReader {
    async fn read_shard(
        &mut self,
        file: &ArtifactFile,
        maximum_bytes: u64,
    ) -> Result<Vec<u8>, ArtifactReadError> {
        Ok(self
            .read_verified_shard(file, maximum_bytes)
            .await?
            .into_bytes())
    }

    async fn read_verified_shard(
        &mut self,
        file: &ArtifactFile,
        maximum_bytes: u64,
    ) -> Result<VerifiedArtifactBytes, ArtifactReadError> {
        let bytes = self
            .fetch_verified_shard_bytes(file, maximum_bytes)
            .await
            .map_err(|error| ArtifactReadError::transport(error.to_string()))?;
        let read = VerifiedArtifactBytes::verify_sha256(file, bytes)?;
        self.control.push(BrowserArtifactEvent::Verified(
            self.progress_path(&file.path),
        ));
        Ok(read)
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
    fn browser_cache_key_binds_url_digest_and_exact_range_correctness() {
        let source = ArtifactSource::Remote {
            base_url: RemoteBaseUrl::new("https://cdn.example/models").unwrap(),
        };
        let request = ArtifactReadRequest::ranged(
            ArtifactPath::new("weights/a.bpk").unwrap(),
            ByteRange::new(8, 4).unwrap(),
        );
        let browser = BrowserRangeRequest::from_source(&source, &request).unwrap();
        let digest = Sha256Digest::calculate(b"sealed object");
        let key = browser_range_cache_key(&browser, digest);
        assert!(key.starts_with("https://burn-image.invalid/.well-known/range-cache/v1/"));
        assert!(key.ends_with("/8-12"));
        assert_eq!(key, browser_range_cache_key(&browser, digest));

        let other_range = BrowserRangeRequest {
            range: ByteRange::new(9, 4).unwrap(),
            range_header: "bytes=9-12".into(),
            ..browser.clone()
        };
        assert_ne!(key, browser_range_cache_key(&other_range, digest));
        let mut other_url = browser;
        other_url.url.push_str("&mirror=1");
        assert_ne!(key, browser_range_cache_key(&other_url, digest));
        assert_ne!(
            key,
            browser_range_cache_key(&other_url, Sha256Digest::calculate(b"replacement object"))
        );
    }

    #[test]
    fn browser_cache_chunks_never_exceed_four_mib_correctness() {
        assert_eq!(
            browser_cache_chunk_length(16 * 1024 * 1024, MAX_BROWSER_CHUNK_BYTES),
            MAX_BROWSER_CACHE_CHUNK_BYTES
        );
        assert_eq!(browser_cache_chunk_length(17, 8), 8);
        assert_eq!(browser_cache_chunk_length(7, 8), 7);
    }

    #[test]
    fn browser_range_cache_policy_is_opt_in_and_required_is_explicit_correctness() {
        assert_eq!(
            BrowserRangeCachePolicy::default(),
            BrowserRangeCachePolicy::Disabled
        );
        assert_eq!(
            serde_json::to_string(&BrowserRangeCachePolicy::Disabled).unwrap(),
            "\"disabled\""
        );
        assert_eq!(
            serde_json::to_string(&BrowserRangeCachePolicy::Required).unwrap(),
            "\"required\""
        );
    }

    #[test]
    fn browser_cache_session_distinguishes_cold_and_lost_entries_correctness() {
        let key = "https://burn-image.invalid/.well-known/range-cache/v1/key";
        let mut active = BrowserRangeCacheSession::default();
        // An empty new control cannot know whether another browser session populated and later
        // evicted this key, so its first miss is cold. Production also calls this marker after a
        // prior-session hit participates in a completely SHA-256-verified object.
        assert!(!active.was_populated(key));
        active.record_populated(key);
        assert!(active.was_populated(key));

        // The state itself is shared by BrowserArtifactControl's Arc/Mutex;
        // cloning a reader therefore cannot erase the continuity marker.
        active.record_populated("second-reader-key");
        assert!(active.was_populated("second-reader-key"));

        let next_engine = BrowserRangeCacheSession::default();
        assert!(!next_engine.was_populated(key));
    }

    #[test]
    fn browser_lost_session_entry_error_names_no_repeat_network_contract_correctness() {
        let error = ArtifactStreamError::BrowserCacheSessionEntryLost {
            cache: BROWSER_ARTIFACT_RANGE_CACHE_NAME,
            path: ArtifactPath::new("weights/a.bpk").unwrap(),
            offset: 8,
            end_exclusive: 12,
        }
        .to_string();
        assert!(error.contains("after this active reader session populated it"));
        assert!(error.contains("refusing a repeated network transfer"));
        assert!(error.contains("earlier browser session"));
    }

    #[test]
    fn browser_traffic_delta_keeps_logical_cache_and_network_counts_distinct_correctness() {
        let earlier = BrowserArtifactTrafficSnapshot {
            object_reads: 1,
            object_read_bytes: 10,
            range_fetch_requests: 2,
            range_response_bytes: 20,
            verified_objects: 1,
            cache_lookup_requests: 2,
            cache_hits: 1,
            cache_misses: 1,
            cache_read_bytes: 8,
            network_fetch_requests: 1,
            network_response_bytes: 12,
            cache_write_requests: 1,
            cache_write_bytes: 12,
            cache_eviction_requests: 0,
            cache_evicted_entries: 0,
            cache_invalid_entries: 0,
            integrity_refetches: 0,
        };
        let later = BrowserArtifactTrafficSnapshot {
            object_reads: 2,
            object_read_bytes: 30,
            range_fetch_requests: 5,
            range_response_bytes: 44,
            verified_objects: 2,
            cache_lookup_requests: 5,
            cache_hits: 4,
            cache_misses: 1,
            cache_read_bytes: 32,
            network_fetch_requests: 1,
            network_response_bytes: 12,
            cache_write_requests: 1,
            cache_write_bytes: 12,
            cache_eviction_requests: 1,
            cache_evicted_entries: 1,
            cache_invalid_entries: 1,
            integrity_refetches: 1,
        };
        let delta = later.checked_delta(earlier).unwrap();
        assert_eq!(delta.range_fetch_requests, 3);
        assert_eq!(delta.range_response_bytes, 24);
        assert_eq!(delta.cache_lookup_requests, 3);
        assert_eq!(delta.cache_hits, 3);
        assert_eq!(delta.cache_misses, 0);
        assert_eq!(delta.cache_read_bytes, 24);
        assert_eq!(delta.network_fetch_requests, 0);
        assert_eq!(delta.network_response_bytes, 0);
        assert_eq!(delta.cache_eviction_requests, 1);
        assert_eq!(delta.cache_evicted_entries, 1);
        assert_eq!(delta.cache_invalid_entries, 1);
        assert_eq!(delta.integrity_refetches, 1);
    }

    #[test]
    fn dependency_bundle_resolves_as_a_same_origin_sibling_correctness() {
        let pipeline =
            RemoteBaseUrl::new("https://cdn.example/model/boogu-image-0.1-turbo").unwrap();
        let qwen = ArtifactBundleId::new("qwen3-vl-shared").unwrap();
        assert_eq!(
            sibling_bundle_base_url(&pipeline, &qwen).unwrap().as_str(),
            "https://cdn.example/model/qwen3-vl-shared"
        );

        let origin_only = RemoteBaseUrl::new("https://cdn.example").unwrap();
        assert!(matches!(
            sibling_bundle_base_url(&origin_only, &qwen),
            Err(ArtifactStreamError::DependencySiblingBase { .. })
        ));
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
