#[cfg(any(test, target_arch = "wasm32"))]
use burn_image::VerifiedArtifactTransportLayout;
#[cfg(any(test, all(target_arch = "wasm32", feature = "boogu-web")))]
use burn_image::{
    ARTIFACT_TRANSPORT_MAX_PART_BYTES, ARTIFACT_TRANSPORT_TARGET_PART_BYTES, ArtifactFileRole,
    ArtifactTransportObject, ArtifactTransportPart,
};
use burn_image::{
    ArtifactBundleId, ArtifactFile, ArtifactPath, ArtifactReadRequest, ArtifactSource,
    ArtifactVerifier, ByteRange, IntegrityPolicy, RemoteBaseUrl, Sha256Digest, VerifiedArtifact,
};
#[cfg(any(test, all(target_arch = "wasm32", feature = "boogu-web")))]
use burn_image::{ArtifactComponentId, ArtifactRequestTransferActivity, ArtifactTransferProgress};
#[cfg(any(test, target_arch = "wasm32"))]
use burn_image::{ArtifactManifest, ArtifactTransportLayout, MAX_ARTIFACT_TRANSPORT_LAYOUT_BYTES};
#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
use burn_image::{
    ArtifactReadError, AsyncArtifactShardReader, VerifiedArtifactBytes,
    VerifiedArtifactBytesBuilder,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
use std::collections::VecDeque;
#[cfg(any(test, all(target_arch = "wasm32", feature = "boogu-web")))]
use std::collections::{BTreeMap, BTreeSet};
#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
use std::sync::{Arc, Mutex};

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
/// Versioned Cache Storage namespace for complete, independently authenticated
/// CDN transport parts. Canonical parts are at most 20 MiB, so one browser-owned
/// response and one bounded Wasm copy replace five serial 4 MiB range/cache
/// operations without retaining more than one physical part at a time.
pub const BROWSER_ARTIFACT_PART_CACHE_NAME: &str = "burn-image-artifact-parts-v2";
/// Free origin quota retained beyond the exact missing model payload. Cache Storage bookkeeping,
/// compact runtime metadata, and one corrupt-part replacement must not turn an admitted model
/// load into a late quota failure.
pub const BROWSER_PERSISTENT_CACHE_RESERVE_BYTES: u64 = 256 * 1024 * 1024;
#[cfg(any(test, all(target_arch = "wasm32", feature = "boogu-web")))]
const BROWSER_PERSISTENT_CACHE_OVERHEAD_DIVISOR: u64 = 100;
/// Hard ceiling for one semantic Burnpack object retained in Wasm linear memory.
pub const MAX_BROWSER_STAGE_BYTES: u64 = 256 * 1024 * 1024;
/// Bootstrap metadata must remain small enough to fetch before the sealed manifest is known.
pub const MAX_BROWSER_MANIFEST_BYTES: u64 = 4 * 1024 * 1024;

/// Browser artifact-cache contract. Disabled preserves isolated qualification readers. Required
/// mode is used by interactive policies so immutable transport parts survive reloads and release
/// switches without depending on the opportunistic HTTP cache.
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

/// Return the zero-based semantic-file position used by `ProgressEvent::ArtifactStarted`.
///
/// Unsharded semantic objects are represented as the sole object in a one-object group. Keeping
/// this conversion beside the transport contract prevents browser adapters from accidentally
/// reporting one-based indices that UI formatters increment a second time. This stage-local count
/// is retained as machine telemetry; it is not a Wasm residency cap or a user-facing transfer
/// denominator. Browser UIs use the aggregate transport closure attached to the progress event.
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
    #[error("browser persistent-storage operation {operation} failed: {message}")]
    BrowserStorageOperation {
        operation: &'static str,
        message: String,
    },
    #[error("browser storage estimate omitted or returned invalid {field}: {actual:?}")]
    BrowserStorageEstimate {
        field: &'static str,
        actual: Option<String>,
    },
    #[error(
        "browser origin has {available_bytes} available storage bytes, but the selected model still needs {missing_bytes} cache bytes plus {reserve_bytes} bytes of safety reserve ({cached_entries}/{total_entries} exact cache entries already present); free origin storage or clear unrelated site data before retrying"
    )]
    BrowserStorageQuotaInsufficient {
        available_bytes: u64,
        missing_bytes: u64,
        reserve_bytes: u64,
        cached_entries: u64,
        total_entries: u64,
    },
    #[error("browser persistent-cache plan is invalid: {0}")]
    BrowserPersistentCachePlan(String),
    #[error("browser range fetch returned HTTP {status} for {url}; expected 206")]
    BrowserHttpStatus { status: u16, url: String },
    #[error("browser complete-object fetch returned HTTP {status} for {url}; expected 200")]
    BrowserCompleteObjectHttpStatus { status: u16, url: String },
    #[error("browser range response has Content-Range {actual:?}; expected {expected}")]
    BrowserContentRange {
        expected: String,
        actual: Option<String>,
    },
    #[error("browser response contains {actual} bytes; expected {expected}")]
    BrowserResponseSize { expected: u64, actual: u64 },
    #[error("browser response has Content-Length {actual:?}; expected exact decimal {expected}")]
    BrowserContentLength {
        expected: u64,
        actual: Option<String>,
    },
    #[error("browser response has non-canonical or missing Content-Length {actual:?}")]
    BrowserMalformedContentLength { actual: Option<String> },
    #[error(
        "browser range response has Content-Encoding {actual:?}; expected absent or identity so Content-Length bounds the decoded body"
    )]
    BrowserContentEncoding { actual: String },
    #[error("browser Content-Range header is malformed: {0:?}")]
    BrowserMalformedContentRange(Option<String>),
    #[error("browser file contains {actual} bytes, above the bounded maximum {maximum}")]
    BrowserFileTooLarge { actual: u64, maximum: u64 },
    #[error("browser artifact transport layout is invalid: {0}")]
    BrowserTransportLayout(String),
    #[error("browser artifact transport layout omits required logical weight object {path}")]
    BrowserTransportObjectMissing { path: ArtifactPath },
    #[error(
        "browser transport part {path} has SHA-256 {actual}; expected independently sealed digest {expected}"
    )]
    BrowserTransportPartIntegrity {
        path: ArtifactPath,
        expected: Sha256Digest,
        actual: Sha256Digest,
    },
    #[error(
        "browser transport reconstruction for {path} has SHA-256 {actual}; expected logical artifact digest {expected}"
    )]
    BrowserTransportReconstructionIntegrity {
        path: ArtifactPath,
        expected: Sha256Digest,
        actual: Sha256Digest,
    },
    #[error(
        "browser transport part {path} contains {actual} bytes, above the bounded part maximum {maximum}"
    )]
    BrowserTransportPartTooLarge {
        path: ArtifactPath,
        actual: u64,
        maximum: u64,
    },
    #[error("browser transport part {path} contains {actual} bytes; expected exactly {expected}")]
    BrowserTransportPartSize {
        path: ArtifactPath,
        expected: u64,
        actual: u64,
    },
    #[error("browser Web Crypto SHA-256 failed for transport part {path}: {message}")]
    BrowserTransportPartCrypto { path: ArtifactPath, message: String },
    #[error("browser Web Crypto returned {actual} SHA-256 bytes for {path}; expected 32")]
    BrowserTransportPartCryptoSize { path: ArtifactPath, actual: u32 },
    #[error(
        "browser transport reconstruction for {path} contains {actual} bytes; expected exactly {expected}"
    )]
    BrowserTransportReconstructionSize {
        path: ArtifactPath,
        expected: u64,
        actual: u64,
    },
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
    fetch_browser_range_with_total(request, None).await
}

/// Fetch one exact browser range and, when the caller already has a sealed
/// object contract, require `Content-Range` to name that exact total size.
///
/// A generic range fetch can only prove that the returned interval exists.
/// Artifact files and transport parts know their complete size in advance, so
/// accepting a different server-side object length would weaken the sealed
/// path/size/digest binding and can hide a mutable or misrouted CDN object.
#[cfg(target_arch = "wasm32")]
async fn fetch_browser_range_with_total(
    request: &BrowserRangeRequest,
    expected_total: Option<u64>,
) -> Result<ArtifactChunk, ArtifactStreamError> {
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
    match expected_total {
        Some(total) => {
            validate_content_range_exact_total(request.range, content_range.as_deref(), total)?;
        }
        None => validate_content_range(request.range, content_range.as_deref())?,
    }
    let bytes = read_browser_response_body_bounded(&response, request.range.length()).await?;
    Ok(ArtifactChunk {
        path: request.path.clone(),
        range: request.range,
        bytes,
    })
}

/// Synthetic key for one complete content-addressed transport part.
///
/// The key binds the cache representation, exact resolved source URL, sealed
/// digest, and declared byte size. It never reaches the network.
#[cfg(any(all(target_arch = "wasm32", feature = "boogu-web"), test))]
fn browser_part_cache_key(url: &str, object_digest: Sha256Digest, object_size: u64) -> String {
    let url_digest = Sha256Digest::calculate(url.as_bytes());
    format!(
        "https://burn-image.invalid/.well-known/part-cache/v2/{url_digest}/{object_digest}/{object_size}"
    )
}

/// Exact Cache Storage objects required by one selected browser model closure.
///
/// Cache names remain part of the identity so representation changes create a new namespace.
#[cfg(any(test, all(target_arch = "wasm32", feature = "boogu-web")))]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct BrowserPersistentCachePlan {
    entries: BTreeMap<&'static str, BTreeMap<String, u64>>,
}

#[cfg(any(test, all(target_arch = "wasm32", feature = "boogu-web")))]
impl BrowserPersistentCachePlan {
    fn register(
        &mut self,
        cache: &'static str,
        key: String,
        size: u64,
    ) -> Result<(), ArtifactStreamError> {
        if size == 0 {
            return Err(ArtifactStreamError::BrowserPersistentCachePlan(format!(
                "cache {cache} contains zero-byte entry {key}"
            )));
        }
        match self.entries.entry(cache).or_default().entry(key) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(size);
                Ok(())
            }
            std::collections::btree_map::Entry::Occupied(entry) if *entry.get() == size => Ok(()),
            std::collections::btree_map::Entry::Occupied(entry) => {
                Err(ArtifactStreamError::BrowserPersistentCachePlan(format!(
                    "cache {cache} entry has conflicting sizes {} and {size}",
                    entry.get()
                )))
            }
        }
    }

    pub(crate) fn entry_count(&self) -> u64 {
        self.entries
            .values()
            .map(|entries| entries.len() as u64)
            .sum()
    }

    pub(crate) fn total_bytes(&self) -> u64 {
        self.entries
            .values()
            .flat_map(|entries| entries.values())
            .copied()
            .sum()
    }
}

/// Result of checking exact selected-model keys against origin Cache Storage and quota.
#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BrowserPersistentCachePreflight {
    pub(crate) total_entries: u64,
    pub(crate) cached_entries: u64,
    pub(crate) missing_entries: u64,
    pub(crate) missing_bytes: u64,
    pub(crate) storage_available_bytes: u64,
    pub(crate) persistent_storage_granted: bool,
}

#[cfg(any(test, all(target_arch = "wasm32", feature = "boogu-web")))]
fn browser_persistent_cache_reserve(missing_bytes: u64) -> u64 {
    if missing_bytes == 0 {
        return 0;
    }
    BROWSER_PERSISTENT_CACHE_RESERVE_BYTES
        .saturating_add(missing_bytes / BROWSER_PERSISTENT_CACHE_OVERHEAD_DIVISOR)
}

#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
async fn open_browser_artifact_cache(
    cache_name: &'static str,
) -> Result<web_sys::Cache, ArtifactStreamError> {
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;

    let window = web_sys::window().ok_or(ArtifactStreamError::BrowserWindowUnavailable)?;
    let storage = window
        .caches()
        .map_err(|value| ArtifactStreamError::BrowserCacheUnavailable(browser_js_message(value)))?;
    JsFuture::from(storage.open(cache_name))
        .await
        .map_err(|value| browser_cache_operation_error(cache_name, "open", value))?
        .dyn_into::<web_sys::Cache>()
        .map_err(|value| browser_cache_operation_error(cache_name, "open result conversion", value))
}

#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
async fn browser_cache_match(
    cache: &web_sys::Cache,
    cache_name: &'static str,
    key: &str,
    expected_bytes: u64,
) -> Result<Option<Vec<u8>>, ArtifactStreamError> {
    use js_sys::Uint8Array;
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;

    let value = JsFuture::from(cache.match_with_str(key))
        .await
        .map_err(|value| browser_cache_operation_error(cache_name, "match", value))?;
    if value.is_null() || value.is_undefined() {
        return Ok(None);
    }
    let response = value.dyn_into::<web_sys::Response>().map_err(|value| {
        browser_cache_operation_error(cache_name, "match result conversion", value)
    })?;
    if response.status() != 200 {
        return Ok(Some(Vec::new()));
    }
    // Inspect the browser-owned Blob length before copying the body into Wasm
    // linear memory. A malicious or stale Cache Storage entry therefore cannot
    // turn a bounded range or <=20 MiB transport-part read into an unbounded
    // Wasm allocation.
    let blob = JsFuture::from(response.blob().map_err(|value| {
        browser_cache_operation_error(cache_name, "read cached response", value)
    })?)
    .await
    .map_err(|value| browser_cache_operation_error(cache_name, "read cached response", value))?
    .dyn_into::<web_sys::Blob>()
    .map_err(|value| browser_cache_operation_error(cache_name, "cached Blob conversion", value))?;
    if blob.size() != expected_bytes as f64 {
        return Ok(Some(Vec::new()));
    }
    let buffer = JsFuture::from(blob.array_buffer()).await.map_err(|value| {
        browser_cache_operation_error(cache_name, "copy cached response", value)
    })?;
    Ok(Some(Uint8Array::new(&buffer).to_vec()))
}

#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
async fn browser_cache_put(
    cache: &web_sys::Cache,
    cache_name: &'static str,
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
    let response =
        Response::new_with_opt_js_u8_array_and_init(Some(&copied), &init).map_err(|value| {
            browser_cache_operation_error(cache_name, "construct status-200 response", value)
        })?;
    JsFuture::from(cache.put_with_str(key, &response))
        .await
        .map_err(|value| browser_cache_operation_error(cache_name, "put required object", value))?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
async fn browser_cache_delete(
    cache: &web_sys::Cache,
    cache_name: &'static str,
    key: &str,
) -> Result<bool, ArtifactStreamError> {
    use wasm_bindgen_futures::JsFuture;

    let value = JsFuture::from(cache.delete_with_str(key))
        .await
        .map_err(|value| browser_cache_operation_error(cache_name, "delete", value))?;
    value
        .as_bool()
        .ok_or_else(|| browser_cache_operation_error(cache_name, "delete result conversion", value))
}

#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
async fn browser_cache_keys(
    cache: &web_sys::Cache,
    cache_name: &'static str,
) -> Result<BTreeSet<String>, ArtifactStreamError> {
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;

    let value = JsFuture::from(cache.keys())
        .await
        .map_err(|value| browser_cache_operation_error(cache_name, "list keys", value))?;
    let requests = value.dyn_into::<js_sys::Array>().map_err(|value| {
        browser_cache_operation_error(cache_name, "keys result conversion", value)
    })?;
    requests
        .iter()
        .map(|value| {
            value
                .dyn_into::<web_sys::Request>()
                .map(|request| request.url())
                .map_err(|value| {
                    browser_cache_operation_error(cache_name, "key request conversion", value)
                })
        })
        .collect()
}

#[cfg(any(test, all(target_arch = "wasm32", feature = "boogu-web")))]
fn browser_storage_estimate_bytes(
    field: &'static str,
    value: Option<f64>,
) -> Result<u64, ArtifactStreamError> {
    const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;
    let Some(value) = value else {
        return Err(ArtifactStreamError::BrowserStorageEstimate {
            field,
            actual: None,
        });
    };
    if !value.is_finite() || !(0.0..=MAX_SAFE_INTEGER).contains(&value) {
        return Err(ArtifactStreamError::BrowserStorageEstimate {
            field,
            actual: Some(value.to_string()),
        });
    }
    Ok(value.floor() as u64)
}

#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
fn browser_storage_operation_error(
    operation: &'static str,
    value: wasm_bindgen::JsValue,
) -> ArtifactStreamError {
    ArtifactStreamError::BrowserStorageOperation {
        operation,
        message: browser_js_message(value),
    }
}

#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
fn browser_storage_estimate_field(
    estimate: &wasm_bindgen::JsValue,
    field: &'static str,
) -> Result<u64, ArtifactStreamError> {
    let value = js_sys::Reflect::get(estimate, &wasm_bindgen::JsValue::from_str(field))
        .map_err(|value| browser_storage_operation_error("read estimate field", value))?;
    browser_storage_estimate_bytes(field, value.as_f64())
}

#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
async fn request_browser_persistent_storage(
    storage: &web_sys::StorageManager,
) -> Result<bool, ArtifactStreamError> {
    use wasm_bindgen_futures::JsFuture;

    let persisted = JsFuture::from(
        storage
            .persisted()
            .map_err(|value| browser_storage_operation_error("query persistence", value))?,
    )
    .await
    .map_err(|value| browser_storage_operation_error("query persistence", value))?
    .as_bool()
    .ok_or_else(|| ArtifactStreamError::BrowserStorageOperation {
        operation: "query persistence",
        message: "StorageManager.persisted() returned a non-boolean value".into(),
    })?;
    if persisted {
        return Ok(true);
    }
    JsFuture::from(
        storage
            .persist()
            .map_err(|value| browser_storage_operation_error("request persistence", value))?,
    )
    .await
    .map_err(|value| browser_storage_operation_error("request persistence", value))?
    .as_bool()
    .ok_or_else(|| ArtifactStreamError::BrowserStorageOperation {
        operation: "request persistence",
        message: "StorageManager.persist() returned a non-boolean value".into(),
    })
}

/// Fail before downloading model weights when this origin cannot hold the exact missing cache
/// closure. Existing entries are recognized by the same URL/digest/size keys used by reads.
#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
pub(crate) async fn preflight_browser_persistent_cache(
    plan: &BrowserPersistentCachePlan,
) -> Result<BrowserPersistentCachePreflight, ArtifactStreamError> {
    use wasm_bindgen_futures::JsFuture;

    let total_entries = plan.entry_count();
    let total_bytes = plan.total_bytes();
    if total_entries == 0 || total_bytes == 0 {
        return Err(ArtifactStreamError::BrowserPersistentCachePlan(
            "selected model produced an empty cache closure".into(),
        ));
    }

    let mut cached_entries = 0_u64;
    let mut cached_bytes = 0_u64;
    for (cache_name, expected) in &plan.entries {
        let cache = open_browser_artifact_cache(cache_name).await?;
        let present = browser_cache_keys(&cache, cache_name).await?;
        for (key, size) in expected {
            if present.contains(key) {
                cached_entries = cached_entries.saturating_add(1);
                cached_bytes = cached_bytes.saturating_add(*size);
            }
        }
    }
    let missing_entries = total_entries.saturating_sub(cached_entries);
    let missing_bytes = total_bytes.saturating_sub(cached_bytes);
    let reserve_bytes = browser_persistent_cache_reserve(missing_bytes);
    let required_available = missing_bytes.checked_add(reserve_bytes).ok_or_else(|| {
        ArtifactStreamError::BrowserPersistentCachePlan(
            "selected model cache byte requirement overflowed u64".into(),
        )
    })?;

    let window = web_sys::window().ok_or(ArtifactStreamError::BrowserWindowUnavailable)?;
    let storage = window.navigator().storage();
    let estimate_value = JsFuture::from(
        storage
            .estimate()
            .map_err(|value| browser_storage_operation_error("estimate quota", value))?,
    )
    .await
    .map_err(|value| browser_storage_operation_error("estimate quota", value))?;
    // `StorageEstimate` is a WebIDL dictionary returned as a plain JavaScript object, not a
    // constructible browser class. A checked `JsCast::dyn_into` therefore rejects valid Chrome
    // results. Read its numeric dictionary fields directly and retain the finite/safe-integer
    // validation before converting them to Rust byte counts.
    let storage_usage_bytes = browser_storage_estimate_field(&estimate_value, "usage")?;
    let storage_quota_bytes = browser_storage_estimate_field(&estimate_value, "quota")?;
    let storage_available_bytes = storage_quota_bytes.saturating_sub(storage_usage_bytes);
    if storage_available_bytes < required_available {
        return Err(ArtifactStreamError::BrowserStorageQuotaInsufficient {
            available_bytes: storage_available_bytes,
            missing_bytes,
            reserve_bytes,
            cached_entries,
            total_entries,
        });
    }

    let persistent_storage_granted = match request_browser_persistent_storage(&storage).await {
        Ok(granted) => granted,
        Err(error) => {
            web_sys::console::warn_1(
                &format!(
                    "burn_image could not request eviction-resistant origin storage; the verified Cache Storage entries remain best-effort: {error}"
                )
                .into(),
            );
            false
        }
    };
    Ok(BrowserPersistentCachePreflight {
        total_entries,
        cached_entries,
        missing_entries,
        missing_bytes,
        storage_available_bytes,
        persistent_storage_granted,
    })
}

#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
fn browser_cache_operation_error(
    cache: &'static str,
    operation: &'static str,
    value: wasm_bindgen::JsValue,
) -> ArtifactStreamError {
    ArtifactStreamError::BrowserCacheOperation {
        cache,
        operation,
        message: browser_js_message(value),
    }
}

#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
fn browser_js_message(value: wasm_bindgen::JsValue) -> String {
    value.as_string().unwrap_or_else(|| format!("{value:?}"))
}

/// Fetch an initially unknown-size compact browser file as one bounded response.
///
/// This is used to bootstrap `manifest.json`. The request bypasses the browser
/// HTTP cache so a mutable manifest URL cannot retain an old sealed release.
/// Any exposed `Content-Length` is checked before the browser-owned Blob; the
/// Blob hard cap is always checked before its one bounded copy into Wasm.
#[cfg(target_arch = "wasm32")]
pub async fn fetch_browser_bounded_file(
    base_url: &RemoteBaseUrl,
    path: ArtifactPath,
    maximum_bytes: u64,
    _config: ArtifactStreamConfig,
) -> Result<Vec<u8>, ArtifactStreamError> {
    fetch_browser_complete_file(base_url, &path, None, maximum_bytes, true).await
}

/// Fetch and authenticate the manifest-declared transport layout.
///
/// A missing declaration, unavailable sidecar, different HTTP object size, sidecar digest
/// mismatch, malformed JSON, or invalid logical-to-physical mapping fails before a browser stage
/// reader is constructed.
#[cfg(target_arch = "wasm32")]
pub async fn fetch_browser_transport_layout(
    base_url: &RemoteBaseUrl,
    manifest: &ArtifactManifest,
    config: ArtifactStreamConfig,
) -> Result<VerifiedArtifactTransportLayout, ArtifactStreamError> {
    let declared = ArtifactTransportLayout::declared_file(manifest)
        .map_err(|error| ArtifactStreamError::BrowserTransportLayout(error.to_string()))?
        .ok_or_else(|| {
            ArtifactStreamError::BrowserTransportLayout(
                "browser manifest omits its sealed transport layout".into(),
            )
        })?;
    let bytes = fetch_browser_declared_file(
        base_url,
        declared,
        MAX_ARTIFACT_TRANSPORT_LAYOUT_BYTES,
        config,
    )
    .await?;
    ArtifactTransportLayout::parse_and_validate(manifest, &bytes)
        .map_err(|error| ArtifactStreamError::BrowserTransportLayout(error.to_string()))
}

#[cfg(target_arch = "wasm32")]
async fn fetch_browser_declared_file(
    base_url: &RemoteBaseUrl,
    file: &ArtifactFile,
    maximum_bytes: u64,
    _config: ArtifactStreamConfig,
) -> Result<Vec<u8>, ArtifactStreamError> {
    if file.size == 0 || file.size > maximum_bytes {
        return Err(ArtifactStreamError::BrowserFileTooLarge {
            actual: file.size,
            maximum: maximum_bytes,
        });
    }
    fetch_browser_complete_file(base_url, &file.path, Some(file.size), maximum_bytes, false).await
}

#[cfg(target_arch = "wasm32")]
async fn fetch_browser_complete_file(
    base_url: &RemoteBaseUrl,
    path: &ArtifactPath,
    expected_bytes: Option<u64>,
    maximum_bytes: u64,
    bypass_http_cache: bool,
) -> Result<Vec<u8>, ArtifactStreamError> {
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{Request, RequestInit, Response};

    let url = base_url.resolve(path);
    let init = RequestInit::new();
    init.set_method("GET");
    if bypass_http_cache {
        init.set_cache(web_sys::RequestCache::NoStore);
    }
    let request = Request::new_with_str_and_init(&url, &init).map_err(browser_request_error)?;
    let window = web_sys::window().ok_or(ArtifactStreamError::BrowserWindowUnavailable)?;
    let response = JsFuture::from(window.fetch_with_request(&request))
        .await
        .map_err(browser_request_error)?
        .dyn_into::<Response>()
        .map_err(browser_request_error)?;
    if response.status() != 200 {
        return Err(ArtifactStreamError::BrowserCompleteObjectHttpStatus {
            status: response.status(),
            url,
        });
    }
    let content_length = response
        .headers()
        .get("Content-Length")
        .map_err(browser_request_error)?;
    match expected_bytes {
        Some(expected) => {
            validate_browser_content_length_if_exposed(expected, content_length.as_deref())?;
        }
        None => {
            if let Some(content_length) = content_length.as_deref() {
                let total = parse_browser_content_length(Some(content_length))?;
                validate_browser_complete_object_size(total, maximum_bytes)?;
            }
        }
    }
    read_browser_complete_response_body_bounded(&response, expected_bytes, maximum_bytes).await
}

#[cfg(target_arch = "wasm32")]
async fn read_browser_complete_response_body_bounded(
    response: &web_sys::Response,
    expected_bytes: Option<u64>,
    maximum_bytes: u64,
) -> Result<Vec<u8>, ArtifactStreamError> {
    use js_sys::Uint8Array;
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;

    let content_length = response
        .headers()
        .get("Content-Length")
        .map_err(browser_request_error)?;
    if let Some(expected_bytes) = expected_bytes {
        validate_browser_content_length_if_exposed(expected_bytes, content_length.as_deref())?;
    } else if let Some(content_length) = content_length.as_deref() {
        let declared = parse_browser_content_length(Some(content_length))?;
        validate_browser_complete_object_size(declared, maximum_bytes)?;
    }
    let content_encoding = response
        .headers()
        .get("Content-Encoding")
        .map_err(browser_request_error)?;
    validate_browser_content_encoding(content_encoding.as_deref())?;

    // The live CDN intentionally allows cross-origin reads but can omit
    // Access-Control-Expose-Headers. Content-Length is CORS-safelisted in most
    // browsers, yet Chrome can still hide it on a cached complete response.
    // Keep the network transaction in browser-owned storage, reject any
    // exposed framing mismatch early, and always enforce the sealed size (or
    // compact-file hard maximum) on Blob.size before the single Wasm copy.
    let blob = JsFuture::from(response.blob().map_err(browser_request_error)?)
        .await
        .map_err(browser_request_error)?
        .dyn_into::<web_sys::Blob>()
        .map_err(browser_request_error)?;
    let actual_bytes = blob.size() as u64;
    validate_browser_complete_object_size(actual_bytes, maximum_bytes)?;
    if let Some(expected_bytes) = expected_bytes {
        validate_browser_response_size(expected_bytes, actual_bytes)?;
    }
    let buffer = JsFuture::from(blob.array_buffer())
        .await
        .map_err(browser_request_error)?;
    let bytes = Uint8Array::new(&buffer);
    validate_browser_response_size(actual_bytes, u64::from(bytes.length()))?;
    Ok(bytes.to_vec())
}

#[cfg(target_arch = "wasm32")]
fn browser_request_error(value: wasm_bindgen::JsValue) -> ArtifactStreamError {
    ArtifactStreamError::BrowserRequest(value.as_string().unwrap_or_else(|| format!("{value:?}")))
}

#[cfg(target_arch = "wasm32")]
async fn read_browser_response_body_bounded(
    response: &web_sys::Response,
    expected_bytes: u64,
) -> Result<Vec<u8>, ArtifactStreamError> {
    use js_sys::Uint8Array;
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;

    let content_length = response
        .headers()
        .get("Content-Length")
        .map_err(browser_request_error)?;
    validate_browser_content_length(expected_bytes, content_length.as_deref())?;
    let content_encoding = response
        .headers()
        .get("Content-Encoding")
        .map_err(browser_request_error)?;
    validate_browser_content_encoding(content_encoding.as_deref())?;

    // Let Fetch finish the HTTP transaction into browser-owned storage. Chrome
    // reports manually drained 206 ReadableStreams as canceled even after all
    // bytes arrive, which makes every non-final range look like a network
    // failure. Exact identity framing bounds this Blob before any allocation in
    // Wasm linear memory; the Blob size is checked before `array_buffer()` makes
    // the single bounded copy into Wasm.
    let blob = JsFuture::from(response.blob().map_err(browser_request_error)?)
        .await
        .map_err(browser_request_error)?
        .dyn_into::<web_sys::Blob>()
        .map_err(browser_request_error)?;
    let blob_size = blob.size();
    if blob_size != expected_bytes as f64 {
        return Err(ArtifactStreamError::BrowserResponseSize {
            expected: expected_bytes,
            actual: blob_size as u64,
        });
    }
    let buffer = JsFuture::from(blob.array_buffer())
        .await
        .map_err(browser_request_error)?;
    let bytes = Uint8Array::new(&buffer);
    validate_browser_response_size(expected_bytes, u64::from(bytes.length()))?;
    Ok(bytes.to_vec())
}

#[cfg(any(target_arch = "wasm32", test))]
fn validate_browser_content_length(
    expected: u64,
    actual: Option<&str>,
) -> Result<(), ArtifactStreamError> {
    let valid = actual.is_some_and(|actual| {
        actual
            .parse::<u64>()
            .is_ok_and(|parsed| parsed == expected && parsed.to_string() == actual)
    });
    if !valid {
        return Err(ArtifactStreamError::BrowserContentLength {
            expected,
            actual: actual.map(str::to_owned),
        });
    }
    Ok(())
}

#[cfg(any(target_arch = "wasm32", test))]
fn validate_browser_content_length_if_exposed(
    expected: u64,
    actual: Option<&str>,
) -> Result<(), ArtifactStreamError> {
    match actual {
        Some(actual) => validate_browser_content_length(expected, Some(actual)),
        None => Ok(()),
    }
}

#[cfg(any(target_arch = "wasm32", test))]
fn parse_browser_content_length(actual: Option<&str>) -> Result<u64, ArtifactStreamError> {
    let parsed = actual.and_then(|value| value.parse::<u64>().ok());
    match (actual, parsed) {
        (Some(raw), Some(value)) if value.to_string() == raw => Ok(value),
        _ => Err(ArtifactStreamError::BrowserMalformedContentLength {
            actual: actual.map(str::to_owned),
        }),
    }
}

#[cfg(any(target_arch = "wasm32", test))]
fn validate_browser_content_encoding(actual: Option<&str>) -> Result<(), ArtifactStreamError> {
    if let Some(actual) = actual
        && !actual.eq_ignore_ascii_case("identity")
    {
        return Err(ArtifactStreamError::BrowserContentEncoding {
            actual: actual.into(),
        });
    }
    Ok(())
}

#[cfg(any(target_arch = "wasm32", test))]
fn validate_browser_response_size(expected: u64, actual: u64) -> Result<(), ArtifactStreamError> {
    if actual != expected {
        return Err(ArtifactStreamError::BrowserResponseSize { expected, actual });
    }
    Ok(())
}

#[cfg(any(target_arch = "wasm32", test))]
fn validate_browser_complete_object_size(
    actual: u64,
    maximum: u64,
) -> Result<(), ArtifactStreamError> {
    if actual == 0 || actual > maximum {
        return Err(ArtifactStreamError::BrowserFileTooLarge { actual, maximum });
    }
    Ok(())
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
fn validate_content_range_exact_total(
    range: ByteRange,
    actual: Option<&str>,
    expected_total: u64,
) -> Result<(), ArtifactStreamError> {
    let expected = format!(
        "bytes {}-{}/{expected_total}",
        range.offset(),
        range.end_exclusive() - 1
    );
    let valid = parse_content_range(actual).is_ok_and(|(start, end, total)| {
        start == range.offset() && end == range.end_exclusive() - 1 && total == expected_total
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

#[cfg(any(test, all(target_arch = "wasm32", feature = "boogu-web")))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BrowserPlannedTransportPart {
    size: u64,
    ranges: u64,
}

/// Monotonic progress through the complete registered transport closure.
///
/// Semantic stages may read the same logical object more than once. This tracker deliberately
/// counts unique qualified logical objects, physical parts, and bounded ranges, which keeps the
/// denominator stable while a stage source reconstructs or revisits an object.
#[cfg(any(test, all(target_arch = "wasm32", feature = "boogu-web")))]
#[derive(Debug)]
struct BrowserArtifactTransferTracker {
    phase: String,
    current_component: Option<ArtifactComponentId>,
    logical_objects: BTreeMap<ArtifactPath, u64>,
    logical_parts: BTreeMap<ArtifactPath, BTreeSet<ArtifactPath>>,
    physical_parts: BTreeMap<ArtifactPath, BrowserPlannedTransportPart>,
    completed_logical_objects: BTreeSet<ArtifactPath>,
    completed_physical_parts: BTreeSet<ArtifactPath>,
    completed_ranges: BTreeSet<(ArtifactPath, u64, u64)>,
    loaded_bytes: u64,
    total_bytes: u64,
    total_ranges: u64,
    last_rate_sample: Option<(f64, u64)>,
    smoothed_bytes_per_second: Option<f64>,
    rate_sample_count: u32,
    request_activity: Option<ArtifactRequestTransferActivity>,
}

#[cfg(any(test, all(target_arch = "wasm32", feature = "boogu-web")))]
impl Default for BrowserArtifactTransferTracker {
    fn default() -> Self {
        Self {
            phase: "Model setup".into(),
            current_component: None,
            logical_objects: BTreeMap::new(),
            logical_parts: BTreeMap::new(),
            physical_parts: BTreeMap::new(),
            completed_logical_objects: BTreeSet::new(),
            completed_physical_parts: BTreeSet::new(),
            completed_ranges: BTreeSet::new(),
            loaded_bytes: 0,
            total_bytes: 0,
            total_ranges: 0,
            last_rate_sample: None,
            smoothed_bytes_per_second: None,
            rate_sample_count: 0,
            request_activity: None,
        }
    }
}

#[cfg(any(test, all(target_arch = "wasm32", feature = "boogu-web")))]
impl BrowserArtifactTransferTracker {
    const MIN_RATE_SAMPLE_INTERVAL_MILLIS: f64 = 250.0;
    const MIN_VISIBLE_RATE_SAMPLES: u32 = 3;
    const RATE_SMOOTHING_ALPHA: f64 = 0.25;

    fn set_phase(&mut self, phase: impl Into<String>) {
        self.phase = phase.into();
    }

    fn start_request_activity(&mut self) {
        self.request_activity = (self.total_bytes > 0 && self.loaded_bytes >= self.total_bytes)
            .then(|| ArtifactRequestTransferActivity {
                phase: "Applying verified cached model stages".into(),
                current_path: None,
                component: None,
                logical_objects_completed: 0,
                bounded_ranges_processed: 0,
                processed_bytes: 0,
            });
    }

    fn register_logical_object(&mut self, path: ArtifactPath, size: u64) -> Result<(), String> {
        match self.logical_objects.entry(path.clone()) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(size);
                Ok(())
            }
            std::collections::btree_map::Entry::Occupied(entry) if *entry.get() == size => Ok(()),
            std::collections::btree_map::Entry::Occupied(entry) => Err(format!(
                "browser transfer logical object {path} was registered with sizes {} and {size}",
                entry.get()
            )),
        }
    }

    fn register_physical_part(
        &mut self,
        path: ArtifactPath,
        size: u64,
        chunk_bytes: u64,
    ) -> Result<(), String> {
        if size == 0 || chunk_bytes == 0 {
            return Err(format!(
                "browser transfer physical part {path} has invalid size/chunk {size}/{chunk_bytes}"
            ));
        }
        let ranges = size.div_ceil(chunk_bytes);
        let part = BrowserPlannedTransportPart { size, ranges };
        match self.physical_parts.entry(path.clone()) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(part);
                self.total_bytes = self.total_bytes.saturating_add(size);
                self.total_ranges = self.total_ranges.saturating_add(ranges);
                Ok(())
            }
            std::collections::btree_map::Entry::Occupied(entry) if *entry.get() == part => Ok(()),
            std::collections::btree_map::Entry::Occupied(entry) => Err(format!(
                "browser transfer physical part {path} was registered with conflicting plans {:?} and {part:?}",
                entry.get()
            )),
        }
    }

    fn register_manifest_plan(
        &mut self,
        manifest: &ArtifactManifest,
        bundle: Option<&ArtifactBundleId>,
        layout: &VerifiedArtifactTransportLayout,
    ) -> Result<(), String> {
        for file in manifest
            .files
            .iter()
            .filter(|file| file.role == ArtifactFileRole::Weights)
        {
            self.register_logical_object(qualified_transfer_path(bundle, &file.path), file.size)?;
        }
        for object in layout.objects() {
            let logical_path = qualified_transfer_path(bundle, &object.path);
            let mut logical_parts = BTreeSet::new();
            for part in &object.parts {
                let part_path = qualified_transfer_path(bundle, &part.path);
                self.register_physical_part(part_path.clone(), part.size, part.size)?;
                logical_parts.insert(part_path);
            }
            self.logical_parts
                .entry(logical_path)
                .or_default()
                .extend(logical_parts);
        }
        Ok(())
    }

    fn retain_logical_objects(&mut self, active: &BTreeSet<ArtifactPath>) -> Result<(), String> {
        if !self.completed_logical_objects.is_empty()
            || !self.completed_physical_parts.is_empty()
            || !self.completed_ranges.is_empty()
            || self.loaded_bytes != 0
        {
            return Err(
                "browser active transfer plan was selected after payload loading began".into(),
            );
        }
        let known = self
            .logical_objects
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();
        let unknown = active.difference(&known).cloned().collect::<Vec<_>>();
        if !unknown.is_empty() {
            return Err(format!(
                "browser active transfer plan names unknown logical objects {unknown:?}"
            ));
        }
        self.logical_objects.retain(|path, _| active.contains(path));
        self.logical_parts.retain(|path, _| active.contains(path));
        let active_parts = self
            .logical_parts
            .values()
            .flat_map(|parts| parts.iter().cloned())
            .collect::<BTreeSet<_>>();
        self.physical_parts
            .retain(|path, _| active_parts.contains(path));
        self.total_bytes = self
            .physical_parts
            .values()
            .fold(0_u64, |total, part| total.saturating_add(part.size));
        self.total_ranges = self
            .physical_parts
            .values()
            .fold(0_u64, |total, part| total.saturating_add(part.ranges));
        Ok(())
    }

    fn object_started(&mut self, file: &ArtifactFile) {
        self.current_component = file.component.clone();
        if let Some(activity) = &mut self.request_activity {
            activity.current_path = Some(file.path.clone());
            activity.component = file.component.clone();
        }
    }

    fn record_bounded_range(
        &mut self,
        path: ArtifactPath,
        offset: u64,
        length: u64,
        now_millis: f64,
    ) {
        let Some(part) = self.physical_parts.get(&path).copied() else {
            return;
        };
        let Some(end) = offset.checked_add(length) else {
            return;
        };
        if length == 0 || end > part.size {
            return;
        }
        if let Some(activity) = &mut self.request_activity {
            activity.bounded_ranges_processed = activity.bounded_ranges_processed.saturating_add(1);
            activity.processed_bytes = activity.processed_bytes.saturating_add(length);
        }
        if self.completed_ranges.insert((path, offset, length)) {
            self.loaded_bytes = self.loaded_bytes.saturating_add(length);
            self.sample_rate(now_millis);
        }
    }

    fn physical_part_verified(&mut self, path: ArtifactPath) {
        if self.physical_parts.contains_key(&path) {
            self.completed_physical_parts.insert(path);
        }
    }

    fn logical_object_verified(&mut self, path: ArtifactPath) {
        if self.logical_objects.contains_key(&path) {
            self.completed_logical_objects.insert(path);
            if let Some(activity) = &mut self.request_activity {
                activity.logical_objects_completed =
                    activity.logical_objects_completed.saturating_add(1);
            }
        }
    }

    fn sample_rate(&mut self, now_millis: f64) {
        let Some((last_millis, last_bytes)) = self.last_rate_sample else {
            self.last_rate_sample = Some((now_millis, self.loaded_bytes));
            return;
        };
        let elapsed_millis = now_millis - last_millis;
        if elapsed_millis < Self::MIN_RATE_SAMPLE_INTERVAL_MILLIS {
            return;
        }
        let bytes = self.loaded_bytes.saturating_sub(last_bytes);
        self.last_rate_sample = Some((now_millis, self.loaded_bytes));
        if bytes == 0 || !elapsed_millis.is_finite() || elapsed_millis <= 0.0 {
            return;
        }
        let instantaneous = bytes as f64 * 1_000.0 / elapsed_millis;
        self.smoothed_bytes_per_second = Some(self.smoothed_bytes_per_second.map_or(
            instantaneous,
            |previous| {
                previous * (1.0 - Self::RATE_SMOOTHING_ALPHA)
                    + instantaneous * Self::RATE_SMOOTHING_ALPHA
            },
        ));
        self.rate_sample_count = self.rate_sample_count.saturating_add(1);
    }

    fn snapshot(&self) -> Option<ArtifactTransferProgress> {
        if self.total_bytes == 0 {
            return None;
        }
        let bytes_per_second = (self.rate_sample_count >= Self::MIN_VISIBLE_RATE_SAMPLES)
            .then_some(self.smoothed_bytes_per_second)
            .flatten()
            .filter(|rate| rate.is_finite() && *rate > 0.0)
            .map(|rate| rate.round() as u64);
        let eta_seconds = bytes_per_second.and_then(|rate| {
            (rate > 0 && self.loaded_bytes < self.total_bytes)
                .then(|| (self.total_bytes - self.loaded_bytes).div_ceil(rate))
        });
        Some(ArtifactTransferProgress {
            phase: self.phase.clone(),
            component: self.current_component.clone(),
            logical_objects_completed: u32::try_from(self.completed_logical_objects.len())
                .unwrap_or(u32::MAX),
            logical_objects_total: u32::try_from(self.logical_objects.len()).unwrap_or(u32::MAX),
            physical_parts_completed: u32::try_from(self.completed_physical_parts.len())
                .unwrap_or(u32::MAX),
            physical_parts_total: u32::try_from(self.physical_parts.len()).unwrap_or(u32::MAX),
            bounded_ranges_completed: u64::try_from(self.completed_ranges.len())
                .unwrap_or(u64::MAX),
            bounded_ranges_total: self.total_ranges,
            loaded_bytes: self.loaded_bytes.min(self.total_bytes),
            total_bytes: self.total_bytes,
            bytes_per_second,
            eta_seconds,
            request_activity: self.request_activity.clone(),
        })
    }
}

#[cfg(any(test, all(target_arch = "wasm32", feature = "boogu-web")))]
fn qualified_transfer_path(bundle: Option<&ArtifactBundleId>, path: &ArtifactPath) -> ArtifactPath {
    bundle.map_or_else(
        || path.clone(),
        |bundle| {
            ArtifactPath::new(format!("{bundle}/{path}"))
                .expect("validated bundle and artifact paths compose into a valid path")
        },
    )
}

#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
fn browser_progress_now_millis() -> f64 {
    js_sys::Date::now()
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
    transfer: BrowserArtifactTransferTracker,
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

    pub fn set_transfer_phase(&self, phase: impl Into<String>) {
        self.inner
            .lock()
            .expect("browser artifact control mutex poisoned")
            .transfer
            .set_phase(phase);
    }

    pub fn start_request_transfer_activity(&self) {
        self.inner
            .lock()
            .expect("browser artifact control mutex poisoned")
            .transfer
            .start_request_activity();
    }

    pub fn transfer_progress(&self) -> Option<ArtifactTransferProgress> {
        self.inner
            .lock()
            .expect("browser artifact control mutex poisoned")
            .transfer
            .snapshot()
    }

    pub fn retain_transfer_logical_objects(
        &self,
        active: &BTreeSet<ArtifactPath>,
    ) -> Result<(), BooguError> {
        self.inner
            .lock()
            .expect("browser artifact control mutex poisoned")
            .transfer
            .retain_logical_objects(active)
            .map_err(BooguError::Artifact)
    }

    fn register_manifest_transfer_plan(
        &self,
        manifest: &ArtifactManifest,
        bundle: Option<&ArtifactBundleId>,
        layout: &VerifiedArtifactTransportLayout,
    ) -> Result<(), BooguError> {
        let mut state = self
            .inner
            .lock()
            .expect("browser artifact control mutex poisoned");
        state
            .transfer
            .register_manifest_plan(manifest, bundle, layout)
            .map_err(BooguError::Artifact)
    }

    fn record_transport_range(&self, path: ArtifactPath, range: ByteRange) {
        self.inner
            .lock()
            .expect("browser artifact control mutex poisoned")
            .transfer
            .record_bounded_range(
                path,
                range.offset(),
                range.length(),
                browser_progress_now_millis(),
            );
    }

    fn record_transport_part_verified(&self, path: ArtifactPath) {
        self.inner
            .lock()
            .expect("browser artifact control mutex poisoned")
            .transfer
            .physical_part_verified(path);
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
                state.transfer.object_started(file);
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
                state.transfer.logical_object_verified(path.clone());
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

#[cfg(test)]
fn verify_browser_transport_part_bytes(
    part: &ArtifactTransportPart,
    bytes: &[u8],
) -> Result<(), ArtifactStreamError> {
    let maximum = ARTIFACT_TRANSPORT_TARGET_PART_BYTES.min(ARTIFACT_TRANSPORT_MAX_PART_BYTES);
    if part.size == 0 {
        return Err(ArtifactStreamError::BrowserTransportLayout(format!(
            "verified transport part {} has zero bytes",
            part.path
        )));
    }
    if part.size > maximum {
        return Err(ArtifactStreamError::BrowserTransportPartTooLarge {
            path: part.path.clone(),
            actual: part.size,
            maximum,
        });
    }
    let actual_size = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    if actual_size != part.size {
        return Err(ArtifactStreamError::BrowserTransportPartSize {
            path: part.path.clone(),
            expected: part.size,
            actual: actual_size,
        });
    }
    let actual = Sha256Digest::calculate(bytes);
    if actual != part.sha256 {
        return Err(ArtifactStreamError::BrowserTransportPartIntegrity {
            path: part.path.clone(),
            expected: part.sha256,
            actual,
        });
    }
    Ok(())
}

/// Authenticate one bounded physical part with the browser's native Web Crypto implementation.
///
/// Cache hits previously ran both the per-part and reconstructed-logical SHA-256 loops on the
/// Wasm main thread. Web Crypto keeps the independent part seal while yielding to the browser;
/// the complete logical Burnpack is still hashed in Rust before any tensor parser sees it.
#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
async fn verify_browser_transport_part_bytes_async(
    part: &ArtifactTransportPart,
    bytes: &[u8],
) -> Result<(), ArtifactStreamError> {
    let maximum = ARTIFACT_TRANSPORT_TARGET_PART_BYTES.min(ARTIFACT_TRANSPORT_MAX_PART_BYTES);
    if part.size == 0 || part.size > maximum {
        return Err(ArtifactStreamError::BrowserTransportPartTooLarge {
            path: part.path.clone(),
            actual: part.size,
            maximum,
        });
    }
    let actual_size = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    if actual_size != part.size {
        return Err(ArtifactStreamError::BrowserTransportPartSize {
            path: part.path.clone(),
            expected: part.size,
            actual: actual_size,
        });
    }
    let window = web_sys::window().ok_or(ArtifactStreamError::BrowserWindowUnavailable)?;
    let crypto =
        window
            .crypto()
            .map_err(|error| ArtifactStreamError::BrowserTransportPartCrypto {
                path: part.path.clone(),
                message: format!("Crypto is unavailable: {error:?}"),
            })?;
    let promise = crypto
        .subtle()
        .digest_with_str_and_u8_array("SHA-256", bytes)
        .map_err(|error| ArtifactStreamError::BrowserTransportPartCrypto {
            path: part.path.clone(),
            message: format!("digest could not start: {error:?}"),
        })?;
    let value = wasm_bindgen_futures::JsFuture::from(promise)
        .await
        .map_err(|error| ArtifactStreamError::BrowserTransportPartCrypto {
            path: part.path.clone(),
            message: format!("digest rejected: {error:?}"),
        })?;
    let digest = js_sys::Uint8Array::new(&value);
    if digest.length() != 32 {
        return Err(ArtifactStreamError::BrowserTransportPartCryptoSize {
            path: part.path.clone(),
            actual: digest.length(),
        });
    }
    let mut digest_bytes = [0u8; 32];
    digest.copy_to(&mut digest_bytes);
    let actual = Sha256Digest::from_bytes(digest_bytes);
    if actual != part.sha256 {
        return Err(ArtifactStreamError::BrowserTransportPartIntegrity {
            path: part.path.clone(),
            expected: part.sha256,
            actual,
        });
    }
    Ok(())
}

#[cfg(any(test, all(target_arch = "wasm32", feature = "boogu-web")))]
fn validate_browser_transport_part_offset(
    file: &ArtifactFile,
    part: &ArtifactTransportPart,
    reconstructed_bytes: usize,
) -> Result<(), ArtifactStreamError> {
    let reconstructed_offset = u64::try_from(reconstructed_bytes).map_err(|_| {
        ArtifactStreamError::BrowserTransportLayout(format!(
            "browser transport reconstruction offset overflowed for {}",
            file.path
        ))
    })?;
    if reconstructed_offset != part.offset {
        return Err(ArtifactStreamError::BrowserTransportLayout(format!(
            "verified part {} starts at {}, reconstructed {} bytes for {}",
            part.path, part.offset, reconstructed_offset, file.path
        )));
    }
    Ok(())
}

#[cfg(test)]
fn verify_browser_transport_reconstruction(
    file: &ArtifactFile,
    bytes: &[u8],
) -> Result<(), ArtifactStreamError> {
    let actual_size = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    if actual_size != file.size {
        return Err(ArtifactStreamError::BrowserTransportReconstructionSize {
            path: file.path.clone(),
            expected: file.size,
            actual: actual_size,
        });
    }
    let actual = Sha256Digest::calculate(bytes);
    if actual != file.sha256 {
        return Err(
            ArtifactStreamError::BrowserTransportReconstructionIntegrity {
                path: file.path.clone(),
                expected: file.sha256,
                actual,
            },
        );
    }
    Ok(())
}

/// Browser reader for one sealed semantic Burnpack at a time.
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
    transport_layout: Option<Arc<VerifiedArtifactTransportLayout>>,
    cache_policy: BrowserRangeCachePolicy,
    part_cache: Option<web_sys::Cache>,
}

#[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
impl BrowserStageShardReader {
    pub fn new(base_url: RemoteBaseUrl, config: ArtifactStreamConfig) -> Self {
        Self {
            source: ArtifactSource::Remote { base_url },
            config,
            control: BrowserArtifactControl::default(),
            progress_bundle: None,
            transport_layout: None,
            cache_policy: BrowserRangeCachePolicy::Disabled,
            part_cache: None,
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
            transport_layout: None,
            cache_policy: BrowserRangeCachePolicy::Disabled,
            part_cache: None,
        }
    }

    /// Construct an unqualified standalone reader that shares progress and cancellation routing
    /// with sibling reader wrappers.
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
            transport_layout: None,
            cache_policy: BrowserRangeCachePolicy::Disabled,
            part_cache: None,
        }
    }

    /// Bind a layout that already carries model-neutral verification typestate.
    /// The manifest digest check prevents accidentally pairing a valid layout
    /// from a sibling bundle or prior immutable release with this reader.
    pub fn with_verified_transport_layout(
        mut self,
        manifest: &ArtifactManifest,
        layout: VerifiedArtifactTransportLayout,
    ) -> Result<Self, BooguError> {
        ArtifactTransportLayout::declared_file(manifest)
            .map_err(|error| BooguError::Artifact(error.to_string()))?
            .ok_or_else(|| {
                BooguError::Artifact(
                    "cannot attach a transport layout to a manifest without a declaration".into(),
                )
            })?;
        let expected_digest = manifest.content_digest.ok_or_else(|| {
            BooguError::Artifact("browser transport manifest is not sealed".into())
        })?;
        if layout.manifest_content_digest() != expected_digest {
            return Err(BooguError::Artifact(format!(
                "browser transport layout is bound to manifest {}, expected {}",
                layout.manifest_content_digest(),
                expected_digest
            )));
        }
        self.control.register_manifest_transfer_plan(
            manifest,
            self.progress_bundle.as_ref(),
            &layout,
        )?;
        self.transport_layout = Some(Arc::new(layout));
        Ok(self)
    }

    /// Bootstrap this reader from its sealed manifest. The declared layout is fetched and
    /// authenticated before this method returns.
    pub async fn with_manifest_transport_layout(
        self,
        manifest: &ArtifactManifest,
    ) -> Result<Self, BooguError> {
        let base_url = match &self.source {
            ArtifactSource::Remote { base_url } => base_url.clone(),
            ArtifactSource::LocalDirectory { .. } => {
                return Err(BooguError::Artifact(
                    ArtifactStreamError::LocalBrowserSource.to_string(),
                ));
            }
        };
        let layout = fetch_browser_transport_layout(&base_url, manifest, self.config)
            .await
            .map_err(|error| BooguError::Artifact(error.to_string()))?;
        self.with_verified_transport_layout(manifest, layout)
    }

    pub fn control(&self) -> BrowserArtifactControl {
        self.control.clone()
    }

    /// Require verified Cache Storage entries, one <=20 MiB entry per physical transport part.
    /// Any cache or quota failure aborts instead of silently repeating network traffic.
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

    /// Add only this reader's active logical weights to the selected-model cache preflight.
    /// Shared physical parts deduplicate by their exact synthetic Cache Storage key.
    pub(crate) fn extend_persistent_cache_plan(
        &self,
        manifest: &ArtifactManifest,
        active: &BTreeSet<ArtifactPath>,
        plan: &mut BrowserPersistentCachePlan,
    ) -> Result<(), BooguError> {
        if self.cache_policy != BrowserRangeCachePolicy::Required {
            return Err(BooguError::Artifact(
                "cannot preflight a browser reader whose persistent cache policy is disabled"
                    .into(),
            ));
        }
        let base_url = match &self.source {
            ArtifactSource::Remote { base_url } => base_url,
            ArtifactSource::LocalDirectory { .. } => {
                return Err(BooguError::Artifact(
                    ArtifactStreamError::LocalBrowserSource.to_string(),
                ));
            }
        };
        let layout = self.transport_layout.as_ref().ok_or_else(|| {
            BooguError::Artifact("browser reader has no verified transport layout".into())
        })?;
        for file in manifest
            .files
            .iter()
            .filter(|file| file.role == ArtifactFileRole::Weights)
        {
            if !active.contains(&self.progress_path(&file.path)) {
                continue;
            }
            let object = layout.object(&file.path).ok_or_else(|| {
                BooguError::Artifact(
                    ArtifactStreamError::BrowserTransportObjectMissing {
                        path: file.path.clone(),
                    }
                    .to_string(),
                )
            })?;
            for part in &object.parts {
                let url = base_url.resolve(&part.path);
                plan.register(
                    BROWSER_ARTIFACT_PART_CACHE_NAME,
                    browser_part_cache_key(&url, part.sha256, part.size),
                    part.size,
                )
                .map_err(|error| BooguError::Artifact(error.to_string()))?;
            }
        }
        Ok(())
    }

    async fn part_cache(&mut self) -> Result<web_sys::Cache, BooguError> {
        if let Some(cache) = &self.part_cache {
            return Ok(cache.clone());
        }
        let cache = open_browser_artifact_cache(BROWSER_ARTIFACT_PART_CACHE_NAME)
            .await
            .map_err(|error| BooguError::Artifact(error.to_string()))?;
        self.part_cache = Some(cache.clone());
        Ok(cache)
    }

    fn complete_object_url(&self, path: &ArtifactPath) -> Result<String, BooguError> {
        match &self.source {
            ArtifactSource::Remote { base_url } => Ok(base_url.resolve(path)),
            ArtifactSource::LocalDirectory { .. } => Err(BooguError::Artifact(
                ArtifactStreamError::LocalBrowserSource.to_string(),
            )),
        }
    }

    async fn fetch_transport_part_complete_attempt(
        &mut self,
        logical_file: &ArtifactFile,
        part: &ArtifactTransportPart,
        force_network: bool,
    ) -> Result<Vec<u8>, BooguError> {
        self.control.check_cancelled()?;
        let base_url = match &self.source {
            ArtifactSource::Remote { base_url } => base_url.clone(),
            ArtifactSource::LocalDirectory { .. } => {
                return Err(BooguError::Artifact(
                    ArtifactStreamError::LocalBrowserSource.to_string(),
                ));
            }
        };
        let url = self.complete_object_url(&part.path)?;
        let key = browser_part_cache_key(&url, part.sha256, part.size);
        let bytes = if self.cache_policy == BrowserRangeCachePolicy::Required {
            let cache = self.part_cache().await?;
            if !force_network {
                let cached =
                    browser_cache_match(&cache, BROWSER_ARTIFACT_PART_CACHE_NAME, &key, part.size)
                        .await
                        .map_err(|error| BooguError::Artifact(error.to_string()))?;
                match cached {
                    Some(bytes) if u64::try_from(bytes.len()).ok() == Some(part.size) => {
                        self.control.record_cache_lookup(Some(part.size), false);
                        self.control.record_logical_range(part.size);
                        bytes
                    }
                    Some(_) => {
                        self.control.record_cache_lookup(None, true);
                        let removed =
                            browser_cache_delete(&cache, BROWSER_ARTIFACT_PART_CACHE_NAME, &key)
                                .await
                                .map_err(|error| BooguError::Artifact(error.to_string()))?;
                        self.control.record_cache_eviction(removed);
                        self.fetch_and_cache_transport_part(&base_url, &cache, &key, part)
                            .await?
                    }
                    None => {
                        self.control.record_cache_lookup(None, false);
                        if self.control.cache_key_was_populated(&key) {
                            return Err(BooguError::Artifact(
                                ArtifactStreamError::BrowserCacheSessionEntryLost {
                                    cache: BROWSER_ARTIFACT_PART_CACHE_NAME,
                                    path: part.path.clone(),
                                    offset: 0,
                                    end_exclusive: part.size,
                                }
                                .to_string(),
                            ));
                        }
                        self.fetch_and_cache_transport_part(&base_url, &cache, &key, part)
                            .await?
                    }
                }
            } else {
                self.fetch_and_cache_transport_part(&base_url, &cache, &key, part)
                    .await?
            }
        } else {
            let bytes = fetch_browser_complete_file(
                &base_url,
                &part.path,
                Some(part.size),
                ARTIFACT_TRANSPORT_TARGET_PART_BYTES.min(ARTIFACT_TRANSPORT_MAX_PART_BYTES),
                false,
            )
            .await
            .map_err(|error| BooguError::Artifact(error.to_string()))?;
            self.control.record_network_fetch(part.size);
            self.control.record_logical_range(part.size);
            bytes
        };

        let range = ByteRange::new(0, part.size)
            .expect("a verified non-empty transport part is one bounded object read");
        self.control
            .record_transport_range(self.progress_path(&part.path), range);
        let loaded_bytes = part.offset.checked_add(part.size).ok_or_else(|| {
            BooguError::Artifact(format!(
                "browser progress byte count overflowed while reading {}",
                part.path
            ))
        })?;
        self.control.push(BrowserArtifactEvent::Progress {
            path: self.progress_path(&logical_file.path),
            loaded_bytes,
            total_bytes: logical_file.size,
        });
        Ok(bytes)
    }

    async fn fetch_and_cache_transport_part(
        &mut self,
        base_url: &RemoteBaseUrl,
        cache: &web_sys::Cache,
        key: &str,
        part: &ArtifactTransportPart,
    ) -> Result<Vec<u8>, BooguError> {
        let bytes = fetch_browser_complete_file(
            base_url,
            &part.path,
            Some(part.size),
            ARTIFACT_TRANSPORT_TARGET_PART_BYTES.min(ARTIFACT_TRANSPORT_MAX_PART_BYTES),
            false,
        )
        .await
        .map_err(|error| BooguError::Artifact(error.to_string()))?;
        self.control.record_network_fetch(part.size);
        browser_cache_put(cache, BROWSER_ARTIFACT_PART_CACHE_NAME, key, &bytes)
            .await
            .map_err(|error| BooguError::Artifact(error.to_string()))?;
        self.control.record_cache_write(key, part.size);
        self.control.record_logical_range(part.size);
        Ok(bytes)
    }

    async fn evict_transport_part(
        &mut self,
        part: &ArtifactTransportPart,
    ) -> Result<(), BooguError> {
        let url = self.complete_object_url(&part.path)?;
        let key = browser_part_cache_key(&url, part.sha256, part.size);
        let cache = self.part_cache().await?;
        let removed = browser_cache_delete(&cache, BROWSER_ARTIFACT_PART_CACHE_NAME, &key)
            .await
            .map_err(|error| BooguError::Artifact(error.to_string()))?;
        self.control.record_cache_eviction(removed);
        Ok(())
    }

    fn protect_verified_transport_part(
        &self,
        part: &ArtifactTransportPart,
    ) -> Result<(), BooguError> {
        if self.cache_policy != BrowserRangeCachePolicy::Required {
            return Ok(());
        }
        let url = self.complete_object_url(&part.path)?;
        self.control
            .protect_verified_cache_key(&browser_part_cache_key(&url, part.sha256, part.size));
        Ok(())
    }

    async fn fetch_direct_complete_file_attempt(
        &mut self,
        file: &ArtifactFile,
        maximum_bytes: u64,
        force_network: bool,
    ) -> Result<Vec<u8>, BooguError> {
        self.control.check_cancelled()?;
        let base_url = match &self.source {
            ArtifactSource::Remote { base_url } => base_url.clone(),
            ArtifactSource::LocalDirectory { .. } => {
                return Err(BooguError::Artifact(
                    ArtifactStreamError::LocalBrowserSource.to_string(),
                ));
            }
        };
        let url = self.complete_object_url(&file.path)?;
        let key = browser_part_cache_key(&url, file.sha256, file.size);
        let bytes = if self.cache_policy == BrowserRangeCachePolicy::Required {
            let cache = self.part_cache().await?;
            if !force_network {
                match browser_cache_match(&cache, BROWSER_ARTIFACT_PART_CACHE_NAME, &key, file.size)
                    .await
                    .map_err(|error| BooguError::Artifact(error.to_string()))?
                {
                    Some(bytes) if u64::try_from(bytes.len()).ok() == Some(file.size) => {
                        self.control.record_cache_lookup(Some(file.size), false);
                        self.control.record_logical_range(file.size);
                        bytes
                    }
                    Some(_) => {
                        self.control.record_cache_lookup(None, true);
                        let removed =
                            browser_cache_delete(&cache, BROWSER_ARTIFACT_PART_CACHE_NAME, &key)
                                .await
                                .map_err(|error| BooguError::Artifact(error.to_string()))?;
                        self.control.record_cache_eviction(removed);
                        self.fetch_and_cache_direct_complete_file(
                            &base_url,
                            &cache,
                            &key,
                            file,
                            maximum_bytes,
                        )
                        .await?
                    }
                    None => {
                        self.control.record_cache_lookup(None, false);
                        if self.control.cache_key_was_populated(&key) {
                            return Err(BooguError::Artifact(
                                ArtifactStreamError::BrowserCacheSessionEntryLost {
                                    cache: BROWSER_ARTIFACT_PART_CACHE_NAME,
                                    path: file.path.clone(),
                                    offset: 0,
                                    end_exclusive: file.size,
                                }
                                .to_string(),
                            ));
                        }
                        self.fetch_and_cache_direct_complete_file(
                            &base_url,
                            &cache,
                            &key,
                            file,
                            maximum_bytes,
                        )
                        .await?
                    }
                }
            } else {
                self.fetch_and_cache_direct_complete_file(
                    &base_url,
                    &cache,
                    &key,
                    file,
                    maximum_bytes,
                )
                .await?
            }
        } else {
            let bytes = fetch_browser_complete_file(
                &base_url,
                &file.path,
                Some(file.size),
                maximum_bytes,
                false,
            )
            .await
            .map_err(|error| BooguError::Artifact(error.to_string()))?;
            self.control.record_network_fetch(file.size);
            self.control.record_logical_range(file.size);
            bytes
        };
        self.control.push(BrowserArtifactEvent::Progress {
            path: self.progress_path(&file.path),
            loaded_bytes: file.size,
            total_bytes: file.size,
        });
        Ok(bytes)
    }

    async fn fetch_and_cache_direct_complete_file(
        &mut self,
        base_url: &RemoteBaseUrl,
        cache: &web_sys::Cache,
        key: &str,
        file: &ArtifactFile,
        maximum_bytes: u64,
    ) -> Result<Vec<u8>, BooguError> {
        let bytes = fetch_browser_complete_file(
            base_url,
            &file.path,
            Some(file.size),
            maximum_bytes,
            false,
        )
        .await
        .map_err(|error| BooguError::Artifact(error.to_string()))?;
        self.control.record_network_fetch(file.size);
        browser_cache_put(cache, BROWSER_ARTIFACT_PART_CACHE_NAME, key, &bytes)
            .await
            .map_err(|error| BooguError::Artifact(error.to_string()))?;
        self.control.record_cache_write(key, file.size);
        self.control.record_logical_range(file.size);
        Ok(bytes)
    }

    async fn evict_direct_complete_file(&mut self, file: &ArtifactFile) -> Result<(), BooguError> {
        let url = self.complete_object_url(&file.path)?;
        let key = browser_part_cache_key(&url, file.sha256, file.size);
        let cache = self.part_cache().await?;
        let removed = browser_cache_delete(&cache, BROWSER_ARTIFACT_PART_CACHE_NAME, &key)
            .await
            .map_err(|error| BooguError::Artifact(error.to_string()))?;
        self.control.record_cache_eviction(removed);
        Ok(())
    }

    fn protect_verified_direct_complete_file(&self, file: &ArtifactFile) -> Result<(), BooguError> {
        if self.cache_policy != BrowserRangeCachePolicy::Required {
            return Ok(());
        }
        let url = self.complete_object_url(&file.path)?;
        self.control
            .protect_verified_cache_key(&browser_part_cache_key(&url, file.sha256, file.size));
        Ok(())
    }

    fn transport_object_for_file(
        &self,
        file: &ArtifactFile,
    ) -> Result<Option<ArtifactTransportObject>, BooguError> {
        if file.role != ArtifactFileRole::Weights {
            return Ok(None);
        }
        let layout = self.transport_layout.as_ref().ok_or_else(|| {
            BooguError::Artifact("browser reader has no verified transport layout".into())
        })?;
        layout.object(&file.path).cloned().map(Some).ok_or_else(|| {
            BooguError::Artifact(
                ArtifactStreamError::BrowserTransportObjectMissing {
                    path: file.path.clone(),
                }
                .to_string(),
            )
        })
    }

    async fn fetch_verified_transport_part_bytes(
        &mut self,
        logical_file: &ArtifactFile,
        part: &ArtifactTransportPart,
    ) -> Result<Vec<u8>, BooguError> {
        let part_maximum =
            ARTIFACT_TRANSPORT_TARGET_PART_BYTES.min(ARTIFACT_TRANSPORT_MAX_PART_BYTES);
        if part.size == 0 || part.size > part_maximum {
            return Err(BooguError::Artifact(
                ArtifactStreamError::BrowserTransportPartTooLarge {
                    path: part.path.clone(),
                    actual: part.size,
                    maximum: part_maximum,
                }
                .to_string(),
            ));
        }
        let bytes = self
            .fetch_transport_part_complete_attempt(logical_file, part, false)
            .await?;
        let actual = match verify_browser_transport_part_bytes_async(part, &bytes).await {
            Ok(()) => {
                self.control
                    .record_transport_part_verified(self.progress_path(&part.path));
                return Ok(bytes);
            }
            Err(ArtifactStreamError::BrowserTransportPartIntegrity { actual, .. }) => actual,
            Err(error) => return Err(BooguError::Artifact(error.to_string())),
        };
        if self.cache_policy != BrowserRangeCachePolicy::Required {
            return Err(BooguError::Artifact(
                ArtifactStreamError::BrowserTransportPartIntegrity {
                    path: part.path.clone(),
                    expected: part.sha256,
                    actual,
                }
                .to_string(),
            ));
        }

        // Cache entries are untrusted independently of the sealed sidecar.
        // Release the failed part, evict its one complete-object entry, and
        // permit one cache-bypassing network replacement.
        drop(bytes);
        self.control.record_integrity_refetch();
        self.evict_transport_part(part).await?;
        let bytes = self
            .fetch_transport_part_complete_attempt(logical_file, part, true)
            .await?;
        match verify_browser_transport_part_bytes_async(part, &bytes).await {
            Ok(()) => {
                self.control
                    .record_transport_part_verified(self.progress_path(&part.path));
                Ok(bytes)
            }
            Err(ArtifactStreamError::BrowserTransportPartIntegrity { actual, .. }) => {
                drop(bytes);
                self.evict_transport_part(part).await?;
                Err(BooguError::Artifact(
                    ArtifactStreamError::BrowserCacheIntegrityRetryFailed {
                        path: part.path.clone(),
                        expected: part.sha256,
                        actual,
                    }
                    .to_string(),
                ))
            }
            Err(error) => Err(BooguError::Artifact(error.to_string())),
        }
    }

    async fn fetch_transport_shard_read(
        &mut self,
        file: &ArtifactFile,
        object: &ArtifactTransportObject,
    ) -> Result<VerifiedArtifactBytes, BooguError> {
        if object.path != file.path || object.size != file.size || object.sha256 != file.sha256 {
            return Err(BooguError::Artifact(
                ArtifactStreamError::BrowserTransportLayout(format!(
                    "verified layout logical identity for {} drifted from the requested manifest file",
                    file.path
                ))
                .to_string(),
            ));
        }
        let mut builder = VerifiedArtifactBytesBuilder::new(file)
            .map_err(|error| BooguError::Artifact(error.to_string()))?;
        for part in &object.parts {
            validate_browser_transport_part_offset(file, part, builder.len())
                .map_err(|error| BooguError::Artifact(error.to_string()))?;
            let part_bytes = self.fetch_verified_transport_part_bytes(file, part).await?;
            builder
                .extend_from_slice(&part_bytes)
                .map_err(|error| BooguError::Artifact(error.to_string()))?;
        }
        let read = match builder.finish() {
            Ok(read) => read,
            Err((_error, bytes)) => {
                let actual_size = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
                if actual_size != file.size {
                    return Err(BooguError::Artifact(
                        ArtifactStreamError::BrowserTransportReconstructionSize {
                            path: file.path.clone(),
                            expected: file.size,
                            actual: actual_size,
                        }
                        .to_string(),
                    ));
                }
                return Err(BooguError::Artifact(
                    ArtifactStreamError::BrowserTransportReconstructionIntegrity {
                        path: file.path.clone(),
                        expected: file.sha256,
                        actual: Sha256Digest::calculate(&bytes),
                    }
                    .to_string(),
                ));
            }
        };
        // Prior-session Cache Storage hits become continuity-protected only
        // after both the physical-part digests and complete logical artifact
        // digest have passed.
        for part in &object.parts {
            self.protect_verified_transport_part(part)?;
        }
        Ok(read)
    }

    async fn fetch_verified_shard_read(
        &mut self,
        file: &ArtifactFile,
        max_bytes: u64,
    ) -> Result<VerifiedArtifactBytes, BooguError> {
        self.control.check_cancelled()?;
        self.control
            .push(BrowserArtifactEvent::Started(self.progress_file(file)));
        let maximum = max_bytes.min(MAX_BROWSER_STAGE_BYTES);
        if file.size > maximum {
            return Err(BooguError::Artifact(format!(
                "browser stage {} is {} bytes, exceeding the bounded maximum {maximum}",
                file.path, file.size
            )));
        }
        if let Some(object) = self.transport_object_for_file(file)? {
            return self.fetch_transport_shard_read(file, &object).await;
        }
        let complete_file_maximum = maximum.min(ARTIFACT_TRANSPORT_MAX_PART_BYTES);
        let bytes = self
            .fetch_direct_complete_file_attempt(file, complete_file_maximum, false)
            .await?;
        let (read, actual) = match VerifiedArtifactBytes::try_verify_sha256(file, bytes) {
            Ok(read) => (Some(read), None),
            Err((_error, bytes)) => (None, Some((Sha256Digest::calculate(&bytes), bytes))),
        };
        if let Some(read) = read {
            self.control
                .record_transport_part_verified(self.progress_path(&file.path));
            self.protect_verified_direct_complete_file(file)?;
            return Ok(read);
        }
        let (actual, bytes) = actual.expect("failed verification retains rejected bytes");

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
        self.evict_direct_complete_file(file).await?;
        let bytes = self
            .fetch_direct_complete_file_attempt(file, complete_file_maximum, true)
            .await?;
        let read = match VerifiedArtifactBytes::try_verify_sha256(file, bytes) {
            Ok(read) => read,
            Err((_error, bytes)) => {
                let actual = Sha256Digest::calculate(&bytes);
                drop(bytes);
                self.evict_direct_complete_file(file).await?;
                return Err(BooguError::Artifact(
                    ArtifactStreamError::BrowserCacheIntegrityRetryFailed {
                        path: file.path.clone(),
                        expected: file.sha256,
                        actual,
                    }
                    .to_string(),
                ));
            }
        };
        self.control
            .record_transport_part_verified(self.progress_path(&file.path));
        self.protect_verified_direct_complete_file(file)?;
        Ok(read)
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
        let read = AsyncStageShardRead::from_verified_artifact_bytes(
            self.fetch_verified_shard_read(file, max_bytes).await?,
        );
        self.control.push(BrowserArtifactEvent::Verified(
            self.progress_path(&file.path),
        ));
        Ok(read)
    }
}

/// Model crates consume the model-neutral reader contract while Boogu's denoiser uses the same
/// verified transport implementation through its stage-reader trait.
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
        let read = self
            .fetch_verified_shard_read(file, maximum_bytes)
            .await
            .map_err(|error| ArtifactReadError::transport(error.to_string()))?;
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
    use std::collections::BTreeMap;

    use burn_image::{
        ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES, ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES_KEY,
        ARTIFACT_TARGET_MAX_SEMANTIC_SHARD_BYTES_KEY,
        ARTIFACT_TARGET_MAX_TRANSPORT_SHARD_BYTES_KEY, ARTIFACT_TRANSPORT_LAYOUT_PATH,
        ARTIFACT_TRANSPORT_LAYOUT_PATH_KEY, ARTIFACT_TRANSPORT_LAYOUT_SCHEMA_KEY,
        ARTIFACT_TRANSPORT_LAYOUT_SCHEMA_VERSION, ARTIFACT_TRANSPORT_PART_TARGET_BYTES_KEY,
        ARTIFACT_TRANSPORT_PARTS_REQUIRED_KEY, ArtifactProfileId, ArtifactShard,
        ArtifactTransportLayoutError, ModelId, NumericFormat,
    };

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

    fn transport_metadata() -> BTreeMap<String, String> {
        BTreeMap::from([
            (
                ARTIFACT_TRANSPORT_LAYOUT_PATH_KEY.into(),
                ARTIFACT_TRANSPORT_LAYOUT_PATH.into(),
            ),
            (
                ARTIFACT_TRANSPORT_LAYOUT_SCHEMA_KEY.into(),
                ARTIFACT_TRANSPORT_LAYOUT_SCHEMA_VERSION.to_string(),
            ),
            (ARTIFACT_TRANSPORT_PARTS_REQUIRED_KEY.into(), "true".into()),
            (
                ARTIFACT_TRANSPORT_PART_TARGET_BYTES_KEY.into(),
                ARTIFACT_TRANSPORT_TARGET_PART_BYTES.to_string(),
            ),
            (
                ARTIFACT_TARGET_MAX_TRANSPORT_SHARD_BYTES_KEY.into(),
                ARTIFACT_TRANSPORT_MAX_PART_BYTES.to_string(),
            ),
            (
                ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES_KEY.into(),
                ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES.to_string(),
            ),
            (
                ARTIFACT_TARGET_MAX_SEMANTIC_SHARD_BYTES_KEY.into(),
                ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES.to_string(),
            ),
        ])
    }

    fn transport_weight(size: u64, digest: Sha256Digest) -> ArtifactFile {
        ArtifactFile {
            path: ArtifactPath::new(format!("objects/{digest}.bpk")).unwrap(),
            size,
            sha256: digest,
            role: ArtifactFileRole::Weights,
            component: None,
            shard: None,
        }
    }

    fn transport_layout(
        weight: &ArtifactFile,
        parts: Vec<ArtifactTransportPart>,
    ) -> ArtifactTransportLayout {
        ArtifactTransportLayout {
            schema_version: ARTIFACT_TRANSPORT_LAYOUT_SCHEMA_VERSION,
            bundle: ArtifactBundleId::new("browser-transport-test").unwrap(),
            profile: ArtifactProfileId::new("test-profile").unwrap(),
            model: ModelId::new("test/browser-transport").unwrap(),
            model_revision: "revision".into(),
            target_part_bytes: ARTIFACT_TRANSPORT_TARGET_PART_BYTES,
            hard_max_part_bytes: ARTIFACT_TRANSPORT_MAX_PART_BYTES,
            objects: vec![ArtifactTransportObject {
                path: weight.path.clone(),
                size: weight.size,
                sha256: weight.sha256,
                parts,
            }],
        }
    }

    fn transport_part(offset: u64, bytes: &[u8]) -> ArtifactTransportPart {
        let sha256 = Sha256Digest::calculate(bytes);
        ArtifactTransportPart {
            path: ArtifactPath::new(format!("transport/{sha256}.part")).unwrap(),
            offset,
            size: bytes.len() as u64,
            sha256,
        }
    }

    fn seal_transport_fixture(
        weight: ArtifactFile,
        layout: &ArtifactTransportLayout,
    ) -> (ArtifactManifest, Vec<u8>) {
        let layout_bytes = serde_json::to_vec(layout).unwrap();
        let mut manifest = ArtifactManifest {
            schema_version: burn_image::ARTIFACT_MANIFEST_SCHEMA_V2,
            bundle: layout.bundle.clone(),
            profile: layout.profile.clone(),
            model: layout.model.clone(),
            model_revision: layout.model_revision.clone(),
            numeric_format: NumericFormat::F16,
            components: Vec::new(),
            files: vec![
                weight,
                ArtifactFile {
                    path: ArtifactPath::new(ARTIFACT_TRANSPORT_LAYOUT_PATH).unwrap(),
                    size: layout_bytes.len() as u64,
                    sha256: Sha256Digest::calculate(&layout_bytes),
                    role: ArtifactFileRole::Metadata,
                    component: None,
                    shard: None,
                },
            ],
            dependencies: Vec::new(),
            metadata: transport_metadata(),
            content_digest: None,
        };
        manifest.seal().unwrap();
        (manifest, layout_bytes)
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
    fn persistent_cache_plan_deduplicates_and_reserves_headroom_correctness() {
        let mut plan = BrowserPersistentCachePlan::default();
        plan.register(
            BROWSER_ARTIFACT_PART_CACHE_NAME,
            "https://burn-image.invalid/part/a".into(),
            20,
        )
        .unwrap();
        plan.register(
            BROWSER_ARTIFACT_PART_CACHE_NAME,
            "https://burn-image.invalid/part/a".into(),
            20,
        )
        .unwrap();
        plan.register(
            BROWSER_ARTIFACT_PART_CACHE_NAME,
            "https://burn-image.invalid/part/b".into(),
            10,
        )
        .unwrap();
        assert_eq!(plan.entry_count(), 2);
        assert_eq!(plan.total_bytes(), 30);
        assert_eq!(browser_persistent_cache_reserve(0), 0);
        assert_eq!(
            browser_persistent_cache_reserve(1_000),
            BROWSER_PERSISTENT_CACHE_RESERVE_BYTES + 10
        );
        let error = plan
            .register(
                BROWSER_ARTIFACT_PART_CACHE_NAME,
                "https://burn-image.invalid/part/a".into(),
                21,
            )
            .unwrap_err();
        assert!(matches!(
            error,
            ArtifactStreamError::BrowserPersistentCachePlan(_)
        ));
    }

    #[test]
    fn storage_estimate_requires_finite_safe_nonnegative_bytes_correctness() {
        assert_eq!(
            browser_storage_estimate_bytes("quota", Some(12_345.75)).unwrap(),
            12_345
        );
        assert_eq!(
            browser_storage_estimate_bytes("quota", Some(50_481_610_932.0)).unwrap(),
            50_481_610_932
        );
        assert_eq!(
            browser_storage_estimate_bytes("usage", Some(39_744_192_692.0)).unwrap(),
            39_744_192_692
        );
        for value in [
            None,
            Some(-1.0),
            Some(f64::NAN),
            Some(f64::INFINITY),
            Some(9_007_199_254_740_992.0),
        ] {
            assert!(matches!(
                browser_storage_estimate_bytes("quota", value),
                Err(ArtifactStreamError::BrowserStorageEstimate { field: "quota", .. })
            ));
        }
    }

    #[test]
    fn browser_part_cache_key_binds_url_digest_and_size_correctness() {
        let url = "https://cdn.example/model/transport/part.part";
        let digest = Sha256Digest::calculate(b"sealed transport part");
        let key = browser_part_cache_key(url, digest, ARTIFACT_TRANSPORT_TARGET_PART_BYTES);
        assert!(key.starts_with("https://burn-image.invalid/.well-known/part-cache/v2/"));
        assert!(key.ends_with(&format!("/{digest}/{ARTIFACT_TRANSPORT_TARGET_PART_BYTES}")));
        assert_eq!(
            key,
            browser_part_cache_key(url, digest, ARTIFACT_TRANSPORT_TARGET_PART_BYTES)
        );
        assert_ne!(
            key,
            browser_part_cache_key(
                "https://mirror.example/model/transport/part.part",
                digest,
                ARTIFACT_TRANSPORT_TARGET_PART_BYTES
            )
        );
        assert_ne!(
            key,
            browser_part_cache_key(url, digest, ARTIFACT_TRANSPORT_TARGET_PART_BYTES - 1)
        );
        assert_ne!(
            key,
            browser_part_cache_key(
                url,
                Sha256Digest::calculate(b"replacement transport part"),
                ARTIFACT_TRANSPORT_TARGET_PART_BYTES
            )
        );
    }

    #[test]
    fn browser_transport_typestate_authenticates_sidecar_and_manifest_correctness() {
        let logical = b"authenticated logical Burnpack";
        let weight = transport_weight(logical.len() as u64, Sha256Digest::calculate(logical));
        let layout = transport_layout(&weight, vec![transport_part(0, logical)]);
        let (manifest, layout_bytes) = seal_transport_fixture(weight.clone(), &layout);
        assert!(layout_bytes.len() as u64 <= MAX_ARTIFACT_TRANSPORT_LAYOUT_BYTES);

        let verified = ArtifactTransportLayout::parse_and_validate(&manifest, &layout_bytes)
            .expect("the manifest-sealed layout must mint verified typestate");
        assert_eq!(
            verified.manifest_content_digest(),
            manifest.content_digest.unwrap()
        );
        assert_eq!(verified.object(&weight.path).unwrap().sha256, weight.sha256);

        let mut corrupted = layout_bytes;
        corrupted[0] ^= 1;
        assert!(matches!(
            ArtifactTransportLayout::parse_and_validate(&manifest, &corrupted),
            Err(ArtifactTransportLayoutError::Integrity(_))
        ));
    }

    #[test]
    fn browser_transport_declaration_rejects_partial_metadata_correctness() {
        let logical = b"direct object";
        let weight = transport_weight(logical.len() as u64, Sha256Digest::calculate(logical));
        let layout = transport_layout(&weight, vec![transport_part(0, logical)]);
        let (mut manifest, _) = seal_transport_fixture(weight, &layout);
        manifest
            .files
            .retain(|file| file.path.as_str() != ARTIFACT_TRANSPORT_LAYOUT_PATH);
        manifest.metadata.clear();
        manifest.seal().unwrap();
        assert!(
            ArtifactTransportLayout::declared_file(&manifest)
                .unwrap()
                .is_none()
        );

        manifest
            .metadata
            .insert(ARTIFACT_TRANSPORT_PARTS_REQUIRED_KEY.into(), "true".into());
        manifest.seal().unwrap();
        assert!(matches!(
            ArtifactTransportLayout::declared_file(&manifest),
            Err(ArtifactTransportLayoutError::InvalidDeclaration(_))
        ));
    }

    #[test]
    fn browser_transport_rejects_oversize_and_noncontiguous_parts_correctness() {
        let oversize = ARTIFACT_TRANSPORT_MAX_PART_BYTES + 1;
        let logical_digest = Sha256Digest::calculate(b"oversize logical identity");
        let part_digest = Sha256Digest::calculate(b"oversize physical identity");
        let weight = transport_weight(oversize, logical_digest);
        let layout = transport_layout(
            &weight,
            vec![ArtifactTransportPart {
                path: ArtifactPath::new(format!("transport/{part_digest}.part")).unwrap(),
                offset: 0,
                size: oversize,
                sha256: part_digest,
            }],
        );
        let (manifest, bytes) = seal_transport_fixture(weight, &layout);
        assert!(matches!(
            ArtifactTransportLayout::parse_and_validate(&manifest, &bytes),
            Err(ArtifactTransportLayoutError::PartSizeOutOfBounds { .. })
        ));

        let logical_size = ARTIFACT_TRANSPORT_TARGET_PART_BYTES + 1;
        let weight = transport_weight(
            logical_size,
            Sha256Digest::calculate(b"noncontiguous logical identity"),
        );
        let first_digest = Sha256Digest::calculate(b"first full physical part identity");
        let final_digest = Sha256Digest::calculate(b"final physical part identity");
        let layout = transport_layout(
            &weight,
            vec![
                ArtifactTransportPart {
                    path: ArtifactPath::new(format!("transport/{first_digest}.part")).unwrap(),
                    offset: 0,
                    size: ARTIFACT_TRANSPORT_TARGET_PART_BYTES,
                    sha256: first_digest,
                },
                ArtifactTransportPart {
                    path: ArtifactPath::new(format!("transport/{final_digest}.part")).unwrap(),
                    offset: ARTIFACT_TRANSPORT_TARGET_PART_BYTES + 1,
                    size: 1,
                    sha256: final_digest,
                },
            ],
        );
        let (manifest, bytes) = seal_transport_fixture(weight, &layout);
        assert!(matches!(
            ArtifactTransportLayout::parse_and_validate(&manifest, &bytes),
            Err(ArtifactTransportLayoutError::PartOffsetMismatch { .. })
        ));
    }

    #[test]
    fn browser_transport_verifies_parts_before_logical_reconstruction_correctness() {
        let logical = b"abcdefgh";
        let logical_file = file(logical);
        let first = transport_part(0, &logical[..4]);
        let second = transport_part(4, &logical[4..]);

        verify_browser_transport_part_bytes(&first, &logical[..4]).unwrap();
        assert!(matches!(
            verify_browser_transport_part_bytes(&first, b"abce"),
            Err(ArtifactStreamError::BrowserTransportPartIntegrity { .. })
        ));
        validate_browser_transport_part_offset(&logical_file, &first, 0).unwrap();
        validate_browser_transport_part_offset(&logical_file, &second, 4).unwrap();
        assert!(matches!(
            validate_browser_transport_part_offset(&logical_file, &second, 3),
            Err(ArtifactStreamError::BrowserTransportLayout(_))
        ));
        verify_browser_transport_reconstruction(&logical_file, logical).unwrap();
        assert!(matches!(
            verify_browser_transport_reconstruction(&logical_file, b"abcdefgi"),
            Err(ArtifactStreamError::BrowserTransportReconstructionIntegrity { .. })
        ));
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
            cache: BROWSER_ARTIFACT_PART_CACHE_NAME,
            path: ArtifactPath::new("transport/a.part").unwrap(),
            offset: 0,
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
    fn sealed_browser_range_requires_exact_object_total_correctness() {
        let range = ByteRange::new(8, 4).unwrap();
        validate_content_range_exact_total(range, Some("bytes 8-11/32"), 32).unwrap();
        for actual in [
            None,
            Some("bytes 8-11/*"),
            Some("bytes 8-11/31"),
            Some("bytes 8-11/33"),
            Some("bytes 7-10/32"),
        ] {
            assert!(matches!(
                validate_content_range_exact_total(range, actual, 32),
                Err(ArtifactStreamError::BrowserContentRange { .. })
            ));
        }
    }

    #[test]
    fn browser_response_requires_exact_identity_framing_before_wasm_copy_correctness() {
        validate_browser_content_length(4, Some("4")).unwrap();
        for actual in [None, Some("3"), Some("5"), Some("04"), Some("invalid")] {
            assert!(matches!(
                validate_browser_content_length(4, actual),
                Err(ArtifactStreamError::BrowserContentLength { expected: 4, .. })
            ));
        }

        validate_browser_content_length_if_exposed(4, None).unwrap();
        validate_browser_content_length_if_exposed(4, Some("4")).unwrap();
        for actual in [Some("3"), Some("5"), Some("04"), Some("invalid")] {
            assert!(matches!(
                validate_browser_content_length_if_exposed(4, actual),
                Err(ArtifactStreamError::BrowserContentLength { expected: 4, .. })
            ));
        }

        assert_eq!(parse_browser_content_length(Some("4")).unwrap(), 4);
        for actual in [None, Some("04"), Some("invalid"), Some(" 4")] {
            assert!(matches!(
                parse_browser_content_length(actual),
                Err(ArtifactStreamError::BrowserMalformedContentLength { .. })
            ));
        }

        validate_browser_content_encoding(None).unwrap();
        validate_browser_content_encoding(Some("identity")).unwrap();
        validate_browser_content_encoding(Some("IDENTITY")).unwrap();
        for actual in ["gzip", "br", "identity, gzip", " identity "] {
            assert!(matches!(
                validate_browser_content_encoding(Some(actual)),
                Err(ArtifactStreamError::BrowserContentEncoding { .. })
            ));
        }

        validate_browser_response_size(4, 4).unwrap();
        assert!(matches!(
            validate_browser_response_size(4, 3),
            Err(ArtifactStreamError::BrowserResponseSize {
                expected: 4,
                actual: 3
            })
        ));
        assert!(matches!(
            validate_browser_response_size(4, 5),
            Err(ArtifactStreamError::BrowserResponseSize {
                expected: 4,
                actual: 5
            })
        ));

        validate_browser_complete_object_size(1, 4).unwrap();
        validate_browser_complete_object_size(4, 4).unwrap();
        for actual in [0, 5] {
            assert!(matches!(
                validate_browser_complete_object_size(actual, 4),
                Err(ArtifactStreamError::BrowserFileTooLarge {
                    actual: rejected,
                    maximum: 4
                }) if rejected == actual
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

    #[test]
    fn aggregate_transport_progress_is_closure_wide_monotonic_and_smoothed_correctness() {
        let mut tracker = BrowserArtifactTransferTracker::default();
        tracker.set_phase("Inference model transfer");
        let logical_a = ArtifactPath::new("qwen/objects/a.bpk").unwrap();
        let logical_b = ArtifactPath::new("vae/objects/b.bpk").unwrap();
        let part_a = ArtifactPath::new("qwen/transport/a.part").unwrap();
        let part_b = ArtifactPath::new("vae/transport/b.part").unwrap();
        tracker
            .register_logical_object(logical_a.clone(), 16)
            .unwrap();
        tracker
            .register_logical_object(logical_b.clone(), 4)
            .unwrap();
        tracker
            .register_physical_part(part_a.clone(), 16, 4)
            .unwrap();
        tracker
            .register_physical_part(part_b.clone(), 4, 4)
            .unwrap();

        let mut current = file(b"stage");
        current.path = logical_a.clone();
        current.component = Some(ArtifactComponentId::new("qwen").unwrap());
        tracker.object_started(&current);

        tracker.record_bounded_range(part_a.clone(), 0, 4, 0.0);
        assert_eq!(tracker.snapshot().unwrap().bytes_per_second, None);
        tracker.record_bounded_range(part_a.clone(), 4, 4, 1_000.0);
        tracker.record_bounded_range(part_a.clone(), 8, 4, 2_000.0);
        assert_eq!(tracker.snapshot().unwrap().bytes_per_second, None);
        tracker.record_bounded_range(part_a.clone(), 12, 4, 3_000.0);
        tracker.physical_part_verified(part_a.clone());
        tracker.logical_object_verified(logical_a.clone());

        let progress = tracker.snapshot().unwrap();
        assert_eq!(progress.phase, "Inference model transfer");
        assert_eq!(progress.component.unwrap().as_str(), "qwen");
        assert_eq!(progress.logical_objects_completed, 1);
        assert_eq!(progress.logical_objects_total, 2);
        assert_eq!(progress.physical_parts_completed, 1);
        assert_eq!(progress.physical_parts_total, 2);
        assert_eq!(progress.bounded_ranges_completed, 4);
        assert_eq!(progress.bounded_ranges_total, 5);
        assert_eq!((progress.loaded_bytes, progress.total_bytes), (16, 20));
        assert_eq!(progress.bytes_per_second, Some(4));
        assert_eq!(progress.eta_seconds, Some(1));
        assert_eq!(progress.request_activity, None);

        // A repeated semantic-stage reconstruction reuses the same physical ranges and must not
        // advance or reset the aggregate denominator.
        tracker.record_bounded_range(part_a.clone(), 0, 4, 4_000.0);
        assert_eq!(tracker.snapshot().unwrap().loaded_bytes, 16);

        tracker.record_bounded_range(part_b.clone(), 0, 4, 5_000.0);
        tracker.physical_part_verified(part_b.clone());
        tracker.logical_object_verified(logical_b.clone());
        let complete = tracker.snapshot().unwrap();
        assert_eq!(complete.loaded_bytes, complete.total_bytes);
        assert_eq!(complete.logical_objects_completed, 2);
        assert_eq!(complete.physical_parts_completed, 2);
        assert_eq!(complete.bounded_ranges_completed, 5);
        assert_eq!(complete.eta_seconds, None);

        tracker.start_request_activity();
        let mut cached = file(b"cached");
        cached.path = logical_b.clone();
        cached.component = Some(ArtifactComponentId::new("flux-vae-decoder").unwrap());
        tracker.object_started(&cached);
        tracker.record_bounded_range(part_b, 0, 4, 6_000.0);
        tracker.logical_object_verified(logical_b);
        let rehydrating = tracker.snapshot().unwrap();
        assert_eq!(rehydrating.loaded_bytes, complete.loaded_bytes);
        assert_eq!(
            rehydrating.physical_parts_completed,
            complete.physical_parts_completed
        );
        let activity = rehydrating.request_activity.unwrap();
        assert_eq!(activity.phase, "Applying verified cached model stages");
        assert_eq!(activity.current_path, Some(cached.path));
        assert_eq!(activity.component.unwrap().as_str(), "flux-vae-decoder");
        assert_eq!(activity.logical_objects_completed, 1);
        assert_eq!(activity.bounded_ranges_processed, 1);
        assert_eq!(activity.processed_bytes, 4);
    }

    #[test]
    fn three_reader_closure_plan_is_stable_and_deduplicates_shared_parts_correctness() {
        let bundle_fixture = |bundle_name: &str, objects: &[(&str, &[u8])]| {
            let bundle = ArtifactBundleId::new(bundle_name).unwrap();
            let profile = ArtifactProfileId::new("test-profile").unwrap();
            let model = ModelId::new(format!("test/{bundle_name}")).unwrap();
            let mut weights = Vec::new();
            let mut transport_objects = Vec::new();
            for (path, bytes) in objects {
                let digest = Sha256Digest::calculate(bytes);
                let weight = ArtifactFile {
                    path: ArtifactPath::new(*path).unwrap(),
                    size: bytes.len() as u64,
                    sha256: digest,
                    role: ArtifactFileRole::Weights,
                    component: None,
                    shard: None,
                };
                transport_objects.push(ArtifactTransportObject {
                    path: weight.path.clone(),
                    size: weight.size,
                    sha256: weight.sha256,
                    parts: vec![transport_part(0, bytes)],
                });
                weights.push(weight);
            }
            transport_objects.sort_by(|left, right| left.path.cmp(&right.path));
            let layout = ArtifactTransportLayout {
                schema_version: ARTIFACT_TRANSPORT_LAYOUT_SCHEMA_VERSION,
                bundle: bundle.clone(),
                profile: profile.clone(),
                model: model.clone(),
                model_revision: "revision".into(),
                target_part_bytes: ARTIFACT_TRANSPORT_TARGET_PART_BYTES,
                hard_max_part_bytes: ARTIFACT_TRANSPORT_MAX_PART_BYTES,
                objects: transport_objects,
            };
            let layout_bytes = serde_json::to_vec(&layout).unwrap();
            weights.push(ArtifactFile {
                path: ArtifactPath::new(ARTIFACT_TRANSPORT_LAYOUT_PATH).unwrap(),
                size: layout_bytes.len() as u64,
                sha256: Sha256Digest::calculate(&layout_bytes),
                role: ArtifactFileRole::Metadata,
                component: None,
                shard: None,
            });
            let mut manifest = ArtifactManifest {
                schema_version: burn_image::ARTIFACT_MANIFEST_SCHEMA_V2,
                bundle,
                profile,
                model,
                model_revision: "revision".into(),
                numeric_format: NumericFormat::F16,
                components: Vec::new(),
                files: weights,
                dependencies: Vec::new(),
                metadata: transport_metadata(),
                content_digest: None,
            };
            manifest.seal().unwrap();
            let verified =
                ArtifactTransportLayout::parse_and_validate(&manifest, &layout_bytes).unwrap();
            (manifest, verified)
        };

        let shared_pipeline_part = b"12345678".as_slice();
        let (pipeline, pipeline_layout) = bundle_fixture(
            "pipeline",
            &[
                ("objects/a.bpk", shared_pipeline_part),
                ("objects/b.bpk", shared_pipeline_part),
            ],
        );
        let (qwen, qwen_layout) =
            bundle_fixture("qwen", &[("objects/qwen.bpk", b"qwen".as_slice())]);
        let (vae, vae_layout) =
            bundle_fixture("vae", &[("objects/vae.bpk", b"vae-payload!".as_slice())]);

        let mut tracker = BrowserArtifactTransferTracker::default();
        for (manifest, layout) in [
            (&pipeline, &pipeline_layout),
            (&qwen, &qwen_layout),
            (&vae, &vae_layout),
        ] {
            tracker
                .register_manifest_plan(manifest, Some(&manifest.bundle), layout)
                .unwrap();
        }
        let first = tracker.snapshot().unwrap();
        assert_eq!(first.logical_objects_total, 4);
        assert_eq!(first.physical_parts_total, 3);
        assert_eq!(first.bounded_ranges_total, 3);
        assert_eq!(first.total_bytes, 24);

        // Reader construction or a cloned reader may present the same sealed plan again. Exact
        // duplicate registration must remain idempotent and cannot inflate the visible total.
        for (manifest, layout) in [
            (&pipeline, &pipeline_layout),
            (&qwen, &qwen_layout),
            (&vae, &vae_layout),
        ] {
            tracker
                .register_manifest_plan(manifest, Some(&manifest.bundle), layout)
                .unwrap();
        }
        assert_eq!(tracker.snapshot().unwrap(), first);

        let active_pipeline = ArtifactPath::new("pipeline/objects/a.bpk").unwrap();
        let active_qwen = ArtifactPath::new("qwen/objects/qwen.bpk").unwrap();
        tracker
            .retain_logical_objects(&BTreeSet::from([
                active_pipeline.clone(),
                active_qwen.clone(),
            ]))
            .unwrap();
        let active = tracker.snapshot().unwrap();
        assert_eq!(active.logical_objects_total, 2);
        assert_eq!(active.physical_parts_total, 2);
        assert_eq!(active.bounded_ranges_total, 2);
        assert_eq!(active.total_bytes, 12);

        for (bundle, layout, logical) in [
            (&pipeline.bundle, &pipeline_layout, active_pipeline),
            (&qwen.bundle, &qwen_layout, active_qwen),
        ] {
            let relative = ArtifactPath::new(
                logical
                    .as_str()
                    .split_once('/')
                    .expect("qualified path has bundle prefix")
                    .1,
            )
            .unwrap();
            let object = layout.object(&relative).unwrap();
            for part in &object.parts {
                let physical = qualified_transfer_path(Some(bundle), &part.path);
                tracker.record_bounded_range(physical.clone(), 0, part.size, 0.0);
                tracker.physical_part_verified(physical);
            }
            tracker.logical_object_verified(logical);
        }
        let complete = tracker.snapshot().unwrap();
        assert_eq!(complete.loaded_bytes, complete.total_bytes);
        assert_eq!(
            complete.logical_objects_completed,
            complete.logical_objects_total
        );
        assert_eq!(
            complete.physical_parts_completed,
            complete.physical_parts_total
        );
        assert_eq!(
            complete.bounded_ranges_completed,
            complete.bounded_ranges_total
        );
    }
}
