//! Verified native CDN cache for sealed Boogu artifact bundles.
//!
//! The cache mirrors the immutable CDN tree under
//! `~/.burn_image/models/<bundle-id>`. A manifest is installed only after its seal and exact
//! variant/profile/bundle identity validate. Every declared payload is streamed through the
//! manifest's size and SHA-256 contract before an atomic rename makes it visible to the runtime.

use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use burn_boogu::artifacts::{
    canonical_published_bundle, validate_canonical_release_artifact_digest,
};
use burn_boogu::{BooguVariant, artifacts::BooguStorageProfile, boogu_model_descriptor};
use burn_image::{
    ArtifactFile, ArtifactManifest, ArtifactPath, ArtifactSource, ArtifactVerifier,
    IntegrityPolicy, NumericFormat, RemoteBaseUrl,
};
use thiserror::Error;

use crate::{
    BOOGU_CDN_ROOT, MAX_BROWSER_MANIFEST_BYTES, boogu_bundle_id, boogu_cdn_base_url,
    boogu_profile_slug,
};

/// Root containing immutable model-name prefixes on the Aberration CDN.
pub const DEFAULT_BURN_IMAGE_MODEL_ROOT_URL: &str = BOOGU_CDN_ROOT;
/// Per-user native cache root, matching the convention used by `burn_jepa`.
pub const DEFAULT_BURN_IMAGE_CACHE_ROOT_DIR: &str = ".burn_image";
/// Model bundle subtree below [`DEFAULT_BURN_IMAGE_CACHE_ROOT_DIR`].
pub const DEFAULT_BURN_IMAGE_MODEL_CACHE_SUBDIR: &str = "models";

const DOWNLOAD_ATTEMPTS: u32 = 4;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(20);
const READ_TIMEOUT: Duration = Duration::from_secs(60);
const WRITE_TIMEOUT: Duration = Duration::from_secs(60);
const COPY_BUFFER_BYTES: usize = 1024 * 1024;

/// Native artifact bootstrap failure. The runtime maps this to its selected model identity.
#[derive(Debug, Error)]
pub enum NativeArtifactCacheError {
    #[error("{0}")]
    Message(String),
}

impl NativeArtifactCacheError {
    fn message(value: impl Into<String>) -> Self {
        Self::Message(value.into())
    }
}

/// Default immutable CDN prefix for one exact manifest bundle identity.
pub fn default_native_boogu_model_base_url(
    variant: BooguVariant,
    profile: BooguStorageProfile,
) -> Result<RemoteBaseUrl, NativeArtifactCacheError> {
    let value = match std::env::var("BURN_IMAGE_MODEL_BASE_URL") {
        Ok(value) => value,
        Err(_) => canonical_native_boogu_model_base_url(variant, profile)?,
    };
    RemoteBaseUrl::new(value).map_err(|error| {
        NativeArtifactCacheError::message(format!("invalid native Boogu model base URL: {error}"))
    })
}

fn canonical_native_boogu_model_base_url(
    variant: BooguVariant,
    profile: BooguStorageProfile,
) -> Result<String, NativeArtifactCacheError> {
    let published = canonical_published_bundle(variant, profile).ok_or_else(|| {
        NativeArtifactCacheError::message(format!(
            "{variant:?}/{profile:?} has no canonical published Aberration CDN bundle; pass --artifacts or set BURN_IMAGE_MODEL_BASE_URL for an explicit diagnostic source"
        ))
    })?;
    let value = boogu_cdn_base_url(variant, profile);
    debug_assert!(value.ends_with(published.bundle_id));
    Ok(value)
}

/// Default exact cache directory for one bundle.
///
/// `BURN_IMAGE_CACHE_DIR` replaces the broad `~/.burn_image` root;
/// `BURN_IMAGE_MODEL_CACHE_DIR` replaces the final bundle directory.
pub fn default_native_boogu_model_cache_root(
    variant: BooguVariant,
    profile: BooguStorageProfile,
) -> Result<PathBuf, NativeArtifactCacheError> {
    if let Some(exact) = std::env::var_os("BURN_IMAGE_MODEL_CACHE_DIR") {
        return Ok(expand_home_path(PathBuf::from(exact)));
    }
    let broad_root = match std::env::var_os("BURN_IMAGE_CACHE_DIR") {
        Some(root) => expand_home_path(PathBuf::from(root)),
        None => user_home_dir()
            .ok_or_else(|| {
                NativeArtifactCacheError::message(
                    "failed to resolve the user home directory for the native model cache",
                )
            })?
            .join(DEFAULT_BURN_IMAGE_CACHE_ROOT_DIR),
    };
    Ok(broad_root
        .join(DEFAULT_BURN_IMAGE_MODEL_CACHE_SUBDIR)
        .join(boogu_bundle_id(variant, profile)))
}

/// Resolve a local override or materialize a remote sealed bundle in the verified native cache.
pub fn resolve_native_boogu_artifact_directory<F>(
    variant: BooguVariant,
    profile: BooguStorageProfile,
    source: &ArtifactSource,
    progress: F,
) -> Result<PathBuf, NativeArtifactCacheError>
where
    F: Fn(&str),
{
    let ArtifactSource::Remote { base_url } = source else {
        return source
            .local_root()
            .map(Path::to_path_buf)
            .ok_or_else(|| NativeArtifactCacheError::message("invalid local artifact source"));
    };
    let base_url = match std::env::var("BURN_IMAGE_MODEL_BASE_URL") {
        Ok(value) => RemoteBaseUrl::new(value).map_err(|error| {
            NativeArtifactCacheError::message(format!("invalid BURN_IMAGE_MODEL_BASE_URL: {error}"))
        })?,
        Err(_) => base_url.clone(),
    };
    let require_canonical_digest = native_boogu_source_requires_canonical_digest(
        variant,
        profile,
        &ArtifactSource::Remote {
            base_url: base_url.clone(),
        },
    )?;
    let cache_root = default_native_boogu_model_cache_root(variant, profile)?;
    cache_remote_bundle(
        variant,
        profile,
        &base_url,
        &cache_root,
        require_canonical_digest,
        &progress,
    )?;
    Ok(cache_root)
}

/// Whether an effective remote source names the pinned canonical CDN prefix.
///
/// Explicit local or custom remote diagnostic sources return false. Invalid URL overrides fail
/// before loading so the caller cannot accidentally fall back to a different origin.
pub fn native_boogu_source_requires_canonical_digest(
    variant: BooguVariant,
    profile: BooguStorageProfile,
    source: &ArtifactSource,
) -> Result<bool, NativeArtifactCacheError> {
    let ArtifactSource::Remote { base_url } = source else {
        return Ok(false);
    };
    let effective = match std::env::var("BURN_IMAGE_MODEL_BASE_URL") {
        Ok(value) => RemoteBaseUrl::new(value).map_err(|error| {
            NativeArtifactCacheError::message(format!("invalid BURN_IMAGE_MODEL_BASE_URL: {error}"))
        })?,
        Err(_) => base_url.clone(),
    };
    Ok(
        canonical_published_bundle(variant, profile).is_some_and(|published| {
            effective.as_str()
                == format!(
                    "{DEFAULT_BURN_IMAGE_MODEL_ROOT_URL}/{}",
                    published.bundle_id
                )
        }),
    )
}

fn cache_remote_bundle<F>(
    variant: BooguVariant,
    profile: BooguStorageProfile,
    base_url: &RemoteBaseUrl,
    cache_root: &Path,
    require_canonical_digest: bool,
    progress: &F,
) -> Result<(), NativeArtifactCacheError>
where
    F: Fn(&str),
{
    fs::create_dir_all(cache_root).map_err(|error| {
        NativeArtifactCacheError::message(format!(
            "create native model cache {}: {error}",
            cache_root.display()
        ))
    })?;
    progress(&format!(
        "resolving native model cache under {}",
        cache_root.display()
    ));

    let manifest_path = cache_root.join("manifest.json");
    let (manifest, manifest_bytes, mut manifest_needs_commit) =
        match read_expected_manifest(&manifest_path, variant, profile, require_canonical_digest) {
            Ok((manifest, bytes)) => {
                progress("using cached sealed model manifest");
                (manifest, bytes, false)
            }
            Err(cached_error) => {
                if manifest_path.exists() {
                    progress(&format!(
                        "refreshing invalid cached model manifest: {cached_error}"
                    ));
                }
                let manifest_url =
                    std::env::var("BURN_IMAGE_MODEL_MANIFEST_URL").unwrap_or_else(|_| {
                        base_url.resolve(
                            &ArtifactPath::new("manifest.json")
                                .expect("the canonical manifest path is valid"),
                        )
                    });
                progress(&format!("downloading sealed model manifest {manifest_url}"));
                let bytes = fetch_bounded_with_retries(
                    &manifest_url,
                    MAX_BROWSER_MANIFEST_BYTES,
                    "model manifest",
                )?;
                let manifest =
                    parse_expected_manifest(&bytes, variant, profile, require_canonical_digest)?;
                // The sealed manifest is the cache commit point. Keep it only in memory until every
                // declared payload has been size- and SHA-verified and atomically installed.
                if manifest_path.exists() {
                    fs::remove_file(&manifest_path).map_err(|error| {
                        NativeArtifactCacheError::message(format!(
                            "remove invalid cached manifest {}: {error}",
                            manifest_path.display()
                        ))
                    })?;
                }
                (manifest, bytes, true)
            }
        };

    let total_files = manifest.files.len();
    let mut cached_files = 0usize;
    for file in &manifest.files {
        let path = cache_root.join(file.path.as_str());
        if cached_file_matches(&path, file)? {
            cached_files += 1;
            continue;
        }
        if !manifest_needs_commit {
            // A missing or corrupt payload makes the prior cache incomplete. Remove its commit
            // point before repairing any file, then restore the same validated manifest only
            // after the complete payload set passes again.
            fs::remove_file(&manifest_path).map_err(|error| {
                NativeArtifactCacheError::message(format!(
                    "remove incomplete cache manifest {}: {error}",
                    manifest_path.display()
                ))
            })?;
            manifest_needs_commit = true;
        }
        progress(&format!(
            "downloading model artifact {}/{}: {} ({} bytes)",
            cached_files + 1,
            total_files,
            file.path,
            file.size
        ));
        let url = base_url.resolve(&file.path);
        download_verified_file_with_retries(&url, &path, file)?;
        cached_files += 1;
    }
    if manifest_needs_commit {
        install_bytes_atomically(&manifest_path, &manifest_bytes)?;
        progress("installed sealed model manifest as the cache commit point");
    }
    progress(&format!(
        "native model cache complete: {cached_files}/{total_files} files"
    ));
    Ok(())
}

fn read_expected_manifest(
    path: &Path,
    variant: BooguVariant,
    profile: BooguStorageProfile,
    require_canonical_digest: bool,
) -> Result<(ArtifactManifest, Vec<u8>), NativeArtifactCacheError> {
    let metadata = fs::metadata(path).map_err(|error| {
        NativeArtifactCacheError::message(format!(
            "read cached manifest metadata {}: {error}",
            path.display()
        ))
    })?;
    if metadata.len() == 0 || metadata.len() > MAX_BROWSER_MANIFEST_BYTES {
        return Err(NativeArtifactCacheError::message(format!(
            "cached manifest size {} is outside 1..={MAX_BROWSER_MANIFEST_BYTES}",
            metadata.len()
        )));
    }
    let bytes = fs::read(path).map_err(|error| {
        NativeArtifactCacheError::message(format!(
            "read cached manifest {}: {error}",
            path.display()
        ))
    })?;
    let manifest = parse_expected_manifest(&bytes, variant, profile, require_canonical_digest)?;
    Ok((manifest, bytes))
}

fn parse_expected_manifest(
    bytes: &[u8],
    variant: BooguVariant,
    profile: BooguStorageProfile,
    require_canonical_digest: bool,
) -> Result<ArtifactManifest, NativeArtifactCacheError> {
    let manifest: ArtifactManifest = serde_json::from_slice(bytes).map_err(|error| {
        NativeArtifactCacheError::message(format!("parse sealed model manifest: {error}"))
    })?;
    manifest.validate_sealed().map_err(|error| {
        NativeArtifactCacheError::message(format!("validate sealed model manifest: {error}"))
    })?;

    let expected_bundle = boogu_bundle_id(variant, profile);
    let expected_profile = boogu_profile_slug(profile);
    let descriptor = boogu_model_descriptor(variant);
    let expected_numeric = numeric_format(profile);
    if manifest.bundle.as_str() != expected_bundle
        || manifest.profile.as_str() != expected_profile
        || manifest.model != descriptor.id
        || manifest.model_revision != descriptor.revision
        || manifest.numeric_format != expected_numeric
    {
        return Err(NativeArtifactCacheError::message(format!(
            "model manifest identity mismatch: expected bundle={expected_bundle}, profile={expected_profile}, model={}, revision={}, numeric={expected_numeric:?}; found bundle={}, profile={}, model={}, revision={}, numeric={:?}",
            descriptor.id,
            descriptor.revision,
            manifest.bundle,
            manifest.profile,
            manifest.model,
            manifest.model_revision,
            manifest.numeric_format
        )));
    }
    if require_canonical_digest {
        validate_canonical_release_artifact_digest(
            variant,
            profile,
            manifest
                .content_digest
                .expect("sealed manifests contain a content digest"),
        )
        .map_err(|error| {
            NativeArtifactCacheError::message(format!(
                "validate canonical published model manifest: {error}"
            ))
        })?;
    }
    Ok(manifest)
}

fn cached_file_matches(path: &Path, file: &ArtifactFile) -> Result<bool, NativeArtifactCacheError> {
    let metadata = match fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(NativeArtifactCacheError::message(format!(
                "stat cached artifact {}: {error}",
                path.display()
            )));
        }
    };
    if !metadata.is_file() || metadata.len() != file.size {
        return Ok(false);
    }
    let mut reader = fs::File::open(path).map_err(|error| {
        NativeArtifactCacheError::message(format!(
            "open cached artifact {}: {error}",
            path.display()
        ))
    })?;
    let mut verifier = ArtifactVerifier::new(file, IntegrityPolicy::RequireSha256);
    let mut buffer = vec![0u8; COPY_BUFFER_BYTES];
    loop {
        let read = reader.read(&mut buffer).map_err(|error| {
            NativeArtifactCacheError::message(format!(
                "read cached artifact {}: {error}",
                path.display()
            ))
        })?;
        if read == 0 {
            break;
        }
        if verifier.update(&buffer[..read]).is_err() {
            return Ok(false);
        }
    }
    Ok(verifier.finish().is_ok())
}

fn download_verified_file_with_retries(
    url: &str,
    path: &Path,
    file: &ArtifactFile,
) -> Result<(), NativeArtifactCacheError> {
    let mut last_error = None;
    for attempt in 1..=DOWNLOAD_ATTEMPTS {
        match download_verified_file(url, path, file) {
            Ok(()) => return Ok(()),
            Err(error) => {
                last_error = Some(error.to_string());
                if attempt < DOWNLOAD_ATTEMPTS {
                    thread::sleep(retry_delay(attempt));
                }
            }
        }
    }
    Err(NativeArtifactCacheError::message(format!(
        "failed to cache model artifact `{url}` after {DOWNLOAD_ATTEMPTS} attempts: {}",
        last_error.unwrap_or_else(|| "unknown download error".into())
    )))
}

fn download_verified_file(
    url: &str,
    path: &Path,
    file: &ArtifactFile,
) -> Result<(), NativeArtifactCacheError> {
    ensure_parent_dir(path)?;
    let temporary = temporary_download_path(path);
    let result = (|| {
        let response = http_get(url)?;
        if let Some(expected) = response
            .header("Content-Length")
            .and_then(|value| value.parse::<u64>().ok())
            && expected != file.size
        {
            return Err(NativeArtifactCacheError::message(format!(
                "HTTP Content-Length for `{url}` is {expected}, manifest requires {}",
                file.size
            )));
        }
        let mut reader = response.into_reader();
        let mut writer = fs::File::create(&temporary).map_err(|error| {
            NativeArtifactCacheError::message(format!(
                "create temporary artifact {}: {error}",
                temporary.display()
            ))
        })?;
        let mut verifier = ArtifactVerifier::new(file, IntegrityPolicy::RequireSha256);
        let mut buffer = vec![0u8; COPY_BUFFER_BYTES];
        loop {
            let read = reader.read(&mut buffer).map_err(|error| {
                NativeArtifactCacheError::message(format!("read response body `{url}`: {error}"))
            })?;
            if read == 0 {
                break;
            }
            verifier.update(&buffer[..read]).map_err(|error| {
                NativeArtifactCacheError::message(format!(
                    "verify downloaded artifact `{url}`: {error}"
                ))
            })?;
            writer.write_all(&buffer[..read]).map_err(|error| {
                NativeArtifactCacheError::message(format!(
                    "write temporary artifact {}: {error}",
                    temporary.display()
                ))
            })?;
        }
        verifier.finish().map_err(|error| {
            NativeArtifactCacheError::message(format!(
                "verify downloaded artifact `{url}`: {error}"
            ))
        })?;
        writer.sync_all().map_err(|error| {
            NativeArtifactCacheError::message(format!(
                "sync temporary artifact {}: {error}",
                temporary.display()
            ))
        })?;
        install_temporary_file(&temporary, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn fetch_bounded_with_retries(
    url: &str,
    max_bytes: u64,
    label: &str,
) -> Result<Vec<u8>, NativeArtifactCacheError> {
    let mut last_error = None;
    for attempt in 1..=DOWNLOAD_ATTEMPTS {
        match fetch_bounded(url, max_bytes) {
            Ok(bytes) => return Ok(bytes),
            Err(error) => {
                last_error = Some(error.to_string());
                if attempt < DOWNLOAD_ATTEMPTS {
                    thread::sleep(retry_delay(attempt));
                }
            }
        }
    }
    Err(NativeArtifactCacheError::message(format!(
        "failed to download {label} `{url}` after {DOWNLOAD_ATTEMPTS} attempts: {}",
        last_error.unwrap_or_else(|| "unknown download error".into())
    )))
}

fn fetch_bounded(url: &str, max_bytes: u64) -> Result<Vec<u8>, NativeArtifactCacheError> {
    let response = http_get(url)?;
    let expected = response
        .header("Content-Length")
        .and_then(|value| value.parse::<u64>().ok());
    if let Some(expected) = expected
        && (expected == 0 || expected > max_bytes)
    {
        return Err(NativeArtifactCacheError::message(format!(
            "response `{url}` declares {expected} bytes; expected 1..={max_bytes}"
        )));
    }
    let mut reader = response.into_reader();
    let capacity = expected.unwrap_or(0).min(max_bytes) as usize;
    let mut bytes = Vec::with_capacity(capacity);
    let mut buffer = vec![0u8; COPY_BUFFER_BYTES.min(max_bytes as usize)];
    loop {
        let read = reader.read(&mut buffer).map_err(|error| {
            NativeArtifactCacheError::message(format!("read response body `{url}`: {error}"))
        })?;
        if read == 0 {
            break;
        }
        let next = (bytes.len() as u64).saturating_add(read as u64);
        if next > max_bytes {
            return Err(NativeArtifactCacheError::message(format!(
                "response `{url}` exceeded the {max_bytes}-byte bound"
            )));
        }
        bytes.extend_from_slice(&buffer[..read]);
    }
    if bytes.is_empty() {
        return Err(NativeArtifactCacheError::message(format!(
            "response `{url}` was empty"
        )));
    }
    if expected.is_some_and(|expected| expected != bytes.len() as u64) {
        return Err(NativeArtifactCacheError::message(format!(
            "response `{url}` Content-Length did not match the delivered body"
        )));
    }
    Ok(bytes)
}

fn http_get(url: &str) -> Result<ureq::Response, NativeArtifactCacheError> {
    ureq::AgentBuilder::new()
        .timeout_connect(CONNECT_TIMEOUT)
        .timeout_read(READ_TIMEOUT)
        .timeout_write(WRITE_TIMEOUT)
        .build()
        .get(url)
        .set("Accept", "*/*")
        .set("Accept-Encoding", "identity")
        .set("Cache-Control", "no-cache")
        .set("Pragma", "no-cache")
        .call()
        .map_err(|error| {
            NativeArtifactCacheError::message(format!("HTTP GET `{url}` failed: {error}"))
        })
}

fn install_bytes_atomically(path: &Path, bytes: &[u8]) -> Result<(), NativeArtifactCacheError> {
    ensure_parent_dir(path)?;
    let temporary = temporary_download_path(path);
    let result = (|| {
        let mut file = fs::File::create(&temporary).map_err(|error| {
            NativeArtifactCacheError::message(format!(
                "create temporary manifest {}: {error}",
                temporary.display()
            ))
        })?;
        file.write_all(bytes).map_err(|error| {
            NativeArtifactCacheError::message(format!(
                "write temporary manifest {}: {error}",
                temporary.display()
            ))
        })?;
        file.sync_all().map_err(|error| {
            NativeArtifactCacheError::message(format!(
                "sync temporary manifest {}: {error}",
                temporary.display()
            ))
        })?;
        install_temporary_file(&temporary, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn install_temporary_file(
    temporary: &Path,
    destination: &Path,
) -> Result<(), NativeArtifactCacheError> {
    install_temporary_file_platform(temporary, destination).map_err(|error| {
        NativeArtifactCacheError::message(format!(
            "install cached artifact {} -> {}: {error}",
            temporary.display(),
            destination.display()
        ))
    })
}

#[cfg(not(target_os = "windows"))]
fn install_temporary_file_platform(temporary: &Path, destination: &Path) -> std::io::Result<()> {
    // POSIX rename replaces an existing non-directory path atomically.
    fs::rename(temporary, destination)
}

#[cfg(target_os = "windows")]
fn install_temporary_file_platform(temporary: &Path, destination: &Path) -> std::io::Result<()> {
    // std::fs::rename cannot replace an existing file on Windows. This fallback has a small
    // replacement window, but the runtime never trusts a cache entry without rechecking its
    // manifest-bound size and SHA-256 digest.
    if destination.exists() {
        fs::remove_file(destination)?;
    }
    fs::rename(temporary, destination)
}

fn ensure_parent_dir(path: &Path) -> Result<(), NativeArtifactCacheError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            NativeArtifactCacheError::message(format!(
                "create artifact parent directory {}: {error}",
                parent.display()
            ))
        })?;
    }
    Ok(())
}

fn temporary_download_path(path: &Path) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_nanos())
        .unwrap_or_default();
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("artifact");
    path.with_file_name(format!(
        ".{file_name}.download-{}-{stamp}.part",
        std::process::id()
    ))
}

fn retry_delay(attempt: u32) -> Duration {
    Duration::from_millis(500_u64.saturating_mul(1_u64 << attempt.min(5)))
}

fn user_home_dir() -> Option<PathBuf> {
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        return Some(home);
    }
    #[cfg(target_os = "windows")]
    {
        if let Some(profile) = std::env::var_os("USERPROFILE").map(PathBuf::from) {
            return Some(profile);
        }
        let drive = std::env::var_os("HOMEDRIVE");
        let path = std::env::var_os("HOMEPATH");
        if let (Some(drive), Some(path)) = (drive, path) {
            return Some(PathBuf::from(format!(
                "{}{}",
                drive.to_string_lossy(),
                path.to_string_lossy()
            )));
        }
    }
    None
}

fn expand_home_path(path: PathBuf) -> PathBuf {
    let value = path.to_string_lossy();
    if value == "~" {
        return user_home_dir().unwrap_or(path);
    }
    if let Some(rest) = value.strip_prefix("~/") {
        return user_home_dir().map(|home| home.join(rest)).unwrap_or(path);
    }
    path
}

fn numeric_format(profile: BooguStorageProfile) -> NumericFormat {
    match profile {
        BooguStorageProfile::F16 => NumericFormat::F16,
        BooguStorageProfile::F16QwenVisionF32 => NumericFormat::Other("f16-qwen-vision-f32".into()),
        BooguStorageProfile::Q8sBlock32F32 => NumericFormat::Other("q8s-block32-f32".into()),
        BooguStorageProfile::Q8sBlock32F32QwenVisionF32 => {
            NumericFormat::Other("q8s-block32-f32-qwen-vision-f32".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        cell::Cell,
        collections::BTreeMap,
        io::{Read, Write},
        net::TcpListener,
        sync::{
            Arc,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        },
    };

    use burn_image::{
        ARTIFACT_MANIFEST_SCHEMA_VERSION, ArtifactBundleId, ArtifactFileRole, ArtifactProfileId,
        ModelId, Sha256Digest,
    };

    use super::*;

    fn write_tiny_remote(root: &Path, variant: BooguVariant, profile: BooguStorageProfile) {
        let payload = b"small verified payload";
        let payload_path = ArtifactPath::new("objects/tiny.bpk").unwrap();
        fs::create_dir_all(root.join("objects")).unwrap();
        fs::write(root.join(payload_path.as_str()), payload).unwrap();
        let descriptor = boogu_model_descriptor(variant);
        let mut manifest = ArtifactManifest {
            schema_version: ARTIFACT_MANIFEST_SCHEMA_VERSION,
            bundle: ArtifactBundleId::new(boogu_bundle_id(variant, profile)).unwrap(),
            profile: ArtifactProfileId::new(boogu_profile_slug(profile)).unwrap(),
            model: ModelId::new(descriptor.id.as_str()).unwrap(),
            model_revision: descriptor.revision,
            numeric_format: numeric_format(profile),
            components: Vec::new(),
            files: vec![ArtifactFile {
                path: payload_path,
                size: payload.len() as u64,
                sha256: Sha256Digest::calculate(payload),
                role: ArtifactFileRole::Metadata,
                component: None,
                shard: None,
            }],
            metadata: BTreeMap::new(),
            content_digest: None,
        };
        manifest.seal().unwrap();
        fs::write(
            root.join("manifest.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn canonical_cdn_and_cache_names_are_bundle_specific_correctness() {
        assert_eq!(
            boogu_bundle_id(
                BooguVariant::Image01Turbo,
                BooguStorageProfile::F16QwenVisionF32
            ),
            "boogu-image-0.1-turbo-f16-qwen-vision-f32"
        );
        let base = RemoteBaseUrl::new(format!(
            "{}/{}",
            DEFAULT_BURN_IMAGE_MODEL_ROOT_URL,
            boogu_bundle_id(
                BooguVariant::Image01EditTurbo1k5,
                BooguStorageProfile::F16QwenVisionF32
            )
        ))
        .unwrap();
        assert_eq!(
            base.as_str(),
            "https://aberration.technology/model/boogu-image-0.1-edit-turbo-1k5-f16-qwen-vision-f32"
        );
        assert!(
            canonical_native_boogu_model_base_url(
                BooguVariant::Image01Turbo,
                BooguStorageProfile::F16
            )
            .unwrap_err()
            .to_string()
            .contains("no canonical published Aberration CDN bundle")
        );
    }

    #[test]
    fn native_cache_manifest_cannot_alias_a_different_profile_correctness() {
        let temp = tempfile::tempdir().unwrap();
        write_tiny_remote(
            temp.path(),
            BooguVariant::Image01Turbo,
            BooguStorageProfile::F16,
        );
        let bytes = fs::read(temp.path().join("manifest.json")).unwrap();
        let error = parse_expected_manifest(
            &bytes,
            BooguVariant::Image01Turbo,
            BooguStorageProfile::F16QwenVisionF32,
            false,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("model manifest identity mismatch"),
            "{error}"
        );
        assert!(error.contains("f16-qwen-vision-f32"), "{error}");
    }

    #[test]
    fn canonical_cdn_manifest_requires_the_pinned_release_digest_correctness() {
        let temp = tempfile::tempdir().unwrap();
        write_tiny_remote(
            temp.path(),
            BooguVariant::Image01Turbo,
            BooguStorageProfile::F16QwenVisionF32,
        );
        let bytes = fs::read(temp.path().join("manifest.json")).unwrap();
        let error = parse_expected_manifest(
            &bytes,
            BooguVariant::Image01Turbo,
            BooguStorageProfile::F16QwenVisionF32,
            true,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("requires sealed artifact digest"), "{error}");
    }

    #[test]
    fn native_cache_downloads_reuses_and_repairs_tiny_bundle_correctness() {
        let temp = tempfile::tempdir().unwrap();
        let remote = temp.path().join("remote");
        fs::create_dir_all(&remote).unwrap();
        let variant = BooguVariant::Image01Turbo;
        let profile = BooguStorageProfile::F16;
        write_tiny_remote(&remote, variant, profile);
        let server = TestServer::serve(remote);
        let base = RemoteBaseUrl::new(&server.base_url).unwrap();
        let cache = temp.path().join("cache");

        cache_remote_bundle(variant, profile, &base, &cache, false, &|_| {}).unwrap();
        assert_eq!(
            fs::read(cache.join("objects/tiny.bpk")).unwrap(),
            b"small verified payload"
        );
        assert_eq!(server.requests.load(Ordering::SeqCst), 2);

        server.requests.store(0, Ordering::SeqCst);
        cache_remote_bundle(variant, profile, &base, &cache, false, &|_| {}).unwrap();
        assert_eq!(server.requests.load(Ordering::SeqCst), 0);

        fs::write(cache.join("objects/tiny.bpk"), b"corrupt").unwrap();
        let observed_repair_without_commit = Cell::new(false);
        cache_remote_bundle(variant, profile, &base, &cache, false, &|message| {
            if message.starts_with("downloading model artifact") {
                assert!(!cache.join("manifest.json").exists());
                observed_repair_without_commit.set(true);
            }
        })
        .unwrap();
        assert!(observed_repair_without_commit.get());
        assert!(cache.join("manifest.json").is_file());
        assert_eq!(server.requests.load(Ordering::SeqCst), 1);
        assert_eq!(
            fs::read(cache.join("objects/tiny.bpk")).unwrap(),
            b"small verified payload"
        );
    }

    struct TestServer {
        base_url: String,
        requests: Arc<AtomicUsize>,
        stop: Arc<AtomicBool>,
        handle: Option<std::thread::JoinHandle<()>>,
    }

    impl TestServer {
        fn serve(root: PathBuf) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(true).unwrap();
            let address = listener.local_addr().unwrap();
            let requests = Arc::new(AtomicUsize::new(0));
            let stop = Arc::new(AtomicBool::new(false));
            let thread_requests = Arc::clone(&requests);
            let thread_stop = Arc::clone(&stop);
            let handle = std::thread::spawn(move || {
                while !thread_stop.load(Ordering::SeqCst) {
                    match listener.accept() {
                        Ok((mut stream, _)) => {
                            thread_requests.fetch_add(1, Ordering::SeqCst);
                            let mut request = [0u8; 2048];
                            let read = stream.read(&mut request).unwrap_or(0);
                            let request = String::from_utf8_lossy(&request[..read]);
                            let path = request
                                .lines()
                                .next()
                                .and_then(|line| line.split_whitespace().nth(1))
                                .unwrap_or("/")
                                .trim_start_matches('/')
                                .split('?')
                                .next()
                                .unwrap_or("");
                            let (status, body) = match fs::read(root.join(path)) {
                                Ok(bytes) => ("200 OK", bytes),
                                Err(_) => ("404 Not Found", Vec::new()),
                            };
                            let header = format!(
                                "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                                body.len()
                            );
                            let _ = stream.write_all(header.as_bytes());
                            let _ = stream.write_all(&body);
                            let _ = stream.flush();
                        }
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            std::thread::sleep(Duration::from_millis(5));
                        }
                        Err(_) => break,
                    }
                }
            });
            Self {
                base_url: format!("http://{address}"),
                requests,
                stop,
                handle: Some(handle),
            }
        }
    }

    impl Drop for TestServer {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::SeqCst);
            let _ = std::net::TcpStream::connect(
                self.base_url
                    .strip_prefix("http://")
                    .expect("test server URL is HTTP"),
            );
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }
}
