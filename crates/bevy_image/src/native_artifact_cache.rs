//! Verified native CDN cache for sealed Boogu artifact bundles.
//!
//! The cache mirrors the immutable CDN tree under
//! `~/.burn_image/models/<bundle-id>`. A manifest is installed only after its seal and exact
//! variant/profile/bundle identity validate. Every direct payload or manifest-sealed transport
//! part is streamed through its size and SHA-256 contract before an atomic rename makes it visible
//! to the runtime.

use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use burn_boogu::artifacts::{
    artifact_bundle_id_matches_selection, canonical_published_bundle,
    validate_canonical_release_artifact_digest,
};
use burn_boogu::{BooguVariant, artifacts::BooguStorageProfile, boogu_model_descriptor};
use burn_image::{
    ArtifactBundleFetcher, ArtifactBundleId, ArtifactDependency, ArtifactFile, ArtifactFileRole,
    ArtifactManifest, ArtifactPath, ArtifactReadError, ArtifactSource, ArtifactTransportLayout,
    ArtifactTransportObject, ArtifactTransportPart, ArtifactVerifier, FilesystemArtifactCache,
    IntegrityPolicy, NumericFormat, RemoteBaseUrl, Sha256Digest, VerifiedArtifactTransportLayout,
};
use thiserror::Error;

use crate::{
    BOOGU_CDN_ROOT, MAX_BROWSER_MANIFEST_BYTES, boogu_bundle_id, boogu_profile_slug,
    boogu_source_bundle_id, sibling_bundle_base_url,
};

const QWEN_DEPENDENCY_ROLE: &str = "qwen";
const VAE_DEPENDENCY_ROLE: &str = "vae";

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

/// Exact local roots used to assemble one Boogu runtime.
///
/// Standalone schema-v1 bundles point all three roots at the same directory. Schema-v2
/// compositions keep independently sealed Qwen and VAE bundles in digest-isolated cache
/// directories.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedNativeBooguArtifactDirectories {
    pipeline_root: PathBuf,
    qwen_root: PathBuf,
    vae_root: PathBuf,
}

impl ResolvedNativeBooguArtifactDirectories {
    pub fn pipeline_root(&self) -> &Path {
        &self.pipeline_root
    }

    pub fn qwen_root(&self) -> &Path {
        &self.qwen_root
    }

    pub fn vae_root(&self) -> &Path {
        &self.vae_root
    }

    pub fn is_standalone(&self) -> bool {
        self.pipeline_root == self.qwen_root && self.pipeline_root == self.vae_root
    }

    fn standalone(root: PathBuf) -> Self {
        Self {
            pipeline_root: root.clone(),
            qwen_root: root.clone(),
            vae_root: root,
        }
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
    Ok(format!(
        "{DEFAULT_BURN_IMAGE_MODEL_ROOT_URL}/{}",
        published.bundle_id
    ))
}

/// Default exact cache directory for one canonical bundle.
///
/// `BURN_IMAGE_CACHE_DIR` replaces the broad `~/.burn_image` root;
/// `BURN_IMAGE_MODEL_CACHE_DIR` replaces the final bundle directory.
/// A custom remote source is isolated under the standalone source tuple unless the exact-directory
/// override is set, so it cannot reuse or populate the canonical production cache by accident.
pub fn default_native_boogu_model_cache_root(
    variant: BooguVariant,
    profile: BooguStorageProfile,
) -> Result<PathBuf, NativeArtifactCacheError> {
    configured_native_boogu_model_cache_root(variant, profile, true)
}

fn configured_native_boogu_model_cache_root(
    variant: BooguVariant,
    profile: BooguStorageProfile,
    canonical_source: bool,
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
    Ok(native_boogu_model_cache_root_under(
        &broad_root,
        variant,
        profile,
        canonical_source,
    ))
}

fn native_boogu_model_cache_root_under(
    broad_root: &Path,
    variant: BooguVariant,
    profile: BooguStorageProfile,
    canonical_source: bool,
) -> PathBuf {
    let cache_key = if canonical_source {
        boogu_bundle_id(variant, profile)
    } else {
        boogu_source_bundle_id(variant, profile)
    };
    broad_root
        .join(DEFAULT_BURN_IMAGE_MODEL_CACHE_SUBDIR)
        .join(cache_key)
}

/// Resolve a local override or materialize a remote sealed bundle in the verified native cache.
pub fn resolve_native_boogu_artifact_directory<F>(
    variant: BooguVariant,
    profile: BooguStorageProfile,
    source: &ArtifactSource,
    progress: F,
) -> Result<ResolvedNativeBooguArtifactDirectories, NativeArtifactCacheError>
where
    F: Fn(&str),
{
    let ArtifactSource::Remote { base_url } = source else {
        let root = source
            .local_root()
            .map(Path::to_path_buf)
            .ok_or_else(|| NativeArtifactCacheError::message("invalid local artifact source"))?;
        return resolve_local_composition(variant, profile, root, &progress);
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
    let cache_root =
        configured_native_boogu_model_cache_root(variant, profile, require_canonical_digest)?;
    let manifest = cache_remote_bundle(
        variant,
        profile,
        &base_url,
        &cache_root,
        require_canonical_digest,
        &progress,
    )?;
    resolve_remote_composition(&base_url, &cache_root, &manifest, &progress)
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
) -> Result<ArtifactManifest, NativeArtifactCacheError>
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
        "resolving pipeline bundle cache under {}",
        cache_root.display()
    ));

    let manifest_path = cache_root.join("manifest.json");
    let (manifest, manifest_bytes, mut manifest_needs_commit) =
        match read_expected_manifest(&manifest_path, variant, profile, require_canonical_digest) {
            Ok((manifest, bytes)) => {
                progress(&format!("using cached sealed {} manifest", manifest.bundle));
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

    let transport = resolve_cached_or_remote_transport_layout(cache_root, base_url, &manifest)?;
    let total_files = manifest.files.len();
    let mut cached_files = 0usize;
    for file in &manifest.files {
        let path = cache_root.join(file.path.as_str());
        let transport_object = match (file.role, transport.as_ref()) {
            (ArtifactFileRole::Weights, Some(layout)) => {
                Some(layout.object(&file.path).ok_or_else(|| {
                    NativeArtifactCacheError::message(format!(
                        "verified transport layout omits logical artifact {}",
                        file.path
                    ))
                })?)
            }
            _ => None,
        };
        let cached = if let Some(object) = transport_object {
            fs::symlink_metadata(&path)
                .is_err_and(|error| error.kind() == std::io::ErrorKind::NotFound)
                && cached_transport_object_matches(cache_root, object)?
        } else {
            cached_file_matches(&path, file)?
        };
        if cached {
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
            "downloading {} artifact {}/{}: {} ({} bytes)",
            manifest.bundle,
            cached_files + 1,
            total_files,
            file.path,
            file.size
        ));
        if let Some(object) = transport_object {
            download_verified_transport_parts(base_url, cache_root, file, object)?;
            remove_stale_cached_logical_weight(&path)?;
        } else {
            let url = base_url.resolve(&file.path);
            download_verified_file_with_retries(&url, &path, file)?;
        }
        cached_files += 1;
    }
    if manifest_needs_commit {
        install_bytes_atomically(&manifest_path, &manifest_bytes)?;
        progress(&format!(
            "installed sealed {} manifest as the cache commit point",
            manifest.bundle
        ));
    }
    progress(&format!(
        "native {} cache complete: {cached_files}/{total_files} files",
        manifest.bundle
    ));
    Ok(manifest)
}

fn resolve_local_composition<F>(
    variant: BooguVariant,
    profile: BooguStorageProfile,
    pipeline_root: PathBuf,
    progress: &F,
) -> Result<ResolvedNativeBooguArtifactDirectories, NativeArtifactCacheError>
where
    F: Fn(&str),
{
    let manifest_path = pipeline_root.join("manifest.json");
    let (manifest, _) = read_expected_manifest(&manifest_path, variant, profile, false)?;
    let Some((qwen, vae)) = required_component_dependencies(&manifest)? else {
        progress(&format!(
            "using standalone artifact directory {}",
            pipeline_root.display()
        ));
        return Ok(ResolvedNativeBooguArtifactDirectories::standalone(
            pipeline_root,
        ));
    };
    let parent = pipeline_root.parent().ok_or_else(|| {
        NativeArtifactCacheError::message(format!(
            "composed local artifact directory has no sibling parent: {}",
            pipeline_root.display()
        ))
    })?;
    let qwen_root = validate_local_dependency(parent, qwen, progress)?;
    let vae_root = validate_local_dependency(parent, vae, progress)?;
    Ok(ResolvedNativeBooguArtifactDirectories {
        pipeline_root,
        qwen_root,
        vae_root,
    })
}

fn resolve_remote_composition<F>(
    pipeline_base_url: &RemoteBaseUrl,
    pipeline_cache_root: &Path,
    manifest: &ArtifactManifest,
    progress: &F,
) -> Result<ResolvedNativeBooguArtifactDirectories, NativeArtifactCacheError>
where
    F: Fn(&str),
{
    let Some((qwen, vae)) = required_component_dependencies(manifest)? else {
        progress("resolved standalone remote bundle");
        return Ok(ResolvedNativeBooguArtifactDirectories::standalone(
            pipeline_cache_root.to_owned(),
        ));
    };
    let cache_parent = pipeline_cache_root.parent().ok_or_else(|| {
        NativeArtifactCacheError::message(format!(
            "pipeline cache has no dependency parent: {}",
            pipeline_cache_root.display()
        ))
    })?;
    let qwen_root = cache_remote_dependency(pipeline_base_url, cache_parent, qwen, progress)?;
    let vae_root = cache_remote_dependency(pipeline_base_url, cache_parent, vae, progress)?;
    Ok(ResolvedNativeBooguArtifactDirectories {
        pipeline_root: pipeline_cache_root.to_owned(),
        qwen_root,
        vae_root,
    })
}

fn required_component_dependencies(
    manifest: &ArtifactManifest,
) -> Result<Option<(&ArtifactDependency, &ArtifactDependency)>, NativeArtifactCacheError> {
    if manifest.dependencies.is_empty() {
        return if manifest.schema_version == burn_image::ARTIFACT_MANIFEST_SCHEMA_V1 {
            Ok(None)
        } else {
            Err(NativeArtifactCacheError::message(format!(
                "schema-v2 composed Boogu manifest {} omits qwen and vae dependencies",
                manifest.bundle
            )))
        };
    }
    if manifest.dependencies.len() != 2 {
        return Err(NativeArtifactCacheError::message(format!(
            "composed Boogu manifest {} must contain exactly qwen and vae dependencies; found {}",
            manifest.bundle,
            manifest.dependencies.len()
        )));
    }
    if manifest
        .metadata
        .get("component_dependency_count")
        .map(String::as_str)
        != Some("2")
    {
        return Err(NativeArtifactCacheError::message(format!(
            "composed Boogu manifest {} does not declare component_dependency_count=2",
            manifest.bundle
        )));
    }
    let dependency = |role: &str| {
        manifest
            .dependencies
            .iter()
            .find(|dependency| dependency.role.as_str() == role)
            .ok_or_else(|| {
                NativeArtifactCacheError::message(format!(
                    "composed Boogu manifest {} omits required {role} dependency",
                    manifest.bundle
                ))
            })
    };
    Ok(Some((
        dependency(QWEN_DEPENDENCY_ROLE)?,
        dependency(VAE_DEPENDENCY_ROLE)?,
    )))
}

fn validate_local_dependency<F>(
    sibling_parent: &Path,
    dependency: &ArtifactDependency,
    progress: &F,
) -> Result<PathBuf, NativeArtifactCacheError>
where
    F: Fn(&str),
{
    let root = sibling_parent.join(dependency.bundle.as_str());
    let manifest_path = root.join("manifest.json");
    let bytes = read_bounded_local_manifest(&manifest_path)?;
    parse_dependency_manifest(&bytes, dependency)?;
    progress(&format!(
        "validated local {} component bundle {} under {}",
        dependency.role,
        dependency.bundle,
        root.display()
    ));
    Ok(root)
}

fn cache_remote_dependency<F>(
    pipeline_base_url: &RemoteBaseUrl,
    cache_parent: &Path,
    dependency: &ArtifactDependency,
    progress: &F,
) -> Result<PathBuf, NativeArtifactCacheError>
where
    F: Fn(&str),
{
    let cache = FilesystemArtifactCache::new(cache_parent, MAX_BROWSER_MANIFEST_BYTES)
        .map_err(|error| NativeArtifactCacheError::message(error.to_string()))?;
    let mut fetcher = UreqSiblingBundleFetcher::new(pipeline_base_url.clone());
    progress(&format!(
        "resolving {} component bundle {} ({})",
        dependency.role, dependency.bundle, dependency.content_digest
    ));
    let directory = cache
        .ensure_dependency(dependency, &mut fetcher)
        .map_err(|error| NativeArtifactCacheError::message(error.to_string()))?;
    progress(&format!(
        "native {} component {} cache complete: {} files",
        dependency.role,
        dependency.bundle,
        directory.manifest().files.len()
    ));
    Ok(directory.root().to_owned())
}

struct UreqSiblingBundleFetcher {
    pipeline_base_url: RemoteBaseUrl,
}

impl UreqSiblingBundleFetcher {
    fn new(pipeline_base_url: RemoteBaseUrl) -> Self {
        Self { pipeline_base_url }
    }

    fn bundle_base_url(
        &self,
        bundle: &ArtifactBundleId,
    ) -> Result<RemoteBaseUrl, ArtifactReadError> {
        sibling_bundle_base_url(&self.pipeline_base_url, bundle)
            .map_err(|error| ArtifactReadError::transport(error.to_string()))
    }
}

impl ArtifactBundleFetcher for UreqSiblingBundleFetcher {
    fn fetch_manifest(
        &mut self,
        bundle: &ArtifactBundleId,
        maximum_bytes: u64,
    ) -> Result<Vec<u8>, ArtifactReadError> {
        let base = self.bundle_base_url(bundle)?;
        let url = base.resolve(
            &ArtifactPath::new("manifest.json").expect("canonical manifest path is valid"),
        );
        let bytes = fetch_bounded_with_retries(&url, maximum_bytes, "dependency manifest")
            .map_err(|error| ArtifactReadError::transport(error.to_string()))?;
        Ok(bytes)
    }

    fn fetch_file(
        &mut self,
        bundle: &ArtifactBundleId,
        file: &ArtifactFile,
        destination: &mut dyn Write,
    ) -> Result<(), ArtifactReadError> {
        let base = self.bundle_base_url(bundle)?;
        let url = base.resolve(&file.path);
        let response =
            http_get(&url).map_err(|error| ArtifactReadError::transport(error.to_string()))?;
        if let Some(expected) = response
            .header("Content-Length")
            .and_then(|value| value.parse::<u64>().ok())
            && expected != file.size
        {
            return Err(ArtifactReadError::transport(format!(
                "HTTP Content-Length for `{url}` is {expected}, manifest requires {}",
                file.size
            )));
        }
        let mut input = response.into_reader();
        io_copy_bounded(&mut input, destination, file.size, &url)
    }
}

fn io_copy_bounded(
    input: &mut dyn Read,
    output: &mut dyn Write,
    exact_bytes: u64,
    url: &str,
) -> Result<(), ArtifactReadError> {
    let mut copied = 0_u64;
    let mut buffer = vec![0_u8; COPY_BUFFER_BYTES];
    loop {
        let read = input
            .read(&mut buffer)
            .map_err(|error| ArtifactReadError::transport(format!("read `{url}`: {error}")))?;
        if read == 0 {
            break;
        }
        copied = copied.saturating_add(read as u64);
        if copied > exact_bytes {
            return Err(ArtifactReadError::transport(format!(
                "response `{url}` exceeded sealed size {exact_bytes}"
            )));
        }
        output
            .write_all(&buffer[..read])
            .map_err(|error| ArtifactReadError::transport(format!("cache `{url}`: {error}")))?;
    }
    if copied != exact_bytes {
        return Err(ArtifactReadError::transport(format!(
            "response `{url}` delivered {copied} bytes; expected {exact_bytes}"
        )));
    }
    Ok(())
}

fn read_bounded_local_manifest(path: &Path) -> Result<Vec<u8>, NativeArtifactCacheError> {
    let metadata = fs::metadata(path).map_err(|error| {
        NativeArtifactCacheError::message(format!("inspect manifest {}: {error}", path.display()))
    })?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > MAX_BROWSER_MANIFEST_BYTES {
        return Err(NativeArtifactCacheError::message(format!(
            "manifest {} is not a regular file within 1..={MAX_BROWSER_MANIFEST_BYTES} bytes",
            path.display()
        )));
    }
    fs::read(path).map_err(|error| {
        NativeArtifactCacheError::message(format!("read manifest {}: {error}", path.display()))
    })
}

fn parse_dependency_manifest(
    bytes: &[u8],
    dependency: &ArtifactDependency,
) -> Result<ArtifactManifest, NativeArtifactCacheError> {
    let manifest: ArtifactManifest = serde_json::from_slice(bytes).map_err(|error| {
        NativeArtifactCacheError::message(format!(
            "parse {} dependency manifest: {error}",
            dependency.role
        ))
    })?;
    dependency
        .validate_resolved_manifest(&manifest)
        .map_err(|error| {
            NativeArtifactCacheError::message(format!(
                "validate sealed {} dependency {}: {error}",
                dependency.role, dependency.bundle
            ))
        })?;
    if !manifest.dependencies.is_empty() {
        return Err(NativeArtifactCacheError::message(format!(
            "model component bundle {} must be a dependency leaf",
            manifest.bundle
        )));
    }
    Ok(manifest)
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
    let source_bundle = boogu_source_bundle_id(variant, profile);
    let expected_profile = boogu_profile_slug(profile);
    let descriptor = boogu_model_descriptor(variant);
    let expected_numeric = numeric_format(profile);
    let bundle_matches = if require_canonical_digest {
        manifest.bundle.as_str() == expected_bundle
    } else {
        artifact_bundle_id_matches_selection(variant, profile, manifest.bundle.as_str())
    };
    if !bundle_matches
        || manifest.profile.as_str() != expected_profile
        || manifest.model != descriptor.id
        || manifest.model_revision != descriptor.revision
        || manifest.numeric_format != expected_numeric
    {
        return Err(NativeArtifactCacheError::message(format!(
            "model manifest identity mismatch: expected bundle={expected_bundle} (an explicit local conversion source may use {source_bundle}), profile={expected_profile}, model={}, revision={}, numeric={expected_numeric:?}; found bundle={}, profile={}, model={}, revision={}, numeric={:?}",
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
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(NativeArtifactCacheError::message(format!(
                "stat cached artifact {}: {error}",
                path.display()
            )));
        }
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() != file.size {
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

fn resolve_cached_or_remote_transport_layout(
    cache_root: &Path,
    base_url: &RemoteBaseUrl,
    manifest: &ArtifactManifest,
) -> Result<Option<VerifiedArtifactTransportLayout>, NativeArtifactCacheError> {
    let Some(file) = ArtifactTransportLayout::declared_file(manifest)
        .map_err(|error| NativeArtifactCacheError::message(error.to_string()))?
    else {
        return Ok(None);
    };
    let cached_path = cache_root.join(file.path.as_str());
    if cached_file_matches(&cached_path, file)? {
        let bytes = fs::read(&cached_path).map_err(|error| {
            NativeArtifactCacheError::message(format!(
                "read cached transport layout {}: {error}",
                cached_path.display()
            ))
        })?;
        let layout = ArtifactTransportLayout::parse_and_validate(manifest, &bytes)
            .map_err(|error| NativeArtifactCacheError::message(error.to_string()))?;
        return Ok(Some(layout));
    }
    fetch_remote_transport_layout(base_url, manifest)
}

fn fetch_remote_transport_layout(
    base_url: &RemoteBaseUrl,
    manifest: &ArtifactManifest,
) -> Result<Option<VerifiedArtifactTransportLayout>, NativeArtifactCacheError> {
    let Some(file) = ArtifactTransportLayout::declared_file(manifest)
        .map_err(|error| NativeArtifactCacheError::message(error.to_string()))?
    else {
        return Ok(None);
    };
    let url = base_url.resolve(&file.path);
    let bytes = fetch_bounded_with_retries(&url, file.size, "artifact transport layout")?;
    ArtifactTransportLayout::parse_and_validate(manifest, &bytes)
        .map(Some)
        .map_err(|error| NativeArtifactCacheError::message(error.to_string()))
}

fn fetch_verified_transport_part_with_retries(
    base_url: &RemoteBaseUrl,
    part: &ArtifactTransportPart,
) -> Result<Vec<u8>, NativeArtifactCacheError> {
    let url = base_url.resolve(&part.path);
    let mut last_error = None;
    for attempt in 1..=DOWNLOAD_ATTEMPTS {
        match fetch_bounded(&url, part.size) {
            Ok(bytes)
                if u64::try_from(bytes.len()).ok() == Some(part.size)
                    && Sha256Digest::calculate(&bytes) == part.sha256 =>
            {
                return Ok(bytes);
            }
            Ok(bytes) => {
                last_error = Some(format!(
                    "transport part integrity mismatch: expected {}/{} bytes, found {}/{}",
                    part.sha256,
                    part.size,
                    Sha256Digest::calculate(&bytes),
                    bytes.len()
                ));
            }
            Err(error) => last_error = Some(error.to_string()),
        }
        if attempt < DOWNLOAD_ATTEMPTS {
            thread::sleep(retry_delay(attempt));
        }
    }
    Err(NativeArtifactCacheError::message(format!(
        "failed to authenticate transport part `{url}` after {DOWNLOAD_ATTEMPTS} attempts: {}",
        last_error.unwrap_or_else(|| "unknown transport error".into())
    )))
}

fn cached_transport_object_matches(
    cache_root: &Path,
    object: &ArtifactTransportObject,
) -> Result<bool, NativeArtifactCacheError> {
    for part in &object.parts {
        if !cached_file_matches(
            &cache_root.join(part.path.as_str()),
            &transport_part_file(part),
        )? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn download_verified_transport_parts(
    base_url: &RemoteBaseUrl,
    cache_root: &Path,
    file: &ArtifactFile,
    object: &ArtifactTransportObject,
) -> Result<(), NativeArtifactCacheError> {
    if object.path != file.path || object.size != file.size || object.sha256 != file.sha256 {
        return Err(NativeArtifactCacheError::message(format!(
            "transport object identity differs from sealed logical artifact {}",
            file.path
        )));
    }
    for part in &object.parts {
        let physical = transport_part_file(part);
        let path = cache_root.join(part.path.as_str());
        if cached_file_matches(&path, &physical)? {
            continue;
        }
        let bytes = fetch_verified_transport_part_with_retries(base_url, part)?;
        install_bytes_atomically(&path, &bytes)?;
        if !cached_file_matches(&path, &physical)? {
            return Err(NativeArtifactCacheError::message(format!(
                "installed transport part {} failed its sealed identity",
                part.path
            )));
        }
    }
    Ok(())
}

fn transport_part_file(part: &ArtifactTransportPart) -> ArtifactFile {
    ArtifactFile {
        path: part.path.clone(),
        size: part.size,
        sha256: part.sha256,
        role: ArtifactFileRole::Other,
        component: None,
        shard: None,
    }
}

fn remove_stale_cached_logical_weight(path: &Path) -> Result<(), NativeArtifactCacheError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_dir() => Err(NativeArtifactCacheError::message(format!(
            "cached logical weight path is a directory: {}",
            path.display()
        ))),
        Ok(_) => fs::remove_file(path).map_err(|error| {
            NativeArtifactCacheError::message(format!(
                "remove stale cached logical weight {}: {error}",
                path.display()
            ))
        }),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(NativeArtifactCacheError::message(format!(
            "inspect stale cached logical weight {}: {error}",
            path.display()
        ))),
    }
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
        BooguStorageProfile::Q4sBlockUpTo128F32 => {
            NumericFormat::Other("q4s-block-up-to128-f32".into())
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
        ARTIFACT_MANIFEST_SCHEMA_V1, ARTIFACT_MANIFEST_SCHEMA_V2, ArtifactBundleId,
        ArtifactComponentId, ArtifactDependency, ArtifactFileRole, ArtifactProfileId, ModelId,
        Sha256Digest,
    };

    use super::*;

    fn write_tiny_remote(root: &Path, variant: BooguVariant, profile: BooguStorageProfile) {
        write_tiny_remote_with_bundle(root, variant, profile, boogu_bundle_id(variant, profile));
    }

    fn write_tiny_remote_with_bundle(
        root: &Path,
        variant: BooguVariant,
        profile: BooguStorageProfile,
        bundle: String,
    ) {
        let payload = b"small verified payload";
        let payload_path = ArtifactPath::new("objects/tiny.bpk").unwrap();
        fs::create_dir_all(root.join("objects")).unwrap();
        fs::write(root.join(payload_path.as_str()), payload).unwrap();
        let descriptor = boogu_model_descriptor(variant);
        let mut manifest = ArtifactManifest {
            schema_version: ARTIFACT_MANIFEST_SCHEMA_V1,
            bundle: ArtifactBundleId::new(bundle).unwrap(),
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
            dependencies: Vec::new(),
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

    fn tiny_dependency(role: &str, manifest: &ArtifactManifest) -> ArtifactDependency {
        ArtifactDependency {
            role: ArtifactComponentId::new(role).unwrap(),
            bundle: manifest.bundle.clone(),
            profile: manifest.profile.clone(),
            model: manifest.model.clone(),
            model_revision: manifest.model_revision.clone(),
            content_digest: manifest.content_digest.unwrap(),
        }
    }

    fn write_tiny_dependency(root: &Path, bundle: &str, payload: &[u8]) -> ArtifactManifest {
        let path = ArtifactPath::new("objects/tiny.bpk").unwrap();
        fs::create_dir_all(root.join("objects")).unwrap();
        fs::write(root.join(path.as_str()), payload).unwrap();
        let mut manifest = ArtifactManifest {
            schema_version: ARTIFACT_MANIFEST_SCHEMA_V1,
            bundle: ArtifactBundleId::new(bundle).unwrap(),
            profile: ArtifactProfileId::new("tiny-component").unwrap(),
            model: ModelId::new(format!("test/{bundle}")).unwrap(),
            model_revision: "exact-revision".into(),
            numeric_format: NumericFormat::F16,
            components: Vec::new(),
            files: vec![ArtifactFile {
                path,
                size: payload.len() as u64,
                sha256: Sha256Digest::calculate(payload),
                role: ArtifactFileRole::Metadata,
                component: None,
                shard: None,
            }],
            dependencies: Vec::new(),
            metadata: BTreeMap::new(),
            content_digest: None,
        };
        manifest.seal().unwrap();
        fs::write(
            root.join("manifest.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        manifest
    }

    fn write_tiny_transport_bundle(
        root: &Path,
        profile: BooguStorageProfile,
        bundle: &str,
        model: &str,
        model_revision: &str,
        payload: &[u8],
    ) -> ArtifactManifest {
        use burn_image::{
            ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES, ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES_KEY,
            ARTIFACT_TARGET_MAX_SEMANTIC_SHARD_BYTES_KEY,
            ARTIFACT_TARGET_MAX_TRANSPORT_SHARD_BYTES_KEY, ARTIFACT_TRANSPORT_LAYOUT_PATH,
            ARTIFACT_TRANSPORT_LAYOUT_PATH_KEY, ARTIFACT_TRANSPORT_LAYOUT_SCHEMA_KEY,
            ARTIFACT_TRANSPORT_LAYOUT_SCHEMA_VERSION, ARTIFACT_TRANSPORT_MAX_PART_BYTES,
            ARTIFACT_TRANSPORT_PART_TARGET_BYTES_KEY, ARTIFACT_TRANSPORT_PARTS_REQUIRED_KEY,
            ARTIFACT_TRANSPORT_TARGET_PART_BYTES, ArtifactTransportLayout, ArtifactTransportObject,
            ArtifactTransportPart,
        };

        let logical_path = ArtifactPath::new("objects/tiny.bpk").unwrap();
        let digest = Sha256Digest::calculate(payload);
        let part_path = ArtifactPath::new(format!("transport/{digest}.part")).unwrap();
        fs::create_dir_all(root.join("transport")).unwrap();
        fs::create_dir_all(root.join("metadata")).unwrap();
        fs::write(root.join(part_path.as_str()), payload).unwrap();
        let layout = ArtifactTransportLayout {
            schema_version: ARTIFACT_TRANSPORT_LAYOUT_SCHEMA_VERSION,
            bundle: ArtifactBundleId::new(bundle).unwrap(),
            profile: ArtifactProfileId::new(boogu_profile_slug(profile)).unwrap(),
            model: ModelId::new(model).unwrap(),
            model_revision: model_revision.into(),
            target_part_bytes: ARTIFACT_TRANSPORT_TARGET_PART_BYTES,
            hard_max_part_bytes: ARTIFACT_TRANSPORT_MAX_PART_BYTES,
            objects: vec![ArtifactTransportObject {
                path: logical_path.clone(),
                size: payload.len() as u64,
                sha256: digest,
                parts: vec![ArtifactTransportPart {
                    path: part_path,
                    offset: 0,
                    size: payload.len() as u64,
                    sha256: digest,
                }],
            }],
        };
        let layout_bytes = serde_json::to_vec_pretty(&layout).unwrap();
        fs::write(root.join(ARTIFACT_TRANSPORT_LAYOUT_PATH), &layout_bytes).unwrap();
        let mut metadata = BTreeMap::from([
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
        ]);
        let mut manifest = ArtifactManifest {
            schema_version: ARTIFACT_MANIFEST_SCHEMA_V1,
            bundle: layout.bundle.clone(),
            profile: layout.profile.clone(),
            model: layout.model.clone(),
            model_revision: layout.model_revision.clone(),
            numeric_format: numeric_format(profile),
            components: Vec::new(),
            files: vec![
                ArtifactFile {
                    path: logical_path,
                    size: payload.len() as u64,
                    sha256: digest,
                    role: ArtifactFileRole::Weights,
                    component: None,
                    shard: None,
                },
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
            metadata: std::mem::take(&mut metadata),
            content_digest: None,
        };
        manifest.seal().unwrap();
        layout.validate_for_manifest(&manifest).unwrap();
        fs::write(
            root.join("manifest.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        manifest
    }

    #[test]
    fn explicit_source_accepts_standalone_bundle_identity_correctness() {
        let temp = tempfile::tempdir().unwrap();
        let variant = BooguVariant::Image01Turbo;
        let profile = BooguStorageProfile::F16QwenVisionF32;
        write_tiny_remote_with_bundle(
            temp.path(),
            variant,
            profile,
            boogu_source_bundle_id(variant, profile),
        );
        let bytes = fs::read(temp.path().join("manifest.json")).unwrap();

        let manifest = parse_expected_manifest(&bytes, variant, profile, false).unwrap();
        assert_eq!(
            manifest.bundle.as_str(),
            "boogu-image-0.1-turbo-f16-qwen-vision-f32"
        );
        let error = parse_expected_manifest(&bytes, variant, profile, true)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("model manifest identity mismatch"),
            "{error}"
        );
        assert!(
            error.contains(
                "expected bundle=boogu-image-0.1-turbo (an explicit local conversion source may use"
            ),
            "{error}"
        );
    }

    #[test]
    fn canonical_cdn_and_cache_names_are_bundle_specific_correctness() {
        assert_eq!(
            boogu_bundle_id(
                BooguVariant::Image01Turbo,
                BooguStorageProfile::F16QwenVisionF32
            ),
            "boogu-image-0.1-turbo"
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
            "https://aberration.technology/model/boogu-image-0.1-edit-turbo-1k5"
        );
        assert_eq!(
            canonical_native_boogu_model_base_url(
                BooguVariant::Image01Turbo,
                BooguStorageProfile::Q4sBlockUpTo128F32,
            )
            .unwrap()
            .as_str(),
            "https://aberration.technology/model/boogu-image-0.1-turbo-q4s-block-up-to128-f32"
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
    fn canonical_and_custom_remote_default_cache_directories_do_not_alias_correctness() {
        let broad_root = Path::new("/test-user-cache");
        let variant = BooguVariant::Image01Turbo;
        let profile = BooguStorageProfile::F16QwenVisionF32;

        let canonical = native_boogu_model_cache_root_under(broad_root, variant, profile, true);
        let custom = native_boogu_model_cache_root_under(broad_root, variant, profile, false);

        assert_ne!(canonical, custom);
        assert_eq!(
            canonical,
            broad_root
                .join(DEFAULT_BURN_IMAGE_MODEL_CACHE_SUBDIR)
                .join("boogu-image-0.1-turbo")
        );
        assert_eq!(
            custom,
            broad_root
                .join(DEFAULT_BURN_IMAGE_MODEL_CACHE_SUBDIR)
                .join("boogu-image-0.1-turbo-f16-qwen-vision-f32")
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
            if message.starts_with("downloading boogu-image") {
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

    #[test]
    fn local_schema_v2_composition_resolves_exact_siblings_correctness() {
        let temp = tempfile::tempdir().unwrap();
        let parent = temp.path().join("models");
        let variant = BooguVariant::Image01Turbo;
        let profile = BooguStorageProfile::F16QwenVisionF32;
        let pipeline_root = parent.join(boogu_bundle_id(variant, profile));
        fs::create_dir_all(&pipeline_root).unwrap();
        write_tiny_remote(&pipeline_root, variant, profile);
        let qwen =
            write_tiny_dependency(&parent.join("shared-qwen"), "shared-qwen", b"qwen payload");
        let vae = write_tiny_dependency(&parent.join("shared-vae"), "shared-vae", b"vae payload");
        let mut pipeline: ArtifactManifest =
            serde_json::from_slice(&fs::read(pipeline_root.join("manifest.json")).unwrap())
                .unwrap();
        pipeline.schema_version = ARTIFACT_MANIFEST_SCHEMA_V2;
        pipeline.dependencies = vec![tiny_dependency("qwen", &qwen), tiny_dependency("vae", &vae)];
        pipeline
            .metadata
            .insert("component_dependency_count".into(), "2".into());
        pipeline.content_digest = None;
        pipeline.seal().unwrap();
        fs::write(
            pipeline_root.join("manifest.json"),
            serde_json::to_vec_pretty(&pipeline).unwrap(),
        )
        .unwrap();

        let resolved =
            resolve_local_composition(variant, profile, pipeline_root.clone(), &|_| {}).unwrap();
        assert_eq!(resolved.pipeline_root(), pipeline_root);
        assert_eq!(resolved.qwen_root(), parent.join("shared-qwen"));
        assert_eq!(resolved.vae_root(), parent.join("shared-vae"));
        assert!(!resolved.is_standalone());
    }

    #[test]
    fn local_composition_fails_closed_for_missing_or_tampered_dependency_correctness() {
        let temp = tempfile::tempdir().unwrap();
        let parent = temp.path().join("models");
        let variant = BooguVariant::Image01Turbo;
        let profile = BooguStorageProfile::F16QwenVisionF32;
        let pipeline_root = parent.join(boogu_bundle_id(variant, profile));
        fs::create_dir_all(&pipeline_root).unwrap();
        write_tiny_remote(&pipeline_root, variant, profile);
        let qwen =
            write_tiny_dependency(&parent.join("shared-qwen"), "shared-qwen", b"qwen payload");
        let vae = write_tiny_dependency(&parent.join("shared-vae"), "shared-vae", b"vae payload");
        let mut pipeline: ArtifactManifest =
            serde_json::from_slice(&fs::read(pipeline_root.join("manifest.json")).unwrap())
                .unwrap();
        pipeline.schema_version = ARTIFACT_MANIFEST_SCHEMA_V2;
        pipeline.dependencies = vec![tiny_dependency("qwen", &qwen), tiny_dependency("vae", &vae)];
        pipeline
            .metadata
            .insert("component_dependency_count".into(), "2".into());
        pipeline.content_digest = None;
        pipeline.seal().unwrap();
        fs::write(
            pipeline_root.join("manifest.json"),
            serde_json::to_vec_pretty(&pipeline).unwrap(),
        )
        .unwrap();

        fs::remove_file(parent.join("shared-vae/manifest.json")).unwrap();
        let missing = resolve_local_composition(variant, profile, pipeline_root.clone(), &|_| {})
            .unwrap_err()
            .to_string();
        assert!(missing.contains("shared-vae/manifest.json"), "{missing}");

        fs::write(
            parent.join("shared-vae/manifest.json"),
            serde_json::to_vec_pretty(&vae).unwrap(),
        )
        .unwrap();
        let mut tampered = vae;
        tampered.model_revision = "wrong-revision".into();
        tampered.content_digest = None;
        tampered.seal().unwrap();
        fs::write(
            parent.join("shared-vae/manifest.json"),
            serde_json::to_vec_pretty(&tampered).unwrap(),
        )
        .unwrap();
        let error = resolve_local_composition(variant, profile, pipeline_root, &|_| {})
            .unwrap_err()
            .to_string();
        assert!(error.contains("dependency"), "{error}");
    }

    #[test]
    fn local_standalone_schema_v1_uses_one_root_correctness() {
        let temp = tempfile::tempdir().unwrap();
        let variant = BooguVariant::Image01Turbo;
        let profile = BooguStorageProfile::F16QwenVisionF32;
        write_tiny_remote(temp.path(), variant, profile);
        let resolved =
            resolve_local_composition(variant, profile, temp.path().to_owned(), &|_| {}).unwrap();
        assert!(resolved.is_standalone());
        assert_eq!(resolved.pipeline_root(), temp.path());
        assert_eq!(resolved.qwen_root(), temp.path());
        assert_eq!(resolved.vae_root(), temp.path());
    }

    #[test]
    fn local_schema_v2_without_dependencies_fails_closed_correctness() {
        let temp = tempfile::tempdir().unwrap();
        let variant = BooguVariant::Image01Turbo;
        let profile = BooguStorageProfile::F16QwenVisionF32;
        write_tiny_remote(temp.path(), variant, profile);
        let path = temp.path().join("manifest.json");
        let mut manifest: ArtifactManifest =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        manifest.schema_version = ARTIFACT_MANIFEST_SCHEMA_V2;
        manifest.content_digest = None;
        manifest.seal().unwrap();
        fs::write(&path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();

        let error = resolve_local_composition(variant, profile, temp.path().to_owned(), &|_| {})
            .unwrap_err()
            .to_string();
        assert!(error.contains("omits qwen and vae dependencies"), "{error}");
    }

    #[test]
    fn remote_component_cache_is_bundle_and_digest_isolated_correctness() {
        let temp = tempfile::tempdir().unwrap();
        let remote = temp.path().join("remote");
        let variant = BooguVariant::Image01Turbo;
        let profile = BooguStorageProfile::F16QwenVisionF32;
        let pipeline_bundle = boogu_bundle_id(variant, profile);
        let pipeline_remote = remote.join(&pipeline_bundle);
        fs::create_dir_all(&pipeline_remote).unwrap();
        write_tiny_remote(&pipeline_remote, variant, profile);
        let qwen =
            write_tiny_dependency(&remote.join("shared-qwen"), "shared-qwen", b"qwen payload");
        let vae = write_tiny_dependency(&remote.join("shared-vae"), "shared-vae", b"vae payload");
        let mut pipeline: ArtifactManifest =
            serde_json::from_slice(&fs::read(pipeline_remote.join("manifest.json")).unwrap())
                .unwrap();
        pipeline.schema_version = ARTIFACT_MANIFEST_SCHEMA_V2;
        pipeline.dependencies = vec![tiny_dependency("qwen", &qwen), tiny_dependency("vae", &vae)];
        pipeline
            .metadata
            .insert("component_dependency_count".into(), "2".into());
        pipeline.content_digest = None;
        pipeline.seal().unwrap();
        fs::write(
            pipeline_remote.join("manifest.json"),
            serde_json::to_vec_pretty(&pipeline).unwrap(),
        )
        .unwrap();

        let server = TestServer::serve(remote);
        let base = RemoteBaseUrl::new(format!("{}/{}", server.base_url, pipeline_bundle)).unwrap();
        let cache_parent = temp.path().join("cache/models");
        let pipeline_cache = cache_parent.join(&pipeline_bundle);
        let fetched =
            cache_remote_bundle(variant, profile, &base, &pipeline_cache, false, &|_| {}).unwrap();
        let resolved =
            resolve_remote_composition(&base, &pipeline_cache, &fetched, &|_| {}).unwrap();
        assert_eq!(resolved.pipeline_root(), pipeline_cache);
        assert_eq!(
            resolved.qwen_root(),
            cache_parent
                .join("shared-qwen")
                .join(qwen.content_digest.unwrap().to_string())
        );
        assert_eq!(
            resolved.vae_root(),
            cache_parent
                .join("shared-vae")
                .join(vae.content_digest.unwrap().to_string())
        );
        assert_ne!(resolved.qwen_root(), resolved.vae_root());
        assert_eq!(
            fs::read(resolved.qwen_root().join("objects/tiny.bpk")).unwrap(),
            b"qwen payload"
        );
        assert_eq!(
            fs::read(resolved.vae_root().join("objects/tiny.bpk")).unwrap(),
            b"vae payload"
        );
    }

    #[test]
    fn native_remote_pipeline_cache_commits_transport_parts_and_reopens_correctness() {
        let temp = tempfile::tempdir().unwrap();
        let remote = temp.path().join("remote");
        let variant = BooguVariant::Image01Turbo;
        let profile = BooguStorageProfile::F16QwenVisionF32;
        let bundle = boogu_bundle_id(variant, profile);
        let descriptor = boogu_model_descriptor(variant);
        let payload = b"part-only native pipeline payload";
        let remote_bundle = remote.join(&bundle);
        fs::create_dir_all(&remote_bundle).unwrap();
        write_tiny_transport_bundle(
            &remote_bundle,
            profile,
            &bundle,
            descriptor.id.as_str(),
            &descriptor.revision,
            payload,
        );

        let server = TestServer::serve(remote);
        let base = RemoteBaseUrl::new(format!("{}/{}", server.base_url, bundle)).unwrap();
        let cache = temp.path().join("cache").join(&bundle);
        let manifest =
            cache_remote_bundle(variant, profile, &base, &cache, false, &|_| {}).unwrap();
        assert!(
            ArtifactTransportLayout::declared_file(&manifest)
                .unwrap()
                .is_some()
        );
        let digest = Sha256Digest::calculate(payload);
        let part_path = format!("transport/{digest}.part");
        assert_eq!(fs::read(cache.join(&part_path)).unwrap(), payload);
        assert!(!cache.join("objects/tiny.bpk").exists());
        assert!(!remote_bundle.join("objects/tiny.bpk").exists());
        assert!(cache.join("metadata/transport-layout.json").is_file());
        server.requests.store(0, Ordering::SeqCst);
        cache_remote_bundle(variant, profile, &base, &cache, false, &|_| {}).unwrap();
        assert_eq!(server.requests.load(Ordering::SeqCst), 0);
        let directory = burn_image::VerifiedArtifactDirectory::open(&cache).unwrap();
        let logical = directory
            .manifest()
            .files
            .iter()
            .find(|file| file.role == ArtifactFileRole::Weights)
            .unwrap();
        let mut reader = directory.shard_reader().unwrap();
        assert_eq!(
            burn_image::ArtifactShardReader::read_shard(&mut reader, logical).unwrap(),
            payload
        );
    }

    #[test]
    fn native_remote_dependency_cache_commits_transport_parts_and_reopens_correctness() {
        let temp = tempfile::tempdir().unwrap();
        let remote = temp.path().join("remote");
        let payload = b"part-only native dependency payload";
        let bundle = "transport-dependency";
        let remote_bundle = remote.join(bundle);
        fs::create_dir_all(&remote_bundle).unwrap();
        let manifest = write_tiny_transport_bundle(
            &remote_bundle,
            BooguStorageProfile::F16,
            bundle,
            "test/transport-dependency",
            "exact-transport-revision",
            payload,
        );
        let dependency = tiny_dependency("qwen", &manifest);
        let server = TestServer::serve(remote);
        let pipeline_base = RemoteBaseUrl::new(format!("{}/pipeline", server.base_url)).unwrap();
        let cache =
            FilesystemArtifactCache::new(temp.path().join("cache"), 4 * 1024 * 1024).unwrap();
        let mut fetcher = UreqSiblingBundleFetcher::new(pipeline_base);
        let directory = cache.ensure_dependency(&dependency, &mut fetcher).unwrap();
        let digest = Sha256Digest::calculate(payload);
        let part_path = format!("transport/{digest}.part");
        assert_eq!(
            fs::read(directory.root().join(&part_path)).unwrap(),
            payload
        );
        assert!(!directory.root().join("objects/tiny.bpk").exists());
        assert!(!remote_bundle.join("objects/tiny.bpk").exists());
        let logical = directory
            .manifest()
            .files
            .iter()
            .find(|file| file.role == ArtifactFileRole::Weights)
            .unwrap();
        let mut reader = directory.shard_reader().unwrap();
        assert_eq!(
            burn_image::ArtifactShardReader::read_shard(&mut reader, logical).unwrap(),
            payload
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
