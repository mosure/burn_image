use std::{
    collections::BTreeMap,
    fs,
    io::Read,
    path::{Path, PathBuf},
};

use thiserror::Error;

use crate::{
    ArtifactFile, ArtifactFileRole, ArtifactManifest, ArtifactPath, ArtifactTransportLayout,
    ArtifactTransportLayoutError, ArtifactTransportObject, ArtifactVerifier, IntegrityPolicy,
    VerificationStatus, VerifiedArtifact, VerifiedArtifactTransportLayout,
};

/// Conservative default cap for a compact sealed artifact manifest read from disk.
///
/// Protocol- and application-specific caches should use [`VerifiedArtifactDirectory::open_bounded`]
/// with their stricter deployment limit.
pub const DEFAULT_MAX_ARTIFACT_MANIFEST_BYTES: u64 = 64 * 1024 * 1024;

/// Transport-neutral failure while reading a manifest-declared artifact.
#[derive(Debug, Error)]
pub enum ArtifactReadError {
    #[error("artifact transport failed: {0}")]
    Transport(String),
    #[error("artifact integrity check failed for {path}: {message}")]
    Integrity { path: ArtifactPath, message: String },
    #[error("artifact reader returned {actual} bytes for {path}, above the {maximum}-byte cap")]
    ResponseTooLarge {
        path: ArtifactPath,
        actual: u64,
        maximum: u64,
    },
    #[error("artifact verification evidence does not match sealed file {0}")]
    EvidenceMismatch(ArtifactPath),
    #[error("artifact directory is invalid: {0}")]
    Directory(String),
    #[error(transparent)]
    TransportLayout(#[from] ArtifactTransportLayoutError),
}

impl ArtifactReadError {
    pub fn transport(message: impl Into<String>) -> Self {
        Self::Transport(message.into())
    }
}

/// Supplies one manifest-declared logical object at a time, reconstructing physical parts when the
/// sealed transport layout requires them.
pub trait ArtifactShardReader {
    fn read_shard(&mut self, file: &ArtifactFile) -> Result<Vec<u8>, ArtifactReadError>;
}

/// Bytes from an asynchronous reader, optionally bound to SHA-256 evidence produced while the
/// transport streamed those exact bytes.
///
/// The fields are private so authenticated evidence cannot be paired with replacement bytes.
pub struct VerifiedArtifactBytes {
    bytes: Vec<u8>,
    verification: Option<VerifiedArtifact>,
}

/// Incrementally reconstruct one logical artifact while binding its final SHA-256 evidence.
///
/// Browser transports use this to hash each already-authenticated physical part while it is hot,
/// avoiding a second full pass over a 200--256 MiB logical Burnpack after its final part arrives.
/// The builder still owns only one logical object and exposes no way to forge verification proof.
pub struct VerifiedArtifactBytesBuilder {
    bytes: Vec<u8>,
    verifier: ArtifactVerifier,
    path: ArtifactPath,
}

impl VerifiedArtifactBytesBuilder {
    pub fn new(file: &ArtifactFile) -> Result<Self, ArtifactReadError> {
        let capacity = usize::try_from(file.size).map_err(|_| {
            ArtifactReadError::transport(format!(
                "artifact {} does not fit the process address space",
                file.path
            ))
        })?;
        Ok(Self {
            bytes: Vec::with_capacity(capacity),
            verifier: ArtifactVerifier::new(file, IntegrityPolicy::RequireSha256),
            path: file.path.clone(),
        })
    }

    pub fn extend_from_slice(&mut self, bytes: &[u8]) -> Result<(), ArtifactReadError> {
        self.verifier
            .update(bytes)
            .map_err(|error| ArtifactReadError::Integrity {
                path: self.path.clone(),
                message: error.to_string(),
            })?;
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub fn finish(self) -> Result<VerifiedArtifactBytes, (ArtifactReadError, Vec<u8>)> {
        let Self {
            bytes,
            verifier,
            path,
        } = self;
        let verification = match verifier.finish() {
            Ok(verification) => verification,
            Err(error) => {
                return Err((
                    ArtifactReadError::Integrity {
                        path,
                        message: error.to_string(),
                    },
                    bytes,
                ));
            }
        };
        Ok(VerifiedArtifactBytes {
            bytes,
            verification: Some(verification),
        })
    }
}

impl VerifiedArtifactBytes {
    /// Wrap bytes that still require SHA-256 verification by the model loader.
    pub fn unverified(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            verification: None,
        }
    }

    /// Verify exact size and SHA-256 once and bind that evidence to these bytes.
    pub fn verify_sha256(file: &ArtifactFile, bytes: Vec<u8>) -> Result<Self, ArtifactReadError> {
        Self::try_verify_sha256(file, bytes).map_err(|(error, _bytes)| error)
    }

    /// Verify exact size and SHA-256 once while returning ownership of a rejected payload.
    ///
    /// Cache-backed transports use the returned bytes only to report the observed digest and then
    /// evict the failed entry. Successful reads retain private evidence exactly like
    /// [`Self::verify_sha256`], so callers cannot pair proof from one payload with another.
    pub fn try_verify_sha256(
        file: &ArtifactFile,
        bytes: Vec<u8>,
    ) -> Result<Self, (ArtifactReadError, Vec<u8>)> {
        let verification = match verify_bytes(file, &bytes) {
            Ok(verification) => verification,
            Err(error) => return Err((error, bytes)),
        };
        Ok(Self {
            bytes,
            verification: Some(verification),
        })
    }

    /// Consume an untrusted read without requiring evidence.
    ///
    /// Verified model loaders should use [`Self::into_verified_bytes`] instead.
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Enforce the response cap and sealed identity, hashing only when the reader did not already
    /// return matching typed SHA-256 evidence.
    pub fn into_verified_bytes(
        self,
        file: &ArtifactFile,
        maximum_bytes: u64,
    ) -> Result<Vec<u8>, ArtifactReadError> {
        let actual = u64::try_from(self.bytes.len()).unwrap_or(u64::MAX);
        if actual > maximum_bytes {
            return Err(ArtifactReadError::ResponseTooLarge {
                path: file.path.clone(),
                actual,
                maximum: maximum_bytes,
            });
        }
        match self.verification {
            Some(verification)
                if verification.path() == &file.path
                    && verification.size() == file.size
                    && verification.size() == actual
                    && verification.digest() == file.sha256
                    && verification.status() == VerificationStatus::Sha256Verified => {}
            Some(_) => return Err(ArtifactReadError::EvidenceMismatch(file.path.clone())),
            None => {
                verify_bytes(file, &self.bytes)?;
            }
        }
        Ok(self.bytes)
    }
}

/// Wasm-friendly bounded reader. Futures deliberately need not be `Send` because browser fetch
/// and WebGPU handles are event-loop local.
#[allow(async_fn_in_trait)]
pub trait AsyncArtifactShardReader {
    /// Read exactly one sealed file without exceeding `maximum_bytes`.
    async fn read_shard(
        &mut self,
        file: &ArtifactFile,
        maximum_bytes: u64,
    ) -> Result<Vec<u8>, ArtifactReadError>;

    /// Read one file with optional typed integrity evidence.
    async fn read_verified_shard(
        &mut self,
        file: &ArtifactFile,
        maximum_bytes: u64,
    ) -> Result<VerifiedArtifactBytes, ArtifactReadError> {
        self.read_shard(file, maximum_bytes)
            .await
            .map(VerifiedArtifactBytes::unverified)
    }
}

/// Native reader rooted at one sealed bundle directory.
#[derive(Clone, Debug)]
pub struct DirectoryArtifactShardReader {
    root: PathBuf,
    manifest_files: Option<BTreeMap<ArtifactPath, ArtifactFile>>,
    transport_layout: Option<VerifiedArtifactTransportLayout>,
}

impl DirectoryArtifactShardReader {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            manifest_files: None,
            transport_layout: None,
        }
    }

    /// Construct a reader bound to one sealed manifest and its optional sealed transport layout.
    ///
    /// Legacy manifests without transport declarations continue reading logical files directly.
    /// Once a layout is declared, every logical weight is reconstructed exclusively from its
    /// verified content-addressed parts; a missing direct logical file is therefore expected.
    pub fn from_manifest(
        root: impl Into<PathBuf>,
        manifest: &ArtifactManifest,
    ) -> Result<Self, ArtifactReadError> {
        let root = root.into();
        let mut reader = Self::new(&root);
        let transport_layout = match ArtifactTransportLayout::declared_file(manifest)? {
            Some(layout_file) => {
                let bytes = reader.read_direct_shard(layout_file)?;
                Some(ArtifactTransportLayout::parse_and_validate(
                    manifest, &bytes,
                )?)
            }
            None => None,
        };
        reader.manifest_files = Some(
            manifest
                .files
                .iter()
                .cloned()
                .map(|file| (file.path.clone(), file))
                .collect(),
        );
        reader.transport_layout = transport_layout;
        Ok(reader)
    }

    pub fn from_verified_directory(
        directory: &VerifiedArtifactDirectory,
    ) -> Result<Self, ArtifactReadError> {
        Self::from_manifest(directory.root(), directory.manifest())
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn transport_layout(&self) -> Option<&VerifiedArtifactTransportLayout> {
        self.transport_layout.as_ref()
    }

    fn read_direct_shard(&self, file: &ArtifactFile) -> Result<Vec<u8>, ArtifactReadError> {
        let path = self.root.join(file.path.as_str());
        reject_symlink_chain(&path)?;
        let metadata = fs::symlink_metadata(&path).map_err(|error| {
            ArtifactReadError::Directory(format!("inspect {}: {error}", path.display()))
        })?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(ArtifactReadError::Directory(format!(
                "artifact is not a regular non-symlink file: {}",
                path.display()
            )));
        }
        if metadata.len() != file.size {
            return Err(ArtifactReadError::Integrity {
                path: file.path.clone(),
                message: format!(
                    "declared {} bytes but local file has {} bytes",
                    file.size,
                    metadata.len()
                ),
            });
        }
        read_bounded_file(&path, file.size, &file.path)
    }

    fn read_transport_object(
        &self,
        file: &ArtifactFile,
        object: &ArtifactTransportObject,
    ) -> Result<Vec<u8>, ArtifactReadError> {
        if file.role != ArtifactFileRole::Weights
            || file.path != object.path
            || file.size != object.size
            || file.sha256 != object.sha256
        {
            return Err(ArtifactReadError::EvidenceMismatch(file.path.clone()));
        }
        let mut logical_verifier = ArtifactVerifier::new(file, IntegrityPolicy::RequireSha256);
        let initial_capacity = usize::try_from(file.size)
            .unwrap_or(usize::MAX)
            .min(1024 * 1024);
        let mut output = Vec::with_capacity(initial_capacity);
        let mut buffer = [0_u8; 1024 * 1024];

        for part in &object.parts {
            let part_path = self.root.join(part.path.as_str());
            reject_symlink_chain(&part_path)?;
            let metadata = fs::symlink_metadata(&part_path).map_err(|error| {
                ArtifactReadError::Directory(format!("inspect {}: {error}", part_path.display()))
            })?;
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                return Err(ArtifactReadError::Directory(format!(
                    "artifact transport part is not a regular non-symlink file: {}",
                    part_path.display()
                )));
            }
            if metadata.len() != part.size {
                return Err(ArtifactReadError::Integrity {
                    path: part.path.clone(),
                    message: format!(
                        "declared {} bytes but local transport part has {} bytes",
                        part.size,
                        metadata.len()
                    ),
                });
            }
            let physical_file = ArtifactFile {
                path: part.path.clone(),
                size: part.size,
                sha256: part.sha256,
                role: ArtifactFileRole::Other,
                component: None,
                shard: None,
            };
            let mut part_verifier =
                ArtifactVerifier::new(&physical_file, IntegrityPolicy::RequireSha256);
            let mut input = fs::File::open(&part_path).map_err(|error| {
                ArtifactReadError::transport(format!("open {}: {error}", part_path.display()))
            })?;
            loop {
                let read = input.read(&mut buffer).map_err(|error| {
                    ArtifactReadError::transport(format!(
                        "read transport part {}: {error}",
                        part_path.display()
                    ))
                })?;
                if read == 0 {
                    break;
                }
                part_verifier.update(&buffer[..read]).map_err(|error| {
                    ArtifactReadError::Integrity {
                        path: part.path.clone(),
                        message: error.to_string(),
                    }
                })?;
                logical_verifier.update(&buffer[..read]).map_err(|error| {
                    ArtifactReadError::Integrity {
                        path: file.path.clone(),
                        message: error.to_string(),
                    }
                })?;
                output.extend_from_slice(&buffer[..read]);
            }
            part_verifier
                .finish()
                .map_err(|error| ArtifactReadError::Integrity {
                    path: part.path.clone(),
                    message: error.to_string(),
                })?;
        }
        logical_verifier
            .finish()
            .map_err(|error| ArtifactReadError::Integrity {
                path: file.path.clone(),
                message: error.to_string(),
            })?;
        Ok(output)
    }
}

impl ArtifactShardReader for DirectoryArtifactShardReader {
    fn read_shard(&mut self, file: &ArtifactFile) -> Result<Vec<u8>, ArtifactReadError> {
        if let Some(manifest_files) = &self.manifest_files {
            match manifest_files.get(&file.path) {
                Some(expected) if expected == file => {}
                _ => return Err(ArtifactReadError::EvidenceMismatch(file.path.clone())),
            }
        }
        if let Some(layout) = &self.transport_layout
            && file.role == ArtifactFileRole::Weights
        {
            let object = layout.object(&file.path).ok_or_else(|| {
                ArtifactReadError::Directory(format!(
                    "verified transport layout omits logical weight {}",
                    file.path
                ))
            })?;
            return self.read_transport_object(file, object);
        }
        self.read_direct_shard(file)
    }
}

fn reject_symlink_chain(path: &Path) -> Result<(), ArtifactReadError> {
    let mut current = PathBuf::new();
    for component in path.components() {
        current.push(component);
        if current.exists()
            && fs::symlink_metadata(&current)
                .is_ok_and(|metadata| metadata.file_type().is_symlink())
        {
            return Err(ArtifactReadError::Directory(format!(
                "artifact path traverses a symlink: {}",
                current.display()
            )));
        }
    }
    Ok(())
}

/// A sealed native bundle directory with exact per-file verification helpers.
#[derive(Clone, Debug)]
pub struct VerifiedArtifactDirectory {
    root: PathBuf,
    manifest: ArtifactManifest,
}

impl VerifiedArtifactDirectory {
    /// Open a sealed directory with a conservative model-neutral manifest cap.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, ArtifactReadError> {
        Self::open_bounded(root, DEFAULT_MAX_ARTIFACT_MANIFEST_BYTES)
    }

    /// Open a sealed directory after rejecting an oversized manifest before allocation.
    pub fn open_bounded(
        root: impl Into<PathBuf>,
        maximum_manifest_bytes: u64,
    ) -> Result<Self, ArtifactReadError> {
        if maximum_manifest_bytes == 0 {
            return Err(ArtifactReadError::Directory(
                "maximum manifest bytes must be positive".into(),
            ));
        }
        let root = root.into();
        let manifest_path = root.join("manifest.json");
        let metadata = fs::symlink_metadata(&manifest_path).map_err(|error| {
            ArtifactReadError::Directory(format!("inspect {}: {error}", manifest_path.display()))
        })?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(ArtifactReadError::Directory(format!(
                "manifest is not a regular non-symlink file: {}",
                manifest_path.display()
            )));
        }
        if metadata.len() == 0 || metadata.len() > maximum_manifest_bytes {
            return Err(ArtifactReadError::Directory(format!(
                "manifest {} is {} bytes, outside 1..={maximum_manifest_bytes}",
                manifest_path.display(),
                metadata.len()
            )));
        }
        let manifest_artifact_path =
            ArtifactPath::new("manifest.json").expect("canonical manifest path is valid");
        let bytes = read_bounded_file(
            &manifest_path,
            maximum_manifest_bytes,
            &manifest_artifact_path,
        )?;
        let manifest: ArtifactManifest = serde_json::from_slice(&bytes).map_err(|error| {
            ArtifactReadError::Directory(format!("parse {}: {error}", manifest_path.display()))
        })?;
        manifest.validate_sealed().map_err(|error| {
            ArtifactReadError::Directory(format!("validate {}: {error}", manifest_path.display()))
        })?;
        Ok(Self { root, manifest })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn manifest(&self) -> &ArtifactManifest {
        &self.manifest
    }

    /// Construct a reader bound to this directory's sealed manifest and optional transport
    /// layout.
    pub fn shard_reader(&self) -> Result<DirectoryArtifactShardReader, ArtifactReadError> {
        DirectoryArtifactShardReader::from_verified_directory(self)
    }

    pub fn read_file(&self, path: &str) -> Result<Vec<u8>, ArtifactReadError> {
        let file = self
            .manifest
            .files
            .iter()
            .find(|file| file.path.as_str() == path)
            .ok_or_else(|| ArtifactReadError::Directory(format!("manifest omits {path}")))?;
        let mut reader = self.shard_reader()?;
        let bytes = reader.read_shard(file)?;
        verify_bytes(file, &bytes)?;
        Ok(bytes)
    }

    pub fn read_text(&self, path: &str) -> Result<String, ArtifactReadError> {
        String::from_utf8(self.read_file(path)?).map_err(|error| {
            ArtifactReadError::Directory(format!("manifest file {path} is not UTF-8: {error}"))
        })
    }
}

fn read_bounded_file(
    path: &Path,
    maximum_bytes: u64,
    artifact_path: &ArtifactPath,
) -> Result<Vec<u8>, ArtifactReadError> {
    let capacity = usize::try_from(maximum_bytes).map_err(|_| {
        ArtifactReadError::Directory(format!(
            "declared size {maximum_bytes} cannot fit this platform for {}",
            path.display()
        ))
    })?;
    let input = fs::File::open(path).map_err(|error| {
        ArtifactReadError::transport(format!("open {}: {error}", path.display()))
    })?;
    let mut bytes = Vec::with_capacity(capacity.min(1024 * 1024));
    input
        .take(maximum_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|error| {
            ArtifactReadError::transport(format!("read {}: {error}", path.display()))
        })?;
    let actual = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    if actual > maximum_bytes {
        return Err(ArtifactReadError::ResponseTooLarge {
            path: artifact_path.clone(),
            actual,
            maximum: maximum_bytes,
        });
    }
    Ok(bytes)
}

#[cfg(test)]
std::thread_local! {
    static VERIFICATION_PASSES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn verify_bytes(file: &ArtifactFile, bytes: &[u8]) -> Result<VerifiedArtifact, ArtifactReadError> {
    #[cfg(test)]
    VERIFICATION_PASSES.with(|passes| passes.set(passes.get() + 1));
    ArtifactVerifier::verify_bytes(file, bytes, IntegrityPolicy::RequireSha256).map_err(|error| {
        ArtifactReadError::Integrity {
            path: file.path.clone(),
            message: error.to_string(),
        }
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::{
        ARTIFACT_LEGACY_TARGET_MAX_SHARD_BYTES_KEY, ARTIFACT_MANIFEST_SCHEMA_V1,
        ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES, ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES_KEY,
        ARTIFACT_TARGET_MAX_TRANSPORT_SHARD_BYTES_KEY, ARTIFACT_TRANSPORT_LAYOUT_PATH,
        ARTIFACT_TRANSPORT_LAYOUT_PATH_KEY, ARTIFACT_TRANSPORT_LAYOUT_SCHEMA_KEY,
        ARTIFACT_TRANSPORT_LAYOUT_SCHEMA_VERSION, ARTIFACT_TRANSPORT_MAX_PART_BYTES,
        ARTIFACT_TRANSPORT_PART_TARGET_BYTES_KEY, ARTIFACT_TRANSPORT_PARTS_REQUIRED_KEY,
        ARTIFACT_TRANSPORT_TARGET_PART_BYTES, ArtifactBundleId, ArtifactFileRole,
        ArtifactProfileId, ArtifactTransportLayout, ArtifactTransportObject, ArtifactTransportPart,
        ModelId, NumericFormat, Sha256Digest,
    };

    use super::*;

    fn file(path: &str, bytes: &[u8]) -> ArtifactFile {
        ArtifactFile {
            path: ArtifactPath::new(path).unwrap(),
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
                ARTIFACT_LEGACY_TARGET_MAX_SHARD_BYTES_KEY.into(),
                ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES.to_string(),
            ),
        ])
    }

    #[test]
    fn incremental_verified_bytes_bind_exact_logical_digest_correctness() {
        let payload = b"two authenticated transport parts";
        let file = file("objects/incremental.bpk", payload);
        let mut builder = VerifiedArtifactBytesBuilder::new(&file).unwrap();
        builder.extend_from_slice(&payload[..11]).unwrap();
        builder.extend_from_slice(&payload[11..]).unwrap();
        assert_eq!(builder.len(), payload.len());
        let read = builder.finish().unwrap();
        assert_eq!(read.into_verified_bytes(&file, file.size).unwrap(), payload);

        let mut corrupt = VerifiedArtifactBytesBuilder::new(&file).unwrap();
        corrupt
            .extend_from_slice(b"two authenticated transport partz")
            .unwrap();
        let (error, rejected) = match corrupt.finish() {
            Ok(_) => panic!("corrupt incremental artifact unexpectedly verified"),
            Err(error) => error,
        };
        assert!(matches!(error, ArtifactReadError::Integrity { .. }));
        assert_eq!(rejected, b"two authenticated transport partz");
    }

    fn write_transport_bundle(
        root: &Path,
        physical_bytes: &[u8],
        logical_sha256: Sha256Digest,
    ) -> (ArtifactManifest, ArtifactFile, ArtifactPath) {
        let bundle = ArtifactBundleId::new("reader-transport-test").unwrap();
        let profile = ArtifactProfileId::new("f16").unwrap();
        let model = ModelId::new("example/reader-transport-test").unwrap();
        let revision = "0123456789abcdef".to_string();
        let part_sha256 = Sha256Digest::calculate(physical_bytes);
        let part_path = ArtifactPath::new(format!("transport/{part_sha256}.part")).unwrap();
        let logical = ArtifactFile {
            path: ArtifactPath::new("objects/logical.bpk").unwrap(),
            size: physical_bytes.len() as u64,
            sha256: logical_sha256,
            role: ArtifactFileRole::Weights,
            component: None,
            shard: None,
        };
        let layout = ArtifactTransportLayout {
            schema_version: ARTIFACT_TRANSPORT_LAYOUT_SCHEMA_VERSION,
            bundle: bundle.clone(),
            profile: profile.clone(),
            model: model.clone(),
            model_revision: revision.clone(),
            target_part_bytes: ARTIFACT_TRANSPORT_TARGET_PART_BYTES,
            hard_max_part_bytes: ARTIFACT_TRANSPORT_MAX_PART_BYTES,
            objects: vec![ArtifactTransportObject {
                path: logical.path.clone(),
                size: logical.size,
                sha256: logical.sha256,
                parts: vec![ArtifactTransportPart {
                    path: part_path.clone(),
                    offset: 0,
                    size: physical_bytes.len() as u64,
                    sha256: part_sha256,
                }],
            }],
        };
        let sidecar = serde_json::to_vec(&layout).unwrap();
        let sidecar_file = ArtifactFile {
            path: ArtifactPath::new(ARTIFACT_TRANSPORT_LAYOUT_PATH).unwrap(),
            size: sidecar.len() as u64,
            sha256: Sha256Digest::calculate(&sidecar),
            role: ArtifactFileRole::Metadata,
            component: None,
            shard: None,
        };
        let config_bytes = b"configuration";
        let config = ArtifactFile {
            path: ArtifactPath::new("metadata/config.json").unwrap(),
            size: config_bytes.len() as u64,
            sha256: Sha256Digest::calculate(config_bytes),
            role: ArtifactFileRole::Config,
            component: None,
            shard: None,
        };
        let mut manifest = ArtifactManifest {
            schema_version: ARTIFACT_MANIFEST_SCHEMA_V1,
            bundle,
            profile,
            model,
            model_revision: revision,
            numeric_format: NumericFormat::F16,
            components: Vec::new(),
            files: vec![logical.clone(), sidecar_file, config],
            dependencies: Vec::new(),
            metadata: transport_metadata(),
            content_digest: None,
        };
        manifest.seal().unwrap();

        fs::create_dir_all(root.join("metadata")).unwrap();
        fs::create_dir_all(root.join("transport")).unwrap();
        fs::write(root.join(ARTIFACT_TRANSPORT_LAYOUT_PATH), sidecar).unwrap();
        fs::write(root.join(part_path.as_str()), physical_bytes).unwrap();
        fs::write(root.join("metadata/config.json"), config_bytes).unwrap();
        fs::write(
            root.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        (manifest, logical, part_path)
    }

    #[test]
    fn typed_evidence_hashes_only_once_correctness() {
        let bytes = b"one bounded object";
        let file = file("objects/one.bpk", bytes);
        VERIFICATION_PASSES.with(|passes| passes.set(0));
        let read = VerifiedArtifactBytes::verify_sha256(&file, bytes.to_vec()).unwrap();
        let result = read.into_verified_bytes(&file, bytes.len() as u64).unwrap();
        assert_eq!(result, bytes);
        VERIFICATION_PASSES.with(|passes| assert_eq!(passes.get(), 1));
    }

    #[test]
    fn unverified_bytes_are_hashed_and_caps_fail_closed_correctness() {
        let bytes = b"generic bounded object";
        let file = file("objects/generic.bpk", bytes);
        VERIFICATION_PASSES.with(|passes| passes.set(0));
        let result = VerifiedArtifactBytes::unverified(bytes.to_vec())
            .into_verified_bytes(&file, bytes.len() as u64)
            .unwrap();
        assert_eq!(result, bytes);
        VERIFICATION_PASSES.with(|passes| assert_eq!(passes.get(), 1));
        let error = VerifiedArtifactBytes::unverified(bytes.to_vec())
            .into_verified_bytes(&file, (bytes.len() - 1) as u64)
            .unwrap_err();
        assert!(matches!(error, ArtifactReadError::ResponseTooLarge { .. }));
    }

    #[test]
    fn verified_directory_binds_manifest_and_payload_correctness() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join("objects")).unwrap();
        let bytes = b"sealed payload";
        fs::write(root.path().join("objects/payload.bpk"), bytes).unwrap();
        let mut manifest = ArtifactManifest {
            schema_version: ARTIFACT_MANIFEST_SCHEMA_V1,
            bundle: ArtifactBundleId::new("reader-test").unwrap(),
            profile: ArtifactProfileId::new("f16").unwrap(),
            model: ModelId::new("example/reader-test").unwrap(),
            model_revision: "0123456789abcdef".into(),
            numeric_format: NumericFormat::F16,
            components: Vec::new(),
            files: vec![file("objects/payload.bpk", bytes)],
            dependencies: Vec::new(),
            metadata: BTreeMap::new(),
            content_digest: None,
        };
        manifest.seal().unwrap();
        fs::write(
            root.path().join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();

        let directory = VerifiedArtifactDirectory::open(root.path()).unwrap();
        assert_eq!(directory.read_file("objects/payload.bpk").unwrap(), bytes);
    }

    #[test]
    fn verified_directory_rejects_oversized_manifest_before_read_correctness() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("manifest.json"), b"12345").unwrap();
        let error = VerifiedArtifactDirectory::open_bounded(root.path(), 4).unwrap_err();
        assert!(error.to_string().contains("outside 1..=4"));
    }

    #[test]
    fn directory_reader_rejects_wrong_size_before_allocation_correctness() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join("objects")).unwrap();
        fs::write(root.path().join("objects/payload.bpk"), b"oversized").unwrap();
        let declared = file("objects/payload.bpk", b"small");
        let mut reader = DirectoryArtifactShardReader::new(root.path());
        let error = reader.read_shard(&declared).unwrap_err();
        assert!(error.to_string().contains("declared 5 bytes"));
    }

    #[test]
    fn directory_reader_reconstructs_verified_logical_weight_from_parts_correctness() {
        let root = tempfile::tempdir().unwrap();
        let bytes = b"logical bytes exist only as a physical transport part";
        let (_, logical, _) =
            write_transport_bundle(root.path(), bytes, Sha256Digest::calculate(bytes));
        assert!(!root.path().join(logical.path.as_str()).exists());

        let directory = VerifiedArtifactDirectory::open(root.path()).unwrap();
        let reader = directory.shard_reader().unwrap();
        assert!(reader.transport_layout().is_some());
        assert_eq!(directory.read_file(logical.path.as_str()).unwrap(), bytes);
        assert_eq!(
            directory.read_text("metadata/config.json").unwrap(),
            "configuration"
        );
    }

    #[test]
    fn directory_reader_rejects_missing_or_tampered_transport_parts_correctness() {
        let root = tempfile::tempdir().unwrap();
        let bytes = b"authenticated physical part";
        let (_, logical, part_path) =
            write_transport_bundle(root.path(), bytes, Sha256Digest::calculate(bytes));
        let mut tampered = bytes.to_vec();
        tampered[0] ^= 1;
        fs::write(root.path().join(part_path.as_str()), tampered).unwrap();
        let directory = VerifiedArtifactDirectory::open(root.path()).unwrap();
        let error = directory.read_file(logical.path.as_str()).unwrap_err();
        assert!(matches!(error, ArtifactReadError::Integrity { .. }));

        fs::remove_file(root.path().join(part_path.as_str())).unwrap();
        let error = directory.read_file(logical.path.as_str()).unwrap_err();
        assert!(matches!(error, ArtifactReadError::Directory(_)));
    }

    #[test]
    fn directory_reader_rejects_valid_parts_with_bad_logical_digest_correctness() {
        let root = tempfile::tempdir().unwrap();
        let bytes = b"part digest is valid but logical identity is not";
        let wrong_logical_sha = Sha256Digest::calculate(b"different logical bytes");
        let (_, logical, _) = write_transport_bundle(root.path(), bytes, wrong_logical_sha);
        let directory = VerifiedArtifactDirectory::open(root.path()).unwrap();
        let error = directory.read_file(logical.path.as_str()).unwrap_err();
        assert!(matches!(error, ArtifactReadError::Integrity { .. }));
        assert!(error.to_string().contains(logical.path.as_str()));
    }

    #[test]
    fn directory_reader_rejects_tampered_sealed_sidecar_correctness() {
        let root = tempfile::tempdir().unwrap();
        let bytes = b"logical bytes";
        let (manifest, _, _) =
            write_transport_bundle(root.path(), bytes, Sha256Digest::calculate(bytes));
        let sidecar_path = root.path().join(ARTIFACT_TRANSPORT_LAYOUT_PATH);
        let mut sidecar = fs::read(&sidecar_path).unwrap();
        sidecar[0] ^= 1;
        fs::write(sidecar_path, sidecar).unwrap();
        let error =
            DirectoryArtifactShardReader::from_manifest(root.path(), &manifest).unwrap_err();
        assert!(matches!(error, ArtifactReadError::TransportLayout(_)));
    }

    #[test]
    fn manifest_bound_directory_reader_rejects_foreign_file_identity_correctness() {
        let root = tempfile::tempdir().unwrap();
        let bytes = b"bound logical bytes";
        let (manifest, mut logical, _) =
            write_transport_bundle(root.path(), bytes, Sha256Digest::calculate(bytes));
        let mut reader =
            DirectoryArtifactShardReader::from_manifest(root.path(), &manifest).unwrap();
        logical.sha256 = Sha256Digest::calculate(b"foreign identity");
        assert!(matches!(
            reader.read_shard(&logical),
            Err(ArtifactReadError::EvidenceMismatch(_))
        ));
    }

    #[cfg(unix)]
    #[test]
    fn directory_reader_rejects_transport_symlink_chain_correctness() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let bytes = b"symlinked part bytes";
        let (manifest, logical, part_path) =
            write_transport_bundle(root.path(), bytes, Sha256Digest::calculate(bytes));
        let outside = root.path().join("outside.part");
        fs::write(&outside, bytes).unwrap();
        fs::remove_file(root.path().join(part_path.as_str())).unwrap();
        symlink(&outside, root.path().join(part_path.as_str())).unwrap();

        let mut reader =
            DirectoryArtifactShardReader::from_manifest(root.path(), &manifest).unwrap();
        let error = reader.read_shard(&logical).unwrap_err();
        assert!(matches!(error, ArtifactReadError::Directory(_)));
        assert!(error.to_string().contains("symlink"));
    }
}
