use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
};

use thiserror::Error;

use crate::{
    ArtifactFile, ArtifactManifest, ArtifactPath, ArtifactVerifier, IntegrityPolicy,
    VerificationStatus, VerifiedArtifact,
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
}

impl ArtifactReadError {
    pub fn transport(message: impl Into<String>) -> Self {
        Self::Transport(message.into())
    }
}

/// Supplies one manifest-declared physical object at a time.
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
        let verification = verify_bytes(file, &bytes)?;
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
}

impl DirectoryArtifactShardReader {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}

impl ArtifactShardReader for DirectoryArtifactShardReader {
    fn read_shard(&mut self, file: &ArtifactFile) -> Result<Vec<u8>, ArtifactReadError> {
        let path = self.root.join(file.path.as_str());
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

    pub fn read_file(&self, path: &str) -> Result<Vec<u8>, ArtifactReadError> {
        let file = self
            .manifest
            .files
            .iter()
            .find(|file| file.path.as_str() == path)
            .ok_or_else(|| ArtifactReadError::Directory(format!("manifest omits {path}")))?;
        let mut reader = DirectoryArtifactShardReader::new(&self.root);
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
        ARTIFACT_MANIFEST_SCHEMA_V1, ArtifactBundleId, ArtifactFileRole, ArtifactProfileId,
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
}
