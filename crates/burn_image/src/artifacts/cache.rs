use std::{
    fs::{self, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use crate::{
    ArtifactBundleId, ArtifactDependency, ArtifactFile, ArtifactManifest, ArtifactReadError,
    ArtifactVerifier, IntegrityPolicy, VerifiedArtifactDirectory,
};

/// Platform adapter used by the reusable native filesystem cache.
///
/// HTTP clients, object-store SDKs, and test fixtures implement only transport. The cache owns
/// manifest identity, SHA-256 verification, atomic installation, and the manifest-last commit
/// rule, so model consumers do not need to reproduce those invariants.
pub trait ArtifactBundleFetcher {
    /// Fetch one bounded manifest for `bundle`.
    fn fetch_manifest(
        &mut self,
        bundle: &ArtifactBundleId,
        maximum_bytes: u64,
    ) -> Result<Vec<u8>, ArtifactReadError>;

    /// Stream one exact file into `destination`. The cache independently verifies size/SHA-256.
    fn fetch_file(
        &mut self,
        bundle: &ArtifactBundleId,
        file: &ArtifactFile,
        destination: &mut dyn Write,
    ) -> Result<(), ArtifactReadError>;
}

/// Generic verified native cache keyed by immutable bundle identity and sealed digest.
#[derive(Clone, Debug)]
pub struct FilesystemArtifactCache {
    root: PathBuf,
    maximum_manifest_bytes: u64,
}

impl FilesystemArtifactCache {
    pub fn new(
        root: impl Into<PathBuf>,
        maximum_manifest_bytes: u64,
    ) -> Result<Self, ArtifactReadError> {
        if maximum_manifest_bytes == 0 {
            return Err(ArtifactReadError::Directory(
                "maximum manifest bytes must be positive".into(),
            ));
        }
        Ok(Self {
            root: root.into(),
            maximum_manifest_bytes,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolve one exact dependency into the verified cache.
    ///
    /// The directory is `<root>/<bundle>/<content-digest>`. Payloads are installed atomically and
    /// `manifest.json` is written only after every declared file verifies, making it the cache
    /// commit point.
    pub fn ensure_dependency<F: ArtifactBundleFetcher>(
        &self,
        dependency: &ArtifactDependency,
        fetcher: &mut F,
    ) -> Result<VerifiedArtifactDirectory, ArtifactReadError> {
        dependency
            .validate()
            .map_err(|error| ArtifactReadError::Directory(error.to_string()))?;
        let bundle_root = self
            .root
            .join(dependency.bundle.as_str())
            .join(dependency.content_digest.to_string());
        reject_symlink_chain(&self.root)?;
        fs::create_dir_all(&bundle_root).map_err(directory_error("create cache directory"))?;
        reject_symlink_chain(&bundle_root)?;
        let manifest_path = bundle_root.join("manifest.json");

        if manifest_path.exists()
            && let Ok(directory) =
                VerifiedArtifactDirectory::open_bounded(&bundle_root, self.maximum_manifest_bytes)
            && dependency
                .validate_resolved_manifest(directory.manifest())
                .is_ok()
            && directory
                .manifest()
                .files
                .iter()
                .all(|file| verified_cached_file(&bundle_root.join(file.path.as_str()), file))
        {
            return Ok(directory);
        }

        let manifest_bytes =
            fetcher.fetch_manifest(&dependency.bundle, self.maximum_manifest_bytes)?;
        let manifest_size = u64::try_from(manifest_bytes.len()).unwrap_or(u64::MAX);
        if manifest_size == 0 || manifest_size > self.maximum_manifest_bytes {
            return Err(ArtifactReadError::ResponseTooLarge {
                path: crate::ArtifactPath::new("manifest.json")
                    .expect("canonical manifest path is valid"),
                actual: manifest_size,
                maximum: self.maximum_manifest_bytes,
            });
        }
        let manifest: ArtifactManifest =
            serde_json::from_slice(&manifest_bytes).map_err(|error| {
                ArtifactReadError::Directory(format!("parse fetched manifest: {error}"))
            })?;
        dependency
            .validate_resolved_manifest(&manifest)
            .map_err(|error| ArtifactReadError::Directory(error.to_string()))?;

        // Invalidate a previous commit point before repairing payloads. An interrupted repair is
        // therefore incomplete rather than apparently committed.
        if manifest_path.exists() {
            fs::remove_file(&manifest_path).map_err(directory_error("remove invalid manifest"))?;
        }
        for file in &manifest.files {
            let destination = bundle_root.join(file.path.as_str());
            if verified_cached_file(&destination, file) {
                continue;
            }
            install_verified_file(&dependency.bundle, file, &destination, fetcher)?;
        }
        install_bytes_atomically(&manifest_path, &manifest_bytes)?;
        let directory =
            VerifiedArtifactDirectory::open_bounded(&bundle_root, self.maximum_manifest_bytes)?;
        dependency
            .validate_resolved_manifest(directory.manifest())
            .map_err(|error| ArtifactReadError::Directory(error.to_string()))?;
        Ok(directory)
    }
}

fn verified_cached_file(path: &Path, file: &ArtifactFile) -> bool {
    if fs::symlink_metadata(path).is_ok_and(|metadata| {
        metadata.is_file() && !metadata.file_type().is_symlink() && metadata.len() == file.size
    }) {
        let Ok(mut input) = fs::File::open(path) else {
            return false;
        };
        let mut verifier = ArtifactVerifier::new(file, IntegrityPolicy::RequireSha256);
        let mut buffer = [0_u8; 1024 * 1024];
        loop {
            let Ok(read) = input.read(&mut buffer) else {
                return false;
            };
            if read == 0 {
                break;
            }
            if verifier.update(&buffer[..read]).is_err() {
                return false;
            }
        }
        verifier.finish().is_ok()
    } else {
        false
    }
}

fn install_verified_file<F: ArtifactBundleFetcher>(
    bundle: &ArtifactBundleId,
    file: &ArtifactFile,
    destination: &Path,
    fetcher: &mut F,
) -> Result<(), ArtifactReadError> {
    let parent = destination.parent().ok_or_else(|| {
        ArtifactReadError::Directory(format!(
            "cache destination has no parent: {}",
            destination.display()
        ))
    })?;
    fs::create_dir_all(parent).map_err(directory_error("create artifact parent"))?;
    reject_symlink_chain(parent)?;
    let temporary = temporary_path(destination);
    let output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(directory_error("create temporary artifact"))?;
    let mut writer = VerifyingWriter::new(output, file);
    let result = fetcher
        .fetch_file(bundle, file, &mut writer)
        .and_then(|()| writer.finish());
    if let Err(error) = result {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    if destination.exists() {
        fs::remove_file(destination).map_err(directory_error("remove invalid cached artifact"))?;
    }
    fs::rename(&temporary, destination).map_err(directory_error("install verified artifact"))?;
    Ok(())
}

struct VerifyingWriter {
    output: fs::File,
    verifier: ArtifactVerifier,
    path: crate::ArtifactPath,
    failed: Option<String>,
}

impl VerifyingWriter {
    fn new(output: fs::File, file: &ArtifactFile) -> Self {
        Self {
            output,
            verifier: ArtifactVerifier::new(file, IntegrityPolicy::RequireSha256),
            path: file.path.clone(),
            failed: None,
        }
    }

    fn finish(mut self) -> Result<(), ArtifactReadError> {
        if let Some(message) = self.failed.take() {
            return Err(ArtifactReadError::transport(message));
        }
        self.verifier
            .finish()
            .map_err(|error| ArtifactReadError::Integrity {
                path: self.path,
                message: error.to_string(),
            })?;
        self.output
            .sync_all()
            .map_err(directory_error("sync verified artifact"))
    }
}

impl Write for VerifyingWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let written = self.output.write(buffer)?;
        if let Err(error) = self.verifier.update(&buffer[..written]) {
            self.failed = Some(error.to_string());
            return Err(io::Error::other(error.to_string()));
        }
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.output.flush()
    }
}

fn install_bytes_atomically(path: &Path, bytes: &[u8]) -> Result<(), ArtifactReadError> {
    let temporary = temporary_path(path);
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(directory_error("create temporary manifest"))?;
    output
        .write_all(bytes)
        .map_err(directory_error("write manifest"))?;
    output
        .sync_all()
        .map_err(directory_error("sync manifest"))?;
    if path.exists() {
        fs::remove_file(path).map_err(directory_error("remove old manifest"))?;
    }
    fs::rename(&temporary, path).map_err(directory_error("commit manifest"))
}

fn temporary_path(path: &Path) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("artifact");
    path.with_file_name(format!(".{name}.{}.{}.part", std::process::id(), nonce))
}

fn reject_symlink(path: &Path) -> Result<(), ArtifactReadError> {
    let metadata = fs::symlink_metadata(path).map_err(directory_error("inspect cache path"))?;
    if metadata.file_type().is_symlink() {
        return Err(ArtifactReadError::Directory(format!(
            "cache path is a symlink: {}",
            path.display()
        )));
    }
    Ok(())
}

fn reject_symlink_chain(path: &Path) -> Result<(), ArtifactReadError> {
    let mut current = PathBuf::new();
    for component in path.components() {
        current.push(component);
        if current.exists() {
            reject_symlink(&current)?;
        }
    }
    Ok(())
}

fn directory_error(operation: &'static str) -> impl FnOnce(io::Error) -> ArtifactReadError {
    move |error| ArtifactReadError::Directory(format!("{operation}: {error}"))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::{
        ARTIFACT_MANIFEST_SCHEMA_V1, ArtifactBundleId, ArtifactComponentId, ArtifactFileRole,
        ArtifactPath, ArtifactProfileId, ModelId, NumericFormat, Sha256Digest,
    };

    use super::*;

    struct MemoryFetcher {
        manifest: Vec<u8>,
        payload: Vec<u8>,
        manifest_reads: usize,
        payload_reads: usize,
    }

    impl ArtifactBundleFetcher for MemoryFetcher {
        fn fetch_manifest(
            &mut self,
            _bundle: &ArtifactBundleId,
            _maximum_bytes: u64,
        ) -> Result<Vec<u8>, ArtifactReadError> {
            self.manifest_reads += 1;
            Ok(self.manifest.clone())
        }

        fn fetch_file(
            &mut self,
            _bundle: &ArtifactBundleId,
            _file: &ArtifactFile,
            destination: &mut dyn Write,
        ) -> Result<(), ArtifactReadError> {
            self.payload_reads += 1;
            destination
                .write_all(&self.payload)
                .map_err(|error| ArtifactReadError::transport(error.to_string()))
        }
    }

    fn fixture() -> (ArtifactDependency, MemoryFetcher) {
        let payload = b"cached component payload".to_vec();
        let bundle = ArtifactBundleId::new("component-cache-test").unwrap();
        let profile = ArtifactProfileId::new("f16").unwrap();
        let model = ModelId::new("tests/component-cache").unwrap();
        let mut manifest = ArtifactManifest {
            schema_version: ARTIFACT_MANIFEST_SCHEMA_V1,
            bundle: bundle.clone(),
            profile: profile.clone(),
            model: model.clone(),
            model_revision: "immutable-component-revision".into(),
            numeric_format: NumericFormat::F16,
            components: Vec::new(),
            files: vec![ArtifactFile {
                path: ArtifactPath::new("objects/component.bpk").unwrap(),
                size: payload.len() as u64,
                sha256: Sha256Digest::calculate(&payload),
                role: ArtifactFileRole::Weights,
                component: None,
                shard: None,
            }],
            dependencies: Vec::new(),
            metadata: BTreeMap::new(),
            content_digest: None,
        };
        let digest = manifest.seal().unwrap();
        let dependency = ArtifactDependency {
            role: ArtifactComponentId::new("component").unwrap(),
            bundle,
            profile,
            model,
            model_revision: manifest.model_revision.clone(),
            content_digest: digest,
        };
        let fetcher = MemoryFetcher {
            manifest: serde_json::to_vec(&manifest).unwrap(),
            payload,
            manifest_reads: 0,
            payload_reads: 0,
        };
        (dependency, fetcher)
    }

    #[test]
    fn dependency_cache_commits_last_and_reuses_verified_payload_correctness() {
        let root = tempfile::tempdir().unwrap();
        let cache = FilesystemArtifactCache::new(root.path(), 1024 * 1024).unwrap();
        let (dependency, mut fetcher) = fixture();
        let first = cache.ensure_dependency(&dependency, &mut fetcher).unwrap();
        assert!(first.root().join("manifest.json").is_file());
        assert_eq!(fetcher.manifest_reads, 1);
        assert_eq!(fetcher.payload_reads, 1);

        let second = cache.ensure_dependency(&dependency, &mut fetcher).unwrap();
        assert_eq!(first.root(), second.root());
        assert_eq!(fetcher.manifest_reads, 1);
        assert_eq!(fetcher.payload_reads, 1);
    }

    #[test]
    fn dependency_cache_rejects_corrupt_payload_without_commit_correctness() {
        let root = tempfile::tempdir().unwrap();
        let cache = FilesystemArtifactCache::new(root.path(), 1024 * 1024).unwrap();
        let (dependency, mut fetcher) = fixture();
        fetcher.payload = b"corrupt".to_vec();
        assert!(cache.ensure_dependency(&dependency, &mut fetcher).is_err());
        let committed = root
            .path()
            .join(dependency.bundle.as_str())
            .join(dependency.content_digest.to_string())
            .join("manifest.json");
        assert!(!committed.exists());
    }

    #[cfg(unix)]
    #[test]
    fn dependency_cache_rejects_symlink_payload_on_cache_hit_correctness() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let cache = FilesystemArtifactCache::new(root.path(), 1024 * 1024).unwrap();
        let (dependency, mut fetcher) = fixture();
        let first = cache.ensure_dependency(&dependency, &mut fetcher).unwrap();
        let payload_path = first.root().join("objects/component.bpk");
        let replacement = root.path().join("replacement.bpk");
        fs::write(&replacement, &fetcher.payload).unwrap();
        fs::remove_file(&payload_path).unwrap();
        symlink(&replacement, &payload_path).unwrap();

        let second = cache.ensure_dependency(&dependency, &mut fetcher).unwrap();
        assert_eq!(first.root(), second.root());
        assert_eq!(fetcher.payload_reads, 2);
        assert!(
            !fs::symlink_metadata(&payload_path)
                .unwrap()
                .file_type()
                .is_symlink()
        );
    }

    #[cfg(unix)]
    #[test]
    fn dependency_cache_rejects_symlinked_bundle_ancestor_correctness() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let redirected = tempfile::tempdir().unwrap();
        let cache = FilesystemArtifactCache::new(root.path(), 1024 * 1024).unwrap();
        let (dependency, mut fetcher) = fixture();
        symlink(
            redirected.path(),
            root.path().join(dependency.bundle.as_str()),
        )
        .unwrap();

        let error = cache
            .ensure_dependency(&dependency, &mut fetcher)
            .unwrap_err();
        assert!(error.to_string().contains("symlink"));
        assert!(!redirected.path().join("manifest.json").exists());
    }
}
