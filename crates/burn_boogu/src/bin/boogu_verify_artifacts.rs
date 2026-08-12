//! Verify every byte in a sealed Boogu deployment directory before publication.

use std::{
    error::Error,
    path::{Path, PathBuf},
};

use burn_boogu::artifacts::{
    VerifiedArtifactDirectory, validate_edit_turbo_1k5_release_artifact_digest,
    verify_published_release_artifact_directory, verify_release_artifact_directory,
};
use clap::Parser;
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(
    about = "Verify every byte and semantic tensor contract in a sealed Boogu release bundle"
)]
struct Args {
    /// Directory containing the sealed manifest and its authenticated payloads.
    #[arg(long)]
    artifacts: PathBuf,
    /// Require the exact mixed-F16 artifact digest qualified for Edit-Turbo 1.5K.
    #[arg(long, default_value_t = false)]
    require_edit_turbo_1k5_release: bool,
    /// Require the exact pinned digest for one of the five canonical published bundles.
    #[arg(long, default_value_t = false)]
    require_published_release: bool,
}

#[derive(Debug, Serialize)]
struct VerificationReport {
    root: PathBuf,
    bundle: String,
    profile: String,
    model: String,
    model_revision: String,
    content_digest: String,
    verified_files: usize,
    verified_bytes: u64,
    verified_weight_objects: usize,
    verified_tensors: usize,
    largest_object_bytes: u64,
    max_shard_bytes: u64,
    published_release_verified: bool,
    semantic_contract_verified: bool,
    artifacts_verified: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let report = verify_artifact_directory(
        &args.artifacts,
        args.require_edit_turbo_1k5_release,
        args.require_published_release,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn verify_artifact_directory(
    root: &Path,
    require_edit_turbo_1k5_release: bool,
    require_published_release: bool,
) -> Result<VerificationReport, Box<dyn Error>> {
    let semantic = if require_published_release {
        verify_published_release_artifact_directory(root)?
    } else {
        verify_release_artifact_directory(root)?
    };
    let directory = VerifiedArtifactDirectory::open(root)?;
    let manifest = directory.manifest();
    let content_digest = manifest
        .content_digest
        .expect("a verified sealed manifest has a digest");
    if require_edit_turbo_1k5_release {
        validate_edit_turbo_1k5_release_artifact_digest(content_digest)?;
    }
    Ok(VerificationReport {
        root: root.to_path_buf(),
        bundle: manifest.bundle.to_string(),
        profile: manifest.profile.to_string(),
        model: manifest.model.to_string(),
        model_revision: manifest.model_revision.clone(),
        content_digest: content_digest.to_string(),
        verified_files: semantic.verified_files,
        verified_bytes: semantic.verified_bytes,
        verified_weight_objects: semantic.verified_weight_objects,
        verified_tensors: semantic.verified_tensors,
        largest_object_bytes: semantic.largest_object_bytes,
        max_shard_bytes: semantic.max_shard_bytes,
        published_release_verified: require_published_release,
        semantic_contract_verified: true,
        artifacts_verified: true,
    })
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, fs};

    use burn_boogu::artifacts::BooguArtifactLoadError;
    use burn_image::{
        ARTIFACT_MANIFEST_SCHEMA_VERSION, ArtifactBundleId, ArtifactFile, ArtifactFileRole,
        ArtifactManifest, ArtifactPath, ArtifactProfileId, ModelId, NumericFormat, Sha256Digest,
    };

    use super::{VerifiedArtifactDirectory, verify_artifact_directory};

    const PAYLOAD: &[u8] = b"bounded artifact verification fixture";

    fn tiny_sealed_directory() -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        let relative = ArtifactPath::new("objects/tiny.bin").unwrap();
        let path = directory.path().join(relative.as_str());
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, PAYLOAD).unwrap();

        let mut manifest = ArtifactManifest {
            schema_version: ARTIFACT_MANIFEST_SCHEMA_VERSION,
            bundle: ArtifactBundleId::new("tiny-bundle").unwrap(),
            profile: ArtifactProfileId::new("tiny-profile").unwrap(),
            model: ModelId::new("tests/tiny-model").unwrap(),
            model_revision: "immutable-test-revision".into(),
            numeric_format: NumericFormat::F16,
            components: Vec::new(),
            files: vec![ArtifactFile {
                path: relative,
                size: PAYLOAD.len() as u64,
                sha256: Sha256Digest::calculate(PAYLOAD),
                role: ArtifactFileRole::Other,
                component: None,
                shard: None,
            }],
            metadata: BTreeMap::new(),
            content_digest: None,
        };
        manifest.seal().unwrap();
        fs::write(
            directory.path().join("manifest.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        directory
    }

    #[test]
    fn generic_bundle_is_not_misreported_as_boogu_release_correctness() {
        let directory = tiny_sealed_directory();
        let error = verify_artifact_directory(directory.path(), false, false).unwrap_err();
        assert!(matches!(
            error.downcast_ref::<BooguArtifactLoadError>(),
            Some(BooguArtifactLoadError::Identity(message))
                if message.contains("unsupported Boogu release model")
        ));
    }

    #[test]
    fn modified_payload_is_rejected_by_sealed_reader_correctness() {
        let directory = tiny_sealed_directory();
        let mut tampered = PAYLOAD.to_vec();
        *tampered.last_mut().unwrap() ^= 1;
        fs::write(directory.path().join("objects/tiny.bin"), tampered).unwrap();
        let sealed = VerifiedArtifactDirectory::open(directory.path()).unwrap();
        let error = sealed.read_file("objects/tiny.bin").unwrap_err();
        assert!(error.to_string().contains("integrity verification failed"));
    }
}
