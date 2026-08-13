//! Verify every byte in a sealed Boogu deployment directory before publication.

use std::{
    error::Error,
    path::{Path, PathBuf},
};

use burn_boogu::artifacts::{
    VerifiedArtifactDirectory, validate_canonical_release_artifact_digest,
    validate_edit_turbo_1k5_release_artifact_digest, verify_modular_release_artifact_directories,
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
    /// Require the exact pinned digest for one of the three canonical production bundles.
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
    dependency_bundles: Vec<String>,
    dependency_closure_verified: bool,
    component_contracts_verified: bool,
    reconstructed_inventory_verified: bool,
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
    let directory = VerifiedArtifactDirectory::open(root)?;
    let manifest = directory.manifest();
    let content_digest = manifest
        .content_digest
        .expect("a verified sealed manifest has a digest");
    let modular = !manifest.dependencies.is_empty();
    let (
        verified_files,
        verified_bytes,
        verified_weight_objects,
        verified_tensors,
        largest_object_bytes,
        max_shard_bytes,
        dependency_bundles,
        dependency_closure_verified,
        component_contracts_verified,
        reconstructed_inventory_verified,
    ) = if modular {
        let bundle_root = root
            .parent()
            .ok_or("modular artifact directory must have a sibling-bundle parent")?;
        let dependency_path = |role: &str| -> Result<PathBuf, Box<dyn Error>> {
            let dependency = manifest
                .dependencies
                .iter()
                .find(|dependency| dependency.role.as_str() == role)
                .ok_or_else(|| format!("composition omits dependency role {role}"))?;
            Ok(bundle_root.join(dependency.bundle.as_str()))
        };
        let semantic = verify_modular_release_artifact_directories(
            root,
            dependency_path("qwen")?,
            dependency_path("vae")?,
        )?;
        if require_published_release {
            validate_canonical_release_artifact_digest(
                semantic.variant,
                semantic.profile,
                content_digest,
            )?;
        }
        (
            semantic.verified_files,
            semantic.verified_bytes,
            semantic.verified_weight_objects,
            semantic.verified_tensors,
            semantic.largest_object_bytes,
            semantic
                .parent
                .max_shard_bytes
                .max(semantic.qwen.max_shard_bytes)
                .max(semantic.vae.max_shard_bytes),
            manifest
                .dependencies
                .iter()
                .map(|dependency| dependency.bundle.to_string())
                .collect(),
            semantic.dependency_closure_verified,
            semantic.component_contracts_verified,
            semantic.reconstructed_inventory_verified,
        )
    } else {
        let semantic = if require_published_release {
            verify_published_release_artifact_directory(root)?
        } else {
            verify_release_artifact_directory(root)?
        };
        (
            semantic.verified_files,
            semantic.verified_bytes,
            semantic.verified_weight_objects,
            semantic.verified_tensors,
            semantic.largest_object_bytes,
            semantic.max_shard_bytes,
            Vec::new(),
            true,
            true,
            true,
        )
    };
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
        verified_files,
        verified_bytes,
        verified_weight_objects,
        verified_tensors,
        largest_object_bytes,
        max_shard_bytes,
        dependency_bundles,
        dependency_closure_verified,
        component_contracts_verified,
        reconstructed_inventory_verified,
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
            dependencies: Vec::new(),
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
