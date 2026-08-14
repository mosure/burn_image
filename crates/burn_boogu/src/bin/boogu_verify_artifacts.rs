//! Verify a sealed Boogu manifest or every payload byte before publication.

use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use burn_boogu::artifacts::{
    BooguStorageProfile, VerifiedArtifactDirectory, promotable_legacy_artifact_digest,
    validate_canonical_release_artifact_digest, validate_edit_turbo_1k5_release_artifact_digest,
    verify_modular_release_artifact_directories, verify_published_release_artifact_directory,
    verify_release_artifact_directory,
};
use burn_boogu::config::BooguVariant;
use burn_image::{ArtifactManifest, Sha256Digest};
use clap::Parser;
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(about = "Verify a sealed Boogu manifest or every payload and semantic tensor contract")]
struct Args {
    /// Directory containing the sealed manifest and its authenticated payloads.
    #[arg(long, required_unless_present = "manifest_only")]
    artifacts: Option<PathBuf>,
    /// Verify only one manifest's structure and sealed content digest without reading payloads.
    #[arg(
        long,
        value_name = "MANIFEST_JSON",
        conflicts_with_all = [
            "artifacts",
            "require_edit_turbo_1k5_release",
            "require_published_release",
            "require_legacy_flat_parity_release"
        ]
    )]
    manifest_only: Option<PathBuf>,
    /// Require the exact mixed-F16 artifact digest qualified for Edit-Turbo 1.5K.
    #[arg(long, default_value_t = false)]
    require_edit_turbo_1k5_release: bool,
    /// Require the exact pinned digest for one of the three canonical production bundles.
    #[arg(long, default_value_t = false)]
    require_published_release: bool,
    /// Require the exact legacy schema-v1 flat digest consumed by native parity binaries.
    #[arg(long, default_value_t = false)]
    require_legacy_flat_parity_release: bool,
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
    legacy_flat_parity_release_verified: bool,
    semantic_contract_verified: bool,
    artifacts_verified: bool,
}

#[derive(Debug, Serialize)]
struct ManifestVerificationReport {
    manifest: PathBuf,
    manifest_sha256: String,
    schema_version: u32,
    bundle: String,
    profile: String,
    model: String,
    model_revision: String,
    content_digest: String,
    declared_files: usize,
    declared_bytes: u64,
    dependency_bundles: Vec<String>,
    verification_scope: &'static str,
    sealed_manifest_verified: bool,
    payloads_verified: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if let Some(manifest) = args.manifest_only.as_deref() {
        let report = verify_manifest_only(manifest)?;
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    let report = verify_artifact_directory(
        args.artifacts
            .as_deref()
            .expect("clap requires --artifacts unless --manifest-only is selected"),
        args.require_edit_turbo_1k5_release,
        args.require_published_release,
        args.require_legacy_flat_parity_release,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn verify_manifest_only(path: &Path) -> Result<ManifestVerificationReport, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let manifest: ArtifactManifest = serde_json::from_slice(&bytes)?;
    manifest.validate_sealed()?;
    let declared_bytes = manifest
        .files
        .iter()
        .try_fold(0_u64, |total, file| total.checked_add(file.size))
        .ok_or("manifest declared byte count overflows u64")?;
    let content_digest = manifest
        .content_digest
        .expect("validate_sealed requires a content digest");

    Ok(ManifestVerificationReport {
        manifest: path.to_path_buf(),
        manifest_sha256: Sha256Digest::calculate(&bytes).to_string(),
        schema_version: manifest.schema_version,
        bundle: manifest.bundle.to_string(),
        profile: manifest.profile.to_string(),
        model: manifest.model.to_string(),
        model_revision: manifest.model_revision,
        content_digest: content_digest.to_string(),
        declared_files: manifest.files.len(),
        declared_bytes,
        dependency_bundles: manifest
            .dependencies
            .into_iter()
            .map(|dependency| dependency.bundle.to_string())
            .collect(),
        verification_scope: "manifest-structure-and-sealed-content-digest-only",
        sealed_manifest_verified: true,
        payloads_verified: false,
    })
}

fn verify_artifact_directory(
    root: &Path,
    require_edit_turbo_1k5_release: bool,
    require_published_release: bool,
    require_legacy_flat_parity_release: bool,
) -> Result<VerificationReport, Box<dyn Error>> {
    if require_published_release && require_legacy_flat_parity_release {
        return Err(
            "canonical modular publication and legacy flat parity are distinct artifact contracts"
                .into(),
        );
    }
    let directory = VerifiedArtifactDirectory::open(root)?;
    let manifest = directory.manifest();
    let content_digest = manifest
        .content_digest
        .expect("a verified sealed manifest has a digest");
    let modular = !manifest.dependencies.is_empty();
    if modular && require_legacy_flat_parity_release {
        return Err("legacy flat parity verification rejects dependency-composed artifacts".into());
    }
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
        if require_legacy_flat_parity_release {
            validate_legacy_flat_parity_release_digest(
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
        legacy_flat_parity_release_verified: require_legacy_flat_parity_release,
        semantic_contract_verified: true,
        artifacts_verified: true,
    })
}

fn validate_legacy_flat_parity_release_digest(
    variant: BooguVariant,
    profile: BooguStorageProfile,
    actual: Sha256Digest,
) -> Result<(), Box<dyn Error>> {
    let expected = promotable_legacy_artifact_digest(variant, profile).ok_or_else(|| {
        format!(
            "{variant:?}/{profile:?} has no exact legacy flat artifact qualified for native parity"
        )
    })?;
    let actual = actual.to_string();
    if actual != expected {
        return Err(format!(
            "legacy flat parity artifact for {variant:?}/{profile:?} requires sealed digest {expected}, found {actual}"
        )
        .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, fs};

    use burn_boogu::artifacts::BooguArtifactLoadError;
    use burn_image::{
        ARTIFACT_MANIFEST_SCHEMA_VERSION, ArtifactBundleId, ArtifactFile, ArtifactFileRole,
        ArtifactManifest, ArtifactPath, ArtifactProfileId, ModelId, NumericFormat, Sha256Digest,
    };

    use burn_boogu::{
        artifacts::{
            BooguStorageProfile, EDIT_TURBO_1K5_F16_QWEN_VISION_F32_CONTENT_DIGEST,
            LEGACY_EDIT_TURBO_1K5_F16_QWEN_VISION_F32_CONTENT_DIGEST,
        },
        config::BooguVariant,
    };

    use super::{
        Args, VerifiedArtifactDirectory, validate_legacy_flat_parity_release_digest,
        verify_artifact_directory, verify_manifest_only,
    };
    use clap::Parser;

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
        let error = verify_artifact_directory(directory.path(), false, false, false).unwrap_err();
        assert!(matches!(
            error.downcast_ref::<BooguArtifactLoadError>(),
            Some(BooguArtifactLoadError::Identity(message))
                if message.contains("unsupported Boogu release model")
        ));
    }

    #[test]
    fn manifest_only_verifies_the_sealed_declaration_without_payload_reads_correctness() {
        let directory = tiny_sealed_directory();
        fs::remove_file(directory.path().join("objects/tiny.bin")).unwrap();

        let report = verify_manifest_only(&directory.path().join("manifest.json")).unwrap();
        assert_eq!(report.bundle, "tiny-bundle");
        assert_eq!(report.declared_files, 1);
        assert_eq!(report.declared_bytes, PAYLOAD.len() as u64);
        assert!(report.sealed_manifest_verified);
        assert!(!report.payloads_verified);
    }

    #[test]
    fn manifest_only_rejects_a_declaration_changed_after_sealing_correctness() {
        let directory = tiny_sealed_directory();
        let path = directory.path().join("manifest.json");
        let mut manifest: ArtifactManifest =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        manifest.model_revision.push_str("-tampered");
        fs::write(&path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();

        let error = verify_manifest_only(&path).unwrap_err();
        assert!(error.to_string().contains("content digest mismatch"));
    }

    #[test]
    fn manifest_only_rejects_payload_verification_flags_correctness() {
        let error = Args::try_parse_from([
            "boogu-verify-artifacts",
            "--manifest-only",
            "manifest.json",
            "--require-published-release",
        ])
        .unwrap_err();
        assert!(error.to_string().contains("cannot be used with"));
    }

    #[test]
    fn legacy_flat_parity_digest_is_distinct_from_canonical_composition_correctness() {
        let variant = BooguVariant::Image01EditTurbo1k5;
        let profile = BooguStorageProfile::F16QwenVisionF32;
        let legacy =
            Sha256Digest::from_hex(LEGACY_EDIT_TURBO_1K5_F16_QWEN_VISION_F32_CONTENT_DIGEST)
                .unwrap();
        validate_legacy_flat_parity_release_digest(variant, profile, legacy).unwrap();

        let canonical =
            Sha256Digest::from_hex(EDIT_TURBO_1K5_F16_QWEN_VISION_F32_CONTENT_DIGEST).unwrap();
        let error =
            validate_legacy_flat_parity_release_digest(variant, profile, canonical).unwrap_err();
        assert!(error.to_string().contains("requires sealed digest"));
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
