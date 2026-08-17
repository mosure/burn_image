//! Verify a sealed Boogu manifest or every payload byte before publication.

use std::{
    collections::BTreeMap,
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
use burn_image::{
    ArtifactFile, ArtifactFileRole, ArtifactManifest, ArtifactShardReader, ArtifactTransportLayout,
    ArtifactVerifier, DirectoryArtifactShardReader, IntegrityPolicy, Sha256Digest,
};
use clap::Parser;
use serde::Serialize;
use sha2::{Digest, Sha256};

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
    /// Verify a manifest-sealed transport layout without fetching its physical part payloads.
    #[arg(long, value_name = "TRANSPORT_LAYOUT_JSON", requires = "manifest_only")]
    transport_layout: Option<PathBuf>,
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
    semantic_weight_objects: usize,
    semantic_payload_bytes: u64,
    largest_semantic_object_bytes: u64,
    maximum_semantic_object_bytes: u64,
    transport_layout_verified: bool,
    transport_payloads_verified: bool,
    transport_object_count: usize,
    transport_part_count: usize,
    transport_logical_bytes: u64,
    transport_payload_bytes: u64,
    largest_transport_part_bytes: u64,
    transport_part_target_bytes: u64,
    maximum_transport_part_bytes: u64,
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
    transport_layout_verified: bool,
    transport_payloads_verified: bool,
    transport_object_count: usize,
    transport_part_count: usize,
    transport_logical_bytes: u64,
    transport_payload_bytes: u64,
    largest_transport_part_bytes: u64,
    transport_part_target_bytes: u64,
    maximum_transport_part_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
struct TransportVerificationStats {
    objects: usize,
    parts: usize,
    logical_bytes: u64,
    payload_bytes: u64,
    largest_part_bytes: u64,
    target_part_bytes: u64,
    maximum_part_bytes: u64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if let Some(manifest) = args.manifest_only.as_deref() {
        let report = verify_manifest_only(manifest, args.transport_layout.as_deref())?;
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

fn verify_manifest_only(
    path: &Path,
    transport_layout: Option<&Path>,
) -> Result<ManifestVerificationReport, Box<dyn Error>> {
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

    let transport = transport_layout
        .map(|path| verify_transport_layout_only(&manifest, path))
        .transpose()?;

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
        verification_scope: if transport.is_some() {
            "manifest-and-sealed-transport-layout-structure-only"
        } else {
            "manifest-structure-and-sealed-content-digest-only"
        },
        sealed_manifest_verified: true,
        payloads_verified: false,
        transport_layout_verified: transport.is_some(),
        transport_payloads_verified: false,
        transport_object_count: transport.map_or(0, |stats| stats.objects),
        transport_part_count: transport.map_or(0, |stats| stats.parts),
        transport_logical_bytes: transport.map_or(0, |stats| stats.logical_bytes),
        transport_payload_bytes: transport.map_or(0, |stats| stats.payload_bytes),
        largest_transport_part_bytes: transport.map_or(0, |stats| stats.largest_part_bytes),
        transport_part_target_bytes: transport.map_or(0, |stats| stats.target_part_bytes),
        maximum_transport_part_bytes: transport.map_or(0, |stats| stats.maximum_part_bytes),
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
    let transport = verify_transport_payload_closure(root, manifest)?;
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
        semantic_weight_objects: verified_weight_objects,
        semantic_payload_bytes: verified_bytes,
        largest_semantic_object_bytes: largest_object_bytes,
        maximum_semantic_object_bytes: max_shard_bytes,
        transport_layout_verified: transport.is_some(),
        transport_payloads_verified: transport.is_some(),
        transport_object_count: transport.map_or(0, |stats| stats.objects),
        transport_part_count: transport.map_or(0, |stats| stats.parts),
        transport_logical_bytes: transport.map_or(0, |stats| stats.logical_bytes),
        transport_payload_bytes: transport.map_or(0, |stats| stats.payload_bytes),
        largest_transport_part_bytes: transport.map_or(0, |stats| stats.largest_part_bytes),
        transport_part_target_bytes: transport.map_or(0, |stats| stats.target_part_bytes),
        maximum_transport_part_bytes: transport.map_or(0, |stats| stats.maximum_part_bytes),
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

fn verify_transport_layout_only(
    manifest: &ArtifactManifest,
    path: &Path,
) -> Result<TransportVerificationStats, Box<dyn Error>> {
    let declaration = ArtifactTransportLayout::declared_file(manifest)?
        .ok_or("sealed manifest does not declare a transport layout")?;
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!(
            "transport layout is not a regular non-symlink file: {}",
            path.display()
        )
        .into());
    }
    if metadata.len() != declaration.size {
        return Err(format!(
            "transport layout {} is {} bytes, expected {}",
            path.display(),
            metadata.len(),
            declaration.size
        )
        .into());
    }
    let bytes = fs::read(path)?;
    let verified = ArtifactTransportLayout::parse_and_validate(manifest, &bytes)?;
    transport_stats(verified.layout())
}

fn verify_transport_payload_closure(
    root: &Path,
    manifest: &ArtifactManifest,
) -> Result<Option<TransportVerificationStats>, Box<dyn Error>> {
    let mut bundles = vec![(root.to_path_buf(), manifest.clone())];
    if !manifest.dependencies.is_empty() {
        let parent = root
            .parent()
            .ok_or("modular artifact directory must have a sibling-bundle parent")?;
        for dependency in &manifest.dependencies {
            let dependency_root = parent.join(dependency.bundle.as_str());
            let directory = VerifiedArtifactDirectory::open(&dependency_root)?;
            dependency.validate_resolved_manifest(directory.manifest())?;
            bundles.push((dependency_root, directory.manifest().clone()));
        }
    }

    let mut total: Option<TransportVerificationStats> = None;
    let mut layouts_present = 0_usize;
    for (bundle_root, bundle_manifest) in &bundles {
        let Some(declaration) = ArtifactTransportLayout::declared_file(bundle_manifest)? else {
            continue;
        };
        layouts_present = layouts_present
            .checked_add(1)
            .ok_or("transport bundle count overflow")?;
        let path = bundle_root.join(declaration.path.as_str());
        let stats = verify_transport_payloads(bundle_root, bundle_manifest, &path)?;
        total = Some(match total {
            Some(accumulated) => accumulated.checked_add(stats)?,
            None => stats,
        });
    }
    if layouts_present != 0 && layouts_present != bundles.len() {
        return Err(format!(
            "transport-part deployment is incomplete: {layouts_present} of {} dependency-closure bundles declare layouts",
            bundles.len()
        )
        .into());
    }
    Ok(total)
}

fn verify_transport_payloads(
    root: &Path,
    manifest: &ArtifactManifest,
    layout_path: &Path,
) -> Result<TransportVerificationStats, Box<dyn Error>> {
    // Authenticate the compact sidecar through the same bounded, non-symlink gate used by
    // manifest-only publication checks before trusting any of its physical paths.
    verify_transport_layout_only(manifest, layout_path)?;
    let layout_bytes = fs::read(layout_path)?;
    let verified = ArtifactTransportLayout::parse_and_validate(manifest, &layout_bytes)?;
    let layout = verified.layout();
    let mut unique_parts = BTreeMap::new();
    let mut reader = DirectoryArtifactShardReader::new(root);
    for object in layout.objects() {
        let mut logical_hasher = Sha256::new();
        let mut logical_bytes = 0_u64;
        for part in &object.parts {
            let part_file = ArtifactFile {
                path: part.path.clone(),
                size: part.size,
                sha256: part.sha256,
                role: ArtifactFileRole::Other,
                component: None,
                shard: None,
            };
            let bytes = reader.read_shard(&part_file)?;
            ArtifactVerifier::verify_bytes(&part_file, &bytes, IntegrityPolicy::RequireSha256)?;
            logical_hasher.update(&bytes);
            logical_bytes = logical_bytes
                .checked_add(part.size)
                .ok_or("reconstructed logical byte count overflow")?;
            if let Some((size, digest)) =
                unique_parts.insert(part.path.clone(), (part.size, part.sha256))
                && (size != part.size || digest != part.sha256)
            {
                return Err(
                    format!("transport part {} has conflicting identities", part.path).into(),
                );
            }
        }
        let logical_digest = Sha256Digest::from_bytes(logical_hasher.finalize().into());
        if logical_bytes != object.size || logical_digest != object.sha256 {
            return Err(format!(
                "reconstructed logical object {} has {logical_bytes}/{logical_digest}, expected {}/{}",
                object.path, object.size, object.sha256
            )
            .into());
        }
    }
    transport_stats_from_parts(layout, &unique_parts)
}

fn transport_stats(
    layout: &ArtifactTransportLayout,
) -> Result<TransportVerificationStats, Box<dyn Error>> {
    let mut parts = BTreeMap::new();
    for object in layout.objects() {
        for part in &object.parts {
            if let Some((size, digest)) = parts.insert(part.path.clone(), (part.size, part.sha256))
                && (size != part.size || digest != part.sha256)
            {
                return Err(
                    format!("transport part {} has conflicting identities", part.path).into(),
                );
            }
        }
    }
    transport_stats_from_parts(layout, &parts)
}

fn transport_stats_from_parts(
    layout: &ArtifactTransportLayout,
    parts: &BTreeMap<burn_image::ArtifactPath, (u64, Sha256Digest)>,
) -> Result<TransportVerificationStats, Box<dyn Error>> {
    let logical_bytes = layout.objects().iter().try_fold(0_u64, |total, object| {
        total
            .checked_add(object.size)
            .ok_or("transport logical byte count overflow")
    })?;
    let (payload_bytes, largest_part_bytes) =
        parts
            .values()
            .try_fold((0_u64, 0_u64), |(total, largest), (size, _)| {
                Ok::<_, Box<dyn Error>>((
                    total
                        .checked_add(*size)
                        .ok_or("transport payload byte count overflow")?,
                    largest.max(*size),
                ))
            })?;
    Ok(TransportVerificationStats {
        objects: layout.objects().len(),
        parts: parts.len(),
        logical_bytes,
        payload_bytes,
        largest_part_bytes,
        target_part_bytes: layout.target_part_bytes,
        maximum_part_bytes: layout.hard_max_part_bytes,
    })
}

impl TransportVerificationStats {
    fn checked_add(self, other: Self) -> Result<Self, Box<dyn Error>> {
        if self.target_part_bytes != other.target_part_bytes
            || self.maximum_part_bytes != other.maximum_part_bytes
        {
            return Err("dependency closure uses inconsistent transport part bounds".into());
        }
        Ok(Self {
            objects: self
                .objects
                .checked_add(other.objects)
                .ok_or("transport object count overflow")?,
            parts: self
                .parts
                .checked_add(other.parts)
                .ok_or("transport part count overflow")?,
            logical_bytes: self
                .logical_bytes
                .checked_add(other.logical_bytes)
                .ok_or("transport logical byte count overflow")?,
            payload_bytes: self
                .payload_bytes
                .checked_add(other.payload_bytes)
                .ok_or("transport payload byte count overflow")?,
            largest_part_bytes: self.largest_part_bytes.max(other.largest_part_bytes),
            target_part_bytes: self.target_part_bytes,
            maximum_part_bytes: self.maximum_part_bytes,
        })
    }
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
        ARTIFACT_MANIFEST_SCHEMA_VERSION, ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES,
        ARTIFACT_TRANSPORT_LAYOUT_PATH, ARTIFACT_TRANSPORT_MAX_PART_BYTES,
        ARTIFACT_TRANSPORT_TARGET_PART_BYTES, ArtifactBundleId, ArtifactFile, ArtifactFileRole,
        ArtifactManifest, ArtifactPath, ArtifactProfileId, ArtifactTransportLayout,
        ArtifactTransportObject, ArtifactTransportPart, ModelId, NumericFormat, Sha256Digest,
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
        verify_artifact_directory, verify_manifest_only, verify_transport_payloads,
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

    fn tiny_transport_directory() -> (tempfile::TempDir, ArtifactManifest) {
        let directory = tempfile::tempdir().unwrap();
        let logical = b"transport reconstruction fixture";
        let logical_path = ArtifactPath::new("objects/logical.bpk").unwrap();
        let part_digest = Sha256Digest::calculate(logical);
        let part_path = ArtifactPath::new(format!("transport/{part_digest}.part")).unwrap();
        fs::create_dir_all(directory.path().join("metadata")).unwrap();
        fs::create_dir_all(directory.path().join("transport")).unwrap();
        fs::write(directory.path().join(part_path.as_str()), logical).unwrap();

        let mut manifest = ArtifactManifest {
            schema_version: ARTIFACT_MANIFEST_SCHEMA_VERSION,
            bundle: ArtifactBundleId::new("tiny-transport-bundle").unwrap(),
            profile: ArtifactProfileId::new("tiny-profile").unwrap(),
            model: ModelId::new("tests/tiny-transport-model").unwrap(),
            model_revision: "immutable-test-revision".into(),
            numeric_format: NumericFormat::F16,
            components: Vec::new(),
            files: vec![ArtifactFile {
                path: logical_path.clone(),
                size: logical.len() as u64,
                sha256: Sha256Digest::calculate(logical),
                role: ArtifactFileRole::Weights,
                component: None,
                shard: None,
            }],
            dependencies: Vec::new(),
            metadata: BTreeMap::from([
                (
                    "target_max_shard_bytes".into(),
                    ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES.to_string(),
                ),
                (
                    "semantic_object_max_bytes".into(),
                    ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES.to_string(),
                ),
                (
                    "transport_layout_path".into(),
                    ARTIFACT_TRANSPORT_LAYOUT_PATH.into(),
                ),
                ("transport_layout_schema".into(), "1".into()),
                ("transport_parts_required".into(), "true".into()),
                (
                    "transport_part_target_bytes".into(),
                    ARTIFACT_TRANSPORT_TARGET_PART_BYTES.to_string(),
                ),
                (
                    "target_max_transport_shard_bytes".into(),
                    ARTIFACT_TRANSPORT_MAX_PART_BYTES.to_string(),
                ),
            ]),
            content_digest: None,
        };
        let layout = ArtifactTransportLayout {
            schema_version: 1,
            bundle: manifest.bundle.clone(),
            profile: manifest.profile.clone(),
            model: manifest.model.clone(),
            model_revision: manifest.model_revision.clone(),
            target_part_bytes: ARTIFACT_TRANSPORT_TARGET_PART_BYTES,
            hard_max_part_bytes: ARTIFACT_TRANSPORT_MAX_PART_BYTES,
            objects: vec![ArtifactTransportObject {
                path: logical_path,
                size: logical.len() as u64,
                sha256: Sha256Digest::calculate(logical),
                parts: vec![ArtifactTransportPart {
                    path: part_path,
                    offset: 0,
                    size: logical.len() as u64,
                    sha256: part_digest,
                }],
            }],
        };
        let mut layout_bytes = serde_json::to_vec_pretty(&layout).unwrap();
        layout_bytes.push(b'\n');
        fs::write(
            directory.path().join(ARTIFACT_TRANSPORT_LAYOUT_PATH),
            &layout_bytes,
        )
        .unwrap();
        manifest.files.push(ArtifactFile {
            path: ArtifactPath::new(ARTIFACT_TRANSPORT_LAYOUT_PATH).unwrap(),
            size: layout_bytes.len() as u64,
            sha256: Sha256Digest::calculate(&layout_bytes),
            role: ArtifactFileRole::Metadata,
            component: None,
            shard: None,
        });
        manifest.seal().unwrap();
        fs::write(
            directory.path().join("manifest.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        (directory, manifest)
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

        let report = verify_manifest_only(&directory.path().join("manifest.json"), None).unwrap();
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

        let error = verify_manifest_only(&path, None).unwrap_err();
        assert!(error.to_string().contains("content digest mismatch"));
    }

    #[test]
    fn manifest_only_authenticates_transport_layout_without_claiming_payloads_correctness() {
        let (directory, _) = tiny_transport_directory();
        let report = verify_manifest_only(
            &directory.path().join("manifest.json"),
            Some(&directory.path().join(ARTIFACT_TRANSPORT_LAYOUT_PATH)),
        )
        .unwrap();
        assert!(report.transport_layout_verified);
        assert!(!report.transport_payloads_verified);
        assert_eq!(report.transport_object_count, 1);
        assert_eq!(report.transport_part_count, 1);
        assert_eq!(report.largest_transport_part_bytes, 32);
    }

    #[test]
    fn full_transport_verification_hashes_parts_and_logical_reconstruction_correctness() {
        let (directory, manifest) = tiny_transport_directory();
        let layout_path = directory.path().join(ARTIFACT_TRANSPORT_LAYOUT_PATH);
        let report = verify_transport_payloads(directory.path(), &manifest, &layout_path).unwrap();
        assert_eq!(report.objects, 1);
        assert_eq!(report.parts, 1);

        let layout: ArtifactTransportLayout =
            serde_json::from_slice(&fs::read(&layout_path).unwrap()).unwrap();
        let part_path = directory
            .path()
            .join(layout.objects[0].parts[0].path.as_str());
        fs::write(&part_path, b"transport reconstruction fixturE").unwrap();
        let error =
            verify_transport_payloads(directory.path(), &manifest, &layout_path).unwrap_err();
        let message = error.to_string();
        assert!(
            message.contains("integrity") || message.contains("SHA-256"),
            "unexpected transport corruption error: {message}"
        );
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
