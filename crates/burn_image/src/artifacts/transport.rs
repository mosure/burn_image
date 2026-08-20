use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    ArtifactBundleId, ArtifactFile, ArtifactFileRole, ArtifactManifest, ArtifactPath,
    ArtifactProfileId, ArtifactVerifier, IntegrityError, IntegrityPolicy, ManifestError, ModelId,
    Sha256Digest,
};

/// Exact manifest path of the sealed transport-layout sidecar.
pub const ARTIFACT_TRANSPORT_LAYOUT_PATH: &str = "metadata/transport-layout.json";
/// Current transport-layout sidecar schema.
pub const ARTIFACT_TRANSPORT_LAYOUT_SCHEMA_VERSION: u32 = 1;
/// Deterministic physical part target: five 4 MiB browser-cache ranges.
pub const ARTIFACT_TRANSPORT_TARGET_PART_BYTES: u64 = 20 * 1024 * 1024;
/// Exact decimal hard ceiling for every physical CDN object.
pub const ARTIFACT_TRANSPORT_MAX_PART_BYTES: u64 = 25_000_000;
/// Existing logical Burnpack object ceiling, distinct from the physical transport-part ceiling.
pub const ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES: u64 = 256 * 1024 * 1024;
/// Bootstrap bound for the manifest-sealed JSON sidecar.
pub const MAX_ARTIFACT_TRANSPORT_LAYOUT_BYTES: u64 = 4 * 1024 * 1024;

pub const ARTIFACT_TRANSPORT_LAYOUT_PATH_KEY: &str = "transport_layout_path";
pub const ARTIFACT_TRANSPORT_LAYOUT_SCHEMA_KEY: &str = "transport_layout_schema";
pub const ARTIFACT_TRANSPORT_PARTS_REQUIRED_KEY: &str = "transport_parts_required";
pub const ARTIFACT_TRANSPORT_PART_TARGET_BYTES_KEY: &str = "transport_part_target_bytes";
pub const ARTIFACT_TARGET_MAX_TRANSPORT_SHARD_BYTES_KEY: &str = "target_max_transport_shard_bytes";
pub const ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES_KEY: &str = "semantic_object_max_bytes";
pub const ARTIFACT_TARGET_MAX_SEMANTIC_SHARD_BYTES_KEY: &str = "target_max_shard_bytes";

/// One content-addressed physical part of a logical manifest weight object.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactTransportPart {
    pub path: ArtifactPath,
    pub offset: u64,
    pub size: u64,
    pub sha256: Sha256Digest,
}

/// Transport reconstruction plan for one logical manifest weight object.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactTransportObject {
    pub path: ArtifactPath,
    pub size: u64,
    pub sha256: Sha256Digest,
    pub parts: Vec<ArtifactTransportPart>,
}

/// Manifest-sealed mapping from logical weight objects to bounded physical parts.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactTransportLayout {
    pub schema_version: u32,
    pub bundle: ArtifactBundleId,
    pub profile: ArtifactProfileId,
    pub model: ModelId,
    pub model_revision: String,
    pub target_part_bytes: u64,
    pub hard_max_part_bytes: u64,
    pub objects: Vec<ArtifactTransportObject>,
}

/// A transport layout whose exact serialized bytes and complete structure were checked against a
/// sealed manifest.
#[derive(Clone, Debug)]
pub struct VerifiedArtifactTransportLayout {
    layout: ArtifactTransportLayout,
    manifest_content_digest: Sha256Digest,
}

impl VerifiedArtifactTransportLayout {
    pub fn layout(&self) -> &ArtifactTransportLayout {
        &self.layout
    }

    pub fn objects(&self) -> &[ArtifactTransportObject] {
        &self.layout.objects
    }

    pub fn object(&self, path: &ArtifactPath) -> Option<&ArtifactTransportObject> {
        self.layout.object(path)
    }

    pub fn manifest_content_digest(&self) -> Sha256Digest {
        self.manifest_content_digest
    }
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ArtifactTransportLayoutError {
    #[error(transparent)]
    Manifest(#[from] ManifestError),
    #[error(transparent)]
    Integrity(#[from] IntegrityError),
    #[error("artifact manifest does not declare '{ARTIFACT_TRANSPORT_LAYOUT_PATH}'")]
    MissingDeclaration,
    #[error("artifact transport layout declaration is invalid: {0}")]
    InvalidDeclaration(String),
    #[error(
        "direct artifact file '{path}' has {size} bytes, above the {maximum}-byte physical CDN object cap"
    )]
    DirectFileSizeOutOfBounds {
        path: ArtifactPath,
        size: u64,
        maximum: u64,
    },
    #[error("artifact transport layout JSON is invalid: {0}")]
    InvalidJson(String),
    #[error("unsupported artifact transport layout schema {actual}; expected {expected}")]
    UnsupportedSchema { expected: u32, actual: u32 },
    #[error("artifact transport layout {field} mismatch: expected '{expected}', got '{actual}'")]
    IdentityMismatch {
        field: &'static str,
        expected: String,
        actual: String,
    },
    #[error("artifact manifest metadata '{key}' must be '{expected}', got {actual:?}")]
    MetadataMismatch {
        key: &'static str,
        expected: String,
        actual: Option<String>,
    },
    #[error(
        "artifact manifest metadata '{key}' is not a canonical positive decimal u64: '{value}'"
    )]
    InvalidMetadataInteger { key: &'static str, value: String },
    #[error("artifact transport layout has no logical weight objects")]
    EmptyObjects,
    #[error("artifact transport logical objects are not strictly path-sorted at '{path}'")]
    ObjectsNotSorted { path: ArtifactPath },
    #[error("artifact transport layout maps unknown logical weight '{path}'")]
    UnknownLogicalObject { path: ArtifactPath },
    #[error("artifact transport layout maps logical weight '{path}' more than once")]
    DuplicateLogicalObject { path: ArtifactPath },
    #[error("artifact transport layout omits logical weight '{path}'")]
    MissingLogicalObject { path: ArtifactPath },
    #[error("artifact transport logical identity mismatch for '{path}': {message}")]
    LogicalIdentityMismatch { path: ArtifactPath, message: String },
    #[error("artifact transport logical object '{path}' has no physical parts")]
    EmptyParts { path: ArtifactPath },
    #[error(
        "artifact transport part '{part}' for '{object}' starts at {actual}, expected {expected}"
    )]
    PartOffsetMismatch {
        object: ArtifactPath,
        part: ArtifactPath,
        expected: u64,
        actual: u64,
    },
    #[error("artifact transport part '{part}' has {size} bytes, outside {minimum}..={maximum}")]
    PartSizeOutOfBounds {
        part: ArtifactPath,
        size: u64,
        minimum: u64,
        maximum: u64,
    },
    #[error(
        "non-final artifact transport part '{part}' has {actual} bytes, expected exactly {expected}"
    )]
    NonFinalPartSize {
        part: ArtifactPath,
        expected: u64,
        actual: u64,
    },
    #[error("artifact transport part path mismatch: expected '{expected}', got '{actual}'")]
    PartPathMismatch {
        expected: ArtifactPath,
        actual: ArtifactPath,
    },
    #[error("artifact transport part path '{path}' collides with a logical manifest file")]
    PartPathCollidesWithManifestFile { path: ArtifactPath },
    #[error("artifact transport part path '{path}' is reused with conflicting identity")]
    PartPathConflict { path: ArtifactPath },
    #[error(
        "artifact transport parts cover {actual} bytes for '{path}', expected exactly {expected}"
    )]
    CoverageMismatch {
        path: ArtifactPath,
        expected: u64,
        actual: u64,
    },
    #[error("artifact transport byte offset overflow for '{path}'")]
    OffsetOverflow { path: ArtifactPath },
}

impl ArtifactTransportLayout {
    /// Return the declared sidecar, rejecting partial or contradictory transport declarations.
    pub fn declared_file(
        manifest: &ArtifactManifest,
    ) -> Result<Option<&ArtifactFile>, ArtifactTransportLayoutError> {
        manifest.validate_sealed()?;
        let file = manifest
            .files
            .iter()
            .find(|file| file.path.as_str() == ARTIFACT_TRANSPORT_LAYOUT_PATH);
        let has_transport_metadata = [
            ARTIFACT_TRANSPORT_LAYOUT_PATH_KEY,
            ARTIFACT_TRANSPORT_LAYOUT_SCHEMA_KEY,
            ARTIFACT_TRANSPORT_PARTS_REQUIRED_KEY,
            ARTIFACT_TRANSPORT_PART_TARGET_BYTES_KEY,
            ARTIFACT_TARGET_MAX_TRANSPORT_SHARD_BYTES_KEY,
            ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES_KEY,
        ]
        .iter()
        .any(|key| manifest.metadata.contains_key(*key));

        let Some(file) = file else {
            if has_transport_metadata {
                return Err(ArtifactTransportLayoutError::InvalidDeclaration(
                    "transport metadata is present without the sealed sidecar".into(),
                ));
            }
            return Ok(None);
        };
        if file.role != ArtifactFileRole::Metadata
            || file.component.is_some()
            || file.shard.is_some()
        {
            return Err(ArtifactTransportLayoutError::InvalidDeclaration(
                "the sidecar must have role=metadata with no component or semantic shard".into(),
            ));
        }
        if file.size > MAX_ARTIFACT_TRANSPORT_LAYOUT_BYTES {
            return Err(ArtifactTransportLayoutError::InvalidDeclaration(format!(
                "the sidecar has {} bytes, above the {}-byte bootstrap cap",
                file.size, MAX_ARTIFACT_TRANSPORT_LAYOUT_BYTES
            )));
        }
        for direct_file in manifest
            .files
            .iter()
            .filter(|direct_file| direct_file.role != ArtifactFileRole::Weights)
        {
            if direct_file.size > ARTIFACT_TRANSPORT_MAX_PART_BYTES {
                return Err(ArtifactTransportLayoutError::DirectFileSizeOutOfBounds {
                    path: direct_file.path.clone(),
                    size: direct_file.size,
                    maximum: ARTIFACT_TRANSPORT_MAX_PART_BYTES,
                });
            }
        }
        validate_transport_metadata(manifest)?;
        Ok(Some(file))
    }

    /// Parse a sidecar only after verifying its exact manifest-declared size and SHA-256 digest.
    pub fn parse_and_validate(
        manifest: &ArtifactManifest,
        bytes: &[u8],
    ) -> Result<VerifiedArtifactTransportLayout, ArtifactTransportLayoutError> {
        let file = Self::declared_file(manifest)?
            .ok_or(ArtifactTransportLayoutError::MissingDeclaration)?;
        ArtifactVerifier::verify_bytes(file, bytes, IntegrityPolicy::RequireSha256)?;
        let layout: Self = serde_json::from_slice(bytes)
            .map_err(|error| ArtifactTransportLayoutError::InvalidJson(error.to_string()))?;
        layout.validate_for_manifest(manifest)?;
        Ok(VerifiedArtifactTransportLayout {
            layout,
            manifest_content_digest: manifest
                .content_digest
                .expect("declared_file validated the sealed manifest"),
        })
    }

    /// Validate a constructed layout against a sealed manifest.
    ///
    /// This structural check deliberately does not return verified typestate because it cannot
    /// prove that these in-memory values came from the manifest-declared sidecar bytes.
    pub fn validate_for_manifest(
        &self,
        manifest: &ArtifactManifest,
    ) -> Result<(), ArtifactTransportLayoutError> {
        Self::declared_file(manifest)?.ok_or(ArtifactTransportLayoutError::MissingDeclaration)?;
        if self.schema_version != ARTIFACT_TRANSPORT_LAYOUT_SCHEMA_VERSION {
            return Err(ArtifactTransportLayoutError::UnsupportedSchema {
                expected: ARTIFACT_TRANSPORT_LAYOUT_SCHEMA_VERSION,
                actual: self.schema_version,
            });
        }
        validate_identity("bundle", manifest.bundle.as_str(), self.bundle.as_str())?;
        validate_identity("profile", manifest.profile.as_str(), self.profile.as_str())?;
        validate_identity("model", manifest.model.as_str(), self.model.as_str())?;
        validate_identity(
            "model_revision",
            &manifest.model_revision,
            &self.model_revision,
        )?;
        if self.target_part_bytes != ARTIFACT_TRANSPORT_TARGET_PART_BYTES {
            return Err(ArtifactTransportLayoutError::IdentityMismatch {
                field: "target_part_bytes",
                expected: ARTIFACT_TRANSPORT_TARGET_PART_BYTES.to_string(),
                actual: self.target_part_bytes.to_string(),
            });
        }
        if self.hard_max_part_bytes != ARTIFACT_TRANSPORT_MAX_PART_BYTES {
            return Err(ArtifactTransportLayoutError::IdentityMismatch {
                field: "hard_max_part_bytes",
                expected: ARTIFACT_TRANSPORT_MAX_PART_BYTES.to_string(),
                actual: self.hard_max_part_bytes.to_string(),
            });
        }

        let semantic_max = semantic_object_max_bytes(manifest)?;
        let logical_files = manifest
            .files
            .iter()
            .filter(|file| file.role == ArtifactFileRole::Weights)
            .map(|file| (file.path.clone(), file))
            .collect::<BTreeMap<_, _>>();
        if logical_files.is_empty() || self.objects.is_empty() {
            return Err(ArtifactTransportLayoutError::EmptyObjects);
        }
        let manifest_paths = manifest
            .files
            .iter()
            .map(|file| file.path.clone())
            .collect::<BTreeSet<_>>();
        let mut seen_objects = BTreeSet::new();
        let mut previous_object: Option<&ArtifactPath> = None;
        let mut physical_parts: BTreeMap<ArtifactPath, (u64, Sha256Digest)> = BTreeMap::new();

        for object in &self.objects {
            if previous_object.is_some_and(|previous| previous >= &object.path) {
                if previous_object == Some(&object.path) {
                    return Err(ArtifactTransportLayoutError::DuplicateLogicalObject {
                        path: object.path.clone(),
                    });
                }
                return Err(ArtifactTransportLayoutError::ObjectsNotSorted {
                    path: object.path.clone(),
                });
            }
            previous_object = Some(&object.path);
            if !seen_objects.insert(object.path.clone()) {
                return Err(ArtifactTransportLayoutError::DuplicateLogicalObject {
                    path: object.path.clone(),
                });
            }
            let logical = logical_files.get(&object.path).ok_or_else(|| {
                ArtifactTransportLayoutError::UnknownLogicalObject {
                    path: object.path.clone(),
                }
            })?;
            if logical.size != object.size || logical.sha256 != object.sha256 {
                return Err(ArtifactTransportLayoutError::LogicalIdentityMismatch {
                    path: object.path.clone(),
                    message: format!(
                        "declared size/SHA-256 is {}/{}, sidecar has {}/{}",
                        logical.size, logical.sha256, object.size, object.sha256
                    ),
                });
            }
            if object.size > semantic_max {
                return Err(ArtifactTransportLayoutError::LogicalIdentityMismatch {
                    path: object.path.clone(),
                    message: format!(
                        "{} bytes exceeds semantic object cap {semantic_max}",
                        object.size
                    ),
                });
            }
            if object.parts.is_empty() {
                return Err(ArtifactTransportLayoutError::EmptyParts {
                    path: object.path.clone(),
                });
            }

            let mut expected_offset = 0_u64;
            let final_index = object.parts.len() - 1;
            for (index, part) in object.parts.iter().enumerate() {
                if part.offset != expected_offset {
                    return Err(ArtifactTransportLayoutError::PartOffsetMismatch {
                        object: object.path.clone(),
                        part: part.path.clone(),
                        expected: expected_offset,
                        actual: part.offset,
                    });
                }
                if part.size == 0 || part.size > ARTIFACT_TRANSPORT_MAX_PART_BYTES {
                    return Err(ArtifactTransportLayoutError::PartSizeOutOfBounds {
                        part: part.path.clone(),
                        size: part.size,
                        minimum: 1,
                        maximum: ARTIFACT_TRANSPORT_MAX_PART_BYTES,
                    });
                }
                if part.size > ARTIFACT_TRANSPORT_TARGET_PART_BYTES {
                    return Err(ArtifactTransportLayoutError::PartSizeOutOfBounds {
                        part: part.path.clone(),
                        size: part.size,
                        minimum: 1,
                        maximum: ARTIFACT_TRANSPORT_TARGET_PART_BYTES,
                    });
                }
                if index != final_index && part.size != ARTIFACT_TRANSPORT_TARGET_PART_BYTES {
                    return Err(ArtifactTransportLayoutError::NonFinalPartSize {
                        part: part.path.clone(),
                        expected: ARTIFACT_TRANSPORT_TARGET_PART_BYTES,
                        actual: part.size,
                    });
                }
                let expected_path = ArtifactPath::new(format!("transport/{}.part", part.sha256))
                    .expect("a lowercase SHA-256 digest is a valid artifact path");
                if part.path != expected_path {
                    return Err(ArtifactTransportLayoutError::PartPathMismatch {
                        expected: expected_path,
                        actual: part.path.clone(),
                    });
                }
                if manifest_paths.contains(&part.path) {
                    return Err(
                        ArtifactTransportLayoutError::PartPathCollidesWithManifestFile {
                            path: part.path.clone(),
                        },
                    );
                }
                match physical_parts.get(&part.path) {
                    Some(identity) if *identity != (part.size, part.sha256) => {
                        return Err(ArtifactTransportLayoutError::PartPathConflict {
                            path: part.path.clone(),
                        });
                    }
                    Some(_) => {}
                    None => {
                        physical_parts.insert(part.path.clone(), (part.size, part.sha256));
                    }
                }
                expected_offset = expected_offset.checked_add(part.size).ok_or_else(|| {
                    ArtifactTransportLayoutError::OffsetOverflow {
                        path: object.path.clone(),
                    }
                })?;
            }
            if expected_offset != object.size {
                return Err(ArtifactTransportLayoutError::CoverageMismatch {
                    path: object.path.clone(),
                    expected: object.size,
                    actual: expected_offset,
                });
            }
        }

        for path in logical_files.keys() {
            if !seen_objects.contains(path) {
                return Err(ArtifactTransportLayoutError::MissingLogicalObject {
                    path: path.clone(),
                });
            }
        }
        Ok(())
    }

    pub fn objects(&self) -> &[ArtifactTransportObject] {
        &self.objects
    }

    pub fn object(&self, path: &ArtifactPath) -> Option<&ArtifactTransportObject> {
        self.objects
            .binary_search_by(|object| object.path.cmp(path))
            .ok()
            .map(|index| &self.objects[index])
    }
}

fn validate_transport_metadata(
    manifest: &ArtifactManifest,
) -> Result<(), ArtifactTransportLayoutError> {
    for (key, expected) in [
        (
            ARTIFACT_TRANSPORT_LAYOUT_PATH_KEY,
            ARTIFACT_TRANSPORT_LAYOUT_PATH,
        ),
        (ARTIFACT_TRANSPORT_LAYOUT_SCHEMA_KEY, "1"),
        (ARTIFACT_TRANSPORT_PARTS_REQUIRED_KEY, "true"),
        (ARTIFACT_TRANSPORT_PART_TARGET_BYTES_KEY, "20971520"),
        (ARTIFACT_TARGET_MAX_TRANSPORT_SHARD_BYTES_KEY, "25000000"),
    ] {
        let actual = manifest.metadata.get(key);
        if actual.map(String::as_str) != Some(expected) {
            return Err(ArtifactTransportLayoutError::MetadataMismatch {
                key,
                expected: expected.into(),
                actual: actual.cloned(),
            });
        }
    }
    semantic_object_max_bytes(manifest)?;
    Ok(())
}

fn semantic_object_max_bytes(
    manifest: &ArtifactManifest,
) -> Result<u64, ArtifactTransportLayoutError> {
    let semantic =
        parse_canonical_positive_u64_metadata(manifest, ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES_KEY)?;
    let declared = parse_canonical_positive_u64_metadata(
        manifest,
        ARTIFACT_TARGET_MAX_SEMANTIC_SHARD_BYTES_KEY,
    )?;
    if semantic != ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES {
        return Err(ArtifactTransportLayoutError::MetadataMismatch {
            key: ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES_KEY,
            expected: ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES.to_string(),
            actual: Some(semantic.to_string()),
        });
    }
    if declared != ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES {
        return Err(ArtifactTransportLayoutError::MetadataMismatch {
            key: ARTIFACT_TARGET_MAX_SEMANTIC_SHARD_BYTES_KEY,
            expected: ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES.to_string(),
            actual: Some(declared.to_string()),
        });
    }
    Ok(semantic)
}

fn parse_canonical_positive_u64_metadata(
    manifest: &ArtifactManifest,
    key: &'static str,
) -> Result<u64, ArtifactTransportLayoutError> {
    let value = manifest.metadata.get(key).cloned().unwrap_or_default();
    let parsed = value.parse::<u64>().ok();
    match parsed {
        Some(parsed) if parsed > 0 && parsed.to_string() == value => Ok(parsed),
        _ => Err(ArtifactTransportLayoutError::InvalidMetadataInteger { key, value }),
    }
}

fn validate_identity(
    field: &'static str,
    expected: &str,
    actual: &str,
) -> Result<(), ArtifactTransportLayoutError> {
    if expected == actual {
        Ok(())
    } else {
        Err(ArtifactTransportLayoutError::IdentityMismatch {
            field,
            expected: expected.into(),
            actual: actual.into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::json;

    use crate::{ARTIFACT_MANIFEST_SCHEMA_V1, ArtifactFileRole, NumericFormat};

    use super::*;

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

    fn logical_file(path: &str, size: u64, sha256: Sha256Digest) -> ArtifactFile {
        ArtifactFile {
            path: ArtifactPath::new(path).unwrap(),
            size,
            sha256,
            role: ArtifactFileRole::Weights,
            component: None,
            shard: None,
        }
    }

    fn part(size: u64, offset: u64, marker: &[u8]) -> ArtifactTransportPart {
        let sha256 = Sha256Digest::calculate(marker);
        ArtifactTransportPart {
            path: ArtifactPath::new(format!("transport/{sha256}.part")).unwrap(),
            offset,
            size,
            sha256,
        }
    }

    fn layout_for_object(object: ArtifactTransportObject) -> ArtifactTransportLayout {
        ArtifactTransportLayout {
            schema_version: ARTIFACT_TRANSPORT_LAYOUT_SCHEMA_VERSION,
            bundle: ArtifactBundleId::new("transport-test").unwrap(),
            profile: ArtifactProfileId::new("f16").unwrap(),
            model: ModelId::new("example/transport-test").unwrap(),
            model_revision: "0123456789abcdef".into(),
            target_part_bytes: ARTIFACT_TRANSPORT_TARGET_PART_BYTES,
            hard_max_part_bytes: ARTIFACT_TRANSPORT_MAX_PART_BYTES,
            objects: vec![object],
        }
    }

    fn manifest_for_layout(layout: &ArtifactTransportLayout) -> (ArtifactManifest, Vec<u8>) {
        let sidecar = serde_json::to_vec(layout).unwrap();
        let logical_files = layout
            .objects
            .iter()
            .map(|object| logical_file(object.path.as_str(), object.size, object.sha256));
        let sidecar_file = ArtifactFile {
            path: ArtifactPath::new(ARTIFACT_TRANSPORT_LAYOUT_PATH).unwrap(),
            size: sidecar.len() as u64,
            sha256: Sha256Digest::calculate(&sidecar),
            role: ArtifactFileRole::Metadata,
            component: None,
            shard: None,
        };
        let mut manifest = ArtifactManifest {
            schema_version: ARTIFACT_MANIFEST_SCHEMA_V1,
            bundle: layout.bundle.clone(),
            profile: layout.profile.clone(),
            model: layout.model.clone(),
            model_revision: layout.model_revision.clone(),
            numeric_format: NumericFormat::F16,
            components: Vec::new(),
            files: logical_files.chain([sidecar_file]).collect(),
            dependencies: Vec::new(),
            metadata: transport_metadata(),
            content_digest: None,
        };
        manifest.seal().unwrap();
        (manifest, sidecar)
    }

    fn small_fixture() -> (ArtifactTransportLayout, ArtifactManifest, Vec<u8>, Vec<u8>) {
        let bytes = b"one logical burnpack transported in one part".to_vec();
        let digest = Sha256Digest::calculate(&bytes);
        let object = ArtifactTransportObject {
            path: ArtifactPath::new("objects/logical.bpk").unwrap(),
            size: bytes.len() as u64,
            sha256: digest,
            parts: vec![ArtifactTransportPart {
                path: ArtifactPath::new(format!("transport/{digest}.part")).unwrap(),
                offset: 0,
                size: bytes.len() as u64,
                sha256: digest,
            }],
        };
        let layout = layout_for_object(object);
        let (manifest, sidecar) = manifest_for_layout(&layout);
        (layout, manifest, sidecar, bytes)
    }

    #[test]
    fn sealed_transport_layout_parses_to_verified_typestate_correctness() {
        let (layout, manifest, sidecar, _) = small_fixture();
        let verified = ArtifactTransportLayout::parse_and_validate(&manifest, &sidecar).unwrap();
        assert_eq!(verified.layout(), &layout);
        assert_eq!(verified.objects().len(), 1);
        assert_eq!(
            verified.object(&layout.objects[0].path),
            Some(&layout.objects[0])
        );
        assert_eq!(
            verified.manifest_content_digest(),
            manifest.content_digest.unwrap()
        );
    }

    #[test]
    fn transport_layout_rejects_unknown_json_fields_correctness() {
        let (layout, mut manifest, _, _) = small_fixture();
        let mut value = serde_json::to_value(layout).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("unexpected".into(), json!(true));
        let sidecar = serde_json::to_vec(&value).unwrap();
        let declared = manifest
            .files
            .iter_mut()
            .find(|file| file.path.as_str() == ARTIFACT_TRANSPORT_LAYOUT_PATH)
            .unwrap();
        declared.size = sidecar.len() as u64;
        declared.sha256 = Sha256Digest::calculate(&sidecar);
        manifest.seal().unwrap();
        let error = ArtifactTransportLayout::parse_and_validate(&manifest, &sidecar).unwrap_err();
        assert!(matches!(
            error,
            ArtifactTransportLayoutError::InvalidJson(_)
        ));
    }

    #[test]
    fn transport_layout_rejects_sidecar_digest_tampering_correctness() {
        let (_, manifest, mut sidecar, _) = small_fixture();
        sidecar[0] ^= 1;
        let error = ArtifactTransportLayout::parse_and_validate(&manifest, &sidecar).unwrap_err();
        assert!(matches!(error, ArtifactTransportLayoutError::Integrity(_)));
    }

    #[test]
    fn transport_declaration_requires_complete_exact_metadata_correctness() {
        let (_, mut manifest, _, _) = small_fixture();
        manifest
            .metadata
            .remove(ARTIFACT_TRANSPORT_PARTS_REQUIRED_KEY);
        manifest.seal().unwrap();
        assert!(matches!(
            ArtifactTransportLayout::declared_file(&manifest),
            Err(ArtifactTransportLayoutError::MetadataMismatch { .. })
        ));

        let (_, mut manifest, _, _) = small_fixture();
        manifest
            .metadata
            .insert(ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES_KEY.into(), "1".into());
        manifest.metadata.insert(
            ARTIFACT_TARGET_MAX_SEMANTIC_SHARD_BYTES_KEY.into(),
            "1".into(),
        );
        manifest.seal().unwrap();
        assert!(matches!(
            ArtifactTransportLayout::declared_file(&manifest),
            Err(ArtifactTransportLayoutError::MetadataMismatch { .. })
        ));
    }

    #[test]
    fn transport_metadata_without_sidecar_fails_closed_correctness() {
        let (_, mut manifest, _, _) = small_fixture();
        manifest
            .files
            .retain(|file| file.path.as_str() != ARTIFACT_TRANSPORT_LAYOUT_PATH);
        manifest.seal().unwrap();
        assert!(matches!(
            ArtifactTransportLayout::declared_file(&manifest),
            Err(ArtifactTransportLayoutError::InvalidDeclaration(_))
        ));
    }

    #[test]
    fn transport_declaration_rejects_every_oversized_direct_file_role_correctness() {
        for (role, suffix) in [
            (ArtifactFileRole::Config, "config"),
            (ArtifactFileRole::Tokenizer, "tokenizer"),
            (ArtifactFileRole::Metadata, "metadata"),
            (ArtifactFileRole::Other, "other"),
        ] {
            let (_, mut manifest, _, _) = small_fixture();
            let path = ArtifactPath::new(format!("metadata/oversized-{suffix}.bin")).unwrap();
            manifest.files.push(ArtifactFile {
                path: path.clone(),
                size: ARTIFACT_TRANSPORT_MAX_PART_BYTES + 1,
                sha256: Sha256Digest::calculate(suffix.as_bytes()),
                role,
                component: None,
                shard: None,
            });
            manifest.seal().unwrap();

            assert_eq!(
                ArtifactTransportLayout::declared_file(&manifest).unwrap_err(),
                ArtifactTransportLayoutError::DirectFileSizeOutOfBounds {
                    path,
                    size: ARTIFACT_TRANSPORT_MAX_PART_BYTES + 1,
                    maximum: ARTIFACT_TRANSPORT_MAX_PART_BYTES,
                }
            );
        }
    }

    #[test]
    fn transport_declaration_accepts_direct_file_at_exact_decimal_cap_correctness() {
        let (_, mut manifest, _, _) = small_fixture();
        manifest.files.push(ArtifactFile {
            path: ArtifactPath::new("metadata/exact-cap.bin").unwrap(),
            size: ARTIFACT_TRANSPORT_MAX_PART_BYTES,
            sha256: Sha256Digest::calculate(b"exact-cap"),
            role: ArtifactFileRole::Other,
            component: None,
            shard: None,
        });
        manifest.seal().unwrap();

        assert!(
            ArtifactTransportLayout::declared_file(&manifest)
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn transport_declaration_keeps_tighter_sidecar_bootstrap_cap_correctness() {
        let (_, mut manifest, _, _) = small_fixture();
        let sidecar = manifest
            .files
            .iter_mut()
            .find(|file| file.path.as_str() == ARTIFACT_TRANSPORT_LAYOUT_PATH)
            .unwrap();
        sidecar.size = MAX_ARTIFACT_TRANSPORT_LAYOUT_BYTES + 1;
        manifest.seal().unwrap();

        let error = ArtifactTransportLayout::declared_file(&manifest).unwrap_err();
        assert!(matches!(
            error,
            ArtifactTransportLayoutError::InvalidDeclaration(message)
                if message.contains("bootstrap cap")
        ));
    }

    #[test]
    fn transport_declaration_does_not_apply_physical_cap_to_logical_weights_correctness() {
        let (_, mut manifest, _, _) = small_fixture();
        let logical = manifest
            .files
            .iter_mut()
            .find(|file| file.role == ArtifactFileRole::Weights)
            .unwrap();
        logical.size = ARTIFACT_TRANSPORT_MAX_PART_BYTES + 1;
        manifest.seal().unwrap();

        assert!(
            ArtifactTransportLayout::declared_file(&manifest)
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn transport_layout_requires_every_logical_weight_exactly_once_correctness() {
        let (mut layout, mut manifest, _, _) = small_fixture();
        let second = logical_file("objects/second.bpk", 7, Sha256Digest::calculate(b"second"));
        manifest.files.push(second.clone());
        manifest.seal().unwrap();
        assert_eq!(
            layout.validate_for_manifest(&manifest).unwrap_err(),
            ArtifactTransportLayoutError::MissingLogicalObject {
                path: second.path.clone()
            }
        );

        layout.objects.push(layout.objects[0].clone());
        assert!(matches!(
            layout.validate_for_manifest(&manifest),
            Err(ArtifactTransportLayoutError::DuplicateLogicalObject { .. })
        ));
    }

    #[test]
    fn transport_layout_rejects_unknown_and_unsorted_logical_objects_correctness() {
        let (layout, manifest, _, _) = small_fixture();
        let mut unknown = layout.clone();
        unknown.objects[0].path = ArtifactPath::new("objects/unknown.bpk").unwrap();
        assert!(matches!(
            unknown.validate_for_manifest(&manifest),
            Err(ArtifactTransportLayoutError::UnknownLogicalObject { .. })
        ));

        let mut unsorted = layout;
        let mut earlier = unsorted.objects[0].clone();
        earlier.path = ArtifactPath::new("objects/a.bpk").unwrap();
        unsorted.objects.push(earlier);
        assert!(matches!(
            unsorted.validate_for_manifest(&manifest),
            Err(ArtifactTransportLayoutError::ObjectsNotSorted { .. })
        ));
    }

    #[test]
    fn transport_layout_enforces_contiguous_deterministic_parts_correctness() {
        let (layout, manifest, _, _) = small_fixture();
        let mut gap = layout.clone();
        gap.objects[0].parts[0].offset = 1;
        assert!(matches!(
            gap.validate_for_manifest(&manifest),
            Err(ArtifactTransportLayoutError::PartOffsetMismatch { .. })
        ));

        let total = ARTIFACT_TRANSPORT_TARGET_PART_BYTES + 8;
        let logical_digest = Sha256Digest::calculate(b"logical identity");
        let mut multi = layout_for_object(ArtifactTransportObject {
            path: ArtifactPath::new("objects/multi.bpk").unwrap(),
            size: total,
            sha256: logical_digest,
            parts: vec![
                part(ARTIFACT_TRANSPORT_TARGET_PART_BYTES, 0, b"first"),
                part(8, ARTIFACT_TRANSPORT_TARGET_PART_BYTES, b"last"),
            ],
        });
        let (multi_manifest, _) = manifest_for_layout(&multi);
        multi.validate_for_manifest(&multi_manifest).unwrap();
        multi.objects[0].parts[0].size -= 1;
        multi.objects[0].parts[1].offset -= 1;
        multi.objects[0].parts[1].size += 1;
        assert!(matches!(
            multi.validate_for_manifest(&multi_manifest),
            Err(ArtifactTransportLayoutError::NonFinalPartSize { .. })
        ));
    }

    #[test]
    fn transport_layout_enforces_target_and_decimal_hard_ceiling_correctness() {
        for size in [
            ARTIFACT_TRANSPORT_TARGET_PART_BYTES,
            25_000_000,
            25_000_001,
            25 * 1024 * 1024,
        ] {
            let logical_digest = Sha256Digest::calculate(format!("logical-{size}").as_bytes());
            let layout = layout_for_object(ArtifactTransportObject {
                path: ArtifactPath::new("objects/boundary.bpk").unwrap(),
                size,
                sha256: logical_digest,
                parts: vec![part(size, 0, format!("part-{size}").as_bytes())],
            });
            let (manifest, _) = manifest_for_layout(&layout);
            let result = layout.validate_for_manifest(&manifest);
            if size == ARTIFACT_TRANSPORT_TARGET_PART_BYTES {
                result.unwrap();
            } else {
                let error = result.unwrap_err();
                assert!(matches!(
                    error,
                    ArtifactTransportLayoutError::PartSizeOutOfBounds { .. }
                ));
                if size > ARTIFACT_TRANSPORT_MAX_PART_BYTES {
                    assert!(error.to_string().contains("25000000"));
                } else {
                    assert!(error.to_string().contains("20971520"));
                }
            }
        }
    }

    #[test]
    fn transport_layout_requires_content_addressed_part_paths_correctness() {
        let (mut layout, manifest, _, _) = small_fixture();
        layout.objects[0].parts[0].path =
            ArtifactPath::new("transport/not-the-digest.part").unwrap();
        assert!(matches!(
            layout.validate_for_manifest(&manifest),
            Err(ArtifactTransportLayoutError::PartPathMismatch { .. })
        ));
    }
}
