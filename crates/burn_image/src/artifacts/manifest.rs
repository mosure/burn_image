use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    ArtifactBundleId, ArtifactComponentId, ArtifactPath, ArtifactProfileId, ManifestError, ModelId,
    NumericFormat, Sha256Digest, ValidationError,
};

/// Original, dependency-free artifact manifest schema.
pub const ARTIFACT_MANIFEST_SCHEMA_V1: u32 = 1;
/// Artifact manifest schema with immutable sibling-bundle dependencies.
pub const ARTIFACT_MANIFEST_SCHEMA_V2: u32 = 2;
/// Schema version used when constructing new artifact manifests.
pub const ARTIFACT_MANIFEST_SCHEMA_VERSION: u32 = ARTIFACT_MANIFEST_SCHEMA_V2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactFileRole {
    Config,
    Tokenizer,
    Weights,
    Metadata,
    Other,
}

impl ArtifactFileRole {
    const fn digest_tag(self) -> u8 {
        match self {
            Self::Config => 0,
            Self::Tokenizer => 1,
            Self::Weights => 2,
            Self::Metadata => 3,
            Self::Other => 4,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactComponent {
    pub id: ArtifactComponentId,
    pub required: bool,
}

/// Immutable identity of a component bundle required by a composed manifest.
///
/// Dependency bundles are resolved as sibling bundle prefixes by transport and
/// cache adapters. Keeping locations out of the sealed contract makes the same
/// manifest mirrorable across CDN roots and local cache parents.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactDependency {
    /// Role this bundle fills in the composition, for example `qwen` or `vae`.
    pub role: ArtifactComponentId,
    pub bundle: ArtifactBundleId,
    pub profile: ArtifactProfileId,
    pub model: ModelId,
    pub model_revision: String,
    pub content_digest: Sha256Digest,
}

impl ArtifactDependency {
    pub fn validate(&self) -> Result<(), ManifestError> {
        if self.model_revision.trim().is_empty() {
            return Err(ValidationError::Empty {
                field: "artifact.dependency.model_revision",
            }
            .into());
        }
        Ok(())
    }

    /// Validate a resolved sibling manifest against every sealed identity
    /// field pinned by this dependency.
    pub fn validate_resolved_manifest(
        &self,
        manifest: &ArtifactManifest,
    ) -> Result<(), ManifestError> {
        self.validate()?;
        manifest.validate_sealed()?;
        validate_dependency_field(
            &self.role,
            "bundle",
            self.bundle.as_str(),
            manifest.bundle.as_str(),
        )?;
        validate_dependency_field(
            &self.role,
            "profile",
            self.profile.as_str(),
            manifest.profile.as_str(),
        )?;
        validate_dependency_field(
            &self.role,
            "model",
            self.model.as_str(),
            manifest.model.as_str(),
        )?;
        validate_dependency_field(
            &self.role,
            "model_revision",
            &self.model_revision,
            &manifest.model_revision,
        )?;
        let actual = manifest
            .content_digest
            .expect("validate_sealed requires a content digest");
        if actual != self.content_digest {
            return Err(ManifestError::DependencyContentDigestMismatch {
                role: self.role.to_string(),
                expected: self.content_digest,
                actual,
            });
        }
        Ok(())
    }
}

/// A dependency/manifest pair whose complete sealed identity has been checked.
#[derive(Clone, Copy, Debug)]
pub struct ResolvedArtifactDependency<'a> {
    dependency: &'a ArtifactDependency,
    manifest: &'a ArtifactManifest,
}

impl<'a> ResolvedArtifactDependency<'a> {
    pub fn new(
        dependency: &'a ArtifactDependency,
        manifest: &'a ArtifactManifest,
    ) -> Result<Self, ManifestError> {
        dependency.validate_resolved_manifest(manifest)?;
        Ok(Self {
            dependency,
            manifest,
        })
    }

    pub fn dependency(self) -> &'a ArtifactDependency {
        self.dependency
    }

    pub fn manifest(self) -> &'a ArtifactManifest {
        self.manifest
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactShard {
    pub index: u32,
    pub count: u32,
    /// Hash-chain state through this shard. [`ArtifactManifest::seal`] fills
    /// this field deterministically in shard-index order.
    pub chain_sha256: Option<Sha256Digest>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactFile {
    pub path: ArtifactPath,
    pub size: u64,
    pub sha256: Sha256Digest,
    pub role: ArtifactFileRole,
    pub component: Option<ArtifactComponentId>,
    pub shard: Option<ArtifactShard>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactManifest {
    pub schema_version: u32,
    pub bundle: ArtifactBundleId,
    pub profile: ArtifactProfileId,
    pub model: ModelId,
    pub model_revision: String,
    pub numeric_format: NumericFormat,
    pub components: Vec<ArtifactComponent>,
    pub files: Vec<ArtifactFile>,
    /// Immutable sibling bundles used by a composed schema-v2 manifest.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dependencies: Vec<ArtifactDependency>,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
    pub content_digest: Option<Sha256Digest>,
}

impl ArtifactManifest {
    pub fn validate(&self) -> Result<(), ManifestError> {
        if !matches!(
            self.schema_version,
            ARTIFACT_MANIFEST_SCHEMA_V1 | ARTIFACT_MANIFEST_SCHEMA_V2
        ) {
            return Err(ManifestError::UnsupportedSchema {
                expected: ARTIFACT_MANIFEST_SCHEMA_VERSION,
                actual: self.schema_version,
            });
        }
        if self.schema_version == ARTIFACT_MANIFEST_SCHEMA_V1 && !self.dependencies.is_empty() {
            return Err(ManifestError::DependenciesRequireSchemaV2);
        }
        if self.model_revision.trim().is_empty() {
            return Err(ValidationError::Empty {
                field: "artifact.model_revision",
            }
            .into());
        }
        self.numeric_format.validate()?;
        if self.files.is_empty() {
            return Err(ManifestError::EmptyFiles);
        }

        let mut dependency_roles = BTreeSet::new();
        let mut dependency_bundles = BTreeSet::new();
        for dependency in &self.dependencies {
            dependency.validate()?;
            if dependency.bundle == self.bundle {
                return Err(ManifestError::SelfDependency {
                    bundle: self.bundle.to_string(),
                });
            }
            if !dependency_roles.insert(dependency.role.clone()) {
                return Err(ManifestError::DuplicateDependencyRole {
                    role: dependency.role.to_string(),
                });
            }
            if !dependency_bundles.insert(dependency.bundle.clone()) {
                return Err(ManifestError::DuplicateDependencyBundle {
                    bundle: dependency.bundle.to_string(),
                });
            }
        }

        let mut component_ids = BTreeSet::new();
        for component in &self.components {
            if !component_ids.insert(component.id.clone()) {
                return Err(ManifestError::DuplicateComponent {
                    component: component.id.to_string(),
                });
            }
        }

        let mut paths = BTreeSet::new();
        let mut present_components = BTreeSet::new();
        let mut layouts: BTreeMap<ArtifactComponentId, ComponentLayout> = BTreeMap::new();
        for file in &self.files {
            if !paths.insert(file.path.clone()) {
                return Err(ManifestError::DuplicatePath(file.path.clone()));
            }
            if file.size == 0 {
                return Err(ManifestError::ZeroLength {
                    path: file.path.clone(),
                });
            }
            if let Some(component) = file.component.as_ref() {
                if !component_ids.contains(component) {
                    return Err(ManifestError::UnknownComponent {
                        component: component.to_string(),
                        path: file.path.clone(),
                    });
                }
                present_components.insert(component.clone());
            }
            if let Some(shard) = file.shard {
                let component =
                    file.component
                        .as_ref()
                        .ok_or_else(|| ManifestError::InvalidShard {
                            path: file.path.clone(),
                            reason: "shards require a component".to_string(),
                        })?;
                if file.role != ArtifactFileRole::Weights {
                    return Err(ManifestError::InvalidShard {
                        path: file.path.clone(),
                        reason: "only weight artifacts may be sharded".to_string(),
                    });
                }
                if shard.count < 2 || shard.index >= shard.count {
                    return Err(ManifestError::InvalidShard {
                        path: file.path.clone(),
                        reason: format!(
                            "index {} must be below count {}, and count must be at least 2",
                            shard.index, shard.count
                        ),
                    });
                }
                layouts
                    .entry(component.clone())
                    .or_default()
                    .add_shard(component, shard)?;
            } else if file.role == ArtifactFileRole::Weights
                && let Some(component) = file.component.as_ref()
            {
                layouts.entry(component.clone()).or_default().unsharded += 1;
            }
        }

        for component in &self.components {
            if component.required && !present_components.contains(&component.id) {
                return Err(ManifestError::MissingComponent {
                    component: component.id.to_string(),
                });
            }
        }
        for (component, layout) in layouts {
            layout.validate(&component)?;
        }
        self.validate_hash_chains(false)?;
        for key in self.metadata.keys() {
            if key.trim().is_empty() {
                return Err(ValidationError::Empty {
                    field: "artifact.metadata.key",
                }
                .into());
            }
        }

        if let Some(expected) = self.content_digest {
            let actual = self.calculate_content_digest();
            if actual != expected {
                return Err(ManifestError::ContentDigestMismatch { expected, actual });
            }
        }
        Ok(())
    }

    /// Validate a deployment-ready manifest. Unlike [`Self::validate`], this
    /// requires both the bundle digest and every shard hash-chain entry.
    pub fn validate_sealed(&self) -> Result<(), ManifestError> {
        self.validate()?;
        if self.content_digest.is_none() {
            return Err(ManifestError::MissingContentDigest);
        }
        self.validate_hash_chains(true)
    }

    /// Validate this sealed manifest and its complete transitive dependency
    /// closure. The resolver supplies sibling manifests by immutable bundle id.
    /// Missing nodes, identity drift, and dependency cycles are rejected.
    pub fn validate_dependency_closure<'a, F>(&'a self, mut resolve: F) -> Result<(), ManifestError>
    where
        F: FnMut(&ArtifactBundleId) -> Option<&'a ArtifactManifest>,
    {
        let mut visiting = Vec::new();
        let mut validated = BTreeMap::new();
        validate_dependency_node(self, &mut resolve, &mut visiting, &mut validated)
    }

    /// Deterministic digest over artifact identity, components, file metadata,
    /// and user metadata. File declaration order does not affect the result.
    pub fn calculate_content_digest(&self) -> Sha256Digest {
        let mut hasher = Sha256::new();
        match self.schema_version {
            ARTIFACT_MANIFEST_SCHEMA_V1 => {
                hasher.update(b"burn_image.artifact_manifest.v1\0");
            }
            _ => {
                // Unknown schemas cannot be sealed, but still receive a
                // domain-separated digest for diagnostic callers.
                hasher.update(b"burn_image.artifact_manifest.v2\0");
            }
        }
        update_u32(&mut hasher, self.schema_version);
        update_str(&mut hasher, self.bundle.as_str());
        update_str(&mut hasher, self.profile.as_str());
        update_str(&mut hasher, self.model.as_str());
        update_str(&mut hasher, &self.model_revision);
        update_str(
            &mut hasher,
            &serde_numeric_format_name(&self.numeric_format),
        );

        let mut components = self.components.iter().collect::<Vec<_>>();
        components.sort_by_key(|component| &component.id);
        update_u64(&mut hasher, components.len() as u64);
        for component in components {
            update_str(&mut hasher, component.id.as_str());
            hasher.update([u8::from(component.required)]);
        }

        if self.schema_version >= ARTIFACT_MANIFEST_SCHEMA_V2 {
            let mut dependencies = self.dependencies.iter().collect::<Vec<_>>();
            dependencies.sort_by(|left, right| {
                left.role
                    .cmp(&right.role)
                    .then_with(|| left.bundle.cmp(&right.bundle))
                    .then_with(|| left.profile.cmp(&right.profile))
                    .then_with(|| left.model.cmp(&right.model))
                    .then_with(|| left.model_revision.cmp(&right.model_revision))
                    .then_with(|| left.content_digest.cmp(&right.content_digest))
            });
            update_u64(&mut hasher, dependencies.len() as u64);
            for dependency in dependencies {
                update_str(&mut hasher, dependency.role.as_str());
                update_str(&mut hasher, dependency.bundle.as_str());
                update_str(&mut hasher, dependency.profile.as_str());
                update_str(&mut hasher, dependency.model.as_str());
                update_str(&mut hasher, &dependency.model_revision);
                hasher.update(dependency.content_digest.as_bytes());
            }
        }

        let mut files = self.files.iter().collect::<Vec<_>>();
        files.sort_by_key(|file| &file.path);
        update_u64(&mut hasher, files.len() as u64);
        for file in files {
            update_str(&mut hasher, file.path.as_str());
            update_u64(&mut hasher, file.size);
            hasher.update(file.sha256.as_bytes());
            hasher.update([file.role.digest_tag()]);
            update_optional_str(
                &mut hasher,
                file.component.as_ref().map(ArtifactComponentId::as_str),
            );
            match file.shard {
                Some(shard) => {
                    hasher.update([1]);
                    update_u32(&mut hasher, shard.index);
                    update_u32(&mut hasher, shard.count);
                    match shard.chain_sha256 {
                        Some(chain) => {
                            hasher.update([1]);
                            hasher.update(chain.as_bytes());
                        }
                        None => hasher.update([0]),
                    }
                }
                None => hasher.update([0]),
            }
        }

        update_u64(&mut hasher, self.metadata.len() as u64);
        for (key, value) in &self.metadata {
            update_str(&mut hasher, key);
            update_str(&mut hasher, value);
        }
        Sha256Digest::from_bytes(hasher.finalize().into())
    }

    pub fn seal(&mut self) -> Result<Sha256Digest, ManifestError> {
        self.content_digest = None;
        let chains = self.calculate_shard_hash_chains();
        for file in &mut self.files {
            if let Some(shard) = file.shard.as_mut() {
                shard.chain_sha256 = chains.get(&file.path).copied();
            }
        }
        self.validate()?;
        let digest = self.calculate_content_digest();
        self.content_digest = Some(digest);
        Ok(digest)
    }

    pub fn file(&self, path: &ArtifactPath) -> Option<&ArtifactFile> {
        self.files.iter().find(|file| &file.path == path)
    }

    /// Calculate the chain state for every shard. The result is keyed by path,
    /// so callers and loaders may process files in any declaration order.
    pub fn calculate_shard_hash_chains(&self) -> BTreeMap<ArtifactPath, Sha256Digest> {
        let mut groups: BTreeMap<ArtifactComponentId, Vec<&ArtifactFile>> = BTreeMap::new();
        for file in &self.files {
            if file.shard.is_some()
                && let Some(component) = file.component.as_ref()
            {
                groups.entry(component.clone()).or_default().push(file);
            }
        }

        let mut output = BTreeMap::new();
        for (component, mut files) in groups {
            files.sort_by_key(|file| file.shard.map(|shard| shard.index).unwrap_or(u32::MAX));
            let mut previous: Option<Sha256Digest> = None;
            for file in files {
                let shard = file.shard.expect("group only contains shards");
                let mut hasher = Sha256::new();
                hasher.update(b"burn_image.artifact_shard_chain.v1\0");
                update_str(&mut hasher, component.as_str());
                match previous {
                    Some(previous_digest) => {
                        hasher.update([1]);
                        hasher.update(previous_digest.as_bytes());
                    }
                    None => hasher.update([0]),
                }
                update_u32(&mut hasher, shard.index);
                update_u32(&mut hasher, shard.count);
                update_str(&mut hasher, file.path.as_str());
                update_u64(&mut hasher, file.size);
                hasher.update(file.sha256.as_bytes());
                let digest = Sha256Digest::from_bytes(hasher.finalize().into());
                output.insert(file.path.clone(), digest);
                previous = Some(digest);
            }
        }
        output
    }

    fn validate_hash_chains(&self, require: bool) -> Result<(), ManifestError> {
        let expected = self.calculate_shard_hash_chains();
        let any_declared = self
            .files
            .iter()
            .filter_map(|file| file.shard)
            .any(|shard| shard.chain_sha256.is_some());
        if !require && !any_declared {
            return Ok(());
        }
        for file in self.files.iter().filter(|file| file.shard.is_some()) {
            let actual = file
                .shard
                .and_then(|shard| shard.chain_sha256)
                .ok_or_else(|| ManifestError::MissingHashChain {
                    path: file.path.clone(),
                })?;
            let expected = expected[&file.path];
            if actual != expected {
                return Err(ManifestError::HashChainMismatch {
                    path: file.path.clone(),
                    expected,
                    actual,
                });
            }
        }
        Ok(())
    }
}

fn validate_dependency_field(
    role: &ArtifactComponentId,
    field: &'static str,
    expected: &str,
    actual: &str,
) -> Result<(), ManifestError> {
    if expected != actual {
        return Err(ManifestError::DependencyIdentityMismatch {
            role: role.to_string(),
            field,
            expected: expected.to_string(),
            actual: actual.to_string(),
        });
    }
    Ok(())
}

fn validate_dependency_node<'a, F>(
    manifest: &'a ArtifactManifest,
    resolve: &mut F,
    visiting: &mut Vec<ArtifactBundleId>,
    validated: &mut BTreeMap<ArtifactBundleId, Sha256Digest>,
) -> Result<(), ManifestError>
where
    F: FnMut(&ArtifactBundleId) -> Option<&'a ArtifactManifest>,
{
    manifest.validate_sealed()?;
    let manifest_digest = manifest
        .content_digest
        .expect("validate_sealed requires a content digest");
    if let Some(expected) = validated.get(&manifest.bundle) {
        if *expected != manifest_digest {
            return Err(ManifestError::DependencyBundleConflict {
                bundle: manifest.bundle.to_string(),
                expected: *expected,
                actual: manifest_digest,
            });
        }
        return Ok(());
    }

    visiting.push(manifest.bundle.clone());
    for dependency in &manifest.dependencies {
        let resolved = resolve(&dependency.bundle).ok_or_else(|| {
            ManifestError::MissingResolvedDependency {
                role: dependency.role.to_string(),
                bundle: dependency.bundle.to_string(),
            }
        })?;
        // Establish the graph edge before validating its cryptographic pin so
        // a topological cycle is reported as a cycle even though circular
        // content-digest fixed points cannot practically be constructed.
        validate_dependency_field(
            &dependency.role,
            "bundle",
            dependency.bundle.as_str(),
            resolved.bundle.as_str(),
        )?;

        if let Some(cycle_start) = visiting
            .iter()
            .position(|bundle| bundle == &resolved.bundle)
        {
            let mut cycle = visiting[cycle_start..]
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>();
            cycle.push(resolved.bundle.to_string());
            return Err(ManifestError::DependencyCycle { cycle });
        }
        dependency.validate_resolved_manifest(resolved)?;
        validate_dependency_node(resolved, resolve, visiting, validated)?;
    }
    let completed = visiting
        .pop()
        .expect("dependency traversal always has a current node");
    validated.insert(completed, manifest_digest);
    Ok(())
}

#[derive(Default)]
struct ComponentLayout {
    unsharded: usize,
    shard_count: Option<u32>,
    shard_indices: BTreeSet<u32>,
}

impl ComponentLayout {
    fn add_shard(
        &mut self,
        component: &ArtifactComponentId,
        shard: ArtifactShard,
    ) -> Result<(), ManifestError> {
        if let Some(expected) = self.shard_count
            && expected != shard.count
        {
            return Err(ManifestError::InconsistentShardCount {
                component: component.to_string(),
                expected,
                actual: shard.count,
            });
        }
        self.shard_count = Some(shard.count);
        if !self.shard_indices.insert(shard.index) {
            return Err(ManifestError::DuplicateShard {
                component: component.to_string(),
                index: shard.index,
            });
        }
        Ok(())
    }

    fn validate(self, component: &ArtifactComponentId) -> Result<(), ManifestError> {
        if self.unsharded > 0 && self.shard_count.is_some() {
            return Err(ManifestError::MixedShardLayout {
                component: component.to_string(),
            });
        }
        if let Some(count) = self.shard_count {
            for index in 0..count {
                if !self.shard_indices.contains(&index) {
                    return Err(ManifestError::MissingShard {
                        component: component.to_string(),
                        index,
                        count,
                    });
                }
            }
        }
        Ok(())
    }
}

fn update_str(hasher: &mut Sha256, value: &str) {
    update_u64(hasher, value.len() as u64);
    hasher.update(value.as_bytes());
}

fn update_optional_str(hasher: &mut Sha256, value: Option<&str>) {
    match value {
        Some(value) => {
            hasher.update([1]);
            update_str(hasher, value);
        }
        None => hasher.update([0]),
    }
}

fn update_u32(hasher: &mut Sha256, value: u32) {
    hasher.update(value.to_le_bytes());
}

fn update_u64(hasher: &mut Sha256, value: u64) {
    hasher.update(value.to_le_bytes());
}

fn serde_numeric_format_name(format: &NumericFormat) -> String {
    match format {
        NumericFormat::F32 => "f32".to_string(),
        NumericFormat::F16 => "f16".to_string(),
        NumericFormat::Bf16 => "bf16".to_string(),
        NumericFormat::I8 => "i8".to_string(),
        NumericFormat::U8 => "u8".to_string(),
        NumericFormat::Other(name) => format!("other:{name}"),
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, collections::BTreeMap};

    use crate::{
        ARTIFACT_MANIFEST_SCHEMA_V1, ARTIFACT_MANIFEST_SCHEMA_VERSION, ArtifactBundleId,
        ArtifactComponent, ArtifactComponentId, ArtifactDependency, ArtifactFile, ArtifactFileRole,
        ArtifactManifest, ArtifactPath, ArtifactProfileId, ArtifactShard, ManifestError, ModelId,
        NumericFormat, ResolvedArtifactDependency, Sha256Digest,
    };

    fn manifest() -> ArtifactManifest {
        let component = ArtifactComponentId::new("transformer").unwrap();
        let files = (0..2)
            .map(|index| ArtifactFile {
                path: ArtifactPath::new(format!("transformer/model.bpk.part-{index:03}")).unwrap(),
                size: 4,
                sha256: Sha256Digest::calculate(&[index as u8; 4]),
                role: ArtifactFileRole::Weights,
                component: Some(component.clone()),
                shard: Some(ArtifactShard {
                    index,
                    count: 2,
                    chain_sha256: None,
                }),
            })
            .collect();
        ArtifactManifest {
            schema_version: ARTIFACT_MANIFEST_SCHEMA_VERSION,
            bundle: ArtifactBundleId::new("example-bundle").unwrap(),
            profile: ArtifactProfileId::new("f16-web").unwrap(),
            model: ModelId::new("owner/model").unwrap(),
            model_revision: "0123456789abcdef".to_string(),
            numeric_format: NumericFormat::F16,
            components: vec![ArtifactComponent {
                id: component,
                required: true,
            }],
            files,
            dependencies: Vec::new(),
            metadata: BTreeMap::new(),
            content_digest: None,
        }
    }

    fn sealed_manifest(bundle: &str) -> ArtifactManifest {
        let mut manifest = manifest();
        manifest.bundle = ArtifactBundleId::new(bundle).unwrap();
        manifest.model = ModelId::new(format!("owner/{bundle}")).unwrap();
        manifest.seal().unwrap();
        manifest
    }

    fn dependency(role: &str, manifest: &ArtifactManifest) -> ArtifactDependency {
        ArtifactDependency {
            role: ArtifactComponentId::new(role).unwrap(),
            bundle: manifest.bundle.clone(),
            profile: manifest.profile.clone(),
            model: manifest.model.clone(),
            model_revision: manifest.model_revision.clone(),
            content_digest: manifest.content_digest.unwrap(),
        }
    }

    #[test]
    fn manifest_seal_is_order_independent_and_detects_mutation_correctness() {
        let mut manifest = manifest();
        let digest = manifest.seal().unwrap();
        assert!(
            manifest
                .files
                .iter()
                .all(|file| file.shard.is_none_or(|shard| shard.chain_sha256.is_some()))
        );
        assert!(manifest.validate_sealed().is_ok());
        manifest.files.reverse();
        assert_eq!(manifest.calculate_content_digest(), digest);
        assert!(manifest.validate().is_ok());

        manifest.files[0].size += 1;
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn manifest_rejects_tampered_hash_chain_correctness() {
        let mut manifest = manifest();
        manifest.seal().unwrap();
        manifest.files[1].shard.as_mut().unwrap().chain_sha256 =
            Some(Sha256Digest::calculate(b"tampered"));
        manifest.content_digest = None;
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn manifest_rejects_missing_shard_correctness() {
        let mut manifest = manifest();
        manifest.files.pop();
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn manifest_v1_component_digest_and_json_are_canonical_correctness() {
        let mut manifest = manifest();
        manifest.schema_version = ARTIFACT_MANIFEST_SCHEMA_V1;
        let digest = manifest.seal().unwrap();
        assert_eq!(
            digest.to_string(),
            "aaaaecdde41723f2203b30a40fb2ba6ee46b8c390b834e1d32c5746a8259abff"
        );

        let json = serde_json::to_value(&manifest).unwrap();
        assert!(json.get("dependencies").is_none());
        let decoded: ArtifactManifest = serde_json::from_value(json).unwrap();
        assert!(decoded.dependencies.is_empty());
        decoded.validate_sealed().unwrap();
        assert_eq!(decoded.calculate_content_digest(), digest);
    }

    #[test]
    fn manifest_v2_dependency_digest_is_order_independent_and_bound_correctness() {
        let qwen = sealed_manifest("qwen-bundle");
        let vae = sealed_manifest("vae-bundle");
        let mut root = manifest();
        root.dependencies = vec![dependency("qwen", &qwen), dependency("vae", &vae)];
        let digest = root.seal().unwrap();

        root.dependencies.reverse();
        assert_eq!(root.calculate_content_digest(), digest);
        root.dependencies[0].model_revision.push_str("-tampered");
        assert_ne!(root.calculate_content_digest(), digest);
        assert!(matches!(
            root.validate(),
            Err(ManifestError::ContentDigestMismatch { .. })
        ));
    }

    #[test]
    fn manifest_dependencies_fail_closed_on_invalid_declarations_correctness() {
        let qwen = sealed_manifest("qwen-bundle");
        let vae = sealed_manifest("vae-bundle");

        let mut root = manifest();
        root.schema_version = ARTIFACT_MANIFEST_SCHEMA_V1;
        root.dependencies = vec![dependency("qwen", &qwen)];
        assert_eq!(
            root.validate(),
            Err(ManifestError::DependenciesRequireSchemaV2)
        );

        root.schema_version = ARTIFACT_MANIFEST_SCHEMA_VERSION;
        root.dependencies = vec![dependency("qwen", &qwen), dependency("qwen", &vae)];
        assert!(matches!(
            root.validate(),
            Err(ManifestError::DuplicateDependencyRole { .. })
        ));

        let mut second_qwen = dependency("vision", &qwen);
        second_qwen.role = ArtifactComponentId::new("vision").unwrap();
        root.dependencies = vec![dependency("qwen", &qwen), second_qwen];
        assert!(matches!(
            root.validate(),
            Err(ManifestError::DuplicateDependencyBundle { .. })
        ));

        let mut self_dependency = dependency("pipeline", &qwen);
        self_dependency.bundle = root.bundle.clone();
        root.dependencies = vec![self_dependency];
        assert!(matches!(
            root.validate(),
            Err(ManifestError::SelfDependency { .. })
        ));

        let mut empty_revision = dependency("qwen", &qwen);
        empty_revision.model_revision = "  ".to_string();
        root.dependencies = vec![empty_revision];
        assert!(root.validate().is_err());
    }

    #[test]
    fn dependency_serde_rejects_missing_invalid_and_unknown_fields_correctness() {
        let qwen = sealed_manifest("qwen-bundle");
        let dependency = dependency("qwen", &qwen);

        let mut missing = serde_json::to_value(&dependency).unwrap();
        missing.as_object_mut().unwrap().remove("profile");
        assert!(serde_json::from_value::<ArtifactDependency>(missing).is_err());

        let mut invalid = serde_json::to_value(&dependency).unwrap();
        invalid["role"] = serde_json::Value::String("bad role".to_string());
        assert!(serde_json::from_value::<ArtifactDependency>(invalid).is_err());

        let mut unknown = serde_json::to_value(&dependency).unwrap();
        unknown["base_url"] = serde_json::Value::String("https://mutable.example".to_string());
        assert!(serde_json::from_value::<ArtifactDependency>(unknown).is_err());
    }

    #[test]
    fn resolved_dependency_requires_exact_sealed_identity_correctness() {
        let qwen = sealed_manifest("qwen-bundle");
        let dependency = dependency("qwen", &qwen);
        let resolved = ResolvedArtifactDependency::new(&dependency, &qwen).unwrap();
        assert_eq!(resolved.dependency(), &dependency);
        assert_eq!(resolved.manifest(), &qwen);

        let mut wrong_profile = qwen.clone();
        wrong_profile.profile = ArtifactProfileId::new("bf16").unwrap();
        wrong_profile.content_digest = None;
        wrong_profile.seal().unwrap();
        assert!(matches!(
            dependency.validate_resolved_manifest(&wrong_profile),
            Err(ManifestError::DependencyIdentityMismatch {
                field: "profile",
                ..
            })
        ));

        let mut wrong_content = qwen.clone();
        wrong_content.files[0].sha256 = Sha256Digest::calculate(b"different payload");
        wrong_content.content_digest = None;
        wrong_content.seal().unwrap();
        assert!(matches!(
            dependency.validate_resolved_manifest(&wrong_content),
            Err(ManifestError::DependencyContentDigestMismatch { .. })
        ));
    }

    #[test]
    fn dependency_closure_validates_transitive_nodes_and_missing_nodes_correctness() {
        let qwen = sealed_manifest("qwen-bundle");
        let vae = sealed_manifest("vae-bundle");
        let mut codec = sealed_manifest("codec-bundle");
        codec.dependencies = vec![dependency("vae", &vae)];
        codec.content_digest = None;
        codec.seal().unwrap();
        let mut root = sealed_manifest("pipeline-bundle");
        root.dependencies = vec![dependency("qwen", &qwen), dependency("codec", &codec)];
        root.content_digest = None;
        root.seal().unwrap();

        let closure = BTreeMap::from([
            (qwen.bundle.clone(), &qwen),
            (codec.bundle.clone(), &codec),
            (vae.bundle.clone(), &vae),
        ]);
        root.validate_dependency_closure(|bundle| closure.get(bundle).copied())
            .unwrap();

        let incomplete = BTreeMap::from([(qwen.bundle.clone(), &qwen)]);
        assert!(matches!(
            root.validate_dependency_closure(|bundle| incomplete.get(bundle).copied()),
            Err(ManifestError::MissingResolvedDependency { .. })
        ));
    }

    #[test]
    fn dependency_closure_rejects_cycles_before_stale_cycle_pins_correctness() {
        let mut first = sealed_manifest("first-bundle");
        let mut second = sealed_manifest("second-bundle");

        second.dependencies = vec![dependency("first", &first)];
        second.content_digest = None;
        second.seal().unwrap();
        first.dependencies = vec![dependency("second", &second)];
        first.content_digest = None;
        first.seal().unwrap();

        let closure = BTreeMap::from([
            (first.bundle.clone(), &first),
            (second.bundle.clone(), &second),
        ]);
        assert!(matches!(
            first.validate_dependency_closure(|bundle| closure.get(bundle).copied()),
            Err(ManifestError::DependencyCycle { .. })
        ));
    }

    #[test]
    fn dependency_closure_rejects_equivocating_bundle_resolver_correctness() {
        let shared_a = sealed_manifest("shared-bundle");
        let mut shared_b = shared_a.clone();
        shared_b.files[0].sha256 = Sha256Digest::calculate(b"different shared payload");
        shared_b.content_digest = None;
        shared_b.seal().unwrap();

        let mut left = sealed_manifest("left-bundle");
        left.dependencies = vec![dependency("shared", &shared_a)];
        left.content_digest = None;
        left.seal().unwrap();
        let mut right = sealed_manifest("right-bundle");
        right.dependencies = vec![dependency("shared", &shared_b)];
        right.content_digest = None;
        right.seal().unwrap();
        let mut root = sealed_manifest("root-bundle");
        root.dependencies = vec![dependency("left", &left), dependency("right", &right)];
        root.content_digest = None;
        root.seal().unwrap();

        let shared_resolutions = Cell::new(0);
        let result = root.validate_dependency_closure(|bundle| match bundle.as_str() {
            "left-bundle" => Some(&left),
            "right-bundle" => Some(&right),
            "shared-bundle" => {
                let index = shared_resolutions.get();
                shared_resolutions.set(index + 1);
                Some(if index == 0 { &shared_a } else { &shared_b })
            }
            _ => None,
        });
        assert!(matches!(
            result,
            Err(ManifestError::DependencyBundleConflict { .. })
        ));
    }
}
