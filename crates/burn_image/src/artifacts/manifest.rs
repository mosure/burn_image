use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    ArtifactBundleId, ArtifactComponentId, ArtifactPath, ArtifactProfileId, ManifestError, ModelId,
    NumericFormat, Sha256Digest, ValidationError,
};

pub const ARTIFACT_MANIFEST_SCHEMA_VERSION: u32 = 1;

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
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
    pub content_digest: Option<Sha256Digest>,
}

impl ArtifactManifest {
    pub fn validate(&self) -> Result<(), ManifestError> {
        if self.schema_version != ARTIFACT_MANIFEST_SCHEMA_VERSION {
            return Err(ManifestError::UnsupportedSchema {
                expected: ARTIFACT_MANIFEST_SCHEMA_VERSION,
                actual: self.schema_version,
            });
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

    /// Deterministic digest over artifact identity, components, file metadata,
    /// and user metadata. File declaration order does not affect the result.
    pub fn calculate_content_digest(&self) -> Sha256Digest {
        let mut hasher = Sha256::new();
        hasher.update(b"burn_image.artifact_manifest.v1\0");
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
    use std::collections::BTreeMap;

    use crate::{
        ARTIFACT_MANIFEST_SCHEMA_VERSION, ArtifactBundleId, ArtifactComponent, ArtifactComponentId,
        ArtifactFile, ArtifactFileRole, ArtifactManifest, ArtifactPath, ArtifactProfileId,
        ArtifactShard, ModelId, NumericFormat, Sha256Digest,
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
            metadata: BTreeMap::new(),
            content_digest: None,
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
}
