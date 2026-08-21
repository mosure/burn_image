//! Sealed, bounded Burnpack loading for standalone Qwen3-VL component bundles.
//!
//! This module owns the semantic mapping from Qwen stages to tensors. Transport and persistent
//! cache policy stay in [`burn_image`], so native and browser runtimes can share the same loader
//! without depending on an image-generation pipeline crate.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    rc::Rc,
};

use burn::{
    nn::{RmsNorm, RmsNormConfig},
    prelude::Backend,
    tensor::{
        Bytes, DType, Tensor, TensorData,
        quantization::{QuantLevel, QuantParam, QuantScheme, QuantStore, QuantValue},
    },
};
use burn_image::{
    ARTIFACT_MANIFEST_SCHEMA_V1, ArtifactBundleId, ArtifactComponentId, ArtifactDependency,
    ArtifactFile, ArtifactFileRole, ArtifactManifest, ArtifactProfileId, ArtifactReadError,
    ArtifactShardReader, AsyncArtifactShardReader, DirectoryArtifactShardReader, ModelId,
    Sha256Digest, VerifiedArtifactBytes, VerifiedArtifactDirectory,
};
use burn_store::{
    ApplyResult, BurnpackStore, ModuleAdapter, ModuleSnapshot, ModuleStore, TensorSnapshot,
    TensorSnapshotError,
};
use thiserror::Error;

use crate::{
    AsyncQwen3VlStageSource, EmbeddingRowChunk, HostRoutedEmbedding, HostRoutedF16EmbeddingState,
    Qwen3VlConfig, Qwen3VlStage, Qwen3VlStageDType, Qwen3VlStageDTypePolicy,
    Qwen3VlStageDescriptor, Qwen3VlStageSource, Qwen3VlStreamingPlan, Qwen3VlVisionPrelude,
    RowChunkSpec,
    text::Qwen3VlDecoderLayer,
    vision::{Qwen3VlVisionBlock, Qwen3VlVisionPatchMerger},
};

const QWEN_METADATA_VALUES: [(&str, &str); 17] = [
    ("component_bundle", "true"),
    ("component_kind", "qwen3-vl-base-conditioning"),
    ("artifact_layout", "semantic-burnpack-v1"),
    ("owner", "qwen3-vl"),
    ("tensor_inventory_schema", "2"),
    ("tensor_count", "749"),
    ("stored_tensor_count", "754"),
    ("tensor_inventory_entries", "755"),
    ("omitted_tensor_count", "1"),
    ("physical_shards_bounded", "true"),
    ("target_max_shard_bytes", "268435456"),
    ("transport_layout_path", "metadata/transport-layout.json"),
    ("transport_layout_schema", "1"),
    ("transport_parts_required", "true"),
    ("transport_part_target_bytes", "20971520"),
    ("target_max_transport_shard_bytes", "25000000"),
    ("semantic_object_max_bytes", "268435456"),
];

const QWEN_METADATA_PATHS: [&str; 22] = [
    "metadata/tensor-inventory.json",
    "metadata/source-files.json",
    "metadata/transport-layout.json",
    "metadata/source/mllm/chat_template.json",
    "metadata/source/mllm/config.json",
    "metadata/source/mllm/generation_config.json",
    "metadata/source/mllm/merges.txt",
    "metadata/source/mllm/model.safetensors.index.json",
    "metadata/source/mllm/preprocessor_config.json",
    "metadata/source/mllm/tokenizer.json",
    "metadata/source/mllm/tokenizer_config.json",
    "metadata/source/mllm/video_preprocessor_config.json",
    "metadata/source/mllm/vocab.json",
    "metadata/source/processor/added_tokens.json",
    "metadata/source/processor/chat_template.jinja",
    "metadata/source/processor/merges.txt",
    "metadata/source/processor/preprocessor_config.json",
    "metadata/source/processor/special_tokens_map.json",
    "metadata/source/processor/tokenizer.json",
    "metadata/source/processor/tokenizer_config.json",
    "metadata/source/processor/video_preprocessor_config.json",
    "metadata/source/processor/vocab.json",
];

/// Canonical storage profile for reusable base-conditioning weights.
///
/// Text and embedding stages are F16, vision stages are F32, and the unused LM head is absent.
pub const QWEN_BASE_CONDITIONING_PROFILE: &str = "f16-text-f32-vision-base";
/// Canonical packed-Q4S storage profile for the reusable base-conditioning weights.
pub const QWEN_Q4S_BASE_CONDITIONING_PROFILE: &str = "q4s-block-up-to128-f32";
/// Dependency role used by composed image-model manifests.
pub const QWEN_COMPONENT_ROLE: &str = "qwen";
/// Canonical reusable component bundle id.
pub const QWEN_COMPONENT_BUNDLE_ID: &str = "qwen3-vl-8b-base-boogu-image-0.1";
/// Canonical reusable packed-Q4S component bundle id.
pub const QWEN_Q4S_COMPONENT_BUNDLE_ID: &str =
    "qwen3-vl-8b-base-boogu-image-0.1-q4s-block-up-to128-f32";
/// Provenance model id for the exact Qwen source shared by the Boogu 0.1 releases.
pub const QWEN_COMPONENT_MODEL_ID: &str = "BooguDerived/Qwen3-VL-8B-Base-0.1";
/// SHA-256 of the canonical sorted declarations for the four shared upstream MLLM shards.
pub const QWEN_COMPONENT_MODEL_REVISION: &str =
    "020ea5b58bd3fc9abf5f23e92e4039864a6d6ff4993db777a713b489bbd6c5a1";
/// Exact sealed digest of the canonical reusable Qwen component manifest.
pub const QWEN_COMPONENT_CONTENT_DIGEST: &str =
    "2bab9d7c378158137c117a43d7a3cc5d66dc94af5dd0856d12348d08b2b9e9da";
/// Exact sealed digest of the canonical reusable packed-Q4S Qwen component manifest.
pub const QWEN_Q4S_COMPONENT_CONTENT_DIGEST: &str =
    "d3e332ebd710d87fa6a2ae97eef3302f5c9f5e7d3f4e27675f0c4c4f5a31c5de";

/// Construct the complete immutable dependency pin for the released Qwen component.
pub fn qwen_component_dependency() -> ArtifactDependency {
    ArtifactDependency {
        role: ArtifactComponentId::new(QWEN_COMPONENT_ROLE).expect("static role is valid"),
        bundle: ArtifactBundleId::new(QWEN_COMPONENT_BUNDLE_ID).expect("static bundle is valid"),
        profile: ArtifactProfileId::new(QWEN_BASE_CONDITIONING_PROFILE)
            .expect("static profile is valid"),
        model: ModelId::new(QWEN_COMPONENT_MODEL_ID).expect("static model id is valid"),
        model_revision: QWEN_COMPONENT_MODEL_REVISION.to_owned(),
        content_digest: Sha256Digest::from_hex(QWEN_COMPONENT_CONTENT_DIGEST)
            .expect("static digest is valid"),
    }
}

/// Construct the complete immutable dependency pin for the released packed-Q4S Qwen component.
pub fn qwen_q4s_component_dependency() -> ArtifactDependency {
    ArtifactDependency {
        role: ArtifactComponentId::new(QWEN_COMPONENT_ROLE).expect("static role is valid"),
        bundle: ArtifactBundleId::new(QWEN_Q4S_COMPONENT_BUNDLE_ID)
            .expect("static bundle is valid"),
        profile: ArtifactProfileId::new(QWEN_Q4S_BASE_CONDITIONING_PROFILE)
            .expect("static profile is valid"),
        model: ModelId::new(QWEN_COMPONENT_MODEL_ID).expect("static model id is valid"),
        model_revision: QWEN_COMPONENT_MODEL_REVISION.to_owned(),
        content_digest: Sha256Digest::from_hex(QWEN_Q4S_COMPONENT_CONTENT_DIGEST)
            .expect("static digest is valid"),
    }
}

/// Stable artifact component name for one independently loadable Qwen stage.
pub fn qwen_streaming_stage_name(stage: &Qwen3VlStage) -> String {
    match stage {
        Qwen3VlStage::EmbeddingRows { chunk } => format!("qwen-embedding-rows-{chunk:02}"),
        Qwen3VlStage::VisionPrelude => "qwen-vision-prelude".into(),
        Qwen3VlStage::VisionBlock { index } => format!("qwen-vision-block-{index:02}"),
        Qwen3VlStage::VisionDeepstackMerger { index, .. } => {
            format!("qwen-vision-deepstack-merger-{index:02}")
        }
        Qwen3VlStage::VisionFinalMerger => "qwen-vision-final-merger".into(),
        Qwen3VlStage::TextBlock { index } => format!("qwen-text-block-{index:02}"),
        Qwen3VlStage::TextFinalNorm => "qwen-text-final-norm".into(),
        Qwen3VlStage::LmHeadRows { chunk } => format!("qwen-lm-head-rows-{chunk:02}"),
    }
}

/// Unique Burnpack snapshot path for one contiguous vocabulary-table row slice.
pub fn qwen_row_slice_target(logical_target: &str, chunk: &RowChunkSpec) -> String {
    format!(
        "{logical_target}.rows.{:02}.{:06}-{:06}",
        chunk.chunk_index, chunk.row_range.start, chunk.row_range.end
    )
}

/// Explicit load conversion after the sealed storage dtype has been checked.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Qwen3VlArtifactFloatPolicy {
    /// Keep the F16/F32 release dtype on a backend that supports it.
    #[default]
    Preserve,
    /// Convert bounded F16 stages to F32 during application.
    AdaptToF32,
    /// Retain rank-two F16 matrices for packed backend kernels while widening norms and biases.
    ///
    /// This policy is for F32-activation backends that can bind F16 bytes but do not expose a
    /// typed F16 shader feature. It never changes the sealed artifact representation.
    PackedF16WeightsF32Auxiliaries,
    /// Convert authenticated rank-two weights to packed signed Q4S block-128/F32 storage while
    /// keeping normalization parameters and biases in F32.
    ///
    /// Execution uses F32 activations and a quantization-aware matmul. Embedding row objects stay
    /// F16 because browser execution selects the required rows on the host before upload.
    PackedQ4sBlock128WeightsF32Auxiliaries,
}

/// Failure while validating or applying a standalone Qwen component bundle.
#[derive(Debug, Error)]
pub enum Qwen3VlArtifactError {
    #[error("invalid Qwen3-VL component manifest: {0}")]
    Manifest(String),
    #[error(transparent)]
    Read(#[from] ArtifactReadError),
    #[error("invalid Burnpack object {path}: {message}")]
    Burnpack { path: String, message: String },
    #[error("Qwen3-VL artifact contract failed for {stage}: {message}")]
    Contract { stage: String, message: String },
    #[error("Qwen3-VL stage initialization failed: {0}")]
    Model(String),
    #[error("device synchronization after Qwen3-VL stage failed: {0}")]
    Synchronize(String),
}

fn contract(stage: impl Into<String>, message: impl Into<String>) -> Qwen3VlArtifactError {
    Qwen3VlArtifactError::Contract {
        stage: stage.into(),
        message: message.into(),
    }
}

/// Validated semantic view of a sealed base-conditioning component manifest.
///
/// Construction is allocation-free with respect to model weights. Every logical Burnpack object is
/// read (and may be reconstructed from physical transport parts), SHA-256 checked, parsed, applied,
/// and dropped only when its semantic stage is requested.
#[derive(Clone)]
pub struct Qwen3VlComponentContract {
    manifest: ArtifactManifest,
    config: Qwen3VlConfig,
    plan: Qwen3VlStreamingPlan,
    stages: BTreeMap<String, Vec<ArtifactFile>>,
    max_shard_bytes: u64,
}

impl Qwen3VlComponentContract {
    /// Validate the canonical released base-conditioning plan (no LM head).
    pub fn released_base(
        manifest: ArtifactManifest,
        config: Qwen3VlConfig,
    ) -> Result<Self, Qwen3VlArtifactError> {
        let plan = Qwen3VlStreamingPlan::released_f16(&config, false)
            .map_err(|error| Qwen3VlArtifactError::Model(error.to_string()))?;
        Self::new(manifest, config, plan)
    }
    /// Validate an explicit base-only row partition.
    pub fn new(
        manifest: ArtifactManifest,
        config: Qwen3VlConfig,
        plan: Qwen3VlStreamingPlan,
    ) -> Result<Self, Qwen3VlArtifactError> {
        manifest
            .validate_sealed()
            .map_err(|error| Qwen3VlArtifactError::Manifest(error.to_string()))?;
        validate_identity(&manifest)?;
        validate_metadata(&manifest)?;
        validate_base_plan(&config, &plan)?;

        let max_shard_bytes = declared_max_shard_bytes(&manifest)?;
        let descriptors = plan
            .stages
            .iter()
            .map(|descriptor| (qwen_streaming_stage_name(&descriptor.stage), descriptor))
            .collect::<BTreeMap<_, _>>();
        let required = descriptors.keys().cloned().collect::<BTreeSet<_>>();
        let declared = manifest
            .components
            .iter()
            .map(|component| component.id.as_str().to_owned())
            .collect::<BTreeSet<_>>();
        if declared != required
            || manifest
                .components
                .iter()
                .any(|component| !component.required)
        {
            return Err(contract(
                "qwen",
                format!(
                    "component declarations differ from the exact required stage set: expected={required:?}, actual={declared:?}"
                ),
            ));
        }
        let mut stages = BTreeMap::<String, Vec<ArtifactFile>>::new();
        for file in manifest
            .files
            .iter()
            .filter(|file| file.role == ArtifactFileRole::Weights)
        {
            let stage = file.component.as_ref().ok_or_else(|| {
                contract(
                    "qwen",
                    format!("weight object {} has no component", file.path),
                )
            })?;
            if !required.contains(stage.as_str()) {
                return Err(contract(
                    stage.as_str(),
                    format!(
                        "weight object {} belongs to an unknown or excluded stage",
                        file.path
                    ),
                ));
            }
            if file.size > max_shard_bytes {
                return Err(contract(
                    stage.as_str(),
                    format!(
                        "object {} is {} bytes, exceeding the declared {max_shard_bytes}-byte bound",
                        file.path, file.size
                    ),
                ));
            }
            let expected_path = format!("objects/{}.bpk", file.sha256);
            if file.path.as_str() != expected_path {
                return Err(contract(
                    stage.as_str(),
                    format!("weight object {} is not content-addressed", file.path),
                ));
            }
            stages
                .entry(stage.as_str().to_owned())
                .or_default()
                .push(file.clone());
        }
        for stage in &required {
            if !stages.contains_key(stage) {
                return Err(contract(stage, "sealed manifest omits the required stage"));
            }
        }
        for files in stages.values_mut() {
            files.sort_by_key(|file| file.shard.map_or(0, |shard| shard.index));
        }

        Ok(Self {
            manifest,
            config,
            plan,
            stages,
            max_shard_bytes,
        })
    }
    pub fn manifest(&self) -> &ArtifactManifest {
        &self.manifest
    }

    pub fn config(&self) -> &Qwen3VlConfig {
        &self.config
    }

    pub fn plan(&self) -> &Qwen3VlStreamingPlan {
        &self.plan
    }

    pub const fn max_shard_bytes(&self) -> u64 {
        self.max_shard_bytes
    }

    fn descriptor(&self, stage: &Qwen3VlStage) -> &Qwen3VlStageDescriptor {
        self.plan
            .stages
            .iter()
            .find(|descriptor| descriptor.stage == *stage)
            .expect("validated plan contains every requested stage")
    }

    fn files(&self, stage: &Qwen3VlStage) -> &[ArtifactFile] {
        self.stages
            .get(&qwen_streaming_stage_name(stage))
            .expect("validated contract contains every stage")
    }
}

fn validate_base_plan(
    config: &Qwen3VlConfig,
    plan: &Qwen3VlStreamingPlan,
) -> Result<(), Qwen3VlArtifactError> {
    if plan.lm_head_rows.is_some()
        || plan
            .stages
            .iter()
            .any(|stage| matches!(stage.stage, Qwen3VlStage::LmHeadRows { .. }))
    {
        return Err(contract(
            "qwen",
            "the reusable base-conditioning bundle must omit the LM head",
        ));
    }
    let canonical = Qwen3VlStreamingPlan::new(config, plan.embedding_rows.clone(), None)
        .map_err(|error| Qwen3VlArtifactError::Model(error.to_string()))?;
    if &canonical != plan {
        return Err(contract(
            "qwen",
            "streaming descriptors differ from the canonical config and row partition",
        ));
    }
    let logical_tensor_count = crate::WeightInventory::for_config(config, false)
        .specs()
        .len();
    let stored_tensor_count = plan
        .stages
        .iter()
        .map(|stage| stage.tensors.len() + usize::from(stage.row_slice.is_some()))
        .sum::<usize>();
    if logical_tensor_count != 749 || stored_tensor_count != 754 {
        return Err(contract(
            "qwen",
            format!(
                "config/plan is not the canonical 749-logical/754-stored base inventory: logical={logical_tensor_count}, stored={stored_tensor_count}"
            ),
        ));
    }

    Ok(())
}

fn validate_identity(manifest: &ArtifactManifest) -> Result<(), Qwen3VlArtifactError> {
    if manifest.schema_version != ARTIFACT_MANIFEST_SCHEMA_V1 {
        return Err(Qwen3VlArtifactError::Manifest(format!(
            "standalone components require schema {ARTIFACT_MANIFEST_SCHEMA_V1}, found {}",
            manifest.schema_version
        )));
    }
    if !manifest.dependencies.is_empty() {
        return Err(Qwen3VlArtifactError::Manifest(
            "a standalone Qwen component must not depend on another bundle".into(),
        ));
    }
    let (expected_bundle, expected_profile, expected_content_digest) =
        match manifest.profile.as_str() {
            QWEN_BASE_CONDITIONING_PROFILE => (
                QWEN_COMPONENT_BUNDLE_ID,
                QWEN_BASE_CONDITIONING_PROFILE,
                QWEN_COMPONENT_CONTENT_DIGEST,
            ),
            QWEN_Q4S_BASE_CONDITIONING_PROFILE => (
                QWEN_Q4S_COMPONENT_BUNDLE_ID,
                QWEN_Q4S_BASE_CONDITIONING_PROFILE,
                QWEN_Q4S_COMPONENT_CONTENT_DIGEST,
            ),
            actual => {
                return Err(Qwen3VlArtifactError::Manifest(format!(
                    "profile {actual} is not a canonical released Qwen component profile"
                )));
            }
        };
    for (field, actual, expected) in [
        ("bundle", manifest.bundle.as_str(), expected_bundle),
        ("profile", manifest.profile.as_str(), expected_profile),
        ("model", manifest.model.as_str(), QWEN_COMPONENT_MODEL_ID),
        (
            "model_revision",
            manifest.model_revision.as_str(),
            QWEN_COMPONENT_MODEL_REVISION,
        ),
    ] {
        if actual != expected {
            return Err(Qwen3VlArtifactError::Manifest(format!(
                "{field} {actual:?} differs from canonical {expected:?}"
            )));
        }
    }
    let expected_digest =
        Sha256Digest::from_hex(expected_content_digest).expect("static component digest is valid");
    if manifest.content_digest != Some(expected_digest) {
        return Err(Qwen3VlArtifactError::Manifest(format!(
            "content digest {:?} differs from canonical {expected_digest}",
            manifest.content_digest
        )));
    }
    Ok(())
}

fn validate_metadata(manifest: &ArtifactManifest) -> Result<(), Qwen3VlArtifactError> {
    for (key, expected) in QWEN_METADATA_VALUES {
        let actual = manifest.metadata.get(key).map(String::as_str);
        if actual != Some(expected) {
            return Err(Qwen3VlArtifactError::Manifest(format!(
                "metadata {key} must be {expected:?}, found {actual:?}"
            )));
        }
    }
    for path in QWEN_METADATA_PATHS {
        if !manifest.files.iter().any(|file| file.path.as_str() == path) {
            return Err(Qwen3VlArtifactError::Manifest(format!(
                "component manifest omits required metadata file {path}"
            )));
        }
    }
    if let Some(file) = manifest
        .files
        .iter()
        .filter(|file| file.role != ArtifactFileRole::Weights)
        .find(|file| !QWEN_METADATA_PATHS.contains(&file.path.as_str()))
    {
        return Err(Qwen3VlArtifactError::Manifest(format!(
            "component manifest contains unexpected metadata file {}",
            file.path
        )));
    }
    Ok(())
}

fn declared_max_shard_bytes(manifest: &ArtifactManifest) -> Result<u64, Qwen3VlArtifactError> {
    let value = manifest
        .metadata
        .get("target_max_shard_bytes")
        .ok_or_else(|| {
            Qwen3VlArtifactError::Manifest("manifest omits target_max_shard_bytes".into())
        })?
        .parse::<u64>()
        .map_err(|error| {
            Qwen3VlArtifactError::Manifest(format!("invalid target_max_shard_bytes: {error}"))
        })?;
    if value == 0 {
        return Err(Qwen3VlArtifactError::Manifest(
            "target_max_shard_bytes must be positive".into(),
        ));
    }
    if manifest.numeric_format != burn_image::NumericFormat::Other(manifest.profile.to_string()) {
        return Err(Qwen3VlArtifactError::Manifest(format!(
            "numeric format {:?} is not the canonical mixed profile",
            manifest.numeric_format
        )));
    }
    Ok(value)
}

/// Synchronous verified stage source backed by any model-neutral shard reader.
pub struct VerifiedBurnpackQwen3VlStageSource<B: Backend, R> {
    contract: Qwen3VlComponentContract,
    device: B::Device,
    reader: R,
    float_policy: Qwen3VlArtifactFloatPolicy,
}

impl<B: Backend, R: ArtifactShardReader> VerifiedBurnpackQwen3VlStageSource<B, R> {
    pub fn new(contract: Qwen3VlComponentContract, device: B::Device, reader: R) -> Self {
        Self {
            contract,
            device,
            reader,
            float_policy: Qwen3VlArtifactFloatPolicy::Preserve,
        }
    }

    pub fn with_float_policy(mut self, policy: Qwen3VlArtifactFloatPolicy) -> Self {
        self.float_policy = policy;
        self
    }

    pub fn contract(&self) -> &Qwen3VlComponentContract {
        &self.contract
    }

    pub fn reader(&self) -> &R {
        &self.reader
    }

    pub fn reader_mut(&mut self) -> &mut R {
        &mut self.reader
    }

    fn verified_bytes(&mut self, file: &ArtifactFile) -> Result<Vec<u8>, Qwen3VlArtifactError> {
        let bytes = self.reader.read_shard(file)?;
        VerifiedArtifactBytes::unverified(bytes)
            .into_verified_bytes(file, self.contract.max_shard_bytes())
            .map_err(Into::into)
    }

    fn load_module<M: ModuleSnapshot<B>>(
        &mut self,
        stage: Qwen3VlStage,
        prefix: &str,
        mut module: M,
    ) -> Result<M, Qwen3VlArtifactError> {
        let descriptor = self.contract.descriptor(&stage).clone();
        let files = self.contract.files(&stage).to_vec();
        let dtype = Qwen3VlStageDTypePolicy::released_hybrid().for_stage(&stage);
        let mut applied = BTreeSet::new();
        for file in files {
            let bytes = self.verified_bytes(&file)?;
            apply_module_object(
                &mut module,
                &descriptor,
                &file,
                bytes,
                prefix,
                dtype,
                self.float_policy,
                &mut applied,
            )?;
        }
        ensure_module_complete(&descriptor, &applied)?;
        Ok(module)
    }

    fn load_rows(&mut self, spec: &RowChunkSpec) -> Result<Tensor<B, 2>, Qwen3VlArtifactError> {
        let stage = Qwen3VlStage::EmbeddingRows {
            chunk: spec.chunk_index,
        };
        let target = qwen_row_slice_target("model.language_model.embed_tokens.weight", spec);
        let files = self.contract.files(&stage).to_vec();
        let mut found = None;
        for file in files {
            let bytes = self.verified_bytes(&file)?;
            let tensor = parse_row_object::<B>(
                &file,
                bytes,
                &target,
                spec,
                self.float_policy,
                &self.device,
            )?;
            if found.replace(tensor).is_some() {
                return Err(contract(
                    qwen_streaming_stage_name(&stage),
                    "row slice appears in more than one logical Burnpack object",
                ));
            }
        }
        found.ok_or_else(|| contract(qwen_streaming_stage_name(&stage), "row stage is empty"))
    }
}

impl<B: Backend> VerifiedBurnpackQwen3VlStageSource<B, DirectoryArtifactShardReader> {
    pub fn from_directory(
        root: impl AsRef<Path>,
        config: Qwen3VlConfig,
        device: B::Device,
    ) -> Result<Self, Qwen3VlArtifactError> {
        let directory = VerifiedArtifactDirectory::open(root.as_ref().to_owned())?;
        let contract =
            Qwen3VlComponentContract::released_base(directory.manifest().clone(), config)?;
        let reader = directory.shard_reader()?;
        Ok(Self::new(contract, device, reader))
    }
}

impl<B: Backend, R: ArtifactShardReader> Qwen3VlStageSource<B>
    for VerifiedBurnpackQwen3VlStageSource<B, R>
{
    type Error = Qwen3VlArtifactError;

    fn load_embedding_rows(
        &mut self,
        spec: &RowChunkSpec,
    ) -> Result<EmbeddingRowChunk<B>, Self::Error> {
        EmbeddingRowChunk::new(spec.clone(), self.load_rows(spec)?)
            .map_err(|error| contract("qwen-embedding", error.to_string()))
    }

    fn load_vision_prelude(&mut self) -> Result<Qwen3VlVisionPrelude<B>, Self::Error> {
        let module =
            Qwen3VlVisionPrelude::new(self.contract.config.vision_config.clone(), &self.device)
                .map_err(|error| Qwen3VlArtifactError::Model(error.to_string()))?;
        self.load_module(Qwen3VlStage::VisionPrelude, "model.visual.", module)
    }

    fn load_vision_block(&mut self, index: usize) -> Result<Qwen3VlVisionBlock<B>, Self::Error> {
        let module = Qwen3VlVisionBlock::new(&self.contract.config.vision_config, &self.device);
        self.load_module(
            Qwen3VlStage::VisionBlock { index },
            &format!("model.visual.blocks.{index}."),
            module,
        )
    }

    fn load_vision_deepstack_merger(
        &mut self,
        index: usize,
    ) -> Result<Qwen3VlVisionPatchMerger<B>, Self::Error> {
        let module =
            Qwen3VlVisionPatchMerger::new(&self.contract.config.vision_config, true, &self.device);
        self.load_module(
            Qwen3VlStage::VisionDeepstackMerger {
                index,
                after_block: *self
                    .contract
                    .config
                    .vision_config
                    .deepstack_visual_indexes
                    .get(index)
                    .ok_or_else(|| contract("qwen", "unknown deepstack merger"))?,
            },
            &format!("model.visual.deepstack_merger_list.{index}."),
            module,
        )
    }

    fn load_vision_final_merger(&mut self) -> Result<Qwen3VlVisionPatchMerger<B>, Self::Error> {
        let module =
            Qwen3VlVisionPatchMerger::new(&self.contract.config.vision_config, false, &self.device);
        self.load_module(
            Qwen3VlStage::VisionFinalMerger,
            "model.visual.merger.",
            module,
        )
    }

    fn load_text_block(&mut self, index: usize) -> Result<Qwen3VlDecoderLayer<B>, Self::Error> {
        let module = Qwen3VlDecoderLayer::new(&self.contract.config.text_config, &self.device);
        self.load_module(
            Qwen3VlStage::TextBlock { index },
            &format!("model.language_model.layers.{index}."),
            module,
        )
    }

    fn load_text_final_norm(&mut self) -> Result<RmsNorm<B>, Self::Error> {
        let module = RmsNormConfig::new(self.contract.config.text_config.hidden_size)
            .with_epsilon(self.contract.config.text_config.rms_norm_eps)
            .init(&self.device);
        self.load_module(
            Qwen3VlStage::TextFinalNorm,
            "model.language_model.norm.",
            module,
        )
    }

    fn synchronize(&mut self) -> Result<(), Self::Error> {
        B::sync(&self.device).map_err(|error| Qwen3VlArtifactError::Synchronize(error.to_string()))
    }
}

/// Asynchronous verified stage source for browser fetch/cache adapters.
pub struct VerifiedAsyncBurnpackQwen3VlStageSource<B: Backend, R> {
    contract: Qwen3VlComponentContract,
    device: B::Device,
    reader: R,
    float_policy: Qwen3VlArtifactFloatPolicy,
}

impl<B: Backend, R: AsyncArtifactShardReader> VerifiedAsyncBurnpackQwen3VlStageSource<B, R> {
    pub fn new(contract: Qwen3VlComponentContract, device: B::Device, reader: R) -> Self {
        Self {
            contract,
            device,
            reader,
            float_policy: Qwen3VlArtifactFloatPolicy::Preserve,
        }
    }

    pub fn with_float_policy(mut self, policy: Qwen3VlArtifactFloatPolicy) -> Self {
        self.float_policy = policy;
        self
    }

    pub fn contract(&self) -> &Qwen3VlComponentContract {
        &self.contract
    }

    pub fn reader(&self) -> &R {
        &self.reader
    }

    pub fn reader_mut(&mut self) -> &mut R {
        &mut self.reader
    }

    async fn verified_bytes(
        &mut self,
        file: &ArtifactFile,
    ) -> Result<Vec<u8>, Qwen3VlArtifactError> {
        self.reader
            .read_verified_shard(file, self.contract.max_shard_bytes())
            .await?
            .into_verified_bytes(file, self.contract.max_shard_bytes())
            .map_err(Into::into)
    }

    async fn load_module<M: ModuleSnapshot<B>>(
        &mut self,
        stage: Qwen3VlStage,
        prefix: &str,
        mut module: M,
    ) -> Result<M, Qwen3VlArtifactError> {
        let descriptor = self.contract.descriptor(&stage).clone();
        let files = self.contract.files(&stage).to_vec();
        let dtype = Qwen3VlStageDTypePolicy::released_hybrid().for_stage(&stage);
        let mut applied = BTreeSet::new();
        for file in files {
            let bytes = self.verified_bytes(&file).await?;
            apply_module_object(
                &mut module,
                &descriptor,
                &file,
                bytes,
                prefix,
                dtype,
                self.float_policy,
                &mut applied,
            )?;
        }
        ensure_module_complete(&descriptor, &applied)?;
        Ok(module)
    }

    async fn load_rows(
        &mut self,
        spec: &RowChunkSpec,
    ) -> Result<Tensor<B, 2>, Qwen3VlArtifactError> {
        let stage = Qwen3VlStage::EmbeddingRows {
            chunk: spec.chunk_index,
        };
        let target = qwen_row_slice_target("model.language_model.embed_tokens.weight", spec);
        let files = self.contract.files(&stage).to_vec();
        let mut found = None;
        for file in files {
            let bytes = self.verified_bytes(&file).await?;
            let tensor = parse_row_object::<B>(
                &file,
                bytes,
                &target,
                spec,
                self.float_policy,
                &self.device,
            )?;
            if found.replace(tensor).is_some() {
                return Err(contract(
                    qwen_streaming_stage_name(&stage),
                    "row slice appears in more than one logical Burnpack object",
                ));
            }
        }
        found.ok_or_else(|| contract(qwen_streaming_stage_name(&stage), "row stage is empty"))
    }
}

impl<B: Backend, R: AsyncArtifactShardReader> AsyncQwen3VlStageSource<B>
    for VerifiedAsyncBurnpackQwen3VlStageSource<B, R>
{
    type Error = Qwen3VlArtifactError;

    async fn load_embedding_rows(
        &mut self,
        spec: &RowChunkSpec,
    ) -> Result<EmbeddingRowChunk<B>, Self::Error> {
        EmbeddingRowChunk::new(spec.clone(), self.load_rows(spec).await?)
            .map_err(|error| contract("qwen-embedding", error.to_string()))
    }

    async fn load_host_routed_f16_embedding_f32(
        &mut self,
        input_ids: &[Vec<i64>],
        device: &B::Device,
    ) -> Result<Option<HostRoutedEmbedding<B>>, Self::Error> {
        let mut state = HostRoutedF16EmbeddingState::new(
            input_ids,
            self.contract.config.text_config.vocab_size,
            self.contract.config.text_config.hidden_size,
        )
        .map_err(|error| contract("qwen-embedding", error.to_string()))?;
        // Clone only compact metadata. Every full object is fetched, authenticated, parsed,
        // selected, and dropped before the following object begins.
        let specs = self.contract.plan.embedding_rows.chunks.clone();
        for spec in specs {
            let stage = Qwen3VlStage::EmbeddingRows {
                chunk: spec.chunk_index,
            };
            let target = qwen_row_slice_target("model.language_model.embed_tokens.weight", &spec);
            let files = self.contract.files(&stage).to_vec();
            let mut applied = false;
            for file in files {
                let bytes = self.verified_bytes(&file).await?;
                let authenticated_object_bytes = u64::try_from(bytes.len()).map_err(|_| {
                    contract(
                        qwen_streaming_stage_name(&stage),
                        "authenticated row object byte count does not fit u64",
                    )
                })?;
                let data = parse_row_object_data(
                    &file,
                    bytes,
                    &target,
                    &spec,
                    Qwen3VlArtifactFloatPolicy::Preserve,
                )?;
                if applied {
                    return Err(contract(
                        qwen_streaming_stage_name(&stage),
                        "row slice appears in more than one logical Burnpack object",
                    ));
                }
                state
                    .apply_chunk_data(&spec, &data, authenticated_object_bytes)
                    .map_err(|error| contract("qwen-embedding", error.to_string()))?;
                applied = true;
            }
            if !applied {
                return Err(contract(
                    qwen_streaming_stage_name(&stage),
                    "row stage is empty",
                ));
            }
        }
        let (data, report) = state
            .finish()
            .map_err(|error| contract("qwen-embedding", error.to_string()))?;
        let tensor = Tensor::<B, 3>::from_data(data, device);
        Ok(Some(HostRoutedEmbedding { tensor, report }))
    }

    async fn load_vision_prelude(&mut self) -> Result<Qwen3VlVisionPrelude<B>, Self::Error> {
        let module =
            Qwen3VlVisionPrelude::new(self.contract.config.vision_config.clone(), &self.device)
                .map_err(|error| Qwen3VlArtifactError::Model(error.to_string()))?;
        self.load_module(Qwen3VlStage::VisionPrelude, "model.visual.", module)
            .await
    }

    async fn load_vision_block(
        &mut self,
        index: usize,
    ) -> Result<Qwen3VlVisionBlock<B>, Self::Error> {
        let module = Qwen3VlVisionBlock::new(&self.contract.config.vision_config, &self.device);
        self.load_module(
            Qwen3VlStage::VisionBlock { index },
            &format!("model.visual.blocks.{index}."),
            module,
        )
        .await
    }

    async fn load_vision_deepstack_merger(
        &mut self,
        index: usize,
    ) -> Result<Qwen3VlVisionPatchMerger<B>, Self::Error> {
        let module =
            Qwen3VlVisionPatchMerger::new(&self.contract.config.vision_config, true, &self.device);
        let after_block = *self
            .contract
            .config
            .vision_config
            .deepstack_visual_indexes
            .get(index)
            .ok_or_else(|| contract("qwen", "unknown deepstack merger"))?;
        self.load_module(
            Qwen3VlStage::VisionDeepstackMerger { index, after_block },
            &format!("model.visual.deepstack_merger_list.{index}."),
            module,
        )
        .await
    }

    async fn load_vision_final_merger(
        &mut self,
    ) -> Result<Qwen3VlVisionPatchMerger<B>, Self::Error> {
        let module =
            Qwen3VlVisionPatchMerger::new(&self.contract.config.vision_config, false, &self.device);
        self.load_module(
            Qwen3VlStage::VisionFinalMerger,
            "model.visual.merger.",
            module,
        )
        .await
    }

    async fn load_text_block(
        &mut self,
        index: usize,
    ) -> Result<Qwen3VlDecoderLayer<B>, Self::Error> {
        let module = Qwen3VlDecoderLayer::new(&self.contract.config.text_config, &self.device);
        self.load_module(
            Qwen3VlStage::TextBlock { index },
            &format!("model.language_model.layers.{index}."),
            module,
        )
        .await
    }

    async fn load_text_final_norm(&mut self) -> Result<RmsNorm<B>, Self::Error> {
        let module = RmsNormConfig::new(self.contract.config.text_config.hidden_size)
            .with_epsilon(self.contract.config.text_config.rms_norm_eps)
            .init(&self.device);
        self.load_module(
            Qwen3VlStage::TextFinalNorm,
            "model.language_model.norm.",
            module,
        )
        .await
    }

    async fn synchronize(&mut self) -> Result<(), Self::Error> {
        B::sync(&self.device).map_err(|error| Qwen3VlArtifactError::Synchronize(error.to_string()))
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_module_object<B: Backend, M: ModuleSnapshot<B>>(
    module: &mut M,
    descriptor: &Qwen3VlStageDescriptor,
    file: &ArtifactFile,
    bytes: Vec<u8>,
    prefix: &str,
    expected_dtype: Qwen3VlStageDType,
    float_policy: Qwen3VlArtifactFloatPolicy,
    applied: &mut BTreeSet<String>,
) -> Result<(), Qwen3VlArtifactError> {
    let stage = qwen_streaming_stage_name(&descriptor.stage);
    let expected = descriptor
        .tensors
        .iter()
        .map(|spec| (spec.target.as_str(), spec.shape.as_slice()))
        .collect::<BTreeMap<_, _>>();
    let mut store = BurnpackStore::from_bytes(Some(Bytes::from_bytes_vec(bytes)));
    let snapshots = store
        .get_all_snapshots()
        .map_err(|error| Qwen3VlArtifactError::Burnpack {
            path: file.path.to_string(),
            message: error.to_string(),
        })?;
    if snapshots.is_empty() {
        return Err(contract(&stage, format!("{} is empty", file.path)));
    }
    let mut local = Vec::with_capacity(snapshots.len());
    for (name, snapshot) in snapshots {
        let shape = expected
            .get(name.as_str())
            .ok_or_else(|| contract(&stage, format!("unknown tensor {name}")))?;
        if !applied.insert(name.clone()) {
            return Err(contract(&stage, format!("duplicate tensor {name}")));
        }
        if snapshot.shape.as_slice() != *shape {
            return Err(contract(
                &stage,
                format!(
                    "tensor {name} shape mismatch: expected {shape:?}, got {:?}",
                    snapshot.shape
                ),
            ));
        }
        validate_artifact_dtype(
            &stage,
            name,
            snapshot.dtype,
            expected_dtype,
            &snapshot.shape,
            float_policy,
        )?;
        let local_name = name
            .strip_prefix(prefix)
            .ok_or_else(|| contract(&stage, format!("tensor {name} lacks prefix {prefix:?}")))?;
        local.push(rename_snapshot(snapshot, local_name));
    }
    let expected_applied = local
        .iter()
        .map(TensorSnapshot::full_path)
        .collect::<BTreeSet<_>>();
    let result = module.apply(local, None, load_adapter(float_policy), false);
    validate_apply(&stage, &result, &expected_applied)
}

fn ensure_module_complete(
    descriptor: &Qwen3VlStageDescriptor,
    applied: &BTreeSet<String>,
) -> Result<(), Qwen3VlArtifactError> {
    let missing = descriptor
        .tensors
        .iter()
        .map(|spec| spec.target.as_str())
        .filter(|name| !applied.contains(*name))
        .take(16)
        .collect::<Vec<_>>();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(contract(
            qwen_streaming_stage_name(&descriptor.stage),
            format!("stage is incomplete; first missing tensors: {missing:?}"),
        ))
    }
}

fn parse_row_object<B: Backend>(
    file: &ArtifactFile,
    bytes: Vec<u8>,
    target: &str,
    spec: &RowChunkSpec,
    float_policy: Qwen3VlArtifactFloatPolicy,
    device: &B::Device,
) -> Result<Tensor<B, 2>, Qwen3VlArtifactError> {
    let mut data = parse_row_object_data(file, bytes, target, spec, float_policy)?;
    match float_policy {
        Qwen3VlArtifactFloatPolicy::AdaptToF32 => {
            data = data.convert_dtype(DType::F32);
        }
        Qwen3VlArtifactFloatPolicy::PackedQ4sBlock128WeightsF32Auxiliaries => match data.dtype {
            DType::F16 | DType::F32 => {
                data = quantize_q4s_block128_f32(data)
                    .map_err(|error| contract("qwen-embedding", error.to_string()))?;
            }
            DType::QFloat(scheme)
                if scheme.value == QuantValue::Q4S
                    && scheme.level == QuantLevel::block([128])
                    && scheme.param == QuantParam::F32
                    && scheme.store == QuantStore::PackedU32(0) => {}
            other => {
                return Err(contract(
                    "qwen-embedding",
                    format!("packed Q4S row object has unsupported dtype {other:?}"),
                ));
            }
        },
        Qwen3VlArtifactFloatPolicy::Preserve
        | Qwen3VlArtifactFloatPolicy::PackedF16WeightsF32Auxiliaries => {}
    }
    let dtype = data.dtype;
    Ok(Tensor::from_data(data, (device, dtype)))
}

/// Parse and validate the complete released row object without allocating a backend tensor.
/// Host-routed browser execution passes [`Qwen3VlArtifactFloatPolicy::Preserve`] to require raw
/// F16 rows, while resident Q4 execution admits only the exact packed block-128/F32 scheme.
fn parse_row_object_data(
    file: &ArtifactFile,
    bytes: Vec<u8>,
    target: &str,
    spec: &RowChunkSpec,
    float_policy: Qwen3VlArtifactFloatPolicy,
) -> Result<TensorData, Qwen3VlArtifactError> {
    let mut store = BurnpackStore::from_bytes(Some(Bytes::from_bytes_vec(bytes)));
    let snapshots = store
        .get_all_snapshots()
        .map_err(|error| Qwen3VlArtifactError::Burnpack {
            path: file.path.to_string(),
            message: error.to_string(),
        })?;
    if snapshots.len() != 1 {
        return Err(contract(
            file.component
                .as_ref()
                .map_or("qwen-embedding", |stage| stage.as_str()),
            format!(
                "row object {} contains {} tensors, expected one",
                file.path,
                snapshots.len()
            ),
        ));
    }
    let (name, snapshot) = snapshots.iter().next().expect("length checked");
    if name.as_str() != target || snapshot.shape.as_slice() != [spec.rows(), spec.hidden_size] {
        return Err(contract(
            file.component
                .as_ref()
                .map_or("qwen-embedding", |stage| stage.as_str()),
            format!(
                "row tensor {name} does not match {target} [{}, {}]",
                spec.rows(),
                spec.hidden_size
            ),
        ));
    }
    validate_artifact_dtype(
        "qwen-embedding",
        name,
        snapshot.dtype,
        Qwen3VlStageDType::F16,
        &snapshot.shape,
        float_policy,
    )?;
    let data = snapshot
        .to_data()
        .map_err(|error| contract("qwen-embedding", format!("read {name}: {error}")))?;
    let expected_bytes = expected_row_payload_bytes(spec, data.dtype)?;
    if data.bytes.len() != expected_bytes {
        return Err(contract(
            "qwen-embedding",
            format!(
                "row tensor {name} has {} payload bytes, expected {expected_bytes}",
                data.bytes.len()
            ),
        ));
    }
    Ok(data)
}

fn is_packed_q4s_block128_f32(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::QFloat(scheme)
            if scheme.value == QuantValue::Q4S
                && scheme.level == QuantLevel::block([128])
                && scheme.param == QuantParam::F32
                && scheme.store == QuantStore::PackedU32(0)
    )
}

fn is_q4s_block128_matrix_shape(shape: &[usize]) -> bool {
    shape.len() == 2 && shape.last().is_some_and(|inner| inner.is_multiple_of(128))
}

fn validate_artifact_dtype(
    stage: &str,
    name: &str,
    actual: DType,
    expected: Qwen3VlStageDType,
    shape: &[usize],
    float_policy: Qwen3VlArtifactFloatPolicy,
) -> Result<(), Qwen3VlArtifactError> {
    if float_policy == Qwen3VlArtifactFloatPolicy::PackedQ4sBlock128WeightsF32Auxiliaries {
        if is_q4s_block128_matrix_shape(shape) && is_packed_q4s_block128_f32(actual) {
            return Ok(());
        }
        // The sealed Q4 release stores source F16 for vision tensors that are not eligible for
        // block-128 packing. The load adapter widens those tensors to the released F32 vision
        // execution dtype before applying them to the module. Validate that explicit conversion
        // policy here without weakening Preserve or the ordinary packed-F16 release contract.
        if expected == Qwen3VlStageDType::F32 && actual == DType::F16 {
            return Ok(());
        }
    }
    validate_dtype(stage, name, actual, expected)
}

fn expected_row_payload_bytes(
    spec: &RowChunkSpec,
    dtype: DType,
) -> Result<usize, Qwen3VlArtifactError> {
    let elements = spec.rows().checked_mul(spec.hidden_size).ok_or_else(|| {
        contract(
            "qwen-embedding",
            "row tensor element count overflowed usize",
        )
    })?;
    match dtype {
        DType::F16 => elements.checked_mul(2).ok_or_else(|| {
            contract(
                "qwen-embedding",
                "row tensor F16 payload byte count overflowed usize",
            )
        }),
        dtype if is_packed_q4s_block128_f32(dtype) && elements.is_multiple_of(128) => elements
            .checked_div(2)
            .and_then(|packed| {
                elements
                    .checked_div(128)
                    .and_then(|blocks| blocks.checked_mul(4))
                    .and_then(|scales| packed.checked_add(scales))
            })
            .ok_or_else(|| {
                contract(
                    "qwen-embedding",
                    "row tensor Q4S payload byte count overflowed usize",
                )
            }),
        other => Err(contract(
            "qwen-embedding",
            format!("row tensor has unsupported validated dtype {other:?}"),
        )),
    }
}

fn validate_dtype(
    stage: &str,
    name: &str,
    actual: DType,
    expected: Qwen3VlStageDType,
) -> Result<(), Qwen3VlArtifactError> {
    let expected = match expected {
        Qwen3VlStageDType::F16 => DType::F16,
        Qwen3VlStageDType::F32 => DType::F32,
    };
    if actual == expected {
        Ok(())
    } else {
        Err(contract(
            stage,
            format!("tensor {name} dtype mismatch: expected {expected:?}, got {actual:?}"),
        ))
    }
}

fn rename_snapshot(snapshot: &TensorSnapshot, name: &str) -> TensorSnapshot {
    TensorSnapshot::from_closure(
        snapshot.clone_data_fn(),
        snapshot.dtype,
        snapshot.shape.clone(),
        name.split('.').map(str::to_owned).collect(),
        snapshot.container_stack.clone().unwrap_or_default(),
        snapshot.tensor_id.unwrap_or_default(),
    )
}

fn validate_apply(
    stage: &str,
    result: &ApplyResult,
    expected: &BTreeSet<String>,
) -> Result<(), Qwen3VlArtifactError> {
    if result.applied.is_empty()
        || !result.skipped.is_empty()
        || !result.unused.is_empty()
        || !result.errors.is_empty()
    {
        return Err(contract(
            stage,
            format!(
                "apply failed: applied={:?}, skipped={:?}, unused={:?}, errors={:?}",
                result.applied, result.skipped, result.unused, result.errors
            ),
        ));
    }
    let actual = result.applied.iter().cloned().collect::<BTreeSet<_>>();
    if actual != *expected || actual.len() != result.applied.len() {
        return Err(contract(
            stage,
            format!("applied path mismatch: expected={expected:?}, actual={actual:?}"),
        ));
    }
    Ok(())
}

fn quantize_q4s_block128_f32(data: TensorData) -> Result<TensorData, TensorSnapshotError> {
    const BLOCK: usize = 128;
    const VALUES_PER_WORD: usize = 8;

    if !matches!(data.dtype, DType::F16 | DType::F32) {
        return Err(TensorSnapshotError::DataError(format!(
            "Qwen runtime Q4S quantization requires F16/F32, found {:?}",
            data.dtype
        )));
    }
    let shape = data.shape.to_vec();
    let Some(&inner) = shape.last() else {
        return Err(TensorSnapshotError::DataError(
            "Qwen runtime Q4S tensor has no dimensions".into(),
        ));
    };
    if !inner.is_multiple_of(BLOCK) {
        return Err(TensorSnapshotError::DataError(format!(
            "Qwen runtime Q4S innermost dimension {inner} is not divisible by {BLOCK}"
        )));
    }
    let values = data
        .convert_dtype(DType::F32)
        .to_vec::<f32>()
        .map_err(|error| TensorSnapshotError::DataError(error.to_string()))?;
    let mut packed = Vec::with_capacity(values.len() / VALUES_PER_WORD);
    let mut scales = Vec::with_capacity(values.len() / BLOCK);
    for block in values.as_chunks::<BLOCK>().0 {
        if block.iter().any(|value| !value.is_finite()) {
            return Err(TensorSnapshotError::DataError(
                "cannot quantize a non-finite Qwen checkpoint value".into(),
            ));
        }
        let alpha = block
            .iter()
            .fold(0.0_f32, |value, element| value.max(element.abs()));
        let scale = if alpha == 0.0 {
            f32::MIN_POSITIVE
        } else {
            alpha / 7.0
        };
        scales.push(scale);
        let inverse = scale.recip();
        for values in block.as_chunks::<VALUES_PER_WORD>().0 {
            let mut word = 0_u32;
            for (lane, value) in values.iter().enumerate() {
                let quantized = (value * inverse).round().clamp(-7.0, 7.0) as i8;
                word |= u32::from((quantized as u8) & 0x0f) << (lane * 4);
            }
            packed.push(word);
        }
    }
    let scheme = QuantScheme::default()
        .with_value(QuantValue::Q4S)
        .with_level(QuantLevel::block([BLOCK as u8]))
        .with_param(QuantParam::F32)
        .with_store(QuantStore::PackedU32(0));
    let mut bytes = Vec::with_capacity(values.len() / 2 + scales.len() * 4);
    for word in packed {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    for scale in scales {
        bytes.extend_from_slice(&scale.to_le_bytes());
    }
    Ok(TensorData::from_bytes(
        Bytes::from_bytes_vec(bytes),
        shape,
        DType::QFloat(scheme),
    ))
}

#[derive(Debug, Clone, Copy)]
struct FloatAdapter(Qwen3VlArtifactFloatPolicy);

impl ModuleAdapter for FloatAdapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        if self.0 == Qwen3VlArtifactFloatPolicy::PackedQ4sBlock128WeightsF32Auxiliaries
            && matches!(snapshot.dtype, DType::F16 | DType::F32)
            && is_q4s_block128_matrix_shape(&snapshot.shape)
        {
            let data = snapshot.clone_data_fn();
            return TensorSnapshot::from_closure(
                Rc::new(move || quantize_q4s_block128_f32(data()?)),
                DType::QFloat(
                    QuantScheme::default()
                        .with_value(QuantValue::Q4S)
                        .with_level(QuantLevel::block([128]))
                        .with_param(QuantParam::F32)
                        .with_store(QuantStore::PackedU32(0)),
                ),
                snapshot.shape.clone(),
                snapshot.path_stack.clone().unwrap_or_default(),
                snapshot.container_stack.clone().unwrap_or_default(),
                snapshot.tensor_id.unwrap_or_default(),
            );
        }
        if snapshot.dtype != DType::F16 {
            return snapshot.clone();
        }
        let output_dtype = match self.0 {
            Qwen3VlArtifactFloatPolicy::Preserve => return snapshot.clone(),
            Qwen3VlArtifactFloatPolicy::AdaptToF32 => DType::F32,
            Qwen3VlArtifactFloatPolicy::PackedF16WeightsF32Auxiliaries
                if snapshot.shape.len() == 2 =>
            {
                return snapshot.clone();
            }
            Qwen3VlArtifactFloatPolicy::PackedF16WeightsF32Auxiliaries => DType::F32,
            // Unaligned matrices, Conv3d weights, norms, and biases in the sealed Q4 profile are
            // the explicit F32 fallback. Eligible matrices returned from the branch above.
            Qwen3VlArtifactFloatPolicy::PackedQ4sBlock128WeightsF32Auxiliaries => DType::F32,
        };
        let data = snapshot.clone_data_fn();
        TensorSnapshot::from_closure(
            Rc::new(move || Ok(data()?.convert_dtype(output_dtype))),
            output_dtype,
            snapshot.shape.clone(),
            snapshot.path_stack.clone().unwrap_or_default(),
            snapshot.container_stack.clone().unwrap_or_default(),
            snapshot.tensor_id.unwrap_or_default(),
        )
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(*self)
    }
}

fn load_adapter(policy: Qwen3VlArtifactFloatPolicy) -> Option<Box<dyn ModuleAdapter>> {
    (policy != Qwen3VlArtifactFloatPolicy::Preserve)
        .then(|| Box::new(FloatAdapter(policy)) as Box<dyn ModuleAdapter>)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{backend::Flex, module::ParamId, tensor::TensorData};
    use burn_image::{
        ArtifactBundleId, ArtifactPath, ArtifactProfileId, ModelId, NumericFormat, Sha256Digest,
    };
    use burn_store::{BurnpackWriter, TensorSnapshot};

    fn identity_manifest() -> ArtifactManifest {
        ArtifactManifest {
            schema_version: ARTIFACT_MANIFEST_SCHEMA_V1,
            bundle: ArtifactBundleId::new(QWEN_COMPONENT_BUNDLE_ID).unwrap(),
            profile: ArtifactProfileId::new(QWEN_BASE_CONDITIONING_PROFILE).unwrap(),
            model: ModelId::new(QWEN_COMPONENT_MODEL_ID).unwrap(),
            model_revision: QWEN_COMPONENT_MODEL_REVISION.into(),
            numeric_format: NumericFormat::Other(QWEN_BASE_CONDITIONING_PROFILE.into()),
            components: Vec::new(),
            files: Vec::new(),
            dependencies: Vec::new(),
            metadata: BTreeMap::new(),
            content_digest: Some(Sha256Digest::from_hex(QWEN_COMPONENT_CONTENT_DIGEST).unwrap()),
        }
    }

    fn metadata_manifest() -> ArtifactManifest {
        let mut manifest = identity_manifest();
        manifest.metadata = QWEN_METADATA_VALUES
            .into_iter()
            .map(|(key, value)| (key.into(), value.into()))
            .collect();
        manifest.files = QWEN_METADATA_PATHS
            .into_iter()
            .map(|path| ArtifactFile {
                path: ArtifactPath::new(path).unwrap(),
                size: 1,
                sha256: Sha256Digest::calculate(b"x"),
                role: ArtifactFileRole::Metadata,
                component: None,
                shard: None,
            })
            .collect();
        manifest
    }

    #[test]
    fn stable_stage_names_and_rows_correctness() {
        assert_eq!(
            qwen_streaming_stage_name(&Qwen3VlStage::VisionBlock { index: 7 }),
            "qwen-vision-block-07"
        );
        let spec = RowChunkSpec {
            chunk_index: 2,
            row_range: 10..20,
            total_rows: 20,
            hidden_size: 4,
            element_bytes: 2,
        };
        assert_eq!(
            qwen_row_slice_target("model.language_model.embed_tokens.weight", &spec),
            "model.language_model.embed_tokens.weight.rows.02.000010-000020"
        );
    }

    #[test]
    fn row_object_float_policy_uploads_directly_with_exact_values_correctness() {
        let spec = RowChunkSpec {
            chunk_index: 0,
            row_range: 0..2,
            total_rows: 2,
            hidden_size: 3,
            element_bytes: 2,
        };
        let target = qwen_row_slice_target("model.language_model.embed_tokens.weight", &spec);
        let values = [-2.0_f32, -0.5, 0.0, 0.25, 1.0, 4.0];
        let data = TensorData::new(values.to_vec(), [spec.rows(), spec.hidden_size])
            .convert_dtype(DType::F16);
        let snapshot =
            TensorSnapshot::from_data(data, vec![target.clone()], Vec::new(), ParamId::new());
        let bytes = BurnpackWriter::new(vec![snapshot])
            .to_bytes()
            .unwrap()
            .to_vec();
        let file = ArtifactFile {
            path: ArtifactPath::new(format!("objects/{}.bpk", Sha256Digest::calculate(&bytes)))
                .unwrap(),
            size: bytes.len() as u64,
            sha256: Sha256Digest::calculate(&bytes),
            role: ArtifactFileRole::Weights,
            component: Some(ArtifactComponentId::new("qwen-embedding-rows-00").unwrap()),
            shard: None,
        };
        let device = Default::default();

        let preserved = parse_row_object::<Flex>(
            &file,
            bytes.clone(),
            &target,
            &spec,
            Qwen3VlArtifactFloatPolicy::Preserve,
            &device,
        )
        .unwrap();
        assert_eq!(preserved.dtype(), DType::F16);
        let preserved = preserved.into_data();
        assert_eq!(
            preserved.convert_dtype(DType::F32).to_vec::<f32>().unwrap(),
            values
        );

        let adapted = parse_row_object::<Flex>(
            &file,
            bytes,
            &target,
            &spec,
            Qwen3VlArtifactFloatPolicy::AdaptToF32,
            &device,
        )
        .unwrap();
        assert_eq!(adapted.dtype(), DType::F32);
        let adapted = adapted.into_data();
        assert_eq!(adapted.to_vec::<f32>().unwrap(), values);
    }

    #[test]
    fn packed_q4s_row_object_preserves_prequantized_release_storage_correctness() {
        let spec = RowChunkSpec {
            chunk_index: 0,
            row_range: 0..2,
            total_rows: 2,
            hidden_size: 128,
            element_bytes: 2,
        };
        let target = qwen_row_slice_target("model.language_model.embed_tokens.weight", &spec);
        let source = TensorData::new(
            (0..256)
                .map(|index| (index as f32 - 127.5) / 256.0)
                .collect::<Vec<_>>(),
            [spec.rows(), spec.hidden_size],
        )
        .convert_dtype(DType::F16);
        let packed = quantize_q4s_block128_f32(source).unwrap();
        let packed_bytes = packed.bytes.clone();
        let snapshot =
            TensorSnapshot::from_data(packed, vec![target.clone()], Vec::new(), ParamId::new());
        let bytes = BurnpackWriter::new(vec![snapshot])
            .to_bytes()
            .unwrap()
            .to_vec();
        let file = ArtifactFile {
            path: ArtifactPath::new(format!("objects/{}.bpk", Sha256Digest::calculate(&bytes)))
                .unwrap(),
            size: bytes.len() as u64,
            sha256: Sha256Digest::calculate(&bytes),
            role: ArtifactFileRole::Weights,
            component: Some(ArtifactComponentId::new("qwen-embedding-rows-00").unwrap()),
            shard: None,
        };

        let parsed = parse_row_object_data(
            &file,
            bytes.clone(),
            &target,
            &spec,
            Qwen3VlArtifactFloatPolicy::PackedQ4sBlock128WeightsF32Auxiliaries,
        )
        .unwrap();
        assert!(is_packed_q4s_block128_f32(parsed.dtype));
        assert_eq!(parsed.bytes, packed_bytes);
        assert_eq!(
            parsed.bytes.len(),
            256 / 2 + (256 / 128) * core::mem::size_of::<f32>()
        );
        let tensor = parse_row_object::<Flex>(
            &file,
            bytes.clone(),
            &target,
            &spec,
            Qwen3VlArtifactFloatPolicy::PackedQ4sBlock128WeightsF32Auxiliaries,
            &Default::default(),
        )
        .unwrap();
        assert!(matches!(
            tensor.dtype(),
            DType::QFloat(scheme)
                if scheme.value == QuantValue::Q4S
                    && scheme.level == QuantLevel::block([128])
                    && scheme.param == QuantParam::F32
                    && scheme.store == QuantStore::Native
        ));

        let error = parse_row_object_data(
            &file,
            bytes,
            &target,
            &spec,
            Qwen3VlArtifactFloatPolicy::Preserve,
        )
        .unwrap_err();
        assert!(error.to_string().contains("dtype mismatch: expected F16"));

        let wrong_scheme = DType::QFloat(
            QuantScheme::default()
                .with_value(QuantValue::Q4S)
                .with_level(QuantLevel::block([64]))
                .with_param(QuantParam::F32)
                .with_store(QuantStore::PackedU32(0)),
        );
        assert!(
            validate_artifact_dtype(
                "qwen-embedding",
                &target,
                wrong_scheme,
                Qwen3VlStageDType::F16,
                &[2, 128],
                Qwen3VlArtifactFloatPolicy::PackedQ4sBlock128WeightsF32Auxiliaries,
            )
            .is_err()
        );
    }

    #[test]
    fn packed_f16_adapter_retains_matrices_and_widens_auxiliaries_correctness() {
        let matrix = TensorSnapshot::from_data(
            TensorData::new(vec![0.25_f32; 8], [2, 4]).convert_dtype(DType::F16),
            vec!["weight".into()],
            Vec::new(),
            ParamId::new(),
        );
        let bias = TensorSnapshot::from_data(
            TensorData::new(vec![0.25_f32; 4], [4]).convert_dtype(DType::F16),
            vec!["bias".into()],
            Vec::new(),
            ParamId::new(),
        );
        let convolution = TensorSnapshot::from_data(
            TensorData::new(vec![0.25_f32; 8], [2, 1, 1, 2, 2]).convert_dtype(DType::F16),
            vec!["conv3d".into()],
            Vec::new(),
            ParamId::new(),
        );
        let adapter = FloatAdapter(Qwen3VlArtifactFloatPolicy::PackedF16WeightsF32Auxiliaries);

        assert_eq!(adapter.adapt(&matrix).dtype, DType::F16);
        assert_eq!(adapter.adapt(&bias).dtype, DType::F32);
        // The Qwen patch Conv3d remains an F32 auxiliary until a mixed-input Conv3d kernel is
        // admitted; keeping it explicit avoids silently routing an unsupported F16 activation.
        assert_eq!(adapter.adapt(&convolution).dtype, DType::F32);
    }

    #[test]
    fn packed_q4s_adapter_packs_matrix_and_widens_auxiliary_correctness() {
        use burn::tensor::quantization::QuantizedBytes;

        let matrix = TensorSnapshot::from_data(
            TensorData::new(
                (0..(128 * 2))
                    .map(|index| (index as f32 - 127.5) / 512.0)
                    .collect::<Vec<_>>(),
                [2, 128],
            )
            .convert_dtype(DType::F16),
            vec!["weight".into()],
            Vec::new(),
            ParamId::new(),
        );
        let bias = TensorSnapshot::from_data(
            TensorData::new(vec![0.25_f32; 2], [2]).convert_dtype(DType::F16),
            vec!["bias".into()],
            Vec::new(),
            ParamId::new(),
        );
        let unaligned_matrix = TensorSnapshot::from_data(
            TensorData::new(vec![0.25_f32; 2 * 129], [2, 129]).convert_dtype(DType::F16),
            vec!["unaligned".into()],
            Vec::new(),
            ParamId::new(),
        );
        let convolution = TensorSnapshot::from_data(
            TensorData::new(vec![0.25_f32; 8], [2, 1, 1, 2, 2]).convert_dtype(DType::F16),
            vec!["conv3d".into()],
            Vec::new(),
            ParamId::new(),
        );
        let adapter =
            FloatAdapter(Qwen3VlArtifactFloatPolicy::PackedQ4sBlock128WeightsF32Auxiliaries);
        let packed_snapshot = adapter.adapt(&matrix);
        let packed = (packed_snapshot.clone_data_fn())().unwrap();
        let DType::QFloat(scheme) = packed.dtype else {
            panic!("matrix must be QFloat");
        };
        let (values, qparams) = QuantizedBytes {
            bytes: packed.bytes.clone(),
            scheme,
            num_elements: 256,
        }
        .into_vec_i8();
        assert_eq!(values.len(), 256);
        assert_eq!(qparams.scales.len(), 2);
        assert_eq!(adapter.adapt(&bias).dtype, DType::F32);
        assert_eq!(adapter.adapt(&unaligned_matrix).dtype, DType::F32);
        assert_eq!(adapter.adapt(&convolution).dtype, DType::F32);

        let prepacked_snapshot =
            TensorSnapshot::from_data(packed, vec!["weight".into()], Vec::new(), ParamId::new());
        assert!(is_packed_q4s_block128_f32(
            adapter.adapt(&prepacked_snapshot).dtype
        ));
        let device = Default::default();
        let mut linear = crate::QwenLinearConfig::new(128, 2)
            .with_bias(false)
            .init::<Flex>(&device);
        let applied = linear.apply(
            vec![prepacked_snapshot],
            None,
            Some(Box::new(adapter)),
            false,
        );
        assert!(applied.errors.is_empty(), "{:?}", applied.errors);
        assert!(matches!(
            linear.weight.val().dtype(),
            DType::QFloat(scheme)
                if scheme.value == QuantValue::Q4S
                    && scheme.level == QuantLevel::block([128])
                    && scheme.param == QuantParam::F32
        ));
    }

    #[test]
    fn packed_q4s_contract_admits_widened_f16_vision_tensors_correctness() {
        for (name, shape) in [
            ("model.visual.patch_embed.proj.bias", vec![1152]),
            (
                "model.visual.patch_embed.proj.weight",
                vec![1152, 3, 2, 16, 16],
            ),
            (
                "model.visual.blocks.0.mlp.linear_fc2.weight",
                vec![1152, 4304],
            ),
        ] {
            validate_artifact_dtype(
                "qwen-vision-prelude",
                name,
                DType::F16,
                Qwen3VlStageDType::F32,
                &shape,
                Qwen3VlArtifactFloatPolicy::PackedQ4sBlock128WeightsF32Auxiliaries,
            )
            .unwrap();

            let preserve_error = validate_artifact_dtype(
                "qwen-vision-prelude",
                name,
                DType::F16,
                Qwen3VlStageDType::F32,
                &shape,
                Qwen3VlArtifactFloatPolicy::Preserve,
            )
            .unwrap_err();
            assert!(preserve_error.to_string().contains("expected F32, got F16"));
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    #[ignore = "requires an explicitly selected native WGPU adapter"]
    fn packed_q4s_column_layout_wgpu_module_reference() {
        use burn::tensor::{Tensor, TensorData, quantization::QuantizedBytes};

        type B = burn_wgpu::CubeBackend<burn_wgpu::WgpuRuntime, f32, i32, u32>;
        let device = burn_wgpu::WgpuDevice::default();
        let input_width = 128;
        let output_width = 256;
        let raw = (0..(output_width * input_width))
            .map(|index| {
                ((index * 37 % 509) as f32 - 254.0) / 1536.0
                    + ((index % input_width) as f32 - 63.5) / 8192.0
            })
            .collect::<Vec<_>>();
        let input = (0..input_width)
            .map(|index| ((index * 19 % 97) as f32 - 48.0) / 128.0)
            .collect::<Vec<_>>();
        let reference = (0..output_width)
            .map(|output| {
                raw[output * input_width..(output + 1) * input_width]
                    .iter()
                    .zip(&input)
                    .map(|(weight, input)| weight * input)
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        let quantized_data = quantize_q4s_block128_f32(
            TensorData::new(raw.clone(), [output_width, input_width]).convert_dtype(DType::F16),
        )
        .unwrap();
        let DType::QFloat(scheme) = quantized_data.dtype else {
            panic!("Q4S helper must return QFloat");
        };
        let (quantized_values, qparams) = QuantizedBytes {
            bytes: quantized_data.bytes,
            scheme,
            num_elements: raw.len(),
        }
        .into_vec_i8();
        let reconstructed = quantized_values
            .as_chunks::<128>()
            .0
            .iter()
            .zip(qparams.scales)
            .flat_map(|(block, scale)| block.iter().map(move |value| f32::from(*value) * scale))
            .collect::<Vec<_>>();
        let selected_rows = [2_i32, 0_i32];
        let selected = Tensor::<B, 2>::from_data(
            quantize_q4s_block128_f32(
                TensorData::new(raw.clone(), [output_width, input_width]).convert_dtype(DType::F16),
            )
            .unwrap(),
            &device,
        )
        .select(
            0,
            Tensor::<B, 1, burn::tensor::Int>::from_data(
                TensorData::new(selected_rows.to_vec(), [selected_rows.len()]),
                &device,
            ),
        )
        .dequantize()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
        let selected_reference = selected_rows
            .into_iter()
            .flat_map(|row| {
                reconstructed[row as usize * input_width..(row as usize + 1) * input_width]
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>();
        assert_eq!(selected.len(), selected_reference.len());
        assert!(
            selected
                .iter()
                .zip(selected_reference)
                .all(|(actual, expected)| (actual - expected).abs() < 1.0e-6),
            "packed-Q4 embedding row selection must widen only the selected rows"
        );
        let quantized_reference = (0..output_width)
            .map(|output| {
                reconstructed[output * input_width..(output + 1) * input_width]
                    .iter()
                    .zip(&input)
                    .map(|(weight, input)| weight * input)
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        let snapshot = TensorSnapshot::from_data(
            TensorData::new(raw, [output_width, input_width]).convert_dtype(DType::F16),
            vec!["weight".into()],
            Vec::new(),
            ParamId::new(),
        );
        let mut linear = crate::QwenLinearConfig::new(input_width, output_width)
            .with_bias(false)
            .init::<B>(&device);
        let applied = linear.apply(
            vec![snapshot],
            None,
            Some(Box::new(FloatAdapter(
                Qwen3VlArtifactFloatPolicy::PackedQ4sBlock128WeightsF32Auxiliaries,
            ))),
            false,
        );
        assert!(applied.errors.is_empty(), "{:?}", applied.errors);
        let actual = crate::linear::qwen_linear_forward(
            &linear,
            Tensor::<B, 2>::from_data(TensorData::new(input.clone(), [1, input_width]), &device),
        )
        .into_data()
        .to_vec::<f32>()
        .unwrap();
        let actual_rank3 = crate::linear::qwen_linear_forward(
            &linear,
            Tensor::<B, 3>::from_data(TensorData::new(input, [1, 1, input_width]), &device),
        )
        .into_data()
        .to_vec::<f32>()
        .unwrap();
        assert_eq!(actual_rank3.len(), actual.len());
        assert!(
            actual_rank3
                .iter()
                .zip(&actual)
                .all(|(rank3, rank2)| (rank3 - rank2).abs() < 1.0e-5)
        );
        let error = actual
            .iter()
            .zip(&reference)
            .map(|(actual, expected)| f64::from(actual - expected).powi(2))
            .sum::<f64>();
        let signal = reference
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>();
        let dot = actual
            .iter()
            .zip(&reference)
            .map(|(actual, expected)| f64::from(*actual) * f64::from(*expected))
            .sum::<f64>();
        let actual_norm = actual
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>()
            .sqrt();
        let relative_rmse = (error / signal).sqrt();
        let cosine = dot / (actual_norm * signal.sqrt());
        let kernel_relative_rmse = {
            let error = actual
                .iter()
                .zip(&quantized_reference)
                .map(|(actual, expected)| f64::from(actual - expected).powi(2))
                .sum::<f64>();
            let signal = quantized_reference
                .iter()
                .map(|value| f64::from(*value).powi(2))
                .sum::<f64>();
            (error / signal).sqrt()
        };
        eprintln!(
            "Qwen column-layout WGPU packed-Q4S: rel-RMSE={relative_rmse} cosine={cosine} kernel-rel-RMSE={kernel_relative_rmse}"
        );
        assert!(actual.iter().all(|value| value.is_finite()));
        assert!(
            kernel_relative_rmse < 1.0e-4,
            "quantized kernel relative RMSE {kernel_relative_rmse}"
        );
        assert!(relative_rmse < 0.12, "relative RMSE {relative_rmse}");
        assert!(cosine > 0.993, "cosine {cosine}");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    #[ignore = "requires BURN_QWEN3_VL_HF_SNAPSHOT and an explicitly selected native WGPU adapter"]
    fn released_qwen_k_projection_packed_q4s_wgpu_module_reference() {
        use std::{
            fs::File,
            io::{Read, Seek, SeekFrom},
            path::PathBuf,
        };

        use burn::tensor::{DType, Tensor, TensorData};

        const SOURCE_OFFSET: u64 = 1_546_680_424;
        const OUTPUT_WIDTH: usize = 1_024;
        const INPUT_WIDTH: usize = 4_096;
        const SOURCE_BYTES: usize = OUTPUT_WIDTH * INPUT_WIDTH * 2;

        type B = burn_wgpu::CubeBackend<burn_wgpu::WgpuRuntime, f32, i32, u32>;
        let snapshot = PathBuf::from(
            std::env::var_os("BURN_QWEN3_VL_HF_SNAPSHOT")
                .expect("set BURN_QWEN3_VL_HF_SNAPSHOT to the pinned Turbo snapshot"),
        );
        let mut file = File::open(snapshot.join("mllm/model-00001-of-00004.safetensors"))
            .expect("open pinned Qwen shard");
        file.seek(SeekFrom::Start(SOURCE_OFFSET))
            .expect("seek pinned Qwen tensor");
        let mut bytes = vec![0_u8; SOURCE_BYTES];
        file.read_exact(&mut bytes)
            .expect("read pinned Qwen tensor");
        let weights = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16))
            .collect::<Vec<_>>();
        assert!(weights.iter().all(|value| value.is_finite()));

        let input = (0..INPUT_WIDTH)
            .map(|index| ((index * 19 % 251) as f32 - 125.0) / 256.0)
            .collect::<Vec<_>>();
        let reference = weights
            .as_chunks::<INPUT_WIDTH>()
            .0
            .iter()
            .map(|row| {
                row.iter()
                    .zip(&input)
                    .map(|(weight, input)| weight * input)
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        let tensor = TensorSnapshot::from_data(
            TensorData::new(weights, [OUTPUT_WIDTH, INPUT_WIDTH]).convert_dtype(DType::F16),
            vec!["weight".into()],
            Vec::new(),
            ParamId::new(),
        );
        let device = burn_wgpu::WgpuDevice::default();
        let mut linear = crate::QwenLinearConfig::new(INPUT_WIDTH, OUTPUT_WIDTH)
            .with_bias(false)
            .init::<B>(&device);
        let applied = linear.apply(
            vec![tensor],
            None,
            Some(Box::new(FloatAdapter(
                Qwen3VlArtifactFloatPolicy::PackedQ4sBlock128WeightsF32Auxiliaries,
            ))),
            false,
        );
        assert!(applied.errors.is_empty(), "{:?}", applied.errors);
        let actual = crate::linear::qwen_linear_forward(
            &linear,
            Tensor::<B, 2>::from_data(TensorData::new(input, [1, INPUT_WIDTH]), &device),
        )
        .into_data()
        .to_vec::<f32>()
        .unwrap();
        let squared_error = actual
            .iter()
            .zip(&reference)
            .map(|(actual, expected)| f64::from(actual - expected).powi(2))
            .sum::<f64>();
        let signal = reference
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>();
        let dot = actual
            .iter()
            .zip(&reference)
            .map(|(actual, expected)| f64::from(*actual) * f64::from(*expected))
            .sum::<f64>();
        let actual_norm = actual
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>()
            .sqrt();
        let relative_rmse = (squared_error / signal).sqrt();
        let cosine = dot / (actual_norm * signal.sqrt());
        eprintln!(
            "released Qwen block-00 k_proj packed-Q4S: rel-RMSE={relative_rmse} cosine={cosine}"
        );
        assert!(actual.iter().all(|value| value.is_finite()));
        assert!(relative_rmse < 0.15, "relative RMSE {relative_rmse}");
        assert!(cosine > 0.985, "cosine {cosine}");

        const BENCHMARK_ROWS: usize = 128;
        const BENCHMARK_REPEATS: usize = 3;
        let benchmark_input = (0..BENCHMARK_ROWS * INPUT_WIDTH)
            .map(|index| ((index * 19 % 251) as f32 - 125.0) / 256.0)
            .collect::<Vec<_>>();
        let benchmark_input = Tensor::<B, 2>::from_data(
            TensorData::new(benchmark_input, [BENCHMARK_ROWS, INPUT_WIDTH]),
            &device,
        );
        let warmup =
            crate::linear::qwen_linear_forward(&linear, benchmark_input.clone()).into_data();
        assert_eq!(warmup.dtype, DType::F32);
        let started = std::time::Instant::now();
        for _ in 0..BENCHMARK_REPEATS {
            let output =
                crate::linear::qwen_linear_forward(&linear, benchmark_input.clone()).into_data();
            assert_eq!(output.shape, [BENCHMARK_ROWS, OUTPUT_WIDTH].into());
            assert_eq!(output.dtype, DType::F32);
        }
        let elapsed = started.elapsed();
        eprintln!(
            "released Qwen block-00 k_proj packed-Q4S benchmark: rows={BENCHMARK_ROWS} repeats={BENCHMARK_REPEATS} total_ms={} mean_ms={:.3}",
            elapsed.as_millis(),
            elapsed.as_secs_f64() * 1_000.0 / BENCHMARK_REPEATS as f64,
        );
    }

    #[test]
    #[ignore = "requires BURN_QWEN3_VL_COMPONENT_ROOT pointing at the released component bundle"]
    fn released_artifact_host_selected_rows_digest_reference() {
        let root = std::env::var("BURN_QWEN3_VL_COMPONENT_ROOT")
            .expect("set BURN_QWEN3_VL_COMPONENT_ROOT to the released Qwen component directory");
        let directory = VerifiedArtifactDirectory::open(root.clone()).unwrap();
        let config_bytes =
            std::fs::read(Path::new(&root).join("metadata/source/mllm/config.json")).unwrap();
        let config = Qwen3VlConfig::from_json(std::str::from_utf8(&config_bytes).unwrap()).unwrap();
        let contract =
            Qwen3VlComponentContract::released_base(directory.manifest().clone(), config.clone())
                .unwrap();
        let ids = vec![vec![
            151_644, 8_948, 198, 2_610, 525, 264, 10_950, 17_847, 429, 26_885, 1_550, 22_092,
            5_335, 3_118, 389, 1_196, 11_221, 13, 576, 11_221, 525, 438, 11_017, 13, 151_645, 198,
            151_644, 872, 198, 32, 14_029, 10_300, 315, 264, 6_303, 42_024, 11_958, 389, 264,
            14_396, 4_158, 1_965, 13, 151_645, 198,
        ]];
        let mut state = HostRoutedF16EmbeddingState::new(
            &ids,
            config.text_config.vocab_size,
            config.text_config.hidden_size,
        )
        .unwrap();
        let mut reader = directory.shard_reader().unwrap();
        for spec in &contract.plan.embedding_rows.chunks {
            let stage = Qwen3VlStage::EmbeddingRows {
                chunk: spec.chunk_index,
            };
            let files = contract.files(&stage);
            assert_eq!(files.len(), 1);
            let file = &files[0];
            let bytes = VerifiedArtifactBytes::unverified(reader.read_shard(file).unwrap())
                .into_verified_bytes(file, contract.max_shard_bytes())
                .unwrap();
            let object_bytes = bytes.len() as u64;
            let target = qwen_row_slice_target("model.language_model.embed_tokens.weight", spec);
            let data = parse_row_object_data(
                file,
                bytes,
                &target,
                spec,
                Qwen3VlArtifactFloatPolicy::Preserve,
            )
            .unwrap();
            state.apply_chunk_data(spec, &data, object_bytes).unwrap();
        }
        let (data, report) = state.finish().unwrap();
        assert_eq!(data.shape.as_slice(), [1, 45, 4096]);
        assert_eq!(report.unique_token_count, 33);
        assert_eq!(report.authenticated_object_count, 6);
        assert_eq!(report.authenticated_object_bytes, 1_244_662_784);
        assert_eq!(report.authenticated_f16_payload_bytes, 1_244_659_712);
        assert_eq!(report.selected_f16_bytes, 368_640);
        assert_eq!(report.host_f32_payload_bytes, 737_280);
        assert_eq!(
            report.host_f32_sha256,
            "a6aa0501f1d6f5a622934ee10a64b526843f723937f6d5abd96058b29ea8b6fe"
        );
        assert!(report.all_finite && report.not_all_zero && report.coverage_complete);
    }

    #[test]
    #[ignore = "requires BURN_QWEN3_VL_Q4_COMPONENT_ROOT pointing at the released Q4 component bundle"]
    fn released_packed_q4s_embedding_text_and_vision_stage_load_reference() {
        let root = std::env::var("BURN_QWEN3_VL_Q4_COMPONENT_ROOT").expect(
            "set BURN_QWEN3_VL_Q4_COMPONENT_ROOT to the released Q4 Qwen component directory",
        );
        let directory = VerifiedArtifactDirectory::open(root.clone()).unwrap();
        let config_bytes =
            std::fs::read(Path::new(&root).join("metadata/source/mllm/config.json")).unwrap();
        let config = Qwen3VlConfig::from_json(std::str::from_utf8(&config_bytes).unwrap()).unwrap();
        let contract =
            Qwen3VlComponentContract::released_base(directory.manifest().clone(), config).unwrap();
        let first_embedding = contract.plan.embedding_rows.chunks[0].clone();
        let reader = directory.shard_reader().unwrap();
        let device = Default::default();
        let mut source =
            VerifiedBurnpackQwen3VlStageSource::<Flex, _>::new(contract, device, reader)
                .with_float_policy(
                    Qwen3VlArtifactFloatPolicy::PackedQ4sBlock128WeightsF32Auxiliaries,
                );

        let embedding = source.load_embedding_rows(&first_embedding).unwrap();
        assert!(matches!(
            embedding.weight.dtype(),
            DType::QFloat(scheme)
                if scheme.value == QuantValue::Q4S
                    && scheme.level == QuantLevel::block([128])
                    && scheme.param == QuantParam::F32
        ));
        drop(embedding);
        drop(source.load_text_block(0).unwrap());
        drop(source.load_vision_prelude().unwrap());
        drop(source.load_vision_block(0).unwrap());
        drop(source.load_vision_deepstack_merger(0).unwrap());
        drop(source.load_vision_final_merger().unwrap());
    }

    #[test]
    fn component_identity_fails_closed_correctness() {
        let manifest = identity_manifest();
        validate_identity(&manifest).unwrap();

        let mut packed_q4s = manifest.clone();
        packed_q4s.bundle = ArtifactBundleId::new(QWEN_Q4S_COMPONENT_BUNDLE_ID).unwrap();
        packed_q4s.profile = ArtifactProfileId::new(QWEN_Q4S_BASE_CONDITIONING_PROFILE).unwrap();
        packed_q4s.numeric_format = NumericFormat::Other(QWEN_Q4S_BASE_CONDITIONING_PROFILE.into());
        packed_q4s.content_digest =
            Some(Sha256Digest::from_hex(QWEN_Q4S_COMPONENT_CONTENT_DIGEST).unwrap());
        validate_identity(&packed_q4s).unwrap();

        let mut wrong_bundle = manifest.clone();
        wrong_bundle.bundle = ArtifactBundleId::new("qwen-unpinned").unwrap();
        assert!(validate_identity(&wrong_bundle).is_err());

        let mut wrong_model = manifest.clone();
        wrong_model.model = ModelId::new("Example/Qwen").unwrap();
        assert!(validate_identity(&wrong_model).is_err());

        let mut wrong_digest = manifest.clone();
        wrong_digest.content_digest = Some(Sha256Digest::calculate(b"wrong qwen component"));
        assert!(validate_identity(&wrong_digest).is_err());

        let mut wrong_revision = manifest;
        wrong_revision.model_revision = "0".repeat(64);
        assert!(validate_identity(&wrong_revision).is_err());

        let mut wrong_profile = identity_manifest();
        wrong_profile.profile = ArtifactProfileId::new("f32").unwrap();
        assert!(validate_identity(&wrong_profile).is_err());
    }

    #[test]
    fn base_conditioning_rejects_lm_head_plan_correctness() {
        let config = crate::config::tiny_config();
        let with_lm_head = Qwen3VlStreamingPlan::released_f16(&config, true).unwrap();
        let error = validate_base_plan(&config, &with_lm_head).unwrap_err();
        assert!(error.to_string().contains("omit the LM head"));
    }

    #[test]
    fn component_metadata_layout_and_counts_fail_closed_correctness() {
        let manifest = metadata_manifest();
        validate_metadata(&manifest).unwrap();

        let mut wrong_count = manifest.clone();
        wrong_count
            .metadata
            .insert("stored_tensor_count".into(), "753".into());
        assert!(validate_metadata(&wrong_count).is_err());

        let mut extra = manifest;
        extra.files.push(ArtifactFile {
            path: ArtifactPath::new("metadata/source/transformer/config.json").unwrap(),
            size: 1,
            sha256: Sha256Digest::calculate(b"y"),
            role: ArtifactFileRole::Metadata,
            component: None,
            shard: None,
        });
        assert!(validate_metadata(&extra).is_err());
    }
}
