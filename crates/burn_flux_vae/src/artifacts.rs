//! Sealed, bounded Burnpack loading for standalone FLUX-compatible VAE bundles.
//!
//! The crate owns the encoder/decoder tensor split, strict inventory checks, and optional
//! device-resident retention. Filesystem, HTTP, and browser cache implementations remain in the
//! model-neutral [`burn_image`] artifact layer.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    rc::Rc,
};

use burn::{
    prelude::Backend,
    tensor::{Bytes, DType},
};
use burn_image::{
    ARTIFACT_MANIFEST_SCHEMA_V1, ArtifactBundleId, ArtifactComponentId, ArtifactDependency,
    ArtifactFile, ArtifactFileRole, ArtifactManifest, ArtifactProfileId, ArtifactReadError,
    ArtifactShardReader, AsyncArtifactShardReader, DirectoryArtifactShardReader, ModelId,
    Sha256Digest, VerifiedArtifactBytes, VerifiedArtifactDirectory,
};
use burn_store::{
    ApplyResult, BurnpackStore, ModuleAdapter, ModuleSnapshot, ModuleStore, TensorSnapshot,
};
use thiserror::Error;

use crate::{AutoencoderKl, AutoencoderKlConfig, TensorInventory, TensorSpec};

const FLUX_VAE_METADATA_VALUES: [(&str, &str); 17] = [
    ("component_bundle", "true"),
    ("component_kind", "flux1-vae"),
    ("artifact_layout", "semantic-burnpack-v1"),
    ("owner", "flux-vae"),
    ("tensor_inventory_schema", "2"),
    ("tensor_count", "244"),
    ("stored_tensor_count", "244"),
    ("tensor_inventory_entries", "244"),
    ("omitted_tensor_count", "0"),
    ("physical_shards_bounded", "true"),
    ("target_max_shard_bytes", "268435456"),
    ("transport_layout_path", "metadata/transport-layout.json"),
    ("transport_layout_schema", "1"),
    ("transport_parts_required", "true"),
    ("transport_part_target_bytes", "20971520"),
    ("target_max_transport_shard_bytes", "25000000"),
    ("semantic_object_max_bytes", "268435456"),
];

const FLUX_VAE_METADATA_PATHS: [&str; 4] = [
    "metadata/tensor-inventory.json",
    "metadata/source-files.json",
    "metadata/source/vae/config.json",
    "metadata/transport-layout.json",
];

pub const FLUX_VAE_ENCODER_STAGE: &str = "flux-vae-encoder";
pub const FLUX_VAE_DECODER_STAGE: &str = "flux-vae-decoder";
pub const FLUX_VAE_COMPONENT_ROLE: &str = "vae";
pub const FLUX_VAE_COMPONENT_BUNDLE_ID: &str = "flux1-vae-boogu-image-0.1";
pub const FLUX_VAE_COMPONENT_PROFILE: &str = "f16";
pub const FLUX_VAE_COMPONENT_MODEL_ID: &str = "BooguDerived/FLUX1-VAE-0.1";
/// SHA-256 of the canonical sorted declaration for the shared upstream VAE source file.
pub const FLUX_VAE_COMPONENT_MODEL_REVISION: &str =
    "5f9271cca82f45ef89910f1a5a4a775745dca788f518d25d93afe5bae9e6b8b8";
/// Exact sealed digest of the canonical reusable FLUX VAE component manifest.
pub const FLUX_VAE_COMPONENT_CONTENT_DIGEST: &str =
    "a7a4758d3334bf3c2749cc9e84bed748fd0dc9b982299708748e1343b08efab9";

/// Construct the complete immutable dependency pin for the released FLUX VAE component.
pub fn flux_vae_component_dependency() -> ArtifactDependency {
    ArtifactDependency {
        role: ArtifactComponentId::new(FLUX_VAE_COMPONENT_ROLE).expect("static role is valid"),
        bundle: ArtifactBundleId::new(FLUX_VAE_COMPONENT_BUNDLE_ID)
            .expect("static bundle is valid"),
        profile: ArtifactProfileId::new(FLUX_VAE_COMPONENT_PROFILE)
            .expect("static profile is valid"),
        model: ModelId::new(FLUX_VAE_COMPONENT_MODEL_ID).expect("static model id is valid"),
        model_revision: FLUX_VAE_COMPONENT_MODEL_REVISION.to_owned(),
        content_digest: Sha256Digest::from_hex(FLUX_VAE_COMPONENT_CONTENT_DIGEST)
            .expect("static digest is valid"),
    }
}

/// Explicit execution dtype policy after the F16 storage contract has been authenticated.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FluxVaeArtifactFloatPolicy {
    /// Preserve F16 weights on a backend and execution path proven for them.
    Preserve,
    /// Apply one bounded stage at a time as F32, matching FLUX `force_upcast` behavior.
    #[default]
    AdaptToF32,
}

#[derive(Debug, Error)]
pub enum FluxVaeArtifactError {
    #[error("invalid FLUX VAE component manifest: {0}")]
    Manifest(String),
    #[error(transparent)]
    Read(#[from] ArtifactReadError),
    #[error("invalid Burnpack object {path}: {message}")]
    Burnpack { path: String, message: String },
    #[error("FLUX VAE artifact contract failed for {stage}: {message}")]
    Contract { stage: String, message: String },
    #[error("FLUX VAE initialization failed: {0}")]
    Model(String),
    #[error("device synchronization after FLUX VAE stage failed: {0}")]
    Synchronize(String),
}

fn contract(stage: impl Into<String>, message: impl Into<String>) -> FluxVaeArtifactError {
    FluxVaeArtifactError::Contract {
        stage: stage.into(),
        message: message.into(),
    }
}

/// Sealed standalone VAE manifest paired with the exact config-derived tensor inventory.
#[derive(Clone)]
pub struct FluxVaeComponentContract {
    manifest: ArtifactManifest,
    config: AutoencoderKlConfig,
    stages: BTreeMap<String, Vec<ArtifactFile>>,
    expected: BTreeMap<String, BTreeMap<String, TensorSpec>>,
    max_shard_bytes: u64,
}

impl FluxVaeComponentContract {
    pub fn new(
        manifest: ArtifactManifest,
        config: AutoencoderKlConfig,
    ) -> Result<Self, FluxVaeArtifactError> {
        manifest
            .validate_sealed()
            .map_err(|error| FluxVaeArtifactError::Manifest(error.to_string()))?;
        validate_identity(&manifest)?;
        validate_metadata(&manifest)?;
        let inventory = TensorInventory::from_config(&config)
            .map_err(|error| FluxVaeArtifactError::Model(error.to_string()))?;
        if inventory.tensors.len() != 244 {
            return Err(contract(
                "flux-vae",
                format!(
                    "config is not the canonical 244-tensor FLUX.1 VAE inventory: found {}",
                    inventory.tensors.len()
                ),
            ));
        }
        let mut expected = BTreeMap::from([
            (FLUX_VAE_ENCODER_STAGE.to_owned(), BTreeMap::new()),
            (FLUX_VAE_DECODER_STAGE.to_owned(), BTreeMap::new()),
        ]);
        for spec in inventory.tensors {
            let stage = stage_for_tensor(&spec);
            expected
                .get_mut(stage)
                .expect("both stages initialized")
                .insert(spec.burn_name.clone(), spec);
        }
        if expected.values().any(BTreeMap::is_empty) {
            return Err(contract("flux-vae", "config produced an empty VAE half"));
        }

        let required = BTreeSet::from([
            FLUX_VAE_ENCODER_STAGE.to_owned(),
            FLUX_VAE_DECODER_STAGE.to_owned(),
        ]);
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
                "flux-vae",
                format!(
                    "component declarations differ from the exact required stage set: expected={required:?}, actual={declared:?}"
                ),
            ));
        }

        let max_shard_bytes = declared_max_shard_bytes(&manifest)?;
        let mut stages = BTreeMap::<String, Vec<ArtifactFile>>::new();
        for file in manifest
            .files
            .iter()
            .filter(|file| file.role == ArtifactFileRole::Weights)
        {
            let stage = file.component.as_ref().ok_or_else(|| {
                contract(
                    "flux-vae",
                    format!("weight object {} has no component", file.path),
                )
            })?;
            if !expected.contains_key(stage.as_str()) {
                return Err(contract(
                    stage.as_str(),
                    format!("weight object {} belongs to an unknown stage", file.path),
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
        for stage in [FLUX_VAE_ENCODER_STAGE, FLUX_VAE_DECODER_STAGE] {
            if !stages.contains_key(stage) {
                return Err(contract(
                    stage,
                    "sealed manifest omits the required VAE half",
                ));
            }
        }
        for files in stages.values_mut() {
            files.sort_by_key(|file| file.shard.map_or(0, |shard| shard.index));
        }
        Ok(Self {
            manifest,
            config,
            stages,
            expected,
            max_shard_bytes,
        })
    }

    pub fn manifest(&self) -> &ArtifactManifest {
        &self.manifest
    }

    pub fn config(&self) -> &AutoencoderKlConfig {
        &self.config
    }

    pub const fn max_shard_bytes(&self) -> u64 {
        self.max_shard_bytes
    }

    fn files(&self, stage: &str) -> &[ArtifactFile] {
        self.stages
            .get(stage)
            .expect("validated contract contains both VAE stages")
    }

    fn expected(&self, stage: &str) -> &BTreeMap<String, TensorSpec> {
        self.expected
            .get(stage)
            .expect("validated contract contains both VAE inventories")
    }
}

fn validate_identity(manifest: &ArtifactManifest) -> Result<(), FluxVaeArtifactError> {
    if manifest.schema_version != ARTIFACT_MANIFEST_SCHEMA_V1 {
        return Err(FluxVaeArtifactError::Manifest(format!(
            "standalone components require schema {ARTIFACT_MANIFEST_SCHEMA_V1}, found {}",
            manifest.schema_version
        )));
    }
    if !manifest.dependencies.is_empty() {
        return Err(FluxVaeArtifactError::Manifest(
            "a standalone VAE component must not depend on another bundle".into(),
        ));
    }
    if manifest.profile.as_str() != FLUX_VAE_COMPONENT_PROFILE {
        return Err(FluxVaeArtifactError::Manifest(format!(
            "profile {} is not the canonical {FLUX_VAE_COMPONENT_PROFILE} profile",
            manifest.profile
        )));
    }
    for (field, actual, expected) in [
        (
            "bundle",
            manifest.bundle.as_str(),
            FLUX_VAE_COMPONENT_BUNDLE_ID,
        ),
        (
            "model",
            manifest.model.as_str(),
            FLUX_VAE_COMPONENT_MODEL_ID,
        ),
        (
            "model_revision",
            manifest.model_revision.as_str(),
            FLUX_VAE_COMPONENT_MODEL_REVISION,
        ),
    ] {
        if actual != expected {
            return Err(FluxVaeArtifactError::Manifest(format!(
                "{field} {actual:?} differs from canonical {expected:?}"
            )));
        }
    }
    let expected_digest = Sha256Digest::from_hex(FLUX_VAE_COMPONENT_CONTENT_DIGEST)
        .expect("static component digest is valid");
    if manifest.content_digest != Some(expected_digest) {
        return Err(FluxVaeArtifactError::Manifest(format!(
            "content digest {:?} differs from canonical {expected_digest}",
            manifest.content_digest
        )));
    }
    Ok(())
}

fn validate_metadata(manifest: &ArtifactManifest) -> Result<(), FluxVaeArtifactError> {
    for (key, expected) in FLUX_VAE_METADATA_VALUES {
        let actual = manifest.metadata.get(key).map(String::as_str);
        if actual != Some(expected) {
            return Err(FluxVaeArtifactError::Manifest(format!(
                "metadata {key} must be {expected:?}, found {actual:?}"
            )));
        }
    }
    for path in FLUX_VAE_METADATA_PATHS {
        if !manifest.files.iter().any(|file| file.path.as_str() == path) {
            return Err(FluxVaeArtifactError::Manifest(format!(
                "component manifest omits required metadata file {path}"
            )));
        }
    }
    if let Some(file) = manifest
        .files
        .iter()
        .filter(|file| file.role != ArtifactFileRole::Weights)
        .find(|file| !FLUX_VAE_METADATA_PATHS.contains(&file.path.as_str()))
    {
        return Err(FluxVaeArtifactError::Manifest(format!(
            "component manifest contains unexpected metadata file {}",
            file.path
        )));
    }
    Ok(())
}

fn stage_for_tensor(spec: &TensorSpec) -> &'static str {
    if spec.burn_name.starts_with("encoder.") || spec.burn_name.starts_with("quant_conv.") {
        FLUX_VAE_ENCODER_STAGE
    } else {
        FLUX_VAE_DECODER_STAGE
    }
}

fn declared_max_shard_bytes(manifest: &ArtifactManifest) -> Result<u64, FluxVaeArtifactError> {
    let value = manifest
        .metadata
        .get("target_max_shard_bytes")
        .ok_or_else(|| {
            FluxVaeArtifactError::Manifest("manifest omits target_max_shard_bytes".into())
        })?
        .parse::<u64>()
        .map_err(|error| {
            FluxVaeArtifactError::Manifest(format!("invalid target_max_shard_bytes: {error}"))
        })?;
    if value == 0 {
        return Err(FluxVaeArtifactError::Manifest(
            "target_max_shard_bytes must be positive".into(),
        ));
    }
    if manifest.numeric_format != burn_image::NumericFormat::F16 {
        return Err(FluxVaeArtifactError::Manifest(format!(
            "numeric format {:?} is not canonical F16 storage",
            manifest.numeric_format
        )));
    }
    Ok(value)
}

/// Source of independently verified VAE encoder and decoder stages.
pub trait FluxVaeStageSource<B: Backend> {
    type Error;

    fn load_encoder(&mut self) -> Result<AutoencoderKl<B>, Self::Error>;
    fn load_decoder(&mut self) -> Result<AutoencoderKl<B>, Self::Error>;
    fn synchronize(&mut self) -> Result<(), Self::Error>;
}

/// Wasm-local asynchronous source of independently verified VAE stages.
#[allow(async_fn_in_trait)]
pub trait AsyncFluxVaeStageSource<B: Backend> {
    type Error;

    async fn load_encoder(&mut self) -> Result<AutoencoderKl<B>, Self::Error>;
    async fn load_decoder(&mut self) -> Result<AutoencoderKl<B>, Self::Error>;
    async fn synchronize(&mut self) -> Result<(), Self::Error>;
}

/// Synchronous verified source backed by any model-neutral shard reader.
pub struct VerifiedBurnpackFluxVaeStageSource<B: Backend, R> {
    contract: FluxVaeComponentContract,
    device: B::Device,
    reader: R,
    float_policy: FluxVaeArtifactFloatPolicy,
}

impl<B: Backend, R: ArtifactShardReader> VerifiedBurnpackFluxVaeStageSource<B, R> {
    pub fn new(contract: FluxVaeComponentContract, device: B::Device, reader: R) -> Self {
        Self {
            contract,
            device,
            reader,
            float_policy: FluxVaeArtifactFloatPolicy::default(),
        }
    }

    pub fn with_float_policy(mut self, policy: FluxVaeArtifactFloatPolicy) -> Self {
        self.float_policy = policy;
        self
    }

    pub fn contract(&self) -> &FluxVaeComponentContract {
        &self.contract
    }

    pub fn reader(&self) -> &R {
        &self.reader
    }

    pub fn reader_mut(&mut self) -> &mut R {
        &mut self.reader
    }

    fn load_stage(&mut self, stage: &str) -> Result<AutoencoderKl<B>, FluxVaeArtifactError> {
        let files = self.contract.files(stage).to_vec();
        let expected = self.contract.expected(stage).clone();
        let mut model = self
            .contract
            .config
            .try_init(&self.device)
            .map_err(|error| FluxVaeArtifactError::Model(error.to_string()))?;
        let mut applied = BTreeSet::new();
        for file in files {
            let bytes = self.reader.read_shard(&file)?;
            let bytes = VerifiedArtifactBytes::unverified(bytes)
                .into_verified_bytes(&file, self.contract.max_shard_bytes())?;
            apply_object(
                &mut model,
                stage,
                &file,
                bytes,
                &expected,
                self.float_policy,
                &mut applied,
            )?;
        }
        ensure_complete(stage, &expected, &applied)?;
        Ok(model)
    }
}

impl<B: Backend> VerifiedBurnpackFluxVaeStageSource<B, DirectoryArtifactShardReader> {
    pub fn from_directory(
        root: impl AsRef<Path>,
        config: AutoencoderKlConfig,
        device: B::Device,
    ) -> Result<Self, FluxVaeArtifactError> {
        let directory = VerifiedArtifactDirectory::open(root.as_ref().to_owned())?;
        let contract = FluxVaeComponentContract::new(directory.manifest().clone(), config)?;
        let reader = directory.shard_reader()?;
        Ok(Self::new(contract, device, reader))
    }
}

impl<B: Backend, R: ArtifactShardReader> FluxVaeStageSource<B>
    for VerifiedBurnpackFluxVaeStageSource<B, R>
{
    type Error = FluxVaeArtifactError;

    fn load_encoder(&mut self) -> Result<AutoencoderKl<B>, Self::Error> {
        self.load_stage(FLUX_VAE_ENCODER_STAGE)
    }

    fn load_decoder(&mut self) -> Result<AutoencoderKl<B>, Self::Error> {
        self.load_stage(FLUX_VAE_DECODER_STAGE)
    }

    fn synchronize(&mut self) -> Result<(), Self::Error> {
        B::sync(&self.device).map_err(|error| FluxVaeArtifactError::Synchronize(error.to_string()))
    }
}

/// Asynchronous verified source for browser transport/cache adapters.
pub struct VerifiedAsyncBurnpackFluxVaeStageSource<B: Backend, R> {
    contract: FluxVaeComponentContract,
    device: B::Device,
    reader: R,
    float_policy: FluxVaeArtifactFloatPolicy,
}

impl<B: Backend, R: AsyncArtifactShardReader> VerifiedAsyncBurnpackFluxVaeStageSource<B, R> {
    pub fn new(contract: FluxVaeComponentContract, device: B::Device, reader: R) -> Self {
        Self {
            contract,
            device,
            reader,
            float_policy: FluxVaeArtifactFloatPolicy::default(),
        }
    }

    pub fn with_float_policy(mut self, policy: FluxVaeArtifactFloatPolicy) -> Self {
        self.float_policy = policy;
        self
    }

    pub fn contract(&self) -> &FluxVaeComponentContract {
        &self.contract
    }

    pub fn reader(&self) -> &R {
        &self.reader
    }

    pub fn reader_mut(&mut self) -> &mut R {
        &mut self.reader
    }

    async fn load_stage(&mut self, stage: &str) -> Result<AutoencoderKl<B>, FluxVaeArtifactError> {
        let files = self.contract.files(stage).to_vec();
        let expected = self.contract.expected(stage).clone();
        let mut model = self
            .contract
            .config
            .try_init(&self.device)
            .map_err(|error| FluxVaeArtifactError::Model(error.to_string()))?;
        let mut applied = BTreeSet::new();
        for file in files {
            let bytes = self
                .reader
                .read_verified_shard(&file, self.contract.max_shard_bytes())
                .await?
                .into_verified_bytes(&file, self.contract.max_shard_bytes())?;
            apply_object(
                &mut model,
                stage,
                &file,
                bytes,
                &expected,
                self.float_policy,
                &mut applied,
            )?;
        }
        ensure_complete(stage, &expected, &applied)?;
        Ok(model)
    }
}

impl<B: Backend, R: AsyncArtifactShardReader> AsyncFluxVaeStageSource<B>
    for VerifiedAsyncBurnpackFluxVaeStageSource<B, R>
{
    type Error = FluxVaeArtifactError;

    async fn load_encoder(&mut self) -> Result<AutoencoderKl<B>, Self::Error> {
        self.load_stage(FLUX_VAE_ENCODER_STAGE).await
    }

    async fn load_decoder(&mut self) -> Result<AutoencoderKl<B>, Self::Error> {
        self.load_stage(FLUX_VAE_DECODER_STAGE).await
    }

    async fn synchronize(&mut self) -> Result<(), Self::Error> {
        B::sync(&self.device).map_err(|error| FluxVaeArtifactError::Synchronize(error.to_string()))
    }
}

/// Opt-in device-resident cache for a synchronous verified source.
pub struct RetainingFluxVaeStageSource<B: Backend, S> {
    source: S,
    encoder: Option<AutoencoderKl<B>>,
    decoder: Option<AutoencoderKl<B>>,
}

impl<B: Backend, S> RetainingFluxVaeStageSource<B, S> {
    pub const fn new(source: S) -> Self {
        Self {
            source,
            encoder: None,
            decoder: None,
        }
    }

    pub fn cached_stage_count(&self) -> usize {
        usize::from(self.encoder.is_some()) + usize::from(self.decoder.is_some())
    }

    pub fn clear(&mut self) {
        self.encoder = None;
        self.decoder = None;
    }

    pub fn source(&self) -> &S {
        &self.source
    }

    pub fn source_mut(&mut self) -> &mut S {
        &mut self.source
    }

    pub fn into_source(self) -> S {
        self.source
    }
}

impl<B, S> FluxVaeStageSource<B> for RetainingFluxVaeStageSource<B, S>
where
    B: Backend,
    S: FluxVaeStageSource<B>,
{
    type Error = S::Error;

    fn load_encoder(&mut self) -> Result<AutoencoderKl<B>, Self::Error> {
        if self.encoder.is_none() {
            self.encoder = Some(self.source.load_encoder()?);
        }
        Ok(self.encoder.as_ref().expect("populated above").clone())
    }

    fn load_decoder(&mut self) -> Result<AutoencoderKl<B>, Self::Error> {
        if self.decoder.is_none() {
            self.decoder = Some(self.source.load_decoder()?);
        }
        Ok(self.decoder.as_ref().expect("populated above").clone())
    }

    fn synchronize(&mut self) -> Result<(), Self::Error> {
        self.source.synchronize()
    }
}

/// Opt-in device-resident cache for an asynchronous verified source.
pub struct RetainingAsyncFluxVaeStageSource<B: Backend, S> {
    source: S,
    retention_enabled: bool,
    encoder: Option<AutoencoderKl<B>>,
    decoder: Option<AutoencoderKl<B>>,
}

impl<B: Backend, S> RetainingAsyncFluxVaeStageSource<B, S> {
    pub fn new(source: S) -> Self {
        Self::with_retention(source, true)
    }

    pub fn passthrough(source: S) -> Self {
        Self::with_retention(source, false)
    }

    fn with_retention(source: S, retention_enabled: bool) -> Self {
        Self {
            source,
            retention_enabled,
            encoder: None,
            decoder: None,
        }
    }

    pub const fn retention_enabled(&self) -> bool {
        self.retention_enabled
    }

    pub fn cached_stage_count(&self) -> usize {
        usize::from(self.encoder.is_some()) + usize::from(self.decoder.is_some())
    }

    pub fn clear(&mut self) {
        self.encoder = None;
        self.decoder = None;
    }

    pub fn source(&self) -> &S {
        &self.source
    }

    pub fn source_mut(&mut self) -> &mut S {
        &mut self.source
    }

    pub fn into_source(self) -> S {
        self.source
    }
}

impl<B, S> AsyncFluxVaeStageSource<B> for RetainingAsyncFluxVaeStageSource<B, S>
where
    B: Backend,
    S: AsyncFluxVaeStageSource<B>,
{
    type Error = S::Error;

    async fn load_encoder(&mut self) -> Result<AutoencoderKl<B>, Self::Error> {
        if !self.retention_enabled {
            return self.source.load_encoder().await;
        }
        if self.encoder.is_none() {
            self.encoder = Some(self.source.load_encoder().await?);
        }
        Ok(self.encoder.as_ref().expect("populated above").clone())
    }

    async fn load_decoder(&mut self) -> Result<AutoencoderKl<B>, Self::Error> {
        if !self.retention_enabled {
            return self.source.load_decoder().await;
        }
        if self.decoder.is_none() {
            self.decoder = Some(self.source.load_decoder().await?);
        }
        Ok(self.decoder.as_ref().expect("populated above").clone())
    }

    async fn synchronize(&mut self) -> Result<(), Self::Error> {
        self.source.synchronize().await
    }
}

fn apply_object<B: Backend>(
    model: &mut AutoencoderKl<B>,
    stage: &str,
    file: &ArtifactFile,
    bytes: Vec<u8>,
    expected: &BTreeMap<String, TensorSpec>,
    float_policy: FluxVaeArtifactFloatPolicy,
    applied: &mut BTreeSet<String>,
) -> Result<(), FluxVaeArtifactError> {
    let mut store = BurnpackStore::from_bytes(Some(Bytes::from_bytes_vec(bytes)));
    let snapshots = store
        .get_all_snapshots()
        .map_err(|error| FluxVaeArtifactError::Burnpack {
            path: file.path.to_string(),
            message: error.to_string(),
        })?;
    if snapshots.is_empty() {
        return Err(contract(stage, format!("{} is empty", file.path)));
    }
    let mut local = Vec::with_capacity(snapshots.len());
    for (name, snapshot) in snapshots {
        let spec = expected
            .get(name)
            .ok_or_else(|| contract(stage, format!("unknown tensor {name}")))?;
        if !applied.insert(name.clone()) {
            return Err(contract(stage, format!("duplicate tensor {name}")));
        }
        if snapshot.shape.as_slice() != spec.burn_shape.as_slice() {
            return Err(contract(
                stage,
                format!(
                    "tensor {name} shape mismatch: expected {:?}, got {:?}",
                    spec.burn_shape, snapshot.shape
                ),
            ));
        }
        if snapshot.dtype != DType::F16 {
            return Err(contract(
                stage,
                format!(
                    "tensor {name} dtype mismatch: expected F16, got {:?}",
                    snapshot.dtype
                ),
            ));
        }
        local.push(snapshot.clone());
    }
    let expected_applied = local
        .iter()
        .map(TensorSnapshot::full_path)
        .collect::<BTreeSet<_>>();
    let result = model.apply(local, None, load_adapter(float_policy), false);
    validate_apply(stage, &result, &expected_applied)
}

fn ensure_complete(
    stage: &str,
    expected: &BTreeMap<String, TensorSpec>,
    applied: &BTreeSet<String>,
) -> Result<(), FluxVaeArtifactError> {
    let missing = expected
        .keys()
        .filter(|name| !applied.contains(*name))
        .take(16)
        .collect::<Vec<_>>();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(contract(
            stage,
            format!("stage is incomplete; first missing tensors: {missing:?}"),
        ))
    }
}

fn validate_apply(
    stage: &str,
    result: &ApplyResult,
    expected: &BTreeSet<String>,
) -> Result<(), FluxVaeArtifactError> {
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

#[derive(Debug, Clone, Copy)]
struct Float32Adapter;

impl ModuleAdapter for Float32Adapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        if snapshot.dtype != DType::F16 {
            return snapshot.clone();
        }
        let data = snapshot.clone_data_fn();
        TensorSnapshot::from_closure(
            Rc::new(move || Ok(data()?.convert_dtype(DType::F32))),
            DType::F32,
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

fn load_adapter(policy: FluxVaeArtifactFloatPolicy) -> Option<Box<dyn ModuleAdapter>> {
    (policy == FluxVaeArtifactFloatPolicy::AdaptToF32)
        .then(|| Box::new(Float32Adapter) as Box<dyn ModuleAdapter>)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_image::{
        ArtifactBundleId, ArtifactPath, ArtifactProfileId, ModelId, NumericFormat, Sha256Digest,
    };

    fn identity_manifest() -> ArtifactManifest {
        ArtifactManifest {
            schema_version: ARTIFACT_MANIFEST_SCHEMA_V1,
            bundle: ArtifactBundleId::new(FLUX_VAE_COMPONENT_BUNDLE_ID).unwrap(),
            profile: ArtifactProfileId::new(FLUX_VAE_COMPONENT_PROFILE).unwrap(),
            model: ModelId::new(FLUX_VAE_COMPONENT_MODEL_ID).unwrap(),
            model_revision: FLUX_VAE_COMPONENT_MODEL_REVISION.into(),
            numeric_format: NumericFormat::F16,
            components: Vec::new(),
            files: Vec::new(),
            dependencies: Vec::new(),
            metadata: BTreeMap::new(),
            content_digest: Some(
                Sha256Digest::from_hex(FLUX_VAE_COMPONENT_CONTENT_DIGEST).unwrap(),
            ),
        }
    }

    fn metadata_manifest() -> ArtifactManifest {
        let mut manifest = identity_manifest();
        manifest.metadata = FLUX_VAE_METADATA_VALUES
            .into_iter()
            .map(|(key, value)| (key.into(), value.into()))
            .collect();
        manifest.files = FLUX_VAE_METADATA_PATHS
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
    fn inventory_split_is_complete_and_disjoint_correctness() {
        let inventory = TensorInventory::from_config(&AutoencoderKlConfig::flux1()).unwrap();
        let encoder = inventory
            .tensors
            .iter()
            .filter(|spec| stage_for_tensor(spec) == FLUX_VAE_ENCODER_STAGE)
            .count();
        let decoder = inventory.tensors.len() - encoder;
        assert_eq!(encoder, 106);
        assert_eq!(decoder, 138);
    }

    #[test]
    fn component_identity_fails_closed_correctness() {
        let manifest = identity_manifest();
        validate_identity(&manifest).unwrap();

        let mut wrong_bundle = manifest.clone();
        wrong_bundle.bundle = ArtifactBundleId::new("vae-unpinned").unwrap();
        assert!(validate_identity(&wrong_bundle).is_err());

        let mut wrong_model = manifest.clone();
        wrong_model.model = ModelId::new("Example/FluxVae").unwrap();
        assert!(validate_identity(&wrong_model).is_err());

        let mut wrong_digest = manifest.clone();
        wrong_digest.content_digest = Some(Sha256Digest::calculate(b"wrong vae component"));
        assert!(validate_identity(&wrong_digest).is_err());

        let mut wrong_revision = manifest;
        wrong_revision.model_revision = "0".repeat(64);
        assert!(validate_identity(&wrong_revision).is_err());

        let mut wrong_profile = identity_manifest();
        wrong_profile.profile = ArtifactProfileId::new("f32").unwrap();
        assert!(validate_identity(&wrong_profile).is_err());
    }

    #[test]
    fn component_metadata_layout_and_counts_fail_closed_correctness() {
        let manifest = metadata_manifest();
        validate_metadata(&manifest).unwrap();

        let mut wrong_count = manifest.clone();
        wrong_count
            .metadata
            .insert("tensor_count".into(), "243".into());
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
