//! Verified packed-F16 denoiser cache for the browser Turbo fallback.
//!
//! Burnpack objects remain the authenticated transport format. Preload verifies one bounded
//! object at a time, copies its canonical F16 payloads into a U32 device arena, and drops all
//! decoded host data before reading the next object. Tensor starts are padded to WebGPU's
//! 256-byte F32 binding alignment. A semantic-stage load widens only that stage's cached objects
//! and maps aligned shared-arena views into an otherwise lazy Boogu module.

use std::collections::{BTreeMap, BTreeSet};

use burn::{
    module::{Module, ModuleMapper, Param},
    prelude::Backend,
    tensor::{
        Bytes, DType, Int, Shape, Tensor, TensorCreationOptions, TensorData, TensorPrimitive,
    },
};
use burn_cubecl::cubecl::Runtime;
use burn_image::{ArtifactFile, ArtifactFileRole, ArtifactManifest, Sha256Digest};
use burn_store::{BurnpackStore, ModuleStore};
use burn_wgpu::{CubeBackend, WgpuDevice, WgpuRuntime};

use crate::artifacts::{
    AsyncStageShardReader, BooguArtifactInventory, BooguArtifactLoadError, BooguReleaseIdentity,
    BooguStorageProfile, SerializedTensorInventory, TensorOwner, declared_target_max_shard_bytes,
    read_verified_async, validate_release_manifest, verify_inventory_contract_async,
};
use crate::{
    AsyncBooguDenoiserStageSource, BooguConfig, BooguDenoiserPrelude, BooguDenoiserTail,
    BooguError, BooguVariant, DoubleStreamBlock, MaterializedF32Object, PackedF16Layout,
    PackedF16Object, SingleStreamBlock, materialize_packed_f16_objects,
};

/// Backend whose raw U32 arenas are accepted by the packed-F16 widening kernel.
pub type PackedF16DenoiserBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;

/// Semantic stages used by ordinary Boogu Image Turbo (reference refiners are not executed).
pub const TURBO_PACKED_F16_STAGE_COUNT: usize = 46;
/// Canonical F16 tensors used by ordinary Boogu Image Turbo.
pub const TURBO_PACKED_F16_TENSOR_COUNT: usize = 912;
/// Logical Burnpack objects containing those tensors.
pub const TURBO_PACKED_F16_OBJECT_COUNT: usize = 106;
/// Authenticated compact F16 payload bytes before device-alignment padding.
pub const TURBO_PACKED_F16_COMPACT_PAYLOAD_BYTES: u64 = 19_869_996_096;
/// Exact bytes transferred for the 106 authenticated Burnpack objects, including framing.
pub const TURBO_PACKED_F16_ARTIFACT_BYTES: u64 = 19_870_166_528;
/// Zero F16 elements inserted so every materialized tensor starts on a 256-byte boundary.
pub const TURBO_PACKED_F16_PADDING_ELEMENTS: u64 = 7_264;
/// Logical F16 elements in all padded device arenas.
pub const TURBO_PACKED_F16_PADDED_ELEMENTS: u64 = 9_935_005_312;
/// Exact retained U32 device bytes after alignment padding.
pub const TURBO_PACKED_F16_RETAINED_BYTES: u64 = 19_870_010_624;
/// F32 bytes written if every Turbo stage is materialized once.
pub const TURBO_PACKED_F16_F32_WRITE_BYTES_PER_DMD: u64 = 39_740_021_248;
/// Largest one-object padded U32 arena.
pub const TURBO_PACKED_F16_MAX_OBJECT_BYTES: u64 = 254_251_904;
/// Largest one-object materialized F32 arena.
pub const TURBO_PACKED_F16_MAX_OBJECT_F32_BYTES: u64 = 508_503_808;
/// Sum of padded U32 arenas read by the largest semantic stage.
pub const TURBO_PACKED_F16_MAX_STAGE_PACKED_BYTES: u64 = 876_827_328;
/// Sum of F32 arenas written by the largest semantic stage.
pub const TURBO_PACKED_F16_MAX_STAGE_F32_BYTES: u64 = 1_753_654_656;
/// Packed device bytes read across all four DMD denoiser evaluations.
pub const TURBO_PACKED_F16_FOUR_DMD_READ_BYTES: u64 = 79_480_042_496;
/// F32 device bytes written across all four DMD denoiser evaluations.
pub const TURBO_PACKED_F16_FOUR_DMD_WRITE_BYTES: u64 = 158_960_084_992;

/// Lifecycle state of the verified packed denoiser cache.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PackedF16DenoiserCacheState {
    /// No packed object is retained.
    #[default]
    Empty,
    /// A bounded object-at-a-time preload is in progress.
    Preloading,
    /// The exact Turbo raw cache is complete and may materialize stages.
    Ready,
    /// Verification or materialization failed; callers must clear before retrying.
    Failed,
}

/// Exact resident and cumulative traffic audit for the packed Turbo denoiser source.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PackedF16DenoiserCacheAudit {
    /// Current cache lifecycle state.
    pub state: PackedF16DenoiserCacheState,
    /// Whether all exact Turbo objects are resident and usable.
    pub packed_cache_ready: bool,
    /// Number of semantic stages represented by retained objects.
    pub cached_stage_count: usize,
    /// Number of retained packed device objects.
    pub cached_object_count: usize,
    /// Number of canonical tensors represented by those objects.
    pub cached_tensor_count: usize,
    /// Physical padded U32 bytes currently retained on device.
    pub retained_packed_bytes: u64,
    /// Cumulative authenticated Burnpack bytes read from the artifact reader.
    pub packed_read_bytes: u64,
    /// Cumulative padded U32 bytes submitted for host-to-device upload.
    pub packed_upload_bytes: u64,
    /// Cumulative padded U32 bytes consumed by widening dispatches.
    pub materialization_packed_read_bytes: u64,
    /// Number of successfully materialized semantic stages.
    pub materialized_stage_count: u64,
    /// Number of object-wide widening dispatches queued for successful stage loads.
    pub object_unpack_count: u64,
    /// Cumulative F32 arena bytes written for successful stage loads.
    pub f32_write_bytes: u64,
    /// Number of explicit preload attempts.
    pub preload_attempt_count: u64,
    /// Number of preload or stage-materialization failures.
    pub failure_count: u64,
}

#[derive(Debug, Clone)]
struct PackedTensorDescriptor {
    target_name: String,
    shape: Vec<usize>,
    offset_elements: usize,
    elements: usize,
    digest: Sha256Digest,
}

#[derive(Debug, Clone)]
struct PackedObjectDescriptor {
    file: ArtifactFile,
    stage: String,
    tensors: Vec<PackedTensorDescriptor>,
    f16_elements: usize,
    packed_bytes: u64,
    f32_bytes: u64,
}

#[derive(Debug)]
struct CachedPackedObject {
    descriptor: PackedObjectDescriptor,
    arena: PackedF16Object<WgpuRuntime>,
}

type PackedF16Catalog = (Vec<PackedObjectDescriptor>, BTreeMap<String, Vec<String>>);

/// Async verified source retaining only padded packed-F16 Turbo objects between requests.
///
/// Returned modules own short-lived shared views into newly materialized F32 object arenas. The
/// caller must await [`AsyncBooguDenoiserStageSource::synchronize`] after executing each stage and
/// before dropping its module. Browser integrations should replace that blocking backend barrier
/// with their event-loop-safe async WebGPU barrier.
pub struct VerifiedAsyncPackedF16DenoiserStageSource<R> {
    config: BooguConfig,
    device: WgpuDevice,
    reader: R,
    max_bytes: u64,
    catalog: Vec<PackedObjectDescriptor>,
    stage_objects: BTreeMap<String, Vec<String>>,
    cache: BTreeMap<String, CachedPackedObject>,
    audit: PackedF16DenoiserCacheAudit,
}

impl<R: AsyncStageShardReader> VerifiedAsyncPackedF16DenoiserStageSource<R> {
    /// Authenticate the release metadata and build the exact 46-stage Turbo object catalog.
    #[allow(clippy::too_many_arguments)]
    pub async fn new(
        identity: &BooguReleaseIdentity,
        manifest: ArtifactManifest,
        inventory: BooguArtifactInventory,
        config: BooguConfig,
        profile: BooguStorageProfile,
        device: WgpuDevice,
        mut reader: R,
    ) -> Result<Self, BooguArtifactLoadError> {
        if identity.variant != BooguVariant::Image01Turbo {
            return Err(contract(
                "packed-f16-denoiser",
                "packed-F16 fallback is exact only for Boogu Image Turbo",
            ));
        }
        if profile != BooguStorageProfile::F16QwenVisionF32 {
            return Err(contract(
                "packed-f16-denoiser",
                "packed-F16 fallback requires the canonical f16-qwen-vision-f32 profile",
            ));
        }
        if config != BooguConfig::default() {
            return Err(contract(
                "packed-f16-denoiser",
                "packed-F16 fallback requires the exact released Turbo denoiser config",
            ));
        }

        validate_release_manifest(identity, &manifest, &inventory, profile)?;
        let max_bytes = declared_target_max_shard_bytes(&manifest)?;
        let entries =
            verify_inventory_contract_async(&manifest, &inventory, profile, &mut reader, max_bytes)
                .await?;
        let (catalog, stage_objects) =
            build_turbo_catalog(&manifest, &inventory, &entries, max_bytes)?;

        Ok(Self {
            config,
            device,
            reader,
            max_bytes,
            catalog,
            stage_objects,
            cache: BTreeMap::new(),
            audit: PackedF16DenoiserCacheAudit::default(),
        })
    }

    /// Load and authenticate all 106 used Turbo Burnpacks into padded U32 device arenas.
    ///
    /// No F32 tensor or widening dispatch is created during preload. Calling this again after a
    /// successful preload is an I/O-free cache hit.
    pub async fn preload_turbo_raw(&mut self) -> Result<PackedF16DenoiserCacheAudit, BooguError> {
        if self.audit.state == PackedF16DenoiserCacheState::Ready {
            return Ok(self.audit());
        }
        if self.audit.state != PackedF16DenoiserCacheState::Empty {
            return Err(BooguError::Artifact(format!(
                "packed-F16 cache cannot preload from {:?}; clear it first",
                self.audit.state
            )));
        }

        self.audit.preload_attempt_count += 1;
        self.audit.state = PackedF16DenoiserCacheState::Preloading;
        let catalog = self.catalog.clone();
        for descriptor in catalog {
            if let Err(error) = self.preload_object(descriptor).await {
                self.cache.clear();
                self.clear_resident_audit();
                self.audit.failure_count += 1;
                self.audit.state = PackedF16DenoiserCacheState::Failed;
                return Err(error);
            }
        }

        if let Err(error) = self.validate_complete_cache() {
            self.cache.clear();
            self.clear_resident_audit();
            self.audit.failure_count += 1;
            self.audit.state = PackedF16DenoiserCacheState::Failed;
            return Err(error);
        }
        self.audit.state = PackedF16DenoiserCacheState::Ready;
        self.audit.packed_cache_ready = true;
        Ok(self.audit())
    }

    /// Borrow the asynchronous artifact transport/cache reader.
    pub const fn reader(&self) -> &R {
        &self.reader
    }

    /// Mutably borrow the asynchronous artifact transport/cache reader.
    pub fn reader_mut(&mut self) -> &mut R {
        &mut self.reader
    }

    /// Maximum response size enforced for every metadata or Burnpack read.
    pub const fn max_shard_bytes(&self) -> u64 {
        self.max_bytes
    }

    /// Snapshot the exact cache and cumulative traffic audit.
    pub fn audit(&self) -> PackedF16DenoiserCacheAudit {
        self.audit
    }

    /// Whether the complete exact Turbo packed cache is ready.
    pub const fn packed_cache_ready(&self) -> bool {
        self.audit.packed_cache_ready
    }

    /// Number of semantic stages represented by the current cache.
    pub const fn cached_stage_count(&self) -> usize {
        self.audit.cached_stage_count
    }

    /// Number of logical Burnpack objects retained by the current cache.
    pub const fn cached_object_count(&self) -> usize {
        self.audit.cached_object_count
    }

    /// Number of canonical tensors represented by the current cache.
    pub const fn cached_tensor_count(&self) -> usize {
        self.audit.cached_tensor_count
    }

    /// Physical padded U32 bytes retained by the current cache.
    pub const fn retained_packed_bytes(&self) -> u64 {
        self.audit.retained_packed_bytes
    }

    /// Cumulative authenticated Burnpack bytes read from the reader.
    pub const fn packed_read_bytes(&self) -> u64 {
        self.audit.packed_read_bytes
    }

    /// Cumulative padded U32 upload bytes.
    pub const fn packed_upload_bytes(&self) -> u64 {
        self.audit.packed_upload_bytes
    }

    /// Cumulative packed device bytes read by widening dispatches.
    pub const fn materialization_packed_read_bytes(&self) -> u64 {
        self.audit.materialization_packed_read_bytes
    }

    /// Number of successfully materialized semantic stages.
    pub const fn materialized_stage_count(&self) -> u64 {
        self.audit.materialized_stage_count
    }

    /// Number of object-wide widening dispatches queued by successful stage loads.
    pub const fn object_unpack_count(&self) -> u64 {
        self.audit.object_unpack_count
    }

    /// Cumulative F32 bytes written by successful stage loads.
    pub const fn f32_write_bytes(&self) -> u64 {
        self.audit.f32_write_bytes
    }

    /// Drop all retained packed arenas while preserving cumulative traffic counters.
    ///
    /// If a materialized stage was submitted, the caller must first await that stage's execution
    /// barrier. The source itself cannot perform an event-loop-safe browser barrier synchronously.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.clear_resident_audit();
        self.audit.state = PackedF16DenoiserCacheState::Empty;
    }

    /// Mark a request failure and drop the cache after the caller's stage execution barrier.
    pub fn fail_and_clear(&mut self) {
        self.cache.clear();
        self.clear_resident_audit();
        if self.audit.state != PackedF16DenoiserCacheState::Failed {
            self.audit.failure_count += 1;
        }
        self.audit.state = PackedF16DenoiserCacheState::Failed;
    }

    async fn preload_object(
        &mut self,
        descriptor: PackedObjectDescriptor,
    ) -> Result<(), BooguError> {
        let path = descriptor.file.path.to_string();
        if self.cache.contains_key(&path) {
            return Err(BooguError::Artifact(format!(
                "packed-F16 preload repeats object {path}"
            )));
        }
        let bytes = read_verified_async(&mut self.reader, &descriptor.file, self.max_bytes).await?;
        self.audit.packed_read_bytes = self
            .audit
            .packed_read_bytes
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| BooguError::Artifact("packed read-byte counter overflow".into()))?;

        let mut store = BurnpackStore::from_bytes(Some(Bytes::from_bytes_vec(bytes)));
        let snapshots = store
            .get_all_snapshots()
            .map_err(|error| BooguError::Artifact(format!("invalid Burnpack {path}: {error}")))?;
        let expected_names = descriptor
            .tensors
            .iter()
            .map(|tensor| tensor.target_name.as_str())
            .collect::<BTreeSet<_>>();
        let actual_names = snapshots
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        validate_object_keyset(&path, &expected_names, &actual_names)?;

        let mut words = PackedWordBuilder::with_capacity(descriptor.f16_elements.div_ceil(2));
        for tensor in &descriptor.tensors {
            words.zero_pad_to(tensor.offset_elements)?;
            let snapshot = snapshots
                .get(&tensor.target_name)
                .expect("Burnpack keyset equality checked");
            let data = snapshot.to_data().map_err(|error| {
                BooguError::Artifact(format!(
                    "failed to decode packed-F16 tensor {} in {path}: {error}",
                    tensor.target_name
                ))
            })?;
            validate_tensor_payload(
                tensor,
                &path,
                snapshot.dtype,
                snapshot.shape.as_slice(),
                data.dtype,
                data.shape.as_slice(),
                data.bytes.as_ref(),
            )?;
            words.push_f16_bytes(data.bytes.as_ref())?;
        }
        words.zero_pad_to(descriptor.f16_elements)?;
        let packed_words = words.finish()?;
        if packed_words.len() != descriptor.f16_elements.div_ceil(2) {
            return Err(BooguError::Artifact(format!(
                "packed-F16 object {path} produced the wrong padded word count"
            )));
        }

        let raw = Tensor::<PackedF16DenoiserBackend, 1, Int>::from_data(
            TensorData::new(packed_words, [descriptor.f16_elements.div_ceil(2)]),
            packed_raw_upload_options(&self.device),
        )
        .into_primitive();
        let arena = PackedF16Object::try_new(raw, descriptor.f16_elements).map_err(|error| {
            BooguError::Artifact(format!("invalid packed arena {path}: {error}"))
        })?;
        WgpuRuntime::client(&self.device)
            .sync()
            .await
            .map_err(|error| {
                BooguError::Artifact(format!(
                    "packed-F16 object {path} upload synchronization failed: {error}"
                ))
            })?;

        self.audit.cached_object_count += 1;
        self.audit.cached_tensor_count += descriptor.tensors.len();
        self.audit.retained_packed_bytes += descriptor.packed_bytes;
        self.audit.packed_upload_bytes += descriptor.packed_bytes;
        self.cache
            .insert(path, CachedPackedObject { descriptor, arena });
        self.audit.cached_stage_count = self
            .cache
            .values()
            .map(|object| object.descriptor.stage.as_str())
            .collect::<BTreeSet<_>>()
            .len();
        Ok(())
    }

    fn validate_complete_cache(&self) -> Result<(), BooguError> {
        let exact = self.audit.cached_stage_count == TURBO_PACKED_F16_STAGE_COUNT
            && self.audit.cached_object_count == TURBO_PACKED_F16_OBJECT_COUNT
            && self.audit.cached_tensor_count == TURBO_PACKED_F16_TENSOR_COUNT
            && self.audit.retained_packed_bytes == TURBO_PACKED_F16_RETAINED_BYTES;
        if !exact {
            return Err(BooguError::Artifact(format!(
                "packed-F16 Turbo cache totals are not exact: {:?}",
                self.audit
            )));
        }
        Ok(())
    }

    fn clear_resident_audit(&mut self) {
        self.audit.packed_cache_ready = false;
        self.audit.cached_stage_count = 0;
        self.audit.cached_object_count = 0;
        self.audit.cached_tensor_count = 0;
        self.audit.retained_packed_bytes = 0;
    }

    fn load_module<M: Module<PackedF16DenoiserBackend>>(
        &mut self,
        stage: &str,
        prefix: &str,
        module: M,
    ) -> Result<M, BooguError> {
        if self.audit.state != PackedF16DenoiserCacheState::Ready {
            return Err(BooguError::Artifact(format!(
                "packed-F16 Turbo cache is not ready (state {:?})",
                self.audit.state
            )));
        }
        let object_paths = self.stage_objects.get(stage).cloned().ok_or_else(|| {
            BooguError::Artifact(format!("packed-F16 catalog has no Turbo stage {stage}"))
        })?;
        let packed_bytes = object_paths
            .iter()
            .map(|path| {
                self.cache
                    .get(path)
                    .map(|object| object.descriptor.packed_bytes)
                    .ok_or_else(|| {
                        BooguError::Artifact(format!(
                            "packed-F16 cache is missing stage {stage} object {path}"
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<u64>();
        let f32_bytes = object_paths
            .iter()
            .map(|path| {
                self.cache
                    .get(path)
                    .expect("stage packed-byte pass checked cache completeness")
                    .descriptor
                    .f32_bytes
            })
            .sum::<u64>();
        let packed_objects = object_paths
            .iter()
            .map(|path| {
                self.cache
                    .get(path)
                    .map(|object| object.arena.clone())
                    .ok_or_else(|| {
                        BooguError::Artifact(format!(
                            "packed-F16 cache is missing stage {stage} object {path}"
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let materialized = materialize_packed_f16_objects(&packed_objects).map_err(|error| {
            self.audit.state = PackedF16DenoiserCacheState::Failed;
            self.audit.packed_cache_ready = false;
            self.audit.failure_count += 1;
            BooguError::Artifact(format!(
                "failed to materialize packed stage {stage}: {error}"
            ))
        })?;
        self.audit.object_unpack_count += object_paths.len() as u64;
        self.audit.materialization_packed_read_bytes += packed_bytes;
        self.audit.f32_write_bytes += f32_bytes;

        let mut bindings = BTreeMap::new();
        for (object_slot, path) in object_paths.iter().enumerate() {
            let object = self
                .cache
                .get(path)
                .expect("cached object paths checked before materialization");
            for tensor in &object.descriptor.tensors {
                let local_name = tensor.target_name.strip_prefix(prefix).ok_or_else(|| {
                    BooguError::Artifact(format!(
                        "packed-F16 stage {stage} tensor {} does not start with {prefix:?}",
                        tensor.target_name
                    ))
                })?;
                if bindings
                    .insert(
                        local_name.to_owned(),
                        PackedTensorBinding {
                            object_slot,
                            offset_elements: tensor.offset_elements,
                            shape: tensor.shape.clone(),
                        },
                    )
                    .is_some()
                {
                    return Err(BooguError::Artifact(format!(
                        "packed-F16 stage {stage} repeats local tensor {local_name}"
                    )));
                }
            }
        }

        let mut mapper = PackedStageMapper::new(materialized, bindings);
        let module = module.map(&mut mapper);
        if let Err(error) = mapper.finish(stage) {
            self.audit.state = PackedF16DenoiserCacheState::Failed;
            self.audit.packed_cache_ready = false;
            self.audit.failure_count += 1;
            return Err(error);
        }

        self.audit.materialized_stage_count += 1;
        Ok(module)
    }
}

impl<R: AsyncStageShardReader> AsyncBooguDenoiserStageSource<PackedF16DenoiserBackend>
    for VerifiedAsyncPackedF16DenoiserStageSource<R>
{
    async fn load_prelude(
        &mut self,
    ) -> Result<BooguDenoiserPrelude<PackedF16DenoiserBackend>, BooguError> {
        let module = BooguDenoiserPrelude::new(self.config.clone(), &self.device)?;
        self.load_module("boogu-prelude", "", module)
    }

    async fn load_context_refiner(
        &mut self,
        index: usize,
    ) -> Result<SingleStreamBlock<PackedF16DenoiserBackend>, BooguError> {
        let stage = format!("boogu-context-refiner-{index:02}");
        let prefix = format!("context_refiner.{index}.");
        let module = single_block(&self.config, false, &self.device);
        self.load_module(&stage, &prefix, module)
    }

    async fn load_noise_refiner(
        &mut self,
        index: usize,
    ) -> Result<SingleStreamBlock<PackedF16DenoiserBackend>, BooguError> {
        let stage = format!("boogu-noise-refiner-{index:02}");
        let prefix = format!("noise_refiner.{index}.");
        let module = single_block(&self.config, true, &self.device);
        self.load_module(&stage, &prefix, module)
    }

    async fn load_reference_refiner(
        &mut self,
        index: usize,
    ) -> Result<SingleStreamBlock<PackedF16DenoiserBackend>, BooguError> {
        Err(BooguError::Artifact(format!(
            "ordinary Turbo packed-F16 cache excludes reference refiner {index}"
        )))
    }

    async fn load_double_stream(
        &mut self,
        index: usize,
    ) -> Result<DoubleStreamBlock<PackedF16DenoiserBackend>, BooguError> {
        let stage = format!("boogu-dual-block-{index:02}");
        let prefix = format!("double_stream_layers.{index}.");
        let module = DoubleStreamBlock::new(
            self.config.hidden_size,
            self.config.ffn_inner_dim(),
            self.config.num_attention_heads,
            self.config.num_kv_heads,
            self.config.hidden_size.min(1024),
            self.config.norm_eps,
            &self.device,
        );
        self.load_module(&stage, &prefix, module)
    }

    async fn load_single_stream(
        &mut self,
        index: usize,
    ) -> Result<SingleStreamBlock<PackedF16DenoiserBackend>, BooguError> {
        let stage = format!("boogu-single-block-{index:02}");
        let prefix = format!("single_stream_layers.{index}.");
        let module = single_block(&self.config, true, &self.device);
        self.load_module(&stage, &prefix, module)
    }

    async fn load_tail(
        &mut self,
    ) -> Result<BooguDenoiserTail<PackedF16DenoiserBackend>, BooguError> {
        let module = BooguDenoiserTail::new(self.config.clone(), &self.device)?;
        self.load_module("boogu-tail", "", module)
    }

    async fn synchronize(&mut self) -> Result<(), BooguError> {
        PackedF16DenoiserBackend::sync(&self.device)
            .map_err(|error| BooguError::Artifact(format!("device sync failed: {error}")))
    }
}

#[derive(Debug, Clone)]
struct PackedTensorBinding {
    object_slot: usize,
    offset_elements: usize,
    shape: Vec<usize>,
}

struct PackedStageMapper {
    materialized: Vec<MaterializedF32Object<WgpuRuntime>>,
    bindings: BTreeMap<String, PackedTensorBinding>,
    path_stack: Vec<String>,
    applied: BTreeSet<String>,
    error: Option<String>,
}

impl PackedStageMapper {
    fn new(
        materialized: Vec<MaterializedF32Object<WgpuRuntime>>,
        bindings: BTreeMap<String, PackedTensorBinding>,
    ) -> Self {
        Self {
            materialized,
            bindings,
            path_stack: Vec::new(),
            applied: BTreeSet::new(),
            error: None,
        }
    }

    fn finish(self, stage: &str) -> Result<(), BooguError> {
        if let Some(error) = self.error {
            return Err(BooguError::Artifact(format!(
                "failed to map packed-F16 stage {stage}: {error}"
            )));
        }
        let expected = self.bindings.keys().cloned().collect::<BTreeSet<_>>();
        if self.applied != expected {
            let missing = expected
                .difference(&self.applied)
                .take(16)
                .cloned()
                .collect::<Vec<_>>();
            return Err(BooguError::Artifact(format!(
                "packed-F16 stage {stage} did not map its exact keyset; missing={missing:?}"
            )));
        }
        Ok(())
    }
}

impl ModuleMapper<PackedF16DenoiserBackend> for PackedStageMapper {
    fn enter_module(&mut self, name: &str, container_type: &str) {
        if !container_type.starts_with("Enum:") {
            self.path_stack.push(name.to_owned());
        }
    }

    fn exit_module(&mut self, _name: &str, container_type: &str) {
        if !container_type.starts_with("Enum:") {
            self.path_stack.pop();
        }
    }

    fn map_float<const D: usize>(
        &mut self,
        param: Param<Tensor<PackedF16DenoiserBackend, D>>,
    ) -> Param<Tensor<PackedF16DenoiserBackend, D>> {
        if self.error.is_some() {
            return param;
        }
        let path = self.path_stack.join(".");
        let Some(binding) = self.bindings.get(&path) else {
            self.error = Some(format!("module contains unknown tensor {path}"));
            return param;
        };
        if !self.applied.insert(path.clone()) {
            self.error = Some(format!("module repeats tensor {path}"));
            return param;
        }
        let target_shape = param.lazy_shape();
        if target_shape.as_slice() != binding.shape.as_slice() {
            self.error = Some(format!(
                "tensor {path} module shape {:?} differs from sealed {:?}",
                target_shape, binding.shape
            ));
            return param;
        }
        let Some(object) = self.materialized.get(binding.object_slot) else {
            self.error = Some(format!(
                "tensor {path} references an unknown materialized object"
            ));
            return param;
        };
        let primitive =
            match object.slice(binding.offset_elements, Shape::from(binding.shape.clone())) {
                Ok(primitive) => primitive,
                Err(error) => {
                    self.error = Some(format!("tensor {path} has an invalid arena view: {error}"));
                    return param;
                }
            };
        let tensor = Tensor::<PackedF16DenoiserBackend, D>::from_primitive(TensorPrimitive::Float(
            primitive,
        ));
        let id = param.id;
        param.transform_for_load(tensor, id)
    }
}

fn build_turbo_catalog(
    manifest: &ArtifactManifest,
    inventory: &BooguArtifactInventory,
    entries: &[SerializedTensorInventory],
    max_bytes: u64,
) -> Result<PackedF16Catalog, BooguArtifactLoadError> {
    let expected_targets = inventory
        .tensors()
        .iter()
        .filter(|spec| {
            spec.owner == TensorOwner::BooguDenoiser
                && !spec.stage.starts_with("boogu-reference-refiner-")
        })
        .map(|spec| spec.target_name.as_str())
        .collect::<BTreeSet<_>>();
    let selected = entries
        .iter()
        .filter(|entry| {
            entry.owner == TensorOwner::BooguDenoiser
                && entry.included
                && !entry.stage.starts_with("boogu-reference-refiner-")
        })
        .collect::<Vec<_>>();
    let actual_targets = selected
        .iter()
        .map(|entry| entry.target_name.as_str())
        .collect::<Vec<_>>();
    validate_exact_names(
        "packed-f16-denoiser",
        &expected_targets,
        actual_targets.iter().copied(),
    )?;

    let mut weight_files = BTreeMap::new();
    for file in manifest
        .files
        .iter()
        .filter(|file| file.role == ArtifactFileRole::Weights)
    {
        if weight_files.insert(file.path.as_str(), file).is_some() {
            return Err(contract(
                "packed-f16-denoiser",
                format!("manifest repeats weight object {}", file.path),
            ));
        }
    }
    let mut by_object = BTreeMap::<String, Vec<&SerializedTensorInventory>>::new();
    for entry in selected {
        if entry.source_row_range.is_some()
            || entry.stored_dtype.as_deref() != Some("f16")
            || entry.quantized
            || entry.stored_shape.is_none()
            || entry.stored_sha256.is_none()
        {
            return Err(contract(
                &entry.stage,
                format!(
                    "packed-F16 tensor {} lacks an exact unsliced F16 shape/digest contract",
                    entry.target_name
                ),
            ));
        }
        let object = entry.burnpack_object.as_ref().ok_or_else(|| {
            contract(
                &entry.stage,
                format!("packed-F16 tensor {} omits its object", entry.target_name),
            )
        })?;
        by_object.entry(object.clone()).or_default().push(entry);
    }

    let mut catalog = Vec::with_capacity(by_object.len());
    let mut stage_objects = BTreeMap::<String, Vec<String>>::new();
    let mut compact_payload_bytes = 0_u64;
    let mut artifact_bytes = 0_u64;
    let mut padding_elements = 0_u64;
    let mut padded_elements = 0_u64;
    for (path, mut object_entries) in by_object {
        object_entries.sort_by(|left, right| left.target_name.cmp(&right.target_name));
        let file = weight_files.get(path.as_str()).ok_or_else(|| {
            contract(
                "packed-f16-denoiser",
                format!("packed-F16 tensor catalog references unknown object {path}"),
            )
        })?;
        if file.size > max_bytes {
            return Err(contract(
                "packed-f16-denoiser",
                format!("packed-F16 object {path} exceeds the sealed read cap"),
            ));
        }
        let stage = object_entries[0].stage.clone();
        validate_object_stage(
            &stage,
            &path,
            object_entries.iter().map(|entry| entry.stage.as_str()),
            file.component.as_ref().map(|component| component.as_str()),
        )?;

        let tensor_shapes = object_entries
            .iter()
            .map(|entry| {
                let shape = entry.stored_shape.clone().expect("shape checked above");
                let elements = shape
                    .iter()
                    .try_fold(1_usize, |count, dimension| count.checked_mul(*dimension))
                    .ok_or_else(|| contract(&stage, "packed-F16 tensor element count overflow"))?;
                Ok((shape, elements))
            })
            .collect::<Result<Vec<_>, BooguArtifactLoadError>>()?;
        let layout = PackedF16Layout::try_from_element_counts(
            tensor_shapes.iter().map(|(_, elements)| elements),
        )
        .map_err(|error| {
            contract(
                &stage,
                format!("packed-F16 object {path} layout failed: {error}"),
            )
        })?;
        let mut tensors = Vec::with_capacity(object_entries.len());
        for ((entry, (shape, elements)), view) in object_entries
            .into_iter()
            .zip(tensor_shapes)
            .zip(layout.tensors())
        {
            tensors.push(PackedTensorDescriptor {
                target_name: entry.target_name.clone(),
                shape,
                offset_elements: view.offset_elements(),
                elements,
                digest: entry.stored_sha256.expect("digest checked above"),
            });
        }
        compact_payload_bytes = compact_payload_bytes
            .checked_add((layout.compact_elements() as u64) * 2)
            .ok_or_else(|| contract(&stage, "packed-F16 payload counter overflow"))?;
        padding_elements = padding_elements
            .checked_add(layout.padding_elements() as u64)
            .ok_or_else(|| contract(&stage, "packed-F16 padding counter overflow"))?;
        padded_elements = padded_elements
            .checked_add(layout.padded_elements() as u64)
            .ok_or_else(|| contract(&stage, "packed-F16 padded counter overflow"))?;
        artifact_bytes += file.size;
        stage_objects
            .entry(stage.clone())
            .or_default()
            .push(path.clone());
        catalog.push(PackedObjectDescriptor {
            file: (*file).clone(),
            stage,
            tensors,
            f16_elements: layout.padded_elements(),
            packed_bytes: layout.raw_bytes(),
            f32_bytes: layout.f32_bytes(),
        });
    }
    catalog.sort_by(|left, right| {
        (
            &left.stage,
            left.file.shard.map(|shard| shard.index),
            left.file.path.as_str(),
        )
            .cmp(&(
                &right.stage,
                right.file.shard.map(|shard| shard.index),
                right.file.path.as_str(),
            ))
    });
    for objects in stage_objects.values_mut() {
        objects.sort_by_key(|path| {
            let file = weight_files[path.as_str()];
            (file.shard.map(|shard| shard.index), path.clone())
        });
    }

    let retained_bytes = catalog
        .iter()
        .map(|object| object.packed_bytes)
        .sum::<u64>();
    let f32_write_bytes = catalog.iter().map(|object| object.f32_bytes).sum::<u64>();
    let tensor_count = catalog
        .iter()
        .map(|object| object.tensors.len())
        .sum::<usize>();
    let exact = stage_objects.len() == TURBO_PACKED_F16_STAGE_COUNT
        && catalog.len() == TURBO_PACKED_F16_OBJECT_COUNT
        && tensor_count == TURBO_PACKED_F16_TENSOR_COUNT
        && compact_payload_bytes == TURBO_PACKED_F16_COMPACT_PAYLOAD_BYTES
        && artifact_bytes == TURBO_PACKED_F16_ARTIFACT_BYTES
        && padding_elements == TURBO_PACKED_F16_PADDING_ELEMENTS
        && padded_elements == TURBO_PACKED_F16_PADDED_ELEMENTS
        && retained_bytes == TURBO_PACKED_F16_RETAINED_BYTES
        && f32_write_bytes == TURBO_PACKED_F16_F32_WRITE_BYTES_PER_DMD;
    if !exact {
        return Err(contract(
            "packed-f16-denoiser",
            format!(
                "Turbo packed catalog totals differ from the exact release: stages={}, objects={}, tensors={tensor_count}, compact={compact_payload_bytes}, artifact={artifact_bytes}, padding={padding_elements}, padded={padded_elements}, retained={retained_bytes}, f32={f32_write_bytes}",
                stage_objects.len(),
                catalog.len(),
            ),
        ));
    }
    Ok((catalog, stage_objects))
}

fn validate_exact_names<'a, I>(
    stage: &str,
    expected: &BTreeSet<&'a str>,
    actual: I,
) -> Result<(), BooguArtifactLoadError>
where
    I: IntoIterator<Item = &'a str>,
{
    let actual = actual.into_iter().collect::<Vec<_>>();
    let actual_set = actual.iter().copied().collect::<BTreeSet<_>>();
    if actual_set != *expected || actual_set.len() != actual.len() {
        return Err(contract(
            stage,
            "tensor set contains an unknown, duplicate, or missing exact path",
        ));
    }
    Ok(())
}

fn validate_object_stage<'a, I>(
    stage: &str,
    path: &str,
    tensor_stages: I,
    object_component: Option<&str>,
) -> Result<(), BooguArtifactLoadError>
where
    I: IntoIterator<Item = &'a str>,
{
    if object_component != Some(stage) || tensor_stages.into_iter().any(|actual| actual != stage) {
        return Err(contract(
            stage,
            format!("packed-F16 object {path} mixes or mislabels semantic stages"),
        ));
    }
    Ok(())
}

fn validate_object_keyset(
    path: &str,
    expected: &BTreeSet<&str>,
    actual: &BTreeSet<&str>,
) -> Result<(), BooguError> {
    if actual != expected {
        return Err(BooguError::Artifact(format!(
            "packed-F16 object {path} tensor keyset differs from its sealed catalog"
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_tensor_payload(
    tensor: &PackedTensorDescriptor,
    path: &str,
    snapshot_dtype: DType,
    snapshot_shape: &[usize],
    data_dtype: DType,
    data_shape: &[usize],
    bytes: &[u8],
) -> Result<(), BooguError> {
    if snapshot_dtype != DType::F16 || data_dtype != DType::F16 {
        return Err(BooguError::Artifact(format!(
            "packed-F16 tensor {} in {path} has dtype {snapshot_dtype:?}/{data_dtype:?}, expected F16",
            tensor.target_name
        )));
    }
    if snapshot_shape != tensor.shape || data_shape != tensor.shape {
        return Err(BooguError::Artifact(format!(
            "packed-F16 tensor {} in {path} differs from sealed shape {:?}",
            tensor.target_name, tensor.shape
        )));
    }
    let expected_bytes = tensor
        .elements
        .checked_mul(2)
        .ok_or_else(|| BooguError::Artifact("F16 tensor byte-size overflow".into()))?;
    if bytes.len() != expected_bytes {
        return Err(BooguError::Artifact(format!(
            "packed-F16 tensor {} in {path} has {} payload bytes, expected {expected_bytes}",
            tensor.target_name,
            bytes.len()
        )));
    }
    let actual_digest = Sha256Digest::calculate(bytes);
    if actual_digest != tensor.digest {
        return Err(BooguError::Artifact(format!(
            "packed-F16 tensor {} in {path} digest {actual_digest} differs from sealed {}",
            tensor.target_name, tensor.digest
        )));
    }
    Ok(())
}

struct PackedWordBuilder {
    words: Vec<u32>,
    low: Option<u16>,
    elements: usize,
}

impl PackedWordBuilder {
    fn with_capacity(words: usize) -> Self {
        Self {
            words: Vec::with_capacity(words),
            low: None,
            elements: 0,
        }
    }

    fn zero_pad_to(&mut self, offset: usize) -> Result<(), BooguError> {
        if offset < self.elements {
            return Err(BooguError::Artifact(format!(
                "packed-F16 layout moved backwards from {} to {offset}",
                self.elements
            )));
        }
        while self.elements < offset {
            self.push_half(0);
        }
        Ok(())
    }

    fn push_f16_bytes(&mut self, bytes: &[u8]) -> Result<(), BooguError> {
        let chunks = bytes.chunks_exact(2);
        if !chunks.remainder().is_empty() {
            return Err(BooguError::Artifact(
                "canonical F16 tensor has an odd byte count".into(),
            ));
        }
        for bytes in chunks {
            self.push_half(u16::from_le_bytes([bytes[0], bytes[1]]));
        }
        Ok(())
    }

    fn push_half(&mut self, half: u16) {
        if let Some(low) = self.low.take() {
            self.words.push(u32::from(low) | (u32::from(half) << 16));
        } else {
            self.low = Some(half);
        }
        self.elements += 1;
    }

    fn finish(mut self) -> Result<Vec<u32>, BooguError> {
        if let Some(low) = self.low.take() {
            self.words.push(u32::from(low));
        }
        Ok(self.words)
    }
}

fn single_block(
    config: &BooguConfig,
    modulation: bool,
    device: &WgpuDevice,
) -> SingleStreamBlock<PackedF16DenoiserBackend> {
    SingleStreamBlock::new(
        config.hidden_size,
        config.ffn_inner_dim(),
        config.num_attention_heads,
        config.num_kv_heads,
        config.hidden_size.min(1024),
        config.norm_eps,
        modulation,
        device,
    )
}

fn packed_raw_upload_options(
    device: &WgpuDevice,
) -> TensorCreationOptions<PackedF16DenoiserBackend> {
    // Passing only `device` converts U32 TensorData to the backend's default IntElem (I32).
    // The widening kernel consumes the bit pattern as unsigned words, so pin the creation dtype.
    TensorCreationOptions::new(device.clone()).with_dtype(DType::U32)
}

fn contract(stage: &str, message: impl Into<String>) -> BooguArtifactLoadError {
    BooguArtifactLoadError::Contract {
        stage: stage.into(),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_turbo_used_inventory_and_padded_totals_correctness() {
        let inventory = BooguArtifactInventory::denoiser(&BooguConfig::default()).unwrap();
        let used = inventory
            .tensors()
            .iter()
            .filter(|spec| !spec.stage.starts_with("boogu-reference-refiner-"))
            .collect::<Vec<_>>();
        assert_eq!(used.len(), TURBO_PACKED_F16_TENSOR_COUNT);
        assert_eq!(
            used.iter()
                .map(|spec| spec.stage.as_str())
                .collect::<BTreeSet<_>>()
                .len(),
            TURBO_PACKED_F16_STAGE_COUNT
        );
        let compact = used
            .iter()
            .map(|spec| spec.target_shape.iter().product::<usize>() as u64 * 2)
            .sum::<u64>();
        assert_eq!(compact, TURBO_PACKED_F16_COMPACT_PAYLOAD_BYTES);
        assert_eq!(
            TURBO_PACKED_F16_RETAINED_BYTES - TURBO_PACKED_F16_COMPACT_PAYLOAD_BYTES,
            TURBO_PACKED_F16_PADDING_ELEMENTS * 2
        );
        assert_eq!(
            TURBO_PACKED_F16_F32_WRITE_BYTES_PER_DMD,
            TURBO_PACKED_F16_PADDED_ELEMENTS * 4
        );
    }

    #[test]
    fn packed_word_builder_preserves_little_endian_halves_and_zero_padding_correctness() {
        let mut builder = PackedWordBuilder::with_capacity(3);
        builder.push_f16_bytes(&[0x34, 0x12]).unwrap();
        builder.zero_pad_to(2).unwrap();
        builder.push_f16_bytes(&[0xcd, 0xab, 0x02, 0x01]).unwrap();
        assert_eq!(builder.finish().unwrap(), vec![0x0000_1234, 0x0102_abcd]);
    }

    #[test]
    fn packed_word_builder_rejects_overlap_and_odd_payload_correctness() {
        let mut builder = PackedWordBuilder::with_capacity(1);
        builder.push_f16_bytes(&[0, 0]).unwrap();
        assert!(builder.zero_pad_to(0).is_err());
        assert!(builder.push_f16_bytes(&[0]).is_err());
    }

    #[test]
    fn packed_raw_upload_explicitly_overrides_i32_backend_default_correctness() {
        assert_eq!(
            std::any::TypeId::of::<
                <PackedF16DenoiserBackend as burn::tensor::backend::BackendTypes>::IntElem,
            >(),
            std::any::TypeId::of::<i32>()
        );
        let data = TensorData::new(vec![0x8000_0001_u32], [1]);
        assert_eq!(data.dtype, DType::U32);
        let device = WgpuDevice::default();
        let default_options: TensorCreationOptions<PackedF16DenoiserBackend> = (&device).into();
        assert_eq!(default_options.dtype, None);
        assert_eq!(packed_raw_upload_options(&device).dtype, Some(DType::U32));
    }

    #[test]
    fn exact_name_and_object_stage_contracts_reject_unknown_duplicate_missing_correctness() {
        let expected = ["tensor.a", "tensor.b"]
            .into_iter()
            .collect::<BTreeSet<_>>();
        validate_exact_names("stage", &expected, ["tensor.a", "tensor.b"]).unwrap();
        assert!(validate_exact_names("stage", &expected, ["tensor.a", "tensor.a"]).is_err());
        assert!(validate_exact_names("stage", &expected, ["tensor.a"]).is_err());
        assert!(validate_exact_names("stage", &expected, ["tensor.a", "tensor.unknown"]).is_err());

        validate_object_stage("stage", "object.bpk", ["stage", "stage"], Some("stage")).unwrap();
        assert!(
            validate_object_stage(
                "stage",
                "object.bpk",
                ["stage", "wrong-stage"],
                Some("stage")
            )
            .is_err()
        );
        assert!(
            validate_object_stage("stage", "object.bpk", ["stage"], Some("wrong-stage")).is_err()
        );
    }

    #[test]
    fn object_keyset_and_payload_contract_reject_shape_dtype_digest_correctness() {
        let expected = ["tensor.a", "tensor.b"]
            .into_iter()
            .collect::<BTreeSet<_>>();
        validate_object_keyset("object.bpk", &expected, &expected).unwrap();
        let missing = ["tensor.a"].into_iter().collect::<BTreeSet<_>>();
        let unknown = ["tensor.a", "tensor.c"]
            .into_iter()
            .collect::<BTreeSet<_>>();
        assert!(validate_object_keyset("object.bpk", &expected, &missing).is_err());
        assert!(validate_object_keyset("object.bpk", &expected, &unknown).is_err());

        let bytes = [0_u8, 0, 1, 0];
        let descriptor = PackedTensorDescriptor {
            target_name: "tensor.a".into(),
            shape: vec![2],
            offset_elements: 0,
            elements: 2,
            digest: Sha256Digest::calculate(&bytes),
        };
        validate_tensor_payload(
            &descriptor,
            "object.bpk",
            DType::F16,
            &[2],
            DType::F16,
            &[2],
            &bytes,
        )
        .unwrap();
        assert!(
            validate_tensor_payload(
                &descriptor,
                "object.bpk",
                DType::F32,
                &[2],
                DType::F16,
                &[2],
                &bytes
            )
            .is_err()
        );
        assert!(
            validate_tensor_payload(
                &descriptor,
                "object.bpk",
                DType::F16,
                &[1, 2],
                DType::F16,
                &[2],
                &bytes
            )
            .is_err()
        );
        assert!(
            validate_tensor_payload(
                &descriptor,
                "object.bpk",
                DType::F16,
                &[2],
                DType::F16,
                &[2],
                &[0, 0]
            )
            .is_err()
        );
        assert!(
            validate_tensor_payload(
                &descriptor,
                "object.bpk",
                DType::F16,
                &[2],
                DType::F16,
                &[2],
                &[0, 0, 2, 0]
            )
            .is_err()
        );
    }
}
