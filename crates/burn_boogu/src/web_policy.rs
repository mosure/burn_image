//! Browser execution policy for Boogu model composition.
//!
//! This module is browser-aware but frontend-agnostic. Bevy owns device sharing, transport, and
//! UI events; the model crate owns which stages, dtypes, quantization paths, and residency form a
//! valid Boogu execution profile.

use burn::tensor::DType;
use burn_qwen3_vl::{
    Qwen3VlEmbeddingExecutionPolicy, Qwen3VlTextBlockLoadSynchronizationPolicy,
    Qwen3VlTextLayerAllocationPolicy,
};

use crate::{
    BooguQuantizedLinearExecutionPolicy, BooguRuntimeDTypes, BooguVariant,
    artifacts::{
        BooguDenoiserRuntimeQuantizationPolicy, BooguFloatLoadPolicy, BooguQuantizedLoadPolicy,
        BooguStorageProfile,
    },
    deployment::{BooguDeploymentSettings, BrowserBooguResidencyPolicy},
};

/// Exact packed-F16 Qwen allocator handoff policy.
pub const PACKED_F16_QWEN_HANDOFF_POLICY: &str = "qwen-per-stage-cleanup-disabled/exact-f32-instruction-host-handoff/async-webgpu-sync/backend-memory-cleanup/async-webgpu-sync/exact-f32-reupload/post-upload-digest-verify/packed-cache-reaudit";
/// Exact packed-F16 DMD-to-VAE allocator handoff policy.
pub const PACKED_F16_DMD_VAE_HANDOFF_POLICY: &str = "exact-f32-final-latent-host-handoff/drop-dmd-input-handles/pre-clear-async-webgpu-sync/clear-packed-source-wrapper-rope/async-webgpu-sync/backend-memory-cleanup/async-webgpu-sync/require-empty-packed-cache/exact-f32-reupload/post-upload-digest-verify";
/// Packed-F16 request rehydration contract.
pub const PACKED_F16_NEXT_REQUEST_REHYDRATION_POLICY: &str =
    "ensure-preloaded-low-vram-denoiser/verified-persistent-cache-storage/bounded-object-replay";
/// Packed-F16 provenance suffix.
pub const PACKED_F16_PROVENANCE_SUFFIX: &str = "request-scoped-packed-cache-evicted-before-vae";
/// Shared-surface provenance suffix.
pub const SURFACE_INFERENCE_PROVENANCE_SUFFIX: &str = "request-scoped-surface-acquire-suspended";
/// Exact packed-F16 Qwen submission policy.
pub const PACKED_F16_QWEN_TEXT_LAYER_SUBMISSION_POLICY: &str = "explicit-pre-forward-upload-barrier/explicit-post-forward-barrier/bounded-task-batches/per-submit-error-scopes";
/// Ordinary backend-managed Qwen submission policy.
pub const DEFAULT_QWEN_TEXT_LAYER_SUBMISSION_POLICY: &str = "backend-default";

/// Physical denoiser source used by one browser execution profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrowserDenoiserExecutionKind {
    /// Standard verified Burn module snapshots.
    VerifiedBurnModule,
    /// Packed-F16 source widened to dense F32 one semantic stage at a time.
    PackedF16DeviceWidenDenseF32,
}

/// Complete model-owned policy selected before a browser runtime is constructed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BrowserExecutionPolicy {
    /// Qwen floating-point artifact mapping.
    pub qwen_float: BooguFloatLoadPolicy,
    /// Qwen stored-quantized artifact mapping.
    pub qwen_quantized: BooguQuantizedLoadPolicy,
    /// Qwen embedding gather execution route.
    pub qwen_embedding_execution: Qwen3VlEmbeddingExecutionPolicy,
    /// Qwen text-layer allocation contract.
    pub qwen_text_layer_allocation: Qwen3VlTextLayerAllocationPolicy,
    /// Qwen text-block synchronization contract.
    pub qwen_text_block_load_synchronization: Qwen3VlTextBlockLoadSynchronizationPolicy,
    /// Qwen text-layer queue-submission provenance label.
    pub qwen_text_layer_submission_policy: &'static str,
    /// VAE floating-point artifact mapping.
    pub vae_float: BooguFloatLoadPolicy,
    /// Denoiser floating-point artifact mapping.
    pub denoiser_float: BooguFloatLoadPolicy,
    /// Denoiser stored-quantized artifact mapping.
    pub denoiser_quantized: BooguQuantizedLoadPolicy,
    /// Optional runtime quantization applied to denoiser weights.
    pub denoiser_runtime_quantization: BooguDenoiserRuntimeQuantizationPolicy,
    /// Adapter used by the outer retained-module wrapper.
    pub denoiser_retaining_wrapper_adapter: BooguQuantizedLinearExecutionPolicy,
    /// Physical denoiser source representation.
    pub denoiser_execution_kind: BrowserDenoiserExecutionKind,
    /// Model-weight residency contract.
    pub residency: BrowserBooguResidencyPolicy,
    /// Whether Qwen stages remain device resident.
    pub retain_qwen_stages: bool,
    /// Whether VAE stages remain device resident.
    pub retain_vae_stages: bool,
    /// Whether denoiser stages remain device resident.
    pub retain_denoiser_stages: bool,
    /// Whether all active stages load before the first request.
    pub eager_preload: bool,
    /// Whether the denoiser loads before each request boundary.
    pub preload_denoiser_before_request: bool,
    /// Whether retained Qwen synchronization is deferred.
    pub defer_retained_qwen_synchronization: bool,
    /// Whether retained denoiser synchronization is deferred.
    pub defer_retained_denoiser_synchronization: bool,
    /// Whether browser Cache Storage is mandatory.
    pub require_persistent_range_cache: bool,
    /// Whether Qwen may release device memory after each stage.
    pub release_unused_qwen_memory_after_stage: bool,
    /// Whether backend memory cleanup runs at model-phase boundaries.
    pub phase_boundary_memory_cleanup: bool,
    /// Whether the exact packed-F16 Qwen host handoff is active.
    pub packed_qwen_instruction_handoff: bool,
    /// Whether rendered surface acquisition is suspended during inference.
    pub request_scoped_surface_acquire_suspended: bool,
}

impl BrowserExecutionPolicy {
    /// Whether the denoiser source uses packed F16 with bounded dense-F32 stage views.
    pub const fn uses_packed_f16_denoiser_source(self) -> bool {
        matches!(
            self.denoiser_execution_kind,
            BrowserDenoiserExecutionKind::PackedF16DeviceWidenDenseF32
        )
    }

    /// Whether Qwen embeddings are selected on the host before one compact upload.
    pub const fn uses_host_routed_qwen_embedding(self) -> bool {
        matches!(
            self.qwen_embedding_execution,
            Qwen3VlEmbeddingExecutionPolicy::ExactHostRoutedF16ToF32 { .. }
        )
    }

    /// Whether a request boundary must preload the packed-F16 denoiser.
    pub const fn requires_packed_f16_request_preload(self) -> bool {
        self.preload_denoiser_before_request && self.uses_packed_f16_denoiser_source()
    }

    /// Stable packed-F16 Qwen handoff policy label, or `disabled`.
    pub const fn packed_qwen_instruction_handoff_policy(self) -> &'static str {
        if self.packed_qwen_instruction_handoff {
            PACKED_F16_QWEN_HANDOFF_POLICY
        } else {
            "disabled"
        }
    }

    /// Stable packed-F16 DMD-to-VAE handoff policy label, or `disabled`.
    pub const fn packed_f16_dmd_vae_handoff_policy(self) -> &'static str {
        if self.uses_packed_f16_denoiser_source() {
            PACKED_F16_DMD_VAE_HANDOFF_POLICY
        } else {
            "disabled"
        }
    }

    /// Stable backend provenance label for this execution policy.
    pub fn provenance_backend(self) -> String {
        let backend = if self.uses_packed_f16_denoiser_source() {
            format!(
                "burn-webgpu/{}/{}",
                self.residency.label(),
                PACKED_F16_PROVENANCE_SUFFIX
            )
        } else {
            format!("burn-webgpu/{}", self.residency.label())
        };
        if self.request_scoped_surface_acquire_suspended {
            format!("{backend}/{SURFACE_INFERENCE_PROVENANCE_SUFFIX}")
        } else {
            backend
        }
    }

    /// Whether every packed-F16 allocator and synchronization invariant is exact.
    pub fn packed_allocator_policy_is_exact(self) -> bool {
        !self.release_unused_qwen_memory_after_stage
            && self.packed_qwen_instruction_handoff == self.uses_packed_f16_denoiser_source()
            && self.qwen_text_layer_allocation
                == if self.uses_packed_f16_denoiser_source() {
                    Qwen3VlTextLayerAllocationPolicy::ExactSizePersistent
                } else {
                    Qwen3VlTextLayerAllocationPolicy::BackendDefault
                }
            && self.qwen_text_block_load_synchronization
                == if self.uses_packed_f16_denoiser_source() {
                    Qwen3VlTextBlockLoadSynchronizationPolicy::PreForwardAndPostForward
                } else {
                    Qwen3VlTextBlockLoadSynchronizationPolicy::PostForwardOnly
                }
            && self.qwen_text_layer_submission_policy
                == if self.uses_packed_f16_denoiser_source() {
                    PACKED_F16_QWEN_TEXT_LAYER_SUBMISSION_POLICY
                } else {
                    DEFAULT_QWEN_TEXT_LAYER_SUBMISSION_POLICY
                }
            && (!self.uses_packed_f16_denoiser_source()
                || (!self.retain_qwen_stages && !self.defer_retained_qwen_synchronization))
            && (matches!(
                self.qwen_embedding_execution,
                Qwen3VlEmbeddingExecutionPolicy::DeviceRoutedChunks
            ) || self.uses_host_routed_qwen_embedding())
    }

    /// Stable Qwen embedding execution label.
    pub const fn qwen_embedding_execution_policy(self) -> &'static str {
        match self.qwen_embedding_execution {
            Qwen3VlEmbeddingExecutionPolicy::DeviceRoutedChunks => "device-routed-row-chunks",
            Qwen3VlEmbeddingExecutionPolicy::ExactHostRoutedF16ToF32 {
                verify_device_roundtrip_before_text: true,
            } => {
                "authenticated-full-f16-row-objects/host-token-row-select/f16-to-f32/one-compact-upload/immediate-device-readback-before-text"
            }
            Qwen3VlEmbeddingExecutionPolicy::ExactHostRoutedF16ToF32 {
                verify_device_roundtrip_before_text: false,
            } => {
                "authenticated-full-f16-row-objects/host-token-row-select/f16-to-f32/one-compact-upload/no-device-readback"
            }
        }
    }

    /// Stable Qwen text-layer allocation label.
    pub const fn qwen_text_layer_allocation_policy(self) -> &'static str {
        self.qwen_text_layer_allocation.label()
    }

    /// Stable Qwen text-block synchronization label.
    pub const fn qwen_text_block_load_synchronization_policy(self) -> &'static str {
        self.qwen_text_block_load_synchronization.label()
    }

    /// Stable Qwen text-layer submission label.
    pub const fn qwen_text_layer_submission_policy(self) -> &'static str {
        self.qwen_text_layer_submission_policy
    }

    /// Apply ordinary interactive-browser cache and optional surface-gate requirements.
    pub const fn for_ordinary_browser_factory(mut self, surface_gate_requested: bool) -> Self {
        self.require_persistent_range_cache = true;
        self.request_scoped_surface_acquire_suspended = surface_gate_requested;
        self
    }

    /// Stable artifact and device weight-traffic contract.
    pub const fn weight_traffic_contract(self) -> &'static str {
        if self.eager_preload
            && matches!(
                self.denoiser_float,
                BooguFloatLoadPolicy::PackedQ4sWeightsF32Auxiliaries
            )
        {
            "eager-preload/qwen+vae+denoiser/resident-q4s-matrices+embedding+packed-f16-convolutions+f32-auxiliaries/zero-inference-artifact-transfers/no-model-unload"
        } else if self.eager_preload
            && matches!(
                self.denoiser_float,
                BooguFloatLoadPolicy::PackedF16WeightsF32Auxiliaries
            )
        {
            "eager-preload/qwen+vae+denoiser/resident-f16-weights/fused-f32-accumulate/zero-inference-artifact-transfers/zero-full-stage-widening"
        } else if self.eager_preload {
            "eager-preload/qwen+vae+denoiser/zero-inference-artifact-transfers"
        } else if self.require_persistent_range_cache
            && matches!(
                self.denoiser_retaining_wrapper_adapter,
                BooguQuantizedLinearExecutionPolicy::DenseF32PerSemanticStage
            )
        {
            "persistent-transport-part-cache/qwen+vae+denoiser-first-request/zero-repeat-network-required/retained-q8-denoiser-cache-hits-dmd-steps-2-through-4/dense-f32-materialized-per-semantic-stage"
        } else if self.require_persistent_range_cache
            && self.preload_denoiser_before_request
            && matches!(
                self.residency,
                BrowserBooguResidencyPolicy::LowVramPreloadedPackedF16Denoiser
            )
        {
            "persistent-transport-part-cache/qwen+vae+packed-f16-denoiser-rehydrated-before-each-request/zero-dmd-artifact-transfers/zero-repeat-network-required/request-scoped-packed-cache-evicted-before-vae/dense-f32-materialized-per-semantic-stage"
        } else if self.require_persistent_range_cache && self.residency.is_low_vram() {
            "persistent-transport-part-cache/qwen+vae+denoiser-first-request/zero-repeat-network-required/retained-q8-direct-matmul-denoiser-cache-hits-dmd-steps-2-through-4"
        } else if self.preload_denoiser_before_request
            && matches!(
                self.residency,
                BrowserBooguResidencyPolicy::LowVramPreloadedPackedF16Denoiser
            )
        {
            "diagnostic-per-request/qwen+vae/packed-f16-denoiser-rehydrated-before-each-request/zero-dmd-artifact-transfers/no-persistent-cache-claim/request-scoped-packed-cache-evicted-before-vae/dense-f32-materialized-per-semantic-stage"
        } else if self.residency.is_low_vram() {
            "per-request/qwen+vae+denoiser-first-dmd-step/denoiser-cache-hits-steps-2-through-4"
        } else if self.retain_denoiser_stages {
            "qualification-per-request/qwen+vae+denoiser-first-dmd-step/denoiser-cache-hits-steps-2-through-4"
        } else {
            "diagnostic-per-request/qwen+vae/denoiser-reloaded-every-dmd-step"
        }
    }

    /// Stable denoiser linear-kernel execution label.
    pub const fn denoiser_linear_execution_policy(self) -> &'static str {
        if self.uses_packed_f16_denoiser_source() {
            "packed-f16-storage/device-widen-f32-per-semantic-stage/dense-f32-matmul"
        } else if matches!(
            self.denoiser_float,
            BooguFloatLoadPolicy::PackedF16WeightsF32Auxiliaries
        ) {
            "resident-f16-storage/integer-unpack/fused-f32-accumulate-matmul"
        } else if matches!(
            self.denoiser_float,
            BooguFloatLoadPolicy::PackedQ4sWeightsF32Auxiliaries
        ) {
            "resident-q4s-storage/direct-quantized-matmul/f32-accumulate"
        } else {
            quantized_linear_execution_policy_name(self.denoiser_retaining_wrapper_adapter)
        }
    }

    /// Stable denoiser device-storage label.
    pub const fn denoiser_storage_policy(self) -> &'static str {
        if self.uses_packed_f16_denoiser_source() {
            "authenticated-compact-f16/padded-u32-retained/dense-f32-per-semantic-stage"
        } else if matches!(
            self.denoiser_float,
            BooguFloatLoadPolicy::PackedF16WeightsF32Auxiliaries
        ) {
            "verified-f16-matrix-convolution-buffers/f32-auxiliaries/no-full-stage-widening"
        } else if matches!(
            self.denoiser_float,
            BooguFloatLoadPolicy::PackedQ4sWeightsF32Auxiliaries
        ) {
            "verified-q4s-matrix-buffers/f32-block-scales/packed-f16-convolutions/f32-auxiliaries"
        } else {
            "standard-verified-burn-module-snapshots"
        }
    }

    /// Stable denoiser quantized-load report label.
    pub const fn denoiser_quantized_load_policy_report(self) -> &'static str {
        if self.uses_packed_f16_denoiser_source() {
            "not-applicable-packed-f16-storage"
        } else if matches!(
            self.residency,
            BrowserBooguResidencyPolicy::ResidentPackedQ4s
        ) && matches!(
            self.denoiser_runtime_quantization,
            BooguDenoiserRuntimeQuantizationPolicy::Disabled
        ) {
            "preserve-stored-q4s-block-up-to-128-f32"
        } else {
            denoiser_quantized_load_policy_name(
                self.denoiser_quantized,
                self.denoiser_runtime_quantization,
            )
        }
    }

    /// Stable retained-wrapper quantized execution label.
    pub const fn denoiser_quantized_linear_execution_policy_report(self) -> &'static str {
        if self.uses_packed_f16_denoiser_source() {
            "not-applicable-packed-f16-storage"
        } else {
            quantized_linear_execution_policy_name(self.denoiser_retaining_wrapper_adapter)
        }
    }

    /// Build the common request-streaming policy before a concrete residency is selected.
    fn request_streaming_base(settings: &BooguDeploymentSettings) -> Self {
        Self {
            qwen_float: BooguFloatLoadPolicy::AdaptToF32,
            qwen_quantized: settings.qwen_quantized_load_policy(),
            qwen_embedding_execution: Qwen3VlEmbeddingExecutionPolicy::DeviceRoutedChunks,
            qwen_text_layer_allocation: Qwen3VlTextLayerAllocationPolicy::BackendDefault,
            qwen_text_block_load_synchronization:
                Qwen3VlTextBlockLoadSynchronizationPolicy::PostForwardOnly,
            qwen_text_layer_submission_policy: DEFAULT_QWEN_TEXT_LAYER_SUBMISSION_POLICY,
            vae_float: BooguFloatLoadPolicy::AdaptToF32,
            denoiser_float: BooguFloatLoadPolicy::AdaptToF32,
            denoiser_quantized: settings.denoiser_quantized_load_policy(),
            denoiser_runtime_quantization: BooguDenoiserRuntimeQuantizationPolicy::Disabled,
            denoiser_retaining_wrapper_adapter:
                BooguQuantizedLinearExecutionPolicy::DirectQuantizedMatmul,
            denoiser_execution_kind: BrowserDenoiserExecutionKind::VerifiedBurnModule,
            residency: BrowserBooguResidencyPolicy::LowVramRuntimeQ8Denoiser,
            retain_qwen_stages: false,
            retain_vae_stages: false,
            retain_denoiser_stages: false,
            eager_preload: false,
            preload_denoiser_before_request: false,
            defer_retained_qwen_synchronization: false,
            defer_retained_denoiser_synchronization: false,
            require_persistent_range_cache: false,
            release_unused_qwen_memory_after_stage: false,
            phase_boundary_memory_cleanup: false,
            packed_qwen_instruction_handoff: false,
            request_scoped_surface_acquire_suspended: false,
        }
    }

    /// Build the verified F32 stage-probe policy used by the no-surface bootstrap check.
    pub fn stage_probe_f32(settings: &BooguDeploymentSettings) -> Self {
        Self::request_streaming_base(settings)
    }

    /// Build the explicit resident dense-F32 diagnostic policy.
    pub fn resident_dense_f32(settings: &BooguDeploymentSettings) -> Result<Self, &'static str> {
        if !matches!(
            settings.storage_profile,
            BooguStorageProfile::F16QwenVisionF32
        ) {
            return Err(
                "browser resident-dense F32 requires profile=f16-qwen-vision-f32; Q4 and Q8 profiles are unsupported for this diagnostic residency",
            );
        }
        let mut policy = Self::request_streaming_base(settings);
        policy.residency = BrowserBooguResidencyPolicy::HighVramResidentDenseF32;
        policy.retain_qwen_stages = true;
        policy.retain_vae_stages = true;
        policy.retain_denoiser_stages = true;
        policy.eager_preload = true;
        policy.preload_denoiser_before_request = true;
        policy.defer_retained_qwen_synchronization = true;
        policy.defer_retained_denoiser_synchronization = true;
        Ok(policy)
    }

    /// Build the resident packed-F16 validation policy.
    pub fn resident_packed_f16(settings: &BooguDeploymentSettings) -> Result<Self, &'static str> {
        if !matches!(
            settings.storage_profile,
            BooguStorageProfile::F16QwenVisionF32
        ) {
            return Err(
                "browser resident packed-F16 requires profile=f16-qwen-vision-f32; quantized and all-F16 profiles remain explicit diagnostics",
            );
        }
        let mut policy = Self::request_streaming_base(settings);
        policy.qwen_float = BooguFloatLoadPolicy::PackedF16WeightsF32Auxiliaries;
        policy.vae_float = BooguFloatLoadPolicy::PackedF16WeightsF32Auxiliaries;
        policy.denoiser_float = BooguFloatLoadPolicy::PackedF16WeightsF32Auxiliaries;
        policy.residency = BrowserBooguResidencyPolicy::HighVramResidentPackedF16;
        policy.retain_qwen_stages = true;
        policy.retain_vae_stages = true;
        policy.retain_denoiser_stages = true;
        policy.eager_preload = true;
        policy.preload_denoiser_before_request = true;
        policy.defer_retained_qwen_synchronization = true;
        policy.defer_retained_denoiser_synchronization = true;
        Ok(policy)
    }

    /// Build the canonical resident packed-Q4S policy for a public variant.
    pub fn resident_packed_q4s(
        _variant: BooguVariant,
        settings: &BooguDeploymentSettings,
    ) -> Result<Self, &'static str> {
        if !matches!(
            settings.storage_profile,
            BooguStorageProfile::Q4sBlockUpTo128F32
        ) {
            return Err("browser resident packed-Q4S requires the canonical Q4 artifact profile");
        }
        let mut policy = Self::request_streaming_base(settings);
        policy.qwen_float = BooguFloatLoadPolicy::PackedQ4sWeightsF32Auxiliaries;
        policy.qwen_quantized = BooguQuantizedLoadPolicy::Preserve;
        policy.vae_float = BooguFloatLoadPolicy::PackedQ4sWeightsF32Auxiliaries;
        policy.denoiser_float = BooguFloatLoadPolicy::PackedQ4sWeightsF32Auxiliaries;
        policy.denoiser_quantized = BooguQuantizedLoadPolicy::Preserve;
        policy.denoiser_runtime_quantization = BooguDenoiserRuntimeQuantizationPolicy::Disabled;
        policy.denoiser_retaining_wrapper_adapter =
            BooguQuantizedLinearExecutionPolicy::DirectQuantizedMatmul;
        policy.residency = BrowserBooguResidencyPolicy::ResidentPackedQ4s;
        policy.retain_qwen_stages = true;
        policy.retain_vae_stages = true;
        policy.retain_denoiser_stages = true;
        policy.eager_preload = true;
        policy.preload_denoiser_before_request = true;
        policy.defer_retained_qwen_synchronization = true;
        policy.defer_retained_denoiser_synchronization = true;
        policy.phase_boundary_memory_cleanup = true;
        Ok(policy)
    }

    /// Build the Edit-only request-scoped runtime-Q8 policy.
    pub fn low_vram_runtime_q8_denoiser(
        variant: BooguVariant,
        settings: &BooguDeploymentSettings,
    ) -> Result<Self, &'static str> {
        if matches!(variant, BooguVariant::Image01Turbo) {
            return Err(
                "Turbo all-eligible direct Q8 matmul is not numerically qualified; use the variant-aware default low-vram policy",
            );
        }
        if !matches!(
            settings.storage_profile,
            BooguStorageProfile::F16QwenVisionF32
        ) {
            return Err(
                "browser low-vram production requires profile=production; stored Q8 profiles remain explicit diagnostics",
            );
        }
        let mut policy = Self::request_streaming_base(settings);
        policy.denoiser_runtime_quantization =
            BooguDenoiserRuntimeQuantizationPolicy::Q8sBlock32F32;
        policy.retain_denoiser_stages = true;
        policy.residency = BrowserBooguResidencyPolicy::LowVramRuntimeQ8Denoiser;
        Ok(policy)
    }

    /// Build the exact Turbo packed-F16 stage-materialization low-VRAM policy.
    pub fn low_vram_preloaded_packed_f16_denoiser(
        variant: BooguVariant,
        settings: &BooguDeploymentSettings,
    ) -> Result<Self, &'static str> {
        if !matches!(variant, BooguVariant::Image01Turbo) {
            return Err(
                "browser preloaded packed-F16 dense-F32-per-stage low-VRAM policy is restricted to Turbo",
            );
        }
        if !matches!(
            settings.storage_profile,
            BooguStorageProfile::F16QwenVisionF32
        ) {
            return Err(
                "browser preloaded packed-F16 dense-F32-per-stage low-VRAM policy requires profile=production",
            );
        }
        let mut policy = Self::request_streaming_base(settings);
        policy.qwen_embedding_execution =
            Qwen3VlEmbeddingExecutionPolicy::ExactHostRoutedF16ToF32 {
                verify_device_roundtrip_before_text: true,
            };
        policy.qwen_text_layer_allocation = Qwen3VlTextLayerAllocationPolicy::ExactSizePersistent;
        policy.qwen_text_block_load_synchronization =
            Qwen3VlTextBlockLoadSynchronizationPolicy::PreForwardAndPostForward;
        policy.qwen_text_layer_submission_policy = PACKED_F16_QWEN_TEXT_LAYER_SUBMISSION_POLICY;
        policy.denoiser_execution_kind = BrowserDenoiserExecutionKind::PackedF16DeviceWidenDenseF32;
        policy.residency = BrowserBooguResidencyPolicy::LowVramPreloadedPackedF16Denoiser;
        policy.preload_denoiser_before_request = true;
        policy.packed_qwen_instruction_handoff = true;
        Ok(policy)
    }

    /// Build the exact 1.5K full-F32 qualification policy.
    pub fn exact_1k5_parity(settings: &BooguDeploymentSettings) -> Self {
        let mut policy = Self::request_streaming_base(settings);
        policy.residency = BrowserBooguResidencyPolicy::QualificationPerRequestF32DenoiserRetained;
        policy.retain_denoiser_stages = true;
        policy
    }

    /// Build the exact 1.5K low-VRAM qualification policy.
    pub fn exact_1k5_low_vram_parity(
        settings: &BooguDeploymentSettings,
    ) -> Result<Self, &'static str> {
        Self::low_vram_runtime_q8_denoiser(BooguVariant::Image01EditTurbo1k5, settings)
    }

    /// Build the explicit Qwen shader-F16 diagnostic policy.
    pub fn preserve_qwen_f16(settings: &BooguDeploymentSettings) -> Self {
        let mut policy = Self::request_streaming_base(settings);
        policy.qwen_float = BooguFloatLoadPolicy::Preserve;
        policy
    }

    /// Resolve runtime activation dtypes from this execution policy.
    pub fn execution_dtypes(self, profile: BooguStorageProfile) -> BooguRuntimeDTypes {
        let mut dtypes = BooguRuntimeDTypes::from_artifact_policies(
            profile,
            self.vae_float,
            self.denoiser_float,
        );
        if !matches!(self.qwen_float, BooguFloatLoadPolicy::Preserve) {
            dtypes.qwen_visual = DType::F32;
        }
        if matches!(
            self.vae_float,
            BooguFloatLoadPolicy::PackedF16WeightsF32Auxiliaries
                | BooguFloatLoadPolicy::PackedQ4sWeightsF32Auxiliaries
        ) {
            dtypes.vae = DType::F32;
        }
        dtypes
    }

    /// Parameter dtype expected by the VAE source for this policy.
    pub const fn vae_parameter_dtype(self) -> DType {
        match self.vae_float {
            BooguFloatLoadPolicy::AdaptToF32 => DType::F32,
            BooguFloatLoadPolicy::Preserve
            | BooguFloatLoadPolicy::PackedF16WeightsF32Auxiliaries
            | BooguFloatLoadPolicy::PackedQ4sWeightsF32Auxiliaries => DType::F16,
        }
    }
}

/// Stable report label for a floating-point artifact load policy.
pub const fn float_load_policy_name(policy: BooguFloatLoadPolicy) -> &'static str {
    match policy {
        BooguFloatLoadPolicy::Preserve => "preserve",
        BooguFloatLoadPolicy::AdaptToF32 => "adapt-to-f32",
        BooguFloatLoadPolicy::PackedF16WeightsF32Auxiliaries => {
            "packed-f16-weights-f32-auxiliaries"
        }
        BooguFloatLoadPolicy::PackedQ4sWeightsF32Auxiliaries => {
            "packed-q4s-weights-f32-auxiliaries"
        }
    }
}

/// Stable report label for a stored-quantized artifact load policy.
pub const fn quantized_load_policy_name(policy: BooguQuantizedLoadPolicy) -> &'static str {
    match policy {
        BooguQuantizedLoadPolicy::Preserve => "preserve",
        BooguQuantizedLoadPolicy::DequantizeF16 => "dequantize-f16",
    }
}

/// Stable report label for the effective denoiser quantization policy.
pub const fn denoiser_quantized_load_policy_name(
    stored: BooguQuantizedLoadPolicy,
    runtime: BooguDenoiserRuntimeQuantizationPolicy,
) -> &'static str {
    match runtime {
        BooguDenoiserRuntimeQuantizationPolicy::Q8sBlock32F32
        | BooguDenoiserRuntimeQuantizationPolicy::Q4sBlockUpTo128F32 => runtime.label(),
        _ => quantized_load_policy_name(stored),
    }
}

/// Stable report label for retained quantized-linear execution.
pub const fn quantized_linear_execution_policy_name(
    policy: BooguQuantizedLinearExecutionPolicy,
) -> &'static str {
    match policy {
        BooguQuantizedLinearExecutionPolicy::DirectQuantizedMatmul => "direct-quantized-matmul",
        BooguQuantizedLinearExecutionPolicy::DenseF32PerSemanticStage => {
            "retained-q8-dense-f32-per-semantic-stage"
        }
    }
}

#[cfg(test)]
mod tests {
    use burn_image::{ArtifactSource, RemoteBaseUrl};

    use super::*;

    fn production() -> BooguDeploymentSettings {
        BooguDeploymentSettings::production(
            BooguVariant::Image01Turbo,
            ArtifactSource::Remote {
                base_url: RemoteBaseUrl::new("https://cdn.example/q4").unwrap(),
            },
        )
    }

    #[test]
    fn public_variants_select_resident_q4_policy_correctness() {
        let settings = production();
        for variant in [
            BooguVariant::Image01Turbo,
            BooguVariant::Image01EditTurbo,
            BooguVariant::Image01EditTurbo1k5,
        ] {
            let policy = BrowserExecutionPolicy::resident_packed_q4s(variant, &settings).unwrap();
            assert_eq!(
                policy.residency,
                BrowserBooguResidencyPolicy::ResidentPackedQ4s
            );
            assert!(policy.eager_preload);
            assert!(policy.retain_qwen_stages && policy.retain_vae_stages);
            assert!(policy.retain_denoiser_stages);
            assert!(policy.weight_traffic_contract().contains("no-model-unload"));
        }
    }
}
