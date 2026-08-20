//! Canonical Boogu release selection and runtime artifact policy.
//!
//! Frontends supply a platform-specific artifact source, but model/profile selection and load
//! policy remain authoritative here. This keeps native CLIs, Bevy, and browser entry points from
//! independently choosing different production formats.

use burn_image::{
    ArtifactCachePolicy, ArtifactProfileId, ArtifactSource, Dimensions, IntegrityPolicy,
    ModelDescriptor, ModelId, NumericFormat, RuntimeConfig,
};
use serde::{Deserialize, Serialize};

use crate::{
    BooguConfig, BooguError, BooguVariant,
    artifacts::{
        BooguFloatLoadPolicy, BooguQuantizedLoadPolicy, BooguStorageProfile,
        preferred_artifact_bundle_id, source_artifact_bundle_id,
    },
    config::{
        BOOGU_1K_NATIVE_POLICY, EDIT_TURBO_1K5_NATIVE_POLICY, NativeAutotunePolicy,
        NativeHighVramPolicy,
    },
    pipeline::VaeDecoderMemoryPolicy,
    processing::boogu_model_descriptor,
};

/// Largest verified tensor buffer applied by the ordinary browser loader.
pub const BROWSER_MAX_APPLIED_BUFFER_BYTES: u64 = 414_892_032;
/// F32 `[1, 256, 1536, 1536]` feature buffer required by an untiled 1.5K VAE tail.
pub const BROWSER_1K5_VAE_FINAL_FEATURE_BUFFER_BYTES: u64 = 2_415_919_104;
/// Conservative maximum buffer for the released strict-F32 striped VAE tail.
pub const BROWSER_1K5_VAE_STRIPED_TAIL_BUFFER_BYTES: u64 = 1_215_832_064;
/// Largest measured F32 denoiser feed-forward buffer in the pinned 1.5K fixture.
pub const BROWSER_1K5_DENOISER_FFN_BUFFER_BYTES: u64 = 522_042_368;
/// Minimum applied buffer limit required by the released 1.5K browser plan.
pub const BROWSER_1K5_MIN_RUNTIME_BUFFER_BYTES: u64 = BROWSER_1K5_VAE_STRIPED_TAIL_BUFFER_BYTES;
/// Device-buffer limit requested for all released browser shapes.
pub const BROWSER_REQUESTED_BUFFER_LIMIT_BYTES: u64 = 1_217_126_400;
/// Minimum applied device-buffer limit required by exact 1.5K qualification.
pub const BROWSER_1K5_MIN_REQUIRED_BUFFER_LIMIT_BYTES: u64 = BROWSER_1K5_MIN_RUNTIME_BUFFER_BYTES;
/// Output side at which browser VAE decode uses the strict-F32 striped tail.
pub const BROWSER_STRIPED_VAE_MIN_OUTPUT_SIDE: u32 = 1024;
/// Exact edge replayed by the dedicated 1.5K browser parity route.
pub const BROWSER_1K5_PARITY_OUTPUT_EDGE: u32 = 1536;
/// Exact pixel count replayed by the dedicated 1.5K browser parity route.
pub const BROWSER_1K5_PARITY_OUTPUT_PIXELS: u64 =
    BROWSER_1K5_PARITY_OUTPUT_EDGE as u64 * BROWSER_1K5_PARITY_OUTPUT_EDGE as u64;

/// VAE decode strategy selected by the browser shape plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrowserVaeDecodePolicy {
    /// Decode the final VAE feature in one strict-F32 allocation.
    FullStrictF32,
    /// Decode the final VAE feature as two exact strict-F32 width slabs.
    StripedTailStrictF32 {
        /// Width coordinate separating the two slabs.
        split_width: usize,
    },
}

/// Largest individual buffers planned for one browser inference shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BrowserBufferPlan {
    /// Selected VAE tail policy.
    pub vae_decode_policy: BrowserVaeDecodePolicy,
    /// Largest VAE decode allocation.
    pub vae_decode_max_buffer_bytes: u64,
    /// Largest denoiser FFN allocation.
    pub denoiser_ffn_max_buffer_bytes: u64,
    /// Required maximum of every planned allocation.
    pub required_buffer_limit_bytes: u64,
}

/// Build the exact maximum-single-buffer plan for one released browser output shape.
pub fn browser_buffer_plan(
    variant: BooguVariant,
    dimensions: Dimensions,
) -> Result<BrowserBufferPlan, BooguError> {
    let model = model_id(variant);
    boogu_model_descriptor(variant)
        .capabilities
        .dimensions
        .supports(dimensions)
        .map_err(|error| BooguError::InvalidRequest(format!("{model}: {error}")))?;

    let width = u64::from(dimensions.width());
    let height = u64::from(dimensions.height());
    let uses_striped_tail = dimensions.width() >= BROWSER_STRIPED_VAE_MIN_OUTPUT_SIDE
        || dimensions.height() >= BROWSER_STRIPED_VAE_MIN_OUTPUT_SIDE;
    let (vae_decode_policy, vae_decode_max_buffer_bytes) = if uses_striped_tail {
        let split_width = dimensions.width() / 2;
        let largest_slab_width = u64::from(split_width.max(dimensions.width() - split_width));
        let bytes = 256_u64
            .checked_mul(height.checked_add(2).ok_or_else(|| {
                BooguError::InvalidRequest("browser VAE height plan overflowed".into())
            })?)
            .and_then(|value| value.checked_mul(largest_slab_width.checked_add(4)?))
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>() as u64))
            .ok_or_else(|| {
                BooguError::InvalidRequest("browser striped VAE buffer plan overflowed".into())
            })?;
        (
            BrowserVaeDecodePolicy::StripedTailStrictF32 {
                split_width: usize::try_from(split_width)
                    .map_err(|error| BooguError::InvalidRequest(error.to_string()))?,
            },
            bytes,
        )
    } else {
        let bytes = 256_u64
            .checked_mul(height)
            .and_then(|value| value.checked_mul(width))
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>() as u64))
            .ok_or_else(|| {
                BooguError::InvalidRequest("browser full VAE buffer plan overflowed".into())
            })?;
        (BrowserVaeDecodePolicy::FullStrictF32, bytes)
    };
    let image_tokens = width
        .checked_div(16)
        .and_then(|width| width.checked_mul(height / 16))
        .ok_or_else(|| {
            BooguError::InvalidRequest("browser denoiser token plan overflowed".into())
        })?;
    let joint_tokens = image_tokens.checked_add(1_280).ok_or_else(|| {
        BooguError::InvalidRequest("browser denoiser sequence plan overflowed".into())
    })?;
    let denoiser_ffn_max_buffer_bytes = joint_tokens
        .checked_mul(BooguConfig::default().ffn_inner_dim() as u64)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>() as u64))
        .ok_or_else(|| BooguError::InvalidRequest("browser denoiser FFN plan overflowed".into()))?;
    let required_buffer_limit_bytes = BROWSER_MAX_APPLIED_BUFFER_BYTES
        .max(denoiser_ffn_max_buffer_bytes)
        .max(vae_decode_max_buffer_bytes);
    Ok(BrowserBufferPlan {
        vae_decode_policy,
        vae_decode_max_buffer_bytes,
        denoiser_ffn_max_buffer_bytes,
        required_buffer_limit_bytes,
    })
}

/// Fail when either applied WebGPU limit cannot cover one selected shape.
pub fn validate_browser_buffer_limits_for_dimensions(
    variant: BooguVariant,
    dimensions: Dimensions,
    max_storage_buffer_binding_size: u64,
    max_buffer_size: u64,
) -> Result<BrowserBufferPlan, BooguError> {
    let plan = browser_buffer_plan(variant, dimensions)?;
    for (name, actual) in [
        (
            "max_storage_buffer_binding_size",
            max_storage_buffer_binding_size,
        ),
        ("max_buffer_size", max_buffer_size),
    ] {
        if actual < plan.required_buffer_limit_bytes {
            return Err(BooguError::InvalidRequest(format!(
                "browser {name} is {actual} bytes; {}x{} requires at least {} bytes (VAE decode plan {}, untiled 1.5K feature {}, shape-aware maximum denoiser FFN {})",
                dimensions.width(),
                dimensions.height(),
                plan.required_buffer_limit_bytes,
                plan.vae_decode_max_buffer_bytes,
                BROWSER_1K5_VAE_FINAL_FEATURE_BUFFER_BYTES,
                plan.denoiser_ffn_max_buffer_bytes,
            )));
        }
    }
    Ok(plan)
}

/// Validate applied limits against every shape advertised by one browser release.
pub fn validate_browser_variant_buffer_limits(
    variant: BooguVariant,
    max_storage_buffer_binding_size: u64,
    max_buffer_size: u64,
) -> Result<(), BooguError> {
    let descriptor = boogu_model_descriptor(variant);
    if let Some(allowed) = descriptor.capabilities.dimensions.allowed_dimensions {
        for dimensions in allowed {
            validate_browser_buffer_limits_for_dimensions(
                variant,
                dimensions,
                max_storage_buffer_binding_size,
                max_buffer_size,
            )?;
        }
    } else {
        let dimensions = Dimensions::new(
            descriptor.capabilities.dimensions.max_width,
            descriptor.capabilities.dimensions.max_height,
        )
        .expect("released Boogu maximum dimensions are valid");
        validate_browser_buffer_limits_for_dimensions(
            variant,
            dimensions,
            max_storage_buffer_binding_size,
            max_buffer_size,
        )?;
    }
    Ok(())
}

/// Artifact and cache policy for one concrete Boogu runtime.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BooguDeploymentSettings {
    /// Local directory or immutable remote bundle selected by the platform adapter.
    pub artifact_source: ArtifactSource,
    /// Exact sealed storage profile.
    pub storage_profile: BooguStorageProfile,
    /// Required artifact integrity policy.
    pub integrity: IntegrityPolicy,
    /// Required platform cache policy.
    pub cache: ArtifactCachePolicy,
}

/// Native model-weight residency selected before constructing a Boogu runtime.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NativeBooguResidencyPolicy {
    /// Retain every required stage on the GPU for warm repeated inference.
    #[default]
    HighVram,
    /// Stream Qwen/VAE by request while keeping the denoiser resident.
    LowVram,
}

impl NativeBooguResidencyPolicy {
    /// Stable provenance label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::HighVram => "native-high-vram-gpu-resident",
            Self::LowVram => "native-low-vram-phase-resident-mixed-f16",
        }
    }

    /// Whether this policy retains the complete request graph on the GPU.
    pub const fn is_gpu_resident(self) -> bool {
        matches!(self, Self::HighVram)
    }

    /// Steady-state model-weight traffic contract for runtime provenance.
    pub const fn weight_traffic_contract(self, production_profile: bool) -> &'static str {
        match self {
            Self::HighVram if production_profile => {
                "gpu-resident/zero-forward-host-weight-transfers"
            }
            Self::HighVram => {
                "diagnostic-gpu-resident-unqualified/zero-forward-host-weight-transfers"
            }
            Self::LowVram if production_profile => {
                "phase-resident/qwen+vae-per-request/denoiser-resident-zero-dmd-weight-reloads"
            }
            Self::LowVram => "unsupported-low-vram-profile/fail-closed-before-model-load",
        }
    }
}

/// Allocation behavior paired with one native resident storage profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeResidentAllocationPolicy {
    /// VAE decoder allocation strategy.
    pub vae_decoder: VaeDecoderMemoryPolicy,
    /// Whether dead activation/workspace pages are cleaned at model-phase boundaries.
    pub phase_boundary_cleanup: bool,
}

/// Preserve resident model weights while bounding dead activation/workspace allocations.
pub const fn native_resident_allocation_policy(
    profile: BooguStorageProfile,
) -> NativeResidentAllocationPolicy {
    if matches!(profile, BooguStorageProfile::Q4sBlockUpTo128F32) {
        NativeResidentAllocationPolicy {
            vae_decoder: VaeDecoderMemoryPolicy::ExactStripedTailWithStageCleanup,
            phase_boundary_cleanup: true,
        }
    } else {
        NativeResidentAllocationPolicy {
            vae_decoder: VaeDecoderMemoryPolicy::BackendDefault,
            phase_boundary_cleanup: false,
        }
    }
}

/// Resolve a parity-qualified native kernel policy for one variant/residency/profile tuple.
pub const fn qualified_native_execution_policy(
    variant: BooguVariant,
    residency: NativeBooguResidencyPolicy,
    profile: BooguStorageProfile,
) -> Option<NativeHighVramPolicy> {
    if !matches!(
        residency,
        NativeBooguResidencyPolicy::HighVram | NativeBooguResidencyPolicy::LowVram
    ) || !matches!(profile, BooguStorageProfile::F16QwenVisionF32)
    {
        return None;
    }
    match variant {
        BooguVariant::Image01Turbo | BooguVariant::Image01EditTurbo => Some(BOOGU_1K_NATIVE_POLICY),
        BooguVariant::Image01EditTurbo1k5 => Some(EDIT_TURBO_1K5_NATIVE_POLICY),
    }
}

/// Interactive 1K Edit query tile used by the explicitly unqualified balanced policy.
pub const NATIVE_BALANCED_1K_QWEN_QUERY_CHUNK_SIZE: usize = 1_024;

/// Resolve the native Qwen query tile without duplicating model policy in a frontend.
pub const fn native_qwen_query_chunk_size(
    variant: BooguVariant,
    residency: NativeBooguResidencyPolicy,
    autotune: NativeAutotunePolicy,
    policy: NativeHighVramPolicy,
) -> usize {
    if matches!(
        variant,
        BooguVariant::Image01Turbo | BooguVariant::Image01EditTurbo
    ) && matches!(residency, NativeBooguResidencyPolicy::HighVram)
        && (!cfg!(feature = "autotune") || matches!(autotune, NativeAutotunePolicy::Balanced))
    {
        NATIVE_BALANCED_1K_QWEN_QUERY_CHUNK_SIZE
    } else {
        policy.qwen_query_chunk_size
    }
}

/// Stable native kernel-policy label for output provenance.
pub fn native_kernel_policy_label(
    variant: BooguVariant,
    _policy: NativeHighVramPolicy,
    autotune: NativeAutotunePolicy,
) -> String {
    let suffix = match variant {
        BooguVariant::Image01Turbo | BooguVariant::Image01EditTurbo => {
            "1k-mixed-f16/qwen-q128/denoiser-padded-blackbox-p4-kv1-q1-q8192-rms-strict-f32-qk-balanced-strict-norm-rope/vae-q4096-f16-storage-f32-accum"
        }
        BooguVariant::Image01EditTurbo1k5 => {
            "1k5-mixed-f16/qwen-q128/denoiser-padded-blackbox-p4-kv1-q1-q16384-rms-strict-f32-qk-composed/vae-q4096-f16-storage-f32-accum"
        }
    };
    format!("{}/{suffix}", native_autotune_policy_label(autotune))
}

/// Stable complete native runtime-policy label for output provenance.
pub fn native_runtime_policy_label(
    policy: NativeHighVramPolicy,
    autotune: NativeAutotunePolicy,
    qwen_query_chunk_size: usize,
) -> String {
    let label = if !cfg!(feature = "autotune") {
        policy
            .provenance_label
            .replacen("full-autotune", "no-autotune-static-kernels", 1)
    } else {
        match autotune {
            NativeAutotunePolicy::Full => policy.provenance_label.to_owned(),
            NativeAutotunePolicy::Balanced => {
                policy
                    .provenance_label
                    .replacen("full-autotune", "balanced-autotune", 1)
            }
        }
    };
    if qwen_query_chunk_size == policy.qwen_query_chunk_size {
        label
    } else {
        label.replacen(
            &format!("qwen-q{}", policy.qwen_query_chunk_size),
            &format!("qwen-q{qwen_query_chunk_size}"),
            1,
        )
    }
}

const fn native_autotune_policy_label(autotune: NativeAutotunePolicy) -> &'static str {
    if !cfg!(feature = "autotune") {
        return "no-autotune-static-kernels";
    }
    match autotune {
        NativeAutotunePolicy::Balanced => "balanced-autotune",
        NativeAutotunePolicy::Full => "full-autotune",
    }
}

/// Browser model-weight residency selected before constructing a Boogu runtime.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BrowserBooguResidencyPolicy {
    /// Retain F16 matrix/convolution weights and execute fused F32-accumulate kernels.
    HighVramResidentPackedF16,
    /// Retain signed Q4S linear weights, packed-F16 convolutions, and F32 auxiliaries.
    #[default]
    ResidentPackedQ4s,
    /// Retain fully materialized dense-F32 stages for diagnostics.
    HighVramResidentDenseF32,
    /// Stream Qwen/VAE per request and retain the exact-fixture F32 denoiser.
    QualificationPerRequestF32DenoiserRetained,
    /// Stream Qwen/VAE with a request-scoped runtime-Q8 denoiser.
    LowVramRuntimeQ8Denoiser,
    /// Retain packed F16 and widen one denoiser stage at a time.
    LowVramPreloadedPackedF16Denoiser,
}

impl BrowserBooguResidencyPolicy {
    /// Stable runtime provenance label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::HighVramResidentPackedF16 => "browser-high-vram-resident-packed-f16",
            Self::ResidentPackedQ4s => "browser-resident-packed-q4s-block-up-to-128",
            Self::HighVramResidentDenseF32 => "browser-high-vram-resident-dense-f32",
            Self::QualificationPerRequestF32DenoiserRetained => {
                "browser-qualification-per-request-f32-denoiser-retained"
            }
            Self::LowVramRuntimeQ8Denoiser => "browser-low-vram-runtime-q8-denoiser",
            Self::LowVramPreloadedPackedF16Denoiser => {
                "browser-low-vram-preloaded-packed-f16-dense-f32-per-stage-denoiser"
            }
        }
    }

    /// Parse one exact public browser residency selector.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "resident" => Some(Self::ResidentPackedQ4s),
            "high-vram-resident-packed-f16" => Some(Self::HighVramResidentPackedF16),
            "resident-q4"
            | "resident-packed-q4s"
            | "browser-resident-packed-q4s-block-up-to-128" => Some(Self::ResidentPackedQ4s),
            "high-vram-resident-dense-f32" => Some(Self::HighVramResidentDenseF32),
            "qualification-f32"
            | "qualification-per-request-f32-denoiser-retained"
            | "browser-qualification-per-request-f32-denoiser-retained" => {
                Some(Self::QualificationPerRequestF32DenoiserRetained)
            }
            "low-vram-runtime-q8-denoiser" => Some(Self::LowVramRuntimeQ8Denoiser),
            "low-vram-preloaded-packed-f16-dense-f32-per-stage-denoiser" => {
                Some(Self::LowVramPreloadedPackedF16Denoiser)
            }
            _ => None,
        }
    }

    /// Whether this is a bounded request-scoped residency.
    pub const fn is_low_vram(self) -> bool {
        matches!(
            self,
            Self::LowVramRuntimeQ8Denoiser | Self::LowVramPreloadedPackedF16Denoiser
        )
    }

    /// Whether all request graph stages remain resident across requests.
    pub const fn is_high_vram_resident(self) -> bool {
        matches!(
            self,
            Self::HighVramResidentPackedF16
                | Self::ResidentPackedQ4s
                | Self::HighVramResidentDenseF32
        )
    }
}

/// Variant-aware fallback for the public `low-vram` browser selector.
pub const fn default_browser_low_vram_residency(
    variant: BooguVariant,
) -> BrowserBooguResidencyPolicy {
    if matches!(variant, BooguVariant::Image01Turbo) {
        BrowserBooguResidencyPolicy::LowVramPreloadedPackedF16Denoiser
    } else {
        BrowserBooguResidencyPolicy::LowVramRuntimeQ8Denoiser
    }
}

/// Resolve the public bounded-memory selector for one concrete released profile.
///
/// Direct Q4S is already the smaller resident Turbo representation, so routing it through the
/// mixed-F16 request-scoped widening path would add work and misstate provenance.
pub const fn browser_bounded_residency_policy(
    variant: BooguVariant,
    profile: BooguStorageProfile,
) -> BrowserBooguResidencyPolicy {
    if matches!(variant, BooguVariant::Image01Turbo)
        && matches!(profile, BooguStorageProfile::Q4sBlockUpTo128F32)
    {
        BrowserBooguResidencyPolicy::ResidentPackedQ4s
    } else {
        default_browser_low_vram_residency(variant)
    }
}

/// Canonical ordinary-browser residency for one released model variant.
pub const fn default_browser_residency(variant: BooguVariant) -> BrowserBooguResidencyPolicy {
    browser_bounded_residency_policy(variant, default_storage_profile(variant))
}

impl BooguDeploymentSettings {
    /// Construct the canonical production policy for one released variant.
    ///
    /// Turbo selects the sealed direct-Q4S transit and execution format. Edit releases retain
    /// mixed F16 until their Q4 execution policy has independent qualification.
    pub fn production(variant: BooguVariant, artifact_source: ArtifactSource) -> Self {
        Self {
            artifact_source,
            storage_profile: default_storage_profile(variant),
            integrity: IntegrityPolicy::RequireSha256,
            cache: ArtifactCachePolicy::UseCached,
        }
    }

    /// Construct the explicit mixed-F16 validation policy.
    pub fn mixed_f16(artifact_source: ArtifactSource) -> Self {
        Self {
            artifact_source,
            storage_profile: BooguStorageProfile::F16QwenVisionF32,
            integrity: IntegrityPolicy::RequireSha256,
            cache: ArtifactCachePolicy::UseCached,
        }
    }

    /// Construct the diagnostic all-F16 policy.
    pub fn f16(artifact_source: ArtifactSource) -> Self {
        Self {
            artifact_source,
            storage_profile: BooguStorageProfile::F16,
            integrity: IntegrityPolicy::RequireSha256,
            cache: ArtifactCachePolicy::UseCached,
        }
    }

    /// Canonical model-neutral runtime configuration for one release.
    pub fn runtime_config(&self, variant: BooguVariant) -> RuntimeConfig {
        RuntimeConfig {
            model: model_id(variant),
            artifact_profile: artifact_profile_id(self.storage_profile),
            artifact_source: self.artifact_source.clone(),
            integrity: self.integrity,
            cache: self.cache,
        }
    }

    /// Numeric-format identity represented by the selected bundle.
    pub fn numeric_format(&self) -> NumericFormat {
        numeric_format(self.storage_profile)
    }

    /// VAE float policy for ordinary execution.
    pub const fn vae_float_load_policy(&self) -> BooguFloatLoadPolicy {
        BooguFloatLoadPolicy::AdaptToF32
    }

    /// Denoiser float policy for the selected storage profile.
    pub const fn denoiser_float_load_policy(&self) -> BooguFloatLoadPolicy {
        match self.storage_profile {
            BooguStorageProfile::F16 | BooguStorageProfile::F16QwenVisionF32 => {
                BooguFloatLoadPolicy::Preserve
            }
            BooguStorageProfile::Q8sBlock32F32
            | BooguStorageProfile::Q8sBlock32F32QwenVisionF32 => BooguFloatLoadPolicy::AdaptToF32,
            BooguStorageProfile::Q4sBlockUpTo128F32 => {
                BooguFloatLoadPolicy::PackedQ4sWeightsF32Auxiliaries
            }
        }
    }

    /// Qwen float policy for the selected storage profile.
    pub const fn qwen_float_load_policy(&self) -> BooguFloatLoadPolicy {
        match self.storage_profile {
            BooguStorageProfile::Q4sBlockUpTo128F32 => {
                BooguFloatLoadPolicy::PackedQ4sWeightsF32Auxiliaries
            }
            _ => BooguFloatLoadPolicy::Preserve,
        }
    }

    /// Qwen quantized policy for the selected storage profile.
    pub const fn qwen_quantized_load_policy(&self) -> BooguQuantizedLoadPolicy {
        match self.storage_profile {
            BooguStorageProfile::F16 | BooguStorageProfile::F16QwenVisionF32 => {
                BooguQuantizedLoadPolicy::Preserve
            }
            BooguStorageProfile::Q8sBlock32F32
            | BooguStorageProfile::Q8sBlock32F32QwenVisionF32 => {
                BooguQuantizedLoadPolicy::DequantizeF16
            }
            BooguStorageProfile::Q4sBlockUpTo128F32 => BooguQuantizedLoadPolicy::Preserve,
        }
    }

    /// Boogu denoiser matrices remain quantized for measured direct kernels.
    pub const fn denoiser_quantized_load_policy(&self) -> BooguQuantizedLoadPolicy {
        BooguQuantizedLoadPolicy::Preserve
    }

    /// Concrete platform adapters currently implement the persistent-cache policy only.
    pub fn validate_concrete_cache_policy(&self) -> Result<(), &'static str> {
        match self.cache {
            ArtifactCachePolicy::UseCached => Ok(()),
            ArtifactCachePolicy::Refresh | ArtifactCachePolicy::Bypass => {
                Err("concrete Boogu factories implement only ArtifactCachePolicy::UseCached")
            }
        }
    }
}

/// Preferred immutable storage profile for ordinary native and browser execution.
pub const fn default_storage_profile(variant: BooguVariant) -> BooguStorageProfile {
    match variant {
        BooguVariant::Image01Turbo => BooguStorageProfile::Q4sBlockUpTo128F32,
        BooguVariant::Image01EditTurbo | BooguVariant::Image01EditTurbo1k5 => {
            BooguStorageProfile::F16QwenVisionF32
        }
    }
}

/// Stable model id for one Boogu release.
pub fn model_id(variant: BooguVariant) -> ModelId {
    boogu_model_descriptor(variant).id
}

/// Resolve a public model id to one Boogu release.
pub fn variant_for_model(model: &ModelId) -> Option<BooguVariant> {
    [
        BooguVariant::Image01Turbo,
        BooguVariant::Image01EditTurbo,
        BooguVariant::Image01EditTurbo1k5,
    ]
    .into_iter()
    .find(|variant| boogu_model_descriptor(*variant).id == *model)
}

/// Stable directory stem for one logical Boogu release.
pub const fn bundle_slug(variant: BooguVariant) -> &'static str {
    match variant {
        BooguVariant::Image01Turbo => "boogu-image-0.1-turbo",
        BooguVariant::Image01EditTurbo => "boogu-image-0.1-edit-turbo",
        BooguVariant::Image01EditTurbo1k5 => "boogu-image-0.1-edit-turbo-1k5",
    }
}

/// Stable storage-profile identity used by manifests and runtime provenance.
pub const fn profile_slug(profile: BooguStorageProfile) -> &'static str {
    match profile {
        BooguStorageProfile::F16 => "f16",
        BooguStorageProfile::F16QwenVisionF32 => "f16-qwen-vision-f32",
        BooguStorageProfile::Q8sBlock32F32 => "q8s-block32-f32",
        BooguStorageProfile::Q8sBlock32F32QwenVisionF32 => "q8s-block32-f32-qwen-vision-f32",
        BooguStorageProfile::Q4sBlockUpTo128F32 => "q4s-block-up-to128-f32",
    }
}

/// Dependency-free source bundle id emitted by the converter.
pub fn source_bundle_id(variant: BooguVariant, profile: BooguStorageProfile) -> String {
    source_artifact_bundle_id(variant, profile)
}

/// Preferred immutable bundle id for one released variant/profile tuple.
pub fn bundle_id(variant: BooguVariant, profile: BooguStorageProfile) -> String {
    preferred_artifact_bundle_id(variant, profile)
}

/// Model-neutral artifact profile id for one Boogu storage profile.
pub fn artifact_profile_id(profile: BooguStorageProfile) -> ArtifactProfileId {
    ArtifactProfileId::new(profile_slug(profile))
        .expect("canonical Boogu artifact profile ids are valid")
}

/// Model-neutral numeric format for one Boogu storage profile.
pub fn numeric_format(profile: BooguStorageProfile) -> NumericFormat {
    match profile {
        BooguStorageProfile::F16 => NumericFormat::F16,
        BooguStorageProfile::F16QwenVisionF32 => NumericFormat::Other("f16-qwen-vision-f32".into()),
        BooguStorageProfile::Q8sBlock32F32 => NumericFormat::Other("q8s-block32-f32".into()),
        BooguStorageProfile::Q8sBlock32F32QwenVisionF32 => {
            NumericFormat::Other("q8s-block32-f32-qwen-vision-f32".into())
        }
        BooguStorageProfile::Q4sBlockUpTo128F32 => {
            NumericFormat::Other("q4s-block-up-to128-f32".into())
        }
    }
}

/// Descriptor narrowed to the exact artifact format loaded by one runtime.
pub fn descriptor(variant: BooguVariant, profile: BooguStorageProfile) -> ModelDescriptor {
    let selected = numeric_format(profile);
    let mut descriptor = boogu_model_descriptor(variant);
    if descriptor.capabilities.numeric_formats.contains(&selected) {
        descriptor.capabilities.numeric_formats = [selected].into_iter().collect();
    }
    descriptor
}

/// Reject a profile that is not declared by the immutable model release.
pub fn validate_variant_profile(
    variant: BooguVariant,
    profile: BooguStorageProfile,
) -> Result<(), String> {
    let descriptor = boogu_model_descriptor(variant);
    let format = numeric_format(profile);
    if descriptor.capabilities.numeric_formats.contains(&format) {
        Ok(())
    } else {
        Err(format!(
            "artifact profile {} is not validated for this immutable release",
            artifact_profile_id(profile).as_str()
        ))
    }
}

#[cfg(test)]
mod tests {
    use burn_image::{ArtifactSource, RemoteBaseUrl};

    use super::*;

    fn remote() -> ArtifactSource {
        ArtifactSource::Remote {
            base_url: RemoteBaseUrl::new("https://cdn.example/model").unwrap(),
        }
    }

    #[test]
    fn production_defaults_prioritize_published_q4_without_overclaiming_edit_correctness() {
        assert_eq!(
            default_storage_profile(BooguVariant::Image01Turbo),
            BooguStorageProfile::Q4sBlockUpTo128F32
        );
        for variant in [
            BooguVariant::Image01EditTurbo,
            BooguVariant::Image01EditTurbo1k5,
        ] {
            assert_eq!(
                default_storage_profile(variant),
                BooguStorageProfile::F16QwenVisionF32
            );
        }
        assert_eq!(
            BooguDeploymentSettings::production(BooguVariant::Image01Turbo, remote())
                .storage_profile,
            BooguStorageProfile::Q4sBlockUpTo128F32
        );
        assert_eq!(
            BrowserBooguResidencyPolicy::default(),
            BrowserBooguResidencyPolicy::ResidentPackedQ4s
        );
        assert_eq!(
            browser_bounded_residency_policy(
                BooguVariant::Image01Turbo,
                BooguStorageProfile::Q4sBlockUpTo128F32,
            ),
            BrowserBooguResidencyPolicy::ResidentPackedQ4s
        );
        assert_eq!(
            default_browser_residency(BooguVariant::Image01Turbo),
            BrowserBooguResidencyPolicy::ResidentPackedQ4s
        );
    }

    #[test]
    fn native_policy_selection_and_q4_allocation_are_model_owned_correctness() {
        assert_eq!(
            qualified_native_execution_policy(
                BooguVariant::Image01EditTurbo,
                NativeBooguResidencyPolicy::HighVram,
                BooguStorageProfile::F16QwenVisionF32,
            ),
            Some(BOOGU_1K_NATIVE_POLICY)
        );
        assert_eq!(
            qualified_native_execution_policy(
                BooguVariant::Image01Turbo,
                NativeBooguResidencyPolicy::HighVram,
                BooguStorageProfile::Q4sBlockUpTo128F32,
            ),
            None
        );
        let allocation = native_resident_allocation_policy(BooguStorageProfile::Q4sBlockUpTo128F32);
        assert_eq!(
            allocation.vae_decoder,
            VaeDecoderMemoryPolicy::ExactStripedTailWithStageCleanup
        );
        assert!(allocation.phase_boundary_cleanup);
    }

    #[test]
    fn profile_identity_and_descriptor_share_one_authority_correctness() {
        for profile in [
            BooguStorageProfile::F16,
            BooguStorageProfile::F16QwenVisionF32,
            BooguStorageProfile::Q8sBlock32F32,
            BooguStorageProfile::Q8sBlock32F32QwenVisionF32,
            BooguStorageProfile::Q4sBlockUpTo128F32,
        ] {
            assert_eq!(artifact_profile_id(profile).as_str(), profile_slug(profile));
        }
        let descriptor = descriptor(
            BooguVariant::Image01Turbo,
            BooguStorageProfile::Q4sBlockUpTo128F32,
        );
        assert_eq!(
            descriptor.capabilities.numeric_formats,
            [numeric_format(BooguStorageProfile::Q4sBlockUpTo128F32)]
                .into_iter()
                .collect()
        );
    }
}
