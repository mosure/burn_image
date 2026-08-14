//! Real-artifact, full-chain Boogu parity on Burn WGPU.

use std::{
    collections::BTreeMap,
    error::Error,
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    process::Command,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use burn::{
    prelude::Backend,
    tensor::{DType, Tensor, TensorData},
};
use burn_boogu::{
    BooguConfig, BooguDenoiser, BooguDenoiserInput, BooguExecution, BooguRuntimeDTypes, BooguTask,
    BooguVariant, DenoiserRmsNormPolicy, DmdDenoiser, DmdSchedule,
    NativeDenoiserQkPreparationPolicy, NativeHighVramPolicy, NativePaddedBlackboxDenoiser,
    StreamingBooguPipeline,
    artifacts::{
        BooguArtifactInventory, BooguFloatLoadPolicy, BooguLoadReport, BooguQuantizedLoadPolicy,
        BooguReleaseIdentity, BooguStorageProfile, VerifiedArtifactDirectory,
        VerifiedBurnpackQwenStageSource, VerifiedDirectoryVaeStageSource,
        load_resident_denoiser_from_directory_with_policies,
    },
    decoder_output_to_host, dmd_prediction, dmd_renoise,
    reference::verify_reference_fixture_file,
    trim_instruction_features,
};
use burn_flux_vae::{AutoencoderKlConfig, DecoderGroupNormPolicy};
use burn_image::{ColorSpace, HostImage, PixelFormat};
use burn_qwen3_vl::{
    BatchEncoding, Grid, Qwen3VlConfig, Qwen3VlModelInput, Qwen3VlStageSource, Qwen3VlVisualInput,
    RetainingQwen3VlStageSource, RetainingSynchronizationPolicy, StreamingForwardError,
    StreamingQwen3Vl,
};
use clap::{Parser, ValueEnum};
use half::{bf16, f16};
use safetensors::{Dtype, tensor::Metadata};
use serde::{Deserialize, Serialize};

type B = burn_wgpu::Wgpu<f32, i32, u32>;
const DEFAULT_QUERY_CHUNK_SIZE: usize = 128;
// Decimal GB is intentional and matches the browser low-VRAM qualification contract.
const NATIVE_LOW_VRAM_STRICT_DEVICE_CEILING_BYTES: u64 = 32_000_000_000;
const NVIDIA_SMI_SAMPLE_INTERVAL: Duration = Duration::from_millis(250);

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ProfileChoice {
    F16,
    F16QwenVisionF32,
    Q8sBlock32F32,
    Q8sBlock32F32QwenVisionF32,
}

impl ProfileChoice {
    const fn is_q8(self) -> bool {
        matches!(self, Self::Q8sBlock32F32 | Self::Q8sBlock32F32QwenVisionF32)
    }

    const fn has_f32_qwen_vision(self) -> bool {
        matches!(
            self,
            Self::F16QwenVisionF32 | Self::Q8sBlock32F32QwenVisionF32
        )
    }

    const fn slug(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::F16QwenVisionF32 => "f16-qwen-vision-f32",
            Self::Q8sBlock32F32 => "q8s-block32-f32",
            Self::Q8sBlock32F32QwenVisionF32 => "q8s-block32-f32-qwen-vision-f32",
        }
    }
}

impl From<ProfileChoice> for BooguStorageProfile {
    fn from(value: ProfileChoice) -> Self {
        match value {
            ProfileChoice::F16 => Self::F16,
            ProfileChoice::F16QwenVisionF32 => Self::F16QwenVisionF32,
            ProfileChoice::Q8sBlock32F32 => Self::Q8sBlock32F32,
            ProfileChoice::Q8sBlock32F32QwenVisionF32 => Self::Q8sBlock32F32QwenVisionF32,
        }
    }
}

/// Named native runtime contracts that may make a qualification claim.
///
/// Omitting this selector preserves the existing component-level diagnostic controls. Selecting a
/// named policy makes the CLI fail closed unless every component control matches that policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
enum NativeRuntimePolicyChoice {
    /// Stream verified Qwen stages, load one unretained VAE half per phase, and retain the exact
    /// qualified mixed-F16 denoiser across all four DMD steps.
    LowVram,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum VaeFloatPolicyChoice {
    ForceF32,
    PreserveF16,
}

impl VaeFloatPolicyChoice {
    const fn load_policy(self) -> BooguFloatLoadPolicy {
        match self {
            Self::ForceF32 => BooguFloatLoadPolicy::AdaptToF32,
            Self::PreserveF16 => BooguFloatLoadPolicy::Preserve,
        }
    }

    const fn id(self) -> &'static str {
        match self {
            Self::ForceF32 => "force-f32",
            Self::PreserveF16 => "preserve-f16",
        }
    }
}

const fn edit_reference_oracles(policy: VaeFloatPolicyChoice) -> (&'static str, &'static str) {
    match policy {
        VaeFloatPolicyChoice::ForceF32 => (
            "vae.reference_f32_scaled_latent",
            "vae.reference_scaled_latent",
        ),
        VaeFloatPolicyChoice::PreserveF16 => (
            "vae.reference_scaled_latent",
            "vae.reference_f32_scaled_latent",
        ),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
enum VaeGroupNormPolicyChoice {
    StrictF32,
    F16StorageF32Accum,
}

impl VaeGroupNormPolicyChoice {
    const fn execution_policy(self) -> DecoderGroupNormPolicy {
        match self {
            Self::StrictF32 => DecoderGroupNormPolicy::StrictF32,
            Self::F16StorageF32Accum => DecoderGroupNormPolicy::F16StorageF32Accum,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
enum DenoiserAttentionPolicy {
    Portable,
    PaddedBlackbox,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
enum DenoiserRmsNormPolicyChoice {
    StrictF32,
    F16StorageF32Accum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
enum DenoiserQkPreparationPolicyChoice {
    Composed,
    FusedStrictQkNormRope,
    FusedRopeGqaPadding,
    BalancedStrictQkNormRope,
}

impl DenoiserQkPreparationPolicyChoice {
    const fn fused_strict_qk_norm_rope(self) -> bool {
        matches!(self, Self::FusedStrictQkNormRope)
    }

    const fn fused_rope_gqa_padding(self) -> bool {
        matches!(self, Self::FusedRopeGqaPadding)
    }

    const fn balanced_strict_qk_norm_rope(self) -> bool {
        matches!(self, Self::BalancedStrictQkNormRope)
    }

    const fn matches_native_policy(self, policy: NativeDenoiserQkPreparationPolicy) -> bool {
        matches!(
            (self, policy),
            (Self::Composed, NativeDenoiserQkPreparationPolicy::Composed)
                | (
                    Self::BalancedStrictQkNormRope,
                    NativeDenoiserQkPreparationPolicy::BalancedStrictQkNormRope
                )
        )
    }
}

impl DenoiserRmsNormPolicyChoice {
    const fn execution_policy(self) -> DenoiserRmsNormPolicy {
        match self {
            Self::StrictF32 => DenoiserRmsNormPolicy::StrictF32,
            Self::F16StorageF32Accum => DenoiserRmsNormPolicy::F16StorageF32Accum,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
enum QwenResidency {
    Retained,
    Streamed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
enum QwenSynchronizationPolicyChoice {
    PerStage,
    Deferred,
}

impl QwenSynchronizationPolicyChoice {
    const fn execution_policy(self) -> RetainingSynchronizationPolicy {
        match self {
            Self::PerStage => RetainingSynchronizationPolicy::PerStage,
            Self::Deferred => RetainingSynchronizationPolicy::Deferred,
        }
    }
}

#[derive(Debug, Parser)]
#[command(about = "Run schema-2 full-chain parity through a sealed Boogu bundle on WGPU")]
struct Args {
    /// Directory containing the sealed manifest and converted Burnpack objects.
    #[arg(long)]
    artifacts: PathBuf,
    /// Directory containing schema-2 metadata.json and tensors.safetensors.
    #[arg(long)]
    fixture: PathBuf,
    /// Exact converted storage profile; it must match the sealed manifest.
    #[arg(long, value_enum, default_value = "f16-qwen-vision-f32")]
    profile: ProfileChoice,
    /// Select a named native qualification contract. `low-vram` requires the exact production
    /// numerical policy, streamed/per-stage Qwen, unretained phase-loaded VAE, a resident
    /// denoiser, `--require`, and in-process PID-scoped peak framebuffer telemetry below 32 GB.
    #[arg(long, value_enum)]
    native_runtime_policy: Option<NativeRuntimePolicyChoice>,
    /// Retain verified Qwen stages after first load or drop each stage immediately.
    #[arg(long, value_enum, default_value = "retained")]
    qwen_residency: QwenResidency,
    /// Synchronize after every retained semantic stage or defer to the explicit terminal Qwen
    /// barrier. Deferred synchronization is unavailable with streamed residency.
    #[arg(long, value_enum, default_value = "per-stage")]
    qwen_synchronization_policy: QwenSynchronizationPolicyChoice,
    /// VAE execution policy. `force-f32` matches Diffusers `force_upcast`; `preserve-f16` keeps
    /// both staged VAE weights and full-chain VAE boundary activations in F16.
    #[arg(long, value_enum, default_value = "force-f32")]
    vae_float_policy: VaeFloatPolicyChoice,
    /// Decoder-only GroupNorm policy. The mixed-storage policy retains F16 activation storage
    /// while Burn/CubeCL accumulates reductions in F32; Edit encoding remains strict F32.
    #[arg(long, value_enum, default_value = "strict-f32")]
    vae_group_norm_policy: VaeGroupNormPolicyChoice,
    /// Bounded query tile used by staged VAE attention.
    #[arg(long, default_value_t = 512)]
    vae_attention_query_chunk_size: usize,
    /// Bounded query tile used by streamed Qwen vision and text attention. Defaults to 128.
    #[arg(long)]
    qwen_query_chunk_size: Option<usize>,
    /// Bounded query tile used by Boogu denoiser attention. Defaults to 128.
    #[arg(long)]
    denoiser_query_chunk_size: Option<usize>,
    /// Deprecated compatibility control that sets both Qwen and denoiser query tiles. It cannot be
    /// combined with either component-specific control.
    #[arg(long)]
    query_chunk_size: Option<usize>,
    /// Native WGPU denoiser attention implementation. The padded blackbox policy requires an F16
    /// denoiser and fails closed if the accelerated Cubek kernel cannot run.
    #[arg(long, value_enum, default_value = "portable")]
    denoiser_attention_policy: DenoiserAttentionPolicy,
    /// Denoiser RMSNorm policy. Mixed F16 storage is a diagnostic padded-blackbox experiment and
    /// must pass this pinned full-chain gate before it can become a released runtime policy.
    #[arg(long, value_enum, default_value = "strict-f32")]
    denoiser_rms_norm_policy: DenoiserRmsNormPolicyChoice,
    /// Q/K preparation policy. Balanced strict Q/K normalization is qualified only for the exact
    /// native 1K release policy; the other non-composed choices remain diagnostic.
    #[arg(long, value_enum, default_value = "composed")]
    denoiser_qk_preparation_policy: DenoiserQkPreparationPolicyChoice,
    /// Apply the shared dual-stream output projection separately to each stream.
    #[arg(long, default_value_t = false)]
    denoiser_split_double_stream_shared_projection: bool,
    /// Plane count for padded-blackbox attention. Two planes may use one or two K/V tiles; four
    /// planes use one tile.
    #[arg(long, default_value_t = 4)]
    blackbox_num_planes: u8,
    /// Number of 16-row key/value tiles per padded-blackbox online-softmax partition.
    #[arg(long, default_value_t = 1)]
    blackbox_seq_kv_tiles: u8,
    /// Number of 16-row query tiles retained per plane. Only 1 is supported; q2 failed the native
    /// WGPU nonzero parity gate.
    #[arg(long, default_value_t = 1)]
    blackbox_seq_q_tiles: u8,
    /// Fail when the selected release/profile full-chain gates are exceeded.
    #[arg(long, default_value_t = false)]
    require: bool,
    /// Permit a diagnostic 1.5K policy instead of the exact released native configuration.
    #[arg(long, default_value_t = false)]
    allow_unvalidated_1k5_policy: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct FixtureMetadata {
    schema_version: u32,
    variant: String,
    model_revision: String,
    upstream_source_revision: String,
    width: usize,
    height: usize,
    dtype: String,
    prompt: String,
    seed: u64,
}

impl FixtureMetadata {
    fn variant(&self) -> Result<BooguVariant, Box<dyn Error>> {
        match self.variant.as_str() {
            "turbo" => Ok(BooguVariant::Image01Turbo),
            "edit-turbo" => Ok(BooguVariant::Image01EditTurbo),
            "edit-turbo-1k5" => Ok(BooguVariant::Image01EditTurbo1k5),
            value => Err(format!("unsupported fixture variant {value:?}").into()),
        }
    }

    fn task(&self) -> Result<BooguTask, Box<dyn Error>> {
        Ok(match self.variant()? {
            BooguVariant::Image01Turbo => BooguTask::Generate,
            BooguVariant::Image01EditTurbo | BooguVariant::Image01EditTurbo1k5 => BooguTask::Edit,
        })
    }
}

#[derive(Debug, Clone, Serialize)]
struct TensorMetric {
    name: String,
    oracle: String,
    shape: Vec<usize>,
    actual_dtype: String,
    max_abs: f32,
    mean_abs: f32,
    rmse: f32,
    relative_rmse: f32,
    cosine_similarity: f32,
    readback_milliseconds: f64,
}

#[derive(Debug, Clone, Serialize)]
struct ImageMetric {
    max_abs_u8: u8,
    mean_abs_u8: f32,
    rmse_u8: f32,
    psnr_db: f32,
    mean_block_ssim_8x8: f32,
    exact_fraction: f32,
}

#[derive(Debug, Serialize)]
struct DmdStepMetric {
    index: usize,
    schedule_sigma: f32,
    fixture_sigma: f32,
    fixture_dtype_oracle_sigma: f32,
    fixture_dtype_oracle_abs_difference: f32,
    /// Dtype drift between the production schedule and the directly constructed fixture schedule.
    sigma_abs_difference: f32,
    execution_milliseconds: f64,
    input: TensorMetric,
    velocity: TensorMetric,
    prediction: TensorMetric,
    #[serde(skip_serializing_if = "Option::is_none")]
    renoised: Option<TensorMetric>,
}

#[derive(Debug, Serialize)]
struct ConditioningMetrics {
    qwen_final_hidden_state: TensorMetric,
    #[serde(skip_serializing_if = "Option::is_none")]
    edit_reference_latent: Option<TensorMetric>,
    #[serde(skip_serializing_if = "Option::is_none")]
    edit_reference_bf16_drift: Option<TensorMetric>,
}

#[derive(Debug, Serialize)]
struct TrajectoryMetrics {
    initial_latent: TensorMetric,
    steps: Vec<DmdStepMetric>,
    final_latent: TensorMetric,
}

#[derive(Debug, Serialize)]
struct FullChainOutputMetrics {
    metric_scope: &'static str,
    decoded_tensor: TensorMetric,
    final_rgb: ImageMetric,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct TensorGate {
    maximum_relative_rmse: f32,
    minimum_cosine_similarity: f32,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct ReferenceGate {
    maximum_abs: f32,
    maximum_rmse: f32,
    minimum_cosine_similarity: f32,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct RgbGate {
    minimum_psnr_db: f32,
    minimum_mean_block_ssim_8x8: f32,
}

#[derive(Debug, Clone, Serialize)]
struct GateSet {
    evidence: String,
    vae_float_policy: &'static str,
    qwen_final: TensorGate,
    edit_reference: ReferenceGate,
    dmd_boundaries: TensorGate,
    dmd_final: TensorGate,
    propagated_decode: TensorGate,
    final_rgb: RgbGate,
}

#[derive(Debug, Serialize)]
struct GateEvaluation {
    supported: bool,
    passed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    unsupported_reason: Option<String>,
    failures: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thresholds: Option<GateSet>,
}

#[derive(Debug, Serialize)]
struct PolicyReport {
    native_autotune: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    native_runtime_policy: Option<NativeRuntimePolicyChoice>,
    native_weight_traffic_contract: &'static str,
    vae_residency: &'static str,
    denoiser_residency: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict_peak_device_ceiling_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    required_peak_telemetry: Option<&'static str>,
    qwen_residency: QwenResidency,
    qwen_synchronization_policy: QwenSynchronizationPolicyChoice,
    qwen_quantized_load: &'static str,
    qwen_query_chunk_size: usize,
    vae_float_load: &'static str,
    vae_group_norm: VaeGroupNormPolicyChoice,
    vae_attention_query_chunk_size: usize,
    denoiser_float_load: &'static str,
    denoiser_quantized_load: &'static str,
    denoiser_attention_policy: DenoiserAttentionPolicy,
    denoiser_rms_norm_policy: DenoiserRmsNormPolicyChoice,
    denoiser_qk_preparation_policy: DenoiserQkPreparationPolicyChoice,
    denoiser_split_double_stream_shared_projection: bool,
    denoiser_query_chunk_size: usize,
    blackbox_num_planes: u8,
    blackbox_seq_kv_tiles: u8,
    blackbox_seq_q_tiles: u8,
    native_release_policy_validated: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    edit_1k5_release_policy_validated: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QueryChunkSizes {
    qwen: usize,
    denoiser: usize,
}

fn resolve_query_chunk_sizes(args: &Args) -> Result<QueryChunkSizes, &'static str> {
    if args.query_chunk_size.is_some()
        && (args.qwen_query_chunk_size.is_some() || args.denoiser_query_chunk_size.is_some())
    {
        return Err(
            "--query-chunk-size cannot be combined with --qwen-query-chunk-size or --denoiser-query-chunk-size",
        );
    }
    if args.query_chunk_size == Some(0) {
        return Err("--query-chunk-size must be greater than zero");
    }
    let sizes = if let Some(query_chunk_size) = args.query_chunk_size {
        QueryChunkSizes {
            qwen: query_chunk_size,
            denoiser: query_chunk_size,
        }
    } else {
        QueryChunkSizes {
            qwen: args
                .qwen_query_chunk_size
                .unwrap_or(DEFAULT_QUERY_CHUNK_SIZE),
            denoiser: args
                .denoiser_query_chunk_size
                .unwrap_or(DEFAULT_QUERY_CHUNK_SIZE),
        }
    };
    if sizes.qwen == 0 {
        return Err("--qwen-query-chunk-size must be greater than zero");
    }
    if sizes.denoiser == 0 {
        return Err("--denoiser-query-chunk-size must be greater than zero");
    }
    Ok(sizes)
}

fn policy_report(
    args: &Args,
    variant: BooguVariant,
    query_chunk_sizes: QueryChunkSizes,
) -> PolicyReport {
    let release_policy_validated = is_exact_native_release_policy(args, variant, query_chunk_sizes)
        && !(variant == BooguVariant::Image01EditTurbo1k5 && args.allow_unvalidated_1k5_policy);
    PolicyReport {
        native_autotune: "full",
        native_runtime_policy: args.native_runtime_policy,
        native_weight_traffic_contract: match args.native_runtime_policy {
            Some(NativeRuntimePolicyChoice::LowVram) => {
                "phase-resident/qwen+vae-per-run/denoiser-resident-zero-dmd-weight-reloads"
            }
            None => "explicit-component-controls/no-named-residency-qualification",
        },
        // This binary deliberately uses the ordinary non-retaining verified VAE source. Each half
        // is synchronized and dropped at its phase boundary by StreamingBooguPipeline.
        vae_residency: "unretained-phase-loaded-encoder-then-decoder",
        // The denoiser is reconstructed once before Qwen starts and remains alive for all four
        // DMD steps. It is never a per-step stage source in this full-chain binary.
        denoiser_residency: "resident-across-all-four-dmd-steps",
        strict_peak_device_ceiling_bytes: args
            .native_runtime_policy
            .map(|_| NATIVE_LOW_VRAM_STRICT_DEVICE_CEILING_BYTES),
        required_peak_telemetry: args.native_runtime_policy.map(|_| {
            "in-process-pid-scoped-nvidia-smi-configured-250ms-delay-total-framebuffer/strictly-less-than-ceiling"
        }),
        qwen_residency: args.qwen_residency,
        qwen_synchronization_policy: args.qwen_synchronization_policy,
        qwen_quantized_load: if args.profile.is_q8() {
            "dequantize-f16-before-col-mapper"
        } else {
            "preserve"
        },
        qwen_query_chunk_size: query_chunk_sizes.qwen,
        vae_float_load: args.vae_float_policy.id(),
        vae_group_norm: args.vae_group_norm_policy,
        vae_attention_query_chunk_size: args.vae_attention_query_chunk_size,
        denoiser_float_load: if args.profile.is_q8() {
            "adapt-to-f32"
        } else {
            "preserve"
        },
        denoiser_quantized_load: if args.profile.is_q8() {
            "preserve-row-layout-q8"
        } else {
            "not-applicable-no-quantized-tensors"
        },
        denoiser_attention_policy: args.denoiser_attention_policy,
        denoiser_rms_norm_policy: args.denoiser_rms_norm_policy,
        denoiser_qk_preparation_policy: args.denoiser_qk_preparation_policy,
        denoiser_split_double_stream_shared_projection: args
            .denoiser_split_double_stream_shared_projection,
        denoiser_query_chunk_size: query_chunk_sizes.denoiser,
        blackbox_num_planes: args.blackbox_num_planes,
        blackbox_seq_kv_tiles: args.blackbox_seq_kv_tiles,
        blackbox_seq_q_tiles: args.blackbox_seq_q_tiles,
        native_release_policy_validated: release_policy_validated,
        edit_1k5_release_policy_validated: (variant == BooguVariant::Image01EditTurbo1k5)
            .then_some(release_policy_validated),
    }
}

const fn native_release_policy(variant: BooguVariant) -> NativeHighVramPolicy {
    match variant {
        BooguVariant::Image01Turbo | BooguVariant::Image01EditTurbo => {
            burn_boogu::BOOGU_1K_NATIVE_POLICY
        }
        BooguVariant::Image01EditTurbo1k5 => burn_boogu::EDIT_TURBO_1K5_NATIVE_POLICY,
    }
}

fn is_exact_native_release_policy(
    args: &Args,
    variant: BooguVariant,
    query_chunk_sizes: QueryChunkSizes,
) -> bool {
    let policy = native_release_policy(variant);
    let qwen_residency_matches = match args.native_runtime_policy {
        Some(NativeRuntimePolicyChoice::LowVram) => {
            args.qwen_residency == QwenResidency::Streamed
                && args.qwen_synchronization_policy == QwenSynchronizationPolicyChoice::PerStage
        }
        None => {
            args.qwen_residency == QwenResidency::Retained
                && args.qwen_synchronization_policy == QwenSynchronizationPolicyChoice::Deferred
        }
    };
    args.profile == ProfileChoice::F16QwenVisionF32
        && qwen_residency_matches
        && args.vae_float_policy == VaeFloatPolicyChoice::PreserveF16
        && args.vae_group_norm_policy == VaeGroupNormPolicyChoice::F16StorageF32Accum
        && args.vae_attention_query_chunk_size == policy.vae_attention_query_chunk_size
        && query_chunk_sizes.qwen == policy.qwen_query_chunk_size
        && query_chunk_sizes.denoiser == policy.denoiser_query_chunk_size
        && args.denoiser_attention_policy == DenoiserAttentionPolicy::PaddedBlackbox
        && args.denoiser_rms_norm_policy == DenoiserRmsNormPolicyChoice::StrictF32
        && args
            .denoiser_qk_preparation_policy
            .matches_native_policy(policy.denoiser_qk_preparation)
        && !args.denoiser_split_double_stream_shared_projection
        && args.blackbox_num_planes == policy.blackbox_num_planes
        && args.blackbox_seq_kv_tiles == policy.blackbox_seq_kv_tiles
        && args.blackbox_seq_q_tiles == policy.blackbox_seq_q_tiles
}

fn validate_native_runtime_policy(
    args: &Args,
    variant: BooguVariant,
    query_chunk_sizes: QueryChunkSizes,
) -> Result<(), &'static str> {
    let Some(NativeRuntimePolicyChoice::LowVram) = args.native_runtime_policy else {
        return Ok(());
    };
    if !args.require {
        return Err(
            "--native-runtime-policy low-vram requires --require; numerical gates may not be disabled for qualification",
        );
    }
    if args.allow_unvalidated_1k5_policy {
        return Err("--native-runtime-policy low-vram rejects --allow-unvalidated-1k5-policy");
    }
    if !is_exact_native_release_policy(args, variant, query_chunk_sizes) {
        return Err(
            "--native-runtime-policy low-vram requires profile=f16-qwen-vision-f32, streamed per-stage Qwen, preserve-F16 VAE q4096 with f16-storage-f32-accum GroupNorm, and the exact release padded-blackbox denoiser policy; its in-process PID-scoped nvidia-smi gate must additionally prove sampled total framebuffer stays strictly below 32,000,000,000 bytes",
        );
    }
    Ok(())
}

fn is_exact_1k5_release_policy(args: &Args, query_chunk_sizes: QueryChunkSizes) -> bool {
    is_exact_native_release_policy(args, BooguVariant::Image01EditTurbo1k5, query_chunk_sizes)
}

fn validate_1k5_release_policy(
    args: &Args,
    variant: BooguVariant,
    query_chunk_sizes: QueryChunkSizes,
) -> Result<(), &'static str> {
    if variant == BooguVariant::Image01EditTurbo1k5
        && !is_exact_1k5_release_policy(args, query_chunk_sizes)
        && !args.allow_unvalidated_1k5_policy
    {
        return Err(
            "Edit-Turbo 1.5K full-chain parity requires the exact native release policy: \
             retained deferred-sync Qwen q128, or explicit low-vram streamed per-stage Qwen q128; preserve-F16 VAE q4096 with f16-storage-f32-accum GroupNorm, \
             and padded-blackbox p4/kv1/q1 denoiser q16384 with strict-f32 RMSNorm and composed Q/K preparation; pass \
             --allow-unvalidated-1k5-policy only for explicitly diagnostic runs",
        );
    }
    Ok(())
}

fn validate_denoiser_attention_policy(
    profile: ProfileChoice,
    policy: DenoiserAttentionPolicy,
) -> Result<(), &'static str> {
    if policy == DenoiserAttentionPolicy::PaddedBlackbox && profile.is_q8() {
        return Err(
            "--denoiser-attention-policy padded-blackbox requires F16 denoiser execution; \
             Q8 profiles adapt the denoiser to F32",
        );
    }
    Ok(())
}

fn validate_denoiser_rms_norm_policy(
    attention_policy: DenoiserAttentionPolicy,
    rms_norm_policy: DenoiserRmsNormPolicyChoice,
) -> Result<(), &'static str> {
    if rms_norm_policy == DenoiserRmsNormPolicyChoice::F16StorageF32Accum
        && attention_policy != DenoiserAttentionPolicy::PaddedBlackbox
    {
        return Err("--denoiser-rms-norm-policy f16-storage-f32-accum requires \
             --denoiser-attention-policy padded-blackbox");
    }
    Ok(())
}

fn validate_denoiser_qk_preparation_policy(
    attention_policy: DenoiserAttentionPolicy,
    rms_norm_policy: DenoiserRmsNormPolicyChoice,
    qk_preparation_policy: DenoiserQkPreparationPolicyChoice,
    num_planes: u8,
    seq_kv_tiles: u8,
    seq_q_tiles: u8,
) -> Result<(), &'static str> {
    if qk_preparation_policy != DenoiserQkPreparationPolicyChoice::Composed
        && (attention_policy != DenoiserAttentionPolicy::PaddedBlackbox
            || rms_norm_policy != DenoiserRmsNormPolicyChoice::StrictF32
            || (num_planes, seq_kv_tiles, seq_q_tiles) != (4, 1, 1))
    {
        return Err(
            "non-composed --denoiser-qk-preparation-policy values require padded-blackbox \
             attention, strict-f32 RMSNorm, and p4/kv1/q1",
        );
    }
    Ok(())
}

fn validate_blackbox_partition_controls(
    policy: DenoiserAttentionPolicy,
    num_planes: u8,
    seq_kv_tiles: u8,
    seq_q_tiles: u8,
) -> Result<(), &'static str> {
    if !matches!(num_planes, 2 | 4) {
        return Err("--blackbox-num-planes must be one of 2 or 4");
    }
    if !matches!(seq_kv_tiles, 1 | 2) {
        return Err("--blackbox-seq-kv-tiles must be one of 1 or 2");
    }
    if seq_q_tiles != 1 {
        return Err(
            "--blackbox-seq-q-tiles must be 1; q2 failed the native WGPU nonzero parity gate",
        );
    }
    if num_planes != 2 && seq_kv_tiles == 2 {
        return Err("multi-KV-tile padded-blackbox configurations require two planes");
    }
    if policy == DenoiserAttentionPolicy::Portable
        && (num_planes != 4 || seq_kv_tiles != 1 || seq_q_tiles != 1)
    {
        return Err("blackbox partition controls apply only to padded-blackbox attention");
    }
    Ok(())
}

enum FullParityDenoiser {
    Portable(Box<BooguDenoiser<B>>),
    PaddedBlackbox(Box<NativePaddedBlackboxDenoiser>),
}

#[derive(Clone, Copy)]
struct DenoiserConfiguration {
    attention_policy: DenoiserAttentionPolicy,
    rms_norm_policy: DenoiserRmsNormPolicyChoice,
    qk_preparation_policy: DenoiserQkPreparationPolicyChoice,
    split_double_stream_shared_projection: bool,
    query_chunk_size: usize,
    blackbox_num_planes: u8,
    blackbox_seq_kv_tiles: u8,
    blackbox_seq_q_tiles: u8,
}

impl DmdDenoiser<B> for FullParityDenoiser {
    fn execution_dtype(&self) -> Option<DType> {
        match self {
            Self::Portable(denoiser) => denoiser.execution_dtype(),
            Self::PaddedBlackbox(denoiser) => denoiser.execution_dtype(),
        }
    }

    fn predict(
        &mut self,
        input: BooguDenoiserInput<B>,
    ) -> Result<Tensor<B, 4>, burn_boogu::BooguError> {
        match self {
            Self::Portable(denoiser) => denoiser.predict(input),
            Self::PaddedBlackbox(denoiser) => denoiser.predict(input),
        }
    }
}

fn configure_denoiser(
    denoiser: BooguDenoiser<B>,
    configuration: DenoiserConfiguration,
) -> FullParityDenoiser {
    match configuration.attention_policy {
        DenoiserAttentionPolicy::Portable => {
            debug_assert_eq!(
                configuration.rms_norm_policy,
                DenoiserRmsNormPolicyChoice::StrictF32
            );
            debug_assert_eq!(
                configuration.qk_preparation_policy,
                DenoiserQkPreparationPolicyChoice::Composed
            );
            let mut denoiser = denoiser;
            denoiser.set_attention_query_chunk_size(configuration.query_chunk_size);
            FullParityDenoiser::Portable(Box::new(denoiser))
        }
        DenoiserAttentionPolicy::PaddedBlackbox => {
            let mut denoiser = NativePaddedBlackboxDenoiser::new(denoiser)
                .with_partition_configuration(
                    configuration.blackbox_num_planes,
                    configuration.blackbox_seq_kv_tiles,
                    configuration.blackbox_seq_q_tiles,
                )
                .with_rms_norm_policy(configuration.rms_norm_policy.execution_policy())
                .with_fused_strict_qk_norm_rope(
                    configuration
                        .qk_preparation_policy
                        .fused_strict_qk_norm_rope(),
                )
                .with_fused_rope_gqa_padding(
                    configuration.qk_preparation_policy.fused_rope_gqa_padding(),
                )
                .with_balanced_strict_qk_norm_rope(
                    configuration
                        .qk_preparation_policy
                        .balanced_strict_qk_norm_rope(),
                )
                .with_split_double_stream_shared_projection(
                    configuration.split_double_stream_shared_projection,
                );
            denoiser.set_attention_query_chunk_size(configuration.query_chunk_size);
            FullParityDenoiser::PaddedBlackbox(Box::new(denoiser))
        }
    }
}

#[derive(Debug, Serialize)]
struct DTypeReport {
    qwen_visual: String,
    qwen_text: &'static str,
    vae: String,
    denoiser: String,
}

#[derive(Debug, Default, Serialize)]
struct TimingReport {
    fixture_and_manifest_verification_milliseconds: f64,
    qwen_source_verification_milliseconds: f64,
    vae_source_verification_milliseconds: f64,
    denoiser_load_milliseconds: f64,
    qwen_execute_milliseconds: f64,
    vae_encode_milliseconds: f64,
    dmd_step_milliseconds: Vec<f64>,
    vae_decode_milliseconds: f64,
    total_milliseconds: f64,
}

#[derive(Clone, Debug, Default)]
struct PidFramebufferSamples {
    attempted_samples: u64,
    matched_samples: u64,
    nonzero_samples: u64,
    peak_total_framebuffer_mib: u64,
    sample_error_count: u64,
    sample_errors: Vec<String>,
}

#[derive(Debug, Serialize)]
struct DeviceMemoryQualification {
    provider: &'static str,
    process_id: u32,
    sample_interval_milliseconds: u64,
    attempted_samples: u64,
    matched_samples: u64,
    nonzero_samples: u64,
    peak_total_framebuffer_mib: u64,
    peak_total_framebuffer_bytes: u64,
    strict_ceiling_bytes: u64,
    strictly_below_ceiling: bool,
    sample_error_count: u64,
    sample_errors: Vec<String>,
    passed: bool,
    failures: Vec<String>,
}

struct NvidiaSmiPidMonitor {
    process_id: u32,
    sample_interval: Duration,
    stop: Arc<AtomicBool>,
    samples: Arc<Mutex<PidFramebufferSamples>>,
    worker: Option<JoinHandle<()>>,
}

impl NvidiaSmiPidMonitor {
    fn start(sample_interval: Duration) -> Result<Self, Box<dyn Error>> {
        let inventory = Command::new("nvidia-smi")
            .args(["--query-gpu=uuid", "--format=csv,noheader,nounits"])
            .output()
            .map_err(|error| {
                format!("native low-vram qualification requires nvidia-smi GPU telemetry: {error}")
            })?;
        if !inventory.status.success() {
            return Err(format!(
                "native low-vram qualification could not inventory NVIDIA GPUs: {}",
                String::from_utf8_lossy(&inventory.stderr).trim()
            )
            .into());
        }
        if String::from_utf8_lossy(&inventory.stdout)
            .lines()
            .all(|line| line.trim().is_empty())
        {
            return Err(
                "native low-vram qualification requires at least one nvidia-smi GPU".into(),
            );
        }

        let process_id = std::process::id();
        let stop = Arc::new(AtomicBool::new(false));
        let samples = Arc::new(Mutex::new(PidFramebufferSamples::default()));
        let worker_stop = stop.clone();
        let worker_samples = samples.clone();
        let worker = thread::Builder::new()
            .name("boogu-low-vram-nvidia-smi".into())
            .spawn(move || {
                loop {
                    let sample = sample_pid_total_framebuffer_mib(process_id);
                    if let Ok(mut samples) = worker_samples.lock() {
                        samples.attempted_samples += 1;
                        match sample {
                            Ok(Some(total_mib)) => {
                                samples.matched_samples += 1;
                                samples.nonzero_samples += u64::from(total_mib > 0);
                                samples.peak_total_framebuffer_mib =
                                    samples.peak_total_framebuffer_mib.max(total_mib);
                            }
                            Ok(None) => {}
                            Err(error) => {
                                samples.sample_error_count += 1;
                                if samples.sample_errors.len() < 8 {
                                    samples.sample_errors.push(error);
                                }
                            }
                        }
                    }
                    if worker_stop.load(Ordering::Acquire) {
                        break;
                    }
                    thread::sleep(sample_interval);
                }
            })?;
        Ok(Self {
            process_id,
            sample_interval,
            stop,
            samples,
            worker: Some(worker),
        })
    }

    fn finish(mut self) -> Result<DeviceMemoryQualification, Box<dyn Error>> {
        self.stop_and_join()?;
        let samples = self
            .samples
            .lock()
            .map_err(|_| "nvidia-smi telemetry state was poisoned")?
            .clone();
        Ok(evaluate_device_memory_qualification(
            self.process_id,
            self.sample_interval,
            samples,
        ))
    }

    fn stop_and_join(&mut self) -> Result<(), Box<dyn Error>> {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            worker
                .join()
                .map_err(|_| "nvidia-smi telemetry worker panicked")?;
        }
        Ok(())
    }
}

impl Drop for NvidiaSmiPidMonitor {
    fn drop(&mut self) {
        let _ = self.stop_and_join();
    }
}

fn sample_pid_total_framebuffer_mib(process_id: u32) -> Result<Option<u64>, String> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .map_err(|error| format!("failed to sample nvidia-smi compute processes: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "nvidia-smi process sample failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    parse_pid_total_framebuffer_mib(&String::from_utf8_lossy(&output.stdout), process_id)
}

fn parse_pid_total_framebuffer_mib(output: &str, process_id: u32) -> Result<Option<u64>, String> {
    let mut matched = false;
    let mut total_mib = 0_u64;
    for line in output.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (pid, used_memory) = line
            .split_once(',')
            .ok_or_else(|| format!("unparseable nvidia-smi process row {line:?}"))?;
        let row_pid = pid
            .trim()
            .parse::<u32>()
            .map_err(|_| format!("unparseable nvidia-smi process PID {pid:?}"))?;
        if row_pid != process_id {
            continue;
        }
        matched = true;
        let used_memory = used_memory.trim().trim_end_matches("MiB").trim();
        let used_memory_mib = used_memory.parse::<u64>().map_err(|_| {
            format!("unparseable nvidia-smi framebuffer value {used_memory:?} for PID {process_id}")
        })?;
        total_mib = total_mib
            .checked_add(used_memory_mib)
            .ok_or_else(|| "PID-scoped framebuffer sum overflowed u64".to_owned())?;
    }
    Ok(matched.then_some(total_mib))
}

fn evaluate_device_memory_qualification(
    process_id: u32,
    sample_interval: Duration,
    samples: PidFramebufferSamples,
) -> DeviceMemoryQualification {
    const MIN_MATCHED_SAMPLES: u64 = 4;
    const MIN_NONZERO_SAMPLES: u64 = 4;
    const MIB: u64 = 1024 * 1024;

    let peak_total_framebuffer_bytes = samples.peak_total_framebuffer_mib.saturating_mul(MIB);
    let strictly_below_ceiling =
        peak_total_framebuffer_bytes < NATIVE_LOW_VRAM_STRICT_DEVICE_CEILING_BYTES;
    let mut failures = Vec::new();
    if samples.matched_samples < MIN_MATCHED_SAMPLES {
        failures.push(format!(
            "nvidia-smi matched PID {process_id} in only {} intervals; at least {MIN_MATCHED_SAMPLES} are required",
            samples.matched_samples
        ));
    }
    if samples.nonzero_samples < MIN_NONZERO_SAMPLES {
        failures.push(format!(
            "nvidia-smi observed nonzero PID framebuffer in only {} intervals; at least {MIN_NONZERO_SAMPLES} are required",
            samples.nonzero_samples
        ));
    }
    if samples.sample_error_count != 0 {
        failures.push(format!(
            "nvidia-smi encountered {} sampling errors",
            samples.sample_error_count
        ));
    }
    if !strictly_below_ceiling {
        failures.push(format!(
            "PID-scoped total framebuffer peak was {} MiB ({} bytes), which is not strictly below {} bytes",
            samples.peak_total_framebuffer_mib,
            peak_total_framebuffer_bytes,
            NATIVE_LOW_VRAM_STRICT_DEVICE_CEILING_BYTES
        ));
    }
    DeviceMemoryQualification {
        provider: "nvidia-smi",
        process_id,
        sample_interval_milliseconds: u64::try_from(sample_interval.as_millis())
            .unwrap_or(u64::MAX),
        attempted_samples: samples.attempted_samples,
        matched_samples: samples.matched_samples,
        nonzero_samples: samples.nonzero_samples,
        peak_total_framebuffer_mib: samples.peak_total_framebuffer_mib,
        peak_total_framebuffer_bytes,
        strict_ceiling_bytes: NATIVE_LOW_VRAM_STRICT_DEVICE_CEILING_BYTES,
        strictly_below_ceiling,
        sample_error_count: samples.sample_error_count,
        sample_errors: samples.sample_errors,
        passed: failures.is_empty(),
        failures,
    }
}

#[derive(Debug, Serialize)]
struct FullChainReport {
    report_schema_version: u32,
    metric_scope: &'static str,
    variant: String,
    fixture_schema_version: u32,
    fixture_dtype: String,
    fixture_prompt: String,
    fixture_seed: u64,
    width: usize,
    height: usize,
    model_revision: String,
    upstream_source_revision: String,
    artifact_content_digest: String,
    artifact_profile: String,
    backend: &'static str,
    policies: PolicyReport,
    execution_dtypes: DTypeReport,
    qwen_stage_count: usize,
    qwen_embedding_row_chunks: usize,
    denoiser_loaded_tensors: usize,
    denoiser_loaded_shards: usize,
    denoiser_reference_refiner_modules_retained: usize,
    effective_instruction_length: usize,
    timings: TimingReport,
    conditioning: ConditioningMetrics,
    trajectory: TrajectoryMetrics,
    full_chain_output: FullChainOutputMetrics,
    gates: GateEvaluation,
    #[serde(skip_serializing_if = "Option::is_none")]
    device_memory_qualification: Option<DeviceMemoryQualification>,
}

#[derive(Debug, Clone)]
struct TensorSpec {
    dtype: Dtype,
    shape: Vec<usize>,
}

trait TensorIndex {
    fn tensor_spec(&self, name: &str) -> Option<TensorSpec>;
}

struct FixtureStore {
    path: PathBuf,
    data_start: u64,
    metadata: Metadata,
}

impl FixtureStore {
    fn open(path: impl AsRef<Path>) -> Result<Self, Box<dyn Error>> {
        let path = path.as_ref();
        let mut file = File::open(path)?;
        let mut length = [0_u8; 8];
        file.read_exact(&mut length)?;
        let header_len = usize::try_from(u64::from_le_bytes(length))?;
        if header_len == 0 || header_len > 100 * 1024 * 1024 {
            return Err("invalid SafeTensors fixture header length".into());
        }
        let mut header = vec![0_u8; header_len];
        file.read_exact(&mut header)?;
        let metadata: Metadata = serde_json::from_slice(&header)?;
        let data_start = 8_u64
            .checked_add(u64::try_from(header_len)?)
            .ok_or("SafeTensors fixture data offset overflow")?;
        let expected_size = data_start
            .checked_add(u64::try_from(metadata.data_len())?)
            .ok_or("SafeTensors fixture size overflow")?;
        if file.metadata()?.len() != expected_size {
            return Err("SafeTensors fixture size differs from its validated header".into());
        }
        Ok(Self {
            path: path.to_owned(),
            data_start,
            metadata,
        })
    }

    fn contains(&self, name: &str) -> bool {
        self.metadata.info(name).is_some()
    }

    fn tensor(&self, name: &str) -> Result<FixtureTensor, Box<dyn Error>> {
        let info = self
            .metadata
            .info(name)
            .ok_or_else(|| format!("fixture omits tensor {name}"))?;
        let start = self
            .data_start
            .checked_add(u64::try_from(info.data_offsets.0)?)
            .ok_or("fixture tensor offset overflow")?;
        let len = info
            .data_offsets
            .1
            .checked_sub(info.data_offsets.0)
            .ok_or("fixture tensor offset underflow")?;
        let element_count = info.shape.iter().try_fold(1_usize, |count, &dimension| {
            count
                .checked_mul(dimension)
                .ok_or("fixture tensor shape overflow")
        })?;
        let expected_bits = element_count
            .checked_mul(info.dtype.bitsize())
            .ok_or("fixture tensor byte length overflow")?;
        let expected_len = expected_bits.div_ceil(8);
        if len != expected_len {
            return Err(format!(
                "fixture tensor {name} stores {len} bytes, expected {expected_len} from {:?} {:?}",
                info.dtype, info.shape
            )
            .into());
        }
        let mut bytes = vec![0_u8; len];
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(start))?;
        file.read_exact(&mut bytes)?;
        Ok(FixtureTensor {
            dtype: info.dtype,
            shape: info.shape.clone(),
            bytes,
        })
    }

    fn i64(&self, name: &str) -> Result<(Vec<usize>, Vec<i64>), Box<dyn Error>> {
        let tensor = self.tensor(name)?;
        if tensor.dtype != Dtype::I64 {
            return Err(format!("fixture tensor {name} is not I64").into());
        }
        let values = tensor
            .bytes
            .chunks_exact(8)
            .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("eight-byte chunk")))
            .collect();
        Ok((tensor.shape, values))
    }

    fn f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), Box<dyn Error>> {
        let tensor = self.tensor(name)?;
        let values = decode_float_bytes(name, tensor.dtype, &tensor.bytes)?;
        Ok((tensor.shape, values))
    }

    fn u8(&self, name: &str) -> Result<(Vec<usize>, Vec<u8>), Box<dyn Error>> {
        let tensor = self.tensor(name)?;
        if tensor.dtype != Dtype::U8 {
            return Err(format!("fixture tensor {name} is not U8").into());
        }
        Ok((tensor.shape, tensor.bytes))
    }
}

impl TensorIndex for FixtureStore {
    fn tensor_spec(&self, name: &str) -> Option<TensorSpec> {
        self.metadata.info(name).map(|info| TensorSpec {
            dtype: info.dtype,
            shape: info.shape.clone(),
        })
    }
}

impl TensorIndex for BTreeMap<String, TensorSpec> {
    fn tensor_spec(&self, name: &str) -> Option<TensorSpec> {
        self.get(name).cloned()
    }
}

struct FixtureTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
enum ExpectedDType {
    I64,
    U8,
    Exact(Dtype),
}

#[derive(Debug, Clone)]
enum ExpectedShape {
    Exact(Vec<usize>),
    Rank(usize),
    Scalar,
}

fn main() -> Result<(), Box<dyn Error>> {
    burn_boogu::configure_native_full_autotune();
    let args = Args::parse();
    let query_chunk_sizes = resolve_query_chunk_sizes(&args)?;
    if args.vae_attention_query_chunk_size == 0 {
        return Err("--vae-attention-query-chunk-size must be greater than zero".into());
    }
    validate_denoiser_attention_policy(args.profile, args.denoiser_attention_policy)?;
    validate_denoiser_rms_norm_policy(
        args.denoiser_attention_policy,
        args.denoiser_rms_norm_policy,
    )?;
    if args.denoiser_split_double_stream_shared_projection
        && args.denoiser_attention_policy != DenoiserAttentionPolicy::PaddedBlackbox
    {
        return Err(
            "--denoiser-split-double-stream-shared-projection requires padded-blackbox attention"
                .into(),
        );
    }
    validate_blackbox_partition_controls(
        args.denoiser_attention_policy,
        args.blackbox_num_planes,
        args.blackbox_seq_kv_tiles,
        args.blackbox_seq_q_tiles,
    )?;
    validate_denoiser_qk_preparation_policy(
        args.denoiser_attention_policy,
        args.denoiser_rms_norm_policy,
        args.denoiser_qk_preparation_policy,
        args.blackbox_num_planes,
        args.blackbox_seq_kv_tiles,
        args.blackbox_seq_q_tiles,
    )?;
    if args.vae_group_norm_policy == VaeGroupNormPolicyChoice::F16StorageF32Accum
        && args.vae_float_policy != VaeFloatPolicyChoice::PreserveF16
    {
        return Err(
            "--vae-group-norm-policy f16-storage-f32-accum requires --vae-float-policy preserve-f16"
                .into(),
        );
    }
    let total_started = Instant::now();
    let verification_started = Instant::now();
    let fixture_metadata_bytes = fs::read(args.fixture.join("metadata.json"))?;
    let metadata: FixtureMetadata = serde_json::from_slice(&fixture_metadata_bytes)?;
    if metadata.schema_version != 2 {
        return Err(format!(
            "full-chain parity requires fixture schema 2, found {}",
            metadata.schema_version
        )
        .into());
    }
    let variant = metadata.variant()?;
    validate_native_runtime_policy(&args, variant, query_chunk_sizes)?;
    validate_1k5_release_policy(&args, variant, query_chunk_sizes)?;
    let identity = BooguReleaseIdentity::canonical(variant);
    if metadata.model_revision != identity.model_revision
        || metadata.upstream_source_revision != identity.upstream_source_revision
    {
        return Err("fixture revisions do not match the canonical immutable release".into());
    }
    let fixture_path = args.fixture.join("tensors.safetensors");
    verify_reference_fixture_file(&fixture_metadata_bytes, &fixture_path)?;
    let fixture = FixtureStore::open(fixture_path)?;
    validate_fixture_contract(&fixture, &metadata)?;

    let artifact_directory = VerifiedArtifactDirectory::open(&args.artifacts)?;
    let manifest = artifact_directory.manifest();
    if manifest.model_revision != identity.model_revision {
        return Err(format!(
            "artifact revision {} does not match fixture release {}",
            manifest.model_revision, identity.model_revision
        )
        .into());
    }
    if manifest.profile.as_str() != args.profile.slug() {
        return Err(format!(
            "sealed manifest profile {:?} does not match --profile {}",
            manifest.profile.as_str(),
            args.profile.slug()
        )
        .into());
    }
    let artifact_content_digest = manifest
        .content_digest
        .ok_or("sealed artifact manifest has no content digest")?;
    if variant == BooguVariant::Image01EditTurbo1k5 && !args.allow_unvalidated_1k5_policy {
        burn_boogu::artifacts::validate_edit_turbo_1k5_release_artifact_digest(
            artifact_content_digest,
        )?;
    }
    let artifact_content_digest = artifact_content_digest.to_string();
    let fixture_and_manifest_verification_milliseconds =
        verification_started.elapsed().as_secs_f64() * 1_000.0;

    let qwen_config = Qwen3VlConfig::from_json(
        &artifact_directory.read_text("metadata/source/mllm/config.json")?,
    )?;
    let mut vae_config = AutoencoderKlConfig::from_diffusers_json(
        &artifact_directory.read_text("metadata/source/vae/config.json")?,
    )?;
    vae_config.attention_query_chunk_size = args.vae_attention_query_chunk_size;
    let denoiser_config = BooguConfig::default();
    let inventory = BooguArtifactInventory::new(&qwen_config, &denoiser_config, &vae_config)?;
    let mut memory_monitor = args
        .native_runtime_policy
        .map(|NativeRuntimePolicyChoice::LowVram| {
            NvidiaSmiPidMonitor::start(NVIDIA_SMI_SAMPLE_INTERVAL)
        })
        .transpose()?;
    let device = burn_boogu::require_native_wgpu_device()?;
    let profile: BooguStorageProfile = args.profile.into();
    let vae_policy = args.vae_float_policy.load_policy();
    let denoiser_policy = if args.profile.is_q8() {
        BooguFloatLoadPolicy::AdaptToF32
    } else {
        BooguFloatLoadPolicy::Preserve
    };
    let denoiser_quantized_policy = BooguQuantizedLoadPolicy::Preserve;
    let qwen_quantized_policy = if args.profile.is_q8() {
        BooguQuantizedLoadPolicy::DequantizeF16
    } else {
        BooguQuantizedLoadPolicy::Preserve
    };
    let execution_dtypes =
        BooguRuntimeDTypes::from_artifact_policies(profile, vae_policy, denoiser_policy);

    let qwen_started = Instant::now();
    let qwen_source = VerifiedBurnpackQwenStageSource::<B, _>::from_directory_auto(
        &identity,
        &args.artifacts,
        inventory.clone(),
        qwen_config.clone(),
        profile,
        device.clone(),
    )?
    .with_quantized_load_policy(qwen_quantized_policy);
    let qwen_plan = qwen_source.plan().clone();
    let qwen_source_verification_milliseconds = qwen_started.elapsed().as_secs_f64() * 1_000.0;

    let vae_started = Instant::now();
    let vae = VerifiedDirectoryVaeStageSource::<B>::new(
        &identity,
        &args.artifacts,
        inventory.clone(),
        vae_config,
        profile,
        vae_policy,
        device.clone(),
    )?;
    let vae_source_verification_milliseconds = vae_started.elapsed().as_secs_f64() * 1_000.0;

    let denoiser_started = Instant::now();
    let (mut denoiser, denoiser_report) = load_resident_denoiser_from_directory_with_policies::<B>(
        &identity,
        &args.artifacts,
        inventory,
        denoiser_config,
        profile,
        denoiser_policy,
        denoiser_quantized_policy,
        &device,
    )?;
    if args.native_runtime_policy == Some(NativeRuntimePolicyChoice::LowVram)
        && variant == BooguVariant::Image01Turbo
    {
        denoiser.ref_image_refiner.clear();
    }
    let denoiser_reference_refiner_modules_retained = denoiser.ref_image_refiner.len();
    let denoiser = configure_denoiser(
        denoiser,
        DenoiserConfiguration {
            attention_policy: args.denoiser_attention_policy,
            rms_norm_policy: args.denoiser_rms_norm_policy,
            qk_preparation_policy: args.denoiser_qk_preparation_policy,
            split_double_stream_shared_projection: args
                .denoiser_split_double_stream_shared_projection,
            query_chunk_size: query_chunk_sizes.denoiser,
            blackbox_num_planes: args.blackbox_num_planes,
            blackbox_seq_kv_tiles: args.blackbox_seq_kv_tiles,
            blackbox_seq_q_tiles: args.blackbox_seq_q_tiles,
        },
    );
    <B as Backend>::sync(&device)?;
    let denoiser_load_milliseconds = denoiser_started.elapsed().as_secs_f64() * 1_000.0;
    let timings = TimingReport {
        fixture_and_manifest_verification_milliseconds,
        qwen_source_verification_milliseconds,
        vae_source_verification_milliseconds,
        denoiser_load_milliseconds,
        ..TimingReport::default()
    };
    let context = RunContext {
        args: &args,
        metadata: &metadata,
        fixture: &fixture,
        qwen_config: &qwen_config,
        device: &device,
        execution_dtypes,
        artifact_content_digest,
        qwen_plan_stage_count: qwen_plan.stages.len(),
        qwen_embedding_row_chunks: qwen_plan.embedding_rows.chunks.len(),
        denoiser_report,
        denoiser_reference_refiner_modules_retained,
        query_chunk_sizes,
        timings,
        total_started,
    };

    let mut report = match args.qwen_residency {
        QwenResidency::Streamed => {
            if args.qwen_synchronization_policy != QwenSynchronizationPolicyChoice::PerStage {
                return Err(
                    "--qwen-synchronization-policy deferred requires --qwen-residency retained"
                        .into(),
                );
            }
            let mut qwen = StreamingQwen3Vl::new(qwen_plan, qwen_source);
            qwen.set_query_chunk_size(query_chunk_sizes.qwen);
            let pipeline =
                StreamingBooguPipeline::new(variant, qwen_config.clone(), qwen, vae, denoiser)
                    .with_decoder_group_norm_policy(args.vae_group_norm_policy.execution_policy());
            run_chain(pipeline, context)?
        }
        QwenResidency::Retained => {
            let qwen_source = RetainingQwen3VlStageSource::new(qwen_source)
                .with_synchronization_policy(args.qwen_synchronization_policy.execution_policy());
            let mut qwen = StreamingQwen3Vl::new(qwen_plan, qwen_source);
            qwen.set_query_chunk_size(query_chunk_sizes.qwen);
            let pipeline =
                StreamingBooguPipeline::new(variant, qwen_config.clone(), qwen, vae, denoiser)
                    .with_decoder_group_norm_policy(args.vae_group_norm_policy.execution_policy());
            run_chain(pipeline, context)?
        }
    };
    report.device_memory_qualification = memory_monitor
        .take()
        .map(NvidiaSmiPidMonitor::finish)
        .transpose()?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if args.require {
        if !report.gates.supported {
            return Err(report
                .gates
                .unsupported_reason
                .clone()
                .unwrap_or_else(|| "full-chain gate is unsupported".into())
                .into());
        }
        if !report.gates.passed {
            return Err(format!(
                "full-chain parity gate failed: {}",
                report.gates.failures.join(", ")
            )
            .into());
        }
    }
    if let Some(memory) = &report.device_memory_qualification
        && !memory.passed
    {
        return Err(format!(
            "native low-vram device-memory gate failed: {}",
            memory.failures.join(", ")
        )
        .into());
    }
    Ok(())
}

struct RunContext<'a> {
    args: &'a Args,
    metadata: &'a FixtureMetadata,
    fixture: &'a FixtureStore,
    qwen_config: &'a Qwen3VlConfig,
    device: &'a burn_wgpu::WgpuDevice,
    execution_dtypes: BooguRuntimeDTypes,
    artifact_content_digest: String,
    qwen_plan_stage_count: usize,
    qwen_embedding_row_chunks: usize,
    denoiser_report: BooguLoadReport,
    denoiser_reference_refiner_modules_retained: usize,
    query_chunk_sizes: QueryChunkSizes,
    timings: TimingReport,
    total_started: Instant,
}

fn run_chain<Q>(
    mut pipeline: StreamingBooguPipeline<
        B,
        Q,
        VerifiedDirectoryVaeStageSource<B>,
        FullParityDenoiser,
    >,
    mut context: RunContext<'_>,
) -> Result<FullChainReport, Box<dyn Error>>
where
    Q: Qwen3VlStageSource<B>,
    Q::Error: core::fmt::Display,
{
    let (qwen_input, effective_instruction_length) = fixture_qwen_input::<B>(
        context.fixture,
        context.qwen_config,
        context.execution_dtypes.qwen_visual,
        context.device,
    )?;
    let qwen_started = Instant::now();
    let qwen_output = pipeline
        .qwen
        .forward_base(context.qwen_config, qwen_input, &mut ())
        .map_err(|error| match error {
            StreamingForwardError::Model(error) => {
                format!("streamed Qwen model failed: {error}")
            }
            StreamingForwardError::Source(error) => {
                format!("verified Qwen stage failed: {error}")
            }
        })?;
    <B as Backend>::sync(context.device)?;
    context.timings.qwen_execute_milliseconds = qwen_started.elapsed().as_secs_f64() * 1_000.0;
    let qwen_final_hidden_state = compare_tensor(
        context.fixture,
        "conditioning.qwen_final_hidden_state",
        "qwen.last_hidden_state",
        qwen_output.last_hidden_state.clone(),
    )?;
    let instruction =
        trim_instruction_features(qwen_output.last_hidden_state, effective_instruction_length)?
            .cast(context.execution_dtypes.denoiser);

    let (reference, edit_reference_latent, edit_reference_bf16_drift) =
        if context.metadata.variant()?.is_edit() {
            let normalized = tensor4::<B>(
                context.fixture,
                "vae.reference_input",
                context.execution_dtypes.vae,
                context.device,
            )?;
            let epsilon = tensor4::<B>(
                context.fixture,
                "vae.reference_epsilon",
                context.execution_dtypes.vae,
                context.device,
            )?;
            let encode_started = Instant::now();
            let encoded = pipeline.encode_reference(normalized, epsilon)?;
            <B as Backend>::sync(context.device)?;
            context.timings.vae_encode_milliseconds =
                encode_started.elapsed().as_secs_f64() * 1_000.0;
            let (primary_oracle, alternate_oracle) =
                edit_reference_oracles(context.args.vae_float_policy);
            let primary = compare_tensor(
                context.fixture,
                "conditioning.edit_reference_latent",
                primary_oracle,
                encoded.clone(),
            )?;
            let bf16_drift = compare_tensor(
                context.fixture,
                "conditioning.edit_reference_alternate_dtype_drift",
                alternate_oracle,
                encoded.clone(),
            )?;
            (
                Some(encoded.cast(context.execution_dtypes.denoiser)),
                Some(primary),
                Some(bf16_drift),
            )
        } else {
            (None, None, None)
        };

    let task = context.metadata.task()?;
    let schedule = DmdSchedule::upstream_for_dtype(task, context.execution_dtypes.denoiser);
    let fixture_sigma_oracle = fixture_sigma_oracle(task, &context.metadata.dtype)?;
    let mut latents = tensor4::<B>(
        context.fixture,
        "dmd.initial_latents",
        context.execution_dtypes.denoiser,
        context.device,
    )?;
    let initial_latent = compare_tensor(
        context.fixture,
        "trajectory.initial_latent",
        "dmd.initial_latents",
        latents.clone(),
    )?;
    let mut steps = Vec::with_capacity(schedule.sigmas().len());
    for (index, &sigma) in schedule.sigmas().iter().enumerate() {
        let fixture_sigma = scalar_f32(context.fixture, &format!("dmd.step.{index}.sigma"))?;
        let fixture_dtype_oracle_sigma = fixture_sigma_oracle[index];
        let input = compare_tensor(
            context.fixture,
            &format!("trajectory.step.{index}.input"),
            &format!("dmd.step.{index}.input"),
            latents.clone(),
        )?;
        let timestep = Tensor::<B, 1>::from_data(TensorData::new(vec![sigma], [1]), context.device)
            .cast(context.execution_dtypes.denoiser);
        let step_started = Instant::now();
        let velocity = DmdDenoiser::predict(
            &mut pipeline.denoiser,
            BooguDenoiserInput {
                latent: latents.clone(),
                timestep,
                instruction: instruction.clone(),
                reference: reference.clone(),
            },
        )?;
        <B as Backend>::sync(context.device)?;
        let execution_milliseconds = step_started.elapsed().as_secs_f64() * 1_000.0;
        context
            .timings
            .dmd_step_milliseconds
            .push(execution_milliseconds);
        let velocity_metric = compare_tensor(
            context.fixture,
            &format!("trajectory.step.{index}.velocity"),
            &format!("dmd.step.{index}.velocity"),
            velocity.clone(),
        )?;
        let prediction = dmd_prediction(latents, velocity, sigma);
        let prediction_oracle = if index + 1 == schedule.sigmas().len() {
            "dmd.final_latents".to_owned()
        } else {
            format!("dmd.step.{index}.prediction")
        };
        let prediction_metric = compare_tensor(
            context.fixture,
            &format!("trajectory.step.{index}.prediction"),
            &prediction_oracle,
            prediction.clone(),
        )?;
        let renoised = if let Some(&next_sigma) = schedule.sigmas().get(index + 1) {
            let noise = tensor4::<B>(
                context.fixture,
                &format!("dmd.step.{index}.noise"),
                context.execution_dtypes.denoiser,
                context.device,
            )?;
            latents = dmd_renoise(prediction, noise, next_sigma);
            Some(compare_tensor(
                context.fixture,
                &format!("trajectory.step.{index}.renoised"),
                &format!("dmd.step.{index}.renoised"),
                latents.clone(),
            )?)
        } else {
            latents = prediction;
            None
        };
        steps.push(DmdStepMetric {
            index,
            schedule_sigma: sigma,
            fixture_sigma,
            fixture_dtype_oracle_sigma,
            fixture_dtype_oracle_abs_difference: (fixture_sigma - fixture_dtype_oracle_sigma).abs(),
            sigma_abs_difference: (sigma - fixture_sigma).abs(),
            execution_milliseconds,
            input,
            velocity: velocity_metric,
            prediction: prediction_metric,
            renoised,
        });
    }
    let final_latent = compare_tensor(
        context.fixture,
        "trajectory.final_latent",
        "dmd.final_latents",
        latents.clone(),
    )?;

    let decode_started = Instant::now();
    let decoded = pipeline.decode(latents.cast(context.execution_dtypes.vae))?;
    <B as Backend>::sync(context.device)?;
    context.timings.vae_decode_milliseconds = decode_started.elapsed().as_secs_f64() * 1_000.0;
    let decoded_tensor = compare_tensor(
        context.fixture,
        "full_chain_output.decoded_tensor",
        "vae.decode_output",
        decoded.clone(),
    )?;
    let final_rgb = compare_output_rgb(
        decoder_output_to_host(decoded)?,
        context.fixture,
        context.metadata.width,
        context.metadata.height,
    )?;

    context.timings.total_milliseconds = context.total_started.elapsed().as_secs_f64() * 1_000.0;
    let conditioning = ConditioningMetrics {
        qwen_final_hidden_state,
        edit_reference_latent,
        edit_reference_bf16_drift,
    };
    let trajectory = TrajectoryMetrics {
        initial_latent,
        steps,
        final_latent,
    };
    let full_chain_output = FullChainOutputMetrics {
        metric_scope: "propagated-qwen-reference-dmd-vae-rgb",
        decoded_tensor,
        final_rgb,
    };
    let gates = evaluate_gates(
        context.metadata.variant()?,
        context.args.profile,
        context.args.vae_float_policy,
        &conditioning,
        &trajectory,
        &full_chain_output,
    );
    Ok(FullChainReport {
        report_schema_version: 1,
        metric_scope: "real-artifact-full-chain-propagated",
        variant: context.metadata.variant.clone(),
        fixture_schema_version: context.metadata.schema_version,
        fixture_dtype: context.metadata.dtype.clone(),
        fixture_prompt: context.metadata.prompt.clone(),
        fixture_seed: context.metadata.seed,
        width: context.metadata.width,
        height: context.metadata.height,
        model_revision: context.metadata.model_revision.clone(),
        upstream_source_revision: context.metadata.upstream_source_revision.clone(),
        artifact_content_digest: context.artifact_content_digest,
        artifact_profile: context.args.profile.slug().into(),
        backend: "burn-wgpu-native",
        policies: policy_report(
            context.args,
            context.metadata.variant()?,
            context.query_chunk_sizes,
        ),
        execution_dtypes: DTypeReport {
            qwen_visual: context.execution_dtypes.qwen_visual.name().into(),
            qwen_text: "f16",
            vae: context.execution_dtypes.vae.name().into(),
            denoiser: context.execution_dtypes.denoiser.name().into(),
        },
        qwen_stage_count: context.qwen_plan_stage_count,
        qwen_embedding_row_chunks: context.qwen_embedding_row_chunks,
        denoiser_loaded_tensors: context.denoiser_report.tensors,
        denoiser_loaded_shards: context.denoiser_report.shards,
        denoiser_reference_refiner_modules_retained: context
            .denoiser_reference_refiner_modules_retained,
        effective_instruction_length,
        timings: context.timings,
        conditioning,
        trajectory,
        full_chain_output,
        gates,
        device_memory_qualification: None,
    })
}

fn decode_float_bytes(name: &str, dtype: Dtype, bytes: &[u8]) -> Result<Vec<f32>, Box<dyn Error>> {
    let values = match dtype {
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
            .collect(),
        Dtype::F16 => bytes
            .chunks_exact(2)
            .map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect(),
        Dtype::BF16 => bytes
            .chunks_exact(2)
            .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect(),
        other => {
            return Err(
                format!("fixture tensor {name} has unsupported float dtype {other:?}").into(),
            );
        }
    };
    Ok(values)
}

fn validate_fixture_contract<I: TensorIndex>(
    fixture: &I,
    metadata: &FixtureMetadata,
) -> Result<(), Box<dyn Error>> {
    let capture_dtype = match metadata.dtype.as_str() {
        "bf16" => Dtype::BF16,
        "f16" => Dtype::F16,
        _ => {
            return Err(format!(
                "fixture dtype must be bf16 or f16, found {:?}",
                metadata.dtype
            )
            .into());
        }
    };
    if metadata.width == 0
        || metadata.height == 0
        || !metadata.width.is_multiple_of(8)
        || !metadata.height.is_multiple_of(8)
    {
        return Err("fixture dimensions must be non-zero multiples of eight".into());
    }
    let input_ids = require_tensor(
        fixture,
        "processor.input_ids",
        ExpectedDType::I64,
        ExpectedShape::Rank(2),
    )?;
    let [batch, sequence]: [usize; 2] = input_ids
        .shape
        .as_slice()
        .try_into()
        .expect("rank was validated");
    if batch != 1 || sequence == 0 {
        return Err("released fixtures must contain one non-empty processor sample".into());
    }
    let token_shape = vec![1, sequence];
    for name in [
        "processor.attention_mask",
        "processor.mm_token_type_ids",
        "qwen.attention_mask",
    ] {
        require_tensor(
            fixture,
            name,
            ExpectedDType::I64,
            ExpectedShape::Exact(token_shape.clone()),
        )?;
    }
    require_tensor(
        fixture,
        "qwen.last_hidden_state",
        ExpectedDType::Exact(capture_dtype),
        ExpectedShape::Exact(vec![1, sequence, 4096]),
    )?;

    let latent_shape = vec![1, 16, metadata.height / 8, metadata.width / 8];
    for name in [
        "dmd.initial_latents",
        "dmd.final_latents",
        "vae.decode_input",
    ] {
        require_tensor(
            fixture,
            name,
            ExpectedDType::Exact(capture_dtype),
            ExpectedShape::Exact(latent_shape.clone()),
        )?;
    }
    require_tensor(
        fixture,
        "vae.decode_output",
        ExpectedDType::Exact(capture_dtype),
        ExpectedShape::Exact(vec![1, 3, metadata.height, metadata.width]),
    )?;
    require_tensor(
        fixture,
        "output.rgb_u8",
        ExpectedDType::U8,
        ExpectedShape::Exact(vec![metadata.height, metadata.width, 3]),
    )?;
    for index in 0..4 {
        require_tensor(
            fixture,
            &format!("dmd.step.{index}.sigma"),
            ExpectedDType::Exact(capture_dtype),
            ExpectedShape::Scalar,
        )?;
        for suffix in ["input", "velocity"] {
            require_tensor(
                fixture,
                &format!("dmd.step.{index}.{suffix}"),
                ExpectedDType::Exact(capture_dtype),
                ExpectedShape::Exact(latent_shape.clone()),
            )?;
        }
        if index < 3 {
            for suffix in ["noise", "prediction", "renoised"] {
                require_tensor(
                    fixture,
                    &format!("dmd.step.{index}.{suffix}"),
                    ExpectedDType::Exact(capture_dtype),
                    ExpectedShape::Exact(latent_shape.clone()),
                )?;
            }
        }
    }

    match metadata.variant()? {
        BooguVariant::Image01Turbo => {
            for unexpected in [
                "processor.pixel_values",
                "processor.image_grid_thw",
                "vae.reference_input",
                "vae.reference_epsilon",
                "vae.reference_scaled_latent",
                "vae.reference_f32_scaled_latent",
            ] {
                if fixture.tensor_spec(unexpected).is_some() {
                    return Err(format!("Turbo fixture unexpectedly contains {unexpected}").into());
                }
            }
        }
        BooguVariant::Image01EditTurbo | BooguVariant::Image01EditTurbo1k5 => {
            require_tensor(
                fixture,
                "processor.pixel_values",
                ExpectedDType::Exact(Dtype::F32),
                ExpectedShape::Rank(2),
            )?;
            let grid = require_tensor(
                fixture,
                "processor.image_grid_thw",
                ExpectedDType::I64,
                ExpectedShape::Rank(2),
            )?;
            if grid.shape[1] != 3 || grid.shape[0] == 0 {
                return Err("processor.image_grid_thw must have non-empty shape [N,3]".into());
            }
            let reference_input = require_tensor(
                fixture,
                "vae.reference_input",
                ExpectedDType::Exact(Dtype::F32),
                ExpectedShape::Rank(4),
            )?;
            let [
                reference_batch,
                reference_channels,
                reference_height,
                reference_width,
            ]: [usize; 4] = reference_input
                .shape
                .as_slice()
                .try_into()
                .expect("rank was validated");
            if reference_batch != 1
                || reference_channels != 3
                || reference_height == 0
                || reference_width == 0
                || !reference_height.is_multiple_of(8)
                || !reference_width.is_multiple_of(8)
            {
                return Err(
                    "Edit fixture VAE reference input must have shape [1,3,H,W] with non-zero multiples of eight"
                        .into(),
                );
            }
            let reference_latent_shape = vec![1, 16, reference_height / 8, reference_width / 8];
            for name in ["vae.reference_epsilon", "vae.reference_scaled_latent"] {
                require_tensor(
                    fixture,
                    name,
                    ExpectedDType::Exact(capture_dtype),
                    ExpectedShape::Exact(reference_latent_shape.clone()),
                )?;
            }
            require_tensor(
                fixture,
                "vae.reference_f32_scaled_latent",
                ExpectedDType::Exact(Dtype::F32),
                ExpectedShape::Exact(reference_latent_shape),
            )?;
        }
    }
    Ok(())
}

fn require_tensor<I: TensorIndex>(
    fixture: &I,
    name: &str,
    expected_dtype: ExpectedDType,
    expected_shape: ExpectedShape,
) -> Result<TensorSpec, Box<dyn Error>> {
    let spec = fixture
        .tensor_spec(name)
        .ok_or_else(|| format!("fixture omits required tensor {name}"))?;
    let dtype_matches = match expected_dtype {
        ExpectedDType::I64 => spec.dtype == Dtype::I64,
        ExpectedDType::U8 => spec.dtype == Dtype::U8,
        ExpectedDType::Exact(dtype) => spec.dtype == dtype,
    };
    if !dtype_matches {
        return Err(format!("fixture tensor {name} has invalid dtype {:?}", spec.dtype).into());
    }
    let shape_matches = match &expected_shape {
        ExpectedShape::Exact(shape) => spec.shape == *shape,
        ExpectedShape::Rank(rank) => spec.shape.len() == *rank,
        ExpectedShape::Scalar => spec.shape.is_empty() || spec.shape == [1],
    };
    if !shape_matches {
        return Err(format!(
            "fixture tensor {name} has shape {:?}, expected {expected_shape:?}",
            spec.shape
        )
        .into());
    }
    Ok(spec)
}

fn fixture_qwen_input<B: Backend>(
    fixture: &FixtureStore,
    config: &Qwen3VlConfig,
    visual_dtype: DType,
    device: &B::Device,
) -> Result<(Qwen3VlModelInput<B>, usize), Box<dyn Error>> {
    let (input_shape, input_ids) = fixture.i64("processor.input_ids")?;
    let [batch, sequence]: [usize; 2] = input_shape
        .try_into()
        .map_err(|_| "processor.input_ids must be rank two")?;
    let (mask_shape, attention) = fixture.i64("processor.attention_mask")?;
    let (qwen_mask_shape, qwen_attention) = fixture.i64("qwen.attention_mask")?;
    let (type_shape, token_types) = fixture.i64("processor.mm_token_type_ids")?;
    if mask_shape != [batch, sequence]
        || qwen_mask_shape != [batch, sequence]
        || type_shape != [batch, sequence]
    {
        return Err("processor/Qwen mask and token-type shapes do not match input_ids".into());
    }
    if attention != qwen_attention {
        return Err("processor.attention_mask differs from captured qwen.attention_mask".into());
    }
    if batch != 1 {
        return Err("released parity fixtures must contain one processor sample".into());
    }
    let input_rows = input_ids
        .chunks_exact(sequence)
        .map(<[i64]>::to_vec)
        .collect::<Vec<_>>();
    let mask_rows = attention
        .chunks_exact(sequence)
        .map(|row| row.iter().map(|&value| value != 0).collect())
        .collect::<Vec<Vec<bool>>>();
    let type_rows = token_types
        .chunks_exact(sequence)
        .map(|row| {
            row.iter()
                .map(|&value| u8::try_from(value))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let effective_instruction_length = right_padded_length(&mask_rows[0])?;

    let grids = if fixture.contains("processor.image_grid_thw") {
        let (shape, values) = fixture.i64("processor.image_grid_thw")?;
        if shape.len() != 2 || shape[1] != 3 {
            return Err("processor.image_grid_thw must have shape [N,3]".into());
        }
        values
            .chunks_exact(3)
            .map(|value| {
                Ok(Grid::new(
                    usize::try_from(value[0])?,
                    usize::try_from(value[1])?,
                    usize::try_from(value[2])?,
                ))
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?
    } else {
        Vec::new()
    };
    let visual_indices = type_rows[0]
        .iter()
        .enumerate()
        .filter_map(|(index, &kind)| (kind != 0).then_some(index))
        .collect::<Vec<_>>();
    let encoding = BatchEncoding {
        input_ids: input_rows,
        attention_mask: mask_rows,
        mm_token_type_ids: type_rows,
        visual_token_indices: vec![visual_indices],
        image_grids: vec![grids.clone()],
        video_grids: vec![Vec::new()],
    };
    let tensors = encoding.to_tensors::<B>(device)?;
    let position_ids = encoding.position_ids(config.vision_config.spatial_merge_size)?;
    let images = if fixture.contains("processor.pixel_values") {
        let (shape, values) = fixture.f32("processor.pixel_values")?;
        let shape: [usize; 2] = shape
            .try_into()
            .map_err(|_| "processor.pixel_values must be rank two")?;
        Some(Qwen3VlVisualInput {
            patches: Tensor::<B, 2>::from_data(TensorData::new(values, shape), device)
                .cast(visual_dtype),
            grids,
            token_indices: encoding.flattened_image_token_indices(),
        })
    } else {
        if !grids.is_empty() || !encoding.flattened_image_token_indices().is_empty() {
            return Err("visual grid/token metadata exists without processor.pixel_values".into());
        }
        None
    };
    Ok((
        Qwen3VlModelInput {
            input_ids: tensors.input_ids,
            attention_mask: Some(tensors.attention_mask),
            position_ids: Some(position_ids),
            images,
            videos: None,
            output_hidden_states: false,
        },
        effective_instruction_length,
    ))
}

fn right_padded_length(mask: &[bool]) -> Result<usize, Box<dyn Error>> {
    let length = mask.iter().take_while(|&&value| value).count();
    if length == 0 || mask[length..].iter().any(|&value| value) {
        return Err("processor attention mask must be non-empty and right padded".into());
    }
    Ok(length)
}

fn tensor4<B: Backend>(
    fixture: &FixtureStore,
    name: &str,
    dtype: DType,
    device: &B::Device,
) -> Result<Tensor<B, 4>, Box<dyn Error>> {
    let (shape, values) = fixture.f32(name)?;
    let shape: [usize; 4] = shape
        .try_into()
        .map_err(|_| format!("fixture tensor {name} must be rank four"))?;
    Ok(Tensor::<B, 4>::from_data(TensorData::new(values, shape), device).cast(dtype))
}

fn scalar_f32(fixture: &FixtureStore, name: &str) -> Result<f32, Box<dyn Error>> {
    let (_, values) = fixture.f32(name)?;
    match values.as_slice() {
        [value] if value.is_finite() => Ok(*value),
        [..] => Err(format!("fixture tensor {name} must contain one finite scalar").into()),
    }
}

/// Direct `torch.linspace(..., dtype=...)[:-1]` values from the pinned upstream schedule.
/// BF16 Turbo step one intentionally differs from an F32 linspace rounded after construction.
fn fixture_sigma_oracle(task: BooguTask, fixture_dtype: &str) -> Result<[f32; 4], Box<dyn Error>> {
    let dtype = match fixture_dtype {
        "bf16" => DType::BF16,
        "f16" => DType::F16,
        other => return Err(format!("unsupported fixture schedule dtype {other:?}").into()),
    };
    let schedule = DmdSchedule::upstream_for_dtype(task, dtype);
    schedule
        .sigmas()
        .try_into()
        .map_err(|_| "released fixture schedule must contain four sigmas".into())
}

struct Comparison {
    max_abs: f32,
    mean_abs: f32,
    rmse: f32,
    relative_rmse: f32,
    cosine: f32,
}

fn compare_tensor<B: Backend, const D: usize>(
    fixture: &FixtureStore,
    name: &str,
    oracle: &str,
    actual: Tensor<B, D>,
) -> Result<TensorMetric, Box<dyn Error>> {
    let started = Instant::now();
    let shape = actual.dims().to_vec();
    let actual_dtype = actual.dtype().name().to_owned();
    let actual = actual.to_data().convert_dtype(DType::F32).to_vec::<f32>()?;
    let (expected_shape, expected) = fixture.f32(oracle)?;
    if shape != expected_shape {
        return Err(format!(
            "oracle {oracle} has shape {expected_shape:?}, Burn produced {shape:?}"
        )
        .into());
    }
    let comparison = compare(&actual, &expected)?;
    Ok(TensorMetric {
        name: name.into(),
        oracle: oracle.into(),
        shape,
        actual_dtype,
        max_abs: comparison.max_abs,
        mean_abs: comparison.mean_abs,
        rmse: comparison.rmse,
        relative_rmse: comparison.relative_rmse,
        cosine_similarity: comparison.cosine,
        readback_milliseconds: started.elapsed().as_secs_f64() * 1_000.0,
    })
}

fn compare(actual: &[f32], expected: &[f32]) -> Result<Comparison, Box<dyn Error>> {
    if actual.len() != expected.len() || actual.is_empty() {
        return Err(format!(
            "comparison length mismatch: actual={} expected={}",
            actual.len(),
            expected.len()
        )
        .into());
    }
    let mut max_abs = 0.0_f32;
    let mut sum_abs = 0.0_f64;
    let mut sum_squared = 0.0_f64;
    let mut dot = 0.0_f64;
    let mut actual_squared = 0.0_f64;
    let mut expected_squared = 0.0_f64;
    for (&actual, &expected) in actual.iter().zip(expected) {
        if !actual.is_finite() || !expected.is_finite() {
            return Err("comparison contains a non-finite value".into());
        }
        let difference = f64::from(actual) - f64::from(expected);
        max_abs = max_abs.max(difference.abs() as f32);
        sum_abs += difference.abs();
        sum_squared += difference * difference;
        dot += f64::from(actual) * f64::from(expected);
        actual_squared += f64::from(actual).powi(2);
        expected_squared += f64::from(expected).powi(2);
    }
    let count = actual.len() as f64;
    let rmse = (sum_squared / count).sqrt();
    let expected_rms = (expected_squared / count).sqrt();
    let denominator = (actual_squared * expected_squared).sqrt();
    let relative_rmse = if expected_rms == 0.0 {
        if rmse == 0.0 { 0.0 } else { f32::MAX }
    } else {
        (rmse / expected_rms) as f32
    };
    let cosine = if actual_squared == 0.0 && expected_squared == 0.0 {
        1.0
    } else if denominator == 0.0 {
        0.0
    } else {
        (dot / denominator).clamp(-1.0, 1.0) as f32
    };
    Ok(Comparison {
        max_abs,
        mean_abs: (sum_abs / count) as f32,
        rmse: rmse as f32,
        relative_rmse,
        cosine,
    })
}

fn compare_output_rgb(
    actual: HostImage,
    fixture: &FixtureStore,
    width: usize,
    height: usize,
) -> Result<ImageMetric, Box<dyn Error>> {
    let HostImage::Pixels(actual) = actual else {
        return Err("decoder postprocess unexpectedly returned an encoded image".into());
    };
    let dimensions = actual.dimensions();
    if dimensions.width() as usize != width
        || dimensions.height() as usize != height
        || actual.format() != PixelFormat::Rgb8
        || actual.color_space() != ColorSpace::Srgb
    {
        return Err(format!(
            "decoder postprocess returned {} {:?}/{:?}, expected {width}x{height} RGB8/sRGB",
            dimensions,
            actual.format(),
            actual.color_space()
        )
        .into());
    }
    let (expected_shape, expected) = fixture.u8("output.rgb_u8")?;
    if expected_shape != [height, width, 3] {
        return Err(format!(
            "output.rgb_u8 has shape {expected_shape:?}, expected [{height},{width},3]"
        )
        .into());
    }
    compare_rgb(actual.bytes(), &expected, width, height)
}

fn compare_rgb(
    actual: &[u8],
    expected: &[u8],
    width: usize,
    height: usize,
) -> Result<ImageMetric, Box<dyn Error>> {
    let expected_len = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or("RGB dimensions overflow")?;
    if actual.len() != expected_len || expected.len() != expected_len || expected_len == 0 {
        return Err(format!(
            "final RGB byte count differs: actual={} expected={} dimensions={width}x{height}",
            actual.len(),
            expected.len()
        )
        .into());
    }
    let mut max_abs = 0_u8;
    let mut exact = 0_usize;
    let mut sum_abs = 0.0_f64;
    let mut sum_squared = 0.0_f64;
    for (&actual, &expected) in actual.iter().zip(expected) {
        let difference = actual.abs_diff(expected);
        max_abs = max_abs.max(difference);
        exact += usize::from(difference == 0);
        sum_abs += f64::from(difference);
        sum_squared += f64::from(difference).powi(2);
    }
    let count = expected_len as f64;
    let rmse = (sum_squared / count).sqrt();
    Ok(ImageMetric {
        max_abs_u8: max_abs,
        mean_abs_u8: (sum_abs / count) as f32,
        rmse_u8: rmse as f32,
        psnr_db: if rmse == 0.0 {
            100.0
        } else {
            (20.0 * (255.0 / rmse).log10()) as f32
        },
        mean_block_ssim_8x8: mean_block_ssim_8x8(actual, expected, width, height)?,
        exact_fraction: (exact as f64 / count) as f32,
    })
}

fn mean_block_ssim_8x8(
    actual: &[u8],
    expected: &[u8],
    width: usize,
    height: usize,
) -> Result<f32, Box<dyn Error>> {
    let expected_len = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or("RGB dimensions overflow")?;
    if actual.len() != expected_len || expected.len() != expected_len {
        return Err("SSIM input length differs from RGB dimensions".into());
    }
    let c1 = (0.01_f64 * 255.0).powi(2);
    let c2 = (0.03_f64 * 255.0).powi(2);
    let mut total = 0.0_f64;
    let mut blocks = 0_usize;
    for top in (0..height).step_by(8) {
        for left in (0..width).step_by(8) {
            let bottom = (top + 8).min(height);
            let right = (left + 8).min(width);
            for channel in 0..3 {
                let samples = (bottom - top) * (right - left);
                let mut actual_mean = 0.0_f64;
                let mut expected_mean = 0.0_f64;
                for y in top..bottom {
                    for x in left..right {
                        let index = (y * width + x) * 3 + channel;
                        actual_mean += f64::from(actual[index]);
                        expected_mean += f64::from(expected[index]);
                    }
                }
                let count = samples as f64;
                actual_mean /= count;
                expected_mean /= count;
                let mut actual_variance = 0.0_f64;
                let mut expected_variance = 0.0_f64;
                let mut covariance = 0.0_f64;
                for y in top..bottom {
                    for x in left..right {
                        let index = (y * width + x) * 3 + channel;
                        let actual_delta = f64::from(actual[index]) - actual_mean;
                        let expected_delta = f64::from(expected[index]) - expected_mean;
                        actual_variance += actual_delta * actual_delta;
                        expected_variance += expected_delta * expected_delta;
                        covariance += actual_delta * expected_delta;
                    }
                }
                actual_variance /= count;
                expected_variance /= count;
                covariance /= count;
                total += ((2.0 * actual_mean * expected_mean + c1) * (2.0 * covariance + c2))
                    / ((actual_mean.powi(2) + expected_mean.powi(2) + c1)
                        * (actual_variance + expected_variance + c2));
                blocks += 1;
            }
        }
    }
    Ok((total / blocks as f64) as f32)
}

fn evaluate_gates(
    variant: BooguVariant,
    profile: ProfileChoice,
    vae_float_policy: VaeFloatPolicyChoice,
    conditioning: &ConditioningMetrics,
    trajectory: &TrajectoryMetrics,
    output: &FullChainOutputMetrics,
) -> GateEvaluation {
    let Some(gates) = gate_set(variant, profile, vae_float_policy) else {
        let reason = if variant == BooguVariant::Image01EditTurbo1k5 && profile.is_q8() {
            "Edit-Turbo 1.5K Q8 is not a supported release profile: no pinned 1.5K Q8 full-chain parity evidence exists"
        } else {
            "Edit-Turbo requires a qwen-vision-f32 profile: F16 Qwen vision diverged from the pinned upstream oracle"
        };
        return GateEvaluation {
            supported: false,
            passed: false,
            unsupported_reason: Some(reason.into()),
            failures: Vec::new(),
            thresholds: None,
        };
    };
    let mut failures = Vec::new();
    check_tensor_gate(
        &conditioning.qwen_final_hidden_state,
        gates.qwen_final,
        &mut failures,
    );
    if variant.is_edit() {
        match &conditioning.edit_reference_latent {
            Some(metric) => check_reference_gate(metric, gates.edit_reference, &mut failures),
            None => failures.push("conditioning.edit_reference_latent is absent".into()),
        }
    }
    check_tensor_gate(
        &trajectory.initial_latent,
        gates.dmd_boundaries,
        &mut failures,
    );
    for step in &trajectory.steps {
        check_tensor_gate(&step.input, gates.dmd_boundaries, &mut failures);
        check_tensor_gate(&step.velocity, gates.dmd_boundaries, &mut failures);
        check_tensor_gate(&step.prediction, gates.dmd_boundaries, &mut failures);
        if let Some(metric) = &step.renoised {
            check_tensor_gate(metric, gates.dmd_boundaries, &mut failures);
        }
        if step.fixture_dtype_oracle_abs_difference > 1.0e-7 {
            failures.push(format!(
                "dmd.step.{}.fixture_sigma differs from its direct fixture-dtype oracle by {}",
                step.index, step.fixture_dtype_oracle_abs_difference
            ));
        }
    }
    check_tensor_gate(&trajectory.final_latent, gates.dmd_final, &mut failures);
    check_tensor_gate(
        &output.decoded_tensor,
        gates.propagated_decode,
        &mut failures,
    );
    if output.final_rgb.psnr_db < gates.final_rgb.minimum_psnr_db
        || output.final_rgb.mean_block_ssim_8x8 < gates.final_rgb.minimum_mean_block_ssim_8x8
    {
        failures.push(format!(
            "final_rgb (PSNR {}, block SSIM {}) misses ({}, {})",
            output.final_rgb.psnr_db,
            output.final_rgb.mean_block_ssim_8x8,
            gates.final_rgb.minimum_psnr_db,
            gates.final_rgb.minimum_mean_block_ssim_8x8
        ));
    }
    GateEvaluation {
        supported: true,
        passed: failures.is_empty(),
        unsupported_reason: None,
        failures,
        thresholds: Some(gates),
    }
}

fn gate_set(
    variant: BooguVariant,
    profile: ProfileChoice,
    vae_float_policy: VaeFloatPolicyChoice,
) -> Option<GateSet> {
    if variant.is_edit() && !profile.has_f32_qwen_vision() {
        return None;
    }
    if variant == BooguVariant::Image01EditTurbo1k5 && profile.is_q8() {
        return None;
    }
    let (qwen_final, dmd_boundaries, dmd_final, propagated_decode, final_rgb, evidence) = match (
        variant,
        profile.is_q8(),
    ) {
        (BooguVariant::Image01Turbo, false) => (
            TensorGate {
                maximum_relative_rmse: 0.03,
                minimum_cosine_similarity: 0.9995,
            },
            TensorGate {
                maximum_relative_rmse: 0.27,
                minimum_cosine_similarity: 0.964,
            },
            TensorGate {
                maximum_relative_rmse: 0.15,
                minimum_cosine_similarity: 0.989,
            },
            TensorGate {
                maximum_relative_rmse: 0.07,
                minimum_cosine_similarity: 0.997,
            },
            RgbGate {
                minimum_psnr_db: 32.0,
                minimum_mean_block_ssim_8x8: 0.975,
            },
            "Pinned Turbo hybrid F16 Qwen final rel-RMSE 0.02035/cos 0.999793. Upstream Turbo F16-vs-BF16 production-schedule floors: worst trajectory rel-RMSE 0.25993/cos 0.96601, final latent 0.13578/cos 0.99074, decode 0.06186/cos 0.99809, RGB PSNR 33.44/SSIM 0.97999. Production schedule drives this full chain; fixture sigma delta is reported as expected dtype drift.",
        ),
        (BooguVariant::Image01Turbo, true) => (
            TensorGate {
                maximum_relative_rmse: 0.04,
                minimum_cosine_similarity: 0.999,
            },
            TensorGate {
                maximum_relative_rmse: 0.30,
                minimum_cosine_similarity: 0.95,
            },
            TensorGate {
                maximum_relative_rmse: 0.22,
                minimum_cosine_similarity: 0.975,
            },
            TensorGate {
                maximum_relative_rmse: 0.15,
                minimum_cosine_similarity: 0.98,
            },
            RgbGate {
                minimum_psnr_db: 27.0,
                minimum_mean_block_ssim_8x8: 0.93,
            },
            "Pinned Turbo Q8 Qwen final rel-RMSE 0.02897/cos 0.999580; pinned Q8 captured-sigma diagnostic worst trajectory rel-RMSE 0.22839/cos 0.97405 and final latent 0.17953/cos 0.98391. Upstream Turbo F16-vs-BF16 output floors are decode rel-RMSE 0.06186/cos 0.99809 and RGB PSNR 33.44/SSIM 0.97999; Q8 budgets include measured quantization headroom. Production schedule drives this full chain.",
        ),
        (BooguVariant::Image01EditTurbo1k5, false) => (
            TensorGate {
                maximum_relative_rmse: 0.10,
                minimum_cosine_similarity: 0.995,
            },
            TensorGate {
                maximum_relative_rmse: 0.13,
                minimum_cosine_similarity: 0.992,
            },
            TensorGate {
                maximum_relative_rmse: 0.085,
                minimum_cosine_similarity: 0.996,
            },
            TensorGate {
                maximum_relative_rmse: 0.09,
                minimum_cosine_similarity: 0.996,
            },
            RgbGate {
                minimum_psnr_db: 33.5,
                minimum_mean_block_ssim_8x8: 0.99,
            },
            "Pinned Edit-Turbo 1.5K hybrid F32-vision/F16-text WGPU Qwen final rel-RMSE 0.093365/cos 0.995632. Its independent upstream F16-vs-BF16 1536 floors are worst trajectory rel-RMSE 0.121971/cos 0.992577, final latent 0.077752/cos 0.996997, decode 0.079907/cos 0.996903, and RGB PSNR 34.005/SSIM 0.991623. The exact-policy exhaustive WGPU full-chain result is worst trajectory 0.121937/0.992576, final 0.077824/0.996991, decode 0.082429/0.996670, and RGB 33.728/0.992010.",
        ),
        (BooguVariant::Image01EditTurbo, false) => (
            TensorGate {
                maximum_relative_rmse: 0.11,
                minimum_cosine_similarity: 0.994,
            },
            TensorGate {
                maximum_relative_rmse: 0.26,
                minimum_cosine_similarity: 0.965,
            },
            TensorGate {
                maximum_relative_rmse: 0.195,
                minimum_cosine_similarity: 0.98,
            },
            TensorGate {
                maximum_relative_rmse: 0.115,
                minimum_cosine_similarity: 0.99,
            },
            RgbGate {
                minimum_psnr_db: 28.5,
                minimum_mean_block_ssim_8x8: 0.95,
            },
            "Pinned Edit hybrid F32-vision/F16-text Qwen final rel-RMSE 0.09357/cos 0.995613. Upstream Edit F16-vs-BF16 production-schedule floors: worst trajectory rel-RMSE 0.24372/cos 0.97032, final latent 0.18122/cos 0.98357, decode 0.10462/cos 0.99453, RGB PSNR 29.74/SSIM 0.95982. Real staged WGPU F32 VAE reference: max 0.004233/RMSE 0.000133/cos 1.0. Production schedule drives this full chain; fixture sigma delta is expected dtype drift.",
        ),
        (BooguVariant::Image01EditTurbo, true) => (
            TensorGate {
                maximum_relative_rmse: 0.12,
                minimum_cosine_similarity: 0.993,
            },
            TensorGate {
                maximum_relative_rmse: 0.30,
                minimum_cosine_similarity: 0.95,
            },
            TensorGate {
                maximum_relative_rmse: 0.24,
                minimum_cosine_similarity: 0.97,
            },
            TensorGate {
                maximum_relative_rmse: 0.16,
                minimum_cosine_similarity: 0.975,
            },
            RgbGate {
                minimum_psnr_db: 26.0,
                minimum_mean_block_ssim_8x8: 0.92,
            },
            "Pinned Edit hybrid Q8 Qwen final rel-RMSE 0.09618/cos 0.995364; pinned Q8 captured-sigma diagnostic worst trajectory rel-RMSE 0.22839/cos 0.97405 and final latent 0.17953/cos 0.98391. Upstream Edit F16-vs-BF16 output floors are decode rel-RMSE 0.10462/cos 0.99453 and RGB PSNR 29.74/SSIM 0.95982; real staged WGPU F32 VAE reference is max 0.004233/RMSE 0.000133/cos 1.0. Q8 budgets include measured quantization headroom.",
        ),
        (BooguVariant::Image01EditTurbo1k5, true) => {
            unreachable!("Edit-Turbo 1.5K Q8 is rejected before release-gate construction")
        }
    };
    let (vae_evidence, edit_reference) = match vae_float_policy {
        VaeFloatPolicyChoice::ForceF32 => (
            "VAE policy force-f32 adapts the verified F16 VAE tensors and VAE boundary activations to F32; Edit reference conditioning is gated against the pinned PyTorch F32 oracle.",
            ReferenceGate {
                maximum_abs: 0.005,
                maximum_rmse: 0.0002,
                minimum_cosine_similarity: 0.999_999,
            },
        ),
        VaeFloatPolicyChoice::PreserveF16 => (
            "VAE policy preserve-f16 keeps the verified F16 VAE tensors and VAE boundary activations in F16. Pinned retained native-WGPU decoder component evidence against the upstream BF16 fixture: max 0.012696, RMSE 0.001590, cosine 0.999997, RGB PSNR 54.51/SSIM 0.99780. Edit reference conditioning is gated against the fixture's upstream BF16 latent under the measured F16/BF16 dtype envelope: max 0.809571, RMSE 0.075922, cosine 0.998654.",
            ReferenceGate {
                maximum_abs: 0.82,
                maximum_rmse: 0.080,
                minimum_cosine_similarity: 0.998_5,
            },
        ),
    };
    Some(GateSet {
        evidence: format!("{evidence} {vae_evidence}"),
        vae_float_policy: vae_float_policy.id(),
        qwen_final,
        edit_reference,
        dmd_boundaries,
        dmd_final,
        propagated_decode,
        final_rgb,
    })
}

fn check_tensor_gate(metric: &TensorMetric, gate: TensorGate, failures: &mut Vec<String>) {
    if metric.relative_rmse > gate.maximum_relative_rmse
        || metric.cosine_similarity < gate.minimum_cosine_similarity
    {
        failures.push(format!(
            "{} (relative RMSE {}, cosine {}) misses ({}, {})",
            metric.name,
            metric.relative_rmse,
            metric.cosine_similarity,
            gate.maximum_relative_rmse,
            gate.minimum_cosine_similarity
        ));
    }
}

fn check_reference_gate(metric: &TensorMetric, gate: ReferenceGate, failures: &mut Vec<String>) {
    if metric.max_abs > gate.maximum_abs
        || metric.rmse > gate.maximum_rmse
        || metric.cosine_similarity < gate.minimum_cosine_similarity
    {
        failures.push(format!(
            "{} (max {}, RMSE {}, cosine {}) misses ({}, {}, {})",
            metric.name,
            metric.max_abs,
            metric.rmse,
            metric.cosine_similarity,
            gate.maximum_abs,
            gate.maximum_rmse,
            gate.minimum_cosine_similarity
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(dtype: Dtype, shape: impl Into<Vec<usize>>) -> TensorSpec {
        TensorSpec {
            dtype,
            shape: shape.into(),
        }
    }

    fn metadata(variant: &str) -> FixtureMetadata {
        FixtureMetadata {
            schema_version: 2,
            variant: variant.into(),
            model_revision: "model".into(),
            upstream_source_revision: "source".into(),
            width: 256,
            height: 256,
            dtype: "bf16".into(),
            prompt: "test".into(),
            seed: 42,
        }
    }

    fn exact_low_vram_args(variant: BooguVariant) -> Args {
        let policy = native_release_policy(variant);
        let qk_preparation = match policy.denoiser_qk_preparation {
            NativeDenoiserQkPreparationPolicy::Composed => "composed",
            NativeDenoiserQkPreparationPolicy::BalancedStrictQkNormRope => {
                "balanced-strict-qk-norm-rope"
            }
        };
        Args::try_parse_from([
            "boogu-full-parity".to_owned(),
            "--artifacts".to_owned(),
            "artifacts".to_owned(),
            "--fixture".to_owned(),
            "fixture".to_owned(),
            "--profile".to_owned(),
            "f16-qwen-vision-f32".to_owned(),
            "--native-runtime-policy".to_owned(),
            "low-vram".to_owned(),
            "--qwen-residency".to_owned(),
            "streamed".to_owned(),
            "--qwen-synchronization-policy".to_owned(),
            "per-stage".to_owned(),
            "--vae-float-policy".to_owned(),
            "preserve-f16".to_owned(),
            "--vae-group-norm-policy".to_owned(),
            "f16-storage-f32-accum".to_owned(),
            "--vae-attention-query-chunk-size".to_owned(),
            policy.vae_attention_query_chunk_size.to_string(),
            "--qwen-query-chunk-size".to_owned(),
            policy.qwen_query_chunk_size.to_string(),
            "--denoiser-query-chunk-size".to_owned(),
            policy.denoiser_query_chunk_size.to_string(),
            "--denoiser-attention-policy".to_owned(),
            "padded-blackbox".to_owned(),
            "--denoiser-rms-norm-policy".to_owned(),
            "strict-f32".to_owned(),
            "--denoiser-qk-preparation-policy".to_owned(),
            qk_preparation.to_owned(),
            "--blackbox-num-planes".to_owned(),
            policy.blackbox_num_planes.to_string(),
            "--blackbox-seq-kv-tiles".to_owned(),
            policy.blackbox_seq_kv_tiles.to_string(),
            "--blackbox-seq-q-tiles".to_owned(),
            policy.blackbox_seq_q_tiles.to_string(),
            "--require".to_owned(),
        ])
        .unwrap()
    }

    fn fixture_specs(variant: &str) -> BTreeMap<String, TensorSpec> {
        let sequence = if variant == "turbo" { 49 } else { 147 };
        let latent = vec![1, 16, 32, 32];
        let mut specs = BTreeMap::from([
            (
                "processor.input_ids".into(),
                spec(Dtype::I64, vec![1, sequence]),
            ),
            (
                "processor.attention_mask".into(),
                spec(Dtype::I64, vec![1, sequence]),
            ),
            (
                "processor.mm_token_type_ids".into(),
                spec(Dtype::I64, vec![1, sequence]),
            ),
            (
                "qwen.attention_mask".into(),
                spec(Dtype::I64, vec![1, sequence]),
            ),
            (
                "qwen.last_hidden_state".into(),
                spec(Dtype::BF16, vec![1, sequence, 4096]),
            ),
            (
                "dmd.initial_latents".into(),
                spec(Dtype::BF16, latent.clone()),
            ),
            (
                "dmd.final_latents".into(),
                spec(Dtype::BF16, latent.clone()),
            ),
            ("vae.decode_input".into(), spec(Dtype::BF16, latent.clone())),
            (
                "vae.decode_output".into(),
                spec(Dtype::BF16, vec![1, 3, 256, 256]),
            ),
            ("output.rgb_u8".into(), spec(Dtype::U8, vec![256, 256, 3])),
        ]);
        for index in 0..4 {
            specs.insert(
                format!("dmd.step.{index}.sigma"),
                spec(Dtype::BF16, Vec::new()),
            );
            for suffix in ["input", "velocity"] {
                specs.insert(
                    format!("dmd.step.{index}.{suffix}"),
                    spec(Dtype::BF16, latent.clone()),
                );
            }
            if index < 3 {
                for suffix in ["noise", "prediction", "renoised"] {
                    specs.insert(
                        format!("dmd.step.{index}.{suffix}"),
                        spec(Dtype::BF16, latent.clone()),
                    );
                }
            }
        }
        if matches!(variant, "edit-turbo" | "edit-turbo-1k5") {
            specs.insert(
                "processor.pixel_values".into(),
                spec(Dtype::F32, vec![256, 1536]),
            );
            specs.insert(
                "processor.image_grid_thw".into(),
                spec(Dtype::I64, vec![1, 3]),
            );
            specs.insert(
                "vae.reference_input".into(),
                spec(Dtype::F32, vec![1, 3, 256, 256]),
            );
            for name in ["vae.reference_epsilon", "vae.reference_scaled_latent"] {
                specs.insert(name.into(), spec(Dtype::BF16, latent.clone()));
            }
            specs.insert(
                "vae.reference_f32_scaled_latent".into(),
                spec(Dtype::F32, latent.clone()),
            );
        }
        specs
    }

    #[test]
    fn schema_two_contract_accepts_all_release_variants_correctness() {
        for variant in ["turbo", "edit-turbo", "edit-turbo-1k5"] {
            validate_fixture_contract(&fixture_specs(variant), &metadata(variant)).unwrap();
        }
    }

    #[test]
    fn edit_reference_shape_is_independent_from_output_shape_correctness() {
        let mut metadata = metadata("edit-turbo-1k5");
        metadata.width = 1536;
        metadata.height = 1536;
        let mut fixture = fixture_specs("edit-turbo-1k5");
        let output_latent = vec![1, 16, 192, 192];
        for name in [
            "dmd.initial_latents",
            "dmd.final_latents",
            "vae.decode_input",
        ] {
            fixture.insert(name.into(), spec(Dtype::BF16, output_latent.clone()));
        }
        fixture.insert(
            "vae.decode_output".into(),
            spec(Dtype::BF16, vec![1, 3, 1536, 1536]),
        );
        fixture.insert("output.rgb_u8".into(), spec(Dtype::U8, vec![1536, 1536, 3]));
        for index in 0..4 {
            for suffix in ["input", "velocity"] {
                fixture.insert(
                    format!("dmd.step.{index}.{suffix}"),
                    spec(Dtype::BF16, output_latent.clone()),
                );
            }
            if index < 3 {
                for suffix in ["noise", "prediction", "renoised"] {
                    fixture.insert(
                        format!("dmd.step.{index}.{suffix}"),
                        spec(Dtype::BF16, output_latent.clone()),
                    );
                }
            }
        }
        validate_fixture_contract(&fixture, &metadata).unwrap();
    }

    #[test]
    fn fixture_contract_rejects_missing_renoise_and_wrong_rgb_dtype_correctness() {
        let mut fixture = fixture_specs("edit-turbo");
        fixture.remove("dmd.step.2.renoised");
        assert!(validate_fixture_contract(&fixture, &metadata("edit-turbo")).is_err());
        let mut fixture = fixture_specs("turbo");
        fixture.insert("output.rgb_u8".into(), spec(Dtype::F32, vec![256, 256, 3]));
        assert!(validate_fixture_contract(&fixture, &metadata("turbo")).is_err());
    }

    #[test]
    fn comparison_is_exact_for_identity_and_rejects_non_finite_values_correctness() {
        let comparison = compare(&[1.0, -2.0, 3.0], &[1.0, -2.0, 3.0]).unwrap();
        assert_eq!(comparison.max_abs, 0.0);
        assert_eq!(comparison.rmse, 0.0);
        assert!((comparison.cosine - 1.0).abs() <= f32::EPSILON);
        assert_eq!(compare(&[1.0], &[0.0]).unwrap().cosine, 0.0);
        assert!(compare(&[f32::NAN], &[0.0]).is_err());
        assert!(compare(&[0.0], &[f32::INFINITY]).is_err());
    }

    #[test]
    fn block_ssim_is_one_for_identity_and_detects_a_changed_channel_correctness() {
        let identity = (0_u8..45).collect::<Vec<_>>();
        let score = mean_block_ssim_8x8(&identity, &identity, 5, 3).unwrap();
        assert!((score - 1.0).abs() <= f32::EPSILON);
        let metric = compare_rgb(&identity, &identity, 5, 3).unwrap();
        assert_eq!(metric.psnr_db, 100.0);
        assert_eq!(metric.exact_fraction, 1.0);
        let expected = vec![128_u8; 8 * 8 * 3];
        let mut actual = expected.clone();
        for pixel in actual.chunks_exact_mut(3) {
            pixel[1] = 0;
        }
        let score = mean_block_ssim_8x8(&actual, &expected, 8, 8).unwrap();
        assert!((0.6..0.7).contains(&score));
    }

    #[test]
    fn edit_gate_requires_f32_vision_and_profiles_have_distinct_floors_correctness() {
        assert!(
            gate_set(
                BooguVariant::Image01EditTurbo,
                ProfileChoice::F16,
                VaeFloatPolicyChoice::ForceF32,
            )
            .is_none()
        );
        let f16 = gate_set(
            BooguVariant::Image01EditTurbo,
            ProfileChoice::F16QwenVisionF32,
            VaeFloatPolicyChoice::ForceF32,
        )
        .unwrap();
        let q8 = gate_set(
            BooguVariant::Image01EditTurbo,
            ProfileChoice::Q8sBlock32F32QwenVisionF32,
            VaeFloatPolicyChoice::ForceF32,
        )
        .unwrap();
        assert!(q8.qwen_final.maximum_relative_rmse > f16.qwen_final.maximum_relative_rmse);
        assert!(q8.dmd_final.maximum_relative_rmse > f16.dmd_final.maximum_relative_rmse);
    }

    #[test]
    fn denoiser_attention_policy_defaults_to_portable_correctness() {
        let args = Args::try_parse_from([
            "boogu-full-parity",
            "--artifacts",
            "artifacts",
            "--fixture",
            "fixture",
        ])
        .unwrap();

        assert_eq!(
            args.denoiser_attention_policy,
            DenoiserAttentionPolicy::Portable
        );
    }

    #[test]
    fn native_low_vram_selector_maps_exact_release_policy_and_fails_closed_correctness() {
        for variant in [
            BooguVariant::Image01Turbo,
            BooguVariant::Image01EditTurbo,
            BooguVariant::Image01EditTurbo1k5,
        ] {
            let exact = exact_low_vram_args(variant);
            let sizes = resolve_query_chunk_sizes(&exact).unwrap();
            validate_native_runtime_policy(&exact, variant, sizes).unwrap();
            assert!(is_exact_native_release_policy(&exact, variant, sizes));
            let report = serde_json::to_value(policy_report(&exact, variant, sizes)).unwrap();
            assert_eq!(report["native_runtime_policy"], "low-vram");
            assert_eq!(
                report["native_weight_traffic_contract"],
                "phase-resident/qwen+vae-per-run/denoiser-resident-zero-dmd-weight-reloads"
            );
            assert_eq!(
                report["strict_peak_device_ceiling_bytes"],
                NATIVE_LOW_VRAM_STRICT_DEVICE_CEILING_BYTES
            );
            assert_eq!(report["native_release_policy_validated"], true);

            let no_required_gates = Args {
                require: false,
                ..exact_low_vram_args(variant)
            };
            assert!(
                validate_native_runtime_policy(&no_required_gates, variant, sizes)
                    .unwrap_err()
                    .contains("requires --require")
            );
            let retained_qwen = Args {
                qwen_residency: QwenResidency::Retained,
                qwen_synchronization_policy: QwenSynchronizationPolicyChoice::Deferred,
                ..exact_low_vram_args(variant)
            };
            assert!(validate_native_runtime_policy(&retained_qwen, variant, sizes).is_err());
        }
    }

    #[test]
    fn native_low_vram_pid_telemetry_sums_gpus_and_enforces_strict_ceiling_correctness() {
        assert_eq!(
            parse_pid_total_framebuffer_mib("41, 20000\n9, 500\n41, 12767\n", 41).unwrap(),
            Some(32_767)
        );
        assert_eq!(
            parse_pid_total_framebuffer_mib("9, 500\n", 41).unwrap(),
            None
        );
        assert!(parse_pid_total_framebuffer_mib("malformed", 41).is_err());

        let qualified = evaluate_device_memory_qualification(
            41,
            NVIDIA_SMI_SAMPLE_INTERVAL,
            PidFramebufferSamples {
                attempted_samples: 4,
                matched_samples: 4,
                nonzero_samples: 4,
                peak_total_framebuffer_mib: 30_517,
                ..PidFramebufferSamples::default()
            },
        );
        assert!(qualified.passed);
        assert!(qualified.strictly_below_ceiling);

        let at_ceiling = evaluate_device_memory_qualification(
            41,
            NVIDIA_SMI_SAMPLE_INTERVAL,
            PidFramebufferSamples {
                attempted_samples: 4,
                matched_samples: 4,
                nonzero_samples: 4,
                peak_total_framebuffer_mib: 30_518,
                ..PidFramebufferSamples::default()
            },
        );
        assert!(!at_ceiling.passed);
        assert!(!at_ceiling.strictly_below_ceiling);
        assert!(
            at_ceiling
                .failures
                .iter()
                .any(|failure| failure.contains("not strictly below"))
        );

        let incomplete = evaluate_device_memory_qualification(
            41,
            NVIDIA_SMI_SAMPLE_INTERVAL,
            PidFramebufferSamples {
                attempted_samples: 1,
                matched_samples: 1,
                nonzero_samples: 1,
                peak_total_framebuffer_mib: 1,
                sample_error_count: 1,
                sample_errors: vec!["sample failed".into()],
            },
        );
        assert!(!incomplete.passed);
        assert_eq!(incomplete.sample_error_count, 1);
    }

    #[test]
    fn padded_blackbox_policy_rejects_f32_q8_denoisers_correctness() {
        for profile in [
            ProfileChoice::Q8sBlock32F32,
            ProfileChoice::Q8sBlock32F32QwenVisionF32,
        ] {
            let error = validate_denoiser_attention_policy(
                profile,
                DenoiserAttentionPolicy::PaddedBlackbox,
            )
            .unwrap_err();
            assert!(error.contains("requires F16 denoiser execution"));
        }

        validate_denoiser_attention_policy(
            ProfileChoice::F16QwenVisionF32,
            DenoiserAttentionPolicy::PaddedBlackbox,
        )
        .unwrap();
        validate_denoiser_attention_policy(
            ProfileChoice::Q8sBlock32F32,
            DenoiserAttentionPolicy::Portable,
        )
        .unwrap();
    }

    #[test]
    fn policy_report_serializes_selected_denoiser_attention_correctness() {
        let args = Args::try_parse_from([
            "boogu-full-parity",
            "--artifacts",
            "artifacts",
            "--fixture",
            "fixture",
            "--denoiser-attention-policy",
            "padded-blackbox",
            "--qwen-query-chunk-size",
            "128",
            "--denoiser-query-chunk-size",
            "16384",
        ])
        .unwrap();
        let query_chunk_sizes = resolve_query_chunk_sizes(&args).unwrap();
        let report = serde_json::to_value(policy_report(
            &args,
            BooguVariant::Image01EditTurbo,
            query_chunk_sizes,
        ))
        .unwrap();

        assert_eq!(
            report["denoiser_attention_policy"],
            serde_json::json!("padded-blackbox")
        );
        assert_eq!(report["qwen_query_chunk_size"], serde_json::json!(128));
        assert_eq!(
            report["denoiser_query_chunk_size"],
            serde_json::json!(16384)
        );
        assert_eq!(report["vae_group_norm"], serde_json::json!("strict-f32"));
        assert_eq!(
            report["denoiser_rms_norm_policy"],
            serde_json::json!("strict-f32")
        );
        assert_eq!(
            report["denoiser_qk_preparation_policy"],
            serde_json::json!("composed")
        );
        assert_eq!(report["native_autotune"], serde_json::json!("full"));
        assert_eq!(
            report["native_release_policy_validated"],
            serde_json::json!(false)
        );
        assert!(report.get("edit_1k5_release_policy_validated").is_none());
    }

    #[test]
    fn turbo_1k_release_policy_selects_balanced_qk_and_vae_q4096_correctness() {
        let exact = Args::try_parse_from([
            "boogu-full-parity",
            "--artifacts",
            "artifacts",
            "--fixture",
            "fixture",
            "--profile",
            "f16-qwen-vision-f32",
            "--qwen-residency",
            "retained",
            "--qwen-synchronization-policy",
            "deferred",
            "--vae-float-policy",
            "preserve-f16",
            "--vae-group-norm-policy",
            "f16-storage-f32-accum",
            "--vae-attention-query-chunk-size",
            "4096",
            "--qwen-query-chunk-size",
            "128",
            "--denoiser-query-chunk-size",
            "8192",
            "--denoiser-attention-policy",
            "padded-blackbox",
            "--denoiser-qk-preparation-policy",
            "balanced-strict-qk-norm-rope",
            "--blackbox-num-planes",
            "4",
            "--blackbox-seq-kv-tiles",
            "1",
            "--blackbox-seq-q-tiles",
            "1",
        ])
        .unwrap();
        let sizes = resolve_query_chunk_sizes(&exact).unwrap();
        assert!(is_exact_native_release_policy(
            &exact,
            BooguVariant::Image01Turbo,
            sizes,
        ));
        let report =
            serde_json::to_value(policy_report(&exact, BooguVariant::Image01Turbo, sizes)).unwrap();
        assert_eq!(
            report["native_release_policy_validated"],
            serde_json::json!(true)
        );
        assert!(report.get("edit_1k5_release_policy_validated").is_none());
    }

    #[test]
    fn fused_qk_preparation_is_explicit_and_fail_closed_correctness() {
        let args = Args::try_parse_from([
            "boogu-full-parity",
            "--artifacts",
            "artifacts",
            "--fixture",
            "fixture",
            "--denoiser-attention-policy",
            "padded-blackbox",
            "--denoiser-qk-preparation-policy",
            "fused-strict-qk-norm-rope",
            "--allow-unvalidated-1k5-policy",
        ])
        .unwrap();
        validate_denoiser_qk_preparation_policy(
            args.denoiser_attention_policy,
            args.denoiser_rms_norm_policy,
            args.denoiser_qk_preparation_policy,
            args.blackbox_num_planes,
            args.blackbox_seq_kv_tiles,
            args.blackbox_seq_q_tiles,
        )
        .unwrap();
        let report = serde_json::to_value(policy_report(
            &args,
            BooguVariant::Image01EditTurbo1k5,
            resolve_query_chunk_sizes(&args).unwrap(),
        ))
        .unwrap();
        assert_eq!(
            report["denoiser_qk_preparation_policy"],
            serde_json::json!("fused-strict-qk-norm-rope")
        );
        assert_eq!(
            report["edit_1k5_release_policy_validated"],
            serde_json::json!(false)
        );
        assert!(
            validate_denoiser_qk_preparation_policy(
                DenoiserAttentionPolicy::Portable,
                DenoiserRmsNormPolicyChoice::StrictF32,
                DenoiserQkPreparationPolicyChoice::FusedStrictQkNormRope,
                4,
                1,
                1,
            )
            .is_err()
        );
    }

    #[test]
    fn mixed_denoiser_rms_policy_is_explicit_diagnostic_json_correctness() {
        let args = Args::try_parse_from([
            "boogu-full-parity",
            "--artifacts",
            "artifacts",
            "--fixture",
            "fixture",
            "--denoiser-attention-policy",
            "padded-blackbox",
            "--denoiser-rms-norm-policy",
            "f16-storage-f32-accum",
            "--allow-unvalidated-1k5-policy",
        ])
        .unwrap();
        validate_denoiser_rms_norm_policy(
            args.denoiser_attention_policy,
            args.denoiser_rms_norm_policy,
        )
        .unwrap();
        let sizes = resolve_query_chunk_sizes(&args).unwrap();
        let report = serde_json::to_value(policy_report(
            &args,
            BooguVariant::Image01EditTurbo1k5,
            sizes,
        ))
        .unwrap();

        assert_eq!(
            report["denoiser_rms_norm_policy"],
            serde_json::json!("f16-storage-f32-accum")
        );
        assert_eq!(
            report["edit_1k5_release_policy_validated"],
            serde_json::json!(false)
        );
        assert!(
            validate_denoiser_rms_norm_policy(
                DenoiserAttentionPolicy::Portable,
                DenoiserRmsNormPolicyChoice::F16StorageF32Accum,
            )
            .is_err()
        );
    }

    #[test]
    fn edit_1k5_release_policy_is_exact_and_diagnostics_are_explicit_correctness() {
        let exact = Args::try_parse_from([
            "boogu-full-parity",
            "--artifacts",
            "artifacts",
            "--fixture",
            "fixture",
            "--profile",
            "f16-qwen-vision-f32",
            "--qwen-residency",
            "retained",
            "--qwen-synchronization-policy",
            "deferred",
            "--vae-float-policy",
            "preserve-f16",
            "--vae-group-norm-policy",
            "f16-storage-f32-accum",
            "--vae-attention-query-chunk-size",
            "4096",
            "--qwen-query-chunk-size",
            "128",
            "--denoiser-query-chunk-size",
            "16384",
            "--denoiser-attention-policy",
            "padded-blackbox",
            "--blackbox-num-planes",
            "4",
            "--blackbox-seq-kv-tiles",
            "1",
            "--blackbox-seq-q-tiles",
            "1",
        ])
        .unwrap();
        let sizes = resolve_query_chunk_sizes(&exact).unwrap();
        assert!(is_exact_1k5_release_policy(&exact, sizes));
        validate_1k5_release_policy(&exact, BooguVariant::Image01EditTurbo1k5, sizes).unwrap();
        let report = serde_json::to_value(policy_report(
            &exact,
            BooguVariant::Image01EditTurbo1k5,
            sizes,
        ))
        .unwrap();
        assert_eq!(
            report["edit_1k5_release_policy_validated"],
            serde_json::json!(true)
        );
        assert_eq!(
            report["native_release_policy_validated"],
            serde_json::json!(true)
        );

        let diagnostic = Args::try_parse_from([
            "boogu-full-parity",
            "--artifacts",
            "artifacts",
            "--fixture",
            "fixture",
            "--allow-unvalidated-1k5-policy",
        ])
        .unwrap();
        let sizes = resolve_query_chunk_sizes(&diagnostic).unwrap();
        assert!(!is_exact_1k5_release_policy(&diagnostic, sizes));
        validate_1k5_release_policy(&diagnostic, BooguVariant::Image01EditTurbo1k5, sizes).unwrap();
        let report = serde_json::to_value(policy_report(
            &diagnostic,
            BooguVariant::Image01EditTurbo1k5,
            sizes,
        ))
        .unwrap();
        assert_eq!(
            report["edit_1k5_release_policy_validated"],
            serde_json::json!(false)
        );
        assert!(
            validate_1k5_release_policy(
                &Args {
                    allow_unvalidated_1k5_policy: false,
                    ..diagnostic
                },
                BooguVariant::Image01EditTurbo1k5,
                sizes,
            )
            .is_err()
        );
    }

    #[test]
    fn component_query_chunk_controls_are_independent_and_legacy_compatible_correctness() {
        let parse = |extra: &[&str]| {
            let mut argv = vec![
                "boogu-full-parity",
                "--artifacts",
                "artifacts",
                "--fixture",
                "fixture",
            ];
            argv.extend_from_slice(extra);
            Args::try_parse_from(argv).unwrap()
        };

        assert_eq!(
            resolve_query_chunk_sizes(&parse(&[])).unwrap(),
            QueryChunkSizes {
                qwen: DEFAULT_QUERY_CHUNK_SIZE,
                denoiser: DEFAULT_QUERY_CHUNK_SIZE,
            }
        );
        assert_eq!(
            resolve_query_chunk_sizes(&parse(&[
                "--qwen-query-chunk-size",
                "64",
                "--denoiser-query-chunk-size",
                "16384",
            ]))
            .unwrap(),
            QueryChunkSizes {
                qwen: 64,
                denoiser: 16_384,
            }
        );
        assert_eq!(
            resolve_query_chunk_sizes(&parse(&["--query-chunk-size", "8192"])).unwrap(),
            QueryChunkSizes {
                qwen: 8192,
                denoiser: 8192,
            }
        );
        assert!(
            resolve_query_chunk_sizes(&parse(&[
                "--query-chunk-size",
                "8192",
                "--qwen-query-chunk-size",
                "128",
            ]))
            .unwrap_err()
            .contains("cannot be combined")
        );
        assert!(
            resolve_query_chunk_sizes(&parse(&["--qwen-query-chunk-size", "0"]))
                .unwrap_err()
                .contains("qwen-query")
        );
        assert!(
            resolve_query_chunk_sizes(&parse(&["--denoiser-query-chunk-size", "0"]))
                .unwrap_err()
                .contains("denoiser-query")
        );
        assert!(
            resolve_query_chunk_sizes(&parse(&["--query-chunk-size", "0"]))
                .unwrap_err()
                .contains("--query-chunk-size")
        );
    }

    #[test]
    fn vae_float_policy_cli_controls_load_execution_and_gate_evidence_correctness() {
        let default = Args::try_parse_from([
            "boogu-full-parity",
            "--artifacts",
            "artifacts",
            "--fixture",
            "fixture",
        ])
        .unwrap();
        assert_eq!(default.vae_float_policy, VaeFloatPolicyChoice::ForceF32);
        assert_eq!(
            default.vae_float_policy.load_policy(),
            BooguFloatLoadPolicy::AdaptToF32
        );

        let preserve = Args::try_parse_from([
            "boogu-full-parity",
            "--artifacts",
            "artifacts",
            "--fixture",
            "fixture",
            "--vae-float-policy",
            "preserve-f16",
        ])
        .unwrap();
        assert_eq!(preserve.vae_float_policy, VaeFloatPolicyChoice::PreserveF16);
        assert_eq!(
            preserve.vae_float_policy.load_policy(),
            BooguFloatLoadPolicy::Preserve
        );
        assert_eq!(
            BooguRuntimeDTypes::from_artifact_policies(
                BooguStorageProfile::F16QwenVisionF32,
                preserve.vae_float_policy.load_policy(),
                BooguFloatLoadPolicy::Preserve,
            )
            .vae,
            DType::F16
        );

        let gates = gate_set(
            BooguVariant::Image01Turbo,
            ProfileChoice::F16QwenVisionF32,
            preserve.vae_float_policy,
        )
        .unwrap();
        assert_eq!(gates.vae_float_policy, "preserve-f16");
        assert!(gates.evidence.contains("VAE policy preserve-f16"));

        let edit_gates = gate_set(
            BooguVariant::Image01EditTurbo,
            ProfileChoice::F16QwenVisionF32,
            preserve.vae_float_policy,
        )
        .unwrap();
        assert_eq!(edit_gates.edit_reference.maximum_abs, 0.82);
        assert_eq!(edit_gates.edit_reference.maximum_rmse, 0.080);
        assert_eq!(
            edit_reference_oracles(preserve.vae_float_policy),
            (
                "vae.reference_scaled_latent",
                "vae.reference_f32_scaled_latent"
            )
        );
    }

    #[test]
    fn optimized_native_policy_cli_is_explicit_and_bounded_correctness() {
        let args = Args::try_parse_from([
            "boogu-full-parity",
            "--artifacts",
            "artifacts",
            "--fixture",
            "fixture",
            "--vae-float-policy",
            "preserve-f16",
            "--vae-group-norm-policy",
            "f16-storage-f32-accum",
            "--vae-attention-query-chunk-size",
            "1024",
            "--denoiser-attention-policy",
            "padded-blackbox",
            "--blackbox-num-planes",
            "2",
            "--blackbox-seq-kv-tiles",
            "2",
        ])
        .unwrap();

        assert_eq!(
            args.vae_group_norm_policy.execution_policy(),
            DecoderGroupNormPolicy::F16StorageF32Accum
        );
        assert_eq!(args.vae_attention_query_chunk_size, 1024);
        assert_eq!(args.blackbox_num_planes, 2);
        assert_eq!(args.blackbox_seq_kv_tiles, 2);
        assert_eq!(args.blackbox_seq_q_tiles, 1);
    }

    #[test]
    fn right_padding_is_strict_correctness() {
        assert_eq!(right_padded_length(&[true, true, false, false]).unwrap(), 2);
        assert!(right_padded_length(&[false, false]).is_err());
        assert!(right_padded_length(&[true, false, true]).is_err());
    }

    #[test]
    fn fixture_sigma_oracle_preserves_expected_bf16_dtype_drift_reference() {
        let fixture = fixture_sigma_oracle(BooguTask::Generate, "bf16").unwrap();
        let production = DmdSchedule::upstream_for_dtype(BooguTask::Generate, DType::F16);
        assert_eq!(fixture[1], 0.251_953_13);
        assert!((production.sigmas()[1] - fixture[1]).abs() > 0.001);
        assert!((production.sigmas()[1] - fixture[1]).abs() < 0.0013);
    }
}
