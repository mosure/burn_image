//! Shared native verified-artifact Boogu inference runner.

use std::{error::Error, fs, path::PathBuf, sync::Arc, time::Instant};

use burn_boogu::{
    BooguDenoiser, BooguImageModel, BooguRuntimeDTypes, BooguRuntimeMetadata, BooguVariant,
    DenoiserRmsNormPolicy, DmdDenoiser, NativeDenoiserQkPreparationPolicy, NativeHighVramPolicy,
    RetainingBooguVaeStageSource, StreamingBooguPipeline,
    artifacts::{
        BooguArtifactInventory, BooguFloatLoadPolicy, BooguQuantizedLoadPolicy,
        BooguReleaseIdentity, BooguStorageProfile, VerifiedArtifactDirectory,
        VerifiedBurnpackQwenStageSource, VerifiedDirectoryVaeStageSource,
        load_resident_denoiser_from_directory_with_policies,
    },
    boogu_model_descriptor, boogu_processor_config,
};
use burn_flux_vae::{AutoencoderKlConfig, DecoderGroupNormPolicy};
use burn_image::{
    ArtifactCachePolicy, ArtifactProfileId, ArtifactSource, ColorSpace, Dimensions, EditRequest,
    EncodedImage, GenerateRequest, GenerationOptions, HostImage, ImageEncoding, ImageRequest,
    ImageRuntime, InputImage, IntegrityPolicy, NumericFormat, PixelFormat, ProgressEvent, Prompt,
    RuntimeConfig,
};
use burn_qwen3_vl::{
    Qwen3VlConfig, Qwen3VlImageProcessor, Qwen3VlImageProcessorConfig, Qwen3VlProcessor,
    Qwen3VlTokenizer, RetainingQwen3VlStageSource, RetainingSynchronizationPolicy,
    StreamingQwen3Vl, tokenizer::HfTokenizer,
};
use clap::{Parser, ValueEnum};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Variant {
    Turbo,
    EditTurbo,
    #[value(name = "edit-turbo-1k5", alias = "edit-turbo1k5")]
    EditTurbo1k5,
}

impl From<Variant> for BooguVariant {
    fn from(value: Variant) -> Self {
        match value {
            Variant::Turbo => Self::Image01Turbo,
            Variant::EditTurbo => Self::Image01EditTurbo,
            Variant::EditTurbo1k5 => Self::Image01EditTurbo1k5,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Profile {
    F16,
    F16QwenVisionF32,
    Q8sBlock32F32,
    Q8sBlock32F32QwenVisionF32,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum VaeFloatPolicy {
    ForceF32,
    PreserveF16,
}

impl VaeFloatPolicy {
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

#[derive(Debug, Clone, Copy, ValueEnum)]
enum VaeGroupNormPolicy {
    StrictF32,
    F16StorageF32Accum,
}

impl VaeGroupNormPolicy {
    const fn execution_policy(self) -> DecoderGroupNormPolicy {
        match self {
            Self::StrictF32 => DecoderGroupNormPolicy::StrictF32,
            Self::F16StorageF32Accum => DecoderGroupNormPolicy::F16StorageF32Accum,
        }
    }

    const fn id(self) -> &'static str {
        match self {
            Self::StrictF32 => "strict-f32",
            Self::F16StorageF32Accum => "f16-storage-f32-accum",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum DenoiserRmsNormPolicyChoice {
    StrictF32,
    F16StorageF32Accum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum DenoiserQkPreparationPolicyChoice {
    /// Preserve the parity-gated sequence of RMSNorm, RoPE, and padded-GQA preparation ops.
    Composed,
    /// Fuse strict-F32 Q/K RMSNorm, RoPE, scaling, GQA expansion, and padding into one dispatch.
    FusedStrictQkNormRope,
    /// Preserve strict RMSNorm and fuse only RoPE, scaling, GQA expansion, and padding.
    FusedRopeGqaPadding,
    /// Normalize Q and K in separate balanced native kernels before fused RoPE/GQA preparation.
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

    const fn id(self) -> &'static str {
        match self {
            Self::Composed => "composed",
            Self::FusedStrictQkNormRope => "fused-strict-qk-norm-rope",
            Self::FusedRopeGqaPadding => "fused-rope-gqa-padding",
            Self::BalancedStrictQkNormRope => "balanced-strict-qk-norm-rope",
        }
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

    const fn id(self) -> &'static str {
        match self {
            Self::StrictF32 => "strict-f32",
            Self::F16StorageF32Accum => "f16-storage-f32-accum",
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ConditioningCachePolicy {
    /// Retain one conditioning tensor keyed by the exact prompt, source image, and sequence length.
    ExactRetained,
    /// Clear conditioning immediately before every inference while retaining loaded model stages.
    Disabled,
}

impl ConditioningCachePolicy {
    const fn id(self) -> &'static str {
        match self {
            Self::ExactRetained => "exact-retained",
            Self::Disabled => "disabled",
        }
    }

    const fn residency(self, iteration: u16) -> &'static str {
        if iteration == 0 {
            "cold-policy"
        } else {
            match self {
                Self::ExactRetained => "warm-retained",
                Self::Disabled => "warm-model-uncached-conditioning",
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum QwenSynchronizationPolicyChoice {
    /// Preserve one backend synchronization after every streamed semantic stage.
    PerStage,
    /// Submit retained stages continuously and rely on the runtime's terminal Qwen-stage barrier.
    Deferred,
}

impl QwenSynchronizationPolicyChoice {
    const fn execution_policy(self) -> RetainingSynchronizationPolicy {
        match self {
            Self::PerStage => RetainingSynchronizationPolicy::PerStage,
            Self::Deferred => RetainingSynchronizationPolicy::Deferred,
        }
    }

    const fn id(self) -> &'static str {
        match self {
            Self::PerStage => "per-stage",
            Self::Deferred => "deferred",
        }
    }
}

impl Profile {
    const fn is_q8(self) -> bool {
        matches!(self, Self::Q8sBlock32F32 | Self::Q8sBlock32F32QwenVisionF32)
    }

    const fn storage(self) -> BooguStorageProfile {
        match self {
            Self::F16 => BooguStorageProfile::F16,
            Self::F16QwenVisionF32 => BooguStorageProfile::F16QwenVisionF32,
            Self::Q8sBlock32F32 => BooguStorageProfile::Q8sBlock32F32,
            Self::Q8sBlock32F32QwenVisionF32 => BooguStorageProfile::Q8sBlock32F32QwenVisionF32,
        }
    }

    fn numeric(self) -> NumericFormat {
        match self {
            Self::F16 => NumericFormat::F16,
            Self::F16QwenVisionF32 => NumericFormat::Other("f16-qwen-vision-f32".into()),
            Self::Q8sBlock32F32 => NumericFormat::Other("q8s-block32-f32".into()),
            Self::Q8sBlock32F32QwenVisionF32 => {
                NumericFormat::Other("q8s-block32-f32-qwen-vision-f32".into())
            }
        }
    }

    const fn id(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::F16QwenVisionF32 => "f16-qwen-vision-f32",
            Self::Q8sBlock32F32 => "q8s-block32-f32",
            Self::Q8sBlock32F32QwenVisionF32 => "q8s-block32-f32-qwen-vision-f32",
        }
    }
}

#[derive(Debug, Parser)]
#[command(about = "Run a sealed Boogu-Image artifact bundle with a native Burn backend")]
struct Args {
    /// Directory containing manifest.json, metadata/source, and content-addressed Burnpacks.
    #[arg(long)]
    artifacts: PathBuf,
    /// Immutable model release represented by the bundle.
    #[arg(long, value_enum)]
    variant: Variant,
    /// Artifact numeric profile.
    #[arg(long, value_enum, default_value = "f16-qwen-vision-f32")]
    profile: Profile,
    /// VAE execution policy. `force-f32` matches Diffusers `force_upcast`; `preserve-f16` is an
    /// explicit native performance policy that must satisfy the fixture-backed parity gate.
    #[arg(long, value_enum, default_value = "force-f32")]
    vae_float_policy: VaeFloatPolicy,
    /// Decoder-only GroupNorm execution policy. The opt-in policy keeps F16 activation storage
    /// while Burn accumulates GroupNorm reductions in F32; the encoder always remains strict F32.
    #[arg(long, value_enum, default_value = "strict-f32")]
    vae_group_norm_policy: VaeGroupNormPolicy,
    /// Denoiser RMSNorm policy. The mixed-storage policy is a diagnostic native-padded-blackbox
    /// experiment until it passes pinned full-chain parity and synchronized performance gates.
    #[arg(long, value_enum, default_value = "strict-f32")]
    denoiser_rms_norm_policy: DenoiserRmsNormPolicyChoice,
    /// Q/K preparation policy. Balanced strict Q/K normalization is qualified only for the exact
    /// native 1K release policy; the other non-composed choices remain diagnostic.
    #[arg(long, value_enum, default_value = "composed")]
    denoiser_qk_preparation_policy: DenoiserQkPreparationPolicyChoice,
    /// Apply the shared dual-stream output projection independently to each token stream.
    #[arg(long, default_value_t = false)]
    denoiser_split_double_stream_shared_projection: bool,
    /// Generation prompt or edit instruction.
    #[arg(long)]
    prompt: String,
    /// Required source image for Edit-Turbo; forbidden for Turbo.
    #[arg(long)]
    source: Option<PathBuf>,
    /// PNG output path.
    #[arg(long, default_value = "boogu-output.png")]
    output: PathBuf,
    /// Output width. Defaults to the selected release's canonical 1024 or 1536 edge.
    #[arg(long)]
    width: Option<u32>,
    /// Output height. Defaults to the selected release's canonical 1024 or 1536 edge.
    #[arg(long)]
    height: Option<u32>,
    /// Deterministic Burn runtime seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Repeat inference for each selected denoiser query tile. Iteration 0 is cold; later
    /// iterations reuse retained Qwen/VAE stages and the resident denoiser. Conditioning reuse is
    /// controlled independently by `--conditioning-cache-policy`.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u16).range(1..))]
    repeat: u16,
    /// Conditioning reuse policy for repeated measurements. `exact-retained` reuses an exact
    /// prompt/source/length match; `disabled` clears only conditioning before every inference so
    /// Qwen executes again against an otherwise warm resident model.
    #[arg(long, value_enum, default_value = "exact-retained")]
    conditioning_cache_policy: ConditioningCachePolicy,
    /// Retained-Qwen synchronization policy. `deferred` removes intermediate device drains and
    /// relies on the runtime's mandatory synchronization at the Qwen stage boundary.
    #[arg(long, value_enum, default_value = "per-stage")]
    qwen_synchronization_policy: QwenSynchronizationPolicyChoice,
    /// Maximum query rows per streamed Qwen attention submission. This is independent from the
    /// denoiser tile so performance experiments cannot silently change the conditioning oracle.
    #[arg(long, default_value_t = 128)]
    qwen_query_chunk_size: usize,
    /// Maximum query rows per denoiser attention submission. The conservative portable default
    /// is 128; larger native-WGPU tiles reduce dispatch overhead at a higher fallback-memory bound.
    #[arg(long, default_value_t = 128)]
    denoiser_query_chunk_size: usize,
    /// Maximum query rows per VAE middle-block attention submission. Larger values preserve the
    /// exact dense attention math while trading a larger bounded score buffer for fewer launches.
    #[arg(long, default_value_t = 512)]
    vae_attention_query_chunk_size: usize,
    /// Optional comma-separated native benchmark sweep. Each tile runs `--repeat` times in the
    /// same retained process; this avoids attributing artifact loading to an attention policy.
    #[arg(long, value_delimiter = ',')]
    benchmark_denoiser_query_chunk_sizes: Vec<usize>,
    /// Plane count for forced padded-blackbox attention. Two planes may use one or two key/value
    /// tiles; four planes use one tile.
    #[arg(long, default_value_t = 4)]
    blackbox_num_planes: u8,
    /// Number of 16-row key/value tiles per forced padded-blackbox partition. This is accepted by
    /// padded-blackbox runners only; 1 preserves the original blueprint.
    #[arg(long, default_value_t = 1)]
    blackbox_seq_kv_tiles: u8,
    /// Number of 16-row query tiles retained per plane. Only 1 is supported; the q2 blueprint
    /// failed the native-WGPU nonzero parity gate.
    #[arg(long, default_value_t = 1)]
    blackbox_seq_q_tiles: u8,
    /// Optional comma-separated padded-blackbox partition sweep. Supported values are 1 and 2;
    /// every value is benchmarked with every selected denoiser query chunk in one process.
    #[arg(long, value_delimiter = ',')]
    benchmark_blackbox_seq_kv_tiles: Vec<u8>,
    /// Permit a diagnostic 1.5K policy other than the released full-autotune p4/kv1/q1 mixed-F16
    /// configuration. Such a run is reported as experimental and is not support evidence.
    #[arg(long, default_value_t = false)]
    allow_unvalidated_1k5_policy: bool,
}

pub(crate) trait RunnerDenoiser<B: burn::prelude::Backend>: DmdDenoiser<B> {
    fn set_query_chunk_size(&mut self, query_chunk_size: usize);

    fn validate_rms_norm_policy(policy: DenoiserRmsNormPolicy) -> Result<(), &'static str> {
        if policy == DenoiserRmsNormPolicy::StrictF32 {
            Ok(())
        } else {
            Err("selected denoiser does not implement mixed-storage RMSNorm")
        }
    }

    fn set_rms_norm_policy(&mut self, policy: DenoiserRmsNormPolicy) -> Result<(), &'static str> {
        Self::validate_rms_norm_policy(policy)
    }

    fn validate_qk_preparation_policy(
        policy: DenoiserQkPreparationPolicyChoice,
    ) -> Result<(), &'static str> {
        if policy == DenoiserQkPreparationPolicyChoice::Composed {
            Ok(())
        } else {
            Err("selected denoiser does not implement fused Q/K preparation")
        }
    }

    fn set_qk_preparation_policy(
        &mut self,
        policy: DenoiserQkPreparationPolicyChoice,
    ) -> Result<(), &'static str> {
        Self::validate_qk_preparation_policy(policy)
    }

    fn validate_split_double_stream_shared_projection(enabled: bool) -> Result<(), &'static str> {
        if enabled {
            Err("selected denoiser does not implement split shared projection")
        } else {
            Ok(())
        }
    }

    fn set_split_double_stream_shared_projection(
        &mut self,
        enabled: bool,
    ) -> Result<(), &'static str> {
        Self::validate_split_double_stream_shared_projection(enabled)
    }

    fn validate_blackbox_configuration(
        num_planes: u8,
        seq_kv_tiles: u8,
        seq_q_tiles: u8,
    ) -> Result<(), &'static str> {
        validate_blackbox_configuration(num_planes, seq_kv_tiles, seq_q_tiles)
    }

    fn set_blackbox_configuration(
        &mut self,
        num_planes: u8,
        seq_kv_tiles: u8,
        seq_q_tiles: u8,
    ) -> Result<(), &'static str> {
        Self::validate_blackbox_configuration(num_planes, seq_kv_tiles, seq_q_tiles)?;
        if (num_planes, seq_kv_tiles, seq_q_tiles) == (4, 1, 1) {
            Ok(())
        } else {
            Err("selected denoiser does not implement padded-blackbox configuration tuning")
        }
    }
}

impl<B: burn::prelude::Backend> RunnerDenoiser<B> for BooguDenoiser<B> {
    fn set_query_chunk_size(&mut self, query_chunk_size: usize) {
        self.set_attention_query_chunk_size(query_chunk_size);
    }
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
impl RunnerDenoiser<burn_boogu::NativeWgpuBackend> for burn_boogu::NativeFlashUnitDenoiser {
    fn set_query_chunk_size(&mut self, query_chunk_size: usize) {
        self.set_attention_query_chunk_size(query_chunk_size);
    }
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
impl RunnerDenoiser<burn_boogu::NativeWgpuBackend> for burn_boogu::NativePaddedBlackboxDenoiser {
    fn set_query_chunk_size(&mut self, query_chunk_size: usize) {
        self.set_attention_query_chunk_size(query_chunk_size);
    }

    fn set_blackbox_configuration(
        &mut self,
        num_planes: u8,
        seq_kv_tiles: u8,
        seq_q_tiles: u8,
    ) -> Result<(), &'static str> {
        Self::validate_blackbox_configuration(num_planes, seq_kv_tiles, seq_q_tiles)?;
        self.set_partition_configuration(num_planes, seq_kv_tiles, seq_q_tiles);
        Ok(())
    }

    fn validate_rms_norm_policy(_policy: DenoiserRmsNormPolicy) -> Result<(), &'static str> {
        Ok(())
    }

    fn set_rms_norm_policy(&mut self, policy: DenoiserRmsNormPolicy) -> Result<(), &'static str> {
        burn_boogu::NativePaddedBlackboxDenoiser::set_rms_norm_policy(self, policy);
        Ok(())
    }

    fn validate_qk_preparation_policy(
        _policy: DenoiserQkPreparationPolicyChoice,
    ) -> Result<(), &'static str> {
        Ok(())
    }

    fn set_qk_preparation_policy(
        &mut self,
        policy: DenoiserQkPreparationPolicyChoice,
    ) -> Result<(), &'static str> {
        self.set_fused_strict_qk_norm_rope(policy.fused_strict_qk_norm_rope());
        self.set_fused_rope_gqa_padding(policy.fused_rope_gqa_padding());
        self.set_balanced_strict_qk_norm_rope(policy.balanced_strict_qk_norm_rope());
        Ok(())
    }

    fn validate_split_double_stream_shared_projection(_enabled: bool) -> Result<(), &'static str> {
        Ok(())
    }

    fn set_split_double_stream_shared_projection(
        &mut self,
        enabled: bool,
    ) -> Result<(), &'static str> {
        burn_boogu::NativePaddedBlackboxDenoiser::set_split_double_stream_shared_projection(
            self, enabled,
        );
        Ok(())
    }
}

#[cfg(all(feature = "cuda-experimental", not(target_arch = "wasm32")))]
impl RunnerDenoiser<burn_boogu::NativeCudaBackend>
    for burn_boogu::NativeCudaPaddedBlackboxDenoiser
{
    fn set_query_chunk_size(&mut self, query_chunk_size: usize) {
        self.set_attention_query_chunk_size(query_chunk_size);
    }

    fn set_blackbox_configuration(
        &mut self,
        num_planes: u8,
        seq_kv_tiles: u8,
        seq_q_tiles: u8,
    ) -> Result<(), &'static str> {
        validate_blackbox_configuration(num_planes, seq_kv_tiles, seq_q_tiles)?;
        if seq_q_tiles != 1 {
            return Err("selected CUDA denoiser has not enabled query-partition tuning");
        }
        self.set_configuration(num_planes, seq_kv_tiles);
        Ok(())
    }
}

fn validate_blackbox_seq_kv_tiles(seq_kv_tiles: u8) -> Result<(), &'static str> {
    if matches!(seq_kv_tiles, 1 | 2) {
        Ok(())
    } else {
        Err("blackbox seq-kv tiles must be one of 1 or 2")
    }
}

fn validate_blackbox_seq_q_tiles(seq_q_tiles: u8) -> Result<(), &'static str> {
    if seq_q_tiles == 1 {
        Ok(())
    } else {
        Err("blackbox seq-q tiles must be 1; q2 failed the native WGPU nonzero parity gate")
    }
}

fn validate_blackbox_configuration(
    num_planes: u8,
    seq_kv_tiles: u8,
    seq_q_tiles: u8,
) -> Result<(), &'static str> {
    if !matches!(num_planes, 2 | 4) {
        return Err("blackbox plane count must be one of 2 or 4");
    }
    validate_blackbox_seq_kv_tiles(seq_kv_tiles)?;
    validate_blackbox_seq_q_tiles(seq_q_tiles)?;
    if num_planes == 4 && seq_kv_tiles == 2 {
        return Err("four-plane/multi-KV-tile configuration failed the native WGPU parity gate");
    }
    Ok(())
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
    denoiser_attention_policy: &str,
) -> bool {
    let policy = native_release_policy(variant);
    denoiser_attention_policy == "required-padded-blackbox-p4-kv1-q1-full-autotune"
        && matches!(args.profile, Profile::F16QwenVisionF32)
        && args.denoiser_rms_norm_policy == DenoiserRmsNormPolicyChoice::StrictF32
        && args
            .denoiser_qk_preparation_policy
            .matches_native_policy(policy.denoiser_qk_preparation)
        && !args.denoiser_split_double_stream_shared_projection
        && args.qwen_synchronization_policy == QwenSynchronizationPolicyChoice::Deferred
        && matches!(args.vae_float_policy, VaeFloatPolicy::PreserveF16)
        && matches!(
            args.vae_group_norm_policy,
            VaeGroupNormPolicy::F16StorageF32Accum
        )
        && args.qwen_query_chunk_size == policy.qwen_query_chunk_size
        && args.denoiser_query_chunk_size == policy.denoiser_query_chunk_size
        && args.vae_attention_query_chunk_size == policy.vae_attention_query_chunk_size
        && args.blackbox_num_planes == policy.blackbox_num_planes
        && args.blackbox_seq_kv_tiles == policy.blackbox_seq_kv_tiles
        && args.blackbox_seq_q_tiles == policy.blackbox_seq_q_tiles
        && args.benchmark_denoiser_query_chunk_sizes.is_empty()
        && args.benchmark_blackbox_seq_kv_tiles.is_empty()
}

pub(crate) fn run<B, D>(
    backend: &'static str,
    create_device: impl FnOnce() -> Result<B::Device, Box<dyn Error>>,
    wrap_denoiser: impl FnOnce(BooguDenoiser<B>) -> D,
    denoiser_attention_policy: &'static str,
) -> Result<(), Box<dyn Error>>
where
    B: burn::prelude::Backend,
    D: RunnerDenoiser<B>,
{
    let process_started = Instant::now();
    let args = Args::parse();
    if args.denoiser_query_chunk_size == 0 {
        return Err("--denoiser-query-chunk-size must be non-zero".into());
    }
    if args.qwen_query_chunk_size == 0 {
        return Err("--qwen-query-chunk-size must be non-zero".into());
    }
    if args.vae_attention_query_chunk_size == 0 {
        return Err("--vae-attention-query-chunk-size must be non-zero".into());
    }
    if args.benchmark_denoiser_query_chunk_sizes.contains(&0) {
        return Err("--benchmark-denoiser-query-chunk-sizes must contain non-zero values".into());
    }
    if matches!(
        args.vae_group_norm_policy,
        VaeGroupNormPolicy::F16StorageF32Accum
    ) && !matches!(args.vae_float_policy, VaeFloatPolicy::PreserveF16)
    {
        return Err(
            "--vae-group-norm-policy f16-storage-f32-accum requires --vae-float-policy preserve-f16"
                .into(),
        );
    }
    D::validate_rms_norm_policy(args.denoiser_rms_norm_policy.execution_policy())
        .map_err(|reason| format!("invalid --denoiser-rms-norm-policy: {reason}"))?;
    D::validate_qk_preparation_policy(args.denoiser_qk_preparation_policy)
        .map_err(|reason| format!("invalid --denoiser-qk-preparation-policy: {reason}"))?;
    D::validate_split_double_stream_shared_projection(
        args.denoiser_split_double_stream_shared_projection,
    )
    .map_err(|reason| {
        format!("invalid --denoiser-split-double-stream-shared-projection: {reason}")
    })?;
    D::validate_blackbox_configuration(
        args.blackbox_num_planes,
        args.blackbox_seq_kv_tiles,
        args.blackbox_seq_q_tiles,
    )
    .map_err(|reason| format!("invalid padded-blackbox configuration: {reason}"))?;
    for &seq_kv_tiles in &args.benchmark_blackbox_seq_kv_tiles {
        D::validate_blackbox_configuration(
            args.blackbox_num_planes,
            seq_kv_tiles,
            args.blackbox_seq_q_tiles,
        )
        .map_err(|reason| {
            format!("invalid --benchmark-blackbox-seq-kv-tiles value {seq_kv_tiles}: {reason}")
        })?;
    }
    if args.denoiser_qk_preparation_policy != DenoiserQkPreparationPolicyChoice::Composed
        && (args.denoiser_rms_norm_policy != DenoiserRmsNormPolicyChoice::StrictF32
            || (
                args.blackbox_num_planes,
                args.blackbox_seq_kv_tiles,
                args.blackbox_seq_q_tiles,
            ) != (4, 1, 1)
            || !args.benchmark_blackbox_seq_kv_tiles.is_empty())
    {
        return Err(
            "non-composed --denoiser-qk-preparation-policy values require strict-f32 RMSNorm, \
             p4/kv1/q1, and no partition sweep"
                .into(),
        );
    }
    let variant: BooguVariant = args.variant.into();
    match (variant, &args.source) {
        (BooguVariant::Image01Turbo, Some(_)) => {
            return Err("--source is forbidden for Turbo".into());
        }
        (BooguVariant::Image01EditTurbo | BooguVariant::Image01EditTurbo1k5, None) => {
            return Err("--source is required for Edit-Turbo".into());
        }
        _ => {}
    }

    // Reject unsupported shapes before hashing or loading tens of gigabytes of artifacts.
    let descriptor = boogu_model_descriptor(variant);
    let requested_numeric_format = args.profile.numeric();
    if !descriptor
        .capabilities
        .numeric_formats
        .contains(&requested_numeric_format)
    {
        return Err(format!(
            "profile {} is not supported by selected release {:?}",
            args.profile.id(),
            variant
        )
        .into());
    }
    let default_edge = if variant == BooguVariant::Image01EditTurbo1k5 {
        burn_boogu::processing::BOOGU_1K5_DEFAULT_EDGE
    } else {
        burn_boogu::processing::BOOGU_DEFAULT_EDGE
    };
    let dimensions = Dimensions::new(
        args.width.unwrap_or(default_edge),
        args.height.unwrap_or(default_edge),
    )?;
    if let Err(reason) = descriptor.capabilities.dimensions.supports(dimensions) {
        return Err(format!("unsupported output dimensions {dimensions:?}: {reason}").into());
    }
    if variant == BooguVariant::Image01EditTurbo1k5
        && !burn_boogu::BOOGU_1K5_OUTPUT_PRESETS
            .contains(&(dimensions.width(), dimensions.height()))
    {
        return Err(format!(
            "Edit-Turbo 1.5K output {}x{} is not an official released preset",
            dimensions.width(),
            dimensions.height()
        )
        .into());
    }
    let release_policy_validated =
        is_exact_native_release_policy(&args, variant, denoiser_attention_policy)
            && !(variant == BooguVariant::Image01EditTurbo1k5 && args.allow_unvalidated_1k5_policy);
    if variant == BooguVariant::Image01EditTurbo1k5
        && !args.allow_unvalidated_1k5_policy
        && !release_policy_validated
    {
        return Err(
            "Edit-Turbo 1.5K requires the released full-autotune mixed-F16 policy: \
                 deferred-sync qwen-q128, denoiser padded-blackbox p4/kv1/q1 q16384 with strict-f32 RMSNorm, \
                 composed Q/K preparation, and VAE q4096 with preserve-f16/f16-storage-f32-accum; pass \
                 --allow-unvalidated-1k5-policy only for explicitly diagnostic runs"
                .into(),
        );
    }

    let artifact_directory = VerifiedArtifactDirectory::open(&args.artifacts)?;
    let manifest = artifact_directory.manifest();
    let identity = BooguReleaseIdentity::canonical(variant);
    if manifest.model_revision != identity.model_revision {
        return Err(format!(
            "bundle revision {} does not match selected release {}",
            manifest.model_revision, identity.model_revision
        )
        .into());
    }
    if variant == BooguVariant::Image01EditTurbo1k5 && !args.allow_unvalidated_1k5_policy {
        let content_digest = manifest
            .content_digest
            .ok_or("sealed Edit-Turbo 1.5K manifest has no content digest")?;
        burn_boogu::artifacts::validate_edit_turbo_1k5_release_artifact_digest(content_digest)?;
    }
    let qwen_config = Qwen3VlConfig::from_json(
        &artifact_directory.read_text("metadata/source/mllm/config.json")?,
    )?;
    let mut vae_config = AutoencoderKlConfig::from_diffusers_json(
        &artifact_directory.read_text("metadata/source/vae/config.json")?,
    )?;
    vae_config.attention_query_chunk_size = args.vae_attention_query_chunk_size;
    let denoiser_config = burn_boogu::BooguConfig::default();
    let inventory = BooguArtifactInventory::new(&qwen_config, &denoiser_config, &vae_config)?;
    let device = create_device()?;
    let profile = args.profile.storage();
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

    eprintln!("verifying streamed Qwen3-VL stages");
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
    eprintln!(
        "verified Qwen: {} streamed stages, {} embedding row chunks",
        qwen_plan.stages.len(),
        qwen_plan.embedding_rows.chunks.len()
    );
    let qwen_source = RetainingQwen3VlStageSource::new(qwen_source)
        .with_synchronization_policy(args.qwen_synchronization_policy.execution_policy());
    let mut qwen = StreamingQwen3Vl::new(qwen_plan, qwen_source);
    qwen.set_query_chunk_size(args.qwen_query_chunk_size);

    eprintln!("verifying independently staged FLUX VAE encoder and decoder");
    let vae = VerifiedDirectoryVaeStageSource::<B>::new(
        &identity,
        &args.artifacts,
        inventory.clone(),
        vae_config,
        profile,
        vae_policy,
        device.clone(),
    )?;
    let vae = RetainingBooguVaeStageSource::new(vae);
    eprintln!(
        "verified VAE stages; each half will load once and retain shared device handles for repeats"
    );

    eprintln!("loading verified Boogu denoiser");
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
    denoiser.set_attention_query_chunk_size(args.denoiser_query_chunk_size);
    eprintln!(
        "loaded denoiser: {} tensors in {} shards",
        denoiser_report.tensors, denoiser_report.shards
    );

    let tokenizer = HfTokenizer::from_bytes(
        &artifact_directory.read_file("metadata/source/mllm/tokenizer.json")?,
    )?;
    let pad_token_id = tokenizer
        .token_to_id("<|endoftext|>")
        .ok_or("Qwen tokenizer is missing <|endoftext|>")?;
    let processor = Qwen3VlProcessor::new(
        tokenizer,
        boogu_processor_config(&qwen_config, pad_token_id),
    )?;
    let image_processor = Qwen3VlImageProcessor::new(Qwen3VlImageProcessorConfig::from_json(
        &artifact_directory.read_text("metadata/source/mllm/preprocessor_config.json")?,
    )?)?;
    let mut denoiser = wrap_denoiser(denoiser);
    denoiser
        .set_rms_norm_policy(args.denoiser_rms_norm_policy.execution_policy())
        .map_err(|reason| format!("cannot apply --denoiser-rms-norm-policy: {reason}"))?;
    denoiser
        .set_qk_preparation_policy(args.denoiser_qk_preparation_policy)
        .map_err(|reason| format!("cannot apply --denoiser-qk-preparation-policy: {reason}"))?;
    denoiser
        .set_split_double_stream_shared_projection(
            args.denoiser_split_double_stream_shared_projection,
        )
        .map_err(|reason| {
            format!("cannot apply --denoiser-split-double-stream-shared-projection: {reason}")
        })?;
    let pipeline = StreamingBooguPipeline::new(variant, qwen_config.clone(), qwen, vae, denoiser)
        .with_decoder_group_norm_policy(args.vae_group_norm_policy.execution_policy());
    let model = BooguImageModel::new(
        pipeline,
        processor,
        image_processor,
        device,
        BooguRuntimeMetadata {
            numeric_format: requested_numeric_format,
            backend: backend.into(),
            artifact_content_digest: manifest.content_digest,
            artifacts_verified: true,
            execution_dtypes,
            default_seed: args.seed,
        },
    )?;
    let mut runtime = ImageRuntime::new(
        RuntimeConfig {
            model: descriptor.id,
            artifact_profile: ArtifactProfileId::new(args.profile.id())?,
            artifact_source: ArtifactSource::LocalDirectory {
                root: args.artifacts.clone(),
            },
            integrity: IntegrityPolicy::RequireSha256,
            cache: ArtifactCachePolicy::UseCached,
        },
        model,
    )?
    .with_observer(Arc::new(|event: &ProgressEvent| eprintln!("{event:?}")));
    let initialization_milliseconds = process_started.elapsed().as_secs_f64() * 1_000.0;

    let options = GenerationOptions {
        dimensions: Some(dimensions),
        steps: Some(4),
        guidance_scale: Some(1.0),
        seed: Some(args.seed),
        batch_size: 1,
    };
    let request = match (variant, args.source) {
        (BooguVariant::Image01Turbo, None) => ImageRequest::Generate(GenerateRequest {
            prompt: Prompt::new(args.prompt)?,
            negative_prompt: None,
            options,
        }),
        (BooguVariant::Image01EditTurbo | BooguVariant::Image01EditTurbo1k5, Some(path)) => {
            ImageRequest::Edit(EditRequest {
                source: read_input_image(path)?,
                instruction: Prompt::new(args.prompt)?,
                negative_prompt: None,
                mask: None,
                strength: None,
                options,
            })
        }
        _ => unreachable!("source/variant contract was checked"),
    };
    let query_chunk_sizes = if args.benchmark_denoiser_query_chunk_sizes.is_empty() {
        vec![args.denoiser_query_chunk_size]
    } else {
        args.benchmark_denoiser_query_chunk_sizes.clone()
    };
    let blackbox_seq_kv_tiles = if args.benchmark_blackbox_seq_kv_tiles.is_empty() {
        vec![args.blackbox_seq_kv_tiles]
    } else {
        args.benchmark_blackbox_seq_kv_tiles.clone()
    };
    let mut runs = Vec::with_capacity(
        usize::from(args.repeat) * query_chunk_sizes.len() * blackbox_seq_kv_tiles.len(),
    );
    let mut result = None;
    for query_chunk_size in query_chunk_sizes {
        runtime
            .model_mut()
            .pipeline_mut()
            .denoiser
            .set_query_chunk_size(query_chunk_size);
        for &seq_kv_tiles in &blackbox_seq_kv_tiles {
            runtime
                .model_mut()
                .pipeline_mut()
                .denoiser
                .set_blackbox_configuration(
                    args.blackbox_num_planes,
                    seq_kv_tiles,
                    args.blackbox_seq_q_tiles,
                )
                .map_err(|reason| {
                    format!(
                        "cannot apply padded-blackbox configuration planes={} seq_kv_tiles={} seq_q_tiles={} \
                         to the selected denoiser: {reason}",
                        args.blackbox_num_planes, seq_kv_tiles, args.blackbox_seq_q_tiles
                    )
                })?;
            for iteration in 0..args.repeat {
                if matches!(
                    args.conditioning_cache_policy,
                    ConditioningCachePolicy::Disabled
                ) {
                    // `pipeline_mut` is the model's explicit conditioning-cache invalidation
                    // boundary. Borrow it only after applying the attention configuration so the
                    // cache is cold immediately before inference without changing denoiser state.
                    let _ = runtime.model_mut().pipeline_mut();
                }
                let started = Instant::now();
                let current = runtime.infer(&request)?;
                runs.push(serde_json::json!({
                    "iteration": iteration,
                    "residency": args.conditioning_cache_policy.residency(iteration),
                    "conditioning_cache_policy": args.conditioning_cache_policy.id(),
                    "qwen_synchronization_policy": args.qwen_synchronization_policy.id(),
                    "denoiser_query_chunk_size": query_chunk_size,
                    "blackbox_num_planes": args.blackbox_num_planes,
                    "blackbox_seq_kv_tiles": seq_kv_tiles,
                    "blackbox_seq_q_tiles": args.blackbox_seq_q_tiles,
                    "denoiser_rms_norm_policy": args.denoiser_rms_norm_policy.id(),
                    "denoiser_qk_preparation_policy": args.denoiser_qk_preparation_policy.id(),
                    "denoiser_split_double_stream_shared_projection": args.denoiser_split_double_stream_shared_projection,
                    "wall_milliseconds": started.elapsed().as_secs_f64() * 1_000.0,
                    "stage_timings": &current.timings,
                }));
                result = Some(current);
            }
        }
    }
    let result = result.expect("--repeat is constrained to at least one inference");
    save_output(&args.output, &result.images[0].image)?;
    let process_wall_milliseconds = process_started.elapsed().as_secs_f64() * 1_000.0;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "output": args.output,
            "dimensions": result.images[0].image.dimensions(),
            "seed": result.seed,
            "initialization_milliseconds": initialization_milliseconds,
            "process_wall_milliseconds": process_wall_milliseconds,
            "timings": result.timings,
            "provenance": result.provenance,
            "repeat": args.repeat,
            "conditioning_cache_policy": args.conditioning_cache_policy.id(),
            "qwen_synchronization_policy": args.qwen_synchronization_policy.id(),
            "release_policy_validated": release_policy_validated,
            "conditioning_cache": match args.conditioning_cache_policy {
                ConditioningCachePolicy::ExactRetained => "single-entry-exact-prompt-source-length",
                ConditioningCachePolicy::Disabled => "cleared-before-every-infer",
            },
            "qwen_residency": "retained-after-first-load",
            "qwen_query_chunk_size": args.qwen_query_chunk_size,
            "vae_residency": "retained-after-first-load",
            "qwen_quantized_load_policy": match qwen_quantized_policy {
                BooguQuantizedLoadPolicy::Preserve => "preserve",
                BooguQuantizedLoadPolicy::DequantizeF16 => "dequantize-f16",
            },
            "vae_float_load_policy": args.vae_float_policy.id(),
            "vae_group_norm_policy": args.vae_group_norm_policy.id(),
            "denoiser_float_load_policy": match denoiser_policy {
                BooguFloatLoadPolicy::Preserve => "preserve",
                BooguFloatLoadPolicy::AdaptToF32 => "adapt-to-f32",
                BooguFloatLoadPolicy::PackedF16WeightsF32Auxiliaries => {
                    "packed-f16-weights-f32-auxiliaries"
                }
                BooguFloatLoadPolicy::PackedQ4sWeightsF32Auxiliaries => {
                    "packed-q4s-weights-f32-auxiliaries"
                }
            },
            "denoiser_quantized_load_policy": "preserve",
            "denoiser_attention_policy": denoiser_attention_policy,
            "denoiser_rms_norm_policy": args.denoiser_rms_norm_policy.id(),
            "denoiser_qk_preparation_policy": args.denoiser_qk_preparation_policy.id(),
            "denoiser_split_double_stream_shared_projection": args.denoiser_split_double_stream_shared_projection,
            "denoiser_query_chunk_size": args.denoiser_query_chunk_size,
            "vae_attention_query_chunk_size": args.vae_attention_query_chunk_size,
            "benchmark_denoiser_query_chunk_sizes": args.benchmark_denoiser_query_chunk_sizes,
            "blackbox_num_planes": args.blackbox_num_planes,
            "blackbox_seq_kv_tiles": args.blackbox_seq_kv_tiles,
            "blackbox_seq_q_tiles": args.blackbox_seq_q_tiles,
            "benchmark_blackbox_seq_kv_tiles": args.benchmark_blackbox_seq_kv_tiles,
            "execution_dtypes": {
                "qwen_visual": execution_dtypes.qwen_visual.name(),
                "vae": execution_dtypes.vae.name(),
                "denoiser": execution_dtypes.denoiser.name(),
            },
            "runs": runs,
        }))?
    );
    Ok(())
}

fn read_input_image(path: PathBuf) -> Result<InputImage, Box<dyn Error>> {
    let dimensions = image::image_dimensions(&path)
        .ok()
        .map(|(width, height)| Dimensions::new(width, height))
        .transpose()?;
    let encoding = match path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("png") => ImageEncoding::Png,
        Some("jpg" | "jpeg") => ImageEncoding::Jpeg,
        Some("webp") => ImageEncoding::Webp,
        _ => ImageEncoding::Other,
    };
    Ok(InputImage::Encoded(EncodedImage::new(
        encoding,
        dimensions,
        fs::read(path)?,
    )?))
}

fn save_output(path: &std::path::Path, output: &HostImage) -> Result<(), Box<dyn Error>> {
    match output {
        HostImage::Encoded(image) => fs::write(path, image.bytes())?,
        HostImage::Pixels(pixels) => {
            if pixels.format() != PixelFormat::Rgb8 || pixels.color_space() != ColorSpace::Srgb {
                return Err("boogu-run currently writes only RGB8 sRGB output".into());
            }
            let dimensions = pixels.dimensions();
            let image = image::RgbImage::from_raw(
                dimensions.width(),
                dimensions.height(),
                pixels.bytes().to_vec(),
            )
            .ok_or("validated output pixel buffer could not be materialized")?;
            image.save_with_format(path, image::ImageFormat::Png)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        Args, DenoiserQkPreparationPolicyChoice, DenoiserRmsNormPolicyChoice,
        is_exact_native_release_policy, validate_blackbox_configuration,
        validate_blackbox_seq_kv_tiles, validate_blackbox_seq_q_tiles,
    };
    use clap::Parser;

    #[test]
    fn blackbox_seq_kv_tile_cli_values_are_bounded_correctness() {
        for value in [1, 2] {
            assert_eq!(validate_blackbox_seq_kv_tiles(value), Ok(()));
        }
        for value in [0, 3, 4, 8, u8::MAX] {
            assert!(validate_blackbox_seq_kv_tiles(value).is_err());
            assert!(validate_blackbox_seq_q_tiles(value).is_err());
        }
        assert_eq!(validate_blackbox_seq_q_tiles(1), Ok(()));
        for configuration in [(2, 1, 1), (2, 2, 1), (4, 1, 1)] {
            assert_eq!(
                validate_blackbox_configuration(configuration.0, configuration.1, configuration.2,),
                Ok(())
            );
        }
        for configuration in [
            (1, 1, 1),
            (2, 1, 2),
            (2, 4, 1),
            (4, 1, 2),
            (4, 2, 1),
            (8, 1, 1),
        ] {
            assert!(
                validate_blackbox_configuration(configuration.0, configuration.1, configuration.2,)
                    .is_err()
            );
        }
    }

    #[test]
    fn denoiser_rms_norm_cli_defaults_strict_and_names_diagnostic_policy_correctness() {
        let strict = Args::try_parse_from([
            "boogu-run",
            "--artifacts",
            "artifacts",
            "--variant",
            "turbo",
            "--prompt",
            "test",
        ])
        .unwrap();
        assert_eq!(
            strict.denoiser_rms_norm_policy,
            DenoiserRmsNormPolicyChoice::StrictF32
        );

        let mixed = Args::try_parse_from([
            "boogu-run",
            "--artifacts",
            "artifacts",
            "--variant",
            "turbo",
            "--prompt",
            "test",
            "--denoiser-rms-norm-policy",
            "f16-storage-f32-accum",
        ])
        .unwrap();
        assert_eq!(mixed.denoiser_rms_norm_policy.id(), "f16-storage-f32-accum");
    }

    #[test]
    fn denoiser_qk_preparation_policy_defaults_safe_and_names_fused_candidate_correctness() {
        let composed = Args::try_parse_from([
            "boogu-run",
            "--artifacts",
            "artifacts",
            "--variant",
            "turbo",
            "--prompt",
            "test",
        ])
        .unwrap();
        assert_eq!(
            composed.denoiser_qk_preparation_policy,
            DenoiserQkPreparationPolicyChoice::Composed
        );

        let fused = Args::try_parse_from([
            "boogu-run",
            "--artifacts",
            "artifacts",
            "--variant",
            "turbo",
            "--prompt",
            "test",
            "--denoiser-qk-preparation-policy",
            "fused-strict-qk-norm-rope",
        ])
        .unwrap();
        assert_eq!(
            fused.denoiser_qk_preparation_policy.id(),
            "fused-strict-qk-norm-rope"
        );
    }

    #[test]
    fn native_release_policy_validation_is_variant_specific_correctness() {
        let exact_1k = Args::try_parse_from([
            "boogu-run",
            "--artifacts",
            "artifacts",
            "--variant",
            "turbo",
            "--prompt",
            "test",
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
            "--denoiser-qk-preparation-policy",
            "balanced-strict-qk-norm-rope",
        ])
        .unwrap();
        assert!(is_exact_native_release_policy(
            &exact_1k,
            burn_boogu::BooguVariant::Image01Turbo,
            "required-padded-blackbox-p4-kv1-q1-full-autotune",
        ));

        let old_1k = Args::try_parse_from([
            "boogu-run",
            "--artifacts",
            "artifacts",
            "--variant",
            "turbo",
            "--prompt",
            "test",
            "--qwen-synchronization-policy",
            "deferred",
            "--vae-float-policy",
            "preserve-f16",
            "--vae-group-norm-policy",
            "f16-storage-f32-accum",
            "--vae-attention-query-chunk-size",
            "1024",
            "--qwen-query-chunk-size",
            "128",
            "--denoiser-query-chunk-size",
            "8192",
        ])
        .unwrap();
        assert!(!is_exact_native_release_policy(
            &old_1k,
            burn_boogu::BooguVariant::Image01Turbo,
            "required-padded-blackbox-p4-kv1-q1-full-autotune",
        ));

        let exact_1k5 = Args::try_parse_from([
            "boogu-run",
            "--artifacts",
            "artifacts",
            "--variant",
            "edit-turbo-1k5",
            "--prompt",
            "test",
            "--source",
            "source.png",
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
        ])
        .unwrap();
        assert!(is_exact_native_release_policy(
            &exact_1k5,
            burn_boogu::BooguVariant::Image01EditTurbo1k5,
            "required-padded-blackbox-p4-kv1-q1-full-autotune",
        ));
    }
}
