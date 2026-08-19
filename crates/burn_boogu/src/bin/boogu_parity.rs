//! Real-fixture DMD and FLUX VAE parity runner.

use std::{
    error::Error,
    fs::{self, File},
    path::{Path, PathBuf},
    time::Instant,
};

use burn::{
    prelude::Backend,
    tensor::{Tensor, TensorData},
};
#[cfg(feature = "wgpu")]
use burn_boogu::require_native_wgpu_device;
use burn_boogu::{
    BooguConfig, BooguVariant,
    artifacts::{
        BooguArtifactInventory, BooguFloatLoadPolicy, BooguReleaseIdentity, BooguStorageProfile,
        VerifiedArtifactDirectory, load_vae_decoder_from_directory,
        load_vae_encoder_from_directory,
    },
    reference::{verify_reference_fixture, verify_reference_fixture_file},
};
use burn_flux_vae::{AutoencoderKl, AutoencoderKlConfig, load_safetensors_file};
use burn_image::HostImage;
use burn_qwen3_vl::Qwen3VlConfig;
use clap::{Parser, ValueEnum};
use half::{bf16, f16};
use memmap2::{Mmap, MmapOptions};
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use serde::{Deserialize, Serialize};

const VAE_F32_SCALED_LATENT_MAX_ABS: f32 = 0.005;
const VAE_F32_REFERENCE_NAMES: [&str; 6] = [
    "vae.reference_f32_moments",
    "vae.reference_f32_mean",
    "vae.reference_f32_logvar",
    "vae.reference_f32_std",
    "vae.reference_f32_raw_latent",
    "vae.reference_f32_scaled_latent",
];
const VAE_BF16_REFERENCE_NAMES: [&str; 6] = [
    "vae.reference_moments",
    "vae.reference_mean",
    "vae.reference_logvar",
    "vae.reference_std",
    "vae.reference_raw_latent",
    "vae.reference_scaled_latent",
];

#[derive(Debug, Clone, Copy, ValueEnum)]
enum BackendChoice {
    Ndarray,
    Wgpu,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ProfileChoice {
    F16,
    F16QwenVisionF32,
    Q8sBlock32F32,
    Q8sBlock32F32QwenVisionF32,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
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

#[derive(Debug, Parser)]
#[command(about = "Compare Burn math with an exported pinned Boogu fixture")]
struct Args {
    /// Directory containing tensors.safetensors and metadata.json.
    #[arg(long)]
    fixture: PathBuf,
    /// Upstream VAE diffusion_pytorch_model.safetensors (F32 execution oracle).
    #[arg(
        long,
        conflicts_with = "artifacts",
        required_unless_present = "artifacts"
    )]
    vae: Option<PathBuf>,
    /// Sealed converted bundle whose independently staged VAE is exercised.
    #[arg(long, conflicts_with = "vae", required_unless_present = "vae")]
    artifacts: Option<PathBuf>,
    /// Exact converted storage profile, required to match `--artifacts`.
    #[arg(long, value_enum, default_value = "f16-qwen-vision-f32")]
    profile: ProfileChoice,
    /// Burn backend used for VAE execution.
    #[arg(long, value_enum, default_value = "ndarray")]
    backend: BackendChoice,
    /// VAE weight and activation policy. The F16 path is native-WGPU-only and is compared against
    /// the same pinned upstream BF16 fixture.
    #[arg(long, value_enum, default_value = "force-f32")]
    vae_float_policy: VaeFloatPolicyChoice,
    /// Load only the sealed VAE encoder stage and report the six authenticated F32-oracle
    /// encoder surfaces. This mode stream-authenticates and memory-maps the fixture without
    /// materializing the decoder input, decoder output, or final pixels.
    #[arg(
        long,
        default_value_t = false,
        requires = "artifacts",
        conflicts_with = "vae"
    )]
    encoder_only: bool,
    /// Fail when the measured metrics exceed the checked-in gates.
    #[arg(long, default_value_t = false)]
    require: bool,
    /// Repeat VAE execution in one loaded process. This repeats the decoder normally and the
    /// encoder under `--encoder-only`. The first sample includes first-use kernel selection;
    /// later samples are retained warm-path measurements.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u16).range(1..))]
    repeat: u16,
}

#[derive(Debug, Serialize)]
struct Metrics {
    variant: String,
    model_revision: String,
    width: usize,
    height: usize,
    backend: String,
    dmd_max_abs: f32,
    dmd_mean_abs: f32,
    vae_max_abs: f32,
    vae_mean_abs: f32,
    vae_rmse: f32,
    vae_psnr_db: f32,
    vae_cosine_similarity: f32,
    output_rgb: ImageComparison,
    #[serde(skip_serializing_if = "Option::is_none")]
    vae_encode: Option<VaeEncodeMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vae_encode_bf16_drift: Option<VaeEncodeMetrics>,
    vae_load_milliseconds: f64,
    vae_decode_milliseconds: f64,
    vae_decode_milliseconds_by_run: Vec<f64>,
    vae_loaded_tensors: usize,
}

#[derive(Debug, Serialize)]
struct VaeEncodeMetrics {
    reference: String,
    moments: Comparison,
    mean: Comparison,
    logvar: Comparison,
    std: Comparison,
    raw_sample: Comparison,
    scaled_latent: Comparison,
    milliseconds: f64,
}

#[derive(Debug, Serialize)]
struct EncoderOnlyMetrics {
    mode: &'static str,
    variant: String,
    model_revision: String,
    width: usize,
    height: usize,
    backend: String,
    vae_encode: VaeEncodeMetrics,
    #[serde(skip_serializing_if = "Option::is_none")]
    vae_encode_bf16_drift: Option<VaeEncodeMetrics>,
    vae_load_milliseconds: f64,
    vae_encode_milliseconds_by_run: Vec<f64>,
    vae_loaded_tensors: usize,
}

#[derive(Debug, Serialize)]
struct ImageComparison {
    max_abs_u8: u8,
    mean_abs_u8: f32,
    rmse_u8: f32,
    psnr_db: f32,
    mean_block_ssim_8x8: f32,
    exact_fraction: f32,
}

#[derive(Debug, Deserialize)]
struct FixtureMetadata {
    variant: String,
    model_revision: String,
    width: usize,
    height: usize,
}

impl FixtureMetadata {
    fn variant(&self) -> Result<BooguVariant, Box<dyn Error>> {
        match self.variant.as_str() {
            "turbo" => Ok(BooguVariant::Image01Turbo),
            "edit-turbo" => Ok(BooguVariant::Image01EditTurbo),
            "edit-turbo-1k5" => Ok(BooguVariant::Image01EditTurbo1k5),
            other => Err(format!("unsupported fixture variant {other:?}").into()),
        }
    }
}

enum VaeSource<'a> {
    Upstream(&'a std::path::Path),
    Artifacts {
        root: &'a std::path::Path,
        profile: BooguStorageProfile,
    },
}

struct LoadedVae<B: Backend> {
    decoder: AutoencoderKl<B>,
    encoder: Option<AutoencoderKl<B>>,
    tensors: usize,
    label: String,
}

struct LoadedVaeEncoder<B: Backend> {
    encoder: AutoencoderKl<B>,
    tensors: usize,
    label: String,
}

struct VaeEncoderOutputs<B: Backend> {
    moments: Tensor<B, 4>,
    mean: Tensor<B, 4>,
    logvar: Tensor<B, 4>,
    std: Tensor<B, 4>,
    raw_sample: Tensor<B, 4>,
    scaled_latent: Tensor<B, 4>,
}

struct VaeRun<'a> {
    metadata: &'a FixtureMetadata,
    source: VaeSource<'a>,
    float_policy: BooguFloatLoadPolicy,
    repeat: u16,
    tensors: &'a SafeTensors<'a>,
    dmd: (f32, f32),
}

struct VaeEncoderRun<'a> {
    metadata: &'a FixtureMetadata,
    source: VaeSource<'a>,
    float_policy: BooguFloatLoadPolicy,
    repeat: u16,
    tensors: &'a SafeTensors<'a>,
}

enum FixtureBytes {
    Owned(Vec<u8>),
    Mapped(Mmap),
}

impl AsRef<[u8]> for FixtureBytes {
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::Owned(bytes) => bytes,
            Self::Mapped(bytes) => bytes,
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let fixture_metadata_bytes = fs::read(args.fixture.join("metadata.json"))?;
    let metadata: FixtureMetadata = serde_json::from_slice(&fixture_metadata_bytes)?;
    let tensor_path = args.fixture.join("tensors.safetensors");
    let fixture_bytes = if args.encoder_only {
        verify_reference_fixture_file(&fixture_metadata_bytes, &tensor_path)?;
        FixtureBytes::Mapped(memory_map_read_only(&tensor_path)?)
    } else {
        let bytes = fs::read(&tensor_path)?;
        verify_reference_fixture(&fixture_metadata_bytes, &bytes)?;
        FixtureBytes::Owned(bytes)
    };
    let tensors = SafeTensors::deserialize(fixture_bytes.as_ref())?;
    let source = if let Some(path) = args.vae.as_deref() {
        VaeSource::Upstream(path)
    } else {
        VaeSource::Artifacts {
            root: args
                .artifacts
                .as_deref()
                .ok_or("either --vae or --artifacts is required")?,
            profile: args.profile.into(),
        }
    };

    if args.encoder_only {
        let metrics = match args.backend {
            BackendChoice::Ndarray => {
                if !matches!(args.vae_float_policy, VaeFloatPolicyChoice::ForceF32) {
                    return Err("NdArray does not implement the native F16 VAE policy".into());
                }
                run_vae_encoder_only::<burn_ndarray::NdArray<f32>>(
                    Default::default(),
                    "ndarray-f32",
                    VaeEncoderRun {
                        metadata: &metadata,
                        source,
                        float_policy: args.vae_float_policy.load_policy(),
                        repeat: args.repeat,
                        tensors: &tensors,
                    },
                )?
            }
            BackendChoice::Wgpu => run_wgpu_encoder_only(
                &metadata,
                source,
                args.vae_float_policy.load_policy(),
                args.repeat,
                &tensors,
            )?,
        };
        println!("{}", serde_json::to_string_pretty(&metrics)?);
        if args.require {
            require_vae_encode_f32(&metrics.vae_encode)?;
        }
        return Ok(());
    }

    let dmd = dmd_metrics(&tensors)?;

    let metrics = match args.backend {
        BackendChoice::Ndarray => {
            if !matches!(args.vae_float_policy, VaeFloatPolicyChoice::ForceF32) {
                return Err("NdArray does not implement the native F16 VAE policy".into());
            }
            run_vae::<burn_ndarray::NdArray<f32>>(
                Default::default(),
                "ndarray-f32",
                VaeRun {
                    metadata: &metadata,
                    source,
                    float_policy: args.vae_float_policy.load_policy(),
                    repeat: args.repeat,
                    tensors: &tensors,
                    dmd,
                },
            )?
        }
        BackendChoice::Wgpu => run_wgpu(
            &metadata,
            source,
            args.vae_float_policy.load_policy(),
            args.repeat,
            &tensors,
            dmd,
        )?,
    };
    println!("{}", serde_json::to_string_pretty(&metrics)?);

    if args.require {
        // DMD compares F32 Burn/host arithmetic against BF16 CUDA fixture values. One BF16
        // rounding unit is expected at each multiply/add boundary.
        if metrics.dmd_max_abs > 0.04 || metrics.dmd_mean_abs > 0.003 {
            return Err(format!(
                "DMD parity gate failed: max_abs={} mean_abs={}",
                metrics.dmd_max_abs, metrics.dmd_mean_abs
            )
            .into());
        }
        // The upstream custom pipeline decodes in BF16, while FLUX force-upcast makes Burn use
        // F32 for portable WGPU/WebGPU. These gates reject topology/layout errors while allowing
        // the expected precision drift.
        if metrics.vae_max_abs > 0.02
            || metrics.vae_mean_abs > 0.002
            || metrics.vae_rmse > 0.0025
            || metrics.vae_cosine_similarity < 0.99999
        {
            return Err(format!(
                "VAE parity gate failed: max_abs={} mean_abs={} rmse={} cosine={}",
                metrics.vae_max_abs,
                metrics.vae_mean_abs,
                metrics.vae_rmse,
                metrics.vae_cosine_similarity
            )
            .into());
        }
        if let Some(encode) = &metrics.vae_encode {
            require_vae_encode_f32(encode)?;
        }
        if metrics.output_rgb.max_abs_u8 > 4
            || metrics.output_rgb.mean_abs_u8 > 0.5
            || metrics.output_rgb.psnr_db < 50.0
            || metrics.output_rgb.mean_block_ssim_8x8 < 0.995
        {
            return Err(format!("final RGB parity gate failed: {:?}", metrics.output_rgb).into());
        }
    }
    Ok(())
}

fn memory_map_read_only(path: &Path) -> Result<Mmap, Box<dyn Error>> {
    let file = File::open(path)?;
    // SAFETY: this process only holds a read-only fixture handle and never mutates the file. The
    // parity contract requires callers not to replace or truncate an authenticated fixture while
    // it is running. Keeping the mapping alive owns the resulting byte region after `file` drops.
    Ok(unsafe { MmapOptions::new().map(&file)? })
}

fn require_vae_encode_f32(encode: &VaeEncodeMetrics) -> Result<(), Box<dyn Error>> {
    if encode.reference != "pytorch-f32-cpu"
        || encode.moments.max_abs > 0.007
        || encode.moments.rmse > 0.0002
        || encode.moments.cosine < 0.999999
        || encode.mean.max_abs > 0.007
        || encode.logvar.max_abs > 0.004
        || encode.std.max_abs > 0.0001
        || encode.raw_sample.max_abs > 0.007
        || encode.scaled_latent.max_abs > VAE_F32_SCALED_LATENT_MAX_ABS
    {
        return Err(format!(
            "VAE encode F32-oracle gate failed: reference={}, moments={:?}, mean={:?}, logvar={:?}, std={:?}, raw_sample={:?}, scaled={:?}",
            encode.reference,
            encode.moments,
            encode.mean,
            encode.logvar,
            encode.std,
            encode.raw_sample,
            encode.scaled_latent
        )
        .into());
    }
    Ok(())
}

#[cfg(feature = "wgpu")]
fn run_wgpu(
    metadata: &FixtureMetadata,
    source: VaeSource<'_>,
    float_policy: BooguFloatLoadPolicy,
    repeat: u16,
    tensors: &SafeTensors<'_>,
    dmd: (f32, f32),
) -> Result<Metrics, Box<dyn Error>> {
    run_vae::<burn_wgpu::Wgpu<f32, i32, u32>>(
        require_native_wgpu_device()?,
        "wgpu-f32",
        VaeRun {
            metadata,
            source,
            float_policy,
            repeat,
            tensors,
            dmd,
        },
    )
}

#[cfg(feature = "wgpu")]
fn run_wgpu_encoder_only(
    metadata: &FixtureMetadata,
    source: VaeSource<'_>,
    float_policy: BooguFloatLoadPolicy,
    repeat: u16,
    tensors: &SafeTensors<'_>,
) -> Result<EncoderOnlyMetrics, Box<dyn Error>> {
    run_vae_encoder_only::<burn_wgpu::Wgpu<f32, i32, u32>>(
        require_native_wgpu_device()?,
        "wgpu-f32",
        VaeEncoderRun {
            metadata,
            source,
            float_policy,
            repeat,
            tensors,
        },
    )
}

#[cfg(not(feature = "wgpu"))]
fn run_wgpu(
    _metadata: &FixtureMetadata,
    _source: VaeSource<'_>,
    _float_policy: BooguFloatLoadPolicy,
    _repeat: u16,
    _tensors: &SafeTensors<'_>,
    _dmd: (f32, f32),
) -> Result<Metrics, Box<dyn Error>> {
    Err("the wgpu backend requires --features wgpu".into())
}

#[cfg(not(feature = "wgpu"))]
fn run_wgpu_encoder_only(
    _metadata: &FixtureMetadata,
    _source: VaeSource<'_>,
    _float_policy: BooguFloatLoadPolicy,
    _repeat: u16,
    _tensors: &SafeTensors<'_>,
) -> Result<EncoderOnlyMetrics, Box<dyn Error>> {
    Err("the wgpu backend requires --features wgpu".into())
}

fn run_vae_encoder_only<B: Backend>(
    device: B::Device,
    backend: &str,
    run: VaeEncoderRun<'_>,
) -> Result<EncoderOnlyMetrics, Box<dyn Error>> {
    let VaeEncoderRun {
        metadata,
        source,
        float_policy,
        repeat,
        tensors,
    } = run;
    for name in [
        "vae.reference_input",
        "vae.reference_epsilon",
        VAE_F32_REFERENCE_NAMES[0],
        VAE_F32_REFERENCE_NAMES[1],
        VAE_F32_REFERENCE_NAMES[2],
        VAE_F32_REFERENCE_NAMES[3],
        VAE_F32_REFERENCE_NAMES[4],
        VAE_F32_REFERENCE_NAMES[5],
    ] {
        tensors.tensor(name).map_err(|error| {
            format!("--encoder-only requires authenticated fixture tensor {name:?}: {error}")
        })?;
    }

    let execution_dtype = match float_policy {
        BooguFloatLoadPolicy::Preserve => burn::tensor::DType::F16,
        BooguFloatLoadPolicy::AdaptToF32
        | BooguFloatLoadPolicy::PackedF16WeightsF32Auxiliaries
        | BooguFloatLoadPolicy::PackedQ4sWeightsF32Auxiliaries => burn::tensor::DType::F32,
    };
    let input_view = tensors.tensor("vae.reference_input")?;
    let input = Tensor::<B, 4>::from_data(
        TensorData::new(decode_f32(&input_view)?, shape4(&input_view)?),
        &device,
    )
    .cast(execution_dtype);
    let epsilon_view = tensors.tensor("vae.reference_epsilon")?;
    let epsilon = Tensor::<B, 4>::from_data(
        TensorData::new(decode_f32(&epsilon_view)?, shape4(&epsilon_view)?),
        &device,
    )
    .cast(execution_dtype);

    let load_started = Instant::now();
    let loaded = load_vae_encoder_model::<B>(&device, metadata, source, float_policy)?;
    B::sync(&device)?;
    let load_milliseconds = load_started.elapsed().as_secs_f64() * 1_000.0;

    let mut milliseconds_by_run = Vec::with_capacity(usize::from(repeat));
    let mut outputs = None;
    for _ in 0..repeat {
        let started = Instant::now();
        let current = run_vae_encoder(&loaded.encoder, input.clone(), epsilon.clone());
        B::sync(&device)?;
        milliseconds_by_run.push(started.elapsed().as_secs_f64() * 1_000.0);
        outputs = Some(current);
    }
    let outputs = outputs.expect("--repeat is constrained to at least one encode");
    let milliseconds = *milliseconds_by_run
        .last()
        .expect("--repeat is constrained to at least one encode");
    let vae_encode = compare_vae_encoder_outputs(
        &outputs,
        tensors,
        "vae.reference_f32_",
        "pytorch-f32-cpu",
        milliseconds,
    )?;
    let vae_encode_bf16_drift = VAE_BF16_REFERENCE_NAMES
        .iter()
        .all(|name| tensors.tensor(name).is_ok())
        .then(|| {
            compare_vae_encoder_outputs(
                &outputs,
                tensors,
                "vae.reference_",
                "upstream-bf16-execution",
                milliseconds,
            )
        })
        .transpose()?;

    Ok(EncoderOnlyMetrics {
        mode: "encoder-only",
        variant: metadata.variant.clone(),
        model_revision: metadata.model_revision.clone(),
        width: metadata.width,
        height: metadata.height,
        backend: format!("{backend}/{}", loaded.label),
        vae_encode,
        vae_encode_bf16_drift,
        vae_load_milliseconds: load_milliseconds,
        vae_encode_milliseconds_by_run: milliseconds_by_run,
        vae_loaded_tensors: loaded.tensors,
    })
}

fn run_vae_encoder<B: Backend>(
    encoder: &AutoencoderKl<B>,
    input: Tensor<B, 4>,
    epsilon: Tensor<B, 4>,
) -> VaeEncoderOutputs<B> {
    let moments = encoder.encode_moments(input);
    let posterior = burn_flux_vae::DiagonalGaussian::from_moments(moments.clone());
    let mean = posterior.mean();
    let logvar = posterior.logvar();
    let std = posterior.std();
    let raw_sample = posterior.sample_with_epsilon(epsilon);
    let scaled_latent = encoder.scale_latents(raw_sample.clone());
    VaeEncoderOutputs {
        moments,
        mean,
        logvar,
        std,
        raw_sample,
        scaled_latent,
    }
}

fn compare_vae_encoder_outputs<B: Backend>(
    outputs: &VaeEncoderOutputs<B>,
    tensors: &SafeTensors<'_>,
    prefix: &str,
    reference: &str,
    milliseconds: f64,
) -> Result<VaeEncodeMetrics, Box<dyn Error>> {
    Ok(VaeEncodeMetrics {
        reference: reference.to_owned(),
        moments: compare_tensor(
            outputs.moments.clone(),
            &tensors.tensor(&format!("{prefix}moments"))?,
            "VAE moments",
        )?,
        mean: compare_tensor(
            outputs.mean.clone(),
            &tensors.tensor(&format!("{prefix}mean"))?,
            "VAE posterior mean",
        )?,
        logvar: compare_tensor(
            outputs.logvar.clone(),
            &tensors.tensor(&format!("{prefix}logvar"))?,
            "VAE posterior logvar",
        )?,
        std: compare_tensor(
            outputs.std.clone(),
            &tensors.tensor(&format!("{prefix}std"))?,
            "VAE posterior std",
        )?,
        raw_sample: compare_tensor(
            outputs.raw_sample.clone(),
            &tensors.tensor(&format!("{prefix}raw_latent"))?,
            "VAE raw sample",
        )?,
        scaled_latent: compare_tensor(
            outputs.scaled_latent.clone(),
            &tensors.tensor(&format!("{prefix}scaled_latent"))?,
            "VAE scaled latent",
        )?,
        milliseconds,
    })
}

fn run_vae<B: Backend>(
    device: B::Device,
    backend: &str,
    run: VaeRun<'_>,
) -> Result<Metrics, Box<dyn Error>> {
    let VaeRun {
        metadata,
        source,
        float_policy,
        repeat,
        tensors,
        dmd: (dmd_max_abs, dmd_mean_abs),
    } = run;
    let input_view = tensors.tensor("vae.decode_input")?;
    let expected_view = tensors.tensor("vae.decode_output")?;
    let input_shape = shape4(&input_view)?;
    let expected_shape = shape4(&expected_view)?;
    let input = decode_f32(&input_view)?;
    let expected = decode_f32(&expected_view)?;
    let execution_dtype = match float_policy {
        BooguFloatLoadPolicy::Preserve => burn::tensor::DType::F16,
        BooguFloatLoadPolicy::AdaptToF32
        | BooguFloatLoadPolicy::PackedF16WeightsF32Auxiliaries
        | BooguFloatLoadPolicy::PackedQ4sWeightsF32Auxiliaries => burn::tensor::DType::F32,
    };
    let input = Tensor::<B, 4>::from_data(TensorData::new(input, input_shape), &device)
        .cast(execution_dtype);

    let load_started = Instant::now();
    let loaded = load_vae_models::<B>(&device, metadata, source, float_policy)?;
    B::sync(&device)?;
    let load_milliseconds = load_started.elapsed().as_secs_f64() * 1_000.0;
    let mut vae_decode_milliseconds_by_run = Vec::with_capacity(usize::from(repeat));
    let mut actual = None;
    for _ in 0..repeat {
        let decode_started = Instant::now();
        let current = loaded.decoder.decode(input.clone());
        B::sync(&device)?;
        vae_decode_milliseconds_by_run.push(decode_started.elapsed().as_secs_f64() * 1_000.0);
        actual = Some(current);
    }
    let actual = actual.expect("--repeat is constrained to at least one decode");
    let decode_milliseconds = *vae_decode_milliseconds_by_run
        .last()
        .expect("--repeat is constrained to at least one decode");
    if actual.dims() != expected_shape {
        return Err(format!(
            "VAE output shape {:?} differs from fixture {expected_shape:?}",
            actual.dims()
        )
        .into());
    }
    let output_rgb = compare_output_rgb(
        burn_boogu::decoder_output_to_host(actual.clone())?,
        &tensors.tensor("output.rgb_u8")?,
        metadata.width,
        metadata.height,
    )?;
    let actual_values = actual
        .to_data()
        .convert_dtype(burn::tensor::DType::F32)
        .to_vec::<f32>()?;
    let comparison = compare(&actual_values, &expected)?;
    let encode_names = [
        "vae.reference_input",
        "vae.reference_epsilon",
        "vae.reference_moments",
        "vae.reference_mean",
        "vae.reference_logvar",
        "vae.reference_raw_latent",
        "vae.reference_scaled_latent",
    ];
    let (encode, encode_bf16_drift) =
        if encode_names.iter().all(|name| tensors.tensor(name).is_ok()) {
            let input_view = tensors.tensor("vae.reference_input")?;
            let input_shape = shape4(&input_view)?;
            let input = Tensor::<B, 4>::from_data(
                TensorData::new(decode_f32(&input_view)?, input_shape),
                &device,
            )
            .cast(execution_dtype);
            let epsilon_view = tensors.tensor("vae.reference_epsilon")?;
            let epsilon_shape = shape4(&epsilon_view)?;
            let epsilon = Tensor::<B, 4>::from_data(
                TensorData::new(decode_f32(&epsilon_view)?, epsilon_shape),
                &device,
            )
            .cast(execution_dtype);
            let started = Instant::now();
            let encoder = loaded.encoder.as_ref().unwrap_or(&loaded.decoder);
            let outputs = run_vae_encoder(encoder, input, epsilon);
            B::sync(&device)?;
            let milliseconds = started.elapsed().as_secs_f64() * 1_000.0;
            if VAE_F32_REFERENCE_NAMES
                .iter()
                .all(|name| tensors.tensor(name).is_ok())
            {
                (
                    Some(compare_vae_encoder_outputs(
                        &outputs,
                        tensors,
                        "vae.reference_f32_",
                        "pytorch-f32-cpu",
                        milliseconds,
                    )?),
                    Some(compare_vae_encoder_outputs(
                        &outputs,
                        tensors,
                        "vae.reference_",
                        "upstream-bf16-execution",
                        milliseconds,
                    )?),
                )
            } else {
                (
                    Some(compare_vae_encoder_outputs(
                        &outputs,
                        tensors,
                        "vae.reference_",
                        "upstream-bf16-execution",
                        milliseconds,
                    )?),
                    None,
                )
            }
        } else {
            (None, None)
        };
    Ok(Metrics {
        variant: metadata.variant.clone(),
        model_revision: metadata.model_revision.clone(),
        width: metadata.width,
        height: metadata.height,
        backend: format!("{backend}/{}", loaded.label),
        dmd_max_abs,
        dmd_mean_abs,
        vae_max_abs: comparison.max_abs,
        vae_mean_abs: comparison.mean_abs,
        vae_rmse: comparison.rmse,
        vae_psnr_db: 20.0 * (2.0 / comparison.rmse.max(f32::MIN_POSITIVE)).log10(),
        vae_cosine_similarity: comparison.cosine,
        output_rgb,
        vae_encode: encode,
        vae_encode_bf16_drift: encode_bf16_drift,
        vae_load_milliseconds: load_milliseconds,
        vae_decode_milliseconds: decode_milliseconds,
        vae_decode_milliseconds_by_run,
        vae_loaded_tensors: loaded.tensors,
    })
}

fn load_vae_models<B: Backend>(
    device: &B::Device,
    metadata: &FixtureMetadata,
    source: VaeSource<'_>,
    float_policy: BooguFloatLoadPolicy,
) -> Result<LoadedVae<B>, Box<dyn Error>> {
    match source {
        VaeSource::Upstream(path) => {
            let (model, report) =
                load_safetensors_file::<B>(device, path, &AutoencoderKlConfig::flux1())?;
            Ok(LoadedVae {
                decoder: model,
                encoder: None,
                tensors: report.applied.len(),
                label: "upstream-bf16-weights-f32".into(),
            })
        }
        VaeSource::Artifacts { root, profile } => {
            let artifact_directory = VerifiedArtifactDirectory::open(root)?;
            let qwen = Qwen3VlConfig::from_json(
                &artifact_directory.read_text("metadata/source/mllm/config.json")?,
            )?;
            let config = AutoencoderKlConfig::from_diffusers_json(
                &artifact_directory.read_text("metadata/source/vae/config.json")?,
            )?;
            let inventory = BooguArtifactInventory::new(&qwen, &BooguConfig::default(), &config)?;
            let identity = BooguReleaseIdentity::canonical(metadata.variant()?);
            let (decoder, decoder_report) = load_vae_decoder_from_directory::<B>(
                &identity,
                root,
                inventory.clone(),
                config.clone(),
                profile,
                float_policy,
                device,
            )?;
            let (encoder, encoder_report) = load_vae_encoder_from_directory::<B>(
                &identity,
                root,
                inventory,
                config,
                profile,
                float_policy,
                device,
            )?;
            Ok(LoadedVae {
                decoder,
                encoder: Some(encoder),
                tensors: decoder_report.tensors + encoder_report.tensors,
                label: format!("burnpack-{}", profile_label(profile)),
            })
        }
    }
}

fn load_vae_encoder_model<B: Backend>(
    device: &B::Device,
    metadata: &FixtureMetadata,
    source: VaeSource<'_>,
    float_policy: BooguFloatLoadPolicy,
) -> Result<LoadedVaeEncoder<B>, Box<dyn Error>> {
    let VaeSource::Artifacts { root, profile } = source else {
        return Err(
            "--encoder-only requires --artifacts so decoder weights are never loaded".into(),
        );
    };
    let artifact_directory = VerifiedArtifactDirectory::open(root)?;
    let qwen = Qwen3VlConfig::from_json(
        &artifact_directory.read_text("metadata/source/mllm/config.json")?,
    )?;
    let config = AutoencoderKlConfig::from_diffusers_json(
        &artifact_directory.read_text("metadata/source/vae/config.json")?,
    )?;
    let inventory = BooguArtifactInventory::new(&qwen, &BooguConfig::default(), &config)?;
    let identity = BooguReleaseIdentity::canonical(metadata.variant()?);
    let (encoder, report) = load_vae_encoder_from_directory::<B>(
        &identity,
        root,
        inventory,
        config,
        profile,
        float_policy,
        device,
    )?;
    Ok(LoadedVaeEncoder {
        encoder,
        tensors: report.tensors,
        label: format!("burnpack-{}", profile_label(profile)),
    })
}

const fn profile_label(profile: BooguStorageProfile) -> &'static str {
    match profile {
        BooguStorageProfile::F16 => "f16",
        BooguStorageProfile::F16QwenVisionF32 => "f16-qwen-vision-f32",
        BooguStorageProfile::Q8sBlock32F32 => "q8s-block32-f32",
        BooguStorageProfile::Q8sBlock32F32QwenVisionF32 => "q8s-block32-f32-qwen-vision-f32",
    }
}

fn dmd_metrics(tensors: &SafeTensors<'_>) -> Result<(f32, f32), Box<dyn Error>> {
    let mut actual = Vec::new();
    let mut expected = Vec::new();
    for step in 0..4 {
        let input = decode_f32(&tensors.tensor(&format!("dmd.step.{step}.input"))?)?;
        let velocity = decode_f32(&tensors.tensor(&format!("dmd.step.{step}.velocity"))?)?;
        let sigma = decode_f32(&tensors.tensor(&format!("dmd.step.{step}.sigma"))?)?[0];
        let prediction = input
            .iter()
            .zip(velocity)
            .map(|(&latent, velocity)| latent + (1.0 - sigma) * velocity)
            .collect::<Vec<_>>();
        if step < 3 {
            let expected_prediction =
                decode_f32(&tensors.tensor(&format!("dmd.step.{step}.prediction"))?)?;
            actual.extend_from_slice(&prediction);
            expected.extend_from_slice(&expected_prediction);
            let noise = decode_f32(&tensors.tensor(&format!("dmd.step.{step}.noise"))?)?;
            let next_input = decode_f32(&tensors.tensor(&format!("dmd.step.{}.input", step + 1))?)?;
            let next_sigma =
                decode_f32(&tensors.tensor(&format!("dmd.step.{}.sigma", step + 1))?)?[0];
            actual.extend(
                noise
                    .into_iter()
                    .zip(prediction)
                    .map(|(noise, prediction)| {
                        (1.0 - next_sigma) * noise + next_sigma * prediction
                    }),
            );
            expected.extend(next_input);
        } else if let Ok(final_view) = tensors.tensor("dmd.final_latents") {
            actual.extend(prediction);
            expected.extend(decode_f32(&final_view)?);
        }
    }
    let metrics = compare(&actual, &expected)?;
    Ok((metrics.max_abs, metrics.mean_abs))
}

#[derive(Debug, Serialize)]
struct Comparison {
    max_abs: f32,
    mean_abs: f32,
    rmse: f32,
    cosine: f32,
}

fn compare_output_rgb(
    actual: HostImage,
    expected: &TensorView<'_>,
    width: usize,
    height: usize,
) -> Result<ImageComparison, Box<dyn Error>> {
    let HostImage::Pixels(actual) = actual else {
        return Err("decoder postprocess unexpectedly returned an encoded image".into());
    };
    if expected.dtype() != Dtype::U8 || expected.shape() != [height, width, 3] {
        return Err(format!(
            "output.rgb_u8 must be U8 [{height},{width},3], got {:?} {:?}",
            expected.dtype(),
            expected.shape()
        )
        .into());
    }
    let actual = actual.bytes();
    let expected = expected.data();
    if actual.len() != expected.len() || actual.is_empty() {
        return Err(format!(
            "final RGB byte count differs: actual={} expected={}",
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
    let count = actual.len() as f64;
    let rmse = (sum_squared / count).sqrt();
    Ok(ImageComparison {
        max_abs_u8: max_abs,
        mean_abs_u8: (sum_abs / count) as f32,
        rmse_u8: rmse as f32,
        psnr_db: (20.0 * (255.0 / rmse.max(f64::MIN_POSITIVE)).log10()) as f32,
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

fn compare_tensor<B: Backend>(
    actual: Tensor<B, 4>,
    expected: &TensorView<'_>,
    label: &str,
) -> Result<Comparison, Box<dyn Error>> {
    let actual_shape = actual.dims();
    let expected_shape = shape4(expected)?;
    if actual_shape != expected_shape {
        return Err(format!(
            "{label} shape {actual_shape:?} differs from fixture {expected_shape:?}"
        )
        .into());
    }
    compare(
        &actual
            .to_data()
            .convert_dtype(burn::tensor::DType::F32)
            .to_vec::<f32>()?,
        &decode_f32(expected)?,
    )
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
    let denominator = (actual_squared * expected_squared).sqrt();
    Ok(Comparison {
        max_abs,
        mean_abs: (sum_abs / count) as f32,
        rmse: (sum_squared / count).sqrt() as f32,
        cosine: if denominator == 0.0 {
            1.0
        } else {
            (dot / denominator) as f32
        },
    })
}

fn shape4(view: &TensorView<'_>) -> Result<[usize; 4], Box<dyn Error>> {
    view.shape()
        .try_into()
        .map_err(|_| format!("expected rank-four tensor, got {:?}", view.shape()).into())
}

fn decode_f32(view: &TensorView<'_>) -> Result<Vec<f32>, Box<dyn Error>> {
    let bytes = view.data();
    let values = match view.dtype() {
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
        other => return Err(format!("unsupported fixture dtype {other:?}").into()),
    };
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_only_cli_requires_sealed_artifacts_correctness() {
        let parsed = Args::try_parse_from([
            "boogu-parity",
            "--fixture",
            "fixture",
            "--artifacts",
            "artifacts",
            "--encoder-only",
        ])
        .unwrap();
        assert!(parsed.encoder_only);
        assert!(parsed.artifacts.is_some());
        assert!(parsed.vae.is_none());

        assert!(
            Args::try_parse_from([
                "boogu-parity",
                "--fixture",
                "fixture",
                "--vae",
                "upstream.safetensors",
                "--encoder-only",
            ])
            .is_err(),
            "encoder-only must reject the upstream loader because it allocates decoder weights"
        );
    }

    #[test]
    fn encoder_only_f32_oracle_surface_contract_correctness() {
        assert_eq!(
            VAE_F32_REFERENCE_NAMES,
            [
                "vae.reference_f32_moments",
                "vae.reference_f32_mean",
                "vae.reference_f32_logvar",
                "vae.reference_f32_std",
                "vae.reference_f32_raw_latent",
                "vae.reference_f32_scaled_latent",
            ]
        );
    }

    #[test]
    fn block_ssim_is_one_for_identical_non_multiple_dimensions_correctness() {
        let pixels = (0_u8..45).collect::<Vec<_>>();
        let actual = mean_block_ssim_8x8(&pixels, &pixels, 5, 3).unwrap();
        assert!((actual - 1.0).abs() <= f32::EPSILON);
    }

    #[test]
    fn block_ssim_detects_a_changed_channel_correctness() {
        let expected = vec![128_u8; 8 * 8 * 3];
        let mut actual = expected.clone();
        for pixel in actual.chunks_exact_mut(3) {
            pixel[1] = 0;
        }
        let score = mean_block_ssim_8x8(&actual, &expected, 8, 8).unwrap();
        assert!(
            score < 0.7,
            "changed channel should lower SSIM, got {score}"
        );
        assert!(
            score > 0.6,
            "two unchanged channels should remain visible, got {score}"
        );
    }

    #[test]
    fn comparison_rejects_non_finite_values_correctness() {
        assert!(compare(&[f32::NAN], &[0.0]).is_err());
        assert!(compare(&[0.0], &[f32::INFINITY]).is_err());
    }
}
