//! Real-checkpoint, block-by-block parity for the streamed Boogu denoiser.

use std::{error::Error, fs, path::PathBuf, time::Instant};

use burn::{
    prelude::Backend,
    tensor::{DType, Tensor, TensorData},
};
#[cfg(feature = "wgpu")]
use burn_boogu::require_native_wgpu_device;
use burn_boogu::{
    BooguConfig, BooguDenoiserInput, BooguError, BooguTask, BooguVariant, DenoiserStageObserver,
    DmdSchedule, StreamingBooguDenoiser,
    artifacts::{
        BooguArtifactInventory, BooguFloatLoadPolicy, BooguQuantizedLoadPolicy,
        BooguReleaseIdentity, BooguStorageProfile, VerifiedArtifactDirectory,
        VerifiedBurnpackStageSource,
    },
    dmd_prediction, dmd_renoise,
    reference::verify_reference_fixture,
};
use burn_flux_vae::AutoencoderKlConfig;
use burn_qwen3_vl::Qwen3VlConfig;
use clap::{Parser, ValueEnum};
use half::{bf16, f16};
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum BackendChoice {
    Ndarray,
    Wgpu,
}

#[derive(Debug, Clone, Copy, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
enum InputMode {
    /// Inject each captured upstream step input independently for block-local diagnostics.
    Isolated,
    /// Start from captured initial latents and propagate Burn predictions with captured noise.
    Trajectory,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
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

#[derive(Debug, Parser)]
#[command(about = "Compare every streamed Boogu denoiser stage with a pinned upstream fixture")]
struct Args {
    /// Directory containing the sealed converted artifact bundle.
    #[arg(long)]
    artifacts: PathBuf,
    /// Directory containing tensors.safetensors and metadata.json from the reference exporter.
    #[arg(long)]
    fixture: PathBuf,
    /// Burn backend used for execution.
    #[arg(long, value_enum, default_value = "wgpu")]
    backend: BackendChoice,
    /// Converted artifact storage profile.
    #[arg(long, value_enum, default_value = "f16-qwen-vision-f32")]
    profile: ProfileChoice,
    /// Number of DMD denoiser calls to compare, from the beginning of the fixture.
    #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u8).range(1..=4))]
    steps: u8,
    /// DMD input semantics. Isolated preserves the original block-local parity behavior.
    #[arg(long, value_enum, default_value = "isolated")]
    input_mode: InputMode,
    /// Compare every internal refiner/block boundary. Step-level replay metrics remain enabled.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    capture_boundaries: bool,
    /// Override the mode-specific maximum relative RMSE gate.
    #[arg(long, value_parser = parse_non_negative_f32)]
    maximum_relative_rmse: Option<f32>,
    /// Override the mode-specific minimum cosine-similarity gate.
    #[arg(long, value_parser = parse_cosine)]
    minimum_cosine: Option<f32>,
    /// Fail when any checked-in numerical gate is exceeded.
    #[arg(long, default_value_t = false)]
    require: bool,
}

#[derive(Debug, Deserialize)]
struct FixtureMetadata {
    variant: String,
    model_revision: String,
    width: usize,
    height: usize,
}

impl FixtureMetadata {
    fn release_variant(&self) -> Result<BooguVariant, Box<dyn Error>> {
        match self.variant.as_str() {
            "turbo" => Ok(BooguVariant::Image01Turbo),
            "edit-turbo" => Ok(BooguVariant::Image01EditTurbo),
            "edit-turbo-1k5" => Ok(BooguVariant::Image01EditTurbo1k5),
            other => Err(format!("unsupported fixture variant {other:?}").into()),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct BoundaryMetric {
    name: String,
    shape: Vec<usize>,
    max_abs: f32,
    mean_abs: f32,
    rmse: f32,
    relative_rmse: f32,
    cosine_similarity: f32,
    readback_milliseconds: f64,
}

#[derive(Debug, Serialize)]
struct Report {
    variant: String,
    model_revision: String,
    backend: String,
    profile: String,
    float_load_policy: String,
    quantized_load_policy: String,
    execution_dtype: String,
    input_mode: InputMode,
    sigma_source: String,
    gate_basis: String,
    gate_maximum_relative_rmse: f32,
    gate_minimum_cosine_similarity: f32,
    width: usize,
    height: usize,
    steps: u8,
    boundary_count: usize,
    worst_max_abs: f32,
    worst_relative_rmse: f32,
    minimum_cosine_similarity: f32,
    load_and_execute_milliseconds: f64,
    boundaries: Vec<BoundaryMetric>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let fixture_metadata_bytes = fs::read(args.fixture.join("metadata.json"))?;
    let metadata: FixtureMetadata = serde_json::from_slice(&fixture_metadata_bytes)?;
    let expected_revision = BooguReleaseIdentity::canonical(metadata.release_variant()?)
        .model_revision
        .to_owned();
    if metadata.model_revision != expected_revision {
        return Err(format!(
            "fixture revision {} does not match canonical release {expected_revision}",
            metadata.model_revision
        )
        .into());
    }
    let fixture_bytes = fs::read(args.fixture.join("tensors.safetensors"))?;
    verify_reference_fixture(&fixture_metadata_bytes, &fixture_bytes)?;
    let fixture = SafeTensors::deserialize(&fixture_bytes)?;
    let report = match args.backend {
        BackendChoice::Ndarray => run::<burn_ndarray::NdArray<f32>>(
            Default::default(),
            "ndarray-f32",
            true,
            &args,
            &metadata,
            &fixture,
        )?,
        BackendChoice::Wgpu => run_wgpu(&args, &metadata, &fixture)?,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    if args.require {
        require_parity(&report)?;
    }
    Ok(())
}

#[cfg(feature = "wgpu")]
fn run_wgpu(
    args: &Args,
    metadata: &FixtureMetadata,
    fixture: &SafeTensors<'_>,
) -> Result<Report, Box<dyn Error>> {
    run::<burn_wgpu::Wgpu<f32, i32, u32>>(
        require_native_wgpu_device()?,
        if args.profile.is_q8() {
            "wgpu-f32-activations"
        } else {
            "wgpu-f16-activations"
        },
        false,
        args,
        metadata,
        fixture,
    )
}

#[cfg(not(feature = "wgpu"))]
fn run_wgpu(
    _args: &Args,
    _metadata: &FixtureMetadata,
    _fixture: &SafeTensors<'_>,
) -> Result<Report, Box<dyn Error>> {
    Err("the WGPU backend requires --features wgpu".into())
}

fn run<B: Backend>(
    device: B::Device,
    backend: &str,
    adapt_float32: bool,
    args: &Args,
    metadata: &FixtureMetadata,
    fixture: &SafeTensors<'_>,
) -> Result<Report, Box<dyn Error>> {
    let variant = metadata.release_variant()?;
    let identity = BooguReleaseIdentity::canonical(variant);
    let artifact_directory = VerifiedArtifactDirectory::open(&args.artifacts)?;
    let qwen_config = Qwen3VlConfig::from_json(
        &artifact_directory.read_text("metadata/source/mllm/config.json")?,
    )?;
    let vae_config = AutoencoderKlConfig::from_diffusers_json(
        &artifact_directory.read_text("metadata/source/vae/config.json")?,
    )?;
    let config = BooguConfig::default();
    let inventory = BooguArtifactInventory::new(&qwen_config, &config, &vae_config)?;
    let float_policy = if adapt_float32 || args.profile.is_q8() {
        BooguFloatLoadPolicy::AdaptToF32
    } else {
        BooguFloatLoadPolicy::Preserve
    };
    let quantized_policy = BooguQuantizedLoadPolicy::Preserve;
    let source = VerifiedBurnpackStageSource::<B, _>::from_directory(
        &identity,
        &args.artifacts,
        inventory,
        config.clone(),
        args.profile.into(),
        device.clone(),
    )?
    .with_float_load_policy(float_policy)
    .with_quantized_load_policy(quantized_policy);
    let mut denoiser = StreamingBooguDenoiser::<B, _>::new(config, source)?;
    let execution_dtype = if float_policy == BooguFloatLoadPolicy::AdaptToF32 {
        DType::F32
    } else {
        DType::F16
    };

    let instruction =
        tensor3::<B>(fixture, "qwen.last_hidden_state", &device)?.cast(execution_dtype);
    let reference = if variant.is_edit() {
        Some(tensor4::<B>(fixture, "vae.reference_scaled_latent", &device)?.cast(execution_dtype))
    } else {
        None
    };
    let started = Instant::now();
    let mut boundaries = Vec::new();
    match args.input_mode {
        InputMode::Isolated => run_isolated(
            &mut denoiser,
            fixture,
            &device,
            execution_dtype,
            instruction,
            reference,
            args.steps,
            args.capture_boundaries,
            &mut boundaries,
        )?,
        InputMode::Trajectory => run_trajectory(
            &mut denoiser,
            fixture,
            &device,
            execution_dtype,
            instruction,
            reference,
            variant,
            args.steps,
            args.capture_boundaries,
            &mut boundaries,
        )?,
    }
    B::sync(&device)?;
    let load_and_execute_milliseconds = started.elapsed().as_secs_f64() * 1_000.0;
    let worst_max_abs = boundaries
        .iter()
        .map(|metric| metric.max_abs)
        .fold(0.0_f32, f32::max);
    let worst_relative_rmse = boundaries
        .iter()
        .map(|metric| metric.relative_rmse)
        .fold(0.0_f32, f32::max);
    let minimum_cosine_similarity = boundaries
        .iter()
        .map(|metric| metric.cosine_similarity)
        .fold(1.0_f32, f32::min);
    let gate = parity_gate(args);
    Ok(Report {
        variant: metadata.variant.clone(),
        model_revision: metadata.model_revision.clone(),
        backend: backend.to_owned(),
        profile: args.profile.slug().to_owned(),
        float_load_policy: match float_policy {
            BooguFloatLoadPolicy::Preserve => "preserve",
            BooguFloatLoadPolicy::AdaptToF32 => "adapt-to-f32",
            BooguFloatLoadPolicy::PackedF16WeightsF32Auxiliaries => {
                "packed-f16-weights-f32-auxiliaries"
            }
            BooguFloatLoadPolicy::PackedQ4sWeightsF32Auxiliaries => {
                "packed-q4s-weights-f32-auxiliaries"
            }
        }
        .into(),
        quantized_load_policy: match quantized_policy {
            BooguQuantizedLoadPolicy::Preserve => "preserve",
            BooguQuantizedLoadPolicy::DequantizeF16 => "dequantize-f16",
        }
        .into(),
        execution_dtype: execution_dtype.name().into(),
        input_mode: args.input_mode,
        sigma_source: "fixture-captured".into(),
        gate_basis: gate.basis,
        gate_maximum_relative_rmse: gate.maximum_relative_rmse,
        gate_minimum_cosine_similarity: gate.minimum_cosine,
        width: metadata.width,
        height: metadata.height,
        steps: args.steps,
        boundary_count: boundaries.len(),
        worst_max_abs,
        worst_relative_rmse,
        minimum_cosine_similarity,
        load_and_execute_milliseconds,
        boundaries,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_isolated<B: Backend, S: burn_boogu::StreamingStageSource<B>>(
    denoiser: &mut StreamingBooguDenoiser<B, S>,
    fixture: &SafeTensors<'_>,
    device: &B::Device,
    execution_dtype: DType,
    instruction: Tensor<B, 3>,
    reference: Option<Tensor<B, 4>>,
    steps: u8,
    capture_boundaries: bool,
    boundaries: &mut Vec<BoundaryMetric>,
) -> Result<(), Box<dyn Error>> {
    for step in 0..usize::from(steps) {
        let input = BooguDenoiserInput {
            latent: tensor4::<B>(fixture, &format!("dmd.step.{step}.input"), device)?
                .cast(execution_dtype),
            timestep: tensor1::<B>(fixture, &format!("dmd.step.{step}.sigma"), device)?
                .cast(execution_dtype),
            instruction: instruction.clone(),
            reference: reference.clone(),
        };
        let velocity = predict_observed(
            denoiser,
            fixture,
            step,
            input,
            capture_boundaries,
            boundaries,
        )?;
        let mut observer = FixtureObserver::<B> {
            fixture,
            step,
            metrics: boundaries,
            _backend: core::marker::PhantomData,
        };
        observer.observe("velocity", velocity, &format!("dmd.step.{step}.velocity"))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_trajectory<B: Backend, S: burn_boogu::StreamingStageSource<B>>(
    denoiser: &mut StreamingBooguDenoiser<B, S>,
    fixture: &SafeTensors<'_>,
    device: &B::Device,
    execution_dtype: DType,
    instruction: Tensor<B, 3>,
    reference: Option<Tensor<B, 4>>,
    variant: BooguVariant,
    steps: u8,
    capture_boundaries: bool,
    boundaries: &mut Vec<BoundaryMetric>,
) -> Result<(), Box<dyn Error>> {
    let task = match variant {
        BooguVariant::Image01Turbo => BooguTask::Generate,
        BooguVariant::Image01EditTurbo | BooguVariant::Image01EditTurbo1k5 => BooguTask::Edit,
    };
    let schedule = DmdSchedule::upstream(task);
    let mut latents = tensor4::<B>(fixture, "dmd.initial_latents", device)?.cast(execution_dtype);
    for step in 0..usize::from(steps) {
        let sigma_name = format!("dmd.step.{step}.sigma");
        let sigma = captured_schedule_sigma(fixture, &sigma_name, schedule.sigmas()[step])?;
        {
            let mut observer = FixtureObserver::<B> {
                fixture,
                step,
                metrics: boundaries,
                _backend: core::marker::PhantomData,
            };
            observer.observe(
                "trajectory.input",
                latents.clone(),
                &format!("dmd.step.{step}.input"),
            )?;
        }
        let input = BooguDenoiserInput {
            latent: latents.clone(),
            timestep: tensor1::<B>(fixture, &format!("dmd.step.{step}.sigma"), device)?
                .cast(execution_dtype),
            instruction: instruction.clone(),
            reference: reference.clone(),
        };
        let velocity = predict_observed(
            denoiser,
            fixture,
            step,
            input,
            capture_boundaries,
            boundaries,
        )?;
        let mut observer = FixtureObserver::<B> {
            fixture,
            step,
            metrics: boundaries,
            _backend: core::marker::PhantomData,
        };
        observer.observe(
            "velocity",
            velocity.clone(),
            &format!("dmd.step.{step}.velocity"),
        )?;
        let prediction = dmd_prediction(latents, velocity, sigma);
        let prediction_oracle = if step + 1 == schedule.sigmas().len() {
            "dmd.final_latents".to_owned()
        } else {
            format!("dmd.step.{step}.prediction")
        };
        observer.observe(
            "trajectory.prediction",
            prediction.clone(),
            &prediction_oracle,
        )?;
        if let Some(&expected_next_sigma) = schedule.sigmas().get(step + 1) {
            let next_sigma = captured_schedule_sigma(
                fixture,
                &format!("dmd.step.{}.sigma", step + 1),
                expected_next_sigma,
            )?;
            let noise = tensor4::<B>(fixture, &format!("dmd.step.{step}.noise"), device)?
                .cast(execution_dtype);
            latents = dmd_renoise(prediction, noise, next_sigma);
            observer.observe(
                "trajectory.renoised",
                latents.clone(),
                &format!("dmd.step.{step}.renoised"),
            )?;
        } else {
            latents = prediction;
        }
    }
    if usize::from(steps) == schedule.sigmas().len() {
        let mut observer = FixtureObserver::<B> {
            fixture,
            step: usize::from(steps) - 1,
            metrics: boundaries,
            _backend: core::marker::PhantomData,
        };
        observer.observe("trajectory.final_latents", latents, "dmd.final_latents")?;
    }
    Ok(())
}

fn predict_observed<B: Backend, S: burn_boogu::StreamingStageSource<B>>(
    denoiser: &mut StreamingBooguDenoiser<B, S>,
    fixture: &SafeTensors<'_>,
    step: usize,
    input: BooguDenoiserInput<B>,
    capture_boundaries: bool,
    boundaries: &mut Vec<BoundaryMetric>,
) -> Result<Tensor<B, 4>, BooguError> {
    if capture_boundaries {
        let mut observer = FixtureObserver::<B> {
            fixture,
            step,
            metrics: boundaries,
            _backend: core::marker::PhantomData,
        };
        denoiser.predict_with_observer(input, &mut observer)
    } else {
        burn_boogu::DmdDenoiser::predict(denoiser, input)
    }
}

struct ParityGate {
    basis: String,
    maximum_relative_rmse: f32,
    minimum_cosine: f32,
}

fn parity_gate(args: &Args) -> ParityGate {
    let (basis, default_maximum_relative_rmse, default_minimum_cosine) = match args.input_mode {
        InputMode::Isolated => ("strict-isolated", 0.05, 0.995),
        // This execution-dtype envelope is independently measured between the pinned upstream
        // Edit F16 and BF16 trajectories. It is intentionally distinct from the strict
        // block-local gate and is reported verbatim when applied to either release variant.
        InputMode::Trajectory => (
            "edit-upstream-f16-vs-bf16-dtype-envelope",
            0.265_799_3,
            0.964_623_4,
        ),
    };
    let overridden = args.maximum_relative_rmse.is_some() || args.minimum_cosine.is_some();
    ParityGate {
        basis: if overridden {
            format!("{basis}+cli-override")
        } else {
            basis.to_owned()
        },
        maximum_relative_rmse: args
            .maximum_relative_rmse
            .unwrap_or(default_maximum_relative_rmse),
        minimum_cosine: args.minimum_cosine.unwrap_or(default_minimum_cosine),
    }
}

fn parse_non_negative_f32(raw: &str) -> Result<f32, String> {
    let value = raw
        .parse::<f32>()
        .map_err(|error| format!("invalid floating-point value {raw:?}: {error}"))?;
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err("maximum relative RMSE must be finite and non-negative".into())
    }
}

fn parse_cosine(raw: &str) -> Result<f32, String> {
    let value = raw
        .parse::<f32>()
        .map_err(|error| format!("invalid floating-point value {raw:?}: {error}"))?;
    if value.is_finite() && (-1.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err("minimum cosine must be finite and within [-1, 1]".into())
    }
}

struct FixtureObserver<'a, B: Backend> {
    fixture: &'a SafeTensors<'a>,
    step: usize,
    metrics: &'a mut Vec<BoundaryMetric>,
    _backend: core::marker::PhantomData<B>,
}

impl<B: Backend> FixtureObserver<'_, B> {
    fn observe<const D: usize>(
        &mut self,
        boundary: &str,
        tensor: Tensor<B, D>,
        expected_name: &str,
    ) -> Result<(), BooguError> {
        let started = Instant::now();
        let shape = tensor.dims().to_vec();
        let actual = tensor
            .to_data()
            .convert_dtype(DType::F32)
            .to_vec::<f32>()
            .map_err(|error| BooguError::Artifact(error.to_string()))?;
        let view = self
            .fixture
            .tensor(expected_name)
            .map_err(|error| BooguError::Artifact(error.to_string()))?;
        if view.shape() != shape {
            return Err(BooguError::InvalidShape(format!(
                "boundary {expected_name} expected {:?}, Burn produced {shape:?}",
                view.shape()
            )));
        }
        let expected = decode_f32(&view).map_err(BooguError::Artifact)?;
        let comparison = compare(&actual, &expected).map_err(BooguError::Artifact)?;
        self.metrics.push(BoundaryMetric {
            name: format!("denoiser.step.{}.{}", self.step, boundary),
            shape,
            max_abs: comparison.max_abs,
            mean_abs: comparison.mean_abs,
            rmse: comparison.rmse,
            relative_rmse: comparison.relative_rmse,
            cosine_similarity: comparison.cosine,
            readback_milliseconds: started.elapsed().as_secs_f64() * 1_000.0,
        });
        Ok(())
    }
}

impl<B: Backend> DenoiserStageObserver<B> for FixtureObserver<'_, B> {
    fn rank2(&mut self, name: &str, tensor: Tensor<B, 2>) -> Result<(), BooguError> {
        self.observe(
            name,
            tensor,
            &format!("denoiser.step.{}.{}", self.step, name),
        )
    }

    fn rank3(&mut self, name: &str, tensor: Tensor<B, 3>) -> Result<(), BooguError> {
        self.observe(
            name,
            tensor,
            &format!("denoiser.step.{}.{}", self.step, name),
        )
    }

    fn rank4(&mut self, name: &str, tensor: Tensor<B, 4>) -> Result<(), BooguError> {
        self.observe(
            name,
            tensor,
            &format!("denoiser.step.{}.{}", self.step, name),
        )
    }
}

fn tensor1<B: Backend>(
    fixture: &SafeTensors<'_>,
    name: &str,
    device: &B::Device,
) -> Result<Tensor<B, 1>, Box<dyn Error>> {
    let view = fixture.tensor(name)?;
    let values = decode_f32(&view)?;
    Ok(Tensor::from_data(TensorData::new(values, [1]), device))
}

fn scalar_f32(fixture: &SafeTensors<'_>, name: &str) -> Result<f32, Box<dyn Error>> {
    let view = fixture.tensor(name)?;
    let values = decode_f32(&view)?;
    match values.as_slice() {
        [value] => Ok(*value),
        _ => Err(format!("{name} must contain exactly one scalar").into()),
    }
}

fn captured_schedule_sigma(
    fixture: &SafeTensors<'_>,
    name: &str,
    expected: f32,
) -> Result<f32, Box<dyn Error>> {
    let view = fixture.tensor(name)?;
    let captured = scalar_f32(fixture, name)?;
    let (expected_at_fixture_dtype, tolerance) = match view.dtype() {
        Dtype::BF16 => {
            let rounded = bf16::from_f32(expected);
            let next = bf16::from_bits(rounded.to_bits() + 1);
            (rounded.to_f32(), next.to_f32() - rounded.to_f32())
        }
        Dtype::F16 => {
            let rounded = f16::from_f32(expected);
            let next = f16::from_bits(rounded.to_bits() + 1);
            (rounded.to_f32(), next.to_f32() - rounded.to_f32())
        }
        Dtype::F32 => (expected, f32::EPSILON * expected.abs().max(1.0)),
        dtype => return Err(format!("{name} has unsupported schedule dtype {dtype:?}").into()),
    };
    // Upstream constructs the schedule directly with torch.linspace at the latent dtype. Its
    // low-precision interpolation can land one representable value away from computing the
    // mathematical F32 schedule first and casting afterward (Turbo BF16 step 1 is the concrete
    // case). Bound validation to that single fixture-dtype ULP, then replay the exact capture.
    if (captured - expected_at_fixture_dtype).abs() > tolerance {
        return Err(format!(
            "captured {name} value {captured} differs by more than one {:?} ULP from schedule {expected} ({expected_at_fixture_dtype} after rounding)",
            view.dtype()
        )
        .into());
    }
    Ok(captured)
}

fn tensor3<B: Backend>(
    fixture: &SafeTensors<'_>,
    name: &str,
    device: &B::Device,
) -> Result<Tensor<B, 3>, Box<dyn Error>> {
    let view = fixture.tensor(name)?;
    let shape: [usize; 3] = view
        .shape()
        .try_into()
        .map_err(|_| format!("{name} is not rank three"))?;
    Ok(Tensor::from_data(
        TensorData::new(decode_f32(&view)?, shape),
        device,
    ))
}

fn tensor4<B: Backend>(
    fixture: &SafeTensors<'_>,
    name: &str,
    device: &B::Device,
) -> Result<Tensor<B, 4>, Box<dyn Error>> {
    let view = fixture.tensor(name)?;
    let shape: [usize; 4] = view
        .shape()
        .try_into()
        .map_err(|_| format!("{name} is not rank four"))?;
    Ok(Tensor::from_data(
        TensorData::new(decode_f32(&view)?, shape),
        device,
    ))
}

fn decode_f32(view: &TensorView<'_>) -> Result<Vec<f32>, String> {
    let bytes = view.data();
    match view.dtype() {
        Dtype::F32 => Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
            .collect()),
        Dtype::F16 => Ok(bytes
            .chunks_exact(2)
            .map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect()),
        Dtype::BF16 => Ok(bytes
            .chunks_exact(2)
            .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect()),
        dtype => Err(format!("unsupported fixture dtype {dtype:?}")),
    }
}

struct Comparison {
    max_abs: f32,
    mean_abs: f32,
    rmse: f32,
    relative_rmse: f32,
    cosine: f32,
}

fn compare(actual: &[f32], expected: &[f32]) -> Result<Comparison, String> {
    if actual.len() != expected.len() || actual.is_empty() {
        return Err(format!(
            "comparison length mismatch: actual={} expected={}",
            actual.len(),
            expected.len()
        ));
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
    Ok(Comparison {
        max_abs,
        mean_abs: (sum_abs / count) as f32,
        rmse: rmse as f32,
        relative_rmse: (rmse / expected_rms.max(f64::MIN_POSITIVE)) as f32,
        cosine: if denominator == 0.0 {
            1.0
        } else {
            (dot / denominator) as f32
        },
    })
}

fn require_parity(report: &Report) -> Result<(), Box<dyn Error>> {
    let failures = report
        .boundaries
        .iter()
        .filter(|metric| {
            metric.relative_rmse > report.gate_maximum_relative_rmse
                || metric.cosine_similarity < report.gate_minimum_cosine_similarity
        })
        .map(|metric| {
            format!(
                "{} (relative RMSE {}, cosine {})",
                metric.name, metric.relative_rmse, metric.cosine_similarity
            )
        })
        .take(16)
        .collect::<Vec<_>>();
    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!("denoiser parity gate failed: {}", failures.join(", ")).into())
    }
}
