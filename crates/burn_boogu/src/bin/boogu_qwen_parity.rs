//! Opt-in real-artifact parity for the row-streamed Qwen3-VL conditioning path.

use std::{
    error::Error,
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    time::Instant,
};

use burn::{
    prelude::Backend,
    tensor::{DType, Tensor, TensorData},
};
use burn_boogu::{
    BooguConfig, BooguVariant,
    artifacts::{
        BooguArtifactInventory, BooguQuantizedLoadPolicy, BooguReleaseIdentity,
        BooguStorageProfile, VerifiedArtifactDirectory, VerifiedBurnpackQwenStageSource,
    },
    reference::verify_reference_fixture_file,
    require_native_wgpu_device,
};
use burn_flux_vae::AutoencoderKlConfig;
use burn_qwen3_vl::{
    BatchEncoding, Grid, Qwen3VlConfig, Qwen3VlError, Qwen3VlModelInput, Qwen3VlStage,
    Qwen3VlStageObserver, Qwen3VlVisualInput, StreamingQwen3Vl,
};
use clap::{Parser, ValueEnum};
use half::{bf16, f16};
use safetensors::{Dtype, tensor::Metadata};
use serde::{Deserialize, Serialize};

type WgpuBackend = burn_wgpu::Wgpu<f32, i32, u32>;

const ARTIFACT_ENV: &str = "BURN_BOOGU_ARTIFACT_DIR";
const FIXTURE_ENV: &str = "BURN_BOOGU_QWEN_FIXTURE_DIR";

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ProfileChoice {
    F16,
    F16QwenVisionF32,
    Q8sBlock32F32,
    Q8sBlock32F32QwenVisionF32,
}

impl ProfileChoice {
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

#[derive(Debug, Clone, Copy, ValueEnum)]
enum QuantizedLoadChoice {
    Auto,
    Preserve,
    DequantizeF16,
}

impl QuantizedLoadChoice {
    fn resolve(self, profile: ProfileChoice) -> Result<BooguQuantizedLoadPolicy, Box<dyn Error>> {
        let quantized = matches!(
            profile,
            ProfileChoice::Q8sBlock32F32 | ProfileChoice::Q8sBlock32F32QwenVisionF32
        );
        match (self, quantized) {
            (Self::Auto | Self::DequantizeF16, true) => {
                Ok(BooguQuantizedLoadPolicy::DequantizeF16)
            }
            (Self::Auto | Self::Preserve, false) => Ok(BooguQuantizedLoadPolicy::Preserve),
            (Self::Preserve, true) => Err(
                "Qwen Q8 Preserve is rejected: Burn 0.21 corrupts block scales in the Col load mapper; use dequantize-f16"
                    .into(),
            ),
            (Self::DequantizeF16, false) => {
                Err("dequantize-f16 requires a Q8 storage profile".into())
            }
        }
    }
}

const fn quantized_policy_slug(policy: BooguQuantizedLoadPolicy) -> &'static str {
    match policy {
        BooguQuantizedLoadPolicy::Preserve => "preserve",
        BooguQuantizedLoadPolicy::DequantizeF16 => "dequantize-f16",
    }
}

#[derive(Debug, Parser)]
#[command(about = "Compare streamed Burn Qwen3-VL with a pinned schema-2 oracle on WGPU")]
struct Args {
    /// Sealed importer output. Falls back to BURN_BOOGU_ARTIFACT_DIR.
    #[arg(long)]
    artifacts: Option<PathBuf>,
    /// Reference directory containing metadata.json and tensors.safetensors. Falls back to
    /// BURN_BOOGU_QWEN_FIXTURE_DIR.
    #[arg(long)]
    fixture: Option<PathBuf>,
    /// Converted storage profile; it must exactly match the sealed manifest.
    #[arg(long, value_enum, default_value = "f16-qwen-vision-f32")]
    profile: ProfileChoice,
    /// Q8 snapshot application policy. Q8 profiles require host dequantization before Col mapping.
    #[arg(long, value_enum, default_value = "auto")]
    quantized_load_policy: QuantizedLoadChoice,
    /// Query tile bound used by both streamed vision and text attention.
    #[arg(long, default_value_t = 128)]
    query_chunk_size: usize,
    /// Read back and compare every semantic stage in addition to the final hidden state.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    capture_stages: bool,
    /// Fail when a captured metric exceeds the supplied numerical gates.
    #[arg(long, default_value_t = false)]
    require: bool,
    /// Minimum accepted cosine similarity for `--require`.
    #[arg(long, default_value_t = 0.99)]
    minimum_cosine: f32,
    /// Maximum accepted relative RMSE for `--require`.
    #[arg(long, default_value_t = 0.2)]
    maximum_relative_rmse: f32,
}

#[derive(Debug, Deserialize)]
struct FixtureMetadata {
    schema_version: u32,
    variant: String,
    model_revision: String,
    upstream_source_revision: String,
    #[serde(default)]
    capture_qwen: bool,
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
}

#[derive(Debug, Clone, Serialize)]
struct ParityMetric {
    name: String,
    shape: Vec<usize>,
    max_abs: f32,
    mean_abs: f32,
    rmse: f32,
    relative_rmse: f32,
    cosine_similarity: f32,
    readback_milliseconds: f64,
}

#[derive(Debug, Clone, Serialize)]
struct DtypeOnlyStage {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    reason: String,
}

#[derive(Debug, Serialize)]
struct Report {
    variant: String,
    model_revision: String,
    upstream_source_revision: String,
    artifact_content_digest: String,
    backend: String,
    profile: String,
    quantized_load_policy: String,
    qwen_text_execution_dtype: String,
    query_chunk_size: usize,
    capture_stages: bool,
    stage_count: usize,
    compared_stage_count: usize,
    dtype_only_stage_count: usize,
    load_and_execute_milliseconds: f64,
    minimum_cosine_similarity: f32,
    worst_relative_rmse: f32,
    final_hidden_state: ParityMetric,
    stages: Vec<ParityMetric>,
    dtype_only_stages: Vec<DtypeOnlyStage>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let artifacts = required_path(args.artifacts, ARTIFACT_ENV)?;
    let fixture_path = required_path(args.fixture, FIXTURE_ENV)?;
    if args.query_chunk_size == 0 {
        return Err("--query-chunk-size must be non-zero".into());
    }
    if !(0.0..=1.0).contains(&args.minimum_cosine)
        || !args.maximum_relative_rmse.is_finite()
        || args.maximum_relative_rmse < 0.0
    {
        return Err("invalid numerical parity gates".into());
    }

    let fixture_metadata_bytes = fs::read(fixture_path.join("metadata.json"))?;
    let metadata: FixtureMetadata = serde_json::from_slice(&fixture_metadata_bytes)?;
    if metadata.schema_version != 2 || !metadata.capture_qwen {
        return Err(
            "Qwen parity requires a schema-2 fixture captured with capture_qwen=true".into(),
        );
    }
    let variant = metadata.variant()?;
    let identity = BooguReleaseIdentity::canonical(variant);
    if metadata.model_revision != identity.model_revision
        || metadata.upstream_source_revision != identity.upstream_source_revision
    {
        return Err("fixture revisions do not match the canonical immutable release".into());
    }
    verify_reference_fixture_file(
        &fixture_metadata_bytes,
        fixture_path.join("tensors.safetensors"),
    )?;

    let artifact_directory = VerifiedArtifactDirectory::open(&artifacts)?;
    let manifest = artifact_directory.manifest();
    let artifact_content_digest = manifest
        .content_digest
        .ok_or("sealed artifact manifest has no content digest")?
        .to_string();
    let qwen_config = Qwen3VlConfig::from_json(
        &artifact_directory.read_text("metadata/source/mllm/config.json")?,
    )?;
    let vae_config = AutoencoderKlConfig::from_diffusers_json(
        &artifact_directory.read_text("metadata/source/vae/config.json")?,
    )?;
    let inventory =
        BooguArtifactInventory::new(&qwen_config, &BooguConfig::default(), &vae_config)?;
    let fixture = FixtureStore::open(fixture_path.join("tensors.safetensors"))?;
    let device = require_native_wgpu_device()?;
    let quantized_load_policy = args.quantized_load_policy.resolve(args.profile)?;
    let source = VerifiedBurnpackQwenStageSource::<WgpuBackend, _>::from_directory_auto(
        &identity,
        &artifacts,
        inventory,
        qwen_config.clone(),
        args.profile.into(),
        device.clone(),
    )?
    .with_quantized_load_policy(quantized_load_policy);
    let plan = source.plan().clone();
    let mut qwen = StreamingQwen3Vl::new(plan, source);
    qwen.set_query_chunk_size(args.query_chunk_size);
    let input = fixture_input::<WgpuBackend>(&fixture, &qwen_config, &device)?;
    let started = Instant::now();
    let mut observer = FixtureObserver::<WgpuBackend> {
        fixture: &fixture,
        capture: args.capture_stages,
        multimodal: fixture.contains("processor.pixel_values"),
        deepstack_count: qwen_config.vision_config.deepstack_visual_indexes.len(),
        metrics: Vec::new(),
        dtype_only: Vec::new(),
        _backend: core::marker::PhantomData,
    };
    let output = qwen
        .forward_base(&qwen_config, input, &mut observer)
        .map_err(|error| format!("streamed Qwen forward failed: {error:?}"))?;
    let final_hidden_state =
        compare_tensor(&fixture, "qwen.last_hidden_state", output.last_hidden_state)?;
    <WgpuBackend as Backend>::sync(&device)?;
    let load_and_execute_milliseconds = started.elapsed().as_secs_f64() * 1_000.0;
    let minimum_cosine_similarity = observer
        .metrics
        .iter()
        .chain(core::iter::once(&final_hidden_state))
        .map(|metric| metric.cosine_similarity)
        .fold(1.0_f32, f32::min);
    let worst_relative_rmse = observer
        .metrics
        .iter()
        .chain(core::iter::once(&final_hidden_state))
        .map(|metric| metric.relative_rmse)
        .fold(0.0_f32, f32::max);
    let report = Report {
        variant: metadata.variant,
        model_revision: metadata.model_revision,
        upstream_source_revision: metadata.upstream_source_revision,
        artifact_content_digest,
        backend: "burn-wgpu".into(),
        profile: args.profile.slug().into(),
        quantized_load_policy: quantized_policy_slug(quantized_load_policy).into(),
        qwen_text_execution_dtype: "F16".into(),
        query_chunk_size: args.query_chunk_size,
        capture_stages: args.capture_stages,
        stage_count: observer.metrics.len() + observer.dtype_only.len(),
        compared_stage_count: observer.metrics.len(),
        dtype_only_stage_count: observer.dtype_only.len(),
        load_and_execute_milliseconds,
        minimum_cosine_similarity,
        worst_relative_rmse,
        final_hidden_state,
        stages: observer.metrics,
        dtype_only_stages: observer.dtype_only,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    if args.require {
        require_parity(&report, args.minimum_cosine, args.maximum_relative_rmse)?;
    }
    Ok(())
}

fn required_path(value: Option<PathBuf>, environment: &str) -> Result<PathBuf, Box<dyn Error>> {
    value
        .or_else(|| std::env::var_os(environment).map(PathBuf::from))
        .ok_or_else(|| format!("provide the path argument or set {environment}").into())
}

fn fixture_input<B: Backend>(
    fixture: &FixtureStore,
    config: &Qwen3VlConfig,
    device: &B::Device,
) -> Result<Qwen3VlModelInput<B>, Box<dyn Error>> {
    let (input_shape, input_ids) = fixture.i64("processor.input_ids")?;
    let [batch, sequence]: [usize; 2] = input_shape
        .try_into()
        .map_err(|_| "processor.input_ids must be rank two")?;
    let (mask_shape, attention) = fixture.i64("processor.attention_mask")?;
    let (type_shape, token_types) = fixture.i64("processor.mm_token_type_ids")?;
    if mask_shape != [batch, sequence] || type_shape != [batch, sequence] {
        return Err("processor mask/type tensors do not match input_ids".into());
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
    if batch != 1 {
        return Err("the released parity fixtures must contain one processor sample".into());
    }
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
            patches: Tensor::from_data(TensorData::new(values, shape), device),
            grids,
            token_indices: encoding.flattened_image_token_indices(),
        })
    } else {
        None
    };
    Ok(Qwen3VlModelInput {
        input_ids: tensors.input_ids,
        attention_mask: Some(tensors.attention_mask),
        position_ids: Some(position_ids),
        images,
        videos: None,
        output_hidden_states: false,
    })
}

struct FixtureObserver<'a, B: Backend> {
    fixture: &'a FixtureStore,
    capture: bool,
    multimodal: bool,
    deepstack_count: usize,
    metrics: Vec<ParityMetric>,
    dtype_only: Vec<DtypeOnlyStage>,
    _backend: core::marker::PhantomData<B>,
}

impl<B: Backend> FixtureObserver<'_, B> {
    fn observe<const D: usize>(
        &mut self,
        stage: &Qwen3VlStage,
        tensor: Tensor<B, D>,
    ) -> burn_qwen3_vl::Result<()> {
        if !self.capture {
            return Ok(());
        }
        let mapping = stage_oracle(self.fixture, stage, self.multimodal, self.deepstack_count);
        let (name, dtype_only_reason) = match mapping {
            OracleMapping::Compare(name) => (name, None),
            OracleMapping::DtypeOnly { name, reason } => (name, Some(reason)),
        };
        eprintln!(
            "observing {name}: dtype={:?}, shape={:?}",
            tensor.dtype(),
            tensor.dims()
        );
        if let Some(reason) = dtype_only_reason {
            self.dtype_only.push(DtypeOnlyStage {
                name,
                dtype: format!("{:?}", tensor.dtype()),
                shape: tensor.dims().to_vec(),
                reason,
            });
            return Ok(());
        }
        let metric = compare_tensor(self.fixture, &name, tensor)
            .map_err(|error| Qwen3VlError::Checkpoint(error.to_string()))?;
        self.metrics.push(metric);
        Ok(())
    }
}

impl<B: Backend> Qwen3VlStageObserver<B> for FixtureObserver<'_, B> {
    fn rank2(
        &mut self,
        stage: &Qwen3VlStage,
        activation: Tensor<B, 2>,
    ) -> burn_qwen3_vl::Result<()> {
        self.observe(stage, activation)
    }

    fn rank3(
        &mut self,
        stage: &Qwen3VlStage,
        activation: Tensor<B, 3>,
    ) -> burn_qwen3_vl::Result<()> {
        self.observe(stage, activation)
    }
}

enum OracleMapping {
    Compare(String),
    DtypeOnly { name: String, reason: String },
}

fn stage_oracle(
    fixture: &FixtureStore,
    stage: &Qwen3VlStage,
    multimodal: bool,
    deepstack_count: usize,
) -> OracleMapping {
    match stage {
        Qwen3VlStage::EmbeddingRows { .. } => {
            OracleMapping::Compare("qwen.text.token_embeddings".into())
        }
        Qwen3VlStage::VisionPrelude => optional_aligned_oracle(
            fixture,
            "qwen.vision.prelude",
            "streamed prelude is post learned-position addition; the source qwen.vision.patch_embed capture is raw Conv3d output",
        ),
        Qwen3VlStage::VisionBlock { index } => {
            OracleMapping::Compare(format!("qwen.vision.block.{index}"))
        }
        Qwen3VlStage::VisionDeepstackMerger { index, .. } => {
            OracleMapping::Compare(format!("qwen.vision.deepstack_merger.{index}"))
        }
        Qwen3VlStage::VisionFinalMerger => {
            OracleMapping::Compare("qwen.vision.final_merger".into())
        }
        Qwen3VlStage::TextBlock { index } if multimodal && *index < deepstack_count => {
            optional_aligned_oracle(
                fixture,
                &format!("qwen.text.layer.{index}.post_deepstack"),
                "streamed observer is post deepstack insertion; the source layer hook is pre insertion",
            )
        }
        Qwen3VlStage::TextBlock { index } => {
            OracleMapping::Compare(format!("qwen.text.layer.{index}"))
        }
        Qwen3VlStage::TextFinalNorm => OracleMapping::Compare("qwen.text.final_norm".into()),
        Qwen3VlStage::LmHeadRows { chunk } => OracleMapping::DtypeOnly {
            name: format!("qwen.lm_head.rows.{chunk}"),
            reason: "base-model parity does not execute the optional LM head".into(),
        },
    }
}

fn optional_aligned_oracle(
    fixture: &FixtureStore,
    name: &str,
    missing_reason: &str,
) -> OracleMapping {
    if fixture.contains(name) {
        OracleMapping::Compare(name.into())
    } else {
        OracleMapping::DtypeOnly {
            name: name.into(),
            reason: missing_reason.into(),
        }
    }
}

fn compare_tensor<B: Backend, const D: usize>(
    fixture: &FixtureStore,
    oracle: &str,
    tensor: Tensor<B, D>,
) -> Result<ParityMetric, Box<dyn Error>> {
    let started = Instant::now();
    let shape = tensor.dims().to_vec();
    let actual = tensor.to_data().convert_dtype(DType::F32).to_vec::<f32>()?;
    let (expected_shape, expected) = fixture.f32(oracle)?;
    if shape != expected_shape {
        return Err(format!(
            "oracle {oracle} has shape {expected_shape:?}, Burn produced {shape:?}"
        )
        .into());
    }
    let comparison = compare(&actual, &expected)?;
    Ok(ParityMetric {
        name: oracle.into(),
        shape,
        max_abs: comparison.max_abs,
        mean_abs: comparison.mean_abs,
        rmse: comparison.rmse,
        relative_rmse: comparison.relative_rmse,
        cosine_similarity: comparison.cosine,
        readback_milliseconds: started.elapsed().as_secs_f64() * 1_000.0,
    })
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
            return Err("invalid SafeTensors header length".into());
        }
        let mut header = vec![0_u8; header_len];
        file.read_exact(&mut header)?;
        let metadata: Metadata = serde_json::from_slice(&header)?;
        let data_start = 8_u64
            .checked_add(u64::try_from(header_len)?)
            .ok_or("SafeTensors data offset overflow")?;
        let expected_size = data_start
            .checked_add(u64::try_from(metadata.data_len())?)
            .ok_or("SafeTensors file size overflow")?;
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
        let values = match tensor.dtype {
            Dtype::F32 => tensor
                .bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
                .collect(),
            Dtype::F16 => tensor
                .bytes
                .chunks_exact(2)
                .map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
                .collect(),
            Dtype::BF16 => tensor
                .bytes
                .chunks_exact(2)
                .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
                .collect(),
            dtype => return Err(format!("fixture tensor {name} has unsupported {dtype:?}").into()),
        };
        Ok((tensor.shape, values))
    }
}

struct FixtureTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

struct Comparison {
    max_abs: f32,
    mean_abs: f32,
    rmse: f32,
    relative_rmse: f32,
    cosine: f32,
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

fn require_parity(
    report: &Report,
    minimum_cosine: f32,
    maximum_relative_rmse: f32,
) -> Result<(), Box<dyn Error>> {
    let failures = report
        .stages
        .iter()
        .chain(core::iter::once(&report.final_hidden_state))
        .filter(|metric| {
            metric.cosine_similarity < minimum_cosine
                || metric.relative_rmse > maximum_relative_rmse
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
        Err(format!("streamed Qwen parity gate failed: {}", failures.join(", ")).into())
    }
}
