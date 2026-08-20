//! Streaming SafeTensors-to-Burnpack converter for the released Boogu checkpoints.

use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs::{self, File},
    io::{BufReader, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use burn::{
    module::ParamId,
    tensor::{
        DType, TensorData,
        quantization::{QuantLevel, QuantParam, QuantValue},
    },
};
use burn_boogu::{
    BooguConfig,
    artifacts::{
        ArtifactTensorSpec, BooguArtifactInventory, EDIT_TURBO_1K5_REVISION, EDIT_TURBO_REVISION,
        TURBO_REVISION, TensorOwner, TensorTransform, UPSTREAM_SOURCE_REVISION,
        q4s_storage_block_and_axis, q4s_stored_dtype, quantize_q4s_block128_f32,
        quantize_q8s_block32_f32, quantize_row_layout_q4s_block_up_to128_f32,
        qwen_row_slice_target, qwen_streaming_stage_name,
    },
};
use burn_flux_vae::AutoencoderKlConfig;
use burn_image::{
    ARTIFACT_MANIFEST_SCHEMA_V1, ArtifactBundleId, ArtifactComponent, ArtifactComponentId,
    ArtifactFile, ArtifactFileRole, ArtifactManifest, ArtifactPath, ArtifactProfileId,
    ArtifactShard, ModelId, NumericFormat, Sha256Digest,
};
use burn_qwen3_vl::{Qwen3VlConfig, Qwen3VlStage, Qwen3VlStreamingPlan, RowChunkPlan};
use burn_store::{BurnpackStore, BurnpackWriter, ModuleStore, TensorSnapshot};
use clap::{Parser, ValueEnum};
use half::{bf16, f16};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const DEFAULT_SHARD_MIB: u64 = 256;
const MAX_METADATA_FILE_BYTES: u64 = 64 * 1024 * 1024;
const BURNPACK_ROW_SLICE_RESERVE_BYTES: u64 = 1024 * 1024;

#[derive(Debug, Parser)]
#[command(about = "Convert a pinned Boogu Hugging Face snapshot to verified Burnpack shards")]
struct Args {
    /// Source model directory downloaded at the immutable revision for `variant`.
    #[arg(long)]
    source: PathBuf,
    /// New destination release directory. Existing directories are refused.
    #[arg(long)]
    output: PathBuf,
    /// Released checkpoint variant.
    #[arg(long, value_enum)]
    variant: VariantArg,
    /// Numeric storage profile.
    #[arg(long, value_enum, default_value_t = ProfileArg::F16QwenVisionF32)]
    profile: ProfileArg,
    /// Target maximum payload size per logical Burnpack object, in MiB.
    #[arg(long, default_value_t = DEFAULT_SHARD_MIB)]
    max_shard_mib: u64,
    /// Permit a tensor larger than the target limit to occupy one explicitly declared native-only
    /// shard. This diagnostic escape hatch is never needed by the production row-sliced plan.
    #[arg(long)]
    allow_oversized_tensors: bool,
    /// Include the streamed causal-LM vocabulary projection. Boogu conditioning does not need it.
    #[arg(long)]
    include_lm_head: bool,
    /// Convert just one qualified source key (for example
    /// `transformer:context_refiner.0.attn.norm_q.weight`) and do not seal a release manifest.
    #[arg(long)]
    smoke_tensor: Option<String>,
    /// With `--smoke-tensor` on a Qwen vocabulary table, convert only this bounded row chunk.
    #[arg(long, requires = "smoke_tensor")]
    smoke_row_chunk: Option<usize>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum VariantArg {
    Image01Turbo,
    Image01EditTurbo,
    #[value(name = "image01-edit-turbo-1k5")]
    Image01EditTurbo1k5,
}

impl VariantArg {
    fn revision(self) -> &'static str {
        match self {
            Self::Image01Turbo => TURBO_REVISION,
            Self::Image01EditTurbo => EDIT_TURBO_REVISION,
            Self::Image01EditTurbo1k5 => EDIT_TURBO_1K5_REVISION,
        }
    }

    fn model_id(self) -> &'static str {
        match self {
            Self::Image01Turbo => "Boogu/Boogu-Image-0.1-Turbo",
            Self::Image01EditTurbo => "Boogu/Boogu-Image-0.1-Edit-Turbo",
            Self::Image01EditTurbo1k5 => "Boogu/Boogu-Image-0.1-Edit-Turbo-1K5",
        }
    }

    fn slug(self) -> &'static str {
        match self {
            Self::Image01Turbo => "boogu-image-0.1-turbo",
            Self::Image01EditTurbo => "boogu-image-0.1-edit-turbo",
            Self::Image01EditTurbo1k5 => "boogu-image-0.1-edit-turbo-1k5",
        }
    }

    fn upstream_model_id(self) -> &'static str {
        match self {
            Self::Image01Turbo => "Boogu/Boogu-Image-0.1-Turbo",
            Self::Image01EditTurbo | Self::Image01EditTurbo1k5 => {
                "Boogu/Boogu-Image-0.1-Edit-Turbo"
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ProfileArg {
    F16,
    F16QwenVisionF32,
    Q8sBlock32F32,
    Q8sBlock32F32QwenVisionF32,
    Q4sBlockUpTo128F32,
}

impl ProfileArg {
    fn slug(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::F16QwenVisionF32 => "f16-qwen-vision-f32",
            Self::Q8sBlock32F32 => "q8s-block32-f32",
            Self::Q8sBlock32F32QwenVisionF32 => "q8s-block32-f32-qwen-vision-f32",
            Self::Q4sBlockUpTo128F32 => "q4s-block-up-to128-f32",
        }
    }

    fn numeric_format(self) -> NumericFormat {
        match self {
            Self::F16 => NumericFormat::F16,
            Self::F16QwenVisionF32
            | Self::Q8sBlock32F32
            | Self::Q8sBlock32F32QwenVisionF32
            | Self::Q4sBlockUpTo128F32 => NumericFormat::Other(self.slug().to_owned()),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct HeaderTensor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

#[derive(Debug, Deserialize)]
struct UpstreamInstructionConfig {
    instruction_feat_dim: usize,
    num_instruction_feature_layers: usize,
    reduce_type: String,
}

#[derive(Debug, Deserialize)]
struct UpstreamDenoiserConfig {
    patch_size: usize,
    in_channels: usize,
    out_channels: Option<usize>,
    hidden_size: usize,
    num_layers: usize,
    num_double_stream_layers: usize,
    num_refiner_layers: usize,
    num_attention_heads: usize,
    num_kv_heads: usize,
    multiple_of: usize,
    norm_eps: f64,
    axes_dim_rope: [usize; 3],
    axes_lens: [usize; 3],
    instruction_feature_configs: UpstreamInstructionConfig,
    timestep_scale: f64,
}

#[derive(Debug, Clone, Serialize)]
struct InventoryTensor {
    source_name: String,
    logical_target_name: String,
    target_name: String,
    owner: TensorOwner,
    component: String,
    stage: String,
    transform: TensorTransform,
    source_file: String,
    source_dtype: String,
    source_shape: Vec<usize>,
    source_row_range: Option<[usize; 2]>,
    included: bool,
    stored_dtype: Option<String>,
    stored_shape: Option<Vec<usize>>,
    source_offset: u64,
    source_bytes: u64,
    quantized: bool,
    stored_sha256: Option<Sha256Digest>,
    burnpack_object: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct SourceFileRecord {
    path: String,
    size: u64,
    sha256: Sha256Digest,
}

#[derive(Debug, Clone)]
struct SourceTensor {
    name: String,
    component: String,
    owner: TensorOwner,
    stage: String,
    file: PathBuf,
    relative_file: String,
    dtype: String,
    full_source_shape: Vec<usize>,
    shape: Vec<usize>,
    source_row_range: Option<[usize; 2]>,
    absolute_offset: u64,
    bytes: u64,
    target_name: String,
    logical_target_name: String,
    target_shape: Vec<usize>,
    transform: TensorTransform,
    quantizable: bool,
}

#[derive(Debug)]
struct PlannedShard {
    component: String,
    tensors: Vec<SourceTensor>,
}

#[derive(Debug)]
struct WrittenShard {
    component: String,
    path: ArtifactPath,
    size: u64,
    sha256: Sha256Digest,
    tensors: Vec<InventoryTensor>,
}

type PreparedStreamingTensors = (
    Vec<SourceTensor>,
    Vec<InventoryTensor>,
    Qwen3VlStreamingPlan,
);

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if args.max_shard_mib == 0 {
        return Err("--max-shard-mib must be non-zero".into());
    }
    let max_bytes = args
        .max_shard_mib
        .checked_mul(1024 * 1024)
        .ok_or("shard size overflow")?;
    validate_source_identity(&args.source, args.variant)?;
    if args.output.exists() {
        return Err(format!(
            "output already exists; refusing to merge or overwrite: {}",
            args.output.display()
        )
        .into());
    }
    let qwen_config =
        Qwen3VlConfig::from_json(&fs::read_to_string(args.source.join("mllm/config.json"))?)?;
    let expected = load_release_inventory(&args.source)?;
    let mut tensors = inventory_snapshot(&args.source)?;
    validate_inventory(&mut tensors, &expected)?;
    validate_weight_indexes(&args.source, &tensors)?;
    if let Some(smoke_tensor) = &args.smoke_tensor {
        fs::create_dir_all(args.output.join("objects"))?;
        fs::create_dir_all(args.output.join("metadata"))?;
        let tensor = if let Some(chunk_index) = args.smoke_row_chunk {
            let include_lm_head = args.include_lm_head || smoke_tensor == "mllm:lm_head.weight";
            let (prepared, _, _) =
                prepare_streaming_tensors(tensors, &qwen_config, max_bytes, include_lm_head)?;
            prepared
                .into_iter()
                .find(|tensor| {
                    format!("{}:{}", tensor.component, tensor.name) == *smoke_tensor
                        && tensor
                            .target_name
                            .contains(&format!(".rows.{chunk_index:02}."))
                })
                .ok_or_else(|| {
                    format!("--smoke-row-chunk {chunk_index} does not exist for {smoke_tensor}")
                })?
        } else {
            tensors
                .into_iter()
                .find(|tensor| format!("{}:{}", tensor.component, tensor.name) == *smoke_tensor)
                .ok_or_else(|| format!("unknown --smoke-tensor {smoke_tensor}"))?
        };
        let shard = PlannedShard {
            component: tensor.stage.clone(),
            tensors: vec![tensor],
        };
        let written = write_shard(
            &args.output,
            shard,
            args.profile,
            max_bytes,
            args.allow_oversized_tensors,
        )?;
        let bytes = serde_json::to_vec_pretty(&written.tensors)?;
        write_metadata_object(
            &args.output,
            "metadata/smoke-tensor.json",
            &bytes,
            ArtifactFileRole::Metadata,
        )?;
        eprintln!(
            "real-checkpoint smoke converted {} into {} ({} bytes); no release manifest was written",
            smoke_tensor, written.path, written.size
        );
        return Ok(());
    }
    let source_files = hash_source_files(&args.source, &tensors)?;
    let (mut tensors, omitted_inventory, qwen_plan) =
        prepare_streaming_tensors(tensors, &qwen_config, max_bytes, args.include_lm_head)?;
    tensors.sort_by(|left, right| {
        (&left.stage, &left.name, &left.target_name).cmp(&(
            &right.stage,
            &right.name,
            &right.target_name,
        ))
    });
    let shards = plan_shards(tensors.clone(), args.profile, max_bytes);
    validate_shard_plan(
        &shards,
        args.profile,
        max_bytes,
        args.allow_oversized_tensors,
    )?;
    fs::create_dir_all(args.output.join("objects"))?;
    fs::create_dir_all(args.output.join("metadata"))?;

    eprintln!(
        "converting {} tensors from {} pinned source files into {} semantic shards ({})",
        expected.tensors().len(),
        source_files.len(),
        shards.len(),
        args.profile.slug(),
    );
    let mut written = Vec::with_capacity(shards.len());
    for (index, shard) in shards.into_iter().enumerate() {
        eprintln!(
            "[{}/{}] {} ({} tensors)",
            index + 1,
            written.capacity(),
            shard.component,
            shard.tensors.len()
        );
        written.push(write_shard(
            &args.output,
            shard,
            args.profile,
            max_bytes,
            args.allow_oversized_tensors,
        )?);
    }

    let mut artifact_files = copy_metadata_files(&args.source, &args.output)?;
    let mut inventory = written
        .iter()
        .flat_map(|shard| shard.tensors.iter().cloned())
        .collect::<Vec<_>>();
    inventory.extend(omitted_inventory);
    inventory.sort_by(|left, right| {
        (&left.component, &left.source_name, left.source_row_range).cmp(&(
            &right.component,
            &right.source_name,
            right.source_row_range,
        ))
    });
    let inventory_bytes = serde_json::to_vec_pretty(&inventory)?;
    let inventory_file = write_metadata_object(
        &args.output,
        "metadata/tensor-inventory.json",
        &inventory_bytes,
        ArtifactFileRole::Metadata,
    )?;
    artifact_files.push(inventory_file);

    let source_bytes = serde_json::to_vec_pretty(&source_files)?;
    let sources_file = write_metadata_object(
        &args.output,
        "metadata/source-files.json",
        &source_bytes,
        ArtifactFileRole::Metadata,
    )?;
    artifact_files.push(sources_file);

    let mut grouped: BTreeMap<String, Vec<&WrittenShard>> = BTreeMap::new();
    for shard in &written {
        grouped
            .entry(shard.component.clone())
            .or_default()
            .push(shard);
    }
    let components = grouped
        .keys()
        .map(|name| {
            Ok(ArtifactComponent {
                id: ArtifactComponentId::new(name.clone())?,
                required: true,
            })
        })
        .collect::<Result<Vec<_>, burn_image::ValidationError>>()?;
    for (component, shards) in grouped {
        let count = u32::try_from(shards.len())?;
        for (index, shard) in shards.into_iter().enumerate() {
            artifact_files.push(ArtifactFile {
                path: shard.path.clone(),
                size: shard.size,
                sha256: shard.sha256,
                role: ArtifactFileRole::Weights,
                component: Some(ArtifactComponentId::new(component.clone())?),
                shard: (count > 1).then_some(ArtifactShard {
                    index: u32::try_from(index)?,
                    count,
                    chain_sha256: None,
                }),
            });
        }
    }
    artifact_files.sort_by(|left, right| left.path.cmp(&right.path));

    let mut metadata = BTreeMap::new();
    metadata.insert("algorithm".to_owned(), "dmd-turbo".to_owned());
    metadata.insert(
        "artifact_layout".to_owned(),
        "semantic-burnpack-v1".to_owned(),
    );
    metadata.insert(
        "conversion_crate".to_owned(),
        env!("CARGO_PKG_VERSION").to_owned(),
    );
    metadata.insert(
        "layout_contract".to_owned(),
        "metadata/tensor-inventory.json".to_owned(),
    );
    metadata.insert("tensor_inventory_schema".to_owned(), "2".to_owned());
    metadata.insert("profile".to_owned(), args.profile.slug().to_owned());
    let oversized_shards = written
        .iter()
        .filter(|shard| shard.size > max_bytes)
        .count();
    metadata.insert("target_max_shard_bytes".to_owned(), max_bytes.to_string());
    metadata.insert(
        "physical_shards_bounded".to_owned(),
        (oversized_shards == 0).to_string(),
    );
    metadata.insert(
        "oversized_tensor_shards".to_owned(),
        oversized_shards.to_string(),
    );
    metadata.insert(
        "source_revision".to_owned(),
        UPSTREAM_SOURCE_REVISION.to_owned(),
    );
    metadata.insert(
        "upstream_model_repository".to_owned(),
        args.variant.upstream_model_id().to_owned(),
    );
    metadata.insert(
        "tensor_count".to_owned(),
        expected.tensors().len().to_string(),
    );
    metadata.insert(
        "stored_tensor_count".to_owned(),
        inventory
            .iter()
            .filter(|entry| entry.included)
            .count()
            .to_string(),
    );
    metadata.insert(
        "omitted_tensor_count".to_owned(),
        inventory
            .iter()
            .filter(|entry| !entry.included)
            .count()
            .to_string(),
    );
    metadata.insert(
        "qwen_embedding_row_chunks".to_owned(),
        qwen_plan.embedding_rows.chunks.len().to_string(),
    );
    metadata.insert(
        "qwen_lm_head".to_owned(),
        if args.include_lm_head {
            "included-row-sliced"
        } else {
            "omitted-base-model"
        }
        .to_owned(),
    );
    let mut manifest = ArtifactManifest {
        // Direct imports are dependency-free conversion sources. The release builder extracts and
        // seals schema-v1 components, then emits the schema-v2 composition lock.
        schema_version: ARTIFACT_MANIFEST_SCHEMA_V1,
        // Conversion sources keep a descriptive identity. The release builder assigns the clean
        // canonical id after authenticating the exact source bundle.
        bundle: ArtifactBundleId::new(format!("{}-{}", args.variant.slug(), args.profile.slug()))?,
        profile: ArtifactProfileId::new(args.profile.slug())?,
        model: ModelId::new(args.variant.model_id())?,
        model_revision: args.variant.revision().to_owned(),
        numeric_format: args.profile.numeric_format(),
        components,
        files: artifact_files,
        dependencies: Vec::new(),
        metadata,
        content_digest: None,
    };
    let digest = manifest.seal()?;
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
    fs::write(args.output.join("manifest.json"), &manifest_bytes)?;
    eprintln!(
        "sealed {} files / {} source tensors / {} stored slices as {}",
        manifest.files.len(),
        expected.tensors().len(),
        inventory.iter().filter(|entry| entry.included).count(),
        digest
    );
    Ok(())
}

fn validate_source_identity(source: &Path, variant: VariantArg) -> Result<(), Box<dyn Error>> {
    if !source.is_dir() {
        return Err(format!("source directory does not exist: {}", source.display()).into());
    }
    let source_text = source.to_string_lossy();
    let marker = source.join(".burn-image-revision");
    let marker_matches = fs::read_to_string(marker)
        .map(|value| value.trim() == variant.revision())
        .unwrap_or(false);
    if !source_text.contains(variant.revision()) && !marker_matches {
        return Err(format!(
            "source path does not prove pinned revision {}; use the exact Hugging Face snapshot path or add .burn-image-revision",
            variant.revision()
        )
        .into());
    }
    for required in [
        "model_index.json",
        "mllm/config.json",
        "mllm/model.safetensors.index.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ] {
        if !source.join(required).is_file() {
            return Err(format!("pinned snapshot is incomplete: missing {required}").into());
        }
    }
    Ok(())
}

fn load_release_inventory(source: &Path) -> Result<BooguArtifactInventory, Box<dyn Error>> {
    let qwen = Qwen3VlConfig::from_json(&fs::read_to_string(source.join("mllm/config.json"))?)?;
    let vae = AutoencoderKlConfig::from_diffusers_json(&fs::read_to_string(
        source.join("vae/config.json"),
    )?)?;
    let denoiser =
        validate_denoiser_config(&fs::read_to_string(source.join("transformer/config.json"))?)?;
    Ok(BooguArtifactInventory::new(&qwen, &denoiser, &vae)?)
}

fn validate_denoiser_config(json: &str) -> Result<BooguConfig, Box<dyn Error>> {
    let source: UpstreamDenoiserConfig = serde_json::from_str(json)?;
    let config = BooguConfig::default();
    let expected_out = source.out_channels.unwrap_or(source.in_channels);
    let matches = source.patch_size == config.patch_size
        && source.in_channels == config.in_channels
        && expected_out == config.out_channels
        && source.hidden_size == config.hidden_size
        && source.num_layers == config.num_layers
        && source.num_double_stream_layers == config.num_double_stream_layers
        && source.num_refiner_layers == config.num_refiner_layers
        && source.num_attention_heads == config.num_attention_heads
        && source.num_kv_heads == config.num_kv_heads
        && source.multiple_of == config.multiple_of
        && source.norm_eps == config.norm_eps
        && source.axes_dim_rope == config.axes_dim_rope
        && source.axes_lens == config.axes_lens
        && source.instruction_feature_configs.instruction_feat_dim
            == config.instruction_feature_dim
        && source
            .instruction_feature_configs
            .num_instruction_feature_layers
            == 1
        && source.instruction_feature_configs.reduce_type == "mean"
        && source.timestep_scale == config.timestep_scale;
    if !matches {
        return Err(
            "transformer/config.json does not match the released Boogu architecture".into(),
        );
    }
    config.validate()?;
    Ok(config)
}

fn inventory_snapshot(source: &Path) -> Result<Vec<SourceTensor>, Box<dyn Error>> {
    let mut safetensors = Vec::new();
    for component in ["mllm", "transformer", "vae"] {
        for entry in fs::read_dir(source.join(component))? {
            let path = entry?.path();
            if path.extension().and_then(|value| value.to_str()) == Some("safetensors") {
                safetensors.push(path);
            }
        }
    }
    safetensors.sort();
    let mut all = Vec::new();
    for path in safetensors {
        let relative = relative_unix(source, &path)?;
        let component = relative
            .split('/')
            .next()
            .ok_or("source tensor has no component")?
            .to_owned();
        let (header_len, header) = read_safetensors_header(&path)?;
        let file_size = fs::metadata(&path)?.len();
        let data_base = 8_u64
            .checked_add(header_len)
            .ok_or("SafeTensors header offset overflow")?;
        let data_len = file_size
            .checked_sub(data_base)
            .ok_or("SafeTensors header exceeds file length")?;
        let mut ranges = Vec::new();
        for (name, info) in header {
            validate_header_tensor(&path, &name, &info, data_len)?;
            ranges.push((info.data_offsets[0], info.data_offsets[1], name.clone()));
            all.push(SourceTensor {
                stage: String::new(),
                name,
                component: component.clone(),
                owner: match component.as_str() {
                    "mllm" => TensorOwner::Qwen3Vl,
                    "transformer" => TensorOwner::BooguDenoiser,
                    "vae" => TensorOwner::FluxVae,
                    _ => return Err(format!("unknown source component {component}").into()),
                },
                file: path.clone(),
                relative_file: relative.clone(),
                dtype: info.dtype,
                full_source_shape: info.shape.clone(),
                shape: info.shape,
                source_row_range: None,
                absolute_offset: data_base + info.data_offsets[0],
                bytes: info.data_offsets[1] - info.data_offsets[0],
                target_name: String::new(),
                logical_target_name: String::new(),
                target_shape: Vec::new(),
                transform: TensorTransform::Identity,
                quantizable: false,
            });
        }
        ranges.sort_by_key(|range| range.0);
        for pair in ranges.windows(2) {
            if pair[0].1 > pair[1].0 {
                return Err(format!(
                    "overlapping SafeTensors ranges {} and {} in {}",
                    pair[0].2,
                    pair[1].2,
                    path.display()
                )
                .into());
            }
        }
    }
    Ok(all)
}

fn hash_source_files(
    source: &Path,
    tensors: &[SourceTensor],
) -> Result<Vec<SourceFileRecord>, Box<dyn Error>> {
    let files = tensors
        .iter()
        .map(|tensor| tensor.relative_file.clone())
        .collect::<BTreeSet<_>>();
    files
        .into_iter()
        .map(|relative| {
            let path = source.join(&relative);
            Ok(SourceFileRecord {
                path: relative,
                size: fs::metadata(&path)?.len(),
                sha256: hash_file(&path)?,
            })
        })
        .collect()
}

fn read_safetensors_header(
    path: &Path,
) -> Result<(u64, BTreeMap<String, HeaderTensor>), Box<dyn Error>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut length = [0_u8; 8];
    reader.read_exact(&mut length)?;
    let header_len = u64::from_le_bytes(length);
    if header_len == 0 || header_len > 100 * 1024 * 1024 {
        return Err(format!("invalid SafeTensors header length in {}", path.display()).into());
    }
    let header_len_usize = usize::try_from(header_len)?;
    let mut bytes = vec![0_u8; header_len_usize];
    reader.read_exact(&mut bytes)?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)?;
    let object = value
        .as_object()
        .ok_or("SafeTensors header must be a JSON object")?;
    let mut tensors = BTreeMap::new();
    for (name, value) in object {
        if name == "__metadata__" {
            continue;
        }
        tensors.insert(name.clone(), serde_json::from_value(value.clone())?);
    }
    Ok((header_len, tensors))
}

fn validate_header_tensor(
    path: &Path,
    name: &str,
    info: &HeaderTensor,
    data_len: u64,
) -> Result<(), Box<dyn Error>> {
    let element_bytes = match info.dtype.as_str() {
        "BF16" | "F16" => 2_u64,
        "F32" => 4_u64,
        other => return Err(format!("unsupported source dtype {other} for {name}").into()),
    };
    if info.shape.is_empty() || info.shape.contains(&0) {
        return Err(format!("invalid scalar/zero shape for {name} in {}", path.display()).into());
    }
    let elements = info.shape.iter().try_fold(1_u64, |product, &dim| {
        product
            .checked_mul(dim as u64)
            .ok_or("tensor shape overflow")
    })?;
    let expected = elements
        .checked_mul(element_bytes)
        .ok_or("tensor byte size overflow")?;
    let [start, end] = info.data_offsets;
    if start >= end || end > data_len || end - start != expected {
        return Err(format!(
            "invalid data range for {name} in {}: {start}..{end}, expected {expected} bytes within {data_len}",
            path.display()
        )
        .into());
    }
    Ok(())
}

fn validate_inventory(
    tensors: &mut [SourceTensor],
    expected: &BooguArtifactInventory,
) -> Result<(), Box<dyn Error>> {
    let mut names = BTreeSet::new();
    for tensor in tensors.iter_mut() {
        let qualified = format!("{}:{}", tensor.component, tensor.name);
        if !names.insert(qualified.clone()) {
            return Err(format!("duplicate tensor {qualified}").into());
        }
        let spec = expected
            .by_source(&tensor.component, &tensor.name)
            .ok_or_else(|| format!("unknown source tensor {qualified}"))?;
        if tensor.dtype != spec.source_dtype.safetensors_name() {
            return Err(format!(
                "dtype mismatch for {qualified}: expected {}, found {}",
                spec.source_dtype.safetensors_name(),
                tensor.dtype
            )
            .into());
        }
        if tensor.shape != spec.source_shape {
            return Err(format!(
                "shape mismatch for {qualified}: expected {:?}, found {:?}",
                spec.source_shape, tensor.shape
            )
            .into());
        }
        if tensor.owner != spec.owner {
            return Err(format!(
                "owner mismatch for {qualified}: expected {:?}, found {:?}",
                spec.owner, tensor.owner
            )
            .into());
        }
        tensor.stage = spec.stage.clone();
        tensor.target_name = spec.target_name.clone();
        tensor.logical_target_name = spec.target_name.clone();
        tensor.target_shape = spec.target_shape.clone();
        tensor.transform = spec.transform;
        tensor.quantizable = spec.quantizable;
    }
    let expected_names = expected
        .tensors()
        .iter()
        .map(ArtifactTensorSpec::qualified_source_name)
        .collect::<BTreeSet<_>>();
    let missing = expected_names
        .difference(&names)
        .take(16)
        .collect::<Vec<_>>();
    if !missing.is_empty() || names.len() != expected_names.len() {
        return Err(format!(
            "incomplete checkpoint inventory: expected {}, found {}; first missing keys: {missing:?}",
            expected_names.len(),
            names.len()
        )
        .into());
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
struct WeightIndex {
    weight_map: BTreeMap<String, String>,
}

fn validate_weight_indexes(source: &Path, tensors: &[SourceTensor]) -> Result<(), Box<dyn Error>> {
    for (component, index_name) in [
        ("mllm", "model.safetensors.index.json"),
        (
            "transformer",
            "diffusion_pytorch_model.safetensors.index.json",
        ),
    ] {
        let index: WeightIndex =
            serde_json::from_slice(&fs::read(source.join(component).join(index_name))?)?;
        let observed = tensors
            .iter()
            .filter(|tensor| tensor.component == component)
            .map(|tensor| (tensor.name.as_str(), tensor.relative_file.as_str()))
            .collect::<BTreeMap<_, _>>();
        if observed.len() != index.weight_map.len() {
            return Err(format!(
                "{component} index contains {} keys but SafeTensors headers contain {}",
                index.weight_map.len(),
                observed.len()
            )
            .into());
        }
        for (name, filename) in &index.weight_map {
            let expected_file = format!("{component}/{filename}");
            match observed.get(name.as_str()) {
                Some(actual) if **actual == expected_file => {}
                Some(actual) => {
                    return Err(format!(
                        "{component} index maps {name} to {expected_file}, header was in {actual}"
                    )
                    .into());
                }
                None => return Err(format!("{component} index key {name} is missing").into()),
            }
        }
    }
    let vae_files = tensors
        .iter()
        .filter(|tensor| tensor.component == "vae")
        .map(|tensor| tensor.relative_file.as_str())
        .collect::<BTreeSet<_>>();
    if vae_files != BTreeSet::from(["vae/diffusion_pytorch_model.safetensors"]) {
        return Err(format!("unexpected VAE SafeTensors files: {vae_files:?}").into());
    }
    Ok(())
}

fn prepare_streaming_tensors(
    tensors: Vec<SourceTensor>,
    config: &Qwen3VlConfig,
    max_bytes: u64,
    include_lm_head: bool,
) -> Result<PreparedStreamingTensors, Box<dyn Error>> {
    let plan = bounded_qwen_plan(config, include_lm_head, max_bytes)?;
    let mut payload = Vec::with_capacity(tensors.len() + plan.embedding_rows.chunks.len());
    let mut omitted = Vec::new();
    for tensor in tensors {
        let row_plan = if tensor.component == "mllm"
            && tensor.name == "model.language_model.embed_tokens.weight"
        {
            Some((&plan.embedding_rows, false))
        } else if tensor.component == "mllm" && tensor.name == "lm_head.weight" {
            plan.lm_head_rows.as_ref().map(|rows| (rows, true))
        } else {
            None
        };
        if let Some((row_plan, lm_head)) = row_plan {
            validate_row_slice_source(&tensor, config)?;
            for chunk in &row_plan.chunks {
                let mut slice = tensor.clone();
                let row_bytes = u64::try_from(chunk.hidden_size)? * 2;
                slice.absolute_offset = tensor
                    .absolute_offset
                    .checked_add(
                        u64::try_from(chunk.row_range.start)?
                            .checked_mul(row_bytes)
                            .ok_or("Qwen row-slice source offset overflow")?,
                    )
                    .ok_or("Qwen row-slice absolute offset overflow")?;
                slice.bytes = u64::try_from(chunk.byte_len())?;
                slice.shape = vec![chunk.rows(), chunk.hidden_size];
                slice.target_shape = slice.shape.clone();
                slice.source_row_range = Some([chunk.row_range.start, chunk.row_range.end]);
                slice.stage = qwen_streaming_stage_name(&if lm_head {
                    Qwen3VlStage::LmHeadRows {
                        chunk: chunk.chunk_index,
                    }
                } else {
                    Qwen3VlStage::EmbeddingRows {
                        chunk: chunk.chunk_index,
                    }
                });
                slice.target_name = qwen_row_slice_target(&tensor.logical_target_name, chunk);
                slice.quantizable = false;
                payload.push(slice);
            }
        } else if tensor.component == "mllm" && tensor.name == "lm_head.weight" && !include_lm_head
        {
            omitted.push(omitted_inventory_tensor(&tensor));
        } else {
            payload.push(tensor);
        }
    }
    if omitted.len() != usize::from(!include_lm_head) {
        return Err(format!(
            "expected exactly one validated Qwen LM-head tensor to be omitted, found {}",
            omitted.len()
        )
        .into());
    }
    validate_qwen_stage_assignments(&payload, &plan)?;
    Ok((payload, omitted, plan))
}

fn bounded_qwen_plan(
    config: &Qwen3VlConfig,
    include_lm_head: bool,
    max_bytes: u64,
) -> Result<Qwen3VlStreamingPlan, Box<dyn Error>> {
    let released = Qwen3VlStreamingPlan::released_f16(config, include_lm_head)?;
    let fits = released
        .embedding_rows
        .chunks
        .iter()
        .chain(released.lm_head_rows.iter().flat_map(|plan| &plan.chunks))
        .all(|chunk| {
            u64::try_from(chunk.byte_len()).is_ok_and(|bytes| {
                bytes.saturating_add(BURNPACK_ROW_SLICE_RESERVE_BYTES) <= max_bytes
            })
        });
    if fits {
        return Ok(released);
    }
    let payload_limit = max_bytes
        .checked_sub(BURNPACK_ROW_SLICE_RESERVE_BYTES)
        .ok_or("--max-shard-mib is too small for Burnpack row-slice metadata")?;
    let payload_limit = usize::try_from(payload_limit)?;
    let embedding_rows = RowChunkPlan::for_max_bytes(
        config.text_config.vocab_size,
        config.text_config.hidden_size,
        2,
        payload_limit,
    )?;
    let lm_head_rows = include_lm_head
        .then(|| {
            RowChunkPlan::for_max_bytes(
                config.text_config.vocab_size,
                config.text_config.hidden_size,
                2,
                payload_limit,
            )
        })
        .transpose()?;
    Ok(Qwen3VlStreamingPlan::new(
        config,
        embedding_rows,
        lm_head_rows,
    )?)
}

fn validate_row_slice_source(
    tensor: &SourceTensor,
    config: &Qwen3VlConfig,
) -> Result<(), Box<dyn Error>> {
    let expected = vec![
        config.text_config.vocab_size,
        config.text_config.hidden_size,
    ];
    if tensor.dtype != "BF16"
        || tensor.full_source_shape != expected
        || tensor.shape != expected
        || tensor.transform != TensorTransform::Identity
    {
        return Err(format!(
            "Qwen row-slice source {} has incompatible dtype/shape/layout: {} {:?} {:?}",
            tensor.name, tensor.dtype, tensor.shape, tensor.transform
        )
        .into());
    }
    Ok(())
}

fn validate_qwen_stage_assignments(
    payload: &[SourceTensor],
    plan: &Qwen3VlStreamingPlan,
) -> Result<(), Box<dyn Error>> {
    let expected = plan
        .stages
        .iter()
        .flat_map(|descriptor| {
            let stage = qwen_streaming_stage_name(&descriptor.stage);
            descriptor
                .tensors
                .iter()
                .map(move |tensor| (tensor.source.clone(), stage.clone()))
        })
        .collect::<BTreeMap<_, _>>();
    for tensor in payload
        .iter()
        .filter(|tensor| tensor.component == "mllm" && tensor.source_row_range.is_none())
    {
        let stage = expected.get(&tensor.name).ok_or_else(|| {
            format!(
                "Qwen streaming plan does not own non-sliced tensor {}",
                tensor.name
            )
        })?;
        if stage != &tensor.stage {
            return Err(format!(
                "Qwen stage mismatch for {}: inventory {}, streaming plan {}",
                tensor.name, tensor.stage, stage
            )
            .into());
        }
    }
    Ok(())
}

fn omitted_inventory_tensor(tensor: &SourceTensor) -> InventoryTensor {
    InventoryTensor {
        source_name: tensor.name.clone(),
        logical_target_name: tensor.logical_target_name.clone(),
        target_name: tensor.target_name.clone(),
        owner: tensor.owner,
        component: tensor.component.clone(),
        stage: tensor.stage.clone(),
        transform: tensor.transform,
        source_file: tensor.relative_file.clone(),
        source_dtype: tensor.dtype.clone(),
        source_shape: tensor.full_source_shape.clone(),
        source_row_range: None,
        included: false,
        stored_dtype: None,
        stored_shape: None,
        source_offset: tensor.absolute_offset,
        source_bytes: tensor.bytes,
        quantized: false,
        stored_sha256: None,
        burnpack_object: None,
    }
}

fn plan_shards(
    tensors: Vec<SourceTensor>,
    profile: ProfileArg,
    max_bytes: u64,
) -> Vec<PlannedShard> {
    let mut planned = Vec::new();
    let mut current: Option<PlannedShard> = None;
    let mut current_bytes = 0_u64;
    for tensor in tensors {
        let estimate = stored_size_estimate(&tensor, profile);
        let must_flush = current.as_ref().is_some_and(|shard| {
            shard.component != tensor.stage
                || (!shard.tensors.is_empty() && current_bytes.saturating_add(estimate) > max_bytes)
        });
        if must_flush {
            planned.push(current.take().expect("checked current shard"));
            current_bytes = 0;
        }
        let shard = current.get_or_insert_with(|| PlannedShard {
            component: tensor.stage.clone(),
            tensors: Vec::new(),
        });
        shard.tensors.push(tensor);
        current_bytes = current_bytes.saturating_add(estimate);
    }
    if let Some(shard) = current {
        planned.push(shard);
    }
    planned
}

fn validate_shard_plan(
    shards: &[PlannedShard],
    profile: ProfileArg,
    max_bytes: u64,
    allow_oversized_tensors: bool,
) -> Result<(), Box<dyn Error>> {
    for shard in shards {
        let estimate = shard
            .tensors
            .iter()
            .map(|tensor| stored_size_estimate(tensor, profile))
            .fold(0_u64, u64::saturating_add);
        if estimate > max_bytes && (!allow_oversized_tensors || shard.tensors.len() != 1) {
            let largest = shard
                .tensors
                .iter()
                .max_by_key(|tensor| stored_size_estimate(tensor, profile))
                .expect("planned shards are non-empty");
            return Err(format!(
                "stage {} cannot satisfy --max-shard-mib: tensor {}:{} alone estimates {} bytes, requested maximum is {} bytes; increase the limit for a native-only artifact or add a chunk-aware module contract",
                shard.component,
                largest.component,
                largest.name,
                stored_size_estimate(largest, profile),
                max_bytes,
            )
            .into());
        }
    }
    Ok(())
}

fn stored_size_estimate(tensor: &SourceTensor, profile: ProfileArg) -> u64 {
    let elements = tensor.target_shape.iter().product::<usize>() as u64;
    if profile == ProfileArg::Q4sBlockUpTo128F32
        && let Some((block, _)) = q4s_storage_block_and_axis(tensor.owner, &tensor.target_shape)
    {
        elements / 2 + elements / block as u64 * 4 + 1024
    } else if should_quantize(tensor, profile) {
        elements + elements.div_ceil(32) * 4 + 1024
    } else if should_store_f32(tensor, profile) {
        elements * 4 + 1024
    } else {
        elements * 2 + 1024
    }
}

fn should_quantize(tensor: &SourceTensor, profile: ProfileArg) -> bool {
    if profile == ProfileArg::Q4sBlockUpTo128F32 {
        return q4s_storage_block_and_axis(tensor.owner, &tensor.target_shape).is_some();
    }
    let q8 = matches!(
        profile,
        ProfileArg::Q8sBlock32F32 | ProfileArg::Q8sBlock32F32QwenVisionF32
    );
    q8 && tensor.quantizable
        && !(profile == ProfileArg::Q8sBlock32F32QwenVisionF32 && is_qwen_vision(tensor))
}

fn should_store_f32(tensor: &SourceTensor, profile: ProfileArg) -> bool {
    matches!(
        profile,
        ProfileArg::F16QwenVisionF32 | ProfileArg::Q8sBlock32F32QwenVisionF32
    ) && is_qwen_vision(tensor)
}

fn is_qwen_vision(tensor: &SourceTensor) -> bool {
    tensor.owner == TensorOwner::Qwen3Vl && tensor.stage.starts_with("qwen-vision-")
}

fn write_shard(
    output: &Path,
    shard: PlannedShard,
    profile: ProfileArg,
    max_bytes: u64,
    allow_oversized_tensors: bool,
) -> Result<WrittenShard, Box<dyn Error>> {
    let oversized_tensor_exception = allow_oversized_tensors
        && shard.tensors.len() == 1
        && stored_size_estimate(&shard.tensors[0], profile) > max_bytes;
    let mut snapshots = Vec::with_capacity(shard.tensors.len());
    let mut pending_inventory = Vec::with_capacity(shard.tensors.len());
    for tensor in &shard.tensors {
        let source_data = read_tensor_bytes(tensor)?;
        let quantized = should_quantize(tensor, profile);
        let (data, stored_dtype, stored_shape) =
            if quantized && profile == ProfileArg::Q4sBlockUpTo128F32 {
                let data = match tensor.owner {
                    TensorOwner::Qwen3Vl => {
                        let (values, shape) = decode_transposed_f32(tensor, &source_data)?;
                        quantize_q4s_block128_f32(values, shape)?
                    }
                    TensorOwner::BooguDenoiser => quantize_row_layout_q4s_block_up_to128_f32(
                        convert_f32(tensor, &source_data)?,
                    )?,
                    TensorOwner::FluxVae => {
                        return Err("FLUX VAE tensors are not eligible for Q4 storage".into());
                    }
                };
                let stored_dtype = q4s_stored_dtype(tensor.owner, &tensor.target_shape)
                    .ok_or("Q4 tensor has no exact storage contract")?;
                (data, stored_dtype, tensor.target_shape.clone())
            } else if quantized {
                let (values, shape) = decode_transposed_f32(tensor, &source_data)?;
                let data = quantize_q8s_block32_f32(values, shape.clone())?;
                (data, "q8s-block32-f32".to_owned(), shape)
            } else if should_store_f32(tensor, profile) {
                (
                    convert_f32(tensor, &source_data)?,
                    "f32".to_owned(),
                    tensor.target_shape.clone(),
                )
            } else {
                (
                    convert_f16(tensor, &source_data)?,
                    "f16".to_owned(),
                    tensor.target_shape.clone(),
                )
            };
        let stored_sha256 = Sha256Digest::calculate(data.bytes.as_ref());
        let target_name = tensor.target_name.clone();
        let id = deterministic_param_id(&format!("{}:{}", tensor.component, target_name));
        snapshots.push(TensorSnapshot::from_data(
            data,
            vec![target_name.clone()],
            vec!["Imported".to_owned()],
            id,
        ));
        pending_inventory.push((tensor, quantized, stored_dtype, stored_shape, stored_sha256));
    }
    let temporary = output.join("objects").join(format!(
        ".{}-{}-{}.tmp",
        shard.component,
        std::process::id(),
        deterministic_param_id(&shard.component).val()
    ));
    BurnpackWriter::new(snapshots)
        .with_metadata("component", &shard.component)
        .with_metadata("profile", profile.slug())
        .with_metadata("source_layout", "pytorch")
        .with_metadata("layout_contract", "metadata/tensor-inventory.json")
        .write_to_file(&temporary)?;
    validate_written_burnpack(&temporary, &pending_inventory)?;
    let size = fs::metadata(&temporary)?.len();
    if size > max_bytes && !oversized_tensor_exception {
        fs::remove_file(&temporary)?;
        return Err(format!(
            "serialized Burnpack stage {} is {size} bytes, exceeding the declared {max_bytes}-byte semantic-object limit",
            shard.component
        )
        .into());
    }
    let sha256 = hash_file(&temporary)?;
    let relative = format!("objects/{sha256}.bpk");
    let final_path = output.join(&relative);
    if final_path.exists() {
        let existing_size = fs::metadata(&final_path)?.len();
        let existing_hash = hash_file(&final_path)?;
        if existing_size != size || existing_hash != sha256 {
            return Err(format!("conflicting content-addressed object {relative}").into());
        }
        fs::remove_file(&temporary)?;
    } else {
        fs::rename(&temporary, &final_path)?;
    }
    let inventory = pending_inventory
        .into_iter()
        .map(
            |(tensor, quantized, stored_dtype, stored_shape, stored_sha256)| InventoryTensor {
                source_name: tensor.name.clone(),
                logical_target_name: tensor.logical_target_name.clone(),
                target_name: tensor.target_name.clone(),
                owner: tensor.owner,
                component: tensor.component.clone(),
                stage: tensor.stage.clone(),
                transform: tensor.transform,
                source_file: tensor.relative_file.clone(),
                source_dtype: tensor.dtype.clone(),
                source_shape: tensor.full_source_shape.clone(),
                source_row_range: tensor.source_row_range,
                included: true,
                stored_dtype: Some(stored_dtype),
                stored_shape: Some(stored_shape),
                source_offset: tensor.absolute_offset,
                source_bytes: tensor.bytes,
                quantized,
                stored_sha256: Some(stored_sha256),
                burnpack_object: Some(relative.clone()),
            },
        )
        .collect();
    Ok(WrittenShard {
        component: shard.component,
        path: ArtifactPath::new(relative)?,
        size,
        sha256,
        tensors: inventory,
    })
}

fn validate_written_burnpack(
    path: &Path,
    expected: &[(&SourceTensor, bool, String, Vec<usize>, Sha256Digest)],
) -> Result<(), Box<dyn Error>> {
    let mut store = BurnpackStore::from_file(path).auto_extension(false);
    let snapshots = store.get_all_snapshots()?;
    if snapshots.len() != expected.len() {
        return Err(format!(
            "Burnpack round-trip tensor count mismatch for {}: expected {}, found {}",
            path.display(),
            expected.len(),
            snapshots.len()
        )
        .into());
    }
    for (tensor, quantized, stored_dtype, stored_shape, expected_digest) in expected {
        let snapshot = snapshots.get(&tensor.target_name).ok_or_else(|| {
            format!(
                "Burnpack round-trip omitted tensor {} from {}",
                tensor.target_name,
                path.display()
            )
        })?;
        if snapshot.shape.as_slice() != stored_shape {
            return Err(format!(
                "Burnpack shape changed for {}: expected {:?}, found {:?}",
                tensor.target_name, stored_shape, snapshot.shape
            )
            .into());
        }
        let dtype_matches = match stored_dtype.as_str() {
            "q8s-block32-f32" if *quantized => matches!(snapshot.dtype, DType::QFloat(scheme)
                if scheme.value == QuantValue::Q8S
                    && scheme.level == QuantLevel::block([32])
                    && scheme.param == QuantParam::F32),
            stored if stored.starts_with("q4s-block") && *quantized => {
                let expected = match snapshot.dtype {
                    DType::QFloat(scheme)
                        if scheme.value == QuantValue::Q4S && scheme.param == QuantParam::F32 =>
                    {
                        match (scheme.level, scheme.store) {
                            (
                                QuantLevel::Block(block),
                                burn::tensor::quantization::QuantStore::PackedU32(0),
                            ) if block.len() == 1 => {
                                format!("q4s-block{}-f32-packed-axis0", block[0])
                            }
                            (
                                QuantLevel::Block(block),
                                burn::tensor::quantization::QuantStore::PackedU32(1),
                            ) if block.len() == 2 && block[1] == 1 => {
                                format!("q4s-block{}x1-f32-packed-axis1", block[0])
                            }
                            _ => String::new(),
                        }
                    }
                    _ => String::new(),
                };
                expected == stored
            }
            "f16" if !quantized => snapshot.dtype == DType::F16,
            "f32" if !quantized => snapshot.dtype == DType::F32,
            _ => false,
        };
        if !dtype_matches {
            return Err(format!(
                "Burnpack dtype changed for {}: inventory {}, found {:?}",
                tensor.target_name, stored_dtype, snapshot.dtype
            )
            .into());
        }
        let data = snapshot.to_data()?;
        let actual_digest = Sha256Digest::calculate(data.bytes.as_ref());
        if &actual_digest != expected_digest {
            return Err(format!(
                "Burnpack payload changed for {}: expected {}, found {}",
                tensor.target_name, expected_digest, actual_digest
            )
            .into());
        }
    }
    Ok(())
}

fn read_tensor_bytes(tensor: &SourceTensor) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut file = File::open(&tensor.file)?;
    file.seek(SeekFrom::Start(tensor.absolute_offset))?;
    let mut bytes = vec![0_u8; usize::try_from(tensor.bytes)?];
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn convert_f16(tensor: &SourceTensor, source: &[u8]) -> Result<TensorData, Box<dyn Error>> {
    let mut output = Vec::with_capacity(tensor.shape.iter().product::<usize>() * 2);
    match tensor.dtype.as_str() {
        "F16" => {
            for pair in source.chunks_exact(2) {
                let value = f16::from_bits(u16::from_le_bytes([pair[0], pair[1]]));
                if !value.is_finite() {
                    return Err(format!(
                        "source tensor {} contains a non-finite F16 value",
                        tensor.name
                    )
                    .into());
                }
                output.extend_from_slice(pair);
            }
        }
        "BF16" => {
            for pair in source.chunks_exact(2) {
                let value = bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]])).to_f32();
                output.extend_from_slice(&checked_f16_bits(tensor, value)?.to_le_bytes());
            }
        }
        "F32" => {
            for word in source.chunks_exact(4) {
                let value = f32::from_le_bytes([word[0], word[1], word[2], word[3]]);
                output.extend_from_slice(&checked_f16_bits(tensor, value)?.to_le_bytes());
            }
        }
        other => return Err(format!("unsupported conversion dtype {other}").into()),
    }
    if tensor.transform == TensorTransform::Transpose2d {
        output = transpose_u16_bytes(&output, tensor.shape[0], tensor.shape[1]);
    }
    Ok(TensorData::from_bytes_vec(
        output,
        tensor.target_shape.clone(),
        DType::F16,
    ))
}

fn convert_f32(tensor: &SourceTensor, source: &[u8]) -> Result<TensorData, Box<dyn Error>> {
    let (values, shape) = decode_transposed_f32(tensor, source)?;
    Ok(TensorData::new(values, shape))
}

fn checked_f16_bits(tensor: &SourceTensor, value: f32) -> Result<u16, Box<dyn Error>> {
    if !value.is_finite() {
        return Err(format!("source tensor {} contains a non-finite value", tensor.name).into());
    }
    let converted = f16::from_f32(value);
    if !converted.is_finite() {
        return Err(format!(
            "source tensor {} contains {value}, which overflows F16",
            tensor.name
        )
        .into());
    }
    Ok(converted.to_bits())
}

fn decode_transposed_f32(
    tensor: &SourceTensor,
    source: &[u8],
) -> Result<(Vec<f32>, Vec<usize>), Box<dyn Error>> {
    let mut values = match tensor.dtype.as_str() {
        "F16" => source
            .chunks_exact(2)
            .map(|pair| f16::from_bits(u16::from_le_bytes([pair[0], pair[1]])).to_f32())
            .collect::<Vec<_>>(),
        "BF16" => source
            .chunks_exact(2)
            .map(|pair| bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]])).to_f32())
            .collect::<Vec<_>>(),
        "F32" => source
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect::<Vec<_>>(),
        other => return Err(format!("unsupported quantization dtype {other}").into()),
    };
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("source tensor {} contains a non-finite value", tensor.name).into());
    }
    if tensor.transform == TensorTransform::Transpose2d {
        values = transpose_values(&values, tensor.shape[0], tensor.shape[1]);
    }
    Ok((values, tensor.target_shape.clone()))
}

fn transpose_u16_bytes(source: &[u8], rows: usize, columns: usize) -> Vec<u8> {
    let mut output = vec![0_u8; source.len()];
    for row in 0..rows {
        for column in 0..columns {
            let source_index = (row * columns + column) * 2;
            let target_index = (column * rows + row) * 2;
            output[target_index..target_index + 2]
                .copy_from_slice(&source[source_index..source_index + 2]);
        }
    }
    output
}

fn transpose_values(source: &[f32], rows: usize, columns: usize) -> Vec<f32> {
    let mut output = vec![0.0; source.len()];
    for row in 0..rows {
        for column in 0..columns {
            output[column * rows + row] = source[row * columns + column];
        }
    }
    output
}

fn deterministic_param_id(name: &str) -> ParamId {
    let digest = Sha256::digest(name.as_bytes());
    let mut bytes = [0_u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    ParamId::from(u64::from_le_bytes(bytes))
}

fn copy_metadata_files(source: &Path, output: &Path) -> Result<Vec<ArtifactFile>, Box<dyn Error>> {
    let mut candidates = vec![source.join("model_index.json")];
    for directory in ["mllm", "processor", "scheduler", "transformer", "vae"] {
        let root = source.join(directory);
        if !root.is_dir() {
            continue;
        }
        for entry in fs::read_dir(root)? {
            let path = entry?.path();
            let extension = path.extension().and_then(|value| value.to_str());
            if path.is_file()
                && matches!(extension, Some("json" | "jinja" | "txt"))
                && fs::metadata(&path)?.len() <= MAX_METADATA_FILE_BYTES
            {
                candidates.push(path);
            }
        }
    }
    candidates.sort();
    candidates.dedup();
    let mut files = Vec::new();
    for source_path in candidates {
        let relative_source = relative_unix(source, &source_path)?;
        let artifact_relative = format!("metadata/source/{relative_source}");
        let bytes = fs::read(&source_path)?;
        let role = if relative_source.contains("tokenizer")
            || relative_source.ends_with("vocab.json")
            || relative_source.ends_with("merges.txt")
            || relative_source.contains("chat_template")
        {
            ArtifactFileRole::Tokenizer
        } else {
            ArtifactFileRole::Config
        };
        files.push(write_metadata_object(
            output,
            &artifact_relative,
            &bytes,
            role,
        )?);
    }
    Ok(files)
}

fn write_metadata_object(
    output: &Path,
    relative: &str,
    bytes: &[u8],
    role: ArtifactFileRole,
) -> Result<ArtifactFile, Box<dyn Error>> {
    if bytes.is_empty() {
        return Err(format!("refusing zero-length metadata object {relative}").into());
    }
    let path = output.join(relative);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    if path.exists() {
        return Err(format!("metadata output collision: {relative}").into());
    }
    fs::write(&path, bytes)?;
    Ok(ArtifactFile {
        path: ArtifactPath::new(relative)?,
        size: bytes.len() as u64,
        sha256: Sha256Digest::calculate(bytes),
        role,
        component: None,
        shard: None,
    })
}

fn hash_file(path: &Path) -> Result<Sha256Digest, Box<dyn Error>> {
    let mut reader = BufReader::with_capacity(1024 * 1024, File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(Sha256Digest::from_bytes(hasher.finalize().into()))
}

fn relative_unix(root: &Path, path: &Path) -> Result<String, Box<dyn Error>> {
    Ok(path
        .strip_prefix(root)?
        .components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn released_qwen_config() -> Qwen3VlConfig {
        Qwen3VlConfig::from_json(
            r#"{
              "text_config": {
                "vocab_size":151936,"hidden_size":4096,"intermediate_size":12288,
                "num_hidden_layers":36,"num_attention_heads":32,"num_key_value_heads":8,
                "head_dim":128,"hidden_act":"silu","rms_norm_eps":1e-6,
                "rope_scaling":{"mrope_section":[24,20,20],"mrope_interleaved":true}
              },
              "vision_config": {
                "depth":27,"hidden_size":1152,"intermediate_size":4304,"num_heads":16,
                "patch_size":16,"temporal_patch_size":2,"spatial_merge_size":2,
                "out_hidden_size":4096,"num_position_embeddings":2304,
                "deepstack_visual_indexes":[8,16,24]
              },
              "tie_word_embeddings":false,"image_token_id":151655,"video_token_id":151656,
              "vision_start_token_id":151652,"vision_end_token_id":151653
            }"#,
        )
        .unwrap()
    }

    fn vocabulary_source(name: &str, offset: u64) -> SourceTensor {
        let shape = vec![151_936, 4096];
        SourceTensor {
            name: name.into(),
            component: "mllm".into(),
            owner: TensorOwner::Qwen3Vl,
            stage: if name == "lm_head.weight" {
                "qwen-lm-head".into()
            } else {
                "qwen-embedding".into()
            },
            file: PathBuf::from("unused.safetensors"),
            relative_file: "mllm/unused.safetensors".into(),
            dtype: "BF16".into(),
            full_source_shape: shape.clone(),
            shape: shape.clone(),
            source_row_range: None,
            absolute_offset: offset,
            bytes: shape.iter().product::<usize>() as u64 * 2,
            target_name: name.into(),
            logical_target_name: name.into(),
            target_shape: shape,
            transform: TensorTransform::Identity,
            quantizable: false,
        }
    }

    #[test]
    fn transpose_matches_burn_linear_layout_correctness() {
        let source = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(
            transpose_values(&source, 2, 3),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }

    #[test]
    fn q8s_quantization_is_finite_and_bounded_correctness() {
        let values = (0..32).map(|value| value as f32 - 16.0).collect();
        let data = quantize_q8s_block32_f32(values, vec![1, 32]).unwrap();
        assert!(matches!(data.dtype, DType::QFloat(scheme)
            if scheme.param == QuantParam::F32
                && scheme.level == QuantLevel::block([32])));
        assert_eq!(data.shape.as_slice(), &[1, 32]);
    }

    #[test]
    fn direct_q4_profile_matches_runtime_layout_and_size_contract_correctness() {
        let mut qwen =
            vocabulary_source("model.language_model.layers.0.self_attn.q_proj.weight", 0);
        qwen.target_shape = vec![256, 128];
        assert!(should_quantize(&qwen, ProfileArg::Q4sBlockUpTo128F32));
        assert_eq!(
            q4s_stored_dtype(qwen.owner, &qwen.target_shape).as_deref(),
            Some("q4s-block128-f32-packed-axis0")
        );
        assert_eq!(
            stored_size_estimate(&qwen, ProfileArg::Q4sBlockUpTo128F32),
            256 * 128 / 2 + 256 * 4 + 1024
        );

        let mut denoiser = qwen.clone();
        denoiser.owner = TensorOwner::BooguDenoiser;
        denoiser.target_shape = vec![64, 3360];
        assert_eq!(
            q4s_stored_dtype(denoiser.owner, &denoiser.target_shape).as_deref(),
            Some("q4s-block64x1-f32-packed-axis1")
        );

        let mut vae = qwen;
        vae.owner = TensorOwner::FluxVae;
        assert!(!should_quantize(&vae, ProfileArg::Q4sBlockUpTo128F32));
        assert_eq!(q4s_stored_dtype(vae.owner, &vae.target_shape), None);
    }

    #[test]
    fn oversized_single_tensor_plan_is_rejected_correctness() {
        let tensor = SourceTensor {
            name: "model.embed_tokens.weight".into(),
            component: "mllm".into(),
            owner: TensorOwner::Qwen3Vl,
            stage: "qwen-text-core".into(),
            file: PathBuf::from("unused.safetensors"),
            relative_file: "mllm/unused.safetensors".into(),
            dtype: "BF16".into(),
            full_source_shape: vec![1, 1024],
            shape: vec![1, 1024],
            source_row_range: None,
            absolute_offset: 0,
            bytes: 2048,
            target_name: "model.embed_tokens.weight".into(),
            logical_target_name: "model.embed_tokens.weight".into(),
            target_shape: vec![1, 1024],
            transform: TensorTransform::Identity,
            quantizable: false,
        };
        let shards = plan_shards(vec![tensor], ProfileArg::F16, 1024);
        let error = validate_shard_plan(&shards, ProfileArg::F16, 1024, false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("cannot satisfy --max-shard-mib"));
        assert!(error.contains("chunk-aware module contract"));
        validate_shard_plan(&shards, ProfileArg::F16, 1024, true).unwrap();
    }

    #[test]
    fn production_qwen_vocabulary_is_bounded_and_lm_head_is_proven_omitted_correctness() {
        let max_bytes = 256 * 1024 * 1024;
        let embedding = vocabulary_source("model.language_model.embed_tokens.weight", 4096);
        let lm_head = vocabulary_source("lm_head.weight", 2_000_000_000);
        let (payload, omitted, plan) = prepare_streaming_tensors(
            vec![embedding, lm_head],
            &released_qwen_config(),
            max_bytes,
            false,
        )
        .unwrap();
        assert_eq!(payload.len(), 6);
        assert_eq!(plan.embedding_rows.chunks.len(), 6);
        assert!(plan.lm_head_rows.is_none());
        assert_eq!(omitted.len(), 1);
        assert_eq!(omitted[0].source_name, "lm_head.weight");
        assert!(!omitted[0].included);
        assert!(omitted[0].burnpack_object.is_none());
        assert!(payload.iter().all(|tensor| {
            tensor.source_row_range.is_some()
                && stored_size_estimate(tensor, ProfileArg::F16) <= max_bytes
        }));
        let ranges = payload
            .iter()
            .map(|tensor| tensor.source_row_range.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(ranges.first().unwrap()[0], 0);
        assert_eq!(ranges.last().unwrap()[1], 151_936);
        for pair in ranges.windows(2) {
            assert_eq!(pair[0][1], pair[1][0]);
        }
        let shards = plan_shards(payload, ProfileArg::F16, max_bytes);
        validate_shard_plan(&shards, ProfileArg::F16, max_bytes, false).unwrap();
    }

    #[test]
    fn hybrid_profiles_keep_qwen_vision_f32_and_quantize_only_non_vision_correctness() {
        let mut vision = vocabulary_source("model.visual.merger.linear_fc1.weight", 0);
        vision.stage = "qwen-vision-final-merger".into();
        vision.quantizable = true;
        let mut text = vocabulary_source("model.language_model.layers.0.mlp.gate_proj.weight", 0);
        text.stage = "qwen-text-block-00".into();
        text.quantizable = true;

        assert!(should_store_f32(&vision, ProfileArg::F16QwenVisionF32));
        assert!(should_store_f32(
            &vision,
            ProfileArg::Q8sBlock32F32QwenVisionF32
        ));
        assert!(!should_quantize(
            &vision,
            ProfileArg::Q8sBlock32F32QwenVisionF32
        ));
        assert!(should_quantize(
            &text,
            ProfileArg::Q8sBlock32F32QwenVisionF32
        ));
        assert_eq!(
            stored_size_estimate(&vision, ProfileArg::F16QwenVisionF32),
            vision.target_shape.iter().product::<usize>() as u64 * 4 + 1024
        );
    }
}
