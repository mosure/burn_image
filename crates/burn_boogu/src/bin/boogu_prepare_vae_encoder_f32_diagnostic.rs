//! Prepare an opt-in 1.5K VAE encoder F32 A/B artifact without changing production defaults.

use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs::{self, File, OpenOptions},
    io::{BufReader, Read, Seek, SeekFrom, Write},
    path::{Component, Path, PathBuf},
};

use burn::{module::ParamId, tensor::TensorData};
use burn_boogu::artifacts::{
    BooguReleaseIdentity, EDIT_TURBO_1K5_VAE_ENCODER_F32_DIAGNOSTIC_TENSORS,
    EDIT_TURBO_1K5_VAE_SOURCE_BYTES, EDIT_TURBO_1K5_VAE_SOURCE_SHA256,
    LEGACY_EDIT_TURBO_1K5_F16_QWEN_VISION_F32_CONTENT_DIGEST, TensorOwner, TensorTransform,
    stamp_edit_turbo_1k5_vae_encoder_f32_diagnostic_metadata,
    validate_edit_turbo_1k5_vae_encoder_f32_diagnostic_manifest,
};
use burn_image::{ArtifactFile, ArtifactFileRole, ArtifactManifest, ArtifactPath, Sha256Digest};
use burn_store::{BurnpackStore, BurnpackWriter, ModuleStore, TensorSnapshot};
use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const BASE_BUNDLE_ID: &str = "boogu-image-0.1-edit-turbo-1k5";
const DIAGNOSTIC_BUNDLE_ID: &str = "boogu-image-0.1-edit-turbo-1k5-f16-qwen-vision-f32";
const ENCODER_STAGE: &str = "flux-vae-encoder";
const DECODER_STAGE: &str = "flux-vae-decoder";
const INVENTORY_PATH: &str = "metadata/tensor-inventory.json";
const OUTPUT_REPORT_PATH: &str = "diagnostic-overlay-report.json";
const DEFAULT_MAX_SHARD_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug, Parser)]
#[command(about = "Build a sealed, explicit 1.5K VAE encoder F32 diagnostic overlay")]
struct Args {
    /// Exact canonical 1.5K bundle whose non-encoder payloads are reused.
    #[arg(long, default_value = ".artifacts/boogu-image-0.1-edit-turbo-1k5")]
    base: PathBuf,
    /// Pinned upstream F32 VAE SafeTensors object.
    #[arg(long)]
    vae_safetensors: PathBuf,
    /// Fresh output directory. Existing paths are refused.
    #[arg(
        long,
        default_value = ".artifacts/diagnostic-boogu-image-0.1-edit-turbo-1k5-vae-encoder-f32"
    )]
    output: PathBuf,
    /// Copy reused payloads rather than hardlinking them.
    #[arg(long, default_value_t = false)]
    copy: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct InventoryEntry {
    source_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    logical_target_name: Option<String>,
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

#[derive(Debug, Deserialize)]
struct HeaderTensor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

#[derive(Debug, Serialize)]
struct DiagnosticOverlayReport {
    schema_version: u32,
    diagnostic_only: bool,
    production_qualified: bool,
    bundle: String,
    content_digest: Sha256Digest,
    base_bundle: &'static str,
    base_content_digest: &'static str,
    base_payloads_reused: usize,
    materialization: &'static str,
    replaced_stage: &'static str,
    replaced_tensors: usize,
    encoder_weight_files: usize,
    encoder_weight_bytes: u64,
    upstream_vae_bytes: u64,
    upstream_vae_sha256: &'static str,
    decoder_files_reused: usize,
    decoder_bytes_reused: u64,
    non_encoder_weight_declarations_unchanged: bool,
    output_directory: String,
    explicit_browser_base_url: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    prepare(&args)
}

fn prepare(args: &Args) -> Result<(), Box<dyn Error>> {
    require_directory(&args.base)?;
    require_regular_file(&args.vae_safetensors)?;
    if args.output.exists() {
        return Err(format!(
            "output already exists; use a fresh path: {}",
            args.output.display()
        )
        .into());
    }
    validate_upstream_vae(&args.vae_safetensors)?;

    let base_manifest: ArtifactManifest =
        serde_json::from_slice(&fs::read(args.base.join("manifest.json"))?)?;
    validate_base_manifest(&base_manifest)?;
    let mut inventory: Vec<InventoryEntry> =
        serde_json::from_slice(&fs::read(args.base.join(INVENTORY_PATH))?)?;
    validate_base_inventory(&inventory, &base_manifest)?;

    let output_parent = args
        .output
        .parent()
        .ok_or("output directory must have a parent")?;
    let output_parent = if output_parent.as_os_str().is_empty() {
        Path::new(".")
    } else {
        output_parent
    };
    fs::create_dir_all(output_parent)?;
    let stage = fresh_stage_directory(output_parent)?;
    let result = prepare_into(&stage, args, &base_manifest, &mut inventory);
    match result {
        Ok(report) => {
            fs::rename(&stage, &args.output)?;
            let final_report = DiagnosticOverlayReport {
                output_directory: args.output.to_string_lossy().into_owned(),
                explicit_browser_base_url: format!(
                    "http://127.0.0.1:8080/{}",
                    args.output
                        .file_name()
                        .and_then(|value| value.to_str())
                        .ok_or("output directory name is not UTF-8")?
                ),
                ..report
            };
            write_new(
                &args.output.join(OUTPUT_REPORT_PATH),
                &json_bytes(&final_report)?,
            )?;
            println!("{}", serde_json::to_string_pretty(&final_report)?);
            Ok(())
        }
        Err(error) => {
            let _ = fs::remove_dir_all(&stage);
            Err(error)
        }
    }
}

fn prepare_into(
    output: &Path,
    args: &Args,
    base_manifest: &ArtifactManifest,
    inventory: &mut [InventoryEntry],
) -> Result<DiagnosticOverlayReport, Box<dyn Error>> {
    let base_encoder = stage_weight_files(base_manifest, ENCODER_STAGE);
    if base_encoder.len() != 1 {
        return Err(format!(
            "canonical base must have exactly one {ENCODER_STAGE} object, found {}",
            base_encoder.len()
        )
        .into());
    }
    let base_encoder_path = base_encoder[0].path.as_str();
    let base_payloads_reused = materialize_immutable_base_payloads(
        &args.base,
        output,
        base_manifest,
        base_encoder_path,
        args.copy,
    )?;

    let (header_bytes, header) = read_safetensors_header(&args.vae_safetensors)?;
    let encoder_indices = inventory
        .iter()
        .enumerate()
        .filter(|(_, entry)| entry.stage == ENCODER_STAGE)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if encoder_indices.len() != EDIT_TURBO_1K5_VAE_ENCODER_F32_DIAGNOSTIC_TENSORS {
        return Err(format!(
            "base inventory has {} encoder tensors; expected {}",
            encoder_indices.len(),
            EDIT_TURBO_1K5_VAE_ENCODER_F32_DIAGNOSTIC_TENSORS
        )
        .into());
    }

    let mut snapshots = Vec::with_capacity(encoder_indices.len());
    let mut stored_digests = Vec::with_capacity(encoder_indices.len());
    let data_base = header_bytes
        .checked_add(8)
        .ok_or("header offset overflow")?;
    for &index in &encoder_indices {
        let entry = &inventory[index];
        let tensor = header
            .get(&entry.source_name)
            .ok_or_else(|| format!("upstream VAE omits encoder tensor {}", entry.source_name))?;
        validate_encoder_tensor_contract(entry, tensor)?;
        let bytes = read_source_tensor(&args.vae_safetensors, data_base, tensor)?;
        let values = bytes
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect::<Vec<_>>();
        if values.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "upstream encoder tensor {} contains a non-finite value",
                entry.source_name
            )
            .into());
        }
        let values = if entry.transform == TensorTransform::Transpose2d {
            transpose_values(&values, tensor.shape[0], tensor.shape[1])
        } else {
            values
        };
        let shape = entry
            .stored_shape
            .clone()
            .ok_or_else(|| format!("{} omits stored shape", entry.target_name))?;
        let data = TensorData::new(values, shape);
        stored_digests.push(Sha256Digest::calculate(data.bytes.as_ref()));
        snapshots.push(TensorSnapshot::from_data(
            data,
            vec![entry.target_name.clone()],
            vec!["Imported".into()],
            deterministic_param_id(&format!("vae:{}", entry.target_name)),
        ));
    }

    fs::create_dir_all(output.join("objects"))?;
    let temporary = output.join("objects/.flux-vae-encoder-f32.tmp");
    BurnpackWriter::new(snapshots)
        .with_metadata("component", ENCODER_STAGE)
        .with_metadata("profile", "f16-qwen-vision-f32+vae-encoder-f32")
        .with_metadata("source_layout", "pytorch")
        .with_metadata("layout_contract", INVENTORY_PATH)
        .write_to_file(&temporary)?;
    validate_encoder_burnpack(&temporary, inventory, &encoder_indices, &stored_digests)?;
    let encoder_size = fs::metadata(&temporary)?.len();
    if encoder_size > DEFAULT_MAX_SHARD_BYTES {
        return Err(format!(
            "F32 VAE encoder Burnpack is {encoder_size} bytes, exceeding browser bound {DEFAULT_MAX_SHARD_BYTES}"
        )
        .into());
    }
    let encoder_sha256 = hash_file(&temporary)?;
    let encoder_relative = format!("objects/{encoder_sha256}.bpk");
    fs::rename(&temporary, output.join(&encoder_relative))?;

    for (entry_index, digest) in encoder_indices.iter().copied().zip(stored_digests) {
        let entry = &mut inventory[entry_index];
        entry.stored_dtype = Some("f32".into());
        entry.stored_sha256 = Some(digest);
        entry.burnpack_object = Some(encoder_relative.clone());
    }
    fs::create_dir_all(output.join("metadata"))?;
    let inventory_bytes = json_bytes(&inventory)?;
    write_new(&output.join(INVENTORY_PATH), &inventory_bytes)?;

    let mut manifest = base_manifest.clone();
    manifest.bundle = burn_image::ArtifactBundleId::new(DIAGNOSTIC_BUNDLE_ID)?;
    manifest
        .files
        .retain(|file| file.path.as_str() != base_encoder_path);
    let inventory_file = manifest
        .files
        .iter_mut()
        .find(|file| file.path.as_str() == INVENTORY_PATH)
        .ok_or("base manifest omits tensor inventory")?;
    inventory_file.size = inventory_bytes.len() as u64;
    inventory_file.sha256 = Sha256Digest::calculate(&inventory_bytes);
    manifest.files.push(ArtifactFile {
        path: ArtifactPath::new(&encoder_relative)?,
        size: encoder_size,
        sha256: encoder_sha256,
        role: ArtifactFileRole::Weights,
        component: Some(burn_image::ArtifactComponentId::new(ENCODER_STAGE)?),
        shard: None,
    });
    manifest
        .files
        .sort_by(|left, right| left.path.cmp(&right.path));
    stamp_edit_turbo_1k5_vae_encoder_f32_diagnostic_metadata(&mut manifest)?;
    let content_digest = manifest.seal()?;
    let manifest_bytes = json_bytes(&manifest)?;
    write_new(&output.join("manifest.json"), &manifest_bytes)?;
    let evidence = validate_edit_turbo_1k5_vae_encoder_f32_diagnostic_manifest(&manifest)?;

    let decoder_files = stage_weight_files(&manifest, DECODER_STAGE);
    let decoder_bytes = decoder_files.iter().try_fold(0_u64, |total, file| {
        total
            .checked_add(file.size)
            .ok_or("decoder byte count overflow")
    })?;
    Ok(DiagnosticOverlayReport {
        schema_version: 1,
        diagnostic_only: true,
        production_qualified: false,
        bundle: manifest.bundle.to_string(),
        content_digest,
        base_bundle: BASE_BUNDLE_ID,
        base_content_digest: LEGACY_EDIT_TURBO_1K5_F16_QWEN_VISION_F32_CONTENT_DIGEST,
        base_payloads_reused,
        materialization: if args.copy { "copy" } else { "hardlink" },
        replaced_stage: ENCODER_STAGE,
        replaced_tensors: evidence.replaced_tensors,
        encoder_weight_files: evidence.encoder_weight_files,
        encoder_weight_bytes: evidence.encoder_weight_bytes,
        upstream_vae_bytes: EDIT_TURBO_1K5_VAE_SOURCE_BYTES,
        upstream_vae_sha256: EDIT_TURBO_1K5_VAE_SOURCE_SHA256,
        decoder_files_reused: decoder_files.len(),
        decoder_bytes_reused: decoder_bytes,
        non_encoder_weight_declarations_unchanged: non_encoder_weights_equal(
            base_manifest,
            &manifest,
        ),
        output_directory: String::new(),
        explicit_browser_base_url: String::new(),
    })
}

fn validate_base_manifest(manifest: &ArtifactManifest) -> Result<(), Box<dyn Error>> {
    manifest.validate_sealed()?;
    let expected_digest =
        Sha256Digest::from_hex(LEGACY_EDIT_TURBO_1K5_F16_QWEN_VISION_F32_CONTENT_DIGEST)?;
    let identity = BooguReleaseIdentity::canonical(burn_boogu::BooguVariant::Image01EditTurbo1k5);
    if manifest.bundle.as_str() != BASE_BUNDLE_ID
        || manifest.profile.as_str() != "f16-qwen-vision-f32"
        || manifest.model.as_str() != "Boogu/Boogu-Image-0.1-Edit-Turbo-1K5"
        || manifest.model_revision != identity.model_revision
        || manifest.content_digest != Some(expected_digest)
        || manifest
            .metadata
            .keys()
            .any(|key| key.starts_with("diagnostic_"))
    {
        return Err("base is not the exact legacy flat 1.5K mixed-F16 bundle".into());
    }
    Ok(())
}

fn validate_base_inventory(
    inventory: &[InventoryEntry],
    manifest: &ArtifactManifest,
) -> Result<(), Box<dyn Error>> {
    let mut encoder = 0_usize;
    let mut decoder = 0_usize;
    let mut targets = BTreeSet::new();
    for entry in inventory {
        if entry.included && !targets.insert(entry.target_name.as_str()) {
            return Err(format!("duplicate inventory target {}", entry.target_name).into());
        }
        if entry.stage == ENCODER_STAGE {
            encoder += 1;
            if entry.owner != TensorOwner::FluxVae
                || entry.component != "vae"
                || entry.source_dtype != "F32"
                || entry.stored_dtype.as_deref() != Some("f16")
                || entry.quantized
                || entry.source_row_range.is_some()
            {
                return Err(format!(
                    "base encoder entry {} violates the F32-source/F16-storage contract",
                    entry.target_name
                )
                .into());
            }
        } else if entry.owner == TensorOwner::FluxVae {
            if entry.stage != DECODER_STAGE || entry.stored_dtype.as_deref() != Some("f16") {
                return Err(format!(
                    "unexpected VAE stage/storage for {}: {}/{}",
                    entry.target_name,
                    entry.stage,
                    entry.stored_dtype.as_deref().unwrap_or("none")
                )
                .into());
            }
            decoder += 1;
        }
    }
    if encoder != EDIT_TURBO_1K5_VAE_ENCODER_F32_DIAGNOSTIC_TENSORS || decoder != 138 {
        return Err(format!(
            "unexpected VAE partition: encoder={encoder}, decoder={decoder}; expected 106/138"
        )
        .into());
    }
    let declared_encoder = stage_weight_files(manifest, ENCODER_STAGE);
    let declared_decoder = stage_weight_files(manifest, DECODER_STAGE);
    if declared_encoder.len() != 1 || declared_decoder.len() != 1 {
        return Err("base manifest must declare one encoder and one decoder object".into());
    }
    Ok(())
}

fn validate_upstream_vae(path: &Path) -> Result<(), Box<dyn Error>> {
    let size = fs::metadata(path)?.len();
    let digest = hash_file(path)?;
    if size != EDIT_TURBO_1K5_VAE_SOURCE_BYTES
        || digest.to_string() != EDIT_TURBO_1K5_VAE_SOURCE_SHA256
    {
        return Err(format!(
            "pinned upstream VAE mismatch: expected {} bytes / {}, found {size} / {digest}",
            EDIT_TURBO_1K5_VAE_SOURCE_BYTES, EDIT_TURBO_1K5_VAE_SOURCE_SHA256
        )
        .into());
    }
    Ok(())
}

fn read_safetensors_header(
    path: &Path,
) -> Result<(u64, BTreeMap<String, HeaderTensor>), Box<dyn Error>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut length = [0_u8; 8];
    reader.read_exact(&mut length)?;
    let header_len = u64::from_le_bytes(length);
    if header_len == 0 || header_len > 100 * 1024 * 1024 {
        return Err("invalid SafeTensors header length".into());
    }
    let mut bytes = vec![0_u8; usize::try_from(header_len)?];
    reader.read_exact(&mut bytes)?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)?;
    let object = value
        .as_object()
        .ok_or("SafeTensors header must be a JSON object")?;
    let mut tensors = BTreeMap::new();
    for (name, value) in object {
        if name != "__metadata__" {
            tensors.insert(name.clone(), serde_json::from_value(value.clone())?);
        }
    }
    Ok((header_len, tensors))
}

fn validate_encoder_tensor_contract(
    entry: &InventoryEntry,
    tensor: &HeaderTensor,
) -> Result<(), Box<dyn Error>> {
    if tensor.dtype != "F32"
        || tensor.shape != entry.source_shape
        || entry.source_dtype != "F32"
        || entry.source_file != "vae/diffusion_pytorch_model.safetensors"
        || entry.owner != TensorOwner::FluxVae
        || entry.stage != ENCODER_STAGE
        || !entry.included
        || entry.quantized
        || entry.source_row_range.is_some()
    {
        return Err(format!(
            "encoder tensor {} does not match the sealed upstream contract",
            entry.source_name
        )
        .into());
    }
    let elements = tensor.shape.iter().try_fold(1_u64, |total, &dimension| {
        total.checked_mul(dimension as u64)
    });
    let expected_bytes = elements.and_then(|total| total.checked_mul(4));
    let [start, end] = tensor.data_offsets;
    if start >= end || expected_bytes != Some(end - start) || entry.source_bytes != end - start {
        return Err(format!("invalid source range for {}", entry.source_name).into());
    }
    Ok(())
}

fn read_source_tensor(
    path: &Path,
    data_base: u64,
    tensor: &HeaderTensor,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let [start, end] = tensor.data_offsets;
    let absolute = data_base
        .checked_add(start)
        .ok_or("source offset overflow")?;
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(absolute))?;
    let mut bytes = vec![0_u8; usize::try_from(end - start)?];
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn validate_encoder_burnpack(
    path: &Path,
    inventory: &[InventoryEntry],
    indices: &[usize],
    digests: &[Sha256Digest],
) -> Result<(), Box<dyn Error>> {
    let mut store = BurnpackStore::from_file(path).auto_extension(false);
    let snapshots = store.get_all_snapshots()?;
    if snapshots.len() != indices.len() || indices.len() != digests.len() {
        return Err("F32 encoder Burnpack tensor count changed during serialization".into());
    }
    for (&index, &expected_digest) in indices.iter().zip(digests) {
        let entry = &inventory[index];
        let snapshot = snapshots
            .get(&entry.target_name)
            .ok_or_else(|| format!("F32 Burnpack omits {}", entry.target_name))?;
        if snapshot.dtype != burn::tensor::DType::F32
            || snapshot.shape.as_slice() != entry.stored_shape.as_deref().unwrap_or(&[])
        {
            return Err(format!(
                "F32 Burnpack dtype/shape mismatch for {}",
                entry.target_name
            )
            .into());
        }
        let data = snapshot.to_data()?;
        if Sha256Digest::calculate(data.bytes.as_ref()) != expected_digest {
            return Err(format!("F32 Burnpack payload changed for {}", entry.target_name).into());
        }
    }
    Ok(())
}

/// Reuse only payloads that the diagnostic will never rewrite.
///
/// Both replacement outputs are excluded here rather than at their individual write sites. This
/// prevents a create/truncate call from following a hardlink back into the canonical base bundle.
fn materialize_immutable_base_payloads(
    base: &Path,
    output: &Path,
    manifest: &ArtifactManifest,
    encoder_path: &str,
    copy: bool,
) -> Result<usize, Box<dyn Error>> {
    let mut count = 0;
    for file in manifest
        .files
        .iter()
        .filter(|file| file.path.as_str() != encoder_path && file.path.as_str() != INVENTORY_PATH)
    {
        let relative = Path::new(file.path.as_str());
        require_safe_relative(relative)?;
        let source = base.join(relative);
        require_regular_file(&source)?;
        let destination = output.join(relative);
        fs::create_dir_all(destination.parent().ok_or("payload has no parent")?)?;
        if copy {
            let mut input = File::open(&source)?;
            let mut target = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&destination)?;
            let copied = std::io::copy(&mut input, &mut target)?;
            target.sync_all()?;
            if copied != file.size {
                return Err(format!("short copy for {}", file.path).into());
            }
        } else {
            fs::hard_link(&source, &destination)?;
        }
        count += 1;
    }
    Ok(count)
}

fn stage_weight_files<'a>(manifest: &'a ArtifactManifest, stage: &str) -> Vec<&'a ArtifactFile> {
    manifest
        .files
        .iter()
        .filter(|file| {
            file.role == ArtifactFileRole::Weights
                && file.component.as_ref().map(|value| value.as_str()) == Some(stage)
        })
        .collect()
}

fn non_encoder_weights_equal(left: &ArtifactManifest, right: &ArtifactManifest) -> bool {
    let select = |manifest: &ArtifactManifest| {
        manifest
            .files
            .iter()
            .filter(|file| {
                file.role == ArtifactFileRole::Weights
                    && file.component.as_ref().map(|value| value.as_str()) != Some(ENCODER_STAGE)
            })
            .map(|file| {
                (
                    file.path.clone(),
                    file.size,
                    file.sha256,
                    file.role,
                    file.component.clone(),
                    file.shard,
                )
            })
            .collect::<Vec<_>>()
    };
    select(left) == select(right)
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

fn hash_file(path: &Path) -> Result<Sha256Digest, Box<dyn Error>> {
    let mut reader = BufReader::with_capacity(1024 * 1024, File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(Sha256Digest::from_bytes(hasher.finalize().into()))
}

fn fresh_stage_directory(parent: &Path) -> Result<PathBuf, Box<dyn Error>> {
    for attempt in 0..100_u32 {
        let path = parent.join(format!(
            ".vae-encoder-f32.prepare.{}.{}",
            std::process::id(),
            attempt
        ));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Err("could not allocate a fresh staging directory".into())
}

fn json_bytes(value: &impl Serialize) -> Result<Vec<u8>, serde_json::Error> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn require_directory(path: &Path) -> Result<(), Box<dyn Error>> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(format!("expected a non-symlink directory: {}", path.display()).into());
    }
    Ok(())
}

fn require_regular_file(path: &Path) -> Result<(), Box<dyn Error>> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() == 0 {
        return Err(format!("expected a non-empty regular file: {}", path.display()).into());
    }
    Ok(())
}

fn require_safe_relative(path: &Path) -> Result<(), Box<dyn Error>> {
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!("unsafe manifest path: {}", path.display()).into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_image::{
        ARTIFACT_MANIFEST_SCHEMA_VERSION, ArtifactBundleId, ArtifactComponent, ArtifactComponentId,
        ArtifactProfileId, ModelId, NumericFormat,
    };

    fn test_file(
        path: &str,
        bytes: &[u8],
        role: ArtifactFileRole,
        component: Option<&str>,
    ) -> ArtifactFile {
        ArtifactFile {
            path: ArtifactPath::new(path).unwrap(),
            size: bytes.len() as u64,
            sha256: Sha256Digest::calculate(bytes),
            role,
            component: component.map(|value| ArtifactComponentId::new(value).unwrap()),
            shard: None,
        }
    }

    fn test_manifest(
        files: Vec<ArtifactFile>,
        components: impl IntoIterator<Item = &'static str>,
    ) -> ArtifactManifest {
        ArtifactManifest {
            schema_version: ARTIFACT_MANIFEST_SCHEMA_VERSION,
            bundle: ArtifactBundleId::new("diagnostic-materialization-test").unwrap(),
            profile: ArtifactProfileId::new("f16-qwen-vision-f32").unwrap(),
            model: ModelId::new("tests/diagnostic-materialization").unwrap(),
            model_revision: "immutable-test-revision".into(),
            numeric_format: NumericFormat::Other("f16-qwen-vision-f32".into()),
            components: components
                .into_iter()
                .map(|id| ArtifactComponent {
                    id: ArtifactComponentId::new(id).unwrap(),
                    required: true,
                })
                .collect(),
            files,
            dependencies: Vec::new(),
            metadata: BTreeMap::new(),
            content_digest: None,
        }
    }

    fn tiny_materialization_manifest(
        inventory: &[u8],
        encoder: &[u8],
        reused: &[u8],
    ) -> ArtifactManifest {
        test_manifest(
            vec![
                test_file(INVENTORY_PATH, inventory, ArtifactFileRole::Metadata, None),
                test_file(
                    "objects/base-encoder.bpk",
                    encoder,
                    ArtifactFileRole::Weights,
                    Some(ENCODER_STAGE),
                ),
                test_file(
                    "metadata/source/vae/config.json",
                    reused,
                    ArtifactFileRole::Config,
                    None,
                ),
            ],
            [ENCODER_STAGE],
        )
    }

    fn synthetic_vae_manifest() -> ArtifactManifest {
        test_manifest(
            vec![
                test_file(INVENTORY_PATH, b"[]", ArtifactFileRole::Metadata, None),
                test_file(
                    "objects/base-encoder.bpk",
                    b"encoder",
                    ArtifactFileRole::Weights,
                    Some(ENCODER_STAGE),
                ),
                test_file(
                    "objects/base-decoder.bpk",
                    b"decoder",
                    ArtifactFileRole::Weights,
                    Some(DECODER_STAGE),
                ),
            ],
            [ENCODER_STAGE, DECODER_STAGE],
        )
    }

    fn synthetic_inventory_entry(stage: &str, index: usize) -> InventoryEntry {
        let prefix = if stage == ENCODER_STAGE {
            "encoder"
        } else {
            "decoder"
        };
        let name = format!("{prefix}.tensor_{index:03}.weight");
        InventoryEntry {
            source_name: name.clone(),
            logical_target_name: None,
            target_name: name,
            owner: TensorOwner::FluxVae,
            component: "vae".into(),
            stage: stage.into(),
            transform: TensorTransform::Identity,
            source_file: "vae/diffusion_pytorch_model.safetensors".into(),
            source_dtype: "F32".into(),
            source_shape: vec![1],
            source_row_range: None,
            included: true,
            stored_dtype: Some("f16".into()),
            stored_shape: Some(vec![1]),
            source_offset: (index as u64) * 4,
            source_bytes: 4,
            quantized: false,
            stored_sha256: Some(Sha256Digest::calculate(&index.to_le_bytes())),
            burnpack_object: Some(format!("objects/base-{prefix}.bpk")),
        }
    }

    fn synthetic_vae_inventory() -> Vec<InventoryEntry> {
        (0..EDIT_TURBO_1K5_VAE_ENCODER_F32_DIAGNOSTIC_TENSORS)
            .map(|index| synthetic_inventory_entry(ENCODER_STAGE, index))
            .chain((0..138).map(|index| synthetic_inventory_entry(DECODER_STAGE, index)))
            .collect()
    }

    #[test]
    fn diagnostic_materialization_never_aliases_mutable_inventory_correctness() {
        let base = tempfile::tempdir().unwrap();
        let output = tempfile::tempdir().unwrap();
        let inventory = br#"[{"stored_dtype":"f16"}]"#;
        let replacement = br#"[{"stored_dtype":"f32"}]"#;
        let encoder = b"base encoder";
        let reused = br#"{"force_upcast":true}"#;
        let inventory_path = base.path().join(INVENTORY_PATH);
        let encoder_path = base.path().join("objects/base-encoder.bpk");
        let reused_path = base.path().join("metadata/source/vae/config.json");
        for (path, bytes) in [
            (&inventory_path, inventory.as_slice()),
            (&encoder_path, encoder.as_slice()),
            (&reused_path, reused.as_slice()),
        ] {
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, bytes).unwrap();
        }
        let manifest = tiny_materialization_manifest(inventory, encoder, reused);
        let before_bytes = fs::read(&inventory_path).unwrap();
        let before_hash = hash_file(&inventory_path).unwrap();
        #[cfg(unix)]
        let before_inode = {
            use std::os::unix::fs::MetadataExt;
            fs::metadata(&inventory_path).unwrap().ino()
        };

        let reused_count = materialize_immutable_base_payloads(
            base.path(),
            output.path(),
            &manifest,
            "objects/base-encoder.bpk",
            false,
        )
        .unwrap();
        assert_eq!(reused_count, 1);
        fs::create_dir_all(output.path().join("metadata")).unwrap();
        write_new(&output.path().join(INVENTORY_PATH), replacement).unwrap();

        assert_eq!(fs::read(&inventory_path).unwrap(), before_bytes);
        assert_eq!(hash_file(&inventory_path).unwrap(), before_hash);
        assert_eq!(
            fs::read(output.path().join(INVENTORY_PATH)).unwrap(),
            replacement
        );
        assert!(!output.path().join("objects/base-encoder.bpk").exists());
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let base_inventory = fs::metadata(&inventory_path).unwrap();
            let output_inventory = fs::metadata(output.path().join(INVENTORY_PATH)).unwrap();
            let base_reused = fs::metadata(&reused_path).unwrap();
            let output_reused =
                fs::metadata(output.path().join("metadata/source/vae/config.json")).unwrap();
            assert_eq!(base_inventory.ino(), before_inode);
            assert_ne!(base_inventory.ino(), output_inventory.ino());
            assert_eq!(base_reused.ino(), output_reused.ino());
        }
    }

    #[test]
    fn only_exact_encoder_partition_is_replaceable_correctness() {
        let manifest = synthetic_vae_manifest();
        let inventory = synthetic_vae_inventory();
        validate_base_inventory(&inventory, &manifest).unwrap();
        assert_eq!(
            inventory
                .iter()
                .filter(|entry| entry.stage == ENCODER_STAGE)
                .count(),
            106
        );
        assert_eq!(
            inventory
                .iter()
                .filter(|entry| entry.stage == DECODER_STAGE)
                .count(),
            138
        );
    }

    #[test]
    fn non_encoder_comparison_rejects_decoder_replacement_correctness() {
        let base = synthetic_vae_manifest();
        let mut changed = base.clone();
        let decoder = changed
            .files
            .iter_mut()
            .find(|file| file.component.as_ref().map(|value| value.as_str()) == Some(DECODER_STAGE))
            .unwrap();
        decoder.sha256 = Sha256Digest::calculate(b"not the decoder");
        assert!(!non_encoder_weights_equal(&base, &changed));
    }

    #[test]
    fn transpose_matches_burn_linear_layout_correctness() {
        assert_eq!(
            transpose_values(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }
}
