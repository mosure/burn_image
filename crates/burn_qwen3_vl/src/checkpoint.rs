//! Strict native Hugging Face SafeTensors inspection and loading.
//!
//! Loading validates the complete cross-shard key and shape set before model allocation, then
//! applies one memory-mapped shard at a time. Runtime/CDN Burnpack streaming belongs to the
//! artifact layer; this module is the deterministic import path for published checkpoints.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    rc::Rc,
};

use burn::tensor::{DType as BurnDType, backend::Backend};
use burn_store::{KeyRemapper, ModuleAdapter, ModuleSnapshot, SafetensorsStore, TensorSnapshot};
use safetensors::Dtype;
use serde::Deserialize;

use crate::{
    Qwen3VlBuilder, Qwen3VlConfig, Qwen3VlError, Qwen3VlForConditionalGeneration, Qwen3VlModel,
    Result, WeightInventory,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointTensor {
    pub key: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub shard: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointInspection {
    pub shards: Vec<PathBuf>,
    pub tensors: Vec<CheckpointTensor>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointLoadReport {
    pub shards_loaded: usize,
    pub tensors_validated: usize,
    pub tensors_applied: usize,
}

/// Floating-point precision used when materializing a Hugging Face checkpoint.
///
/// `Source` is useful for diagnosing backend support. Native WGPU/WebGPU deployments normally
/// convert published BF16 tensors to F32 or F16 because browser WebGPU does not expose BF16.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CheckpointDType {
    #[default]
    Source,
    F32,
    F16,
}

/// Resolved Hugging Face checkpoint shards and optional index ownership map.
#[derive(Debug, Clone)]
pub struct HfCheckpoint {
    shards: Vec<PathBuf>,
    weight_map: Option<BTreeMap<String, String>>,
}

#[derive(Debug, Deserialize)]
struct HfIndex {
    weight_map: BTreeMap<String, String>,
}

impl HfCheckpoint {
    pub fn from_shards(shards: impl IntoIterator<Item = PathBuf>) -> Result<Self> {
        let mut shards = shards.into_iter().collect::<Vec<_>>();
        shards.sort();
        shards.dedup();
        if shards.is_empty() {
            return Err(Qwen3VlError::Checkpoint(
                "at least one SafeTensors shard is required".into(),
            ));
        }
        Ok(Self {
            shards,
            weight_map: None,
        })
    }

    pub fn from_index(index_path: impl AsRef<Path>) -> Result<Self> {
        let index_path = index_path.as_ref();
        let bytes = std::fs::read(index_path).map_err(|error| {
            Qwen3VlError::Checkpoint(format!(
                "failed to read checkpoint index {}: {error}",
                index_path.display()
            ))
        })?;
        let index: HfIndex = serde_json::from_slice(&bytes).map_err(|error| {
            Qwen3VlError::Checkpoint(format!(
                "invalid checkpoint index {}: {error}",
                index_path.display()
            ))
        })?;
        if index.weight_map.is_empty() {
            return Err(Qwen3VlError::Checkpoint(
                "checkpoint index weight_map must not be empty".into(),
            ));
        }
        let directory = index_path.parent().unwrap_or_else(|| Path::new("."));
        let shards = index
            .weight_map
            .values()
            .map(|name| directory.join(name))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        Ok(Self {
            shards,
            weight_map: Some(index.weight_map),
        })
    }

    pub fn shards(&self) -> &[PathBuf] {
        &self.shards
    }

    /// Read only SafeTensors headers and reject missing, unknown, duplicate, misplaced, or
    /// shape-incompatible tensors before model memory is allocated.
    pub fn inspect(&self, inventory: &WeightInventory) -> Result<CheckpointInspection> {
        let mut tensors = Vec::new();
        let mut observed = BTreeSet::new();
        for shard in &self.shards {
            let header = read_safetensors_header(shard)?;
            let shard_name = shard
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| {
                    Qwen3VlError::Checkpoint(format!(
                        "checkpoint shard has a non-UTF8 filename: {}",
                        shard.display()
                    ))
                })?;
            for (key, info) in header.tensors() {
                if !observed.insert(key.clone()) {
                    return Err(Qwen3VlError::Checkpoint(format!(
                        "tensor {key:?} occurs in more than one shard"
                    )));
                }
                if let Some(weight_map) = &self.weight_map {
                    match weight_map.get(&key) {
                        Some(expected_shard) if expected_shard == shard_name => {}
                        Some(expected_shard) => {
                            return Err(Qwen3VlError::Checkpoint(format!(
                                "index assigns {key:?} to {expected_shard:?}, found in {shard_name:?}"
                            )));
                        }
                        None => {
                            return Err(Qwen3VlError::Checkpoint(format!(
                                "tensor {key:?} is absent from index weight_map"
                            )));
                        }
                    }
                }
                if !matches!(info.dtype, Dtype::BF16 | Dtype::F16 | Dtype::F32) {
                    return Err(Qwen3VlError::Checkpoint(format!(
                        "unsupported dtype {:?} for tensor {key:?}",
                        info.dtype
                    )));
                }
                tensors.push(CheckpointTensor {
                    key,
                    shape: info.shape.clone(),
                    dtype: format!("{:?}", info.dtype),
                    shard: shard.clone(),
                });
            }
        }
        if let Some(weight_map) = &self.weight_map
            && weight_map.keys().cloned().collect::<BTreeSet<_>>() != observed
        {
            let missing_files = weight_map
                .keys()
                .filter(|key| !observed.contains(*key))
                .take(16)
                .cloned()
                .collect::<Vec<_>>();
            return Err(Qwen3VlError::Checkpoint(format!(
                "index references tensors absent from shard headers: {missing_files:?}"
            )));
        }
        inventory
            .validate_entries(
                tensors
                    .iter()
                    .map(|tensor| (tensor.key.as_str(), tensor.shape.as_slice())),
            )
            .into_result()?;
        tensors.sort_by(|left, right| left.key.cmp(&right.key));
        Ok(CheckpointInspection {
            shards: self.shards.clone(),
            tensors,
        })
    }
}

pub fn load_causal_lm_from_safetensors<B: Backend>(
    config: Qwen3VlConfig,
    device: &B::Device,
    checkpoint: &HfCheckpoint,
) -> Result<(Qwen3VlForConditionalGeneration<B>, CheckpointLoadReport)> {
    load_causal_lm_from_safetensors_with_dtype(config, device, checkpoint, CheckpointDType::Source)
}

pub fn load_causal_lm_from_safetensors_with_dtype<B: Backend>(
    config: Qwen3VlConfig,
    device: &B::Device,
    checkpoint: &HfCheckpoint,
    dtype: CheckpointDType,
) -> Result<(Qwen3VlForConditionalGeneration<B>, CheckpointLoadReport)> {
    let builder = Qwen3VlBuilder::new(config)?;
    let inventory = builder.causal_lm_inventory();
    let inspection = checkpoint.inspect(&inventory)?;
    validate_backend_dtype::<B>(device, dtype, &inspection)?;
    let mut model = builder.build_causal_lm(device)?;
    let applied = apply_shards(
        &mut model,
        &inspection.shards,
        &inventory,
        dtype,
        &BTreeSet::new(),
    )?;
    Ok((
        model,
        CheckpointLoadReport {
            shards_loaded: inspection.shards.len(),
            tensors_validated: inspection.tensors.len(),
            tensors_applied: applied,
        },
    ))
}

pub fn load_base_from_safetensors<B: Backend>(
    config: Qwen3VlConfig,
    device: &B::Device,
    checkpoint: &HfCheckpoint,
) -> Result<(Qwen3VlModel<B>, CheckpointLoadReport)> {
    load_base_from_safetensors_with_dtype(config, device, checkpoint, CheckpointDType::Source)
}

pub fn load_base_from_safetensors_with_dtype<B: Backend>(
    config: Qwen3VlConfig,
    device: &B::Device,
    checkpoint: &HfCheckpoint,
    dtype: CheckpointDType,
) -> Result<(Qwen3VlModel<B>, CheckpointLoadReport)> {
    let builder = Qwen3VlBuilder::new(config)?;
    let inventory = builder.base_inventory();
    // Published conditional checkpoints contain an untied LM head. Validate it strictly at the
    // header level, but never allocate it for base-model conditioning.
    let checkpoint_inventory = builder.causal_lm_inventory();
    let inspection = checkpoint.inspect(&checkpoint_inventory)?;
    validate_backend_dtype::<B>(device, dtype, &inspection)?;
    let mut model = builder.build_base(device)?;
    let allowed_unused = checkpoint_inventory
        .specs()
        .iter()
        .filter(|spec| inventory.source_to_target(&spec.source).is_none())
        .map(|spec| spec.source.clone())
        .collect::<BTreeSet<_>>();
    let applied = apply_shards(
        &mut model,
        &inspection.shards,
        &inventory,
        dtype,
        &allowed_unused,
    )?;
    Ok((
        model,
        CheckpointLoadReport {
            shards_loaded: inspection.shards.len(),
            tensors_validated: inspection.tensors.len(),
            tensors_applied: applied,
        },
    ))
}

fn apply_shards<B: Backend, M: ModuleSnapshot<B>>(
    model: &mut M,
    shards: &[PathBuf],
    inventory: &WeightInventory,
    dtype: CheckpointDType,
    allowed_unused: &BTreeSet<String>,
) -> Result<usize> {
    let remapper = checkpoint_remapper(inventory)?;
    let mut applied = BTreeSet::new();
    for shard in shards {
        let mut store = SafetensorsStore::from_file(shard)
            .allow_partial(true)
            .remap(remapper.clone())
            .with_from_adapter(CheckpointDTypeAdapter(dtype))
            .validate(true);
        let result = model.load_from(&mut store).map_err(|error| {
            Qwen3VlError::Checkpoint(format!(
                "failed to apply shard {}: {error:?}",
                shard.display()
            ))
        })?;
        let unexpected_unused = result
            .unused
            .iter()
            .filter(|name| !allowed_unused.contains(*name))
            .collect::<Vec<_>>();
        if !result.errors.is_empty() || !unexpected_unused.is_empty() || !result.skipped.is_empty()
        {
            return Err(Qwen3VlError::Checkpoint(format!(
                "shard {} was not applied strictly: errors={:?}, unexpected_unused={:?}, skipped={:?}",
                shard.display(),
                result.errors,
                unexpected_unused,
                result.skipped
            )));
        }
        applied.extend(result.applied);
    }
    if applied.len() != inventory.specs().len() {
        return Err(Qwen3VlError::Checkpoint(format!(
            "applied {} unique tensors, expected {}",
            applied.len(),
            inventory.specs().len()
        )));
    }
    Ok(applied.len())
}

fn validate_backend_dtype<B: Backend>(
    device: &B::Device,
    dtype: CheckpointDType,
    inspection: &CheckpointInspection,
) -> Result<()> {
    let target = match dtype {
        CheckpointDType::Source => None,
        CheckpointDType::F32 => Some(BurnDType::F32),
        CheckpointDType::F16 => Some(BurnDType::F16),
    };
    if let Some(target) = target
        && !B::supports_dtype(device, target)
    {
        return Err(Qwen3VlError::Checkpoint(format!(
            "backend {} does not generally support requested checkpoint dtype {target:?}",
            B::name(device)
        )));
    }
    let has_bf16 = inspection
        .tensors
        .iter()
        .any(|tensor| tensor.dtype == "BF16");
    let backend_name = B::name(device);
    if dtype == CheckpointDType::Source
        && has_bf16
        && backend_name.to_ascii_lowercase().contains("wgpu")
    {
        return Err(Qwen3VlError::Checkpoint(format!(
            "source BF16 execution is rejected on {backend_name}: Burn 0.21 WGPU layer matmuls overflow on the pinned Qwen checkpoint; choose CheckpointDType::F16 for WebGPU or F32 for native parity"
        )));
    }
    if dtype == CheckpointDType::Source && has_bf16 && !B::supports_dtype(device, BurnDType::BF16) {
        return Err(Qwen3VlError::Checkpoint(format!(
            "backend {backend_name} cannot materialize source BF16; choose an explicit checkpoint dtype"
        )));
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct CheckpointDTypeAdapter(CheckpointDType);

impl ModuleAdapter for CheckpointDTypeAdapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        let target = match self.0 {
            CheckpointDType::Source => return snapshot.clone(),
            CheckpointDType::F32 => BurnDType::F32,
            CheckpointDType::F16 => BurnDType::F16,
        };
        if snapshot.dtype == target
            || !matches!(
                snapshot.dtype,
                BurnDType::BF16 | BurnDType::F16 | BurnDType::F32
            )
        {
            return snapshot.clone();
        }
        let data = snapshot.clone_data_fn();
        TensorSnapshot::from_closure(
            Rc::new(move || Ok(data()?.convert_dtype(target))),
            target,
            snapshot.shape.clone(),
            snapshot.path_stack.clone().unwrap_or_default(),
            snapshot.container_stack.clone().unwrap_or_default(),
            snapshot.tensor_id.unwrap_or_default(),
        )
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

fn checkpoint_remapper(inventory: &WeightInventory) -> Result<KeyRemapper> {
    let mut remapper = KeyRemapper::new();
    for spec in inventory
        .specs()
        .iter()
        .filter(|spec| spec.source != spec.target)
    {
        let escaped = spec.source.replace('.', r"\.");
        let pattern = format!(r"^{escaped}$");
        remapper = remapper
            .add_pattern(&pattern, &spec.target)
            .map_err(|error| {
                Qwen3VlError::Checkpoint(format!(
                    "invalid key mapping {} -> {}: {error}",
                    spec.source, spec.target
                ))
            })?;
    }
    Ok(remapper)
}

fn read_safetensors_header(path: &Path) -> Result<safetensors::tensor::Metadata> {
    let mut file = File::open(path).map_err(|error| {
        Qwen3VlError::Checkpoint(format!("failed to open {}: {error}", path.display()))
    })?;
    let file_len = file
        .metadata()
        .map_err(|error| {
            Qwen3VlError::Checkpoint(format!("failed to stat {}: {error}", path.display()))
        })?
        .len();
    let mut length_bytes = [0_u8; 8];
    file.read_exact(&mut length_bytes).map_err(|error| {
        Qwen3VlError::Checkpoint(format!(
            "failed to read SafeTensors header length {}: {error}",
            path.display()
        ))
    })?;
    let header_len = usize::try_from(u64::from_le_bytes(length_bytes)).map_err(|_| {
        Qwen3VlError::Checkpoint(format!("header length overflow in {}", path.display()))
    })?;
    if header_len > 100_000_000 {
        return Err(Qwen3VlError::Checkpoint(format!(
            "unreasonably large SafeTensors header ({header_len} bytes) in {}",
            path.display()
        )));
    }
    let mut header = vec![0_u8; 8 + header_len];
    header[..8].copy_from_slice(&length_bytes);
    file.read_exact(&mut header[8..]).map_err(|error| {
        Qwen3VlError::Checkpoint(format!(
            "failed to read SafeTensors header {}: {error}",
            path.display()
        ))
    })?;
    let metadata: safetensors::tensor::Metadata =
        serde_json::from_slice(&header[8..]).map_err(|error| {
            Qwen3VlError::Checkpoint(format!(
                "invalid SafeTensors header {}: {error}",
                path.display()
            ))
        })?;
    let data_offset = 8 + header_len;
    let expected_len = u64::try_from(data_offset + metadata.data_len()).map_err(|_| {
        Qwen3VlError::Checkpoint(format!("file length overflow in {}", path.display()))
    })?;
    if expected_len != file_len {
        return Err(Qwen3VlError::Checkpoint(format!(
            "SafeTensors length mismatch in {}: header describes {expected_len} bytes, file has {file_len}",
            path.display()
        )));
    }
    Ok(metadata)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        MropePositionIds, Qwen3VlModelInput, Qwen3VlVisualInput,
        config::tiny_config,
        processor::Grid,
        vision::{Qwen3VlVisionModel, VisionPositionPlan},
    };
    use burn::tensor::{Bool, Int, Tensor, TensorData};
    use burn_ndarray::NdArray;
    use safetensors::tensor::{TensorView, serialize_to_file};
    use std::time::Instant;

    #[test]
    fn strict_tiny_checkpoint_loads_all_tensors_reference() {
        type B = NdArray<f32>;
        let config = tiny_config();
        let inventory = WeightInventory::for_config(&config, true);
        let buffers = inventory
            .specs()
            .iter()
            .map(|spec| {
                let elements = spec.shape.iter().product::<usize>();
                (
                    spec.source.clone(),
                    spec.shape.clone(),
                    vec![0_u8; elements * size_of::<f32>()],
                )
            })
            .collect::<Vec<_>>();
        let views = buffers
            .iter()
            .map(|(name, shape, bytes)| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let directory = tempfile::tempdir().unwrap();
        let shard = directory.path().join("model.safetensors");
        serialize_to_file(views, None, &shard).unwrap();

        let checkpoint = HfCheckpoint::from_shards([shard]).unwrap();
        let (_, report) =
            load_causal_lm_from_safetensors::<B>(config.clone(), &Default::default(), &checkpoint)
                .unwrap();
        assert_eq!(report.shards_loaded, 1);
        assert_eq!(report.tensors_validated, inventory.specs().len());
        assert_eq!(report.tensors_applied, inventory.specs().len());

        let base_inventory = WeightInventory::for_base_model(&config);
        let (_, base_report) = load_base_from_safetensors_with_dtype::<B>(
            config,
            &Default::default(),
            &checkpoint,
            CheckpointDType::F32,
        )
        .unwrap();
        assert_eq!(base_report.tensors_validated, inventory.specs().len());
        assert_eq!(base_report.tensors_applied, base_inventory.specs().len());
    }

    /// Opt-in validation against an unpacked Hugging Face model directory. Keeping the path in
    /// an environment variable makes the test useful for local caches without making CI depend
    /// on a multi-gigabyte checkpoint.
    #[test]
    fn real_checkpoint_strict_reference() {
        let Ok(directory) = std::env::var("QWEN3_VL_CHECKPOINT_DIR") else {
            return;
        };
        let directory = PathBuf::from(directory);
        let config = Qwen3VlConfig::from_json(
            &std::fs::read_to_string(directory.join("config.json")).unwrap(),
        )
        .unwrap();
        let inventory = WeightInventory::for_config(&config, true);
        let checkpoint =
            HfCheckpoint::from_index(directory.join("model.safetensors.index.json")).unwrap();

        let started = Instant::now();
        let inspection = checkpoint.inspect(&inventory).unwrap();
        assert_eq!(inspection.shards.len(), 4);
        assert_eq!(inspection.tensors.len(), 750);
        assert!(
            inspection
                .tensors
                .iter()
                .all(|tensor| tensor.dtype == "BF16")
        );
        eprintln!(
            "strictly validated {} BF16 tensors across {} shards in {:?}",
            inspection.tensors.len(),
            inspection.shards.len(),
            started.elapsed()
        );

        if std::env::var_os("QWEN3_VL_FULL_LOAD").is_none() {
            return;
        }
        type B = NdArray<f32>;
        let started = Instant::now();
        let (_model, report) = load_causal_lm_from_safetensors_with_dtype::<B>(
            config,
            &Default::default(),
            &checkpoint,
            CheckpointDType::F32,
        )
        .unwrap();
        assert_eq!(report.shards_loaded, 4);
        assert_eq!(report.tensors_validated, 750);
        assert_eq!(report.tensors_applied, 750);
        eprintln!(
            "strict full NdArray load completed in {:?}",
            started.elapsed()
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn fixture_values(view: safetensors::tensor::TensorView<'_>) -> Vec<f32> {
        match view.dtype() {
            Dtype::F32 => view
                .data()
                .chunks_exact(size_of::<f32>())
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect(),
            Dtype::BF16 => view
                .data()
                .chunks_exact(size_of::<u16>())
                .map(|bytes| {
                    let bits = u16::from_le_bytes(bytes.try_into().unwrap());
                    f32::from_bits(u32::from(bits) << 16)
                })
                .collect(),
            Dtype::F16 => view
                .data()
                .chunks_exact(size_of::<u16>())
                .map(|bytes| f16_bits_to_f32(u16::from_le_bytes(bytes.try_into().unwrap())))
                .collect(),
            dtype => panic!("unsupported fixture float dtype {dtype:?}"),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn f16_bits_to_f32(bits: u16) -> f32 {
        let sign = (u32::from(bits & 0x8000)) << 16;
        let exponent = (bits >> 10) & 0x1f;
        let mantissa = u32::from(bits & 0x03ff);
        let float_bits = match exponent {
            0 if mantissa == 0 => sign,
            0 => {
                let leading = mantissa.leading_zeros() - 22;
                let normalized = (mantissa << (leading + 1)) & 0x03ff;
                let exponent = 127_u32 - 15 - leading;
                sign | (exponent << 23) | (normalized << 13)
            }
            0x1f => sign | 0x7f80_0000 | (mantissa << 13),
            exponent => sign | ((u32::from(exponent) + 112) << 23) | (mantissa << 13),
        };
        f32::from_bits(float_bits)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn fixture_i64(view: safetensors::tensor::TensorView<'_>) -> Vec<i64> {
        assert_eq!(view.dtype(), Dtype::I64);
        view.data()
            .chunks_exact(size_of::<i64>())
            .map(|bytes| i64::from_le_bytes(bytes.try_into().unwrap()))
            .collect()
    }

    #[derive(Debug, Clone, Copy)]
    struct ParityMetrics {
        max_abs: f64,
        mean_abs: f64,
        rmse: f64,
        relative_rmse: f64,
        cosine: f64,
        actual_rms: f64,
        expected_rms: f64,
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn parity_metrics(actual: &[f32], expected: &[f32]) -> ParityMetrics {
        assert_eq!(actual.len(), expected.len());
        let mut max_abs = 0.0_f64;
        let mut sum_abs = 0.0_f64;
        let mut sum_square = 0.0_f64;
        let mut dot = 0.0_f64;
        let mut actual_square = 0.0_f64;
        let mut expected_square = 0.0_f64;
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert!(actual.is_finite() && expected.is_finite());
            let actual = f64::from(actual);
            let expected = f64::from(expected);
            let difference = actual - expected;
            max_abs = max_abs.max(difference.abs());
            sum_abs += difference.abs();
            sum_square += difference * difference;
            dot += actual * expected;
            actual_square += actual * actual;
            expected_square += expected * expected;
        }
        let count = actual.len() as f64;
        let actual_rms = (actual_square / count).sqrt();
        let expected_rms = (expected_square / count).sqrt();
        let rmse = (sum_square / count).sqrt();
        ParityMetrics {
            max_abs,
            mean_abs: sum_abs / count,
            rmse,
            relative_rmse: rmse / expected_rms,
            cosine: dot / (actual_square.sqrt() * expected_square.sqrt()),
            actual_rms,
            expected_rms,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn report_rank2<B: Backend>(
        name: &str,
        actual: Tensor<B, 2>,
        tensors: &safetensors::SafeTensors<'_>,
    ) -> ParityMetrics {
        let expected_view = tensors.tensor(name).unwrap();
        assert_eq!(actual.dims().as_slice(), expected_view.shape());
        let actual = actual
            .cast(BurnDType::F32)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let expected = fixture_values(expected_view);
        let metrics = parity_metrics(&actual, &expected);
        eprintln!(
            "WGPU stage {name}: max={:.6e}, mean={:.6e}, rmse={:.6e}, rel_rmse={:.4}%, cosine={:.9}",
            metrics.max_abs,
            metrics.mean_abs,
            metrics.rmse,
            100.0 * metrics.relative_rmse,
            metrics.cosine,
        );
        metrics
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn report_rank3<B: Backend>(
        name: &str,
        actual: Tensor<B, 3>,
        tensors: &safetensors::SafeTensors<'_>,
    ) -> ParityMetrics {
        let expected_view = tensors.tensor(name).unwrap();
        assert_eq!(actual.dims().as_slice(), expected_view.shape());
        let actual = actual
            .cast(BurnDType::F32)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let expected = fixture_values(expected_view);
        let metrics = parity_metrics(&actual, &expected);
        eprintln!(
            "WGPU stage {name}: max={:.6e}, mean={:.6e}, rmse={:.6e}, rel_rmse={:.4}%, cosine={:.9}",
            metrics.max_abs,
            metrics.mean_abs,
            metrics.rmse,
            100.0 * metrics.relative_rmse,
            metrics.cosine,
        );
        metrics
    }

    /// Positive stage oracle used by fixtures exported with `--capture-qwen`. This deliberately
    /// reads each boundary back: it is a diagnostic/release test, not the production path.
    #[cfg(not(target_arch = "wasm32"))]
    fn report_vision_stages<B: Backend>(
        visual: &Qwen3VlVisionModel<B>,
        patches: Tensor<B, 2>,
        grids: &[Grid],
        tensors: &safetensors::SafeTensors<'_>,
    ) {
        if tensors.tensor("qwen.vision.patch_embed").is_err() {
            return;
        }
        let config = visual.config();
        let plan = VisionPositionPlan::new(
            grids,
            config.spatial_merge_size,
            config.num_position_embeddings,
        )
        .unwrap();
        let mut hidden = visual.patch_embed.forward(patches);
        report_rank2("qwen.vision.patch_embed", hidden.clone(), tensors);
        let device = hidden.device();
        hidden = hidden + visual.interpolate_position_embeddings(&plan, &device);
        let (cos, sin) = plan
            .vision_cos_sin::<B>(config.head_dim(), &device)
            .unwrap();
        for (index, block) in visual.blocks.iter().enumerate() {
            hidden = block.forward(hidden, &plan.frame_ranges, cos.clone(), sin.clone());
            report_rank2(
                &format!("qwen.vision.block.{index}"),
                hidden.clone(),
                tensors,
            );
            if let Some(merger_index) = config
                .deepstack_visual_indexes
                .iter()
                .position(|&after| after == index)
            {
                report_rank2(
                    &format!("qwen.vision.deepstack_merger.{merger_index}"),
                    visual.deepstack_merger_list[merger_index].forward(hidden.clone()),
                    tensors,
                );
            }
        }
        report_rank2(
            "qwen.vision.final_merger",
            visual.merger.forward(hidden),
            tensors,
        );
    }

    /// Opt-in positive parity run using real model weights and Transformers-exported processor
    /// inputs/hidden states. Multiple fixture directories are separated with `:`.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn real_wgpu_forward_reference() {
        let Ok(directory) = std::env::var("QWEN3_VL_CHECKPOINT_DIR") else {
            return;
        };
        let Ok(fixture_directories) = std::env::var("QWEN3_VL_WGPU_FORWARD_FIXTURES") else {
            return;
        };
        type B = burn_wgpu::Wgpu<f32, i32, u32>;
        let device = burn_wgpu::WgpuDevice::default();
        let directory = PathBuf::from(directory);
        let config = Qwen3VlConfig::from_json(
            &std::fs::read_to_string(directory.join("config.json")).unwrap(),
        )
        .unwrap();
        let checkpoint =
            HfCheckpoint::from_index(directory.join("model.safetensors.index.json")).unwrap();

        let checkpoint_dtype = match std::env::var("QWEN3_VL_CHECKPOINT_DTYPE")
            .as_deref()
            .unwrap_or("f32")
        {
            "source" => CheckpointDType::Source,
            "f32" => CheckpointDType::F32,
            "f16" => CheckpointDType::F16,
            value => panic!("unknown QWEN3_VL_CHECKPOINT_DTYPE {value:?}"),
        };
        let load_started = Instant::now();
        let (mut model, report) = load_causal_lm_from_safetensors_with_dtype::<B>(
            config.clone(),
            &device,
            &checkpoint,
            checkpoint_dtype,
        )
        .unwrap();
        B::sync(&device).unwrap();
        eprintln!(
            "WGPU load ({checkpoint_dtype:?}): {:?}, shards={}, tensors={}",
            load_started.elapsed(),
            report.shards_loaded,
            report.tensors_applied
        );
        assert_eq!(report.tensors_applied, 750);
        model.set_query_chunk_size(64);

        let embedding_probe = model
            .model
            .language_model
            .embed_tokens
            .weight
            .val()
            .slice([0..1, 0..16])
            .cast(burn::tensor::DType::F32)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let final_norm_probe = model
            .model
            .language_model
            .norm
            .gamma
            .val()
            .slice_dim(0, 0..16)
            .cast(burn::tensor::DType::F32)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        eprintln!(
            "WGPU loaded probes: embed[0,0..16]={embedding_probe:?}, final_norm[0..16]={final_norm_probe:?}"
        );
        let embedding_forward_probe = model
            .model
            .language_model
            .embed(Tensor::<B, 2, Int>::from_data([[0_i64, 151_644]], &device))
            .slice([0..1, 0..2, 0..16])
            .cast(burn::tensor::DType::F32)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        eprintln!("WGPU embedding forward probe={embedding_forward_probe:?}");

        for fixture_directory in fixture_directories.split(':') {
            let fixture_directory = PathBuf::from(fixture_directory);
            let bytes = std::fs::read(fixture_directory.join("tensors.safetensors")).unwrap();
            let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
            let input_view = tensors.tensor("processor.input_ids").unwrap();
            let input_shape = input_view.shape();
            assert_eq!(input_shape.len(), 2);
            let [batch, sequence] = [input_shape[0], input_shape[1]];
            assert_eq!(batch, 1, "real parity fixtures currently cover batch one");
            let input_ids = fixture_i64(input_view);
            let mask_values = fixture_i64(tensors.tensor("processor.attention_mask").unwrap())
                .into_iter()
                .map(|value| value != 0)
                .collect::<Vec<_>>();
            let token_types = fixture_i64(tensors.tensor("processor.mm_token_type_ids").unwrap())
                .into_iter()
                .map(|value| u8::try_from(value).unwrap())
                .collect::<Vec<_>>();

            let image_grids = match tensors.tensor("processor.image_grid_thw") {
                Ok(view) => fixture_i64(view)
                    .chunks_exact(3)
                    .map(|grid| {
                        Grid::new(
                            usize::try_from(grid[0]).unwrap(),
                            usize::try_from(grid[1]).unwrap(),
                            usize::try_from(grid[2]).unwrap(),
                        )
                    })
                    .collect::<Vec<_>>(),
                Err(_) => Vec::new(),
            };
            let position_ids = MropePositionIds::from_batch(
                std::slice::from_ref(&token_types),
                std::slice::from_ref(&mask_values),
                std::slice::from_ref(&image_grids),
                &[vec![]],
                config.vision_config.spatial_merge_size,
            )
            .unwrap();
            let image_indices = token_types
                .iter()
                .enumerate()
                .filter_map(|(index, &kind)| (kind == 1).then_some(index))
                .collect::<Vec<_>>();
            let images = tensors.tensor("processor.pixel_values").ok().map(|view| {
                let shape = view.shape().to_vec();
                assert_eq!(shape.len(), 2);
                Qwen3VlVisualInput {
                    patches: Tensor::<B, 2>::from_data(
                        TensorData::new(fixture_values(view), [shape[0], shape[1]]),
                        &device,
                    ),
                    grids: image_grids,
                    token_indices: image_indices,
                }
            });
            if let Some(images) = &images {
                report_vision_stages(
                    &model.model.visual,
                    images.patches.clone(),
                    &images.grids,
                    &tensors,
                );
            }
            let has_images = images.is_some();
            let input_ids = Tensor::<B, 2, Int>::from_data(
                TensorData::new(input_ids, [batch, sequence]),
                &device,
            );
            if tensors.tensor("qwen.text.token_embeddings").is_ok() {
                report_rank3(
                    "qwen.text.token_embeddings",
                    model.model.language_model.embed(input_ids.clone()),
                    &tensors,
                );
            }
            if tensors.tensor("qwen.text.rope.0").is_ok() {
                let (cos, sin) = position_ids
                    .cos_sin::<B>(&config.text_config, &device)
                    .unwrap();
                report_rank3("qwen.text.rope.0", cos, &tensors);
                report_rank3("qwen.text.rope.1", sin, &tensors);
            }
            let input = Qwen3VlModelInput {
                input_ids,
                attention_mask: Some(Tensor::<B, 2, Bool>::from_data(
                    TensorData::new(mask_values, [batch, sequence]),
                    &device,
                )),
                position_ids: Some(position_ids),
                images,
                videos: None,
                output_hidden_states: true,
            };

            let forward_started = Instant::now();
            let output = model.model.forward(input).unwrap();
            B::sync(&device).unwrap();
            let forward_elapsed = forward_started.elapsed();
            let layer_probes = output
                .hidden_states
                .as_ref()
                .unwrap()
                .iter()
                .map(|hidden| {
                    hidden
                        .clone()
                        .slice([0..1, 0..1, 0..16])
                        .cast(burn::tensor::DType::F32)
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap()
                })
                .collect::<Vec<_>>();
            eprintln!(
                "WGPU layer first-token probe RMS={:?}",
                layer_probes
                    .iter()
                    .map(|values| {
                        (values
                            .iter()
                            .map(|value| f64::from(*value).powi(2))
                            .sum::<f64>()
                            / values.len() as f64)
                            .sqrt()
                    })
                    .collect::<Vec<_>>()
            );
            if tensors.tensor("qwen.text.token_embeddings").is_ok() {
                let hidden_states = output.hidden_states.as_ref().unwrap();
                if !has_images {
                    report_rank3(
                        "qwen.text.token_embeddings",
                        hidden_states[0].clone(),
                        &tensors,
                    );
                    for index in 0..config.text_config.num_hidden_layers - 1 {
                        report_rank3(
                            &format!("qwen.text.layer.{index}"),
                            hidden_states[index + 1].clone(),
                            &tensors,
                        );
                    }
                } else {
                    // Module hooks capture the decoder output before DeepStack is added. From
                    // layer three onward there is no addition, so the next layer input is the
                    // exact preceding module output.
                    for index in config.vision_config.deepstack_visual_indexes.len()
                        ..config.text_config.num_hidden_layers - 1
                    {
                        report_rank3(
                            &format!("qwen.text.layer.{index}"),
                            hidden_states[index + 1].clone(),
                            &tensors,
                        );
                    }
                }
                report_rank3(
                    "qwen.text.final_norm",
                    hidden_states.last().unwrap().clone(),
                    &tensors,
                );
            }
            let actual = output
                .last_hidden_state
                .cast(burn::tensor::DType::F32)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            let expected = fixture_values(tensors.tensor("qwen.last_hidden_state").unwrap());
            assert_eq!(actual.len(), expected.len());
            let metrics = parity_metrics(&actual, &expected);
            eprintln!(
                "WGPU parity {}: shape=[{batch},{sequence},{}], forward={:?}, actual_rms={:e}, expected_rms={:e}, max_abs={:e}, mean_abs={:e}, rmse={:e}, rel_rmse={:.4}%, cosine={:.9}",
                fixture_directory.display(),
                config.text_config.hidden_size,
                forward_elapsed,
                metrics.actual_rms,
                metrics.expected_rms,
                metrics.max_abs,
                metrics.mean_abs,
                metrics.rmse,
                100.0 * metrics.relative_rmse,
                metrics.cosine,
            );
            // Evidence-based BF16->F16 envelopes come from the same pinned Transformers model
            // rerun with F16 weights. Same-precision captures use a substantially tighter gate.
            let expected_dtype = tensors.tensor("qwen.last_hidden_state").unwrap().dtype();
            match (checkpoint_dtype, expected_dtype, has_images) {
                (CheckpointDType::F16, Dtype::BF16, false) => assert!(
                    metrics.cosine >= 0.9997
                        && metrics.relative_rmse <= 0.025
                        && metrics.max_abs <= 10.0
                ),
                (CheckpointDType::F16, Dtype::BF16, true) => assert!(
                    metrics.cosine >= 0.995
                        && metrics.relative_rmse <= 0.11
                        && metrics.max_abs <= 24.0
                ),
                (CheckpointDType::F16, Dtype::F16, _) => assert!(
                    metrics.cosine >= 0.9999
                        && metrics.relative_rmse <= 0.02
                        && metrics.max_abs <= 5.0
                ),
                _ => assert!(metrics.cosine >= 0.999 && metrics.relative_rmse <= 0.05),
            }
        }
    }
}
