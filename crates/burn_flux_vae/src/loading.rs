use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
    rc::Rc,
};

use burn::{
    prelude::Backend,
    tensor::{Bytes, DType},
};
use burn_store::{
    ApplyResult, BurnpackStore, KeyRemapper, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter,
    SafetensorsStore, TensorSnapshot,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{AutoencoderKl, AutoencoderKlConfig, TensorInventory};

/// Validation controls for loading trusted local artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadOptions {
    /// Permit a store to contain only a subset of the module.
    pub allow_partial: bool,
    /// Ask BurnStore to validate checksums and structure where the format supports it.
    pub validate: bool,
    /// Convert F16/BF16/F64 source tensors to F32 before backend allocation.
    ///
    /// This is enabled by default because FLUX.1 sets `force_upcast=true`, and it also permits
    /// loading the public BF16 checkpoint on backends such as NdArray that do not implement BF16.
    pub force_f32: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            allow_partial: false,
            validate: true,
            force_f32: true,
        }
    }
}

/// Auditable result of applying one or more weight stores.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoadReport {
    pub applied: Vec<String>,
    pub missing: Vec<String>,
    pub skipped: Vec<String>,
    pub errors: Vec<String>,
}

impl LoadReport {
    pub fn is_complete(&self) -> bool {
        !self.applied.is_empty()
            && self.missing.is_empty()
            && self.skipped.is_empty()
            && self.errors.is_empty()
    }

    pub fn ensure_complete(&self, label: &str) -> Result<(), LoadError> {
        if self.is_complete() {
            Ok(())
        } else {
            Err(LoadError::Incomplete {
                label: label.to_string(),
                applied: self.applied.len(),
                missing: self.missing.clone(),
                skipped: self.skipped.clone(),
                errors: self.errors.clone(),
            })
        }
    }

    fn append(&mut self, mut other: Self) {
        self.applied.append(&mut other.applied);
        self.missing.append(&mut other.missing);
        self.skipped.append(&mut other.skipped);
        self.errors.append(&mut other.errors);
    }
}

#[derive(Debug, Error)]
pub enum LoadError {
    #[error("invalid AutoencoderKL config: {0}")]
    Config(#[from] crate::AutoencoderKlConfigError),
    #[error("failed to load {path}: {message}")]
    Store { path: PathBuf, message: String },
    #[error(
        "incomplete {label}: applied {applied}, missing {missing:?}, skipped {skipped:?}, errors {errors:?}"
    )]
    Incomplete {
        label: String,
        applied: usize,
        missing: Vec<String>,
        skipped: Vec<String>,
        errors: Vec<String>,
    },
    #[error("burnpack shard loader is poisoned after an earlier failed apply")]
    Poisoned,
    #[error("burnpack shard applied duplicate tensor '{0}'")]
    DuplicateTensor(String),
    #[error("burnpack shard applied tensor '{0}' outside the configured inventory")]
    UnexpectedTensor(String),
}

/// Load a complete upstream Diffusers SafeTensors file with strict defaults.
pub fn load_safetensors_file<B: Backend>(
    device: &B::Device,
    path: impl AsRef<Path>,
    config: &AutoencoderKlConfig,
) -> Result<(AutoencoderKl<B>, LoadReport), LoadError> {
    load_safetensors_file_with_options(device, path, config, LoadOptions::default())
}

pub fn load_safetensors_file_with_options<B: Backend>(
    device: &B::Device,
    path: impl AsRef<Path>,
    config: &AutoencoderKlConfig,
    options: LoadOptions,
) -> Result<(AutoencoderKl<B>, LoadReport), LoadError> {
    let path = path.as_ref();
    let mut model = config.try_init(device)?;
    let mut remapper = KeyRemapper::new();
    for &(from, to) in diffusers_key_remap_rules() {
        remapper = remapper
            .add_pattern(from, to)
            .map_err(|error| LoadError::Store {
                path: path.to_path_buf(),
                message: format!("invalid key remap {from}->{to}: {error}"),
            })?;
    }
    let mut store = SafetensorsStore::from_file(path)
        .with_from_adapter(PyTorchToBurnAdapter.chain(Float32Adapter {
            enabled: options.force_f32,
        }))
        .allow_partial(options.allow_partial)
        .remap(remapper)
        .validate(options.validate);
    let result = model
        .load_from(&mut store)
        .map_err(|error| LoadError::Store {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
    let report = report_from_apply(result);
    if !options.allow_partial {
        report.ensure_complete("AutoencoderKL safetensors")?;
    }
    Ok((model, report))
}

/// Load a complete Burnpack file with strict defaults.
pub fn load_burnpack_file<B: Backend>(
    device: &B::Device,
    path: impl AsRef<Path>,
    config: &AutoencoderKlConfig,
) -> Result<(AutoencoderKl<B>, LoadReport), LoadError> {
    load_burnpack_file_with_options(device, path, config, LoadOptions::default())
}

pub fn load_burnpack_file_with_options<B: Backend>(
    device: &B::Device,
    path: impl AsRef<Path>,
    config: &AutoencoderKlConfig,
    options: LoadOptions,
) -> Result<(AutoencoderKl<B>, LoadReport), LoadError> {
    let path = path.as_ref();
    let mut model = config.try_init(device)?;
    let mut store = BurnpackStore::from_file(path)
        .with_from_adapter(Float32Adapter {
            enabled: options.force_f32,
        })
        .allow_partial(options.allow_partial)
        .validate(options.validate);
    let result = model
        .load_from(&mut store)
        .map_err(|error| LoadError::Store {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
    let report = report_from_apply(result);
    if !options.allow_partial {
        report.ensure_complete("AutoencoderKL burnpack")?;
    }
    Ok((model, report))
}

/// Save the exact Burn module layout into a Burnpack file.
pub fn save_burnpack_file<B: Backend>(
    model: &AutoencoderKl<B>,
    path: impl AsRef<Path>,
    overwrite: bool,
) -> Result<(), LoadError> {
    let path = path.as_ref();
    let mut store = BurnpackStore::from_file(path).overwrite(overwrite);
    model
        .save_into(&mut store)
        .map_err(|error| LoadError::Store {
            path: path.to_path_buf(),
            message: error.to_string(),
        })
}

/// Apply one partial Burnpack payload to an existing model.
///
/// Callers that need duplicate/unknown/completeness enforcement across multiple shards should use
/// [`BurnpackShardLoader`].
pub fn apply_burnpack_part_bytes<B: Backend>(
    model: &mut AutoencoderKl<B>,
    bytes: Vec<u8>,
    validate: bool,
) -> Result<LoadReport, LoadError> {
    let mut store = BurnpackStore::from_bytes(Some(Bytes::from_bytes_vec(bytes)))
        .with_from_adapter(Float32Adapter { enabled: true })
        .allow_partial(true)
        .validate(validate);
    let result = model
        .load_from(&mut store)
        .map_err(|error| LoadError::Store {
            path: PathBuf::from("<burnpack-bytes>"),
            message: error.to_string(),
        })?;
    let report = report_from_apply(result);
    if report.applied.is_empty() || !report.skipped.is_empty() || !report.errors.is_empty() {
        return Err(LoadError::Incomplete {
            label: "AutoencoderKL burnpack part".to_string(),
            applied: report.applied.len(),
            missing: Vec::new(),
            skipped: report.skipped.clone(),
            errors: report.errors.clone(),
        });
    }
    Ok(report)
}

/// Stateful completeness and duplicate guard for sequential Burnpack shard application.
pub struct BurnpackShardLoader<B: Backend> {
    model: AutoencoderKl<B>,
    expected: BTreeSet<String>,
    applied: BTreeSet<String>,
    report: LoadReport,
    validate: bool,
    poisoned: bool,
}

impl<B: Backend> BurnpackShardLoader<B> {
    pub fn new(
        device: &B::Device,
        config: &AutoencoderKlConfig,
        validate: bool,
    ) -> Result<Self, LoadError> {
        let inventory = TensorInventory::from_config(config)?;
        Ok(Self {
            model: config.try_init(device)?,
            expected: inventory.burn_names().map(str::to_string).collect(),
            applied: BTreeSet::new(),
            report: LoadReport::default(),
            validate,
            poisoned: false,
        })
    }

    pub fn apply(&mut self, bytes: Vec<u8>) -> Result<LoadReport, LoadError> {
        if self.poisoned {
            return Err(LoadError::Poisoned);
        }
        let report = match apply_burnpack_part_bytes(&mut self.model, bytes, self.validate) {
            Ok(report) => report,
            Err(error) => {
                self.poisoned = true;
                return Err(error);
            }
        };
        for name in &report.applied {
            if !self.expected.contains(name) {
                self.poisoned = true;
                return Err(LoadError::UnexpectedTensor(name.clone()));
            }
            if !self.applied.insert(name.clone()) {
                self.poisoned = true;
                return Err(LoadError::DuplicateTensor(name.clone()));
            }
        }
        self.report.append(LoadReport {
            applied: report.applied.clone(),
            missing: Vec::new(),
            skipped: report.skipped.clone(),
            errors: report.errors.clone(),
        });
        Ok(report)
    }

    pub fn loaded_tensor_count(&self) -> usize {
        self.applied.len()
    }

    pub fn expected_tensor_count(&self) -> usize {
        self.expected.len()
    }

    pub fn finish(mut self) -> Result<(AutoencoderKl<B>, LoadReport), LoadError> {
        if self.poisoned {
            return Err(LoadError::Poisoned);
        }
        self.report.missing = self.expected.difference(&self.applied).cloned().collect();
        self.report
            .ensure_complete("AutoencoderKL burnpack shards")?;
        Ok((self.model, self.report))
    }
}

/// Key rewrites from upstream Diffusers state dictionaries into Burn module parameter names.
pub fn diffusers_key_remap_rules() -> &'static [(&'static str, &'static str)] {
    &[
        (r"^(.*\.to_out)\.0\.(weight|bias)$", "$1.$2"),
        (r"^(.+\.norm[12])\.weight$", "$1.gamma"),
        (r"^(.+\.norm[12])\.bias$", "$1.beta"),
        (r"^(.+\.group_norm)\.weight$", "$1.gamma"),
        (r"^(.+\.group_norm)\.bias$", "$1.beta"),
        (r"^(.+\.conv_norm_out)\.weight$", "$1.gamma"),
        (r"^(.+\.conv_norm_out)\.bias$", "$1.beta"),
    ]
}

fn report_from_apply(result: ApplyResult) -> LoadReport {
    LoadReport {
        applied: result.applied,
        missing: result
            .missing
            .into_iter()
            .map(|(name, reason)| format!("{name}: {reason}"))
            .collect(),
        skipped: result.skipped,
        errors: result
            .errors
            .into_iter()
            .map(|error| format!("{error:?}"))
            .collect(),
    }
}

#[derive(Debug, Clone, Copy)]
struct Float32Adapter {
    enabled: bool,
}

impl ModuleAdapter for Float32Adapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        if !self.enabled || snapshot.dtype == DType::F32 {
            return snapshot.clone();
        }
        if !matches!(snapshot.dtype, DType::F16 | DType::BF16 | DType::F64) {
            return snapshot.clone();
        }
        let data_fn = snapshot.clone_data_fn();
        TensorSnapshot::from_closure(
            Rc::new(move || Ok(data_fn()?.convert_dtype(DType::F32))),
            DType::F32,
            snapshot.shape.clone(),
            snapshot.path_stack.clone().unwrap_or_default(),
            snapshot.container_stack.clone().unwrap_or_default(),
            snapshot.tensor_id.unwrap_or_default(),
        )
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(*self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestBackend = burn::backend::NdArray<f32>;

    fn remap(key: &str) -> String {
        let mut remapper = KeyRemapper::new();
        for &(from, to) in diffusers_key_remap_rules() {
            remapper = remapper.add_pattern(from, to).expect("valid rule");
        }
        let mut output = key.to_string();
        for (pattern, replacement) in &remapper.patterns {
            if pattern.is_match(&output) {
                output = pattern
                    .replace_all(&output, replacement.as_str())
                    .to_string();
            }
        }
        output
    }

    #[test]
    fn diffusers_key_remap_correctness() {
        assert_eq!(
            remap("encoder.down_blocks.1.resnets.0.norm1.weight"),
            "encoder.down_blocks.1.resnets.0.norm1.gamma"
        );
        assert_eq!(
            remap("decoder.mid_block.attentions.0.to_out.0.bias"),
            "decoder.mid_block.attentions.0.to_out.bias"
        );
        assert_eq!(
            remap("decoder.conv_norm_out.weight"),
            "decoder.conv_norm_out.gamma"
        );
    }

    #[test]
    fn burnpack_round_trip_smoke() {
        let device = Default::default();
        let config = AutoencoderKlConfig::tiny();
        let model = config.init::<TestBackend>(&device);
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("tiny.bpk");
        save_burnpack_file(&model, &path, true).expect("save burnpack");
        let (_loaded, report) =
            load_burnpack_file::<TestBackend>(&device, &path, &config).expect("load burnpack");
        assert!(report.is_complete(), "{report:?}");
    }

    #[test]
    fn real_diffusers_safetensors_reference() {
        let Ok(path) = std::env::var("BURN_FLUX_VAE_SAFETENSORS") else {
            eprintln!("SKIP real_diffusers_safetensors_reference: set BURN_FLUX_VAE_SAFETENSORS");
            return;
        };
        let device = Default::default();
        let config = AutoencoderKlConfig::flux1();
        let (model, report) =
            load_safetensors_file::<TestBackend>(&device, path, &config).expect("load FLUX VAE");
        assert_eq!(report.applied.len(), 244, "{report:?}");
        assert!(report.is_complete(), "{report:?}");
        let moments = model.encode_moments(burn::tensor::Tensor::zeros([1, 3, 8, 8], &device));
        assert_eq!(moments.dims(), [1, 32, 1, 1]);
        let decoded = model.decode(crate::DiagonalGaussian::from_moments(moments).mode());
        assert_eq!(decoded.dims(), [1, 3, 8, 8]);
        assert!(decoded.is_finite().all().into_scalar());
    }
}
