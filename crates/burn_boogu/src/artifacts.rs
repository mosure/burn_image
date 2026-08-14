//! Immutable release identities, exact checkpoint inventories, and staged Burnpack loading.

use std::collections::BTreeSet;

use burn_flux_vae::{AutoencoderKlConfig, TensorInventory as FluxTensorInventory};
use burn_qwen3_vl::{
    Qwen3VlConfig, Qwen3VlStage, RowChunkSpec, WeightInventory as QwenWeightInventory, WeightRole,
};
use serde::{Deserialize, Serialize};

use crate::{BooguConfig, BooguError, BooguTask, BooguVariant};

/// Immutable upstream source revision used by the initial artifact release.
pub const UPSTREAM_SOURCE_REVISION: &str = "25f8f888298224a94e5ec2abafb98abea9031a0d";
/// Immutable Turbo Hugging Face revision.
pub const TURBO_REVISION: &str = "53ad54522023f64d049f7f38e4d679359ef3fb92";
/// Immutable Edit-Turbo Hugging Face revision.
pub const EDIT_TURBO_REVISION: &str = "132a0ab9051b42c1d9be4919a68873d1f132c0c8";
/// Immutable Edit-Turbo 1.5K Hugging Face revision.
pub const EDIT_TURBO_1K5_REVISION: &str = "60981c49e48cffadf2c169532a4ba3f6108afd5e";
/// Converter version that produced the immutable payloads reused by the canonical bundles.
///
/// Artifact compatibility follows the sealed converter contract, not this runtime crate's version.
pub const PUBLISHED_BUNDLE_CONVERTER_VERSION: &str = "0.1.0";
/// Converter version emitted by the current importer for explicit local or custom bundles.
pub const CURRENT_BUNDLE_CONVERTER_VERSION: &str = env!("CARGO_PKG_VERSION");
/// Converter versions whose layout/schema contract this runtime can verify.
pub const SUPPORTED_BUNDLE_CONVERTER_VERSIONS: &[&str] = &[
    PUBLISHED_BUNDLE_CONVERTER_VERSION,
    CURRENT_BUNDLE_CONVERTER_VERSION,
];

/// Require a converter whose sealed layout/schema contract this runtime understands.
pub fn validate_supported_bundle_converter_version(actual: &str) -> Result<(), BooguError> {
    if SUPPORTED_BUNDLE_CONVERTER_VERSIONS.contains(&actual) {
        Ok(())
    } else {
        Err(BooguError::Artifact(format!(
            "unsupported release converter {actual:?}; expected one of {SUPPORTED_BUNDLE_CONVERTER_VERSIONS:?}"
        )))
    }
}
/// Exact sealed mixed-F16 Turbo bundle qualified for the canonical production release.
pub const TURBO_F16_QWEN_VISION_F32_CONTENT_DIGEST: &str =
    "555019af867a80bb4d7cec5dc2f0ba60ae799071994a5fd24d7e71918cb9ce36";
/// Exact sealed hybrid-Q8 Turbo legacy bundle retained as evidence.
pub const TURBO_Q8S_BLOCK32_F32_QWEN_VISION_F32_CONTENT_DIGEST: &str =
    "8685559e73cf836e98e1ebdf80815e3d66765f7d620624408148d5f98c87c0dd";
/// Exact sealed mixed-F16 Edit-Turbo 1K bundle qualified for the canonical production release.
pub const EDIT_TURBO_F16_QWEN_VISION_F32_CONTENT_DIGEST: &str =
    "28b1b51f2fb152557b11a9f0ef8e872ae7d163bcab7abd42f9eaf4bfef10e7aa";
/// Exact sealed hybrid-Q8 Edit-Turbo 1K legacy bundle retained as evidence.
pub const EDIT_TURBO_Q8S_BLOCK32_F32_QWEN_VISION_F32_CONTENT_DIGEST: &str =
    "ffde989bb66df3a541d44957422f996790633dab46ca3547a59dfdfb871f0b7a";
/// Exact sealed mixed-F16 Edit-Turbo 1.5K bundle qualified for the canonical production release.
pub const EDIT_TURBO_1K5_F16_QWEN_VISION_F32_CONTENT_DIGEST: &str =
    "4eb95001708becebeab5bb7417b02003e9dbe704775bb49557b681a5b617fd5a";
/// Storage-policy marker for the opt-in 1.5K VAE encoder F32 A/B artifact.
///
/// This is deliberately metadata rather than a production storage profile: callers must
/// select the ordinary mixed-F16 profile and an explicit custom artifact URL, then separately
/// authenticate this diagnostic overlay contract.
pub const EDIT_TURBO_1K5_VAE_ENCODER_F32_DIAGNOSTIC_PROFILE: &str =
    "f16-qwen-vision-f32+vae-encoder-f32";
/// Exact number of ordinary FLUX VAE encoder tensors replaced by the diagnostic F32 overlay.
pub const EDIT_TURBO_1K5_VAE_ENCODER_F32_DIAGNOSTIC_TENSORS: usize = 106;
/// SHA-256 of the pinned upstream F32 FLUX VAE SafeTensors object used by the diagnostic.
pub const EDIT_TURBO_1K5_VAE_SOURCE_SHA256: &str =
    "8c717328c8ad41faab2ccfd52ae17332505c6833cf176aad56e7b58f2c4d4c94";
/// Exact byte length of the pinned upstream F32 FLUX VAE SafeTensors object.
pub const EDIT_TURBO_1K5_VAE_SOURCE_BYTES: u64 = 335_306_212;
/// Legacy descriptive mixed-F16 Turbo digest accepted only for explicit/local migration.
pub const LEGACY_TURBO_F16_QWEN_VISION_F32_CONTENT_DIGEST: &str =
    "4f94cf68c00af12d5de486db4d316ce889d6d21e78913a1c74edab4bd0119ce3";
/// Legacy descriptive mixed-F16 Edit-Turbo 1K digest accepted only for explicit/local migration.
pub const LEGACY_EDIT_TURBO_F16_QWEN_VISION_F32_CONTENT_DIGEST: &str =
    "14acbafd13dc9b79757e7d554b504396bee30ea7ed231f533919c6c82a6e6a32";
/// Legacy descriptive mixed-F16 Edit-Turbo 1.5K digest accepted only for explicit/local migration.
pub const LEGACY_EDIT_TURBO_1K5_F16_QWEN_VISION_F32_CONTENT_DIGEST: &str =
    "4e8b12ac5ca95272f9009080a23baf1bc52d1b0e7aebf2e9e5f394a492369213";
/// Largest physical Burnpack object accepted for browser-deployable releases.
pub const BOOGU_RELEASE_MAX_SHARD_BYTES: u64 = 256 * 1024 * 1024;
/// Largest compact config, tokenizer, template, or inventory object accepted in a release.
pub const BOOGU_RELEASE_MAX_METADATA_BYTES: u64 = 64 * 1024 * 1024;

/// Algorithm encoded into a Boogu release manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Algorithm {
    /// Four-step distribution-matching-distillation path.
    DmdTurbo,
}

/// Identity portion of a Boogu artifact release.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BooguReleaseIdentity {
    /// Schema version.
    pub schema_version: u32,
    /// Model variant.
    pub variant: BooguVariant,
    /// Supported task.
    pub task: BooguTask,
    /// Required algorithm.
    pub algorithm: Algorithm,
    /// Pinned upstream source revision.
    pub upstream_source_revision: String,
    /// Pinned Hugging Face revision.
    pub model_revision: String,
}

impl BooguReleaseIdentity {
    /// Canonical identity for a released variant.
    pub fn canonical(variant: BooguVariant) -> Self {
        let (task, revision) = match variant {
            BooguVariant::Image01Turbo => (BooguTask::Generate, TURBO_REVISION),
            BooguVariant::Image01EditTurbo => (BooguTask::Edit, EDIT_TURBO_REVISION),
            BooguVariant::Image01EditTurbo1k5 => (BooguTask::Edit, EDIT_TURBO_1K5_REVISION),
        };
        Self {
            schema_version: 1,
            variant,
            task,
            algorithm: Algorithm::DmdTurbo,
            upstream_source_revision: UPSTREAM_SOURCE_REVISION.into(),
            model_revision: revision.into(),
        }
    }

    /// Ensure mutable or inconsistent metadata cannot select the wrong pipeline.
    pub fn validate(&self) -> Result<(), BooguError> {
        let expected = Self::canonical(self.variant);
        if self != &expected {
            return Err(BooguError::Artifact(format!(
                "release identity differs from canonical {expected:?}"
            )));
        }
        Ok(())
    }
}

/// Model that owns a checkpoint tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TensorOwner {
    /// Ordinary Qwen3-VL conditional-generation model.
    Qwen3Vl,
    /// Boogu-specific diffusion transformer.
    BooguDenoiser,
    /// Ordinary FLUX-compatible AutoencoderKL.
    FluxVae,
}

/// Element type required in the pinned upstream BF16 releases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum SourceDType {
    /// BFloat16 storage.
    Bf16,
    /// IEEE single precision storage.
    F32,
}

impl SourceDType {
    /// SafeTensors spelling of this element type.
    pub const fn safetensors_name(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::F32 => "F32",
        }
    }
}

/// Storage layout conversion performed before writing Burnpack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TensorTransform {
    /// Preserve source element order and shape.
    Identity,
    /// Transpose an upstream `[out, in]` matrix to Burn's row-layout `[in, out]` matrix.
    Transpose2d,
}

/// One exact source-to-Burn tensor contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactTensorSpec {
    /// Component directory in the Hugging Face snapshot.
    pub source_component: String,
    /// SafeTensors key.
    pub source_name: String,
    /// Burn module record path.
    pub target_name: String,
    /// Semantic shard stage. A shard may contain tensors from only one stage.
    pub stage: String,
    /// Owning Burn model.
    pub owner: TensorOwner,
    /// Required source element type.
    pub source_dtype: SourceDType,
    /// Exact source shape.
    pub source_shape: Vec<usize>,
    /// Exact stored/Burn shape.
    pub target_shape: Vec<usize>,
    /// Required layout conversion.
    pub transform: TensorTransform,
    /// Whether the released Q8S profile may keep this tensor quantized on device.
    pub quantizable: bool,
}

impl ArtifactTensorSpec {
    /// Qualified source key used to disambiguate the three upstream state dictionaries.
    pub fn qualified_source_name(&self) -> String {
        format!("{}:{}", self.source_component, self.source_name)
    }
}

/// Complete deterministic inventory for all three models in a Boogu pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BooguArtifactInventory {
    tensors: Vec<ArtifactTensorSpec>,
}

impl BooguArtifactInventory {
    /// Build the complete source-to-record inventory from the three validated model configs.
    pub fn new(
        qwen: &Qwen3VlConfig,
        denoiser: &BooguConfig,
        vae: &AutoencoderKlConfig,
    ) -> Result<Self, BooguError> {
        qwen.validate()
            .map_err(|error| BooguError::Artifact(error.to_string()))?;
        denoiser.validate()?;
        vae.validate()
            .map_err(|error| BooguError::Artifact(error.to_string()))?;

        let mut tensors = qwen_specs(qwen);
        tensors.extend(boogu_specs(denoiser));
        tensors.extend(flux_vae_specs(vae)?);
        tensors.sort_by(|left, right| {
            (&left.source_component, &left.source_name)
                .cmp(&(&right.source_component, &right.source_name))
        });
        let inventory = Self { tensors };
        inventory.validate_internal()?;
        Ok(inventory)
    }

    /// Build the exact standalone Boogu denoiser inventory from its validated config.
    ///
    /// This is useful to audit denoiser-only residency plans without constructing unrelated Qwen
    /// or VAE configurations. The returned contracts are identical to the denoiser subset of
    /// [`Self::new`].
    pub fn denoiser(denoiser: &BooguConfig) -> Result<Self, BooguError> {
        denoiser.validate()?;
        let mut tensors = boogu_specs(denoiser);
        tensors.sort_by(|left, right| {
            (&left.source_component, &left.source_name)
                .cmp(&(&right.source_component, &right.source_name))
        });
        let inventory = Self { tensors };
        inventory.validate_internal()?;
        Ok(inventory)
    }

    /// Exact tensor contracts in deterministic source-key order.
    pub fn tensors(&self) -> &[ArtifactTensorSpec] {
        &self.tensors
    }

    /// Find a tensor by component-qualified upstream name.
    pub fn by_source(&self, component: &str, name: &str) -> Option<&ArtifactTensorSpec> {
        self.tensors
            .iter()
            .find(|spec| spec.source_component == component && spec.source_name == name)
    }

    /// Find a tensor by its globally unique Burn record path.
    pub fn by_target(&self, name: &str) -> Option<&ArtifactTensorSpec> {
        self.tensors.iter().find(|spec| spec.target_name == name)
    }

    /// Required tensor count for one upstream component.
    pub fn component_len(&self, component: &str) -> usize {
        self.tensors
            .iter()
            .filter(|spec| spec.source_component == component)
            .count()
    }

    /// All required semantic stages.
    pub fn stages(&self) -> BTreeSet<&str> {
        self.tensors
            .iter()
            .map(|spec| spec.stage.as_str())
            .collect()
    }

    fn validate_internal(&self) -> Result<(), BooguError> {
        let mut sources = BTreeSet::new();
        let mut targets = BTreeSet::new();
        for spec in &self.tensors {
            if !sources.insert(spec.qualified_source_name()) {
                return Err(BooguError::Artifact(format!(
                    "duplicate source tensor {}",
                    spec.qualified_source_name()
                )));
            }
            if !targets.insert(spec.target_name.clone()) {
                return Err(BooguError::Artifact(format!(
                    "duplicate Burn tensor path {}",
                    spec.target_name
                )));
            }
            if spec.source_shape.is_empty()
                || spec.source_shape.contains(&0)
                || spec.target_shape.is_empty()
                || spec.target_shape.contains(&0)
            {
                return Err(BooguError::Artifact(format!(
                    "invalid shape contract for {}",
                    spec.qualified_source_name()
                )));
            }
            let source_elements = spec.source_shape.iter().product::<usize>();
            let target_elements = spec.target_shape.iter().product::<usize>();
            if source_elements != target_elements {
                return Err(BooguError::Artifact(format!(
                    "element count changes for {}",
                    spec.qualified_source_name()
                )));
            }
            if spec.transform == TensorTransform::Transpose2d
                && (spec.source_shape.len() != 2
                    || spec.target_shape != [spec.source_shape[1], spec.source_shape[0]])
            {
                return Err(BooguError::Artifact(format!(
                    "invalid transpose contract for {}",
                    spec.qualified_source_name()
                )));
            }
            if spec.quantizable
                && (spec.target_shape.len() != 2 || !spec.target_shape[1].is_multiple_of(32))
            {
                return Err(BooguError::Artifact(format!(
                    "Q8S block contract is not aligned for {}",
                    spec.target_name
                )));
            }
        }
        Ok(())
    }

    /// Derive the exact denoiser parameter payload for runtime Q8S block-32/F32 execution.
    ///
    /// Inventory-qualified matrices use one signed byte per value plus one F32 scale per block of
    /// 32 values. All other denoiser tensors are widened to F32 by the runtime load policy. Turbo
    /// excludes edit-only reference-refiner stages. This is a parameter-payload count; allocator
    /// alignment, activations, quantization workspace, and backend kernel scratch are separate.
    pub fn denoiser_runtime_q8s_block32_f32_footprint(
        &self,
        variant: BooguVariant,
    ) -> Result<BooguDenoiserRuntimeQ8Footprint, BooguError> {
        self.denoiser_runtime_q8s_block32_f32_footprint_with_scope(
            variant,
            BooguRuntimeQ8Scope::AllInventoryEligible,
        )
    }

    /// Derive the exact denoiser parameter payload for one closed runtime-Q8 execution scope.
    ///
    /// The scope changes only the runtime adaptation of already-authenticated floating-point
    /// tensors. It does not alter artifact eligibility, inventories, manifests, or content
    /// digests.
    pub fn denoiser_runtime_q8s_block32_f32_footprint_with_scope(
        &self,
        variant: BooguVariant,
        scope: BooguRuntimeQ8Scope,
    ) -> Result<BooguDenoiserRuntimeQ8Footprint, BooguError> {
        scope.validate_variant(variant)?;
        let mut footprint = BooguDenoiserRuntimeQ8Footprint::default();
        for spec in self
            .tensors
            .iter()
            .filter(|spec| spec.owner == TensorOwner::BooguDenoiser)
            .filter(|spec| variant.is_edit() || !spec.stage.starts_with("boogu-reference-refiner-"))
        {
            let elements = spec
                .target_shape
                .iter()
                .try_fold(1_u64, |total, &dimension| {
                    total.checked_mul(dimension as u64)
                })
                .ok_or_else(|| {
                    BooguError::Artifact(format!(
                        "runtime Q8 footprint element count overflowed for {}",
                        spec.target_name
                    ))
                })?;
            if spec.quantizable && scope.quantizes_target(&spec.target_name) {
                if !elements.is_multiple_of(32) {
                    return Err(BooguError::Artifact(format!(
                        "runtime Q8 footprint is not block-32 aligned for {}",
                        spec.target_name
                    )));
                }
                footprint.quantized_elements = footprint
                    .quantized_elements
                    .checked_add(elements)
                    .ok_or_else(|| {
                    BooguError::Artifact("runtime Q8 element count overflowed".into())
                })?;
                footprint.quantized_tensor_count = footprint
                    .quantized_tensor_count
                    .checked_add(1)
                    .ok_or_else(|| {
                        BooguError::Artifact("runtime Q8 tensor count overflowed".into())
                    })?;
                let packed_bytes = elements.checked_add(elements / 32 * 4).ok_or_else(|| {
                    BooguError::Artifact("runtime Q8 byte count overflowed".into())
                })?;
                footprint.quantized_payload_bytes = footprint
                    .quantized_payload_bytes
                    .checked_add(packed_bytes)
                    .ok_or_else(|| {
                        BooguError::Artifact("runtime Q8 byte count overflowed".into())
                    })?;
            } else {
                footprint.f32_tensor_count =
                    footprint.f32_tensor_count.checked_add(1).ok_or_else(|| {
                        BooguError::Artifact("runtime F32 tensor count overflowed".into())
                    })?;
                footprint.f32_elements =
                    footprint
                        .f32_elements
                        .checked_add(elements)
                        .ok_or_else(|| {
                            BooguError::Artifact("runtime F32 element count overflowed".into())
                        })?;
                footprint.f32_payload_bytes = footprint
                    .f32_payload_bytes
                    .checked_add(elements.checked_mul(4).ok_or_else(|| {
                        BooguError::Artifact("runtime F32 byte count overflowed".into())
                    })?)
                    .ok_or_else(|| {
                        BooguError::Artifact("runtime F32 byte count overflowed".into())
                    })?;
            }
            footprint.tensor_count = footprint
                .tensor_count
                .checked_add(1)
                .ok_or_else(|| BooguError::Artifact("runtime Q8 tensor count overflowed".into()))?;
        }
        footprint.total_payload_bytes = footprint
            .quantized_payload_bytes
            .checked_add(footprint.f32_payload_bytes)
            .ok_or_else(|| BooguError::Artifact("runtime Q8 total byte count overflowed".into()))?;
        if footprint.tensor_count == 0 || footprint.total_payload_bytes == 0 {
            return Err(BooguError::Artifact(
                "runtime Q8 denoiser footprint contains no tensors".into(),
            ));
        }
        Ok(footprint)
    }
}

/// Closed selection of inventory-eligible matrices adapted to Q8 at runtime.
///
/// Artifact eligibility remains immutable. A narrower scope keeps selected authenticated F16
/// production weights in F32 only while applying a runtime execution policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BooguRuntimeQ8Scope {
    /// Quantize every matrix marked eligible by the sealed artifact inventory.
    #[default]
    AllInventoryEligible,
    /// Turbo-only runtime stability scope retaining caption and final projection linears in F32.
    TurboCaptionAndTailF32,
    /// Historical Turbo diagnostic scope quantizing attention and feed-forward matrices only.
    ///
    /// Prelude projections, timestep/caption conditioning, every adaptive modulation projection,
    /// and the output tail remain F32. Those comparatively small, high-leverage matrices control
    /// residual scales and gates across the full denoiser. This scope remains available for
    /// reproducible artifact accounting, but is not the current Turbo browser policy.
    TurboAttentionFfnCoreQ8,
    /// Historical Turbo scope quantizing all feed-forward matrices while retaining attention in F32.
    ///
    /// Only the ordinary, image, and instruction feed-forward module families are selected. Every
    /// attention, conditioning, adaptive-modulation, prelude, and output-tail matrix remains F32.
    /// It remains available for reproducible accounting but is not the current browser policy.
    TurboFfnCoreQ8,
    /// Historical Turbo scope quantizing every feed-forward gate and up projection.
    ///
    /// Within the ordinary, image, and instruction feed-forward module families, only
    /// `linear_1.weight` and `linear_3.weight` are selected. The `linear_2.weight` down projection,
    /// attention, conditioning, adaptive modulation, prelude, and output tail remain F32. This
    /// remains available for reproducible accounting but is not the current browser policy.
    TurboFfnGateUpQ8,
    /// Evidence-calibrated Turbo scope selecting the exact 96 main-core gate/up matrices.
    ///
    /// All single-stream gate/up matrices and all dual-stream image/instruction gate/up matrices
    /// are selected. Every down projection plus every context/noise refiner matrix remains F32.
    /// This cap-tight policy is pending a complete real-browser measurement and is not a
    /// numerical-parity claim.
    TurboMainCoreFfnGateUpQ8,
}

impl BooguRuntimeQ8Scope {
    /// Stable provenance label for the runtime-only selection.
    pub const fn label(self) -> &'static str {
        match self {
            Self::AllInventoryEligible => "all-inventory-eligible",
            Self::TurboCaptionAndTailF32 => "turbo-caption-tail-f32",
            Self::TurboAttentionFfnCoreQ8 => "turbo-attention-ffn-core-q8",
            Self::TurboFfnCoreQ8 => "turbo-ffn-core-q8",
            Self::TurboFfnGateUpQ8 => "turbo-ffn-gate-up-q8",
            Self::TurboMainCoreFfnGateUpQ8 => "turbo-main-core-ffn-gate-up-q8",
        }
    }

    fn quantizes_target(self, target: &str) -> bool {
        match self {
            Self::AllInventoryEligible => true,
            Self::TurboCaptionAndTailF32 => !matches!(
                target,
                "time_caption_embed.caption_linear.weight"
                    | "norm_out.linear_1.weight"
                    | "norm_out.linear_2.weight"
            ),
            Self::TurboAttentionFfnCoreQ8 => turbo_attention_ffn_core_q8_target(target),
            Self::TurboFfnCoreQ8 => turbo_ffn_core_q8_target(target),
            Self::TurboFfnGateUpQ8 => turbo_ffn_gate_up_q8_target(target),
            Self::TurboMainCoreFfnGateUpQ8 => turbo_main_core_ffn_gate_up_q8_target(target),
        }
    }

    fn validate_variant(self, variant: BooguVariant) -> Result<(), BooguError> {
        if matches!(
            self,
            Self::TurboCaptionAndTailF32
                | Self::TurboAttentionFfnCoreQ8
                | Self::TurboFfnCoreQ8
                | Self::TurboFfnGateUpQ8
                | Self::TurboMainCoreFfnGateUpQ8
        ) && variant != BooguVariant::Image01Turbo
        {
            return Err(BooguError::Artifact(format!(
                "runtime Q8 scope {} is restricted to Image01Turbo, found {variant:?}",
                self.label()
            )));
        }
        Ok(())
    }
}

fn turbo_attention_ffn_core_q8_target(target: &str) -> bool {
    // Fail closed: only the exact attention and feed-forward module families are quantized. A
    // newly added inventory-eligible projection therefore stays F32 until it is explicitly
    // classified and numerically qualified.
    target.ends_with(".weight")
        && (target.contains(".attn.")
            || target.contains(".joint_attn.")
            || target.contains(".image_self_attn.")
            || target.contains(".feed_forward.")
            || target.contains(".image_ffn.")
            || target.contains(".instruction_ffn."))
}

fn turbo_ffn_core_q8_target(target: &str) -> bool {
    // Fail closed: attention and every newly introduced matrix family remain F32 until an exact
    // classifier and real-checkpoint numerical evidence explicitly admit them.
    target.ends_with(".weight")
        && (target.contains(".feed_forward.")
            || target.contains(".image_ffn.")
            || target.contains(".instruction_ffn."))
}

fn turbo_ffn_gate_up_q8_target(target: &str) -> bool {
    // Fail closed: admit only the two explicitly qualified gate/up projections. In particular,
    // the feed-forward down projection (`linear_2`) stays F32 so accumulated quantization error is
    // not injected directly into the residual stream.
    (target.ends_with(".linear_1.weight") || target.ends_with(".linear_3.weight"))
        && (target.contains(".feed_forward.")
            || target.contains(".image_ffn.")
            || target.contains(".instruction_ffn."))
}

fn canonical_indexed_ffn_projection_target(
    target: &str,
    prefix: &str,
    layer_count: usize,
    module_and_projection: &[&str],
) -> bool {
    let Some(index_and_path) = target.strip_prefix(prefix) else {
        return false;
    };
    let Some((index, path)) = index_and_path.split_once('.') else {
        return false;
    };
    if index.is_empty()
        || (index.len() > 1 && index.starts_with('0'))
        || !index.bytes().all(|byte| byte.is_ascii_digit())
    {
        return false;
    }
    let Ok(index) = index.parse::<usize>() else {
        return false;
    };
    index < layer_count && module_and_projection.contains(&path)
}

fn turbo_main_core_ffn_gate_up_q8_target(target: &str) -> bool {
    const SINGLE_STREAM_GATE_UP: &[&str] = &[
        "feed_forward.linear_1.weight",
        "feed_forward.linear_3.weight",
    ];
    const DOUBLE_STREAM_GATE_UP: &[&str] = &[
        "image_ffn.linear_1.weight",
        "image_ffn.linear_3.weight",
        "instruction_ffn.linear_1.weight",
        "instruction_ffn.linear_3.weight",
    ];

    canonical_indexed_ffn_projection_target(
        target,
        "single_stream_layers.",
        32,
        SINGLE_STREAM_GATE_UP,
    ) || canonical_indexed_ffn_projection_target(
        target,
        "double_stream_layers.",
        8,
        DOUBLE_STREAM_GATE_UP,
    )
}

/// Inventory-derived denoiser parameter payload for runtime Q8S block-32/F32 execution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BooguDenoiserRuntimeQ8Footprint {
    /// Number of included denoiser tensor contracts.
    pub tensor_count: usize,
    /// Number of inventory-qualified tensors stored as blockwise Q8S.
    pub quantized_tensor_count: usize,
    /// Number of tensors widened to ordinary F32.
    pub f32_tensor_count: usize,
    /// Values stored in blockwise Q8S tensors.
    pub quantized_elements: u64,
    /// Values kept as ordinary F32 tensors.
    pub f32_elements: u64,
    /// Packed signed values plus one F32 scale per 32-value block.
    pub quantized_payload_bytes: u64,
    /// Four bytes per non-quantized value.
    pub f32_payload_bytes: u64,
    /// Sum of packed-Q8 and F32 parameter payloads.
    pub total_payload_bytes: u64,
}

fn qwen_specs(config: &Qwen3VlConfig) -> Vec<ArtifactTensorSpec> {
    QwenWeightInventory::for_config(config, true)
        .specs()
        .iter()
        .map(|source| {
            let quantizable = source.role == WeightRole::LinearWeight
                && source.target != "lm_head.weight"
                && source.shape.len() == 2
                && source.shape[1].is_multiple_of(32);
            ArtifactTensorSpec {
                source_component: "mllm".into(),
                source_name: source.source.clone(),
                target_name: source.target.clone(),
                stage: qwen_stage(&source.source),
                owner: TensorOwner::Qwen3Vl,
                source_dtype: SourceDType::Bf16,
                source_shape: source.shape.clone(),
                target_shape: source.shape.clone(),
                // Qwen modules intentionally use Burn's column layout, matching HF storage.
                transform: TensorTransform::Identity,
                quantizable,
            }
        })
        .collect()
}

fn qwen_stage(name: &str) -> String {
    if name == "lm_head.weight" {
        "qwen-lm-head".into()
    } else if name == "model.language_model.embed_tokens.weight" {
        "qwen-embedding".into()
    } else if let Some(index) = indexed_segment(name, "model.visual.blocks.") {
        format!("qwen-vision-block-{index:02}")
    } else if let Some(index) = indexed_segment(name, "model.language_model.layers.") {
        format!("qwen-text-block-{index:02}")
    } else if indexed_segment(name, "model.visual.deepstack_merger_list.").is_some() {
        let index =
            indexed_segment(name, "model.visual.deepstack_merger_list.").expect("segment checked");
        format!("qwen-vision-deepstack-merger-{index:02}")
    } else if name.starts_with("model.visual.merger.") {
        "qwen-vision-final-merger".into()
    } else if name.starts_with("model.visual.patch_embed.")
        || name == "model.visual.pos_embed.weight"
    {
        "qwen-vision-prelude".into()
    } else if name.starts_with("model.language_model.norm.") {
        "qwen-text-final-norm".into()
    } else {
        unreachable!("Qwen inventory emitted an unclassified tensor {name}")
    }
}

#[cfg(feature = "burnpack")]
fn legacy_qwen_stage(name: &str) -> String {
    if name == "lm_head.weight" {
        "qwen-lm-head".into()
    } else if let Some(index) = indexed_segment(name, "model.visual.blocks.") {
        format!("qwen-vision-block-{index:02}")
    } else if let Some(index) = indexed_segment(name, "model.language_model.layers.") {
        format!("qwen-text-block-{index:02}")
    } else if name.starts_with("model.visual") {
        "qwen-vision-core".into()
    } else {
        "qwen-text-core".into()
    }
}

/// Stable artifact component name for one reusable Qwen streaming stage.
pub fn qwen_streaming_stage_name(stage: &Qwen3VlStage) -> String {
    match stage {
        Qwen3VlStage::EmbeddingRows { chunk } => format!("qwen-embedding-rows-{chunk:02}"),
        Qwen3VlStage::VisionPrelude => "qwen-vision-prelude".into(),
        Qwen3VlStage::VisionBlock { index } => format!("qwen-vision-block-{index:02}"),
        Qwen3VlStage::VisionDeepstackMerger { index, .. } => {
            format!("qwen-vision-deepstack-merger-{index:02}")
        }
        Qwen3VlStage::VisionFinalMerger => "qwen-vision-final-merger".into(),
        Qwen3VlStage::TextBlock { index } => format!("qwen-text-block-{index:02}"),
        Qwen3VlStage::TextFinalNorm => "qwen-text-final-norm".into(),
        Qwen3VlStage::LmHeadRows { chunk } => format!("qwen-lm-head-rows-{chunk:02}"),
    }
}

/// Unique Burnpack snapshot path for one row slice of a canonical Qwen table.
pub fn qwen_row_slice_target(logical_target: &str, chunk: &RowChunkSpec) -> String {
    format!(
        "{logical_target}.rows.{:02}.{:06}-{:06}",
        chunk.chunk_index, chunk.row_range.start, chunk.row_range.end
    )
}

fn flux_vae_specs(config: &AutoencoderKlConfig) -> Result<Vec<ArtifactTensorSpec>, BooguError> {
    let inventory = FluxTensorInventory::from_config(config)
        .map_err(|error| BooguError::Artifact(error.to_string()))?;
    Ok(inventory
        .tensors
        .into_iter()
        .map(|source| {
            let transform = if source.diffusers_shape.len() == 2
                && source.burn_shape == [source.diffusers_shape[1], source.diffusers_shape[0]]
            {
                TensorTransform::Transpose2d
            } else {
                TensorTransform::Identity
            };
            let stage = if source.diffusers_name.starts_with("encoder.")
                || source.diffusers_name.starts_with("quant_conv.")
            {
                "flux-vae-encoder"
            } else {
                "flux-vae-decoder"
            };
            ArtifactTensorSpec {
                source_component: "vae".into(),
                source_name: source.diffusers_name,
                target_name: source.burn_name,
                stage: stage.into(),
                owner: TensorOwner::FluxVae,
                source_dtype: SourceDType::F32,
                source_shape: source.diffusers_shape,
                target_shape: source.burn_shape,
                transform,
                // The VAE is only 320 MiB and force-upcasts. Preserve it at F16 in Q8 profiles.
                quantizable: false,
            }
        })
        .collect())
}

fn boogu_specs(config: &BooguConfig) -> Vec<ArtifactTensorSpec> {
    let mut specs = Vec::with_capacity(942);
    let width = config.hidden_size;
    let inner = config.ffn_inner_dim();
    let head_dim = config.head_dim();
    let kv_width = config.num_kv_heads * config.head_dim();
    let condition = width.min(1024);
    let patch = config.patch_size * config.patch_size * config.in_channels;
    let patch_out = config.patch_size * config.patch_size * config.out_channels;

    linear(
        &mut specs,
        "x_embedder",
        "x_embedder",
        patch,
        width,
        true,
        "boogu-prelude",
    );
    linear(
        &mut specs,
        "ref_image_patch_embedder",
        "ref_image_patch_embedder",
        patch,
        width,
        true,
        "boogu-prelude",
    );
    embedding(
        &mut specs,
        "image_index_embedding",
        "image_index_embedding.weight",
        vec![5, width],
        "boogu-prelude",
    );
    rms(
        &mut specs,
        "time_caption_embed.caption_embedder.0",
        "time_caption_embed.caption_norm",
        config.instruction_feature_dim,
        "boogu-prelude",
    );
    linear(
        &mut specs,
        "time_caption_embed.caption_embedder.1",
        "time_caption_embed.caption_linear",
        config.instruction_feature_dim,
        width,
        true,
        "boogu-prelude",
    );
    linear(
        &mut specs,
        "time_caption_embed.timestep_embedder.linear_1",
        "time_caption_embed.time_linear_1",
        256,
        condition,
        true,
        "boogu-prelude",
    );
    linear(
        &mut specs,
        "time_caption_embed.timestep_embedder.linear_2",
        "time_caption_embed.time_linear_2",
        condition,
        condition,
        true,
        "boogu-prelude",
    );
    linear(
        &mut specs,
        "norm_out.linear_1",
        "norm_out.linear_1",
        condition,
        width,
        true,
        "boogu-tail",
    );
    linear(
        &mut specs,
        "norm_out.linear_2",
        "norm_out.linear_2",
        width,
        patch_out,
        true,
        "boogu-tail",
    );

    for index in 0..config.num_refiner_layers {
        let source = format!("context_refiner.{index}");
        let target = source.clone();
        let stage = format!("boogu-context-refiner-{index:02}");
        gqa_attention(
            &mut specs,
            &format!("{source}.attn"),
            &format!("{target}.attn"),
            width,
            kv_width,
            head_dim,
            &stage,
        );
        ffn(
            &mut specs,
            &format!("{source}.feed_forward"),
            &format!("{target}.feed_forward"),
            width,
            inner,
            &stage,
        );
        rms(
            &mut specs,
            &format!("{source}.norm1"),
            &format!("{target}.plain_norm1"),
            width,
            &stage,
        );
        for name in ["norm2", "ffn_norm1", "ffn_norm2"] {
            rms(
                &mut specs,
                &format!("{source}.{name}"),
                &format!("{target}.{name}"),
                width,
                &stage,
            );
        }
    }

    for (collection, label) in [
        ("noise_refiner", "boogu-noise-refiner"),
        ("ref_image_refiner", "boogu-reference-refiner"),
    ] {
        for index in 0..config.num_refiner_layers {
            modulated_single_block(
                &mut specs,
                &format!("{collection}.{index}"),
                width,
                kv_width,
                head_dim,
                inner,
                condition,
                &format!("{label}-{index:02}"),
            );
        }
    }

    for index in 0..config.num_double_stream_layers {
        double_block(
            &mut specs, index, width, kv_width, head_dim, inner, condition,
        );
    }
    for index in 0..config.num_single_stream_layers() {
        modulated_single_block(
            &mut specs,
            &format!("single_stream_layers.{index}"),
            width,
            kv_width,
            head_dim,
            inner,
            condition,
            &format!("boogu-single-block-{index:02}"),
        );
    }
    specs
}

#[allow(clippy::too_many_arguments)]
fn modulated_single_block(
    specs: &mut Vec<ArtifactTensorSpec>,
    prefix: &str,
    width: usize,
    kv_width: usize,
    head_dim: usize,
    inner: usize,
    condition: usize,
    stage: &str,
) {
    gqa_attention(
        specs,
        &format!("{prefix}.attn"),
        &format!("{prefix}.attn"),
        width,
        kv_width,
        head_dim,
        stage,
    );
    ffn(
        specs,
        &format!("{prefix}.feed_forward"),
        &format!("{prefix}.feed_forward"),
        width,
        inner,
        stage,
    );
    adaptive_rms(
        specs,
        &format!("{prefix}.norm1"),
        &format!("{prefix}.norm1"),
        width,
        condition,
        stage,
    );
    for name in ["norm2", "ffn_norm1", "ffn_norm2"] {
        rms(
            specs,
            &format!("{prefix}.{name}"),
            &format!("{prefix}.{name}"),
            width,
            stage,
        );
    }
}

fn double_block(
    specs: &mut Vec<ArtifactTensorSpec>,
    index: usize,
    width: usize,
    kv_width: usize,
    head_dim: usize,
    inner: usize,
    condition: usize,
) {
    let source = format!("double_stream_layers.{index}");
    let target = source.clone();
    let stage = format!("boogu-dual-block-{index:02}");
    let source_joint = format!("{source}.img_instruct_attn");
    let target_joint = format!("{target}.joint_attn");
    for (source_name, target_name, output) in [
        ("processor.img_to_q", "img_to_q", width),
        ("processor.img_to_k", "img_to_k", kv_width),
        ("processor.img_to_v", "img_to_v", kv_width),
        ("processor.instruct_to_q", "instruct_to_q", width),
        ("processor.instruct_to_k", "instruct_to_k", kv_width),
        ("processor.instruct_to_v", "instruct_to_v", kv_width),
        ("processor.img_out", "img_out", width),
        ("processor.instruct_out", "instruct_out", width),
        ("to_out.0", "to_out", width),
    ] {
        linear(
            specs,
            &format!("{source_joint}.{source_name}"),
            &format!("{target_joint}.{target_name}"),
            width,
            output,
            false,
            &stage,
        );
    }
    for name in ["norm_q", "norm_k"] {
        rms(
            specs,
            &format!("{source_joint}.{name}"),
            &format!("{target_joint}.{name}"),
            head_dim,
            &stage,
        );
    }

    gqa_attention(
        specs,
        &format!("{source}.img_self_attn"),
        &format!("{target}.image_self_attn"),
        width,
        kv_width,
        head_dim,
        &stage,
    );
    ffn(
        specs,
        &format!("{source}.img_feed_forward"),
        &format!("{target}.image_ffn"),
        width,
        inner,
        &stage,
    );
    ffn(
        specs,
        &format!("{source}.instruct_feed_forward"),
        &format!("{target}.instruction_ffn"),
        width,
        inner,
        &stage,
    );
    for number in 1..=3 {
        adaptive_rms(
            specs,
            &format!("{source}.img_norm{number}"),
            &format!("{target}.image_norm{number}"),
            width,
            condition,
            &stage,
        );
    }
    for number in 1..=2 {
        adaptive_rms(
            specs,
            &format!("{source}.instruct_norm{number}"),
            &format!("{target}.instruction_norm{number}"),
            width,
            condition,
            &stage,
        );
    }
    for (source_name, target_name) in [
        ("img_attn_norm", "image_attn_norm"),
        ("img_self_attn_norm", "image_self_norm"),
        ("img_ffn_norm1", "image_ffn_norm1"),
        ("img_ffn_norm2", "image_ffn_norm2"),
        ("instruct_attn_norm", "instruction_attn_norm"),
        ("instruct_ffn_norm1", "instruction_ffn_norm1"),
        ("instruct_ffn_norm2", "instruction_ffn_norm2"),
    ] {
        rms(
            specs,
            &format!("{source}.{source_name}"),
            &format!("{target}.{target_name}"),
            width,
            &stage,
        );
    }
}

fn gqa_attention(
    specs: &mut Vec<ArtifactTensorSpec>,
    source: &str,
    target: &str,
    width: usize,
    kv_width: usize,
    head_dim: usize,
    stage: &str,
) {
    for (name, output) in [
        ("to_q", width),
        ("to_k", kv_width),
        ("to_v", kv_width),
        ("to_out.0", width),
    ] {
        let target_name = if name == "to_out.0" { "to_out" } else { name };
        linear(
            specs,
            &format!("{source}.{name}"),
            &format!("{target}.{target_name}"),
            width,
            output,
            false,
            stage,
        );
    }
    for name in ["norm_q", "norm_k"] {
        rms(
            specs,
            &format!("{source}.{name}"),
            &format!("{target}.{name}"),
            head_dim,
            stage,
        );
    }
}

fn ffn(
    specs: &mut Vec<ArtifactTensorSpec>,
    source: &str,
    target: &str,
    width: usize,
    inner: usize,
    stage: &str,
) {
    for (name, input, output) in [
        ("linear_1", width, inner),
        ("linear_2", inner, width),
        ("linear_3", width, inner),
    ] {
        linear(
            specs,
            &format!("{source}.{name}"),
            &format!("{target}.{name}"),
            input,
            output,
            false,
            stage,
        );
    }
}

fn adaptive_rms(
    specs: &mut Vec<ArtifactTensorSpec>,
    source: &str,
    target: &str,
    width: usize,
    condition: usize,
    stage: &str,
) {
    linear(
        specs,
        &format!("{source}.linear"),
        &format!("{target}.linear"),
        condition,
        4 * width,
        true,
        stage,
    );
    rms(
        specs,
        &format!("{source}.norm"),
        &format!("{target}.norm"),
        width,
        stage,
    );
}

#[allow(clippy::too_many_arguments)]
fn linear(
    specs: &mut Vec<ArtifactTensorSpec>,
    source: &str,
    target: &str,
    input: usize,
    output: usize,
    bias: bool,
    stage: &str,
) {
    push_spec(
        specs,
        source,
        target,
        vec![output, input],
        vec![input, output],
        TensorTransform::Transpose2d,
        output.is_multiple_of(32),
        stage,
    );
    if bias {
        push_spec(
            specs,
            &format!("{source}.bias"),
            &format!("{target}.bias"),
            vec![output],
            vec![output],
            TensorTransform::Identity,
            false,
            stage,
        );
    }
}

fn rms(specs: &mut Vec<ArtifactTensorSpec>, source: &str, target: &str, width: usize, stage: &str) {
    push_spec(
        specs,
        &format!("{source}.weight"),
        &format!("{target}.gamma"),
        vec![width],
        vec![width],
        TensorTransform::Identity,
        false,
        stage,
    );
}

fn embedding(
    specs: &mut Vec<ArtifactTensorSpec>,
    source: &str,
    target: &str,
    shape: Vec<usize>,
    stage: &str,
) {
    specs.push(ArtifactTensorSpec {
        source_component: "transformer".into(),
        source_name: source.into(),
        target_name: target.into(),
        stage: stage.into(),
        owner: TensorOwner::BooguDenoiser,
        source_dtype: SourceDType::Bf16,
        source_shape: shape.clone(),
        target_shape: shape,
        transform: TensorTransform::Identity,
        quantizable: false,
    });
}

#[allow(clippy::too_many_arguments)]
fn push_spec(
    specs: &mut Vec<ArtifactTensorSpec>,
    source: &str,
    target: &str,
    source_shape: Vec<usize>,
    target_shape: Vec<usize>,
    transform: TensorTransform,
    quantizable: bool,
    stage: &str,
) {
    let source_name = if source.ends_with(".bias") || source.ends_with(".weight") {
        source.to_owned()
    } else {
        format!("{source}.weight")
    };
    let target_name = if target.ends_with(".bias") || target.ends_with(".gamma") {
        target.to_owned()
    } else {
        format!("{target}.weight")
    };
    specs.push(ArtifactTensorSpec {
        source_component: "transformer".into(),
        source_name,
        target_name,
        stage: stage.into(),
        owner: TensorOwner::BooguDenoiser,
        source_dtype: SourceDType::Bf16,
        source_shape,
        target_shape,
        transform,
        quantizable,
    });
}

fn indexed_segment(name: &str, prefix: &str) -> Option<usize> {
    name.strip_prefix(prefix)?.split('.').next()?.parse().ok()
}

#[cfg(feature = "burnpack")]
mod loading {
    use std::{
        collections::{BTreeMap, BTreeSet},
        fs,
        path::{Path, PathBuf},
        rc::Rc,
    };

    use burn::{
        nn::{RmsNorm, RmsNormConfig},
        prelude::Backend,
        tensor::{
            Bytes, DType, Tensor, TensorData,
            quantization::{
                QuantLevel, QuantParam, QuantScheme, QuantStore, QuantValue, QuantizedBytes,
            },
        },
    };
    use burn_flux_vae::{
        AutoencoderKl, AutoencoderKlConfig, FLUX_VAE_COMPONENT_ROLE, FluxVaeComponentContract,
    };
    use burn_image::{
        ArtifactFile, ArtifactFileRole, ArtifactManifest, ArtifactVerifier, IntegrityPolicy,
        NumericFormat, Sha256Digest, VerificationStatus, VerifiedArtifact,
    };
    use burn_qwen3_vl::{
        AsyncQwen3VlCausalLmStageSource, AsyncQwen3VlStageSource, EmbeddingRowChunk,
        OutputProjectionRowChunk, QWEN_COMPONENT_ROLE, Qwen3VlCausalLmStageSource,
        Qwen3VlComponentContract, Qwen3VlForConditionalGeneration, Qwen3VlImageProcessorConfig,
        Qwen3VlModel, Qwen3VlStage, Qwen3VlStageSource, Qwen3VlStreamingPlan, Qwen3VlVisionPrelude,
        RowChunkPlan, RowChunkSpec,
        text::Qwen3VlDecoderLayer,
        vision::{Qwen3VlVisionBlock, Qwen3VlVisionPatchMerger},
    };
    use burn_store::{
        ApplyResult, BurnpackStore, ModuleAdapter, ModuleSnapshot, ModuleStore, TensorSnapshot,
        TensorSnapshotError,
    };
    use serde::{Deserialize, Serialize};
    use thiserror::Error;

    use super::{
        BOOGU_RELEASE_MAX_METADATA_BYTES, BOOGU_RELEASE_MAX_SHARD_BYTES, BooguArtifactInventory,
        BooguConfig, BooguReleaseIdentity, BooguRuntimeQ8Scope, SourceDType, TensorOwner,
        TensorTransform, legacy_qwen_stage, qwen_row_slice_target, qwen_streaming_stage_name,
        validate_supported_bundle_converter_version,
    };
    use crate::{
        AsyncBooguDenoiserStageSource, AsyncBooguVaeStageSource, BooguDenoiser,
        BooguDenoiserPrelude, BooguDenoiserTail, BooguError, BooguVaeStageSource, BooguVariant,
        DoubleStreamBlock, SingleStreamBlock, StreamingStageSource,
    };

    /// Storage profile whose dtype contract is checked before any tensor is applied.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "kebab-case")]
    pub enum BooguStorageProfile {
        /// All checkpoint tensors stored as F16.
        F16,
        /// Parity profile: Qwen vision stages use F32; all other tensors use F16.
        F16QwenVisionF32,
        /// Aligned matrices stored as Q8S block-32 with F32 scales; all other tensors use F16.
        Q8sBlock32F32,
        /// Qwen vision uses F32; eligible non-vision matrices use Q8S; all else uses F16.
        Q8sBlock32F32QwenVisionF32,
    }

    /// Authenticated identity evidence for the opt-in 1.5K VAE encoder F32 A/B overlay.
    ///
    /// This does not make the overlay a published bundle or a production storage profile. The
    /// browser must receive it through an explicit custom URL and label its result diagnostic.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct BooguVaeEncoderF32DiagnosticManifest {
        /// Sealed content digest of the complete diagnostic overlay manifest.
        pub content_digest: Sha256Digest,
        /// Legacy flat mixed-F16 digest from which all non-encoder payloads are reused.
        pub base_content_digest: Sha256Digest,
        /// Number of tensors whose storage changes from F16 to F32.
        pub replaced_tensors: usize,
        /// Number of bounded Burnpack files holding the replacement encoder.
        pub encoder_weight_files: usize,
        /// Total bytes in the replacement encoder Burnpack files.
        pub encoder_weight_bytes: u64,
    }

    const VAE_ENCODER_F32_DIAGNOSTIC_METADATA: [(&str, &str); 12] = [
        ("diagnostic_manifest_schema", "1"),
        ("diagnostic_kind", "vae-encoder-f32-overlay"),
        (
            "diagnostic_storage_profile",
            super::EDIT_TURBO_1K5_VAE_ENCODER_F32_DIAGNOSTIC_PROFILE,
        ),
        ("diagnostic_base_bundle", "boogu-image-0.1-edit-turbo-1k5"),
        (
            "diagnostic_base_content_digest",
            super::LEGACY_EDIT_TURBO_1K5_F16_QWEN_VISION_F32_CONTENT_DIGEST,
        ),
        ("diagnostic_replaced_stage", "flux-vae-encoder"),
        ("diagnostic_replaced_tensor_count", "106"),
        ("diagnostic_replaced_stored_dtype", "f32"),
        (
            "diagnostic_upstream_vae_repository",
            "Boogu/Boogu-Image-0.1-Edit-Turbo",
        ),
        (
            "diagnostic_upstream_vae_revision",
            super::EDIT_TURBO_REVISION,
        ),
        (
            "diagnostic_upstream_vae_sha256",
            super::EDIT_TURBO_1K5_VAE_SOURCE_SHA256,
        ),
        ("diagnostic_intended_mode", "browser-vae-reference"),
    ];

    /// Stamp the exact diagnostic metadata used by the F32 VAE encoder A/B overlay.
    ///
    /// The caller is still responsible for replacing exactly the encoder object and inventory
    /// entries before sealing. Existing diagnostic metadata is rejected rather than overwritten.
    pub fn stamp_edit_turbo_1k5_vae_encoder_f32_diagnostic_metadata(
        manifest: &mut ArtifactManifest,
    ) -> Result<(), BooguError> {
        if manifest
            .metadata
            .keys()
            .any(|key| key.starts_with("diagnostic_"))
        {
            return Err(BooguError::Artifact(
                "base manifest already contains diagnostic metadata".into(),
            ));
        }
        for (key, value) in VAE_ENCODER_F32_DIAGNOSTIC_METADATA {
            manifest.metadata.insert(key.into(), value.into());
        }
        manifest.metadata.insert(
            "diagnostic_upstream_vae_bytes".into(),
            super::EDIT_TURBO_1K5_VAE_SOURCE_BYTES.to_string(),
        );
        manifest
            .metadata
            .insert("production_qualified".into(), "false".into());
        Ok(())
    }

    /// Authenticate the sealed manifest-level identity of an explicit 1.5K VAE encoder F32 A/B.
    ///
    /// Exact tensor ownership and the rule that only the 106 encoder tensors use F32 are checked
    /// by the normal sealed tensor-inventory verifier. This helper is the narrow selection gate a
    /// browser diagnostic can use instead of accepting an arbitrary non-canonical digest.
    pub fn validate_edit_turbo_1k5_vae_encoder_f32_diagnostic_manifest(
        manifest: &ArtifactManifest,
    ) -> Result<BooguVaeEncoderF32DiagnosticManifest, BooguError> {
        manifest
            .validate_sealed()
            .map_err(|error| BooguError::Artifact(error.to_string()))?;
        let expected_profile = "f16-qwen-vision-f32";
        if manifest.bundle.as_str() != "boogu-image-0.1-edit-turbo-1k5-f16-qwen-vision-f32"
            || manifest.profile.as_str() != expected_profile
            || manifest.model.as_str() != "Boogu/Boogu-Image-0.1-Edit-Turbo-1K5"
            || manifest.model_revision != super::EDIT_TURBO_1K5_REVISION
            || manifest.numeric_format != NumericFormat::Other(expected_profile.into())
            || manifest.metadata.get("profile").map(String::as_str) != Some(expected_profile)
        {
            return Err(BooguError::Artifact(
                "VAE encoder F32 diagnostic manifest has the wrong bundle, release, or base profile"
                    .into(),
            ));
        }
        let expected_keys = VAE_ENCODER_F32_DIAGNOSTIC_METADATA
            .iter()
            .map(|(key, _)| *key)
            .chain(["diagnostic_upstream_vae_bytes"])
            .collect::<BTreeSet<_>>();
        let actual_keys = manifest
            .metadata
            .keys()
            .filter(|key| key.starts_with("diagnostic_"))
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if actual_keys != expected_keys
            || VAE_ENCODER_F32_DIAGNOSTIC_METADATA
                .iter()
                .any(|(key, value)| manifest.metadata.get(*key).map(String::as_str) != Some(*value))
            || manifest
                .metadata
                .get("diagnostic_upstream_vae_bytes")
                .map(String::as_str)
                != Some("335306212")
            || manifest
                .metadata
                .get("production_qualified")
                .map(String::as_str)
                != Some("false")
        {
            return Err(BooguError::Artifact(
                "VAE encoder F32 diagnostic metadata is incomplete or not exact".into(),
            ));
        }
        let encoder_files = manifest
            .files
            .iter()
            .filter(|file| {
                file.role == ArtifactFileRole::Weights
                    && file.component.as_ref().map(|value| value.as_str())
                        == Some("flux-vae-encoder")
            })
            .collect::<Vec<_>>();
        if encoder_files.len() != 1 || encoder_files[0].shard.is_some() {
            return Err(BooguError::Artifact(format!(
                "VAE encoder F32 diagnostic requires one bounded unsharded encoder object, found {}",
                encoder_files.len()
            )));
        }
        let encoder_weight_bytes = encoder_files[0].size;
        if encoder_weight_bytes == 0 || encoder_weight_bytes > super::BOOGU_RELEASE_MAX_SHARD_BYTES
        {
            return Err(BooguError::Artifact(format!(
                "VAE encoder F32 diagnostic object is {encoder_weight_bytes} bytes; limit is {}",
                super::BOOGU_RELEASE_MAX_SHARD_BYTES
            )));
        }
        let content_digest = manifest
            .content_digest
            .ok_or_else(|| BooguError::Artifact("diagnostic manifest is not sealed".into()))?;
        let base_content_digest =
            Sha256Digest::from_hex(super::LEGACY_EDIT_TURBO_1K5_F16_QWEN_VISION_F32_CONTENT_DIGEST)
                .map_err(|error| BooguError::Artifact(error.to_string()))?;
        if content_digest == base_content_digest {
            return Err(BooguError::Artifact(
                "diagnostic overlay aliases the canonical production digest".into(),
            ));
        }
        Ok(BooguVaeEncoderF32DiagnosticManifest {
            content_digest,
            base_content_digest,
            replaced_tensors: super::EDIT_TURBO_1K5_VAE_ENCODER_F32_DIAGNOSTIC_TENSORS,
            encoder_weight_files: encoder_files.len(),
            encoder_weight_bytes,
        })
    }

    /// One immutable bundle published beneath the canonical CDN root.
    ///
    /// Absence from [`PUBLISHED_ARTIFACT_BUNDLES`] means callers must require an explicit local or
    /// custom remote source rather than synthesizing a canonical CDN URL.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct CanonicalBooguArtifactBundle {
        /// Released model variant.
        pub variant: BooguVariant,
        /// Exact storage profile.
        pub profile: BooguStorageProfile,
        /// Single canonical CDN path segment and manifest bundle identity.
        pub bundle_id: &'static str,
        /// Manifest content digest sealed by the release conversion.
        pub content_digest: &'static str,
        /// Exact converter version recorded in the published manifest.
        pub converter_version: &'static str,
    }

    /// The three immutable production parent bundles published beneath the canonical CDN root.
    ///
    /// The shared Qwen and VAE component bundles are also canonical published entries, but parent
    /// dependency pins select them instead of a variant/profile lookup. Hybrid-Q8 and all-F16
    /// bundles remain available as explicitly selected diagnostics and are intentionally absent
    /// from this canonical parent selection set.
    pub const PUBLISHED_ARTIFACT_BUNDLES: [CanonicalBooguArtifactBundle; 3] = [
        CanonicalBooguArtifactBundle {
            variant: BooguVariant::Image01Turbo,
            profile: BooguStorageProfile::F16QwenVisionF32,
            bundle_id: "boogu-image-0.1-turbo",
            content_digest: super::TURBO_F16_QWEN_VISION_F32_CONTENT_DIGEST,
            converter_version: super::PUBLISHED_BUNDLE_CONVERTER_VERSION,
        },
        CanonicalBooguArtifactBundle {
            variant: BooguVariant::Image01EditTurbo,
            profile: BooguStorageProfile::F16QwenVisionF32,
            bundle_id: "boogu-image-0.1-edit-turbo",
            content_digest: super::EDIT_TURBO_F16_QWEN_VISION_F32_CONTENT_DIGEST,
            converter_version: super::PUBLISHED_BUNDLE_CONVERTER_VERSION,
        },
        CanonicalBooguArtifactBundle {
            variant: BooguVariant::Image01EditTurbo1k5,
            profile: BooguStorageProfile::F16QwenVisionF32,
            bundle_id: "boogu-image-0.1-edit-turbo-1k5",
            content_digest: super::EDIT_TURBO_1K5_F16_QWEN_VISION_F32_CONTENT_DIGEST,
            converter_version: super::PUBLISHED_BUNDLE_CONVERTER_VERSION,
        },
    ];

    /// Return the pinned release contract for an actually published variant/profile tuple.
    pub fn canonical_published_bundle(
        variant: BooguVariant,
        profile: BooguStorageProfile,
    ) -> Option<CanonicalBooguArtifactBundle> {
        PUBLISHED_ARTIFACT_BUNDLES
            .iter()
            .copied()
            .find(|bundle| bundle.variant == variant && bundle.profile == profile)
    }

    /// Return the legacy descriptive bundle id for a variant and exact storage profile.
    ///
    /// These identities remain valid for explicit local/custom artifacts and as the source of a
    /// canonical promotion. They are never used to construct a canonical CDN URL.
    pub fn legacy_artifact_bundle_id(
        variant: BooguVariant,
        profile: BooguStorageProfile,
    ) -> String {
        format!(
            "{}-{}",
            release_variant_name(variant),
            release_profile_name(profile)
        )
    }

    /// Return the exact legacy digest eligible for promotion to a canonical production bundle.
    pub const fn promotable_legacy_artifact_digest(
        variant: BooguVariant,
        profile: BooguStorageProfile,
    ) -> Option<&'static str> {
        match (variant, profile) {
            (BooguVariant::Image01Turbo, BooguStorageProfile::F16QwenVisionF32) => {
                Some(super::LEGACY_TURBO_F16_QWEN_VISION_F32_CONTENT_DIGEST)
            }
            (BooguVariant::Image01EditTurbo, BooguStorageProfile::F16QwenVisionF32) => {
                Some(super::LEGACY_EDIT_TURBO_F16_QWEN_VISION_F32_CONTENT_DIGEST)
            }
            (BooguVariant::Image01EditTurbo1k5, BooguStorageProfile::F16QwenVisionF32) => {
                Some(super::LEGACY_EDIT_TURBO_1K5_F16_QWEN_VISION_F32_CONTENT_DIGEST)
            }
            _ => None,
        }
    }

    /// Select the preferred bundle id for a variant/profile tuple.
    ///
    /// The three published mixed-F16 tuples use their clean canonical identities. Diagnostics
    /// retain the explicit legacy-derived dtype suffix and require an explicit source. Importers
    /// must use [`legacy_artifact_bundle_id`] until a candidate is promoted separately.
    pub fn preferred_artifact_bundle_id(
        variant: BooguVariant,
        profile: BooguStorageProfile,
    ) -> String {
        canonical_published_bundle(variant, profile)
            .map(|bundle| bundle.bundle_id.to_owned())
            .unwrap_or_else(|| legacy_artifact_bundle_id(variant, profile))
    }

    /// Whether an explicit/local manifest uses the canonical id or its compatible legacy id.
    pub fn artifact_bundle_id_is_compatible(
        variant: BooguVariant,
        profile: BooguStorageProfile,
        actual: &str,
    ) -> bool {
        actual == legacy_artifact_bundle_id(variant, profile)
            || canonical_published_bundle(variant, profile)
                .is_some_and(|bundle| actual == bundle.bundle_id)
    }

    /// Require the exact content digest pinned for a published variant/profile tuple.
    ///
    /// Diagnostic tuples return an error because they have no canonical published artifact. They
    /// may still be loaded from an explicitly selected local or custom remote source without
    /// presenting that source as a published release.
    pub fn validate_canonical_release_artifact_digest(
        variant: BooguVariant,
        profile: BooguStorageProfile,
        actual: Sha256Digest,
    ) -> Result<(), BooguError> {
        let Some(expected) = canonical_published_bundle(variant, profile) else {
            return Err(BooguError::Artifact(format!(
                "{variant:?}/{profile:?} has no canonical published artifact bundle; use an explicit local or custom remote source for diagnostics"
            )));
        };
        let actual = actual.to_string();
        if actual != expected.content_digest {
            return Err(BooguError::Artifact(format!(
                "published bundle {} requires sealed artifact digest {}, found {actual}",
                expected.bundle_id, expected.content_digest
            )));
        }
        Ok(())
    }

    /// Require an exact artifact manifest qualified by the native Edit-Turbo 1.5K release gates.
    ///
    /// This accepts either the dependency-composed canonical CDN release or the exact legacy flat
    /// monolith used by explicit/local compatibility and diagnostic tooling. Canonical CDN callers
    /// should use [`validate_canonical_release_artifact_digest`] to require canonical identity.
    pub fn validate_edit_turbo_1k5_release_artifact_digest(
        actual: Sha256Digest,
    ) -> Result<(), BooguError> {
        let actual = actual.to_string();
        if matches!(
            actual.as_str(),
            super::EDIT_TURBO_1K5_F16_QWEN_VISION_F32_CONTENT_DIGEST
                | super::LEGACY_EDIT_TURBO_1K5_F16_QWEN_VISION_F32_CONTENT_DIGEST
        ) {
            Ok(())
        } else {
            Err(BooguError::Artifact(format!(
                "Edit-Turbo 1.5K requires canonical composition digest {} or compatible legacy flat digest {}, found {actual}",
                super::EDIT_TURBO_1K5_F16_QWEN_VISION_F32_CONTENT_DIGEST,
                super::LEGACY_EDIT_TURBO_1K5_F16_QWEN_VISION_F32_CONTENT_DIGEST,
            )))
        }
    }

    /// Policy for floating-point snapshots after their release dtype has been verified.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub enum BooguFloatLoadPolicy {
        /// Keep F16 tensors in F16 on backends that support the released dtype.
        #[default]
        Preserve,
        /// Convert floating-point tensors to F32 during application.
        ///
        /// This explicit compatibility mode is useful for CPU backends without F16 support.
        /// Q8S tensors are never dequantized by this policy.
        AdaptToF32,
    }

    /// Policy for verified quantized snapshots when applying a Burnpack stage.
    ///
    /// Burn 0.21's Q8S row-layout kernel is accurate with F32 activations, so Boogu's native
    /// row-layout denoiser can preserve device quantization. Its column-layout load mapper does
    /// not preserve block quantization parameters while transposing, so Qwen must dequantize each
    /// bounded stage on the host before the normal float mapper performs the saved `[out, in]` to
    /// internal `[in, out]` transpose.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub enum BooguQuantizedLoadPolicy {
        /// Keep the sealed Q8S payload quantized on device.
        #[default]
        Preserve,
        /// Dequantize a verified Q8S block-32 payload through F16 before module application.
        ///
        /// When [`BooguFloatLoadPolicy::AdaptToF32`] is also selected, the F16-rounded values are
        /// subsequently widened to F32 for a backend that cannot materialize F16.
        DequantizeF16,
    }

    /// Backend allocator policy while populating a resident denoiser from verified Burnpacks.
    ///
    /// The default preserves allocator caching for existing native/high-VRAM callers. A bounded
    /// residency runtime can explicitly synchronize and release wholly unused upload and allocator
    /// pages after each physical shard, preventing transient buffers from accumulating until the
    /// complete model is loaded.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    #[non_exhaustive]
    pub enum BooguResidentLoadMemoryPolicy {
        /// Preserve the backend allocator cache until the complete resident model is loaded.
        #[default]
        PreserveAllocatorCache,
        /// Synchronize and release transient backend pages after each verified physical shard.
        ReleaseTransientBuffersPerShard,
    }

    impl BooguResidentLoadMemoryPolicy {
        const fn releases_transient_buffers(self) -> bool {
            matches!(self, Self::ReleaseTransientBuffersPerShard)
        }
    }

    /// Runtime conversion policy for verified floating-point Boogu denoiser snapshots.
    ///
    /// This policy is intentionally separate from [`BooguQuantizedLoadPolicy`], which describes
    /// how an already-quantized artifact payload is loaded. Runtime conversion leaves the sealed
    /// production Burnpacks unchanged and is accepted only by verified Boogu denoiser stage
    /// sources. Qwen and VAE loaders cannot enable it.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    #[non_exhaustive]
    pub enum BooguDenoiserRuntimeQuantizationPolicy {
        /// Keep authenticated floating-point denoiser snapshots floating point.
        #[default]
        Disabled,
        /// Convert inventory-eligible F16/F32 denoiser matrices to Q8S block-32/F32 after
        /// integrity, dtype, shape, and inventory eligibility have been verified.
        Q8sBlock32F32,
    }

    impl BooguDenoiserRuntimeQuantizationPolicy {
        /// Stable provenance label for runtime reports.
        pub const fn label(self) -> &'static str {
            match self {
                Self::Disabled => "disabled",
                Self::Q8sBlock32F32 => "runtime-quantize-q8s-block32-f32",
            }
        }
    }

    const fn qwen_quantized_policy(profile: BooguStorageProfile) -> BooguQuantizedLoadPolicy {
        match profile {
            BooguStorageProfile::F16 | BooguStorageProfile::F16QwenVisionF32 => {
                BooguQuantizedLoadPolicy::Preserve
            }
            BooguStorageProfile::Q8sBlock32F32
            | BooguStorageProfile::Q8sBlock32F32QwenVisionF32 => {
                BooguQuantizedLoadPolicy::DequantizeF16
            }
        }
    }

    #[derive(Debug, Clone, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(crate) struct SerializedTensorInventory {
        pub(crate) source_name: String,
        #[serde(default)]
        pub(crate) logical_target_name: Option<String>,
        pub(crate) target_name: String,
        pub(crate) owner: TensorOwner,
        pub(crate) component: String,
        pub(crate) stage: String,
        pub(crate) transform: TensorTransform,
        pub(crate) source_file: String,
        pub(crate) source_dtype: String,
        pub(crate) source_shape: Vec<usize>,
        #[serde(default)]
        pub(crate) source_row_range: Option<[usize; 2]>,
        #[serde(default = "included_by_default")]
        pub(crate) included: bool,
        pub(crate) stored_dtype: Option<String>,
        pub(crate) stored_shape: Option<Vec<usize>>,
        pub(crate) source_offset: u64,
        pub(crate) source_bytes: u64,
        pub(crate) quantized: bool,
        pub(crate) stored_sha256: Option<Sha256Digest>,
        pub(crate) burnpack_object: Option<String>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct SerializedSourceFile {
        path: String,
        size: u64,
        sha256: Sha256Digest,
    }

    #[derive(Debug, Deserialize)]
    struct ReleaseInstructionConfig {
        instruction_feat_dim: usize,
        num_instruction_feature_layers: usize,
        reduce_type: String,
    }

    #[derive(Debug, Deserialize)]
    struct ReleaseDenoiserConfig {
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
        instruction_feature_configs: ReleaseInstructionConfig,
        timestep_scale: f64,
    }

    const fn included_by_default() -> bool {
        true
    }

    /// The three initialized modules populated by a complete artifact load.
    #[derive(Debug)]
    pub struct BooguModels<B: Backend> {
        /// Qwen3-VL conditional-generation model.
        pub qwen: Qwen3VlForConditionalGeneration<B>,
        /// Boogu diffusion transformer.
        pub denoiser: BooguDenoiser<B>,
        /// FLUX AutoencoderKL.
        pub vae: AutoencoderKl<B>,
    }

    /// Auditable staged load statistics.
    #[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
    pub struct BooguLoadReport {
        /// Number of accepted physical shards.
        pub shards: usize,
        /// Number of applied tensors.
        pub tensors: usize,
        /// Applied tensor count by semantic stage.
        pub by_stage: std::collections::BTreeMap<String, usize>,
    }

    /// Result of a complete, bounded semantic verification of one deployment bundle.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize)]
    pub struct BooguReleaseVerification {
        /// Canonical released model variant proven by the manifest and source metadata.
        pub variant: BooguVariant,
        /// Exact artifact storage profile.
        pub profile: BooguStorageProfile,
        /// Manifest-declared files whose bytes were authenticated.
        pub verified_files: usize,
        /// Sum of authenticated manifest payload sizes.
        pub verified_bytes: u64,
        /// Physical Burnpack objects parsed and checked.
        pub verified_weight_objects: usize,
        /// Stored tensor entries checked against Burnpack names, shapes, dtypes, and payload hashes.
        pub verified_tensors: usize,
        /// Largest physical object read during verification.
        pub largest_object_bytes: u64,
        /// Exact declared physical-object ceiling.
        pub max_shard_bytes: u64,
    }

    /// Bounded semantic verification statistics for one reusable dependency bundle.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize)]
    pub struct BooguComponentVerification {
        /// Exact sealed bundle id.
        pub bundle: String,
        /// Exact sealed manifest digest.
        pub content_digest: Sha256Digest,
        /// Manifest-declared payloads whose complete bytes were authenticated.
        pub verified_files: usize,
        /// Sum of authenticated payload sizes.
        pub verified_bytes: u64,
        /// Physical Burnpack objects parsed and checked.
        pub verified_weight_objects: usize,
        /// Stored tensor entries checked through the exact model-owned inventory.
        pub verified_tensors: usize,
        /// Largest physical object read during verification.
        pub largest_object_bytes: u64,
        /// Exact declared physical-object ceiling.
        pub max_shard_bytes: u64,
    }

    /// Complete semantic proof for a schema-v2 Boogu composition and both sealed dependencies.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize)]
    pub struct BooguModularReleaseVerification {
        /// Canonical released model variant proven by the parent manifest.
        pub variant: BooguVariant,
        /// Exact composed storage profile.
        pub profile: BooguStorageProfile,
        /// Parent denoiser/composition verification statistics.
        pub parent: BooguReleaseVerification,
        /// Qwen3-VL base-conditioning component verification statistics.
        pub qwen: BooguComponentVerification,
        /// FLUX VAE component verification statistics.
        pub vae: BooguComponentVerification,
        /// Both dependency edges matched their complete sealed component identities.
        pub dependency_closure_verified: bool,
        /// Both reusable crates accepted their exact component manifests and configs.
        pub component_contracts_verified: bool,
        /// The three owner inventories exactly reconstructed the compiled pipeline inventory.
        pub reconstructed_inventory_verified: bool,
        /// Total authenticated payload declarations across all three bundle prefixes.
        pub verified_files: usize,
        /// Total logical payload bytes across the composed release.
        pub verified_bytes: u64,
        /// Total parsed Burnpack objects across the composed release.
        pub verified_weight_objects: usize,
        /// Total stored tensor entries authenticated across the composed release.
        pub verified_tensors: usize,
        /// Largest bounded object read across the composed release.
        pub largest_object_bytes: u64,
    }

    /// Strict staged artifact loading error.
    #[derive(Debug, Error)]
    pub enum BooguArtifactLoadError {
        /// The immutable release identity did not validate.
        #[error("invalid release identity: {0}")]
        Identity(String),
        /// A model could not be initialized.
        #[error("model initialization failed: {0}")]
        Model(String),
        /// Burnpack parsing or tensor materialization failed.
        #[error("invalid Burnpack shard for {stage}: {message}")]
        Burnpack {
            /// Semantic component being loaded.
            stage: String,
            /// Burnpack parser or materialization diagnostic.
            message: String,
        },
        /// A shard contained an unknown, duplicate, misplaced, or incompatible tensor.
        #[error("artifact tensor contract failed for {stage}: {message}")]
        Contract {
            /// Semantic component being loaded.
            stage: String,
            /// Exact contract violation.
            message: String,
        },
        /// Loading cannot continue after a failed apply operation.
        #[error("artifact loader is poisoned after an earlier failed shard")]
        Poisoned,
        /// Not every required tensor was supplied.
        #[error("artifact set is incomplete; missing {count} tensors, first entries: {sample:?}")]
        Incomplete {
            /// Number of absent required tensors.
            count: usize,
            /// Bounded sample of absent Burn record paths.
            sample: Vec<String>,
        },
    }

    /// Supplies one physical shard at a time to the verified semantic-stage loader.
    ///
    /// The loader, rather than the reader, checks the sealed manifest size and SHA-256 contract.
    /// A browser integration can implement this over a bounded cache; a native integration can
    /// use [`DirectoryStageShardReader`]. The interface is synchronous because Burn's current
    /// [`StreamingStageSource`] execution seam is synchronous.
    pub trait StageShardReader {
        /// Read exactly one manifest-declared file.
        fn read_shard(&mut self, file: &ArtifactFile) -> Result<Vec<u8>, BooguError>;
    }

    /// Bytes returned by an asynchronous shard reader, optionally carrying unforgeable SHA-256
    /// evidence produced from those same bytes.
    ///
    /// Generic readers can use [`Self::unverified`]; the verified source will hash that payload
    /// before parsing it. Readers that already authenticate a response can use
    /// [`Self::verify_sha256`] and pass the resulting evidence through without making the source
    /// hash the object a second time. The payload and evidence fields remain private so callers
    /// cannot replace authenticated bytes while retaining their proof.
    pub struct AsyncStageShardRead {
        bytes: Vec<u8>,
        verification: Option<VerifiedArtifact>,
    }

    impl AsyncStageShardRead {
        /// Wrap bytes from a generic reader. The verified source remains responsible for hashing
        /// them before any payload is parsed.
        pub fn unverified(bytes: Vec<u8>) -> Self {
            Self {
                bytes,
                verification: None,
            }
        }

        /// Verify exact size and SHA-256 once and bind the resulting evidence to these bytes.
        pub fn verify_sha256(file: &ArtifactFile, bytes: Vec<u8>) -> Result<Self, BooguError> {
            let verification = verify_async_stage_bytes(file, &bytes).map_err(|error| {
                BooguError::Artifact(format!(
                    "integrity verification failed for {}: {error}",
                    file.path
                ))
            })?;
            Ok(Self {
                bytes,
                verification: Some(verification),
            })
        }

        /// Consume the wrapper and return its payload bytes, discarding verification evidence.
        ///
        /// Verified model sources do not use this escape hatch; it exists so direct reader users
        /// can preserve the original `read_shard` behavior.
        pub fn into_bytes(self) -> Vec<u8> {
            self.bytes
        }

        fn into_verified_bytes(
            self,
            file: &ArtifactFile,
            max_bytes: u64,
        ) -> Result<Vec<u8>, BooguError> {
            let received = u64::try_from(self.bytes.len()).unwrap_or(u64::MAX);
            if received > max_bytes {
                return Err(BooguError::Artifact(format!(
                    "reader returned {received} bytes for {}, exceeding the per-read cap of {max_bytes}",
                    file.path
                )));
            }
            match self.verification {
                Some(verification)
                    if verification.path() == &file.path
                        && verification.size() == file.size
                        && verification.size() == received
                        && verification.digest() == file.sha256
                        && verification.status() == VerificationStatus::Sha256Verified =>
                {
                    Ok(self.bytes)
                }
                Some(_) => Err(BooguError::Artifact(format!(
                    "reader SHA-256 evidence does not match sealed file {}",
                    file.path
                ))),
                None => {
                    verify_async_stage_bytes(file, &self.bytes).map_err(|error| {
                        BooguError::Artifact(format!(
                            "integrity verification failed for {}: {error}",
                            file.path
                        ))
                    })?;
                    Ok(self.bytes)
                }
            }
        }
    }

    #[cfg(test)]
    std::thread_local! {
        static ASYNC_STAGE_SHA256_PASSES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    fn verify_async_stage_bytes(
        file: &ArtifactFile,
        bytes: &[u8],
    ) -> Result<VerifiedArtifact, burn_image::IntegrityError> {
        #[cfg(test)]
        ASYNC_STAGE_SHA256_PASSES.with(|passes| passes.set(passes.get() + 1));
        ArtifactVerifier::verify_bytes(file, bytes, IntegrityPolicy::RequireSha256)
    }

    /// Wasm-friendly source of one bounded manifest-declared shard at a time.
    ///
    /// Futures intentionally do not require [`Send`]: browser fetch/cache handles and WebGPU
    /// resources commonly remain on one JavaScript event loop. `max_bytes` is supplied before
    /// the read starts so an HTTP/range implementation can abort a response before it exceeds the
    /// release's sealed physical-shard bound. Verified sources independently check the response
    /// cap and either validate typed SHA-256 evidence or hash an unverified response before parsing
    /// any payload.
    #[allow(async_fn_in_trait)]
    pub trait AsyncStageShardReader {
        /// Read exactly the supplied sealed file without exceeding `max_bytes`.
        async fn read_shard(
            &mut self,
            file: &ArtifactFile,
            max_bytes: u64,
        ) -> Result<Vec<u8>, BooguError>;

        /// Read one shard with optional integrity evidence.
        ///
        /// The default preserves generic reader behavior: [`Self::read_shard`] returns ordinary
        /// bytes and the verified source hashes them. Browser transports that already perform
        /// exact SHA-256 verification should override this method and return
        /// [`AsyncStageShardRead::verify_sha256`] so the source validates the evidence instead of
        /// hashing the object again.
        async fn read_stage_shard(
            &mut self,
            file: &ArtifactFile,
            max_bytes: u64,
        ) -> Result<AsyncStageShardRead, BooguError> {
            self.read_shard(file, max_bytes)
                .await
                .map(AsyncStageShardRead::unverified)
        }
    }

    #[cfg(test)]
    mod async_stage_shard_read_tests {
        use burn_image::{ArtifactFileRole, ArtifactPath};

        use super::*;

        fn sealed_file(path: &str, bytes: &[u8]) -> ArtifactFile {
            ArtifactFile {
                path: ArtifactPath::new(path).unwrap(),
                size: bytes.len() as u64,
                sha256: Sha256Digest::calculate(bytes),
                role: ArtifactFileRole::Weights,
                component: None,
                shard: None,
            }
        }

        #[test]
        fn typed_async_shard_evidence_hashes_exactly_once_correctness() {
            let bytes = b"sealed browser object";
            let file = sealed_file("objects/sealed.bpk", bytes);
            ASYNC_STAGE_SHA256_PASSES.with(|passes| passes.set(0));

            let read = AsyncStageShardRead::verify_sha256(&file, bytes.to_vec()).unwrap();
            ASYNC_STAGE_SHA256_PASSES.with(|passes| assert_eq!(passes.get(), 1));
            assert_eq!(
                read.into_verified_bytes(&file, bytes.len() as u64).unwrap(),
                bytes
            );
            ASYNC_STAGE_SHA256_PASSES.with(|passes| assert_eq!(passes.get(), 1));
        }

        #[test]
        fn generic_async_shard_is_hashed_once_by_source_correctness() {
            let bytes = b"generic reader object";
            let file = sealed_file("objects/generic.bpk", bytes);
            ASYNC_STAGE_SHA256_PASSES.with(|passes| passes.set(0));

            let read = AsyncStageShardRead::unverified(bytes.to_vec());
            assert_eq!(
                read.into_verified_bytes(&file, bytes.len() as u64).unwrap(),
                bytes
            );
            ASYNC_STAGE_SHA256_PASSES.with(|passes| assert_eq!(passes.get(), 1));
        }

        #[test]
        fn typed_async_shard_rejects_corrupt_size_and_wrong_identity_correctness() {
            let bytes = b"authenticated object";
            let file = sealed_file("objects/authenticated.bpk", bytes);
            let corrupt =
                match AsyncStageShardRead::verify_sha256(&file, b"corrupted object".to_vec()) {
                    Ok(_) => panic!("corrupt payload must fail SHA-256 verification"),
                    Err(error) => error,
                };
            assert!(
                corrupt
                    .to_string()
                    .contains("integrity verification failed")
            );

            let short = match AsyncStageShardRead::verify_sha256(&file, bytes[..8].to_vec()) {
                Ok(_) => panic!("short payload must fail exact-size verification"),
                Err(error) => error,
            };
            assert!(short.to_string().contains("integrity verification failed"));

            let read = AsyncStageShardRead::verify_sha256(&file, bytes.to_vec()).unwrap();
            let other = sealed_file("objects/other.bpk", bytes);
            let wrong_identity = read
                .into_verified_bytes(&other, bytes.len() as u64)
                .unwrap_err();
            assert!(
                wrong_identity
                    .to_string()
                    .contains("evidence does not match sealed file")
            );

            let read = AsyncStageShardRead::verify_sha256(&file, bytes.to_vec()).unwrap();
            let over_cap = read.into_verified_bytes(&file, (bytes.len() - 1) as u64);
            assert!(over_cap.unwrap_err().to_string().contains("per-read cap"));
        }
    }

    /// Native filesystem reader for sealed artifact bundle directories.
    #[derive(Debug, Clone)]
    pub struct DirectoryStageShardReader {
        root: PathBuf,
    }

    impl DirectoryStageShardReader {
        /// Use `root` as the directory containing the sealed manifest's relative paths.
        pub fn new(root: impl Into<PathBuf>) -> Self {
            Self { root: root.into() }
        }
    }

    impl StageShardReader for DirectoryStageShardReader {
        fn read_shard(&mut self, file: &ArtifactFile) -> Result<Vec<u8>, BooguError> {
            let path = self.root.join(file.path.as_str());
            fs::read(&path).map_err(|error| {
                BooguError::Artifact(format!("failed to read {}: {error}", path.display()))
            })
        }
    }

    /// A native artifact directory whose sealed manifest has been validated.
    ///
    /// Every file returned by [`Self::read_file`] is matched against the manifest and checked for
    /// its exact byte length and SHA-256 digest before the bytes leave this type. This is the
    /// required path for model configs, tokenizers, processor configs, inventories, and weights
    /// whenever a runtime reports `artifacts_verified = true`.
    #[derive(Debug, Clone)]
    pub struct VerifiedArtifactDirectory {
        root: PathBuf,
        manifest: ArtifactManifest,
    }

    impl VerifiedArtifactDirectory {
        /// Open `root/manifest.json` and require a valid sealed content digest.
        pub fn open(root: impl Into<PathBuf>) -> Result<Self, BooguArtifactLoadError> {
            let root = root.into();
            let manifest = read_directory_manifest(&root)?;
            manifest.validate_sealed().map_err(|error| {
                BooguArtifactLoadError::Identity(format!("invalid sealed manifest: {error}"))
            })?;
            Ok(Self { root, manifest })
        }

        /// The validated sealed manifest.
        pub fn manifest(&self) -> &ArtifactManifest {
            &self.manifest
        }

        /// Root containing the manifest-authenticated artifact tree.
        pub fn root(&self) -> &Path {
            &self.root
        }

        /// Read and verify one exact manifest-declared file.
        pub fn read_file(&self, relative_path: &str) -> Result<Vec<u8>, BooguArtifactLoadError> {
            let file = self
                .manifest
                .files
                .iter()
                .find(|file| file.path.as_str() == relative_path)
                .ok_or_else(|| {
                    BooguArtifactLoadError::Identity(format!(
                        "sealed manifest omits required file {relative_path}"
                    ))
                })?;
            let mut reader = DirectoryStageShardReader::new(&self.root);
            let bytes = reader.read_shard(file).map_err(|error| {
                BooguArtifactLoadError::Identity(format!(
                    "failed to read manifest file {relative_path}: {error}"
                ))
            })?;
            ArtifactVerifier::verify_bytes(file, &bytes, IntegrityPolicy::RequireSha256).map_err(
                |error| {
                    BooguArtifactLoadError::Identity(format!(
                        "integrity verification failed for {relative_path}: {error}"
                    ))
                },
            )?;
            Ok(bytes)
        }

        /// Read and verify a UTF-8 manifest-declared file.
        pub fn read_text(&self, relative_path: &str) -> Result<String, BooguArtifactLoadError> {
            String::from_utf8(self.read_file(relative_path)?).map_err(|error| {
                BooguArtifactLoadError::Identity(format!(
                    "manifest file {relative_path} is not UTF-8: {error}"
                ))
            })
        }
    }

    /// Authenticate a complete browser-deployable Boogu release without constructing a model.
    ///
    /// Verification is deliberately stricter than opening a generic [`ArtifactManifest`]. It
    /// proves the canonical Boogu release identity and profile, parses the pinned source configs,
    /// reconstructs the exact compiled tensor inventory, checks every source range, requires the
    /// browser-safe physical-object bound, then parses and authenticates one Burnpack object at a
    /// time. No more than one declared shard enters host memory.
    pub fn verify_release_artifact_directory(
        root: impl AsRef<Path>,
    ) -> Result<BooguReleaseVerification, BooguArtifactLoadError> {
        let directory = VerifiedArtifactDirectory::open(root.as_ref())?;
        let manifest = directory.manifest();
        let variant = release_variant(manifest)?;
        let profile = release_profile(manifest)?;
        validate_release_deployment_bounds(manifest, variant, profile)?;

        let qwen_config = burn_qwen3_vl::Qwen3VlConfig::from_json(&required_release_text(
            &directory,
            "metadata/source/mllm/config.json",
            ArtifactFileRole::Config,
        )?)
        .map_err(|error| contract("source-config", format!("invalid Qwen config: {error}")))?;
        let vae_config = burn_flux_vae::AutoencoderKlConfig::from_diffusers_json(
            &required_release_text(
                &directory,
                "metadata/source/vae/config.json",
                ArtifactFileRole::Config,
            )?,
        )
        .map_err(|error| contract("source-config", format!("invalid FLUX VAE config: {error}")))?;
        let denoiser_config = validate_release_denoiser_config(&required_release_text(
            &directory,
            "metadata/source/transformer/config.json",
            ArtifactFileRole::Config,
        )?)?;
        validate_release_processor_metadata(&directory, &directory, variant)?;

        let inventory = BooguArtifactInventory::new(&qwen_config, &denoiser_config, &vae_config)
            .map_err(|error| {
                contract(
                    "tensor-inventory",
                    format!("invalid model inventory: {error}"),
                )
            })?;
        let identity = BooguReleaseIdentity::canonical(variant);
        validate_release_manifest(&identity, manifest, &inventory, profile)?;

        let mut reader = DirectoryStageShardReader::new(directory.root());
        let entries = verify_inventory_contract(manifest, &inventory, profile, &mut reader)?;
        let (verified_weight_objects, verified_tensors, largest_weight_bytes) =
            verify_release_burnpacks(manifest, &entries, &mut reader)?;

        let mut largest_object_bytes = largest_weight_bytes;
        for file in manifest
            .files
            .iter()
            .filter(|file| file.role != ArtifactFileRole::Weights)
        {
            let bytes = directory.read_file(file.path.as_str())?;
            largest_object_bytes = largest_object_bytes.max(bytes.len() as u64);
        }
        let verified_bytes = manifest.files.iter().try_fold(0_u64, |total, file| {
            total.checked_add(file.size).ok_or_else(|| {
                BooguArtifactLoadError::Identity("manifest payload byte count overflow".into())
            })
        })?;

        Ok(BooguReleaseVerification {
            variant,
            profile,
            verified_files: manifest.files.len(),
            verified_bytes,
            verified_weight_objects,
            verified_tensors,
            largest_object_bytes,
            max_shard_bytes: declared_target_max_shard_bytes(manifest)?,
        })
    }

    /// Authenticate a modular Boogu release and both reusable component bundles.
    ///
    /// The parent must be a schema-v2 composition containing only Boogu denoiser weights and two
    /// exact sibling dependencies. Qwen and VAE identities are validated by their owning crates.
    /// Every compact file is SHA-256 checked, and every Burnpack is parsed one bounded object at a
    /// time against the complete config-derived three-model inventory.
    pub fn verify_modular_release_artifact_directories(
        parent_root: impl AsRef<Path>,
        qwen_root: impl AsRef<Path>,
        vae_root: impl AsRef<Path>,
    ) -> Result<BooguModularReleaseVerification, BooguArtifactLoadError> {
        let parent = VerifiedArtifactDirectory::open(parent_root.as_ref())?;
        let qwen = VerifiedArtifactDirectory::open(qwen_root.as_ref())?;
        let vae = VerifiedArtifactDirectory::open(vae_root.as_ref())?;
        let parent_manifest = parent.manifest();
        let qwen_manifest = qwen.manifest();
        let vae_manifest = vae.manifest();

        let variant = release_variant(parent_manifest)?;
        let profile = release_profile(parent_manifest)?;
        validate_modular_parent_contract(parent_manifest, qwen_manifest, vae_manifest)?;
        validate_release_deployment_bounds(parent_manifest, variant, profile)?;

        let qwen_config = burn_qwen3_vl::Qwen3VlConfig::from_json(&required_release_text(
            &qwen,
            "metadata/source/mllm/config.json",
            ArtifactFileRole::Config,
        )?)
        .map_err(|error| contract("qwen-source-config", error.to_string()))?;
        let vae_config = AutoencoderKlConfig::from_diffusers_json(&required_release_text(
            &vae,
            "metadata/source/vae/config.json",
            ArtifactFileRole::Config,
        )?)
        .map_err(|error| contract("vae-source-config", error.to_string()))?;
        let denoiser_config = validate_release_denoiser_config(&required_release_text(
            &parent,
            "metadata/source/transformer/config.json",
            ArtifactFileRole::Config,
        )?)?;

        Qwen3VlComponentContract::released_base(qwen_manifest.clone(), qwen_config.clone())
            .map_err(|error| contract("qwen-component", error.to_string()))?;
        FluxVaeComponentContract::new(vae_manifest.clone(), vae_config.clone())
            .map_err(|error| contract("vae-component", error.to_string()))?;
        validate_release_processor_metadata(&qwen, &parent, variant)?;

        let inventory = BooguArtifactInventory::new(&qwen_config, &denoiser_config, &vae_config)
            .map_err(|error| contract("tensor-inventory", error.to_string()))?;
        let identity = BooguReleaseIdentity::canonical(variant);
        validate_release_manifest(&identity, parent_manifest, &inventory, profile)?;

        let parent_verification =
            verify_one_modular_owner(&parent, &inventory, profile, Some((variant, profile)))?;
        let qwen_verification = verify_one_modular_owner(&qwen, &inventory, profile, None)?;
        let vae_verification = verify_one_modular_owner(&vae, &inventory, profile, None)?;

        let verified_tensors = parent_verification
            .verified_tensors
            .checked_add(qwen_verification.verified_tensors)
            .and_then(|count| count.checked_add(vae_verification.verified_tensors))
            .ok_or_else(|| contract("verification", "tensor count overflow"))?;
        let expected_stored = inventory
            .tensors()
            .len()
            .checked_sub(1)
            // The one embedding tensor is represented by six row slices (+5 physical entries),
            // while the LM head remains one present-but-omitted inventory entry.
            .and_then(|count| count.checked_add(5))
            .ok_or_else(|| contract("verification", "inventory count overflow"))?;
        if verified_tensors != expected_stored {
            return Err(contract(
                "verification",
                format!(
                    "component tensor closure has {verified_tensors} stored entries, expected {expected_stored}"
                ),
            ));
        }

        let verified_files = parent_verification
            .verified_files
            .checked_add(qwen_verification.verified_files)
            .and_then(|count| count.checked_add(vae_verification.verified_files))
            .ok_or_else(|| contract("verification", "file count overflow"))?;
        let verified_bytes = parent_verification
            .verified_bytes
            .checked_add(qwen_verification.verified_bytes)
            .and_then(|count| count.checked_add(vae_verification.verified_bytes))
            .ok_or_else(|| contract("verification", "byte count overflow"))?;
        let verified_weight_objects = parent_verification
            .verified_weight_objects
            .checked_add(qwen_verification.verified_weight_objects)
            .and_then(|count| count.checked_add(vae_verification.verified_weight_objects))
            .ok_or_else(|| contract("verification", "object count overflow"))?;
        let largest_object_bytes = parent_verification
            .largest_object_bytes
            .max(qwen_verification.largest_object_bytes)
            .max(vae_verification.largest_object_bytes);
        Ok(BooguModularReleaseVerification {
            variant,
            profile,
            parent: BooguReleaseVerification {
                variant,
                profile,
                verified_files: parent_verification.verified_files,
                verified_bytes: parent_verification.verified_bytes,
                verified_weight_objects: parent_verification.verified_weight_objects,
                verified_tensors: parent_verification.verified_tensors,
                largest_object_bytes: parent_verification.largest_object_bytes,
                max_shard_bytes: parent_verification.max_shard_bytes,
            },
            qwen: qwen_verification,
            vae: vae_verification,
            dependency_closure_verified: true,
            component_contracts_verified: true,
            reconstructed_inventory_verified: true,
            verified_files,
            verified_bytes,
            verified_weight_objects,
            verified_tensors,
            largest_object_bytes,
        })
    }

    fn validate_modular_parent_contract(
        parent: &ArtifactManifest,
        qwen: &ArtifactManifest,
        vae: &ArtifactManifest,
    ) -> Result<(), BooguArtifactLoadError> {
        if parent.schema_version != burn_image::ARTIFACT_MANIFEST_SCHEMA_V2
            || parent.dependencies.len() != 2
            || parent.metadata.get("artifact_layout").map(String::as_str)
                != Some("semantic-burnpack-composition-v2")
            || parent
                .metadata
                .get("component_dependency_count")
                .map(String::as_str)
                != Some("2")
        {
            return Err(contract(
                "composition",
                "parent is not the exact two-dependency schema-v2 layout",
            ));
        }
        if parent
            .components
            .iter()
            .any(|component| !component.required || !component.id.as_str().starts_with("boogu-"))
            || parent.files.iter().any(|file| {
                file.role == ArtifactFileRole::Weights
                    && file
                        .component
                        .as_ref()
                        .is_none_or(|component| !component.as_str().starts_with("boogu-"))
            })
        {
            return Err(contract(
                "composition",
                "parent contains a non-denoiser component or weight object",
            ));
        }
        let qwen_dependency = parent
            .dependencies
            .iter()
            .find(|dependency| dependency.role.as_str() == QWEN_COMPONENT_ROLE)
            .ok_or_else(|| contract("composition", "parent omits the qwen dependency"))?;
        let vae_dependency = parent
            .dependencies
            .iter()
            .find(|dependency| dependency.role.as_str() == FLUX_VAE_COMPONENT_ROLE)
            .ok_or_else(|| contract("composition", "parent omits the vae dependency"))?;
        qwen_dependency
            .validate_resolved_manifest(qwen)
            .map_err(|error| contract("qwen-dependency", error.to_string()))?;
        vae_dependency
            .validate_resolved_manifest(vae)
            .map_err(|error| contract("vae-dependency", error.to_string()))?;
        parent
            .validate_dependency_closure(|bundle| {
                if bundle == &qwen.bundle {
                    Some(qwen)
                } else if bundle == &vae.bundle {
                    Some(vae)
                } else {
                    None
                }
            })
            .map_err(|error| contract("dependency-closure", error.to_string()))?;
        Ok(())
    }

    fn verify_one_modular_owner(
        directory: &VerifiedArtifactDirectory,
        inventory: &BooguArtifactInventory,
        profile: BooguStorageProfile,
        parent_identity: Option<(BooguVariant, BooguStorageProfile)>,
    ) -> Result<BooguComponentVerification, BooguArtifactLoadError> {
        let manifest = directory.manifest();
        if let Some((variant, profile)) = parent_identity {
            validate_release_deployment_bounds(manifest, variant, profile)?;
        }
        let mut reader = DirectoryStageShardReader::new(directory.root());
        let entries = verify_inventory_contract(manifest, inventory, profile, &mut reader)?;
        let (verified_weight_objects, verified_tensors, largest_weight_bytes) =
            verify_release_burnpacks(manifest, &entries, &mut reader)?;
        let mut largest_object_bytes = largest_weight_bytes;
        for file in manifest
            .files
            .iter()
            .filter(|file| file.role != ArtifactFileRole::Weights)
        {
            let bytes = directory.read_file(file.path.as_str())?;
            largest_object_bytes = largest_object_bytes.max(bytes.len() as u64);
        }
        let verified_bytes = manifest.files.iter().try_fold(0_u64, |total, file| {
            total
                .checked_add(file.size)
                .ok_or_else(|| contract("verification", "payload byte count overflow"))
        })?;
        Ok(BooguComponentVerification {
            bundle: manifest.bundle.to_string(),
            content_digest: manifest
                .content_digest
                .expect("verified sealed manifest has a content digest"),
            verified_files: manifest.files.len(),
            verified_bytes,
            verified_weight_objects,
            verified_tensors,
            largest_object_bytes,
            max_shard_bytes: declared_target_max_shard_bytes(manifest)?,
        })
    }

    /// Authenticate one of the three exact bundles published beneath the canonical CDN root.
    ///
    /// This performs the complete bounded semantic verification and additionally pins the tuple's
    /// manifest bundle id, converter version, and sealed content digest. Use
    /// [`verify_release_artifact_directory`] for an explicitly selected compatible diagnostic or
    /// custom bundle that must not be reported as a published release.
    pub fn verify_published_release_artifact_directory(
        root: impl AsRef<Path>,
    ) -> Result<BooguReleaseVerification, BooguArtifactLoadError> {
        let verification = verify_release_artifact_directory(root.as_ref())?;
        let directory = VerifiedArtifactDirectory::open(root.as_ref())?;
        let manifest = directory.manifest();
        let published = canonical_published_bundle(verification.variant, verification.profile)
            .ok_or_else(|| {
                BooguArtifactLoadError::Identity(format!(
                    "{:?}/{:?} has no canonical published artifact bundle",
                    verification.variant, verification.profile
                ))
            })?;
        if manifest.bundle.as_str() != published.bundle_id {
            return Err(BooguArtifactLoadError::Identity(format!(
                "published manifest bundle {} differs from pinned {}",
                manifest.bundle, published.bundle_id
            )));
        }
        let converter = manifest
            .metadata
            .get("conversion_crate")
            .map(String::as_str);
        if converter != Some(published.converter_version) {
            return Err(BooguArtifactLoadError::Identity(format!(
                "published bundle {} requires converter {}, found {converter:?}",
                published.bundle_id, published.converter_version
            )));
        }
        validate_published_release_content_digest(
            verification.variant,
            verification.profile,
            manifest.content_digest,
        )?;
        Ok(verification)
    }

    fn release_variant(
        manifest: &ArtifactManifest,
    ) -> Result<BooguVariant, BooguArtifactLoadError> {
        match manifest.model.as_str() {
            "Boogu/Boogu-Image-0.1-Turbo" => Ok(BooguVariant::Image01Turbo),
            "Boogu/Boogu-Image-0.1-Edit-Turbo" => Ok(BooguVariant::Image01EditTurbo),
            "Boogu/Boogu-Image-0.1-Edit-Turbo-1K5" => Ok(BooguVariant::Image01EditTurbo1k5),
            model => Err(BooguArtifactLoadError::Identity(format!(
                "unsupported Boogu release model {model}"
            ))),
        }
    }

    fn release_profile(
        manifest: &ArtifactManifest,
    ) -> Result<BooguStorageProfile, BooguArtifactLoadError> {
        match manifest.profile.as_str() {
            "f16" => Ok(BooguStorageProfile::F16),
            "f16-qwen-vision-f32" => Ok(BooguStorageProfile::F16QwenVisionF32),
            "q8s-block32-f32" => Ok(BooguStorageProfile::Q8sBlock32F32),
            "q8s-block32-f32-qwen-vision-f32" => {
                Ok(BooguStorageProfile::Q8sBlock32F32QwenVisionF32)
            }
            profile => Err(BooguArtifactLoadError::Identity(format!(
                "unsupported Boogu storage profile {profile}"
            ))),
        }
    }

    fn release_variant_name(variant: BooguVariant) -> &'static str {
        match variant {
            BooguVariant::Image01Turbo => "boogu-image-0.1-turbo",
            BooguVariant::Image01EditTurbo => "boogu-image-0.1-edit-turbo",
            BooguVariant::Image01EditTurbo1k5 => "boogu-image-0.1-edit-turbo-1k5",
        }
    }

    fn release_profile_name(profile: BooguStorageProfile) -> &'static str {
        match profile {
            BooguStorageProfile::F16 => "f16",
            BooguStorageProfile::F16QwenVisionF32 => "f16-qwen-vision-f32",
            BooguStorageProfile::Q8sBlock32F32 => "q8s-block32-f32",
            BooguStorageProfile::Q8sBlock32F32QwenVisionF32 => "q8s-block32-f32-qwen-vision-f32",
        }
    }

    pub(super) fn validate_published_release_content_digest(
        variant: BooguVariant,
        profile: BooguStorageProfile,
        content_digest: Option<Sha256Digest>,
    ) -> Result<(), BooguArtifactLoadError> {
        if canonical_published_bundle(variant, profile).is_none() {
            return Ok(());
        }
        let content_digest = content_digest.ok_or_else(|| {
            BooguArtifactLoadError::Identity(
                "published release manifest omits its sealed content digest".into(),
            )
        })?;
        validate_canonical_release_artifact_digest(variant, profile, content_digest)
            .map_err(|error| BooguArtifactLoadError::Identity(error.to_string()))
    }

    fn validate_release_deployment_bounds(
        manifest: &ArtifactManifest,
        variant: BooguVariant,
        profile: BooguStorageProfile,
    ) -> Result<(), BooguArtifactLoadError> {
        let legacy_bundle = legacy_artifact_bundle_id(variant, profile);
        let published_bundle =
            canonical_published_bundle(variant, profile).map(|bundle| bundle.bundle_id);
        if !artifact_bundle_id_is_compatible(variant, profile, manifest.bundle.as_str()) {
            let accepted = published_bundle
                .map(|bundle| format!("{legacy_bundle} or {bundle}"))
                .unwrap_or(legacy_bundle);
            return Err(BooguArtifactLoadError::Identity(format!(
                "bundle {} is not an accepted explicit/canonical release identity; expected {accepted}",
                manifest.bundle,
            )));
        }
        let converter = manifest
            .metadata
            .get("conversion_crate")
            .map(String::as_str);
        if converter
            .is_none_or(|version| validate_supported_bundle_converter_version(version).is_err())
            || manifest
                .metadata
                .get("tensor_inventory_schema")
                .map(String::as_str)
                != Some("2")
            || manifest
                .metadata
                .get("physical_shards_bounded")
                .map(String::as_str)
                != Some("true")
            || manifest
                .metadata
                .get("oversized_tensor_shards")
                .map(String::as_str)
                != Some("0")
        {
            return Err(BooguArtifactLoadError::Identity(format!(
                "release requires supported converter {:?}, tensor inventory v2, and no oversized shards",
                super::SUPPORTED_BUNDLE_CONVERTER_VERSIONS
            )));
        }
        let max_shard_bytes = declared_target_max_shard_bytes(manifest)?;
        if max_shard_bytes > BOOGU_RELEASE_MAX_SHARD_BYTES {
            return Err(BooguArtifactLoadError::Identity(format!(
                "declared shard bound {max_shard_bytes} exceeds browser release maximum {BOOGU_RELEASE_MAX_SHARD_BYTES}"
            )));
        }
        for file in &manifest.files {
            match file.role {
                ArtifactFileRole::Weights => {
                    if file.size > max_shard_bytes {
                        return Err(contract(
                            "release-bounds",
                            format!(
                                "weight object {} is {} bytes, exceeding {max_shard_bytes}",
                                file.path, file.size
                            ),
                        ));
                    }
                    let expected = format!("objects/{}.bpk", file.sha256);
                    if file.path.as_str() != expected {
                        return Err(contract(
                            "release-layout",
                            format!(
                                "weight object {} is not content-addressed as {expected}",
                                file.path
                            ),
                        ));
                    }
                }
                ArtifactFileRole::Config
                | ArtifactFileRole::Tokenizer
                | ArtifactFileRole::Metadata => {
                    if file.size > BOOGU_RELEASE_MAX_METADATA_BYTES {
                        return Err(contract(
                            "release-bounds",
                            format!(
                                "compact object {} is {} bytes, exceeding {BOOGU_RELEASE_MAX_METADATA_BYTES}",
                                file.path, file.size
                            ),
                        ));
                    }
                    if !file.path.as_str().starts_with("metadata/") {
                        return Err(contract(
                            "release-layout",
                            format!("compact object {} is outside metadata/", file.path),
                        ));
                    }
                }
                ArtifactFileRole::Other => {
                    return Err(contract(
                        "release-layout",
                        format!("release object {} has an unsupported role", file.path),
                    ));
                }
            }
        }
        Ok(())
    }

    fn required_release_text(
        directory: &VerifiedArtifactDirectory,
        path: &str,
        role: ArtifactFileRole,
    ) -> Result<String, BooguArtifactLoadError> {
        let file = directory
            .manifest()
            .files
            .iter()
            .find(|file| file.path.as_str() == path)
            .ok_or_else(|| {
                BooguArtifactLoadError::Identity(format!("manifest omits required {path}"))
            })?;
        if file.role != role {
            return Err(BooguArtifactLoadError::Identity(format!(
                "required file {path} has role {:?}, expected {role:?}",
                file.role
            )));
        }
        directory.read_text(path)
    }

    fn validate_release_denoiser_config(json: &str) -> Result<BooguConfig, BooguArtifactLoadError> {
        let source: ReleaseDenoiserConfig = serde_json::from_str(json).map_err(|error| {
            contract(
                "source-config",
                format!("invalid Boogu denoiser config: {error}"),
            )
        })?;
        let config = BooguConfig::default();
        let output_channels = source.out_channels.unwrap_or(source.in_channels);
        let matches = source.patch_size == config.patch_size
            && source.in_channels == config.in_channels
            && output_channels == config.out_channels
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
            return Err(contract(
                "source-config",
                "transformer/config.json differs from the released Boogu architecture",
            ));
        }
        config
            .validate()
            .map_err(|error| contract("source-config", error.to_string()))?;
        Ok(config)
    }

    fn validate_release_processor_metadata(
        qwen_directory: &VerifiedArtifactDirectory,
        pipeline_directory: &VerifiedArtifactDirectory,
        variant: BooguVariant,
    ) -> Result<(), BooguArtifactLoadError> {
        let processor = required_release_text(
            qwen_directory,
            "metadata/source/mllm/preprocessor_config.json",
            ArtifactFileRole::Config,
        )?;
        Qwen3VlImageProcessorConfig::from_json(&processor).map_err(|error| {
            contract(
                "source-config",
                format!("invalid Qwen image processor config: {error}"),
            )
        })?;
        let tokenizer = required_release_text(
            qwen_directory,
            "metadata/source/mllm/tokenizer.json",
            ArtifactFileRole::Tokenizer,
        )?;
        let tokenizer: serde_json::Value = serde_json::from_str(&tokenizer).map_err(|error| {
            contract("source-config", format!("invalid tokenizer JSON: {error}"))
        })?;
        if tokenizer.get("model").is_none() {
            return Err(contract(
                "source-config",
                "Qwen tokenizer JSON omits its model",
            ));
        }
        let template = required_release_text(
            qwen_directory,
            "metadata/source/mllm/chat_template.json",
            ArtifactFileRole::Tokenizer,
        )?;
        let template: serde_json::Value = serde_json::from_str(&template).map_err(|error| {
            contract(
                "source-config",
                format!("invalid chat template JSON: {error}"),
            )
        })?;
        if template.is_null() {
            return Err(contract("source-config", "Qwen chat template is null"));
        }

        let model_index = required_release_text(
            pipeline_directory,
            "metadata/source/model_index.json",
            ArtifactFileRole::Config,
        )?;
        let model_index: serde_json::Value =
            serde_json::from_str(&model_index).map_err(|error| {
                contract(
                    "source-config",
                    format!("invalid model_index.json: {error}"),
                )
            })?;
        let expected_pipeline = match variant {
            BooguVariant::Image01Turbo => "BooguImageTurboPipeline",
            // The pinned Edit repository metadata names the parent pipeline. Runtime execution is
            // still explicitly routed through the DMD Turbo implementation.
            BooguVariant::Image01EditTurbo => "BooguImagePipeline",
            BooguVariant::Image01EditTurbo1k5 => "BooguImagePipeline",
        };
        let expected_components = [
            ("mllm", "transformers", "Qwen3VLForConditionalGeneration"),
            ("processor", "transformers", "Qwen3VLProcessor"),
            (
                "transformer",
                "transformer_boogu",
                "BooguImageTransformer2DModel",
            ),
            ("vae", "diffusers", "AutoencoderKL"),
        ];
        if model_index
            .get("_class_name")
            .and_then(serde_json::Value::as_str)
            != Some(expected_pipeline)
            || expected_components.iter().any(|(key, library, class)| {
                model_index
                    .get(key)
                    .and_then(serde_json::Value::as_array)
                    .is_none_or(|entry| {
                        entry.first().and_then(serde_json::Value::as_str) != Some(*library)
                            || entry.get(1).and_then(serde_json::Value::as_str) != Some(*class)
                    })
            })
        {
            return Err(contract(
                "source-config",
                "model_index.json differs from the pinned Boogu component graph",
            ));
        }
        Ok(())
    }

    pub(super) fn verify_release_burnpacks<R: StageShardReader>(
        manifest: &ArtifactManifest,
        entries: &[SerializedTensorInventory],
        reader: &mut R,
    ) -> Result<(usize, usize, u64), BooguArtifactLoadError> {
        let mut by_object = BTreeMap::<&str, Vec<&SerializedTensorInventory>>::new();
        for entry in entries.iter().filter(|entry| entry.included) {
            let object = entry.burnpack_object.as_deref().ok_or_else(|| {
                contract(
                    "tensor-inventory",
                    format!("{} omits its Burnpack object", entry.target_name),
                )
            })?;
            by_object.entry(object).or_default().push(entry);
        }

        let mut objects = 0_usize;
        let mut tensors = 0_usize;
        let mut largest = 0_u64;
        for file in manifest
            .files
            .iter()
            .filter(|file| file.role == ArtifactFileRole::Weights)
        {
            let bytes =
                reader
                    .read_shard(file)
                    .map_err(|error| BooguArtifactLoadError::Burnpack {
                        stage: file.path.to_string(),
                        message: error.to_string(),
                    })?;
            ArtifactVerifier::verify_bytes(file, &bytes, IntegrityPolicy::RequireSha256).map_err(
                |error| BooguArtifactLoadError::Burnpack {
                    stage: file.path.to_string(),
                    message: format!("integrity verification failed: {error}"),
                },
            )?;
            largest = largest.max(bytes.len() as u64);
            let mut store = BurnpackStore::from_bytes(Some(Bytes::from_bytes_vec(bytes)));
            let snapshots =
                store
                    .get_all_snapshots()
                    .map_err(|error| BooguArtifactLoadError::Burnpack {
                        stage: file.path.to_string(),
                        message: error.to_string(),
                    })?;
            let expected = by_object.get(file.path.as_str()).ok_or_else(|| {
                contract(
                    "tensor-inventory",
                    format!("weight object {} has no tensor entries", file.path),
                )
            })?;
            let expected_names = expected
                .iter()
                .map(|entry| entry.target_name.as_str())
                .collect::<BTreeSet<_>>();
            let actual_names = snapshots
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>();
            if actual_names != expected_names {
                return Err(contract(
                    "burnpack",
                    format!(
                        "object {} tensor keyset differs from its sealed inventory",
                        file.path
                    ),
                ));
            }
            for entry in expected {
                let snapshot = snapshots
                    .get(&entry.target_name)
                    .expect("Burnpack keyset equality checked");
                let expected_shape = entry.stored_shape.as_ref().ok_or_else(|| {
                    contract(
                        "tensor-inventory",
                        format!("stored tensor {} omits its shape", entry.target_name),
                    )
                })?;
                if snapshot.shape.as_slice() != expected_shape.as_slice() {
                    return Err(contract(
                        "burnpack",
                        format!(
                            "tensor {} shape {:?} differs from sealed {:?}",
                            entry.target_name, snapshot.shape, expected_shape
                        ),
                    ));
                }
                validate_entry_dtype(entry, snapshot.dtype).map_err(|message| {
                    contract(
                        "burnpack",
                        format!("tensor {} dtype mismatch: {message}", entry.target_name),
                    )
                })?;
                let data =
                    snapshot
                        .to_data()
                        .map_err(|error| BooguArtifactLoadError::Burnpack {
                            stage: file.path.to_string(),
                            message: format!(
                                "failed to materialize {}: {error}",
                                entry.target_name
                            ),
                        })?;
                let actual_digest = Sha256Digest::calculate(data.bytes.as_ref());
                let expected_digest = entry.stored_sha256.ok_or_else(|| {
                    contract(
                        "tensor-inventory",
                        format!(
                            "stored tensor {} omits its payload digest",
                            entry.target_name
                        ),
                    )
                })?;
                if actual_digest != expected_digest {
                    return Err(contract(
                        "burnpack",
                        format!(
                            "tensor {} payload digest {actual_digest} differs from sealed {expected_digest}",
                            entry.target_name
                        ),
                    ));
                }
                tensors += 1;
            }
            objects += 1;
            // The store, snapshots, and object bytes are dropped before the next shard is read.
        }
        Ok((objects, tensors, largest))
    }

    pub(crate) async fn read_verified_async<R: AsyncStageShardReader + ?Sized>(
        reader: &mut R,
        file: &ArtifactFile,
        max_bytes: u64,
    ) -> Result<Vec<u8>, BooguError> {
        if file.size > max_bytes {
            return Err(BooguError::Artifact(format!(
                "sealed file {} is {} bytes, exceeding the per-read cap of {max_bytes}",
                file.path, file.size
            )));
        }
        reader
            .read_stage_shard(file, max_bytes)
            .await?
            .into_verified_bytes(file, max_bytes)
    }

    #[derive(Default)]
    struct PrefetchedStageShardReader {
        files: BTreeMap<String, (ArtifactFile, Vec<u8>)>,
    }

    impl StageShardReader for PrefetchedStageShardReader {
        fn read_shard(&mut self, file: &ArtifactFile) -> Result<Vec<u8>, BooguError> {
            let (expected, bytes) = self.files.get(file.path.as_str()).ok_or_else(|| {
                BooguError::Artifact(format!("metadata file {} was not prefetched", file.path))
            })?;
            if expected != file {
                return Err(BooguError::Artifact(format!(
                    "prefetched metadata identity differs for {}",
                    file.path
                )));
            }
            Ok(bytes.clone())
        }
    }

    pub(crate) async fn verify_inventory_contract_async<R: AsyncStageShardReader + ?Sized>(
        manifest: &ArtifactManifest,
        inventory: &BooguArtifactInventory,
        profile: BooguStorageProfile,
        reader: &mut R,
        max_bytes: u64,
    ) -> Result<Vec<SerializedTensorInventory>, BooguArtifactLoadError> {
        let mut prefetched = PrefetchedStageShardReader::default();
        for path in [
            "metadata/tensor-inventory.json",
            "metadata/source-files.json",
        ] {
            let file = manifest
                .files
                .iter()
                .find(|file| file.path.as_str() == path)
                .ok_or_else(|| {
                    BooguArtifactLoadError::Identity(format!("manifest omits {path}"))
                })?;
            let bytes = read_verified_async(reader, file, max_bytes)
                .await
                .map_err(|error| BooguArtifactLoadError::Burnpack {
                    stage: path.into(),
                    message: error.to_string(),
                })?;
            prefetched.files.insert(path.into(), (file.clone(), bytes));
        }
        verify_inventory_contract(manifest, inventory, profile, &mut prefetched)
    }

    /// Verified reusable native source for independently loading the FLUX VAE encoder/decoder.
    ///
    /// Each call rereads only its selected bounded stage. This permits edit pipelines to load and
    /// drop the encoder before DMD, then load the decoder without keeping both VAE halves resident.
    pub struct VerifiedDirectoryVaeStageSource<B: Backend> {
        identity: BooguReleaseIdentity,
        root: PathBuf,
        inventory: BooguArtifactInventory,
        config: burn_flux_vae::AutoencoderKlConfig,
        profile: BooguStorageProfile,
        float_policy: BooguFloatLoadPolicy,
        device: B::Device,
    }

    impl<B: Backend> VerifiedDirectoryVaeStageSource<B> {
        /// Validate the sealed directory and both exact VAE stage inventories up front.
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            identity: &BooguReleaseIdentity,
            root: impl AsRef<Path>,
            inventory: BooguArtifactInventory,
            config: burn_flux_vae::AutoencoderKlConfig,
            profile: BooguStorageProfile,
            float_policy: BooguFloatLoadPolicy,
            device: B::Device,
        ) -> Result<Self, BooguArtifactLoadError> {
            let root = root.as_ref();
            let manifest = read_directory_manifest(root)?;
            validate_release_manifest(identity, &manifest, &inventory, profile)?;
            let mut reader = DirectoryStageShardReader::new(root);
            verify_inventory_contract(&manifest, &inventory, profile, &mut reader)?;
            let max_bytes = declared_target_max_shard_bytes(&manifest)?;
            for stage in ["flux-vae-encoder", "flux-vae-decoder"] {
                let files = manifest
                    .files
                    .iter()
                    .filter(|file| {
                        file.role == ArtifactFileRole::Weights
                            && file.component.as_ref().map(|value| value.as_str()) == Some(stage)
                    })
                    .collect::<Vec<_>>();
                if files.is_empty() {
                    return Err(contract(stage, "sealed manifest omits VAE stage"));
                }
                if let Some(file) = files.iter().find(|file| file.size > max_bytes) {
                    return Err(contract(
                        stage,
                        format!(
                            "VAE shard {} is {} bytes, exceeding {max_bytes}",
                            file.path, file.size
                        ),
                    ));
                }
            }
            Ok(Self {
                identity: identity.clone(),
                root: root.to_owned(),
                inventory,
                config,
                profile,
                float_policy,
                device,
            })
        }

        /// Borrow the sealed artifact directory.
        pub fn root(&self) -> &Path {
            &self.root
        }
    }

    impl<B: Backend> BooguVaeStageSource<B> for VerifiedDirectoryVaeStageSource<B> {
        fn load_encoder(&mut self) -> Result<AutoencoderKl<B>, BooguError> {
            load_vae_encoder_from_directory(
                &self.identity,
                &self.root,
                self.inventory.clone(),
                self.config.clone(),
                self.profile,
                self.float_policy,
                &self.device,
            )
            .map(|(model, _)| model)
            .map_err(|error| BooguError::Artifact(error.to_string()))
        }

        fn load_decoder(&mut self) -> Result<AutoencoderKl<B>, BooguError> {
            load_vae_decoder_from_directory(
                &self.identity,
                &self.root,
                self.inventory.clone(),
                self.config.clone(),
                self.profile,
                self.float_policy,
                &self.device,
            )
            .map(|(model, _)| model)
            .map_err(|error| BooguError::Artifact(error.to_string()))
        }
    }

    /// Async verified source for independently loading the FLUX VAE encoder and decoder.
    ///
    /// Only the selected half is populated. Physical Burnpacks are fetched, bounded, verified,
    /// applied, and released sequentially; the opposite half remains lazy and unfetched.
    pub struct VerifiedAsyncBurnpackVaeStageSource<B: Backend, R> {
        config: burn_flux_vae::AutoencoderKlConfig,
        device: B::Device,
        entries: Vec<SerializedTensorInventory>,
        stages: BTreeMap<String, Vec<ArtifactFile>>,
        reader: R,
        max_bytes: u64,
        float_policy: BooguFloatLoadPolicy,
    }

    impl<B: Backend, R: AsyncStageShardReader> VerifiedAsyncBurnpackVaeStageSource<B, R> {
        /// Validate both VAE half inventories and attach a bounded asynchronous reader.
        #[allow(clippy::too_many_arguments)]
        pub async fn new(
            identity: &BooguReleaseIdentity,
            manifest: ArtifactManifest,
            inventory: BooguArtifactInventory,
            config: burn_flux_vae::AutoencoderKlConfig,
            profile: BooguStorageProfile,
            float_policy: BooguFloatLoadPolicy,
            device: B::Device,
            mut reader: R,
        ) -> Result<Self, BooguArtifactLoadError> {
            validate_release_manifest(identity, &manifest, &inventory, profile)?;
            let max_bytes = declared_target_max_shard_bytes(&manifest)?;
            let entries = verify_inventory_contract_async(
                &manifest,
                &inventory,
                profile,
                &mut reader,
                max_bytes,
            )
            .await?;
            let required_stages = inventory
                .tensors()
                .iter()
                .filter(|spec| spec.owner == TensorOwner::FluxVae)
                .map(|spec| spec.stage.clone())
                .collect::<BTreeSet<_>>();
            let expected_stages =
                BTreeSet::from(["flux-vae-encoder".to_owned(), "flux-vae-decoder".to_owned()]);
            if required_stages != expected_stages {
                return Err(contract(
                    "vae",
                    format!("compiled VAE inventory has unexpected stages {required_stages:?}"),
                ));
            }
            let actual_stages = entries
                .iter()
                .filter(|entry| entry.owner == TensorOwner::FluxVae && entry.included)
                .map(|entry| entry.stage.clone())
                .collect::<BTreeSet<_>>();
            if actual_stages != required_stages {
                return Err(contract(
                    "vae",
                    format!(
                        "stored VAE stages differ from exact inventory: expected={required_stages:?}, actual={actual_stages:?}"
                    ),
                ));
            }
            let mut stages = BTreeMap::<String, Vec<ArtifactFile>>::new();
            for file in manifest
                .files
                .iter()
                .filter(|file| file.role == ArtifactFileRole::Weights)
            {
                let Some(stage) = file.component.as_ref().map(|value| value.as_str()) else {
                    continue;
                };
                if required_stages.contains(stage) {
                    if file.size > max_bytes {
                        return Err(contract(
                            stage,
                            format!(
                                "streamed VAE shard {} is {} bytes, exceeding {max_bytes}",
                                file.path, file.size
                            ),
                        ));
                    }
                    stages
                        .entry(stage.to_owned())
                        .or_default()
                        .push(file.clone());
                }
            }
            for stage in &required_stages {
                if !stages.contains_key(stage) {
                    return Err(contract(stage, "sealed manifest omits required VAE stage"));
                }
            }
            for files in stages.values_mut() {
                files.sort_by_key(|file| file.shard.map(|shard| shard.index).unwrap_or(0));
            }
            Ok(Self {
                config,
                device,
                entries,
                stages,
                reader,
                max_bytes,
                float_policy,
            })
        }

        /// Borrow the asynchronous transport/cache reader.
        pub const fn reader(&self) -> &R {
            &self.reader
        }

        /// Mutably borrow the asynchronous transport/cache reader.
        pub fn reader_mut(&mut self) -> &mut R {
            &mut self.reader
        }

        /// Maximum response size passed to and enforced around every read.
        pub const fn max_shard_bytes(&self) -> u64 {
            self.max_bytes
        }

        async fn load_stage(&mut self, stage: &str) -> Result<AutoencoderKl<B>, BooguError> {
            let files = self.stages.get(stage).cloned().ok_or_else(|| {
                BooguError::Artifact(format!("manifest has no VAE stage {stage}"))
            })?;
            let expected = self
                .entries
                .iter()
                .filter(|entry| {
                    entry.owner == TensorOwner::FluxVae
                        && entry.included
                        && entry.stage == stage
                        && entry.source_row_range.is_none()
                })
                .map(|entry| (entry.target_name.clone(), entry.clone()))
                .collect::<BTreeMap<_, _>>();
            if expected.is_empty() {
                return Err(BooguError::Artifact(format!(
                    "VAE stage {stage} has no exact tensor contracts"
                )));
            }
            let mut module = self
                .config
                .clone()
                .try_init(&self.device)
                .map_err(|error| BooguError::Artifact(error.to_string()))?;
            let mut applied = BTreeSet::new();
            for file in files {
                let bytes = read_verified_async(&mut self.reader, &file, self.max_bytes).await?;
                let mut store = BurnpackStore::from_bytes(Some(Bytes::from_bytes_vec(bytes)));
                let snapshots = store.get_all_snapshots().map_err(|error| {
                    BooguError::Artifact(format!("invalid Burnpack {}: {error}", file.path))
                })?;
                if snapshots.is_empty() {
                    return Err(BooguError::Artifact(format!(
                        "VAE stage {stage} contains an empty Burnpack {}",
                        file.path
                    )));
                }
                let mut local = Vec::with_capacity(snapshots.len());
                let mut local_paths = BTreeSet::new();
                for (name, snapshot) in snapshots {
                    let entry = expected.get(name).ok_or_else(|| {
                        BooguError::Artifact(format!(
                            "VAE stage {stage} contains unknown tensor {name}"
                        ))
                    })?;
                    if entry.burnpack_object.as_deref() != Some(file.path.as_str()) {
                        return Err(BooguError::Artifact(format!(
                            "VAE stage {stage} tensor {name} is in {}, but its sealed object is {:?}",
                            file.path, entry.burnpack_object
                        )));
                    }
                    if !applied.insert(name.clone()) {
                        return Err(BooguError::Artifact(format!(
                            "VAE stage {stage} repeats tensor {name}"
                        )));
                    }
                    if entry.stored_shape.as_deref() != Some(snapshot.shape.as_slice()) {
                        return Err(BooguError::Artifact(format!(
                            "VAE stage {stage} tensor {name} shape differs from sealed inventory"
                        )));
                    }
                    validate_entry_dtype(entry, snapshot.dtype).map_err(|message| {
                        BooguError::Artifact(format!(
                            "VAE stage {stage} tensor {name} dtype mismatch: {message}"
                        ))
                    })?;
                    if !local_paths.insert(name.clone()) {
                        return Err(BooguError::Artifact(format!(
                            "VAE stage {stage} repeats local tensor {name}"
                        )));
                    }
                    local.push(snapshot.clone());
                }
                let result = module.apply(
                    local,
                    None,
                    load_adapter(self.float_policy, BooguQuantizedLoadPolicy::Preserve),
                    false,
                );
                validate_partial_apply(stage, &result, &local_paths).map_err(|error| {
                    BooguError::Artifact(format!("failed to apply VAE stage {stage}: {error}"))
                })?;
            }
            let missing = expected
                .keys()
                .filter(|name| !applied.contains(*name))
                .take(16)
                .cloned()
                .collect::<Vec<_>>();
            if !missing.is_empty() {
                return Err(BooguError::Artifact(format!(
                    "VAE stage {stage} is incomplete; first missing tensors: {missing:?}"
                )));
            }
            Ok(module)
        }
    }

    impl<B: Backend, R: AsyncStageShardReader> AsyncBooguVaeStageSource<B>
        for VerifiedAsyncBurnpackVaeStageSource<B, R>
    {
        async fn load_encoder(&mut self) -> Result<AutoencoderKl<B>, BooguError> {
            self.load_stage("flux-vae-encoder").await
        }

        async fn load_decoder(&mut self) -> Result<AutoencoderKl<B>, BooguError> {
            self.load_stage("flux-vae-decoder").await
        }

        async fn synchronize(&mut self) -> Result<(), BooguError> {
            B::sync(&self.device).map_err(|error| {
                BooguError::Artifact(format!("device sync after VAE stage failed: {error}"))
            })
        }
    }

    /// Hash-verifying one-block-at-a-time source for [`crate::StreamingBooguDenoiser`].
    ///
    /// The source validates the sealed manifest once. Every stage then reads, verifies, applies,
    /// and drops one physical Burnpack shard before reading the next. Returned modules contain
    /// exactly one refiner/block (or the small prelude/tail), so the executor never constructs the
    /// complete 10B denoiser.
    pub struct VerifiedBurnpackStageSource<B: Backend, R> {
        config: BooguConfig,
        inventory: BooguArtifactInventory,
        variant: BooguVariant,
        profile: BooguStorageProfile,
        device: B::Device,
        stages: BTreeMap<String, Vec<ArtifactFile>>,
        reader: R,
        float_policy: BooguFloatLoadPolicy,
        quantized_policy: BooguQuantizedLoadPolicy,
        runtime_quantization_policy: BooguDenoiserRuntimeQuantizationPolicy,
        runtime_q8_scope: BooguRuntimeQ8Scope,
    }

    impl<B: Backend, R: StageShardReader> VerifiedBurnpackStageSource<B, R> {
        /// Validate a sealed release manifest and attach a bounded physical-shard reader.
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            identity: &BooguReleaseIdentity,
            manifest: ArtifactManifest,
            inventory: BooguArtifactInventory,
            config: BooguConfig,
            profile: BooguStorageProfile,
            device: B::Device,
            mut reader: R,
        ) -> Result<Self, BooguArtifactLoadError> {
            validate_release_manifest(identity, &manifest, &inventory, profile)?;
            verify_inventory_contract(&manifest, &inventory, profile, &mut reader)?;
            let required_stages = inventory
                .tensors()
                .iter()
                .filter(|spec| spec.owner == TensorOwner::BooguDenoiser)
                .map(|spec| spec.stage.as_str())
                .collect::<BTreeSet<_>>();
            let mut stages = BTreeMap::<String, Vec<ArtifactFile>>::new();
            for file in manifest
                .files
                .iter()
                .filter(|file| file.role == ArtifactFileRole::Weights)
            {
                let Some(component) = file.component.as_ref() else {
                    continue;
                };
                if required_stages.contains(component.as_str()) {
                    stages
                        .entry(component.as_str().to_owned())
                        .or_default()
                        .push(file.clone());
                }
            }
            for stage in &required_stages {
                if !stages.contains_key(*stage) {
                    return Err(contract(stage, "sealed manifest omits required stage"));
                }
            }
            for files in stages.values_mut() {
                files.sort_by_key(|file| file.shard.map(|shard| shard.index).unwrap_or(0));
            }
            let target_max = declared_target_max_shard_bytes(&manifest)?;
            for (stage, files) in &stages {
                if let Some(file) = files.iter().find(|file| file.size > target_max) {
                    return Err(contract(
                        stage,
                        format!(
                            "streamed denoiser shard {} is {} bytes, exceeding the manifest target bound of {target_max} bytes",
                            file.path, file.size
                        ),
                    ));
                }
            }
            Ok(Self {
                config,
                inventory,
                variant: identity.variant,
                profile,
                device,
                stages,
                reader,
                float_policy: BooguFloatLoadPolicy::Preserve,
                quantized_policy: BooguQuantizedLoadPolicy::Preserve,
                runtime_quantization_policy: BooguDenoiserRuntimeQuantizationPolicy::Disabled,
                runtime_q8_scope: BooguRuntimeQ8Scope::AllInventoryEligible,
            })
        }

        /// Explicitly adapt F16/BF16/F64 snapshots to F32 before applying them.
        ///
        /// This is intended for CPU backends such as `NdArray` that cannot materialize F16.
        /// Quantized snapshots remain quantized. WGPU/WebGPU callers should retain the default
        /// `false` value so the released profile stays resident in its declared storage dtype.
        pub fn with_float_load_policy(mut self, policy: BooguFloatLoadPolicy) -> Self {
            self.float_policy = policy;
            self
        }

        /// Select how already-quantized, verified Q8S denoiser matrices are loaded.
        ///
        /// Production Boogu row-layout stages use [`BooguQuantizedLoadPolicy::Preserve`] together
        /// with [`BooguFloatLoadPolicy::AdaptToF32`] and F32 activations.
        pub fn with_quantized_load_policy(mut self, policy: BooguQuantizedLoadPolicy) -> Self {
            self.quantized_policy = policy;
            self
        }

        /// Select whether eligible verified float denoiser matrices are quantized at runtime.
        ///
        /// The default is [`BooguDenoiserRuntimeQuantizationPolicy::Disabled`]. Enabling Q8S
        /// conversion does not alter artifact identity and remains constrained by the sealed
        /// inventory plus [`Self::with_runtime_q8_scope`].
        pub fn with_runtime_quantization_policy(
            mut self,
            policy: BooguDenoiserRuntimeQuantizationPolicy,
        ) -> Self {
            self.runtime_quantization_policy = policy;
            self
        }

        /// Select the closed subset of inventory-eligible matrices quantized at runtime.
        pub fn with_runtime_q8_scope(mut self, scope: BooguRuntimeQ8Scope) -> Self {
            self.runtime_q8_scope = scope;
            self
        }

        /// Enable or disable the explicit F32 compatibility adapter.
        ///
        /// Prefer [`Self::with_float_load_policy`] when propagating a typed runtime policy.
        pub fn with_float32_adapter(self, enabled: bool) -> Self {
            let policy = if enabled {
                BooguFloatLoadPolicy::AdaptToF32
            } else {
                BooguFloatLoadPolicy::Preserve
            };
            self.with_float_load_policy(policy)
        }

        /// Borrow the underlying shard reader for cache and transport statistics.
        pub const fn reader(&self) -> &R {
            &self.reader
        }

        /// Mutably borrow the underlying shard reader.
        pub fn reader_mut(&mut self) -> &mut R {
            &mut self.reader
        }

        fn load_module<M: ModuleSnapshot<B>>(
            &mut self,
            stage: &str,
            prefix: &str,
            mut module: M,
        ) -> Result<M, BooguError> {
            validate_runtime_denoiser_quantization_policy(
                self.profile,
                self.float_policy,
                self.quantized_policy,
                self.runtime_quantization_policy,
                self.runtime_q8_scope,
                self.variant,
                stage,
            )?;
            let files = self.stages.get(stage).cloned().ok_or_else(|| {
                BooguError::Artifact(format!("manifest has no denoiser stage {stage}"))
            })?;
            let expected = self
                .inventory
                .tensors()
                .iter()
                .filter(|spec| spec.stage == stage)
                .map(|spec| spec.target_name.clone())
                .collect::<BTreeSet<_>>();
            if let Some(name) = expected.iter().find(|name| !name.starts_with(prefix)) {
                return Err(BooguError::Artifact(format!(
                    "stage {stage} tensor {name} does not start with {prefix:?}"
                )));
            }

            let mut applied = BTreeSet::new();
            for file in files {
                let bytes = self.reader.read_shard(&file)?;
                ArtifactVerifier::verify_bytes(&file, &bytes, IntegrityPolicy::RequireSha256)
                    .map_err(|error| {
                        BooguError::Artifact(format!(
                            "integrity verification failed for {}: {error}",
                            file.path
                        ))
                    })?;
                let mut store = BurnpackStore::from_bytes(Some(Bytes::from_bytes_vec(bytes)));
                let raw = store.get_all_snapshots().map_err(|error| {
                    BooguError::Artifact(format!("invalid Burnpack {}: {error}", file.path))
                })?;
                if raw.is_empty() {
                    return Err(BooguError::Artifact(format!(
                        "stage {stage} contains an empty Burnpack {}",
                        file.path
                    )));
                }
                let mut local = Vec::with_capacity(raw.len());
                let mut runtime_quantizable_paths = BTreeSet::new();
                for (name, snapshot) in raw {
                    let spec = self.inventory.by_target(name).ok_or_else(|| {
                        BooguError::Artifact(format!(
                            "stage {stage} contains unknown tensor {name}"
                        ))
                    })?;
                    if spec.stage != stage {
                        return Err(BooguError::Artifact(format!(
                            "stage {stage} contains tensor {name} assigned to {}",
                            spec.stage
                        )));
                    }
                    if !applied.insert(name.clone()) {
                        return Err(BooguError::Artifact(format!(
                            "stage {stage} repeats tensor {name}"
                        )));
                    }
                    if snapshot.shape.as_slice() != spec.target_shape {
                        return Err(BooguError::Artifact(format!(
                            "stage {stage} tensor {name} shape mismatch: expected {:?}, found {:?}",
                            spec.target_shape, snapshot.shape
                        )));
                    }
                    validate_spec_dtype(self.profile, spec, snapshot.dtype).map_err(|message| {
                        BooguError::Artifact(format!(
                            "stage {stage} tensor {name} dtype mismatch: {message}"
                        ))
                    })?;
                    let local_name = name.strip_prefix(prefix).ok_or_else(|| {
                        BooguError::Artifact(format!(
                            "stage {stage} tensor {name} does not start with {prefix:?}"
                        ))
                    })?;
                    if spec.quantizable && self.runtime_q8_scope.quantizes_target(&spec.target_name)
                    {
                        runtime_quantizable_paths.insert(local_name.to_owned());
                    }
                    local.push(rename_snapshot(snapshot, local_name));
                }
                let expected_applied = local
                    .iter()
                    .map(TensorSnapshot::full_path)
                    .collect::<BTreeSet<_>>();
                let adapter = denoiser_load_adapter(
                    self.float_policy,
                    self.quantized_policy,
                    self.runtime_quantization_policy,
                    runtime_quantizable_paths,
                );
                let result = module.apply(local, None, adapter, false);
                validate_partial_apply(stage, &result, &expected_applied).map_err(|error| {
                    BooguError::Artifact(format!("failed to apply stage {stage}: {error}"))
                })?;
                // `bytes`, its store, and all host snapshots are dropped before the next file.
            }
            let missing = expected.difference(&applied).cloned().collect::<Vec<_>>();
            if !missing.is_empty() {
                return Err(BooguError::Artifact(format!(
                    "stage {stage} is incomplete; missing {} tensors: {:?}",
                    missing.len(),
                    missing.iter().take(16).collect::<Vec<_>>()
                )));
            }
            Ok(module)
        }
    }

    impl<B: Backend> VerifiedBurnpackStageSource<B, DirectoryStageShardReader> {
        /// Load a sealed native artifact directory containing `manifest.json`.
        pub fn from_directory(
            identity: &BooguReleaseIdentity,
            root: impl AsRef<Path>,
            inventory: BooguArtifactInventory,
            config: BooguConfig,
            profile: BooguStorageProfile,
            device: B::Device,
        ) -> Result<Self, BooguArtifactLoadError> {
            let root = root.as_ref();
            let manifest = read_directory_manifest(root)?;
            Self::new(
                identity,
                manifest,
                inventory,
                config,
                profile,
                device,
                DirectoryStageShardReader::new(root),
            )
        }
    }

    /// Hash-verifying source for the reusable one-stage-at-a-time Qwen3-VL executor.
    ///
    /// Vocabulary tables are materialized one bounded row slice at a time. Every ordinary module
    /// is initialized lazily and populated without calling `collect`, preserving Burn's saved
    /// `[out, in]` contract for column-layout linear parameters.
    pub struct VerifiedBurnpackQwenStageSource<B: Backend, R> {
        config: burn_qwen3_vl::Qwen3VlConfig,
        plan: Qwen3VlStreamingPlan,
        device: B::Device,
        entries: Vec<SerializedTensorInventory>,
        stages: BTreeMap<String, Vec<ArtifactFile>>,
        reader: R,
        float_policy: BooguFloatLoadPolicy,
        quantized_policy: BooguQuantizedLoadPolicy,
    }

    impl<B: Backend, R: StageShardReader> VerifiedBurnpackQwenStageSource<B, R> {
        /// Validate a sealed release, exact source inventory, and Qwen streaming plan.
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            identity: &BooguReleaseIdentity,
            manifest: ArtifactManifest,
            inventory: BooguArtifactInventory,
            config: burn_qwen3_vl::Qwen3VlConfig,
            plan: Qwen3VlStreamingPlan,
            profile: BooguStorageProfile,
            device: B::Device,
            mut reader: R,
        ) -> Result<Self, BooguArtifactLoadError> {
            validate_release_manifest(identity, &manifest, &inventory, profile)?;
            let entries = verify_inventory_contract(&manifest, &inventory, profile, &mut reader)?;
            let canonical = Qwen3VlStreamingPlan::new(
                &config,
                plan.embedding_rows.clone(),
                plan.lm_head_rows.clone(),
            )
            .map_err(|error| BooguArtifactLoadError::Model(error.to_string()))?;
            if canonical != plan {
                return Err(contract(
                    "qwen",
                    "streaming descriptors differ from the canonical config/row plan",
                ));
            }
            let required_stages = plan
                .stages
                .iter()
                .map(|descriptor| qwen_streaming_stage_name(&descriptor.stage))
                .collect::<BTreeSet<_>>();
            let actual_stages = entries
                .iter()
                .filter(|entry| entry.owner == TensorOwner::Qwen3Vl && entry.included)
                .map(|entry| entry.stage.clone())
                .collect::<BTreeSet<_>>();
            if actual_stages != required_stages {
                return Err(contract(
                    "qwen",
                    format!(
                        "stored Qwen stages differ from the selected streaming plan: expected={required_stages:?}, actual={actual_stages:?}"
                    ),
                ));
            }
            let target_max = declared_target_max_shard_bytes(&manifest)?;
            let mut stages = BTreeMap::<String, Vec<ArtifactFile>>::new();
            for file in manifest
                .files
                .iter()
                .filter(|file| file.role == ArtifactFileRole::Weights)
            {
                let Some(stage) = file.component.as_ref().map(|value| value.as_str()) else {
                    continue;
                };
                if required_stages.contains(stage) {
                    if file.size > target_max {
                        return Err(contract(
                            stage,
                            format!(
                                "streamed Qwen shard {} is {} bytes, exceeding {target_max}",
                                file.path, file.size
                            ),
                        ));
                    }
                    stages
                        .entry(stage.to_owned())
                        .or_default()
                        .push(file.clone());
                }
            }
            for stage in &required_stages {
                if !stages.contains_key(stage) {
                    return Err(contract(stage, "sealed manifest omits required Qwen stage"));
                }
            }
            for files in stages.values_mut() {
                files.sort_by_key(|file| file.shard.map(|shard| shard.index).unwrap_or(0));
            }
            Ok(Self {
                config,
                plan,
                device,
                entries,
                stages,
                reader,
                float_policy: BooguFloatLoadPolicy::Preserve,
                quantized_policy: qwen_quantized_policy(profile),
            })
        }

        /// Explicit compatibility conversion for CPU backends without F16 support.
        pub fn with_float_load_policy(mut self, policy: BooguFloatLoadPolicy) -> Self {
            self.float_policy = policy;
            self
        }

        /// Override the profile-derived Q8S application policy.
        ///
        /// Both production Q8 profiles default to [`BooguQuantizedLoadPolicy::DequantizeF16`]
        /// because Burn 0.21 cannot transpose a block-quantized Col parameter correctly.
        pub fn with_quantized_load_policy(mut self, policy: BooguQuantizedLoadPolicy) -> Self {
            self.quantized_policy = policy;
            self
        }

        /// Borrow the validated streaming plan used by this source.
        pub const fn plan(&self) -> &Qwen3VlStreamingPlan {
            &self.plan
        }

        /// Borrow the underlying transport/cache reader.
        pub const fn reader(&self) -> &R {
            &self.reader
        }

        /// Mutably borrow the underlying transport/cache reader.
        pub fn reader_mut(&mut self) -> &mut R {
            &mut self.reader
        }

        fn load_qwen_module<M: ModuleSnapshot<B>>(
            &mut self,
            stage: &str,
            prefix: &str,
            mut module: M,
        ) -> Result<M, BooguError> {
            let files = self.stages.get(stage).cloned().ok_or_else(|| {
                BooguError::Artifact(format!("manifest has no Qwen stage {stage}"))
            })?;
            let expected = self
                .entries
                .iter()
                .filter(|entry| {
                    entry.owner == TensorOwner::Qwen3Vl
                        && entry.included
                        && entry.stage == stage
                        && entry.source_row_range.is_none()
                })
                .map(|entry| (entry.target_name.clone(), entry))
                .collect::<BTreeMap<_, _>>();
            if expected.is_empty() {
                return Err(BooguError::Artifact(format!(
                    "Qwen module stage {stage} has no exact tensor contracts"
                )));
            }
            let mut applied = BTreeSet::new();
            for file in files {
                let bytes = self.reader.read_shard(&file)?;
                ArtifactVerifier::verify_bytes(&file, &bytes, IntegrityPolicy::RequireSha256)
                    .map_err(|error| {
                        BooguError::Artifact(format!(
                            "integrity verification failed for {}: {error}",
                            file.path
                        ))
                    })?;
                let mut store = BurnpackStore::from_bytes(Some(Bytes::from_bytes_vec(bytes)));
                let snapshots = store.get_all_snapshots().map_err(|error| {
                    BooguError::Artifact(format!("invalid Burnpack {}: {error}", file.path))
                })?;
                if snapshots.is_empty() {
                    return Err(BooguError::Artifact(format!(
                        "Qwen stage {stage} contains an empty Burnpack {}",
                        file.path
                    )));
                }
                let mut local = Vec::with_capacity(snapshots.len());
                for (name, snapshot) in snapshots {
                    let entry = expected.get(name).ok_or_else(|| {
                        BooguError::Artifact(format!(
                            "Qwen stage {stage} contains unknown tensor {name}"
                        ))
                    })?;
                    if !applied.insert(name.clone()) {
                        return Err(BooguError::Artifact(format!(
                            "Qwen stage {stage} repeats tensor {name}"
                        )));
                    }
                    if entry.stored_shape.as_deref() != Some(snapshot.shape.as_slice()) {
                        return Err(BooguError::Artifact(format!(
                            "Qwen stage {stage} tensor {name} shape differs from sealed inventory"
                        )));
                    }
                    validate_entry_dtype(entry, snapshot.dtype).map_err(|message| {
                        BooguError::Artifact(format!(
                            "Qwen stage {stage} tensor {name} dtype mismatch: {message}"
                        ))
                    })?;
                    let local_name = name.strip_prefix(prefix).ok_or_else(|| {
                        BooguError::Artifact(format!(
                            "Qwen stage {stage} tensor {name} lacks prefix {prefix:?}"
                        ))
                    })?;
                    local.push(rename_snapshot(snapshot, local_name));
                }
                let expected_applied = local
                    .iter()
                    .map(TensorSnapshot::full_path)
                    .collect::<BTreeSet<_>>();
                let result = module.apply(
                    local,
                    None,
                    load_adapter(self.float_policy, self.quantized_policy),
                    false,
                );
                validate_partial_apply(stage, &result, &expected_applied).map_err(|error| {
                    BooguError::Artifact(format!("failed to apply Qwen stage {stage}: {error}"))
                })?;
            }
            let missing = expected
                .keys()
                .filter(|name| !applied.contains(*name))
                .take(16)
                .cloned()
                .collect::<Vec<_>>();
            if !missing.is_empty() {
                return Err(BooguError::Artifact(format!(
                    "Qwen stage {stage} is incomplete; first missing tensors: {missing:?}"
                )));
            }
            Ok(module)
        }

        fn load_row_tensor(
            &mut self,
            spec: &RowChunkSpec,
            lm_head: bool,
        ) -> Result<Tensor<B, 2>, BooguError> {
            let stage = qwen_streaming_stage_name(&if lm_head {
                Qwen3VlStage::LmHeadRows {
                    chunk: spec.chunk_index,
                }
            } else {
                Qwen3VlStage::EmbeddingRows {
                    chunk: spec.chunk_index,
                }
            });
            let logical = if lm_head {
                "lm_head.weight"
            } else {
                "model.language_model.embed_tokens.weight"
            };
            let target = qwen_row_slice_target(logical, spec);
            let entry = self
                .entries
                .iter()
                .find(|entry| {
                    entry.owner == TensorOwner::Qwen3Vl
                        && entry.included
                        && entry.stage == stage
                        && entry.target_name == target
                        && entry.source_row_range
                            == Some([spec.row_range.start, spec.row_range.end])
                })
                .ok_or_else(|| {
                    BooguError::Artifact(format!(
                        "sealed inventory omits Qwen row slice {stage}:{target}"
                    ))
                })?;
            if entry.stored_shape.as_deref() != Some([spec.rows(), spec.hidden_size].as_slice()) {
                return Err(BooguError::Artifact(format!(
                    "sealed Qwen row slice {target} has the wrong shape"
                )));
            }
            let files = self.stages.get(&stage).cloned().ok_or_else(|| {
                BooguError::Artifact(format!("manifest has no Qwen row stage {stage}"))
            })?;
            let mut found = None;
            for file in files {
                let bytes = self.reader.read_shard(&file)?;
                ArtifactVerifier::verify_bytes(&file, &bytes, IntegrityPolicy::RequireSha256)
                    .map_err(|error| {
                        BooguError::Artifact(format!(
                            "integrity verification failed for {}: {error}",
                            file.path
                        ))
                    })?;
                let mut store = BurnpackStore::from_bytes(Some(Bytes::from_bytes_vec(bytes)));
                let snapshots = store.get_all_snapshots().map_err(|error| {
                    BooguError::Artifact(format!("invalid Burnpack {}: {error}", file.path))
                })?;
                for (name, snapshot) in snapshots {
                    if name.as_str() != target || found.is_some() {
                        return Err(BooguError::Artifact(format!(
                            "Qwen row stage {stage} contains an unknown or duplicate tensor {name}"
                        )));
                    }
                    if snapshot.dtype != DType::F16
                        || snapshot.shape.as_slice() != [spec.rows(), spec.hidden_size]
                    {
                        return Err(BooguError::Artifact(format!(
                            "Qwen row tensor {name} violates its F16 shape contract"
                        )));
                    }
                    let mut data = snapshot.to_data().map_err(|error| {
                        BooguError::Artifact(format!("failed to read Qwen row {name}: {error}"))
                    })?;
                    if self.float_policy == BooguFloatLoadPolicy::AdaptToF32 {
                        data = data.convert_dtype(DType::F32);
                    }
                    // `Tensor::from_data` uses the backend's default float element (F32 for the
                    // WGPU type used here), even when the sealed row payload is F16. Reapply the
                    // verified release dtype so embeddings and the streamed text-stage weights
                    // have one exact execution dtype.
                    let dtype = data.dtype;
                    found = Some(Tensor::<B, 2>::from_data(data, (&self.device, dtype)));
                }
            }
            found.ok_or_else(|| BooguError::Artifact(format!("Qwen row stage {stage} is empty")))
        }
    }

    impl<B: Backend> VerifiedBurnpackQwenStageSource<B, DirectoryStageShardReader> {
        /// Attach a native directory and reconstruct its exact sealed row-slice plan.
        ///
        /// Unlike [`Self::from_directory`], callers do not need to duplicate the converter's row
        /// partitioning policy. The plan is derived from the hash-verified tensor inventory and
        /// then checked by the normal constructor against every stored Qwen semantic stage.
        #[allow(clippy::too_many_arguments)]
        pub fn from_directory_auto(
            identity: &BooguReleaseIdentity,
            root: impl AsRef<Path>,
            inventory: BooguArtifactInventory,
            config: burn_qwen3_vl::Qwen3VlConfig,
            profile: BooguStorageProfile,
            device: B::Device,
        ) -> Result<Self, BooguArtifactLoadError> {
            let root = root.as_ref();
            let manifest = read_directory_manifest(root)?;
            validate_release_manifest(identity, &manifest, &inventory, profile)?;
            let mut reader = DirectoryStageShardReader::new(root);
            let entries = verify_inventory_contract(&manifest, &inventory, profile, &mut reader)?;
            let plan = qwen_streaming_plan_from_entries(&config, &entries)?;
            Self::new(
                identity,
                manifest,
                inventory,
                config,
                plan,
                profile,
                device,
                DirectoryStageShardReader::new(root),
            )
        }

        /// Attach the native directory reader to a sealed Qwen streaming artifact.
        #[allow(clippy::too_many_arguments)]
        pub fn from_directory(
            identity: &BooguReleaseIdentity,
            root: impl AsRef<Path>,
            inventory: BooguArtifactInventory,
            config: burn_qwen3_vl::Qwen3VlConfig,
            plan: Qwen3VlStreamingPlan,
            profile: BooguStorageProfile,
            device: B::Device,
        ) -> Result<Self, BooguArtifactLoadError> {
            let root = root.as_ref();
            let manifest = read_directory_manifest(root)?;
            Self::new(
                identity,
                manifest,
                inventory,
                config,
                plan,
                profile,
                device,
                DirectoryStageShardReader::new(root),
            )
        }
    }

    fn qwen_streaming_plan_from_entries(
        config: &burn_qwen3_vl::Qwen3VlConfig,
        entries: &[SerializedTensorInventory],
    ) -> Result<Qwen3VlStreamingPlan, BooguArtifactLoadError> {
        fn rows(
            config: &burn_qwen3_vl::Qwen3VlConfig,
            entries: &[SerializedTensorInventory],
            source_name: &str,
            required: bool,
        ) -> Result<Option<RowChunkPlan>, BooguArtifactLoadError> {
            let source = entries
                .iter()
                .filter(|entry| {
                    entry.owner == TensorOwner::Qwen3Vl
                        && entry.component == "mllm"
                        && entry.source_name == source_name
                })
                .collect::<Vec<_>>();
            if source
                .iter()
                .any(|entry| entry.included && entry.source_row_range.is_none())
            {
                return Err(contract(
                    "qwen",
                    format!(
                        "{source_name} is stored as one resident tensor; production streaming requires sealed row slices"
                    ),
                ));
            }
            let mut ranges = source
                .into_iter()
                .filter(|entry| entry.included)
                .filter_map(|entry| entry.source_row_range)
                .collect::<Vec<_>>();
            ranges.sort_unstable();
            if ranges.is_empty() {
                if required {
                    return Err(contract(
                        "qwen",
                        format!("sealed inventory omits streamed {source_name} rows"),
                    ));
                }
                return Ok(None);
            }
            let chunks = ranges
                .into_iter()
                .enumerate()
                .map(|(chunk_index, [start, end])| RowChunkSpec {
                    chunk_index,
                    row_range: start..end,
                    total_rows: config.text_config.vocab_size,
                    hidden_size: config.text_config.hidden_size,
                    element_bytes: 2,
                })
                .collect();
            Ok(Some(RowChunkPlan { chunks }))
        }

        let embedding_rows = rows(
            config,
            entries,
            "model.language_model.embed_tokens.weight",
            true,
        )?
        .expect("required row plan was checked");
        let lm_head_rows = rows(config, entries, "lm_head.weight", false)?;
        Qwen3VlStreamingPlan::new(config, embedding_rows, lm_head_rows)
            .map_err(|error| contract("qwen", error.to_string()))
    }

    impl<B: Backend, R: StageShardReader> Qwen3VlStageSource<B>
        for VerifiedBurnpackQwenStageSource<B, R>
    {
        type Error = BooguError;

        fn load_embedding_rows(
            &mut self,
            spec: &RowChunkSpec,
        ) -> Result<EmbeddingRowChunk<B>, Self::Error> {
            let tensor = self.load_row_tensor(spec, false)?;
            EmbeddingRowChunk::new(spec.clone(), tensor)
                .map_err(|error| BooguError::Artifact(error.to_string()))
        }

        fn load_vision_prelude(&mut self) -> Result<Qwen3VlVisionPrelude<B>, Self::Error> {
            let module = Qwen3VlVisionPrelude::new(self.config.vision_config.clone(), &self.device)
                .map_err(|error| BooguError::Artifact(error.to_string()))?;
            self.load_qwen_module("qwen-vision-prelude", "model.visual.", module)
        }

        fn load_vision_block(
            &mut self,
            index: usize,
        ) -> Result<Qwen3VlVisionBlock<B>, Self::Error> {
            let stage = format!("qwen-vision-block-{index:02}");
            let prefix = format!("model.visual.blocks.{index}.");
            let module = Qwen3VlVisionBlock::new(&self.config.vision_config, &self.device);
            self.load_qwen_module(&stage, &prefix, module)
        }

        fn load_vision_deepstack_merger(
            &mut self,
            index: usize,
        ) -> Result<Qwen3VlVisionPatchMerger<B>, Self::Error> {
            let stage = format!("qwen-vision-deepstack-merger-{index:02}");
            let prefix = format!("model.visual.deepstack_merger_list.{index}.");
            let module =
                Qwen3VlVisionPatchMerger::new(&self.config.vision_config, true, &self.device);
            self.load_qwen_module(&stage, &prefix, module)
        }

        fn load_vision_final_merger(&mut self) -> Result<Qwen3VlVisionPatchMerger<B>, Self::Error> {
            let module =
                Qwen3VlVisionPatchMerger::new(&self.config.vision_config, false, &self.device);
            self.load_qwen_module("qwen-vision-final-merger", "model.visual.merger.", module)
        }

        fn load_text_block(&mut self, index: usize) -> Result<Qwen3VlDecoderLayer<B>, Self::Error> {
            let stage = format!("qwen-text-block-{index:02}");
            let prefix = format!("model.language_model.layers.{index}.");
            let module = Qwen3VlDecoderLayer::new(&self.config.text_config, &self.device);
            self.load_qwen_module(&stage, &prefix, module)
        }

        fn load_text_final_norm(&mut self) -> Result<RmsNorm<B>, Self::Error> {
            let module = RmsNormConfig::new(self.config.text_config.hidden_size)
                .with_epsilon(self.config.text_config.rms_norm_eps)
                .init(&self.device);
            self.load_qwen_module("qwen-text-final-norm", "model.language_model.norm.", module)
        }

        fn synchronize(&mut self) -> Result<(), Self::Error> {
            B::sync(&self.device).map_err(|error| {
                BooguError::Artifact(format!("device sync after Qwen stage failed: {error}"))
            })
        }
    }

    impl<B: Backend, R: StageShardReader> Qwen3VlCausalLmStageSource<B>
        for VerifiedBurnpackQwenStageSource<B, R>
    {
        fn load_lm_head_rows(
            &mut self,
            spec: &RowChunkSpec,
        ) -> Result<OutputProjectionRowChunk<B>, Self::Error> {
            let weight = self.load_row_tensor(spec, true)?;
            Ok(OutputProjectionRowChunk {
                spec: spec.clone(),
                weight,
            })
        }
    }

    /// Async hash-verifying Qwen stage source for browser fetch/WebGPU orchestration.
    ///
    /// Construction fetches and verifies only the two sealed metadata inventories. Every load
    /// thereafter requests one exact manifest file with the manifest's byte cap, verifies its
    /// length and SHA-256, applies it to a fresh lazy module, and drops its host bytes before the
    /// following physical shard is requested.
    pub struct VerifiedAsyncBurnpackQwenStageSource<B: Backend, R> {
        config: burn_qwen3_vl::Qwen3VlConfig,
        plan: Qwen3VlStreamingPlan,
        device: B::Device,
        entries: Vec<SerializedTensorInventory>,
        stages: BTreeMap<String, Vec<ArtifactFile>>,
        reader: R,
        max_bytes: u64,
        float_policy: BooguFloatLoadPolicy,
        quantized_policy: BooguQuantizedLoadPolicy,
    }

    impl<B: Backend, R: AsyncStageShardReader> VerifiedAsyncBurnpackQwenStageSource<B, R> {
        /// Validate a sealed release, its exact inventories, and an explicit streaming plan.
        #[allow(clippy::too_many_arguments)]
        pub async fn new(
            identity: &BooguReleaseIdentity,
            manifest: ArtifactManifest,
            inventory: BooguArtifactInventory,
            config: burn_qwen3_vl::Qwen3VlConfig,
            plan: Qwen3VlStreamingPlan,
            profile: BooguStorageProfile,
            device: B::Device,
            reader: R,
        ) -> Result<Self, BooguArtifactLoadError> {
            Self::build(
                identity,
                manifest,
                inventory,
                config,
                Some(plan),
                profile,
                device,
                reader,
            )
            .await
        }

        /// Validate a sealed release and reconstruct its exact row-slice plan from inventory.
        #[allow(clippy::too_many_arguments)]
        pub async fn new_auto(
            identity: &BooguReleaseIdentity,
            manifest: ArtifactManifest,
            inventory: BooguArtifactInventory,
            config: burn_qwen3_vl::Qwen3VlConfig,
            profile: BooguStorageProfile,
            device: B::Device,
            reader: R,
        ) -> Result<Self, BooguArtifactLoadError> {
            Self::build(
                identity, manifest, inventory, config, None, profile, device, reader,
            )
            .await
        }

        #[allow(clippy::too_many_arguments)]
        async fn build(
            identity: &BooguReleaseIdentity,
            manifest: ArtifactManifest,
            inventory: BooguArtifactInventory,
            config: burn_qwen3_vl::Qwen3VlConfig,
            plan: Option<Qwen3VlStreamingPlan>,
            profile: BooguStorageProfile,
            device: B::Device,
            mut reader: R,
        ) -> Result<Self, BooguArtifactLoadError> {
            validate_release_manifest(identity, &manifest, &inventory, profile)?;
            let max_bytes = declared_target_max_shard_bytes(&manifest)?;
            let entries = verify_inventory_contract_async(
                &manifest,
                &inventory,
                profile,
                &mut reader,
                max_bytes,
            )
            .await?;
            let plan = match plan {
                Some(plan) => plan,
                None => qwen_streaming_plan_from_entries(&config, &entries)?,
            };
            let canonical = Qwen3VlStreamingPlan::new(
                &config,
                plan.embedding_rows.clone(),
                plan.lm_head_rows.clone(),
            )
            .map_err(|error| BooguArtifactLoadError::Model(error.to_string()))?;
            if canonical != plan {
                return Err(contract(
                    "qwen",
                    "streaming descriptors differ from the canonical config/row plan",
                ));
            }
            let required_stages = plan
                .stages
                .iter()
                .map(|descriptor| qwen_streaming_stage_name(&descriptor.stage))
                .collect::<BTreeSet<_>>();
            let actual_stages = entries
                .iter()
                .filter(|entry| entry.owner == TensorOwner::Qwen3Vl && entry.included)
                .map(|entry| entry.stage.clone())
                .collect::<BTreeSet<_>>();
            if actual_stages != required_stages {
                return Err(contract(
                    "qwen",
                    format!(
                        "stored Qwen stages differ from the selected streaming plan: expected={required_stages:?}, actual={actual_stages:?}"
                    ),
                ));
            }
            let mut stages = BTreeMap::<String, Vec<ArtifactFile>>::new();
            for file in manifest
                .files
                .iter()
                .filter(|file| file.role == ArtifactFileRole::Weights)
            {
                let Some(stage) = file.component.as_ref().map(|value| value.as_str()) else {
                    continue;
                };
                if required_stages.contains(stage) {
                    if file.size > max_bytes {
                        return Err(contract(
                            stage,
                            format!(
                                "streamed Qwen shard {} is {} bytes, exceeding {max_bytes}",
                                file.path, file.size
                            ),
                        ));
                    }
                    stages
                        .entry(stage.to_owned())
                        .or_default()
                        .push(file.clone());
                }
            }
            for stage in &required_stages {
                if !stages.contains_key(stage) {
                    return Err(contract(stage, "sealed manifest omits required Qwen stage"));
                }
            }
            for files in stages.values_mut() {
                files.sort_by_key(|file| file.shard.map(|shard| shard.index).unwrap_or(0));
            }
            Ok(Self {
                config,
                plan,
                device,
                entries,
                stages,
                reader,
                max_bytes,
                float_policy: BooguFloatLoadPolicy::Preserve,
                quantized_policy: qwen_quantized_policy(profile),
            })
        }

        /// Explicit compatibility conversion for CPU backends without F16 support.
        pub fn with_float_load_policy(mut self, policy: BooguFloatLoadPolicy) -> Self {
            self.float_policy = policy;
            self
        }

        /// Override the profile-derived Q8S application policy.
        ///
        /// Browser Q8 profiles default to stage-local host dequantization because applying a
        /// block-quantized Qwen Col parameter directly would corrupt its scale layout.
        pub fn with_quantized_load_policy(mut self, policy: BooguQuantizedLoadPolicy) -> Self {
            self.quantized_policy = policy;
            self
        }

        /// Borrow the validated streaming plan.
        pub const fn plan(&self) -> &Qwen3VlStreamingPlan {
            &self.plan
        }

        /// Borrow the async transport/cache reader.
        pub const fn reader(&self) -> &R {
            &self.reader
        }

        /// Mutably borrow the async transport/cache reader.
        pub fn reader_mut(&mut self) -> &mut R {
            &mut self.reader
        }

        /// Maximum response size passed to and enforced around every read.
        pub const fn max_shard_bytes(&self) -> u64 {
            self.max_bytes
        }

        async fn load_qwen_module<M: ModuleSnapshot<B>>(
            &mut self,
            stage: &str,
            prefix: &str,
            mut module: M,
        ) -> Result<M, BooguError> {
            let files = self.stages.get(stage).cloned().ok_or_else(|| {
                BooguError::Artifact(format!("manifest has no Qwen stage {stage}"))
            })?;
            let expected = self
                .entries
                .iter()
                .filter(|entry| {
                    entry.owner == TensorOwner::Qwen3Vl
                        && entry.included
                        && entry.stage == stage
                        && entry.source_row_range.is_none()
                })
                .map(|entry| (entry.target_name.clone(), entry.clone()))
                .collect::<BTreeMap<_, _>>();
            if expected.is_empty() {
                return Err(BooguError::Artifact(format!(
                    "Qwen module stage {stage} has no exact tensor contracts"
                )));
            }
            let mut applied = BTreeSet::new();
            for file in files {
                let bytes = read_verified_async(&mut self.reader, &file, self.max_bytes).await?;
                let mut store = BurnpackStore::from_bytes(Some(Bytes::from_bytes_vec(bytes)));
                let snapshots = store.get_all_snapshots().map_err(|error| {
                    BooguError::Artifact(format!("invalid Burnpack {}: {error}", file.path))
                })?;
                if snapshots.is_empty() {
                    return Err(BooguError::Artifact(format!(
                        "Qwen stage {stage} contains an empty Burnpack {}",
                        file.path
                    )));
                }
                let mut local = Vec::with_capacity(snapshots.len());
                for (name, snapshot) in snapshots {
                    let entry = expected.get(name).ok_or_else(|| {
                        BooguError::Artifact(format!(
                            "Qwen stage {stage} contains unknown tensor {name}"
                        ))
                    })?;
                    if entry.burnpack_object.as_deref() != Some(file.path.as_str()) {
                        return Err(BooguError::Artifact(format!(
                            "Qwen stage {stage} tensor {name} is in {}, but its sealed object is {:?}",
                            file.path, entry.burnpack_object
                        )));
                    }
                    if !applied.insert(name.clone()) {
                        return Err(BooguError::Artifact(format!(
                            "Qwen stage {stage} repeats tensor {name}"
                        )));
                    }
                    if entry.stored_shape.as_deref() != Some(snapshot.shape.as_slice()) {
                        return Err(BooguError::Artifact(format!(
                            "Qwen stage {stage} tensor {name} shape differs from sealed inventory"
                        )));
                    }
                    validate_entry_dtype(entry, snapshot.dtype).map_err(|message| {
                        BooguError::Artifact(format!(
                            "Qwen stage {stage} tensor {name} dtype mismatch: {message}"
                        ))
                    })?;
                    let local_name = name.strip_prefix(prefix).ok_or_else(|| {
                        BooguError::Artifact(format!(
                            "Qwen stage {stage} tensor {name} lacks prefix {prefix:?}"
                        ))
                    })?;
                    local.push(rename_snapshot(snapshot, local_name));
                }
                let expected_applied = local
                    .iter()
                    .map(TensorSnapshot::full_path)
                    .collect::<BTreeSet<_>>();
                let result = module.apply(
                    local,
                    None,
                    load_adapter(self.float_policy, self.quantized_policy),
                    false,
                );
                validate_partial_apply(stage, &result, &expected_applied).map_err(|error| {
                    BooguError::Artifact(format!("failed to apply Qwen stage {stage}: {error}"))
                })?;
                // Burnpack bytes, store, and host snapshots leave scope before the next await.
            }
            let missing = expected
                .keys()
                .filter(|name| !applied.contains(*name))
                .take(16)
                .cloned()
                .collect::<Vec<_>>();
            if !missing.is_empty() {
                return Err(BooguError::Artifact(format!(
                    "Qwen stage {stage} is incomplete; first missing tensors: {missing:?}"
                )));
            }
            Ok(module)
        }

        async fn load_row_tensor(
            &mut self,
            spec: &RowChunkSpec,
            lm_head: bool,
        ) -> Result<Tensor<B, 2>, BooguError> {
            let stage = qwen_streaming_stage_name(&if lm_head {
                Qwen3VlStage::LmHeadRows {
                    chunk: spec.chunk_index,
                }
            } else {
                Qwen3VlStage::EmbeddingRows {
                    chunk: spec.chunk_index,
                }
            });
            let logical = if lm_head {
                "lm_head.weight"
            } else {
                "model.language_model.embed_tokens.weight"
            };
            let target = qwen_row_slice_target(logical, spec);
            let entry = self
                .entries
                .iter()
                .find(|entry| {
                    entry.owner == TensorOwner::Qwen3Vl
                        && entry.included
                        && entry.stage == stage
                        && entry.target_name == target
                        && entry.source_row_range
                            == Some([spec.row_range.start, spec.row_range.end])
                })
                .cloned()
                .ok_or_else(|| {
                    BooguError::Artifact(format!(
                        "sealed inventory omits Qwen row slice {stage}:{target}"
                    ))
                })?;
            if entry.stored_shape.as_deref() != Some([spec.rows(), spec.hidden_size].as_slice()) {
                return Err(BooguError::Artifact(format!(
                    "sealed Qwen row slice {target} has the wrong shape"
                )));
            }
            let files = self.stages.get(&stage).cloned().ok_or_else(|| {
                BooguError::Artifact(format!("manifest has no Qwen row stage {stage}"))
            })?;
            let mut found = None;
            for file in files {
                let bytes = read_verified_async(&mut self.reader, &file, self.max_bytes).await?;
                let mut store = BurnpackStore::from_bytes(Some(Bytes::from_bytes_vec(bytes)));
                let snapshots = store.get_all_snapshots().map_err(|error| {
                    BooguError::Artifact(format!("invalid Burnpack {}: {error}", file.path))
                })?;
                for (name, snapshot) in snapshots {
                    if name.as_str() != target || found.is_some() {
                        return Err(BooguError::Artifact(format!(
                            "Qwen row stage {stage} contains an unknown or duplicate tensor {name}"
                        )));
                    }
                    if entry.burnpack_object.as_deref() != Some(file.path.as_str()) {
                        return Err(BooguError::Artifact(format!(
                            "Qwen row tensor {name} is in {}, but its sealed object is {:?}",
                            file.path, entry.burnpack_object
                        )));
                    }
                    if snapshot.dtype != DType::F16
                        || snapshot.shape.as_slice() != [spec.rows(), spec.hidden_size]
                    {
                        return Err(BooguError::Artifact(format!(
                            "Qwen row tensor {name} violates its F16 shape contract"
                        )));
                    }
                    let mut data = snapshot.to_data().map_err(|error| {
                        BooguError::Artifact(format!("failed to read Qwen row {name}: {error}"))
                    })?;
                    if self.float_policy == BooguFloatLoadPolicy::AdaptToF32 {
                        data = data.convert_dtype(DType::F32);
                    }
                    let dtype = data.dtype;
                    found = Some(Tensor::<B, 2>::from_data(data, (&self.device, dtype)));
                }
                // The current plan stores one row tensor per file; any additional file would be
                // rejected as a duplicate on the following iteration.
            }
            found.ok_or_else(|| BooguError::Artifact(format!("Qwen row stage {stage} is empty")))
        }
    }

    impl<B: Backend, R: AsyncStageShardReader> AsyncQwen3VlStageSource<B>
        for VerifiedAsyncBurnpackQwenStageSource<B, R>
    {
        type Error = BooguError;

        async fn load_embedding_rows(
            &mut self,
            spec: &RowChunkSpec,
        ) -> Result<EmbeddingRowChunk<B>, Self::Error> {
            let tensor = self.load_row_tensor(spec, false).await?;
            EmbeddingRowChunk::new(spec.clone(), tensor)
                .map_err(|error| BooguError::Artifact(error.to_string()))
        }

        async fn load_vision_prelude(&mut self) -> Result<Qwen3VlVisionPrelude<B>, Self::Error> {
            let module = Qwen3VlVisionPrelude::new(self.config.vision_config.clone(), &self.device)
                .map_err(|error| BooguError::Artifact(error.to_string()))?;
            self.load_qwen_module("qwen-vision-prelude", "model.visual.", module)
                .await
        }

        async fn load_vision_block(
            &mut self,
            index: usize,
        ) -> Result<Qwen3VlVisionBlock<B>, Self::Error> {
            let stage = format!("qwen-vision-block-{index:02}");
            let prefix = format!("model.visual.blocks.{index}.");
            let module = Qwen3VlVisionBlock::new(&self.config.vision_config, &self.device);
            self.load_qwen_module(&stage, &prefix, module).await
        }

        async fn load_vision_deepstack_merger(
            &mut self,
            index: usize,
        ) -> Result<Qwen3VlVisionPatchMerger<B>, Self::Error> {
            let stage = format!("qwen-vision-deepstack-merger-{index:02}");
            let prefix = format!("model.visual.deepstack_merger_list.{index}.");
            let module =
                Qwen3VlVisionPatchMerger::new(&self.config.vision_config, true, &self.device);
            self.load_qwen_module(&stage, &prefix, module).await
        }

        async fn load_vision_final_merger(
            &mut self,
        ) -> Result<Qwen3VlVisionPatchMerger<B>, Self::Error> {
            let module =
                Qwen3VlVisionPatchMerger::new(&self.config.vision_config, false, &self.device);
            self.load_qwen_module("qwen-vision-final-merger", "model.visual.merger.", module)
                .await
        }

        async fn load_text_block(
            &mut self,
            index: usize,
        ) -> Result<Qwen3VlDecoderLayer<B>, Self::Error> {
            let stage = format!("qwen-text-block-{index:02}");
            let prefix = format!("model.language_model.layers.{index}.");
            let module = Qwen3VlDecoderLayer::new(&self.config.text_config, &self.device);
            self.load_qwen_module(&stage, &prefix, module).await
        }

        async fn load_text_final_norm(&mut self) -> Result<RmsNorm<B>, Self::Error> {
            let module = RmsNormConfig::new(self.config.text_config.hidden_size)
                .with_epsilon(self.config.text_config.rms_norm_eps)
                .init(&self.device);
            self.load_qwen_module("qwen-text-final-norm", "model.language_model.norm.", module)
                .await
        }

        async fn synchronize(&mut self) -> Result<(), Self::Error> {
            B::sync(&self.device).map_err(|error| {
                BooguError::Artifact(format!("device sync after Qwen stage failed: {error}"))
            })
        }
    }

    impl<B: Backend, R: AsyncStageShardReader> AsyncQwen3VlCausalLmStageSource<B>
        for VerifiedAsyncBurnpackQwenStageSource<B, R>
    {
        async fn load_lm_head_rows(
            &mut self,
            spec: &RowChunkSpec,
        ) -> Result<OutputProjectionRowChunk<B>, Self::Error> {
            let weight = self.load_row_tensor(spec, true).await?;
            Ok(OutputProjectionRowChunk {
                spec: spec.clone(),
                weight,
            })
        }
    }

    /// Load the complete denoiser from a sealed native Burnpack artifact directory.
    ///
    /// This is the all-resident path for native parity and high-memory deployments. Files are
    /// still read, SHA-256 verified, parsed, applied, synchronized, and dropped one physical shard
    /// at a time. Browser deployments should use [`VerifiedBurnpackStageSource`] with
    /// [`crate::StreamingBooguDenoiser`] so they never allocate all denoiser parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn load_resident_denoiser_from_directory<B: Backend>(
        identity: &BooguReleaseIdentity,
        root: impl AsRef<Path>,
        inventory: BooguArtifactInventory,
        config: BooguConfig,
        profile: BooguStorageProfile,
        float_policy: BooguFloatLoadPolicy,
        device: &B::Device,
    ) -> Result<(BooguDenoiser<B>, BooguLoadReport), BooguArtifactLoadError> {
        load_resident_denoiser_from_directory_with_policies(
            identity,
            root,
            inventory,
            config,
            profile,
            float_policy,
            BooguQuantizedLoadPolicy::Preserve,
            device,
        )
    }

    /// Load the complete denoiser with independently selected float and Q8S policies.
    ///
    /// The production Q8 path uses [`BooguFloatLoadPolicy::AdaptToF32`] and
    /// [`BooguQuantizedLoadPolicy::Preserve`], retaining accurate row-layout device Q8 while
    /// making every non-quantized parameter compatible with F32 activations.
    #[allow(clippy::too_many_arguments)]
    pub fn load_resident_denoiser_from_directory_with_policies<B: Backend>(
        identity: &BooguReleaseIdentity,
        root: impl AsRef<Path>,
        inventory: BooguArtifactInventory,
        config: BooguConfig,
        profile: BooguStorageProfile,
        float_policy: BooguFloatLoadPolicy,
        quantized_policy: BooguQuantizedLoadPolicy,
        device: &B::Device,
    ) -> Result<(BooguDenoiser<B>, BooguLoadReport), BooguArtifactLoadError> {
        load_resident_denoiser_from_directory_with_memory_policy(
            identity,
            root,
            inventory,
            config,
            profile,
            float_policy,
            quantized_policy,
            BooguResidentLoadMemoryPolicy::PreserveAllocatorCache,
            device,
        )
    }

    /// Load the complete denoiser with explicit snapshot and allocator policies.
    ///
    /// [`BooguResidentLoadMemoryPolicy::ReleaseTransientBuffersPerShard`] is intended for a
    /// strictly bounded native initialization path. It preserves the loaded model and numerical
    /// policy, but retires completed materialization/upload work and releases wholly unused backend
    /// pages before reading the next physical shard.
    #[allow(clippy::too_many_arguments)]
    pub fn load_resident_denoiser_from_directory_with_memory_policy<B: Backend>(
        identity: &BooguReleaseIdentity,
        root: impl AsRef<Path>,
        inventory: BooguArtifactInventory,
        config: BooguConfig,
        profile: BooguStorageProfile,
        float_policy: BooguFloatLoadPolicy,
        quantized_policy: BooguQuantizedLoadPolicy,
        memory_policy: BooguResidentLoadMemoryPolicy,
        device: &B::Device,
    ) -> Result<(BooguDenoiser<B>, BooguLoadReport), BooguArtifactLoadError> {
        let root = root.as_ref();
        let manifest = read_directory_manifest(root)?;
        let reader = DirectoryStageShardReader::new(root);
        load_resident_denoiser(
            identity,
            &manifest,
            inventory,
            config.clone(),
            profile,
            float_policy,
            quantized_policy,
            memory_policy,
            device,
            reader,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn load_resident_denoiser<B: Backend, R: StageShardReader>(
        identity: &BooguReleaseIdentity,
        manifest: &ArtifactManifest,
        inventory: BooguArtifactInventory,
        config: BooguConfig,
        profile: BooguStorageProfile,
        float_policy: BooguFloatLoadPolicy,
        quantized_policy: BooguQuantizedLoadPolicy,
        memory_policy: BooguResidentLoadMemoryPolicy,
        device: &B::Device,
        mut reader: R,
    ) -> Result<(BooguDenoiser<B>, BooguLoadReport), BooguArtifactLoadError> {
        validate_release_manifest(identity, manifest, &inventory, profile)?;
        verify_inventory_contract(manifest, &inventory, profile, &mut reader)?;

        let expected_specs = inventory
            .tensors()
            .iter()
            .filter(|spec| spec.owner == TensorOwner::BooguDenoiser)
            .collect::<Vec<_>>();
        let expected = expected_specs
            .iter()
            .map(|spec| spec.target_name.clone())
            .collect::<BTreeSet<_>>();
        let required_stages = expected_specs
            .iter()
            .map(|spec| spec.stage.as_str())
            .collect::<BTreeSet<_>>();
        let mut files = manifest
            .files
            .iter()
            .filter(|file| {
                file.role == ArtifactFileRole::Weights
                    && file
                        .component
                        .as_ref()
                        .is_some_and(|stage| required_stages.contains(stage.as_str()))
            })
            .cloned()
            .collect::<Vec<_>>();
        let present_stages = files
            .iter()
            .filter_map(|file| file.component.as_ref().map(|value| value.as_str()))
            .collect::<BTreeSet<_>>();
        if present_stages != required_stages {
            let missing = required_stages
                .difference(&present_stages)
                .copied()
                .collect::<Vec<_>>();
            return Err(contract(
                "denoiser",
                format!("sealed manifest omits required stages: {missing:?}"),
            ));
        }
        files.sort_by(|left, right| {
            let left_stage = left.component.as_ref().map(|value| value.as_str());
            let right_stage = right.component.as_ref().map(|value| value.as_str());
            (
                left_stage,
                left.shard.map(|shard| shard.index).unwrap_or(0),
                left.path.as_str(),
            )
                .cmp(&(
                    right_stage,
                    right.shard.map(|shard| shard.index).unwrap_or(0),
                    right.path.as_str(),
                ))
        });

        let mut model = BooguDenoiser::new(config, device)
            .map_err(|error| BooguArtifactLoadError::Model(error.to_string()))?;

        let mut applied = BTreeSet::new();
        let mut report = BooguLoadReport::default();
        for file in files {
            let stage = file
                .component
                .as_ref()
                .expect("selected weight file has a component")
                .as_str()
                .to_owned();
            let bytes =
                reader
                    .read_shard(&file)
                    .map_err(|error| BooguArtifactLoadError::Burnpack {
                        stage: stage.clone(),
                        message: error.to_string(),
                    })?;
            ArtifactVerifier::verify_bytes(&file, &bytes, IntegrityPolicy::RequireSha256).map_err(
                |error| BooguArtifactLoadError::Burnpack {
                    stage: stage.clone(),
                    message: format!("integrity verification failed for {}: {error}", file.path),
                },
            )?;
            let mut store = BurnpackStore::from_bytes(Some(Bytes::from_bytes_vec(bytes)));
            let snapshots =
                store
                    .get_all_snapshots()
                    .map_err(|error| BooguArtifactLoadError::Burnpack {
                        stage: stage.clone(),
                        message: format!("failed to parse {}: {error}", file.path),
                    })?;
            if snapshots.is_empty() {
                return Err(contract(&stage, format!("empty Burnpack {}", file.path)));
            }
            let mut shard_paths = BTreeSet::new();
            for (name, snapshot) in snapshots.iter() {
                let spec = inventory
                    .by_target(name)
                    .ok_or_else(|| contract(&stage, format!("unknown tensor {name}")))?;
                if spec.owner != TensorOwner::BooguDenoiser {
                    return Err(contract(
                        &stage,
                        format!(
                            "tensor {name} belongs to {:?}, not the denoiser",
                            spec.owner
                        ),
                    ));
                }
                if spec.stage != stage {
                    return Err(contract(
                        &stage,
                        format!("tensor {name} belongs to stage {}", spec.stage),
                    ));
                }
                if !shard_paths.insert(name.clone()) || !applied.insert(name.clone()) {
                    return Err(contract(&stage, format!("duplicate tensor {name}")));
                }
                if snapshot.shape.as_slice() != spec.target_shape.as_slice() {
                    return Err(contract(
                        &stage,
                        format!(
                            "shape mismatch for {name}: expected {:?}, found {:?}",
                            spec.target_shape, snapshot.shape
                        ),
                    ));
                }
                validate_spec_dtype(profile, spec, snapshot.dtype).map_err(|message| {
                    contract(&stage, format!("dtype mismatch for {name}: {message}"))
                })?;
            }
            let result = model.apply(
                snapshots.values().cloned().collect(),
                None,
                load_adapter(float_policy, quantized_policy),
                false,
            );
            validate_apply_result(&stage, &result, &shard_paths)?;
            report.shards += 1;
            report.tensors += shard_paths.len();
            *report.by_stage.entry(stage).or_default() += shard_paths.len();
            if memory_policy.releases_transient_buffers() {
                // `apply` queues tensor materialization and upload work. Drain it on this loader
                // thread, then release only wholly unused transient pages and wait for cleanup
                // before parsing and uploading the next physical shard.
                drop(result);
                drop(store);
                B::sync(device).map_err(|error| {
                    BooguArtifactLoadError::Model(format!(
                        "device sync after applying denoiser shard failed: {error}"
                    ))
                })?;
                B::memory_cleanup(device);
                B::sync(device).map_err(|error| {
                    BooguArtifactLoadError::Model(format!(
                        "device sync after denoiser shard allocator cleanup failed: {error}"
                    ))
                })?;
            }
            // Bytes, the Burnpack store, and host snapshots are dropped before the next file.
        }
        let missing = expected.difference(&applied).cloned().collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(BooguArtifactLoadError::Incomplete {
                count: missing.len(),
                sample: missing.into_iter().take(16).collect(),
            });
        }
        B::sync(device).map_err(|error| {
            BooguArtifactLoadError::Model(format!(
                "device sync after denoiser load failed: {error}"
            ))
        })?;
        Ok((model, report))
    }

    /// Load only the Qwen3-VL base model required for Boogu conditioning.
    ///
    /// The sealed release is still checked against the complete 750-tensor inventory, but the
    /// separate `qwen-lm-head` component is neither fetched nor allocated because Boogu consumes
    /// hidden states rather than vocabulary logits.
    #[allow(clippy::too_many_arguments)]
    pub fn load_resident_qwen_base_from_directory<B: Backend>(
        identity: &BooguReleaseIdentity,
        root: impl AsRef<Path>,
        inventory: BooguArtifactInventory,
        config: burn_qwen3_vl::Qwen3VlConfig,
        profile: BooguStorageProfile,
        float_policy: BooguFloatLoadPolicy,
        device: &B::Device,
    ) -> Result<(Qwen3VlModel<B>, BooguLoadReport), BooguArtifactLoadError> {
        let root = root.as_ref();
        let manifest = read_directory_manifest(root)?;
        validate_release_manifest(identity, &manifest, &inventory, profile)?;
        let expected = inventory
            .tensors()
            .iter()
            .filter(|spec| {
                spec.owner == TensorOwner::Qwen3Vl && spec.target_name != "lm_head.weight"
            })
            .map(|spec| spec.target_name.clone())
            .collect::<BTreeSet<_>>();
        let model = Qwen3VlModel::new(config, device)
            .map_err(|error| BooguArtifactLoadError::Model(error.to_string()))?;
        load_resident_owner_module(
            &manifest,
            &inventory,
            TensorOwner::Qwen3Vl,
            &expected,
            "model.",
            profile,
            float_policy,
            device,
            DirectoryStageShardReader::new(root),
            model,
        )
    }

    /// Load only the ordinary FLUX-compatible VAE from a sealed native artifact directory.
    ///
    /// Pass [`BooguFloatLoadPolicy::AdaptToF32`] for the released `force_upcast=true` VAE.
    #[allow(clippy::too_many_arguments)]
    pub fn load_resident_vae_from_directory<B: Backend>(
        identity: &BooguReleaseIdentity,
        root: impl AsRef<Path>,
        inventory: BooguArtifactInventory,
        config: burn_flux_vae::AutoencoderKlConfig,
        profile: BooguStorageProfile,
        float_policy: BooguFloatLoadPolicy,
        device: &B::Device,
    ) -> Result<(AutoencoderKl<B>, BooguLoadReport), BooguArtifactLoadError> {
        let root = root.as_ref();
        let manifest = read_directory_manifest(root)?;
        validate_release_manifest(identity, &manifest, &inventory, profile)?;
        let expected = inventory
            .tensors()
            .iter()
            .filter(|spec| spec.owner == TensorOwner::FluxVae)
            .map(|spec| spec.target_name.clone())
            .collect::<BTreeSet<_>>();
        let model = config
            .try_init(device)
            .map_err(|error| BooguArtifactLoadError::Model(error.to_string()))?;
        load_resident_owner_module(
            &manifest,
            &inventory,
            TensorOwner::FluxVae,
            &expected,
            "",
            profile,
            float_policy,
            device,
            DirectoryStageShardReader::new(root),
            model,
        )
    }

    /// Load only the VAE encoder and `quant_conv` tensors from a sealed directory.
    ///
    /// The returned `AutoencoderKl` is suitable for encode calls only. Its decoder remains lazy
    /// and unpopulated, and decoder Burnpacks are neither read nor allocated.
    #[allow(clippy::too_many_arguments)]
    pub fn load_vae_encoder_from_directory<B: Backend>(
        identity: &BooguReleaseIdentity,
        root: impl AsRef<Path>,
        inventory: BooguArtifactInventory,
        config: burn_flux_vae::AutoencoderKlConfig,
        profile: BooguStorageProfile,
        float_policy: BooguFloatLoadPolicy,
        device: &B::Device,
    ) -> Result<(AutoencoderKl<B>, BooguLoadReport), BooguArtifactLoadError> {
        let root = root.as_ref();
        let manifest = read_directory_manifest(root)?;
        load_vae_encoder(
            identity,
            &manifest,
            inventory,
            config,
            profile,
            float_policy,
            device,
            DirectoryStageShardReader::new(root),
        )
    }

    /// Load only the VAE encoder from a sealed manifest and bounded custom shard reader.
    #[allow(clippy::too_many_arguments)]
    pub fn load_vae_encoder<B: Backend, R: StageShardReader>(
        identity: &BooguReleaseIdentity,
        manifest: &ArtifactManifest,
        inventory: BooguArtifactInventory,
        config: burn_flux_vae::AutoencoderKlConfig,
        profile: BooguStorageProfile,
        float_policy: BooguFloatLoadPolicy,
        device: &B::Device,
        reader: R,
    ) -> Result<(AutoencoderKl<B>, BooguLoadReport), BooguArtifactLoadError> {
        load_vae_stage(
            identity,
            manifest,
            inventory,
            config,
            profile,
            float_policy,
            device,
            reader,
            "flux-vae-encoder",
        )
    }

    /// Load only the VAE decoder and `post_quant_conv` tensors from a sealed directory.
    ///
    /// The returned `AutoencoderKl` is suitable for decode calls only. Its encoder remains lazy
    /// and unpopulated, and encoder Burnpacks are neither read nor allocated.
    #[allow(clippy::too_many_arguments)]
    pub fn load_vae_decoder_from_directory<B: Backend>(
        identity: &BooguReleaseIdentity,
        root: impl AsRef<Path>,
        inventory: BooguArtifactInventory,
        config: burn_flux_vae::AutoencoderKlConfig,
        profile: BooguStorageProfile,
        float_policy: BooguFloatLoadPolicy,
        device: &B::Device,
    ) -> Result<(AutoencoderKl<B>, BooguLoadReport), BooguArtifactLoadError> {
        let root = root.as_ref();
        let manifest = read_directory_manifest(root)?;
        load_vae_decoder(
            identity,
            &manifest,
            inventory,
            config,
            profile,
            float_policy,
            device,
            DirectoryStageShardReader::new(root),
        )
    }

    /// Load only the VAE decoder from a sealed manifest and bounded custom shard reader.
    #[allow(clippy::too_many_arguments)]
    pub fn load_vae_decoder<B: Backend, R: StageShardReader>(
        identity: &BooguReleaseIdentity,
        manifest: &ArtifactManifest,
        inventory: BooguArtifactInventory,
        config: burn_flux_vae::AutoencoderKlConfig,
        profile: BooguStorageProfile,
        float_policy: BooguFloatLoadPolicy,
        device: &B::Device,
        reader: R,
    ) -> Result<(AutoencoderKl<B>, BooguLoadReport), BooguArtifactLoadError> {
        load_vae_stage(
            identity,
            manifest,
            inventory,
            config,
            profile,
            float_policy,
            device,
            reader,
            "flux-vae-decoder",
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn load_vae_stage<B: Backend, R: StageShardReader>(
        identity: &BooguReleaseIdentity,
        manifest: &ArtifactManifest,
        inventory: BooguArtifactInventory,
        config: burn_flux_vae::AutoencoderKlConfig,
        profile: BooguStorageProfile,
        float_policy: BooguFloatLoadPolicy,
        device: &B::Device,
        reader: R,
        stage: &str,
    ) -> Result<(AutoencoderKl<B>, BooguLoadReport), BooguArtifactLoadError> {
        validate_release_manifest(identity, manifest, &inventory, profile)?;
        let expected = inventory
            .tensors()
            .iter()
            .filter(|spec| spec.owner == TensorOwner::FluxVae && spec.stage == stage)
            .map(|spec| spec.target_name.clone())
            .collect::<BTreeSet<_>>();
        let model = config
            .try_init(device)
            .map_err(|error| BooguArtifactLoadError::Model(error.to_string()))?;
        load_resident_owner_module(
            manifest,
            &inventory,
            TensorOwner::FluxVae,
            &expected,
            "",
            profile,
            float_policy,
            device,
            reader,
            model,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn load_resident_owner_module<B, M, R>(
        manifest: &ArtifactManifest,
        inventory: &BooguArtifactInventory,
        owner: TensorOwner,
        expected: &BTreeSet<String>,
        prefix: &str,
        profile: BooguStorageProfile,
        float_policy: BooguFloatLoadPolicy,
        device: &B::Device,
        mut reader: R,
        mut module: M,
    ) -> Result<(M, BooguLoadReport), BooguArtifactLoadError>
    where
        B: Backend,
        M: ModuleSnapshot<B>,
        R: StageShardReader,
    {
        let serialized = verify_inventory_contract(manifest, inventory, profile, &mut reader)?;
        if expected.is_empty() {
            return Err(contract("component", format!("empty {owner:?} inventory")));
        }
        if let Some(name) = expected.iter().find(|name| !name.starts_with(prefix)) {
            return Err(contract(
                "component",
                format!("{owner:?} tensor {name} lacks prefix {prefix:?}"),
            ));
        }

        if serialized.iter().any(|entry| {
            entry.owner == owner
                && entry.included
                && entry.source_row_range.is_some()
                && expected.contains(
                    entry
                        .logical_target_name
                        .as_deref()
                        .unwrap_or(entry.target_name.as_str()),
                )
        }) {
            return Err(contract(
                "component",
                format!(
                    "resident {owner:?} loading does not concatenate row-sliced tables; use VerifiedBurnpackQwenStageSource"
                ),
            ));
        }
        let requested_entries = serialized
            .iter()
            .filter(|entry| {
                entry.owner == owner
                    && entry.included
                    && entry.source_row_range.is_none()
                    && expected.contains(
                        entry
                            .logical_target_name
                            .as_deref()
                            .unwrap_or(entry.target_name.as_str()),
                    )
            })
            .map(|entry| (entry.target_name.clone(), entry))
            .collect::<BTreeMap<_, _>>();
        let required_stages = requested_entries
            .values()
            .map(|entry| entry.stage.as_str())
            .collect::<BTreeSet<_>>();
        let mut files = manifest
            .files
            .iter()
            .filter(|file| {
                file.role == ArtifactFileRole::Weights
                    && file
                        .component
                        .as_ref()
                        .is_some_and(|stage| required_stages.contains(stage.as_str()))
            })
            .cloned()
            .collect::<Vec<_>>();
        let present_stages = files
            .iter()
            .filter_map(|file| file.component.as_ref().map(|value| value.as_str()))
            .collect::<BTreeSet<_>>();
        if present_stages != required_stages {
            return Err(contract(
                "component",
                format!(
                    "sealed manifest stages for {owner:?} differ: expected={required_stages:?}, present={present_stages:?}"
                ),
            ));
        }
        files.sort_by(|left, right| {
            (
                left.component.as_ref().map(|value| value.as_str()),
                left.shard.map(|shard| shard.index).unwrap_or(0),
                left.path.as_str(),
            )
                .cmp(&(
                    right.component.as_ref().map(|value| value.as_str()),
                    right.shard.map(|shard| shard.index).unwrap_or(0),
                    right.path.as_str(),
                ))
        });

        let mut applied = BTreeSet::new();
        let mut report = BooguLoadReport::default();
        for file in files {
            let stage = file
                .component
                .as_ref()
                .expect("selected weight file has a component")
                .as_str()
                .to_owned();
            let bytes =
                reader
                    .read_shard(&file)
                    .map_err(|error| BooguArtifactLoadError::Burnpack {
                        stage: stage.clone(),
                        message: error.to_string(),
                    })?;
            ArtifactVerifier::verify_bytes(&file, &bytes, IntegrityPolicy::RequireSha256).map_err(
                |error| BooguArtifactLoadError::Burnpack {
                    stage: stage.clone(),
                    message: format!("integrity verification failed for {}: {error}", file.path),
                },
            )?;
            let mut store = BurnpackStore::from_bytes(Some(Bytes::from_bytes_vec(bytes)));
            let snapshots =
                store
                    .get_all_snapshots()
                    .map_err(|error| BooguArtifactLoadError::Burnpack {
                        stage: stage.clone(),
                        message: format!("failed to parse {}: {error}", file.path),
                    })?;
            if snapshots.is_empty() {
                return Err(contract(&stage, format!("empty Burnpack {}", file.path)));
            }
            let mut local = Vec::with_capacity(snapshots.len());
            let mut local_paths = BTreeSet::new();
            for (name, snapshot) in snapshots.iter() {
                let entry = requested_entries
                    .get(name)
                    .ok_or_else(|| contract(&stage, format!("unknown tensor {name}")))?;
                let logical_target = entry
                    .logical_target_name
                    .as_deref()
                    .unwrap_or(entry.target_name.as_str());
                if entry.owner != owner || !expected.contains(logical_target) {
                    return Err(contract(
                        &stage,
                        format!("tensor {name} is outside the requested {owner:?} subset"),
                    ));
                }
                if entry.stage != stage {
                    return Err(contract(
                        &stage,
                        format!("tensor {name} belongs to stage {}", entry.stage),
                    ));
                }
                if !applied.insert(name.clone()) {
                    return Err(contract(&stage, format!("duplicate tensor {name}")));
                }
                if entry.stored_shape.as_deref() != Some(snapshot.shape.as_slice()) {
                    return Err(contract(
                        &stage,
                        format!(
                            "shape mismatch for {name}: expected {:?}, found {:?}",
                            entry.stored_shape, snapshot.shape
                        ),
                    ));
                }
                validate_entry_dtype(entry, snapshot.dtype).map_err(|message| {
                    contract(&stage, format!("dtype mismatch for {name}: {message}"))
                })?;
                let local_name = name.strip_prefix(prefix).ok_or_else(|| {
                    contract(
                        &stage,
                        format!("tensor {name} lacks module prefix {prefix:?}"),
                    )
                })?;
                if !local_paths.insert(local_name.to_owned()) {
                    return Err(contract(
                        &stage,
                        format!("duplicate local tensor path {local_name}"),
                    ));
                }
                local.push(rename_snapshot(snapshot, local_name));
            }
            let quantized_policy = if owner == TensorOwner::Qwen3Vl {
                qwen_quantized_policy(profile)
            } else {
                BooguQuantizedLoadPolicy::Preserve
            };
            let result = module.apply(
                local,
                None,
                load_adapter(float_policy, quantized_policy),
                false,
            );
            validate_apply_result(&stage, &result, &local_paths)?;
            report.shards += 1;
            report.tensors += local_paths.len();
            *report.by_stage.entry(stage).or_default() += local_paths.len();
        }
        let missing = expected.difference(&applied).cloned().collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(BooguArtifactLoadError::Incomplete {
                count: missing.len(),
                sample: missing.into_iter().take(16).collect(),
            });
        }
        B::sync(device).map_err(|error| {
            BooguArtifactLoadError::Model(format!(
                "device sync after {owner:?} load failed: {error}"
            ))
        })?;
        Ok((module, report))
    }

    impl<B: Backend, R: StageShardReader> StreamingStageSource<B>
        for VerifiedBurnpackStageSource<B, R>
    {
        fn load_prelude(&mut self) -> Result<BooguDenoiserPrelude<B>, BooguError> {
            let module = BooguDenoiserPrelude::new(self.config.clone(), &self.device)?;
            self.load_module("boogu-prelude", "", module)
        }

        fn load_context_refiner(
            &mut self,
            index: usize,
        ) -> Result<SingleStreamBlock<B>, BooguError> {
            let stage = format!("boogu-context-refiner-{index:02}");
            let prefix = format!("context_refiner.{index}.");
            let module = single_block(&self.config, false, &self.device);
            self.load_module(&stage, &prefix, module)
        }

        fn load_noise_refiner(&mut self, index: usize) -> Result<SingleStreamBlock<B>, BooguError> {
            let stage = format!("boogu-noise-refiner-{index:02}");
            let prefix = format!("noise_refiner.{index}.");
            let module = single_block(&self.config, true, &self.device);
            self.load_module(&stage, &prefix, module)
        }

        fn load_reference_refiner(
            &mut self,
            index: usize,
        ) -> Result<SingleStreamBlock<B>, BooguError> {
            let stage = format!("boogu-reference-refiner-{index:02}");
            let prefix = format!("ref_image_refiner.{index}.");
            let module = single_block(&self.config, true, &self.device);
            self.load_module(&stage, &prefix, module)
        }

        fn load_double_stream(&mut self, index: usize) -> Result<DoubleStreamBlock<B>, BooguError> {
            let stage = format!("boogu-dual-block-{index:02}");
            let prefix = format!("double_stream_layers.{index}.");
            let module = DoubleStreamBlock::new(
                self.config.hidden_size,
                self.config.ffn_inner_dim(),
                self.config.num_attention_heads,
                self.config.num_kv_heads,
                self.config.hidden_size.min(1024),
                self.config.norm_eps,
                &self.device,
            );
            self.load_module(&stage, &prefix, module)
        }

        fn load_single_stream(&mut self, index: usize) -> Result<SingleStreamBlock<B>, BooguError> {
            let stage = format!("boogu-single-block-{index:02}");
            let prefix = format!("single_stream_layers.{index}.");
            let module = single_block(&self.config, true, &self.device);
            self.load_module(&stage, &prefix, module)
        }

        fn load_tail(&mut self) -> Result<BooguDenoiserTail<B>, BooguError> {
            let module = BooguDenoiserTail::new(self.config.clone(), &self.device)?;
            self.load_module("boogu-tail", "", module)
        }

        fn synchronize(&mut self) -> Result<(), BooguError> {
            B::sync(&self.device)
                .map_err(|error| BooguError::Artifact(format!("device sync failed: {error}")))
        }
    }

    /// Async hash-verifying one-block-at-a-time denoiser source for browser execution.
    ///
    /// The source retains only sealed metadata and file descriptors between calls. A load awaits,
    /// verifies, applies, and releases each physical Burnpack before requesting the next one; the
    /// returned device module therefore contains exactly one semantic denoiser stage.
    pub struct VerifiedAsyncBurnpackDenoiserStageSource<B: Backend, R> {
        config: BooguConfig,
        variant: BooguVariant,
        profile: BooguStorageProfile,
        device: B::Device,
        entries: Vec<SerializedTensorInventory>,
        runtime_quantizable_targets: BTreeSet<String>,
        stages: BTreeMap<String, Vec<ArtifactFile>>,
        reader: R,
        max_bytes: u64,
        float_policy: BooguFloatLoadPolicy,
        quantized_policy: BooguQuantizedLoadPolicy,
        runtime_quantization_policy: BooguDenoiserRuntimeQuantizationPolicy,
        runtime_q8_scope: BooguRuntimeQ8Scope,
    }

    impl<B: Backend, R: AsyncStageShardReader> VerifiedAsyncBurnpackDenoiserStageSource<B, R> {
        /// Validate a sealed release and attach an asynchronous bounded shard reader.
        #[allow(clippy::too_many_arguments)]
        pub async fn new(
            identity: &BooguReleaseIdentity,
            manifest: ArtifactManifest,
            inventory: BooguArtifactInventory,
            config: BooguConfig,
            profile: BooguStorageProfile,
            device: B::Device,
            mut reader: R,
        ) -> Result<Self, BooguArtifactLoadError> {
            validate_release_manifest(identity, &manifest, &inventory, profile)?;
            let max_bytes = declared_target_max_shard_bytes(&manifest)?;
            let entries = verify_inventory_contract_async(
                &manifest,
                &inventory,
                profile,
                &mut reader,
                max_bytes,
            )
            .await?;
            let required_stages = inventory
                .tensors()
                .iter()
                .filter(|spec| spec.owner == TensorOwner::BooguDenoiser)
                .map(|spec| spec.stage.clone())
                .collect::<BTreeSet<_>>();
            let runtime_quantizable_targets = inventory
                .tensors()
                .iter()
                .filter(|spec| spec.owner == TensorOwner::BooguDenoiser && spec.quantizable)
                .map(|spec| spec.target_name.clone())
                .collect::<BTreeSet<_>>();
            let actual_stages = entries
                .iter()
                .filter(|entry| entry.owner == TensorOwner::BooguDenoiser && entry.included)
                .map(|entry| entry.stage.clone())
                .collect::<BTreeSet<_>>();
            if actual_stages != required_stages {
                return Err(contract(
                    "denoiser",
                    format!(
                        "stored denoiser stages differ from exact inventory: expected={required_stages:?}, actual={actual_stages:?}"
                    ),
                ));
            }
            let mut stages = BTreeMap::<String, Vec<ArtifactFile>>::new();
            for file in manifest
                .files
                .iter()
                .filter(|file| file.role == ArtifactFileRole::Weights)
            {
                let Some(stage) = file.component.as_ref().map(|value| value.as_str()) else {
                    continue;
                };
                if required_stages.contains(stage) {
                    if file.size > max_bytes {
                        return Err(contract(
                            stage,
                            format!(
                                "streamed denoiser shard {} is {} bytes, exceeding {max_bytes}",
                                file.path, file.size
                            ),
                        ));
                    }
                    stages
                        .entry(stage.to_owned())
                        .or_default()
                        .push(file.clone());
                }
            }
            for stage in &required_stages {
                if !stages.contains_key(stage) {
                    return Err(contract(
                        stage,
                        "sealed manifest omits required denoiser stage",
                    ));
                }
            }
            for files in stages.values_mut() {
                files.sort_by_key(|file| file.shard.map(|shard| shard.index).unwrap_or(0));
            }
            Ok(Self {
                config,
                variant: identity.variant,
                profile,
                device,
                entries,
                runtime_quantizable_targets,
                stages,
                reader,
                max_bytes,
                float_policy: BooguFloatLoadPolicy::Preserve,
                quantized_policy: BooguQuantizedLoadPolicy::Preserve,
                runtime_quantization_policy: BooguDenoiserRuntimeQuantizationPolicy::Disabled,
                runtime_q8_scope: BooguRuntimeQ8Scope::AllInventoryEligible,
            })
        }

        /// Explicit compatibility conversion for CPU backends without F16 support.
        pub fn with_float_load_policy(mut self, policy: BooguFloatLoadPolicy) -> Self {
            self.float_policy = policy;
            self
        }

        /// Select how already-quantized, verified Q8S denoiser matrices are loaded.
        ///
        /// Production WebGPU uses [`BooguQuantizedLoadPolicy::Preserve`] with F32 floating-point
        /// snapshots and activations; host dequantization remains available for diagnostics.
        pub fn with_quantized_load_policy(mut self, policy: BooguQuantizedLoadPolicy) -> Self {
            self.quantized_policy = policy;
            self
        }

        /// Select whether eligible verified float denoiser matrices are quantized at runtime.
        ///
        /// The default is [`BooguDenoiserRuntimeQuantizationPolicy::Disabled`]. Enabling Q8S
        /// conversion does not alter artifact identity and remains constrained by the sealed
        /// inventory plus [`Self::with_runtime_q8_scope`].
        pub fn with_runtime_quantization_policy(
            mut self,
            policy: BooguDenoiserRuntimeQuantizationPolicy,
        ) -> Self {
            self.runtime_quantization_policy = policy;
            self
        }

        /// Select the closed subset of inventory-eligible matrices quantized at runtime.
        pub fn with_runtime_q8_scope(mut self, scope: BooguRuntimeQ8Scope) -> Self {
            self.runtime_q8_scope = scope;
            self
        }

        /// Borrow the asynchronous transport/cache reader.
        pub const fn reader(&self) -> &R {
            &self.reader
        }

        /// Mutably borrow the asynchronous transport/cache reader.
        pub fn reader_mut(&mut self) -> &mut R {
            &mut self.reader
        }

        /// Maximum response size passed to and enforced around every read.
        pub const fn max_shard_bytes(&self) -> u64 {
            self.max_bytes
        }

        async fn load_module<M: ModuleSnapshot<B>>(
            &mut self,
            stage: &str,
            prefix: &str,
            mut module: M,
        ) -> Result<M, BooguError> {
            validate_runtime_denoiser_quantization_policy(
                self.profile,
                self.float_policy,
                self.quantized_policy,
                self.runtime_quantization_policy,
                self.runtime_q8_scope,
                self.variant,
                stage,
            )?;
            let files = self.stages.get(stage).cloned().ok_or_else(|| {
                BooguError::Artifact(format!("manifest has no denoiser stage {stage}"))
            })?;
            let expected = self
                .entries
                .iter()
                .filter(|entry| {
                    entry.owner == TensorOwner::BooguDenoiser
                        && entry.included
                        && entry.stage == stage
                        && entry.source_row_range.is_none()
                })
                .map(|entry| (entry.target_name.clone(), entry.clone()))
                .collect::<BTreeMap<_, _>>();
            if expected.is_empty() {
                return Err(BooguError::Artifact(format!(
                    "denoiser stage {stage} has no exact tensor contracts"
                )));
            }
            if let Some(name) = expected.keys().find(|name| !name.starts_with(prefix)) {
                return Err(BooguError::Artifact(format!(
                    "stage {stage} tensor {name} does not start with {prefix:?}"
                )));
            }
            let mut applied = BTreeSet::new();
            for file in files {
                let bytes = read_verified_async(&mut self.reader, &file, self.max_bytes).await?;
                let mut store = BurnpackStore::from_bytes(Some(Bytes::from_bytes_vec(bytes)));
                let snapshots = store.get_all_snapshots().map_err(|error| {
                    BooguError::Artifact(format!("invalid Burnpack {}: {error}", file.path))
                })?;
                if snapshots.is_empty() {
                    return Err(BooguError::Artifact(format!(
                        "stage {stage} contains an empty Burnpack {}",
                        file.path
                    )));
                }
                let mut local = Vec::with_capacity(snapshots.len());
                let mut runtime_quantizable_paths = BTreeSet::new();
                for (name, snapshot) in snapshots {
                    let entry = expected.get(name).ok_or_else(|| {
                        BooguError::Artifact(format!(
                            "stage {stage} contains unknown tensor {name}"
                        ))
                    })?;
                    if entry.burnpack_object.as_deref() != Some(file.path.as_str()) {
                        return Err(BooguError::Artifact(format!(
                            "stage {stage} tensor {name} is in {}, but its sealed object is {:?}",
                            file.path, entry.burnpack_object
                        )));
                    }
                    if !applied.insert(name.clone()) {
                        return Err(BooguError::Artifact(format!(
                            "stage {stage} repeats tensor {name}"
                        )));
                    }
                    if entry.stored_shape.as_deref() != Some(snapshot.shape.as_slice()) {
                        return Err(BooguError::Artifact(format!(
                            "stage {stage} tensor {name} shape differs from sealed inventory"
                        )));
                    }
                    validate_entry_dtype(entry, snapshot.dtype).map_err(|message| {
                        BooguError::Artifact(format!(
                            "stage {stage} tensor {name} dtype mismatch: {message}"
                        ))
                    })?;
                    let local_name = name.strip_prefix(prefix).ok_or_else(|| {
                        BooguError::Artifact(format!(
                            "stage {stage} tensor {name} does not start with {prefix:?}"
                        ))
                    })?;
                    if self.runtime_quantizable_targets.contains(name)
                        && self.runtime_q8_scope.quantizes_target(name)
                    {
                        runtime_quantizable_paths.insert(local_name.to_owned());
                    }
                    local.push(rename_snapshot(snapshot, local_name));
                }
                let expected_applied = local
                    .iter()
                    .map(TensorSnapshot::full_path)
                    .collect::<BTreeSet<_>>();
                let result = module.apply(
                    local,
                    None,
                    denoiser_load_adapter(
                        self.float_policy,
                        self.quantized_policy,
                        self.runtime_quantization_policy,
                        runtime_quantizable_paths,
                    ),
                    false,
                );
                validate_partial_apply(stage, &result, &expected_applied).map_err(|error| {
                    BooguError::Artifact(format!("failed to apply stage {stage}: {error}"))
                })?;
            }
            let missing = expected
                .keys()
                .filter(|name| !applied.contains(*name))
                .take(16)
                .cloned()
                .collect::<Vec<_>>();
            if !missing.is_empty() {
                return Err(BooguError::Artifact(format!(
                    "stage {stage} is incomplete; first missing tensors: {missing:?}"
                )));
            }
            Ok(module)
        }
    }

    impl<B: Backend, R: AsyncStageShardReader> AsyncBooguDenoiserStageSource<B>
        for VerifiedAsyncBurnpackDenoiserStageSource<B, R>
    {
        async fn load_prelude(&mut self) -> Result<BooguDenoiserPrelude<B>, BooguError> {
            let module = BooguDenoiserPrelude::new(self.config.clone(), &self.device)?;
            self.load_module("boogu-prelude", "", module).await
        }

        async fn load_context_refiner(
            &mut self,
            index: usize,
        ) -> Result<SingleStreamBlock<B>, BooguError> {
            let stage = format!("boogu-context-refiner-{index:02}");
            let prefix = format!("context_refiner.{index}.");
            let module = single_block(&self.config, false, &self.device);
            self.load_module(&stage, &prefix, module).await
        }

        async fn load_noise_refiner(
            &mut self,
            index: usize,
        ) -> Result<SingleStreamBlock<B>, BooguError> {
            let stage = format!("boogu-noise-refiner-{index:02}");
            let prefix = format!("noise_refiner.{index}.");
            let module = single_block(&self.config, true, &self.device);
            self.load_module(&stage, &prefix, module).await
        }

        async fn load_reference_refiner(
            &mut self,
            index: usize,
        ) -> Result<SingleStreamBlock<B>, BooguError> {
            let stage = format!("boogu-reference-refiner-{index:02}");
            let prefix = format!("ref_image_refiner.{index}.");
            let module = single_block(&self.config, true, &self.device);
            self.load_module(&stage, &prefix, module).await
        }

        async fn load_double_stream(
            &mut self,
            index: usize,
        ) -> Result<DoubleStreamBlock<B>, BooguError> {
            let stage = format!("boogu-dual-block-{index:02}");
            let prefix = format!("double_stream_layers.{index}.");
            let module = DoubleStreamBlock::new(
                self.config.hidden_size,
                self.config.ffn_inner_dim(),
                self.config.num_attention_heads,
                self.config.num_kv_heads,
                self.config.hidden_size.min(1024),
                self.config.norm_eps,
                &self.device,
            );
            self.load_module(&stage, &prefix, module).await
        }

        async fn load_single_stream(
            &mut self,
            index: usize,
        ) -> Result<SingleStreamBlock<B>, BooguError> {
            let stage = format!("boogu-single-block-{index:02}");
            let prefix = format!("single_stream_layers.{index}.");
            let module = single_block(&self.config, true, &self.device);
            self.load_module(&stage, &prefix, module).await
        }

        async fn load_tail(&mut self) -> Result<BooguDenoiserTail<B>, BooguError> {
            let module = BooguDenoiserTail::new(self.config.clone(), &self.device)?;
            self.load_module("boogu-tail", "", module).await
        }

        async fn synchronize(&mut self) -> Result<(), BooguError> {
            B::sync(&self.device)
                .map_err(|error| BooguError::Artifact(format!("device sync failed: {error}")))
        }
    }

    /// Sequential semantic-shard loader shared by native and browser orchestration.
    ///
    /// Each call retains only the supplied shard bytes plus the device tensors. Callers verify
    /// manifest hashes before calling this type; this layer additionally validates every key,
    /// stage, shape, dtype, duplicate, and Burn module application result.
    pub struct BooguBurnpackLoader<B: Backend> {
        models: BooguModels<B>,
        inventory: BooguArtifactInventory,
        profile: BooguStorageProfile,
        applied: BTreeSet<String>,
        report: BooguLoadReport,
        poisoned: bool,
        float_policy: BooguFloatLoadPolicy,
        qwen_quantized_policy: BooguQuantizedLoadPolicy,
        denoiser_quantized_policy: BooguQuantizedLoadPolicy,
    }

    impl<B: Backend> BooguBurnpackLoader<B> {
        /// Initialize all three modules and a strict loader for a pinned release.
        pub fn new(
            identity: &BooguReleaseIdentity,
            qwen_config: &burn_qwen3_vl::Qwen3VlConfig,
            denoiser_config: &BooguConfig,
            vae_config: &burn_flux_vae::AutoencoderKlConfig,
            profile: BooguStorageProfile,
            device: &B::Device,
        ) -> Result<Self, BooguArtifactLoadError> {
            identity
                .validate()
                .map_err(|error| BooguArtifactLoadError::Identity(error.to_string()))?;
            let inventory = BooguArtifactInventory::new(qwen_config, denoiser_config, vae_config)
                .map_err(|error| BooguArtifactLoadError::Model(error.to_string()))?;
            let models = BooguModels {
                qwen: Qwen3VlForConditionalGeneration::new(qwen_config.clone(), device)
                    .map_err(|error| BooguArtifactLoadError::Model(error.to_string()))?,
                denoiser: BooguDenoiser::new(denoiser_config.clone(), device)
                    .map_err(|error| BooguArtifactLoadError::Model(error.to_string()))?,
                vae: vae_config
                    .try_init(device)
                    .map_err(|error| BooguArtifactLoadError::Model(error.to_string()))?,
            };
            Ok(Self::from_models(models, inventory, profile))
        }

        /// Attach validation state to modules initialized by an application-specific allocator.
        pub fn from_models(
            models: BooguModels<B>,
            inventory: BooguArtifactInventory,
            profile: BooguStorageProfile,
        ) -> Self {
            Self {
                models,
                inventory,
                profile,
                applied: BTreeSet::new(),
                report: BooguLoadReport::default(),
                poisoned: false,
                float_policy: BooguFloatLoadPolicy::Preserve,
                qwen_quantized_policy: qwen_quantized_policy(profile),
                denoiser_quantized_policy: BooguQuantizedLoadPolicy::Preserve,
            }
        }

        /// Select an explicit float conversion policy before applying any shard.
        pub fn with_float_load_policy(mut self, policy: BooguFloatLoadPolicy) -> Self {
            self.float_policy = policy;
            self
        }

        /// Override the Qwen-specific quantized snapshot policy.
        pub fn with_qwen_quantized_load_policy(mut self, policy: BooguQuantizedLoadPolicy) -> Self {
            self.qwen_quantized_policy = policy;
            self
        }

        /// Override the denoiser-specific quantized snapshot policy.
        pub fn with_denoiser_quantized_load_policy(
            mut self,
            policy: BooguQuantizedLoadPolicy,
        ) -> Self {
            self.denoiser_quantized_policy = policy;
            self
        }

        /// Apply one already hash-verified Burnpack shard for `stage`.
        pub fn apply_shard(
            &mut self,
            stage: &str,
            bytes: Vec<u8>,
        ) -> Result<usize, BooguArtifactLoadError> {
            if self.poisoned {
                return Err(BooguArtifactLoadError::Poisoned);
            }
            match self.apply_shard_inner(stage, bytes) {
                Ok(count) => Ok(count),
                Err(error) => {
                    self.poisoned = true;
                    Err(error)
                }
            }
        }

        fn apply_shard_inner(
            &mut self,
            stage: &str,
            bytes: Vec<u8>,
        ) -> Result<usize, BooguArtifactLoadError> {
            if !self.inventory.stages().contains(stage) {
                return Err(contract(stage, "unknown semantic stage"));
            }
            let mut store = BurnpackStore::from_bytes(Some(Bytes::from_bytes_vec(bytes)));
            let snapshots =
                store
                    .get_all_snapshots()
                    .map_err(|error| BooguArtifactLoadError::Burnpack {
                        stage: stage.into(),
                        message: error.to_string(),
                    })?;
            if snapshots.is_empty() {
                return Err(contract(stage, "empty shard"));
            }
            let mut owner = None;
            for (name, snapshot) in snapshots.iter() {
                let spec = self
                    .inventory
                    .by_target(name)
                    .ok_or_else(|| contract(stage, format!("unknown tensor {name}")))?;
                if spec.stage != stage {
                    return Err(contract(
                        stage,
                        format!("tensor {name} belongs to stage {}", spec.stage),
                    ));
                }
                if self.applied.contains(name) {
                    return Err(contract(stage, format!("duplicate tensor {name}")));
                }
                if owner
                    .replace(spec.owner)
                    .is_some_and(|value| value != spec.owner)
                {
                    return Err(contract(stage, "one shard spans multiple model owners"));
                }
                if snapshot.shape.as_slice() != spec.target_shape {
                    return Err(contract(
                        stage,
                        format!(
                            "shape mismatch for {name}: expected {:?}, found {:?}",
                            spec.target_shape, snapshot.shape
                        ),
                    ));
                }
                validate_spec_dtype(self.profile, spec, snapshot.dtype).map_err(|message| {
                    contract(stage, format!("dtype mismatch for {name}: {message}"))
                })?;
            }
            let expected_applied = snapshots.keys().cloned().collect::<BTreeSet<_>>();
            let snapshots = snapshots.values().cloned().collect::<Vec<_>>();
            let owner = owner.expect("non-empty snapshots established owner");
            let quantized_policy = match owner {
                TensorOwner::Qwen3Vl => self.qwen_quantized_policy,
                TensorOwner::BooguDenoiser => self.denoiser_quantized_policy,
                TensorOwner::FluxVae => BooguQuantizedLoadPolicy::Preserve,
            };
            let adapter = load_adapter(self.float_policy, quantized_policy);
            let result = match owner {
                TensorOwner::Qwen3Vl => self.models.qwen.apply(snapshots, None, adapter, false),
                TensorOwner::BooguDenoiser => {
                    self.models.denoiser.apply(snapshots, None, adapter, false)
                }
                TensorOwner::FluxVae => self.models.vae.apply(snapshots, None, adapter, false),
            };
            validate_apply_result(stage, &result, &expected_applied)?;
            for name in &result.applied {
                if !self.applied.insert(name.clone()) {
                    return Err(contract(stage, format!("duplicate tensor {name}")));
                }
            }
            let count = result.applied.len();
            self.report.shards += 1;
            self.report.tensors += count;
            *self.report.by_stage.entry(stage.into()).or_default() += count;
            Ok(count)
        }

        /// Current progress without exposing partially loaded modules.
        pub fn report(&self) -> &BooguLoadReport {
            &self.report
        }

        /// Require the full inventory and release the populated models.
        pub fn finish(self) -> Result<(BooguModels<B>, BooguLoadReport), BooguArtifactLoadError> {
            if self.poisoned {
                return Err(BooguArtifactLoadError::Poisoned);
            }
            let expected = self
                .inventory
                .tensors()
                .iter()
                .map(|spec| spec.target_name.clone())
                .collect::<BTreeSet<_>>();
            let missing = expected
                .difference(&self.applied)
                .cloned()
                .collect::<Vec<_>>();
            if !missing.is_empty() {
                return Err(BooguArtifactLoadError::Incomplete {
                    count: missing.len(),
                    sample: missing.into_iter().take(16).collect(),
                });
            }
            Ok((self.models, self.report))
        }
    }

    fn read_directory_manifest(root: &Path) -> Result<ArtifactManifest, BooguArtifactLoadError> {
        let manifest_path = root.join("manifest.json");
        serde_json::from_slice(&fs::read(&manifest_path).map_err(|error| {
            BooguArtifactLoadError::Identity(format!(
                "failed to read {}: {error}",
                manifest_path.display()
            ))
        })?)
        .map_err(|error| {
            BooguArtifactLoadError::Identity(format!(
                "failed to parse {}: {error}",
                manifest_path.display()
            ))
        })
    }

    pub(super) fn verify_inventory_contract<R: StageShardReader>(
        manifest: &ArtifactManifest,
        inventory: &BooguArtifactInventory,
        profile: BooguStorageProfile,
        reader: &mut R,
    ) -> Result<Vec<SerializedTensorInventory>, BooguArtifactLoadError> {
        let schema = tensor_inventory_schema(manifest)?;
        let inventory_file = manifest
            .files
            .iter()
            .find(|file| file.path.as_str() == "metadata/tensor-inventory.json")
            .ok_or_else(|| {
                BooguArtifactLoadError::Identity(
                    "manifest omits metadata/tensor-inventory.json".into(),
                )
            })?;
        let bytes = reader.read_shard(inventory_file).map_err(|error| {
            BooguArtifactLoadError::Burnpack {
                stage: "tensor-inventory".into(),
                message: error.to_string(),
            }
        })?;
        ArtifactVerifier::verify_bytes(inventory_file, &bytes, IntegrityPolicy::RequireSha256)
            .map_err(|error| BooguArtifactLoadError::Burnpack {
                stage: "tensor-inventory".into(),
                message: format!("integrity verification failed: {error}"),
            })?;
        let entries: Vec<SerializedTensorInventory> =
            serde_json::from_slice(&bytes).map_err(|error| BooguArtifactLoadError::Contract {
                stage: "tensor-inventory".into(),
                message: format!("invalid exact tensor inventory JSON: {error}"),
            })?;
        let vae_encoder_f32_diagnostic = manifest
            .metadata
            .keys()
            .any(|key| key.starts_with("diagnostic_"));
        if vae_encoder_f32_diagnostic {
            validate_edit_turbo_1k5_vae_encoder_f32_diagnostic_manifest(manifest)
                .map_err(|error| contract("tensor-inventory", error.to_string()))?;
        }
        verify_source_file_contract(manifest, &entries, reader)?;
        let weight_files = manifest
            .files
            .iter()
            .filter(|file| file.role == ArtifactFileRole::Weights)
            .map(|file| (file.path.as_str(), file))
            .collect::<BTreeMap<_, _>>();
        let expected_specs = manifest_inventory_specs(manifest, inventory);
        let mut targets = BTreeSet::new();
        let mut sources = BTreeMap::<String, Vec<&SerializedTensorInventory>>::new();
        let mut referenced_objects = BTreeSet::new();
        let mut included_count = 0_usize;
        let mut omitted_count = 0_usize;
        for entry in &entries {
            if entry.included && !targets.insert(entry.target_name.clone()) {
                return Err(contract(
                    "tensor-inventory",
                    format!("duplicate target {}", entry.target_name),
                ));
            }
            let qualified_source = format!("{}:{}", entry.component, entry.source_name);
            let spec = inventory
                .by_source(&entry.component, &entry.source_name)
                .ok_or_else(|| {
                    contract(
                        "tensor-inventory",
                        format!("unknown source {qualified_source}"),
                    )
                })?;
            let logical_target = entry
                .logical_target_name
                .as_deref()
                .unwrap_or(entry.target_name.as_str());
            let source_element_bytes = match spec.source_dtype {
                SourceDType::Bf16 => 2_u64,
                SourceDType::F32 => 4_u64,
            };
            let full_source_bytes = spec
                .source_shape
                .iter()
                .try_fold(1_u64, |size, &dimension| size.checked_mul(dimension as u64))
                .and_then(|size| size.checked_mul(source_element_bytes))
                .ok_or_else(|| contract("tensor-inventory", "source byte size overflow"))?;
            if entry.source_name != spec.source_name
                || entry.owner != spec.owner
                || entry.component != spec.source_component
                || entry.transform != spec.transform
                || entry.source_dtype != spec.source_dtype.safetensors_name()
                || entry.source_shape != spec.source_shape
                || logical_target != spec.target_name
                || !entry
                    .source_file
                    .starts_with(&format!("{}/", spec.source_component))
                || entry
                    .source_offset
                    .checked_add(entry.source_bytes)
                    .is_none()
            {
                return Err(contract(
                    "tensor-inventory",
                    format!("semantic contract mismatch for {}", entry.target_name),
                ));
            }
            if let Some([start, end]) = entry.source_row_range {
                let valid_row_source = spec.owner == TensorOwner::Qwen3Vl
                    && matches!(
                        spec.source_name.as_str(),
                        "model.language_model.embed_tokens.weight" | "lm_head.weight"
                    )
                    && spec.source_shape.len() == 2
                    && spec.transform == TensorTransform::Identity;
                let expected_shape = vec![end.saturating_sub(start), spec.source_shape[1]];
                let expected_bytes = u64::try_from(expected_shape.iter().product::<usize>())
                    .ok()
                    .and_then(|elements| elements.checked_mul(source_element_bytes));
                if !entry.included
                    || !valid_row_source
                    || start >= end
                    || end > spec.source_shape[0]
                    || entry.stored_shape.as_ref() != Some(&expected_shape)
                    || entry.stored_dtype.as_deref() != Some("f16")
                    || entry.quantized
                    || Some(entry.source_bytes) != expected_bytes
                {
                    return Err(contract(
                        "tensor-inventory",
                        format!("invalid row-slice contract for {qualified_source}"),
                    ));
                }
            } else if entry.included {
                let (expected_stored_dtype, expected_quantized) =
                    expected_spec_storage(profile, spec, vae_encoder_f32_diagnostic);
                let expected_stage = if schema == 1 && spec.owner == TensorOwner::Qwen3Vl {
                    legacy_qwen_stage(&spec.source_name)
                } else {
                    spec.stage.clone()
                };
                if entry.target_name != spec.target_name
                    || entry.stage != expected_stage
                    || entry.stored_dtype.as_deref() != Some(expected_stored_dtype)
                    || entry.stored_shape.as_ref() != Some(&spec.target_shape)
                    || entry.quantized != expected_quantized
                    || entry.source_bytes != full_source_bytes
                {
                    return Err(contract(
                        "tensor-inventory",
                        format!("stored tensor contract mismatch for {qualified_source}"),
                    ));
                }
            } else if spec.owner != TensorOwner::Qwen3Vl
                || spec.source_name != "lm_head.weight"
                || entry.target_name != spec.target_name
                || entry.stage != spec.stage
                || entry.source_bytes != full_source_bytes
                || entry.stored_dtype.is_some()
                || entry.stored_shape.is_some()
                || entry.stored_sha256.is_some()
                || entry.burnpack_object.is_some()
                || entry.quantized
            {
                return Err(contract(
                    "tensor-inventory",
                    format!("only the validated Qwen LM head may be omitted: {qualified_source}"),
                ));
            }

            if entry.included {
                included_count += 1;
                let object_path = entry.burnpack_object.as_deref().ok_or_else(|| {
                    contract(
                        "tensor-inventory",
                        format!("{} omits its Burnpack object", entry.target_name),
                    )
                })?;
                let object = weight_files.get(object_path).ok_or_else(|| {
                    contract(
                        "tensor-inventory",
                        format!(
                            "{} references undeclared Burnpack {object_path}",
                            entry.target_name
                        ),
                    )
                })?;
                referenced_objects.insert(object_path);
                if entry.stored_sha256.is_none() {
                    return Err(contract(
                        "tensor-inventory",
                        format!("{} omits its stored payload digest", entry.target_name),
                    ));
                }
                if object.component.as_ref().map(|value| value.as_str())
                    != Some(spec.stage.as_str())
                {
                    let expected_stage = entry.stage.as_str();
                    if object.component.as_ref().map(|value| value.as_str()) != Some(expected_stage)
                    {
                        return Err(contract(
                            "tensor-inventory",
                            format!(
                                "{} references Burnpack in the wrong semantic stage",
                                entry.target_name
                            ),
                        ));
                    }
                }
            } else {
                omitted_count += 1;
            }
            sources.entry(qualified_source).or_default().push(entry);
        }

        let expected_sources = expected_specs
            .iter()
            .map(|spec| spec.qualified_source_name())
            .collect::<BTreeSet<_>>();
        if sources.keys().cloned().collect::<BTreeSet<_>>() != expected_sources {
            return Err(contract(
                "tensor-inventory",
                "source set differs from the compiled exact inventory",
            ));
        }
        if referenced_objects != weight_files.keys().copied().collect::<BTreeSet<_>>() {
            return Err(contract(
                "tensor-inventory",
                "sealed weight object set contains an unreferenced or missing Burnpack",
            ));
        }
        for spec in &expected_specs {
            let qualified = spec.qualified_source_name();
            let group = sources
                .get(&qualified)
                .expect("source set equality checked above");
            let has_slices = group.iter().any(|entry| entry.source_row_range.is_some());
            if !has_slices {
                if group.len() != 1 {
                    return Err(contract(
                        "tensor-inventory",
                        format!("non-sliced source {qualified} has {} entries", group.len()),
                    ));
                }
                continue;
            }
            if group.iter().any(|entry| entry.source_row_range.is_none()) {
                return Err(contract(
                    "tensor-inventory",
                    format!("source {qualified} mixes full and row-sliced entries"),
                ));
            }
            let mut slices = group.clone();
            slices.sort_by_key(|entry| entry.source_row_range.expect("checked row range")[0]);
            let row_bytes = u64::try_from(spec.source_shape[1])
                .ok()
                .and_then(|hidden| {
                    hidden.checked_mul(match spec.source_dtype {
                        SourceDType::Bf16 => 2,
                        SourceDType::F32 => 4,
                    })
                })
                .ok_or_else(|| contract("tensor-inventory", "row byte size overflow"))?;
            let first_range = slices[0].source_row_range.expect("checked row range");
            let first_byte_offset = u64::try_from(first_range[0])
                .ok()
                .and_then(|row| row.checked_mul(row_bytes))
                .ok_or_else(|| contract("tensor-inventory", "row source offset overflow"))?;
            let base_offset = slices[0]
                .source_offset
                .checked_sub(first_byte_offset)
                .ok_or_else(|| contract("tensor-inventory", "row source offset underflow"))?;
            let mut cursor = 0_usize;
            for (chunk_index, entry) in slices.iter().enumerate() {
                let [start, end] = entry.source_row_range.expect("checked row range");
                let chunk = RowChunkSpec {
                    chunk_index,
                    row_range: start..end,
                    total_rows: spec.source_shape[0],
                    hidden_size: spec.source_shape[1],
                    element_bytes: 2,
                };
                let lm_head = spec.source_name == "lm_head.weight";
                let expected_stage = qwen_streaming_stage_name(&if lm_head {
                    Qwen3VlStage::LmHeadRows { chunk: chunk_index }
                } else {
                    Qwen3VlStage::EmbeddingRows { chunk: chunk_index }
                });
                let expected_target = qwen_row_slice_target(&spec.target_name, &chunk);
                let relative_offset = u64::try_from(start)
                    .ok()
                    .and_then(|row| row.checked_mul(row_bytes))
                    .ok_or_else(|| contract("tensor-inventory", "row source offset overflow"))?;
                let expected_offset = base_offset
                    .checked_add(relative_offset)
                    .ok_or_else(|| contract("tensor-inventory", "row source offset overflow"))?;
                if start != cursor
                    || entry.stage != expected_stage
                    || entry.target_name != expected_target
                    || entry.source_offset != expected_offset
                {
                    return Err(contract(
                        "tensor-inventory",
                        format!("row slices are not canonical/contiguous for {qualified}"),
                    ));
                }
                cursor = end;
            }
            if cursor != spec.source_shape[0] {
                return Err(contract(
                    "tensor-inventory",
                    format!("row slices do not completely cover {qualified}"),
                ));
            }
        }
        if schema == 2 {
            validate_declared_count(manifest, "stored_tensor_count", included_count)?;
            validate_declared_count(manifest, "omitted_tensor_count", omitted_count)?;
            if let Some(embedding) = sources.get("mllm:model.language_model.embed_tokens.weight") {
                let row_chunks = embedding
                    .iter()
                    .filter(|entry| entry.source_row_range.is_some())
                    .count();
                if row_chunks > 0 {
                    validate_declared_count(manifest, "qwen_embedding_row_chunks", row_chunks)?;
                }
            }
            if omitted_count == 1
                && manifest.metadata.get("qwen_lm_head").map(String::as_str)
                    != Some("omitted-base-model")
            {
                return Err(contract(
                    "tensor-inventory",
                    "omitted Qwen LM head lacks explicit manifest policy",
                ));
            }
        } else if entries.len() != expected_specs.len() || omitted_count != 0 {
            return Err(contract(
                "tensor-inventory",
                "legacy schema must contain exactly one stored entry per source tensor",
            ));
        }
        Ok(entries)
    }

    fn verify_source_file_contract<R: StageShardReader>(
        manifest: &ArtifactManifest,
        entries: &[SerializedTensorInventory],
        reader: &mut R,
    ) -> Result<(), BooguArtifactLoadError> {
        let file = manifest
            .files
            .iter()
            .find(|file| file.path.as_str() == "metadata/source-files.json")
            .ok_or_else(|| {
                BooguArtifactLoadError::Identity("manifest omits metadata/source-files.json".into())
            })?;
        if file.role != ArtifactFileRole::Metadata {
            return Err(BooguArtifactLoadError::Identity(
                "metadata/source-files.json has the wrong artifact role".into(),
            ));
        }
        let bytes = reader
            .read_shard(file)
            .map_err(|error| BooguArtifactLoadError::Burnpack {
                stage: "source-files".into(),
                message: error.to_string(),
            })?;
        ArtifactVerifier::verify_bytes(file, &bytes, IntegrityPolicy::RequireSha256).map_err(
            |error| BooguArtifactLoadError::Burnpack {
                stage: "source-files".into(),
                message: format!("integrity verification failed: {error}"),
            },
        )?;
        let records: Vec<SerializedSourceFile> =
            serde_json::from_slice(&bytes).map_err(|error| {
                contract(
                    "source-files",
                    format!("invalid exact source-file inventory JSON: {error}"),
                )
            })?;
        let mut by_path = BTreeMap::new();
        for record in &records {
            if record.size == 0 || by_path.insert(record.path.as_str(), record).is_some() {
                return Err(contract(
                    "source-files",
                    format!("duplicate or empty source-file record {}", record.path),
                ));
            }
            let _source_digest = record.sha256;
        }
        let expected = entries
            .iter()
            .map(|entry| entry.source_file.as_str())
            .collect::<BTreeSet<_>>();
        if by_path.keys().copied().collect::<BTreeSet<_>>() != expected {
            return Err(contract(
                "source-files",
                "source-file record set differs from tensor inventory references",
            ));
        }
        for entry in entries {
            let record = by_path
                .get(entry.source_file.as_str())
                .expect("source-file set equality checked");
            if entry
                .source_offset
                .checked_add(entry.source_bytes)
                .is_none_or(|end| end > record.size)
            {
                return Err(contract(
                    "source-files",
                    format!(
                        "tensor {} range exceeds source file {}",
                        entry.source_name, entry.source_file
                    ),
                ));
            }
        }
        Ok(())
    }

    fn tensor_inventory_schema(manifest: &ArtifactManifest) -> Result<u32, BooguArtifactLoadError> {
        let Some(value) = manifest.metadata.get("tensor_inventory_schema") else {
            return Ok(1);
        };
        let schema = value.parse::<u32>().map_err(|error| {
            BooguArtifactLoadError::Identity(format!(
                "manifest tensor_inventory_schema is invalid: {error}"
            ))
        })?;
        if !matches!(schema, 1 | 2) {
            return Err(BooguArtifactLoadError::Identity(format!(
                "unsupported tensor_inventory_schema {schema}"
            )));
        }
        Ok(schema)
    }

    fn validate_declared_count(
        manifest: &ArtifactManifest,
        key: &str,
        actual: usize,
    ) -> Result<(), BooguArtifactLoadError> {
        let declared = manifest
            .metadata
            .get(key)
            .ok_or_else(|| BooguArtifactLoadError::Identity(format!("manifest omits {key}")))?
            .parse::<usize>()
            .map_err(|error| {
                BooguArtifactLoadError::Identity(format!("manifest {key} is invalid: {error}"))
            })?;
        if declared != actual {
            return Err(contract(
                "tensor-inventory",
                format!("manifest declares {declared} {key}, inventory has {actual}"),
            ));
        }
        Ok(())
    }

    pub(crate) fn validate_release_manifest(
        identity: &BooguReleaseIdentity,
        manifest: &ArtifactManifest,
        inventory: &BooguArtifactInventory,
        profile: BooguStorageProfile,
    ) -> Result<(), BooguArtifactLoadError> {
        identity
            .validate()
            .map_err(|error| BooguArtifactLoadError::Identity(error.to_string()))?;
        manifest
            .validate_sealed()
            .map_err(|error| BooguArtifactLoadError::Identity(error.to_string()))?;
        validate_manifest_identity(identity, manifest, profile)?;
        let metadata = &manifest.metadata;
        let expected_layout = match (manifest.schema_version, manifest.dependencies.is_empty()) {
            (burn_image::ARTIFACT_MANIFEST_SCHEMA_V1, true) => "semantic-burnpack-v1",
            (burn_image::ARTIFACT_MANIFEST_SCHEMA_V2, false) => "semantic-burnpack-composition-v2",
            (schema, empty) => {
                return Err(BooguArtifactLoadError::Identity(format!(
                    "manifest schema {schema} and dependency state ({}) do not form a supported release layout",
                    if empty { "empty" } else { "non-empty" }
                )));
            }
        };
        if metadata.get("artifact_layout").map(String::as_str) != Some(expected_layout)
            || metadata.get("layout_contract").map(String::as_str)
                != Some("metadata/tensor-inventory.json")
            || metadata
                .get("conversion_crate")
                .is_none_or(|value| validate_supported_bundle_converter_version(value).is_err())
        {
            return Err(BooguArtifactLoadError::Identity(format!(
                "manifest omits a supported semantic Burnpack layout/converter contract; supported converters are {:?}",
                super::SUPPORTED_BUNDLE_CONVERTER_VERSIONS
            )));
        }
        tensor_inventory_schema(manifest)?;
        let tensor_count = metadata
            .get("tensor_count")
            .ok_or_else(|| BooguArtifactLoadError::Identity("manifest omits tensor_count".into()))?
            .parse::<usize>()
            .map_err(|error| {
                BooguArtifactLoadError::Identity(format!(
                    "manifest tensor_count is invalid: {error}"
                ))
            })?;
        let expected_tensor_count = manifest_inventory_specs(manifest, inventory).len();
        if tensor_count != expected_tensor_count {
            return Err(BooguArtifactLoadError::Identity(format!(
                "manifest declares {tensor_count} tensors, local exact inventory requires {}",
                expected_tensor_count
            )));
        }
        let inventory_path = metadata
            .get("layout_contract")
            .expect("layout contract presence checked");
        if !manifest.files.iter().any(|file| {
            file.path.as_str() == inventory_path && file.role == ArtifactFileRole::Metadata
        }) {
            return Err(BooguArtifactLoadError::Identity(format!(
                "manifest does not seal its declared tensor inventory {inventory_path}"
            )));
        }
        let target_max = declared_target_max_shard_bytes(manifest)?;
        let bounded = metadata
            .get("physical_shards_bounded")
            .map(String::as_str)
            .ok_or_else(|| {
                BooguArtifactLoadError::Identity("manifest omits physical_shards_bounded".into())
            })?;
        if !matches!(bounded, "true" | "false") {
            return Err(BooguArtifactLoadError::Identity(
                "physical_shards_bounded must be true or false".into(),
            ));
        }
        if bounded == "true"
            && let Some(file) = manifest
                .files
                .iter()
                .filter(|file| file.role == ArtifactFileRole::Weights)
                .find(|file| file.size > target_max)
        {
            return Err(BooguArtifactLoadError::Identity(format!(
                "manifest claims bounded shards but {} is {} bytes (target {target_max})",
                file.path, file.size
            )));
        }
        Ok(())
    }

    /// Schema-v2 Boogu composition manifests own only the variant-specific denoiser inventory;
    /// their sealed Qwen/VAE dependencies are validated and loaded by the reusable model crates.
    /// Legacy schema-v1 monoliths continue to bind the complete three-model inventory.
    fn manifest_inventory_specs<'a>(
        manifest: &ArtifactManifest,
        inventory: &'a BooguArtifactInventory,
    ) -> Vec<&'a super::ArtifactTensorSpec> {
        let owner = match manifest.metadata.get("component_kind").map(String::as_str) {
            Some("qwen3-vl-base-conditioning") => Some(TensorOwner::Qwen3Vl),
            Some("flux1-vae") => Some(TensorOwner::FluxVae),
            _ if !manifest.dependencies.is_empty() => Some(TensorOwner::BooguDenoiser),
            _ => None,
        };
        inventory
            .tensors()
            .iter()
            .filter(|spec| owner.is_none_or(|owner| spec.owner == owner))
            .collect()
    }

    pub(crate) fn declared_target_max_shard_bytes(
        manifest: &ArtifactManifest,
    ) -> Result<u64, BooguArtifactLoadError> {
        let value = manifest
            .metadata
            .get("target_max_shard_bytes")
            .ok_or_else(|| {
                BooguArtifactLoadError::Identity("manifest omits target_max_shard_bytes".into())
            })?
            .parse::<u64>()
            .map_err(|error| {
                BooguArtifactLoadError::Identity(format!(
                    "manifest target_max_shard_bytes is invalid: {error}"
                ))
            })?;
        if value == 0 {
            return Err(BooguArtifactLoadError::Identity(
                "manifest target_max_shard_bytes must be non-zero".into(),
            ));
        }
        Ok(value)
    }

    fn validate_manifest_identity(
        identity: &BooguReleaseIdentity,
        manifest: &ArtifactManifest,
        profile: BooguStorageProfile,
    ) -> Result<(), BooguArtifactLoadError> {
        let expected_model = match identity.variant {
            crate::BooguVariant::Image01Turbo => "Boogu/Boogu-Image-0.1-Turbo",
            crate::BooguVariant::Image01EditTurbo => "Boogu/Boogu-Image-0.1-Edit-Turbo",
            crate::BooguVariant::Image01EditTurbo1k5 => "Boogu/Boogu-Image-0.1-Edit-Turbo-1K5",
        };
        if manifest.model.as_str() != expected_model
            || manifest.model_revision != identity.model_revision
            || manifest.metadata.get("source_revision").map(String::as_str)
                != Some(identity.upstream_source_revision.as_str())
            || manifest.metadata.get("algorithm").map(String::as_str) != Some("dmd-turbo")
        {
            return Err(BooguArtifactLoadError::Identity(
                "manifest model, revisions, or algorithm differ from the canonical release".into(),
            ));
        }
        let (profile_name, numeric) = match profile {
            BooguStorageProfile::F16 => ("f16", NumericFormat::F16),
            BooguStorageProfile::F16QwenVisionF32 => (
                "f16-qwen-vision-f32",
                NumericFormat::Other("f16-qwen-vision-f32".into()),
            ),
            BooguStorageProfile::Q8sBlock32F32 => (
                "q8s-block32-f32",
                NumericFormat::Other("q8s-block32-f32".into()),
            ),
            BooguStorageProfile::Q8sBlock32F32QwenVisionF32 => (
                "q8s-block32-f32-qwen-vision-f32",
                NumericFormat::Other("q8s-block32-f32-qwen-vision-f32".into()),
            ),
        };
        if manifest.profile.as_str() != profile_name
            || manifest.numeric_format != numeric
            || manifest.metadata.get("profile").map(String::as_str) != Some(profile_name)
        {
            return Err(BooguArtifactLoadError::Identity(format!(
                "manifest profile {:?}/{:?} does not match requested {profile:?}",
                manifest.profile, manifest.numeric_format
            )));
        }
        Ok(())
    }

    fn single_block<B: Backend>(
        config: &BooguConfig,
        modulation: bool,
        device: &B::Device,
    ) -> SingleStreamBlock<B> {
        SingleStreamBlock::new(
            config.hidden_size,
            config.ffn_inner_dim(),
            config.num_attention_heads,
            config.num_kv_heads,
            config.hidden_size.min(1024),
            config.norm_eps,
            modulation,
            device,
        )
    }

    fn rename_snapshot(snapshot: &TensorSnapshot, name: &str) -> TensorSnapshot {
        TensorSnapshot::from_closure(
            snapshot.clone_data_fn(),
            snapshot.dtype,
            snapshot.shape.clone(),
            name.split('.').map(str::to_owned).collect(),
            snapshot.container_stack.clone().unwrap_or_default(),
            snapshot.tensor_id.unwrap_or_default(),
        )
    }

    fn validate_partial_apply(
        stage: &str,
        result: &ApplyResult,
        expected_applied: &BTreeSet<String>,
    ) -> Result<(), String> {
        if result.applied.is_empty() {
            return Err(format!("Burn applied no tensors for {stage}"));
        }
        if !result.skipped.is_empty() || !result.unused.is_empty() || !result.errors.is_empty() {
            return Err(format!(
                "skipped={:?}, unused={:?}, errors={:?}",
                result.skipped, result.unused, result.errors
            ));
        }
        let actual = result.applied.iter().cloned().collect::<BTreeSet<_>>();
        if &actual != expected_applied || actual.len() != result.applied.len() {
            return Err(format!(
                "applied path mismatch: expected={expected_applied:?}, actual={actual:?}"
            ));
        }
        Ok(())
    }

    fn validate_apply_result(
        stage: &str,
        result: &ApplyResult,
        expected_applied: &BTreeSet<String>,
    ) -> Result<(), BooguArtifactLoadError> {
        if result.applied.is_empty() {
            return Err(contract(stage, "Burn applied no tensors"));
        }
        if !result.skipped.is_empty() || !result.unused.is_empty() || !result.errors.is_empty() {
            return Err(contract(
                stage,
                format!(
                    "Burn apply rejected tensors: skipped={:?}, unused={:?}, errors={:?}",
                    result.skipped, result.unused, result.errors
                ),
            ));
        }
        let actual = result.applied.iter().cloned().collect::<BTreeSet<_>>();
        if &actual != expected_applied || actual.len() != result.applied.len() {
            return Err(contract(
                stage,
                format!(
                    "Burn applied the wrong paths: expected={expected_applied:?}, actual={actual:?}"
                ),
            ));
        }
        Ok(())
    }

    fn expected_spec_storage(
        profile: BooguStorageProfile,
        spec: &super::ArtifactTensorSpec,
        vae_encoder_f32_diagnostic: bool,
    ) -> (&'static str, bool) {
        if vae_encoder_f32_diagnostic
            && spec.owner == TensorOwner::FluxVae
            && spec.stage == "flux-vae-encoder"
        {
            return ("f32", false);
        }
        let qwen_vision =
            spec.owner == TensorOwner::Qwen3Vl && spec.stage.starts_with("qwen-vision-");
        let mixed_vision = matches!(
            profile,
            BooguStorageProfile::F16QwenVisionF32 | BooguStorageProfile::Q8sBlock32F32QwenVisionF32
        );
        if mixed_vision && qwen_vision {
            return ("f32", false);
        }
        let quantized = spec.quantizable
            && matches!(
                profile,
                BooguStorageProfile::Q8sBlock32F32
                    | BooguStorageProfile::Q8sBlock32F32QwenVisionF32
            );
        if quantized {
            ("q8s-block32-f32", true)
        } else {
            ("f16", false)
        }
    }

    fn validate_spec_dtype(
        profile: BooguStorageProfile,
        spec: &super::ArtifactTensorSpec,
        actual: DType,
    ) -> Result<(), String> {
        let (stored_dtype, quantized) = expected_spec_storage(profile, spec, false);
        validate_stored_dtype(stored_dtype, quantized, actual)
    }

    fn validate_entry_dtype(
        entry: &SerializedTensorInventory,
        actual: DType,
    ) -> Result<(), String> {
        let stored_dtype = entry
            .stored_dtype
            .as_deref()
            .ok_or_else(|| "stored tensor omits dtype".to_owned())?;
        validate_stored_dtype(stored_dtype, entry.quantized, actual)
    }

    fn validate_stored_dtype(
        stored_dtype: &str,
        quantized: bool,
        actual: DType,
    ) -> Result<(), String> {
        let matches = match (stored_dtype, quantized, actual) {
            ("f16", false, DType::F16) | ("f32", false, DType::F32) => true,
            ("q8s-block32-f32", true, DType::QFloat(scheme)) => {
                scheme.value == burn::tensor::quantization::QuantValue::Q8S
                    && scheme.level == burn::tensor::quantization::QuantLevel::block([32])
                    && scheme.param == burn::tensor::quantization::QuantParam::F32
            }
            _ => false,
        };
        if matches {
            Ok(())
        } else {
            Err(format!(
                "sealed dtype {stored_dtype} (quantized={quantized}) does not permit {actual:?}"
            ))
        }
    }

    fn contract(stage: &str, message: impl Into<String>) -> BooguArtifactLoadError {
        BooguArtifactLoadError::Contract {
            stage: stage.into(),
            message: message.into(),
        }
    }

    pub(super) fn validate_runtime_denoiser_quantization_policy(
        profile: BooguStorageProfile,
        float_policy: BooguFloatLoadPolicy,
        quantized_policy: BooguQuantizedLoadPolicy,
        runtime_quantization_policy: BooguDenoiserRuntimeQuantizationPolicy,
        runtime_q8_scope: BooguRuntimeQ8Scope,
        variant: BooguVariant,
        stage: &str,
    ) -> Result<(), BooguError> {
        if runtime_quantization_policy == BooguDenoiserRuntimeQuantizationPolicy::Q8sBlock32F32
            && !matches!(
                profile,
                BooguStorageProfile::F16 | BooguStorageProfile::F16QwenVisionF32
            )
        {
            return Err(BooguError::Artifact(format!(
                "stage {stage} runtime Q8S quantization requires a sealed F16 production denoiser; profile {profile:?} already stores quantized tensors"
            )));
        }
        if runtime_quantization_policy == BooguDenoiserRuntimeQuantizationPolicy::Q8sBlock32F32
            && quantized_policy != BooguQuantizedLoadPolicy::Preserve
        {
            return Err(BooguError::Artifact(format!(
                "stage {stage} runtime Q8S quantization requires the stored-payload quantized load policy Preserve"
            )));
        }
        runtime_q8_scope.validate_variant(variant)?;
        if runtime_q8_scope != BooguRuntimeQ8Scope::AllInventoryEligible
            && (profile != BooguStorageProfile::F16QwenVisionF32
                || float_policy != BooguFloatLoadPolicy::AdaptToF32
                || quantized_policy != BooguQuantizedLoadPolicy::Preserve
                || runtime_quantization_policy
                    != BooguDenoiserRuntimeQuantizationPolicy::Q8sBlock32F32)
        {
            return Err(BooguError::Artifact(format!(
                "stage {stage} runtime Q8 scope {} requires the production F16/Qwen-vision-F32 profile, AdaptToF32, and runtime Q8S quantization",
                runtime_q8_scope.label()
            )));
        }
        Ok(())
    }

    /// Quantize one row-layout floating-point payload with the canonical Boogu Q8S algorithm.
    ///
    /// Both the offline importer and the verified runtime denoiser adapter call this function, so
    /// block scaling, rounding, clamping, zero-block handling, and packed storage stay byte-for-byte
    /// identical for identical input values. The caller remains responsible for authenticating the
    /// source tensor and proving that its inventory contract permits quantization.
    pub fn quantize_q8s_block32_f32(
        values: Vec<f32>,
        shape: Vec<usize>,
    ) -> Result<TensorData, TensorSnapshotError> {
        let elements = shape.iter().try_fold(1_usize, |product, dimension| {
            product.checked_mul(*dimension)
        });
        if elements != Some(values.len()) {
            return Err(TensorSnapshotError::DataError(format!(
                "Q8S shape {shape:?} describes {elements:?} elements, received {} values",
                values.len()
            )));
        }
        if !values.len().is_multiple_of(32) {
            return Err(TensorSnapshotError::DataError(
                "Q8S block tensor length is not divisible by 32".into(),
            ));
        }
        let mut quantized = Vec::with_capacity(values.len());
        let mut scales = Vec::with_capacity(values.len() / 32);
        for block in values.chunks_exact(32) {
            if block.iter().any(|value| !value.is_finite()) {
                return Err(TensorSnapshotError::DataError(
                    "cannot quantize a non-finite checkpoint value".into(),
                ));
            }
            let alpha = block
                .iter()
                .fold(0.0_f32, |value, element| value.max(element.abs()));
            let scale = if alpha == 0.0 {
                f32::MIN_POSITIVE
            } else {
                alpha / 127.0
            };
            scales.push(scale);
            let inverse = scale.recip();
            quantized.extend(
                block
                    .iter()
                    .map(|value| (value * inverse).round().clamp(-127.0, 127.0) as i8),
            );
        }
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_level(QuantLevel::block([32]))
            .with_param(QuantParam::F32)
            .with_store(QuantStore::PackedU32(0));
        Ok(TensorData::quantized(quantized, shape, scheme, &scales))
    }

    pub(super) fn quantize_verified_float_q8s_block32_f32(
        data: TensorData,
    ) -> Result<TensorData, TensorSnapshotError> {
        if !matches!(data.dtype, DType::F16 | DType::F32) {
            return Err(TensorSnapshotError::DataError(format!(
                "runtime Q8S denoiser quantization requires a verified F16/F32 snapshot, found {:?}",
                data.dtype
            )));
        }
        let shape = data.shape.to_vec();
        let values = data
            .convert_dtype(DType::F32)
            .to_vec::<f32>()
            .map_err(|error| TensorSnapshotError::DataError(error.to_string()))?;
        quantize_q8s_block32_f32(values, shape)
    }

    #[derive(Debug, Clone)]
    pub(super) struct ArtifactLoadAdapter {
        pub(super) float_policy: BooguFloatLoadPolicy,
        pub(super) quantized_policy: BooguQuantizedLoadPolicy,
        pub(super) runtime_quantization_policy: BooguDenoiserRuntimeQuantizationPolicy,
        pub(super) runtime_quantizable_paths: Option<BTreeSet<String>>,
    }

    pub(super) fn load_adapter(
        float_policy: BooguFloatLoadPolicy,
        quantized_policy: BooguQuantizedLoadPolicy,
    ) -> Option<Box<dyn ModuleAdapter>> {
        if float_policy == BooguFloatLoadPolicy::Preserve
            && quantized_policy == BooguQuantizedLoadPolicy::Preserve
        {
            None
        } else {
            Some(Box::new(ArtifactLoadAdapter {
                float_policy,
                quantized_policy,
                runtime_quantization_policy: BooguDenoiserRuntimeQuantizationPolicy::Disabled,
                runtime_quantizable_paths: None,
            }) as Box<dyn ModuleAdapter>)
        }
    }

    fn denoiser_load_adapter(
        float_policy: BooguFloatLoadPolicy,
        quantized_policy: BooguQuantizedLoadPolicy,
        runtime_quantization_policy: BooguDenoiserRuntimeQuantizationPolicy,
        runtime_quantizable_paths: BTreeSet<String>,
    ) -> Option<Box<dyn ModuleAdapter>> {
        if float_policy == BooguFloatLoadPolicy::Preserve
            && quantized_policy == BooguQuantizedLoadPolicy::Preserve
            && runtime_quantization_policy == BooguDenoiserRuntimeQuantizationPolicy::Disabled
        {
            None
        } else {
            Some(Box::new(ArtifactLoadAdapter {
                float_policy,
                quantized_policy,
                runtime_quantization_policy,
                runtime_quantizable_paths: Some(runtime_quantizable_paths),
            }) as Box<dyn ModuleAdapter>)
        }
    }

    fn dequantize_q8s_block32(
        data: TensorData,
        float_policy: BooguFloatLoadPolicy,
    ) -> Result<TensorData, TensorSnapshotError> {
        let DType::QFloat(scheme) = data.dtype else {
            return Err(TensorSnapshotError::DataError(
                "DequantizeF16 received a non-quantized snapshot".into(),
            ));
        };
        if scheme.value != burn::tensor::quantization::QuantValue::Q8S
            || scheme.level != burn::tensor::quantization::QuantLevel::block([32])
            || scheme.param != burn::tensor::quantization::QuantParam::F32
        {
            return Err(TensorSnapshotError::DataError(format!(
                "DequantizeF16 only accepts sealed Q8S block-32/F32, found {scheme:?}"
            )));
        }
        let shape = data.shape.clone();
        let num_elements = data.num_elements();
        let (values, qparams) = QuantizedBytes {
            bytes: data.bytes,
            scheme,
            num_elements,
        }
        .into_vec_i8();
        if values.len() != num_elements || !values.len().is_multiple_of(32) {
            return Err(TensorSnapshotError::DataError(format!(
                "invalid Q8S block-32 payload length: {} values for {num_elements} elements",
                values.len()
            )));
        }
        if qparams.scales.len() != values.len() / 32 {
            return Err(TensorSnapshotError::DataError(format!(
                "invalid Q8S block-32 scale count: expected {}, found {}",
                values.len() / 32,
                qparams.scales.len()
            )));
        }
        let mut dequantized = Vec::with_capacity(values.len());
        for (block, scale) in values.chunks_exact(32).zip(qparams.scales) {
            if !scale.is_finite() {
                return Err(TensorSnapshotError::DataError(
                    "Q8S block-32 payload contains a non-finite scale".into(),
                ));
            }
            for value in block {
                let value = f32::from(*value) * scale;
                if !value.is_finite() || value.abs() > 65_504.0 {
                    return Err(TensorSnapshotError::DataError(format!(
                        "Q8S value {value} cannot be represented as F16"
                    )));
                }
                dequantized.push(value);
            }
        }
        let data = TensorData::new(dequantized, shape).convert_dtype(DType::F16);
        Ok(match float_policy {
            BooguFloatLoadPolicy::Preserve => data,
            BooguFloatLoadPolicy::AdaptToF32 => data.convert_dtype(DType::F32),
        })
    }

    impl ModuleAdapter for ArtifactLoadAdapter {
        fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
            if self.runtime_quantization_policy
                == BooguDenoiserRuntimeQuantizationPolicy::Q8sBlock32F32
            {
                let Some(runtime_quantizable_paths) = &self.runtime_quantizable_paths else {
                    return TensorSnapshot::from_closure(
                        Rc::new(|| {
                            Err(TensorSnapshotError::DataError(
                                "runtime Q8S policy is restricted to inventory-qualified Boogu denoiser stages"
                                    .into(),
                            ))
                        }),
                        snapshot.dtype,
                        snapshot.shape.clone(),
                        snapshot.path_stack.clone().unwrap_or_default(),
                        snapshot.container_stack.clone().unwrap_or_default(),
                        snapshot.tensor_id.unwrap_or_default(),
                    );
                };
                if runtime_quantizable_paths.contains(&snapshot.full_path()) {
                    let data_fn = snapshot.clone_data_fn();
                    return TensorSnapshot::from_closure(
                        Rc::new(move || quantize_verified_float_q8s_block32_f32(data_fn()?)),
                        DType::QFloat(
                            QuantScheme::default()
                                .with_value(QuantValue::Q8S)
                                .with_level(QuantLevel::block([32]))
                                .with_param(QuantParam::F32)
                                .with_store(QuantStore::PackedU32(0)),
                        ),
                        snapshot.shape.clone(),
                        snapshot.path_stack.clone().unwrap_or_default(),
                        snapshot.container_stack.clone().unwrap_or_default(),
                        snapshot.tensor_id.unwrap_or_default(),
                    );
                }
            }
            let output_dtype = match (snapshot.dtype, self.quantized_policy, self.float_policy) {
                (DType::QFloat(_), BooguQuantizedLoadPolicy::DequantizeF16, float_policy) => {
                    let data_fn = snapshot.clone_data_fn();
                    return TensorSnapshot::from_closure(
                        Rc::new(move || dequantize_q8s_block32(data_fn()?, float_policy)),
                        match float_policy {
                            BooguFloatLoadPolicy::Preserve => DType::F16,
                            BooguFloatLoadPolicy::AdaptToF32 => DType::F32,
                        },
                        snapshot.shape.clone(),
                        snapshot.path_stack.clone().unwrap_or_default(),
                        snapshot.container_stack.clone().unwrap_or_default(),
                        snapshot.tensor_id.unwrap_or_default(),
                    );
                }
                (DType::F16 | DType::BF16 | DType::F64, _, BooguFloatLoadPolicy::AdaptToF32) => {
                    DType::F32
                }
                _ => return snapshot.clone(),
            };
            let data_fn = snapshot.clone_data_fn();
            TensorSnapshot::from_closure(
                Rc::new(move || Ok(data_fn()?.convert_dtype(output_dtype))),
                output_dtype,
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
}

#[cfg(feature = "burnpack")]
pub use loading::{
    AsyncStageShardRead, AsyncStageShardReader, BooguArtifactLoadError, BooguBurnpackLoader,
    BooguComponentVerification, BooguDenoiserRuntimeQuantizationPolicy, BooguFloatLoadPolicy,
    BooguLoadReport, BooguModels, BooguModularReleaseVerification, BooguQuantizedLoadPolicy,
    BooguReleaseVerification, BooguResidentLoadMemoryPolicy, BooguStorageProfile,
    BooguVaeEncoderF32DiagnosticManifest, CanonicalBooguArtifactBundle, DirectoryStageShardReader,
    PUBLISHED_ARTIFACT_BUNDLES, StageShardReader, VerifiedArtifactDirectory,
    VerifiedAsyncBurnpackDenoiserStageSource, VerifiedAsyncBurnpackQwenStageSource,
    VerifiedAsyncBurnpackVaeStageSource, VerifiedBurnpackQwenStageSource,
    VerifiedBurnpackStageSource, VerifiedDirectoryVaeStageSource, artifact_bundle_id_is_compatible,
    canonical_published_bundle, legacy_artifact_bundle_id, load_resident_denoiser_from_directory,
    load_resident_denoiser_from_directory_with_memory_policy,
    load_resident_denoiser_from_directory_with_policies, load_resident_qwen_base_from_directory,
    load_resident_vae_from_directory, load_vae_decoder, load_vae_decoder_from_directory,
    load_vae_encoder, load_vae_encoder_from_directory, preferred_artifact_bundle_id,
    promotable_legacy_artifact_digest, quantize_q8s_block32_f32,
    stamp_edit_turbo_1k5_vae_encoder_f32_diagnostic_metadata,
    validate_canonical_release_artifact_digest, validate_edit_turbo_1k5_release_artifact_digest,
    validate_edit_turbo_1k5_vae_encoder_f32_diagnostic_manifest,
    verify_modular_release_artifact_directories, verify_published_release_artifact_directory,
    verify_release_artifact_directory,
};

#[cfg(all(feature = "burnpack", feature = "wgpu"))]
pub(crate) use loading::{
    SerializedTensorInventory, declared_target_max_shard_bytes, read_verified_async,
    validate_release_manifest, verify_inventory_contract_async,
};

#[cfg(test)]
mod tests {
    #[cfg(feature = "burnpack")]
    use std::{collections::BTreeMap, path::PathBuf};

    use super::*;
    #[cfg(feature = "burnpack")]
    use burn_image::Sha256Digest;
    #[cfg(feature = "burnpack")]
    use burn_image::{ArtifactFile, ArtifactManifest};

    #[test]
    fn turbo_ffn_core_q8_scope_is_exact_and_fail_closed_correctness() {
        let scope = BooguRuntimeQ8Scope::TurboFfnCoreQ8;
        assert_eq!(scope.label(), "turbo-ffn-core-q8");
        for target in [
            "single_stream_layers.0.feed_forward.linear_1.weight",
            "double_stream_layers.0.image_ffn.linear_2.weight",
            "double_stream_layers.0.instruction_ffn.linear_3.weight",
        ] {
            assert!(scope.quantizes_target(target), "did not select {target}");
        }
        for target in [
            "single_stream_layers.0.attn.to_q.weight",
            "double_stream_layers.0.joint_attn.to_q.weight",
            "double_stream_layers.0.image_self_attn.to_q.weight",
            "single_stream_layers.0.feed_forward.linear_1.bias",
            "time_caption_embed.caption_linear.weight",
            "norm_out.linear_1.weight",
            "feed_forward.weight",
        ] {
            assert!(
                !scope.quantizes_target(target),
                "unexpectedly selected {target}"
            );
        }

        let inventory = BooguArtifactInventory::denoiser(&BooguConfig::default()).unwrap();
        let quantized = inventory
            .tensors()
            .iter()
            .filter(|spec| {
                spec.quantizable
                    && !spec.stage.starts_with("boogu-reference-refiner-")
                    && scope.quantizes_target(&spec.target_name)
            })
            .map(|spec| spec.target_name.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(quantized.len(), 156);
        assert!(
            quantized
                .iter()
                .all(|target| turbo_ffn_core_q8_target(target))
        );

        let footprint = inventory
            .denoiser_runtime_q8s_block32_f32_footprint_with_scope(
                BooguVariant::Image01Turbo,
                scope,
            )
            .unwrap();
        assert_eq!(footprint.tensor_count, 912);
        assert_eq!(footprint.quantized_tensor_count, 156);
        assert_eq!(footprint.f32_tensor_count, 756);
        assert_eq!(footprint.quantized_elements, 7_111_802_880);
        assert_eq!(footprint.f32_elements, 2_823_195_168);
        assert_eq!(footprint.quantized_payload_bytes, 8_000_778_240);
        assert_eq!(footprint.f32_payload_bytes, 11_292_780_672);
        assert_eq!(footprint.total_payload_bytes, 19_293_558_912);
        for variant in [
            BooguVariant::Image01EditTurbo,
            BooguVariant::Image01EditTurbo1k5,
        ] {
            assert!(
                inventory
                    .denoiser_runtime_q8s_block32_f32_footprint_with_scope(variant, scope)
                    .is_err()
            );
        }
    }

    #[test]
    fn turbo_ffn_gate_up_q8_scope_is_exact_and_fail_closed_correctness() {
        let scope = BooguRuntimeQ8Scope::TurboFfnGateUpQ8;
        assert_eq!(scope.label(), "turbo-ffn-gate-up-q8");
        for target in [
            "single_stream_layers.0.feed_forward.linear_1.weight",
            "single_stream_layers.0.feed_forward.linear_3.weight",
            "double_stream_layers.0.image_ffn.linear_1.weight",
            "double_stream_layers.0.image_ffn.linear_3.weight",
            "double_stream_layers.0.instruction_ffn.linear_1.weight",
            "double_stream_layers.0.instruction_ffn.linear_3.weight",
        ] {
            assert!(scope.quantizes_target(target), "did not select {target}");
        }
        for target in [
            "single_stream_layers.0.feed_forward.linear_2.weight",
            "double_stream_layers.0.image_ffn.linear_2.weight",
            "double_stream_layers.0.instruction_ffn.linear_2.weight",
            "single_stream_layers.0.attn.linear_1.weight",
            "single_stream_layers.0.feed_forward.linear_1.bias",
            "single_stream_layers.0.feed_forward.other_linear_1.weight",
            "feed_forward.linear_1.weight",
        ] {
            assert!(
                !scope.quantizes_target(target),
                "unexpectedly selected {target}"
            );
        }

        let inventory = BooguArtifactInventory::denoiser(&BooguConfig::default()).unwrap();
        let quantized = inventory
            .tensors()
            .iter()
            .filter(|spec| {
                spec.quantizable
                    && !spec.stage.starts_with("boogu-reference-refiner-")
                    && scope.quantizes_target(&spec.target_name)
            })
            .map(|spec| spec.target_name.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(quantized.len(), 104);
        assert!(
            quantized
                .iter()
                .all(|target| turbo_ffn_gate_up_q8_target(target))
        );

        let footprint = inventory
            .denoiser_runtime_q8s_block32_f32_footprint_with_scope(
                BooguVariant::Image01Turbo,
                scope,
            )
            .unwrap();
        assert_eq!(footprint.tensor_count, 912);
        assert_eq!(footprint.quantized_tensor_count, 104);
        assert_eq!(footprint.f32_tensor_count, 808);
        assert_eq!(footprint.quantized_elements, 4_741_201_920);
        assert_eq!(footprint.f32_elements, 5_193_796_128);
        assert_eq!(footprint.quantized_payload_bytes, 5_333_852_160);
        assert_eq!(footprint.f32_payload_bytes, 20_775_184_512);
        assert_eq!(footprint.total_payload_bytes, 26_109_036_672);
        for variant in [
            BooguVariant::Image01EditTurbo,
            BooguVariant::Image01EditTurbo1k5,
        ] {
            assert!(
                inventory
                    .denoiser_runtime_q8s_block32_f32_footprint_with_scope(variant, scope)
                    .is_err()
            );
        }
    }

    #[test]
    fn turbo_main_core_ffn_gate_up_q8_scope_matches_exact_canonical_target_set_correctness() {
        let scope = BooguRuntimeQ8Scope::TurboMainCoreFfnGateUpQ8;
        assert_eq!(scope.label(), "turbo-main-core-ffn-gate-up-q8");

        let mut expected = BTreeSet::<String>::new();
        for layer in 0..32 {
            for projection in [1, 3] {
                expected.insert(format!(
                    "single_stream_layers.{layer}.feed_forward.linear_{projection}.weight"
                ));
            }
        }
        for layer in 0..8 {
            for module in ["image_ffn", "instruction_ffn"] {
                for projection in [1, 3] {
                    expected.insert(format!(
                        "double_stream_layers.{layer}.{module}.linear_{projection}.weight"
                    ));
                }
            }
        }
        assert_eq!(expected.len(), 96);
        assert!(expected.iter().all(|target| scope.quantizes_target(target)));

        for target in [
            "single_stream_layers.0.feed_forward.linear_2.weight",
            "single_stream_layers.32.feed_forward.linear_1.weight",
            "single_stream_layers.00.feed_forward.linear_1.weight",
            "single_stream_layers.+1.feed_forward.linear_1.weight",
            "double_stream_layers.0.image_ffn.linear_2.weight",
            "double_stream_layers.8.image_ffn.linear_1.weight",
            "double_stream_layers.01.image_ffn.linear_1.weight",
            "double_stream_layers.0.feed_forward.linear_1.weight",
            "context_refiner.0.feed_forward.linear_1.weight",
            "context_refiner.0.feed_forward.linear_3.weight",
            "noise_refiner.0.feed_forward.linear_1.weight",
            "noise_refiner.0.feed_forward.linear_3.weight",
            "single_stream_layers.0.feed_forward.linear_1.bias",
            "single_stream_layers.0.feed_forward.linear_1.weight.extra",
            "prefix.single_stream_layers.0.feed_forward.linear_1.weight",
        ] {
            assert!(
                !scope.quantizes_target(target),
                "unexpectedly selected {target}"
            );
        }

        let inventory = BooguArtifactInventory::denoiser(&BooguConfig::default()).unwrap();
        let selected = inventory
            .tensors()
            .iter()
            .filter(|spec| {
                spec.quantizable
                    && !spec.stage.starts_with("boogu-reference-refiner-")
                    && scope.quantizes_target(&spec.target_name)
            })
            .map(|spec| spec.target_name.clone())
            .collect::<BTreeSet<_>>();
        assert_eq!(selected, expected);

        let footprint = inventory
            .denoiser_runtime_q8s_block32_f32_footprint_with_scope(
                BooguVariant::Image01Turbo,
                scope,
            )
            .unwrap();
        assert_eq!(footprint.tensor_count, 912);
        assert_eq!(footprint.quantized_tensor_count, 96);
        assert_eq!(footprint.f32_tensor_count, 816);
        assert_eq!(footprint.quantized_elements, 4_376_494_080);
        assert_eq!(footprint.f32_elements, 5_558_503_968);
        assert_eq!(footprint.quantized_payload_bytes, 4_923_555_840);
        assert_eq!(footprint.f32_payload_bytes, 22_234_015_872);
        assert_eq!(footprint.total_payload_bytes, 27_157_571_712);
        for variant in [
            BooguVariant::Image01EditTurbo,
            BooguVariant::Image01EditTurbo1k5,
        ] {
            assert!(
                inventory
                    .denoiser_runtime_q8s_block32_f32_footprint_with_scope(variant, scope)
                    .is_err()
            );
        }
    }

    #[test]
    fn runtime_q8_denoiser_footprint_is_inventory_derived_and_variant_exact_correctness() {
        let inventory = BooguArtifactInventory::denoiser(&BooguConfig::default()).unwrap();
        assert!(
            inventory
                .tensors()
                .iter()
                .all(|spec| spec.owner == TensorOwner::BooguDenoiser)
        );
        let turbo = inventory
            .denoiser_runtime_q8s_block32_f32_footprint(BooguVariant::Image01Turbo)
            .unwrap();
        let edit = inventory
            .denoiser_runtime_q8s_block32_f32_footprint(BooguVariant::Image01EditTurbo)
            .unwrap();
        let edit_1k5 = inventory
            .denoiser_runtime_q8s_block32_f32_footprint(BooguVariant::Image01EditTurbo1k5)
            .unwrap();
        let turbo_caption_tail_f32 = inventory
            .denoiser_runtime_q8s_block32_f32_footprint_with_scope(
                BooguVariant::Image01Turbo,
                BooguRuntimeQ8Scope::TurboCaptionAndTailF32,
            )
            .unwrap();
        let turbo_attention_ffn_core_q8 = inventory
            .denoiser_runtime_q8s_block32_f32_footprint_with_scope(
                BooguVariant::Image01Turbo,
                BooguRuntimeQ8Scope::TurboAttentionFfnCoreQ8,
            )
            .unwrap();
        let excluded = inventory
            .tensors()
            .iter()
            .filter(|spec| {
                spec.quantizable
                    && !BooguRuntimeQ8Scope::TurboCaptionAndTailF32
                        .quantizes_target(&spec.target_name)
            })
            .map(|spec| spec.target_name.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            excluded,
            BTreeSet::from([
                "norm_out.linear_1.weight",
                "norm_out.linear_2.weight",
                "time_caption_embed.caption_linear.weight",
            ])
        );
        assert_eq!(turbo.total_payload_bytes, 12_155_919_232);
        assert_eq!(edit.total_payload_bytes, 12_590_785_792);
        assert_eq!(edit_1k5, edit);
        assert_eq!(turbo_caption_tail_f32.tensor_count, 912);
        assert_eq!(turbo_caption_tail_f32.quantized_tensor_count, 362);
        assert_eq!(turbo_caption_tail_f32.f32_tensor_count, 550);
        assert_eq!(turbo_caption_tail_f32.quantized_elements, 9_577_041_920);
        assert_eq!(turbo_caption_tail_f32.f32_elements, 357_956_128);
        assert_eq!(
            turbo_caption_tail_f32.quantized_payload_bytes,
            10_774_172_160
        );
        assert_eq!(turbo_caption_tail_f32.f32_payload_bytes, 1_431_824_512);
        assert_eq!(turbo_caption_tail_f32.total_payload_bytes, 12_205_996_672);
        assert_eq!(
            turbo_caption_tail_f32.total_payload_bytes - turbo.total_payload_bytes,
            50_077_440
        );
        let core_scope_excluded = inventory
            .tensors()
            .iter()
            .filter(|spec| {
                spec.quantizable
                    && !spec.stage.starts_with("boogu-reference-refiner-")
                    && !BooguRuntimeQ8Scope::TurboAttentionFfnCoreQ8
                        .quantizes_target(&spec.target_name)
            })
            .map(|spec| spec.target_name.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(core_scope_excluded.len(), 81);
        assert!(
            core_scope_excluded
                .iter()
                .all(|target| { !turbo_attention_ffn_core_q8_target(target) })
        );
        let core_scope_quantized = inventory
            .tensors()
            .iter()
            .filter(|spec| {
                spec.quantizable
                    && !spec.stage.starts_with("boogu-reference-refiner-")
                    && BooguRuntimeQ8Scope::TurboAttentionFfnCoreQ8
                        .quantizes_target(&spec.target_name)
            })
            .map(|spec| spec.target_name.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(core_scope_quantized.len(), 284);
        assert!(
            core_scope_quantized
                .iter()
                .all(|target| turbo_attention_ffn_core_q8_target(target))
        );
        assert_eq!(
            core_scope_quantized
                .iter()
                .filter(|target| {
                    target.contains(".attn.")
                        || target.contains(".joint_attn.")
                        || target.contains(".image_self_attn.")
                })
                .count(),
            128
        );
        assert_eq!(
            core_scope_quantized
                .iter()
                .filter(|target| target.contains(".feed_forward."))
                .chain(core_scope_quantized.iter().filter(|target| {
                    target.contains(".image_ffn.") || target.contains(".instruction_ffn.")
                }),)
                .count(),
            156
        );
        for required in [
            "x_embedder.weight",
            "time_caption_embed.caption_linear.weight",
            "time_caption_embed.time_linear_1.weight",
            "time_caption_embed.time_linear_2.weight",
            "noise_refiner.0.norm1.linear.weight",
            "double_stream_layers.0.image_norm1.linear.weight",
            "double_stream_layers.7.instruction_norm2.linear.weight",
            "single_stream_layers.31.norm1.linear.weight",
            "norm_out.linear_1.weight",
            "norm_out.linear_2.weight",
        ] {
            assert!(core_scope_excluded.contains(required), "missing {required}");
        }
        assert_eq!(turbo_attention_ffn_core_q8.tensor_count, 912);
        assert_eq!(turbo_attention_ffn_core_q8.quantized_tensor_count, 284);
        assert_eq!(turbo_attention_ffn_core_q8.f32_tensor_count, 628);
        assert_eq!(
            turbo_attention_ffn_core_q8.quantized_elements,
            8_556_871_680
        );
        assert_eq!(turbo_attention_ffn_core_q8.f32_elements, 1_378_126_368);
        assert_eq!(
            turbo_attention_ffn_core_q8.quantized_payload_bytes,
            9_626_480_640
        );
        assert_eq!(turbo_attention_ffn_core_q8.f32_payload_bytes, 5_512_505_472);
        assert_eq!(
            turbo_attention_ffn_core_q8.total_payload_bytes,
            15_138_986_112
        );
        assert!(
            inventory
                .denoiser_runtime_q8s_block32_f32_footprint_with_scope(
                    BooguVariant::Image01EditTurbo1k5,
                    BooguRuntimeQ8Scope::TurboCaptionAndTailF32,
                )
                .is_err()
        );
        assert!(
            inventory
                .denoiser_runtime_q8s_block32_f32_footprint_with_scope(
                    BooguVariant::Image01EditTurbo,
                    BooguRuntimeQ8Scope::TurboAttentionFfnCoreQ8,
                )
                .is_err()
        );
        assert_eq!(
            turbo.total_payload_bytes,
            turbo.quantized_payload_bytes + turbo.f32_payload_bytes
        );
        assert_eq!(
            edit.total_payload_bytes,
            edit.quantized_payload_bytes + edit.f32_payload_bytes
        );

        let reference_refiner_bytes = inventory
            .tensors()
            .iter()
            .filter(|spec| spec.stage.starts_with("boogu-reference-refiner-"))
            .map(|spec| {
                let elements = spec.target_shape.iter().product::<usize>() as u64;
                if spec.quantizable {
                    elements + elements / 32 * 4
                } else {
                    elements * 4
                }
            })
            .sum::<u64>();
        assert!(reference_refiner_bytes > 0);
        assert_eq!(
            edit.total_payload_bytes - turbo.total_payload_bytes,
            reference_refiner_bytes
        );
    }

    #[cfg(feature = "burnpack")]
    #[test]
    fn turbo_caption_tail_f32_runtime_q8_scope_is_fail_closed_correctness() {
        let scope = BooguRuntimeQ8Scope::TurboCaptionAndTailF32;
        loading::validate_runtime_denoiser_quantization_policy(
            BooguStorageProfile::F16QwenVisionF32,
            BooguFloatLoadPolicy::AdaptToF32,
            BooguQuantizedLoadPolicy::Preserve,
            BooguDenoiserRuntimeQuantizationPolicy::Q8sBlock32F32,
            scope,
            BooguVariant::Image01Turbo,
            "boogu-prelude",
        )
        .unwrap();

        for result in [
            loading::validate_runtime_denoiser_quantization_policy(
                BooguStorageProfile::F16,
                BooguFloatLoadPolicy::AdaptToF32,
                BooguQuantizedLoadPolicy::Preserve,
                BooguDenoiserRuntimeQuantizationPolicy::Q8sBlock32F32,
                scope,
                BooguVariant::Image01Turbo,
                "boogu-prelude",
            ),
            loading::validate_runtime_denoiser_quantization_policy(
                BooguStorageProfile::F16QwenVisionF32,
                BooguFloatLoadPolicy::Preserve,
                BooguQuantizedLoadPolicy::Preserve,
                BooguDenoiserRuntimeQuantizationPolicy::Q8sBlock32F32,
                scope,
                BooguVariant::Image01Turbo,
                "boogu-prelude",
            ),
            loading::validate_runtime_denoiser_quantization_policy(
                BooguStorageProfile::F16QwenVisionF32,
                BooguFloatLoadPolicy::AdaptToF32,
                BooguQuantizedLoadPolicy::Preserve,
                BooguDenoiserRuntimeQuantizationPolicy::Disabled,
                scope,
                BooguVariant::Image01Turbo,
                "boogu-prelude",
            ),
            loading::validate_runtime_denoiser_quantization_policy(
                BooguStorageProfile::F16QwenVisionF32,
                BooguFloatLoadPolicy::AdaptToF32,
                BooguQuantizedLoadPolicy::Preserve,
                BooguDenoiserRuntimeQuantizationPolicy::Q8sBlock32F32,
                scope,
                BooguVariant::Image01EditTurbo1k5,
                "boogu-prelude",
            ),
        ] {
            assert!(result.is_err());
        }
    }

    #[test]
    fn canonical_release_identity_correctness() {
        BooguReleaseIdentity::canonical(BooguVariant::Image01Turbo)
            .validate()
            .unwrap();
        BooguReleaseIdentity::canonical(BooguVariant::Image01EditTurbo)
            .validate()
            .unwrap();
        let one_k_five = BooguReleaseIdentity::canonical(BooguVariant::Image01EditTurbo1k5);
        one_k_five.validate().unwrap();
        assert_eq!(one_k_five.model_revision, EDIT_TURBO_1K5_REVISION);
    }

    #[cfg(feature = "burnpack")]
    #[test]
    fn edit_turbo_1k5_release_artifact_digest_is_exact_correctness() {
        let expected =
            Sha256Digest::from_hex(EDIT_TURBO_1K5_F16_QWEN_VISION_F32_CONTENT_DIGEST).unwrap();
        validate_edit_turbo_1k5_release_artifact_digest(expected).unwrap();
        validate_edit_turbo_1k5_release_artifact_digest(
            Sha256Digest::from_hex(LEGACY_EDIT_TURBO_1K5_F16_QWEN_VISION_F32_CONTENT_DIGEST)
                .unwrap(),
        )
        .unwrap();
        assert!(
            validate_edit_turbo_1k5_release_artifact_digest(Sha256Digest::calculate(b"other"))
                .is_err()
        );
    }

    #[cfg(feature = "burnpack")]
    #[test]
    fn canonical_published_bundle_matrix_is_exact_correctness() {
        use std::collections::BTreeSet;

        assert_eq!(PUBLISHED_ARTIFACT_BUNDLES.len(), 3);
        let bundle_ids = PUBLISHED_ARTIFACT_BUNDLES
            .iter()
            .map(|bundle| bundle.bundle_id)
            .collect::<BTreeSet<_>>();
        let digests = PUBLISHED_ARTIFACT_BUNDLES
            .iter()
            .map(|bundle| bundle.content_digest)
            .collect::<BTreeSet<_>>();
        assert_eq!(bundle_ids.len(), PUBLISHED_ARTIFACT_BUNDLES.len());
        assert_eq!(digests.len(), PUBLISHED_ARTIFACT_BUNDLES.len());
        assert_eq!(
            bundle_ids,
            BTreeSet::from([
                "boogu-image-0.1-turbo",
                "boogu-image-0.1-edit-turbo",
                "boogu-image-0.1-edit-turbo-1k5",
            ])
        );

        for bundle in PUBLISHED_ARTIFACT_BUNDLES {
            assert_eq!(bundle.converter_version, PUBLISHED_BUNDLE_CONVERTER_VERSION);
            let parsed = Sha256Digest::from_hex(bundle.content_digest).unwrap();
            assert_eq!(
                canonical_published_bundle(bundle.variant, bundle.profile),
                Some(bundle)
            );
            validate_canonical_release_artifact_digest(bundle.variant, bundle.profile, parsed)
                .unwrap();
        }

        for diagnostic in [
            BooguStorageProfile::F16,
            BooguStorageProfile::Q8sBlock32F32,
            BooguStorageProfile::Q8sBlock32F32QwenVisionF32,
        ] {
            assert!(canonical_published_bundle(BooguVariant::Image01Turbo, diagnostic).is_none());
            let error = validate_canonical_release_artifact_digest(
                BooguVariant::Image01Turbo,
                diagnostic,
                Sha256Digest::calculate(b"diagnostic"),
            )
            .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("explicit local or custom remote source")
            );
        }

        assert_eq!(
            preferred_artifact_bundle_id(
                BooguVariant::Image01Turbo,
                BooguStorageProfile::F16QwenVisionF32,
            ),
            "boogu-image-0.1-turbo"
        );
        assert_eq!(
            preferred_artifact_bundle_id(
                BooguVariant::Image01Turbo,
                BooguStorageProfile::Q8sBlock32F32QwenVisionF32,
            ),
            "boogu-image-0.1-turbo-q8s-block32-f32-qwen-vision-f32"
        );
        assert_eq!(
            promotable_legacy_artifact_digest(
                BooguVariant::Image01Turbo,
                BooguStorageProfile::F16QwenVisionF32,
            ),
            Some(LEGACY_TURBO_F16_QWEN_VISION_F32_CONTENT_DIGEST)
        );
        assert_eq!(
            promotable_legacy_artifact_digest(
                BooguVariant::Image01Turbo,
                BooguStorageProfile::Q8sBlock32F32QwenVisionF32,
            ),
            None
        );
        assert!(artifact_bundle_id_is_compatible(
            BooguVariant::Image01Turbo,
            BooguStorageProfile::F16QwenVisionF32,
            "boogu-image-0.1-turbo",
        ));
        assert!(artifact_bundle_id_is_compatible(
            BooguVariant::Image01Turbo,
            BooguStorageProfile::F16QwenVisionF32,
            "boogu-image-0.1-turbo-f16-qwen-vision-f32",
        ));
        assert!(!artifact_bundle_id_is_compatible(
            BooguVariant::Image01Turbo,
            BooguStorageProfile::F16QwenVisionF32,
            "boogu-image-0.1-turbo-arbitrary",
        ));
        assert!(!artifact_bundle_id_is_compatible(
            BooguVariant::Image01Turbo,
            BooguStorageProfile::Q8sBlock32F32QwenVisionF32,
            "boogu-image-0.1-turbo",
        ));
    }

    #[cfg(feature = "burnpack")]
    #[test]
    fn canonical_published_bundle_rejects_wrong_digest_correctness() {
        let error = validate_canonical_release_artifact_digest(
            BooguVariant::Image01Turbo,
            BooguStorageProfile::F16QwenVisionF32,
            Sha256Digest::calculate(b"wrong published bundle"),
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains(TURBO_F16_QWEN_VISION_F32_CONTENT_DIGEST)
        );
    }

    #[cfg(feature = "burnpack")]
    #[test]
    fn release_converter_compatibility_is_explicit_correctness() {
        validate_supported_bundle_converter_version(PUBLISHED_BUNDLE_CONVERTER_VERSION).unwrap();
        validate_supported_bundle_converter_version(CURRENT_BUNDLE_CONVERTER_VERSION).unwrap();
        let error = validate_supported_bundle_converter_version("0.0.0-unknown").unwrap_err();
        assert!(error.to_string().contains("unsupported release converter"));
    }

    #[cfg(feature = "burnpack")]
    #[test]
    fn strict_verifier_pins_published_but_allows_diagnostic_digest_correctness() {
        let published = canonical_published_bundle(
            BooguVariant::Image01Turbo,
            BooguStorageProfile::F16QwenVisionF32,
        )
        .unwrap();
        loading::validate_published_release_content_digest(
            published.variant,
            published.profile,
            Some(Sha256Digest::from_hex(published.content_digest).unwrap()),
        )
        .unwrap();
        assert!(
            loading::validate_published_release_content_digest(
                published.variant,
                published.profile,
                Some(Sha256Digest::calculate(b"wrong")),
            )
            .is_err()
        );
        assert!(
            loading::validate_published_release_content_digest(
                published.variant,
                published.profile,
                None,
            )
            .is_err()
        );
        loading::validate_published_release_content_digest(
            BooguVariant::Image01Turbo,
            BooguStorageProfile::F16,
            Some(Sha256Digest::calculate(b"explicit diagnostic")),
        )
        .unwrap();
    }

    #[cfg(feature = "burnpack")]
    #[test]
    fn verified_artifact_directory_authenticates_metadata_files_correctness() {
        use burn_image::{
            ARTIFACT_MANIFEST_SCHEMA_VERSION, ArtifactBundleId, ArtifactFileRole, ArtifactPath,
            ArtifactProfileId, ModelId, NumericFormat, Sha256Digest,
        };

        let directory = tempfile::tempdir().unwrap();
        let relative = "metadata/source/mllm/config.json";
        let bytes = br#"{"model_type":"qwen3_vl"}"#;
        std::fs::create_dir_all(directory.path().join("metadata/source/mllm")).unwrap();
        std::fs::write(directory.path().join(relative), bytes).unwrap();
        let mut manifest = ArtifactManifest {
            schema_version: ARTIFACT_MANIFEST_SCHEMA_VERSION,
            bundle: ArtifactBundleId::new("verified-directory-test").unwrap(),
            profile: ArtifactProfileId::new("f16").unwrap(),
            model: ModelId::new("Boogu/Boogu-Image-0.1-Turbo").unwrap(),
            model_revision: TURBO_REVISION.into(),
            numeric_format: NumericFormat::F16,
            components: Vec::new(),
            files: vec![ArtifactFile {
                path: ArtifactPath::new(relative).unwrap(),
                size: bytes.len() as u64,
                sha256: Sha256Digest::calculate(bytes),
                role: ArtifactFileRole::Config,
                component: None,
                shard: None,
            }],
            dependencies: Vec::new(),
            metadata: BTreeMap::new(),
            content_digest: None,
        };
        manifest.seal().unwrap();
        std::fs::write(
            directory.path().join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();

        let verified = VerifiedArtifactDirectory::open(directory.path()).unwrap();
        assert_eq!(verified.read_file(relative).unwrap(), bytes);
        std::fs::write(directory.path().join(relative), b"tampered").unwrap();
        assert!(verified.read_file(relative).is_err());
    }

    #[cfg(feature = "burnpack")]
    fn tiny_qwen_config() -> Qwen3VlConfig {
        Qwen3VlConfig::from_json(
            r#"{
                "text_config": {
                    "vocab_size": 64, "hidden_size": 8, "intermediate_size": 16,
                    "num_hidden_layers": 1, "num_attention_heads": 2,
                    "num_key_value_heads": 1, "head_dim": 4,
                    "rope_scaling": {"mrope_section": [2, 0, 0], "mrope_interleaved": true}
                },
                "vision_config": {
                    "depth": 1, "hidden_size": 8, "intermediate_size": 16,
                    "num_heads": 2, "patch_size": 2, "temporal_patch_size": 1,
                    "spatial_merge_size": 2, "out_hidden_size": 8,
                    "num_position_embeddings": 4, "deepstack_visual_indexes": [0]
                },
                "tie_word_embeddings": false,
                "image_token_id": 60, "video_token_id": 61,
                "vision_start_token_id": 62, "vision_end_token_id": 63
            }"#,
        )
        .unwrap()
    }

    #[cfg(feature = "burnpack")]
    #[derive(Default)]
    struct AsyncMemoryShardReader {
        sealed: BTreeMap<String, ArtifactFile>,
        objects: BTreeMap<String, Vec<u8>>,
        requests: Vec<(String, u64)>,
        largest_response: usize,
        corrupt_path: Option<String>,
        append_byte_path: Option<String>,
        oversize_response: bool,
    }

    #[cfg(feature = "burnpack")]
    impl AsyncMemoryShardReader {
        fn from_directory(directory: &tempfile::TempDir, manifest: &ArtifactManifest) -> Self {
            let sealed = manifest
                .files
                .iter()
                .map(|file| (file.path.as_str().to_owned(), file.clone()))
                .collect::<BTreeMap<_, _>>();
            let objects = manifest
                .files
                .iter()
                .map(|file| {
                    (
                        file.path.as_str().to_owned(),
                        std::fs::read(directory.path().join(file.path.as_str())).unwrap(),
                    )
                })
                .collect::<BTreeMap<_, _>>();
            Self {
                sealed,
                objects,
                ..Self::default()
            }
        }
    }

    #[cfg(feature = "burnpack")]
    impl AsyncStageShardReader for AsyncMemoryShardReader {
        async fn read_shard(
            &mut self,
            file: &ArtifactFile,
            max_bytes: u64,
        ) -> Result<Vec<u8>, BooguError> {
            let expected = self.sealed.get(file.path.as_str()).ok_or_else(|| {
                BooguError::Artifact(format!("unsealed request for {}", file.path))
            })?;
            if expected != file {
                return Err(BooguError::Artifact(format!(
                    "request identity differs from sealed file {}",
                    file.path
                )));
            }
            self.requests
                .push((file.path.as_str().to_owned(), max_bytes));
            if self.oversize_response {
                let length = usize::try_from(max_bytes)
                    .ok()
                    .and_then(|value| value.checked_add(1))
                    .ok_or_else(|| {
                        BooguError::Artifact("test byte cap is not addressable".into())
                    })?;
                let bytes = vec![0; length];
                self.largest_response = self.largest_response.max(bytes.len());
                return Ok(bytes);
            }
            let mut bytes = self
                .objects
                .get(file.path.as_str())
                .cloned()
                .ok_or_else(|| BooguError::Artifact(format!("missing {}", file.path)))?;
            if self.corrupt_path.as_deref() == Some(file.path.as_str()) && !bytes.is_empty() {
                bytes[0] ^= 0x80;
            }
            if self.append_byte_path.as_deref() == Some(file.path.as_str()) {
                bytes.push(0);
            }
            self.largest_response = self.largest_response.max(bytes.len());
            Ok(bytes)
        }
    }

    #[cfg(feature = "burnpack")]
    fn write_tiny_float_artifact(
        inventory: &BooguArtifactInventory,
        snapshots: Vec<burn_store::TensorSnapshot>,
        profile: BooguStorageProfile,
    ) -> (tempfile::TempDir, burn_image::ArtifactManifest) {
        use burn::{module::ParamId, tensor::DType};
        use burn_image::{
            ARTIFACT_MANIFEST_SCHEMA_V1, ArtifactBundleId, ArtifactComponent, ArtifactComponentId,
            ArtifactFile, ArtifactFileRole, ArtifactManifest, ArtifactPath, ArtifactProfileId,
            ModelId, NumericFormat, Sha256Digest,
        };
        use burn_store::{BurnpackWriter, TensorSnapshot};

        let (profile_name, numeric_format) = match profile {
            BooguStorageProfile::F16 => ("f16", NumericFormat::F16),
            BooguStorageProfile::F16QwenVisionF32 => (
                "f16-qwen-vision-f32",
                NumericFormat::Other("f16-qwen-vision-f32".into()),
            ),
            other => panic!("tiny float fixture does not implement {other:?}"),
        };
        let stored_dtype = |spec: &ArtifactTensorSpec| {
            if profile == BooguStorageProfile::F16QwenVisionF32
                && spec.owner == TensorOwner::Qwen3Vl
                && spec.stage.starts_with("qwen-vision-")
            {
                DType::F32
            } else {
                DType::F16
            }
        };

        let directory = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(directory.path().join("objects")).unwrap();
        std::fs::create_dir_all(directory.path().join("metadata")).unwrap();
        let source = snapshots
            .into_iter()
            .map(|snapshot| (snapshot.full_path(), snapshot))
            .collect::<BTreeMap<_, _>>();
        let mut by_stage = BTreeMap::<String, Vec<TensorSnapshot>>::new();
        let mut stored_digests = BTreeMap::new();
        for spec in inventory.tensors() {
            let original = source.get(&spec.target_name).unwrap();
            let data = original
                .to_data()
                .unwrap()
                .convert_dtype(stored_dtype(spec));
            stored_digests.insert(
                spec.target_name.clone(),
                Sha256Digest::calculate(data.bytes.as_ref()),
            );
            by_stage
                .entry(spec.stage.clone())
                .or_default()
                .push(TensorSnapshot::from_data(
                    data,
                    vec![spec.target_name.clone()],
                    Vec::new(),
                    ParamId::new(),
                ));
        }
        let mut files = Vec::new();
        let mut components = Vec::new();
        for (stage, snapshots) in by_stage {
            let bytes = BurnpackWriter::new(snapshots).to_bytes().unwrap().to_vec();
            let path = format!("objects/{stage}.bpk");
            std::fs::write(directory.path().join(&path), &bytes).unwrap();
            files.push(ArtifactFile {
                path: ArtifactPath::new(&path).unwrap(),
                size: bytes.len() as u64,
                sha256: Sha256Digest::calculate(&bytes),
                role: ArtifactFileRole::Weights,
                component: Some(ArtifactComponentId::new(&stage).unwrap()),
                shard: None,
            });
            components.push(ArtifactComponent {
                id: ArtifactComponentId::new(&stage).unwrap(),
                required: true,
            });
        }
        let inventory_bytes = serde_json::to_vec(
            &inventory
                .tensors()
                .iter()
                .map(|spec| {
                    let dtype = stored_dtype(spec);
                    let object = files
                        .iter()
                        .find(|file| {
                            file.role == ArtifactFileRole::Weights
                                && file.component.as_ref().map(|value| value.as_str())
                                    == Some(spec.stage.as_str())
                        })
                        .unwrap();
                    let element_bytes = match spec.source_dtype {
                        SourceDType::Bf16 => 2_u64,
                        SourceDType::F32 => 4_u64,
                    };
                    serde_json::json!({
                        "source_name": spec.source_name,
                        "target_name": spec.target_name,
                        "owner": spec.owner,
                        "component": spec.source_component,
                        "stage": spec.stage,
                        "transform": spec.transform,
                        "source_file": format!("{}/fixture.safetensors", spec.source_component),
                        "source_dtype": spec.source_dtype.safetensors_name(),
                        "source_shape": spec.source_shape,
                        "stored_dtype": if dtype == DType::F32 { "f32" } else { "f16" },
                        "stored_shape": spec.target_shape,
                        "source_offset": 0,
                        "source_bytes": spec.source_shape.iter().product::<usize>() as u64 * element_bytes,
                        "quantized": false,
                        "stored_sha256": stored_digests[&spec.target_name],
                        "burnpack_object": object.path.as_str(),
                    })
                })
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let inventory_path = "metadata/tensor-inventory.json";
        std::fs::write(directory.path().join(inventory_path), &inventory_bytes).unwrap();
        files.push(ArtifactFile {
            path: ArtifactPath::new(inventory_path).unwrap(),
            size: inventory_bytes.len() as u64,
            sha256: Sha256Digest::calculate(&inventory_bytes),
            role: ArtifactFileRole::Metadata,
            component: None,
            shard: None,
        });
        let source_records = inventory
            .tensors()
            .iter()
            .map(|spec| spec.source_component.as_str())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(|component| {
                serde_json::json!({
                    "path": format!("{component}/fixture.safetensors"),
                    "size": 1024_u64 * 1024 * 1024,
                    "sha256": Sha256Digest::calculate(component.as_bytes()),
                })
            })
            .collect::<Vec<_>>();
        let source_bytes = serde_json::to_vec(&source_records).unwrap();
        let source_path = "metadata/source-files.json";
        std::fs::write(directory.path().join(source_path), &source_bytes).unwrap();
        files.push(ArtifactFile {
            path: ArtifactPath::new(source_path).unwrap(),
            size: source_bytes.len() as u64,
            sha256: Sha256Digest::calculate(&source_bytes),
            role: ArtifactFileRole::Metadata,
            component: None,
            shard: None,
        });
        let mut metadata = BTreeMap::new();
        metadata.insert("source_revision".into(), UPSTREAM_SOURCE_REVISION.into());
        metadata.insert("algorithm".into(), "dmd-turbo".into());
        metadata.insert("artifact_layout".into(), "semantic-burnpack-v1".into());
        metadata.insert("tensor_inventory_schema".into(), "2".into());
        metadata.insert("layout_contract".into(), inventory_path.into());
        metadata.insert(
            "conversion_crate".into(),
            CURRENT_BUNDLE_CONVERTER_VERSION.into(),
        );
        metadata.insert("profile".into(), profile_name.into());
        metadata.insert("tensor_count".into(), inventory.tensors().len().to_string());
        metadata.insert(
            "stored_tensor_count".into(),
            inventory.tensors().len().to_string(),
        );
        metadata.insert("omitted_tensor_count".into(), "0".into());
        metadata.insert(
            "target_max_shard_bytes".into(),
            (64 * 1024 * 1024).to_string(),
        );
        metadata.insert("physical_shards_bounded".into(), "true".into());
        let mut manifest = ArtifactManifest {
            schema_version: ARTIFACT_MANIFEST_SCHEMA_V1,
            bundle: ArtifactBundleId::new("tiny-component-test").unwrap(),
            profile: ArtifactProfileId::new(profile_name).unwrap(),
            model: ModelId::new("Boogu/Boogu-Image-0.1-Turbo").unwrap(),
            model_revision: TURBO_REVISION.into(),
            numeric_format,
            components,
            files,
            dependencies: Vec::new(),
            metadata,
            content_digest: None,
        };
        manifest.seal().unwrap();
        std::fs::write(
            directory.path().join("manifest.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        (directory, manifest)
    }

    #[cfg(feature = "burnpack")]
    #[test]
    fn release_burnpacks_match_sealed_tensor_contract_correctness() {
        use burn::backend::NdArray;
        use burn_store::ModuleSnapshot;

        let config = tiny_qwen_config();
        let device = Default::default();
        let model = burn_qwen3_vl::Qwen3VlForConditionalGeneration::<NdArray<f32>>::new(
            config.clone(),
            &device,
        )
        .unwrap();
        let inventory = BooguArtifactInventory {
            tensors: qwen_specs(&config),
        };
        let (directory, manifest) = write_tiny_float_artifact(
            &inventory,
            model.collect(None, None, false),
            BooguStorageProfile::F16,
        );
        let mut reader = DirectoryStageShardReader::new(directory.path());
        let entries = loading::verify_inventory_contract(
            &manifest,
            &inventory,
            BooguStorageProfile::F16,
            &mut reader,
        )
        .unwrap();
        let (objects, tensors, largest) =
            loading::verify_release_burnpacks(&manifest, &entries, &mut reader).unwrap();
        assert_eq!(objects, manifest.components.len());
        assert_eq!(tensors, inventory.tensors().len());
        assert!(largest > 0);
    }

    #[cfg(feature = "burnpack")]
    #[test]
    fn release_burnpacks_reject_tampered_tensor_digest_correctness() {
        use burn::backend::NdArray;
        use burn_image::Sha256Digest;
        use burn_store::ModuleSnapshot;

        let config = tiny_qwen_config();
        let device = Default::default();
        let model = burn_qwen3_vl::Qwen3VlForConditionalGeneration::<NdArray<f32>>::new(
            config.clone(),
            &device,
        )
        .unwrap();
        let inventory = BooguArtifactInventory {
            tensors: qwen_specs(&config),
        };
        let (directory, mut manifest) = write_tiny_float_artifact(
            &inventory,
            model.collect(None, None, false),
            BooguStorageProfile::F16,
        );
        let inventory_path = directory.path().join("metadata/tensor-inventory.json");
        let mut serialized: Vec<serde_json::Value> =
            serde_json::from_slice(&std::fs::read(&inventory_path).unwrap()).unwrap();
        serialized[0]["stored_sha256"] =
            serde_json::json!(Sha256Digest::calculate(b"tampered tensor digest"));
        let bytes = serde_json::to_vec(&serialized).unwrap();
        std::fs::write(&inventory_path, &bytes).unwrap();
        let inventory_file = manifest
            .files
            .iter_mut()
            .find(|file| file.path.as_str() == "metadata/tensor-inventory.json")
            .unwrap();
        inventory_file.size = bytes.len() as u64;
        inventory_file.sha256 = Sha256Digest::calculate(&bytes);
        manifest.content_digest = None;
        manifest.seal().unwrap();

        let mut reader = DirectoryStageShardReader::new(directory.path());
        let entries = loading::verify_inventory_contract(
            &manifest,
            &inventory,
            BooguStorageProfile::F16,
            &mut reader,
        )
        .unwrap();
        let error =
            loading::verify_release_burnpacks(&manifest, &entries, &mut reader).unwrap_err();
        assert!(error.to_string().contains("payload digest"));
    }

    #[cfg(feature = "burnpack")]
    fn rewrite_tiny_qwen_vocabulary_as_rows(
        directory: &tempfile::TempDir,
        mut manifest: burn_image::ArtifactManifest,
        inventory: &BooguArtifactInventory,
        snapshots: &[burn_store::TensorSnapshot],
        plan: &burn_qwen3_vl::Qwen3VlStreamingPlan,
    ) -> burn_image::ArtifactManifest {
        use burn::{module::ParamId, tensor::DType};
        use burn_image::{
            ArtifactComponent, ArtifactComponentId, ArtifactFile, ArtifactFileRole, ArtifactPath,
            Sha256Digest,
        };
        use burn_store::{BurnpackWriter, TensorSnapshot};

        let removed_stages = BTreeSet::from(["qwen-embedding", "qwen-lm-head"]);
        for file in manifest.files.iter().filter(|file| {
            file.component
                .as_ref()
                .is_some_and(|stage| removed_stages.contains(stage.as_str()))
        }) {
            std::fs::remove_file(directory.path().join(file.path.as_str())).unwrap();
        }
        manifest.files.retain(|file| {
            !file
                .component
                .as_ref()
                .is_some_and(|stage| removed_stages.contains(stage.as_str()))
        });
        manifest
            .components
            .retain(|component| !removed_stages.contains(component.id.as_str()));

        let embed_spec = inventory
            .by_source("mllm", "model.language_model.embed_tokens.weight")
            .unwrap();
        let lm_head_spec = inventory.by_source("mllm", "lm_head.weight").unwrap();
        let embed_data = snapshots
            .iter()
            .find(|snapshot| snapshot.full_path() == embed_spec.target_name)
            .unwrap()
            .to_data()
            .unwrap()
            .convert_dtype(DType::F16);
        let embed_bytes: &[u8] = embed_data.bytes.as_ref();
        let row_bytes = embed_spec.source_shape[1] * 2;
        let mut serialized: Vec<serde_json::Value> = serde_json::from_slice(
            &std::fs::read(directory.path().join("metadata/tensor-inventory.json")).unwrap(),
        )
        .unwrap();
        serialized.retain(|entry| {
            !matches!(
                entry["source_name"].as_str(),
                Some("model.language_model.embed_tokens.weight" | "lm_head.weight")
            )
        });
        for chunk in &plan.embedding_rows.chunks {
            let start = chunk.row_range.start * row_bytes;
            let end = chunk.row_range.end * row_bytes;
            let bytes = embed_bytes[start..end].to_vec();
            let target = qwen_row_slice_target(&embed_spec.target_name, chunk);
            let stage = qwen_streaming_stage_name(&Qwen3VlStage::EmbeddingRows {
                chunk: chunk.chunk_index,
            });
            let snapshot = TensorSnapshot::from_data(
                burn::tensor::TensorData::from_bytes_vec(
                    bytes.clone(),
                    vec![chunk.rows(), chunk.hidden_size],
                    DType::F16,
                ),
                vec![target.clone()],
                Vec::new(),
                ParamId::new(),
            );
            let burnpack = BurnpackWriter::new(vec![snapshot])
                .to_bytes()
                .unwrap()
                .to_vec();
            let path = format!("objects/{stage}.bpk");
            std::fs::write(directory.path().join(&path), &burnpack).unwrap();
            manifest.files.push(ArtifactFile {
                path: ArtifactPath::new(&path).unwrap(),
                size: burnpack.len() as u64,
                sha256: Sha256Digest::calculate(&burnpack),
                role: ArtifactFileRole::Weights,
                component: Some(ArtifactComponentId::new(&stage).unwrap()),
                shard: None,
            });
            manifest.components.push(ArtifactComponent {
                id: ArtifactComponentId::new(&stage).unwrap(),
                required: true,
            });
            serialized.push(serde_json::json!({
                "source_name": embed_spec.source_name,
                "logical_target_name": embed_spec.target_name,
                "target_name": target,
                "owner": embed_spec.owner,
                "component": embed_spec.source_component,
                "stage": stage,
                "transform": embed_spec.transform,
                "source_file": "mllm/fixture.safetensors",
                "source_dtype": embed_spec.source_dtype.safetensors_name(),
                "source_shape": embed_spec.source_shape,
                "source_row_range": [chunk.row_range.start, chunk.row_range.end],
                "included": true,
                "stored_dtype": "f16",
                "stored_shape": [chunk.rows(), chunk.hidden_size],
                "source_offset": 4096 + start as u64,
                "source_bytes": bytes.len() as u64,
                "quantized": false,
                "stored_sha256": Sha256Digest::calculate(&bytes),
                "burnpack_object": path,
            }));
        }
        serialized.push(serde_json::json!({
            "source_name": lm_head_spec.source_name,
            "logical_target_name": lm_head_spec.target_name,
            "target_name": lm_head_spec.target_name,
            "owner": lm_head_spec.owner,
            "component": lm_head_spec.source_component,
            "stage": lm_head_spec.stage,
            "transform": lm_head_spec.transform,
            "source_file": "mllm/fixture.safetensors",
            "source_dtype": lm_head_spec.source_dtype.safetensors_name(),
            "source_shape": lm_head_spec.source_shape,
            "source_row_range": null,
            "included": false,
            "stored_dtype": null,
            "stored_shape": null,
            "source_offset": 0,
            "source_bytes": lm_head_spec.source_shape.iter().product::<usize>() as u64 * 2,
            "quantized": false,
            "stored_sha256": null,
            "burnpack_object": null,
        }));
        serialized.sort_by(|left, right| {
            (
                left["component"].as_str(),
                left["source_name"].as_str(),
                left["source_row_range"].to_string(),
            )
                .cmp(&(
                    right["component"].as_str(),
                    right["source_name"].as_str(),
                    right["source_row_range"].to_string(),
                ))
        });
        let bytes = serde_json::to_vec(&serialized).unwrap();
        std::fs::write(
            directory.path().join("metadata/tensor-inventory.json"),
            &bytes,
        )
        .unwrap();
        let inventory_file = manifest
            .files
            .iter_mut()
            .find(|file| file.path.as_str() == "metadata/tensor-inventory.json")
            .unwrap();
        inventory_file.size = bytes.len() as u64;
        inventory_file.sha256 = Sha256Digest::calculate(&bytes);
        manifest.metadata.insert(
            "stored_tensor_count".into(),
            (inventory.tensors().len() - 2 + plan.embedding_rows.chunks.len()).to_string(),
        );
        manifest
            .metadata
            .insert("omitted_tensor_count".into(), "1".into());
        manifest.metadata.insert(
            "qwen_embedding_row_chunks".into(),
            plan.embedding_rows.chunks.len().to_string(),
        );
        manifest
            .metadata
            .insert("qwen_lm_head".into(), "omitted-base-model".into());
        manifest
            .files
            .sort_by(|left, right| left.path.cmp(&right.path));
        manifest
            .components
            .sort_by(|left, right| left.id.cmp(&right.id));
        manifest.content_digest = None;
        manifest.seal().unwrap();
        std::fs::write(
            directory.path().join("manifest.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        manifest
    }

    #[cfg(feature = "burnpack")]
    #[test]
    fn denoiser_inventory_matches_burn_record_paths_correctness() {
        use burn::backend::NdArray;
        use burn_store::ModuleSnapshot;

        let config = BooguConfig {
            patch_size: 2,
            in_channels: 4,
            out_channels: 4,
            hidden_size: 8,
            num_layers: 2,
            num_double_stream_layers: 1,
            num_refiner_layers: 1,
            num_attention_heads: 2,
            num_kv_heads: 1,
            multiple_of: 8,
            norm_eps: 1.0e-5,
            axes_dim_rope: [2, 2, 0],
            axes_lens: [16, 16, 16],
            instruction_feature_dim: 8,
            timestep_scale: 1000.0,
        };
        let device = Default::default();
        let model = crate::BooguDenoiser::<NdArray<f32>>::new(config.clone(), &device).unwrap();
        let actual = model
            .collect(None, None, false)
            .into_iter()
            .map(|snapshot| (snapshot.full_path(), snapshot.shape.to_vec()))
            .collect::<BTreeMap<_, _>>();
        let expected = boogu_specs(&config)
            .into_iter()
            .map(|spec| (spec.target_name, spec.target_shape))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(actual, expected);
    }

    #[cfg(feature = "burnpack")]
    #[test]
    fn qwen_inventory_matches_burn_record_paths_correctness() {
        use burn::backend::NdArray;
        use burn_store::ModuleSnapshot;

        let config = tiny_qwen_config();
        let device = Default::default();
        let model = burn_qwen3_vl::Qwen3VlForConditionalGeneration::<NdArray<f32>>::new(
            config.clone(),
            &device,
        )
        .unwrap();
        let actual = model
            .collect(None, None, false)
            .into_iter()
            .map(|snapshot| (snapshot.full_path(), snapshot.shape.to_vec()))
            .collect::<BTreeMap<_, _>>();
        let expected = qwen_specs(&config)
            .into_iter()
            .map(|spec| (spec.target_name, spec.target_shape))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(actual, expected);
    }

    #[cfg(feature = "burnpack")]
    #[test]
    fn lazy_col_linear_raw_burnpack_forward_correctness() {
        use burn::{
            backend::NdArray,
            module::ParamId,
            nn::{LinearConfig, LinearLayout},
            tensor::{Tensor, TensorData},
        };
        use burn_store::{
            BurnpackStore, BurnpackWriter, ModuleSnapshot, ModuleStore, TensorSnapshot,
        };

        let device = Default::default();
        let mut linear = LinearConfig::new(2, 3)
            .with_bias(false)
            .with_layout(LinearLayout::Col)
            .init::<NdArray<f32>>(&device);
        // PyTorch/saved Col layout is [out, in]. Do not inspect or collect the lazy target first.
        let raw = TensorSnapshot::from_data(
            TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]),
            vec!["weight".into()],
            Vec::new(),
            ParamId::new(),
        );
        let bytes = BurnpackWriter::new(vec![raw]).to_bytes().unwrap();
        let mut store = BurnpackStore::from_bytes(Some(bytes));
        let snapshots = store
            .get_all_snapshots()
            .unwrap()
            .values()
            .cloned()
            .collect();
        let result = linear.apply(snapshots, None, None, false);
        assert_eq!(result.applied, ["weight"]);
        assert!(result.missing.is_empty());
        assert!(result.unused.is_empty());
        assert!(result.errors.is_empty());

        let output = linear
            .forward(Tensor::<NdArray<f32>, 2>::from_data(
                TensorData::new(vec![10.0_f32, 20.0], [1, 2]),
                &device,
            ))
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(output, vec![50.0, 110.0, 170.0]);
    }

    #[cfg(feature = "burnpack")]
    #[test]
    fn verified_qwen_stage_source_loads_rows_and_every_module_correctness() {
        use burn::backend::NdArray;
        use burn_qwen3_vl::{Qwen3VlStageSource, Qwen3VlStreamingPlan, RowChunkPlan};
        use burn_store::ModuleSnapshot;

        let config = tiny_qwen_config();
        let device = Default::default();
        let resident = burn_qwen3_vl::Qwen3VlForConditionalGeneration::<NdArray<f32>>::new(
            config.clone(),
            &device,
        )
        .unwrap();
        let snapshots = resident.collect(None, None, false);
        let inventory = BooguArtifactInventory {
            tensors: qwen_specs(&config),
        };
        let (directory, manifest) = write_tiny_float_artifact(
            &inventory,
            snapshots.clone(),
            BooguStorageProfile::F16QwenVisionF32,
        );
        let rows = RowChunkPlan::even(
            config.text_config.vocab_size,
            config.text_config.hidden_size,
            3,
            2,
        )
        .unwrap();
        let plan = Qwen3VlStreamingPlan::new(&config, rows, None).unwrap();
        let manifest = rewrite_tiny_qwen_vocabulary_as_rows(
            &directory, manifest, &inventory, &snapshots, &plan,
        );
        assert!(
            manifest
                .files
                .iter()
                .filter(|file| file.role == burn_image::ArtifactFileRole::Weights)
                .all(|file| file.size <= 64 * 1024 * 1024)
        );

        let mut source = VerifiedBurnpackQwenStageSource::<NdArray<f32>, _>::from_directory_auto(
            &BooguReleaseIdentity::canonical(BooguVariant::Image01Turbo),
            directory.path(),
            inventory,
            config,
            BooguStorageProfile::F16QwenVisionF32,
            device,
        )
        .unwrap()
        .with_float_load_policy(BooguFloatLoadPolicy::AdaptToF32);
        assert_eq!(source.plan(), &plan);
        let mut loaded_rows = Vec::new();
        for row in &plan.embedding_rows.chunks {
            let chunk = source.load_embedding_rows(row).unwrap();
            loaded_rows.extend(chunk.weight.into_data().to_vec::<f32>().unwrap());
        }
        let expected = snapshots
            .iter()
            .find(|snapshot| snapshot.full_path() == "model.language_model.embed_tokens.weight")
            .unwrap()
            .to_data()
            .unwrap()
            .convert_dtype(burn::tensor::DType::F16)
            .convert_dtype(burn::tensor::DType::F32)
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(loaded_rows, expected);
        source.load_vision_prelude().unwrap();
        source.load_vision_block(0).unwrap();
        source.load_vision_deepstack_merger(0).unwrap();
        source.load_vision_final_merger().unwrap();
        source.load_text_block(0).unwrap();
        source.load_text_final_norm().unwrap();
        source.synchronize().unwrap();
    }

    #[cfg(feature = "burnpack")]
    #[test]
    fn async_qwen_stage_source_verifies_bounded_memory_reader_correctness() {
        use burn::backend::NdArray;
        use burn_qwen3_vl::{AsyncQwen3VlStageSource, Qwen3VlStreamingPlan, RowChunkPlan};
        use burn_store::ModuleSnapshot;
        use futures::executor::block_on;

        let config = tiny_qwen_config();
        let device = Default::default();
        let resident = burn_qwen3_vl::Qwen3VlForConditionalGeneration::<NdArray<f32>>::new(
            config.clone(),
            &device,
        )
        .unwrap();
        let snapshots = resident.collect(None, None, false);
        let inventory = BooguArtifactInventory {
            tensors: qwen_specs(&config),
        };
        let (directory, manifest) = write_tiny_float_artifact(
            &inventory,
            snapshots.clone(),
            BooguStorageProfile::F16QwenVisionF32,
        );
        let rows = RowChunkPlan::even(
            config.text_config.vocab_size,
            config.text_config.hidden_size,
            3,
            2,
        )
        .unwrap();
        let plan = Qwen3VlStreamingPlan::new(&config, rows, None).unwrap();
        let manifest = rewrite_tiny_qwen_vocabulary_as_rows(
            &directory, manifest, &inventory, &snapshots, &plan,
        );
        let reader = AsyncMemoryShardReader::from_directory(&directory, &manifest);
        let mut source = block_on(
            VerifiedAsyncBurnpackQwenStageSource::<NdArray<f32>, _>::new_auto(
                &BooguReleaseIdentity::canonical(BooguVariant::Image01Turbo),
                manifest.clone(),
                inventory,
                config,
                BooguStorageProfile::F16QwenVisionF32,
                device,
                reader,
            ),
        )
        .unwrap()
        .with_float_load_policy(BooguFloatLoadPolicy::AdaptToF32);
        assert_eq!(source.plan(), &plan);
        let max_bytes = source.max_shard_bytes();

        block_on(async {
            let mut loaded_rows = Vec::new();
            for row in &plan.embedding_rows.chunks {
                let chunk = source.load_embedding_rows(row).await.unwrap();
                loaded_rows.extend(chunk.weight.into_data().to_vec::<f32>().unwrap());
            }
            let expected = snapshots
                .iter()
                .find(|snapshot| snapshot.full_path() == "model.language_model.embed_tokens.weight")
                .unwrap()
                .to_data()
                .unwrap()
                .convert_dtype(burn::tensor::DType::F16)
                .convert_dtype(burn::tensor::DType::F32)
                .to_vec::<f32>()
                .unwrap();
            assert_eq!(loaded_rows, expected);
            source.load_vision_prelude().await.unwrap();
            source.load_vision_block(0).await.unwrap();
            source.load_vision_deepstack_merger(0).await.unwrap();
            source.load_vision_final_merger().await.unwrap();
            source.load_text_block(0).await.unwrap();
            source.load_text_final_norm().await.unwrap();
            source.synchronize().await.unwrap();
        });

        assert!(!source.reader().requests.is_empty());
        assert!(
            source
                .reader()
                .requests
                .iter()
                .all(|(path, cap)| *cap == max_bytes && source.reader().sealed.contains_key(path))
        );
        assert!(source.reader().largest_response as u64 <= max_bytes);
    }

    #[cfg(feature = "burnpack")]
    #[test]
    fn async_reader_rejects_digest_size_and_cap_violations_correctness() {
        use burn::backend::NdArray;
        use burn_qwen3_vl::{AsyncQwen3VlStageSource, Qwen3VlStreamingPlan, RowChunkPlan};
        use burn_store::ModuleSnapshot;
        use futures::executor::block_on;

        let fixture = || {
            let config = tiny_qwen_config();
            let device = Default::default();
            let resident = burn_qwen3_vl::Qwen3VlForConditionalGeneration::<NdArray<f32>>::new(
                config.clone(),
                &device,
            )
            .unwrap();
            let snapshots = resident.collect(None, None, false);
            let inventory = BooguArtifactInventory {
                tensors: qwen_specs(&config),
            };
            let (directory, manifest) = write_tiny_float_artifact(
                &inventory,
                snapshots.clone(),
                BooguStorageProfile::F16QwenVisionF32,
            );
            let rows = RowChunkPlan::even(
                config.text_config.vocab_size,
                config.text_config.hidden_size,
                3,
                2,
            )
            .unwrap();
            let plan = Qwen3VlStreamingPlan::new(&config, rows, None).unwrap();
            let manifest = rewrite_tiny_qwen_vocabulary_as_rows(
                &directory, manifest, &inventory, &snapshots, &plan,
            );
            (config, inventory, directory, manifest)
        };

        let (config, inventory, directory, manifest) = fixture();
        let corrupt_path = manifest
            .files
            .iter()
            .find(|file| {
                file.component.as_ref().map(|value| value.as_str()) == Some("qwen-vision-prelude")
            })
            .unwrap()
            .path
            .as_str()
            .to_owned();
        let mut reader = AsyncMemoryShardReader::from_directory(&directory, &manifest);
        reader.corrupt_path = Some(corrupt_path);
        let mut source = block_on(
            VerifiedAsyncBurnpackQwenStageSource::<NdArray<f32>, _>::new_auto(
                &BooguReleaseIdentity::canonical(BooguVariant::Image01Turbo),
                manifest,
                inventory,
                config,
                BooguStorageProfile::F16QwenVisionF32,
                Default::default(),
                reader,
            ),
        )
        .unwrap();
        let error = block_on(source.load_vision_prelude()).unwrap_err();
        assert!(error.to_string().contains("integrity verification failed"));

        let (config, inventory, directory, mut manifest) = fixture();
        let cap = manifest.files.iter().map(|file| file.size).max().unwrap();
        manifest
            .metadata
            .insert("target_max_shard_bytes".into(), cap.to_string());
        manifest.content_digest = None;
        manifest.seal().unwrap();
        let mut reader = AsyncMemoryShardReader::from_directory(&directory, &manifest);
        reader.oversize_response = true;
        let error = block_on(
            VerifiedAsyncBurnpackQwenStageSource::<NdArray<f32>, _>::new_auto(
                &BooguReleaseIdentity::canonical(BooguVariant::Image01Turbo),
                manifest,
                inventory,
                config,
                BooguStorageProfile::F16QwenVisionF32,
                Default::default(),
                reader,
            ),
        )
        .err()
        .expect("oversized response must be rejected");
        assert!(error.to_string().contains("exceeding the per-read cap"));

        let (config, inventory, directory, manifest) = fixture();
        let mut reader = AsyncMemoryShardReader::from_directory(&directory, &manifest);
        reader.append_byte_path = Some("metadata/tensor-inventory.json".into());
        let error = block_on(
            VerifiedAsyncBurnpackQwenStageSource::<NdArray<f32>, _>::new_auto(
                &BooguReleaseIdentity::canonical(BooguVariant::Image01Turbo),
                manifest,
                inventory,
                config,
                BooguStorageProfile::F16QwenVisionF32,
                Default::default(),
                reader,
            ),
        )
        .err()
        .expect("size-mismatched response must be rejected");
        assert!(error.to_string().contains("integrity verification failed"));

        let (config, inventory, directory, mut manifest) = fixture();
        let wrong_dtype_file = manifest
            .files
            .iter()
            .find(|file| {
                file.component.as_ref().map(|value| value.as_str()) == Some("qwen-vision-prelude")
            })
            .unwrap()
            .clone();
        let mut store =
            burn_store::BurnpackStore::from_bytes(Some(burn::tensor::Bytes::from_bytes_vec(
                std::fs::read(directory.path().join(wrong_dtype_file.path.as_str())).unwrap(),
            )));
        let snapshots = burn_store::ModuleStore::get_all_snapshots(&mut store)
            .unwrap()
            .iter()
            .enumerate()
            .map(|(index, (name, snapshot))| {
                if index == 0 {
                    burn_store::TensorSnapshot::from_data(
                        snapshot
                            .to_data()
                            .unwrap()
                            .convert_dtype(burn::tensor::DType::F16),
                        vec![name.clone()],
                        Vec::new(),
                        burn::module::ParamId::new(),
                    )
                } else {
                    snapshot.clone()
                }
            })
            .collect::<Vec<_>>();
        let wrong_dtype_bytes = burn_store::BurnpackWriter::new(snapshots)
            .to_bytes()
            .unwrap()
            .to_vec();
        std::fs::write(
            directory.path().join(wrong_dtype_file.path.as_str()),
            &wrong_dtype_bytes,
        )
        .unwrap();
        let manifest_file = manifest
            .files
            .iter_mut()
            .find(|file| file.path == wrong_dtype_file.path)
            .unwrap();
        manifest_file.size = wrong_dtype_bytes.len() as u64;
        manifest_file.sha256 = burn_image::Sha256Digest::calculate(&wrong_dtype_bytes);
        manifest.content_digest = None;
        manifest.seal().unwrap();
        let reader = AsyncMemoryShardReader::from_directory(&directory, &manifest);
        let mut source = block_on(
            VerifiedAsyncBurnpackQwenStageSource::<NdArray<f32>, _>::new_auto(
                &BooguReleaseIdentity::canonical(BooguVariant::Image01Turbo),
                manifest,
                inventory,
                config,
                BooguStorageProfile::F16QwenVisionF32,
                Default::default(),
                reader,
            ),
        )
        .unwrap();
        let error = block_on(source.load_vision_prelude()).unwrap_err();
        assert!(error.to_string().contains("dtype mismatch"));
    }

    #[cfg(feature = "burnpack")]
    #[test]
    fn real_streamed_qwen_burnpack_load_reference() {
        use burn::backend::NdArray;
        use burn_qwen3_vl::{Qwen3VlConfig, Qwen3VlStageSource};

        let Some(root) = std::env::var_os("BURN_BOOGU_ARTIFACT_DIR").map(PathBuf::from) else {
            eprintln!("BURN_BOOGU_ARTIFACT_DIR is unset; skipping opt-in real artifact fixture");
            return;
        };
        let manifest: burn_image::ArtifactManifest = serde_json::from_slice(
            &std::fs::read(root.join("manifest.json")).expect("read real manifest"),
        )
        .expect("parse real manifest");
        let profile = match manifest.profile.as_str() {
            "f16" => BooguStorageProfile::F16,
            "f16-qwen-vision-f32" => BooguStorageProfile::F16QwenVisionF32,
            "q8s-block32-f32" => BooguStorageProfile::Q8sBlock32F32,
            "q8s-block32-f32-qwen-vision-f32" => BooguStorageProfile::Q8sBlock32F32QwenVisionF32,
            value => panic!("unsupported real fixture profile {value}"),
        };
        let qwen_config = Qwen3VlConfig::from_json(
            &std::fs::read_to_string(root.join("metadata/source/mllm/config.json"))
                .expect("read real Qwen config"),
        )
        .expect("parse real Qwen config");
        let vae_config = burn_flux_vae::AutoencoderKlConfig::from_diffusers_json(
            &std::fs::read_to_string(root.join("metadata/source/vae/config.json"))
                .expect("read real VAE config"),
        )
        .expect("parse real VAE config");
        let inventory =
            BooguArtifactInventory::new(&qwen_config, &BooguConfig::default(), &vae_config)
                .expect("build exact real inventory");
        let device = Default::default();
        let mut source = VerifiedBurnpackQwenStageSource::<NdArray<f32>, _>::from_directory_auto(
            &BooguReleaseIdentity::canonical(BooguVariant::Image01Turbo),
            &root,
            inventory,
            qwen_config,
            profile,
            device,
        )
        .expect("verify real streamed Qwen source")
        .with_float_load_policy(BooguFloatLoadPolicy::AdaptToF32);
        source
            .load_vision_prelude()
            .expect("load and strictly apply the real Qwen vision prelude");
        source.synchronize().expect("synchronize real fixture");
    }

    #[cfg(feature = "burnpack")]
    #[test]
    fn qwen_base_directory_loader_skips_lm_head_correctness() {
        use burn::backend::NdArray;
        use burn_store::ModuleSnapshot;

        let config = tiny_qwen_config();
        let device = Default::default();
        let source = burn_qwen3_vl::Qwen3VlForConditionalGeneration::<NdArray<f32>>::new(
            config.clone(),
            &device,
        )
        .unwrap();
        let inventory = BooguArtifactInventory {
            tensors: qwen_specs(&config),
        };
        let (directory, manifest) = write_tiny_float_artifact(
            &inventory,
            source.collect(None, None, false),
            BooguStorageProfile::F16,
        );
        let lm_head_files = manifest
            .files
            .iter()
            .filter(|file| {
                file.component.as_ref().map(|value| value.as_str()) == Some("qwen-lm-head")
            })
            .collect::<Vec<_>>();
        assert_eq!(lm_head_files.len(), 1);
        std::fs::remove_file(directory.path().join(lm_head_files[0].path.as_str())).unwrap();

        let (_, report) = load_resident_qwen_base_from_directory::<NdArray<f32>>(
            &BooguReleaseIdentity::canonical(BooguVariant::Image01Turbo),
            directory.path(),
            inventory.clone(),
            config,
            BooguStorageProfile::F16,
            BooguFloatLoadPolicy::AdaptToF32,
            &device,
        )
        .unwrap();
        assert_eq!(report.tensors + 1, inventory.tensors().len());
        assert!(!report.by_stage.contains_key("qwen-lm-head"));
    }

    #[cfg(feature = "burnpack")]
    #[test]
    fn vae_directory_loader_applies_exact_inventory_correctness() {
        use burn::backend::NdArray;
        use burn_store::ModuleSnapshot;

        let config = AutoencoderKlConfig::tiny();
        let device = Default::default();
        let source = config.clone().try_init::<NdArray<f32>>(&device).unwrap();
        let inventory = BooguArtifactInventory {
            tensors: flux_vae_specs(&config).unwrap(),
        };
        let (directory, mut manifest) = write_tiny_float_artifact(
            &inventory,
            source.collect(None, None, false),
            BooguStorageProfile::F16,
        );
        let (_, report) = load_resident_vae_from_directory::<NdArray<f32>>(
            &BooguReleaseIdentity::canonical(BooguVariant::Image01Turbo),
            directory.path(),
            inventory.clone(),
            config.clone(),
            BooguStorageProfile::F16,
            BooguFloatLoadPolicy::AdaptToF32,
            &device,
        )
        .unwrap();
        assert_eq!(report.tensors, inventory.tensors().len());
        assert_eq!(report.by_stage.len(), 2);

        let inventory_path = directory.path().join("metadata/tensor-inventory.json");
        let mut serialized: Vec<serde_json::Value> =
            serde_json::from_slice(&std::fs::read(&inventory_path).unwrap()).unwrap();
        serialized[0]["stored_shape"] = serde_json::json!([999]);
        let corrupted = serde_json::to_vec(&serialized).unwrap();
        std::fs::write(&inventory_path, &corrupted).unwrap();
        let manifest_entry = manifest
            .files
            .iter_mut()
            .find(|file| file.path.as_str() == "metadata/tensor-inventory.json")
            .unwrap();
        manifest_entry.size = corrupted.len() as u64;
        manifest_entry.sha256 = burn_image::Sha256Digest::calculate(&corrupted);
        manifest.content_digest = None;
        manifest.seal().unwrap();
        std::fs::write(
            directory.path().join("manifest.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        let error = load_resident_vae_from_directory::<NdArray<f32>>(
            &BooguReleaseIdentity::canonical(BooguVariant::Image01Turbo),
            directory.path(),
            inventory,
            config,
            BooguStorageProfile::F16,
            BooguFloatLoadPolicy::AdaptToF32,
            &device,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            BooguArtifactLoadError::Contract { stage, .. } if stage == "tensor-inventory"
        ));
    }

    #[cfg(feature = "burnpack")]
    #[test]
    fn vae_encoder_and_decoder_stage_loaders_do_not_fetch_opposite_stage_correctness() {
        use crate::BooguVaeStageSource;
        use burn::backend::NdArray;
        use burn_store::ModuleSnapshot;

        let config = AutoencoderKlConfig::tiny();
        let device = Default::default();
        let source = config.clone().try_init::<NdArray<f32>>(&device).unwrap();
        let snapshots = source.collect(None, None, false);
        let inventory = BooguArtifactInventory {
            tensors: flux_vae_specs(&config).unwrap(),
        };
        let identity = BooguReleaseIdentity::canonical(BooguVariant::Image01Turbo);

        let (encoder_directory, encoder_manifest) =
            write_tiny_float_artifact(&inventory, snapshots.clone(), BooguStorageProfile::F16);
        for file in encoder_manifest.files.iter().filter(|file| {
            file.component.as_ref().map(|value| value.as_str()) == Some("flux-vae-decoder")
        }) {
            std::fs::remove_file(encoder_directory.path().join(file.path.as_str())).unwrap();
        }
        let (_, encoder_report) = load_vae_encoder_from_directory::<NdArray<f32>>(
            &identity,
            encoder_directory.path(),
            inventory.clone(),
            config.clone(),
            BooguStorageProfile::F16,
            BooguFloatLoadPolicy::AdaptToF32,
            &device,
        )
        .unwrap();
        assert_eq!(
            encoder_report.by_stage.keys().collect::<Vec<_>>(),
            ["flux-vae-encoder"]
        );
        assert_eq!(
            encoder_report.tensors,
            inventory
                .tensors()
                .iter()
                .filter(|spec| spec.stage == "flux-vae-encoder")
                .count()
        );

        let (decoder_directory, decoder_manifest) =
            write_tiny_float_artifact(&inventory, snapshots, BooguStorageProfile::F16);
        for file in decoder_manifest.files.iter().filter(|file| {
            file.component.as_ref().map(|value| value.as_str()) == Some("flux-vae-encoder")
        }) {
            std::fs::remove_file(decoder_directory.path().join(file.path.as_str())).unwrap();
        }
        let (_, decoder_report) = load_vae_decoder_from_directory::<NdArray<f32>>(
            &identity,
            decoder_directory.path(),
            inventory.clone(),
            config.clone(),
            BooguStorageProfile::F16,
            BooguFloatLoadPolicy::AdaptToF32,
            &device,
        )
        .unwrap();
        assert_eq!(
            decoder_report.by_stage.keys().collect::<Vec<_>>(),
            ["flux-vae-decoder"]
        );
        assert_eq!(
            decoder_report.tensors,
            inventory
                .tensors()
                .iter()
                .filter(|spec| spec.stage == "flux-vae-decoder")
                .count()
        );

        let source = config.try_init::<NdArray<f32>>(&device).unwrap();
        let (stage_directory, _) = write_tiny_float_artifact(
            &inventory,
            source.collect(None, None, false),
            BooguStorageProfile::F16,
        );
        let mut stage_source = VerifiedDirectoryVaeStageSource::<NdArray<f32>>::new(
            &identity,
            stage_directory.path(),
            inventory,
            config,
            BooguStorageProfile::F16,
            BooguFloatLoadPolicy::AdaptToF32,
            device,
        )
        .unwrap();
        stage_source.load_encoder().unwrap();
        stage_source.load_decoder().unwrap();
    }

    #[cfg(feature = "burnpack")]
    #[test]
    fn async_vae_source_fetches_only_selected_half_correctness() {
        use crate::AsyncBooguVaeStageSource;
        use burn::backend::NdArray;
        use burn_store::ModuleSnapshot;
        use futures::executor::block_on;

        let config = AutoencoderKlConfig::tiny();
        let device = Default::default();
        let resident = config.clone().try_init::<NdArray<f32>>(&device).unwrap();
        let inventory = BooguArtifactInventory {
            tensors: flux_vae_specs(&config).unwrap(),
        };
        let (directory, manifest) = write_tiny_float_artifact(
            &inventory,
            resident.collect(None, None, false),
            BooguStorageProfile::F16,
        );
        let reader = AsyncMemoryShardReader::from_directory(&directory, &manifest);
        let mut source = block_on(VerifiedAsyncBurnpackVaeStageSource::<NdArray<f32>, _>::new(
            &BooguReleaseIdentity::canonical(BooguVariant::Image01Turbo),
            manifest.clone(),
            inventory,
            config,
            BooguStorageProfile::F16,
            BooguFloatLoadPolicy::AdaptToF32,
            device,
            reader,
        ))
        .unwrap();

        source.reader_mut().requests.clear();
        block_on(source.load_encoder()).unwrap();
        let requested = source
            .reader()
            .requests
            .iter()
            .map(|(path, _)| path.as_str())
            .collect::<BTreeSet<_>>();
        let encoder_files = manifest
            .files
            .iter()
            .filter(|file| {
                file.component.as_ref().map(|value| value.as_str()) == Some("flux-vae-encoder")
            })
            .map(|file| file.path.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(requested, encoder_files);

        source.reader_mut().requests.clear();
        block_on(source.load_decoder()).unwrap();
        let requested = source
            .reader()
            .requests
            .iter()
            .map(|(path, _)| path.as_str())
            .collect::<BTreeSet<_>>();
        let decoder_files = manifest
            .files
            .iter()
            .filter(|file| {
                file.component.as_ref().map(|value| value.as_str()) == Some("flux-vae-decoder")
            })
            .map(|file| file.path.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(requested, decoder_files);
        block_on(source.synchronize()).unwrap();
    }

    #[cfg(feature = "burnpack")]
    #[test]
    fn async_denoiser_source_loads_one_verified_stage_at_a_time_correctness() {
        use crate::{AsyncBooguDenoiserStageSource, StreamingStageSource};
        use burn::{backend::NdArray, tensor::DType};
        use burn_store::ModuleSnapshot;
        use futures::executor::block_on;

        let config = BooguConfig {
            patch_size: 2,
            in_channels: 4,
            out_channels: 4,
            hidden_size: 32,
            num_layers: 2,
            num_double_stream_layers: 1,
            num_refiner_layers: 1,
            num_attention_heads: 4,
            num_kv_heads: 1,
            multiple_of: 32,
            norm_eps: 1.0e-5,
            axes_dim_rope: [4, 4, 0],
            axes_lens: [16, 16, 16],
            instruction_feature_dim: 32,
            timestep_scale: 1000.0,
        };
        let device = Default::default();
        let resident = crate::BooguDenoiser::<NdArray<f32>>::new(config.clone(), &device).unwrap();
        let inventory = BooguArtifactInventory {
            tensors: boogu_specs(&config),
        };
        let context_quantizable = inventory
            .tensors()
            .iter()
            .filter(|spec| spec.stage == "boogu-context-refiner-00" && spec.quantizable)
            .map(|spec| {
                spec.target_name
                    .strip_prefix("context_refiner.0.")
                    .unwrap()
                    .to_owned()
            })
            .collect::<BTreeSet<_>>();
        assert!(!context_quantizable.is_empty());
        let prelude_quantizable = inventory
            .tensors()
            .iter()
            .filter(|spec| spec.stage == "boogu-prelude" && spec.quantizable)
            .map(|spec| spec.target_name.clone())
            .collect::<BTreeSet<_>>();
        assert!(prelude_quantizable.contains("time_caption_embed.caption_linear.weight"));
        let tail_quantizable = inventory
            .tensors()
            .iter()
            .filter(|spec| spec.stage == "boogu-tail" && spec.quantizable)
            .map(|spec| spec.target_name.clone())
            .collect::<BTreeSet<_>>();
        assert!(tail_quantizable.contains("norm_out.linear_1.weight"));
        let (directory, manifest) = write_tiny_float_artifact(
            &inventory,
            resident.collect(None, None, false),
            BooguStorageProfile::F16QwenVisionF32,
        );
        let identity = BooguReleaseIdentity::canonical(BooguVariant::Image01Turbo);
        let mut sync_source =
            VerifiedBurnpackStageSource::<NdArray<f32>, DirectoryStageShardReader>::from_directory(
                &identity,
                directory.path(),
                inventory.clone(),
                config.clone(),
                BooguStorageProfile::F16QwenVisionF32,
                device,
            )
            .unwrap()
            .with_float_load_policy(BooguFloatLoadPolicy::AdaptToF32)
            .with_runtime_quantization_policy(BooguDenoiserRuntimeQuantizationPolicy::Q8sBlock32F32)
            .with_runtime_q8_scope(BooguRuntimeQ8Scope::TurboCaptionAndTailF32);
        let sync_prelude = sync_source.load_prelude().unwrap();
        for snapshot in sync_prelude.collect(None, None, false) {
            let expected_q8 = prelude_quantizable.contains(&snapshot.full_path())
                && snapshot.full_path() != "time_caption_embed.caption_linear.weight";
            assert_eq!(
                matches!(snapshot.dtype, DType::QFloat(_)),
                expected_q8,
                "unexpected scoped sync runtime dtype for {}",
                snapshot.full_path()
            );
            if snapshot.full_path() == "time_caption_embed.caption_linear.weight" {
                assert_eq!(snapshot.dtype, DType::F32);
            }
        }
        let sync_tail = sync_source.load_tail().unwrap();
        for snapshot in sync_tail.collect(None, None, false) {
            let expected_q8 = tail_quantizable.contains(&snapshot.full_path())
                && !matches!(
                    snapshot.full_path().as_str(),
                    "norm_out.linear_1.weight" | "norm_out.linear_2.weight"
                );
            assert_eq!(
                matches!(snapshot.dtype, DType::QFloat(_)),
                expected_q8,
                "unexpected scoped sync runtime dtype for {}",
                snapshot.full_path()
            );
            if matches!(
                snapshot.full_path().as_str(),
                "norm_out.linear_1.weight" | "norm_out.linear_2.weight"
            ) {
                assert_eq!(snapshot.dtype, DType::F32);
            }
        }
        let reader = AsyncMemoryShardReader::from_directory(&directory, &manifest);
        let mut source = block_on(
            VerifiedAsyncBurnpackDenoiserStageSource::<NdArray<f32>, _>::new(
                &identity,
                manifest.clone(),
                inventory,
                config,
                BooguStorageProfile::F16QwenVisionF32,
                device,
                reader,
            ),
        )
        .unwrap()
        .with_float_load_policy(BooguFloatLoadPolicy::AdaptToF32)
        .with_runtime_quantization_policy(BooguDenoiserRuntimeQuantizationPolicy::Q8sBlock32F32)
        .with_runtime_q8_scope(BooguRuntimeQ8Scope::TurboCaptionAndTailF32);

        source.reader_mut().requests.clear();
        let prelude = block_on(source.load_prelude()).unwrap();
        for snapshot in prelude.collect(None, None, false) {
            let expected_q8 = prelude_quantizable.contains(&snapshot.full_path())
                && snapshot.full_path() != "time_caption_embed.caption_linear.weight";
            assert_eq!(
                matches!(snapshot.dtype, DType::QFloat(_)),
                expected_q8,
                "unexpected scoped runtime dtype for {}",
                snapshot.full_path()
            );
            if snapshot.full_path() == "time_caption_embed.caption_linear.weight" {
                assert_eq!(snapshot.dtype, DType::F32);
            }
        }
        let prelude_files = manifest
            .files
            .iter()
            .filter(|file| {
                file.component.as_ref().map(|value| value.as_str()) == Some("boogu-prelude")
            })
            .map(|file| file.path.as_str())
            .collect::<BTreeSet<_>>();
        let requested = source
            .reader()
            .requests
            .iter()
            .map(|(path, _)| path.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(requested, prelude_files);

        let context = block_on(source.load_context_refiner(0)).unwrap();
        let context_snapshots = context.collect(None, None, false);
        assert!(
            context_snapshots
                .iter()
                .any(|snapshot| matches!(snapshot.dtype, DType::QFloat(_)))
        );
        for snapshot in context_snapshots {
            assert_eq!(
                matches!(snapshot.dtype, DType::QFloat(_)),
                context_quantizable.contains(&snapshot.full_path()),
                "unexpected runtime dtype for {}",
                snapshot.full_path()
            );
        }

        let tail = block_on(source.load_tail()).unwrap();
        for snapshot in tail.collect(None, None, false) {
            let expected_q8 = tail_quantizable.contains(&snapshot.full_path())
                && !matches!(
                    snapshot.full_path().as_str(),
                    "norm_out.linear_1.weight" | "norm_out.linear_2.weight"
                );
            assert_eq!(
                matches!(snapshot.dtype, DType::QFloat(_)),
                expected_q8,
                "unexpected scoped runtime dtype for {}",
                snapshot.full_path()
            );
            if matches!(
                snapshot.full_path().as_str(),
                "norm_out.linear_1.weight" | "norm_out.linear_2.weight"
            ) {
                assert_eq!(snapshot.dtype, DType::F32);
            }
        }

        block_on(async {
            source.load_noise_refiner(0).await.unwrap();
            source.load_reference_refiner(0).await.unwrap();
            source.load_double_stream(0).await.unwrap();
            source.load_single_stream(0).await.unwrap();
            source.synchronize().await.unwrap();
        });
        assert!(source.reader().largest_response as u64 <= source.max_shard_bytes());
    }

    #[cfg(feature = "burnpack")]
    #[test]
    fn verified_stage_source_applies_prefix_stripped_burnpacks_correctness() {
        use crate::StreamingStageSource;
        use burn::{backend::NdArray, module::ParamId, tensor::DType};
        use burn_image::{
            ARTIFACT_MANIFEST_SCHEMA_V1, ArtifactBundleId, ArtifactComponent, ArtifactComponentId,
            ArtifactFile, ArtifactFileRole, ArtifactManifest, ArtifactPath, ArtifactProfileId,
            ModelId, NumericFormat, Sha256Digest,
        };
        use burn_store::{BurnpackWriter, ModuleSnapshot, TensorSnapshot};

        #[derive(Default)]
        struct MemoryReader(BTreeMap<String, Vec<u8>>);

        impl StageShardReader for MemoryReader {
            fn read_shard(&mut self, file: &ArtifactFile) -> Result<Vec<u8>, BooguError> {
                self.0
                    .get(file.path.as_str())
                    .cloned()
                    .ok_or_else(|| BooguError::Artifact(format!("missing {}", file.path)))
            }
        }

        let config = BooguConfig {
            patch_size: 2,
            in_channels: 4,
            out_channels: 4,
            hidden_size: 8,
            num_layers: 2,
            num_double_stream_layers: 1,
            num_refiner_layers: 1,
            num_attention_heads: 2,
            num_kv_heads: 1,
            multiple_of: 8,
            norm_eps: 1.0e-5,
            axes_dim_rope: [2, 2, 0],
            axes_lens: [16, 16, 16],
            instruction_feature_dim: 8,
            timestep_scale: 1000.0,
        };
        let inventory = BooguArtifactInventory {
            tensors: boogu_specs(&config),
        };
        let sync_context_quantizable = inventory
            .tensors()
            .iter()
            .filter(|spec| spec.stage == "boogu-context-refiner-00" && spec.quantizable)
            .map(|spec| {
                spec.target_name
                    .strip_prefix("context_refiner.0.")
                    .unwrap()
                    .to_owned()
            })
            .collect::<BTreeSet<_>>();
        assert!(!sync_context_quantizable.is_empty());
        let device = Default::default();
        let model = crate::BooguDenoiser::<NdArray<f32>>::new(config.clone(), &device).unwrap();
        let source = model
            .collect(None, None, false)
            .into_iter()
            .map(|snapshot| (snapshot.full_path(), snapshot))
            .collect::<BTreeMap<_, _>>();
        let mut by_stage = BTreeMap::<String, Vec<TensorSnapshot>>::new();
        for spec in inventory.tensors() {
            let original = source.get(&spec.target_name).unwrap();
            let data = original.to_data().unwrap().convert_dtype(DType::F16);
            by_stage
                .entry(spec.stage.clone())
                .or_default()
                .push(TensorSnapshot::from_data(
                    data,
                    vec![spec.target_name.clone()],
                    Vec::new(),
                    ParamId::new(),
                ));
        }
        let mut reader = MemoryReader::default();
        let mut files = Vec::new();
        let mut components = Vec::new();
        for (stage, snapshots) in by_stage {
            let bytes = BurnpackWriter::new(snapshots).to_bytes().unwrap();
            let bytes = bytes.to_vec();
            let path = format!("objects/{stage}.bpk");
            files.push(ArtifactFile {
                path: ArtifactPath::new(&path).unwrap(),
                size: bytes.len() as u64,
                sha256: Sha256Digest::calculate(&bytes),
                role: ArtifactFileRole::Weights,
                component: Some(ArtifactComponentId::new(&stage).unwrap()),
                shard: None,
            });
            components.push(ArtifactComponent {
                id: ArtifactComponentId::new(&stage).unwrap(),
                required: true,
            });
            reader.0.insert(path, bytes);
        }
        let inventory_metadata = serde_json::to_vec(
            &inventory
                .tensors()
                .iter()
                .map(|spec| {
                    let burnpack_object = files
                        .iter()
                        .find(|file| {
                            file.role == ArtifactFileRole::Weights
                                && file.component.as_ref().map(|value| value.as_str())
                                    == Some(spec.stage.as_str())
                        })
                        .unwrap()
                        .path
                        .as_str();
                    let source_element_bytes = match spec.source_dtype {
                        SourceDType::Bf16 => 2_u64,
                        SourceDType::F32 => 4_u64,
                    };
                    let source_bytes =
                        spec.source_shape.iter().product::<usize>() as u64 * source_element_bytes;
                    serde_json::json!({
                        "source_name": spec.source_name,
                        "target_name": spec.target_name,
                        "owner": spec.owner,
                        "component": spec.source_component,
                        "stage": spec.stage,
                        "transform": spec.transform,
                        "source_file": format!("{}/fixture.safetensors", spec.source_component),
                        "source_dtype": spec.source_dtype.safetensors_name(),
                        "source_shape": spec.source_shape,
                        "stored_dtype": "f16",
                        "stored_shape": spec.target_shape,
                        "source_offset": 0,
                        "source_bytes": source_bytes,
                        "quantized": false,
                        "stored_sha256": Sha256Digest::calculate(spec.target_name.as_bytes()),
                        "burnpack_object": burnpack_object,
                    })
                })
                .collect::<Vec<_>>(),
        )
        .unwrap();
        files.push(ArtifactFile {
            path: ArtifactPath::new("metadata/tensor-inventory.json").unwrap(),
            size: inventory_metadata.len() as u64,
            sha256: Sha256Digest::calculate(&inventory_metadata),
            role: ArtifactFileRole::Metadata,
            component: None,
            shard: None,
        });
        reader
            .0
            .insert("metadata/tensor-inventory.json".into(), inventory_metadata);
        let source_metadata = serde_json::to_vec(&[serde_json::json!({
            "path": "transformer/fixture.safetensors",
            "size": 1024_u64 * 1024 * 1024,
            "sha256": Sha256Digest::calculate(b"transformer"),
        })])
        .unwrap();
        files.push(ArtifactFile {
            path: ArtifactPath::new("metadata/source-files.json").unwrap(),
            size: source_metadata.len() as u64,
            sha256: Sha256Digest::calculate(&source_metadata),
            role: ArtifactFileRole::Metadata,
            component: None,
            shard: None,
        });
        reader
            .0
            .insert("metadata/source-files.json".into(), source_metadata);
        let identity = BooguReleaseIdentity::canonical(BooguVariant::Image01Turbo);
        let mut metadata = BTreeMap::new();
        metadata.insert("source_revision".into(), UPSTREAM_SOURCE_REVISION.into());
        metadata.insert("algorithm".into(), "dmd-turbo".into());
        metadata.insert("artifact_layout".into(), "semantic-burnpack-v1".into());
        metadata.insert("tensor_inventory_schema".into(), "2".into());
        metadata.insert(
            "layout_contract".into(),
            "metadata/tensor-inventory.json".into(),
        );
        metadata.insert(
            "conversion_crate".into(),
            CURRENT_BUNDLE_CONVERTER_VERSION.into(),
        );
        metadata.insert("profile".into(), "f16".into());
        metadata.insert("tensor_count".into(), inventory.tensors().len().to_string());
        metadata.insert(
            "stored_tensor_count".into(),
            inventory.tensors().len().to_string(),
        );
        metadata.insert("omitted_tensor_count".into(), "0".into());
        metadata.insert("target_max_shard_bytes".into(), (1024 * 1024).to_string());
        metadata.insert("physical_shards_bounded".into(), "true".into());
        let mut manifest = ArtifactManifest {
            schema_version: ARTIFACT_MANIFEST_SCHEMA_V1,
            bundle: ArtifactBundleId::new("tiny-stream-test").unwrap(),
            profile: ArtifactProfileId::new("f16").unwrap(),
            model: ModelId::new("Boogu/Boogu-Image-0.1-Turbo").unwrap(),
            model_revision: TURBO_REVISION.into(),
            numeric_format: NumericFormat::F16,
            components,
            files,
            dependencies: Vec::new(),
            metadata,
            content_digest: None,
        };
        manifest.seal().unwrap();

        let resident_directory = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(resident_directory.path().join("objects")).unwrap();
        for (path, bytes) in &reader.0 {
            let destination = resident_directory.path().join(path);
            if let Some(parent) = destination.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(destination, bytes).unwrap();
        }
        std::fs::write(
            resident_directory.path().join("manifest.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        let (resident, report) = load_resident_denoiser_from_directory::<NdArray<f32>>(
            &identity,
            resident_directory.path(),
            inventory.clone(),
            config.clone(),
            BooguStorageProfile::F16,
            BooguFloatLoadPolicy::AdaptToF32,
            &device,
        )
        .unwrap();
        assert_eq!(report.tensors, inventory.tensors().len());
        assert_eq!(report.shards, manifest.components.len());
        let (cleanup_model, cleanup_report) =
            load_resident_denoiser_from_directory_with_memory_policy::<NdArray<f32>>(
                &identity,
                resident_directory.path(),
                inventory.clone(),
                config.clone(),
                BooguStorageProfile::F16,
                BooguFloatLoadPolicy::AdaptToF32,
                BooguQuantizedLoadPolicy::Preserve,
                BooguResidentLoadMemoryPolicy::ReleaseTransientBuffersPerShard,
                &device,
            )
            .unwrap();
        assert_eq!(cleanup_report, report);
        assert_eq!(
            BooguResidentLoadMemoryPolicy::default(),
            BooguResidentLoadMemoryPolicy::PreserveAllocatorCache
        );
        let mut resident_snapshots = resident.collect(None, None, false);
        let mut cleanup_snapshots = cleanup_model.collect(None, None, false);
        resident_snapshots.sort_by_key(|snapshot| snapshot.full_path());
        cleanup_snapshots.sort_by_key(|snapshot| snapshot.full_path());
        assert_eq!(resident_snapshots.len(), cleanup_snapshots.len());
        for (resident_snapshot, cleanup_snapshot) in
            resident_snapshots.into_iter().zip(cleanup_snapshots)
        {
            assert_eq!(resident_snapshot.full_path(), cleanup_snapshot.full_path());
            assert_eq!(resident_snapshot.dtype, cleanup_snapshot.dtype);
            assert_eq!(resident_snapshot.shape, cleanup_snapshot.shape);
            assert_eq!(
                resident_snapshot.to_data().unwrap(),
                cleanup_snapshot.to_data().unwrap()
            );
        }

        let mut source = VerifiedBurnpackStageSource::<NdArray<f32>, _>::new(
            &identity,
            manifest,
            inventory,
            config,
            BooguStorageProfile::F16,
            device,
            reader,
        )
        .unwrap()
        .with_float_load_policy(BooguFloatLoadPolicy::AdaptToF32)
        .with_runtime_quantization_policy(BooguDenoiserRuntimeQuantizationPolicy::Q8sBlock32F32);
        source.load_prelude().unwrap();
        let context = source.load_context_refiner(0).unwrap();
        for snapshot in context.collect(None, None, false) {
            assert_eq!(
                matches!(snapshot.dtype, DType::QFloat(_)),
                sync_context_quantizable.contains(&snapshot.full_path()),
                "unexpected runtime dtype for {}",
                snapshot.full_path()
            );
        }
        source.load_noise_refiner(0).unwrap();
        source.load_reference_refiner(0).unwrap();
        source.load_double_stream(0).unwrap();
        source.load_single_stream(0).unwrap();
        source.load_tail().unwrap();
    }

    #[cfg(feature = "burnpack")]
    #[test]
    fn q8s_dequantize_f16_preserves_non_square_col_forward_correctness() {
        use burn::{
            backend::NdArray,
            module::ParamId,
            nn::{LinearConfig, LinearLayout},
            tensor::{Tensor, TensorData, quantization::*},
        };
        use burn_store::{ModuleSnapshot, TensorSnapshot};
        use half::f16;

        let device = Default::default();
        let values = (-16_i8..16).collect::<Vec<_>>();
        let scale = 0.013_37_f32;
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_level(QuantLevel::block([32]))
            .with_param(QuantParam::F32)
            .with_store(QuantStore::PackedU32(0));
        let snapshot = TensorSnapshot::from_data(
            TensorData::quantized(values.clone(), [4, 8], scheme, &[scale]),
            vec!["weight".into()],
            Vec::new(),
            ParamId::new(),
        );
        let mut linear = LinearConfig::new(8, 4)
            .with_bias(false)
            .with_layout(LinearLayout::Col)
            .init::<NdArray<f32>>(&device);
        let result = linear.apply(
            vec![snapshot],
            None,
            loading::load_adapter(
                BooguFloatLoadPolicy::AdaptToF32,
                BooguQuantizedLoadPolicy::DequantizeF16,
            ),
            false,
        );
        assert!(result.errors.is_empty(), "{:?}", result.errors);
        assert_eq!(result.applied, ["weight"]);

        let saved = values
            .iter()
            .map(|value| f16::from_f32(f32::from(*value) * scale).to_f32())
            .collect::<Vec<_>>();
        let internal = linear.weight.val().into_data().to_vec::<f32>().unwrap();
        let mut expected_internal = Vec::with_capacity(saved.len());
        for input in 0..8 {
            for output in 0..4 {
                expected_internal.push(saved[output * 8 + input]);
            }
        }
        assert_eq!(internal, expected_internal);

        let input = vec![-0.75_f32, -0.5, -0.25, 0.0, 0.125, 0.25, 0.5, 0.75];
        let expected = (0..4)
            .map(|output| {
                input
                    .iter()
                    .zip(&saved[output * 8..(output + 1) * 8])
                    .map(|(input, weight)| input * weight)
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        let actual = linear
            .forward(Tensor::<NdArray<f32>, 2>::from_data(
                TensorData::new(input, [1, 8]),
                &device,
            ))
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 1.0e-6,
                "{actual} != {expected}"
            );
        }
    }

    #[cfg(feature = "burnpack")]
    #[test]
    fn runtime_q8s_matches_canonical_importer_bytes_and_scales_correctness() {
        use burn::{
            module::ParamId,
            tensor::{DType, TensorData, quantization::QuantizedBytes},
        };
        use burn_store::{ModuleAdapter, TensorSnapshot};

        let values = (-32..32)
            .map(|value| value as f32 * 0.031_25)
            .collect::<Vec<_>>();
        let expected = quantize_q8s_block32_f32(values.clone(), vec![2, 32]).unwrap();
        let source = TensorSnapshot::from_data(
            TensorData::new(values, [2, 32]).convert_dtype(DType::F16),
            vec!["weight".into()],
            Vec::new(),
            ParamId::new(),
        );
        let adapter = loading::ArtifactLoadAdapter {
            float_policy: BooguFloatLoadPolicy::AdaptToF32,
            quantized_policy: BooguQuantizedLoadPolicy::Preserve,
            runtime_quantization_policy: BooguDenoiserRuntimeQuantizationPolicy::Q8sBlock32F32,
            runtime_quantizable_paths: Some(BTreeSet::from(["weight".into()])),
        };
        let actual = adapter.adapt(&source).to_data().unwrap();
        assert_eq!(actual.dtype, expected.dtype);
        assert_eq!(actual.shape, expected.shape);

        let DType::QFloat(scheme) = actual.dtype else {
            unreachable!()
        };
        let actual_parts = QuantizedBytes {
            bytes: actual.bytes,
            scheme,
            num_elements: 64,
        }
        .into_vec_i8();
        let expected_parts = QuantizedBytes {
            bytes: expected.bytes,
            scheme,
            num_elements: 64,
        }
        .into_vec_i8();
        assert_eq!(actual_parts.0, expected_parts.0);
        assert_eq!(actual_parts.1.scales, expected_parts.1.scales);
    }

    #[cfg(feature = "burnpack")]
    #[test]
    fn runtime_q8s_quantizes_only_inventory_eligible_paths_correctness() {
        use burn::{
            module::ParamId,
            tensor::{DType, TensorData},
        };
        use burn_store::{ModuleAdapter, TensorSnapshot};

        let adapter = loading::ArtifactLoadAdapter {
            float_policy: BooguFloatLoadPolicy::AdaptToF32,
            quantized_policy: BooguQuantizedLoadPolicy::Preserve,
            runtime_quantization_policy: BooguDenoiserRuntimeQuantizationPolicy::Q8sBlock32F32,
            runtime_quantizable_paths: Some(BTreeSet::from(["weight".into()])),
        };
        let eligible = TensorSnapshot::from_data(
            TensorData::new(vec![0.25_f32; 32], [1, 32]).convert_dtype(DType::F16),
            vec!["weight".into()],
            Vec::new(),
            ParamId::new(),
        );
        let bias = TensorSnapshot::from_data(
            TensorData::new(vec![0.25_f32; 32], [32]).convert_dtype(DType::F16),
            vec!["bias".into()],
            Vec::new(),
            ParamId::new(),
        );
        assert!(matches!(adapter.adapt(&eligible).dtype, DType::QFloat(_)));
        assert_eq!(adapter.adapt(&bias).dtype, DType::F32);

        let disabled = loading::ArtifactLoadAdapter {
            float_policy: BooguFloatLoadPolicy::AdaptToF32,
            quantized_policy: BooguQuantizedLoadPolicy::Preserve,
            runtime_quantization_policy: BooguDenoiserRuntimeQuantizationPolicy::Disabled,
            runtime_quantizable_paths: Some(BTreeSet::from(["weight".into()])),
        };
        assert_eq!(disabled.adapt(&eligible).dtype, DType::F32);
        assert_eq!(
            BooguDenoiserRuntimeQuantizationPolicy::default(),
            BooguDenoiserRuntimeQuantizationPolicy::Disabled
        );
        assert_eq!(
            BooguDenoiserRuntimeQuantizationPolicy::Q8sBlock32F32.label(),
            "runtime-quantize-q8s-block32-f32"
        );
    }

    #[cfg(feature = "burnpack")]
    #[test]
    fn runtime_q8s_rejects_nonfinite_misaligned_and_nonfloat_sources_correctness() {
        use burn::tensor::{
            TensorData,
            quantization::{QuantLevel, QuantParam, QuantScheme, QuantStore, QuantValue},
        };

        assert!(quantize_q8s_block32_f32(vec![f32::NAN; 32], vec![1, 32]).is_err());
        assert!(quantize_q8s_block32_f32(vec![0.0; 31], vec![1, 31]).is_err());
        assert!(quantize_q8s_block32_f32(vec![0.0; 32], vec![2, 32]).is_err());

        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_level(QuantLevel::block([32]))
            .with_param(QuantParam::F32)
            .with_store(QuantStore::PackedU32(0));
        let q8 = TensorData::quantized(vec![0_i8; 32], [1, 32], scheme, &[1.0]);
        assert!(loading::quantize_verified_float_q8s_block32_f32(q8).is_err());
    }

    #[cfg(feature = "burnpack")]
    #[test]
    #[ignore = "requires the pinned real Q8 bundle and Hugging Face snapshot"]
    fn real_q8_layout_policy_and_wgpu_row_forward_reference() {
        use std::{
            fs::File,
            io::{Read, Seek, SeekFrom},
        };

        use burn::{
            backend::NdArray,
            module::ParamId,
            nn::{LinearConfig, LinearLayout},
            tensor::{Tensor, TensorData},
        };
        use burn_store::{BurnpackStore, ModuleSnapshot, ModuleStore, TensorSnapshot};
        use half::bf16;

        const TARGET: &str = "model.language_model.layers.0.self_attn.k_proj.weight";
        let Some(artifact_root) = std::env::var_os("BURN_BOOGU_Q8_ARTIFACT_DIR").map(PathBuf::from)
        else {
            eprintln!("BURN_BOOGU_Q8_ARTIFACT_DIR is unset; skipping opt-in Q8 probe");
            return;
        };
        let Some(snapshot_root) = std::env::var_os("BURN_BOOGU_HF_SNAPSHOT").map(PathBuf::from)
        else {
            eprintln!("BURN_BOOGU_HF_SNAPSHOT is unset; skipping opt-in Q8 probe");
            return;
        };
        let inventory: Vec<serde_json::Value> = serde_json::from_slice(
            &std::fs::read(artifact_root.join("metadata/tensor-inventory.json")).unwrap(),
        )
        .unwrap();
        let entry = inventory
            .iter()
            .find(|entry| entry["target_name"].as_str() == Some(TARGET))
            .unwrap();
        assert_eq!(entry["transform"].as_str(), Some("identity"));
        assert_eq!(entry["stored_dtype"].as_str(), Some("q8s-block32-f32"));
        let shape = entry["stored_shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as usize)
            .collect::<Vec<_>>();
        assert_eq!(shape, [1024, 4096]);

        let source_path = snapshot_root.join(entry["source_file"].as_str().unwrap());
        let source_offset = entry["source_offset"].as_u64().unwrap();
        let source_bytes = entry["source_bytes"].as_u64().unwrap() as usize;
        let mut source_file = File::open(&source_path).unwrap();
        source_file.seek(SeekFrom::Start(source_offset)).unwrap();
        let mut source_raw = vec![0_u8; source_bytes];
        source_file.read_exact(&mut source_raw).unwrap();
        let source = source_raw
            .chunks_exact(2)
            .map(|pair| bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]])).to_f32())
            .collect::<Vec<_>>();
        assert!(source.iter().all(|value| value.is_finite()));

        let object = entry["burnpack_object"].as_str().unwrap();
        let manifest: ArtifactManifest =
            serde_json::from_slice(&std::fs::read(artifact_root.join("manifest.json")).unwrap())
                .unwrap();
        let object_file = manifest
            .files
            .iter()
            .find(|file| file.path.as_str() == object)
            .unwrap();
        let object_bytes = std::fs::read(artifact_root.join(object)).unwrap();
        burn_image::ArtifactVerifier::verify_bytes(
            object_file,
            &object_bytes,
            burn_image::IntegrityPolicy::RequireSha256,
        )
        .unwrap();
        let mut store =
            BurnpackStore::from_bytes(Some(burn::tensor::Bytes::from_bytes_vec(object_bytes)));
        let snapshot = store.get_all_snapshots().unwrap().get(TARGET).unwrap();
        assert_eq!(snapshot.shape.as_slice(), shape);
        assert!(matches!(snapshot.dtype, burn::tensor::DType::QFloat(_)));

        fn metrics(actual: &[f32], expected: &[f32]) -> (f32, f64, f64) {
            assert_eq!(actual.len(), expected.len());
            let mut max_abs = 0.0_f32;
            let mut squared = 0.0_f64;
            let mut dot = 0.0_f64;
            let mut actual_norm = 0.0_f64;
            let mut expected_norm = 0.0_f64;
            for (&actual, &expected) in actual.iter().zip(expected) {
                max_abs = max_abs.max((actual - expected).abs());
                squared += f64::from(actual - expected).powi(2);
                dot += f64::from(actual) * f64::from(expected);
                actual_norm += f64::from(actual).powi(2);
                expected_norm += f64::from(expected).powi(2);
            }
            (
                max_abs,
                (squared / actual.len() as f64).sqrt(),
                dot / (actual_norm.sqrt() * expected_norm.sqrt()),
            )
        }

        let device = Default::default();
        let dequantized =
            Tensor::<NdArray<f32>, 2>::from_data(snapshot.to_data().unwrap(), &device)
                .dequantize()
                .into_data()
                .to_vec::<f32>()
                .unwrap();
        assert!(dequantized.iter().all(|value| value.is_finite()));
        let raw_metrics = metrics(&dequantized, &source);
        eprintln!(
            "raw Q8 [out,in] vs BF16: max_abs={} rmse={} cosine={}",
            raw_metrics.0, raw_metrics.1, raw_metrics.2
        );

        let local = TensorSnapshot::from_data(
            snapshot.to_data().unwrap(),
            vec!["weight".into()],
            Vec::new(),
            ParamId::new(),
        );
        let mut linear = LinearConfig::new(4096, 1024)
            .with_bias(false)
            .with_layout(LinearLayout::Col)
            .init::<NdArray<f32>>(&device);
        let applied = linear.apply(vec![local], None, None, false);
        assert!(applied.errors.is_empty(), "{:?}", applied.errors);
        assert_eq!(applied.applied, ["weight"]);
        let internal = linear
            .weight
            .val()
            .dequantize()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert!(internal.iter().all(|value| value.is_finite()));
        let mut source_transposed = Vec::with_capacity(source.len());
        for input in 0..4096 {
            for output in 0..1024 {
                source_transposed.push(source[output * 4096 + input]);
            }
        }
        let mapped_metrics = metrics(&internal, &source_transposed);
        eprintln!(
            "mapped Q8 [in,out] vs BF16 transpose: max_abs={} rmse={} cosine={}",
            mapped_metrics.0, mapped_metrics.1, mapped_metrics.2
        );

        let input = (0..4096)
            .map(|index| ((index % 251) as f32 - 125.0) / 512.0)
            .collect::<Vec<_>>();
        let output = linear
            .forward(Tensor::<NdArray<f32>, 2>::from_data(
                TensorData::new(input.clone(), [1, 4096]),
                &device,
            ))
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let reference = (0..1024)
            .map(|output| {
                source[output * 4096..(output + 1) * 4096]
                    .iter()
                    .zip(&input)
                    .map(|(&weight, &value)| weight * value)
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        assert!(output.iter().all(|value| value.is_finite()));
        let output_metrics = metrics(&output, &reference);
        eprintln!(
            "NdArray Q8 linear vs BF16 matmul: max_abs={} rmse={} cosine={}",
            output_metrics.0, output_metrics.1, output_metrics.2
        );

        #[cfg(feature = "wgpu")]
        if std::env::var_os("BURN_BOOGU_Q8_RUN_WGPU").is_some() {
            type WgpuBackend = burn_wgpu::CubeBackend<burn_wgpu::WgpuRuntime, f32, i32, u32>;

            let wgpu_device = crate::require_native_wgpu_device().unwrap();
            let local = TensorSnapshot::from_data(
                snapshot.to_data().unwrap(),
                vec!["weight".into()],
                Vec::new(),
                ParamId::new(),
            );
            let mut wgpu_linear = LinearConfig::new(4096, 1024)
                .with_bias(false)
                .with_layout(LinearLayout::Col)
                .init::<WgpuBackend>(&wgpu_device);
            let applied = wgpu_linear.apply(vec![local], None, None, false);
            assert!(applied.errors.is_empty(), "{:?}", applied.errors);
            let wgpu_internal = wgpu_linear
                .weight
                .val()
                .dequantize()
                .cast(burn::tensor::DType::F32)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            let wgpu_mapped_metrics = metrics(&wgpu_internal, &source_transposed);
            eprintln!(
                "raw WGPU mapped Q8 [in,out] vs BF16 transpose: finite={} max_abs={} rmse={} cosine={}",
                wgpu_internal.iter().all(|value| value.is_finite()),
                wgpu_mapped_metrics.0,
                wgpu_mapped_metrics.1,
                wgpu_mapped_metrics.2
            );
            let wgpu_input = Tensor::<WgpuBackend, 2>::from_data(
                TensorData::new(input.clone(), [1, 4096]),
                &wgpu_device,
            );
            let wgpu_output_f32 = wgpu_linear
                .forward(wgpu_input.clone())
                .cast(burn::tensor::DType::F32)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            let wgpu_output_f16 = wgpu_linear
                .forward(wgpu_input.cast(burn::tensor::DType::F16))
                .cast(burn::tensor::DType::F32)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            let wgpu_f32_metrics = metrics(&wgpu_output_f32, &reference);
            let wgpu_f16_metrics = metrics(&wgpu_output_f16, &reference);
            eprintln!(
                "raw WGPU F32 x Q8 linear: finite={} max_abs={} rmse={} cosine={}",
                wgpu_output_f32.iter().all(|value| value.is_finite()),
                wgpu_f32_metrics.0,
                wgpu_f32_metrics.1,
                wgpu_f32_metrics.2
            );
            eprintln!(
                "raw WGPU F16 x Q8 linear: finite={} max_abs={} rmse={} cosine={}",
                wgpu_output_f16.iter().all(|value| value.is_finite()),
                wgpu_f16_metrics.0,
                wgpu_f16_metrics.1,
                wgpu_f16_metrics.2
            );

            const ROW_TARGET: &str = "x_embedder.weight";
            let row_entry = inventory
                .iter()
                .find(|entry| entry["target_name"].as_str() == Some(ROW_TARGET))
                .unwrap();
            assert_eq!(row_entry["transform"].as_str(), Some("transpose2d"));
            assert_eq!(row_entry["stored_shape"], serde_json::json!([64, 3360]));
            let mut row_source_file =
                File::open(snapshot_root.join(row_entry["source_file"].as_str().unwrap())).unwrap();
            row_source_file
                .seek(SeekFrom::Start(
                    row_entry["source_offset"].as_u64().unwrap(),
                ))
                .unwrap();
            let mut row_source_raw =
                vec![0_u8; row_entry["source_bytes"].as_u64().unwrap() as usize];
            row_source_file.read_exact(&mut row_source_raw).unwrap();
            let row_source_raw = row_source_raw
                .chunks_exact(2)
                .map(|pair| bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]])).to_f32())
                .collect::<Vec<_>>();
            let mut row_source = Vec::with_capacity(row_source_raw.len());
            for input in 0..64 {
                for output in 0..3360 {
                    row_source.push(row_source_raw[output * 64 + input]);
                }
            }
            let row_object = row_entry["burnpack_object"].as_str().unwrap();
            let row_file = manifest
                .files
                .iter()
                .find(|file| file.path.as_str() == row_object)
                .unwrap();
            let row_object_bytes = std::fs::read(artifact_root.join(row_object)).unwrap();
            burn_image::ArtifactVerifier::verify_bytes(
                row_file,
                &row_object_bytes,
                burn_image::IntegrityPolicy::RequireSha256,
            )
            .unwrap();
            let mut row_store = BurnpackStore::from_bytes(Some(
                burn::tensor::Bytes::from_bytes_vec(row_object_bytes),
            ));
            let row_snapshot = row_store
                .get_all_snapshots()
                .unwrap()
                .get(ROW_TARGET)
                .unwrap();
            let row_raw_dequantized =
                Tensor::<NdArray<f32>, 2>::from_data(row_snapshot.to_data().unwrap(), &device)
                    .dequantize()
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap();
            let row_raw_metrics = metrics(&row_raw_dequantized, &row_source);
            eprintln!(
                "Boogu Row raw Q8 [in,out] vs BF16 transpose: max_abs={} rmse={} cosine={}",
                row_raw_metrics.0, row_raw_metrics.1, row_raw_metrics.2
            );
            let row_local = TensorSnapshot::from_data(
                row_snapshot.to_data().unwrap(),
                vec!["weight".into()],
                Vec::new(),
                ParamId::new(),
            );
            let mut row_linear = LinearConfig::new(64, 3360)
                .with_bias(false)
                .with_layout(LinearLayout::Row)
                .init::<WgpuBackend>(&wgpu_device);
            let applied = row_linear.apply(vec![row_local], None, None, false);
            assert!(applied.errors.is_empty(), "{:?}", applied.errors);
            let row_input = (0..64)
                .map(|index| (index as f32 - 31.5) / 64.0)
                .collect::<Vec<_>>();
            let row_reference = (0..3360)
                .map(|output| {
                    (0..64)
                        .map(|input| row_input[input] * row_source[input * 3360 + output])
                        .sum::<f32>()
                })
                .collect::<Vec<_>>();
            let row_wgpu_input = Tensor::<WgpuBackend, 2>::from_data(
                TensorData::new(row_input, [1, 64]),
                &wgpu_device,
            );
            let row_f32 = crate::model::linear::linear_forward(&row_linear, row_wgpu_input.clone())
                .cast(burn::tensor::DType::F32)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            let row_f16 = crate::model::linear::linear_forward(
                &row_linear,
                row_wgpu_input.cast(burn::tensor::DType::F16),
            )
            .cast(burn::tensor::DType::F32)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
            let row_f32_metrics = metrics(&row_f32, &row_reference);
            let row_f16_metrics = metrics(&row_f16, &row_reference);
            eprintln!(
                "raw WGPU Boogu Row F32 x Q8: finite={} max_abs={} rmse={} cosine={}",
                row_f32.iter().all(|value| value.is_finite()),
                row_f32_metrics.0,
                row_f32_metrics.1,
                row_f32_metrics.2
            );
            eprintln!(
                "raw WGPU Boogu Row F16 x Q8: finite={} max_abs={} rmse={} cosine={}",
                row_f16.iter().all(|value| value.is_finite()),
                row_f16_metrics.0,
                row_f16_metrics.1,
                row_f16_metrics.2
            );
            // Burn's column-layout mapper cannot transpose block-quantized parameters while
            // preserving their scale geometry. This negative control is why production Qwen
            // dequantizes each bounded stage before applying the normal float transpose. The
            // Boogu denoiser uses row-layout parameters and must remain packed through matmul.
            assert!(!wgpu_mapped_metrics.2.is_finite() || wgpu_mapped_metrics.2 < 0.99);
            assert!(!wgpu_f16_metrics.2.is_finite() || wgpu_f16_metrics.2 < 0.99);
            assert!(
                row_f32.iter().all(|value| value.is_finite())
                    && row_raw_metrics.2 > 0.999
                    && row_f32_metrics.2 > 0.999
            );
        }
        assert!(raw_metrics.2 > 0.999);
        assert!(mapped_metrics.2 < 0.99 && output_metrics.2 < 0.99);
    }
}
