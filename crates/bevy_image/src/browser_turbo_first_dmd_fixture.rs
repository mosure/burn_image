//! Authenticated, range-bounded browser oracle for Turbo's first DMD step.
//!
//! This diagnostic reader is deliberately separate from the exhaustive Edit-Turbo 1.5K reader.
//! It binds the exact compact Turbo fixture used by the release parity workflow, validates the
//! complete 320-entry SafeTensors header against the authenticated metadata, and exposes only the
//! five tensors required to localize the first DMD prediction. It never downloads the full tensor
//! body or represents those five range checks as whole-file authentication.

#![cfg_attr(
    all(test, not(target_arch = "wasm32")),
    allow(dead_code, unused_imports)
)]

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
};

use burn_image::{ArtifactPath, ByteRange, ModelId, RemoteBaseUrl, RuntimeError, Sha256Digest};
use half::bf16;
use serde::{Deserialize, Deserializer, Serialize, de};

#[cfg(target_arch = "wasm32")]
use crate::DEFAULT_BROWSER_CHUNK_BYTES;

pub(crate) const TURBO_FIRST_DMD_FIXTURE_SCHEMA_VERSION: u32 = 2;
pub(crate) const TURBO_FIRST_DMD_FIXTURE_VARIANT: &str = "turbo";
pub(crate) const TURBO_FIRST_DMD_FIXTURE_MODEL_ID: &str = "Boogu/Boogu-Image-0.1-Turbo";
pub(crate) const TURBO_FIRST_DMD_FIXTURE_MODEL_REVISION: &str =
    "53ad54522023f64d049f7f38e4d679359ef3fb92";
pub(crate) const TURBO_FIRST_DMD_FIXTURE_UPSTREAM_SOURCE_REVISION: &str =
    "25f8f888298224a94e5ec2abafb98abea9031a0d";
pub(crate) const TURBO_FIRST_DMD_FIXTURE_PROMPT: &str =
    "A matte red cube centered on a plain white studio background, soft shadow, front view.";
pub(crate) const TURBO_FIRST_DMD_FIXTURE_EDGE: usize = 256;
pub(crate) const TURBO_FIRST_DMD_FIXTURE_SEED: u64 = 42;
pub(crate) const TURBO_FIRST_DMD_FIXTURE_TENSOR_COUNT: usize = 320;
pub(crate) const TURBO_FIRST_DMD_REQUIRED_TENSOR_COUNT: usize = 5;
pub(crate) const TURBO_FIRST_DMD_REQUIRED_TENSOR_BYTES: u64 = 499_714;

pub(crate) const TURBO_FIRST_DMD_FIXTURE_METADATA_SIZE: u64 = 73_936;
pub(crate) const TURBO_FIRST_DMD_FIXTURE_TENSORS_SIZE: u64 = 375_578_272;
pub(crate) const TURBO_FIRST_DMD_FIXTURE_OUTPUT_SIZE: u64 = 61_874;
pub(crate) const TURBO_FIRST_DMD_FIXTURE_METADATA_SHA256: &str =
    "b85f757d18c5dfdb94b3366692d0b3ceb6f73d699e98820ba5177d90d36d838d";
pub(crate) const TURBO_FIRST_DMD_FIXTURE_TENSORS_SHA256: &str =
    "d26178de365bea3a7e4fef6c9a292e8220af5481dfe398e74491817adc1fcd75";
pub(crate) const TURBO_FIRST_DMD_FIXTURE_OUTPUT_SHA256: &str =
    "80beb2ebd1f15ead37b48bab2741beb0e06befe1530730eba304596b5a1d5554";

pub(crate) const TURBO_FIRST_DMD_1K_PROFILE: &str = "qualification-1024-bf16";
pub(crate) const TURBO_FIRST_DMD_1K_RESOLUTION_PROFILE: &str = "1k";
pub(crate) const TURBO_FIRST_DMD_1K_PROMPT: &str =
    "A studio photograph of a blue ceramic bird on a plain white table.";
pub(crate) const TURBO_FIRST_DMD_1K_EDGE: usize = 1024;
pub(crate) const TURBO_FIRST_DMD_1K_SEED: u64 = 0;
pub(crate) const TURBO_FIRST_DMD_1K_TENSOR_COUNT: usize = 323;
pub(crate) const TURBO_FIRST_DMD_1K_REQUIRED_TENSOR_BYTES: u64 = 1_941_506;
pub(crate) const TURBO_FIRST_DMD_1K_METADATA_SIZE: u64 = 82_947;
pub(crate) const TURBO_FIRST_DMD_1K_TENSORS_SIZE: u64 = 4_829_366_000;
pub(crate) const TURBO_FIRST_DMD_1K_OUTPUT_SIZE: u64 = 1_153_523;
pub(crate) const TURBO_FIRST_DMD_1K_METADATA_SHA256: &str =
    "a7cf73b0ea0183d58b25f5c41eb732c28ed7a0aef52465365387d84cf2af0758";
pub(crate) const TURBO_FIRST_DMD_1K_TENSORS_SHA256: &str =
    "eb3a81e7285f25df69a4e20a9f7d71d318bf9ccc5f84f2819c38fc2c1311f40e";
pub(crate) const TURBO_FIRST_DMD_1K_OUTPUT_SHA256: &str =
    "4abd717984140ace64143617f1981025917c1f35ceb2271501880b350961d703";

pub(crate) const TURBO_FIRST_DMD_QWEN: &str = "qwen.last_hidden_state";
pub(crate) const TURBO_FIRST_DMD_INPUT: &str = "dmd.step.0.input";
pub(crate) const TURBO_FIRST_DMD_SIGMA: &str = "dmd.step.0.sigma";
pub(crate) const TURBO_FIRST_DMD_VELOCITY: &str = "dmd.step.0.velocity";
pub(crate) const TURBO_FIRST_DMD_PREDICTION: &str = "dmd.step.0.prediction";

const METADATA_PATH: &str = "metadata.json";
const TENSORS_PATH: &str = "tensors.safetensors";
const OUTPUT_PATH: &str = "output.png";
const MAX_SAFETENSORS_HEADER_BYTES: u64 = 4 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum BrowserTurboFirstDmdFixtureProfile {
    #[default]
    Release256,
    Qualification1024,
}

impl BrowserTurboFirstDmdFixtureProfile {
    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value {
            "release-256-bf16" => Some(Self::Release256),
            TURBO_FIRST_DMD_1K_PROFILE => Some(Self::Qualification1024),
            _ => None,
        }
    }

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Release256 => "release-256-bf16",
            Self::Qualification1024 => TURBO_FIRST_DMD_1K_PROFILE,
        }
    }

    const fn tensor_count(self) -> usize {
        match self {
            Self::Release256 => TURBO_FIRST_DMD_FIXTURE_TENSOR_COUNT,
            Self::Qualification1024 => TURBO_FIRST_DMD_1K_TENSOR_COUNT,
        }
    }

    const fn required_tensor_bytes(self) -> u64 {
        match self {
            Self::Release256 => TURBO_FIRST_DMD_REQUIRED_TENSOR_BYTES,
            Self::Qualification1024 => TURBO_FIRST_DMD_1K_REQUIRED_TENSOR_BYTES,
        }
    }
}

fn required_tensor_names() -> BTreeSet<&'static str> {
    BTreeSet::from([
        TURBO_FIRST_DMD_QWEN,
        TURBO_FIRST_DMD_INPUT,
        TURBO_FIRST_DMD_SIGMA,
        TURBO_FIRST_DMD_VELOCITY,
        TURBO_FIRST_DMD_PREDICTION,
    ])
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct BrowserTurboFixtureFileIdentity {
    pub(crate) path: String,
    pub(crate) size: u64,
    pub(crate) sha256: String,
}

/// Complete pinned identity of the compact Turbo fixture used by release parity.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct BrowserTurboFirstDmdFixtureIdentity {
    pub(crate) profile: String,
    pub(crate) schema_version: u32,
    pub(crate) variant: String,
    pub(crate) model_revision: String,
    pub(crate) upstream_source_revision: String,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) seed: u64,
    pub(crate) metadata: BrowserTurboFixtureFileIdentity,
    pub(crate) tensors: BrowserTurboFixtureFileIdentity,
    pub(crate) output: BrowserTurboFixtureFileIdentity,
}

impl BrowserTurboFirstDmdFixtureIdentity {
    pub(crate) fn exact_release_fixture() -> Self {
        Self {
            profile: BrowserTurboFirstDmdFixtureProfile::Release256
                .label()
                .into(),
            schema_version: TURBO_FIRST_DMD_FIXTURE_SCHEMA_VERSION,
            variant: TURBO_FIRST_DMD_FIXTURE_VARIANT.into(),
            model_revision: TURBO_FIRST_DMD_FIXTURE_MODEL_REVISION.into(),
            upstream_source_revision: TURBO_FIRST_DMD_FIXTURE_UPSTREAM_SOURCE_REVISION.into(),
            width: TURBO_FIRST_DMD_FIXTURE_EDGE,
            height: TURBO_FIRST_DMD_FIXTURE_EDGE,
            seed: TURBO_FIRST_DMD_FIXTURE_SEED,
            metadata: file_identity(
                METADATA_PATH,
                TURBO_FIRST_DMD_FIXTURE_METADATA_SIZE,
                TURBO_FIRST_DMD_FIXTURE_METADATA_SHA256,
            ),
            tensors: file_identity(
                TENSORS_PATH,
                TURBO_FIRST_DMD_FIXTURE_TENSORS_SIZE,
                TURBO_FIRST_DMD_FIXTURE_TENSORS_SHA256,
            ),
            output: file_identity(
                OUTPUT_PATH,
                TURBO_FIRST_DMD_FIXTURE_OUTPUT_SIZE,
                TURBO_FIRST_DMD_FIXTURE_OUTPUT_SHA256,
            ),
        }
    }

    pub(crate) fn exact_qualification_1024_fixture() -> Self {
        Self {
            profile: BrowserTurboFirstDmdFixtureProfile::Qualification1024
                .label()
                .into(),
            schema_version: TURBO_FIRST_DMD_FIXTURE_SCHEMA_VERSION,
            variant: TURBO_FIRST_DMD_FIXTURE_VARIANT.into(),
            model_revision: TURBO_FIRST_DMD_FIXTURE_MODEL_REVISION.into(),
            upstream_source_revision: TURBO_FIRST_DMD_FIXTURE_UPSTREAM_SOURCE_REVISION.into(),
            width: TURBO_FIRST_DMD_1K_EDGE,
            height: TURBO_FIRST_DMD_1K_EDGE,
            seed: TURBO_FIRST_DMD_1K_SEED,
            metadata: file_identity(
                METADATA_PATH,
                TURBO_FIRST_DMD_1K_METADATA_SIZE,
                TURBO_FIRST_DMD_1K_METADATA_SHA256,
            ),
            tensors: file_identity(
                TENSORS_PATH,
                TURBO_FIRST_DMD_1K_TENSORS_SIZE,
                TURBO_FIRST_DMD_1K_TENSORS_SHA256,
            ),
            output: file_identity(
                OUTPUT_PATH,
                TURBO_FIRST_DMD_1K_OUTPUT_SIZE,
                TURBO_FIRST_DMD_1K_OUTPUT_SHA256,
            ),
        }
    }
}

fn file_identity(path: &str, size: u64, sha256: &str) -> BrowserTurboFixtureFileIdentity {
    BrowserTurboFixtureFileIdentity {
        path: path.into(),
        size,
        sha256: sha256.into(),
    }
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct BrowserTurboFirstDmdMetadata {
    pub(crate) schema_version: u32,
    pub(crate) variant: String,
    #[serde(default)]
    pub(crate) resolution_profile: Option<String>,
    pub(crate) model_revision: String,
    pub(crate) upstream_source_revision: String,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) dtype: String,
    pub(crate) prompt: String,
    pub(crate) seed: u64,
    pub(crate) capture_blocks: bool,
    pub(crate) capture_qwen: bool,
    pub(crate) pipeline: String,
    pub(crate) vision_token_ids: Vec<i64>,
    pub(crate) mask_vision_tokens_feature: bool,
    pub(crate) tensors: BTreeMap<String, BrowserTurboTensorDigest>,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct BrowserTurboTensorDigest {
    pub(crate) dtype: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) sha256: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TurboFixtureDType {
    Bf16,
    I64,
    U8,
}

impl TurboFixtureDType {
    fn element_bytes(self) -> u64 {
        match self {
            Self::Bf16 => 2,
            Self::I64 => 8,
            Self::U8 => 1,
        }
    }

    fn metadata_name(self) -> &'static str {
        match self {
            Self::Bf16 => "torch.bfloat16",
            Self::I64 => "torch.int64",
            Self::U8 => "torch.uint8",
        }
    }
}

#[derive(Clone, Debug)]
struct TurboFixtureTensorSpec {
    name: String,
    dtype: TurboFixtureDType,
    shape: Vec<usize>,
    file_offset: u64,
    byte_length: u64,
    sha256: Sha256Digest,
}

/// Evidence for the bounded fixture reads. Whole-file SafeTensors verification stays false
/// because this reader intentionally fetches only five authenticated tensor ranges.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct BrowserTurboFirstDmdVerificationSnapshot {
    pub(crate) verified_metadata_files: u64,
    pub(crate) verified_metadata_bytes: u64,
    pub(crate) verified_metadata_sha256: Option<String>,
    pub(crate) verified_safetensors_headers: u64,
    pub(crate) verified_safetensors_header_bytes: u64,
    pub(crate) whole_safetensors_identity_pinned: bool,
    pub(crate) whole_safetensors_file_verified: bool,
    pub(crate) verified_required_tensors: u64,
    pub(crate) verified_required_tensor_bytes: u64,
    pub(crate) expected_required_tensors: u64,
    pub(crate) expected_required_tensor_bytes: u64,
    pub(crate) verified_required_tensor_names: Vec<String>,
}

impl BrowserTurboFirstDmdVerificationSnapshot {
    pub(crate) fn required_inputs_authenticated_for(
        &self,
        identity: &BrowserTurboFirstDmdFixtureIdentity,
    ) -> bool {
        self.verified_metadata_files == 1
            && self.verified_metadata_bytes == identity.metadata.size
            && self.verified_metadata_sha256.as_deref() == Some(identity.metadata.sha256.as_str())
            && self.verified_safetensors_headers == 1
            && self.verified_safetensors_header_bytes > 8
            && self.whole_safetensors_identity_pinned
            && !self.whole_safetensors_file_verified
            && self.verified_required_tensors == TURBO_FIRST_DMD_REQUIRED_TENSOR_COUNT as u64
            && self.verified_required_tensor_bytes == self.expected_required_tensor_bytes
            && self.expected_required_tensors == TURBO_FIRST_DMD_REQUIRED_TENSOR_COUNT as u64
            && self.expected_required_tensor_bytes > 0
            && self
                .verified_required_tensor_names
                .iter()
                .map(String::as_str)
                .collect::<BTreeSet<_>>()
                == required_tensor_names()
    }
}

#[derive(Debug)]
struct VerificationState {
    snapshot: BrowserTurboFirstDmdVerificationSnapshot,
    verified_names: BTreeSet<String>,
}

/// Cloneable reader for only the exact five first-DMD oracle tensors.
#[derive(Clone)]
pub(crate) struct BrowserTurboFirstDmdFixture {
    base_url: RemoteBaseUrl,
    identity: BrowserTurboFirstDmdFixtureIdentity,
    metadata: Arc<BrowserTurboFirstDmdMetadata>,
    required_tensors: Arc<BTreeMap<String, TurboFixtureTensorSpec>>,
    verification: Arc<Mutex<VerificationState>>,
}

impl BrowserTurboFirstDmdFixture {
    #[cfg(target_arch = "wasm32")]
    pub(crate) async fn open(
        base_url: RemoteBaseUrl,
        profile: BrowserTurboFirstDmdFixtureProfile,
    ) -> Result<Self, RuntimeError> {
        let identity = match profile {
            BrowserTurboFirstDmdFixtureProfile::Release256 => {
                BrowserTurboFirstDmdFixtureIdentity::exact_release_fixture()
            }
            BrowserTurboFirstDmdFixtureProfile::Qualification1024 => {
                BrowserTurboFirstDmdFixtureIdentity::exact_qualification_1024_fixture()
            }
        };
        let metadata_bytes = fetch_exact_file(
            &base_url,
            &artifact_path(METADATA_PATH)?,
            identity.metadata.size,
        )
        .await?;
        verify_digest(METADATA_PATH, &metadata_bytes, &identity.metadata.sha256)?;
        let metadata: BrowserTurboFirstDmdMetadata = serde_json::from_slice(&metadata_bytes)
            .map_err(|error| {
                fixture_error(format!("invalid authenticated metadata.json: {error}"))
            })?;
        validate_metadata(&metadata, profile)?;

        let tensors_path = artifact_path(TENSORS_PATH)?;
        let prefix =
            fetch_exact_span(&base_url, &tensors_path, 0, 8, identity.tensors.size).await?;
        let header_length = parse_header_length(&prefix, identity.tensors.size)?;
        let header = fetch_exact_span(
            &base_url,
            &tensors_path,
            8,
            header_length,
            identity.tensors.size,
        )
        .await?;
        let data_start = 8_u64
            .checked_add(header_length)
            .ok_or_else(|| fixture_error("SafeTensors data offset overflow"))?;
        let required_tensors = build_required_tensor_index(
            &header,
            data_start,
            identity.tensors.size,
            &metadata.tensors,
            profile,
        )?;
        let expected_required_tensor_bytes =
            required_tensors.values().try_fold(0_u64, |total, tensor| {
                total
                    .checked_add(tensor.byte_length)
                    .ok_or_else(|| fixture_error("required tensor byte count overflow"))
            })?;
        if expected_required_tensor_bytes != profile.required_tensor_bytes() {
            return Err(fixture_error(format!(
                "required tensor bytes are {expected_required_tensor_bytes}; expected {}",
                profile.required_tensor_bytes()
            )));
        }

        let verified_metadata_bytes = identity.metadata.size;
        let verified_metadata_sha256 = identity.metadata.sha256.clone();
        Ok(Self {
            base_url,
            identity,
            metadata: Arc::new(metadata),
            required_tensors: Arc::new(required_tensors),
            verification: Arc::new(Mutex::new(VerificationState {
                snapshot: BrowserTurboFirstDmdVerificationSnapshot {
                    verified_metadata_files: 1,
                    verified_metadata_bytes,
                    verified_metadata_sha256: Some(verified_metadata_sha256),
                    verified_safetensors_headers: 1,
                    verified_safetensors_header_bytes: data_start,
                    whole_safetensors_identity_pinned: true,
                    whole_safetensors_file_verified: false,
                    expected_required_tensors: TURBO_FIRST_DMD_REQUIRED_TENSOR_COUNT as u64,
                    expected_required_tensor_bytes: profile.required_tensor_bytes(),
                    ..BrowserTurboFirstDmdVerificationSnapshot::default()
                },
                verified_names: BTreeSet::new(),
            })),
        })
    }

    pub(crate) fn identity(&self) -> BrowserTurboFirstDmdFixtureIdentity {
        self.identity.clone()
    }

    pub(crate) fn metadata(&self) -> &BrowserTurboFirstDmdMetadata {
        self.metadata.as_ref()
    }

    pub(crate) fn snapshot(
        &self,
    ) -> Result<BrowserTurboFirstDmdVerificationSnapshot, RuntimeError> {
        self.verification
            .lock()
            .map(|state| state.snapshot.clone())
            .map_err(|_| fixture_error("Turbo fixture verification state is poisoned"))
    }

    /// Fetch one of the five allowed tensors and widen its authenticated BF16 body to finite F32.
    #[cfg(target_arch = "wasm32")]
    pub(crate) async fn f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), RuntimeError> {
        let spec =
            self.required_tensors.get(name).cloned().ok_or_else(|| {
                fixture_error(format!("first-DMD fixture does not expose {name:?}"))
            })?;
        if spec.dtype != TurboFixtureDType::Bf16 {
            return Err(fixture_error(format!(
                "first-DMD tensor {name:?} has {:?}; expected BF16",
                spec.dtype
            )));
        }
        let bytes = fetch_exact_span(
            &self.base_url,
            &artifact_path(TENSORS_PATH)?,
            spec.file_offset,
            spec.byte_length,
            self.identity.tensors.size,
        )
        .await?;
        let actual = Sha256Digest::calculate(&bytes);
        if actual != spec.sha256 {
            return Err(fixture_error(format!(
                "first-DMD tensor {name:?} SHA-256 {actual} differs from authenticated {}",
                spec.sha256
            )));
        }
        let values = decode_bf16(name, &bytes)?
            .into_iter()
            .map(bf16::to_f32)
            .collect();
        self.record_verified(&spec)?;
        Ok((spec.shape, values))
    }

    fn record_verified(&self, spec: &TurboFixtureTensorSpec) -> Result<(), RuntimeError> {
        let mut state = self
            .verification
            .lock()
            .map_err(|_| fixture_error("Turbo fixture verification state is poisoned"))?;
        if state.verified_names.insert(spec.name.clone()) {
            state.snapshot.verified_required_tensors = state
                .snapshot
                .verified_required_tensors
                .checked_add(1)
                .ok_or_else(|| fixture_error("verified tensor counter overflow"))?;
            state.snapshot.verified_required_tensor_bytes = state
                .snapshot
                .verified_required_tensor_bytes
                .checked_add(spec.byte_length)
                .ok_or_else(|| fixture_error("verified tensor byte counter overflow"))?;
            state.snapshot.verified_required_tensor_names =
                state.verified_names.iter().cloned().collect();
        }
        Ok(())
    }
}

fn validate_metadata(
    metadata: &BrowserTurboFirstDmdMetadata,
    profile: BrowserTurboFirstDmdFixtureProfile,
) -> Result<(), RuntimeError> {
    let (resolution_profile, edge, prompt, seed) = match profile {
        BrowserTurboFirstDmdFixtureProfile::Release256 => (
            None,
            TURBO_FIRST_DMD_FIXTURE_EDGE,
            TURBO_FIRST_DMD_FIXTURE_PROMPT,
            TURBO_FIRST_DMD_FIXTURE_SEED,
        ),
        BrowserTurboFirstDmdFixtureProfile::Qualification1024 => (
            Some(TURBO_FIRST_DMD_1K_RESOLUTION_PROFILE),
            TURBO_FIRST_DMD_1K_EDGE,
            TURBO_FIRST_DMD_1K_PROMPT,
            TURBO_FIRST_DMD_1K_SEED,
        ),
    };
    let valid = metadata.schema_version == TURBO_FIRST_DMD_FIXTURE_SCHEMA_VERSION
        && metadata.variant == TURBO_FIRST_DMD_FIXTURE_VARIANT
        && metadata.resolution_profile.as_deref() == resolution_profile
        && metadata.model_revision == TURBO_FIRST_DMD_FIXTURE_MODEL_REVISION
        && metadata.upstream_source_revision == TURBO_FIRST_DMD_FIXTURE_UPSTREAM_SOURCE_REVISION
        && metadata.width == edge
        && metadata.height == edge
        && metadata.dtype == "bf16"
        && metadata.prompt == prompt
        && metadata.seed == seed
        && metadata.capture_blocks
        && metadata.capture_qwen
        && metadata.pipeline == "BooguImageTurboPipeline"
        && metadata.vision_token_ids.is_empty()
        && !metadata.mask_vision_tokens_feature
        && metadata.tensors.len() == profile.tensor_count();
    if !valid {
        return Err(fixture_error(
            "authenticated Turbo metadata differs from the pinned first-DMD fixture contract",
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Deserialize)]
enum SafeTensorsDType {
    BF16,
    I64,
    U8,
}

impl From<SafeTensorsDType> for TurboFixtureDType {
    fn from(value: SafeTensorsDType) -> Self {
        match value {
            SafeTensorsDType::BF16 => Self::Bf16,
            SafeTensorsDType::I64 => Self::I64,
            SafeTensorsDType::U8 => Self::U8,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SafeTensorsEntry {
    dtype: SafeTensorsDType,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

#[derive(Clone, Debug)]
struct SafeTensorsHeader {
    tensors: BTreeMap<String, SafeTensorsEntry>,
}

impl<'de> Deserialize<'de> for SafeTensorsHeader {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct HeaderVisitor;

        impl<'de> de::Visitor<'de> for HeaderVisitor {
            type Value = SafeTensorsHeader;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a SafeTensors header object")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: de::MapAccess<'de>,
            {
                let mut tensors = BTreeMap::new();
                let mut saw_metadata = false;
                while let Some(name) = map.next_key::<String>()? {
                    if name == "__metadata__" {
                        if saw_metadata {
                            return Err(de::Error::duplicate_field("__metadata__"));
                        }
                        saw_metadata = true;
                        let _: BTreeMap<String, String> = map.next_value()?;
                    } else {
                        if tensors.contains_key(&name) {
                            return Err(de::Error::custom(format!(
                                "duplicate SafeTensors key {name:?}"
                            )));
                        }
                        tensors.insert(name, map.next_value()?);
                    }
                }
                Ok(SafeTensorsHeader { tensors })
            }
        }

        deserializer.deserialize_map(HeaderVisitor)
    }
}

fn parse_header_length(prefix: &[u8], file_size: u64) -> Result<u64, RuntimeError> {
    let prefix: [u8; 8] = prefix
        .try_into()
        .map_err(|_| fixture_error("SafeTensors length prefix must contain eight bytes"))?;
    let header_length = u64::from_le_bytes(prefix);
    let data_start = 8_u64
        .checked_add(header_length)
        .ok_or_else(|| fixture_error("SafeTensors header length overflow"))?;
    if header_length == 0 || header_length > MAX_SAFETENSORS_HEADER_BYTES || data_start >= file_size
    {
        return Err(fixture_error(format!(
            "invalid SafeTensors header length {header_length} for {file_size}-byte fixture"
        )));
    }
    Ok(header_length)
}

fn build_required_tensor_index(
    header_bytes: &[u8],
    data_start: u64,
    file_size: u64,
    authenticated: &BTreeMap<String, BrowserTurboTensorDigest>,
    profile: BrowserTurboFirstDmdFixtureProfile,
) -> Result<BTreeMap<String, TurboFixtureTensorSpec>, RuntimeError> {
    let header: SafeTensorsHeader = serde_json::from_slice(header_bytes)
        .map_err(|error| fixture_error(format!("invalid SafeTensors header JSON: {error}")))?;
    if header.tensors.len() != profile.tensor_count()
        || authenticated.len() != profile.tensor_count()
    {
        return Err(fixture_error(format!(
            "SafeTensors/authenticated tensor counts are {}/{}; expected {}",
            header.tensors.len(),
            authenticated.len(),
            profile.tensor_count()
        )));
    }
    if header.tensors.keys().collect::<BTreeSet<_>>()
        != authenticated.keys().collect::<BTreeSet<_>>()
    {
        return Err(fixture_error(
            "SafeTensors tensor keyset differs from authenticated metadata",
        ));
    }
    let data_length = file_size
        .checked_sub(data_start)
        .ok_or_else(|| fixture_error("SafeTensors data start exceeds fixed file size"))?;
    let mut ordered = header.tensors.iter().collect::<Vec<_>>();
    // SafeTensors permits zero-element tensors. Sort an empty range before a
    // non-empty range that starts at the same byte so both remain contiguous.
    ordered.sort_by_key(|(_, entry)| (entry.data_offsets[0], entry.data_offsets[1]));
    let required_names = required_tensor_names();
    let mut required = BTreeMap::new();
    let mut cursor = 0_u64;
    for (name, entry) in ordered {
        let [start, end] = entry.data_offsets;
        if start != cursor || end < start || end > data_length {
            return Err(fixture_error(format!(
                "SafeTensors tensor {name:?} has non-contiguous/out-of-bounds offsets [{start},{end})"
            )));
        }
        let dtype = TurboFixtureDType::from(entry.dtype);
        let elements = entry.shape.iter().try_fold(1_u64, |count, &dimension| {
            count
                .checked_mul(dimension as u64)
                .ok_or_else(|| fixture_error(format!("tensor {name:?} shape overflows")))
        })?;
        let expected_length = elements
            .checked_mul(dtype.element_bytes())
            .ok_or_else(|| fixture_error(format!("tensor {name:?} byte length overflows")))?;
        let byte_length = end - start;
        if byte_length != expected_length {
            return Err(fixture_error(format!(
                "SafeTensors tensor {name:?} stores {byte_length} bytes; dtype/shape require {expected_length}"
            )));
        }
        let expected = authenticated
            .get(name)
            .expect("equal keysets guarantee authenticated metadata");
        if expected.dtype != dtype.metadata_name() || expected.shape != entry.shape {
            return Err(fixture_error(format!(
                "SafeTensors tensor {name:?} dtype/shape differs from authenticated metadata"
            )));
        }
        let sha256 = Sha256Digest::from_hex(&expected.sha256).map_err(|error| {
            fixture_error(format!(
                "authenticated tensor {name:?} has invalid SHA-256: {error}"
            ))
        })?;
        if required_names.contains(name.as_str()) {
            required.insert(
                name.clone(),
                TurboFixtureTensorSpec {
                    name: name.clone(),
                    dtype,
                    shape: entry.shape.clone(),
                    file_offset: data_start
                        .checked_add(start)
                        .ok_or_else(|| fixture_error("required tensor offset overflow"))?,
                    byte_length,
                    sha256,
                },
            );
        }
        cursor = end;
    }
    if cursor != data_length {
        return Err(fixture_error(format!(
            "SafeTensors header accounts for {cursor} bytes; file contains {data_length}"
        )));
    }
    validate_required_tensor_contract(&required, profile)?;
    Ok(required)
}

fn validate_required_tensor_contract(
    required: &BTreeMap<String, TurboFixtureTensorSpec>,
    profile: BrowserTurboFirstDmdFixtureProfile,
) -> Result<(), RuntimeError> {
    let release_256 = [
        (
            TURBO_FIRST_DMD_QWEN,
            vec![1, 49, 4096],
            "2aed1527c0b348a0b008b68225beb82ed1e6249fe5b6c07f3dfd228ba83a7c58",
        ),
        (
            TURBO_FIRST_DMD_INPUT,
            vec![1, 16, 32, 32],
            "1166b0e7e49e48258bd0393f53ef22e20c96099d32b801343ff289286cdb03d5",
        ),
        (
            TURBO_FIRST_DMD_SIGMA,
            vec![],
            "b122c09cdd7935ffcaed22e316bbc8e73770b61a392f958bf93018a3e417c573",
        ),
        (
            TURBO_FIRST_DMD_VELOCITY,
            vec![1, 16, 32, 32],
            "6326d535bff3a2e9678cf05fe316ee57cbf9d4e7b9d143564963f4e6ddd8fb2d",
        ),
        (
            TURBO_FIRST_DMD_PREDICTION,
            vec![1, 16, 32, 32],
            "c2edbe325a61b4be66c6c77475c6b2c7db38513ee69e9c0c828f4c440f23ecf5",
        ),
    ];
    let qualification_1024 = [
        (
            TURBO_FIRST_DMD_QWEN,
            vec![1, 45, 4096],
            "ea0c1f8223591a5b03a096564080522b9f6e01bd3010b8ca1ad60d25866d58d5",
        ),
        (
            TURBO_FIRST_DMD_INPUT,
            vec![1, 16, 128, 128],
            "4927f2e660051659d6d85bc5264b4d514ee13cab49002f0ae84a73d9e81b0453",
        ),
        (
            TURBO_FIRST_DMD_SIGMA,
            vec![],
            "b122c09cdd7935ffcaed22e316bbc8e73770b61a392f958bf93018a3e417c573",
        ),
        (
            TURBO_FIRST_DMD_VELOCITY,
            vec![1, 16, 128, 128],
            "c6a21125e4510956b728baca5f24d5263a342766299d10edbbc32f815562e254",
        ),
        (
            TURBO_FIRST_DMD_PREDICTION,
            vec![1, 16, 128, 128],
            "9ab797b46ebae8bc39f8c90440781c1a6cefc456f2b84591a7a8eba0e7fbb656",
        ),
    ];
    let expected = match profile {
        BrowserTurboFirstDmdFixtureProfile::Release256 => release_256,
        BrowserTurboFirstDmdFixtureProfile::Qualification1024 => qualification_1024,
    };
    if required.len() != expected.len() {
        return Err(fixture_error(format!(
            "first-DMD index exposes {} tensors; expected {}",
            required.len(),
            expected.len()
        )));
    }
    for (name, shape, digest) in expected {
        let spec = required
            .get(name)
            .ok_or_else(|| fixture_error(format!("first-DMD index omits {name:?}")))?;
        if spec.dtype != TurboFixtureDType::Bf16
            || spec.shape != shape
            || spec.sha256.to_hex() != digest
        {
            return Err(fixture_error(format!(
                "first-DMD tensor {name:?} differs from its pinned dtype/shape/digest"
            )));
        }
    }
    Ok(())
}

fn decode_bf16(name: &str, bytes: &[u8]) -> Result<Vec<bf16>, RuntimeError> {
    if !bytes.len().is_multiple_of(2) {
        return Err(fixture_error(format!(
            "BF16 tensor {name:?} has a partial element"
        )));
    }
    bytes
        .chunks_exact(2)
        .enumerate()
        .map(|(index, word)| {
            let value = bf16::from_bits(u16::from_le_bytes([word[0], word[1]]));
            if !value.is_finite() {
                return Err(fixture_error(format!(
                    "BF16 tensor {name:?} contains a non-finite value at element {index}"
                )));
            }
            Ok(value)
        })
        .collect()
}

fn verify_digest(name: &str, bytes: &[u8], expected: &str) -> Result<(), RuntimeError> {
    let expected = Sha256Digest::from_hex(expected)
        .map_err(|error| fixture_error(format!("invalid pinned {name} SHA-256: {error}")))?;
    let actual = Sha256Digest::calculate(bytes);
    if actual != expected {
        return Err(fixture_error(format!(
            "fixture {name} SHA-256 {actual} differs from pinned {expected}"
        )));
    }
    Ok(())
}

fn artifact_path(value: &str) -> Result<ArtifactPath, RuntimeError> {
    ArtifactPath::new(value)
        .map_err(|error| fixture_error(format!("invalid fixture path {value:?}: {error}")))
}

fn fixture_error(message: impl Into<String>) -> RuntimeError {
    RuntimeError::ModelExecution {
        model: ModelId::new(TURBO_FIRST_DMD_FIXTURE_MODEL_ID)
            .expect("fixed Turbo model id is valid"),
        message: message.into(),
    }
}

fn exact_range_chunks(
    offset: u64,
    length: u64,
    total: u64,
    maximum_chunk_bytes: u64,
) -> Result<Vec<ByteRange>, RuntimeError> {
    if length == 0 || maximum_chunk_bytes == 0 {
        return Err(fixture_error(
            "exact browser ranges and their chunk bound must be non-zero",
        ));
    }
    let end = offset
        .checked_add(length)
        .ok_or_else(|| fixture_error("exact browser range overflows"))?;
    if end > total {
        return Err(fixture_error(format!(
            "exact browser range [{offset},{end}) exceeds {total}-byte file"
        )));
    }
    let mut ranges = Vec::new();
    let mut cursor = offset;
    while cursor < end {
        let chunk_length = (end - cursor).min(maximum_chunk_bytes);
        ranges.push(
            ByteRange::new(cursor, chunk_length)
                .map_err(|error| fixture_error(format!("invalid exact range: {error}")))?,
        );
        cursor = cursor
            .checked_add(chunk_length)
            .ok_or_else(|| fixture_error("exact browser chunk cursor overflow"))?;
    }
    Ok(ranges)
}

#[cfg(target_arch = "wasm32")]
async fn fetch_exact_file(
    base_url: &RemoteBaseUrl,
    path: &ArtifactPath,
    size: u64,
) -> Result<Vec<u8>, RuntimeError> {
    fetch_exact_span(base_url, path, 0, size, size).await
}

#[cfg(target_arch = "wasm32")]
async fn fetch_exact_span(
    base_url: &RemoteBaseUrl,
    path: &ArtifactPath,
    offset: u64,
    length: u64,
    total: u64,
) -> Result<Vec<u8>, RuntimeError> {
    let capacity = usize::try_from(length).map_err(|_| {
        fixture_error(format!(
            "requested fixture span of {length} bytes cannot fit in Wasm memory"
        ))
    })?;
    let mut output = Vec::with_capacity(capacity);
    for range in exact_range_chunks(offset, length, total, DEFAULT_BROWSER_CHUNK_BYTES)? {
        let bytes = fetch_one_exact_range(base_url, path, range, total).await?;
        output.extend_from_slice(&bytes);
    }
    if output.len() != capacity {
        return Err(fixture_error(format!(
            "fixture span returned {} bytes; expected {capacity}",
            output.len()
        )));
    }
    Ok(output)
}

#[cfg(target_arch = "wasm32")]
async fn fetch_one_exact_range(
    base_url: &RemoteBaseUrl,
    path: &ArtifactPath,
    range: ByteRange,
    total: u64,
) -> Result<Vec<u8>, RuntimeError> {
    use js_sys::Uint8Array;
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{Headers, Request, RequestInit, Response};

    let headers = Headers::new().map_err(browser_js_error)?;
    headers
        .set("Range", &range.http_range_header())
        .map_err(browser_js_error)?;
    let init = RequestInit::new();
    init.set_method("GET");
    init.set_headers_headers(&headers);
    let url = base_url.resolve(path);
    let request = Request::new_with_str_and_init(&url, &init).map_err(browser_js_error)?;
    let window = web_sys::window().ok_or_else(|| fixture_error("fixture fetch requires Window"))?;
    let response = JsFuture::from(window.fetch_with_request(&request))
        .await
        .map_err(browser_js_error)?
        .dyn_into::<Response>()
        .map_err(browser_js_error)?;
    if response.status() != 206 {
        return Err(fixture_error(format!(
            "fixture range returned HTTP {} for {url}; expected 206",
            response.status()
        )));
    }
    let expected_content_range = format!(
        "bytes {}-{}/{total}",
        range.offset(),
        range.end_exclusive() - 1
    );
    let content_range = response
        .headers()
        .get("Content-Range")
        .map_err(browser_js_error)?;
    if content_range.as_deref() != Some(expected_content_range.as_str()) {
        return Err(fixture_error(format!(
            "fixture range has Content-Range {content_range:?}; expected {expected_content_range:?}"
        )));
    }
    let buffer = JsFuture::from(response.array_buffer().map_err(browser_js_error)?)
        .await
        .map_err(browser_js_error)?;
    let bytes = Uint8Array::new(&buffer).to_vec();
    if bytes.len() as u64 != range.length() {
        return Err(fixture_error(format!(
            "fixture range returned {} bytes; expected {}",
            bytes.len(),
            range.length()
        )));
    }
    Ok(bytes)
}

#[cfg(target_arch = "wasm32")]
fn browser_js_error(value: wasm_bindgen::JsValue) -> RuntimeError {
    fixture_error(
        value
            .as_string()
            .unwrap_or_else(|| format!("browser JavaScript error: {value:?}")),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Seek, SeekFrom};

    #[test]
    fn release_workflow_and_reader_pin_the_same_turbo_fixture_correctness() {
        let workflow = include_str!("../../../.github/workflows/parity.yml");
        for digest in [
            TURBO_FIRST_DMD_FIXTURE_METADATA_SHA256,
            TURBO_FIRST_DMD_FIXTURE_TENSORS_SHA256,
            TURBO_FIRST_DMD_FIXTURE_OUTPUT_SHA256,
        ] {
            assert!(workflow.contains(digest), "parity workflow omits {digest}");
        }
    }

    #[test]
    fn required_tensor_contract_is_exact_and_bounded_correctness() {
        assert_eq!(required_tensor_names().len(), 5);
        assert_eq!(TURBO_FIRST_DMD_REQUIRED_TENSOR_BYTES, 499_714);
        const {
            assert!(
                TURBO_FIRST_DMD_REQUIRED_TENSOR_BYTES * 700 < TURBO_FIRST_DMD_FIXTURE_TENSORS_SIZE,
                "diagnostic must not approach a full fixture-body download"
            );
        }
        let snapshot = BrowserTurboFirstDmdVerificationSnapshot {
            verified_metadata_files: 1,
            verified_metadata_bytes: TURBO_FIRST_DMD_FIXTURE_METADATA_SIZE,
            verified_metadata_sha256: Some(TURBO_FIRST_DMD_FIXTURE_METADATA_SHA256.into()),
            verified_safetensors_headers: 1,
            verified_safetensors_header_bytes: 35_192,
            whole_safetensors_identity_pinned: true,
            whole_safetensors_file_verified: false,
            verified_required_tensors: 5,
            verified_required_tensor_bytes: TURBO_FIRST_DMD_REQUIRED_TENSOR_BYTES,
            expected_required_tensors: 5,
            expected_required_tensor_bytes: TURBO_FIRST_DMD_REQUIRED_TENSOR_BYTES,
            verified_required_tensor_names: required_tensor_names()
                .into_iter()
                .map(str::to_owned)
                .collect(),
        };
        let identity = BrowserTurboFirstDmdFixtureIdentity::exact_release_fixture();
        assert!(snapshot.required_inputs_authenticated_for(&identity));
        let mut false_whole_file_claim = snapshot;
        false_whole_file_claim.whole_safetensors_file_verified = true;
        assert!(!false_whole_file_claim.required_inputs_authenticated_for(&identity));
    }

    #[test]
    fn full_resolution_first_dmd_identity_is_separate_and_bounded_correctness() {
        let identity = BrowserTurboFirstDmdFixtureIdentity::exact_qualification_1024_fixture();
        assert_eq!(identity.profile, TURBO_FIRST_DMD_1K_PROFILE);
        assert_eq!(
            (identity.width, identity.height, identity.seed),
            (1024, 1024, 0)
        );
        assert_eq!(identity.metadata.size, 82_947);
        assert_eq!(identity.tensors.size, 4_829_366_000);
        assert_eq!(identity.output.size, 1_153_523);
        assert_eq!(TURBO_FIRST_DMD_1K_REQUIRED_TENSOR_BYTES, 1_941_506);
        const {
            assert!(
                TURBO_FIRST_DMD_1K_REQUIRED_TENSOR_BYTES * 2_000 < TURBO_FIRST_DMD_1K_TENSORS_SIZE,
                "full-resolution diagnostic must remain a bounded range subset"
            );
        }
        for digest in [
            TURBO_FIRST_DMD_1K_METADATA_SHA256,
            TURBO_FIRST_DMD_1K_TENSORS_SHA256,
            TURBO_FIRST_DMD_1K_OUTPUT_SHA256,
        ] {
            assert_eq!(digest.len(), 64);
            assert!(digest.bytes().all(|byte| byte.is_ascii_hexdigit()));
        }
        assert_eq!(
            BrowserTurboFirstDmdFixtureProfile::parse(TURBO_FIRST_DMD_1K_PROFILE),
            Some(BrowserTurboFirstDmdFixtureProfile::Qualification1024)
        );
        assert!(BrowserTurboFirstDmdFixtureProfile::parse("1024-ish").is_none());
    }

    #[test]
    fn opt_in_full_resolution_metadata_and_header_match_strict_identity_correctness() {
        let Ok(directory) = std::env::var("BURN_IMAGE_TURBO_FIRST_DMD_1K_FIXTURE_DIR") else {
            return;
        };
        let directory = std::path::Path::new(&directory);
        let metadata_bytes = std::fs::read(directory.join(METADATA_PATH)).unwrap();
        assert_eq!(
            metadata_bytes.len() as u64,
            TURBO_FIRST_DMD_1K_METADATA_SIZE
        );
        assert_eq!(
            Sha256Digest::calculate(&metadata_bytes).to_hex(),
            TURBO_FIRST_DMD_1K_METADATA_SHA256
        );
        let metadata: BrowserTurboFirstDmdMetadata =
            serde_json::from_slice(&metadata_bytes).unwrap();
        validate_metadata(
            &metadata,
            BrowserTurboFirstDmdFixtureProfile::Qualification1024,
        )
        .unwrap();

        let mut tensors = std::fs::File::open(directory.join(TENSORS_PATH)).unwrap();
        assert_eq!(
            tensors.metadata().unwrap().len(),
            TURBO_FIRST_DMD_1K_TENSORS_SIZE
        );
        let mut prefix = [0_u8; 8];
        tensors.read_exact(&mut prefix).unwrap();
        let header_length = parse_header_length(&prefix, TURBO_FIRST_DMD_1K_TENSORS_SIZE).unwrap();
        let mut header = vec![0_u8; header_length as usize];
        tensors.seek(SeekFrom::Start(8)).unwrap();
        tensors.read_exact(&mut header).unwrap();
        let index = build_required_tensor_index(
            &header,
            8 + header_length,
            TURBO_FIRST_DMD_1K_TENSORS_SIZE,
            &metadata.tensors,
            BrowserTurboFirstDmdFixtureProfile::Qualification1024,
        )
        .unwrap();
        assert_eq!(index.len(), TURBO_FIRST_DMD_REQUIRED_TENSOR_COUNT);
        assert_eq!(
            index.values().map(|tensor| tensor.byte_length).sum::<u64>(),
            TURBO_FIRST_DMD_1K_REQUIRED_TENSOR_BYTES
        );
    }

    #[test]
    fn range_math_and_bf16_decode_fail_closed_correctness() {
        let ranges = exact_range_chunks(7, 10, 17, 4).unwrap();
        assert_eq!(ranges.len(), 3);
        assert_eq!(ranges[0].http_range_header(), "bytes=7-10");
        assert_eq!(ranges[2].http_range_header(), "bytes=15-16");
        assert!(exact_range_chunks(16, 2, 17, 4).is_err());
        assert!(decode_bf16("partial", &[0]).is_err());
        assert!(decode_bf16("nan", &bf16::NAN.to_bits().to_le_bytes()).is_err());
        assert_eq!(
            decode_bf16("sigma", &bf16::from_f32(0.001).to_bits().to_le_bytes()).unwrap()[0]
                .to_f32(),
            0.000_999_450_7
        );
    }
}
