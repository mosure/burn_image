//! Authenticated, range-streamed browser oracle for the exact Edit-Turbo 1.5K release.
//!
//! The fixture is deliberately separate from the ordinary browser model loader. It reads only
//! exact HTTP ranges, keeps SafeTensors offsets as `u64` (the sealed file is larger than Wasm's
//! address space), and authenticates every requested tensor against the already authenticated
//! metadata before exposing its bytes.

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
#[cfg(target_arch = "wasm32")]
use sha2::{Digest, Sha256};

pub(crate) const EDIT_1K5_FIXTURE_SCHEMA_VERSION: u32 = 2;
pub(crate) const EDIT_1K5_FIXTURE_VARIANT: &str = "edit-turbo-1k5";
pub(crate) const EDIT_1K5_FIXTURE_RESOLUTION_PROFILE: &str = "1k5";
pub(crate) const EDIT_1K5_FIXTURE_MODEL_ID: &str = "Boogu/Boogu-Image-0.1-Edit-Turbo-1K5";
pub(crate) const EDIT_1K5_FIXTURE_MODEL_REVISION: &str = "60981c49e48cffadf2c169532a4ba3f6108afd5e";
pub(crate) const EDIT_1K5_FIXTURE_UPSTREAM_SOURCE_REVISION: &str =
    "25f8f888298224a94e5ec2abafb98abea9031a0d";
pub(crate) const EDIT_1K5_FIXTURE_EDGE: usize = 1536;
pub(crate) const EDIT_1K5_FIXTURE_SEED: u64 = 42;
pub(crate) const EDIT_1K5_FIXTURE_TENSOR_COUNT: usize = 372;

pub(crate) const EDIT_1K5_FIXTURE_METADATA_SIZE: u64 = 93_942;
pub(crate) const EDIT_1K5_FIXTURE_TENSORS_SIZE: u64 = 11_258_528_368;
pub(crate) const EDIT_1K5_FIXTURE_SOURCE_SIZE: u64 = 6_367;
pub(crate) const EDIT_1K5_FIXTURE_OUTPUT_SIZE: u64 = 1_803_055;

pub(crate) const EDIT_1K5_FIXTURE_METADATA_SHA256: &str =
    "1e78233c703ed32ee351c25d54ca4b05e3efeb898ee2836d1cc96c522e2abcae";
pub(crate) const EDIT_1K5_FIXTURE_TENSORS_SHA256: &str =
    "2585ddf2337e41f884218a4abeceb8a10baa7553e43d37f33016be68edc3eeb9";
pub(crate) const EDIT_1K5_FIXTURE_SOURCE_SHA256: &str =
    "96534b93904478caf92c1d0e1b431396f81e7b62f09bb5505443378f245d9647";
pub(crate) const EDIT_1K5_FIXTURE_OUTPUT_SHA256: &str =
    "8e88d6c3580593da723049ef4027a60c5d730b6006ef766d49971a23c6446a70";

pub(crate) const EDIT_1K5_VAE_FIXTURE_TENSOR_COUNT: usize = 47;
pub(crate) const EDIT_1K5_VAE_FIXTURE_METADATA_SIZE: u64 = 19_604;
pub(crate) const EDIT_1K5_VAE_FIXTURE_TENSORS_SIZE: u64 = 49_120_176;
pub(crate) const EDIT_1K5_VAE_FIXTURE_OUTPUT_SIZE: u64 = 1_799_947;
pub(crate) const EDIT_1K5_VAE_FIXTURE_METADATA_SHA256: &str =
    "4a3847347adefd38f5978844f311b934606f9b8a6be0013235dd1fcaf5393ebb";
pub(crate) const EDIT_1K5_VAE_FIXTURE_TENSORS_SHA256: &str =
    "bdd429af5b8f146fea3ac05238cd1d711d3be7f974dc54544ae85c149874a2df";
pub(crate) const EDIT_1K5_VAE_FIXTURE_OUTPUT_SHA256: &str =
    "f6d8e1b45351bfe203136da075b43afaf6f80c9eda481f529bc6707eb91787bc";

const METADATA_PATH: &str = "metadata.json";
const TENSORS_PATH: &str = "tensors.safetensors";
const SOURCE_PATH: &str = "source.png";
const OUTPUT_PATH: &str = "output.png";
const MAX_SAFETENSORS_HEADER_BYTES: u64 = 4 * 1024 * 1024;

/// One immutable file identity carried by the browser qualification report.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct BrowserParityFileIdentity {
    pub(crate) path: String,
    pub(crate) size: u64,
    pub(crate) sha256: String,
}

/// Complete identity of the one fixture accepted by this qualification route.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct BrowserParityFixtureIdentity {
    pub(crate) schema_version: u32,
    pub(crate) variant: String,
    pub(crate) model_revision: String,
    pub(crate) upstream_source_revision: String,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) seed: u64,
    pub(crate) metadata: BrowserParityFileIdentity,
    pub(crate) tensors: BrowserParityFileIdentity,
    pub(crate) source: BrowserParityFileIdentity,
    pub(crate) output: BrowserParityFileIdentity,
}

impl BrowserParityFixtureIdentity {
    pub(crate) fn exact_edit_1k5() -> Self {
        Self {
            schema_version: EDIT_1K5_FIXTURE_SCHEMA_VERSION,
            variant: EDIT_1K5_FIXTURE_VARIANT.into(),
            model_revision: EDIT_1K5_FIXTURE_MODEL_REVISION.into(),
            upstream_source_revision: EDIT_1K5_FIXTURE_UPSTREAM_SOURCE_REVISION.into(),
            width: EDIT_1K5_FIXTURE_EDGE,
            height: EDIT_1K5_FIXTURE_EDGE,
            seed: EDIT_1K5_FIXTURE_SEED,
            metadata: file_identity(
                METADATA_PATH,
                EDIT_1K5_FIXTURE_METADATA_SIZE,
                EDIT_1K5_FIXTURE_METADATA_SHA256,
            ),
            tensors: file_identity(
                TENSORS_PATH,
                EDIT_1K5_FIXTURE_TENSORS_SIZE,
                EDIT_1K5_FIXTURE_TENSORS_SHA256,
            ),
            source: file_identity(
                SOURCE_PATH,
                EDIT_1K5_FIXTURE_SOURCE_SIZE,
                EDIT_1K5_FIXTURE_SOURCE_SHA256,
            ),
            output: file_identity(
                OUTPUT_PATH,
                EDIT_1K5_FIXTURE_OUTPUT_SIZE,
                EDIT_1K5_FIXTURE_OUTPUT_SHA256,
            ),
        }
    }

    pub(crate) fn exact_edit_1k5_vae_reference() -> Self {
        Self {
            schema_version: EDIT_1K5_FIXTURE_SCHEMA_VERSION,
            variant: EDIT_1K5_FIXTURE_VARIANT.into(),
            model_revision: EDIT_1K5_FIXTURE_MODEL_REVISION.into(),
            upstream_source_revision: EDIT_1K5_FIXTURE_UPSTREAM_SOURCE_REVISION.into(),
            width: EDIT_1K5_FIXTURE_EDGE,
            height: EDIT_1K5_FIXTURE_EDGE,
            seed: EDIT_1K5_FIXTURE_SEED,
            metadata: file_identity(
                METADATA_PATH,
                EDIT_1K5_VAE_FIXTURE_METADATA_SIZE,
                EDIT_1K5_VAE_FIXTURE_METADATA_SHA256,
            ),
            tensors: file_identity(
                TENSORS_PATH,
                EDIT_1K5_VAE_FIXTURE_TENSORS_SIZE,
                EDIT_1K5_VAE_FIXTURE_TENSORS_SHA256,
            ),
            source: file_identity(
                SOURCE_PATH,
                EDIT_1K5_FIXTURE_SOURCE_SIZE,
                EDIT_1K5_FIXTURE_SOURCE_SHA256,
            ),
            output: file_identity(
                OUTPUT_PATH,
                EDIT_1K5_VAE_FIXTURE_OUTPUT_SIZE,
                EDIT_1K5_VAE_FIXTURE_OUTPUT_SHA256,
            ),
        }
    }
}

fn file_identity(path: &str, size: u64, sha256: &str) -> BrowserParityFileIdentity {
    BrowserParityFileIdentity {
        path: path.into(),
        size,
        sha256: sha256.into(),
    }
}

/// Authenticated subset of the upstream schema-2 fixture metadata.
///
/// Unknown exporter/environment fields are intentionally ignored only after the entire JSON file
/// has matched its release-pinned digest.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct BrowserParityMetadata {
    pub(crate) schema_version: u32,
    pub(crate) variant: String,
    pub(crate) resolution_profile: String,
    pub(crate) model_revision: String,
    pub(crate) upstream_source_revision: String,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) dtype: String,
    pub(crate) prompt: String,
    pub(crate) seed: u64,
    pub(crate) capture_blocks: bool,
    pub(crate) capture_qwen: bool,
    pub(crate) output: BrowserParityOutputMetadata,
    pub(crate) provenance: BrowserParityProvenance,
    pub(crate) tensors: BTreeMap<String, BrowserParityTensorDigest>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct BrowserParityOutputMetadata {
    pub(crate) align_res: bool,
    pub(crate) requested: BrowserParityDimensions,
    pub(crate) actual: BrowserParityImageDescription,
    pub(crate) validated: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct BrowserParityDimensions {
    pub(crate) width: usize,
    pub(crate) height: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct BrowserParityImageDescription {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) mode: String,
    pub(crate) image_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct BrowserParityProvenance {
    pub(crate) source_image: BrowserParitySourceImage,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct BrowserParitySourceImage {
    pub(crate) fixture_path: String,
    pub(crate) sha256: String,
    pub(crate) size: u64,
    pub(crate) copy_verified: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct BrowserParityTensorDigest {
    pub(crate) dtype: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) sha256: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum BrowserParityDType {
    Bf16,
    F32,
    I64,
    U8,
}

impl BrowserParityDType {
    fn element_bytes(self) -> u64 {
        match self {
            Self::Bf16 => 2,
            Self::F32 => 4,
            Self::I64 => 8,
            Self::U8 => 1,
        }
    }

    pub(crate) fn metadata_name(self) -> &'static str {
        match self {
            Self::Bf16 => "torch.bfloat16",
            Self::F32 => "torch.float32",
            Self::I64 => "torch.int64",
            Self::U8 => "torch.uint8",
        }
    }
}

/// One validated SafeTensors range. Offsets remain 64-bit even on wasm32.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct BrowserParityTensorSpec {
    pub(crate) name: String,
    pub(crate) dtype: BrowserParityDType,
    pub(crate) shape: Vec<usize>,
    pub(crate) file_offset: u64,
    pub(crate) byte_length: u64,
    pub(crate) sha256: Sha256Digest,
}

/// Shared, serializable evidence of which fixture objects have actually been authenticated.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct BrowserParityVerificationSnapshot {
    pub(crate) verified_metadata_files: u64,
    pub(crate) verified_metadata_bytes: u64,
    pub(crate) verified_metadata_sha256: Option<String>,
    pub(crate) verified_safetensors_headers: u64,
    pub(crate) verified_safetensors_header_bytes: u64,
    pub(crate) verified_safetensors_files: u64,
    pub(crate) verified_safetensors_file_bytes: u64,
    pub(crate) verified_safetensors_sha256: Option<String>,
    pub(crate) verified_source_files: u64,
    pub(crate) verified_source_bytes: u64,
    pub(crate) verified_source_sha256: Option<String>,
    pub(crate) verified_output_files: u64,
    pub(crate) verified_output_bytes: u64,
    pub(crate) verified_output_sha256: Option<String>,
    pub(crate) verified_tensors: u64,
    pub(crate) verified_tensor_bytes: u64,
    pub(crate) expected_tensors: u64,
    pub(crate) expected_tensor_bytes: u64,
}

impl BrowserParityVerificationSnapshot {
    pub(crate) fn all_tensors_verified(&self) -> bool {
        self.verified_tensors == self.expected_tensors
            && self.verified_tensor_bytes == self.expected_tensor_bytes
    }

    pub(crate) fn qualification_inputs_verified(&self) -> bool {
        self.qualification_inputs_verified_for(&BrowserParityFixtureIdentity::exact_edit_1k5())
    }

    pub(crate) fn qualification_inputs_verified_for(
        &self,
        identity: &BrowserParityFixtureIdentity,
    ) -> bool {
        self.verified_metadata_files == 1
            && self.verified_metadata_bytes == identity.metadata.size
            && self.verified_metadata_sha256.as_deref() == Some(identity.metadata.sha256.as_str())
            && self.verified_safetensors_headers == 1
            && self.verified_safetensors_header_bytes > 8
            && self.verified_safetensors_files == 1
            && self.verified_safetensors_file_bytes == identity.tensors.size
            && self.verified_safetensors_sha256.as_deref() == Some(identity.tensors.sha256.as_str())
            && self.verified_source_files == 1
            && self.verified_source_bytes == identity.source.size
            && self.verified_source_sha256.as_deref() == Some(identity.source.sha256.as_str())
            && self.verified_output_files == 1
            && self.verified_output_bytes == identity.output.size
            && self.verified_output_sha256.as_deref() == Some(identity.output.sha256.as_str())
            && self.all_tensors_verified()
    }
}

#[derive(Debug)]
struct BrowserParityVerificationState {
    snapshot: BrowserParityVerificationSnapshot,
    verified_tensor_names: BTreeSet<String>,
}

/// Cloneable browser fixture reader. Clones share unique tensor-verification evidence.
#[derive(Clone)]
pub(crate) struct BrowserParityFixture {
    base_url: RemoteBaseUrl,
    identity: BrowserParityFixtureIdentity,
    metadata: Arc<BrowserParityMetadata>,
    tensors: Arc<BTreeMap<String, BrowserParityTensorSpec>>,
    verification: Arc<Mutex<BrowserParityVerificationState>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BrowserParityFixtureScope {
    Exhaustive,
    VaeReference,
}

impl BrowserParityFixture {
    /// Open and authenticate the fixed metadata plus the SafeTensors header without downloading
    /// the multi-gigabyte tensor body.
    #[cfg(target_arch = "wasm32")]
    pub(crate) async fn open(base_url: RemoteBaseUrl) -> Result<Self, RuntimeError> {
        Self::open_scoped(base_url, BrowserParityFixtureScope::Exhaustive).await
    }

    /// Open the pinned compact 47-tensor fixture used only by the encoder-only diagnostic.
    #[cfg(target_arch = "wasm32")]
    pub(crate) async fn open_vae_reference(base_url: RemoteBaseUrl) -> Result<Self, RuntimeError> {
        Self::open_scoped(base_url, BrowserParityFixtureScope::VaeReference).await
    }

    #[cfg(target_arch = "wasm32")]
    async fn open_scoped(
        base_url: RemoteBaseUrl,
        scope: BrowserParityFixtureScope,
    ) -> Result<Self, RuntimeError> {
        let (identity, expected_tensor_count) = match scope {
            BrowserParityFixtureScope::Exhaustive => (
                BrowserParityFixtureIdentity::exact_edit_1k5(),
                EDIT_1K5_FIXTURE_TENSOR_COUNT,
            ),
            BrowserParityFixtureScope::VaeReference => (
                BrowserParityFixtureIdentity::exact_edit_1k5_vae_reference(),
                EDIT_1K5_VAE_FIXTURE_TENSOR_COUNT,
            ),
        };
        let metadata_path = artifact_path(METADATA_PATH)?;
        let metadata_bytes =
            fetch_exact_file(&base_url, &metadata_path, identity.metadata.size).await?;
        verify_digest(METADATA_PATH, &metadata_bytes, &identity.metadata.sha256)?;
        let metadata: BrowserParityMetadata =
            serde_json::from_slice(&metadata_bytes).map_err(|error| {
                parity_error(format!("invalid authenticated metadata.json: {error}"))
            })?;
        validate_metadata(&metadata, scope, expected_tensor_count)?;

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
            .ok_or_else(|| parity_error("SafeTensors data offset overflow"))?;
        let tensors = build_tensor_index(
            &header,
            data_start,
            identity.tensors.size,
            &metadata.tensors,
            expected_tensor_count,
        )?;
        let expected_tensor_bytes = tensors.values().try_fold(0_u64, |total, tensor| {
            total
                .checked_add(tensor.byte_length)
                .ok_or_else(|| parity_error("fixture tensor byte-count overflow"))
        })?;

        Ok(Self {
            base_url,
            identity: identity.clone(),
            metadata: Arc::new(metadata),
            tensors: Arc::new(tensors),
            verification: Arc::new(Mutex::new(BrowserParityVerificationState {
                snapshot: BrowserParityVerificationSnapshot {
                    verified_metadata_files: 1,
                    verified_metadata_bytes: identity.metadata.size,
                    verified_metadata_sha256: Some(identity.metadata.sha256),
                    verified_safetensors_headers: 1,
                    verified_safetensors_header_bytes: data_start,
                    expected_tensors: expected_tensor_count as u64,
                    expected_tensor_bytes,
                    ..BrowserParityVerificationSnapshot::default()
                },
                verified_tensor_names: BTreeSet::new(),
            })),
        })
    }

    pub(crate) fn identity(&self) -> BrowserParityFixtureIdentity {
        self.identity.clone()
    }

    pub(crate) fn metadata(&self) -> &BrowserParityMetadata {
        self.metadata.as_ref()
    }

    pub(crate) fn tensor_specs(&self) -> &BTreeMap<String, BrowserParityTensorSpec> {
        self.tensors.as_ref()
    }

    pub(crate) fn snapshot(&self) -> Result<BrowserParityVerificationSnapshot, RuntimeError> {
        self.verification
            .lock()
            .map(|state| state.snapshot.clone())
            .map_err(|_| parity_error("browser fixture verification state is poisoned"))
    }

    /// Stream and authenticate the complete identity-pinned SafeTensors container.
    ///
    /// The pass keeps one bounded transport chunk plus two SHA-256 states in memory. In addition
    /// to the pinned whole-file digest, it authenticates all tensor bodies independently and
    /// advances the shared unique-tensor counters. A partial/failed pass never marks the whole
    /// container verified.
    #[cfg(target_arch = "wasm32")]
    pub(crate) async fn authenticate_all_tensors(
        &self,
    ) -> Result<BrowserParityVerificationSnapshot, RuntimeError> {
        let path = artifact_path(TENSORS_PATH)?;
        let data_start = self
            .tensors
            .values()
            .map(|tensor| tensor.file_offset)
            .min()
            .ok_or_else(|| parity_error("fixture tensor index is empty"))?;
        let mut whole_hasher = Sha256::new();
        for range in exact_range_chunks(
            0,
            data_start,
            self.identity.tensors.size,
            DEFAULT_BROWSER_CHUNK_BYTES,
        )? {
            let bytes =
                fetch_one_exact_range(&self.base_url, &path, range, self.identity.tensors.size)
                    .await?;
            whole_hasher.update(&bytes);
        }

        let mut ordered = self.tensors.values().cloned().collect::<Vec<_>>();
        ordered.sort_by_key(|tensor| tensor.file_offset);
        let mut cursor = data_start;
        for tensor in ordered {
            if tensor.file_offset != cursor {
                return Err(parity_error(format!(
                    "tensor {:?} begins at {}; expected contiguous file offset {cursor}",
                    tensor.name, tensor.file_offset
                )));
            }
            let mut tensor_hasher = Sha256::new();
            for range in exact_range_chunks(
                tensor.file_offset,
                tensor.byte_length,
                self.identity.tensors.size,
                DEFAULT_BROWSER_CHUNK_BYTES,
            )? {
                let bytes =
                    fetch_one_exact_range(&self.base_url, &path, range, self.identity.tensors.size)
                        .await?;
                whole_hasher.update(&bytes);
                tensor_hasher.update(&bytes);
            }
            let actual = Sha256Digest::from_bytes(tensor_hasher.finalize().into());
            if actual != tensor.sha256 {
                return Err(parity_error(format!(
                    "fixture tensor {:?} SHA-256 {actual} differs from authenticated {}",
                    tensor.name, tensor.sha256
                )));
            }
            self.record_verified_tensor(&tensor)?;
            cursor = cursor
                .checked_add(tensor.byte_length)
                .ok_or_else(|| parity_error("authenticated tensor cursor overflow"))?;
        }
        if cursor != self.identity.tensors.size {
            return Err(parity_error(format!(
                "authenticated tensor stream ended at {cursor}; expected {}",
                self.identity.tensors.size
            )));
        }
        let actual = Sha256Digest::from_bytes(whole_hasher.finalize().into());
        let expected = Sha256Digest::from_hex(&self.identity.tensors.sha256)
            .expect("identity SafeTensors SHA-256 is valid");
        if actual != expected {
            return Err(parity_error(format!(
                "fixture {TENSORS_PATH} SHA-256 {actual} differs from pinned {expected}"
            )));
        }

        let mut state = self
            .verification
            .lock()
            .map_err(|_| parity_error("browser fixture verification state is poisoned"))?;
        state.snapshot.verified_safetensors_files = 1;
        state.snapshot.verified_safetensors_file_bytes = self.identity.tensors.size;
        state.snapshot.verified_safetensors_sha256 = Some(actual.to_hex());
        Ok(state.snapshot.clone())
    }

    /// Fetch an authenticated floating-point tensor and widen BF16 values to finite F32 hosts.
    #[cfg(target_arch = "wasm32")]
    pub(crate) async fn f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), RuntimeError> {
        let tensor = self.fetch_tensor(name).await?;
        let values = decode_float(name, tensor.spec.dtype, &tensor.bytes)?;
        Ok((tensor.spec.shape, values))
    }

    /// Fetch an authenticated I64 tensor as a host vector.
    #[cfg(target_arch = "wasm32")]
    pub(crate) async fn i64(&self, name: &str) -> Result<(Vec<usize>, Vec<i64>), RuntimeError> {
        let tensor = self.fetch_tensor(name).await?;
        if tensor.spec.dtype != BrowserParityDType::I64 {
            return Err(dtype_error(name, tensor.spec.dtype, "I64"));
        }
        Ok((tensor.spec.shape, decode_i64(&tensor.bytes)))
    }

    /// Fetch an authenticated U8 tensor as a host vector.
    #[cfg(target_arch = "wasm32")]
    pub(crate) async fn u8(&self, name: &str) -> Result<(Vec<usize>, Vec<u8>), RuntimeError> {
        let tensor = self.fetch_tensor(name).await?;
        if tensor.spec.dtype != BrowserParityDType::U8 {
            return Err(dtype_error(name, tensor.spec.dtype, "U8"));
        }
        Ok((tensor.spec.shape, tensor.bytes))
    }

    /// Fetch and authenticate the canonical source image used for Edit conditioning.
    #[cfg(target_arch = "wasm32")]
    pub(crate) async fn source_png(&self) -> Result<Vec<u8>, RuntimeError> {
        let bytes = fetch_exact_file(
            &self.base_url,
            &artifact_path(SOURCE_PATH)?,
            self.identity.source.size,
        )
        .await?;
        verify_digest(SOURCE_PATH, &bytes, &self.identity.source.sha256)?;
        let mut state = self
            .verification
            .lock()
            .map_err(|_| parity_error("browser fixture verification state is poisoned"))?;
        if state.snapshot.verified_source_files == 0 {
            state.snapshot.verified_source_files = 1;
            state.snapshot.verified_source_bytes = self.identity.source.size;
            state.snapshot.verified_source_sha256 = Some(self.identity.source.sha256.clone());
        }
        Ok(bytes)
    }

    /// Fetch and authenticate the encoded upstream image artifact.
    #[cfg(target_arch = "wasm32")]
    pub(crate) async fn output_png(&self) -> Result<Vec<u8>, RuntimeError> {
        let bytes = fetch_exact_file(
            &self.base_url,
            &artifact_path(OUTPUT_PATH)?,
            self.identity.output.size,
        )
        .await?;
        verify_digest(OUTPUT_PATH, &bytes, &self.identity.output.sha256)?;
        let mut state = self
            .verification
            .lock()
            .map_err(|_| parity_error("browser fixture verification state is poisoned"))?;
        if state.snapshot.verified_output_files == 0 {
            state.snapshot.verified_output_files = 1;
            state.snapshot.verified_output_bytes = self.identity.output.size;
            state.snapshot.verified_output_sha256 = Some(self.identity.output.sha256.clone());
        }
        Ok(bytes)
    }

    #[cfg(target_arch = "wasm32")]
    async fn fetch_tensor(&self, name: &str) -> Result<FetchedTensor, RuntimeError> {
        let spec = self
            .tensors
            .get(name)
            .cloned()
            .ok_or_else(|| parity_error(format!("fixture omits tensor {name:?}")))?;
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
            return Err(parity_error(format!(
                "fixture tensor {name:?} SHA-256 {actual} differs from authenticated {}",
                spec.sha256
            )));
        }

        self.record_verified_tensor(&spec)?;

        Ok(FetchedTensor { spec, bytes })
    }

    fn record_verified_tensor(&self, spec: &BrowserParityTensorSpec) -> Result<(), RuntimeError> {
        let mut state = self
            .verification
            .lock()
            .map_err(|_| parity_error("browser fixture verification state is poisoned"))?;
        if state.verified_tensor_names.insert(spec.name.clone()) {
            state.snapshot.verified_tensors = state
                .snapshot
                .verified_tensors
                .checked_add(1)
                .ok_or_else(|| parity_error("verified tensor counter overflow"))?;
            state.snapshot.verified_tensor_bytes = state
                .snapshot
                .verified_tensor_bytes
                .checked_add(spec.byte_length)
                .ok_or_else(|| parity_error("verified tensor byte counter overflow"))?;
        }
        Ok(())
    }
}

#[cfg(target_arch = "wasm32")]
struct FetchedTensor {
    spec: BrowserParityTensorSpec,
    bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct FloatMetrics {
    pub(crate) element_count: u64,
    pub(crate) max_abs: f32,
    pub(crate) mean_abs: f32,
    pub(crate) rmse: f32,
    pub(crate) relative_rmse: f32,
    pub(crate) cosine_similarity: f32,
}

/// Compare two finite floating-point buffers using the release parity definitions.
pub(crate) fn compare_float(
    actual: &[f32],
    expected: &[f32],
) -> Result<FloatMetrics, RuntimeError> {
    if actual.is_empty() || actual.len() != expected.len() {
        return Err(parity_error(format!(
            "float comparison length mismatch: actual={} expected={}",
            actual.len(),
            expected.len()
        )));
    }
    let mut max_abs = 0.0_f64;
    let mut sum_abs = 0.0_f64;
    let mut sum_squared = 0.0_f64;
    let mut dot = 0.0_f64;
    let mut actual_squared = 0.0_f64;
    let mut expected_squared = 0.0_f64;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if !actual.is_finite() || !expected.is_finite() {
            return Err(parity_error(format!(
                "float comparison contains a non-finite value at element {index}"
            )));
        }
        let actual = f64::from(actual);
        let expected = f64::from(expected);
        let difference = actual - expected;
        max_abs = max_abs.max(difference.abs());
        sum_abs += difference.abs();
        sum_squared += difference * difference;
        dot += actual * expected;
        actual_squared += actual * actual;
        expected_squared += expected * expected;
    }
    let count = actual.len() as f64;
    let rmse = (sum_squared / count).sqrt();
    let expected_rms = (expected_squared / count).sqrt();
    let relative_rmse = if expected_rms == 0.0 {
        if rmse == 0.0 {
            0.0
        } else {
            f64::from(f32::MAX)
        }
    } else {
        rmse / expected_rms
    };
    let denominator = (actual_squared * expected_squared).sqrt();
    let cosine = if actual_squared == 0.0 && expected_squared == 0.0 {
        1.0
    } else if denominator == 0.0 {
        0.0
    } else {
        (dot / denominator).clamp(-1.0, 1.0)
    };
    Ok(FloatMetrics {
        element_count: actual.len() as u64,
        max_abs: metric_f32("max_abs", max_abs)?,
        mean_abs: metric_f32("mean_abs", sum_abs / count)?,
        rmse: metric_f32("rmse", rmse)?,
        relative_rmse: metric_f32("relative_rmse", relative_rmse)?,
        cosine_similarity: metric_f32("cosine_similarity", cosine)?,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct RgbMetrics {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) max_abs_u8: u8,
    pub(crate) mean_abs_u8: f32,
    pub(crate) rmse_u8: f32,
    pub(crate) psnr_db: f32,
    pub(crate) mean_block_ssim_8x8: f32,
    pub(crate) exact_fraction: f32,
}

/// Compare packed RGB8 buffers, including the release's channel-wise 8x8 block SSIM.
pub(crate) fn compare_rgb(
    actual: &[u8],
    expected: &[u8],
    width: usize,
    height: usize,
) -> Result<RgbMetrics, RuntimeError> {
    let expected_length = rgb_length(width, height)?;
    if actual.len() != expected_length || expected.len() != expected_length {
        return Err(parity_error(format!(
            "RGB comparison length mismatch: actual={} expected={} dimensions={width}x{height}",
            actual.len(),
            expected.len()
        )));
    }
    let mut max_abs = 0_u8;
    let mut exact = 0_u64;
    let mut sum_abs = 0.0_f64;
    let mut sum_squared = 0.0_f64;
    for (&actual, &expected) in actual.iter().zip(expected) {
        let difference = actual.abs_diff(expected);
        max_abs = max_abs.max(difference);
        exact += u64::from(difference == 0);
        sum_abs += f64::from(difference);
        sum_squared += f64::from(difference).powi(2);
    }
    let count = expected_length as f64;
    let rmse = (sum_squared / count).sqrt();
    Ok(RgbMetrics {
        width,
        height,
        max_abs_u8: max_abs,
        mean_abs_u8: metric_f32("RGB mean_abs", sum_abs / count)?,
        rmse_u8: metric_f32("RGB rmse", rmse)?,
        psnr_db: if rmse == 0.0 {
            100.0
        } else {
            metric_f32("RGB PSNR", 20.0 * (255.0 / rmse).log10())?
        },
        mean_block_ssim_8x8: mean_block_ssim_8x8(actual, expected, width, height)?,
        exact_fraction: metric_f32("RGB exact fraction", exact as f64 / count)?,
    })
}

fn mean_block_ssim_8x8(
    actual: &[u8],
    expected: &[u8],
    width: usize,
    height: usize,
) -> Result<f32, RuntimeError> {
    let expected_length = rgb_length(width, height)?;
    if actual.len() != expected_length || expected.len() != expected_length {
        return Err(parity_error(
            "SSIM input length differs from the declared RGB dimensions",
        ));
    }
    let c1 = (0.01_f64 * 255.0).powi(2);
    let c2 = (0.03_f64 * 255.0).powi(2);
    let mut total = 0.0_f64;
    let mut blocks = 0_u64;
    for top in (0..height).step_by(8) {
        for left in (0..width).step_by(8) {
            let bottom = (top + 8).min(height);
            let right = (left + 8).min(width);
            for channel in 0..3 {
                let samples = (bottom - top) * (right - left);
                let count = samples as f64;
                let mut actual_mean = 0.0_f64;
                let mut expected_mean = 0.0_f64;
                for y in top..bottom {
                    for x in left..right {
                        let index = (y * width + x) * 3 + channel;
                        actual_mean += f64::from(actual[index]);
                        expected_mean += f64::from(expected[index]);
                    }
                }
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
    metric_f32("mean block SSIM", total / blocks as f64)
}

fn rgb_length(width: usize, height: usize) -> Result<usize, RuntimeError> {
    width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(3))
        .filter(|length| *length > 0)
        .ok_or_else(|| parity_error("RGB dimensions are empty or overflow"))
}

fn metric_f32(name: &str, value: f64) -> Result<f32, RuntimeError> {
    if !value.is_finite() || value.abs() > f64::from(f32::MAX) {
        return Err(parity_error(format!(
            "computed {name} is not a finite F32 value"
        )));
    }
    Ok(value as f32)
}

fn validate_metadata(
    metadata: &BrowserParityMetadata,
    scope: BrowserParityFixtureScope,
    expected_tensor_count: usize,
) -> Result<(), RuntimeError> {
    if metadata.schema_version != EDIT_1K5_FIXTURE_SCHEMA_VERSION
        || metadata.variant != EDIT_1K5_FIXTURE_VARIANT
        || metadata.resolution_profile != EDIT_1K5_FIXTURE_RESOLUTION_PROFILE
        || metadata.model_revision != EDIT_1K5_FIXTURE_MODEL_REVISION
        || metadata.upstream_source_revision != EDIT_1K5_FIXTURE_UPSTREAM_SOURCE_REVISION
        || metadata.width != EDIT_1K5_FIXTURE_EDGE
        || metadata.height != EDIT_1K5_FIXTURE_EDGE
        || metadata.seed != EDIT_1K5_FIXTURE_SEED
        || metadata.dtype != "bf16"
    {
        return Err(parity_error(
            "fixture metadata is not the fixed schema-2 Edit-Turbo 1.5K release identity",
        ));
    }
    let expected_exhaustive_capture = scope == BrowserParityFixtureScope::Exhaustive;
    if metadata.prompt.trim().is_empty()
        || metadata.capture_blocks != expected_exhaustive_capture
        || metadata.capture_qwen != expected_exhaustive_capture
    {
        return Err(parity_error(
            "fixture metadata capture policy differs from its pinned diagnostic scope",
        ));
    }
    let dimensions = BrowserParityDimensions {
        width: EDIT_1K5_FIXTURE_EDGE,
        height: EDIT_1K5_FIXTURE_EDGE,
    };
    if metadata.output.align_res
        || !metadata.output.validated
        || metadata.output.requested != dimensions
        || metadata.output.actual.width != EDIT_1K5_FIXTURE_EDGE
        || metadata.output.actual.height != EDIT_1K5_FIXTURE_EDGE
        || metadata.output.actual.mode != "RGB"
        || metadata.output.actual.image_count != 1
    {
        return Err(parity_error(
            "fixture output metadata is not one validated 1536x1536 RGB image with align_res=false",
        ));
    }
    let source = &metadata.provenance.source_image;
    if source.fixture_path != SOURCE_PATH
        || source.size != EDIT_1K5_FIXTURE_SOURCE_SIZE
        || source.sha256 != EDIT_1K5_FIXTURE_SOURCE_SHA256
        || !source.copy_verified
    {
        return Err(parity_error(
            "fixture source-image provenance differs from the authenticated release source",
        ));
    }
    if metadata.tensors.len() != expected_tensor_count {
        return Err(parity_error(format!(
            "fixture metadata contains {} tensors; pinned diagnostic scope requires {}",
            metadata.tensors.len(),
            expected_tensor_count
        )));
    }
    validate_required_tensor(
        metadata,
        "dmd.initial_latents",
        BrowserParityDType::Bf16,
        &[1, 16, 192, 192],
    )?;
    validate_required_tensor(
        metadata,
        "dmd.final_latents",
        BrowserParityDType::Bf16,
        &[1, 16, 192, 192],
    )?;
    validate_required_tensor(
        metadata,
        "output.rgb_u8",
        BrowserParityDType::U8,
        &[EDIT_1K5_FIXTURE_EDGE, EDIT_1K5_FIXTURE_EDGE, 3],
    )?;
    validate_required_tensor(
        metadata,
        "vae.reference_input",
        BrowserParityDType::F32,
        &[1, 3, 256, 256],
    )?;
    validate_required_tensor(
        metadata,
        "vae.reference_epsilon",
        BrowserParityDType::Bf16,
        &[1, 16, 32, 32],
    )?;
    for component in ["mean", "logvar", "std", "raw_latent", "scaled_latent"] {
        validate_required_tensor(
            metadata,
            &format!("vae.reference_f32_{component}"),
            BrowserParityDType::F32,
            &[1, 16, 32, 32],
        )?;
        validate_required_tensor(
            metadata,
            &format!("vae.reference_{component}"),
            BrowserParityDType::Bf16,
            &[1, 16, 32, 32],
        )?;
    }
    validate_required_tensor(
        metadata,
        "vae.reference_f32_moments",
        BrowserParityDType::F32,
        &[1, 32, 32, 32],
    )?;
    validate_required_tensor(
        metadata,
        "vae.reference_moments",
        BrowserParityDType::Bf16,
        &[1, 32, 32, 32],
    )?;
    Ok(())
}

fn validate_required_tensor(
    metadata: &BrowserParityMetadata,
    name: &str,
    dtype: BrowserParityDType,
    shape: &[usize],
) -> Result<(), RuntimeError> {
    let tensor = metadata
        .tensors
        .get(name)
        .ok_or_else(|| parity_error(format!("fixture metadata omits required tensor {name:?}")))?;
    if tensor.dtype != dtype.metadata_name() || tensor.shape != shape {
        return Err(parity_error(format!(
            "fixture tensor {name:?} has metadata dtype/shape {:?} {:?}; expected {} {shape:?}",
            tensor.dtype,
            tensor.shape,
            dtype.metadata_name()
        )));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
enum SafeTensorsDType {
    #[serde(rename = "BF16")]
    Bf16,
    #[serde(rename = "F32")]
    F32,
    #[serde(rename = "I64")]
    I64,
    #[serde(rename = "U8")]
    U8,
}

impl From<SafeTensorsDType> for BrowserParityDType {
    fn from(value: SafeTensorsDType) -> Self {
        match value {
            SafeTensorsDType::Bf16 => Self::Bf16,
            SafeTensorsDType::F32 => Self::F32,
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
        .map_err(|_| parity_error("SafeTensors length prefix must contain exactly eight bytes"))?;
    let header_length = u64::from_le_bytes(prefix);
    let data_start = 8_u64
        .checked_add(header_length)
        .ok_or_else(|| parity_error("SafeTensors header length overflow"))?;
    if header_length == 0 || header_length > MAX_SAFETENSORS_HEADER_BYTES || data_start >= file_size
    {
        return Err(parity_error(format!(
            "invalid SafeTensors header length {header_length} for {file_size}-byte fixture"
        )));
    }
    Ok(header_length)
}

fn build_tensor_index(
    header_bytes: &[u8],
    data_start: u64,
    file_size: u64,
    authenticated: &BTreeMap<String, BrowserParityTensorDigest>,
    expected_tensor_count: usize,
) -> Result<BTreeMap<String, BrowserParityTensorSpec>, RuntimeError> {
    let header: SafeTensorsHeader = serde_json::from_slice(header_bytes)
        .map_err(|error| parity_error(format!("invalid SafeTensors header JSON: {error}")))?;
    if header.tensors.len() != expected_tensor_count || authenticated.len() != expected_tensor_count
    {
        return Err(parity_error(format!(
            "SafeTensors/authenticated tensor counts are {}/{}; expected {expected_tensor_count}",
            header.tensors.len(),
            authenticated.len()
        )));
    }
    let header_names = header.tensors.keys().collect::<BTreeSet<_>>();
    let authenticated_names = authenticated.keys().collect::<BTreeSet<_>>();
    if header_names != authenticated_names {
        return Err(parity_error(
            "SafeTensors tensor keyset differs from authenticated metadata",
        ));
    }
    let data_length = file_size
        .checked_sub(data_start)
        .ok_or_else(|| parity_error("SafeTensors data start exceeds fixed file size"))?;
    let mut ordered = header.tensors.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|(_, entry)| entry.data_offsets[0]);
    let mut cursor = 0_u64;
    let mut tensors = BTreeMap::new();
    for (name, entry) in ordered {
        let [start, end] = entry.data_offsets;
        if start != cursor || end <= start || end > data_length {
            return Err(parity_error(format!(
                "SafeTensors tensor {name:?} has non-contiguous/out-of-bounds offsets [{start},{end})"
            )));
        }
        if entry.shape.contains(&0) {
            return Err(parity_error(format!(
                "SafeTensors tensor {name:?} contains a zero shape dimension"
            )));
        }
        let dtype = BrowserParityDType::from(entry.dtype);
        let elements = entry.shape.iter().try_fold(1_u64, |count, &dimension| {
            count
                .checked_mul(dimension as u64)
                .ok_or_else(|| parity_error(format!("tensor {name:?} shape overflows")))
        })?;
        let expected_length = elements
            .checked_mul(dtype.element_bytes())
            .ok_or_else(|| parity_error(format!("tensor {name:?} byte length overflows")))?;
        let byte_length = end - start;
        if byte_length != expected_length {
            return Err(parity_error(format!(
                "SafeTensors tensor {name:?} stores {byte_length} bytes; dtype/shape require {expected_length}"
            )));
        }
        let expected = authenticated
            .get(name)
            .expect("equal keysets guarantee authenticated tensor metadata");
        if expected.dtype != dtype.metadata_name() || expected.shape != entry.shape {
            return Err(parity_error(format!(
                "SafeTensors tensor {name:?} dtype/shape differs from authenticated metadata"
            )));
        }
        let sha256 = Sha256Digest::from_hex(&expected.sha256).map_err(|error| {
            parity_error(format!(
                "authenticated tensor {name:?} has invalid SHA-256: {error}"
            ))
        })?;
        let file_offset = data_start
            .checked_add(start)
            .ok_or_else(|| parity_error(format!("tensor {name:?} file offset overflows")))?;
        tensors.insert(
            name.clone(),
            BrowserParityTensorSpec {
                name: name.clone(),
                dtype,
                shape: entry.shape.clone(),
                file_offset,
                byte_length,
                sha256,
            },
        );
        cursor = end;
    }
    if cursor != data_length {
        return Err(parity_error(format!(
            "SafeTensors header accounts for {cursor} data bytes; fixed file contains {data_length}"
        )));
    }
    Ok(tensors)
}

fn decode_float(
    name: &str,
    dtype: BrowserParityDType,
    bytes: &[u8],
) -> Result<Vec<f32>, RuntimeError> {
    match dtype {
        BrowserParityDType::Bf16 => Ok(decode_bf16(name, bytes)?
            .into_iter()
            .map(bf16::to_f32)
            .collect()),
        BrowserParityDType::F32 => decode_f32(name, bytes),
        other => Err(dtype_error(name, other, "BF16 or F32")),
    }
}

fn decode_bf16(name: &str, bytes: &[u8]) -> Result<Vec<bf16>, RuntimeError> {
    if !bytes.len().is_multiple_of(2) {
        return Err(parity_error(format!(
            "BF16 tensor {name:?} has a partial element"
        )));
    }
    bytes
        .chunks_exact(2)
        .enumerate()
        .map(|(index, word)| {
            let value = bf16::from_bits(u16::from_le_bytes([word[0], word[1]]));
            if !value.is_finite() {
                return Err(parity_error(format!(
                    "BF16 tensor {name:?} contains a non-finite value at element {index}"
                )));
            }
            Ok(value)
        })
        .collect()
}

fn decode_f32(name: &str, bytes: &[u8]) -> Result<Vec<f32>, RuntimeError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(parity_error(format!(
            "F32 tensor {name:?} has a partial element"
        )));
    }
    bytes
        .chunks_exact(4)
        .enumerate()
        .map(|(index, word)| {
            let value = f32::from_le_bytes([word[0], word[1], word[2], word[3]]);
            if !value.is_finite() {
                return Err(parity_error(format!(
                    "F32 tensor {name:?} contains a non-finite value at element {index}"
                )));
            }
            Ok(value)
        })
        .collect()
}

fn decode_i64(bytes: &[u8]) -> Vec<i64> {
    bytes
        .chunks_exact(8)
        .map(|word| {
            i64::from_le_bytes([
                word[0], word[1], word[2], word[3], word[4], word[5], word[6], word[7],
            ])
        })
        .collect()
}

fn dtype_error(name: &str, actual: BrowserParityDType, expected: &str) -> RuntimeError {
    parity_error(format!(
        "fixture tensor {name:?} has dtype {actual:?}; expected {expected}"
    ))
}

fn verify_digest(name: &str, bytes: &[u8], expected: &str) -> Result<(), RuntimeError> {
    let expected = Sha256Digest::from_hex(expected)
        .map_err(|error| parity_error(format!("invalid pinned {name} SHA-256: {error}")))?;
    let actual = Sha256Digest::calculate(bytes);
    if actual != expected {
        return Err(parity_error(format!(
            "fixture {name} SHA-256 {actual} differs from pinned {expected}"
        )));
    }
    Ok(())
}

fn artifact_path(value: &str) -> Result<ArtifactPath, RuntimeError> {
    ArtifactPath::new(value)
        .map_err(|error| parity_error(format!("invalid fixed fixture path {value:?}: {error}")))
}

fn parity_error(message: impl Into<String>) -> RuntimeError {
    RuntimeError::ModelExecution {
        model: ModelId::new(EDIT_1K5_FIXTURE_MODEL_ID)
            .expect("fixed Edit-Turbo 1.5K model id is valid"),
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
        return Err(parity_error(
            "exact browser ranges and their chunk bound must be non-zero",
        ));
    }
    let end = offset
        .checked_add(length)
        .ok_or_else(|| parity_error("exact browser range overflows"))?;
    if end > total {
        return Err(parity_error(format!(
            "exact browser range [{offset},{end}) exceeds {total}-byte file"
        )));
    }
    let mut ranges = Vec::new();
    let mut cursor = offset;
    while cursor < end {
        let chunk_length = (end - cursor).min(maximum_chunk_bytes);
        ranges.push(
            ByteRange::new(cursor, chunk_length)
                .map_err(|error| parity_error(format!("invalid exact browser range: {error}")))?,
        );
        cursor = cursor
            .checked_add(chunk_length)
            .ok_or_else(|| parity_error("exact browser chunk cursor overflow"))?;
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
        parity_error(format!(
            "requested fixture span of {length} bytes cannot fit in Wasm memory"
        ))
    })?;
    let mut output = Vec::with_capacity(capacity);
    for range in exact_range_chunks(offset, length, total, DEFAULT_BROWSER_CHUNK_BYTES)? {
        let bytes = fetch_one_exact_range(base_url, path, range, total).await?;
        output.extend_from_slice(&bytes);
    }
    if output.len() != capacity {
        return Err(parity_error(format!(
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

    let range_header = range.http_range_header();
    let headers = Headers::new().map_err(browser_js_error)?;
    headers
        .set("Range", &range_header)
        .map_err(browser_js_error)?;
    let init = RequestInit::new();
    init.set_method("GET");
    init.set_headers_headers(&headers);
    let url = base_url.resolve(path);
    let request = Request::new_with_str_and_init(&url, &init).map_err(browser_js_error)?;
    let window =
        web_sys::window().ok_or_else(|| parity_error("browser fixture fetch requires Window"))?;
    let response = JsFuture::from(window.fetch_with_request(&request))
        .await
        .map_err(browser_js_error)?
        .dyn_into::<Response>()
        .map_err(browser_js_error)?;
    if response.status() != 206 {
        return Err(parity_error(format!(
            "fixture range request returned HTTP {} for {url}; expected 206",
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
        return Err(parity_error(format!(
            "fixture range response has Content-Range {content_range:?}; expected {expected_content_range:?}"
        )));
    }
    let buffer = JsFuture::from(response.array_buffer().map_err(browser_js_error)?)
        .await
        .map_err(browser_js_error)?;
    let bytes = Uint8Array::new(&buffer).to_vec();
    if bytes.len() as u64 != range.length() {
        return Err(parity_error(format!(
            "fixture range returned {} bytes; expected {}",
            bytes.len(),
            range.length()
        )));
    }
    Ok(bytes)
}

#[cfg(target_arch = "wasm32")]
fn browser_js_error(value: wasm_bindgen::JsValue) -> RuntimeError {
    parity_error(
        value
            .as_string()
            .unwrap_or_else(|| format!("browser JavaScript error: {value:?}")),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn browser_webgpu_vae_f32_oracle_envelope_sources_stay_synchronized_correctness() {
        let rust = include_str!("browser_boogu.rs");
        let javascript_source = include_str!("../tests/wasm_browser_1k5_parity.mjs");
        assert!(javascript_source.contains("const F32_EPSILON = 2 ** -23;"));
        assert!(
            javascript_source
                .contains("Math.fround(processing.pixel_values?.max_abs) > F32_EPSILON")
        );
        let rust = rust
            .split_once("const BROWSER_WEBGPU_VAE_F32_ORACLE_ENVELOPE")
            .unwrap()
            .1
            .split_once("};")
            .unwrap()
            .0;
        let javascript = javascript_source
            .split_once("const BROWSER_WEBGPU_VAE_F32_ORACLE_ENVELOPE")
            .unwrap()
            .1
            .split_once("});")
            .unwrap()
            .0;

        for needle in [
            "BrowserWebGpu/raw-cubecl-no-fusion",
            "weight_storage_dtype: \"f16\"",
            "weight_load_policy: \"adapt-to-f32\"",
            "execution_dtype: \"f32\"",
            "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
            "calibrated_device: \"0x2bb1\"",
            "calibrated_driver: \"610.43.02\"",
            "no-cross-adapter-portability-claim",
            "maximum_abs: 0.016",
            "maximum_rmse: 0.000_75",
            "maximum_abs: 0.013",
            "maximum_abs: 0.000_1",
            "maximum_abs: 0.005",
            "maximum_rmse: 0.000_2",
            "minimum_cosine_similarity: 0.999_999",
        ] {
            assert!(rust.contains(needle), "Rust envelope omits {needle}");
            assert!(
                javascript.contains(needle),
                "JavaScript envelope omits {needle}"
            );
        }
        assert_eq!(rust.matches("maximum_abs: 0.016").count(), 2);
        assert_eq!(javascript.matches("maximum_abs: 0.016").count(), 2);
        assert_eq!(rust.matches("maximum_abs: 0.013").count(), 2);
        assert_eq!(javascript.matches("maximum_abs: 0.013").count(), 2);
        assert!(rust.contains("artifact_profile: \"f16-qwen-vision-f32\""));
        assert!(javascript.contains("artifact_profile: CANONICAL_PROFILE"));
        assert!(rust.contains("BROWSER_WEBGPU_VAE_F32_ORACLE_LEGACY_FLAT_CONTENT_DIGEST"));
        assert!(
            javascript_source
                .contains("5d7e25b1d9be1fdf4a6372bfb9db28cf62ef90253082cef22af09653047e3a7b")
        );
    }

    fn digest(dtype: &str, shape: &[usize]) -> BrowserParityTensorDigest {
        BrowserParityTensorDigest {
            dtype: dtype.into(),
            shape: shape.into(),
            sha256: Sha256Digest::calculate(b"test").to_hex(),
        }
    }

    #[test]
    fn range_math_stays_exact_above_wasm_address_space_correctness() {
        let offset = u64::from(u32::MAX) + 1024;
        let ranges = exact_range_chunks(offset, 10, offset + 10, 4).unwrap();
        assert_eq!(ranges.len(), 3);
        assert_eq!(ranges[0].http_range_header(), "bytes=4294968319-4294968322");
        assert_eq!(ranges[1].http_range_header(), "bytes=4294968323-4294968326");
        assert_eq!(ranges[2].http_range_header(), "bytes=4294968327-4294968328");
        assert!(exact_range_chunks(u64::MAX, 1, u64::MAX, 4).is_err());
        assert!(exact_range_chunks(9, 2, 10, 4).is_err());
    }

    #[test]
    fn safetensors_header_uses_u64_offsets_and_authenticates_contract_correctness() {
        let data_start = u64::from(u32::MAX) + 100;
        let file_size = data_start + 20;
        let header = br#"{
            "floats":{"dtype":"F32","shape":[3],"data_offsets":[0,12]},
            "ids":{"dtype":"I64","shape":[1],"data_offsets":[12,20]}
        }"#;
        let authenticated = BTreeMap::from([
            ("floats".into(), digest("torch.float32", &[3])),
            ("ids".into(), digest("torch.int64", &[1])),
        ]);
        let index = build_tensor_index(header, data_start, file_size, &authenticated, 2).unwrap();
        assert_eq!(index["floats"].file_offset, data_start);
        assert_eq!(index["ids"].file_offset, data_start + 12);
        assert_eq!(index["ids"].byte_length, 8);

        let gap = br#"{
            "floats":{"dtype":"F32","shape":[3],"data_offsets":[0,12]},
            "ids":{"dtype":"I64","shape":[1],"data_offsets":[13,21]}
        }"#;
        assert!(build_tensor_index(gap, data_start, file_size + 1, &authenticated, 2).is_err());
    }

    #[test]
    fn safetensors_header_rejects_duplicates_and_unsupported_dtype_correctness() {
        let duplicate = br#"{
            "value":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},
            "value":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}
        }"#;
        let authenticated = BTreeMap::from([("value".into(), digest("torch.uint8", &[1]))]);
        assert!(build_tensor_index(duplicate, 8, 9, &authenticated, 1).is_err());

        let unsupported = br#"{"value":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}"#;
        assert!(build_tensor_index(unsupported, 8, 10, &authenticated, 1).is_err());
    }

    #[test]
    fn header_length_is_bounded_and_file_relative_correctness() {
        assert_eq!(parse_header_length(&64_u64.to_le_bytes(), 100).unwrap(), 64);
        assert!(parse_header_length(&0_u64.to_le_bytes(), 100).is_err());
        assert!(parse_header_length(&100_u64.to_le_bytes(), 100).is_err());
        assert!(parse_header_length(&[0; 7], 100).is_err());
    }

    #[test]
    fn float_metrics_are_exact_and_fail_closed_on_nonfinite_values_correctness() {
        let metric = compare_float(&[1.0, -2.0, 3.0], &[1.0, -2.0, 3.0]).unwrap();
        assert_eq!(metric.max_abs, 0.0);
        assert_eq!(metric.rmse, 0.0);
        assert_eq!(metric.relative_rmse, 0.0);
        assert!((metric.cosine_similarity - 1.0).abs() <= f32::EPSILON);
        assert_eq!(
            compare_float(&[1.0], &[0.0]).unwrap().cosine_similarity,
            0.0
        );
        assert!(compare_float(&[f32::NAN], &[0.0]).is_err());
        assert!(compare_float(&[0.0], &[f32::INFINITY]).is_err());
        assert!(compare_float(&[], &[]).is_err());
    }

    #[test]
    fn rgb_metrics_include_psnr_and_channelwise_block_ssim_correctness() {
        let identity = (0_u8..45).collect::<Vec<_>>();
        let metric = compare_rgb(&identity, &identity, 5, 3).unwrap();
        assert_eq!(metric.max_abs_u8, 0);
        assert_eq!(metric.psnr_db, 100.0);
        assert_eq!(metric.exact_fraction, 1.0);
        assert!((metric.mean_block_ssim_8x8 - 1.0).abs() <= f32::EPSILON);

        let expected = vec![128_u8; 8 * 8 * 3];
        let mut actual = expected.clone();
        for pixel in actual.chunks_exact_mut(3) {
            pixel[1] = 0;
        }
        let metric = compare_rgb(&actual, &expected, 8, 8).unwrap();
        assert_eq!(metric.max_abs_u8, 128);
        assert!(metric.psnr_db.is_finite());
        assert!((0.6..0.7).contains(&metric.mean_block_ssim_8x8));
        assert!(compare_rgb(&actual[..actual.len() - 1], &expected, 8, 8).is_err());
    }

    #[test]
    fn decoders_reject_nonfinite_float_payloads_correctness() {
        assert_eq!(
            decode_float(
                "bf16",
                BrowserParityDType::Bf16,
                &bf16::from_f32(1.5).to_bits().to_le_bytes()
            )
            .unwrap(),
            vec![1.5]
        );
        assert!(
            decode_float(
                "bf16",
                BrowserParityDType::Bf16,
                &bf16::NAN.to_bits().to_le_bytes()
            )
            .is_err()
        );
        assert!(
            decode_float("f32", BrowserParityDType::F32, &f32::INFINITY.to_le_bytes()).is_err()
        );
    }

    #[test]
    fn verification_snapshot_requires_every_input_correctness() {
        let mut snapshot = BrowserParityVerificationSnapshot {
            verified_metadata_files: 1,
            verified_metadata_bytes: EDIT_1K5_FIXTURE_METADATA_SIZE,
            verified_metadata_sha256: Some(EDIT_1K5_FIXTURE_METADATA_SHA256.into()),
            verified_safetensors_headers: 1,
            verified_safetensors_header_bytes: 9,
            verified_safetensors_files: 1,
            verified_safetensors_file_bytes: EDIT_1K5_FIXTURE_TENSORS_SIZE,
            verified_safetensors_sha256: Some(EDIT_1K5_FIXTURE_TENSORS_SHA256.into()),
            verified_source_files: 1,
            verified_source_bytes: EDIT_1K5_FIXTURE_SOURCE_SIZE,
            verified_source_sha256: Some(EDIT_1K5_FIXTURE_SOURCE_SHA256.into()),
            verified_output_files: 1,
            verified_output_bytes: EDIT_1K5_FIXTURE_OUTPUT_SIZE,
            verified_output_sha256: Some(EDIT_1K5_FIXTURE_OUTPUT_SHA256.into()),
            verified_tensors: 372,
            verified_tensor_bytes: 123,
            expected_tensors: 372,
            expected_tensor_bytes: 123,
        };
        assert!(snapshot.qualification_inputs_verified());
        snapshot.verified_tensors -= 1;
        assert!(!snapshot.qualification_inputs_verified());
    }

    #[test]
    fn compact_vae_reference_identity_is_distinct_and_fail_closed_correctness() {
        let identity = BrowserParityFixtureIdentity::exact_edit_1k5_vae_reference();
        assert_eq!(identity.metadata.size, EDIT_1K5_VAE_FIXTURE_METADATA_SIZE);
        assert_eq!(identity.tensors.size, EDIT_1K5_VAE_FIXTURE_TENSORS_SIZE);
        assert_eq!(identity.output.size, EDIT_1K5_VAE_FIXTURE_OUTPUT_SIZE);
        assert_ne!(identity, BrowserParityFixtureIdentity::exact_edit_1k5());

        let mut snapshot = BrowserParityVerificationSnapshot {
            verified_metadata_files: 1,
            verified_metadata_bytes: identity.metadata.size,
            verified_metadata_sha256: Some(identity.metadata.sha256.clone()),
            verified_safetensors_headers: 1,
            verified_safetensors_header_bytes: 4_400,
            verified_safetensors_files: 1,
            verified_safetensors_file_bytes: identity.tensors.size,
            verified_safetensors_sha256: Some(identity.tensors.sha256.clone()),
            verified_source_files: 1,
            verified_source_bytes: identity.source.size,
            verified_source_sha256: Some(identity.source.sha256.clone()),
            verified_output_files: 1,
            verified_output_bytes: identity.output.size,
            verified_output_sha256: Some(identity.output.sha256.clone()),
            verified_tensors: EDIT_1K5_VAE_FIXTURE_TENSOR_COUNT as u64,
            verified_tensor_bytes: 49_115_776,
            expected_tensors: EDIT_1K5_VAE_FIXTURE_TENSOR_COUNT as u64,
            expected_tensor_bytes: 49_115_776,
        };
        assert!(snapshot.qualification_inputs_verified_for(&identity));
        assert!(!snapshot.qualification_inputs_verified());
        snapshot.verified_safetensors_sha256 = Some(EDIT_1K5_FIXTURE_TENSORS_SHA256.into());
        assert!(!snapshot.qualification_inputs_verified_for(&identity));
    }
}
