//! Authentication of exported upstream numerical reference fixtures.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

use safetensors::{Dtype, SafeTensors, tensor::Metadata};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::BooguError;

#[derive(Debug, Deserialize)]
struct FixtureDigestEnvelope {
    tensors: BTreeMap<String, FixtureTensorDigest>,
}

#[derive(Debug, Deserialize)]
struct FixtureTensorDigest {
    dtype: String,
    shape: Vec<usize>,
    sha256: String,
}

/// Authenticate every tensor in an exported upstream fixture.
///
/// The metadata and SafeTensors keysets must be identical. Each tensor is then checked for its
/// exact shape, PyTorch dtype spelling, and SHA-256 digest over the raw contiguous tensor bytes.
/// Callers must complete this check before making any numerical parity claim.
pub fn verify_reference_fixture(
    metadata_json: &[u8],
    safetensors_bytes: &[u8],
) -> Result<usize, BooguError> {
    let metadata: FixtureDigestEnvelope =
        serde_json::from_slice(metadata_json).map_err(|error| {
            BooguError::Artifact(format!(
                "reference metadata does not contain a valid tensor digest map: {error}"
            ))
        })?;
    if metadata.tensors.is_empty() {
        return Err(BooguError::Artifact(
            "reference metadata tensor digest map is empty".into(),
        ));
    }
    let tensors = SafeTensors::deserialize(safetensors_bytes).map_err(|error| {
        BooguError::Artifact(format!("invalid reference SafeTensors payload: {error}"))
    })?;
    let actual_names = tensors
        .names()
        .into_iter()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    let expected_names = metadata.tensors.keys().cloned().collect::<BTreeSet<_>>();
    if actual_names != expected_names {
        let missing = expected_names
            .difference(&actual_names)
            .take(8)
            .cloned()
            .collect::<Vec<_>>();
        let unexpected = actual_names
            .difference(&expected_names)
            .take(8)
            .cloned()
            .collect::<Vec<_>>();
        return Err(BooguError::Artifact(format!(
            "reference tensor keyset differs from metadata; missing {missing:?}, unexpected {unexpected:?}"
        )));
    }
    for (name, expected) in metadata.tensors {
        let tensor = tensors.tensor(&name).map_err(|error| {
            BooguError::Artifact(format!("failed to read reference tensor {name}: {error}"))
        })?;
        if tensor.shape() != expected.shape {
            return Err(BooguError::Artifact(format!(
                "reference tensor {name} shape {:?} differs from authenticated {:?}",
                tensor.shape(),
                expected.shape
            )));
        }
        let dtype = pytorch_dtype_name(tensor.dtype()).ok_or_else(|| {
            BooguError::Artifact(format!(
                "reference tensor {name} uses unsupported dtype {:?}",
                tensor.dtype()
            ))
        })?;
        if dtype != expected.dtype {
            return Err(BooguError::Artifact(format!(
                "reference tensor {name} dtype {dtype} differs from authenticated {}",
                expected.dtype
            )));
        }
        let actual_digest = hex::encode(Sha256::digest(tensor.data()));
        if actual_digest != expected.sha256 {
            return Err(BooguError::Artifact(format!(
                "reference tensor {name} SHA-256 {actual_digest} differs from authenticated {}",
                expected.sha256
            )));
        }
    }
    Ok(actual_names.len())
}

/// Streaming variant of [`verify_reference_fixture`] for large release oracles.
///
/// Only the SafeTensors header and a 1 MiB hashing buffer are held in memory. File length and every
/// tensor byte range are checked before hashing, so malformed offsets cannot escape the validated
/// container.
pub fn verify_reference_fixture_file(
    metadata_json: &[u8],
    safetensors_path: impl AsRef<Path>,
) -> Result<usize, BooguError> {
    let expected: FixtureDigestEnvelope =
        serde_json::from_slice(metadata_json).map_err(|error| {
            BooguError::Artifact(format!(
                "reference metadata does not contain a valid tensor digest map: {error}"
            ))
        })?;
    if expected.tensors.is_empty() {
        return Err(BooguError::Artifact(
            "reference metadata tensor digest map is empty".into(),
        ));
    }
    let path = safetensors_path.as_ref();
    let mut file = File::open(path).map_err(|error| {
        BooguError::Artifact(format!(
            "failed to open reference fixture {}: {error}",
            path.display()
        ))
    })?;
    let mut length = [0_u8; 8];
    file.read_exact(&mut length).map_err(|error| {
        BooguError::Artifact(format!("failed to read reference header length: {error}"))
    })?;
    let header_len = usize::try_from(u64::from_le_bytes(length)).map_err(|error| {
        BooguError::Artifact(format!(
            "reference header length is not addressable: {error}"
        ))
    })?;
    if header_len == 0 || header_len > 100 * 1024 * 1024 {
        return Err(BooguError::Artifact(format!(
            "invalid reference SafeTensors header length {header_len}"
        )));
    }
    let mut header = vec![0_u8; header_len];
    file.read_exact(&mut header).map_err(|error| {
        BooguError::Artifact(format!("failed to read reference header: {error}"))
    })?;
    let actual: Metadata = serde_json::from_slice(&header).map_err(|error| {
        BooguError::Artifact(format!("invalid reference SafeTensors header: {error}"))
    })?;
    let data_start = 8_u64
        .checked_add(u64::try_from(header_len).unwrap_or(u64::MAX))
        .ok_or_else(|| BooguError::Artifact("reference data offset overflow".into()))?;
    let expected_size = data_start
        .checked_add(u64::try_from(actual.data_len()).unwrap_or(u64::MAX))
        .ok_or_else(|| BooguError::Artifact("reference file size overflow".into()))?;
    let actual_size = file
        .metadata()
        .map_err(|error| BooguError::Artifact(format!("failed to stat reference: {error}")))?
        .len();
    if actual_size != expected_size {
        return Err(BooguError::Artifact(format!(
            "reference file size {actual_size} differs from header-derived {expected_size}"
        )));
    }
    let actual_names = actual.tensors().keys().cloned().collect::<BTreeSet<_>>();
    let expected_names = expected.tensors.keys().cloned().collect::<BTreeSet<_>>();
    verify_keysets(&expected_names, &actual_names)?;

    let mut buffer = vec![0_u8; 1024 * 1024];
    for (name, digest) in expected.tensors {
        let info = actual
            .info(&name)
            .ok_or_else(|| BooguError::Artifact(format!("reference header omits tensor {name}")))?;
        verify_tensor_contract(&name, &digest, info.shape.as_slice(), info.dtype)?;
        let start = data_start
            .checked_add(u64::try_from(info.data_offsets.0).unwrap_or(u64::MAX))
            .ok_or_else(|| {
                BooguError::Artifact(format!("reference tensor {name} offset overflow"))
            })?;
        let byte_len = info
            .data_offsets
            .1
            .checked_sub(info.data_offsets.0)
            .ok_or_else(|| {
                BooguError::Artifact(format!("reference tensor {name} offset underflow"))
            })?;
        file.seek(SeekFrom::Start(start)).map_err(|error| {
            BooguError::Artifact(format!("failed to seek reference tensor {name}: {error}"))
        })?;
        let mut remaining = byte_len;
        let mut hasher = Sha256::new();
        while remaining > 0 {
            let count = remaining.min(buffer.len());
            file.read_exact(&mut buffer[..count]).map_err(|error| {
                BooguError::Artifact(format!("failed to read reference tensor {name}: {error}"))
            })?;
            hasher.update(&buffer[..count]);
            remaining -= count;
        }
        let actual_digest = hex::encode(hasher.finalize());
        if actual_digest != digest.sha256 {
            return Err(BooguError::Artifact(format!(
                "reference tensor {name} SHA-256 {actual_digest} differs from authenticated {}",
                digest.sha256
            )));
        }
    }
    Ok(actual_names.len())
}

fn verify_keysets(
    expected_names: &BTreeSet<String>,
    actual_names: &BTreeSet<String>,
) -> Result<(), BooguError> {
    if actual_names == expected_names {
        return Ok(());
    }
    let missing = expected_names
        .difference(actual_names)
        .take(8)
        .cloned()
        .collect::<Vec<_>>();
    let unexpected = actual_names
        .difference(expected_names)
        .take(8)
        .cloned()
        .collect::<Vec<_>>();
    Err(BooguError::Artifact(format!(
        "reference tensor keyset differs from metadata; missing {missing:?}, unexpected {unexpected:?}"
    )))
}

fn verify_tensor_contract(
    name: &str,
    expected: &FixtureTensorDigest,
    shape: &[usize],
    dtype: Dtype,
) -> Result<(), BooguError> {
    if shape != expected.shape {
        return Err(BooguError::Artifact(format!(
            "reference tensor {name} shape {shape:?} differs from authenticated {:?}",
            expected.shape
        )));
    }
    let dtype = pytorch_dtype_name(dtype).ok_or_else(|| {
        BooguError::Artifact(format!(
            "reference tensor {name} uses unsupported dtype {dtype:?}"
        ))
    })?;
    if dtype != expected.dtype {
        return Err(BooguError::Artifact(format!(
            "reference tensor {name} dtype {dtype} differs from authenticated {}",
            expected.dtype
        )));
    }
    Ok(())
}

fn pytorch_dtype_name(dtype: Dtype) -> Option<&'static str> {
    match dtype {
        Dtype::BF16 => Some("torch.bfloat16"),
        Dtype::F16 => Some("torch.float16"),
        Dtype::F32 => Some("torch.float32"),
        Dtype::F64 => Some("torch.float64"),
        Dtype::I64 => Some("torch.int64"),
        Dtype::I32 => Some("torch.int32"),
        Dtype::U8 => Some("torch.uint8"),
        Dtype::BOOL => Some("torch.bool"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use safetensors::{Dtype, tensor::TensorView};
    use sha2::{Digest, Sha256};

    use super::verify_reference_fixture;

    fn fixture(value: &[u8]) -> (Vec<u8>, Vec<u8>) {
        let view = TensorView::new(Dtype::U8, vec![value.len()], value).unwrap();
        let tensors = HashMap::from([("value".to_owned(), view)]);
        let bytes = safetensors::serialize(tensors, None).unwrap();
        let metadata = serde_json::to_vec(&serde_json::json!({
            "tensors": {
                "value": {
                    "dtype": "torch.uint8",
                    "shape": [value.len()],
                    "sha256": hex::encode(Sha256::digest(value)),
                }
            }
        }))
        .unwrap();
        (metadata, bytes)
    }

    #[test]
    fn exact_tensor_digest_map_is_authenticated_correctness() {
        let (metadata, bytes) = fixture(&[1, 2, 3, 4]);
        assert_eq!(verify_reference_fixture(&metadata, &bytes).unwrap(), 1);
    }

    #[test]
    fn modified_reference_tensor_is_rejected_correctness() {
        let (metadata, _) = fixture(&[1, 2, 3, 4]);
        let (_, modified) = fixture(&[1, 2, 3, 5]);
        let error = verify_reference_fixture(&metadata, &modified).unwrap_err();
        assert!(error.to_string().contains("SHA-256"));
    }
}
