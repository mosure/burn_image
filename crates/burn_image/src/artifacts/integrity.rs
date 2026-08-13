use std::{collections::BTreeMap, fmt::Display};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    ArtifactFile, ArtifactManifest, ArtifactPath, ByteRange, IntegrityError, ManifestError,
    ValidationError,
};

/// A complete SHA-256 digest, serialized as 64 lowercase hexadecimal digits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Sha256Digest([u8; 32]);

impl Sha256Digest {
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub fn calculate(bytes: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        Self(hasher.finalize().into())
    }

    pub fn from_hex(value: &str) -> Result<Self, ValidationError> {
        if value.len() != 64 {
            return Err(ValidationError::OutOfRange {
                field: "sha256",
                range: "exactly 64 hexadecimal digits",
                value: value.len().to_string(),
            });
        }
        let mut bytes = [0u8; 32];
        for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
            let high = decode_hex(pair[0]).ok_or(ValidationError::InvalidCharacter {
                field: "sha256",
                index: index * 2,
            })?;
            let low = decode_hex(pair[1]).ok_or(ValidationError::InvalidCharacter {
                field: "sha256",
                index: index * 2 + 1,
            })?;
            bytes[index] = (high << 4) | low;
        }
        Ok(Self(bytes))
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn to_hex(self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut output = String::with_capacity(64);
        for byte in self.0 {
            output.push(HEX[usize::from(byte >> 4)] as char);
            output.push(HEX[usize::from(byte & 0x0f)] as char);
        }
        output
    }
}

fn decode_hex(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

impl Display for Sha256Digest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl TryFrom<String> for Sha256Digest {
    type Error = ValidationError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::from_hex(&value)
    }
}

impl From<Sha256Digest> for String {
    fn from(value: Sha256Digest) -> Self {
        value.to_hex()
    }
}

/// Integrity level requested by a loader.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntegrityPolicy {
    #[default]
    RequireSha256,
    /// Explicit development escape hatch. This status must not be represented
    /// as cryptographically verified in result provenance.
    SizeOnlyForDevelopment,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationStatus {
    Sha256Verified,
    SizeOnly,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedArtifact {
    path: ArtifactPath,
    size: u64,
    digest: Sha256Digest,
    status: VerificationStatus,
}

impl VerifiedArtifact {
    pub fn path(&self) -> &ArtifactPath {
        &self.path
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn digest(&self) -> Sha256Digest {
        self.digest
    }

    pub fn status(&self) -> VerificationStatus {
        self.status
    }
}

/// Accumulates independently verified files in any arrival order and only
/// yields a bundle after every sealed-manifest entry is present.
pub struct BundleVerifier {
    expected: BTreeMap<ArtifactPath, ArtifactFile>,
    verified: BTreeMap<ArtifactPath, VerifiedArtifact>,
    content_digest: Sha256Digest,
    policy: IntegrityPolicy,
}

impl BundleVerifier {
    pub fn new(
        manifest: &ArtifactManifest,
        policy: IntegrityPolicy,
    ) -> Result<Self, ManifestError> {
        manifest.validate_sealed()?;
        Ok(Self {
            expected: manifest
                .files
                .iter()
                .cloned()
                .map(|file| (file.path.clone(), file))
                .collect(),
            verified: BTreeMap::new(),
            content_digest: manifest
                .content_digest
                .expect("sealed manifests have a content digest"),
            policy,
        })
    }

    pub fn verify_bytes(
        &mut self,
        path: &ArtifactPath,
        bytes: &[u8],
    ) -> Result<(), IntegrityError> {
        let file = self
            .expected
            .get(path)
            .cloned()
            .ok_or_else(|| IntegrityError::UnknownArtifact { path: path.clone() })?;
        let verified = ArtifactVerifier::verify_bytes(&file, bytes, self.policy)?;
        self.record(verified)
    }

    pub fn record(&mut self, artifact: VerifiedArtifact) -> Result<(), IntegrityError> {
        let path = artifact.path.clone();
        let expected = self
            .expected
            .get(&path)
            .ok_or_else(|| IntegrityError::UnknownArtifact { path: path.clone() })?;
        if self.verified.contains_key(&path) {
            return Err(IntegrityError::DuplicateArtifact { path });
        }
        if artifact.size != expected.size
            || (self.policy == IntegrityPolicy::RequireSha256 && artifact.digest != expected.sha256)
        {
            return Err(IntegrityError::VerificationMetadataMismatch { path });
        }
        if self.policy == IntegrityPolicy::RequireSha256
            && artifact.status != VerificationStatus::Sha256Verified
        {
            return Err(IntegrityError::InsufficientVerification { path });
        }
        self.verified.insert(path, artifact);
        Ok(())
    }

    pub fn verified_count(&self) -> usize {
        self.verified.len()
    }

    pub fn expected_count(&self) -> usize {
        self.expected.len()
    }

    pub fn finish(self) -> Result<VerifiedBundle, IntegrityError> {
        for path in self.expected.keys() {
            if !self.verified.contains_key(path) {
                return Err(IntegrityError::MissingArtifact { path: path.clone() });
            }
        }
        let cryptographically_verified = self
            .verified
            .values()
            .all(|artifact| artifact.status == VerificationStatus::Sha256Verified);
        Ok(VerifiedBundle {
            files: self.verified,
            content_digest: self.content_digest,
            cryptographically_verified,
        })
    }
}

/// Complete verification result for a sealed artifact manifest.
pub struct VerifiedBundle {
    files: BTreeMap<ArtifactPath, VerifiedArtifact>,
    content_digest: Sha256Digest,
    cryptographically_verified: bool,
}

impl VerifiedBundle {
    pub fn file(&self, path: &ArtifactPath) -> Option<&VerifiedArtifact> {
        self.files.get(path)
    }

    pub fn len(&self) -> usize {
        self.files.len()
    }

    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    pub fn content_digest(&self) -> Sha256Digest {
        self.content_digest
    }

    pub fn is_cryptographically_verified(&self) -> bool {
        self.cryptographically_verified
    }
}

/// Bounded streaming verifier for an artifact file.
pub struct ArtifactVerifier {
    path: ArtifactPath,
    expected_size: u64,
    expected_digest: Sha256Digest,
    observed_size: u64,
    hasher: Sha256,
    policy: IntegrityPolicy,
}

impl ArtifactVerifier {
    pub fn new(file: &ArtifactFile, policy: IntegrityPolicy) -> Self {
        Self {
            path: file.path.clone(),
            expected_size: file.size,
            expected_digest: file.sha256,
            observed_size: 0,
            hasher: Sha256::new(),
            policy,
        }
    }

    pub fn update(&mut self, bytes: &[u8]) -> Result<(), IntegrityError> {
        let chunk_size =
            u64::try_from(bytes.len()).map_err(|_| IntegrityError::ByteCountOverflow)?;
        let next = self
            .observed_size
            .checked_add(chunk_size)
            .ok_or(IntegrityError::ByteCountOverflow)?;
        if next > self.expected_size {
            return Err(IntegrityError::SizeExceeded {
                path: self.path.clone(),
                expected: self.expected_size,
            });
        }
        self.hasher.update(bytes);
        self.observed_size = next;
        Ok(())
    }

    /// Verify that a fetched range is the exact next interval before hashing
    /// it. SHA-256 is sequential, so out-of-order ranges must not be silently
    /// concatenated in arrival order.
    pub fn update_range(&mut self, range: ByteRange, bytes: &[u8]) -> Result<(), IntegrityError> {
        if range.offset() != self.observed_size {
            return Err(IntegrityError::UnexpectedRangeOffset {
                path: self.path.clone(),
                expected: self.observed_size,
                actual: range.offset(),
            });
        }
        let actual = u64::try_from(bytes.len()).map_err(|_| IntegrityError::ByteCountOverflow)?;
        if actual != range.length() {
            return Err(IntegrityError::RangeLengthMismatch {
                path: self.path.clone(),
                expected: range.length(),
                actual,
            });
        }
        self.update(bytes)
    }

    pub fn observed_size(&self) -> u64 {
        self.observed_size
    }

    pub fn finish(self) -> Result<VerifiedArtifact, IntegrityError> {
        if self.observed_size != self.expected_size {
            return Err(IntegrityError::SizeMismatch {
                path: self.path,
                expected: self.expected_size,
                actual: self.observed_size,
            });
        }
        let digest = Sha256Digest::from_bytes(self.hasher.finalize().into());
        if self.policy == IntegrityPolicy::RequireSha256 && digest != self.expected_digest {
            return Err(IntegrityError::DigestMismatch {
                path: self.path,
                expected: self.expected_digest,
                actual: digest,
            });
        }
        Ok(VerifiedArtifact {
            path: self.path,
            size: self.observed_size,
            digest,
            status: match self.policy {
                IntegrityPolicy::RequireSha256 => VerificationStatus::Sha256Verified,
                IntegrityPolicy::SizeOnlyForDevelopment => VerificationStatus::SizeOnly,
            },
        })
    }

    pub fn verify_bytes(
        file: &ArtifactFile,
        bytes: &[u8],
        policy: IntegrityPolicy,
    ) -> Result<VerifiedArtifact, IntegrityError> {
        let mut verifier = Self::new(file, policy);
        verifier.update(bytes)?;
        verifier.finish()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::{
        ARTIFACT_MANIFEST_SCHEMA_VERSION, ArtifactBundleId, ArtifactComponent, ArtifactComponentId,
        ArtifactFile, ArtifactFileRole, ArtifactManifest, ArtifactPath, ArtifactProfileId,
        ArtifactShard, BundleVerifier, ByteRange, IntegrityPolicy, ModelId, NumericFormat,
        Sha256Digest,
    };

    use super::{ArtifactVerifier, VerificationStatus};

    fn file(bytes: &[u8]) -> ArtifactFile {
        ArtifactFile {
            path: ArtifactPath::new("weights/model.bpk").unwrap(),
            size: bytes.len() as u64,
            sha256: Sha256Digest::calculate(bytes),
            role: ArtifactFileRole::Weights,
            component: None,
            shard: None,
        }
    }

    #[test]
    fn streaming_verifier_accepts_multiple_bounded_chunks_correctness() {
        let bytes = b"abcdef";
        let file = file(bytes);
        let mut verifier = ArtifactVerifier::new(&file, IntegrityPolicy::RequireSha256);
        verifier.update(&bytes[..2]).unwrap();
        verifier.update(&bytes[2..]).unwrap();
        let verified = verifier.finish().unwrap();
        assert_eq!(verified.status(), VerificationStatus::Sha256Verified);
    }

    #[test]
    fn streaming_verifier_rejects_wrong_digest_and_oversize_correctness() {
        let bytes = b"abcdef";
        let mut wrong = file(bytes);
        wrong.sha256 = Sha256Digest::calculate(b"xxxxxx");
        assert!(
            ArtifactVerifier::verify_bytes(&wrong, bytes, IntegrityPolicy::RequireSha256).is_err()
        );

        let file = file(bytes);
        let mut verifier = ArtifactVerifier::new(&file, IntegrityPolicy::RequireSha256);
        assert!(verifier.update(b"abcdefg").is_err());
    }

    #[test]
    fn ranged_verification_requires_sequential_exact_intervals_correctness() {
        let bytes = b"abcdef";
        let file = file(bytes);
        let mut verifier = ArtifactVerifier::new(&file, IntegrityPolicy::RequireSha256);
        assert!(
            verifier
                .update_range(ByteRange::new(2, 2).unwrap(), b"cd")
                .is_err()
        );
        verifier
            .update_range(ByteRange::new(0, 2).unwrap(), b"ab")
            .unwrap();
        assert!(
            verifier
                .update_range(ByteRange::new(2, 3).unwrap(), b"cd")
                .is_err()
        );
        verifier
            .update_range(ByteRange::new(2, 4).unwrap(), b"cdef")
            .unwrap();
        assert!(verifier.finish().is_ok());
    }

    #[test]
    fn bundle_verifier_accepts_files_in_any_order_and_requires_all_files_correctness() {
        let component = ArtifactComponentId::new("transformer").unwrap();
        let first = b"first";
        let second = b"second";
        let mut manifest = ArtifactManifest {
            schema_version: ARTIFACT_MANIFEST_SCHEMA_VERSION,
            bundle: ArtifactBundleId::new("bundle").unwrap(),
            profile: ArtifactProfileId::new("f16").unwrap(),
            model: ModelId::new("owner/model").unwrap(),
            model_revision: "revision".to_string(),
            numeric_format: NumericFormat::F16,
            components: vec![ArtifactComponent {
                id: component.clone(),
                required: true,
            }],
            files: vec![
                ArtifactFile {
                    path: ArtifactPath::new("weights/part-000").unwrap(),
                    size: first.len() as u64,
                    sha256: Sha256Digest::calculate(first),
                    role: ArtifactFileRole::Weights,
                    component: Some(component.clone()),
                    shard: Some(ArtifactShard {
                        index: 0,
                        count: 2,
                        chain_sha256: None,
                    }),
                },
                ArtifactFile {
                    path: ArtifactPath::new("weights/part-001").unwrap(),
                    size: second.len() as u64,
                    sha256: Sha256Digest::calculate(second),
                    role: ArtifactFileRole::Weights,
                    component: Some(component),
                    shard: Some(ArtifactShard {
                        index: 1,
                        count: 2,
                        chain_sha256: None,
                    }),
                },
            ],
            dependencies: Vec::new(),
            metadata: BTreeMap::new(),
            content_digest: None,
        };
        manifest.seal().unwrap();

        let mut incomplete =
            BundleVerifier::new(&manifest, IntegrityPolicy::RequireSha256).unwrap();
        incomplete
            .verify_bytes(&ArtifactPath::new("weights/part-001").unwrap(), second)
            .unwrap();
        assert!(incomplete.finish().is_err());

        let mut verifier = BundleVerifier::new(&manifest, IntegrityPolicy::RequireSha256).unwrap();
        verifier
            .verify_bytes(&ArtifactPath::new("weights/part-001").unwrap(), second)
            .unwrap();
        verifier
            .verify_bytes(&ArtifactPath::new("weights/part-000").unwrap(), first)
            .unwrap();
        let verified = verifier.finish().unwrap();
        assert_eq!(verified.len(), 2);
        assert!(verified.is_cryptographically_verified());
        assert_eq!(verified.content_digest(), manifest.content_digest.unwrap());
    }
}
