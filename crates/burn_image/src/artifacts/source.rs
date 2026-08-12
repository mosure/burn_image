use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{ArtifactFile, ArtifactPath, ValidationError};

/// Half-open byte range `[offset, offset + length)` used by native readers and
/// HTTP range requests.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "ByteRangeWire", into = "ByteRangeWire")]
pub struct ByteRange {
    offset: u64,
    length: u64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct ByteRangeWire {
    offset: u64,
    length: u64,
}

impl ByteRange {
    pub fn new(offset: u64, length: u64) -> Result<Self, ValidationError> {
        if length == 0 {
            return Err(ValidationError::MustBePositive {
                field: "artifact.byte_range.length",
            });
        }
        offset
            .checked_add(length)
            .ok_or(ValidationError::OutOfRange {
                field: "artifact.byte_range",
                range: "a non-overflowing u64 interval",
                value: format!("{offset}+{length}"),
            })?;
        Ok(Self { offset, length })
    }

    pub fn offset(self) -> u64 {
        self.offset
    }

    pub fn length(self) -> u64 {
        self.length
    }

    pub fn end_exclusive(self) -> u64 {
        self.offset + self.length
    }

    pub fn http_range_header(self) -> String {
        format!("bytes={}-{}", self.offset, self.end_exclusive() - 1)
    }

    pub fn validate_for_size(self, size: u64) -> Result<(), ValidationError> {
        if self.end_exclusive() > size {
            return Err(ValidationError::OutOfRange {
                field: "artifact.byte_range",
                range: "within the declared artifact size",
                value: format!("{}..{} of {size}", self.offset, self.end_exclusive()),
            });
        }
        Ok(())
    }
}

impl TryFrom<ByteRangeWire> for ByteRange {
    type Error = ValidationError;

    fn try_from(value: ByteRangeWire) -> Result<Self, Self::Error> {
        Self::new(value.offset, value.length)
    }
}

impl From<ByteRange> for ByteRangeWire {
    fn from(value: ByteRange) -> Self {
        Self {
            offset: value.offset,
            length: value.length,
        }
    }
}

/// One transport read operation. A missing range requests the complete file.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactReadRequest {
    pub path: ArtifactPath,
    pub range: Option<ByteRange>,
}

impl ArtifactReadRequest {
    pub fn full(path: ArtifactPath) -> Self {
        Self { path, range: None }
    }

    pub fn ranged(path: ArtifactPath, range: ByteRange) -> Self {
        Self {
            path,
            range: Some(range),
        }
    }

    pub fn validate_for_file(&self, file: &ArtifactFile) -> Result<(), ValidationError> {
        if self.path != file.path {
            return Err(ValidationError::OutOfRange {
                field: "artifact.read_request.path",
                range: "the selected manifest file path",
                value: self.path.to_string(),
            });
        }
        if let Some(range) = self.range {
            range.validate_for_size(file.size)?;
        }
        Ok(())
    }
}

/// Fully resolved request consumed by a native filesystem or HTTP fetcher.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolvedArtifactRequest {
    Local {
        path: PathBuf,
        range: Option<ByteRange>,
    },
    Remote {
        url: String,
        range: Option<ByteRange>,
        range_header: Option<String>,
    },
}

/// Validated HTTP(S) base URL for remote artifact resolution.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct RemoteBaseUrl(String);

impl RemoteBaseUrl {
    pub fn new(value: impl Into<String>) -> Result<Self, ValidationError> {
        let mut value = value.into();
        if !(value.starts_with("https://") || value.starts_with("http://")) {
            return Err(ValidationError::OutOfRange {
                field: "artifact.remote_base_url",
                range: "an http:// or https:// base URL",
                value,
            });
        }
        if value.contains('?') || value.contains('#') || value.chars().any(char::is_whitespace) {
            return Err(ValidationError::InvalidCharacter {
                field: "artifact.remote_base_url",
                index: 0,
            });
        }
        while value.ends_with('/') {
            value.pop();
        }
        let scheme_end = value.find("://").expect("validated URL scheme") + 3;
        let authority = value[scheme_end..].split('/').next().unwrap_or_default();
        if authority.is_empty() || authority.starts_with(':') || authority.contains('@') {
            return Err(ValidationError::Empty {
                field: "artifact.remote_base_url.host",
            });
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn resolve(&self, path: &ArtifactPath) -> String {
        format!("{}/{}", self.0, path.as_str())
    }
}

impl TryFrom<String> for RemoteBaseUrl {
    type Error = ValidationError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<RemoteBaseUrl> for String {
    fn from(value: RemoteBaseUrl) -> Self {
        value.0
    }
}

/// Transport-neutral artifact origin. Fetching and caching are implemented by
/// native or browser adapters outside this crate.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ArtifactSource {
    LocalDirectory { root: PathBuf },
    Remote { base_url: RemoteBaseUrl },
}

impl ArtifactSource {
    pub fn local_path(&self, path: &ArtifactPath) -> Option<PathBuf> {
        match self {
            Self::LocalDirectory { root } => Some(root.join(path.as_str())),
            Self::Remote { .. } => None,
        }
    }

    pub fn remote_url(&self, path: &ArtifactPath) -> Option<String> {
        match self {
            Self::Remote { base_url } => Some(base_url.resolve(path)),
            Self::LocalDirectory { .. } => None,
        }
    }

    pub fn local_root(&self) -> Option<&Path> {
        match self {
            Self::LocalDirectory { root } => Some(root.as_path()),
            Self::Remote { .. } => None,
        }
    }

    pub fn resolve_request(&self, request: &ArtifactReadRequest) -> ResolvedArtifactRequest {
        match self {
            Self::LocalDirectory { root } => ResolvedArtifactRequest::Local {
                path: root.join(request.path.as_str()),
                range: request.range,
            },
            Self::Remote { base_url } => ResolvedArtifactRequest::Remote {
                url: base_url.resolve(&request.path),
                range: request.range,
                range_header: request.range.map(ByteRange::http_range_header),
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactCachePolicy {
    #[default]
    UseCached,
    Refresh,
    Bypass,
}

#[cfg(test)]
mod tests {
    use super::{ArtifactReadRequest, ByteRange, RemoteBaseUrl, ResolvedArtifactRequest};
    use crate::{ArtifactPath, ArtifactSource};

    #[test]
    fn remote_base_url_resolves_validated_paths_correctness() {
        let base = RemoteBaseUrl::new("https://cdn.example/models/").unwrap();
        let path = ArtifactPath::new("model/part-000.bpk").unwrap();
        assert_eq!(
            base.resolve(&path),
            "https://cdn.example/models/model/part-000.bpk"
        );
    }

    #[test]
    fn remote_base_url_rejects_query_and_non_http_schemes_correctness() {
        assert!(RemoteBaseUrl::new("file:///tmp/models").is_err());
        assert!(RemoteBaseUrl::new("https://example/models?token=x").is_err());
        assert!(RemoteBaseUrl::new("https:///models").is_err());
    }

    #[test]
    fn byte_range_has_unambiguous_half_open_and_http_forms_correctness() {
        let range = ByteRange::new(64, 32).unwrap();
        assert_eq!(range.end_exclusive(), 96);
        assert_eq!(range.http_range_header(), "bytes=64-95");
        assert!(range.validate_for_size(96).is_ok());
        assert!(range.validate_for_size(95).is_err());
        assert!(ByteRange::new(u64::MAX, 2).is_err());
    }

    #[test]
    fn source_resolution_preserves_range_contract_correctness() {
        let source = ArtifactSource::Remote {
            base_url: RemoteBaseUrl::new("https://cdn.example/models").unwrap(),
        };
        let request = ArtifactReadRequest::ranged(
            ArtifactPath::new("weights/part-000").unwrap(),
            ByteRange::new(10, 5).unwrap(),
        );
        assert_eq!(
            source.resolve_request(&request),
            ResolvedArtifactRequest::Remote {
                url: "https://cdn.example/models/weights/part-000".to_string(),
                range: request.range,
                range_header: Some("bytes=10-14".to_string()),
            }
        );
    }
}
