use crate::{ArtifactPath, Dimensions, ImageTaskKind, ModelId, Sha256Digest};
use thiserror::Error;

/// Errors produced while constructing or validating portable API values.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ValidationError {
    #[error("{field} must not be empty")]
    Empty { field: &'static str },
    #[error("{field} exceeds the maximum length of {max} bytes (got {actual})")]
    TooLong {
        field: &'static str,
        max: usize,
        actual: usize,
    },
    #[error("{field} contains an invalid character at byte {index}")]
    InvalidCharacter { field: &'static str, index: usize },
    #[error("image dimensions must be non-zero (got {width}x{height})")]
    ZeroDimensions { width: u32, height: u32 },
    #[error("image area overflow for {width}x{height}")]
    DimensionOverflow { width: u32, height: u32 },
    #[error("{field} must be finite (got {value})")]
    NonFinite { field: &'static str, value: String },
    #[error("{field} must be in {range} (got {value})")]
    OutOfRange {
        field: &'static str,
        range: &'static str,
        value: String,
    },
    #[error("{field} must be greater than zero")]
    MustBePositive { field: &'static str },
    #[error("pixel byte length mismatch: expected {expected}, got {actual}")]
    PixelLengthMismatch { expected: usize, actual: usize },
    #[error("mask dimensions {mask} do not match source dimensions {source_dimensions}")]
    MaskDimensionMismatch {
        mask: Dimensions,
        source_dimensions: Dimensions,
    },
    #[error("output list must not be empty")]
    EmptyOutput,
    #[error("duplicate output index {index}")]
    DuplicateOutputIndex { index: u32 },
    #[error("invalid timing interval for stage '{stage}': finish precedes start")]
    InvalidTimingInterval { stage: String },
}

/// Errors produced by an artifact manifest before any bytes are loaded.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ManifestError {
    #[error("unsupported artifact manifest schema {actual}; expected {expected}")]
    UnsupportedSchema { expected: u32, actual: u32 },
    #[error("artifact dependencies require manifest schema 2")]
    DependenciesRequireSchemaV2,
    #[error("artifact manifest declares duplicate dependency role '{role}'")]
    DuplicateDependencyRole { role: String },
    #[error("artifact manifest declares dependency bundle '{bundle}' more than once")]
    DuplicateDependencyBundle { bundle: String },
    #[error("artifact bundle '{bundle}' depends on itself")]
    SelfDependency { bundle: String },
    #[error("dependency role '{role}' resolved {field} '{actual}', expected '{expected}'")]
    DependencyIdentityMismatch {
        role: String,
        field: &'static str,
        expected: String,
        actual: String,
    },
    #[error("dependency role '{role}' content digest mismatch: expected {expected}, got {actual}")]
    DependencyContentDigestMismatch {
        role: String,
        expected: Sha256Digest,
        actual: Sha256Digest,
    },
    #[error("dependency role '{role}' could not resolve bundle '{bundle}'")]
    MissingResolvedDependency { role: String, bundle: String },
    #[error(
        "dependency closure resolved bundle '{bundle}' to conflicting digests {expected} and {actual}"
    )]
    DependencyBundleConflict {
        bundle: String,
        expected: Sha256Digest,
        actual: Sha256Digest,
    },
    #[error("artifact dependency cycle detected: {cycle:?}")]
    DependencyCycle { cycle: Vec<String> },
    #[error("artifact manifest contains no files")]
    EmptyFiles,
    #[error("duplicate artifact path '{0}'")]
    DuplicatePath(ArtifactPath),
    #[error("artifact '{path}' has zero bytes")]
    ZeroLength { path: ArtifactPath },
    #[error("artifact '{path}' has invalid shard metadata: {reason}")]
    InvalidShard { path: ArtifactPath, reason: String },
    #[error("component '{component}' has inconsistent shard counts {expected} and {actual}")]
    InconsistentShardCount {
        component: String,
        expected: u32,
        actual: u32,
    },
    #[error("component '{component}' is missing shard index {index} of {count}")]
    MissingShard {
        component: String,
        index: u32,
        count: u32,
    },
    #[error("component '{component}' contains duplicate shard index {index}")]
    DuplicateShard { component: String, index: u32 },
    #[error("artifact '{path}' is missing its shard hash-chain digest")]
    MissingHashChain { path: ArtifactPath },
    #[error("artifact '{path}' hash-chain mismatch: expected {expected}, got {actual}")]
    HashChainMismatch {
        path: ArtifactPath,
        expected: Sha256Digest,
        actual: Sha256Digest,
    },
    #[error("component '{component}' mixes sharded and unsharded weight files")]
    MixedShardLayout { component: String },
    #[error("artifact manifest declares duplicate component '{component}'")]
    DuplicateComponent { component: String },
    #[error("artifact manifest is missing files for required component '{component}'")]
    MissingComponent { component: String },
    #[error("artifact manifest does not declare component '{component}' used by '{path}'")]
    UnknownComponent {
        component: String,
        path: ArtifactPath,
    },
    #[error("artifact bundle content digest mismatch: expected {expected}, got {actual}")]
    ContentDigestMismatch {
        expected: Sha256Digest,
        actual: Sha256Digest,
    },
    #[error("sealed artifact manifest is missing its content digest")]
    MissingContentDigest,
    #[error(transparent)]
    Validation(#[from] ValidationError),
}

/// Errors produced while checking artifact bytes.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum IntegrityError {
    #[error("artifact '{path}' exceeded its declared size of {expected} bytes")]
    SizeExceeded { path: ArtifactPath, expected: u64 },
    #[error("artifact '{path}' size mismatch: expected {expected} bytes, got {actual}")]
    SizeMismatch {
        path: ArtifactPath,
        expected: u64,
        actual: u64,
    },
    #[error("artifact '{path}' SHA-256 mismatch: expected {expected}, got {actual}")]
    DigestMismatch {
        path: ArtifactPath,
        expected: Sha256Digest,
        actual: Sha256Digest,
    },
    #[error("artifact byte count overflow")]
    ByteCountOverflow,
    #[error("artifact '{path}' expected the next range at byte {expected}, got {actual}")]
    UnexpectedRangeOffset {
        path: ArtifactPath,
        expected: u64,
        actual: u64,
    },
    #[error("artifact '{path}' range declared {expected} bytes but delivered {actual}")]
    RangeLengthMismatch {
        path: ArtifactPath,
        expected: u64,
        actual: u64,
    },
    #[error("artifact '{path}' is not declared by the manifest")]
    UnknownArtifact { path: ArtifactPath },
    #[error("artifact '{path}' was verified more than once")]
    DuplicateArtifact { path: ArtifactPath },
    #[error("artifact '{path}' verification metadata does not match the manifest")]
    VerificationMetadataMismatch { path: ArtifactPath },
    #[error("artifact '{path}' has not been verified")]
    MissingArtifact { path: ArtifactPath },
    #[error("artifact '{path}' was only size-checked but SHA-256 verification is required")]
    InsufficientVerification { path: ArtifactPath },
}

/// Errors produced by capability checks and runtime routing.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum RuntimeError {
    #[error("model '{0}' is not registered")]
    UnknownModel(ModelId),
    #[error("model '{0}' is already registered")]
    DuplicateModel(ModelId),
    #[error("model '{model}' does not support task {task}")]
    UnsupportedTask { model: ModelId, task: ImageTaskKind },
    #[error("model '{model}' does not support edit masks")]
    MasksUnsupported { model: ModelId },
    #[error("model '{model}' does not support requested dimensions {requested}: {reason}")]
    UnsupportedDimensions {
        model: ModelId,
        requested: Dimensions,
        reason: String,
    },
    #[error("model '{model}' does not support {steps} inference steps (allowed {min}..={max})")]
    UnsupportedSteps {
        model: ModelId,
        steps: u32,
        min: u32,
        max: u32,
    },
    #[error("model '{model}' supports batches up to {max}, requested {requested}")]
    UnsupportedBatchSize {
        model: ModelId,
        requested: u32,
        max: u32,
    },
    #[error("model descriptor id '{descriptor}' does not match runtime selection '{selected}'")]
    ModelSelectionMismatch {
        selected: ModelId,
        descriptor: ModelId,
    },
    #[error("inference was cancelled")]
    Cancelled,
    #[error("model '{model}' execution failed: {message}")]
    ModelExecution { model: ModelId, message: String },
    #[error(transparent)]
    Validation(#[from] ValidationError),
}

/// Top-level error for portable `burn_image` operations.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum ImageError {
    #[error(transparent)]
    Validation(#[from] ValidationError),
    #[error(transparent)]
    Manifest(#[from] ManifestError),
    #[error(transparent)]
    Integrity(#[from] IntegrityError),
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
}
