use serde::{Deserialize, Serialize};

use crate::{ArtifactCachePolicy, ArtifactProfileId, ArtifactSource, IntegrityPolicy, ModelId};

/// Transport and artifact selection shared by model-specific runtime adapters.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeConfig {
    pub model: ModelId,
    pub artifact_profile: ArtifactProfileId,
    pub artifact_source: ArtifactSource,
    pub integrity: IntegrityPolicy,
    pub cache: ArtifactCachePolicy,
}
