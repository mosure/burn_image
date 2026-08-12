use std::collections::{BTreeMap, btree_map::Entry};

use crate::{ImageRequest, ModelDescriptor, ModelId, RuntimeError};

/// Deterministic registry of available model descriptors.
#[derive(Clone, Debug, Default)]
pub struct ModelRegistry {
    models: BTreeMap<ModelId, ModelDescriptor>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, descriptor: ModelDescriptor) -> Result<(), RuntimeError> {
        descriptor.validate()?;
        match self.models.entry(descriptor.id.clone()) {
            Entry::Vacant(entry) => {
                entry.insert(descriptor);
                Ok(())
            }
            Entry::Occupied(_) => Err(RuntimeError::DuplicateModel(descriptor.id)),
        }
    }

    pub fn descriptor(&self, model: &ModelId) -> Option<&ModelDescriptor> {
        self.models.get(model)
    }

    pub fn require(&self, model: &ModelId) -> Result<&ModelDescriptor, RuntimeError> {
        self.descriptor(model)
            .ok_or_else(|| RuntimeError::UnknownModel(model.clone()))
    }

    pub fn validate_request(
        &self,
        model: &ModelId,
        request: &ImageRequest,
    ) -> Result<(), RuntimeError> {
        let descriptor = self.require(model)?;
        descriptor.capabilities.validate_request(model, request)
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&ModelId, &ModelDescriptor)> {
        self.models.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    pub fn len(&self) -> usize {
        self.models.len()
    }
}
