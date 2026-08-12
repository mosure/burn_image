//! Validating model and weight-inventory builder.

use burn::tensor::backend::Backend;

use crate::{
    Qwen3VlConfig, Qwen3VlForConditionalGeneration, Qwen3VlModel, Result, weights::WeightInventory,
};

/// A single source of truth for validation, module construction, and checkpoint tensor inventory.
#[derive(Debug, Clone)]
pub struct Qwen3VlBuilder {
    config: Qwen3VlConfig,
}

impl Qwen3VlBuilder {
    pub fn new(config: Qwen3VlConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    pub fn from_json(json: &str) -> Result<Self> {
        Self::new(Qwen3VlConfig::from_json(json)?)
    }

    pub fn config(&self) -> &Qwen3VlConfig {
        &self.config
    }

    pub fn base_inventory(&self) -> WeightInventory {
        WeightInventory::for_base_model(&self.config)
    }

    pub fn causal_lm_inventory(&self) -> WeightInventory {
        WeightInventory::for_config(&self.config, true)
    }

    pub fn build_base<B: Backend>(&self, device: &B::Device) -> Result<Qwen3VlModel<B>> {
        Qwen3VlModel::new(self.config.clone(), device)
    }

    pub fn build_causal_lm<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<Qwen3VlForConditionalGeneration<B>> {
        Qwen3VlForConditionalGeneration::new(self.config.clone(), device)
    }
}
