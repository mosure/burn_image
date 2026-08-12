use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::config::{AutoencoderKlConfig, AutoencoderKlConfigError};

/// One expected parameter in both Burn and upstream Diffusers naming/layout conventions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorSpec {
    pub burn_name: String,
    pub diffusers_name: String,
    pub burn_shape: Vec<usize>,
    pub diffusers_shape: Vec<usize>,
}

/// Complete deterministic tensor inventory for an AutoencoderKL configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorInventory {
    pub tensors: Vec<TensorSpec>,
}

impl TensorInventory {
    pub fn from_config(config: &AutoencoderKlConfig) -> Result<Self, AutoencoderKlConfigError> {
        config.validate()?;
        let mut builder = InventoryBuilder::default();
        let first = config.block_out_channels[0];
        let last = *config
            .block_out_channels
            .last()
            .expect("validated block channels");

        builder.conv("encoder.conv_in", config.in_channels, first, 3);
        let mut input_channels = first;
        for (block, &output_channels) in config.block_out_channels.iter().enumerate() {
            for layer in 0..config.layers_per_block {
                let layer_in = if layer == 0 {
                    input_channels
                } else {
                    output_channels
                };
                builder.resnet(
                    &format!("encoder.down_blocks.{block}.resnets.{layer}"),
                    layer_in,
                    output_channels,
                );
            }
            if block + 1 != config.block_out_channels.len() {
                builder.conv(
                    &format!("encoder.down_blocks.{block}.downsamplers.0.conv"),
                    output_channels,
                    output_channels,
                    3,
                );
            }
            input_channels = output_channels;
        }
        builder.mid_block("encoder.mid_block", last, config.mid_block_add_attention);
        builder.norm("encoder.conv_norm_out", last);
        builder.conv("encoder.conv_out", last, config.moment_channels(), 3);

        builder.conv("decoder.conv_in", config.latent_channels, last, 3);
        builder.mid_block("decoder.mid_block", last, config.mid_block_add_attention);
        let reversed = config
            .block_out_channels
            .iter()
            .copied()
            .rev()
            .collect::<Vec<_>>();
        let mut previous_channels = reversed[0];
        for (block, output_channels) in reversed.iter().copied().enumerate() {
            for layer in 0..(config.layers_per_block + 1) {
                let layer_in = if layer == 0 {
                    previous_channels
                } else {
                    output_channels
                };
                builder.resnet(
                    &format!("decoder.up_blocks.{block}.resnets.{layer}"),
                    layer_in,
                    output_channels,
                );
            }
            if block + 1 != reversed.len() {
                builder.conv(
                    &format!("decoder.up_blocks.{block}.upsamplers.0.conv"),
                    output_channels,
                    output_channels,
                    3,
                );
            }
            previous_channels = output_channels;
        }
        builder.norm("decoder.conv_norm_out", first);
        builder.conv("decoder.conv_out", first, config.out_channels, 3);

        if config.use_quant_conv {
            builder.conv(
                "quant_conv",
                config.moment_channels(),
                config.moment_channels(),
                1,
            );
        }
        if config.use_post_quant_conv {
            builder.conv(
                "post_quant_conv",
                config.latent_channels,
                config.latent_channels,
                1,
            );
        }

        let inventory = Self {
            tensors: builder.tensors,
        };
        debug_assert!(inventory.has_unique_names());
        Ok(inventory)
    }

    pub fn burn_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().map(|tensor| tensor.burn_name.as_str())
    }

    pub fn diffusers_names(&self) -> impl Iterator<Item = &str> {
        self.tensors
            .iter()
            .map(|tensor| tensor.diffusers_name.as_str())
    }

    pub fn get_by_burn_name(&self, name: &str) -> Option<&TensorSpec> {
        self.tensors.iter().find(|tensor| tensor.burn_name == name)
    }

    pub fn get_by_diffusers_name(&self, name: &str) -> Option<&TensorSpec> {
        self.tensors
            .iter()
            .find(|tensor| tensor.diffusers_name == name)
    }

    pub fn has_unique_names(&self) -> bool {
        let burn = self.burn_names().collect::<BTreeSet<_>>();
        let diffusers = self.diffusers_names().collect::<BTreeSet<_>>();
        burn.len() == self.tensors.len() && diffusers.len() == self.tensors.len()
    }
}

#[derive(Default)]
struct InventoryBuilder {
    tensors: Vec<TensorSpec>,
}

impl InventoryBuilder {
    fn tensor(
        &mut self,
        burn_name: impl Into<String>,
        diffusers_name: impl Into<String>,
        burn_shape: impl Into<Vec<usize>>,
        diffusers_shape: impl Into<Vec<usize>>,
    ) {
        self.tensors.push(TensorSpec {
            burn_name: burn_name.into(),
            diffusers_name: diffusers_name.into(),
            burn_shape: burn_shape.into(),
            diffusers_shape: diffusers_shape.into(),
        });
    }

    fn conv(&mut self, prefix: &str, input: usize, output: usize, kernel: usize) {
        let shape = vec![output, input, kernel, kernel];
        self.tensor(
            format!("{prefix}.weight"),
            format!("{prefix}.weight"),
            shape.clone(),
            shape,
        );
        self.tensor(
            format!("{prefix}.bias"),
            format!("{prefix}.bias"),
            vec![output],
            vec![output],
        );
    }

    fn norm(&mut self, prefix: &str, channels: usize) {
        self.tensor(
            format!("{prefix}.gamma"),
            format!("{prefix}.weight"),
            vec![channels],
            vec![channels],
        );
        self.tensor(
            format!("{prefix}.beta"),
            format!("{prefix}.bias"),
            vec![channels],
            vec![channels],
        );
    }

    fn linear(&mut self, prefix: &str, input: usize, output: usize, diffusers_prefix: &str) {
        self.tensor(
            format!("{prefix}.weight"),
            format!("{diffusers_prefix}.weight"),
            vec![input, output],
            vec![output, input],
        );
        self.tensor(
            format!("{prefix}.bias"),
            format!("{diffusers_prefix}.bias"),
            vec![output],
            vec![output],
        );
    }

    fn resnet(&mut self, prefix: &str, input: usize, output: usize) {
        self.norm(&format!("{prefix}.norm1"), input);
        self.conv(&format!("{prefix}.conv1"), input, output, 3);
        self.norm(&format!("{prefix}.norm2"), output);
        self.conv(&format!("{prefix}.conv2"), output, output, 3);
        if input != output {
            self.conv(&format!("{prefix}.conv_shortcut"), input, output, 1);
        }
    }

    fn attention(&mut self, prefix: &str, channels: usize) {
        self.norm(&format!("{prefix}.group_norm"), channels);
        self.linear(
            &format!("{prefix}.to_q"),
            channels,
            channels,
            &format!("{prefix}.to_q"),
        );
        self.linear(
            &format!("{prefix}.to_k"),
            channels,
            channels,
            &format!("{prefix}.to_k"),
        );
        self.linear(
            &format!("{prefix}.to_v"),
            channels,
            channels,
            &format!("{prefix}.to_v"),
        );
        self.linear(
            &format!("{prefix}.to_out"),
            channels,
            channels,
            &format!("{prefix}.to_out.0"),
        );
    }

    fn mid_block(&mut self, prefix: &str, channels: usize, attention: bool) {
        self.resnet(&format!("{prefix}.resnets.0"), channels, channels);
        if attention {
            self.attention(&format!("{prefix}.attentions.0"), channels);
        }
        self.resnet(&format!("{prefix}.resnets.1"), channels, channels);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flux1_inventory_correctness() {
        let inventory =
            TensorInventory::from_config(&AutoencoderKlConfig::flux1()).expect("FLUX inventory");
        assert_eq!(inventory.tensors.len(), 244);
        assert!(inventory.has_unique_names());
        assert_eq!(
            inventory
                .get_by_diffusers_name("encoder.conv_in.weight")
                .expect("encoder input")
                .diffusers_shape,
            [128, 3, 3, 3]
        );
        assert_eq!(
            inventory
                .get_by_burn_name("encoder.mid_block.attentions.0.to_out.weight")
                .expect("attention output")
                .diffusers_name,
            "encoder.mid_block.attentions.0.to_out.0.weight"
        );
        assert!(inventory.get_by_burn_name("quant_conv.weight").is_none());
    }

    #[test]
    fn tiny_inventory_includes_optional_convs_correctness() {
        let inventory =
            TensorInventory::from_config(&AutoencoderKlConfig::tiny()).expect("tiny inventory");
        assert!(inventory.get_by_burn_name("quant_conv.weight").is_some());
        assert!(
            inventory
                .get_by_burn_name("post_quant_conv.weight")
                .is_some()
        );
    }
}
