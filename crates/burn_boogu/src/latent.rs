use burn::prelude::*;

use crate::BooguError;

/// Validate an output image size for the VAE and denoiser patching contract.
pub fn validate_image_size(height: usize, width: usize) -> Result<(), BooguError> {
    if height == 0 || width == 0 {
        return Err(BooguError::InvalidShape(
            "image dimensions must be non-zero".into(),
        ));
    }
    if !height.is_multiple_of(16) || !width.is_multiple_of(16) {
        return Err(BooguError::InvalidShape(
            "Boogu image dimensions must be divisible by 16".into(),
        ));
    }
    Ok(())
}

/// Patch BCHW latents into `[batch, tokens, patch*patch*channels]`.
pub fn patchify<B: Backend>(
    latent: Tensor<B, 4>,
    patch: usize,
) -> Result<Tensor<B, 3>, BooguError> {
    let [batch, channels, height, width] = latent.dims();
    if patch == 0 || !height.is_multiple_of(patch) || !width.is_multiple_of(patch) {
        return Err(BooguError::InvalidShape(format!(
            "latent {height}x{width} is not divisible by patch {patch}"
        )));
    }
    let rows = height / patch;
    let columns = width / patch;
    Ok(latent
        .reshape([batch, channels, rows, patch, columns, patch])
        .permute([0, 2, 4, 3, 5, 1])
        .reshape([batch, rows * columns, patch * patch * channels]))
}

/// Restore patched tokens to BCHW latents.
pub fn unpatchify<B: Backend>(
    tokens: Tensor<B, 3>,
    channels: usize,
    rows: usize,
    columns: usize,
    patch: usize,
) -> Result<Tensor<B, 4>, BooguError> {
    let [batch, token_count, token_width] = tokens.dims();
    if token_count != rows * columns || token_width != patch * patch * channels {
        return Err(BooguError::InvalidShape(format!(
            "cannot unpatch [{batch}, {token_count}, {token_width}] as {rows}x{columns}, patch {patch}, channels {channels}"
        )));
    }
    Ok(tokens
        .reshape([batch, rows, columns, patch, patch, channels])
        .permute([0, 5, 1, 3, 2, 4])
        .reshape([batch, channels, rows * patch, columns * patch]))
}
