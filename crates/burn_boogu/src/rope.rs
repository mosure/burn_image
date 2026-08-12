use serde::{Deserialize, Serialize};

use crate::BooguError;

/// Height and width of a latent map before denoiser patching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LatentSize {
    /// Latent height.
    pub height: usize,
    /// Latent width.
    pub width: usize,
}

/// Host-side three-axis position IDs used to select Boogu RoPE frequencies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PositionIds {
    /// Sequence-major `[axis0, axis1, axis2]` coordinates.
    pub values: Vec<[u32; 3]>,
    /// Number of text tokens.
    pub text_len: usize,
    /// Number of reference-image tokens.
    pub reference_len: usize,
    /// Number of generated-image tokens.
    pub generated_len: usize,
}

/// Construct exact upstream ordering: text, references, generated image.
pub fn position_ids(
    text_len: usize,
    references: &[LatentSize],
    generated: LatentSize,
    patch: usize,
) -> Result<PositionIds, BooguError> {
    if patch == 0 {
        return Err(BooguError::InvalidShape(
            "RoPE patch must be non-zero".into(),
        ));
    }
    let mut values = Vec::new();
    for index in 0..text_len {
        let index = u32::try_from(index)
            .map_err(|_| BooguError::InvalidShape("text position exceeds u32".into()))?;
        values.push([index, index, index]);
    }

    let mut axis_zero = text_len;
    let mut reference_len = 0;
    for reference in references {
        if !reference.height.is_multiple_of(patch) || !reference.width.is_multiple_of(patch) {
            return Err(BooguError::InvalidShape(
                "reference latent is not divisible by patch size".into(),
            ));
        }
        let rows = reference.height / patch;
        let columns = reference.width / patch;
        for row in 0..rows {
            for column in 0..columns {
                values.push([
                    u32::try_from(axis_zero).unwrap(),
                    u32::try_from(row).unwrap(),
                    u32::try_from(column).unwrap(),
                ]);
            }
        }
        reference_len += rows * columns;
        axis_zero += rows.max(columns);
    }

    if !generated.height.is_multiple_of(patch) || !generated.width.is_multiple_of(patch) {
        return Err(BooguError::InvalidShape(
            "generated latent is not divisible by patch size".into(),
        ));
    }
    let rows = generated.height / patch;
    let columns = generated.width / patch;
    for row in 0..rows {
        for column in 0..columns {
            values.push([
                u32::try_from(axis_zero).unwrap(),
                u32::try_from(row).unwrap(),
                u32::try_from(column).unwrap(),
            ]);
        }
    }

    Ok(PositionIds {
        values,
        text_len,
        reference_len,
        generated_len: rows * columns,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_reference_generated_order_reference() {
        let ids = position_ids(
            2,
            &[LatentSize {
                height: 4,
                width: 6,
            }],
            LatentSize {
                height: 4,
                width: 4,
            },
            2,
        )
        .unwrap();
        assert_eq!(ids.values[0], [0, 0, 0]);
        assert_eq!(ids.values[1], [1, 1, 1]);
        assert_eq!(ids.values[2], [2, 0, 0]);
        assert_eq!(ids.values[7], [2, 1, 2]);
        assert_eq!(ids.values[8], [5, 0, 0]);
        assert_eq!(ids.reference_len, 6);
        assert_eq!(ids.generated_len, 4);
    }
}
