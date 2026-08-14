use alloc::vec::Vec;
use core::ops::Range;

/// Stay below Dawn's large-write shared-memory path when Rust is targeting
/// the browser WebGPU backend. Two MiB is aligned for WebGPU buffer copies and
/// leaves ample headroom below the four-MiB transport threshold used by both
/// `writeBuffer` and `writeTexture`.
pub(crate) const WRITE_BUFFER_CHUNK_BYTES: usize = 2 * 1024 * 1024;
pub(crate) const WRITE_TEXTURE_CHUNK_BYTES: usize = WRITE_BUFFER_CHUNK_BYTES;

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct WriteBufferChunk {
    pub(crate) buffer_offset: u64,
    pub(crate) data_range: Range<usize>,
}

pub(crate) fn write_buffer_chunks(
    buffer_offset: u64,
    data_len: usize,
) -> impl Iterator<Item = WriteBufferChunk> {
    (0..data_len)
        .step_by(WRITE_BUFFER_CHUNK_BYTES)
        .map(move |start| {
            let end = start.saturating_add(WRITE_BUFFER_CHUNK_BYTES).min(data_len);
            let relative_offset = u64::try_from(start).expect("write-buffer offset exceeds u64");
            let buffer_offset = buffer_offset
                .checked_add(relative_offset)
                .expect("write-buffer destination offset overflowed u64");
            WriteBufferChunk {
                buffer_offset,
                data_range: start..end,
            }
        })
}

/// Geometry and source layout needed to split one valid `Queue::write_texture` call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct WriteTextureCopy {
    pub(crate) data_offset: u64,
    pub(crate) bytes_per_row: Option<u32>,
    pub(crate) rows_per_image: Option<u32>,
    pub(crate) block_size_bytes: u32,
    pub(crate) block_width_texels: u32,
    pub(crate) block_height_texels: u32,
    pub(crate) origin_y: u32,
    pub(crate) origin_z: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) depth_or_array_layers: u32,
}

/// One contiguous source slab and its corresponding texture destination.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct WriteTextureChunk {
    pub(crate) data_range: Range<usize>,
    pub(crate) origin_y: u32,
    pub(crate) origin_z: u32,
    pub(crate) height: u32,
}

/// Split a texture write into complete block-row slabs below Dawn's large-write threshold.
///
/// `None` preserves the original call for small, invalid, or unsupported layouts. The browser
/// backend has already relied on WebGPU validation for those calls, so this helper deliberately
/// does not turn malformed input into a different operation. Every returned slab has depth one,
/// keeps the original row stride, and resets the layout offset to zero after slicing the source.
pub(crate) fn write_texture_chunks(
    copy: WriteTextureCopy,
    data_len: usize,
) -> Option<Vec<WriteTextureChunk>> {
    write_texture_chunks_with_limit(copy, data_len, WRITE_TEXTURE_CHUNK_BYTES)
}

fn write_texture_chunks_with_limit(
    copy: WriteTextureCopy,
    data_len: usize,
    chunk_bytes: usize,
) -> Option<Vec<WriteTextureChunk>> {
    if data_len <= chunk_bytes
        || chunk_bytes == 0
        || copy.width == 0
        || copy.height == 0
        || copy.depth_or_array_layers == 0
        || copy.block_size_bytes == 0
        || copy.block_width_texels == 0
        || copy.block_height_texels == 0
    {
        return None;
    }

    let block_size_bytes = u64::from(copy.block_size_bytes);
    let block_width_texels = u64::from(copy.block_width_texels);
    let block_height_texels = u64::from(copy.block_height_texels);
    if u64::from(copy.origin_y) % block_height_texels != 0
        || copy.data_offset % block_size_bytes != 0
    {
        return None;
    }

    let width_blocks = u64::from(copy.width).div_ceil(block_width_texels);
    let height_blocks = u64::from(copy.height).div_ceil(block_height_texels);
    let row_bytes_dense = width_blocks.checked_mul(block_size_bytes)?;
    let row_stride_bytes = copy.bytes_per_row.map(u64::from).unwrap_or(row_bytes_dense);
    if row_stride_bytes < row_bytes_dense || row_stride_bytes % block_size_bytes != 0 {
        return None;
    }
    // WebGPU requires these explicit strides when the corresponding dimension has more than one
    // row/image. Falling back preserves the validation behavior of an invalid original call.
    if (height_blocks > 1 || copy.depth_or_array_layers > 1) && copy.bytes_per_row.is_none() {
        return None;
    }
    if copy.depth_or_array_layers > 1 && copy.rows_per_image.is_none() {
        return None;
    }

    let image_stride_rows = copy.rows_per_image.map(u64::from).unwrap_or(height_blocks);
    if image_stride_rows < height_blocks {
        return None;
    }
    let image_stride_bytes = row_stride_bytes.checked_mul(image_stride_rows)?;
    let image_bytes_dense = height_blocks
        .checked_sub(1)?
        .checked_mul(row_stride_bytes)?
        .checked_add(row_bytes_dense)?;
    let bytes_in_copy = u64::from(copy.depth_or_array_layers)
        .checked_sub(1)?
        .checked_mul(image_stride_bytes)?
        .checked_add(image_bytes_dense)?;
    let source_end = copy.data_offset.checked_add(bytes_in_copy)?;
    if source_end > u64::try_from(data_len).ok()? {
        return None;
    }

    let chunk_bytes = u64::try_from(chunk_bytes).ok()?;
    if row_bytes_dense > chunk_bytes {
        // A row cannot be split without changing the destination x coordinate and copy width.
        // Current WebGPU texture-dimension limits keep every supported texel row below two MiB.
        return None;
    }
    let rows_per_chunk = 1_u64
        .checked_add(chunk_bytes.checked_sub(row_bytes_dense)? / row_stride_bytes)?
        .min(height_blocks);
    let chunk_capacity = u64::from(copy.depth_or_array_layers)
        .checked_mul(height_blocks.div_ceil(rows_per_chunk))?;
    let mut chunks = Vec::with_capacity(usize::try_from(chunk_capacity).ok()?);

    for layer in 0..u64::from(copy.depth_or_array_layers) {
        let layer_offset = copy
            .data_offset
            .checked_add(layer.checked_mul(image_stride_bytes)?)?;
        let mut block_row = 0_u64;
        while block_row < height_blocks {
            let rows = rows_per_chunk.min(height_blocks - block_row);
            let source_start =
                layer_offset.checked_add(block_row.checked_mul(row_stride_bytes)?)?;
            let source_bytes = rows
                .checked_sub(1)?
                .checked_mul(row_stride_bytes)?
                .checked_add(row_bytes_dense)?;
            let source_end = source_start.checked_add(source_bytes)?;
            let texel_row = block_row.checked_mul(block_height_texels)?;
            let height = rows
                .checked_mul(block_height_texels)?
                .min(u64::from(copy.height).checked_sub(texel_row)?);
            chunks.push(WriteTextureChunk {
                data_range: usize::try_from(source_start).ok()?
                    ..usize::try_from(source_end).ok()?,
                origin_y: copy.origin_y.checked_add(u32::try_from(texel_row).ok()?)?,
                origin_z: copy.origin_z.checked_add(u32::try_from(layer).ok()?)?,
                height: u32::try_from(height).ok()?,
            });
            block_row = block_row.checked_add(rows)?;
        }
    }

    (!chunks.is_empty()).then_some(chunks)
}

#[cfg(test)]
mod tests {
    use alloc::{vec, vec::Vec};

    use super::{
        write_buffer_chunks, write_texture_chunks, write_texture_chunks_with_limit,
        WriteBufferChunk, WriteTextureChunk, WriteTextureCopy, WRITE_BUFFER_CHUNK_BYTES,
        WRITE_TEXTURE_CHUNK_BYTES,
    };

    #[test]
    fn write_buffer_chunks_cover_source_and_advance_destination_correctness() {
        const START: u64 = 12;
        let data_len = WRITE_BUFFER_CHUNK_BYTES * 2 + 12;

        let chunks = write_buffer_chunks(START, data_len).collect::<Vec<_>>();

        assert_eq!(
            chunks,
            vec![
                WriteBufferChunk {
                    buffer_offset: START,
                    data_range: 0..WRITE_BUFFER_CHUNK_BYTES,
                },
                WriteBufferChunk {
                    buffer_offset: START + WRITE_BUFFER_CHUNK_BYTES as u64,
                    data_range: WRITE_BUFFER_CHUNK_BYTES..WRITE_BUFFER_CHUNK_BYTES * 2,
                },
                WriteBufferChunk {
                    buffer_offset: START + (WRITE_BUFFER_CHUNK_BYTES * 2) as u64,
                    data_range: WRITE_BUFFER_CHUNK_BYTES * 2..data_len,
                },
            ]
        );
        assert_eq!(
            chunks
                .iter()
                .map(|chunk| chunk.data_range.len())
                .sum::<usize>(),
            data_len
        );
        assert!(chunks
            .iter()
            .all(|chunk| chunk.data_range.len() <= WRITE_BUFFER_CHUNK_BYTES));
        assert!(chunks.iter().all(|chunk| {
            chunk.buffer_offset.is_multiple_of(4)
                && chunk.data_range.start.is_multiple_of(4)
                && chunk.data_range.len().is_multiple_of(4)
        }));
    }

    #[test]
    fn write_buffer_chunks_handle_empty_and_exact_boundary_correctness() {
        assert_eq!(write_buffer_chunks(0, 0).count(), 0);
        assert_eq!(
            write_buffer_chunks(0, WRITE_BUFFER_CHUNK_BYTES).collect::<Vec<_>>(),
            vec![WriteBufferChunk {
                buffer_offset: 0,
                data_range: 0..WRITE_BUFFER_CHUNK_BYTES,
            }]
        );
    }

    #[test]
    fn write_texture_chunks_split_exact_four_mib_rgba_image_correctness() {
        let chunks = write_texture_chunks(
            WriteTextureCopy {
                data_offset: 0,
                bytes_per_row: Some(4_096),
                rows_per_image: Some(1_024),
                block_size_bytes: 4,
                block_width_texels: 1,
                block_height_texels: 1,
                origin_y: 7,
                origin_z: 3,
                width: 1_024,
                height: 1_024,
                depth_or_array_layers: 1,
            },
            4 * 1024 * 1024,
        )
        .expect("the exact Dawn large-write boundary must be split");

        assert_eq!(
            chunks,
            vec![
                WriteTextureChunk {
                    data_range: 0..WRITE_TEXTURE_CHUNK_BYTES,
                    origin_y: 7,
                    origin_z: 3,
                    height: 512,
                },
                WriteTextureChunk {
                    data_range: WRITE_TEXTURE_CHUNK_BYTES..4 * 1024 * 1024,
                    origin_y: 519,
                    origin_z: 3,
                    height: 512,
                },
            ]
        );
        assert!(chunks.iter().all(|chunk| {
            !chunk.data_range.is_empty()
                && chunk.data_range.len() <= WRITE_TEXTURE_CHUNK_BYTES
                && chunk.data_range.len() < 4 * 1024 * 1024
        }));
    }

    #[test]
    fn write_texture_chunks_preserve_offset_padding_and_array_layers_correctness() {
        let chunks = write_texture_chunks_with_limit(
            WriteTextureCopy {
                data_offset: 8,
                bytes_per_row: Some(16),
                rows_per_image: Some(4),
                block_size_bytes: 4,
                block_width_texels: 1,
                block_height_texels: 1,
                origin_y: 5,
                origin_z: 7,
                width: 3,
                height: 2,
                depth_or_array_layers: 2,
            },
            WRITE_TEXTURE_CHUNK_BYTES + 1,
            20,
        )
        .expect("padded array texture should split by block row");

        assert_eq!(
            chunks,
            vec![
                WriteTextureChunk {
                    data_range: 8..20,
                    origin_y: 5,
                    origin_z: 7,
                    height: 1,
                },
                WriteTextureChunk {
                    data_range: 24..36,
                    origin_y: 6,
                    origin_z: 7,
                    height: 1,
                },
                WriteTextureChunk {
                    data_range: 72..84,
                    origin_y: 5,
                    origin_z: 8,
                    height: 1,
                },
                WriteTextureChunk {
                    data_range: 88..100,
                    origin_y: 6,
                    origin_z: 8,
                    height: 1,
                },
            ]
        );
    }

    #[test]
    fn write_texture_chunks_preserve_partial_compressed_block_extent_correctness() {
        let chunks = write_texture_chunks_with_limit(
            WriteTextureCopy {
                data_offset: 0,
                bytes_per_row: Some(48),
                rows_per_image: Some(2),
                block_size_bytes: 16,
                block_width_texels: 4,
                block_height_texels: 4,
                origin_y: 8,
                origin_z: 0,
                width: 6,
                height: 6,
                depth_or_array_layers: 1,
            },
            WRITE_TEXTURE_CHUNK_BYTES + 1,
            48,
        )
        .expect("compressed rows should remain complete blocks");

        assert_eq!(
            chunks,
            vec![
                WriteTextureChunk {
                    data_range: 0..32,
                    origin_y: 8,
                    origin_z: 0,
                    height: 4,
                },
                WriteTextureChunk {
                    data_range: 48..80,
                    origin_y: 12,
                    origin_z: 0,
                    height: 2,
                },
            ]
        );
    }

    #[test]
    fn write_texture_chunks_leave_invalid_or_unsplittable_calls_unchanged_correctness() {
        let base = WriteTextureCopy {
            data_offset: 0,
            bytes_per_row: Some(16),
            rows_per_image: Some(2),
            block_size_bytes: 4,
            block_width_texels: 1,
            block_height_texels: 1,
            origin_y: 0,
            origin_z: 0,
            width: 4,
            height: 2,
            depth_or_array_layers: 1,
        };
        assert!(write_texture_chunks_with_limit(base, 64, 8).is_none());
        assert!(write_texture_chunks_with_limit(
            WriteTextureCopy {
                bytes_per_row: None,
                ..base
            },
            64,
            32,
        )
        .is_none());
        assert!(write_texture_chunks_with_limit(
            WriteTextureCopy {
                data_offset: 2,
                ..base
            },
            64,
            32,
        )
        .is_none());
    }
}
