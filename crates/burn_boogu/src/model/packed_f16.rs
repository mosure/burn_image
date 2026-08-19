//! Exact device-side materialization of packed IEEE binary16 arenas.
//!
//! Browser WebGPU does not guarantee the optional `shader-f16` feature. This module therefore
//! stores two canonical binary16 bit patterns in each `u32` and widens them with integer
//! operations before reinterpreting the resulting IEEE binary32 bits. No floating-point
//! arithmetic participates in the conversion.

use core::borrow::Borrow;

use burn::tensor::{DType, Shape};
use burn_cubecl::{
    CubeRuntime, cubecl, ops::numeric::empty_device_contiguous_dtype, tensor::CubeTensor,
};
use cubecl::{calculate_cube_count_elemwise, prelude::*};

/// Largest raw or materialized arena accepted by this primitive.
///
/// The value is the measured browser binding/allocation ceiling used by the Boogu Turbo loader.
/// Keeping the check here makes it impossible for a source to accidentally turn a semantic stage
/// into one oversized WebGPU storage buffer.
pub const PACKED_F16_MAX_BUFFER_BYTES: u64 = 1_217_126_400;

/// Byte alignment required for a materialized F32 tensor view's WebGPU storage binding.
///
/// WebGPU's default minimum storage-buffer offset alignment is 256 bytes. CubeCL also suballocates
/// its arena handles on this boundary. Aligning every logical tensor start to this conservative
/// boundary makes a shared object arena usable without a follow-up device copy.
pub const PACKED_F16_F32_VIEW_ALIGNMENT_BYTES: u64 = 256;

/// F32 element alignment corresponding to [`PACKED_F16_F32_VIEW_ALIGNMENT_BYTES`].
pub const PACKED_F16_F32_VIEW_ALIGNMENT_ELEMENTS: usize =
    PACKED_F16_F32_VIEW_ALIGNMENT_BYTES as usize / core::mem::size_of::<f32>();

const U32_BYTES: u64 = core::mem::size_of::<u32>() as u64;
const F32_BYTES: u64 = core::mem::size_of::<f32>() as u64;

/// Validation failure for a packed-F16 arena or one of its materialized tensor views.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PackedF16Error {
    /// An object with no logical binary16 values was supplied.
    #[error("packed-F16 objects must contain at least one element")]
    EmptyObject,
    /// The raw arena did not use the required unsigned 32-bit storage type.
    #[error("packed-F16 raw arena must use U32 storage, found {found:?}")]
    RawDType {
        /// Actual raw-arena dtype.
        found: DType,
    },
    /// The raw arena was not a single contiguous word array.
    #[error("packed-F16 raw arena must be rank-one, found rank {found}")]
    RawRank {
        /// Actual rank.
        found: usize,
    },
    /// The raw arena's only dimension did not have unit stride.
    #[error("packed-F16 raw arena must have unit stride, found {found}")]
    RawStride {
        /// Actual stride in `u32` words.
        found: usize,
    },
    /// The raw word count did not exactly cover the declared number of binary16 values.
    #[error("packed-F16 raw arena word count mismatch: expected {expected}, found {found}")]
    RawWordCount {
        /// Required `ceil(f16_elements / 2)` word count.
        expected: usize,
        /// Actual word count.
        found: usize,
    },
    /// A byte-size calculation overflowed the host index width.
    #[error("packed-F16 arena byte-size calculation overflowed")]
    SizeOverflow,
    /// A raw or materialized arena exceeded the browser's per-buffer ceiling.
    #[error("packed-F16 {kind} arena requires {bytes} bytes, exceeding the {limit}-byte ceiling")]
    BufferLimit {
        /// Whether this was the raw or materialized arena.
        kind: &'static str,
        /// Required byte count.
        bytes: u64,
        /// Enforced byte ceiling.
        limit: u64,
    },
    /// CubeCL returned a logical allocation length different from the requested exact arena.
    #[error("packed-F16 {kind} allocation size mismatch: expected {expected} bytes, found {found}")]
    AllocationSize {
        /// Whether this was the raw or materialized arena.
        kind: &'static str,
        /// Exact requested bytes.
        expected: u64,
        /// Actual logical handle bytes.
        found: u64,
    },
    /// A tensor view extended beyond the materialized F32 arena.
    #[error("packed-F16 F32 view [{offset}, {end}) exceeds the {arena_elements}-element arena")]
    SliceOutOfBounds {
        /// View offset in F32 elements.
        offset: usize,
        /// Exclusive view end in F32 elements.
        end: usize,
        /// Logical arena length in F32 elements.
        arena_elements: usize,
    },
    /// A tensor view did not begin on a valid WebGPU storage-buffer binding boundary.
    #[error("packed-F16 F32 view offset {offset} is not aligned to {alignment_elements} elements")]
    SliceUnaligned {
        /// View offset in F32 elements.
        offset: usize,
        /// Required F32 element alignment.
        alignment_elements: usize,
    },
    /// A layout contained a parameter with no elements.
    #[error("packed-F16 tensor {index} must contain at least one element")]
    EmptyTensor {
        /// Zero-based tensor index in deterministic layout order.
        index: usize,
    },
}

/// Round an object-arena element offset up to a bindable materialized-F32 view boundary.
///
/// Artifact sources should call this before appending each tensor, and fill the skipped raw F16
/// elements with canonical positive-zero bits. Since the widening kernel preserves indices, the
/// same aligned offset is then valid in the materialized F32 arena.
pub fn align_packed_f16_f32_view_offset(offset: usize) -> Result<usize, PackedF16Error> {
    let alignment = PACKED_F16_F32_VIEW_ALIGNMENT_ELEMENTS;
    offset
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .ok_or(PackedF16Error::SizeOverflow)
}

/// One tensor's deterministic compact-to-padded arena mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedF16TensorLayout {
    compact_offset_elements: usize,
    offset_elements: usize,
    elements: usize,
}

impl PackedF16TensorLayout {
    /// Offset in the original compact sequence of verified F16 tensor payloads.
    pub const fn compact_offset_elements(&self) -> usize {
        self.compact_offset_elements
    }

    /// Aligned offset in both the padded raw-F16 and materialized-F32 arenas.
    pub const fn offset_elements(&self) -> usize {
        self.offset_elements
    }

    /// Number of logical values belonging to this tensor.
    pub const fn elements(&self) -> usize {
        self.elements
    }

    /// Exclusive end of this tensor in the padded arena.
    pub const fn end_elements(&self) -> usize {
        self.offset_elements + self.elements
    }

    /// Cumulative positive-zero padding elements preceding this tensor in the padded arena.
    pub const fn cumulative_padding_before_elements(&self) -> usize {
        self.offset_elements - self.compact_offset_elements
    }
}

/// Deterministic object-wide layout for compact verified F16 tensors in a padded device arena.
///
/// Input counts must already be in the source's authenticated deterministic order (the production
/// source sorts by exact target path). Each tensor starts at a 64-F32-element/256-byte boundary;
/// gaps are represented by canonical positive-zero F16 values. Tensor hashes continue to cover
/// only the original compact payloads, never these runtime padding bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedF16Layout {
    tensors: Vec<PackedF16TensorLayout>,
    compact_elements: usize,
    padded_elements: usize,
    packed_words: usize,
    raw_bytes: u64,
    f32_bytes: u64,
}

impl PackedF16Layout {
    /// Build the exact aligned layout for an object in deterministic tensor order.
    pub fn try_from_element_counts<I>(element_counts: I) -> Result<Self, PackedF16Error>
    where
        I: IntoIterator,
        I::Item: Borrow<usize>,
    {
        let mut tensors = Vec::new();
        let mut compact_elements = 0usize;
        let mut padded_elements = 0usize;

        for (index, elements) in element_counts.into_iter().enumerate() {
            let elements = *elements.borrow();
            if elements == 0 {
                return Err(PackedF16Error::EmptyTensor { index });
            }
            let offset_elements = align_packed_f16_f32_view_offset(padded_elements)?;
            tensors.push(PackedF16TensorLayout {
                compact_offset_elements: compact_elements,
                offset_elements,
                elements,
            });
            compact_elements = compact_elements
                .checked_add(elements)
                .ok_or(PackedF16Error::SizeOverflow)?;
            padded_elements = offset_elements
                .checked_add(elements)
                .ok_or(PackedF16Error::SizeOverflow)?;
        }
        if tensors.is_empty() {
            return Err(PackedF16Error::EmptyObject);
        }

        let packed_words = padded_elements
            .checked_add(1)
            .ok_or(PackedF16Error::SizeOverflow)?
            / 2;
        let raw_bytes = words_to_bytes(packed_words)?;
        let f32_bytes = elements_to_f32_bytes(padded_elements)?;
        check_buffer_limit("raw", raw_bytes)?;
        check_buffer_limit("materialized", f32_bytes)?;

        Ok(Self {
            tensors,
            compact_elements,
            padded_elements,
            packed_words,
            raw_bytes,
            f32_bytes,
        })
    }

    /// Per-tensor compact-to-padded mappings in deterministic input order.
    pub fn tensors(&self) -> &[PackedF16TensorLayout] {
        &self.tensors
    }

    /// Number of tensors in this object.
    pub const fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Number of authenticated F16 values before runtime alignment padding.
    pub const fn compact_elements(&self) -> usize {
        self.compact_elements
    }

    /// Number of logical values in both padded raw and materialized arenas.
    pub const fn padded_elements(&self) -> usize {
        self.padded_elements
    }

    /// Number of zero-valued F16 gaps visible to the object-wide widening dispatch.
    pub const fn padding_elements(&self) -> usize {
        self.padded_elements - self.compact_elements
    }

    /// Number of physical U32 words in the padded raw arena.
    pub const fn packed_words(&self) -> usize {
        self.packed_words
    }

    /// Exact physical padded raw-arena size in bytes.
    pub const fn raw_bytes(&self) -> u64 {
        self.raw_bytes
    }

    /// Exact materialized F32-arena size in bytes.
    pub const fn f32_bytes(&self) -> u64 {
        self.f32_bytes
    }
}

/// A verified raw device arena containing two canonical binary16 values per `u32` word.
///
/// Element `2 * i` occupies the low 16 bits of word `i`; element `2 * i + 1` occupies the high
/// 16 bits. This is also the word value obtained by interpreting four canonical little-endian F16
/// bytes as one little-endian `u32`.
#[derive(Debug, Clone)]
pub struct PackedF16Object<R: CubeRuntime> {
    raw: CubeTensor<R>,
    f16_elements: usize,
}

impl<R: CubeRuntime> PackedF16Object<R> {
    /// Validate and wrap a raw rank-one U32 arena.
    ///
    /// `f16_elements` is the padded logical length, normally
    /// [`PackedF16Layout::padded_elements`], and therefore includes zero-filled alignment gaps.
    pub fn try_new(raw: CubeTensor<R>, f16_elements: usize) -> Result<Self, PackedF16Error> {
        if f16_elements == 0 {
            return Err(PackedF16Error::EmptyObject);
        }
        if raw.dtype != DType::U32 {
            return Err(PackedF16Error::RawDType { found: raw.dtype });
        }
        let rank = raw.meta.rank();
        if rank != 1 {
            return Err(PackedF16Error::RawRank { found: rank });
        }
        let stride = raw.meta.strides()[0];
        if stride != 1 {
            return Err(PackedF16Error::RawStride { found: stride });
        }

        let expected_words = f16_elements
            .checked_add(1)
            .ok_or(PackedF16Error::SizeOverflow)?
            / 2;
        let found_words = raw.meta.num_elements();
        if found_words != expected_words {
            return Err(PackedF16Error::RawWordCount {
                expected: expected_words,
                found: found_words,
            });
        }

        let expected_raw_bytes = words_to_bytes(expected_words)?;
        check_buffer_limit("raw", expected_raw_bytes)?;
        let actual_raw_bytes = raw.handle.size_in_used();
        if actual_raw_bytes != expected_raw_bytes {
            return Err(PackedF16Error::AllocationSize {
                kind: "raw",
                expected: expected_raw_bytes,
                found: actual_raw_bytes,
            });
        }
        check_buffer_limit("materialized", elements_to_f32_bytes(f16_elements)?)?;

        Ok(Self { raw, f16_elements })
    }

    /// Borrow the rank-one U32 device arena.
    pub const fn raw(&self) -> &CubeTensor<R> {
        &self.raw
    }

    /// Padded number of logical binary16 values stored in this object.
    pub const fn f16_elements(&self) -> usize {
        self.f16_elements
    }

    /// Number of physical U32 words in the raw arena.
    pub const fn packed_words(&self) -> usize {
        self.f16_elements.div_ceil(2)
    }
}

/// One object-wide contiguous F32 arena produced by the widening kernel.
///
/// Tensor views returned by [`Self::slice`] clone the managed device handle. They therefore keep
/// the allocation alive even if this owner is dropped. The caller must still synchronize stage
/// execution before releasing the final views and requesting allocator cleanup.
#[derive(Debug, Clone)]
pub struct MaterializedF32Object<R: CubeRuntime> {
    arena: CubeTensor<R>,
    f32_elements: usize,
}

impl<R: CubeRuntime> MaterializedF32Object<R> {
    /// Borrow the full rank-one contiguous F32 arena.
    pub const fn arena(&self) -> &CubeTensor<R> {
        &self.arena
    }

    /// Consume the owner and return the full rank-one contiguous F32 arena.
    pub fn into_arena(self) -> CubeTensor<R> {
        self.arena
    }

    /// Number of F32 elements in the materialized arena.
    pub const fn f32_elements(&self) -> usize {
        self.f32_elements
    }

    /// Create a shaped, contiguous F32 view into this object's materialized arena.
    ///
    /// `offset_elements` and the returned bounds are expressed in F32 elements, not bytes.
    pub fn slice(
        &self,
        offset_elements: usize,
        shape: Shape,
    ) -> Result<CubeTensor<R>, PackedF16Error> {
        if !offset_elements.is_multiple_of(PACKED_F16_F32_VIEW_ALIGNMENT_ELEMENTS) {
            return Err(PackedF16Error::SliceUnaligned {
                offset: offset_elements,
                alignment_elements: PACKED_F16_F32_VIEW_ALIGNMENT_ELEMENTS,
            });
        }
        let view_elements = checked_shape_elements(&shape)?;
        let end = offset_elements
            .checked_add(view_elements)
            .ok_or(PackedF16Error::SizeOverflow)?;
        if end > self.f32_elements {
            return Err(PackedF16Error::SliceOutOfBounds {
                offset: offset_elements,
                end,
                arena_elements: self.f32_elements,
            });
        }

        let start_bytes = elements_to_f32_bytes(offset_elements)?;
        let view_bytes = elements_to_f32_bytes(view_elements)?;
        let arena_bytes = elements_to_f32_bytes(self.f32_elements)?;
        let trailing_bytes = arena_bytes
            .checked_sub(start_bytes)
            .and_then(|remaining| remaining.checked_sub(view_bytes))
            .ok_or(PackedF16Error::SizeOverflow)?;
        let handle = self
            .arena
            .handle
            .clone()
            .offset_start(start_bytes)
            .offset_end(trailing_bytes);

        Ok(CubeTensor::new_contiguous(
            self.arena.client.clone(),
            self.arena.device.clone(),
            shape,
            handle,
            DType::F32,
        ))
    }
}

/// Materialize one packed object into a contiguous F32 device arena.
///
/// The dispatch is queued on the object's CubeCL client and does not synchronize. It uses one raw
/// storage binding and one output storage binding, and allocates exactly four logical bytes per
/// F32 value.
pub fn materialize_packed_f16_object<R: CubeRuntime>(
    object: &PackedF16Object<R>,
) -> Result<MaterializedF32Object<R>, PackedF16Error> {
    let output = empty_device_contiguous_dtype::<R>(
        object.raw.client.clone(),
        object.raw.device.clone(),
        Shape::new([object.f16_elements]),
        DType::F32,
    );
    let expected_output_bytes = elements_to_f32_bytes(object.f16_elements)?;
    let actual_output_bytes = output.handle.size_in_used();
    if actual_output_bytes != expected_output_bytes {
        return Err(PackedF16Error::AllocationSize {
            kind: "materialized",
            expected: expected_output_bytes,
            found: actual_output_bytes,
        });
    }

    let cube_dim = CubeDim::new(&object.raw.client, object.packed_words());
    let cube_count =
        calculate_cube_count_elemwise(&object.raw.client, object.packed_words(), cube_dim);
    unpack_packed_f16_kernel::launch(
        &object.raw.client,
        cube_count,
        cube_dim,
        object.raw.clone().into_tensor_arg(),
        output.clone().into_tensor_arg(),
    );

    Ok(MaterializedF32Object {
        arena: output,
        f32_elements: object.f16_elements,
    })
}

/// Materialize a batch of independently bounded packed objects.
///
/// One output arena and one dispatch are created per logical Burnpack object. This preserves
/// semantic object boundaries, ensuring no combined stage buffer can cross the WebGPU binding
/// ceiling.
pub fn materialize_packed_f16_objects<R: CubeRuntime>(
    objects: &[PackedF16Object<R>],
) -> Result<Vec<MaterializedF32Object<R>>, PackedF16Error> {
    objects.iter().map(materialize_packed_f16_object).collect()
}

#[cube]
fn widen_f16_bits_to_f32(bits: u32) -> f32 {
    let sign = (bits & 0x8000u32) << 16u32;
    let exponent = (bits >> 10u32) & 0x1fu32;
    let mantissa = bits & 0x03ffu32;

    let widened_bits = if exponent == 0u32 {
        if mantissa == 0u32 {
            sign
        } else {
            let shift = mantissa.leading_zeros() - 21u32;
            let widened_exponent = 113u32 - shift;
            let widened_mantissa = ((mantissa << shift) & 0x03ffu32) << 13u32;
            sign | (widened_exponent << 23u32) | widened_mantissa
        }
    } else if exponent == 0x1fu32 {
        if mantissa == 0u32 {
            sign | 0x7f80_0000u32
        } else {
            // IEEE widening preserves the payload and quiets a signaling half NaN, matching the
            // software reference used during artifact conversion.
            sign | 0x7fc0_0000u32 | (mantissa << 13u32)
        }
    } else {
        sign | ((exponent + 112u32) << 23u32) | (mantissa << 13u32)
    };

    f32::reinterpret(widened_bits)
}

#[cube(launch)]
fn unpack_packed_f16_kernel(packed: &Tensor<u32>, output: &mut Tensor<f32>) {
    if ABSOLUTE_POS < packed.len() {
        let word = packed[ABSOLUTE_POS];
        let output_offset = ABSOLUTE_POS * 2;
        output[output_offset] = widen_f16_bits_to_f32(word & 0xffffu32);
        if output_offset + 1 < output.len() {
            output[output_offset + 1] = widen_f16_bits_to_f32(word >> 16u32);
        }
    }
}

fn checked_shape_elements(shape: &Shape) -> Result<usize, PackedF16Error> {
    shape
        .iter()
        .try_fold(1usize, |count, dim| count.checked_mul(*dim))
        .ok_or(PackedF16Error::SizeOverflow)
}

fn words_to_bytes(words: usize) -> Result<u64, PackedF16Error> {
    u64::try_from(words)
        .ok()
        .and_then(|words| words.checked_mul(U32_BYTES))
        .ok_or(PackedF16Error::SizeOverflow)
}

fn elements_to_f32_bytes(elements: usize) -> Result<u64, PackedF16Error> {
    u64::try_from(elements)
        .ok()
        .and_then(|elements| elements.checked_mul(F32_BYTES))
        .ok_or(PackedF16Error::SizeOverflow)
}

fn check_buffer_limit(kind: &'static str, bytes: u64) -> Result<(), PackedF16Error> {
    if bytes > PACKED_F16_MAX_BUFFER_BYTES {
        return Err(PackedF16Error::BufferLimit {
            kind,
            bytes,
            limit: PACKED_F16_MAX_BUFFER_BYTES,
        });
    }
    Ok(())
}

#[cfg(test)]
fn reference_f16_bits_to_f32_bits(bits: u16) -> u32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = u32::from(bits & 0x03ff);
    match (exponent, mantissa) {
        (0, 0) => sign,
        (0, mantissa) => {
            let shift = mantissa.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | (((mantissa << shift) & 0x03ff) << 13)
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, mantissa) => sign | 0x7fc0_0000 | (mantissa << 13),
        (exponent, mantissa) => sign | ((u32::from(exponent) + 112) << 23) | (mantissa << 13),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_widening_is_exhaustively_bit_exact_correctness() {
        for bits in 0u16..=u16::MAX {
            let expected = half::f16::from_bits(bits).to_f32_const().to_bits();
            let actual = reference_f16_bits_to_f32_bits(bits);
            assert_eq!(
                actual, expected,
                "binary16 bit pattern 0x{bits:04x} widened incorrectly"
            );
        }
    }

    #[test]
    fn f16_widening_covers_ieee_classes_correctness() {
        let cases = [
            0x0000u16, 0x8000, // signed zero
            0x0001, 0x03ff, 0x8001, // subnormal
            0x0400, 0x3c00, 0x7bff, 0xbc00, // normal
            0x7c00, 0xfc00, // infinity
            0x7c01, 0x7e00, 0x7fff, 0xfe01, // signaling/quiet NaN payloads
        ];
        for bits in cases {
            assert_eq!(
                reference_f16_bits_to_f32_bits(bits),
                half::f16::from_bits(bits).to_f32_const().to_bits()
            );
        }
    }

    #[test]
    fn packed_word_order_is_low_then_high_correctness() {
        let halves = [0x3c00u16, 0xc000, 0x0001, 0x7c00, 0x7e35];
        let words: Vec<u32> = halves
            .chunks(2)
            .map(|pair| u32::from(pair[0]) | (u32::from(*pair.get(1).unwrap_or(&0)) << 16))
            .collect();
        let unpacked: Vec<u32> = (0..halves.len())
            .map(|index| {
                let bits = ((words[index / 2] >> ((index % 2) * 16)) & 0xffff) as u16;
                reference_f16_bits_to_f32_bits(bits)
            })
            .collect();
        let expected: Vec<u32> = halves
            .iter()
            .map(|bits| half::f16::from_bits(*bits).to_f32_const().to_bits())
            .collect();
        assert_eq!(unpacked, expected);
    }

    #[test]
    fn buffer_ceiling_is_fail_closed_correctness() {
        let max_f32_elements = (PACKED_F16_MAX_BUFFER_BYTES / F32_BYTES) as usize;
        assert!(elements_to_f32_bytes(max_f32_elements).is_ok());
        let error = check_buffer_limit(
            "materialized",
            elements_to_f32_bytes(max_f32_elements + 1).unwrap(),
        )
        .unwrap_err();
        assert!(matches!(error, PackedF16Error::BufferLimit { .. }));
    }

    #[test]
    fn f32_view_offsets_are_conservatively_aligned_correctness() {
        assert_eq!(PACKED_F16_F32_VIEW_ALIGNMENT_ELEMENTS, 64);
        assert_eq!(align_packed_f16_f32_view_offset(0).unwrap(), 0);
        assert_eq!(align_packed_f16_f32_view_offset(1).unwrap(), 64);
        assert_eq!(align_packed_f16_f32_view_offset(63).unwrap(), 64);
        assert_eq!(align_packed_f16_f32_view_offset(64).unwrap(), 64);
        assert_eq!(align_packed_f16_f32_view_offset(120).unwrap(), 128);
        assert_eq!(align_packed_f16_f32_view_offset(3360).unwrap(), 3392);
    }

    #[test]
    fn object_layout_padding_and_counters_are_exact_correctness() {
        let counts = [120, 120, 3360];
        let layout = PackedF16Layout::try_from_element_counts(counts).unwrap();
        assert_eq!(layout.tensor_count(), 3);
        assert_eq!(
            layout.tensors(),
            &[
                PackedF16TensorLayout {
                    compact_offset_elements: 0,
                    offset_elements: 0,
                    elements: 120,
                },
                PackedF16TensorLayout {
                    compact_offset_elements: 120,
                    offset_elements: 128,
                    elements: 120,
                },
                PackedF16TensorLayout {
                    compact_offset_elements: 240,
                    offset_elements: 256,
                    elements: 3360,
                },
            ]
        );
        assert_eq!(layout.compact_elements(), 3600);
        assert_eq!(layout.padded_elements(), 3616);
        assert_eq!(layout.padding_elements(), 16);
        assert_eq!(layout.packed_words(), 1808);
        assert_eq!(layout.raw_bytes(), 7232);
        assert_eq!(layout.f32_bytes(), 14_464);
    }

    #[test]
    fn object_layout_counts_odd_tail_storage_exactly_correctness() {
        let layout = PackedF16Layout::try_from_element_counts([1]).unwrap();
        assert_eq!(layout.compact_elements(), 1);
        assert_eq!(layout.padded_elements(), 1);
        assert_eq!(layout.packed_words(), 1);
        assert_eq!(layout.raw_bytes(), 4);
        assert_eq!(layout.f32_bytes(), 4);
    }

    #[test]
    fn object_layout_rejects_empty_tensor_and_oversized_buffer_correctness() {
        assert!(matches!(
            PackedF16Layout::try_from_element_counts([1, 0]),
            Err(PackedF16Error::EmptyTensor { index: 1 })
        ));
        let max_f32_elements = (PACKED_F16_MAX_BUFFER_BYTES / F32_BYTES) as usize;
        assert!(PackedF16Layout::try_from_element_counts([max_f32_elements]).is_ok());
        assert!(matches!(
            PackedF16Layout::try_from_element_counts([max_f32_elements + 1]),
            Err(PackedF16Error::BufferLimit {
                kind: "materialized",
                ..
            })
        ));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    #[ignore = "opt-in WGPU device correctness test"]
    fn packed_f16_wgpu_materialization_correctness() {
        use burn::tensor::{Bytes, Tensor, TensorPrimitive};
        use burn_cubecl::CubeBackend;
        use burn_wgpu::{WgpuDevice, WgpuRuntime};

        type Backend = CubeBackend<WgpuRuntime, f32, i32, u32>;

        let halves = [
            0x0000u16, 0x8000, 0x0001, 0x03ff, 0x3c00, 0xbc00, 0x7bff, 0x7c00, 0xfc00, 0x7c01,
            0x7e35,
        ];
        let words: Vec<u32> = halves
            .chunks(2)
            .map(|pair| u32::from(pair[0]) | (u32::from(*pair.get(1).unwrap_or(&0)) << 16))
            .collect();
        let word_count = words.len();
        let device = WgpuDevice::DefaultDevice;
        let client = WgpuRuntime::client(&device);
        let allocation = client.create_tensor(
            Bytes::from_elems(words),
            Shape::new([word_count]),
            core::mem::size_of::<u32>(),
        );
        let raw = CubeTensor::new_contiguous(
            client,
            device,
            Shape::new([word_count]),
            allocation.memory,
            DType::U32,
        );
        let packed = PackedF16Object::try_new(raw, halves.len()).unwrap();
        let materialized = materialize_packed_f16_object(&packed).unwrap();
        let tensor =
            Tensor::<Backend, 1>::from_primitive(TensorPrimitive::Float(materialized.into_arena()));
        let actual = futures::executor::block_on(tensor.into_data_async())
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let expected: Vec<u32> = halves
            .iter()
            .map(|bits| half::f16::from_bits(*bits).to_f32_const().to_bits())
            .collect();
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    #[ignore = "opt-in WGPU packed-resident kernel correctness test"]
    fn packed_f16_wgpu_linear_embedding_and_convolution_correctness() {
        use burn::tensor::{DType, Int, Tensor, TensorData, TensorPrimitive, ops::ConvOptions};
        use burn_cubecl::{
            CubeBackend,
            kernel::packed_f16::{
                packed_f16_conv2d, packed_f16_rhs_matmul, packed_f16_select_rows,
            },
        };
        use burn_wgpu::{WgpuDevice, WgpuRuntime};

        type Backend = CubeBackend<WgpuRuntime, f32, i32, u32>;

        fn float_primitive<const D: usize>(
            tensor: Tensor<Backend, D>,
        ) -> burn_cubecl::tensor::CubeTensor<WgpuRuntime> {
            match tensor.into_primitive() {
                TensorPrimitive::Float(tensor) => tensor,
                TensorPrimitive::QFloat(_) => panic!("test tensor unexpectedly quantized"),
            }
        }

        fn f16(values: &[f32]) -> Vec<half::f16> {
            values.iter().copied().map(half::f16::from_f32).collect()
        }

        let device = WgpuDevice::DefaultDevice;

        // Exercise the exact Qwen load shape: checkpoint [out, in] becomes a zero-copy
        // non-contiguous [in, out] view before the fused matmul reads its F16 bytes.
        let lhs = float_primitive(Tensor::<Backend, 2>::from_data(
            TensorData::new(vec![1.0_f32, 2.0, 3.0, -1.0, 0.5, 2.0], [2, 3]),
            &device,
        ));
        let rhs = float_primitive(
            Tensor::<Backend, 2>::from_data(
                TensorData::new(
                    f16(&[
                        1.0, 0.0, 2.0, // output 0
                        0.0, 1.0, -1.0, // output 1
                        2.0, -1.0, 0.5, // output 2
                        -1.0, 2.0, 1.0, // output 3
                    ]),
                    [4, 3],
                ),
                (&device, DType::F16),
            )
            .transpose(),
        );
        let output = Tensor::<Backend, 2>::from_primitive(TensorPrimitive::Float(
            packed_f16_rhs_matmul(lhs, rhs),
        ));
        let actual = futures::executor::block_on(output.into_data_async())
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(actual, vec![7.0, -1.0, 1.5, 6.0, 3.0, -1.5, -1.5, 4.0]);

        let table = float_primitive(Tensor::<Backend, 2>::from_data(
            TensorData::new(
                f16(&[
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, -2.0, -3.0, 7.0, 8.0, 9.0,
                ]),
                [4, 3],
            ),
            (&device, DType::F16),
        ));
        let indices =
            Tensor::<Backend, 1, Int>::from_data(TensorData::new(vec![2_i32, 0_i32], [2]), &device)
                .into_primitive();
        let selected = Tensor::<Backend, 2>::from_primitive(TensorPrimitive::Float(
            packed_f16_select_rows(table, 0, indices),
        ));
        let actual = futures::executor::block_on(selected.into_data_async())
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(actual, vec![-1.0, -2.0, -3.0, 1.0, 2.0, 3.0]);

        let input = float_primitive(Tensor::<Backend, 4>::from_data(
            TensorData::new(
                vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [1, 1, 3, 3],
            ),
            &device,
        ));
        let weight = float_primitive(Tensor::<Backend, 4>::from_data(
            TensorData::new(
                f16(&[1.0, 0.0, 0.0, -1.0, 0.5, 0.5, 0.5, 0.5]),
                [2, 1, 2, 2],
            ),
            (&device, DType::F16),
        ));
        let bias = float_primitive(Tensor::<Backend, 1>::from_data(
            TensorData::new(vec![0.5_f32, -1.0], [2]),
            &device,
        ));
        let convolved =
            Tensor::<Backend, 4>::from_primitive(TensorPrimitive::Float(packed_f16_conv2d(
                input,
                weight,
                Some(bias),
                ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
            )));
        let actual = futures::executor::block_on(convolved.into_data_async())
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(actual, vec![-3.5, -3.5, -3.5, -3.5, 5.0, 7.0, 11.0, 13.0]);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    #[ignore = "opt-in representative WGPU packed-F16 matmul timing"]
    fn packed_f16_wgpu_matmul_throughput_smoke() {
        use std::time::Instant;

        use burn::tensor::{DType, Tensor, TensorData, TensorPrimitive, backend::Backend as _};
        use burn_cubecl::{
            CubeBackend, kernel::packed_f16::packed_f16_rhs_matmul, tensor::CubeTensor,
        };
        use burn_wgpu::{WgpuDevice, WgpuRuntime};

        type TestBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;

        fn primitive<const D: usize>(tensor: Tensor<TestBackend, D>) -> CubeTensor<WgpuRuntime> {
            match tensor.into_primitive() {
                TensorPrimitive::Float(tensor) => tensor,
                TensorPrimitive::QFloat(_) => panic!("benchmark tensor unexpectedly quantized"),
            }
        }

        const ROWS: usize = 4096;
        const INNER: usize = 4096;
        const COLS: usize = 4096;
        const ATTEMPTS: usize = 3;
        let device = WgpuDevice::DefaultDevice;
        let lhs = primitive(Tensor::<TestBackend, 2>::ones([ROWS, INNER], &device));
        let checkpoint = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(
                vec![half::f16::from_f32(1.0 / INNER as f32); INNER * COLS],
                [COLS, INNER],
            ),
            (&device, DType::F16),
        );
        let rhs = primitive(checkpoint.transpose());

        let warm = packed_f16_rhs_matmul(lhs.clone(), rhs.clone());
        TestBackend::sync(&device).unwrap();
        drop(warm);

        let mut fastest = std::time::Duration::MAX;
        let mut last = None;
        for _ in 0..ATTEMPTS {
            let started = Instant::now();
            let output = packed_f16_rhs_matmul(lhs.clone(), rhs.clone());
            TestBackend::sync(&device).unwrap();
            fastest = fastest.min(started.elapsed());
            last = Some(output);
        }
        let operations = 2.0 * ROWS as f64 * INNER as f64 * COLS as f64;
        let throughput_tflops = operations / fastest.as_secs_f64() / 1.0e12;
        eprintln!(
            "packed-F16 compatibility matmul {ROWS}x{INNER}x{COLS}: {:.3} ms, {:.3} TFLOP/s",
            fastest.as_secs_f64() * 1_000.0,
            throughput_tflops
        );
        assert!(throughput_tflops.is_finite() && throughput_tflops > 0.0);

        let output = Tensor::<TestBackend, 2>::from_primitive(TensorPrimitive::Float(
            last.expect("at least one timing attempt"),
        ));
        let sample = output
            .slice([0..1, 0..1])
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0];
        assert!((sample - 1.0).abs() <= 0.001, "unexpected sample {sample}");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    #[ignore = "opt-in representative WGPU packed-F16 convolution timing"]
    fn packed_f16_wgpu_conv2d_throughput_smoke() {
        use std::time::Instant;

        use burn::tensor::{
            DType, Tensor, TensorData, TensorPrimitive, backend::Backend as _, ops::ConvOptions,
        };
        use burn_cubecl::{CubeBackend, kernel::packed_f16::packed_f16_conv2d, tensor::CubeTensor};
        use burn_wgpu::{WgpuDevice, WgpuRuntime};

        type TestBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;

        fn primitive<const D: usize>(tensor: Tensor<TestBackend, D>) -> CubeTensor<WgpuRuntime> {
            match tensor.into_primitive() {
                TensorPrimitive::Float(tensor) => tensor,
                TensorPrimitive::QFloat(_) => panic!("benchmark tensor unexpectedly quantized"),
            }
        }

        const BATCH: usize = 1;
        const CHANNELS: usize = 512;
        const SPATIAL: usize = 64;
        const KERNEL: usize = 3;
        const ATTEMPTS: usize = 3;
        let device = WgpuDevice::DefaultDevice;
        let input = primitive(Tensor::<TestBackend, 4>::ones(
            [BATCH, CHANNELS, SPATIAL, SPATIAL],
            &device,
        ));
        let weight_scale = 1.0 / (CHANNELS * KERNEL * KERNEL) as f32;
        let weight = primitive(Tensor::<TestBackend, 4>::from_data(
            TensorData::new(
                vec![half::f16::from_f32(weight_scale); CHANNELS * CHANNELS * KERNEL * KERNEL],
                [CHANNELS, CHANNELS, KERNEL, KERNEL],
            ),
            (&device, DType::F16),
        ));
        let options = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);

        let warm = packed_f16_conv2d(input.clone(), weight.clone(), None, options.clone());
        TestBackend::sync(&device).unwrap();
        drop(warm);

        let mut fastest = std::time::Duration::MAX;
        let mut last = None;
        for _ in 0..ATTEMPTS {
            let started = Instant::now();
            let output = packed_f16_conv2d(input.clone(), weight.clone(), None, options.clone());
            TestBackend::sync(&device).unwrap();
            fastest = fastest.min(started.elapsed());
            last = Some(output);
        }
        let operations = 2.0
            * BATCH as f64
            * CHANNELS as f64
            * SPATIAL as f64
            * SPATIAL as f64
            * CHANNELS as f64
            * KERNEL as f64
            * KERNEL as f64;
        let throughput_tflops = operations / fastest.as_secs_f64() / 1.0e12;
        eprintln!(
            "packed-F16 compatibility conv2d {CHANNELS}x{CHANNELS}x{KERNEL}x{KERNEL} at {SPATIAL}x{SPATIAL}: {:.3} ms, {:.3} TFLOP/s",
            fastest.as_secs_f64() * 1_000.0,
            throughput_tflops
        );
        assert!(throughput_tflops.is_finite() && throughput_tflops > 0.0);

        let output = Tensor::<TestBackend, 4>::from_primitive(TensorPrimitive::Float(
            last.expect("at least one timing attempt"),
        ));
        let sample = output
            .slice([
                0..1,
                0..1,
                SPATIAL / 2..SPATIAL / 2 + 1,
                SPATIAL / 2..SPATIAL / 2 + 1,
            ])
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0];
        assert!((sample - 1.0).abs() <= 0.002, "unexpected sample {sample}");
    }
}
