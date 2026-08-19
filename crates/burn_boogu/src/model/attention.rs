use burn::{
    nn,
    prelude::*,
    tensor::{module::attention, ops::AttentionModuleOptions},
};

use super::{
    linear::linear_forward,
    norm::{DenoiserRmsNormPolicy, rms_norm_with_policy},
};

#[cfg(all(feature = "cuda-experimental", not(target_arch = "wasm32")))]
use super::native_flash::{
    NativeCudaBackend, required_chunked_cuda_flash_unit_attention,
    required_chunked_cuda_padded_blackbox_attention_partitioned,
};
#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
use super::native_flash::{
    NativeWgpuBackend, required_chunked_flash_unit_attention,
    required_chunked_gqa_wgpu_balanced_strict_qk_norm_rope_padded_blackbox_attention,
    required_chunked_gqa_wgpu_fused_rope_padded_blackbox_attention,
    required_chunked_gqa_wgpu_fused_strict_qk_norm_rope_padded_blackbox_attention,
    required_chunked_gqa_wgpu_padded_blackbox_attention_partitioned,
    required_chunked_wgpu_padded_blackbox_attention_partitioned,
};

/// Bounds every fallback score tensor while retaining the complete key/value context.
const DEFAULT_QUERY_CHUNK_SIZE: usize = 128;
/// Minimum query partitions retained by portable attention for sequences longer than 128 rows.
pub const PORTABLE_ATTENTION_MINIMUM_IMAGE_QUERY_PARTITIONS: usize = 4;

/// Cap a requested portable-attention tile so an image-scale query never becomes dense.
///
/// The short instruction refiners are intentionally exempt: their released sequence is only 45
/// rows, so splitting them adds submissions without providing a meaningful memory bound. Above
/// the default 128-row tile, every query is divided into at least four partitions. This lets the
/// browser amortize WebGPU dispatch overhead with a large requested tile while preserving the
/// production ban on a full image-sequence-squared score tensor at smaller output sizes.
fn effective_portable_query_chunk_size(query_len: usize, requested: usize) -> usize {
    assert!(requested > 0, "attention query chunk must be non-zero");
    assert!(query_len > 0, "attention query sequence must be non-empty");
    if query_len <= DEFAULT_QUERY_CHUNK_SIZE {
        return requested.min(query_len);
    }

    let image_scale_cap = query_len.div_ceil(PORTABLE_ATTENTION_MINIMUM_IMAGE_QUERY_PARTITIONS);
    requested.min(image_scale_cap.max(DEFAULT_QUERY_CHUNK_SIZE))
}

pub(crate) trait AttentionKernel<B: Backend> {
    /// Whether dual-stream attention should apply its final shared token-wise projection to each
    /// stream separately instead of concatenating, projecting, and narrowing the streams.
    ///
    /// The conservative default retains the established graph. Native diagnostic wrappers may
    /// opt in because [`nn::Linear`] acts independently on the final feature dimension.
    const SPLIT_DOUBLE_STREAM_SHARED_PROJECTION: bool = false;

    fn execute(
        query: Tensor<B, 4>,
        key: Tensor<B, 4>,
        value: Tensor<B, 4>,
        query_chunk_size: usize,
    ) -> Tensor<B, 4>;

    /// Execute grouped-query attention from token-major, unexpanded Q/K/V tensors.
    ///
    /// The default retains the portable path's established materialized head expansion. Native
    /// kernels may override this seam when they can expand grouped key/value heads as part of a
    /// backend-specific preparation operation.
    fn execute_gqa(
        query: Tensor<B, 4>,
        key: Tensor<B, 4>,
        value: Tensor<B, 4>,
        query_chunk_size: usize,
    ) -> Tensor<B, 4> {
        let [batch, _query_len, query_heads, head_dim] = query.dims();
        let [key_batch, key_len, key_value_heads, key_head_dim] = key.dims();
        let [value_batch, value_len, value_heads, value_dim] = value.dims();
        assert_eq!(key_batch, batch, "GQA query/key batch mismatch");
        assert_eq!(value_batch, batch, "GQA query/value batch mismatch");
        assert_eq!(value_len, key_len, "GQA key/value length mismatch");
        assert_eq!(value_heads, key_value_heads, "GQA key/value head mismatch");
        assert_eq!(key_head_dim, head_dim, "GQA query/key width mismatch");
        assert_eq!(value_dim, head_dim, "GQA query/value width mismatch");
        assert_eq!(
            query_heads % key_value_heads,
            0,
            "GQA query heads must be divisible by key/value heads"
        );

        let groups = query_heads / key_value_heads;
        let key = key
            .reshape([batch, key_len, key_value_heads, 1, head_dim])
            .repeat_dim(3, groups)
            .reshape([batch, key_len, query_heads, head_dim]);
        let value = value
            .reshape([batch, key_len, key_value_heads, 1, head_dim])
            .repeat_dim(3, groups)
            .reshape([batch, key_len, query_heads, head_dim]);

        Self::execute(
            query.permute([0, 2, 1, 3]),
            key.permute([0, 2, 1, 3]),
            value.permute([0, 2, 1, 3]),
            query_chunk_size,
        )
        .permute([0, 2, 1, 3])
    }

    /// Normalize and rotate token-major, unexpanded Q/K before grouped-query attention.
    ///
    /// Keeping this preparation on the kernel seam lets a native implementation combine Q/K
    /// normalization, RoPE, GQA head expansion, attention scaling, and head-width padding. The
    /// default is deliberately the established portable composition, so reusable and browser
    /// backends retain exactly the same graph.
    #[allow(clippy::too_many_arguments)]
    fn execute_gqa_with_qk_norm_rope(
        query: Tensor<B, 4>,
        key: Tensor<B, 4>,
        value: Tensor<B, 4>,
        norm_q: &nn::RmsNorm<B>,
        norm_k: &nn::RmsNorm<B>,
        rope: Option<(Tensor<B, 3>, Tensor<B, 3>)>,
        query_chunk_size: usize,
        rms_norm_policy: DenoiserRmsNormPolicy,
    ) -> Tensor<B, 4> {
        let query = rms_norm_with_policy(norm_q, query, rms_norm_policy);
        let key = rms_norm_with_policy(norm_k, key, rms_norm_policy);
        let (query, key) = match rope {
            Some((cos, sin)) => (
                apply_rope(query, cos.clone(), sin.clone()),
                apply_rope(key, cos, sin),
            ),
            None => (query, key),
        };
        Self::execute_gqa(query, key, value, query_chunk_size)
    }
}

/// Native policy wrapper that removes the dual-stream shared-projection cat/narrow round trip.
///
/// Every attention operation is delegated unchanged to `K`; only the token-wise projection after
/// joint attention observes this marker.
#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
pub(crate) struct SplitDoubleStreamSharedProjection<K>(core::marker::PhantomData<K>);

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
impl<B: Backend, K: AttentionKernel<B>> AttentionKernel<B>
    for SplitDoubleStreamSharedProjection<K>
{
    const SPLIT_DOUBLE_STREAM_SHARED_PROJECTION: bool = true;

    fn execute(
        query: Tensor<B, 4>,
        key: Tensor<B, 4>,
        value: Tensor<B, 4>,
        query_chunk_size: usize,
    ) -> Tensor<B, 4> {
        K::execute(query, key, value, query_chunk_size)
    }

    fn execute_gqa(
        query: Tensor<B, 4>,
        key: Tensor<B, 4>,
        value: Tensor<B, 4>,
        query_chunk_size: usize,
    ) -> Tensor<B, 4> {
        K::execute_gqa(query, key, value, query_chunk_size)
    }

    fn execute_gqa_with_qk_norm_rope(
        query: Tensor<B, 4>,
        key: Tensor<B, 4>,
        value: Tensor<B, 4>,
        norm_q: &nn::RmsNorm<B>,
        norm_k: &nn::RmsNorm<B>,
        rope: Option<(Tensor<B, 3>, Tensor<B, 3>)>,
        query_chunk_size: usize,
        rms_norm_policy: DenoiserRmsNormPolicy,
    ) -> Tensor<B, 4> {
        K::execute_gqa_with_qk_norm_rope(
            query,
            key,
            value,
            norm_q,
            norm_k,
            rope,
            query_chunk_size,
            rms_norm_policy,
        )
    }
}

pub(crate) struct PortableChunkedAttention;

impl<B: Backend> AttentionKernel<B> for PortableChunkedAttention {
    fn execute(
        query: Tensor<B, 4>,
        key: Tensor<B, 4>,
        value: Tensor<B, 4>,
        query_chunk_size: usize,
    ) -> Tensor<B, 4> {
        query_chunked_attention(query, key, value, query_chunk_size)
    }
}

#[cfg(all(
    any(feature = "wgpu", feature = "cuda-experimental"),
    not(target_arch = "wasm32")
))]
pub(crate) struct NativeFlashUnitAttention;

#[cfg(all(
    any(feature = "wgpu", feature = "cuda-experimental"),
    not(target_arch = "wasm32")
))]
pub(crate) struct NativePaddedBlackboxAttention<
    const NUM_PLANES: u8,
    const SEQ_KV_TILES: u8 = 1,
    const SEQ_Q_TILES: u8 = 1,
    // Opt-in candidate; release aliases leave this false until real-artifact parity succeeds.
    const FUSED_STRICT_QK_NORM_ROPE: bool = false,
    // Narrower opt-in candidate that preserves the stock strict RMSNorm graph.
    const FUSED_ROPE_GQA_PADDING: bool = false,
    // Balanced Q/K reductions feeding the already-validated narrow preparation fusion.
    const BALANCED_STRICT_QK_NORM_ROPE: bool = false,
>;

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
impl AttentionKernel<NativeWgpuBackend> for NativeFlashUnitAttention {
    fn execute(
        query: Tensor<NativeWgpuBackend, 4>,
        key: Tensor<NativeWgpuBackend, 4>,
        value: Tensor<NativeWgpuBackend, 4>,
        query_chunk_size: usize,
    ) -> Tensor<NativeWgpuBackend, 4> {
        required_chunked_flash_unit_attention(query, key, value, query_chunk_size)
    }
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
impl<
    const NUM_PLANES: u8,
    const SEQ_KV_TILES: u8,
    const SEQ_Q_TILES: u8,
    const FUSED_STRICT_QK_NORM_ROPE: bool,
    const FUSED_ROPE_GQA_PADDING: bool,
    const BALANCED_STRICT_QK_NORM_ROPE: bool,
> AttentionKernel<NativeWgpuBackend>
    for NativePaddedBlackboxAttention<
        NUM_PLANES,
        SEQ_KV_TILES,
        SEQ_Q_TILES,
        FUSED_STRICT_QK_NORM_ROPE,
        FUSED_ROPE_GQA_PADDING,
        BALANCED_STRICT_QK_NORM_ROPE,
    >
{
    fn execute(
        query: Tensor<NativeWgpuBackend, 4>,
        key: Tensor<NativeWgpuBackend, 4>,
        value: Tensor<NativeWgpuBackend, 4>,
        query_chunk_size: usize,
    ) -> Tensor<NativeWgpuBackend, 4> {
        required_chunked_wgpu_padded_blackbox_attention_partitioned(
            query,
            key,
            value,
            query_chunk_size,
            NUM_PLANES,
            SEQ_KV_TILES,
            SEQ_Q_TILES,
        )
    }

    fn execute_gqa(
        query: Tensor<NativeWgpuBackend, 4>,
        key: Tensor<NativeWgpuBackend, 4>,
        value: Tensor<NativeWgpuBackend, 4>,
        query_chunk_size: usize,
    ) -> Tensor<NativeWgpuBackend, 4> {
        required_chunked_gqa_wgpu_padded_blackbox_attention_partitioned(
            query.permute([0, 2, 1, 3]),
            key.permute([0, 2, 1, 3]),
            value.permute([0, 2, 1, 3]),
            query_chunk_size,
            NUM_PLANES,
            SEQ_KV_TILES,
            SEQ_Q_TILES,
        )
        .permute([0, 2, 1, 3])
    }

    fn execute_gqa_with_qk_norm_rope(
        query: Tensor<NativeWgpuBackend, 4>,
        key: Tensor<NativeWgpuBackend, 4>,
        value: Tensor<NativeWgpuBackend, 4>,
        norm_q: &nn::RmsNorm<NativeWgpuBackend>,
        norm_k: &nn::RmsNorm<NativeWgpuBackend>,
        rope: Option<(Tensor<NativeWgpuBackend, 3>, Tensor<NativeWgpuBackend, 3>)>,
        query_chunk_size: usize,
        rms_norm_policy: DenoiserRmsNormPolicy,
    ) -> Tensor<NativeWgpuBackend, 4> {
        assert!(
            u8::from(FUSED_STRICT_QK_NORM_ROPE)
                + u8::from(FUSED_ROPE_GQA_PADDING)
                + u8::from(BALANCED_STRICT_QK_NORM_ROPE)
                <= 1,
            "native Q/K preparation candidates are mutually exclusive"
        );
        if FUSED_STRICT_QK_NORM_ROPE {
            assert_eq!(
                (NUM_PLANES, SEQ_KV_TILES, SEQ_Q_TILES),
                (4, 1, 1),
                "fused strict Q/K norm+RoPE preparation is gated to native p4/kv1/q1"
            );
            assert_eq!(
                rms_norm_policy,
                DenoiserRmsNormPolicy::StrictF32,
                "fused strict Q/K norm+RoPE preparation requires StrictF32 RMSNorm"
            );
            let (cos, sin) = rope
                .expect("fused strict Q/K norm+RoPE preparation requires explicit RoPE tensors");
            return required_chunked_gqa_wgpu_fused_strict_qk_norm_rope_padded_blackbox_attention(
                query.permute([0, 2, 1, 3]),
                key.permute([0, 2, 1, 3]),
                value.permute([0, 2, 1, 3]),
                norm_q.gamma.val(),
                norm_k.gamma.val(),
                cos,
                sin,
                norm_q.epsilon,
                norm_k.epsilon,
                query_chunk_size,
            )
            .permute([0, 2, 1, 3]);
        }

        if BALANCED_STRICT_QK_NORM_ROPE {
            assert_eq!(
                (NUM_PLANES, SEQ_KV_TILES, SEQ_Q_TILES),
                (4, 1, 1),
                "balanced strict Q/K RMSNorm preparation is gated to native p4/kv1/q1"
            );
            assert_eq!(
                rms_norm_policy,
                DenoiserRmsNormPolicy::StrictF32,
                "balanced strict Q/K RMSNorm preparation requires StrictF32 RMSNorm"
            );
            let (cos, sin) = rope
                .expect("balanced strict Q/K RMSNorm preparation requires explicit RoPE tensors");
            return required_chunked_gqa_wgpu_balanced_strict_qk_norm_rope_padded_blackbox_attention(
                query.permute([0, 2, 1, 3]),
                key.permute([0, 2, 1, 3]),
                value.permute([0, 2, 1, 3]),
                norm_q.gamma.val(),
                norm_k.gamma.val(),
                cos,
                sin,
                norm_q.epsilon,
                norm_k.epsilon,
                query_chunk_size,
            )
            .permute([0, 2, 1, 3]);
        }

        let query = rms_norm_with_policy(norm_q, query, rms_norm_policy);
        let key = rms_norm_with_policy(norm_k, key, rms_norm_policy);
        if FUSED_ROPE_GQA_PADDING {
            assert_eq!(
                (NUM_PLANES, SEQ_KV_TILES, SEQ_Q_TILES),
                (4, 1, 1),
                "fused RoPE+GQA padding preparation is gated to native p4/kv1/q1"
            );
            assert_eq!(
                rms_norm_policy,
                DenoiserRmsNormPolicy::StrictF32,
                "fused RoPE+GQA padding preparation requires stock StrictF32 RMSNorm"
            );
            let (cos, sin) =
                rope.expect("fused RoPE+GQA padding preparation requires explicit RoPE tensors");
            return required_chunked_gqa_wgpu_fused_rope_padded_blackbox_attention(
                query.permute([0, 2, 1, 3]),
                key.permute([0, 2, 1, 3]),
                value.permute([0, 2, 1, 3]),
                cos,
                sin,
                query_chunk_size,
            )
            .permute([0, 2, 1, 3]);
        }
        let (query, key) = match rope {
            Some((cos, sin)) => (
                apply_rope(query, cos.clone(), sin.clone()),
                apply_rope(key, cos, sin),
            ),
            None => (query, key),
        };
        Self::execute_gqa(query, key, value, query_chunk_size)
    }
}

#[cfg(all(feature = "cuda-experimental", not(target_arch = "wasm32")))]
impl AttentionKernel<NativeCudaBackend> for NativeFlashUnitAttention {
    fn execute(
        query: Tensor<NativeCudaBackend, 4>,
        key: Tensor<NativeCudaBackend, 4>,
        value: Tensor<NativeCudaBackend, 4>,
        query_chunk_size: usize,
    ) -> Tensor<NativeCudaBackend, 4> {
        required_chunked_cuda_flash_unit_attention(query, key, value, query_chunk_size)
    }
}

#[cfg(all(feature = "cuda-experimental", not(target_arch = "wasm32")))]
impl<const NUM_PLANES: u8, const SEQ_KV_TILES: u8, const SEQ_Q_TILES: u8>
    AttentionKernel<NativeCudaBackend>
    for NativePaddedBlackboxAttention<NUM_PLANES, SEQ_KV_TILES, SEQ_Q_TILES, false, false, false>
{
    fn execute(
        query: Tensor<NativeCudaBackend, 4>,
        key: Tensor<NativeCudaBackend, 4>,
        value: Tensor<NativeCudaBackend, 4>,
        query_chunk_size: usize,
    ) -> Tensor<NativeCudaBackend, 4> {
        required_chunked_cuda_padded_blackbox_attention_partitioned(
            query,
            key,
            value,
            query_chunk_size,
            NUM_PLANES,
            SEQ_KV_TILES,
            SEQ_Q_TILES,
        )
    }
}

/// Multi-head attention with grouped key/value projections and Q/K RMSNorm.
#[derive(Module, Debug)]
pub struct GqaAttention<B: Backend> {
    /// Query projection.
    pub to_q: nn::Linear<B>,
    /// Key projection.
    pub to_k: nn::Linear<B>,
    /// Value projection.
    pub to_v: nn::Linear<B>,
    /// Output projection.
    pub to_out: nn::Linear<B>,
    /// Query RMSNorm.
    pub norm_q: nn::RmsNorm<B>,
    /// Key RMSNorm.
    pub norm_k: nn::RmsNorm<B>,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    query_chunk_size: usize,
}

/// Separate image/instruction projections used by Boogu's leading dual-stream blocks.
///
/// The upstream processor does not reuse the ordinary attention projections: it owns
/// eight independent matrices, concatenates the projected streams, attends jointly,
/// splits the result, applies stream-specific output matrices, and only then applies
/// the enclosing attention output projection. Keeping this topology explicit is
/// required both for checkpoint key compatibility and numerical parity.
#[derive(Module, Debug)]
pub struct DoubleStreamAttention<B: Backend> {
    /// Image query projection.
    pub img_to_q: nn::Linear<B>,
    /// Image key projection.
    pub img_to_k: nn::Linear<B>,
    /// Image value projection.
    pub img_to_v: nn::Linear<B>,
    /// Instruction query projection.
    pub instruct_to_q: nn::Linear<B>,
    /// Instruction key projection.
    pub instruct_to_k: nn::Linear<B>,
    /// Instruction value projection.
    pub instruct_to_v: nn::Linear<B>,
    /// Image-stream output projection.
    pub img_out: nn::Linear<B>,
    /// Instruction-stream output projection.
    pub instruct_out: nn::Linear<B>,
    /// Final shared output projection (`img_instruct_attn.to_out.0` upstream).
    pub to_out: nn::Linear<B>,
    /// Query RMS normalization shared after concatenation.
    pub norm_q: nn::RmsNorm<B>,
    /// Key RMS normalization shared after concatenation.
    pub norm_k: nn::RmsNorm<B>,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    query_chunk_size: usize,
}

impl<B: Backend> DoubleStreamAttention<B> {
    /// Create the exact bias-free dual-stream projection topology.
    pub fn new(
        width: usize,
        heads: usize,
        kv_heads: usize,
        epsilon: f64,
        device: &B::Device,
    ) -> Self {
        let head_dim = width / heads;
        let kv_width = kv_heads * head_dim;
        let no_bias = |input, output| {
            nn::LinearConfig::new(input, output)
                .with_bias(false)
                .init(device)
        };
        Self {
            img_to_q: no_bias(width, width),
            img_to_k: no_bias(width, kv_width),
            img_to_v: no_bias(width, kv_width),
            instruct_to_q: no_bias(width, width),
            instruct_to_k: no_bias(width, kv_width),
            instruct_to_v: no_bias(width, kv_width),
            img_out: no_bias(width, width),
            instruct_out: no_bias(width, width),
            to_out: no_bias(width, width),
            norm_q: nn::RmsNormConfig::new(head_dim)
                .with_epsilon(epsilon)
                .init(device),
            norm_k: nn::RmsNormConfig::new(head_dim)
                .with_epsilon(epsilon)
                .init(device),
            heads,
            kv_heads,
            head_dim,
            query_chunk_size: DEFAULT_QUERY_CHUNK_SIZE,
        }
    }

    /// Set the maximum number of query rows submitted to one attention operation.
    ///
    /// Smaller values retain a tighter fallback score-memory bound. Native WGPU callers may use
    /// larger values to amortize dispatch overhead when the backend's flash-attention kernel is
    /// available.
    pub fn set_query_chunk_size(&mut self, query_chunk_size: usize) {
        assert!(
            query_chunk_size > 0,
            "attention query chunk must be non-zero"
        );
        self.query_chunk_size = query_chunk_size;
    }

    /// Attend over `[instruction, image]` and return the two projected streams.
    ///
    /// This uniform-batch overload is deliberately allocation-light and is the exact
    /// path exercised by the batch-one parity harness. Variable-length batches are
    /// packed by the pipeline before entering this method.
    pub fn forward(
        &self,
        image: Tensor<B, 3>,
        instruction: Tensor<B, 3>,
        rope: (Tensor<B, 3>, Tensor<B, 3>),
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        self.forward_with_kernel::<PortableChunkedAttention>(image, instruction, rope)
    }

    pub(crate) fn forward_with_kernel<K: AttentionKernel<B>>(
        &self,
        image: Tensor<B, 3>,
        instruction: Tensor<B, 3>,
        rope: (Tensor<B, 3>, Tensor<B, 3>),
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        self.forward_with_kernel_and_rms_norm_policy::<K>(
            image,
            instruction,
            rope,
            DenoiserRmsNormPolicy::StrictF32,
        )
    }

    pub(crate) fn forward_with_kernel_and_rms_norm_policy<K: AttentionKernel<B>>(
        &self,
        image: Tensor<B, 3>,
        instruction: Tensor<B, 3>,
        rope: (Tensor<B, 3>, Tensor<B, 3>),
        rms_norm_policy: DenoiserRmsNormPolicy,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let [batch, image_len, _] = image.dims();
        let instruction_len = instruction.dims()[1];
        let query = Tensor::cat(
            vec![
                linear_forward(&self.instruct_to_q, instruction.clone()),
                linear_forward(&self.img_to_q, image.clone()),
            ],
            1,
        );
        let key = Tensor::cat(
            vec![
                linear_forward(&self.instruct_to_k, instruction.clone()),
                linear_forward(&self.img_to_k, image.clone()),
            ],
            1,
        );
        let value = Tensor::cat(
            vec![
                linear_forward(&self.instruct_to_v, instruction),
                linear_forward(&self.img_to_v, image),
            ],
            1,
        );
        let sequence = instruction_len + image_len;
        let query = query.reshape([batch, sequence, self.heads, self.head_dim]);
        let key = key.reshape([batch, sequence, self.kv_heads, self.head_dim]);
        let value = value.reshape([batch, sequence, self.kv_heads, self.head_dim]);
        let (cos, sin) = rope;
        let attended = K::execute_gqa_with_qk_norm_rope(
            query,
            key,
            value,
            &self.norm_q,
            &self.norm_k,
            Some((cos, sin)),
            self.query_chunk_size,
            rms_norm_policy,
        )
        .reshape([batch, sequence, self.heads * self.head_dim]);

        let instruction = linear_forward(
            &self.instruct_out,
            attended.clone().narrow(1, 0, instruction_len),
        );
        let image = linear_forward(
            &self.img_out,
            attended.narrow(1, instruction_len, image_len),
        );
        self.project_shared_output(instruction, image, K::SPLIT_DOUBLE_STREAM_SHARED_PROJECTION)
    }

    /// Apply the same shared feature projection either jointly or one stream at a time.
    ///
    /// `to_out` is a bias-free [`nn::Linear`] over the final feature dimension. It neither mixes
    /// tokens nor depends on sequence length, so splitting along the token axis is algebraically
    /// equivalent while avoiding the joint cat followed by two narrows.
    fn project_shared_output(
        &self,
        instruction: Tensor<B, 3>,
        image: Tensor<B, 3>,
        split_streams: bool,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        if split_streams {
            return (
                linear_forward(&self.to_out, image),
                linear_forward(&self.to_out, instruction),
            );
        }

        let instruction_len = instruction.dims()[1];
        let image_len = image.dims()[1];
        let merged = linear_forward(&self.to_out, Tensor::cat(vec![instruction, image], 1));
        (
            merged.clone().narrow(1, instruction_len, image_len),
            merged.narrow(1, 0, instruction_len),
        )
    }
}

impl<B: Backend> GqaAttention<B> {
    /// Create bias-free Boogu attention projections.
    pub fn new(
        width: usize,
        heads: usize,
        kv_heads: usize,
        epsilon: f64,
        device: &B::Device,
    ) -> Self {
        let head_dim = width / heads;
        let no_bias = |input, output| {
            nn::LinearConfig::new(input, output)
                .with_bias(false)
                .init(device)
        };
        Self {
            to_q: no_bias(width, width),
            to_k: no_bias(width, kv_heads * head_dim),
            to_v: no_bias(width, kv_heads * head_dim),
            to_out: no_bias(width, width),
            norm_q: nn::RmsNormConfig::new(head_dim)
                .with_epsilon(epsilon)
                .init(device),
            norm_k: nn::RmsNormConfig::new(head_dim)
                .with_epsilon(epsilon)
                .init(device),
            heads,
            kv_heads,
            head_dim,
            query_chunk_size: DEFAULT_QUERY_CHUNK_SIZE,
        }
    }

    /// Set the maximum number of query rows submitted to one attention operation.
    ///
    /// Smaller values retain a tighter fallback score-memory bound. Native WGPU callers may use
    /// larger values to amortize dispatch overhead when the backend's flash-attention kernel is
    /// available.
    pub fn set_query_chunk_size(&mut self, query_chunk_size: usize) {
        assert!(
            query_chunk_size > 0,
            "attention query chunk must be non-zero"
        );
        self.query_chunk_size = query_chunk_size;
    }

    /// Project tokens and execute attention. `rope` contains repeated real cos/sin values.
    pub fn forward(
        &self,
        query_tokens: Tensor<B, 3>,
        key_value_tokens: Tensor<B, 3>,
        rope: Option<(Tensor<B, 3>, Tensor<B, 3>)>,
    ) -> Tensor<B, 3> {
        self.forward_with_kernel::<PortableChunkedAttention>(query_tokens, key_value_tokens, rope)
    }

    pub(crate) fn forward_with_kernel<K: AttentionKernel<B>>(
        &self,
        query_tokens: Tensor<B, 3>,
        key_value_tokens: Tensor<B, 3>,
        rope: Option<(Tensor<B, 3>, Tensor<B, 3>)>,
    ) -> Tensor<B, 3> {
        self.forward_with_kernel_and_rms_norm_policy::<K>(
            query_tokens,
            key_value_tokens,
            rope,
            DenoiserRmsNormPolicy::StrictF32,
        )
    }

    pub(crate) fn forward_with_kernel_and_rms_norm_policy<K: AttentionKernel<B>>(
        &self,
        query_tokens: Tensor<B, 3>,
        key_value_tokens: Tensor<B, 3>,
        rope: Option<(Tensor<B, 3>, Tensor<B, 3>)>,
        rms_norm_policy: DenoiserRmsNormPolicy,
    ) -> Tensor<B, 3> {
        let [batch, query_len, _] = query_tokens.dims();
        let key_len = key_value_tokens.dims()[1];
        let query = linear_forward(&self.to_q, query_tokens).reshape([
            batch,
            query_len,
            self.heads,
            self.head_dim,
        ]);
        let key = linear_forward(&self.to_k, key_value_tokens.clone()).reshape([
            batch,
            key_len,
            self.kv_heads,
            self.head_dim,
        ]);
        let value = linear_forward(&self.to_v, key_value_tokens).reshape([
            batch,
            key_len,
            self.kv_heads,
            self.head_dim,
        ]);

        let output = K::execute_gqa_with_qk_norm_rope(
            query,
            key,
            value,
            &self.norm_q,
            &self.norm_k,
            rope,
            self.query_chunk_size,
            rms_norm_policy,
        )
        .reshape([batch, query_len, self.heads * self.head_dim]);
        linear_forward(&self.to_out, output)
    }
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
impl DoubleStreamAttention<NativeWgpuBackend> {
    /// Execute bounded-query native WGPU attention with required Cubek `FlashUnit` kernels.
    ///
    /// The configured query chunk bounds each forced-FlashUnit submission. The method fails closed
    /// if FlashUnit cannot be prepared or launched.
    pub fn forward_native_flash_unit(
        &self,
        image: Tensor<NativeWgpuBackend, 3>,
        instruction: Tensor<NativeWgpuBackend, 3>,
        rope: (Tensor<NativeWgpuBackend, 3>, Tensor<NativeWgpuBackend, 3>),
    ) -> (Tensor<NativeWgpuBackend, 3>, Tensor<NativeWgpuBackend, 3>) {
        self.forward_with_kernel::<NativeFlashUnitAttention>(image, instruction, rope)
    }
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
impl GqaAttention<NativeWgpuBackend> {
    /// Execute bounded-query native WGPU attention with required Cubek `FlashUnit` kernels.
    ///
    /// The configured query chunk bounds each forced-FlashUnit submission. The method fails closed
    /// if FlashUnit cannot be prepared or launched.
    pub fn forward_native_flash_unit(
        &self,
        query_tokens: Tensor<NativeWgpuBackend, 3>,
        key_value_tokens: Tensor<NativeWgpuBackend, 3>,
        rope: Option<(Tensor<NativeWgpuBackend, 3>, Tensor<NativeWgpuBackend, 3>)>,
    ) -> Tensor<NativeWgpuBackend, 3> {
        self.forward_with_kernel::<NativeFlashUnitAttention>(query_tokens, key_value_tokens, rope)
    }
}

#[cfg(all(feature = "cuda-experimental", not(target_arch = "wasm32")))]
impl DoubleStreamAttention<NativeCudaBackend> {
    /// Execute bounded-query native CUDA attention with required Cubek `FlashUnit` kernels.
    pub fn forward_native_cuda_flash_unit(
        &self,
        image: Tensor<NativeCudaBackend, 3>,
        instruction: Tensor<NativeCudaBackend, 3>,
        rope: (Tensor<NativeCudaBackend, 3>, Tensor<NativeCudaBackend, 3>),
    ) -> (Tensor<NativeCudaBackend, 3>, Tensor<NativeCudaBackend, 3>) {
        self.forward_with_kernel::<NativeFlashUnitAttention>(image, instruction, rope)
    }
}

#[cfg(all(feature = "cuda-experimental", not(target_arch = "wasm32")))]
impl GqaAttention<NativeCudaBackend> {
    /// Execute bounded-query native CUDA attention with required Cubek `FlashUnit` kernels.
    pub fn forward_native_cuda_flash_unit(
        &self,
        query_tokens: Tensor<NativeCudaBackend, 3>,
        key_value_tokens: Tensor<NativeCudaBackend, 3>,
        rope: Option<(Tensor<NativeCudaBackend, 3>, Tensor<NativeCudaBackend, 3>)>,
    ) -> Tensor<NativeCudaBackend, 3> {
        self.forward_with_kernel::<NativeFlashUnitAttention>(query_tokens, key_value_tokens, rope)
    }
}

/// Exact unmasked attention evaluated in bounded query tiles.
///
/// Burn's attention operation preserves the input activation dtype and applies the default
/// `1 / sqrt(head_dim)` scaling. Calling that same operation once per query tile leaves every
/// row's key/value context and softmax unchanged while bounding fallback score storage to
/// `batch * heads * query_chunk_size * key_len`. Only the attended output tiles are retained for
/// the final concatenation.
fn query_chunked_attention<B: Backend>(
    query: Tensor<B, 4>,
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
    query_chunk_size: usize,
) -> Tensor<B, 4> {
    assert!(
        query_chunk_size > 0,
        "attention query chunk must be non-zero"
    );
    let [batch, heads, query_len, head_dim] = query.dims();
    let [key_batch, key_heads, key_len, key_head_dim] = key.dims();
    let [value_batch, value_heads, value_len, _value_dim] = value.dims();
    assert!(query_len > 0, "attention query sequence must be non-empty");
    assert!(key_len > 0, "attention key sequence must be non-empty");
    assert_eq!(key_batch, batch, "attention query/key batch mismatch");
    assert_eq!(value_batch, batch, "attention query/value batch mismatch");
    assert_eq!(key_heads, heads, "attention query/key head mismatch");
    assert_eq!(value_heads, heads, "attention query/value head mismatch");
    assert_eq!(key_head_dim, head_dim, "attention query/key width mismatch");
    assert_eq!(value_len, key_len, "attention key/value length mismatch");

    let query_chunk_size = effective_portable_query_chunk_size(query_len, query_chunk_size);
    let mut outputs = Vec::with_capacity(query_len.div_ceil(query_chunk_size));
    for start in (0..query_len).step_by(query_chunk_size) {
        let end = start.saturating_add(query_chunk_size).min(query_len);
        let query = query
            .clone()
            .slice([0..batch, 0..heads, start..end, 0..head_dim]);
        outputs.push(attention(
            query,
            key.clone(),
            value.clone(),
            None,
            None,
            AttentionModuleOptions::default(),
        ));
    }
    Tensor::cat(outputs, 2)
}

fn apply_rope<B: Backend>(x: Tensor<B, 4>, cos: Tensor<B, 3>, sin: Tensor<B, 3>) -> Tensor<B, 4> {
    let [batch, tokens, heads, head_dim] = x.dims();
    let pairs = head_dim / 2;
    let paired = x.clone().reshape([batch, tokens, heads, pairs, 2]);
    let real = paired
        .clone()
        .slice([0..batch, 0..tokens, 0..heads, 0..pairs, 0..1]);
    let imag = paired.slice([0..batch, 0..tokens, 0..heads, 0..pairs, 1..2]);
    let rotated = Tensor::cat(vec![imag.neg(), real], 4).reshape([batch, tokens, heads, head_dim]);
    let cos = cos.unsqueeze_dim(2);
    let sin = sin.unsqueeze_dim(2);
    x * cos + rotated * sin
}

#[cfg(test)]
mod tests {
    use core::sync::atomic::{AtomicUsize, Ordering};

    use burn::tensor::{DType, TensorData};
    use burn_ndarray::NdArray;

    use super::*;

    type TestBackend = NdArray<f32>;

    static OBSERVED_QUERY_CHUNK: AtomicUsize = AtomicUsize::new(0);

    struct ObserveQueryChunk;

    impl AttentionKernel<TestBackend> for ObserveQueryChunk {
        fn execute(
            query: Tensor<TestBackend, 4>,
            _key: Tensor<TestBackend, 4>,
            _value: Tensor<TestBackend, 4>,
            query_chunk_size: usize,
        ) -> Tensor<TestBackend, 4> {
            OBSERVED_QUERY_CHUNK.store(query_chunk_size, Ordering::Relaxed);
            query
        }
    }

    fn deterministic_tensor<B: Backend, const D: usize>(
        shape: [usize; D],
        offset: usize,
        device: &B::Device,
    ) -> Tensor<B, D> {
        let elements = shape.iter().product();
        let values = (0..elements)
            .map(|index| {
                let integer = ((index + offset) * 17 + 5) % 43;
                (integer as f32 - 21.0) / 13.0
            })
            .collect::<Vec<_>>();
        Tensor::from_data(TensorData::new(values, shape), device)
    }

    #[test]
    fn query_chunked_attention_matches_dense_reference_correctness() {
        let device = Default::default();
        let query = deterministic_tensor::<TestBackend, 4>([2, 3, 5, 4], 0, &device);
        let key = deterministic_tensor::<TestBackend, 4>([2, 3, 7, 4], 3, &device);
        let value = deterministic_tensor::<TestBackend, 4>([2, 3, 7, 6], 9, &device);
        let expected = attention(
            query.clone(),
            key.clone(),
            value.clone(),
            None,
            None,
            AttentionModuleOptions::default(),
        )
        .into_data()
        .to_vec::<f32>()
        .expect("dense attention values");
        let actual = query_chunked_attention(query, key, value, 2)
            .into_data()
            .to_vec::<f32>()
            .expect("chunked attention values");
        let max_abs = expected
            .iter()
            .zip(actual)
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_abs <= 1.0e-6, "attention chunking max_abs={max_abs}");
    }

    #[test]
    fn query_chunked_attention_preserves_shape_and_dtype_correctness() {
        let device = Default::default();
        let query = deterministic_tensor::<TestBackend, 4>([1, 2, 5, 4], 1, &device);
        let key = deterministic_tensor::<TestBackend, 4>([1, 2, 7, 4], 2, &device);
        let value = deterministic_tensor::<TestBackend, 4>([1, 2, 7, 3], 4, &device);
        let output = query_chunked_attention(query, key, value, 2);
        assert_eq!(output.dims(), [1, 2, 5, 3]);
        assert_eq!(output.dtype(), DType::F32);
        assert!(output.is_finite().all().into_scalar());
    }

    #[test]
    fn portable_query_chunk_keeps_image_attention_partitioned_correctness() {
        assert_eq!(effective_portable_query_chunk_size(45, 1_024), 45);
        assert_eq!(effective_portable_query_chunk_size(129, 1_024), 128);
        assert_eq!(effective_portable_query_chunk_size(1_024, 1_024), 256);
        assert_eq!(effective_portable_query_chunk_size(2_304, 1_024), 576);
        assert_eq!(effective_portable_query_chunk_size(4_096, 1_024), 1_024);
        assert_eq!(effective_portable_query_chunk_size(4_141, 1_024), 1_024);
        assert_eq!(effective_portable_query_chunk_size(9_261, 1_024), 1_024);
    }

    #[test]
    fn portable_gqa_kernel_seam_matches_materialized_reference_correctness() {
        let device = Default::default();
        let query = deterministic_tensor::<TestBackend, 4>([1, 5, 4, 4], 1, &device);
        let key = deterministic_tensor::<TestBackend, 4>([1, 7, 2, 4], 4, &device);
        let value = deterministic_tensor::<TestBackend, 4>([1, 7, 2, 4], 8, &device);
        let key_expanded = key
            .clone()
            .reshape([1, 7, 2, 1, 4])
            .repeat_dim(3, 2)
            .reshape([1, 7, 4, 4]);
        let value_expanded = value
            .clone()
            .reshape([1, 7, 2, 1, 4])
            .repeat_dim(3, 2)
            .reshape([1, 7, 4, 4]);
        let expected = query_chunked_attention(
            query.clone().permute([0, 2, 1, 3]),
            key_expanded.permute([0, 2, 1, 3]),
            value_expanded.permute([0, 2, 1, 3]),
            2,
        )
        .permute([0, 2, 1, 3]);
        let actual = <PortableChunkedAttention as AttentionKernel<TestBackend>>::execute_gqa(
            query, key, value, 2,
        );
        let max_abs = expected.sub(actual).abs().max().into_scalar();
        assert!(max_abs <= 1.0e-6, "portable GQA seam max_abs={max_abs}");
    }

    #[test]
    fn public_forward_uses_portable_kernel_correctness() {
        let device = Default::default();
        let attention = GqaAttention::<TestBackend>::new(8, 2, 1, 1.0e-5, &device);
        let query = deterministic_tensor::<TestBackend, 3>([1, 5, 8], 2, &device);
        let key_value = deterministic_tensor::<TestBackend, 3>([1, 7, 8], 11, &device);
        let expected = attention
            .forward_with_kernel::<PortableChunkedAttention>(query.clone(), key_value.clone(), None)
            .into_data()
            .to_vec::<f32>()
            .expect("policy-routed attention values");
        let actual = attention
            .forward(query, key_value, None)
            .into_data()
            .to_vec::<f32>()
            .expect("public attention values");

        assert_eq!(actual, expected);
    }

    #[test]
    fn configured_query_chunk_reaches_attention_kernel_correctness() {
        let device = Default::default();
        let mut attention = GqaAttention::<TestBackend>::new(8, 2, 1, 1.0e-5, &device);
        let query = deterministic_tensor::<TestBackend, 3>([1, 5, 8], 2, &device);
        let key_value = deterministic_tensor::<TestBackend, 3>([1, 5, 8], 11, &device);

        for query_chunk_size in [128, 256, 512] {
            attention.set_query_chunk_size(query_chunk_size);
            let _ = attention.forward_with_kernel::<ObserveQueryChunk>(
                query.clone(),
                key_value.clone(),
                None,
            );

            assert_eq!(
                OBSERVED_QUERY_CHUNK.load(Ordering::Relaxed),
                query_chunk_size
            );
        }
    }

    #[test]
    fn split_double_stream_shared_projection_matches_joint_projection_correctness() {
        let device = Default::default();
        let attention = DoubleStreamAttention::<TestBackend>::new(8, 2, 1, 1.0e-5, &device);
        assert!(
            attention.to_out.bias.is_none(),
            "shared dual-stream projection must remain bias-free"
        );
        let instruction = deterministic_tensor::<TestBackend, 3>([1, 3, 8], 5, &device);
        let image = deterministic_tensor::<TestBackend, 3>([1, 5, 8], 13, &device);

        let (expected_image, expected_instruction) =
            attention.project_shared_output(instruction.clone(), image.clone(), false);
        let (actual_image, actual_instruction) =
            attention.project_shared_output(instruction, image, true);

        for (label, expected, actual) in [
            ("image", expected_image, actual_image),
            ("instruction", expected_instruction, actual_instruction),
        ] {
            assert_eq!(expected.dims(), actual.dims(), "{label} shape mismatch");
            let max_abs = expected.sub(actual).abs().max().into_scalar();
            assert!(
                max_abs <= 1.0e-6,
                "split {label} shared projection max_abs={max_abs}"
            );
        }
    }
}
