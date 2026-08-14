//! Exact four-step Turbo orchestration over reusable Qwen3-VL, FLUX VAE, and Boogu modules.
//!
//! The individual stage functions are intentionally public. A constrained WebGPU runtime can
//! encode the instruction, drop Qwen, encode and drop the VAE encoder, stream denoiser layers for
//! each step, and finally load only the VAE decoder. [`ResidentBooguPipeline`] is the convenient
//! all-resident native form and uses the same numerical path.

use burn::{
    prelude::Backend,
    tensor::{DType, Tensor, TensorData},
};
use burn_flux_vae::{AutoencoderKl, DecoderGroupNormPolicy};
use burn_qwen3_vl::{
    Qwen3VlConfig, Qwen3VlModel, Qwen3VlModelInput, Qwen3VlStageSource, StreamingForwardError,
    StreamingQwen3Vl,
};

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
use crate::DenoiserRmsNormPolicy;
use crate::{
    BooguDenoiser, BooguDenoiserInput, BooguError, BooguTask, BooguVariant, DmdSchedule,
    dmd_prediction, dmd_renoise,
};

/// A denoiser implementation usable by the DMD loop.
///
/// The trait is the layer-streaming seam: native applications normally use [`BooguDenoiser`],
/// while browser runtimes may implement it with verified, short-lived stage modules.
pub trait DmdDenoiser<B: Backend> {
    /// Activation dtype required by the loaded denoiser, when it can be inspected cheaply.
    ///
    /// Resident models return the dtype of a non-quantized parameter. Streamed sources may
    /// return `None` and rely on [`BooguDmdInput::execution_dtype`], which is derived from the
    /// verified artifact profile and load policy by the runtime that owns the source.
    fn execution_dtype(&self) -> Option<DType> {
        None
    }

    /// Predict the rectified-flow velocity for one sigma.
    fn predict(&mut self, input: BooguDenoiserInput<B>) -> Result<Tensor<B, 4>, BooguError>;
}

impl<B: Backend> DmdDenoiser<B> for BooguDenoiser<B> {
    fn execution_dtype(&self) -> Option<DType> {
        self.x_embedder.bias.as_ref().map(|bias| bias.val().dtype())
    }

    fn predict(&mut self, input: BooguDenoiserInput<B>) -> Result<Tensor<B, 4>, BooguError> {
        self.forward(input)
    }
}

/// Native resident denoiser using Burn's portable bounded-attention graph.
///
/// This adapter leaves the verified dense weights on the shared WGPU device and caches the exact
/// step-invariant RoPE tensors across all four DMD predictions. It is used by native storage
/// profiles that do not select the separately qualified padded-blackbox execution policy.
#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
pub struct NativePortableDenoiser {
    denoiser: BooguDenoiser<crate::model::NativeWgpuBackend>,
    rope_geometry: Option<crate::model::BooguRoPeGeometry<crate::model::NativeWgpuBackend>>,
    rope_cache_misses: usize,
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
impl NativePortableDenoiser {
    /// Wrap an already verified, fully resident native WGPU denoiser.
    pub const fn new(denoiser: BooguDenoiser<crate::model::NativeWgpuBackend>) -> Self {
        Self {
            denoiser,
            rope_geometry: None,
            rope_cache_misses: 0,
        }
    }

    /// Access the resident denoiser.
    pub const fn denoiser(&self) -> &BooguDenoiser<crate::model::NativeWgpuBackend> {
        &self.denoiser
    }

    /// Mutably access the resident denoiser and invalidate shape-derived cached tensors.
    pub fn denoiser_mut(&mut self) -> &mut BooguDenoiser<crate::model::NativeWgpuBackend> {
        self.rope_geometry = None;
        &mut self.denoiser
    }

    /// Number of exact input geometries built and uploaded since construction.
    pub const fn rope_cache_misses(&self) -> usize {
        self.rope_cache_misses
    }

    /// Consume the adapter and return its resident denoiser.
    pub fn into_inner(self) -> BooguDenoiser<crate::model::NativeWgpuBackend> {
        self.denoiser
    }
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
impl DmdDenoiser<crate::model::NativeWgpuBackend> for NativePortableDenoiser {
    fn execution_dtype(&self) -> Option<DType> {
        self.denoiser
            .x_embedder
            .bias
            .as_ref()
            .map(|bias| bias.val().dtype())
    }

    fn predict(
        &mut self,
        input: BooguDenoiserInput<crate::model::NativeWgpuBackend>,
    ) -> Result<Tensor<crate::model::NativeWgpuBackend, 4>, BooguError> {
        if !self
            .rope_geometry
            .as_ref()
            .is_some_and(|geometry| geometry.matches(&input))
        {
            self.rope_geometry = Some(self.denoiser.prepare_rope_geometry(&input)?);
            self.rope_cache_misses += 1;
        }
        self.denoiser.forward_with_prepared_rope(
            input,
            self.rope_geometry
                .as_ref()
                .expect("native RoPE geometry was populated above"),
        )
    }
}

/// Native WGPU adapter that requires Cubek `FlashUnit` for every denoiser attention operation.
///
/// Constructing this adapter is an explicit execution-policy choice. The wrapped model and its
/// checkpoint record are unchanged, while [`DmdDenoiser::predict`] dispatches the native-only
/// fail-closed FlashUnit path rather than the generic bounded attention path.
#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
pub struct NativeFlashUnitDenoiser {
    denoiser: BooguDenoiser<crate::model::NativeWgpuBackend>,
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
impl NativeFlashUnitDenoiser {
    /// Wrap an already loaded native WGPU denoiser without modifying its parameters.
    pub const fn new(denoiser: BooguDenoiser<crate::model::NativeWgpuBackend>) -> Self {
        Self { denoiser }
    }

    /// Access the wrapped denoiser.
    pub const fn denoiser(&self) -> &BooguDenoiser<crate::model::NativeWgpuBackend> {
        &self.denoiser
    }

    /// Mutably access the wrapped denoiser.
    pub fn denoiser_mut(&mut self) -> &mut BooguDenoiser<crate::model::NativeWgpuBackend> {
        &mut self.denoiser
    }

    /// Set the maximum query rows submitted to each required-FlashUnit operation.
    pub fn set_attention_query_chunk_size(&mut self, query_chunk_size: usize) {
        self.denoiser
            .set_attention_query_chunk_size(query_chunk_size);
    }

    /// Return the wrapped denoiser.
    pub fn into_inner(self) -> BooguDenoiser<crate::model::NativeWgpuBackend> {
        self.denoiser
    }
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
impl DmdDenoiser<crate::model::NativeWgpuBackend> for NativeFlashUnitDenoiser {
    fn execution_dtype(&self) -> Option<DType> {
        self.denoiser
            .x_embedder
            .bias
            .as_ref()
            .map(|bias| bias.val().dtype())
    }

    fn predict(
        &mut self,
        input: BooguDenoiserInput<crate::model::NativeWgpuBackend>,
    ) -> Result<Tensor<crate::model::NativeWgpuBackend, 4>, BooguError> {
        self.denoiser.forward_native_flash_unit(input)
    }
}

/// Native WGPU adapter using padded, accelerated Cubek blackbox FlashAttention.
///
/// The adapter retains Boogu's configured bounded query chunks, pads 120-wide attention heads to
/// the CMMA-compatible width 128, and corrects the query scale before each required accelerated
/// kernel. It never routes through attention autotuning or a dense fallback.
#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
pub struct NativePaddedBlackboxDenoiser {
    denoiser: BooguDenoiser<crate::model::NativeWgpuBackend>,
    rope_geometry: Option<crate::model::BooguRoPeGeometry<crate::model::NativeWgpuBackend>>,
    rope_cache_misses: usize,
    num_planes: u8,
    seq_kv_tiles: u8,
    seq_q_tiles: u8,
    rms_norm_policy: DenoiserRmsNormPolicy,
    fused_strict_qk_norm_rope: bool,
    fused_rope_gqa_padding: bool,
    balanced_strict_qk_norm_rope: bool,
    split_double_stream_shared_projection: bool,
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
impl NativePaddedBlackboxDenoiser {
    /// Wrap a native WGPU denoiser with the default four-plane accelerated strategy.
    pub fn new(denoiser: BooguDenoiser<crate::model::NativeWgpuBackend>) -> Self {
        Self {
            denoiser,
            rope_geometry: None,
            rope_cache_misses: 0,
            num_planes: 4,
            seq_kv_tiles: 1,
            seq_q_tiles: 1,
            rms_norm_policy: DenoiserRmsNormPolicy::StrictF32,
            fused_strict_qk_norm_rope: false,
            fused_rope_gqa_padding: false,
            balanced_strict_qk_norm_rope: false,
            split_double_stream_shared_projection: false,
        }
    }

    /// Set the accelerated stage plane count to 2 or 4 on native WGPU.
    pub fn with_num_planes(mut self, num_planes: u8) -> Self {
        self.set_num_planes(num_planes);
        self
    }

    /// Set the key/value partition width to 1 or 2 CMMA tiles; two requires two planes.
    pub fn with_seq_kv_tiles(mut self, seq_kv_tiles: u8) -> Self {
        self.set_seq_kv_tiles(seq_kv_tiles);
        self
    }

    /// Set the plane count and key/value partition width atomically.
    pub fn with_configuration(mut self, num_planes: u8, seq_kv_tiles: u8) -> Self {
        self.set_configuration(num_planes, seq_kv_tiles);
        self
    }

    /// Set the query partition width. Only one tile is currently validated; other values fail
    /// closed.
    pub fn with_seq_q_tiles(mut self, seq_q_tiles: u8) -> Self {
        self.set_seq_q_tiles(seq_q_tiles);
        self
    }

    /// Set the plane count and both partition widths atomically.
    pub fn with_partition_configuration(
        mut self,
        num_planes: u8,
        seq_kv_tiles: u8,
        seq_q_tiles: u8,
    ) -> Self {
        self.set_partition_configuration(num_planes, seq_kv_tiles, seq_q_tiles);
        self
    }

    /// Select the denoiser RMSNorm numerical policy.
    ///
    /// The mixed-storage policy is diagnostic until it passes the pinned full-chain parity and
    /// synchronized performance gates. This adapter defaults to [`DenoiserRmsNormPolicy::StrictF32`].
    pub fn with_rms_norm_policy(mut self, policy: DenoiserRmsNormPolicy) -> Self {
        self.set_rms_norm_policy(policy);
        self
    }

    /// Select the opt-in fused strict Q/K RMSNorm+RoPE preparation candidate.
    ///
    /// Enabling this requires p4/kv1/q1 and [`DenoiserRmsNormPolicy::StrictF32`]. It defaults to
    /// false and remains diagnostic until pinned full-chain parity and performance gates pass.
    pub fn with_fused_strict_qk_norm_rope(mut self, enabled: bool) -> Self {
        self.set_fused_strict_qk_norm_rope(enabled);
        self
    }

    /// Select the opt-in fused RoPE, query-scale, GQA-expansion, and padding candidate.
    ///
    /// The stock strict-F32 RMSNorm operations remain unchanged. Enabling this requires p4/kv1/q1
    /// and is mutually exclusive with full Q/K fusion. It defaults to false pending real gates.
    pub fn with_fused_rope_gqa_padding(mut self, enabled: bool) -> Self {
        self.set_fused_rope_gqa_padding(enabled);
        self
    }

    /// Select balanced strict-F32 Q/K normalization feeding narrow RoPE+GQA preparation.
    ///
    /// This diagnostic candidate uses separate Q and K reduction dispatches and retains F32
    /// arithmetic through the affine multiply before its F16 output. Enabling it requires
    /// p4/kv1/q1 and excludes the other fused preparation candidates.
    pub fn with_balanced_strict_qk_norm_rope(mut self, enabled: bool) -> Self {
        self.set_balanced_strict_qk_norm_rope(enabled);
        self
    }

    /// Select separate application of the final shared dual-stream token projection.
    ///
    /// This default-off diagnostic candidate is independent of Q/K preparation policy. It applies
    /// the exact same bias-free feature projection to each stream separately, avoiding a joint
    /// token cat and two output narrows without introducing any cross-token operation.
    pub fn with_split_double_stream_shared_projection(mut self, enabled: bool) -> Self {
        self.set_split_double_stream_shared_projection(enabled);
        self
    }

    /// Access the wrapped denoiser.
    pub const fn denoiser(&self) -> &BooguDenoiser<crate::model::NativeWgpuBackend> {
        &self.denoiser
    }

    /// Mutably access the wrapped denoiser.
    pub fn denoiser_mut(&mut self) -> &mut BooguDenoiser<crate::model::NativeWgpuBackend> {
        self.rope_geometry = None;
        &mut self.denoiser
    }

    /// Number of exact input geometries built and uploaded since construction.
    ///
    /// A four-step DMD request with stable shape/dtype/device increments this once. A later request
    /// with the same contract reuses the same device tensors; an exact geometry change increments
    /// it again.
    pub const fn rope_cache_misses(&self) -> usize {
        self.rope_cache_misses
    }

    /// Return the configured accelerated stage plane count.
    pub const fn num_planes(&self) -> u8 {
        self.num_planes
    }

    /// Return the number of 16-row key/value tiles processed in each partition.
    pub const fn seq_kv_tiles(&self) -> u8 {
        self.seq_kv_tiles
    }

    /// Return the number of 16-row query tiles retained by each plane.
    pub const fn seq_q_tiles(&self) -> u8 {
        self.seq_q_tiles
    }

    /// Return the configured denoiser RMSNorm numerical policy.
    pub const fn rms_norm_policy(&self) -> DenoiserRmsNormPolicy {
        self.rms_norm_policy
    }

    /// Report whether fused strict Q/K RMSNorm+RoPE preparation is selected.
    pub const fn fused_strict_qk_norm_rope(&self) -> bool {
        self.fused_strict_qk_norm_rope
    }

    /// Report whether narrow RoPE+GQA padding fusion is selected.
    pub const fn fused_rope_gqa_padding(&self) -> bool {
        self.fused_rope_gqa_padding
    }

    /// Report whether balanced strict Q/K RMSNorm plus narrow preparation is selected.
    pub const fn balanced_strict_qk_norm_rope(&self) -> bool {
        self.balanced_strict_qk_norm_rope
    }

    /// Report whether the final shared dual-stream projection is applied per stream.
    pub const fn split_double_stream_shared_projection(&self) -> bool {
        self.split_double_stream_shared_projection
    }

    /// Set the accelerated stage plane count to one of the native-WGPU validated values.
    pub fn set_num_planes(&mut self, num_planes: u8) {
        crate::model::assert_supported_wgpu_blackbox_configuration(num_planes, self.seq_kv_tiles);
        self.assert_fused_preparation_configuration(
            num_planes,
            self.seq_kv_tiles,
            1,
            self.rms_norm_policy,
            self.fused_strict_qk_norm_rope,
            self.fused_rope_gqa_padding,
            self.balanced_strict_qk_norm_rope,
        );
        self.num_planes = num_planes;
        self.seq_q_tiles = 1;
    }

    /// Set the number of 16-row key/value tiles processed in each partition.
    pub fn set_seq_kv_tiles(&mut self, seq_kv_tiles: u8) {
        crate::model::assert_supported_wgpu_blackbox_configuration(self.num_planes, seq_kv_tiles);
        self.assert_fused_preparation_configuration(
            self.num_planes,
            seq_kv_tiles,
            1,
            self.rms_norm_policy,
            self.fused_strict_qk_norm_rope,
            self.fused_rope_gqa_padding,
            self.balanced_strict_qk_norm_rope,
        );
        self.seq_kv_tiles = seq_kv_tiles;
        self.seq_q_tiles = 1;
    }

    /// Set the plane count and key/value partition width atomically.
    pub fn set_configuration(&mut self, num_planes: u8, seq_kv_tiles: u8) {
        crate::model::assert_supported_wgpu_blackbox_configuration(num_planes, seq_kv_tiles);
        self.assert_fused_preparation_configuration(
            num_planes,
            seq_kv_tiles,
            1,
            self.rms_norm_policy,
            self.fused_strict_qk_norm_rope,
            self.fused_rope_gqa_padding,
            self.balanced_strict_qk_norm_rope,
        );
        self.num_planes = num_planes;
        self.seq_kv_tiles = seq_kv_tiles;
        self.seq_q_tiles = 1;
    }

    /// Set the number of 16-row query tiles retained by each plane.
    pub fn set_seq_q_tiles(&mut self, seq_q_tiles: u8) {
        crate::model::assert_supported_wgpu_blackbox_partition_configuration(
            self.num_planes,
            self.seq_kv_tiles,
            seq_q_tiles,
        );
        self.assert_fused_preparation_configuration(
            self.num_planes,
            self.seq_kv_tiles,
            seq_q_tiles,
            self.rms_norm_policy,
            self.fused_strict_qk_norm_rope,
            self.fused_rope_gqa_padding,
            self.balanced_strict_qk_norm_rope,
        );
        self.seq_q_tiles = seq_q_tiles;
    }

    /// Set the plane count and both partition widths atomically.
    pub fn set_partition_configuration(
        &mut self,
        num_planes: u8,
        seq_kv_tiles: u8,
        seq_q_tiles: u8,
    ) {
        crate::model::assert_supported_wgpu_blackbox_partition_configuration(
            num_planes,
            seq_kv_tiles,
            seq_q_tiles,
        );
        self.assert_fused_preparation_configuration(
            num_planes,
            seq_kv_tiles,
            seq_q_tiles,
            self.rms_norm_policy,
            self.fused_strict_qk_norm_rope,
            self.fused_rope_gqa_padding,
            self.balanced_strict_qk_norm_rope,
        );
        self.num_planes = num_planes;
        self.seq_kv_tiles = seq_kv_tiles;
        self.seq_q_tiles = seq_q_tiles;
    }

    /// Select the denoiser RMSNorm numerical policy.
    ///
    /// Selecting mixed F16 storage does not by itself establish numerical or release parity.
    pub fn set_rms_norm_policy(&mut self, policy: DenoiserRmsNormPolicy) {
        self.assert_fused_preparation_configuration(
            self.num_planes,
            self.seq_kv_tiles,
            self.seq_q_tiles,
            policy,
            self.fused_strict_qk_norm_rope,
            self.fused_rope_gqa_padding,
            self.balanced_strict_qk_norm_rope,
        );
        self.rms_norm_policy = policy;
    }

    /// Enable or disable fused strict Q/K RMSNorm+RoPE preparation.
    ///
    /// Enabling fails closed unless the adapter is configured for p4/kv1/q1 with StrictF32
    /// RMSNorm. Disabling always succeeds and restores the established preparation graph.
    pub fn set_fused_strict_qk_norm_rope(&mut self, enabled: bool) {
        self.assert_fused_preparation_configuration(
            self.num_planes,
            self.seq_kv_tiles,
            self.seq_q_tiles,
            self.rms_norm_policy,
            enabled,
            self.fused_rope_gqa_padding,
            self.balanced_strict_qk_norm_rope,
        );
        self.fused_strict_qk_norm_rope = enabled;
    }

    /// Enable or disable narrow RoPE+GQA padding fusion.
    ///
    /// Enabling fails closed unless the adapter is p4/kv1/q1 with stock StrictF32 RMSNorm and the
    /// rejected full Q/K fusion candidate is disabled.
    pub fn set_fused_rope_gqa_padding(&mut self, enabled: bool) {
        self.assert_fused_preparation_configuration(
            self.num_planes,
            self.seq_kv_tiles,
            self.seq_q_tiles,
            self.rms_norm_policy,
            self.fused_strict_qk_norm_rope,
            enabled,
            self.balanced_strict_qk_norm_rope,
        );
        self.fused_rope_gqa_padding = enabled;
    }

    /// Enable or disable balanced strict Q/K RMSNorm feeding narrow preparation fusion.
    ///
    /// Enabling fails closed unless the adapter is p4/kv1/q1 with StrictF32 RMSNorm and both
    /// other preparation candidates are disabled.
    pub fn set_balanced_strict_qk_norm_rope(&mut self, enabled: bool) {
        self.assert_fused_preparation_configuration(
            self.num_planes,
            self.seq_kv_tiles,
            self.seq_q_tiles,
            self.rms_norm_policy,
            self.fused_strict_qk_norm_rope,
            self.fused_rope_gqa_padding,
            enabled,
        );
        self.balanced_strict_qk_norm_rope = enabled;
    }

    /// Enable or disable separate application of the final shared dual-stream projection.
    ///
    /// This policy is independent of the native attention preparation candidates because it acts
    /// only after joint attention and the stream-specific output projections.
    pub fn set_split_double_stream_shared_projection(&mut self, enabled: bool) {
        self.split_double_stream_shared_projection = enabled;
    }

    #[allow(clippy::too_many_arguments)]
    fn assert_fused_preparation_configuration(
        &self,
        num_planes: u8,
        seq_kv_tiles: u8,
        seq_q_tiles: u8,
        rms_norm_policy: DenoiserRmsNormPolicy,
        fused_strict_qk_norm_rope: bool,
        fused_rope_gqa_padding: bool,
        balanced_strict_qk_norm_rope: bool,
    ) {
        assert!(
            u8::from(fused_strict_qk_norm_rope)
                + u8::from(fused_rope_gqa_padding)
                + u8::from(balanced_strict_qk_norm_rope)
                <= 1,
            "native Q/K preparation candidates are mutually exclusive"
        );
        if !(fused_strict_qk_norm_rope || fused_rope_gqa_padding || balanced_strict_qk_norm_rope) {
            return;
        }
        assert_eq!(
            (num_planes, seq_kv_tiles, seq_q_tiles),
            (4, 1, 1),
            "fused native Q/K preparation requires p4/kv1/q1"
        );
        assert_eq!(
            rms_norm_policy,
            DenoiserRmsNormPolicy::StrictF32,
            "fused native Q/K preparation requires StrictF32 RMSNorm"
        );
    }

    /// Set the maximum query rows submitted to each required blackbox operation.
    pub fn set_attention_query_chunk_size(&mut self, query_chunk_size: usize) {
        self.denoiser
            .set_attention_query_chunk_size(query_chunk_size);
    }

    /// Return the wrapped denoiser.
    pub fn into_inner(self) -> BooguDenoiser<crate::model::NativeWgpuBackend> {
        self.denoiser
    }
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
impl DmdDenoiser<crate::model::NativeWgpuBackend> for NativePaddedBlackboxDenoiser {
    fn execution_dtype(&self) -> Option<DType> {
        self.denoiser
            .x_embedder
            .bias
            .as_ref()
            .map(|bias| bias.val().dtype())
    }

    fn predict(
        &mut self,
        input: BooguDenoiserInput<crate::model::NativeWgpuBackend>,
    ) -> Result<Tensor<crate::model::NativeWgpuBackend, 4>, BooguError> {
        if !self
            .rope_geometry
            .as_ref()
            .is_some_and(|geometry| geometry.matches(&input))
        {
            self.rope_geometry = Some(self.denoiser.prepare_rope_geometry(&input)?);
            self.rope_cache_misses += 1;
        }
        let geometry = self
            .rope_geometry
            .as_ref()
            .expect("native RoPE geometry was populated above");
        self.denoiser
            .forward_native_padded_blackbox_partitioned_with_prepared_rope_and_policies(
                input,
                geometry,
                self.num_planes,
                self.seq_kv_tiles,
                self.seq_q_tiles,
                self.rms_norm_policy,
                self.fused_strict_qk_norm_rope,
                self.fused_rope_gqa_padding,
                self.balanced_strict_qk_norm_rope,
                self.split_double_stream_shared_projection,
            )
    }
}

/// Native CUDA adapter that requires Cubek `FlashUnit` for every denoiser attention operation.
///
/// The wrapped model and checkpoint record are unchanged. Prediction uses bounded-query,
/// fail-closed FlashUnit submissions and never invokes attention autotuning or dense fallback.
#[cfg(all(feature = "cuda-experimental", not(target_arch = "wasm32")))]
pub struct NativeCudaFlashUnitDenoiser {
    denoiser: BooguDenoiser<crate::model::NativeCudaBackend>,
}

#[cfg(all(feature = "cuda-experimental", not(target_arch = "wasm32")))]
impl NativeCudaFlashUnitDenoiser {
    /// Wrap an already loaded native CUDA denoiser without modifying its parameters.
    pub const fn new(denoiser: BooguDenoiser<crate::model::NativeCudaBackend>) -> Self {
        Self { denoiser }
    }

    /// Access the wrapped denoiser.
    pub const fn denoiser(&self) -> &BooguDenoiser<crate::model::NativeCudaBackend> {
        &self.denoiser
    }

    /// Mutably access the wrapped denoiser.
    pub fn denoiser_mut(&mut self) -> &mut BooguDenoiser<crate::model::NativeCudaBackend> {
        &mut self.denoiser
    }

    /// Set the maximum query rows submitted to each required-FlashUnit operation.
    pub fn set_attention_query_chunk_size(&mut self, query_chunk_size: usize) {
        self.denoiser
            .set_attention_query_chunk_size(query_chunk_size);
    }

    /// Return the wrapped denoiser.
    pub fn into_inner(self) -> BooguDenoiser<crate::model::NativeCudaBackend> {
        self.denoiser
    }
}

#[cfg(all(feature = "cuda-experimental", not(target_arch = "wasm32")))]
impl DmdDenoiser<crate::model::NativeCudaBackend> for NativeCudaFlashUnitDenoiser {
    fn execution_dtype(&self) -> Option<DType> {
        self.denoiser
            .x_embedder
            .bias
            .as_ref()
            .map(|bias| bias.val().dtype())
    }

    fn predict(
        &mut self,
        input: BooguDenoiserInput<crate::model::NativeCudaBackend>,
    ) -> Result<Tensor<crate::model::NativeCudaBackend, 4>, BooguError> {
        self.denoiser.forward_native_cuda_flash_unit(input)
    }
}

/// Experimental native CUDA adapter using padded, accelerated Cubek blackbox FlashAttention.
///
/// The adapter retains Boogu's configured bounded query chunks, pads 120-wide attention heads to
/// 128, corrects the query scale, and forces a 16-by-16-by-16 CMMA blueprint. It accepts only F16
/// activations and never routes through attention autotuning or a dense fallback. CUDA numerical
/// parity remains an explicit prerequisite for any production support claim.
#[cfg(all(feature = "cuda-experimental", not(target_arch = "wasm32")))]
pub struct NativeCudaPaddedBlackboxDenoiser {
    denoiser: BooguDenoiser<crate::model::NativeCudaBackend>,
    num_planes: u8,
    seq_kv_tiles: u8,
}

#[cfg(all(feature = "cuda-experimental", not(target_arch = "wasm32")))]
impl NativeCudaPaddedBlackboxDenoiser {
    /// Wrap a native CUDA denoiser with the default four-plane experimental strategy.
    pub const fn new(denoiser: BooguDenoiser<crate::model::NativeCudaBackend>) -> Self {
        Self {
            denoiser,
            num_planes: 4,
            seq_kv_tiles: 1,
        }
    }

    /// Set the accelerated stage plane count to 2 or 4.
    pub fn with_num_planes(mut self, num_planes: u8) -> Self {
        self.set_num_planes(num_planes);
        self
    }

    /// Set the key/value partition width to 1 or 2 CMMA tiles; two requires two planes.
    pub fn with_seq_kv_tiles(mut self, seq_kv_tiles: u8) -> Self {
        self.set_seq_kv_tiles(seq_kv_tiles);
        self
    }

    /// Set the plane count and key/value partition width atomically.
    pub fn with_configuration(mut self, num_planes: u8, seq_kv_tiles: u8) -> Self {
        self.set_configuration(num_planes, seq_kv_tiles);
        self
    }

    /// Access the wrapped denoiser.
    pub const fn denoiser(&self) -> &BooguDenoiser<crate::model::NativeCudaBackend> {
        &self.denoiser
    }

    /// Mutably access the wrapped denoiser.
    pub fn denoiser_mut(&mut self) -> &mut BooguDenoiser<crate::model::NativeCudaBackend> {
        &mut self.denoiser
    }

    /// Return the configured accelerated stage plane count.
    pub const fn num_planes(&self) -> u8 {
        self.num_planes
    }

    /// Return the number of 16-row key/value tiles processed in each partition.
    pub const fn seq_kv_tiles(&self) -> u8 {
        self.seq_kv_tiles
    }

    /// Set the accelerated stage plane count to 2 or 4.
    pub fn set_num_planes(&mut self, num_planes: u8) {
        crate::model::assert_supported_blackbox_configuration(num_planes, self.seq_kv_tiles);
        self.num_planes = num_planes;
    }

    /// Set the number of 16-row key/value tiles processed in each partition.
    pub fn set_seq_kv_tiles(&mut self, seq_kv_tiles: u8) {
        crate::model::assert_supported_blackbox_configuration(self.num_planes, seq_kv_tiles);
        self.seq_kv_tiles = seq_kv_tiles;
    }

    /// Set the plane count and key/value partition width atomically.
    pub fn set_configuration(&mut self, num_planes: u8, seq_kv_tiles: u8) {
        crate::model::assert_supported_blackbox_configuration(num_planes, seq_kv_tiles);
        self.num_planes = num_planes;
        self.seq_kv_tiles = seq_kv_tiles;
    }

    /// Set the maximum query rows submitted to each required blackbox operation.
    pub fn set_attention_query_chunk_size(&mut self, query_chunk_size: usize) {
        self.denoiser
            .set_attention_query_chunk_size(query_chunk_size);
    }

    /// Return the wrapped denoiser.
    pub fn into_inner(self) -> BooguDenoiser<crate::model::NativeCudaBackend> {
        self.denoiser
    }
}

#[cfg(all(feature = "cuda-experimental", not(target_arch = "wasm32")))]
impl DmdDenoiser<crate::model::NativeCudaBackend> for NativeCudaPaddedBlackboxDenoiser {
    fn execution_dtype(&self) -> Option<DType> {
        self.denoiser
            .x_embedder
            .bias
            .as_ref()
            .map(|bias| bias.val().dtype())
    }

    fn predict(
        &mut self,
        input: BooguDenoiserInput<crate::model::NativeCudaBackend>,
    ) -> Result<Tensor<crate::model::NativeCudaBackend, 4>, BooguError> {
        self.denoiser.forward_native_cuda_padded_blackbox_tiled(
            input,
            self.num_planes,
            self.seq_kv_tiles,
        )
    }
}

/// Fully deterministic input to the DMD student loop.
///
/// Supplying every noise tensor explicitly avoids backend RNG differences and is required by the
/// cross-framework parity harness. There must be exactly `schedule.len() - 1` renoise tensors.
pub struct BooguDmdInput<B: Backend> {
    /// Activation dtype required by the selected denoiser artifact/load policy.
    pub execution_dtype: DType,
    /// Initial standard-normal latent `[1, 16, H/8, W/8]`.
    pub initial_latents: Tensor<B, 4>,
    /// Trimmed, unpadded final Qwen hidden state `[1, T, 4096]`.
    pub instruction: Tensor<B, 3>,
    /// Scaled VAE latent for the single edit reference, when editing.
    pub reference: Option<Tensor<B, 4>>,
    /// Fresh standard-normal tensors used between adjacent DMD steps.
    pub renoise: Vec<Tensor<B, 4>>,
    /// Explicit schedule, normally [`DmdSchedule::upstream_for_dtype`].
    pub schedule: DmdSchedule,
}

/// Result before and after VAE decoding.
pub struct BooguPipelineOutput<B: Backend> {
    /// Final scaled FLUX latent, useful for parity checks and resumable runtimes.
    pub latents: Tensor<B, 4>,
    /// Raw VAE decoder output in the upstream `[-1, 1]` convention.
    pub image: Tensor<B, 4>,
}

/// Encode one Qwen processor result and remove right-padding before the denoiser.
///
/// Boogu's released checkpoints consume the base model's final hidden state, not LM logits. The
/// upstream default keeps image placeholder features in edit prompts, so this function does not
/// filter vision-token positions.
pub fn encode_instruction<B: Backend>(
    qwen: &Qwen3VlModel<B>,
    input: Qwen3VlModelInput<B>,
    effective_length: usize,
) -> Result<Tensor<B, 3>, BooguError> {
    let output = qwen.forward(input).map_err(|error| {
        BooguError::InvalidRequest(format!("Qwen conditioning failed: {error}"))
    })?;
    trim_instruction_features(output.last_hidden_state, effective_length)
}

/// Validate and remove right-padding from an already-computed Qwen base-model output.
///
/// Resident and semantic-stage-streamed Qwen execution share this exact boundary so the Boogu
/// denoiser never depends on which weight residency strategy produced the instruction features.
pub fn trim_instruction_features<B: Backend>(
    hidden_states: Tensor<B, 3>,
    effective_length: usize,
) -> Result<Tensor<B, 3>, BooguError> {
    let [batch, padded_length, width] = hidden_states.dims();
    if batch != 1 {
        return Err(BooguError::InvalidShape(
            "the initial Turbo parity path requires one instruction".into(),
        ));
    }
    if effective_length == 0 || effective_length > padded_length {
        return Err(BooguError::InvalidShape(format!(
            "effective instruction length {effective_length} is outside 1..={padded_length}"
        )));
    }
    if width != 4096 {
        return Err(BooguError::InvalidShape(format!(
            "released Boogu checkpoints require 4096-wide Qwen features, got {width}"
        )));
    }
    Ok(hidden_states.narrow(1, 0, effective_length))
}

/// Loads one short-lived half of the ordinary FLUX VAE.
///
/// Implementations must verify the complete selected stage before returning it. The opposite
/// stage should remain lazy and unfetched so an edit can encode, release the encoder, run DMD,
/// then load only the decoder.
pub trait BooguVaeStageSource<B: Backend> {
    /// Verify, load, and return the encoder half for one encode call.
    fn load_encoder(&mut self) -> Result<AutoencoderKl<B>, BooguError>;

    /// Verify, load, and return the decoder half for one decode call.
    fn load_decoder(&mut self) -> Result<AutoencoderKl<B>, BooguError>;
}

/// Thin compatibility adapter from the reusable FLUX VAE artifact source into Boogu composition.
///
/// Shard verification, loading, and device residency remain owned by `burn_flux_vae`; this type
/// only translates its model-neutral error at the pipeline boundary.
#[cfg(feature = "burnpack")]
pub struct FluxVaeStageSourceAdapter<S> {
    source: S,
}

#[cfg(feature = "burnpack")]
impl<S> FluxVaeStageSourceAdapter<S> {
    /// Wrap a reusable synchronous FLUX VAE artifact source.
    pub const fn new(source: S) -> Self {
        Self { source }
    }

    /// Borrow the underlying reusable source.
    pub const fn source(&self) -> &S {
        &self.source
    }

    /// Mutably borrow the underlying reusable source.
    pub const fn source_mut(&mut self) -> &mut S {
        &mut self.source
    }

    /// Unwrap and return the reusable source.
    pub fn into_source(self) -> S {
        self.source
    }
}

#[cfg(feature = "burnpack")]
impl<B, S> BooguVaeStageSource<B> for FluxVaeStageSourceAdapter<S>
where
    B: Backend,
    S: burn_flux_vae::FluxVaeStageSource<B>,
    S::Error: std::fmt::Display,
{
    fn load_encoder(&mut self) -> Result<AutoencoderKl<B>, BooguError> {
        self.source
            .load_encoder()
            .map_err(|error| BooguError::Artifact(error.to_string()))
    }

    fn load_decoder(&mut self) -> Result<AutoencoderKl<B>, BooguError> {
        self.source
            .load_decoder()
            .map_err(|error| BooguError::Artifact(error.to_string()))
    }
}

/// Native high-residency wrapper that retains each verified VAE half after its first load.
///
/// Burn modules clone initialized backend tensor handles, so returning a clone does not reread
/// the artifact or duplicate the underlying WGPU allocation. The wrapper remains opt-in: browser
/// and lower-residency runtimes should keep using their stage source directly so only one VAE half
/// is resident at a time.
pub struct RetainingBooguVaeStageSource<B: Backend, S> {
    source: S,
    encoder: Option<AutoencoderKl<B>>,
    decoder: Option<AutoencoderKl<B>>,
}

impl<B: Backend, S> RetainingBooguVaeStageSource<B, S> {
    /// Wrap a verified stage source with an initially empty encoder/decoder cache.
    pub const fn new(source: S) -> Self {
        Self {
            source,
            encoder: None,
            decoder: None,
        }
    }

    /// Number of VAE halves currently retained (`0..=2`).
    pub fn cached_stage_count(&self) -> usize {
        usize::from(self.encoder.is_some()) + usize::from(self.decoder.is_some())
    }

    /// Drop both retained module handles without changing the verified underlying source.
    ///
    /// Callers must ensure prior device work has completed before clearing a live GPU cache.
    pub fn clear(&mut self) {
        self.encoder = None;
        self.decoder = None;
    }

    /// Borrow the verified source wrapped by this cache.
    pub const fn source(&self) -> &S {
        &self.source
    }

    /// Mutably borrow the verified source wrapped by this cache.
    pub const fn source_mut(&mut self) -> &mut S {
        &mut self.source
    }

    /// Consume the wrapper and return its verified source.
    pub fn into_source(self) -> S {
        self.source
    }
}

impl<B, S> BooguVaeStageSource<B> for RetainingBooguVaeStageSource<B, S>
where
    B: Backend,
    S: BooguVaeStageSource<B>,
{
    fn load_encoder(&mut self) -> Result<AutoencoderKl<B>, BooguError> {
        if self.encoder.is_none() {
            self.encoder = Some(self.source.load_encoder()?);
        }
        Ok(self
            .encoder
            .as_ref()
            .expect("encoder was populated above")
            .clone())
    }

    fn load_decoder(&mut self) -> Result<AutoencoderKl<B>, BooguError> {
        if self.decoder.is_none() {
            self.decoder = Some(self.source.load_decoder()?);
        }
        Ok(self
            .decoder
            .as_ref()
            .expect("decoder was populated above")
            .clone())
    }
}

/// Wasm-local asynchronous source of one verified FLUX VAE half at a time.
///
/// Futures deliberately have no [`Send`] requirement so browser fetch/cache and WebGPU handles
/// can stay on one JavaScript event loop. Implementations must fetch and apply physical shards
/// sequentially and leave the opposite VAE half lazy and unfetched.
#[allow(async_fn_in_trait)]
pub trait AsyncBooguVaeStageSource<B: Backend> {
    /// Verify, load, and return the encoder half for one encode call.
    async fn load_encoder(&mut self) -> Result<AutoencoderKl<B>, BooguError>;

    /// Verify, load, and return the decoder half for one decode call.
    async fn load_decoder(&mut self) -> Result<AutoencoderKl<B>, BooguError>;

    /// Await submitted device work before the selected VAE half is dropped.
    async fn synchronize(&mut self) -> Result<(), BooguError>;
}

/// Wasm-local compatibility adapter for the reusable asynchronous FLUX VAE source.
#[cfg(feature = "burnpack")]
pub struct AsyncFluxVaeStageSourceAdapter<S> {
    source: S,
}

#[cfg(feature = "burnpack")]
impl<S> AsyncFluxVaeStageSourceAdapter<S> {
    /// Wrap a reusable asynchronous FLUX VAE artifact source.
    pub const fn new(source: S) -> Self {
        Self { source }
    }

    /// Borrow the underlying reusable source.
    pub const fn source(&self) -> &S {
        &self.source
    }

    /// Mutably borrow the underlying reusable source.
    pub const fn source_mut(&mut self) -> &mut S {
        &mut self.source
    }

    /// Unwrap and return the reusable source.
    pub fn into_source(self) -> S {
        self.source
    }
}

#[cfg(feature = "burnpack")]
impl<B, S> AsyncBooguVaeStageSource<B> for AsyncFluxVaeStageSourceAdapter<S>
where
    B: Backend,
    S: burn_flux_vae::AsyncFluxVaeStageSource<B>,
    S::Error: std::fmt::Display,
{
    async fn load_encoder(&mut self) -> Result<AutoencoderKl<B>, BooguError> {
        self.source
            .load_encoder()
            .await
            .map_err(|error| BooguError::Artifact(error.to_string()))
    }

    async fn load_decoder(&mut self) -> Result<AutoencoderKl<B>, BooguError> {
        self.source
            .load_decoder()
            .await
            .map_err(|error| BooguError::Artifact(error.to_string()))
    }

    async fn synchronize(&mut self) -> Result<(), BooguError> {
        self.source
            .synchronize()
            .await
            .map_err(|error| BooguError::Artifact(error.to_string()))
    }
}

/// Opt-in GPU-resident cache for an asynchronous verified FLUX VAE source.
///
/// A cache miss delegates to the wrapped source and retains only a clone of the initialized Burn
/// module. WGPU/WebGPU clones share device-buffer handles, so no Burnpack payload or decoded host
/// tensor is retained. Synchronization always reaches the wrapped source, including cache hits.
///
/// [`Self::new`] enables retention. [`Self::passthrough`] provides an explicitly non-retaining
/// wrapper for diagnostic low-memory runtimes that need the same concrete type.
pub struct RetainingAsyncBooguVaeStageSource<B: Backend, S> {
    source: S,
    retention_enabled: bool,
    encoder: Option<AutoencoderKl<B>>,
    decoder: Option<AutoencoderKl<B>>,
}

impl<B: Backend, S> RetainingAsyncBooguVaeStageSource<B, S> {
    /// Create an initially empty cache that retains each verified VAE half after first load.
    pub fn new(source: S) -> Self {
        Self::with_retention(source, true)
    }

    /// Wrap a source without retaining either VAE half.
    pub fn passthrough(source: S) -> Self {
        Self::with_retention(source, false)
    }

    fn with_retention(source: S, retention_enabled: bool) -> Self {
        Self {
            source,
            retention_enabled,
            encoder: None,
            decoder: None,
        }
    }

    /// Whether successfully loaded halves are retained for later requests.
    pub const fn retention_enabled(&self) -> bool {
        self.retention_enabled
    }

    /// Number of retained VAE halves (`0..=2`).
    pub fn cached_stage_count(&self) -> usize {
        usize::from(self.encoder.is_some()) + usize::from(self.decoder.is_some())
    }

    /// Drop both retained device handles while preserving the wrapped verified source.
    ///
    /// Callers must await the final submitted synchronization before clearing a live GPU cache.
    pub fn clear(&mut self) {
        self.encoder = None;
        self.decoder = None;
    }

    /// Borrow the wrapped verified source.
    pub const fn source(&self) -> &S {
        &self.source
    }

    /// Mutably borrow the wrapped verified source.
    pub fn source_mut(&mut self) -> &mut S {
        &mut self.source
    }

    /// Consume the wrapper, dropping retained handles and returning the source.
    pub fn into_source(self) -> S {
        self.source
    }
}

impl<B, S> AsyncBooguVaeStageSource<B> for RetainingAsyncBooguVaeStageSource<B, S>
where
    B: Backend,
    S: AsyncBooguVaeStageSource<B>,
{
    async fn load_encoder(&mut self) -> Result<AutoencoderKl<B>, BooguError> {
        if let Some(encoder) = &self.encoder {
            return Ok(encoder.clone());
        }
        let encoder = self.source.load_encoder().await?;
        if self.retention_enabled {
            self.encoder = Some(encoder.clone());
        }
        Ok(encoder)
    }

    async fn load_decoder(&mut self) -> Result<AutoencoderKl<B>, BooguError> {
        if let Some(decoder) = &self.decoder {
            return Ok(decoder.clone());
        }
        let decoder = self.source.load_decoder().await?;
        if self.retention_enabled {
            self.decoder = Some(decoder.clone());
        }
        Ok(decoder)
    }

    async fn synchronize(&mut self) -> Result<(), BooguError> {
        self.source.synchronize().await
    }
}

/// Component execution boundary shared by the model-neutral host adapter.
///
/// The resident implementation keeps all weights alive. [`StreamingBooguPipeline`] implements
/// the same numerical stages with row-routed Qwen, one denoiser layer at a time, and independent
/// VAE encoder/decoder loads.
pub trait BooguExecution<B: Backend> {
    /// Immutable release executed by this component composition.
    fn variant(&self) -> BooguVariant;

    /// Execute Qwen conditioning and trim right-padding for the denoiser.
    fn encode_instruction(
        &mut self,
        input: Qwen3VlModelInput<B>,
        effective_length: usize,
    ) -> Result<Tensor<B, 3>, BooguError>;

    /// Encode and scale the optional edit reference with exact posterior noise.
    fn encode_reference(
        &mut self,
        normalized_image: Tensor<B, 4>,
        posterior_epsilon: Tensor<B, 4>,
    ) -> Result<Tensor<B, 4>, BooguError>;

    /// Execute the four-step DMD student and report completed step boundaries.
    fn denoise_with_observer<F>(
        &mut self,
        input: BooguDmdInput<B>,
        after_step: F,
    ) -> Result<Tensor<B, 4>, BooguError>
    where
        F: FnMut(usize, f32) -> Result<(), BooguError>;

    /// Load/execute the VAE decoder for the final scaled latent.
    fn decode(&mut self, scaled_latents: Tensor<B, 4>) -> Result<Tensor<B, 4>, BooguError>;
}

/// Sample and scale the single edit reference latent with caller-provided VAE epsilon.
pub fn encode_reference<B: Backend>(
    vae: &AutoencoderKl<B>,
    normalized_image: Tensor<B, 4>,
    posterior_epsilon: Tensor<B, 4>,
) -> Result<Tensor<B, 4>, BooguError> {
    let vae_dtype: DType = vae.encoder_float_dtype().into();
    validate_tensor_dtype("normalized VAE image", normalized_image.dtype(), vae_dtype)?;
    validate_tensor_dtype(
        "VAE posterior epsilon",
        posterior_epsilon.dtype(),
        vae_dtype,
    )?;
    let [batch, channels, height, width] = normalized_image.dims();
    if batch != 1 || channels != 3 || height == 0 || width == 0 {
        return Err(BooguError::InvalidShape(format!(
            "reference must be non-empty [1,3,H,W], got [{batch},{channels},{height},{width}]"
        )));
    }
    Ok(vae.encode_scaled_with_epsilon(normalized_image, posterior_epsilon))
}

/// Execute the exact Turbo DMD update sequence.
pub fn run_dmd<B: Backend, D: DmdDenoiser<B>>(
    denoiser: &mut D,
    input: BooguDmdInput<B>,
) -> Result<Tensor<B, 4>, BooguError> {
    run_dmd_with_observer(denoiser, input, |_, _| Ok(()))
}

/// Execute DMD and call `after_step` at a safe cancellation/progress boundary.
pub fn run_dmd_with_observer<B, D, F>(
    denoiser: &mut D,
    input: BooguDmdInput<B>,
    mut after_step: F,
) -> Result<Tensor<B, 4>, BooguError>
where
    B: Backend,
    D: DmdDenoiser<B>,
    F: FnMut(usize, f32) -> Result<(), BooguError>,
{
    let sigmas = input.schedule.sigmas();
    let expected_noise = sigmas.len().saturating_sub(1);
    if input.renoise.len() != expected_noise {
        return Err(BooguError::InvalidRequest(format!(
            "DMD schedule has {} steps and requires {expected_noise} renoise tensors, got {}",
            sigmas.len(),
            input.renoise.len()
        )));
    }
    validate_dmd_shapes(&input)?;
    validate_dmd_dtypes(&input)?;
    if let Some(loaded_dtype) = denoiser.execution_dtype()
        && loaded_dtype != input.execution_dtype
    {
        return Err(BooguError::InvalidRequest(format!(
            "DMD execution dtype {} does not match loaded denoiser dtype {}",
            input.execution_dtype.name(),
            loaded_dtype.name()
        )));
    }

    let device = input.initial_latents.device();
    let execution_dtype = input.execution_dtype;
    let mut latents = input.initial_latents;
    let mut noises = input.renoise.into_iter();
    for (index, &sigma) in sigmas.iter().enumerate() {
        let timestep = Tensor::<B, 1>::from_data(TensorData::new(vec![sigma], [1]), &device)
            .cast(execution_dtype);
        let model_prediction = denoiser.predict(BooguDenoiserInput {
            latent: latents.clone(),
            timestep,
            instruction: input.instruction.clone(),
            reference: input.reference.clone(),
        })?;
        validate_tensor_dtype(
            "denoiser prediction",
            model_prediction.dtype(),
            execution_dtype,
        )?;
        latents = dmd_prediction(latents, model_prediction, sigma);
        if let Some(&next_sigma) = sigmas.get(index + 1) {
            let noise = noises.next().expect("renoise count was validated");
            latents = dmd_renoise(latents, noise, next_sigma);
        }
        after_step(index, sigma)?;
    }
    Ok(latents)
}

fn validate_dmd_shapes<B: Backend>(input: &BooguDmdInput<B>) -> Result<(), BooguError> {
    let latent_shape = input.initial_latents.dims();
    if latent_shape[0] != 1 || latent_shape[1] != 16 {
        return Err(BooguError::InvalidShape(format!(
            "released Turbo latent must have shape [1,16,H,W], got {latent_shape:?}"
        )));
    }
    if input.instruction.dims()[0] != 1 || input.instruction.dims()[2] != 4096 {
        return Err(BooguError::InvalidShape(format!(
            "released Turbo instruction must have shape [1,T,4096], got {:?}",
            input.instruction.dims()
        )));
    }
    if let Some(reference) = &input.reference
        && (reference.dims()[0] != 1 || reference.dims()[1] != 16)
    {
        return Err(BooguError::InvalidShape(format!(
            "released Edit-Turbo reference must have shape [1,16,H,W], got {:?}",
            reference.dims()
        )));
    }
    for (index, noise) in input.renoise.iter().enumerate() {
        if noise.dims() != latent_shape {
            return Err(BooguError::InvalidShape(format!(
                "renoise tensor {index} has shape {:?}, expected {latent_shape:?}",
                noise.dims()
            )));
        }
    }
    Ok(())
}

fn validate_dmd_dtypes<B: Backend>(input: &BooguDmdInput<B>) -> Result<(), BooguError> {
    if !input.execution_dtype.is_float() {
        return Err(BooguError::InvalidRequest(format!(
            "DMD execution dtype must be floating point, got {}",
            input.execution_dtype.name()
        )));
    }
    validate_tensor_dtype(
        "initial latent",
        input.initial_latents.dtype(),
        input.execution_dtype,
    )?;
    validate_tensor_dtype(
        "instruction conditioning",
        input.instruction.dtype(),
        input.execution_dtype,
    )?;
    if let Some(reference) = &input.reference {
        validate_tensor_dtype(
            "reference conditioning",
            reference.dtype(),
            input.execution_dtype,
        )?;
    }
    for (index, noise) in input.renoise.iter().enumerate() {
        validate_tensor_dtype(
            &format!("renoise tensor {index}"),
            noise.dtype(),
            input.execution_dtype,
        )?;
    }
    Ok(())
}

fn validate_tensor_dtype(name: &str, actual: DType, expected: DType) -> Result<(), BooguError> {
    if actual == expected {
        Ok(())
    } else {
        Err(BooguError::InvalidRequest(format!(
            "{name} has dtype {}, expected execution dtype {}",
            actual.name(),
            expected.name()
        )))
    }
}

/// Input to the convenient all-resident native pipeline.
pub struct ResidentBooguInput<B: Backend> {
    /// Fully processed Qwen model input, including visual patches for edits.
    pub qwen: Qwen3VlModelInput<B>,
    /// Number of valid right-padded Qwen tokens.
    pub instruction_length: usize,
    /// Initial DMD noise latent.
    pub initial_latents: Tensor<B, 4>,
    /// Optional normalized edit image and exact posterior epsilon.
    pub reference: Option<(Tensor<B, 4>, Tensor<B, 4>)>,
    /// Exact inter-step noise tensors.
    pub renoise: Vec<Tensor<B, 4>>,
}

/// All-resident composition of the three reusable model crates.
///
/// This form is suitable for high-memory native GPUs. Browser applications should call the stage
/// functions separately and release each component at stage boundaries.
pub struct ResidentBooguPipeline<B: Backend> {
    /// Ordinary Qwen3-VL instruction encoder.
    pub qwen: Qwen3VlModel<B>,
    /// Ordinary FLUX VAE.
    pub vae: AutoencoderKl<B>,
    /// Boogu-specific DMD denoiser.
    pub denoiser: BooguDenoiser<B>,
    variant: BooguVariant,
}

impl<B: Backend> ResidentBooguPipeline<B> {
    /// Compose already-loaded, verified model components.
    pub fn new(
        variant: BooguVariant,
        qwen: Qwen3VlModel<B>,
        vae: AutoencoderKl<B>,
        denoiser: BooguDenoiser<B>,
    ) -> Self {
        Self {
            qwen,
            vae,
            denoiser,
            variant,
        }
    }

    /// Selected immutable checkpoint identity.
    pub const fn variant(&self) -> BooguVariant {
        self.variant
    }

    /// Run conditioning, optional reference encoding, DMD, and VAE decode.
    pub fn forward(
        &mut self,
        input: ResidentBooguInput<B>,
    ) -> Result<BooguPipelineOutput<B>, BooguError> {
        let task = match self.variant {
            BooguVariant::Image01Turbo => BooguTask::Generate,
            BooguVariant::Image01EditTurbo | BooguVariant::Image01EditTurbo1k5 => BooguTask::Edit,
        };
        if matches!(task, BooguTask::Generate) && input.reference.is_some() {
            return Err(BooguError::InvalidRequest(
                "Boogu-Image-0.1-Turbo does not accept a reference image".into(),
            ));
        }
        if matches!(task, BooguTask::Edit) && input.reference.is_none() {
            return Err(BooguError::InvalidRequest(
                "Boogu-Image-0.1-Edit-Turbo requires exactly one reference image".into(),
            ));
        }

        let instruction = encode_instruction(&self.qwen, input.qwen, input.instruction_length)?;
        let reference = input
            .reference
            .map(|(image, epsilon)| encode_reference(&self.vae, image, epsilon))
            .transpose()?;
        let execution_dtype = input.initial_latents.dtype();
        let latents = run_dmd(
            &mut self.denoiser,
            BooguDmdInput {
                execution_dtype,
                initial_latents: input.initial_latents,
                instruction,
                reference,
                renoise: input.renoise,
                schedule: DmdSchedule::upstream_for_dtype(task, execution_dtype),
            },
        )?;
        let image = self.vae.decode_scaled(latents.clone());
        Ok(BooguPipelineOutput { latents, image })
    }
}

impl<B: Backend> BooguExecution<B> for ResidentBooguPipeline<B> {
    fn variant(&self) -> BooguVariant {
        self.variant
    }

    fn encode_instruction(
        &mut self,
        input: Qwen3VlModelInput<B>,
        effective_length: usize,
    ) -> Result<Tensor<B, 3>, BooguError> {
        encode_instruction(&self.qwen, input, effective_length)
    }

    fn encode_reference(
        &mut self,
        normalized_image: Tensor<B, 4>,
        posterior_epsilon: Tensor<B, 4>,
    ) -> Result<Tensor<B, 4>, BooguError> {
        encode_reference(&self.vae, normalized_image, posterior_epsilon)
    }

    fn denoise_with_observer<F>(
        &mut self,
        input: BooguDmdInput<B>,
        after_step: F,
    ) -> Result<Tensor<B, 4>, BooguError>
    where
        F: FnMut(usize, f32) -> Result<(), BooguError>,
    {
        run_dmd_with_observer(&mut self.denoiser, input, after_step)
    }

    fn decode(&mut self, scaled_latents: Tensor<B, 4>) -> Result<Tensor<B, 4>, BooguError> {
        let vae_dtype: DType = self.vae.decoder_float_dtype().into();
        validate_tensor_dtype("VAE decoder latent", scaled_latents.dtype(), vae_dtype)?;
        Ok(self.vae.decode_scaled(scaled_latents))
    }
}

/// Backend allocation policy for one streamed VAE decoder execution.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum VaeDecoderMemoryPolicy {
    /// Preserve the backend's ordinary allocator behavior.
    #[default]
    BackendDefault,
    /// Allocate decoder intermediates at their exact size and release dead pre-tail buffers.
    ///
    /// The decoder graph is synchronized while exact transient allocation mode is active, with a
    /// second synchronized cleanup immediately before its final full-resolution residual block.
    ExactTransientWithTailCleanup,
}

/// Memory-bounded composition of reusable Qwen3-VL, FLUX VAE, and Boogu executors.
///
/// Only activations survive semantic stage transitions. `Q` supplies verified Qwen row/module
/// stages, `V` independently supplies the VAE encoder and decoder, and `D` normally is a
/// [`crate::StreamingBooguDenoiser`]. Each source is transport-neutral: native readers can use a
/// directory while a browser orchestration layer can populate a bounded verified cache from CDN
/// range requests.
pub struct StreamingBooguPipeline<B: Backend, Q, V, D> {
    variant: BooguVariant,
    qwen_config: Qwen3VlConfig,
    decoder_group_norm_policy: DecoderGroupNormPolicy,
    decoder_memory_policy: VaeDecoderMemoryPolicy,
    /// Verified semantic-stage Qwen executor.
    pub qwen: StreamingQwen3Vl<B, Q>,
    /// Independent verified VAE encoder/decoder source.
    pub vae: V,
    /// Usually a layer-streamed Boogu denoiser.
    pub denoiser: D,
}

impl<B: Backend, Q, V, D> StreamingBooguPipeline<B, Q, V, D> {
    /// Compose already-validated streaming component sources for one release.
    pub fn new(
        variant: BooguVariant,
        qwen_config: Qwen3VlConfig,
        qwen: StreamingQwen3Vl<B, Q>,
        vae: V,
        denoiser: D,
    ) -> Self {
        Self {
            variant,
            qwen_config,
            decoder_group_norm_policy: DecoderGroupNormPolicy::StrictF32,
            decoder_memory_policy: VaeDecoderMemoryPolicy::BackendDefault,
            qwen,
            vae,
            denoiser,
        }
    }

    /// Selected immutable Boogu release.
    pub const fn variant(&self) -> BooguVariant {
        self.variant
    }

    /// Select the decoder-only GroupNorm execution policy.
    ///
    /// The default is [`DecoderGroupNormPolicy::StrictF32`]. This does not affect the Edit VAE
    /// encoder, which always retains the established strict-F32 normalization path.
    pub const fn with_decoder_group_norm_policy(mut self, policy: DecoderGroupNormPolicy) -> Self {
        self.decoder_group_norm_policy = policy;
        self
    }

    /// Return the decoder-only GroupNorm execution policy.
    pub const fn decoder_group_norm_policy(&self) -> DecoderGroupNormPolicy {
        self.decoder_group_norm_policy
    }

    /// Select the VAE decoder's backend allocation policy.
    pub const fn with_decoder_memory_policy(mut self, policy: VaeDecoderMemoryPolicy) -> Self {
        self.decoder_memory_policy = policy;
        self
    }

    /// Return the selected VAE decoder backend allocation policy.
    pub const fn decoder_memory_policy(&self) -> VaeDecoderMemoryPolicy {
        self.decoder_memory_policy
    }
}

impl<B, Q, V, D> BooguExecution<B> for StreamingBooguPipeline<B, Q, V, D>
where
    B: Backend,
    Q: Qwen3VlStageSource<B>,
    Q::Error: core::fmt::Display,
    V: BooguVaeStageSource<B>,
    D: DmdDenoiser<B>,
{
    fn variant(&self) -> BooguVariant {
        self.variant
    }

    fn encode_instruction(
        &mut self,
        input: Qwen3VlModelInput<B>,
        effective_length: usize,
    ) -> Result<Tensor<B, 3>, BooguError> {
        let output = self
            .qwen
            .forward_base(&self.qwen_config, input, &mut ())
            .map_err(|error| match error {
                StreamingForwardError::Model(error) => BooguError::InvalidRequest(format!(
                    "streamed Qwen conditioning failed: {error}"
                )),
                StreamingForwardError::Source(error) => {
                    BooguError::Artifact(format!("streamed Qwen stage failed: {error}"))
                }
            })?;
        trim_instruction_features(output.last_hidden_state, effective_length)
    }

    fn encode_reference(
        &mut self,
        normalized_image: Tensor<B, 4>,
        posterior_epsilon: Tensor<B, 4>,
    ) -> Result<Tensor<B, 4>, BooguError> {
        let encoder = self.vae.load_encoder()?;
        let device = normalized_image.device();
        let latent = encode_reference(&encoder, normalized_image, posterior_epsilon)?;
        B::sync(&device).map_err(|error| {
            BooguError::Artifact(format!("VAE encoder synchronization failed: {error}"))
        })?;
        drop(encoder);
        Ok(latent)
    }

    fn denoise_with_observer<F>(
        &mut self,
        input: BooguDmdInput<B>,
        after_step: F,
    ) -> Result<Tensor<B, 4>, BooguError>
    where
        F: FnMut(usize, f32) -> Result<(), BooguError>,
    {
        run_dmd_with_observer(&mut self.denoiser, input, after_step)
    }

    fn decode(&mut self, scaled_latents: Tensor<B, 4>) -> Result<Tensor<B, 4>, BooguError> {
        let decoder = self.vae.load_decoder()?;
        let vae_dtype: DType = decoder.decoder_float_dtype().into();
        validate_tensor_dtype("VAE decoder latent", scaled_latents.dtype(), vae_dtype)?;
        let device = scaled_latents.device();
        let image = match self.decoder_memory_policy {
            VaeDecoderMemoryPolicy::BackendDefault => decoder.decode_scaled_with_group_norm_policy(
                scaled_latents,
                self.decoder_group_norm_policy,
            ),
            VaeDecoderMemoryPolicy::ExactTransientWithTailCleanup => {
                let (image, final_sync) = B::memory_persistent_allocations(
                    &device,
                    scaled_latents,
                    |scaled_latents| {
                        let image = decoder
                            .decode_scaled_with_group_norm_policy_and_tail_barrier(
                                scaled_latents,
                                self.decoder_group_norm_policy,
                                |device| {
                                    B::sync(device).map_err(|error| {
                                        BooguError::Artifact(format!(
                                            "VAE decoder pre-tail synchronization failed: {error}"
                                        ))
                                    })?;
                                    B::memory_cleanup(device);
                                    B::sync(device).map_err(|error| {
                                        BooguError::Artifact(format!(
                                            "VAE decoder pre-tail allocator cleanup synchronization failed: {error}"
                                        ))
                                    })
                                },
                            );
                        let sync = B::sync(&device).map_err(|error| {
                            BooguError::Artifact(format!(
                                "VAE decoder exact-allocation execution synchronization failed: {error}"
                            ))
                        });
                        (image, sync)
                    },
                );
                final_sync?;
                image?
            }
        };
        B::sync(&device).map_err(|error| {
            BooguError::Artifact(format!("VAE decoder synchronization failed: {error}"))
        })?;
        drop(decoder);
        Ok(image)
    }
}

#[cfg(test)]
mod tests {
    use burn::{backend::NdArray, tensor::Tensor};
    use burn_flux_vae::AutoencoderKlConfig;
    use burn_qwen3_vl::{
        Qwen3VlStageSource, Qwen3VlStreamingPlan, Qwen3VlTextConfig, Qwen3VlVisionConfig,
        RowChunkPlan, config::MropeConfig,
    };

    use super::*;

    type B = NdArray<f32>;

    struct NeverQwenSource;

    impl Qwen3VlStageSource<B> for NeverQwenSource {
        type Error = core::convert::Infallible;

        fn load_embedding_rows(
            &mut self,
            _spec: &burn_qwen3_vl::RowChunkSpec,
        ) -> Result<burn_qwen3_vl::EmbeddingRowChunk<B>, Self::Error> {
            panic!("policy propagation does not execute Qwen")
        }

        fn load_vision_prelude(
            &mut self,
        ) -> Result<burn_qwen3_vl::Qwen3VlVisionPrelude<B>, Self::Error> {
            panic!("policy propagation does not execute Qwen")
        }

        fn load_vision_block(
            &mut self,
            _index: usize,
        ) -> Result<burn_qwen3_vl::Qwen3VlVisionBlock<B>, Self::Error> {
            panic!("policy propagation does not execute Qwen")
        }

        fn load_vision_deepstack_merger(
            &mut self,
            _index: usize,
        ) -> Result<burn_qwen3_vl::Qwen3VlVisionPatchMerger<B>, Self::Error> {
            panic!("policy propagation does not execute Qwen")
        }

        fn load_vision_final_merger(
            &mut self,
        ) -> Result<burn_qwen3_vl::Qwen3VlVisionPatchMerger<B>, Self::Error> {
            panic!("policy propagation does not execute Qwen")
        }

        fn load_text_block(
            &mut self,
            _index: usize,
        ) -> Result<burn_qwen3_vl::Qwen3VlDecoderLayer<B>, Self::Error> {
            panic!("policy propagation does not execute Qwen")
        }

        fn load_text_final_norm(&mut self) -> Result<burn::nn::RmsNorm<B>, Self::Error> {
            panic!("policy propagation does not execute Qwen")
        }

        fn synchronize(&mut self) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    fn tiny_qwen_config() -> Qwen3VlConfig {
        Qwen3VlConfig {
            text_config: Qwen3VlTextConfig {
                vocab_size: 64,
                hidden_size: 8,
                intermediate_size: 16,
                num_hidden_layers: 2,
                num_attention_heads: 2,
                num_key_value_heads: 1,
                head_dim: Some(4),
                hidden_act: "silu".into(),
                rms_norm_eps: 1.0e-6,
                max_position_embeddings: 128,
                rope_theta: 10_000.0,
                rope_scaling: Some(MropeConfig {
                    mrope_section: [2, 0, 0],
                    mrope_interleaved: true,
                    rope_type: Some("default".into()),
                }),
                rope_parameters: None,
            },
            vision_config: Qwen3VlVisionConfig {
                depth: 1,
                hidden_size: 8,
                intermediate_size: 16,
                num_heads: 2,
                patch_size: 2,
                temporal_patch_size: 1,
                spatial_merge_size: 2,
                out_hidden_size: 8,
                in_channels: 3,
                num_position_embeddings: 16,
                deepstack_visual_indexes: vec![0],
                hidden_act: "gelu_pytorch_tanh".into(),
                layer_norm_eps: 1.0e-6,
            },
            tie_word_embeddings: false,
            image_token_id: 60,
            video_token_id: 61,
            vision_start_token_id: 62,
            vision_end_token_id: 63,
        }
    }

    struct CountingVaeSource {
        encoder_loads: usize,
        decoder_loads: usize,
    }

    impl BooguVaeStageSource<B> for CountingVaeSource {
        fn load_encoder(&mut self) -> Result<AutoencoderKl<B>, BooguError> {
            self.encoder_loads += 1;
            let device = Default::default();
            Ok(AutoencoderKl::new(&device, &AutoencoderKlConfig::tiny()))
        }

        fn load_decoder(&mut self) -> Result<AutoencoderKl<B>, BooguError> {
            self.decoder_loads += 1;
            let device = Default::default();
            Ok(AutoencoderKl::new(&device, &AutoencoderKlConfig::tiny()))
        }
    }

    impl AsyncBooguVaeStageSource<B> for CountingVaeSource {
        async fn load_encoder(&mut self) -> Result<AutoencoderKl<B>, BooguError> {
            BooguVaeStageSource::load_encoder(self)
        }

        async fn load_decoder(&mut self) -> Result<AutoencoderKl<B>, BooguError> {
            BooguVaeStageSource::load_decoder(self)
        }

        async fn synchronize(&mut self) -> Result<(), BooguError> {
            Ok(())
        }
    }

    fn block_on_immediate<F: core::future::Future>(future: F) -> F::Output {
        let mut future = std::pin::pin!(future);
        let waker = std::task::Waker::noop();
        let mut context = std::task::Context::from_waker(waker);
        loop {
            match future.as_mut().poll(&mut context) {
                std::task::Poll::Ready(output) => return output,
                std::task::Poll::Pending => std::thread::yield_now(),
            }
        }
    }

    #[test]
    fn retaining_vae_source_loads_each_half_once_correctness() {
        let source = CountingVaeSource {
            encoder_loads: 0,
            decoder_loads: 0,
        };
        let mut retaining = RetainingBooguVaeStageSource::new(source);

        drop(retaining.load_encoder().unwrap());
        drop(retaining.load_encoder().unwrap());
        drop(retaining.load_decoder().unwrap());
        drop(retaining.load_decoder().unwrap());

        assert_eq!(retaining.cached_stage_count(), 2);
        assert_eq!(retaining.source().encoder_loads, 1);
        assert_eq!(retaining.source().decoder_loads, 1);

        retaining.clear();
        assert_eq!(retaining.cached_stage_count(), 0);
        drop(retaining.load_decoder().unwrap());
        assert_eq!(retaining.source().decoder_loads, 2);
    }

    #[test]
    fn async_retaining_vae_source_loads_each_half_once_and_clears_correctness() {
        let source = CountingVaeSource {
            encoder_loads: 0,
            decoder_loads: 0,
        };
        let mut retaining = RetainingAsyncBooguVaeStageSource::new(source);

        block_on_immediate(retaining.load_encoder()).unwrap();
        block_on_immediate(retaining.load_encoder()).unwrap();
        block_on_immediate(retaining.load_decoder()).unwrap();
        block_on_immediate(retaining.load_decoder()).unwrap();

        assert!(retaining.retention_enabled());
        assert_eq!(retaining.cached_stage_count(), 2);
        assert_eq!(retaining.source().encoder_loads, 1);
        assert_eq!(retaining.source().decoder_loads, 1);
        retaining.clear();
        assert_eq!(retaining.cached_stage_count(), 0);
        block_on_immediate(retaining.load_decoder()).unwrap();
        assert_eq!(retaining.source().decoder_loads, 2);

        let source = CountingVaeSource {
            encoder_loads: 0,
            decoder_loads: 0,
        };
        let mut passthrough = RetainingAsyncBooguVaeStageSource::passthrough(source);
        block_on_immediate(passthrough.load_decoder()).unwrap();
        block_on_immediate(passthrough.load_decoder()).unwrap();
        assert!(!passthrough.retention_enabled());
        assert_eq!(passthrough.cached_stage_count(), 0);
        assert_eq!(passthrough.source().decoder_loads, 2);
    }

    #[test]
    fn streaming_pipeline_decoder_group_norm_policy_propagation_correctness() {
        let config = tiny_qwen_config();
        let plan =
            Qwen3VlStreamingPlan::new(&config, RowChunkPlan::even(64, 8, 2, 4).unwrap(), None)
                .unwrap();
        let qwen = StreamingQwen3Vl::<B, NeverQwenSource>::new(plan, NeverQwenSource);
        let source = CountingVaeSource {
            encoder_loads: 0,
            decoder_loads: 0,
        };

        let strict = StreamingBooguPipeline::new(
            BooguVariant::Image01Turbo,
            config.clone(),
            qwen,
            source,
            IdentityVelocity,
        );
        assert_eq!(
            strict.decoder_group_norm_policy(),
            DecoderGroupNormPolicy::StrictF32
        );
        assert_eq!(
            strict.decoder_memory_policy(),
            VaeDecoderMemoryPolicy::BackendDefault
        );

        let plan =
            Qwen3VlStreamingPlan::new(&config, RowChunkPlan::even(64, 8, 2, 4).unwrap(), None)
                .unwrap();
        let qwen = StreamingQwen3Vl::<B, NeverQwenSource>::new(plan, NeverQwenSource);
        let mixed = StreamingBooguPipeline::new(
            BooguVariant::Image01Turbo,
            config,
            qwen,
            CountingVaeSource {
                encoder_loads: 0,
                decoder_loads: 0,
            },
            IdentityVelocity,
        )
        .with_decoder_group_norm_policy(DecoderGroupNormPolicy::F16StorageF32Accum)
        .with_decoder_memory_policy(VaeDecoderMemoryPolicy::ExactTransientWithTailCleanup);
        assert_eq!(
            mixed.decoder_group_norm_policy(),
            DecoderGroupNormPolicy::F16StorageF32Accum
        );
        assert_eq!(
            mixed.decoder_memory_policy(),
            VaeDecoderMemoryPolicy::ExactTransientWithTailCleanup
        );
    }

    struct IdentityVelocity;

    impl DmdDenoiser<B> for IdentityVelocity {
        fn predict(&mut self, input: BooguDenoiserInput<B>) -> Result<Tensor<B, 4>, BooguError> {
            Ok(input.latent)
        }
    }

    struct F16VelocityMustNotDispatch;

    impl DmdDenoiser<B> for F16VelocityMustNotDispatch {
        fn execution_dtype(&self) -> Option<DType> {
            Some(DType::F16)
        }

        fn predict(&mut self, _input: BooguDenoiserInput<B>) -> Result<Tensor<B, 4>, BooguError> {
            panic!("dtype validation must run before denoiser dispatch")
        }
    }

    #[test]
    fn four_step_loop_matches_scalar_reference() {
        let device = Default::default();
        let initial = Tensor::<B, 4>::ones([1, 16, 2, 2], &device);
        let instruction = Tensor::<B, 3>::zeros([1, 2, 4096], &device);
        let noises = (0..3)
            .map(|index| Tensor::<B, 4>::full([1, 16, 2, 2], index as f32, &device))
            .collect();
        let schedule = DmdSchedule::upstream(BooguTask::Generate);
        let mut expected = 1.0_f32;
        for (index, &sigma) in schedule.sigmas().iter().enumerate() {
            expected += (1.0 - sigma) * expected;
            if let Some(&next) = schedule.sigmas().get(index + 1) {
                expected = (1.0 - next) * index as f32 + next * expected;
            }
        }
        let actual = run_dmd(
            &mut IdentityVelocity,
            BooguDmdInput {
                execution_dtype: DType::F32,
                initial_latents: initial,
                instruction,
                reference: None,
                renoise: noises,
                schedule,
            },
        )
        .unwrap()
        .to_data()
        .to_vec::<f32>()
        .unwrap();
        assert!(
            actual
                .iter()
                .all(|value| (value - expected).abs() <= 1.0e-6)
        );
    }

    #[test]
    fn loop_rejects_wrong_renoise_count_before_dispatch_correctness() {
        let device = Default::default();
        let error = run_dmd(
            &mut IdentityVelocity,
            BooguDmdInput {
                execution_dtype: DType::F32,
                initial_latents: Tensor::zeros([1, 16, 2, 2], &device),
                instruction: Tensor::zeros([1, 1, 4096], &device),
                reference: None,
                renoise: Vec::new(),
                schedule: DmdSchedule::upstream(BooguTask::Generate),
            },
        )
        .err()
        .unwrap();
        assert!(matches!(error, BooguError::InvalidRequest(_)));
    }

    #[test]
    fn loop_rejects_f32_latents_for_loaded_f16_denoiser_before_dispatch_correctness() {
        let device = Default::default();
        let error = run_dmd(
            &mut F16VelocityMustNotDispatch,
            BooguDmdInput {
                execution_dtype: DType::F16,
                initial_latents: Tensor::zeros([1, 16, 2, 2], &device),
                instruction: Tensor::zeros([1, 1, 4096], &device),
                reference: None,
                renoise: (0..3)
                    .map(|_| Tensor::zeros([1, 16, 2, 2], &device))
                    .collect(),
                schedule: DmdSchedule::upstream(BooguTask::Generate),
            },
        )
        .expect_err("F32 host tensors must not reach an F16 denoiser");
        assert!(
            error
                .to_string()
                .contains("initial latent has dtype f32, expected execution dtype f16")
        );
    }
}
