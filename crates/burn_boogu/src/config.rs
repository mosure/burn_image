use serde::{Deserialize, Serialize};

use crate::BooguError;

/// Supported Boogu checkpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BooguVariant {
    /// Boogu-Image-0.1-Turbo text-to-image checkpoint.
    Image01Turbo,
    /// Boogu-Image-0.1-Edit-Turbo 1K checkpoint.
    Image01EditTurbo,
    /// Boogu-Image-0.1-Edit-Turbo 1.5K checkpoint.
    Image01EditTurbo1k5,
}

impl BooguVariant {
    /// Whether this checkpoint consumes exactly one edit reference image.
    pub const fn is_edit(self) -> bool {
        matches!(self, Self::Image01EditTurbo | Self::Image01EditTurbo1k5)
    }
}

/// Task selected for a pipeline execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BooguTask {
    /// Generate an image from text.
    Generate,
    /// Edit a single reference image from an instruction.
    Edit,
}

/// Autotune contract attached to a parity- and performance-qualified native policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeAutotunePolicy {
    /// Shape-bucketed CubeCL tuning for interactive latency and cache reuse across nearby shapes.
    Balanced,
    /// Exhaustive CubeCL candidate search configured before WGPU device creation.
    Full,
}

/// Attention implementation attached to a parity- and performance-qualified native policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeDenoiserAttentionPolicy {
    /// Required native padded-blackbox attention with an explicit partition configuration.
    PaddedBlackbox,
}

/// Activation precision presented to the native denoiser attention kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeDenoiserAttentionPrecisionPolicy {
    /// The surrounding denoiser already executes with F16 activations.
    PreserveF16,
    /// Keep Q4 linear/residual execution in F32, but run normalized Q/K/V attention in F16.
    F32ToF16Bridge,
}

/// Denoiser normalization contract attached to a qualified native policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeDenoiserRmsNormPolicy {
    /// Preserve the accepted whole-input F32 normalization path.
    StrictF32,
}

/// Q/K preparation contract attached to a qualified native policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeDenoiserQkPreparationPolicy {
    /// Preserve the stock sequence of strict-F32 RMSNorm, RoPE, and padded-GQA preparation.
    Composed,
    /// Fuse strict-F32 RMSNorm, RoPE, GQA expansion, and padding into the F16 attention boundary.
    FusedStrictF32ToF16,
    /// Normalize Q and K in balanced native kernels before fused RoPE/GQA preparation.
    BalancedStrictQkNormRope,
}

/// VAE precision contract attached to a parity- and performance-qualified native policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeVaeExecutionPolicy {
    /// Preserve F16 storage while accumulating decoder GroupNorm reductions in F32.
    PreserveF16StorageF32GroupNorm,
}

/// Retained-Qwen synchronization contract attached to a qualified native policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeQwenSynchronizationPolicy {
    /// Drain queued work at every semantic-stage boundary.
    PerStage,
    /// Coalesce semantic-stage drains and rely on the runtime's mandatory terminal Qwen barrier.
    DeferredToStageBoundary,
}

/// Exact native high-VRAM execution controls qualified by numerical and performance gates.
///
/// This value centralizes every policy knob that must not drift between standalone runners,
/// parity gates, and production integrations. Each integration maps the typed attention, Q/K
/// preparation, VAE, and autotune contracts to its concrete runtime types and reports
/// [`Self::provenance_label`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeHighVramPolicy {
    /// Stable backend-provenance suffix naming the complete qualified policy.
    pub provenance_label: &'static str,
    /// Required CubeCL autotune policy.
    pub autotune: NativeAutotunePolicy,
    /// Required denoiser attention implementation.
    pub denoiser_attention: NativeDenoiserAttentionPolicy,
    /// Precision boundary applied immediately around the native attention kernel.
    pub denoiser_attention_precision: NativeDenoiserAttentionPrecisionPolicy,
    /// Required denoiser RMSNorm execution policy.
    pub denoiser_rms_norm: NativeDenoiserRmsNormPolicy,
    /// Required denoiser Q/K preparation implementation.
    pub denoiser_qk_preparation: NativeDenoiserQkPreparationPolicy,
    /// Required VAE storage and reduction policy.
    pub vae_execution: NativeVaeExecutionPolicy,
    /// Retained-Qwen synchronization behavior used by high-VRAM residency. Phase-streamed
    /// runtimes keep their independent per-stage source barrier.
    pub qwen_synchronization: NativeQwenSynchronizationPolicy,
    /// Streamed Qwen attention query rows per submission.
    pub qwen_query_chunk_size: usize,
    /// Native padded-blackbox denoiser query rows per submission.
    pub denoiser_query_chunk_size: usize,
    /// Staged VAE attention query rows per submission.
    pub vae_attention_query_chunk_size: usize,
    /// Cooperative-matrix attention planes.
    pub blackbox_num_planes: u8,
    /// Key/value tiles per online-softmax partition.
    pub blackbox_seq_kv_tiles: u8,
    /// Query tiles retained per plane.
    pub blackbox_seq_q_tiles: u8,
}

/// Parity- and performance-qualified native Turbo and Edit-Turbo 1K execution controls.
///
/// The kernel, dtype, and query-chunk controls are shared by the native high- and low-VRAM
/// production routes. `provenance_label` and retained-Qwen synchronization describe high-VRAM
/// residency specifically; low-VRAM provenance supplies its own streamed-per-stage label.
/// Browser, Q8, and all-F16 execution retain independent policies.
pub const BOOGU_1K_NATIVE_POLICY: NativeHighVramPolicy = NativeHighVramPolicy {
    provenance_label: "native-high-vram-retained-qwen-deferred-sync/full-autotune/1k-mixed-f16/qwen-q128/denoiser-padded-blackbox-p4-kv1-q1-split-rows-q8192-rms-strict-f32-qk-balanced-strict-norm-rope/vae-q4096-f16-storage-f32-accum",
    autotune: NativeAutotunePolicy::Full,
    denoiser_attention: NativeDenoiserAttentionPolicy::PaddedBlackbox,
    denoiser_attention_precision: NativeDenoiserAttentionPrecisionPolicy::PreserveF16,
    denoiser_rms_norm: NativeDenoiserRmsNormPolicy::StrictF32,
    denoiser_qk_preparation: NativeDenoiserQkPreparationPolicy::BalancedStrictQkNormRope,
    vae_execution: NativeVaeExecutionPolicy::PreserveF16StorageF32GroupNorm,
    qwen_synchronization: NativeQwenSynchronizationPolicy::DeferredToStageBoundary,
    qwen_query_chunk_size: 128,
    denoiser_query_chunk_size: 8_192,
    vae_attention_query_chunk_size: 4_096,
    blackbox_num_planes: 4,
    blackbox_seq_kv_tiles: 1,
    blackbox_seq_q_tiles: 1,
};

/// Parity- and performance-qualified native Edit-Turbo 1.5K execution controls.
pub const EDIT_TURBO_1K5_NATIVE_POLICY: NativeHighVramPolicy = NativeHighVramPolicy {
    provenance_label: "native-high-vram-retained-qwen-deferred-sync/full-autotune/1k5-mixed-f16/qwen-q128/denoiser-padded-blackbox-p4-kv1-q1-split-rows-q16384-rms-strict-f32-qk-composed/vae-q4096-f16-storage-f32-accum",
    autotune: NativeAutotunePolicy::Full,
    denoiser_attention: NativeDenoiserAttentionPolicy::PaddedBlackbox,
    denoiser_attention_precision: NativeDenoiserAttentionPrecisionPolicy::PreserveF16,
    denoiser_rms_norm: NativeDenoiserRmsNormPolicy::StrictF32,
    denoiser_qk_preparation: NativeDenoiserQkPreparationPolicy::Composed,
    vae_execution: NativeVaeExecutionPolicy::PreserveF16StorageF32GroupNorm,
    qwen_synchronization: NativeQwenSynchronizationPolicy::DeferredToStageBoundary,
    qwen_query_chunk_size: 128,
    denoiser_query_chunk_size: 16_384,
    vae_attention_query_chunk_size: 4_096,
    blackbox_num_planes: 4,
    blackbox_seq_kv_tiles: 1,
    blackbox_seq_q_tiles: 1,
};

/// Native packed-Q4S controls for 1K generation and editing.
///
/// Matrix and residual execution remains Q4S/F32. Only normalized Q/K/V and the attended output
/// cross the explicit F16 bridge required by the accelerated padded-blackbox attention kernel.
pub const BOOGU_Q4_1K_NATIVE_POLICY: NativeHighVramPolicy = NativeHighVramPolicy {
    provenance_label: "native-high-vram-retained-qwen-deferred-sync/full-autotune/1k-q4s-f32/qwen-q128/denoiser-q4s-f32-padded-blackbox-f16-attention-bridge-p4-kv1-q1-split-rows-q8192-rms-strict-f32-qk-fused-f32-to-f16/vae-q4096-f16-storage-f32-accum",
    autotune: NativeAutotunePolicy::Full,
    denoiser_attention: NativeDenoiserAttentionPolicy::PaddedBlackbox,
    denoiser_attention_precision: NativeDenoiserAttentionPrecisionPolicy::F32ToF16Bridge,
    denoiser_rms_norm: NativeDenoiserRmsNormPolicy::StrictF32,
    denoiser_qk_preparation: NativeDenoiserQkPreparationPolicy::FusedStrictF32ToF16,
    vae_execution: NativeVaeExecutionPolicy::PreserveF16StorageF32GroupNorm,
    qwen_synchronization: NativeQwenSynchronizationPolicy::DeferredToStageBoundary,
    qwen_query_chunk_size: 128,
    denoiser_query_chunk_size: 8_192,
    vae_attention_query_chunk_size: 4_096,
    blackbox_num_planes: 4,
    blackbox_seq_kv_tiles: 1,
    blackbox_seq_q_tiles: 1,
};

/// Native packed-Q4S controls for Edit-Turbo 1.5K.
pub const EDIT_TURBO_1K5_Q4_NATIVE_POLICY: NativeHighVramPolicy = NativeHighVramPolicy {
    provenance_label: "native-high-vram-retained-qwen-deferred-sync/full-autotune/1k5-q4s-f32/qwen-q128/denoiser-q4s-f32-padded-blackbox-f16-attention-bridge-p4-kv1-q1-split-rows-q16384-rms-strict-f32-qk-fused-f32-to-f16/vae-q4096-f16-storage-f32-accum",
    autotune: NativeAutotunePolicy::Full,
    denoiser_attention: NativeDenoiserAttentionPolicy::PaddedBlackbox,
    denoiser_attention_precision: NativeDenoiserAttentionPrecisionPolicy::F32ToF16Bridge,
    denoiser_rms_norm: NativeDenoiserRmsNormPolicy::StrictF32,
    denoiser_qk_preparation: NativeDenoiserQkPreparationPolicy::FusedStrictF32ToF16,
    vae_execution: NativeVaeExecutionPolicy::PreserveF16StorageF32GroupNorm,
    qwen_synchronization: NativeQwenSynchronizationPolicy::DeferredToStageBoundary,
    qwen_query_chunk_size: 128,
    denoiser_query_chunk_size: 16_384,
    vae_attention_query_chunk_size: 4_096,
    blackbox_num_planes: 4,
    blackbox_seq_kv_tiles: 1,
    blackbox_seq_q_tiles: 1,
};

/// Exact configuration of the released 10B Boogu denoiser.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BooguConfig {
    /// Latent patch edge.
    pub patch_size: usize,
    /// VAE latent channels.
    pub in_channels: usize,
    /// Output latent channels.
    pub out_channels: usize,
    /// Transformer width.
    pub hidden_size: usize,
    /// Total dual plus single stream layers.
    pub num_layers: usize,
    /// Leading dual-stream layers.
    pub num_double_stream_layers: usize,
    /// Layers in each context/noise/reference refiner.
    pub num_refiner_layers: usize,
    /// Query heads.
    pub num_attention_heads: usize,
    /// Key/value heads.
    pub num_kv_heads: usize,
    /// FFN width alignment.
    pub multiple_of: usize,
    /// Normalization epsilon.
    pub norm_eps: f64,
    /// Per-axis rotary dimensions.
    pub axes_dim_rope: [usize; 3],
    /// Per-axis rotary table lengths.
    pub axes_lens: [usize; 3],
    /// Qwen feature width.
    pub instruction_feature_dim: usize,
    /// Timestep sinusoid scale.
    pub timestep_scale: f64,
}

impl Default for BooguConfig {
    fn default() -> Self {
        Self {
            patch_size: 2,
            in_channels: 16,
            out_channels: 16,
            hidden_size: 3360,
            num_layers: 40,
            num_double_stream_layers: 8,
            num_refiner_layers: 2,
            num_attention_heads: 28,
            num_kv_heads: 7,
            multiple_of: 256,
            norm_eps: 1.0e-5,
            axes_dim_rope: [40, 40, 40],
            axes_lens: [2048, 1664, 1664],
            instruction_feature_dim: 4096,
            timestep_scale: 1000.0,
        }
    }
}

impl BooguConfig {
    /// Number of joint single-stream layers.
    pub const fn num_single_stream_layers(&self) -> usize {
        self.num_layers - self.num_double_stream_layers
    }

    /// Dimension of one attention head.
    pub const fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Rounded SwiGLU intermediate width.
    pub fn ffn_inner_dim(&self) -> usize {
        let requested = 4 * self.hidden_size;
        self.multiple_of * requested.div_ceil(self.multiple_of)
    }

    /// Validate invariants required by the upstream implementation.
    pub fn validate(&self) -> Result<(), BooguError> {
        if self.patch_size == 0 || self.in_channels == 0 || self.out_channels == 0 {
            return Err(BooguError::InvalidConfig(
                "patch and latent channel counts must be non-zero".into(),
            ));
        }
        if self.num_attention_heads == 0 || self.num_kv_heads == 0 {
            return Err(BooguError::InvalidConfig(
                "attention head counts must be non-zero".into(),
            ));
        }
        if !self.num_attention_heads.is_multiple_of(self.num_kv_heads) {
            return Err(BooguError::InvalidConfig(
                "query heads must be divisible by KV heads".into(),
            ));
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(BooguError::InvalidConfig(
                "hidden size must be divisible by query heads".into(),
            ));
        }
        if self.head_dim() != self.axes_dim_rope.iter().sum::<usize>() {
            return Err(BooguError::InvalidConfig(
                "head dimension must equal the sum of RoPE axis dimensions".into(),
            ));
        }
        if self.num_double_stream_layers > self.num_layers {
            return Err(BooguError::InvalidConfig(
                "dual-stream layers exceed total layers".into(),
            ));
        }
        if self.axes_dim_rope.iter().any(|dim| !dim.is_multiple_of(2)) {
            return Err(BooguError::InvalidConfig(
                "each RoPE dimension must be even".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn released_config_correctness() {
        let config = BooguConfig::default();
        config.validate().unwrap();
        assert_eq!(config.head_dim(), 120);
        assert_eq!(config.num_single_stream_layers(), 32);
        assert_eq!(config.ffn_inner_dim(), 13_568);
    }

    #[test]
    fn qualified_native_policy_controls_are_exact_correctness() {
        assert_eq!(BOOGU_1K_NATIVE_POLICY.qwen_query_chunk_size, 128);
        assert_eq!(BOOGU_1K_NATIVE_POLICY.denoiser_query_chunk_size, 8_192);
        assert_eq!(BOOGU_1K_NATIVE_POLICY.vae_attention_query_chunk_size, 4_096);
        assert_eq!(BOOGU_1K_NATIVE_POLICY.blackbox_num_planes, 4);
        assert_eq!(BOOGU_1K_NATIVE_POLICY.blackbox_seq_kv_tiles, 1);
        assert_eq!(BOOGU_1K_NATIVE_POLICY.blackbox_seq_q_tiles, 1);
        assert_eq!(
            BOOGU_1K_NATIVE_POLICY.provenance_label,
            "native-high-vram-retained-qwen-deferred-sync/full-autotune/1k-mixed-f16/qwen-q128/denoiser-padded-blackbox-p4-kv1-q1-split-rows-q8192-rms-strict-f32-qk-balanced-strict-norm-rope/vae-q4096-f16-storage-f32-accum"
        );
        assert_eq!(BOOGU_1K_NATIVE_POLICY.autotune, NativeAutotunePolicy::Full);
        assert_eq!(
            BOOGU_1K_NATIVE_POLICY.denoiser_attention,
            NativeDenoiserAttentionPolicy::PaddedBlackbox
        );
        assert_eq!(
            BOOGU_1K_NATIVE_POLICY.denoiser_attention_precision,
            NativeDenoiserAttentionPrecisionPolicy::PreserveF16
        );
        assert_eq!(
            BOOGU_1K_NATIVE_POLICY.denoiser_rms_norm,
            NativeDenoiserRmsNormPolicy::StrictF32
        );
        assert_eq!(
            BOOGU_1K_NATIVE_POLICY.denoiser_qk_preparation,
            NativeDenoiserQkPreparationPolicy::BalancedStrictQkNormRope
        );
        assert_eq!(
            BOOGU_1K_NATIVE_POLICY.vae_execution,
            NativeVaeExecutionPolicy::PreserveF16StorageF32GroupNorm
        );
        assert_eq!(
            BOOGU_1K_NATIVE_POLICY.qwen_synchronization,
            NativeQwenSynchronizationPolicy::DeferredToStageBoundary
        );
        assert_eq!(EDIT_TURBO_1K5_NATIVE_POLICY.qwen_query_chunk_size, 128);
        assert_eq!(
            EDIT_TURBO_1K5_NATIVE_POLICY.denoiser_query_chunk_size,
            16_384
        );
        assert_eq!(
            EDIT_TURBO_1K5_NATIVE_POLICY.vae_attention_query_chunk_size,
            4_096
        );
        assert_eq!(EDIT_TURBO_1K5_NATIVE_POLICY.blackbox_num_planes, 4);
        assert_eq!(EDIT_TURBO_1K5_NATIVE_POLICY.blackbox_seq_kv_tiles, 1);
        assert_eq!(EDIT_TURBO_1K5_NATIVE_POLICY.blackbox_seq_q_tiles, 1);
        assert_eq!(
            EDIT_TURBO_1K5_NATIVE_POLICY.denoiser_attention_precision,
            NativeDenoiserAttentionPrecisionPolicy::PreserveF16
        );
        assert_eq!(
            EDIT_TURBO_1K5_NATIVE_POLICY.denoiser_rms_norm,
            NativeDenoiserRmsNormPolicy::StrictF32
        );
        assert_eq!(
            EDIT_TURBO_1K5_NATIVE_POLICY.denoiser_qk_preparation,
            NativeDenoiserQkPreparationPolicy::Composed
        );
        assert_eq!(
            EDIT_TURBO_1K5_NATIVE_POLICY.provenance_label,
            "native-high-vram-retained-qwen-deferred-sync/full-autotune/1k5-mixed-f16/qwen-q128/denoiser-padded-blackbox-p4-kv1-q1-split-rows-q16384-rms-strict-f32-qk-composed/vae-q4096-f16-storage-f32-accum"
        );
        assert_eq!(
            EDIT_TURBO_1K5_NATIVE_POLICY.qwen_synchronization,
            NativeQwenSynchronizationPolicy::DeferredToStageBoundary
        );
        assert_eq!(
            BOOGU_Q4_1K_NATIVE_POLICY.denoiser_attention_precision,
            NativeDenoiserAttentionPrecisionPolicy::F32ToF16Bridge
        );
        assert_eq!(
            BOOGU_Q4_1K_NATIVE_POLICY.denoiser_qk_preparation,
            NativeDenoiserQkPreparationPolicy::FusedStrictF32ToF16
        );
        assert_eq!(
            EDIT_TURBO_1K5_Q4_NATIVE_POLICY.denoiser_attention_precision,
            NativeDenoiserAttentionPrecisionPolicy::F32ToF16Bridge
        );
        assert_eq!(
            EDIT_TURBO_1K5_Q4_NATIVE_POLICY.denoiser_qk_preparation,
            NativeDenoiserQkPreparationPolicy::FusedStrictF32ToF16
        );
        assert_eq!(
            EDIT_TURBO_1K5_Q4_NATIVE_POLICY.denoiser_query_chunk_size,
            16_384
        );
    }
}
