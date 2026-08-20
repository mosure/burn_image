use burn::{nn, prelude::*};

use super::attention::{AttentionKernel, PortableChunkedAttention};
use super::norm::{DenoiserRmsNormPolicy, rms_norm_with_policy, rms_normalized_with_policy};
use super::{DoubleStreamAttention, GqaAttention, LuminaFeedForward, RmsNormZero};

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
use super::attention::NativeFlashUnitAttention;
#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
use super::native_flash::NativeWgpuBackend;

/// A modulated or unmodulated single-stream transformer/refiner block.
#[derive(Module, Debug)]
pub struct SingleStreamBlock<B: Backend> {
    /// Self-attention module.
    pub attn: GqaAttention<B>,
    /// Feed-forward module.
    pub feed_forward: LuminaFeedForward<B>,
    /// Optional adaptive normalization for modulated blocks.
    pub norm1: Option<RmsNormZero<B>>,
    /// Ordinary input norm for context refinement.
    pub plain_norm1: Option<nn::RmsNorm<B>>,
    /// Post-attention norm.
    pub norm2: nn::RmsNorm<B>,
    /// Pre-FFN norm.
    pub ffn_norm1: nn::RmsNorm<B>,
    /// Post-FFN norm.
    pub ffn_norm2: nn::RmsNorm<B>,
}

impl<B: Backend> SingleStreamBlock<B> {
    /// Create a block matching Boogu's Lumina-style layout.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        width: usize,
        inner: usize,
        heads: usize,
        kv_heads: usize,
        conditioning_width: usize,
        epsilon: f64,
        modulation: bool,
        device: &B::Device,
    ) -> Self {
        let norm = || {
            nn::RmsNormConfig::new(width)
                .with_epsilon(epsilon)
                .init(device)
        };
        Self {
            attn: GqaAttention::new(width, heads, kv_heads, 1.0e-5, device),
            feed_forward: LuminaFeedForward::new(width, inner, device),
            norm1: modulation.then(|| RmsNormZero::new(width, conditioning_width, epsilon, device)),
            plain_norm1: (!modulation).then(norm),
            norm2: norm(),
            ffn_norm1: norm(),
            ffn_norm2: norm(),
        }
    }

    /// Apply one block.
    pub fn forward(
        &self,
        tokens: Tensor<B, 3>,
        rope: Option<(Tensor<B, 3>, Tensor<B, 3>)>,
        conditioning: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        self.forward_with_kernel::<PortableChunkedAttention>(tokens, rope, conditioning)
    }

    pub(crate) fn forward_with_kernel<K: AttentionKernel<B>>(
        &self,
        tokens: Tensor<B, 3>,
        rope: Option<(Tensor<B, 3>, Tensor<B, 3>)>,
        conditioning: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        self.forward_with_kernel_and_rms_norm_policy::<K>(
            tokens,
            rope,
            conditioning,
            DenoiserRmsNormPolicy::StrictF32,
        )
    }

    pub(crate) fn forward_with_kernel_and_rms_norm_policy<K: AttentionKernel<B>>(
        &self,
        tokens: Tensor<B, 3>,
        rope: Option<(Tensor<B, 3>, Tensor<B, 3>)>,
        conditioning: Option<Tensor<B, 2>>,
        rms_norm_policy: DenoiserRmsNormPolicy,
    ) -> Tensor<B, 3> {
        if let Some(norm1) = &self.norm1 {
            let (normalized, gate_attn, scale_mlp, gate_mlp) = norm1.forward_with_policy(
                tokens.clone(),
                conditioning.expect("modulated Boogu block requires timestep embedding"),
                rms_norm_policy,
            );
            let attended = self.attn.forward_with_kernel_and_rms_norm_policy::<K>(
                normalized.clone(),
                normalized,
                rope,
                rms_norm_policy,
            );
            let tokens = tokens
                + rms_norm_with_policy(&self.norm2, attended, rms_norm_policy)
                    * gate_attn.tanh().unsqueeze_dim(1);
            let mlp_in = rms_norm_with_policy(&self.ffn_norm1, tokens.clone(), rms_norm_policy)
                * (scale_mlp.unsqueeze_dim(1) + 1.0);
            let mlp = self.feed_forward.forward(mlp_in);
            tokens
                + rms_norm_with_policy(&self.ffn_norm2, mlp, rms_norm_policy)
                    * gate_mlp.tanh().unsqueeze_dim(1)
        } else {
            let normalized = rms_norm_with_policy(
                self.plain_norm1
                    .as_ref()
                    .expect("plain block must have an input norm"),
                tokens.clone(),
                rms_norm_policy,
            );
            let attended = self.attn.forward_with_kernel_and_rms_norm_policy::<K>(
                normalized.clone(),
                normalized,
                rope,
                rms_norm_policy,
            );
            let tokens = tokens + rms_norm_with_policy(&self.norm2, attended, rms_norm_policy);
            let mlp = self.feed_forward.forward(rms_norm_with_policy(
                &self.ffn_norm1,
                tokens.clone(),
                rms_norm_policy,
            ));
            tokens + rms_norm_with_policy(&self.ffn_norm2, mlp, rms_norm_policy)
        }
    }
}

/// Leading block that retains separate instruction and image streams.
#[derive(Module, Debug)]
pub struct DoubleStreamBlock<B: Backend> {
    /// Joint instruction/image attention.
    pub joint_attn: DoubleStreamAttention<B>,
    /// Image-only self attention.
    pub image_self_attn: GqaAttention<B>,
    /// Image FFN.
    pub image_ffn: LuminaFeedForward<B>,
    /// Instruction FFN.
    pub instruction_ffn: LuminaFeedForward<B>,
    /// Image joint-attention modulation.
    pub image_norm1: RmsNormZero<B>,
    /// Image MLP modulation.
    pub image_norm2: RmsNormZero<B>,
    /// Image self-attention modulation.
    pub image_norm3: RmsNormZero<B>,
    /// Instruction attention modulation.
    pub instruction_norm1: RmsNormZero<B>,
    /// Instruction MLP modulation.
    pub instruction_norm2: RmsNormZero<B>,
    /// Image post-joint norm.
    pub image_attn_norm: nn::RmsNorm<B>,
    /// Image post-self norm.
    pub image_self_norm: nn::RmsNorm<B>,
    /// Image pre-FFN norm.
    pub image_ffn_norm1: nn::RmsNorm<B>,
    /// Image post-FFN norm.
    pub image_ffn_norm2: nn::RmsNorm<B>,
    /// Instruction post-attention norm.
    pub instruction_attn_norm: nn::RmsNorm<B>,
    /// Instruction pre-FFN norm.
    pub instruction_ffn_norm1: nn::RmsNorm<B>,
    /// Instruction post-FFN norm.
    pub instruction_ffn_norm2: nn::RmsNorm<B>,
}

impl<B: Backend> DoubleStreamBlock<B> {
    /// Create a dual-stream block.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        width: usize,
        inner: usize,
        heads: usize,
        kv_heads: usize,
        conditioning_width: usize,
        epsilon: f64,
        device: &B::Device,
    ) -> Self {
        let rms = || {
            nn::RmsNormConfig::new(width)
                .with_epsilon(epsilon)
                .init(device)
        };
        let adaptive = || RmsNormZero::new(width, conditioning_width, epsilon, device);
        Self {
            joint_attn: DoubleStreamAttention::new(width, heads, kv_heads, 1.0e-5, device),
            image_self_attn: GqaAttention::new(width, heads, kv_heads, 1.0e-5, device),
            image_ffn: LuminaFeedForward::new(width, inner, device),
            instruction_ffn: LuminaFeedForward::new(width, inner, device),
            image_norm1: adaptive(),
            image_norm2: adaptive(),
            image_norm3: adaptive(),
            instruction_norm1: adaptive(),
            instruction_norm2: adaptive(),
            image_attn_norm: rms(),
            image_self_norm: rms(),
            image_ffn_norm1: rms(),
            image_ffn_norm2: rms(),
            instruction_attn_norm: rms(),
            instruction_ffn_norm1: rms(),
            instruction_ffn_norm2: rms(),
        }
    }

    /// Apply the exact dual-stream residual topology for uniform, unpadded batches.
    pub fn forward(
        &self,
        image: Tensor<B, 3>,
        instruction: Tensor<B, 3>,
        image_rope: (Tensor<B, 3>, Tensor<B, 3>),
        joint_rope: (Tensor<B, 3>, Tensor<B, 3>),
        conditioning: Tensor<B, 2>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        self.forward_with_kernel::<PortableChunkedAttention>(
            image,
            instruction,
            image_rope,
            joint_rope,
            conditioning,
        )
    }

    pub(crate) fn forward_with_kernel<K: AttentionKernel<B>>(
        &self,
        image: Tensor<B, 3>,
        instruction: Tensor<B, 3>,
        image_rope: (Tensor<B, 3>, Tensor<B, 3>),
        joint_rope: (Tensor<B, 3>, Tensor<B, 3>),
        conditioning: Tensor<B, 2>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        self.forward_with_kernel_and_rms_norm_policy::<K>(
            image,
            instruction,
            image_rope,
            joint_rope,
            conditioning,
            DenoiserRmsNormPolicy::StrictF32,
        )
    }

    pub(crate) fn forward_with_kernel_and_rms_norm_policy<K: AttentionKernel<B>>(
        &self,
        image: Tensor<B, 3>,
        instruction: Tensor<B, 3>,
        image_rope: (Tensor<B, 3>, Tensor<B, 3>),
        joint_rope: (Tensor<B, 3>, Tensor<B, 3>),
        conditioning: Tensor<B, 2>,
        rms_norm_policy: DenoiserRmsNormPolicy,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        assert_eq!(
            self.image_norm1.norm.epsilon, self.image_norm2.norm.epsilon,
            "shared image RMS normalization requires equal epsilons"
        );
        assert_eq!(
            self.image_norm1.norm.epsilon, self.image_norm3.norm.epsilon,
            "shared image RMS normalization requires equal epsilons"
        );
        assert_eq!(
            self.instruction_norm1.norm.epsilon, self.instruction_norm2.norm.epsilon,
            "shared instruction RMS normalization requires equal epsilons"
        );
        let image_rms = rms_normalized_with_policy(
            image.clone(),
            self.image_norm1.norm.epsilon,
            rms_norm_policy,
        );
        let instruction_rms = rms_normalized_with_policy(
            instruction.clone(),
            self.instruction_norm1.norm.epsilon,
            rms_norm_policy,
        );
        let (image_joint, image_gate_attn, image_scale_mlp, image_gate_mlp) = self
            .image_norm1
            .forward_from_rms_normalized(image_rms.clone(), conditioning.clone());
        let (image_mlp_base, image_shift_mlp, _, _) = self
            .image_norm2
            .forward_from_rms_normalized(image_rms.clone(), conditioning.clone());
        let (image_self, image_gate_self, _, _) = self
            .image_norm3
            .forward_from_rms_normalized(image_rms, conditioning.clone());
        let (instruction_joint, instruction_gate_attn, instruction_scale_mlp, instruction_gate_mlp) =
            self.instruction_norm1
                .forward_from_rms_normalized(instruction_rms.clone(), conditioning.clone());
        let (instruction_mlp_base, instruction_shift_mlp, _, _) = self
            .instruction_norm2
            .forward_from_rms_normalized(instruction_rms, conditioning);

        let (image_attn, instruction_attn) = self
            .joint_attn
            .forward_with_kernel_and_rms_norm_policy::<K>(
                image_joint,
                instruction_joint,
                joint_rope,
                rms_norm_policy,
            );
        let image_self_output = self
            .image_self_attn
            .forward_with_kernel_and_rms_norm_policy::<K>(
                image_self.clone(),
                image_self,
                Some(image_rope),
                rms_norm_policy,
            );

        let image = image
            + rms_norm_with_policy(&self.image_attn_norm, image_attn, rms_norm_policy)
                * image_gate_attn.tanh().unsqueeze_dim(1);
        let image = image
            + rms_norm_with_policy(&self.image_self_norm, image_self_output, rms_norm_policy)
                * image_gate_self.tanh().unsqueeze_dim(1);
        let image_mlp_input = image_mlp_base * (image_scale_mlp.unsqueeze_dim(1) + 1.0)
            + image_shift_mlp.unsqueeze_dim(1);
        let image_mlp = self.image_ffn.forward(rms_norm_with_policy(
            &self.image_ffn_norm1,
            image_mlp_input,
            rms_norm_policy,
        ));
        let image = image
            + rms_norm_with_policy(&self.image_ffn_norm2, image_mlp, rms_norm_policy)
                * image_gate_mlp.tanh().unsqueeze_dim(1);

        let instruction = instruction
            + rms_norm_with_policy(
                &self.instruction_attn_norm,
                instruction_attn,
                rms_norm_policy,
            ) * instruction_gate_attn.tanh().unsqueeze_dim(1);
        let instruction_mlp_input = instruction_mlp_base
            * (instruction_scale_mlp.unsqueeze_dim(1) + 1.0)
            + instruction_shift_mlp.unsqueeze_dim(1);
        let instruction_mlp = self.instruction_ffn.forward(rms_norm_with_policy(
            &self.instruction_ffn_norm1,
            instruction_mlp_input,
            rms_norm_policy,
        ));
        let instruction = instruction
            + rms_norm_with_policy(
                &self.instruction_ffn_norm2,
                instruction_mlp,
                rms_norm_policy,
            ) * instruction_gate_mlp.tanh().unsqueeze_dim(1);

        (image, instruction)
    }
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
impl SingleStreamBlock<NativeWgpuBackend> {
    /// Apply one block using required native Cubek `FlashUnit` attention.
    pub fn forward_native_flash_unit(
        &self,
        tokens: Tensor<NativeWgpuBackend, 3>,
        rope: Option<(Tensor<NativeWgpuBackend, 3>, Tensor<NativeWgpuBackend, 3>)>,
        conditioning: Option<Tensor<NativeWgpuBackend, 2>>,
    ) -> Tensor<NativeWgpuBackend, 3> {
        self.forward_with_kernel::<NativeFlashUnitAttention>(tokens, rope, conditioning)
    }
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
impl DoubleStreamBlock<NativeWgpuBackend> {
    /// Apply the dual-stream block using required native Cubek `FlashUnit` attention.
    pub fn forward_native_flash_unit(
        &self,
        image: Tensor<NativeWgpuBackend, 3>,
        instruction: Tensor<NativeWgpuBackend, 3>,
        image_rope: (Tensor<NativeWgpuBackend, 3>, Tensor<NativeWgpuBackend, 3>),
        joint_rope: (Tensor<NativeWgpuBackend, 3>, Tensor<NativeWgpuBackend, 3>),
        conditioning: Tensor<NativeWgpuBackend, 2>,
    ) -> (Tensor<NativeWgpuBackend, 3>, Tensor<NativeWgpuBackend, 3>) {
        self.forward_with_kernel::<NativeFlashUnitAttention>(
            image,
            instruction,
            image_rope,
            joint_rope,
            conditioning,
        )
    }
}
