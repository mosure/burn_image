use burn::{nn, prelude::*, tensor::TensorData};

use crate::{
    BooguConfig, BooguError,
    latent::{patchify, unpatchify},
    rope::{LatentSize, position_ids},
};

use super::{
    CombinedTimestepCaptionEmbedding, DoubleStreamBlock, FinalProjection, SingleStreamBlock,
    attention::{AttentionKernel, PortableChunkedAttention},
    linear::linear_forward,
    norm::DenoiserRmsNormPolicy,
};

/// Select the one edit-image index row while it is still packed, then widen only that row.
///
/// Q4 releases retain this embedding as a quantized table. Adding a selected `QFloat` row to the
/// floating reference activation is invalid, while widening the entire table would add avoidable
/// traffic to every edit request. Floating releases follow the same path because `dequantize` is
/// an identity for ordinary float tensors.
pub(super) fn selected_image_index_embedding<B: Backend>(
    embedding: &nn::Embedding<B>,
    hidden_size: usize,
    dtype: burn::tensor::DType,
) -> Tensor<B, 3> {
    embedding
        .weight
        .val()
        .clone()
        .slice([0..1, 0..hidden_size])
        .dequantize()
        .cast(dtype)
        .reshape([1, 1, hidden_size])
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
use super::attention::NativeFlashUnitAttention;
#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
use super::attention::SplitDoubleStreamSharedProjection;
#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
use super::attention::{NativeF32ToF16PaddedBlackboxAttention, NativePaddedBlackboxAttention};
#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
use super::native_flash::NativeWgpuBackend;
#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
use super::native_flash::{
    assert_supported_wgpu_blackbox_configuration,
    assert_supported_wgpu_blackbox_partition_configuration,
};

/// Uniform batch input to the first production Boogu denoiser path.
pub struct BooguDenoiserInput<B: Backend> {
    /// Generated-noise latent `[B,16,H,W]`.
    pub latent: Tensor<B, 4>,
    /// Scalar DMD sigma per batch item.
    pub timestep: Tensor<B, 1>,
    /// Trimmed Qwen instruction hidden state `[B,T,4096]`.
    pub instruction: Tensor<B, 3>,
    /// Optional single edit reference latent `[B,16,Hr,Wr]`.
    pub reference: Option<Tensor<B, 4>>,
}

/// Exact shape/dtype identity of a reusable denoiser RoPE table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BooguRoPeKey {
    text_len: usize,
    reference_size: Option<LatentSize>,
    generated_size: LatentSize,
    dtype: burn::tensor::DType,
}

/// Device-resident RoPE tensors and their exact position layout.
#[derive(Clone)]
pub(crate) struct BooguRoPeGeometry<B: Backend> {
    key: BooguRoPeKey,
    pub(crate) reference_len: usize,
    pub(crate) generated_len: usize,
    pub(crate) joint_cos: Tensor<B, 3>,
    pub(crate) joint_sin: Tensor<B, 3>,
}

impl<B: Backend> BooguRoPeGeometry<B> {
    pub(crate) fn prepare(
        config: &BooguConfig,
        input: &BooguDenoiserInput<B>,
    ) -> Result<Self, BooguError> {
        let key = BooguRoPeKey::from_input(input);
        let ids = position_ids(
            key.text_len,
            key.reference_size.as_slice(),
            key.generated_size,
            config.patch_size,
        )?;
        let device = input.latent.device();
        let (joint_cos, joint_sin) = rope_tensors::<B>(
            &ids.values,
            config.axes_dim_rope,
            10_000.0,
            key.dtype,
            &device,
        );
        Ok(Self {
            key,
            reference_len: ids.reference_len,
            generated_len: ids.generated_len,
            joint_cos,
            joint_sin,
        })
    }

    /// Whether these tensors exactly describe the supplied input on the same device.
    pub(crate) fn matches(&self, input: &BooguDenoiserInput<B>) -> bool {
        self.key == BooguRoPeKey::from_input(input)
            && self.joint_cos.device() == input.latent.device()
    }
}

impl BooguRoPeKey {
    fn from_input<B: Backend>(input: &BooguDenoiserInput<B>) -> Self {
        let [_batch, _channels, height, width] = input.latent.dims();
        let reference_size = input.reference.as_ref().map(|reference| {
            let dims = reference.dims();
            LatentSize {
                height: dims[2],
                width: dims[3],
            }
        });
        Self {
            text_len: input.instruction.dims()[1],
            reference_size,
            generated_size: LatentSize { height, width },
            dtype: input.latent.dtype(),
        }
    }
}

/// Mixed dual/single-stream Boogu diffusion transformer.
#[derive(Module, Debug)]
pub struct BooguDenoiser<B: Backend> {
    /// Generated latent patch projection.
    pub x_embedder: nn::Linear<B>,
    /// Reference latent patch projection.
    pub ref_image_patch_embedder: nn::Linear<B>,
    /// Timestep/caption embeddings.
    pub time_caption_embed: CombinedTimestepCaptionEmbedding<B>,
    /// Context refiners.
    pub context_refiner: Vec<SingleStreamBlock<B>>,
    /// Generated latent refiners.
    pub noise_refiner: Vec<SingleStreamBlock<B>>,
    /// Reference latent refiners.
    pub ref_image_refiner: Vec<SingleStreamBlock<B>>,
    /// Leading dual-stream blocks.
    pub double_stream_layers: Vec<DoubleStreamBlock<B>>,
    /// Joint single-stream blocks.
    pub single_stream_layers: Vec<SingleStreamBlock<B>>,
    /// Reference-image index embeddings.
    pub image_index_embedding: nn::Embedding<B>,
    /// Final norm/projection.
    pub norm_out: FinalProjection<B>,
    config: BooguConfig,
}

impl<B: Backend> BooguDenoiser<B> {
    /// Declare a model with lazy parameters populated later from verified artifacts.
    pub fn new(config: BooguConfig, device: &B::Device) -> Result<Self, BooguError> {
        config.validate()?;
        let width = config.hidden_size;
        let patch_width = config.patch_size * config.patch_size * config.in_channels;
        let inner = config.ffn_inner_dim();
        let conditioning_width = width.min(1024);
        let make_refiner = |modulation| {
            SingleStreamBlock::new(
                width,
                inner,
                config.num_attention_heads,
                config.num_kv_heads,
                conditioning_width,
                config.norm_eps,
                modulation,
                device,
            )
        };
        Ok(Self {
            x_embedder: nn::LinearConfig::new(patch_width, width).init(device),
            ref_image_patch_embedder: nn::LinearConfig::new(patch_width, width).init(device),
            time_caption_embed: CombinedTimestepCaptionEmbedding::new(
                width,
                config.instruction_feature_dim,
                256,
                conditioning_width,
                config.norm_eps,
                config.timestep_scale,
                device,
            ),
            context_refiner: (0..config.num_refiner_layers)
                .map(|_| make_refiner(false))
                .collect(),
            noise_refiner: (0..config.num_refiner_layers)
                .map(|_| make_refiner(true))
                .collect(),
            ref_image_refiner: (0..config.num_refiner_layers)
                .map(|_| make_refiner(true))
                .collect(),
            double_stream_layers: (0..config.num_double_stream_layers)
                .map(|_| {
                    DoubleStreamBlock::new(
                        width,
                        inner,
                        config.num_attention_heads,
                        config.num_kv_heads,
                        conditioning_width,
                        config.norm_eps,
                        device,
                    )
                })
                .collect(),
            single_stream_layers: (0..config.num_single_stream_layers())
                .map(|_| make_refiner(true))
                .collect(),
            image_index_embedding: nn::EmbeddingConfig::new(5, width).init(device),
            norm_out: FinalProjection::new(
                width,
                conditioning_width,
                config.patch_size * config.patch_size * config.out_channels,
                1.0e-6,
                device,
            ),
            config,
        })
    }

    /// Immutable architecture configuration used to build this denoiser.
    pub const fn config(&self) -> &BooguConfig {
        &self.config
    }

    /// Set the query tile used by every denoiser attention module.
    ///
    /// The checkpoint inventory is unaffected because this is an execution policy rather than a
    /// learned parameter. The default remains the conservative bounded value selected by each
    /// attention module.
    pub fn set_attention_query_chunk_size(&mut self, query_chunk_size: usize) {
        assert!(
            query_chunk_size > 0,
            "attention query chunk must be non-zero"
        );
        for block in &mut self.context_refiner {
            block.attn.set_query_chunk_size(query_chunk_size);
        }
        for block in &mut self.noise_refiner {
            block.attn.set_query_chunk_size(query_chunk_size);
        }
        for block in &mut self.ref_image_refiner {
            block.attn.set_query_chunk_size(query_chunk_size);
        }
        for block in &mut self.double_stream_layers {
            block.joint_attn.set_query_chunk_size(query_chunk_size);
            block.image_self_attn.set_query_chunk_size(query_chunk_size);
        }
        for block in &mut self.single_stream_layers {
            block.attn.set_query_chunk_size(query_chunk_size);
        }
    }

    /// Execute the uniform, batch-one parity path.
    pub fn forward(&self, input: BooguDenoiserInput<B>) -> Result<Tensor<B, 4>, BooguError> {
        self.forward_with_kernel::<PortableChunkedAttention>(input)
    }

    pub(crate) fn forward_with_kernel<K: AttentionKernel<B>>(
        &self,
        input: BooguDenoiserInput<B>,
    ) -> Result<Tensor<B, 4>, BooguError> {
        let geometry = self.prepare_rope_geometry(&input)?;
        self.forward_with_kernel_and_rope::<K>(input, &geometry)
    }

    /// Build the step-invariant joint RoPE table for one exact input geometry.
    pub(crate) fn prepare_rope_geometry(
        &self,
        input: &BooguDenoiserInput<B>,
    ) -> Result<BooguRoPeGeometry<B>, BooguError> {
        self.validate_input(input)?;
        BooguRoPeGeometry::prepare(&self.config, input)
    }

    /// Execute with a previously prepared, exact-match RoPE table.
    pub(crate) fn forward_with_kernel_and_rope<K: AttentionKernel<B>>(
        &self,
        input: BooguDenoiserInput<B>,
        geometry: &BooguRoPeGeometry<B>,
    ) -> Result<Tensor<B, 4>, BooguError> {
        self.forward_with_kernel_and_rope_and_rms_norm_policy::<K>(
            input,
            geometry,
            DenoiserRmsNormPolicy::StrictF32,
        )
    }

    /// Execute portable bounded attention with a caller-retained, exact-match RoPE table.
    ///
    /// Native DMD adapters use this seam to keep step-invariant trigonometric tables on device
    /// across all four predictions instead of rebuilding and uploading them on every step.
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    pub(crate) fn forward_with_prepared_rope(
        &self,
        input: BooguDenoiserInput<B>,
        geometry: &BooguRoPeGeometry<B>,
    ) -> Result<Tensor<B, 4>, BooguError> {
        self.forward_with_kernel_and_rope::<PortableChunkedAttention>(input, geometry)
    }

    pub(crate) fn forward_with_kernel_and_rope_and_rms_norm_policy<K: AttentionKernel<B>>(
        &self,
        input: BooguDenoiserInput<B>,
        geometry: &BooguRoPeGeometry<B>,
        rms_norm_policy: DenoiserRmsNormPolicy,
    ) -> Result<Tensor<B, 4>, BooguError> {
        self.validate_input(&input)?;
        if !geometry.matches(&input) {
            return Err(BooguError::InvalidShape(
                "cached RoPE geometry does not match denoiser input".into(),
            ));
        }
        let instruction = self.prepare_instruction_with_kernel::<K>(
            input.instruction.clone(),
            geometry,
            rms_norm_policy,
        );
        self.forward_with_kernel_and_rope_and_prepared_instruction::<K>(
            input,
            geometry,
            rms_norm_policy,
            instruction,
        )
    }

    pub(crate) fn prepare_instruction_with_kernel<K: AttentionKernel<B>>(
        &self,
        instruction: Tensor<B, 3>,
        geometry: &BooguRoPeGeometry<B>,
        rms_norm_policy: DenoiserRmsNormPolicy,
    ) -> Tensor<B, 3> {
        let mut instruction = self
            .time_caption_embed
            .embed_caption_with_rms_norm_policy(instruction, rms_norm_policy);
        let text_len = instruction.dims()[1];
        assert_eq!(
            text_len, geometry.key.text_len,
            "prepared instruction length must match cached RoPE geometry"
        );
        let text_rope = (
            geometry.joint_cos.clone().narrow(1, 0, text_len),
            geometry.joint_sin.clone().narrow(1, 0, text_len),
        );
        for block in &self.context_refiner {
            instruction = block.forward_with_kernel_and_rms_norm_policy::<K>(
                instruction,
                Some(text_rope.clone()),
                None,
                rms_norm_policy,
            );
        }
        instruction
    }

    /// Project the step-invariant edit reference once for an entire DMD schedule.
    ///
    /// The timestep-conditioned reference refiner still executes on every step. Only patchifying,
    /// the input projection, and the fixed image-index embedding are retained here.
    pub(crate) fn prepare_reference_embedding(
        &self,
        reference: Tensor<B, 4>,
    ) -> Result<Tensor<B, 3>, BooguError> {
        let mut reference = linear_forward(
            &self.ref_image_patch_embedder,
            patchify(reference, self.config.patch_size)?,
        );
        let embedding = selected_image_index_embedding(
            &self.image_index_embedding,
            self.config.hidden_size,
            reference.dtype(),
        );
        reference = reference + embedding;
        Ok(reference)
    }

    fn forward_with_kernel_and_rope_and_prepared_instruction<K: AttentionKernel<B>>(
        &self,
        input: BooguDenoiserInput<B>,
        geometry: &BooguRoPeGeometry<B>,
        rms_norm_policy: DenoiserRmsNormPolicy,
        instruction: Tensor<B, 3>,
    ) -> Result<Tensor<B, 4>, BooguError> {
        self.forward_with_kernel_and_rope_and_prepared_conditioning::<K>(
            input,
            geometry,
            rms_norm_policy,
            instruction,
            None,
        )
    }

    fn forward_with_kernel_and_rope_and_prepared_conditioning<K: AttentionKernel<B>>(
        &self,
        input: BooguDenoiserInput<B>,
        geometry: &BooguRoPeGeometry<B>,
        rms_norm_policy: DenoiserRmsNormPolicy,
        mut instruction: Tensor<B, 3>,
        prepared_reference: Option<Tensor<B, 3>>,
    ) -> Result<Tensor<B, 4>, BooguError> {
        self.validate_input(&input)?;
        if !geometry.matches(&input) {
            return Err(BooguError::InvalidShape(
                "cached RoPE geometry does not match denoiser input".into(),
            ));
        }
        let [_, _, height, width] = input.latent.dims();
        let time = self.time_caption_embed.embed_timestep(input.timestep);
        let text_len = instruction.dims()[1];
        if text_len != input.instruction.dims()[1] {
            return Err(BooguError::InvalidShape(format!(
                "prepared instruction has {text_len} tokens, expected {}",
                input.instruction.dims()[1]
            )));
        }
        let joint_cos = geometry.joint_cos.clone();
        let joint_sin = geometry.joint_sin.clone();

        let mut generated = linear_forward(
            &self.x_embedder,
            patchify(input.latent, self.config.patch_size)?,
        );
        let generated_start = text_len + geometry.reference_len;
        let generated_rope = (
            joint_cos
                .clone()
                .narrow(1, generated_start, geometry.generated_len),
            joint_sin
                .clone()
                .narrow(1, generated_start, geometry.generated_len),
        );
        for block in &self.noise_refiner {
            generated = block.forward_with_kernel_and_rms_norm_policy::<K>(
                generated,
                Some(generated_rope.clone()),
                Some(time.clone()),
                rms_norm_policy,
            );
        }

        let mut image = if let Some(reference) = input.reference {
            let mut reference = if let Some(prepared) = prepared_reference {
                let [batch, tokens, width] = prepared.dims();
                if batch != 1
                    || tokens != geometry.reference_len
                    || width != self.config.hidden_size
                {
                    return Err(BooguError::InvalidShape(format!(
                        "prepared reference has shape [{batch},{tokens},{width}], expected [1,{},{}]",
                        geometry.reference_len, self.config.hidden_size
                    )));
                }
                prepared
            } else {
                self.prepare_reference_embedding(reference)?
            };
            let reference_rope = (
                joint_cos
                    .clone()
                    .narrow(1, text_len, geometry.reference_len),
                joint_sin
                    .clone()
                    .narrow(1, text_len, geometry.reference_len),
            );
            for block in &self.ref_image_refiner {
                reference = block.forward_with_kernel_and_rms_norm_policy::<K>(
                    reference,
                    Some(reference_rope.clone()),
                    Some(time.clone()),
                    rms_norm_policy,
                );
            }
            Tensor::cat(vec![reference, generated], 1)
        } else {
            if prepared_reference.is_some() {
                return Err(BooguError::InvalidShape(
                    "prepared reference was supplied without an edit reference latent".into(),
                ));
            }
            generated
        };

        let image_rope = (
            joint_cos
                .clone()
                .narrow(1, text_len, geometry.reference_len + geometry.generated_len),
            joint_sin
                .clone()
                .narrow(1, text_len, geometry.reference_len + geometry.generated_len),
        );
        for block in &self.double_stream_layers {
            (image, instruction) = block.forward_with_kernel_and_rms_norm_policy::<K>(
                image,
                instruction,
                image_rope.clone(),
                (joint_cos.clone(), joint_sin.clone()),
                time.clone(),
                rms_norm_policy,
            );
        }

        let mut joint = Tensor::cat(vec![instruction, image], 1);
        for block in &self.single_stream_layers {
            joint = block.forward_with_kernel_and_rms_norm_policy::<K>(
                joint,
                Some((joint_cos.clone(), joint_sin.clone())),
                Some(time.clone()),
                rms_norm_policy,
            );
        }
        let patches =
            self.norm_out
                .forward(joint, time)
                .narrow(1, generated_start, geometry.generated_len);
        unpatchify(
            patches,
            self.config.out_channels,
            height / self.config.patch_size,
            width / self.config.patch_size,
            self.config.patch_size,
        )
    }

    fn validate_input(&self, input: &BooguDenoiserInput<B>) -> Result<(), BooguError> {
        let [batch, channels, _height, _width] = input.latent.dims();
        if batch != 1 {
            return Err(BooguError::InvalidShape(
                "the initial parity path requires batch size one".into(),
            ));
        }
        if channels != self.config.in_channels {
            return Err(BooguError::InvalidShape(format!(
                "expected {} latent channels, got {channels}",
                self.config.in_channels
            )));
        }
        if input.instruction.dims()[0] != batch
            || input.instruction.dims()[2] != self.config.instruction_feature_dim
        {
            return Err(BooguError::InvalidShape(
                "instruction tensor does not match batch/feature dimensions".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
impl BooguDenoiser<NativeWgpuBackend> {
    /// Execute the full denoiser with bounded-query, required Cubek `FlashUnit` attention.
    ///
    /// This path is native-only, requires preserved-F16 activations, honors every attention
    /// module's configured query chunk, and fails closed if the adapter cannot launch FlashUnit.
    /// It never falls back to dense attention and leaves the generic [`Self::forward`] path
    /// unchanged for WebGPU and other Burn backends.
    pub fn forward_native_flash_unit(
        &self,
        input: BooguDenoiserInput<NativeWgpuBackend>,
    ) -> Result<Tensor<NativeWgpuBackend, 4>, BooguError> {
        self.forward_with_kernel::<NativeFlashUnitAttention>(input)
    }

    /// Execute the denoiser with bounded, padded WGPU blackbox FlashAttention.
    ///
    /// Every 120-wide Q/K/V head is transformed to the scale-equivalent 128-wide representation
    /// required by CMMA. `num_planes` must be 2 or 4. The path accepts only F16 activations and
    /// fails closed instead of invoking attention autotuning or a dense fallback.
    pub fn forward_native_padded_blackbox(
        &self,
        input: BooguDenoiserInput<NativeWgpuBackend>,
        num_planes: u8,
    ) -> Result<Tensor<NativeWgpuBackend, 4>, BooguError> {
        self.forward_native_padded_blackbox_tiled(input, num_planes, 1)
    }

    /// Execute padded WGPU blackbox FlashAttention with an explicit key/value partition width.
    ///
    /// `seq_kv_tiles` must be 1 or 2, with two tiles restricted to two planes. Each tile adds 16
    /// key/value rows to one online-softmax partition.
    pub fn forward_native_padded_blackbox_tiled(
        &self,
        input: BooguDenoiserInput<NativeWgpuBackend>,
        num_planes: u8,
        seq_kv_tiles: u8,
    ) -> Result<Tensor<NativeWgpuBackend, 4>, BooguError> {
        assert_supported_wgpu_blackbox_configuration(num_planes, seq_kv_tiles);
        self.forward_native_padded_blackbox_partitioned(input, num_planes, seq_kv_tiles, 1)
    }

    /// Execute padded WGPU blackbox FlashAttention with explicit query and key/value partition
    /// widths. `seq_q_tiles` must be 1; the two-tile blueprint failed native WGPU parity.
    pub(crate) fn forward_native_padded_blackbox_partitioned(
        &self,
        input: BooguDenoiserInput<NativeWgpuBackend>,
        num_planes: u8,
        seq_kv_tiles: u8,
        seq_q_tiles: u8,
    ) -> Result<Tensor<NativeWgpuBackend, 4>, BooguError> {
        self.forward_native_padded_blackbox_partitioned_with_rms_norm_policy(
            input,
            num_planes,
            seq_kv_tiles,
            seq_q_tiles,
            DenoiserRmsNormPolicy::StrictF32,
        )
    }

    pub(crate) fn forward_native_padded_blackbox_partitioned_with_rms_norm_policy(
        &self,
        input: BooguDenoiserInput<NativeWgpuBackend>,
        num_planes: u8,
        seq_kv_tiles: u8,
        seq_q_tiles: u8,
        rms_norm_policy: DenoiserRmsNormPolicy,
    ) -> Result<Tensor<NativeWgpuBackend, 4>, BooguError> {
        self.forward_native_padded_blackbox_partitioned_with_policies(
            input,
            num_planes,
            seq_kv_tiles,
            seq_q_tiles,
            rms_norm_policy,
            false,
            false,
            false,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_native_padded_blackbox_partitioned_with_policies(
        &self,
        input: BooguDenoiserInput<NativeWgpuBackend>,
        num_planes: u8,
        seq_kv_tiles: u8,
        seq_q_tiles: u8,
        rms_norm_policy: DenoiserRmsNormPolicy,
        fused_strict_qk_norm_rope: bool,
        fused_rope_gqa_padding: bool,
        balanced_strict_qk_norm_rope: bool,
        split_double_stream_shared_projection: bool,
    ) -> Result<Tensor<NativeWgpuBackend, 4>, BooguError> {
        let geometry = self.prepare_rope_geometry(&input)?;
        self.forward_native_padded_blackbox_partitioned_with_prepared_rope_and_policies(
            input,
            &geometry,
            num_planes,
            seq_kv_tiles,
            seq_q_tiles,
            rms_norm_policy,
            fused_strict_qk_norm_rope,
            fused_rope_gqa_padding,
            balanced_strict_qk_norm_rope,
            split_double_stream_shared_projection,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_native_padded_blackbox_partitioned_with_prepared_rope_and_policies(
        &self,
        input: BooguDenoiserInput<NativeWgpuBackend>,
        geometry: &BooguRoPeGeometry<NativeWgpuBackend>,
        num_planes: u8,
        seq_kv_tiles: u8,
        seq_q_tiles: u8,
        rms_norm_policy: DenoiserRmsNormPolicy,
        fused_strict_qk_norm_rope: bool,
        fused_rope_gqa_padding: bool,
        balanced_strict_qk_norm_rope: bool,
        split_double_stream_shared_projection: bool,
    ) -> Result<Tensor<NativeWgpuBackend, 4>, BooguError> {
        assert_supported_wgpu_blackbox_partition_configuration(
            num_planes,
            seq_kv_tiles,
            seq_q_tiles,
        );
        assert!(
            u8::from(fused_strict_qk_norm_rope)
                + u8::from(fused_rope_gqa_padding)
                + u8::from(balanced_strict_qk_norm_rope)
                <= 1,
            "native Q/K preparation candidates are mutually exclusive"
        );
        if fused_strict_qk_norm_rope {
            assert_eq!(
                (num_planes, seq_kv_tiles, seq_q_tiles),
                (4, 1, 1),
                "fused strict Q/K norm+RoPE preparation requires p4/kv1/q1"
            );
            assert_eq!(
                rms_norm_policy,
                DenoiserRmsNormPolicy::StrictF32,
                "fused strict Q/K norm+RoPE preparation requires StrictF32 RMSNorm"
            );
            return self.forward_with_native_projection_policy::<
                NativePaddedBlackboxAttention<4, 1, 1, true, false>,
            >(
                input,
                geometry,
                rms_norm_policy,
                split_double_stream_shared_projection,
            );
        }
        if fused_rope_gqa_padding {
            assert_eq!(
                (num_planes, seq_kv_tiles, seq_q_tiles),
                (4, 1, 1),
                "fused RoPE+GQA padding preparation requires p4/kv1/q1"
            );
            assert_eq!(
                rms_norm_policy,
                DenoiserRmsNormPolicy::StrictF32,
                "fused RoPE+GQA padding preparation requires stock StrictF32 RMSNorm"
            );
            return self.forward_with_native_projection_policy::<
                NativePaddedBlackboxAttention<4, 1, 1, false, true>,
            >(
                input,
                geometry,
                rms_norm_policy,
                split_double_stream_shared_projection,
            );
        }
        if balanced_strict_qk_norm_rope {
            assert_eq!(
                (num_planes, seq_kv_tiles, seq_q_tiles),
                (4, 1, 1),
                "balanced strict Q/K RMSNorm preparation requires p4/kv1/q1"
            );
            assert_eq!(
                rms_norm_policy,
                DenoiserRmsNormPolicy::StrictF32,
                "balanced strict Q/K RMSNorm preparation requires StrictF32 RMSNorm"
            );
            return self.forward_with_native_projection_policy::<
                NativePaddedBlackboxAttention<4, 1, 1, false, false, true>,
            >(
                input,
                geometry,
                rms_norm_policy,
                split_double_stream_shared_projection,
            );
        }
        match (num_planes, seq_kv_tiles, seq_q_tiles) {
            (2, 1, 1) => self
                .forward_with_native_projection_policy::<NativePaddedBlackboxAttention<2, 1, 1>>(
                    input,
                    geometry,
                    rms_norm_policy,
                    split_double_stream_shared_projection,
                ),
            (2, 2, 1) => self
                .forward_with_native_projection_policy::<NativePaddedBlackboxAttention<2, 2, 1>>(
                    input,
                    geometry,
                    rms_norm_policy,
                    split_double_stream_shared_projection,
                ),
            (4, 1, 1) => self
                .forward_with_native_projection_policy::<NativePaddedBlackboxAttention<4, 1, 1>>(
                    input,
                    geometry,
                    rms_norm_policy,
                    split_double_stream_shared_projection,
                ),
            _ => unreachable!("validated padded blackbox blueprint"),
        }
    }

    /// Execute an F32 packed-Q4 denoiser through the native F16 padded-blackbox attention core.
    ///
    /// Q4 projections, residuals, norms, and FFNs remain in their released F32 execution policy.
    /// The attention marker narrows only normalized/rotated Q/K/V and widens the attended result.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_native_q4_padded_blackbox_with_prepared_rope(
        &self,
        input: BooguDenoiserInput<NativeWgpuBackend>,
        geometry: &BooguRoPeGeometry<NativeWgpuBackend>,
        num_planes: u8,
        seq_kv_tiles: u8,
        seq_q_tiles: u8,
        rms_norm_policy: DenoiserRmsNormPolicy,
        split_double_stream_shared_projection: bool,
    ) -> Result<Tensor<NativeWgpuBackend, 4>, BooguError> {
        assert_supported_wgpu_blackbox_partition_configuration(
            num_planes,
            seq_kv_tiles,
            seq_q_tiles,
        );
        assert_eq!(
            input.latent.dtype(),
            burn::tensor::DType::F32,
            "packed-Q4 native attention bridge requires F32 denoiser execution"
        );
        match (num_planes, seq_kv_tiles, seq_q_tiles) {
            (2, 1, 1) => self.forward_with_native_projection_policy::<
                NativeF32ToF16PaddedBlackboxAttention<2, 1, 1>,
            >(
                input,
                geometry,
                rms_norm_policy,
                split_double_stream_shared_projection,
            ),
            (2, 2, 1) => self.forward_with_native_projection_policy::<
                NativeF32ToF16PaddedBlackboxAttention<2, 2, 1>,
            >(
                input,
                geometry,
                rms_norm_policy,
                split_double_stream_shared_projection,
            ),
            (4, 1, 1) => self.forward_with_native_projection_policy::<
                NativeF32ToF16PaddedBlackboxAttention<4, 1, 1>,
            >(
                input,
                geometry,
                rms_norm_policy,
                split_double_stream_shared_projection,
            ),
            _ => unreachable!("validated packed-Q4 padded blackbox blueprint"),
        }
    }

    /// Prepare the timestep-invariant caption projection and context-refiner graph once per DMD
    /// run for a packed-Q4 native attention policy.
    pub(crate) fn prepare_native_q4_instruction_with_prepared_rope(
        &self,
        instruction: Tensor<NativeWgpuBackend, 3>,
        geometry: &BooguRoPeGeometry<NativeWgpuBackend>,
        num_planes: u8,
        seq_kv_tiles: u8,
        seq_q_tiles: u8,
        rms_norm_policy: DenoiserRmsNormPolicy,
    ) -> Tensor<NativeWgpuBackend, 3> {
        match (num_planes, seq_kv_tiles, seq_q_tiles) {
            (2, 1, 1) => self
                .prepare_instruction_with_kernel::<NativeF32ToF16PaddedBlackboxAttention<2, 1, 1>>(
                    instruction,
                    geometry,
                    rms_norm_policy,
                ),
            (2, 2, 1) => self
                .prepare_instruction_with_kernel::<NativeF32ToF16PaddedBlackboxAttention<2, 2, 1>>(
                    instruction,
                    geometry,
                    rms_norm_policy,
                ),
            (4, 1, 1) => self
                .prepare_instruction_with_kernel::<NativeF32ToF16PaddedBlackboxAttention<4, 1, 1>>(
                    instruction,
                    geometry,
                    rms_norm_policy,
                ),
            _ => unreachable!("validated packed-Q4 padded blackbox blueprint"),
        }
    }

    /// Execute one packed-Q4 denoiser step with a caption/context tensor prepared once for the
    /// current four-step DMD run.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_native_q4_padded_blackbox_with_prepared_instruction(
        &self,
        input: BooguDenoiserInput<NativeWgpuBackend>,
        geometry: &BooguRoPeGeometry<NativeWgpuBackend>,
        num_planes: u8,
        seq_kv_tiles: u8,
        seq_q_tiles: u8,
        rms_norm_policy: DenoiserRmsNormPolicy,
        split_double_stream_shared_projection: bool,
        instruction: Tensor<NativeWgpuBackend, 3>,
        prepared_reference: Option<Tensor<NativeWgpuBackend, 3>>,
    ) -> Result<Tensor<NativeWgpuBackend, 4>, BooguError> {
        match (num_planes, seq_kv_tiles, seq_q_tiles) {
            (2, 1, 1) => self
                .forward_with_native_projection_policy_and_prepared_instruction::<
                    NativeF32ToF16PaddedBlackboxAttention<2, 1, 1>,
                >(
                    input,
                    geometry,
                    rms_norm_policy,
                    split_double_stream_shared_projection,
                    instruction,
                    prepared_reference,
                ),
            (2, 2, 1) => self
                .forward_with_native_projection_policy_and_prepared_instruction::<
                    NativeF32ToF16PaddedBlackboxAttention<2, 2, 1>,
                >(
                    input,
                    geometry,
                    rms_norm_policy,
                    split_double_stream_shared_projection,
                    instruction,
                    prepared_reference,
                ),
            (4, 1, 1) => self
                .forward_with_native_projection_policy_and_prepared_instruction::<
                    NativeF32ToF16PaddedBlackboxAttention<4, 1, 1>,
                >(
                    input,
                    geometry,
                    rms_norm_policy,
                    split_double_stream_shared_projection,
                    instruction,
                    prepared_reference,
                ),
            _ => unreachable!("validated packed-Q4 padded blackbox blueprint"),
        }
    }

    fn forward_with_native_projection_policy<K: AttentionKernel<NativeWgpuBackend>>(
        &self,
        input: BooguDenoiserInput<NativeWgpuBackend>,
        geometry: &BooguRoPeGeometry<NativeWgpuBackend>,
        rms_norm_policy: DenoiserRmsNormPolicy,
        split_double_stream_shared_projection: bool,
    ) -> Result<Tensor<NativeWgpuBackend, 4>, BooguError> {
        if split_double_stream_shared_projection {
            self.forward_with_kernel_and_rope_and_rms_norm_policy::<
                SplitDoubleStreamSharedProjection<K>,
            >(input, geometry, rms_norm_policy)
        } else {
            self.forward_with_kernel_and_rope_and_rms_norm_policy::<K>(
                input,
                geometry,
                rms_norm_policy,
            )
        }
    }

    fn forward_with_native_projection_policy_and_prepared_instruction<
        K: AttentionKernel<NativeWgpuBackend>,
    >(
        &self,
        input: BooguDenoiserInput<NativeWgpuBackend>,
        geometry: &BooguRoPeGeometry<NativeWgpuBackend>,
        rms_norm_policy: DenoiserRmsNormPolicy,
        split_double_stream_shared_projection: bool,
        instruction: Tensor<NativeWgpuBackend, 3>,
        prepared_reference: Option<Tensor<NativeWgpuBackend, 3>>,
    ) -> Result<Tensor<NativeWgpuBackend, 4>, BooguError> {
        if split_double_stream_shared_projection {
            self.forward_with_kernel_and_rope_and_prepared_conditioning::<
                SplitDoubleStreamSharedProjection<K>,
            >(
                input,
                geometry,
                rms_norm_policy,
                instruction,
                prepared_reference,
            )
        } else {
            self.forward_with_kernel_and_rope_and_prepared_conditioning::<K>(
                input,
                geometry,
                rms_norm_policy,
                instruction,
                prepared_reference,
            )
        }
    }
}

pub(super) fn rope_tensors<B: Backend>(
    ids: &[[u32; 3]],
    axis_dims: [usize; 3],
    theta: f64,
    dtype: burn::tensor::DType,
    device: &B::Device,
) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let (cos, sin) = rope_values(ids, axis_dims, theta);
    let head_dim = axis_dims.iter().sum::<usize>();
    let sequence = ids.len();
    (
        Tensor::<B, 3>::from_data(TensorData::new(cos, [1, sequence, head_dim]), device)
            .cast(dtype),
        Tensor::<B, 3>::from_data(TensorData::new(sin, [1, sequence, head_dim]), device)
            .cast(dtype),
    )
}

fn rope_values(ids: &[[u32; 3]], axis_dims: [usize; 3], theta: f64) -> (Vec<f32>, Vec<f32>) {
    type AxisCoordinateValues = std::collections::HashMap<u32, (Vec<f32>, Vec<f32>)>;

    let inverse_frequencies: [Vec<f64>; 3] = core::array::from_fn(|axis| {
        let dim = axis_dims[axis];
        (0..dim / 2)
            .map(|pair| theta.powf(-((2 * pair) as f64) / dim as f64))
            .collect()
    });
    let mut coordinate_values: [AxisCoordinateValues; 3] =
        core::array::from_fn(|_| AxisCoordinateValues::new());

    for id in ids {
        for (axis, axis_values) in coordinate_values.iter_mut().enumerate() {
            axis_values.entry(id[axis]).or_insert_with(|| {
                let mut cos = Vec::with_capacity(axis_dims[axis]);
                let mut sin = Vec::with_capacity(axis_dims[axis]);
                for &inverse_frequency in &inverse_frequencies[axis] {
                    let phase = id[axis] as f64 * inverse_frequency;
                    let cosine = phase.cos() as f32;
                    let sine = phase.sin() as f32;
                    cos.extend([cosine, cosine]);
                    sin.extend([sine, sine]);
                }
                (cos, sin)
            });
        }
    }

    let head_dim = axis_dims.iter().sum::<usize>();
    let mut cos = Vec::with_capacity(ids.len() * head_dim);
    let mut sin = Vec::with_capacity(ids.len() * head_dim);
    for id in ids {
        for (axis, axis_values) in coordinate_values.iter().enumerate() {
            let (axis_cos, axis_sin) = axis_values
                .get(&id[axis])
                .expect("RoPE coordinate was prepared from the same position IDs");
            cos.extend_from_slice(axis_cos);
            sin.extend_from_slice(axis_sin);
        }
    }
    (cos, sin)
}

#[cfg(all(test, feature = "ndarray"))]
fn scalar_rope_values_reference(
    ids: &[[u32; 3]],
    axis_dims: [usize; 3],
    theta: f64,
) -> (Vec<f32>, Vec<f32>) {
    let head_dim = axis_dims.iter().sum::<usize>();
    let mut cos = Vec::with_capacity(ids.len() * head_dim);
    let mut sin = Vec::with_capacity(ids.len() * head_dim);
    for id in ids {
        for (axis, dim) in axis_dims.into_iter().enumerate() {
            for pair in 0..dim / 2 {
                let inverse_frequency = theta.powf(-((2 * pair) as f64) / dim as f64);
                let phase = id[axis] as f64 * inverse_frequency;
                let cosine = phase.cos() as f32;
                let sine = phase.sin() as f32;
                cos.extend([cosine, cosine]);
                sin.extend([sine, sine]);
            }
        }
    }
    (cos, sin)
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use burn::{
        module::Param,
        tensor::{DType, TensorData, quantization::*},
    };
    use burn_ndarray::{NdArray, NdArrayDevice};

    use super::*;

    type TestBackend = NdArray<f32>;

    #[test]
    fn quantized_image_index_selects_before_widening_correctness() {
        let device = Default::default();
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_level(QuantLevel::Tensor)
            .with_param(QuantParam::F32)
            .with_store(QuantStore::PackedU32(0));
        let weight = Tensor::<TestBackend, 2>::from_data(
            TensorData::quantized(vec![1_i8; 160], [5, 32], scheme, &[0.125_f32]),
            &device,
        );
        let embedding = nn::Embedding {
            weight: Param::from_tensor(weight),
        };

        let selected = selected_image_index_embedding(&embedding, 32, DType::F32);

        assert_eq!(selected.dims(), [1, 1, 32]);
        assert_eq!(selected.dtype(), DType::F32);
        assert!(
            selected
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );
        assert!(matches!(embedding.weight.val().dtype(), DType::QFloat(_)));
    }

    fn tiny_config() -> BooguConfig {
        BooguConfig {
            patch_size: 2,
            in_channels: 4,
            out_channels: 4,
            hidden_size: 8,
            num_layers: 2,
            num_double_stream_layers: 1,
            num_refiner_layers: 1,
            num_attention_heads: 2,
            num_kv_heads: 1,
            multiple_of: 8,
            norm_eps: 1.0e-5,
            axes_dim_rope: [2, 2, 0],
            axes_lens: [16, 16, 16],
            instruction_feature_dim: 8,
            timestep_scale: 1000.0,
        }
    }

    fn tiny_input(
        device: &NdArrayDevice,
        text_len: usize,
        reference: bool,
        latent_size: [usize; 2],
    ) -> BooguDenoiserInput<TestBackend> {
        let [height, width] = latent_size;
        BooguDenoiserInput {
            latent: Tensor::from_data(
                TensorData::new(
                    (0..4 * height * width)
                        .map(|index| (index as f32 - 31.0) / 64.0)
                        .collect::<Vec<_>>(),
                    [1, 4, height, width],
                ),
                device,
            ),
            timestep: Tensor::from_data([0.375_f32], device),
            instruction: Tensor::from_data(
                TensorData::new(
                    (0..text_len * 8)
                        .map(|index| (index as f32 - 12.0) / 32.0)
                        .collect::<Vec<_>>(),
                    [1, text_len, 8],
                ),
                device,
            ),
            reference: reference.then(|| Tensor::zeros([1, 4, 4, 4], device)),
        }
    }

    #[test]
    fn precomputed_rope_values_match_scalar_1k5_bits_parity() {
        let ids = position_ids(
            147,
            &[LatentSize {
                height: 32,
                width: 32,
            }],
            LatentSize {
                height: 192,
                width: 192,
            },
            2,
        )
        .unwrap();
        let expected = scalar_rope_values_reference(&ids.values, [40, 40, 40], 10_000.0);
        let actual = rope_values(&ids.values, [40, 40, 40], 10_000.0);

        assert_eq!(actual.0.len(), expected.0.len());
        assert_eq!(actual.1.len(), expected.1.len());
        assert!(
            actual
                .0
                .iter()
                .zip(&expected.0)
                .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
        );
        assert!(
            actual
                .1
                .iter()
                .zip(&expected.1)
                .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
        );
    }

    #[test]
    fn rope_geometry_key_is_exact_correctness() {
        let device = Default::default();
        let model = BooguDenoiser::<TestBackend>::new(tiny_config(), &device).unwrap();
        let input = tiny_input(&device, 3, true, [4, 4]);
        let geometry = model.prepare_rope_geometry(&input).unwrap();

        assert!(geometry.matches(&input));
        assert!(!geometry.matches(&tiny_input(&device, 4, true, [4, 4])));
        assert!(!geometry.matches(&tiny_input(&device, 3, false, [4, 4])));
        assert!(!geometry.matches(&tiny_input(&device, 3, true, [4, 6])));

        let mut different_dtype = geometry.key;
        different_dtype.dtype = DType::F16;
        assert_ne!(geometry.key, different_dtype);
    }

    #[test]
    fn prepared_rope_matches_uncached_forward_correctness() {
        let device = Default::default();
        TestBackend::seed(&device, 71);
        let model = BooguDenoiser::<TestBackend>::new(tiny_config(), &device).unwrap();
        let geometry = model
            .prepare_rope_geometry(&tiny_input(&device, 3, true, [4, 4]))
            .unwrap();
        let expected = model
            .forward(tiny_input(&device, 3, true, [4, 4]))
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let actual = model
            .forward_with_kernel_and_rope::<PortableChunkedAttention>(
                tiny_input(&device, 3, true, [4, 4]),
                &geometry,
            )
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        assert_eq!(actual, expected);
        assert!(
            model
                .forward_with_kernel_and_rope::<PortableChunkedAttention>(
                    tiny_input(&device, 4, true, [4, 4]),
                    &geometry,
                )
                .is_err()
        );
    }

    #[test]
    fn prepared_instruction_matches_per_step_context_forward_correctness() {
        let device = Default::default();
        TestBackend::seed(&device, 73);
        let model = BooguDenoiser::<TestBackend>::new(tiny_config(), &device).unwrap();
        let geometry = model
            .prepare_rope_geometry(&tiny_input(&device, 3, true, [4, 4]))
            .unwrap();
        let expected = model
            .forward_with_kernel_and_rope::<PortableChunkedAttention>(
                tiny_input(&device, 3, true, [4, 4]),
                &geometry,
            )
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let instruction = model.prepare_instruction_with_kernel::<PortableChunkedAttention>(
            tiny_input(&device, 3, true, [4, 4]).instruction,
            &geometry,
            DenoiserRmsNormPolicy::StrictF32,
        );
        let actual = model
            .forward_with_kernel_and_rope_and_prepared_instruction::<PortableChunkedAttention>(
                tiny_input(&device, 3, true, [4, 4]),
                &geometry,
                DenoiserRmsNormPolicy::StrictF32,
                instruction,
            )
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn prepared_reference_projection_matches_per_step_projection_correctness() {
        let device = Default::default();
        TestBackend::seed(&device, 79);
        let model = BooguDenoiser::<TestBackend>::new(tiny_config(), &device).unwrap();
        let input = tiny_input(&device, 3, true, [4, 4]);
        let geometry = model.prepare_rope_geometry(&input).unwrap();
        let expected = model
            .forward_with_kernel_and_rope::<PortableChunkedAttention>(
                tiny_input(&device, 3, true, [4, 4]),
                &geometry,
            )
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let instruction = model.prepare_instruction_with_kernel::<PortableChunkedAttention>(
            input.instruction.clone(),
            &geometry,
            DenoiserRmsNormPolicy::StrictF32,
        );
        let prepared_reference = model
            .prepare_reference_embedding(
                input
                    .reference
                    .clone()
                    .expect("the edit fixture carries a reference"),
            )
            .unwrap();
        let actual = model
            .forward_with_kernel_and_rope_and_prepared_conditioning::<PortableChunkedAttention>(
                input,
                &geometry,
                DenoiserRmsNormPolicy::StrictF32,
                instruction,
                Some(prepared_reference),
            )
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        assert_eq!(actual, expected);
    }
}
