use burn::{nn, prelude::*};

use crate::{
    BooguConfig, BooguDenoiserInput, BooguError,
    latent::{patchify, unpatchify},
    pipeline::DmdDenoiser,
    rope::{LatentSize, position_ids},
};

use super::{
    CombinedTimestepCaptionEmbedding, DoubleStreamBlock, FinalProjection, SingleStreamBlock,
    denoiser::rope_tensors,
};

/// Optional observer for numerical checks at streamed denoiser boundaries.
///
/// Production callers use the default no-op implementation on `()`. Reference tools can read
/// back named tensors without changing the model's execution order or retaining stage weights.
pub trait DenoiserStageObserver<B: Backend> {
    /// Observe a rank-two boundary tensor.
    fn rank2(&mut self, _name: &str, _tensor: Tensor<B, 2>) -> Result<(), BooguError> {
        Ok(())
    }

    /// Observe a rank-three boundary tensor.
    fn rank3(&mut self, _name: &str, _tensor: Tensor<B, 3>) -> Result<(), BooguError> {
        Ok(())
    }

    /// Observe a rank-four boundary tensor.
    fn rank4(&mut self, _name: &str, _tensor: Tensor<B, 4>) -> Result<(), BooguError> {
        Ok(())
    }
}

impl<B: Backend> DenoiserStageObserver<B> for () {}

/// Small, persistent projections needed to initialize one streamed denoiser step.
#[derive(Module, Debug)]
pub struct BooguDenoiserPrelude<B: Backend> {
    /// Generated latent patch projection.
    pub x_embedder: nn::Linear<B>,
    /// Reference latent patch projection.
    pub ref_image_patch_embedder: nn::Linear<B>,
    /// Timestep and caption projections.
    pub time_caption_embed: CombinedTimestepCaptionEmbedding<B>,
    /// Embeddings distinguishing reference image indices.
    pub image_index_embedding: nn::Embedding<B>,
    #[module(skip)]
    config: BooguConfig,
}

impl<B: Backend> BooguDenoiserPrelude<B> {
    /// Allocate the prelude. A stage source replaces its parameters from a verified shard.
    pub fn new(config: BooguConfig, device: &B::Device) -> Result<Self, BooguError> {
        config.validate()?;
        let patch_width = config.patch_size * config.patch_size * config.in_channels;
        let conditioning_width = config.hidden_size.min(1024);
        Ok(Self {
            x_embedder: nn::LinearConfig::new(patch_width, config.hidden_size).init(device),
            ref_image_patch_embedder: nn::LinearConfig::new(patch_width, config.hidden_size)
                .init(device),
            time_caption_embed: CombinedTimestepCaptionEmbedding::new(
                config.hidden_size,
                config.instruction_feature_dim,
                256,
                conditioning_width,
                config.norm_eps,
                config.timestep_scale,
                device,
            ),
            image_index_embedding: nn::EmbeddingConfig::new(5, config.hidden_size).init(device),
            config,
        })
    }

    /// Initialize activation state, after which this module can be dropped from VRAM.
    pub fn begin(&self, input: BooguDenoiserInput<B>) -> Result<BooguStreamState<B>, BooguError> {
        self.begin_with_observer(input, &mut ())
    }

    /// Initialize activation state and expose the checkpoint-relevant prelude boundaries.
    pub fn begin_with_observer<O: DenoiserStageObserver<B>>(
        &self,
        input: BooguDenoiserInput<B>,
        observer: &mut O,
    ) -> Result<BooguStreamState<B>, BooguError> {
        let [batch, channels, height, width] = input.latent.dims();
        if batch != 1 || channels != self.config.in_channels {
            return Err(BooguError::InvalidShape(format!(
                "streaming denoiser expects [1,{},H,W], got {:?}",
                self.config.in_channels,
                input.latent.dims()
            )));
        }
        if input.instruction.dims()[0] != 1
            || input.instruction.dims()[2] != self.config.instruction_feature_dim
        {
            return Err(BooguError::InvalidShape(format!(
                "instruction shape {:?} does not match [1,T,{}]",
                input.instruction.dims(),
                self.config.instruction_feature_dim
            )));
        }
        let device = input.latent.device();
        let (time, instruction) = self
            .time_caption_embed
            .forward(input.timestep, input.instruction);
        observer.rank2("time_caption_embed.0", time.clone())?;
        observer.rank3("time_caption_embed.1", instruction.clone())?;
        let text_len = instruction.dims()[1];
        let reference_size = input.reference.as_ref().map(|reference| {
            let dims = reference.dims();
            LatentSize {
                height: dims[2],
                width: dims[3],
            }
        });
        let ids = position_ids(
            text_len,
            reference_size.as_slice(),
            LatentSize { height, width },
            self.config.patch_size,
        )?;
        let (joint_cos, joint_sin) = rope_tensors::<B>(
            &ids.values,
            self.config.axes_dim_rope,
            10_000.0,
            input.latent.dtype(),
            &device,
        );
        let text_rope = (
            joint_cos.clone().narrow(1, 0, text_len),
            joint_sin.clone().narrow(1, 0, text_len),
        );
        let generated_start = text_len + ids.reference_len;
        let generated_rope = (
            joint_cos
                .clone()
                .narrow(1, generated_start, ids.generated_len),
            joint_sin
                .clone()
                .narrow(1, generated_start, ids.generated_len),
        );
        let generated = self
            .x_embedder
            .forward(patchify(input.latent, self.config.patch_size)?);
        observer.rank3("x_embedder", generated.clone())?;
        let (reference, reference_rope) = if let Some(reference) = input.reference {
            let mut reference = self
                .ref_image_patch_embedder
                .forward(patchify(reference, self.config.patch_size)?);
            observer.rank3("ref_image_patch_embedder", reference.clone())?;
            let index_embedding = self
                .image_index_embedding
                .weight
                .val()
                .slice([0..1, 0..self.config.hidden_size])
                .reshape([1, 1, self.config.hidden_size]);
            reference = reference + index_embedding;
            (
                Some(reference),
                Some((
                    joint_cos.clone().narrow(1, text_len, ids.reference_len),
                    joint_sin.clone().narrow(1, text_len, ids.reference_len),
                )),
            )
        } else {
            (None, None)
        };
        let image_rope = (
            joint_cos
                .clone()
                .narrow(1, text_len, ids.reference_len + ids.generated_len),
            joint_sin
                .clone()
                .narrow(1, text_len, ids.reference_len + ids.generated_len),
        );
        Ok(BooguStreamState {
            instruction: Some(instruction),
            generated: Some(generated),
            reference,
            image: None,
            joint: None,
            time,
            text_rope,
            generated_rope,
            reference_rope,
            image_rope,
            joint_rope: (joint_cos, joint_sin),
            config: self.config.clone(),
            generated_start,
            generated_len: ids.generated_len,
            latent_height: height,
            latent_width: width,
            context_refiners: 0,
            noise_refiners: 0,
            reference_refiners: 0,
            double_layers: 0,
            single_layers: 0,
        })
    }
}

/// Activation-only state retained while verified layer shards are loaded and dropped.
pub struct BooguStreamState<B: Backend> {
    instruction: Option<Tensor<B, 3>>,
    generated: Option<Tensor<B, 3>>,
    reference: Option<Tensor<B, 3>>,
    image: Option<Tensor<B, 3>>,
    joint: Option<Tensor<B, 3>>,
    time: Tensor<B, 2>,
    text_rope: (Tensor<B, 3>, Tensor<B, 3>),
    generated_rope: (Tensor<B, 3>, Tensor<B, 3>),
    reference_rope: Option<(Tensor<B, 3>, Tensor<B, 3>)>,
    image_rope: (Tensor<B, 3>, Tensor<B, 3>),
    joint_rope: (Tensor<B, 3>, Tensor<B, 3>),
    config: BooguConfig,
    generated_start: usize,
    generated_len: usize,
    latent_height: usize,
    latent_width: usize,
    context_refiners: usize,
    noise_refiners: usize,
    reference_refiners: usize,
    double_layers: usize,
    single_layers: usize,
}

impl<B: Backend> BooguStreamState<B> {
    /// Apply the next context-refiner shard.
    pub fn apply_context_refiner(
        &mut self,
        block: &SingleStreamBlock<B>,
    ) -> Result<(), BooguError> {
        if self.context_refiners >= self.config.num_refiner_layers || self.image.is_some() {
            return Err(BooguError::InvalidRequest(
                "context refiner applied out of order".into(),
            ));
        }
        let instruction = self.instruction.take().ok_or_else(|| {
            BooguError::InvalidRequest("instruction stream is no longer available".into())
        })?;
        self.instruction = Some(block.forward(instruction, Some(self.text_rope.clone()), None));
        self.context_refiners += 1;
        Ok(())
    }

    /// Apply the next generated-noise refiner shard.
    pub fn apply_noise_refiner(&mut self, block: &SingleStreamBlock<B>) -> Result<(), BooguError> {
        if self.noise_refiners >= self.config.num_refiner_layers || self.image.is_some() {
            return Err(BooguError::InvalidRequest(
                "noise refiner applied out of order".into(),
            ));
        }
        let generated = self.generated.take().ok_or_else(|| {
            BooguError::InvalidRequest("generated stream is no longer available".into())
        })?;
        self.generated = Some(block.forward(
            generated,
            Some(self.generated_rope.clone()),
            Some(self.time.clone()),
        ));
        self.noise_refiners += 1;
        Ok(())
    }

    /// Apply the next single-reference refiner shard.
    pub fn apply_reference_refiner(
        &mut self,
        block: &SingleStreamBlock<B>,
    ) -> Result<(), BooguError> {
        if self.reference.is_none() {
            return Err(BooguError::InvalidRequest(
                "reference refiner supplied for a generation request".into(),
            ));
        }
        if self.reference_refiners >= self.config.num_refiner_layers || self.image.is_some() {
            return Err(BooguError::InvalidRequest(
                "reference refiner applied out of order".into(),
            ));
        }
        let reference = self.reference.take().expect("reference presence checked");
        self.reference = Some(block.forward(
            reference,
            self.reference_rope.clone(),
            Some(self.time.clone()),
        ));
        self.reference_refiners += 1;
        Ok(())
    }

    /// Fuse reference and generated streams after all three refiner sequences.
    pub fn finish_refiners(&mut self) -> Result<(), BooguError> {
        let expected_reference = if self.reference.is_some() {
            self.config.num_refiner_layers
        } else {
            0
        };
        if self.context_refiners != self.config.num_refiner_layers
            || self.noise_refiners != self.config.num_refiner_layers
            || self.reference_refiners != expected_reference
        {
            return Err(BooguError::InvalidRequest(format!(
                "incomplete refiners: context={}/{}, noise={}/{}, reference={}/{}",
                self.context_refiners,
                self.config.num_refiner_layers,
                self.noise_refiners,
                self.config.num_refiner_layers,
                self.reference_refiners,
                expected_reference
            )));
        }
        let generated = self
            .generated
            .take()
            .expect("noise refiners retained stream");
        self.image = Some(if let Some(reference) = self.reference.take() {
            Tensor::cat(vec![reference, generated], 1)
        } else {
            generated
        });
        Ok(())
    }

    /// Apply the next leading dual-stream layer shard.
    pub fn apply_double_stream(&mut self, block: &DoubleStreamBlock<B>) -> Result<(), BooguError> {
        if self.image.is_none() || self.joint.is_some() {
            return Err(BooguError::InvalidRequest(
                "double-stream layer applied before refiner fusion or after joint fusion".into(),
            ));
        }
        if self.double_layers >= self.config.num_double_stream_layers {
            return Err(BooguError::InvalidRequest(
                "too many double-stream layers".into(),
            ));
        }
        let image = self.image.take().expect("image presence checked");
        let instruction = self
            .instruction
            .take()
            .expect("instruction is retained through double-stream layers");
        let (image, instruction) = block.forward(
            image,
            instruction,
            self.image_rope.clone(),
            self.joint_rope.clone(),
            self.time.clone(),
        );
        self.image = Some(image);
        self.instruction = Some(instruction);
        self.double_layers += 1;
        Ok(())
    }

    /// Fuse instruction and image streams before single-stream layers.
    pub fn finish_double_stream(&mut self) -> Result<(), BooguError> {
        if self.double_layers != self.config.num_double_stream_layers {
            return Err(BooguError::InvalidRequest(format!(
                "incomplete double stream: {}/{} layers",
                self.double_layers, self.config.num_double_stream_layers
            )));
        }
        let instruction = self.instruction.take().ok_or_else(|| {
            BooguError::InvalidRequest("instruction stream is unavailable".into())
        })?;
        let image = self
            .image
            .take()
            .ok_or_else(|| BooguError::InvalidRequest("image stream is unavailable".into()))?;
        self.joint = Some(Tensor::cat(vec![instruction, image], 1));
        Ok(())
    }

    /// Apply the next joint single-stream layer shard.
    pub fn apply_single_stream(&mut self, block: &SingleStreamBlock<B>) -> Result<(), BooguError> {
        if self.single_layers >= self.config.num_single_stream_layers() {
            return Err(BooguError::InvalidRequest(
                "too many single-stream layers".into(),
            ));
        }
        let joint = self.joint.take().ok_or_else(|| {
            BooguError::InvalidRequest("single-stream layer applied before joint fusion".into())
        })?;
        self.joint = Some(block.forward(
            joint,
            Some(self.joint_rope.clone()),
            Some(self.time.clone()),
        ));
        self.single_layers += 1;
        Ok(())
    }
}

/// Final adaptive normalization and latent projection for a streamed step.
#[derive(Module, Debug)]
pub struct BooguDenoiserTail<B: Backend> {
    /// Checkpoint-compatible final module.
    pub norm_out: FinalProjection<B>,
    #[module(skip)]
    config: BooguConfig,
}

impl<B: Backend> BooguDenoiserTail<B> {
    /// Allocate the tail. A stage source replaces its parameters from a verified shard.
    pub fn new(config: BooguConfig, device: &B::Device) -> Result<Self, BooguError> {
        config.validate()?;
        Ok(Self {
            norm_out: FinalProjection::new(
                config.hidden_size,
                config.hidden_size.min(1024),
                config.patch_size * config.patch_size * config.out_channels,
                1.0e-6,
                device,
            ),
            config,
        })
    }

    /// Finish a complete streamed state and restore BCHW latent layout.
    pub fn finish(&self, state: BooguStreamState<B>) -> Result<Tensor<B, 4>, BooguError> {
        self.finish_with_observer(state, &mut ())
    }

    /// Finish a streamed state while exposing the final patch projection.
    pub fn finish_with_observer<O: DenoiserStageObserver<B>>(
        &self,
        mut state: BooguStreamState<B>,
        observer: &mut O,
    ) -> Result<Tensor<B, 4>, BooguError> {
        if state.single_layers != state.config.num_single_stream_layers() {
            return Err(BooguError::InvalidRequest(format!(
                "incomplete single stream: {}/{} layers",
                state.single_layers,
                state.config.num_single_stream_layers()
            )));
        }
        if state.config != self.config {
            return Err(BooguError::InvalidConfig(
                "streaming tail configuration differs from activation state".into(),
            ));
        }
        let joint = state
            .joint
            .take()
            .ok_or_else(|| BooguError::InvalidRequest("joint stream is unavailable".into()))?;
        let projected = self.norm_out.forward(joint, state.time);
        observer.rank3("norm_out", projected.clone())?;
        let patches = projected.narrow(1, state.generated_start, state.generated_len);
        unpatchify(
            patches,
            self.config.out_channels,
            state.latent_height / self.config.patch_size,
            state.latent_width / self.config.patch_size,
            self.config.patch_size,
        )
    }
}

/// Source of verified, short-lived denoiser stage modules.
///
/// Implementations normally fetch one content-addressed Burnpack shard, verify it, construct the
/// corresponding module, and return it. The executor drops every module before requesting the
/// next, bounding peak device weight residency to one transformer block.
pub trait StreamingStageSource<B: Backend> {
    /// Load projections and embeddings for the start of one DMD step.
    fn load_prelude(&mut self) -> Result<BooguDenoiserPrelude<B>, BooguError>;
    /// Load a context refiner by index.
    fn load_context_refiner(&mut self, index: usize) -> Result<SingleStreamBlock<B>, BooguError>;
    /// Load a generated-noise refiner by index.
    fn load_noise_refiner(&mut self, index: usize) -> Result<SingleStreamBlock<B>, BooguError>;
    /// Load a reference-image refiner by index.
    fn load_reference_refiner(&mut self, index: usize) -> Result<SingleStreamBlock<B>, BooguError>;
    /// Load a dual-stream transformer layer by index.
    fn load_double_stream(&mut self, index: usize) -> Result<DoubleStreamBlock<B>, BooguError>;
    /// Load a joint single-stream transformer layer by index.
    fn load_single_stream(&mut self, index: usize) -> Result<SingleStreamBlock<B>, BooguError>;
    /// Load final normalization and projection.
    fn load_tail(&mut self) -> Result<BooguDenoiserTail<B>, BooguError>;
    /// Synchronize submitted device work before a stage module is dropped.
    fn synchronize(&mut self) -> Result<(), BooguError>;
}

/// Wasm-local asynchronous source of verified, short-lived denoiser stage modules.
///
/// Futures deliberately have no `Send` requirement so browser fetch, Cache Storage, and WebGPU
/// handles can stay on one event loop. [`StreamingBooguDenoiser::predict_with_observer_async`]
/// awaits synchronization and drops each returned module before requesting the following stage.
#[allow(async_fn_in_trait)]
pub trait AsyncBooguDenoiserStageSource<B: Backend> {
    /// Load projections and embeddings for the start of one DMD step.
    async fn load_prelude(&mut self) -> Result<BooguDenoiserPrelude<B>, BooguError>;
    /// Load a context refiner by index.
    async fn load_context_refiner(
        &mut self,
        index: usize,
    ) -> Result<SingleStreamBlock<B>, BooguError>;
    /// Load a generated-noise refiner by index.
    async fn load_noise_refiner(
        &mut self,
        index: usize,
    ) -> Result<SingleStreamBlock<B>, BooguError>;
    /// Load a reference-image refiner by index.
    async fn load_reference_refiner(
        &mut self,
        index: usize,
    ) -> Result<SingleStreamBlock<B>, BooguError>;
    /// Load a dual-stream transformer layer by index.
    async fn load_double_stream(
        &mut self,
        index: usize,
    ) -> Result<DoubleStreamBlock<B>, BooguError>;
    /// Load a joint single-stream transformer layer by index.
    async fn load_single_stream(
        &mut self,
        index: usize,
    ) -> Result<SingleStreamBlock<B>, BooguError>;
    /// Load final normalization and projection.
    async fn load_tail(&mut self) -> Result<BooguDenoiserTail<B>, BooguError>;
    /// Await submitted device work before the current stage module is dropped.
    async fn synchronize(&mut self) -> Result<(), BooguError>;
}

/// Concrete one-block-at-a-time denoiser executor for constrained WGPU/WebGPU devices.
pub struct StreamingBooguDenoiser<B: Backend, S> {
    config: BooguConfig,
    source: S,
    _backend: core::marker::PhantomData<B>,
}

impl<B: Backend, S> StreamingBooguDenoiser<B, S> {
    /// Create an executor around a verified stage source.
    pub fn new(config: BooguConfig, source: S) -> Result<Self, BooguError> {
        config.validate()?;
        Ok(Self {
            config,
            source,
            _backend: core::marker::PhantomData,
        })
    }

    /// Access the stage source, for cache statistics and lifecycle control.
    pub const fn source(&self) -> &S {
        &self.source
    }

    /// Mutably access the stage source.
    pub fn source_mut(&mut self) -> &mut S {
        &mut self.source
    }
}

impl<B, S> StreamingBooguDenoiser<B, S>
where
    B: Backend,
    S: StreamingStageSource<B>,
{
    /// Predict one velocity while reporting every checkpoint-relevant streamed boundary.
    pub fn predict_with_observer<O: DenoiserStageObserver<B>>(
        &mut self,
        input: BooguDenoiserInput<B>,
        observer: &mut O,
    ) -> Result<Tensor<B, 4>, BooguError> {
        let has_reference = input.reference.is_some();
        let prelude = self.source.load_prelude()?;
        let mut state = prelude.begin_with_observer(input, observer)?;
        self.source.synchronize()?;
        drop(prelude);

        for index in 0..self.config.num_refiner_layers {
            let block = self.source.load_context_refiner(index)?;
            state.apply_context_refiner(&block)?;
            observer.rank3(
                &format!("context_refiner.{index}"),
                state
                    .instruction
                    .as_ref()
                    .expect("context refiner retains instruction")
                    .clone(),
            )?;
            self.source.synchronize()?;
            drop(block);
        }
        for index in 0..self.config.num_refiner_layers {
            let block = self.source.load_noise_refiner(index)?;
            state.apply_noise_refiner(&block)?;
            observer.rank3(
                &format!("noise_refiner.{index}"),
                state
                    .generated
                    .as_ref()
                    .expect("noise refiner retains generated stream")
                    .clone(),
            )?;
            self.source.synchronize()?;
            drop(block);
        }
        if has_reference {
            for index in 0..self.config.num_refiner_layers {
                let block = self.source.load_reference_refiner(index)?;
                state.apply_reference_refiner(&block)?;
                observer.rank3(
                    &format!("ref_image_refiner.{index}"),
                    state
                        .reference
                        .as_ref()
                        .expect("reference refiner retains reference stream")
                        .clone(),
                )?;
                self.source.synchronize()?;
                drop(block);
            }
        }
        state.finish_refiners()?;

        for index in 0..self.config.num_double_stream_layers {
            let block = self.source.load_double_stream(index)?;
            state.apply_double_stream(&block)?;
            observer.rank3(
                &format!("double_stream_layers.{index}.0"),
                state
                    .image
                    .as_ref()
                    .expect("double stream retains image")
                    .clone(),
            )?;
            observer.rank3(
                &format!("double_stream_layers.{index}.1"),
                state
                    .instruction
                    .as_ref()
                    .expect("double stream retains instruction")
                    .clone(),
            )?;
            self.source.synchronize()?;
            drop(block);
        }
        state.finish_double_stream()?;
        for index in 0..self.config.num_single_stream_layers() {
            let block = self.source.load_single_stream(index)?;
            state.apply_single_stream(&block)?;
            observer.rank3(
                &format!("single_stream_layers.{index}"),
                state
                    .joint
                    .as_ref()
                    .expect("single stream retains joint tokens")
                    .clone(),
            )?;
            self.source.synchronize()?;
            drop(block);
        }
        let tail = self.source.load_tail()?;
        let output = tail.finish_with_observer(state, observer)?;
        self.source.synchronize()?;
        Ok(output)
    }
}

impl<B, S> StreamingBooguDenoiser<B, S>
where
    B: Backend,
    S: AsyncBooguDenoiserStageSource<B>,
{
    /// Asynchronously predict one velocity while reporting every streamed numerical boundary.
    ///
    /// Each module fetch and synchronization is awaited. A module is explicitly dropped before
    /// the following fetch, preserving one-semantic-stage weight residency in browsers.
    pub async fn predict_with_observer_async<O: DenoiserStageObserver<B>>(
        &mut self,
        input: BooguDenoiserInput<B>,
        observer: &mut O,
    ) -> Result<Tensor<B, 4>, BooguError> {
        let has_reference = input.reference.is_some();
        let prelude = self.source.load_prelude().await?;
        let mut state = prelude.begin_with_observer(input, observer)?;
        self.source.synchronize().await?;
        drop(prelude);

        for index in 0..self.config.num_refiner_layers {
            let block = self.source.load_context_refiner(index).await?;
            state.apply_context_refiner(&block)?;
            observer.rank3(
                &format!("context_refiner.{index}"),
                state
                    .instruction
                    .as_ref()
                    .expect("context refiner retains instruction")
                    .clone(),
            )?;
            self.source.synchronize().await?;
            drop(block);
        }
        for index in 0..self.config.num_refiner_layers {
            let block = self.source.load_noise_refiner(index).await?;
            state.apply_noise_refiner(&block)?;
            observer.rank3(
                &format!("noise_refiner.{index}"),
                state
                    .generated
                    .as_ref()
                    .expect("noise refiner retains generated stream")
                    .clone(),
            )?;
            self.source.synchronize().await?;
            drop(block);
        }
        if has_reference {
            for index in 0..self.config.num_refiner_layers {
                let block = self.source.load_reference_refiner(index).await?;
                state.apply_reference_refiner(&block)?;
                observer.rank3(
                    &format!("ref_image_refiner.{index}"),
                    state
                        .reference
                        .as_ref()
                        .expect("reference refiner retains reference stream")
                        .clone(),
                )?;
                self.source.synchronize().await?;
                drop(block);
            }
        }
        state.finish_refiners()?;

        for index in 0..self.config.num_double_stream_layers {
            let block = self.source.load_double_stream(index).await?;
            state.apply_double_stream(&block)?;
            observer.rank3(
                &format!("double_stream_layers.{index}.0"),
                state
                    .image
                    .as_ref()
                    .expect("double stream retains image")
                    .clone(),
            )?;
            observer.rank3(
                &format!("double_stream_layers.{index}.1"),
                state
                    .instruction
                    .as_ref()
                    .expect("double stream retains instruction")
                    .clone(),
            )?;
            self.source.synchronize().await?;
            drop(block);
        }
        state.finish_double_stream()?;
        for index in 0..self.config.num_single_stream_layers() {
            let block = self.source.load_single_stream(index).await?;
            state.apply_single_stream(&block)?;
            observer.rank3(
                &format!("single_stream_layers.{index}"),
                state
                    .joint
                    .as_ref()
                    .expect("single stream retains joint tokens")
                    .clone(),
            )?;
            self.source.synchronize().await?;
            drop(block);
        }
        let tail = self.source.load_tail().await?;
        let output = tail.finish_with_observer(state, observer)?;
        self.source.synchronize().await?;
        drop(tail);
        Ok(output)
    }

    /// Asynchronously predict one velocity without retaining observer activations.
    pub async fn predict_async(
        &mut self,
        input: BooguDenoiserInput<B>,
    ) -> Result<Tensor<B, 4>, BooguError> {
        self.predict_with_observer_async(input, &mut ()).await
    }
}

impl<B, S> DmdDenoiser<B> for StreamingBooguDenoiser<B, S>
where
    B: Backend,
    S: StreamingStageSource<B>,
{
    fn predict(&mut self, input: BooguDenoiserInput<B>) -> Result<Tensor<B, 4>, BooguError> {
        self.predict_with_observer(input, &mut ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn_ndarray::{NdArray, NdArrayDevice};

    type B = NdArray<f32>;

    struct MemoryStageSource {
        model: super::super::BooguDenoiser<B>,
        synchronizations: usize,
    }

    impl MemoryStageSource {
        fn prelude(&self) -> BooguDenoiserPrelude<B> {
            BooguDenoiserPrelude {
                x_embedder: self.model.x_embedder.clone(),
                ref_image_patch_embedder: self.model.ref_image_patch_embedder.clone(),
                time_caption_embed: self.model.time_caption_embed.clone(),
                image_index_embedding: self.model.image_index_embedding.clone(),
                config: self.model.config().clone(),
            }
        }

        fn tail(&self) -> BooguDenoiserTail<B> {
            BooguDenoiserTail {
                norm_out: self.model.norm_out.clone(),
                config: self.model.config().clone(),
            }
        }

        fn synchronized(&mut self) {
            self.synchronizations += 1;
        }
    }

    impl StreamingStageSource<B> for MemoryStageSource {
        fn load_prelude(&mut self) -> Result<BooguDenoiserPrelude<B>, BooguError> {
            Ok(self.prelude())
        }

        fn load_context_refiner(
            &mut self,
            index: usize,
        ) -> Result<SingleStreamBlock<B>, BooguError> {
            Ok(self.model.context_refiner[index].clone())
        }

        fn load_noise_refiner(&mut self, index: usize) -> Result<SingleStreamBlock<B>, BooguError> {
            Ok(self.model.noise_refiner[index].clone())
        }

        fn load_reference_refiner(
            &mut self,
            index: usize,
        ) -> Result<SingleStreamBlock<B>, BooguError> {
            Ok(self.model.ref_image_refiner[index].clone())
        }

        fn load_double_stream(&mut self, index: usize) -> Result<DoubleStreamBlock<B>, BooguError> {
            Ok(self.model.double_stream_layers[index].clone())
        }

        fn load_single_stream(&mut self, index: usize) -> Result<SingleStreamBlock<B>, BooguError> {
            Ok(self.model.single_stream_layers[index].clone())
        }

        fn load_tail(&mut self) -> Result<BooguDenoiserTail<B>, BooguError> {
            Ok(self.tail())
        }

        fn synchronize(&mut self) -> Result<(), BooguError> {
            self.synchronized();
            Ok(())
        }
    }

    impl AsyncBooguDenoiserStageSource<B> for MemoryStageSource {
        async fn load_prelude(&mut self) -> Result<BooguDenoiserPrelude<B>, BooguError> {
            Ok(self.prelude())
        }

        async fn load_context_refiner(
            &mut self,
            index: usize,
        ) -> Result<SingleStreamBlock<B>, BooguError> {
            Ok(self.model.context_refiner[index].clone())
        }

        async fn load_noise_refiner(
            &mut self,
            index: usize,
        ) -> Result<SingleStreamBlock<B>, BooguError> {
            Ok(self.model.noise_refiner[index].clone())
        }

        async fn load_reference_refiner(
            &mut self,
            index: usize,
        ) -> Result<SingleStreamBlock<B>, BooguError> {
            Ok(self.model.ref_image_refiner[index].clone())
        }

        async fn load_double_stream(
            &mut self,
            index: usize,
        ) -> Result<DoubleStreamBlock<B>, BooguError> {
            Ok(self.model.double_stream_layers[index].clone())
        }

        async fn load_single_stream(
            &mut self,
            index: usize,
        ) -> Result<SingleStreamBlock<B>, BooguError> {
            Ok(self.model.single_stream_layers[index].clone())
        }

        async fn load_tail(&mut self) -> Result<BooguDenoiserTail<B>, BooguError> {
            Ok(self.tail())
        }

        async fn synchronize(&mut self) -> Result<(), BooguError> {
            self.synchronized();
            Ok(())
        }
    }

    #[derive(Debug, PartialEq)]
    struct Boundary {
        name: String,
        shape: Vec<usize>,
        values: Vec<f32>,
    }

    #[derive(Default)]
    struct RecordingObserver {
        boundaries: Vec<Boundary>,
    }

    impl RecordingObserver {
        fn push<const D: usize>(&mut self, name: &str, tensor: Tensor<B, D>) {
            self.boundaries.push(Boundary {
                name: name.into(),
                shape: tensor.dims().to_vec(),
                values: tensor.into_data().to_vec::<f32>().unwrap(),
            });
        }
    }

    impl DenoiserStageObserver<B> for RecordingObserver {
        fn rank2(&mut self, name: &str, tensor: Tensor<B, 2>) -> Result<(), BooguError> {
            self.push(name, tensor);
            Ok(())
        }

        fn rank3(&mut self, name: &str, tensor: Tensor<B, 3>) -> Result<(), BooguError> {
            self.push(name, tensor);
            Ok(())
        }

        fn rank4(&mut self, name: &str, tensor: Tensor<B, 4>) -> Result<(), BooguError> {
            self.push(name, tensor);
            Ok(())
        }
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

    fn tiny_input(device: &NdArrayDevice) -> BooguDenoiserInput<B> {
        let latent_values = (0..64)
            .map(|index| (index as f32 - 31.0) / 64.0)
            .collect::<Vec<_>>();
        let reference_values = (0..64)
            .map(|index| (17.0 - index as f32) / 80.0)
            .collect::<Vec<_>>();
        let instruction_values = (0..24)
            .map(|index| (index as f32 - 12.0) / 32.0)
            .collect::<Vec<_>>();
        BooguDenoiserInput {
            latent: Tensor::from_data(TensorData::new(latent_values, [1, 4, 4, 4]), device),
            timestep: Tensor::from_data([0.375_f32], device),
            instruction: Tensor::from_data(TensorData::new(instruction_values, [1, 3, 8]), device),
            reference: Some(Tensor::from_data(
                TensorData::new(reference_values, [1, 4, 4, 4]),
                device,
            )),
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

    fn max_delta(left: &[f32], right: &[f32]) -> f32 {
        left.iter()
            .zip(right)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn async_streaming_matches_sync_boundaries_output_and_residency_correctness() {
        let config = tiny_config();
        let device = Default::default();
        B::seed(&device, 71);
        let resident = super::super::BooguDenoiser::<B>::new(config.clone(), &device).unwrap();
        let expected = resident.forward(tiny_input(&device)).unwrap();
        let expected = expected.into_data().to_vec::<f32>().unwrap();

        let source = MemoryStageSource {
            model: resident.clone(),
            synchronizations: 0,
        };
        let mut synchronous = StreamingBooguDenoiser::new(config.clone(), source).unwrap();
        let mut synchronous_observer = RecordingObserver::default();
        let synchronous_output = synchronous
            .predict_with_observer(tiny_input(&device), &mut synchronous_observer)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        let source = MemoryStageSource {
            model: resident.clone(),
            synchronizations: 0,
        };
        let mut asynchronous = StreamingBooguDenoiser::new(config.clone(), source).unwrap();
        let mut asynchronous_observer = RecordingObserver::default();
        let asynchronous_output = block_on_immediate(
            asynchronous
                .predict_with_observer_async(tiny_input(&device), &mut asynchronous_observer),
        )
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();

        let source = MemoryStageSource {
            model: resident,
            synchronizations: 0,
        };
        let mut unobserved = StreamingBooguDenoiser::new(config.clone(), source).unwrap();
        let unobserved_output = block_on_immediate(unobserved.predict_async(tiny_input(&device)))
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        assert!(max_delta(&synchronous_output, &expected) < 1.0e-5);
        assert!(max_delta(&asynchronous_output, &synchronous_output) < 1.0e-6);
        assert!(max_delta(&unobserved_output, &asynchronous_output) < 1.0e-6);
        assert_eq!(
            asynchronous_observer.boundaries.len(),
            synchronous_observer.boundaries.len()
        );
        for (asynchronous, synchronous) in asynchronous_observer
            .boundaries
            .iter()
            .zip(&synchronous_observer.boundaries)
        {
            assert_eq!(asynchronous.name, synchronous.name);
            assert_eq!(asynchronous.shape, synchronous.shape);
            assert!(max_delta(&asynchronous.values, &synchronous.values) < 1.0e-6);
        }

        let expected_synchronizations = 2
            + 3 * config.num_refiner_layers
            + config.num_double_stream_layers
            + config.num_single_stream_layers();
        assert_eq!(
            synchronous.source().synchronizations,
            expected_synchronizations
        );
        assert_eq!(
            asynchronous.source().synchronizations,
            expected_synchronizations
        );
        assert_eq!(
            unobserved.source().synchronizations,
            expected_synchronizations
        );
    }
}
