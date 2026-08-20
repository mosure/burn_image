//! Complete host request adapter for an already loaded resident Boogu pipeline.

use std::time::{Duration, Instant};

#[cfg(not(target_arch = "wasm32"))]
use std::thread::{self, JoinHandle};

use burn::{
    prelude::Backend,
    tensor::{DType, Tensor, TensorData},
};
use burn_image::{
    GeneratedImage, ImageModel, ImageOutput, ImageRequest, InferenceContext, InputImage,
    ModelDescriptor, ModelProvenance, NumericFormat, RuntimeError, Sha256Digest, StageTiming,
    StageTimings,
};
use burn_qwen3_vl::{Qwen3VlImageProcessor, Qwen3VlProcessor, Qwen3VlTokenizer};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::{
    BooguDmdInput, BooguError, BooguExecution, DmdSchedule, ResidentBooguPipeline,
    boogu_model_descriptor, decode_input_image, decoder_output_to_host, prepare_instruction,
    prepare_vae_reference, resolve_request,
};

/// Floating-point activation dtypes at the three host/model boundaries.
///
/// Storage dtype is not sufficient to infer these values. Native Q8 Qwen stages are dequantized
/// through F16, native Q8 denoiser matrices remain quantized with F32 activations, and browser
/// runtimes may adapt all stored floats to F32. The released FLUX VAE follows its upstream
/// `force_upcast=true` policy and executes in F32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BooguRuntimeDTypes {
    /// Preprocessed visual patch dtype expected by the loaded Qwen vision stages.
    pub qwen_visual: DType,
    /// Normalized pixels, posterior noise, and decoder-latent dtype expected by the loaded VAE.
    pub vae: DType,
    /// Latent, timestep, instruction, reference, and renoise dtype expected by the denoiser.
    pub denoiser: DType,
}

impl BooguRuntimeDTypes {
    /// Define an explicit runtime boundary policy.
    pub const fn new(qwen_visual: DType, vae: DType, denoiser: DType) -> Self {
        Self {
            qwen_visual,
            vae,
            denoiser,
        }
    }

    /// Derive activation dtypes from a verified storage profile and the actual float-load
    /// policies selected for the VAE and denoiser.
    #[cfg(feature = "burnpack")]
    pub const fn from_artifact_policies(
        profile: crate::artifacts::BooguStorageProfile,
        vae_policy: crate::artifacts::BooguFloatLoadPolicy,
        denoiser_policy: crate::artifacts::BooguFloatLoadPolicy,
    ) -> Self {
        use crate::artifacts::{BooguFloatLoadPolicy, BooguStorageProfile};

        let qwen_visual = match profile {
            BooguStorageProfile::F16 | BooguStorageProfile::Q8sBlock32F32 => DType::F16,
            BooguStorageProfile::F16QwenVisionF32
            | BooguStorageProfile::Q8sBlock32F32QwenVisionF32
            | BooguStorageProfile::Q4sBlockUpTo128F32 => DType::F32,
        };
        let vae = match vae_policy {
            BooguFloatLoadPolicy::Preserve
            | BooguFloatLoadPolicy::PackedF16WeightsF32Auxiliaries
            | BooguFloatLoadPolicy::PackedQ4sWeightsF32Auxiliaries => DType::F16,
            BooguFloatLoadPolicy::AdaptToF32 => DType::F32,
        };
        let denoiser = match denoiser_policy {
            BooguFloatLoadPolicy::Preserve => DType::F16,
            BooguFloatLoadPolicy::AdaptToF32
            | BooguFloatLoadPolicy::PackedF16WeightsF32Auxiliaries
            | BooguFloatLoadPolicy::PackedQ4sWeightsF32Auxiliaries => DType::F32,
        };
        Self::new(qwen_visual, vae, denoiser)
    }

    fn validate(self) -> Result<(), BooguError> {
        for (name, dtype) in [
            ("Qwen visual", self.qwen_visual),
            ("VAE", self.vae),
            ("denoiser", self.denoiser),
        ] {
            if !dtype.is_float() {
                return Err(BooguError::InvalidConfig(format!(
                    "{name} execution dtype must be floating point, got {}",
                    dtype.name()
                )));
            }
        }
        Ok(())
    }
}

/// Provenance fixed when an artifact bundle and backend are selected.
#[derive(Debug, Clone)]
pub struct BooguRuntimeMetadata {
    /// Artifact storage profile represented in the result.
    pub numeric_format: NumericFormat,
    /// Concrete backend/adapter label (for example `wgpu-vulkan/RTX-PRO-6000`).
    pub backend: String,
    /// Sealed artifact bundle content digest.
    pub artifact_content_digest: Option<Sha256Digest>,
    /// Whether every required artifact was SHA-256 verified before construction.
    pub artifacts_verified: bool,
    /// Activation dtypes derived from the selected verified profile and load policies.
    pub execution_dtypes: BooguRuntimeDTypes,
    /// Host-provided entropy used only when a request omits its seed.
    pub default_seed: u64,
}

impl BooguRuntimeMetadata {
    /// Validate provenance against a concrete immutable model descriptor.
    pub fn validate_for(&self, descriptor: &ModelDescriptor) -> Result<(), BooguError> {
        if self.backend.trim().is_empty() {
            return Err(BooguError::InvalidConfig(
                "runtime backend label must not be empty".into(),
            ));
        }
        self.numeric_format
            .validate()
            .map_err(|error| BooguError::InvalidConfig(error.to_string()))?;
        self.execution_dtypes.validate()?;
        if self.artifacts_verified && self.artifact_content_digest.is_none() {
            return Err(BooguError::InvalidConfig(
                "verified runtime provenance requires the sealed bundle digest".into(),
            ));
        }
        if !descriptor
            .capabilities
            .numeric_formats
            .contains(&self.numeric_format)
        {
            return Err(BooguError::InvalidConfig(format!(
                "artifact numeric format {:?} is not advertised by {}",
                self.numeric_format, descriptor.id
            )));
        }
        Ok(())
    }
}

/// A resident Boogu pipeline exposed through the model-neutral image API.
///
/// Weight loading remains separate so native applications can choose a verified resident loader,
/// while browser applications can implement the same public request contract with streamed
/// component executors.
pub struct BooguImageModel<B: Backend, T, E = ResidentBooguPipeline<B>> {
    descriptor: ModelDescriptor,
    pipeline: E,
    processor: Qwen3VlProcessor<T>,
    image_processor: Qwen3VlImageProcessor,
    device: B::Device,
    metadata: BooguRuntimeMetadata,
    conditioning_cache: Option<ConditioningCache<B>>,
    release_unused_memory_at_phase_boundaries: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ConditioningCacheKey {
    prompt: String,
    source: Option<InputImage>,
    effective_length: usize,
}

struct ConditioningCache<B: Backend> {
    key: ConditioningCacheKey,
    instruction: Tensor<B, 3>,
}

impl<B: Backend> ConditioningCache<B> {
    fn instruction_for(&self, key: &ConditioningCacheKey) -> Option<Tensor<B, 3>> {
        (self.key == *key).then(|| self.instruction.clone())
    }
}

impl<B, T, E> BooguImageModel<B, T, E>
where
    B: Backend,
    T: Qwen3VlTokenizer,
    E: BooguExecution<B>,
{
    /// Compose a loaded resident model with deterministic host processing.
    pub fn new(
        pipeline: E,
        processor: Qwen3VlProcessor<T>,
        image_processor: Qwen3VlImageProcessor,
        device: B::Device,
        metadata: BooguRuntimeMetadata,
    ) -> Result<Self, BooguError> {
        let descriptor = boogu_model_descriptor(pipeline.variant());
        metadata.validate_for(&descriptor)?;
        Ok(Self {
            descriptor,
            pipeline,
            processor,
            image_processor,
            device,
            metadata,
            conditioning_cache: None,
            release_unused_memory_at_phase_boundaries: false,
        })
    }

    /// Release unused backend allocator pages after each completed model phase.
    ///
    /// This is intended for explicitly memory-bounded runtimes whose phase-resident weights are
    /// already synchronized before the next component begins. Live tensors and resident model
    /// parameters remain referenced; only allocator pages with no live handles are eligible for
    /// release. The default preserves the backend allocator cache.
    pub const fn with_phase_boundary_memory_cleanup(mut self, enabled: bool) -> Self {
        self.release_unused_memory_at_phase_boundaries = enabled;
        self
    }

    /// Whether unused allocator pages are released at synchronized model-phase boundaries.
    pub const fn phase_boundary_memory_cleanup_enabled(&self) -> bool {
        self.release_unused_memory_at_phase_boundaries
    }

    /// Access the composed resident pipeline.
    pub fn pipeline(&self) -> &E {
        &self.pipeline
    }

    /// Mutably access the composed resident pipeline.
    pub fn pipeline_mut(&mut self) -> &mut E {
        // A caller may replace or mutate the Qwen executor through this handle. Any resident
        // conditioning was produced by the previous pipeline state and must not survive that
        // mutation boundary.
        self.conditioning_cache = None;
        &mut self.pipeline
    }

    fn infer_inner(
        &mut self,
        request: &ImageRequest,
        context: &InferenceContext,
    ) -> Result<ImageOutput, BooguError> {
        let total_started = Instant::now();
        let resolved =
            resolve_request(self.pipeline.variant(), request, self.metadata.default_seed)?;
        let source = resolved
            .source
            .as_ref()
            .map(decode_input_image)
            .transpose()?;
        context.check_cancelled().map_err(cancelled)?;

        let latent_shape = [
            1,
            16,
            resolved.dimensions.height() as usize / 8,
            resolved.dimensions.width() as usize / 8,
        ];
        let mut timings = Vec::new();
        context.stage_started("processing", Some(1));
        let started = Instant::now();
        let mut prepared = prepare_instruction::<B, T>(
            &resolved,
            source.as_ref(),
            &self.processor,
            &self.image_processor,
            &self.device,
        )?;
        cast_visual_inputs(
            &mut prepared.model_input,
            self.metadata.execution_dtypes.qwen_visual,
        );
        let conditioning_key = ConditioningCacheKey {
            prompt: resolved.prompt.clone(),
            source: resolved.source.clone(),
            effective_length: prepared.effective_length,
        };
        finish_stage::<B>(
            &self.device,
            context,
            &mut timings,
            "processing",
            started,
            false,
        )?;

        context.check_cancelled().map_err(cancelled)?;
        let cached_instruction = self
            .conditioning_cache
            .as_ref()
            .and_then(|cached| cached.instruction_for(&conditioning_key));
        // These four deterministic host buffers are independent of conditioning. Generate them
        // while Qwen (and, for edits, the VAE encoder) occupies the GPU, then perform the same
        // device uploads at the original point immediately before DMD sampling.
        let noise_worker = DmdNoiseWorker::spawn(
            latent_shape,
            resolved.seed,
            should_overlap_dmd_noise(cached_instruction.is_some(), source.is_some()),
        );
        context.stage_started("qwen", Some(1));
        let started = Instant::now();
        let instruction = if let Some(instruction) = cached_instruction {
            instruction
        } else {
            let instruction = self
                .pipeline
                .encode_instruction(prepared.model_input, prepared.effective_length)?
                .cast(self.metadata.execution_dtypes.denoiser);
            self.conditioning_cache = Some(ConditioningCache {
                key: conditioning_key,
                instruction: instruction.clone(),
            });
            instruction
        };
        finish_stage::<B>(
            &self.device,
            context,
            &mut timings,
            "qwen",
            started,
            self.release_unused_memory_at_phase_boundaries,
        )?;

        let reference = if let Some(source) = source.as_ref() {
            context.check_cancelled().map_err(cancelled)?;
            context.stage_started("vae-encode", Some(1));
            let started = Instant::now();
            let normalized = prepare_vae_reference::<B>(source, &self.device)?
                .cast(self.metadata.execution_dtypes.vae);
            let [_, _, height, width] = normalized.dims();
            let epsilon = normal_tensor::<B, 4>(
                [1, 16, height / 8, width / 8],
                domain_seed(resolved.seed, 0x5641_452d_454e_434f),
                self.metadata.execution_dtypes.vae,
                &self.device,
            );
            let reference = self
                .pipeline
                .encode_reference(normalized, epsilon)?
                .cast(self.metadata.execution_dtypes.denoiser);
            finish_stage::<B>(
                &self.device,
                context,
                &mut timings,
                "vae-encode",
                started,
                self.release_unused_memory_at_phase_boundaries,
            )?;
            Some(reference)
        } else {
            None
        };

        let noise = noise_worker.join()?;
        context.check_cancelled().map_err(cancelled)?;
        let initial_latents = tensor_from_f32_values::<B, 4>(
            noise.initial_latents,
            latent_shape,
            self.metadata.execution_dtypes.denoiser,
            &self.device,
        );
        let renoise = noise
            .renoise
            .into_iter()
            .map(|values| {
                tensor_from_f32_values::<B, 4>(
                    values,
                    latent_shape,
                    self.metadata.execution_dtypes.denoiser,
                    &self.device,
                )
            })
            .collect();

        context.check_cancelled().map_err(cancelled)?;
        context.stage_started("dmd", Some(resolved.steps));
        let started = Instant::now();
        let latents = self.pipeline.denoise_with_observer(
            BooguDmdInput {
                execution_dtype: self.metadata.execution_dtypes.denoiser,
                initial_latents,
                instruction,
                reference,
                renoise,
                schedule: DmdSchedule::upstream_for_dtype(
                    resolved.task,
                    self.metadata.execution_dtypes.denoiser,
                ),
            },
            |index, _sigma| {
                context.check_cancelled().map_err(cancelled)?;
                context.step(
                    "dmd",
                    index as u32 + 1,
                    resolved.steps,
                    elapsed_micros(started.elapsed()),
                );
                Ok(())
            },
        )?;
        finish_stage::<B>(
            &self.device,
            context,
            &mut timings,
            "dmd",
            started,
            self.release_unused_memory_at_phase_boundaries,
        )?;

        context.check_cancelled().map_err(cancelled)?;
        context.stage_started("vae-decode", Some(1));
        let started = Instant::now();
        let decoded = self
            .pipeline
            .decode(latents.cast(self.metadata.execution_dtypes.vae))?;
        finish_stage::<B>(
            &self.device,
            context,
            &mut timings,
            "vae-decode",
            started,
            self.release_unused_memory_at_phase_boundaries,
        )?;

        context.check_cancelled().map_err(cancelled)?;
        context.stage_started("output", Some(1));
        let started = Instant::now();
        let image = decoder_output_to_host(decoded)?;
        finish_stage::<B>(
            &self.device,
            context,
            &mut timings,
            "output",
            started,
            self.release_unused_memory_at_phase_boundaries,
        )?;

        let output = ImageOutput {
            images: vec![GeneratedImage { index: 0, image }],
            seed: resolved.seed,
            timings: StageTimings {
                stages: timings,
                total_micros: elapsed_micros(total_started.elapsed()),
            },
            provenance: ModelProvenance {
                model: self.descriptor.id.clone(),
                model_revision: self.descriptor.revision.clone(),
                artifact_content_digest: self.metadata.artifact_content_digest,
                numeric_format: self.metadata.numeric_format.clone(),
                backend: self.metadata.backend.clone(),
                artifacts_verified: self.metadata.artifacts_verified,
            },
        };
        output
            .validate()
            .map_err(|error| BooguError::InvalidShape(error.to_string()))?;
        Ok(output)
    }
}

impl<B, T, E> ImageModel for BooguImageModel<B, T, E>
where
    B: Backend,
    T: Qwen3VlTokenizer,
    E: BooguExecution<B>,
{
    type Output = ImageOutput;

    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn infer(
        &mut self,
        request: &ImageRequest,
        context: &InferenceContext,
    ) -> Result<Self::Output, RuntimeError> {
        let result = self.infer_inner(request, context);
        let cleanup_error = if result.is_err() && self.release_unused_memory_at_phase_boundaries {
            release_unused_backend_memory::<B>(&self.device, "failed inference").err()
        } else {
            None
        };

        result.map_err(|error| {
            if let Some(cleanup_error) = cleanup_error {
                return RuntimeError::ModelExecution {
                    model: self.descriptor.id.clone(),
                    message: format!(
                        "{error}; backend cleanup after failed or cancelled inference also failed: {cleanup_error}"
                    ),
                };
            }
            match error {
                BooguError::Cancelled => RuntimeError::Cancelled,
                error => RuntimeError::ModelExecution {
                    model: self.descriptor.id.clone(),
                    message: error.to_string(),
                },
            }
        })
    }
}

fn release_unused_backend_memory<B: Backend>(
    device: &B::Device,
    phase: &str,
) -> Result<(), BooguError> {
    B::sync(device).map_err(|error| {
        BooguError::InvalidRequest(format!(
            "backend synchronization before {phase} allocator cleanup failed: {error}"
        ))
    })?;
    B::memory_cleanup(device);
    B::sync(device).map_err(|error| {
        BooguError::InvalidRequest(format!(
            "backend synchronization after {phase} allocator cleanup failed: {error}"
        ))
    })?;
    Ok(())
}

fn finish_stage<B: Backend>(
    device: &B::Device,
    context: &InferenceContext,
    timings: &mut Vec<StageTiming>,
    name: &str,
    started: Instant,
    release_unused_memory: bool,
) -> Result<(), BooguError> {
    if release_unused_memory {
        release_unused_backend_memory::<B>(device, name)?;
    } else {
        B::sync(device).map_err(|error| {
            BooguError::InvalidRequest(format!("backend synchronization failed: {error}"))
        })?;
    }
    let elapsed = elapsed_micros(started.elapsed());
    timings.push(StageTiming {
        stage: name.into(),
        elapsed_micros: elapsed,
    });
    context.stage_completed(name, elapsed);
    Ok(())
}

fn normal_tensor<B: Backend, const D: usize>(
    shape: [usize; D],
    seed: u64,
    dtype: DType,
    device: &B::Device,
) -> Tensor<B, D> {
    let count = shape.iter().product();
    let values = normal_values(count, seed);
    tensor_from_f32_values(values, shape, dtype, device)
}

fn normal_values(count: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha12Rng::seed_from_u64(seed);
    StandardNormal.sample_iter(&mut rng).take(count).collect()
}

fn tensor_from_f32_values<B: Backend, const D: usize>(
    values: Vec<f32>,
    shape: [usize; D],
    dtype: DType,
    device: &B::Device,
) -> Tensor<B, D> {
    Tensor::<B, D>::from_data(TensorData::new(values, shape), device).cast(dtype)
}

struct DmdNoiseHostBuffers {
    initial_latents: Vec<f32>,
    renoise: [Vec<f32>; 3],
}

impl DmdNoiseHostBuffers {
    fn generate(shape: [usize; 4], seed: u64) -> Self {
        let count = shape.into_iter().product();
        let initial_latents = normal_values(count, domain_seed(seed, 0x444d_442d_494e_4954));
        let renoise = std::array::from_fn(|index| {
            normal_values(
                count,
                domain_seed(seed, 0x444d_442d_4e4f_4953 ^ index as u64),
            )
        });
        Self {
            initial_latents,
            renoise,
        }
    }
}

fn should_overlap_dmd_noise(has_cached_instruction: bool, has_reference: bool) -> bool {
    !has_cached_instruction || has_reference
}

#[cfg(not(target_arch = "wasm32"))]
enum DmdNoiseWorker {
    Threaded(Option<JoinHandle<DmdNoiseHostBuffers>>),
    Deferred { shape: [usize; 4], seed: u64 },
}

#[cfg(not(target_arch = "wasm32"))]
impl DmdNoiseWorker {
    fn spawn(shape: [usize; 4], seed: u64, overlap: bool) -> Self {
        if !overlap {
            return Self::Deferred { shape, seed };
        }
        match thread::Builder::new()
            .name("boogu-dmd-noise".into())
            .spawn(move || DmdNoiseHostBuffers::generate(shape, seed))
        {
            Ok(handle) => Self::Threaded(Some(handle)),
            // Thread creation is only an optimization. Preserve availability and exact output by
            // falling back to generation at the original serial point.
            Err(_) => Self::Deferred { shape, seed },
        }
    }

    fn join(mut self) -> Result<DmdNoiseHostBuffers, BooguError> {
        match &mut self {
            Self::Threaded(handle) => handle
                .take()
                .expect("DMD noise worker handle must exist until join")
                .join()
                .map_err(|_| {
                    BooguError::InvalidRequest("deterministic DMD noise worker panicked".into())
                }),
            Self::Deferred { shape, seed } => Ok(DmdNoiseHostBuffers::generate(*shape, *seed)),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for DmdNoiseWorker {
    fn drop(&mut self) {
        if let Self::Threaded(handle) = self
            && let Some(handle) = handle.take()
        {
            // Error and cancellation paths still wait for the bounded worker instead of
            // detaching it. A panic is reported by `join` on the success path; there is no
            // useful error channel left while another error is already being returned.
            let _ = handle.join();
        }
    }
}

#[cfg(target_arch = "wasm32")]
struct DmdNoiseWorker {
    shape: [usize; 4],
    seed: u64,
}

#[cfg(target_arch = "wasm32")]
impl DmdNoiseWorker {
    fn spawn(shape: [usize; 4], seed: u64, _overlap: bool) -> Self {
        Self { shape, seed }
    }

    fn join(self) -> Result<DmdNoiseHostBuffers, BooguError> {
        Ok(DmdNoiseHostBuffers::generate(self.shape, self.seed))
    }
}

fn cast_visual_inputs<B: Backend>(input: &mut burn_qwen3_vl::Qwen3VlModelInput<B>, dtype: DType) {
    for visual in [&mut input.images, &mut input.videos].into_iter().flatten() {
        visual.patches = visual.patches.clone().cast(dtype);
    }
}

fn domain_seed(seed: u64, domain: u64) -> u64 {
    // SplitMix64 finalizer: stable domain separation without relying on process hash state.
    let mut value = seed ^ domain;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn elapsed_micros(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

fn cancelled(_error: RuntimeError) -> BooguError {
    BooguError::Cancelled
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;

    use super::*;

    type B = NdArray<f32>;

    #[test]
    fn noise_is_domain_separated_and_reproducible_correctness() {
        let device = Default::default();
        let first = normal_tensor::<B, 1>([128], domain_seed(42, 1), DType::F32, &device)
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        let again = normal_tensor::<B, 1>([128], domain_seed(42, 1), DType::F32, &device)
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        let other = normal_tensor::<B, 1>([128], domain_seed(42, 2), DType::F32, &device)
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(first, again);
        assert_ne!(first, other);
        assert!(first.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn overlapped_dmd_noise_matches_serial_tensor_generation_correctness() {
        fn serial_reference<const D: usize>(
            shape: [usize; D],
            seed: u64,
            dtype: DType,
        ) -> Tensor<B, D> {
            let mut rng = ChaCha12Rng::seed_from_u64(seed);
            let values = StandardNormal
                .sample_iter(&mut rng)
                .take(shape.iter().product())
                .collect::<Vec<f32>>();
            Tensor::<B, D>::from_data(TensorData::new(values, shape), &Default::default())
                .cast(dtype)
        }

        let shape = [1, 16, 7, 11];
        let seed = 0x0123_4567_89ab_cdef;
        let actual = DmdNoiseWorker::spawn(shape, seed, true).join().unwrap();
        let deferred = DmdNoiseWorker::spawn(shape, seed, false).join().unwrap();
        assert_eq!(actual.initial_latents, deferred.initial_latents);
        assert_eq!(actual.renoise, deferred.renoise);
        let expected_seeds = [
            domain_seed(seed, 0x444d_442d_494e_4954),
            domain_seed(seed, 0x444d_442d_4e4f_4953),
            domain_seed(seed, 0x444d_442d_4e4f_4953 ^ 1),
            domain_seed(seed, 0x444d_442d_4e4f_4953 ^ 2),
        ];
        let actual_values = std::iter::once(&actual.initial_latents).chain(actual.renoise.iter());
        let device = Default::default();

        for (values, noise_seed) in actual_values.zip(expected_seeds) {
            let expected_f32 = serial_reference(shape, noise_seed, DType::F32);
            let actual_f32 =
                tensor_from_f32_values::<B, 4>(values.clone(), shape, DType::F32, &device);
            assert_eq!(actual_f32.into_data(), expected_f32.into_data());
        }
    }

    #[test]
    fn dmd_noise_overlap_requires_gpu_conditioning_work_correctness() {
        assert!(should_overlap_dmd_noise(false, false));
        assert!(should_overlap_dmd_noise(false, true));
        assert!(should_overlap_dmd_noise(true, true));
        assert!(!should_overlap_dmd_noise(true, false));
    }

    #[test]
    fn stage_completion_requires_successful_backend_sync_correctness() {
        let source = include_str!("runtime.rs");
        let finish_stage = source
            .split("fn finish_stage<B: Backend>(")
            .nth(1)
            .expect("finish_stage must remain present")
            .split("fn normal_tensor<")
            .next()
            .expect("finish_stage must end before normal_tensor");

        assert!(finish_stage.contains("release_unused_backend_memory::<B>(device, name)?;"));
        let direct_sync = finish_stage
            .find("B::sync(device).map_err")
            .expect("ordinary stage completion must synchronize the backend");
        let completion = finish_stage
            .find("context.stage_completed(name, elapsed);")
            .expect("successful synchronization must report stage completion");
        assert!(direct_sync < completion);
        assert!(finish_stage[direct_sync..completion].contains(")?;"));
    }

    #[test]
    fn runtime_metadata_rejects_unidentified_verified_artifacts_correctness() {
        let metadata = BooguRuntimeMetadata {
            numeric_format: NumericFormat::F16,
            backend: "wgpu-test".into(),
            artifact_content_digest: None,
            artifacts_verified: true,
            execution_dtypes: BooguRuntimeDTypes::new(DType::F32, DType::F32, DType::F32),
            default_seed: 0,
        };
        let descriptor = boogu_model_descriptor(crate::BooguVariant::Image01Turbo);
        assert!(metadata.validate_for(&descriptor).is_err());
    }

    #[test]
    fn conditioning_cache_requires_exact_prompt_source_and_length_correctness() {
        let device = Default::default();
        let key = ConditioningCacheKey {
            prompt: "same prompt".into(),
            source: None,
            effective_length: 7,
        };
        let cache = ConditioningCache::<B> {
            key: key.clone(),
            instruction: Tensor::ones([1, 7, 4], &device),
        };

        assert!(cache.instruction_for(&key).is_some());
        let mut different_prompt = key.clone();
        different_prompt.prompt.push('!');
        assert!(cache.instruction_for(&different_prompt).is_none());
        let mut different_length = key;
        different_length.effective_length += 1;
        assert!(cache.instruction_for(&different_length).is_none());
    }

    #[cfg(feature = "burnpack")]
    #[test]
    fn released_mixed_profile_derives_model_execution_dtypes_from_load_policies_correctness() {
        use crate::artifacts::{BooguFloatLoadPolicy, BooguStorageProfile};

        let dtypes = BooguRuntimeDTypes::from_artifact_policies(
            BooguStorageProfile::F16QwenVisionF32,
            BooguFloatLoadPolicy::AdaptToF32,
            BooguFloatLoadPolicy::Preserve,
        );
        assert_eq!(dtypes.qwen_visual, DType::F32);
        assert_eq!(dtypes.vae, DType::F32);
        assert_eq!(dtypes.denoiser, DType::F16);
    }

    #[cfg(feature = "burnpack")]
    #[test]
    fn packed_q4_profile_uses_f16_vae_and_f32_quantized_matmul_boundaries_correctness() {
        use crate::artifacts::{BooguFloatLoadPolicy, BooguStorageProfile};

        let dtypes = BooguRuntimeDTypes::from_artifact_policies(
            BooguStorageProfile::Q4sBlockUpTo128F32,
            BooguFloatLoadPolicy::PackedQ4sWeightsF32Auxiliaries,
            BooguFloatLoadPolicy::PackedQ4sWeightsF32Auxiliaries,
        );
        assert_eq!(dtypes.qwen_visual, DType::F32);
        assert_eq!(dtypes.vae, DType::F16);
        assert_eq!(dtypes.denoiser, DType::F32);
    }
}
