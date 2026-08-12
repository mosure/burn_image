use std::{sync::Arc, time::Instant};

use crate::{
    CancellationToken, ImageRequest, InferenceContext, ModelDescriptor, NoopProgressObserver,
    ProgressEvent, ProgressObserver, RunId, RuntimeConfig, RuntimeError,
};

/// Model adapter implemented by concrete model crates.
///
/// The associated output permits a WGPU implementation to return a
/// device-resident tensor without introducing a Burn dependency here.
pub trait ImageModel {
    type Output;

    fn descriptor(&self) -> &ModelDescriptor;

    fn infer(
        &mut self,
        request: &ImageRequest,
        context: &InferenceContext,
    ) -> Result<Self::Output, RuntimeError>;
}

/// Validating runtime wrapper around one loaded model implementation.
pub struct ImageRuntime<M> {
    config: RuntimeConfig,
    model: M,
    observer: Arc<dyn ProgressObserver>,
    cancellation: CancellationToken,
    next_run_id: u64,
}

impl<M: ImageModel> ImageRuntime<M> {
    pub fn new(config: RuntimeConfig, model: M) -> Result<Self, RuntimeError> {
        model.descriptor().validate()?;
        if config.model != model.descriptor().id {
            return Err(RuntimeError::ModelSelectionMismatch {
                selected: config.model,
                descriptor: model.descriptor().id.clone(),
            });
        }
        Ok(Self {
            config,
            model,
            observer: Arc::new(NoopProgressObserver),
            cancellation: CancellationToken::default(),
            next_run_id: 1,
        })
    }

    pub fn with_observer(mut self, observer: Arc<dyn ProgressObserver>) -> Self {
        self.observer = observer;
        self
    }

    pub fn set_observer(&mut self, observer: Arc<dyn ProgressObserver>) {
        self.observer = observer;
    }

    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    pub fn model(&self) -> &M {
        &self.model
    }

    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancellation.clone()
    }

    pub fn infer(&mut self, request: &ImageRequest) -> Result<M::Output, RuntimeError> {
        self.model
            .descriptor()
            .capabilities
            .validate_request(&self.config.model, request)?;
        self.cancellation.check()?;

        let run_id = RunId(self.next_run_id);
        self.next_run_id = self.next_run_id.wrapping_add(1).max(1);
        let context = InferenceContext::new(
            run_id,
            Arc::clone(&self.observer),
            self.cancellation.clone(),
        );
        context.emit(ProgressEvent::RunStarted {
            run_id,
            model: self.config.model.clone(),
            task: request.task_kind(),
        });
        let started = Instant::now();
        let result = self.model.infer(request, &context);
        match result {
            Ok(_output) if self.cancellation.is_cancelled() => {
                context.emit(ProgressEvent::RunCancelled { run_id });
                Err(RuntimeError::Cancelled)
            }
            Ok(output) => {
                context.emit(ProgressEvent::RunCompleted {
                    run_id,
                    elapsed_micros: duration_micros(started.elapsed()),
                });
                Ok(output)
            }
            Err(RuntimeError::Cancelled) => {
                context.emit(ProgressEvent::RunCancelled { run_id });
                Err(RuntimeError::Cancelled)
            }
            Err(error) => {
                context.emit(ProgressEvent::RunFailed {
                    run_id,
                    message: error.to_string(),
                });
                Err(error)
            }
        }
    }

    pub fn into_model(self) -> M {
        self.model
    }
}

fn duration_micros(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeSet,
        sync::{Arc, Mutex},
    };

    use crate::{
        ArtifactCachePolicy, ArtifactProfileId, ArtifactSource, DimensionConstraints,
        GenerateRequest, GenerationOptions, ImageModel, ImageRequest, ImageRuntime, ImageTaskKind,
        IntegrityPolicy, ModelCapabilities, ModelDescriptor, ModelId, NumericFormat, ProgressEvent,
        Prompt, RemoteBaseUrl, RuntimeConfig, RuntimeError,
    };

    struct FakeModel {
        descriptor: ModelDescriptor,
        calls: usize,
    }

    impl ImageModel for FakeModel {
        type Output = usize;

        fn descriptor(&self) -> &ModelDescriptor {
            &self.descriptor
        }

        fn infer(
            &mut self,
            _request: &ImageRequest,
            context: &crate::InferenceContext,
        ) -> Result<Self::Output, RuntimeError> {
            context.stage_started("fake", Some(1));
            context.check_cancelled()?;
            self.calls += 1;
            context.step("fake", 1, 1, 1);
            context.stage_completed("fake", 1);
            Ok(self.calls)
        }
    }

    fn descriptor() -> ModelDescriptor {
        ModelDescriptor {
            id: ModelId::new("test/model").unwrap(),
            display_name: "Test Model".to_string(),
            revision: "abc123".to_string(),
            capabilities: ModelCapabilities {
                tasks: BTreeSet::from([ImageTaskKind::Generate]),
                supports_masks: false,
                dimensions: DimensionConstraints {
                    min_width: 64,
                    max_width: 1024,
                    min_height: 64,
                    max_height: 1024,
                    width_multiple: 64,
                    height_multiple: 64,
                    max_pixels: None,
                    allowed_dimensions: None,
                },
                min_steps: 1,
                max_steps: 10,
                max_batch_size: 1,
                numeric_formats: BTreeSet::from([NumericFormat::F16]),
            },
        }
    }

    fn runtime() -> ImageRuntime<FakeModel> {
        let descriptor = descriptor();
        let config = RuntimeConfig {
            model: descriptor.id.clone(),
            artifact_profile: ArtifactProfileId::new("test-f16").unwrap(),
            artifact_source: ArtifactSource::Remote {
                base_url: RemoteBaseUrl::new("https://example.test/models").unwrap(),
            },
            integrity: IntegrityPolicy::RequireSha256,
            cache: ArtifactCachePolicy::UseCached,
        };
        ImageRuntime::new(
            config,
            FakeModel {
                descriptor,
                calls: 0,
            },
        )
        .unwrap()
    }

    fn request(steps: u32) -> ImageRequest {
        ImageRequest::Generate(GenerateRequest {
            prompt: Prompt::new("test prompt").unwrap(),
            negative_prompt: None,
            options: GenerationOptions {
                steps: Some(steps),
                ..GenerationOptions::default()
            },
        })
    }

    #[test]
    fn runtime_validates_before_model_dispatch_and_emits_progress_correctness() {
        let events = Arc::new(Mutex::new(Vec::<ProgressEvent>::new()));
        let observer_events = Arc::clone(&events);
        let observer = Arc::new(move |event: &ProgressEvent| {
            observer_events.lock().unwrap().push(event.clone());
        });
        let mut runtime = runtime().with_observer(observer);

        assert!(runtime.infer(&request(11)).is_err());
        assert_eq!(runtime.model().calls, 0);
        assert_eq!(runtime.infer(&request(2)).unwrap(), 1);

        let events = events.lock().unwrap();
        assert!(matches!(
            events.first(),
            Some(ProgressEvent::RunStarted { .. })
        ));
        assert!(matches!(
            events.last(),
            Some(ProgressEvent::RunCompleted { .. })
        ));
    }

    #[test]
    fn cancellation_prevents_dispatch_correctness() {
        let mut runtime = runtime();
        runtime.cancellation_token().cancel();
        assert_eq!(runtime.infer(&request(2)), Err(RuntimeError::Cancelled));
        assert_eq!(runtime.model().calls, 0);
    }
}
