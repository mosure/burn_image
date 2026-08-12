use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Mutex,
};

use bevy::prelude::*;
use burn_image::{
    CancellationToken, ImageOutput, ImageRequest, ModelDescriptor, ModelId, ProgressEvent,
    RuntimeError,
};
use serde::{Deserialize, Serialize};

use crate::{
    CompleteImageJob, FailImageJob, FrontendError, ImageFrontendSet, ImageJobCancellationRequested,
    ImageJobDispatched, ImageJobId, ReportImageProgress,
};

/// The only execution modes accepted by the frontend. There is deliberately
/// no CPU or mock variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WgpuExecutionKind {
    NativeWgpu,
    BrowserWebGpu,
}

/// Canonical model descriptors plus runner behavior required by the UI.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageRunnerCapabilities {
    pub execution: WgpuExecutionKind,
    pub models: Vec<ModelDescriptor>,
    pub streams_progress: bool,
    pub cooperative_cancellation: bool,
    pub returns_host_images: bool,
}

impl ImageRunnerCapabilities {
    pub fn validate(&self) -> Result<(), FrontendError> {
        if !self.streams_progress || !self.cooperative_cancellation || !self.returns_host_images {
            return Err(FrontendError::model_runtime(
                "runner must stream progress, support cooperative cancellation, and return canonical host images",
            ));
        }
        if self.models.is_empty() {
            return Err(FrontendError::model_runtime(
                "runner advertises no burn_image model descriptors",
            ));
        }
        let mut ids = BTreeSet::new();
        for descriptor in &self.models {
            descriptor.validate()?;
            if !ids.insert(descriptor.id.clone()) {
                return Err(FrontendError::model_runtime(format!(
                    "runner advertises model '{}' more than once",
                    descriptor.id
                )));
            }
        }
        Ok(())
    }

    pub fn descriptor(&self, model: &ModelId) -> Option<&ModelDescriptor> {
        self.models
            .iter()
            .find(|descriptor| descriptor.id == *model)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ImageRunnerState {
    Missing,
    Initializing {
        message: String,
    },
    Ready {
        capabilities: ImageRunnerCapabilities,
    },
    Failed {
        error: FrontendError,
    },
}

#[derive(Resource, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageRunnerStatus {
    pub state: ImageRunnerState,
}

impl Default for ImageRunnerStatus {
    fn default() -> Self {
        Self {
            state: ImageRunnerState::Missing,
        }
    }
}

impl ImageRunnerStatus {
    pub fn initializing(message: impl Into<String>) -> Self {
        Self {
            state: ImageRunnerState::Initializing {
                message: message.into(),
            },
        }
    }

    pub fn ready(capabilities: ImageRunnerCapabilities) -> Result<Self, FrontendError> {
        capabilities.validate()?;
        Ok(Self {
            state: ImageRunnerState::Ready { capabilities },
        })
    }

    pub fn validate_request(
        &self,
        model: &ModelId,
        request: &ImageRequest,
    ) -> Result<(), FrontendError> {
        match &self.state {
            ImageRunnerState::Missing => Err(FrontendError::model_runtime(
                "no image runner is installed; no placeholder generation is available",
            )),
            ImageRunnerState::Initializing { message } => {
                Err(FrontendError::model_runtime(message.clone()))
            }
            ImageRunnerState::Failed { error } => Err(error.clone()),
            ImageRunnerState::Ready { capabilities } => {
                let descriptor = capabilities.descriptor(model).ok_or_else(|| {
                    FrontendError::model_runtime(format!(
                        "model '{model}' is not advertised by the installed runner"
                    ))
                })?;
                descriptor
                    .capabilities
                    .validate_request(model, request)
                    .map_err(FrontendError::from)
            }
        }
    }
}

/// Work accepted by a runner. The returned canonical cancellation token must
/// be the token checked by its `burn_image` runtime/model execution path.
#[derive(Clone, Debug)]
pub struct ImageRunnerJob {
    pub id: ImageJobId,
    pub model: ModelId,
    pub request: ImageRequest,
}

/// Poll result built entirely from `burn_image` runtime/output contracts.
#[derive(Clone, Debug)]
pub enum ImageRunnerEvent {
    Progress {
        id: ImageJobId,
        event: ProgressEvent,
    },
    Completed {
        id: ImageJobId,
        output: ImageOutput,
    },
    Failed {
        id: ImageJobId,
        error: RuntimeError,
    },
    Cancelled {
        id: ImageJobId,
    },
}

/// Asynchronous boundary implemented by a native worker or a browser-local
/// WebGPU task. `submit` must return the exact canonical cancellation token
/// observed by inference. `poll` must be non-blocking and bounded per call.
pub trait ImageRunner: Send + Sync + 'static {
    fn capabilities(&self) -> ImageRunnerCapabilities;

    fn submit(&mut self, job: ImageRunnerJob) -> Result<CancellationToken, RuntimeError>;

    fn cancel(&mut self, id: ImageJobId) -> Result<(), RuntimeError>;

    fn poll(&mut self, emit: &mut dyn FnMut(ImageRunnerEvent));
}

#[derive(Resource)]
struct InstalledImageRunner(Box<dyn ImageRunner>);

#[derive(Resource, Default)]
struct ActiveRunnerJobs(BTreeMap<ImageJobId, CancellationToken>);

/// Installs a concrete runner and connects it to the frontend messages.
///
/// This plugin accepts ownership exactly once; reusing the same plugin value
/// in multiple apps is a configuration error and panics during app setup.
pub struct ImageRunnerPlugin<R: ImageRunner> {
    runner: Mutex<Option<R>>,
}

impl<R: ImageRunner> ImageRunnerPlugin<R> {
    pub fn new(runner: R) -> Self {
        Self {
            runner: Mutex::new(Some(runner)),
        }
    }
}

impl<R: ImageRunner> Plugin for ImageRunnerPlugin<R> {
    fn build(&self, app: &mut App) {
        let runner = self
            .runner
            .lock()
            .expect("image runner plugin mutex poisoned")
            .take()
            .expect("an ImageRunnerPlugin can only be installed once");
        let capabilities = runner.capabilities();
        let status =
            ImageRunnerStatus::ready(capabilities).unwrap_or_else(|error| ImageRunnerStatus {
                state: ImageRunnerState::Failed { error },
            });

        app.insert_resource(status)
            .insert_resource(InstalledImageRunner(Box::new(runner)))
            .init_resource::<ActiveRunnerJobs>()
            .add_systems(
                Update,
                (submit_dispatched_jobs, cancel_runner_jobs, poll_runner)
                    .chain()
                    .in_set(ImageFrontendSet::Dispatch),
            );
    }
}

fn submit_dispatched_jobs(
    mut runner: ResMut<InstalledImageRunner>,
    mut active: ResMut<ActiveRunnerJobs>,
    mut dispatched: MessageReader<ImageJobDispatched>,
    mut failed: MessageWriter<FailImageJob>,
) {
    for dispatch in dispatched.read() {
        match runner.0.submit(ImageRunnerJob {
            id: dispatch.id,
            model: dispatch.model.clone(),
            request: dispatch.request.clone(),
        }) {
            Ok(token) => {
                active.0.insert(dispatch.id, token);
            }
            Err(error) => {
                failed.write(FailImageJob {
                    id: dispatch.id,
                    error: FrontendError::from(error),
                });
            }
        }
    }
}

fn cancel_runner_jobs(
    mut runner: ResMut<InstalledImageRunner>,
    mut active: ResMut<ActiveRunnerJobs>,
    mut cancellations: MessageReader<ImageJobCancellationRequested>,
) {
    for cancellation in cancellations.read() {
        if let Some(token) = active.0.remove(&cancellation.id) {
            token.cancel();
        }
        let _ = runner.0.cancel(cancellation.id);
    }
}

fn poll_runner(
    mut runner: ResMut<InstalledImageRunner>,
    mut active: ResMut<ActiveRunnerJobs>,
    mut progress: MessageWriter<ReportImageProgress>,
    mut completed: MessageWriter<CompleteImageJob>,
    mut failed: MessageWriter<FailImageJob>,
) {
    runner.0.poll(&mut |event| match event {
        ImageRunnerEvent::Progress { id, event } => {
            progress.write(ReportImageProgress { id, event });
        }
        ImageRunnerEvent::Completed { id, output } => {
            active.0.remove(&id);
            completed.write(CompleteImageJob { id, output });
        }
        ImageRunnerEvent::Failed { id, error } => {
            active.0.remove(&id);
            failed.write(FailImageJob {
                id,
                error: FrontendError::from(error),
            });
        }
        ImageRunnerEvent::Cancelled { id } => {
            active.0.remove(&id);
        }
    });
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::BTreeSet;

    use burn_image::{DimensionConstraints, ImageTaskKind, ModelCapabilities, NumericFormat};

    use super::*;

    pub(crate) fn test_capabilities(model: &str) -> ImageRunnerCapabilities {
        ImageRunnerCapabilities {
            execution: WgpuExecutionKind::NativeWgpu,
            models: vec![ModelDescriptor {
                id: ModelId::new(model).unwrap(),
                display_name: "Test".into(),
                revision: "test-revision".into(),
                capabilities: ModelCapabilities {
                    tasks: BTreeSet::from([ImageTaskKind::Generate, ImageTaskKind::Edit]),
                    supports_masks: true,
                    dimensions: DimensionConstraints {
                        min_width: 1,
                        max_width: 4096,
                        min_height: 1,
                        max_height: 4096,
                        width_multiple: 1,
                        height_multiple: 1,
                        max_pixels: Some(4096 * 4096),
                        allowed_dimensions: None,
                    },
                    min_steps: 1,
                    max_steps: 100,
                    max_batch_size: 4,
                    numeric_formats: BTreeSet::from([NumericFormat::F16]),
                },
            }],
            streams_progress: true,
            cooperative_cancellation: true,
            returns_host_images: true,
        }
    }

    #[test]
    fn missing_runner_never_claims_placeholder_generation_correctness() {
        let status = ImageRunnerStatus::default();
        let error = status
            .validate_request(
                &ModelId::new("test/model").unwrap(),
                &ImageRequest::Generate(burn_image::GenerateRequest {
                    prompt: burn_image::Prompt::new("test").unwrap(),
                    negative_prompt: None,
                    options: burn_image::GenerationOptions::default(),
                }),
            )
            .unwrap_err();
        assert!(error.message.contains("no placeholder generation"));
    }

    #[test]
    fn runner_capabilities_use_canonical_descriptors_correctness() {
        let capabilities = test_capabilities("test/model");
        assert!(capabilities.validate().is_ok());
        assert_eq!(
            capabilities.descriptor(&ModelId::new("test/model").unwrap()),
            capabilities.models.first()
        );
    }
}
