use std::collections::BTreeMap;

use bevy::prelude::*;
use burn_image::{ImageOutput, ImageRequest, ModelId, ProgressEvent};
use serde::{Deserialize, Serialize};

use crate::{BackendStatus, FrontendError, ImageRunnerStatus};

const MAX_RETAINED_TERMINAL_JOBS: usize = 8;

/// Frontend-owned identifier, independent of model-runtime `RunId` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ImageJobId(pub u64);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub enum ImageJobPhase {
    Queued,
    Running,
    Completed,
    Failed { error: FrontendError },
    Cancelled,
}

impl ImageJobPhase {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed { .. } | Self::Cancelled
        )
    }
}

/// Lightweight job record. Completed image bytes flow through messages into
/// Bevy assets rather than being duplicated here.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImageJobRecord {
    pub id: ImageJobId,
    pub model: ModelId,
    /// Present only while the job can still execute. Terminal records release
    /// request-owned image payloads while retaining progress and provenance context.
    pub request: Option<ImageRequest>,
    pub phase: ImageJobPhase,
    pub last_progress: Option<ProgressEvent>,
}

#[derive(Resource, Default)]
pub struct ImageJobs {
    next_id: u64,
    records: BTreeMap<ImageJobId, ImageJobRecord>,
}

impl ImageJobs {
    pub fn reserve_id(&mut self) -> ImageJobId {
        self.next_id = self.next_id.wrapping_add(1).max(1);
        ImageJobId(self.next_id)
    }

    pub fn get(&self, id: ImageJobId) -> Option<&ImageJobRecord> {
        self.records.get(&id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &ImageJobRecord> {
        self.records.values()
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

/// UI or host request to validate and route one generation/edit operation.
#[derive(Message, Clone, Debug)]
pub struct SubmitImageJob {
    pub id: ImageJobId,
    pub model: ModelId,
    pub request: ImageRequest,
}

/// Validated request consumed by a concrete model integration plugin.
#[derive(Message, Clone, Debug)]
pub struct ImageJobDispatched {
    pub id: ImageJobId,
    pub model: ModelId,
    pub request: ImageRequest,
}

#[derive(Message, Clone, Copy, Debug)]
pub struct CancelImageJob {
    pub id: ImageJobId,
}

/// Cancellation signal consumed by a model integration plugin.
#[derive(Message, Clone, Copy, Debug)]
pub struct ImageJobCancellationRequested {
    pub id: ImageJobId,
}

#[derive(Message, Clone, Debug)]
pub struct ReportImageProgress {
    pub id: ImageJobId,
    pub event: ProgressEvent,
}

#[derive(Message, Clone, Debug)]
pub struct CompleteImageJob {
    pub id: ImageJobId,
    pub output: ImageOutput,
}

#[derive(Message, Clone, Debug)]
pub struct FailImageJob {
    pub id: ImageJobId,
    pub error: FrontendError,
}

/// Observable rejection for duplicate, invalid, or backend-unavailable jobs.
#[derive(Message, Clone, Debug)]
pub struct ImageJobRejected {
    pub id: ImageJobId,
    pub error: FrontendError,
}

/// Scheduling sets allow model crates to dispatch between input and feedback.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImageFrontendSet {
    Input,
    Dispatch,
    Feedback,
    Display,
}

pub struct ImageJobPlugin;

impl Plugin for ImageJobPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ImageJobs>()
            .init_resource::<ImageRunnerStatus>()
            .add_message::<SubmitImageJob>()
            .add_message::<ImageJobDispatched>()
            .add_message::<CancelImageJob>()
            .add_message::<ImageJobCancellationRequested>()
            .add_message::<ReportImageProgress>()
            .add_message::<CompleteImageJob>()
            .add_message::<FailImageJob>()
            .add_message::<ImageJobRejected>()
            .configure_sets(
                Update,
                (
                    ImageFrontendSet::Input,
                    ImageFrontendSet::Dispatch,
                    ImageFrontendSet::Feedback,
                    ImageFrontendSet::Display,
                )
                    .chain(),
            )
            .add_systems(
                Update,
                (accept_submissions, accept_cancellations)
                    .chain()
                    .in_set(ImageFrontendSet::Input),
            )
            .add_systems(
                Update,
                (apply_progress, apply_completions, apply_failures)
                    .chain()
                    .in_set(ImageFrontendSet::Feedback),
            );
    }
}

fn accept_submissions(
    backend: Res<BackendStatus>,
    runner: Res<ImageRunnerStatus>,
    mut jobs: ResMut<ImageJobs>,
    mut submitted: MessageReader<SubmitImageJob>,
    mut dispatched: MessageWriter<ImageJobDispatched>,
    mut rejected: MessageWriter<ImageJobRejected>,
) {
    for submission in submitted.read() {
        if jobs.records.contains_key(&submission.id) {
            rejected.write(ImageJobRejected {
                id: submission.id,
                error: FrontendError::invalid_request(format!(
                    "image job {} already exists",
                    submission.id.0
                )),
            });
            continue;
        }
        prune_terminal_jobs(
            &mut jobs.records,
            MAX_RETAINED_TERMINAL_JOBS.saturating_sub(1),
        );

        let validation = submission.request.validate().map_err(FrontendError::from);
        let availability = backend
            .unavailable_message()
            .map(FrontendError::backend)
            .map_or(Ok(()), Err);
        let runner_validation = runner.validate_request(&submission.model, &submission.request);
        let error = validation
            .err()
            .or_else(|| availability.err())
            .or_else(|| runner_validation.err());

        let phase = error
            .clone()
            .map_or(ImageJobPhase::Queued, |error| ImageJobPhase::Failed {
                error,
            });
        let retained_request = error.is_none().then(|| submission.request.clone());
        jobs.records.insert(
            submission.id,
            ImageJobRecord {
                id: submission.id,
                model: submission.model.clone(),
                request: retained_request,
                phase,
                last_progress: None,
            },
        );

        if let Some(error) = error {
            rejected.write(ImageJobRejected {
                id: submission.id,
                error,
            });
        } else {
            dispatched.write(ImageJobDispatched {
                id: submission.id,
                model: submission.model.clone(),
                request: submission.request.clone(),
            });
        }
    }
}

fn prune_terminal_jobs(
    records: &mut BTreeMap<ImageJobId, ImageJobRecord>,
    maximum_terminal_jobs: usize,
) {
    let remove_count = records
        .values()
        .filter(|record| record.phase.is_terminal())
        .count()
        .saturating_sub(maximum_terminal_jobs);
    if remove_count == 0 {
        return;
    }
    let stale = records
        .iter()
        .filter(|(_, record)| record.phase.is_terminal())
        .map(|(id, _)| *id)
        .take(remove_count)
        .collect::<Vec<_>>();
    for id in stale {
        records.remove(&id);
    }
}

fn accept_cancellations(
    mut jobs: ResMut<ImageJobs>,
    mut requested: MessageReader<CancelImageJob>,
    mut routed: MessageWriter<ImageJobCancellationRequested>,
) {
    for cancellation in requested.read() {
        let Some(record) = jobs.records.get_mut(&cancellation.id) else {
            continue;
        };
        if !record.phase.is_terminal() {
            record.phase = ImageJobPhase::Cancelled;
            record.request = None;
            routed.write(ImageJobCancellationRequested {
                id: cancellation.id,
            });
        }
    }
}

fn apply_progress(mut jobs: ResMut<ImageJobs>, mut progress: MessageReader<ReportImageProgress>) {
    for update in progress.read() {
        let Some(record) = jobs.records.get_mut(&update.id) else {
            continue;
        };
        if record.phase.is_terminal() {
            continue;
        }
        record.phase = ImageJobPhase::Running;
        record.last_progress = Some(update.event.clone());
    }
}

fn apply_completions(mut jobs: ResMut<ImageJobs>, mut completed: MessageReader<CompleteImageJob>) {
    for completion in completed.read() {
        let Some(record) = jobs.records.get_mut(&completion.id) else {
            continue;
        };
        if record.phase.is_terminal() {
            continue;
        }
        let validation = completion.output.validate().map_err(FrontendError::from);
        let provenance_matches = completion.output.provenance.model == record.model;
        record.phase = match (validation, provenance_matches) {
            (Err(error), _) => ImageJobPhase::Failed { error },
            (Ok(()), false) => ImageJobPhase::Failed {
                error: FrontendError::invalid_request(format!(
                    "output model '{}' does not match selected model '{}'",
                    completion.output.provenance.model, record.model
                )),
            },
            (Ok(()), true) => ImageJobPhase::Completed,
        };
        record.request = None;
    }
}

fn apply_failures(mut jobs: ResMut<ImageJobs>, mut failed: MessageReader<FailImageJob>) {
    for failure in failed.read() {
        let Some(record) = jobs.records.get_mut(&failure.id) else {
            continue;
        };
        if !record.phase.is_terminal() {
            record.phase = ImageJobPhase::Failed {
                error: failure.error.clone(),
            };
            record.request = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use bevy::prelude::*;
    use burn_image::{
        GenerateRequest, GenerationOptions, ImageRequest, ImageTaskKind, ModelId, ProgressEvent,
        Prompt, RunId,
    };

    use crate::{BackendDeviceInfo, BackendStatus, ImageRunnerStatus};

    use super::{
        CancelImageJob, ImageJobId, ImageJobPhase, ImageJobPlugin, ImageJobRecord, ImageJobs,
        ReportImageProgress, SubmitImageJob,
    };

    fn request() -> ImageRequest {
        ImageRequest::Generate(GenerateRequest {
            prompt: Prompt::new("a lighthouse").unwrap(),
            negative_prompt: None,
            options: GenerationOptions::default(),
        })
    }

    fn ready_backend() -> BackendStatus {
        BackendStatus::ready(BackendDeviceInfo {
            adapter_name: "test".into(),
            backend: "test".into(),
            device_type: "test".into(),
            driver: "test".into(),
            max_storage_buffer_binding_size: 128 * 1024 * 1024,
            max_buffer_size: 256 * 1024 * 1024,
            shared_adapter_device_queue: true,
        })
    }

    fn ready_runner() -> ImageRunnerStatus {
        ImageRunnerStatus::ready(crate::runner::tests::test_capabilities("test/model")).unwrap()
    }

    #[test]
    fn ready_gpu_routes_valid_request_correctness() {
        let mut app = App::new();
        app.insert_resource(ready_backend())
            .insert_resource(ready_runner())
            .add_plugins(ImageJobPlugin);
        app.world_mut()
            .resource_mut::<Messages<SubmitImageJob>>()
            .write(SubmitImageJob {
                id: ImageJobId(7),
                model: ModelId::new("test/model").unwrap(),
                request: request(),
            });
        app.update();
        let jobs = app.world().resource::<ImageJobs>();
        assert!(matches!(
            jobs.get(ImageJobId(7)).unwrap().phase,
            ImageJobPhase::Queued
        ));
    }

    #[test]
    fn unavailable_gpu_rejects_without_cpu_fallback_correctness() {
        let mut app = App::new();
        app.init_resource::<BackendStatus>()
            .insert_resource(ready_runner())
            .add_plugins(ImageJobPlugin);
        app.world_mut()
            .resource_mut::<Messages<SubmitImageJob>>()
            .write(SubmitImageJob {
                id: ImageJobId(8),
                model: ModelId::new("test/model").unwrap(),
                request: request(),
            });
        app.update();
        let jobs = app.world().resource::<ImageJobs>();
        assert!(matches!(
            jobs.get(ImageJobId(8)).unwrap().phase,
            ImageJobPhase::Failed { .. }
        ));
        assert!(jobs.get(ImageJobId(8)).unwrap().request.is_none());
    }

    #[test]
    fn progress_and_cancellation_transitions_are_explicit_correctness() {
        let mut app = App::new();
        app.insert_resource(ready_backend())
            .insert_resource(ready_runner())
            .add_plugins(ImageJobPlugin);
        let model = ModelId::new("test/model").unwrap();
        app.world_mut()
            .resource_mut::<Messages<SubmitImageJob>>()
            .write(SubmitImageJob {
                id: ImageJobId(9),
                model: model.clone(),
                request: request(),
            });
        app.update();

        app.world_mut()
            .resource_mut::<Messages<ReportImageProgress>>()
            .write(ReportImageProgress {
                id: ImageJobId(9),
                event: ProgressEvent::RunStarted {
                    run_id: RunId(1),
                    model,
                    task: ImageTaskKind::Generate,
                },
            });
        app.update();
        assert!(matches!(
            app.world()
                .resource::<ImageJobs>()
                .get(ImageJobId(9))
                .unwrap()
                .phase,
            ImageJobPhase::Running
        ));

        app.world_mut()
            .resource_mut::<Messages<CancelImageJob>>()
            .write(CancelImageJob { id: ImageJobId(9) });
        app.update();
        assert!(matches!(
            app.world()
                .resource::<ImageJobs>()
                .get(ImageJobId(9))
                .unwrap()
                .phase,
            ImageJobPhase::Cancelled
        ));
        assert!(
            app.world()
                .resource::<ImageJobs>()
                .get(ImageJobId(9))
                .unwrap()
                .request
                .is_none()
        );
    }

    #[test]
    fn terminal_job_history_is_bounded_without_pruning_active_jobs_correctness() {
        let model = ModelId::new("test/model").unwrap();
        let mut records = BTreeMap::new();
        for raw_id in 1..=12 {
            let id = ImageJobId(raw_id);
            records.insert(
                id,
                super::ImageJobRecord {
                    id,
                    model: model.clone(),
                    request: (raw_id == 6).then_some(request()),
                    phase: if raw_id == 6 {
                        ImageJobPhase::Running
                    } else {
                        ImageJobPhase::Completed
                    },
                    last_progress: None,
                },
            );
        }

        super::prune_terminal_jobs(&mut records, 3);
        assert_eq!(
            records
                .values()
                .filter(|record| record.phase.is_terminal())
                .count(),
            3
        );
        assert!(records.contains_key(&ImageJobId(6)));
        assert_eq!(records.len(), 4);
    }

    #[test]
    fn job_record_serde_releases_only_terminal_request_payloads_correctness() {
        let model = ModelId::new("test/model").unwrap();
        let active = ImageJobRecord {
            id: ImageJobId(1),
            model: model.clone(),
            request: Some(request()),
            phase: ImageJobPhase::Running,
            last_progress: None,
        };
        let active_json = serde_json::to_value(&active).unwrap();
        assert_eq!(active_json["request"]["task"], "generate");
        let active_roundtrip: ImageJobRecord = serde_json::from_value(active_json).unwrap();
        assert!(active_roundtrip.request.is_some());

        let terminal = ImageJobRecord {
            id: ImageJobId(2),
            model,
            request: None,
            phase: ImageJobPhase::Completed,
            last_progress: None,
        };
        let terminal_json = serde_json::to_value(&terminal).unwrap();
        assert!(terminal_json["request"].is_null());
        let terminal_roundtrip: ImageJobRecord = serde_json::from_value(terminal_json).unwrap();
        assert!(terminal_roundtrip.request.is_none());
    }
}
