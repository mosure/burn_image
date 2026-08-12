use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use serde::{Deserialize, Serialize};

use crate::{ArtifactComponentId, ArtifactPath, ImageTaskKind, ModelId, RuntimeError};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RunId(pub u64);

/// Structured load and inference progress shared by CLI, browser, and UI adapters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum ProgressEvent {
    RunStarted {
        run_id: RunId,
        model: ModelId,
        task: ImageTaskKind,
    },
    ArtifactStarted {
        run_id: RunId,
        path: ArtifactPath,
        component: Option<ArtifactComponentId>,
        file_index: u32,
        file_count: u32,
        total_bytes: u64,
    },
    ArtifactProgress {
        run_id: RunId,
        path: ArtifactPath,
        loaded_bytes: u64,
        total_bytes: u64,
    },
    ArtifactVerified {
        run_id: RunId,
        path: ArtifactPath,
    },
    StageStarted {
        run_id: RunId,
        stage: String,
        total_steps: Option<u32>,
    },
    Step {
        run_id: RunId,
        stage: String,
        step: u32,
        total_steps: u32,
        elapsed_micros: u64,
    },
    StageCompleted {
        run_id: RunId,
        stage: String,
        elapsed_micros: u64,
    },
    Warning {
        run_id: RunId,
        message: String,
    },
    RunCompleted {
        run_id: RunId,
        elapsed_micros: u64,
    },
    RunFailed {
        run_id: RunId,
        message: String,
    },
    RunCancelled {
        run_id: RunId,
    },
}

/// Thread-safe observer usable by native workers. Wasm adapters may forward
/// events through a channel before invoking JavaScript.
pub trait ProgressObserver: Send + Sync {
    fn on_progress(&self, event: &ProgressEvent);
}

impl<F> ProgressObserver for F
where
    F: Fn(&ProgressEvent) + Send + Sync,
{
    fn on_progress(&self, event: &ProgressEvent) {
        self(event);
    }
}

#[derive(Default)]
pub struct NoopProgressObserver;

impl ProgressObserver for NoopProgressObserver {
    fn on_progress(&self, _event: &ProgressEvent) {}
}

/// Cooperative cancellation checked at model-defined safe boundaries.
#[derive(Clone, Debug, Default)]
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub fn reset(&self) {
        self.0.store(false, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }

    pub fn check(&self) -> Result<(), RuntimeError> {
        if self.is_cancelled() {
            Err(RuntimeError::Cancelled)
        } else {
            Ok(())
        }
    }
}

/// Per-run context passed to a model implementation.
#[derive(Clone)]
pub struct InferenceContext {
    run_id: RunId,
    observer: Arc<dyn ProgressObserver>,
    cancellation: CancellationToken,
}

impl InferenceContext {
    pub(crate) fn new(
        run_id: RunId,
        observer: Arc<dyn ProgressObserver>,
        cancellation: CancellationToken,
    ) -> Self {
        Self {
            run_id,
            observer,
            cancellation,
        }
    }

    pub fn run_id(&self) -> RunId {
        self.run_id
    }

    pub fn cancellation(&self) -> &CancellationToken {
        &self.cancellation
    }

    pub fn check_cancelled(&self) -> Result<(), RuntimeError> {
        self.cancellation.check()
    }

    pub fn emit(&self, event: ProgressEvent) {
        self.observer.on_progress(&event);
    }

    pub fn stage_started(&self, stage: impl Into<String>, total_steps: Option<u32>) {
        self.emit(ProgressEvent::StageStarted {
            run_id: self.run_id,
            stage: stage.into(),
            total_steps,
        });
    }

    pub fn step(&self, stage: impl Into<String>, step: u32, total_steps: u32, elapsed_micros: u64) {
        self.emit(ProgressEvent::Step {
            run_id: self.run_id,
            stage: stage.into(),
            step,
            total_steps,
            elapsed_micros,
        });
    }

    pub fn stage_completed(&self, stage: impl Into<String>, elapsed_micros: u64) {
        self.emit(ProgressEvent::StageCompleted {
            run_id: self.run_id,
            stage: stage.into(),
            elapsed_micros,
        });
    }
}
