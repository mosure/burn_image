use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use serde::{Deserialize, Serialize};

use crate::{ArtifactComponentId, ArtifactPath, ImageTaskKind, ModelId, RuntimeError};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RunId(pub u64);

/// Request-local work performed from an already verified browser transport cache.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArtifactRequestTransferActivity {
    pub phase: String,
    pub current_path: Option<ArtifactPath>,
    pub component: Option<ArtifactComponentId>,
    pub logical_objects_completed: u32,
    pub bounded_ranges_processed: u64,
    pub processed_bytes: u64,
}

/// Aggregate progress through one immutable artifact transport closure.
///
/// Logical objects are the semantic model files consumed by a runtime. Physical parts and
/// bounded ranges describe their browser/CDN representation. The counters are monotonic across
/// semantic-stage boundaries, so a UI never mistakes a stage-local file count for the complete
/// transfer denominator.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArtifactTransferProgress {
    pub phase: String,
    pub component: Option<ArtifactComponentId>,
    pub logical_objects_completed: u32,
    pub logical_objects_total: u32,
    pub physical_parts_completed: u32,
    pub physical_parts_total: u32,
    pub bounded_ranges_completed: u64,
    pub bounded_ranges_total: u64,
    pub loaded_bytes: u64,
    pub total_bytes: u64,
    /// Smoothed aggregate authenticated-byte throughput. It remains absent until enough samples
    /// have accumulated to avoid a misleading first-range estimate.
    pub bytes_per_second: Option<u64>,
    pub eta_seconds: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_activity: Option<ArtifactRequestTransferActivity>,
}

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
        #[serde(default, skip_serializing_if = "Option::is_none")]
        transfer: Option<ArtifactTransferProgress>,
    },
    ArtifactProgress {
        run_id: RunId,
        path: ArtifactPath,
        loaded_bytes: u64,
        total_bytes: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        transfer: Option<ArtifactTransferProgress>,
    },
    ArtifactVerified {
        run_id: RunId,
        path: ArtifactPath,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        transfer: Option<ArtifactTransferProgress>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_transfer_telemetry_is_optional_for_native_progress_correctness() {
        let native = serde_json::json!({
            "event": "artifact_progress",
            "run_id": 7,
            "path": "objects/current.bpk",
            "loaded_bytes": 4,
            "total_bytes": 8
        });
        let decoded: ProgressEvent = serde_json::from_value(native).unwrap();
        assert!(matches!(
            decoded,
            ProgressEvent::ArtifactProgress { transfer: None, .. }
        ));

        let event = ProgressEvent::ArtifactProgress {
            run_id: RunId(7),
            path: ArtifactPath::new("objects/current.bpk").unwrap(),
            loaded_bytes: 4,
            total_bytes: 8,
            transfer: Some(ArtifactTransferProgress {
                phase: "Model setup".into(),
                component: Some(ArtifactComponentId::new("qwen").unwrap()),
                logical_objects_completed: 1,
                logical_objects_total: 2,
                physical_parts_completed: 2,
                physical_parts_total: 4,
                bounded_ranges_completed: 8,
                bounded_ranges_total: 16,
                loaded_bytes: 32,
                total_bytes: 64,
                bytes_per_second: None,
                eta_seconds: None,
                request_activity: None,
            }),
        };
        let encoded = serde_json::to_value(event).unwrap();
        assert_eq!(encoded["event"], "artifact_progress");
        assert_eq!(encoded["transfer"]["physical_parts_total"], 4);
        assert!(encoded["transfer"]["bytes_per_second"].is_null());
    }
}
