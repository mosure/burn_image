//! First-class native CLI automation through the ordinary Bevy image-job path.

use std::{
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use bevy::window::PrimaryWindow;
use bevy::{app::AppExit, prelude::*};
use burn_boogu::BooguVariant;
use burn_image::{ImageEncoding, ImageRequest, ProgressEvent, Sha256Digest};

use crate::{
    CompleteImageJob, FailImageJob, ImageFrontendSet, ImageJobId, ImageJobPhase, ImageJobRejected,
    ImageJobs, ImageRunnerState, ImageRunnerStatus, NativeOutputQualificationRequestIdentity,
    ReportImageProgress, SubmitImageJob, boogu_model_id, encode_host_image,
};

/// One non-interactive request executed through the same runtime and validation path as the UI.
#[derive(Clone, Debug)]
pub struct NativeAutomatedRun {
    pub variant: BooguVariant,
    pub request: ImageRequest,
    pub request_identity: NativeOutputQualificationRequestIdentity,
    pub output_path: PathBuf,
    pub report_path: PathBuf,
    pub timeout: Duration,
    pub show_window: bool,
    /// Number of identical requests to execute against one retained runtime.
    pub repeat: u16,
}

/// Submit a native request when the model becomes ready, save its PNG/report, and exit.
pub struct NativeAutomatedRunPlugin {
    configuration: NativeAutomatedRun,
}

impl NativeAutomatedRunPlugin {
    pub fn new(configuration: NativeAutomatedRun) -> Self {
        Self { configuration }
    }
}

impl Plugin for NativeAutomatedRunPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(AutomatedRunHost {
            configuration: self.configuration.clone(),
            state: AutomatedRunState::WaitingForRuntime,
            progress: Vec::new(),
            completed_runs: Vec::new(),
            started: Instant::now(),
            request_started: None,
            runtime_ready_and_submit: None,
        })
        .add_systems(
            Update,
            drive_native_automated_run.in_set(ImageFrontendSet::Display),
        );
        if !self.configuration.show_window {
            app.add_systems(Startup, hide_automated_window);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AutomatedRunState {
    WaitingForRuntime,
    Submitted { id: ImageJobId, index: u16 },
    Finished,
}

#[derive(Resource)]
struct AutomatedRunHost {
    configuration: NativeAutomatedRun,
    state: AutomatedRunState,
    progress: Vec<ProgressEvent>,
    completed_runs: Vec<serde_json::Value>,
    started: Instant,
    request_started: Option<Instant>,
    runtime_ready_and_submit: Option<Duration>,
}

fn hide_automated_window(mut windows: Query<&mut Window, With<PrimaryWindow>>) {
    for mut window in &mut windows {
        window.visible = false;
    }
}

#[allow(clippy::too_many_arguments)]
fn drive_native_automated_run(
    mut host: ResMut<AutomatedRunHost>,
    runner: Res<ImageRunnerStatus>,
    mut jobs: ResMut<ImageJobs>,
    mut submit: MessageWriter<SubmitImageJob>,
    mut completions: MessageReader<CompleteImageJob>,
    mut failures: MessageReader<FailImageJob>,
    mut rejected: MessageReader<ImageJobRejected>,
    mut progress: MessageReader<ReportImageProgress>,
    mut exits: MessageWriter<AppExit>,
) {
    if host.state == AutomatedRunState::Finished {
        return;
    }
    if host.started.elapsed() >= host.configuration.timeout {
        let timeout_seconds = host.configuration.timeout.as_secs_f64();
        finish_failure(
            &mut host,
            format!("native automated run exceeded its {timeout_seconds:.3}-second timeout"),
            &mut exits,
        );
        return;
    }
    let active_submission = match host.state {
        AutomatedRunState::Submitted { id, index } => Some((id, index)),
        AutomatedRunState::WaitingForRuntime | AutomatedRunState::Finished => None,
    };
    let active_id = active_submission.map(|(id, _)| id);
    for update in progress.read() {
        if active_id == Some(update.id) {
            host.progress.push(update.event.clone());
        }
    }
    for failure in failures.read() {
        if active_id == Some(failure.id) {
            finish_failure(
                &mut host,
                format!("native inference failed: {}", failure.error),
                &mut exits,
            );
            return;
        }
    }
    for rejection in rejected.read() {
        if active_id == Some(rejection.id) {
            finish_failure(
                &mut host,
                format!("native request was rejected: {}", rejection.error),
                &mut exits,
            );
            return;
        }
    }
    for completion in completions.read() {
        if active_id != Some(completion.id) {
            continue;
        }
        if !jobs
            .get(completion.id)
            .is_some_and(|job| job.phase == ImageJobPhase::Completed)
        {
            finish_failure(
                &mut host,
                "output did not pass ordinary frontend completion checks".into(),
                &mut exits,
            );
            return;
        }
        if let Err(error) = validate_automated_output(&completion.output) {
            finish_failure(&mut host, error, &mut exits);
            return;
        }
        let (_, index) = active_submission.expect("completion belongs to active submission");
        let run = automated_run_record(&host, index, completion.id, &completion.output);
        if index + 1 < host.configuration.repeat {
            host.completed_runs.push(run);
            let id = jobs.reserve_id();
            submit.write(SubmitImageJob {
                id,
                model: boogu_model_id(host.configuration.variant),
                request: host.configuration.request.clone(),
            });
            host.request_started = Some(Instant::now());
            host.state = AutomatedRunState::Submitted {
                id,
                index: index + 1,
            };
        } else {
            match finish_output(&host, completion.id, &completion.output, run) {
                Ok(()) => {
                    host.state = AutomatedRunState::Finished;
                    exits.write(AppExit::Success);
                }
                Err(error) => finish_failure(&mut host, error, &mut exits),
            }
        }
        return;
    }

    match (&runner.state, host.state) {
        (ImageRunnerState::Failed { error }, AutomatedRunState::WaitingForRuntime) => {
            finish_failure(
                &mut host,
                format!("native runtime initialization failed: {error}"),
                &mut exits,
            );
        }
        (ImageRunnerState::Ready { capabilities }, AutomatedRunState::WaitingForRuntime) => {
            let model = boogu_model_id(host.configuration.variant);
            if capabilities.descriptor(&model).is_none() {
                finish_failure(
                    &mut host,
                    format!("native runtime became ready without selected model {model}"),
                    &mut exits,
                );
                return;
            }
            let id = jobs.reserve_id();
            submit.write(SubmitImageJob {
                id,
                model,
                request: host.configuration.request.clone(),
            });
            host.request_started = Some(Instant::now());
            host.runtime_ready_and_submit = Some(host.started.elapsed());
            host.state = AutomatedRunState::Submitted { id, index: 0 };
        }
        _ => {}
    }
}

fn finish_output(
    host: &AutomatedRunHost,
    job_id: ImageJobId,
    output: &burn_image::ImageOutput,
    final_run: serde_json::Value,
) -> Result<(), String> {
    validate_automated_output(output)?;
    let png = encode_host_image(&output.images[0].image, ImageEncoding::Png)
        .map_err(|error| format!("encode automated PNG: {error}"))?;
    write_atomic(&host.configuration.output_path, &png)?;
    let output_path = canonical_or_original(&host.configuration.output_path);
    let request_milliseconds = host
        .request_started
        .map(|started| started.elapsed().as_secs_f64() * 1_000.0);
    let startup_milliseconds = host
        .runtime_ready_and_submit
        .map(|elapsed| elapsed.as_secs_f64() * 1_000.0);
    let runs = host
        .completed_runs
        .iter()
        .cloned()
        .chain(std::iter::once(final_run))
        .collect::<Vec<_>>();
    let report = serde_json::json!({
        "schema_version": 1,
        "test": "burn_image_native_automated_run",
        "ok": true,
        "job_id": job_id.0,
        "request": &host.configuration.request_identity,
        "repeat": host.configuration.repeat,
        "runs": runs,
        "output": {
            "path": output_path,
            "bytes": png.len(),
            "sha256": Sha256Digest::calculate(&png).to_hex(),
            "encoding": "png",
            "seed": output.seed,
        },
        "timings": &output.timings,
        "provenance": &output.provenance,
        "progress_events": &host.progress,
        "host_timings": {
            "runtime_ready_and_submit_milliseconds": startup_milliseconds,
            "request_milliseconds": request_milliseconds,
            "total_milliseconds": host.started.elapsed().as_secs_f64() * 1_000.0,
        },
        "completed_unix_milliseconds": unix_milliseconds(),
    });
    write_json_atomic(&host.configuration.report_path, &report)?;
    println!("{}", host.configuration.report_path.display());
    Ok(())
}

fn validate_automated_output(output: &burn_image::ImageOutput) -> Result<(), String> {
    output
        .validate()
        .map_err(|error| format!("automated output validation failed: {error}"))?;
    if output.images.len() != 1 || output.images[0].index != 0 {
        return Err("automated output must contain exactly image index zero".into());
    }
    Ok(())
}

fn automated_run_record(
    host: &AutomatedRunHost,
    index: u16,
    job_id: ImageJobId,
    output: &burn_image::ImageOutput,
) -> serde_json::Value {
    serde_json::json!({
        "index": index,
        "residency": if index == 0 { "cold-kernels" } else { "warm-retained" },
        "job_id": job_id.0,
        "request_milliseconds": host.request_started.map(|started| {
            started.elapsed().as_secs_f64() * 1_000.0
        }),
        "timings": &output.timings,
        "seed": output.seed,
    })
}

fn finish_failure(host: &mut AutomatedRunHost, error: String, exits: &mut MessageWriter<AppExit>) {
    if host.state == AutomatedRunState::Finished {
        return;
    }
    let report = serde_json::json!({
        "schema_version": 1,
        "test": "burn_image_native_automated_run",
        "ok": false,
        "failures": [&error],
        "request": &host.configuration.request_identity,
        "repeat": host.configuration.repeat,
        "completed_runs": &host.completed_runs,
        "progress_events": &host.progress,
        "host_timings": {
            "runtime_ready_and_submit_milliseconds": host.runtime_ready_and_submit.map(|elapsed| {
                elapsed.as_secs_f64() * 1_000.0
            }),
            "request_milliseconds": host.request_started.map(|started| {
                started.elapsed().as_secs_f64() * 1_000.0
            }),
            "total_milliseconds": host.started.elapsed().as_secs_f64() * 1_000.0,
        },
        "completed_unix_milliseconds": unix_milliseconds(),
    });
    if let Err(write_error) = write_json_atomic(&host.configuration.report_path, &report) {
        eprintln!("native automated report write failed: {write_error}");
    }
    eprintln!("native automated run failed: {error}");
    host.state = AutomatedRunState::Finished;
    exits.write(AppExit::error());
}

fn write_json_atomic(path: &Path, value: &serde_json::Value) -> Result<(), String> {
    let mut bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("serialize automated report: {error}"))?;
    bytes.push(b'\n');
    write_atomic(path, &bytes)
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .map_err(|error| format!("create output directory {}: {error}", parent.display()))?;
    let file_name = path
        .file_name()
        .ok_or_else(|| format!("output path has no file name: {}", path.display()))?;
    let temporary = parent.join(format!(
        ".{}.{}.tmp",
        file_name.to_string_lossy(),
        std::process::id()
    ));
    fs::write(&temporary, bytes)
        .map_err(|error| format!("write temporary output {}: {error}", temporary.display()))?;
    fs::rename(&temporary, path)
        .map_err(|error| format!("commit output {}: {error}", path.display()))
}

fn canonical_or_original(path: &Path) -> PathBuf {
    fs::canonicalize(path).unwrap_or_else(|_| path.to_owned())
}

fn unix_milliseconds() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automated_report_path_supports_relative_outputs_correctness() {
        assert_eq!(
            Path::new("output.png").with_extension("report.json"),
            PathBuf::from("output.report.json")
        );
    }
}
