//! Thin public surface for the browser-specific Bevy adapter.
//!
//! Model/profile selection and execution policy live in `burn_boogu`. This module retains only
//! browser transport, shared-device integration, progress/events, and the runtime bridge owned by
//! the Bevy frontend.

mod runtime;

pub use runtime::*;
pub(crate) use runtime::{
    report_browser_runtime_failure, report_browser_runtime_preparing,
    report_browser_surface_inference_gate_failure, report_browser_surface_inference_resumed,
    report_browser_surface_inference_suspended,
};
