//! Bevy 0.19 frontend contracts for `burn_image` runtimes.
//!
//! This crate owns the ECS boundary, shared WGPU device initialization, host
//! image display and I/O, and bounded artifact transport. Concrete models live
//! in model crates and consume [`ImageJobDispatched`] messages.

#![forbid(unsafe_code)]

pub mod artifact_stream;
pub mod backend;
pub mod display;
pub mod editor;
pub mod error;
pub mod io;
pub mod jobs;
pub mod runner;

#[cfg(feature = "boogu")]
pub mod boogu;
#[cfg(all(feature = "boogu-web", target_arch = "wasm32"))]
pub mod browser_boogu;
#[cfg(all(feature = "boogu-native", not(target_arch = "wasm32")))]
pub mod native_artifact_cache;
#[cfg(all(feature = "boogu-native", not(target_arch = "wasm32")))]
pub mod native_boogu;

#[cfg(feature = "app")]
pub mod app;
#[cfg(feature = "app")]
pub mod controls;

pub use artifact_stream::*;
pub use backend::*;
pub use display::*;
pub use editor::*;
pub use error::*;
pub use io::*;
pub use jobs::*;
pub use runner::*;

#[cfg(feature = "boogu")]
pub use boogu::*;
#[cfg(all(feature = "boogu-web", target_arch = "wasm32"))]
pub use browser_boogu::*;
#[cfg(all(feature = "boogu-native", not(target_arch = "wasm32")))]
pub use native_artifact_cache::*;
#[cfg(all(feature = "boogu-native", not(target_arch = "wasm32")))]
pub use native_boogu::*;

use bevy::prelude::*;

/// Human-readable application name used by native and browser shells.
pub const APP_NAME: &str = "burn image";

/// Installs the model-neutral frontend and, by default, the shared WGPU bridge.
///
/// The bridge initializes Burn from Bevy's exact adapter, device, instance,
/// and queue. Setting [`Self::install_shared_wgpu_bridge`] to `false` is useful
/// for contract-only tests, but leaves [`BackendStatus`] failed until an
/// embedding application explicitly supplies a ready status.
#[derive(Clone, Debug)]
pub struct BurnImageFrontendPlugin {
    pub install_shared_wgpu_bridge: bool,
}

impl Default for BurnImageFrontendPlugin {
    fn default() -> Self {
        Self {
            install_shared_wgpu_bridge: true,
        }
    }
}

impl Plugin for BurnImageFrontendPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            ImageJobPlugin,
            ImageEditorPlugin,
            ImageIoPlugin,
            ImageDisplayPlugin,
        ));

        #[cfg(feature = "gpu-interop")]
        if self.install_shared_wgpu_bridge {
            app.add_plugins(
                bevy_burn::BevyBurnBridgePlugin::<backend::SharedWgpuBackend>::default(),
            );
        }

        app.add_plugins(BackendStatusPlugin {
            bridge_expected: self.install_shared_wgpu_bridge,
        });
    }
}

/// Starts the browser shell when this library is built as a Wasm `cdylib`.
// A Boogu-enabled browser host uses the concrete query-configured factory below.
#[cfg(all(feature = "app", target_arch = "wasm32", not(feature = "boogu")))]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start_web() {
    console_error_panic_hook::set_once();
    app::build_app().run();
}

/// Start the concrete browser Boogu runtime.
///
/// Query parameters select `variant`, `profile`, and the absolute remote `artifacts` base URL.
/// When `artifacts` is omitted, the app uses the exact manifest bundle id beneath
/// `https://aberration.technology/model/`. The default is Turbo with
/// `f16-qwen-vision-f32`. `headless=bootstrap` selects an opt-in no-surface F32 compute diagnostic;
/// `headless=f16-probe` runs the same final-norm probe while requiring and preserving WebGPU
/// `shader-f16`.
#[cfg(all(feature = "boogu-web", target_arch = "wasm32"))]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start_boogu_web() -> Result<(), wasm_bindgen::JsValue> {
    use burn_boogu::{BooguVariant, artifacts::BooguStorageProfile};
    use burn_image::{ArtifactSource, RemoteBaseUrl};
    use wasm_bindgen::JsValue;

    console_error_panic_hook::set_once();
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("Window is unavailable"))?;
    let search = window.location().search()?;
    let params = web_sys::UrlSearchParams::new_with_str(&search)?;
    let headless = params.get("headless");
    if !matches!(
        headless.as_deref(),
        None | Some("bootstrap") | Some("f16-probe") | Some("infer")
    ) {
        return Err(JsValue::from_str(
            "unsupported headless mode; use headless=bootstrap, headless=f16-probe, or headless=infer",
        ));
    }
    let variant = match params.get("variant").as_deref().unwrap_or("turbo") {
        "turbo" => BooguVariant::Image01Turbo,
        "edit" | "edit-turbo" => BooguVariant::Image01EditTurbo,
        "1k5" | "edit-turbo-1k5" => {
            return Err(JsValue::from_str(
                "Boogu Edit-Turbo 1.5K is native-WGPU only; browser WebGPU numerical and performance parity have not been validated",
            ));
        }
        value => {
            return Err(JsValue::from_str(&format!(
                "unsupported Boogu variant {value:?}; browser supports turbo or edit-turbo"
            )));
        }
    };
    let profile = match params
        .get("profile")
        .as_deref()
        .unwrap_or("f16-qwen-vision-f32")
    {
        "f16" => BooguStorageProfile::F16,
        "f16-qwen-vision-f32" => BooguStorageProfile::F16QwenVisionF32,
        "q8s-block32-f32" => BooguStorageProfile::Q8sBlock32F32,
        "q8s-block32-f32-qwen-vision-f32" => BooguStorageProfile::Q8sBlock32F32QwenVisionF32,
        value => {
            return Err(JsValue::from_str(&format!(
                "unsupported Boogu profile {value:?}"
            )));
        }
    };
    let artifacts = match params.get("artifacts") {
        Some(artifacts) => artifacts,
        None => {
            let published = burn_boogu::artifacts::canonical_published_bundle(variant, profile)
                .ok_or_else(|| {
                    JsValue::from_str(
                        "the selected variant/profile has no canonical published artifact; pass an explicit artifacts= custom URL for diagnostics",
                    )
                })?;
            debug_assert_eq!(
                published.bundle_id,
                crate::boogu::boogu_bundle_id(variant, profile)
            );
            crate::boogu::boogu_cdn_base_url(variant, profile)
        }
    };
    let base_url =
        RemoteBaseUrl::new(artifacts).map_err(|error| JsValue::from_str(&error.to_string()))?;
    let mut settings = BooguAdapterSettings::verified_default(ArtifactSource::Remote { base_url });
    settings.storage_profile = profile;
    if matches!(headless.as_deref(), Some("bootstrap" | "f16-probe")) {
        let preserve_f16 = headless.as_deref() == Some("f16-probe");
        let status = window
            .document()
            .and_then(|document| document.get_element_by_id("status"));
        wasm_bindgen_futures::spawn_local(async move {
            let result = if preserve_f16 {
                BrowserBooguFactory::bootstrap_no_surface_preserve_f16(variant, settings).await
            } else {
                BrowserBooguFactory::bootstrap_no_surface(variant, settings).await
            };
            let (message, failed) = match result {
                Ok(report) => match serde_json::to_string(&report) {
                    Ok(json) => (format!("BURN_IMAGE_HEADLESS_BOOTSTRAP_OK {json}"), false),
                    Err(error) => (
                        format!("BURN_IMAGE_HEADLESS_BOOTSTRAP_FAILED report JSON: {error}"),
                        true,
                    ),
                },
                Err(error) => (
                    format!("BURN_IMAGE_HEADLESS_BOOTSTRAP_FAILED {error}"),
                    true,
                ),
            };
            if let Some(status) = status {
                status.set_text_content(Some(&message));
            }
            if failed {
                web_sys::console::error_1(&JsValue::from_str(&message));
            } else {
                web_sys::console::info_1(&JsValue::from_str(&message));
            }
        });
        return Ok(());
    }
    if headless.as_deref() == Some("infer") {
        let request = boogu::prepare_headless_generate_request(
            variant,
            params.get("prompt"),
            params.get("seed"),
            params.get("width"),
            params.get("height"),
        )
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
        let status = window
            .document()
            .and_then(|document| document.get_element_by_id("status"));
        wasm_bindgen_futures::spawn_local(async move {
            let result = BrowserBooguFactory::infer_no_surface(variant, settings, request).await;
            let (message, failed) = match result {
                Ok(result) => {
                    let attached =
                        attach_headless_inference_png(&result.png, &result.report.png_file_name);
                    match (attached, serde_json::to_string(&result.report)) {
                        (Ok(()), Ok(json)) => {
                            (format!("BURN_IMAGE_HEADLESS_INFER_OK {json}"), false)
                        }
                        (Err(error), _) => (
                            format!("BURN_IMAGE_HEADLESS_INFER_FAILED output attach: {error:?}"),
                            true,
                        ),
                        (_, Err(error)) => (
                            format!("BURN_IMAGE_HEADLESS_INFER_FAILED report JSON: {error}"),
                            true,
                        ),
                    }
                }
                Err(error) => (format!("BURN_IMAGE_HEADLESS_INFER_FAILED {error}"), true),
            };
            if let Some(status) = status {
                status.set_text_content(Some(&message));
            }
            if failed {
                web_sys::console::error_1(&JsValue::from_str(&message));
            } else {
                web_sys::console::info_1(&JsValue::from_str(&message));
            }
        });
        return Ok(());
    }
    app::build_boogu_app(settings, BrowserBooguFactory::new(variant)).run();
    Ok(())
}
