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
pub mod file_dialog;
pub mod io;
pub mod jobs;
pub mod runner;

#[cfg(feature = "boogu")]
pub mod boogu;
#[cfg(all(feature = "boogu-web", target_arch = "wasm32"))]
pub mod browser_boogu;
#[cfg(all(feature = "boogu-web", any(test, target_arch = "wasm32")))]
mod browser_parity_fixture;
#[cfg(all(feature = "boogu-native", not(target_arch = "wasm32")))]
pub mod native_artifact_cache;
#[cfg(all(feature = "boogu-native", not(target_arch = "wasm32")))]
pub mod native_boogu;

#[cfg(feature = "app")]
pub mod app;
#[cfg(feature = "app")]
pub mod controls;
#[cfg(feature = "app")]
pub mod viewer;

pub use artifact_stream::*;
pub use backend::*;
pub use display::*;
pub use editor::*;
pub use error::*;
pub use file_dialog::*;
pub use io::*;
pub use jobs::*;
pub use runner::*;
#[cfg(feature = "app")]
pub use viewer::*;

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

/// Preferred user-facing selector for the parity-qualified mixed-F16 storage profile.
#[cfg(feature = "boogu")]
pub const BOOGU_PRODUCTION_PROFILE_SELECTOR: &str = "production";

/// Resolve a user-facing profile selector without changing the sealed manifest profile identity.
///
/// `production` is the concise public selector. The precise legacy spelling remains accepted so
/// existing browser URLs and automation continue to select the same mixed-F16 storage contract.
#[cfg(feature = "boogu")]
pub fn parse_boogu_storage_profile(
    value: &str,
) -> Option<burn_boogu::artifacts::BooguStorageProfile> {
    use burn_boogu::artifacts::BooguStorageProfile;

    match value {
        "production" | "f16-qwen-vision-f32" => Some(BooguStorageProfile::F16QwenVisionF32),
        "f16" => Some(BooguStorageProfile::F16),
        "q8s-block32-f32" => Some(BooguStorageProfile::Q8sBlock32F32),
        "q8s-block32-f32-qwen-vision-f32" => Some(BooguStorageProfile::Q8sBlock32F32QwenVisionF32),
        _ => None,
    }
}

#[cfg(any(
    all(test, feature = "boogu"),
    all(feature = "boogu-web", target_arch = "wasm32")
))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BrowserHeadlessMode {
    Bootstrap,
    F16Probe,
    Infer,
    Parity,
    VaeReference,
}

#[cfg(any(
    all(test, feature = "boogu"),
    all(feature = "boogu-web", target_arch = "wasm32")
))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum BrowserResidencySelector {
    #[default]
    Resident,
    LayerStreamedDiagnostic,
}

#[cfg(any(
    all(test, feature = "boogu"),
    all(feature = "boogu-web", target_arch = "wasm32")
))]
fn parse_browser_residency(value: Option<&str>) -> Result<BrowserResidencySelector, String> {
    match value {
        None | Some("resident") | Some("high-vram-resident-dense-f32") => {
            Ok(BrowserResidencySelector::Resident)
        }
        Some("layer-streamed-diagnostic") => {
            Ok(BrowserResidencySelector::LayerStreamedDiagnostic)
        }
        Some(_) => Err(
            "unsupported browser residency; use residency=resident or the explicit residency=layer-streamed-diagnostic"
                .into(),
        ),
    }
}

#[cfg(any(
    all(test, feature = "boogu"),
    all(feature = "boogu-web", target_arch = "wasm32")
))]
fn parse_browser_headless_mode(value: Option<&str>) -> Result<Option<BrowserHeadlessMode>, String> {
    match value {
        None => Ok(None),
        Some("bootstrap") => Ok(Some(BrowserHeadlessMode::Bootstrap)),
        Some("f16-probe") => Ok(Some(BrowserHeadlessMode::F16Probe)),
        Some("infer") => Ok(Some(BrowserHeadlessMode::Infer)),
        Some("parity") => Ok(Some(BrowserHeadlessMode::Parity)),
        Some("vae-reference") => Ok(Some(BrowserHeadlessMode::VaeReference)),
        Some(_) => Err(
            "unsupported headless mode; use headless=bootstrap, headless=f16-probe, headless=infer, headless=vae-reference, or headless=parity"
                .into(),
        ),
    }
}

#[cfg(any(
    all(test, feature = "boogu"),
    all(feature = "boogu-web", target_arch = "wasm32")
))]
fn parse_browser_boogu_variant(
    value: Option<&str>,
    headless: Option<BrowserHeadlessMode>,
) -> Result<burn_boogu::BooguVariant, String> {
    use burn_boogu::BooguVariant;

    let variant = match value.unwrap_or("turbo") {
        "turbo" => BooguVariant::Image01Turbo,
        "edit" | "edit-turbo" => BooguVariant::Image01EditTurbo,
        "1k5" | "edit-turbo-1k5"
            if matches!(
                headless,
                Some(BrowserHeadlessMode::Parity | BrowserHeadlessMode::VaeReference)
            ) =>
        {
            BooguVariant::Image01EditTurbo1k5
        }
        "1k5" | "edit-turbo-1k5" => {
            return Err(
                "Boogu Edit-Turbo 1.5K is available only through the explicit headless=parity qualification or headless=vae-reference diagnostic route"
                    .into(),
            );
        }
        value => {
            return Err(format!(
                "unsupported Boogu variant {value:?}; browser supports turbo or edit-turbo, while headless=parity requires edit-turbo-1k5"
            ));
        }
    };
    if matches!(
        headless,
        Some(BrowserHeadlessMode::Parity | BrowserHeadlessMode::VaeReference)
    ) && variant != BooguVariant::Image01EditTurbo1k5
    {
        return Err(format!(
            "headless={} requires variant=edit-turbo-1k5",
            match headless {
                Some(BrowserHeadlessMode::Parity) => "parity",
                Some(BrowserHeadlessMode::VaeReference) => "vae-reference",
                _ => unreachable!("guard restricts this branch to fixture modes"),
            }
        ));
    }
    Ok(variant)
}

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
            ImageFileDialogPlugin,
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

#[cfg(all(feature = "boogu-web", target_arch = "wasm32"))]
fn browser_js_error(error: &wasm_bindgen::JsValue) -> String {
    error.as_string().unwrap_or_else(|| format!("{error:?}"))
}

#[cfg(all(feature = "boogu-web", target_arch = "wasm32"))]
fn report_browser_parity_terminal_failure(window: &web_sys::Window, error: &str) {
    use wasm_bindgen::JsValue;

    let message = format!("BURN_IMAGE_HEADLESS_PARITY_FAILED {error}");
    if let Some(status) = window
        .document()
        .and_then(|document| document.get_element_by_id("status"))
    {
        status.set_text_content(Some(&message));
    }
    web_sys::console::error_1(&JsValue::from_str(&message));
}

#[cfg(all(feature = "boogu-web", target_arch = "wasm32"))]
fn report_browser_vae_reference_terminal_failure(window: &web_sys::Window, error: &str) {
    use wasm_bindgen::JsValue;

    let message = format!("BURN_IMAGE_HEADLESS_VAE_REFERENCE_FAILED {error}");
    if let Some(status) = window
        .document()
        .and_then(|document| document.get_element_by_id("status"))
    {
        status.set_text_content(Some(&message));
    }
    web_sys::console::error_1(&JsValue::from_str(&message));
}

/// Start the concrete browser Boogu runtime.
///
/// Query parameters select `variant`, `profile`, `residency`, and the absolute remote `artifacts`
/// base URL.
/// When `artifacts` is omitted, the app uses the exact manifest bundle id beneath
/// `https://aberration.technology/model/`. The default is Turbo with
/// `profile=production` and `residency=resident`; the legacy
/// `f16-qwen-vision-f32` profile selector remains accepted. The production browser path verifies,
/// materializes, and retains its exact request graph on WebGPU before reporting ready. The
/// intentionally host-heavy `residency=layer-streamed-diagnostic` path requires an explicit
/// `artifacts=` URL and is not a supported production mode.
/// `headless=bootstrap` selects an opt-in no-surface F32 compute diagnostic;
/// `headless=f16-probe` runs the same final-norm probe while requiring and preserving WebGPU
/// `shader-f16`. `headless=vae-reference` runs a diagnostic-only repeated VAE encoder probe over
/// the authenticated 1.5K fixture. `headless=parity&variant=edit-turbo-1k5&fixture=https://...` is
/// the only browser surface allowed to make a complete 1.5K qualification claim.
#[cfg(all(feature = "boogu-web", target_arch = "wasm32"))]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start_boogu_web() -> Result<(), wasm_bindgen::JsValue> {
    use burn_image::{ArtifactSource, RemoteBaseUrl};
    use wasm_bindgen::JsValue;

    console_error_panic_hook::set_once();
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("Window is unavailable"))?;
    let search = window.location().search()?;
    let params = web_sys::UrlSearchParams::new_with_str(&search)?;
    let parity_requested = params.get("headless").as_deref() == Some("parity");
    let vae_reference_requested = params.get("headless").as_deref() == Some("vae-reference");
    let configuration = (|| {
        let headless = parse_browser_headless_mode(params.get("headless").as_deref())
            .map_err(|error| JsValue::from_str(&error))?;
        let residency = parse_browser_residency(params.get("residency").as_deref())
            .map_err(|error| JsValue::from_str(&error))?;
        if residency == BrowserResidencySelector::LayerStreamedDiagnostic
            && params.get("artifacts").is_none()
        {
            return Err(JsValue::from_str(
                "residency=layer-streamed-diagnostic requires an explicit artifacts= URL",
            ));
        }
        if residency == BrowserResidencySelector::LayerStreamedDiagnostic
            && !matches!(headless, None | Some(BrowserHeadlessMode::Infer))
        {
            return Err(JsValue::from_str(
                "residency=layer-streamed-diagnostic is available only to the ordinary UI or headless=infer; other headless diagnostics own an exact execution policy",
            ));
        }
        let variant = parse_browser_boogu_variant(params.get("variant").as_deref(), headless)
            .map_err(|error| JsValue::from_str(&error))?;
        let profile_selector = params
            .get("profile")
            .unwrap_or_else(|| BOOGU_PRODUCTION_PROFILE_SELECTOR.to_owned());
        let profile = parse_boogu_storage_profile(&profile_selector).ok_or_else(|| {
            JsValue::from_str(&format!(
                "unsupported Boogu profile {profile_selector:?}; use production or pass an explicit diagnostic profile with artifacts="
            ))
        })?;
        if matches!(
            headless,
            Some(BrowserHeadlessMode::Parity | BrowserHeadlessMode::VaeReference)
        ) && profile != burn_boogu::artifacts::BooguStorageProfile::F16QwenVisionF32
        {
            return Err(JsValue::from_str(
                "1.5K fixture diagnostics require profile=production",
            ));
        }
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
                    .expect("published bundles have a canonical CDN base URL")
            }
        };
        let base_url =
            RemoteBaseUrl::new(artifacts).map_err(|error| JsValue::from_str(&error.to_string()))?;
        let mut settings =
            BooguAdapterSettings::verified_default(ArtifactSource::Remote { base_url });
        settings.storage_profile = profile;
        Ok::<_, JsValue>((headless, variant, residency, settings))
    })();
    let (headless, variant, residency, settings) = match configuration {
        Ok(configuration) => configuration,
        Err(error) if parity_requested => {
            report_browser_parity_terminal_failure(&window, &browser_js_error(&error));
            return Ok(());
        }
        Err(error) if vae_reference_requested => {
            report_browser_vae_reference_terminal_failure(&window, &browser_js_error(&error));
            return Ok(());
        }
        Err(error) => return Err(error),
    };
    browser_boogu::report_browser_runtime_preparing(
        "Initializing WebGPU and the verified GPU-resident model runtime",
    );
    if matches!(
        headless,
        Some(BrowserHeadlessMode::Bootstrap | BrowserHeadlessMode::F16Probe)
    ) {
        let preserve_f16 = headless == Some(BrowserHeadlessMode::F16Probe);
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
    if headless == Some(BrowserHeadlessMode::Infer) {
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
            let policy = match residency {
                BrowserResidencySelector::Resident => {
                    BrowserBooguResidencyPolicy::HighVramResidentDenseF32
                }
                BrowserResidencySelector::LayerStreamedDiagnostic => {
                    BrowserBooguResidencyPolicy::LayerStreamedDiagnostic
                }
            };
            let result = BrowserBooguFactory::infer_no_surface_with_residency(
                variant, settings, policy, request,
            )
            .await;
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
    if headless == Some(BrowserHeadlessMode::VaeReference) {
        let fixture_base = match params
            .get("fixture")
            .ok_or_else(|| {
                JsValue::from_str(
                    "headless=vae-reference requires an absolute HTTP(S) fixture= base URL",
                )
            })
            .and_then(|fixture| {
                RemoteBaseUrl::new(fixture)
                    .map_err(|error| JsValue::from_str(&format!("invalid fixture= URL: {error}")))
            }) {
            Ok(fixture_base) => fixture_base,
            Err(error) => {
                report_browser_vae_reference_terminal_failure(&window, &browser_js_error(&error));
                return Ok(());
            }
        };
        let status = window
            .document()
            .and_then(|document| document.get_element_by_id("status"));
        wasm_bindgen_futures::spawn_local(async move {
            let result =
                BrowserBooguFactory::vae_reference_no_surface(variant, settings, fixture_base)
                    .await;
            let (message, failed) = match result {
                Ok(report) => match serde_json::to_string(&report) {
                    Ok(json) if report.diagnostic_passed => (
                        format!("BURN_IMAGE_HEADLESS_VAE_REFERENCE_OK {json}"),
                        false,
                    ),
                    Ok(json) => (
                        format!("BURN_IMAGE_HEADLESS_VAE_REFERENCE_FAILED {json}"),
                        true,
                    ),
                    Err(error) => (
                        format!("BURN_IMAGE_HEADLESS_VAE_REFERENCE_FAILED report JSON: {error}"),
                        true,
                    ),
                },
                Err(error) => (
                    format!("BURN_IMAGE_HEADLESS_VAE_REFERENCE_FAILED {error}"),
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
    if headless == Some(BrowserHeadlessMode::Parity) {
        let fixture_base = match params
            .get("fixture")
            .ok_or_else(|| {
                JsValue::from_str("headless=parity requires an absolute HTTP(S) fixture= base URL")
            })
            .and_then(|fixture| {
                RemoteBaseUrl::new(fixture)
                    .map_err(|error| JsValue::from_str(&format!("invalid fixture= URL: {error}")))
            }) {
            Ok(fixture_base) => fixture_base,
            Err(error) => {
                report_browser_parity_terminal_failure(&window, &browser_js_error(&error));
                return Ok(());
            }
        };
        let status = window
            .document()
            .and_then(|document| document.get_element_by_id("status"));
        wasm_bindgen_futures::spawn_local(async move {
            let result =
                BrowserBooguFactory::parity_no_surface(variant, settings, fixture_base).await;
            let (message, failed) = match result {
                Ok(report) => match serde_json::to_string(&report) {
                    Ok(json) if report.numerical_parity_claimed => {
                        (format!("BURN_IMAGE_HEADLESS_PARITY_OK {json}"), false)
                    }
                    Ok(json) => (format!("BURN_IMAGE_HEADLESS_PARITY_FAILED {json}"), true),
                    Err(error) => (
                        format!("BURN_IMAGE_HEADLESS_PARITY_FAILED report JSON: {error}"),
                        true,
                    ),
                },
                Err(error) => (format!("BURN_IMAGE_HEADLESS_PARITY_FAILED {error}"), true),
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
    let factory = match residency {
        BrowserResidencySelector::Resident => BrowserBooguFactory::new(variant),
        BrowserResidencySelector::LayerStreamedDiagnostic => BrowserBooguFactory::with_residency(
            variant,
            BrowserBooguResidencyPolicy::LayerStreamedDiagnostic,
        ),
    };
    app::build_boogu_app(settings, factory).run();
    Ok(())
}

#[cfg(test)]
mod web_shell_tests {
    #[cfg(feature = "boogu")]
    #[test]
    fn browser_1k5_routes_are_explicit_and_fail_closed_correctness() {
        use burn_boogu::BooguVariant;

        let parity = super::parse_browser_headless_mode(Some("parity")).unwrap();
        assert_eq!(parity, Some(super::BrowserHeadlessMode::Parity));
        assert_eq!(
            super::parse_browser_boogu_variant(Some("edit-turbo-1k5"), parity).unwrap(),
            BooguVariant::Image01EditTurbo1k5
        );
        assert_eq!(
            super::parse_browser_boogu_variant(Some("1k5"), parity).unwrap(),
            BooguVariant::Image01EditTurbo1k5
        );
        let vae_reference = super::parse_browser_headless_mode(Some("vae-reference")).unwrap();
        assert_eq!(
            vae_reference,
            Some(super::BrowserHeadlessMode::VaeReference)
        );
        assert_eq!(
            super::parse_browser_boogu_variant(Some("edit-turbo-1k5"), vae_reference).unwrap(),
            BooguVariant::Image01EditTurbo1k5
        );

        for headless in [
            None,
            Some(super::BrowserHeadlessMode::Bootstrap),
            Some(super::BrowserHeadlessMode::F16Probe),
            Some(super::BrowserHeadlessMode::Infer),
        ] {
            let error =
                super::parse_browser_boogu_variant(Some("edit-turbo-1k5"), headless).unwrap_err();
            assert!(error.contains("explicit headless=parity"), "{error}");
        }
        for headless in [parity, vae_reference] {
            for variant in [None, Some("turbo"), Some("edit-turbo")] {
                let error = super::parse_browser_boogu_variant(variant, headless).unwrap_err();
                assert!(error.contains("requires variant=edit-turbo-1k5"), "{error}");
            }
        }
        assert!(super::parse_browser_headless_mode(Some("parity-ish")).is_err());
    }

    #[cfg(feature = "boogu")]
    #[test]
    fn production_profile_selector_keeps_legacy_alias_correctness() {
        use burn_boogu::artifacts::BooguStorageProfile;

        assert_eq!(
            super::parse_boogu_storage_profile(super::BOOGU_PRODUCTION_PROFILE_SELECTOR),
            Some(BooguStorageProfile::F16QwenVisionF32)
        );
        assert_eq!(
            super::parse_boogu_storage_profile("f16-qwen-vision-f32"),
            Some(BooguStorageProfile::F16QwenVisionF32)
        );
        assert_eq!(super::parse_boogu_storage_profile("f32"), None);
    }

    #[cfg(feature = "boogu")]
    #[test]
    fn browser_residency_defaults_to_resident_and_streaming_is_explicit_correctness() {
        assert_eq!(
            super::parse_browser_residency(None).unwrap(),
            super::BrowserResidencySelector::Resident
        );
        assert_eq!(
            super::parse_browser_residency(Some("resident")).unwrap(),
            super::BrowserResidencySelector::Resident
        );
        assert_eq!(
            super::parse_browser_residency(Some("layer-streamed-diagnostic")).unwrap(),
            super::BrowserResidencySelector::LayerStreamedDiagnostic
        );
        assert!(super::parse_browser_residency(Some("streamed")).is_err());
    }

    #[test]
    fn web_shell_exposes_streaming_progress_contract_correctness() {
        let html = include_str!("../www/index.html");
        for required in [
            "id=\"model-loader\"",
            "id=\"artifact-progress\"",
            "aria-live=\"polite\"",
            "burn-image-runtime",
            "burn-image-progress",
            "verifiedObjects",
            "transferredBytes",
            "Reload runtime",
            "id=\"burn-image-reference-input\"",
            "loadReferenceFile",
            "MAX_REFERENCE_BYTES",
            "failRun",
        ] {
            assert!(html.contains(required), "web shell omits {required}");
        }
        assert!(html.contains("preloaded to WebGPU during preparation"));
    }

    #[test]
    fn authored_shell_text_uses_default_font_safe_ascii_correctness() {
        for (path, source) in [
            ("app.rs", include_str!("app.rs")),
            ("controls.rs", include_str!("controls.rs")),
            ("file_dialog.rs", include_str!("file_dialog.rs")),
            ("viewer.rs", include_str!("viewer.rs")),
            ("www/index.html", include_str!("../www/index.html")),
        ] {
            assert!(
                source.is_ascii(),
                "authored shell source {path} contains non-ASCII text that may render as tofu"
            );
        }
    }
}
