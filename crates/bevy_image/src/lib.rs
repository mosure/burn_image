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
#[cfg(all(feature = "boogu-web", any(test, target_arch = "wasm32")))]
mod browser_turbo_first_dmd_fixture;
#[cfg(all(feature = "boogu-native", not(target_arch = "wasm32")))]
pub mod native_artifact_cache;
#[cfg(all(feature = "boogu-native", not(target_arch = "wasm32")))]
pub mod native_automation;
#[cfg(all(feature = "boogu-native", not(target_arch = "wasm32")))]
pub mod native_boogu;
#[cfg(all(feature = "boogu-native", not(target_arch = "wasm32")))]
pub mod native_output_qualification;

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
pub use native_automation::*;
#[cfg(all(feature = "boogu-native", not(target_arch = "wasm32")))]
pub use native_boogu::*;
#[cfg(all(feature = "boogu-native", not(target_arch = "wasm32")))]
pub use native_output_qualification::*;

use bevy::prelude::*;

/// Human-readable application name used by native and browser shells.
pub const APP_NAME: &str = "burn image";

/// Internal stage prefix used to carry native model-switch setup detail through the existing
/// model-neutral progress channel without presenting the encoded stage name to users.
#[cfg(any(
    feature = "app",
    all(feature = "boogu-native", not(target_arch = "wasm32"))
))]
pub(crate) const MODEL_SWITCH_PROGRESS_STAGE_PREFIX: &str = "model-switch-progress:";

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
    TurboFirstDmd,
    Parity,
    VaeReference,
}

#[cfg(any(
    all(test, feature = "boogu"),
    all(feature = "boogu-web", target_arch = "wasm32")
))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum BrowserResidencySelector {
    Resident,
    QualificationF32,
    #[default]
    LowVram,
    RuntimeQ8,
    PreloadedPackedF16,
    LayerStreamedDiagnostic,
}

#[cfg(any(
    all(test, feature = "boogu"),
    all(feature = "boogu-web", target_arch = "wasm32")
))]
fn parse_browser_residency(value: Option<&str>) -> Result<BrowserResidencySelector, String> {
    match value {
        None | Some("low-vram") => Ok(BrowserResidencySelector::LowVram),
        Some("low-vram-runtime-q8-denoiser") => Ok(BrowserResidencySelector::RuntimeQ8),
        Some("low-vram-preloaded-packed-f16-dense-f32-per-stage-denoiser") => {
            Ok(BrowserResidencySelector::PreloadedPackedF16)
        }
        Some("low-vram-preloaded-main-core-ffn-gate-up-q8-denoiser") => Err(
            "the Turbo main-core gate/up-Q8 browser policy is retired after failing its final numerical gate; use residency=low-vram or residency=low-vram-preloaded-packed-f16-dense-f32-per-stage-denoiser"
                .into(),
        ),
        Some("low-vram-retained-q8-dense-f32-per-stage-denoiser") => Err(
            "the retained-Q8 dense-F32-per-stage browser policy is retired after real-browser non-finite output; use residency=low-vram"
                .into(),
        ),
        Some("resident") | Some("high-vram-resident-dense-f32") => {
            Ok(BrowserResidencySelector::Resident)
        }
        Some("qualification-f32")
        | Some("qualification-per-request-f32-denoiser-retained") => {
            Ok(BrowserResidencySelector::QualificationF32)
        }
        Some("layer-streamed-diagnostic") => {
            Ok(BrowserResidencySelector::LayerStreamedDiagnostic)
        }
        Some(_) => Err(
            "unsupported browser residency; use residency=resident, residency=qualification-f32, residency=low-vram, residency=low-vram-runtime-q8-denoiser, residency=low-vram-preloaded-packed-f16-dense-f32-per-stage-denoiser, or the explicit residency=layer-streamed-diagnostic"
                .into(),
        ),
    }
}

#[cfg(any(
    all(test, feature = "boogu"),
    all(feature = "boogu-web", target_arch = "wasm32")
))]
fn resolve_browser_residency(
    value: Option<&str>,
    headless: Option<BrowserHeadlessMode>,
) -> Result<BrowserResidencySelector, String> {
    if value.is_none() && headless.is_none() {
        // The interactive default prioritizes warm-session throughput. Its mandatory shared-device
        // allocation preflight fails before model-weight download on devices that cannot admit the
        // resident plan; `residency=low-vram` remains the explicit bounded-memory alternative.
        Ok(BrowserResidencySelector::Resident)
    } else {
        parse_browser_residency(value)
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
        Some("turbo-first-dmd") => Ok(Some(BrowserHeadlessMode::TurboFirstDmd)),
        Some("parity") => Ok(Some(BrowserHeadlessMode::Parity)),
        Some("vae-reference") => Ok(Some(BrowserHeadlessMode::VaeReference)),
        Some(_) => Err(
            "unsupported headless mode; use headless=bootstrap, headless=f16-probe, headless=infer, headless=turbo-first-dmd, headless=vae-reference, or headless=parity"
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
        "1k5" | "edit-turbo-1k5" => BooguVariant::Image01EditTurbo1k5,
        value => {
            return Err(format!(
                "unsupported Boogu variant {value:?}; browser supports turbo, edit-turbo, or edit-turbo-1k5"
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
    if headless == Some(BrowserHeadlessMode::TurboFirstDmd) && variant != BooguVariant::Image01Turbo
    {
        return Err("headless=turbo-first-dmd requires variant=turbo".into());
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

#[cfg(all(feature = "boogu-web", target_arch = "wasm32"))]
fn report_browser_turbo_first_dmd_terminal_failure(window: &web_sys::Window, error: &str) {
    use wasm_bindgen::JsValue;

    let message = format!("BURN_IMAGE_HEADLESS_TURBO_FIRST_DMD_FAILED {error}");
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
/// `profile=production` and warm `residency=resident`; the legacy
/// `f16-qwen-vision-f32` profile selector remains accepted. The interactive default authenticates,
/// materializes, and retains the selected request graph before reporting ready, so later requests
/// reuse the device-resident pipeline. Its mandatory shared-device VRAM preflight fails before
/// weight download when the conservative plan cannot be committed. Explicit `residency=low-vram`
/// verifies the complete manifest closure before reporting ready. Turbo preloads a
/// complete authenticated packed-F16 denoiser and materializes exactly one semantic stage to F32
/// at a time; this is storage compression, not on-device quantized execution. Qwen and VAE stages
/// remain streamed. Edit variants retain a request-scoped, all-eligible runtime-Q8 denoiser and use
/// direct quantized matmul. Both policies have conservative sub-32-GB
/// resource plans and exact per-request cache/network traffic reports. Ordinary Turbo UI loading
/// and both Edit variants require the integrity-checked persistent object cache. Startup compares
/// the selected executable closure with exact existing Cache Storage keys and fails before model
/// transfer when origin quota cannot hold the missing bytes plus safety reserve.
/// `headless=parity&residency=low-vram` replays the low-VRAM policy against the exact
/// authenticated 1.5K fixture. Exact F32 fixture replay uses
/// `headless=parity&residency=qualification-f32`: Qwen/VAE are streamed per request and only the
/// F32 denoiser is retained through the four DMD steps, so its evidence does not claim all-stage
/// resident-dense execution. The
/// intentionally host-heavy `residency=layer-streamed-diagnostic` path requires an explicit
/// `artifacts=` URL and is not a supported production mode.
/// `headless=bootstrap` selects an opt-in no-surface F32 compute diagnostic;
/// `headless=f16-probe` runs the same final-norm probe while requiring and preserving WebGPU
/// `shader-f16`. `headless=infer` accepts the full Turbo 256-1024 release range; edit inference,
/// including 1.5K, uses the ordinary UI because it requires a reference image.
/// `headless=vae-reference` and `headless=parity&variant=edit-turbo-1k5&fixture=https://...` retain
/// their exact authenticated-fixture contracts.
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
    let turbo_first_dmd_requested = params.get("headless").as_deref() == Some("turbo-first-dmd");
    let configuration = (|| {
        let headless = parse_browser_headless_mode(params.get("headless").as_deref())
            .map_err(|error| JsValue::from_str(&error))?;
        let residency = resolve_browser_residency(params.get("residency").as_deref(), headless)
            .map_err(|error| JsValue::from_str(&error))?;
        if residency == BrowserResidencySelector::LayerStreamedDiagnostic
            && params.get("artifacts").is_none()
        {
            return Err(JsValue::from_str(
                "residency=layer-streamed-diagnostic requires an explicit artifacts= URL",
            ));
        }
        if residency == BrowserResidencySelector::QualificationF32
            && headless != Some(BrowserHeadlessMode::Parity)
        {
            return Err(JsValue::from_str(
                "residency=qualification-f32 is reserved for headless=parity exact-fixture replay",
            ));
        }
        if headless == Some(BrowserHeadlessMode::Parity)
            && !matches!(
                residency,
                BrowserResidencySelector::QualificationF32
                    | BrowserResidencySelector::LowVram
                    | BrowserResidencySelector::RuntimeQ8
            )
        {
            return Err(JsValue::from_str(
                "headless=parity requires residency=qualification-f32 or residency=low-vram",
            ));
        }
        if headless == Some(BrowserHeadlessMode::TurboFirstDmd)
            && !matches!(
                residency,
                BrowserResidencySelector::LowVram | BrowserResidencySelector::PreloadedPackedF16
            )
        {
            return Err(JsValue::from_str(
                "headless=turbo-first-dmd requires residency=low-vram or the exact preloaded packed-F16 dense-F32-per-stage policy selector",
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
        if variant == burn_boogu::BooguVariant::Image01Turbo
            && residency == BrowserResidencySelector::RuntimeQ8
        {
            return Err(JsValue::from_str(
                "ordinary Turbo rejects residency=low-vram-runtime-q8-denoiser because its direct-Q8 path is not numerically qualified; use residency=low-vram",
            ));
        }
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
            Some(
                BrowserHeadlessMode::Parity
                    | BrowserHeadlessMode::VaeReference
                    | BrowserHeadlessMode::TurboFirstDmd
            )
        ) && profile != burn_boogu::artifacts::BooguStorageProfile::F16QwenVisionF32
        {
            return Err(JsValue::from_str(
                "fixture diagnostics require profile=production",
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
        Err(error) if turbo_first_dmd_requested => {
            report_browser_turbo_first_dmd_terminal_failure(&window, &browser_js_error(&error));
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
    if headless == Some(BrowserHeadlessMode::TurboFirstDmd) {
        let fixture_profile_value = params
            .get("fixture-profile")
            .unwrap_or_else(|| "release-256-bf16".into());
        let fixture_profile =
            match browser_turbo_first_dmd_fixture::BrowserTurboFirstDmdFixtureProfile::parse(
                &fixture_profile_value,
            ) {
                Some(profile) => profile,
                None => {
                    report_browser_turbo_first_dmd_terminal_failure(
                        &window,
                        "unsupported fixture-profile; use release-256-bf16 or qualification-1024-bf16",
                    );
                    return Ok(());
                }
            };
        let fixture_base = match params
            .get("fixture")
            .ok_or_else(|| {
                JsValue::from_str(
                    "headless=turbo-first-dmd requires an absolute HTTP(S) fixture= base URL",
                )
            })
            .and_then(|fixture| {
                RemoteBaseUrl::new(fixture)
                    .map_err(|error| JsValue::from_str(&format!("invalid fixture= URL: {error}")))
            }) {
            Ok(fixture_base) => fixture_base,
            Err(error) => {
                report_browser_turbo_first_dmd_terminal_failure(&window, &browser_js_error(&error));
                return Ok(());
            }
        };
        let status = window
            .document()
            .and_then(|document| document.get_element_by_id("status"));
        wasm_bindgen_futures::spawn_local(async move {
            let result = BrowserBooguFactory::turbo_first_dmd_no_surface(
                variant,
                settings,
                fixture_base,
                fixture_profile,
            )
            .await;
            let (message, failed) = match result {
                Ok(report) => match serde_json::to_string(&report) {
                    Ok(json) if report.diagnostic_passed && !report.numerical_parity_claimed => (
                        format!("BURN_IMAGE_HEADLESS_TURBO_FIRST_DMD_OK {json}"),
                        false,
                    ),
                    Ok(json) => (
                        format!("BURN_IMAGE_HEADLESS_TURBO_FIRST_DMD_FAILED {json}"),
                        true,
                    ),
                    Err(error) => (
                        format!("BURN_IMAGE_HEADLESS_TURBO_FIRST_DMD_FAILED report JSON: {error}"),
                        true,
                    ),
                },
                Err(error) => (
                    format!("BURN_IMAGE_HEADLESS_TURBO_FIRST_DMD_FAILED {error}"),
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
                BrowserResidencySelector::QualificationF32 => {
                    unreachable!("configuration reserves qualification-f32 for parity")
                }
                BrowserResidencySelector::LowVram => {
                    browser_boogu::default_browser_low_vram_residency(variant)
                }
                BrowserResidencySelector::RuntimeQ8 => {
                    BrowserBooguResidencyPolicy::LowVramRuntimeQ8Denoiser
                }
                BrowserResidencySelector::PreloadedPackedF16 => {
                    BrowserBooguResidencyPolicy::LowVramPreloadedPackedF16Denoiser
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
            let result = match residency {
                BrowserResidencySelector::QualificationF32 => {
                    BrowserBooguFactory::parity_no_surface(variant, settings, fixture_base).await
                }
                BrowserResidencySelector::LowVram => {
                    BrowserBooguFactory::parity_no_surface_with_residency(
                        variant,
                        settings,
                        browser_boogu::default_browser_low_vram_residency(variant),
                        fixture_base,
                    )
                    .await
                }
                BrowserResidencySelector::RuntimeQ8 => {
                    BrowserBooguFactory::parity_no_surface_with_residency(
                        variant,
                        settings,
                        BrowserBooguResidencyPolicy::LowVramRuntimeQ8Denoiser,
                        fixture_base,
                    )
                    .await
                }
                BrowserResidencySelector::Resident
                | BrowserResidencySelector::PreloadedPackedF16
                | BrowserResidencySelector::LayerStreamedDiagnostic => {
                    unreachable!("configuration accepts only qualification residencies for parity")
                }
            };
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
        BrowserResidencySelector::Resident => BrowserBooguFactory::with_residency(
            variant,
            BrowserBooguResidencyPolicy::HighVramResidentDenseF32,
        ),
        BrowserResidencySelector::QualificationF32 => {
            unreachable!("configuration reserves qualification-f32 for parity")
        }
        BrowserResidencySelector::LowVram => BrowserBooguFactory::with_residency(
            variant,
            browser_boogu::default_browser_low_vram_residency(variant),
        ),
        BrowserResidencySelector::RuntimeQ8 => BrowserBooguFactory::with_residency(
            variant,
            BrowserBooguResidencyPolicy::LowVramRuntimeQ8Denoiser,
        ),
        BrowserResidencySelector::PreloadedPackedF16 => BrowserBooguFactory::with_residency(
            variant,
            BrowserBooguResidencyPolicy::LowVramPreloadedPackedF16Denoiser,
        ),
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
    fn browser_1k5_ui_and_fixture_routes_are_explicit_and_fail_closed_correctness() {
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
            assert_eq!(
                super::parse_browser_boogu_variant(Some("edit-turbo-1k5"), headless).unwrap(),
                BooguVariant::Image01EditTurbo1k5
            );
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
    fn browser_turbo_first_dmd_route_is_exact_and_variant_scoped_correctness() {
        use burn_boogu::BooguVariant;

        let mode = super::parse_browser_headless_mode(Some("turbo-first-dmd")).unwrap();
        assert_eq!(mode, Some(super::BrowserHeadlessMode::TurboFirstDmd));
        assert_eq!(
            super::parse_browser_boogu_variant(Some("turbo"), mode).unwrap(),
            BooguVariant::Image01Turbo
        );
        for edit in ["edit", "edit-turbo", "1k5", "edit-turbo-1k5"] {
            let error = super::parse_browser_boogu_variant(Some(edit), mode).unwrap_err();
            assert!(error.contains("requires variant=turbo"), "{error}");
        }
        let source = include_str!("lib.rs");
        for required in [
            "BURN_IMAGE_HEADLESS_TURBO_FIRST_DMD_OK",
            "BURN_IMAGE_HEADLESS_TURBO_FIRST_DMD_FAILED",
            "BrowserBooguFactory::turbo_first_dmd_no_surface",
        ] {
            assert!(
                source.contains(required),
                "Turbo first-DMD route omits {required}"
            );
        }
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
    fn browser_interactive_residency_defaults_warm_and_diagnostics_stay_bounded_correctness() {
        assert_eq!(
            super::parse_browser_residency(None).unwrap(),
            super::BrowserResidencySelector::LowVram
        );
        assert_eq!(
            super::resolve_browser_residency(None, None).unwrap(),
            super::BrowserResidencySelector::Resident
        );
        assert_eq!(
            super::resolve_browser_residency(None, Some(super::BrowserHeadlessMode::Bootstrap))
                .unwrap(),
            super::BrowserResidencySelector::LowVram
        );
        assert_eq!(
            super::parse_browser_residency(Some("resident")).unwrap(),
            super::BrowserResidencySelector::Resident
        );
        assert_eq!(
            super::parse_browser_residency(Some("qualification-f32")).unwrap(),
            super::BrowserResidencySelector::QualificationF32
        );
        assert_eq!(
            super::parse_browser_residency(Some("low-vram")).unwrap(),
            super::BrowserResidencySelector::LowVram
        );
        assert_eq!(
            super::parse_browser_residency(Some("low-vram-runtime-q8-denoiser")).unwrap(),
            super::BrowserResidencySelector::RuntimeQ8
        );
        assert!(
            super::parse_browser_residency(Some(
                "low-vram-retained-q8-dense-f32-per-stage-denoiser"
            ))
            .unwrap_err()
            .contains("retired")
        );
        assert!(
            super::parse_browser_residency(Some(
                "low-vram-preloaded-main-core-ffn-gate-up-q8-denoiser"
            ))
            .unwrap_err()
            .contains("retired")
        );
        assert_eq!(
            super::parse_browser_residency(Some(
                "low-vram-preloaded-packed-f16-dense-f32-per-stage-denoiser"
            ))
            .unwrap(),
            super::BrowserResidencySelector::PreloadedPackedF16
        );
        for retired_or_internal in [
            "browser-low-vram-preloaded-main-core-ffn-gate-up-q8-denoiser",
            "low-vram-preloaded-ffn-gate-up-q8-denoiser",
            "browser-low-vram-preloaded-ffn-gate-up-q8-denoiser",
            "low-vram-preloaded-ffn-core-q8-denoiser",
            "browser-low-vram-preloaded-ffn-core-q8-denoiser",
            "low-vram-preloaded-attention-ffn-core-q8-denoiser",
            "browser-low-vram-preloaded-attention-ffn-core-q8-denoiser",
        ] {
            assert!(super::parse_browser_residency(Some(retired_or_internal)).is_err());
        }
        assert_eq!(
            super::parse_browser_residency(Some("layer-streamed-diagnostic")).unwrap(),
            super::BrowserResidencySelector::LayerStreamedDiagnostic
        );
        assert!(super::parse_browser_residency(Some("streamed")).is_err());
    }

    #[test]
    fn explicit_runtime_q8_routes_do_not_reenter_variant_default_correctness() {
        let source = include_str!("lib.rs");
        for anchor in [
            "if headless == Some(BrowserHeadlessMode::Infer)",
            "let factory = match residency",
        ] {
            let route = source
                .split_once(anchor)
                .unwrap_or_else(|| panic!("missing browser routing anchor {anchor}"))
                .1
                .split_once("BrowserResidencySelector::RuntimeQ8 =>")
                .unwrap_or_else(|| panic!("missing explicit runtime-Q8 route after {anchor}"))
                .1;
            let branch = &route[..route.len().min(400)];
            assert!(
                branch.contains("BrowserBooguResidencyPolicy::LowVramRuntimeQ8Denoiser"),
                "explicit runtime-Q8 route after {anchor} does not select the direct-Q policy"
            );
            assert!(
                !branch.contains("default_browser_low_vram_residency"),
                "explicit runtime-Q8 route after {anchor} silently reentered the variant default"
            );
        }
    }

    #[test]
    fn web_shell_uses_bevy_as_its_only_visible_interface_correctness() {
        let html = include_str!("../www/index.html");
        for required in [
            "id=\"status\" aria-hidden=\"true\"",
            "id=\"burn-image-reference-input\"",
            "id=\"burn-image\"",
            ".JPG",
            "HEIC/HEIF decoding is not available in this build",
            "rel=\"icon\"",
            "./out/burn-image-icon.png",
            "provide_reference_image_error",
            "reportReferenceError",
            "loadReferenceFile",
            "MAX_REFERENCE_BYTES",
            "await init()",
            "The visible interface is exclusively Bevy on every platform",
        ] {
            assert!(html.contains(required), "web shell omits {required}");
        }
        for forbidden in [
            "id=\"model-loader\"",
            "id=\"artifact-progress\"",
            "loader-panel",
            "burn-image-runtime",
            "burn-image-progress",
            "configureModelReleaseSelector",
            "./model_selector.mjs",
            "surface_inference_suspended",
            "requestAnimationFrame",
        ] {
            assert!(
                !html.contains(forbidden),
                "web shell retains fragmented browser UI behavior {forbidden}"
            );
        }
        assert!(
            html.contains("#status,\n      #burn-image-reference-input"),
            "the headless status terminal and browser picker must remain nonvisual"
        );
    }

    #[test]
    fn browser_transport_uses_native_async_part_hashing_correctness() {
        let source = include_str!("artifact_stream.rs");
        let runtime = include_str!("browser_boogu.rs");
        for required in [
            "verify_browser_transport_part_bytes_async",
            "digest_with_str_and_u8_array(\"SHA-256\", bytes)",
            "wasm_bindgen_futures::JsFuture::from(promise)",
            "verify_browser_transport_reconstruction(file, &bytes)",
            "fn browser_dom_event_stream_requested()",
            "if !browser_dom_event_stream_requested()",
            "Interactive progress goes directly to ImageRunnerEvent/Bevy",
        ] {
            assert!(
                source.contains(required) || runtime.contains(required),
                "browser cache path omits {required}"
            );
        }
        assert_eq!(
            source
                .matches("verify_browser_transport_part_bytes_async(part, &bytes).await")
                .count(),
            2,
            "cache hit and one integrity refetch must share the native async part verifier"
        );
    }

    #[test]
    fn browser_model_switching_is_owned_by_the_bevy_selector_correctness() {
        let runtime = include_str!("browser_boogu.rs");
        let controls = include_str!("controls.rs");
        for required in [
            "fn browser_release_switching_enabled()",
            "!params.has(\"artifacts\") && !params.has(\"headless\")",
            "pub(crate) fn request_browser_model_release(",
            ".assign(&url.href())",
            "the previous browser model is unloading",
        ] {
            assert!(
                runtime.contains(required) || controls.contains(required),
                "Bevy-owned browser model switching omits {required}"
            );
        }
        assert!(
            !include_str!("../www/index.html").contains("model-release"),
            "the browser shell must not reintroduce a second model selector"
        );
    }

    #[test]
    fn canonical_turbo_transfer_plan_uses_only_executable_artifacts_correctness() {
        let source = include_str!("browser_boogu.rs");
        for required in [
            "BROWSER_TURBO_ACTIVE_LOGICAL_OBJECTS: u32 = 186",
            "BROWSER_TURBO_ACTIVE_UNIQUE_TRANSPORT_PARTS: u32 = 1_751",
            "BROWSER_TURBO_ACTIVE_TRANSPORT_BYTES: u64 = 35_106_151_424",
            "active_manifest_weight_artifacts",
            "browser_resident_artifact_required",
            "retain_transfer_logical_objects",
            "validate_browser_turbo_active_transfer_plan",
        ] {
            assert!(
                source.contains(required),
                "canonical Turbo active transfer contract omits {required}"
            );
        }
    }

    #[test]
    fn ordinary_browser_cache_is_selected_model_scoped_and_quota_preflighted_correctness() {
        let stream = include_str!("artifact_stream.rs");
        let runtime = include_str!("browser_boogu.rs");
        for required in [
            "self.require_persistent_range_cache = true",
            "BrowserPersistentCachePlan",
            "extend_persistent_cache_plan",
            "active_transfer_objects",
            "cache.keys()",
            "window.navigator().storage()",
            ".estimate()",
            "browser_storage_estimate_field",
            "js_sys::Reflect::get",
            ".persisted()",
            ".persist()",
            "BrowserStorageQuotaInsufficient",
            "selected model cache plan {cache_shape:?} differs from transfer plan {transfer_shape:?}",
            "Loading selected model from persistent cache",
            "Downloading selected model into persistent cache",
        ] {
            assert!(
                stream.contains(required) || runtime.contains(required),
                "ordinary browser persistent-cache contract omits {required}"
            );
        }
        assert!(
            !stream.contains("dyn_into::<web_sys::StorageEstimate>()"),
            "WebIDL StorageEstimate dictionaries must not use a checked JavaScript class cast"
        );
        let build = runtime
            .split_once("let mut cache_plan = BrowserPersistentCachePlan::default()")
            .expect("browser runtime omits selected-model cache preflight")
            .1;
        assert!(
            build
                .find("preflight_browser_persistent_cache(&cache_plan)")
                .unwrap()
                < build.find("let qwen_config_bytes =").unwrap(),
            "browser runtime reads model files before persistent-cache admission"
        );
    }

    #[test]
    fn browser_vram_preflight_precedes_weight_stage_loading_correctness() {
        let source = include_str!("browser_boogu.rs");
        for required in [
            "run_browser_vram_preflight(",
            "create_buffer(&wgpu::BufferDescriptor",
            "encoder.clear_buffer(&buffer, 0, None)",
            "queue.on_submitted_work_done",
            "BROWSER_VRAM_PREFLIGHT_TIMEOUT_MS",
            "buffer.destroy()",
            "failed before model-weight download",
            "shared_device_and_queue: true",
        ] {
            assert!(
                source.contains(required),
                "browser VRAM preflight contract omits {required}"
            );
        }
        let build = source
            .split_once("let vram_preflight =")
            .expect("browser engine omits model-specific VRAM admission")
            .1;
        let preflight = build
            .find("run_browser_vram_preflight(")
            .expect("browser engine omits the allocation preflight call");
        for weight_loader in [
            "VerifiedAsyncPackedF16DenoiserStageSource::new(",
            "VerifiedAsyncBurnpackDenoiserStageSource::new(",
        ] {
            let loader = build
                .find(weight_loader)
                .unwrap_or_else(|| panic!("browser engine omits {weight_loader}"));
            assert!(
                preflight < loader,
                "browser engine can construct {weight_loader} before GPU memory admission"
            );
        }
    }

    #[test]
    fn browser_resident_requests_remain_warm_without_artifact_io_correctness() {
        let source = include_str!("browser_boogu.rs");
        for required in [
            "policies.retain_qwen_stages = true",
            "policies.retain_vae_stages = true",
            "policies.retain_denoiser_stages = true",
            "policies.eager_preload = true",
            "eager-preload/qwen+vae+denoiser/zero-inference-artifact-transfers",
            "resident browser request performed artifact I/O after its eager preload",
        ] {
            assert!(
                source.contains(required),
                "browser resident warm-session contract omits {required}"
            );
        }
        assert!(
            source.matches("self.validate_resident_caches()?;").count() >= 2,
            "resident caches must be validated both before and after inference"
        );
    }

    #[test]
    fn authored_shell_text_uses_default_font_safe_ascii_correctness() {
        for (path, source) in [
            ("app.rs", include_str!("app.rs")),
            ("controls.rs", include_str!("controls.rs")),
            ("file_dialog.rs", include_str!("file_dialog.rs")),
            ("viewer.rs", include_str!("viewer.rs")),
            ("www/index.html", include_str!("../www/index.html")),
            (
                "www/model_selector.mjs",
                include_str!("../www/model_selector.mjs"),
            ),
        ] {
            assert!(
                source.is_ascii(),
                "authored shell source {path} contains non-ASCII text that may render as tofu"
            );
        }
    }
}
