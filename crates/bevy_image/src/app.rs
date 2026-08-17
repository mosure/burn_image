//! Small native/web shell. Model crates add their own dispatch systems and UI
//! controls while reusing the status and image-view surfaces here.

use bevy::{
    camera::visibility::RenderLayers,
    prelude::*,
    render::{
        RenderPlugin,
        settings::{RenderCreation, WgpuFeatures, WgpuSettings, WgpuSettingsPriority},
    },
    window::WindowPlugin,
};

use crate::{APP_NAME, BackendState, BackendStatus, BurnImageFrontendPlugin};

const APP_WINDOW_ID: &str = "burn-image";

#[cfg(not(target_arch = "wasm32"))]
const APP_ICON_PNG: &[u8] = include_bytes!("../www/burn-image-icon.png");

/// Browser event emitted by the same system that updates the visible Bevy GPU status label.
///
/// The rendered-surface qualification harness listens before the Wasm module starts, so it can
/// distinguish a real shared-device-ready Bevy frame from a merely non-empty HTML canvas.
#[cfg(any(target_arch = "wasm32", test))]
pub(crate) const BROWSER_BACKEND_EVENT_NAME: &str = "burn-image-backend";

#[derive(Component)]
struct BackendStatusText;

pub struct BurnImageShellPlugin;

impl Plugin for BurnImageShellPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            crate::viewer::ImageViewerPlugin,
            crate::controls::ImageControlPanelPlugin,
        ))
        .add_systems(Startup, setup_shell)
        .add_systems(Update, update_backend_text);
        #[cfg(not(target_arch = "wasm32"))]
        app.add_systems(Update, set_native_window_icon);
    }
}

pub fn build_app() -> App {
    build_app_with_wgpu_settings(browser_safe_wgpu_settings())
}

fn build_app_with_wgpu_settings(wgpu_settings: WgpuSettings) -> App {
    let mut app = App::new();
    let plugins = DefaultPlugins
        .set(WindowPlugin {
            primary_window: Some(primary_window()),
            ..default()
        })
        .set(RenderPlugin {
            render_creation: RenderCreation::Automatic(Box::new(wgpu_settings)),
            ..default()
        });
    app.add_plugins(plugins)
        .add_plugins((BurnImageFrontendPlugin::default(), BurnImageShellPlugin));
    app
}

fn primary_window() -> Window {
    Window {
        title: APP_NAME.to_string(),
        name: Some(APP_WINDOW_ID.to_string()),
        resolution: (1280, 800).into(),
        canvas: Some("#burn-image".to_string()),
        fit_canvas_to_parent: true,
        ..default()
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn decode_app_icon() -> Result<(Vec<u8>, u32, u32), String> {
    let image = image::load_from_memory_with_format(APP_ICON_PNG, image::ImageFormat::Png)
        .map_err(|error| format!("embedded PNG could not be decoded: {error}"))?
        .into_rgba8();
    let (width, height) = image.dimensions();
    Ok((image.into_raw(), width, height))
}

#[cfg(not(target_arch = "wasm32"))]
fn set_native_window_icon(
    mut created_windows: MessageReader<bevy::window::WindowCreated>,
    primary_windows: Query<(), With<bevy::window::PrimaryWindow>>,
    // WINIT_WINDOWS is thread-local and winit requires window mutations on the main thread.
    _main_thread_marker: bevy::ecs::system::NonSendMarker,
) {
    for created in created_windows.read() {
        if primary_windows.get(created.window).is_err() {
            continue;
        }
        let (rgba, width, height) = match decode_app_icon() {
            Ok(decoded) => decoded,
            Err(error) => {
                warn!("failed to prepare the {APP_NAME} window icon: {error}");
                continue;
            }
        };
        let icon = match winit::window::Icon::from_rgba(rgba, width, height) {
            Ok(icon) => icon,
            Err(error) => {
                warn!("failed to prepare the {APP_NAME} window icon: {error}");
                continue;
            }
        };
        bevy::winit::WINIT_WINDOWS.with_borrow(|windows| {
            if let Some(window) = windows.get_window(created.window) {
                window.set_window_icon(Some(icon));
            }
        });
    }
}

fn browser_safe_wgpu_settings() -> WgpuSettings {
    browser_safe_wgpu_settings_for(cfg!(target_arch = "wasm32"))
}

fn browser_safe_wgpu_settings_for(browser_webgpu: bool) -> WgpuSettings {
    let mut settings = WgpuSettings::default();
    if browser_webgpu {
        // The model-neutral frontend requests only WebGPU's portable baseline. Model plugins that
        // require elevated limits must do so explicitly in their own app constructor.
        settings.priority = WgpuSettingsPriority::WebGPU;
        // Bevy defaults to TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES, which is not part of the
        // browser descriptor contract. Start from an exact empty WebGPU optional-feature set.
        settings.features = WgpuFeatures::empty();
    }
    settings
}

#[cfg(feature = "boogu")]
fn boogu_wgpu_settings() -> WgpuSettings {
    boogu_wgpu_settings_for(cfg!(target_arch = "wasm32"))
}

#[cfg(feature = "boogu")]
fn boogu_wgpu_settings_for(browser_webgpu: bool) -> WgpuSettings {
    let mut settings = browser_safe_wgpu_settings_for(browser_webgpu);
    if browser_webgpu {
        // The Boogu browser descriptor exposes the same 1K and 1.5K release shapes as native.
        // Request the maximum shape-aware single-buffer plan across those released presets; the
        // runtime verifies the limits actually applied to the shared Bevy/Burn device.
        settings.limits.max_storage_buffer_binding_size =
            crate::boogu::BOOGU_BROWSER_REQUESTED_BUFFER_LIMIT_BYTES;
        settings.limits.max_buffer_size = crate::boogu::BOOGU_BROWSER_REQUESTED_BUFFER_LIMIT_BYTES;
        #[cfg(feature = "boogu-web")]
        {
            // CubeCL cannot use its synchronous system-timing profiler in a browser. Requiring the
            // timestamp-query feature on the one Bevy/Burn device makes CubeCL select device
            // timing; an adapter without that production requirement fails during device creation.
            settings.features.insert(WgpuFeatures::TIMESTAMP_QUERY);
        }
    }
    settings
}

/// Build the native or browser shell with a real, injected Boogu runtime
/// factory. The factory is initialized only after the shared Burn WGPU device
/// has been attested.
#[cfg(feature = "boogu")]
pub fn build_boogu_app<F>(settings: crate::boogu::BooguAdapterSettings, factory: F) -> App
where
    F: crate::boogu::BooguRuntimeFactory,
{
    let mut app = build_app_with_wgpu_settings(boogu_wgpu_settings());
    app.add_plugins(crate::boogu::BooguAdapterPlugin::new(settings, factory));
    app
}

fn setup_shell(mut commands: Commands) {
    commands.spawn((
        Camera2d,
        Camera {
            order: 1,
            clear_color: ClearColorConfig::None,
            ..default()
        },
        RenderLayers::layer(0),
        IsDefaultUiCamera,
    ));
    commands.spawn((
        Text::new("GPU - connecting shared device"),
        TextFont {
            font_size: bevy::text::FontSize::Px(13.0),
            ..default()
        },
        TextColor(Color::srgb(0.68, 0.74, 0.82)),
        bevy::text::LineHeight::RelativeToFont(1.2),
        Node {
            position_type: PositionType::Absolute,
            left: px(16),
            top: px(16),
            ..default()
        },
        BackendStatusText,
    ));
}

fn update_backend_text(
    status: Res<BackendStatus>,
    mut labels: Query<&mut Text, With<BackendStatusText>>,
) {
    if !status.is_changed() {
        return;
    }
    let visible_value = backend_visible_status(&status.state);
    let event_value = backend_event_message(&status.state);
    dispatch_browser_backend_status(&status.state, &event_value);
    for mut text in &mut labels {
        **text = visible_value.clone();
    }
}

fn backend_visible_status(state: &BackendState) -> String {
    match state {
        BackendState::Initializing => "GPU - connecting shared device".into(),
        BackendState::Ready { device } => format!("GPU ready - {}", device.backend),
        BackendState::Failed { .. } => "GPU unavailable - see runtime status".into(),
    }
}

fn backend_event_message(state: &BackendState) -> String {
    match state {
        BackendState::Initializing => "GPU: initializing shared Bevy/Burn WGPU device".into(),
        BackendState::Ready { device } => format!(
            "GPU: {} ({}, {}) - shared device ready",
            device.adapter_name, device.backend, device.device_type
        ),
        BackendState::Failed { reason } => format!("GPU unavailable: {reason}"),
    }
}

#[cfg(target_arch = "wasm32")]
fn dispatch_browser_backend_status(state: &BackendState, visible_status: &str) {
    let result = (|| {
        let detail = js_sys::Object::new();
        let set = |name: &str, value: wasm_bindgen::JsValue| {
            js_sys::Reflect::set(&detail, &name.into(), &value)
                .map(|_| ())
                .map_err(|error| format!("{error:?}"))
        };
        set("event", backend_event_phase(state).into())?;
        set("message", visible_status.into())?;
        if let BackendState::Ready { device } = state {
            set("adapter_name", device.adapter_name.as_str().into())?;
            set("backend", device.backend.as_str().into())?;
            set("device_type", device.device_type.as_str().into())?;
            set(
                "shared_adapter_device_queue",
                device.shared_adapter_device_queue.into(),
            )?;
        }

        let init = web_sys::CustomEventInit::new();
        init.set_detail(detail.as_ref());
        let event =
            web_sys::CustomEvent::new_with_event_init_dict(BROWSER_BACKEND_EVENT_NAME, &init)
                .map_err(|error| format!("{error:?}"))?;
        let window = web_sys::window().ok_or_else(|| "Window is unavailable".to_owned())?;
        window
            .dispatch_event(event.as_ref())
            .map_err(|error| format!("{error:?}"))?;
        Ok::<(), String>(())
    })();
    if let Err(error) = result {
        web_sys::console::warn_1(
            &format!("failed to dispatch browser event {BROWSER_BACKEND_EVENT_NAME}: {error}")
                .into(),
        );
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn dispatch_browser_backend_status(_state: &BackendState, _visible_status: &str) {}

#[cfg(any(target_arch = "wasm32", test))]
fn backend_event_phase(state: &BackendState) -> &'static str {
    match state {
        BackendState::Initializing => "initializing",
        BackendState::Ready { .. } => "ready",
        BackendState::Failed { .. } => "failed",
    }
}

#[cfg(test)]
mod tests {
    use super::{
        APP_WINDOW_ID, BROWSER_BACKEND_EVENT_NAME, backend_event_phase,
        browser_safe_wgpu_settings_for, primary_window,
    };
    use crate::{BackendDeviceInfo, BackendFailure, BackendState};
    use bevy::render::settings::WgpuFeatures;

    #[test]
    fn generic_browser_shell_requests_exactly_no_optional_features_correctness() {
        assert_eq!(
            browser_safe_wgpu_settings_for(true).features,
            WgpuFeatures::empty()
        );
    }

    #[test]
    fn primary_window_has_stable_native_identity_correctness() {
        let window = primary_window();
        assert_eq!(window.title, crate::APP_NAME);
        assert_eq!(window.name.as_deref(), Some(APP_WINDOW_ID));
        assert_eq!(window.canvas.as_deref(), Some("#burn-image"));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn embedded_window_icon_is_valid_rgba_correctness() {
        let (rgba, width, height) = super::decode_app_icon().unwrap();
        assert_eq!((width, height), (512, 512));
        assert_eq!(rgba.len(), width as usize * height as usize * 4);
        assert!(winit::window::Icon::from_rgba(rgba, width, height).is_ok());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn native_window_icon_system_is_main_thread_pinned_correctness() {
        use bevy::ecs::system::{IntoSystem, System};

        let mut world = bevy::prelude::World::new();
        let mut system = IntoSystem::into_system(super::set_native_window_icon);
        system.initialize(&mut world);
        assert!(!system.is_send());
    }

    #[cfg(feature = "boogu-web")]
    #[test]
    fn boogu_browser_shell_requests_exactly_timestamp_query_correctness() {
        assert_eq!(
            super::boogu_wgpu_settings_for(true).features,
            WgpuFeatures::TIMESTAMP_QUERY
        );
    }

    #[test]
    fn browser_backend_event_phases_match_rendered_smoke_harness_correctness() {
        let ready = BackendState::Ready {
            device: BackendDeviceInfo {
                adapter_name: "NVIDIA test adapter".into(),
                backend: "BrowserWebGpu".into(),
                device_type: "DiscreteGpu".into(),
                driver: "test driver".into(),
                max_storage_buffer_binding_size: 1,
                max_buffer_size: 1,
                shared_adapter_device_queue: true,
            },
        };
        assert_eq!(
            backend_event_phase(&BackendState::Initializing),
            "initializing"
        );
        assert_eq!(backend_event_phase(&ready), "ready");
        assert_eq!(
            backend_event_phase(&BackendState::Failed {
                reason: BackendFailure::BurnDeviceLost,
            }),
            "failed"
        );

        let contract = include_str!("../tests/wasm_rendered_surface_contract.mjs");
        assert!(contract.contains(BROWSER_BACKEND_EVENT_NAME));
        assert!(contract.contains("shared_adapter_device_queue"));
        assert!(BROWSER_BACKEND_EVENT_NAME.is_ascii());
    }

    #[test]
    fn visible_backend_status_stays_concise_while_event_keeps_identity_correctness() {
        let ready = BackendState::Ready {
            device: BackendDeviceInfo {
                adapter_name: "NVIDIA RTX PRO 6000 Blackwell Workstation Edition".into(),
                backend: "Vulkan".into(),
                device_type: "DiscreteGpu".into(),
                driver: "test driver".into(),
                max_storage_buffer_binding_size: 1,
                max_buffer_size: 1,
                shared_adapter_device_queue: true,
            },
        };
        assert_eq!(super::backend_visible_status(&ready), "GPU ready - Vulkan");
        let event = super::backend_event_message(&ready);
        assert!(event.contains("NVIDIA RTX PRO 6000 Blackwell Workstation Edition"));
        assert!(event.contains("shared device ready"));
    }
}
