//! Small native/web shell. Model crates add their own dispatch systems and UI
//! controls while reusing the status and image-view surfaces here.

use bevy::{
    prelude::*,
    render::{
        RenderPlugin,
        settings::{RenderCreation, WgpuSettings, WgpuSettingsPriority},
    },
    window::WindowPlugin,
};

use crate::{APP_NAME, BackendState, BackendStatus, BurnImageFrontendPlugin};

#[derive(Component)]
struct BackendStatusText;

pub struct BurnImageShellPlugin;

impl Plugin for BurnImageShellPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(crate::controls::ImageControlPanelPlugin)
            .add_systems(Startup, setup_shell)
            .add_systems(Update, update_backend_text);
    }
}

pub fn build_app() -> App {
    build_app_with_wgpu_settings(browser_safe_wgpu_settings())
}

fn build_app_with_wgpu_settings(wgpu_settings: WgpuSettings) -> App {
    let mut app = App::new();
    let plugins = DefaultPlugins
        .set(WindowPlugin {
            primary_window: Some(Window {
                title: APP_NAME.to_string(),
                resolution: (1280, 800).into(),
                canvas: Some("#burn-image".to_string()),
                fit_canvas_to_parent: true,
                ..default()
            }),
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

fn browser_safe_wgpu_settings() -> WgpuSettings {
    let mut settings = WgpuSettings::default();
    if cfg!(target_arch = "wasm32") {
        // The model-neutral frontend requests only WebGPU's portable baseline. Model plugins that
        // require elevated limits must do so explicitly in their own app constructor.
        settings.priority = WgpuSettingsPriority::WebGPU;
    }
    settings
}

#[cfg(feature = "boogu")]
fn boogu_wgpu_settings() -> WgpuSettings {
    let mut settings = browser_safe_wgpu_settings();
    if cfg!(target_arch = "wasm32") {
        settings.limits.max_storage_buffer_binding_size =
            crate::boogu::BOOGU_BROWSER_REQUESTED_BUFFER_LIMIT_BYTES;
        settings.limits.max_buffer_size = crate::boogu::BOOGU_BROWSER_REQUESTED_BUFFER_LIMIT_BYTES;
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
    commands.spawn(Camera2d);
    commands.spawn((
        Text::new("GPU: initializing shared Bevy/Burn WGPU device"),
        TextFont {
            font_size: bevy::text::FontSize::Px(18.0),
            ..default()
        },
        Node {
            position_type: PositionType::Absolute,
            left: px(18),
            top: px(14),
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
    let value = match &status.state {
        BackendState::Initializing => "GPU: initializing shared Bevy/Burn WGPU device".to_string(),
        BackendState::Ready { device } => format!(
            "GPU: {} ({}, {}) — shared device ready",
            device.adapter_name, device.backend, device.device_type
        ),
        BackendState::Failed { reason } => format!("GPU unavailable: {reason}"),
    };
    for mut text in &mut labels {
        **text = value.clone();
    }
}
