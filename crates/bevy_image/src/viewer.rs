//! Camera-backed image viewing for native and browser shells.

use bevy::{
    camera::{Viewport, visibility::RenderLayers},
    input::{gestures::PinchGesture, mouse::MouseWheel},
    input_focus::InputFocus,
    prelude::*,
    text::EditableText,
    window::PrimaryWindow,
};
use bevy_pancam::{DirectionKeys, PanCam, PanCamPlugin, PanCamSystems};
use burn_image::{Dimensions, HostImage, InputImage};

use crate::{
    ImageBytesLoaded, ImageDisplayReady, ImageFrontendSet, ImageIoId, LatestGeneratedImageView,
    controls::image_control_panel_layout, host_image_to_bevy_image,
};

const IMAGE_RENDER_LAYER: usize = 1;
const FIT_PADDING: f32 = 1.05;
const MIN_VIEW_SCALE: f32 = 0.01;
const MAX_VIEW_SCALE: f32 = 256.0;

/// Stable I/O identity used for the edit-reference picker and image preview.
pub const REFERENCE_IMAGE_IO_ID: ImageIoId = ImageIoId(1);

/// Center the current image and contain it inside the viewer viewport.
#[derive(Message, Clone, Copy, Debug, Default)]
pub struct FitImageView;

/// Center the current image at one image pixel per logical display pixel.
#[derive(Message, Clone, Copy, Debug, Default)]
pub struct ActualSizeImageView;

#[derive(Component)]
pub struct ImageViewCamera;

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
struct ImageViewState {
    dimensions: Option<Dimensions>,
    mode: ImageViewMode,
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
struct ImageViewportState {
    logical_size: Option<Vec2>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum ImageViewMode {
    #[default]
    Fit,
    ActualSize,
    Manual,
}

#[derive(Resource, Default)]
struct ImageViewPointerCapture(bool);

#[derive(Resource, Default)]
struct ReferencePreviewImage {
    handle: Option<Handle<Image>>,
}

/// Installs a responsive world-space sprite viewer driven by `bevy_pancam`.
pub struct ImageViewerPlugin;

impl Plugin for ImageViewerPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(PanCamPlugin)
            .init_resource::<ImageViewState>()
            .init_resource::<ImageViewportState>()
            .init_resource::<ImageViewPointerCapture>()
            .init_resource::<ReferencePreviewImage>()
            .add_message::<FitImageView>()
            .add_message::<ActualSizeImageView>()
            .add_systems(Startup, setup_image_viewer)
            .add_systems(
                Update,
                (
                    sync_image_viewport,
                    preview_reference_image,
                    capture_generated_image,
                    receive_view_requests,
                    apply_view_mode,
                    gate_image_camera_input,
                )
                    .chain()
                    .after(ImageFrontendSet::Display)
                    .before(PanCamSystems),
            );
    }
}

fn setup_image_viewer(mut commands: Commands) {
    commands.spawn((
        Camera2d,
        Camera {
            order: 0,
            ..default()
        },
        RenderLayers::layer(IMAGE_RENDER_LAYER),
        image_pan_cam(),
        ImageViewCamera,
    ));
    commands.spawn((
        Sprite::default(),
        Visibility::Hidden,
        RenderLayers::layer(IMAGE_RENDER_LAYER),
        LatestGeneratedImageView,
    ));
}

fn image_pan_cam() -> PanCam {
    PanCam {
        grab_buttons: vec![MouseButton::Left, MouseButton::Middle],
        move_keys: DirectionKeys::NONE,
        enabled: false,
        zoom_to_cursor: true,
        min_scale: MIN_VIEW_SCALE,
        max_scale: MAX_VIEW_SCALE,
        ..default()
    }
}

fn sync_image_viewport(
    windows: Query<&Window, With<PrimaryWindow>>,
    mut cameras: Query<&mut Camera, With<ImageViewCamera>>,
    mut viewport_state: ResMut<ImageViewportState>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let scale_factor = window.scale_factor().max(f32::EPSILON);
    let viewport = image_viewport(window.physical_size(), scale_factor);
    let logical_size = viewport
        .as_ref()
        .map(|viewport| viewport.physical_size.as_vec2() / scale_factor);
    for mut camera in &mut cameras {
        if !viewports_equal(camera.viewport.as_ref(), viewport.as_ref()) {
            camera.viewport = viewport.clone();
        }
    }
    if viewport_state.logical_size != logical_size {
        viewport_state.logical_size = logical_size;
    }
}

fn viewports_equal(left: Option<&Viewport>, right: Option<&Viewport>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => {
            left.physical_position == right.physical_position
                && left.physical_size == right.physical_size
                && left.depth.start == right.depth.start
                && left.depth.end == right.depth.end
        }
        _ => false,
    }
}

fn image_viewport(physical_size: UVec2, scale_factor: f32) -> Option<Viewport> {
    if physical_size.x == 0 || physical_size.y == 0 {
        return None;
    }
    let scale_factor = scale_factor.max(f32::EPSILON);
    let logical_size = physical_size.as_vec2() / scale_factor;
    let layout = image_control_panel_layout(logical_size);
    let position = Vec2::new(layout.viewer_left, layout.viewer_top) * scale_factor;
    let size = Vec2::new(layout.viewer_width, layout.viewer_height) * scale_factor;
    let physical_position = position.round().as_uvec2().min(physical_size - UVec2::ONE);
    let available = physical_size - physical_position;
    let physical_size = size.round().as_uvec2().max(UVec2::ONE).min(available);
    Some(Viewport {
        physical_position,
        physical_size,
        depth: 0.0..1.0,
    })
}

fn preview_reference_image(
    mut loaded: MessageReader<ImageBytesLoaded>,
    mut assets: ResMut<Assets<Image>>,
    mut preview: ResMut<ReferencePreviewImage>,
    mut sprites: Query<(&mut Sprite, &mut Visibility), With<LatestGeneratedImageView>>,
    mut state: ResMut<ImageViewState>,
) {
    for loaded in loaded.read() {
        if loaded.id != REFERENCE_IMAGE_IO_ID {
            continue;
        }
        let host = match &loaded.image {
            InputImage::Pixels(pixels) => HostImage::Pixels(pixels.clone()),
            InputImage::Encoded(encoded) => HostImage::Encoded(encoded.clone()),
        };
        let Ok((dimensions, image)) = host_image_to_bevy_image(&host) else {
            continue;
        };
        let handle = replace_reference_preview_image(&mut assets, &mut preview, image);
        for (mut sprite, mut visibility) in &mut sprites {
            sprite.image = handle.clone();
            sprite.custom_size = Some(image_size(dimensions));
            *visibility = Visibility::Visible;
        }
        state.dimensions = Some(dimensions);
        state.mode = ImageViewMode::Fit;
    }
}

fn replace_reference_preview_image(
    assets: &mut Assets<Image>,
    preview: &mut ReferencePreviewImage,
    image: Image,
) -> Handle<Image> {
    if let Some(previous) = preview.handle.take() {
        assets.remove(previous.id());
    }
    let handle = assets.add(image);
    preview.handle = Some(handle.clone());
    handle
}

fn capture_generated_image(
    mut ready: MessageReader<ImageDisplayReady>,
    mut state: ResMut<ImageViewState>,
) {
    for image in ready.read() {
        state.dimensions = Some(image.dimensions);
        state.mode = ImageViewMode::Fit;
    }
}

fn receive_view_requests(
    mut fit: MessageReader<FitImageView>,
    mut actual_size: MessageReader<ActualSizeImageView>,
    mut state: ResMut<ImageViewState>,
) {
    if fit.read().next().is_some() {
        state.mode = ImageViewMode::Fit;
    }
    if actual_size.read().next().is_some() {
        state.mode = ImageViewMode::ActualSize;
    }
}

fn apply_view_mode(
    state: Res<ImageViewState>,
    viewport: Res<ImageViewportState>,
    mut projections: Query<(&mut Projection, &mut Transform), With<ImageViewCamera>>,
) {
    if !state.is_changed() && !viewport.is_changed() {
        return;
    }
    let Some(dimensions) = state.dimensions else {
        return;
    };
    let Some(viewport_size) = viewport.logical_size else {
        return;
    };
    let scale = match state.mode {
        ImageViewMode::Fit => fit_scale(dimensions, viewport_size),
        ImageViewMode::ActualSize => 1.0,
        ImageViewMode::Manual => return,
    };
    for (mut projection, mut transform) in &mut projections {
        if let Projection::Orthographic(projection) = &mut *projection {
            projection.scale = scale.clamp(MIN_VIEW_SCALE, MAX_VIEW_SCALE);
            transform.translation.x = 0.0;
            transform.translation.y = 0.0;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn gate_image_camera_input(
    windows: Query<&Window, With<PrimaryWindow>>,
    mouse: Res<ButtonInput<MouseButton>>,
    focus: Res<InputFocus>,
    editable: Query<(), With<EditableText>>,
    mut wheel: MessageReader<MouseWheel>,
    mut pinch: MessageReader<PinchGesture>,
    mut capture: ResMut<ImageViewPointerCapture>,
    mut state: ResMut<ImageViewState>,
    mut cameras: Query<(&Camera, &mut PanCam), With<ImageViewCamera>>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let cursor = window.cursor_position();
    let any_pressed = mouse.pressed(MouseButton::Left) || mouse.pressed(MouseButton::Middle);
    let just_pressed =
        mouse.just_pressed(MouseButton::Left) || mouse.just_pressed(MouseButton::Middle);
    let text_focused = focus.get().is_some_and(|entity| editable.contains(entity));
    let had_wheel = wheel.read().next().is_some();
    let had_pinch = pinch.read().next().is_some();

    for (camera, mut pan_cam) in &mut cameras {
        let cursor_over = cursor
            .zip(camera.logical_viewport_rect())
            .is_some_and(|(cursor, viewport)| viewport.contains(cursor));
        let next_capture = if just_pressed {
            cursor_over && !text_focused
        } else if !any_pressed {
            false
        } else {
            capture.0
        };
        if capture.0 != next_capture {
            capture.0 = next_capture;
        }
        let enabled = !text_focused && (cursor_over || capture.0);
        if pan_cam.enabled != enabled {
            pan_cam.enabled = enabled;
        }
        if enabled
            && (had_wheel || had_pinch || (capture.0 && any_pressed))
            && state.mode != ImageViewMode::Manual
        {
            state.mode = ImageViewMode::Manual;
        }
    }
}

fn image_size(dimensions: Dimensions) -> Vec2 {
    Vec2::new(dimensions.width() as f32, dimensions.height() as f32)
}

fn fit_scale(dimensions: Dimensions, viewport_size: Vec2) -> f32 {
    let image = image_size(dimensions);
    (image.x / viewport_size.x.max(1.0)).max(image.y / viewport_size.y.max(1.0)) * FIT_PADDING
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_fit_contains_landscape_and_portrait_correctness() {
        let viewport = Vec2::new(800.0, 600.0);
        let landscape = fit_scale(Dimensions::new(1600, 400).unwrap(), viewport);
        let portrait = fit_scale(Dimensions::new(400, 1600).unwrap(), viewport);
        assert!((landscape - 2.1).abs() < 1.0e-6);
        assert!((portrait - 2.8).abs() < 1.0e-6);
    }

    #[test]
    fn viewer_camera_has_no_keyboard_movement_correctness() {
        let pan_cam = image_pan_cam();
        assert_eq!(pan_cam.move_keys, DirectionKeys::NONE);
        assert_eq!(
            pan_cam.grab_buttons,
            [MouseButton::Left, MouseButton::Middle]
        );
        assert!(!pan_cam.enabled);
    }

    #[test]
    fn physical_viewport_respects_panel_and_scale_factor_correctness() {
        let viewport = image_viewport(UVec2::new(2560, 1600), 2.0).unwrap();
        assert_eq!(viewport.physical_position, UVec2::new(768, 104));
        assert_eq!(viewport.physical_size, UVec2::new(1768, 1472));
    }

    #[test]
    fn viewport_equality_tracks_only_render_geometry_correctness() {
        let original = image_viewport(UVec2::new(2560, 1600), 2.0).unwrap();
        let mut resized = original.clone();
        resized.physical_size.x += 1;
        assert!(viewports_equal(Some(&original), Some(&original)));
        assert!(!viewports_equal(Some(&original), Some(&resized)));
        assert!(!viewports_equal(Some(&original), None));
        assert!(viewports_equal(None, None));
    }

    #[test]
    fn replacing_reference_preview_evicts_only_the_previous_preview_correctness() {
        let mut assets = Assets::<Image>::default();
        let generated = assets.add(Image::default());
        let mut preview = ReferencePreviewImage::default();

        let first = replace_reference_preview_image(&mut assets, &mut preview, Image::default());
        let first_id = first.id();
        let second = replace_reference_preview_image(&mut assets, &mut preview, Image::default());

        assert!(assets.get(generated.id()).is_some());
        assert!(assets.get(first_id).is_none());
        assert!(assets.get(second.id()).is_some());
        assert_eq!(preview.handle.as_ref().unwrap().id(), second.id());
        assert_eq!(assets.len(), 2);
    }
}
