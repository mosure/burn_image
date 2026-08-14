//! Usable model-neutral Bevy controls for generation and editing.

#[cfg(target_arch = "wasm32")]
use bevy::input_focus::InputFocus;
use bevy::{
    input::mouse::{MouseScrollUnit, MouseWheel},
    prelude::*,
    text::{EditableText, EditableTextFilter, TextCursorStyle},
    ui::InteractionDisabled,
    window::PrimaryWindow,
};
#[cfg(test)]
use burn_image::DimensionConstraints;
use burn_image::{
    Dimensions, HostImage, ImageEncoding, ImageTaskKind, ModelDescriptor, ProgressEvent,
};

use crate::{
    ActualSizeImageView, CancelImageJob, CompleteImageJob, EditorMode, FitImageView,
    ImageBytesLoaded, ImageDisplayFailed, ImageEditorState, ImageFrontendSet, ImageIoFailed,
    ImageJobId, ImageJobPhase, ImageJobRejected, ImageJobs, ImageRunnerState, ImageRunnerStatus,
    LoadImageBytes, PrepareImageDownload, REFERENCE_IMAGE_IO_ID,
};

#[cfg(any(target_arch = "wasm32", not(feature = "native-io")))]
use crate::ImageIoId;

#[cfg(all(feature = "native-io", not(target_arch = "wasm32")))]
use crate::{
    ImageFileDialogCancelled, ImageFileDialogFailed, ImageFileDialogSelected, ImageFileDialogState,
    OpenImageFileDialog, ReadDroppedImageFile, sanitize_image_file_name,
};

#[cfg(target_arch = "wasm32")]
use crate::ImageDownloadReady;

const MAX_REFERENCE_BYTES: usize = 64 * 1024 * 1024;
#[cfg(any(target_arch = "wasm32", not(feature = "native-io")))]
const DOWNLOAD_IO_ID: ImageIoId = ImageIoId(2);
const DESKTOP_PANEL_WIDTH: f32 = 360.0;
const DESKTOP_LAYOUT_MIN_WIDTH: f32 = 820.0;
const NARROW_PANEL_HEIGHT_RATIO: f32 = 0.48;
const NARROW_PANEL_MIN_HEIGHT: f32 = 260.0;
const NARROW_PANEL_MAX_HEIGHT: f32 = 430.0;
const MIN_VIEWER_HEIGHT: f32 = 160.0;
const PANEL_MARGIN: f32 = 12.0;
const PANEL_TOP: f32 = 52.0;
const SIZE_PRESETS: &[(u32, u32)] = &[
    (256, 256),
    (512, 512),
    (768, 768),
    (1024, 1024),
    (1024, 768),
    (768, 1024),
    // Keep this suffix equal to burn_boogu::BOOGU_1K5_OUTPUT_PRESETS. Capability filtering keeps
    // these out of the 1K and exact-256 browser controls.
    (1536, 1536),
    (1264, 1856),
    (1856, 1264),
    (1344, 1744),
    (1744, 1344),
    (1392, 1696),
    (1696, 1392),
    (1152, 2032),
    (2032, 1152),
    (2368, 992),
];
const DEFAULT_SIZE_PRESET: (u32, u32) = (512, 512);
#[cfg(any(target_arch = "wasm32", test))]
const BROWSER_UI_CONTRACT_EVENT_NAME: &str = "burn-image-ui-contract";

#[derive(Resource)]
pub struct ImageControlPanelState {
    pub latest_job: Option<ImageJobId>,
    pub latest_output: Option<(ImageJobId, HostImage)>,
    pub notice: String,
    size_index: usize,
    seed_valid: bool,
}

impl Default for ImageControlPanelState {
    fn default() -> Self {
        Self {
            latest_job: None,
            latest_output: None,
            notice: String::new(),
            size_index: SIZE_PRESETS
                .iter()
                .position(|preset| *preset == DEFAULT_SIZE_PRESET)
                .expect("default size preset must be listed"),
            // The seed field starts at `0`; it is valid before the first
            // EditableText change event is observed.
            seed_valid: true,
        }
    }
}

#[derive(Component, Default)]
struct ModeButton;
#[derive(Component, Default)]
struct ModelButton;
#[derive(Component, Default)]
struct SizeButton;
#[derive(Component, Default)]
struct ReferenceButton;
#[derive(Component, Default)]
struct RunButton;
#[derive(Component, Default)]
struct CancelButton;
#[derive(Component, Default)]
struct SaveButton;
#[derive(Component, Default)]
struct FitButton;
#[derive(Component, Default)]
struct ActualSizeButton;
#[derive(Component)]
struct PromptInput;
#[derive(Component)]
struct SeedInput;
#[derive(Component, Default)]
struct ModeButtonLabel;
#[derive(Component, Default)]
struct ModelButtonLabel;
#[derive(Component, Default)]
struct SizeButtonLabel;
#[derive(Component)]
struct ReferenceLabel;
#[derive(Component)]
struct ProgressLabel;
#[derive(Component)]
struct ProgressDetailLabel;
#[derive(Component)]
struct ProgressFill;

#[derive(Component)]
pub(crate) struct ImageControlPanel;

#[derive(Component)]
struct ImageControlPanelScroll;

#[derive(Component, Clone, Copy)]
struct ButtonPalette {
    idle: Color,
    pressed: Color,
    disabled: Color,
}

impl ButtonPalette {
    const fn neutral() -> Self {
        Self {
            idle: Color::srgb(0.14, 0.18, 0.27),
            pressed: Color::srgb(0.24, 0.32, 0.48),
            disabled: Color::srgb(0.085, 0.095, 0.12),
        }
    }

    const fn action(idle: Color) -> Self {
        Self {
            idle,
            pressed: Color::srgb(0.24, 0.32, 0.48),
            disabled: Color::srgb(0.085, 0.095, 0.12),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ImageControlPanelLayout {
    pub(crate) narrow: bool,
    pub(crate) panel_width: f32,
    pub(crate) panel_height: f32,
    pub(crate) viewer_left: f32,
    pub(crate) viewer_top: f32,
    pub(crate) viewer_width: f32,
    pub(crate) viewer_height: f32,
}

pub(crate) fn image_control_panel_layout(logical_size: Vec2) -> ImageControlPanelLayout {
    let width = logical_size.x.max(1.0);
    let height = logical_size.y.max(1.0);
    if width >= DESKTOP_LAYOUT_MIN_WIDTH {
        ImageControlPanelLayout {
            narrow: false,
            panel_width: DESKTOP_PANEL_WIDTH,
            panel_height: (height - PANEL_TOP - PANEL_MARGIN).max(1.0),
            viewer_left: DESKTOP_PANEL_WIDTH + 2.0 * PANEL_MARGIN,
            viewer_top: PANEL_TOP,
            viewer_width: (width - DESKTOP_PANEL_WIDTH - 3.0 * PANEL_MARGIN).max(1.0),
            viewer_height: (height - PANEL_TOP - PANEL_MARGIN).max(1.0),
        }
    } else {
        let available_height = (height - PANEL_TOP - 2.0 * PANEL_MARGIN).max(1.0);
        let desired_panel_height = (height * NARROW_PANEL_HEIGHT_RATIO).clamp(
            NARROW_PANEL_MIN_HEIGHT.min(available_height),
            NARROW_PANEL_MAX_HEIGHT.min(available_height),
        );
        let panel_height =
            desired_panel_height.min((available_height - MIN_VIEWER_HEIGHT).max(1.0));
        ImageControlPanelLayout {
            narrow: true,
            panel_width: (width - 2.0 * PANEL_MARGIN).max(1.0),
            panel_height,
            viewer_left: PANEL_MARGIN,
            viewer_top: PANEL_TOP,
            viewer_width: (width - 2.0 * PANEL_MARGIN).max(1.0),
            viewer_height: (available_height - panel_height).max(1.0),
        }
    }
}

pub struct ImageControlPanelPlugin;

impl Plugin for ImageControlPanelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ImageControlPanelState>()
            .add_systems(Startup, setup_controls)
            .add_systems(
                Update,
                (
                    sync_control_panel_layout,
                    scroll_control_panel,
                    select_initial_model,
                    sync_text_inputs,
                    handle_mode_button,
                    handle_model_button,
                    handle_size_button,
                    handle_reference_button,
                    accept_native_file_dialog,
                    handle_view_buttons,
                    handle_run_button,
                    handle_cancel_button,
                    handle_save_button,
                    accept_reference_images,
                    capture_frontend_errors,
                    update_control_labels,
                    update_progress_panel,
                    update_action_availability,
                    update_button_colors,
                )
                    .chain(),
            )
            .add_systems(Update, capture_outputs.after(ImageFrontendSet::Feedback));

        #[cfg(all(feature = "native-io", not(target_arch = "wasm32")))]
        app.add_systems(
            Update,
            accept_native_file_drop.before(accept_reference_images),
        );

        #[cfg(target_arch = "wasm32")]
        app.add_systems(
            Update,
            (
                drain_browser_reference_queue,
                complete_browser_download,
                dispatch_browser_ui_contract.after(update_action_availability),
            ),
        );
    }
}

fn setup_controls(mut commands: Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(PANEL_MARGIN),
                top: px(PANEL_TOP),
                bottom: px(PANEL_MARGIN),
                width: px(DESKTOP_PANEL_WIDTH),
                padding: px(14).all(),
                row_gap: px(9),
                flex_direction: FlexDirection::Column,
                overflow: Overflow::scroll_y(),
                scrollbar_width: 7.0,
                border_radius: BorderRadius::all(px(8)),
                ..default()
            },
            ScrollPosition(Vec2::ZERO),
            BackgroundColor(Color::srgba(0.055, 0.065, 0.085, 0.96)),
            ImageControlPanel,
            ImageControlPanelScroll,
        ))
        .with_children(|panel| {
            panel.spawn((
                Text::new("IMAGE GENERATION AND EDITING"),
                TextFont::from_font_size(18.0),
                TextColor(Color::srgb(0.78, 0.86, 1.0)),
            ));

            spawn_labeled_button::<ModeButton, ModeButtonLabel>(panel, "Mode", "Generate");
            spawn_labeled_button::<ModelButton, ModelButtonLabel>(panel, "Model", "waiting...");

            panel.spawn((
                Text::new("Prompt / instruction"),
                TextFont::from_font_size(13.0),
                TextColor(Color::srgb(0.7, 0.74, 0.8)),
            ));
            panel.spawn((
                EditableText {
                    visible_width: Some(35.0),
                    visible_lines: Some(5.0),
                    allow_newlines: true,
                    max_characters: Some(8_192),
                    ..default()
                },
                PromptInput,
                TextFont::from_font_size(15.0),
                TextColor(Color::WHITE),
                TextCursorStyle::default(),
                TextLayout {
                    linebreak: LineBreak::WordOrCharacter,
                    ..default()
                },
                Node {
                    width: percent(100),
                    min_height: px(105),
                    padding: px(8).all(),
                    border: px(1).all(),
                    ..default()
                },
                BorderColor::all(Color::srgb(0.26, 0.3, 0.4)),
                BackgroundColor(Color::srgb(0.09, 0.105, 0.14)),
            ));

            spawn_labeled_button::<SizeButton, SizeButtonLabel>(panel, "Size", "model default");

            panel
                .spawn(Node {
                    width: percent(100),
                    column_gap: px(8),
                    align_items: AlignItems::Center,
                    ..default()
                })
                .with_children(|row| {
                    row.spawn((
                        Text::new("Seed"),
                        TextFont::from_font_size(13.0),
                        TextColor(Color::srgb(0.7, 0.74, 0.8)),
                        Node {
                            width: px(74),
                            ..default()
                        },
                    ));
                    row.spawn((
                        EditableText::new("0"),
                        EditableTextFilter::new(|character| character.is_ascii_digit()),
                        SeedInput,
                        TextFont::from_font_size(15.0),
                        TextColor(Color::WHITE),
                        TextCursorStyle::default(),
                        TextLayout::no_wrap(),
                        Node {
                            flex_grow: 1.0,
                            min_height: px(34),
                            padding: px(7).all(),
                            border: px(1).all(),
                            ..default()
                        },
                        BorderColor::all(Color::srgb(0.26, 0.3, 0.4)),
                        BackgroundColor(Color::srgb(0.09, 0.105, 0.14)),
                    ));
                });

            panel
                .spawn((
                    Button,
                    ReferenceButton,
                    control_button_node(),
                    ButtonPalette::neutral(),
                    BackgroundColor(ButtonPalette::neutral().idle),
                ))
                .with_children(|button| {
                    button.spawn((
                        Text::new(reference_button_text()),
                        TextFont::from_font_size(14.0),
                        TextColor(Color::WHITE),
                        ReferenceLabel,
                    ));
                });

            panel
                .spawn(Node {
                    width: percent(100),
                    column_gap: px(7),
                    ..default()
                })
                .with_children(|row| {
                    spawn_action_button::<RunButton>(row, "Run", Color::srgb(0.15, 0.42, 0.75));
                    spawn_action_button::<CancelButton>(
                        row,
                        "Cancel",
                        Color::srgb(0.54, 0.2, 0.22),
                    );
                    spawn_action_button::<SaveButton>(
                        row,
                        "Save PNG",
                        Color::srgb(0.18, 0.42, 0.3),
                    );
                });

            panel
                .spawn(Node {
                    width: percent(100),
                    column_gap: px(7),
                    ..default()
                })
                .with_children(|row| {
                    spawn_action_button::<FitButton>(
                        row,
                        "Fit image",
                        Color::srgb(0.18, 0.25, 0.36),
                    );
                    spawn_action_button::<ActualSizeButton>(
                        row,
                        "100%",
                        Color::srgb(0.18, 0.25, 0.36),
                    );
                });

            panel.spawn((
                Text::new("Preparing model runtime"),
                TextFont::from_font_size(13.0),
                TextColor(Color::srgb(0.84, 0.88, 0.94)),
                TextLayout {
                    linebreak: LineBreak::WordOrCharacter,
                    ..default()
                },
                ProgressLabel,
            ));

            panel
                .spawn((
                    Node {
                        position_type: PositionType::Relative,
                        width: percent(100),
                        height: px(7),
                        overflow: Overflow::clip(),
                        border_radius: BorderRadius::all(px(4)),
                        ..default()
                    },
                    BackgroundColor(Color::srgb(0.13, 0.15, 0.19)),
                ))
                .with_children(|track| {
                    track.spawn((
                        Node {
                            position_type: PositionType::Absolute,
                            left: px(0),
                            top: px(0),
                            width: percent(28),
                            height: percent(100),
                            border_radius: BorderRadius::all(px(4)),
                            ..default()
                        },
                        BackgroundColor(Color::srgb(0.32, 0.68, 0.83)),
                        ProgressFill,
                    ));
                });

            panel.spawn((
                Text::new("Waiting for the shared GPU"),
                TextFont::from_font_size(11.0),
                TextColor(Color::srgb(0.58, 0.64, 0.72)),
                TextLayout {
                    linebreak: LineBreak::WordOrCharacter,
                    ..default()
                },
                ProgressDetailLabel,
            ));
        });
}

fn sync_control_panel_layout(
    windows: Query<&Window, With<PrimaryWindow>>,
    mut panels: Query<&mut Node, With<ImageControlPanel>>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let layout = image_control_panel_layout(window.size());
    for mut node in &mut panels {
        let (right, top, width, height) = if layout.narrow {
            (
                px(PANEL_MARGIN),
                Val::Auto,
                Val::Auto,
                px(layout.panel_height),
            )
        } else {
            (Val::Auto, px(PANEL_TOP), px(layout.panel_width), Val::Auto)
        };
        if node.left != px(PANEL_MARGIN)
            || node.bottom != px(PANEL_MARGIN)
            || node.right != right
            || node.top != top
            || node.width != width
            || node.height != height
        {
            node.left = px(PANEL_MARGIN);
            node.bottom = px(PANEL_MARGIN);
            node.right = right;
            node.top = top;
            node.width = width;
            node.height = height;
        }
    }
}

fn scroll_control_panel(
    windows: Query<&Window, With<PrimaryWindow>>,
    mut wheel: MessageReader<MouseWheel>,
    panels: Query<(&ComputedNode, &UiGlobalTransform), With<ImageControlPanel>>,
    mut scroll_areas: Query<&mut ScrollPosition, With<ImageControlPanelScroll>>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    if !panels
        .iter()
        .any(|(node, transform)| node.contains_point(*transform, cursor))
    {
        return;
    }

    let delta = wheel.read().fold(0.0, |total, event| {
        let scale = match event.unit {
            MouseScrollUnit::Line => 42.0,
            MouseScrollUnit::Pixel => 1.0,
        };
        total + event.y * scale
    });
    if delta == 0.0 {
        return;
    }
    for mut scroll in &mut scroll_areas {
        scroll.0.y = (scroll.0.y - delta).max(0.0);
    }
}

fn spawn_labeled_button<M: Component + Default, L: Component + Default>(
    panel: &mut ChildSpawnerCommands,
    caption: &str,
    value: &str,
) {
    panel
        .spawn((
            Button,
            M::default(),
            control_button_node(),
            ButtonPalette::neutral(),
            BackgroundColor(ButtonPalette::neutral().idle),
        ))
        .with_children(|button| {
            button.spawn((
                Text::new(format!("{caption}: {value}")),
                TextFont::from_font_size(14.0),
                TextColor(Color::WHITE),
                L::default(),
            ));
        });
}

fn spawn_action_button<M: Component + Default>(
    row: &mut ChildSpawnerCommands,
    label: &str,
    color: Color,
) {
    let palette = ButtonPalette::action(color);
    row.spawn((
        Button,
        InteractionDisabled,
        M::default(),
        Node {
            flex_grow: 1.0,
            min_height: px(38),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            padding: px(7).all(),
            border: px(1).all(),
            ..default()
        },
        palette,
        BackgroundColor(palette.disabled),
        BorderColor::all(Color::srgb(0.35, 0.4, 0.52)),
    ))
    .with_children(|button| {
        button.spawn((
            Text::new(label),
            TextFont::from_font_size(13.0),
            TextColor(Color::WHITE),
        ));
    });
}

fn control_button_node() -> Node {
    Node {
        width: percent(100),
        min_height: px(38),
        align_items: AlignItems::Center,
        justify_content: JustifyContent::FlexStart,
        padding: px(8).all(),
        border: px(1).all(),
        ..default()
    }
}

#[cfg(target_arch = "wasm32")]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn dispatch_browser_ui_contract(
    runner: Res<ImageRunnerStatus>,
    editor: Res<ImageEditorState>,
    input_focus: Res<InputFocus>,
    prompts: Query<
        (
            Entity,
            &ComputedNode,
            &UiGlobalTransform,
            Has<InteractionDisabled>,
        ),
        With<PromptInput>,
    >,
    seeds: Query<
        (
            Entity,
            &ComputedNode,
            &UiGlobalTransform,
            Has<InteractionDisabled>,
        ),
        (With<SeedInput>, Without<PromptInput>),
    >,
    runs: Query<
        (&ComputedNode, &UiGlobalTransform, Has<InteractionDisabled>),
        (With<RunButton>, Without<PromptInput>, Without<SeedInput>),
    >,
    saves: Query<
        (&ComputedNode, &UiGlobalTransform, Has<InteractionDisabled>),
        (
            With<SaveButton>,
            Without<PromptInput>,
            Without<SeedInput>,
            Without<RunButton>,
        ),
    >,
    mut last_contract: Local<Option<(bool, bool, bool, bool, bool, bool, u32, u32)>>,
) {
    if !browser_model_smoke_requested() {
        return;
    }
    if !matches!(runner.state, ImageRunnerState::Ready { .. }) {
        return;
    }
    let (
        Ok((prompt_entity, prompt_node, prompt_transform, prompt_disabled)),
        Ok((seed_entity, seed_node, seed_transform, seed_disabled)),
        Ok((run_node, run_transform, run_disabled)),
        Ok((save_node, save_transform, save_disabled)),
    ) = (
        prompts.single(),
        seeds.single(),
        runs.single(),
        saves.single(),
    )
    else {
        return;
    };
    if prompt_node.is_empty() || seed_node.is_empty() || run_node.is_empty() || save_node.is_empty()
    {
        return;
    }
    let (Some(model), Some(dimensions)) = (&editor.model, editor.options.dimensions) else {
        return;
    };
    let prompt_enabled = !prompt_disabled;
    let seed_enabled = !seed_disabled;
    let prompt_focused = input_focus.get() == Some(prompt_entity);
    let seed_focused = input_focus.get() == Some(seed_entity);
    let run_enabled = !run_disabled;
    let save_enabled = !save_disabled;
    let contract = (
        prompt_enabled,
        seed_enabled,
        prompt_focused,
        seed_focused,
        run_enabled,
        save_enabled,
        dimensions.width(),
        dimensions.height(),
    );
    if last_contract.as_ref() == Some(&contract) {
        return;
    }
    let prompt_center = prompt_transform.to_scale_angle_translation().2;
    let seed_center = seed_transform.to_scale_angle_translation().2;
    let run_center = run_transform.to_scale_angle_translation().2;
    let save_center = save_transform.to_scale_angle_translation().2;
    let result = (|| {
        let detail = js_sys::Object::new();
        let set = |name: &str, value: wasm_bindgen::JsValue| {
            js_sys::Reflect::set(&detail, &name.into(), &value)
                .map(|_| ())
                .map_err(|error| format!("{error:?}"))
        };
        set("event", "ready".into())?;
        set("model", model.as_str().into())?;
        set("width", dimensions.width().into())?;
        set("height", dimensions.height().into())?;
        set("prompt_x", prompt_center.x.into())?;
        set("prompt_y", prompt_center.y.into())?;
        set("prompt_enabled", prompt_enabled.into())?;
        set("prompt_focused", prompt_focused.into())?;
        set("seed_x", seed_center.x.into())?;
        set("seed_y", seed_center.y.into())?;
        set("seed_enabled", seed_enabled.into())?;
        set("seed_focused", seed_focused.into())?;
        set("run_x", run_center.x.into())?;
        set("run_y", run_center.y.into())?;
        set("run_enabled", run_enabled.into())?;
        set("save_x", save_center.x.into())?;
        set("save_y", save_center.y.into())?;
        set("save_enabled", save_enabled.into())?;

        let init = web_sys::CustomEventInit::new();
        init.set_detail(detail.as_ref());
        let event =
            web_sys::CustomEvent::new_with_event_init_dict(BROWSER_UI_CONTRACT_EVENT_NAME, &init)
                .map_err(|error| format!("{error:?}"))?;
        let window = web_sys::window().ok_or_else(|| "Window is unavailable".to_owned())?;
        window
            .dispatch_event(event.as_ref())
            .map_err(|error| format!("{error:?}"))?;
        Ok::<(), String>(())
    })();
    match result {
        Ok(()) => *last_contract = Some(contract),
        Err(error) => web_sys::console::warn_1(
            &format!("failed to dispatch browser event {BROWSER_UI_CONTRACT_EVENT_NAME}: {error}")
                .into(),
        ),
    }
}

#[cfg(target_arch = "wasm32")]
fn browser_model_smoke_requested() -> bool {
    let Some(window) = web_sys::window() else {
        return false;
    };
    let Ok(search) = window.location().search() else {
        return false;
    };
    web_sys::UrlSearchParams::new_with_str(&search)
        .ok()
        .is_some_and(|params| params.get("rendered-model-smoke").as_deref() == Some("1"))
}

#[cfg(target_arch = "wasm32")]
fn dispatch_browser_text_value(event_name: &str, value: &str) {
    let result = (|| {
        let detail = js_sys::Object::new();
        js_sys::Reflect::set(&detail, &"event".into(), &event_name.into())
            .map_err(|error| format!("{error:?}"))?;
        js_sys::Reflect::set(&detail, &"value".into(), &value.into())
            .map_err(|error| format!("{error:?}"))?;
        let init = web_sys::CustomEventInit::new();
        init.set_detail(detail.as_ref());
        let event =
            web_sys::CustomEvent::new_with_event_init_dict(BROWSER_UI_CONTRACT_EVENT_NAME, &init)
                .map_err(|error| format!("{error:?}"))?;
        let window = web_sys::window().ok_or_else(|| "Window is unavailable".to_owned())?;
        window
            .dispatch_event(event.as_ref())
            .map_err(|error| format!("{error:?}"))?;
        Ok::<(), String>(())
    })();
    if let Err(error) = result {
        web_sys::console::warn_1(
            &format!("failed to dispatch browser event {BROWSER_UI_CONTRACT_EVENT_NAME}: {error}")
                .into(),
        );
    }
}

fn select_initial_model(
    status: Res<ImageRunnerStatus>,
    mut editor: ResMut<ImageEditorState>,
    mut panel: ResMut<ImageControlPanelState>,
) {
    if editor.model.is_some() {
        return;
    }
    let ImageRunnerState::Ready { capabilities } = &status.state else {
        return;
    };
    if let Some(descriptor) = capabilities.models.first() {
        editor.model = Some(descriptor.id.clone());
        if descriptor
            .capabilities
            .tasks
            .contains(&ImageTaskKind::Generate)
        {
            editor.mode = EditorMode::Generate;
        } else if descriptor.capabilities.tasks.contains(&ImageTaskKind::Edit) {
            editor.mode = EditorMode::Edit;
        }
        apply_descriptor_size(descriptor, &mut editor, &mut panel);
        editor.options.seed = Some(0);
    }
}

fn sync_text_inputs(
    prompts: Query<&EditableText, (With<PromptInput>, Changed<EditableText>)>,
    seeds: Query<&EditableText, (With<SeedInput>, Changed<EditableText>)>,
    mut editor: ResMut<ImageEditorState>,
    mut panel: ResMut<ImageControlPanelState>,
) {
    if let Ok(prompt) = prompts.single() {
        let value = prompt.value().to_string();
        if editor.prompt_or_instruction != value {
            editor.prompt_or_instruction = value;
            #[cfg(target_arch = "wasm32")]
            if browser_model_smoke_requested() {
                dispatch_browser_text_value("prompt_changed", &editor.prompt_or_instruction);
            }
        }
    }
    if let Ok(seed) = seeds.single() {
        let value = seed.value().to_string();
        if value.is_empty() {
            let changed = editor.options.seed.is_some();
            if changed {
                editor.options.seed = None;
                #[cfg(target_arch = "wasm32")]
                if browser_model_smoke_requested() {
                    dispatch_browser_text_value("seed_changed", &value);
                }
            }
            panel.seed_valid = true;
        } else {
            match value.parse::<u64>() {
                Ok(seed) => {
                    let changed = editor.options.seed != Some(seed);
                    if changed {
                        editor.options.seed = Some(seed);
                        #[cfg(target_arch = "wasm32")]
                        if browser_model_smoke_requested() {
                            dispatch_browser_text_value("seed_changed", &value);
                        }
                    }
                    panel.seed_valid = true;
                }
                Err(error) => {
                    panel.seed_valid = false;
                    panel.notice = format!("Invalid seed: {error}");
                }
            }
        }
    }
}

fn handle_mode_button(
    interactions: Query<&Interaction, (Changed<Interaction>, With<ModeButton>)>,
    status: Res<ImageRunnerStatus>,
    mut editor: ResMut<ImageEditorState>,
    mut panel: ResMut<ImageControlPanelState>,
) {
    if !interactions
        .iter()
        .any(|interaction| *interaction == Interaction::Pressed)
    {
        return;
    }
    let requested_mode = match editor.mode {
        EditorMode::Generate => EditorMode::Edit,
        EditorMode::Edit => EditorMode::Generate,
    };
    let Some(descriptor) = descriptor_for_mode(&status.state, requested_mode) else {
        panel.notice = format!(
            "The loaded runtime does not support {} mode",
            editor_mode_label(requested_mode)
        );
        return;
    };
    editor.mode = requested_mode;
    editor.model = Some(descriptor.id.clone());
    apply_descriptor_size(descriptor, &mut editor, &mut panel);
}

fn handle_model_button(
    interactions: Query<&Interaction, (Changed<Interaction>, With<ModelButton>)>,
    status: Res<ImageRunnerStatus>,
    mut editor: ResMut<ImageEditorState>,
    mut panel: ResMut<ImageControlPanelState>,
) {
    if !interactions
        .iter()
        .any(|interaction| *interaction == Interaction::Pressed)
    {
        return;
    }
    let ImageRunnerState::Ready { capabilities } = &status.state else {
        return;
    };
    let Some(descriptor) = next_model_descriptor(&capabilities.models, editor.model.as_ref())
    else {
        return;
    };
    editor.model = Some(descriptor.id.clone());
    if descriptor
        .capabilities
        .tasks
        .contains(&ImageTaskKind::Generate)
    {
        editor.mode = EditorMode::Generate;
    } else if descriptor.capabilities.tasks.contains(&ImageTaskKind::Edit) {
        editor.mode = EditorMode::Edit;
    }
    apply_descriptor_size(descriptor, &mut editor, &mut panel);
}

fn next_model_descriptor<'a>(
    models: &'a [ModelDescriptor],
    current: Option<&burn_image::ModelId>,
) -> Option<&'a ModelDescriptor> {
    if models.len() < 2 {
        return None;
    }
    let current =
        current.and_then(|model| models.iter().position(|descriptor| descriptor.id == *model));
    models.get(current.map_or(0, |index| index + 1) % models.len())
}

fn descriptor_for_mode(state: &ImageRunnerState, mode: EditorMode) -> Option<&ModelDescriptor> {
    let ImageRunnerState::Ready { capabilities } = state else {
        return None;
    };
    let task = editor_mode_task(mode);
    capabilities
        .models
        .iter()
        .find(|descriptor| descriptor.capabilities.tasks.contains(&task))
}

const fn editor_mode_task(mode: EditorMode) -> ImageTaskKind {
    match mode {
        EditorMode::Generate => ImageTaskKind::Generate,
        EditorMode::Edit => ImageTaskKind::Edit,
    }
}

const fn editor_mode_label(mode: EditorMode) -> &'static str {
    match mode {
        EditorMode::Generate => "Generate",
        EditorMode::Edit => "Edit",
    }
}

fn handle_size_button(
    interactions: Query<&Interaction, (Changed<Interaction>, With<SizeButton>)>,
    status: Res<ImageRunnerStatus>,
    mut editor: ResMut<ImageEditorState>,
    mut panel: ResMut<ImageControlPanelState>,
) {
    if !interactions
        .iter()
        .any(|interaction| *interaction == Interaction::Pressed)
    {
        return;
    }
    let ImageRunnerState::Ready { capabilities } = &status.state else {
        return;
    };
    let Some(descriptor) = editor.model.as_ref().and_then(|model| {
        capabilities
            .models
            .iter()
            .find(|descriptor| descriptor.id == *model)
    }) else {
        return;
    };
    let current = editor
        .options
        .dimensions
        .and_then(preset_index)
        .unwrap_or(panel.size_index);
    if let Some(index) = next_supported_size_index_for_descriptor(current, descriptor) {
        panel.size_index = index;
        editor.options.dimensions = Some(preset_dimensions(index));
    }
}

fn handle_reference_button(
    interactions: Query<&Interaction, (Changed<Interaction>, With<ReferenceButton>)>,
    mut panel: ResMut<ImageControlPanelState>,
    #[cfg(all(feature = "native-io", not(target_arch = "wasm32")))] dialog_state: Res<
        ImageFileDialogState,
    >,
    #[cfg(all(feature = "native-io", not(target_arch = "wasm32")))] mut open: MessageWriter<
        OpenImageFileDialog,
    >,
) {
    if !interactions
        .iter()
        .any(|interaction| *interaction == Interaction::Pressed)
    {
        return;
    }
    #[cfg(target_arch = "wasm32")]
    {
        if let Err(error) = click_browser_reference_input() {
            panel.notice = format!("Could not open the browser image picker: {error:?}");
        }
    }
    #[cfg(all(feature = "native-io", not(target_arch = "wasm32")))]
    {
        if dialog_state.is_open {
            panel.notice = "An image picker is already open".into();
        } else {
            open.write(OpenImageFileDialog {
                id: REFERENCE_IMAGE_IO_ID,
                max_bytes: MAX_REFERENCE_BYTES,
            });
            panel.notice = "Opening image picker".into();
        }
    }
    #[cfg(all(not(feature = "native-io"), not(target_arch = "wasm32")))]
    {
        panel.notice = "Drop a PNG, JPEG, or WebP file on the window".into();
    }
}

#[cfg(all(feature = "native-io", not(target_arch = "wasm32")))]
fn accept_native_file_dialog(
    mut selected: MessageReader<ImageFileDialogSelected>,
    mut cancelled: MessageReader<ImageFileDialogCancelled>,
    mut failed: MessageReader<ImageFileDialogFailed>,
    mut load: MessageWriter<LoadImageBytes>,
    mut panel: ResMut<ImageControlPanelState>,
) {
    for selected in selected.read() {
        if selected.id != REFERENCE_IMAGE_IO_ID {
            continue;
        }
        load.write(LoadImageBytes {
            id: selected.id,
            bytes: selected.bytes.clone(),
            encoding: None,
        });
        panel.notice = format!("Loading {}", selected.file_name);
    }
    if cancelled
        .read()
        .any(|cancelled| cancelled.id == REFERENCE_IMAGE_IO_ID)
    {
        panel.notice = "Image selection cancelled".into();
    }
    for failed in failed.read() {
        if failed.id == REFERENCE_IMAGE_IO_ID {
            panel.notice = failed.error.to_string();
        }
    }
}

#[cfg(any(target_arch = "wasm32", not(feature = "native-io")))]
fn accept_native_file_dialog() {}

#[allow(clippy::type_complexity)]
fn handle_view_buttons(
    fit_interactions: Query<
        &Interaction,
        (
            Changed<Interaction>,
            With<FitButton>,
            Without<InteractionDisabled>,
        ),
    >,
    actual_size_interactions: Query<
        &Interaction,
        (
            Changed<Interaction>,
            With<ActualSizeButton>,
            Without<InteractionDisabled>,
        ),
    >,
    mut fit: MessageWriter<FitImageView>,
    mut actual_size: MessageWriter<ActualSizeImageView>,
) {
    if fit_interactions
        .iter()
        .any(|interaction| *interaction == Interaction::Pressed)
    {
        fit.write(FitImageView);
    }
    if actual_size_interactions
        .iter()
        .any(|interaction| *interaction == Interaction::Pressed)
    {
        actual_size.write(ActualSizeImageView);
    }
}

#[allow(clippy::type_complexity)]
fn handle_run_button(
    interactions: Query<
        &Interaction,
        (
            Changed<Interaction>,
            With<RunButton>,
            Without<InteractionDisabled>,
        ),
    >,
    editor: Res<ImageEditorState>,
    mut jobs: ResMut<ImageJobs>,
    mut panel: ResMut<ImageControlPanelState>,
    mut submit: MessageWriter<crate::SubmitImageJob>,
) {
    if !interactions
        .iter()
        .any(|interaction| *interaction == Interaction::Pressed)
    {
        return;
    }
    if !panel.seed_valid {
        panel.notice = "Fix the seed before running".into();
        return;
    }
    if jobs.iter().any(|job| !job.phase.is_terminal()) || has_pending_submission(&panel, &jobs) {
        panel.notice = "An image job is already active".into();
        return;
    }
    let id = jobs.reserve_id();
    match editor.submission(id) {
        Ok(request) => {
            submit.write(request);
            panel.latest_job = Some(id);
            panel.notice.clear();
        }
        Err(error) => panel.notice = error.to_string(),
    }
}

#[allow(clippy::type_complexity)]
fn handle_cancel_button(
    interactions: Query<
        &Interaction,
        (
            Changed<Interaction>,
            With<CancelButton>,
            Without<InteractionDisabled>,
        ),
    >,
    jobs: Res<ImageJobs>,
    mut panel: ResMut<ImageControlPanelState>,
    mut cancel: MessageWriter<CancelImageJob>,
) {
    if !interactions
        .iter()
        .any(|interaction| *interaction == Interaction::Pressed)
    {
        return;
    }
    let id = panel
        .latest_job
        .filter(|id| jobs.get(*id).is_some_and(|job| !job.phase.is_terminal()))
        .or_else(|| {
            jobs.iter()
                .filter(|job| !job.phase.is_terminal())
                .map(|job| job.id)
                .max()
        });
    if let Some(id) = id
        && jobs.get(id).is_some_and(|job| !job.phase.is_terminal())
    {
        cancel.write(CancelImageJob { id });
        panel.notice = format!("Cancellation requested for job {}", id.0);
    } else {
        panel.notice = "No running job to cancel".into();
    }
}

#[allow(clippy::type_complexity)]
fn handle_save_button(
    interactions: Query<
        &Interaction,
        (
            Changed<Interaction>,
            With<SaveButton>,
            Without<InteractionDisabled>,
        ),
    >,
    mut panel: ResMut<ImageControlPanelState>,
    mut download: MessageWriter<PrepareImageDownload>,
) {
    if !interactions
        .iter()
        .any(|interaction| *interaction == Interaction::Pressed)
    {
        return;
    }
    let Some((job, image)) = panel.latest_output.clone() else {
        panel.notice = "No completed output to save".into();
        return;
    };

    #[cfg(all(feature = "native-io", not(target_arch = "wasm32")))]
    let _ = &mut download;

    #[cfg(all(feature = "native-io", not(target_arch = "wasm32")))]
    {
        let file_name = format!("burn-image-{}.png", job.0);
        match std::env::current_dir()
            .map(|directory| directory.join(&file_name))
            .map_err(crate::FrontendError::from)
            .and_then(|path| {
                crate::save_image_file(&path, &image, ImageEncoding::Png).map(|()| path)
            }) {
            Ok(_) => panel.notice = format!("Saved {file_name}"),
            Err(error) => panel.notice = error.to_string(),
        }
    }
    #[cfg(any(target_arch = "wasm32", not(feature = "native-io")))]
    {
        download.write(PrepareImageDownload {
            id: DOWNLOAD_IO_ID,
            image,
            encoding: ImageEncoding::Png,
            file_stem: format!("burn-image-{}", job.0),
        });
        panel.notice = "Preparing PNG download".into();
    }
}

fn accept_reference_images(
    mut loaded: MessageReader<ImageBytesLoaded>,
    runner: Res<ImageRunnerStatus>,
    mut editor: ResMut<ImageEditorState>,
    mut panel: ResMut<ImageControlPanelState>,
) {
    for loaded in loaded.read() {
        if loaded.id != REFERENCE_IMAGE_IO_ID {
            continue;
        }
        let dimensions = loaded.image.dimensions();
        editor.source = Some(loaded.image.clone());
        let edit_descriptor = descriptor_for_mode(&runner.state, EditorMode::Edit);
        if let Some(descriptor) = edit_descriptor {
            editor.mode = EditorMode::Edit;
            editor.model = Some(descriptor.id.clone());
            apply_descriptor_size(descriptor, &mut editor, &mut panel);
        }
        let loaded_notice = dimensions.map_or_else(
            || "Reference image loaded".into(),
            |size| format!("Reference loaded: {} x {}", size.width(), size.height()),
        );
        panel.notice = if edit_descriptor.is_some() {
            loaded_notice
        } else {
            format!("{loaded_notice}; the loaded runtime does not support Edit mode")
        };
    }
}

fn apply_descriptor_size(
    descriptor: &ModelDescriptor,
    editor: &mut ImageEditorState,
    panel: &mut ImageControlPanelState,
) {
    if editor.options.dimensions.is_none()
        && let Some(dimensions) = model_default_dimensions(descriptor)
        && descriptor_supports_dimensions(descriptor, dimensions)
    {
        if let Some(index) = preset_index(dimensions) {
            panel.size_index = index;
        }
        editor.options.dimensions = Some(dimensions);
        return;
    }
    if let Some(dimensions) = editor.options.dimensions
        && descriptor_supports_dimensions(descriptor, dimensions)
    {
        if let Some(index) = preset_index(dimensions) {
            panel.size_index = index;
        }
        return;
    }
    if let Some(index) = preferred_size_index_for_descriptor(descriptor) {
        panel.size_index = index;
        editor.options.dimensions = Some(preset_dimensions(index));
    } else {
        // A model may accept valid dimensions that are not represented by this compact UI.
        // Leaving the field empty delegates to that model's validated default.
        editor.options.dimensions = None;
    }
}

fn descriptor_supports_dimensions(descriptor: &ModelDescriptor, dimensions: Dimensions) -> bool {
    if descriptor
        .capabilities
        .dimensions
        .supports(dimensions)
        .is_err()
    {
        return false;
    }
    #[cfg(feature = "boogu")]
    if crate::boogu::variant_for_model(&descriptor.id)
        == Some(burn_boogu::BooguVariant::Image01EditTurbo1k5)
    {
        return burn_boogu::BOOGU_1K5_OUTPUT_PRESETS
            .contains(&(dimensions.width(), dimensions.height()));
    }
    true
}

fn model_default_dimensions(_descriptor: &ModelDescriptor) -> Option<Dimensions> {
    #[cfg(feature = "boogu")]
    if let Some(variant) = crate::boogu::variant_for_model(&_descriptor.id) {
        let edge = if variant == burn_boogu::BooguVariant::Image01EditTurbo1k5 {
            burn_boogu::BOOGU_1K5_DEFAULT_EDGE
        } else {
            burn_boogu::BOOGU_DEFAULT_EDGE
        };
        return Dimensions::new(edge, edge).ok();
    }

    None
}

#[cfg(test)]
fn preferred_size_index(constraints: &DimensionConstraints) -> Option<usize> {
    let default = SIZE_PRESETS
        .iter()
        .position(|preset| *preset == DEFAULT_SIZE_PRESET)
        .expect("default size preset must be listed");
    if constraints.supports(preset_dimensions(default)).is_ok() {
        return Some(default);
    }
    (0..SIZE_PRESETS.len()).find(|index| constraints.supports(preset_dimensions(*index)).is_ok())
}

fn preferred_size_index_for_descriptor(descriptor: &ModelDescriptor) -> Option<usize> {
    if let Some(dimensions) = model_default_dimensions(descriptor)
        && descriptor_supports_dimensions(descriptor, dimensions)
        && let Some(index) = preset_index(dimensions)
    {
        return Some(index);
    }
    let default = SIZE_PRESETS
        .iter()
        .position(|preset| *preset == DEFAULT_SIZE_PRESET)
        .expect("default size preset must be listed");
    if descriptor_supports_dimensions(descriptor, preset_dimensions(default)) {
        return Some(default);
    }
    (0..SIZE_PRESETS.len())
        .find(|index| descriptor_supports_dimensions(descriptor, preset_dimensions(*index)))
}

#[cfg(test)]
fn next_supported_size_index(current: usize, constraints: &DimensionConstraints) -> Option<usize> {
    (1..=SIZE_PRESETS.len())
        .map(|offset| (current + offset) % SIZE_PRESETS.len())
        .find(|index| constraints.supports(preset_dimensions(*index)).is_ok())
}

fn next_supported_size_index_for_descriptor(
    current: usize,
    descriptor: &ModelDescriptor,
) -> Option<usize> {
    (1..=SIZE_PRESETS.len())
        .map(|offset| (current + offset) % SIZE_PRESETS.len())
        .find(|index| descriptor_supports_dimensions(descriptor, preset_dimensions(*index)))
}

fn preset_index(dimensions: Dimensions) -> Option<usize> {
    SIZE_PRESETS
        .iter()
        .position(|(width, height)| dimensions.width() == *width && dimensions.height() == *height)
}

fn preset_dimensions(index: usize) -> Dimensions {
    let (width, height) = SIZE_PRESETS[index];
    Dimensions::new(width, height).expect("size presets are valid")
}

fn capture_outputs(
    jobs: Res<ImageJobs>,
    mut outputs: MessageReader<CompleteImageJob>,
    mut panel: ResMut<ImageControlPanelState>,
) {
    for output in outputs.read() {
        if !jobs
            .get(output.id)
            .is_some_and(|job| job.phase == ImageJobPhase::Completed)
        {
            continue;
        }
        if let Some(image) = output.output.images.first() {
            panel.latest_output = Some((output.id, image.image.clone()));
            panel.latest_job = Some(output.id);
            panel.notice.clear();
        }
    }
}

fn capture_frontend_errors(
    mut rejected: MessageReader<ImageJobRejected>,
    mut io_failed: MessageReader<ImageIoFailed>,
    mut display_failed: MessageReader<ImageDisplayFailed>,
    mut panel: ResMut<ImageControlPanelState>,
) {
    for error in rejected.read() {
        panel.latest_job = Some(error.id);
        panel.notice = error.error.to_string();
    }
    for error in io_failed.read() {
        panel.notice = error.error.to_string();
    }
    for error in display_failed.read() {
        panel.notice = error.error.to_string();
    }
}

// The four marker-filtered mutable Text queries must be a ParamSet: Bevy
// correctly rejects them as ordinary parameters because their access could
// overlap, even though each marker is unique in this plugin.
#[allow(clippy::type_complexity)]
fn update_control_labels(
    editor: Res<ImageEditorState>,
    runner: Res<ImageRunnerStatus>,
    mut labels: ParamSet<(
        Query<&mut Text, With<ModeButtonLabel>>,
        Query<&mut Text, With<ModelButtonLabel>>,
        Query<&mut Text, With<SizeButtonLabel>>,
        Query<&mut Text, With<ReferenceLabel>>,
    )>,
) {
    if !editor.is_changed() && !runner.is_changed() {
        return;
    }
    if let Ok(mut label) = labels.p0().single_mut() {
        let value = format!(
            "Mode: {}",
            match editor.mode {
                EditorMode::Generate => "Generate",
                EditorMode::Edit => "Edit",
            }
        );
        if label.0 != value {
            label.0 = value;
        }
    }
    if let Ok(mut label) = labels.p1().single_mut() {
        let value = model_control_label(&editor, &runner.state);
        if label.0 != value {
            label.0 = value;
        }
    }
    if let Ok(mut label) = labels.p2().single_mut() {
        let value = editor.options.dimensions.map_or_else(
            || "Size: model default".into(),
            |size| format!("Size: {} x {}", size.width(), size.height()),
        );
        if label.0 != value {
            label.0 = value;
        }
    }
    if let Ok(mut label) = labels.p3().single_mut() {
        let value = if editor.source.is_some() {
            "Reference: loaded (click to replace)".into()
        } else {
            reference_button_text().into()
        };
        if label.0 != value {
            label.0 = value;
        }
    }
}

fn model_control_label(editor: &ImageEditorState, runner: &ImageRunnerState) -> String {
    let Some(model) = &editor.model else {
        return runner_state_label(runner);
    };
    let ImageRunnerState::Ready { capabilities } = runner else {
        return format!("Loaded model: {model}");
    };
    let display_name = capabilities
        .descriptor(model)
        .map(|descriptor| descriptor.display_name.as_str())
        .unwrap_or_else(|| model.as_str());
    if capabilities.models.len() < 2 {
        format!("Loaded model: {display_name}")
    } else {
        format!("Model: {display_name}")
    }
}

fn can_cycle_models(runner: &ImageRunnerState) -> bool {
    matches!(
        runner,
        ImageRunnerState::Ready { capabilities } if capabilities.models.len() > 1
    )
}

fn can_change_mode(runner: &ImageRunnerState, current: EditorMode) -> bool {
    let requested = match current {
        EditorMode::Generate => EditorMode::Edit,
        EditorMode::Edit => EditorMode::Generate,
    };
    descriptor_for_mode(runner, requested).is_some()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProgressTone {
    Normal,
    Complete,
    Warning,
    Failed,
}

#[derive(Clone, Debug, PartialEq)]
struct ProgressPresentation {
    headline: String,
    detail: String,
    fraction: Option<f32>,
    tone: ProgressTone,
}

fn progress_presentation(
    runner: &ImageRunnerState,
    job: Option<&crate::ImageJobRecord>,
    notice: &str,
) -> ProgressPresentation {
    let mut value = job.map_or_else(
        || runner_progress_presentation(runner),
        job_progress_presentation,
    );
    if !notice.is_empty() {
        value.detail = if value.detail.is_empty() {
            notice.to_owned()
        } else {
            format!("{}\n{notice}", value.detail)
        };
    }
    value
}

fn runner_progress_presentation(state: &ImageRunnerState) -> ProgressPresentation {
    match state {
        ImageRunnerState::Missing => ProgressPresentation {
            headline: "Model runtime unavailable".into(),
            detail: "Install a WGPU model runtime to generate an image".into(),
            fraction: Some(0.0),
            tone: ProgressTone::Failed,
        },
        ImageRunnerState::Initializing { message } => ProgressPresentation {
            headline: "Preparing model runtime".into(),
            detail: message.clone(),
            fraction: setup_progress_fraction(message),
            tone: ProgressTone::Normal,
        },
        ImageRunnerState::Ready { .. } => ProgressPresentation {
            headline: "Model runtime ready".into(),
            detail: "Enter a prompt, choose settings, and run".into(),
            fraction: Some(1.0),
            tone: ProgressTone::Complete,
        },
        ImageRunnerState::Failed { error } => ProgressPresentation {
            headline: "Model runtime failed".into(),
            detail: error.to_string(),
            fraction: Some(0.0),
            tone: ProgressTone::Failed,
        },
    }
}

fn setup_progress_fraction(message: &str) -> Option<f32> {
    let remainder = message.strip_prefix("Model setup ")?;
    let fraction = remainder.split(':').next()?;
    let (completed, total) = fraction.split_once('/')?;
    let completed = completed.trim().parse::<u32>().ok()?;
    let total = total.trim().parse::<u32>().ok()?;
    (total > 0 && completed <= total).then(|| completed as f32 / total as f32)
}

fn job_progress_presentation(job: &crate::ImageJobRecord) -> ProgressPresentation {
    let prefix = format!("Job {}", job.id.0);
    match &job.phase {
        ImageJobPhase::Queued => ProgressPresentation {
            headline: format!("{prefix}: queued"),
            detail: "Waiting for the GPU runtime".into(),
            fraction: Some(0.0),
            tone: ProgressTone::Normal,
        },
        ImageJobPhase::Running => job.last_progress.as_ref().map_or_else(
            || ProgressPresentation {
                headline: format!("{prefix}: running"),
                detail: "Preparing inference".into(),
                fraction: None,
                tone: ProgressTone::Normal,
            },
            |progress| event_progress_presentation(&prefix, progress),
        ),
        ImageJobPhase::Completed => ProgressPresentation {
            headline: format!("{prefix}: complete"),
            detail: "Output is ready to view or save".into(),
            fraction: Some(1.0),
            tone: ProgressTone::Complete,
        },
        ImageJobPhase::Failed { error } => ProgressPresentation {
            headline: format!("{prefix}: failed"),
            detail: error.to_string(),
            fraction: Some(0.0),
            tone: ProgressTone::Failed,
        },
        ImageJobPhase::Cancelled => ProgressPresentation {
            headline: format!("{prefix}: cancelled"),
            detail: "The request was stopped".into(),
            fraction: Some(0.0),
            tone: ProgressTone::Warning,
        },
    }
}

fn event_progress_presentation(prefix: &str, progress: &ProgressEvent) -> ProgressPresentation {
    let (headline, detail, fraction, tone) = match progress {
        ProgressEvent::RunStarted { .. } => (
            "Starting inference".into(),
            "Preparing model inputs".into(),
            Some(0.0),
            ProgressTone::Normal,
        ),
        ProgressEvent::ArtifactStarted {
            path,
            component,
            file_index,
            file_count,
            total_bytes,
            ..
        } => (
            "Loading model data".into(),
            format!(
                "{}: file {} of {} - {} ({})",
                component
                    .as_ref()
                    .map(ToString::to_string)
                    .unwrap_or_else(|| "artifact".into()),
                file_index + 1,
                file_count,
                path,
                format_bytes(*total_bytes)
            ),
            Some(*file_index as f32 / (*file_count).max(1) as f32),
            ProgressTone::Normal,
        ),
        ProgressEvent::ArtifactProgress {
            path,
            loaded_bytes,
            total_bytes,
            ..
        } => (
            "Loading model data".into(),
            format!(
                "{} - {} of {}",
                path,
                format_bytes(*loaded_bytes),
                format_bytes(*total_bytes)
            ),
            Some(*loaded_bytes as f32 / (*total_bytes).max(1) as f32),
            ProgressTone::Normal,
        ),
        ProgressEvent::ArtifactVerified { path, .. } => (
            "Model object verified".into(),
            path.to_string(),
            None,
            ProgressTone::Normal,
        ),
        ProgressEvent::StageStarted {
            stage, total_steps, ..
        } => (
            format!("Running {}", humanize_stage(stage)),
            total_steps.map_or_else(
                || "Stage started".into(),
                |steps| format!("0 of {steps} steps"),
            ),
            total_steps.map(|_| 0.0),
            ProgressTone::Normal,
        ),
        ProgressEvent::Step {
            stage,
            step,
            total_steps,
            ..
        } => (
            format!("Running {}", humanize_stage(stage)),
            format!("Step {step} of {total_steps}"),
            Some(*step as f32 / (*total_steps).max(1) as f32),
            ProgressTone::Normal,
        ),
        ProgressEvent::StageCompleted { stage, .. } => (
            format!("{} complete", humanize_stage(stage)),
            "Moving to the next stage".into(),
            Some(1.0),
            ProgressTone::Normal,
        ),
        ProgressEvent::Warning { message, .. } => (
            "Runtime warning".into(),
            message.clone(),
            None,
            ProgressTone::Warning,
        ),
        ProgressEvent::RunCompleted { .. } => (
            "Inference complete".into(),
            "Preparing the output image".into(),
            Some(1.0),
            ProgressTone::Complete,
        ),
        ProgressEvent::RunFailed { message, .. } => (
            "Inference failed".into(),
            message.clone(),
            Some(0.0),
            ProgressTone::Failed,
        ),
        ProgressEvent::RunCancelled { .. } => (
            "Inference cancelled".into(),
            "The request was stopped".into(),
            Some(0.0),
            ProgressTone::Warning,
        ),
    };
    ProgressPresentation {
        headline: format!("{prefix}: {headline}"),
        detail,
        fraction,
        tone,
    }
}

fn format_bytes(bytes: u64) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = 1024.0 * MIB;
    if bytes as f64 >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB)
    } else if bytes as f64 >= MIB {
        format!("{:.1} MiB", bytes as f64 / MIB)
    } else if bytes >= 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

fn humanize_stage(stage: &str) -> String {
    stage.replace(['-', '_'], " ")
}

#[allow(clippy::type_complexity)]
fn update_progress_panel(
    time: Res<Time>,
    runner: Res<ImageRunnerStatus>,
    jobs: Res<ImageJobs>,
    panel: Res<ImageControlPanelState>,
    mut labels: ParamSet<(
        Query<&mut Text, With<ProgressLabel>>,
        Query<&mut Text, With<ProgressDetailLabel>>,
    )>,
    mut fills: Query<(&mut Node, &mut BackgroundColor), With<ProgressFill>>,
) {
    let presentation = progress_presentation(
        &runner.state,
        panel.latest_job.and_then(|id| jobs.get(id)),
        &panel.notice,
    );
    if presentation.fraction.is_some()
        && !runner.is_changed()
        && !jobs.is_changed()
        && !panel.is_changed()
    {
        return;
    }
    if let Ok(mut label) = labels.p0().single_mut()
        && label.0 != presentation.headline
    {
        label.0 = presentation.headline.clone();
    }
    if let Ok(mut label) = labels.p1().single_mut()
        && label.0 != presentation.detail
    {
        label.0 = presentation.detail.clone();
    }
    for (mut node, mut background) in &mut fills {
        let (left, width) = match presentation.fraction {
            Some(fraction) => (px(0), percent(100.0 * fraction.clamp(0.0, 1.0))),
            None => (
                percent((time.elapsed_secs() * 42.0) % 128.0 - 28.0),
                percent(28),
            ),
        };
        if node.left != left || node.width != width {
            node.left = left;
            node.width = width;
        }
        let color = match presentation.tone {
            ProgressTone::Normal => Color::srgb(0.32, 0.68, 0.83),
            ProgressTone::Complete => Color::srgb(0.34, 0.74, 0.52),
            ProgressTone::Warning => Color::srgb(0.84, 0.61, 0.24),
            ProgressTone::Failed => Color::srgb(0.78, 0.28, 0.3),
        };
        if background.0 != color {
            background.0 = color;
        }
    }
}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn update_action_availability(
    mut commands: Commands,
    runner: Res<ImageRunnerStatus>,
    editor: Res<ImageEditorState>,
    jobs: Res<ImageJobs>,
    panel: Res<ImageControlPanelState>,
    mode_buttons: Query<(Entity, Has<InteractionDisabled>), With<ModeButton>>,
    model_buttons: Query<
        (Entity, Has<InteractionDisabled>),
        (With<ModelButton>, Without<ModeButton>),
    >,
    run_buttons: Query<(Entity, Has<InteractionDisabled>), With<RunButton>>,
    cancel_buttons: Query<
        (Entity, Has<InteractionDisabled>),
        (With<CancelButton>, Without<RunButton>),
    >,
    save_buttons: Query<
        (Entity, Has<InteractionDisabled>),
        (With<SaveButton>, Without<RunButton>, Without<CancelButton>),
    >,
    view_buttons: Query<
        (Entity, Has<InteractionDisabled>),
        (
            Or<(With<FitButton>, With<ActualSizeButton>)>,
            Without<RunButton>,
            Without<CancelButton>,
            Without<SaveButton>,
        ),
    >,
) {
    if !runner.is_changed() && !editor.is_changed() && !jobs.is_changed() && !panel.is_changed() {
        return;
    }
    let running = jobs.iter().any(|job| !job.phase.is_terminal());
    let can_run = !running
        && !has_pending_submission(&panel, &jobs)
        && panel.seed_valid
        && matches!(runner.state, ImageRunnerState::Ready { .. })
        && editor.model.is_some()
        && editor.validate_request().is_ok();
    let can_save = panel.latest_output.is_some();
    let can_adjust_view = can_save || editor.source.is_some();
    let can_select_model = !running && can_cycle_models(&runner.state);
    let can_select_mode = !running && can_change_mode(&runner.state, editor.mode);

    for (entity, disabled) in &mode_buttons {
        set_button_disabled(&mut commands, entity, disabled, !can_select_mode);
    }
    for (entity, disabled) in &model_buttons {
        set_button_disabled(&mut commands, entity, disabled, !can_select_model);
    }
    for (entity, disabled) in &run_buttons {
        set_button_disabled(&mut commands, entity, disabled, !can_run);
    }
    for (entity, disabled) in &cancel_buttons {
        set_button_disabled(&mut commands, entity, disabled, !running);
    }
    for (entity, disabled) in &save_buttons {
        set_button_disabled(&mut commands, entity, disabled, !can_save);
    }
    for (entity, disabled) in &view_buttons {
        set_button_disabled(&mut commands, entity, disabled, !can_adjust_view);
    }
}

fn has_pending_submission(panel: &ImageControlPanelState, jobs: &ImageJobs) -> bool {
    panel.latest_job.is_some_and(|id| {
        jobs.get(id).is_none()
            && panel
                .latest_output
                .as_ref()
                .is_none_or(|(completed_id, _)| *completed_id != id)
    })
}

fn set_button_disabled(
    commands: &mut Commands,
    entity: Entity,
    currently_disabled: bool,
    disabled: bool,
) {
    if disabled == currently_disabled {
        return;
    }
    if disabled {
        commands.entity(entity).insert(InteractionDisabled);
    } else {
        commands.entity(entity).remove::<InteractionDisabled>();
    }
}

#[cfg(test)]
fn format_progress(progress: &ProgressEvent) -> String {
    match progress {
        ProgressEvent::RunStarted { .. } => "starting".into(),
        ProgressEvent::ArtifactStarted {
            file_index,
            file_count,
            ..
        } => format!("artifact {}/{file_count}", file_index + 1),
        ProgressEvent::ArtifactProgress {
            loaded_bytes,
            total_bytes,
            ..
        } => format!(
            "artifact {:.1}%",
            100.0 * *loaded_bytes as f64 / (*total_bytes).max(1) as f64
        ),
        ProgressEvent::ArtifactVerified { path, .. } => format!("verified {path}"),
        ProgressEvent::StageStarted { stage, .. } => format!("{stage}: starting"),
        ProgressEvent::Step {
            stage,
            step,
            total_steps,
            ..
        } => format!("{stage}: {step}/{total_steps}"),
        ProgressEvent::StageCompleted { stage, .. } => format!("{stage}: complete"),
        ProgressEvent::Warning { message, .. } => format!("warning: {message}"),
        ProgressEvent::RunCompleted { .. } => "completed".into(),
        ProgressEvent::RunFailed { message, .. } => format!("failed: {message}"),
        ProgressEvent::RunCancelled { .. } => "cancelled".into(),
    }
}

fn runner_state_label(state: &ImageRunnerState) -> String {
    match state {
        ImageRunnerState::Missing => "No model runtime installed".into(),
        ImageRunnerState::Initializing { message } => message.clone(),
        ImageRunnerState::Ready { .. } => "Model runtime ready".into(),
        ImageRunnerState::Failed { error } => format!("Model runtime failed: {error}"),
    }
}

fn update_button_colors(
    mut buttons: Query<(
        &Interaction,
        Has<InteractionDisabled>,
        &ButtonPalette,
        &mut BackgroundColor,
    )>,
) {
    for (interaction, disabled, palette, mut background) in &mut buttons {
        let color = if disabled {
            palette.disabled
        } else if *interaction == Interaction::Pressed {
            palette.pressed
        } else {
            palette.idle
        };
        if background.0 != color {
            background.0 = color;
        }
    }
}

const fn reference_button_text() -> &'static str {
    if cfg!(target_arch = "wasm32") {
        "Reference: choose image..."
    } else {
        "Reference: choose or drop image"
    }
}

#[cfg(all(feature = "native-io", not(target_arch = "wasm32")))]
fn accept_native_file_drop(
    mut drops: MessageReader<FileDragAndDrop>,
    mut read: MessageWriter<ReadDroppedImageFile>,
    mut panel: ResMut<ImageControlPanelState>,
) {
    for drop in drops.read() {
        let FileDragAndDrop::DroppedFile { path_buf, .. } = drop else {
            continue;
        };
        let file_name = path_buf
            .file_name()
            .and_then(|name| name.to_str())
            .map(sanitize_image_file_name)
            .unwrap_or_else(|| "image".to_owned());
        read.write(ReadDroppedImageFile {
            id: REFERENCE_IMAGE_IO_ID,
            path: path_buf.clone(),
            max_bytes: MAX_REFERENCE_BYTES,
        });
        panel.notice = format!("Reading {file_name}");
    }
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    static BROWSER_REFERENCE_QUEUE: std::cell::RefCell<Option<Vec<u8>>> = const {
        std::cell::RefCell::new(None)
    };
}

/// Receive browser-selected image bytes from the JavaScript host.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn provide_reference_image(bytes: Vec<u8>) -> Result<(), wasm_bindgen::JsValue> {
    if bytes.is_empty() || bytes.len() > MAX_REFERENCE_BYTES {
        return Err(wasm_bindgen::JsValue::from_str(
            "reference image must contain 1..=64 MiB",
        ));
    }
    // The picker has one reference slot. If the host submits several files
    // before Bevy's next update, retain only the newest bounded payload.
    BROWSER_REFERENCE_QUEUE.with(|queue| *queue.borrow_mut() = Some(bytes));
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn drain_browser_reference_queue(mut load: MessageWriter<LoadImageBytes>) {
    BROWSER_REFERENCE_QUEUE.with(|queue| {
        if let Some(bytes) = queue.borrow_mut().take() {
            load.write(LoadImageBytes {
                id: REFERENCE_IMAGE_IO_ID,
                bytes: bytes.into(),
                encoding: None,
            });
        }
    });
}

#[cfg(target_arch = "wasm32")]
fn click_browser_reference_input() -> Result<(), wasm_bindgen::JsValue> {
    use wasm_bindgen::JsCast;

    let document = web_sys::window()
        .and_then(|window| window.document())
        .ok_or_else(|| wasm_bindgen::JsValue::from_str("browser document is unavailable"))?;
    let input = document
        .get_element_by_id("burn-image-reference-input")
        .ok_or_else(|| {
            wasm_bindgen::JsValue::from_str(
                "host is missing #burn-image-reference-input file element",
            )
        })?
        .dyn_into::<web_sys::HtmlElement>()?;
    input.click();
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn complete_browser_download(
    mut ready: MessageReader<ImageDownloadReady>,
    mut panel: ResMut<ImageControlPanelState>,
) {
    for download in ready.read() {
        if download.id != DOWNLOAD_IO_ID {
            continue;
        }
        match trigger_browser_download(download) {
            Ok(()) => panel.notice = format!("Downloaded {}", download.file_name),
            Err(error) => panel.notice = format!("Browser download failed: {error:?}"),
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn trigger_browser_download(download: &ImageDownloadReady) -> Result<(), wasm_bindgen::JsValue> {
    use wasm_bindgen::JsCast;

    let bytes = js_sys::Uint8Array::from(download.bytes.as_slice());
    let parts = js_sys::Array::new();
    parts.push(&bytes);
    let options = web_sys::BlobPropertyBag::new();
    options.set_type(download.mime_type);
    let blob = web_sys::Blob::new_with_u8_array_sequence_and_options(&parts, &options)?;
    let url = web_sys::Url::create_object_url_with_blob(&blob)?;
    let document = web_sys::window()
        .and_then(|window| window.document())
        .ok_or_else(|| wasm_bindgen::JsValue::from_str("browser document is unavailable"))?;
    let anchor = document
        .create_element("a")?
        .dyn_into::<web_sys::HtmlAnchorElement>()?;
    anchor.set_href(&url);
    anchor.set_download(&download.file_name);
    let body = document
        .body()
        .ok_or_else(|| wasm_bindgen::JsValue::from_str("browser body is unavailable"))?;
    body.append_child(&anchor)?;
    anchor.click();
    anchor.remove();
    web_sys::Url::revoke_object_url(&url)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[cfg(any(target_arch = "wasm32", test))]
    use super::BROWSER_UI_CONTRACT_EVENT_NAME;
    use super::{
        MIN_VIEWER_HEIGHT, can_change_mode, can_cycle_models, descriptor_for_mode,
        event_progress_presentation, format_progress, image_control_panel_layout,
        model_control_label, next_model_descriptor, next_supported_size_index,
        preferred_size_index, preset_dimensions, preset_index, runner_progress_presentation,
        runner_state_label, setup_progress_fraction,
    };
    #[cfg(feature = "boogu")]
    use super::{apply_descriptor_size, next_supported_size_index_for_descriptor};
    use bevy::{prelude::Vec2, ui::InteractionDisabled};
    use burn_image::{
        DimensionConstraints, Dimensions, ImageTaskKind, ModelId, ProgressEvent, RunId,
    };

    fn runner_with_models(models: &[(&str, &str, &[ImageTaskKind])]) -> crate::ImageRunnerState {
        let mut capabilities = crate::runner::tests::test_capabilities(models[0].0);
        capabilities.models = models
            .iter()
            .map(|(id, display_name, tasks)| {
                let mut descriptor = crate::runner::tests::test_capabilities(id)
                    .models
                    .into_iter()
                    .next()
                    .unwrap();
                descriptor.display_name = (*display_name).into();
                descriptor.capabilities.tasks = tasks.iter().copied().collect();
                descriptor.capabilities.supports_masks =
                    descriptor.capabilities.tasks.contains(&ImageTaskKind::Edit);
                descriptor
            })
            .collect();
        crate::ImageRunnerState::Ready { capabilities }
    }

    fn dimensions(minimum: u32, maximum: u32, max_pixels: Option<u64>) -> DimensionConstraints {
        DimensionConstraints {
            min_width: minimum,
            max_width: maximum,
            min_height: minimum,
            max_height: maximum,
            width_multiple: 32,
            height_multiple: 32,
            max_pixels,
            allowed_dimensions: None,
        }
    }

    #[test]
    fn progress_text_keeps_stage_and_step_correctness() {
        let event = ProgressEvent::Step {
            run_id: RunId(1),
            stage: "denoise".into(),
            step: 3,
            total_steps: 4,
            elapsed_micros: 5,
        };
        assert_eq!(format_progress(&event), "denoise: 3/4");
        let presentation = event_progress_presentation("Job 1", &event);
        assert_eq!(presentation.headline, "Job 1: Running denoise");
        assert_eq!(presentation.detail, "Step 3 of 4");
        assert_eq!(presentation.fraction, Some(0.75));
    }

    #[test]
    fn model_setup_progress_is_determinate_when_numbered_correctness() {
        assert_eq!(
            setup_progress_fraction("Model setup 2/3: loading the denoiser"),
            Some(2.0 / 3.0)
        );
        assert_eq!(setup_progress_fraction("Loading model artifacts"), None);

        let state = crate::ImageRunnerState::Initializing {
            message: "Model setup 1/3: loading Qwen".into(),
        };
        assert_eq!(
            runner_progress_presentation(&state).fraction,
            Some(1.0 / 3.0)
        );
    }

    #[test]
    fn controls_layout_preserves_viewer_on_wide_and_narrow_windows_correctness() {
        let desktop = image_control_panel_layout(Vec2::new(1_280.0, 800.0));
        assert!(!desktop.narrow);
        assert_eq!(desktop.panel_width, 360.0);
        assert!(desktop.viewer_width > desktop.panel_width);
        assert!(desktop.viewer_height > 700.0);

        let narrow = image_control_panel_layout(Vec2::new(600.0, 800.0));
        assert!(narrow.narrow);
        assert!(narrow.panel_height >= 260.0);
        assert!(narrow.viewer_width > 500.0);
        assert!(narrow.viewer_height > 250.0);

        let short = image_control_panel_layout(Vec2::new(600.0, 360.0));
        assert!(short.narrow);
        assert!(short.viewer_height >= MIN_VIEWER_HEIGHT);
        assert!(short.panel_height > 100.0);
    }

    #[test]
    fn authored_control_source_uses_default_font_safe_ascii_correctness() {
        assert!(
            include_str!("controls.rs").is_ascii(),
            "authored control text must stay within the default Bevy font's ASCII-safe subset"
        );
    }

    #[test]
    fn rendered_model_smoke_uses_real_browser_input_and_download_correctness() {
        let contract = include_str!("../tests/wasm_rendered_surface_contract.mjs");
        let harness = include_str!("../tests/wasm_rendered_surface_smoke.mjs");
        assert!(contract.contains(BROWSER_UI_CONTRACT_EVENT_NAME));
        assert!(harness.contains("UI_CONTRACT_EVENT_NAME"));
        assert!(harness.contains("Input.dispatchMouseEvent"));
        assert!(harness.contains("Input.dispatchKeyEvent"));
        assert!(harness.contains("Input.dispatchKeyEvent(per-character text)"));
        assert!(!harness.contains("Input.insertText"));
        assert!(harness.contains("Browser.setDownloadBehavior"));
        assert!(harness.contains("Browser.downloadProgress"));
        assert!(!harness.contains("__burnImageDriveModelSmoke"));

        let manifest = include_str!("../Cargo.toml");
        let app_features = manifest
            .split_once("app = [")
            .and_then(|(_, suffix)| suffix.split_once(']'))
            .map(|(features, _)| features)
            .expect("bevy_image app feature must remain a literal array");
        assert!(app_features.contains("\"bevy/bevy_picking\""));
        assert!(app_features.contains("\"bevy/ui_picking\""));
    }

    #[test]
    fn prompt_input_remains_unicode_capable_correctness() {
        let prompt = "\u{732b}\u{3068}\u{6708} - caf\u{e9}";
        let editor = crate::ImageEditorState {
            prompt_or_instruction: prompt.into(),
            ..Default::default()
        };
        let burn_image::ImageRequest::Generate(request) = editor.build_request().unwrap() else {
            panic!("default editor mode must generate");
        };
        assert_eq!(request.prompt.as_str(), prompt);
    }

    #[test]
    fn missing_runtime_label_does_not_claim_generation_correctness() {
        assert_eq!(
            runner_state_label(&crate::ImageRunnerState::Missing),
            "No model runtime installed"
        );
        let _ = ModelId::new("test/model").unwrap();
    }

    #[test]
    fn singleton_runtime_labels_and_disables_model_selection_correctness() {
        let runner =
            runner_with_models(&[("test/turbo", "Test Turbo", &[ImageTaskKind::Generate])]);
        let editor = crate::ImageEditorState {
            model: Some(ModelId::new("test/turbo").unwrap()),
            ..Default::default()
        };

        assert_eq!(
            model_control_label(&editor, &runner),
            "Loaded model: Test Turbo"
        );
        assert!(!can_cycle_models(&runner));
        let crate::ImageRunnerState::Ready { capabilities } = &runner else {
            unreachable!();
        };
        assert!(next_model_descriptor(&capabilities.models, editor.model.as_ref()).is_none());
    }

    #[test]
    fn singleton_runtime_model_button_is_disabled_in_the_ecs_correctness() {
        let runner =
            runner_with_models(&[("test/turbo", "Test Turbo", &[ImageTaskKind::Generate])]);
        let mut app = bevy::prelude::App::new();
        app.insert_resource(crate::ImageRunnerStatus { state: runner })
            .insert_resource(crate::ImageEditorState {
                model: Some(ModelId::new("test/turbo").unwrap()),
                ..Default::default()
            })
            .init_resource::<crate::ImageJobs>()
            .init_resource::<super::ImageControlPanelState>()
            .add_systems(bevy::prelude::Update, super::update_action_availability);
        let button = app.world_mut().spawn(super::ModelButton).id();

        app.update();

        assert!(app.world().get::<InteractionDisabled>(button).is_some());
    }

    #[test]
    fn genuine_multi_model_runtime_keeps_model_cycling_correctness() {
        let runner = runner_with_models(&[
            ("test/turbo", "Test Turbo", &[ImageTaskKind::Generate]),
            ("test/edit", "Test Edit", &[ImageTaskKind::Edit]),
        ]);
        let editor = crate::ImageEditorState {
            model: Some(ModelId::new("test/turbo").unwrap()),
            ..Default::default()
        };

        assert_eq!(model_control_label(&editor, &runner), "Model: Test Turbo");
        assert!(can_cycle_models(&runner));
        let crate::ImageRunnerState::Ready { capabilities } = &runner else {
            unreachable!();
        };
        assert_eq!(
            next_model_descriptor(&capabilities.models, editor.model.as_ref())
                .unwrap()
                .id
                .as_str(),
            "test/edit"
        );
    }

    #[test]
    fn mode_selection_never_enters_an_unsupported_task_correctness() {
        let generation_only =
            runner_with_models(&[("test/turbo", "Test Turbo", &[ImageTaskKind::Generate])]);
        assert!(!can_change_mode(
            &generation_only,
            crate::EditorMode::Generate
        ));
        assert!(descriptor_for_mode(&generation_only, crate::EditorMode::Edit).is_none());

        let multi = runner_with_models(&[
            ("test/turbo", "Test Turbo", &[ImageTaskKind::Generate]),
            ("test/edit", "Test Edit", &[ImageTaskKind::Edit]),
        ]);
        assert!(can_change_mode(&multi, crate::EditorMode::Generate));
        assert_eq!(
            descriptor_for_mode(&multi, crate::EditorMode::Edit)
                .unwrap()
                .id
                .as_str(),
            "test/edit"
        );
    }

    #[test]
    fn unsupported_mode_press_keeps_the_editor_dispatchable_correctness() {
        let runner =
            runner_with_models(&[("test/turbo", "Test Turbo", &[ImageTaskKind::Generate])]);
        let mut app = bevy::prelude::App::new();
        app.insert_resource(crate::ImageRunnerStatus { state: runner })
            .insert_resource(crate::ImageEditorState {
                model: Some(ModelId::new("test/turbo").unwrap()),
                ..Default::default()
            })
            .init_resource::<super::ImageControlPanelState>()
            .add_systems(bevy::prelude::Update, super::handle_mode_button);
        app.world_mut()
            .spawn((super::ModeButton, bevy::prelude::Interaction::Pressed));

        app.update();

        assert_eq!(
            app.world().resource::<crate::ImageEditorState>().mode,
            crate::EditorMode::Generate
        );
        assert_eq!(
            app.world()
                .resource::<super::ImageControlPanelState>()
                .notice,
            "The loaded runtime does not support Edit mode"
        );
    }

    #[test]
    fn preferred_size_uses_512_when_the_model_supports_it_correctness() {
        let constraints = dimensions(256, 1_024, Some(1_024 * 1_024));
        let index = preferred_size_index(&constraints).unwrap();
        assert_eq!(preset_dimensions(index), Dimensions::new(512, 512).unwrap());
    }

    #[test]
    fn exact_256_model_initializes_and_cycles_only_256_correctness() {
        let constraints = dimensions(256, 256, Some(256 * 256));
        let initial = preferred_size_index(&constraints).unwrap();
        assert_eq!(
            preset_dimensions(initial),
            Dimensions::new(256, 256).unwrap()
        );
        let next = next_supported_size_index(initial, &constraints).unwrap();
        assert_eq!(next, initial);
        assert_eq!(preset_index(Dimensions::new(512, 512).unwrap()), Some(1));
        assert!(
            constraints
                .supports(Dimensions::new(512, 512).unwrap())
                .is_err()
        );
    }

    #[cfg(feature = "boogu")]
    #[test]
    fn turbo_controls_start_at_core_1k_default_correctness() {
        for variant in [
            burn_boogu::BooguVariant::Image01Turbo,
            burn_boogu::BooguVariant::Image01EditTurbo,
        ] {
            let descriptor = burn_boogu::boogu_model_descriptor(variant);
            let mut editor = crate::ImageEditorState::default();
            let mut panel = super::ImageControlPanelState::default();
            apply_descriptor_size(&descriptor, &mut editor, &mut panel);

            assert_eq!(
                editor.options.dimensions,
                Some(Dimensions::new(1024, 1024).unwrap())
            );
        }
    }

    #[cfg(feature = "boogu")]
    #[test]
    fn edit_turbo_1k5_controls_start_at_released_default_and_expose_presets_correctness() {
        let descriptor =
            burn_boogu::boogu_model_descriptor(burn_boogu::BooguVariant::Image01EditTurbo1k5);
        let mut editor = crate::ImageEditorState::default();
        editor.options.dimensions = Some(Dimensions::new(512, 512).unwrap());
        let mut panel = super::ImageControlPanelState::default();
        apply_descriptor_size(&descriptor, &mut editor, &mut panel);

        assert_eq!(
            editor.options.dimensions,
            Some(Dimensions::new(1536, 1536).unwrap())
        );
        for &(width, height) in &super::SIZE_PRESETS[6..] {
            assert!(
                descriptor
                    .capabilities
                    .dimensions
                    .supports(Dimensions::new(width, height).unwrap())
                    .is_ok(),
                "released 1.5K preset {width}x{height} must be selectable"
            );
        }
        assert_eq!(
            &super::SIZE_PRESETS[6..],
            burn_boogu::BOOGU_1K5_OUTPUT_PRESETS.as_slice()
        );
        let initial = super::preset_index(Dimensions::new(1536, 1536).unwrap()).unwrap();
        let next = next_supported_size_index_for_descriptor(initial, &descriptor).unwrap();
        assert_eq!(
            super::preset_dimensions(next),
            Dimensions::new(1264, 1856).unwrap()
        );
    }

    #[cfg(feature = "boogu")]
    #[test]
    fn switching_from_1k5_to_turbo_restores_the_core_1k_default_correctness() {
        let descriptor = burn_boogu::boogu_model_descriptor(burn_boogu::BooguVariant::Image01Turbo);
        let mut editor = crate::ImageEditorState::default();
        editor.options.dimensions = Some(Dimensions::new(1536, 1536).unwrap());
        let mut panel = super::ImageControlPanelState::default();

        apply_descriptor_size(&descriptor, &mut editor, &mut panel);

        assert_eq!(
            editor.options.dimensions,
            Some(Dimensions::new(1024, 1024).unwrap())
        );
        assert_eq!(
            super::preset_dimensions(panel.size_index),
            Dimensions::new(1024, 1024).unwrap()
        );
    }
}
