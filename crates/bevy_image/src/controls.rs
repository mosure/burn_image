//! Usable model-neutral Bevy controls for generation and editing.

#[cfg(target_arch = "wasm32")]
use bevy::input_focus::InputFocus;
use bevy::{
    input::mouse::{MouseScrollUnit, MouseWheel},
    picking::Pickable,
    prelude::*,
    text::{EditableText, EditableTextFilter, LineHeight, TextCursorStyle, TextEdit},
    ui::InteractionDisabled,
    window::PrimaryWindow,
};
#[cfg(test)]
use burn_image::DimensionConstraints;
use burn_image::{
    ArtifactTransferProgress, Dimensions, HostImage, ImageEncoding, ImageTaskKind, InputImage,
    ModelDescriptor, ProgressEvent,
};

use crate::{
    CancelImageJob, CompleteImageJob, EditorMode, ImageBytesLoaded, ImageDisplayFailed,
    ImageEditorState, ImageFrontendSet, ImageIoFailed, ImageJobId, ImageJobPhase, ImageJobRejected,
    ImageJobs, ImageRunnerReadiness, ImageRunnerState, ImageRunnerStatus, LoadImageBytes,
    MODEL_SWITCH_PROGRESS_STAGE_PREFIX, PrepareImageDownload, REFERENCE_IMAGE_IO_ID,
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
const DESKTOP_PANEL_WIDTH: f32 = 380.0;
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

#[derive(Resource, Default)]
struct SizeDropdownState {
    open: bool,
}

#[derive(Resource, Default)]
struct ModelDropdownState {
    open: bool,
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
struct ModelButton;
#[derive(Component, Default)]
struct ModelDropdown;
#[derive(Component, Clone, Copy)]
struct ModelOption {
    index: usize,
}
#[derive(Component)]
struct ModelOptionLabel;
#[derive(Component, Default)]
struct SizeButton;
#[derive(Component, Default)]
struct SizeDropdown;
#[derive(Component, Clone, Copy)]
struct SizeOption {
    index: usize,
}
#[derive(Component, Default)]
struct SizeDropdownHint;
#[derive(Component, Default)]
struct ReferenceButton;
#[derive(Component, Default)]
struct SelectorAffordance;
#[derive(Component, Default)]
struct RunButton;
#[derive(Component)]
struct RunRequirementsLabel;
#[derive(Component, Default)]
struct CancelButton;
#[derive(Component, Default)]
struct SaveButton;
#[derive(Component, Default)]
struct UseOutputReferenceButton;
#[derive(Component, Default)]
struct UseOutputReferenceAction;
#[derive(Component)]
struct PromptInput;
#[derive(Component)]
struct SeedInput;
#[derive(Component, Default)]
struct RandomSeedButton;
#[derive(Component, Default)]
struct ModelButtonLabel;
#[derive(Component, Default)]
struct SizeButtonLabel;
#[derive(Component, Default)]
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
    hovered: Color,
    pressed: Color,
    disabled: Color,
}

impl ButtonPalette {
    const fn neutral() -> Self {
        Self {
            idle: Color::srgb(0.14, 0.18, 0.27),
            hovered: Color::srgb(0.19, 0.25, 0.38),
            pressed: Color::srgb(0.24, 0.32, 0.48),
            disabled: Color::srgb(0.085, 0.095, 0.12),
        }
    }

    const fn action(idle: Color, hovered: Color) -> Self {
        Self {
            idle,
            hovered,
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
            .init_resource::<ImageRunnerReadiness>()
            .init_resource::<ModelDropdownState>()
            .init_resource::<SizeDropdownState>()
            .add_systems(Startup, setup_controls)
            .add_systems(
                Update,
                (
                    sync_control_panel_layout,
                    scroll_control_panel,
                    select_initial_model,
                    sync_text_inputs,
                    handle_model_button,
                    handle_size_button,
                    handle_reference_button,
                    handle_use_output_reference_button,
                    accept_native_file_dialog,
                    handle_run_button,
                    handle_cancel_button,
                    handle_save_button,
                    accept_reference_images,
                    capture_frontend_errors,
                    update_control_labels,
                    update_progress_panel,
                    update_action_availability,
                    update_control_affordances,
                    update_button_colors,
                )
                    .chain(),
            )
            .add_systems(
                Update,
                (
                    handle_random_seed_button
                        .after(sync_text_inputs)
                        .before(update_action_availability),
                    handle_model_option.after(handle_model_button),
                    close_model_dropdown_on_outside_click.after(handle_model_option),
                    sync_model_dropdown
                        .after(update_action_availability)
                        .after(close_model_dropdown_on_outside_click)
                        .before(update_control_affordances),
                    handle_size_option.after(handle_size_button),
                    close_size_dropdown_on_outside_click.after(handle_size_option),
                    sync_size_dropdown
                        .after(update_action_availability)
                        .after(close_size_dropdown_on_outside_click)
                        .before(update_control_affordances),
                    sync_run_requirements.after(update_action_availability),
                    sync_output_action_visibility
                        .after(update_action_availability)
                        .after(capture_outputs),
                ),
            )
            .add_systems(
                Update,
                sync_reference_control_visibility
                    .after(handle_size_button)
                    .before(handle_reference_button),
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
                padding: px(16).all(),
                row_gap: px(10),
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
                Text::new("CREATE / EDIT"),
                TextFont::from_font_size(16.0),
                TextColor(Color::srgb(0.78, 0.86, 1.0)),
                LineHeight::RelativeToFont(1.2),
            ));

            spawn_labeled_button::<ModelButton, ModelButtonLabel>(
                panel,
                "Model",
                "waiting...",
                "v",
            );
            spawn_model_dropdown(panel);

            panel.spawn((
                Text::new("PROMPT / INSTRUCTION"),
                TextFont::from_font_size(11.0),
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
                TextFont::from_font_size(14.0),
                TextColor(Color::WHITE),
                TextCursorStyle::default(),
                TextLayout {
                    linebreak: LineBreak::WordOrCharacter,
                    ..default()
                },
                LineHeight::RelativeToFont(1.35),
                Node {
                    width: percent(100),
                    min_height: px(108),
                    padding: px(10).all(),
                    border: px(1).all(),
                    ..default()
                },
                BorderColor::all(Color::srgb(0.26, 0.3, 0.4)),
                BackgroundColor(Color::srgb(0.09, 0.105, 0.14)),
            ));

            spawn_labeled_button::<SizeButton, SizeButtonLabel>(
                panel,
                "Size",
                "model default",
                "v",
            );
            spawn_size_dropdown(panel);

            panel
                .spawn(Node {
                    width: percent(100),
                    column_gap: px(8),
                    align_items: AlignItems::Center,
                    ..default()
                })
                .with_children(|row| {
                    row.spawn((
                        Text::new("SEED"),
                        TextFont::from_font_size(11.0),
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
                        TextFont::from_font_size(14.0),
                        TextColor(Color::WHITE),
                        TextCursorStyle::default(),
                        TextLayout::no_wrap(),
                        Node {
                            flex_grow: 1.0,
                            height: px(28),
                            padding: UiRect::axes(px(8), px(3)),
                            border: px(1).all(),
                            border_radius: BorderRadius::all(px(4)),
                            ..default()
                        },
                        BorderColor::all(Color::srgb(0.26, 0.3, 0.4)),
                        BackgroundColor(Color::srgb(0.09, 0.105, 0.14)),
                    ));
                    let palette = ButtonPalette::neutral();
                    row.spawn((
                        Button,
                        RandomSeedButton,
                        Node {
                            width: px(82),
                            min_height: px(34),
                            align_items: AlignItems::Center,
                            justify_content: JustifyContent::Center,
                            padding: UiRect::axes(px(8), px(6)),
                            border_radius: BorderRadius::all(px(4)),
                            flex_shrink: 0.0,
                            ..default()
                        },
                        palette,
                        BackgroundColor(palette.idle),
                    ))
                    .with_child((
                        Text::new("Random"),
                        TextFont::from_font_size(11.0),
                        TextColor(Color::WHITE),
                        Pickable::IGNORE,
                    ));
                });

            spawn_labeled_button::<ReferenceButton, ReferenceLabel>(
                panel,
                "Reference",
                reference_button_text(),
                ">",
            );

            panel
                .spawn((
                    Node {
                        display: Display::None,
                        width: percent(100),
                        ..default()
                    },
                    UseOutputReferenceAction,
                ))
                .with_children(|row| {
                    spawn_action_button::<UseOutputReferenceButton>(
                        row,
                        "Use output as reference",
                        Color::srgb(0.18, 0.35, 0.5),
                        Color::srgb(0.24, 0.47, 0.66),
                    );
                });

            panel.spawn((
                Text::new(""),
                TextFont::from_font_size(10.5),
                TextColor(Color::srgb(0.82, 0.67, 0.36)),
                TextLayout {
                    linebreak: LineBreak::WordOrCharacter,
                    ..default()
                },
                LineHeight::RelativeToFont(1.3),
                Visibility::Hidden,
                RunRequirementsLabel,
            ));

            panel
                .spawn(Node {
                    width: percent(100),
                    column_gap: px(7),
                    ..default()
                })
                .with_children(|row| {
                    spawn_action_button::<RunButton>(
                        row,
                        "Run",
                        Color::srgb(0.15, 0.42, 0.75),
                        Color::srgb(0.2, 0.53, 0.9),
                    );
                    spawn_action_button::<CancelButton>(
                        row,
                        "Cancel",
                        Color::srgb(0.54, 0.2, 0.22),
                        Color::srgb(0.68, 0.28, 0.31),
                    );
                    spawn_action_button::<SaveButton>(
                        row,
                        "Save PNG",
                        Color::srgb(0.18, 0.42, 0.3),
                        Color::srgb(0.24, 0.55, 0.39),
                    );
                });

            panel
                .spawn((
                    Node {
                        width: percent(100),
                        padding: px(10).all(),
                        row_gap: px(7),
                        flex_direction: FlexDirection::Column,
                        border_radius: BorderRadius::all(px(7)),
                        ..default()
                    },
                    BackgroundColor(Color::srgb(0.075, 0.09, 0.12)),
                ))
                .with_children(|status| {
                    status.spawn((
                        Text::new("Preparing model runtime"),
                        TextFont::from_font_size(12.0),
                        TextColor(Color::srgb(0.84, 0.88, 0.94)),
                        TextLayout {
                            linebreak: LineBreak::WordOrCharacter,
                            ..default()
                        },
                        LineHeight::RelativeToFont(1.3),
                        ProgressLabel,
                    ));

                    status
                        .spawn((
                            Node {
                                position_type: PositionType::Relative,
                                width: percent(100),
                                height: px(6),
                                overflow: Overflow::clip(),
                                border_radius: BorderRadius::all(px(3)),
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
                                    border_radius: BorderRadius::all(px(3)),
                                    ..default()
                                },
                                BackgroundColor(Color::srgb(0.32, 0.68, 0.83)),
                                ProgressFill,
                            ));
                        });

                    status.spawn((
                        Text::new("Waiting for the shared GPU"),
                        TextFont::from_font_size(10.5),
                        TextColor(Color::srgb(0.62, 0.68, 0.76)),
                        TextLayout {
                            linebreak: LineBreak::WordOrCharacter,
                            ..default()
                        },
                        LineHeight::RelativeToFont(1.35),
                        ProgressDetailLabel,
                    ));
                });
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
    affordance: &str,
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
                Text::new(caption.to_ascii_uppercase()),
                TextFont::from_font_size(10.0),
                TextColor(Color::srgb(0.58, 0.65, 0.75)),
                Pickable::IGNORE,
                Node {
                    width: px(68),
                    flex_shrink: 0.0,
                    ..default()
                },
            ));
            button.spawn((
                Text::new(value),
                TextFont::from_font_size(13.0),
                TextColor(Color::WHITE),
                TextLayout {
                    linebreak: LineBreak::WordOrCharacter,
                    ..default()
                },
                LineHeight::RelativeToFont(1.25),
                Pickable::IGNORE,
                Node {
                    min_width: px(0),
                    flex_grow: 1.0,
                    ..default()
                },
                L::default(),
            ));
            button.spawn((
                Text::new(affordance),
                TextFont::from_font_size(13.0),
                TextColor(Color::srgb(0.58, 0.65, 0.75)),
                Pickable::IGNORE,
                Visibility::Inherited,
                SelectorAffordance,
            ));
        });
}

fn spawn_size_dropdown(panel: &mut ChildSpawnerCommands) {
    panel
        .spawn((
            Node {
                display: Display::None,
                width: percent(100),
                padding: px(6).all(),
                row_gap: px(6),
                column_gap: px(6),
                flex_wrap: FlexWrap::Wrap,
                border: px(1).all(),
                border_radius: BorderRadius::all(px(6)),
                ..default()
            },
            BorderColor::all(Color::srgb(0.28, 0.34, 0.46)),
            BackgroundColor(Color::srgb(0.07, 0.085, 0.12)),
            SizeDropdown,
        ))
        .with_children(|dropdown| {
            for (index, (width, height)) in SIZE_PRESETS.iter().copied().enumerate() {
                let palette = ButtonPalette::neutral();
                dropdown
                    .spawn((
                        Button,
                        SizeOption { index },
                        Node {
                            display: Display::Flex,
                            flex_grow: 1.0,
                            flex_basis: percent(46),
                            min_width: px(132),
                            min_height: px(32),
                            align_items: AlignItems::Center,
                            justify_content: JustifyContent::Center,
                            padding: UiRect::axes(px(8), px(6)),
                            border_radius: BorderRadius::all(px(4)),
                            ..default()
                        },
                        palette,
                        BackgroundColor(palette.idle),
                    ))
                    .with_child((
                        Text::new(format!("{width} x {height}")),
                        TextFont::from_font_size(12.0),
                        TextColor(Color::WHITE),
                        Pickable::IGNORE,
                    ));
            }
            dropdown.spawn((
                Text::new(""),
                TextFont::from_font_size(10.0),
                TextColor(Color::srgb(0.68, 0.72, 0.8)),
                TextLayout {
                    linebreak: LineBreak::WordOrCharacter,
                    ..default()
                },
                LineHeight::RelativeToFont(1.25),
                Pickable::IGNORE,
                Visibility::Hidden,
                Node {
                    width: percent(100),
                    flex_basis: percent(100),
                    ..default()
                },
                SizeDropdownHint,
            ));
        });
}

fn spawn_model_dropdown(panel: &mut ChildSpawnerCommands) {
    panel
        .spawn((
            Node {
                display: Display::None,
                width: percent(100),
                padding: px(6).all(),
                row_gap: px(6),
                flex_direction: FlexDirection::Column,
                border: px(1).all(),
                border_radius: BorderRadius::all(px(6)),
                ..default()
            },
            BorderColor::all(Color::srgb(0.28, 0.34, 0.46)),
            BackgroundColor(Color::srgb(0.07, 0.085, 0.12)),
            ModelDropdown,
        ))
        .with_children(|dropdown| {
            for index in 0..3 {
                let palette = ButtonPalette::neutral();
                dropdown
                    .spawn((
                        Button,
                        ModelOption { index },
                        Node {
                            display: Display::None,
                            width: percent(100),
                            min_height: px(34),
                            align_items: AlignItems::Center,
                            padding: UiRect::axes(px(10), px(7)),
                            border_radius: BorderRadius::all(px(4)),
                            ..default()
                        },
                        palette,
                        BackgroundColor(palette.idle),
                    ))
                    .with_child((
                        Text::new(""),
                        TextFont::from_font_size(12.0),
                        TextColor(Color::WHITE),
                        Pickable::IGNORE,
                        ModelOptionLabel,
                    ));
            }
        });
}

fn spawn_action_button<M: Component + Default>(
    row: &mut ChildSpawnerCommands,
    label: &str,
    color: Color,
    hovered: Color,
) {
    let palette = ButtonPalette::action(color, hovered);
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
            Pickable::IGNORE,
        ));
    });
}

fn control_button_node() -> Node {
    Node {
        width: percent(100),
        min_height: px(42),
        align_items: AlignItems::Center,
        justify_content: JustifyContent::FlexStart,
        column_gap: px(8),
        padding: UiRect::axes(px(10), px(8)),
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
        if let Some(mode) = descriptor_mode(descriptor) {
            editor.mode = mode;
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

#[allow(clippy::type_complexity)]
fn handle_random_seed_button(
    interactions: Query<
        &Interaction,
        (
            Changed<Interaction>,
            With<RandomSeedButton>,
            Without<InteractionDisabled>,
        ),
    >,
    mut seeds: Query<&mut EditableText, With<SeedInput>>,
    mut editor: ResMut<ImageEditorState>,
    mut panel: ResMut<ImageControlPanelState>,
) {
    if !interactions
        .iter()
        .any(|interaction| *interaction == Interaction::Pressed)
    {
        return;
    }
    let value = distinct_random_seed(editor.options.seed, rand::random());
    if let Ok(mut seed) = seeds.single_mut() {
        seed.clear();
        seed.editor_mut().set_text(&value.to_string());
        seed.queue_edit(TextEdit::TextEnd(false));
    }
    editor.options.seed = Some(value);
    panel.seed_valid = true;
    panel.notice = format!("Random seed: {value}");
    #[cfg(target_arch = "wasm32")]
    if browser_model_smoke_requested() {
        dispatch_browser_text_value("seed_changed", &value.to_string());
    }
}

fn distinct_random_seed(current: Option<u64>, candidate: u64) -> u64 {
    if current == Some(candidate) {
        candidate.wrapping_add(1)
    } else {
        candidate
    }
}

fn handle_model_button(
    interactions: Query<&Interaction, (Changed<Interaction>, With<ModelButton>)>,
    mut dropdown: ResMut<ModelDropdownState>,
    mut size_dropdown: ResMut<SizeDropdownState>,
) {
    if !interactions
        .iter()
        .any(|interaction| *interaction == Interaction::Pressed)
    {
        return;
    }
    dropdown.open = !dropdown.open;
    size_dropdown.open = false;
}

#[cfg(test)]
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

fn descriptor_for_mode_prefer_model<'a>(
    state: &'a ImageRunnerState,
    mode: EditorMode,
    preferred: Option<&burn_image::ModelId>,
) -> Option<&'a ModelDescriptor> {
    let ImageRunnerState::Ready { capabilities } = state else {
        return None;
    };
    let task = editor_mode_task(mode);
    preferred
        .and_then(|model| capabilities.descriptor(model))
        .filter(|descriptor| descriptor.capabilities.tasks.contains(&task))
        .or_else(|| descriptor_for_mode(state, mode))
}

const fn editor_mode_task(mode: EditorMode) -> ImageTaskKind {
    match mode {
        EditorMode::Generate => ImageTaskKind::Generate,
        EditorMode::Edit => ImageTaskKind::Edit,
    }
}

fn descriptor_mode(descriptor: &ModelDescriptor) -> Option<EditorMode> {
    let generate = descriptor
        .capabilities
        .tasks
        .contains(&ImageTaskKind::Generate);
    let edit = descriptor.capabilities.tasks.contains(&ImageTaskKind::Edit);
    match (generate, edit) {
        (true, _) => Some(EditorMode::Generate),
        (false, true) => Some(EditorMode::Edit),
        (false, false) => None,
    }
}

fn handle_size_button(
    interactions: Query<&Interaction, (Changed<Interaction>, With<SizeButton>)>,
    mut dropdown: ResMut<SizeDropdownState>,
    mut model_dropdown: ResMut<ModelDropdownState>,
) {
    if !interactions
        .iter()
        .any(|interaction| *interaction == Interaction::Pressed)
    {
        return;
    }
    dropdown.open = !dropdown.open;
    model_dropdown.open = false;
}

#[allow(clippy::type_complexity)]
fn handle_model_option(
    interactions: Query<
        (&Interaction, &ModelOption),
        (Changed<Interaction>, With<Button>, Without<ModelButton>),
    >,
    status: Res<ImageRunnerStatus>,
    mut editor: ResMut<ImageEditorState>,
    mut panel: ResMut<ImageControlPanelState>,
    mut dropdown: ResMut<ModelDropdownState>,
) {
    let Some(option) = interactions
        .iter()
        .find_map(|(interaction, option)| (*interaction == Interaction::Pressed).then_some(option))
    else {
        return;
    };
    let ImageRunnerState::Ready { capabilities } = &status.state else {
        dropdown.open = false;
        return;
    };
    let Some(descriptor) = capabilities.models.get(option.index) else {
        dropdown.open = false;
        return;
    };
    editor.model = Some(descriptor.id.clone());
    if let Some(mode) = descriptor_mode(descriptor) {
        editor.mode = mode;
    }
    apply_descriptor_size(descriptor, &mut editor, &mut panel);
    #[cfg(all(target_arch = "wasm32", feature = "boogu-web"))]
    match crate::browser_boogu::request_browser_model_release(&descriptor.id) {
        Ok(true) => {
            panel.notice = format!(
                "Switching to {}; the previous browser model is unloading",
                descriptor.display_name
            );
            dropdown.open = false;
            return;
        }
        Ok(false) => {}
        Err(error) => {
            panel.notice = error;
            dropdown.open = false;
            return;
        }
    }
    panel.notice = format!(
        "{} selected; it will load on the next Run",
        descriptor.display_name
    );
    dropdown.open = false;
}

fn close_model_dropdown_on_outside_click(
    mouse: Res<ButtonInput<MouseButton>>,
    model_buttons: Query<&Interaction, With<ModelButton>>,
    model_options: Query<&Interaction, (With<ModelOption>, Without<ModelButton>)>,
    mut dropdown: ResMut<ModelDropdownState>,
) {
    if !dropdown.open || !mouse.just_pressed(MouseButton::Left) {
        return;
    }
    let clicked_selector = model_buttons
        .iter()
        .chain(model_options.iter())
        .any(|interaction| *interaction == Interaction::Pressed);
    if !clicked_selector {
        dropdown.open = false;
    }
}

fn sync_model_dropdown(
    mut dropdown_state: ResMut<ModelDropdownState>,
    runner: Res<ImageRunnerStatus>,
    editor: Res<ImageEditorState>,
    jobs: Res<ImageJobs>,
    mut dropdowns: Query<&mut Node, (With<ModelDropdown>, Without<ModelOption>)>,
    mut options: Query<(&ModelOption, &mut Node, &mut ButtonPalette), Without<ModelDropdown>>,
    mut labels: Query<(&ChildOf, &mut Text), With<ModelOptionLabel>>,
) {
    let running = jobs.iter().any(|job| !job.phase.is_terminal());
    let capabilities = match &runner.state {
        ImageRunnerState::Ready { capabilities } => Some(capabilities),
        _ => None,
    };
    if running || capabilities.is_none_or(|capabilities| capabilities.models.len() < 2) {
        dropdown_state.open = false;
    }
    let display = if dropdown_state.open {
        Display::Flex
    } else {
        Display::None
    };
    for mut node in &mut dropdowns {
        node.display = display;
    }
    for (option, mut node, mut palette) in &mut options {
        let descriptor =
            capabilities.and_then(|capabilities| capabilities.models.get(option.index));
        let supported = descriptor.is_some();
        node.display = if supported {
            Display::Flex
        } else {
            Display::None
        };
        palette.idle =
            if descriptor.is_some_and(|descriptor| editor.model.as_ref() == Some(&descriptor.id)) {
                Color::srgb(0.2, 0.34, 0.54)
            } else {
                ButtonPalette::neutral().idle
            };
    }
    for (parent, mut text) in &mut labels {
        let label = options
            .get(parent.parent())
            .ok()
            .and_then(|(option, _, _)| {
                capabilities.and_then(|capabilities| capabilities.models.get(option.index))
            })
            .map(|descriptor| descriptor.display_name.as_str())
            .unwrap_or_default();
        if text.0 != label {
            text.0 = label.into();
        }
    }
}

#[allow(clippy::type_complexity)]
fn handle_size_option(
    interactions: Query<
        (&Interaction, &SizeOption),
        (Changed<Interaction>, With<Button>, Without<SizeButton>),
    >,
    status: Res<ImageRunnerStatus>,
    mut editor: ResMut<ImageEditorState>,
    mut panel: ResMut<ImageControlPanelState>,
    mut dropdown: ResMut<SizeDropdownState>,
) {
    let Some(option) = interactions
        .iter()
        .find_map(|(interaction, option)| (*interaction == Interaction::Pressed).then_some(option))
    else {
        return;
    };
    let ImageRunnerState::Ready { capabilities } = &status.state else {
        dropdown.open = false;
        return;
    };
    let Some(descriptor) = editor.model.as_ref().and_then(|model| {
        capabilities
            .models
            .iter()
            .find(|descriptor| descriptor.id == *model)
    }) else {
        dropdown.open = false;
        return;
    };
    if option.index < SIZE_PRESETS.len()
        && descriptor_supports_dimensions(descriptor, preset_dimensions(option.index))
    {
        panel.size_index = option.index;
        editor.options.dimensions = Some(preset_dimensions(option.index));
    }
    dropdown.open = false;
}

fn close_size_dropdown_on_outside_click(
    mouse: Res<ButtonInput<MouseButton>>,
    size_buttons: Query<&Interaction, With<SizeButton>>,
    size_options: Query<&Interaction, (With<SizeOption>, Without<SizeButton>)>,
    mut dropdown: ResMut<SizeDropdownState>,
) {
    if !dropdown.open || !mouse.just_pressed(MouseButton::Left) {
        return;
    }
    let clicked_selector = size_buttons
        .iter()
        .chain(size_options.iter())
        .any(|interaction| *interaction == Interaction::Pressed);
    if !clicked_selector {
        dropdown.open = false;
    }
}

fn sync_size_dropdown(
    mut dropdown_state: ResMut<SizeDropdownState>,
    runner: Res<ImageRunnerStatus>,
    editor: Res<ImageEditorState>,
    jobs: Res<ImageJobs>,
    mut dropdowns: Query<&mut Node, (With<SizeDropdown>, Without<SizeOption>)>,
    mut options: Query<(&SizeOption, &mut Node, &mut ButtonPalette), Without<SizeDropdown>>,
    mut hints: Query<(&mut Text, &mut Visibility), With<SizeDropdownHint>>,
) {
    let running = jobs.iter().any(|job| !job.phase.is_terminal());
    let descriptor = match &runner.state {
        ImageRunnerState::Ready { capabilities } => editor
            .model
            .as_ref()
            .and_then(|model| capabilities.descriptor(model)),
        _ => None,
    };
    if running || descriptor.is_none() {
        dropdown_state.open = false;
    }
    let display = if dropdown_state.open {
        Display::Flex
    } else {
        Display::None
    };
    for mut node in &mut dropdowns {
        node.display = display;
    }

    let selected = editor.options.dimensions.and_then(preset_index);
    for (option, mut node, mut palette) in &mut options {
        let supported = descriptor.is_some_and(|descriptor| {
            option.index < SIZE_PRESETS.len()
                && descriptor_supports_dimensions(descriptor, preset_dimensions(option.index))
        });
        node.display = if supported {
            Display::Flex
        } else {
            Display::None
        };
        palette.idle = if selected == Some(option.index) {
            Color::srgb(0.2, 0.34, 0.54)
        } else {
            ButtonPalette::neutral().idle
        };
    }

    let hint = descriptor.and_then(size_dropdown_hint);
    for (mut text, mut visibility) in &mut hints {
        let message = hint.unwrap_or_default();
        if text.0 != message {
            text.0 = message.into();
        }
        let next = if hint.is_some() {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        if *visibility != next {
            *visibility = next;
        }
    }
}

fn size_dropdown_hint(_descriptor: &ModelDescriptor) -> Option<&'static str> {
    #[cfg(feature = "boogu")]
    if crate::boogu::variant_for_model(&_descriptor.id)
        == Some(burn_boogu::BooguVariant::Image01EditTurbo)
    {
        return Some("For 1.5K sizes, choose Edit - Turbo 1.5K from the Model menu.");
    }
    None
}

fn sync_reference_control_visibility(
    runner: Res<ImageRunnerStatus>,
    editor: Res<ImageEditorState>,
    panel: Res<ImageControlPanelState>,
    mut reference_buttons: Query<
        &mut Node,
        (With<ReferenceButton>, Without<UseOutputReferenceAction>),
    >,
    mut use_output_actions: Query<
        &mut Node,
        (With<UseOutputReferenceAction>, Without<ReferenceButton>),
    >,
) {
    if !runner.is_changed() && !editor.is_changed() && !panel.is_changed() {
        return;
    }
    let relevant = reference_control_relevant(&editor, &runner.state);
    let reference_display = if relevant {
        Display::Flex
    } else {
        Display::None
    };
    for mut node in &mut reference_buttons {
        if node.display != reference_display {
            node.display = reference_display;
        }
    }
    let use_output_display = if relevant && panel.latest_output.is_some() {
        Display::Flex
    } else {
        Display::None
    };
    for mut node in &mut use_output_actions {
        if node.display != use_output_display {
            node.display = use_output_display;
        }
    }
}

pub(crate) fn reference_control_relevant(
    editor: &ImageEditorState,
    runner: &ImageRunnerState,
) -> bool {
    if editor.mode != EditorMode::Edit {
        return false;
    }
    let (Some(model), ImageRunnerState::Ready { capabilities }) = (&editor.model, runner) else {
        return false;
    };
    capabilities
        .descriptor(model)
        .is_some_and(|descriptor| descriptor.capabilities.tasks.contains(&ImageTaskKind::Edit))
}

fn handle_reference_button(
    interactions: Query<&Interaction, (Changed<Interaction>, With<ReferenceButton>)>,
    runner: Res<ImageRunnerStatus>,
    editor: Res<ImageEditorState>,
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
    if !reference_control_relevant(&editor, &runner.state) {
        panel.notice = "Reference images are available only in Edit mode".into();
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

#[allow(clippy::type_complexity)]
fn handle_use_output_reference_button(
    interactions: Query<
        &Interaction,
        (
            Changed<Interaction>,
            With<UseOutputReferenceButton>,
            Without<InteractionDisabled>,
        ),
    >,
    runner: Res<ImageRunnerStatus>,
    mut editor: ResMut<ImageEditorState>,
    mut panel: ResMut<ImageControlPanelState>,
) {
    if !interactions
        .iter()
        .any(|interaction| *interaction == Interaction::Pressed)
    {
        return;
    }
    if !reference_control_relevant(&editor, &runner.state) {
        panel.notice = "Output can become a reference only in Edit mode".into();
        return;
    }
    let Some((_, output)) = panel.latest_output.as_ref() else {
        panel.notice = "No completed output is available as a reference".into();
        return;
    };
    editor.source = Some(match output {
        HostImage::Pixels(pixels) => InputImage::Pixels(pixels.clone()),
        HostImage::Encoded(encoded) => InputImage::Encoded(encoded.clone()),
    });
    // A mask is spatially bound to the prior reference and cannot safely follow a replacement.
    editor.mask = None;
    panel.notice = "Current output is now the edit reference".into();
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
        let edit_descriptor = descriptor_for_mode_prefer_model(
            &runner.state,
            EditorMode::Edit,
            editor.model.as_ref(),
        );
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

// The three marker-filtered mutable Text queries must be a ParamSet: Bevy
// correctly rejects them as ordinary parameters because their access could
// overlap, even though each marker is unique in this plugin.
#[allow(clippy::type_complexity)]
fn update_control_labels(
    editor: Res<ImageEditorState>,
    runner: Res<ImageRunnerStatus>,
    mut labels: ParamSet<(
        Query<&mut Text, With<ModelButtonLabel>>,
        Query<&mut Text, With<SizeButtonLabel>>,
        Query<&mut Text, With<ReferenceLabel>>,
    )>,
) {
    if !editor.is_changed() && !runner.is_changed() {
        return;
    }
    if let Ok(mut label) = labels.p0().single_mut() {
        let value = model_control_value(&editor, &runner.state);
        if label.0 != value {
            label.0 = value;
        }
    }
    if let Ok(mut label) = labels.p1().single_mut() {
        let value = editor.options.dimensions.map_or_else(
            || "Model default".into(),
            |size| format!("{} x {}", size.width(), size.height()),
        );
        if label.0 != value {
            label.0 = value;
        }
    }
    if let Ok(mut label) = labels.p2().single_mut() {
        let value = if editor.source.is_some() {
            "Loaded - click to replace".into()
        } else {
            reference_button_text().into()
        };
        if label.0 != value {
            label.0 = value;
        }
    }
}

fn model_control_value(editor: &ImageEditorState, runner: &ImageRunnerState) -> String {
    let Some(model) = &editor.model else {
        return runner_control_value(runner);
    };
    let ImageRunnerState::Ready { capabilities } = runner else {
        return model.to_string();
    };
    let display_name = capabilities
        .descriptor(model)
        .map(|descriptor| descriptor.display_name.as_str())
        .unwrap_or_else(|| model.as_str());
    display_name.to_owned()
}

fn runner_control_value(state: &ImageRunnerState) -> String {
    match state {
        ImageRunnerState::Missing => "Runtime unavailable".into(),
        ImageRunnerState::Initializing { .. } => "Preparing runtime".into(),
        ImageRunnerState::Ready { .. } => "Runtime ready".into(),
        ImageRunnerState::Failed { .. } => "Runtime failed".into(),
    }
}

fn can_cycle_models(runner: &ImageRunnerState) -> bool {
    matches!(
        runner,
        ImageRunnerState::Ready { capabilities } if capabilities.models.len() > 1
    )
}

fn can_cycle_size(
    runner: &ImageRunnerState,
    editor: &ImageEditorState,
    panel: &ImageControlPanelState,
) -> bool {
    let ImageRunnerState::Ready { capabilities } = runner else {
        return false;
    };
    let Some(descriptor) = editor
        .model
        .as_ref()
        .and_then(|model| capabilities.descriptor(model))
    else {
        return false;
    };
    let current = editor
        .options
        .dimensions
        .and_then(preset_index)
        .unwrap_or(panel.size_index);
    next_supported_size_index_for_descriptor(current, descriptor)
        .is_some_and(|next| next != current)
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

fn readiness_transfer_presentation(
    state: &ImageRunnerState,
    readiness: &ImageRunnerReadiness,
) -> Option<ProgressPresentation> {
    let ImageRunnerState::Ready { capabilities } = state else {
        return None;
    };
    if capabilities.execution != crate::WgpuExecutionKind::BrowserWebGpu {
        return None;
    }
    let transfer = readiness.transfer.as_ref()?;
    if transfer.total_bytes == 0 {
        return None;
    }
    if transfer.loaded_bytes >= transfer.total_bytes {
        return Some(ProgressPresentation {
            headline: if readiness.selected_model_device_resident {
                "Selected model warm on GPU".into()
            } else {
                "Selected model cached".into()
            },
            detail: if readiness.selected_model_device_resident {
                format!(
                    "{} verified | {}/{} parts | reusable stages remain resident for repeat Runs",
                    format_bytes(transfer.total_bytes),
                    transfer.physical_parts_completed,
                    transfer.physical_parts_total,
                )
            } else {
                format!(
                    "{} verified | {}/{} parts | stages stream from browser cache per Run",
                    format_bytes(transfer.total_bytes),
                    transfer.physical_parts_completed,
                    transfer.physical_parts_total,
                )
            },
            fraction: Some(1.0),
            tone: ProgressTone::Complete,
        });
    }
    let remaining = transfer.total_bytes.saturating_sub(transfer.loaded_bytes);
    Some(ProgressPresentation {
        headline: "Ready to run; selected model setup continues".into(),
        detail: format!(
            "{} / {} cached | {}/{} parts | {} remaining for the first Run",
            format_bytes(transfer.loaded_bytes),
            format_bytes(transfer.total_bytes),
            transfer.physical_parts_completed,
            transfer.physical_parts_total,
            format_bytes(remaining),
        ),
        fraction: Some(transfer.loaded_bytes as f32 / transfer.total_bytes as f32),
        tone: ProgressTone::Normal,
    })
}

fn setup_progress_fraction(message: &str) -> Option<f32> {
    if let Some(percent) = message
        .strip_prefix("Model transfer ")
        .and_then(|message| message.split_once('%').map(|(percent, _)| percent))
        .and_then(|percent| percent.trim().parse::<f32>().ok())
    {
        return percent
            .is_finite()
            .then(|| (percent / 100.0).clamp(0.0, 1.0));
    }
    let remainder = message.strip_prefix("Model setup ")?;
    let fraction = remainder.split(':').next()?;
    let (completed, total) = fraction.split_once('/')?;
    let completed = completed.trim().parse::<u64>().ok()?;
    let total = total.trim().parse::<u64>().ok()?;
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
    if let ProgressEvent::StageStarted { stage, .. } = progress
        && let Some(encoded) = stage.strip_prefix(MODEL_SWITCH_PROGRESS_STAGE_PREFIX)
    {
        return model_switch_progress_presentation(prefix, encoded);
    }
    if let Some(transfer) = match progress {
        ProgressEvent::ArtifactStarted { transfer, .. }
        | ProgressEvent::ArtifactProgress { transfer, .. }
        | ProgressEvent::ArtifactVerified { transfer, .. } => transfer.as_ref(),
        _ => None,
    } {
        return transfer_progress_presentation(prefix, transfer);
    }
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
            total_bytes,
            ..
        } => (
            "Loading model data".into(),
            format!(
                "{} | {} | {}",
                component
                    .as_ref()
                    .map(|component| humanize_stage(&component.to_string()))
                    .unwrap_or_else(|| "Model artifact".into()),
                compact_artifact_name(path),
                format_bytes(*total_bytes)
            ),
            None,
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
                "{} | {} / {}",
                compact_artifact_name(path),
                format_bytes(*loaded_bytes),
                format_bytes(*total_bytes)
            ),
            Some(*loaded_bytes as f32 / (*total_bytes).max(1) as f32),
            ProgressTone::Normal,
        ),
        ProgressEvent::ArtifactVerified { path, .. } => (
            "Model object verified".into(),
            compact_artifact_name(path),
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

fn model_switch_progress_presentation(prefix: &str, encoded: &str) -> ProgressPresentation {
    let message = encoded
        .split_once(':')
        .and_then(|(steps, message)| {
            steps
                .parse::<u32>()
                .ok()
                .filter(|steps| *steps > 0)
                .map(|_| message)
        })
        .unwrap_or(encoded);
    ProgressPresentation {
        headline: format!("{prefix}: Switching model"),
        detail: compact_model_switch_message(message),
        fraction: setup_progress_fraction(message),
        tone: ProgressTone::Normal,
    }
}

fn compact_model_switch_message(message: &str) -> String {
    let nested = message.strip_prefix("Model setup: ").unwrap_or(message);
    if let Some(download) = nested.strip_prefix("downloading ")
        && let Some((_, artifact)) = download.split_once(" artifact ")
        && let Some((counts, description)) = artifact.split_once(": ")
        && let Some((index, total)) = counts.split_once('/')
        && index.parse::<u32>().is_ok()
        && total.parse::<u32>().is_ok()
    {
        let (path, bytes) = description
            .rsplit_once(" (")
            .and_then(|(path, suffix)| {
                suffix
                    .strip_suffix(" bytes)")
                    .and_then(|bytes| bytes.parse::<u64>().ok())
                    .map(|bytes| (path, bytes))
            })
            .unwrap_or((description, 0));
        let name = path.rsplit('/').next().unwrap_or(path);
        return if bytes > 0 {
            format!(
                "Downloading model file {index} of {total} | {name} | {}",
                format_bytes(bytes)
            )
        } else {
            format!("Downloading model file {index} of {total} | {name}")
        };
    }
    if let Some((_, detail)) = message
        .strip_prefix("Model setup ")
        .and_then(|message| message.split_once(": "))
    {
        return detail.to_owned();
    }
    nested.to_owned()
}

fn transfer_progress_presentation(
    prefix: &str,
    transfer: &ArtifactTransferProgress,
) -> ProgressPresentation {
    if transfer.loaded_bytes >= transfer.total_bytes
        && transfer.total_bytes > 0
        && let Some(activity) = &transfer.request_activity
    {
        let component = activity
            .component
            .as_ref()
            .or(transfer.component.as_ref())
            .map(|component| format!(" | {}", humanize_stage(&component.to_string())))
            .unwrap_or_default();
        return ProgressPresentation {
            headline: format!("{prefix} | {}{component}", activity.phase),
            detail: format!(
                "{} from cache | {} objects | {} reads",
                format_bytes(activity.processed_bytes),
                activity.logical_objects_completed,
                activity.bounded_ranges_processed,
            ),
            fraction: None,
            tone: ProgressTone::Normal,
        };
    }
    let component = transfer
        .component
        .as_ref()
        .map(|component| format!(" | {}", humanize_stage(&component.to_string())))
        .unwrap_or_default();
    let rate = transfer
        .bytes_per_second
        .map(|bytes| format!(" | {}/s", format_bytes(bytes)))
        .unwrap_or_default();
    let eta = transfer
        .eta_seconds
        .map(|seconds| format!(" | ETA {}", format_duration(seconds)))
        .unwrap_or_default();
    ProgressPresentation {
        headline: format!("{prefix} | {}{component}", transfer.phase),
        detail: format!(
            "{} / {} | {}/{} parts{rate}{eta}",
            format_bytes(transfer.loaded_bytes),
            format_bytes(transfer.total_bytes),
            transfer.physical_parts_completed,
            transfer.physical_parts_total,
        ),
        fraction: Some(transfer.loaded_bytes as f32 / transfer.total_bytes.max(1) as f32),
        tone: ProgressTone::Normal,
    }
}

fn compact_artifact_name(path: &burn_image::ArtifactPath) -> String {
    let value = path.to_string();
    value.rsplit('/').next().unwrap_or(&value).to_owned()
}

fn format_duration(seconds: u64) -> String {
    if seconds < 60 {
        format!("{}s", seconds.max(1))
    } else if seconds < 3_600 {
        format!("{}m {}s", seconds / 60, seconds % 60)
    } else {
        format!("{}h {}m", seconds / 3_600, (seconds % 3_600) / 60)
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
    readiness: Res<ImageRunnerReadiness>,
    jobs: Res<ImageJobs>,
    panel: Res<ImageControlPanelState>,
    mut labels: ParamSet<(
        Query<&mut Text, With<ProgressLabel>>,
        Query<&mut Text, With<ProgressDetailLabel>>,
    )>,
    mut fills: Query<(&mut Node, &mut BackgroundColor), With<ProgressFill>>,
) {
    let job = panel.latest_job.and_then(|id| jobs.get(id));
    let presentation = if job.is_none() && panel.notice.is_empty() {
        readiness_transfer_presentation(&runner.state, &readiness)
            .unwrap_or_else(|| progress_presentation(&runner.state, None, ""))
    } else {
        progress_presentation(&runner.state, job, &panel.notice)
    };
    if presentation.fraction.is_some()
        && !runner.is_changed()
        && !readiness.is_changed()
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
    model_buttons: Query<(Entity, Has<InteractionDisabled>), With<ModelButton>>,
    size_buttons: Query<
        (Entity, Has<InteractionDisabled>),
        (With<SizeButton>, Without<ModelButton>),
    >,
    reference_buttons: Query<
        (Entity, Has<InteractionDisabled>),
        (
            With<ReferenceButton>,
            Without<UseOutputReferenceButton>,
            Without<ModelButton>,
        ),
    >,
    use_output_buttons: Query<
        (Entity, Has<InteractionDisabled>),
        (
            With<UseOutputReferenceButton>,
            Without<ReferenceButton>,
            Without<RunButton>,
        ),
    >,
    random_seed_buttons: Query<
        (Entity, Has<InteractionDisabled>),
        (With<RandomSeedButton>, Without<RunButton>),
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
    let reference_relevant = reference_control_relevant(&editor, &runner.state);
    let can_use_output = !running && reference_relevant && panel.latest_output.is_some();
    let can_select_model = !running && can_cycle_models(&runner.state);
    let can_select_size = !running && can_cycle_size(&runner.state, &editor, &panel);
    let can_choose_reference = !running && reference_relevant;

    for (entity, disabled) in &model_buttons {
        set_button_disabled(&mut commands, entity, disabled, !can_select_model);
    }
    for (entity, disabled) in &size_buttons {
        set_button_disabled(&mut commands, entity, disabled, !can_select_size);
    }
    for (entity, disabled) in &reference_buttons {
        set_button_disabled(&mut commands, entity, disabled, !can_choose_reference);
    }
    for (entity, disabled) in &use_output_buttons {
        set_button_disabled(&mut commands, entity, disabled, !can_use_output);
    }
    for (entity, disabled) in &random_seed_buttons {
        set_button_disabled(&mut commands, entity, disabled, running);
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
}

fn run_requirement_message(
    runner: &ImageRunnerState,
    editor: &ImageEditorState,
    jobs: &ImageJobs,
    panel: &ImageControlPanelState,
) -> Option<String> {
    if jobs.iter().any(|job| !job.phase.is_terminal())
        || has_pending_submission(panel, jobs)
        || !matches!(runner, ImageRunnerState::Ready { .. })
        || editor.model.is_none()
    {
        return None;
    }

    let prompt_missing = editor.prompt_or_instruction.trim().is_empty();
    if editor.mode == EditorMode::Edit {
        match (prompt_missing, editor.source.is_none()) {
            (true, true) => {
                return Some(
                    "Enter an instruction and choose a reference image to enable Run.".into(),
                );
            }
            (true, false) => return Some("Enter an instruction to enable Run.".into()),
            (false, true) => return Some("Choose a reference image to enable Run.".into()),
            (false, false) => {}
        }
    } else if prompt_missing {
        return Some("Enter a prompt to enable Run.".into());
    }
    if !panel.seed_valid {
        return Some("Enter a valid numeric seed to enable Run.".into());
    }
    editor.validate_request().err().map(|error| error.message)
}

fn sync_run_requirements(
    runner: Res<ImageRunnerStatus>,
    editor: Res<ImageEditorState>,
    jobs: Res<ImageJobs>,
    panel: Res<ImageControlPanelState>,
    mut labels: Query<(&mut Text, &mut Visibility), With<RunRequirementsLabel>>,
) {
    if !runner.is_changed() && !editor.is_changed() && !jobs.is_changed() && !panel.is_changed() {
        return;
    }
    let message = run_requirement_message(&runner.state, &editor, &jobs, &panel);
    for (mut text, mut visibility) in &mut labels {
        let value = message.as_deref().unwrap_or_default();
        if text.0 != value {
            text.0 = value.into();
        }
        let next = if message.is_some() {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        if *visibility != next {
            *visibility = next;
        }
    }
}

fn sync_output_action_visibility(
    panel: Res<ImageControlPanelState>,
    mut save_buttons: Query<&mut Node, With<SaveButton>>,
) {
    if !panel.is_changed() {
        return;
    }
    let display = if panel.latest_output.is_some() {
        Display::Flex
    } else {
        Display::None
    };
    for mut node in &mut save_buttons {
        if node.display != display {
            node.display = display;
        }
    }
}

fn update_control_affordances(
    buttons: Query<Has<InteractionDisabled>, With<Button>>,
    mut affordances: Query<(&ChildOf, &mut Visibility), With<SelectorAffordance>>,
) {
    for (parent, mut visibility) in &mut affordances {
        let Ok(disabled) = buttons.get(parent.parent()) else {
            continue;
        };
        let next = if disabled {
            Visibility::Hidden
        } else {
            Visibility::Inherited
        };
        if *visibility != next {
            *visibility = next;
        }
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
        } else if *interaction == Interaction::Hovered {
            palette.hovered
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
        "Choose image..."
    } else {
        "Choose or drop image"
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
    static BROWSER_REFERENCE_ERROR: std::cell::RefCell<Option<String>> = const {
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
    BROWSER_REFERENCE_ERROR.with(|error| *error.borrow_mut() = None);
    Ok(())
}

/// Route browser file-picker validation failures into the same Bevy notice surface used natively.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn provide_reference_image_error(message: String) -> Result<(), wasm_bindgen::JsValue> {
    let message = message.trim();
    if message.is_empty() || message.len() > 1_024 {
        return Err(wasm_bindgen::JsValue::from_str(
            "reference image error must contain 1..=1024 UTF-8 bytes",
        ));
    }
    BROWSER_REFERENCE_QUEUE.with(|queue| *queue.borrow_mut() = None);
    BROWSER_REFERENCE_ERROR.with(|error| *error.borrow_mut() = Some(message.to_owned()));
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn drain_browser_reference_queue(
    mut load: MessageWriter<LoadImageBytes>,
    mut panel: ResMut<ImageControlPanelState>,
) {
    BROWSER_REFERENCE_ERROR.with(|error| {
        if let Some(message) = error.borrow_mut().take() {
            panel.notice = message;
        }
    });
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
        MIN_VIEWER_HEIGHT, can_cycle_models, descriptor_mode, event_progress_presentation,
        format_progress, image_control_panel_layout, model_control_value, next_model_descriptor,
        next_supported_size_index, preferred_size_index, preset_dimensions, preset_index,
        readiness_transfer_presentation, reference_control_relevant, runner_control_value,
        runner_progress_presentation, setup_progress_fraction,
    };
    #[cfg(feature = "boogu")]
    use super::{apply_descriptor_size, next_supported_size_index_for_descriptor};
    use bevy::{picking::Pickable, prelude::*, ui::InteractionDisabled};
    use burn_image::{
        ArtifactComponentId, ArtifactPath, ArtifactRequestTransferActivity,
        ArtifactTransferProgress, DimensionConstraints, Dimensions, ImageTaskKind, ModelId,
        ProgressEvent, RunId,
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

        assert_eq!(
            setup_progress_fraction("Model transfer 37.5%: Qwen - 1.0 GiB / 2.7 GiB"),
            Some(0.375)
        );
    }

    #[test]
    fn model_switch_progress_replaces_the_stale_stage_label_correctness() {
        let event = ProgressEvent::StageStarted {
            run_id: RunId(3),
            stage: format!(
                "{}5:Model setup 3/5: loading Qwen stages to GPU",
                super::MODEL_SWITCH_PROGRESS_STAGE_PREFIX
            ),
            total_steps: None,
        };
        let presentation = event_progress_presentation("Job 3", &event);
        assert_eq!(presentation.headline, "Job 3: Switching model");
        assert_eq!(presentation.detail, "loading Qwen stages to GPU");
        assert_eq!(presentation.fraction, Some(3.0 / 5.0));
    }

    #[test]
    fn model_switch_download_status_names_the_bundle_local_file_count_correctness() {
        let event = ProgressEvent::StageStarted {
            run_id: RunId(4),
            stage: format!(
                "{}5:Model setup: downloading Boogu artifact 17/110: pipeline/objects/abcdef.bpk (20971520 bytes)",
                super::MODEL_SWITCH_PROGRESS_STAGE_PREFIX
            ),
            total_steps: None,
        };
        let presentation = event_progress_presentation("Job 4", &event);
        assert_eq!(presentation.headline, "Job 4: Switching model");
        assert_eq!(
            presentation.detail,
            "Downloading model file 17 of 110 | abcdef.bpk | 20.0 MiB"
        );
        // A bundle-local file count is useful status, but is not misrepresented as whole-switch
        // progress because the dependency bundles have separate denominators.
        assert_eq!(presentation.fraction, None);
    }

    #[test]
    fn stage_local_file_count_is_not_presented_as_overall_progress_correctness() {
        let event = ProgressEvent::ArtifactStarted {
            run_id: RunId(7),
            path: ArtifactPath::new("pipeline/objects/current.bpk").unwrap(),
            component: Some(ArtifactComponentId::new("boogu-denoiser-blocks").unwrap()),
            file_index: 1,
            file_count: 4,
            total_bytes: 32,
            transfer: None,
        };
        let presentation = event_progress_presentation("Job 7", &event);
        assert_eq!(presentation.fraction, None);
        assert_eq!(
            presentation.detail,
            "boogu denoiser blocks | current.bpk | 32 B"
        );
        assert!(!presentation.detail.contains("file 2 of 4"));
    }

    #[test]
    fn aggregate_transfer_presentation_uses_bytes_and_unique_closure_counts_correctness() {
        let transfer = ArtifactTransferProgress {
            phase: "Inference model transfer".into(),
            component: Some(ArtifactComponentId::new("qwen").unwrap()),
            logical_objects_completed: 41,
            logical_objects_total: 223,
            physical_parts_completed: 300,
            physical_parts_total: 1_904,
            bounded_ranges_completed: 1_500,
            bounded_ranges_total: 9_520,
            loaded_bytes: 6 * 1024 * 1024 * 1024,
            total_bytes: 24 * 1024 * 1024 * 1024,
            bytes_per_second: Some(96 * 1024 * 1024),
            eta_seconds: Some(192),
            request_activity: None,
        };
        let event = ProgressEvent::ArtifactProgress {
            run_id: RunId(7),
            path: ArtifactPath::new("qwen/objects/current.bpk").unwrap(),
            loaded_bytes: 8,
            total_bytes: 16,
            transfer: Some(transfer),
        };
        let presentation = event_progress_presentation("Job 7", &event);
        assert_eq!(
            presentation.headline,
            "Job 7 | Inference model transfer | qwen"
        );
        assert_eq!(presentation.fraction, Some(0.25));
        assert!(presentation.detail.contains("6.00 GiB / 24.00 GiB"));
        assert!(presentation.detail.contains("300/1904 parts"));
        assert!(!presentation.detail.contains("41/223"));
        assert!(!presentation.detail.contains("1500/9520"));
        assert!(presentation.detail.contains("96.0 MiB/s"));
        assert!(presentation.detail.contains("ETA 3m 12s"));
        assert!(presentation.detail.len() <= 80);
    }

    #[test]
    fn browser_request_ready_preserves_incomplete_selected_model_progress_correctness() {
        let mut capabilities = crate::runner::tests::test_capabilities("test/browser-model");
        capabilities.execution = crate::WgpuExecutionKind::BrowserWebGpu;
        let state = crate::ImageRunnerState::Ready { capabilities };
        let readiness = crate::ImageRunnerReadiness {
            transfer: Some(ArtifactTransferProgress {
                phase: "Model setup".into(),
                component: Some(ArtifactComponentId::new("boogu-denoiser-blocks").unwrap()),
                logical_objects_completed: 106,
                logical_objects_total: 186,
                physical_parts_completed: 1_001,
                physical_parts_total: 1_751,
                bounded_ranges_completed: 1_001,
                bounded_ranges_total: 1_751,
                loaded_bytes: 19_870_166_528,
                total_bytes: 35_106_151_424,
                bytes_per_second: Some(100_000_000),
                eta_seconds: Some(153),
                request_activity: None,
            }),
            selected_model_device_resident: false,
        };
        let presentation = readiness_transfer_presentation(&state, &readiness).unwrap();
        assert_eq!(
            presentation.headline,
            "Ready to run; selected model setup continues"
        );
        assert_eq!(
            presentation.fraction,
            Some(19_870_166_528_f32 / 35_106_151_424_f32)
        );
        assert!(presentation.detail.contains("18.51 GiB / 32.70 GiB cached"));
        assert!(presentation.detail.contains("1001/1751 parts"));
        assert!(
            presentation
                .detail
                .contains("14.19 GiB remaining for the first Run")
        );

        let mut warm = readiness.clone();
        let warm_transfer = warm.transfer.as_mut().unwrap();
        warm_transfer.loaded_bytes = warm_transfer.total_bytes;
        warm_transfer.logical_objects_completed = warm_transfer.logical_objects_total;
        warm_transfer.physical_parts_completed = warm_transfer.physical_parts_total;
        warm_transfer.bounded_ranges_completed = warm_transfer.bounded_ranges_total;
        warm_transfer.eta_seconds = None;
        warm.selected_model_device_resident = true;
        let presentation = readiness_transfer_presentation(&state, &warm).unwrap();
        assert_eq!(presentation.headline, "Selected model warm on GPU");
        assert_eq!(presentation.fraction, Some(1.0));
        assert!(
            presentation
                .detail
                .contains("reusable stages remain resident for repeat Runs")
        );

        let native_state = crate::ImageRunnerState::Ready {
            capabilities: crate::runner::tests::test_capabilities("test/native-model"),
        };
        assert!(readiness_transfer_presentation(&native_state, &readiness).is_none());
    }

    #[test]
    fn complete_transport_presents_cache_rehydration_as_indeterminate_correctness() {
        let total = 35_106_151_424;
        let transfer = ArtifactTransferProgress {
            phase: "Inference model transfer".into(),
            component: Some(ArtifactComponentId::new("boogu-denoiser-blocks").unwrap()),
            logical_objects_completed: 186,
            logical_objects_total: 186,
            physical_parts_completed: 1_751,
            physical_parts_total: 1_751,
            bounded_ranges_completed: 9_000,
            bounded_ranges_total: 9_000,
            loaded_bytes: total,
            total_bytes: total,
            bytes_per_second: Some(96 * 1024 * 1024),
            eta_seconds: None,
            request_activity: Some(ArtifactRequestTransferActivity {
                phase: "Applying verified cached model stages".into(),
                current_path: Some(
                    ArtifactPath::new("pipeline/objects/denoiser-block-02.bpk").unwrap(),
                ),
                component: Some(ArtifactComponentId::new("boogu-denoiser-blocks").unwrap()),
                logical_objects_completed: 3,
                bounded_ranges_processed: 17,
                processed_bytes: 68 * 1024 * 1024,
            }),
        };
        let event = ProgressEvent::ArtifactProgress {
            run_id: RunId(8),
            path: ArtifactPath::new("pipeline/objects/denoiser-block-02.bpk").unwrap(),
            loaded_bytes: 4,
            total_bytes: 8,
            transfer: Some(transfer),
        };
        let presentation = event_progress_presentation("Job 8", &event);
        assert_eq!(
            presentation.headline,
            "Job 8 | Applying verified cached model stages | boogu denoiser blocks"
        );
        assert_eq!(presentation.fraction, None);
        assert_eq!(
            presentation.detail,
            "68.0 MiB from cache | 3 objects | 17 reads"
        );
        assert!(!presentation.detail.contains("lifetime transport"));
        assert!(presentation.detail.len() <= 64);
    }

    #[test]
    fn controls_layout_preserves_viewer_on_wide_and_narrow_windows_correctness() {
        let desktop = image_control_panel_layout(Vec2::new(1_280.0, 800.0));
        assert!(!desktop.narrow);
        assert_eq!(desktop.panel_width, 380.0);
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
    fn random_seed_button_updates_editor_and_text_input_correctness() {
        let mut app = App::new();
        app.insert_resource(crate::ImageEditorState {
            options: burn_image::GenerationOptions {
                seed: Some(5),
                ..Default::default()
            },
            ..Default::default()
        })
        .init_resource::<super::ImageControlPanelState>()
        .add_systems(Update, super::handle_random_seed_button);
        app.world_mut()
            .spawn((Button, super::RandomSeedButton, Interaction::Pressed));
        let seed_input = app
            .world_mut()
            .spawn((super::SeedInput, bevy::text::EditableText::new("5")))
            .id();

        app.update();

        let value = app
            .world()
            .resource::<crate::ImageEditorState>()
            .options
            .seed
            .expect("random seed button must set a seed");
        assert_ne!(value, 5);
        assert_eq!(
            app.world()
                .get::<bevy::text::EditableText>(seed_input)
                .unwrap()
                .value()
                .to_string(),
            value.to_string()
        );
        assert!(
            app.world()
                .resource::<super::ImageControlPanelState>()
                .seed_valid
        );
    }

    #[test]
    fn random_seed_collision_is_advanced_without_overflow_correctness() {
        assert_eq!(super::distinct_random_seed(Some(7), 7), 8);
        assert_eq!(super::distinct_random_seed(Some(u64::MAX), u64::MAX), 0);
        assert_eq!(super::distinct_random_seed(Some(7), 9), 9);
    }

    #[test]
    fn missing_runtime_label_does_not_claim_generation_correctness() {
        assert_eq!(
            runner_control_value(&crate::ImageRunnerState::Missing),
            "Runtime unavailable"
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

        assert_eq!(model_control_value(&editor, &runner), "Test Turbo");
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
    fn run_control_stays_disabled_until_edit_input_is_valid_correctness() {
        let runner = runner_with_models(&[("test/edit", "Test Edit", &[ImageTaskKind::Edit])]);
        let editor = crate::ImageEditorState {
            mode: crate::EditorMode::Edit,
            model: Some(ModelId::new("test/edit").unwrap()),
            prompt_or_instruction: "Improve this image".into(),
            source: None,
            ..Default::default()
        };
        assert!(editor.validate_request().is_err());
        let mut app = App::new();
        app.insert_resource(crate::ImageRunnerStatus { state: runner })
            .insert_resource(editor)
            .init_resource::<crate::ImageJobs>()
            .init_resource::<super::ImageControlPanelState>()
            .add_systems(Update, super::update_action_availability);
        let run = app
            .world_mut()
            .spawn((super::RunButton, InteractionDisabled))
            .id();

        app.update();

        assert!(app.world().get::<InteractionDisabled>(run).is_some());
    }

    #[test]
    fn generate_run_enables_only_after_a_prompt_is_entered_correctness() {
        let runner =
            runner_with_models(&[("test/turbo", "Test Turbo", &[ImageTaskKind::Generate])]);
        let mut app = App::new();
        app.insert_resource(crate::ImageRunnerStatus { state: runner })
            .insert_resource(crate::ImageEditorState {
                model: Some(ModelId::new("test/turbo").unwrap()),
                ..Default::default()
            })
            .init_resource::<crate::ImageJobs>()
            .init_resource::<super::ImageControlPanelState>()
            .add_systems(Update, super::update_action_availability);
        let run = app
            .world_mut()
            .spawn((super::RunButton, InteractionDisabled))
            .id();

        app.update();
        assert!(app.world().get::<InteractionDisabled>(run).is_some());

        app.world_mut()
            .resource_mut::<crate::ImageEditorState>()
            .prompt_or_instruction = "A ceramic bird".into();
        app.update();
        assert!(app.world().get::<InteractionDisabled>(run).is_none());
    }

    #[test]
    fn disabled_run_explains_the_missing_mode_inputs_correctness() {
        let runner = runner_with_models(&[(
            "test/edit",
            "Test Edit",
            &[ImageTaskKind::Generate, ImageTaskKind::Edit],
        )]);
        let jobs = crate::ImageJobs::default();
        let panel = super::ImageControlPanelState::default();
        let mut editor = crate::ImageEditorState {
            mode: crate::EditorMode::Edit,
            model: Some(ModelId::new("test/edit").unwrap()),
            ..Default::default()
        };
        assert_eq!(
            super::run_requirement_message(&runner, &editor, &jobs, &panel).as_deref(),
            Some("Enter an instruction and choose a reference image to enable Run.")
        );

        editor.mode = crate::EditorMode::Generate;
        assert_eq!(
            super::run_requirement_message(&runner, &editor, &jobs, &panel).as_deref(),
            Some("Enter a prompt to enable Run.")
        );
    }

    #[test]
    fn save_action_is_hidden_until_an_output_exists_correctness() {
        let mut app = App::new();
        app.init_resource::<super::ImageControlPanelState>()
            .add_systems(Update, super::sync_output_action_visibility);
        let save = app
            .world_mut()
            .spawn((super::SaveButton, Node::default()))
            .id();

        app.update();
        assert_eq!(
            app.world().get::<Node>(save).unwrap().display,
            Display::None
        );

        let dimensions = burn_image::Dimensions::new(1, 1).unwrap();
        let pixels = burn_image::PixelBuffer::new(
            dimensions,
            burn_image::PixelFormat::Rgba8,
            burn_image::ColorSpace::Srgb,
            vec![20, 40, 60, 255],
        )
        .unwrap();
        app.world_mut()
            .resource_mut::<super::ImageControlPanelState>()
            .latest_output = Some((crate::ImageJobId(1), burn_image::HostImage::Pixels(pixels)));
        app.update();
        assert_eq!(
            app.world().get::<Node>(save).unwrap().display,
            Display::Flex
        );
    }

    #[test]
    fn enabled_buttons_have_a_distinct_hover_palette_correctness() {
        let palette = super::ButtonPalette::neutral();
        let mut app = App::new();
        app.add_systems(Update, super::update_button_colors);
        let button = app
            .world_mut()
            .spawn((Interaction::Hovered, palette, BackgroundColor(palette.idle)))
            .id();

        app.update();
        assert_eq!(
            app.world().get::<BackgroundColor>(button).unwrap().0,
            palette.hovered
        );

        app.world_mut()
            .entity_mut(button)
            .insert(InteractionDisabled);
        app.update();
        assert_eq!(
            app.world().get::<BackgroundColor>(button).unwrap().0,
            palette.disabled
        );
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

        assert_eq!(model_control_value(&editor, &runner), "Test Turbo");
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
    fn model_dropdown_selects_an_explicit_model_and_derives_edit_mode_correctness() {
        let runner = runner_with_models(&[
            (
                "Boogu/Boogu-Image-0.1-Turbo",
                "Generate - Turbo 1K",
                &[ImageTaskKind::Generate],
            ),
            (
                "Boogu/Boogu-Image-0.1-Edit-Turbo",
                "Edit - Turbo 1K",
                &[ImageTaskKind::Edit],
            ),
        ]);
        let mut app = App::new();
        app.insert_resource(crate::ImageRunnerStatus { state: runner })
            .insert_resource(crate::ImageEditorState {
                model: Some(ModelId::new("Boogu/Boogu-Image-0.1-Turbo").unwrap()),
                ..Default::default()
            })
            .init_resource::<super::ImageControlPanelState>()
            .init_resource::<super::ModelDropdownState>()
            .init_resource::<super::SizeDropdownState>()
            .add_systems(
                Update,
                (super::handle_model_button, super::handle_model_option).chain(),
            );
        app.world_mut()
            .spawn((super::ModelButton, Interaction::Pressed));
        app.world_mut().spawn((
            Button,
            super::ModelOption { index: 1 },
            Interaction::Pressed,
        ));

        app.update();

        let editor = app.world().resource::<crate::ImageEditorState>();
        assert_eq!(editor.mode, crate::EditorMode::Edit);
        assert_eq!(
            editor.model.as_ref().unwrap().as_str(),
            "Boogu/Boogu-Image-0.1-Edit-Turbo"
        );
    }

    #[test]
    fn reference_control_tracks_mode_and_selected_model_capability_correctness() {
        let runner = runner_with_models(&[
            (
                "test/hybrid",
                "Hybrid",
                &[ImageTaskKind::Generate, ImageTaskKind::Edit],
            ),
            ("test/generate", "Generate", &[ImageTaskKind::Generate]),
        ]);
        let mut app = App::new();
        app.insert_resource(crate::ImageRunnerStatus { state: runner })
            .insert_resource(crate::ImageEditorState {
                model: Some(ModelId::new("test/hybrid").unwrap()),
                ..Default::default()
            })
            .init_resource::<super::ImageControlPanelState>()
            .add_systems(Update, super::sync_reference_control_visibility);
        let button = app
            .world_mut()
            .spawn((super::ReferenceButton, Node::default()))
            .id();
        let use_output = app
            .world_mut()
            .spawn((super::UseOutputReferenceAction, Node::default()))
            .id();

        app.update();
        assert_eq!(
            app.world().get::<Node>(button).unwrap().display,
            Display::None
        );
        assert_eq!(
            app.world().get::<Node>(use_output).unwrap().display,
            Display::None
        );

        app.world_mut()
            .resource_mut::<crate::ImageEditorState>()
            .mode = crate::EditorMode::Edit;
        app.update();
        assert_eq!(
            app.world().get::<Node>(button).unwrap().display,
            Display::Flex
        );
        assert_eq!(
            app.world().get::<Node>(use_output).unwrap().display,
            Display::None
        );
        assert!(reference_control_relevant(
            app.world().resource::<crate::ImageEditorState>(),
            &app.world().resource::<crate::ImageRunnerStatus>().state,
        ));

        let dimensions = burn_image::Dimensions::new(1, 1).unwrap();
        let pixels = burn_image::PixelBuffer::new(
            dimensions,
            burn_image::PixelFormat::Rgba8,
            burn_image::ColorSpace::Srgb,
            vec![1, 2, 3, 255],
        )
        .unwrap();
        app.world_mut()
            .resource_mut::<super::ImageControlPanelState>()
            .latest_output = Some((crate::ImageJobId(1), burn_image::HostImage::Pixels(pixels)));
        app.update();
        assert_eq!(
            app.world().get::<Node>(use_output).unwrap().display,
            Display::Flex
        );

        app.world_mut()
            .resource_mut::<crate::ImageEditorState>()
            .model = Some(ModelId::new("test/generate").unwrap());
        app.update();
        assert_eq!(
            app.world().get::<Node>(button).unwrap().display,
            Display::None
        );
        assert_eq!(
            app.world().get::<Node>(use_output).unwrap().display,
            Display::None
        );
    }

    #[test]
    fn edit_mode_can_explicitly_reuse_the_current_output_as_reference_correctness() {
        let runner = runner_with_models(&[(
            "test/hybrid",
            "Hybrid",
            &[ImageTaskKind::Generate, ImageTaskKind::Edit],
        )]);
        let dimensions = burn_image::Dimensions::new(1, 1).unwrap();
        let output_pixels = burn_image::PixelBuffer::new(
            dimensions,
            burn_image::PixelFormat::Rgba8,
            burn_image::ColorSpace::Srgb,
            vec![20, 40, 60, 255],
        )
        .unwrap();
        let mask = burn_image::InputMask::new(
            dimensions,
            burn_image::MaskSemantics::WhiteEdits,
            vec![255],
        )
        .unwrap();
        let panel = super::ImageControlPanelState {
            latest_output: Some((
                crate::ImageJobId(2),
                burn_image::HostImage::Pixels(output_pixels.clone()),
            )),
            ..Default::default()
        };
        let mut app = App::new();
        app.insert_resource(crate::ImageRunnerStatus { state: runner })
            .insert_resource(crate::ImageEditorState {
                mode: crate::EditorMode::Edit,
                model: Some(ModelId::new("test/hybrid").unwrap()),
                mask: Some(mask),
                ..Default::default()
            })
            .insert_resource(panel)
            .add_systems(Update, super::handle_use_output_reference_button);
        app.world_mut()
            .spawn((super::UseOutputReferenceButton, Interaction::Pressed));

        app.update();

        let editor = app.world().resource::<crate::ImageEditorState>();
        assert_eq!(
            editor.source,
            Some(burn_image::InputImage::Pixels(output_pixels))
        );
        assert!(editor.mask.is_none());
        assert_eq!(
            app.world()
                .resource::<super::ImageControlPanelState>()
                .notice,
            "Current output is now the edit reference"
        );
    }

    #[test]
    fn generate_mode_never_overwrites_the_reference_from_current_output_correctness() {
        let runner = runner_with_models(&[(
            "test/hybrid",
            "Hybrid",
            &[ImageTaskKind::Generate, ImageTaskKind::Edit],
        )]);
        let dimensions = burn_image::Dimensions::new(1, 1).unwrap();
        let original = burn_image::PixelBuffer::new(
            dimensions,
            burn_image::PixelFormat::Rgba8,
            burn_image::ColorSpace::Srgb,
            vec![1, 2, 3, 255],
        )
        .unwrap();
        let output = burn_image::PixelBuffer::new(
            dimensions,
            burn_image::PixelFormat::Rgba8,
            burn_image::ColorSpace::Srgb,
            vec![20, 40, 60, 255],
        )
        .unwrap();
        let panel = super::ImageControlPanelState {
            latest_output: Some((crate::ImageJobId(3), burn_image::HostImage::Pixels(output))),
            ..Default::default()
        };
        let mut app = App::new();
        app.insert_resource(crate::ImageRunnerStatus { state: runner })
            .insert_resource(crate::ImageEditorState {
                mode: crate::EditorMode::Generate,
                model: Some(ModelId::new("test/hybrid").unwrap()),
                source: Some(burn_image::InputImage::Pixels(original.clone())),
                ..Default::default()
            })
            .insert_resource(panel)
            .add_systems(Update, super::handle_use_output_reference_button);
        app.world_mut()
            .spawn((super::UseOutputReferenceButton, Interaction::Pressed));

        app.update();

        assert_eq!(
            app.world().resource::<crate::ImageEditorState>().source,
            Some(burn_image::InputImage::Pixels(original))
        );
    }

    #[test]
    fn decorative_button_text_never_blocks_parent_picking_correctness() {
        let mut app = App::new();
        app.add_systems(Startup, super::setup_controls);
        app.update();

        let world = app.world_mut();
        let selector = {
            let mut query = world.query_filtered::<Entity, With<super::ModelButton>>();
            query.single(world).unwrap()
        };
        let action = {
            let mut query = world.query_filtered::<Entity, With<super::RunButton>>();
            query.single(world).unwrap()
        };
        for button in [selector, action] {
            let children: Vec<_> = world.get::<Children>(button).unwrap().iter().collect();
            assert!(!children.is_empty());
            for child in children {
                assert!(world.get::<Text>(child).is_some());
                assert_eq!(world.get::<Pickable>(child), Some(&Pickable::IGNORE));
            }
        }
    }

    #[test]
    fn seed_input_uses_a_compact_single_line_height_correctness() {
        let mut app = App::new();
        app.add_systems(Startup, super::setup_controls);
        app.update();

        let world = app.world_mut();
        let input = {
            let mut query = world.query_filtered::<&Node, With<super::SeedInput>>();
            query.single(world).unwrap().clone()
        };
        assert_eq!(input.height, px(28));
        assert_eq!(input.padding.top, px(3));
        assert_eq!(input.padding.bottom, px(3));
    }

    #[test]
    fn selected_model_capability_is_the_single_mode_authority_correctness() {
        let runner = runner_with_models(&[
            ("test/turbo", "Test Turbo", &[ImageTaskKind::Generate]),
            ("test/edit", "Test Edit", &[ImageTaskKind::Edit]),
            (
                "test/hybrid",
                "Hybrid",
                &[ImageTaskKind::Generate, ImageTaskKind::Edit],
            ),
        ]);
        let crate::ImageRunnerState::Ready { capabilities } = runner else {
            unreachable!();
        };
        assert_eq!(
            descriptor_mode(
                capabilities
                    .descriptor(&ModelId::new("test/turbo").unwrap())
                    .unwrap()
            ),
            Some(crate::EditorMode::Generate)
        );
        assert_eq!(
            descriptor_mode(
                capabilities
                    .descriptor(&ModelId::new("test/edit").unwrap())
                    .unwrap()
            ),
            Some(crate::EditorMode::Edit)
        );
        // A hybrid descriptor has one deterministic UI mode rather than introducing another
        // independent selector; hosts may still author explicit Edit requests through the API.
        assert_eq!(
            descriptor_mode(
                capabilities
                    .descriptor(&ModelId::new("test/hybrid").unwrap())
                    .unwrap()
            ),
            Some(crate::EditorMode::Generate)
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

    #[test]
    fn size_dropdown_selects_the_explicit_pressed_preset_correctness() {
        let mut runner =
            runner_with_models(&[("test/turbo", "Test Turbo", &[ImageTaskKind::Generate])]);
        let crate::ImageRunnerState::Ready { capabilities } = &mut runner else {
            unreachable!();
        };
        capabilities.models[0].capabilities.dimensions = dimensions(256, 1024, Some(1024 * 1024));
        let mut app = App::new();
        app.insert_resource(crate::ImageRunnerStatus { state: runner })
            .insert_resource(crate::ImageEditorState {
                model: Some(ModelId::new("test/turbo").unwrap()),
                ..Default::default()
            })
            .init_resource::<super::ImageControlPanelState>()
            .insert_resource(super::SizeDropdownState { open: true })
            .add_systems(Update, super::handle_size_option);
        app.world_mut()
            .spawn((Button, super::SizeOption { index: 4 }, Interaction::Pressed));

        app.update();

        assert_eq!(
            app.world()
                .resource::<crate::ImageEditorState>()
                .options
                .dimensions,
            Some(Dimensions::new(1024, 768).unwrap())
        );
        assert_eq!(
            app.world()
                .resource::<super::ImageControlPanelState>()
                .size_index,
            4
        );
        assert!(!app.world().resource::<super::SizeDropdownState>().open);
    }

    #[test]
    fn size_dropdown_lists_only_model_supported_presets_correctness() {
        let mut runner =
            runner_with_models(&[("test/turbo", "Test Turbo", &[ImageTaskKind::Generate])]);
        let crate::ImageRunnerState::Ready { capabilities } = &mut runner else {
            unreachable!();
        };
        capabilities.models[0].capabilities.dimensions = dimensions(256, 1024, Some(1024 * 1024));
        let mut app = App::new();
        app.insert_resource(crate::ImageRunnerStatus { state: runner })
            .insert_resource(crate::ImageEditorState {
                model: Some(ModelId::new("test/turbo").unwrap()),
                options: burn_image::GenerationOptions {
                    dimensions: Some(Dimensions::new(1024, 1024).unwrap()),
                    ..Default::default()
                },
                ..Default::default()
            })
            .init_resource::<crate::ImageJobs>()
            .insert_resource(super::SizeDropdownState { open: true })
            .add_systems(Update, super::sync_size_dropdown);
        let dropdown = app
            .world_mut()
            .spawn((super::SizeDropdown, Node::default()))
            .id();
        let supported = app
            .world_mut()
            .spawn((
                super::SizeOption { index: 3 },
                Node::default(),
                super::ButtonPalette::neutral(),
            ))
            .id();
        let unsupported = app
            .world_mut()
            .spawn((
                super::SizeOption { index: 6 },
                Node::default(),
                super::ButtonPalette::neutral(),
            ))
            .id();

        app.update();

        assert_eq!(
            app.world().get::<Node>(dropdown).unwrap().display,
            Display::Flex
        );
        assert_eq!(
            app.world().get::<Node>(supported).unwrap().display,
            Display::Flex
        );
        assert_eq!(
            app.world().get::<Node>(unsupported).unwrap().display,
            Display::None
        );
        assert_eq!(
            app.world()
                .get::<super::ButtonPalette>(supported)
                .unwrap()
                .idle,
            Color::srgb(0.2, 0.34, 0.54)
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
            if variant == burn_boogu::BooguVariant::Image01EditTurbo {
                assert_eq!(
                    super::size_dropdown_hint(&descriptor),
                    Some("For 1.5K sizes, choose Edit - Turbo 1.5K from the Model menu.")
                );
            } else {
                assert_eq!(super::size_dropdown_hint(&descriptor), None);
            }
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
