//! Usable model-neutral Bevy controls for generation and editing.

use bevy::{
    prelude::*,
    text::{EditableText, EditableTextFilter, TextCursorStyle},
    ui::widget::NodeImageMode,
};
#[cfg(test)]
use burn_image::DimensionConstraints;
use burn_image::{
    Dimensions, HostImage, ImageEncoding, ImageTaskKind, ModelDescriptor, ProgressEvent,
};

use crate::{
    CancelImageJob, CompleteImageJob, EditorMode, ImageBytesLoaded, ImageDisplayFailed,
    ImageEditorState, ImageIoFailed, ImageIoId, ImageJobId, ImageJobPhase, ImageJobRejected,
    ImageJobs, ImageRunnerState, ImageRunnerStatus, LatestGeneratedImageView, LoadImageBytes,
    PrepareImageDownload,
};

#[cfg(target_arch = "wasm32")]
use crate::ImageDownloadReady;

const MAX_REFERENCE_BYTES: usize = 64 * 1024 * 1024;
const REFERENCE_IO_ID: ImageIoId = ImageIoId(1);
#[cfg(any(target_arch = "wasm32", not(feature = "native-io")))]
const DOWNLOAD_IO_ID: ImageIoId = ImageIoId(2);
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

pub struct ImageControlPanelPlugin;

impl Plugin for ImageControlPanelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ImageControlPanelState>()
            .add_systems(Startup, setup_controls)
            .add_systems(
                Update,
                (
                    select_initial_model,
                    sync_text_inputs,
                    handle_mode_button,
                    handle_model_button,
                    handle_size_button,
                    handle_reference_button,
                    handle_run_button,
                    handle_cancel_button,
                    handle_save_button,
                    accept_reference_images,
                    capture_outputs,
                    capture_frontend_errors,
                    update_control_labels,
                    update_button_colors,
                )
                    .chain(),
            );

        #[cfg(all(feature = "native-io", not(target_arch = "wasm32")))]
        app.add_systems(
            Update,
            accept_native_file_drop.before(accept_reference_images),
        );

        #[cfg(target_arch = "wasm32")]
        app.add_systems(
            Update,
            (drain_browser_reference_queue, complete_browser_download),
        );
    }
}

fn setup_controls(mut commands: Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(12),
                top: px(52),
                bottom: px(12),
                width: px(360),
                padding: px(14).all(),
                row_gap: px(9),
                flex_direction: FlexDirection::Column,
                overflow: Overflow::clip_y(),
                border_radius: BorderRadius::all(px(8)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.055, 0.065, 0.085, 0.96)),
        ))
        .with_children(|panel| {
            panel.spawn((
                Text::new("IMAGE GENERATION / EDIT"),
                TextFont::from_font_size(18.0),
                TextColor(Color::srgb(0.78, 0.86, 1.0)),
            ));

            spawn_labeled_button::<ModeButton, ModeButtonLabel>(panel, "Mode", "Generate");
            spawn_labeled_button::<ModelButton, ModelButtonLabel>(panel, "Model", "waiting…");

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
                    BackgroundColor(button_color(false)),
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

            panel.spawn((
                Text::new("Waiting for a WGPU model runtime"),
                TextFont::from_font_size(13.0),
                TextColor(Color::srgb(0.74, 0.78, 0.84)),
                TextLayout {
                    linebreak: LineBreak::WordOrCharacter,
                    ..default()
                },
                ProgressLabel,
            ));
        });

    commands.spawn((
        ImageNode {
            image_mode: NodeImageMode::Auto,
            ..default()
        },
        Node {
            position_type: PositionType::Absolute,
            left: px(390),
            right: px(12),
            top: px(52),
            bottom: px(12),
            max_width: percent(100),
            max_height: percent(100),
            margin: auto().all(),
            ..default()
        },
        LatestGeneratedImageView,
    ));
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
            BackgroundColor(button_color(false)),
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
    row.spawn((
        Button,
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
        BackgroundColor(color),
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
        editor.prompt_or_instruction = prompt.value().to_string();
    }
    if let Ok(seed) = seeds.single() {
        let value = seed.value().to_string();
        if value.is_empty() {
            editor.options.seed = None;
            panel.seed_valid = true;
        } else {
            match value.parse::<u64>() {
                Ok(seed) => {
                    editor.options.seed = Some(seed);
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
    editor.mode = match editor.mode {
        EditorMode::Generate => EditorMode::Edit,
        EditorMode::Edit => EditorMode::Generate,
    };
    let task = match editor.mode {
        EditorMode::Generate => ImageTaskKind::Generate,
        EditorMode::Edit => ImageTaskKind::Edit,
    };
    if let ImageRunnerState::Ready { capabilities } = &status.state
        && let Some(descriptor) = capabilities
            .models
            .iter()
            .find(|descriptor| descriptor.capabilities.tasks.contains(&task))
    {
        editor.model = Some(descriptor.id.clone());
        apply_descriptor_size(descriptor, &mut editor, &mut panel);
    }
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
    if capabilities.models.is_empty() {
        return;
    }
    let current = editor.model.as_ref().and_then(|model| {
        capabilities
            .models
            .iter()
            .position(|descriptor| descriptor.id == *model)
    });
    let descriptor =
        &capabilities.models[current.map_or(0, |index| index + 1) % capabilities.models.len()];
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
    #[cfg(not(target_arch = "wasm32"))]
    {
        panel.notice = "Drop a PNG, JPEG, or WebP file anywhere on the window".into();
    }
}

fn handle_run_button(
    interactions: Query<&Interaction, (Changed<Interaction>, With<RunButton>)>,
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
    let id = jobs.reserve_id();
    match editor.submission(id) {
        Ok(request) => {
            submit.write(request);
            panel.latest_job = Some(id);
            panel.notice = format!("Queued job {}", id.0);
        }
        Err(error) => panel.notice = error.to_string(),
    }
}

fn handle_cancel_button(
    interactions: Query<&Interaction, (Changed<Interaction>, With<CancelButton>)>,
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
    let id = panel.latest_job.or_else(|| {
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

fn handle_save_button(
    interactions: Query<&Interaction, (Changed<Interaction>, With<SaveButton>)>,
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
            Ok(path) => panel.notice = format!("Saved {}", path.display()),
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
        if loaded.id != REFERENCE_IO_ID {
            continue;
        }
        let dimensions = loaded.image.dimensions();
        editor.source = Some(loaded.image.clone());
        editor.mode = EditorMode::Edit;
        if let ImageRunnerState::Ready { capabilities } = &runner.state
            && let Some(descriptor) = capabilities
                .models
                .iter()
                .find(|descriptor| descriptor.capabilities.tasks.contains(&ImageTaskKind::Edit))
        {
            editor.model = Some(descriptor.id.clone());
            apply_descriptor_size(descriptor, &mut editor, &mut panel);
        }
        panel.notice = dimensions.map_or_else(
            || "Reference image loaded".into(),
            |size| format!("Reference loaded: {} × {}", size.width(), size.height()),
        );
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
    if crate::boogu::variant_for_model(&_descriptor.id)
        == Some(burn_boogu::BooguVariant::Image01EditTurbo1k5)
    {
        let edge = burn_boogu::BOOGU_1K5_DEFAULT_EDGE;
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
    mut outputs: MessageReader<CompleteImageJob>,
    mut panel: ResMut<ImageControlPanelState>,
) {
    for output in outputs.read() {
        if let Some(image) = output.output.images.first() {
            panel.latest_output = Some((output.id, image.image.clone()));
            panel.latest_job = Some(output.id);
            panel.notice = format!("Job {} completed", output.id.0);
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

// The five marker-filtered mutable Text queries must be a ParamSet: Bevy
// correctly rejects them as ordinary parameters because their access could
// overlap, even though each marker is unique in this plugin.
#[allow(clippy::type_complexity)]
fn update_control_labels(
    editor: Res<ImageEditorState>,
    runner: Res<ImageRunnerStatus>,
    jobs: Res<ImageJobs>,
    panel: Res<ImageControlPanelState>,
    mut labels: ParamSet<(
        Query<&mut Text, With<ModeButtonLabel>>,
        Query<&mut Text, With<ModelButtonLabel>>,
        Query<&mut Text, With<SizeButtonLabel>>,
        Query<&mut Text, With<ReferenceLabel>>,
        Query<&mut Text, With<ProgressLabel>>,
    )>,
) {
    if let Ok(mut label) = labels.p0().single_mut() {
        label.0 = format!(
            "Mode: {}",
            match editor.mode {
                EditorMode::Generate => "Generate",
                EditorMode::Edit => "Edit",
            }
        );
    }
    if let Ok(mut label) = labels.p1().single_mut() {
        label.0 = match &editor.model {
            Some(model) => format!("Model: {model}"),
            None => runner_state_label(&runner.state),
        };
    }
    if let Ok(mut label) = labels.p2().single_mut() {
        label.0 = editor.options.dimensions.map_or_else(
            || "Size: model default".into(),
            |size| format!("Size: {} × {}", size.width(), size.height()),
        );
    }
    if let Ok(mut label) = labels.p3().single_mut() {
        label.0 = if editor.source.is_some() {
            "Reference: loaded (click to replace)".into()
        } else {
            reference_button_text().into()
        };
    }
    if let Ok(mut label) = labels.p4().single_mut() {
        let job_status = panel
            .latest_job
            .and_then(|id| jobs.get(id))
            .map(format_job_status);
        label.0 = if panel.notice.is_empty() {
            job_status.unwrap_or_else(|| runner_state_label(&runner.state))
        } else if let Some(job_status) = job_status {
            format!("{job_status}\n{}", panel.notice)
        } else {
            panel.notice.clone()
        };
    }
}

fn format_job_status(job: &crate::ImageJobRecord) -> String {
    let phase = match &job.phase {
        ImageJobPhase::Queued => "queued".into(),
        ImageJobPhase::Running => job
            .last_progress
            .as_ref()
            .map(format_progress)
            .unwrap_or_else(|| "running".into()),
        ImageJobPhase::Completed => "completed".into(),
        ImageJobPhase::Failed { error } => format!("failed: {error}"),
        ImageJobPhase::Cancelled => "cancelled".into(),
    };
    format!("Job {}: {phase}", job.id.0)
}

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
    mut buttons: Query<(&Interaction, &mut BackgroundColor), Changed<Interaction>>,
) {
    for (interaction, mut background) in &mut buttons {
        background.0 = button_color(*interaction == Interaction::Pressed);
    }
}

fn button_color(pressed: bool) -> Color {
    if pressed {
        Color::srgb(0.24, 0.32, 0.48)
    } else {
        Color::srgb(0.14, 0.18, 0.27)
    }
}

const fn reference_button_text() -> &'static str {
    if cfg!(target_arch = "wasm32") {
        "Reference: choose image…"
    } else {
        "Reference: drop image on window"
    }
}

#[cfg(all(feature = "native-io", not(target_arch = "wasm32")))]
fn accept_native_file_drop(
    mut drops: MessageReader<FileDragAndDrop>,
    mut load: MessageWriter<LoadImageBytes>,
    mut panel: ResMut<ImageControlPanelState>,
) {
    for drop in drops.read() {
        let FileDragAndDrop::DroppedFile { path_buf, .. } = drop else {
            continue;
        };
        let result = std::fs::metadata(path_buf)
            .map_err(crate::FrontendError::from)
            .and_then(|metadata| {
                if metadata.len() > MAX_REFERENCE_BYTES as u64 {
                    Err(crate::FrontendError::invalid_request(format!(
                        "reference image exceeds the {} MiB limit",
                        MAX_REFERENCE_BYTES / (1024 * 1024)
                    )))
                } else {
                    std::fs::read(path_buf).map_err(crate::FrontendError::from)
                }
            });
        match result {
            Ok(bytes) => {
                load.write(LoadImageBytes {
                    id: REFERENCE_IO_ID,
                    bytes,
                    encoding: None,
                });
                panel.notice = format!("Loading {}", path_buf.display());
            }
            Err(error) => panel.notice = error.to_string(),
        }
    }
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    static BROWSER_REFERENCE_QUEUE: std::cell::RefCell<Vec<Vec<u8>>> = const { std::cell::RefCell::new(Vec::new()) };
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
    BROWSER_REFERENCE_QUEUE.with(|queue| queue.borrow_mut().push(bytes));
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn drain_browser_reference_queue(mut load: MessageWriter<LoadImageBytes>) {
    BROWSER_REFERENCE_QUEUE.with(|queue| {
        for bytes in queue.borrow_mut().drain(..) {
            load.write(LoadImageBytes {
                id: REFERENCE_IO_ID,
                bytes,
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
    #[cfg(feature = "boogu")]
    use super::{apply_descriptor_size, next_supported_size_index_for_descriptor};
    use super::{
        format_progress, next_supported_size_index, preferred_size_index, preset_dimensions,
        preset_index, runner_state_label,
    };
    use burn_image::{DimensionConstraints, Dimensions, ModelId, ProgressEvent, RunId};

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
}
