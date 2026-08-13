use std::collections::{BTreeMap, VecDeque};

#[cfg(feature = "app")]
use bevy::image::TRANSPARENT_IMAGE_HANDLE;
use bevy::{
    asset::{Assets, Handle, RenderAssetUsages},
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use burn_image::{Dimensions, HostImage};

use crate::{
    CompleteImageJob, FrontendError, ImageFrontendSet, ImageJobId, ImageJobPhase, ImageJobs,
    host_image_rgba8,
};

/// Maximum number of generated output textures retained by the frontend.
///
/// Exact [`GeneratedImageView`] bindings remain available for this recent
/// window. Older textures are removed from [`Assets<Image>`], and their strong
/// handles are released by the built-in view binding system.
pub const DISPLAYED_IMAGE_HISTORY_LIMIT: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DisplayKey {
    pub job: ImageJobId,
    pub output_index: u32,
}

#[derive(Clone, Debug)]
pub struct DisplayedImage {
    pub handle: Handle<Image>,
    pub dimensions: Dimensions,
}

#[derive(Resource, Default)]
pub struct DisplayedImages {
    images: BTreeMap<DisplayKey, DisplayedImage>,
    insertion_order: VecDeque<DisplayKey>,
    latest: Option<DisplayKey>,
}

impl DisplayedImages {
    pub fn get(&self, key: DisplayKey) -> Option<&DisplayedImage> {
        self.images.get(&key)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&DisplayKey, &DisplayedImage)> {
        self.images.iter()
    }

    pub fn latest(&self) -> Option<(DisplayKey, &DisplayedImage)> {
        let key = self.latest?;
        self.images.get(&key).map(|image| (key, image))
    }

    pub fn len(&self) -> usize {
        self.images.len()
    }

    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    fn insert(&mut self, key: DisplayKey, image: DisplayedImage) -> Vec<DisplayedImage> {
        let mut evicted = Vec::with_capacity(1);
        if let Some(previous) = self.images.insert(key, image) {
            self.insertion_order.retain(|candidate| *candidate != key);
            evicted.push(previous);
        }
        self.insertion_order.push_back(key);
        self.latest = Some(key);

        while self.images.len() > DISPLAYED_IMAGE_HISTORY_LIMIT {
            let Some(oldest) = self.insertion_order.pop_front() else {
                break;
            };
            if let Some(image) = self.images.remove(&oldest) {
                evicted.push(image);
            }
        }
        evicted
    }
}

#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct GeneratedImageView {
    pub key: DisplayKey,
}

#[cfg(feature = "app")]
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LatestGeneratedImageView;

#[derive(Message, Clone, Debug)]
pub struct ImageDisplayReady {
    pub key: DisplayKey,
    pub handle: Handle<Image>,
    pub dimensions: Dimensions,
}

#[derive(Message, Clone, Debug)]
pub struct ImageDisplayFailed {
    pub key: DisplayKey,
    pub error: FrontendError,
}

pub fn host_image_to_bevy_image(image: &HostImage) -> Result<(Dimensions, Image), FrontendError> {
    let (dimensions, rgba) = host_image_rgba8(image)?;
    let image = Image::new(
        Extent3d {
            width: dimensions.width(),
            height: dimensions.height(),
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        rgba,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    );
    Ok((dimensions, image))
}

pub struct ImageDisplayPlugin;

impl Plugin for ImageDisplayPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Assets<Image>>()
            .init_resource::<DisplayedImages>()
            .add_message::<ImageDisplayReady>()
            .add_message::<ImageDisplayFailed>()
            .add_systems(
                Update,
                materialize_completed_images.in_set(ImageFrontendSet::Display),
            );

        #[cfg(feature = "app")]
        app.add_systems(Update, bind_image_views.after(materialize_completed_images));
    }
}

fn materialize_completed_images(
    jobs: Res<ImageJobs>,
    mut completed: MessageReader<CompleteImageJob>,
    mut assets: ResMut<Assets<Image>>,
    mut displayed: ResMut<DisplayedImages>,
    mut ready: MessageWriter<ImageDisplayReady>,
    mut failed: MessageWriter<ImageDisplayFailed>,
) {
    for completion in completed.read() {
        let accepted = jobs
            .get(completion.id)
            .is_some_and(|job| job.phase == ImageJobPhase::Completed);
        if !accepted {
            continue;
        }
        for generated in &completion.output.images {
            let key = DisplayKey {
                job: completion.id,
                output_index: generated.index,
            };
            match host_image_to_bevy_image(&generated.image) {
                Ok((dimensions, image)) => {
                    let handle = assets.add(image);
                    let evicted = displayed.insert(
                        key,
                        DisplayedImage {
                            handle: handle.clone(),
                            dimensions,
                        },
                    );
                    for stale in evicted {
                        assets.remove(stale.handle.id());
                    }
                    ready.write(ImageDisplayReady {
                        key,
                        handle,
                        dimensions,
                    });
                }
                Err(error) => {
                    failed.write(ImageDisplayFailed { key, error });
                }
            }
        }
    }
}

#[cfg(feature = "app")]
#[allow(clippy::type_complexity)]
fn bind_image_views(
    displayed: Res<DisplayedImages>,
    mut ready: MessageReader<ImageDisplayReady>,
    mut exact_views: Query<(&GeneratedImageView, &mut ImageNode)>,
    mut latest_views: Query<
        &mut ImageNode,
        (With<LatestGeneratedImageView>, Without<GeneratedImageView>),
    >,
    mut latest_sprites: Query<
        (&mut Sprite, &mut Visibility),
        (With<LatestGeneratedImageView>, Without<ImageNode>),
    >,
) {
    for (view, mut node) in &mut exact_views {
        let target = displayed.get(view.key).map_or_else(
            || TRANSPARENT_IMAGE_HANDLE.clone(),
            |image| image.handle.clone(),
        );
        if node.image.id() != target.id() {
            node.image = target;
        }
    }

    for image in ready.read() {
        for mut node in &mut latest_views {
            if node.image.id() != image.handle.id() {
                node.image = image.handle.clone();
            }
        }
        for (mut sprite, mut visibility) in &mut latest_sprites {
            if sprite.image.id() != image.handle.id() {
                sprite.image = image.handle.clone();
                sprite.custom_size = Some(Vec2::new(
                    image.dimensions.width() as f32,
                    image.dimensions.height() as f32,
                ));
            }
            *visibility = Visibility::Visible;
        }
    }
}

#[cfg(test)]
mod tests {
    use burn_image::{ColorSpace, PixelBuffer, PixelFormat};

    use super::*;

    #[test]
    fn host_image_becomes_renderable_texture_correctness() {
        let dimensions = Dimensions::new(2, 1).unwrap();
        let image = HostImage::Pixels(
            PixelBuffer::new(
                dimensions,
                PixelFormat::Rgb8,
                ColorSpace::Srgb,
                vec![1, 2, 3, 4, 5, 6],
            )
            .unwrap(),
        );
        let (actual_dimensions, texture) = host_image_to_bevy_image(&image).unwrap();
        assert_eq!(actual_dimensions, dimensions);
        assert_eq!(
            texture.data.as_deref(),
            Some(&[1, 2, 3, 255, 4, 5, 6, 255][..])
        );
        assert_eq!(
            texture.texture_descriptor.format,
            TextureFormat::Rgba8UnormSrgb
        );
        assert_eq!(texture.asset_usage, RenderAssetUsages::RENDER_WORLD);
    }

    fn test_displayed_image(assets: &mut Assets<Image>) -> DisplayedImage {
        DisplayedImage {
            handle: assets.add(Image::default()),
            dimensions: Dimensions::new(1, 1).unwrap(),
        }
    }

    #[test]
    fn displayed_image_history_is_bounded_and_latest_safe_correctness() {
        let mut assets = Assets::<Image>::default();
        let mut displayed = DisplayedImages::default();
        let total = DISPLAYED_IMAGE_HISTORY_LIMIT + 3;

        for raw_id in 0..total {
            let key = DisplayKey {
                job: ImageJobId(raw_id as u64),
                output_index: 0,
            };
            let image = test_displayed_image(&mut assets);
            for stale in displayed.insert(key, image) {
                assert!(assets.remove(stale.handle.id()).is_some());
            }
        }

        assert_eq!(displayed.len(), DISPLAYED_IMAGE_HISTORY_LIMIT);
        assert_eq!(assets.len(), DISPLAYED_IMAGE_HISTORY_LIMIT);
        assert!(
            displayed
                .get(DisplayKey {
                    job: ImageJobId(0),
                    output_index: 0,
                })
                .is_none()
        );
        let (latest_key, _) = displayed.latest().unwrap();
        assert_eq!(latest_key.job, ImageJobId((total - 1) as u64));
    }

    #[test]
    fn replacing_an_exact_display_key_evicts_only_the_old_texture_correctness() {
        let mut assets = Assets::<Image>::default();
        let mut displayed = DisplayedImages::default();
        let key = DisplayKey {
            job: ImageJobId(7),
            output_index: 2,
        };
        let first = test_displayed_image(&mut assets);
        let first_id = first.handle.id();
        assert!(displayed.insert(key, first).is_empty());

        let second = test_displayed_image(&mut assets);
        let second_id = second.handle.id();
        let evicted = displayed.insert(key, second);
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].handle.id(), first_id);
        assert_eq!(displayed.get(key).unwrap().handle.id(), second_id);
        assert_eq!(displayed.len(), 1);
        assert_eq!(displayed.latest().unwrap().0, key);
    }

    #[cfg(feature = "app")]
    #[test]
    fn image_views_release_evicted_handles_and_follow_retained_outputs_correctness() {
        let mut app = App::new();
        app.init_resource::<Assets<Image>>()
            .add_message::<ImageDisplayReady>()
            .add_systems(Update, bind_image_views);

        let retained_key = DisplayKey {
            job: ImageJobId(11),
            output_index: 0,
        };
        let missing_key = DisplayKey {
            job: ImageJobId(10),
            output_index: 0,
        };
        let (retained, stale) = {
            let mut assets = app.world_mut().resource_mut::<Assets<Image>>();
            (
                test_displayed_image(&mut assets),
                test_displayed_image(&mut assets),
            )
        };
        let retained_id = retained.handle.id();
        let retained_handle = retained.handle.clone();
        let stale_id = stale.handle.id();

        let mut displayed = DisplayedImages::default();
        assert!(displayed.insert(retained_key, retained).is_empty());
        app.insert_resource(displayed);

        let retained_view = app
            .world_mut()
            .spawn((
                ImageNode::new(stale.handle.clone()),
                GeneratedImageView { key: retained_key },
            ))
            .id();
        let evicted_view = app
            .world_mut()
            .spawn((
                ImageNode::new(stale.handle.clone()),
                GeneratedImageView { key: missing_key },
            ))
            .id();
        let latest_view = app
            .world_mut()
            .spawn((
                ImageNode::new(stale.handle.clone()),
                LatestGeneratedImageView,
            ))
            .id();
        assert!(
            app.world_mut()
                .resource_mut::<Assets<Image>>()
                .remove(stale_id)
                .is_some()
        );
        drop(stale);
        app.world_mut()
            .resource_mut::<Messages<ImageDisplayReady>>()
            .write(ImageDisplayReady {
                key: retained_key,
                handle: retained_handle,
                dimensions: Dimensions::new(1, 1).unwrap(),
            });

        app.update();

        let image_id = |entity: Entity| {
            app.world()
                .entity(entity)
                .get::<ImageNode>()
                .unwrap()
                .image
                .id()
        };
        assert_eq!(image_id(retained_view), retained_id);
        assert_eq!(image_id(latest_view), retained_id);
        assert_eq!(image_id(evicted_view), TRANSPARENT_IMAGE_HANDLE.id());
    }
}
