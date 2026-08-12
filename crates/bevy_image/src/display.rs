use std::collections::BTreeMap;

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
}

impl DisplayedImages {
    pub fn get(&self, key: DisplayKey) -> Option<&DisplayedImage> {
        self.images.get(&key)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&DisplayKey, &DisplayedImage)> {
        self.images.iter()
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
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
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
                    displayed.images.insert(
                        key,
                        DisplayedImage {
                            handle: handle.clone(),
                            dimensions,
                        },
                    );
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
fn bind_image_views(
    mut ready: MessageReader<ImageDisplayReady>,
    mut exact_views: Query<(&GeneratedImageView, &mut ImageNode)>,
    mut latest_views: Query<
        &mut ImageNode,
        (With<LatestGeneratedImageView>, Without<GeneratedImageView>),
    >,
) {
    for image in ready.read() {
        for (view, mut node) in &mut exact_views {
            if view.key == image.key {
                node.image = image.handle.clone();
            }
        }
        for mut node in &mut latest_views {
            node.image = image.handle.clone();
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
    }
}
