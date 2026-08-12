use bevy::prelude::*;
use burn_image::{
    EditRequest, GenerateRequest, GenerationOptions, ImageRequest, InputImage, InputMask, ModelId,
    Prompt,
};

use crate::{FrontendError, ImageJobId, SubmitImageJob};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EditorMode {
    #[default]
    Generate,
    Edit,
}

/// Native/web-neutral state edited by UI widgets or an embedding host.
#[derive(Resource, Clone, Debug)]
pub struct ImageEditorState {
    pub mode: EditorMode,
    pub model: Option<ModelId>,
    pub prompt_or_instruction: String,
    pub negative_prompt: String,
    pub options: GenerationOptions,
    pub source: Option<InputImage>,
    pub mask: Option<InputMask>,
    pub edit_strength: Option<f32>,
}

impl Default for ImageEditorState {
    fn default() -> Self {
        Self {
            mode: EditorMode::Generate,
            model: None,
            prompt_or_instruction: String::new(),
            negative_prompt: String::new(),
            options: GenerationOptions::default(),
            source: None,
            mask: None,
            edit_strength: None,
        }
    }
}

impl ImageEditorState {
    pub fn build_request(&self) -> Result<ImageRequest, FrontendError> {
        let prompt = Prompt::new(self.prompt_or_instruction.clone())?;
        let negative_prompt = if self.negative_prompt.trim().is_empty() {
            None
        } else {
            Some(Prompt::new(self.negative_prompt.clone())?)
        };
        let request = match self.mode {
            EditorMode::Generate => ImageRequest::Generate(GenerateRequest {
                prompt,
                negative_prompt,
                options: self.options.clone(),
            }),
            EditorMode::Edit => ImageRequest::Edit(EditRequest {
                source: self.source.clone().ok_or_else(|| {
                    FrontendError::invalid_request("edit mode requires a source image")
                })?,
                instruction: prompt,
                negative_prompt,
                mask: self.mask.clone(),
                strength: self.edit_strength,
                options: self.options.clone(),
            }),
        };
        request.validate()?;
        Ok(request)
    }

    pub fn submission(&self, id: ImageJobId) -> Result<SubmitImageJob, FrontendError> {
        let model = self
            .model
            .clone()
            .ok_or_else(|| FrontendError::invalid_request("no image model is selected"))?;
        Ok(SubmitImageJob {
            id,
            model,
            request: self.build_request()?,
        })
    }
}

pub struct ImageEditorPlugin;

impl Plugin for ImageEditorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ImageEditorState>();
    }
}

#[cfg(test)]
mod tests {
    use burn_image::{ColorSpace, Dimensions, InputImage, ModelId, PixelBuffer, PixelFormat};

    use super::{EditorMode, ImageEditorState};

    #[test]
    fn editor_preserves_generation_text_correctness() {
        let editor = ImageEditorState {
            model: Some(ModelId::new("test/model").unwrap()),
            prompt_or_instruction: "  exact prompt text  ".into(),
            ..Default::default()
        };
        let request = editor.build_request().unwrap();
        let burn_image::ImageRequest::Generate(request) = request else {
            panic!("expected generation request");
        };
        assert_eq!(request.prompt.as_str(), "  exact prompt text  ");
    }

    #[test]
    fn editor_requires_edit_source_correctness() {
        let editor = ImageEditorState {
            mode: EditorMode::Edit,
            prompt_or_instruction: "make it dusk".into(),
            ..Default::default()
        };
        assert!(editor.build_request().is_err());

        let dimensions = Dimensions::new(1, 1).unwrap();
        let mut editor = editor;
        editor.source = Some(InputImage::Pixels(
            PixelBuffer::new(
                dimensions,
                PixelFormat::Rgba8,
                ColorSpace::Srgb,
                vec![0, 0, 0, 255],
            )
            .unwrap(),
        ));
        assert!(editor.build_request().is_ok());
    }
}
