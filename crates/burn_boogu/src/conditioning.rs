use crate::{BooguError, BooguTask};

/// Literal upstream T2I system prompt.
pub const T2I_SYSTEM_PROMPT: &str = "You are a helpful assistant that generates high-quality images based on user instructions. The instructions are as follows.";

/// Literal upstream edit system prompt.
pub const EDIT_SYSTEM_PROMPT: &str = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.";

/// Boogu-specific instruction construction policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstructionPolicy {
    /// Task whose system prompt and media ordering are required.
    pub task: BooguTask,
    /// Maximum length passed to the processor.
    pub max_sequence_length: usize,
    /// Whether the processor may truncate.
    pub truncate: bool,
    /// Number of input images.
    pub image_count: usize,
}

impl InstructionPolicy {
    /// Upstream-compatible policy for a task.
    pub fn upstream(task: BooguTask, image_count: usize) -> Result<Self, BooguError> {
        match task {
            BooguTask::Generate if image_count != 0 => {
                return Err(BooguError::InvalidRequest(
                    "generation conditioning must not contain reference images".into(),
                ));
            }
            BooguTask::Edit if image_count != 1 => {
                return Err(BooguError::InvalidRequest(
                    "the initial Edit-Turbo contract requires exactly one reference image".into(),
                ));
            }
            _ => {}
        }
        Ok(Self {
            task,
            max_sequence_length: 1280,
            truncate: false,
            image_count,
        })
    }

    /// Literal system prompt for the task.
    pub const fn system_prompt(&self) -> &'static str {
        match self.task {
            BooguTask::Generate => T2I_SYSTEM_PROMPT,
            BooguTask::Edit => EDIT_SYSTEM_PROMPT,
        }
    }

    /// Whether multimodal user content must place images before text.
    pub const fn images_precede_text(&self) -> bool {
        matches!(self.task, BooguTask::Edit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upstream_prompt_policy_reference() {
        let generation = InstructionPolicy::upstream(BooguTask::Generate, 0).unwrap();
        assert_eq!(generation.system_prompt(), T2I_SYSTEM_PROMPT);
        assert!(!generation.truncate);

        let edit = InstructionPolicy::upstream(BooguTask::Edit, 1).unwrap();
        assert_eq!(edit.system_prompt(), EDIT_SYSTEM_PROMPT);
        assert!(edit.images_precede_text());
    }
}
