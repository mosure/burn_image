//! Generic Qwen chat messages and template rendering.

use serde::{Deserialize, Serialize};

use crate::{Qwen3VlError, Result};

/// A role understood by Qwen's ChatML-style template.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

impl ChatRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

/// One content item. Images and videos are deliberately handles rather than decoded pixels; the
/// caller supplies corresponding preprocessed grids and patch tensors separately.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatContent {
    Text { text: String },
    Image,
    Video,
}

impl ChatContent {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }
}

/// A role and its ordered multimodal content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: Vec<ChatContent>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
}

impl ChatMessage {
    pub fn new(role: ChatRole, content: Vec<ChatContent>) -> Self {
        Self {
            role,
            content,
            tool_calls: Vec::new(),
        }
    }

    pub fn text(role: ChatRole, text: impl Into<String>) -> Self {
        Self::new(role, vec![ChatContent::text(text)])
    }

    pub fn with_tool_calls(mut self, tool_calls: Vec<ToolCall>) -> Self {
        self.tool_calls = tool_calls;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Tokens controlling the ordinary Qwen ChatML renderer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatTemplateConfig {
    pub im_start: String,
    pub im_end: String,
    pub vision_start: String,
    pub vision_end: String,
    pub image_pad: String,
    pub video_pad: String,
    /// Match Qwen's optional `add_vision_id` presentation (`Picture 1:`, `Video 1:`).
    pub add_vision_id: bool,
}

impl Default for ChatTemplateConfig {
    fn default() -> Self {
        Self {
            im_start: "<|im_start|>".into(),
            im_end: "<|im_end|>".into(),
            vision_start: "<|vision_start|>".into(),
            vision_end: "<|vision_end|>".into(),
            image_pad: "<|image_pad|>".into(),
            video_pad: "<|video_pad|>".into(),
            add_vision_id: false,
        }
    }
}

/// Stateless ordinary Qwen chat template renderer.
#[derive(Debug, Clone, Default)]
pub struct ChatTemplate {
    config: ChatTemplateConfig,
}

impl ChatTemplate {
    pub fn new(config: ChatTemplateConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &ChatTemplateConfig {
        &self.config
    }

    pub fn render(&self, messages: &[ChatMessage], add_generation_prompt: bool) -> Result<String> {
        self.render_with_tools(messages, &[], add_generation_prompt)
    }

    /// Render the released Qwen template, including optional JSON function definitions and
    /// assistant/tool-call turns. Tool definitions are ordinary JSON schema objects.
    pub fn render_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[serde_json::Value],
        add_generation_prompt: bool,
    ) -> Result<String> {
        if messages.is_empty() {
            return Err(Qwen3VlError::InvalidInput(
                "chat must contain at least one message".into(),
            ));
        }
        let mut output = String::new();
        let mut image_number = 0;
        let mut video_number = 0;
        if tools.is_empty() {
            if let Some(system) = messages
                .first()
                .filter(|message| message.role == ChatRole::System)
            {
                output.push_str(&self.config.im_start);
                output.push_str("system\n");
                render_text_only(&mut output, &system.content);
                output.push_str(&self.config.im_end);
                output.push('\n');
            }
        } else {
            output.push_str(&self.config.im_start);
            output.push_str("system\n");
            if let Some(system) = messages
                .first()
                .filter(|message| message.role == ChatRole::System)
            {
                render_text_only(&mut output, &system.content);
                output.push_str("\n\n");
            }
            output.push_str("# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>");
            for tool in tools {
                output.push('\n');
                output.push_str(&serde_json::to_string(tool).map_err(|error| {
                    Qwen3VlError::InvalidInput(format!("tool JSON cannot be rendered: {error}"))
                })?);
            }
            output.push_str("\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>");
            output.push_str(&self.config.im_end);
            output.push('\n');
        }

        for (index, message) in messages.iter().enumerate() {
            match message.role {
                ChatRole::System => {}
                ChatRole::User => {
                    output.push_str(&self.config.im_start);
                    output.push_str("user\n");
                    render_multimodal(
                        &mut output,
                        &message.content,
                        &self.config,
                        &mut image_number,
                        &mut video_number,
                    );
                    output.push_str(&self.config.im_end);
                    output.push('\n');
                }
                ChatRole::Assistant => {
                    output.push_str(&self.config.im_start);
                    output.push_str("assistant\n");
                    render_text_only(&mut output, &message.content);
                    for (call_index, call) in message.tool_calls.iter().enumerate() {
                        let has_text = message
                            .content
                            .iter()
                            .any(|content| matches!(content, ChatContent::Text { .. }));
                        if (call_index == 0 && has_text) || call_index != 0 {
                            output.push('\n');
                        }
                        output.push_str("<tool_call>\n{\"name\": \"");
                        output.push_str(&call.name);
                        output.push_str("\", \"arguments\": ");
                        output.push_str(&serde_json::to_string(&call.arguments).map_err(
                            |error| {
                                Qwen3VlError::InvalidInput(format!(
                                    "tool-call arguments cannot be rendered: {error}"
                                ))
                            },
                        )?);
                        output.push_str("}\n</tool_call>");
                    }
                    output.push_str(&self.config.im_end);
                    output.push('\n');
                }
                ChatRole::Tool => {
                    let previous_is_tool = index > 0 && messages[index - 1].role == ChatRole::Tool;
                    let next_is_tool =
                        index + 1 < messages.len() && messages[index + 1].role == ChatRole::Tool;
                    if !previous_is_tool {
                        output.push_str(&self.config.im_start);
                        output.push_str("user");
                    }
                    output.push_str("\n<tool_response>\n");
                    render_multimodal(
                        &mut output,
                        &message.content,
                        &self.config,
                        &mut image_number,
                        &mut video_number,
                    );
                    output.push_str("\n</tool_response>");
                    if !next_is_tool {
                        output.push_str(&self.config.im_end);
                        output.push('\n');
                    }
                }
            }
        }
        if add_generation_prompt {
            output.push_str(&self.config.im_start);
            output.push_str("assistant\n");
        }
        Ok(output)
    }
}

fn render_text_only(output: &mut String, content: &[ChatContent]) {
    for item in content {
        if let ChatContent::Text { text } = item {
            output.push_str(text);
        }
    }
}

fn render_multimodal(
    output: &mut String,
    content: &[ChatContent],
    config: &ChatTemplateConfig,
    image_number: &mut usize,
    video_number: &mut usize,
) {
    for item in content {
        match item {
            ChatContent::Text { text } => output.push_str(text),
            ChatContent::Image => {
                *image_number += 1;
                if config.add_vision_id {
                    output.push_str(&format!("Picture {image_number}: "));
                }
                output.push_str(&config.vision_start);
                output.push_str(&config.image_pad);
                output.push_str(&config.vision_end);
            }
            ChatContent::Video => {
                *video_number += 1;
                if config.add_vision_id {
                    output.push_str(&format!("Video {video_number}: "));
                }
                output.push_str(&config.vision_start);
                output.push_str(&config.video_pad);
                output.push_str(&config.vision_end);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_chat_template_reference() {
        let template = ChatTemplate::default();
        let rendered = template
            .render(
                &[ChatMessage::new(
                    ChatRole::User,
                    vec![ChatContent::Image, ChatContent::text("Describe this.")],
                )],
                true,
            )
            .unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this.<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn tool_calls_and_grouped_responses_match_reference() {
        let template = ChatTemplate::default();
        let messages = [
            ChatMessage::text(ChatRole::Assistant, "checking").with_tool_calls(vec![ToolCall {
                name: "weather".into(),
                arguments: serde_json::json!({"city":"Paris"}),
            }]),
            ChatMessage::text(ChatRole::Tool, "sunny"),
            ChatMessage::text(ChatRole::Tool, "18 C"),
        ];
        let rendered = template.render(&messages, false).unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>assistant\nchecking\n<tool_call>\n{\"name\": \"weather\", \"arguments\": {\"city\":\"Paris\"}}\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\nsunny\n</tool_response>\n<tool_response>\n18 C\n</tool_response><|im_end|>\n"
        );
    }
}
