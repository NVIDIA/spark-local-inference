/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Official DeepSeek conversation encoding with normalized message positions.

use super::{Frontend, Json, Map, PromptEncoder, RenderedPrompt};
use crate::{Error, Failure, FrontendError, Result};
use deepseek_recipe_core::{
    conversation::{Conversation, ReasoningEffort, ResponseFormat},
    messages::{InputMessage, ToolCall},
    multimodal::{ImageDetail, ImageSource},
    prompt::Prompt,
    special_tokens::SpecialToken,
    tools::{ToolChoice, ToolDefinition},
};
use deepseek_recipe_encoding::{
    PromptEncoding, dsv4::DeepseekV4Encoding, dsv41::DeepseekV41Encoding,
};

const IMAGE_SPECIAL_TOKEN: &str = SpecialToken::Image.default_str();

pub(super) fn invalid(message: impl Into<String>) -> Error {
    Error::frontend(FrontendError::Input(Failure::new(message.into())))
}

fn content(value: &Json, hints: &[Json]) -> Result<(Prompt, Vec<ImageSource>)> {
    let mut prompt = Prompt::new();
    let mut images = Vec::new();
    if value.is_null() {
        return Ok((prompt, images));
    }
    if let Some(text) = value.as_str() {
        prompt.push_plain(text);
        for _ in hints {
            prompt.mark_checkpoint(text.len());
        }
        return Ok((prompt, images));
    }
    let parts = value
        .as_array()
        .ok_or_else(|| invalid("message content must be text or content blocks"))?;
    for (index, part) in parts.iter().enumerate() {
        if index != 0 {
            prompt.push_plain("\n\n");
        }
        match part["type"].as_str() {
            Some("text") => prompt.push_plain(
                part["text"]
                    .as_str()
                    .ok_or_else(|| invalid("text content must be a string"))?,
            ),
            Some("image_url") => {
                let url = part["image_url"]["url"]
                    .as_str()
                    .ok_or_else(|| invalid("image_url.url must be a string"))?;
                if !url.starts_with("data:image/") {
                    return Err(invalid("image input requires a data URL"));
                }
                prompt.push_special(IMAGE_SPECIAL_TOKEN);
                images.push(ImageSource::DataUrl {
                    data_url: url.to_owned(),
                    detail: ImageDetail::Auto,
                    path: String::new(),
                });
            }
            _ => return Err(invalid("unsupported DeepSeek content block")),
        }
        if hints.contains(&Json::from(index + 1))
            || (hints == [Json::from(0)] && index + 1 == parts.len())
        {
            prompt.mark_checkpoint(prompt.as_str().len());
        }
    }
    Ok((prompt, images))
}

fn tool_arguments(value: &Json) -> Result<String> {
    let parsed = if let Some(text) = value.as_str() {
        serde_json::from_str::<Json>(text)
            .map_err(|_| invalid("tool arguments must contain a JSON object"))?
    } else {
        value.clone()
    };
    if !parsed.is_object() {
        return Err(invalid("tool arguments must contain a JSON object"));
    }
    serde_json::to_string(&parsed).map_err(|e| invalid(e.to_string()))
}

fn conversation(
    frontend: &Frontend,
    messages: &[Json],
    options: &Map<String, Json>,
) -> Result<(Conversation, Vec<(String, Vec<usize>)>, Option<u8>)> {
    if messages.is_empty() {
        return Err(invalid("cannot encode an empty conversation"));
    }
    for key in options.keys() {
        if matches!(
            key.as_str(),
            "bos_token"
                | "eos_token"
                | "unk_token"
                | "sep_token"
                | "pad_token"
                | "cls_token"
                | "mask_token"
                | "additional_special_tokens"
        ) {
            return Err(invalid(format!(
                "unsupported DeepSeek prompt option: {key}"
            )));
        }
    }
    let mut options = {
        let mut merged = frontend.defaults.clone();
        if options.contains_key("enable_thinking") && !options.contains_key("thinking_mode") {
            merged.remove("thinking_mode");
        }
        if options.contains_key("thinking_mode") {
            merged.remove("enable_thinking");
        }
        merged.extend(options.clone());
        merged
    };
    for key in options.keys() {
        if !matches!(
            key.as_str(),
            "thinking_mode"
                | "enable_thinking"
                | "reasoning_effort"
                | "add_generation_prompt"
                | "continue_final_message"
                | "drop_thinking"
                | "tools"
                | "tool_choice"
                | "response_format"
                | "documents"
                | "bos_token"
                | "eos_token"
                | "unk_token"
                | "sep_token"
                | "pad_token"
                | "cls_token"
                | "mask_token"
                | "additional_special_tokens"
        ) {
            return Err(invalid(format!(
                "unsupported DeepSeek prompt option: {key}"
            )));
        }
    }
    if !options.contains_key("thinking_mode") && !options.contains_key("enable_thinking") {
        options.insert("thinking_mode".into(), "thinking".into());
    }
    if let Some(value) = options.get("continue_final_message") {
        if !matches!(value, Json::Null | Json::Bool(false)) && value.as_str() != Some("") {
            return Err(invalid(
                "DeepSeek prompt encoding does not support continue_final_message",
            ));
        }
    }
    if options.get("add_generation_prompt") != Some(&Json::Bool(true)) {
        return Err(invalid(
            "DeepSeek prompt encoding requires add_generation_prompt=true",
        ));
    }
    if options.contains_key("drop_thinking") {
        return Err(invalid(
            "DeepSeek prompt encoding does not support drop_thinking",
        ));
    }
    if options.get("documents").is_some_and(|v| !v.is_null()) {
        return Err(invalid(
            "DeepSeek prompt encoding does not support documents",
        ));
    }
    let thinking_mode = match options.get("thinking_mode").filter(|v| !v.is_null()) {
        Some(Json::String(value)) if value == "thinking" => true,
        Some(Json::String(value)) if value == "chat" => false,
        Some(_) => return Err(invalid("thinking_mode must be thinking or chat")),
        None => match options.get("enable_thinking").filter(|v| !v.is_null()) {
            None => true,
            Some(Json::Bool(value)) => *value,
            _ => return Err(invalid("enable_thinking must be a boolean")),
        },
    };
    let effort_error = || {
        invalid(if frontend.prompt_encoder == PromptEncoder::DeepseekV41 {
            "reasoning_effort must be an integer from 1 through 100 or low, high, xhigh, or max"
        } else {
            "reasoning_effort must be low, high, xhigh, or max for DeepSeek V4"
        })
    };
    let (reasoning_effort, reasoning_effort_score) =
        match options.get("reasoning_effort").filter(|v| !v.is_null()) {
            None => (Some(ReasoningEffort::High), None),
            Some(Json::String(value)) => (
                Some(match value.as_str() {
                    "low" => ReasoningEffort::Low,
                    "high" => ReasoningEffort::High,
                    "xhigh" => ReasoningEffort::Xhigh,
                    "max" => ReasoningEffort::Max,
                    _ => return Err(effort_error()),
                }),
                None,
            ),
            Some(Json::Number(value)) if frontend.prompt_encoder == PromptEncoder::DeepseekV41 => {
                let score = value
                    .as_u64()
                    .filter(|score| (1..=100).contains(score))
                    .ok_or_else(effort_error)?;
                (None, Some(score as u8))
            }
            _ => return Err(effort_error()),
        };
    let tool_choice = match options
        .get("tool_choice")
        .filter(|v| !v.is_null())
        .map(|v| {
            v.as_str()
                .ok_or_else(|| invalid("tool_choice must be a string"))
        })
        .transpose()?
        .unwrap_or("auto")
    {
        "auto" => ToolChoice::Auto,
        "none" => ToolChoice::None,
        "required" => ToolChoice::Required,
        _ => return Err(invalid("unsupported tool_choice")),
    };
    let mut tools = Vec::new();
    if let Some(value) = options.get("tools").filter(|v| !v.is_null()) {
        for tool in value
            .as_array()
            .ok_or_else(|| invalid("tools must be an array"))?
        {
            let function = tool.get("function").unwrap_or(tool);
            tools.push(ToolDefinition {
                name: function["name"]
                    .as_str()
                    .ok_or_else(|| invalid("tool requires a name"))?
                    .to_owned(),
                description: function
                    .get("description")
                    .filter(|v| !v.is_null())
                    .map(|v| {
                        v.as_str()
                            .map(str::to_owned)
                            .ok_or_else(|| invalid("tool description must be text"))
                    })
                    .transpose()?,
                parameters: function
                    .get("parameters")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({"type":"object","properties":{}})),
                defer_loading: false,
                strict: function
                    .get("strict")
                    .filter(|v| !v.is_null())
                    .map(|v| {
                        v.as_bool()
                            .ok_or_else(|| invalid("tool strict must be a boolean"))
                    })
                    .transpose()?,
            });
        }
    }
    if tool_choice == ToolChoice::Required && tools.is_empty() {
        return Err(invalid("required tool choice needs at least one tool"));
    }
    // Required calls are checked at completion; thinking starts before the tool block.
    let tool_choice = if tool_choice == ToolChoice::Required && thinking_mode {
        ToolChoice::Auto
    } else {
        tool_choice
    };
    let response_format = match options.get("response_format").filter(|v| !v.is_null()) {
        None => ResponseFormat::Text,
        Some(value) if value == "text" || value["type"] == "text" => ResponseFormat::Text,
        Some(value) if value == "json_object" || value["type"] == "json_object" => {
            ResponseFormat::JsonObject
        }
        _ => return Err(invalid("unsupported DeepSeek response_format")),
    };
    let mut converted = Vec::new();
    let mut assistant_hints = Vec::new();
    for message in messages {
        if message.get("task").is_some() || message.get("prefix").is_some() {
            return Err(invalid(
                "DeepSeek task and assistant-prefill fields are unsupported",
            ));
        }
        let block_hints = message["_checkpoint_hints"]
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let (prompt, image_sources) = content(&message["content"], block_hints)?;
        let role = message["role"]
            .as_str()
            .ok_or_else(|| invalid("message requires a role"))?;
        if !image_sources.is_empty() && !matches!(role, "user" | "tool") {
            return Err(invalid("images require user or tool messages"));
        }
        converted.push(match role {
            "system" | "developer" => InputMessage::System {
                content: prompt,
                image_sources,
                tools: None,
            },
            "user" => InputMessage::User {
                content: prompt,
                image_sources,
            },
            "tool" => InputMessage::Tool {
                content: prompt,
                image_sources,
                tool_call_id: Some(
                    message["tool_call_id"]
                        .as_str()
                        .ok_or_else(|| invalid("tool message requires tool_call_id"))?
                        .to_owned(),
                ),
            },
            "assistant" => {
                assistant_hints.push((
                    prompt.as_str().to_owned(),
                    prompt.checkpoint_hints().to_vec(),
                ));
                let reasoning_content = message
                    .get("reasoning_content")
                    .or_else(|| message.get("thinking"))
                    .filter(|v| !v.is_null())
                    .map(|v| {
                        v.as_str()
                            .map(str::to_owned)
                            .ok_or_else(|| invalid("assistant reasoning must be text"))
                    })
                    .transpose()?;
                let tool_calls = message
                    .get("tool_calls")
                    .filter(|v| !v.is_null())
                    .map(|calls| {
                        calls
                            .as_array()
                            .ok_or_else(|| invalid("tool_calls must be an array"))?
                            .iter()
                            .map(|call| {
                                let function = &call["function"];
                                Ok(ToolCall {
                                    id: call["id"].as_str().unwrap_or("").to_owned(),
                                    name: function["name"]
                                        .as_str()
                                        .ok_or_else(|| invalid("tool call requires a name"))?
                                        .to_owned(),
                                    arguments: tool_arguments(&function["arguments"])?,
                                })
                            })
                            .collect::<Result<Vec<_>>>()
                    })
                    .transpose()?;
                InputMessage::Assistant {
                    content: prompt.as_str().to_owned(),
                    reasoning_content,
                    tool_calls,
                    prefix: false,
                }
            }
            _ => return Err(invalid("unsupported DeepSeek message role")),
        });
    }
    Ok((
        Conversation {
            messages: converted,
            thinking_mode,
            tools,
            tool_choice,
            reasoning_effort,
            response_format,
        },
        assistant_hints,
        reasoning_effort_score,
    ))
}

pub(super) fn validate_defaults(frontend: &Frontend) -> Result<()> {
    conversation(
        frontend,
        &[serde_json::json!({"role":"user","content":""})],
        &Map::new(),
    )
    .map(|_| ())
}

fn official(
    encoder: PromptEncoder,
    conversation: &Conversation,
    reasoning_effort_score: Option<u8>,
) -> Result<deepseek_recipe_encoding::RenderedPrompt> {
    Ok(match encoder {
        PromptEncoder::DeepseekV4 => DeepseekV4Encoding::new().render_conversation(conversation),
        PromptEncoder::DeepseekV41 => {
            let mut encoder = DeepseekV41Encoding::new();
            if let Some(score) = reasoning_effort_score {
                encoder = encoder
                    .with_reasoning_effort_score(score)
                    .map_err(|error| invalid(error.to_string()))?;
            }
            encoder.render_conversation(conversation)
        }
        PromptEncoder::Jinja => unreachable!(),
    })
}

pub(super) fn render(
    frontend: &Frontend,
    messages: &[Json],
    options: &Map<String, Json>,
) -> Result<RenderedPrompt> {
    let (conversation, assistant_hints, reasoning_effort_score) =
        conversation(frontend, messages, options)?;
    let rendered = official(
        frontend.prompt_encoder,
        &conversation,
        reasoning_effort_score,
    )?;
    let image_sources = rendered
        .image_sources
        .into_iter()
        .map(|source| match source {
            ImageSource::DataUrl { data_url, .. } => Ok(data_url),
            _ => Err(invalid("image input requires a data URL")),
        })
        .collect::<Result<Vec<_>>>()?;
    let mut token_ids = Vec::new();
    let mut offsets = Vec::new();
    let plain = frontend
        .ordinary_tokenizer
        .as_ref()
        .expect("DeepSeek plain tokenizer is initialized");
    let mut start = 0;
    let mut image_count = 0;
    for range in &rendered.special_ranges {
        let span = &rendered.prompt[start..range.start];
        let encoding = plain.encode(span, false).map_err(|error| {
            Error::frontend(FrontendError::Tokenizer(Failure::boxed(
                error.to_string(),
                error,
            )))
        })?;
        token_ids.extend_from_slice(encoding.get_ids());
        offsets.extend(
            encoding
                .get_offsets()
                .iter()
                .map(|&(a, b)| (start + a, start + b)),
        );
        let special = &rendered.prompt[range.clone()];
        let id = if special == IMAGE_SPECIAL_TOKEN {
            image_count += 1;
            frontend.image_placeholder_token_id.ok_or_else(|| {
                invalid("image inputs require a prepared image placeholder token ID")
            })?
        } else {
            frontend.tokenizer.get_added_vocabulary().get_vocab().get(special).copied()
                .ok_or_else(|| invalid(format!("missing DeepSeek special token: {special}")))?
        };
        token_ids.push(id);
        offsets.push((range.start, range.end));
        start = range.end;
    }
    let encoding = plain
        .encode(&rendered.prompt[start..], false)
        .map_err(|error| {
            Error::frontend(FrontendError::Tokenizer(Failure::boxed(
                error.to_string(),
                error,
            )))
        })?;
    token_ids.extend_from_slice(encoding.get_ids());
    offsets.extend(
        encoding
            .get_offsets()
            .iter()
            .map(|&(a, b)| (start + a, start + b)),
    );
    if image_count != image_sources.len() {
        return Err(invalid(
            "image placeholder count does not match image sources",
        ));
    }
    let mut checkpoint_hints = rendered.checkpoint_hints.clone();
    if assistant_hints.iter().any(|(_, hints)| !hints.is_empty()) {
        let assistant = SpecialToken::Assistant.default_str();
        let thinking_end = SpecialToken::ThinkingEnd.default_str();
        let mut messages = assistant_hints.iter();
        for bounds in rendered.message_boundaries.windows(2) {
            let ranges = rendered
                .special_ranges
                .iter()
                .filter(|r| r.start >= bounds[0] && r.end <= bounds[1]);
            let Some(marker) = ranges
                .clone()
                .find(|r| &rendered.prompt[(*r).clone()] == assistant)
            else {
                continue;
            };
            let Some((content, hints)) = messages.next() else {
                return Err(invalid("unexpected rendered assistant message"));
            };
            if hints.is_empty() {
                continue;
            }
            let end = ranges
                .filter(|r| r.start >= marker.end)
                .find(|r| &rendered.prompt[(*r).clone()] == thinking_end)
                .ok_or_else(|| invalid("rendered assistant message lacks a thinking delimiter"))?
                .end;
            if rendered.prompt.get(end..end + content.len()) != Some(content.as_str()) {
                return Err(invalid(
                    "rendered assistant content does not match checkpoint hints",
                ));
            }
            checkpoint_hints.extend(hints.iter().map(|&offset| end + offset));
        }
        if messages.next().is_some() {
            return Err(invalid("rendered assistant message is missing"));
        }
    }
    let message_boundaries = super::boundaries::token_ends(
        &rendered.message_boundaries,
        &checkpoint_hints,
        &offsets,
        rendered.prompt.len(),
    )?;
    Ok(RenderedPrompt {
        text: rendered.prompt,
        token_ids,
        image_sources,
        image_checkpoint_hints: Vec::new(),
        message_boundaries,
    })
}

impl Frontend {
    pub fn image_placeholder_token_id(&self) -> Result<u32> {
        if let Some(id) = self.image_placeholder_token_id {
            return Ok(id);
        }
        let ids = self.encode("<｜deepseek_image｜>", false)?;
        if ids.len() != 1 {
            return Err(invalid("image placeholder must encode to one token"));
        }
        Ok(ids[0])
    }
}
