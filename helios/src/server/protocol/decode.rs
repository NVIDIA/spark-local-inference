/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use super::{Decoded, Protocol};
use crate::{
    Result,
    server::{inference::*, observation::ObservationOptions},
};
use serde_json::{Value, json};

fn field<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{key} must be a string").into())
}
fn array<'a>(v: &'a Value, key: &str) -> Result<&'a [Value]> {
    v.get(key)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| format!("{key} must be an array").into())
}
fn optional_id(v: &Value, key: &str) -> Result<Option<String>> {
    v.get(key)
        .filter(|v| !v.is_null())
        .map(|_| field(v, key).map(str::to_owned))
        .transpose()
}
fn flag(v: &Value, key: &str, default: bool) -> Result<bool> {
    match v.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Bool(v)) => Ok(*v),
        _ => Err(format!("{key} must be boolean").into()),
    }
}
fn present(v: &Value, key: &str) -> bool {
    v.get(key).is_some_and(|v| !v.is_null())
}
fn reject(v: &Value, keys: &[&str]) -> Result<()> {
    for key in keys {
        if present(v, key) && v[key] != false {
            return Err(unsupported(format!("{key} is not supported")));
        }
    }
    Ok(())
}
fn strings(v: Option<&Value>) -> Result<Vec<String>> {
    match v {
        None | Some(Value::Null) => Ok(vec![]),
        Some(Value::String(s)) => Ok(vec![s.clone()]),
        Some(Value::Array(a)) => a
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| "stop strings must be strings".into())
            })
            .collect(),
        _ => Err("stops must be a string or an array of strings".into()),
    }
}
fn arguments(v: &Value) -> Result<Value> {
    let value = if let Some(text) = v.as_str() {
        serde_json::from_str(text).map_err(|e| format!("invalid tool arguments JSON: {e}"))?
    } else {
        v.clone()
    };
    if !value.is_object() {
        return Err("tool arguments must be a JSON object".into());
    }
    Ok(value)
}
fn qualified(name: &str, namespace: Option<&str>) -> String {
    namespace.map_or_else(|| name.to_owned(), |ns| format!("{ns}__{name}"))
}
fn plain_parts(value: &Value, protocol: Protocol) -> Result<Vec<Part>> {
    if value.is_null() {
        return Ok(vec![]);
    }
    if let Some(text) = value.as_str() {
        return Ok(vec![Part::Text(text.into())]);
    }
    value
        .as_array()
        .ok_or(if protocol == Protocol::Chat {
            "Unexpected content type."
        } else {
            "content must be text or an array"
        })?
        .iter()
        .map(|part| match field(part, "type")? {
            "text" | "input_text" | "output_text" => Ok(Part::Text(field(part, "text")?.into())),
            "image_url" => Ok(Part::Image(field(&part["image_url"], "url")?.into())),
            "input_image" => Ok(Part::Image(field(part, "image_url")?.into())),
            "image" if protocol == Protocol::Messages => {
                let source = &part["source"];
                if field(source, "type")? != "base64" {
                    return Err(unsupported("images require base64 PNG, JPEG, or WebP"));
                }
                let media = field(source, "media_type")?;
                if !["image/png", "image/jpeg", "image/webp"].contains(&media) {
                    return Err(unsupported("images require PNG, JPEG, or WebP"));
                }
                Ok(Part::Image(format!(
                    "data:{media};base64,{}",
                    field(source, "data")?
                )))
            }
            kind => Err(unsupported(if protocol == Protocol::Chat {
                "Unexpected item type in content.".into()
            } else {
                format!("unsupported content type {kind}")
            })),
        })
        .collect()
}
/// Reads tool-message `tool_error` parts, tml-renderers' OpenAI form of a
/// failed result, as the text of a failed result.
fn tool_error_parts(content: &Value) -> Result<(Value, bool)> {
    let Some(parts) = content.as_array() else {
        return Ok((content.clone(), false));
    };
    if !parts.iter().any(|part| part["type"] == "tool_error") {
        return Ok((content.clone(), false));
    }
    if parts.iter().any(|part| part["type"] == "text") {
        return Err("tool message content cannot mix text and tool_error parts".into());
    }
    let parts = parts
        .iter()
        .map(|part| {
            if part["type"] != "tool_error" {
                return Ok(part.clone());
            }
            Ok(json!({"type":"text","text":field(part, "text")?}))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((Value::Array(parts), true))
}
fn thinking(text: &str, signature: Option<&str>) -> Result<Part> {
    if let Some(signature) = signature.filter(|s| s.starts_with("flare-thinking-v1:")) {
        if signature != super::encode::thinking_signature(text) {
            return Err("thinking marker does not match text".into());
        }
    }
    Ok(Part::Reasoning(text.into()))
}
fn chat_messages(body: &Value) -> Result<Vec<Message>> {
    array(body, "messages")?
        .iter()
        .map(|v| {
            let role = serde_json::from_value(v["role"].clone())
                .map_err(|_| "unsupported message role")?;
            let mut parts = vec![];
            if let Some(text) = v.get("reasoning_content").filter(|v| !v.is_null()) {
                parts.push(Part::Reasoning(
                    text.as_str()
                        .ok_or("reasoning_content must be text")?
                        .into(),
                ));
            }
            if role == Role::Tool {
                let (content, failed) = tool_error_parts(&v["content"])?;
                parts.push(Part::ToolResult {
                    id: optional_id(v, "tool_call_id")?,
                    name: optional_id(v, "name")?,
                    content: plain_parts(&content, Protocol::Chat)?,
                    is_error: failed || flag(v, "is_error", false)?,
                });
            } else {
                parts.extend(plain_parts(&v["content"], Protocol::Chat)?);
                if present(v, "tool_calls") {
                    for call in array(v, "tool_calls")? {
                        parts.push(Part::ToolCall {
                            id: optional_id(call, "id")?,
                            name: field(&call["function"], "name")?.into(),
                            arguments: arguments(&call["function"]["arguments"])?,
                            arguments_source: call["function"]["arguments"].as_str().map(str::to_owned),
                        });
                    }
                }
            }
            Ok(Message {
                role,
                parts,
                phase: v.get("phase").and_then(Value::as_str).map(str::to_owned),
            })
        })
        .collect()
}
fn anthropic_messages(body: &Value, hints: &mut Vec<(usize, usize)>) -> Result<Vec<Message>> {
    let mut messages = vec![];
    if present(body, "system") {
        messages.push(Message {
            role: Role::System,
            parts: plain_parts(&body["system"], Protocol::Messages)?,
            phase: None,
        });
    }
    for v in array(body, "messages")? {
        let role = match field(v, "role")? {
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "system" => Role::System,
            _ => return Err("Messages roles must be user, assistant, or system".into()),
        };
        let mut parts = vec![];
        if let Some(blocks) = v["content"].as_array() {
            for block in blocks {
                parts.extend(match field(block, "type")? {
                    "thinking" => vec![thinking(
                        field(block, "thinking")?,
                        block.get("signature").and_then(Value::as_str),
                    )?],
                    "tool_use" => vec![Part::ToolCall {
                        id: Some(field(block, "id")?.into()),
                        name: field(block, "name")?.into(),
                        arguments: arguments(&block["input"])?,
                        arguments_source: block["input"].as_str().map(str::to_owned),
                    }],
                    "tool_result" => vec![Part::ToolResult {
                        id: Some(field(block, "tool_use_id")?.into()),
                        name: None,
                        content: plain_parts(&block["content"], Protocol::Messages)?,
                        is_error: flag(block, "is_error", false)?,
                    }],
                    _ => plain_parts(&json!([block]), Protocol::Messages)?,
                });
                if role == Role::User
                    && hints.len() < 4
                    && present(block, "cache_control")
                    && matches!(
                        block["type"].as_str(),
                        Some("text" | "image" | "tool_result")
                    )
                {
                    hints.push((messages.len(), parts.len()));
                }
            }
        } else {
            parts = plain_parts(&v["content"], Protocol::Messages)?;
        }
        if role == Role::System {
            if parts.iter().any(|part| !matches!(part, Part::Text(_))) {
                return Err("system messages require text content".into());
            }
            parts.retain(|part| matches!(part, Part::Text(text) if !text.is_empty()));
            if parts.is_empty() {
                continue;
            }
        }
        messages.push(Message {
            role,
            parts,
            phase: None,
        });
    }
    Ok(messages)
}
fn responses_messages(body: &Value) -> Result<Vec<Message>> {
    let mut messages = vec![];
    if present(body, "instructions") {
        messages.push(Message {
            role: Role::System,
            parts: vec![Part::Text(field(body, "instructions")?.into())],
            phase: None,
        });
    }
    let items = if let Some(text) = body["input"].as_str() {
        vec![json!({"role":"user","content":text})]
    } else {
        array(body, "input")?.to_vec()
    };
    for item in items {
        let kind = item
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("message");
        let (role, parts) = match kind {
            "message" => (
                serde_json::from_value(item["role"].clone()).map_err(|_| "invalid input role")?,
                plain_parts(&item["content"], Protocol::Responses)?,
            ),
            "function_call" => (
                Role::Assistant,
                vec![Part::ToolCall {
                    id: Some(field(&item, "call_id")?.into()),
                    name: qualified(
                        field(&item, "name")?,
                        item.get("namespace").and_then(Value::as_str),
                    ),
                    arguments: arguments(&item["arguments"])?,
                    arguments_source: item["arguments"].as_str().map(str::to_owned),
                }],
            ),
            "custom_tool_call" => (
                Role::Assistant,
                vec![Part::ToolCall {
                    id: Some(field(&item, "call_id")?.into()),
                    name: qualified(
                        field(&item, "name")?,
                        item.get("namespace").and_then(Value::as_str),
                    ),
                    arguments: json!({"input":field(&item,"input")?}),
                    arguments_source: None,
                }],
            ),
            "function_call_output" | "custom_tool_call_output" => (
                Role::Tool,
                vec![Part::ToolResult {
                    id: Some(field(&item, "call_id")?.into()),
                    name: None,
                    content: plain_parts(&item["output"], Protocol::Responses)?,
                    is_error: false,
                }],
            ),
            "reasoning" => {
                let source = item
                    .get("content")
                    .and_then(Value::as_array)
                    .filter(|v| !v.is_empty())
                    .or_else(|| item.get("summary").and_then(Value::as_array));
                let mut parts = vec![];
                if let Some(source) = source {
                    for part in source {
                        let text = field(part, "text")?;
                        if !text.is_empty() {
                            parts.push(Part::Reasoning(text.into()));
                        }
                    }
                }
                if parts.is_empty() && present(&item, "encrypted_content") {
                    return Err(unsupported(
                        "encrypted-only reasoning history cannot be rendered",
                    ));
                }
                (Role::Assistant, parts)
            }
            _ => {
                return Err(unsupported(format!(
                    "unsupported Responses input item {kind}"
                )));
            }
        };
        // Reasoning and tool calls belong to the same assistant turn as its text.
        if role == Role::Assistant && messages.last().is_some_and(|m| m.role == Role::Assistant) {
            messages.last_mut().unwrap().parts.extend(parts);
        } else {
            messages.push(Message {
                role,
                parts,
                phase: item.get("phase").and_then(Value::as_str).map(str::to_owned),
            });
        }
    }
    Ok(messages)
}
fn tool(v: &Value, protocol: Protocol, namespace: Option<&str>) -> Result<Tool> {
    if protocol == Protocol::Messages && present(v, "type") && v["type"] != "custom" {
        return Err(unsupported(
            "only client tools are supported; hosted tool types cannot execute locally",
        ));
    }
    let custom = protocol == Protocol::Responses && v["type"] == "custom";
    let definition = if protocol == Protocol::Chat {
        &v["function"]
    } else {
        v
    };
    if flag(definition, "strict", false)? {
        return Err(unsupported(
            "strict tool schemas require constrained tool generation",
        ));
    }
    let grammar = if custom && present(v, "format") && v["format"]["type"] != "text" {
        if v["format"]["type"] != "grammar"
            || !["lark", "regex"].contains(&field(&v["format"], "syntax")?)
        {
            return Err(unsupported(
                "custom tool format must be text, Lark, or regex",
            ));
        }
        Some(ToolGrammar {
            syntax: field(&v["format"], "syntax")?.into(),
            source: field(&v["format"], "definition")?.into(),
        })
    } else {
        None
    };
    if protocol != Protocol::Messages && !custom && v["type"] != "function" {
        return Err(unsupported(
            "only client function and text custom tools are supported",
        ));
    }
    let declared = definition
        .get(if protocol == Protocol::Messages {
            "input_schema"
        } else {
            "parameters"
        })
        .filter(|v| !v.is_null());
    let parameters = if custom {
        json!({"type":"object","properties":{"input":{"type":"string"}},"required":["input"]})
    } else {
        declared
            .cloned()
            .unwrap_or_else(|| json!({"type":"object","properties":{}}))
    };
    Ok(Tool {
        name: field(definition, "name")?.into(),
        namespace: namespace.map(str::to_owned),
        description: definition
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("")
            .into(),
        parameters,
        declared_parameters: custom || declared.is_some(),
        custom,
        grammar,
    })
}
fn thinking_budget(value: &Value) -> Result<u32> {
    let budget = value
        .as_u64()
        .and_then(|value| u32::try_from(value).ok())
        .ok_or("thinking_token_budget must be a positive 32-bit integer")?;
    if budget == 0 {
        return Err("thinking_token_budget must be positive; disable thinking with enable_thinking=false or reasoning effort none".into());
    }
    Ok(budget)
}

fn options(body: &Value, protocol: Protocol) -> Result<Options> {
    let mut options = Options::default();
    options.sampling = serde_json::from_value(body.clone())
        .map_err(|e| format!("invalid sampling options: {e}"))?;
    if let Some(budget) = body.get("thinking_token_budget").filter(|v| !v.is_null()) {
        if protocol == Protocol::Completions {
            return Err("thinking_token_budget is not supported by raw Completions".into());
        }
        options.thinking_budget = Some(ThinkingBudget::Required(u64::from(thinking_budget(
            budget,
        )?)));
    }
    let limit = match protocol {
        Protocol::Responses => body.get("max_output_tokens"),
        Protocol::Chat => body
            .get("max_completion_tokens")
            .filter(|v| !v.is_null())
            .or_else(|| body.get("max_tokens")),
        _ => body.get("max_tokens"),
    };
    options.output_limit = limit
        .filter(|v| !v.is_null())
        .map(|v| {
            v.as_u64()
                .ok_or("output limit must be a nonnegative integer")
        })
        .transpose()?;
    options.stops = strings(body.get(if protocol == Protocol::Messages {
        "stop_sequences"
    } else {
        "stop"
    }))?;
    if present(body, "stop_token_sequences") {
        options.token_stops = serde_json::from_value(body["stop_token_sequences"].clone())
            .map_err(|_| "invalid stop_token_sequences")?;
    }
    if present(body, "stop_token_ids") {
        let ids: Vec<u32> = serde_json::from_value(body["stop_token_ids"].clone())
            .map_err(|_| "invalid stop_token_ids")?;
        options
            .token_stops
            .extend(ids.into_iter().map(|id| vec![id]));
    }
    options.ignore_eos = flag(body, "ignore_eos", false)?;
    let format = match protocol {
        Protocol::Responses => &body["text"]["format"],
        Protocol::Messages => &body["output_config"]["format"],
        _ => &body["response_format"],
    };
    if !format.is_null() {
        options.constraint = match field(format, "type")? {
            "text" => None,
            "json_object" => Some(Constraint::JsonObject),
            "json_schema" => Some(Constraint::Json(
                if protocol == Protocol::Chat {
                    &format["json_schema"]["schema"]
                } else {
                    &format["schema"]
                }
                .clone(),
            )),
            _ => return Err(unsupported("unsupported output format")),
        };
    }
    if present(body, "grammar") {
        if !format.is_null() {
            return Err("grammar and output format are mutually exclusive".into());
        }
        options.constraint = Some(Constraint::Lark(field(body, "grammar")?.into()));
    }
    let effort = if protocol == Protocol::Responses {
        body["reasoning"].get("effort")
    } else if protocol == Protocol::Messages {
        body["output_config"].get("effort")
    } else {
        None
    };
    if let Some(effort) = effort.filter(|v| !v.is_null()) {
        let native = match effort.as_str() {
            Some("none") => {
                options
                    .template
                    .insert("enable_thinking".into(), false.into());
                None
            }
            Some("minimal") => Some(Effort::Minimal),
            Some("low") => Some(Effort::Low),
            Some("medium" | "high") => Some(Effort::High),
            Some("xhigh" | "max") => Some(Effort::Max),
            _ => return Err("unsupported reasoning effort".into()),
        };
        if let Some(native) = native {
            options.effort = Some(native);
            if protocol == Protocol::Responses {
                options
                    .template
                    .insert("enable_thinking".into(), true.into());
            }
        }
    }
    if protocol == Protocol::Messages && present(body, "thinking") {
        let enabled = match field(&body["thinking"], "type")? {
            "disabled" => false,
            "enabled" | "adaptive" => true,
            _ => return Err("unsupported thinking mode".into()),
        };
        options
            .template
            .insert("enable_thinking".into(), enabled.into());
        if let Some(budget) = body["thinking"].get("budget_tokens").filter(|v| !v.is_null()) {
            let budget = budget
                .as_u64()
                .ok_or("thinking budget must be a positive integer")?;
            if budget == 0 {
                return Err("thinking budget must be positive".into());
            }
            if enabled {
                if options
                    .thinking_budget
                    .is_some_and(|value| value.tokens() != budget)
                {
                    return Err(
                        "thinking_token_budget and thinking.budget_tokens must match".into(),
                    );
                }
                options
                    .thinking_budget
                    .get_or_insert(ThinkingBudget::Advisory(budget));
            }
            options.effort.get_or_insert(if budget <= 5000 {
                Effort::Low
            } else if budget <= 20000 {
                Effort::High
            } else {
                Effort::Max
            });
        }
    }
    for key in [
        "documents",
        "enable_thinking",
        "reasoning_effort",
        "continue_final_message",
        "add_generation_prompt",
    ] {
        if let Some(value) = body.get(key) {
            options.template.insert(key.into(), value.clone());
        }
    }
    if present(body, "chat_template_kwargs") {
        options.template.extend(
            body["chat_template_kwargs"]
                .as_object()
                .ok_or("chat_template_kwargs must be an object")?
                .clone(),
        );
    }
    Ok(options)
}
fn raw_prompts(value: &Value) -> Result<Vec<RawPrompt>> {
    if let Some(text) = value.as_str() {
        return Ok(vec![RawPrompt::Text(text.into())]);
    }
    let items = value
        .as_array()
        .filter(|v| !v.is_empty())
        .ok_or("prompt must be text or a nonempty prompt/token array")?;
    if items[0].is_number() {
        return Ok(vec![RawPrompt::Tokens(
            serde_json::from_value(value.clone()).map_err(|_| "prompt token IDs must be uint32")?,
        )]);
    }
    if !items.iter().all(|v| v.is_string()) && !items.iter().all(|v| v.is_array()) {
        return Err("prompt batch must contain either only strings or only token arrays".into());
    }
    items
        .iter()
        .map(|v| {
            if let Some(text) = v.as_str() {
                Ok(RawPrompt::Text(text.into()))
            } else {
                let ids: Vec<u32> = serde_json::from_value(v.clone())
                    .map_err(|_| "prompt batch must contain only text or token arrays")?;
                if ids.is_empty() {
                    return Err("token prompts cannot be empty".into());
                }
                Ok(RawPrompt::Tokens(ids))
            }
        })
        .collect()
}
fn generation_logprobs(protocol: Protocol, body: &Value) -> Result<Option<LogprobOptions>> {
    let integer = |key: &str| -> Result<usize> {
        match body.get(key).filter(|v| !v.is_null()) {
            None => Ok(0),
            Some(v) => v.as_u64().filter(|n| *n <= crate::execution::MAX_TOP_LOGPROBS as u64).map(|n| n as usize)
                .ok_or_else(|| format!("{key} must be an integer from 0 through {}", crate::execution::MAX_TOP_LOGPROBS).into()),
        }
    };
    let token_ids = flag(body, "return_tokens_as_token_ids", false)?;
    let (enabled, top) = match protocol {
        Protocol::Chat => {
            let enabled = flag(body, "logprobs", false)?;
            let top = integer("top_logprobs")?;
            if top > 0 && !enabled { return Err("top_logprobs requires logprobs: true".into()); }
            (enabled, top)
        }
        Protocol::Completions => {
            if present(body, "top_logprobs") { return Err("completions use integer logprobs".into()); }
            let enabled = present(body, "logprobs");
            let top = integer("logprobs")?;
            (enabled, top)
        }
        Protocol::Responses => {
            if present(body, "logprobs") { return Err("responses use include and top_logprobs".into()); }
            let enabled = body.get("include").and_then(Value::as_array).is_some_and(|items|
                items.iter().any(|v| v == "message.output_text.logprobs"));
            let top = integer("top_logprobs")?;
            if top > 0 && !enabled { return Err("top_logprobs requires include: [\"message.output_text.logprobs\"]".into()); }
            (enabled, top)
        }
        Protocol::Messages => {
            if body.get("logprob_token_ids").is_some()
                || ["logprobs", "top_logprobs", "return_tokens_as_token_ids"]
                    .iter()
                    .any(|key| present(body, key))
            {
                return Err(unsupported("/v1/messages does not support logprob fields"));
            }
            (false, 0)
        }
    };
    let mut ids = Vec::new();
    if let Some(value) = body.get("logprob_token_ids").filter(|v| !v.is_null()) {
        let values = value
            .as_array()
            .ok_or("logprob_token_ids must be an array of integers")?;
        if values.len() > crate::execution::MAX_LOGPROB_TOKEN_IDS {
            return Err("logprob_token_ids accepts at most 128 elements".into());
        }
        for value in values {
            let id = value
                .as_u64()
                .and_then(|id| i32::try_from(id).ok())
                .ok_or("logprob_token_ids must contain nonnegative int32 integers")?
                as u32;
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
    }
    if !ids.is_empty() && !enabled {
        return Err("logprob_token_ids requires generation logprobs to be enabled".into());
    }
    Ok(enabled.then_some(LogprobOptions {
        top,
        token_ids,
        ids,
    }))
}

pub(crate) fn decode(protocol: Protocol, body: Value, count: bool) -> Result<Decoded> {
    body.as_object().ok_or("request must be a JSON object")?;
    reject(
        &body,
        &["logit_bias", "guided_json"],
    )?;
    if protocol == Protocol::Responses {
        reject(
            &body,
            &[
                "previous_response_id",
                "conversation",
                "background",
                "store",
            ],
        )?;
        if present(&body, "truncation") && body["truncation"] != "disabled" {
            return Err(unsupported("automatic truncation is not supported"));
        }
        if present(&body, "service_tier")
            && body["service_tier"] != "auto"
            && body["service_tier"] != "default"
        {
            return Err(unsupported("service tiers are not supported"));
        }
    }
    if protocol == Protocol::Completions {
        reject(&body, &["suffix"])?;
    }
    let logprobs = generation_logprobs(protocol, &body)?;
    let prompt_logprobs = if present(&body, "prompt_logprobs") {
        if !matches!(protocol, Protocol::Completions | Protocol::Chat) {
            return Err("prompt_logprobs requires completions or chat completions".into());
        }
        if flag(&body, "stream", false)? {
            return Err("prompt_logprobs are not available when stream=true; use echo with logprobs".into());
        }
        Some(body["prompt_logprobs"].as_u64()
            .filter(|n| *n <= crate::execution::MAX_TOP_LOGPROBS as u64)
            .ok_or("prompt_logprobs must be an integer from 0 through 20")? as usize)
    } else if protocol == Protocol::Completions && flag(&body, "echo", false)? {
        logprobs.as_ref().map(|v| v.top)
    } else { None };
    let choices = body
        .get("n")
        .filter(|v| !v.is_null())
        .map(|v| {
            v.as_u64()
                .and_then(|v| usize::try_from(v).ok())
                .filter(|v| *v > 0)
                .ok_or("n must be positive")
        })
        .transpose()?
        .unwrap_or(1);
    if protocol != Protocol::Completions && choices != 1 {
        return Err("n must be 1".into());
    }
    if present(&body, "best_of") && body["best_of"].as_u64() != Some(choices as u64) {
        return Err(unsupported(
            "best_of must equal n; ranked sampling is not supported",
        ));
    }
    let tool_choice = match body.get("tool_choice").filter(|v| !v.is_null()) {
        None => ToolChoice::Auto,
        Some(v) if v == "auto" || v["type"] == "auto" => ToolChoice::Auto,
        Some(v) if v == "none" || v["type"] == "none" => ToolChoice::None,
        Some(v) if v == "required" => ToolChoice::Required,
        Some(v) if protocol == Protocol::Messages && v["type"] == "any" => ToolChoice::Required,
        Some(v) if protocol == Protocol::Chat && v["type"] == "function" => {
            ToolChoice::Named(field(&v["function"], "name")?.into())
        }
        Some(v) if protocol == Protocol::Messages && v["type"] == "tool" => {
            ToolChoice::Named(field(v, "name")?.into())
        }
        Some(v)
            if protocol == Protocol::Responses
                && (v["type"] == "function" || v["type"] == "custom") =>
        {
            ToolChoice::Named(field(v, "name")?.into())
        }
        _ => {
            return Err(unsupported(
                "tool_choice supports auto, none, required (Anthropic any), or a named tool",
            ));
        }
    };
    let parallel_tool_calls = flag(&body, "parallel_tool_calls", true)?;
    let disable_parallel_tool_use = flag(&body["tool_choice"], "disable_parallel_tool_use", false)?;
    let parallel_tool_calls = parallel_tool_calls && !disable_parallel_tool_use;
    reject(
        &body,
        &[
            "audio",
            "mcp_servers",
            "container",
            "prediction",
            "max_tool_calls",
        ],
    )?;
    if present(&body, "context_management") {
        let context = &body["context_management"];
        let preserve = protocol == Protocol::Messages
            && context.as_object().is_some_and(|v| v.len() == 1)
            && context["edits"].as_array().is_some_and(|edits| {
                edits
                    .iter()
                    .all(|edit| edit == &json!({"type":"clear_thinking_20251015","keep":"all"}))
            });
        if !preserve {
            return Err(unsupported(
                "context_management supports only preserving all thinking; server-side context edits are not supported",
            ));
        }
    }
    if protocol == Protocol::Completions {
        reject(&body, &["tools", "tool_choice"])?;
    }
    let mut tools = vec![];
    if present(&body, "tools") {
        for v in array(&body, "tools")? {
            if protocol == Protocol::Responses && v["type"] == "namespace" {
                let namespace = field(v, "name")?;
                for nested in array(v, "tools")? {
                    let mut t = tool(nested, protocol, Some(namespace))?;
                    if let Some(description) = v["description"].as_str() {
                        t.description = format!("{description}\n{}", t.description);
                    }
                    tools.push(t);
                }
            } else {
                tools.push(tool(v, protocol, None)?);
            }
        }
    }
    let mut options = options(&body, protocol)?;
    if protocol == Protocol::Messages && !count && options.output_limit.is_none() {
        return Err("max_tokens is required".into());
    }
    let mut checkpoint_hints = Vec::new();
    let input = match protocol {
        Protocol::Chat => Input::Conversation(chat_messages(&body)?),
        Protocol::Messages => {
            let messages = anthropic_messages(&body, &mut checkpoint_hints)?;
            if messages.last().is_some_and(|m| m.role == Role::Assistant) {
                options
                    .template
                    .entry("continue_final_message")
                    .or_insert(json!(true));
            }
            Input::Conversation(messages)
        }
        Protocol::Responses => Input::Conversation(responses_messages(&body)?),
        Protocol::Completions => Input::Raw(raw_prompts(&body["prompt"])?),
    };
    options.logprobs = logprobs;
    options.prompt_logprobs = prompt_logprobs;
    let request = Request {
        input,
        checkpoint_hints,
        options,
        tools,
        tool_choice,
        parallel_tool_calls,
        choices,
    };
    request.validate()?;
    if protocol == Protocol::Responses {
        if let ToolChoice::Named(name) = &request.tool_choice {
            let selected = request.tools.iter()
                .find(|tool| tool.native_name() == *name)
                .ok_or("selected tool is not declared in tools")?;
            if selected.custom != (body["tool_choice"]["type"] == "custom") {
                return Err("tool_choice type must match the selected tool definition".into());
            }
        }
    }
    let observation = ObservationOptions::parse(body.as_object().unwrap())?;
    if protocol != Protocol::Chat && (observation.token_ids || observation.timestamps) {
        return Err(unsupported(
            "token observation extensions require /v1/chat/completions",
        ));
    }
    Ok(Decoded {
        request,
        stream: flag(&body, "stream", false)?,
        echo: flag(&body, "echo", false)?,
        observation,
        metadata: body.get("metadata").cloned().unwrap_or_else(|| json!({})),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn string_stops(value: Option<&Value>) -> Result<Vec<String>> {
        let decoded = decode(
            Protocol::Chat,
            json!({"messages":[{"role":"user","content":"x"}],"stop":value}),
            false,
        )?;
        crate::server::stops::validate_strings(&decoded.request.options.stops, false, false)?;
        Ok(decoded.request.options.stops)
    }
    fn normalize_tool_arguments(message: &mut Value) -> Result<()> {
        let decoded = chat_messages(&json!({"messages":[message]}))?;
        *message = template_messages(&decoded, ToolChoice::Auto, &[], true)?.remove(0);
        Ok(())
    }
    #[test]
    fn standard_string_stop_values_are_validated_without_coercion() {
        assert!(string_stops(None).unwrap().is_empty());
        assert!(string_stops(Some(&Value::Null)).unwrap().is_empty());
        assert_eq!(
            string_stops(Some(&serde_json::json!("\nEND_EDIT"))).unwrap(),
            vec!["\nEND_EDIT"]
        );
        assert_eq!(
            string_stops(Some(&serde_json::json!(["a", "世界"]))).unwrap(),
            vec!["a", "世界"]
        );
        for value in [
            serde_json::json!(false),
            serde_json::json!(1),
            serde_json::json!(["ok", 2]),
            serde_json::json!(""),
            serde_json::json!(["a", "b", "c", "d", "e"]),
        ] {
            assert!(string_stops(Some(&value)).is_err(), "{value}");
        }
    }
    #[test]
    fn response_tool_arguments_round_trip_without_changing_json_types() {
        let expected = serde_json::json!({"query":"café 世界","options":{"count":2,"enabled":true,"none":null,"items":[1,"✓"]}});
        let mut message = serde_json::json!({"role":"assistant","tool_calls":[{"type":"function","id":"call_1","function":{"name":"search","arguments":expected.to_string()}}]});
        normalize_tool_arguments(&mut message).unwrap();
        assert_eq!(message["tool_calls"][0]["function"]["arguments"], expected);
        assert_eq!(message["tool_calls"][0]["id"], "call_1");
        normalize_tool_arguments(&mut message).unwrap();
        for invalid in ["{", "null", "[]", "\"text\"", "1"] {
            let mut message =
                serde_json::json!({"tool_calls":[{"function":{"arguments":invalid}}]});
            assert!(normalize_tool_arguments(&mut message).is_err());
        }
    }
}
#[cfg(test)]
mod native_prompt_tests {
    use super::*;
    use crate::{
        frontend::{
            Asset, Frontend, FrontendAssets, FrontendSettings, PromptEncoder, TemplateEncoding,
            TokenizerSettings,
        },
        grammar::GrammarCompiler,
    };
    use sha2::{Digest, Sha256};
    use tokenizers::{AddedToken, Tokenizer};

    fn native(encoder: PromptEncoder) -> (Frontend, GrammarCompiler, crate::model::ModelConfig) {
        renderer(encoder, None, TemplateEncoding::Text)
    }

    fn renderer(
        encoder: PromptEncoder,
        template: Option<&str>,
        encoding: TemplateEncoding,
    ) -> (Frontend, GrammarCompiler, crate::model::ModelConfig) {
        let mut alphabet: Vec<_> = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet()
            .into_iter()
            .collect();
        alphabet.sort_unstable();
        let mut vocabulary = alphabet
            .into_iter()
            .enumerate()
            .map(|(id, c)| (c.to_string(), id as u32))
            .collect::<tokenizers::models::bpe::Vocab>();
        for text in ["ĊĊ", "ĊĊĊ"] {
            vocabulary.insert(text.into(), vocabulary.len() as u32);
        }
        let model = tokenizers::models::bpe::BPE::builder()
            .vocab_and_merges(
                vocabulary,
                vec![("Ċ".into(), "Ċ".into()), ("ĊĊ".into(), "Ċ".into())],
            )
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(
            tokenizers::pre_tokenizers::byte_level::ByteLevel::new(false, true, false),
        ));
        tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
        tokenizer
            .add_special_tokens(
                [
                    "<｜begin▁of▁sentence｜>",
                    "<｜end▁of▁sentence｜>",
                    "<｜Assistant｜>",
                    "<think>",
                    "</think>",
                    "<｜deepseek_image｜>",
                    "<|im_start|>",
                    "<|im_end|>",
                    "<|vision_start|>",
                    "<|image_pad|>",
                    "<|video_pad|>",
                    "<|vision_end|>",
                    "<|open|>",
                    "<|close|>",
                    "<|sep|>",
                    "<|end_of_msg|>",
                    "<|media_pad|>",
                    "<|media_begin|>",
                    "<|media_content|>",
                    "<|media_end|>",
                    "<｜System｜>",
                    "<｜User｜>",
                ]
                .map(|s| AddedToken::from(s, true)),
            )
            .unwrap();
        tokenizer.add_tokens([AddedToken::from("｜DSML｜", false)]).unwrap();
        let eos = tokenizer.token_to_id("<｜end▁of▁sentence｜>").unwrap();
        let bytes = tokenizer.to_string(false).unwrap();
        let path = std::env::temp_dir().join(format!(
            "flare-native-protocol-{}-{:?}.json",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::write(&path, &bytes).unwrap();
        let template = (encoder == PromptEncoder::Jinja).then(|| {
            let text = template.unwrap_or(concat!(
                "{% for m in messages %}<{{ m.role }}>",
                "{% if m.content is string %}{{ m.content }}{% else %}",
                "{% for block in m.content %}{{ block.text }}{% if not loop.last %}{{ '\\n\\n' }}{% endif %}{% endfor %}{% endif %}",
                "</{{ m.role }}>{% mark %}{% endfor %}<assistant>"
            ));
            let path = path.with_extension("jinja");
            std::fs::write(&path, text).unwrap();
            Asset {
                path,
                range: None,
                sha256: format!("{:x}", Sha256::digest(text.as_bytes())),
            }
        });
        let template_path = template.as_ref().map(|asset| asset.path.clone());
        let settings: FrontendSettings = serde_json::from_value(
            json!({"tokenizer":{},"eos_token_ids":[eos],"prompt_encoder":i32::from(encoder),
                "image_placeholder_token_id":tokenizer.token_to_id("<｜deepseek_image｜>").unwrap()}),
        )
        .unwrap();
        let frontend = Frontend::load_with_settings(
            FrontendAssets {
                tokenizer: Asset {
                    path: path.clone(),
                    sha256: format!("{:x}", Sha256::digest(bytes.as_bytes())),
                    range: None,
                },
                tokens: TokenizerSettings::default(),
                chat_template: template,
            },
            Default::default(),
            &settings,
        )
        .unwrap()
        .with_template_encoding(encoding);
        std::fs::remove_file(path).unwrap();
        if let Some(path) = template_path {
            std::fs::remove_file(path).unwrap();
        }
        let mut config = crate::frontend::test_support::model_config(
            crate::frontend::test_support::contract("deepseek41"),
        );
        config.eos_token_ids = vec![eos];
        let compiler = GrammarCompiler::new(
            frontend.tokenizer(),
            config.contract.vocabulary,
            &config.eos_token_ids,
            &[],
        )
        .unwrap();
        (frontend, compiler, config)
    }

    #[test]
    fn tool_argument_sources_survive_protocol_decoding_and_native_preparation() {
        let source = r#"{"z":1e2,"a":[1,  2],"z":-0,"s":"café\n"}"#;
        let (frontend, compiler, mut config) = renderer(
            PromptEncoder::Jinja,
            Some(include_str!("../../../../ignition/recipes/kimi31/assets/chat_template.jinja")),
            TemplateEncoding::TokenSegments,
        );
        config.contract.frontend.rejected_tokens.clear();
        let expected_argument = "<|open|>argument key=\"z\" type=\"number\"<|sep|>";
        for (protocol, body) in [
            (Protocol::Chat, json!({"messages":[{"role":"assistant","tool_calls":[{"id":"a","type":"function","function":{"name":"lookup","arguments":source}}]}, {"role":"user","content":"Continue"}]})),
            (Protocol::Responses, json!({"input":[{"type":"function_call","call_id":"a","name":"lookup","arguments":source},{"role":"user","content":"Continue"}]})),
            (Protocol::Messages, json!({"max_tokens":4,"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"a","name":"lookup","input":source}]},{"role":"user","content":"Continue"}]})),
        ] {
            for invalid in ["{", "null", "[]", "1", r#"{"x":1} trailing"#] {
                let mut malformed = body.clone();
                let pointer = match protocol {
                    Protocol::Chat => "/messages/0/tool_calls/0/function/arguments",
                    Protocol::Responses => "/input/0/arguments",
                    Protocol::Messages => "/messages/0/content/0/input",
                    _ => unreachable!(),
                };
                *malformed.pointer_mut(pointer).unwrap() = json!(invalid);
                let error = decode(protocol, malformed, false).err().unwrap().to_string();
                assert!(error.contains("invalid tool arguments JSON") || error.contains("tool arguments must be a JSON object"), "{error}");
            }
            let decoded = decode(protocol, body, false).unwrap();
            let Input::Conversation(messages) = &decoded.request.input else { panic!() };
            assert!(matches!(&messages[0].parts[0], Part::ToolCall { arguments_source: Some(original), arguments, .. }
                if original == source && arguments["a"] == json!([1,2])));
            let prepared = crate::server::request::prepare_inference(
                &frontend, None, &compiler, &config, &decoded.request, 0, false,
            ).unwrap();
            assert!(prepared.rendered_prompt.contains(&format!("{expected_argument}1e2<|close|>argument")), "{}", prepared.rendered_prompt);
            assert!(prepared.rendered_prompt.contains(&format!("{expected_argument}-0<|close|>argument")));
            assert!(prepared.rendered_prompt.contains("<|sep|>[1,  2]<|close|>argument"));
            assert!(prepared.rendered_prompt.contains("<|sep|>café\n<|close|>argument"));
        }
    }

    #[test]
    fn messages_image_hints_end_after_expansion_for_text_and_segmented_templates() {
        use crate::frontend::images::{
            ImageLimits, ImageProcessor, ImageSettings, QwenImageSettings,
        };
        use crate::model::contract::{ImageLayoutKind, ImageTokenLayout};
        use base64::Engine;
        let mut png = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut png, 64, 64);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            encoder
                .write_header()
                .unwrap()
                .write_image_data(&vec![127; 64 * 64 * 3])
                .unwrap();
        }
        let image = json!({"type":"image","source":{"type":"base64",
            "media_type":"image/png", "data":base64::engine::general_purpose::STANDARD.encode(png)}});
        for (template, encoding, layout, placeholder) in [
            (
                &crate::template_fixtures::template("qwen38", "chat_template")[..],
                TemplateEncoding::Text,
                ImageLayoutKind::Qwen,
                "<|image_pad|>",
            ),
            (
                include_str!("../../../../ignition/recipes/kimi3/assets/chat_template.jinja"),
                TemplateEncoding::TokenSegments,
                ImageLayoutKind::Moonvit,
                "<|media_pad|>",
            ),
        ] {
            let (frontend, compiler, mut config) =
                renderer(PromptEncoder::Jinja, Some(template), encoding);
            let token = |text: &str| frontend.encode(text, false).unwrap()[0];
            let settings = if layout == ImageLayoutKind::Qwen {
                ImageSettings::Qwen(QwenImageSettings {
                    min_pixels: 4096,
                    max_pixels: 65536,
                    position_side: 48,
                    image_token_id: token(placeholder),
                    video_token_id: token("<|video_pad|>"),
                })
            } else {
                ImageSettings::Moonvit {}
            };
            let processor = ImageProcessor::load(settings, ImageLimits::default()).unwrap();
            let geometry = processor.geometry(64, 64).unwrap();
            config.contract.vision = crate::model::VisionCapabilities {
                feature_slots: 2,
                features_per_request: geometry.tokens * 2,
                max_images: 2,
                patches_per_feature: 4,
                patch_capacities: vec![geometry.patches],
            };
            config.contract.frontend.image_layout = Some(layout);
            config.contract.frontend.position_dimensions = if layout == ImageLayoutKind::Qwen {
                3
            } else {
                0
            };
            config.contract.frontend.image_tokens = Some(ImageTokenLayout {
                feature_token: token(placeholder),
                span: None,
            });
            config.contract.frontend.rejected_tokens.clear();
            let mut marked_image = image.clone();
            marked_image["cache_control"] = json!({"type":"ephemeral"});
            let body = json!({"max_tokens":4,"messages":[{"role":"user","content":[
                image, {"type":"text","text":"Between images","cache_control":{}}, marked_image,
                {"type":"text","text":"Changing suffix"}]}]});
            let decoded = decode(Protocol::Messages, body.clone(), false).unwrap();
            let prepare = |request| {
                crate::server::request::prepare_inference(
                    &frontend,
                    Some(&processor),
                    &compiler,
                    &config,
                    request,
                    0,
                    false,
                )
                .unwrap()
            };
            let marked = prepare(&decoded.request);
            let mut plain_request = decoded.request.clone();
            plain_request.checkpoint_hints.clear();
            let plain = prepare(&plain_request);
            assert_eq!(marked.rendered_prompt, plain.rendered_prompt);
            assert_eq!(marked.prompt.tokens, plain.prompt.tokens);
            assert_eq!(marked.prompt.conditioning, plain.prompt.conditioning);
            assert_eq!(marked.prompt.conditioned_inputs.len(), 2);
            let cut = marked.prompt.conditioned_inputs[1].token_span.end;
            assert!(marked.prompt.conditioned_inputs[1].token_span.len() > 1);
            assert!(!plain.prompt.message_boundaries.contains(&cut));
            assert!(
                marked.prompt.message_boundaries.contains(&cut),
                "{layout:?}"
            );
            assert_eq!(
                marked.prompt.message_boundaries.len(),
                plain.prompt.message_boundaries.len() + 2
            );
            let mut next = body;
            next["messages"][0]["content"][3]["text"] = json!("Different suffix");
            let next = prepare(&decode(Protocol::Messages, next, false).unwrap().request);
            assert!(next.prompt.tokens.starts_with(&marked.prompt.tokens[..cut]));
            assert!(next.prompt.message_boundaries.contains(&cut));
            assert!(marked.prompt.message_boundaries.iter().all(|end| {
                marked
                    .prompt
                    .conditioned_inputs
                    .iter()
                    .all(|image| *end <= image.token_span.start || *end >= image.token_span.end)
            }));
        }
    }

    #[test]
    fn classifier_history_hints_survive_separator_merges_in_the_next_call() {
        for encoder in [
            PromptEncoder::DeepseekV4,
            PromptEncoder::DeepseekV41,
            PromptEncoder::Jinja,
        ] {
            let (frontend, compiler, config) = native(encoder);
            let mut history: Vec<_> = (0..20)
                .map(|index| {
                    json!({
                        "type": "text",
                        "text": format!("Earlier action {index}: read file and check output.\n"),
                    })
                })
                .collect();
            history.last_mut().unwrap()["cache_control"] = json!({"type": "ephemeral"});
            let action = json!({"type": "text", "text": "Proposed action: run the tests.\n",
                "cache_control": {"type": "ephemeral"}});
            let prepare = |blocks| {
                let decoded = decode(
                    Protocol::Messages,
                    json!({"max_tokens": 4,
                        "system": "Review actions for safety.",
                        "messages": [{"role": "user", "content": blocks}]}),
                    false,
                )
                .unwrap();
                let prepared = crate::server::request::prepare_inference(
                    &frontend,
                    None,
                    &compiler,
                    &config,
                    &decoded.request,
                    0,
                    false,
                )
                .unwrap();
                let mut plain = decoded.request.clone();
                plain.checkpoint_hints.clear();
                let plain = crate::server::request::prepare_inference(
                    &frontend, None, &compiler, &config, &plain, 0, false,
                )
                .unwrap();
                assert_eq!(prepared.prompt.tokens, plain.prompt.tokens);
                assert_eq!(prepared.rendered_prompt, plain.rendered_prompt);
                (prepared, plain)
            };
            let mut blocks = history.clone();
            blocks.extend([
                action.clone(),
                json!({"type": "text", "text": "Classify this action."}),
            ]);
            let (first, plain) = prepare(blocks);
            history.extend([
                action,
                json!({"type": "text", "text": "New action: inspect results."}),
                json!({"type": "text", "text": "Classify this action."}),
            ]);
            let (next, _) = prepare(history);
            let last = "Earlier action 19: read file and check output.\n";
            let end = first.rendered_prompt.find(last).unwrap() + last.len();
            let encoding = frontend
                .tokenizer()
                .encode(first.rendered_prompt.as_str(), false)
                .unwrap();
            let offsets = encoding.get_offsets();
            let cut = offsets.iter().take_while(|&&(_, stop)| stop <= end).count();
            assert!(cut > 500, "{encoder:?}");
            assert!(offsets[cut].0 < end && offsets[cut].1 > end, "{encoder:?}");
            assert!(
                first.prompt.message_boundaries.contains(&cut),
                "{encoder:?}"
            );
            assert!(!plain.prompt.message_boundaries.contains(&cut));
            assert!(next.prompt.tokens.starts_with(&first.prompt.tokens[..cut]));
            assert_eq!(
                first.prompt.message_boundaries.len(),
                plain.prompt.message_boundaries.len() + 2
            );
        }
    }

    #[test]
    fn messages_hint_cap_ignores_system_assistant_and_nested_result_controls() {
        let image = json!({"type": "image", "source": {"type": "base64",
            "media_type": "image/png", "data": "a"}, "cache_control": {}});
        let body = json!({"max_tokens": 4,
            "system": [{"type": "text", "text": "Policy", "cache_control": {}}],
            "messages": [
                {"role": "assistant", "content": [{"type": "text", "text": "Calls", "cache_control": {}},
                    {"type": "tool_use", "id": "a", "name": "lookup", "input": {}},
                    {"type": "tool_use", "id": "b", "name": "lookup", "input": {}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "b", "content": [
                        {"type": "text", "text": "B", "cache_control": {}}]},
                    {"type": "tool_result", "tool_use_id": "a", "is_error": true,
                        "content": [image], "cache_control": {}},
                    {"type": "text", "text": "History", "cache_control": {}},image,
                    {"type": "text", "text": "Action", "cache_control": {}},
                    {"type": "text", "text": "Ignored fifth hint", "cache_control": {}}]}]});
        let decoded = decode(Protocol::Messages, body, false).unwrap();
        assert_eq!(
            decoded.request.checkpoint_hints,
            [(2, 2), (2, 3), (2, 4), (2, 5)]
        );
        let Input::Conversation(messages) = &decoded.request.input else {
            panic!()
        };
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[2].parts.len(), 6);
        for encoder in [PromptEncoder::DeepseekV4, PromptEncoder::DeepseekV41] {
            let (frontend, _, _) = native(encoder);
            let render = |hints: &[(usize, usize)]| {
                frontend
                    .render_chat(
                        &template_messages(messages, ToolChoice::Auto, hints, true).unwrap(),
                        &serde_json::Map::new(),
                    )
                    .unwrap()
            };
            let marked = render(&decoded.request.checkpoint_hints);
            let plain = render(&[]);
            assert_eq!(marked.text, plain.text);
            assert_eq!(marked.token_ids, plain.token_ids);
            assert_eq!(marked.image_sources, plain.image_sources);
            assert_eq!(marked.image_sources.len(), 2);
            let end = marked
                .token_ids
                .iter()
                .rposition(|&id| id == frontend.image_placeholder_token_id().unwrap())
                .unwrap()
                + 1;
            assert!(marked.message_boundaries.contains(&end));
            assert!(!plain.message_boundaries.contains(&end));
        }
    }

    #[test]
    fn cache_controls_do_not_change_other_protocols() {
        for (protocol, body) in [
            (
                Protocol::Chat,
                json!({"messages": [{"role": "user", "content": [
                {"type": "text", "text": "History", "cache_control": {}},
                {"type": "text", "text": "Action", "cache_control": {}}]}]}),
            ),
            (
                Protocol::Responses,
                json!({"input": [{"role": "user", "content": [
                {"type": "input_text", "text": "History", "cache_control": {}}]}]}),
            ),
            (
                Protocol::Completions,
                json!({"prompt": "History", "cache_control": {}}),
            ),
        ] {
            let mut plain = body.clone();
            plain.as_object_mut().unwrap().remove("cache_control");
            for key in ["messages", "input"] {
                if let Some(messages) = plain[key].as_array_mut() {
                    for message in messages {
                        for block in message["content"].as_array_mut().unwrap() {
                            block.as_object_mut().unwrap().remove("cache_control");
                        }
                    }
                }
            }
            let decoded = decode(protocol, body, false).unwrap();
            let plain = decode(protocol, plain, false).unwrap();
            assert!(decoded.request.checkpoint_hints.is_empty());
            assert_eq!(decoded.request, plain.request);
        }
    }

    #[test]
    fn every_protocol_clamps_sampling_before_native_admission() {
        use crate::model::{SamplingRange, SamplingRanges};
        let (frontend, compiler, mut config) = native(PromptEncoder::DeepseekV4);
        for (protocol, input) in [
            (Protocol::Completions, json!({"prompt":"hello"})),
            (
                Protocol::Chat,
                json!({"messages":[{"role":"user","content":"hello"}]}),
            ),
            (
                Protocol::Messages,
                json!({"messages":[{"role":"user","content":"hello"}]}),
            ),
            (Protocol::Responses, json!({"input":"hello"})),
        ] {
            for bounded in [false, true] {
                config.sampling_ranges = if bounded {
                    SamplingRanges {
                        temperature: Some(SamplingRange { min: 0.5, max: 1.5 }),
                        top_k: Some(SamplingRange { min: -1, max: 100 }),
                        ..Default::default()
                    }
                } else {
                    SamplingRanges::default()
                };
                for temperature in [0., 1e100] {
                    let mut body = input.clone();
                    body.as_object_mut().unwrap().extend(
                        json!({
                            "model":config.model_id, "max_tokens":4, "max_output_tokens":4,
                            "temperature":temperature, "top_k":u64::MAX, "seed":u64::MAX,
                            "top_p":-1e100, "min_p":1e100, "repetition_penalty":0.,
                            "presence_penalty":1e100, "frequency_penalty":-1e100,
                        })
                        .as_object()
                        .unwrap()
                        .clone(),
                    );
                    let decoded = decode(protocol, body, false).unwrap();
                    let prepared = crate::server::request::prepare_inference(
                        &frontend,
                        None,
                        &compiler,
                        &config,
                        &decoded.request,
                        0,
                        false,
                    )
                    .unwrap();
                    let sampling = prepared.prompt.sampling;
                    let mut expected = config.sampling_defaults.clone();
                    if bounded {
                        expected.temperature = if temperature == 0. { 0.5 } else { 1.5 };
                        expected.top_k = 100;
                    }
                    assert_eq!(sampling.values(), expected.values(), "{protocol:?}");
                    assert_eq!(sampling.top_k, expected.top_k);
                    assert_eq!(sampling.seed, u64::MAX);
                }
            }
        }
    }

    #[test]
    fn prompt_only_completions_and_chat_zero_limit_follow_endpoint_rules() {
        let (frontend, compiler, mut config) = native(PromptEncoder::DeepseekV41);
        config.contract.graphs.push(crate::model::contract::GraphContract { tuned_bindings: vec![],
            graph: "score".into(), kind: crate::model::contract::GraphOperation::ScoredPrefill,
            inputs: vec![], state: vec![], state_rows_per_request: 1,
            draft_cursor: None, embedding_tables: vec![], pending_token: None,
        });
        let completion = decode(Protocol::Completions,
            json!({"prompt":[7,42],"max_tokens":0,"prompt_logprobs":0}), false).unwrap();
        let prepared = crate::server::request::prepare_inference(
            &frontend, None, &compiler, &config, &completion.request, 0, false).unwrap();
        assert_eq!(prepared.prompt.max_new_tokens, 0);
        assert!(prepared.prompt.prompt_logprobs);
        let chat = decode(Protocol::Chat,
            json!({"messages":[{"role":"user","content":"Hello"}],"max_tokens":0,"prompt_logprobs":0}), false).unwrap();
        let error = crate::server::request::prepare_inference(
            &frontend, None, &compiler, &config, &chat.request, 0, false).err().unwrap();
        assert!(error.to_string().contains("output limit must be between 1"), "{error}");
    }

    #[test]
    fn raw_completion_bypasses_native_conversation_options_and_keeps_token_ids() {
        for encoder in [PromptEncoder::DeepseekV4, PromptEncoder::DeepseekV41] {
            let (frontend, compiler, config) = native(encoder);
            for (prompt, expected) in [
                (
                    json!("Raw prompt"),
                    frontend.encode("Raw prompt", true).unwrap(),
                ),
                (json!([7, 42, 3]), vec![7, 42, 3]),
            ] {
                let decoded = decode(
                    Protocol::Completions,
                    json!({"model":config.model_id,"prompt":prompt,"max_tokens":0,"echo":true,
                    "chat_template_kwargs":{"drop_thinking":true,"add_generation_prompt":false}}),
                    false,
                )
                .unwrap();
                let prepared = crate::server::request::prepare_inference(
                    &frontend,
                    None,
                    &compiler,
                    &config,
                    &decoded.request,
                    0,
                    false,
                )
                .unwrap();
                assert_eq!(prepared.prompt.tokens, expected);
                assert_eq!(prepared.prompt.max_new_tokens, 0);
                assert!(prepared.output.is_none());
                assert!(prepared.prompt.message_boundaries.is_empty());
                assert!(prepared.prompt.grammar.is_none());
            }
        }
    }

    #[test]
    fn anthropic_assistant_prefill_is_rejected_by_native_prompt_preparation() {
        for encoder in [PromptEncoder::DeepseekV4, PromptEncoder::DeepseekV41] {
            let (frontend, compiler, config) = native(encoder);
            let decoded = decode(
                Protocol::Messages,
                json!({"model":config.model_id,"max_tokens":32,"messages":[
                {"role":"user","content":"Hello"},{"role":"assistant","content":"Prefix"}]}),
                false,
            )
            .unwrap();
            assert_eq!(
                decoded.request.options.template["continue_final_message"],
                true
            );
            let error = crate::server::request::prepare_inference(
                &frontend,
                None,
                &compiler,
                &config,
                &decoded.request,
                0,
                false,
            )
            .err()
            .unwrap();
            assert!(error.to_string().contains("does not support continue_final_message"), "{error}");
        }
    }

    #[test]
    fn thinking_token_budget_decodes_on_conversation_protocols() {
        for protocol in [Protocol::Chat, Protocol::Responses, Protocol::Messages] {
            let mut body = match protocol {
                Protocol::Responses => json!({"input":"Hello"}),
                _ => json!({"messages":[{"role":"user","content":"Hello"}],"max_tokens":32}),
            };
            body["thinking_token_budget"] = json!(16);
            assert_eq!(
                decode(protocol, body.clone(), false)
                    .unwrap()
                    .request
                    .options
                    .thinking_budget,
                Some(ThinkingBudget::Required(16))
            );
            body["thinking_token_budget"] = Value::Null;
            assert_eq!(
                decode(protocol, body.clone(), false)
                    .unwrap()
                    .request
                    .options
                    .thinking_budget,
                None
            );
            for value in [
                json!(0),
                json!(-1),
                json!(1.5),
                json!("16"),
                json!(true),
                json!(u64::MAX),
            ] {
                body["thinking_token_budget"] = value;
                let error = decode(protocol, body.clone(), false).err().unwrap();
                assert!(
                    error.to_string().contains("thinking_token_budget"),
                    "{error}"
                );
            }
        }
        assert!(
            decode(
                Protocol::Completions,
                json!({"prompt":"Hello","thinking_token_budget":null}),
                false
            )
            .is_ok()
        );
        let error = decode(
            Protocol::Completions,
            json!({"prompt":"Hello","thinking_token_budget":16}),
            false,
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains("thinking_token_budget"));
    }

    #[test]
    fn messages_budgets_agree_and_keep_effort_mapping() {
        for mode in ["enabled", "adaptive"] {
            for (budget, effort) in [
                (32, Effort::Low),
                (6000, Effort::High),
                (21000, Effort::Max),
            ] {
                let mut body = json!({"max_tokens":64,"messages":[{"role":"user","content":"Hello"}],
                    "thinking":{"type":mode,"budget_tokens":budget}});
                let decoded = decode(Protocol::Messages, body.clone(), false).unwrap();
                assert_eq!(
                    decoded.request.options.thinking_budget,
                    Some(ThinkingBudget::Advisory(budget))
                );
                assert_eq!(decoded.request.options.effort, Some(effort));
                body["thinking_token_budget"] = json!(budget);
                body["output_config"] = json!({"effort":"max"});
                assert_eq!(
                    decode(Protocol::Messages, body.clone(), false)
                        .unwrap()
                        .request
                        .options
                        .effort,
                    Some(Effort::Max)
                );
                body["thinking_token_budget"] = json!(budget + 1);
                assert!(
                    decode(Protocol::Messages, body, false)
                        .err()
                        .unwrap()
                        .to_string()
                        .contains("must match")
                );
            }
        }
        let decoded = decode(
            Protocol::Messages,
            json!({"max_tokens":64,
            "messages":[{"role":"user","content":"Hello"}],
            "thinking":{"type":"disabled","budget_tokens":32}}),
            false,
        )
        .unwrap();
        assert_eq!(decoded.request.options.thinking_budget, None);
    }

    #[test]
    fn messages_budget_errors_and_large_values_preserve_messages_semantics() {
        let (frontend, compiler, config) = native(PromptEncoder::DeepseekV41);
        let mut body = json!({"max_tokens":32,"messages":[{"role":"user","content":"Hello"}],
            "thinking":{"type":"enabled","budget_tokens":null}});
        assert_eq!(
            decode(Protocol::Messages, body.clone(), false)
                .unwrap()
                .request
                .options
                .thinking_budget,
            None
        );
        for value in [json!(0), json!(-1), json!(1.5), json!("32"), json!(true)] {
            body["thinking"]["budget_tokens"] = value;
            let error = decode(Protocol::Messages, body.clone(), false)
                .err()
                .unwrap()
                .to_string();
            assert!(error.starts_with("thinking budget must be"), "{error}");
            assert!(!error.contains("thinking_token_budget") && !error.contains("enable_thinking"));
        }
        let budget = u64::MAX;
        body["thinking"]["budget_tokens"] = json!(budget);
        let decoded = decode(Protocol::Messages, body.clone(), false).unwrap();
        assert_eq!(
            decoded.request.options.thinking_budget,
            Some(ThinkingBudget::Advisory(budget))
        );
        assert_eq!(decoded.request.options.effort, Some(Effort::Max));
        let prepared = crate::server::request::prepare_inference(
            &frontend,
            None,
            &compiler,
            &config,
            &decoded.request,
            0,
            false,
        )
        .unwrap();
        assert!(prepared.prompt.reasoning_cut.is_none());
        assert_eq!(prepared.prompt.max_new_tokens, 32);
        body["thinking_token_budget"] = json!(16);
        assert!(
            decode(Protocol::Messages, body.clone(), false)
                .err()
                .unwrap()
                .to_string()
                .contains("must match")
        );
    }

    #[test]
    fn thinking_budget_preparation_preserves_prompts_and_limits_each_pass() {
        for encoder in [PromptEncoder::DeepseekV4, PromptEncoder::DeepseekV41] {
            let (frontend, compiler, config) = native(encoder);
            let decoded = decode(
                Protocol::Chat,
                json!({"max_tokens":32,"seed":7,
                "messages":[{"role":"user","content":"Hello"}]}),
                false,
            )
            .unwrap();
            let prepare = |request: &crate::server::inference::Request| {
                crate::server::request::prepare_inference(
                    &frontend, None, &compiler, &config, request, 0, false,
                )
            };
            let plain = prepare(&decoded.request).unwrap();
            assert!(plain.prompt.reasoning_cut.is_none());
            for budget in [16, 32, 64] {
                let mut request = decoded.request.clone();
                request.options.thinking_budget = Some(ThinkingBudget::Required(budget));
                let prepared = prepare(&request).unwrap();
                assert_eq!(prepared.rendered_prompt, plain.rendered_prompt);
                assert_eq!(prepared.prompt.tokens, plain.prompt.tokens);
                assert_eq!(
                    prepared.prompt.message_boundaries,
                    plain.prompt.message_boundaries
                );
                assert_eq!(
                    serde_json::to_value(&prepared.prompt.sampling).unwrap(),
                    serde_json::to_value(&plain.prompt.sampling).unwrap()
                );
                assert_eq!(prepared.prompt.max_new_tokens, budget.min(32) as u32);
                assert_eq!(
                    prepared.prompt.reasoning_cut.as_ref().map(|cut| cut.after),
                    (budget < 32).then_some(16)
                );
                if let Some(cut) = prepared.prompt.reasoning_cut {
                    assert_eq!(frontend.encode("</think>", false).unwrap(), vec![cut.close]);
                }
            }
        }
        for (encoder, options) in [
            (PromptEncoder::DeepseekV41, json!({"enable_thinking":false})),
            (PromptEncoder::DeepseekV4, json!({"thinking_mode":"chat"})),
            (PromptEncoder::Jinja, json!({})),
        ] {
            let (frontend, compiler, mut config) = native(encoder);
            if encoder == PromptEncoder::Jinja {
                config.contract.frontend.reasoning = None;
            }
            let mut decoded = decode(
                Protocol::Chat,
                json!({"max_tokens":32,"thinking_token_budget":16,
                "messages":[{"role":"user","content":"Hello"}]}),
                false,
            )
            .unwrap();
            decoded
                .request
                .options
                .template
                .extend(options.as_object().unwrap().clone());
            let prepared = crate::server::request::prepare_inference(
                &frontend,
                None,
                &compiler,
                &config,
                &decoded.request,
                0,
                false,
            )
            .unwrap();
            assert_eq!(
                prepared.output.unwrap().content.start,
                crate::output::tagged::OutputStart::Content
            );
            assert!(prepared.prompt.reasoning_cut.is_none());
            assert_eq!(prepared.prompt.max_new_tokens, 32);
        }
    }

    #[test]
    fn tools_without_parameters_keep_the_empty_object_schema_for_templates_and_deepseek() {
        let template = "{% for m in messages %}{{ m.content }}{% mark %}{% endfor %}{{ tools | tojson }}";
        for encoder in [PromptEncoder::Jinja, PromptEncoder::DeepseekV4, PromptEncoder::DeepseekV41] {
            let (frontend, compiler, mut config) = renderer(encoder, Some(template), TemplateEncoding::Text);
            config.contract.frontend.rejected_tokens.clear();
            let render = |parameters: Option<Value>| {
                let mut function = json!({"name":"now","description":"Current time."});
                if let Some(parameters) = parameters {
                    function["parameters"] = parameters;
                }
                let decoded = decode(Protocol::Chat, json!({"messages":[{"role":"user","content":"Time?"}],
                    "tools":[{"type":"function","function":function}]}), false).unwrap();
                crate::server::request::prepare_inference(
                    &frontend, None, &compiler, &config, &decoded.request, 0, false,
                )
                .unwrap()
                .rendered_prompt
            };
            let declared = render(Some(json!({"type":"object","properties":{}})));
            assert_eq!(render(None), declared, "{encoder:?}");
            assert_eq!(render(Some(Value::Null)), declared, "{encoder:?}");
        }
    }

    #[test]
    fn templates_receive_failed_tool_results_with_failure_text() {
        let (frontend, compiler, config) = renderer(PromptEncoder::Jinja, None, TemplateEncoding::Text);
        let render = |result: Value| {
            let decoded = decode(Protocol::Chat, json!({"messages":[{"role":"user","content":"Go"},
                {"role":"assistant","tool_calls":[{"id":"c","type":"function","function":{"name":"f","arguments":"{}"}}]},
                result]}), false).unwrap();
            crate::server::request::prepare_inference(&frontend, None, &compiler, &config, &decoded.request, 0, false)
                .unwrap()
                .rendered_prompt
        };
        let flagged = render(json!({"role":"tool","tool_call_id":"c","is_error":true,"content":"timeout"}));
        assert!(flagged.contains("<tool>Tool execution failed:\n\ntimeout</tool>"), "{flagged}");
        assert_eq!(render(json!({"role":"tool","tool_call_id":"c",
            "content":[{"type":"tool_error","text":"timeout"}]})), flagged);
        assert!(decode(Protocol::Chat, json!({"messages":[{"role":"tool","tool_call_id":"c",
            "content":[{"type":"text","text":"a"},{"type":"tool_error","text":"b"}]}]}), false).is_err());
    }

    #[test]
    fn minimal_effort_renders_low_where_the_prompt_format_has_no_minimal_level() {
        let template = "{% for m in messages %}{{ m.content }}{% mark %}{% endfor %}effort={{ reasoning_effort }}";
        for encoder in [PromptEncoder::Jinja, PromptEncoder::DeepseekV4, PromptEncoder::DeepseekV41] {
            let (frontend, compiler, mut config) = renderer(encoder, Some(template), TemplateEncoding::Text);
            config.contract.frontend.rejected_tokens.clear();
            let render = |effort: &str| {
                let decoded = decode(Protocol::Responses,
                    json!({"input":"Hello","reasoning":{"effort":effort}}), false).unwrap();
                assert_eq!(decoded.request.options.effort,
                    Some(if effort == "minimal" { Effort::Minimal } else { Effort::Low }));
                crate::server::request::prepare_inference(
                    &frontend, None, &compiler, &config, &decoded.request, 0, false,
                )
                .unwrap()
                .rendered_prompt
            };
            let low = render("low");
            assert_eq!(render("minimal"), low, "{encoder:?}");
            if encoder == PromptEncoder::Jinja {
                assert!(low.ends_with("effort=low"));
            }
        }
    }

    #[test]
    fn thinking_budget_keeps_messages_advisory_on_unsupported_formats() {
        for close in [None, Some("</think>extra")] {
            let (frontend, compiler, mut config) = native(PromptEncoder::DeepseekV41);
            if let Some(close) = close {
                config.contract.frontend.reasoning.as_mut().unwrap().close = close.into();
            } else {
                config.contract.frontend.reasoning = None;
            }
            let mut body = json!({"max_tokens":32,"messages":[{"role":"user","content":"Hello"}],
                "thinking":{"type":"enabled","budget_tokens":16}});
            for explicit in [false, true] {
                if explicit {
                    body["thinking_token_budget"] = json!(16);
                }
                let decoded = decode(Protocol::Messages, body.clone(), false).unwrap();
                assert_eq!(decoded.request.options.effort, Some(Effort::Low));
                let prepared = crate::server::request::prepare_inference(
                    &frontend,
                    None,
                    &compiler,
                    &config,
                    &decoded.request,
                    0,
                    false,
                );
                if explicit && close.is_some() {
                    assert!(
                        prepared
                            .err()
                            .unwrap()
                            .to_string()
                            .contains("thinking_token_budget is not supported")
                    );
                } else {
                    let prepared = prepared.unwrap();
                    assert!(prepared.prompt.reasoning_cut.is_none());
                    assert_eq!(prepared.prompt.max_new_tokens, 32);
                }
            }
        }
    }

    #[test]
    fn anthropic_thinking_budget_maps_to_native_effort_without_continuation() {
        for encoder in [PromptEncoder::DeepseekV4, PromptEncoder::DeepseekV41] {
            let (frontend, compiler, config) = native(encoder);
            let decoded = decode(
                Protocol::Messages,
                json!({"model":config.model_id,"max_tokens":32,
                "thinking":{"type":"enabled","budget_tokens":4000},
                "messages":[{"role":"user","content":"Hello"}]}),
                false,
            )
            .unwrap();
            let prepared = crate::server::request::prepare_inference(
                &frontend,
                None,
                &compiler,
                &config,
                &decoded.request,
                0,
                false,
            )
            .unwrap();
            assert_eq!(
                decoded.request.options.thinking_budget,
                Some(ThinkingBudget::Advisory(4000))
            );
            assert!(prepared.rendered_prompt.ends_with("<｜Assistant｜><think>"));
            assert_eq!(
                prepared.output.unwrap().content.start,
                crate::output::tagged::OutputStart::Reasoning
            );
        }
    }
}
