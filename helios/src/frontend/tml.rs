/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! TMLv0 prompt encoding: typed message frames delimited by control tokens.
//!
//! Each frame is a role header, an optional author or tool name, one content-kind
//! token, an ordinary-text body and `<|end_message|>`. Bodies are encoded without
//! the added vocabulary, so control-token spellings in conversation text stay
//! ordinary text. The reference is Thinking Machines' tml-renderers 0.4.1
//! `renderers.Inkling2` over OpenAI Chat messages, recorded in
//! tests/frontend/inkling2_tml.json.

use super::{Frontend, Json, Map, RenderedPrompt, deepseek::invalid};
use crate::{Error, Failure, FrontendError, Result};

/// Named efforts accepted by the checkpoint chat template; TMLv0 encodes the number.
const EFFORTS: [(&str, f64); 7] = [
    ("none", 0.0),
    ("minimal", 0.1),
    ("low", 0.2),
    ("medium", 0.7),
    ("high", 0.9),
    ("xhigh", 0.99),
    ("max", 0.99),
];
const DEFAULT_EFFORT: f64 = 0.9;
/// tml-renderers accepts efforts up to 1, but SGLang's Inkling serving rejects
/// efforts above 0.99 (`parse_reasoning_effort` in
/// python/sglang/srt/entrypoints/openai/serving_inkling.py at 237ba6b), as the
/// checkpoint chat template does; 0.99 is also its `xhigh` and `max` level.
const MAX_EFFORT: f64 = 0.99;
const EFFORT_ERROR: &str =
    "reasoning_effort must be none, minimal, low, medium, high, xhigh, max, or a number from 0 through 0.99";

pub(super) fn effort_level(name: &str) -> bool {
    EFFORTS.iter().any(|(level, _)| *level == name)
}

/// The effort as TMLv0 displays it: thousandths, then hundredths, both rounded
/// half to even in binary floating point, without trailing zeros.
pub(crate) fn effort_text(effort: f64) -> String {
    let thousandths = (effort * 1000.0).round_ties_even();
    match (thousandths / 1000.0 * 100.0).round_ties_even() as u32 {
        0 => "0".into(),
        100 => "1".into(),
        hundredths if hundredths % 10 == 0 => format!("0.{}", hundredths / 10),
        hundredths => format!("0.{hundredths:02}"),
    }
}

fn effort(options: &Map<String, Json>) -> Result<f64> {
    match options.get("enable_thinking").filter(|v| !v.is_null()) {
        Some(Json::Bool(false)) => return Ok(0.0),
        Some(Json::Bool(true)) | None => {}
        Some(_) => return Err(invalid("enable_thinking must be a boolean")),
    }
    let value = match options.get("reasoning_effort").filter(|v| !v.is_null()) {
        None => DEFAULT_EFFORT,
        Some(Json::String(name)) => EFFORTS
            .iter()
            .find(|(level, _)| *level == name.trim())
            .map(|(_, value)| *value)
            .ok_or_else(|| invalid(EFFORT_ERROR))?,
        Some(Json::Number(number)) => number.as_f64().ok_or_else(|| invalid(EFFORT_ERROR))?,
        Some(_) => return Err(invalid(EFFORT_ERROR)),
    };
    if !(0.0..=MAX_EFFORT).contains(&value) {
        return Err(invalid(EFFORT_ERROR));
    }
    Ok(value)
}

/// Compact JSON as tml-renderers writes it. Tool declarations sort object keys
/// recursively and spell floats with an explicit sign on positive exponents
/// (`1e+20`); call arguments keep request key order and spell floats as `1e20`.
/// Both use the shortest round-trip digits.
fn write_json(value: &Json, sort_keys: bool, exponent_sign: bool, output: &mut String) {
    match value {
        Json::Number(number) if number.is_f64() => {
            let mut buffer = ryu::Buffer::new();
            let text = buffer.format_finite(number.as_f64().unwrap());
            match text.split_once('e') {
                Some((mantissa, exponent)) if exponent_sign && !exponent.starts_with('-') => {
                    output.push_str(mantissa);
                    output.push_str("e+");
                    output.push_str(exponent);
                }
                _ => output.push_str(text),
            }
        }
        Json::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                write_json(value, sort_keys, exponent_sign, output);
            }
            output.push(']');
        }
        Json::Object(values) => {
            let mut entries: Vec<_> = values.iter().collect();
            if sort_keys {
                entries.sort_by(|left, right| left.0.cmp(right.0));
            }
            output.push('{');
            for (index, (key, value)) in entries.into_iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                output.push_str(&serde_json::to_string(key).expect("strings serialize"));
                output.push(':');
                write_json(value, sort_keys, exponent_sign, output);
            }
            output.push('}');
        }
        other => output.push_str(&other.to_string()),
    }
}

fn declarations(tools: &[Json]) -> Result<String> {
    let mut specs = Vec::with_capacity(tools.len());
    for tool in tools {
        let function = tool
            .get("function")
            .filter(|v| v.is_object())
            .ok_or_else(|| invalid("TMLv0 tools must declare a function object"))?;
        let name = function["name"]
            .as_str()
            .ok_or_else(|| invalid("tool requires a name"))?;
        let description = match function.get("description").filter(|v| !v.is_null()) {
            None => "",
            Some(Json::String(text)) => text,
            Some(_) => return Err(invalid("tool description must be text")),
        };
        let parameters = match function.get("parameters").filter(|v| !v.is_null()) {
            None => Json::Object(Map::new()),
            Some(value @ Json::Object(_)) => value.clone(),
            Some(_) => return Err(invalid("tool parameters must be a JSON object")),
        };
        specs.push(serde_json::json!({
            "description": description, "name": name, "parameters": parameters, "type": "function",
        }));
    }
    let mut output = String::new();
    write_json(&Json::Array(specs), true, true, &mut output);
    Ok(output)
}

/// Call arguments with sorted top-level keys and nested values in request order.
fn arguments(name: &str, value: &Json) -> Result<String> {
    let parsed;
    let object = match value {
        Json::String(text) => {
            parsed = serde_json::from_str::<Json>(text)
                .map_err(|_| invalid("tool arguments must contain a JSON object"))?;
            &parsed
        }
        other => other,
    }
    .as_object()
    .ok_or_else(|| invalid("tool arguments must contain a JSON object"))?;
    let mut entries: Vec<_> = object.iter().collect();
    entries.sort_by(|left, right| left.0.cmp(right.0));
    let mut output = format!("{{\"name\":{},\"args\":{{", serde_json::to_string(name).unwrap());
    for (index, (key, value)) in entries.into_iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        output.push_str(&serde_json::to_string(key).unwrap());
        output.push(':');
        write_json(value, false, false, &mut output);
    }
    output.push_str("}}");
    Ok(output)
}

struct Tokens {
    user: u32,
    model: u32,
    system: u32,
    tool: u32,
    text: u32,
    thinking: u32,
    xml: u32,
    image: u32,
    invoke: u32,
    end: u32,
    end_sampling: u32,
}

impl Tokens {
    fn new(frontend: &Frontend) -> Result<Self> {
        let vocabulary = frontend.tokenizer.get_added_vocabulary().get_vocab();
        let id = |name: &str| {
            vocabulary
                .get(name)
                .copied()
                .ok_or_else(|| invalid(format!("TMLv0 tokenizer lacks the {name} control token")))
        };
        Ok(Self {
            user: id("<|message_user|>")?,
            model: id("<|message_model|>")?,
            system: id("<|message_system|>")?,
            tool: id("<|message_tool|>")?,
            text: id("<|content_text|>")?,
            thinking: id("<|content_thinking|>")?,
            xml: id("<|content_xml|>")?,
            image: id("<|content_image|>")?,
            invoke: id("<|content_invoke_tool_json|>")?,
            end: id("<|end_message|>")?,
            end_sampling: id("<|content_model_end_sampling|>")?,
        })
    }
}

struct Prompt<'a> {
    frontend: &'a Frontend,
    tokens: Tokens,
    token_ids: Vec<u32>,
    text: String,
    boundaries: Vec<usize>,
    image_sources: Vec<String>,
}

impl Prompt<'_> {
    fn control(&mut self, id: u32) -> Result<()> {
        self.token_ids.push(id);
        self.text.push_str(&self.frontend.decode(&[id], false)?);
        Ok(())
    }

    fn ordinary(&mut self, text: &str) -> Result<()> {
        let tokenizer = self
            .frontend
            .ordinary_tokenizer
            .as_ref()
            .expect("TMLv0 ordinary tokenizer is initialized");
        let encoding = tokenizer.encode(text, false).map_err(|error| {
            Error::frontend(FrontendError::Tokenizer(Failure::boxed(error.to_string(), error)))
        })?;
        self.token_ids.extend_from_slice(encoding.get_ids());
        self.text.push_str(text);
        Ok(())
    }

    fn frame(&mut self, role: u32, name: &str, kind: u32, body: &str) -> Result<()> {
        self.control(role)?;
        self.ordinary(name)?;
        self.control(kind)?;
        self.ordinary(body)?;
        self.control(self.tokens.end)
    }

    fn boundary(&mut self) {
        self.boundaries.push(self.token_ids.len());
    }

    /// Frames for message content: text parts and image placeholders, one frame each.
    fn content(&mut self, role: u32, name: &str, content: &Json, hints: &[Json]) -> Result<()> {
        let parts = match content {
            Json::Null => return Ok(()),
            Json::String(text) => {
                self.frame(role, name, self.tokens.text, text)?;
                if hints.contains(&Json::from(1)) {
                    self.boundary();
                }
                return Ok(());
            }
            Json::Array(parts) => parts,
            _ => return Err(invalid("message content must be text or content blocks")),
        };
        for (index, part) in parts.iter().enumerate() {
            match part["type"].as_str() {
                Some("text") => {
                    let text = part["text"]
                        .as_str()
                        .ok_or_else(|| invalid("text content must be a string"))?;
                    self.frame(role, name, self.tokens.text, text)?;
                }
                Some("image_url") => {
                    let url = part["image_url"]["url"]
                        .as_str()
                        .ok_or_else(|| invalid("image_url.url must be a string"))?;
                    let placeholder = self.frontend.image_placeholder_token_id.ok_or_else(|| {
                        invalid("image inputs require a prepared image placeholder token ID")
                    })?;
                    self.control(role)?;
                    self.ordinary(name)?;
                    self.control(self.tokens.image)?;
                    self.control(placeholder)?;
                    self.control(self.tokens.end)?;
                    self.image_sources.push(url.to_owned());
                }
                _ => return Err(invalid("unsupported TMLv0 content block")),
            }
            if hints.contains(&Json::from(index + 1)) {
                self.boundary();
            }
        }
        Ok(())
    }
}

fn text_field<'a>(message: &'a Json, field: &str) -> Result<Option<&'a str>> {
    match message.get(field).filter(|v| !v.is_null()) {
        None => Ok(None),
        Some(Json::String(text)) => Ok(Some(text)),
        Some(_) => Err(invalid(format!("{field} must be text"))),
    }
}

fn continuation(options: &Map<String, Json>) -> Result<Option<String>> {
    match options.get("continue_final_message") {
        None | Some(Json::Null | Json::Bool(false)) => Ok(None),
        Some(Json::Bool(true)) => Ok(Some("content".into())),
        Some(Json::String(field)) if field.is_empty() => Ok(None),
        Some(Json::String(field)) if field == "content" || field == "reasoning_content" => {
            Ok(Some(field.clone()))
        }
        Some(_) => Err(invalid(
            "continue_final_message must be a boolean, content, or reasoning_content",
        )),
    }
}

const OPTIONS: [&str; 7] = [
    "reasoning_effort", "enable_thinking", "add_generation_prompt", "continue_final_message", "tools",
    "tool_choice", "documents",
];
const TOKEN_DEFAULTS: [&str; 8] = [
    "bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token",
    "additional_special_tokens",
];

fn merged_options(frontend: &Frontend, options: &Map<String, Json>) -> Result<Map<String, Json>> {
    if let Some(key) = options.keys().find(|key| !OPTIONS.contains(&key.as_str())) {
        return Err(invalid(format!("unsupported TMLv0 prompt option: {key}")));
    }
    let mut merged = frontend.defaults.clone();
    if let Some(key) = merged
        .keys()
        .find(|key| !OPTIONS.contains(&key.as_str()) && !TOKEN_DEFAULTS.contains(&key.as_str()))
    {
        return Err(invalid(format!("unsupported TMLv0 prompt option: {key}")));
    }
    merged.extend(options.clone());
    if merged.get("documents").is_some_and(|v| !v.is_null()) {
        return Err(invalid("TMLv0 prompt encoding does not support documents"));
    }
    match merged.get("tool_choice").filter(|v| !v.is_null()) {
        None => {}
        Some(Json::String(choice)) if matches!(choice.as_str(), "auto" | "none" | "required") => {}
        Some(_) => return Err(invalid("tool_choice must be auto, none, or required")),
    }
    if !matches!(merged.get("add_generation_prompt"), None | Some(Json::Null | Json::Bool(_))) {
        return Err(invalid("add_generation_prompt must be a boolean"));
    }
    Ok(merged)
}

pub(super) fn validate_defaults(frontend: &Frontend) -> Result<()> {
    Tokens::new(frontend)?;
    render(frontend, &[serde_json::json!({"role":"user","content":""})], &Map::new()).map(|_| ())
}

pub(super) fn render(
    frontend: &Frontend,
    messages: &[Json],
    options: &Map<String, Json>,
) -> Result<RenderedPrompt> {
    if messages.is_empty() {
        return Err(invalid("cannot encode an empty conversation"));
    }
    let options = merged_options(frontend, options)?;
    let effort = effort(&options)?;
    let continued = continuation(&options)?;
    if continued.is_some() && options.get("add_generation_prompt") == Some(&Json::Bool(true)) {
        return Err(invalid(
            "continue_final_message and add_generation_prompt are not compatible",
        ));
    }
    if continued.is_some() && messages.last().is_none_or(|m| m["role"] != "assistant") {
        return Err(invalid("continue_final_message requires a final assistant message"));
    }
    let tools = match options.get("tools").filter(|v| !v.is_null()) {
        None => &[][..],
        Some(Json::Array(tools)) => tools.as_slice(),
        Some(_) => return Err(invalid("tools must be an array")),
    };
    let mut prompt = Prompt {
        frontend,
        tokens: Tokens::new(frontend)?,
        token_ids: Vec::new(),
        text: String::new(),
        boundaries: Vec::new(),
        image_sources: Vec::new(),
    };
    let declaration = (!tools.is_empty()).then(|| declarations(tools)).transpose()?;
    let first_turn = messages
        .iter()
        .position(|m| m["role"] != "system")
        .unwrap_or(messages.len());
    let thinking = format!("Thinking effort level: {}", effort_text(effort));
    // The leading system messages come first so that a shared system prompt stays a
    // reusable prefix when the tools differ; SGLang and tinker-cookbook use this order.
    let preamble = |prompt: &mut Prompt| -> Result<()> {
        if let Some(body) = &declaration {
            prompt.frame(prompt.tokens.system, "tool_declare", prompt.tokens.xml, body)?;
            prompt.boundary();
        }
        prompt.frame(prompt.tokens.system, "", prompt.tokens.text, &thinking)?;
        prompt.boundary();
        Ok(())
    };
    // A tool result without a name takes the most recent preceding call with its ID.
    let mut calls: std::collections::HashMap<&str, &str> = Default::default();
    for (index, message) in messages.iter().enumerate() {
        if index == first_turn {
            preamble(&mut prompt)?;
        }
        let role = message["role"]
            .as_str()
            .ok_or_else(|| invalid("message requires a role"))?;
        let hints = message["_checkpoint_hints"]
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let content = message.get("content").unwrap_or(&Json::Null);
        match role {
            "system" | "user" => {
                let header = if role == "system" { prompt.tokens.system } else { prompt.tokens.user };
                prompt.content(header, "", content, hints)?;
            }
            "tool" => {
                // A result's own name wins over the name of its call, as in tml-renderers.
                let name = match text_field(message, "name")?.filter(|name| !name.is_empty()) {
                    Some(name) => name,
                    None => text_field(message, "tool_call_id")?
                        .and_then(|id| calls.get(id).copied())
                        .unwrap_or(""),
                };
                prompt.content(prompt.tokens.tool, name, content, hints)?;
            }
            "assistant" => {
                let last = index + 1 == messages.len();
                let open = continued.as_deref().filter(|_| last);
                let reasoning = text_field(message, "reasoning_content")?.filter(|text| !text.is_empty());
                let tool_calls = match message.get("tool_calls").filter(|v| !v.is_null()) {
                    None => &[][..],
                    Some(Json::Array(calls)) => calls.as_slice(),
                    Some(_) => return Err(invalid("tool_calls must be an array")),
                };
                let has_content = !matches!(content, Json::Null)
                    && content != &Json::String(String::new())
                    && content.as_array().is_none_or(|parts| !parts.is_empty());
                match open {
                    Some("reasoning_content") if has_content || !tool_calls.is_empty() => {
                        return Err(invalid(
                            "continue_final_message=reasoning_content requires a final assistant message without content or tool calls",
                        ));
                    }
                    Some("content") if !tool_calls.is_empty() => {
                        return Err(invalid(
                            "continue_final_message requires a final assistant message without tool calls",
                        ));
                    }
                    _ => {}
                }
                if let Some(text) = reasoning.or(open.filter(|f| *f == "reasoning_content").map(|_| "")) {
                    prompt.frame(prompt.tokens.model, "", prompt.tokens.thinking, text)?;
                }
                let text_frame = match open {
                    Some("content") => true,
                    Some(_) => false,
                    None => has_content || (reasoning.is_none() && tool_calls.is_empty()),
                };
                if text_frame {
                    let content = if has_content { content } else { &Json::String(String::new()) };
                    if open == Some("content")
                        && content
                            .as_array()
                            .is_some_and(|parts| parts.last().is_none_or(|part| part["type"] != "text"))
                    {
                        return Err(invalid("continue_final_message requires the final assistant content to end in text"));
                    }
                    prompt.content(prompt.tokens.model, "", content, hints)?;
                }
                for call in tool_calls {
                    let function = &call["function"];
                    let name = function["name"]
                        .as_str()
                        .ok_or_else(|| invalid("tool call requires a name"))?;
                    let body = arguments(name, &function["arguments"])?;
                    prompt.frame(prompt.tokens.model, name, prompt.tokens.invoke, &body)?;
                    if let Some(id) = call["id"].as_str() {
                        calls.insert(id, name);
                    }
                }
                if open.is_some() {
                    // The final frame stays open for the continuation.
                    let end = frontend.decode(&[prompt.tokens.end], false)?;
                    prompt.token_ids.pop();
                    prompt.text.truncate(prompt.text.len() - end.len());
                    break;
                }
                prompt.control(prompt.tokens.end_sampling)?;
            }
            _ => return Err(invalid("unsupported TMLv0 message role")),
        }
        prompt.boundary();
    }
    if first_turn == messages.len() {
        preamble(&mut prompt)?;
    }
    let length = prompt.token_ids.len();
    let mut boundaries = std::mem::take(&mut prompt.boundaries);
    boundaries.retain(|&boundary| 0 < boundary && boundary < length);
    boundaries.sort_unstable();
    boundaries.dedup();
    Ok(RenderedPrompt {
        text: prompt.text,
        token_ids: prompt.token_ids,
        image_sources: prompt.image_sources,
        image_checkpoint_hints: Vec::new(),
        message_boundaries: boundaries,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effort_display_rounds_thousandths_then_hundredths_half_to_even() {
        for (value, text) in [
            (0.0, "0"), (-0.0, "0"), (0.005, "0"), (0.0145, "0.01"), (0.015, "0.02"), (0.1, "0.1"),
            (0.123, "0.12"), (0.125, "0.12"), (0.545, "0.55"), (0.575, "0.57"), (0.9, "0.9"),
            (0.99, "0.99"), (0.995, "1"),
        ] {
            assert_eq!(effort_text(value), text, "{value}");
        }
    }

    #[test]
    fn effort_display_matches_the_reference_table() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/frontend/inkling2_tml.json");
        let golden: Json = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let efforts = golden["efforts"].as_array().unwrap();
        assert!(efforts.len() > 2000);
        for entry in efforts {
            assert_eq!(effort_text(entry[0].as_f64().unwrap()), entry[1].as_str().unwrap(), "{entry}");
        }
    }

    #[test]
    fn declarations_sort_keys_recursively_and_sign_positive_exponents() {
        let tools = serde_json::json!([{"type":"function","function":{"name":"g","description":null,
            "parameters":{"z":{"b":1e20,"a":1e-7},"y":[0.00001,1e16,2.0,-0.0,3]}}}]);
        assert_eq!(
            declarations(tools.as_array().unwrap()).unwrap(),
            r#"[{"description":"","name":"g","parameters":{"y":[0.00001,1e+16,2.0,-0.0,3],"z":{"a":1e-7,"b":1e+20}},"type":"function"}]"#
        );
    }

    #[test]
    fn call_arguments_sort_only_top_level_keys() {
        assert_eq!(
            arguments("f", &serde_json::json!(r#"{"b":{"z":1,"y":1e20},"a":[1.50]}"#)).unwrap(),
            r#"{"name":"f","args":{"a":[1.5],"b":{"z":1,"y":1e20}}}"#
        );
        assert!(arguments("f", &serde_json::json!("[]")).is_err());
    }
}
