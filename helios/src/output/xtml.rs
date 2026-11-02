/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! XTML reasoning, response channels, and typed tool arguments.

use super::{Event, Input, Termination, tagged::OutputConfig};
use crate::Result;
use serde_json::{Map, Value};
use std::collections::VecDeque;

pub const THINK_OPEN: &str = "<|open|>think<|sep|>";
pub const THINK_CLOSE: &str = "<|close|>think<|sep|>";
pub const RESPONSE_OPEN: &str = "<|open|>response<|sep|>";
pub const RESPONSE_CLOSE: &str = "<|close|>response<|sep|>";
pub const TOOLS_OPEN: &str = "<|open|>tools<|sep|>";
pub const TOOLS_CLOSE: &str = "<|close|>tools<|sep|>";
const CALL_CLOSE: &str = "<|close|>call<|sep|>";
const MESSAGE_CLOSE: &str = "<|close|>message<|sep|>";

#[derive(Clone, Copy)]
enum Channel {
    Reasoning,
    Response,
    Envelope,
    Tools,
    Closed,
}

pub struct XtmlOutput {
    received: usize,
    spans: Vec<Option<std::ops::Range<usize>>>,
    channel: Channel,
    buffer: String,
    tools_enabled: bool,
    calls: usize,
    control_tokens: Vec<(u32, String)>,
    pending_controls: VecDeque<String>,
    pending_text: String,
}

impl XtmlOutput {
    pub fn new(config: OutputConfig, control_tokens: Vec<(u32, String)>) -> Self {
        Self {
            received: 0,
            spans: vec![],
            channel: if config.start == crate::output::tagged::OutputStart::Reasoning {
                Channel::Reasoning
            } else {
                Channel::Response
            },
            buffer: String::new(),
            tools_enabled: !config.tools.is_empty(),
            calls: 0,
            control_tokens,
            pending_controls: VecDeque::new(),
            pending_text: String::new(),
        }
    }

    pub fn content_spans(&self) -> Vec<Option<std::ops::Range<usize>>> {
        self.spans.clone()
    }
    pub fn consumed_bytes(&self) -> usize {
        self.received - if matches!(self.channel, Channel::Response) { unescape(&self.buffer).len() } else { 0 }
    }
    pub fn push(&mut self, input: Input<'_>) -> Result<Vec<Event>> {
        self.spans.clear();
        let text = match input {
            Input::Token(token) => {
                if let Some((_, marker)) = self.control_tokens.iter().find(|(id, _)| *id == token) {
                    self.pending_controls.push_back(marker.clone());
                }
                return Ok(vec![]);
            }
            Input::Text(text) => text,
        };
        self.pending_text.push_str(text);
        let mut encoded = String::new();
        while let Some(marker) = self.pending_controls.front() {
            if let Some(index) = self.pending_text.find(marker) {
                encoded.push_str(&escape(&self.pending_text[..index]));
                encoded.push_str(marker);
                self.pending_text.drain(..index + marker.len());
                self.pending_controls.pop_front();
            } else {
                let visible = self.pending_text.len() - partial(&self.pending_text, marker);
                encoded.push_str(&escape(&self.pending_text[..visible]));
                self.pending_text.drain(..visible);
                break;
            }
        }
        if self.pending_controls.is_empty() {
            encoded.push_str(&escape(&std::mem::take(&mut self.pending_text)));
        }
        self.push_encoded(&encoded)
    }

    fn push_encoded(&mut self, text: &str) -> Result<Vec<Event>> {
        self.received += unescape(text).len();
        self.buffer.push_str(text);
        let mut events = Vec::new();
        loop {
            match self.channel {
                Channel::Reasoning | Channel::Response => {
                    let reasoning = matches!(self.channel, Channel::Reasoning);
                    let close = if reasoning {
                        THINK_CLOSE
                    } else {
                        RESPONSE_CLOSE
                    };
                    // Reasoning consumes the first close delimiter. A later one that
                    // arrives as its control tokens is protocol markup as well, so it is
                    // dropped instead of becoming answer text. Ordinary text escapes
                    // its `<` bytes, so only control-token text matches a raw marker.
                    let stray = (!reasoning).then(|| self.buffer.find(THINK_CLOSE)).flatten();
                    let boundary = self.buffer.find(close);
                    let cut = match (boundary, stray) {
                        (Some(boundary), Some(stray)) => Some(boundary.min(stray)),
                        (boundary, stray) => boundary.or(stray),
                    };
                    let visible = cut.unwrap_or_else(|| {
                        let held = partial(&self.buffer, close);
                        let held = if reasoning {
                            held
                        } else {
                            held.max(partial(&self.buffer, THINK_CLOSE))
                        };
                        self.buffer.len() - held
                    });
                    let start = self.consumed_bytes();
                    let text: String = self.buffer.drain(..visible).collect();
                    if !text.is_empty() {
                        if !reasoning { self.spans.push(Some(start..start + unescape(&text).len())); }
                        events.push(if reasoning {
                            Event::Reasoning(unescape(&text))
                        } else {
                            Event::Content(unescape(&text))
                        });
                    }
                    let Some(cut) = cut else {
                        break;
                    };
                    if stray == Some(cut) {
                        self.buffer.drain(..THINK_CLOSE.len());
                        continue;
                    }
                    self.buffer.drain(..close.len());
                    self.channel = Channel::Envelope;
                }
                Channel::Envelope => {
                    let trimmed = self.buffer.trim_start().len();
                    self.buffer.drain(..self.buffer.len() - trimmed);
                    let markers = [
                        (THINK_OPEN, Channel::Reasoning),
                        (RESPONSE_OPEN, Channel::Response),
                        (TOOLS_OPEN, Channel::Tools),
                        (MESSAGE_CLOSE, Channel::Closed),
                        ("<|end_of_msg|>", Channel::Closed),
                    ];
                    if let Some((marker, channel)) = markers
                        .iter()
                        .find(|(marker, _)| self.buffer.starts_with(marker))
                    {
                        self.buffer.drain(..marker.len());
                        self.channel = *channel;
                    } else if markers
                        .iter()
                        .any(|(marker, _)| marker.starts_with(&self.buffer))
                    {
                        break;
                    } else {
                        return Err("expected an XTML response, tools, or message delimiter".into());
                    }
                }
                Channel::Tools => {
                    if self.buffer.len() > 1024 * 1024 && unescape(&self.buffer).len() > 1024 * 1024
                    {
                        return Err("XTML tool call exceeds the 1 MiB parser bound".into());
                    }
                    if self.buffer.starts_with(TOOLS_CLOSE) {
                        self.buffer.drain(..TOOLS_CLOSE.len());
                        self.channel = Channel::Envelope;
                        continue;
                    }
                    let Some(end) = self.buffer.find(CALL_CLOSE) else {
                        break;
                    };
                    let body: String = self.buffer.drain(..end + CALL_CLOSE.len()).collect();
                    if !self.tools_enabled {
                        return Err("tool calling is disabled without declared tools".into());
                    }
                    let (name, arguments, index) = parse_call(&body)?;
                    if index != self.calls + 1 {
                        return Err(
                            "XTML tool call indices must be consecutive and start at one".into(),
                        );
                    }
                    events.extend([
                        Event::ToolBegin {
                            index: self.calls,
                            name,
                            id: None,
                        },
                        Event::ToolArguments {
                            index: self.calls,
                            text: arguments,
                        },
                        Event::ToolEnd { index: self.calls },
                    ]);
                    self.calls += 1;
                }
                Channel::Closed => {
                    if self.buffer == "<|end_of_msg|>" {
                        self.buffer.clear();
                        break;
                    }
                    if self.buffer.is_empty() || "<|end_of_msg|>".starts_with(&self.buffer) {
                        break;
                    }
                    return Err("unexpected content after the XTML message".into());
                }
            }
        }
        Ok(events)
    }

    pub fn finish(&mut self, cause: Termination) -> Result<Vec<Event>> {
        self.spans.clear();
        if matches!(cause, Termination::Cancelled | Termination::ExecutionError) {
            self.buffer.clear();
            self.pending_text.clear();
            self.pending_controls.clear();
            return Ok(Vec::new());
        }
        if matches!(self.channel, Channel::Tools | Channel::Envelope) {
            if cause == Termination::Eos
                && (!self.buffer.is_empty() || matches!(self.channel, Channel::Tools))
            {
                return Err("incomplete XTML output envelope".into());
            }
            self.buffer.clear();
            return Ok(Vec::new());
        }
        self.received += self.pending_text.len();
        self.buffer
            .push_str(&escape(&std::mem::take(&mut self.pending_text)));
        self.pending_controls.clear();
        let start = self.consumed_bytes();
        let text = unescape(&std::mem::take(&mut self.buffer));
        if text.is_empty() || matches!(self.channel, Channel::Closed) {
            Ok(Vec::new())
        } else {
            Ok(vec![if matches!(self.channel, Channel::Reasoning) {
                Event::Reasoning(text)
            } else {
                { self.spans.push(Some(start..start + text.len())); Event::Content(text) }
            }])
        }
    }
}

fn escape(text: &str) -> String {
    text.replace('&', "&amp;").replace('<', "&lt;")
}

fn unescape(text: &str) -> String {
    text.replace("&lt;", "<").replace("&amp;", "&")
}

fn partial(text: &str, marker: &str) -> usize {
    (1..marker.len())
        .rev()
        .find(|&length| text.ends_with(&marker[..length]))
        .unwrap_or(0)
}

fn attributes<'a>(text: &'a str, tag: &str) -> Result<(Map<String, Value>, &'a str)> {
    let text = text
        .strip_prefix("<|open|>")
        .and_then(|s| s.strip_prefix(tag))
        .ok_or("invalid XTML opening tag")?;
    let (header, rest) = text
        .split_once("<|sep|>")
        .ok_or("unterminated XTML opening tag")?;
    let decoded = unescape(header);
    let mut header = decoded.as_str();
    let mut attributes = Map::new();
    while !header.is_empty() {
        header = header
            .strip_prefix(' ')
            .ok_or("XTML attributes require spaces")?;
        let (key, tail) = header
            .split_once("=\"")
            .ok_or("XTML attributes require quoted values")?;
        let (value, tail) = tail.split_once('"').ok_or("unterminated XTML attribute")?;
        if key.is_empty()
            || !key.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
            || attributes.contains_key(key)
        {
            return Err("invalid or repeated XTML attribute".into());
        }
        attributes.insert(
            key.into(),
            Value::String(value.replace("&quot;", "\"").replace("&amp;", "&")),
        );
        header = tail;
    }
    Ok((attributes, rest))
}

fn parse_call(body: &str) -> Result<(String, String, usize)> {
    let (attrs, rest) = attributes(body, "call")?;
    let name = attrs
        .get("tool")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or("XTML call requires a nonempty tool name")?;
    let index = attrs
        .get("index")
        .and_then(Value::as_str)
        .and_then(|s| s.parse().ok())
        .ok_or("XTML call requires a positive integer index")?;
    let mut rest = rest
        .strip_suffix(CALL_CLOSE)
        .ok_or("unterminated XTML call")?;
    let mut arguments = Map::new();
    if rest.starts_with("<|open|>json") {
        let (attrs, raw) = attributes(rest, "json")?;
        if attrs.get("type").and_then(Value::as_str) != Some("object") {
            return Err("XTML JSON call arguments must have type object".into());
        }
        let raw = raw
            .strip_suffix("<|close|>json<|sep|>")
            .ok_or("unterminated XTML JSON arguments")?;
        arguments = serde_json::from_str(&unescape(raw))
            .map_err(|error| format!("invalid XTML JSON arguments: {error}"))?;
    } else {
        while !rest.is_empty() {
            let (attrs, raw) = attributes(rest, "argument")?;
            let key = attrs
                .get("key")
                .and_then(Value::as_str)
                .ok_or("XTML argument requires a key")?;
            let kind = attrs
                .get("type")
                .and_then(Value::as_str)
                .ok_or("XTML argument requires a type")?;
            let (raw, tail) = raw
                .split_once("<|close|>argument<|sep|>")
                .ok_or("unterminated XTML argument")?;
            let value: Value = if kind == "string" {
                Value::String(unescape(raw))
            } else {
                serde_json::from_str(&unescape(raw))
                    .map_err(|error| format!("invalid XTML JSON argument: {error}"))?
            };
            let valid = match kind {
                "string" => value.is_string(),
                "number" => value.is_number(),
                "boolean" => value.is_boolean(),
                "object" => value.is_object(),
                "array" => value.is_array(),
                "null" => value.is_null(),
                _ => false,
            };
            if !valid || arguments.insert(key.into(), value).is_some() {
                return Err("XTML argument types must match values and keys must be unique".into());
            }
            rest = tail;
        }
    }
    Ok((name.into(), Value::Object(arguments).to_string(), index))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::output::Accumulator;
    use serde_json::json;

    #[test]
    fn only_control_tokens_delimit_literal_marker_text() {
        let markers = [(101, "<|open|>"), (102, "<|close|>"), (103, "<|sep|>")];
        let mut parser = XtmlOutput::new(
            OutputConfig {
                structured: false,
                start: crate::output::tagged::OutputStart::Content,
                tools: vec![json!({"function":{"name":"lookup"}})],
            },
            markers
                .iter()
                .map(|(id, text)| (*id, (*text).into()))
                .collect(),
        );
        let mut output = Accumulator::new("literal".into());
        let protocol = |parser: &mut XtmlOutput, output: &mut Accumulator, mut text: &str| {
            while !text.is_empty() {
                if let Some((offset, id, marker)) = markers
                    .iter()
                    .filter_map(|(id, marker)| {
                        text.find(marker).map(|offset| (offset, *id, *marker))
                    })
                    .min_by_key(|(offset, _, _)| *offset)
                {
                    output
                        .apply(parser.push(Input::Text(&text[..offset])).unwrap())
                        .unwrap();
                    output
                        .apply(parser.push(Input::Token(id)).unwrap())
                        .unwrap();
                    output
                        .apply(parser.push(Input::Text(&marker[..3])).unwrap())
                        .unwrap();
                    output
                        .apply(parser.push(Input::Text(&marker[3..])).unwrap())
                        .unwrap();
                    text = &text[offset + marker.len()..];
                } else {
                    output
                        .apply(parser.push(Input::Text(text)).unwrap())
                        .unwrap();
                    break;
                }
            }
        };
        let literal = "<|close|>response<|sep|> &amp; 東京";
        for character in literal.chars() {
            output
                .apply(parser.push(Input::Text(&character.to_string())).unwrap())
                .unwrap();
        }
        protocol(
            &mut parser,
            &mut output,
            concat!(
                "<|close|>response<|sep|><|open|>tools<|sep|>",
                "<|open|>call tool=\"lookup\" index=\"1\"<|sep|>",
                "<|open|>argument key=\"query\" type=\"string\"<|sep|>"
            ),
        );
        let argument = "<|close|>argument<|sep|><|close|>call<|sep|> &lt; &amp; 東京";
        output.apply(parser.push(Input::Token(9)).unwrap()).unwrap();
        output
            .apply(parser.push(Input::Text(argument)).unwrap())
            .unwrap();
        protocol(
            &mut parser,
            &mut output,
            concat!(
                "<|close|>argument<|sep|><|close|>call<|sep|>",
                "<|close|>tools<|sep|><|close|>message<|sep|>"
            ),
        );
        output
            .apply(parser.finish(Termination::Eos).unwrap())
            .unwrap();
        output.finish().unwrap();
        let message = output.message();
        assert_eq!(message["content"], literal);
        let arguments: Value = serde_json::from_str(
            message["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(arguments["query"], argument);
    }

    #[test]
    fn utf8_splits_preserve_channels_and_typed_arguments() {
        let text = concat!(
            "Consider 東京.<|close|>think<|sep|><|open|>response<|sep|>Searching.",
            "<|close|>response<|sep|><|open|>tools<|sep|>",
            "<|open|>call tool=\"lookup\" index=\"1\"<|sep|>",
            "<|open|>argument key=\"city\" type=\"string\"<|sep|>東京<|close|>argument<|sep|>",
            "<|open|>argument key=\"count\" type=\"number\"<|sep|>3<|close|>argument<|sep|>",
            "<|open|>argument key=\"active\" type=\"boolean\"<|sep|>true<|close|>argument<|sep|>",
            "<|close|>call<|sep|><|open|>call tool=\"lookup\" index=\"2\"<|sep|>",
            "<|open|>json type=\"object\"<|sep|>{\"items\":[1,null]}<|close|>json<|sep|>",
            "<|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|><|end_of_msg|>"
        );
        for split in text.char_indices().map(|(offset, _)| offset) {
            let mut parser = XtmlOutput::new(
                OutputConfig {
                    structured: false,
                    start: crate::output::tagged::OutputStart::Reasoning,
                    tools: vec![json!({"function":{"name":"lookup"}})],
                },
                vec![],
            );
            let mut output = Accumulator::new("request".into());
            output
                .apply(parser.push_encoded(&text[..split]).unwrap())
                .unwrap();
            output
                .apply(parser.push_encoded(&text[split..]).unwrap())
                .unwrap();
            output
                .apply(parser.finish(Termination::Eos).unwrap())
                .unwrap();
            output.finish().unwrap();
            let message = output.message();
            assert_eq!(message["reasoning_content"], "Consider 東京.");
            assert_eq!(message["content"], "Searching.");
            assert_eq!(message["tool_calls"].as_array().unwrap().len(), 2);
            let first: Value = serde_json::from_str(
                message["tool_calls"][0]["function"]["arguments"]
                    .as_str()
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(first, json!({"city":"東京", "count":3, "active":true}));
            let second: Value = serde_json::from_str(
                message["tool_calls"][1]["function"]["arguments"]
                    .as_str()
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(second, json!({"items":[1,null]}));
        }
    }

    #[test]
    fn argument_types_and_tool_names_are_validated() {
        let call = |kind: &str, value: &str| {
            format!(
                "<|open|>call tool=\"lookup\" index=\"1\"<|sep|><|open|>argument key=\"x\" type=\"{kind}\"<|sep|>{value}<|close|>argument<|sep|>{CALL_CLOSE}"
            )
        };
        for (kind, value) in [
            ("number", "false"),
            ("array", "{}"),
            ("object", "[]"),
            ("null", "0"),
            ("unknown", "1"),
        ] {
            assert!(parse_call(&call(kind, value)).is_err());
        }
        assert!(parse_call(&call("number", "3").replace("lookup", "")).is_err());
        assert_eq!(
            parse_call(&call("string", "&quot;literal\"")).unwrap().1,
            "{\"x\":\"&quot;literal\\\"\"}"
        );
    }

    #[test]
    fn complete_calls_require_tool_parsing_to_be_enabled() {
        let mut parser = XtmlOutput::new(
            OutputConfig {
                structured: false,
                start: crate::output::tagged::OutputStart::Content,
                tools: vec![],
            },
            vec![],
        );
        assert!(
            parser
                .push_encoded(concat!(
                    "<|close|>response<|sep|><|open|>tools<|sep|>",
                    "<|open|>call tool=\"lookup\" index=\"1\"<|sep|>",
                    "<|close|>call<|sep|><|close|>tools<|sep|>"
                ))
                .unwrap_err()
                .to_string()
                .contains("tool calling is disabled")
        );
    }

    #[test]
    fn output_limits_allow_incomplete_tools_but_eos_rejects_them() {
        for cause in [
            Termination::OutputLimit,
            Termination::UserStop,
            Termination::Eos,
        ] {
            let mut parser = XtmlOutput::new(
                OutputConfig {
                    structured: false,
                    start: crate::output::tagged::OutputStart::Content,
                    tools: vec![],
                },
                vec![],
            );
            parser
                .push_encoded("<|close|>response<|sep|><|open|>tools<|sep|><|open|>call")
                .unwrap();
            assert_eq!(parser.finish(cause).is_err(), cause == Termination::Eos);
        }
    }
}
