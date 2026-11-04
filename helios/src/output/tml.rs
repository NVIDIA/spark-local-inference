/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Token-delimited content headers with named JSON tool invocations.

use super::{
    Event, Input, Termination,
    tagged::{OutputConfig, OutputStart},
};
use crate::Result;
use serde_json::Value;
use std::collections::VecDeque;

pub const START: &str = "<|message_model|>";
pub const TEXT: &str = "<|content_text|>";
pub const THINK: &str = "<|content_thinking|>";
pub const TOOL: &str = "<|content_invoke_tool_json|>";
pub const END: &str = "<|end_message|>";
pub const TURN_END: &str = "<|content_model_end_sampling|>";

#[derive(Clone, Copy, PartialEq)]
enum State {
    Frame,
    Header,
    Content,
    Reasoning,
    Tool,
    Closed,
}

pub struct TmlOutput {
    state: State,
    buffer: String,
    recipient: Option<String>,
    tools: Vec<Value>,
    calls: usize,
    received: usize,
    spans: Vec<Option<std::ops::Range<usize>>>,
    controls: Vec<(u32, String)>,
    pending_controls: VecDeque<String>,
    pending_text: String,
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
        .find(|&n| text.ends_with(&marker[..n]))
        .unwrap_or(0)
}

impl TmlOutput {
    pub fn new(config: OutputConfig, controls: Vec<(u32, String)>) -> Self {
        Self {
            state: match config.start {
                OutputStart::RecipientHeader => State::Header,
                OutputStart::RecipientFrame => State::Frame,
                OutputStart::Content => State::Content,
                OutputStart::Reasoning => State::Reasoning,
                OutputStart::ToolBody => State::Tool,
            },
            buffer: String::new(),
            recipient: None,
            tools: config.tools,
            calls: 0,
            received: 0,
            spans: vec![],
            controls,
            pending_controls: VecDeque::new(),
            pending_text: String::new(),
        }
    }

    pub fn reasoning_open(&self) -> bool {
        self.state == State::Reasoning
    }
    pub fn content_spans(&self) -> Vec<Option<std::ops::Range<usize>>> {
        self.spans.clone()
    }
    pub fn consumed_bytes(&self) -> usize {
        self.received
            .saturating_sub(unescape(&self.buffer).len() + self.pending_text.len())
    }

    pub fn push(&mut self, input: Input<'_>) -> Result<Vec<Event>> {
        self.spans.clear();
        let text = match input {
            Input::Token(id) => {
                if let Some((_, marker)) = self.controls.iter().find(|(token, _)| *token == id) {
                    self.pending_controls.push_back(marker.clone());
                }
                return Ok(vec![]);
            }
            Input::Text(text) => text,
        };
        self.received += text.len();
        self.pending_text.push_str(text);
        while let Some(marker) = self.pending_controls.front() {
            if let Some(at) = self.pending_text.find(marker) {
                self.buffer.push_str(&escape(&self.pending_text[..at]));
                self.buffer.push_str(marker);
                self.pending_text.drain(..at + marker.len());
                self.pending_controls.pop_front();
            } else {
                let n = self.pending_text.len() - partial(&self.pending_text, marker);
                self.buffer.push_str(&escape(&self.pending_text[..n]));
                self.pending_text.drain(..n);
                break;
            }
        }
        if self.pending_controls.is_empty() {
            self.buffer
                .push_str(&escape(&std::mem::take(&mut self.pending_text)));
        }
        self.drain(false)
    }

    fn drain(&mut self, finishing: bool) -> Result<Vec<Event>> {
        let mut events = vec![];
        loop {
            match self.state {
                State::Frame => {
                    if self.buffer.starts_with(TURN_END) {
                        self.buffer.drain(..TURN_END.len());
                        self.state = State::Closed;
                    } else if self.buffer.starts_with(START) {
                        self.buffer.drain(..START.len());
                        self.state = State::Header;
                    } else if self.buffer.is_empty()
                        || (!finishing
                            && [START, TURN_END]
                                .iter()
                                .any(|m| m.starts_with(&self.buffer)))
                    {
                        break;
                    } else {
                        return Err("TML output requires a model message header".into());
                    }
                }
                State::Header => {
                    let boundary = [TEXT, THINK, TOOL]
                        .into_iter()
                        .filter_map(|m| self.buffer.find(m).map(|at| (at, m)))
                        .min_by_key(|(at, _)| *at);
                    let Some((at, marker)) = boundary else { break };
                    let recipient = unescape(&self.buffer[..at]);
                    self.recipient = None;
                    self.state = if marker == TOOL {
                        if recipient.is_empty()
                            || !self
                                .tools
                                .iter()
                                .any(|tool| tool["function"]["name"] == recipient)
                        {
                            return Err("TML header names an unavailable tool".into());
                        }
                        self.recipient = Some(recipient);
                        State::Tool
                    } else {
                        if !recipient.is_empty() {
                            return Err("TML text and thinking headers cannot name a tool".into());
                        }
                        if marker == THINK {
                            State::Reasoning
                        } else {
                            State::Content
                        }
                    };
                    self.buffer.drain(..at + marker.len());
                }
                State::Content | State::Reasoning | State::Tool => {
                    let boundary = [END, TURN_END]
                        .into_iter()
                        .filter_map(|m| self.buffer.find(m).map(|at| (at, m)))
                        .min_by_key(|(at, _)| *at);
                    if self.state == State::Tool {
                        if self.buffer.len() > 1 << 20 {
                            return Err("TML tool call exceeds the 1 MiB parser bound".into());
                        }
                        let Some((at, marker)) = boundary else { break };
                        events.extend(self.call(&unescape(&self.buffer[..at]))?);
                        self.buffer.drain(..at + marker.len());
                        self.state = if marker == END {
                            State::Frame
                        } else {
                            State::Closed
                        };
                        continue;
                    }
                    let visible = boundary.map(|(at, _)| at).unwrap_or_else(|| {
                        self.buffer.len()
                            - if finishing {
                                0
                            } else {
                                [END, TURN_END]
                                    .iter()
                                    .map(|m| partial(&self.buffer, m))
                                    .max()
                                    .unwrap_or(0)
                            }
                    });
                    if visible > 0 {
                        let start = self.consumed_bytes();
                        let text = unescape(&self.buffer[..visible]);
                        self.buffer.drain(..visible);
                        if self.state == State::Reasoning {
                            events.push(Event::Reasoning(text));
                        } else {
                            self.spans.push(Some(start..start + text.len()));
                            events.push(Event::Content(text));
                        }
                    }
                    if let Some((_, marker)) = boundary {
                        self.buffer.drain(..marker.len());
                        self.state = if marker == END {
                            State::Frame
                        } else {
                            State::Closed
                        };
                    } else {
                        break;
                    }
                }
                State::Closed => {
                    if !self.buffer.is_empty() {
                        return Err("content after TML end of turn".into());
                    }
                    break;
                }
            }
        }
        Ok(events)
    }

    fn call(&mut self, body: &str) -> Result<Vec<Event>> {
        let value: Value = serde_json::from_str(body)
            .map_err(|error| format!("invalid TML tool JSON: {error}"))?;
        let object = value
            .as_object()
            .ok_or("TML tool call must be a JSON object")?;
        let name = object
            .get("name")
            .and_then(Value::as_str)
            .ok_or("TML tool call requires a name")?;
        let args = object
            .get("args")
            .filter(|v| v.is_object())
            .ok_or("TML tool args must be an object")?;
        if object.len() != 2
            || !self
                .tools
                .iter()
                .any(|tool| tool["function"]["name"] == name)
            || self
                .recipient
                .as_ref()
                .is_some_and(|recipient| recipient != name)
        {
            return Err("TML tool call does not match its registered header".into());
        }
        let index = self.calls;
        self.calls += 1;
        Ok(vec![
            Event::ToolBegin {
                index,
                name: name.to_owned(),
                id: None,
            },
            Event::ToolArguments {
                index,
                text: args.to_string(),
            },
            Event::ToolEnd { index },
        ])
    }

    pub fn finish(&mut self, cause: Termination) -> Result<Vec<Event>> {
        self.spans.clear();
        if matches!(cause, Termination::Cancelled | Termination::ExecutionError)
            || (cause != Termination::Eos
                && matches!(self.state, State::Frame | State::Header | State::Tool))
        {
            self.buffer.clear();
            self.pending_text.clear();
            self.pending_controls.clear();
            return Ok(vec![]);
        }
        self.buffer
            .push_str(&escape(&std::mem::take(&mut self.pending_text)));
        self.pending_controls.clear();
        let mut events = self.drain(true)?;
        if self.state == State::Tool && cause == Termination::Eos {
            events.extend(self.call(&unescape(&self.buffer))?);
            self.buffer.clear();
        } else if self.state == State::Header && cause == Termination::Eos {
            return Err("unterminated TML content header".into());
        }
        Ok(events)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parser(start: OutputStart) -> TmlOutput {
        TmlOutput::new(
            OutputConfig {
                structured: false,
                start,
                tools: vec![json!({"type":"function","function":{"name":"lookup"}})],
            },
            [START, TEXT, THINK, TOOL, END, TURN_END]
                .iter()
                .enumerate()
                .map(|(i, s)| (i as u32 + 1, (*s).to_owned()))
                .collect(),
        )
    }

    fn feed(p: &mut TmlOutput, marker: &str, fragmented: bool) -> Result<Vec<Event>> {
        let id = p.controls.iter().find(|(_, m)| m == marker).unwrap().0;
        let mut events = p.push(Input::Token(id))?;
        if fragmented {
            for c in marker.chars() {
                events.extend(p.push(Input::Text(&c.to_string()))?);
            }
        } else {
            events.extend(p.push(Input::Text(marker))?);
        }
        Ok(events)
    }

    #[test]
    fn content_headers_stream_reasoning_content_and_parallel_tools() {
        for fragmented in [false, true] {
            let mut p = parser(OutputStart::RecipientHeader);
            assert!(feed(&mut p, THINK, fragmented).unwrap().is_empty());
            assert!(p.reasoning_open());
            assert_eq!(
                p.push(Input::Text("Plan 世界")).unwrap(),
                vec![Event::Reasoning("Plan 世界".into())]
            );
            feed(&mut p, END, fragmented).unwrap();
            for index in 0..2 {
                feed(&mut p, START, fragmented).unwrap();
                p.push(Input::Text("lookup")).unwrap();
                feed(&mut p, TOOL, fragmented).unwrap();
                p.push(Input::Text(
                    r#"{"name":"lookup","args":{"q":"世界 <|end_message|>","n":3}}"#,
                ))
                .unwrap();
                assert_eq!(
                    feed(&mut p, END, fragmented).unwrap(),
                    vec![
                        Event::ToolBegin {
                            index,
                            name: "lookup".into(),
                            id: None
                        },
                        Event::ToolArguments {
                            index,
                            text: r#"{"q":"世界 <|end_message|>","n":3}"#.into()
                        },
                        Event::ToolEnd { index },
                    ]
                );
            }
            feed(&mut p, START, fragmented).unwrap();
            feed(&mut p, TEXT, fragmented).unwrap();
            let before = p.received;
            assert_eq!(
                p.push(Input::Text("Answer & café")).unwrap(),
                vec![Event::Content("Answer & café".into())]
            );
            assert_eq!(
                p.content_spans(),
                vec![Some(before..before + "Answer & café".len())]
            );
            feed(&mut p, END, fragmented).unwrap();
            assert!(p.finish(Termination::Eos).unwrap().is_empty());
        }
    }

    #[test]
    fn ordinary_delimiter_spellings_remain_content() {
        let mut p = parser(OutputStart::Content);
        let text = "literal <|end_message|><|message_model|><|content_thinking|> &lt;|";
        let mut result = String::new();
        for c in text.chars() {
            for event in p.push(Input::Text(&c.to_string())).unwrap() {
                let Event::Content(value) = event else {
                    panic!("ordinary text changed framing")
                };
                result.push_str(&value);
            }
        }
        assert_eq!(result, text);
        assert!(p.finish(Termination::OutputLimit).unwrap().is_empty());
    }

    #[test]
    fn malformed_calls_and_headers_fail_without_leaking_tool_content() {
        for body in [
            r#"{"name":"other","args":{}}"#,
            r#"{"name":"lookup","args":[]}"#,
            r#"{"name":"lookup","args":{},"extra":1}"#,
            "incomplete",
        ] {
            let mut p = parser(OutputStart::ToolBody);
            assert!(p.push(Input::Text(body)).unwrap().is_empty());
            assert!(p.finish(Termination::Eos).is_err());
        }
        for cause in [
            Termination::OutputLimit,
            Termination::UserStop,
            Termination::Cancelled,
            Termination::ExecutionError,
        ] {
            let mut p = parser(OutputStart::ToolBody);
            p.push(Input::Text("incomplete")).unwrap();
            assert!(p.finish(cause).unwrap().is_empty());
            let mut p = parser(OutputStart::RecipientFrame);
            p.push(Input::Token(1)).unwrap();
            p.push(Input::Text("<|mess")).unwrap();
            assert!(p.finish(cause).unwrap().is_empty());
        }
        let mut p = parser(OutputStart::RecipientHeader);
        p.push(Input::Text("other")).unwrap();
        assert!(feed(&mut p, TOOL, false).is_err());
        let mut p = parser(OutputStart::RecipientFrame);
        assert!(feed(&mut p, TEXT, false).is_err());
    }

    #[test]
    fn suppressed_turn_eos_completes_json_but_rejects_an_incomplete_header() {
        let mut p = parser(OutputStart::RecipientHeader);
        p.push(Input::Text("lookup")).unwrap();
        feed(&mut p, TOOL, false).unwrap();
        p.push(Input::Text(r#"{"name":"lookup","args":{"ok":true}}"#))
            .unwrap();
        assert_eq!(
            p.finish(Termination::Eos).unwrap(),
            vec![
                Event::ToolBegin {
                    index: 0,
                    name: "lookup".into(),
                    id: None
                },
                Event::ToolArguments {
                    index: 0,
                    text: r#"{"ok":true}"#.into()
                },
                Event::ToolEnd { index: 0 },
            ]
        );
        let mut p = parser(OutputStart::RecipientHeader);
        p.push(Input::Text("lookup")).unwrap();
        assert!(p.finish(Termination::Eos).is_err());
    }
}
