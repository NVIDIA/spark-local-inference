/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! TMLv0 output frames: content-kind headers with named JSON tool invocations.
//!
//! A frame is an optional role header with an optional author or tool name, one
//! content-kind token, a body, and `<|end_message|>`. Framing tokens take effect
//! only when generated as control tokens; inside a body every control token other
//! than the frame and turn ends is literal text, as tml-renderers parses it.
//! Output that TMLv0 rejects stays visible: ordinary text at a frame boundary is
//! model text, and a JSON invocation that is malformed, unregistered or names a
//! different tool than its header is shown as its payload text.

use super::{
    Event, Input, Termination,
    tagged::{OutputConfig, OutputStart},
};
use crate::Result;
use serde_json::Value;
use std::collections::VecDeque;

const MODEL: &str = "<|message_model|>";
const USER: &str = "<|message_user|>";
const SYSTEM: &str = "<|message_system|>";
const TOOL: &str = "<|message_tool|>";
const TEXT: &str = "<|content_text|>";
const THINK: &str = "<|content_thinking|>";
const JSON_CALL: &str = "<|content_invoke_tool_json|>";
const TEXT_CALL: &str = "<|content_invoke_tool_text|>";
const TOOL_ERROR: &str = "<|content_tool_error|>";
const XML: &str = "<|content_xml|>";
const IMAGE: &str = "<|content_image|>";
const END: &str = "<|end_message|>";
const TURN_END: &str = "<|content_model_end_sampling|>";
const BEGIN: &str = "<|begin_of_text|>";

/// Framing markers; each must be one tokenizer token and frames output only as that token.
pub(crate) const CONTROL_MARKERS: [&str; 14] = [
    MODEL, USER, SYSTEM, TOOL, TEXT, THINK, JSON_CALL, TEXT_CALL, TOOL_ERROR, XML, IMAGE, END,
    TURN_END, BEGIN,
];

#[derive(Clone, Copy, PartialEq)]
enum Body {
    Text,
    Thinking,
    Call,
    /// Tool errors, text invocations and other content kinds, shown as text.
    Visible,
}

fn body_kind(marker: &str) -> Option<Body> {
    match marker {
        TEXT => Some(Body::Text),
        THINK => Some(Body::Thinking),
        JSON_CALL => Some(Body::Call),
        TEXT_CALL | TOOL_ERROR | XML | IMAGE => Some(Body::Visible),
        _ => None,
    }
}

#[derive(Clone, Copy, PartialEq)]
enum State {
    /// Between frames.
    Frame,
    /// After a leading control token that frames nothing; text waits for a header.
    Skip,
    /// Ordinary text generated at a frame boundary, ended by the next control token.
    Unframed,
    Header { model: bool },
    Body { kind: Body, model: bool },
    Closed,
}

/// The parser start after a prompt, from the last framing control token in it.
pub(crate) fn prompt_start(prompt: &[u32], control_tokens: &[(u32, String)]) -> OutputStart {
    let marker = prompt.iter().rev().find_map(|id| {
        control_tokens
            .iter()
            .find(|(token, marker)| token == id && CONTROL_MARKERS.contains(&marker.as_str()))
            .map(|(_, marker)| marker.as_str())
    });
    match marker {
        Some(MODEL | USER | SYSTEM | TOOL) => OutputStart::RecipientHeader,
        Some(THINK) => OutputStart::Reasoning,
        Some(JSON_CALL) => OutputStart::ToolBody,
        Some(TEXT | TEXT_CALL | TOOL_ERROR | XML | IMAGE) => OutputStart::Content,
        _ => OutputStart::RecipientFrame,
    }
}

pub struct TmlOutput {
    state: State,
    /// Escaped generated text with framing markers inserted only for control tokens.
    buffer: String,
    name: Option<String>,
    tools: Vec<Value>,
    calls: usize,
    received: usize,
    spans: Vec<Option<std::ops::Range<usize>>>,
    control_tokens: Vec<(u32, String)>,
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
fn first(buffer: &str, markers: &[&'static str]) -> Option<(usize, &'static str)> {
    markers
        .iter()
        .filter_map(|&marker| buffer.find(marker).map(|at| (at, marker)))
        .min_by_key(|(at, _)| *at)
}

impl TmlOutput {
    pub fn new(config: OutputConfig, control_tokens: Vec<(u32, String)>) -> Self {
        Self {
            state: match config.start {
                OutputStart::RecipientHeader => State::Header { model: true },
                OutputStart::RecipientFrame => State::Frame,
                OutputStart::Content => State::Body { kind: Body::Text, model: true },
                OutputStart::Reasoning => State::Body { kind: Body::Thinking, model: true },
                OutputStart::ToolBody => State::Body { kind: Body::Call, model: true },
            },
            buffer: String::new(),
            name: None,
            tools: config.tools,
            calls: 0,
            received: 0,
            spans: vec![],
            control_tokens,
            pending_controls: VecDeque::new(),
            pending_text: String::new(),
        }
    }

    pub fn reasoning_open(&self) -> bool {
        self.state == State::Body { kind: Body::Thinking, model: true }
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
                if let Some((_, marker)) =
                    self.control_tokens.iter().find(|(token, _)| *token == id)
                {
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
        self.drain()
    }

    /// Takes `visible` buffered body bytes and emits them for a model frame.
    fn emit(&mut self, kind: Body, model: bool, visible: usize, events: &mut Vec<Event>) {
        if visible == 0 {
            return;
        }
        let start = self.consumed_bytes();
        let text = unescape(&self.buffer[..visible]);
        self.buffer.drain(..visible);
        if !model {
            return;
        }
        if kind == Body::Thinking {
            events.push(Event::Reasoning(text));
        } else {
            self.spans.push(Some(start..start + text.len()));
            events.push(Event::Content(text));
        }
    }

    fn drain(&mut self) -> Result<Vec<Event>> {
        let mut events = vec![];
        loop {
            match self.state {
                State::Closed => {
                    // Generation ends at the turn end; later text has no frame.
                    self.buffer.clear();
                    break;
                }
                State::Frame | State::Skip => {
                    let skipping = self.state == State::Skip;
                    let Some((at, marker)) = first(&self.buffer, &CONTROL_MARKERS) else {
                        if !skipping && !self.buffer.is_empty() {
                            self.state = State::Unframed;
                            continue;
                        }
                        self.buffer.clear();
                        break;
                    };
                    if at > 0 && !skipping {
                        self.state = State::Unframed;
                        continue;
                    }
                    self.buffer.drain(..at + marker.len());
                    self.name = None;
                    self.state = match marker {
                        MODEL => State::Header { model: true },
                        USER | SYSTEM | TOOL => State::Header { model: false },
                        TURN_END => State::Closed,
                        // A frame end between frames closes nothing.
                        END => self.state,
                        other => body_kind(other)
                            .map_or(State::Skip, |kind| State::Body { kind, model: true }),
                    };
                }
                State::Unframed => {
                    let end = first(&self.buffer, &CONTROL_MARKERS);
                    let visible = end.map_or(self.buffer.len(), |(at, _)| at);
                    self.emit(Body::Text, true, visible, &mut events);
                    match end {
                        Some((_, END)) => {
                            self.buffer.drain(..END.len());
                            self.state = State::Frame;
                        }
                        Some(_) => self.state = State::Frame,
                        None => break,
                    }
                }
                State::Header { model } => {
                    let Some((at, marker)) = first(&self.buffer, &CONTROL_MARKERS) else {
                        break;
                    };
                    let name = unescape(&self.buffer[..at]);
                    self.buffer.drain(..at + marker.len());
                    self.state = match marker {
                        MODEL => State::Header { model: true },
                        USER | SYSTEM | TOOL => State::Header { model: false },
                        END => State::Frame,
                        TURN_END => State::Closed,
                        other => match body_kind(other) {
                            Some(kind) => {
                                self.name = (!name.is_empty()).then_some(name);
                                State::Body { kind, model }
                            }
                            None => State::Header { model },
                        },
                    };
                }
                State::Body { kind, model } => {
                    let end = first(&self.buffer, &[END, TURN_END]);
                    if kind == Body::Call {
                        if self.buffer.len() > 1 << 20 {
                            return Err("TML tool call exceeds the 1 MiB parser bound".into());
                        }
                        let Some((at, marker)) = end else { break };
                        let start = self.consumed_bytes();
                        let body = unescape(&self.buffer[..at]);
                        self.buffer.drain(..at + marker.len());
                        if model {
                            events.extend(self.call_or_text(&body, start));
                        }
                    } else {
                        let visible = end.map_or(self.buffer.len(), |(at, _)| at);
                        self.emit(kind, model, visible, &mut events);
                        let Some((_, marker)) = end else { break };
                        self.buffer.drain(..marker.len());
                    }
                    self.state = if end.is_some_and(|(_, marker)| marker == END) {
                        State::Frame
                    } else {
                        State::Closed
                    };
                }
            }
        }
        Ok(events)
    }

    /// A registered invocation, or its payload as visible text when TMLv0 rejects it.
    fn call_or_text(&mut self, body: &str, start: usize) -> Vec<Event> {
        if let Some(events) = self.call(body) {
            return events;
        }
        if body.is_empty() {
            return vec![];
        }
        self.spans.push(Some(start..start + body.len()));
        vec![Event::Content(body.to_owned())]
    }

    fn call(&mut self, body: &str) -> Option<Vec<Event>> {
        let value: Value = serde_json::from_str(body).ok()?;
        let object = value.as_object()?;
        let name = object.get("name")?.as_str()?;
        let args = object.get("args").filter(|v| v.is_object())?;
        if object.len() != 2
            || !self
                .tools
                .iter()
                .any(|tool| tool["function"]["name"] == name)
            || self.name.as_ref().is_some_and(|header| header != name)
        {
            return None;
        }
        let index = self.calls;
        self.calls += 1;
        Some(vec![
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
        if matches!(cause, Termination::Cancelled | Termination::ExecutionError) {
            self.buffer.clear();
            self.pending_text.clear();
            self.pending_controls.clear();
            return Ok(vec![]);
        }
        self.buffer
            .push_str(&escape(&std::mem::take(&mut self.pending_text)));
        self.pending_controls.clear();
        let mut events = self.drain()?;
        if let State::Body { kind: Body::Call, model: true } = self.state {
            let start = self.consumed_bytes();
            let body = unescape(&std::mem::take(&mut self.buffer));
            // A call cut off by the output limit or a stop string is never executed.
            if cause == Termination::Eos {
                events.extend(self.call_or_text(&body, start));
            }
        }
        // An unterminated header carries no content.
        self.buffer.clear();
        self.state = State::Closed;
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
            CONTROL_MARKERS
                .iter()
                .enumerate()
                .map(|(i, s)| (i as u32 + 1, (*s).to_owned()))
                .collect(),
        )
    }

    fn feed(p: &mut TmlOutput, marker: &str, fragmented: bool) -> Result<Vec<Event>> {
        let id = p
            .control_tokens
            .iter()
            .find(|(_, m)| m == marker)
            .unwrap()
            .0;
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

    type Parsed = (String, String, Vec<(String, String)>);

    /// Feeds framing markers as control tokens and other pieces as ordinary text.
    fn run(start: OutputStart, pieces: &[&str], cause: Termination) -> Parsed {
        let mut p = parser(start);
        let mut events = vec![];
        for piece in pieces {
            if CONTROL_MARKERS.contains(piece) {
                events.extend(feed(&mut p, piece, true).unwrap());
            } else {
                events.extend(p.push(Input::Text(piece)).unwrap());
            }
        }
        events.extend(p.finish(cause).unwrap());
        let (mut reasoning, mut content, mut calls) = (String::new(), String::new(), vec![]);
        for event in events {
            match event {
                Event::Reasoning(text) => reasoning.push_str(&text),
                Event::Content(text) => content.push_str(&text),
                Event::ToolBegin { name, .. } => calls.push((name, String::new())),
                Event::ToolArguments { text, .. } => calls.last_mut().unwrap().1.push_str(&text),
                _ => {}
            }
        }
        (reasoning, content, calls)
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
                feed(&mut p, MODEL, fragmented).unwrap();
                p.push(Input::Text("lookup")).unwrap();
                feed(&mut p, JSON_CALL, fragmented).unwrap();
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
            feed(&mut p, MODEL, fragmented).unwrap();
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
    fn frames_without_headers_and_frames_of_other_authors() {
        let frame = OutputStart::RecipientFrame;
        let call = r#"{"name":"lookup","args":{}}"#;
        assert_eq!(
            run(frame, &[THINK, "t", END, TEXT, "a", END, JSON_CALL, call, END], Termination::Eos),
            ("t".into(), "a".into(), vec![("lookup".into(), "{}".into())])
        );
        assert_eq!(
            run(frame, &[USER, TEXT, "echo", END, MODEL, "bob", TEXT, "hi", END, TOOL, "lookup", TEXT, "x", END],
                Termination::Eos),
            (String::new(), "hi".into(), vec![])
        );
        let skipped = [BEGIN, "noise", END, "echo", MODEL, TEXT, "x", END];
        assert_eq!(run(frame, &skipped, Termination::Eos).1, "x");
        assert_eq!(run(frame, &["plain", END, MODEL, TEXT, " more", END], Termination::Eos).1, "plain more");
        assert_eq!(
            run(frame, &["plain", MODEL, THINK, "t", END], Termination::Eos),
            ("t".into(), "plain".into(), vec![])
        );
        let visible = [MODEL, TOOL_ERROR, "bad", END, MODEL, "lookup", TEXT_CALL, "raw", END, MODEL, TEXT, "a",
                       MODEL, "b", END];
        assert_eq!(run(frame, &visible, Termination::Eos).1, "badrawa<|message_model|>b");
    }

    #[test]
    fn rejected_invocations_surface_as_text_and_open_calls_complete_only_at_eos() {
        let frame = OutputStart::RecipientFrame;
        for body in [
            r#"{"name":"other","args":{}}"#,
            r#"{"name":"lookup","args":[]}"#,
            r#"{"name":"lookup","args":{},"extra":1}"#,
            r#"{"name":7,"args":{}}"#,
            "incomplete",
        ] {
            assert_eq!(run(frame, &[MODEL, JSON_CALL, body, END], Termination::Eos).1, body);
            assert_eq!(run(OutputStart::ToolBody, &[body], Termination::Eos).1, body);
        }
        assert_eq!(
            run(frame, &[MODEL, "other", JSON_CALL, r#"{"name":"lookup","args":{}}"#, END], Termination::Eos).1,
            r#"{"name":"lookup","args":{}}"#
        );
        let open = [MODEL, "lookup", JSON_CALL, r#"{"name":"lookup","args":{"ok":true}}"#];
        assert_eq!(run(frame, &open, Termination::Eos).2, vec![("lookup".into(), r#"{"ok":true}"#.into())]);
        for cause in [Termination::OutputLimit, Termination::UserStop, Termination::Cancelled, Termination::ExecutionError] {
            assert_eq!(run(frame, &open, cause), Parsed::default());
        }
        assert_eq!(run(frame, &[MODEL, "lookup"], Termination::Eos), Parsed::default());
    }

    #[test]
    fn prompt_tail_selects_the_parser_start() {
        let controls: Vec<_> = CONTROL_MARKERS
            .iter()
            .enumerate()
            .map(|(i, s)| (i as u32 + 1, (*s).to_owned()))
            .collect();
        let id = |marker: &str| controls.iter().find(|(_, m)| m == marker).unwrap().0;
        for (tail, start) in [
            (vec![id(TEXT), 100, id(END)], OutputStart::RecipientFrame),
            (vec![id(END), id(TURN_END)], OutputStart::RecipientFrame),
            (vec![id(MODEL)], OutputStart::RecipientHeader),
            (vec![id(MODEL), id(TEXT), 100, 101], OutputStart::Content),
            (vec![id(MODEL), id(THINK), 100], OutputStart::Reasoning),
            (vec![id(MODEL), 100, id(JSON_CALL)], OutputStart::ToolBody),
            (vec![100], OutputStart::RecipientFrame),
        ] {
            assert_eq!(prompt_start(&tail, &controls), start);
        }
    }
}
