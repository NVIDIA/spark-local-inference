/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Recipient headers with internal message boundaries and ATEM function arguments.

use super::{Event, Input, Termination, tagged::OutputConfig};
use crate::Result;
use serde_json::{Map, Value};
use std::collections::VecDeque;

const START: &str = "<|start|>";
const MESSAGE: &str = "<|message|>";
const END: &str = "<|eom|>";
const TURN_END: &str = "<|eot|>";

#[derive(Clone, Copy)]
enum State {
    Header,
    Reasoning,
    Content,
    Tool,
    Closed,
}

pub struct RecipientOutput {
    buffer: String,
    header: String,
    state: State,
    tools: Vec<Value>,
    calls: usize,
    recipient: Option<String>,
    received: usize,
    spans: Vec<Option<std::ops::Range<usize>>>,
    control_tokens: Vec<(u32, String)>,
    pending_controls: VecDeque<String>,
    pending_text: String,
}

fn escape(text: &str) -> String {
    text.replace('&', "&amp;").replace("<|", "&lt;|")
}
fn unescape(text: &str) -> String {
    text.replace("&lt;|", "<|").replace("&amp;", "&")
}
fn partial(text: &str, marker: &str) -> usize {
    (1..marker.len())
        .rev()
        .find(|&n| text.ends_with(&marker[..n]))
        .unwrap_or(0)
}

impl RecipientOutput {
    pub fn new(config: OutputConfig, control_tokens: Vec<(u32, String)>) -> Self {
        Self {
            buffer: String::new(),
            header: if config.start == super::tagged::OutputStart::RecipientFrame {
                String::new()
            } else {
                "assistant".into()
            },
            state: match config.start {
                super::tagged::OutputStart::RecipientHeader
                | super::tagged::OutputStart::RecipientFrame => State::Header,
                super::tagged::OutputStart::Reasoning => State::Reasoning,
                super::tagged::OutputStart::Content => State::Content,
                super::tagged::OutputStart::ToolBody => State::Tool,
            },
            tools: config.tools,
            calls: 0,
            recipient: None,
            received: 0,
            spans: vec![],
            control_tokens,
            pending_controls: VecDeque::new(),
            pending_text: String::new(),
        }
    }
    pub fn reasoning_open(&self) -> bool {
        matches!(self.state, State::Reasoning)
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
            Input::Token(token) => {
                if let Some((_, marker)) = self.control_tokens.iter().find(|(id, _)| *id == token) {
                    self.pending_controls.push_back(marker.clone());
                }
                return Ok(vec![]);
            }
            Input::Text(text) => text,
        };
        self.received += text.len();
        self.pending_text.push_str(text);
        let mut encoded = String::new();
        while let Some(marker) = self.pending_controls.front() {
            if let Some(at) = self.pending_text.find(marker) {
                encoded.push_str(&escape(&self.pending_text[..at]));
                encoded.push_str(marker);
                self.pending_text.drain(..at + marker.len());
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
        self.buffer.push_str(&encoded);
        self.drain(false)
    }
    fn drain(&mut self, finishing: bool) -> Result<Vec<Event>> {
        let mut events = vec![];
        loop {
            match self.state {
                State::Header => {
                    let Some(at) = self.buffer.find(MESSAGE) else {
                        break;
                    };
                    if self.buffer.starts_with(START) {
                        self.header.clear();
                    }
                    self.header.push_str(&unescape(&self.buffer[..at]));
                    self.buffer.drain(..at + MESSAGE.len());
                    let header = self.header.strip_prefix(START).unwrap_or(&self.header);
                    let header = header
                        .strip_prefix("assistant")
                        .ok_or("recipient frame requires assistant role")?;
                    let recipient = header.trim().strip_prefix("to=").unwrap_or("user").trim();
                    self.recipient = None;
                    if !matches!(recipient, "self" | "user" | "commentary") {
                        if self.tools.is_empty()
                            || (recipient != "tool"
                                && !self
                                    .tools
                                    .iter()
                                    .any(|tool| tool["function"]["name"] == recipient))
                        {
                            return Err("recipient names an unavailable tool".into());
                        }
                        if recipient != "tool" {
                            self.recipient = Some(recipient.to_owned());
                        }
                    }
                    self.state = match recipient {
                        "self" => State::Reasoning,
                        "user" | "commentary" => State::Content,
                        "" => return Err("recipient header is empty".into()),
                        _ => State::Tool,
                    };
                    self.header.clear();
                }
                State::Reasoning | State::Content | State::Tool => {
                    let boundary = [END, TURN_END]
                        .into_iter()
                        .filter_map(|m| self.buffer.find(m).map(|at| (at, m)))
                        .min_by_key(|(at, _)| *at);
                    if matches!(self.state, State::Tool) {
                        let Some((at, marker)) = boundary else { break };
                        let body = unescape(&self.buffer[..at]);
                        for (name, arguments) in self.parse_calls(&body)? {
                            let index = self.calls;
                            self.calls += 1;
                            events.extend([
                                Event::ToolBegin {
                                    index,
                                    name,
                                    id: None,
                                },
                                Event::ToolArguments {
                                    index,
                                    text: arguments,
                                },
                                Event::ToolEnd { index },
                            ]);
                        }
                        self.buffer.drain(..at + marker.len());
                        self.state = if marker == TURN_END {
                            State::Closed
                        } else {
                            State::Header
                        };
                        continue;
                    }
                    let visible = boundary.map(|(at, _)| at).unwrap_or_else(|| {
                        if finishing {
                            self.buffer.len()
                        } else {
                            self.buffer.len()
                                - [END, TURN_END]
                                    .into_iter()
                                    .map(|m| partial(&self.buffer, m))
                                    .max()
                                    .unwrap_or(0)
                        }
                    });
                    if visible > 0 {
                        let start = self.consumed_bytes();
                        let text = unescape(&self.buffer[..visible]);
                        self.buffer.drain(..visible);
                        if matches!(self.state, State::Reasoning) {
                            events.push(Event::Reasoning(text));
                        } else {
                            self.spans.push(Some(start..start + text.len()));
                            events.push(Event::Content(text));
                        }
                    }
                    if let Some((_, marker)) = boundary {
                        self.buffer.drain(..marker.len());
                        self.state = if marker == TURN_END {
                            State::Closed
                        } else {
                            State::Header
                        };
                    } else {
                        break;
                    }
                }
                State::Closed => {
                    self.buffer.clear();
                    break;
                }
            }
        }
        Ok(events)
    }
    fn parse_calls(&self, body: &str) -> Result<Vec<(String, String)>> {
        let calls = parse_atem(body, &self.tools)?;
        if calls.len() != 1 {
            return Err("recipient frames require exactly one ATEM invocation".into());
        }
        if calls.iter().any(|(name, _)| {
            self.recipient
                .as_ref()
                .is_some_and(|recipient| recipient != name)
                || !self
                    .tools
                    .iter()
                    .any(|tool| tool["function"]["name"] == *name)
        }) {
            return Err("ATEM invocation does not match its registered tool recipient".into());
        }
        Ok(calls)
    }

    pub fn finish(&mut self, cause: Termination) -> Result<Vec<Event>> {
        self.spans.clear();
        self.buffer
            .push_str(&escape(&std::mem::take(&mut self.pending_text)));
        self.pending_controls.clear();
        let mut events = self.drain(true)?;
        if matches!(self.state, State::Tool) && cause == Termination::Eos {
            // EOS is suppressed before presentation, so it closes the final tool frame.
            for (name, arguments) in self.parse_calls(&unescape(&self.buffer))? {
                let index = self.calls;
                self.calls += 1;
                events.extend([
                    Event::ToolBegin {
                        index,
                        name,
                        id: None,
                    },
                    Event::ToolArguments {
                        index,
                        text: arguments,
                    },
                    Event::ToolEnd { index },
                ]);
            }
            self.buffer.clear();
        } else if matches!(self.state, State::Header)
            && !self.buffer.is_empty()
            && cause == Termination::Eos
        {
            return Err("unterminated recipient header".into());
        }
        Ok(events)
    }
}

fn attribute(value: &str) -> String {
    value
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
}

fn parse_atem(text: &str, tools: &[Value]) -> Result<Vec<(String, String)>> {
    let mut rest = text
        .trim()
        .strip_prefix("<atem:function_calls>")
        .ok_or("tool recipient requires ATEM function calls")?;
    let mut calls = vec![];
    loop {
        rest = rest.trim_start();
        if let Some(tail) = rest.strip_prefix("</atem:function_calls>") {
            if !tail.trim().is_empty() {
                return Err("unexpected content after ATEM calls".into());
            }
            break;
        }
        let tail = rest
            .strip_prefix("<atem:invoke name=\"")
            .ok_or("ATEM call requires invoke name")?;
        let (name, tail) = tail
            .split_once("\">")
            .ok_or("unterminated ATEM invoke name")?;
        let name = attribute(name);
        if name.is_empty() {
            return Err("ATEM function name is empty".into());
        }
        rest = tail;
        let function = tools
            .iter()
            .filter_map(|t| t.get("function"))
            .find(|f| f["name"] == name);
        let properties = function
            .and_then(|f| f.get("parameters"))
            .and_then(|p| p.get("properties"));
        let mut args = Map::new();
        loop {
            rest = rest.trim_start();
            if let Some(tail) = rest.strip_prefix("</atem:invoke>") {
                rest = tail;
                break;
            }
            let tail = rest
                .strip_prefix("<atem:parameter name=\"")
                .ok_or("ATEM invoke requires parameter or closing tag")?;
            let (key, tail) = tail
                .split_once("\">")
                .ok_or("unterminated ATEM parameter name")?;
            let key = attribute(key);
            if key.is_empty() || args.contains_key(&key) {
                return Err("ATEM parameter names must be nonempty and unique".into());
            }
            let (raw, tail) = tail
                .split_once("</atem:parameter>")
                .ok_or("unterminated ATEM parameter")?;
            args.insert(
                key.clone(),
                super::chatml::parameter_value(raw, properties.and_then(|p| p.get(&key)))?,
            );
            rest = tail;
        }
        calls.push((
            name,
            serde_json::to_string(&args).map_err(|e| e.to_string())?,
        ));
    }
    if calls.is_empty() {
        return Err("ATEM block contains no calls".into());
    }
    Ok(calls)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output::tagged::OutputStart;
    use serde_json::json;

    fn parser() -> RecipientOutput {
        parser_with_start(OutputStart::RecipientHeader)
    }

    fn parser_with_start(start: OutputStart) -> RecipientOutput {
        RecipientOutput::new(
            OutputConfig {
                structured: false,
                start,
                tools: vec![
                    json!({"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{
                "text":{"type":"string"},"n":{"type":"integer"},"ok":{"type":"boolean"},"list":{"type":"array","items":{"type":"integer"}},"obj":{"type":"object","properties":{"n":{"type":"integer"}}},"maybe":{"type":["string","null"]}}}}}),
                ],
            },
            vec![
                (1, START.into()),
                (2, MESSAGE.into()),
                (3, END.into()),
                (4, TURN_END.into()),
            ],
        )
    }

    fn feed(parser: &mut RecipientOutput, text: &str, split: usize) -> Vec<Event> {
        let mut events = vec![];
        // Token observations precede their decoded bytes, including a decoder
        // delta that completes several previously observed tokens.
        let mut at = 0;
        for (offset, _) in text.match_indices('<') {
            let id = [(1, START), (2, MESSAGE), (3, END), (4, TURN_END)]
                .into_iter()
                .find(|(_, marker)| text[offset..].starts_with(marker));
            if let Some((id, marker)) = id {
                for cut in [split, offset] {
                    if cut > at && cut <= offset {
                        events.extend(parser.push(Input::Text(&text[at..cut])).unwrap());
                        at = cut;
                    }
                }
                events.extend(parser.push(Input::Token(id)).unwrap());
                let end = offset + marker.len();
                if split > at && split < end {
                    events.extend(parser.push(Input::Text(&text[at..split])).unwrap());
                    at = split;
                }
                events.extend(parser.push(Input::Text(&text[at..end])).unwrap());
                at = end;
            }
        }
        if split > at {
            events.extend(parser.push(Input::Text(&text[at..split])).unwrap());
            at = split;
        }
        events.extend(parser.push(Input::Text(&text[at..])).unwrap());
        events.extend(parser.finish(Termination::Eos).unwrap());
        events
    }

    #[test]
    fn recipient_channels_and_calls_survive_every_utf8_split() {
        let text = concat!(
            " to=self<|message|>Plan café 世界<|eom|>",
            "<|start|>assistant to=commentary<|message|>Checking.\n<|eom|>",
            "<|start|>assistant to=lookup<|message|><atem:function_calls>\n<atem:invoke name=\"lookup\">\n",
            "<atem:parameter name=\"text\">  café 世界\n</atem:parameter>\n",
            "<atem:parameter name=\"n\">0003</atem:parameter>\n",
            "<atem:parameter name=\"ok\">true</atem:parameter>\n",
            "<atem:parameter name=\"list\">[1,2]</atem:parameter>\n",
            "<atem:parameter name=\"obj\">{\"n\":\"4\"}</atem:parameter>\n",
            "<atem:parameter name=\"maybe\">null</atem:parameter>\n",
            "</atem:invoke>\n</atem:function_calls><|eom|>",
            "<|start|>assistant to=self<|message|>Done thinking.<|eom|>",
            "<|start|>assistant to=user<|message|>{\"answer\":\"世界\"}"
        );
        for split in text
            .char_indices()
            .map(|(at, _)| at)
            .chain(std::iter::once(text.len()))
        {
            let mut parser = parser();
            let events = feed(&mut parser, text, split);
            let content = events
                .iter()
                .filter_map(|e| {
                    if let Event::Content(s) = e {
                        Some(s.as_str())
                    } else {
                        None
                    }
                })
                .collect::<String>();
            let reasoning = events
                .iter()
                .filter_map(|e| {
                    if let Event::Reasoning(s) = e {
                        Some(s.as_str())
                    } else {
                        None
                    }
                })
                .collect::<String>();
            assert_eq!(content, "Checking.\n{\"answer\":\"世界\"}", "split {split}");
            assert_eq!(reasoning, "Plan café 世界Done thinking.", "split {split}");
            let args = events
                .iter()
                .find_map(|e| {
                    if let Event::ToolArguments { text, .. } = e {
                        Some(serde_json::from_str::<Value>(text).unwrap())
                    } else {
                        None
                    }
                })
                .unwrap();
            assert_eq!(
                args,
                json!({"text":"  café 世界\n","n":3,"ok":true,"list":[1,2],"obj":{"n":4},"maybe":null})
            );
            assert_eq!(
                events
                    .iter()
                    .filter(|e| matches!(e, Event::ToolBegin { .. }))
                    .count(),
                1
            );
        }
    }

    #[test]
    fn literal_boundary_spellings_do_not_close_content() {
        let mut parser = parser();
        parser.state = State::Content;
        let text = "Literal <|eom|><|start|>assistant to=self<|message|> text &amp; café";
        let events = parser.push(Input::Text(text)).unwrap();
        assert_eq!(events, vec![Event::Content(text.into())]);
        assert_eq!(parser.content_spans(), vec![Some(0..text.len())]);
    }

    #[test]
    fn final_tool_frame_closes_at_suppressed_eos() {
        let text = " to=lookup<|message|><atem:function_calls><atem:invoke name=\"lookup\"><atem:parameter name=\"text\">abc</atem:parameter></atem:invoke></atem:function_calls>";
        let events = feed(&mut parser(), text, text.len());
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Event::ToolEnd { index: 0 }))
        );
    }

    #[test]
    fn incomplete_calls_fail_at_eos_and_remain_incomplete_at_limits() {
        for cause in [
            Termination::Eos,
            Termination::OutputLimit,
            Termination::Cancelled,
            Termination::ExecutionError,
            Termination::UserStop,
        ] {
            let mut parser = parser();
            parser.state = State::Tool;
            parser
                .push(Input::Text(
                    "<atem:function_calls><atem:invoke name=\"lookup\">",
                ))
                .unwrap();
            assert_eq!(parser.finish(cause).is_err(), cause == Termination::Eos);
        }
    }

    #[test]
    fn atem_attributes_decode_entities_and_preserve_scalar_strings() {
        let calls=parse_atem("<atem:function_calls><atem:invoke name=\"a&amp;&quot;&#39;&lt;&gt;\"><atem:parameter name=\"p&amp;\">  3\n</atem:parameter></atem:invoke></atem:function_calls>",&[]).unwrap();
        assert_eq!(calls[0].0, "a&\"'<>");
        assert_eq!(
            serde_json::from_str::<Value>(&calls[0].1).unwrap(),
            json!({"p&":"  3\n"})
        );
    }
    #[test]
    fn recipient_and_invocation_must_name_the_selected_tool() {
        for (recipient, invoke) in [("other", "lookup"), ("lookup", "other")] {
            let text = format!(
                " to={recipient}<|message|><atem:function_calls><atem:invoke name=\"{invoke}\"></atem:invoke></atem:function_calls>"
            );
            let mut parser = parser();
            let (header, body) = text.split_once(MESSAGE).unwrap();
            parser.push(Input::Text(header)).unwrap();
            parser.push(Input::Token(2)).unwrap();
            let header_result = parser.push(Input::Text(MESSAGE));
            if recipient == "other" {
                assert!(header_result.is_err());
            } else {
                header_result.unwrap();
                parser.push(Input::Text(body)).unwrap();
                assert!(parser.finish(Termination::Eos).is_err());
            }
        }
    }
    #[test]
    fn complete_assistant_frames_parse_after_a_closed_prompt() {
        let mut parser = parser_with_start(OutputStart::RecipientFrame);
        let events = feed(
            &mut parser,
            "<|start|>assistant to=self<|message|>Plan<|eom|><|start|>assistant to=user<|message|>Done",
            0,
        );
        assert_eq!(
            events,
            vec![
                Event::Reasoning("Plan".into()),
                Event::Content("Done".into())
            ]
        );
    }
    #[test]
    fn generic_tool_recipient_uses_one_registered_invocation_per_frame() {
        let call = "<atem:function_calls><atem:invoke name=\"lookup\"></atem:invoke></atem:function_calls>";
        let text = format!(
            " to=tool<|message|>{call}<|eom|><|start|>assistant to=lookup<|message|>{call}"
        );
        let events = feed(&mut parser(), &text, 0);
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, Event::ToolBegin { .. }))
                .count(),
            2
        );
        let mut parser = parser();
        parser.push(Input::Text(" to=tool")).unwrap();
        parser.push(Input::Token(2)).unwrap();
        parser.push(Input::Text("<|message|><atem:function_calls><atem:invoke name=\"lookup\"></atem:invoke><atem:invoke name=\"lookup\"></atem:invoke></atem:function_calls>")).unwrap();
        assert!(parser.finish(Termination::Eos).is_err());
    }
}
