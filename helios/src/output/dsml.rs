/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Translate the official DeepSeek parser's actions into shared output events.

use super::{
    Event, Input, ModelOutput, Termination,
    tagged::{OutputConfig, OutputStart},
};
use crate::Result;
use deepseek_recipe::stream::{
    ParsingOptions, ThinkingInitialStage,
    output_parser::{ChunkKind, OutputAction, OutputActionSegment, OutputParser},
};
use serde::de::{Deserializer, MapAccess, Visitor};
use serde_json::Value;
use std::collections::HashSet;

const BUFFER_LIMIT: usize = 1024 * 1024;

#[derive(Default)]
struct Call {
    name: String,
    arguments: String,
}

pub struct DsmlOutput<const SPACED: bool> {
    consumed: usize,
    reasoning_open: bool,
    spans: Vec<Option<std::ops::Range<usize>>>,
    machine: Option<OutputParser>,
    pending: String,
    last_action: OutputAction,
    call: Option<Call>,
    calls: usize,
    structured: bool,
    verbatim: bool,
    special_tokens: Vec<(u32, String)>,
    last_token: Option<u32>,
}

impl<const SPACED: bool> DsmlOutput<SPACED> {
    fn apply(&mut self, actions: Vec<OutputActionSegment>, events: &mut Vec<Event>) -> Result<()> {
        let mut consumed = 0;
        for segment in actions {
            let end = consumed + segment.byte_len;
            let source = self
                .pending
                .get(consumed..end)
                .ok_or("DeepSeek parser returned an invalid source span")?;
            // The upstream parser skips leading newlines; other non-reasoning actions end reasoning.
            if self.reasoning_open
                && segment.action != OutputAction::Reasoning
                && !(segment.action == OutputAction::Skip && source.bytes().all(|b| b == b'\n'))
            {
                self.reasoning_open = false;
            }
            let repeated = self.last_action == segment.action;
            self.last_action = segment.action;
            match segment.action {
                OutputAction::Content => {
                    if !source.is_empty() {
                        self.spans
                            .push(Some(self.consumed + consumed..self.consumed + end));
                        events.push(Event::Content(source.to_owned()));
                        self.verbatim = self.structured && !source.trim().is_empty();
                    }
                }
                OutputAction::Reasoning => {
                    if !source.is_empty() {
                        events.push(Event::Reasoning(source.to_owned()));
                    }
                }
                OutputAction::ToolCallBegin if !repeated => {
                    if self.call.is_some() {
                        return Err(
                            "DeepSeek parser began a tool call before completing its predecessor"
                                .into(),
                        );
                    }
                    self.call = Some(Call::default());
                }
                OutputAction::ToolName => {
                    self.call
                        .as_mut()
                        .ok_or("DeepSeek tool name has no call")?
                        .name
                        .push_str(source);
                }
                OutputAction::ToolNameEnd if !repeated => {
                    let name = &self
                        .call
                        .as_ref()
                        .ok_or("DeepSeek tool name has no call")?
                        .name;
                    if name.is_empty() {
                        return Err("tool function name is empty".into());
                    }
                }
                OutputAction::LabelToolCallArguments(label) if !repeated => {
                    self.call
                        .as_mut()
                        .ok_or("DeepSeek tool arguments have no call")?
                        .arguments
                        .push_str(label.as_str());
                }
                OutputAction::ToolCallArgumentsText {
                    is_string_value: string,
                } => {
                    let arguments = &mut self
                        .call
                        .as_mut()
                        .ok_or("DeepSeek tool arguments have no call")?
                        .arguments;
                    if string {
                        let escaped = serde_json::to_string(source)
                            .expect("serializing a string cannot fail");
                        arguments.push_str(&escaped[1..escaped.len() - 1]);
                    } else {
                        arguments.push_str(source);
                    }
                }
                OutputAction::ToolCallArgumentsEnd { closing_text } if !repeated => {
                    let mut call = self
                        .call
                        .take()
                        .ok_or("DeepSeek tool arguments have no call")?;
                    call.arguments.push_str(closing_text);
                    validate_arguments(&call.arguments)?;
                    let index = self.calls;
                    events.extend([
                        Event::ToolBegin {
                            index,
                            name: call.name,
                            id: None,
                        },
                        Event::ToolArguments {
                            index,
                            text: call.arguments,
                        },
                        Event::ToolEnd { index },
                    ]);
                    self.calls += 1;
                }
                OutputAction::Label(label) if !repeated => {
                    self.spans.push(None);
                    events.push(Event::Content(label.into()))
                }
                OutputAction::ToSpace => {
                    self.spans
                        .push(Some(self.consumed + consumed..self.consumed + end));
                    events.push(Event::Content(" ".into()));
                }
                _ => {}
            }
            if self.call.as_ref().is_some_and(|call| {
                call.name.len().saturating_add(call.arguments.len()) > BUFFER_LIMIT
            }) {
                return Err("DeepSeek tool call exceeds the 1 MiB buffer bound".into());
            }
            consumed = end;
            if self.verbatim {
                break;
            }
        }
        self.pending.drain(..consumed);
        self.consumed += consumed;
        if self.verbatim && !self.pending.is_empty() {
            self.spans
                .push(Some(self.consumed..self.consumed + self.pending.len()));
            self.consumed += self.pending.len();
            events.push(Event::Content(std::mem::take(&mut self.pending)));
        }
        if self.pending.len() > BUFFER_LIMIT {
            return Err("DeepSeek parser exceeds the 1 MiB buffer bound".into());
        }
        Ok(())
    }

    pub fn reasoning_open(&self) -> bool {
        self.reasoning_open
    }

    pub fn with_special_tokens(mut self, tokens: Vec<(u32, String)>) -> Self {
        self.special_tokens = tokens;
        self
    }

    fn present(&mut self, text: &str, kind: ChunkKind, events: &mut Vec<Event>) -> Result<()> {
        let mut rest = text;
        while self.structured
            && !self.verbatim
            && !rest.is_empty()
            && kind != ChunkKind::SpecialToken
        {
            let length = rest.chars().next().unwrap().len_utf8();
            self.feed(&rest[..length], kind, events)?;
            rest = &rest[length..];
        }
        if !rest.is_empty() {
            if self.verbatim {
                self.spans
                    .push(Some(self.consumed..self.consumed + rest.len()));
                self.consumed += rest.len();
                events.push(Event::Content(rest.to_owned()));
            } else {
                self.feed(rest, kind, events)?;
            }
        }
        Ok(())
    }

    fn feed(&mut self, text: &str, kind: ChunkKind, events: &mut Vec<Event>) -> Result<()> {
        self.pending.push_str(text);
        let actions = self.machine.as_mut().unwrap().feed(text, kind);
        self.apply(actions, events)
    }
}

impl<const SPACED: bool> ModelOutput for DsmlOutput<SPACED> {
    type Config = OutputConfig;

    fn new(config: OutputConfig) -> Self {
        Self {
            consumed: 0,
            reasoning_open: config.start == OutputStart::Reasoning,
            spans: vec![],
            machine: Some(OutputParser::new(ParsingOptions {
                parse_tool_calls: !config.tools.is_empty(),
                starts_in_tool_calls: config.start == OutputStart::ToolBody,
                // Helios grammars also support scalar JSON and non-JSON languages.
                parse_json_output: false,
                thinking_initial_stage: (config.start == OutputStart::Reasoning)
                    .then_some(ThinkingInitialStage::ReasoningStart),
                ..ParsingOptions::default()
            })),
            pending: String::new(),
            last_action: OutputAction::Skip,
            call: None,
            calls: 0,
            structured: config.structured,
            verbatim: false,
            special_tokens: Vec::new(),
            last_token: None,
        }
    }

    fn grammar_envelope(config: &Self::Config) -> Option<crate::grammar::OutputEnvelope> {
        Some(crate::grammar::OutputEnvelope {
            recipient: None,
            reasoning_end: (config.start == OutputStart::Reasoning).then_some("</think>"),
            tool: (!config.tools.is_empty()).then_some(if SPACED {
                ("<｜DSML｜ calls>", "</｜DSML｜ calls>")
            } else {
                ("<｜DSML｜tool_calls>", "</｜DSML｜tool_calls>")
            }),
            function: None,
            tool_ends_reasoning: false,
            response: None,
            message_close: None,
            tool_body_start: config.start == OutputStart::ToolBody,
        })
    }

    fn content_spans(&self) -> Vec<Option<std::ops::Range<usize>>> {
        self.spans.clone()
    }
    fn consumed_bytes(&self) -> usize {
        self.consumed
    }
    fn push(&mut self, input: Input<'_>) -> Result<Vec<Event>> {
        self.spans.clear();
        if self.machine.is_none() {
            return Err("output presentation is finished".into());
        }
        let text = match input {
            Input::Token(token) => {
                self.last_token = Some(token);
                return Ok(vec![]);
            }
            Input::Text(text) => text,
        };
        // A completed control token ends its decoded span; clipped stop spans carry no token ID.
        let special = self.last_token.take().and_then(|token| {
            self.special_tokens
                .iter()
                .find(|(id, _)| *id == token)
                .map(|(_, text)| text.clone())
        });
        let plain_kind = if self.special_tokens.is_empty() {
            ChunkKind::Text
        } else {
            ChunkKind::PlainTokens
        };
        let mut events = Vec::new();
        if let Some(marker) = special {
            if !text.ends_with(&marker) {
                return Err("DeepSeek control token does not match decoded text".into());
            }
            let end = text.len() - marker.len();
            self.present(&text[..end], plain_kind, &mut events)?;
            self.present(&text[end..], ChunkKind::SpecialToken, &mut events)?;
        } else {
            self.present(text, plain_kind, &mut events)?;
        }
        Ok(events)
    }

    fn finish(&mut self, cause: Termination) -> Result<Vec<Event>> {
        self.spans.clear();
        let machine = self
            .machine
            .take()
            .ok_or("output presentation is finished")?;
        if matches!(cause, Termination::Cancelled | Termination::ExecutionError) {
            self.pending.clear();
            self.call = None;
            return Ok(vec![]);
        }
        let mut events = Vec::new();
        if !self.verbatim {
            self.apply(machine.finish(), &mut events)?;
        }
        if self.call.take().is_some() && cause == Termination::Eos {
            return Err("incomplete DeepSeek tool call at end of response".into());
        }
        Ok(events)
    }
}

fn validate_arguments(arguments: &str) -> Result<()> {
    struct Arguments;
    impl<'de> Visitor<'de> for Arguments {
        type Value = ();
        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a tool argument object with nonempty, unique keys")
        }
        fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> std::result::Result<(), M::Error> {
            let mut keys = HashSet::new();
            while let Some(key) = map.next_key::<String>()? {
                if key.is_empty() || !keys.insert(key) {
                    return Err(serde::de::Error::custom(
                        "tool argument names must be nonempty and unique",
                    ));
                }
                map.next_value::<Value>()?;
            }
            Ok(())
        }
    }
    let mut decoder = serde_json::Deserializer::from_str(arguments);
    decoder
        .deserialize_map(Arguments)
        .map_err(|error| format!("invalid DeepSeek tool arguments: {error}"))?;
    decoder
        .end()
        .map_err(|error| format!("invalid DeepSeek tool arguments: {error}"))?;
    Ok(())
}

#[cfg(test)]
mod deepseek41_syntax_tests {
    use super::*;
    use crate::output::tagged::OutputConfig;
    type DeepSeekOutput = DsmlOutput<true>;
    use crate::output::{Input, ModelOutput, Termination};
    use crate::server::output::Accumulator;
    use serde_json::json;

    #[test]
    fn every_utf8_split_preserves_multiple_calls_and_typed_values() {
        let text = "thought</think><｜DSML｜ calls>\n<｜DSML｜ invoke name=\"lookup\">\n<｜DSML｜ parameter name=\"city\" string=\"true\">東京</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"lookup\"><｜DSML｜ parameter name=\"n\" string=\"false\">3</｜DSML｜ parameter></｜DSML｜ invoke>\n</｜DSML｜ calls>";
        for split in text.char_indices().map(|(offset, _)| offset) {
            let mut parser = DeepSeekOutput::new(OutputConfig {
                structured: false,
                start: crate::output::tagged::OutputStart::Reasoning,
                tools: vec![json!({"type":"function", "function":{"name":"lookup"}})],
            });
            let mut output = Accumulator::new("request".into());
            output
                .apply(parser.push(Input::Text(&text[..split])).unwrap())
                .unwrap();
            output
                .apply(parser.push(Input::Text(&text[split..])).unwrap())
                .unwrap();
            output
                .apply(parser.finish(Termination::Eos).unwrap())
                .unwrap();
            output.finish().unwrap();
            let message = output.message();
            assert_eq!(message["reasoning_content"], "thought");
            assert_eq!(message["tool_calls"].as_array().unwrap().len(), 2);
            assert_eq!(
                message["tool_calls"][0]["function"]["arguments"],
                "{\"city\": \"東京\"}"
            );
            assert_eq!(
                message["tool_calls"][1]["function"]["arguments"],
                "{\"n\": 3}"
            );
        }
    }
}

#[cfg(test)]
mod deepseek4_syntax_tests {
    use super::*;
    use crate::output::tagged::OutputConfig;
    type DeepSeekOutput = DsmlOutput<false>;
    use crate::output::{Input, ModelOutput, Termination};
    use crate::server::output::Accumulator;
    use serde_json::json;

    #[test]
    fn every_utf8_split_preserves_multiple_calls_and_typed_values() {
        let text = "thought</think><｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"lookup\">\n<｜DSML｜parameter name=\"city\" string=\"true\">東京</｜DSML｜parameter>\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"lookup\"><｜DSML｜parameter name=\"n\" string=\"false\">3</｜DSML｜parameter></｜DSML｜invoke>\n</｜DSML｜tool_calls>";
        for split in text.char_indices().map(|(offset, _)| offset) {
            let mut parser = DeepSeekOutput::new(OutputConfig {
                structured: false,
                start: crate::output::tagged::OutputStart::Reasoning,
                tools: vec![json!({"type":"function", "function":{"name":"lookup"}})],
            });
            let mut output = Accumulator::new("request".into());
            output
                .apply(parser.push(Input::Text(&text[..split])).unwrap())
                .unwrap();
            output
                .apply(parser.push(Input::Text(&text[split..])).unwrap())
                .unwrap();
            output
                .apply(parser.finish(Termination::Eos).unwrap())
                .unwrap();
            output.finish().unwrap();
            let message = output.message();
            assert_eq!(message["reasoning_content"], "thought");
            assert_eq!(message["tool_calls"].as_array().unwrap().len(), 2);
            assert_eq!(
                message["tool_calls"][0]["function"]["arguments"],
                "{\"city\": \"東京\"}"
            );
            assert_eq!(
                message["tool_calls"][1]["function"]["arguments"],
                "{\"n\": 3}"
            );
        }
    }
}

#[cfg(test)]
mod reasoning_close_tests {
    use super::*;
    use crate::output::tagged::OutputConfig;
    use crate::output::{Input, ModelOutput, Termination};
    use crate::server::output::Accumulator;

    /// The reported shape: reasoning, its close delimiter token, a short answer, a
    /// second close delimiter token, and the rest of the answer. The DeepSeek
    /// parser skips a thinking-end token outside reasoning, so no close delimiter
    /// reaches content. A plain-text spelling is plain tokens and stays content.
    fn check<const SPACED: bool>(feed: usize, spelled: bool) {
        let close = (128822, "</think>".to_owned());
        let mut parser = DsmlOutput::<SPACED>::new(OutputConfig {
            structured: false,
            start: OutputStart::Reasoning,
            tools: vec![],
        })
        .with_special_tokens(vec![(128821, "<think>".to_owned()), close.clone()]);
        let mut output = Accumulator::new("close".into());
        let second = if spelled { None } else { Some(close.0) };
        let pieces = [
            (None, "Reason."),
            (Some(close.0), close.1.as_str()),
            (None, "Answer."),
            (second, close.1.as_str()),
            (None, "More."),
        ];
        for (token, text) in pieces {
            if let Some(token) = token {
                output
                    .apply(parser.push(Input::Token(token)).unwrap())
                    .unwrap();
                output
                    .apply(parser.push(Input::Text(text)).unwrap())
                    .unwrap();
            } else if feed == 0 {
                output
                    .apply(parser.push(Input::Text(text)).unwrap())
                    .unwrap();
            } else {
                for character in text.chars() {
                    output
                        .apply(parser.push(Input::Text(&character.to_string())).unwrap())
                        .unwrap();
                }
            }
        }
        output
            .apply(parser.finish(Termination::Eos).unwrap())
            .unwrap();
        output.finish().unwrap();
        let message = output.message();
        assert_eq!(message["reasoning_content"], "Reason.");
        assert_eq!(
            message["content"],
            if spelled { "Answer.</think>More." } else { "Answer.More." }
        );
    }

    #[test]
    fn compact_close_delimiter_tokens_after_reasoning_never_become_content() {
        for feed in 0..2 {
            check::<false>(feed, false);
            check::<false>(feed, true);
        }
    }

    #[test]
    fn spaced_close_delimiter_tokens_after_reasoning_never_become_content() {
        for feed in 0..2 {
            check::<true>(feed, false);
            check::<true>(feed, true);
        }
    }
}

#[cfg(test)]
mod prefixed_tool_tests {
    use super::*;
    use crate::output::tagged::{OutputConfig, OutputStart};
    use crate::output::{Input, ModelOutput, Termination};
    use crate::server::output::Accumulator;
    use serde_json::json;

    fn check<const SPACED: bool>() {
        let separator = if SPACED { " " } else { "" };
        let close = if SPACED {
            "</｜DSML｜ calls>"
        } else {
            "</｜DSML｜tool_calls>"
        };
        let text = format!(
            "\n<｜DSML｜{separator}invoke name=\"lookup\"><｜DSML｜{separator}parameter name=\"city\" string=\"true\">東京</｜DSML｜{separator}parameter></｜DSML｜{separator}invoke>{close}"
        );
        for split in text
            .char_indices()
            .map(|(offset, _)| offset)
            .chain([text.len()])
        {
            let mut parser = DsmlOutput::<SPACED>::new(OutputConfig {
                structured: true,
                start: OutputStart::ToolBody,
                tools: vec![json!({"type":"function", "function":{"name":"lookup"}})],
            });
            let mut output = Accumulator::new("request".into());
            output
                .apply(parser.push(Input::Text(&text[..split])).unwrap())
                .unwrap();
            output
                .apply(parser.push(Input::Text(&text[split..])).unwrap())
                .unwrap();
            output
                .apply(parser.finish(Termination::Eos).unwrap())
                .unwrap();
            output.finish().unwrap();
            let message = output.message();
            assert_eq!(message["tool_calls"][0]["function"]["name"], "lookup");
            assert_eq!(
                message["tool_calls"][0]["function"]["arguments"],
                "{\"city\": \"東京\"}"
            );
            assert!(message.get("reasoning_content").is_none_or(Value::is_null));
        }
    }

    #[test]
    fn compact_tool_body_accepts_every_utf8_split() {
        check::<false>();
    }

    #[test]
    fn spaced_tool_body_accepts_every_utf8_split() {
        check::<true>();
    }
}
