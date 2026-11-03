/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Native token decoding and protocol-neutral output, with a Chat formatter.

use crate::{
    Result,
    frontend::{Frontend, IncrementalDecoder},
    output::{Event, Input, ModelOutput, Termination},
};
use serde_json::{Value, json};

struct Call {
    id: String,
    name: String,
    arguments: String,
    ended: bool,
}

pub(crate) struct Accumulator {
    id: String,
    content: String,
    reasoning: String,
    calls: Vec<Call>,
}
impl Accumulator {
    pub(crate) fn new(id: String) -> Self {
        Self {
            id,
            content: String::new(),
            reasoning: String::new(),
            calls: Vec::new(),
        }
    }
    #[cfg(test)]
    pub(crate) fn has_tools(&self) -> bool {
        !self.calls.is_empty()
    }
    pub(crate) fn apply(&mut self, events: Vec<Event>) -> Result<Vec<Value>> {
        let mut deltas: Vec<Value> = Vec::new();
        for event in events {
            match event {
                Event::TokenLogprobs(_) | Event::ContentLogprobs(_) => {},
                Event::Content(text) => {
                    if !text.is_empty() {
                        self.content.push_str(&text);
                        deltas.push(json!({"content":text}));
                    }
                }
                Event::Reasoning(text) => {
                    if !text.is_empty() {
                        self.reasoning.push_str(&text);
                        deltas.push(json!({"reasoning_content":text}));
                    }
                }
                Event::ToolBegin { index, name, id } => {
                    if index != self.calls.len() || name.is_empty() {
                        return Err(
                            "tool call must begin once with a sequential index and nonempty name"
                                .into(),
                        );
                    }
                    let id = id.unwrap_or_else(|| format!("call_{}_{}", self.id, index));
                    if id.is_empty() || self.calls.iter().any(|call| call.id == id) {
                        return Err("tool call IDs must be nonempty and unique".into());
                    }
                    self.calls.push(Call {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: String::new(),
                        ended: false,
                    });
                    deltas.push(json!({"tool_calls":[{"index":index,"id":id,"type":"function","function":{"name":name,"arguments":""}}]}));
                }
                Event::ToolArguments { index, text } => {
                    let call = self
                        .calls
                        .get_mut(index)
                        .filter(|call| !call.ended)
                        .ok_or("tool arguments require an open call")?;
                    if call.arguments.len().saturating_add(text.len()) > 8 * 1024 * 1024 {
                        return Err("tool arguments exceed the 8 MiB response bound".into());
                    }
                    call.arguments.push_str(&text);
                    if let Some(previous) = deltas
                        .last_mut()
                        .and_then(|v| v.get_mut("tool_calls"))
                        .and_then(Value::as_array_mut)
                        .and_then(|calls| (calls.len() == 1).then(|| &mut calls[0]))
                        .filter(|call| call["index"].as_u64() == Some(index as u64))
                    {
                        let args = previous["function"]["arguments"]
                            .as_str()
                            .unwrap_or("")
                            .to_owned()
                            + &text;
                        previous["function"]["arguments"] = Value::String(args);
                    } else {
                        deltas.push(
                            json!({"tool_calls":[{"index":index,"function":{"arguments":text}}]}),
                        );
                    }
                }
                Event::ToolEnd { index } => {
                    let call = self
                        .calls
                        .get_mut(index)
                        .filter(|call| !call.ended)
                        .ok_or("tool end requires an open call")?;
                    let value: Value = serde_json::from_str(&call.arguments).map_err(|error| {
                        crate::ExecutionError::State(crate::Failure::caused_by(
                            format!("tool arguments are not valid JSON: {error}"),
                            error,
                        ))
                    })?;
                    if !value.is_object() {
                        return Err("tool arguments must encode a JSON object".into());
                    }
                    call.ended = true;
                }
            }
        }
        Ok(deltas)
    }
    pub(crate) fn finish(&self) -> Result<()> {
        if self.calls.iter().any(|call| !call.ended) {
            return Err("incomplete emitted tool call".into());
        }
        Ok(())
    }
    pub(crate) fn message(&self) -> Value {
        let mut message = json!({"role":"assistant","content":if self.content.is_empty(){Value::Null}else{Value::String(self.content.clone())}});
        if !self.reasoning.is_empty() {
            message["reasoning_content"] = Value::String(self.reasoning.clone());
        }
        if !self.calls.is_empty() {
            message["tool_calls"] = Value::Array(self.calls.iter().map(|call| json!({"id":call.id,"type":"function","function":{"name":call.name,"arguments":call.arguments}})).collect());
        }
        message
    }
}

pub(super) struct Decoder<'a, P: ModelOutput> {
    decoder: IncrementalDecoder<'a>,
    tracking: Option<super::token_spans::TokenSpans<'a>>,
    attribute_content: bool,
    next_entries: Vec<super::token_spans::ReturnedLogprob>,
    pending_entries: Vec<super::token_spans::ReturnedLogprob>,
    text_bytes: usize,
    reasoning_tokens: u32,
    unattributed_tokens: u32,

    parser: Option<P>,
    eos: &'a [u32],
    finished: bool,
    worker_decoded: bool,
    local_decoded: bool,
}
impl Decoder<'_, crate::output::OutputPipeline> {
    pub fn reasoning_open(&self) -> bool {
        self.parser
            .as_ref()
            .is_some_and(|parser| parser.reasoning_open())
    }
}

impl<'a, P: ModelOutput> Decoder<'a, P> {
    pub fn new(frontend: &'a Frontend, eos: &'a [u32], config: Option<P::Config>) -> Self {
        Self {
            decoder: frontend.incremental_decoder(false),
            tracking: None,
            attribute_content: false,
            next_entries: vec![],
            pending_entries: vec![],
            text_bytes: 0,
            reasoning_tokens: 0,
            unattributed_tokens: 0,
            parser: config.map(P::new),
            eos,
            finished: false,
            worker_decoded: false,
            local_decoded: false,
        }
    }
    pub fn track_logprobs(&mut self, frontend: &'a Frontend, attribute_content: bool) {
        self.attribute_content = attribute_content;
        self.tracking = Some(super::token_spans::TokenSpans::new(frontend));
    }
    fn entries(&mut self, entries: impl IntoIterator<Item = super::token_spans::ReturnedLogprob>) {
        for entry in entries {
            if self.attribute_content && !entry.silent { self.pending_entries.push(entry.clone()); }
            self.next_entries.push(entry);
        }
    }
    pub fn push_logprob(&mut self, value: crate::execution::TokenLogprobs) -> Result<Vec<Event>> {
        if self.finished || self.worker_decoded { return Err("invalid token decoding state".into()); }
        self.local_decoded = true;
        let token = value.token;
        let is_end_of_turn = self.eos.contains(&token);
        let mut events = self.parse(Input::Token(token))?;
        let tracking = self.tracking.as_mut().ok_or("logprob decoder was not initialized")?;
        if is_end_of_turn {
            if let Some(span) = tracking.silent(token, Some(value)) {
                self.entries(span.tokens.into_iter().filter_map(|t| t.logprob));
                events.extend(self.parse(Input::Text(""))?);
            }
        } else if let Some(span) = tracking.push(token, Some(value))? {
            self.entries(span.tokens.into_iter().filter_map(|t| t.logprob));
            events.extend(self.parse(Input::Text(&span.text))?);
        }
        self.attribute(u32::from(!is_end_of_turn), &events);
        Ok(events)
    }
    pub fn push(&mut self, token: u32) -> Result<Vec<Event>> {
        if self.finished {
            return Err("output presentation is finished".into());
        }
        if self.worker_decoded {
            return Err("cannot mix worker-decoded protocol pieces with token decoding".into());
        }
        self.local_decoded = true;
        let is_end_of_turn = self.eos.contains(&token);
        let mut events = self.parse(Input::Token(token))?;
        if !is_end_of_turn {
            if let Some(tracking) = &mut self.tracking {
                if let Some(span) = tracking.push(token, None)? {
                    self.entries(span.tokens.into_iter().filter_map(|token| token.logprob));
                    events.extend(self.parse(Input::Text(&span.text))?);
                }
            } else if let Some(text) = self.decoder.push(&[token])? {
                events.extend(self.parse(Input::Text(&text))?);
            }
        }
        // The end-of-turn token publishes nothing in any region, so it stays
        // out of the pool that a withheld span later joins.
        self.attribute(u32::from(!is_end_of_turn), &events);
        Ok(events)
    }
    pub fn push_protocol(&mut self, pieces: Vec<crate::output::Piece>) -> Result<Vec<Event>> {
        if self.finished {
            return Err("output presentation is finished".into());
        }
        if self.local_decoded {
            return Err("cannot mix token decoding with worker-decoded protocol pieces".into());
        }
        self.worker_decoded = true;
        let mut events = Vec::new();
        for piece in pieces {
            match piece {
                crate::output::Piece::Logprob(entry) => self.entries([entry]),
                crate::output::Piece::Token(token) => {
                    let parsed = self.parse(Input::Token(token))?;
                    self.attribute(1, &parsed);
                    events.extend(parsed);
                }
                crate::output::Piece::Text(text) => {
                    let parsed = self.parse(Input::Text(&text))?;
                    self.attribute(0, &parsed);
                    events.extend(parsed);
                }
            }
        }
        Ok(events)
    }
    /// Credit generated tokens to the region whose spans the pipeline published.
    ///
    /// A token's decoded text can stay in a parser buffer while the parser is
    /// still ruling out a delimiter, so a token that published nothing joins the
    /// region of the next published span. A span that carries both regions
    /// credits reasoning, which is the region it opens. Tokens that never
    /// publish, such as the end-of-turn token, stay uncredited, and a span
    /// published only from `finish()` credits the pool the same way. Worker-decoded
    /// pieces are credited one piece at a time, so a batch that closes reasoning
    /// and opens content credits only the pieces whose spans published
    /// reasoning.
    fn attribute(&mut self, tokens: u32, events: &[Event]) {
        self.unattributed_tokens += tokens;
        let mut reasoning = false;
        let mut content = false;
        for event in events {
            match event {
                Event::Reasoning(text) if !text.is_empty() => reasoning = true,
                Event::Content(text) if !text.is_empty() => content = true,
                _ => {}
            }
        }
        if reasoning {
            self.reasoning_tokens += self.unattributed_tokens;
            self.unattributed_tokens = 0;
        } else if content {
            self.unattributed_tokens = 0;
        }
    }
    /// Generated tokens the pipeline published as reasoning content.
    pub fn reasoning_tokens(&self) -> u32 {
        self.reasoning_tokens
    }
    pub fn finish(&mut self, cause: Termination) -> Result<Vec<Event>> {
        if self.finished {
            return Err("output presentation is finished".into());
        }
        if matches!(cause, Termination::Cancelled | Termination::ExecutionError) {
            self.abort(cause);
            return Ok(vec![]);
        }
        let final_text = if let Some(tracking) = &mut self.tracking {
            if self.worker_decoded {
                Ok(vec![])
            } else {
                let span = tracking.finish()?;
                self.entries(span.tokens.into_iter().filter_map(|t| t.logprob));
                self.parse(Input::Text(&span.text))
            }
        } else if self.worker_decoded {
            Ok(vec![])
        } else {
            self.decoder
                .finish()
                .and_then(|text| self.parse(Input::Text(&text)))
        };
        let mut events = match final_text {
            Ok(events) => events,
            Err(error) => {
                self.abort(Termination::ExecutionError);
                return Err(error);
            }
        };
        self.finished = true;
        if let Some(parser) = &mut self.parser {
            let final_events = parser.finish(cause)?;
            let spans = if self.attribute_content {
                parser.content_spans()
            } else {
                vec![]
            };
            let consumed = parser.consumed_bytes();
            events.extend(self.decorate(final_events, spans, consumed));
        }
        // A parser may withhold a span until termination and publish it from
        // finish(), including the flush of the incremental decoder. Credit the
        // pool to the region those final events opened, exactly as an
        // in-stream span would; the end-of-turn token is already outside it.
        self.attribute(0, &events);
        Ok(events)
    }
    pub fn abort(&mut self, cause: Termination) {
        debug_assert!(matches!(
            cause,
            Termination::Cancelled | Termination::ExecutionError
        ));
        if !self.finished {
            self.finished = true;
            // Failed or dropped responses cannot publish additional output.
            if let Some(parser) = &mut self.parser {
                let _ = parser.finish(cause);
            }
        }
    }
    fn parse(&mut self, input: Input<'_>) -> Result<Vec<Event>> {
        if self.tracking.is_none() {
            return if let Some(parser) = &mut self.parser {
                parser.push(input)
            } else {
                Ok(match input {
                    Input::Text(text) if !text.is_empty() => vec![Event::Content(text.into())],
                    _ => vec![],
                })
            };
        }
        let start = self.text_bytes;
        if let Input::Text(text) = input {
            self.text_bytes += text.len();
        }
        let (events, spans, consumed) = if let Some(parser) = &mut self.parser {
            let events = parser.push(input)?;
            let spans = if self.attribute_content {
                parser.content_spans()
            } else {
                vec![]
            };
            (events, spans, parser.consumed_bytes())
        } else {
            let events = match input {
                Input::Text(text) if !text.is_empty() => vec![Event::Content(text.into())],
                _ => vec![],
            };
            (events, vec![Some(start..self.text_bytes)], self.text_bytes)
        };
        Ok(self.decorate(events, spans, consumed))
    }
    fn decorate(
        &mut self,
        events: Vec<Event>,
        spans: Vec<Option<std::ops::Range<usize>>>,
        consumed: usize,
    ) -> Vec<Event> {
        if self.tracking.is_none() {
            return events;
        }
        let mut result = Vec::new();
        if !self.next_entries.is_empty() {
            result.push(Event::TokenLogprobs(std::mem::take(&mut self.next_entries)));
        }
        let mut spans = spans.into_iter();
        for event in events {
            if self.attribute_content && matches!(event, Event::Content(_)) {
                let range = spans.next().flatten();
                let mut entries = Vec::new();
                self.pending_entries.retain(|entry| {
                    if range.as_ref().is_some_and(|r| r.contains(&entry.start)) {
                        entries.push(entry.clone());
                        false
                    } else {
                        true
                    }
                });
                result.push(Event::ContentLogprobs(entries));
            }
            result.push(event);
        }
        self.pending_entries.retain(|entry| entry.start >= consumed);
        result
    }

}

impl<P: ModelOutput> Drop for Decoder<'_, P> {
    fn drop(&mut self) {
        self.abort(Termination::Cancelled);
    }
}

#[cfg(test)]
struct Presentation<'a, P: ModelOutput> {
    decoder: Decoder<'a, P>,
    output: Accumulator,
}
#[cfg(test)]
impl<'a, P: ModelOutput> Presentation<'a, P> {
    fn new(frontend: &'a Frontend, eos: &'a [u32], config: P::Config, id: String) -> Self {
        Self {
            decoder: Decoder::new(frontend, eos, Some(config)),
            output: Accumulator::new(id),
        }
    }
    fn push(&mut self, token: u32) -> Result<Vec<Value>> {
        self.output.apply(self.decoder.push(token)?)
    }
    fn push_protocol(&mut self, pieces: Vec<crate::output::Piece>) -> Result<Vec<Value>> {
        self.output.apply(self.decoder.push_protocol(pieces)?)
    }
    fn finish(&mut self, cause: Termination) -> Result<Vec<Value>> {
        let deltas = self.output.apply(self.decoder.finish(cause)?)?;
        self.output.finish()?;
        Ok(deltas)
    }
    fn abort(&mut self, cause: Termination) {
        self.decoder.abort(cause)
    }
    fn message(&self) -> Value {
        self.output.message()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{
        frontend::{Asset, FrontendAssets},
        output::chatml::{ChatMlOutput, OutputConfig},
        output::{Progress, ReasoningParser, ToolCallParser},
    };
    use sha2::{Digest, Sha256};
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    };

    pub(crate) fn frontend(pieces: &[&str]) -> Frontend {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "flare-output-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        let vocab: tokenizers::models::bpe::Vocab = pieces
            .iter()
            .enumerate()
            .map(|(i, s)| ((*s).to_owned(), i as u32))
            .collect();
        let model = tokenizers::models::bpe::BPE::builder()
            .vocab_and_merges(vocab, vec![])
            .build()
            .unwrap();
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer.with_decoder(Some(tokenizers::decoders::sequence::Sequence::new(vec![
            tokenizers::decoders::byte_fallback::ByteFallback::default().into(),
            tokenizers::decoders::fuse::Fuse::default().into(),
        ])));
        let asset = |name: &str, bytes: &[u8]| {
            let path = root.join(name);
            std::fs::write(&path, bytes).unwrap();
            Asset {
                range: None,
                path,
                sha256: format!("{:x}", Sha256::digest(bytes)),
            }
        };
        let result = Frontend::load(
            FrontendAssets {
                tokenizer: asset(
                    "tokenizer.json",
                    tokenizer.to_string(false).unwrap().as_bytes(),
                ),
                tokens: Default::default(),
                chat_template: Some(asset("chat_template.jinja", b"{% mark %}{{ messages[0].content }}")),
            },
            Default::default(),
        )
        .unwrap();
        std::fs::remove_dir_all(root).unwrap();
        result
    }

    fn reconstruct(deltas: &[Value]) -> Value {
        let mut content = String::new();
        let mut reasoning = String::new();
        let mut calls: Vec<Value> = vec![];
        for delta in deltas {
            if let Some(text) = delta["content"].as_str() {
                content.push_str(text);
            }
            if let Some(text) = delta["reasoning_content"].as_str() {
                reasoning.push_str(text);
            }
            for call in delta["tool_calls"].as_array().into_iter().flatten() {
                let index = call["index"].as_u64().unwrap() as usize;
                if index == calls.len() {
                    calls.push(json!({"id":call["id"],"type":call["type"],"function":{"name":call["function"]["name"],"arguments":""}}));
                }
                if let Some(args) = call["function"]["arguments"].as_str() {
                    let prior = calls[index]["function"]["arguments"]
                        .as_str()
                        .unwrap()
                        .to_owned();
                    calls[index]["function"]["arguments"] = json!(prior + args);
                }
            }
        }
        let mut result = json!({"role":"assistant","content":if content.is_empty(){Value::Null}else{json!(content)}});
        if !reasoning.is_empty() {
            result["reasoning_content"] = json!(reasoning);
        }
        if !calls.is_empty() {
            result["tool_calls"] = json!(calls);
        }
        result
    }

    #[test]
    fn json_and_stream_reconstruction_share_incremental_reasoning_and_multiple_calls() {
        let pieces = [
            "<thi",
            "nk>\nwhy\n</th",
            "ink>\n\nanswer\n\n<tool_",
            "call><function=lookup><parameter=query>",
            "quoted \"é\" <think>literal</think>",
            "</parameter><parameter=data>{\"a\":[1,true,null]}</parameter></function></tool_call>",
            "\n<tool_call><function=lookup><parameter=query>second</parameter></function></tool_call>",
        ];
        let frontend = frontend(&pieces);
        let config = OutputConfig {
            start: crate::output::tagged::OutputStart::Content,
            structured: false,
            tools: vec![
                json!({"function":{"name":"lookup","parameters":{"properties":{"query":{"type":"string"},"data":{"type":"object"}}}}}),
            ],
        };
        let mut streamed =
            Presentation::<ChatMlOutput>::new(&frontend, &[99], config.clone(), "test".into());
        let mut collected =
            Presentation::<ChatMlOutput>::new(&frontend, &[99], config, "test".into());
        let mut deltas = Vec::new();
        for token in (0..pieces.len() as u32).chain([99]) {
            deltas.extend(streamed.push(token).unwrap());
            collected.push(token).unwrap();
        }
        deltas.extend(streamed.finish(Termination::Eos).unwrap());
        collected.finish(Termination::Eos).unwrap();
        assert_eq!(reconstruct(&deltas), collected.message());
        assert_eq!(streamed.message(), collected.message());
        assert_eq!(streamed.message()["reasoning_content"], "why");
        assert_eq!(streamed.message()["content"], "answer");
        assert_eq!(
            streamed.message()["tool_calls"].as_array().unwrap().len(),
            2
        );
        assert_eq!(
            deltas
                .iter()
                .filter(|delta| delta.get("tool_calls").is_some())
                .count(),
            2
        );
        let args: Value = serde_json::from_str(
            streamed.message()["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            args,
            json!({"query":"quoted \"é\" <think>literal</think>","data":{"a":[1,true,null]}})
        );
    }

    #[test]
    fn append_only_argument_fragments_are_rendered_without_reencoding_or_retraction() {
        let mut output = Accumulator::new("x".into());
        let mut deltas = output
            .apply(vec![Event::ToolBegin {
                index: 0,
                name: "synthetic".into(),
                id: Some("model-id".into()),
            }])
            .unwrap();
        for text in ["{\"value\":\"a\\", "\"b\",\"nested\":", "[true,null]}"] {
            deltas.extend(
                output
                    .apply(vec![Event::ToolArguments {
                        index: 0,
                        text: text.into(),
                    }])
                    .unwrap(),
            );
        }
        assert!(output.finish().is_err());
        output.apply(vec![Event::ToolEnd { index: 0 }]).unwrap();
        output.finish().unwrap();
        assert_eq!(reconstruct(&deltas), output.message());
        assert_eq!(output.message()["tool_calls"][0]["id"], "model-id");
        assert!(
            output
                .apply(vec![Event::ToolArguments {
                    index: 0,
                    text: "extra".into()
                }])
                .is_err()
        );
        assert!(
            output
                .apply(vec![Event::ToolBegin {
                    index: 1,
                    name: "x".into(),
                    id: Some("model-id".into())
                }])
                .is_err()
        );
    }

    #[test]
    fn tool_end_requires_object_arguments_and_open_call() {
        for text in ["[1]", "null", "{broken"] {
            let mut output = Accumulator::new("x".into());
            output
                .apply(vec![
                    Event::ToolBegin {
                        index: 0,
                        name: "x".into(),
                        id: None,
                    },
                    Event::ToolArguments {
                        index: 0,
                        text: text.into(),
                    },
                ])
                .unwrap();
            assert!(output.apply(vec![Event::ToolEnd { index: 0 }]).is_err());
        }
        assert!(
            Accumulator::new("x".into())
                .apply(vec![Event::ToolEnd { index: 0 }])
                .is_err()
        );
    }

    struct ControlReasoning;
    impl ReasoningParser for ControlReasoning {
        fn push(&mut self, input: Input<'_>) -> Result<Progress> {
            Ok(Progress {
                events: if matches!(input, Input::Token(0)) {
                    vec![Event::Reasoning("control".into())]
                } else {
                    vec![]
                },
                remainder: None,
            })
        }
        fn finish(&mut self, _: Termination) -> Result<Vec<Event>> {
            Ok(vec![])
        }
    }
    struct ControlTools;
    impl ToolCallParser for ControlTools {
        fn push(&mut self, input: Input<'_>) -> Result<Progress> {
            Ok(Progress {
                events: if matches!(input, Input::Token(98)) {
                    vec![
                        Event::ToolBegin {
                            index: 0,
                            name: "partial".into(),
                            id: None,
                        },
                        Event::ToolArguments {
                            index: 0,
                            text: "{\"value\":".into(),
                        },
                    ]
                } else if matches!(input, Input::Token(99)) {
                    vec![
                        Event::ToolBegin {
                            index: 0,
                            name: "control-tool".into(),
                            id: None,
                        },
                        Event::ToolArguments {
                            index: 0,
                            text: "{}".into(),
                        },
                        Event::ToolEnd { index: 0 },
                    ]
                } else {
                    vec![]
                },
                remainder: None,
            })
        }
        fn finish(&mut self, _: Termination) -> Result<Vec<Event>> {
            Ok(vec![])
        }
    }
    struct Controls {
        content: bool,
        observed: Arc<Mutex<Vec<String>>>,
        reasoning: ControlReasoning,
        tools: ControlTools,
    }
    impl ModelOutput for Controls {
        fn content_spans(&self) -> Vec<Option<std::ops::Range<usize>>> {
            if self.content {
                vec![None]
            } else {
                vec![]
            }
        }
        fn consumed_bytes(&self) -> usize {
            0
        }
        type Config = Arc<Mutex<Vec<String>>>;
        fn new(observed: Self::Config) -> Self {
            Self {
                content: false,
                observed,
                reasoning: ControlReasoning,
                tools: ControlTools,
            }
        }
        fn push(&mut self, input: Input<'_>) -> Result<Vec<Event>> {
            self.content = false;
            self.observed.lock().unwrap().push(format!("{input:?}"));
            let mut events = self.tools.push(input)?.events;
            events.extend(self.reasoning.push(input)?.events);
            if let Input::Text(text) = input {
                if !text.is_empty() {
                    self.content = true;
                    events.push(Event::Content(text.into()));
                }
            }
            Ok(events)
        }
        fn finish(&mut self, cause: Termination) -> Result<Vec<Event>> {
            self.content = false;
            self.observed
                .lock()
                .unwrap()
                .push(format!("finish:{cause:?}"));
            Ok(vec![])
        }
    }

    /// Publishes each decoded span as reasoning or content by its leading
    /// character, and withholds a span that could still be a delimiter the
    /// way a real parser does while it rules one out.
    struct Regions {
        held: String,
    }
    impl ModelOutput for Regions {
        type Config = ();
        fn new(_: Self::Config) -> Self {
            Self { held: String::new() }
        }
        fn content_spans(&self) -> Vec<Option<std::ops::Range<usize>>> {
            vec![]
        }
        fn consumed_bytes(&self) -> usize {
            0
        }
        fn push(&mut self, input: Input<'_>) -> Result<Vec<Event>> {
            let Input::Text(text) = input else {
                return Ok(vec![]);
            };
            let span = std::mem::take(&mut self.held) + text;
            if span.is_empty() {
                return Ok(vec![]);
            }
            if span.ends_with('?') {
                self.held = span;
                return Ok(vec![]);
            }
            Ok(if span.starts_with('R') {
                vec![Event::Reasoning(span)]
            } else {
                vec![Event::Content(span)]
            })
        }
        fn finish(&mut self, _: Termination) -> Result<Vec<Event>> {
            Ok(vec![])
        }
    }

    #[test]
    fn reasoning_count_credits_tokens_to_the_region_their_spans_publish_in() {
        let frontend = frontend(&["Ra", "R?", "x", "Ca"]);
        let mut decoder = Decoder::<Regions>::new(&frontend, &[99], Some(()));
        assert_eq!(decoder.reasoning_tokens(), 0, "no token has published yet");
        for token in [0, 1, 2, 3] {
            decoder.push(token).unwrap();
        }
        // "Ra" and "x" publish as reasoning; the withheld "R?" token joins the
        // span its text completed, so it credits reasoning too. "Ca" is content.
        assert_eq!(decoder.reasoning_tokens(), 3);
        decoder.push(99).unwrap();
        decoder.finish(Termination::Eos).unwrap();
        assert_eq!(decoder.reasoning_tokens(), 3, "the end-of-turn token publishes nothing");
    }

    #[test]
    fn protocol_pieces_batch_splits_attribution_between_regions() {
        use crate::output::Piece;
        let frontend = frontend(&["Ra", "R?", "x", "Ca"]);
        let mut decoder = Decoder::<Regions>::new(&frontend, &[99], Some(()));
        // One batch that closes reasoning and immediately opens content. Batch-
        // level attribution would credit the whole batch to reasoning because
        // the batch carries a reasoning span; per-piece attribution follows the
        // region each piece's spans published in.
        decoder
            .push_protocol(vec![
                Piece::Token(0),
                Piece::Text("Ra".to_string()),
                Piece::Token(1),
                Piece::Text("R?".to_string()),
                Piece::Token(2),
                Piece::Text("x".to_string()),
                Piece::Token(3),
                Piece::Text("Ca".to_string()),
            ])
            .unwrap();
        assert_eq!(
            decoder.reasoning_tokens(),
            3,
            "the content piece's token credits content, not reasoning"
        );
        decoder.push_protocol(vec![Piece::Token(99)]).unwrap();
        decoder.finish(Termination::Eos).unwrap();
        assert_eq!(decoder.reasoning_tokens(), 3, "the unpublished end-of-turn token stays uncredited");
    }

    /// Withholds every span until termination, publishing them together as
    /// reasoning the way a real parser flushes a buffer it still considered a
    /// possible delimiter.
    struct HoldUntilFinish {
        held: String,
    }
    impl ModelOutput for HoldUntilFinish {
        type Config = ();
        fn new(_: Self::Config) -> Self {
            Self { held: String::new() }
        }
        fn content_spans(&self) -> Vec<Option<std::ops::Range<usize>>> {
            vec![]
        }
        fn consumed_bytes(&self) -> usize {
            0
        }
        fn push(&mut self, input: Input<'_>) -> Result<Vec<Event>> {
            if let Input::Text(text) = input {
                self.held.push_str(text);
            }
            Ok(vec![])
        }
        fn finish(&mut self, _: Termination) -> Result<Vec<Event>> {
            let held = std::mem::take(&mut self.held);
            if held.is_empty() {
                Ok(vec![])
            } else {
                Ok(vec![Event::Reasoning(held)])
            }
        }
    }

    #[test]
    fn reasoning_published_only_at_termination_is_credited() {
        let frontend = frontend(&["Th", "ink"]);
        let mut decoder = Decoder::<HoldUntilFinish>::new(&frontend, &[99], Some(()));
        for token in [0, 1] {
            decoder.push(token).unwrap();
        }
        assert_eq!(
            decoder.reasoning_tokens(),
            0,
            "the parser withholds every span until termination"
        );
        let events = decoder.finish(Termination::Eos).unwrap();
        assert!(
            matches!(events.as_slice(), [Event::Reasoning(text)] if text == "Think"),
            "the terminal flush publishes the withheld reasoning"
        );
        assert_eq!(
            decoder.reasoning_tokens(),
            2,
            "tokens withheld until the terminal flush still credit reasoning"
        );
    }

    #[test]
    fn end_of_turn_token_never_joins_a_reasoning_flush() {
        let frontend = frontend(&["Th", "ink"]);
        let mut decoder = Decoder::<HoldUntilFinish>::new(&frontend, &[99], Some(()));
        for token in [0, 1, 99] {
            decoder.push(token).unwrap();
        }
        let events = decoder.finish(Termination::Eos).unwrap();
        assert!(
            matches!(events.as_slice(), [Event::Reasoning(_)]),
            "the terminal flush publishes reasoning"
        );
        assert_eq!(
            decoder.reasoning_tokens(),
            2,
            "the end-of-turn token stays outside every region even when a terminal flush credits the pool"
        );
    }

    /// Publishes reasoning and content from a single span, as a parser does
    /// when one withheld buffer turns out to close reasoning and open content.
    struct BothRegions;
    impl ModelOutput for BothRegions {
        type Config = ();
        fn new(_: Self::Config) -> Self {
            Self
        }
        fn content_spans(&self) -> Vec<Option<std::ops::Range<usize>>> {
            vec![]
        }
        fn consumed_bytes(&self) -> usize {
            0
        }
        fn push(&mut self, input: Input<'_>) -> Result<Vec<Event>> {
            match input {
                Input::Text(text) if text == "B" => Ok(vec![
                    Event::Reasoning("R".into()),
                    Event::Content("C".into()),
                ]),
                Input::Text(_) | Input::Token(_) => Ok(vec![]),
            }
        }
        fn finish(&mut self, _: Termination) -> Result<Vec<Event>> {
            Ok(vec![])
        }
    }

    #[test]
    fn a_span_publishing_both_regions_credits_reasoning() {
        let frontend = frontend(&["B"]);
        let mut decoder = Decoder::<BothRegions>::new(&frontend, &[99], Some(()));
        decoder.push(0).unwrap();
        assert_eq!(
            decoder.reasoning_tokens(),
            1,
            "a span that carries reasoning and content alike is credited to the region it opens"
        );
    }

    #[test]
    fn real_parser_terminal_flush_credits_every_generated_token_to_reasoning() {
        use crate::output::tagged::OutputStart;
        let frontend = frontend(&["thin", "k<|im_", "\n"]);
        let mut decoder = Decoder::<ChatMlOutput>::new(
            &frontend,
            &[99],
            Some(OutputConfig {
                structured: false,
                start: OutputStart::Reasoning,
                tools: vec![],
            }),
        );
        for token in [0, 1, 2] {
            decoder.push(token).unwrap();
        }
        let events = decoder.finish(Termination::OutputLimit).unwrap();
        let published: String = events
            .iter()
            .filter_map(|event| match event {
                Event::Reasoning(text) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            decoder.reasoning_tokens(),
            3,
            "tokens withheld while a delimiter is ruled out join reasoning at the terminal flush; published reasoning: {published:?}"
        );
    }

    #[test]
    fn token_only_controls_and_split_utf8_reach_model_protocol_without_qwen_syntax() {
        let frontend = frontend(&["<0xE4>", "<0xB8>", "<0x96>"]);
        let observed = Arc::new(Mutex::new(Vec::new()));
        let mut presentation =
            Presentation::<Controls>::new(&frontend, &[99], observed.clone(), "control".into());
        for token in [0, 1, 2, 99] {
            presentation.push(token).unwrap();
        }
        presentation.finish(Termination::Eos).unwrap();
        assert_eq!(
            *observed.lock().unwrap(),
            vec![
                "Token(0)",
                "Token(1)",
                "Token(2)",
                "Text(\"世\")",
                "Token(99)",
                "Text(\"\")",
                "finish:Eos"
            ]
        );
        assert_eq!(presentation.message()["content"], "世");
        assert_eq!(presentation.message()["reasoning_content"], "control");
        assert_eq!(
            presentation.message()["tool_calls"][0]["function"]["name"],
            "control-tool"
        );
        assert!(presentation.push(0).is_err());
        assert!(presentation.finish(Termination::Eos).is_err());
    }

    #[test]
    fn aborts_deliver_exactly_one_cause_without_flushing_incomplete_utf8() {
        let frontend = frontend(&["<0xE4>"]);
        for cause in [Termination::Cancelled, Termination::ExecutionError] {
            let observed = Arc::new(Mutex::new(Vec::new()));
            {
                let mut presentation = Presentation::<Controls>::new(
                    &frontend,
                    &[99],
                    observed.clone(),
                    "abort".into(),
                );
                presentation.push(0).unwrap();
                assert!(presentation.finish(cause).unwrap().is_empty());
                presentation.abort(cause);
            }
            assert_eq!(
                *observed.lock().unwrap(),
                vec!["Token(0)".to_owned(), format!("finish:{cause:?}")]
            );
        }
        let observed = Arc::new(Mutex::new(Vec::new()));
        {
            let mut presentation =
                Presentation::<Controls>::new(&frontend, &[99], observed.clone(), "dropped".into());
            presentation.push(0).unwrap();
        }
        assert_eq!(
            *observed.lock().unwrap(),
            vec!["Token(0)", "finish:Cancelled"]
        );
    }
    #[test]
    fn string_stop_preserves_complete_calls_and_json_stream_equality() {
        let pieces = [
            "<think>reason</think>\n\nanswer\n\n<tool_call><function=lookup><parameter=query>first</parameter></function></tool_call>",
            "\n<tool_call><function=lookup><parameter=query>partialST",
            "OPdiscard</parameter></function></tool_call>",
        ];
        let frontend = frontend(&pieces);
        let config = OutputConfig {
            start: crate::output::tagged::OutputStart::Content,
            structured: false,
            tools: vec![
                json!({"function":{"name":"lookup","parameters":{"properties":{"query":{"type":"string"}}}}}),
            ],
        };
        let mut streamed =
            Presentation::<ChatMlOutput>::new(&frontend, &[99], config.clone(), "stop".into());
        let mut collected =
            Presentation::<ChatMlOutput>::new(&frontend, &[99], config.clone(), "stop".into());
        let mut filter =
            super::super::stops::TextStops::new(&frontend, vec!["STOP".into()]).unwrap();
        let mut deltas = vec![];
        for token in 0..3 {
            let (visible, matched) = filter.push(token).unwrap();
            deltas.extend(streamed.push_protocol(visible.clone()).unwrap());
            collected.push_protocol(visible).unwrap();
            assert_eq!(matched, token == 2);
        }
        deltas.extend(streamed.finish(Termination::UserStop).unwrap());
        collected.finish(Termination::UserStop).unwrap();
        assert_eq!(reconstruct(&deltas), collected.message());
        assert_eq!(collected.message()["content"], "answer");
        assert_eq!(collected.message()["reasoning_content"], "reason");
        assert_eq!(
            collected.message()["tool_calls"].as_array().unwrap().len(),
            1
        );
        assert_eq!(
            collected.message()["tool_calls"][0]["function"]["arguments"],
            "{\"query\":\"first\"}"
        );
        for cause in [Termination::Eos, Termination::OutputLimit] {
            let mut presentation = Presentation::<ChatMlOutput>::new(
                &frontend,
                &[99],
                config.clone(),
                "invalid".into(),
            );
            presentation
                .push_protocol(vec![crate::output::Piece::Text(
                    "<tool_call><function=lookup>".into(),
                )])
                .unwrap();
            if cause == Termination::Eos {
                assert!(presentation.finish(cause).is_err());
            } else {
                assert!(presentation.finish(cause).unwrap().is_empty());
                assert!(presentation.message().get("tool_calls").is_none());
            }
        }
    }

    #[test]
    fn worker_decoded_spans_do_not_reorder_a_later_control_before_held_text() {
        let frontend = frontend(&["beforeST", "<control>"]);
        let observed = Arc::new(Mutex::new(Vec::new()));
        let mut presentation =
            Presentation::<Controls>::new(&frontend, &[99], observed.clone(), "ordered".into());
        let mut stops =
            super::super::stops::TextStops::new(&frontend, vec!["STOP".into()]).unwrap();
        for token in [0, 1] {
            let (pieces, matched) = stops.push(token).unwrap();
            assert!(!matched);
            presentation.push_protocol(pieces).unwrap();
        }
        let (pieces, matched) = stops.finish(Some(99)).unwrap();
        assert!(!matched);
        presentation.push_protocol(pieces).unwrap();
        presentation.finish(Termination::Eos).unwrap();
        assert_eq!(
            *observed.lock().unwrap(),
            vec![
                "Token(0)",
                "Text(\"beforeST\")",
                "Token(1)",
                "Text(\"<control>\")",
                "Token(99)",
                "finish:Eos"
            ]
        );
    }
    #[test]
    fn user_stop_cannot_retract_an_already_emitted_partial_tool_call() {
        let frontend = frontend(&["x"]);
        let observed = Arc::new(Mutex::new(Vec::new()));
        let mut presentation =
            Presentation::<Controls>::new(&frontend, &[98], observed, "partial".into());
        let deltas = presentation.push(98).unwrap();
        assert_eq!(deltas[0]["tool_calls"][0]["function"]["name"], "partial");
        assert_eq!(
            presentation
                .finish(Termination::UserStop)
                .unwrap_err()
                .to_string(),
            "incomplete emitted tool call"
        );
    }
    #[test]
    fn decoding_sources_cannot_be_mixed_in_either_order() {
        let frontend = frontend(&["x"]);
        for worker_first in [false, true] {
            let config = OutputConfig {
                start: crate::output::tagged::OutputStart::Content,
                structured: false,
                tools: vec![],
            };
            let mut presentation =
                Presentation::<ChatMlOutput>::new(&frontend, &[99], config, "mixed".into());
            if worker_first {
                presentation
                    .push_protocol(vec![crate::output::Piece::Text("x".into())])
                    .unwrap();
                assert!(presentation.push(0).is_err());
            } else {
                presentation.push(0).unwrap();
                assert!(
                    presentation
                        .push_protocol(vec![crate::output::Piece::Text("x".into())])
                        .is_err()
                );
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Block {
    Text(String),
    Reasoning(String),
    Tool {
        index: usize,
        id: String,
        name: String,
        arguments: String,
        ended: bool,
    },
}
#[derive(Clone, Debug)]
pub(super) enum Change {
    Start(usize),
    Delta(usize, String),
    End(usize),
}
#[derive(Default)]
pub(super) struct ResponseOutput {
    pub blocks: Vec<Block>,
    open: Option<usize>,
    calls: Vec<usize>,
}
impl ResponseOutput {
    pub fn has_tools(&self) -> bool {
        !self.calls.is_empty()
    }
    pub fn text(&self) -> String {
        self.blocks
            .iter()
            .filter_map(|b| {
                if let Block::Text(text) = b {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect()
    }
    fn close(&mut self, changes: &mut Vec<Change>) -> Result<()> {
        if let Some(index) = self.open.take() {
            if let Block::Tool { ended: false, .. } = &self.blocks[index] {
                return Err("incomplete emitted tool call".into());
            }
            changes.push(Change::End(index));
        }
        Ok(())
    }
    pub fn finish(&mut self) -> Result<Vec<Change>> {
        let mut changes = vec![];
        self.close(&mut changes)?;
        Ok(changes)
    }
    pub fn apply(&mut self, event: Event, request_id: &str) -> Result<Vec<Change>> {
        let mut changes = vec![];
        match event {
            Event::TokenLogprobs(_) | Event::ContentLogprobs(_) => {},
            Event::Content(text) | Event::Reasoning(text) if text.is_empty() => {}
            event @ (Event::Content(_) | Event::Reasoning(_)) => {
                let reasoning = matches!(event, Event::Reasoning(_));
                let text = match event {
                    Event::Content(t) | Event::Reasoning(t) => t,
                    _ => unreachable!(),
                };
                let same = self.open.is_some_and(|i| {
                    matches!(
                        (&self.blocks[i], reasoning),
                        (Block::Text(_), false) | (Block::Reasoning(_), true)
                    )
                });
                if !same {
                    self.close(&mut changes)?;
                    let i = self.blocks.len();
                    self.blocks.push(if reasoning {
                        Block::Reasoning(String::new())
                    } else {
                        Block::Text(String::new())
                    });
                    self.open = Some(i);
                    changes.push(Change::Start(i));
                }
                let i = self.open.unwrap();
                match &mut self.blocks[i] {
                    Block::Text(t) | Block::Reasoning(t) => t.push_str(&text),
                    _ => unreachable!(),
                }
                changes.push(Change::Delta(i, text));
            }
            Event::ToolBegin { index, name, id } => {
                if index != self.calls.len() || name.is_empty() {
                    return Err("tool calls require sequential indices and names".into());
                }
                self.close(&mut changes)?;
                let id = id.unwrap_or_else(|| format!("call_{request_id}_{index}"));
                if id.is_empty()
                    || self
                        .blocks
                        .iter()
                        .any(|b| matches!(b,Block::Tool{id:existing,..} if existing==&id))
                {
                    return Err("tool IDs must be unique and nonempty".into());
                }
                let i = self.blocks.len();
                self.blocks.push(Block::Tool {
                    index,
                    id,
                    name,
                    arguments: String::new(),
                    ended: false,
                });
                self.calls.push(i);
                self.open = Some(i);
                changes.push(Change::Start(i));
            }
            Event::ToolArguments { index, text } => {
                let i = *self
                    .calls
                    .get(index)
                    .ok_or("tool arguments require an open call")?;
                let Block::Tool {
                    arguments,
                    ended: false,
                    ..
                } = &mut self.blocks[i]
                else {
                    return Err("tool arguments require an open call".into());
                };
                if arguments.len().saturating_add(text.len()) > 8 * 1024 * 1024 {
                    return Err("tool arguments exceed the 8 MiB response bound".into());
                }
                arguments.push_str(&text);
                changes.push(Change::Delta(i, text));
            }
            Event::ToolEnd { index } => {
                let i = *self
                    .calls
                    .get(index)
                    .ok_or("tool end requires an open call")?;
                let Block::Tool {
                    arguments, ended, ..
                } = &mut self.blocks[i]
                else {
                    unreachable!()
                };
                if *ended {
                    return Err("tool already ended".into());
                }
                let value: Value = serde_json::from_str(arguments)
                    .map_err(|e| format!("invalid tool arguments: {e}"))?;
                if !value.is_object() {
                    return Err("tool arguments must encode a JSON object".into());
                }
                *ended = true;
                self.close(&mut changes)?;
            }
        }
        Ok(changes)
    }
}

#[cfg(test)]
mod logprob_retention_tests {
    use super::*;
    use crate::{execution::TokenLogprobs, output::{OutputConfig, OutputPipeline, tagged::{OutputConfig as Tags, OutputStart}}};
    #[test]
    fn forced_reasoning_close_preserves_decoding_and_sampled_logprob_offsets() {
        let frontend = super::tests::frontend(&["plan", "</think>", "answer", "tail"]);
        let mut syntax = crate::model::testing::metadata().frontend;
        syntax.reasoning = Some(crate::model::contract::ReasoningParser {
            kind: crate::model::contract::ReasoningParserKind::ThinkTags,
            open: "<think>".into(),
            close: "</think>".into(),
            wrapping_newlines: 0,
            terminal_markers: vec![],
        });
        let config = OutputConfig {
            syntax,
            control_tokens: vec![(1, "</think>".into())],
            content: Tags {
                structured: false,
                start: OutputStart::Reasoning,
                tools: vec![],
            },
        };
        let mut decoder = Decoder::<OutputPipeline>::new(&frontend, &[], Some(config));
        decoder.track_logprobs(&frontend, true);
        let mut events = decoder
            .push_logprob(TokenLogprobs {
                token: 0,
                logprob: -1.,
                top: vec![],
            })
            .unwrap();
        events.extend(decoder.push(1).unwrap());
        for token in [2, 3] {
            events.extend(
                decoder
                    .push_logprob(TokenLogprobs {
                        token,
                        logprob: -1.,
                        top: vec![],
                    })
                    .unwrap(),
            );
        }
        events.extend(decoder.finish(Termination::OutputLimit).unwrap());
        let (mut reasoning, mut content, mut entries) = (String::new(), String::new(), vec![]);
        for event in &events {
            match event {
                Event::Reasoning(text) => reasoning.push_str(text),
                Event::Content(text) => content.push_str(text),
                Event::TokenLogprobs(values) => entries.extend(values),
                _ => {}
            }
        }
        assert_eq!(reasoning, "plan");
        assert_eq!(content, "answertail");
        assert_eq!(
            entries
                .iter()
                .map(|entry| entry.value.token)
                .collect::<Vec<_>>(),
            vec![0, 2, 3]
        );
        assert_eq!(
            entries.iter().map(|entry| entry.start).collect::<Vec<_>>(),
            vec![0, 12, 18]
        );
    }

    #[test]
    fn responses_discard_unfinished_tool_body_entries_as_the_body_arrives() {
        let frontend = super::tests::frontend(&["<tool_call>", "<function=lookup>", "<parameter=q>", "xxxxxxxx"]);
        let mut syntax = crate::model::testing::metadata().frontend;
        syntax.tools = Some(crate::model::contract::ToolParser {
            kind: crate::model::contract::ToolParserKind::QwenCoder,
            wrapping_newlines: 0, tag_separator: String::new(),
        });
        let config = OutputConfig { syntax, control_tokens: vec![], content: Tags {
            structured: false, start: OutputStart::Content,
            tools: vec![json!({"type":"function","function":{"name":"lookup"}})],
        } };
        let mut decoder = Decoder::<OutputPipeline>::new(&frontend, &[], Some(config));
        decoder.track_logprobs(&frontend, true);
        for (index, token) in [vec![0,1,2],vec![3;1024]].concat().into_iter().enumerate() {
            decoder.push_logprob(TokenLogprobs { token, logprob: -1., top: vec![] }).unwrap();
            // The opener and header stay pending until the parameter tag confirms the call.
            assert_eq!(decoder.pending_entries.is_empty(), index >= 2);
        }
        assert!(decoder.finish(Termination::OutputLimit).unwrap().is_empty());
    }
}
