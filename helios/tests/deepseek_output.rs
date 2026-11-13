/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#![cfg(feature = "server")]

use deepseek_recipe::stream::{
    ChunkGenerator, InferenceChunk, InferenceFinishReason, OutputChunk, ParsingOptions,
    StreamProcessor, ThinkingInitialStage,
};
use flare_helios::{
    model::PreparedContract,
    output::{Event, Input, OutputConfig, OutputPipeline, Termination, tagged::OutputStart},
};
use futures_util::StreamExt;
use serde_json::{Value, json};

#[derive(Debug, Default, PartialEq)]
struct Parsed {
    content: String,
    reasoning: String,
    calls: Vec<(String, String)>,
    ended: usize,
}

impl Parsed {
    fn event(&mut self, event: Event) {
        match event {
            Event::TokenLogprobs(_) | Event::ContentLogprobs(_) => {
                panic!("unexpected logprobs from syntax parser")
            }
            Event::Content(text) => self.content.push_str(&text),
            Event::Reasoning(text) => self.reasoning.push_str(&text),
            Event::ToolBegin { index, name, .. } => {
                assert_eq!(index, self.calls.len());
                self.calls.push((name, String::new()));
            }
            Event::ToolArguments { index, text } => self.calls[index].1.push_str(&text),
            Event::ToolEnd { index } => {
                assert_eq!(index, self.ended);
                self.ended += 1;
            }
        }
    }

    fn arguments(&self) -> Vec<Value> {
        self.calls
            .iter()
            .map(|(_, args)| serde_json::from_str(args).unwrap())
            .collect()
    }
}

struct OfficialChunks;
impl ChunkGenerator for OfficialChunks {
    type Chunk = OutputChunk;
    async fn generate(&mut self, chunk: OutputChunk) -> Vec<Self::Chunk> {
        vec![chunk]
    }
}

fn options(start: OutputStart, structured: bool) -> ParsingOptions {
    ParsingOptions {
        parse_tool_calls: true,
        starts_in_tool_calls: start == OutputStart::ToolBody,
        parse_json_output: structured,
        thinking_initial_stage: (start == OutputStart::Reasoning)
            .then_some(ThinkingInitialStage::ReasoningStart),
        ..ParsingOptions::default()
    }
}

async fn official(chunks: &[&str], start: OutputStart, structured: bool) -> Parsed {
    official_with_options(chunks, options(start, structured)).await
}

async fn official_with_options(chunks: &[&str], parsing: ParsingOptions) -> Parsed {
    let input = chunks
        .iter()
        .map(|text| InferenceChunk::Text {
            content: (*text).to_owned(),
            token_count: 1,
        })
        .chain([InferenceChunk::Finish {
            finish_reason: InferenceFinishReason::Stop,
        }]);
    let stream = StreamProcessor::new(OfficialChunks, parsing)
        .into_protocol_stream(futures_util::stream::iter(input));
    futures_util::pin_mut!(stream);
    let mut parsed = Parsed::default();
    while let Some(chunk) = stream.next().await {
        match chunk.unwrap() {
            OutputChunk::Content { content, .. } => parsed.content.push_str(&content),
            OutputChunk::Reasoning { content, .. } => parsed.reasoning.push_str(&content),
            OutputChunk::ToolCallNamed { tool_name } => {
                parsed.calls.push((tool_name, String::new()))
            }
            OutputChunk::ToolCallArgumentsDelta { arguments: content } => {
                parsed.calls.last_mut().unwrap().1.push_str(&content)
            }
            OutputChunk::Start { .. } | OutputChunk::ToolCallBegin | OutputChunk::Finish { .. } => {
            }
        }
    }
    parsed.ended = parsed.calls.len();
    parsed
}

fn pipeline(spaced: bool, start: OutputStart, structured: bool) -> OutputPipeline {
    pipeline_with_tools(
        spaced,
        start,
        structured,
        vec![
            json!({"type":"function","function":{"name":"lookup"}}),
            json!({"type":"function","function":{"name":"record"}}),
        ],
    )
}

fn pipeline_with_tools(
    spaced: bool,
    start: OutputStart,
    structured: bool,
    tools: Vec<Value>,
) -> OutputPipeline {
    let contract: PreparedContract = serde_json::from_str(if spaced {
        include_str!("support/contracts/deepseek41.json")
    } else {
        include_str!("support/contracts/deepseek4.json")
    })
    .unwrap();
    OutputPipeline::new(OutputConfig {
        syntax: contract.frontend,
        control_tokens: vec![],
        content: flare_helios::output::tagged::OutputConfig {
            structured,
            start,
            tools,
        },
    })
}

fn actual(
    chunks: &[&str],
    spaced: bool,
    start: OutputStart,
    structured: bool,
    end: Termination,
) -> flare_helios::Result<Parsed> {
    let mut output = pipeline(spaced, start, structured);
    let mut parsed = Parsed::default();
    for text in chunks {
        for event in output.push(Input::Text(text))? {
            parsed.event(event);
        }
    }
    for event in output.finish(end)? {
        parsed.event(event);
    }
    Ok(parsed)
}

fn block(spaced: bool) -> (String, String, String) {
    let s = if spaced { " " } else { "" };
    let outer = if spaced { " calls" } else { "tool_calls" };
    let open = format!("<｜DSML｜{outer}>");
    let close = format!("</｜DSML｜{outer}>");
    let body = format!(
        "\n<｜DSML｜{s}invoke name=\"lookup\">\n<｜DSML｜{s}parameter name=\"city\" string=\"true\">東京🙂\\\nquoted \"value\"</｜DSML｜{s}parameter>\n<｜DSML｜{s}parameter name=\"limit\" string=\"false\">3</｜DSML｜{s}parameter>\n</｜DSML｜{s}invoke>\n<｜DSML｜{s}invoke name=\"record\"><｜DSML｜{s}parameter name=\"items\" string=\"false\">[true,null,{{\"x\":2}}]</｜DSML｜{s}parameter></｜DSML｜{s}invoke>\n"
    );
    (open, body, close)
}

fn compare(actual: Parsed, expected: &Parsed) {
    assert_eq!(actual.content, expected.content);
    assert_eq!(actual.reasoning, expected.reasoning);
    assert_eq!(
        actual
            .calls
            .iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>(),
        expected
            .calls
            .iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>()
    );
    assert_eq!(actual.arguments(), expected.arguments());
    assert_eq!(actual.ended, expected.ended);
}

#[tokio::test]
async fn multiple_calls_match_official_processing_at_every_utf8_boundary() {
    for spaced in [false, true] {
        let (open, body, close) = block(spaced);
        let text = format!("分析🙂</think>\n\n{open}{body}{close}");
        let expected = official(&[&text], OutputStart::Reasoning, false).await;
        assert_eq!(expected.reasoning, "分析🙂");
        assert_eq!(
            expected.arguments(),
            vec![
                json!({"city":"東京🙂\\\nquoted \"value\"","limit":3}),
                json!({"items":[true,null,{"x":2}]})
            ]
        );
        for split in text.char_indices().map(|(i, _)| i).chain([text.len()]) {
            let chunks = [&text[..split], &text[split..]];
            compare(
                actual(
                    &chunks,
                    spaced,
                    OutputStart::Reasoning,
                    false,
                    Termination::Eos,
                )
                .unwrap(),
                &expected,
            );
        }
        let chunks: Vec<_> = text
            .char_indices()
            .map(|(i, c)| &text[i..i + c.len_utf8()])
            .collect();
        compare(
            actual(
                &chunks,
                spaced,
                OutputStart::Reasoning,
                false,
                Termination::Eos,
            )
            .unwrap(),
            &expected,
        );
        compare(
            official(&chunks, OutputStart::Reasoning, false).await,
            &expected,
        );
    }
}

#[tokio::test]
async fn prompt_opened_tool_block_and_implicit_reasoning_transition_match_official() {
    for spaced in [false, true] {
        let (open, body, close) = block(spaced);
        for (text, start) in [
            (format!("{body}{close}"), OutputStart::ToolBody),
            (
                format!("reasoning{open}{body}{close}"),
                OutputStart::Reasoning,
            ),
        ] {
            let chunks: Vec<_> = text
                .char_indices()
                .map(|(i, c)| &text[i..i + c.len_utf8()])
                .collect();
            let expected = official(&chunks, start, false).await;
            compare(
                actual(&chunks, spaced, start, false, Termination::Eos).unwrap(),
                &expected,
            );
        }
    }
}

#[tokio::test]
async fn json_literals_keep_markup_as_content_and_tool_blocks_end_output() {
    for spaced in [false, true] {
        let (open, body, close) = block(spaced);
        let text = json!({"literal":format!("{open} </think> <think> {close} 東京🙂")}).to_string();
        let chunks: Vec<_> = text
            .char_indices()
            .map(|(i, c)| &text[i..i + c.len_utf8()])
            .collect();
        let expected = official(&chunks, OutputStart::Content, true).await;
        assert_eq!(expected.content, text);
        assert!(expected.calls.is_empty());
        compare(
            actual(
                &chunks,
                spaced,
                OutputStart::Content,
                true,
                Termination::Eos,
            )
            .unwrap(),
            &expected,
        );
        let text = format!("{open}{body}{close}discarded trailing answer");
        let expected = official(&[&text], OutputStart::Content, false).await;
        assert!(expected.content.is_empty());
        compare(
            actual(
                &[&text],
                spaced,
                OutputStart::Content,
                false,
                Termination::Eos,
            )
            .unwrap(),
            &expected,
        );
    }
}

#[test]
fn complete_calls_validate_nonempty_names_and_json_arguments() {
    for spaced in [false, true] {
        let (open, body, close) = block(spaced);
        for invalid in [
            body.replace("name=\"lookup\"", "name=\"\""),
            body.replace(">3</", ">not-json</"),
        ] {
            let text = format!("{open}{invalid}{close}");
            assert!(
                actual(
                    &[&text],
                    spaced,
                    OutputStart::Content,
                    false,
                    Termination::Eos
                )
                .is_err(),
                "{text}"
            );
        }
    }
}

#[test]
fn incomplete_calls_do_not_finish_on_eos_and_can_be_cancelled_or_truncated() {
    for spaced in [false, true] {
        let (open, _, _) = block(spaced);
        let s = if spaced { " " } else { "" };
        let text = format!(
            "{open}<｜DSML｜{s}invoke name=\"lookup\"><｜DSML｜{s}parameter name=\"x\" string=\"true\">partial🙂"
        );
        assert!(
            actual(
                &[&text],
                spaced,
                OutputStart::Content,
                false,
                Termination::Eos
            )
            .is_err()
        );
        for end in [
            Termination::OutputLimit,
            Termination::UserStop,
            Termination::Cancelled,
            Termination::ExecutionError,
        ] {
            let parsed = actual(&[&text], spaced, OutputStart::Content, false, end).unwrap();
            assert_eq!(parsed.ended, 0);
            assert!(
                parsed
                    .calls
                    .iter()
                    .all(|(_, args)| serde_json::from_str::<Value>(args).is_err())
            );
            assert!(parsed.content.is_empty());
        }
    }
}

#[test]
fn structured_scalars_lark_and_markup_literals_remain_verbatim_after_answer_start() {
    for spaced in [false, true] {
        let (open, _, close) = block(spaced);
        let scalar_string =
            json!(format!("literal {open} </think> <think> {close} 東京🙂")).to_string();
        let lark = format!("answer: 東京🙂 {open} literal </think> <think> {close}");
        for answer in [
            "42",
            "-1.25e2",
            "true",
            "false",
            "null",
            scalar_string.as_str(),
            lark.as_str(),
        ] {
            for (start, prefix, reasoning) in [
                (OutputStart::Content, "", ""),
                (
                    OutputStart::Reasoning,
                    "analysis🙂</think>\n\n",
                    "analysis🙂",
                ),
            ] {
                let text = format!("{prefix}{answer}");
                for split in text.char_indices().map(|(i, _)| i).chain([text.len()]) {
                    let parsed = actual(
                        &[&text[..split], &text[split..]],
                        spaced,
                        start,
                        true,
                        Termination::Eos,
                    )
                    .unwrap();
                    assert_eq!(parsed.content, answer, "split={split}, spaced={spaced}");
                    assert_eq!(parsed.reasoning, reasoning);
                    assert!(parsed.calls.is_empty());
                    assert_eq!(parsed.ended, 0);
                }
                let chars: Vec<_> = text
                    .char_indices()
                    .map(|(i, c)| &text[i..i + c.len_utf8()])
                    .collect();
                let parsed = actual(&chars, spaced, start, true, Termination::Eos).unwrap();
                assert_eq!(parsed.content, answer);
                assert_eq!(parsed.reasoning, reasoning);
                assert!(parsed.calls.is_empty());
            }
        }
    }
}

#[test]
fn empty_and_duplicate_top_level_parameter_names_are_rejected() {
    for spaced in [false, true] {
        let (open, _, close) = block(spaced);
        let s = if spaced { " " } else { "" };
        let parameter = |name: &str, value: &str| {
            format!(
                "<｜DSML｜{s}parameter name=\"{name}\" string=\"false\">{value}</｜DSML｜{s}parameter>"
            )
        };
        for parameters in [
            parameter("", "1"),
            format!("{}{}", parameter("x", "1"), parameter("x", "2")),
        ] {
            let text = format!(
                "{open}<｜DSML｜{s}invoke name=\"lookup\">{parameters}</｜DSML｜{s}invoke>{close}"
            );
            for chunks in [
                vec![text.as_str()],
                text.char_indices()
                    .map(|(i, c)| &text[i..i + c.len_utf8()])
                    .collect(),
            ] {
                let mut output = pipeline(spaced, OutputStart::Content, false);
                let mut published = vec![];
                let mut error = None;
                for chunk in chunks {
                    match output.push(Input::Text(chunk)) {
                        Ok(events) => published.extend(events),
                        Err(failure) => {
                            error = Some(failure);
                            break;
                        }
                    }
                }
                assert!(error.unwrap().to_string().contains("nonempty and unique"));
                assert!(published.is_empty());
            }
        }
        let nested = parameter("object", "{\"x\":1,\"x\":2}");
        let text =
            format!("{open}<｜DSML｜{s}invoke name=\"lookup\">{nested}</｜DSML｜{s}invoke>{close}");
        let parsed = actual(
            &[&text],
            spaced,
            OutputStart::Content,
            false,
            Termination::Eos,
        )
        .unwrap();
        assert_eq!(parsed.arguments(), vec![json!({"object":{"x":2}})]);
    }
}

#[test]
fn tool_call_buffers_are_bounded_without_limiting_plain_content() {
    for spaced in [false, true] {
        let (open, _, close) = block(spaced);
        let s = if spaced { " " } else { "" };
        let large = "x".repeat(1024 * 1024 + 1);
        let plain = actual(
            &[&large],
            spaced,
            OutputStart::Content,
            false,
            Termination::Eos,
        )
        .unwrap();
        assert_eq!(plain.content, large);
        let escaped = "\\".repeat(600_000);
        for text in [
            format!("{open}<｜DSML｜{s}invoke name=\"{large}"),
            format!(
                "{open}<｜DSML｜{s}invoke name=\"lookup\"><｜DSML｜{s}parameter name=\"x\" string=\"true\">{large}</｜DSML｜{s}parameter></｜DSML｜{s}invoke>{close}"
            ),
            format!(
                "{open}<｜DSML｜{s}invoke name=\"lookup\"><｜DSML｜{s}parameter name=\"x\" string=\"true\">{escaped}</｜DSML｜{s}parameter></｜DSML｜{s}invoke>{close}"
            ),
        ] {
            let error = actual(
                &[&text],
                spaced,
                OutputStart::Content,
                false,
                Termination::Eos,
            )
            .err()
            .unwrap();
            assert!(error.to_string().contains("1 MiB"), "{error}");
        }
    }
}

#[tokio::test]
async fn no_tool_definitions_preserve_dsml_text_and_official_leading_newlines() {
    for spaced in [false, true] {
        let (open, body, close) = block(spaced);
        for (text, start) in [
            (format!("\n\n{open}{body}{close}"), OutputStart::Content),
            (
                format!("\n\nanalysis🙂\n\n</think>\n\n{open}{body}{close}"),
                OutputStart::Reasoning,
            ),
        ] {
            let chunks: Vec<_> = text
                .char_indices()
                .map(|(i, c)| &text[i..i + c.len_utf8()])
                .collect();
            let mut parsing = options(start, false);
            parsing.parse_tool_calls = false;
            let expected = official_with_options(&chunks, parsing).await;
            assert!(expected.calls.is_empty());
            assert!(expected.content.contains(&open));
            let mut output = pipeline_with_tools(spaced, start, false, vec![]);
            let mut parsed = Parsed::default();
            for chunk in &chunks {
                for event in output.push(Input::Text(chunk)).unwrap() {
                    parsed.event(event);
                }
            }
            for event in output.finish(Termination::Eos).unwrap() {
                parsed.event(event);
            }
            compare(parsed, &expected);
        }
        for (text, start) in [
            ("\n\nanswer🙂", OutputStart::Content),
            (
                "\n\nanalysis🙂\n\n</think>\n\nanswer",
                OutputStart::Reasoning,
            ),
        ] {
            let expected = official(&[text], start, false).await;
            let chunks: Vec<_> = text
                .char_indices()
                .map(|(i, c)| &text[i..i + c.len_utf8()])
                .collect();
            compare(
                actual(&chunks, spaced, start, false, Termination::Eos).unwrap(),
                &expected,
            );
        }
    }
}

#[test]
fn partial_control_markers_flush_on_eof_limit_and_stop_but_not_cancel() {
    for spaced in [false, true] {
        for partial in ["<｜DSM", "</thi"] {
            for end in [
                Termination::Eos,
                Termination::OutputLimit,
                Termination::UserStop,
                Termination::Cancelled,
                Termination::ExecutionError,
            ] {
                let mut output = pipeline(spaced, OutputStart::Content, false);
                let mut parsed = Parsed::default();
                for (i, c) in partial.char_indices() {
                    for event in output
                        .push(Input::Text(&partial[i..i + c.len_utf8()]))
                        .unwrap()
                    {
                        parsed.event(event);
                    }
                }
                assert!(parsed.content.is_empty());
                for event in output.finish(end).unwrap() {
                    parsed.event(event);
                }
                assert_eq!(
                    parsed.content,
                    if matches!(end, Termination::Cancelled | Termination::ExecutionError) {
                        ""
                    } else {
                        partial
                    }
                );
                assert!(parsed.calls.is_empty());
                assert!(output.push(Input::Text("after finish")).is_err());
                assert!(output.finish(Termination::Eos).is_err());
            }
        }
    }
}

#[test]
fn deepseek_control_markers_require_special_tokens_for_token_input() {
    for spaced in [false, true] {
        let contract: PreparedContract = serde_json::from_str(if spaced {
            include_str!("support/contracts/deepseek41.json")
        } else {
            include_str!("support/contracts/deepseek4.json")
        })
        .unwrap();
        let config = OutputConfig {
            syntax: contract.frontend,
            control_tokens: vec![(101, "</think>".into()), (102, "｜DSML｜".into())],
            content: flare_helios::output::tagged::OutputConfig {
                structured: false,
                start: OutputStart::Reasoning,
                tools: vec![json!({"type":"function","function":{"name":"lookup"}})],
            },
        };
        let mut parser = OutputPipeline::new(config);
        let mut parsed = Parsed::default();
        for ch in "literal </think> ".chars() {
            parser.push(Input::Token(1)).unwrap();
            for e in parser.push(Input::Text(&ch.to_string())).unwrap() {
                parsed.event(e);
            }
        }
        parser.push(Input::Token(101)).unwrap();
        for e in parser.push(Input::Text("</think>")).unwrap() {
            parsed.event(e);
        }
        for ch in "answer <｜DSML｜ calls>".chars() {
            parser.push(Input::Token(1)).unwrap();
            for e in parser.push(Input::Text(&ch.to_string())).unwrap() {
                parsed.event(e);
            }
        }
        for e in parser.finish(Termination::Eos).unwrap() {
            parsed.event(e);
        }
        assert_eq!(parsed.reasoning, "literal </think> ");
        assert_eq!(parsed.content, "answer <｜DSML｜ calls>");
        assert!(parsed.calls.is_empty());
    }
}

#[test]
fn deepseek_tool_markup_uses_special_dsml_token() {
    for spaced in [false, true] {
        let contract: PreparedContract = serde_json::from_str(if spaced {
            include_str!("support/contracts/deepseek41.json")
        } else {
            include_str!("support/contracts/deepseek4.json")
        })
        .unwrap();
        let config = OutputConfig {
            syntax: contract.frontend,
            control_tokens: vec![(101, "</think>".into()), (102, "｜DSML｜".into())],
            content: flare_helios::output::tagged::OutputConfig {
                structured: false,
                start: OutputStart::Content,
                tools: vec![json!({"type":"function","function":{"name":"lookup"}})],
            },
        };
        let (block, call) = if spaced {
            (" calls", " invoke")
        } else {
            ("tool_calls", "invoke")
        };
        let text = format!(
            "<｜DSML｜{block}><｜DSML｜{call} name=\"lookup\"></｜DSML｜{call}></｜DSML｜{block}>"
        );
        let mut parser = OutputPipeline::new(config);
        let mut parsed = Parsed::default();
        let parts: Vec<_> = text.split("｜DSML｜").collect();
        for (i, part) in parts.iter().enumerate() {
            if i > 0 {
                parser.push(Input::Token(102)).unwrap();
                for event in parser.push(Input::Text("｜DSML｜")).unwrap() {
                    parsed.event(event);
                }
            }
            for ch in part.chars() {
                parser.push(Input::Token(1)).unwrap();
                for event in parser.push(Input::Text(&ch.to_string())).unwrap() {
                    parsed.event(event);
                }
            }
        }
        for event in parser.finish(Termination::Eos).unwrap() {
            parsed.event(event);
        }
        assert_eq!(parsed.calls, [("lookup".into(), "{}".into())]);
        assert_eq!(parsed.ended, 1);
    }
}
