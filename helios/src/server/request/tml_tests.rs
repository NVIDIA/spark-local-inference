/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CPU request preparation with the canonical content-header checkpoint frontend.

use super::*;
use crate::frontend::{Asset, FrontendAssets};
use crate::output::tagged::OutputStart;
use crate::server::protocol::{Protocol, decode};
use serde_json::json;
use sha2::{Digest, Sha256};

#[path = "../../../tests/support/frontend.rs"]
mod support;

#[test]
#[ignore = "requires FLARE_INKLING_SNAPSHOT and FLARE_TEST_TEMPLATE_DIR checkpoint assets"]
fn content_headers_preserve_requests_tools_and_constrained_answers() {
    let snapshot = std::path::PathBuf::from(std::env::var_os("FLARE_INKLING_SNAPSHOT").unwrap());
    let asset = |path: std::path::PathBuf| Asset {
        sha256: format!("{:x}", Sha256::digest(std::fs::read(&path).unwrap())),
        path,
        range: None,
    };
    let frontend = Frontend::load(
        FrontendAssets {
            tokenizer: asset(snapshot.join("tokenizer.json")),
            tokens: crate::frontend::test_support::tokens(&asset(
                snapshot.join("tokenizer_config.json"),
            ))
            .unwrap(),
            chat_template: Some(asset(crate::template_fixtures::path(
                "inkling",
                "chat_template",
            ))),
        },
        json!({"add_generation_prompt":true})
            .as_object()
            .unwrap()
            .clone(),
    )
    .unwrap();
    let mut config = support::model_config(crate::model::testing::metadata());
    config.contract.vocabulary = frontend.tokenizer().get_vocab_size(true);
    config.contract.input_vocabulary = config.contract.vocabulary;
    config.contract.vision.max_images = 0;
    config.contract.frontend = serde_json::from_value(json!({
        "tools":{"kind":8,"wrapping_newlines":0,"tag_separator":""},
        "reasoning":{"kind":6,"open":"<|content_thinking|>","close":"<|end_message|>","wrapping_newlines":0,"terminal_markers":[]},
        "assistant_marker":"<|message_model|>","message_end_markers":["<|end_message|>","<|content_model_end_sampling|>"],
        "rejected_tokens":[200005,200020,200023,200053,200054],"position_dimensions":0
    })).unwrap();
    config.eos_token_ids = vec![200006];
    let compiler = crate::grammar::GrammarCompiler::new(
        frontend.tokenizer(),
        config.contract.vocabulary,
        &config.eos_token_ids,
        &crate::model::reasoning_delimiter_tokens(&frontend, &config.contract.frontend).unwrap(),
    )
    .unwrap();
    let prepare = |protocol, mut body: Value| {
        body["model"] = json!(config.model_id);
        let decoded = decode(protocol, body, false)?;
        prepare_inference(
            &frontend,
            None,
            &compiler,
            &config,
            &decoded.request,
            0,
            false,
        )
    };
    for (protocol, body) in [
        (
            Protocol::Chat,
            json!({"messages":[{"role":"user","content":"Hello 世界"}]}),
        ),
        (
            Protocol::Messages,
            json!({"max_tokens":32,"messages":[{"role":"user","content":"Hello 世界"}]}),
        ),
        (Protocol::Responses, json!({"input":"Hello 世界"})),
    ] {
        let request = prepare(protocol, body).unwrap();
        assert!(request.rendered_prompt.ends_with("<|message_model|>"));
        assert!(
            request
                .rendered_prompt
                .contains("Thinking effort level: 0.9")
        );
        assert_eq!(
            request.output.unwrap().content.start,
            OutputStart::RecipientHeader
        );
    }
    let tools = json!([{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"query":{"type":"string"}}}}}]);
    let history = json!([{"role":"user","content":"Find it"},
        {"role":"assistant","tool_calls":[{"id":"a","type":"function","function":{"name":"lookup","arguments":"{\"query\":\"世界\"}"}}]},
        {"role":"tool","tool_call_id":"a","content":"Found"},{"role":"user","content":"Explain"}]);
    let request = prepare(Protocol::Chat, json!({"messages":history,"tools":tools})).unwrap();
    assert!(request.rendered_prompt.contains(
        "lookup<|content_invoke_tool_json|>{\"name\":\"lookup\",\"args\":{\"query\":\"世界\"}}"
    ));
    assert!(
        request
            .rendered_prompt
            .contains("<|message_tool|>lookup<|content_text|>Found<|end_message|>")
    );
    assert!(request.prompt.message_boundaries.len() >= 4);
    for (message, field, start) in [
        (
            json!({"role":"assistant","content":"Partial café"}),
            json!(true),
            OutputStart::Content,
        ),
        (
            json!({"role":"assistant","content":"","reasoning_content":"Plan café"}),
            json!("reasoning_content"),
            OutputStart::Reasoning,
        ),
    ] {
        let request = prepare(
            Protocol::Chat,
            json!({"messages":[{"role":"user","content":"Hello"},message],
            "chat_template_kwargs":{"continue_final_message":field,"add_generation_prompt":false}}),
        )
        .unwrap();
        assert_eq!(request.output.unwrap().content.start, start);
    }
    let closed = prepare(
        Protocol::Chat,
        json!({"messages":[{"role":"user","content":"Hello"}],
        "chat_template_kwargs":{"add_generation_prompt":false}}),
    )
    .unwrap();
    assert_eq!(
        closed.output.unwrap().content.start,
        OutputStart::RecipientFrame
    );
    assert!(
        prepare(
            Protocol::Chat,
            json!({"messages":[{"role":"user","content":"<|content_image|>"}]})
        )
        .is_err()
    );
    for choice in [
        json!("none"),
        json!("auto"),
        json!("required"),
        json!({"type":"function","function":{"name":"lookup"}}),
    ] {
        let request = prepare(Protocol::Chat, json!({"messages":[{"role":"user","content":"Find"}],"tools":tools,"tool_choice":choice,
            "response_format":{"type":"json_schema","json_schema":{"name":"result","schema":{"const":{"ok":true}}}}})).unwrap();
        assert!(request.prompt.grammar.is_some());
        assert_eq!(
            request.output.unwrap().content.tools.is_empty(),
            choice == "none"
        );
    }
    let answer = "<|content_text|>{\"ok\":true}<|end_message|>";
    let thinking = "<|content_thinking|>Plan 世界<|end_message|><|message_model|>";
    let call = "lookup<|content_invoke_tool_json|>{\"name\":\"lookup\",\"args\":{\"query\":\"café\"}}<|end_message|>";
    for (text, valid) in [
        (answer.to_owned(), true),
        (format!("{thinking}{answer}"), true),
        (call.to_owned(), true),
        (format!("{call}<|message_model|>{thinking}{call}"), true),
        (answer.replace("true", "false"), false),
        (call.replace("lookup", "other"), false),
        (
            call.replace("\"name\":\"lookup\"", "\"name\":\"other\""),
            false,
        ),
        (
            call.replace("\"args\":{\"query\":\"café\"}", "\"args\":[]"),
            false,
        ),
    ] {
        let mut request = prepare(Protocol::Chat, json!({"messages":[{"role":"user","content":"Find"}],"tools":tools,
            "response_format":{"type":"json_schema","json_schema":{"name":"result","schema":{"const":{"ok":true}}}}})).unwrap();
        let grammar = request.prompt.grammar.as_mut().unwrap();
        let accepted = frontend
            .encode(&text, false)
            .unwrap()
            .into_iter()
            .all(|id| grammar.advance(id).is_ok())
            && grammar.advance(200006).is_ok();
        assert_eq!(accepted, valid, "{text}");
    }
}
