/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CPU request preparation with the canonical recipient-header checkpoint frontend.

use super::*;
use crate::frontend::{Asset, FrontendAssets};
use crate::output::tagged::OutputStart;
use crate::server::protocol::{Protocol, decode};
use serde_json::json;
use sha2::{Digest, Sha256};

#[path = "../../../tests/support/frontend.rs"]
mod support;

#[test]
#[ignore = "requires FLARE_MUSE_SPARK_SNAPSHOT checkpoint tokenizer"]
fn recipient_request_preparation_preserves_protocols_tools_and_continuation() {
    let snapshot = std::path::PathBuf::from(std::env::var_os("FLARE_MUSE_SPARK_SNAPSHOT").unwrap());
    let asset = |path: std::path::PathBuf| {
        let bytes = std::fs::read(&path).unwrap();
        Asset {
            path,
            range: None,
            sha256: format!("{:x}", Sha256::digest(bytes)),
        }
    };
    let frontend = Frontend::load(
        FrontendAssets {
            tokenizer: asset(snapshot.join("tokenizer.json")),
            tokens: crate::frontend::test_support::tokens(&asset(
                snapshot.join("tokenizer_config.json"),
            ))
            .unwrap(),
            chat_template: Some(asset(
                crate::template_fixtures::path("muse_spark", "chat_template"),
            )),
        },
        json!({"add_generation_prompt":true,"current_date":"2026-10-27"})
            .as_object()
            .unwrap()
            .clone(),
    )
    .unwrap();
    let mut config = support::model_config(crate::model::testing::metadata());
    config.contract.vocabulary = frontend.tokenizer().get_vocab_size(true);
    config.contract.input_vocabulary = config.contract.vocabulary;
    config.contract.vision.max_images = 0;
    config.contract.frontend=serde_json::from_value(json!({
        "tools":{"kind":7,"wrapping_newlines":0,"tag_separator":""},
        "reasoning":{"kind":5,"open":"<|start|>assistant to=self<|message|>","close":"<|eom|>","wrapping_newlines":0,"terminal_markers":[]},
        "assistant_marker":"<|start|>assistant","message_end_markers":["<|eom|>","<|eot|>"],
        "rejected_tokens":[],"position_dimensions":0
    })).unwrap();
    config.eos_token_ids = vec![200001, 200008];
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
        assert!(request.rendered_prompt.ends_with("<|start|>assistant"));
        assert_eq!(
            request.output.unwrap().content.start,
            OutputStart::RecipientHeader
        );
        assert!(!request.prompt.tokens.is_empty());
    }
    let messages_body = json!({
        "max_tokens":2048,
        "messages":[{"role":"user","content":"Reply with only PROTOCOL-MESSAGES."}],
        "model":"Muse-Spark-Open-816B-A42B",
        "temperature":1,"top_p":1,"top_k":-1,
        "chat_template_kwargs":{"reasoning_effort":"low","current_date":"2026-10-27"}
    });
    let decoded = decode(Protocol::Messages, messages_body.clone(), false).unwrap();
    let crate::server::inference::Input::Conversation(messages) = &decoded.request.input else {
        panic!("Messages must lower to a conversation");
    };
    let normalized = crate::server::inference::template_messages(
        messages,
        decoded.request.tool_choice.clone(),
        &decoded.request.checkpoint_hints,
    )
    .unwrap();
    assert_eq!(&normalized, messages_body["messages"].as_array().unwrap());
    assert_eq!(
        &decoded.request.options.template,
        messages_body["chat_template_kwargs"].as_object().unwrap()
    );
    let request = prepare_inference(
        &frontend,
        None,
        &compiler,
        &config,
        &decoded.request,
        0,
        false,
    )
    .unwrap();
    // Tokenizer IDs from meta-models/Muse-Spark-Open-816B-A42B-NVFP4
    // at revision 4239e4db19376904f2e82c167e78f70abb5455fe for the Messages body above.
    let expected_tokens = [
        200000, 200022, 15651, 200023, 4662, 583, 262, 19933, 12133, 43910, 1574, 111198, 67059,
        38, 220, 837, 34, 25, 1761, 25, 2468, 1574, 8048, 4282, 38, 220, 837, 34, 25, 580, 25,
        1704, 1574, 34956, 300, 9762, 38, 220, 36, 1574, 7533, 290, 10116, 41712, 393, 1789, 3623,
        4568, 25, 392, 2540, 1217, 1566, 35538, 341, 7935, 14479, 335, 25, 392, 13929, 811, 1217,
        1941, 130639, 26001, 18870, 2467, 290, 43910, 917, 6639, 6050, 24, 4414, 12485, 4298, 328,
        290, 1941, 3020, 537, 1867, 7935, 13915, 335, 25, 392, 1556, 1217, 12485, 511, 1352, 290,
        43910, 4651, 24, 2203, 486, 262, 13544, 3495, 537, 113026, 4097, 511, 110118, 393, 290,
        46103, 22383, 39, 641, 702, 1161, 393, 853, 55985, 18870, 537, 10453, 20276, 1574, 15,
        14757, 73965, 38, 392, 2540, 706, 392, 13929, 811, 706, 392, 1556, 4205, 200008, 200022,
        1556, 200023, 33692, 517, 1809, 198964, 4872, 132753, 26, 200008, 200022, 140680,
    ];
    assert_eq!(request.prompt.tokens.as_slice(), &expected_tokens);
    assert_eq!(request.prompt.max_new_tokens, 2048);
    assert!(request.rendered_prompt.contains("Reasoning strength: 8."));
    assert!(
        request
            .rendered_prompt
            .ends_with("Reply with only PROTOCOL-MESSAGES.<|eot|><|start|>assistant")
    );
    let tools = json!([{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"query":{"type":"string"}}}}}]);
    for choice in [
        json!("none"),
        json!("auto"),
        json!("required"),
        json!({"type":"function","function":{"name":"lookup"}}),
    ] {
        let request=prepare(Protocol::Chat,json!({"messages":[{"role":"user","content":"Find it"}],"tools":tools,"tool_choice":choice,
            "response_format":{"type":"json_schema","json_schema":{"name":"result","schema":{"type":"object"}}}})).unwrap();
        assert!(request.prompt.grammar.is_some());
        assert_eq!(
            request.output.unwrap().content.tools.is_empty(),
            choice == "none"
        );
        if choice == "required" {
            assert!(
                request
                    .rendered_prompt
                    .contains("You must call at least one")
            );
        }
    }
    let history = json!([{"role":"user","content":"Find it"},{"role":"assistant","tool_calls":[{"id":"a","type":"function","function":{"name":"lookup","arguments":"{\"query\":\"世界\"}"}}]},
        {"role":"tool","tool_call_id":"a","content":"Found"},{"role":"user","content":"Explain"}]);
    let request = prepare(Protocol::Chat, json!({"messages":history,"tools":tools})).unwrap();
    assert!(
        request
            .rendered_prompt
            .contains("<atem:parameter name=\"query\">世界</atem:parameter>")
    );
    assert!(
        request
            .rendered_prompt
            .contains("<tool_output name=\"lookup\">\nFound\n</tool_output>")
    );
    assert!(request.prompt.message_boundaries.len() >= 4);
    for (message, start) in [
        (
            json!({"role":"assistant","content":"Partial café"}),
            OutputStart::Content,
        ),
        (
            json!({"role":"assistant","content":"","reasoning_content":"Plan café"}),
            OutputStart::Reasoning,
        ),
    ] {
        let field = if start == OutputStart::Reasoning {
            json!("reasoning_content")
        } else {
            json!(true)
        };
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
    let image = prepare(
        Protocol::Chat,
        json!({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}]}),
    );
    assert!(image.is_err());
}
