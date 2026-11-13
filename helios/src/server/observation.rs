/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::Result;
use serde_json::{Map, Value, json};
use std::sync::OnceLock;
use std::time::Instant;

static ORIGIN: OnceLock<Instant> = OnceLock::new();

pub(super) fn initialize_clock() {
    ORIGIN.get_or_init(Instant::now);
}

pub(super) fn monotonic_ns() -> u64 {
    ORIGIN.get_or_init(Instant::now).elapsed().as_nanos() as u64
}

#[derive(Clone, Copy, Default)]
pub(crate) struct ObservationOptions {
    pub token_ids: bool,
    pub timestamps: bool,
    pub include_usage: bool,
    pub continuous_usage: bool,
}

impl ObservationOptions {
    pub fn parse(object: &Map<String, Value>) -> Result<Self> {
        let token_ids = flag(object.get("return_token_ids"), "return_token_ids")?;
        let timestamps = flag(
            object.get("return_token_timestamps"),
            "return_token_timestamps",
        )?;
        if timestamps && !token_ids {
            return Err("return_token_timestamps requires return_token_ids=true".into());
        }
        let (include_usage, continuous_usage) = match object.get("stream_options") {
            None | Some(Value::Null) => (false, false),
            Some(Value::Object(options)) => {
                (flag(options.get("include_usage"), "stream_options.include_usage")?,
                 flag(options.get("continuous_usage_stats"), "stream_options.continuous_usage_stats")?)
            }
            _ => return Err("stream_options must be an object or null".into()),
        };
        Ok(Self {
            token_ids,
            timestamps,
            include_usage,
            continuous_usage: include_usage && continuous_usage,
        })
    }

    pub fn finish(
        &self,
        response: &mut Value,
        usage: &Value,
        reused: u64,
        timing: (Option<u64>, Option<u64>),
        stream: bool,
    ) {
        if self.token_ids {
            response["reused_prompt_tokens"] = json!(reused);
            if !stream || !self.include_usage {
                response["usage"] = usage.clone();
            }
        }
        if self.timestamps {
            let (ingress_ns, enqueue_ns) = timing;
            response["performance"] = json!({"ingress_ns":ingress_ns,"enqueue_ns":enqueue_ns,
                "reused_prompt_tokens":reused});
        }
    }
}

fn flag(value: Option<&Value>, name: &str) -> Result<bool> {
    match value {
        None | Some(Value::Null) => Ok(false),
        Some(Value::Bool(value)) => Ok(*value),
        _ => Err(format!("{name} must be boolean or null").into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observation_flags_are_explicit_booleans_and_timestamps_require_ids() {
        for body in [
            json!({}),
            json!({"return_token_ids":null,"return_token_timestamps":null,"stream_options":null}),
            json!({"return_token_ids":false,"stream_options":{"include_usage":null}}),
        ] {
            let options = ObservationOptions::parse(body.as_object().unwrap()).unwrap();
            assert!(!options.token_ids && !options.timestamps && !options.include_usage);
        }
        let body = json!({"return_token_ids":true,"return_token_timestamps":true,"stream_options":{"include_usage":true}});
        let options = ObservationOptions::parse(body.as_object().unwrap()).unwrap();
        assert!(options.token_ids && options.timestamps && options.include_usage);
        for body in [
            json!({"return_token_timestamps":true}),
            json!({"return_token_ids":1}),
            json!({"return_token_ids":"true"}),
            json!({"return_token_ids":true,"return_token_timestamps":0}),
            json!({"stream_options":false}),
            json!({"stream_options":{"include_usage":"true"}}),
            json!({"stream_options":{"continuous_usage_stats":"true"}}),
        ] {
            assert!(ObservationOptions::parse(body.as_object().unwrap()).is_err());
        }
    }

    #[test]
    fn continuous_usage_requires_explicit_usage_opt_in() {
        for (body, expected) in [
            (json!({"stream_options":{"include_usage":true,"continuous_usage_stats":true}}), true),
            (json!({"stream_options":{"include_usage":true}}), false),
            (json!({"stream_options":{"continuous_usage_stats":true}}), false),
            (json!({"stream_options":{"include_usage":false,"continuous_usage_stats":true}}), false),
        ] {
            let options = ObservationOptions::parse(body.as_object().unwrap()).unwrap();
            assert_eq!(options.continuous_usage, expected);
        }
    }

    #[test]
    fn terminal_metadata_preserves_presentation_and_counts_raw_tokens() {
        let original = json!({"choices":[{"message":{"content":"answer"},"finish_reason":"stop"}]});
        let usage = json!({"prompt_tokens":17,"completion_tokens":3,"total_tokens":20,
            "prompt_tokens_details":{"cached_tokens":16,"multimodal_tokens":{"image":8}}});
        let mut value = original.clone();
        ObservationOptions::default().finish(&mut value, &usage, 16, (None, None), true);
        assert_eq!(value, original);
        let options = ObservationOptions {
            token_ids: true,
            timestamps: true,
            include_usage: false,
            continuous_usage: false,
        };
        options.finish(&mut value, &usage, 16, (Some(10), Some(20)), true);
        assert_eq!(value["choices"], original["choices"]);
        assert_eq!(value["usage"], usage);
        assert_eq!(
            value["performance"],
            json!({"ingress_ns":10,"enqueue_ns":20,"reused_prompt_tokens":16})
        );
        assert_eq!(value["reused_prompt_tokens"], 16);
        let mut separate = original.clone();
        ObservationOptions {
            include_usage: true,
            ..options
        }
        .finish(&mut separate, &usage, 16, (Some(10), Some(20)), true);
        assert!(separate.get("usage").is_none());
        options.finish(&mut value, &usage, 16, (Some(10), Some(20)), false);
        assert_eq!(value["usage"], usage);
    }
}
