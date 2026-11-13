/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::jinja::{CompiledTemplate, Error};
use serde::Deserialize;
use sha2::{Digest, Sha256};

const QWEN: &str = include_str!("../../../ignition/recipes/qwen38/assets/chat_template.patch.json");
const STEP: &str = include_str!("../../../ignition/recipes/step5/assets/chat_template.patch.json");
const KIMI3: &str = include_str!("../../../ignition/recipes/kimi3/assets/chat_template.jinja");

#[derive(Deserialize)]
struct AnnotationPatch {
    source_sha256: String,
    result_sha256: String,
    insertions: Vec<usize>,
}

pub(super) fn compile(source: &str) -> Result<CompiledTemplate, Error> {
    // Upgrade only the exact previously deployed assets. No upstream template
    // text is embedded in the binary; the core supplies the source bytes.
    let digest = format!("{:x}", Sha256::digest(source.as_bytes()));
    for encoded in [QWEN, STEP] {
        let patch: AnnotationPatch =
            serde_json::from_str(encoded).expect("bundled annotation patch");
        if digest != patch.source_sha256 {
            continue;
        }
        let mut annotated = String::new();
        let mut previous = 0;
        for offset in patch.insertions {
            let Some(part) = source.get(previous..offset) else {
                return Err(Error {
                    message: "invalid bundled template insertion".into(),
                    offset: 0,
                });
            };
            annotated.push_str(part);
            annotated.push_str("{%- mark -%}");
            previous = offset;
        }
        annotated.push_str(&source[previous..]);
        if format!("{:x}", Sha256::digest(annotated.as_bytes())) != patch.result_sha256 {
            return Err(Error {
                message: "bundled template patch digest mismatch".into(),
                offset: 0,
            });
        }
        return CompiledTemplate::compile(&annotated);
    }
    let kimi_raw = KIMI3.replace("{%- mark ns.segments | length -%}", "");
    let (_, kimi_legacy) = kimi_raw.split_once("#}\n").expect("Kimi copyright header");
    if source == kimi_raw || source == kimi_legacy {
        return CompiledTemplate::compile(KIMI3);
    }
    let compiled = CompiledTemplate::compile(source)?;
    if compiled.annotation_spans().is_empty() {
        return Err(Error { message: "chat template requires message-boundary mark annotations; only the deployed Qwen3.8, Step, and Kimi K3 templates can be upgraded automatically".into(), offset: 0 });
    }
    Ok(compiled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compatibility_requires_exact_legacy_bytes() {
        for annotated in [
            crate::template_fixtures::template("qwen38", "chat_template"),
            crate::template_fixtures::template("step5", "chat_template"),
            KIMI3.to_owned(),
        ] {
            let mut raw = annotated.to_owned();
            for span in CompiledTemplate::compile(&annotated)
                .unwrap()
                .annotation_spans()
                .iter()
                .rev()
            {
                raw.replace_range(span.clone(), "");
            }
            assert!(!compile(&raw).unwrap().annotation_spans().is_empty());
            assert!(compile(&(raw + " ")).is_err());
            assert!(compile(&annotated).is_ok());
        }
        let (_, kimi_legacy) = KIMI3.split_once("#}\n").unwrap();
        let kimi_legacy = kimi_legacy.replace("{%- mark ns.segments | length -%}", "");
        assert!(!compile(&kimi_legacy).unwrap().annotation_spans().is_empty());
        assert!(compile(&(kimi_legacy + " ")).is_err());
        assert!(compile("{{ messages }}").is_err());
    }
}
