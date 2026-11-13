# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record exact-checkpoint Transformers tokenizer and chat-template outputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from transformers import AutoTokenizer

from ignition.preparation.qwen38 import FRONTEND_SHA256, MODEL_ID, REVISION, TRANSFORMERS_REVISION


TRANSFORMERS_COMMIT = TRANSFORMERS_REVISION


def chat_cases():
    user = [{"role": "user", "content": "Hello, café 世界 👋.\nReturn a concise answer."}]
    yield "empty_conversation", [], {}
    yield "default", user, {}
    yield "generation_prompt", user, {"add_generation_prompt": True}
    yield "thinking_disabled", user, {"add_generation_prompt": True, "enable_thinking": False}
    yield "thinking_enabled", user, {"add_generation_prompt": True, "enable_thinking": True}
    for effort in ("low", "medium", "xhigh"):
        yield f"reasoning_{effort}", user, {"add_generation_prompt": True, "reasoning_effort": effort}
    yield "system", [{"role": "system", "content": "  Be precise.  "}, *user], {}
    yield "empty_system", [{"role": "system", "content": " "}, *user], {"enable_thinking": False}
    assistant = {"role": "assistant", "content": "The answer is", "reasoning_content": "  Consider the inputs.  "}
    yield "assistant_continuation", [*user, assistant], {"continue_final_message": True}
    yield "reasoning_continuation", [*user, assistant], {"continue_final_message": "reasoning_content"}
    yield "continuation_trailing_spaces", [*user, {"role": "assistant", "content": "The answer is   "}], {"continue_final_message": True}
    yield "continuation_text_blocks", [*user, {"role": "assistant", "content": [
        {"type": "text", "text": "First block. "}, {"type": "text", "text": "Second block  "},
    ]}], {"continue_final_message": True}
    yield "continuation_missing_field", [*user, {"role": "assistant", "content": "Answer"}], {"continue_final_message": "reasoning_content"}
    yield "continuation_image_only", [*user, {"role": "assistant", "content": [{"type": "image", "image": "fixture://only.png"}]}], {"continue_final_message": True}
    history = [*user, assistant, {"role": "user", "content": "Continue."}]
    yield "discard_past_thinking", history, {"preserve_thinking": False, "add_generation_prompt": True}
    yield "preserve_past_thinking", history, {"preserve_thinking": True, "add_generation_prompt": True}
    images = [{"role": "user", "content": [
        {"type": "text", "text": "Compare: "},
        {"type": "image", "image": "fixture://first.png"},
        {"type": "image_url", "image_url": {"url": "fixture://second.png"}},
    ]}]
    yield "image_placeholders", images, {"add_generation_prompt": True}
    yield "image_numbering", images, {"add_vision_id": True, "add_generation_prompt": True}
    tools = [{"type": "function", "function": {
        "name": "search", "description": "Find <items> & café data.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "options": {"type": "object"},
        }, "required": ["query"]},
    }}]
    yield "tool_schema", user, {"tools": tools, "add_generation_prompt": True}
    calls = {"role": "assistant", "content": "", "reasoning_content": "Need two sources.", "tool_calls": [
        {"type": "function", "function": {"name": "search", "arguments": {
            "query": "café 世界", "options": {"limit": 2, "ratio": 1.25, "enabled": True, "nullable": None,
                "threshold": 1e-6, "large": 1e16, "negative_zero": -0.0},
        }}},
        {"type": "function", "function": {"name": "search", "arguments": {"query": "a\nb", "options": [1, "✓"]}}},
    ]}
    yield "tool_calls", [*user, calls], {"tools": tools}
    yield "tool_results", [*user, calls,
        {"role": "tool", "content": "First result."},
        {"role": "tool", "content": "Second result."},
    ], {"tools": tools, "add_generation_prompt": True}
    yield "empty_assistant", [*user, {"role": "assistant", "content": None}], {}
    yield "invalid_system_image", [{"role": "system", "content": images[0]["content"]}, *user], {}
    yield "invalid_role", [*user, {"role": "developer", "content": "Invalid role."}], {}
    yield "invalid_content", [{"role": "user", "content": 42}], {}
    yield "invalid_content_item", [{"role": "user", "content": [{"type": "audio"}]}], {}
    yield "invalid_reasoning_effort", user, {"reasoning_effort": "high"}
    yield "misplaced_system", [*user, {"role": "system", "content": "Invalid position."}], {}
    yield "missing_user", [{"role": "assistant", "content": "No query."}], {}
    yield "incompatible_continuation", [*user, assistant], {"continue_final_message": True, "add_generation_prompt": True}


def generate(snapshot: Path) -> dict:
    distribution = importlib.metadata.distribution("transformers")
    provenance = json.loads(distribution.read_text("direct_url.json") or "{}")
    if provenance.get("vcs_info", {}).get("commit_id") != TRANSFORMERS_COMMIT:
        raise RuntimeError(f"fixture generation requires Transformers {TRANSFORMERS_COMMIT}")
    assets = {}
    for name, expected in FRONTEND_SHA256.items():
        actual = hashlib.sha256((snapshot / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"{name}: checkpoint fingerprint mismatch")
        assets[name] = actual
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    cases = []
    for name, messages, options in chat_cases():
        case = {"name": name, "messages": messages, "options": options}
        try:
            case["rendered"] = tokenizer.apply_chat_template(messages, tokenize=False, **options)
            case["token_ids"] = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, **options)
        except (ValueError, TypeError) as error:
            case["error"] = str(error)
        except Exception as error:
            from jinja2 import TemplateError
            if not isinstance(error, TemplateError):
                raise
            case["error"] = str(error)
        cases.append(case)
    encode_cases = []
    for name, text in (
        ("unicode", "café 世界 👋\n\t🚀"),
        ("whitespace", "  multiple   spaces\n\ntrailing "),
        ("special_tokens", "<|im_start|>assistant\n<think>test</think><|im_end|>"),
        ("incomplete_sentence", "The number is 1.25 and the path is /tmp/a.py."),
    ):
        for special in (False, True):
            encode_cases.append({"name": f"{name}_{special}", "text": text,
                "add_special_tokens": special, "token_ids": tokenizer.encode(text, add_special_tokens=special)})
    decode_cases = [{"name": case["name"], "token_ids": case["token_ids"], "skip_special_tokens": skip,
        "text": tokenizer.decode(case["token_ids"], skip_special_tokens=skip, clean_up_tokenization_spaces=False)}
        for case in encode_cases for skip in (False, True)]
    unicode_ids = tokenizer.encode("🧬👨‍👩‍👧‍👦", add_special_tokens=False)
    for end in range(1, len(unicode_ids) + 1):
        ids = unicode_ids[:end]
        decode_cases.append({"name": f"unicode_prefix_{end}", "token_ids": ids,
            "skip_special_tokens": False, "text": tokenizer.decode(ids, clean_up_tokenization_spaces=False)})
    ids = tokenizer.encode("<|im_start|><|im_end|><|endoftext|>", add_special_tokens=False)
    for skip in (False, True):
        decode_cases.append({"name": f"special_only_{skip}", "token_ids": ids, "skip_special_tokens": skip,
            "text": tokenizer.decode(ids, skip_special_tokens=skip, clean_up_tokenization_spaces=False)})
    return {"model_id": MODEL_ID, "revision": REVISION, "transformers_commit": TRANSFORMERS_COMMIT,
        "tokenizers_version": importlib.metadata.version("tokenizers"), "assets": assets,
        "cases": cases, "encode_cases": encode_cases, "decode_cases": decode_cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = generate(args.snapshot)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {len(result['cases'])} chat, {len(result['encode_cases'])} encode, and {len(result['decode_cases'])} decode cases")


if __name__ == "__main__":
    main()
