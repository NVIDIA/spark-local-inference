# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native TML rendering against the pinned checkpoint template and tokenizer."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = Path(os.environ.get("FLARE_INKLING_SNAPSHOT", "/data/models/Inkling-Small"))


def test_native_inkling_matches_checkpoint(tmp_path):
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast
    from transformers.utils.chat_template_utils import _compile_jinja_template
    from ignition.recipes.inkling.frontend import prepare_frontend, runtime_assets

    if not (SNAPSHOT / "tokenizer.json").is_file():
        pytest.skip("requires pinned Inkling-Small frontend assets")
    assets = runtime_assets(SNAPSHOT)
    settings = prepare_frontend(SNAPSHOT, eos_token_ids=[200006])
    tokenizer = Tokenizer.from_file(str(SNAPSHOT / "tokenizer.json"))
    hf = PreTrainedTokenizerFast.from_pretrained(SNAPSHOT)
    source = (SNAPSHOT / "chat_template.jinja").read_text()
    annotated = assets["chat_template"]["text"]
    assert annotated.replace("{%- mark -%}", "") == source
    reference = _compile_jinja_template(source)
    marked = _compile_jinja_template(annotated.replace("{%- mark -%}", "{{- message_position() -}}"))
    cases = []

    def add(name, messages, **options):
        options = dict(dict(add_generation_prompt=True), **options)
        rendered = reference.render(messages=messages, **settings["tokenizer"], **options)
        marker = "\0message-position\0"
        chunks = marked.render(messages=messages, **settings["tokenizer"], **options,
            message_position=lambda: marker).split(marker)
        assert "".join(chunks) == rendered
        if options.get("continue_final_message"):
            rendered = hf.apply_chat_template(messages, tokenize=False, chat_template=source, **options)
        positions, end = [], 0
        for chunk in chunks[:-1]:
            end += len(chunk)
            positions.append(end)
        encoding = tokenizer.encode(rendered, add_special_tokens=False)
        preceding = [0]
        for _, end in encoding.offsets:
            preceding.append(max(preceding[-1], end))
        following = [len(rendered)] * (len(encoding.ids) + 1)
        for index in range(len(encoding.ids) - 1, -1, -1):
            following[index] = min(following[index + 1], encoding.offsets[index][0])
        boundaries = sorted({index for position in positions for index in range(1, len(encoding.ids))
            if preceding[index] <= position <= following[index]})
        cases.append(dict(name=name, messages=messages, options=options, rendered=rendered,
            token_ids=encoding.ids, message_boundaries=boundaries))

    user = dict(role="user", content="Hello 世界 café á 👨‍👩‍👧‍👦")
    add("default", [user])
    add("empty", [dict(role="user", content="")])
    add("system", [dict(role="system", content="Be precise."), user])
    add("system_only", [dict(role="system", content="Be precise.")])
    add("closed_header", [user], add_generation_prompt=False)
    for effort in ["none", "minimal", "low", "medium", "high", "max", 0.0, 0.5, 0.99]:
        add(f"effort_{effort}", [user], reasoning_effort=effort)
    for text in ["Partial café", " repeated repeated "]:
        add("continuation_" + text, [user, dict(role="assistant", content=text)],
            continue_final_message=True, add_generation_prompt=False)
    add("reasoning", [user, dict(role="assistant", content="Answer", reasoning_content="Plan 世界"), user])
    add("blocks", [dict(role="user", content=[dict(type="text", text="Repeated"),
        dict(type="text", text="Repeated"), dict(type="text", text=" café ")])])
    tools = [dict(type="function", function=dict(name="lookup", description="Find things",
        parameters=dict(type="object", properties=dict(query=dict(type="string")))))]
    calls = [dict(id=str(i), type="function", function=dict(name="lookup", arguments=dict(
        query="世界 & \"quoted\"", n=i, yes=True, absent=None))) for i in range(2)]
    add("tools", [user], tools=tools)
    add("tool_history", [user, dict(role="assistant", content="", tool_calls=calls),
        dict(role="tool", tool_call_id="1", content="Second"),
        dict(role="tool", tool_call_id="0", content="First"), user], tools=tools)
    for effort, error in [("unknown", "Unknown reasoning_effort: unknown"), (1.0, "reasoning_effort must be in [0.0, 0.99]")]:
        cases.append(dict(name=f"invalid_{effort}", messages=[user], options=dict(reasoning_effort=effort), error=error))
    with pytest.raises(ValueError, match="sole EOS"):
        prepare_frontend(SNAPSHOT, eos_token_ids=[200010])

    manifest = str(ROOT / "helios/Cargo.toml")
    subprocess.run(["cargo", "build", "--locked", "--quiet", "--manifest-path", manifest,
        "--features", "frontend,test-tools", "--bin", "test_frontend"], check=True)
    target = json.loads(subprocess.check_output(["cargo", "metadata", "--format-version", "1", "--no-deps",
        "--manifest-path", manifest], text=True))["target_directory"]
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    for name in ["tokenizer.json", "tokenizer_config.json"]:
        (runtime / name).symlink_to(SNAPSHOT / name)
    (runtime / "chat_template.jinja").write_text(annotated)
    fingerprints = {name: hashlib.sha256((runtime / name).read_bytes()).hexdigest()
        for name in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]}
    fixture = tmp_path / "chat.json"
    encode_cases, decode_cases = [], []
    for text in [user["content"], "<|message_model|><|content_text|> café <|end_message|>", "", " repeated repeated "]:
        for special in [False, True]:
            ids = hf.encode(text, add_special_tokens=special)
            encode_cases.append(dict(name=f"encode_{text}_{special}", text=text,
                add_special_tokens=special, token_ids=ids))
            decode_cases.append(dict(name=f"decode_{text}_{special}", token_ids=ids,
                skip_special_tokens=special, text=hf.decode(ids, skip_special_tokens=special)))
    fixture.write_text(json.dumps(dict(assets=fingerprints, cases=cases,
        template_defaults=dict(add_generation_prompt=True),
        default_overrides=[["default", {}], ["closed_header", dict(add_generation_prompt=False)]],
        encode_cases=encode_cases, decode_cases=decode_cases)))
    result = subprocess.run([str(Path(target) / "debug/test_frontend"), str(runtime), str(fixture)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
