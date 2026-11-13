# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check reusable native token prefixes across assistant reasoning history."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("directory,variable,recipe", [
    ("Qwen3.8-27B-MXFP8", "FLARE_QWEN38_SNAPSHOT", "qwen38"),
    ("NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB", "FLARE_NEMOTRON_CHECKPOINT", "nemotron_h"),
])
def test_native_message_anchor_survives_reasoning_history(directory, variable, recipe, tmp_path):
    from transformers import AutoTokenizer
    from transformers.utils.chat_template_utils import _compile_jinja_template

    from ignition.preparation.frontend import annotated_template_asset

    snapshot = Path(os.environ.get(variable, f"/data/models/{directory}"))
    if not (snapshot / "tokenizer.json").is_file():
        pytest.skip(f"requires {directory} checkpoint frontend assets")
    manifest = str(ROOT / "helios/Cargo.toml")
    subprocess.run(["cargo", "build", "--quiet", "--manifest-path", manifest,
                    "--features", "frontend,test-tools", "--bin", "test_frontend"], check=True)
    target = json.loads(subprocess.check_output([
        "cargo", "metadata", "--format-version", "1", "--no-deps", "--manifest-path", manifest,
    ], text=True))["target_directory"]
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    annotated = annotated_template_asset((snapshot / "chat_template.jinja").read_bytes(),
        ROOT / f"ignition/recipes/{recipe}/assets/chat_template.patch.json")["text"]
    marker = "\0message-position\0"
    marked_template = _compile_jinja_template(
        annotated.replace("{%- mark -%}", "{{- message_position() -}}"))
    options = dict(add_generation_prompt=True, enable_thinking=True)
    history = [dict(role="system", content="Be precise."),
               dict(role="user", content="What is 17 times 23?")]
    cases = []

    def add(name, messages, **overrides):
        case_options = {**options, **overrides}
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, **case_options)
        marked = marked_template.render(messages=messages, **case_options, message_position=lambda: marker)
        chunks = marked.split(marker)
        assert "".join(chunks) == rendered
        positions, position = [], 0
        for chunk in chunks[:-1]:
            position += len(chunk)
            positions.append(position)
        encoding = tokenizer.backend_tokenizer.encode(rendered, add_special_tokens=False)
        prefix_ends = [0]
        for _, end in encoding.offsets:
            prefix_ends.append(max(prefix_ends[-1], end))
        suffix_starts = [len(rendered)] * (len(encoding.ids) + 1)
        for index in range(len(encoding.ids) - 1, -1, -1):
            suffix_starts[index] = min(suffix_starts[index + 1], encoding.offsets[index][0])
        boundaries = sorted({index for position in positions for index in range(1, len(encoding.ids))
            if prefix_ends[index] <= position <= suffix_starts[index]})
        case = dict(name=name, messages=messages, options=case_options, rendered=rendered,
                    token_ids=encoding.ids, message_boundaries=boundaries)
        cases.append(case)
        return case

    initial = add("initial_user", history)
    anchor = initial["message_boundaries"][-1]
    assert 0 < anchor < len(initial["token_ids"])
    for truncate in (None, True, False):
        overrides = {} if truncate is None else dict(truncate_history_thinking=truncate)
        for reasoning in (None, "Multiply 17 by 23."):
            assistant = dict(role="assistant", content="391")
            if reasoning is not None:
                assistant["reasoning_content"] = reasoning
            followup = add(f"followup_{truncate}_{reasoning is not None}", [*history, assistant,
                dict(role="user", content="Explain the calculation.")], **overrides)
            before, after = initial["token_ids"], followup["token_ids"]
            common = next((index for index, pair in enumerate(zip(before, after))
                           if pair[0] != pair[1]), min(len(before), len(after)))
            rewritten = reasoning is None or (recipe == "nemotron_h" and truncate is not False)
            if rewritten:
                assert common < len(before)
            else:
                assert common == len(before)
            assert anchor in followup["message_boundaries"]
            assert anchor <= common
            assert before[:anchor] == after[:anchor]

    assets = tmp_path / "assets"
    assets.mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (assets / name).symlink_to(snapshot / name)
    (assets / "chat_template.jinja").write_text(annotated)
    fixture = tmp_path / "cases.json"
    fixture.write_text(json.dumps(dict(cases=cases, encode_cases=[], decode_cases=[],
        default_overrides=[], assets={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in assets.iterdir()})))
    result = subprocess.run([str(Path(target) / "debug/test_frontend"), str(assets), str(fixture)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["cases"] == 7
