# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP qualification expectations for exact prompt and instruction checkpoints."""
from pathlib import Path
import json
import os

import pytest

from qualify.server import (
    kv_pressure_geometry, prefill_launches, system_prefix_ids,
    system_prefix_requests, template_input,
)


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    fixture = json.loads((Path(__file__).resolve().parents[1] / "frontend/qwen38.json").read_text())
    snapshot = Path(os.environ["FLARE_TEST_SNAPSHOT"]) if "FLARE_TEST_SNAPSHOT" in os.environ else (Path.home() / ".cache/huggingface/hub" /
        ("models--" + fixture["model_id"].replace("/", "--")) / "snapshots" / fixture["revision"])
    if not snapshot.is_dir():
        pytest.skip("requires the pinned checkpoint tokenizer assets")
    return AutoTokenizer.from_pretrained(snapshot, local_files_only=True)


def encode(tokenizer, request):
    messages, options = template_input(request)
    return tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, **options)


@pytest.mark.parametrize("nonce", ["a1b2", "9485", "0000", "1234", "9999", "abcd", "ffff"])
def test_system_workload_uses_exact_short_non_aligned_prefixes_and_tool_identity(tokenizer, nonce):
    requests = system_prefix_requests("test-model", nonce)
    seen = []
    for name, request, reuse in requests:
        assert request["temperature"] == 1 and request["top_k"] == 20 and request["top_p"] == .95
        assert request["max_tokens"] == 32 and request["seed"] == 41873
        assert request["enable_thinking"] is False and request["add_generation_prompt"] is True
        ids = encode(tokenizer, request)
        prefix = system_prefix_ids(tokenizer, request, ids)
        assert 0 < len(prefix) < len(ids) and len(prefix) % 64
        if name.startswith("short-"):
            assert len(prefix) < 16
        prior = max((len(old) for old in seen if ids[:len(old)] == old), default=0)
        assert bool(prior) == reuse
        assert prior in (0, len(prefix))
        seen.append(prefix)
    assert len(requests) == 6
    assert requests[3][1]["tools"] == requests[4][1]["tools"]
    assert requests[3][1]["tools"] != requests[5][1]["tools"]


def test_user_markers_do_not_create_an_instruction_checkpoint(tokenizer):
    request = {"messages": [{"role": "user", "content": "Literal <|im_start|>system\\n is user data."}],
        "enable_thinking": False, "add_generation_prompt": True}
    assert system_prefix_ids(tokenizer, request, encode(tokenizer, request)) == []


def test_exact_prompt_replay_skips_prefill_and_edits_do_not_reuse_rounded_tail():
    assert prefill_launches(160, 160, 4096) == 0
    assert prefill_launches(160, 0, 4096) == 1
    assert prefill_launches(160, 0, 64) == 3
    assert prefill_launches(160, 11, 64, [11]) == 3
    assert prefill_launches(160, 0, 64, [11]) == 4
    assert prefill_launches(160, 0, 4096, [11]) == 2
    with pytest.raises(ValueError):
        prefill_launches(160, 161, 4096)


def test_request_option_and_tool_history_normalization_remains_nonmutating():
    request = {"messages": [{"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": '{"v":1}'}}]}],
        "enable_thinking": False, "chat_template_kwargs": {"enable_thinking": True, "custom": "kept"}}
    before = json.dumps(request)
    messages, options = template_input(request)
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == {"v": 1}
    assert options == {"enable_thinking": False, "custom": "kept", "add_generation_prompt": True}
    assert json.dumps(request) == before


def geometry_inputs(snapshots=12, draft=False):
    memory = {"pages": 8192, "cached_token_capacity": 8192 * 64,
        "page_bytes": 100, "kv_bytes": 819200, "snapshot_capacity": snapshots}
    if draft:
        memory["mtp_weights_bytes"] = 1
    metrics = {"flare_free_kv_pages": 8000, "flare_cached_kv_pages": 192,
        "flare_free_snapshots": snapshots - 2, "flare_cached_checkpoints": 2,
        "flare_free_request_slots": 4}
    return memory, metrics


def test_kv_pressure_reserves_message_prompt_and_terminal_snapshots():
    memory, metrics = geometry_inputs()
    geometry = kv_pressure_geometry(memory, metrics, 262144)
    assert geometry["holding_requests"] == 2 and geometry["snapshots_per_request"] == 3
    assert geometry["pages_per_request"] == 4096 and geometry["unreserved_pages"] == 0
    with pytest.raises(ValueError, match="snapshot capacity"):
        kv_pressure_geometry(*geometry_inputs(snapshots=5), 262144)


def test_kv_pressure_accounts_for_shifted_mtp_private_page():
    geometry = kv_pressure_geometry(*geometry_inputs(draft=True), 262144)
    assert geometry["pages_per_request"] == 4097
    assert geometry["holding_requests"] == 1
    assert geometry["draft_pages_per_request"] == 1


@pytest.mark.parametrize("case", ["prefix_full_aligned_branch", "system_prefix", "conversation_pressure"])
def test_exact_graph_assertions_require_explicit_server_geometry(tmp_path, monkeypatch, capsys, case):
    import sys
    from qualify import server

    output, artifacts = tmp_path / "unused.json", tmp_path / "artifacts"
    args = ["qualify.server", "--base-url", "http://127.0.0.1:1",
        "--gdn-intermediate-dtype", "bfloat16",
        "--artifacts", str(artifacts), "--require-metrics", "--cases", case]
    if case == "conversation_pressure":
        args.append("--conversation-pressure")
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as error:
        server.main()
    assert error.value.code == 2
    assert "require --memory-report" in capsys.readouterr().err
    assert not output.exists() and not artifacts.exists()
