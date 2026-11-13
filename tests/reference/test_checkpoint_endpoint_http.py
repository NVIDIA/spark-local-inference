# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact-token HTTP checkpoint fixtures and counter acceptance conditions."""

import copy

import pytest

from qualify.checkpoint_http import (
    PROMPT_LENGTHS, counter_checks, endpoint_request, observed_tokens,
    prompt_ids, terminal_branch,
)
from qualify.server import common_prefix, system_prefix_ids
from .test_exact_checkpoint_http import tokenizer as tokenizer


@pytest.mark.parametrize("nonce", ["000000000000", "abcdef123456", "999999999999"])
def test_five_residues_render_exact_independent_native_prompt_inputs(tokenizer, nonce):
    seen = []
    for length in PROMPT_LENGTHS:
        request, ids = endpoint_request(tokenizer, "model", nonce, f"endpoint-{length}", length)
        assert len(ids) == length and ids == prompt_ids(tokenizer, request)
        assert request["temperature"] == 1 and request["top_k"] == 20 and request["top_p"] == .95
        assert request["seed"] == 41873 and request["max_tokens"] == 1
        assert request["return_token_ids"] is True and request["grammar"] == 'start: "!"'
        assert not system_prefix_ids(tokenizer, request, ids)
        assert all(common_prefix(previous, ids) < min(len(previous), len(ids)) for previous in seen)
        seen.append(ids)
    assert [len(ids) % 64 for ids in seen] == [0, 1, 2, 3, 63]


def test_append_uses_observed_consumed_generation_excludes_pending_eos_and_preserves_source(tokenizer):
    request, ids = endpoint_request(tokenizer, "model", "abcdef123456", "generated-terminal", 194, max_tokens=2)
    original = copy.deepcopy(request)
    literal, = tokenizer.encode("!", add_special_tokens=False)
    generated = [literal, tokenizer.eos_token_id]
    branches = []
    for suffix in ("A", "B"):
        branch, branch_ids, consumed = terminal_branch(
            tokenizer, request, ids, generated, {"role": "assistant", "content": "!"}, suffix)
        assert len(consumed) == 195 and consumed == ids + [literal]
        assert branch_ids[:195] == consumed and len(branch_ids) > 195
        assert branch["max_tokens"] == 1 and branch["messages"][-1]["role"] == "user"
        assert branch["messages"][-2] == {"role": "assistant", "content": "!"}
        branches.append(branch_ids)
    assert request == original
    assert 195 < common_prefix(*branches) < min(map(len, branches))


@pytest.mark.parametrize("mutation", ["token", "reasoning", "tool", "content"])
def test_append_rejects_context_or_presentation_that_cannot_reproduce_consumed_tokens(tokenizer, mutation):
    request, ids = endpoint_request(tokenizer, "model", "abcdef123456", "generated-terminal", 194, max_tokens=2)
    generated = [tokenizer.encode("!", add_special_tokens=False)[0], tokenizer.eos_token_id]
    message = {"content": "!"}
    if mutation == "token":
        generated[0] = tokenizer.encode("?", add_special_tokens=False)[0]
    elif mutation == "reasoning":
        message["reasoning_content"] = "hidden"
    elif mutation == "tool":
        message["tool_calls"] = [{"function": {"name": "unused"}}]
    else:
        message["content"] = "different"
    with pytest.raises(ValueError):
        terminal_branch(tokenizer, request, ids, generated, message, "A")


def response():
    return {"body": {"prompt_token_ids": [5, 6], "choices": [{"token_ids": [0, 248046]}],
                     "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}}}


@pytest.mark.parametrize("field", ["prompt", "missing_ids", "count", "usage", "boolean_id"])
def test_observations_require_native_full_ids_and_exact_usage(field):
    record = response()
    assert observed_tokens(record, [5, 6], 2) == [0, 248046]
    if field == "prompt":
        record["body"]["prompt_token_ids"][1] = 7
    elif field == "missing_ids":
        del record["body"]["choices"][0]["token_ids"]
    elif field == "count":
        record["body"]["choices"][0]["token_ids"].pop()
    elif field == "usage":
        record["body"]["usage"]["completion_tokens"] = 1
    else:
        record["body"]["choices"][0]["token_ids"][0] = False
    with pytest.raises(ValueError, match="observed prompt/generated"):
        observed_tokens(record, [5, 6], 2)


def duplicate_counters():
    return {"flare_requests_admitted_total": 1, "flare_requests_completed_total": 1,
        "flare_requests_failed_total": 0, "flare_requests_cancelled_total": 0,
        "flare_cache_hits_total": 1, "flare_cache_misses_total": 0,
        "flare_cache_reused_tokens_total": 194,
        'flare_graph_launches_total{kind="restore"}': 1,
        'flare_graph_launches_total{kind="init"}': 0,
        'flare_graph_launches_total{kind="prefill"}': 0,
        'flare_graph_launches_total{kind="resume"}': 1,
        'flare_graph_launches_total{kind="snapshot"}': 1,
        "flare_snapshot_publications_total": 0, "flare_prompt_checkpoints_total": 0,
        "flare_message_checkpoints_total": 0, "flare_snapshot_evictions_total": 0}


@pytest.mark.parametrize("key,value,failed", [
    ('flare_graph_launches_total{kind="snapshot"}', 0, "exact_publication"),
    ("flare_snapshot_publications_total", 1, "exact_publication"),
    ("flare_cache_reused_tokens_total", 192, "exact_admission"),
    ('flare_graph_launches_total{kind="prefill"}', 1, "exact_prompt_execution"),
    ("flare_requests_admitted_total", 2, "one_completed_request"),
    ("flare_requests_cancelled_total", 1, "one_completed_request"),
    ("flare_snapshot_evictions_total", 1, "no_eviction"),
])
def test_duplicate_proof_requires_executed_snapshot_without_publication_or_pressure(key, value, failed):
    delta = duplicate_counters()
    options = dict(prompt=194, reused=194, chunk_size=4096, snapshots=1, publications=0, prompts=0)
    assert all(counter_checks(delta, **options).values())
    delta[key] = value
    assert counter_checks(delta, **options)[failed] is False


@pytest.mark.parametrize("case", ["prompt_endpoint_residues", "generated_terminal_prefix"])
def test_endpoint_cli_requires_geometry_before_any_artifact_or_connection(tmp_path, monkeypatch, capsys, case):
    import sys
    from qualify import server

    output, artifacts = tmp_path / "unused.json", tmp_path / "raw"
    monkeypatch.setattr(sys, "argv", ["qualify.server", "--base-url", "http://127.0.0.1:1",
        "--gdn-intermediate-dtype", "bfloat16", "--artifacts", str(artifacts),
        "--require-metrics", "--checkpoint-endpoints", "--cases", case])
    with pytest.raises(SystemExit) as error:
        server.main()
    assert error.value.code == 2 and "require --memory-report" in capsys.readouterr().err
    assert not output.exists() and not artifacts.exists()
