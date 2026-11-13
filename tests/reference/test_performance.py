# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for exact workload geometry and token-aware performance accounting."""

import io
import json
import threading
import zlib
from copy import deepcopy

import pytest

from qualify import performance as p
from qualify.replay_floor import decode_groups


class Tokenizer:
    def encode(self, text, **_):
        return [zlib.crc32(word.encode()) % p.VOCABULARY for word in text.split()]

    def apply_chat_template(self, messages, *, tokenize, **options):
        text = " ".join(f'{m["role"]}: {m["content"]}' for m in messages)
        if options.get("add_generation_prompt"):
            text += " assistant:"
        return self.encode(text) if tokenize else text


def test_complete_matrix_and_exact_prefix_geometry():
    corpus = p.build_corpus(Tokenizer(), repetitions=5, nonce="frozen123")
    assert len(corpus["groups"]) == 41
    assert len(corpus["cases"]) == 64
    assert sum(len(c["prompt_tokens"]) for c in corpus["cases"]) == 411703
    assert len(decode_groups(corpus)) == 15
    for case in corpus["cases"]:
        request = case["request"]
        assert request["temperature"] == 1 and request["top_k"] == 20 and request["top_p"] == .95
    for group in corpus["groups"]:
        if group["kind"].startswith("decode"):
            assert group["batch"] in (1, 2, 4)
    cases = {c["id"]: c for c in corpus["cases"]}
    for i in range(5):
        seed, hit = cases[f"prefix-seed-r{i}"], cases[f"prefix-hit-r{i}"]
        assert len(hit["prompt_tokens"]) == 4107
        assert hit["prompt_tokens"][:4096] == seed["prompt_tokens"]
        assert hit["request"]["continue_final_message"] is True
        assert hit["request"]["add_generation_prompt"] is False


@pytest.mark.parametrize("mutation", ["missing_group", "modified_tokens", "modified_request", "wrong_length", "shared_cold"])
def test_corpus_rejects_incomplete_or_changed_workload(mutation):
    corpus = p.build_corpus(Tokenizer(), repetitions=1, nonce="frozen123")
    if mutation == "missing_group":
        corpus["groups"].pop()
    elif mutation == "modified_tokens":
        corpus["cases"][0]["prompt_tokens"][70] += 1
    elif mutation == "modified_request":
        corpus["cases"][0]["request"]["max_tokens"] = 255
    elif mutation == "wrong_length":
        corpus["cases"][0]["prompt_tokens"].pop()
    else:
        a, b = corpus["cases"][:2]
        b["prompt_tokens"][:64] = a["prompt_tokens"][:64]
        b["prompt_tokens_sha256"] = p.sha_bytes(p.token_bytes(b["prompt_tokens"]))
    with pytest.raises(ValueError):
        p.validate_corpus(corpus)


def observations(*, groups=((10,), (11,), (12,), (13,)), core=True, cached=0):
    case = {"prompt_tokens": [1, 2, 3], "output_tokens": 4, "expected_cache": "cold"}
    events = [{"elapsed_ns": 100, "data": {"prompt_token_ids": [1, 2, 3], "choices": [
        {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}}]
    index = 0
    for offset, tokens in enumerate(groups):
        choice = {"index": 0, "delta": {}, "token_ids": list(tokens), "finish_reason": None}
        if core:
            choice["token_timestamps_ns"] = [1000 + 7 * (index + i) for i in range(len(tokens))]
        index += len(tokens)
        events.append({"elapsed_ns": 110 + offset * 10, "data": {"choices": [choice]}})
        # Parser output carries text without repeating the raw token IDs.
        events.append({"elapsed_ns": 111 + offset * 10, "data": {"choices": [
            {"index": 0, "delta": {"content": "x"}, "finish_reason": None}]}})
    terminal = {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
                "reused_prompt_tokens": cached}
    if core:
        terminal["performance"] = {"ingress_ns": 900, "enqueue_ns": 950, "reused_prompt_tokens": cached}
    events.append({"elapsed_ns": 160, "data": terminal})
    events.append({"elapsed_ns": 170, "data": {"choices": [], "usage": {
        "prompt_tokens": 3, "completion_tokens": index, "total_tokens": 3 + index,
        "prompt_tokens_details": {"cached_tokens": cached}}}})
    events.append({"elapsed_ns": 180, "data": "[DONE]"})
    return case, events


def test_metadata_only_token_events_and_parser_deltas_are_not_double_counted():
    case, events = observations()
    result = p.analyze_events(events, case, backend="flare")
    assert result["passed"]
    assert result["generated_token_ids"] == [10, 11, 12, 13]
    assert result["delivery_ttft_ns"] == 110
    assert result["delivery_itl_ns"] == [10, 10, 10]
    assert result["core_itl_ns"] == [7, 7, 7]
    assert result["core_ttft_ns"] == 100


def test_coalesced_chunks_never_create_per_token_arrival_times():
    case, events = observations(groups=((10, 11), (12, 13)), core=False)
    result = p.analyze_events(events, case, backend="http")
    assert result["passed"]
    assert result["delivery_itl_exact"] is False
    assert result["delivery_itl_ns"] == []
    assert result["delivery_group_itl_ns"] == [10]
    assert result["coalesced_groups"] == 2
    assert result["core_itl_ns"] == []


def test_coalesced_delivery_can_still_have_exact_native_core_intervals():
    case, events = observations(groups=((10, 11), (12, 13)))
    result = p.analyze_events(events, case, backend="flare")
    assert result["passed"]
    assert result["delivery_itl_ns"] == []
    assert result["core_itl_ns"] == [7, 7, 7]


@pytest.mark.parametrize("mutation,check", [
    ("short", "exact_output_length"), ("eos", "finish_length"),
    ("prompt", "prompt_ids_equal"), ("usage", "usage_exact"),
    ("missing_core", "native_core_timestamps"), ("partial_core", "valid_core_timestamps"),
    ("nonmonotonic", "valid_core_timestamps"), ("cached", "cold_cache"),
    ("stream_error", "no_stream_error"), ("no_done", "terminal_done"),
])
def test_failed_conditions_are_retained(mutation, check):
    case, events = observations(groups=((10,), (11,)) if mutation == "short" else ((10,), (11,), (12,), (13,)),
                                core=mutation != "missing_core", cached=64 if mutation == "cached" else 0)
    if mutation == "eos":
        events[-3]["data"]["choices"][0]["finish_reason"] = "stop"
    elif mutation == "prompt":
        events[0]["data"]["prompt_token_ids"] = [1, 2, 4]
    elif mutation == "usage":
        events[-2]["data"]["usage"]["completion_tokens"] = 3
    elif mutation == "partial_core":
        del events[1]["data"]["choices"][0]["token_timestamps_ns"]
    elif mutation == "nonmonotonic":
        events[3]["data"]["choices"][0]["token_timestamps_ns"] = [999]
    elif mutation == "stream_error":
        events.insert(-1, {"elapsed_ns": 175, "data": {"error": {"message": "failure"}}})
    elif mutation == "no_done":
        events.pop()
    result = p.analyze_events(events, case, backend="flare")
    assert not result["passed"]
    assert not result["checks"][check]
    assert result["generated_token_ids"]


def test_http_prefix_reports_real_hybrid_cache_granularity():
    case, events = observations(cached=3040, core=False)
    case["expected_cache"] = "prefix-4096"
    case["prompt_tokens"] = [1] * 4107
    events[0]["data"]["prompt_token_ids"] = case["prompt_tokens"]
    events[-2]["data"]["usage"].update(prompt_tokens=4107, total_tokens=4111)
    result = p.analyze_events(events, case, backend="http")
    assert result["passed"]
    assert result["reused_prompt_tokens"] == 3040
    assert result["checks"]["prefix_reused"]
    assert "prefix_4096_reused" not in result["checks"]


def test_nearest_rank_p99_and_empty_distributions():
    assert p.distribution([])["p99"] is None
    result = p.distribution(list(range(1, 101)))
    assert result == {"count": 100, "median": 50.5, "p99": 99, "minimum": 1, "maximum": 100}


def test_concurrent_throughput_excludes_first_generated_token_per_request():
    case, events = observations()
    a = p.analyze_events(events, case, backend="flare")
    b = deepcopy(a)
    group = {"id": "group", "kind": "decode-b2", "batch": 2, "measured": True}
    summary = p.group_summary(group, [
        {"case": "a", "request_start_perf_counter_ns": 1000, "analysis": a},
        {"case": "b", "request_start_perf_counter_ns": 1005, "analysis": b}])
    assert summary["generated_after_first_tokens"] == 6
    assert summary["delivery_decode_span_ns"] == 35
    assert summary["delivery_decode_tokens_per_second"] == 6e9 / 35
    assert summary["request_release_skew_ns"] == 5


@pytest.mark.parametrize("summaries,expected", [([], False), ([False], False), ([True], True)])
def test_measurement_acceptance_includes_group_cache_checks_and_coverage(summaries, expected):
    case, events = observations()
    analysis = p.analyze_events(events, case, backend="flare")
    assert analysis["passed"]
    corpus = {"groups": [{"id": "group", "kind": "decode-b1", "measured": True}],
              "cases": [{"id": "case", "group": "group"}]}
    groups = [{"kind": "decode-b1", "passed": passed, "delivery_decode_tokens_per_second": 1}
              for passed in summaries]
    result = p.summarize_measurements(corpus, groups, [{"case": "case", "analysis": analysis}])
    assert result["decode-b1"]["passed"] is expected
    assert result["decode-b1"]["delivery_itl_ns"]["count"] == 3


class Response(io.BytesIO):
    status = 200

    def getheaders(self):
        return [("Content-Type", "text/event-stream")]


class Connection:
    def __init__(self, response):
        self.response = response
        self.calls = []
        self.closed = False

    def connect(self):
        pass

    def request(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    def getresponse(self):
        return self.response

    def close(self):
        self.closed = True


@pytest.mark.parametrize("backend", ["flare", "http"])
def test_client_preserves_raw_stream_and_sends_only_supported_observation_fields(tmp_path, monkeypatch, backend):
    case, events = observations(core=backend == "flare")
    case.update(id="request", request={"model": p.MODEL_ID, "max_tokens": 4})
    raw = b": keepalive\n\n" + b"".join(b"data: " + (e["data"].encode() if isinstance(e["data"], str)
                else json.dumps(e["data"]).encode()) + b"\n\n" for e in events)
    conn = Connection(Response(raw))
    monkeypatch.setattr(p, "connection", lambda *_: (conn, ""))
    observation = p.stream_request("http://unused", case, tmp_path / "request", backend=backend,
                                    model=None, timeout=1, barrier=threading.Barrier(1))
    assert observation["analysis"]["passed"]
    assert (tmp_path / "request/response.sse").read_bytes() == raw
    body = json.loads(conn.calls[0][1]["body"])
    assert body["return_token_ids"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body.get("return_token_timestamps", False) == (backend == "flare")
    assert conn.closed


def test_truncated_sse_is_saved_with_failure(tmp_path, monkeypatch):
    case, _ = observations()
    case.update(id="request", request={"model": p.MODEL_ID})
    raw = b'data: {"choices":[]}'
    conn = Connection(Response(raw))
    monkeypatch.setattr(p, "connection", lambda *_: (conn, ""))
    result = p.stream_request("http://unused", case, tmp_path / "request", backend="flare",
                              model=None, timeout=1, barrier=threading.Barrier(1))
    assert not result["analysis"]["passed"]
    assert "truncated" in result["client_error"]
    assert (tmp_path / "request/response.sse").read_bytes() == raw
