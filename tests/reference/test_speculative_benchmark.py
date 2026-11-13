# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU parser/contracts; fabricated streams are not serving measurements."""

from copy import deepcopy
import io
import json
import threading

import pytest

from qualify import speculative_benchmark as sb
from qualify import speculative_workload as sw


def case(name="chat-capacity"):
    value = next(c for c in sw.definitions() if c["id"] == name)
    value["prompt_tokens"] = [1, 2, 3]
    value["request_sha256"] = sw.value_sha(value["request"])
    return value


def events(*, ids=(10, 248044), finish="stop", text="An answer.", grouped=False, tool_fragments=()):
    def event(at, payload):
        return {"elapsed_ns": at, "data": payload}
    result = [event(20, {"model": sw.MODEL_ID, "prompt_token_ids": [1, 2, 3],
                        "choices": [{"index": 0, "delta": {"role": "assistant"}}]})]
    chunks = [list(ids)] if grouped else [[token] for token in ids]
    offset = 0
    for i, chunk in enumerate(chunks):
        result.append(event(30 + i, {"choices": [{"index": 0, "delta": {}, "token_ids": chunk,
                       "token_timestamps_ns": [1000 + offset + j for j in range(len(chunk))]}]}))
        offset += len(chunk)
    result.append(event(50, {"choices": [{"index": 0, "delta": {"content": text}}]}))
    for fragment in tool_fragments:
        result.append(event(55, {"choices": [{"index": 0, "delta": {"tool_calls": [fragment]}}]}))
    result += [event(60, {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                         "reused_prompt_tokens": 0,
                         "performance": {"ingress_ns": 900, "enqueue_ns": 950, "reused_prompt_tokens": 0}}),
               event(70, {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": len(ids), "total_tokens": 3 + len(ids)}}),
               event(80, "[DONE]")]
    return result


def test_early_eos_is_valid_and_retained_without_a_budget_equality_requirement():
    result = sb.analyze_events(events(), case())
    assert result["passed"]
    assert result["generated_token_ids"] == [10, 248044]
    assert result["ended_with_eos"]
    assert result["finish_reason"] == "stop"
    assert result["usage"]["completion_tokens"] == 2
    assert not result["output_budget_reached"]
    assert result["output_semantics"]["manual_review_required"]
    assert result["output_semantics"]["automatic_passed"]
    assert not result["output_semantics"]["equivalent_completed_work_established"]


def test_coalesced_vector_has_no_invented_delivery_itl_but_keeps_core_times():
    result = sb.analyze_events(events(grouped=True), case())
    assert result["passed"]
    assert not result["delivery_itl_exact"]
    assert result["delivery_itl_ns"] == []
    assert result["core_itl_ns"] == [1]
    assert result["coalesced_groups"] == 1


def test_budget_finish_is_transport_success_but_not_equivalent_completed_work():
    fixture = case()
    fixture["request"]["max_tokens"] = 2
    result = sb.analyze_events(events(ids=(10, 11), finish="length"), fixture)
    assert result["passed"]
    assert result["output_budget_reached"]
    assert not result["output_semantics"]["automatic_passed"]


def test_eos_at_budget_boundary_is_still_a_valid_natural_completion():
    fixture = case()
    fixture["request"]["max_tokens"] = 2
    result = sb.analyze_events(events(), fixture)
    assert result["passed"]
    assert result["output_budget_reached"]
    assert result["ended_with_eos"]
    assert result["output_semantics"]["automatic_passed"]


def test_fragmented_tool_call_is_assembled_without_counting_presentation_as_tokens():
    fragments = [{"index": 0, "id": "call_7", "type": "function", "function": {"name": "read_", "arguments": '{"incident_id":'}},
                 {"index": 0, "function": {"name": "incident", "arguments": '"INC-204"}'}}]
    result = sb.analyze_events(events(text="", finish="tool_calls", tool_fragments=fragments), case("incident-turn-1"))
    assert result["passed"]
    assert result["output_semantics"]["automatic_passed"]
    assert result["tool_calls"] == [{"index": 0, "id": "call_7", "type": "function",
                                     "function": {"name": "read_incident", "arguments": '{"incident_id":"INC-204"}'}}]
    assert result["generated_token_ids"] == [10, 248044]


@pytest.mark.parametrize("text,passed", [
    ('{"service":"billing","severity":2,"retry_limit":1}', True),
    ('{"service":"billing","severity":2.0,"retry_limit":1}', True),
    ('{"service":"search","severity":2,"retry_limit":1}', False),
    ('{"service":"billing","severity":true,"retry_limit":1}', False),
    ('{"service":"billing","severity":2,"retry_limit":1,"extra":0}', False),
    ('{"service":"billing","severity":2}', False),
    ('["billing",2,1]', False),
    ('{"service":', False),
])
def test_json_structure_and_values_are_distinct_from_transport(text, passed):
    result = sb.analyze_events(events(text=text), case("json-incident"))
    assert result["passed"]
    assert result["output_semantics"]["automatic_passed"] is passed


def test_literal_tags_are_checked_as_json_strings_and_not_parsed_as_tools():
    result = sb.analyze_events(events(text='{"text":"<think> <tool_call>","count":2}'), case("json-literal-tags"))
    assert result["passed"]
    assert result["output_semantics"]["automatic_passed"]
    assert result["tool_calls"] == []


def test_freeform_json_plan_steps_still_need_semantic_review():
    result = sb.analyze_events(events(text='{"steps":["test","review"],"approved":false}'), case("json-plan"))
    assert result["passed"]
    assert result["output_semantics"]["automatic_passed"]
    assert result["output_semantics"]["manual_review_required"]


@pytest.mark.parametrize("mutation,check", [
    (lambda e: e[0]["data"].update(prompt_token_ids=[4]), "exact_prompt_ids"),
    (lambda e: e[0]["data"].update(model="wrong"), "model_identity"),
    (lambda e: e[1]["data"]["choices"][0].update(token_timestamps_ns=[]), "complete_core_timestamps"),
    (lambda e: e[2]["data"]["choices"][0].update(token_timestamps_ns=[1]), "complete_core_timestamps"),
    (lambda e: e[-2]["data"]["usage"].update(completion_tokens=3), "usage_exact"),
    (lambda e: e[-3]["data"]["choices"][0].update(finish_reason="length"), "length_uses_budget"),
    (lambda e: e[-3]["data"]["performance"].update(reused_prompt_tokens=1), "cache_count_valid"),
    (lambda e: e.insert(-1, {"elapsed_ns": 75, "data": {"error": {"message": "failed"}}}), "no_stream_error"),
    (lambda e: e.pop(), "done_once"),
    (lambda e: e.append({"elapsed_ns": 81, "data": {"choices": []}}), "done_last"),
])
def test_invalid_stream_contracts_are_retained_as_failures(mutation, check):
    stream = events()
    mutation(stream)
    result = sb.analyze_events(stream, case())
    assert not result["passed"]
    assert not result["checks"][check]


def test_tokens_after_eos_fail_even_if_usage_and_terminal_match():
    result = sb.analyze_events(events(ids=(248044, 10)), case())
    assert not result["passed"]
    assert not result["checks"]["no_tokens_after_eos"]


def test_sse_reader_preserves_comments_multiline_data_and_raw_offsets():
    data = b': keepalive\n\ndata: {"choices":\ndata: []}\r\n\r\ndata: [DONE]\n\n'
    output = io.BytesIO()
    result = list(sb.read_sse(io.BytesIO(data), output, elapsed_ns=lambda: 1))
    assert output.getvalue() == data
    assert [e["data"] for e in result] == [{"choices": []}, "[DONE]"]
    assert result[-1]["raw_end"] == len(data)
    assert data[result[0]["raw_start"]:result[0]["raw_end"]].startswith(b"data:")


def test_truncated_and_overlarge_sse_are_retained_before_error():
    for data, limit, message in [(b'data: {"choices":[]}', 100, "truncated"), (b'data: [DONE]\n\n', 4, "byte bound")]:
        output = io.BytesIO()
        with pytest.raises(ValueError, match=message):
            list(sb.read_sse(io.BytesIO(data), output, elapsed_ns=lambda: 1, limit=limit))
        assert data.startswith(output.getvalue())
        assert output.getvalue()


def memory(arm):
    value = {"tuning": {"namespace": {"checkpoint": sw.REVISION}}, "pages": 100, "snapshot_capacity": 10}
    if arm["mtp"]:
        value.update({k: v for k, v in arm.items() if k not in ("id", "mtp")}, speculative_widths=list(range(1, 9)),
                     speculative_shapes=[[b, w] for b in (1, 2, 4) for w in range(1, 9) if b * w <= 8])
    return value


def test_actual_prepared_arm_must_match_fixed_width_and_catalog():
    for arm in sw.arms():
        actual = memory(arm)
        sb.validate_arm(actual, arm)
        if arm["mtp"]:
            wrong = deepcopy(actual)
            wrong["speculative_fixed_width"] = 9
            with pytest.raises(ValueError, match="speculative_fixed_width"):
                sb.validate_arm(wrong, arm)
            wrong = deepcopy(actual)
            wrong["speculative_shapes"].append([4, 8])
            with pytest.raises(ValueError, match="B/W catalog"):
                sb.validate_arm(wrong, arm)
    with pytest.raises(ValueError, match="catalog differs"):
        sb.validate_arm(memory(sw.arms()[-1]), sw.arms()[0])


def metrics(admitted=0):
    return {"flare_active_requests": 0, "flare_pending_requests": 0,
            "flare_free_request_slots": 4, "flare_retiring_request_slots": 0,
            "flare_free_kv_pages": 90, "flare_cached_kv_pages": 10,
            "flare_free_snapshots": 9, "flare_cached_checkpoints": 1,
            "flare_requests_admitted_total": admitted, "flare_requests_completed_total": admitted,
            "flare_requests_failed_total": 0, "flare_requests_cancelled_total": 0,
            "flare_cache_reused_tokens_total": 0,
            'flare_speculative_rounds_total{batch="4",width="2"}': admitted * 3,
            'flare_speculative_gpu_nanoseconds_total{batch="4",width="2"}': admitted * 1000}


def test_actual_round_shape_counters_are_not_inferred_from_client_concurrency():
    analysis = sb.analyze_events(events(), case())
    observation = {"case": "chat-capacity", "analysis": analysis, "request_start_perf_counter_ns": 1, "elapsed_ns": 100}
    group = {"stage": "chat", "case_ids": ["chat-capacity"], "concurrency": 1}
    result = sb.summarize_group(group, [observation], metrics(), metrics(1), memory(sw.arms()[0]))
    assert result["passed"]
    assert result["concurrency"] == 1
    assert result["speculation"]["shapes"] == [{"batch": 4, "width": 2, "rounds": 3, "gpu_nanoseconds": 1000}]
    assert result["actual_output_tokens"] == 2
    assert result["manual_review_cases"] == ["chat-capacity"]
    with pytest.raises(ValueError, match="reset"):
        sb.metric_delta(metrics(1), metrics())


def test_vector_emission_and_completed_round_counts_are_cross_checked():
    analysis = sb.analyze_events(events(), case())
    observation = {"case": "chat-capacity", "analysis": analysis, "request_start_perf_counter_ns": 1, "elapsed_ns": 100}
    group = {"stage": "chat", "case_ids": ["chat-capacity"], "concurrency": 1}
    before, after = metrics(), metrics(1)
    before['flare_graph_launches_total{kind="speculative"}'] = 0
    after['flare_graph_launches_total{kind="speculative"}'] = 3
    before['flare_speculative_emitted_tokens_total{batch="4",width="2"}'] = 0
    after['flare_speculative_emitted_tokens_total{batch="4",width="2"}'] = 1
    result = sb.summarize_group(group, [observation], before, after, memory(sw.arms()[-1]))
    assert result["passed"]
    after['flare_speculative_emitted_tokens_total{batch="4",width="2"}'] = 2
    result = sb.summarize_group(group, [observation], before, after, memory(sw.arms()[-1]))
    assert not result["passed"]
    assert not result["checks"]["speculative_emissions_match_raw_ids"]


def test_summary_retains_semantic_failures_and_actual_eos_counts():
    fixture = case()
    analysis = sb.analyze_events(events(), fixture)
    observation = {"case": fixture["id"], "analysis": analysis, "request_start_perf_counter_ns": 1, "elapsed_ns": 100}
    result = sb.measurement_summary({"cases": [fixture]}, [observation], [])
    category = result["chat_reasoning"]
    assert category["actual_output_tokens"] == [2]
    assert category["finish_reasons"] == ["stop"]
    assert category["transport_passed_requests"] == 1
    assert category["manual_review_cases"] == [fixture["id"]]


class Response(io.BytesIO):
    status = 200

    def getheaders(self):
        return [("Content-Type", "text/event-stream")]


class Connection:
    sock = None

    def __init__(self, response):
        self.response, self.calls, self.closed = response, [], False

    def connect(self):
        pass

    def request(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    def getresponse(self):
        return self.response

    def close(self):
        self.closed = True


def test_http_writer_uses_exact_frozen_bytes_with_no_model_or_sampling_override(tmp_path, monkeypatch):
    fixture = case()
    payload = sw.json_bytes(fixture["request"])
    raw = b"".join(b"data: " + (e["data"].encode() if isinstance(e["data"], str) else sw.json_bytes(e["data"])) + b"\n\n" for e in events())
    conn = Connection(Response(raw))
    monkeypatch.setattr(sb, "connection", lambda *_: (conn, "/prefix"))
    result = sb.stream_request("http://unused", fixture, payload, tmp_path / "request", timeout=2, barrier=threading.Barrier(1))
    assert result["analysis"]["passed"]
    assert conn.calls[0][0] == ("POST", "/prefix/v1/chat/completions")
    assert conn.calls[0][1]["body"] is payload
    assert (tmp_path / "request/request.json").read_bytes() == payload
    assert (tmp_path / "request/response.sse").read_bytes() == raw
    assert conn.closed


def test_changed_payload_fails_before_connecting(tmp_path, monkeypatch):
    monkeypatch.setattr(sb, "connection", lambda *_: pytest.fail("must not connect"))
    fixture = case()
    with pytest.raises(ValueError, match="frozen request"):
        sb.stream_request("http://unused", fixture, sw.json_bytes(fixture["request"]) + b"\n", tmp_path / "request", timeout=1, barrier=threading.Barrier(1))


def test_partial_network_failure_preserves_events_and_raw_bytes(tmp_path, monkeypatch):
    fixture = case()
    raw = b'data: {"choices":[]}\n\ndata: {"error":'
    conn = Connection(Response(raw))
    monkeypatch.setattr(sb, "connection", lambda *_: (conn, ""))
    result = sb.stream_request("http://unused", fixture, sw.json_bytes(fixture["request"]), tmp_path / "request", timeout=2, barrier=threading.Barrier(1))
    assert not result["analysis"]["passed"]
    assert "truncated" in result["client_error"]
    assert len(result["events"]) == 1
    assert (tmp_path / "request/response.sse").read_bytes() == raw


def test_absolute_deadline_failure_is_preserved_even_if_response_finishes(tmp_path, monkeypatch):
    class Timer:
        def __init__(self, timeout, callback):
            self.callback = callback

        def start(self):
            self.callback()

        def cancel(self):
            pass

        def join(self):
            pass

    fixture = case()
    raw = b"".join(b"data: " + (e["data"].encode() if isinstance(e["data"], str) else sw.json_bytes(e["data"])) + b"\n\n" for e in events())
    conn = Connection(Response(raw))
    monkeypatch.setattr(sb, "connection", lambda *_: (conn, ""))
    monkeypatch.setattr(sb.threading, "Timer", Timer)
    result = sb.stream_request("http://unused", fixture, sw.json_bytes(fixture["request"]), tmp_path / "request", timeout=1, barrier=threading.Barrier(1))
    assert result["deadline_expired"]
    assert not result["analysis"]["passed"]
    assert "TimeoutError" in result["client_error"]
