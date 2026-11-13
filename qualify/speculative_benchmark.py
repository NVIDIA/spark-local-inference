# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record one configured speculation arm through the actual native HTTP service.

Requests come from frozen files. This client neither starts a model nor changes
its policy, cache, sampling parameters, tool history, or generated output budget.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import re
import socket
import threading
import time
import traceback

from qualify.performance import connection, distribution
from qualify.provenance import write_json
from qualify.speculative_workload import (
    MODEL_ID, REVISION, arms, json_bytes, request_groups, validate_artifacts,
)

VOCABULARY = 248320
EOS = {248046, 248044}
_ROUND = re.compile(r'^flare_speculative_(rounds|gpu_nanoseconds|emitted_tokens|offered_drafts|accepted_drafts)_total\{batch="([124])",width="([1-8])"\}$')


def read_sse(response, raw, *, elapsed_ns, limit=32 << 20):
    """Yield complete data events while retaining every received byte and offset."""
    total, start, data = 0, 0, []
    while line := response.readline(1 << 20):
        raw.write(line)
        total += len(line)
        if total > limit:
            raise ValueError("SSE response exceeds byte bound")
        if not line.endswith(b"\n"):
            raise ValueError("SSE line exceeds bound or is truncated")
        stripped = line.rstrip(b"\r\n")
        if not stripped:
            if data:
                text = b"\n".join(data).decode("utf-8")
                payload = text if text == "[DONE]" else json.loads(text)
                yield {"elapsed_ns": elapsed_ns(), "raw_start": start, "raw_end": total,
                       "data_text": text, "data": payload}
                data.clear()
            start = total
        elif stripped.startswith(b"data:"):
            data.append(stripped[5:].removeprefix(b" "))
    if data:
        raise ValueError("SSE stream ended inside an event")


def fixture_json_shape(schema, value):
    """Check the exact type/field/array constraints used by the frozen fixtures."""
    kind = schema["type"]
    allowed = {"type"}
    if kind == "object":
        allowed |= {"properties", "required", "additionalProperties"}
        valid = (isinstance(value, dict) and set(schema["required"]) <= value.keys()
                 and (schema["additionalProperties"] or value.keys() <= schema["properties"].keys()))
        valid = valid and all(fixture_json_shape(schema["properties"][k], v) for k, v in value.items())
    elif kind == "array":
        allowed |= {"items", "minItems", "maxItems"}
        valid = (isinstance(value, list) and len(value) >= schema.get("minItems", 0)
                 and len(value) <= schema.get("maxItems", len(value)))
        valid = valid and all(fixture_json_shape(schema["items"], v) for v in value)
    elif kind == "integer":
        valid = type(value) is int or (type(value) is float and math.isfinite(value) and value.is_integer())
    elif kind == "boolean":
        valid = type(value) is bool
    elif kind == "string":
        valid = isinstance(value, str)
    else:
        raise ValueError(f"unsupported fixture JSON type: {kind}")
    if set(schema) - allowed:
        raise ValueError("fixture JSON constraints exceed the implemented checks")
    return valid


def output_semantics(case, content, calls, finish, generated, *, ended_with_eos=False):
    """Check objective fixture requirements; leave prose and code review explicit."""
    expected = case["expected_output_semantics"]
    checks, errors = {}, []
    checks["not_budget_exhausted"] = finish == "stop" or (finish == "tool_calls" and
        (ended_with_eos or generated < case["request"]["max_tokens"]))
    parsed = None
    if expected["kind"] == "tool_call":
        checks["one_expected_call"] = len(calls) == 1 and calls[0].get("function", {}).get("name") == expected["allowed_function"]
        try:
            parsed = json.loads(calls[0]["function"]["arguments"]) if len(calls) == 1 else None
            checks["expected_arguments"] = parsed == expected["expected_arguments"]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            errors.append(str(error))
            checks["expected_arguments"] = False
        review = False
    elif expected["kind"] == "json_schema":
        try:
            parsed = json.loads(content)
            checks["json_fixture_shape"] = fixture_json_shape(expected["schema"], parsed)
            checks["expected_values"] = isinstance(parsed, dict) and all(parsed.get(k) == v for k, v in expected["expected_values"].items())
            checks["literal_strings"] = isinstance(parsed, dict) and all(isinstance(parsed.get(k), str) and all(s in parsed[k] for s in strings)
                                            for k, strings in expected.get("string_contains", {}).items())
            checks["no_tool_calls"] = not calls
        except (json.JSONDecodeError, ValueError, TypeError) as error:
            errors.append(str(error))
            checks["json_fixture_shape"] = False
        review = bool(set(expected["schema"]["properties"]) - set(expected["expected_values"])
                      - set(expected.get("string_contains", {})))
    else:
        checks["nonempty_text"] = bool(content.strip())
        checks["no_tool_calls"] = not calls
        review = True
    return {"automatic_checks": checks, "automatic_passed": all(checks.values()),
            "manual_review_required": review, "expected": expected, "parsed": parsed, "errors": errors,
            "equivalent_completed_work_established": False}


def analyze_events(events, case):
    ids, core, groups, prompts, finishes, usages, timings, cached, errors = [], [], [], [], [], [], [], [], []
    content, reasoning, calls, models, done = [], [], {}, [], 0
    terminal_seen = False
    for event in events:
        payload = event["data"]
        if payload == "[DONE]":
            done += 1
            continue
        if done:
            errors.append("data after DONE")
        if not isinstance(payload, dict):
            errors.append("SSE data is not an object")
            continue
        if "error" in payload:
            errors.append(payload["error"])
        if "model" in payload:
            models.append(payload["model"])
        if payload.get("prompt_token_ids") is not None:
            prompts.append(payload["prompt_token_ids"])
        if payload.get("usage") is not None:
            usages.append(payload["usage"])
        if payload.get("performance") is not None:
            timings.append(payload["performance"])
        if payload.get("reused_prompt_tokens") is not None:
            cached.append(payload["reused_prompt_tokens"])
        choices = payload.get("choices", [])
        if not isinstance(choices, list):
            errors.append("choices is not an array")
            continue
        for choice in choices:
            if not isinstance(choice, dict) or choice.get("index") != 0:
                errors.append("unexpected choice")
                continue
            finish = choice.get("finish_reason")
            if finish is not None:
                finishes.append(finish)
                terminal_seen = True
            tokens, timestamps = choice.get("token_ids"), choice.get("token_timestamps_ns")
            if tokens is not None:
                if (not isinstance(tokens, list) or any(type(t) is not int or not 0 <= t < VOCABULARY for t in tokens)):
                    errors.append("invalid raw token IDs")
                else:
                    if terminal_seen and tokens:
                        errors.append("raw token IDs at/after terminal")
                    ids.extend(tokens)
                    if tokens:
                        groups.append({"elapsed_ns": event["elapsed_ns"], "token_ids": tokens})
                    if (not isinstance(timestamps, list) or len(timestamps) != len(tokens)
                            or any(type(t) is not int or t < 0 for t in timestamps)):
                        errors.append("missing or misaligned core timestamps")
                    else:
                        core.extend(timestamps)
            elif timestamps is not None:
                errors.append("timestamps without raw IDs")
            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                errors.append("delta is not an object")
                continue
            for key, target in (("content", content), ("reasoning_content", reasoning)):
                if delta.get(key) is not None:
                    if not isinstance(delta[key], str):
                        errors.append(f"non-string {key}")
                    else:
                        if terminal_seen and delta[key]:
                            errors.append("presentation data at/after terminal")
                        target.append(delta[key])
            fragments = delta.get("tool_calls", [])
            if not isinstance(fragments, list):
                errors.append("tool_calls is not an array")
                continue
            for fragment in fragments:
                if not isinstance(fragment, dict) or type(fragment.get("index")) is not int or fragment["index"] < 0:
                    errors.append("invalid tool-call index")
                    continue
                index = fragment["index"]
                call = calls.setdefault(index, {"index": index, "id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                if terminal_seen:
                    errors.append("tool data at/after terminal")
                if fragment.get("type", "function") != "function":
                    errors.append("unexpected tool-call type")
                if fragment.get("id"):
                    if not isinstance(fragment["id"], str) or (call["id"] and call["id"] != fragment["id"]):
                        errors.append("conflicting tool-call ID")
                    else:
                        call["id"] = fragment["id"]
                function = fragment.get("function", {})
                if not isinstance(function, dict):
                    errors.append("invalid tool function")
                    continue
                for key in ("name", "arguments"):
                    part = function.get(key, "")
                    if not isinstance(part, str):
                        errors.append("non-string tool-call fragment")
                    else:
                        call["function"][key] += part
    finish = finishes[0] if len(finishes) == 1 else None
    usage = usages[0] if len(usages) == 1 and isinstance(usages[0], dict) else {}
    timing = timings[0] if len(timings) == 1 and isinstance(timings[0], dict) else {}
    reuse = cached[0] if len(cached) == 1 else None
    prompt_count = len(case["prompt_tokens"])
    complete_core = bool(ids) and len(core) == len(ids) and all(a <= b for a, b in zip(core, core[1:]))
    ingress, enqueue = timing.get("ingress_ns"), timing.get("enqueue_ns")
    timing_order = (complete_core and type(ingress) is int and type(enqueue) is int and 0 <= ingress <= enqueue <= core[0])
    checks = {"no_stream_error": not errors, "done_once": done == 1, "done_last": bool(events) and events[-1]["data"] == "[DONE]",
        "exact_prompt_ids": len(prompts) == 1 and prompts[0] == case["prompt_tokens"],
        "prompt_ids_first_event": bool(events) and isinstance(events[0]["data"], dict) and events[0]["data"].get("prompt_token_ids") == case["prompt_tokens"],
        "model_identity": bool(models) and all(model == MODEL_ID for model in models),
        "raw_count_within_budget": 0 < len(ids) <= case["request"]["max_tokens"],
        "one_valid_terminal": len(finishes) == 1 and finish in ("stop", "length", "tool_calls"),
        "length_uses_budget": finish != "length" or len(ids) == case["request"]["max_tokens"],
        "no_tokens_after_eos": all(token not in EOS for token in ids[:-1]),
        "usage_exact": len(usages) == 1 and all(type(usage.get(k)) is int and usage[k] == v for k, v in
            {"prompt_tokens": prompt_count, "completion_tokens": len(ids), "total_tokens": prompt_count + len(ids)}.items()),
        "complete_core_timestamps": complete_core, "native_timing_order": timing_order,
        "single_timing_metadata": len(timings) == 1,
        "cache_count_valid": type(reuse) is int and 0 <= reuse <= prompt_count and timing.get("reused_prompt_tokens") == reuse,
        "monotonic_delivery": all(a["elapsed_ns"] <= b["elapsed_ns"] for a, b in zip(events, events[1:])),
        "tool_indices_contiguous": sorted(calls) == list(range(len(calls)))}
    tool_calls = [calls[i] for i in sorted(calls)]
    if tool_calls:
        checks["tools_match_terminal"] = finish == "tool_calls"
        checks["tool_calls_complete"] = all(c["id"] and c["function"]["name"] and c["function"]["arguments"] for c in tool_calls)
    elif finish == "tool_calls":
        checks["tools_match_terminal"] = False
    singleton = bool(groups) and all(len(group["token_ids"]) == 1 for group in groups)
    delivery_itl = [b["elapsed_ns"] - a["elapsed_ns"] for a, b in zip(groups, groups[1:])] if singleton else []
    core_itl = [b - a for a, b in zip(core, core[1:])] if complete_core else []
    text = "".join(content)
    return {"passed": all(checks.values()), "checks": checks, "errors": errors,
        "generated_token_ids": ids, "core_token_timestamps_ns": core, "token_groups": groups,
        "content": text, "reasoning_content": "".join(reasoning), "tool_calls": tool_calls,
        "usage": usage, "finish_reason": finish, "performance": timing, "reused_prompt_tokens": reuse,
        "ended_with_eos": bool(ids) and ids[-1] in EOS, "output_budget_reached": len(ids) == case["request"]["max_tokens"],
        "delivery_ttft_ns": groups[0]["elapsed_ns"] if groups else None,
        "core_ttft_ns": core[0] - ingress if timing_order else None,
        "delivery_itl_exact": singleton, "delivery_itl_ns": delivery_itl, "core_itl_ns": core_itl,
        "delivery_group_itl_ns": [b["elapsed_ns"] - a["elapsed_ns"] for a, b in zip(groups, groups[1:])],
        "coalesced_groups": sum(len(group["token_ids"]) > 1 for group in groups),
        "output_semantics": output_semantics(case, text, tool_calls, finish, len(ids),
                                             ended_with_eos=bool(ids) and ids[-1] in EOS)}


def stream_request(base_url, case, payload, folder, *, timeout, barrier):
    if payload != json_bytes(case["request"]) or hashlib.sha256(payload).hexdigest() != case["request_sha256"]:
        raise ValueError("HTTP payload differs from frozen request bytes")
    folder.mkdir()
    (folder / "request.json").write_bytes(payload)
    result = {"case": case["id"], "request_sha256": case["request_sha256"], "events": []}
    conn, prefix = connection(base_url, timeout)
    start, timer, transport = None, None, None
    expired = threading.Event()

    def abort():
        expired.set()
        if transport is not None:
            try:
                transport.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    try:
        connecting = time.perf_counter_ns()
        conn.connect()
        transport = conn.sock
        result["connection_ns"] = time.perf_counter_ns() - connecting
        barrier.wait(timeout=timeout)
        start = time.perf_counter_ns()
        result["request_start_perf_counter_ns"] = start
        timer = threading.Timer(timeout, abort)
        timer.daemon = True
        timer.start()
        conn.request("POST", prefix + "/v1/chat/completions", body=payload,
                     headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
        response = conn.getresponse()
        result.update(status=response.status, headers=response.getheaders(), headers_ns=time.perf_counter_ns() - start)
        with (folder / "response.sse").open("wb") as raw:
            if response.status != 200:
                raw.write(response.read(8 << 20))
                raise ValueError(f"HTTP status {response.status}")
            if not dict((k.lower(), v) for k, v in response.getheaders()).get("content-type", "").startswith("text/event-stream"):
                raw.write(response.read(8 << 20))
                raise ValueError("response is not text/event-stream")
            for event in read_sse(response, raw, elapsed_ns=lambda: time.perf_counter_ns() - start):
                result["events"].append(event)
        if expired.is_set():
            raise TimeoutError("absolute request deadline exceeded")
    except Exception as error:
        barrier.abort()
        result["client_error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    finally:
        if timer is not None:
            timer.cancel()
            timer.join()
        conn.close()
        result["elapsed_ns"] = time.perf_counter_ns() - start if start is not None else None
        result["deadline_expired"] = expired.is_set()
        if expired.is_set() and "client_error" not in result:
            result["client_error"] = "TimeoutError: absolute request deadline exceeded"
    result["analysis"] = analyze_events(result["events"], case)
    if "client_error" in result:
        result["analysis"]["checks"]["client_completed"] = False
        result["analysis"]["passed"] = False
    write_json(folder / "observation.json", result)
    return result


def read_metrics(base_url, path, timeout):
    conn, prefix = connection(base_url, timeout)
    try:
        conn.request("GET", prefix + "/metrics")
        response = conn.getresponse()
        raw = response.read(8 << 20)
        path.write_bytes(raw)
        if response.status != 200:
            raise ValueError(f"metrics HTTP status {response.status}")
        values = {}
        for line in raw.decode().splitlines():
            if line and not line.startswith("#"):
                key, value = line.rsplit(" ", 1)
                number = int(value) if value.isdigit() else float(value)
                if not math.isfinite(number) or key in values:
                    raise ValueError("duplicate or non-finite native metric")
                values[key] = number
        return values
    finally:
        conn.close()


def is_idle(values):
    return all(values.get(key) == expected for key, expected in {
        "flare_active_requests": 0, "flare_pending_requests": 0,
        "flare_retiring_request_slots": 0, "flare_free_request_slots": 4}.items())


def idle_metrics(base_url, folder, label, timeout):
    deadline = time.monotonic() + min(30, timeout)
    attempt = 0
    while True:
        values = read_metrics(base_url, folder / f"metrics-{label}-{attempt:03}.txt", min(timeout, 10))
        if is_idle(values):
            return values
        if time.monotonic() >= deadline:
            raise TimeoutError("native scheduler did not become idle within bounded wait")
        attempt += 1
        time.sleep(0.05)


def validate_arm(memory, arm):
    actual = bool(memory.get("speculative_widths"))
    if actual != arm["mtp"]:
        raise ValueError("actual prepared speculative catalog differs from arm")
    if arm["mtp"]:
        for key in ("speculative_fixed_width",):
            if key not in memory or memory[key] != arm[key] or type(memory[key]) is not type(arm[key]):
                raise ValueError(f"actual prepared {key} differs from arm")
        widths = memory["speculative_widths"]
        if widths != list(range(1, 9)):
            raise ValueError("paired workload requires the complete bounded width catalog")
        expected = {(b, w) for b in (1, 2, 4) for w in widths if b * w <= 8}
        if {tuple(shape) for shape in memory.get("speculative_shapes", [])} != expected:
            raise ValueError("actual speculative B/W catalog differs")
    if memory.get("tuning", {}).get("namespace", {}).get("checkpoint") != REVISION:
        raise ValueError("memory report checkpoint identity differs")


def metric_delta(before, after):
    counters = {key: after[key] - value for key, value in before.items()
                if "_total" in key and key in after}
    if any(value < 0 for value in counters.values()) or any(key not in after for key in before):
        raise ValueError("native counters reset or disappeared during group")
    return counters


def round_metrics(delta):
    shapes = {}
    for key, value in delta.items():
        if match := _ROUND.fullmatch(key):
            metric, batch, width = match.groups()
            row = shapes.setdefault((int(batch), int(width)), {"batch": int(batch), "width": int(width)})
            row[metric] = value
    return {"shapes": [row for _, row in sorted(shapes.items()) if any(value for key, value in row.items() if key not in ("batch", "width"))],
            "source_depth_outcomes": {key: value for key, value in delta.items() if key.startswith("flare_speculative_drafts_total{")},
            "graph_launches": {key: value for key, value in delta.items() if key.startswith("flare_graph_launches_total{")},
            "scope": "Completed graph capacity B/W, not live-lane count. Duration covers the full speculative graph and excludes transfers; separate draft/verify/commit durations are unavailable."}


def summarize_group(group, observations, before, after, memory):
    delta = metric_delta(before, after)
    count = len(observations)
    checks = {"requests_observed": count == len(group["case_ids"]),
              "http_integrity": all(o["analysis"]["passed"] for o in observations),
              "admitted_count": delta.get("flare_requests_admitted_total") == count,
              "completed_count": delta.get("flare_requests_completed_total") == count,
              "no_runtime_failure": delta.get("flare_requests_failed_total") == 0,
              "no_cancellation": delta.get("flare_requests_cancelled_total") == 0,
              "idle_after": is_idle(after),
              "pages_conserved": after.get("flare_free_kv_pages", -1) + after.get("flare_cached_kv_pages", -1) == memory["pages"],
              "snapshots_conserved": after.get("flare_free_snapshots", -1) + after.get("flare_cached_checkpoints", -1) == memory["snapshot_capacity"]}
    analyses = [o["analysis"] for o in observations]
    reuse = [a["reused_prompt_tokens"] for a in analyses]
    checks["reuse_counter_matches_requests"] = all(type(v) is int for v in reuse) and delta.get("flare_cache_reused_tokens_total") == sum(reuse)
    spans = [(o.get("request_start_perf_counter_ns"), o.get("elapsed_ns")) for o in observations]
    span = max(s + e for s, e in spans) - min(s for s, _ in spans) if all(s is not None and e is not None for s, e in spans) else None
    emitted = sum(len(a["generated_token_ids"]) for a in analyses)
    speculative = round_metrics(delta)
    if memory.get("speculative_widths"):
        shapes = speculative["shapes"]
        checks["speculative_shape_band"] = all(row["batch"] * row["width"] <= 8 for row in shapes)
        checks["completed_rounds_match_launches"] = sum(row.get("rounds", 0) for row in shapes) == delta.get('flare_graph_launches_total{kind="speculative"}')
        checks["speculative_emissions_match_raw_ids"] = sum(row.get("emitted_tokens", 0) for row in shapes) + count == emitted
    semantics_pass = all(a["output_semantics"]["automatic_passed"] for a in analyses)
    return {**group, "passed": all(checks.values()) and semantics_pass,
            "transport_passed": all(checks.values()), "checks": checks, "metrics_delta": delta,
            "speculation": speculative, "reused_prompt_tokens": reuse,
            "actual_output_tokens": emitted, "end_to_end_group_span_ns": span,
            "observed_output_tokens_per_second": emitted * 1e9 / span if span and checks["http_integrity"] else None,
            "delivery_ttft_ns": distribution([a["delivery_ttft_ns"] for a in analyses if a["delivery_ttft_ns"] is not None]),
            "core_ttft_ns": distribution([a["core_ttft_ns"] for a in analyses if a["core_ttft_ns"] is not None]),
            "end_to_end_request_ns": distribution([o["elapsed_ns"] for o in observations if o.get("elapsed_ns") is not None]),
            "finish_reasons": [a["finish_reason"] for a in analyses],
            "output_budget_reached": [a["output_budget_reached"] for a in analyses],
            "automatic_semantic_checks_pass": semantics_pass,
            "manual_review_cases": [o["case"] for o in observations if o["analysis"]["output_semantics"]["manual_review_required"]]}


def measurement_summary(workload, observations, groups):
    cases = {case["id"]: case for case in workload["cases"]}
    result = {}
    for category in dict.fromkeys(case["category"] for case in workload["cases"]):
        records = [o for o in observations if cases[o["case"]]["category"] == category]
        analyses = [o["analysis"] for o in records]
        category_groups = [g for g in groups if g["category"] == category]
        totals = {}
        for group in category_groups:
            for key, value in group["metrics_delta"].items():
                totals[key] = totals.get(key, 0) + value
        result[category] = {
            "expected_requests": sum(case["category"] == category for case in workload["cases"]),
            "observed_requests": len(records),
            "transport_passed_requests": sum(a["passed"] for a in analyses),
            "automatic_semantic_passed_requests": sum(a["output_semantics"]["automatic_passed"] for a in analyses),
            "manual_review_cases": [o["case"] for o in records if o["analysis"]["output_semantics"]["manual_review_required"]],
            "actual_output_tokens": [len(a["generated_token_ids"]) for a in analyses],
            "finish_reasons": [a["finish_reason"] for a in analyses],
            "reused_prompt_tokens": [a["reused_prompt_tokens"] for a in analyses],
            "end_to_end_request_ns": distribution([o["elapsed_ns"] for o in records if o.get("elapsed_ns") is not None]),
            "delivery_ttft_ns": distribution([a["delivery_ttft_ns"] for a in analyses if a["delivery_ttft_ns"] is not None]),
            "core_ttft_ns": distribution([a["core_ttft_ns"] for a in analyses if a["core_ttft_ns"] is not None]),
            "delivery_itl_ns": distribution([v for a in analyses for v in a["delivery_itl_ns"]]),
            "core_itl_ns": distribution([v for a in analyses for v in a["core_itl_ns"]]),
            "coalesced_delivery_groups": sum(a["coalesced_groups"] for a in analyses),
            "observed_group_output_tokens_per_second": distribution([g["observed_output_tokens_per_second"] for g in category_groups
                                                                     if g["observed_output_tokens_per_second"] is not None]),
            "speculation": round_metrics(totals),
        }
    return result


def run(args):
    workload = validate_artifacts(args.workload)
    arm = next(a for a in arms() if a["id"] == args.arm)
    memory = json.loads(args.memory_report.read_text())
    validate_arm(memory, arm)
    args.output.mkdir(parents=True, exist_ok=False)
    measurements = {"arm": arm, "concurrency_ceiling": args.concurrency,
                    "repetition": args.repetition, "base_url": args.base_url,
                    "groups": [], "observations": [], "errors": [], "passed": False}
    cases = {c["id"]: c for c in workload["cases"]}
    groups = request_groups(workload, args.concurrency)
    try:
        initial = idle_metrics(args.base_url, args.output, "initial", args.timeout)
        measurements["initial_metrics"] = initial
        expected_history = int(arm.get("history_drafts", False))
        if initial.get("flare_history_drafts_enabled") != expected_history:
            raise ValueError("runtime history_drafts differs from the selected benchmark arm")
        if (initial.get("flare_cached_checkpoints") != 0 or initial.get("flare_cached_kv_pages") != 0
                or initial.get("flare_requests_admitted_total") != 0):
            raise ValueError("paired workload requires a fresh unused server/cache for this arm/concurrency trial")
        for index, group in enumerate(groups):
            folder = args.output / f"group-{index:03}-{group['stage']}"
            folder.mkdir()
            before = idle_metrics(args.base_url, folder, "before", args.timeout)
            barrier = threading.Barrier(group["concurrency"])

            def submit(identifier):
                case = cases[identifier]
                payload = (args.workload / "cases" / identifier / "request.json").read_bytes()
                return stream_request(args.base_url, case, payload, folder / identifier, timeout=args.timeout, barrier=barrier)

            with ThreadPoolExecutor(max_workers=group["concurrency"]) as pool:
                results = list(pool.map(submit, group["case_ids"]))
            measurements["observations"].extend(results)
            after = idle_metrics(args.base_url, folder, "after", args.timeout)
            summary = summarize_group(group, results, before, after, memory)
            summary["category"] = cases[group["case_ids"][0]]["category"]
            measurements["groups"].append(summary)
            write_json(folder / "group.json", {**summary, "metrics_before": before, "metrics_after": after})
            write_json(args.output / "measurements.json", measurements)
        measurements["final_metrics"] = idle_metrics(args.base_url, args.output, "final", args.timeout)
        complete = len(measurements["observations"]) == len(cases) and len(measurements["groups"]) == len(groups)
        measurements["transport_passed"] = complete and all(g["transport_passed"] for g in measurements["groups"])
        measurements["passed"] = complete and all(g["passed"] for g in measurements["groups"])
        measurements["automatic_semantic_checks_pass"] = all(g["automatic_semantic_checks_pass"] for g in measurements["groups"])
        measurements["manual_review_cases"] = [c for g in measurements["groups"] for c in g["manual_review_cases"]]
        measurements["paired_performance_conclusion"] = None
    except Exception as error:
        measurements["errors"].append(f"{type(error).__name__}: {error}")
        measurements["traceback"] = traceback.format_exc()
        measurements["passed"] = False
    finally:
        measurements["measurements"] = measurement_summary(workload, measurements["observations"], measurements["groups"])
        write_json(args.output / "measurements.json", measurements)
    return measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--arm", choices=[a["id"] for a in arms()], required=True)
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--memory-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if args.timeout <= 0 or not math.isfinite(args.timeout) or args.repetition < 0:
        parser.error("timeout must be positive; repetition must be nonnegative")
    measurements = run(args)
    print(json.dumps({"passed": measurements["passed"], "requests": len(measurements["observations"]),
                      "transport_passed": measurements.get("transport_passed", False),
                      "automatic_semantic_checks_pass": measurements.get("automatic_semantic_checks_pass", False),
                      "groups": len(measurements["groups"]), "output": str(args.output)}))
    raise SystemExit(0 if measurements["passed"] else 1)


if __name__ == "__main__":
    main()
