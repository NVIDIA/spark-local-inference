# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Freeze exact token workloads and measure real HTTP token delivery.

Run ``prepare`` once and reuse its corpus across server configurations. SSE
events containing multiple token IDs remain timestamped groups; their arrival
times never become invented per-token intervals. Native completion timestamps,
when present, form a separate clock domain from client delivery timestamps.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import importlib.metadata
import json
import math
import re
import statistics
import struct
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlsplit

from ignition.preparation.qwen38 import (
    CONFIG_SHA256,
    FRONTEND_SHA256,
    MODEL_ID,
    REVISION,
    TRANSFORMERS_REVISION,
)

from .provenance import digest, write_json

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "flare-http-performance-v1"
VOCABULARY = 248320
SAMPLING = {"temperature": 1, "top_p": .95, "top_k": 20, "min_p": 0,
            "repetition_penalty": 1, "presence_penalty": 0, "frequency_penalty": 0,
            "seed": 42}


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def token_bytes(tokens):
    return struct.pack(f"<{len(tokens)}I", *tokens)


def render(tokenizer, request, *, tokenize):
    options = dict(request["chat_template_kwargs"])
    options.update({key: request[key] for key in ("add_generation_prompt", "continue_final_message")
                    if key in request})
    return tokenizer.apply_chat_template(request["messages"], tokenize=tokenize,
                                         return_dict=False, **options)


def make_prompt(tokenizer, identifier, target, output_tokens, *, continuation=False):
    marker = f"Performance context {identifier}."
    instruction = (marker + " Write a detailed numbered list of at least 400 programming facts."
                   " Continue the list without a conclusion. Context:")
    request = {"model": MODEL_ID, **SAMPLING, "max_tokens": output_tokens,
               "messages": [{"role": "user", "content": instruction}],
               "chat_template_kwargs": {"enable_thinking": False},
               "add_generation_prompt": not continuation}
    if continuation:
        request["messages"].append({"role": "assistant", "content": "1. Programming context:"})
        request["continue_final_message"] = True
    message = request["messages"][-1]
    prefix = message["content"]
    count = target - len(render(tokenizer, request, tokenize=True))
    for _ in range(8):
        if count < 0:
            raise ValueError("prompt target is too short for the explicit instruction")
        message["content"] = prefix + " apple" * count
        tokens = render(tokenizer, request, tokenize=True)
        if len(tokens) == target:
            break
        count += target - len(tokens)
    else:
        raise ValueError(f"cannot render exactly {target} tokens: {identifier}")
    text = render(tokenizer, request, tokenize=False)
    if tokenizer.encode(text, add_special_tokens=False) != tokens:
        raise ValueError("template tokenization differs from rendered text encoding")
    if not any(identifier in message["content"] for message in request["messages"]):
        raise ValueError("missing cold-cache identity")
    return request, tokens, text


def build_corpus(tokenizer, *, repetitions=5, nonce=None):
    if type(repetitions) is not int or repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    nonce = uuid.uuid4().hex if nonce is None else nonce
    if not isinstance(nonce, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", nonce):
        raise ValueError("nonce must contain 1..64 ASCII identifier characters")
    groups, cases = [], []

    def add_group(name, kind, length, batch, output, repetition, *, measured=True):
        ids = []
        for lane in range(batch):
            name_id = f"{name}-{lane}"
            request, tokens, text = make_prompt(tokenizer, f"{nonce}-{name_id}", length, output)
            cases.append({"id": name_id, "group": name, "request": request,
                          "prompt_tokens": tokens, "rendered_prompt": text,
                          "expected_cache": "cold", "output_tokens": output})
            ids.append(name_id)
        groups.append({"id": name, "kind": kind, "batch": batch,
                       "repetition": repetition, "measured": measured, "cases": ids})

    add_group("warmup", "decode", 512, 4, 256, -1, measured=False)
    for repetition in range(repetitions):
        for length in (512, 4096):
            add_group(f"ttft-{length}-r{repetition}", f"ttft-{length}", length, 1, 1, repetition)
        for batch in (1, 2, 4):
            add_group(f"decode-b{batch}-r{repetition}", f"decode-b{batch}", 512, batch, 256, repetition)
        add_group(f"prefill-65536-r{repetition}", "prefill-65536", 65536, 1, 1, repetition)
        seed_id = f"prefix-seed-r{repetition}"
        request, tokens, text = make_prompt(tokenizer, f"{nonce}-{seed_id}", 4096, 1, continuation=True)
        cases.append({"id": seed_id, "group": seed_id, "request": request,
                      "prompt_tokens": tokens, "rendered_prompt": text,
                      "expected_cache": "cold", "output_tokens": 1})
        groups.append({"id": seed_id, "kind": "prefix-seed", "batch": 1,
                       "repetition": repetition, "measured": False, "cases": [seed_id]})
        hit_id = f"prefix-hit-r{repetition}"
        hit = deepcopy(request)
        hit["messages"][-1]["content"] += " apple" * 11
        hit_tokens = render(tokenizer, hit, tokenize=True)
        if len(hit_tokens) != 4107 or hit_tokens[:4096] != tokens:
            raise ValueError("assistant continuation does not preserve the exact 4096-token prefix")
        cases.append({"id": hit_id, "group": hit_id, "request": hit,
                      "prompt_tokens": hit_tokens, "rendered_prompt": render(tokenizer, hit, tokenize=False),
                      "expected_cache": "prefix-4096", "prefix_seed": seed_id, "output_tokens": 1})
        groups.append({"id": hit_id, "kind": "prefix-hit", "batch": 1,
                       "repetition": repetition, "measured": True, "cases": [hit_id]})
    for case in cases:
        case.update(request_sha256=sha_bytes(encoded(case["request"])),
                    prompt_tokens_sha256=sha_bytes(token_bytes(case["prompt_tokens"])),
                    rendered_prompt_sha256=sha_bytes(case["rendered_prompt"].encode()))
    result = {"schema": SCHEMA + "-corpus", "model_id": MODEL_ID, "revision": REVISION,
              "nonce": nonce, "repetitions": repetitions, "groups": groups, "cases": cases,
              "generation_policy": "Greedy, production EOS, neutral penalties; exact requested output length is required."}
    validate_corpus(result)
    return result


def validate_corpus(corpus):
    if (corpus.get("schema") != SCHEMA + "-corpus" or corpus.get("model_id") != MODEL_ID
            or corpus.get("revision") != REVISION):
        raise ValueError("performance corpus checkpoint or schema differs")
    cases = {case["id"]: case for case in corpus["cases"]}
    if len(cases) != len(corpus["cases"]):
        raise ValueError("duplicate performance case ID")
    used, cold = [], set()
    repetitions = corpus.get("repetitions")
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("invalid corpus repetition count")
    expected_groups = [("warmup", "decode", 4, -1, False)]
    for repetition in range(repetitions):
        expected_groups.extend((f"ttft-{length}-r{repetition}", f"ttft-{length}", 1, repetition, True)
                               for length in (512, 4096))
        expected_groups.extend((f"decode-b{batch}-r{repetition}", f"decode-b{batch}", batch, repetition, True)
                               for batch in (1, 2, 4))
        expected_groups.extend(((f"prefill-65536-r{repetition}", "prefill-65536", 1, repetition, True),
                                (f"prefix-seed-r{repetition}", "prefix-seed", 1, repetition, False),
                                (f"prefix-hit-r{repetition}", "prefix-hit", 1, repetition, True)))
    if [(g["id"], g["kind"], g["batch"], g["repetition"], g["measured"]) for g in corpus["groups"]] != expected_groups:
        raise ValueError("corpus does not contain the complete ordered performance matrix")
    for group in corpus["groups"]:
        if group["batch"] != len(group["cases"]) or group["batch"] not in (1, 2, 4):
            raise ValueError("invalid simultaneous request group")
        used.extend(group["cases"])
        length = 4107 if group["kind"] == "prefix-hit" else (65536 if group["kind"] == "prefill-65536"
                 else 4096 if group["kind"] in ("ttft-4096", "prefix-seed") else 512)
        output = 256 if group["kind"].startswith("decode") else 1
        if any(cases[name]["group"] != group["id"] or len(cases[name]["prompt_tokens"]) != length
               or cases[name]["output_tokens"] != output for name in group["cases"]):
            raise ValueError("case token geometry differs from its measurement group")
    if sorted(used) != sorted(cases):
        raise ValueError("case/group coverage differs")
    for case in cases.values():
        tokens, request = case["prompt_tokens"], case["request"]
        if (not tokens or any(type(t) is not int or not 0 <= t < VOCABULARY for t in tokens)
                or len(tokens) + case["output_tokens"] > 262144):
            raise ValueError("invalid prompt token geometry")
        if (any(request.get(k) != v for k, v in SAMPLING.items()) or request.get("model") != MODEL_ID
                or request.get("max_tokens") != case["output_tokens"]
                or case["output_tokens"] not in (1, 256)):
            raise ValueError("performance sampling contract differs")
        if (case["request_sha256"] != sha_bytes(encoded(request))
                or case["prompt_tokens_sha256"] != sha_bytes(token_bytes(tokens))
                or case["rendered_prompt_sha256"] != sha_bytes(case["rendered_prompt"].encode())):
            raise ValueError("performance corpus fingerprint mismatch")
        if case["expected_cache"] == "cold":
            prefix = tuple(tokens[:64])
            if prefix in cold:
                raise ValueError("cold prompts share an entire 64-token cache block")
            cold.add(prefix)
        elif case["expected_cache"] == "prefix-4096":
            seed = cases[case["prefix_seed"]]
            if len(tokens) != 4107 or len(seed["prompt_tokens"]) != 4096 or tokens[:4096] != seed["prompt_tokens"]:
                raise ValueError("prefix fixture token identity differs")
        else:
            raise ValueError("unknown cache condition")


def prepare(args):
    if args.output.exists():
        raise ValueError("corpus destination already exists")
    import tokenizers
    from transformers import AutoTokenizer

    origin = json.loads(importlib.metadata.distribution("transformers").read_text("direct_url.json") or "{}")
    if origin.get("vcs_info", {}).get("commit_id") != TRANSFORMERS_REVISION or tokenizers.__version__ != "0.23.2":
        raise ValueError("corpus generation requires the pinned Transformers/tokenizers versions")
    assets = {**FRONTEND_SHA256, "config.json": CONFIG_SHA256}
    for name, expected in assets.items():
        if digest(args.snapshot / name) != expected:
            raise ValueError(f"checkpoint asset mismatch: {name}")
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    corpus = build_corpus(tokenizer, repetitions=args.repetitions, nonce=args.nonce)
    corpus["snapshot"] = str(args.snapshot.resolve())
    args.output.mkdir(parents=True)
    for case in corpus["cases"]:
        folder = args.output / case["id"]
        folder.mkdir()
        (folder / "request.json").write_bytes(encoded(case["request"]))
        (folder / "prompt.u32le").write_bytes(token_bytes(case["prompt_tokens"]))
        (folder / "rendered.txt").write_text(case["rendered_prompt"])
    write_json(args.output / "corpus.json", corpus)
    print(json.dumps({"corpus": str(args.output / "corpus.json"), "groups": len(corpus["groups"]),
                      "requests": len(corpus["cases"])}))
    return 0


def distribution(values):
    if not values:
        return {"count": 0, "median": None, "p99": None, "minimum": None, "maximum": None}
    ordered = sorted(values)
    return {"count": len(values), "median": statistics.median(values),
            "p99": ordered[math.ceil(.99 * len(values)) - 1], "minimum": ordered[0], "maximum": ordered[-1]}


def analyze_events(events, case, *, backend):
    """Analyze parsed events without conflating process and network clocks."""
    ids, groups, core = [], [], []
    prompts, finishes, usages, errors, performance, reported_cache = [], [], [], [], [], []
    done = 0
    for event in events:
        payload = event["data"]
        if payload == "[DONE]":
            done += 1
            continue
        if not isinstance(payload, dict):
            errors.append("SSE data is not a JSON object")
            continue
        if "error" in payload:
            errors.append(payload["error"])
        if payload.get("prompt_token_ids") is not None:
            prompts.append(payload["prompt_token_ids"])
        if payload.get("usage") is not None:
            usages.append(payload["usage"])
        if payload.get("performance") is not None:
            performance.append(payload["performance"])
        if payload.get("reused_prompt_tokens") is not None:
            reported_cache.append(payload["reused_prompt_tokens"])
        for choice in payload.get("choices", []):
            if choice.get("index") != 0:
                errors.append("unexpected choice index")
                continue
            if choice.get("finish_reason") is not None:
                finishes.append(choice["finish_reason"])
            tokens = choice.get("token_ids")
            timestamps = choice.get("token_timestamps_ns")
            if tokens is None:
                if timestamps:
                    errors.append("timestamps without token IDs")
                continue
            if (not isinstance(tokens, list)
                    or any(type(t) is not int or not 0 <= t < VOCABULARY for t in tokens)):
                errors.append("invalid generated token IDs")
                continue
            if tokens:
                ids.extend(tokens)
                groups.append({"timestamp_ns": event["elapsed_ns"], "token_ids": tokens})
            if timestamps is not None:
                if (not isinstance(timestamps, list) or len(timestamps) != len(tokens)
                        or any(type(t) is not int or t < 0 for t in timestamps)):
                    errors.append("invalid aligned completion timestamps")
                else:
                    core.extend(timestamps)
    checks = {"no_stream_error": not errors, "terminal_done": done == 1,
              "prompt_ids_equal": len(prompts) == 1 and prompts[0] == case["prompt_tokens"],
              "exact_output_length": len(ids) == case["output_tokens"],
              "finish_length": finishes == ["length"],
              "usage_exact": bool(usages) and usages[-1].get("prompt_tokens") == len(case["prompt_tokens"])
                  and usages[-1].get("completion_tokens") == len(ids)
                  and usages[-1].get("total_tokens") == len(ids) + len(case["prompt_tokens"]),
              "monotonic_delivery": all(a["timestamp_ns"] <= b["timestamp_ns"] for a, b in zip(groups, groups[1:]))}
    singleton = bool(groups) and all(len(group["token_ids"]) == 1 for group in groups)
    delivery_itl = [b["timestamp_ns"] - a["timestamp_ns"] for a, b in zip(groups, groups[1:])] if singleton else []
    core_complete = bool(core) and len(core) == len(ids) and all(a <= b for a, b in zip(core, core[1:]))
    checks["valid_core_timestamps"] = not core or core_complete
    if backend == "flare":
        checks["native_core_timestamps"] = core_complete
    metadata = performance[-1] if performance else {}
    usage = usages[-1] if usages else {}
    cached = metadata.get("reused_prompt_tokens", reported_cache[-1] if reported_cache else
                          (usage.get("prompt_tokens_details") or {}).get("cached_tokens"))
    checks["cache_count_reported"] = type(cached) is int and 0 <= cached <= len(case["prompt_tokens"])
    if case["expected_cache"] == "cold":
        checks["cold_cache"] = cached == 0
    elif backend == "flare":
        checks["prefix_4096_reused"] = cached == 4096
    else:
        checks["prefix_reused"] = type(cached) is int and 0 < cached <= 4096
    core_itl = [b - a for a, b in zip(core, core[1:])] if core_complete else []
    ingress, enqueue = metadata.get("ingress_ns"), metadata.get("enqueue_ns")
    native_ttft = None
    if core_complete and type(ingress) is int and type(enqueue) is int:
        checks["native_ingress_order"] = 0 <= ingress <= enqueue <= core[0]
        if checks["native_ingress_order"]:
            native_ttft = core[0] - ingress
    if backend == "flare":
        checks["native_timing_metadata_complete"] = native_ttft is not None
    return {"passed": all(checks.values()), "checks": checks, "errors": errors,
            "generated_token_ids": ids, "token_groups": groups, "core_token_timestamps_ns": core,
            "finish_reasons": finishes, "usage": usage, "performance": metadata,
            "reused_prompt_tokens": cached, "delivery_ttft_ns": groups[0]["timestamp_ns"] if groups else None,
            "delivery_itl_exact": singleton, "delivery_itl_ns": delivery_itl,
            "delivery_group_itl_ns": [b["timestamp_ns"] - a["timestamp_ns"] for a, b in zip(groups, groups[1:])],
            "coalesced_groups": sum(len(g["token_ids"]) > 1 for g in groups),
            "core_itl_ns": core_itl, "core_ttft_ns": native_ttft,
            "delivery_itl_summary_ns": distribution(delivery_itl), "core_itl_summary_ns": distribution(core_itl)}


def connection(base_url, timeout):
    url = urlsplit(base_url)
    if url.scheme not in ("http", "https") or not url.hostname or url.username or url.query or url.fragment:
        raise ValueError("base URL must be HTTP(S), with no credentials, query, or fragment")
    cls = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    return cls(url.hostname, url.port, timeout=timeout), url.path.rstrip("/")


def stream_request(base_url, case, folder, *, backend, model, timeout, barrier):
    folder.mkdir()
    request = deepcopy(case["request"])
    request.update(stream=True, return_token_ids=True, stream_options={"include_usage": True})
    if backend == "flare":
        request["return_token_timestamps"] = True
    if model:
        request["model"] = model
    payload = encoded(request)
    (folder / "request.json").write_bytes(payload)
    result = {"case": case["id"], "request_sha256": sha_bytes(payload), "events": []}
    conn, prefix = connection(base_url, timeout)
    raw = bytearray()
    start = None
    try:
        connecting = time.perf_counter_ns()
        conn.connect()
        result["connection_ns"] = time.perf_counter_ns() - connecting
        barrier.wait(timeout=timeout)
        start = time.perf_counter_ns()
        result["request_start_perf_counter_ns"] = start
        conn.request("POST", prefix + "/v1/chat/completions", body=payload,
                     headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
        response = conn.getresponse()
        result.update(status=response.status, headers=dict(response.getheaders()),
                      headers_ns=time.perf_counter_ns() - start)
        if response.status != 200:
            raw.extend(response.read(8 << 20))
            raise ValueError(f"HTTP status {response.status}")
        data = []
        while line := response.readline(1 << 20):
            raw.extend(line)
            if len(raw) > 32 << 20:
                raise ValueError("response exceeds 32 MiB")
            if not line.endswith(b"\n"):
                raise ValueError("SSE line exceeds bound or stream is truncated")
            stripped = line.rstrip(b"\r\n")
            if not stripped:
                if data:
                    timestamp = time.perf_counter_ns() - start
                    value = b"\n".join(data).decode("utf-8")
                    result["events"].append({"elapsed_ns": timestamp,
                                              "data": value if value == "[DONE]" else json.loads(value)})
                    data.clear()
                    if value == "[DONE]":
                        break
            elif stripped.startswith(b"data:"):
                data.append(stripped[5:].removeprefix(b" "))
        if data:
            raise ValueError("SSE stream ended inside an event")
    except Exception as exc:
        barrier.abort()
        result["client_error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        result["elapsed_ns"] = time.perf_counter_ns() - start if start is not None else None
        conn.close()
        (folder / "response.sse").write_bytes(raw)
    result["analysis"] = analyze_events(result["events"], case, backend=backend)
    if "client_error" in result:
        result["analysis"]["checks"]["client_completed"] = False
        result["analysis"]["passed"] = False
    write_json(folder / "observation.json", result)
    return result


def metrics(base_url, path, timeout):
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
                name, value, *_ = line.split()
                number = float(value)
                values[name] = number if math.isfinite(number) else None
        return values
    finally:
        conn.close()


def group_summary(group, observations):
    analyses = [value["analysis"] for value in observations]
    delivery = [value for a in analyses for value in a["delivery_itl_ns"]]
    core = [value for a in analyses for value in a["core_itl_ns"]]
    start = [o.get("request_start_perf_counter_ns") for o in observations]
    last = [s + a["token_groups"][-1]["timestamp_ns"] for s, a in zip(start, analyses)
            if s is not None and a["token_groups"]]
    first = [s + a["token_groups"][0]["timestamp_ns"] for s, a in zip(start, analyses)
             if s is not None and a["token_groups"]]
    span = max(last) - min(first) if len(last) == len(observations) else 0
    count = sum(max(0, len(a["generated_token_ids"]) - 1) for a in analyses)
    return {"id": group["id"], "kind": group["kind"], "batch": group["batch"],
            "measured": group["measured"], "passed": all(a["passed"] for a in analyses),
            "requests": [o["case"] for o in observations],
            "delivery_ttft_ns": distribution([a["delivery_ttft_ns"] for a in analyses if a["delivery_ttft_ns"] is not None]),
            "delivery_itl_ns": distribution(delivery), "core_itl_ns": distribution(core),
            "generated_after_first_tokens": count, "delivery_decode_span_ns": span,
            "delivery_decode_tokens_per_second": count * 1e9 / span if count and span > 0 else None,
            "request_release_skew_ns": max(start) - min(start) if all(s is not None for s in start) else None}


def idle_metrics(args, folder, name):
    deadline = time.monotonic() + min(args.timeout, 30)
    attempt = 0
    while True:
        path = folder / f"metrics-{name}-{attempt:03}.txt"
        values = metrics(args.base_url, path, args.timeout)
        if args.backend != "flare" or (values.get("flare_active_requests") == 0
                and values.get("flare_free_request_slots") == 4
                and values.get("flare_retiring_request_slots") == 0
                and values.get("flare_pending_requests") == 0):
            return values
        if time.monotonic() >= deadline:
            raise ValueError("native scheduler did not become idle within 30 seconds")
        attempt += 1
        time.sleep(.05)


def summarize_measurements(corpus, group_summaries, observations):
    grouped = {group["id"]: group for group in corpus["groups"]}
    cases = {case["id"]: case for case in corpus["cases"]}
    measurements = {}
    for kind in sorted({group["kind"] for group in corpus["groups"] if group["measured"]}):
        values = [o["analysis"] for o in observations if grouped[cases[o["case"]]["group"]]["kind"] == kind]
        summaries = [g for g in group_summaries if g["kind"] == kind]
        expected_groups = sum(g["measured"] and g["kind"] == kind for g in corpus["groups"])
        measurements[kind] = {
            "requests": len(values), "expected_groups": expected_groups, "completed_groups": len(summaries),
            "passed": len(summaries) == expected_groups and all(g["passed"] for g in summaries),
            "delivery_ttft_ns": distribution([a["delivery_ttft_ns"] for a in values if a["delivery_ttft_ns"] is not None]),
            "core_ttft_ns": distribution([a["core_ttft_ns"] for a in values if a["core_ttft_ns"] is not None]),
            "delivery_itl_ns": distribution([t for a in values for t in a["delivery_itl_ns"]]),
            "core_itl_ns": distribution([t for a in values for t in a["core_itl_ns"]]),
            "coalesced_delivery_groups": sum(a["coalesced_groups"] for a in values),
            "delivery_decode_tokens_per_second": distribution([g["delivery_decode_tokens_per_second"] for g in summaries
                                                                if g["delivery_decode_tokens_per_second"] is not None]),
            "http_prefill_tokens_per_second": distribution([a["http_prefill_tokens_per_second"] for a in values
                                                              if "http_prefill_tokens_per_second" in a]),
            "reused_prompt_tokens": [a["reused_prompt_tokens"] for a in values],
        }
    return measurements


def run(args):
    if args.output.exists():
        raise ValueError("run destination already exists")
    corpus = json.loads((args.corpus / "corpus.json").read_text())
    validate_corpus(corpus)
    for case in corpus["cases"]:
        folder = args.corpus / case["id"]
        if ((folder / "request.json").read_bytes() != encoded(case["request"])
                or (folder / "prompt.u32le").read_bytes() != token_bytes(case["prompt_tokens"])
                or (folder / "rendered.txt").read_text() != case["rendered_prompt"]):
            raise ValueError(f"corpus sidecar mismatch: {case['id']}")
    if args.backend == "flare":
        if args.memory_report is None:
            raise ValueError("Flare measurement requires its actual preparation memory report")
        memory = json.loads(args.memory_report.read_text())
        if memory.get("prefill_chunk_size") != args.prefill_chunk_size:
            raise ValueError("actual prepared prefill capacity differs from the measurement label")
    args.output.mkdir(parents=True)
    measurements = {"backend": args.backend, "base_url": args.base_url,
                    "prefill_chunk_size": args.prefill_chunk_size,
               "groups": [], "errors": [], "passed": False,
               "measurement": {"clock": "client perf_counter_ns; native token timestamps are a separate process-relative domain",
                   "p99": "nearest rank ceil(0.99*n)", "connection": "connections open before the simultaneous send barrier; connection duration is reported separately",
                   "decode_throughput": "sum(output_tokens-1) / (last delivered token - earliest first delivered token)",
                   "prefill_throughput": "prompt_tokens / delivery TTFT; includes frontend, sampling and delivery, not isolated GPU prefill",
                   "core_ttft": "native chat-handler entry after JSON extraction to CPU token observation after scheduler completion",
                   "core_itl": "CPU completion-observation timestamp differences; includes completion polling and scheduler processing",
                   "observation_overhead": "native token observation adds a channel event and metadata SSE chunk per token; unobserved profiling is separate",
                   "batch": "requested concurrency; actual GPU batch requires the separate executor/Nsight trace"}}
    cases = {case["id"]: case for case in corpus["cases"]}
    all_observations = []
    for group in corpus["groups"]:
        folder = args.output / group["id"]
        folder.mkdir()
        try:
            before = idle_metrics(args, folder, "before")
            barrier = threading.Barrier(group["batch"])
            with ThreadPoolExecutor(max_workers=group["batch"]) as executor:
                futures = [executor.submit(stream_request, args.base_url, cases[name], folder / name,
                            backend=args.backend, model=args.model, timeout=args.timeout, barrier=barrier)
                           for name in group["cases"]]
                observations = [future.result() for future in futures]
            all_observations.extend(observations)
            after = idle_metrics(args, folder, "after")
            summary = group_summary(group, observations)
            summary["metric_deltas"] = {key: after[key] - before[key] for key in before.keys() & after.keys()
                                        if after[key] is not None and before[key] is not None
                                        and "_total" in key}
            if args.backend == "flare":
                expected = 4096 if group["kind"] == "prefix-hit" else 0
                summary["native_cache_metric_exact"] = summary["metric_deltas"].get("flare_cache_reused_tokens_total") == expected
                summary["native_requests_completed"] = summary["metric_deltas"].get("flare_requests_completed_total") == group["batch"]
                summary["native_no_failures"] = all(summary["metric_deltas"].get(name) == 0 for name in (
                    "flare_requests_failed_total", "flare_requests_cancelled_total"))
                summary["native_resources_conserved"] = all(
                    before[f"flare_free_{resource}"] + before[f"flare_cached_{cached}"]
                    == after[f"flare_free_{resource}"] + after[f"flare_cached_{cached}"]
                    for resource, cached in (("kv_pages", "kv_pages"), ("snapshots", "checkpoints")))
                summary["passed"] &= all(summary[key] for key in ("native_cache_metric_exact", "native_requests_completed",
                                         "native_no_failures", "native_resources_conserved"))
            for observation in observations:
                ttft = observation["analysis"]["delivery_ttft_ns"]
                if ttft and group["kind"] in ("ttft-4096", "prefill-65536"):
                    observation["analysis"]["http_prefill_tokens_per_second"] = len(cases[observation["case"]]["prompt_tokens"]) * 1e9 / ttft
            measurements["groups"].append(summary)
            print(json.dumps(summary), flush=True)
        except Exception as exc:
            measurements["errors"].append({"group": group["id"], "error": f"{type(exc).__name__}: {exc}",
                                      "traceback": traceback.format_exc()})
        write_json(args.output / "measurements.json", measurements)
    measurements["observations"] = all_observations
    measurements["passed"] = (not measurements["errors"]
        and len(measurements["groups"]) == len(corpus["groups"])
        and all(group["passed"] for group in measurements["groups"]))
    measurements["summary"] = {"groups": len(measurements["groups"]), "requests": len(all_observations),
                          "failed_groups": sum(not group["passed"] for group in measurements["groups"]),
                          "errors": len(measurements["errors"])}
    measurements["measurements"] = summarize_measurements(corpus, measurements["groups"], all_observations)
    write_json(args.output / "measurements.json", measurements)
    return 0 if measurements["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    build.add_argument("--snapshot", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--repetitions", type=int, default=5)
    build.add_argument("--nonce")
    measure = commands.add_parser("run")
    measure.add_argument("--corpus", type=Path, required=True)
    measure.add_argument("--output", type=Path, required=True)
    measure.add_argument("--backend", choices=("flare", "http"), required=True)
    measure.add_argument("--base-url", required=True)
    measure.add_argument("--prefill-chunk-size", type=int, choices=(512, 4096), required=True)
    measure.add_argument("--model")
    measure.add_argument("--memory-report", type=Path)
    measure.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    return prepare(args) if args.command == "prepare" else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
