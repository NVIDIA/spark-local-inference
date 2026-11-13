# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify a running Qwen HTTP server and preserve requests, responses and raw SSE."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import copy
from datetime import datetime, timezone
import hashlib
import http.client
import importlib.metadata
import importlib.machinery
import json
from pathlib import Path
import socket
import threading
import time
import traceback
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def common_prefix(left: list[int], right: list[int]) -> int:
    return next((i for i, pair in enumerate(zip(left, right)) if pair[0] != pair[1]), min(len(left), len(right)))


def template_input(request):
    options = {**request.get("chat_template_kwargs", {}), "enable_thinking": request.get("enable_thinking", False),
        "add_generation_prompt": request.get("add_generation_prompt", True)}
    for key in ("tools", "reasoning_effort", "continue_final_message"):
        if key in request:
            options[key] = request[key]
    messages = copy.deepcopy(request["messages"])
    for message in messages:
        for call in message.get("tool_calls", []):
            function = call.get("function", {})
            if isinstance(function.get("arguments"), str):
                function["arguments"] = json.loads(function["arguments"])
    return messages, options


def rendered_prefix_ids(tokenizer, messages, options):
    from jinja2 import TemplateError
    from transformers.utils.chat_template_utils import _compile_jinja_template

    template = _compile_jinja_template(tokenizer.get_chat_template())
    options = {**options, "add_generation_prompt": False, "continue_final_message": False}
    chunks = []
    try:
        for chunk in template.generate(messages=messages, **tokenizer.special_tokens_map, **options):
            chunks.append(chunk)
    except TemplateError:
        pass
    return tokenizer.encode("".join(chunks), add_special_tokens=False)


def message_prefixes(tokenizer, request, prompt_tokens):
    messages, options = template_input(request)
    prefixes = {}
    for count in range(len(messages) + 1):
        ids = rendered_prefix_ids(tokenizer, messages[:count], options)
        if 0 < len(ids) < len(prompt_tokens) and prompt_tokens[:len(ids)] == ids:
            prefixes[len(ids)] = ids
    return [prefixes[length] for length in sorted(prefixes)]


def system_prefix_ids(tokenizer, request, prompt_tokens):
    """Return the complete leading instruction endpoint for instruction-identity tests."""
    messages, options = template_input(request)
    count = next((i for i, message in enumerate(messages)
        if message.get("role") not in ("system", "developer")), len(messages))
    ids = rendered_prefix_ids(tokenizer, messages[:count], options)
    return ids if 0 < len(ids) < len(prompt_tokens) and prompt_tokens[:len(ids)] == ids else []


def prefill_launches(prompt_tokens, reused_tokens, chunk_size, message_boundaries=()):
    if not 0 <= reused_tokens <= prompt_tokens or chunk_size <= 0:
        raise ValueError("invalid exact checkpoint prefill geometry")
    boundaries = sorted({n for n in message_boundaries if reused_tokens < n < prompt_tokens}) + [prompt_tokens]
    launches, position = 0, reused_tokens
    for boundary in boundaries:
        launches += (boundary - position + chunk_size - 1) // chunk_size
        position = boundary
    return launches


def system_prefix_requests(model_id, nonce):
    instruction = f"Brief {nonce[:3]}."
    base = {"model": model_id, "temperature": 1, "top_k": 20, "top_p": 0.95,
        "seed": 41873, "max_tokens": 32, "enable_thinking": False, "add_generation_prompt": True}
    tool = {"type": "function", "function": {"name": "unused_alpha", "description": "An unused lookup tool.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}}
    result = []
    for name, instructions, question, tools, expected_reuse in (
        ("short-seed", instruction, "Return only OK for request A.", None, False),
        ("short-branch", instruction, "Return only OK for request B.", None, True),
        ("instructions-change", f"Terse {nonce[:3]}.", "Return only OK for request C.", None, False),
        ("tools-seed", instruction, "Return only OK for request D. Do not call a tool.", [tool], False),
        ("tools-branch", instruction, "Return only OK for request E. Do not call a tool.", [tool], True),
        ("tools-change", instruction, "Return only OK for request F. Do not call a tool.",
            [{**tool, "function": {**tool["function"], "name": "unused_beta"}}], False),
    ):
        request = {**base, "messages": [{"role": "system", "content": instructions}, {"role": "user", "content": question}]}
        if tools is not None:
            request["tools"] = copy.deepcopy(tools)
        result.append((name, request, expected_reuse))
    return result


def metric_delta(before: dict, after: dict) -> dict:
    return {name: after[name] - before[name] for name in before.keys() & after.keys() if "_total" in name}


def resource_totals(metrics: dict) -> dict:
    return {"pages": metrics["flare_free_kv_pages"] + metrics["flare_cached_kv_pages"],
        "snapshots": metrics["flare_free_snapshots"] + metrics["flare_cached_checkpoints"],
        "slots": metrics["flare_free_request_slots"]}


def kv_pressure_geometry(memory: dict, metrics: dict, context: int) -> dict:
    totals = resource_totals(metrics)
    pages = memory["pages"]
    if not isinstance(pages, int) or pages <= 0 or memory["cached_token_capacity"] != pages * 64:
        raise ValueError("KV pressure requires the memory report's 64-token page geometry")
    if memory["kv_bytes"] != pages * memory["page_bytes"] or totals["pages"] != pages:
        raise ValueError("memory report KV capacity differs from idle native metrics")
    if totals["snapshots"] != memory["snapshot_capacity"] or totals["slots"] != 4:
        raise ValueError("KV pressure requires the reported snapshot capacity and four idle request slots")
    if context <= 0 or context % 64:
        raise ValueError("prepared context must be positive and 64-token aligned")
    draft_pages = int(memory.get("mtp_weights_bytes", 0) > 0)
    reservation = context // 64 + draft_pages
    holders = pages // reservation
    if not 1 <= holders < totals["slots"]:
        raise ValueError("KV exhaustion cannot be isolated before the slot limit using context-sized reservations")
    if totals["snapshots"] < 3 * (holders + 1):
        raise ValueError("snapshot capacity cannot isolate KV exhaustion")
    return {"page_tokens": 64, "pages": pages, "context_tokens": context,
        "pages_per_request": reservation, "holding_requests": holders,
        "unreserved_pages": pages - holders * reservation, "max_active": totals["slots"],
        "draft_pages_per_request": draft_pages, "snapshots_per_request": 3}


class StreamProgress:
    def __init__(self):
        self.condition = threading.Condition()
        self.visible_events = 0
        self.finished = False

    def observe(self):
        with self.condition:
            self.visible_events += 1
            self.condition.notify_all()

    def finish(self):
        with self.condition:
            self.finished = True
            self.condition.notify_all()

    def count(self):
        with self.condition:
            return self.visible_events

    def wait_after(self, count, timeout):
        with self.condition:
            self.condition.wait_for(lambda: self.visible_events > count or self.finished, timeout)
            return self.visible_events > count


def aggregate_sse(events: list) -> dict:
    message = {"role": "assistant", "content": ""}
    calls: dict[int, dict] = {}
    finishes = []
    errors = []
    for event in events:
        if "error" in event:
            errors.append(event["error"])
            continue
        for choice in event.get("choices", []):
            if choice.get("index") != 0:
                raise ValueError("SSE returned an unexpected choice index")
            delta = choice.get("delta", {})
            for key in ("content", "reasoning_content"):
                if delta.get(key) is not None:
                    if not isinstance(delta[key], str):
                        raise ValueError(f"SSE {key} is not a string")
                    message[key] = message.get(key, "") + delta[key]
            for call in delta.get("tool_calls", []):
                target = calls.setdefault(call["index"], {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                if call.get("id"):
                    target["id"] = call["id"]
                for key in ("name", "arguments"):
                    target["function"][key] += call.get("function", {}).get(key, "")
            if choice.get("finish_reason") is not None:
                finishes.append(choice["finish_reason"])
    if calls:
        message["tool_calls"] = [calls[index] for index in sorted(calls)]
    return {"message": message, "finish_reasons": finishes, "errors": errors}


def comparable_message(message: dict) -> dict:
    value = {"content": message.get("content") or "", "reasoning_content": message.get("reasoning_content") or ""}
    value["tool_calls"] = [call["function"] for call in message.get("tool_calls", [])]
    return value


class Run:
    def __init__(self, args):
        self.args = args
        self.url = urlsplit(args.base_url)
        if self.url.scheme not in ("http", "https") or not self.url.hostname or self.url.query or self.url.fragment or self.url.username:
            raise ValueError("base URL must be an HTTP(S) server URL without credentials, query or fragment")
        self.lock = threading.RLock()
        self.counter = 0
        self.results = {"started_at": datetime.now(timezone.utc).isoformat(),
                        "requests": [], "checks": [], "case_errors": []}

    def check(self, name, condition, **evidence):
        with self.lock:
            self.results["checks"].append({"name": name, "passed": bool(condition), **evidence})
        print(f"{'PASS' if condition else 'FAIL'} {name}", flush=True)

    def case(self, name, function):
        if self.args.cases and name not in self.args.cases:
            return None
        print(f"RUN {name}", flush=True)
        try:
            return function()
        except Exception as exc:
            with self.lock:
                self.results["case_errors"].append({"case": name, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
            print(f"ERROR {name}: {type(exc).__name__}: {exc}", flush=True)
            return None

    def request(self, name, body=None, *, path="/v1/chat/completions", stream=False, cancel=False, barrier=None,
                paused=None, resume=None, receive_buffer=None, progress=None, cancel_event=None):
        with self.lock:
            self.counter += 1
            number = self.counter
        stem = f"{number:03d}-{name}"
        if body is not None:
            body = {**body, "stream": stream}
            write_json(self.args.artifacts / f"{stem}.request.json", body)
        if barrier is not None:
            barrier.wait(timeout=30)
        started = time.monotonic()
        record = {"name": name, "method": "POST" if body is not None else "GET", "path": path,
            "started_monotonic": started, "stream": stream, "cancel_requested": cancel,
            "request_artifact": str(self.args.artifacts / f"{stem}.request.json") if body is not None else None}
        connection = None
        try:
            cls = http.client.HTTPSConnection if self.url.scheme == "https" else http.client.HTTPConnection
            connection = cls(self.url.hostname, self.url.port, timeout=self.args.timeout)
            if receive_buffer is not None:
                connection.connect()
                connection.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, receive_buffer)
                if hasattr(socket, "TCP_WINDOW_CLAMP"):
                    connection.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_WINDOW_CLAMP, receive_buffer)
                record["socket_receive_buffer_bytes"] = connection.sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
            encoded = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
            connection.request(record["method"], self.url.path.rstrip("/") + path, body=encoded,
                headers={"Content-Type": "application/json", "Accept": "text/event-stream" if stream else "application/json"})
            response = connection.getresponse()
            record.update(status=response.status, headers=dict(response.getheaders()), headers_seconds=time.monotonic() - started)
            sse = stream and response.status == 200 and response.getheader("Content-Type", "").startswith("text/event-stream")
            raw_path = self.args.artifacts / f"{stem}.{'sse' if sse else 'response.json'}"
            record["response_artifact"] = str(raw_path)
            if not sse:
                raw = response.read(self.args.max_response_bytes + 1)
                raw_path.write_bytes(raw)
                if len(raw) > self.args.max_response_bytes:
                    raise ValueError("response exceeded byte bound")
                record["body"] = {"prometheus": raw.decode("utf-8")} if path == "/metrics" and response.status == 200 else json.loads(raw)
            else:
                events, data, raw_bytes, done_count, cancelled = [], [], 0, 0, False
                with raw_path.open("wb") as artifact:
                    while True:
                        line = response.readline()
                        if not line:
                            break
                        artifact.write(line)
                        artifact.flush()
                        raw_bytes += len(line)
                        if raw_bytes > self.args.max_response_bytes:
                            raise ValueError("SSE response exceeded byte bound")
                        stripped = line.rstrip(b"\r\n")
                        if stripped.startswith(b"data:"):
                            data.append(stripped[5:].removeprefix(b" "))
                        elif not stripped and data:
                            payload = b"\n".join(data).decode("utf-8")
                            data.clear()
                            if payload == "[DONE]":
                                done_count += 1
                            else:
                                event = json.loads(payload)
                                events.append(event)
                                visible = any(any(choice.get("delta", {}).get(key) for key in ("content", "reasoning_content", "tool_calls")) for choice in event.get("choices", []))
                                if visible:
                                    record.setdefault("first_visible_delta_seconds", time.monotonic() - started)
                                    if progress is not None:
                                        progress.observe()
                                if visible and paused is not None:
                                    pause_started = time.monotonic()
                                    paused.set()
                                    if not resume.wait(timeout=self.args.timeout):
                                        raise TimeoutError("stalled-reader controller did not release the connection")
                                    record["paused_read_seconds"] = time.monotonic() - pause_started
                                    paused = None
                                if visible and (cancel or (cancel_event is not None and cancel_event.is_set())):
                                    cancelled = True
                                    break
                record.update(events=events, sse=aggregate_sse(events), done_count=done_count,
                    cancelled=cancelled, raw_bytes=raw_bytes, incomplete_event=bool(data))
                if cancelled:
                    response.close()
            record["response_sha256"] = digest(raw_path)
        except Exception as exc:
            record["client_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if connection is not None:
                connection.close()
            if progress is not None:
                progress.finish()
            record["elapsed_seconds"] = time.monotonic() - started
            record["finished_monotonic"] = time.monotonic()
            with self.lock:
                self.results["requests"].append(record)
        return record

    def metrics(self, name, *, wait_idle=True):
        if not self.args.require_metrics:
            return None
        deadline = time.monotonic() + self.args.timeout
        attempt = 0
        while True:
            record = self.request(f"{name}-metrics-{attempt}", path="/metrics")
            if record.get("status") != 200 or "client_error" in record:
                raise ValueError("required native /metrics endpoint failed")
            metrics = {}
            for line in record["body"]["prometheus"].splitlines():
                if line and not line.startswith("#"):
                    key, value = line.rsplit(" ", 1)
                    metrics[key] = int(value)
            required = ["flare_active_requests", "flare_retiring_request_slots", "flare_pending_requests", "flare_http_inflight_requests"]
            if not wait_idle or all(metrics[key] == 0 for key in required):
                return metrics
            if time.monotonic() >= deadline:
                raise TimeoutError("model did not reach observable idle state")
            attempt += 1
            time.sleep(0.01)

    def reclaimed(self, name, before, after):
        if before is not None:
            self.check(name + ".resource_accounting", resource_totals(before) == resource_totals(after)
                and after["flare_active_requests"] == after["flare_retiring_request_slots"] == 0,
                before=resource_totals(before), after=resource_totals(after))

    def concurrent(self, name, requests):
        before = self.metrics(name + "-before")
        samples = []
        barrier = threading.Barrier(len(requests))
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(requests)) as pool:
            futures = [pool.submit(self.request, f"{name}-{i}", request, barrier=barrier) for i, request in enumerate(requests)]
            if before is not None:
                while not all(future.done() for future in futures):
                    samples.append(self.metrics(name + f"-active-{len(samples)}", wait_idle=False))
                    time.sleep(0.01)
            results = [future.result() for future in futures]
        after = self.metrics(name + "-after")
        if before is not None:
            delta = metric_delta(before, after)
            peak = max((sample["flare_active_requests"] for sample in samples), default=0)
            self.check(name + ".all_model_slots_observed", peak == len(requests), peak_active_requests=peak)
            self.check(name + ".admitted_and_completed", delta["flare_requests_admitted_total"] == len(requests)
                and delta["flare_requests_completed_total"] == len(requests)
                and delta["flare_requests_cancelled_total"] == delta["flare_requests_failed_total"] == 0, delta=delta)
            self.check(name + ".capacity_bounds", all(sample["flare_active_requests"] <= before["flare_free_request_slots"]
                and sample["flare_frontend_active_workers"] <= sample["flare_frontend_worker_capacity"]
                and sample["flare_http_inflight_requests"] <= sample["flare_http_request_capacity"] for sample in samples))
            self.reclaimed(name, before, after)
        return results

    def complete(self, record, prompt_tokens=None):
        body = record.get("body", {})
        valid = record.get("status") == 200 and "client_error" not in record and "error" not in body and len(body.get("choices", [])) == 1
        self.check(f"{record['name']}.http_completion", valid, status=record.get("status"), error=record.get("client_error") or body.get("error"))
        if not valid:
            raise ValueError(f"{record['name']} did not return a successful completion")
        choice = body["choices"][0]
        if prompt_tokens is not None:
            self.check(f"{record['name']}.prompt_token_count", body.get("usage", {}).get("prompt_tokens") == prompt_tokens,
                expected=prompt_tokens, observed=body.get("usage", {}).get("prompt_tokens"))
        self.check(f"{record['name']}.terminal_reason", choice.get("finish_reason") in ("stop", "length", "tool_calls"), reason=choice.get("finish_reason"))
        return choice

    def pair(self, name, body, prompt_tokens=None):
        before = self.metrics(name + "-before")
        baseline = self.request(name + "-json", body)
        expected = self.complete(baseline, prompt_tokens)
        between = self.metrics(name + "-between")
        streamed = self.request(name + "-stream", body, stream=True)
        self.check(name + ".sse_complete", streamed.get("status") == 200 and not streamed.get("client_error") and streamed.get("done_count") == 1
            and not streamed.get("incomplete_event") and not streamed.get("sse", {}).get("errors"), status=streamed.get("status"), errors=streamed.get("sse", {}).get("errors"), client_error=streamed.get("client_error"))
        combined = streamed.get("sse", {})
        self.check(name + ".greedy_stream_equivalence", comparable_message(expected["message"]) == comparable_message(combined.get("message", {}))
            and combined.get("finish_reasons") == [expected["finish_reason"]], json_choice=expected, stream=combined)
        after = self.metrics(name + "-after")
        if before is not None:
            self.results.setdefault("model_metric_deltas", []).append({"case": name, "json": metric_delta(before, between), "stream": metric_delta(between, after)})
        return baseline, streamed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--memory-report", type=Path)
    parser.add_argument("--gdn-intermediate-dtype", choices=("bfloat16", "float32"), required=True)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-response-bytes", type=int, default=64 << 20)
    parser.add_argument("--cases", nargs="+", choices=("initial_health", "models", "greedy_text", "canonical_chat", "concurrent_four",
        "thinking_output", "tool_call_and_result", "prefix_full_aligned_branch", "system_prefix", "structured_literal_tags", "png_image", "png_full_capacity", "image_cache_identity",
        "cancellation_and_slot_reuse", "mixed_grammar", "cache_pressure", "conversation_pressure", "kv_pressure", "slow_consumer", "final_health",
        "prompt_endpoint_residues", "generated_terminal_prefix"), help="Run only these named qualification cases.")
    parser.add_argument("--image-repetitions", type=int, default=1)
    parser.add_argument("--full-vision-png", type=Path, help="PNG encoding of the pinned bucket_16384 RGB fixture; exercise all 16384 image patches.")
    parser.add_argument("--mixed-grammar", action="store_true", help="Observe constrained and unconstrained requests in the same decode submission.")
    parser.add_argument("--image-cache-identity", action="store_true", help="Check cache separation and reuse for identical text with different pinned images.")
    parser.add_argument("--checkpoint-endpoints", action="store_true", help="Check prompt residues, duplicate publication and consumed generated-prefix reuse with observed token IDs.")
    parser.add_argument("--require-metrics", action="store_true", help="Require native Prometheus metrics and record idle counter deltas.")
    parser.add_argument("--cache-pressure", action="store_true", help="Publish enough unique prefixes to exceed the exposed snapshot capacity.")
    parser.add_argument("--conversation-pressure", action="store_true", help="Exercise edited conversation prompts beyond the exact-checkpoint capacity.")
    parser.add_argument("--kv-pressure", action="store_true", help="Reserve context-sized KV extents and observe excess admission rejection.")
    parser.add_argument("--prepared-context-tokens", type=int, help="Actual server --context value, required for KV pressure and verified against HTTP admission.")
    parser.add_argument("--slow-consumer-seconds", type=float, default=0, help="Pause one real SSE reader for this duration while checking unrelated progress.")
    args = parser.parse_args()
    if args.artifacts.exists() and any(args.artifacts.iterdir()):
        parser.error("output and artifact destinations must not contain a previous run")
    if args.timeout <= 0 or args.max_response_bytes <= 0 or args.image_repetitions <= 0:
        parser.error("timeout, response byte bound and image repetitions must be positive")
    if args.slow_consumer_seconds < 0 or args.slow_consumer_seconds >= args.timeout:
        parser.error("slow-consumer duration must be nonnegative and shorter than the request timeout")
    if args.checkpoint_endpoints and not args.require_metrics:
        parser.error("checkpoint-endpoint qualification requires --require-metrics")
    if (args.cache_pressure or args.conversation_pressure or args.kv_pressure or args.slow_consumer_seconds or args.full_vision_png or args.mixed_grammar or args.image_cache_identity) and not args.require_metrics:
        parser.error("pressure, slow-consumer, full-vision and mixed-grammar qualification require --require-metrics")
    if args.cases and "system_prefix" in args.cases and not args.require_metrics:
        parser.error("system-prefix qualification requires --require-metrics")
    exact_cache_cases = {"prefix_full_aligned_branch", "system_prefix", "conversation_pressure", "prompt_endpoint_residues", "generated_terminal_prefix"}
    if (args.require_metrics and args.memory_report is None
            and (args.cases is None or exact_cache_cases.intersection(args.cases))):
        parser.error("exact-checkpoint graph assertions require --memory-report for prefill geometry")
    if args.kv_pressure and (args.memory_report is None or args.prepared_context_tokens is None):
        parser.error("KV pressure requires --memory-report and --prepared-context-tokens from the server launch")
    if args.cases and (("cache_pressure" in args.cases and not args.cache_pressure)
            or ("conversation_pressure" in args.cases and not args.conversation_pressure)
            or ("kv_pressure" in args.cases and not args.kv_pressure)
            or ("png_full_capacity" in args.cases and not args.full_vision_png)
            or ("mixed_grammar" in args.cases and not args.mixed_grammar)
            or ("image_cache_identity" in args.cases and not args.image_cache_identity)
            or (bool({"prompt_endpoint_residues", "generated_terminal_prefix"}.intersection(args.cases)) and not args.checkpoint_endpoints)
            or ("slow_consumer" in args.cases and not args.slow_consumer_seconds)):
        parser.error("selected pressure/slow-consumer/full-vision cases require their enabling option")
    args.artifacts.mkdir(parents=True, exist_ok=True)

    text_fixture = json.loads((ROOT / "tests/frontend/qwen38.json").read_text())
    image_fixture = json.loads((ROOT / "tests/frontend/qwen38_images.json").read_text())
    model_id, revision = text_fixture["model_id"], text_fixture["revision"]
    snapshot = (args.snapshot or Path.home() / ".cache/huggingface/hub" / ("models--" + model_id.replace("/", "--")) / "snapshots" / revision).resolve()
    assets = {name: digest(snapshot / name) for name in image_fixture["assets"]}
    if assets != image_fixture["assets"]:
        raise ValueError("snapshot frontend assets differ from pinned canonical fixtures")
    import transformers
    import tokenizers

    origin = json.loads(importlib.metadata.distribution("transformers").read_text("direct_url.json"))
    if origin.get("vcs_info", {}).get("commit_id") != text_fixture["transformers_commit"] or tokenizers.__version__ != text_fixture["tokenizers_version"]:
        raise ValueError("qualification requires the pinned Transformers/tokenizers preparation environment")
    tokenizer = transformers.AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    run = Run(args)

    def body(messages, **extra):
        return {"model": model_id, "messages": messages, "temperature": 0, "seed": 1234, "max_tokens": 32,
            "enable_thinking": False, "add_generation_prompt": True, **extra}

    def prompt_ids(request):
        messages, options = template_input(request)
        return tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, **options)

    memory = json.loads(args.memory_report.read_text()) if args.memory_report else {}
    chunk_size = memory.get("prefill_chunk_size")
    if chunk_size is not None and (type(chunk_size) is not int or chunk_size < 64 or chunk_size % 64):
        raise ValueError("memory report requires a positive 64-aligned prefill chunk size")

    def check_cache(name, delta, request, ids, reused):
        if delta is None:
            return
        boundaries = [len(prefix) for prefix in message_prefixes(tokenizer, request, ids)]
        launches = prefill_launches(len(ids), reused, chunk_size, boundaries)
        run.check(name + ".exact_cache_admission", delta["flare_cache_hits_total"] == int(reused > 0)
            and delta["flare_cache_misses_total"] == int(reused == 0)
            and delta["flare_cache_reused_tokens_total"] == reused
            and delta['flare_graph_launches_total{kind="restore"}'] == int(reused > 0)
            and delta['flare_graph_launches_total{kind="init"}'] == int(reused == 0),
            reused_tokens=reused, system_tokens=system, prompt_tokens=len(ids), delta=delta)
        run.check(name + ".exact_suffix_graph", delta['flare_graph_launches_total{kind="prefill"}'] == launches
            and delta['flare_graph_launches_total{kind="resume"}'] == int(reused == len(ids)),
            expected_prefill_launches=launches, delta=delta)

    def health(name):
        result = run.request(name, path="/health")
        run.check(name, result.get("status") == 200 and result.get("body", {}).get("status") == "ok", result=result)

    run.case("initial_health", lambda: health("health-before"))
    def models():
        result = run.request("models", path="/v1/models")
        run.check("exact_model_is_served", result.get("status") == 200 and model_id in [model.get("id") for model in result.get("body", {}).get("data", [])], result=result)
    run.case("models", models)
    capital = body([{"role": "user", "content": "What is the capital of France? Answer with only the city name."}], max_tokens=16)
    def text():
        baseline, _ = run.pair("greedy-text", capital, len(prompt_ids(capital)))
        run.check("capital_answer", "paris" in baseline["body"]["choices"][0]["message"].get("content", "").casefold(), response=baseline["body"])
    run.case("greedy_text", text)
    fixture_case = next(case for case in text_fixture["cases"] if case["name"] == "thinking_disabled")
    chat = body(copy.deepcopy(fixture_case["messages"]), chat_template_kwargs=fixture_case["options"])
    run.case("canonical_chat", lambda: run.pair("canonical-chat", chat, len(prompt_ids(chat))))

    def concurrent_requests():
        requests = [body([{"role": "user", "content": f"Write a short factual paragraph about {subject}."}], max_tokens=64)
            for subject in ("Saturn", "photosynthesis", "volcanoes", "binary search")]
        expected = [run.complete(run.request(f"concurrent-baseline-{i}", request), len(prompt_ids(request))) for i, request in enumerate(requests)]
        results = run.concurrent("concurrent-four", requests)
        run.check("four_simultaneous_requests_admitted", all(result.get("status") == 200 and not result.get("client_error") for result in results), statuses=[result.get("status") for result in results])
        run.check("four_client_intervals_overlap", max(result["started_monotonic"] for result in results) < min(result["finished_monotonic"] for result in results), intervals=[[result["started_monotonic"], result["finished_monotonic"]] for result in results])
        for i, result in enumerate(results):
            try:
                observed = run.complete(result, len(prompt_ids(requests[i])))
                run.check(f"concurrent-{i}.greedy_equivalence", comparable_message(expected[i]["message"]) == comparable_message(observed["message"])
                    and expected[i]["finish_reason"] == observed["finish_reason"], expected=expected[i], observed=observed)
            except ValueError:
                pass
    run.case("concurrent_four", concurrent_requests)

    def thinking():
        request = body([{"role": "user", "content": "What is 7 times 8? Give a brief explanation and the answer."}], enable_thinking=True, max_tokens=256)
        response, _ = run.pair("thinking-output", request, len(prompt_ids(request)))
        message = response["body"]["choices"][0]["message"]
        run.check("thinking_has_separate_reasoning", bool(message.get("reasoning_content")), message=message)
        run.check("thinking_answer", "56" in message.get("content", ""), message=message)
    run.case("thinking_output", thinking)

    def tools():
        schema = [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"], "additionalProperties": False}}}]
        request = body([{"role": "user", "content": "Use get_weather to find the current weather in Paris. Do not guess."}], tools=schema, max_tokens=192)
        response, _ = run.pair("tool-call", request, len(prompt_ids(request)))
        choice = response["body"]["choices"][0]
        calls = choice["message"].get("tool_calls", [])
        valid = len(calls) == 1 and calls[0].get("function", {}).get("name") == "get_weather"
        run.check("tool_call_is_present", valid and choice["finish_reason"] == "tool_calls", choice=choice)
        if valid:
            arguments = json.loads(calls[0]["function"]["arguments"])
            run.check("tool_call_arguments", arguments == {"city": "Paris"}, arguments=arguments)
            messages = copy.deepcopy(request["messages"]) + [choice["message"], {"role": "tool", "tool_call_id": calls[0]["id"], "content": '{"city":"Paris","temperature_c":17,"condition":"sunny"}'}]
            continuation = body(messages, tools=schema, max_tokens=96)
            try:
                answer, _ = run.pair("tool-result-continuation", continuation, len(prompt_ids(continuation)))
                run.check("tool_result_used", "17" in answer["body"]["choices"][0]["message"].get("content", ""), response=answer["body"])
            except ValueError:
                pass
        fixture = next(case for case in text_fixture["cases"] if case["name"] == "tool_results")
        request = body(copy.deepcopy(fixture["messages"]), tools=fixture["options"]["tools"], max_tokens=96)
        run.pair("canonical-tool-results", request, len(prompt_ids(request)))
    run.case("tool_call_and_result", tools)

    def aligned_request(group, target):
        for repeats in range(target + 64):
            content = f"Cache qualification {group}. Read this context and answer DONE.\nContext:" + " apple" * repeats + "\nAnswer:"
            request = body([{"role": "user", "content": content}], max_tokens=1)
            ids = prompt_ids(request)
            if len(ids) == target:
                return request, ids
        raise ValueError(f"cannot construct exact {target}-token canonical chat prompt")

    def prefixes():
        def measured(name, request, ids):
            before = run.metrics(name + "-before")
            completion = run.complete(run.request(name, request), len(ids))
            after = run.metrics(name + "-after")
            delta = metric_delta(before, after) if before is not None else None
            if delta is not None:
                run.results.setdefault("model_metric_deltas", []).append({"case": name, "delta": delta})
            return completion, delta

        full, full_ids = aligned_request("full", 128)
        seed, _ = measured("prefix-full-seed", full, full_ids)
        replay, delta = measured("prefix-full-reuse", full, full_ids)
        check_cache("prefix-full-reuse", delta, full, full_ids, len(full_ids))
        run.check("full_prompt_reuse_equivalence", comparable_message(seed["message"]) == comparable_message(replay["message"]), prompt_tokens=128,
            checkpoint_tokens=128, cache_hit_observed=delta is not None and delta["flare_cache_hits_total"] == 1)
        partial, partial_ids = aligned_request("branch", 160)
        seed, _ = measured("prefix-aligned-seed", partial, partial_ids)
        replay, delta = measured("prefix-aligned-reuse", partial, partial_ids)
        check_cache("prefix-aligned-reuse", delta, partial, partial_ids, len(partial_ids))
        run.check("exact_partial_page_prompt_reuse_equivalence", comparable_message(seed["message"]) == comparable_message(replay["message"]), prompt_tokens=160,
            checkpoint_tokens=160, tail_tokens=0, cache_hit_observed=delta is not None and delta["flare_cache_hits_total"] == 1)
        for name in ("A", "B"):
            branch = copy.deepcopy(partial)
            branch["messages"][0]["content"] += f"\nBranch {name}: return {name}."
            ids = prompt_ids(branch)
            shared = common_prefix(partial_ids, ids)
            run.check(f"branch-{name}.edit_precedes_exact_prompt_endpoint", 128 <= shared < len(partial_ids) and ids != partial_ids, common_prefix_tokens=shared, seed_checkpoint_tokens=len(partial_ids),
                seed_token_ids=partial_ids, branch_token_ids=ids)
            first, delta = measured(f"prefix-branch-{name}", branch, ids)
            system = system_prefix_ids(tokenizer, branch, ids)
            reused = len(system) if partial_ids[:len(system)] == system else 0
            check_cache(f"prefix-branch-{name}", delta, branch, ids, reused)
            repeated, repeat_delta = measured(f"prefix-branch-{name}-repeat", branch, ids)
            check_cache(f"prefix-branch-{name}-repeat", repeat_delta, branch, ids, len(ids))
            run.check(f"branch-{name}.repeat_equivalence", comparable_message(first["message"]) == comparable_message(repeated["message"]), first=first, repeated=repeated)
    run.case("prefix_full_aligned_branch", prefixes)

    def system_prefix():
        nonce = hashlib.sha256(run.results["started_at"].encode()).hexdigest()[:4]
        seen = []
        for name, request, expected_reuse in system_prefix_requests(model_id, nonce):
            ids = prompt_ids(request)
            prefix = system_prefix_ids(tokenizer, request, ids)
            if not prefix or len(prefix) % 64 == 0:
                raise ValueError("system-prefix fixture requires a nonaligned exact instruction endpoint")
            if name.startswith("short-") and len(prefix) >= 16:
                raise ValueError("short system-prefix fixture must contain fewer than 16 tokens")
            reusable = max((len(old) for old in seen if ids[:len(old)] == old), default=0)
            if bool(reusable) != expected_reuse or reusable not in (0, len(prefix)):
                raise ValueError("system-prefix fixture does not isolate instruction/tool identity")
            before = run.metrics("system-" + name + "-before")
            response = run.request("system-" + name, request)
            run.complete(response, len(ids))
            after = run.metrics("system-" + name + "-after")
            delta = metric_delta(before, after)
            check_cache("system-" + name, delta, request, ids, reusable)
            generated = response["body"]["usage"]["completion_tokens"]
            messages = sum(len(prefix) > reusable for prefix in message_prefixes(tokenizer, request, ids))
            run.check("system-" + name + ".retained_endpoints",
                delta["flare_message_checkpoints_total"] == messages
                and delta["flare_prompt_checkpoints_total"] == 1
                and delta["flare_snapshot_publications_total"] == messages + 1 + int(generated > 1),
                system_token_ids=prefix, prompt_tokens=len(ids), generated_tokens=generated, delta=delta)
            run.reclaimed("system-" + name, before, after)
            seen.append(prefix)
    if args.require_metrics:
        run.case("system_prefix", system_prefix)

    if args.checkpoint_endpoints:
        from qualify.checkpoint_http import run_endpoint_cases

        generation = json.loads((snapshot / "generation_config.json").read_text())
        eos_ids = generation["eos_token_id"]
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        run_endpoint_cases(run, tokenizer, model_id,
                           hashlib.sha256(run.results["started_at"].encode()).hexdigest()[:12],
                           chunk_size, set(eos_ids))

    def structured():
        literal = "<think>analysis</think><tool_call><function=lookup></function></tool_call>"
        expected = {"literal": literal, "ok": True}
        request = body([{"role": "user", "content": "Return exactly the object required by the JSON schema."}], max_tokens=256,
            response_format={"type": "json_schema", "json_schema": {"name": "literal_tags", "strict": True,
                "schema": {"type": "object", "properties": {"literal": {"enum": [literal]}, "ok": {"enum": [True]}}, "required": ["literal", "ok"], "additionalProperties": False}}})
        response, stream = run.pair("structured-literal-tags", request, len(prompt_ids(request)))
        message = response["body"]["choices"][0]["message"]
        decoded = json.loads(message["content"])
        run.check("structured_literal_tags_are_verbatim", decoded == expected and not message.get("reasoning_content") and not message.get("tool_calls"), expected=expected, observed=decoded)
        run.check("structured_stream_json_is_valid", json.loads(stream.get("sse", {}).get("message", {}).get("content", "")) == expected)
    run.case("structured_literal_tags", structured)

    def image():
        fixture = next(case for case in image_fixture["cases"] if case["name"] == "png_RGB")
        source = ROOT / "tests/frontend" / fixture["inputs"][0]["png"]
        if digest(source) != fixture["inputs"][0]["sha256"]:
            raise ValueError("PNG fixture bytes differ from canonical processor results")
        messages = copy.deepcopy(fixture["messages"])
        encoded = base64.b64encode(source.read_bytes()).decode()
        for message in messages:
            if isinstance(message.get("content"), list):
                for i, part in enumerate(message["content"]):
                    if part.get("type") == "image":
                        message["content"][i] = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}}
        request = body(messages, max_tokens=32)
        tokens = len(prompt_ids(request)) + sum(t * h * w // 4 - 1 for t, h, w in fixture["grid_thw"])
        for repeat in range(args.image_repetitions):
            run.pair("png-image" if repeat == 0 else f"png-image-repeat-{repeat}", request, tokens)
    run.case("png_image", image)

    def full_vision():
        from PIL import Image

        fixture = next(case for case in image_fixture["cases"] if case["name"] == "bucket_16384")
        source = args.full_vision_png.resolve()
        descriptor = fixture["inputs"][0]
        with Image.open(source) as image:
            if image.format != "PNG" or image.size != (descriptor["width"], descriptor["height"]):
                raise ValueError("full-vision PNG geometry differs from the pinned fixture")
            rgb_hash = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
        if rgb_hash != fixture["input_rgb_sha256"][0] or fixture["grid_thw"] != [[1, 128, 128]]:
            raise ValueError("full-vision RGB content or patch geometry differs from the pinned fixture")
        messages = copy.deepcopy(fixture["messages"])
        encoded = base64.b64encode(source.read_bytes()).decode()
        image_count = 0
        for message in messages:
            for index, part in enumerate(message["content"]):
                if part.get("type") == "image":
                    message["content"][index] = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}}
                    image_count += 1
        if image_count != 1:
            raise ValueError("full-vision fixture must contain exactly one image")
        request = body(messages, max_tokens=32)
        tokens = len(prompt_ids(request)) + 4096 - 1
        before = run.metrics("full-vision-before")
        run.pair("full-vision", request, tokens)
        after = run.metrics("full-vision-after")
        delta = metric_delta(before, after)
        run.check("full_vision.completes_both_encodes", delta['flare_graph_launches_total{kind="vision"}'] == 2
            and delta["flare_requests_admitted_total"] == delta["flare_requests_completed_total"] == 2
            and delta["flare_requests_failed_total"] == delta["flare_requests_cancelled_total"] == 0,
            delta=delta, patches_per_image=16384, features_per_image=4096)
        run.reclaimed("full-vision", before, after)
    if args.full_vision_png:
        run.case("png_full_capacity", full_vision)

    def image_cache_identity():
        nonce = hashlib.sha256(run.results["started_at"].encode()).hexdigest()[:12]
        requests, fixtures = [], []
        for name in ("png_RGB", "png_L"):
            fixture = next(case for case in image_fixture["cases"] if case["name"] == name)
            source = ROOT / "tests/frontend" / fixture["inputs"][0]["png"]
            if digest(source) != fixture["inputs"][0]["sha256"]:
                raise ValueError("image identity fixture PNG differs from the pinned input")
            messages = copy.deepcopy(fixture["messages"])
            encoded = base64.b64encode(source.read_bytes()).decode()
            for message in messages:
                for index, part in enumerate(message["content"]):
                    if part.get("type") == "text":
                        part["text"] = f"Image identity {nonce}. " + part["text"]
                    elif part.get("type") == "image":
                        message["content"][index] = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}}
            requests.append(body(messages, max_tokens=8))
            fixtures.append(fixture)
        first, second = (prompt_ids(request) for request in requests)
        if first != second or fixtures[0]["grid_thw"] != fixtures[1]["grid_thw"]:
            raise ValueError("image identity requests must have identical text tokens and image geometry")
        if first.count(248056) != 1 or first.index(248056) >= 64:
            raise ValueError("image identity must distinguish the first cache block")
        if fixtures[0]["input_rgb_sha256"] == fixtures[1]["input_rgb_sha256"]:
            raise ValueError("image identity requests require different decoded pixels")
        grid = fixtures[0]["grid_thw"][0]
        tokens = len(first) + grid[0] * grid[1] * grid[2] // 4 - 1
        before = run.metrics("image-identity-before")
        last = before
        cold = {}
        for repeated in (False, True):
            for index, request in enumerate(requests):
                name = f"image-identity-{index}-{'repeat' if repeated else 'cold'}"
                response = run.request(name, request)
                messages = len(message_prefixes(tokenizer, request, prompt_ids(request)))
                choice = run.complete(response, tokens)
                after = run.metrics(name + "-after")
                delta = metric_delta(last, after)
                run.check(name + ".conditioned_prefix", delta["flare_cache_hits_total"] == int(repeated)
                    and delta["flare_cache_misses_total"] == int(not repeated)
                    and delta["flare_cache_reused_tokens_total"] == (tokens if repeated else 0)
                    and delta['flare_graph_launches_total{kind="restore"}'] == int(repeated)
                    and delta['flare_graph_launches_total{kind="vision"}'] == 1
                    and delta["flare_prompt_checkpoints_total"] == int(not repeated)
                    and delta["flare_snapshot_publications_total"] == (0 if repeated else messages + 1 + int(response["body"]["usage"]["completion_tokens"] > 1)),
                    delta=delta, checkpoint_tokens=tokens)
                if repeated:
                    run.check(name + ".greedy_cache_equivalence",
                        comparable_message(choice["message"]) == comparable_message(cold[index]["message"])
                        and choice["finish_reason"] == cold[index]["finish_reason"], expected=cold[index], observed=choice)
                else:
                    cold[index] = choice
                run.reclaimed(name, before, after)
                last = after
        delta = metric_delta(before, last)
        run.check("image_identity.all_requests_complete", delta["flare_requests_admitted_total"] == 4
            and delta["flare_requests_completed_total"] == 4
            and delta["flare_requests_failed_total"] == delta["flare_requests_cancelled_total"] == 0,
            delta=delta)
    if args.image_cache_identity:
        run.case("image_cache_identity", image_cache_identity)

    def mixed_grammar():
        literal = "alpha " * 512
        constrained = body([{"role": "user", "content": "Return the exact string required by the response schema."}],
            response_format={"type": "json_schema", "json_schema": {"name": "mixed_grammar", "strict": True,
                "schema": {"enum": [literal]}}}, max_tokens=1024)
        before = run.metrics("mixed-grammar-before")
        progress = StreamProgress()
        cancel = threading.Event()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(run.request, "mixed-grammar-constrained", constrained,
                stream=True, progress=progress, cancel_event=cancel)
            try:
                if not progress.wait_after(0, min(args.timeout, 30)):
                    raise ValueError("constrained generation did not produce visible output")
                plain = run.complete(run.request("mixed-grammar-plain", capital), len(prompt_ids(capital)))
                constrained_result = future.result()
            except BaseException:
                cancel.set()
                raise
        combined = constrained_result.get("sse", {})
        run.check("mixed_grammar.constrained_stream_complete", constrained_result.get("status") == 200
            and constrained_result.get("done_count") == 1 and not constrained_result.get("client_error")
            and not constrained_result.get("incomplete_event") and not combined.get("errors")
            and combined.get("finish_reasons") == ["stop"], response=constrained_result)
        content = combined.get("message", {}).get("content", "")
        try:
            decoded = json.loads(content)
        except json.JSONDecodeError:
            decoded = None
        run.check("mixed_grammar.exact_constrained_output", decoded == literal, content=content)
        run.check("mixed_grammar.unconstrained_answer", "paris" in plain["message"].get("content", "").casefold(), choice=plain)
        after = run.metrics("mixed-grammar-after")
        delta = metric_delta(before, after)
        run.check("mixed_grammar.shared_decode_observed", delta["flare_mixed_grammar_decode_steps_total"] > 0,
            delta=delta)
        run.check("mixed_grammar.one_submission_inflight", after["flare_constrained_max_inflight"] == 1,
            observed_maximum=after["flare_constrained_max_inflight"])
        run.check("mixed_grammar.both_requests_complete", delta["flare_requests_admitted_total"] == 2
            and delta["flare_requests_completed_total"] == 2
            and delta["flare_requests_failed_total"] == delta["flare_requests_cancelled_total"] == 0,
            delta=delta)
        run.reclaimed("mixed-grammar", before, after)
    if args.mixed_grammar:
        run.case("mixed_grammar", mixed_grammar)

    def cancellation():
        request = body([{"role": "user", "content": "List every integer from 1 through 10000 in order, separated by commas. Do not summarize or stop early."}], max_tokens=2048)
        before = run.metrics("cancel-before")
        cancelled = run.request("cancel-live-stream", request, stream=True, cancel=True)
        run.check("client_disconnected_after_visible_output", cancelled.get("status") == 200 and cancelled.get("cancelled") is True and cancelled.get("done_count") == 0,
            visible_seconds=cancelled.get("first_visible_delta_seconds"), received_events=len(cancelled.get("events", [])), client_error=cancelled.get("client_error"))
        after = run.metrics("cancel-after")
        if before is not None:
            delta = metric_delta(before, after)
            run.check("disconnect_cancels_active_generation", delta["flare_requests_admitted_total"] == 1
                and delta["flare_requests_cancelled_total"] == 1 and delta["flare_requests_completed_total"] == 0,
                delta=delta)
            run.reclaimed("cancel", before, after)
        health("health-after-cancel")
        for index in range(4):
            run.complete(run.request(f"after-cancel-{index}", capital), len(prompt_ids(capital)))
        reused = run.metrics("slot-reuse-after")
        if after is not None:
            delta = metric_delta(after, reused)
            run.check("reclaimed_slots_accept_four_generations", delta["flare_requests_admitted_total"] == 4
                and delta["flare_requests_completed_total"] == 4
                and delta["flare_requests_cancelled_total"] == delta["flare_requests_failed_total"] == 0, delta=delta)
            run.reclaimed("slot-reuse", before, reused)
    run.case("cancellation_and_slot_reuse", cancellation)

    def cache_pressure():
        before = run.metrics("pressure-before")
        capacity = resource_totals(before)["snapshots"]
        if not 2 <= capacity <= 256:
            raise ValueError(f"snapshot pressure sweep requires 2..256 slots; server exposes {capacity}")
        if resource_totals(before)["pages"] < 2 * capacity + 1:
            raise ValueError("KV capacity cannot isolate snapshot pressure with 128-token prefixes")
        nonce = hashlib.sha256(run.results["started_at"].encode()).hexdigest()[:12]
        inputs = [aligned_request(f"pressure-{nonce}-{index}", 128) for index in range(capacity + 1)]
        if any(system_prefix_ids(tokenizer, request, ids) for request, ids in inputs):
            raise ValueError("snapshot-pressure fixtures require no synthesized instruction checkpoint")
        if len({tuple(ids[:64]) for _, ids in inputs}) != len(inputs):
            raise ValueError("snapshot pressure inputs share a complete cache block")
        first = None
        last = before
        for index, (request, ids) in enumerate(inputs):
            result = run.complete(run.request(f"pressure-fill-{index}", request), len(ids))
            first = result if first is None else first
            after = run.metrics(f"pressure-fill-{index}-after")
            delta = metric_delta(last, after)
            run.check(f"pressure-fill-{index}.unique_publication", delta["flare_cache_misses_total"] == 1
                and delta["flare_cache_hits_total"] == 0 and delta["flare_snapshot_publications_total"] == 2
                and delta['flare_graph_launches_total{kind="snapshot"}'] == 2, delta=delta)
            last = after
        delta = metric_delta(before, last)
        run.check("snapshot_capacity_is_exercised", last["flare_cached_checkpoints"] == capacity
            and last["flare_free_snapshots"] == 0 and delta["flare_snapshot_evictions_total"] > 0
            and delta["flare_snapshot_publications_total"] == 2 * (capacity + 1),
            capacity=capacity, published_prefixes=len(inputs), delta=delta)
        run.reclaimed("pressure-fill", before, last)
        request, ids = inputs[0]
        repeated = run.complete(run.request("pressure-evicted-prefix", request), len(ids))
        after = run.metrics("pressure-evicted-prefix-after")
        delta = metric_delta(last, after)
        run.check("oldest_prefix_was_evicted", delta["flare_cache_misses_total"] == 1
            and delta["flare_cache_hits_total"] == 0 and delta['flare_graph_launches_total{kind="init"}'] == 1
            and delta['flare_graph_launches_total{kind="restore"}'] == 0
            and delta["flare_snapshot_evictions_total"] == delta["flare_snapshot_publications_total"] == 2, delta=delta)
        run.check("evicted_prefix_cold_replay_equivalence", comparable_message(first["message"]) == comparable_message(repeated["message"])
            and first["finish_reason"] == repeated["finish_reason"], expected=first, observed=repeated)
        delta = metric_delta(before, after)
        run.check("pressure_requests_complete_without_failure", delta["flare_requests_admitted_total"] == capacity + 2
            and delta["flare_requests_completed_total"] == capacity + 2
            and delta["flare_requests_cancelled_total"] == delta["flare_requests_failed_total"] == 0, delta=delta)
        run.reclaimed("pressure-evicted-prefix", before, after)
    if args.cache_pressure:
        run.case("cache_pressure", cache_pressure)

    def conversation_pressure():
        before = run.metrics("conversation-before")
        capacity = resource_totals(before)["snapshots"]
        if not 2 <= capacity <= 256:
            raise ValueError(f"conversation snapshot pressure requires 2..256 slots; server exposes {capacity}")
        turns = capacity + 3
        chunk = memory["prefill_chunk_size"]
        if type(chunk) is not int or chunk <= 0:
            raise ValueError("conversation pressure requires the reported prefill chunk size")
        # Edits before each prompt endpoint do not reuse complete page ownership.
        max_pages = sum((160 + 64 * turn) // 64 for turn in range(turns - capacity, turns))
        max_pages += (160 + 64 * (turns - 1) + 1 + 63) // 64
        if resource_totals(before)["pages"] < max_pages:
            raise ValueError("KV capacity cannot isolate independent exact-prompt snapshot pressure")
        nonce = hashlib.sha256(run.results["started_at"].encode()).hexdigest()[:12]
        request, ids = aligned_request(f"conversation-{nonce}", 160)
        retained_lengths = []
        last = before
        for turn in range(turns):
            if turn:
                previous = ids
                content = request["messages"][0]["content"]
                if not content.endswith("\nAnswer:"):
                    raise ValueError("conversation fixture has no append boundary")
                request["messages"][0]["content"] = content[:-len("\nAnswer:")] + " apple" * 64 + "\nAnswer:"
                ids = prompt_ids(request)
                if len(ids) != len(previous) + 64 or not 128 <= common_prefix(previous, ids) < len(previous):
                    raise ValueError("conversation edit must precede the previous exact prompt endpoint")
            if system_prefix_ids(tokenizer, request, ids):
                raise ValueError("conversation-pressure fixtures require no synthesized instruction checkpoint")
            run.complete(run.request(f"conversation-turn-{turn}", request), len(ids))
            after = run.metrics(f"conversation-turn-{turn}-after")
            delta = metric_delta(last, after)
            check_cache(f"conversation-turn-{turn}", delta, request, ids, 0)
            run.check(f"conversation-turn-{turn}.exact_prompt_publication",
                delta["flare_snapshot_publications_total"] == 2
                and delta['flare_graph_launches_total{kind="snapshot"}'] == 2
                and delta["flare_message_checkpoints_total"] == 1
                and delta["flare_prompt_checkpoints_total"] == 0,
                prompt_tokens=len(ids), terminal_checkpoint_tokens=len(ids), delta=delta)
            run.reclaimed(f"conversation-turn-{turn}", before, after)
            retained_lengths.extend([len(prefix) for prefix in message_prefixes(tokenizer, request, ids)] + [len(ids)])
            last = after
        delta = metric_delta(before, last)
        pairs = capacity // 2
        retained_pages = sum(length // 64 for length in retained_lengths[-2 * pairs + 1::2])
        if capacity % 2:
            retained_pages += retained_lengths[-2 * (pairs + 1)] // 64
        run.check("conversation_exceeds_snapshot_capacity",
            delta["flare_requests_admitted_total"] == turns
            and delta["flare_requests_completed_total"] == turns
            and delta["flare_requests_failed_total"] == delta["flare_requests_cancelled_total"] == 0
            and delta["flare_snapshot_evictions_total"] >= 2 * turns - capacity
            and last["flare_cached_checkpoints"] == capacity
            and last["flare_cached_kv_pages"] == retained_pages,
            turns=turns, snapshot_capacity=capacity, independent_checkpoint_pages=retained_pages, delta=delta)
        run.complete(run.request("conversation-repeat", request), len(ids))
        after = run.metrics("conversation-repeat-after")
        delta = metric_delta(last, after)
        check_cache("conversation-repeat", delta, request, ids, len(ids))
        run.check("conversation_repeat_restores_newest_checkpoint",
            delta["flare_snapshot_publications_total"] == 0
            and delta['flare_graph_launches_total{kind="snapshot"}'] == 0,
            restored_tokens=len(ids), delta=delta)
        run.reclaimed("conversation-repeat", before, after)
    if args.conversation_pressure:
        run.case("conversation_pressure", conversation_pressure)

    def kv_pressure():
        before = run.metrics("kv-before")
        geometry = kv_pressure_geometry(memory, before, args.prepared_context_tokens)
        run.results["kv_pressure_geometry"] = geometry
        count = geometry["holding_requests"]
        nonce = hashlib.sha256(run.results["started_at"].encode()).hexdigest()[:12]

        def reserved_request(index, literal):
            request = body([{"role": "user", "content": f"KV reservation {nonce} {index}. Follow the response schema."}],
                response_format={"type": "json_schema", "json_schema": {"name": "kv_reservation", "strict": True,
                    "schema": {"enum": [literal]}}})
            tokens = len(prompt_ids(request))
            if tokens >= 64 or tokens >= geometry["context_tokens"]:
                raise ValueError("KV pressure requires a prompt shorter than one cache block and the context limit")
            if system_prefix_ids(tokenizer, request, prompt_ids(request)):
                raise ValueError("KV pressure fixtures require no synthesized instruction checkpoint")
            request["max_tokens"] = geometry["context_tokens"] - tokens
            return request

        boundary = reserved_request("context-boundary", "x")
        boundary["max_tokens"] += 1
        invalid = run.request("kv-context-plus-one", boundary, stream=True, cancel=True)
        valid_boundary = invalid.get("status") == 400 and "exceeds this server's context of" in invalid.get("body", {}).get("error", {}).get("message", "")
        run.check("kv.prepared_context_upper_bound", valid_boundary, response=invalid)
        if not valid_boundary:
            run.metrics("kv-context-probe-cleanup")
            raise ValueError("configured context upper bound was not confirmed by HTTP admission")

        progress = [StreamProgress() for _ in range(count)]
        cancel = threading.Event()
        futures = []
        results = []
        expected_error = "live request resources are exhausted or awaiting GPU retirement"

        def observed(name, predicate):
            deadline = time.monotonic() + min(args.timeout, 30)
            attempt = 0
            while True:
                metrics = run.metrics(f"{name}-{attempt}", wait_idle=False)
                if predicate(metrics):
                    return metrics
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{name}: native metrics did not reach the required admission state")
                attempt += 1
                time.sleep(0.01)

        with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
            try:
                for index in range(count):
                    future = pool.submit(run.request, f"kv-holder-{index}", reserved_request(index, "x " * 4096),
                        stream=True, progress=progress[index], cancel_event=cancel)
                    futures.append(future)
                    if not progress[index].wait_after(0, min(args.timeout, 30)):
                        raise ValueError(f"KV holder {index} did not produce visible output")
                held = observed("kv-held", lambda metrics: metrics["flare_active_requests"] == count
                    and resource_totals(metrics)["pages"] == geometry["unreserved_pages"]
                    and metrics["flare_prompt_checkpoints_total"] == before["flare_prompt_checkpoints_total"] + count)
                run.check("kv.context_reservations_observed", held["flare_free_request_slots"] == 4 - count
                    and resource_totals(held)["snapshots"] == resource_totals(before)["snapshots"] - count,
                    geometry=geometry, held=held)
                rejected = run.request("kv-rejected-json", reserved_request("rejected-json", "x"))
                run.check("kv.excess_json_admission_rejected", rejected.get("status") == 503
                    and rejected.get("body", {}).get("error", {}).get("message") == expected_error,
                    status=rejected.get("status"), response=rejected.get("body"), client_error=rejected.get("client_error"))
                rejected_stream = run.request("kv-rejected-stream", reserved_request("rejected-stream", "x"), stream=True)
                run.check("kv.excess_stream_admission_rejected", rejected_stream.get("status") == 200
                    and rejected_stream.get("sse", {}).get("errors") == [{"message": expected_error}]
                    and rejected_stream.get("done_count") == 0 and not rejected_stream.get("client_error")
                    and not rejected_stream.get("sse", {}).get("message", {}).get("content"), response=rejected_stream)
                rejected_state = observed("kv-rejected", lambda metrics:
                    metrics["flare_requests_failed_total"] >= before["flare_requests_failed_total"] + 2)
                delta = metric_delta(before, rejected_state)
                run.check("kv.rejection_preserves_existing_reservations", rejected_state["flare_active_requests"] == count
                    and rejected_state["flare_free_request_slots"] == 4 - count
                    and rejected_state["flare_retiring_request_slots"] == rejected_state["flare_pending_requests"] == 0
                    and rejected_state["flare_free_kv_pages"] == geometry["unreserved_pages"]
                    and rejected_state["flare_cached_kv_pages"] == 0
                    and rejected_state["flare_cached_checkpoints"] == 2 * count
                    and rejected_state["flare_free_snapshots"] == resource_totals(before)["snapshots"] - 3 * count
                    and delta["flare_prompt_checkpoints_total"] == count
                    and delta["flare_message_checkpoints_total"] == count
                    and delta["flare_requests_admitted_total"] == count and delta["flare_requests_failed_total"] == 2
                    and delta["flare_requests_completed_total"] == delta["flare_requests_cancelled_total"] == 0
                    and delta["flare_cache_hits_total"] == delta["flare_cache_reused_tokens_total"] == 0,
                    delta=delta, metrics=rejected_state)
                checkpoints = [observer.count() for observer in progress]
                for index, observer in enumerate(progress):
                    continued = observer.wait_after(checkpoints[index], min(args.timeout, 10))
                    run.check(f"kv.holder-{index}-progresses-after-rejection", continued,
                        visible_events_before=checkpoints[index], visible_events_after=observer.count())
                health("health-under-kv-pressure")
            finally:
                cancel.set()
                results = [future.result() for future in futures]
                after = run.metrics("kv-cancelled")
                run.reclaimed("kv-cancellation", before, after)
        for index, result in enumerate(results):
            run.check(f"kv.holder-{index}-cancelled-without-error", result.get("status") == 200
                and result.get("cancelled") is True and result.get("done_count") == 0
                and not result.get("client_error") and not result.get("sse", {}).get("errors")
                and not result.get("sse", {}).get("finish_reasons"), response=result,
                received_visible_events=progress[index].count())
        delta = metric_delta(before, after)
        run.check("kv.all_holders_retired_after_cancellation", delta["flare_requests_admitted_total"] == count
            and delta["flare_requests_cancelled_total"] == count and delta["flare_requests_failed_total"] == 2
            and delta["flare_requests_completed_total"] == 0
            and delta["flare_snapshot_publications_total"] == 2 * count
            and delta["flare_prompt_checkpoints_total"] == delta["flare_message_checkpoints_total"] == count, delta=delta)
        answer = run.complete(run.request("kv-followup", capital), len(prompt_ids(capital)))
        run.check("kv.healthy_followup_answer", "paris" in answer["message"].get("content", "").casefold(), choice=answer)
        final = run.metrics("kv-followup-after")
        delta = metric_delta(after, final)
        run.check("kv.reclaimed_resources_serve_followup", delta["flare_requests_admitted_total"] == delta["flare_requests_completed_total"] == 1
            and delta["flare_requests_failed_total"] == delta["flare_requests_cancelled_total"] == 0, delta=delta)
        run.reclaimed("kv-followup", before, final)
        health("health-after-kv-pressure")
    if args.kv_pressure:
        run.case("kv_pressure", kv_pressure)

    def slow_consumer():
        literal = "x " * 4096
        request = body([{"role": "user", "content": "Return the exact JSON string required by the schema."}], max_tokens=8192,
            response_format={"type": "json_schema", "json_schema": {"name": "stalled_reader", "strict": True,
                "schema": {"enum": [literal]}}})
        before = run.metrics("slow-reader-before")
        paused, resume = threading.Event(), threading.Event()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(run.request, "slow-reader-stream", request, stream=True, cancel=True,
                paused=paused, resume=resume, receive_buffer=1024)
            try:
                deadline = time.monotonic() + args.timeout
                while not paused.wait(timeout=0.02):
                    if future.done():
                        raise ValueError(f"SSE reader ended before visible output: {future.result()}")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("SSE reader did not reach visible output")
                started = time.monotonic()
                held = run.metrics("slow-reader-held", wait_idle=False)
                followup = run.request("progress-with-stalled-reader", capital)
                answer = run.complete(followup, len(prompt_ids(capital)))
                run.check("unrelated_request_completes_with_stalled_reader", "paris" in answer["message"].get("content", "").casefold()
                    and not future.done(), choice=answer, elapsed_seconds=followup["elapsed_seconds"])
                remaining = args.slow_consumer_seconds - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
                preclose = run.metrics("slow-reader-preclose", wait_idle=False)
            finally:
                resume.set()
            cancelled = future.result()
        after = run.metrics("slow-reader-after")
        delta = metric_delta(before, after)
        held_delta = metric_delta(before, preclose)
        run.check("stalled_reader_observed_during_other_progress", held["flare_http_inflight_requests"] >= 1
            and cancelled.get("paused_read_seconds", 0) >= args.slow_consumer_seconds
            and cancelled.get("cancelled") is True and cancelled.get("done_count") == 0, stream=cancelled)
        run.check("stalled_reader_cancelled_and_other_request_completed", delta["flare_requests_admitted_total"] == 2
            and delta["flare_requests_cancelled_total"] == 1 and delta["flare_requests_completed_total"] == 1,
            delta=delta, preclose_delta=held_delta)
        run.results["slow_consumer_observation"] = {
            "native_output_backpressure_observed": held_delta["flare_requests_failed_total"] > 0
                and held_delta["flare_requests_cancelled_total"] > 0,
            "preclose_delta": held_delta, "final_delta": delta,
            "paused_read_seconds": cancelled.get("paused_read_seconds"),
            "socket_receive_buffer_bytes": cancelled.get("socket_receive_buffer_bytes")}
        run.reclaimed("stalled-reader", before, after)
        health("health-after-stalled-reader")
    if args.slow_consumer_seconds:
        run.case("slow_consumer", slow_consumer)
    run.case("final_health", lambda: health("health-after"))
    run.results["finished_at"] = datetime.now(timezone.utc).isoformat()
    run.results["summary"] = {"requests": len(run.results["requests"]), "checks": len(run.results["checks"]),
        "failed_checks": sum(not check["passed"] for check in run.results["checks"]), "case_errors": len(run.results["case_errors"])}
    run.results["passed"] = run.results["summary"]["failed_checks"] == 0 and not run.results["case_errors"]
    print(json.dumps({**run.results["summary"], "passed": run.results["passed"]}), flush=True)
    return 0 if run.results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
