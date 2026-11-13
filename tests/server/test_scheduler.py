# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concurrent streams through a native service with four slots and 4096-token prefill.

Set FLARE_RUN_CUDA_TESTS=1 and FLARE_SCHEDULER_TEST_URL to an otherwise idle
service. Run against ordinary, fixed-width, and adaptive speculative execution.
Printed measurements are client SSE arrival times and server counter deltas.
"""

import json
import os
import re
import threading
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1"
    or not os.environ.get("FLARE_SCHEDULER_TEST_URL"),
    reason="requires an idle native service with four slots and 4096-token prefill",
)


@dataclass
class Stream:
    name: str
    prompt: str
    max_tokens: int
    first: threading.Event = field(default_factory=threading.Event)
    started: float = 0
    finished: float = 0
    events: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    finish_reason: str | None = None

    @property
    def text(self):
        return "".join(text for _, text in self.events)

    def measurement(self):
        times = [at for at, _ in self.events]
        return {
            "name": self.name,
            "usage": self.usage,
            "ttft_ms": round(1000 * (times[0] - self.started), 3),
            "elapsed_ms": round(1000 * (self.finished - self.started), 3),
            "max_sse_gap_ms": round(
                1000 * max((b - a for a, b in zip(times, times[1:])), default=0), 3
            ),
        }


@pytest.fixture(scope="module")
def service():
    url = os.environ["FLARE_SCHEDULER_TEST_URL"].rstrip("/")
    with urllib.request.urlopen(url + "/v1/models", timeout=10) as response:
        model = json.load(response)["data"][0]["id"]
    return url, model


def metrics(service):
    with urllib.request.urlopen(service[0] + "/metrics", timeout=10) as response:
        return {
            key: float(value)
            for line in response.read().decode().splitlines()
            if line and not line.startswith("#")
            for key, value in [line.rsplit(" ", 1)]
        }


def stream(service, request, barrier=None, cancel_after=None):
    body = {
        "model": service[1],
        "messages": [{"role": "user", "content": request.prompt}],
        "temperature": 1,
        "top_p": 1,
        "top_k": 0,
        "min_p": 0,
        "repetition_penalty": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "seed": 42,
        "max_tokens": request.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if barrier:
        barrier.wait(timeout=10)
    request.started = time.monotonic()
    http = urllib.request.Request(
        service[0] + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    done = False
    with urllib.request.urlopen(http, timeout=180) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            assert "error" not in event, event
            if event.get("usage"):
                request.usage = event["usage"]
            for choice in event.get("choices", []):
                if choice.get("finish_reason"):
                    request.finish_reason = choice["finish_reason"]
                text = choice.get("delta", {}).get("content")
                if text:
                    request.events.append((time.monotonic(), text))
                    request.first.set()
            if cancel_after and len(request.events) >= cancel_after:
                request.finished = time.monotonic()
                return request
    request.finished = time.monotonic()
    assert done and request.usage and request.events, request
    assert request.finish_reason in ("stop", "length"), request
    return request


def prompt_request(name, words=16384):
    prompt = (
        f"Context identifier {uuid.uuid4().hex}. Ignore the following filler.\n"
        + " apple" * words
        + "\nWhat is 19 multiplied by 23? Reply with the number only."
    )
    return Stream(name, prompt, 32)


def decode_request(name):
    prompt = (
        f"Context identifier {uuid.uuid4().hex}. "
        "Count from 1 through 256 inclusive, separated by commas. "
        "Output only the numbers, with no omissions."
    )
    return Stream(name, prompt, 2048)


def check_answer(request):
    if request.name.startswith("prefill"):
        assert request.text.strip() == "437", request.text
    else:
        assert [int(n) for n in re.findall(r"\d+", request.text)] == list(range(1, 257)), request.text


def measure(service, name, requests, *, stagger_prefill=False):
    before = metrics(service)
    assert before["flare_active_requests"] == 0, before
    with ThreadPoolExecutor(max_workers=len(requests)) as workers:
        barrier = threading.Barrier(len(requests)) if not stagger_prefill else None
        futures = []
        for request in requests:
            if stagger_prefill and request.name.startswith("prefill"):
                for earlier in requests[:len(futures)]:
                    assert earlier.first.wait(timeout=30), earlier.name
            futures.append(workers.submit(stream, service, request, barrier))
        for future in futures:
            check_answer(future.result(timeout=180))
    deadline = time.monotonic() + 10
    after = metrics(service)
    while after["flare_active_requests"] and time.monotonic() < deadline:
        time.sleep(0.01)
        after = metrics(service)
    assert after["flare_active_requests"] == 0
    delta = {key: value - before.get(key, 0) for key, value in after.items()}
    assert delta["flare_requests_completed_total"] == len(requests), delta
    assert delta.get("flare_requests_failed_total", 0) == 0, delta
    launches = {
        key: int(value) for key, value in delta.items()
        if value and (key.startswith("flare_graph_launches_total")
                      or key.startswith("flare_speculative_rounds_total"))
    }
    print(json.dumps({"scenario": name, "requests": [r.measurement() for r in requests],
                      "prefill_tokens": delta["flare_prefill_tokens_total"],
                      "launches": launches}, sort_keys=True))
    return delta


def test_two_long_prefills(service):
    requests = [prompt_request(f"prefill-{i}") for i in range(2)]
    delta = measure(service, "two_long_prefills", requests)
    assert all(r.usage["prompt_tokens"] > 16000 for r in requests)
    prefill_launches = delta['flare_graph_launches_total{kind="prefill"}']
    if not delta['flare_graph_launches_total{kind="mixed"}']:
        assert delta["flare_prefill_tokens_total"] / prefill_launches > 1024, delta


@pytest.mark.parametrize("count", [2, 4])
def test_concurrent_decodes(service, count):
    requests = [decode_request(f"decode-{i}") for i in range(count)]
    measure(service, f"{count}_long_decodes", requests)
    assert max(r.events[0][0] for r in requests) < min(r.events[-1][0] for r in requests)


@pytest.mark.parametrize("decode_count", [1, 3])
def test_prefill_during_decode(service, decode_count):
    decodes = [decode_request(f"decode-{i}") for i in range(decode_count)]
    prefill = prompt_request("prefill-long", words=32768)
    measure(service, f"long_prefill_{decode_count}_decodes", [*decodes, prefill], stagger_prefill=True)
    for decode in decodes:
        assert any(prefill.started < at < prefill.events[0][0] for at, _ in decode.events)


def test_small_prompt_during_long_prefill(service):
    measure(service, "long_and_short_prefill", [prompt_request("prefill-long", words=32768),
                                               prompt_request("prefill-short", words=16)])


def test_cancel_and_reuse_slot(service):
    canceled = decode_request("decode-canceled")
    stream(service, canceled, cancel_after=16)
    deadline = time.monotonic() + 10
    while metrics(service)["flare_active_requests"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert metrics(service)["flare_active_requests"] == 0
    measure(service, "after_cancel", [decode_request(f"decode-{i}") for i in range(4)])
