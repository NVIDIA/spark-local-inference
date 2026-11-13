#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise native speculative serving, batching, prefix reuse, and cancellation.

Run against an already loaded Flare service. Output contains HTTP replies and
raw timing/counter deltas so the same workload can compare execution catalogs.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import re
import threading
import time
import urllib.error
import urllib.request


PROMPT = "\n".join(
    f"Service {i}: workers={2 + i % 4}, jobs_per_second={30 + i}, queue_limit={128 + i * 8}."
    for i in range(24)
) + "\nExplain how to assess these services for queue stability. Give calculations and practical tests."


def get(url, path):
    with urllib.request.urlopen(url + path, timeout=30) as response:
        return response.read().decode()


def metrics(url):
    return {key: int(value) for key, value in re.findall(
        r"^(flare_[^\s]+) (\d+)$", get(url, "/metrics"), re.M)}


def body(model, seed, tokens, *, stream=False):
    return dict(model=model, messages=[dict(role="user", content=PROMPT)],
        temperature=1.0, top_p=.95, top_k=0, seed=seed, max_tokens=tokens,
        return_token_ids=True, return_token_timestamps=True, stream=stream,
        chat_template_kwargs=dict(thinking=True, enable_thinking=True))


def post(url, request):
    try:
        return urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions",
            json.dumps(request).encode(), {"Content-Type": "application/json"}), timeout=600)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode()}") from error


def request(url, model, seed, tokens, barrier=None):
    if barrier is not None:
        barrier.wait()
    start = time.monotonic_ns()
    with post(url, body(model, seed, tokens)) as response:
        reply = json.load(response)
    elapsed = time.monotonic_ns() - start
    choice = reply["choices"][0]
    ids, timestamps = choice["token_ids"], choice["token_timestamps_ns"]
    assert ids and len(ids) == reply["usage"]["completion_tokens"] <= tokens, reply
    assert len(timestamps) == len(ids) and timestamps == sorted(timestamps), reply
    assert choice["finish_reason"] in ("stop", "length", "tool_calls"), reply
    return dict(seed=seed, elapsed_ns=elapsed, response=reply)


def run(args):
    model = json.loads(get(args.url, "/v1/models"))["data"][0]["id"]
    print(json.dumps(dict(model=model, tokens=args.tokens, concurrency=args.concurrency)), flush=True)
    for count in args.concurrency:
        # Publish the identical prompt before simultaneously restoring every request.
        request(args.url, model, 700, 1)
        for repeat in range(args.repeats):
            before = metrics(args.url)
            barrier = threading.Barrier(count)
            start = time.monotonic_ns()
            with ThreadPoolExecutor(count) as pool:
                futures = [pool.submit(request, args.url, model, 701 + i, args.tokens, barrier)
                           for i in range(count)]
                replies = [future.result() for future in futures]
            elapsed = time.monotonic_ns() - start
            after = metrics(args.url)
            delta = {key: value - before.get(key, 0) for key, value in after.items()
                     if value != before.get(key, 0)}
            assert not delta.get("flare_requests_failed_total", 0), delta
            assert all(reply["response"].get("reused_prompt_tokens", 0) > 0 for reply in replies), replies
            print(json.dumps(dict(concurrency=count, repeat=repeat, elapsed_ns=elapsed,
                                  metrics=delta, replies=replies)), flush=True)
    if args.cancel:
        before = metrics(args.url)
        observed = 0
        with post(args.url, body(model, 800, 4096, stream=True)) as response:
            for line in response:
                if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                    chunk = json.loads(line[6:])
                    observed += sum(len(choice.get("token_ids", [])) for choice in chunk.get("choices", []))
                    if observed >= 16:
                        break
            else:
                raise AssertionError("stream ended before sixteen generated tokens")
        deadline = time.monotonic() + 30
        while metrics(args.url).get("flare_requests_cancelled_total", 0) == before.get("flare_requests_cancelled_total", 0):
            assert time.monotonic() < deadline, "disconnected stream did not cancel"
            time.sleep(.05)
        count = max(args.concurrency)
        barrier = threading.Barrier(count)
        with ThreadPoolExecutor(count) as pool:
            futures = [pool.submit(request, args.url, model, 900 + i, 16, barrier) for i in range(count)]
            replies = [future.result() for future in futures]
        after = metrics(args.url)
        assert after.get("flare_requests_failed_total", 0) == before.get("flare_requests_failed_total", 0), after
        print(json.dumps(dict(cancelled=True, observed_tokens=observed,
            metrics={key: value - before.get(key, 0) for key, value in after.items()
                     if value != before.get(key, 0)}, replacements=replies)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--cancel", action="store_true")
    run(parser.parse_args())
