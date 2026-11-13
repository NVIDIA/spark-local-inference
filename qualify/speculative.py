# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify a startup-selected speculative policy through native HTTP serving."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import signal
import time
from urllib.request import Request, urlopen



def http(base, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = Request(base + path, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=180) as response:
        value = response.read().decode()
    return value if path == "/metrics" else json.loads(value)


def metrics(base):
    raw = http(base, "/metrics")
    values = {}
    for line in raw.splitlines():
        if line and not line.startswith("#"):
            key, value = line.rsplit(" ", 1)
            values[key] = float(value)
    return raw, values


def idle(base):
    deadline = time.monotonic() + 30
    while True:
        raw, values = metrics(base)
        if (values.get("flare_active_requests", -1) == 0
                and values.get("flare_pending_requests", -1) == 0
                and values.get("flare_free_request_slots", -1) == 4):
            return raw, values
        if time.monotonic() >= deadline:
            raise TimeoutError("native server did not reclaim all requests within 30 seconds")
        time.sleep(.01)


def literal_request(model, text, *, seed=42, budget=256):
    return {"model": model, "messages": [
        {"role": "system", "content": "Copy the supplied text exactly."},
        {"role": "user", "content": text}],
        "chat_template_kwargs": {"enable_thinking": False},
        "grammar": "start: " + json.dumps(text),
        "temperature": 1.0, "top_k": 20, "top_p": .95,
        "seed": seed, "max_tokens": budget,
        "return_token_ids": True, "return_token_timestamps": True}


def run(args, model, server):
    base = "http://" + server.address
    identity = http(base, "/v1/models")["data"][0]["id"]
    text = " ".join(["alpha beta gamma delta epsilon zeta eta theta"] * 8)
    checks = 0

    def check(name, passed, **details):
        nonlocal checks
        if not passed:
            raise AssertionError(f"{name}: {details}")
        checks += 1

    def complete(body):
        return http(base, "/v1/chat/completions", body)

    idle(base)
    for concurrency in (1, 2, 4):
        requests = [literal_request(identity, text, seed=42 + lane) for lane in range(concurrency)]
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            results = list(executor.map(complete, requests))
        for lane, result in enumerate(results):
            choice = result["choices"][0]
            actual = choice["message"]["content"]
            name = f"concurrency-{concurrency}/lane-{lane}"
            check(name + "/literal", actual == text, actual=actual, expected=text)
            check(name + "/stop", choice["finish_reason"] == "stop", actual=choice["finish_reason"])
    for budget in (1, 2, 3, 7, 8, 9, 15):
        response = complete(literal_request(identity, text, budget=budget))
        choice = response["choices"][0]
        check(f"budget-{budget}/count", response["usage"]["completion_tokens"] == budget,
              actual=response["usage"]["completion_tokens"], expected=budget)
        check(f"budget-{budget}/length", choice["finish_reason"] == "length", actual=choice["finish_reason"])
        check(f"budget-{budget}/prefix", text.startswith(choice["message"]["content"]))
    idle(base)
    print(f"{checks} speculative HTTP checks passed", flush=True)


def main():
    from ignition.recipes.qwen38.bootstrap import PreparationConfig, prepare_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--bind", default="127.0.0.1:8000")
    parser.add_argument("--fixed-width", type=int, choices=range(1, 9))
    parser.add_argument("--history-drafts", action="store_true")
    parser.add_argument("--keep-serving", action="store_true")
    args = parser.parse_args()
    config = PreparationConfig(mtp=True, speculative_fixed_width=args.fixed_width)
    with prepare_model(args.snapshot, config, progress=lambda text: print(text, flush=True)) as model:
        server = model.start_server(bind=args.bind, options={"history_drafts": args.history_drafts})
        print(f"Listening on {server.address}", flush=True)
        run(args, model, server)
        if args.keep_serving:
            def stop(signum, frame):
                raise KeyboardInterrupt
            previous = signal.signal(signal.SIGTERM, stop)
            try:
                while server.wait(1000):
                    pass
                raise RuntimeError("native server stopped unexpectedly")
            except KeyboardInterrupt:
                pass
            finally:
                signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
