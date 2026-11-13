# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure final-response delay while another request is still generating.

Run against an idle server using throughput scheduling and at least two request
slots: python3 scripts/check_response_completion.py --port 8000 --output result.json.
The script warms the completion path, starts a long response, and submits a short
response after generation begins. It saves the original SSE events and requires
the long response to emit tokens after the short stream closes, with less than
one second from the short response's final visible token to finish_reason.
"""

import argparse
import json
import threading
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    base = f"http://127.0.0.1:{args.port}"
    with urllib.request.urlopen(base + "/v1/models", timeout=10) as response:
        model = json.load(response)["data"][0]["id"]
    start = time.monotonic()
    first_long = threading.Event()
    results = {}

    def generate(name, prompt, limit):
        record = {"max_tokens": limit, "events": [], "started": time.monotonic() - start}
        results[name] = record
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": limit, "temperature": 1, "top_p": 1, "seed": 42,
                   "return_token_ids": True, "stream": True,
                   "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
        request = urllib.request.Request(base + "/v1/chat/completions",
                                         json.dumps(payload).encode(),
                                         {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                for line in response:
                    if not line.startswith(b"data: "):
                        continue
                    timestamp = time.monotonic() - start
                    data = line[6:].strip()
                    if data == b"[DONE]":
                        record["done"] = timestamp
                        break
                    event = json.loads(data)
                    record["events"].append({"time": timestamp, "data": event})
                    if event.get("usage"):
                        record["usage"] = event["usage"]
                    for choice in event.get("choices", []):
                        if choice.get("token_ids"):
                            record.setdefault("tokens", []).extend(
                                {"time": timestamp, "id": token} for token in choice["token_ids"]
                            )
                        delta = choice.get("delta", {})
                        if any(delta.get(key) for key in ("content", "reasoning", "reasoning_content")):
                            record.setdefault("first_visible", timestamp)
                            record["last_visible"] = timestamp
                            if name == "long":
                                first_long.set()
                        if choice.get("finish_reason") is not None:
                            record["finish"] = timestamp
                            record["finish_reason"] = choice["finish_reason"]
            record["closed"] = time.monotonic() - start
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
        print(name, json.dumps({key: value for key, value in record.items() if key not in ("events", "tokens")}), flush=True)

    # Prime terminal-snapshot costs before requests compete for scheduling.
    generate("warmup", "Write the word WARMUP.", 16)
    long = threading.Thread(target=generate, args=("long",
        "Make me an amazing mind blowing tetris clone with cool music and awesome visuals.", 8192))
    long.start()
    if not first_long.wait(60):
        long.join()
        raise RuntimeError("long request did not start streaming")
    time.sleep(0.25)
    generate("short", "Write only the word READY.", 16)
    long.join()
    short = results["short"]
    long_result = results["long"]
    hold = short.get("finish", float("inf")) - short.get("last_visible", 0)
    long_times = [token["time"] for token in long_result.get("tokens", [])]
    continued = any(at > short.get("done", float("inf")) for at in long_times)
    passed = (not any("error" in value for value in results.values())
              and "done" in short and "done" in long_result
              and short["done"] < long_result["done"]
              and continued and hold < 1.0)
    output = {"model": model, "port": args.port, "passed": passed,
              "short_finish_after_last_visible_seconds": hold,
              "long_emitted_after_short_finished": continued,
              "long_max_token_event_gap_seconds": max(
                  (b - a for a, b in zip(long_times, long_times[1:])), default=0),
              "long_output_tokens_per_second": long_result.get("usage", {}).get("completion_tokens", 0)
                  / max(1e-9, long_result["closed"] - long_result["started"]),
              "requests": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "requests"}), flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
