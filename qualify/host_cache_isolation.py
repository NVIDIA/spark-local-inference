# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host prefix cache isolation measurements: serving latency with and without
the host store under the copy-out and restore load the store creates.

Each workload runs against a server without a host store (the baseline) and
against a server with the configured host store, same core, flags, and GPU.
Per-token latency comes from the server's native completion timestamps
(`return_token_timestamps`), prefill throughput from prompt tokens over the
time from ingress to the first token, and restore latency from the first-token
time of a request whose prefix the host store restores compared with the same
prefix computed cold. Raw observations are written beside the summary.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import statistics
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from qualify.host_cache import Server, body, long_prompt, run_for, wait_quiescent
from qualify.server import metric_delta, resource_totals

ROOT = Path(__file__).resolve().parents[1]


def stream(base_url: str, request: dict, timeout: float) -> dict:
    """One streaming completion with native token timestamps; returns the observation."""
    url = urlsplit(base_url)
    request = {**request, "stream": True, "return_token_ids": True, "return_token_timestamps": True,
               "stream_options": {"include_usage": True}}
    payload = json.dumps(request).encode()
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=timeout)
    started = time.perf_counter_ns()
    tokens, core, metadata, usage, reused, errors = [], [], {}, {}, None, []
    deliveries, content = [], []
    try:
        conn.request("POST", "/v1/chat/completions", body=payload,
                     headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
        response = conn.getresponse()
        if response.status != 200:
            errors.append(f"HTTP {response.status}: {response.read(1 << 16)[:200]!r}")
        data = []
        while response.status == 200 and (line := response.readline(1 << 20)):
            stripped = line.rstrip(b"\r\n")
            if stripped:
                if stripped.startswith(b"data:"):
                    data.append(stripped[5:].removeprefix(b" "))
                continue
            if not data:
                continue
            value = b"\n".join(data).decode()
            data.clear()
            if value == "[DONE]":
                break
            event = json.loads(value)
            if "error" in event:
                errors.append(str(event["error"]))
            if event.get("performance") is not None:
                metadata = event["performance"]
            if event.get("usage") is not None:
                usage = event["usage"]
            if event.get("reused_prompt_tokens") is not None:
                reused = event["reused_prompt_tokens"]
            for choice in event.get("choices", []):
                if (choice.get("delta") or {}).get("content"):
                    content.append(choice["delta"]["content"])
                ids = choice.get("token_ids") or []
                stamps = choice.get("token_timestamps_ns") or []
                if ids:
                    tokens.extend(ids)
                    deliveries.append((time.perf_counter_ns() - started, len(ids)))
                    core.extend(stamps)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    finally:
        conn.close()
    return {"tokens": tokens, "content": "".join(content), "core_ns": core, "deliveries": deliveries,
            "performance": metadata, "usage": usage, "reused": reused, "errors": errors,
            "elapsed_ns": time.perf_counter_ns() - started}


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))]


ROUND_GAP_NS = 1_000_000


def round_intervals(stamps: list[int]) -> list[int]:
    """Intervals between speculative rounds: tokens accepted in one round carry
    timestamps a few hundred nanoseconds apart, so gaps under one millisecond
    belong to the same round and per-token intervals would be meaningless."""
    rounds = [stamp for index, stamp in enumerate(stamps) if index == 0 or stamp - stamps[index - 1] >= ROUND_GAP_NS]
    return [b - a for a, b in zip(rounds, rounds[1:])]


def summarize_itl(observations: list[dict]) -> dict:
    core = [interval for o in observations for interval in round_intervals(o["core_ns"])]
    ttft = [o["core_ns"][0] - o["performance"]["ingress_ns"] for o in observations
            if o["core_ns"] and isinstance(o["performance"].get("ingress_ns"), int)]
    return {"requests": len(observations), "tokens": sum(len(o["tokens"]) for o in observations),
            "errors": sum(len(o["errors"]) for o in observations),
            "round_p50_us": (percentile(core, .5) or 0) / 1e3, "round_p99_us": (percentile(core, .99) or 0) / 1e3,
            "ttft_p50_ms": (percentile(ttft, .5) or 0) / 1e6, "ttft_p99_ms": (percentile(ttft, .99) or 0) / 1e6}


def decode_workload(server: Server, run, model_id: str, args, label: str) -> dict:
    """Continuous short exchanges at the configured concurrency for the configured duration."""
    before = wait_quiescent(run, f"{label}-decode-before", args.timeout)
    observations, lock = [], threading.Lock()
    deadline = time.monotonic() + args.seconds
    def worker(index: int):
        turn = 0
        while time.monotonic() < deadline:
            nonce = f"{args.nonce}-{index}-{turn}"
            messages = [{"role": "user", "content": long_prompt(nonce, args.prompt_sentences)}]
            first = stream(server.url, body(model_id, messages, args.answer_tokens), args.timeout)
            messages += [{"role": "assistant", "content": first["content"]},
                         {"role": "user", "content": "Continue with one more sentence about the series."}]
            second = stream(server.url, body(model_id, messages, args.answer_tokens), args.timeout)
            with lock:
                observations.extend([first, second])
            turn += 1
    threads = [threading.Thread(target=worker, args=(index,)) for index in range(args.concurrency)]
    started = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.monotonic() - started
    after = wait_quiescent(run, f"{label}-decode-after", args.timeout)
    delta = metric_delta(before, after)
    summary = summarize_itl(observations)
    summary.update(seconds=elapsed, tokens_per_second=summary["tokens"] / elapsed,
                   publications=delta.get("flare_snapshot_publications_total", 0),
                   host_stored=delta.get("flare_host_cache_stored_total", 0),
                   host_store_skipped=delta.get("flare_host_cache_store_skipped_total", 0),
                   host_bytes_out=delta.get("flare_host_cache_bytes_out_total", 0),
                   speculative_gpu_ns=sum(v for k, v in delta.items() if k.startswith("flare_speculative_gpu_nanoseconds_total")),
                   speculative_rounds=sum(v for k, v in delta.items() if k.startswith("flare_speculative_rounds_total")))
    (args.artifacts / f"{label}-decode.json").write_text(json.dumps({"summary": summary, "observations": observations}, indent=1))
    return summary


def prefill_workload(server: Server, run, model_id: str, args, label: str) -> dict:
    """Long prompts admitted back to back, one token each; throughput from ingress to the first token."""
    before = wait_quiescent(run, f"{label}-prefill-before", args.timeout)
    rates, observations = [], []
    for index in range(args.prefill_count):
        nonce = f"{args.nonce}-prefill-{index}"
        observation = stream(server.url, body(model_id, [{"role": "user", "content": long_prompt(nonce, args.prefill_sentences)}], 1), args.timeout)
        observations.append(observation)
        prompt = observation["usage"].get("prompt_tokens", 0)
        if observation["core_ns"] and isinstance(observation["performance"].get("ingress_ns"), int):
            seconds = (observation["core_ns"][0] - observation["performance"]["ingress_ns"]) / 1e9
            rates.append(prompt / seconds)
    after = wait_quiescent(run, f"{label}-prefill-after", args.timeout)
    delta = metric_delta(before, after)
    summary = {"requests": len(observations), "prompt_tokens": [o["usage"].get("prompt_tokens") for o in observations],
               "prefill_tokens_per_second_median": statistics.median(rates) if rates else None,
               "prefill_tokens_per_second_min": min(rates) if rates else None,
               "errors": sum(len(o["errors"]) for o in observations),
               "host_stored": delta.get("flare_host_cache_stored_total", 0),
               "host_bytes_out": delta.get("flare_host_cache_bytes_out_total", 0)}
    (args.artifacts / f"{label}-prefill.json").write_text(json.dumps({"summary": summary, "observations": observations}, indent=1))
    return summary


def inbound_latency(server: Server, run, model_id: str, args, label: str) -> dict:
    """First-token time of a long prefix computed cold, hit on the GPU, and restored from the host store."""
    results = {}
    nonce = f"{args.nonce}-inbound"
    messages = [{"role": "user", "content": long_prompt(nonce, args.prefill_sentences)}]
    cold = stream(server.url, body(model_id, messages, 1), args.timeout)
    results["cold"] = cold
    wait_quiescent(run, f"{label}-inbound-cold", args.timeout)
    warm = stream(server.url, body(model_id, messages, 1), args.timeout)
    results["gpu"] = warm
    before = wait_quiescent(run, f"{label}-inbound-evict-before", args.timeout)
    capacity = resource_totals(before)["snapshots"]
    for index in range(capacity + 1):
        stream(server.url, body(model_id, [{"role": "user", "content": f"Filler {nonce} {index}. Reply with one word."}], 1), args.timeout)
    wait_quiescent(run, f"{label}-inbound-evict-after", args.timeout)
    restored = stream(server.url, body(model_id, messages, 1), args.timeout)
    after = wait_quiescent(run, f"{label}-inbound-after", args.timeout)
    results["host"] = restored
    def ttft_ms(observation):
        if observation["core_ns"] and isinstance(observation["performance"].get("ingress_ns"), int):
            return (observation["core_ns"][0] - observation["performance"]["ingress_ns"]) / 1e6
        return None
    summary = {"prompt_tokens": cold["usage"].get("prompt_tokens"),
               "cold_ttft_ms": ttft_ms(cold), "gpu_hit_ttft_ms": ttft_ms(warm), "host_restore_ttft_ms": ttft_ms(restored),
               "reused": {name: results[name]["reused"] for name in results},
               "host_hits": metric_delta(before, after).get("flare_host_cache_hits_total", 0),
               "host_bytes_in": metric_delta(before, after).get("flare_host_cache_bytes_in_total", 0),
               "host_inbound_bytes_per_second": after.get("flare_host_cache_inbound_bytes_per_second", 0)}
    (args.artifacts / f"{label}-inbound.json").write_text(json.dumps({"summary": summary, "observations": results}, indent=1))
    return summary


def measure(args, label: str, extra: list[str]) -> dict:
    server = Server(args.flare, args.core, args.gpus, args.port, args.artifacts / f"{label}.log", extra)
    try:
        server.wait_ready(args.startup_timeout)
        run = run_for(server, args.artifacts / label, args.timeout)
        model_id = server.model_id()
        summary = {"decode": decode_workload(server, run, model_id, args, label),
                   "prefill": prefill_workload(server, run, model_id, args, label)}
        if not label.startswith("baseline"):
            summary["inbound"] = inbound_latency(server, run, model_id, args, label)
        return summary
    finally:
        server.stop()


def compare(baseline: dict, offload: dict) -> list[tuple[str, float, float, float, bool]]:
    rows = []
    for name, key, higher_is_better in (("decode round p50 (us)", ("decode", "round_p50_us"), False),
                                        ("decode round p99 (us)", ("decode", "round_p99_us"), False),
                                        ("first token p50 (ms)", ("decode", "ttft_p50_ms"), False),
                                        ("first token p99 (ms)", ("decode", "ttft_p99_ms"), False),
                                        ("decode tokens/s", ("decode", "tokens_per_second"), True),
                                        ("prefill tokens/s (median)", ("prefill", "prefill_tokens_per_second_median"), True)):
        base, value = baseline[key[0]][key[1]], offload[key[0]][key[1]]
        if not base or value is None:
            continue
        change = (value - base) / base
        within = change >= -0.02 if higher_is_better else change <= 0.02
        rows.append((name, base, value, change, within))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flare", type=Path, default=ROOT / "flare-cli/target/release/flare")
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--port", type=int, default=8061)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--memory", default="24G")
    parser.add_argument("--disk", type=Path, help="record directory for the host store; omitted: memory tier only")
    parser.add_argument("--disk-limit", default="100G")
    parser.add_argument("--seconds", type=float, default=120, help="duration of the decode workload per server")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--prompt-sentences", type=int, default=16, help="about 16 tokens each")
    parser.add_argument("--answer-tokens", type=int, default=64)
    parser.add_argument("--prefill-sentences", type=int, default=2000, help="about 32K tokens")
    parser.add_argument("--prefill-count", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--flare-option", action="append", default=[], metavar="FLAG=VALUE")
    parser.add_argument("--repeat", type=int, default=1,
                        help="alternate baseline and offload servers this many times; drift between repeats bounds the noise")
    args = parser.parse_args()
    if args.artifacts.exists() and any(args.artifacts.iterdir()):
        parser.error("the artifact directory must not contain a previous run")
    args.artifacts.mkdir(parents=True, exist_ok=True)
    args.nonce = hashlib.sha256(f"isolation-{time.time_ns()}".encode()).hexdigest()[:8]
    common = [part for option in args.flare_option for part in option.split("=", 1)]
    offload = common + ["--prefix-cache-memory", args.memory]
    if args.disk is not None:
        offload += ["--prefix-cache-disk", str(args.disk), "--prefix-cache-disk-limit", args.disk_limit]
    runs = []
    for repeat in range(args.repeat):
        suffix = "" if args.repeat == 1 else f"-{repeat}"
        baseline = measure(args, f"baseline{suffix}", common + ["--prefix-cache-memory", "0"])
        measured = measure(args, f"offload{suffix}", offload)
        runs.append({"baseline": baseline, "offload": measured, "comparison": [
            {"metric": n, "baseline": b, "offload": v, "change": c, "within_2_percent": w} for n, b, v, c, w in compare(baseline, measured)]})
    report = {"configuration": {"core": str(args.core), "gpus": args.gpus, "memory": args.memory,
                                "disk": str(args.disk) if args.disk else None, "disk_limit": args.disk_limit,
                                "seconds": args.seconds, "concurrency": args.concurrency, "flare_options": common},
              "runs": runs}
    (args.artifacts / "report.json").write_text(json.dumps(report, indent=1))
    print(f"{'metric':30} " + " ".join(f"{'base':>10} {'offload':>10} {'change':>8}" for _ in runs))
    names = [row["metric"] for row in runs[0]["comparison"]]
    for name in names:
        cells = []
        for run in runs:
            row = next(r for r in run["comparison"] if r["metric"] == name)
            cells.append(f"{row['baseline']:10.1f} {row['offload']:10.1f} {row['change']:+8.2%}")
        print(f"{name:30} " + " ".join(cells))
    rows = [(r["metric"], r["baseline"], r["offload"], r["change"], r["within_2_percent"]) for r in runs[-1]["comparison"]]
    measured = runs[-1]["offload"]
    inbound = measured.get("inbound", {})
    print(f"inbound: prompt {inbound.get('prompt_tokens')} tokens, cold TTFT {inbound.get('cold_ttft_ms')} ms, "
          f"GPU hit TTFT {inbound.get('gpu_hit_ttft_ms')} ms, host restore TTFT {inbound.get('host_restore_ttft_ms')} ms, "
          f"host hits {inbound.get('host_hits')}, inbound {inbound.get('host_inbound_bytes_per_second', 0) / 1e9:.2f} GB/s")
    print(f"copy-out during decode: {measured['decode']['host_stored']} records, {measured['decode']['host_bytes_out'] / 1e9:.2f} GB, "
          f"{measured['decode']['host_store_skipped']} skipped")
    return 0 if all(within for *_, within in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
