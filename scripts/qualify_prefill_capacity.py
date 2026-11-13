# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify configured chunking and prefix restoration through native HTTP."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qualify.server import Run, comparable_message, digest, metric_delta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--memory-report", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    if args.artifacts.exists() and any(args.artifacts.iterdir()):
        parser.error("output destinations must not contain an earlier qualification")
    memory = json.loads(args.memory_report.read_text())
    chunk = memory["prefill_chunk_size"]
    capacities = memory["prefill_capacities"]
    if capacities[-1] != chunk or memory["graphs"] != 25 + 2 * len(capacities):
        raise ValueError("memory report has an inconsistent prefill graph catalog")
    fixture = json.loads((args.fixtures / "manifest.json").read_text())
    bodies = {}
    for name, case in fixture["cases"].items():
        path = args.fixtures / f"{name}.request.json"
        tokens = args.fixtures / f"{name}.tokens.u32le"
        if digest(path) != case["request_sha256"] or digest(tokens) != case["token_ids_sha256"]:
            raise ValueError(f"saved fixture {name} differs from its manifest")
        if tokens.stat().st_size != case["prompt_tokens"] * 4:
            raise ValueError(f"saved fixture {name} has an invalid token extent")
        body = json.loads(path.read_text())
        if body["temperature"] != 0 or body["max_tokens"] != 1 or body["model"] != fixture["checkpoint"]:
            raise ValueError("fixtures require greedy single-token generation for the declared checkpoint")
        bodies[name] = body
    args.artifacts.mkdir(parents=True, exist_ok=True)
    args.require_metrics, args.cases = True, None
    args.timeout, args.max_response_bytes = 120, 8 << 20
    run = Run(args)

    def graph(delta, kind):
        return delta[f'flare_graph_launches_total{{kind="{kind}"}}']

    def check_window(name, before, after, *, length, cached):
        delta = metric_delta(before, after)
        boundary = length // 64 * 64
        tail = length - boundary
        expected_prefill = int(tail > 0) if cached else (length + chunk - 1) // chunk
        run.check(name + ".graph_counts", graph(delta, "prefill") == expected_prefill
            and graph(delta, "decode") == 0 and graph(delta, "restore") == int(cached)
            and graph(delta, "init") == int(not cached)
            and graph(delta, "resume") == int(cached and tail == 0)
            and graph(delta, "vision") == 0, delta=delta, expected_prefill=expected_prefill)
        run.check(name + ".prefix_identity", delta["flare_cache_hits_total"] == int(cached)
            and delta["flare_cache_reused_tokens_total"] == (boundary if cached else 0), delta=delta)
        run.check(name + ".completed", delta["flare_requests_admitted_total"] == 1
            and delta["flare_requests_completed_total"] == 1
            and delta["flare_requests_failed_total"] == delta["flare_requests_cancelled_total"] == 0,
            delta=delta)
        run.reclaimed(name, before, after)

    def case(name, body):
        length = fixture["cases"][name]["prompt_tokens"]
        before = run.metrics(name + "-before")
        cold = run.request(name + "-cold", body)
        cold_choice = run.complete(cold, length)
        between = run.metrics(name + "-between")
        check_window(name + ".cold", before, between, length=length, cached=False)
        restored = run.request(name + "-restored", body)
        restored_choice = run.complete(restored, length)
        after = run.metrics(name + "-after")
        check_window(name + ".restored", between, after, length=length, cached=True)
        run.check(name + ".output_equivalence",
            comparable_message(cold_choice["message"]) == comparable_message(restored_choice["message"])
            and cold_choice["finish_reason"] == restored_choice["finish_reason"],
            cold=cold_choice, restored=restored_choice)
        run.check(name + ".single_token", all(record["body"]["usage"]["completion_tokens"] == 1
            for record in (cold, restored)))

    for name, body in bodies.items():
        run.case(name, lambda name=name, body=body: case(name, body))
    health = run.request("final-health", path="/health")
    run.check("final_health", health.get("status") == 200 and "client_error" not in health)
    run.results["finished_at"] = datetime.now(timezone.utc).isoformat()
    run.results["summary"] = {"checks": len(run.results["checks"]),
        "failed_checks": sum(not item["passed"] for item in run.results["checks"]),
        "case_errors": len(run.results["case_errors"]), "requests": len(run.results["requests"])}
    run.results["passed"] = not run.results["summary"]["failed_checks"] and not run.results["case_errors"]
    print(json.dumps(run.results["summary"]), flush=True)
    return 0 if run.results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
