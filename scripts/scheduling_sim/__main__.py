# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare scheduling objectives over deterministic workload definitions."""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

from .model import Graph, Value
from .simulator import run
from .utility import UtilityConfig, UtilityScheduler
from .workloads import catalog, workloads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cases = workloads()
    parser.add_argument("--workload", choices=["all", *cases], default="all")
    parser.add_argument("--decode-values", required=True, help="comma-separated values per emitted token")
    parser.add_argument("--policy", choices=("value", "utility"), default="value")
    parser.add_argument("--token-fairness", type=float)
    parser.add_argument("--delivery-fairness", type=float)
    parser.add_argument("--adaptation-ms", type=float)
    parser.add_argument("--age-price", type=float)
    parser.add_argument("--rate-history", choices=("exponential", "cumulative", "service"))
    parser.add_argument("--mixed", action="store_true")
    parser.add_argument("--catalog", type=Path, help="JSON array of Graph fields, durations in ms")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon-ms", type=float, default=10_000)
    parser.add_argument("--output", type=Path, help="summary.csv, requests.csv, and trace.jsonl")
    args = parser.parse_args()
    preferences = (args.token_fairness, args.delivery_fairness, args.adaptation_ms, args.age_price)
    if args.policy == "utility" and any(v is None for v in preferences):
        parser.error("utility requires --token-fairness, --delivery-fairness, --adaptation-ms, and --age-price")
    if args.policy == "value" and (any(v is not None for v in preferences) or args.rate_history is not None):
        parser.error("fairness and age preferences require --policy utility")
    config = UtilityConfig(*preferences, rate_history=args.rate_history or "exponential") if args.policy == "utility" else None
    graphs = tuple(Graph(**row) for row in json.loads(args.catalog.read_text())) if args.catalog else catalog(mixed=args.mixed)
    values = [Value(float(v)) for v in args.decode_values.split(",")]
    selected = cases if args.workload == "all" else {args.workload: cases[args.workload]}
    summaries, rows, traces = [], [], []
    print(f"{args.policy} policy; synthetic timings unless --catalog supplies measurements.")
    print("Preferences are experimental inputs; there is no hard output-gap guarantee.")
    for name, requests in selected.items():
        for value in values:
            scheduler = UtilityScheduler(value, config) if config is not None else None
            result = run(requests, graphs, value, seed=args.seed, horizon_ms=args.horizon_ms,
                         scheduler=scheduler)
            case = {"workload": name, "decode_value": value.decode_value, "seed": args.seed,
                    "policy": args.policy} | (asdict(config) if config is not None else {})
            summaries.append(case | result.summary)
            rows.extend(case | row for row in result.requests)
            traces.extend(case | event for event in result.trace)
            summary = result.summary
            print(f"{name:28} v={value.decode_value:6g} value/ms={summary['useful_work_per_ms']:8.2f} "
                  f"prefill={summary['prefill_tok_s']:8.0f}tok/s output={summary['output_tok_s']:8.1f}tok/s "
                  f"gap={summary['max_output_gap_ms']:8.1f}ms unfinished={summary['unfinished']}")
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        for filename, data in (("summary.csv", summaries), ("requests.csv", rows)):
            with (args.output / filename).open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
        with (args.output / "trace.jsonl").open("w") as handle:
            for event in traces:
                handle.write(json.dumps(event) + "\n")
        (args.output / "catalog.json").write_text(json.dumps([asdict(g) for g in graphs], indent=2) + "\n")


if __name__ == "__main__":
    main()
