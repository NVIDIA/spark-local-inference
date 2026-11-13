# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare complete conditional schedules under explicit time and waiting prices."""

import argparse
import csv
from dataclasses import asdict
from itertools import product
from pathlib import Path

from .model import Graph
from .oracle import Job, Objective, Oracle
from .continuation import SerialReference


def cases():
    prefills = (Graph("p512", 10, prefill_capacity=512), Graph("p4096", 64, prefill_capacity=4096))
    decodes = tuple(Graph(f"b{batch}w{width}", cost + 2 * (batch - 1), decode_capacity=batch, width=width)
                    for batch in (1, 2) for width, cost in ((1, 15), (3, 17.3), (8, 30)))
    prefill = Job("p", 8192, ((1, 1.0),))
    low = Job("low", 0, ((3, 1.0),), acceptance=(0.0,) * 7)
    high = Job("high", 0, ((3, 1.0),))
    return {
        "tail": ((Job("p", 129, ((1, 1.0),)),),
                 (Graph("p64", 2, prefill_capacity=64), Graph("p128", 3, prefill_capacity=128),
                  Graph("p256", 4, prefill_capacity=256))),
        "two_prefills": ((prefill, Job("other", 8192, ((1, 1.0),))), prefills),
        "prefill_decode_low": ((prefill, low), (*prefills, *decodes)),
        "prefill_decode_high": ((prefill, high), (*prefills, *decodes)),
        "two_decodes_low": ((low, Job("other", 0, ((3, 1.0),), acceptance=(0.0,) * 7)), decodes),
        "two_decodes_high": ((high, Job("other", 0, ((3, 1.0),))), decodes),
        "two_decodes_heterogeneous": ((low, high), decodes),
        "hidden_length": ((Job("d", 0, ((1, 0.5), (4, 0.5))),),
                          (Graph("w1", 1, decode_capacity=1), Graph("w4", 2.5, decode_capacity=1, width=4))),
        "stalled_decode": ((Job("fresh", 0, ((1, 1.0),)), Job("stalled", 0, ((1, 1.0),), age_ms=100)),
                           (Graph("b1", 10, decode_capacity=1),)),
        "mixed_tail": ((Job("p", 10, ((1, 1.0),)), Job("a", 0, ((2, 1.0),)), Job("b", 0, ((2, 1.0),))),
                       (Graph("p2", 3, prefill_capacity=2), Graph("p8", 8, prefill_capacity=8),
                        Graph("decode", 3, decode_capacity=2, width=2),
                        Graph("mixed2", 3.5, prefill_capacity=2, decode_capacity=2, width=2),
                        Graph("mixed8", 12, prefill_capacity=8, decode_capacity=2, width=2))),
        "penalty_dilution": ((Job("p", 64, ((1, 1.0),)), Job("d", 0, ((1, 1.0),), age_ms=1000)),
                             (Graph("fast", 1, prefill_capacity=64), Graph("slow", 100, prefill_capacity=64),
                              Graph("decode", 1, decode_capacity=1))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    workloads = cases()
    parser.add_argument("--case", choices=["all", *workloads], default="all")
    parser.add_argument("--decode-values", required=True)
    parser.add_argument("--time-values", required=True, help="comma-separated useful-work units per GPU ms")
    parser.add_argument("--age-prices", required=True, help="comma-separated useful-work units per ms squared")
    parser.add_argument("--prefill-wait-price", type=float, default=0)
    parser.add_argument("--unfinished-wait-price", type=float, default=0)
    parser.add_argument("--include-ratio", action="store_true", help="also evaluate episode normalization")
    parser.add_argument("--max-states", type=int, default=100_000)
    parser.add_argument("--output", type=Path, help="CSV of expected metrics and objective values")
    args = parser.parse_args()
    selected = workloads if args.case == "all" else {args.case: workloads[args.case]}
    dimensions = [[float(item) for item in arg.split(",")]
                  for arg in (args.decode_values, args.time_values, args.age_prices)]
    rows = []
    print("Synthetic timings; exact finite-workload expectations, not GPU measurements.")
    for name, (jobs, graphs) in selected.items():
        oracle = Oracle(jobs, graphs, max_states=args.max_states)
        for decode_value, time_value, age_price in product(*dimensions):
            objective = Objective(decode_value, args.prefill_wait_price, args.unfinished_wait_price, age_price)
            best = oracle.solve_priced(objective, time_value=time_value)
            rollout = oracle.rollout(objective, time_value=time_value)
            serial = SerialReference(oracle, objective, time_value=time_value)
            results = {
                "oracle": best,
                "value_rate": oracle.evaluate(oracle.immediate(objective), objective),
                "myopic_wait": oracle.evaluate(oracle.immediate(objective, waiting=True), objective),
                "rollout": oracle.evaluate(rollout, objective),
                "rollout_twice": oracle.evaluate(oracle.rollout(objective, time_value=time_value, baseline=rollout), objective),
                "serial_value": oracle.evaluate(serial.choose, objective),
            }
            if args.include_ratio:
                results["episode_ratio"] = oracle.solve(objective)
            best_surplus = objective.net_work(best.metrics) - time_value * best.metrics.elapsed_ms
            for method, result in results.items():
                surplus = objective.net_work(result.metrics) - time_value * result.metrics.elapsed_ms
                action = result.first
                row = {"case": name, "method": method, "decode_value": decode_value,
                       "time_value": time_value, "age_price": age_price,
                       "prefill_wait_price": args.prefill_wait_price,
                       "unfinished_wait_price": args.unfinished_wait_price,
                       "first_graph": action.graph.name,
                       "first_prefill": jobs[action.prefill].name if action.prefill is not None else "",
                       "first_decodes": ",".join(jobs[i].name for i in action.decodes),
                       "expected_surplus": surplus, "surplus_gap": best_surplus - surplus,
                       "useful_work_per_ms": objective.useful_work(result.metrics) / result.metrics.elapsed_ms,
                       "net_work_per_ms": result.rate, **asdict(result.metrics), "states": result.states}
                rows.append(row)
                print(f"{name:24} {method:13} beta={decode_value:g} time={time_value:g} age={age_price:g} "
                      f"first={action.graph.name:9} elapsed={result.metrics.elapsed_ms:8.2f}ms "
                      f"surplus={surplus:10.3f} gap={row['surplus_gap']:8.3f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
