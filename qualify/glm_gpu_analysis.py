# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize TP graph envelopes from GLM HTTP benchmark Nsight traces.

Flare graph names come from its prepared catalog. Peer launches are matched by
the ordered native dispatch sequence, with equal launch counts required.
Durations are wall time on the GPU; concurrent rank durations are never added
together.
"""

import argparse
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path


def distribution(values):
    values = sorted(values)
    return {"count": len(values), "mean_ms": statistics.mean(values) / 1e6,
            "median_ms": statistics.median(values) / 1e6,
            "p95_ms": values[int((len(values) - 1) * .95)] / 1e6}


def pid(global_id):
    return (global_id >> 24) & 0xffffff


def read_trace(path):
    db = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    table = "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"
    data = {table: [dict(row) for row in db.execute("select * from " + table)]}
    db.close()
    return data


def flare(data, directory):
    catalog = json.loads((directory / "graphs.json").read_text())
    names = {graph["id"]: graph["name"] for graph in catalog["graphs"]}
    ranks = defaultdict(list)
    for graph in data["CUPTI_ACTIVITY_KIND_GRAPH_TRACE"]:
        ranks[pid(graph["globalPid"])].append(graph)
    for graphs in ranks.values():
        graphs.sort(key=lambda graph: graph["start"])
    root = ranks[catalog["pid"]]
    if not root or any(len(graphs) != len(root) for graphs in ranks.values()):
        raise ValueError("rank graph launch counts differ")
    indices = [i for i, graph in enumerate(root) if names[graph["graphExecId"]] == "decode1"]
    if len(indices) != 765:
        raise ValueError(f"expected 765 batch-one decode graphs; found {len(indices)}")
    report = {"ranks": {process: distribution([graphs[i]["end"] - graphs[i]["start"] for i in indices])
                        for process, graphs in ranks.items()}, "root_pid": catalog["pid"]}
    envelopes = []
    for i in indices:
        graphs = [rank[i] for rank in ranks.values()]
        if max(g["start"] for g in graphs) >= min(g["end"] for g in graphs):
            raise ValueError("corresponding rank decode graphs do not overlap")
        envelopes.append(max(g["end"] for g in graphs) - min(g["start"] for g in graphs))
    report["all_rank_envelope"] = distribution(envelopes)
    report["repeat_mean_ms"] = [statistics.mean(envelopes[i:i + 255]) / 1e6 for i in range(0, 765, 255)]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("directory", type=Path, help="HTTP results and Flare graph catalog")
    args = parser.parse_args()
    results = json.loads((args.directory / "results.json").read_text())
    if [row["phase"] for row in results] != ["warmup", *["unprofiled"] * 3, *["profiled"] * 3]:
        raise ValueError("HTTP measurements are incomplete")
    data = read_trace(args.trace)
    report = flare(data, args.directory)
    report["http_tokens_per_second"] = [row["delivery"]["tokens_per_second"] for row in results
                                        if row["phase"] == "unprofiled"]
    (args.directory / "timings.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
