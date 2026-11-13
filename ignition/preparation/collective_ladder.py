# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Final compiled geometries and runtime profiles for PCIe reductions and column gathers."""

import json
from pathlib import Path

from ignition.errors import UsageError

MODES = {
    "joint_latency": ("full", "payload_rsag", "pull"),
    "joint_bandwidth": ("payload_rsag", "bulk_push", "bulk_pull", "pull"),
}
CLUSTER_MODES = {
    "joint_latency": ("full_phased", "payload_rsag_phased"),
    "joint_bandwidth": ("payload_rsag_phased",),
}
GATHER_MODES = {
    "gather_joint": ("barrier", "word_exchange"),
    "gather_selected_barrier": ("barrier",),
    "gather_selected_word_exchange": ("word_exchange",),
}


def _validate_gathers(layout, world_size):
    def require(condition, message):
        if not condition:
            raise UsageError(f"collective ladder all-gather: {message}")

    if layout is None:
        return
    require(world_size in (2, 4, 8, 10), "unsupported gather TP size")
    require(isinstance(layout, dict) and set(layout) == {"min_bytes", "buckets"},
            "expected min_bytes and buckets")
    lower = layout["min_bytes"]
    require(type(lower) is int and lower >= 16 and lower % 16 == 0, "invalid min_bytes")
    require(isinstance(layout["buckets"], list) and bool(layout["buckets"]), "buckets must be nonempty")
    previous = lower - 16
    for row in layout["buckets"]:
        require(isinstance(row, dict) and set(row) == {"max_bytes", "implementation", "threads", "blocks", "profiles"},
                "invalid bucket fields")
        upper = row["max_bytes"]
        require(type(upper) is int and upper > previous and upper % 16 == 0,
                "bucket boundaries must increase and be 16-byte aligned")
        previous = upper
        family = row["implementation"]
        require(isinstance(family, str) and family in GATHER_MODES, "unknown gather kernel")
        require(type(row["threads"]) is int and row["threads"] in (32, 64, 128, 256, 512), "invalid threads")
        require(type(row["blocks"]) is int and 1 <= row["blocks"] <= 64, "invalid physical grid")
        if family != "gather_selected_barrier":
            require(world_size in (2, 4) and upper <= 16384, "word exchange requires TP2/TP4 and at most 16 KiB")
        if family == "gather_selected_word_exchange":
            require(row["threads"] <= 256, "word exchange supports at most 256 threads")
        require(isinstance(row["profiles"], dict) and bool(row["profiles"]), "profiles must be nonempty")
        for mode, counts in row["profiles"].items():
            require(mode in GATHER_MODES[family], "mode does not belong to gather kernel")
            require(isinstance(counts, list) and bool(counts)
                    and all(type(v) is int and 1 <= v <= row["blocks"] for v in counts), "invalid active CTA counts")
            require(len(set(counts)) == len(counts), "duplicate active CTA counts")


def select_gather(data, size_bytes, *, world_size, capability):
    """Select an inclusive per-rank byte bucket independently of row geometry."""
    if world_size != data["world_size"] or tuple(capability) != tuple(data["compute_capability"]):
        raise UsageError("collective ladder does not match the gather's TP size or compute capability")
    ladder = data["all_gather"]
    if (ladder is None or type(size_bytes) is not int or size_bytes % 16
            or not ladder["min_bytes"] <= size_bytes <= ladder["buckets"][-1]["max_bytes"]):
        raise UsageError(f"collective ladder has no all-gather bucket for {size_bytes} bytes")
    return next(row for row in ladder["buckets"] if size_bytes <= row["max_bytes"])


def validate(data):
    """Validate the JSON contract without importing CUDA or preparation backends."""
    def require(condition, message):
        if not condition:
            raise UsageError(f"collective ladder: {message}")

    require(isinstance(data, dict), "expected an object")
    require(set(data) == {"version", "name", "compute_capability", "world_size", "dtype",
                          "inplace", "cache_policy", "min_bytes", "buckets", "all_gather"}, "invalid fields")
    require(type(data["version"]) is int and data["version"] == 2, "unsupported version")
    require(isinstance(data["name"], str) and bool(data["name"]), "name must be nonempty")
    require(isinstance(data["compute_capability"], list) and len(data["compute_capability"]) == 2
            and all(type(v) is int and v >= 0 for v in data["compute_capability"]), "invalid compute capability")
    require(type(data["world_size"]) is int and data["world_size"] in (2, 3, 4, 6, 8, 10), "unsupported TP size")
    require(data["dtype"] == "bfloat16", "only bfloat16 reductions are supported")
    require(type(data["inplace"]) is bool, "inplace must be boolean")
    require(data["cache_policy"] in ("cold", "warm"), "cache_policy must be cold or warm")
    require(type(data["min_bytes"]) is int and data["min_bytes"] >= 16
            and data["min_bytes"] % 16 == 0, "min_bytes must be positive and 16-byte aligned")
    require(isinstance(data["buckets"], list) and bool(data["buckets"]), "buckets must be nonempty")
    previous = data["min_bytes"] - 16
    for row in data["buckets"]:
        required = {"max_bytes", "implementation", "threads", "blocks", "profiles"}
        require(isinstance(row, dict) and required <= set(row) <= required | {"cluster_size"},
                "invalid bucket fields")
        upper = row["max_bytes"]
        require(type(upper) is int and upper > previous and upper % 16 == 0,
                "bucket max_bytes must increase and be 16-byte aligned")
        previous = upper
        require(isinstance(row["implementation"], str) and row["implementation"] in MODES, "unknown joint kernel")
        require(type(row["threads"]) is int and row["threads"] in (32, 64, 128, 256, 512), "invalid threads")
        require(type(row["blocks"]) is int and 1 <= row["blocks"] <= 64, "invalid physical grid")
        cluster_size = row.get("cluster_size", 1)
        require(type(cluster_size) is int and cluster_size in (1, 8), "cluster_size must be 1 or 8")
        if cluster_size == 8:
            require(tuple(data["compute_capability"]) in ((12, 0), (12, 1)),
                    "clustered joint reductions require SM120 or SM121")
            require(row["threads"] == 512 and row["blocks"] == 64,
                    "clustered joint reductions require 512 threads and 64 physical CTAs")
        require(isinstance(row["profiles"], dict) and bool(row["profiles"]), "profiles must be nonempty")
        for mode, counts in row["profiles"].items():
            phased = mode in CLUSTER_MODES[row["implementation"]]
            require(mode in MODES[row["implementation"]] or (cluster_size == 8 and phased),
                    "mode does not belong to the prepared joint kernel")
            require(isinstance(counts, list) and bool(counts)
                    and all(type(v) is int and 1 <= v <= row["blocks"] for v in counts), "invalid active CTA counts")
            require(len(set(counts)) == len(counts), "duplicate active CTA counts")
            if phased:
                require(counts == [cluster_size], "phased profiles require exactly one complete active cluster")
    _validate_gathers(data["all_gather"], data["world_size"])
    return data


def load(path):
    try:
        return validate(json.loads(Path(path).read_text()))
    except (OSError, ValueError) as error:
        raise UsageError(f"cannot read collective ladder {path}: {error}") from error


def select(data, size_bytes, *, world_size, capability, inplace):
    if world_size != data["world_size"] or tuple(capability) != tuple(data["compute_capability"]):
        raise UsageError("collective ladder does not match the reduction's TP size or compute capability")
    if inplace != data["inplace"]:
        raise UsageError("collective ladder does not match the reduction's input/output aliasing")
    if type(size_bytes) is not int or size_bytes % 16 or not data["min_bytes"] <= size_bytes <= data["buckets"][-1]["max_bytes"]:
        raise UsageError(f"collective ladder has no bucket for {size_bytes} bytes")
    return next(row for row in data["buckets"] if size_bytes <= row["max_bytes"])


def resolve(mode, path):
    if mode == "runtime":
        if path is None:
            raise UsageError("--collective-tuning runtime requires --collective-ladder PATH")
        return load(path)
    if path is not None:
        raise UsageError("--collective-ladder requires --collective-tuning runtime")
    return None
