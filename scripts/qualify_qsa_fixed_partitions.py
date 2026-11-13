# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare captured selected-position attention on saved Qwen4Exp operands."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
GPU = "GPU-afd4b1ad-8a64-7057-4bde-241822724c7f"
ARTIFACT_SHA = "1c9e3fdfc57e0cdc1f50420034ea03fd77e370b593f8aecfde3c62be72835397"
REFERENCE_SHA = "3ef0c80ee8de7fa21d2d9bc844c56ab9942624cbe3fb72c0a31b05e0b000fa09"
TAIL_CAPACITIES = (1, 2, 4, 16, 64, 128, 512)
PREFILL_CAPACITIES = (16, 64, 128, 512)
SPLITS = (1, 16, 32, 64)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def tensor_info(value):
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "strides": list(value.stride()), "sha256": tensor_sha(value)}


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def comparison(actual, expected):
    a, b = actual.contiguous(), expected.contiguous()
    equal = torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    error = a.float() - b.float()
    different = (a.view(torch.int16) != b.view(torch.int16))
    return {"bitwise_equal": equal, "different_elements": int(different.sum()),
            "different_rows": int(different.reshape(a.shape[0], -1).any(dim=1).sum()),
            "relative_l2": float(error.norm() / b.float().norm().clamp_min(1e-30)),
            "max_absolute": float(error.abs().max()),
            "rows": [{"row": row, "bitwise_equal": not bool(different[row].any()),
                      "different_elements": int(different[row].sum()),
                      "max_absolute": float(error[row].abs().max())}
                     for row in range(a.shape[0])]}


def quantiles(values):
    ordered = sorted(values)
    def percentile(p):
        at = p * (len(ordered) - 1)
        lo, hi = math.floor(at), math.ceil(at)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (at - lo)
    return {"count": len(values), "median_us": statistics.median(values),
            "min_us": min(values), "p10_us": percentile(.1),
            "p90_us": percentile(.9), "max_us": max(values)}


def load_inputs(args):
    if sha(args.artifact) != ARTIFACT_SHA:
        raise ValueError("saved QSA input artifact SHA256 does not match the fixed provenance")
    reference_receipt = json.loads(args.reference_receipt.read_text())
    if reference_receipt["input_sha256"] != ARTIFACT_SHA:
        raise ValueError("durable QSA baseline receipt identifies another input artifact")
    if sha(args.reference) != REFERENCE_SHA or reference_receipt["tensor_sha256"] != REFERENCE_SHA:
        raise ValueError("saved direct512/split16 reference SHA256 does not match its durable receipt")
    observations = torch.load(args.artifact, map_location="cpu", weights_only=True)
    references = torch.load(args.reference, map_location="cpu", weights_only=True)
    records, inventory = {}, {}
    for name, expected_count in (("tail11", 11), ("prefill459", 459)):
        source = "restored_full_capacity_tail" if name == "tail11" else "cold"
        record = observations[source]["qsa"]
        ids = record["metadata"]["request_ids"]
        count = int((ids >= 0).sum())
        if count != expected_count or not bool((ids[:count] == 0).all()) or not bool((ids[count:] == -1).all()):
            raise ValueError(f"{source} active row layout differs from the recorded contract")
        selected = record["tensors"]["selected_positions"][:count]
        positions = record["metadata"]["query_positions"][:count]
        table = record["metadata"]["main_block_table"]
        if bool(((selected >= 0) & (selected > positions[:, None])).any()):
            raise ValueError("saved selection contains future positions")
        used = set(table[0, torch.div(selected[selected >= 0].long(), 64, rounding_mode="floor")].tolist())
        for field in ("main_k_cache", "main_v_cache"):
            cache = record["cache_after"][field]
            if not used <= set(cache["page_ids"]):
                raise ValueError("saved selection refers to an unrecorded physical cache page")
        records[name] = record
        inventory[name] = {"observation": source, "active_rows": count,
            "query": tensor_info(record["tensors"]["query"][:count]),
            "selected_positions": tensor_info(selected),
            "metadata": {key: tensor_info(value) for key, value in record["metadata"].items()},
            "cache_after": {key: {"page_ids": value["page_ids"],
                "recorded_pool_shape": list(value["pool_shape"]),
                "values": tensor_info(value["values"])}
                for key, value in record["cache_after"].items()},
            "recorded_attended": tensor_info(record["tensors"]["attended"][:count]),
            "query_positions": positions.tolist(), "used_physical_pages": sorted(used)}
    if not torch.equal(references["512"]["attended"], records["tail11"]["tensors"]["attended"][:11]):
        raise ValueError("direct512 reference disagrees with the full-capacity recorded attention output")
    if not torch.equal(references["16"]["selected"], records["tail11"]["tensors"]["selected_positions"][:11]):
        raise ValueError("split16 baseline uses a different selection")
    return records, references, inventory


class ImmutableCache:
    def __init__(self, record):
        self.buffers = {"block_table": record["metadata"]["main_block_table"].cuda()}
        for field in ("main_k_cache", "main_v_cache"):
            saved = record["cache_after"][field]
            pages = max(saved["page_ids"]) + 1
            tensor = torch.zeros((pages, *saved["values"].shape[1:]), dtype=saved["values"].dtype, device="cuda")
            for page, value in zip(saved["page_ids"], saved["values"]):
                tensor[page].copy_(value)
            self.buffers[field] = tensor
        self.initial_hashes = {key: tensor_sha(value) for key, value in self.buffers.items()}


class CapturedAttention:
    def __init__(self, workload, record, cache, capacity, splits, stream, kernels):
        self.workload, self.capacity, self.splits = workload, capacity, splits
        self.name = f"{workload}-capacity{capacity}-" + ("direct1" if splits == 1 else f"split{splits}")
        self.count = int((record["metadata"]["request_ids"] >= 0).sum())
        self.groups, self.buffers, self.immutable = [], {}, {}
        self.cache, self.kernels = cache, kernels
        self.partial = self.lse = None
        if splits > 1:
            self.partial = torch.full((capacity, splits, 24, 256), float("nan"), dtype=torch.float32, device="cuda")
            self.lse = torch.full((capacity, splits, 24), float("nan"), dtype=torch.float32, device="cuda")
            self.buffers.update(partial=self.partial, partial_lse=self.lse)
        for start in range(0, self.count, capacity):
            active = min(capacity, self.count - start)
            query = torch.zeros((capacity, 24, 256), dtype=torch.bfloat16, device="cuda")
            selected = torch.full((capacity, 2051), -1, dtype=torch.int32, device="cuda")
            request_ids = torch.full((capacity,), -1, dtype=torch.int64, device="cuda")
            positions = torch.full((capacity,), -1, dtype=torch.int64, device="cuda")
            query[:active].copy_(record["tensors"]["query"][start:start + active])
            selected[:active].copy_(record["tensors"]["selected_positions"][start:start + active])
            request_ids[:active].copy_(record["metadata"]["request_ids"][start:start + active])
            positions[:active].copy_(record["metadata"]["query_positions"][start:start + active])
            output = torch.full_like(query, float("nan"))
            group = dict(query=query, selected_positions=selected, request_ids=request_ids,
                         query_positions=positions, output=output, active=active, start=start)
            self.groups.append(group)
            for key, value in group.items():
                if isinstance(value, torch.Tensor):
                    name = f"group{start}.{key}"
                    self.buffers[name] = value
                    if key != "output":
                        self.immutable[name] = tensor_sha(value)
        self.start_event = torch.cuda.Event(enable_timing=True, external=True)
        self.end_event = torch.cuda.Event(enable_timing=True, external=True)
        self.start_event.record()
        self.end_event.record()
        self.run()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.start_event.record(stream)
            self.run()
            self.end_event.record(stream)
        torch.cuda.synchronize()

    def run(self):
        for group in self.groups:
            operands = {key: group[key] for key in ("query", "selected_positions", "request_ids", "query_positions")}
            operands.update(key_cache=self.cache.buffers["main_k_cache"],
                value_cache=self.cache.buffers["main_v_cache"],
                block_table=self.cache.buffers["block_table"], k_descale=None, v_descale=None, softmax_scale=1 / 16)
            if self.splits == 1:
                self.kernels.launch_selected_paged_gqa_direct(**operands, output=group["output"], output_lse=None, kv_warps=2)
            else:
                self.kernels.launch_sparse_gqa_split(**operands, partial_output=self.partial, partial_lse=self.lse, splits=self.splits)
                self.kernels.launch_sparse_gqa_merge(partial_output=self.partial, partial_lse=self.lse,
                    output=group["output"], output_lse=None, rows=self.capacity, splits=self.splits)

    def readback(self):
        values, padding = [], []
        for group in self.groups:
            value = group["output"].cpu()
            values.append(value[:group["active"]])
            if group["active"] < self.capacity:
                padding.append(value[group["active"]:])
        return torch.cat(values), all(bool((value == 0).all()) for value in padding)

    def description(self):
        return {"workload": self.workload, "capacity": self.capacity, "splits": self.splits,
            "active_rows": self.count, "groups": [{"first_logical_row": group["start"], "active_rows": group["active"]} for group in self.groups],
            "launches_per_graph": len(self.groups) * (1 if self.splits == 1 else 2),
            "merge_variant": None if self.splits == 1 else "serial" if self.capacity > self.splits else "cooperative",
            "partial_workspace_bytes": 0 if self.partial is None else self.partial.nbytes + self.lse.nbytes,
            "owned_tensor_bytes": sum(value.nbytes for value in self.buffers.values()),
            "input_sha256": self.immutable,
            "graph_exec": self.graph.raw_cuda_graph_exec()}


def gpu_state():
    return subprocess.run(["nvidia-smi", "-i", GPU,
        "--query-gpu=uuid,name,driver_version,memory.used,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw",
        "--format=csv,noheader"], check=True, text=True, capture_output=True).stdout.strip()


def execute(args, records, references, receipt):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU or os.environ.get("CUDA_HOME") != "/opt/cuda":
        raise RuntimeError("explicit assigned GPU UUID and CUDA_HOME=/opt/cuda are required")
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv,noheader"],
                               check=True, text=True, capture_output=True).stdout
    if any(line.startswith(GPU) for line in processes.splitlines()):
        raise RuntimeError(f"assigned GPU is occupied before preparation: {processes}")
    import b12x.attention.paged._selected_forward as kernels
    import flare_heliopause
    from ignition.preparation.graphs import PreparedGraphBundle
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    torch.cuda.set_device(0)
    receipt["device_before"] = gpu_state()
    receipt["versions"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
        "flashinfer": importlib.metadata.version("flashinfer-python"),
        "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl")}
    native_path = next(Path(flare_heliopause.__file__).parent.glob("*.so"))
    source_root = Path(kernels.__file__).resolve().parents[2]
    receipt["installed_native"] = {"path": str(native_path), "sha256": sha(native_path)}
    receipt["b12x_source_root"] = str(source_root)
    receipt["b12x_sources"] = {str(path.relative_to(source_root)): sha(path) for path in sorted(source_root.rglob("*.py"))}
    receipt["selected_forward_source"] = {"path": str(Path(kernels.__file__).resolve()), "sha256": sha(kernels.__file__)}
    source_copy = args.artifacts / "installed-selected-forward.py"
    shutil.copyfile(kernels.__file__, source_copy)
    stream = torch.cuda.Stream()
    caches = {name: ImmutableCache(record) for name, record in records.items()}
    owners = []
    for workload, capacities in (("tail11", TAIL_CAPACITIES), ("prefill459", PREFILL_CAPACITIES)):
        for splits in SPLITS:
            for capacity in capacities:
                print(f"CAPTURE {workload} capacity={capacity} splits={splits}", flush=True)
                owners.append(CapturedAttention(workload, records[workload], caches[workload], capacity, splits, stream, kernels))
    by_name = {owner.name: owner for owner in owners}
    graphs = {owner.name: owner.graph for owner in owners}
    buffers = {owner.name + "." + key: value for owner in owners for key, value in owner.buffers.items()}
    buffers.update({name + ".cache." + key: value for name, cache in caches.items() for key, value in cache.buffers.items()})
    bundle = PreparedGraphBundle(device=0, graphs=graphs, buffers=buffers, resources=(*owners, caches, stream, kernels))
    receipt["cases"] = {owner.name: owner.description() for owner in owners}
    receipt["cuda_memory_after_capture"] = {"allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved(),
                                             "max_allocated": torch.cuda.max_memory_allocated()}
    receipt["cache_storage"] = {name: {key: tensor_info(value) for key, value in cache.buffers.items()} for name, cache in caches.items()}
    receipt["observations"], receipt["baseline_checks"] = [], {}
    samples = {owner.name: [] for owner in owners}
    outputs, reference_hashes = {}, {}
    worker = None
    def replay(owner):
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        ticket = worker.submit(bundle.graph_indices[owner.name])
        result = worker.poll(timeout_ms=60_000)
        if result is None or result.ticket != ticket or result.error is not None:
            raise RuntimeError(f"native replay failed for {owner.name}: {result}")
        after = torch.cuda.memory_stats()["allocation.all.allocated"]
        elapsed_us = owner.start_event.elapsed_time(owner.end_event) * 1000
        observed, padding_zero = owner.readback()
        details = {"allocation_count_before": before, "allocation_count_after": after,
            "elapsed_us": elapsed_us, "output_sha256": tensor_sha(observed), "inactive_output_zero": padding_zero}
        if before != after or not padding_zero or not bool(torch.isfinite(observed).all()):
            raise AssertionError(f"replay allocation/output contract failed: {owner.name}: {details}")
        if not math.isfinite(elapsed_us) or elapsed_us <= 0:
            raise AssertionError(f"invalid timing event duration: {details}")
        return observed, details
    try:
        worker = bundle.start_worker()
        for name, expected in (
            ("tail11-capacity512-direct1", references["512"]["attended"]),
            ("tail11-capacity16-split16", references["16"]["attended"]),
            ("prefill459-capacity512-direct1", records["prefill459"]["tensors"]["attended"][:459]),
        ):
            observed, details = replay(by_name[name])
            result = comparison(observed, expected)
            receipt["baseline_checks"][name] = {**details, **result}
            save(args.output, receipt)
            if not result["bitwise_equal"]:
                torch.save({"actual": observed, "expected": expected}, args.artifacts / (name + "-baseline-failure.pt"))
                raise AssertionError(f"recorded attention baseline reproduction failed: {name}")
        print("RECORDED BASELINES reproduced bitwise: direct512, split16, cold459 direct512", flush=True)
        for iteration in range(args.warmups + args.repetitions):
            order = owners if iteration % 2 == 0 else list(reversed(owners))
            for owner in order:
                observed, details = replay(owner)
                name = owner.name
                prior = reference_hashes.setdefault(name, details["output_sha256"])
                stable = prior == details["output_sha256"]
                receipt["observations"].append({"case": name, "iteration": iteration,
                    "warmup": iteration < args.warmups, "repeat_bitwise_equal": stable, **details})
                if name not in outputs:
                    outputs[name] = observed
                if not stable:
                    torch.save({"actual": observed, "first": outputs[name]}, args.artifacts / (name + "-repeat-failure.pt"))
                    raise AssertionError(f"captured replay is not repeatable: {name}")
                if iteration >= args.warmups:
                    samples[name].append(details["elapsed_us"])
            save(args.output, receipt)
            print(f"REPLAY {iteration + 1}/{args.warmups + args.repetitions}: {len(owners)} captured variants", flush=True)
        receipt["timing_us"] = {name: {"raw": values, **quantiles(values)} for name, values in samples.items()}
        receipt["comparisons"] = {}
        for workload, capacities in (("tail11", TAIL_CAPACITIES), ("prefill459", PREFILL_CAPACITIES)):
            direct = outputs[f"{workload}-capacity512-direct1"]
            for splits in SPLITS:
                variant = "direct1" if splits == 1 else f"split{splits}"
                same_split = outputs[f"{workload}-capacity{capacities[0]}-{variant}"]
                for capacity in capacities:
                    name = f"{workload}-capacity{capacity}-{variant}"
                    receipt["comparisons"][name] = {"same_partition_across_capacity": comparison(outputs[name], same_split),
                        "versus_recorded_direct512": comparison(outputs[name], direct)}
        receipt["immutable_input_checks"] = {}
        for owner in owners:
            receipt["immutable_input_checks"][owner.name] = all(tensor_sha(owner.buffers[key]) == expected for key, expected in owner.immutable.items())
        for name, cache in caches.items():
            receipt["immutable_input_checks"][name + ".cache"] = all(tensor_sha(cache.buffers[key]) == expected for key, expected in cache.initial_hashes.items())
        receipt["fixed_partition_capacity_invariant"] = all(value["same_partition_across_capacity"]["bitwise_equal"] for value in receipt["comparisons"].values())
        receipt["passed"] = receipt["fixed_partition_capacity_invariant"] and all(receipt["immutable_input_checks"].values())
        torch.save(outputs, args.artifacts / "attention-outputs.pt")
        receipt["output_tensors"] = {"path": str(args.artifacts / "attention-outputs.pt"), "sha256": sha(args.artifacts / "attention-outputs.pt")}
        receipt["device_after"] = gpu_state()
    finally:
        if worker is not None:
            worker.close()
            receipt["native_worker_closed"] = True
        for graph in graphs.values():
            graph.reset()
        receipt["graph_handles_released"] = True
        save(args.output, receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=Path("/tmp/sparkgap-prefix-qsa-recording.pt"))
    parser.add_argument("--reference", type=Path, default=Path("/tmp/sparkgap-qsa-capacity-isolation.pt"))
    parser.add_argument("--reference-receipt", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("/tmp/sparkgap-qsa-fixed-partitions"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.warmups != 3 or args.repetitions != 20:
        parser.error("this qualification requires exactly three warmups and twenty measured repetitions")
    records, references, inventory = load_inputs(args)
    if args.validate_only:
        print(json.dumps({"input_sha256": ARTIFACT_SHA, "reference_sha256": REFERENCE_SHA,
            "active_rows": {key: value["active_rows"] for key, value in inventory.items()},
            "cases": len(SPLITS) * (len(TAIL_CAPACITIES) + len(PREFILL_CAPACITIES)),
            "cuda_initialized": torch.cuda.is_initialized()}, indent=2))
        return
    if args.output.exists():
        parser.error("output receipt already exists; preserve it and choose another output path")
    args.artifacts.mkdir(parents=True, exist_ok=True)
    if (args.artifacts / "attention-outputs.pt").exists():
        parser.error("artifact output directory already contains tensor results")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, args.artifacts / Path(__file__).name)
    receipt = {"schema": "flare.qsa-fixed-partitions.v1", "passed": False,
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)},
        "command": sys.argv, "input_artifact": {"path": str(args.artifact), "sha256": ARTIFACT_SHA},
        "reference_artifact": {"path": str(args.reference), "sha256": REFERENCE_SHA},
        "reference_receipt": {"path": str(args.reference_receipt), "sha256": sha(args.reference_receipt)},
        "gpu_uuid": GPU, "warmups": args.warmups, "repetitions": args.repetitions,
        "inputs": inventory, "scope": "Attention-only recorded layer3 QSA: selected positions, query, and main KV pages held immutable. No selector, projections, model state, cache writer, HTTP, or Python replay timing.",
        "timing_scope": "CUDA events captured around all attention launches needed for each complete saved workload; consecutive groups share one split workspace. Native driver replay; readbacks and preparation excluded.",
        "padding": "Inactive request/query-position/selection entries=-1, inactive queries=0; inactive outputs must be zero.",
        "cache_reconstruction": "Recorded physical page IDs are preserved in a compact eight-page pool; every referenced page is validated present. Block-table values and active operands are unchanged.",
        "baseline_policy": "Recorded direct512 and split16 must reproduce bitwise before the capacity matrix. Partition comparisons against direct are diagnostic differences, not model-accuracy acceptance."}
    started = time.monotonic()
    try:
        save(args.output, receipt)
        execute(args, records, references, receipt)
    except BaseException:
        receipt["failure"] = traceback.format_exc()
        raise
    finally:
        receipt["wall_seconds"] = time.monotonic() - started
        save(args.output, receipt)
        shutil.copyfile(args.output, args.artifacts / "receipt.json")
    print(json.dumps({"passed": receipt["passed"], "baseline_checks": list(receipt["baseline_checks"]),
        "fixed_partition_capacity_invariant": receipt["fixed_partition_capacity_invariant"],
        "cases": len(receipt["cases"]), "native_replays": len(receipt["observations"]) + len(receipt["baseline_checks"]),
        "receipt": str(args.output)}, indent=2))
    if not receipt["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
