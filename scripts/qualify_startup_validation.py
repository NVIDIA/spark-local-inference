# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise installed native startup rejection before CUDA or socket operations.

Manifest addresses are synthetic metadata, never valid allocation handles. Every
case must reject before CUDA driver loading or socket creation. strace guards
socket and ioctl calls while recording attempted operations in each subprocess.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ignition.preparation.qwen38 import (
    FRONTEND_SHA256, MODEL_ID, REVISION,
)

SNAPSHOT = Path.home() / ".cache/huggingface/hub" / (
    "models--" + MODEL_ID.replace("/", "--")
) / "snapshots" / REVISION
ABI_VERSION = 6
GRAPH_CAPACITIES = {
    **{kind: ((1, 1), (2, 1), (4, 1)) for kind in ("init", "restore", "snapshot", "resume", "decode")},
    "prefill": tuple((1, n) for n in (128, 256, 512)),
    "vision": tuple((1, n) for n in (256, 1024, 4096, 16384)),
}


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fixture(snapshot):
    """Return shape-valid metadata for the declared native Qwen contract."""
    import torch
    from ignition.components.state import StateLayout

    layout = StateLayout.from_specs((("consumed_tokens", (1,), torch.int64),))
    vocabulary, context = 248320, 262144
    descriptions = [
        ("row_slot_ids", "int32", [4]),
        ("row_generations", "int64", [4]),
        ("live_rows", "int32", [1]),
        ("query_lengths", "int32", [4]),
        ("request_state_rows", "int64", [4, 8]),
        ("request_pending_token", "int64", [4]),
        ("state_pool", "uint8", [9, layout.row_bytes]),
        ("memory_capacity", "int64", [2]),
        ("token_ids", "int64", [512]),
        ("prefill_positions", "int64", [512, 3]),
        ("page_tables", "int32", [4, context // 64]),
        ("sample_enabled", "int32", [4]),
        ("sampling_grammar_mask", "int32", [4, vocabulary // 32]),
        ("vision_token_indices", "int32", [512]),
        ("vision_pixel_values", "float32", [16384, 1536]),
        ("vision_interpolation_indices", "int64", [16384, 4]),
        ("vision_interpolation_weights", "float32", [16384, 4]),
        ("vision_position_ids", "int64", [16384, 2]),
        ("vision_cu_seqlens", "int32", [2]),
        ("vision_num_patches", "int32", [1]),
        ("vision_slot", "int32", [1]),
        ("vision_generation", "int64", [1]),
        ("vision_feature_offset", "int32", [1]),
        ("vision_feature_slot", "int32", [1]),
        ("admission_count", "int32", [1]),
        ("admission_slots", "int32", [4]),
        ("admission_generations", "int64", [4]),
        ("admission_seed_bits", "int64", [4]),
        ("admission_parameters", "float32", [4, 6]),
        ("admission_top_k", "int32", [4]),
        ("admission_max_new_tokens", "int64", [4]),
        ("admission_eos_token_ids", "int64", [4, 8]),
        ("admission_eos_count", "int32", [4]),
        ("admission_histogram", "int32", [4, vocabulary]),
        ("admission_grammar_mask", "int32", [4, vocabulary // 32]),
        ("admission_grammar_enabled", "int32", [4]),
        ("state_source_slots", "int64", [4]),
        ("state_destination_slots", "int64", [4]),
        ("state_operation", "int32", [4]),
        ("state_qsa_slots", "int64", [4]),
        ("state_accepted_tokens", "int32", [4]),
        ("state_expected_consumed", "int64", [4]),
        ("state_num_ops", "int32", [1]),
        ("frame_index", "int32", [1]),
        ("submission_sequence", "int64", [1]),
        ("completion_records", "int64", [2, 4, 24]),
    ]
    buffers, bindings = [], []
    address, host = 0x10000, 0
    for name, dtype, shape in descriptions:
        nbytes = math.prod(shape) * {"int64": 8, "uint8": 1}.get(dtype, 4)
        buffers.append({"name": name, "address": address, "nbytes": nbytes,
            "dtype": dtype, "shape": shape,
            "strides": [math.prod(shape[i + 1:]) for i in range(len(shape))]})
        address += (nbytes + 7) & ~7
        if name not in ("completion_records", "request_pending_token", "state_pool", "memory_capacity"):
            bindings.append({"buffer_name": name, "host_offset": host, "nbytes": nbytes})
            host += (nbytes + 7) & ~7
    from ignition.preparation.execution import GraphSignature
    from ignition.preparation.contract import text_inputs
    from ignition.recipes.qwen38.contract import prepare

    graphs, signatures = [], {}
    for kind, capacities in GRAPH_CAPACITIES.items():
        for batch, tokens in capacities:
            name = f"{kind}-{batch}-{tokens}"
            selected = ({b["buffer_name"] for b in bindings if b["buffer_name"].startswith("vision_")
                         and b["buffer_name"] != "vision_token_indices"} if kind == "vision" else set(text_inputs(kind)))
            if kind in ("init", "restore", "snapshot"):
                selected.update(("state_qsa_slots", "state_accepted_tokens"))
            if kind == "resume":
                selected.update(("sample_enabled", "page_tables"))
            fields = tuple(b["buffer_name"] for b in bindings if b["buffer_name"] in selected)
            signatures[name] = GraphSignature(kind, batch, tokens, fields)
            graphs.append({"name": name, "kind": kind,
                "sampling_mode": "general", "handle": len(graphs) + 1, "estimated_duration_ns": 1000,
                "batch_capacity": batch, "token_capacity": tokens,
                "prefill_capacity": 0, "confidence_width": 0,
                "input_bindings": [deepcopy(b) for b in bindings if b["buffer_name"] in
                                   (*signatures[name].inputs, "frame_index", "submission_sequence")]})
    manifest = {"abi_version": ABI_VERSION, "completion_token_capacity": 8, "context": 1, "device": 0, "max_active": 4,
        "input_frame_bytes": host, "buffers": buffers, "graphs": graphs}
    from ignition.preparation.frontend import annotated_template_asset, prepare_frontend
    from ignition.recipes.qwen38.recipe import RECIPE

    pool = SimpleNamespace(layout=layout)
    contract = prepare(signatures, dict.fromkeys(item["name"] for item in buffers), pool,
        page_tokens=64, draft_lag=None)
    config = {"model_id": MODEL_ID, "identity": "01" * 32, "max_sequence_tokens": context,
        "prefill_chunk_size": 512,
        "pages": 4096, "snapshots": 4, "state_rows": 9, "page_tokens": 64,
        "frontend": prepare_frontend(snapshot, eos_token_ids=[248046, 248044], image_profile="qwen"),
        "sampling_defaults": RECIPE.sampling_defaults.as_dict(),
        "model_contract": contract,
        "bind": "127.0.0.1:0",
        "assets": {"tokenizer": {"path": str((snapshot / "tokenizer.json").resolve(strict=True)),
            "sha256": FRONTEND_SHA256["tokenizer.json"]},
            "chat_template": annotated_template_asset((snapshot / "chat_template.jinja").read_bytes(),
                ROOT / "ignition/recipes/qwen38/assets/chat_template.patch.json")}}
    return manifest, config


def cases(snapshot):
    manifest, config = fixture(snapshot)
    result = []

    def add(name, expected, change):
        m, c = deepcopy(manifest), deepcopy(config)
        change(m, c)
        result.append({"name": name, "expected_error": expected, "manifest": m, "config": c})

    def buffer(m, name):
        return next(item for item in m["buffers"] if item["name"] == name)

    add("abi-version", "unsupported execution ABI version", lambda m, c: m.update(abi_version=2))
    add("null-context", "nonnull context", lambda m, c: m.update(context=0))
    add("overlapping-buffers", "overlap", lambda m, c:
        buffer(m, "row_generations").update(address=buffer(m, "row_slot_ids")["address"]))
    add("completion-layout", "required buffer layout", lambda m, c:
        buffer(m, "completion_records").update(shape=[4, 2, 24], strides=[48, 24, 1]))
    add("missing-resume4", "lacks Resume/General graph with batch 4, tokens 1", lambda m, c:
        m.update(graphs=[g for g in m["graphs"] if g["name"] != "resume-4-1"]))
    add("missing-prefill512", "lacks Prefill/General graph with batch 1, tokens 512", lambda m, c:
        m.update(graphs=[g for g in m["graphs"] if g["name"] != "prefill-1-512"]))

    add("missing-sampling-mode", "'sampling_mode'", lambda m, c: m["graphs"][0].pop("sampling_mode"))
    add("unknown-sampling-mode", 'unsupported graph sampling mode "unknown"', lambda m, c:
        m["graphs"][0].update(sampling_mode="unknown"))
    def ambiguous_variant(m, c):
        graph = deepcopy(next(g for g in m["graphs"] if g["name"] == "prefill-1-512"))
        graph.update(name="duplicate-prefill-1-512", handle=len(m["graphs"]) + 1)
        m["graphs"].append(graph)
        contract = deepcopy(next(g for g in c["model_contract"]["graphs"]
                                 if g["graph"] == "prefill-1-512"))
        contract["graph"] = graph["name"]
        c["model_contract"]["graphs"].append(contract)

    add("ambiguous-graph-capacity", "ambiguous prepared model graph capacity", ambiguous_variant)

    def missing_input(m, c):
        graph = next(g for g in m["graphs"] if g["name"] == "prefill-1-512")
        graph["input_bindings"] = [b for b in graph["input_bindings"]
            if b["buffer_name"] != "vision_token_indices"]

    add("missing-semantic-input", "semantic input must have one graph transport binding", missing_input)
    add("wrong-page-table-layout", "buffer page_tables must be contiguous row-major", lambda m, c:
        buffer(m, "page_tables").update(strides=[1, 4]))
    add("state-view-extent", "persistent state view exceeds its prepared storage", lambda m, c:
        c["model_contract"]["cache"]["regions"][0]["storage"].update(
            nbytes=c["model_contract"]["cache"]["regions"][0]["storage"]["stride"] + 1))
    def pending_consumer(m, c):
        graph = next(g for g in c["model_contract"]["graphs"] if g["graph"] == "init-1-1")
        graph["pending_token"] = dict(buffer="request_pending_token", offset=0, stride=8, nbytes=8)

    add("write-only-pending-consumer", "pending token requires a declared device-state read", pending_consumer)
    add("invalid-model-context", "invalid prepared model configuration", lambda m, c:
        c.update(max_sequence_tokens=262145))
    add("valid-shapes-reach-asset-validation", "assets must contain verified frontend descriptors",
        lambda m, c: c.pop("assets"))
    add("asset-checksum", "frontend asset SHA256 mismatch", lambda m, c:
        c["assets"]["chat_template"].update(sha256="00" * 32))
    add("frontend-worker-capacity", "invalid HTTP admission limits", lambda m, c:
        c.update(frontend_workers=0))
    add("request-queue-capacity", "invalid native request/event queue capacity", lambda m, c:
        c.update(queue_capacity=0))
    add("event-queue-capacity", "invalid native request/event queue capacity", lambda m, c:
        c.update(event_capacity=1))
    return result


def child(path):
    import flare_heliopause

    case = json.loads(path.read_text())

    class Owner:
        pass

    owner = Owner()
    retained = weakref.ref(owner)
    result = {"name": case["name"]}
    try:
        server = flare_heliopause.Server(owner, case["manifest"], json.dumps(case["config"]))
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error),
            resources_retained=getattr(error, "resources_retained", False),
            rejected=True, expected_error=case["expected_error"])
    else:
        result.update(rejected=False, error="unexpected native server construction")
        server.close()
    del owner
    gc.collect()
    result["owner_released"] = retained() is None
    result["passed"] = bool(result["rejected"] and
        case["expected_error"] in result["error"] and result["owner_released"] and
        not result.get("resources_retained"))
    print(json.dumps(result), flush=True)
    return 0 if result["passed"] else 1


def run(args):
    if not shutil.which("strace"):
        raise RuntimeError("strace is required to reject CUDA and networking attempts")
    passed = True
    with TemporaryDirectory(prefix="flare-startup-validation-") as directory:
        for case in cases(args.snapshot):
            path = Path(directory) / (case["name"] + ".json")
            save(path, case)
            command = ["strace", "-f", "-qq", "-s", "512",
                "-e", "trace=network,open,openat,ioctl",
                "-e", "inject=socket:error=EPERM", "-e", "inject=bind:error=EPERM",
                "-e", "inject=listen:error=EPERM", "-e", "inject=ioctl:error=EPERM",
                sys.executable, "-c",
                'import runpy, sys; runpy.run_path(sys.argv.pop(1), run_name="__main__")',
                str(Path(__file__).resolve()), "--child", str(path)]
            process = subprocess.run(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                text=True, capture_output=True, timeout=60)
            try:
                observed = json.loads(process.stdout)
            except ValueError:
                raise AssertionError(f"{case['name']}: child did not return JSON: {process.stdout}\n{process.stderr}")
            network = re.search(r"\b(?:socket|bind|listen|connect)\(", process.stderr)
            cuda = re.search(r"libcuda(?:rt)?[./]|/dev/nvidia", process.stderr)
            observed["passed"] &= process.returncode == 0 and not network and not cuda
            print(json.dumps(observed), flush=True)
            if not observed["passed"]:
                print(process.stderr, file=sys.stderr)
                passed = False
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--child", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child is not None:
        return child(args.child)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
