# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a signed two-rank legacy core with a custom primary basename.

The core is ordinary test data: both ranks have distinct weights, kernels,
module globals, and graph executables, plus one shared tokenizer asset. Its
signed ModelIndex carries a model identity and the cluster identity used by
distribution rank closure selection.
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

from ignition.preparation import runtime_core_pb2 as proto
import legacy_core
from signing import publisher


def build(path: Path, *, shard_size: int = 4096) -> None:
    index = proto.ModelIndex(
        version=4,
        contract=proto.ServingContract(
            model_id="owner/cluster-legacy",
            identity="cluster-legacy",
            model_contract=proto.ModelContract(),
        ),
        frontend=proto.FrontendSettings(),
        sampling_defaults=proto.DefaultSamplingParameters(),
    )
    index.cluster.world = 2
    index.cluster.export_id = "export-legacy"

    payload = []
    cursor = 0

    def add(span: proto.Span, name: str, data: bytes) -> None:
        nonlocal cursor
        source = Path(work) / name
        source.write_bytes(data)
        span.offset = cursor
        span.nbytes = len(data)
        payload.append((cursor, source, 0, len(data)))
        cursor += len(data)

    with tempfile.TemporaryDirectory(prefix="flare-legacy-cluster-") as work:
        for rank in range(2):
            bundle = proto.BundleIndex(architecture=[12, 0])
            weights = (f"rank-{rank}-weights-" * 16).encode()
            kernel = (f"rank-{rank}-kernel" * 16).encode()
            constant = f"constant-{rank}".encode()
            graph = (f"rank-{rank}-graph-" * 8).encode()
            add(bundle.resources["weights"], f"r{rank}-weights", weights)
            add(bundle.resources["kernel"], f"r{rank}-kernel", kernel)
            add(bundle.resources["constant"], f"r{rank}-constant", constant)
            add(bundle.resources["graph"], f"r{rank}-graph", graph)
            bundle.allocations.add(
                name="weights", file="weights", nbytes=len(weights)
            )
            bundle.modules.add(
                name="kernel",
                file="kernel",
                globals=[
                    proto.Global(
                        symbol="constant", file="constant", nbytes=len(constant)
                    )
                ],
            )
            bundle.graphs.add(name="decode", resource="graph")
            index.ranks.add().CopyFrom(bundle)

        tokenizer = b'{"model":{"type":"BPE"}}'
        asset = index.assets.add(name="tokenizer")
        add(asset.payload, "tokenizer", tokenizer)
        asset.sha256 = hashlib.sha256(tokenizer).hexdigest()

        path.parent.mkdir(parents=True, exist_ok=True)
        legacy_core.publish_legacy(
            path,
            index,
            payload,
            publisher=publisher(),
            shard_size=shard_size,
        )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} OUTPUT.safetensors")
    destination = Path(sys.argv[1])
    if destination.exists():
        raise SystemExit(f"destination exists: {destination}")
    build(destination)


if __name__ == "__main__":
    main()
