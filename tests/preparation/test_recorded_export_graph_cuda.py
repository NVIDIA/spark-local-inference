# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Retained export graphs preserve storage and serialize their execution metadata."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires explicit CUDA test opt-in")
def test_export_capture_retains_cuda_handles_and_preserves_storage(tmp_path):
    subprocess.run([sys.executable, "-c",
        "from pathlib import Path; import sys; "
        "from tests.preparation.test_recorded_export_graph_cuda import check_export; "
        "check_export(Path(sys.argv[1]))", str(tmp_path)],
        check=True, timeout=120, cwd=Path(__file__).resolve().parents[2])


def check_export(tmp_path):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import torch
    from ignition.preparation.capture import capture_forwards
    from ignition.preparation.execution import GraphSignature, PreparedExecution, TuningGraph
    from ignition.preparation.module_images import ModuleImages
    from ignition.preparation.core_file import read_sources
    from ignition.preparation.core_storage import CoreStorage
    from ignition.preparation.runtime_core import (
        Allocation, RuntimeCore, capture_graph, instantiate_graph,
    )

    with ModuleImages() as images:
        torch.cuda.set_device(0)
        storage = CoreStorage()
        with storage:
            source = torch.arange(1024, device="cuda", dtype=torch.float32)
            output = torch.empty_like(source)
            frame = torch.zeros(1, device="cuda", dtype=torch.int32)
            sequence = torch.zeros(1, device="cuda", dtype=torch.int64)
            records = torch.zeros((2, 1, 24), device="cuda", dtype=torch.int64)
            selector = torch.zeros(1, device="cuda", dtype=torch.int32)
            status = torch.ones(1, device="cuda", dtype=torch.int32)
            def forward():
                output.copy_(source * 3.0 + 4.0)
            stream = torch.cuda.Stream()
            graphs, _ = capture_forwards({"decode": forward}, {"decode": GraphSignature("decode", 1, 1)},
                session=SimpleNamespace(capture=nullcontext), device=0, stream=stream,
                pool=storage.graph_pool.id, warmup_storage=storage, retain_graphs=True, compile_decoder=False)
            graph = graphs["decode"]
            assert isinstance(graph, torch.cuda.CUDAGraph)
            assert graph._keep_graph
            assert storage.graph_pool.use_count() >= 1
            before = [(segment["address"], segment["total_size"])
                      for segment in storage.graph_pool.snapshot(include_traces=False)]
            torch.cuda.empty_cache()
            assert before == [(segment["address"], segment["total_size"])
                              for segment in storage.graph_pool.snapshot(include_traces=False)]
            tuner = capture_graph(retain_graphs=True)
            with torch.cuda.graph(tuner, pool=storage.graph_pool.id, stream=stream):
                selector.fill_(1)
            instantiate_graph(tuner)
            buffers = dict(source=source, output=output, frame_index=frame, submission_sequence=sequence,
                completion_records=records, selector=selector, status=status)
            bundle = PreparedExecution(device=0, graphs=graphs, buffers=buffers, max_active=1,
                signatures={"decode": GraphSignature("decode", 1, 1, ("source",))},
                tuning_graphs={"tune": TuningGraph(tuner, ("selector",), "status")})
            manifest = bundle.export_manifest()
            assert all("handle" not in value for value in (*manifest["graphs"], *manifest["tuning_graphs"]))
            runtime_manifest = bundle.manifest()
            assert runtime_manifest["graphs"][0]["kind"] == "decode"
            assert "handle" in runtime_manifest["graphs"][0]
            path = tmp_path / "rank.json"
            allocations = [Allocation(name, tensor.data_ptr(), tensor.nbytes)
                           for name, tensor in buffers.items()]
            allocations.extend(Allocation(f"graph-pool-{index}", segment["address"], segment["total_size"])
                for index, segment in enumerate(storage.graph_pool.snapshot(include_traces=False)))
            package = RuntimeCore(allocations=allocations, modules={}, kernels=(),
                recorder=images.native, owners=(bundle, storage))
            package.write(path, bundle.graphs, execution=manifest, sources=True)
            index, _ = read_sources(path)
            assert index.ranks[0].graphs[0].flags == 1
            assert index.ranks[0].graphs[0].node_count > 0
            assert {value.name for value in index.ranks[0].graphs} == {"decode", "tune"}
            graph.reset()
            graph.reset()
            tuner.reset()
