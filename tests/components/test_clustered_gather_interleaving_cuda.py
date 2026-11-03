# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Clustered reductions feed device-selected gathers across captured row sizes."""

from datetime import timedelta
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
                               reason="requires four explicitly assigned GPUs")


def _rank(rank, directory):
    import torch
    import torch.distributed as dist
    from ignition.components.collectives import PreparedAllGather, PreparedAllReduce, _all_gather, _all_reduce
    from ignition.preparation.aot import StorageBackend
    from ignition.preparation.collective_ladder import load
    from ignition.preparation.collectives import (
        CollectivePreparation, create_all_gather_runtime, create_all_reduce_runtime,
    )
    from ignition.preparation.startup_tuning import capture_consumers
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=(Path(directory) / "group").as_uri(),
                            rank=rank, world_size=4, timeout=timedelta(minutes=3))
    coordinator = CollectivePreparation(dist.group.WORLD)
    reduction = create_all_reduce_runtime(dist.group.WORLD, device, max_tokens=32,
                                         hidden_size=5120, collective_tuning="runtime")
    gather = create_all_gather_runtime(dist.group.WORLD, device, max_rows=32, columns=4 * 5120)
    session = RecordedPreparationSession(device=device, checkpoint="collective-dependencies",
        workload_schema="reduce-then-gather", numerical_policy="exact", coordinator=coordinator,
        collective_tuning="runtime", compile_workers=4, autotune_effort="default",
        collective_ladder=load(Path(__file__).resolve().parents[2] /
            "ignition/preparation/ladders/gb202_pcie_tp4_clustered_full.json"))
    cases, graphs = {}, {}
    try:
        for rows in (8, 16, 32):
            source = torch.zeros((rows, 5120), dtype=torch.bfloat16, device=device)
            reduced = torch.empty_like(source)
            gathered = torch.empty((rows, 4 * 5120), dtype=source.dtype, device=device)
            r = PreparedAllReduce(source, reduced, runtime=reduction, session=session,
                                  coordinator=coordinator, name=f"reduce-{rows}")
            g = PreparedAllGather(reduced, gathered, runtime=gather, session=session,
                                  coordinator=coordinator, name=f"gather-{rows}")
            cases[rows] = source, reduced, gathered, r, g
        session.tune(show_progress=False)

        def forward(reduce_handle, gather_handle):
            def run(source, reduced, gathered):
                _all_reduce(source, reduced, reduce_handle)
                _all_gather(reduced, gathered, gather_handle)
            return run

        for rows, (source, reduced, gathered, r, g) in cases.items():
            assert r.plan.prepared.state.query.call["joint_cluster_size"] == 8
            compiled = StorageBackend(forward(r.handle, g.handle), (source, reduced, gathered))
            compiled(source, reduced, gathered)
            torch.cuda.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), capture_consumers(session, graph), torch.cuda.graph(graph):
                compiled(source, reduced, gathered)
            assert session._startup_collectives.captures[id(graph)] == {id(r), id(g)}
            graphs[rows] = graph, compiled
        session.freeze()
        assert {p.gather for p in session._startup_collectives.groups.values()} == {False, True}
        for producer in session._startup_collectives.groups.values():
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                producer.run()
            graph.replay()
            torch.cuda.synchronize()
            assert producer.status.item() == 0
            choices = [None] * 4
            dist.all_gather_object(choices, tuple(int(s.selector.item()) for s in producer.specs))
            assert all(choice == choices[0] for choice in choices)
            graph.reset()

        for step, rows in enumerate((8, 32, 16, 8, 16, 32) * 3):
            source, reduced, gathered, _, _ = cases[rows]
            values = torch.arange(rows * 5120).reshape(1, rows, 5120)
            operands = ((values + 3 * torch.arange(4).reshape(4, 1, 1) + step) % 16).to(torch.bfloat16)
            expected = operands.float().sum(0).to(torch.bfloat16)
            source.copy_(operands[rank])
            reduced.fill_(float("nan"))
            gathered.fill_(float("nan"))
            before = torch.cuda.memory_stats()
            graphs[rows][0].replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated"):
                assert before[key] == after[key], key
            assert torch.equal(reduced.cpu(), expected)
            assert torch.equal(gathered.cpu(), expected.repeat(1, 4))
            assert torch.equal(source.cpu(), operands[rank])
    finally:
        torch.cuda.synchronize()
        for graph, _ in graphs.values():
            graph.reset()
        session.close()
        gather.close()
        reduction.close()
        dist.destroy_process_group()


def test_clustered_reduction_and_gather_share_captured_dependencies(tmp_path):
    import torch
    import torch.multiprocessing as mp

    assert torch.cuda.device_count() == 4
    mp.spawn(_rank, args=(str(tmp_path),), nprocs=4, join=True)
