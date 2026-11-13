"""Runtime gather profiles preserve arbitrary payload bits across captured calls."""

from datetime import timedelta
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
                               reason="requires four explicitly assigned GPUs")


def _rank(rank, directory):
    import torch
    import torch.distributed as dist
    from ignition.components.collectives import PreparedAllGather
    from ignition.preparation.collectives import CollectivePreparation, create_all_gather_runtime
    from ignition.preparation.collective_ladder import load
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.preparation.startup_tuning import capture_consumers
    from ignition.preparation.aot import StorageBackend

    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=(Path(directory) / "group").as_uri(),
                            rank=rank, world_size=4, timeout=timedelta(minutes=3))
    coordinator = CollectivePreparation(dist.group.WORLD)
    runtimes = [create_all_gather_runtime(dist.group.WORLD, device, max_rows=256, columns=columns)
                for columns in (256, 512, 1024, 3072, 4096, 16384)]
    session = RecordedPreparationSession(device=device, checkpoint="runtime-gather",
        workload_schema="column-gather-bits", numerical_policy="exact", coordinator=coordinator,
        collective_tuning="runtime", compile_workers=4, autotune_effort="default",
        collective_ladder=load(Path(__file__).resolve().parents[2] / "ignition/preparation/ladders/gb202_pcie_tp4.json"))
    operations, graphs = [], []
    try:
        cases = [(dtype, rows, 128, 1 if dtype == torch.float8_e4m3fn else 0)
                 for dtype in (torch.bfloat16, torch.float32, torch.float8_e4m3fn)
                 for rows in (1, 64, 256)]
        cases += [(torch.bfloat16, payload // width, width, channel)
                  for width, channel in ((512, 2), (2048, 4), (8192, 5))
                  for payload in (8192, 32768)]
        cases += [(torch.bfloat16, rows, 1536, 3) for rows in (8, 32)]
        for dtype, rows, width, channel in cases:
            source = torch.empty((rows, width // dtype.itemsize), device=device, dtype=dtype)
            output = torch.empty((rows, source.shape[1] * 4), device=device, dtype=dtype)
            operation = PreparedAllGather(source, output, runtime=runtimes[channel], session=session,
                coordinator=coordinator, name=f"gather-{dtype}-{width}-{rows}")
            operations.append((operation, source, output))
        session.tune(show_progress=False)
        for operation, source, output in operations:
            assert operation.plan.contract.component_id == "comm.pcie"
            assert operation.plan.prepared.state.selector.numel() == 1
            graph = torch.cuda.CUDAGraph()
            from ignition.components.collectives import _all_gather
            def forward(source, output):
                _all_gather(source, output, [], operation.handle)
            compiled = StorageBackend(forward, (source, output))
            compiled(source, output)
            torch.cuda.synchronize()
            dist.barrier()
            with session.capture(), capture_consumers(session, graph), torch.cuda.graph(graph):
                compiled(source, output)
            assert id(operation) in session._startup_collectives.captures[id(graph)]
            graphs.append((graph, compiled))
        session.freeze()
        for producer in session._startup_collectives.groups.values():
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                producer.run()
            graph.replay()
            torch.cuda.synchronize()
            assert producer.status.item() == 0
            values = [None] * 4
            dist.all_gather_object(values, tuple(int(s.selector.item()) for s in producer.specs))
            assert all(v == values[0] for v in values)
            graph.reset()
        for step in range(3):
            for (operation, source, output), (graph, compiled) in zip(operations, graphs, strict=True):
                rows, width = source.shape[0], source.shape[1] * source.element_size()
                values = ((torch.arange(4 * rows * width, device=device).reshape(4, rows, width)
                           + step * 53) % 256).to(torch.uint8)
                source.view(torch.uint8).copy_(values[rank])
                output.view(torch.uint8).fill_(0xff)
                expected = values.permute(1, 0, 2).contiguous().reshape(rows, 4 * width)
                before = torch.cuda.memory_stats()
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                for name in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated"):
                    assert before[name] == after[name], name
                assert torch.equal(output.view(torch.uint8), expected)
                assert torch.equal(source.view(torch.uint8), values[rank])
    finally:
        for graph, _ in graphs:
            graph.reset()
        session.close()
        for runtime in runtimes:
            runtime.close()
        dist.destroy_process_group()


def test_runtime_gather_bitwise_profiles_and_capture(tmp_path):
    import torch
    import torch.multiprocessing as mp

    assert torch.cuda.device_count() == 4
    mp.spawn(_rank, args=(str(tmp_path),), nprocs=4, join=True)
