"""TP16 staged latent bytes and BF16 attention against direct peer gathering.

Both paths consume identical selected IDs in their original order. No checkpoint
weights or alternate indexer arithmetic participate in this transport fixture.
"""

from datetime import timedelta
import os

import pytest
import torch
from qualify.kimi_staged_prefill import SelectedHistory, WORLD, PAGE, WIDTH, TOPK

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
                              reason="requires sixteen explicitly assigned CUDA devices")




def _rank(rank, rendezvous, rows, maximum, compiled):
    from ignition import runtime_patches
    runtime_patches.apply()
    import torch.distributed as dist
    from ignition.components.shared_pages import SharedPages
    from ignition.preparation.collectives import CollectivePreparation, create_all_reduce_runtime
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", rank=rank, world_size=WORLD, init_method=rendezvous,
                            timeout=timedelta(minutes=15))
    graph = transfer_graph = runtime = shared_cache = stage = None
    try:
        with PackageStorage(rank=rank, world=WORLD, device=rank) as storage:
            coordinator = CollectivePreparation(dist.group.WORLD)
            shared_cache = SharedPages(dist.group.WORLD, device, maximum // WORLD * WIDTH, ipc=storage.peer_storage)
            stage = SharedPages(dist.group.WORLD, device, maximum // WORLD * WIDTH, ipc=storage.peer_storage)
            cache = shared_cache.tensor.view(torch.float8_e4m3fn).view(-1, PAGE, WIDTH)
            output = torch.empty((maximum, WIDTH), dtype=cache.dtype, device=device)
            signals = tuple(torch.zeros((1, 8), dtype=torch.bfloat16, device=device) for _ in range(4))
            runtime = create_all_reduce_runtime(dist.group.WORLD, device,
                collective_tuning="static", max_tokens=1, hidden_size=8, ipc=storage.peer_storage)
            with RecordedPreparationSession(device=device, checkpoint="kimi31-staged-prefill-transport",
                    workload_schema="direct-and-staged-selected-records", numerical_policy="exact-bytes-bf16-query",
                    autotune_effort="off", coordinator=coordinator) as session:
                arena = ScratchArena(device)
                session.register_scratch(arena)
                operation = SelectedHistory(cache=cache, direct_peers=shared_cache.addresses,
                    stage=stage, output=output, signals=signals, rows=rows, maximum=maximum,
                    runtime=runtime, coordinator=coordinator, session=session, arena=arena, device=device)
                session.tune(show_progress=False)
                # Transport-only replay preserves every FP8 bit pattern, including
                # NaNs and signed zero. Attention is tested separately with finite bytes.
                transfer = compile_forward(operation.transfer) if compiled else operation.transfer
                transfer()
                torch.cuda.synchronize(device)
                transfer_graph = torch.cuda.CUDAGraph()
                with session.capture(), torch.cuda.graph(transfer_graph):
                    transfer()
                channels = torch.arange(WIDTH, dtype=torch.int32, device=device)
                physical = torch.arange(maximum // WORLD, dtype=torch.int32, device=device)
                raw = ((physical[:, None] * 7 + channels[None] * 3 + rank * 13) % 256).byte()
                for visible in (0, 1, 63, 64, 127, 128, 129, maximum, 129, 0):
                    cache.view(torch.uint8).view(-1, WIDTH).copy_(raw)
                    operation.end.fill_(visible)
                    output.view(torch.uint8).fill_(254)
                    stage.tensor.fill_(255)
                    dist.barrier()
                    before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                    transfer_graph.replay()
                    torch.cuda.synchronize(device)
                    assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == before
                    tokens = torch.arange(visible, dtype=torch.int64, device=device)
                    local = tokens // WORLD
                    slots = operation.table.long()[local // PAGE] * PAGE + local % PAGE
                    expected = ((slots[:, None] * 7 + channels[None] * 3 + tokens[:, None] % WORLD * 13) % 256).byte()
                    assert torch.equal(output.view(torch.uint8)[:visible], expected), (rank, visible)
                    assert (output.view(torch.uint8)[visible:] == 254).all()
                transfer_graph.reset()
                transfer_graph = None
                del transfer, raw, expected
                operation.end.zero_()
                call = compile_forward(operation) if compiled else operation
                call()
                torch.cuda.synchronize(device)
                graph = torch.cuda.CUDAGraph()
                with session.capture(), torch.cuda.graph(graph):
                    call()
                channels = torch.arange(WIDTH, dtype=torch.int32, device=device)
                physical = torch.arange(maximum // WORLD, dtype=torch.int32, device=device)
                seed = ((physical[:, None] * 7 + channels[None] * 3 + rank * 13) % 120).byte()
                columns = torch.arange(TOPK, dtype=torch.int32, device=device).flip(0)
                row_ids = torch.arange(rows, dtype=torch.int64, device=device)

                def inputs(visible, active, pattern):
                    cache.view(torch.uint8).view(-1, WIDTH).copy_((seed + pattern).remainder(127))
                    operation.end.fill_(visible)
                    start = max(visible - active, 0)
                    operation.positions.copy_(torch.where(row_ids < active, start + row_ids, -1))
                    ids = (columns[None] + row_ids[:, None] * 31).remainder(max(visible, 1)).int()
                    operation.selected.copy_(torch.where(columns[None] < min(visible, TOPK), ids, -1))
                    operation.current.fill_((rank + 1 + pattern) / 64)

                for visible, active in ((0, 0), (1, 1), (63, 1), (64, 1), (127, 1), (128, 1),
                                        (129, 1), (32768, rows - 7), (maximum, rows), (129, 1), (0, 0)):
                    inputs(visible, active, 0)
                    stage.tensor.fill_(255)
                    output.view(torch.uint8).fill_(255)
                    for record in operation.records:
                        record.view(torch.uint8).fill_(255)
                    dist.barrier()
                    before = torch.cuda.memory_stats(device)
                    graph.replay()
                    torch.cuda.synchronize(device)
                    after = torch.cuda.memory_stats(device)
                    for field in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                        assert after[field] == before[field], (rank, field)
                    assert operation.exact_records.all(), (rank, visible)
                    assert operation.exact_counts.all(), (rank, visible)
                    torch.testing.assert_close(operation.staged_output, operation.reference_output, rtol=0, atol=0)
                    assert not operation.staged_output[active:].count_nonzero()
                    assert (output.view(torch.uint8)[visible:] == 255).all()

                # Missing page entries stay invalid in both address spaces.
                operation.table[0] = -1
                inputs(129, 1, 0)
                dist.barrier()
                graph.replay()
                torch.cuda.synchronize(device)
                assert operation.exact_records.all() and operation.exact_counts.all()
                assert not operation.staged_output.count_nonzero()
                operation.table[0] = cache.shape[0] - 1

                # Queue changing owner bytes with one delayed rank; every replay
                # must finish peer reads before the next overwrites its stage.
                observations = torch.empty((12, 2), dtype=torch.bool, device=device)
                dist.barrier()
                if rank == WORLD - 1:
                    torch.cuda._sleep(2_000_000)
                for iteration in range(12):
                    inputs(32768 if iteration % 2 else 129, min(rows, 128), iteration % 3)
                    stage.tensor.fill_(255)
                    output.view(torch.uint8).fill_(255)
                    graph.replay()
                    observations[iteration, 0] = operation.exact_records.all()
                    observations[iteration, 1] = torch.eq(operation.staged_output, operation.reference_output).all()
                torch.cuda.synchronize(device)
                assert observations.all(), rank
                graph.reset()
                graph = None
            runtime.close()
            runtime = None
            stage.close()
            shared_cache.close()
            stage = shared_cache = None
    finally:
        if transfer_graph is not None:
            transfer_graph.reset()
        if graph is not None:
            graph.reset()
        if runtime is not None:
            runtime.close()
        if stage is not None:
            stage.close()
        if shared_cache is not None:
            shared_cache.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("rows,maximum,compiled", [(128, 65536, False), (4096, 1048576, False),
                                                  (4096, 1048576, True)],
                         ids=("eager128", "eager4096", "compiled4096"))
def test_staged_prefill_preserves_selected_records_and_bf16_attention(tmp_path, rows, maximum, compiled):
    import torch.multiprocessing as mp

    assert torch.cuda.device_count() == WORLD, "assign exactly sixteen CUDA devices"
    mp.spawn(_rank, args=((tmp_path / "gloo").as_uri(), rows, maximum, compiled), nprocs=WORLD, join=True)
