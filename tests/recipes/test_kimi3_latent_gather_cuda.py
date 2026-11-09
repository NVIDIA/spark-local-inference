"""TP16 bit-preserving latent/router exchange across changing prepared capacities."""

from datetime import timedelta
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_TP_CUDA_TESTS') != '1',
                              reason='requires sixteen explicitly assigned CUDA devices')


def fields(rows, rank, variation, device):
    latent = (torch.arange(rows * 224, device=device, dtype=torch.int32)
              + 257 * rank + 101 * variation).short().view(rows, 224).view(torch.bfloat16)
    router = (torch.arange(rows * 56, device=device, dtype=torch.int32)
              * 1048583 + 1069547520 + 193 * rank + variation).view(rows, 56).view(torch.float32)
    latent.view(torch.int16)[0, :4] = torch.tensor([0, -32768, 32704, -64], dtype=torch.int16, device=device)
    router.view(torch.int32)[0, :4] = torch.tensor([0, -2147483648, 2143289345, -4194303], dtype=torch.int32, device=device)
    return latent, router


def worker(rank, rendezvous):
    from ignition import runtime_patches
    runtime_patches.apply()
    import torch.distributed as dist
    from ignition.components.column_pair_gather import PreparedColumnPairGather
    from ignition.preparation.collectives import CollectivePreparation, create_head_exchange
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.core_storage import CoreStorage
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    dist.init_process_group('gloo', rank=rank, world_size=16, init_method=rendezvous,
                            timeout=timedelta(minutes=15))
    runtime, graphs = None, []
    try:
        with CoreStorage(rank=rank, world=16, device=rank) as storage:
            runtime = create_head_exchange(dist.group.WORLD, device, max_rows=1024, heads=16, head_dim=336,
                                           ipc=storage.peer_storage)
            coordinator = CollectivePreparation(dist.group.WORLD)
            with RecordedPreparationSession(device=device, checkpoint='latent-router-column-pairs',
                    workload_schema='tp16-prefill-packing', numerical_policy='exact-bits',
                    compile_workers=16, coordinator=coordinator) as session:
                buffers, cases, calls = {}, [], []
                for rows in (4096, 1025, 128, 16):
                    first, second = fields(rows, rank, 0, device)
                    output_first = torch.empty((rows, 3584), dtype=torch.bfloat16, device=device)
                    output_second = torch.empty((rows, 896), dtype=torch.float32, device=device)
                    operation = PreparedColumnPairGather(first, second, output_first, output_second,
                        runtime=runtime, session=session, coordinator=coordinator, name=f'rows{rows}', buffers=buffers)
                    cases.append((rows, first, second, output_first, output_second, operation))
                session.tune(show_progress=False)
                for index in (0, 1, 2, 3, 0, 3):
                    operation = cases[index][-1]
                    call = compile_forward(operation) if len(graphs) >= 4 else operation
                    calls.append(call)
                    call()
                    torch.cuda.synchronize(device)
                    graph = torch.cuda.CUDAGraph()
                    dist.barrier()
                    with session.capture(), torch.cuda.graph(graph):
                        call()
                    graphs.append(graph)
                assert len({case[-1].send.untyped_storage()._cdata for case in cases}) == 1
                assert len({case[-1].received.untyped_storage()._cdata for case in cases}) == 1
                assert sum(value.numel() * value.element_size() for value in buffers.values()) == 11698176
                case_indices = (0, 1, 2, 3, 0, 3)
                for graph_index in (0, 3, 1, 2, 4, 5, 4):
                    rows, first, second, output_first, output_second, operation = cases[case_indices[graph_index]]
                    for variation in range(3):
                        left, right = fields(rows, rank, variation, device)
                        active = rows if variation == 0 else max(1, rows - 7)
                        left[active:].zero_()
                        right[active:].zero_()
                        first.copy_(left)
                        second.copy_(right)
                        operation.send.fill_(torch.nan)
                        operation.received.fill_(torch.nan)
                        output_first.fill_(torch.nan)
                        output_second.fill_(torch.nan)
                        torch.cuda.synchronize(device)
                        dist.barrier()
                        before = torch.cuda.memory_stats(device)
                        graphs[graph_index].replay()
                        torch.cuda.synchronize(device)
                        after = torch.cuda.memory_stats(device)
                        for field in ('allocation.all.allocated', 'allocation.all.freed',
                                      'segment.all.allocated', 'segment.all.freed'):
                            assert before[field] == after[field], field
                        expected_first = torch.cat([fields(rows, peer, variation, device)[0] for peer in range(16)], dim=1)
                        expected_second = torch.cat([fields(rows, peer, variation, device)[1] for peer in range(16)], dim=1)
                        expected_first[active:].zero_()
                        expected_second[active:].zero_()
                        assert torch.equal(output_first.view(torch.int16), expected_first.view(torch.int16))
                        assert torch.equal(output_second.view(torch.int32), expected_second.view(torch.int32))
                        assert torch.equal(first.view(torch.int16), left.view(torch.int16))
                        assert torch.equal(second.view(torch.int32), right.view(torch.int32))
                rows, first, second, output_first, output_second, operation = cases[0]
                snapshots_first = torch.empty((6, rows, 3584), dtype=torch.bfloat16, device=device)
                snapshots_second = torch.empty((6, rows, 896), dtype=torch.float32, device=device)
                patterns = [fields(rows, rank, variation, device) for variation in range(3)]
                dist.barrier()
                if rank == 15:
                    torch.cuda._sleep(2000000)
                for iteration in range(6):
                    first.copy_(patterns[iteration % 3][0])
                    second.copy_(patterns[iteration % 3][1])
                    operation.send.fill_(torch.nan)
                    operation.received.fill_(torch.nan)
                    graphs[4].replay()
                    snapshots_first[iteration].copy_(output_first)
                    snapshots_second[iteration].copy_(output_second)
                torch.cuda.synchronize(device)
                for iteration in range(6):
                    expected_first = torch.cat([fields(rows, peer, iteration % 3, device)[0] for peer in range(16)], dim=1)
                    expected_second = torch.cat([fields(rows, peer, iteration % 3, device)[1] for peer in range(16)], dim=1)
                    assert torch.equal(snapshots_first[iteration].view(torch.int16), expected_first.view(torch.int16))
                    assert torch.equal(snapshots_second[iteration].view(torch.int32), expected_second.view(torch.int32))
                for graph in graphs:
                    graph.reset()
                graphs.clear()
            runtime.close()
            runtime = None
    finally:
        for graph in graphs:
            graph.reset()
        if runtime is not None:
            runtime.close()
        dist.destroy_process_group()


def test_latent_router_prefill_gather(tmp_path):
    import torch.multiprocessing as mp

    assert torch.cuda.device_count() == 16, 'assign exactly sixteen CUDA devices'
    mp.spawn(worker, args=((tmp_path / 'gloo').as_uri(),), nprocs=16, join=True)
