# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tune shared PCIe declarations, reload selections, and replay their serving calls."""

import os
from pathlib import Path

import pytest


def _worker(rank, directory, head_major):
    import torch
    import torch.distributed as dist
    from b12x.comm.pcie import AllReduce, DcpAllToAll
    from ignition.components.collectives import (
        PreparedAllReduce, PreparedAllReduceRMSNorm, PreparedAllGather, PreparedLseReduceScatter,
    )
    from ignition.preparation.collectives import CollectivePreparation
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.set_num_threads(1)
    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)
    dist.init_process_group('gloo', init_method=(Path(directory) / 'group').as_uri(), rank=rank, world_size=4)
    coordinator = CollectivePreparation(dist.group.WORLD)
    manager = AllReduce.from_exchange_group(exchange_group=dist.group.WORLD, device=device,
        single_channel=True, max_size=2 << 20, eager_buffer_bytes=2 << 20)
    manager.prepare_channels(('tuning',))
    dcp = DcpAllToAll.from_exchange_group(exchange_group=dist.group.WORLD, device=device,
        max_batch_size=8, total_heads=16, head_dim=512, query_head_dim=512, stream_affine=False)
    for repetition in range(2):
        session = RecordedPreparationSession(device=device, checkpoint='collective-operator-test',
            workload_schema='pcie-collectives', numerical_policy='declared-dtypes', coordinator=coordinator,
            compile_workers=4, autotune_effort='default')
        session.cache_dir = Path(directory) / 'cache'
        operations = []
        for index, rows in enumerate((1, 1, 8, 128)):
            source = torch.ones((rows, 5120), dtype=torch.bfloat16, device=device)
            output = torch.empty_like(source)
            operation = PreparedAllReduce(source, output, runtime=manager, session=session,
                coordinator=coordinator, name=f'reduction-{index}')
            operations.append((operation, (source,), output, 4.0))
        source = torch.ones((8, 5120), dtype=torch.bfloat16, device=device)
        inplace = PreparedAllReduce(source, source, runtime=manager, session=session,
            coordinator=coordinator, name='inplace')
        operations.append((inplace, (source,), source, 4.0))
        source = torch.ones((8, 5120), dtype=torch.bfloat16, device=device)
        residual = torch.zeros_like(source, dtype=torch.float32)
        output = torch.empty_like(source)
        weight = torch.ones(5120, dtype=torch.bfloat16, device=device)
        fused = PreparedAllReduceRMSNorm(source, residual, weight, output, eps=1e-6,
            runtime=manager, session=session, coordinator=coordinator, name='fused')
        operations.append((fused, (source, residual), output, 1.0))
        source = torch.ones((8, 4 * 512), dtype=torch.bfloat16, device=device)
        output = torch.empty((8, 16 * 512), dtype=torch.bfloat16, device=device)
        gather = PreparedAllGather(source, output, runtime=dcp, session=session,
                                   coordinator=coordinator, name='gather')
        operations.append((gather, (source,), output, 1.0))
        source = (torch.ones((16, 12, 512), dtype=torch.bfloat16, device=device)[:, :8].transpose(0, 1)
                  if head_major else torch.ones((8, 16, 512), dtype=torch.bfloat16, device=device))
        lse = torch.zeros((8, 16), device=device)
        output = (torch.empty((4, 12, 512), dtype=torch.bfloat16, device=device)[:, :8].transpose(0, 1)
                  if head_major else torch.empty((8, 4, 512), dtype=torch.bfloat16, device=device))
        reduce = PreparedLseReduceScatter(source, lse, output, runtime=dcp, session=session,
                                          coordinator=coordinator, name='lse')
        operations.append((reduce, (source, lse), output, 1.0))
        session.tune()
        assert operations[0][0].plan.selection.config == operations[1][0].plan.selection.config
        assert all(op.plan.selection.source == ('tuned' if repetition == 0 else 'cached') for op, *_ in operations)
        chosen = [op.plan.contract.config_payload(op.plan.selection.config).to_dict() for op, *_ in operations]
        peers = [None] * 4
        dist.all_gather_object(peers, chosen)
        assert all(value == chosen for value in peers)
        graphs = []
        for operation, arguments, output, expected in operations:
            graph = torch.cuda.CUDAGraph()
            with session.capture(), manager.capture(), torch.cuda.graph(graph):
                operation(*arguments)
            graphs.append(graph)
        session.freeze()
        for step in range(6):
            indices = range(len(graphs)) if step % 2 else reversed(range(len(graphs)))
            for index in indices:
                operation, arguments, output, expected = operations[index]
                arguments[0].fill_(1)
                if operation is fused:
                    arguments[1].zero_()
                if operation is reduce:
                    arguments[1].zero_()
                before = torch.cuda.memory_stats()
                graphs[index].replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                for key in ('allocation.all.allocated', 'allocation.all.freed', 'segment.all.allocated', 'segment.all.freed'):
                    assert before[key] == after[key], key
                torch.testing.assert_close(output, torch.full_like(output, expected), rtol=0, atol=0)
        for graph in graphs:
            graph.reset()
        if rank == 0:
            print('collective selections', repetition, chosen, flush=True)
        session.close()
    dcp.close()
    manager.close()
    dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1', reason='requires four peer-accessible CUDA devices')
@pytest.mark.parametrize('head_major', (False, True))
def test_collective_tuning_reuses_equivalent_declarations_and_cached_winners(tmp_path, head_major):
    import torch
    import torch.multiprocessing as mp

    if torch.cuda.device_count() < 4:
        pytest.skip('requires four CUDA devices')
    mp.spawn(_worker, args=(str(tmp_path), head_major), nprocs=4, join=True)
