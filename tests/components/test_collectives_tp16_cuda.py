# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TP16 reductions and column gathers through core-owned CUDA memory."""
from datetime import timedelta
import os
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(os.getenv('FLARE_RUN_TP_CUDA_TESTS') != '1', reason='requires sixteen assigned GPUs')


def _rank(rank, rendezvous):
    from ignition.preparation.core_storage import CoreStorage
    from ignition.preparation.collectives import CollectivePreparation, create_all_reduce_runtime, create_all_gather_runtime
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.components.collectives import PreparedAllReduce, PreparedAllGather

    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=16, timeout=timedelta(minutes=5))
    group = dist.group.WORLD
    with CoreStorage(rank=rank, world=16, device=rank) as storage:
        coordinator = CollectivePreparation(group)
        reduce = create_all_reduce_runtime(group, device, max_tokens=4096, hidden_size=7168, ipc=storage.peer_storage)
        gather = create_all_gather_runtime(group, device, max_rows=4096, columns=7168, ipc=storage.peer_storage)
        router_gather = create_all_gather_runtime(group, device, max_rows=4096, columns=2048, ipc=storage.peer_storage)
        with RecordedPreparationSession(device=device, checkpoint='tp16-collectives', workload_schema='tp16', numerical_policy='default', coordinator=coordinator) as session:
            operations = []
            routers = []
            for rows in (1,4,128,4096):
                source = torch.full((rows,7168), rank+1, device=device, dtype=torch.bfloat16)
                output = torch.empty_like(source)
                local = torch.full((rows,448), rank, device=device, dtype=torch.bfloat16)
                gathered = torch.empty_like(source)
                reduction = PreparedAllReduce(source, output, runtime=reduce, session=session, coordinator=coordinator, name=f'reduce{rows}')
                gathering = PreparedAllGather(local, gathered, runtime=gather, session=session, coordinator=coordinator, name=f'gather{rows}')
                operations.append((source,output,local,gathered,reduction,gathering))
                logits = (torch.arange(rows * 64, device=device, dtype=torch.float32).view(rows,64) + rank) / 17
                router_logits = torch.empty((rows,1024),device=device,dtype=torch.float32)
                route_gather = PreparedAllGather(logits.view(torch.bfloat16),router_logits.view(torch.bfloat16),
                    runtime=router_gather,session=session,coordinator=coordinator,name=f'router{rows}')
                routers.append((logits,router_logits,route_gather))
            session.tune(show_progress=False)
            stream = torch.cuda.Stream(device=device)
            for source,output,local,gathered,reduction,gathering in operations:
                reduction(source);gathering(local)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                stream.wait_stream(torch.cuda.current_stream())
                with reduce.capture(stream=stream), session.capture(), torch.cuda.graph(graph, stream=stream):
                    reduction(source);gathering(local)
                dist.barrier()
                graph.replay();torch.cuda.synchronize()
                torch.testing.assert_close(output,torch.full_like(output,136),atol=0,rtol=0)
                expected = torch.arange(16,device=device,dtype=torch.bfloat16).repeat_interleave(448).expand_as(gathered)
                torch.testing.assert_close(gathered,expected,atol=0,rtol=0)
                graph.reset()
            for local, gathered, gathering in routers:
                gathering(local.view(torch.bfloat16))
                graph = torch.cuda.CUDAGraph()
                stream.wait_stream(torch.cuda.current_stream())
                with session.capture(), torch.cuda.graph(graph, stream=stream):
                    gathering(local.view(torch.bfloat16))
                dist.barrier()
                graph.replay(); torch.cuda.synchronize()
                base = torch.arange(local.numel(),device=device,dtype=torch.float32).view_as(local)
                expected = torch.cat([(base + peer) / 17 for peer in range(16)],dim=-1)
                torch.testing.assert_close(gathered,expected,atol=0,rtol=0)
                graph.reset()
            session.freeze()
        router_gather.close();gather.close();reduce.close()
    dist.destroy_process_group()


def test_core_owned_tp16_collectives(tmp_path):
    assert torch.cuda.device_count() == 16
    mp.spawn(_rank, args=((tmp_path / 'rendezvous').as_uri(),), nprocs=16, join=True)
