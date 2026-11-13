# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compiled diagnostic startup producers complete on every rank before requests."""

from datetime import timedelta
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_TP_CUDA_TESTS') != '1',
                               reason='requires explicitly assigned CUDA devices')


def _rank(rank, world, rendezvous):
    from ignition.direct.executor import StepRunner
    from ignition.preparation.collectives import CollectivePreparation
    from ignition.preparation.execution import GraphSignature, TuningGraph

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(minutes=2))
    graphs = []
    try:
        coordinator = CollectivePreparation(dist.group.WORLD)
        selector = torch.zeros(1, dtype=torch.int32, device=device)
        status, count, failure, output, frame = [torch.zeros_like(selector) for _ in range(5)]
        sequence = torch.zeros(1, dtype=torch.int64, device=device)

        def produce(selector, status, count, failure):
            selector.fill_(17)
            count.add_(1)
            status.copy_(failure)

        def consume(selector, output, frame, sequence):
            output.copy_(selector + frame + sequence.to(torch.int32))

        producer = torch.compile(produce, fullgraph=True)
        consumer = torch.compile(consume, fullgraph=True)
        producer(selector, status, count, failure)
        consumer(selector, output, frame, sequence)
        torch.cuda.synchronize(device)
        stream = torch.cuda.Stream(device=device)
        for eager in (False, True):
            for fail_peer in (False, True):
                stream.wait_stream(torch.cuda.current_stream(device))
                tuning = torch.cuda.CUDAGraph()
                with torch.cuda.graph(tuning, stream=stream):
                    producer(selector, status, count, failure)
                serving = torch.cuda.CUDAGraph()
                with torch.cuda.graph(serving, stream=stream):
                    consumer(selector, output, frame, sequence)
                graphs.extend((tuning, serving))
                torch.cuda.synchronize(device)
                selector.fill_(-100)
                status.fill_(99)
                count.zero_()
                failure.fill_(7 if fail_peer and rank == world - 1 else 0)
                output.fill_(-100)
                frame.zero_()
                sequence.zero_()
                buffers = {'selector': selector, 'count': count, 'status': status,
                           'frame_index': frame, 'submission_sequence': sequence}
                manifest = {
                    'buffers': [dict(name=name, address=value.data_ptr(), nbytes=value.nbytes,
                                     dtype=str(value.dtype).removeprefix('torch.'),
                                     shape=list(value.shape), strides=list(value.stride()))
                                for name, value in buffers.items()],
                    'graphs': [{'name': 'decode', 'tuned_bindings': [
                        {'buffer': 'selector', 'tuning_graph': 'tune'}]}],
                    'tuning_graphs': [{'name': 'tune', 'outputs': ['selector', 'count'],
                                      'status_buffer': 'status'}],
                }
                model = SimpleNamespace(device=rank, resources=(),
                    session=SimpleNamespace(_coordinator=coordinator), manifest=lambda: manifest,
                    buffers=buffers,
                    signatures={'decode': GraphSignature('decode', 1, 1, ())},
                    tuning_graphs={'tune': TuningGraph(tuning, ('selector', 'count'), 'status')},
                    graphs={'decode': serving, 'tune': tuning},
                    forwards={'decode': lambda: consumer(selector, output, frame, sequence)})
                try:
                    runner = StepRunner(model, eager=eager)
                except RuntimeError as error:
                    assert fail_peer and 'failed: local status' in str(error)
                    assert output.item() == -100
                    ready = False
                else:
                    assert not fail_peer
                    assert selector.item() == 17 and count.item() == 1
                    runner.run({'graph': 'decode', 'uploads': [], 'frame': 1, 'sequence': 3})
                    assert output.item() == 21
                    runner.close()
                    ready = True
                states = [None] * world
                dist.all_gather_object(states, ready)
                assert states == [not fail_peer] * world
                assert count.item() == 1
                torch.cuda.synchronize(device)
                for graph in graphs:
                    graph.reset()
                graphs.clear()
    finally:
        torch.cuda.synchronize(device)
        for graph in graphs:
            graph.reset()
        dist.destroy_process_group()


@pytest.mark.parametrize('world', [2, 4])
def test_direct_startup_compiled_graphs_and_peer_failure(tmp_path, world):
    assert os.environ.get('CUDA_VISIBLE_DEVICES'), 'assign CUDA devices explicitly'
    assert torch.cuda.device_count() >= world
    mp.spawn(_rank, args=(world, (tmp_path / f'tp{world}').as_uri()), nprocs=world, join=True)
