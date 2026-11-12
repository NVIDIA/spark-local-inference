# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""RoCEnante reduce-scatter and shard gather reproduce the two-shot all-reduce in captured graphs.

Run the test on two or four assigned Spark hosts with FLARE_RUN_ROCE_TESTS=1, FLARE_TEST_WORLD=2
or 4, FLARE_TEST_RANK=0 through FLARE_TEST_WORLD - 1, and the same FLARE_TEST_RENDEZVOUS=tcp://ADDRESS:PORT
on every host. Each rank reduce-scatters a chunk's partial sums to its owned rows and gathers every
rank's owned rows back; through compiled forwards and native graph replay at 2048 and 8192 rows of
Inkling 2 Small's 4608-wide hidden state, the owned rows equal the matching rows of the replicated
two-shot all-reduce bitwise, the gather equals the whole all-reduce, and replay allocates nothing.
"""

from datetime import timedelta
import os
import socket
from urllib.parse import urlsplit

import pytest
import torch
import torch.distributed as dist

from ignition.components.collectives import PreparedAllReduce, PreparedReduceScatter, PreparedShardGather
from ignition.preparation.collectives import CollectivePreparation, create_roce_runtime
from ignition.preparation.compilation import compile_forward
from ignition.preparation.host_ranks import _use_interface
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_ROCE_TESTS") != "1",
                               reason="requires two or four assigned Spark hosts")

HIDDEN = 4608
CAPACITIES = (8192, 2048)


@torch.no_grad()
def test_reduce_scatter_and_shard_gather_match_two_shot_all_reduce():
    rank, world = int(os.environ["FLARE_TEST_RANK"]), int(os.environ.get("FLARE_TEST_WORLD", "2"))
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    rendezvous = os.environ["FLARE_TEST_RENDEZVOUS"]
    address = urlsplit(rendezvous)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
        route.connect((address.hostname, address.port))
        _use_interface(route.getsockname()[0])
    dist.init_process_group("gloo", init_method=rendezvous,
                            rank=rank, world_size=world, timeout=timedelta(minutes=5))
    coordinator = CollectivePreparation(dist.group.WORLD)
    assert coordinator.spans_hosts
    runtime = create_roce_runtime(dist.group.WORLD, device, max_tokens=64, pipe_tokens=max(CAPACITIES),
                                 hidden_size=HIDDEN, gather_bytes=64 * HIDDEN * 2)
    graphs, cases = [], []
    try:
        with RecordedPreparationSession(device=device, checkpoint="synthetic-shard-collectives-roce",
                workload_schema="collectives.token-shards.v1", numerical_policy="bf16",
                coordinator=coordinator, compile_workers=2) as session:
            for tokens in CAPACITIES:
                rows = tokens // world
                partial = torch.zeros((tokens, HIDDEN), dtype=torch.bfloat16, device=device)
                assert PreparedReduceScatter.fits(runtime, partial)
                reduced_all = torch.empty_like(partial)
                scattered = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device)
                assert PreparedShardGather.fits(runtime, scattered)
                gathered = torch.empty_like(partial)
                reduce = PreparedAllReduce(partial, reduced_all, runtime=runtime, session=session,
                                           coordinator=coordinator, name=f"reduce{tokens}")
                scatter = PreparedReduceScatter(partial, scattered, runtime=runtime, session=session,
                                                coordinator=coordinator, name=f"scatter{tokens}")
                gather = PreparedShardGather(scattered, gathered, runtime=runtime, session=session,
                                             coordinator=coordinator, name=f"gather{tokens}")

                def forward(partial=partial, reduce=reduce, scatter=scatter, gather=gather, scattered=scattered):
                    reduce(partial)
                    scatter(partial)
                    gather(scattered)

                cases.append(dict(tokens=tokens, rows=rows, partial=partial, reduced_all=reduced_all,
                                  scattered=scattered, gathered=gathered, forward=forward))
            session.tune(show_progress=False)
            pool = torch.cuda.graph_pool_handle()
            for case in cases:
                compiled = compile_forward(case["forward"])
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    compiled()
                torch.cuda.synchronize(device)
                graph = torch.cuda.CUDAGraph()
                with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(graph, pool=pool, stream=stream):
                    compiled()
                graphs.append((case, graph, compiled))
            session.freeze()

            for iteration in range(3):
                for case, graph, _ in (graphs if iteration % 2 == 0 else reversed(graphs)):
                    tokens, rows = case["tokens"], case["rows"]
                    generator = torch.Generator(device=device).manual_seed(7 * iteration + rank + tokens)
                    case["partial"].normal_(std=.15, generator=generator)
                    for value in (case["reduced_all"], case["scattered"], case["gathered"]):
                        value.fill_(float("nan"))
                    torch.cuda.synchronize(device)
                    dist.barrier()
                    before = torch.cuda.memory_stats(device)
                    graph.replay()
                    torch.cuda.synchronize(device)
                    after = torch.cuda.memory_stats(device)
                    runtime.check_health()
                    for counter in ("allocation.all.allocated", "allocation.all.freed",
                                    "segment.all.allocated", "segment.all.freed"):
                        assert before[counter] == after[counter], (tokens, counter)
                    expected_sum = case["partial"].float().cpu()
                    dist.all_reduce(expected_sum)
                    torch.testing.assert_close(case["reduced_all"].float().cpu(), expected_sum,
                                               rtol=1e-2, atol=2e-2 * world)
                    assert torch.equal(case["scattered"], case["reduced_all"][rank * rows:(rank + 1) * rows])
                    assert torch.equal(case["gathered"], case["reduced_all"])
                    peers = [None] * world
                    dist.all_gather_object(peers, case["gathered"].cpu())
                    for peer in peers:
                        assert torch.equal(peer, peers[0])
    finally:
        for _, graph, _ in graphs:
            graph.reset()
        runtime.close()
        dist.destroy_process_group()
