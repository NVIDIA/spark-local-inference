# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM query overlap captures alongside a shared RoCEnante transport.

Run the test on two assigned Spark hosts with FLARE_RUN_ROCE_TESTS=1,
FLARE_TEST_RANK=0 or 1, and the same FLARE_TEST_RENDEZVOUS=tcp://ADDRESS:PORT.
"""

from datetime import timedelta
import os
import socket
from urllib.parse import urlsplit

import pytest
import torch
import torch.distributed as dist

from ignition.components.collectives import PreparedAllGather, PreparedAllReduce
from ignition.components.linear import BatchedQuantizedProjection, QuantizedProjection
from ignition.components.weights import quantize_bf16_linear_as_mxfp8
from ignition.preparation.collectives import CollectivePreparation, create_roce_runtime
from ignition.preparation.compilation import compile_forward
from ignition.preparation.host_ranks import _use_interface
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.glm53.mla import QueryChain

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_ROCE_TESTS") != "1",
                               reason="requires two assigned Spark hosts")


@torch.no_grad()
def test_query_chain_with_main_stream_collectives():
    rank = int(os.environ["FLARE_TEST_RANK"])
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    rendezvous = os.environ["FLARE_TEST_RENDEZVOUS"]
    address = urlsplit(rendezvous)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
        route.connect((address.hostname, address.port))
        _use_interface(route.getsockname()[0])
    dist.init_process_group("gloo", init_method=rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(minutes=5))
    coordinator = CollectivePreparation(dist.group.WORLD)
    assert coordinator.spans_hosts
    runtime = create_roce_runtime(dist.group.WORLD, device, max_tokens=96, pipe_tokens=96,
                                 hidden_size=2048, gather_bytes=96 * 32 * 512 * 2)
    graphs = []
    try:
        torch.manual_seed(171 + rank)
        weight = quantize_bf16_linear_as_mxfp8(
            torch.randn((32 * 256, 2048), dtype=torch.bfloat16, device=device) * .01)
        # Exact, finite MXFP8 values and unit scales exercise the native absorption BMM.
        values = torch.randint(-4, 5, (32, 256, 512), device=device).to(torch.float8_e4m3fn)
        scales = torch.full((32, 256, 16), 127, dtype=torch.uint8, device=device)
        arena = ScratchArena(device)
        cases = []
        with RecordedPreparationSession(device=device, checkpoint="glm-query-chain-roce",
                workload_schema="query-overlap-v1", numerical_policy="default",
                coordinator=coordinator, compile_workers=4) as session:
            session.register_scratch(arena)
            for rows in (1, 16, 96):
                source = torch.zeros((rows, 2048), dtype=torch.bfloat16, device=device)
                query = torch.empty((rows, 32, 256), dtype=torch.bfloat16, device=device)
                absorbed = torch.empty((rows, 32, 512), dtype=torch.bfloat16, device=device)
                gathered = torch.empty((rows, 64 * 512), dtype=torch.bfloat16, device=device)
                main_output = torch.empty_like(query)
                reduced = torch.empty_like(source)
                projection = QuantizedProjection(source, weight, query.view(rows, -1), session=session,
                    name=f"query{rows}", workspace=arena, activation_mode="auto")
                main_projection = QuantizedProjection(source, weight, main_output.view(rows, -1),
                    session=session, name=f"main{rows}", workspace=arena, activation_mode="auto")
                absorb = BatchedQuantizedProjection(query.transpose(0, 1), (values, scales),
                    absorbed.transpose(0, 1), major="n", session=session, name=f"absorb{rows}")
                gather = PreparedAllGather(absorbed.view(rows, -1), gathered, runtime=runtime,
                    session=session, coordinator=coordinator, name=f"gather{rows}")
                reduce = PreparedAllReduce(source, reduced, runtime=runtime, session=session,
                    coordinator=coordinator, name=f"reduce{rows}")
                chain = QueryChain(projection, absorb, source, query, absorbed, gather, gathered,
                    session=session, workspace=arena, parallel_plans=(main_projection.plan,))

                chain.functional(session=session)
                main_projection.functional(session=session)
                reduce.functional(session=session)

                def forward(chain=chain, reduce=reduce, main_projection=main_projection, source=source,
                            query=query, absorbed=absorbed, gathered=gathered, main_output=main_output, reduced=reduced):
                    reduced.copy_(reduce(source))
                    local_query = torch.empty_like(query)
                    local_absorbed = torch.empty_like(absorbed)
                    local_gathered = torch.empty_like(gathered)
                    pending = chain.start(source, local_query, local_absorbed, local_gathered)
                    main = main_projection(source)
                    reduced.copy_(reduce(source))
                    result = chain.finish(pending, main)
                    query.copy_(local_query)
                    absorbed.copy_(local_absorbed)
                    gathered.copy_(result)
                    main_output.copy_(main)

                cases.append((rows, source, query, absorbed, gathered, main_output, reduced,
                              projection, absorb, forward))
            session.tune(show_progress=False)
            arena.release_preparation()
            assert arena.auxiliary_stream != torch.cuda.current_stream(device)
            pool = torch.cuda.graph_pool_handle()
            for case in cases:
                rows, *_, forward = case
                for compiled in (False, True):
                    function = compile_forward(forward) if compiled else forward
                    stream = torch.cuda.Stream(device=device)
                    stream.wait_stream(torch.cuda.current_stream(device))
                    with torch.cuda.stream(stream):
                        function()
                    torch.cuda.synchronize(device)
                    graph = torch.cuda.CUDAGraph()
                    with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(
                            graph, pool=pool, stream=stream):
                        function()
                    graphs.append((case, compiled, graph, function))
            session.freeze()

            # Alternate capacities and compiled/uncompiled graphs, overwriting all outputs.
            for iteration in range(3):
                for case, compiled, graph, _ in (graphs if iteration % 2 == 0 else reversed(graphs)):
                    rows, source, query, absorbed, gathered, main_output, reduced, projection, absorb, _ = case
                    source.normal_(std=.1)
                    if rows > 1:
                        source[-1].zero_()
                    query.copy_(projection(source).view_as(query))
                    absorbed.copy_(absorb(query.transpose(0, 1)).transpose(0, 1))
                    expected_query = query.clone()
                    local = absorbed.cpu().reshape(rows, -1)
                    parts = [torch.empty_like(local) for _ in range(2)]
                    dist.all_gather(parts, local)
                    expected_gathered = torch.cat(parts, dim=-1).to(device)
                    expected_reduced = source.cpu()
                    dist.all_reduce(expected_reduced)
                    for output in (query, absorbed, gathered, main_output, reduced):
                        output.fill_(float("nan"))
                    torch.cuda.synchronize(device)
                    dist.barrier()
                    before = torch.cuda.memory_stats(device)
                    graph.replay()
                    torch.cuda.synchronize(device)
                    after = torch.cuda.memory_stats(device)
                    runtime.check_health()
                    for counter in ("allocation.all.allocated", "allocation.all.freed",
                                    "segment.all.allocated", "segment.all.freed"):
                        assert before[counter] == after[counter], (rows, compiled, counter)
                    torch.testing.assert_close(query, expected_query, rtol=0, atol=0)
                    torch.testing.assert_close(main_output, expected_query, rtol=0, atol=0)
                    torch.testing.assert_close(gathered, expected_gathered, rtol=0, atol=0)
                    torch.testing.assert_close(reduced.cpu(), expected_reduced, rtol=0, atol=0)
    finally:
        for _, _, graph, _ in graphs:
            graph.reset()
        runtime.close()
        dist.destroy_process_group()
