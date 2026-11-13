# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Block proposal synchronization reconstructs exact conditional confidence inputs."""

import os
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("batch,depth", [(1, 1), (1, 7), (8, 7), (32, 7)])
@torch.no_grad()
def test_canonical_predecessors_replace_divergent_local_chain_under_native_replay(batch, depth):
    from ignition.components.split_draft import PreparedDraftConfidence
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.graphs import PreparedGraphBundle
    from ignition.preparation.tuning import RecordedPreparationSession

    device = torch.device("cuda", 0)
    generator = torch.Generator().manual_seed(9824 + batch + depth)
    table = torch.randn((257, 32), generator=generator).to(device, torch.bfloat16)
    hidden = torch.randn((batch * depth, 128), generator=generator).to(device, torch.bfloat16)
    weight = torch.randn((1, 160), generator=generator).to(device)
    pending = torch.arange(batch, device=device, dtype=torch.int64)
    drafts = torch.zeros((batch, depth), dtype=torch.int64, device=device)
    active = torch.ones(batch, dtype=torch.bool, device=device)
    candidates = torch.zeros((batch, depth + 1), dtype=torch.int64, device=device)
    candidates[:, 0].copy_(pending)
    outputs = [torch.empty((batch, depth), device=device) for _ in range(2)]
    graphs, worker = {}, None
    with RecordedPreparationSession(device=device, checkpoint="canonical-proposal-confidence",
            workload_schema="canonical-proposal-confidence-v1", numerical_policy="default") as session:
        owner = SimpleNamespace(block_size=depth, logit_offset=0, session=session)
        source = SimpleNamespace(owner=owner, batch=batch, width=depth + 1,
            draft=SimpleNamespace(head_hidden=hidden),
            markov=SimpleNamespace(embedding=torch.empty((batch, 32), device=device, dtype=torch.bfloat16),
                                  embed=SimpleNamespace(weight=table)),
            candidates=candidates, drafts=drafts, active=active, proposal_block=None)
        serial = PreparedDraftConfidence(source, weight, outputs[0], name="serial")
        source.proposal_block = object()
        block = PreparedDraftConfidence(source, weight, outputs[1], name="block")
        session.tune(show_progress=False)

        def serial_forward():
            serial.record_embedding(0, table[torch.where(active, candidates[:, 0], 0)])
            for position in range(1, depth):
                serial.record_embedding(position, table[drafts[:, position - 1]])
            serial()

        calls = (compile_forward(serial_forward), compile_forward(block))
        for index, call in enumerate(calls):
            call()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                call()
            graphs[str(index)] = graph
        session.freeze()
        bundle = PreparedGraphBundle(device=0, graphs=graphs, buffers={},
            resources=(session, serial, block, source, table, hidden, weight, calls))
        worker = bundle.start_worker()
        try:
            for round_id in range(12):
                candidates[:, 0].copy_((pending + 17 * round_id) % 257)
                drafts.copy_((torch.arange(batch * depth, device=device).view(batch, depth) + 11 * round_id) % 257)
                active.copy_((pending + round_id) % 3 != 0)
                serial.embeddings.fill_(float("nan"))
                block.embeddings.fill_(float("nan"))
                block.preceding.fill_(256)
                torch.cuda.synchronize()
                allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
                for index in range(2):
                    ticket = worker.submit(index)
                    completion = worker.poll(10000)
                    assert completion is not None and completion.ticket == ticket and completion.error is None
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
                assert torch.equal(serial.embeddings, block.embeddings)
                assert torch.equal(serial.features, block.features)
                assert torch.equal(serial.logits.view(torch.int32), block.logits.view(torch.int32))
                assert torch.equal(outputs[0].view(torch.int32), outputs[1].view(torch.int32))
        finally:
            worker.close()
            for graph in graphs.values():
                graph.reset()
