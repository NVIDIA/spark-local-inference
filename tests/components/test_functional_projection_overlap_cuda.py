# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Projection-chain temporaries remain live through auxiliary-stream joins."""

import os

import pytest
import torch

from ignition.components.linear import UnquantizedProjection
from ignition.components.projection_overlap import ConcurrentProjectionChain
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an assigned CUDA device")


@pytest.mark.parametrize("rows", [3, 129])
@torch.no_grad()
def test_functional_projection_chain_replay_preserves_local_input(rows):
    device = torch.device("cuda:0")
    torch.manual_seed(321)
    seed = torch.randn(rows, 128, dtype=torch.bfloat16, device=device)
    template = torch.empty_like(seed)
    weights = [torch.randn(128, 128, dtype=torch.bfloat16, device=device) * .1 for _ in range(3)]
    expected_side, expected_main = torch.empty_like(seed), torch.empty_like(seed)
    side, main_output = torch.empty_like(seed), torch.empty_like(seed)
    arena = ScratchArena(device)
    with RecordedPreparationSession(device=device, checkpoint="projection-chain",
            workload_schema="projection-chain", numerical_policy="bf16") as session:
        session.register_scratch(arena)
        first = UnquantizedProjection(template, weights[0], torch.empty_like(seed), session=session, name="first")
        second = UnquantizedProjection(first.output, weights[1], torch.empty_like(seed), session=session, name="second")
        main = UnquantizedProjection(template, weights[2], torch.empty_like(seed), session=session, name="main")
        chain = ConcurrentProjectionChain((first, second), parallel_plans=(main.plan,), session=session, workspace=arena)
        chain.functional(session=session)
        main.functional(session=session)
        session.tune()
        arena.release_preparation()
        assert chain.projections is chain.workspace is chain.arena is None

        def forward():
            source = seed.clone()
            pending = chain.start(source)
            dependency = main(source)
            result = chain.finish(pending, dependency)
            side.copy_(result)
            main_output.copy_(dependency)
            source.fill_(float("nan"))

        compiled = compile_forward(forward)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        try:
            for gain in (1., -.5, 2.):
                seed.mul_(gain)
                expected_side.copy_(second(first(seed)))
                expected_main.copy_(main(seed))
                side.fill_(float("nan"))
                main_output.fill_(float("nan"))
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                for field in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                    assert after[field] == before[field]
                torch.testing.assert_close(side, expected_side, rtol=0, atol=0)
                torch.testing.assert_close(main_output, expected_main, rtol=0, atol=0)
        finally:
            graph.reset()
