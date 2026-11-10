# SPDX-License-Identifier: Apache-2.0

"""Graph replay preserves residuals when consumed embedding storage holds the prefix."""

import os
import pytest
import torch
from ignition.recipes.kimi3.residual import ADD, REPLACE, ResidualStream

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an assigned CUDA device")

@pytest.mark.parametrize("rows", [1, 8, 32, 4096])
@torch.no_grad()
def test_running_prefix_may_reuse_consumed_embedding(rows):
    """The initial REPLACE consumes embedding before subsequent prefix updates."""
    from ignition.preparation.compilation import compile_forward

    rng = torch.Generator(device="cuda").manual_seed(1807)
    shape = (rows, 7168)
    embedding = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=rng)
    delta = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=rng)
    score = torch.randn(7168, device="cuda", generator=rng) * .01
    norm = torch.randn(7168, device="cuda", dtype=torch.bfloat16, generator=rng)
    streams = [ResidualStream(torch.empty_like(embedding),
        torch.empty((rows, 8, 7168), device="cuda", dtype=torch.bfloat16),
        torch.empty_like(embedding)) for _ in range(2)]

    def run():
        streams[1].prefix.copy_(embedding)
        for index, stream in enumerate(streams):
            first = embedding if index == 0 else stream.prefix
            stream.mix(first, REPLACE, sources=0, write=0, score=score, norm=norm, eps=1e-5, output_eps=1e-5)
            stream.mix(delta, ADD, sources=1, write=-1, score=score, norm=norm, eps=1e-5, output_eps=1e-5)
            stream.mix(delta, ADD, sources=1, write=1, score=score, norm=norm, eps=1e-5, output_eps=1e-5)

    forward = compile_forward(run)
    forward()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
    for multiplier in (1., -2., .01, 0.):
        embedding.mul_(multiplier)
        delta.neg_()
        for stream in streams:
            stream.prefix.fill_(float("nan"))
            stream.blocks.fill_(float("nan"))
            stream.output.fill_(float("nan"))
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert allocated == torch.cuda.memory_allocated()
        for field in ("prefix", "blocks", "output"):
            assert torch.equal(getattr(streams[0], field).view(torch.int16),
                               getattr(streams[1], field).view(torch.int16))
    graph.reset()
