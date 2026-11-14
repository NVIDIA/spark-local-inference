# SPDX-License-Identifier: Apache-2.0

"""Direct batched MLA projections equal materialized bmm-plus-copy references."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an assigned CUDA device")


@pytest.mark.parametrize("rows", [1, 5, 8, 32])
@torch.no_grad()
def test_direct_mla_projection_matches_bmm_copy_and_replay(rows):
    from ignition.components.linear import BatchedUnquantizedProjection
    from ignition.preparation.compilation import compile_forward

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    heads = 8
    generator = torch.Generator().manual_seed(3300 + rows)

    def randn(*shape, scale=1.0):
        return (torch.randn(shape, generator=generator) * scale).to(torch.bfloat16).to(device)

    query = randn(rows, heads, 192)
    attended = randn(rows, heads, 512)
    key = randn(heads, 128, 512, scale=0.03)
    value = randn(heads, 512, 128, scale=0.03)
    absorbed = torch.full((rows, heads, 576), 19, dtype=torch.bfloat16, device=device)
    expanded = torch.full((rows, heads * 128), -23, dtype=torch.bfloat16, device=device)
    absorb = BatchedUnquantizedProjection(
        query[:, :, :128].transpose(0, 1), key, absorbed[:, :, :512].transpose(0, 1),
        major="n", session=None, name="test.absorb")
    expand = BatchedUnquantizedProjection(
        attended.transpose(0, 1), value, expanded.view(rows, heads, 128).transpose(0, 1),
        major="n", session=None, name="test.expand")

    def forward():
        absorb(query[:, :, :128].transpose(0, 1))
        expand(attended.transpose(0, 1))

    def check():
        expected_absorbed = torch.bmm(query[:, :, :128].transpose(0, 1), key).transpose(0, 1)
        expected_expanded = torch.bmm(attended.transpose(0, 1), value).transpose(0, 1).flatten(1)
        torch.cuda.synchronize(device)
        torch.testing.assert_close(absorbed[:, :, :512], expected_absorbed, rtol=0, atol=0)
        torch.testing.assert_close(expanded, expected_expanded, rtol=0, atol=0)
        assert absorbed[:, :, 512:].eq(19).all()

    compiled = compile_forward(forward)
    compiled()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    for replay in range(3):
        query.copy_(randn(rows, heads, 192))
        attended.copy_(randn(rows, heads, 512))
        absorbed.fill_(19)
        expanded.fill_(-23)
        before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize(device)
        assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == before
        check()
    graph.reset()
