# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Context-parallel head combination across hosts matches an exact log-sum-exp merge."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("world", [2, 4])
def test_each_rank_merges_its_head_shard_from_every_ranks_partial(world):
    from ignition.components.head_exchange import merge, pack, pack_width

    torch.manual_seed(world)
    rows, heads, dim = 5, 64, 512
    local = heads // world
    partials = [torch.randn(rows, heads, dim, device="cuda").bfloat16() for _ in range(world)]
    lses = [torch.randn(rows, heads, device="cuda") * 4 for _ in range(world)]
    # Some ranks attended nothing for some heads, and one head nothing anywhere.
    lses[0][1, 3] = -float("inf")
    lses[-1][2, 5:9] = -float("inf")
    for lse in lses:
        lse[4, local + 1] = -float("inf")
        lse[4, 1] = -float("inf")
    width = pack_width(world, heads, dim)
    sends = []
    for rank in range(world):
        send = torch.full((rows, width), float("nan"), dtype=torch.bfloat16, device="cuda")
        pack(partials[rank], lses[rank], send, torch.tensor([rank], device="cuda"), world)
        sends.append(send)
    gathered = torch.cat(sends, dim=1)
    stacked = torch.stack([p.float() for p in partials])
    total = torch.logsumexp(torch.stack(lses), dim=0)
    weights = torch.where(torch.isinf(total)[None], 0.0, torch.exp(torch.stack(lses) - total[None]))
    expected = (weights[..., None] * stacked).sum(0)
    for rank in range(world):
        output = torch.full((rows, local, dim), float("nan"), dtype=torch.bfloat16, device="cuda")
        merge(gathered, partials[rank], lses[rank], output, torch.tensor([rank], device="cuda"), world)
        shard = expected[:, rank * local:(rank + 1) * local]
        torch.testing.assert_close(output.float(), shard, rtol=1e-2, atol=1e-2)
        assert not output.isnan().any()
    # A head that no rank attended is zero.
    owner = 0
    output = torch.empty((rows, local, dim), dtype=torch.bfloat16, device="cuda")
    merge(gathered, partials[owner], lses[owner], output, torch.tensor([owner], device="cuda"), world)
    assert (output[4, 1] == 0).all()
