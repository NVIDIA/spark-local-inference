# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Fused GLM draft metadata placement matches the unfused reference for proposals and absorption."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not (os.environ.get("FLARE_RUN_CUDA_TESTS") and torch.cuda.is_available()),
                                reason="set FLARE_RUN_CUDA_TESTS=1 on a CUDA host")


@pytest.mark.parametrize("world", [1, 2])
@pytest.mark.parametrize("capacity", [1, 4])
@pytest.mark.parametrize("requests", [4, 3])
@pytest.mark.parametrize("compiled", [False, True])
def test_fused_draft_metadata_matches_reference(world, capacity, requests, compiled):
    from ignition.recipes.glm53.dcp import ContextShards
    from ignition.recipes.glm53.mtp import DraftMetadata

    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(world * 10 + capacity)
    blocks, slots_capacity = 40, 8
    shards = ContextShards.on(1, world, device) if world > 1 else ContextShards()
    reference = DraftMetadata(requests * capacity, requests, blocks, device, shards=shards)
    slots = torch.tensor([3, 0, 5, 6][:requests], dtype=torch.int64, device=device)
    lengths = torch.randint(0, capacity + 1, (requests,), generator=generator).to(device)
    lengths[2] = 0
    starts = torch.randint(0, blocks * 256 * world - 8, (requests,), generator=generator).to(device)
    if capacity == 1:
        # Proposal steps read a strided column of the drafts so far.
        drafts = torch.randint(0, 150000, (requests, 3), generator=generator).to(device)
        tokens = drafts[:, 1]
        hidden = torch.randn((requests, 4096), generator=generator).to(torch.bfloat16).to(device)
    else:
        tokens = torch.randint(0, 150000, (requests, capacity), generator=generator).to(device)
        hidden = torch.randn((requests, capacity, 4096), generator=generator).to(torch.bfloat16).to(device)
    tables = torch.randint(-1, 500, (slots_capacity, blocks), generator=generator, dtype=torch.int32).to(device)
    state_rows = torch.randint(0, 100, (slots_capacity, 8), generator=generator).to(device)
    def pack(lengths):
        return DraftMetadata.pack(slots, lengths, starts, tokens, hidden, tables,
            capacity=capacity, state_rows=state_rows, shards=shards)
    operation = torch.compile(pack, fullgraph=True) if compiled else pack
    fused = operation(lengths)
    reference.set_reference(slots, lengths, starts, tokens, hidden, tables,
        capacity=capacity, state_rows=state_rows)
    torch.cuda.synchronize()
    for name in ("token_ids", "hidden", "output_indices", "num_tokens"):
        assert torch.equal(getattr(fused, name), getattr(reference, name)), name
    for name in ("query_start_loc", "state_slots", "requests", "positions", "block_table", "token_slots"):
        assert torch.equal(getattr(fused.indexed, name), getattr(reference.indexed, name)), name
