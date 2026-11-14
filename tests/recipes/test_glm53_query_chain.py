# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concurrent query scratch has identical rank shapes and disjoint live storage."""

from types import SimpleNamespace
from functools import partial

import torch
import torch.distributed as dist

from ignition.preparation.scratch import ScratchArena
from ignition.recipes.glm53.mla import QueryChain


def test_query_workspace_normalizes_zero_and_nonzero_rank_requirements(monkeypatch):
    def plan(size):
        return SimpleNamespace(memory_requirements=lambda: SimpleNamespace(scratch_nbytes=size))

    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    layouts = [(((128,), torch.uint8), ((0,), torch.uint8)),
               (((64,), torch.uint8), ((567552,), torch.uint8))]
    shapes = []
    for rank in range(2):
        arena = ScratchArena("cpu")
        chain = object.__new__(QueryChain)
        chain.arena = arena
        chain.parallel_plans = (plan(layouts[rank][0][0][0]),)
        chain.projection = SimpleNamespace(plan=plan(layouts[rank][1][0][0]))
        arena.register_finalizer(partial(chain.refresh, chain.parallel_plans, arena))

        def gather(ranks, value, group):
            ranks[:] = [(256, [layouts[0]]), (567808, [layouts[1]])]

        monkeypatch.setattr(dist, "all_gather_object", gather)
        arena.finalize(group=object())
        arena.freeze()
        assert chain.workspace.shape == (567552,)
        assert chain.workspace.storage_offset() == 256
        chain.workspace.fill_(17)
        arena.tensor[:256].zero_()
        assert chain.workspace.eq(17).all()
        shapes.append((chain.workspace.shape, chain.workspace.stride(), chain.workspace.storage_offset()))
    assert shapes[0] == shapes[1]
