# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Continuation forks preserve both compressed records in partial 256-token pages."""

import os

import pytest
import torch

from ignition.components.sampling import DeviceRequestState
from ignition.components.state_transfer import PreparedStateTransfer, StateCommands
from ignition.recipes.deepseek4.pool import PAGE_TOKENS, ContinuationPool
from .test_deepseek4_pool import model

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an assigned CUDA device")


@pytest.mark.parametrize("endpoint", [1, 127, 128, 129, 255, 256, 257, 511, 512])
@torch.no_grad()
def test_partial_page_snapshot_restores_independent_forks(endpoint):
    device = "cuda:0"
    pool = ContinuationPool.allocate(model(), 8, 5, device)
    commands = StateCommands.allocate(device, capacity=4)
    tables = torch.full((4, 3), -1, dtype=torch.int32, device=device)
    consumed = pool.region("consumed_tokens")
    requests = DeviceRequestState.allocate(device, vocab_size=128, capacity=4,
                                          consumed_tokens=consumed[1:5, 0])
    rows = torch.arange(1, 5, device=device)[:, None].expand(4, 8).contiguous()
    transfer = PreparedStateTransfer(pool, commands, tables, pages=pool.partial_pages(),
        page_tokens=PAGE_TOKENS, state_rows=rows, requests=requests)
    transfer.prepare()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        transfer.validate()
        transfer.apply()

    def replay(operations):
        commands.num_ops.fill_(len(operations))
        for index, (source, destination, slot) in enumerate(operations):
            commands.source_slots[index] = source
            commands.destination_slots[index] = destination
            commands.operation[index] = 0
            commands.expected_consumed[index] = endpoint
            commands.request_slots[index] = slot
        torch.cuda.synchronize()
        before = torch.cuda.memory_stats()
        graph.replay()
        torch.cuda.synchronize()
        after = torch.cuda.memory_stats()
        for key in ("allocation.all.allocated", "allocation.all.freed",
                    "segment.all.allocated", "segment.all.freed"):
            assert before[key] == after[key]
        assert not commands.errors.count_nonzero()

    pool.backing[1].random_(0, 256)
    consumed[1] = endpoint
    tables[0].copy_(torch.tensor([3, 0, 4], device=device))
    tables[1].fill_(2)
    tables[2].fill_(1)
    for cache in pool.pages.values():
        cache.random_(0, 256)
    tail = tables[0, endpoint // PAGE_TOKENS].item() if endpoint % PAGE_TOKENS else None
    expected_pages = {name: cache[tail].clone() for name, cache in pool.partial_pages()} if tail is not None else {}
    expected = pool.backing[1].clone()
    for name, value in expected_pages.items():
        region = pool.layout.region(name)
        expected[region.offset:region.offset + region.nbytes].copy_(value)
    replay([(1, 5, 0)])
    assert torch.equal(pool.backing[5], expected)
    pool.backing[1].fill_(255)
    for cache in pool.pages.values():
        cache.zero_()
    replay([(5, 2, 1), (5, 3, 2)])
    for row in (2, 3, 5):
        assert torch.equal(pool.backing[row], expected)
    for name, cache in pool.partial_pages():
        if tail is not None:
            assert torch.equal(cache[1], expected_pages[name])
            assert torch.equal(cache[2], expected_pages[name])
        assert not cache[3:].count_nonzero()
    graph.reset()
