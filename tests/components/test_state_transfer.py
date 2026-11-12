# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordinary execution preserves unused draft bytes during continuation transfers."""

import os

import pytest
import torch
from triton.runtime.interpreter import InterpretedFunction

from ignition.components.state_transfer import _copy_rows


def test_shifted_draft_pages_copy_one_cache_page_per_part(monkeypatch):
    """_copy_pages strides the cache and offsets the second saved page by one cache page."""
    from types import SimpleNamespace

    from torch._subclasses.fake_tensor import FakeTensorMode

    from ignition.components.sampling import DeviceRequestState
    from ignition.components.state import StateLayout
    from ignition.components.state_transfer import PreparedStateTransfer, StateCommands

    layout = StateLayout.from_specs([("target", (128,), torch.uint8), ("draft", (2, 128), torch.uint8),
                                     ("consumed_tokens", (1,), torch.int64)])
    monkeypatch.setattr("ignition.components.state_transfer._pointer_table",
        lambda values, device: torch.empty((len(values), 2), dtype=torch.int64, device=device))
    with FakeTensorMode():
        device = torch.device("cuda:0")
        consumed = torch.empty((8, 1), dtype=torch.int64, device=device)
        pool = SimpleNamespace(layout=layout, region=lambda name: consumed,
            backing=torch.empty((8, layout.row_bytes), dtype=torch.uint8, device=device),
            memory_capacity=torch.empty(2, dtype=torch.int64, device=device))
        tables = torch.zeros((4, 2), dtype=torch.int32, device=device)
        pages = torch.empty((4, 128), dtype=torch.uint8, device=device)
        transfer = PreparedStateTransfer(pool, StateCommands.allocate(device), tables,
            pages=[("target", pages)], page_tokens=256,
            state_rows=torch.zeros((4, 8), dtype=torch.int64, device=device),
            requests=DeviceRequestState.allocate(device, vocab_size=128),
            trailing_pages=[("draft", pages)], trailing_tables=tables)
    assert list(transfer.trailing_pointers) == [128]


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an explicitly assigned CUDA device"))])
def test_ordinary_state_copy_and_reset_leave_draft_regions_untouched(device):
    backing = torch.full((4, 8192), 231, dtype=torch.uint8, device=device)
    backing[1].fill_(17)
    source = torch.tensor([1, -1], device=device)
    destination = torch.tensor([2, 3], device=device)
    operation = torch.tensor([0, 1], dtype=torch.int32, device=device)
    count = torch.tensor([2], dtype=torch.int32, device=device)
    errors = torch.zeros(2, dtype=torch.int32, device=device)
    args = (backing, source, destination, operation, count, errors, 8192, 1024, ((768, 1792), (4096, 5120)))
    kernel = _copy_rows if device == "cuda" else InterpretedFunction(_copy_rows.fn)
    kernel[(2, 8)](*args)
    if device == "cuda":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            kernel[(2, 8)](*args)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        graph.reset()
    wanted = torch.full((4, 8192), 231, dtype=torch.uint8)
    wanted[1:3].fill_(17)
    wanted[3].zero_()
    for start, end in args[-1]:
        wanted[2:, start:end].fill_(231)
    assert torch.equal(backing.cpu(), wanted)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned GPU")
@pytest.mark.parametrize("endpoint", [127, 128, 129, 130])
@torch.no_grad()
def test_excluded_partial_pages_survive_snapshot_and_restore(endpoint):
    from ignition.components.paged_state import PagedStateStorage
    from ignition.components.sampling import DeviceRequestState
    from ignition.components.state import StateLayout
    from ignition.components.state_transfer import PreparedStateTransfer, StateCommands

    device = torch.device("cuda:0")
    names = ("target", "draft", "target_trailing", "draft_trailing")
    specs = [(name, (128,)) for name in names]
    layout = StateLayout.from_specs([(name, (2, 128) if name.endswith("_trailing") else (128,), torch.uint8)
                                    for name in names]
                                   + [("consumed_tokens", (1,), torch.int64)])
    pool = PagedStateStorage.allocate(layout, specs, 8, 4, device)
    commands = StateCommands.allocate(device, capacity=4)
    tables = torch.tensor([[0, 1], [2, 3], [0, 0], [0, 0]], device=device, dtype=torch.int32)
    trailing_tables = tables.flip(1).contiguous()
    consumed = pool.region("consumed_tokens")
    requests = DeviceRequestState.allocate(device, vocab_size=128, capacity=4, consumed_tokens=consumed[1:5, 0])
    state_rows = torch.arange(1, 5, device=device)[:, None].expand(4, 8).contiguous()
    transfer = PreparedStateTransfer(pool, commands, tables,
        pages=[(name, cache) for name, cache in pool.partial_pages() if not name.endswith("_trailing")],
        trailing_pages=[(name, cache) for name, cache in pool.partial_pages() if name.endswith("_trailing")],
        trailing_tables=trailing_tables, page_tokens=128, state_rows=state_rows, requests=requests,
        excluded_regions=("draft", "draft_trailing"))
    transfer.prepare()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        transfer.validate()
        transfer.apply()

    def replay():
        allocated = torch.cuda.memory_allocated(device)
        graph.replay()
        torch.cuda.synchronize(device)
        assert torch.cuda.memory_allocated(device) == allocated
        assert not commands.errors.count_nonzero()

    try:
        pool.backing[5].fill_(73)
        consumed[1] = endpoint
        pool.region("target")[1].fill_(11)
        pool.region("target_trailing")[1, 0].fill_(12)
        pool.region("target_trailing")[1, 1].fill_(13)
        for name, value in (("target", 21), ("draft", 22), ("target_trailing", 31), ("draft_trailing", 32)):
            pool.pages[name].fill_(value)
        for page in range(4):
            pool.pages["target_trailing"][page].fill_(31 + page)
        commands.num_ops.fill_(1)
        commands.source_slots[0], commands.destination_slots[0], commands.request_slots[0] = 1, 5, 0
        commands.expected_consumed[0] = endpoint
        replay()
        target = 21 if endpoint % 128 else 11
        trailing = {127: (12, 32), 128: (32, 13), 129: (32, 13), 130: (32, 31)}[endpoint]
        assert (pool.region("target")[5] == target).all()
        for part, value in enumerate(trailing):
            assert (pool.region("target_trailing")[5, part] == value).all()
        assert (pool.region("draft")[5] == 73).all()
        assert (pool.region("draft_trailing")[5] == 73).all()

        pool.region("draft")[2].fill_(91)
        pool.region("draft_trailing")[2].fill_(94)
        for name, value in (("target", 17), ("draft", 92), ("target_trailing", 18), ("draft_trailing", 93)):
            pool.pages[name].fill_(value)
        expected_pages = {name: cache.clone() for name, cache in pool.partial_pages()}
        if endpoint % 128:
            expected_pages["target"][int(tables[1, endpoint // 128])].fill_(target)
        if endpoint in (128, 129, 130):
            expected_pages["target_trailing"][3].fill_(trailing[0])
        if endpoint in (127, 130):
            expected_pages["target_trailing"][3 if endpoint == 127 else 2].fill_(trailing[1])
        commands.source_slots[0], commands.destination_slots[0], commands.request_slots[0] = 5, 2, 1
        replay()
        assert (pool.region("target")[2] == target).all()
        for part, value in enumerate(trailing):
            assert (pool.region("target_trailing")[2, part] == value).all()
        assert (pool.region("draft")[2] == 91).all()
        assert (pool.region("draft_trailing")[2] == 94).all()
        for name, cache in pool.partial_pages():
            assert torch.equal(cache, expected_pages[name]), name
    finally:
        graph.reset()
