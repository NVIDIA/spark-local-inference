# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3.1 complete continuation rows and latent/index partial pages under CUDA replay."""

import os

import pytest

from .test_kimi31 import config, raw_config

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("endpoint", [1023, 1024, 1025, 2047, 2048, 2049])
def test_snapshot_forks_restore_kda_latents_index_keys_and_scales(config, endpoint):
    import torch
    from ignition.components.sampling import DeviceRequestState
    from ignition.components.state_transfer import PreparedStateTransfer, StateCommands
    from ignition.recipes.kimi3.pool import ContinuationPool

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    pool = ContinuationPool.for_model(config, 16, 8, 5, device)
    assert pool.layout.row_bytes == 30_275_328
    commands = StateCommands.allocate(device, capacity=4)
    tables = torch.tensor([[3, 0, 4], [2, 2, 2], [1, 1, 1], [-1, -1, -1]],
                          device=device, dtype=torch.int32)
    consumed = pool.region("consumed_tokens")
    requests = DeviceRequestState.allocate(device, vocab_size=128, capacity=4, consumed_tokens=consumed[1:5, 0])
    state_rows = torch.arange(1, 5, device=device)[:, None].expand(4, 8).contiguous()
    transfer = PreparedStateTransfer(pool, commands, tables, pages=pool.partial_pages(),
        page_tokens=2048, state_rows=state_rows, requests=requests)
    transfer.prepare()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        transfer.validate()
        transfer.apply()

    def replay(operations):
        commands.source_slots.fill_(500)
        commands.destination_slots.fill_(500)
        commands.request_slots.fill_(-1)
        commands.expected_consumed.fill_(-1)
        commands.operation.zero_()
        commands.num_ops.fill_(len(operations))
        for index, (source, destination, slot) in enumerate(operations):
            commands.source_slots[index] = source
            commands.destination_slots[index] = destination
            commands.request_slots[index] = slot
            commands.expected_consumed[index] = endpoint
        torch.cuda.synchronize(device)
        before = torch.cuda.memory_stats(device)
        graph.replay()
        torch.cuda.synchronize(device)
        after = torch.cuda.memory_stats(device)
        for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
            assert after[key] == before[key], key

    try:
        generator = torch.Generator(device=device).manual_seed(421)
        pool.backing[1].random_(0, 256, generator=generator)
        consumed[1] = endpoint
        for name, cache in pool.partial_pages():
            cache.random_(0, 256, generator=generator)
            if name.endswith(".indexer"):
                subpages = cache.view(5, 2, 64 * 132)
                keys = subpages[..., :64 * 128].view(5, 2, 64, 128)
                scales = subpages[..., 64 * 128:].view(torch.float32)
                scales.copy_(torch.linspace(.125, 3.5, scales.numel(), device=device).view_as(scales))
                keys[:, 0].fill_(0x31)
                keys[:, 1].fill_(0xA7)
        partial = endpoint % 2048 != 0
        tail = int(tables[0, endpoint // 2048]) if partial else None
        saved_pages = {name: cache[tail].clone() for name, cache in pool.partial_pages()} if partial else {}
        source_row = pool.backing[1].clone()
        expected_row = source_row.clone()
        for name, page in saved_pages.items():
            region = pool.layout.region(name)
            expected_row[region.offset:region.offset + region.nbytes].copy_(page)
        replay([(1, 5, 0)])
        assert not commands.errors.count_nonzero()
        assert torch.equal(pool.backing[5], expected_row)
        assert torch.equal(pool.backing[1], source_row)

        pool.backing[1].fill_(255)
        for cache in pool.pages.values():
            cache.fill_(0xC3)
        expected_pages = {name: cache.clone() for name, cache in pool.partial_pages()}
        if partial:
            for name, page in saved_pages.items():
                expected_pages[name][1].copy_(page)
                expected_pages[name][2].copy_(page)
        replay([(5, 2, 1), (5, 3, 2)])
        assert not commands.errors.count_nonzero()
        for row in (2, 3, 5):
            assert torch.equal(pool.backing[row], expected_row)
        for name, cache in pool.partial_pages():
            assert torch.equal(cache, expected_pages[name]), name
            if partial and name.endswith(".indexer"):
                layer = int(name.split(".")[0].removeprefix("layer"))
                saved = saved_pages[name].view(2, 64 * 132)
                for page in (1, 2):
                    restored = pool.indexer_pages(layer)[2 * page:2 * page + 2].view(2, 64 * 132)
                    assert torch.equal(restored[..., :64 * 128], saved[..., :64 * 128])
                    assert torch.equal(restored[..., 64 * 128:], saved[..., 64 * 128:])
        assert not pool.backing[[0, 4, 6, 7]].count_nonzero()
        assert (pool.backing[1] == 255).all()

        if partial:
            before_rows = pool.backing.clone()
            tables[1, endpoint // 2048] = -1
            replay([(5, 2, 1)])
            assert commands.errors.tolist() == [6, 0, 0, 0]
            assert torch.equal(pool.backing, before_rows)
            for name, cache in pool.partial_pages():
                assert torch.equal(cache, expected_pages[name]), name
    finally:
        graph.reset()
