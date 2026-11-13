# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Partially backed recipe caches preserve snapshots and grow without recapture."""

import os
from functools import partial

import pytest
import torch

from ignition.components.parallel import TensorParallelShard
from ignition.components.sampling import DeviceRequestState
from ignition.components.state_transfer import PreparedStateTransfer, StateCommands
from ignition.preparation.virtual_storage import VirtualStorage


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="explicit CUDA assignment required")


def pool_factory(recipe):
    if recipe == "kimi3":
        from dataclasses import replace
        from ignition.recipes.kimi3.bootstrap import PreparationConfig
        from ignition.recipes.kimi3.config import ModelConfig
        from ignition.recipes.kimi3.config import DSparkConfig
        from ignition.recipes.kimi3.pool import ContinuationPool
        from .test_kimi3 import CHECKPOINT as TARGET, DRAFT

        if not TARGET.is_dir() or not DRAFT.is_dir():
            pytest.skip("Kimi target and Red Hat DSpark checkpoints are unavailable")
        config = ModelConfig.load(TARGET)
        config = replace(config, draft=DSparkConfig.load(DRAFT, config))
        return partial(ContinuationPool.for_model, config, 16, speculative_tokens=7), PreparationConfig().page_tokens
    if recipe == "deepseek41":
        from ignition.recipes.deepseek41.pool import ContinuationPool, PAGE_TOKENS
        from .test_deepseek41_pool import config

        return partial(ContinuationPool.allocate, config()), PAGE_TOKENS
    if recipe == "deepseek4":
        from ignition.recipes.deepseek4.pool import ContinuationPool, PAGE_TOKENS
        from .test_deepseek4_pool import model

        return partial(ContinuationPool.allocate, model()), PAGE_TOKENS
    if recipe == "mimo_v2":
        from ignition.recipes.mimo_v2.pool import ContinuationPool, PAGE_TOKENS
        from .test_mimo_v2 import config

        return partial(ContinuationPool.allocate, config(), TensorParallelShard(0, 8)), PAGE_TOKENS
    if recipe == "qwen38_dense":
        from ignition.recipes.qwen38_dense.config import ModelConfig
        from ignition.recipes.qwen38_dense.pool import ContinuationPool, PAGE_TOKENS

        config = ModelConfig(dict(layer_types=["linear_attention", "full_attention"],
                                  hidden_size=128, num_key_value_heads=1, head_dim=256), {})
        return partial(ContinuationPool.for_model, config), PAGE_TOKENS
    from ignition.recipes.step5.config import ModelConfig, PAGE_TOKENS
    from ignition.recipes.step5.pool import ContinuationPool

    config = ModelConfig(dict(num_hidden_layers=2, num_nextn_predict_layers=3, hidden_size=128,
        layer_types=["full_attention"] + ["sliding_attention"] * 4, sliding_window=512,
        num_attention_heads=4, num_sliding_attention_heads=4, num_attention_groups_full_attention=1,
        num_attention_groups_sliding_attention=1, head_dim_full_attention=256,
        head_dim_sliding_attention=256, partial_rotary_factors=[.25] * 5, rope_theta=[10000] * 5,
        sparse_config=dict(region_block_size=8, proxy_dim=256)), {}, {})
    return partial(ContinuationPool.for_model, config, TensorParallelShard(0, 4)), PAGE_TOKENS


@pytest.mark.parametrize("recipe", ["step5", "deepseek41", "deepseek4", "mimo_v2", "qwen38_dense", "kimi3"])
@torch.no_grad()
def test_snapshot_restore_and_growth_with_unbacked_prefix_slots(recipe):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    torch.cuda.set_device(0)
    allocate, page_tokens = pool_factory(recipe)
    capacity = 4 if recipe == "kimi3" else 16
    backed, declared = 2 * capacity + 1, 145
    pages = 2 * capacity
    if recipe == "kimi3":
        from ignition.recipes.kimi3.bootstrap import PreparationConfig

        pages = PreparationConfig(max_active=capacity).preparation_pages
    storage = VirtualStorage("cuda:0")
    graph = torch.cuda.CUDAGraph()
    try:
        pool = allocate(declared, pages, "cuda:0", storage=storage,
                        maximum_pages=64, maximum_rows=160, backed_slots=backed)
        region = next(r for r in storage.dynamic_regions if r.group == "STATE_ROWS")
        rounded = lambda size: -(-size // storage.granularity) * storage.granularity
        assert region.nbytes == rounded(backed * pool.layout.row_bytes)
        for cache_region in storage.dynamic_regions:
            if cache_region.group == "KV_PAGES":
                assert cache_region.nbytes == rounded(pages * cache_region.unit_bytes)
        assert pool.backing.shape == (declared, pool.layout.row_bytes)
        assert pool.reserved_backing.shape == (160, pool.layout.row_bytes)
        pool.clear_for_preparation()

        requests = DeviceRequestState.allocate("cuda:0", vocab_size=128, capacity=capacity)
        commands = StateCommands.allocate("cuda:0", capacity=capacity)
        rows = torch.arange(1, capacity + 1, device="cuda")
        snapshots = rows + capacity
        states = rows[:, None].expand(-1, 8).contiguous()
        tables = torch.arange(2 * capacity, dtype=torch.int32, device="cuda").view(2, capacity).T.contiguous()
        transfer = PreparedStateTransfer(pool, commands, tables, pages=pool.partial_pages(),
            page_tokens=page_tokens, state_rows=states, requests=requests)
        transfer.prepare()
        with torch.cuda.graph(graph):
            transfer.validate()
            transfer.apply()

        endpoint = page_tokens + 1
        pool.backing[1:capacity + 1].random_(0, 256)
        pool.region("consumed_tokens")[1:capacity + 1].fill_(endpoint)
        requests.consumed_tokens.fill_(endpoint)
        commands.request_slots.copy_(rows - 1)
        commands.expected_consumed.fill_(endpoint)
        commands.num_ops.fill_(capacity)
        expected = pool.backing[1:capacity + 1].clone()
        saved_pages = {}
        for name, cache in pool.pages.items():
            cache[:2 * capacity].random_(0, 256)
            saved_pages[name] = cache[capacity:2 * capacity].clone()
            field = pool.layout.region(name)
            expected[:, field.offset:field.offset + field.nbytes].copy_(saved_pages[name])

        def replay(sources, destinations):
            commands.source_slots.copy_(sources)
            commands.destination_slots.copy_(destinations)
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated"):
                assert after[key] == before[key]

        replay(rows, snapshots)
        assert not commands.errors.any()
        assert torch.equal(pool.backing[capacity + 1:backed], expected)
        pool.backing[1:capacity + 1].zero_()
        for cache in pool.pages.values():
            cache[:2 * capacity].zero_()
        replay(snapshots, rows)
        assert not commands.errors.any()
        assert torch.equal(pool.backing[1:capacity + 1], expected)
        for name, cache in pool.pages.items():
            assert torch.equal(cache[capacity:2 * capacity], saved_pages[name])
            assert not cache[:capacity].any()

        addresses = {name: cache.data_ptr() for name, cache in pool.pages.items()}
        storage.grow("KV_PAGES", pages + 1)
        pool.memory_capacity[1] = pages + 1
        tables[0, 1] = pages
        replay(snapshots, rows)
        assert not commands.errors.any()
        for name, cache in pool.pages.items():
            assert cache.data_ptr() == addresses[name]
            assert torch.equal(cache[pages], saved_pages[name][0])

        commands.num_ops.fill_(1)
        destinations = snapshots.clone()
        destinations[0] = backed
        replay(rows, destinations)
        assert commands.errors[0].item() == 2
        address = pool.backing.data_ptr()
        storage.grow("STATE_ROWS", backed + 1)
        pool.memory_capacity[0] = backed + 1
        replay(rows, destinations)
        assert not commands.errors.any()
        assert pool.backing.data_ptr() == address
        assert torch.equal(pool.backing[backed], expected[0])
        assert region.nbytes == rounded((backed + 1) * pool.layout.row_bytes)
        pool.clear_for_preparation()
        assert not pool.backing[:backed + 1].any()
    finally:
        graph.reset()
        storage.close()
