# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Heterogeneous physical records behind a shared logical page identity."""

import torch

from ignition.recipes.deepseek41.config import ModelConfig
from ignition.recipes.deepseek41.pool import ContinuationPool, PAGE_TOKENS, page_bytes


def config():
    return ModelConfig({
        "num_hidden_layers": 40, "hidden_size": 5120, "sliding_window": 128,
        "compress_ratios": [0, 0] + [2] * 18 + [1] * 20,
        "kv_source_layer_ids": [2, 8, 14, 20],
        "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
    }, {})


def test_shared_sources_preserve_compression_geometry():
    model = config()
    assert model.source(0) is None and model.source(1) is None
    assert model.source(7) == model.source(2)
    assert model.source(19) == model.source(14)
    assert model.source(39) == model.source(20)
    assert model.index_source(27) == 24
    assert model.index_source(39) == 36
    assert page_bytes(model) == (3 * 64 + 128) * (288 + 68)
    assert PAGE_TOKENS == 128


def test_contract_registers_compressed_planes_from_their_storage_owner():
    from ignition.recipes.deepseek41.contract import prepare

    pool = ContinuationPool.allocate(config(), 2, 3, "cpu")
    buffers = {}
    contract = prepare({}, buffers, pool, page_tokens=PAGE_TOKENS, draft_lag=None)
    regions = {r["storage"]["buffer"]: r for r in contract["cache"]["regions"]}
    assert len(regions) == 1 + len(pool.pages)
    for name, tensor in pool.pages.items():
        region = regions["cache." + name]
        assert buffers["cache." + name] is tensor
        assert region["storage"]["nbytes"] == tensor.shape[1]
        assert region["tokens_per_record"] == (1 if name.startswith("layer20.") else 2)


def test_continuation_copies_window_compressor_and_hidden_state():
    pool = ContinuationPool.allocate(config(), 3, 2, "cpu")
    for name in ("layer39.window", "layer2.pending_values", "layer14.pending_gates",
                 "layer8.pending_position", "last_hidden", "consumed_tokens"):
        region = pool.region(name)
        assert region.stride(0) * region.element_size() == pool.layout.row_bytes
        region[1].fill_(17)
    pool.backing[2].copy_(pool.backing[1])
    pool.backing[1].zero_()
    assert (pool.region("layer39.window")[2] == 17).all()
    assert (pool.region("layer2.pending_values")[2] == 17).all()
    assert (pool.region("last_hidden")[2] == 17).all()
    for name, pages in pool.partial_pages():
        assert pages.shape == (2, pool.layout.region(name).nbytes)
        assert pages.dtype == torch.uint8 and pages.is_contiguous()
        pages.fill_(23)
    assert pool.pages["layer2.latent_page"].shape[1] * 2 == pool.pages["layer20.latent_page"].shape[1]
    pool.clear_for_preparation()
    assert not pool.backing.count_nonzero()
    assert all(not value.count_nonzero() for value in pool.pages.values())
