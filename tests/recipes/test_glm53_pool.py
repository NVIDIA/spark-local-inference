# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rank-sharded continuation storage and exact partial-page geometry."""

import pytest

from ignition.components.sampling import DeviceRequestState
from ignition.recipes.glm53.pool import ContinuationPool, continuation_layout


@pytest.mark.parametrize("world", [2, 4])
def test_continuation_pool_binds_strided_kda_and_index_state(world):
    pool = ContinuationPool.allocate(("linear_attention", "indexed_attention"), world, 6, 3, "cpu")
    delta, mla = pool.decoder.layers
    assert delta.recurrent.shape == (6, 64 // world, 128, 128)
    assert delta.convolution.shape == (6, 3 * (64 // world) * 128, 3)
    assert mla.index.tail.shape == (6, 2, 4, 128)
    for value in (delta.recurrent, delta.convolution, mla.index.tail, pool.last_hidden, pool.consumed_tokens):
        assert value.untyped_storage().data_ptr() == pool.backing.data_ptr()
        assert value.stride(0) * value.element_size() == pool.layout.row_bytes
    assert mla.cache.shape == (12, 64, 528)
    assert mla.index.cache.shape == (3, 64, 132)
    assert pool.region("layer1.latent_page").shape == (6, 4, 64, 528)
    assert pool.region("layer1.index_page").shape == (6, 64, 132)
    requests = DeviceRequestState.allocate("cpu", vocab_size=154880, consumed_tokens=pool.consumed_tokens[1:5, 0])
    requests.consumed_tokens[2] = 257
    assert pool.consumed_tokens[3, 0] == 257
    assert not pool.consumed_tokens[[0, 1, 2, 4, 5]].count_nonzero()
    delta.recurrent[1].fill_(1)
    mla.index.tail[1].fill_(2)
    pool.backing[5].copy_(pool.backing[1])
    pool.backing[1].zero_()
    assert (delta.recurrent[5] == 1).all() and (mla.index.tail[5] == 2).all()
    mla.cache.fill_(3)
    mla.index.cache.fill_(4)
    pool.clear_for_preparation()
    assert not pool.backing.count_nonzero()
    assert not mla.cache.count_nonzero() and not mla.index.cache.count_nonzero()


@pytest.mark.parametrize("world", [2, 4])
def test_complete_model_continuation_budget(world):
    kinds = tuple("indexed_attention" if index % 4 == 3 else "linear_attention" for index in range(45))
    layout = continuation_layout(kinds, world)
    assert len(layout.regions) == 34 * 2 + 11 * 3 + 2
    assert layout.row_bytes % 256 == 0
    assert layout.snapshot_capacity() * layout.row_bytes <= 4 << 30
    assert (layout.snapshot_capacity() + 1) * layout.row_bytes > 4 << 30
    for first, second in zip(layout.regions, layout.regions[1:]):
        assert first.offset + first.nbytes <= second.offset
        assert second.offset % 256 == 0
