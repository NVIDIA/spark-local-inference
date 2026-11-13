# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Snapshot checks select committed columns and exclude invalid tail storage."""

from types import SimpleNamespace

import pytest
import torch

from qualify.checkpoint_state import checkpoint_transfer_checks


def fixture(consumed, accepted):
    slot, live, frozen = 2, 3, 5
    pages = [1, 4, 2]
    page = pages[consumed // 64]
    regions = {
        "gdn.0.recurrent": torch.arange(40 * 4, dtype=torch.float32).reshape(40, 2, 2),
        "gdn.0.convolution": torch.arange(40 * 20, dtype=torch.float32).reshape(40, 2, 10),
        "ple_convolution": torch.arange(40 * 32, dtype=torch.float32).reshape(40, 2, 16),
        "consumed_tokens": torch.zeros((40, 1), dtype=torch.int64),
        "last_hidden": torch.arange(80, dtype=torch.float32).reshape(40, 2),
    }
    cache = SimpleNamespace(
        main_k_cache=torch.arange(5 * 64 * 2, dtype=torch.float32).reshape(5, 64, 2),
        main_v_cache=torch.arange(5 * 64 * 2, dtype=torch.float32).reshape(5, 64, 2) + 1,
        compressed_k_cache=torch.arange(5 * 16 * 2, dtype=torch.float32).reshape(5, 16, 2),
        raw_logical_positions=torch.tensor([[-1, consumed - 2, consumed - 1, consumed]] * 5),
        raw_k_ring=torch.arange(5 * 4 * 2, dtype=torch.float32).reshape(5, 4, 2),
        raw_rope_positions=torch.arange(5 * 4 * 3).reshape(5, 4, 3),
    )
    for name, source in (("raw_keys", cache.raw_k_ring[live]),
                         ("raw_positions", cache.raw_logical_positions[live]),
                         ("raw_rope", cache.raw_rope_positions[live]),
                         ("boundary_key", cache.main_k_cache[page]),
                         ("boundary_value", cache.main_v_cache[page]),
                         ("boundary_selector", cache.compressed_k_cache[page])):
        regions["qsa.3." + name] = torch.zeros((40, *source.shape), dtype=source.dtype)
        regions["qsa.3." + name][frozen].copy_(source)
    regions["qsa.3.raw_positions"][frozen, -1] = -1
    regions["qsa.3.raw_anchor"] = torch.full((40, 1), consumed - 1, dtype=torch.int64)
    # Rounds commit accepted recurrent state into the live row.
    regions["gdn.0.recurrent"][frozen].copy_(regions["gdn.0.recurrent"][live])
    for name, history in (("gdn.0.convolution", 3), ("ple_convolution", 9)):
        regions[name][frozen].zero_()
        regions[name][frozen, :, :history].copy_(regions[name][live, :, accepted - 1:accepted - 1 + history])
    regions["last_hidden"][frozen].copy_(regions["last_hidden"][live])
    regions["consumed_tokens"][[live, frozen]] = consumed
    pool = SimpleNamespace(
        layout=SimpleNamespace(regions=[SimpleNamespace(name=name) for name in regions]),
        region=regions.__getitem__, consumed_tokens=regions["consumed_tokens"],
        ple_convolution=regions["ple_convolution"],
        gdn={0: SimpleNamespace(recurrent=regions["gdn.0.recurrent"], convolution=regions["gdn.0.convolution"])}
    )
    return pool, {3: cache}, dict(slot=slot, snapshot=0, pages=pages, consumed=consumed, accepted=accepted)


@pytest.mark.parametrize("consumed", [64, 65, 66, 67, 127])
@pytest.mark.parametrize("accepted", [1, 2, 8])
def test_committed_columns_and_exact_endpoint(consumed, accepted):
    pool, caches, args = fixture(consumed, accepted)
    assert all(checkpoint_transfer_checks(pool, caches, **args).values())
    pool.gdn[0].recurrent[5, 0, 0] += 1
    assert not checkpoint_transfer_checks(pool, caches, **args)["gdn.0.recurrent"]


def test_invalid_raw_entries_and_unused_partial_rows_do_not_define_state():
    pool, caches, args = fixture(66, 8)
    pool.region("qsa.3.raw_keys")[5, [0, 3]] = float("nan")
    pool.region("qsa.3.boundary_key")[5, 2:] = float("nan")
    pool.region("qsa.3.boundary_selector")[5] = float("nan")
    assert all(checkpoint_transfer_checks(pool, caches, **args).values())
    pool.region("qsa.3.boundary_key")[5, 1, 0] += 1
    assert not checkpoint_transfer_checks(pool, caches, **args)["qsa.3.boundary_key"]


@pytest.mark.parametrize("consumed", [1, 2, 3, 10, 15, 63])
def test_short_instruction_endpoints(consumed):
    pool, caches, args = fixture(consumed, 1)
    assert all(checkpoint_transfer_checks(pool, caches, **args).values())


def test_rejected_tags_and_nonzero_history_padding_fail():
    pool, caches, args = fixture(67, 8)
    pool.region("qsa.3.raw_positions")[5, -1] = 67
    pool.ple_convolution[5, 0, 9] = 1
    checks = checkpoint_transfer_checks(pool, caches, **args)
    assert not checks["qsa.3.raw_positions"]
    assert not checks["ple_convolution_tail_zero"]
