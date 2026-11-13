# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prefill capacity and speculative catalog configuration."""

import pytest

from ignition.recipes.qwen38.bootstrap import PreparationConfig


def test_hyperconnection_sharding_requires_multiple_ranks():
    assert not PreparationConfig().shard_hyperconnections
    with pytest.raises(ValueError, match="tensor parallelism"):
        PreparationConfig(shard_hyperconnections=True)
    for devices, hosts in (((0, 1), None), ((0, 1, 2, 3), None), ((0, 0), ("a", "b"))):
        config = PreparationConfig(devices=devices, hosts=hosts, shard_hyperconnections=True)
        assert config.shard_hyperconnections


@pytest.mark.parametrize("size, expected", [
    (64, (64,)),
    (512, (128, 256, 512)),
    (3072, (128, 256, 512, 1024, 1536, 2048, 2560, 3072)),
    (4096, (128, 256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096)),
    (8192, (128, 256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192)),
])
def test_prefill_catalog_covers_aligned_limit_and_partial_tails(size, expected):
    config = PreparationConfig(prefill_chunk_size=size)
    assert config.prefill_capacities == expected
    for tokens in range(1, size + 1):
        assert next(capacity for capacity in config.prefill_capacities if capacity >= tokens) <= size


@pytest.mark.parametrize("size", [True, 4096.0, "4096"])
def test_prefill_chunk_requires_integer(size):
    with pytest.raises(TypeError, match="prefill_chunk_size"):
        PreparationConfig(prefill_chunk_size=size)


@pytest.mark.parametrize("size", [0, 16, 65, 4097, 262208])
def test_prefill_chunk_requires_page_alignment_and_context_bound(size):
    with pytest.raises(ValueError, match="prefill chunk size"):
        PreparationConfig(prefill_chunk_size=size)


def test_prefill_chunk_must_fit_explicit_context():
    with pytest.raises(ValueError, match="prefill chunk size"):
        PreparationConfig(context=2048, prefill_chunk_size=4096)
    assert PreparationConfig(context=64, prefill_chunk_size=64).prefill_capacities == (64,)


@pytest.mark.parametrize("widths,fixed", [((1, 9), None), ((2, 4), None), ((1, 4, 2), None),
                                         ((1, 2, 2), None), ((1, 2), 8), ((1, True), None)])
def test_speculative_policy_requires_a_prepared_width_catalog(widths, fixed):
    with pytest.raises(ValueError):
        PreparationConfig(mtp=True, speculative_widths=widths, speculative_fixed_width=fixed)


@pytest.mark.parametrize("concurrency", [1, 2, 3, 4, 64])
@pytest.mark.parametrize("maximum", range(1, 9))
def test_capture_covers_every_runtime_reduction(concurrency, maximum):
    from ignition.components.capacity import batch_capacities

    widths = tuple(range(1, maximum + 1))
    config = PreparationConfig(max_active=concurrency, mtp=True, speculative_widths=widths)
    assert set(config.speculative_shapes) == {
        (batch, width) for batch in batch_capacities(concurrency) for width in widths
        if width == 1 or batch * width <= max(batch_capacities(concurrency)[-1], maximum)}
    for active in range(1, concurrency + 1):
        for width in widths:
            assert all((batch, width) in config.speculative_shapes for batch in batch_capacities(active)
                       if width == 1 or batch * width <= max(batch_capacities(concurrency)[-1], maximum))




@pytest.mark.parametrize("drafts", range(8))
def test_fixed_serving_default_does_not_remove_capture_variants(drafts):
    from ignition.preparation.execution import GraphSignature

    config = PreparationConfig(mtp=True, speculative_widths=tuple(range(1, 9)),
                               speculative_fixed_width=drafts + 1)
    assert config.speculative_shapes == tuple((b, w) for b in (1, 2, 4) for w in range(8, 0, -1) if w == 1 or b * w <= 8)
    for batch, width in config.speculative_shapes:
        GraphSignature("speculative", batch, width)


def test_ranks_on_separate_hosts_may_share_a_local_device_ordinal():
    config = PreparationConfig(devices=(0, 0), hosts=("boot-a", "boot-b"))
    assert config.devices == (0, 0) and config.hosts == ("boot-a", "boot-b")
    for devices, hosts in (((0, 0), None), ((0, 0), ("boot-a", "boot-a")), ((0, 0), ("boot-a",)),
                           ((0, 0), ["boot-a", "boot-b"]), (None, ("boot-a",))):
        with pytest.raises(ValueError):
            PreparationConfig(devices=devices, hosts=hosts)
