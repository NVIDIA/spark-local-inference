# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The same target-token envelope governs every speculative recipe."""

import pytest

from ignition.components.capacity import batch_capacities, proposal_shapes, speculative_shapes
from ignition.recipes.qwen38.bootstrap import PreparationConfig as Qwen
from ignition.recipes.glm53.bootstrap import PreparationConfig as Glm
from ignition.recipes.deepseek4.bootstrap import PreparationConfig as DeepSeek4
from ignition.recipes.deepseek41.bootstrap import PreparationConfig as DeepSeek41
from ignition.recipes.mimo_v2.bootstrap import PreparationConfig as Mimo
from ignition.recipes.step5.bootstrap import PreparationConfig as Step5


@pytest.mark.parametrize("concurrency", [1, 3, 4, 5, 8, 16, 33, 64])
@pytest.mark.parametrize("drafts", [0, 1, 3, 5, 7])
@pytest.mark.parametrize("override", [None, 8, 32])
def test_recipes_share_bounded_target_coverage(concurrency, drafts, override):
    widths = tuple(range(1, drafts + 2))
    capacity = max(batch_capacities(concurrency)[-1], drafts + 1) if override is None else override
    expected = {(b, w) for b in batch_capacities(concurrency) for w in widths
                if w == 1 or b * w <= capacity}
    for recipe, enabled in ((Qwen, "mtp"), (Glm, "mtp")):
        config = recipe(max_active=concurrency, speculative_widths=widths,
                        speculative_token_capacity=override, **{enabled: True})
        assert set(config.speculative_shapes) == expected
    for recipe, enabled in ((DeepSeek4, "dspark"), (Mimo, "dflash")):
        config = recipe(max_active=concurrency, speculative_widths=widths,
                        speculative_token_capacity=override, **{enabled: True})
        assert {(batch, width) for batch, width, packed in config.speculative_round_shapes if width == 1} == {
            (batch, 1) for batch in batch_capacities(concurrency)}
        assert all(width == 1 or (width == drafts + 1 and batch < packed <= min(capacity, batch * width))
                   for batch, width, packed in config.speculative_round_shapes)
    config = DeepSeek41(max_active=concurrency, speculative_widths=widths,
                       speculative_token_capacity=override)
    assert set(config.speculative_shapes) == {(b, 1) for b in batch_capacities(concurrency)}
    assert all(b < p <= min(capacity, b * (drafts + 1)) for b, p in config.packed_speculative_shapes)
    if drafts <= 3:
        config = Step5(max_active=concurrency, speculative_widths=widths,
                       speculative_token_capacity=override)
        assert set(config.speculative_shapes) == expected
    assert (1, drafts + 1) in expected
    assert all((b, 1) in expected for b in batch_capacities(concurrency))


def test_proposals_keep_full_depth_independently_of_target_coverage():
    assert len(proposal_shapes(4, 7)) == 3
    assert len(speculative_shapes(4, tuple(range(1, 9)), 8)) == 14
    assert set(proposal_shapes(4, 7)) == {(b, 7) for b in (1, 2, 4)}
    assert {w for b, w in speculative_shapes(4, tuple(range(1, 9)), 8) if b == 4} == {1, 2}


@pytest.mark.parametrize("capacity", [0, -1, 7, True, 8.0])
def test_insufficient_or_noninteger_capacity_is_rejected(capacity):
    with pytest.raises(ValueError):
        speculative_shapes(4, tuple(range(1, 9)), capacity)


@pytest.mark.parametrize("recipe,enabled", [(DeepSeek4, "dspark"), (Mimo, "dflash")])
def test_parallel_rounds_bound_physical_rows_independently_of_samples(recipe, enabled):
    config = recipe(max_active=2, speculative_widths=(1, 2, 3, 4),
                    speculative_token_capacity=4, **{enabled: True})
    assert set(config.speculative_round_shapes) == {(1, 1, 1), (2, 1, 2),
                                                   (1, 4, 2), (1, 4, 3), (1, 4, 4),
                                                   (2, 4, 3), (2, 4, 4)}
    assert config.decode_tokens == 4 and config.decode_samples == 8
    fixed = recipe(max_active=2, speculative_widths=(1, 2, 3, 4),
                   speculative_token_capacity=4, speculative_fixed_width=3, **{enabled: True})
    assert {width for _, width, _ in fixed.speculative_round_shapes} == {1, 3}
    assert fixed.decode_samples == 6


def test_kimi_parallel_rounds_have_packed_target_capacity():
    from ignition.recipes.kimi3.bootstrap import PreparationConfig

    config = PreparationConfig(max_active=2, speculative_widths=(1, 2, 3, 4),
                               speculative_token_capacity=4, draft_checkpoint="/tmp/dspark")
    assert (2, 4, 3) in config.speculative_round_shapes
    assert config.decode_tokens == 4 and config.decode_samples == 8


def test_shared_catalogs_cover_fifteen_drafts_with_bounded_target_rows():
    from ignition.components.capacity import parallel_speculative_shapes

    widths = tuple(range(1, 17))
    shapes = speculative_shapes(4, widths, 32)
    assert (1, 16) in shapes and (2, 16) in shapes and (4, 16) not in shapes
    assert set(proposal_shapes(4, 15)) == {(1, 15), (2, 15), (4, 15)}
    rounds = parallel_speculative_shapes(4, widths, 32)
    assert (1, 16, 16) in rounds and (2, 16, 32) in rounds and (4, 16, 32) in rounds
    assert all(width == 1 or batch < packed <= min(32, batch * width)
               for batch, width, packed in rounds)
    with pytest.raises(ValueError, match="Int32"):
        speculative_shapes(4, (*widths, 1 << 31), 32)
    with pytest.raises(ValueError, match="proposal depth"):
        proposal_shapes(4, (1 << 31) - 1)


@pytest.mark.parametrize("recipe,flag", [(Qwen, "mtp"), (Glm, "mtp"), (DeepSeek4, "dspark"),
    (DeepSeek41, "dspark"), (Mimo, "dflash"), (Step5, "mtp")])
def test_recipe_completion_capacity_tracks_enabled_native_catalog(recipe, flag):
    widths = (1, 2, 3, 4)
    active = recipe(speculative_widths=widths, **{flag: True})
    assert active.completion_token_capacity == 4
    ordinary = recipe(speculative_widths=widths, **{flag: False})
    assert ordinary.completion_token_capacity == 1


@pytest.mark.parametrize("width", [1, 8, 13, 16])
def test_qwen_dense_native_capacity_remains_independent_of_recurrent_state_columns(width):
    from pathlib import Path
    from ignition.recipes.qwen38_dense.bootstrap import PreparationConfig

    settings = PreparationConfig(draft_snapshot=Path("/tmp/dflash"), speculative_widths=tuple(range(1, width + 1)))
    assert settings.completion_token_capacity == width
    with pytest.raises(ValueError, match="fifteen"):
        PreparationConfig(draft_snapshot=Path("/tmp/dflash"), speculative_widths=(1, 32))
