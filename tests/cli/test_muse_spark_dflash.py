# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse Spark DFlash2 plan selection and ordinary-decoding preparation."""

import pytest

from ignition import planning
from ignition.cli import _parser, preparation
from ignition.errors import UsageError
from ignition.recipes import resolve


ALIAS = "muse-spark-open-nvfp4"
GIB = 1 << 30


def speculation(*flags):
    args = _parser().parse_args(["export", ALIAS, *flags])
    return preparation.speculation(args, resolve(ALIAS))


def test_muse_defaults_to_seven_adaptive_dflash_drafts():
    selected = speculation()
    assert selected.draft_tokens == 7
    assert selected.adaptive and selected.fixed_width is None
    assert selected.widths == tuple(range(1, 9))
    assert selected.token_capacity == 32
    assert selected.packed_verification and selected.graph_budget == 64
    assert speculation("--no-mtp") is None


def test_muse_width_overrides_preserve_standard_speculation_semantics():
    reduced = speculation("--mtp-draft-tokens", "3")
    assert reduced.widths == (1, 2, 3, 4)
    fixed = speculation("--no-mtp-adaptive-verification")
    assert fixed.draft_tokens == 7 and fixed.fixed_width == 8
    zero = speculation("--mtp-draft-tokens", "0")
    assert zero is not None and zero.widths == (1,)
    assert not zero.packed_verification
    with pytest.raises(UsageError, match="0 through 7"):
        speculation("--mtp-draft-tokens", "8")
    with pytest.raises(UsageError, match="--no-mtp disables speculation"):
        speculation("--no-mtp", "--mtp-draft-tokens", "3")


def test_muse_planning_omits_draft_weights_and_state_without_mtp():
    model = resolve(ALIAS)
    gpus = tuple(planning.Gpu(index, "NVIDIA SM120", f"GPU-{index}", 96 * GIB,
                             95 * GIB, (12, 0)) for index in range(8))
    plain = planning.plan(model.recipe, model.checkpoint, gpus, selected=tuple(range(8)),
                          concurrency=4, speculation=None, reserve_bytes=0)
    draft = planning.plan(model.recipe, model.checkpoint, gpus, selected=tuple(range(8)),
                          concurrency=4, speculation=speculation(), reserve_bytes=0)
    assert plain.fits and draft.fits
    assert draft.world_size == plain.world_size == 8
    assert draft.speculative_token_capacity == 32
    assert draft.weights_bytes - plain.weights_bytes == 1_596_065_024
    assert draft.costs.state_row_bytes - plain.costs.state_row_bytes == 5 * 2 * 2048 * 2 * 64
    assert draft.costs.page_bytes == plain.costs.page_bytes == 128 * 16 * 2 * 2 * 64
    assert model.recipe.live_rows(4, draft.speculation) == model.recipe.live_rows(4, None) == 5
    assert plain.context == draft.context == 262144
    assert plain.prefill_chunk == draft.prefill_chunk == 4096
    assert plain.max_images == draft.max_images == 0
