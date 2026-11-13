# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preparation KV backing covers graph shapes without sizing the serving cache."""

import pytest

from ignition.recipes.qwen38.bootstrap import PreparationConfig


@pytest.mark.parametrize('concurrency', [1, 3, 4, 32, 64])
@pytest.mark.parametrize('prefill', [64, 4096, 16384])
@pytest.mark.parametrize('width', [1, 6, 8])
def test_preparation_pages_cover_prefill_and_partial_verification_pages(concurrency, prefill, width):
    config = PreparationConfig(max_active=concurrency, prefill_chunk_size=prefill,
                               mtp=True, speculative_widths=tuple(sorted({1, width})))
    # A row beginning at the last token of a page needs a second page for wider verification.
    row_pages = (63 + width + 63) // 64
    assert config.preparation_pages >= prefill // 64 + config.request_capacity * row_pages
    assert config.preparation_pages <= prefill // 64 + 2 * config.request_capacity


def test_preparation_budget_does_not_grow_with_serving_context_or_available_memory(monkeypatch):
    def unexpected(*args):
        pytest.fail('preparation page count must not depend on free GPU memory')

    monkeypatch.setattr('torch.cuda.mem_get_info', unexpected)
    short = PreparationConfig(context=4096)
    long = PreparationConfig(context=262144)
    assert short.preparation_pages == long.preparation_pages == 72
    assert long.preparation_pages * 64 < long.context


def test_explicit_preparation_capacity_is_preserved():
    assert PreparationConfig(pages=128).preparation_pages == 128
    for pages in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match='page capacity'):
            PreparationConfig(pages=pages)


@pytest.mark.parametrize('concurrency', [1, 3, 4, 16, 64])
def test_preparation_state_backing_covers_live_rows_and_snapshot_destinations(concurrency):
    config = PreparationConfig(max_active=concurrency, mtp=True, mixed_prefill=True)
    assert config.preparation_state_rows == 1 + 2 * config.request_capacity
    assert config.preparation_state_rows == PreparationConfig(
        max_active=concurrency, snapshot_budget_bytes=32 << 30).preparation_state_rows
