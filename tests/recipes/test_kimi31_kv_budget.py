# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3.1 latent and sparse-index pages have complete continuation storage."""

import pytest

from .test_kimi31 import config, raw_config


def test_rank_costs_include_latent_and_index_pages(config):
    import torch
    from ignition.recipes import resolve
    from ignition.recipes.kimi3.pool import ContinuationPool, continuation_layout, page_bytes, page_specs

    recipe = resolve("kimi-k3.1-pro").recipe
    costs = recipe.costs(16, None)
    specifications = dict(page_specs(config))
    assert len(specifications) == 48
    assert page_bytes(config) == costs.page_bytes == 24 * 128 * (576 + 132) == 2_174_976
    assert costs.page_tokens == 2048
    layout = continuation_layout(config, 16)
    assert layout.row_bytes == costs.state_row_bytes == 30_275_328
    assert layout.row_bytes - 29_869_824 == 24 * 128 * 132
    pool = ContinuationPool.for_model(config, 16, 2, 3, "meta")
    assert pool.layout.row_bytes == costs.state_row_bytes
    assert set(dict(pool.partial_pages())) == set(specifications)
    assert sum(value[0].nbytes for value in pool.pages.values()) == costs.page_bytes
    for layer, kind in enumerate(config.layers):
        if kind == "kda":
            assert layout.region(f"layer{layer}.recurrent").shape == (6, 128, 128)
            continue
        latent, index = pool.attention(layer), pool.indexer_pages(layer)
        assert latent.shape == (3, 128, 576) and latent.dtype == torch.float8_e4m3fn
        assert index.shape == (6, 64, 132) and index.dtype == torch.uint8
    for name, shape in specifications.items():
        saved = layout.region(name)
        assert saved.shape == shape and saved.dtype == torch.uint8
        assert saved.nbytes == pool.pages[name][0].nbytes


@pytest.mark.parametrize("chunk,pages", [(128, 9), (2048, 9), (2176, 10), (4096, 10)])
def test_preparation_pages_follow_global_context_geometry(chunk, pages):
    from ignition.recipes.kimi3.bootstrap import PreparationConfig

    config = PreparationConfig(prefill_chunk_size=chunk)
    assert config.page_tokens == 2048 and config.preparation_pages == pages
    assert not config.drafting
    assert (pages - 8) * config.page_tokens >= chunk
    assert (pages - 9) * config.page_tokens < chunk
