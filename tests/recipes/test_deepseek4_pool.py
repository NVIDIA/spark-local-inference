# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""C4/C128 physical geometry and complete immutable continuation rows."""

from pathlib import Path

import pytest

from ignition.recipes.deepseek4.config import REVISION, ModelConfig
from ignition.recipes.deepseek4.pool import (
    PAGE_TOKENS,
    ContinuationPool,
    continuation_layout,
    latent_page_bytes,
    page_bytes,
    page_specs,
)
from ignition.recipes.deepseek4.recipe import RECIPE


def model():
    return ModelConfig(
        {
            "num_hidden_layers": 5,
            "hidden_size": 4096,
            "sliding_window": 128,
            "head_dim": 512,
            "index_head_dim": 128,
            "compress_ratios": [0, 0, 4, 128, 4, 0, 0, 0],
            "num_nextn_predict_layers": 3,
            "dspark_block_size": 5,
            "dspark_noise_token_id": 128799,
            "dspark_target_layer_ids": [2, 3, 4],
            "dspark_markov_rank": 256,
            "n_routed_experts": 256,
            "num_experts_per_tok": 6,
        },
        {},
    )


def test_pages_hold_two_c128_records_or_sixty_four_c4_records():
    specs = dict(page_specs(model()))
    assert PAGE_TOKENS == 256
    assert specs["layer2.latent_page"] == (latent_page_bytes(64),)
    assert specs["layer3.latent_page"] == (latent_page_bytes(2),)
    assert specs["layer4.index_page"] == (64 * 132,)
    assert "layer3.index_page" not in specs
    assert "layer0.latent_page" not in specs
    for records in (1, 2, 32, 64, 128, 4096):
        size = latent_page_bytes(records)
        assert size % 576 == 0
        assert records * 584 <= size < records * 584 + 576


def test_contract_registers_compressed_planes_from_their_storage_owner():
    from ignition.recipes.deepseek4.contract import prepare

    pool = ContinuationPool.allocate(model(), 2, 3, "cpu", dspark=False)
    buffers = {}
    contract = prepare({}, buffers, pool, page_tokens=PAGE_TOKENS, draft_lag=None)
    regions = {r["storage"]["buffer"]: r for r in contract["cache"]["regions"]}
    assert len(regions) == 1 + len(pool.pages)
    for name, tensor in pool.pages.items():
        region = regions["cache." + name]
        assert buffers["cache." + name] is tensor
        assert region["storage"]["nbytes"] == tensor.shape[1]
        assert region["tokens_per_record"] == (128 if name.startswith("layer3.") else 4)


def test_compressor_rows_preserve_overlap_and_the_complete_c128_interval():
    config = model()
    pool = ContinuationPool.allocate(config, 3, 2, "cpu")
    assert config.compressors[0].state_shape == (8, 1024)
    assert config.compressors[1].state_shape == (8, 256)
    assert config.compressors[2].state_shape == (128, 512)
    for index, region in enumerate(pool.layout.regions):
        values = pool.region(region.name)
        reserved = pool.reserved_region(region.name)
        assert reserved.dtype == values.dtype and reserved.shape == values.shape
        assert reserved.stride() == values.stride()
        assert values.stride(0) * values.element_size() == pool.layout.row_bytes
        values[1].fill_(index + 1)
    pool.backing[2].copy_(pool.backing[1])
    pool.backing[1].zero_()
    for index, region in enumerate(pool.layout.regions):
        assert (pool.region(region.name)[2] == index + 1).all(), region.name
    for name, pages in pool.partial_pages():
        assert pages.shape == (2, pool.layout.region(name).nbytes)
        pages.fill_(17)
    pool.clear_for_preparation()
    assert not pool.backing.count_nonzero()
    assert all(not pages.count_nonzero() for pages in pool.pages.values())


def test_disabling_drafts_retains_target_compressors_and_omits_draft_windows():
    pool = ContinuationPool.allocate(model(), 3, 2, "cpu", dspark=False)
    assert pool.region("layer3.compressor.values").shape == (3, 128, 512)
    assert pool.region("last_hidden").shape == (3, 4096)
    assert not any(r.name.startswith("draft") for r in pool.layout.regions)


def test_cached_vision_checkpoint_declares_integrated_dspark_and_both_compressors():
    path = (
        Path.home()
        / ".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-Vision-Exp"
        / "snapshots"
        / REVISION
    )
    if not (path / "config.json").is_file():
        pytest.skip("DeepSeek V4 Flash Vision checkpoint metadata is unavailable")
    config = ModelConfig.load(path)
    assert config.layers == 43 and config.draft.block_size == 5
    assert config.draft.target_layers == (40, 41, 42)
    assert (config.draft.experts, config.draft.top_k) == (256, 6)
    assert sum(c.ratio == 128 for c in config.compressors) == 20
    assert sum(c.ratio == 4 and not c.index for c in config.compressors) == 21
    for drafts, ranks in ((True, RECIPE.ranks), (False, RECIPE.plain_ranks)):
        for costs in ranks.values():
            assert costs.state_row_bytes == continuation_layout(config, dspark=drafts).row_bytes
            assert costs.page_bytes == page_bytes(config)
    for changes, expected in (
        ({"vision_n_layers": 0}, "vision_n_layers"),
        ({"dspark_block_size": 0}, "DSpark metadata"),
        ({"compress_ratios": [0] * 45}, "C4/C128"),
        ({"dspark_target_layer_ids": [42, 40]}, "DSpark metadata"),
        ({"expert_dtype": "fp8"}, "MXFP4"),
    ):
        with pytest.raises(ValueError, match=expected):
            ModelConfig.from_dict(config.text | changes)
