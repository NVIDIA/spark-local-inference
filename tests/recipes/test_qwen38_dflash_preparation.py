# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flash Next proposal-state geometry and preparation option boundaries."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ignition.components.dflash2_geometry import DraftAttentionGeometry
from ignition.recipes.qwen38.bootstrap import PreparationConfig
from ignition.recipes.qwen38.recipe import RECIPE
from ignition.recipes.qwen38.state import HybridStateLayout, HybridStatePool, checkpoint_copy_layout


@pytest.mark.parametrize("world", [1, 2, 4])
def test_dflash_rings_belong_to_complete_continuation_rows(world):
    config = SimpleNamespace(layers=5, attention=DraftAttentionGeometry(32, 8, 128, 128, 128, 10000, 2048))
    layout = HybridStateLayout.build(speculative_tokens=7, world_size=world, draft_config=config)
    assert layout.row_bytes == RECIPE.speculative_methods["dflash2"].ranks[world].state_row_bytes
    for layer in range(5):
        for field in ("key", "value"):
            ring = layout.region(f"draft{layer}.{field}")
            assert ring.shape == (2048, 8 // world, 128)
            assert ring.dtype == torch.float8_e4m3fn
    assert not any(region.name.startswith(("draft_", "target_residual")) for region in layout.regions)
    transfers = checkpoint_copy_layout(layout, block=256)
    for region in layout.regions:
        if region.name.startswith("draft"):
            first, last = region.offset // 256, (region.offset + region.nbytes + 255) // 256
            assert not transfers[first:last].any()
    assert layout.region("gdn.0.convolution").shape[-1] == 10
    assert layout.region("ple_convolution").shape[-1] == 16
    pool = HybridStatePool.allocate(3, "meta", speculative_tokens=7, world_size=world, draft_config=config)
    assert pool.reserved_region("draft0.key").shape == (3, 2048, 8 // world, 128)
    mtp = HybridStateLayout.build(speculative_tokens=7, world_size=world)
    assert mtp.region("target_residual").shape == (10240,)
    assert mtp.region("draft_raw_anchor").shape == (1,)


def test_dflash_preparation_options_validate_before_weight_loading():
    base = dict(mtp=True, speculative_method="dflash2", draft_snapshot=Path("draft"))
    config = PreparationConfig(**base, online_mxfp8=True, draft_head_quantization="nvfp4",
                               force_moe_decode_a16=True)
    assert config.speculative_method == "dflash2" and config.online_mxfp8
    with pytest.raises(ValueError, match="at least one state row"):
        PreparationConfig(**base, snapshot_budget_bytes=142_821_119)
    with pytest.raises(ValueError, match="separate prefill"):
        PreparationConfig(**base, mixed_prefill=True)
    with pytest.raises(ValueError, match="require DFlash2"):
        PreparationConfig(mtp=True, online_mxfp8=True)
    with pytest.raises(ValueError, match="require DFlash2"):
        PreparationConfig(mtp=True, draft_head_quantization="mxfp8")


def test_prefix_identity_separates_draft_context_and_target_arithmetic():
    from dataclasses import replace
    from ignition.recipes.qwen38.bootstrap import _identity

    manifest = SimpleNamespace(headers_sha256={"model": "header"},
                               metadata_sha256={"config.json": "config"})
    draft = SimpleNamespace(text={"hidden_size": 2560, "sliding_window": 2048})
    ordinary = PreparationConfig()
    mtp = replace(ordinary, mtp=True)
    dflash = replace(mtp, speculative_method="dflash2", draft_snapshot=Path("draft"))
    def identity(config, draft_config=None):
        return _identity(manifest, config=config, draft_config=draft_config)
    baseline = identity(dflash, draft)
    variants = [identity(ordinary), identity(mtp), baseline,
        identity(replace(dflash, online_mxfp8=True), draft),
        identity(replace(dflash, force_moe_decode_a16=True), draft),
        identity(replace(dflash, native_dense_activations=True), draft),
        identity(dflash, SimpleNamespace(text={"hidden_size": 2560, "sliding_window": 1024}))]
    assert len(set(variants)) == len(variants)
    assert identity(replace(dflash, speculative_widths=(1,)), draft) == baseline
