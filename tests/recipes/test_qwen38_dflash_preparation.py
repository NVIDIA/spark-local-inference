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


@pytest.mark.parametrize("fixed", [False, True])
def test_dflash_catalog_shares_physical_rows_without_shrinking_per_request_width(fixed):
    from ignition.recipes.qwen38.decoder import DecoderScratch

    config = PreparationConfig(mtp=True, speculative_method="dflash2", draft_snapshot=Path("draft"),
        max_active=64, context=256, prefill_chunk_size=64, speculative_widths=tuple(range(1, 9)),
        speculative_token_capacity=129, speculative_graph_budget=32,
        speculative_fixed_width=8 if fixed else None)
    shapes = config.speculative_round_shapes
    assert len(shapes) == 32 + len(config.batch_capacities)
    assert (64, 8, 65) in shapes and (64, 8, 129) in shapes
    assert all((batch, 1, batch) in shapes for batch in config.batch_capacities)
    assert all(batch <= packed <= min(129, batch * width) for batch, width, packed in shapes)
    assert config.token_capacity == 129 and config.completion_token_capacity == 8
    samples = max(batch * width for batch, width in config.speculative_shapes)
    assert samples == 512
    scratch = DecoderScratch.allocate(config.token_capacity, "meta", sample_rows=samples)
    view = scratch.rows(65, 512)
    assert view.hidden.shape == (65, 2560) and view.logits.shape == (512, 248320)


def test_dflash_graph_budget_rejects_missing_capacity_endpoints_before_loading():
    with pytest.raises(ValueError, match="capacity endpoints"):
        PreparationConfig(mtp=True, speculative_method="dflash2", draft_snapshot=Path("draft"),
            max_active=8, speculative_widths=tuple(range(1, 9)), speculative_token_capacity=64,
            speculative_graph_budget=7)


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
