# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Declaration tensors preserve geometry without owning serving temporaries."""

import pytest
import torch

from ignition.recipes.deepseek41.bootstrap import PreparationConfig
from ignition.recipes.deepseek41.decoder import Scratch


def _overlap(left, right):
    return max(left.data_ptr(), right.data_ptr()) < min(
        left.data_ptr() + left.nbytes, right.data_ptr() + right.nbytes)


def test_declaration_stages_do_not_overwrite_residuals_or_selections():
    scratch = Scratch.allocate(128, 1, 4, "cpu", candidate_tokens=16)
    retained = (scratch.residual, scratch.engram_residual, scratch.attention_hc.residual,
                scratch.ffn_hc.residual, scratch.attention.selected, scratch.attention.candidates,
                scratch.attention.candidate_counts, scratch.mlp_output)
    concurrent = (scratch.shared.gate_up, scratch.shared.intermediate, scratch.shared.output,
                  scratch.routed, scratch.mlp_partial)
    for index, value in enumerate(retained):
        value.fill_(index + 1)
    for index, value in enumerate(concurrent):
        assert all(not _overlap(value, other) for other in (*retained, *concurrent[:index]))
    assert all(not _overlap(value, other) for value in concurrent
               for other in (scratch.embedding, scratch.engram.gathered, scratch.engram.projected))
    engram = (scratch.engram.gathered, scratch.engram.local, scratch.engram.partial, scratch.engram.projected)
    for index, value in enumerate(engram):
        assert all(not _overlap(value, other) for other in engram[:index])
    for value in (scratch.embedding, *engram, *concurrent):
        value.fill_(42)
    for index, value in enumerate(retained):
        assert value.eq(index + 1).all()
    assert scratch.rows(64, 1).attention.candidates.shape == (16, 16384)
    assert scratch.rows(8, 1).attention.candidates.shape == (8, 16384)


def test_suffix_declarations_preserve_full_prefill_sources():
    prefill = Scratch.allocate(128, 1, 4, "cpu")
    suffix = Scratch.allocate(32, 1, 4, "cpu")
    pairs = [(getattr(prefill.attention_hc, name), getattr(suffix.attention_hc, name))
             for name in ("normalized", "residual", "post", "comb")]
    pairs.append((prefill.attn_mix, suffix.attn_mix))
    for source, destination in pairs:
        assert not _overlap(source, destination)
        source.fill_(7)
        destination.fill_(9)
        assert source.eq(7).all()
    first = prefill.rows(128, 1)
    second = prefill.rows(32, 1)
    assert first.residual.data_ptr() == second.residual.data_ptr() == prefill.residual.data_ptr()
    assert second.attention_hc.normalized.shape == (32, 5120)
    for source, destination in pairs:
        assert source.eq(7).all() and destination.eq(9).all()


@pytest.mark.parametrize("capacity,dspark,draft_width,expected", [
    (32, True, 6, 160), (32, False, 1, 32), (1, True, 6, 6),
])
def test_oneshot_capacity_excludes_prefills_assigned_to_dma(capacity, dspark, draft_width, expected):
    config = PreparationConfig(max_active=capacity, dspark=dspark,
        speculative_widths=tuple(range(1, draft_width + 1)), speculative_token_capacity=64)
    assert config.all_reduce_tokens == expected


def test_short_prefill_still_fits_the_oneshot_channel():
    config = PreparationConfig(max_active=1, dspark=False, prefill_chunk_size=64)
    assert config.all_reduce_tokens == 64


def test_three_rank_scratch_preserves_grouped_attention_and_unpadded_experts():
    config = PreparationConfig(devices=(0, 1, 2), max_active=32,
        speculative_widths=tuple(range(1, 9)), image_concurrency=4)
    scratch = Scratch.allocate(128, 32, 3, "meta")
    assert config.draft_tokens == 7
    assert scratch.attention.query.shape == (128, 24, 512)
    assert scratch.attention.attended.view(128, 3, 4096).shape == (128, 3, 4096)
    assert scratch.local_logits.shape == (32, 43136)
    assert scratch.logits.shape == (32, 129280)
    assert scratch.shared.intermediate.shape == (128, 768)
    assert scratch.engram.local.shape == (128, 2048)


@pytest.mark.parametrize("world", [3, 4])
def test_target_logits_remain_fp32_across_capacity_views(world):
    scratch = Scratch.allocate(128, 32, world, "meta")
    assert scratch.hidden.dtype == torch.bfloat16
    for view in (scratch, scratch.rows(8, 8), scratch.rows(32, 1)):
        assert view.local_logits.dtype == torch.float32
        assert view.logits.dtype == torch.float32
        assert view.local_logits.element_size() == 4


def test_draft_logits_keep_bf16_storage_independently_of_target():
    from ignition.recipes.deepseek41.decoder import DraftScratch

    scratch = DraftScratch.allocate(32, 32, 4, "meta")
    assert scratch.local_logits.dtype == scratch.logits.dtype == torch.bfloat16
    sliced = scratch.rows(8, 8)
    assert isinstance(sliced, DraftScratch)
    assert sliced.local_logits.dtype == sliced.logits.dtype == torch.bfloat16
