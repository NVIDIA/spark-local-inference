# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3.1 draft rotary arithmetic, addressing, and attention merging."""

import math
from types import SimpleNamespace

import pytest
import torch

from ignition.components.parallel import TensorParallelShard
from ignition.recipes.kimi3.dcp import ContextShards
from ignition.recipes.kimi31 import dspark


@pytest.fixture
def config():
    rope = dict(rope_theta=1_000_000., factor=32., original_max_position_embeddings=32768,
                beta_fast=32, beta_slow=1, mscale=1., mscale_all_dim=0., attn_factor=.7426255848312643)
    text = dict(q_lora_rank=1536, qk_rope_head_dim=64, qk_head_dim=192, v_head_dim=128,
                intermediate_size=14336, rope_parameters=rope)
    return SimpleNamespace(hidden=7168, vocabulary=163840, draft=SimpleNamespace(heads=96, text=text))


def test_yarn_frequencies_and_unit_rotary_gain(config):
    rope = config.draft.text["rope_parameters"]
    low = math.floor(64 * math.log(32768 / (32 * 2 * math.pi)) / (2 * math.log(1_000_000)))
    high = math.ceil(64 * math.log(32768 / (2 * math.pi)) / (2 * math.log(1_000_000)))
    expected = []
    for index in range(32):
        ramp = min(1., max(0., (index - low) / (high - low)))
        inverse = 1_000_000. ** (-index / 32.)
        expected.append(inverse * (1 - ramp) + inverse / 32 * ramp)
    torch.testing.assert_close(dspark.inverse_frequencies(config, "cpu"), torch.tensor(expected), rtol=2e-6, atol=0.)
    assert dspark.rotary_gain(config) == 1.
    rope["attn_factor"] = 1.
    assert dspark.rotary_gain(config) == 1 + .1 * math.log(32.)


@pytest.mark.parametrize("heads", [None, 6])
def test_interleaved_rope_rounds_after_float_arithmetic(config, heads):
    shape = (5, 64) if heads is None else (5, heads, 64)
    generator = torch.Generator().manual_seed(13)
    values = torch.randn(shape, generator=generator).to(torch.bfloat16)
    positions = torch.tensor([0, 1, 32767, 32768, 1048575])
    inverse = dspark.inverse_frequencies(config, "cpu")
    angles = positions.float()[:, None] * inverse[None]
    cos, sin = angles.cos(), angles.sin()
    if heads is not None:
        cos, sin = cos[:, None], sin[:, None]
    reference = torch.empty_like(values)
    reference[..., ::2] = values[..., ::2].float() * cos - values[..., 1::2].float() * sin
    reference[..., 1::2] = values[..., 1::2].float() * cos + values[..., ::2].float() * sin
    torch.testing.assert_close(dspark.rotate(values, positions, inverse), reference, rtol=0., atol=0.)
    torch.testing.assert_close(dspark.rotate(values, torch.zeros_like(positions), inverse), values, rtol=0., atol=0.)


@pytest.mark.parametrize("start", [0, 1, 15, 16, 31, 2039, 2047, 2048, 2050])
def test_proposal_pages_exclude_history_and_future_positions(start):
    starts = torch.tensor([start, start + 3, 0])
    requests = torch.tensor([0] * 7 + [1] * 7 + [2] * 7)
    positions = torch.cat((torch.arange(start, start + 7), torch.arange(start + 3, start + 10),
                           torch.full((7,), -1)))
    metadata = SimpleNamespace(starts=starts, requests=requests, positions=positions)
    for rank in range(16):
        shards = ContextShards.on(rank, 16, "cpu")
        slots, history, current = dspark.proposal_metadata(metadata, shards)
        for index, position in enumerate(positions.tolist()):
            request = requests[index].item()
            first = starts[request].item()
            prefix = sum(token % 16 == rank for token in range(first))
            seen = sum(token % 16 == rank for token in range(first, position + 1))
            if position < 0:
                assert slots[index] == -1 and history[index] == 0 and current[index] == 0
            else:
                assert history[index] == prefix
                assert current[index] == seen
                expected = request * 128 + seen - 1 if position % 16 == rank else -1
                assert slots[index] == expected


def test_accepted_slots_ignore_rejected_and_unreserved_suffix():
    starts = torch.tensor([2046, 4094])
    requests = torch.tensor([0] * 8 + [1] * 8)
    positions = torch.cat((torch.arange(2046, 2054), torch.arange(4094, 4102)))
    tables = torch.tensor([[3, -1, -1]] * 8 + [[4, 5, -1]] * 8, dtype=torch.int32)
    metadata = SimpleNamespace(starts=starts, requests=requests, positions=positions, token_page_table=tables)
    counts = torch.tensor([2, 2])
    for rank in range(16):
        shards = ContextShards.on(rank, 16, "cpu")
        slots = dspark.accepted_slots(metadata, counts, shards)
        for index, position in enumerate(positions.tolist()):
            request = requests[index].item()
            accepted = position < starts[request].item() + counts[request].item()
            if accepted and position % 16 == rank:
                local = position // 16
                page = tables[index, local // 128].item()
                assert slots[index] == page * 128 + local % 128
            else:
                assert slots[index] == -1


def test_attention_partition_merge_handles_empty_and_inactive_rows():
    generator = torch.Generator().manual_seed(7)
    histories, currents = [0, 3, 2, 0, 9], [4, 0, 5, 0, 1]
    history = torch.full((5, 3, 8), torch.nan)
    current = torch.full_like(history, torch.nan)
    history_lse = torch.full((5, 3), torch.nan)
    current_lse = torch.full_like(history_lse, torch.nan)
    expected = torch.zeros_like(history)
    expected_lse = torch.full_like(history_lse, -torch.inf)
    for row, (first, second) in enumerate(zip(histories, currents)):
        scores = torch.randn(3, first + second, generator=generator) * 10
        values = torch.randn(3, first + second, 8, generator=generator)
        if first:
            history[row] = (scores[:, :first].softmax(-1)[..., None] * values[:, :first]).sum(1)
            history_lse[row] = scores[:, :first].logsumexp(-1)
        if second:
            current[row] = (scores[:, first:].softmax(-1)[..., None] * values[:, first:]).sum(1)
            current_lse[row] = scores[:, first:].logsumexp(-1)
        if first + second:
            expected[row] = (scores.softmax(-1)[..., None] * values).sum(1)
            expected_lse[row] = scores.logsumexp(-1)
    dspark.merge_attention(history, history_lse, current, current_lse, torch.tensor(histories),
                           torch.tensor(currents), history, history_lse)
    torch.testing.assert_close(history, expected, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(history_lse, expected_lse)


def test_draft_capacity_views_share_mutable_storage_and_keep_constants(config):
    from ignition.preparation.disposable import storage_ranges

    scratch = dspark.DraftScratch.allocate(config, TensorParallelShard(0, 16), 14, 7, "cpu")
    rows = scratch.rows(7, 7)
    assert rows.embedding.data_ptr() == scratch.embedding.data_ptr()
    assert rows.mlp.output.data_ptr() == scratch.mlp.output.data_ptr()
    assert rows.projection is scratch.projection
    assert rows.attention is scratch.attention
    assert rows.constants is scratch.constants
    assert scratch.projection["local"].shape == (14, 136)
    assert scratch.attention["block"].shape == (2, 128, 576)
    ranges = storage_ranges(scratch.scratch_tensors())
    for constant in scratch.constants.values():
        assert all(not start <= constant.data_ptr() < start + size for start, size in ranges)
    assert scratch.constants["starts"].tolist() == list(range(15))
    with pytest.raises(ValueError, match="every block position"):
        scratch.rows(7, 1)
