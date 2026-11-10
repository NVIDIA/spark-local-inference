# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Decoder suffix metadata preserves live positions, cache pages, and replay bounds."""

import pytest
import torch
from types import SimpleNamespace

from ignition.components.paged_batch import Metadata
from ignition.recipes.deepseek41.decoder_prefill import PreparedDecoderSuffix


@pytest.mark.parametrize("capacity", [256, 512, 4096])
@pytest.mark.parametrize("start", [0, 127, 4095])
def test_decoder_suffix_alternates_full_partial_and_inactive_chunks(capacity, start):
    source = Metadata.allocate(capacity, 1, 40, "cpu")
    packing = PreparedDecoderSuffix(source).functional()
    values = torch.arange(capacity * 12).reshape(capacity, 3, 4)
    output = torch.empty((128, 3, 4), dtype=values.dtype)
    pages = torch.arange(40).flip(0).view(1, -1)
    source.page_table.copy_(pages)
    source.starts.fill_(start)
    source.query_start_loc[0].zero_()
    source.token_ids.copy_(torch.arange(capacity) + 1000)
    for count in (capacity, 129, 128, 1, 127, 0, capacity - 1):
        source.lengths.fill_(count)
        source.query_start_loc[1] = count
        source.state_rows.fill_(7 if count else 0)
        source.live_counts.copy_(torch.tensor([count, 1 if count else 0]))
        source.sample_indices.fill_(max(0, count - 1))
        source.positions.fill_(-1)
        source.positions[:count] = torch.arange(start, start + count)
        output.fill_(-999)
        packed = packing.begin(source)
        output.copy_(packed.gather(values))
        target = packed.metadata
        retained = list(range(max(0, count - 128), count))
        kept = len(retained)
        assert target.positions.tolist() == [start + index for index in retained] + [-1] * (128 - kept)
        assert target.token_ids.tolist() == [1000 + index for index in retained] + [0] * (128 - kept)
        assert target.starts.item() == start + max(0, count - 128)
        assert target.lengths.item() == kept
        assert target.query_start_loc.tolist() == [0, kept]
        assert target.sample_indices.item() == max(0, kept - 1)
        assert target.state_rows.item() == (7 if count else 0)
        assert target.live_counts.tolist() == [kept, 1 if count else 0]
        assert target.requests.eq(0).all()
        assert torch.equal(target.page_table, pages)
        assert torch.equal(target.token_page_table, pages.expand(128, -1))
        expected = torch.zeros_like(output)
        if retained:
            expected[:kept] = values[retained]
        assert torch.equal(output, expected)
        assert packed.replay_start.item() == (start + retained[0] if count > 128 else 0)


@pytest.mark.parametrize("rows,tokens", [(2, 256), (1, 128)])
def test_decoder_suffix_rejects_unsupported_packing_geometry(rows, tokens):
    with pytest.raises(ValueError, match="one prefill chunk larger"):
        PreparedDecoderSuffix(Metadata.allocate(tokens, rows, 2, "cpu"))


def test_decoder_feedback_at_replay_boundary_is_rejected_before_component_construction(monkeypatch):
    from ignition.recipes.deepseek41 import decoder

    monkeypatch.setattr(decoder, "PreparedTableRequests", lambda *args, **kwargs: None)
    monkeypatch.setattr(decoder.Scratch, "allocate", lambda *args, **kwargs: None)
    monkeypatch.setattr(decoder, "PreparedEmbedding",
                        lambda *args, **kwargs: pytest.fail("unsupported feedback reached component construction"))
    weights = SimpleNamespace(config=SimpleNamespace(decoder_start=20,
        sources=(SimpleNamespace(layer=20),), draft=SimpleNamespace(target_layers=(20,))),
        draft=object(), layers=tuple(SimpleNamespace(engram=None) for _ in range(40)),
        shard=SimpleNamespace(world_size=4), embedding=None)
    scratch = SimpleNamespace(embedding=torch.zeros(1), hidden=torch.zeros((1, 5120)),
        local_logits=torch.zeros((1, 1)), logits=torch.zeros((1, 1)))
    with pytest.raises(ValueError, match="draft target layers after"):
        decoder.PreparedDecoder(weights, None, scratch, Metadata.allocate(256, 1, 2, "cpu"), None,
            mode="prefill", runtime=None, vocabulary_runtime=None, coordinator=None,
            session=None, name="boundary", workspace=None)
