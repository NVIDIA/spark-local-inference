# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compression geometry stays exact and disjoint from reusable layer scratch."""

import pytest
import torch

from ignition.components.paged_batch import Metadata
from ignition.recipes.deepseek41.attention import AttentionIndices
from ignition.recipes.deepseek41.decoder_prefill import PreparedDecoderSuffix


def set_metadata(metadata, starts, lengths):
    rows = len(starts)
    pages = torch.arange(rows * metadata.page_table.shape[1], dtype=torch.int32).view_as(metadata.page_table)
    tokens = torch.arange(rows * metadata.max_query_tokens).view(rows, -1)
    metadata.set(torch.tensor(starts), torch.tensor(lengths, dtype=torch.int32),
                 torch.arange(rows) + 1, pages, tokens, torch.tensor([sum(n > 0 for n in lengths)]))


def tensors(indices):
    yield indices.position
    yield indices.window_slots
    yield indices.window_indices
    yield indices.window_lengths
    for shared in indices.ratios.values():
        yield from vars(shared).values()


def assert_compression(indices, metadata):
    positions, requests, pages = metadata.positions.tolist(), metadata.requests.tolist(), metadata.page_table.tolist()
    assert indices.position.tolist() == [max(p, 0) for p in positions]
    for ratio, shared in indices.ratios.items():
        lengths, slots, rotary = [], [], []
        for position, request in zip(positions, requests, strict=True):
            positive = max(position, 0)
            completed = (positive + 1) // ratio
            lengths.append(completed if position >= 0 else 0)
            page = pages[request][min(positive // 128, len(pages[request]) - 1)]
            slot = page * (128 // ratio) + (positive % 128) // ratio
            slots.append(slot if position >= 0 else -1)
            rotary.append((completed - 1) * ratio)
        assert shared.lengths.dtype == shared.index_lengths.dtype == shared.active_width.dtype == torch.int32
        assert shared.position.dtype == shared.slots.dtype == torch.int64
        assert shared.lengths.tolist() == lengths
        assert shared.index_lengths.tolist() == [min(n, 512) for n in lengths]
        assert shared.active_width.tolist() == [max(lengths)]
        assert shared.slots.tolist() == slots
        assert shared.position.tolist() == rotary


@pytest.mark.parametrize("mode,tokens,rows,capacity", [
    ("decode", 1, 1, 1), ("decode", 8, 8, 1), ("verify", 6, 1, 6),
    ("verify", 24, 8, 6), ("verify", 48, 8, 6),
    ("draft", 40, 8, 5), ("prefill", 4096, 1, 4096),
])
def test_shared_compression_geometry_across_frames_and_private_storage(mode, tokens, rows, capacity):
    metadata = Metadata.allocate(tokens, rows, 40, "cpu", max_query_tokens=capacity)
    page_size = 128 if mode == "decode" else 128 + capacity
    indices = AttentionIndices(metadata, mode=mode, tokens=tokens, page_size=page_size,
        width=page_size if mode == "draft" else 128, ratios=(1, 2)).functional()
    private_tensors = [torch.zeros((128, 512), dtype=dtype) for dtype in (torch.int32, torch.int64)]
    previous = None
    for turn in range(6):
        starts = [value + turn for value in (0, 1, 126, 127, 128, 511, 4095, 4096)][:rows]
        lengths = [tokens // rows] * rows
        if turn % 2:
            lengths[-1] = 0
        set_metadata(metadata, starts, lengths)
        if previous is not None:
            for tensor in tensors(previous):
                tensor.fill_(-777)
        result = indices(metadata)
        assert_compression(result, metadata)
        saved = [tensor.clone() for tensor in tensors(result)]
        for tensor in private_tensors:
            tensor.fill_(turn + 17)
        for _ in range(40):
            for actual, expected in zip(tensors(result), saved, strict=True):
                assert torch.equal(actual, expected)
        assert result.compressed(1) is result.ratios[1]
        previous = result


def test_compression_geometry_tracks_packed_decoder_suffix_and_preserves_negative_rotary_positions():
    source = Metadata.allocate(4096, 1, 40, "cpu")
    packing = PreparedDecoderSuffix(source)
    full = AttentionIndices(source, mode="prefill", tokens=4096, page_size=4224,
                            width=128, ratios=(1, 2)).functional()
    suffix = AttentionIndices(packing.metadata, mode="prefill", tokens=128, page_size=256,
        width=128, ratios=(1,), replay_start=packing.replay_start).functional()
    packing.functional()
    for start, count in ((0, 4096), (127, 129), (128, 128), (4095, 1), (4096, 0), (0, 1)):
        set_metadata(source, [start], [count])
        full_result = full(source)
        packed = packing.begin(source)
        suffix_result = suffix(packed.metadata, replay_start=packed.replay_start)
        assert_compression(full_result, source)
        assert_compression(suffix_result, packed.metadata)
        if start == 0:
            assert full_result.compressed(2).position[0].item() == -2


@pytest.mark.parametrize("mode,capacity", [("decode", 1), ("verify", 8), ("draft", 8), ("prefill", 8)])
def test_compression_geometry_traces_as_one_graph(mode, capacity):
    metadata = Metadata.allocate(2 * capacity, 2, 40, "cpu", max_query_tokens=capacity)
    page_size = 128 if mode == "decode" else 128 + capacity
    indices = AttentionIndices(metadata, mode=mode, tokens=2 * capacity, page_size=page_size,
        width=page_size if mode == "draft" else 128, ratios=(1, 2)).functional()
    compiled = torch.compile(indices, backend="eager", fullgraph=True)
    for starts, lengths in (([127, 4095], [capacity, capacity]), ([128, 4096], [capacity, 0])):
        set_metadata(metadata, starts, lengths)
        result = compiled(metadata)
        assert_compression(result, metadata)
        expected = indices(metadata)
        for actual, reference in zip(tensors(result), tensors(expected), strict=True):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("ratio", [0, 3, 4, -1, True])
def test_compression_geometry_rejects_unsupported_ratios(ratio):
    metadata = Metadata.allocate(1, 1, 1, "cpu")
    with pytest.raises(ValueError, match="ratios must be one or two"):
        AttentionIndices(metadata, mode="decode", tokens=1, page_size=128, width=128, ratios=(ratio,))
