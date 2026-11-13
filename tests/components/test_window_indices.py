# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Window address equality across packed capacities and decoder replay bounds."""

from types import SimpleNamespace

import pytest
import torch

from ignition.components.paged_batch import Metadata
from ignition.components.window_indices import PreparedWindowIndices
from ignition.recipes.deepseek4.attention import AttentionIndices
from ignition.recipes.deepseek41.decoder import PreparedDecoderBoundary
from ignition.recipes.deepseek41.decoder_prefill import PreparedDecoderSuffix
from ignition.recipes.deepseek41.attention import PreparedAttention
from ignition.recipes.deepseek41.attention import AttentionIndices as DeepSeek41Indices
from .window_indices_reference import LiteralDeepSeek4WindowIndices, LiteralDeepSeek41WindowIndices


FIELDS = ("window_slots", "window_indices", "window_lengths")


def assert_metadata_equal(actual, expected):
    for field in FIELDS:
        a, b = getattr(actual, field), getattr(expected, field)
        assert a.dtype == b.dtype and a.shape == b.shape
        assert torch.equal(a, b), field


def poison(indices):
    for field in FIELDS:
        getattr(indices, field).fill_(-777)
    if isinstance(indices, DeepSeek41Indices):
        indices.position.fill_(-777)
        for shared in indices.ratios.values():
            for value in vars(shared).values():
                value.fill_(-777)


def set_metadata(metadata, starts, lengths):
    device = metadata.positions.device
    starts = torch.tensor(starts, dtype=torch.int64, device=device)
    lengths = torch.tensor(lengths, dtype=torch.int32, device=device)
    rows, width = len(starts), metadata.max_query_tokens
    pages = torch.arange(rows * metadata.page_table.shape[1], dtype=torch.int32,
                         device=device).view_as(metadata.page_table)
    tokens = torch.arange(rows * width, device=device).view(rows, width)
    metadata.set(starts, lengths, torch.arange(rows, device=device) + 2, pages, tokens,
                 (lengths > 0).sum().view(1))


@pytest.mark.parametrize("mode,tokens,rows,query_capacity", [
    ("decode", 8, 8, 1), ("verify", 24, 8, 6), ("verify", 8, 8, 6),
    ("verify", 128, 32, 6), ("draft", 40, 8, 5), ("prefill", 4096, 1, 4096),
])
def test_deepseek41_window_formula_full_buffers(mode, tokens, rows, query_capacity):
    metadata = Metadata.allocate(tokens, rows, 40, "cpu", max_query_tokens=query_capacity)
    page_size = 128 if mode == "decode" else 128 + query_capacity
    width = page_size if mode == "draft" else 128
    replay_start = torch.zeros(rows, dtype=torch.int64) if mode == "prefill" else None
    actual = PreparedWindowIndices(metadata, mode=mode, tokens=tokens, page_size=page_size,
                                   width=width, replay_start=replay_start)
    expected = LiteralDeepSeek41WindowIndices(metadata, mode=mode, tokens=tokens, page_size=page_size,
                                             width=width, replay_start=replay_start)
    actual.functional()
    for start in (0, 127, 128, 4095):
        for partial in (False, True, None):
            per_row = min(query_capacity, tokens // rows)
            lengths = [per_row] * rows
            if partial is True:
                lengths[0] = max(0, per_row - 1)
                lengths[-1] = 0
            elif partial is None:
                lengths = [0] * rows
            set_metadata(metadata, [start + i for i in range(rows)], lengths)
            if replay_start is not None:
                replay_start.copy_(metadata.starts if partial is False else torch.zeros_like(replay_start))
            poison(expected)
            packet = actual(metadata, replay_start=replay_start)
            expected()
            assert_metadata_equal(packet, expected)
    if tokens == 24 and rows == 8:
        assert actual.window_page_size == 134


@pytest.mark.parametrize("mode,visible", [("decode", False), ("verify", False),
    ("draft", False), ("prefill", False), ("prefill", True)])
def test_deepseek4_wrapper_preserves_literal_window_formula(mode, visible):
    rows, width = (1, 256) if mode == "prefill" else (3, 1 if mode == "decode" else 5)
    tokens = rows * width
    metadata = Metadata.allocate(tokens, rows, 40, "cpu")
    visibility = None
    if visible:
        visibility = (torch.arange(tokens) % 193, torch.arange(tokens).flip(0) % 193)
    actual = AttentionIndices(metadata, mode=mode, tokens=tokens, visible=visibility)
    expected = LiteralDeepSeek4WindowIndices(metadata, mode=mode, tokens=tokens,
        page_size=actual.window_page_size, width=actual.width, visible=visibility)
    actual.compressed(4)
    actual.functional()
    for start in (0, 127, 128, 4095):
        for lengths in ([width] * rows, [width - 1] + [0] * (rows - 1), [0] * rows):
            set_metadata(metadata, [start] * rows, lengths)
            poison(expected)
            packet = actual(metadata, visible=visibility)
            expected()
            assert_metadata_equal(packet, expected)
            positions = metadata.positions.clamp_min(0)
            assert torch.equal(packet.ratios[4].lengths, torch.where(metadata.positions >= 0, (positions + 1) // 4, 0))


@pytest.mark.parametrize("start", [0, 127, 128, 4095])
def test_decoder_boundary_refreshes_suffix_after_packing(start):
    metadata = Metadata.allocate(4096, 1, 40, "cpu")
    packing = PreparedDecoderSuffix(metadata).functional()
    boundary = object.__new__(PreparedDecoderBoundary)
    boundary.packing = packing
    source = torch.arange(4096 * 2).view(4096, 2).float()
    mix = torch.ones(4096, 4)
    observed = []
    boundary.mhc = lambda *args, **kwargs: (source, source + 1, source + 2, source + 3)
    class GlobalKV:
        ratio = 1
        def __call__(self, normalized, indexer, full, geometry):
            assert full is metadata
            assert torch.equal(normalized, source + 3)
            observed.append(full.lengths.clone())
    boundary.global_kv = GlobalKV()
    indices = SimpleNamespace(compressed=lambda ratio: None)
    shared = PreparedWindowIndices(Metadata.allocate(128, 1, 40, "cpu"),
        mode="prefill", tokens=128, page_size=256, width=128).functional()
    for count in (4096, 129, 128, 1, 127, 0, 4095):
        set_metadata(metadata, [start], [count])
        suffix, gathered, _ = boundary((source,), mix, metadata, indices, None)
        expected = LiteralDeepSeek41WindowIndices(suffix.metadata, mode="prefill", tokens=128,
            page_size=256, width=128, replay_start=suffix.replay_start)
        expected()
        assert_metadata_equal(shared(suffix.metadata, replay_start=suffix.replay_start), expected)
        assert suffix.replay_start.item() == (start + max(0, count - 128) if count > 128 else 0)
        assert torch.equal(gathered[0], suffix.gather(source))
    assert len(observed) == 7


def test_shared_window_buffers_survive_layer_consumption_and_capacity_switches():
    groups = []
    for tokens in (8, 24, 128):
        metadata = Metadata.allocate(tokens, 8, 40, "cpu", max_query_tokens=6)
        shared = PreparedWindowIndices(metadata, mode="verify", tokens=tokens, page_size=134, width=128)
        oracle = LiteralDeepSeek41WindowIndices(metadata, mode="verify", tokens=tokens, page_size=134, width=128)
        groups.append((metadata, shared.functional(), oracle))
    # Distinct prepared capacities can run sequentially without retaining stale addresses.
    for turn, index in enumerate((0, 1, 2, 1, 0, 2, 0)):
        metadata, shared, oracle = groups[index]
        lengths = [min(6, metadata.positions.numel() // 8)] * 8
        if turn % 2:
            lengths[-1] = 0
        set_metadata(metadata, [4095 + turn] * 8, lengths)
        packet = shared(metadata)
        oracle()
        before = tuple(getattr(packet, field).clone() for field in FIELDS)
        for _ in range(40):
            assert_metadata_equal(packet, oracle)
        for field, snapshot in zip(FIELDS, before):
            assert torch.equal(getattr(packet, field), snapshot)


@pytest.mark.parametrize("mismatch", ["metadata", "mode", "page_size", "width", "replay_start", "visible"])
def test_attention_rejects_incompatible_shared_window_indices(mismatch):
    metadata = Metadata.allocate(24, 8, 1, "cpu", max_query_tokens=6)
    other = Metadata.allocate(24, 8, 1, "cpu", max_query_tokens=6)
    arguments = dict(metadata=metadata, mode="verify", tokens=24, page_size=134, width=128)
    if mismatch == "metadata":
        arguments["metadata"] = other
    elif mismatch == "mode":
        arguments["mode"] = "prefill"
    elif mismatch == "page_size":
        arguments["page_size"] = 131
    elif mismatch == "width":
        arguments["width"] = 134
    elif mismatch == "replay_start":
        arguments["replay_start"] = torch.zeros_like(metadata.starts)
    else:
        arguments["visible"] = (torch.zeros(24), torch.zeros(24))
    helper = DeepSeek41Indices(**arguments)
    config = SimpleNamespace(text={"compress_ratios": [0], "rms_norm_eps": 1e-6, "candidate_source_layer_id": 0}, source=lambda layer: None)
    pool = SimpleNamespace(config=config, window=lambda *args, **kwargs: torch.zeros((1, 128 * 528), dtype=torch.uint8))
    with pytest.raises(ValueError, match="shared window indices must match"):
        PreparedAttention(0, {}, torch.zeros((24, 1)), None, metadata, pool, None,
            mode="verify", world=1, session=None, name="attention", workspace=None, window_metadata=helper)


def test_standalone_attention_refreshes_its_own_window_metadata(monkeypatch):
    from ignition.recipes.deepseek41 import attention
    metadata = Metadata.allocate(24, 8, 1, "cpu", max_query_tokens=6)
    config = SimpleNamespace(text={"compress_ratios": [0], "rms_norm_eps": 1e-6, "candidate_source_layer_id": 0}, source=lambda layer: None)
    pool = SimpleNamespace(config=config, window=lambda *args, **kwargs: torch.zeros((10, 128 * 528), dtype=torch.uint8))
    scratch = SimpleNamespace(projected=torch.zeros((24, 1792)),
        query_low=torch.zeros((24, 1280)), query=torch.zeros((24, 16, 512)),
        window_values=torch.zeros((24, 512)), attended=torch.zeros((24, 16, 512)),
        output=torch.zeros((24, 5120)))
    class Operation:
        encoded_layout = None
        def __init__(self, output):
            self.output = output
        def functional(self, **kwargs):
            return self
        def __call__(self, *args, **kwargs):
            return self.output.clone()
    monkeypatch.setattr(attention, "QuantizedProjection", lambda source, weight, out, **kwargs: Operation(out))
    monkeypatch.setattr(attention, "PreparedRMSNorm", lambda source, weight, out, **kwargs: Operation(out))
    monkeypatch.setattr(attention, "PreparedGroupedOutputProjection", lambda source, weight, out, **kwargs: Operation(out))
    monkeypatch.setattr(attention, "PreparedCompressedCacheWriter", lambda source, *args, **kwargs: Operation(source))
    consumed = []
    class Consumer(Operation):
        def __call__(self, query, indices, lengths, **kwargs):
            consumed.append((indices.clone(), lengths.clone()))
            return query.clone()
    monkeypatch.setattr(attention, "PreparedCompressedMLA", lambda query, *args, **kwargs: Consumer(query))
    weights = dict.fromkeys(("qkv_a", "q_b", "q_norm", "kv_norm", "output", "sink"))
    operation = PreparedAttention(0, weights, torch.zeros((24, 5120)), scratch, metadata, pool,
        SimpleNamespace(apply=lambda *args, **kwargs: None), mode="verify", world=4,
        session=None, name="attention", workspace=None)
    assert operation.owns_window_metadata
    oracle = LiteralDeepSeek41WindowIndices(metadata, mode="verify", tokens=24, page_size=134, width=128)
    for start, lengths in ((127, [3] * 8), (4095, [2, 3, 3, 3, 3, 3, 3, 0])):
        set_metadata(metadata, [start] * 8, lengths)
        operation(torch.zeros((24, 5120)), metadata, (torch.full((24, 512), -1), None, None))
        oracle()
        assert torch.equal(consumed[-1][0], oracle.window_indices)
        assert torch.equal(consumed[-1][1], oracle.window_lengths)


def test_decoder_and_draft_window_packets_preserve_live_outputs():
    groups = []
    for tokens, mode, width in ((256, "prefill", 128), (128, "prefill", 128), (15, "draft", 133)):
        rows = 3 if mode == "draft" else 1
        metadata = Metadata.allocate(tokens, rows, 4, "cpu", max_query_tokens=5 if rows == 3 else tokens)
        page_size = 133 if mode == "draft" else 128 + tokens
        group = DeepSeek41Indices(metadata, mode=mode, tokens=tokens, page_size=page_size, width=width).functional()
        set_metadata(metadata, [127] * rows, [5] * rows)
        packet = group(metadata)
        expected = tuple(getattr(packet, field).clone() for field in FIELDS)
        groups.append((metadata, group, packet, expected))
    for metadata, group, packet, expected in groups:
        set_metadata(metadata, [4095] * metadata.starts.numel(), [1] * metadata.starts.numel())
        following = group(metadata)
        for field, value in zip(FIELDS, expected, strict=True):
            getattr(following, field).fill_(42)
            assert torch.equal(getattr(packet, field), value)


def test_deepseek4_ragged_window_staging_uses_request_width():
    metadata = Metadata.allocate(5, 2, 4, "cpu", max_query_tokens=4)
    set_metadata(metadata, [127, 255], [4, 1])
    actual = AttentionIndices(metadata, mode="verify", tokens=5, visible=None)
    assert actual.window_page_size == 132
    actual.functional()
    packet = actual(metadata)
    assert packet.window_slots.tolist() == [128, 129, 130, 131, 260]
    expected = LiteralDeepSeek4WindowIndices(metadata, mode="verify", tokens=5,
        page_size=132, width=128, visible=None)
    expected()
    assert_metadata_equal(packet, expected)
