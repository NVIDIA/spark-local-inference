# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Packed window and compression metadata are values of each decoder invocation."""

import pytest
import torch

from ignition.components.paged_batch import Metadata
from ignition.recipes.deepseek4.attention import AttentionIndices
from tests.components.window_indices_reference import LiteralDeepSeek4WindowIndices


def fill_metadata(metadata, start, lengths):
    rows = len(lengths)
    metadata.starts.copy_(torch.arange(rows) + start)
    metadata.lengths.copy_(torch.tensor(lengths, dtype=torch.int32))
    metadata.state_rows.copy_(torch.arange(rows) + 1)
    metadata.query_start_loc[0] = 0
    metadata.query_start_loc[1:].copy_(metadata.lengths.cumsum(0))
    metadata.positions.fill_(-1)
    metadata.requests.zero_()
    cursor = 0
    for row, length in enumerate(lengths):
        metadata.positions[cursor:cursor + length] = torch.arange(length) + metadata.starts[row]
        metadata.requests[cursor:cursor + length] = row
        cursor += length
    if metadata.token_page_table.stride(0) != 0:
        metadata.token_page_table.copy_(metadata.page_table[metadata.requests])


@pytest.mark.parametrize("mode,rows,width", [("decode", 3, 1), ("verify", 3, 5),
    ("draft", 3, 5), ("prefill", 1, 256)])
def test_window_and_compression_values_follow_runtime_metadata(mode, rows, width):
    tokens = rows * width
    declaration = Metadata.allocate(tokens, rows, 4, "cpu", max_query_tokens=width)
    operation = AttentionIndices(declaration, mode=mode, tokens=tokens)
    operation.compressed(4)
    operation.compressed(128)
    operation.dense_pages(torch.empty((tokens, 8), dtype=torch.int32))
    page_size, window_width = operation.window_page_size, operation.width
    operation.functional()
    metadata = Metadata.allocate(tokens, rows, 4, "cpu", max_query_tokens=width)
    metadata.page_table.copy_(torch.arange(rows * 4).view(rows, 4).flip(1))
    oracle = LiteralDeepSeek4WindowIndices(metadata, mode=mode, tokens=tokens,
        page_size=page_size, width=window_width)
    retained = []
    for start, lengths in [(0, [width] * rows), (127, [width] + [0] * (rows - 1)),
                            (256, [0] * rows), (511, [max(0, width - 1)] * rows)]:
        fill_metadata(metadata, start, lengths)
        actual = operation(metadata)
        oracle()
        for field in ("window_slots", "window_indices", "window_lengths"):
            assert torch.equal(getattr(actual, field), getattr(oracle, field))
        position, live = metadata.positions.clamp_min(0), metadata.positions >= 0
        pages = metadata.token_page_table.gather(1, (position // 256).clamp(max=3)[:, None]).squeeze(1)
        for ratio in (4, 128):
            compressed = actual.ratios[ratio]
            assert torch.equal(compressed.lengths, torch.where(live, (position + 1) // ratio, 0))
            slots = torch.where(live, pages.long() * (256 // ratio) + position.remainder(256) // ratio, -1)
            assert torch.equal(compressed.slots, slots)
            emitted = live & ((metadata.positions + 1) % ratio == 0)
            assert torch.equal(compressed.emitted_slots, torch.where(emitted, slots, -1))
        expected_dense = (metadata.token_page_table[:, :, None] * 2 + torch.arange(2)).flatten(1)
        assert torch.equal(actual.dense, expected_dense)
        for packet, saved in retained:
            assert torch.equal(packet.window_indices, saved)
        retained.append((actual, actual.window_indices.clone()))


@pytest.mark.parametrize("tracer", ["fx", "dynamo"])
def test_window_and_compression_metadata_trace_without_bound_inputs(tracer):
    from dataclasses import replace
    from torch.fx.experimental.proxy_tensor import make_fx

    declaration = Metadata.allocate(15, 3, 4, "cpu", max_query_tokens=5)
    operation = AttentionIndices(declaration, mode="verify", tokens=15)
    operation.compressed(4)
    operation.functional()
    runtime = Metadata.allocate(15, 3, 4, "cpu", max_query_tokens=5)
    runtime.page_table.copy_(torch.arange(12).view(3, 4))
    fill_metadata(runtime, 127, [5, 5, 5])
    fields = ("starts", "lengths", "positions", "requests", "state_rows", "page_table", "token_page_table")

    def forward(*values):
        metadata = replace(runtime, **dict(zip(fields, values, strict=True)))
        packet = operation(metadata)
        compressed = packet.ratios[4]
        return (packet.window_slots, packet.window_indices, packet.window_lengths,
                compressed.slots, compressed.lengths, compressed.active_width, compressed.emitted_slots)

    graph = (make_fx(forward)(*(getattr(runtime, field) for field in fields)) if tracer == "fx"
             else torch.compile(forward, backend="eager", fullgraph=True))
    for start, lengths in [(127, [5, 5, 5]), (256, [3, 0, 2]), (511, [0, 0, 0])]:
        fill_metadata(runtime, start, lengths)
        values = tuple(getattr(runtime, field) for field in fields)
        actual, expected = graph(*values), forward(*values)
        for result, wanted in zip(actual, expected, strict=True):
            torch.testing.assert_close(result, wanted, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["prefill", "decode", "verify", "draft"])
def test_window_only_layer_omits_decoder_shared_dense_indices(monkeypatch, mode):
    from types import SimpleNamespace
    from ignition.recipes.deepseek4 import attention as module

    monkeypatch.setattr(module, "normalize_rotate", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "transfer_window", lambda *args, **kwargs: None)
    source = torch.zeros((3, 4096), dtype=torch.bfloat16)
    operation = object.__new__(module.PreparedAttention)
    operation.owns_indices, operation.mode, operation.ratio = False, mode, 0
    operation.window_shape = (1, 128 * 584)
    operation.window = torch.zeros(operation.window_shape, dtype=torch.uint8)
    operation.window_page_size, operation.heads, operation.eps = 128, 8, 1e-6
    operation.concurrent_compression = operation.concurrent_index = operation.angles = None
    operation.rotary = SimpleNamespace(frequencies={False: torch.zeros(32)})
    operation.qkv = lambda value: torch.zeros((value.shape[0], 1536), dtype=value.dtype)
    operation.qnorm = lambda value: value
    operation.kvnorm = lambda value: value
    operation.qb = lambda value: torch.zeros((value.shape[0], 4096), dtype=value.dtype)
    operation.rotate = operation.window_writer = operation.commit = lambda *args, **kwargs: None
    received = []

    def attend(query, indices, lengths, *, indexed_indices, indexed_lengths, window):
        assert (indexed_indices is None) == (indexed_lengths is None)
        received.append((indexed_indices, indexed_lengths))
        return query

    operation.attention = attend
    operation.output_projection = lambda value: torch.zeros_like(source)
    metadata = Metadata.allocate(source.shape[0], 1, 1, "cpu")
    metadata.positions.copy_(torch.arange(source.shape[0]))
    metadata.lengths.fill_(source.shape[0])
    indices = SimpleNamespace(ratios={128: object()}, angles={},
        window_slots=torch.zeros(source.shape[0], dtype=torch.int64),
        window_indices=torch.zeros((source.shape[0], 128), dtype=torch.int32),
        window_lengths=torch.zeros(source.shape[0], dtype=torch.int32),
        dense=torch.full((source.shape[0], 8), 7, dtype=torch.int32))
    output, _ = operation(source, metadata, indices)
    assert output.shape == source.shape
    assert received == [(None, None)]
