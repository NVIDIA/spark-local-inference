# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU identity and geometry checks for serial sparse-attention storage views."""

from types import SimpleNamespace

import pytest
import torch

from ignition.recipes.kimi31 import indexer
from ignition.recipes.kimi31.dcp import ContextShards


def _assert_geometry(actual, expected):
    assert actual.untyped_storage()._cdata == expected.untyped_storage()._cdata
    assert actual.storage_offset() == expected.storage_offset()
    assert actual.shape == expected.shape
    assert actual.stride() == expected.stride()
    assert actual.dtype == expected.dtype
    assert actual._version == expected._version


class _OffsetAllocator:
    def zeros(self, shape, dtype):
        return torch.zeros((shape[0] + 2, *shape[1:]), dtype=dtype)[1:-1]


@pytest.mark.parametrize("allocator", [None, _OffsetAllocator()])
def test_shared_buffers_retain_equal_prefix_views(allocator):
    values = {}
    first = indexer.SharedBuffers(values, "cpu", allocator)
    capacity = first.buffer("scores", (128, 8), torch.float32)
    partial = first.buffer("scores", (65, 8), torch.float32)
    second = indexer.SharedBuffers(values, "cpu", allocator)
    assert second.buffer("scores", (128, 8), torch.float32) is capacity
    assert second.buffer("scores", (65, 8), torch.float32) is partial
    _assert_geometry(partial, capacity[:65])
    before = capacity._version
    partial.add_(1)
    assert capacity._version == before + 1
    assert second.buffer("scores", (65, 8), torch.float32)._version == capacity._version
    with pytest.raises(ValueError, match="matching non-row"):
        second.buffer("scores", (65, 7), torch.float32)
    with pytest.raises(ValueError, match="matching non-row"):
        second.buffer("scores", (65, 8), torch.int32)


def test_shared_buffers_growth_does_not_reuse_prior_storage_views():
    values = {}
    first = indexer.SharedBuffers(values, "cpu")
    original = first.buffer("scores", (2, 8), torch.float32)
    original.fill_(7)
    second = indexer.SharedBuffers(values, "cpu")
    larger = second.buffer("scores", (4, 8), torch.float32)
    rebound = first.buffer("scores", (2, 8), torch.float32)
    assert rebound is second.buffer("scores", (2, 8), torch.float32)
    assert rebound is not original
    _assert_geometry(rebound, larger[:2])
    assert not rebound.count_nonzero()
    assert (original == 7).all()


def test_equal_selection_stages_share_scratch_and_preserve_input_offsets(monkeypatch):
    def operation(*args, **kwargs):
        return SimpleNamespace(args=args, kwargs=kwargs, plan=object())

    from ignition.components import candidate_selection
    monkeypatch.setattr(indexer, "PreparedFP8Indexer", operation)
    for name in ("PreparedAllGather", "PreparedOrderedRowTopK"):
        monkeypatch.setattr(candidate_selection, name, operation)
    rows = 257
    query = torch.zeros((rows, 4096), dtype=torch.bfloat16)
    weights = torch.zeros((rows, 32), dtype=torch.bfloat16)
    cache = torch.zeros((4, 64 * 132), dtype=torch.uint8)
    metadata = SimpleNamespace(positions=torch.arange(rows),
        token_page_table=torch.zeros((rows, 2), dtype=torch.int32))
    shards = ContextShards.on(0, 1, "cpu")
    buffers = {}
    bounds = ((0, 128), (128, 256), (256, 257))
    parts = tuple(indexer.PreparedIndexSelection(query, weights, cache, metadata, shards, None,
        buffers, start=start, end=end, mode="prefill", session=None, name=f"selection{start}",
        workspace=None, allocator=None, latent_cache=None) for start, end in bounds)
    names = ("pages", "lengths", "local", "scores", "packed", "gathered", "global_scores",
             "global_ids", "fold_lengths", "active", "selected_scores", "selected")
    for name in names:
        assert getattr(parts[0], name) is getattr(parts[1], name)
    assert parts[2].lengths is not parts[0].lengths
    _assert_geometry(parts[2].lengths, parts[0].lengths[:1])
    for part, (start, end) in zip(parts, bounds, strict=True):
        _assert_geometry(part.positions, metadata.positions[start:end])
        _assert_geometry(part.tables, metadata.token_page_table[start:end])
        _assert_geometry(part.indexer.args[0], query[start:end].view(end - start, 32, 128))
        _assert_geometry(part.indexer.args[1], weights[start:end])
        _assert_geometry(part.indexer.args[3], part.pages.expand(end - start, -1))
        assert part.indexer.args[5] is None
        assert part.indexer.args[6] is part.local
        assert part.indexer.kwargs["scores"] is part.scores
    assert parts[0].positions.storage_offset() != parts[1].positions.storage_offset()
    assert parts[0].indexer.args[0].storage_offset() != parts[1].indexer.args[0].storage_offset()
    assert len({id(part.indexer.plan) for part in parts}) == len(parts)
    assert len({id(part.fold.plan) for part in parts}) == len(parts)
    for name in ("counts", "exceptional"):
        full = getattr(parts[0].candidate_rows, name)
        assert getattr(parts[1].candidate_rows, name) is full
        _assert_geometry(getattr(parts[2].candidate_rows, name), full[:1])


def test_worker_serialization_preserves_shared_prefix_identity():
    from ignition.preparation.forward_metadata import dumps, loads

    storage = indexer.SharedBuffers({}, "cpu")
    full = storage.buffer("ids", (128, 8), torch.int32)
    partial = storage.buffer("ids", (65, 8), torch.int32)
    restored = loads(dumps((full, full, partial, partial)))
    assert restored[0] is restored[1]
    assert restored[2] is restored[3]
    assert restored[0] is not restored[2]
    for actual, expected in zip(restored, (full, full, partial, partial), strict=True):
        assert actual.shape == expected.shape
        assert actual.stride() == expected.stride()
        assert actual.storage_offset() == expected.storage_offset()
        assert actual.dtype == expected.dtype
    assert restored[0].untyped_storage()._cdata == restored[2].untyped_storage()._cdata
    canonical = restored[0]
    derived = canonical[:65]
    before = canonical._version
    with canonical.fake_mode:
        derived.add_(1)
    assert canonical._version == before + 1
    assert restored[1]._version == canonical._version


def test_selected_attention_reuses_only_identical_record_and_constant_views(monkeypatch):
    from ignition.recipes.kimi31 import attention
    from ignition.recipes.kimi31.dcp import RECORD_BYTES, TOPK

    bounds = ((0, 2), (2, 4), (4, 5))

    def projection(self, weights, source, metadata, cache, **kwargs):
        self.absorbed = torch.zeros((5, 2, RECORD_BYTES), dtype=torch.bfloat16)
        self.attended = torch.zeros((5, 2, 512), dtype=torch.bfloat16)
        self.query_norm = torch.zeros((5, 1536), dtype=torch.bfloat16)
        self.buffer = lambda name, dtype: torch.zeros(5, dtype=dtype)

    def selection(*args, **kwargs):
        return SimpleNamespace(bounds=bounds, parts=(object(), object(), object()))

    def operation(*args, **kwargs):
        return SimpleNamespace(args=args, kwargs=kwargs, plan=object())

    monkeypatch.setattr(attention.PreparedMLAProjection, "__init__", projection)
    monkeypatch.setattr(attention, "PreparedDSAIndexer", selection)
    monkeypatch.setattr(attention, "PreparedDenseMLA", operation)
    monkeypatch.setattr(attention, "PreparedPeerRecordGather", operation)
    constants = SimpleNamespace(starts=torch.arange(3, dtype=torch.int32),
        pages=torch.arange(2 * TOPK, dtype=torch.int32).view(2, TOPK), scale=torch.ones(1))
    context = SimpleNamespace(shards=ContextShards.on(0, 16, "cpu"), attention_constants=constants)
    cache = torch.zeros((2, 128, RECORD_BYTES), dtype=torch.float8_e4m3fn)
    prepared = attention.PreparedDSA({}, torch.zeros((5, 8)), None, cache,
        index_cache=torch.zeros((4, 64 * 132), dtype=torch.uint8), eps=1e-5, mode="prefill",
        session=None, name="attention", workspace=None, buffers={}, context=context,
        peers=torch.zeros(16, dtype=torch.int64))
    first, second, tail = (part["attention"] for part in prepared.parts)
    assert first.args[1] is second.args[1]
    assert first.args[3] is second.args[3]
    assert first.args[5] is second.args[5]
    assert tail.args[1] is not first.args[1]
    for part, (start, end) in zip(prepared.parts, bounds, strict=True):
        rows = end - start
        call = part["attention"]
        _assert_geometry(call.args[0], prepared.absorbed[start:end])
        _assert_geometry(call.args[2], prepared.attended[start:end])
        _assert_geometry(call.args[3], constants.pages[:rows])
        _assert_geometry(call.args[5], constants.starts[:rows + 1])
        _assert_geometry(call.args[1], part["gather"].args[4].view(-1, 1, RECORD_BYTES))
    assert first.args[0].storage_offset() != second.args[0].storage_offset()
    assert first.args[2].storage_offset() != second.args[2].storage_offset()
    assert len({id(part["attention"].plan) for part in prepared.parts}) == len(bounds)
