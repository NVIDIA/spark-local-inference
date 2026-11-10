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


def test_equal_selection_stages_reuse_owners_and_preserve_representative_geometry(monkeypatch):
    def operation(*args, **kwargs):
        return SimpleNamespace(args=args, kwargs=kwargs, plan=object())

    for name in ("PreparedFP8Indexer", "PreparedAllGather", "PreparedRowTopK", "PreparedBroadcast",
                 "UnquantizedProjection"):
        monkeypatch.setattr(indexer, name, operation)
    rows = 257
    source = torch.empty((rows, 7168), dtype=torch.bfloat16)
    query = torch.empty((rows, 1536), dtype=torch.bfloat16)
    weights = {field: torch.empty(shape, dtype=torch.bfloat16) for field, shape in (
        ("indexer.wq_b", (4096, 1536)), ("indexer.wk", (128, 7168)),
        ("indexer.weights_proj", (32, 7168)), ("indexer.k_norm.weight", (128,)),
        ("indexer.k_norm.bias", (128,)))}
    cache = torch.zeros((6, 64, 132), dtype=torch.uint8)
    metadata = SimpleNamespace(positions=torch.arange(rows),
        token_page_table=torch.zeros((rows, 3), dtype=torch.int32))
    shards = ContextShards.on(0, 1, "cpu")
    prepared = indexer.PreparedDSAIndexer(weights, source, query, metadata, cache, shards,
        context=None, eps=1e-5, mode="prefill", session=None, name="selection", workspace=None,
        buffers={}, latent_cache=None)
    assert prepared.bounds == ((0, 128), (128, 256), (256, 257))
    assert tuple(prepared.selections) == (128, 1)
    full, tail = prepared.selections[128], prepared.selections[1]
    _assert_geometry(tail.lengths, full.lengths[:1])
    for part, (start, end) in ((full, (0, 128)), (tail, (256, 257))):
        _assert_geometry(part.indexer.args[0], prepared.query[start:end].view(end - start, 32, 128))
        _assert_geometry(part.indexer.args[1], prepared.head_weights[start:end])
        _assert_geometry(part.indexer.args[3], part.pages.expand(end - start, -1))
        assert part.indexer.args[5] is None
        assert part.indexer.args[6] is part.local
        assert part.indexer.kwargs["scores"] is part.scores
    assert full.indexer.plan is not tail.indexer.plan
    assert full.fold.plan is not tail.fold.plan


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
        return SimpleNamespace(bounds=bounds, selections={2: object(), 1: object()})

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
    assert tuple(prepared.parts) == (2, 1)
    first, tail = (part["attention"] for part in prepared.parts.values())
    assert first is prepared.parts[2]["attention"]
    assert tail.args[1] is not first.args[1]
    for rows, (start, end) in ((2, (0, 2)), (1, (4, 5))):
        part = prepared.parts[rows]
        call = part["attention"]
        _assert_geometry(call.args[0], prepared.absorbed[start:end])
        _assert_geometry(call.args[2], prepared.attended[start:end])
        _assert_geometry(call.args[3], constants.pages[:rows])
        _assert_geometry(call.args[5], constants.starts[:rows + 1])
        _assert_geometry(call.args[1], part["gather"].args[4].view(-1, 1, RECORD_BYTES))
    assert first.plan is not tail.plan
