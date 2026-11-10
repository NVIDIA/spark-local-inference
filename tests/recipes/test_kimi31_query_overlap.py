# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Query gathers retain explicit fork/join dependencies and serialized peer exchange."""

from types import SimpleNamespace

import pytest
import torch

from ignition.components import collectives
from ignition.preparation.scratch import ScratchArena
from ignition.recipes.kimi31 import attention, indexer
from ignition.recipes.kimi31.dcp import ContextShards, TOPK, RECORD_BYTES


class Plan:
    def __init__(self, scratch=0):
        self.scratch = scratch

    def memory_requirements(self):
        return SimpleNamespace(scratch_nbytes=self.scratch)


def test_concurrent_gather_declares_lifetimes_and_rejects_caller_scratch(monkeypatch):
    arena = ScratchArena('cpu')
    arena._auxiliary_stream = SimpleNamespace(cuda_stream=17)
    source, output, cache, local = (torch.zeros(4) for _ in range(4))
    plan = Plan()
    gather = SimpleNamespace(single=False, roce=False, plan=plan, output=output,
                             dependencies=[cache], handle=23)
    local_plans = (Plan(1024), Plan(2048))
    operation = collectives.ConcurrentAllGather(gather, workspace=arena, session=None,
                                                parallel_plans=local_plans)
    assert arena._parallel_plans == [(plan, local_plans[0]), (plan, local_plans[1])]
    calls = []
    monkeypatch.setattr(collectives, '_start_all_gather', lambda *args: calls.append(('start', args)))
    monkeypatch.setattr(collectives, '_join_all_gather', lambda *args: calls.append(('join', args)))
    pending = operation.start(source)
    assert operation.finish(pending, local) is output
    assert calls == [('start', (source, output, [cache], 23, 17)),
                     ('join', (source, output, [cache], local, 17))]
    plan.scratch = 256
    with pytest.raises(ValueError, match='without caller scratch'):
        operation.validate_scratch()


@pytest.mark.parametrize('single,roce,overlap', [(True, False, True), (False, True, True), (False, False, False)])
def test_gather_fork_rejects_unsupported_channel_or_disabled_overlap(single, roce, overlap):
    gather = SimpleNamespace(single=single, roce=roce)
    with pytest.raises(ValueError):
        collectives.ConcurrentAllGather(gather, workspace=ScratchArena('cpu', overlap=overlap),
                                        session=None, parallel_plans=(Plan(),))


def test_fx_fork_join_and_candidate_exchange_name_all_storage_dependencies():
    from torch.fx.experimental.proxy_tensor import make_fx

    def forward(source, cache, local, gathered, candidate):
        cache.copy_(source)
        collectives._start_all_gather(source, gathered, [cache], 1, 2)
        local.copy_(source + 1)
        collectives._join_all_gather(source, gathered, [cache], local, 2)
        collectives._all_gather(local, candidate, [gathered], 3)
        source.zero_()
        return candidate

    module = make_fx(forward, tracing_mode='fake')(*(torch.ones(4) for _ in range(5)))
    inputs = [n for n in module.graph.nodes if n.op == 'placeholder']
    start = next(n for n in module.graph.nodes if n.target is torch.ops.flare.start_all_gather.default)
    join = next(n for n in module.graph.nodes if n.target is torch.ops.flare.join_all_gather.default)
    exchange = next(n for n in module.graph.nodes if n.target is torch.ops.flare.pcie_all_gather.default)
    cache_write, = start.args[2]
    local_write = join.args[3]
    assert join.args[2] == [cache_write]
    assert cache_write.target is torch.ops.aten.copy_.default and cache_write.args[0] is inputs[1]
    assert local_write.target is torch.ops.aten.copy_.default and local_write.args[0] is inputs[2]
    assert start.args[:2] == (inputs[0], inputs[3])
    assert join.args[:2] == (inputs[0], inputs[3])
    assert exchange.args[:3] == (local_write, inputs[4], [inputs[3]])
    assert torch.ops.flare.start_all_gather.default._schema.arguments[1].alias_info.is_write
    assert torch.ops.flare.join_all_gather.default._schema.arguments[1].alias_info.is_write


@pytest.mark.parametrize('rows,mode,overlap,expected', [
    (1, 'decode', True, True), (1, 'decode', False, False),
    (2, 'decode', True, False), (1, 'verify', True, False), (1, 'prefill', True, False),
])
def test_query_overlap_only_declared_for_single_row_decode(monkeypatch, rows, mode, overlap, expected):
    def projection(self, weights, source, metadata, cache, **kwargs):
        self.absorbed = torch.zeros((rows, 6, RECORD_BYTES), dtype=torch.bfloat16)
        self.attended = torch.zeros((rows, 6, 512), dtype=torch.bfloat16)
        self.query_norm = torch.zeros((rows, 1536), dtype=torch.bfloat16)
        self.buffer = lambda name, dtype: torch.zeros(rows, dtype=dtype)

    selector = SimpleNamespace(merge=SimpleNamespace(dependencies=[]))
    plans = (Plan(), Plan())
    monkeypatch.setattr(attention.PreparedMLAProjection, '__init__', projection)
    monkeypatch.setattr(attention.PreparedMLAProjection, 'functional', lambda self, **kwargs: self)
    class Functional(SimpleNamespace):
        def functional(self, **kwargs):
            return self
    monkeypatch.setattr(attention, 'PreparedDSAIndexer', lambda *args, **kwargs:
                        SimpleNamespace(bounds=((0, rows),), selections={rows: selector}, sharded=None,
                                        local_plans=plans))
    monkeypatch.setattr(attention, 'PreparedDenseMLA', lambda *args, **kwargs: Functional(plan=Plan()))
    monkeypatch.setattr(attention, 'PreparedLseReduceScatter', lambda *args, **kwargs: Functional())
    monkeypatch.setattr(attention, 'PreparedPeerRecordGather', lambda *args, **kwargs: Functional())
    monkeypatch.setattr(attention, 'PreparedAllGather', lambda source, output, **kwargs:
                        Functional(output=output, dependencies=[], plan=Plan(), single=False, roce=False))
    forks = []
    monkeypatch.setattr(attention, 'ConcurrentAllGather', lambda gather, **kwargs:
                        forks.append((gather, kwargs)) or Functional())
    constants = SimpleNamespace(starts=torch.arange(rows + 1, dtype=torch.int32),
        pages=torch.arange(rows * TOPK, dtype=torch.int32).view(rows, TOPK), scale=torch.ones(1))
    context = SimpleNamespace(shards=ContextShards.on(0, 16, 'cpu'), attention_constants=constants,
                              heads=object(), coordinator=None)
    cache = torch.zeros((1, 128, RECORD_BYTES), dtype=torch.float8_e4m3fn)
    operation = attention.PreparedDSA({}, torch.zeros((rows, 8)), None, cache,
        index_cache=torch.zeros((2, 64 * 132), dtype=torch.uint8), eps=1e-5, mode=mode,
        session=None, name='attention', workspace=SimpleNamespace(overlap=overlap),
        buffers={}, context=context, peers=torch.zeros(16, dtype=torch.int64))
    assert (operation.query_overlap is not None) is expected
    assert len(forks) == int(expected)
    if expected:
        gather, options = forks[0]
        assert options['parallel_plans'] is plans
        assert gather.dependencies == [cache]
        assert selector.merge.dependencies == [gather.output]
    else:
        assert selector.merge.dependencies == []


def test_split_selection_keeps_sequential_composition():
    operation = indexer.PreparedIndexSelection.__new__(indexer.PreparedIndexSelection)
    calls = []
    query, weights, positions, tables = (object() for _ in range(4))
    packed, output = object(), object()
    operation.prepare_local = lambda *args: calls.append(('local', args)) or packed
    operation.exchange = lambda *args: calls.append(('exchange', args)) or output
    assert operation(query, weights, positions, tables) is output
    assert calls == [('local', (query, weights, positions, tables)), ('exchange', (packed, positions))]


def test_candidate_alternatives_share_query_join_dependency(monkeypatch):
    from ignition.components import candidate_selection

    monkeypatch.setattr(candidate_selection, 'PreparedAllGather', lambda *args, **kwargs:
                        SimpleNamespace(dependencies=[]))
    monkeypatch.setattr(candidate_selection, 'PreparedOrderedRowTopK', lambda *args, **kwargs: object())
    monkeypatch.setattr(candidate_selection, 'PreparedOwnerTopK', lambda *args, **kwargs:
                        SimpleNamespace(transport=SimpleNamespace(dependencies=[])))
    buffers = indexer.SharedBuffers({}, 'cpu')
    cache, joined_query = torch.zeros(4), torch.ones(4)
    selection = candidate_selection.PreparedCandidateSelection(
        torch.zeros((1, 4), dtype=torch.int32), torch.zeros((1, 4)), torch.zeros(1, dtype=torch.int64),
        torch.zeros(1, dtype=torch.int64), torch.zeros((1, 4), dtype=torch.int32), world=16,
        gather_runtime=None, owner_runtime=SimpleNamespace(world_size=16), reduction_runtime=object(),
        session=None, coordinator=None, name='selection', buffers=buffers, dependencies=(cache,))
    selection.dependencies.append(joined_query)
    assert selection.gather.dependencies is selection.dependencies
    assert selection.owner.transport.dependencies is selection.dependencies
    assert selection.gather.dependencies == [cache, joined_query]


@pytest.mark.parametrize('overlap', [False, True])
def test_attention_joins_before_candidate_exchange_and_preserves_output(monkeypatch, overlap):
    events = []
    op = attention.PreparedDSA.__new__(attention.PreparedDSA)
    op.absorbed = torch.ones((1, 1, RECORD_BYTES), dtype=torch.bfloat16)
    op.latent = torch.ones((1, RECORD_BYTES), dtype=torch.bfloat16)
    op.slots = torch.zeros(1, dtype=torch.int64)
    op.cache = torch.zeros((1, 128, RECORD_BYTES), dtype=torch.float8_e4m3fn)
    op.metadata = SimpleNamespace(positions=torch.zeros(1, dtype=torch.int64), token_page_table=torch.zeros((1, 1)))
    op.shards = SimpleNamespace(token_slots=lambda *args: op.slots)
    op.mode = 'decode'
    op.history = None
    projected = SimpleNamespace(latent=op.latent, absorbed=op.absorbed, query_norm=torch.ones(1))
    op.heads = 1
    op.shards.world = 1
    op.project = lambda source, metadata: events.append('projection') or projected
    monkeypatch.setattr(attention, 'write_records', lambda *args: events.append('cache_write'))
    op.indexer = SimpleNamespace(project=lambda *args: events.append('index_projection') or
        (torch.zeros((1, 4096)), torch.zeros((1, 32))), sharded=None,
        bounds=((0, 1),), query=torch.zeros((1, 4096)), head_weights=torch.zeros((1, 32)))
    packed, selected = torch.ones(4), torch.ones(4)
    query = torch.zeros_like(op.absorbed, dtype=torch.float8_e4m3fn)
    gathered = torch.zeros_like(query)
    pending = []

    class Selection:
        def prepare_local(self, query, weights, positions, tables):
            events.append('local_selection')
            return packed
        def exchange(self, packed_arg, positions, **kwargs):
            assert packed_arg is packed
            assert not pending
            events.append('candidate_exchange')
            return selected
        def __call__(self, query, weights, positions, tables):
            self.prepare_local(query, weights, positions, tables)
            return self.exchange(packed, positions)

    class Fork:
        def start(self, source, **kwargs):
            events.append('query_start')
            pending.append(source)
            return source
        def finish(self, source, dependency):
            assert dependency is packed and len(pending) == 1
            gathered.copy_(pending.pop().view_as(gathered))
            events.append('query_join')
            return gathered

    def gather(source, **kwargs):
        events.append('query_gather')
        gathered.copy_(source.view_as(gathered))
        return gathered
    monkeypatch.setattr(attention, 'selection_slots', lambda *args, **kwargs: events.append('slots'))
    result = torch.zeros(1)
    def attend(query, **kwargs):
        assert not pending
        events.append('attention')
        result.copy_(gathered.float().sum().reshape(1))
        return torch.zeros((1, 1, 512)), torch.zeros((1, 1))
    op.parts = {1: dict(selector=Selection(), quantized=query, gathered=gathered, gather=gather, attention=attend,
                     owners=None, slots=None, counts=None, partial=None, lse=None,
                     combine=lambda *args: events.append('combine') or torch.zeros((1, 1, 512)), starts=None)}
    op.query_overlap = Fork() if overlap else None
    op.finish = lambda *args: result
    assert op(torch.ones((1, 1)), op.metadata).item() == RECORD_BYTES
    if overlap:
        assert events == ['projection', 'cache_write', 'query_start', 'index_projection',
                          'local_selection', 'query_join', 'candidate_exchange', 'slots', 'attention', 'combine']
    else:
        assert events == ['projection', 'cache_write', 'index_projection', 'local_selection',
                          'candidate_exchange', 'slots', 'query_gather', 'attention', 'combine']
