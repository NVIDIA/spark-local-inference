# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Packed target rows preserve long per-request windows and accepted cache records."""
import os
from copy import copy
from pathlib import Path
import pytest
import torch
from cuda.bindings import driver as cuda
from b12x.attention._shared.mla.compressed_reference import pack_deepseek_v41_cache_reference
from ignition.components.parallel import TensorParallelShard
from ignition.components.paged_batch import Metadata
from ignition.components.ring_commit import PreparedRingCommit
from ignition.preparation.aot import aot_forward
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.deepseek41.attention import AttentionScratch, PreparedAttention, Rotary
from ignition.recipes.deepseek41.pool import ContinuationPool
from ignition.recipes.deepseek41.weights import DeepSeekWeightLoader

pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1'
    or not os.environ.get('FLARE_TEST_DS41_CHECKPOINT'), reason='requires an assigned GPU and a DS41 checkpoint')

@pytest.fixture(scope='module')
def weights():
    manifest = inspect_indexed_checkpoint(Path(os.environ['FLARE_TEST_DS41_CHECKPOINT']))
    device = torch.device('cuda:0')
    with DeepSeekWeightLoader(manifest, device, shard=TensorParallelShard(0,4), prefixes=('layers.0.attn.',)) as loader:
        prefix='layers.0.attn'
        loaded = dict(qkv_a=loader.load_linear_group((prefix+'.wq_a',prefix+'.wkv')),
            q_b=loader.load_linear(prefix+'.wq_b',rows=(0,8192)),
            output=loader.load_attention_output(prefix,(0,2048)),
            q_norm=loader.load_tensor(prefix+'.q_norm.weight'),
            kv_norm=loader.load_tensor(prefix+'.kv_norm.weight'),
            sink=loader.load_tensor(prefix+'.attn_sink',extent=slice(0,16)))
        config=loader.config
    return loaded,config

class Forward:
    def __init__(self, attention, commit, source, metadata, scratch):
        self.attention, self.commit, self.source = attention, commit, source
        self.metadata, self.scratch = metadata, scratch

    def __call__(self):
        selected = torch.full_like(self.scratch.selected, -1)
        output, records, _, observed = self.attention._run(
            self.source, self.metadata, (selected, None, None))
        for name in ("query", "attended", "window_values", "query_low"):
            getattr(self.scratch, name).copy_(observed[name])
        self.scratch.output.copy_(output)
        if self.commit is not None:
            self.commit(self.metadata.lengths, staging=records[0].unsqueeze(0),
                        rows=self.metadata.state_rows, starts=self.metadata.starts)


class QueryReuseForward(Forward):
    def __init__(self, attention, commit, source, metadata, scratch, arena):
        super().__init__(attention, commit, source, metadata, scratch)
        self.arena = arena

    def __call__(self):
        super().__call__()
        # Later decoder work may immediately reuse the branch input and arena.
        self.scratch.query_low.fill_(float('nan'))
        self.arena.tensor.fill_(255)

@pytest.mark.parametrize('overlap',[False,True])
@pytest.mark.parametrize('lengths',[(6,2),(6,5,4,3,2,2,1,1),(6,4,1,1,1,1,1,1)])
@torch.no_grad()
def test_packed_attention_and_ring_commit_cross_window_boundary(weights,lengths,overlap):
    torch._dynamo.reset()
    data,config=weights
    device=torch.device('cuda:0');batch=len(lengths);tokens=sum(lengths);width=max(lengths)
    pool=ContinuationPool.allocate(config,batch+1,batch*2+1,device)
    metadata=Metadata.allocate(tokens,batch,2,device,max_query_tokens=width)
    counts=torch.tensor(lengths,dtype=torch.int32,device=device)
    starts=torch.full((batch,),126,dtype=torch.int64,device=device)
    states=torch.arange(1,batch+1,dtype=torch.int64,device=device)
    pages=torch.arange(1,batch*2+1,dtype=torch.int32,device=device).view(batch,2)
    metadata.set(starts,counts,states,pages,torch.zeros((batch,width),dtype=torch.int64,device=device),
                 torch.tensor([batch],dtype=torch.int32,device=device))
    scratch=AttentionScratch.allocate(tokens,4,device)
    source=torch.randn((tokens,5120),dtype=torch.bfloat16,device=device)*.1
    commit=PreparedRingCommit(pool.reserved_backing,[pool.layout.region('layer0.window').offset],
        metadata.state_rows,metadata.starts,token_capacity=width,window_tokens=128,record_bytes=528)
    arena=ScratchArena(device,overlap=overlap)
    with RecordedPreparationSession(device=device,checkpoint='ds41-packed-window',
            workload_schema='packed-prefix-window',numerical_policy='checkpoint',compile_workers=16) as session:
        session.register_scratch(arena)
        attention=PreparedAttention(0,data,source,scratch,metadata,pool,Rotary(config,device),mode='verify',world=4,
            session=session,name=f'window.{batch}.{tokens}',workspace=arena,window_staging=commit.staging[0])
        session.tune(show_progress=False)
        commit.functional()
        forward=Forward(attention,commit,source,metadata,scratch)
        compiled=aot_forward(forward)
        compiled(forward);torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with session.capture(),torch.cuda.graph(graph):compiled(forward)
        try:
            for _ in range(3):
                pool.clear_for_preparation()
                before=torch.cuda.memory_stats()['allocation.all.allocated']
                graph.replay();torch.cuda.synchronize()
                assert torch.cuda.memory_stats()['allocation.all.allocated']==before
                assert torch.isfinite(scratch.attended).all()
                expected=pack_deepseek_v41_cache_reference(scratch.window_values,page_size=1,cache_kind='swa').reshape(tokens,528)
                offset=0
                for row,count in enumerate(lengths):
                    for column in range(count):
                        start=((126+column)%128)*528
                        assert torch.equal(pool.window(0)[row+1,start:start+528],expected[offset+column])
                    offset+=count
        finally:graph.reset()


@pytest.fixture(scope='module')
def draft_weights():
    manifest = inspect_indexed_checkpoint(Path(os.environ['FLARE_TEST_DS41_CHECKPOINT']))
    device = torch.device('cuda:0')
    prefix = 'mtp.0.attn'
    with DeepSeekWeightLoader(manifest, device, shard=TensorParallelShard(0, 4),
            prefixes=(prefix + '.',), dspark=True) as loader:
        loaded = dict(qkv_a=loader.load_linear_group((prefix + '.wq_a', prefix + '.wkv')),
            q_b=loader.load_linear_group((prefix + '.wq_b',), row_ranges=((0, 8192),), pad=True),
            output=loader.load_attention_output(prefix, (0, 2048), pad=True),
            q_norm=loader.load_tensor(prefix + '.q_norm.weight'),
            kv_norm=loader.load_tensor(prefix + '.kv_norm.weight'),
            sink=loader.load_padded_tensor(prefix + '.attn_sink', rows=(0, 16)))
        config = loader.config
    return loaded, config


def _graph_nodes_and_edges(graph):
    raw = cuda.CUgraph(graph.raw_cuda_graph())
    status, _, _, _, edge_count = cuda.cuGraphGetEdges(raw)
    assert status == cuda.CUresult.CUDA_SUCCESS
    status, _, node_count = cuda.cuGraphGetNodes(raw)
    assert status == cuda.CUresult.CUDA_SUCCESS
    status, nodes, _ = cuda.cuGraphGetNodes(raw, node_count)
    assert status == cuda.CUresult.CUDA_SUCCESS
    status, parents, children, _, _ = cuda.cuGraphGetEdges(raw, edge_count)
    assert status == cuda.CUresult.CUDA_SUCCESS
    index = {int(node): position for position, node in enumerate(nodes)}
    return node_count, tuple(sorted((index[int(parent)], index[int(child)])
                                   for parent, child in zip(parents, children, strict=True)))


@pytest.mark.parametrize('mode,first_lengths,length_sets', (
    ('decode', (1,), ((1,), (0,), (1,))),
    ('verify', (8,), ((8,), (1,), (7,))),
    ('draft', (7,), ((7,), (1,), (6,))),
    ('verify', (8, 4), ((8, 4), (1, 0), (5, 2))),
))
@torch.no_grad()
def test_query_scheduling_preserves_native_attention_and_cache_bytes(
        weights, request, mode, first_lengths, length_sets):
    """Shared plans preserve query, output and cache bytes across prepared row modes."""
    torch._dynamo.reset()
    data, config = request.getfixturevalue('draft_weights') if mode == 'draft' else weights
    device = torch.device('cuda:0')
    batch, tokens, width = len(first_lengths), sum(first_lengths), max(first_lengths)
    pool = ContinuationPool.allocate(config, batch + 1, 3, device, dspark=mode == 'draft')
    metadata = Metadata.allocate(tokens, batch, 1, device, max_query_tokens=width)
    scratch = AttentionScratch.allocate(tokens, 4, device)
    source = torch.randn((tokens, 5120), dtype=torch.bfloat16, device=device) * .1
    counts = torch.tensor(first_lengths, dtype=torch.int32, device=device)
    starts = torch.full((batch,), 4093, dtype=torch.int64, device=device)
    states = torch.arange(1, batch + 1, dtype=torch.int64, device=device)
    pages = torch.ones((batch, 1), dtype=torch.int32, device=device)
    inputs = torch.zeros((batch, width), dtype=torch.int64, device=device)
    live = torch.tensor([batch], dtype=torch.int32, device=device)
    metadata.set(starts, counts, states, pages, inputs, live)
    commit = (PreparedRingCommit(pool.reserved_backing, [pool.layout.region('layer0.window').offset],
        metadata.state_rows, metadata.starts, token_capacity=width, window_tokens=128, record_bytes=528)
        if mode == 'verify' else None)
    arena = ScratchArena(device, overlap=True)
    graphs, owners = {}, []
    with RecordedPreparationSession(device=device, checkpoint='ds41-query-overlap',
            workload_schema='packed-native-query', numerical_policy='checkpoint', compile_workers=16) as session:
        session.register_scratch(arena)
        attention = PreparedAttention(0, data, source, scratch, metadata, pool, Rotary(config, device),
            mode=mode, world=4, session=session, name='attention', workspace=arena,
            window_staging=None if commit is None else commit.staging[0])
        assert (attention.concurrent_query is not None) is (mode != 'decode')
        serial = copy(attention)
        serial.projections = dict(attention.projections)
        serial.concurrent_query = None
        if commit is not None:
            commit.functional()
        session.tune(show_progress=False)
        for name, operation in (('serial', serial), ('overlap', attention)):
            forward = QueryReuseForward(operation, commit, source, metadata, scratch, arena)
            compiled = aot_forward(forward)
            compiled(forward)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with session.capture(), torch.cuda.graph(graph):
                compiled(forward)
            graph.instantiate()
            graphs[name] = graph
            owners.extend((forward, compiled))
        if mode == 'decode':
            assert _graph_nodes_and_edges(graphs['serial']) == _graph_nodes_and_edges(graphs['overlap'])
        session.freeze()
        bundle = PreparedGraphBundle(device=0, graphs=graphs, buffers={'source': source},
            resources=(data, pool, scratch, metadata, commit, arena, session, owners))
        worker = bundle.start_worker()
        try:
            for lengths in length_sets:
                counts.copy_(torch.tensor(lengths, dtype=torch.int32, device=device))
                metadata.set(starts, counts, states, pages, inputs, live)
                source.normal_(std=.1)
                pool.clear_for_preparation()
                window = pool.window(0, draft=mode == 'draft')
                history = torch.randn((window.shape[0] * 128, 512), dtype=torch.bfloat16, device=device) * .1
                window.copy_(pack_deepseek_v41_cache_reference(history, page_size=128,
                             cache_kind='swa').reshape_as(window))
                saved = pool.backing.clone()
                expected = None
                for name in ('serial', 'overlap', 'overlap', 'serial'):
                    pool.backing.copy_(saved)
                    arena.tensor.fill_(0xA5)
                    scratch.output.fill_(float('nan'))
                    torch.cuda.synchronize()
                    before = torch.cuda.memory_stats()
                    ticket = worker.submit(bundle.graph_indices[name])
                    result = worker.poll(timeout_ms=120000)
                    assert result is not None and result.ticket == ticket and result.error is None
                    after = torch.cuda.memory_stats()
                    for counter in ('allocation.all.allocated', 'allocation.all.freed',
                                    'segment.all.allocated', 'segment.all.freed'):
                        assert before[counter] == after[counter]
                    actual = (scratch.query, scratch.attended, scratch.output, pool.backing)
                    if expected is None:
                        expected = tuple(value.clone() for value in actual)
                    else:
                        for value, reference in zip(actual, expected, strict=True):
                            assert torch.equal(value, reference)
        finally:
            worker.close()
            for graph in graphs.values():
                graph.reset()
