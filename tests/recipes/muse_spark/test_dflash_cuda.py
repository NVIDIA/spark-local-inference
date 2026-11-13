# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Captured DFlash context injection and FP8 private-block attention."""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1', reason='requires assigned CUDA GPU')


def norm(x, w):
    y = x.float()
    return (y * (y.square().mean(-1, keepdim=True) + 1e-6).rsqrt() * w.float()).bfloat16()


def rotary(x, positions):
    inv = 500000. ** (-torch.arange(0, 64, 2, device=x.device).float() / 64)
    phase = positions.clamp_min(0).float()[:, None] * inv
    phase = torch.cat((phase, phase), -1)
    c, s = phase.cos().bfloat16()[:, None], phase.sin().bfloat16()[:, None]
    a, b = x.chunk(2, -1)
    return x * c + torch.cat((-b, a), -1) * s


def fp8(x):
    return x.clamp(-448, 448).to(torch.float8_e4m3fn)


class Geometry:
    window, theta, rotary_dim, key_dim, value_dim = 16, 500000., 64, 64, 64

    def local(self, shard):
        return 4, 2


class Pool:
    def __init__(self):
        self.values = {f'draft{i}.{kind}': torch.zeros((6, 16, 2, 64), device='cuda', dtype=torch.float8_e4m3fn)
                       for i in range(5) for kind in ('key', 'value')}

    def reserved_region(self, name):
        return self.values[name]


def fixture():
    from ignition.components.parallel import TensorParallelShard
    from ignition.components.weights import MLPWeights
    torch.manual_seed(821)
    def rand(*shape):
        return (torch.randn(shape, device='cuda') * .04).bfloat16()
    def ones(n):
        return torch.ones(n, device='cuda', dtype=torch.bfloat16)
    text = dict(hidden_size=256, intermediate_size=512, vocab_size=128, rms_norm_eps=1e-6,
        dflash_config=dict(selector_top_k=16, output_multiplier=.17677669529663687, final_logit_softcapping=20.))
    config = SimpleNamespace(text=text, attention=Geometry(), block_size=8, layers=5)
    layers = [SimpleNamespace(qkv=rand(512, 256), output=rand(256, 256), query_norm=ones(64),
        key_norm=ones(64), input_norm=ones(256), post_attention_norm=ones(256),
        mlp=MLPWeights(rand(1024, 256), rand(256, 512)),
        attention_conv=dict(base=rand(2, 2, 256), projection=rand(64, 256)),
        mlp_conv=dict(base=rand(2, 2, 256), projection=rand(64, 256))) for _ in range(5)]
    draft = SimpleNamespace(projection=rand(256, 1280), hidden_norm=ones(256),
        context_kv=rand(1280, 256), norm=ones(256), layers=layers,
        selector=dict(projection=rand(256, 256), predecessor=rand(128, 256), successor=rand(128, 256)))
    return SimpleNamespace(config=SimpleNamespace(draft=config), draft=draft, shard=TensorParallelShard(0, 1),
                           embedding=rand(128, 256), head=rand(128, 256))


def metadata(batch, width=8):
    from ignition.components.paged_batch import Metadata
    m = Metadata.allocate(batch * width, batch, 1, 'cuda', samples=batch * width)
    def set_values(starts, lengths):
        m.set(torch.tensor(starts, device='cuda', dtype=torch.int64),
              torch.tensor(lengths, device='cuda', dtype=torch.int32),
              torch.arange(1, batch + 1, device='cuda'), torch.zeros((batch, 1), device='cuda', dtype=torch.int32),
              torch.arange(batch * width, device='cuda').view(batch, width) % 128,
              torch.tensor([batch], device='cuda', dtype=torch.int32))
    set_values([15] * batch, [width] * batch)
    return m, set_values


@pytest.mark.parametrize("width", [8, 32])
@torch.no_grad()
def test_context_accepted_fp8_capture(width):
    from b12x.preparation import PreparationSession
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import GraphBufferPool, ScratchArena
    from ignition.recipes.muse_spark.dflash import PreparedContext
    weights, pool = fixture(), Pool()
    m, set_values = metadata(4, width)
    tokens = 4 * width
    aux = torch.randn((tokens, 1280), device='cuda', dtype=torch.bfloat16)
    counts = torch.tensor([width, 3, 0, 5], device='cuda')
    buffers = GraphBufferPool('cuda')
    with PreparationSession(device='cuda:0', autotune=False) as session:
        arena = ScratchArena('cuda')
        context = PreparedContext(weights, aux, m, pool, session=session, name='ctx', workspace=arena,
                                  runtime=None, coordinator=None, buffers=buffers.cursor())
        arena.finalize()
        fn = compile_forward(lambda: context(aux, counts))
        fn()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            fn()
        for starts, lengths, accepted in [([15, 16, 17, 31], [width]*4, [width, 3, 0, 5]),
                                          ([33, 44, 0, 0], [width, width, 0, 0], [1, width-1, 0, 0])]:
            set_values(starts, lengths)
            counts.copy_(torch.tensor(accepted, device='cuda'))
            aux.normal_()
            for value in pool.values.values():
                value.fill_(7)
            hidden = norm(F.linear(aux, weights.draft.projection), weights.draft.hidden_norm)
            kv = F.linear(hidden, weights.draft.context_kv).view(tokens, 5, 256)
            expected = {name: value.clone() for name, value in pool.values.items()}
            for i in range(5):
                key = fp8(rotary(norm(kv[:, i, :128].reshape(tokens, 2, 64), weights.draft.layers[i].key_norm), m.positions))
                val = fp8(kv[:, i, 128:].reshape(tokens, 2, 64))
                for t in range(sum(lengths)):
                    r, pos = int(m.requests[t]), int(m.positions[t])
                    if max(0, accepted[r]-16) <= pos - starts[r] < accepted[r]:
                        expected[f'draft{i}.key'][r+1, pos % 16] = key[t]
                        expected[f'draft{i}.value'][r+1, pos % 16] = val[t]
            before = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == before
            for name, value in pool.values.items():
                torch.testing.assert_close(value.float(), expected[name].float(), rtol=0, atol=0)
        graph.reset()


@torch.no_grad()
def test_private_attention_alternating_capacities():
    from b12x.preparation import PreparationSession
    from ignition.preparation.scratch import GraphBufferPool, ScratchArena
    from ignition.recipes.muse_spark.dflash import DraftScratch, DraftAttention
    weights, pool = fixture(), Pool()
    for value in pool.values.values():
        value.copy_(fp8(torch.randn(value.shape, device='cuda') * .2))
    original = {name: value.clone() for name, value in pool.values.items()}
    buffers = GraphBufferPool('cuda')
    scratch = DraftScratch.allocate(weights.config.draft, weights.shard, 32, 'cuda', buffers=buffers.cursor())
    with PreparationSession(device='cuda:0', autotune=False) as session:
        arena = ScratchArena('cuda')
        prepared = []
        for batch in (1, 4):
            m, setter = metadata(batch)
            s = scratch.rows(batch*8, batch*8)
            attention = DraftAttention(0, weights.draft.layers[0], s.normalized, weights.config.draft,
                weights.shard, s, m, pool, runtime=None, coordinator=None, session=session,
                name=f'attn{batch}', workspace=arena, buffers=buffers.cursor())
            prepared.append((s, m, setter, attention))
        arena.finalize()
        graphs = []
        for s, m, setter, op in prepared:
            op(s.normalized)
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                op(s.normalized)
            graphs.append(graph)
        for turn in range(4):
            index = turn % 2
            s, m, setter, op = prepared[index]
            batch = len(m.lengths)
            lengths = [8] * batch
            if batch > 1:
                lengths[-1] = 0
            setter([17 + turn] * batch, lengths)
            s.normalized.normal_()
            q, k, v = F.linear(s.normalized, weights.draft.layers[0].qkv).split((256, 128, 128), -1)
            q = rotary(norm(q.reshape(-1, 4, 64), weights.draft.layers[0].query_norm), m.positions)
            k = fp8(rotary(norm(k.reshape(-1, 2, 64), weights.draft.layers[0].key_norm), m.positions)).float()
            v = fp8(v.reshape(-1, 2, 64)).float()
            expected = []
            offset = 0
            for r, count in enumerate(lengths):
                if not count:
                    continue
                start = int(m.starts[r])
                positions = torch.arange(start-16, start, device='cuda') % 16
                keys = torch.cat((original['draft0.key'][r+1, positions].float(), k[offset:offset+count]))
                vals = torch.cat((original['draft0.value'][r+1, positions].float(), v[offset:offset+count]))
                scores = torch.einsum('qhd,khd->hqk', q[offset:offset+count].float(), keys.repeat_interleave(2, 1)) / 8
                visible = torch.arange(keys.shape[0], device='cuda')[None] >= (1 + torch.arange(count, device='cuda'))[:, None]
                scores.masked_fill_(~visible[None], -torch.inf)
                att = torch.einsum('hqk,khd->qhd', scores.softmax(-1), vals.repeat_interleave(2, 1)).bfloat16()
                expected.append(F.linear(att.flatten(1), weights.draft.layers[0].output))
                offset += count
            before = torch.cuda.memory_allocated()
            graphs[index].replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == before
            torch.testing.assert_close(s.output[:offset], torch.cat(expected), rtol=.03, atol=.025)
            assert torch.equal(op.pages, torch.arange(batch, device='cuda', dtype=torch.int32)[:, None])
            for name, value in pool.values.items():
                assert torch.equal(value.view(torch.uint8), original[name].view(torch.uint8))
        for graph in graphs:
            graph.reset()



@torch.no_grad()
def test_five_layer_draft_compiled_replay(tmp_path):
    from b12x.preparation import PreparationSession
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import GraphBufferPool, ScratchArena
    from ignition.recipes.muse_spark.dflash import DraftScratch, PreparedDraft
    weights, pool = fixture(), Pool()
    buffers = GraphBufferPool('cuda')
    scratch = DraftScratch.allocate(weights.config.draft, weights.shard, 32, 'cuda', buffers=buffers.cursor())
    before_pool = {name: value.clone() for name, value in pool.values.items()}
    with PreparationSession(device='cuda:0', autotune=False) as session:
        arena = ScratchArena('cuda')
        prepared = []
        for batch in (4, 1):
            m, setter = metadata(batch)
            op = PreparedDraft(weights, scratch.rows(batch*8, batch*8), m, pool,
                runtime=None, vocabulary_runtime=None, embedding_runtime=None, coordinator=None,
                session=session, name=f'draft{batch}', workspace=arena, buffers=buffers.cursor())
            prepared.append((m, setter, op))
        arena.finalize()
        graphs = []
        for m, setter, op in prepared:
            fn = compile_forward(op)
            fn()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                fn()
            graphs.append(graph)
        for turn in range(4):
            index = turn % 2
            m, setter, op = prepared[index]
            batch = len(m.lengths)
            starts = [18+turn] * batch
            setter(starts, [8] * batch)
            ids = torch.randint(0, 128, (batch, 8), device='cuda')
            m.token_ids.copy_(ids.flatten())
            op()
            expected_hidden = op.scratch.hidden.clone()
            expected_logits = op.scratch.logits.clone()
            expected_selector = op.selector.projected.clone()
            head = F.linear(expected_hidden, weights.head).float()
            transformed = 20 * torch.tanh(head * .17677669529663687 / 20)
            torch.testing.assert_close(expected_logits, transformed, rtol=0, atol=0)
            coded = F.linear(expected_hidden, weights.draft.selector['projection'])
            torch.testing.assert_close(expected_selector.flatten(0, 1), coded, rtol=0, atol=0)
            before = torch.cuda.memory_allocated()
            graphs[index].replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == before
            torch.testing.assert_close(op.scratch.hidden, expected_hidden, rtol=0, atol=0)
            torch.testing.assert_close(op.scratch.logits, expected_logits, rtol=2e-6, atol=2e-6)
            torch.testing.assert_close(op.selector.projected, expected_selector, rtol=0, atol=0)
            unary = op.scratch.logits.view(batch, 8, -1)[:, 3].clone()
            predecessor = ids[:, 2]
            shortlist, candidates = unary.topk(16, sorted=False)
            coded = weights.draft.selector['predecessor'][predecessor] * expected_selector[:, 3]
            correction = torch.bmm(weights.draft.selector['successor'][candidates], coded[:, :, None]).squeeze(-1)
            selected = torch.full_like(unary, -torch.inf)
            selected.scatter_(1, candidates, shortlist + correction.float())
            op.selector(unary, predecessor, 3)
            torch.testing.assert_close(unary, selected, rtol=0, atol=0)
            for name, value in pool.values.items():
                assert torch.equal(value.view(torch.uint8), before_pool[name].view(torch.uint8))
        assert len(scratch.selectors) == 2
        from ignition.preparation.disposable import storage_ranges

        disposable = dict(storage_ranges(scratch.scratch_tensors()))
        for _, _, op in prepared:
            assert op.selector.projected.untyped_storage().data_ptr() in disposable
            assert dict(storage_ranges(op.scratch_tensors())) == disposable
        assert weights.head.untyped_storage().data_ptr() not in disposable
        for graph in graphs:
            graph.reset()
