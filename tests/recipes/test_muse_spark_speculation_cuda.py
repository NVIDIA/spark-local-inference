# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tentative target attention commits accepted prefixes without corrupting circular history."""

import os
from types import SimpleNamespace

import pytest
import torch

from ignition.components.paged_batch import Metadata
from ignition.components.parallel import TensorParallelShard
from ignition.recipes.muse_spark.attention import AttentionScratch, PreparedAttention
from ignition.recipes.muse_spark.pool import ContinuationPool

from .test_muse_spark_attention import configuration

pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1',
                               reason='requires an explicitly assigned CUDA device')


@pytest.mark.parametrize('window', [-1, 2048])
@torch.no_grad()
def test_verify_accepted_prefix_then_decode(window):
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from transformers.models.muse_spark.modeling_muse_spark import MuseSparkRMSNorm

    device, shard = torch.device('cuda', 0), TensorParallelShard(0, 1)
    torch.manual_seed(17031)
    config = configuration(window)
    pool = ContinuationPool.allocate(config, shard, 6, 80, device)
    weights = SimpleNamespace(qkv=torch.randn(2304, 128, device=device).bfloat16() * .025,
                              output=torch.randn(128, 1024, device=device).bfloat16() * .025)
    source = torch.randn(32, 128, device=device).bfloat16()
    metadata = Metadata.allocate(32, 4, 20, device, max_query_tokens=8)
    decode_metadata = Metadata.allocate(4, 4, 20, device)
    starts = torch.tensor([127, 2047, 2048, 2049], device=device)
    states = torch.arange(1, 5, device=device)
    pages = torch.arange(80, device=device, dtype=torch.int32).view(4, 20)
    lengths = torch.tensor([8, 0, 3, 8], device=device, dtype=torch.int32)
    accepted = torch.tensor([1, 0, 2, 8], device=device)
    metadata.set(starts, lengths, states, pages, torch.zeros(4, 8, device=device, dtype=torch.int64),
                 torch.tensor([4], device=device))
    decode_lengths = torch.tensor([1, 0, 1, 1], device=device, dtype=torch.int32)
    decode_metadata.set(starts + accepted, decode_lengths, states, pages,
                        torch.zeros(4, 1, device=device, dtype=torch.int64), torch.tensor([4], device=device))
    with RecordedPreparationSession(device=device, checkpoint='muse-target-verify',
            workload_schema='muse-target-verify', numerical_policy='default', compile_workers=1) as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        scratch = AttentionScratch.allocate(config, shard, 32, device, requests=4)
        decode_scratch = AttentionScratch.allocate(config, shard, 4, device, requests=4)
        attention = PreparedAttention(0, weights, config, shard, source, scratch, metadata, pool,
            mode='verify', runtime=None, coordinator=None, session=session, name='verify', workspace=arena)
        decode = PreparedAttention(0, weights, config, shard, source[:4], decode_scratch, decode_metadata, pool,
            mode='decode', runtime=None, coordinator=None, session=session, name='decode', workspace=arena)
        session.tune()
        forwards = [compile_forward(lambda: attention(source)),
                    compile_forward(lambda: attention.commit(accepted)),
                    compile_forward(lambda: decode(source[:4]))]
        graphs = []
        for index, forward in enumerate(forwards):
            forward()
            if index == 1 and window < 0:
                graphs.append(None)
                continue
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                forward()
            graphs.append(graph)
        session.freeze()
        for kind in ('key', 'value'):
            cache = pool.cache(0, kind)
            cache.copy_(torch.randn(cache.shape, device=device).clamp(-448, 448) * .3)
        before = {kind: pool.cache(0, kind).clone() for kind in ('key', 'value')}
        graphs[0].replay()
        torch.cuda.synchronize()
        produced = {kind: values.clamp(-448, 448).to(pool.cache(0, kind).dtype)
                    for kind, values in (('key', scratch.key), ('value', scratch.value))}
        norm = MuseSparkRMSNorm(eps=1e-5, with_scale=False)
        packed = 0
        for row, length in enumerate(lengths.tolist()):
            if not length:
                continue
            start = int(starts[row])
            prefix = torch.arange(max(0, start - window) if window > 0 else 0, start, device=device)
            histories = []
            for kind in before:
                previous = (before[kind][row + 1] if window > 0
                            else before[kind][row * 20:(row + 1) * 20].flatten(0, 1))
                history = previous[prefix % window if window > 0 else prefix].float()
                histories.append(torch.cat((history, produced[kind][packed:packed + length].float())).repeat_interleave(8, 1))
            key, value = histories
            positions = torch.cat((prefix, start + torch.arange(length, device=device)))
            ends = start + torch.arange(length, device=device) + 1
            valid = positions[None] < ends[:, None]
            if window > 0:
                valid &= positions[None] >= (ends - window)[:, None]
            scores = torch.einsum('qhd,thd->hqt', scratch.query[packed:packed + length].float(), key) / 8
            attended = torch.einsum('hqt,thd->qhd', scores.masked_fill(~valid[None], -float('inf')).softmax(-1), value).bfloat16()
            gate = scratch.projected[packed:packed + length, -1024:].view(length, 16, 64)
            expected = (norm(attended) * gate.sigmoid()).flatten(1) @ weights.output.T
            torch.testing.assert_close(scratch.output[packed:packed + length], expected, atol=.02, rtol=.04)
            packed += length
        if window > 0:
            for kind in before:
                assert torch.equal(pool.cache(0, kind), before[kind])
            scratch.key.fill_(71)
            scratch.value.fill_(73)
        if graphs[1] is not None:
            graphs[1].replay()
        torch.cuda.synchronize()
        packed = 0
        for row, (length, count) in enumerate(zip(lengths.tolist(), accepted.tolist(), strict=True)):
            for kind in before:
                cache = pool.cache(0, kind)
                previous = before[kind][row + 1] if window > 0 else before[kind][row * 20:(row + 1) * 20].flatten(0, 1)
                actual = cache[row + 1] if window > 0 else cache[row * 20:(row + 1) * 20].flatten(0, 1)
                expected = previous.clone()
                positions = torch.arange(int(starts[row]), int(starts[row]) + count, device=device)
                expected[positions % window if window > 0 else positions] = produced[kind][packed:packed + count]
                if window > 0 or not length:
                    assert torch.equal(actual, expected)
                else:
                    assert torch.equal(actual[:int(starts[row]) + count], expected[:int(starts[row]) + count])
            packed += length
        committed = {kind: pool.cache(0, kind).clone() for kind in before}
        allocated = torch.cuda.memory_stats()
        graphs[2].replay()
        torch.cuda.synchronize()
        after = torch.cuda.memory_stats()
        for key in ('allocation.all.allocated', 'allocation.all.freed', 'segment.all.allocated', 'segment.all.freed'):
            assert allocated[key] == after[key]
        gate = decode_scratch.projected[:, -1024:].view(4, 16, 64)
        packed = 0
        for row, live in enumerate(decode_lengths.tolist()):
            if not live:
                for kind in before:
                    actual = pool.cache(0, kind)[row + 1] if window > 0 else pool.cache(0, kind)[row * 20:(row + 1) * 20]
                    previous = committed[kind][row + 1] if window > 0 else committed[kind][row * 20:(row + 1) * 20]
                    assert torch.equal(actual, previous)
                continue
            end = int(starts[row] + accepted[row]) + 1
            positions = torch.arange(max(0, end - window) if window > 0 else 0, end, device=device)
            histories = []
            for kind in before:
                cache = pool.cache(0, kind)
                history = cache[row + 1] if window > 0 else cache[row * 20:(row + 1) * 20].flatten(0, 1)
                histories.append(history[positions % window if window > 0 else positions].repeat_interleave(8, 1).float())
            key, value = histories
            scores = torch.einsum('hd,thd->ht', decode_scratch.query[packed].float(), key) / 8
            attended = torch.einsum('ht,thd->hd', scores.softmax(-1), value).bfloat16()
            expected = (norm(attended) * gate[packed].sigmoid()).flatten() @ weights.output.T
            torch.testing.assert_close(decode_scratch.output[packed], expected, atol=.02, rtol=.04)
            packed += 1
        for graph in graphs:
            if graph is not None:
                graph.reset()
