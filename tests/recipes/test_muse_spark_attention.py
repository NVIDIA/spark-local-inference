# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse Spark paged attention preserves canonical normalization and FP8 histories."""

import os
from types import SimpleNamespace

import pytest
import torch

from ignition.components.paged_batch import Metadata
from ignition.components.parallel import TensorParallelShard
from ignition.recipes.muse_spark.attention import AttentionScratch, PreparedAttention
from ignition.recipes.muse_spark.pool import ContinuationPool, PAGE_TOKENS, page_bytes


class Geometry:
    heads, kv_heads, key_dim, value_dim, theta = 16, 2, 64, 64, 500000

    def __init__(self, window):
        self.window, self.rotary_dim = window, 64 if window > 0 else 0

    def local(self, shard):
        return self.heads, self.kv_heads


def configuration(window, hidden=128):
    return SimpleNamespace(layers=1, attention=lambda layer: Geometry(window),
        text=dict(hidden_size=hidden, rms_norm_eps=1e-5, qk_scale_factor=5.4730064863875))


def test_fp8_page_storage_and_complete_continuation():
    config, shard = configuration(-1), TensorParallelShard(0, 1)
    pool = ContinuationPool.allocate(config, shard, 3, 4, 'cpu')
    assert page_bytes(config, shard) == 2 * PAGE_TOKENS * 2 * 64
    assert pool.cache(0, 'key').shape == (4, 128, 2, 64)
    assert pool.layout.region('layer0.key').nbytes == PAGE_TOKENS * 2 * 64
    assert pool.layout.region('last_hidden').shape == (128,)
    assert pool.layout.region('consumed_tokens').dtype == torch.int64


@pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1', reason='requires assigned CUDA GPU')
@pytest.mark.parametrize('window', [-1, 2048])
@pytest.mark.parametrize('checkpoint,capacity', [(False, 64), (False, 129), (True, 129), (True, 4096)])
def test_attention_matches_canonical_across_cache_boundaries(window, checkpoint, capacity):
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from transformers.models.muse_spark.modeling_muse_spark import MuseSparkRMSNorm, apply_rotary_pos_emb

    device, shard = torch.device('cuda', 0), TensorParallelShard(0, 1)
    torch.manual_seed(2381)
    hidden = 8192 if checkpoint else 128
    config = configuration(window, hidden)
    blocks = max(20, (capacity + 128) // 128 + 1)
    pool = ContinuationPool.allocate(config, shard, 3, blocks, device)
    if checkpoint:
        import json
        from pathlib import Path
        from safetensors import safe_open
        snapshot = os.environ.get('FLARE_MUSE_SPARK_CHECKPOINT')
        if snapshot is None:
            pytest.skip('requires FLARE_MUSE_SPARK_CHECKPOINT')
        snapshot = Path(snapshot)
        index = json.loads((snapshot / 'model.safetensors.index.json').read_text())['weight_map']
        def load(name, rows=None, columns=None):
            key = f'model.language_model.layers.0.self_attn.{name}.weight'
            with safe_open(snapshot / index[key], framework='pt') as file:
                return file.get_slice(key)[:rows, :columns].to(device)
        weights = SimpleNamespace(qkv=torch.cat([load('q_proj', 1024), load('k_proj', 128),
            load('v_proj', 128), load('gate_proj', 1024)]), output=load('o_proj', columns=1024))
    else:
        weights = SimpleNamespace(qkv=torch.randn((2304, hidden), device=device).bfloat16() * .025,
                                 output=torch.randn((hidden, 1024), device=device).bfloat16() * .025)
    source = torch.randn((capacity, hidden), device=device).bfloat16()
    metadata = Metadata.allocate(capacity, 1, blocks, device)
    metadata.set(torch.tensor([0], device=device), torch.tensor([capacity], dtype=torch.int32, device=device),
        torch.tensor([1], device=device), torch.arange(blocks, dtype=torch.int32, device=device)[None],
        torch.zeros((1, capacity), device=device, dtype=torch.int64), torch.tensor([1], device=device))
    with RecordedPreparationSession(device=device, checkpoint='muse-attention-component',
            workload_schema='muse-attention', numerical_policy='default', compile_workers=1) as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        scratch = AttentionScratch.allocate(config, shard, capacity, device)
        attention = PreparedAttention(0, weights, config, shard, source, scratch, metadata, pool,
            mode='prefill', runtime=None, coordinator=None, session=session, name='prefill', workspace=arena)
        session.tune()
        compiled = compile_forward(lambda: attention(source))
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        pool.clear_for_preparation()
        keys, values = [], []
        stored = dict(key=[], value=[])
        norm = MuseSparkRMSNorm(eps=1e-5, with_scale=False)
        start = 0
        chunks = ((63, 1, 1, 62, 1, 1, *([64] * 29), 62, 1, 1, 0, 3) if capacity == 64 else
                  (127, 1, 1, *([129] * 14), 112, 1, 1, 0, 3) if capacity == 129 else (4096, 0, 3))
        for count in chunks:
            source.normal_()
            metadata.set(torch.tensor([start], device=device), torch.tensor([count], dtype=torch.int32, device=device),
                torch.tensor([1], device=device), torch.arange(blocks, dtype=torch.int32, device=device)[None],
                torch.zeros((1, capacity), device=device, dtype=torch.int64), torch.tensor([1], device=device))
            projected = source @ weights.qkv.T
            q, k, v, gate = projected.split((1024, 128, 128, 1024), -1)
            q = norm(q.view(capacity, 16, 64)) * config.text['qk_scale_factor']
            k, v = norm(k.view(capacity, 2, 64)), v.view(capacity, 2, 64)
            if window > 0:
                inv = 1 / (500000 ** (torch.arange(0, 64, 2, device=device).float() / 64))
                freq = (start + torch.arange(capacity, device=device)).float()[:, None] * inv
                cos, sin = torch.cat((freq, freq), -1).cos().bfloat16(), torch.cat((freq, freq), -1).sin().bfloat16()
                q_ref, k_ref = apply_rotary_pos_emb(q.transpose(0, 1)[None], k.transpose(0, 1)[None], cos[None], sin[None])
                q, k = q_ref[0].transpose(0, 1), k_ref[0].transpose(0, 1)
            keys.append(k[:count].clamp(-448, 448).to(torch.float8_e4m3fn).bfloat16())
            values.append(v[:count].clamp(-448, 448).to(torch.float8_e4m3fn).bfloat16())
            full_k, full_v = torch.cat(keys), torch.cat(values)
            expected = None
            if count:
                ks, vs = full_k.repeat_interleave(8, 1).float(), full_v.repeat_interleave(8, 1).float()
                scores = torch.einsum('qhd,thd->hqt', q[:count].float(), ks) / 8
                ends = start + torch.arange(count, device=device) + 1
                positions = torch.arange(full_k.shape[0], device=device)
                valid = positions[None] < ends[:, None]
                if window > 0:
                    valid &= positions[None] >= (ends - window)[:, None]
                prob = scores.masked_fill(~valid[None], float('-inf')).softmax(-1)
                attended = torch.einsum('hqt,thd->qhd', prob, vs).bfloat16()
                gated = norm(attended).flatten(1) * gate[:count].sigmoid()
                expected = gated @ weights.output.T
            allocated = torch.cuda.memory_allocated(device)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated(device) == allocated
            for kind, produced in (('key', scratch.key), ('value', scratch.value)):
                stored[kind].append(produced[:count].clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8).clone())
            if count:
                torch.testing.assert_close(scratch.output[:count], expected, atol=.02, rtol=.04)
            if window > 0:
                positions = torch.arange(max(0, start + count - window), start + count, device=device)
                for kind in stored:
                    expected_history = torch.cat(stored[kind])[-window:]
                    actual_history = pool.cache(0, kind)[1, positions % window].view(torch.uint8)
                    assert torch.equal(actual_history, expected_history)
                assert pool.cache(0, 'key')[0].count_nonzero() == pool.cache(0, 'key')[2].count_nonzero() == 0
            if count:
                torch.testing.assert_close(scratch.key[:count], k[:count], rtol=0, atol=.03125)
                slots = (torch.arange(start, start + count, device=device) % window if window > 0
                         else torch.arange(start, start + count, device=device))
                actual_key = (pool.cache(0, 'key')[1, slots] if window > 0
                              else pool.cache(0, 'key').flatten(0, 1)[slots])
                actual_value = (pool.cache(0, 'value')[1, slots] if window > 0
                                else pool.cache(0, 'value').flatten(0, 1)[slots])
                retained = min(count, window) if window > 0 else count
                expected_key = scratch.key[count-retained:count].clamp(-448, 448).to(torch.float8_e4m3fn)
                expected_value = scratch.value[count-retained:count].clamp(-448, 448).to(torch.float8_e4m3fn)
                assert torch.equal(actual_key[-retained:].view(torch.uint8), expected_key.view(torch.uint8))
                assert torch.equal(actual_value[-retained:].view(torch.uint8), expected_value.view(torch.uint8))
            start += count
        assert start > 2049
        graph.reset()


@pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1', reason='requires assigned CUDA GPU')
@pytest.mark.parametrize('window', [-1, 2048])
def test_decode_wrap_and_inactive_lanes(window):
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from transformers.models.muse_spark.modeling_muse_spark import MuseSparkRMSNorm

    device, shard = torch.device('cuda', 0), TensorParallelShard(0, 1)
    config = configuration(window)
    pool = ContinuationPool.allocate(config, shard, 5, 80, device)
    weights = SimpleNamespace(qkv=torch.randn((2304, 128), device=device).bfloat16() * .025,
                             output=torch.randn((128, 1024), device=device).bfloat16() * .025)
    source = torch.randn((4, 128), device=device).bfloat16()
    metadata = Metadata.allocate(4, 4, 20, device)
    pages = torch.arange(80, device=device, dtype=torch.int32).view(4, 20)
    starts = torch.tensor([127, 2047, 2048, 2049], device=device)
    states = torch.arange(1, 5, device=device)
    def set_counts(counts):
        metadata.set(starts, torch.tensor(counts, dtype=torch.int32, device=device), states, pages,
            torch.zeros((4, 1), device=device, dtype=torch.int64), torch.tensor([4], device=device))
    set_counts([1, 0, 1, 1])
    with RecordedPreparationSession(device=device, checkpoint='muse-decode-component',
            workload_schema='muse-decode', numerical_policy='default', compile_workers=1) as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        scratch = AttentionScratch.allocate(config, shard, 4, device, requests=4)
        attention = PreparedAttention(0, weights, config, shard, source, scratch, metadata, pool,
            mode='decode', runtime=None, coordinator=None, session=session, name='decode', workspace=arena)
        session.tune()
        compiled = compile_forward(lambda: attention(source))
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        for kind in ('key', 'value'):
            cache = pool.cache(0, kind)
            cache.copy_(torch.randn(cache.shape, device=device) * .3)
        norm = MuseSparkRMSNorm(eps=1e-5, with_scale=False)
        for counts in ([1, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 1]):
            set_counts(counts)
            before = {kind: pool.cache(0, kind).clone() for kind in ('key', 'value')}
            allocated = torch.cuda.memory_allocated(device)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated(device) == allocated
            projected = source @ weights.qkv.T
            q, k, v, gate = projected.split((1024, 128, 128, 1024), -1)
            torch.testing.assert_close(scratch.value[:sum(counts)], v[:sum(counts)].view(-1, 2, 64), atol=.002, rtol=.02)
            packed = 0
            for row, count in enumerate(counts):
                if not count:
                    for kind in ('key', 'value'):
                        cache = pool.cache(0, kind)
                        actual = cache[row + 1] if window > 0 else cache[row * 20:(row + 1) * 20]
                        previous = before[kind][row + 1] if window > 0 else before[kind][row * 20:(row + 1) * 20]
                        torch.testing.assert_close(actual, previous, rtol=0, atol=0)
                    continue
                length = min(int(starts[row]) + 1, window) if window > 0 else int(starts[row]) + 1
                caches = []
                for kind, produced in (('key', scratch.key), ('value', scratch.value)):
                    cache = pool.cache(0, kind)
                    history = cache[row + 1] if window > 0 else cache[row * 20:(row + 1) * 20].flatten(0, 1)
                    slot = int(starts[row]) % window if window > 0 else int(starts[row])
                    quantized = produced[packed].clamp(-448, 448).to(torch.float8_e4m3fn)
                    assert torch.equal(history[slot].view(torch.uint8), quantized.view(torch.uint8))
                    caches.append(history[:length].repeat_interleave(8, 1).float())
                ks, vs = caches
                scores = torch.einsum('hd,thd->ht', scratch.query[packed].float(), ks) / 8
                attended = torch.einsum('ht,thd->hd', scores.softmax(-1), vs).bfloat16()
                gated = norm(attended).flatten() * gate[packed].sigmoid()
                expected = gated @ weights.output.T
                torch.testing.assert_close(scratch.output[packed], expected, atol=.02, rtol=.04)
                packed += 1
            starts.add_(torch.tensor(counts, device=device))
        graph.reset()
