# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Captured Glimmer attention preserves BF16 cache pages, masks and inactive rows."""

import os
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1',
    reason='requires an explicitly assigned CUDA device')


def configuration(window):
    from ignition.recipes.muse_glimmer.config import ModelConfig
    return ModelConfig(dict(hidden_size=256, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, head_dim=128, rms_norm_eps=1e-5, qk_scale_factor=3.87,
        layer_rope_theta=[500000 if window > 0 else 0], sliding_window=window,
        layer_types=['sliding_attention' if window > 0 else 'full_attention']), {})


@pytest.mark.parametrize('window', [-1, 256])
@torch.no_grad()
def test_chunked_prefill_matches_canonical_cache_and_gate(window):
    from ignition.components.paged_batch import Metadata
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.muse_glimmer.attention import AttentionScratch, PreparedAttention
    from ignition.recipes.muse_glimmer.pool import ContinuationPool
    from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerRMSNorm, apply_rotary_pos_emb

    device, capacity, blocks = torch.device('cuda', 0), 129, 5
    torch.manual_seed(2381)
    config = configuration(window)
    pool = ContinuationPool.allocate(config, 3, blocks, device)
    weights = SimpleNamespace(qkv=torch.randn((1536, 256), device=device).bfloat16() * .025,
                             output=torch.randn((256, 512), device=device).bfloat16() * .025)
    source = torch.randn((capacity, 256), device=device).bfloat16()
    metadata = Metadata.allocate(capacity, 1, blocks, device)
    pages = torch.arange(blocks, dtype=torch.int32, device=device)[None]
    def set_metadata(start, count):
        metadata.set(torch.tensor([start], device=device), torch.tensor([count], dtype=torch.int32, device=device),
            torch.tensor([1], device=device), pages, torch.zeros((1, capacity), dtype=torch.int64, device=device),
            torch.tensor([1], device=device))
    set_metadata(0, capacity)
    with RecordedPreparationSession(device=device, checkpoint='glimmer-attention', workload_schema='glimmer-attention',
            numerical_policy='default', compile_workers=1, autotune_effort='off') as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        scratch = AttentionScratch.allocate(config, capacity, device)
        attention = PreparedAttention(0, weights, config, source, scratch, metadata, pool,
            mode='prefill', session=session, name='prefill', workspace=arena)
        session.tune()
        compiled = compile_forward(lambda: attention(source))
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        pool.clear_for_preparation()
        norm = MuseGlimmerRMSNorm(eps=1e-5, with_scale=False)
        keys, values, start = [], [], 0
        for count in (127, 1, 1, 126, 1, 1, 0, 3):
            source.normal_()
            set_metadata(start, count)
            q, k, v, gate = (source @ weights.qkv.T).split((512, 256, 256, 512), -1)
            q, k = norm(q.view(capacity, 4, 128)) * 3.87, norm(k.view(capacity, 2, 128))
            v = v.view(capacity, 2, 128)
            if window > 0:
                inv = 1 / (500000 ** (torch.arange(0, 128, 2, device=device).float() / 128))
                freq = (start + torch.arange(capacity, device=device)).float()[:, None] * inv
                angles = torch.cat((freq, freq), -1)
                q_ref, k_ref = apply_rotary_pos_emb(q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                    angles.cos().bfloat16()[None], angles.sin().bfloat16()[None])
                q, k = q_ref[0].transpose(0, 1), k_ref[0].transpose(0, 1)
            keys.append(k[:count])
            values.append(v[:count])
            if count:
                ks, vs = torch.cat(keys).repeat_interleave(2, 1).float(), torch.cat(values).repeat_interleave(2, 1).float()
                scores = torch.einsum('qhd,thd->hqt', q[:count].float(), ks) / 128**.5
                ends = start + torch.arange(count, device=device) + 1
                positions = torch.arange(ks.shape[0], device=device)
                valid = positions[None] < ends[:, None]
                if window > 0:
                    valid &= positions[None] >= (ends - window)[:, None]
                attended = torch.einsum('hqt,thd->qhd', scores.masked_fill(~valid[None], -torch.inf).softmax(-1), vs).bfloat16()
                expected = (attended.flatten(1) * gate[:count].sigmoid()) @ weights.output.T
            before = {kind: pool.cache(0, kind).clone() for kind in ('key', 'value')}
            allocations = torch.cuda.memory_stats()['allocation.all.allocated']
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocations
            if count:
                # BF16 attention rounding: measured max abs 0.001953125, relative L2 0.00312.
                torch.testing.assert_close(scratch.output[:count], expected, atol=.002, rtol=.004)
            for kind, produced in [('key', scratch.key), ('value', scratch.value)]:
                cache = pool.cache(0, kind).flatten(0, 1)
                assert torch.equal(cache[start:start + count], produced[:count])
                assert torch.equal(cache[:start], before[kind].flatten(0, 1)[:start])
                assert torch.equal(cache[start + count:], before[kind].flatten(0, 1)[start + count:])
            start += count
        graph.reset()
