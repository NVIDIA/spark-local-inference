# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual DFlash2 context and five-layer proposals on matched target features."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1',
    reason='requires assigned CUDA_VISIBLE_DEVICES and Muse Spark checkpoints')

CACHE = Path('/data/cache/huggingface/hub')
TARGET = CACHE / 'models--meta-models--Muse-Spark-Open-816B-A42B-NVFP4/snapshots/4239e4db19376904f2e82c167e78f70abb5455fe'
DRAFT = CACHE / 'models--nvidia--Muse-Spark-Open-816B-A42B-DFlash2/snapshots/fbec887a79866e9ecff6c57adfd9fd02aa3cadea'
FEATURES = Path('/tmp/muse_tp8_fixed_routes_layers.pt')


def rms(x, weight):
    y = x.float()
    return weight * (y * (y.square().mean(-1, keepdim=True) + 1e-6).rsqrt()).to(x.dtype)


def rotary(x, positions):
    inverse = 500000. ** (-torch.arange(0, 64, 2, device=x.device).float() / 64)
    angle = positions.float()[:, None] * inverse
    angle = torch.cat((angle, angle), -1)
    left, right = x.chunk(2, -1)
    return x * angle.cos().bfloat16()[:, None] + torch.cat((-right, left), -1) * angle.sin().bfloat16()[:, None]


def fp8(x):
    return x.clamp(-448, 448).to(torch.float8_e4m3fn)


def convolve(x, delta, base):
    """Apply the serving convolution with BF16 coefficient addition before products."""
    grouped = x.reshape(-1, 512, 16)
    coefficients = base.reshape(1, 2, 512, 16) + delta.unsqueeze(-1)
    result = coefficients[:, 0] * grouped
    shifted = F.pad(grouped[:-1], (0, 0, 0, 0, 1, 0))
    valid = (torch.arange(x.shape[0], device=x.device) % 8 >= 1).reshape(-1, 1, 1)
    result += coefficients[:, 1] * shifted * valid
    return result.flatten(1)


def conv_prepare(x, weights):
    delta = F.linear(x, weights['projection']).reshape(-1, 2, 2, 512)
    return convolve(x, delta[:, 0], weights['base'][0]), delta[:, 1]


def attention(x, layer, history, positions):
    q, k, v = F.linear(x, layer.qkv).split((8192, 1024, 1024), -1)
    q = rotary(rms(q.reshape(-1, 128, 64), layer.query_norm), positions)
    k = fp8(rotary(rms(k.reshape(-1, 16, 64), layer.key_norm), positions)).bfloat16()
    v = fp8(v.reshape(-1, 16, 64)).bfloat16()
    k = torch.cat((history[0].bfloat16(), k)).repeat_interleave(8, 1)
    v = torch.cat((history[1].bfloat16(), v)).repeat_interleave(8, 1)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        attended = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                                                  v.transpose(0, 1)[None], is_causal=False)
    return F.linear(attended[0].transpose(0, 1).reshape(-1, 8192), layer.output)


def reference(weights, auxiliary, ids):
    d = weights.draft
    projected = rms(F.linear(auxiliary, d.projection), d.hidden_norm)
    kv = F.linear(projected, d.context_kv).reshape(152, 5, 2048)
    history = []
    for index, layer in enumerate(d.layers):
        k = fp8(rotary(rms(kv[:, index, :1024].reshape(152, 16, 64), layer.key_norm),
                       torch.arange(152, device=auxiliary.device)))
        v = fp8(kv[:, index, 1024:].reshape(152, 16, 64))
        history.append((k, v))
    x = F.embedding(ids, weights.embedding)
    outputs = []
    positions = torch.arange(152, 160, device=x.device)
    for layer, cached in zip(d.layers, history, strict=True):
        before, delta = conv_prepare(rms(x, layer.input_norm), layer.attention_conv)
        x = x + convolve(attention(before, layer, cached, positions), delta, layer.attention_conv['base'][1])
        before, delta = conv_prepare(rms(x, layer.post_attention_norm), layer.mlp_conv)
        gate, up = F.linear(before, layer.mlp.gate_up).chunk(2, -1)
        mlp = F.linear(F.silu(gate) * up, layer.mlp.down)
        x = x + convolve(mlp, delta, layer.mlp_conv['base'][1])
        outputs.append(x.clone())
    hidden = rms(x, d.norm)
    options = weights.config.draft.text['dflash_config']
    logits = F.linear(hidden, weights.head).float() * options['output_multiplier']
    logits = (logits / options['final_logit_softcapping']).tanh() * options['final_logit_softcapping']
    return hidden, logits, history, outputs


def relative_error(actual, expected):
    return float((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12))


class Pool:
    def __init__(self):
        self.values = {f'draft{i}.{kind}': torch.zeros((3, 2048, 16, 64), device='cuda', dtype=torch.float8_e4m3fn)
                       for i in range(5) for kind in ('key', 'value')}

    def reserved_region(self, name):
        return self.values[name]


@pytest.mark.parametrize("compiled", [False, True])
@torch.no_grad()
def test_actual_context_and_five_layer_draft(compiled):
    if not all(path.exists() for path in (TARGET, DRAFT, FEATURES)):
        pytest.skip('requires actual target/draft checkpoints and matched 152-token target features')
    from b12x.preparation import PreparationSession
    from ignition.components.paged_batch import Metadata
    from ignition.components.parallel import TensorParallelShard
    from ignition.components.weights import DirectWeightLoader
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import GraphBufferPool, ScratchArena
    from ignition.recipes.muse_spark.config import ModelConfig
    from ignition.recipes.muse_spark.draft_weights import DraftWeights
    from ignition.recipes.muse_spark.dflash import PreparedContext, PreparedDraft, DraftScratch

    fixture = torch.load(FEATURES, map_location='cpu', weights_only=True)
    token_ids = fixture['prompt_tokens'] + fixture['teacher_tokens']
    assert len(token_ids) == 152
    assert fixture['layer_outputs'].shape == (152, 62, 8192)
    config = ModelConfig.load(TARGET, draft_snapshot=DRAFT)
    auxiliary = fixture['layer_outputs'][:, list(config.draft.target_layers)].reshape(152, 40960).cuda()
    pending = int(fixture['logits'][-1].argmax())
    del fixture
    shard = TensorParallelShard(0, 1)
    draft = DraftWeights.load(DRAFT, config.draft, torch.device('cuda:0'), shard)
    with DirectWeightLoader(inspect_indexed_checkpoint(TARGET), 'cuda:0',
            prefixes=('model.language_model.embed_tokens', 'lm_head')) as loader:
        embedding, head = loader.load_tensors(('model.language_model.embed_tokens.weight', 'lm_head.weight'))
    weights = SimpleNamespace(config=config, shard=shard, draft=draft, embedding=embedding, head=head)
    pool = Pool()
    context_metadata = Metadata.allocate(152, 1, 2, 'cuda', samples=1)
    context_metadata.set(torch.tensor([0], device='cuda'), torch.tensor([152], device='cuda', dtype=torch.int32),
        torch.tensor([1], device='cuda'), torch.zeros((1, 2), device='cuda', dtype=torch.int32),
        torch.zeros((1, 152), device='cuda', dtype=torch.int64), torch.tensor([1], device='cuda', dtype=torch.int32))
    ids = torch.full((8,), config.draft.noise_token, device='cuda', dtype=torch.int64)
    ids[0] = pending
    metadata = Metadata.allocate(8, 1, 2, 'cuda', samples=8)
    metadata.set(torch.tensor([152], device='cuda'), torch.tensor([8], device='cuda', dtype=torch.int32),
        torch.tensor([1], device='cuda'), torch.zeros((1, 2), device='cuda', dtype=torch.int32),
        ids.view(1, 8), torch.tensor([1], device='cuda', dtype=torch.int32))
    buffers, context_buffers, private = (GraphBufferPool('cuda') for _ in range(3))
    with PreparationSession(device='cuda:0', autotune=False) as session:
        arena = ScratchArena('cuda')
        scratch = DraftScratch.allocate(config.draft, shard, 8, 'cuda', buffers=buffers.cursor())
        context = PreparedContext(weights, auxiliary, context_metadata, pool, session=session, name='actual.context',
            workspace=arena, runtime=None, coordinator=None, buffers=context_buffers.cursor())
        prepared = PreparedDraft(weights, scratch, metadata, pool, runtime=None, vocabulary_runtime=None,
            embedding_runtime=None, coordinator=None, session=session, name='actual.draft', workspace=arena,
            buffers=private.cursor())
        arena.finalize()
        def forward():
            context(auxiliary, context_metadata.lengths)
            prepared()
        run = compile_forward(forward) if compiled else forward
        run()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            run()
        before = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == before
        actual_hidden, actual_logits = scratch.hidden.clone(), scratch.logits.clone()
        hidden, logits, history, layers = reference(weights, auxiliary, ids)
        context_errors = []
        for i, (key, value) in enumerate(history):
            for name, expected in (('key', key), ('value', value)):
                actual = pool.values[f'draft{i}.{name}'][1, :152]
                error = relative_error(actual, expected)
                context_errors.append(error)
                assert error < .008, (i, name, error)
                assert not bool(pool.values[f'draft{i}.{name}'][0].float().any())
        component_errors = []
        x = F.embedding(ids, weights.embedding)
        positions = torch.arange(152, 160, device=x.device)
        for index, (layer, stages) in enumerate(zip(draft.layers, prepared.layers, strict=True)):
            norm, conv, attn, post_norm, mlp_conv, mlp, reduction = stages
            scratch.normalized.copy_(norm(x))
            expected, delta = conv_prepare(scratch.normalized, layer.attention_conv)
            before = conv.prepare(scratch.normalized)
            convolution_error = relative_error(before, expected)
            cached = tuple(pool.values[f'draft{index}.{kind}'][1, :152] for kind in ('key', 'value'))
            expected_attended = attention(before, layer, cached, positions)
            attended = attn(before)
            attention_error = relative_error(attended, expected_attended)
            expected_finished = convolve(attended, delta, layer.attention_conv['base'][1])
            finished = conv.finish(attended)
            finish_error = relative_error(finished, expected_finished)
            x = x + finished
            scratch.normalized.copy_(post_norm(x))
            expected, delta = conv_prepare(scratch.normalized, layer.mlp_conv)
            before = mlp_conv.prepare(scratch.normalized)
            mlp_convolution_error = relative_error(before, expected)
            gate, up = F.linear(before, layer.mlp.gate_up).chunk(2, -1)
            expected_mlp = F.linear(F.silu(gate) * up, layer.mlp.down)
            actual_mlp = reduction(mlp(before))
            mlp_error = relative_error(actual_mlp, expected_mlp)
            expected_finished = convolve(actual_mlp, delta, layer.mlp_conv['base'][1])
            finished = mlp_conv.finish(actual_mlp)
            mlp_finish_error = relative_error(finished, expected_finished)
            x = x + finished
            errors = (convolution_error, attention_error, finish_error, mlp_convolution_error,
                      mlp_error, mlp_finish_error)
            component_errors.append(errors)
            assert max(errors) < .008, (index, errors)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(scratch.hidden, actual_hidden)
        assert torch.equal(scratch.logits, actual_logits)
        assert torch.isfinite(actual_hidden).all() and torch.isfinite(actual_logits).all()
        print(dict(compiled=compiled, component_relative_l2=component_errors, context_relative_l2=context_errors, hidden_relative_l2=relative_error(actual_hidden, hidden),
            logits_rmse=float((actual_logits - logits).square().mean().sqrt()),
            top1_agreement=float((actual_logits.argmax(-1) == logits.argmax(-1)).float().mean())))
        # Selector arithmetic is checked on the same prepared hidden rows.
        projected = F.linear(actual_hidden, draft.selector['projection']).view(1, 8, 256)
        assert relative_error(prepared.selector.projected, projected) < .008
        predecessor = ids[:1]
        for position in range(1, 8):
            unary = actual_logits[position:position+1].clone()
            top, indices = unary.topk(16, -1, sorted=False)
            coded = draft.selector['predecessor'][predecessor] * projected[:, position]
            correction = torch.bmm(draft.selector['successor'][indices], coded.unsqueeze(-1)).squeeze(-1)
            expected = torch.full_like(unary, -torch.inf).scatter_(1, indices, top + correction.float())
            prepared.selector(unary, predecessor, position)
            torch.testing.assert_close(unary, expected, rtol=.008, atol=.008)
            assert torch.equal(unary.argmax(-1), expected.argmax(-1))
            predecessor = unary.argmax(-1)
        graph.reset()
