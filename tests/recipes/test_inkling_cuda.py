# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real prepared Inkling decoder replay across prefill, decode, and restored state."""

import os

import pytest
import torch
import torch.nn.functional as F

from ignition.components.paged_batch import Metadata
from ignition.components.parallel import TensorParallelShard
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import GraphBufferPool, ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.inkling.decoder import PreparedDecoder, Scratch
from ignition.recipes.inkling.model import norm
from ignition.recipes.inkling.pool import ContinuationPool
from ignition.recipes.inkling.weights import ModelWeights
from tests.components.inkling_nvfp4_helpers import nvfp4_routes
from .inkling_helpers import checkpoint

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned CUDA GPU")


def convolution(x, weight):
    y = F.conv1d(x.float().T[None], weight[:, None], padding=weight.shape[1]-1, groups=x.shape[1])
    return (y[0, :, :x.shape[0]].T + x.float()).to(x.dtype)


def assert_decoder_close(actual, expected):
    # Tiled and dense FP32 softmax reductions can straddle a BF16 probability
    # rounding boundary. Residuals and later KV history propagate that difference;
    # normalize error by the vector scale so cancellation near zero stays meaningful.
    epsilon = torch.finfo(torch.bfloat16).eps
    reference = expected.float()
    torch.testing.assert_close(actual, expected, atol=epsilon * reference.abs().max().item(), rtol=0)
    assert (actual.float() - reference).norm().item() <= (epsilon / 2) * reference.norm().item()


def reference(weights, tokens, original):
    from transformers.models.inkling.configuration_inkling import InklingTextConfig
    from transformers.models.inkling.modeling_inkling import InklingAttention

    c = weights.config.text
    config = InklingTextConfig.from_dict(c)
    config._attn_implementation = "eager"
    x = norm(weights.embedding_norm, c["rms_norm_eps"])(F.embedding(tokens, weights.embedding))
    for index, w in enumerate(weights.layers):
        a = weights.config.attention(index)
        attention = InklingAttention(config, index).to(device=tokens.device, dtype=torch.bfloat16).eval()
        sizes = a.heads * a.key_dim, a.kv_heads * a.key_dim, a.kv_heads * a.key_dim, a.heads * c["d_rel"]
        for name, value in zip(("q_proj", "k_proj", "v_proj", "r_proj"), w.qkv_relative.split(sizes)):
            getattr(attention, name).weight = torch.nn.Parameter(value, requires_grad=False)
        attention.o_proj.weight = torch.nn.Parameter(w.output, requires_grad=False)
        attention.q_norm.weight = torch.nn.Parameter(w.query_norm, requires_grad=False)
        attention.k_norm.weight = torch.nn.Parameter(w.key_norm, requires_grad=False)
        attention.rel_logits_proj.proj = torch.nn.Parameter(w.relative.T, requires_grad=False)
        attention.k_sconv.conv1d.weight = torch.nn.Parameter(w.key_conv[:, None], requires_grad=False)
        attention.v_sconv.conv1d.weight = torch.nn.Parameter(w.value_conv[:, None], requires_grad=False)
        positions = torch.arange(tokens.numel(), device=tokens.device)
        distance = positions[:, None] - positions[None]
        allowed = distance >= 0
        if a.window > 0:
            allowed &= distance < a.window
        mask = torch.where(allowed, 0., float("-inf")).bfloat16()[None, None]
        attended = attention(norm(w.input_norm, c["rms_norm_eps"])(x)[None], attention_mask=mask)[0][0]
        x = x + convolution(attended, w.attention_conv)
        source = norm(w.post_attention_norm, c["rms_norm_eps"])(x)
        def mlp(gate_up, down, gamma=None):
            gate, up = F.linear(source, gate_up).float().chunk(2, dim=-1)
            activated = F.silu(gate) * up
            if gamma is not None:
                activated *= gamma[:, None]
            return F.linear(activated.bfloat16(), down)
        if w.dense is not None:
            out = mlp(w.dense.gate_up, w.dense.down) * w.scale
        else:
            # SGLang InklingGate retains FP32 logits, correction and global scale.
            logits = source.float() @ w.router.float().T
            ids = torch.topk(logits[:, :-c["n_shared_experts"]].sigmoid() + w.correction,
                             c["num_experts_per_tok"], sorted=False).indices
            selected = torch.cat((logits[:, :-c["n_shared_experts"]].gather(1, ids), logits[:, -c["n_shared_experts"]:]), -1)
            logp = F.logsigmoid(selected)
            probabilities = (logp - logp.logsumexp(-1, keepdim=True)).exp() * c["route_scale"] * w.scale
            routed = torch.zeros_like(source)
            quantized = index in weights.config.quantized_routed_layers
            if quantized:
                routes = nvfp4_routes(source, ids, original, f"model.llm.layers.{index}.mlp.experts")
            for expert in range(c["n_routed_experts"]):
                selection = ids == expert
                if quantized:
                    for slot in range(ids.shape[1]):
                        weight = probabilities[:, slot] * selection[:, slot]
                        routed += (routes[:, slot] * weight[:, None]).bfloat16()
                else:
                    expert_out = mlp(w.experts.gate_up[expert], w.experts.down[expert])
                    weight = (probabilities[:, :c["num_experts_per_tok"]] * selection).sum(-1)
                    routed += (expert_out * weight[:, None]).bfloat16()
            shared = torch.stack([mlp(expert.gate_up, expert.down,
                probabilities[:, c["num_experts_per_tok"] + i]) for i, expert in enumerate(w.shared)])
            out = routed + shared.float().sum(0).bfloat16()
        x = x + convolution(out, w.mlp_conv)
    hidden = norm(weights.norm, c["rms_norm_eps"])(x) / c["logits_mup_width_multiplier"]
    return F.linear(hidden, weights.head)[:, :weights.config.vocabulary], hidden


@torch.no_grad()
def test_original_checkpoint_loader_shards_interleaved_rows_without_losing_precision(tmp_path):
    config, original = checkpoint(tmp_path)
    manifest = inspect_indexed_checkpoint(tmp_path)
    from ignition.recipes.inkling.weights import InklingWeightLoader

    with InklingWeightLoader(manifest, "cuda:0", shard=TensorParallelShard(1, 2)) as loader:
        layer = loader.load_layer(1)
    raw = original["model.llm.layers.1.mlp.experts.w13_weight"]
    assert torch.equal(layer.experts.gate_up.cpu(), torch.cat((raw[:, 128::2], raw[:, 129::2]), dim=1))
    assert torch.equal(layer.correction.cpu(), original["model.llm.layers.1.mlp.gate.bias"])
    assert torch.equal(layer.scale.cpu(), original["model.llm.layers.1.mlp.gate.global_scale"])
    assert layer.correction.dtype == layer.scale.dtype == layer.key_conv.dtype == torch.float32
    assert torch.equal(layer.key_conv.cpu(), original["model.llm.layers.1.attn.k_sconv.weight"].float().flatten(1))


@pytest.mark.parametrize("layer,excluded,error", [(1, False, "incomplete NVFP4"), (2, True, "interleaved BF16")])
@torch.no_grad()
def test_loader_rejects_checkpoint_precision_disagreeing_with_exclusions(tmp_path, layer, excluded, error):
    import json
    from ignition.recipes.inkling.weights import InklingWeightLoader

    checkpoint(tmp_path)
    path = tmp_path / "hf_quant_config.json"
    metadata = json.loads(path.read_text())
    prefix = f"model.llm.layers.{layer}.mlp.experts"
    exclusions = metadata["quantization"]["exclude_modules"]
    if excluded:
        exclusions.append(prefix)
        exclusions.remove("model.llm.layers.1.mlp.experts")
    else:
        exclusions.remove(prefix)
    path.write_text(json.dumps(metadata))
    with InklingWeightLoader(inspect_indexed_checkpoint(tmp_path), "cuda:0", shard=TensorParallelShard(0, 1)) as loader:
        with pytest.raises(ValueError, match=error):
            loader.load_layer(layer)


@torch.no_grad()
def test_decoder_compiled_replay_cache_boundaries_and_restored_branch(tmp_path):
    _, original = checkpoint(tmp_path)
    device, shard = torch.device("cuda", 0), TensorParallelShard(0, 1)
    weights = ModelWeights.load(inspect_indexed_checkpoint(tmp_path), device, shard=shard)
    pool = ContinuationPool.allocate(weights.config, shard, 4, 5, device)
    buffers = GraphBufferPool(device, chunk_bytes=2 << 20)
    catalog = {}
    with RecordedPreparationSession(device=device, checkpoint="inkling-decoder-test", workload_schema="inkling",
                                    numerical_policy="default", compile_workers=1) as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        for mode, capacity in (("prefill", 64), ("decode", 1)):
            cursor = buffers.cursor()
            metadata = Metadata.allocate(capacity, 1, 5, device, buffers=cursor)
            scratch = Scratch.allocate(weights.config, shard, capacity, 1, device, buffers=cursor)
            decoder = PreparedDecoder(weights, scratch, metadata, pool, mode=mode, runtime=None,
                embedding_runtime=None, vocabulary_runtime=None, coordinator=None, session=session,
                name=mode, workspace=arena, buffers=cursor)
            catalog[mode] = metadata, scratch, decoder
        session.tune()
        graphs = {}
        for mode, (metadata, scratch, decoder) in catalog.items():
            # Bare decoder tests must initialize metadata that PreparedBatch owns in serving.
            for field in metadata.__dataclass_fields__:
                value = getattr(metadata, field)
                if isinstance(value, torch.Tensor):
                    value.fill_(-1 if field in ("positions", "page_table", "token_page_table") else 0)
            compiled = compile_forward(decoder)
            compiled()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                compiled()
            graphs[mode] = graph, compiled
        session.freeze()
        pool.clear_for_preparation()
        tokens = (torch.arange(259, device=device) * 17 % weights.config.vocabulary).long()
        expected, expected_hidden = reference(weights, tokens, original)
        start = 0
        for mode, count in (("prefill", 63), ("decode", 1), ("decode", 1), ("prefill", 62),
                            ("decode", 1), ("decode", 1), ("prefill", 64), ("prefill", 64), ("decode", 1), ("decode", 1)):
            metadata, scratch, decoder = catalog[mode]
            values = torch.zeros((1, metadata.token_ids.numel()), dtype=torch.int64, device=device)
            values[0, :count] = tokens[start:start+count]
            metadata.set(torch.tensor([start], device=device), torch.tensor([count], device=device, dtype=torch.int32),
                torch.tensor([1], device=device), torch.arange(5, device=device, dtype=torch.int32)[None],
                values, torch.ones(1, device=device, dtype=torch.int32))
            metadata.sample_indices.fill_(count - 1)
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graphs[mode][0].replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            start += count
            assert_decoder_close(scratch.logits[0], expected[start - 1])
            assert_decoder_close(scratch.hidden[0], expected_hidden[start - 1])
        assert start == tokens.numel()
        saved = pool.backing.clone()
        pages = {name: value.clone() for name, value in pool.pages.items()}
        metadata, scratch, decoder = catalog["decode"]
        metadata.starts.fill_(start)
        metadata.positions.fill_(start)
        metadata.token_ids.fill_(13)
        graphs["decode"][0].replay()
        torch.cuda.synchronize()
        branch = scratch.logits.clone()
        pool.backing.copy_(saved)
        for name, value in pages.items():
            pool.pages[name].copy_(value)
        graphs["decode"][0].replay()
        torch.cuda.synchronize()
        assert torch.equal(branch, scratch.logits)
        saved = pool.backing.clone()
        metadata.lengths.zero_()
        metadata.query_start_loc.zero_()
        metadata.live_counts.zero_()
        metadata.positions.fill_(-1)
        metadata.state_rows.zero_()
        graphs["decode"][0].replay()
        torch.cuda.synchronize()
        assert torch.equal(saved, pool.backing)


@torch.no_grad()
def test_decoder_batched_decode_with_inactive_rows_after_window_wrap(tmp_path):
    _, original = checkpoint(tmp_path)
    device, shard = torch.device("cuda", 0), TensorParallelShard(0, 1)
    weights = ModelWeights.load(inspect_indexed_checkpoint(tmp_path), device, shard=shard)
    pool = ContinuationPool.allocate(weights.config, shard, 5, 10, device)
    buffers = GraphBufferPool(device, chunk_bytes=2 << 20)
    catalog = {}
    with RecordedPreparationSession(device=device, checkpoint="inkling-batched-test", workload_schema="inkling",
                                    numerical_policy="default", compile_workers=1) as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        for mode, capacity, rows in (("prefill", 192, 1), ("decode", 4, 4)):
            cursor = buffers.cursor()
            metadata = Metadata.allocate(capacity, rows, 5, device, buffers=cursor)
            scratch = Scratch.allocate(weights.config, shard, capacity, rows, device, buffers=cursor)
            decoder = PreparedDecoder(weights, scratch, metadata, pool, mode=mode, runtime=None,
                embedding_runtime=None, vocabulary_runtime=None, coordinator=None, session=session,
                name=mode, workspace=arena, buffers=cursor)
            catalog[mode] = metadata, scratch, decoder
        session.tune()
        graphs = {}
        for mode, (metadata, _, decoder) in catalog.items():
            # Bare decoder tests must initialize metadata that PreparedBatch owns in serving.
            for field in metadata.__dataclass_fields__:
                value = getattr(metadata, field)
                if isinstance(value, torch.Tensor):
                    value.fill_(-1 if field in ("positions", "page_table", "token_page_table") else 0)
            compiled = compile_forward(decoder)
            compiled()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                compiled()
            graphs[mode] = graph, compiled
        session.freeze()
        pool.clear_for_preparation()
        sequences = [(torch.arange(length + 1, device=device) * (17 + row) % weights.config.vocabulary).long()
                     for row, length in enumerate((129, 131))]
        for row, sequence in enumerate(sequences):
            m, s, _ = catalog["prefill"]
            values = torch.zeros((1, 192), dtype=torch.int64, device=device)
            values[0, :sequence.numel()-1] = sequence[:-1]
            m.set(torch.zeros(1, device=device, dtype=torch.int64),
                torch.tensor([sequence.numel()-1], device=device, dtype=torch.int32),
                torch.tensor([row+1], device=device), (torch.arange(5, device=device, dtype=torch.int32) + row*5)[None],
                values, torch.ones(1, device=device, dtype=torch.int32))
            m.sample_indices.fill_(sequence.numel()-2)
            graphs["prefill"][0].replay()
        untouched = pool.backing[[0, 3, 4]].clone()
        m, s, _ = catalog["decode"]
        m.set(torch.tensor([129, 0, 131, 0], device=device), torch.tensor([1, 0, 1, 0], device=device, dtype=torch.int32),
            torch.tensor([1, 3, 2, 4], device=device),
            torch.tensor([[0, 1, 2, 3, 4], [-1]*5, [5, 6, 7, 8, 9], [-1]*5], dtype=torch.int32, device=device),
            torch.stack((sequences[0][-1:], torch.zeros(1, dtype=torch.int64, device=device),
                         sequences[1][-1:], torch.zeros(1, dtype=torch.int64, device=device))),
            torch.tensor([4], device=device, dtype=torch.int32))
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        graphs["decode"][0].replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
        assert torch.equal(untouched, pool.backing[[0, 3, 4]])
        for row, sequence in zip((0, 2), sequences):
            expected, _ = reference(weights, sequence, original)
            assert_decoder_close(s.logits[row], expected[-1])
