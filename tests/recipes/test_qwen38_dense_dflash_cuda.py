# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint DFlash2 context arithmetic and captured complete-state restores."""

from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from ignition.recipes import resolve

CUDA = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires assigned CUDA GPU")
HUB = Path.home() / ".cache/huggingface/hub"
TARGET = resolve("qwen3.8-27b").snapshot(HUB)
DRAFT = Path(os.environ.get("FLARE_QWEN_DFLASH_CHECKPOINT", str(HUB / "models--incoai--Qwen3.8-27B-DFlash2/snapshots/015e795645c74b1a0eeef3b570031fb62e769bc5")))


def relative_error(actual, expected):
    return float((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12))


def rms(x, weight):
    source = x.float()
    return weight * (source * (source.square().mean(-1, keepdim=True) + 1e-6).rsqrt()).to(x.dtype)


def rotary(x, positions):
    inverse = 10000000. ** (-torch.arange(0, 128, 2, device=x.device).float() / 128)
    angle = positions.float()[:, None] * inverse
    angle = torch.cat((angle, angle), -1)
    left, right = x.chunk(2, -1)
    return x * angle.cos().bfloat16()[:, None] + torch.cat((-right, left), -1) * angle.sin().bfloat16()[:, None]


@pytest.fixture(scope="module")
def weights():
    from ignition.components.dflash2_weights import DraftWeights
    from ignition.components.parallel import TensorParallelShard
    from ignition.recipes.qwen38_dense.config import ModelConfig
    from ignition.recipes.qwen38_dense import draft_config
    if not all((path / "config.json").is_file() for path in (TARGET, DRAFT)):
        pytest.skip("requires pinned Qwen3.8-27B target and DFlash2 checkpoints")
    target = ModelConfig.load(TARGET)
    config = replace(target, draft=draft_config.load(DRAFT, target))
    shard = TensorParallelShard(0, 1)
    draft = DraftWeights.load(DRAFT, config.draft, torch.device("cuda:0"), shard)
    return SimpleNamespace(config=config, draft=draft, shard=shard)


@CUDA
@torch.no_grad()
def test_checkpoint_context_accepted_counts_ring_wrap_inactive_and_reordered_rows(weights):
    from b12x.preparation import PreparationSession
    from ignition.components.dflash2 import PreparedContext
    from ignition.components.paged_batch import Metadata
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import GraphBufferPool, ScratchArena

    torch.manual_seed(5027)
    width, rows, window = 8, 4, 2048
    auxiliary = torch.randn((width * rows, 25600), device="cuda", dtype=torch.bfloat16)
    metadata = Metadata.allocate(width * rows, rows, 1, "cuda", samples=width * rows)
    rings = {f"draft{i}.{kind}": torch.zeros((6, window, 8, 128), device="cuda", dtype=torch.float8_e4m3fn)
             for i in range(5) for kind in ("key", "value")}
    pool = SimpleNamespace(reserved_region=rings.__getitem__)
    counts = torch.zeros(rows, device="cuda", dtype=torch.int32)
    buffers = GraphBufferPool("cuda")
    with PreparationSession(device="cuda:0", autotune=False) as session:
        arena = ScratchArena("cuda")
        prepared = PreparedContext(weights, auxiliary, metadata, pool, session=session, name="qwen.context",
            workspace=arena, runtime=None, coordinator=None, buffers=buffers.cursor())
        arena.finalize()
        forward = compile_forward(lambda: prepared(auxiliary, counts))
        forward()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()
        for accepted in range(9):
            starts = torch.tensor([2047, 2048, 4095, 0], device="cuda")
            lengths = torch.tensor([8, 8, 8, 0], device="cuda", dtype=torch.int32)
            states = torch.tensor([4, 1, 3, 5], device="cuda")
            metadata.set(starts, lengths, states, torch.zeros((rows, 1), device="cuda", dtype=torch.int32),
                torch.zeros((rows, width), device="cuda", dtype=torch.int64),
                torch.tensor([rows], device="cuda", dtype=torch.int32))
            counts.copy_(torch.tensor([accepted, 8 - accepted, 0, 0], device="cuda"))
            auxiliary.normal_()
            for tensor in rings.values():
                tensor.fill_(7)
            hidden = rms(F.linear(auxiliary, weights.draft.projection), weights.draft.hidden_norm)
            kv = F.linear(hidden, weights.draft.context_kv).view(width * rows, 5, 2048)
            expected = {name: value.clone() for name, value in rings.items()}
            for layer in range(5):
                key = rotary(rms(kv[:, layer, :1024].reshape(-1, 8, 128),
                    weights.draft.layers[layer].key_norm), metadata.positions).clamp(-448, 448).to(torch.float8_e4m3fn)
                value = kv[:, layer, 1024:].reshape(-1, 8, 128).clamp(-448, 448).to(torch.float8_e4m3fn)
                for row in range(2):
                    for token in range(int(counts[row])):
                        index = row * width + token
                        position = int(starts[row]) + token
                        for kind, tensor in (("key", key), ("value", value)):
                            expected[f"draft{layer}.{kind}"][states[row], position % window] = tensor[index]
            allocated = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            for row in range(2):
                for token in range(int(counts[row])):
                    index = row * width + token
                    position = (int(starts[row]) + token) % window
                    for kind, actual_source in (("key", prepared.key), ("value", prepared.value)):
                        expected_write = actual_source[index].clamp(-448, 448).to(torch.float8_e4m3fn)
                        assert torch.equal(rings[f"draft4.{kind}"][states[row], position].view(torch.uint8), expected_write.view(torch.uint8))
            for name, actual in rings.items():
                changed = expected[name].view(torch.uint8) != 78
                assert torch.equal(actual.view(torch.uint8)[~changed], expected[name].view(torch.uint8)[~changed]), (accepted, name)
                expected_values = expected[name].float()[changed]
                actual_values = actual.float()[changed]
                error = float((actual_values - expected_values).norm() / expected_values.norm().clamp_min(1e-12))
                print(dict(accepted=accepted, ring=name, relative_l2=error))
                assert error < .008, (accepted, name, error)
        graph.reset()


@CUDA
@pytest.mark.parametrize("endpoint", [1, 127, 128, 129, 2047, 2048, 2049])
@torch.no_grad()
def test_complete_draft_rings_snapshot_branch_and_restore(weights, endpoint):
    from ignition.components.sampling import DeviceRequestState
    from ignition.components.state_transfer import PreparedStateTransfer, StateCommands
    from ignition.recipes.qwen38_dense.pool import ContinuationPool

    pool = ContinuationPool.for_model(weights.config, 4, 36, "cuda", mtp=True)
    commands = StateCommands.allocate("cuda", capacity=2)
    tables = torch.arange(36, device="cuda", dtype=torch.int32).view(2, 18)
    requests = DeviceRequestState.allocate("cuda", vocab_size=128, capacity=2)
    requests.consumed_tokens.fill_(endpoint)
    transfer = PreparedStateTransfer(pool, commands, tables, pages=pool.partial_pages(), page_tokens=128,
        state_rows=torch.tensor([[1] * 8, [2] * 8], device="cuda"), requests=requests)
    transfer.prepare()
    pool.backing[1].random_(0, 256)
    pool.region("consumed_tokens")[1] = endpoint
    for page in pool.pages.values():
        page.random_(0, 256)
    commands.request_slots[0] = 0
    commands.source_slots[0], commands.destination_slots[0] = 1, 3
    commands.expected_consumed[0], commands.num_ops[0] = endpoint, 1
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        transfer.validate()
        transfer.apply()
    graph.replay()
    torch.cuda.synchronize()
    assert not commands.errors.count_nonzero()
    for region in pool.layout.regions:
        if region.name not in pool.pages:
            assert torch.equal(pool.region(region.name)[3].view(torch.uint8), pool.region(region.name)[1].view(torch.uint8)), region.name
    saved = pool.backing[3].clone()
    commands.request_slots[0] = 1
    commands.source_slots[0], commands.destination_slots[0] = 3, 2
    graph.replay()
    torch.cuda.synchronize()
    assert not commands.errors.count_nonzero()
    assert torch.equal(pool.backing[3], saved)
    for layer in range(5):
        for kind in ("key", "value"):
            ring = pool.reserved_region(f"draft{layer}.{kind}").view(torch.uint8)
            assert torch.equal(ring[1], ring[2]) and torch.equal(ring[1], ring[3])
    graph.reset()


@CUDA
@torch.no_grad()
def test_checkpoint_non_square_qkv_and_replicated_weights_are_exact(weights):
    from safetensors import safe_open
    with safe_open(DRAFT / "model.safetensors", framework="pt", device="cpu") as source:
        def check(actual, name):
            assert actual.dtype == torch.bfloat16
            assert torch.equal(actual.cpu(), source.get_tensor(name)), name
        check(weights.draft.projection, "fc.weight")
        check(weights.draft.hidden_norm, "hidden_norm.weight")
        check(weights.draft.norm, "norm.weight")
        for i, layer in enumerate(weights.draft.layers):
            prefix = f"layers.{i}."
            for actual, name in zip(layer.qkv.split((4096, 1024, 1024)), ("q", "k", "v"), strict=True):
                check(actual, prefix + f"self_attn.{name}_proj.weight")
            assert layer.output.shape == (5120, 4096)
            check(layer.output, prefix + "self_attn.o_proj.weight")
            assert torch.equal(weights.draft.context_kv[i * 2048:(i + 1) * 2048], layer.qkv[4096:])
            for name, conv in (("attention_conv", layer.attention_conv), ("mlp_conv", layer.mlp_conv)):
                check(conv["base"], prefix + name + ".base_kernel")
                check(conv["projection"], prefix + name + ".kernel_projection.weight")
        for name, actual in weights.draft.selector.items():
            tensor_name = dict(projection="hidden_projection.weight", predecessor="predecessor_codebook",
                               successor="successor_codebook")[name]
            check(actual, "candidate_selector." + tensor_name)


def convolve(x, delta, base):
    grouped = x.reshape(-1, 320, 16)
    coefficient = base.reshape(1, 2, 320, 16) + delta.unsqueeze(-1)
    shifted = F.pad(grouped[:-1], (0, 0, 0, 0, 1, 0))
    valid = (torch.arange(x.shape[0], device=x.device) % 8 >= 1).reshape(-1, 1, 1)
    return (coefficient[:, 0] * grouped + coefficient[:, 1] * shifted * valid).flatten(1)


def before_convolution(x, weights):
    delta = F.linear(x, weights["projection"]).reshape(-1, 2, 2, 320)
    return convolve(x, delta[:, 0], weights["base"][0]), delta[:, 1]


def head_reference(hidden):
    """Decode checkpoint E2M1/E4M3 blocks, then apply the FP32 global gain."""
    import json
    from safetensors import safe_open
    index = json.loads((TARGET / "model.safetensors.index.json").read_text())["weight_map"]
    result = []
    table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.], device="cuda")
    with safe_open(TARGET / index["lm_head.weight"], framework="pt", device="cpu") as source:
        gain = source.get_tensor("lm_head.weight_scale_2").cuda()
        for start in range(0, 248320, 4096):
            end = min(start + 4096, 248320)
            packed = source.get_slice("lm_head.weight")[start:end].cuda()
            scales = source.get_slice("lm_head.weight_scale")[start:end].cuda().float()
            codes = torch.stack((packed & 15, packed >> 4), -1).long().flatten(1)
            decoded = (table[codes] * scales.repeat_interleave(16, -1)).bfloat16()
            result.append((F.linear(hidden.float(), decoded.float()) * gain).bfloat16().float())
    return torch.cat(result, -1)


@CUDA
@torch.no_grad()
def test_checkpoint_five_layer_operator_arithmetic_and_raw_head(weights):
    from b12x.preparation import PreparationSession
    from ignition.components.dflash2 import DraftScratch, PreparedDraft
    from ignition.components.paged_batch import Metadata
    from ignition.components.weights import DirectWeightLoader
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import GraphBufferPool, ScratchArena
    from ignition.recipes.qwen38_dense.config import TEXT_PREFIX

    with DirectWeightLoader(inspect_indexed_checkpoint(TARGET), "cuda:0",
            prefixes=(TEXT_PREFIX + ".embed_tokens", "lm_head")) as loader:
        embedding = loader.load_tensor(TEXT_PREFIX + ".embed_tokens.weight")
        head = loader.load_linear("lm_head")
    torch.manual_seed(2718)
    metadata = Metadata.allocate(8, 1, 1, "cuda", samples=8)
    ids = torch.full((1, 8), weights.config.draft.noise_token, device="cuda", dtype=torch.int64)
    ids[0, 0] = 11
    metadata.set(torch.tensor([17], device="cuda"), torch.tensor([8], device="cuda", dtype=torch.int32),
        torch.tensor([1], device="cuda"), torch.zeros((1, 1), device="cuda", dtype=torch.int32), ids,
        torch.tensor([1], device="cuda", dtype=torch.int32))
    rings = {f"draft{i}.{kind}": (torch.randn((2, 2048, 8, 128), device="cuda") * .1).to(torch.float8_e4m3fn)
             for i in range(5) for kind in ("key", "value")}
    saved = {name: tensor.view(torch.uint8).clone() for name, tensor in rings.items()}
    pool = SimpleNamespace(reserved_region=rings.__getitem__)
    buffers, private = GraphBufferPool("cuda"), GraphBufferPool("cuda")
    with PreparationSession(device="cuda:0", autotune=False) as session:
        arena = ScratchArena("cuda")
        scratch = DraftScratch.allocate(weights.config.draft, weights.shard, 8, "cuda", buffers=buffers.cursor())
        prepared = PreparedDraft(weights, scratch, metadata, pool, runtime=None, vocabulary_runtime=None,
            embedding_runtime=None, coordinator=None, session=session, name="qwen.draft", workspace=arena,
            buffers=private.cursor(), embedding=embedding, head=head, output_multiplier=1., final_logit_softcapping=None)
        arena.finalize()
        forward = compile_forward(prepared)
        forward()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        actual = scratch.hidden.clone()
        logits_before = scratch.logits.clone()
        prepared()
        print(dict(eager_compiled_hidden_relative_l2=relative_error(scratch.hidden, actual),
            eager_compiled_logits_relative_l2=relative_error(scratch.logits, logits_before)))
        x = F.embedding(ids.flatten(), embedding)
        for index, layer in enumerate(weights.draft.layers):
            normalized, delta = before_convolution(rms(x, layer.input_norm), layer.attention_conv)
            q, k, v = F.linear(normalized, layer.qkv).split((4096, 1024, 1024), -1)
            q = rotary(rms(q.reshape(8, 32, 128), layer.query_norm), metadata.positions)
            k = rotary(rms(k.reshape(8, 8, 128), layer.key_norm), metadata.positions).clamp(-448, 448).to(torch.float8_e4m3fn)
            v = v.reshape(8, 8, 128).clamp(-448, 448).to(torch.float8_e4m3fn)
            key = torch.cat((rings[f"draft{index}.key"][1, :17].bfloat16(), k.bfloat16())).repeat_interleave(4, 1)
            value = torch.cat((rings[f"draft{index}.value"][1, :17].bfloat16(), v.bfloat16())).repeat_interleave(4, 1)
            scores = torch.einsum("qhd,khd->hqk", q.float(), key.float()) * 128 ** -.5
            attended = torch.einsum("hqk,khd->qhd", scores.softmax(-1), value.float()).bfloat16().flatten(1)
            norm, conv, attn, post_norm, mlp_conv, mlp, reduction = prepared.layers[index]
            diagnostic = conv.prepare(rms(x, layer.input_norm)).clone()
            attention_reference = F.linear(attended, layer.output)
            attention_actual = attn(diagnostic).clone()
            relative = lambda a, b: float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))
            expected_finish = convolve(attention_actual, delta, layer.attention_conv["base"][1])
            finish_actual = conv.finish(attention_actual).clone()
            errors = dict(norm_error=relative(norm(x), rms(x, layer.input_norm)), convolution_error=relative(diagnostic, normalized),
                attention_error=relative(attention_actual, attention_reference), convolution_finish_error=relative(finish_actual, expected_finish))
            x = x + convolve(attention_reference, delta, layer.attention_conv["base"][1])
            normalized, delta = before_convolution(rms(x, layer.post_attention_norm), layer.mlp_conv)
            diagnostic = mlp_conv.prepare(rms(x, layer.post_attention_norm)).clone()
            gate, up = F.linear(normalized, layer.mlp.gate_up).chunk(2, -1)
            reference_mlp = F.linear(F.silu(gate) * up, layer.mlp.down)
            actual_mlp = reduction(mlp(diagnostic)).clone()
            errors.update(mlp_conv_error=relative(diagnostic, normalized), mlp_error=relative(actual_mlp, reference_mlp),
                mlp_finish_error=relative(mlp_conv.finish(actual_mlp), convolve(actual_mlp, delta, layer.mlp_conv["base"][1])))
            print(dict(layer=index, **errors))
            assert max(errors.values()) < .008, (index, errors)
            x = x + convolve(reference_mlp, delta, layer.mlp_conv["base"][1])
        expected = rms(x, weights.draft.norm)
        error = float((actual.float() - expected.float()).norm() / expected.float().norm())
        print(dict(hidden_relative_l2=error))
        graph.replay()
        torch.cuda.synchronize()
        print(dict(replay_hidden_relative_l2=relative_error(scratch.hidden, actual),
            replay_logits_relative_l2=relative_error(scratch.logits, logits_before)))
        expected_head = head_reference(scratch.hidden)
        head_error = float((scratch.logits - expected_head).norm() / expected_head.norm())
        print(dict(head_relative_l2=head_error))
        assert head_error < .008
        # The prepared projection returns BF16 logits; Qwen applies no multiplier or softcap.
        assert torch.equal(scratch.logits, scratch.gathered_logits[:, :248320].float())
        projected = F.linear(scratch.hidden, weights.draft.selector["projection"]).view(1, 8, 256)
        selector_error = float((prepared.selector.projected.float() - projected.float()).norm() / projected.float().norm())
        assert selector_error < .008
        predecessor = ids[:, 0]
        for position in range(1, 8):
            logits = scratch.logits[position:position + 1].clone()
            top, indices = logits.topk(16, -1, sorted=False)
            coded = weights.draft.selector["predecessor"][predecessor] * prepared.selector.projected[:, position]
            correction = torch.bmm(weights.draft.selector["successor"][indices], coded.unsqueeze(-1)).squeeze(-1)
            expected_logits = torch.full_like(logits, -torch.inf).scatter_(1, indices, top + correction.float())
            prepared.selector(logits, predecessor, position)
            torch.testing.assert_close(logits, expected_logits, rtol=.008, atol=.008)
            assert torch.equal(logits.argmax(-1), expected_logits.argmax(-1))
            predecessor = logits.argmax(-1)
        for name, tensor in rings.items():
            assert torch.equal(tensor.view(torch.uint8), saved[name]), name
        graph.reset()
