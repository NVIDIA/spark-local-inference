# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint draft context and backbone against decoded PyTorch projections."""

from dataclasses import replace
import gc
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an explicitly assigned CUDA device")


def norm(x, weight, eps=1e-6):
    x32 = x.float()
    return (x32 * (x32.square().mean(-1, keepdim=True) + eps).rsqrt() * weight.float()).bfloat16()


def rotary(x, positions):
    frequencies = 10000. ** (-torch.arange(0, 128, 2, device=x.device).float() / 128)
    angles = positions.clamp_min(0).float()[:, None] * frequencies
    cosine, sine = angles.cos().bfloat16().float()[:, None], angles.sin().bfloat16().float()[:, None]
    left, right = x.float().chunk(2, -1)
    return torch.cat((left * cosine - right * sine, right * cosine + left * sine), -1).bfloat16()


def convolve(x, coefficients, base, side):
    shape = x.shape
    x = x.view(-1, 8, shape[-1])
    coeff = coefficients.view(*x.shape[:2], 2, 2, -1)[:, :, side].repeat_interleave(16, -1)
    output = torch.zeros_like(x)
    for tap in range(2):
        output[:, tap:] += (base[side, tap] + coeff[:, tap:, tap]) * x[:, :8 - tap]
    return output.reshape(shape)


@pytest.fixture(scope="module")
@torch.no_grad()
def checkpoint_weights():
    from safetensors import safe_open
    from ignition.components.parallel import TensorParallelShard
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.recipes.nemotron_h.config import ModelConfig
    from ignition.recipes.nemotron_h.draft_config import DraftConfig
    from ignition.recipes.nemotron_h.draft_weights import DraftWeights
    from ignition.recipes.nemotron_h.weights import NemotronWeightLoader
    from ignition.reference.qwen38_quant import decode_nvfp4

    target = Path(os.environ.get("FLARE_NEMOTRON_CHECKPOINT", "/data/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB"))
    draft = Path(os.environ.get("FLARE_LILICORR_CHECKPOINT", str(target) + "-LiLiCorr"))
    if not target.is_dir() or not draft.is_dir():
        pytest.skip("requires the Nemotron Bolt and LiLiCorr checkpoints")
    config = ModelConfig.load(target)
    config = replace(config, draft=DraftConfig.load(draft, config))
    with NemotronWeightLoader(inspect_indexed_checkpoint(target), "cuda:0", shard=TensorParallelShard(0, 1),
            prefixes=("language_model.backbone.embeddings.", "language_model.lm_head.")) as loader:
        embedding = loader.load_tensor("language_model.backbone.embeddings.weight")
        head = loader.load_linear("language_model.lm_head")
    weights = SimpleNamespace(config=config, draft=DraftWeights.load(draft, config.draft, "cuda:0"),
                              embedding=embedding, head=head)
    references = {}
    with safe_open(draft / "model.safetensors", framework="pt", device="cuda:0") as reader:
        names = [name for name in reader.keys() if not name.startswith("lilicorr.")]
        for name in names:
            if name.endswith(("weight_scale", "weight_scale_2")):
                continue
            value = reader.get_tensor(name)
            if value.dtype == torch.uint8:
                p = name.removesuffix(".weight")
                scale = reader.get_tensor(p + ".weight_scale_2")
                value = decode_nvfp4(value, reader.get_tensor(p + ".weight_scale"), torch.ones_like(scale)).bfloat16()
                references[name] = value, scale
            else:
                references[name] = value
    return weights, references


def linear(weights, x, name):
    weight = weights[name + ".weight"]
    if isinstance(weight, tuple):
        value, scale = weight
        return (F.linear(x.float(), value.float()) * scale).bfloat16()
    return F.linear(x, weight)


class ContextReadback:
    def __init__(self, context, auxiliary, counts, metadata, output):
        self.context, self.auxiliary, self.counts = context, auxiliary, counts
        self.metadata, self.output = metadata, output

    def __call__(self):
        self.output.copy_(self.context(self.auxiliary, self.counts, self.metadata))


class DraftReadback:
    def __init__(self, draft, metadata, hidden, logits, features):
        self.draft, self.metadata = draft, metadata
        self.hidden, self.logits, self.features = hidden, logits, features

    def __call__(self):
        hidden, logits, features = self.draft.forward(self.metadata)
        self.hidden.copy_(hidden)
        self.logits.copy_(logits)
        self.features.copy_(features)


@pytest.mark.parametrize("batch", [1, 3, 8])
@torch.no_grad()
def test_checkpoint_context_and_backbone(checkpoint_weights, batch, tmp_path):
    from b12x.preparation import PreparationSession
    from ignition.components.paged_batch import Metadata
    from ignition.preparation.compile_pool import CompilePool
    from ignition.preparation.scratch import ScratchArena
    from ignition.recipes.nemotron_h.decoder import Scratch
    from ignition.recipes.nemotron_h.dflash import PreparedContext, PreparedDraft
    from ignition.recipes.nemotron_h.pool import ContinuationPool

    weights, raw = checkpoint_weights
    c, tokens = weights.config, 8 * batch
    torch.manual_seed(777)
    pool = ContinuationPool.for_model(c, batch + 2, 2, "cuda:0")
    starts = torch.tensor([1019, 0, 4, 1021, 2047, 128, 1023, 0][:batch], device="cuda", dtype=torch.int64)
    counts = torch.tensor([7, 5, 7, 5, 3, 8, 1, 0][:batch], device="cuda", dtype=torch.int64)
    if batch > 1:
        counts[-1] = 0
    rows = torch.arange(1, batch + 1, device="cuda", dtype=torch.int64)
    table = torch.zeros((batch, 16), device="cuda", dtype=torch.int32)
    ids = torch.randint(0, c.vocabulary, (batch, 8), device="cuda", dtype=torch.int64)
    live = torch.tensor([batch], device="cuda", dtype=torch.int32)
    m = Metadata.allocate(tokens, batch, 16, "cuda:0", samples=tokens)
    m.set(starts, torch.full_like(starts, 8), rows, table, ids, live)
    auxiliary = torch.randn((tokens, 6 * c.hidden), device="cuda", dtype=torch.bfloat16) * .1
    for i in range(c.draft.layers):
        for field in ("key", "value"):
            history = pool.reserved_region(f"draft{i}.{field}")
            history.copy_(torch.randn(history.shape, dtype=torch.bfloat16, device=history.device) * .1)
    initial = pool.backing.clone()
    with PreparationSession(device="cuda:0", autotune=False) as session:
        arena = ScratchArena("cuda:0")
        context = PreparedContext(weights, auxiliary, m, pool, session=session,
            name="context", workspace=arena)
        scratch = Scratch.allocate(tokens, tokens, c, "cuda:0")
        draft = PreparedDraft(weights, scratch, m, pool, vocabulary_runtime=None, coordinator=None,
            session=session, name="draft", workspace=arena)
        context.functional()
        draft.release_outputs()
        arena.finalize()
        normalized = torch.empty_like(scratch.hidden)
        selector_scores = torch.empty((batch, 7, 8, 8), device="cuda", dtype=torch.float32)
        context_operation = ContextReadback(context, auxiliary, counts, m, normalized)
        draft_operation = DraftReadback(draft, m, scratch.hidden, scratch.logits, selector_scores)
        paths = (tmp_path / "context.bin", tmp_path / "draft.bin")
        with CompilePool(auxiliary.device) as compiler:
            futures = [compiler.submit(operation, path)
                       for operation, path in zip((context_operation, draft_operation), paths, strict=True)]
            for future in futures:
                future.result()
        artifacts = []
        for path in paths:
            with path.open("rb") as file:
                artifacts.append(torch.compiler.load_compiled_function(file))
        context_fn = lambda: artifacts[0](context_operation)
        draft_fn = lambda: artifacts[1](draft_operation)
        context_fn()
        draft_fn()
        session.freeze()
        context_graph, draft_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(context_graph):
            context_fn()
        with session.capture(), torch.cuda.graph(draft_graph):
            draft_fn()
        pool.backing.copy_(initial)
        context_graph.replay()
        torch.cuda.synchronize()
        projected = norm(linear(raw, auxiliary, "fc"), raw["hidden_norm.weight"])
        delta = normalized.float() - projected.float()
        assert delta.norm() / projected.float().norm() < .002
        torch.testing.assert_close(pool.reserved_region("draft_anchor")[rows[counts > 0]],
            projected[(m.query_start_loc[:-1] + counts - 1)[counts > 0]], rtol=.03, atol=.03)
        for i in range(c.draft.layers):
            prefix = f"layers.{i}.self_attn."
            key = rotary(norm(linear(raw, projected, prefix + "k_proj").view(tokens, 2, 128), raw[prefix + "k_norm.weight"]), m.positions)
            value = linear(raw, projected, prefix + "v_proj").view(tokens, 2, 128)
            for row in range(batch):
                n = counts[row].item()
                offsets = (starts[row] + torch.arange(n, device="cuda")) % 1024
                for field, expected in (("key", key), ("value", value)):
                    actual = pool.reserved_region(f"draft{i}.{field}")[rows[row], offsets]
                    target = expected[row * 8:row * 8 + n].to(actual.dtype).float()
                    if n:
                        difference = actual.float() - target
                        assert difference.norm() / target.norm() < .01
                        # Projection error and FP8 bin-boundary rounding both contribute.
                        ulp = torch.exp2(torch.floor(torch.log2(target.abs().clamp_min(2 ** -6))) - 3)
                        assert (difference.abs() <= .02 * target.abs().max() + ulp).all()
        starts += counts
        lengths = torch.where(counts > 0, 8, 0)
        m.set(starts, lengths, rows, table, ids, live)
        gc.collect()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
        for _ in range(3):
            draft_graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
        x = weights.embedding[m.token_ids]
        for i in range(c.draft.layers):
            p = f"layers.{i}."
            normalized = norm(x, raw[p + "input_layernorm.weight"])
            coeff = linear(raw, normalized, p + "attention_conv.kernel_projection")
            before = convolve(normalized, coeff, raw[p + "attention_conv.base_kernel"], 0)
            q, k, v = (linear(raw, before, p + "self_attn." + field) for field in ("q_proj", "k_proj", "v_proj"))
            q = rotary(norm(q.view(tokens, 32, 128), raw[p + "self_attn.q_norm.weight"]), m.positions)
            k = rotary(norm(k.view(tokens, 2, 128), raw[p + "self_attn.k_norm.weight"]), m.positions)
            v = v.view(tokens, 2, 128)
            attended = torch.zeros_like(q)
            for row in range(batch):
                if not lengths[row]:
                    continue
                start = starts[row].item()
                positions = torch.arange(max(0, start - 1024), start + 8, device="cuda")
                past = positions[positions < start] % 1024
                key = torch.cat((pool.reserved_region(f"draft{i}.key")[rows[row], past].bfloat16(),
                                  k[row * 8:(row + 1) * 8].to(torch.float8_e4m3fn).bfloat16()))
                value = torch.cat((pool.reserved_region(f"draft{i}.value")[rows[row], past].bfloat16(),
                                    v[row * 8:(row + 1) * 8].to(torch.float8_e4m3fn).bfloat16()))
                key, value = (t.repeat_interleave(16, 1).transpose(0, 1).float() for t in (key, value))
                scores = torch.matmul(q[row * 8:(row + 1) * 8].transpose(0, 1).float(), key.transpose(1, 2)) / 128 ** .5
                scores.masked_fill_(positions[None, None] < (start + torch.arange(8, device="cuda"))[None, :, None] - 1023, -torch.inf)
                sink = raw[p + "self_attn.attention_sink_bias"].float()[:, None, None].expand(-1, 8, 1)
                probs = torch.cat((scores, sink), -1).softmax(-1)[..., :-1]
                attended[row * 8:(row + 1) * 8] = (probs @ value).transpose(0, 1)
            output = linear(raw, attended.flatten(1), p + "self_attn.o_proj")
            x = x + convolve(output, coeff, raw[p + "attention_conv.base_kernel"], 1)
            normalized = norm(x, raw[p + "post_attention_layernorm.weight"])
            coeff = linear(raw, normalized, p + "mlp_conv.kernel_projection")
            before = convolve(normalized, coeff, raw[p + "mlp_conv.base_kernel"], 0)
            gate, up = (linear(raw, before, p + "mlp." + field) for field in ("gate_proj", "up_proj"))
            output = linear(raw, (F.silu(gate.float()) * up.float()).bfloat16(), p + "mlp.down_proj")
            x = x + convolve(output, coeff, raw[p + "mlp_conv.base_kernel"], 1)
        expected = norm(x, raw["norm.weight"])[m.sample_indices]
        active = torch.arange(tokens, device="cuda") < lengths.sum()
        error = (scratch.hidden[active].float() - expected[active].float()).norm() / expected[active].float().norm()
        assert error < .025, error.item()
        assert torch.isfinite(scratch.logits[active]).all()
        assert torch.isfinite(selector_scores).all()
