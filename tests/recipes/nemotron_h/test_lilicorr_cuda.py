# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LiLiCorr checkpoint scores against independently decoded PyTorch computation."""

import gc
import os
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("batch", [1, 4, 8])
@torch.no_grad()
def test_selector_returns_local_masked_logits(batch):
    from ignition.preparation.compilation import compile_forward
    from ignition.recipes.nemotron_h.lilicorr import PreparedLatticeSelector

    selector = object.__new__(PreparedLatticeSelector)
    selector.batch, selector.top_k, selector.slots = batch, 8, 7
    logits = torch.randn((batch, 8, 131072), device="cuda", dtype=torch.float32)
    features = torch.randn((batch, 7, 8, 8), device="cuda", dtype=torch.float32)
    predecessor = torch.zeros((batch, 7), device="cuda", dtype=torch.int64)
    ids = (torch.arange(56, device="cuda").view(1, 7, 8).expand(batch, -1, -1) * 101) % 131072
    logits[:, 1:].scatter_(2, ids, 2.0)
    unary, selected_ids = selector.prepare_shortlists(logits, features=features)
    selected_ids[:, 1:].copy_(ids)
    retained = torch.empty((batch, 7, 131072), device="cuda", dtype=torch.float32)

    def forward():
        for step in range(7):
            retained[:, step].copy_(selector.apply_shortlist(logits[:, step + 1], predecessor[:, step],
                step + 1, features, unary, selected_ids))

    compiled = compile_forward(forward)
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    for candidate in (0, 5, None):
        features.normal_()
        predecessor.copy_(torch.roll(selected_ids[:, 1:, candidate or 0], 1, 1))
        if candidate is None:
            predecessor.fill_(-1)
        retained.fill_(torch.nan)
        torch.cuda.synchronize()
        allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
        expected = torch.full_like(retained, -torch.inf)
        for step in range(7):
            row = candidate if step and candidate is not None else 0
            expected[:, step].scatter_(1, selected_ids[:, step + 1], features[:, step, row])
        torch.testing.assert_close(retained, expected, rtol=0, atol=0)


def reference(weights, config, tokens, logp, passed, anchor, valid):
    b, slots, k, _ = tokens.shape
    d = config.text["dflash_config"]
    h, heads = d["lilicorr_hidden_size"], d["lilicorr_num_heads"]

    def linear(x, prefix):
        weight, scale = weights[prefix + ".weight"]
        bias = weights.get(prefix + ".bias")
        if isinstance(scale, float):
            return F.linear(x, weight, bias)
        result = (F.linear(x.float(), weight.float()) * scale).bfloat16()
        return result if bias is None else result + bias

    def norm(x, prefix):
        return F.rms_norm(x, (h,), weights[prefix + ".weight"][0], config.text["rms_norm_eps"])

    def mlp(x, prefix):
        return linear(F.silu(linear(x, prefix + ".up_proj")), prefix + ".down_proj")

    rank = torch.arange(k, device=tokens.device).float()
    features = torch.stack((logp, logp.exp(), logp - logp.amax(-1, keepdim=True),
        (rank / (k - 1)).expand_as(logp), (rank == 0).float().expand_as(logp)), -1).bfloat16()
    features = F.layer_norm(features, (5,), weights["feature_norm.weight"][0], weights["feature_norm.bias"], 1e-5)
    x = linear(tokens, "token_proj") + linear(passed, "pass_hidden_proj")[:, :, None]
    x = x + mlp(features, "feature_mlp")
    x = x + weights["slot_embedding"][:, 0, :slots]
    x = x + weights["rank_embedding"][:, 0]
    x = x.view(b, slots * k, h)
    sid = torch.arange(slots, device=tokens.device).repeat_interleave(k)
    relative = sid[:, None] - sid[None]
    bias = weights["relative_slot_bias"][:, relative + config.block_size - 1]
    bias = bias + (relative == 0).bfloat16()[None] * weights["same_slot_bias"][:, None, None]
    for i in range(d["lilicorr_num_layers"]):
        p = f"layers.{i}."
        qkv = F.linear(norm(x, p + "attn_norm"), weights[p + "attn.in_proj_weight"], weights[p + "attn.in_proj_bias"])
        q, key, v = (t.view(b, slots * k, heads, h // heads).transpose(1, 2) for t in qkv.chunk(3, -1))
        attended = F.scaled_dot_product_attention(q, key, v, attn_mask=bias[None]).transpose(1, 2).reshape_as(x)
        x = x + linear(attended, p + "attn.out_proj")
        x = x + mlp(norm(x, p + "mlp_norm"), p + "mlp")
    x = norm(x, "output_norm").view(b, slots, k, h)
    a = norm(linear(anchor, "context_proj") * valid[:, None].bfloat16(), "anchor_norm")
    w = weights["factor_input_proj.weight"][0]
    wx, wa, wc = w.split(h, -1)
    pre = F.linear(x, wx, weights["factor_input_proj.bias"]) + F.linear(a[:, None, None], wa)
    pre = pre + F.linear(x * a[:, None, None], wc)
    f = F.silu(pre)
    out = F.normalize(linear(f, "out_head"), dim=-1, eps=d["lilicorr_vector_eps"])
    incoming = F.normalize(linear(f, "in_head"), dim=-1, eps=d["lilicorr_vector_eps"])
    a = F.normalize(linear(a, "anchor_out_head"), dim=-1, eps=d["lilicorr_vector_eps"])
    first = (a[:, None] * incoming[:, 0]).sum(-1).float() * d["lilicorr_logit_scale"]
    pairs = torch.matmul(out[:, :-1], incoming[:, 1:].transpose(-1, -2)).float() * d["lilicorr_logit_scale"]
    return torch.cat((first[:, None, None].expand(-1, 1, k, -1), pairs), 1)


@pytest.mark.parametrize("batch,slots", [(1, 1), (1, 7), (3, 7), (4, 7)])
@torch.no_grad()
def test_checkpoint_lattice_scores(batch, slots, tmp_path):
    from b12x.preparation import PreparationSession
    from safetensors import safe_open
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.compile_pool import CompilePool
    from ignition.preparation.scratch import ScratchArena
    from ignition.recipes.nemotron_h.config import ModelConfig
    from ignition.recipes.nemotron_h.draft_config import DraftConfig
    from ignition.recipes.nemotron_h.draft_weights import DraftWeightLoader
    from ignition.recipes.nemotron_h.lilicorr import PreparedLattice
    from ignition.reference.qwen38_quant import decode_nvfp4

    target = Path(os.environ.get("FLARE_NEMOTRON_CHECKPOINT", "/data/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB"))
    path = Path(os.environ.get("FLARE_LILICORR_CHECKPOINT", str(target) + "-LiLiCorr"))
    if not target.is_dir() or not path.is_dir():
        pytest.skip("requires the Nemotron Bolt target and LiLiCorr checkpoints")
    config = DraftConfig.load(path, ModelConfig.load(target))
    manifest = inspect_indexed_checkpoint(path)
    with DraftWeightLoader(manifest, "cuda:0", prefixes=("lilicorr.",)) as loader:
        weights = loader.lattice(config)
    raw = {}
    with safe_open(path / "model.safetensors", framework="pt", device="cuda:0") as reader:
        for name in reader.keys():
            if name.startswith("lilicorr."):
                raw[name.removeprefix("lilicorr.")] = reader.get_tensor(name)
    references = {}
    for name, value in raw.items():
        if name.endswith(".weight"):
            prefix = name.removesuffix(".weight")
            references[name] = ((decode_nvfp4(value, raw[prefix + ".weight_scale"],
                torch.ones_like(raw[prefix + ".weight_scale_2"])).bfloat16(), raw[prefix + ".weight_scale_2"])
                if value.dtype == torch.uint8 else (value, 1.))
        elif not name.endswith(("weight_scale", "weight_scale_2")):
            references[name] = value
    torch.manual_seed(817)
    tokens = torch.randn((batch, slots, 8, 4096), device="cuda", dtype=torch.bfloat16) * .2
    passed = torch.randn((batch, slots, 4096), device="cuda", dtype=torch.bfloat16) * .2
    anchor = torch.randn((batch, 4096), device="cuda", dtype=torch.bfloat16) * .2
    logp = torch.randn((batch, slots, 8), device="cuda").log_softmax(-1) - 2
    valid = torch.arange(batch, device="cuda") % 2 == 0
    with PreparationSession(device="cuda:0", autotune=False) as session:
        arena = ScratchArena("cuda:0")
        lattice = PreparedLattice(weights, config, tokens, logp, passed, anchor, session=session,
            name="lilicorr", workspace=arena)
        lattice.functional(session=session)
        arena.finalize()
        operation = partial(lattice, tokens, logp, passed, anchor, valid)
        path = tmp_path / "lattice.bin"
        with CompilePool(tokens.device) as compiler:
            compiler.submit(operation, path).result()
        with path.open("rb") as file:
            artifact = torch.compiler.load_compiled_function(file)
        forward = lambda: artifact(operation)
        forward()
        session.freeze()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            actual = forward()
        for _ in range(2):
            tokens.normal_(std=.2)
            valid.logical_not_()
            gc.collect()
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            expected = reference(references, config, tokens, logp, passed, anchor, valid)
            error = (actual - expected).norm() / expected.norm()
            assert error < .03, error.item()
            torch.testing.assert_close(actual, expected, rtol=.05, atol=.06)
