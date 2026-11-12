# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decoder composition selects expert overlap from physical rows and execution mode."""

from types import SimpleNamespace

import pytest
import torch

from ignition.recipes.deepseek41 import decoder


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 8, 33, 48, 64, 96, 128, 129])
@pytest.mark.parametrize("mode", ["decode", "verify", "draft", "prefill"])
@pytest.mark.parametrize("overlap", [False, True])
def test_layer_overlap_uses_physical_capacity_and_enabled_workspace(monkeypatch, rows, mode, overlap):
    scratch = decoder.Scratch.allocate(rows, 1, 4, "meta")
    calls = []

    class Component:
        plan = object()

        def functional(self, **kwargs):
            return self

    def component(*args, **kwargs):
        return Component()

    def concurrent(shared, **kwargs):
        calls.append((shared, kwargs))
        return Component()

    for name in ("PreparedMHC", "PreparedAttention", "PreparedAllReduce", "UnquantizedProjection",
                 "PreparedRouter", "PreparedRoutedExperts", "PreparedMLP", "PreparedMHCPost"):
        monkeypatch.setattr(decoder, name, component)
    monkeypatch.setattr(decoder, "_mhc_weights", lambda *args: None)
    monkeypatch.setattr(decoder, "register_finalizer", lambda *args, **kwargs: None)
    monkeypatch.setattr(decoder, "ConcurrentSharedExpert", concurrent)
    text = dict(rms_norm_eps=1e-6, hc_eps=1e-6, hc_sinkhorn_iters=20,
                scoring_func="sigmoid", norm_topk_prob=True, routed_scaling_factor=1., swiglu_limit=10.)
    weights = SimpleNamespace(engram=None, attention=None, experts=SimpleNamespace(
        vision_correction=None, router=None, correction=None, experts=None, shared=None))
    workspace = SimpleNamespace(overlap=overlap)
    session = object()
    layer = decoder.PreparedLayer(3, weights, None, scratch,
        SimpleNamespace(token_ids=torch.empty(rows, device="meta")),
        SimpleNamespace(config=SimpleNamespace(text=text)), None,
        shard=SimpleNamespace(world_size=4), mode=mode, runtime=None, coordinator=None,
        session=session, name="layer", workspace=workspace, carry=False, defer_post=True)
    expected = overlap and mode in ("decode", "verify", "draft") and rows <= 128
    assert (layer.concurrent_shared is not None) == expected
    if expected:
        shared, options = calls.pop()
        assert shared is layer.shared
        assert options == dict(parallel_plans=(layer.router_projection.plan, layer.router.plan,
            layer.experts.plan), session=session, workspace=workspace, defer_launch=rows > 4)
    assert not calls


def test_layer_consumes_standalone_router_results_with_runtime_image_mask():
    layer = object.__new__(decoder.PreparedLayer)
    layer.engram = layer.concurrent_shared = None
    layer.has_image_mask = True
    # Built without __init__, so give the layer an unsharded layer's shard state.
    layer.row_shard, layer.reseed = None, False
    layer.attention_shard = False
    layer.attention_gather = layer.ffn_gather = None
    layer.residual_gather = layer.mix_gather = None
    layer.selection_move = layer.router_gather = None
    residual = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    metadata = SimpleNamespace(token_ids=torch.tensor([7, 129280, 9]))
    selection = (torch.zeros(3, 1, dtype=torch.int32), None, None)
    layer.attention_hc = lambda source, **kwargs: (source, source, source, source)
    layer.attention = SimpleNamespace(_run=lambda source, m, selected, **kwargs:
        (source * 2, source, selected, {}))
    layer.attention_reduce = lambda source: source
    layer.ffn_hc = lambda source, *args, **kwargs: (source, source, source, source)
    layer.router_projection = lambda source: source + 1
    observed = []
    def route(logits, *, image_mask):
        observed.append((logits.clone(), image_mask.clone()))
        return SimpleNamespace(ids=torch.tensor([[0], [1], [0]]), weights=logits[:, :1],
                               selected_logits=logits[:, :1])
    layer.router = route
    def experts(source, ids, weights, **kwargs):
        assert torch.equal(ids, torch.tensor([[0], [1], [0]]))
        assert torch.equal(weights, source[:, :1] + 1)
        return source * weights
    layer.experts = experts
    layer.shared = lambda source: source + 3
    layer.mlp_reduce = lambda source: source
    layer.post = lambda source, *args: source
    incoming, _, records, actual_selection = layer((residual,), torch.zeros(3, 4),
                                                  metadata, None, selection)
    normalized = residual * 2
    assert torch.equal(incoming[0], normalized * (normalized[:, :1] + 1) + normalized + 3)
    assert torch.equal(observed[0][0], normalized + 1)
    assert torch.equal(observed[0][1], torch.tensor([False, True, False]))
    assert records is residual
    assert actual_selection is selection
