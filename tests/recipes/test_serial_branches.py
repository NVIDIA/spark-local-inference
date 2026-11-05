# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Recipes compose each concurrent branch on the main stream when stream overlap is disabled."""

from types import SimpleNamespace

import pytest
import torch

from ignition.preparation.scratch import ScratchArena


class Operator:
    """A prepared operator stand-in that records its launch and returns its output."""

    def __init__(self, launches, name, output=None):
        self.launches, self.name, self.output = launches, name, output
        self.plan = SimpleNamespace(name=name)

    def __call__(self, *operands):
        self.launches.append(self.name)
        return self.output


def operator(launches, name, output=None):
    return lambda *args, **kwargs: Operator(launches, name, output)


class SharedExpertOverlap:
    """Records construction and fork/join placement in place of ConcurrentSharedExpert."""

    def __init__(self, launches, constructed, mlp, *, parallel_plans, **options):
        self.launches, self.mlp = launches, mlp
        constructed.append(tuple(plan.name for plan in parallel_plans))

    def start(self, source):
        self.launches.append("fork")

    def finish(self, source, routed):
        self.launches.append("join")
        return self.mlp.output


@pytest.mark.parametrize("overlap", [True, False])
def test_glm_shared_expert_overlaps_routing_or_follows_routed_experts(monkeypatch, overlap):
    from ignition.recipes.glm53 import moe

    launches, constructed, rows = [], [], 4
    routed = torch.full((rows, 4096), 2, dtype=torch.bfloat16)
    shared = torch.full((rows, 4096), 3, dtype=torch.bfloat16)
    monkeypatch.setattr(moe, "UnquantizedProjection", operator(launches, "router"))
    monkeypatch.setattr(moe, "PreparedRouter", operator(launches, "route"))
    monkeypatch.setattr(moe, "PreparedMLP", operator(launches, "shared", shared))
    monkeypatch.setattr(moe, "PreparedRoutedExperts", operator(launches, "experts", routed))
    monkeypatch.setattr(moe, "ConcurrentSharedExpert",
                        lambda mlp, **options: SharedExpertOverlap(launches, constructed, mlp, **options))
    gate = SimpleNamespace(num_group=1, topk_group=1, top_k=8, num_experts=288, norm_topk_prob=True,
                           routed_scaling_factor=2.5)
    weights = SimpleNamespace(router=torch.zeros((288, 4096), dtype=torch.bfloat16), correction=torch.zeros(288),
                              shared=None, experts_a4_prefill=False, trellismx_release=None,
                              experts=SimpleNamespace(num_experts=288, hidden_size=4096, intermediate_size=1024))
    source = torch.zeros((rows, 4096), dtype=torch.bfloat16)
    prepared = moe.PreparedGlmMoe(SimpleNamespace(gate=gate, config=SimpleNamespace(swiglu_limit=None)), source,
                                  weights, moe.MoeScratch.allocate(rows, 2, "cpu"), session=None, name="moe",
                                  workspace=ScratchArena("cpu", overlap=overlap))
    output = prepared(source)
    if overlap:
        assert constructed == [("route", "experts")]
        assert launches == ["router", "fork", "route", "experts", "join"]
    else:
        assert constructed == [] and prepared.concurrent_shared is None
        assert launches == ["router", "route", "experts", "shared"]
    assert torch.equal(output, routed + shared)


@pytest.mark.parametrize("overlap", [True, False])
def test_qwen_shared_expert_overlaps_routing_or_precedes_it_on_the_main_stream(monkeypatch, overlap):
    from ignition.recipes.qwen38 import moe

    launches, constructed, rows = [], [], 2
    routed = torch.full((rows, 2560), 2, dtype=torch.bfloat16)

    class SharedExpert:
        def __init__(self, source, weights, **options):
            self.gate_projection = Operator(launches, "gate")
            self.output = None

        def project(self, source):
            launches.append("shared")

        def finish(self, routed=None, *, out=None):
            launches.append("gated sum")
            return out.copy_(routed)

    monkeypatch.setattr(moe, "UnquantizedProjection", operator(launches, "router"))
    monkeypatch.setattr(moe, "PreparedRouter", operator(launches, "route"))
    monkeypatch.setattr(moe, "PreparedSharedExpert", SharedExpert)
    monkeypatch.setattr(moe, "PreparedRoutedExperts", operator(launches, "experts", routed))
    monkeypatch.setattr(moe, "register_finalizer", lambda session, callback, *, workspace: None)
    monkeypatch.setattr(moe, "ConcurrentSharedExpert",
                        lambda mlp, **options: SharedExpertOverlap(launches, constructed, mlp, **options))
    weights = SimpleNamespace(router=torch.zeros((512, 2560), dtype=torch.bfloat16), shared=None,
                              experts=SimpleNamespace(num_experts=512, hidden_size=2560, intermediate_size=512))
    arena = ScratchArena("cpu", overlap=overlap)
    source = torch.zeros((rows, 2560), dtype=torch.bfloat16)
    prepared = moe.PreparedQwenMoe(source, weights, session=None, name="moe", workspace=arena,
                                   projection_workspace=arena)
    output = prepared(source)
    if overlap:
        assert constructed == [("router", "route", "experts")]
        assert launches == ["fork", "router", "route", "experts", "join", "gated sum"]
    else:
        assert constructed == [] and prepared.concurrent_shared is None
        assert launches == ["shared", "router", "route", "experts", "gated sum"]
    assert torch.equal(output, routed)


@pytest.mark.parametrize("overlap", [True, False])
def test_glm_kda_gate_chain_overlaps_input_projections_or_follows_them(monkeypatch, overlap):
    from ignition.recipes.glm53 import kda

    launches = []
    scratch = kda.KDAScratch.allocate(2, 16, "cpu")

    class Projection(Operator):
        def __init__(self, name, source, output):
            super().__init__(launches, name, output)
            self.source = source

        def __call__(self, source):
            assert source is self.source, self.name
            return super().__call__(source)

    class GateOverlap:
        def start(self, source):
            launches.append("fork")

        def finish(self, source, dependency):
            assert dependency is scratch.raw_g
            launches.append("join")

    source = torch.zeros((2, 4096), dtype=torch.bfloat16)
    layer = object.__new__(kda.PreparedGlmKDA)
    layer.scratch, layer.tokens, layer.mode = scratch, 2, "decode"
    layer.verification = layer.records = None
    layer.conv_binding, layer.recurrence, layer.options = "conv", "recurrence", {}
    layer.projections = {key: Projection(key, input, output) for key, input, output in (
        ("input_proj", source, scratch.projected_input),
        ("f_b_proj", scratch.forget_input, scratch.raw_g),
        ("g_a_proj", source, scratch.gate_input),
        ("g_b_proj", scratch.gate_input, scratch.z),
        ("o_proj", scratch.gated, scratch.output))}
    layer.gate_overlap = GateOverlap() if overlap else None
    monkeypatch.setattr(kda.short_conv, "run", lambda binding: launches.append(binding))
    monkeypatch.setattr(kda.gdn_decode, "run_kda", lambda binding, **options: launches.append(binding))
    assert layer(source) is scratch.output
    gate = ["fork"] if overlap else []
    joined = ["join"] if overlap else ["g_a_proj", "g_b_proj"]
    # The main stream's order is unchanged; the output-gate chain runs where the main stream would join it.
    assert launches == [*gate, "input_proj", "f_b_proj", *joined, "conv", "recurrence", "o_proj"]
