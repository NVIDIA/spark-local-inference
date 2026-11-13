# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The graph qualifier rejects stale scratch pointers and escaping arena outputs."""

from types import SimpleNamespace

import pytest
import torch
from b12x._lib.scratch import ScratchBufferSpec
from b12x.moe import fused_moe
from b12x.preparation import MemoryRequirements, Plan
from b12x.preparation.types import _CompositePlan, _Prepared

from ignition.preparation.scratch import ScratchArena
from qualify import graph_execution


@pytest.fixture
def audit_type():
    return graph_execution._ScratchArenaAudit


def prepared_plan(arena, *, owners=None):
    spec = ScratchBufferSpec(name="scratch", shape=(32,), dtype=torch.uint8, device=arena.device)
    plan = Plan(contract=SimpleNamespace(component_id="test.scratch"), query=None,
        _compile_jobs=lambda *args: (), _memory_requirements=lambda *args: MemoryRequirements(scratch=(spec,)),
        _materialize=lambda *args: None)
    payload = _Prepared(state=None, selection=SimpleNamespace(config=None), programs=frozenset(),
        retained=None, owners=(arena,) if owners is None else owners, closers=(), device=None, scratch=(spec,))
    object.__setattr__(plan, "_prepared", payload)
    payload.users.append(plan)
    return plan


def test_arena_audit_uses_strided_byte_extent_and_preserves_contents(audit_type):
    arena = ScratchArena("cpu")
    arena.reserve(1024).fill_(173)
    arena.freeze()
    audit = audit_type(arena)
    view = arena.tensor[256:512].view(torch.float32).as_strided((2, 3), (8, 1))
    audit.view("binding.internal", view)
    audit.view("binding.empty", arena.tensor[1024:])
    report = audit.report()
    assert report["views"]["binding.internal"] == {
        "dtype": "torch.float32", "shape": [2, 3], "strides": [8, 1],
        "offset_bytes": 256, "extent_bytes": 44, "logical_bytes": 24,
    }
    assert report["highest_view_end_bytes"] == 1024
    assert torch.all(arena.tensor == 173)


def test_arena_audit_rejects_stale_internal_view_after_outer_binding_refresh(audit_type):
    arena = ScratchArena("cpu")
    previous = arena.reserve(256)
    arena.reserve(512)
    arena.freeze()
    binding = SimpleNamespace(scratch=arena.tensor, request_ids=previous[:64].view(torch.int32))
    audit = audit_type(arena)
    with pytest.raises(AssertionError, match=r"binding.request_ids does not fit"):
        audit.binding("binding", "convolution", binding, ("scratch", "request_ids"))


def test_arena_audit_rejects_same_storage_view_outside_declared_arena(audit_type):
    arena = ScratchArena("cpu")
    whole = arena.reserve(512)
    arena.tensor = whole[:256]
    arena.freeze()
    with pytest.raises(AssertionError, match="does not fit"):
        audit_type(arena).view("binding.internal", whole[255:257])


def test_arena_audit_rejects_escaping_output_in_workspace(audit_type):
    arena = ScratchArena("cpu")
    arena.reserve(512)
    arena.freeze()
    with pytest.raises(AssertionError, match="must survive scratch arena reuse"):
        audit_type(arena).external("projection.output", arena.tensor[:128])


@pytest.mark.parametrize("stale", [False, True])
def test_arena_audit_checks_public_moe_views_but_excludes_weight_scales(audit_type, stale):
    arena = ScratchArena("cpu")
    old = arena.reserve(256)
    arena.reserve(512)
    arena.freeze()
    scales = torch.ones(2)
    output = torch.zeros((1, 4), dtype=torch.bfloat16)
    binding = fused_moe.Binding(
        a=torch.zeros_like(output), experts=SimpleNamespace(a1_gscale=scales, a2_gscale=scales),
        topk_ids=torch.zeros((1, 1), dtype=torch.int32), topk_weights=torch.ones((1, 1)),
        implementation="dynamic", state_E=2, weight_E=2, max_rows=1, k=4, n=4,
        num_topk=1, device=torch.device("cpu"), dtype=torch.bfloat16, output=output,
        row_counts=(old if stale else arena.tensor)[:8].view(torch.int32),
        input_gs=scales.view(2), down_input_scale=scales.view(2),
    )
    operator = SimpleNamespace(scratch=arena.tensor, binding=binding, output=output, plan=prepared_plan(arena))
    audit = audit_type(arena)
    if stale:
        with pytest.raises(AssertionError, match=r"experts.binding.row_counts does not fit"):
            audit.moe("experts", operator)
    else:
        audit.moe("experts", operator)
        assert set(audit.views) == {"experts.scratch", "experts.binding.row_counts"}
        assert audit.families == {"moe": 1}


def test_arena_audit_rejects_projection_with_owned_scratch(audit_type):
    from ignition.components.linear import QuantizedProjection

    arena = ScratchArena("cpu")
    arena.reserve(512)
    arena.freeze()
    projection = QuantizedProjection.__new__(QuantizedProjection)
    projection.plan = SimpleNamespace(query=SimpleNamespace(workspace_form="owned"),
        memory_requirements=lambda: SimpleNamespace(scratch_nbytes=0))
    projection.workspace = arena.tensor
    projection.output = torch.zeros((1, 4))
    with pytest.raises(AssertionError, match="privately owned projection scratch"):
        audit_type(arena).projection("projection", projection)


@pytest.mark.parametrize("stale", [False, True])
def test_arena_audit_unwraps_public_varlen_binding(audit_type, stale):
    from b12x.attention._shared.contiguous.api import VarlenAttentionBinding
    from b12x.attention.varlen import VarlenBinding

    arena = ScratchArena("cpu")
    old = arena.reserve(256)
    arena.reserve(512)
    arena.freeze()
    source = torch.zeros((1, 1, 4))
    lengths = torch.tensor([0, 1], dtype=torch.int32)
    kernel_binding = VarlenAttentionBinding(q=source, k=source, v=source,
        cu_seqlens_q=lengths, cu_seqlens_k=lengths, plan=None,
        output=arena.tensor[:16].view(torch.float32).view_as(source),
        lse=(old if stale else arena.tensor)[16:20].view(torch.float32))
    binding = VarlenBinding(plan=prepared_plan(arena), binding=kernel_binding)
    audit = audit_type(arena)
    if stale:
        with pytest.raises(AssertionError, match=r"vision.lse does not fit"):
            audit.vision_binding("vision", binding)
    else:
        audit.vision_binding("vision", binding)
        assert set(audit.views) == {"vision.output", "vision.lse"}
        assert audit.families == {"vision_attention": 1}


@pytest.mark.parametrize("retention", ["arena", "obsolete_tensor", "binding", "missing"])
def test_prepared_owner_audit_catches_leak_despite_correct_final_pointer(audit_type, retention):
    arena = ScratchArena("cpu")
    previous = arena.reserve(256)
    arena.reserve(512)
    arena.trim(256)
    arena.freeze()
    owners = {"arena": (arena,), "obsolete_tensor": (arena, previous),
        "binding": (SimpleNamespace(scratch=previous),), "missing": ()}[retention]
    plan = prepared_plan(arena, owners=owners)
    audit = audit_type(arena)
    audit.view("final_binding.scratch", arena.tensor)
    if retention == "arena":
        audit.plan_owners("operator", plan)
        assert audit.report()["plans"]["operator"]["arena_owner_count"] == 1
    else:
        message = "no arena owner" if retention == "missing" else "concrete priming owner"
        with pytest.raises(AssertionError, match=message):
            audit.plan_owners("operator", plan)


@pytest.mark.parametrize("stale_child", [False, True])
def test_prepared_owner_audit_checks_composite_children(audit_type, stale_child):
    arena = ScratchArena("cpu")
    old = arena.reserve(256)
    arena.reserve(512)
    arena.freeze()
    children = {1: prepared_plan(arena), 4: prepared_plan(arena, owners=(old,) if stale_child else (arena,))}
    plan = _CompositePlan(component_id="test.composite", capacity_metadata={}, variants=children,
        _assemble=lambda *args: None)
    payload = _Prepared(state=None, selection=None, programs=frozenset(), retained=None,
        owners=(), closers=(), device=None, scratch=children[4].scratch_specs(), variants=plan.variants)
    object.__setattr__(plan, "_prepared", payload)
    audit = audit_type(arena)
    if stale_child:
        with pytest.raises(AssertionError, match=r"composite.variant.4 retains a concrete"):
            audit.plan_owners("composite", plan)
    else:
        audit.plan_owners("composite", plan)
        assert set(audit.plans) == {"composite", "composite.variant.1", "composite.variant.4"}
        assert audit.report()["distinct_prepared_payloads"] == 3
