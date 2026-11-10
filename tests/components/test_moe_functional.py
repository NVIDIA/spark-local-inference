# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Forward-local route, expert, and asynchronous operand ownership."""

from types import SimpleNamespace
import weakref
import gc

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from ignition.components import block_fp8, expert_overlap, mlp, moe, moe_sm103


@pytest.mark.parametrize("arena_owned", [False, True])
def test_scale_priming_drops_binding_templates_from_arena_owned_payload(arena_owned):
    from b12x._lib.quant.csf_blocks_tuning import _Binding

    references = {}

    class Decoder:
        def __init__(self, active):
            self.active = active

        def decode(self, ids, out13, out2):
            assert not ids.count_nonzero()
            out13.fill_(3)
            out2.fill_(5)

    class State:
        def bind(self, *, ids, out13, out2, scratch):
            binding = _Binding(Decoder(scratch), ids, out13, out2)
            references.update(ids=weakref.ref(ids), active=weakref.ref(scratch),
                out13=weakref.ref(out13), out2=weakref.ref(out2), binding=weakref.ref(binding))
            return binding

    workspace = moe.ScratchArena("cpu") if arena_owned else None
    experts = SimpleNamespace(num_experts=2, _impl=SimpleNamespace(
        w1_blockscale=torch.empty(8), w2_blockscale=torch.empty(4)))
    call = moe._prepare_scale_expansion(State(), ids=torch.ones((3, 2), dtype=torch.int32),
        experts=experts, workspace=workspace)
    call.invoke()
    assert experts._impl.w1_blockscale.eq(3).all()
    assert experts._impl.w2_blockscale.eq(5).all()
    retained_owners = tuple(call.owners)
    del call, experts
    gc.collect()
    if arena_owned:
        assert retained_owners == (workspace,)
        assert all(reference() is None for reference in references.values())
    else:
        assert all(reference() is not None for reference in references.values())
    del retained_owners
    gc.collect()
    assert all(reference() is None for reference in references.values())


@pytest.mark.parametrize("counts", [False, True])
def test_router_returns_local_routes_and_requires_declared_metadata(monkeypatch, counts):
    declarations = []
    monkeypatch.setattr(moe.fused_moe, "plan_route_topk", lambda query: SimpleNamespace(
        query=query, request=lambda **kwargs: SimpleNamespace(**kwargs)))
    logits = torch.zeros((3, 8))
    templates = moe.RouterScratch.allocate(3, "cpu", experts=8, top_k=2)
    live = torch.tensor([2], dtype=torch.int32) if counts else None
    owner = moe.PreparedRouter(logits, templates,
        session=SimpleNamespace(prepare=lambda requests: declarations.extend(requests)),
        name="route", num_valid_tokens=live)
    old_logits, old_ids = weakref.ref(logits), weakref.ref(templates.ids)
    owner.functional(session=None)
    declarations.clear()
    del logits, templates, live
    assert old_logits() is None and old_ids() is None
    seen = []
    def route(plan, logits, selected, ids, weights, **options):
        seen.append((selected, ids, weights, options))
        ids.fill_(1)
        weights.fill_(0.5)
        selected.fill_(2)
    monkeypatch.setattr(moe.fused_moe, "route_topk", route)
    runtime = torch.ones((3, 8))
    live = torch.tensor([1], dtype=torch.int32)
    options = {"num_valid_tokens": live} if counts else {}
    first, second = owner(runtime, **options), owner(runtime, **options)
    assert first.ids.tolist() == [[1, 1]] * 3
    assert second.ids.data_ptr() != first.ids.data_ptr()
    assert owner.scratch is owner.logits is None
    if counts:
        assert seen[-1][-1]["num_valid_tokens"] is live
        with pytest.raises(ValueError, match="requires num_valid_tokens"):
            owner(runtime)


def _expert_owner(grouped=False):
    owner = moe.PreparedRoutedExperts.__new__(moe.PreparedRoutedExperts)
    owner.output = owner.source = owner.output_source = None
    owner.output_layout = ((2, 3, 8), (24, 8, 1), torch.bfloat16) if grouped else ((2, 8), (8, 1), torch.bfloat16)
    owner.workspace_nbytes = 256
    owner.scale_inputs, owner.scale_layouts = [], ()
    owner.integrated_routing, owner.grouped_output = False, grouped
    owner.num_experts, owner.precision = 16, {"a4_prefill": False}
    owner.plan = SimpleNamespace(handle=1)
    return owner


@pytest.mark.parametrize("grouped", [False, True])
def test_expert_lowering_exposes_local_output_and_workspace(grouped):
    owner = _expert_owner(grouped)
    source = torch.ones((2, 8), dtype=torch.bfloat16)
    ids = torch.zeros((2, 3), dtype=torch.int32)
    weights = torch.ones((2, 3))
    def forward(source, ids, weights):
        output = owner(source, ids, weights)
        return output.values if grouped else output
    graph = make_fx(forward, tracing_mode="fake")(source, ids, weights)
    calls = [node for node in graph.graph.nodes if node.target == torch.ops.b12x.prepared_moe_out.default]
    assert len(calls) == 1
    assert calls[0].args[3].target == torch.ops.aten.empty_strided.default
    assert calls[0].args[4].target == torch.ops.aten.empty.memory_format
    assert owner.output is owner.source is None


def test_overlap_pending_retains_branch_operands_until_join(monkeypatch):
    owner = expert_overlap.ConcurrentSharedExpert.__new__(expert_overlap.ConcurrentSharedExpert)
    owner.outputs = None
    owner.output_layouts = tuple(((2, 8), (8, 1), torch.bfloat16) for _ in range(4))
    owner.workspace_nbytes = 256
    owner.encoded_layouts = [((2, 8), torch.float8_e4m3fn), ((2, 1), torch.float32)]
    owner.weights, owner.scales, owner.gains, owner.scale_kinds = [], [], [], []
    owner.plans, owner.encoding_offsets, owner.quantizers = [], [], []
    owner.activation, owner.limit, owner.round_silu = "swiglu", float("inf"), True
    owner.stream_handle, owner.launch_order = 1, 0
    calls = []
    monkeypatch.setattr(expert_overlap, "_start_shared_expert", lambda *args: calls.append(("start", args)))
    monkeypatch.setattr(expert_overlap, "_join_shared_expert", lambda *args: calls.append(("join", args)))
    source = torch.ones((2, 8), dtype=torch.bfloat16)
    pending = owner.start(source)
    result = owner.finish(pending, torch.zeros_like(source))
    assert result is pending.outputs[2] and pending.gate is pending.outputs[3]
    assert calls[0][1][0] is calls[1][1][0] is source
    assert calls[0][1][5] is calls[1][1][6] is pending.outputs
    assert calls[0][1][6] is calls[1][1][7] is pending.workspace
    assert calls[0][1][-3] is calls[1][1][-3] is pending.encoded
    assert owner.outputs is None


def test_sm103_staging_returns_explicit_rows_for_each_invocation(monkeypatch):
    owner = moe_sm103.Sm103RoutedExperts.__new__(moe_sm103.Sm103RoutedExperts)
    owner.output = None
    owner.plan = SimpleNamespace(handle=1, query=SimpleNamespace(routed_inputs=False), contract=moe_sm103.TRTLLM_CONTRACT)
    owner.workspace_nbytes = 128
    owner.buffer_layouts = (((1,), torch.int32),) * 18
    layouts = list(owner.buffer_layouts)
    layouts[1], layouts[13], layouts[17] = ((6,), torch.int32), ((8, 8), torch.bfloat16), ((2, 0), torch.bfloat16)
    owner.buffer_layouts = tuple(layouts)
    seen = []
    monkeypatch.setattr(moe_sm103, "_experts_out", lambda *args: seen.append(args))
    monkeypatch.setattr(moe_sm103, "_reduce_rows", lambda *args: seen.append(args))
    source = torch.ones((2, 8), dtype=torch.bfloat16)
    ids, weights = torch.zeros((2, 3), dtype=torch.int32), torch.ones((2, 3))
    first, second = owner.stage_rows(source, ids, weights), owner.stage_rows(source, ids, weights)
    assert first.rows.data_ptr() != second.rows.data_ptr()
    assert first.permutation.data_ptr() != second.permutation.data_ptr()
    shared = torch.zeros_like(source)
    result = owner.reduce(shared, staged=first)
    assert seen[-1][:4] == (first.rows, first.permutation, weights, shared)
    assert seen[-1][4] is result


def test_integrated_expert_lowering_exposes_mutable_routes_and_scale_planes():
    owner = _expert_owner()
    owner.integrated_routing = True
    owner.scale_layouts = (((32,), (1,), torch.uint8), ((16,), (1,), torch.uint8),
                           ((8,), (1,), torch.int32))
    owner.router = SimpleNamespace(runtime_options=lambda **options: options)
    source = torch.ones((2, 8), dtype=torch.bfloat16)
    ids, weights, selected = torch.zeros((2, 3), dtype=torch.int32), torch.ones((2, 3)), torch.zeros((2, 3))
    logits, count = torch.ones((2, 16)), torch.tensor([1], dtype=torch.int32)
    def forward(source, ids, weights, selected, logits, count):
        return owner(source, ids, weights, selected_logits=selected, router_logits=logits,
                     num_valid_tokens=count)
    graph = make_fx(forward, tracing_mode="fake")(source, ids, weights, selected, logits, count)
    call = next(node for node in graph.graph.nodes
                if node.target == torch.ops.b12x.prepared_routed_moe_out.default)
    inputs = [node for node in graph.graph.nodes if node.op == "placeholder"]
    assert call.args[2:5] == (inputs[3], inputs[2], inputs[1])
    assert len(call.args[8]) == 3
    assert all(node.target == torch.ops.aten.empty_strided.default for node in call.args[8])


def test_block_fp8_lowering_exposes_quantization_output_and_workspace(monkeypatch):
    monkeypatch.setattr(block_fp8, "plan_from_handle", lambda handle: SimpleNamespace(component_id="block_fp8"))
    source = torch.ones((2, 128), dtype=torch.bfloat16)
    values, scales = torch.ones((128, 128), dtype=torch.float8_e4m3fn), torch.ones((1, 1))
    graph = make_fx(lambda source, values, scales:
        block_fp8._project(source, values, scales, 1, workspace_nbytes=512),
        tracing_mode="fake")(source, values, scales)
    quantize = next(node for node in graph.graph.nodes
                    if node.target == torch.ops.flare.block_fp8_quantize_out.default)
    project = next(node for node in graph.graph.nodes
                   if node.target == torch.ops.flare.block_fp8_projection_out.default)
    assert project.args[:2] == quantize.args[1:]
    assert project.args[4].target == project.args[5].target == torch.ops.aten.empty.memory_format


@pytest.mark.parametrize("activation", ["relu2", "silu"])
def test_ungated_functional_mlp_preserves_bf16_rounding(activation):
    class Projection:
        def __init__(self, weight):
            self.weight = weight
            self.output = torch.empty((2, weight.shape[0]), dtype=torch.bfloat16)

        def functional(self, *, session):
            self.output = None
            return self

        def __call__(self, source):
            result = source @ self.weight.T
            return result if self.output is None else self.output.copy_(result)

    owner = mlp.PreparedUngatedMLP.__new__(mlp.PreparedUngatedMLP)
    owner.activation = activation
    owner.intermediate = torch.empty((2, 16), dtype=torch.bfloat16)
    owner.up_projection = Projection(torch.randn((16, 8), dtype=torch.bfloat16))
    owner.down_projection = Projection(torch.randn((8, 16), dtype=torch.bfloat16))
    source = torch.randn((2, 8), dtype=torch.bfloat16)
    expected = owner(source).clone()
    intermediate = weakref.ref(owner.intermediate)
    owner.functional(session=None)
    assert intermediate() is None
    torch.testing.assert_close(owner(source), expected, atol=0, rtol=0)


def test_overlap_binding_finalizers_can_run_repeatedly(monkeypatch):
    class Arena(expert_overlap.ScratchArena):
        def __init__(self):
            pass

        @property
        def auxiliary_stream(self):
            return SimpleNamespace(cuda_stream=1)

        def register_parallel(self, *plans):
            pass

        def views(self, *layouts):
            return tuple(torch.empty(shape, dtype=dtype) for shape, dtype in layouts)

    class Projection:
        def __init__(self):
            self.weight = torch.ones((8, 8), dtype=torch.bfloat16)
            self.bias = None
            self.output = torch.empty((2, 8), dtype=torch.bfloat16)
            self.plan = SimpleNamespace(handle=1,
                memory_requirements=lambda: SimpleNamespace(scratch_nbytes=64))
            self.workspace_nbytes = 64

        def functional(self, *, session):
            self.output = None
            return self

    callbacks = []
    monkeypatch.setattr(expert_overlap, "register_finalizer",
                        lambda session, callback, **kwargs: callbacks.append(callback))
    mlp_owner = SimpleNamespace(up_projection=Projection(), down_projection=Projection(),
        intermediate=torch.empty((2, 8), dtype=torch.bfloat16),
        activation_plan=SimpleNamespace(handle=2), functional=lambda **kwargs: None)
    owner = expert_overlap.ConcurrentSharedExpert(mlp_owner, session=None,
        parallel_plans=(mlp_owner.up_projection.plan,), workspace=Arena()).functional(session=None)
    for _ in range(2):
        for callback in callbacks:
            callback()
        assert owner.workspace is owner.arena is owner.parallel_plans is None


@pytest.mark.parametrize("sm103", [False, True])
def test_standalone_experts_release_refreshed_arena_binding(sm103):
    arena = expert_overlap.ScratchArena("cpu")
    arena.reserve(256)
    owner = (moe_sm103.Sm103RoutedExperts.__new__(moe_sm103.Sm103RoutedExperts)
             if sm103 else _expert_owner())
    owner.output = torch.empty((2, 8), dtype=torch.bfloat16)
    owner.output_source = owner.source = owner.output
    owner.preparation_arena, owner.router, owner.route_inputs = arena, None, {}
    if sm103:
        binding = SimpleNamespace(state=SimpleNamespace(buffer_layouts=(), scratch_bytes=lambda: 64))
    else:
        impl = SimpleNamespace(w1_fp4=torch.empty(0), w2_fp4=torch.empty(0), nvfp4_csf=None)
        owner.experts = SimpleNamespace(_impl=impl, has_csf_blocks=False)
        owner.plan = SimpleNamespace(scratch_specs=lambda: (SimpleNamespace(dtype=torch.uint8, nbytes=64),))
        binding = SimpleNamespace(experts=impl)
    def bind():
        owner.binding, owner.scratch = binding, arena.tensor[:64]
    bind()
    arena.register_finalizer(bind)
    owner.functional(session=None)
    arena.finalize()
    arena.freeze()
    arena.release_preparation()
    assert owner.binding is owner.scratch is owner.preparation_arena is None
    assert arena.nbytes == 0
