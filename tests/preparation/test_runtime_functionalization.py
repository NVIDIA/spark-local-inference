# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Auto-functionalization preserves mutations, aliases, and graph metadata."""

import operator
from unittest.mock import patch

import pytest
import torch
from torch._inductor import config, pattern_matcher as pm
from torch._inductor.fx_passes import post_grad as pg
from torch.fx import Graph, GraphModule
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.utils import _pytree as pytree

from ignition.runtime_patches import functionalization


fake_calls = []
fake_grad_modes = []


@torch.library.custom_op("flare_runtime_functionalization::no_result", mutates_args={"x"})
def no_result(
    x: torch.Tensor, y: torch.Tensor, scale: float = 1.0, *, bias: float = 0.0
) -> None:
    x.add_(y, alpha=scale)
    x.add_(bias)


@no_result.register_fake
def no_result_fake(x, y, scale=1.0, *, bias=0.0):
    fake_calls.append((tuple(x.shape), tuple(y.shape), scale, bias))
    fake_grad_modes.append(torch.is_grad_enabled())
    if x.shape != y.shape:
        raise ValueError("mutable and source tensors must have equal shapes")


@torch.library.custom_op(
    "flare_runtime_functionalization::overlapping_views", mutates_args={"x", "z"}
)
def overlapping_views(x: torch.Tensor, z: torch.Tensor) -> None:
    x.add_(1)
    z.mul_(2)


@overlapping_views.register_fake
def overlapping_views_fake(x, z):
    pass


lib = torch.library.Library("flare_runtime_functionalization", "FRAGMENT")
lib.define("with_result(Tensor(a!) x, Tensor y) -> Tensor")


@torch.library.impl(lib, "with_result", "CPU")
def with_result(x, y):
    x.add_(y)
    return x * 2


@torch.library.register_fake("flare_runtime_functionalization::with_result")
def with_result_fake(x, y):
    return torch.empty_like(x)


NO_RESULT = torch.ops.flare_runtime_functionalization.no_result.default
WITH_RESULT = torch.ops.flare_runtime_functionalization.with_result.default
HOPS = (
    torch.ops.higher_order.auto_functionalized,
    torch.ops.higher_order.auto_functionalized_v2,
)
CONTEXT = {"stream": 3, "mempool": 7, "mempool_device": 0}


@pytest.fixture
def transformed(monkeypatch):
    original = pg.decompose_auto_functionalized
    built = functionalization.replacements()
    assert pg.decompose_auto_functionalized is original
    assert len(built) == 1
    owner, name, replacement = built[0]
    assert replacement.__globals__ is original.__globals__
    monkeypatch.setattr(owner, name, replacement)
    with config.patch(disable_cpp_codegen=False):
        yield replacement


def make_graph(count=3, *, v2=True, clone=False, op=NO_RESULT, context=False,
               scale=1.0, bias=0.0):
    graph = Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    state = x
    outputs = []
    for _ in range(count):
        kwargs = (
            {"y": y, "_all_bases": [state], "_x_base_index": 0}
            if v2 else {"x": state, "y": y}
        )
        if op is NO_RESULT:
            kwargs.update(scale=scale, bias=bias)
        wrapper = graph.call_function(HOPS[v2], (op,), kwargs)
        wrapper.meta["only_clone_these_tensors"] = (
            ([0] if v2 else ["x"]) if clone else []
        )
        if context:
            wrapper.meta["custom"] = dict(CONTEXT)
        outputs.append(graph.call_function(operator.getitem, (wrapper, 0)))
        state = graph.call_function(operator.getitem, (wrapper, 1))
    graph.output((state, outputs))
    module = GraphModule({}, graph)
    FakeTensorProp(module).propagate(torch.ones(4), torch.full((4,), 2.0))
    return module


def assert_execution(module, count=3, *, clone=False, op=NO_RESULT,
                     scale=1.0, bias=0.0):
    module.recompile()
    value = torch.ones(4)
    state, outputs = module(value, torch.full((4,), 2.0))
    increment = 2.0 * scale + bias
    torch.testing.assert_close(state, torch.full((4,), 1.0 + increment * count))
    assert (state.data_ptr() == value.data_ptr()) == (not clone)
    torch.testing.assert_close(
        value, torch.full((4,), 1.0 if clone else 1.0 + increment * count)
    )
    if op is NO_RESULT:
        assert outputs == [None] * count
    else:
        for index, output in enumerate(outputs):
            torch.testing.assert_close(
                output, torch.full((4,), 2.0 * (1.0 + 2.0 * (index + 1)))
            )
    assert not any(node.target in HOPS for node in module.graph.nodes)


@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize("clone", [False, True])
@pytest.mark.parametrize("op", [WITH_RESULT, NO_RESULT])
def test_outputs_aliasing_and_stream_metadata(transformed, v2, clone, op):
    module = make_graph(v2=v2, clone=clone, op=op, context=True)
    transformed(module.graph)
    assert_execution(module, clone=clone, op=op)
    calls = [node for node in module.graph.nodes if node.target is op]
    assert len(calls) == 3
    assert all(node.meta["custom"] == CONTEXT for node in calls)
    assert all("val" in node.meta for node in calls)
    if op is NO_RESULT and v2 and not clone:
        assert all(node.meta["val"] is None for node in calls)


@pytest.mark.parametrize("count", [1, 12])
def test_single_region_scan_and_fresh_ids(transformed, count):
    module = make_graph(count)
    with patch.object(pm, "compute_mutation_region_ids", wraps=pm.compute_mutation_region_ids) as scans:
        transformed(module.graph)
    assert scans.call_count == 1
    ids = [node.meta["mutation_region_id"] for node in module.graph.nodes]
    pm.compute_mutation_region_ids(module.graph)
    assert ids == [node.meta["mutation_region_id"] for node in module.graph.nodes]
    assert_execution(module, count)


def test_no_clone_avoids_fx_tracing_and_runs_fake(transformed):
    module = make_graph(scale=2.0, bias=3.0)
    before = len(fake_calls)
    with patch.object(pm, "fwd_only", wraps=pm.fwd_only) as traces:
        transformed(module.graph)
    assert traces.call_count == 0
    assert len(fake_calls) - before == 3
    assert_execution(module, scale=2.0, bias=3.0)
    assert not any(node.target is torch.ops.aten.clone.default for node in module.graph.nodes)


@pytest.mark.parametrize("clone", [False, True])
def test_fake_validation_uses_inference_grad_mode(transformed, clone):
    module = make_graph(1, clone=clone)
    before = len(fake_grad_modes)
    with torch.enable_grad():
        transformed(module.graph)
        assert torch.is_grad_enabled()
    assert fake_grad_modes[before:] == [False]
    assert_execution(module, 1, clone=clone)


@pytest.mark.parametrize("scale,bias", [(1.0, 0.0), (2.0, 0.0), (1.0, 3.0)])
def test_default_scalar_normalization(transformed, scale, bias):
    module = make_graph(1, scale=scale, bias=bias)
    transformed(module.graph)
    call, = [node for node in module.graph.nodes if node.target is NO_RESULT]
    assert len(call.args) == (2 if scale == 1.0 else 3)
    assert call.kwargs == ({} if bias == 0.0 else {"bias": bias})
    assert_execution(module, 1, scale=scale, bias=bias)


def test_clone_and_returning_ops_keep_fx_tracing(transformed):
    for clone, op in [(True, NO_RESULT), (False, WITH_RESULT)]:
        module = make_graph(1, clone=clone, op=op)
        with patch.object(pm, "fwd_only", wraps=pm.fwd_only) as traces:
            transformed(module.graph)
        assert traces.call_count == 1
        assert_execution(module, 1, clone=clone, op=op)


def test_fake_validation_is_not_bypassed(transformed):
    module = make_graph(1)
    y = next(node for node in module.graph.nodes if node.target == "y")
    mode = y.meta["val"].fake_mode
    with mode:
        y.meta["val"] = torch.ones(3)
    with pytest.raises(ValueError, match="must have equal shapes"):
        transformed(module.graph)


@pytest.mark.parametrize("clone", [False, True])
def test_mutable_views_keep_fx_and_shared_base(transformed, clone):
    graph = Graph()
    x = graph.placeholder("x")
    wrapper = graph.call_function(HOPS[1], (overlapping_views._opoverload,), {
        "_all_bases": [x], "_x_base_index": 0, "_x_slice_dim": 0,
        "_x_slice_start": 0, "_x_slice_end": 3, "_z_base_index": 0,
        "_z_slice_dim": 0, "_z_slice_start": 1, "_z_slice_end": 4,
    })
    wrapper.meta["only_clone_these_tensors"] = [0] if clone else []
    base = graph.call_function(operator.getitem, (wrapper, 1))
    graph.output(base)
    module = GraphModule({}, graph)
    FakeTensorProp(module).propagate(torch.arange(1.0, 5.0))
    with patch.object(pm, "fwd_only", wraps=pm.fwd_only) as traces:
        transformed(graph)
    assert traces.call_count == 1
    module.recompile()
    value = torch.arange(1.0, 5.0)
    base = module(value)
    torch.testing.assert_close(base, torch.tensor([2.0, 6.0, 8.0, 8.0]))
    assert (base.data_ptr() == value.data_ptr()) == (not clone)
    torch.testing.assert_close(value, torch.arange(1.0, 5.0) if clone else base)


def test_generic_matcher_rejects_mutation_and_stream_boundaries(transformed):
    graph = Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    sine = graph.call_function(torch.ops.aten.sin.default, (x,))
    wrapper = graph.call_function(HOPS[1], (NO_RESULT,), {
        "y": y, "scale": 1.0, "bias": 0.0,
        "_all_bases": [x], "_x_base_index": 0,
    })
    wrapper.meta["only_clone_these_tensors"] = []
    state = graph.call_function(operator.getitem, (wrapper, 1))
    cosine = graph.call_function(torch.ops.aten.cos.default, (sine,))
    graph.output((cosine, state))
    module = GraphModule({}, graph)
    FakeTensorProp(module).propagate(torch.ones(4), torch.ones(4))
    transformed(graph)
    matcher = pm.PatternMatcherPass()
    accepted = []

    @pm.register_graph_pattern(
        pm.CallFunction(torch.ops.aten.cos.default,
            pm.CallFunction(torch.ops.aten.sin.default, pm.KeywordArg("x"))),
        pass_dict=matcher,
    )
    def handler(match, x):
        accepted.append(match)

    assert matcher.apply(module) == 0
    assert not accepted
    sine.append(cosine)
    pm.compute_mutation_region_ids(graph)
    cosine.meta["custom"] = {"stream": 3}
    assert matcher.apply(module) == 0
    assert not accepted
    cosine.meta.pop("custom")
    assert matcher.apply(module) == 1
    assert len(accepted) == 1


def test_control_deps_recursion_and_unused_attributes(transformed):
    root = torch.nn.Module()
    root.add_module("body", make_graph(2))
    root.add_module("unused", make_graph(1))
    graph = Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    body = graph.get_attr("body")
    graph.get_attr("unused")
    result = graph.call_function(pg.control_deps, ([], body, x, y))
    graph.output(result)
    module = GraphModule(root, graph)
    with patch.object(pm, "compute_mutation_region_ids", wraps=pm.compute_mutation_region_ids) as scans:
        transformed(graph)
    assert scans.call_count == 2
    assert not hasattr(module, "unused")
    assert hasattr(module, "body")
    module.body.recompile()
    assert_execution(module, 2)


def test_callback_order_and_match_counters(transformed):
    module = make_graph(4)
    expected = [node.name for node in module.graph.nodes if node.target is HOPS[1]][::-1]
    observed = []
    real_apply = pm.GraphPatternEntry.apply

    def observed_apply(self, match, graph, node):
        observed.append(node.name)
        return real_apply(self, match, graph, node)

    before_count = pm.counters[pm.backend]["pattern_matcher_count"]
    before_nodes = pm.counters[pm.backend]["pattern_matcher_nodes"]
    with patch.object(pm.GraphPatternEntry, "apply", observed_apply):
        transformed(module.graph)
    assert observed == expected
    assert pm.counters[pm.backend]["pattern_matcher_count"] - before_count == 4
    assert pm.counters[pm.backend]["pattern_matcher_nodes"] - before_nodes == 4
    assert_execution(module, 4)


def test_selected_decomposition_keeps_fx_trace(transformed, monkeypatch):
    module = make_graph(1)
    table = pm.select_decomp_table()
    def decomposition(x, y, scale=1.0, *, bias=0.0):
        x.add_(y, alpha=scale)
        x.add_(bias)

    monkeypatch.setattr(
        pm, "select_decomp_table", lambda: {**table, NO_RESULT: decomposition}
    )
    with patch.object(pm, "fwd_only", wraps=pm.fwd_only) as traces:
        transformed(module.graph)
    assert traces.call_count == 1
    assert_execution(module, 1)


def test_trace_selection_rejects_constants_and_missing_metadata():
    module = make_graph(1)
    wrapper = next(node for node in module.graph.nodes if node.target is HOPS[1])
    flat_args, spec = pytree.tree_flatten((wrapper.args, wrapper.kwargs))
    select = functionalization._auto_functionalized_v2_no_clone_trace
    assert select(NO_RESULT, flat_args, spec, []) is not None
    assert select(NO_RESULT, flat_args, spec, None) is None
    y = next(node for node in module.graph.nodes if node.target == "y")
    y.meta.clear()
    assert select(NO_RESULT, flat_args, spec, []) is None
    y.meta["val"] = torch.ones(4)
    constant_args = [torch.ones(4) if value is y else value for value in flat_args]
    assert select(NO_RESULT, constant_args, spec, []) is None


@torch.library.custom_op(
    "flare_runtime_functionalization::exact_strides",
    mutates_args={"x"}, tags=(torch.Tag.needs_exact_strides,),
)
def exact_strides(x: torch.Tensor, y: torch.Tensor) -> None:
    x.add_(y)


exact_fake_strides = []


@exact_strides.register_fake
def exact_strides_fake(x, y):
    exact_fake_strides.append((x.stride(), y.stride()))


def test_eager_input_metadata_retains_exact_strides(transformed):
    graph = Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    wrapper = graph.call_function(HOPS[1], (exact_strides._opoverload,), {
        "y": y, "_all_bases": [x], "_x_base_index": 0,
    })
    wrapper.meta["only_clone_these_tensors"] = []
    state = graph.call_function(operator.getitem, (wrapper, 1))
    graph.output(state)
    module = GraphModule({}, graph)
    FakeTensorProp(module).propagate(torch.ones(4), torch.ones(4))
    mode = x.meta["val"].fake_mode
    with mode:
        eager_x = torch.empty_strided((4,), (2,))
        eager_y = torch.empty_strided((4,), (3,))
    wrapper.meta["eager_input_vals"] = ((exact_strides._opoverload,), {
        "y": eager_y, "_all_bases": torch.fx.immutable_collections.immutable_list([eager_x]), "_x_base_index": 0,
    })
    before = len(exact_fake_strides)
    with patch.object(pm, "fwd_only", wraps=pm.fwd_only) as traces:
        transformed(graph)
    assert traces.call_count == 0
    assert exact_fake_strides[before:] == [((2,), (3,)), ((1,), (1,))]
    call, = [node for node in graph.nodes if node.target is exact_strides._opoverload]
    eager_args, eager_kwargs = call.meta["eager_input_vals"]
    assert eager_kwargs == {}
    assert [value.stride() for value in eager_args] == [(2,), (3,)]
    assert [value.meta["val"].stride() for value in call.args] == [(1,), (1,)]
    module.recompile()
    value = torch.ones(4)
    result = module(value, torch.full((4,), 2.0))
    torch.testing.assert_close(result, torch.full((4,), 3.0))
    assert result.data_ptr() == value.data_ptr()


@torch.library.custom_op(
    "flare_runtime_functionalization::tensor_list", mutates_args={"xs", "z"}
)
def tensor_list(xs: list[torch.Tensor], z: torch.Tensor | None) -> None:
    for value in xs:
        value.add_(1)
    if z is not None:
        z.mul_(2)


@tensor_list.register_fake
def tensor_list_fake(xs, z):
    pass


@pytest.mark.parametrize("optional", [False, True])
def test_tensor_lists_and_optional_mutable_bases(transformed, optional):
    graph = Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    z = graph.placeholder("z")
    bases = [x, y, z] if optional else [x, y]
    wrapper = graph.call_function(HOPS[1], (tensor_list._opoverload,), {
        "_all_bases": bases, "_xs_length": 2,
        "_xs_0_base_index": 0, "_xs_1_base_index": 1,
        "_z_base_index": 2 if optional else None,
    })
    wrapper.meta["only_clone_these_tensors"] = []
    states = [graph.call_function(operator.getitem, (wrapper, i + 1))
              for i in range(len(bases))]
    graph.output(states)
    module = GraphModule({}, graph)
    FakeTensorProp(module).propagate(torch.ones(4), torch.ones(4), torch.ones(4))
    with patch.object(pm, "fwd_only", wraps=pm.fwd_only) as traces:
        transformed(graph)
    assert traces.call_count == 0
    module.recompile()
    values = [torch.ones(4) for _ in range(3)]
    result = module(*values)
    for index, value in enumerate(result):
        torch.testing.assert_close(value, torch.full((4,), 2.0))
        assert value.data_ptr() == values[index].data_ptr()
    if not optional:
        torch.testing.assert_close(values[-1], torch.ones(4))


def test_symbolic_geometry_keeps_fx_trace(transformed):
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    module = make_graph(1)
    mode = FakeTensorMode(shape_env=ShapeEnv())
    x = next(node for node in module.graph.nodes if node.target == "x")
    y = next(node for node in module.graph.nodes if node.target == "y")
    x.meta["val"] = mode.from_tensor(torch.ones(4), static_shapes=False)
    y.meta["val"] = mode.from_tensor(torch.ones(4), static_shapes=False)
    assert isinstance(x.meta["val"].shape[0], torch.SymInt)
    with patch.object(pm, "fwd_only", wraps=pm.fwd_only) as traces:
        transformed(module.graph)
    assert traces.call_count == 1
    assert_execution(module, 1)
