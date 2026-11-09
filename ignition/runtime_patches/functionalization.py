# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torch auto-functionalization fixes used during forward compilation."""

from . import _replace


def _apply_auto_functionalized_patterns(graph_pass, graph):
    import itertools

    import torch
    from torch._inductor.fx_passes import post_grad

    pm = post_grad.pattern_matcher
    targets = (
        torch.ops.higher_order.auto_functionalized,
        torch.ops.higher_order.auto_functionalized_v2,
    )
    for (op, target), entries in graph_pass.patterns.items():
        assert op == "call_function" and target in targets
        assert all(
            isinstance(entry, pm.GraphPatternEntry)
            and isinstance(entry.pattern, pm.CallFunctionVarArgs)
            for entry in entries
        )
    nodes = itertools.chain.from_iterable(
        graph.find_nodes(op="call_function", target=target, sort=False)
        for target in targets
    )
    count = 0
    with pm.GraphTransformObserver(
        graph.owning_module, "pattern_matcher", graph_pass.subsystem
    ):
        for node in sorted(nodes, reverse=True):
            if pm.fallback_node_due_to_unsupported_type(node, allow_cpu_inputs=False):
                continue
            for entry in graph_pass.patterns[(node.op, node.target)]:
                if node._erased:
                    break
                match = entry.pattern.match(node)
                if pm._should_debug_node(node.name):
                    pm.log.warning("%s%s %s %s", node, node.args, match, entry.pattern)
                if not pm.is_match(match):
                    continue
                assert match.nodes == [node]
                if not pm.guard_or_false(entry.extra_check(match)):
                    continue
                # Single-node matches cannot cross mutation or stream boundaries.
                with pm._track_graph_mutation_ops(
                    graph, pm._common_custom_context(match.nodes)
                ):
                    entry.apply(match, graph, node)
                count += 1
                pm.counters[pm.backend]["pattern_matcher_count"] += 1
                pm.counters[pm.backend]["pattern_matcher_nodes"] += len(match.nodes)
    pm.compute_mutation_region_ids(graph)
    return count


def _auto_functionalized_v2_no_clone_trace(
    mutable_op, flat_args, spec, only_clone_these_bases
):
    """Trace one no-result custom-op call whose mutable tensors are whole bases."""
    import torch
    from torch._higher_order_ops.auto_functionalize import (
        _generate_new_op_kwargs_from_bases,
        get_mutable_args_from_schema,
        NotView,
        read_view_information_from_args,
    )
    from torch._inductor.fx_passes import post_grad
    from torch._library.custom_ops import _maybe_get_opdef
    from torch.utils import _pytree as pytree

    if (
        only_clone_these_bases is None
        or len(only_clone_these_bases) != 0
        or not isinstance(mutable_op, torch._ops.OpOverload)
        or _maybe_get_opdef(mutable_op) is None
        or len(mutable_op._schema.returns) != 0
        or mutable_op in post_grad.pattern_matcher.select_decomp_table()
    ):
        return None

    # Constants that need lifting or symbolic bindings remain on make_fx's path.
    literal_types = tuple(
        kind
        for kind in torch.fx.node.base_types
        if kind
        not in (
            torch.Tensor,
            torch._ops.OpOverload,
            torch.SymInt,
            torch.SymBool,
            torch.SymFloat,
        )
    )
    if any(
        not isinstance(value, torch.fx.Node)
        and value is not mutable_op
        and value is not None
        and type(value) not in literal_types
        for value in flat_args
    ):
        return None

    for value in flat_args:
        if not isinstance(value, torch.fx.Node):
            continue
        if "val" in value.meta:
            example = value.meta["val"]
        elif "example_value" in value.meta:
            example = value.meta["example_value"]
        else:
            return None
        if isinstance(example, torch.Tensor):
            if example.layout != torch.strided or any(
                type(size) is not int
                for size in (*example.shape, *example.stride(), example.storage_offset())
            ):
                return None
        elif (
            example is not None
            and type(example) not in literal_types
            and not isinstance(example, torch.SymInt)
        ):
            return None

    _, kwargs = pytree.tree_unflatten(flat_args, spec)
    bases = kwargs.get("_all_bases")
    if not bases or not all(isinstance(base, torch.fx.Node) for base in bases):
        return None
    view_kwargs = dict(kwargs)
    view_kwargs.pop("_all_bases")
    names, types = get_mutable_args_from_schema(mutable_op._schema)
    views = read_view_information_from_args(names, types, view_kwargs, bases)
    if any(
        view is not None and type(view) is not NotView
        for view in pytree.tree_leaves(views)
    ):
        return None

    @torch.no_grad()
    def trace_one_call(fn, examples):
        from torch._dispatch.python import enable_python_dispatcher
        from torch.compiler import config as compiler_config
        from torch.fx.traceback import preserve_node_meta
        from torch._subclasses.fake_tensor import is_fake_tensor
        from torch.fx.experimental.proxy_tensor import (
            _should_save_eager_input_vals,
            extract_val,
        )

        # Fake implementations retain layout validation and shape guards.
        with (
            compiler_config.patch(compile_on_one_rank=False),
            enable_python_dispatcher(),
            preserve_node_meta(),
        ):
            fn(*examples)

        graph = torch.fx.Graph()
        leaves = []
        for index, (original, example) in enumerate(zip(flat_args, examples)):
            placeholder = graph.placeholder(f"arg{index}")
            placeholder.meta["val"] = example
            leaves.append(
                placeholder if isinstance(original, torch.fx.Node) else original
            )
        _, op_kwargs = pytree.tree_unflatten(leaves, spec)
        all_bases = op_kwargs.pop("_all_bases")
        op_kwargs, all_bases = _generate_new_op_kwargs_from_bases(
            mutable_op._schema, op_kwargs, all_bases, (), False
        )
        positional = [
            op_kwargs.pop(argument.name)
            for argument in mutable_op._schema.arguments
            if not argument.kwarg_only
        ]
        positional_schema = [
            argument
            for argument in mutable_op._schema.arguments
            if not argument.kwarg_only
        ]
        scalar_types = (bool, int, float, str, type(None))
        while positional:
            value = positional[-1]
            argument = positional_schema[len(positional) - 1]
            default = argument.default_value
            if (
                not argument.has_default_value()
                or type(value) not in scalar_types
                or type(value) is not type(default)
                or value != default
            ):
                break
            positional.pop()
        for argument in mutable_op._schema.arguments:
            if argument.kwarg_only and argument.name in op_kwargs:
                value = op_kwargs[argument.name]
                default = argument.default_value
                if (
                    argument.has_default_value()
                    and type(value) in scalar_types
                    and type(value) is type(default)
                    and value == default
                ):
                    del op_kwargs[argument.name]

        call = graph.call_function(mutable_op, tuple(positional), op_kwargs)
        call.meta["val"] = None
        if _should_save_eager_input_vals(mutable_op, (call.args, call.kwargs)):

            def eager_value(value):
                if not isinstance(value, torch.fx.Node) or "val" not in value.meta:
                    return None
                example = value.meta["val"]
                if isinstance(example, torch.Tensor) and not is_fake_tensor(example):
                    return None
                return extract_val(example)

            call.meta["eager_input_vals"] = torch.fx.node.map_aggregate(
                (call.args, call.kwargs), eager_value
            )
        graph.output((None, *all_bases))
        return torch.fx.GraphModule({}, graph)

    return trace_one_call


def replacements():
    from torch._inductor.fx_passes import post_grad

    replacement = _replace(
        post_grad.decompose_auto_functionalized,
        (
            (
                "    apply_pass_to_control_deps_subgraphs(decompose_auto_functionalized, graph)",
                "    from ignition.runtime_patches.functionalization import (\n"
                "        _apply_auto_functionalized_patterns,\n"
                "        _auto_functionalized_v2_no_clone_trace,\n"
                "    )\n\n"
                "    apply_pass_to_control_deps_subgraphs(decompose_auto_functionalized, graph)",
            ),
            (
                "                mutable_op, only_clone_these_bases, **kwargs\n"
                "            )\n\n"
                "        # pyrefly: ignore [bad-argument-type]\n"
                "        match.replace_by_example(decomp, flat_args, run_functional_passes=False)",
                "                mutable_op, only_clone_these_bases, **kwargs\n"
                "            )\n\n"
                "        trace_fn = _auto_functionalized_v2_no_clone_trace(\n"
                "            args[0], flat_args, spec, only_clone_these_bases\n"
                "        )\n"
                "        # pyrefly: ignore [bad-argument-type]\n"
                "        match.replace_by_example(\n"
                "            decomp, flat_args, trace_fn=trace_fn, run_functional_passes=False\n"
                "        )",
            ),
            (
                "    graph_pass.apply(graph)",
                "    _apply_auto_functionalized_patterns(graph_pass, graph)",
            ),
        ),
    )
    return [(post_grad, "decompose_auto_functionalized", replacement)]
