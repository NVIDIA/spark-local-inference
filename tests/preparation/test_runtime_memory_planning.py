# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Workspace lifetimes and memory-cost metadata preserve scheduler behavior."""

from types import SimpleNamespace

import pytest
import sympy
import torch
from torch._inductor import config, memory
from torch._inductor.codegen.common import WorkspaceArg, WorkspaceZeroMode
from torch._inductor.codegen.memory_planning import MemoryPlanner
from torch._inductor.dependencies import StarDep
from torch.utils._ordered_set import OrderedSet

from ignition.runtime_patches import memory_planning


@pytest.mark.parametrize("mode,expected", [
    (WorkspaceZeroMode.UNINITIALIZED, []),
    (WorkspaceZeroMode.ZERO_ON_CALL, []),
    (WorkspaceZeroMode.ZERO_PER_GRAPH, [1]),
])
def test_workspace_live_range_tracks_graph_wide_zeroing(mode, expected):
    node = WorkspaceArg(count=sympy.Integer(256), zero_mode=mode,
                        device=torch.device("cpu"), outer_name="workspace")
    updates = []
    group = SimpleNamespace(node=node, is_output=False, update_usage=updates.append)
    replacement = next(function for owner, name, function in memory_planning.replacements()
                       if owner is MemoryPlanner and name == "compute_live_ranges")
    planner = MemoryPlanner(wrapper=None, buffer_groups=[group])
    replacement(planner, [])
    assert updates == expected


class _Node:
    def __init__(self, index):
        self.name = f"node{index}"
        self.outputs = []
        self.unmet_dependencies = OrderedSet()

    def get_name(self):
        return self.name

    def get_outputs(self):
        return self.outputs

    def get_buffer_names(self):
        return OrderedSet(buffer.name for buffer in self.outputs)


class _Buffer:
    def __init__(self, name, allocated, freed, successors, ordering_successors=None):
        self.name = name
        self.mpi_buffer = memory.MemoryPlanningInfoForBuffer(
            size_alloc=allocated, size_free=freed, succ_nodes=OrderedSet(successors),
            succ_nodes_for_ordering=OrderedSet(
                successors if ordering_successors is None else ordering_successors))

    def get_name(self):
        return self.name


@pytest.mark.parametrize("count", [3, 6, 12])
@pytest.mark.parametrize("retained_outputs", [False, True])
def test_zero_cost_aliases_preserve_ordering_and_memory_estimates(count, retained_outputs):
    replacement = next(function for owner, name, function in memory_planning.replacements()
                       if owner is memory and name == "assign_memory_planning_info_for_scheduler_nodes")
    nodes = [_Node(index) for index in range(count)]
    for index, node in enumerate(nodes):
        node.outputs.extend(_Buffer(f"zero{index}_{alias}", 0, 0, nodes[index + 1:])
                            for alias in range(10))
    nodes[0].outputs.append(_Buffer("real", 64, 64, nodes[1:]))
    nodes[1].outputs.extend((_Buffer("multi_parent", 17, 0, nodes[2:]),
                            _Buffer("multi_child", 0, 17, nodes[2:])))
    nodes[0].outputs.append(_Buffer("order_only", 0, 0, [], nodes[1:]))
    buffers = {buffer.name: buffer for node in nodes for buffer in node.outputs}
    names = {node.name: node for node in nodes}
    inputs = {name: memory.FreeableInputBuffer(name, memory.MemoryPlanningInfoForBuffer(
        size_free=size, succ_nodes=OrderedSet(successors),
        succ_nodes_for_ordering=OrderedSet(successors)))
        for name, size, successors in (("input_real", 24, nodes[:2]), ("input_zero", 0, nodes))}
    for node in nodes:
        for buffer in node.outputs:
            for successor in buffer.mpi_buffer.succ_nodes_for_ordering:
                successor.unmet_dependencies.add(StarDep(buffer.name))
    outputs = OrderedSet(["real", "zero0_0"] if retained_outputs else [])

    def measurements():
        orders = {}
        for method in (memory.topological_sort_lpmf, memory.topological_sort_bfs,
                       memory.topological_sort_dfs):
            order = (method(nodes, inputs, buffers, outputs)
                     if method is memory.topological_sort_lpmf else method(nodes))
            orders[method.__name__] = tuple(node.name for node in order)
        region_peak = memory.estimate_region_peak_memory(nodes, region_start=0, region_end=count - 1,
            step_of=nodes.index, graph_outputs=outputs, cur_memory=24)
        edges = tuple((node.name, tuple(pred.name for pred in node.mpi_node.pred_nodes),
                       tuple(succ.name for succ in node.mpi_node.succ_nodes)) for node in nodes)
        return orders, region_peak, memory.estimate_peak_memory(nodes, inputs, outputs), edges

    def predecessors():
        return tuple(tuple(buffer.name for buffer in node.mpi_node.pred_buffers) for node in nodes)

    # Debug export retains every predecessor even when another test has installed the patch.
    with config.patch(reorder_for_peak_memory_debug=True):
        memory.assign_memory_planning_info_for_scheduler_nodes(nodes, names, buffers, inputs)
    baseline, baseline_predecessors = measurements(), predecessors()
    with config.patch(reorder_for_peak_memory_debug=False):
        replacement(nodes, names, buffers, inputs)
    assert measurements() == baseline
    assert all(buffer.mpi_buffer.size_alloc != 0 or buffer.mpi_buffer.size_free != 0
               for node in nodes for buffer in node.mpi_node.pred_buffers)
    for node in nodes[2:]:
        assert buffers["multi_parent"] in node.mpi_node.pred_buffers
        assert buffers["multi_child"] in node.mpi_node.pred_buffers
    assert inputs["input_real"] in nodes[0].mpi_node.pred_buffers
    assert sum(map(len, predecessors())) < sum(map(len, baseline_predecessors))
    with config.patch(reorder_for_peak_memory_debug=True):
        replacement(nodes, names, buffers, inputs)
    assert predecessors() == baseline_predecessors
    assert measurements() == baseline
