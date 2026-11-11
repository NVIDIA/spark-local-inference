# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Preserve temporary lifetimes and retain compiler pool sizes for capture."""

from . import _replace


def replacements():
    from torch._inductor import memory
    from torch._inductor.codegen.memory_planning import MemoryPlanner
    from torch._inductor.output_code import CompiledFxGraph

    plan = _replace(MemoryPlanner.plan, ((
        "    self.allocate_groups()\n    self.mark_first_last_usage(lines)\n",
        "    self.allocate_groups()\n"
        "    V.graph._flare_pool_sizes = tuple(\n"
        "        (device.type, pool.root.get_symbolic_size())\n"
        "        for device, pools in self.pools.device_to_pools.items()\n"
        "        for pool in pools\n"
        "    )\n"
        "    self.mark_first_last_usage(lines)\n",
    ),), allow_applied=True)
    compiled_graph = _replace(CompiledFxGraph.__init__, ((
        "    self.metrics_deltas = metrics_deltas\n    self.counter_deltas = counter_deltas\n",
        "    self.metrics_deltas = metrics_deltas\n"
        "    self._flare_pool_sizes = getattr(graph, '_flare_pool_sizes', ())\n"
        "    self.counter_deltas = counter_deltas\n",
    ),), allow_applied=True)

    compute_live_ranges = _replace(MemoryPlanner.compute_live_ranges, (
        (
            "    for group in self.buffer_groups:\n",
            "    from torch._inductor.codegen.common import WorkspaceArg, WorkspaceZeroMode\n"
            "    for group in self.buffer_groups:\n",
        ),
        (
            "        if group.is_output:\n            group.update_usage(timestep)\n",
            "        if group.is_output or (\n"
            "            isinstance(group.node, WorkspaceArg)\n"
            "            and group.node.zero_mode == WorkspaceZeroMode.ZERO_PER_GRAPH\n"
            "        ):\n"
            "            group.update_usage(timestep)\n",
        ),
    ), allow_applied=True)
    assign_memory_planning_info = _replace(memory.assign_memory_planning_info_for_scheduler_nodes, (
        (
            "        for buffer in node.get_outputs():\n"
            "            for succ_node in buffer.mpi_buffer.succ_nodes:\n",
            "        for buffer in node.get_outputs():\n"
            "            if not config.reorder_for_peak_memory_debug and (\n"
            "                buffer.mpi_buffer.size_alloc == 0 and buffer.mpi_buffer.size_free == 0\n"
            "            ):\n"
            "                continue\n"
            "            for succ_node in buffer.mpi_buffer.succ_nodes:\n",
        ),
        (
            "    for freeable_buffer in name_to_freeable_input_buf.values():\n"
            "        for succ_node in freeable_buffer.mpi_buffer.succ_nodes:\n",
            "    for freeable_buffer in name_to_freeable_input_buf.values():\n"
            "        if not config.reorder_for_peak_memory_debug and (\n"
            "            freeable_buffer.mpi_buffer.size_alloc == 0\n"
            "            and freeable_buffer.mpi_buffer.size_free == 0\n"
            "        ):\n"
            "            continue\n"
            "        for succ_node in freeable_buffer.mpi_buffer.succ_nodes:\n",
        ),
    ), allow_applied=True)
    return [(MemoryPlanner, "plan", plan),
            (CompiledFxGraph, "__init__", compiled_graph),
            (MemoryPlanner, "compute_live_ranges", compute_live_ranges),
            (memory, "assign_memory_planning_info_for_scheduler_nodes", assign_memory_planning_info)]
