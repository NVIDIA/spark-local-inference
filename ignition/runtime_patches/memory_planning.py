# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Keep graph-wide zeroed workspaces live through the final graph operation."""

from . import _replace


def replacements():
    from torch._inductor.codegen.memory_planning import MemoryPlanner

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
    return [(MemoryPlanner, "compute_live_ranges", compute_live_ranges)]
