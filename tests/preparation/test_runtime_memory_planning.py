# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Graph-wide zeroed workspaces remain live through Inductor memory planning."""

from types import SimpleNamespace

import pytest
import sympy
import torch
from torch._inductor.codegen.common import WorkspaceArg, WorkspaceZeroMode
from torch._inductor.codegen.memory_planning import MemoryPlanner

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
    [(owner, name, replacement)] = memory_planning.replacements()
    assert owner is MemoryPlanner and name == "compute_live_ranges"
    planner = MemoryPlanner(wrapper=None, buffer_groups=[group])
    replacement(planner, [])
    assert updates == expected
