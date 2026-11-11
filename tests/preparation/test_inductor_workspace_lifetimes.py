# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Combined memory pools preserve graph-wide workspace lifetimes."""

from types import SimpleNamespace

import sympy
import torch
from torch._inductor import config
from torch._inductor.codegen.common import WorkspaceArg, WorkspaceZeroMode
from torch._inductor.codegen.memory_planning import (
    AllocFromPoolLine, BufferGroup, DeallocFromPoolLine, MemoryPlanner,
)
from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.virtualized import V

from ignition import runtime_patches


def workspace(name, count, zero_mode, dtype=torch.uint8):
    return BufferGroup(WorkspaceArg(count=sympy.Integer(count), zero_mode=zero_mode,
        device=torch.device('cuda:0'), outer_name=name, dtype=dtype))


def test_graph_wide_semaphore_cannot_alias_kernel_workspace():
    runtime_patches.apply()
    semaphore = workspace('semaphores', 1, WorkspaceZeroMode.ZERO_PER_GRAPH, torch.uint32)
    first = workspace('first', 64, WorkspaceZeroMode.UNINITIALIZED)
    second = workspace('second', 64, WorkspaceZeroMode.UNINITIALIZED)
    wrapper = SimpleNamespace()
    lines = [AllocFromPoolLine(wrapper, semaphore), 'semaphores.zero_()',
        AllocFromPoolLine(wrapper, first), 'first_kernel(semaphores, first)',
        DeallocFromPoolLine(wrapper, first), 'between_kernels',
        AllocFromPoolLine(wrapper, second), 'second_kernel(semaphores, second)',
        DeallocFromPoolLine(wrapper, second)]
    with V.set_graph_handler(SimpleNamespace(sizevars=SizeVarAllocator())), config.patch(memory_pool='combined'):
        planner = MemoryPlanner(wrapper, buffer_groups=[semaphore, first, second])
        planner.compute_live_ranges(lines)
        planner.allocate_groups()
    for scratch in (first, second):
        assert semaphore.live_range.contains(scratch.live_range)
        a, b = semaphore.allocation, scratch.allocation
        assert a.pool is not b.pool or a.offset + a.size_hint <= b.offset or b.offset + b.size_hint <= a.offset
    assert first.allocation.pool is second.allocation.pool
    assert first.allocation.offset == second.allocation.offset
