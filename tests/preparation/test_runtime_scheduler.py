# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scheduler membership checks preserve code generation without copying sets."""

from inspect import unwrap

import pytest
import torch
from torch._inductor import scheduler as upstream
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from ignition.runtime_patches import scheduler


def test_preparing_scheduler_replacement_leaves_owning_method_unchanged():
    original = upstream.BaseSchedulerNode.decide_inplace_update
    template_original = upstream.Scheduler._replace_node
    [(owner, name, replacement), (template_owner, template_name, template_replacement)] = scheduler.replacements()
    assert owner is upstream.BaseSchedulerNode
    assert name == 'decide_inplace_update'
    assert upstream.BaseSchedulerNode.decide_inplace_update is original
    assert upstream.SchedulerNode.decide_inplace_update is original
    assert replacement is not original
    assert template_owner is upstream.Scheduler
    assert template_name == '_replace_node'
    assert upstream.Scheduler._replace_node is template_original
    assert template_replacement is not template_original


@pytest.mark.parametrize('node_type', [upstream.BaseSchedulerNode, upstream.FusedSchedulerNode,
                                       upstream.NopKernelSchedulerNode])
def test_scheduler_siblings_keep_their_inplace_fast_return(node_type, monkeypatch):
    for owner, name, replacement in scheduler.replacements():
        monkeypatch.setattr(owner, name, replacement)
    node = object.__new__(node_type)
    assert node.decide_inplace_update() is None


@pytest.mark.parametrize('patched', [False, True])
def test_cpu_inductor_checks_membership_without_copying_dependency_sets(monkeypatch, patched):
    original = unwrap(upstream.BaseSchedulerNode.decide_inplace_update)
    [(owner, name, replacement), _] = scheduler.replacements()
    decide = replacement if patched else original
    decisions = []

    class NoUnion(OrderedSet):
        def __or__(self, other):
            raise AssertionError('scheduler copied dependency sets')

    def guarded(self):
        if not isinstance(self, upstream.SchedulerNode):
            return decide(self)
        fields = [(self, 'ancestors'), (V.graph, 'removed_operations'),
                  (self.scheduler, 'completed_operations')]
        originals = [getattr(owner, name) for owner, name in fields]
        try:
            for (owner, name), values in zip(fields, originals):
                setattr(owner, name, NoUnion(values))
            decide(self)
            decisions.append(dict(getattr(V.kernel, 'inplace_update_buffers', {})))
        finally:
            for (owner, name), values in zip(fields, originals):
                setattr(owner, name, values)

    def fn(x):
        a = x + 1
        return a.relu() * 3

    torch._dynamo.reset()
    monkeypatch.setattr(owner, name, guarded)
    try:
        with torch._inductor.config.patch(force_disable_caches=True, max_fusion_size=1):
            compiled = torch.compile(fn, backend='inductor', fullgraph=True)
            x = torch.tensor([-3., 2., 5.])
            if patched:
                torch.testing.assert_close(compiled(x), fn(x))
                assert decisions
            else:
                with pytest.raises(torch._inductor.exc.InductorError, match='scheduler copied dependency sets'):
                    compiled(x)
    finally:
        torch._dynamo.reset()
