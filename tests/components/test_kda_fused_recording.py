# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Fused KDA recording exposes committed-state reads to forward compilation."""

from types import SimpleNamespace

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._pytree import tree_leaves

from ignition.recipes.kimi3.kda import PreparedKDA


def _owner():
    owner = object.__new__(PreparedKDA)
    owner.recording, owner.heads, owner.width, owner.handle = True, 1, 128, 0
    owner.eps = 1e-6
    owner.fused = object()
    owner.recurrent_state = torch.zeros((3, 1, 128, 128))
    owner.state = SimpleNamespace(convolution=torch.zeros((3, 384, 6), dtype=torch.bfloat16))
    owner.weights = {
        "A_log": torch.zeros(1), "dt_bias": torch.zeros(128),
        "norm": torch.ones(128, dtype=torch.bfloat16),
        "convolution": torch.ones((384, 4), dtype=torch.bfloat16),
    }
    owner.projections = {"input": lambda value: value, "forget": lambda value: value,
                         "output": lambda value: value}
    return owner


def test_fused_forward_trace_retains_committed_recurrent_state_operand():
    owner = _owner()
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        convolution_slots=torch.ones(1, dtype=torch.int64),
        fresh=torch.zeros(1, dtype=torch.bool),
        accepted_tokens=torch.ones(1, dtype=torch.int32),
        state_indices=torch.ones((1, 4), dtype=torch.int32),
        num_seqs=torch.ones(1, dtype=torch.int32),
        num_tokens=torch.tensor([2], dtype=torch.int32),
    )
    records = (torch.empty((3, 1, 4, 128)),
               torch.empty((3, 1, 4, 256), dtype=torch.bfloat16))

    def forward(projected, recurrent_state):
        owner.recurrent_state = recurrent_state
        return owner(projected, metadata, records=records)

    graph = make_fx(forward, tracing_mode="fake", _allow_non_fake_inputs=True)(
        torch.zeros((2, 648), dtype=torch.bfloat16), owner.recurrent_state).graph
    _, recurrent = [node for node in graph.nodes if node.op == "placeholder"]
    recording, = [node for node in graph.nodes
                  if node.target == torch.ops.flare.kda_record_fused.default]
    assert recurrent in tree_leaves(recording.args)


def test_fused_launch_binds_caller_recurrent_state():
    owner = _owner()
    observed = []
    owner.fused = SimpleNamespace(run_dynamic=lambda binding, conv, **options:
                                 observed.append(binding.recurrent_state))
    tensors = [torch.empty(1) for _ in range(15)]
    state = torch.ones_like(owner.recurrent_state)
    owner.run_fused(tensors, state, torch.empty(1), torch.empty(1), torch.empty(1),
                    owner.state.convolution)
    assert observed == [state]
