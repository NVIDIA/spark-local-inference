# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime row operands for prepared token-state kernels."""

from types import SimpleNamespace

import pytest
import torch

from ignition.components.token_state import PreparedTokenState


@pytest.mark.parametrize("dynamic", [False, True])
def test_functional_token_state_routes_row_operands(monkeypatch, dynamic):
    from ignition.preparation import triton as prepared_triton

    operation = PreparedTokenState.__new__(PreparedTokenState)
    rows = torch.tensor([3], dtype=torch.int32)
    generations = torch.tensor([7], dtype=torch.int64)
    live = torch.tensor([1], dtype=torch.int32)
    tokens = torch.tensor([11], dtype=torch.int64)
    boundaries = torch.tensor([0, 1], dtype=torch.int32)
    operation.state = SimpleNamespace(generation=torch.zeros(4, dtype=torch.int64))
    operation._rows, operation._token_capacity, operation._prepared = 1, 1, True
    state_values = tuple(torch.zeros(4, dtype=torch.int64) for _ in range(4))
    operation.pending_tokens = torch.empty(1, dtype=torch.int64)
    operation.positions = torch.empty(1, dtype=torch.int64)
    operation.history = None
    operation.active = torch.empty(1, dtype=torch.int32)
    operation._gather_args = (rows, generations, live, *state_values, None,
        operation.pending_tokens, operation.positions, None, operation.active, 1, 1, 0, False, 0, 4)
    operation._consume_args = (tokens, boundaries, rows, generations, live, *state_values,
                               None, 1, 1, 1, 0, False, 4)
    operation._gather = SimpleNamespace(kernel="gather", compiled=None, args=operation._gather_args,
                                        grid=(1, 1, 1), options={})
    operation._consume = SimpleNamespace(kernel="consume", compiled=None, args=operation._consume_args,
                                         grid=(1, 1, 1), options={})
    launches = []
    monkeypatch.setattr(prepared_triton, "launch_triton", lambda kernel, compiled, args, grid, **options:
                        launches.append((kernel, args)))
    operation.functional(dynamic_inputs=dynamic)
    replacement_rows = torch.tensor([1], dtype=torch.int32)
    replacement_generations = torch.tensor([9], dtype=torch.int64)
    replacement_live = torch.tensor([0], dtype=torch.int32)
    if dynamic:
        assert operation._gather_inputs[:3] == (None, None, None)
        assert operation._consume_inputs[:3] == (None, None, None)
        with pytest.raises(ValueError, match="runtime row operands"):
            operation.gather()
        with pytest.raises(ValueError, match="row_slot_ids"):
            operation.gather(replacement_rows.to(torch.int64), replacement_generations, replacement_live)
        operands = dict(row_slot_ids=replacement_rows, row_generations=replacement_generations,
                        live_rows=replacement_live)
    else:
        operands = {}
    operation.gather(**operands)
    operation.consume(tokens, boundaries, **operands)
    expected = (replacement_rows, replacement_generations, replacement_live) if dynamic else (rows, generations, live)
    assert all(actual is wanted for actual, wanted in zip(launches[0][1][:3], expected, strict=True))
    assert all(actual is wanted for actual, wanted in zip(launches[1][1][2:5], expected, strict=True))
