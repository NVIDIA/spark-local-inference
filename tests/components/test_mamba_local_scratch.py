# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from ignition.components import mamba


def test_functional_scan_allocates_plan_scratch_and_output_locally(monkeypatch):
    scratch_spec = SimpleNamespace(
        nbytes=13,
        device=torch.device("cpu"),
        shape=(13,),
        dtype=torch.uint8,
    )
    operator = mamba.PreparedMambaScan.__new__(mamba.PreparedMambaScan)
    operator.args = operator.output = None
    operator.output_shape = (2, 4)
    operator.output_dtype = torch.float32
    operator.plan = SimpleNamespace(scratch_specs=lambda: (scratch_spec,))
    captured = {}

    def run(*args, out, plan, workspace):
        captured.update(out=out, plan=plan, workspace=workspace)
        return out

    monkeypatch.setattr(mamba.mamba_ssd, "run", run)
    operands = {
        "x": torch.empty((2, 4)),
        "dt": torch.empty((2, 1)),
        "A": torch.empty((1,)),
        "B": torch.empty((2, 1)),
        "C": torch.empty((2, 1)),
        "D": torch.empty((1,)),
        "dt_bias": torch.empty((1,)),
        "recurrent_state": torch.empty((1, 1, 1, 1)),
        "cu_seqlens": torch.empty((2,), dtype=torch.int32),
        "initial_state_indices": torch.empty((1,), dtype=torch.int64),
        "final_state_indices": torch.empty((1,), dtype=torch.int64),
        "state_is_fresh": torch.empty((1,), dtype=torch.bool),
        "num_seqs": torch.empty((1,), dtype=torch.int32),
        "num_tokens": torch.empty((1,), dtype=torch.int32),
    }

    result = operator(operands)

    assert result is captured["out"]
    assert tuple(result.shape) == operator.output_shape
    assert result.dtype == operator.output_dtype
    assert captured["workspace"].dtype == torch.uint8
    assert captured["workspace"].numel() == scratch_spec.nbytes
