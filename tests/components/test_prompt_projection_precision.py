# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prompt-scoring vocabulary tiles honor the caller's output precision."""

from types import SimpleNamespace

import pytest
import torch

from ignition.components import prompt_logprobs


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_prompt_projection_and_reduction_share_requested_dtype(monkeypatch, dtype):
    observed = []

    class Projection:
        def __init__(self, source, weight, output, **kwargs):
            assert weight.dtype == torch.bfloat16
            observed.append(("projection", output.dtype))
            self.shape, self.dtype = tuple(output.shape), output.dtype

        def functional(self, *, session):
            return self

        def __call__(self, source):
            assert source.dtype == torch.bfloat16
            return torch.zeros(self.shape, dtype=self.dtype)

    class Reduction:
        def __init__(self, logits, *args, **kwargs):
            observed.append(("reduction", logits.dtype))
            self.dtype = logits.dtype

        def functional(self, *, session):
            return self

        def __call__(self, logits):
            assert logits.dtype == self.dtype

    monkeypatch.setattr(prompt_logprobs, "UnquantizedProjection", Projection)
    monkeypatch.setattr(prompt_logprobs, "PreparedPromptReduction", Reduction)
    owner = SimpleNamespace(
        outputs=SimpleNamespace(
            targets=torch.zeros(9, dtype=torch.int32),
            values=torch.zeros(9, 4),
        ),
        frame=torch.zeros(1, dtype=torch.int32),
        rank=0,
        runtime=None,
    )
    source = torch.zeros(9, 128, dtype=torch.bfloat16)
    weight = torch.zeros(64, 128, dtype=torch.bfloat16)
    operation = prompt_logprobs.PromptScoring.prepare(
        owner,
        source,
        weight,
        vocabulary=64,
        session=None,
        coordinator=None,
        name="prompt-head",
        logits_dtype=dtype,
    )
    operation(source)
    assert observed == [("projection", dtype), ("reduction", dtype)]
