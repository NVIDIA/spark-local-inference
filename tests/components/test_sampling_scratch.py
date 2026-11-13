# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conditional scoring accepts invocation-local scratch and candidate prefixes."""

import torch


def test_conditional_score_scratch_can_be_supplied_per_call():
    from ignition.components.sampling import DeviceRequestState, RUNNING
    from ignition.components.speculation import PreparedSpeculativeDistributions

    state = DeviceRequestState.allocate("cpu", vocab_size=64, capacity=4)
    state.status.fill_(RUNNING)
    state.generation.fill_(1)
    state.parameters[:] = torch.tensor([1., .9, 0., 1.2, .1, .2])
    state.histogram[0, 3] = 2
    slots = torch.tensor([0, 1], dtype=torch.int32)
    generations = torch.ones(2, dtype=torch.int64)
    live = torch.tensor([2], dtype=torch.int32)
    candidates = torch.tensor([[3, 7], [2, 5]], dtype=torch.int64)
    logits = torch.linspace(-2, 2, 2 * 3 * 64).reshape(2, 3, 64)
    operation = PreparedSpeculativeDistributions(logits, state, slots, generations, live, candidates)
    operation.prepare_scores()
    expected = {name: getattr(operation, name).clone() for name in
        ("scores", "maximum", "valid", "packed_top_k", "packed_top_p")}
    operation.functional()
    scratch = {name: torch.empty(shape, dtype=dtype) for name, (shape, dtype) in operation.layout.items()}
    operation._prepare_scores(scratch, logits, candidates=candidates)
    for name in ("scores", "maximum", "valid", "packed_top_k", "packed_top_p"):
        torch.testing.assert_close(scratch[name], expected[name], rtol=0, atol=0)
