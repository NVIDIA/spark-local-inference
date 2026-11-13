# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared GPU prefix allocation must match the scalar frontier oracle exactly."""

import os
import random

import pytest
import torch

from ignition.components.causal_prefix import PreparedCausalPrefixAllocation
from tests.components.causal_prefix_oracle import allocate

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("batch,depth", [(1, 1), (1, 7), (1, 15), (8, 15), (8, 3), (32, 1), (32, 5), (64, 7)])
@torch.no_grad()
def test_eager_captured_prefix_oracle_and_allocation_stability(batch, depth):
    rng = random.Random(batch * 100 + depth)
    # Match a sliced host-mapped confidence plane with a larger retained depth.
    backing = torch.empty((batch, depth + 3), dtype=torch.float32, device="cuda")
    confidence = backing[:, :depth]
    slots = torch.empty(batch, dtype=torch.int64, device="cuda")
    submitted = torch.empty(batch, dtype=torch.bool, device="cuda")
    source_valid = torch.empty_like(submitted)
    limits = torch.empty(batch, dtype=torch.int32, device="cuda")
    lengths = torch.empty_like(limits)
    valid = torch.empty_like(submitted)
    budget = 128 if (batch, depth) == (32, 1) else batch * (depth + 1) // 2
    operation = PreparedCausalPrefixAllocation(confidence, slots, submitted, source_valid, limits,
                                               lengths, valid, token_capacity=budget, request_capacity=64)
    confidence.fill_(1.)
    slots.copy_(torch.arange(batch, device="cuda"))
    submitted.fill_(True)
    source_valid.fill_(True)
    limits.fill_(depth + 1)
    valid.copy_(operation.functional()(confidence, slots, submitted, source_valid, limits, lengths))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        valid.copy_(operation(confidence, slots, submitted, source_valid, limits, lengths))
    values = [0., -0., 1., .5, 2. ** -149, 2. ** -126, 1. - 2. ** -24]
    for iteration in range(40):
        probabilities = [[rng.choice(values + [rng.random()]) for _ in range(depth)] for _ in range(batch)]
        identities = rng.sample(range(64), batch)
        active = [rng.random() > .2 for _ in range(batch)]
        checked = [True] * batch
        caps = [rng.randint(1, depth + 1) for _ in range(batch)]
        if iteration % 7 == 0:
            active[0] = True
            probabilities[0][-1] = float("nan")
        elif iteration % 7 == 1:
            active[0] = True
            checked[0] = False
        elif iteration % 7 == 2 and batch > 1:
            active[:2] = [True, True]
            identities[1] = identities[0]
        confidence.copy_(torch.tensor(probabilities, dtype=torch.float32, device="cuda"))
        slots.copy_(torch.tensor(identities, device="cuda"))
        submitted.copy_(torch.tensor(active, device="cuda"))
        source_valid.copy_(torch.tensor(checked, device="cuda"))
        limits.copy_(torch.tensor(caps, dtype=torch.int32, device="cuda"))
        lengths.fill_(-111)
        valid.fill_(True)
        torch.cuda.synchronize()
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
        assert (lengths.tolist(), valid.tolist()) == allocate(confidence.cpu().tolist(), identities,
                                                               active, checked, caps, budget, 64)
    graph.reset()


@torch.no_grad()
def test_signed_zero_frontier_order_and_fp64_underflow_survival():
    confidence = torch.tensor([[-0.] * 7, [0.] * 7, [2. ** -149] * 7], device="cuda")
    slots = torch.tensor([0, 1, 2], device="cuda")
    submitted = torch.ones(3, dtype=torch.bool, device="cuda")
    limits = torch.full((3,), 8, dtype=torch.int32, device="cuda")
    lengths = torch.empty_like(limits)
    valid = torch.empty_like(submitted)
    for budget in range(3, 25):
        operation = PreparedCausalPrefixAllocation(confidence, slots, submitted, submitted, limits,
            lengths, valid, token_capacity=budget, request_capacity=3)
        valid.copy_(operation.functional()(confidence, slots, submitted, submitted, limits, lengths))
        assert (lengths.tolist(), valid.tolist()) == allocate(confidence.tolist(), slots.tolist(),
            submitted.tolist(), submitted.tolist(), limits.tolist(), budget, 3)


def test_compiler_access_analysis_preserves_read_only_prefix_inputs():
    from torch._higher_order_ops.triton_kernel_wrap import identify_accessed_tensors
    from ignition.components.causal_prefix import _allocate

    confidence = torch.full((4, 2), .75, device="cuda")
    slots = torch.arange(4, device="cuda", dtype=torch.int64)
    submitted = torch.ones(4, device="cuda", dtype=torch.bool)
    source_valid = torch.ones_like(submitted)
    limits = torch.full((4,), 3, device="cuda", dtype=torch.int32)
    lengths = torch.empty_like(limits)
    valid = torch.empty_like(submitted)
    operation = PreparedCausalPrefixAllocation(confidence, slots, submitted, source_valid, limits,
        lengths, valid, token_capacity=6, request_capacity=4)
    arguments = dict(zip(_allocate.arg_names, operation.args, strict=True))
    arguments.update(num_warps=4, enable_fp_fusion=False)
    accesses = identify_accessed_tensors(_allocate, arguments, {})
    assert {dependency.name for dependency in accesses.read_writes.writes} == {"Lengths", "Valid"}
