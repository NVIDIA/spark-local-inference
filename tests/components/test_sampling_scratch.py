# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speculative scratch follows graph capacities and operation lifetimes."""

import pytest
import torch

from ignition.components.capacity import speculative_shapes
from ignition.components.parallel_draft import prepare_sampling_scratch
from ignition.preparation.scratch import GraphBufferPool


@pytest.mark.parametrize("width,proposal,verification,keys", [
    (6, True, True, {"proposal", "target", "verify"}),
    (6, True, False, {"proposal"}),
    (6, False, True, {"target", "verify"}),
    (1, True, True, {"target", "verify"}),
])
def test_sampling_allocates_only_executed_operations(width, proposal, verification, keys):
    pool = GraphBufferPool("cpu", chunk_bytes=256)
    work = prepare_sampling_scratch(4, width, 64, "cpu", buffers=pool.cursor(),
                                   proposal=proposal, verification=verification)
    assert work.keys() == keys
    for name, layout in work.items():
        live = list(layout.values())
        if name == "verify":
            live.extend(work["target"][key] for key in ("probabilities", "valid"))
        tensors = [tensor for tensor in live if tensor.numel()]
        spans = sorted((t.data_ptr(), t.data_ptr() + t.nbytes) for t in tensors)
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
        for index, tensor in enumerate(tensors, 1):
            tensor.fill_(bool(index % 2) if tensor.dtype == torch.bool else index)
        for index, tensor in enumerate(tensors, 1):
            assert torch.all(tensor == (bool(index % 2) if tensor.dtype == torch.bool else index))


def test_qwen_sampling_catalog_fits_bounded_shared_backing():
    pool = GraphBufferPool("meta", chunk_bytes=2 << 20)
    shapes = sorted(speculative_shapes(32, (1, 2, 3, 4, 5, 6), 32),
                    key=lambda shape: shape[0] * shape[1], reverse=True)
    for batch, width in shapes:
        cursor = pool.cursor()
        work = prepare_sampling_scratch(batch, width, 248320, "meta", buffers=cursor)
        cursor.zeros((batch, width - 1, 248320), torch.float32)
        assert work["target"]["scores"].numel() <= 32 * 248320
    assert pool.nbytes < 288 << 20
    capacity = pool.nbytes
    for batch, width in reversed(shapes):
        cursor = pool.cursor()
        prepare_sampling_scratch(batch, width, 248320, "meta", buffers=cursor)
        cursor.zeros((batch, width - 1, 248320), torch.float32)
    assert pool.nbytes == capacity


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
    scratch = {name: torch.empty(shape, dtype=dtype) for name, (shape, dtype) in operation.layout.items()}
    operation._prepare_scores(scratch, logits)
    for name in ("scores", "maximum", "valid", "packed_top_k", "packed_top_p"):
        torch.testing.assert_close(scratch[name], getattr(operation, name), rtol=0, atol=0)
