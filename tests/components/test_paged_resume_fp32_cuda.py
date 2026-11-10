# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Cached-prefix resume keeps FP32 target logits through compiled sampling."""

import os
from types import SimpleNamespace

import pytest
import torch
from b12x.preparation import PreparationSession

from ignition.components.paged_resume import PreparedResume
from ignition.components.sampling import DeviceRequestState, RequestAdmission
from ignition.preparation.compilation import compile_forward
from tests.components.completion_support import completion_outputs

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned CUDA device",
)


@pytest.mark.parametrize("rows", [1, 4])
@torch.no_grad()
def test_cached_resume_fp32_head_projection_and_capture(rows):
    device = torch.device("cuda:0")
    torch.manual_seed(159)
    capacity, hidden, vocab = 4, 5120, 1024
    saved = torch.randn(6, hidden, dtype=torch.bfloat16, device=device) * 0.125
    weights = SimpleNamespace(
        head=torch.randn(vocab, hidden, dtype=torch.bfloat16, device=device) * 0.03125,
        shard=SimpleNamespace(world_size=1),
    )
    pool = SimpleNamespace(
        backing=torch.empty(1, device=device, dtype=torch.uint8), last_hidden=saved
    )
    requests = DeviceRequestState.allocate(device, vocab_size=vocab, capacity=capacity)
    inputs = SimpleNamespace(
        row_slot_ids=torch.arange(rows, device=device, dtype=torch.int32),
        row_generations=torch.full((rows,), 7, device=device, dtype=torch.int64),
        live_rows=torch.zeros(1, device=device, dtype=torch.int32),
        request_state_rows=torch.arange(1, capacity + 1, device=device)[:, None]
        .expand(-1, 8)
        .contiguous(),
    )
    records = torch.zeros(2, capacity, 32, device=device, dtype=torch.int64)
    frame = torch.zeros(1, device=device, dtype=torch.int32)
    sequence = torch.ones(1, device=device, dtype=torch.int64)

    def admit():
        for slot in range(capacity):
            requests.admit(
                slot,
                RequestAdmission(7, 13, temperature=1.0, top_k=1, top_p=1.0, max_new_tokens=100),
                [1, 2, 3],
                consumed_tokens=3,
            )
        inputs.live_rows.fill_(rows)

    with PreparationSession(
        device=device, autotune=False, compile_workers=2
    ) as session:
        operation = PreparedResume(
            weights,
            requests,
            pool,
            inputs,
            completion_outputs(records),
            frame,
            sequence,
            rows=rows,
            runtime=None,
            vocabulary_runtime=None,
            coordinator=None,
            session=session,
            name="resume-fp32",
            logits_dtype=torch.float32,
        )
        assert operation.projection.projection.plan.query.output_dtype == "float32"
        assert operation.completion.logits is None and operation.logits is None
        assert operation.local_logits is None
        observed_logits = torch.empty((rows, vocab), dtype=torch.float32, device=device)
        observed_local = torch.empty_like(observed_logits)
        observed_hidden = torch.empty((rows, hidden), dtype=torch.bfloat16, device=device)
        project = operation.projection
        local_project = project.projection
        def observe_local(source):
            logits = local_project(source)
            observed_local.copy_(logits)
            return logits
        project.projection = observe_local
        def observe(hidden):
            observed_hidden.copy_(hidden)
            logits = project(hidden)
            observed_logits.copy_(logits)
            return logits
        operation.projection = observe
        admit()
        call = compile_forward(operation)
        call()
        torch.cuda.synchronize()
        admit()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            call()
        session.freeze()
        try:
            for step in range(3):
                saved.add_(0.00390625)
                admit()
                expected_hidden = saved[inputs.request_state_rows[:rows, 0]]
                expected = torch.mm(
                    expected_hidden, weights.head.T, out_dtype=torch.float32
                )
                observed_logits.fill_(float("nan"))
                observed_local.fill_(float("nan"))
                observed_hidden.fill_(float("nan"))
                before = torch.cuda.memory_stats()
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                for key in (
                    "allocation.all.allocated",
                    "allocation.all.freed",
                    "segment.all.allocated",
                    "segment.all.freed",
                ):
                    assert before[key] == after[key]
                torch.testing.assert_close(
                    observed_logits, expected, rtol=2e-5, atol=2e-5
                )
                torch.testing.assert_close(
                    observed_logits, observed_local, rtol=0, atol=0
                )
                assert (observed_logits != observed_logits.bfloat16().float()).any()
                assert requests.consumed_tokens.eq(3).all()
                torch.testing.assert_close(
                    observed_hidden, expected_hidden, rtol=0, atol=0
                )
        finally:
            graph.reset()
