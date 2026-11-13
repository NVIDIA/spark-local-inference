# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Target feedback capture preserves row means and accepted-tail order in native replay."""

import os

import pytest
import torch

from ignition.components.dspark import PreparedFeedbackCapture
from ignition.components.paged_batch import Metadata
from ignition.preparation.compilation import compile_forward
from ignition.preparation.graphs import PreparedGraphBundle

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an assigned CUDA device")


@torch.no_grad()
def test_feedback_capture_alternates_native_capacities_without_allocation():
    device = torch.device("cuda", 0)
    torch.manual_seed(2167)
    source = torch.randn((8192, 4, 4096), device=device, dtype=torch.bfloat16)
    seed = source.clone()
    variants, graphs, owners = {}, {}, []
    for capacity in (8192, 4096, 256, 128, 8):
        metadata = Metadata.allocate(capacity, 1, 1, device)
        capture = PreparedFeedbackCapture(metadata, 3 * 4096, compact_prefill=True)
        observed = torch.empty(capture.shape, dtype=torch.bfloat16, device=device)
        inputs = source[:capacity]

        def forward(inputs=inputs, capture=capture, metadata=metadata, observed=observed):
            state = capture.begin(metadata)
            for column in range(3):
                capture(state, inputs, column * 4096)
                # Subsequent target layers replace the residual before their feedback is captured.
                inputs.add_(.125)
            observed.copy_(state[0])

        compiled = compile_forward(forward)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled()
        key = str(capacity)
        graphs[key] = graph
        variants[capacity] = metadata, observed
        owners.append(compiled)
    bundle = PreparedGraphBundle(device=0, graphs=graphs, buffers={"source": source},
                                 resources=(variants, owners))
    worker = bundle.start_worker()
    try:
        for capacity in (8192, 8, 4096, 128, 256, 8192):
            metadata, observed = variants[capacity]
            counts = sorted({0, 1, min(127, capacity), min(128, capacity),
                             min(129, capacity), capacity - 7, capacity})
            for step, count in enumerate(counts):
                begin = 7 if count <= capacity - 7 else 0
                start = (0, 127, 999983)[step % 3]
                metadata.lengths.fill_(count)
                metadata.query_start_loc.copy_(torch.tensor([begin, begin + count], device=device))
                metadata.positions.fill_(-1)
                metadata.positions[begin:begin + count].copy_(
                    torch.arange(start, start + count, device=device))
                reference = seed[:capacity].clone()
                expected = torch.empty_like(observed)
                for column in range(3):
                    means = reference.float().mean(1).bfloat16()
                    target = expected[:, column * 4096:(column + 1) * 4096]
                    if capacity > 128:
                        kept = min(count, 128)
                        target.zero_()
                        target[:kept].copy_(means[begin + count - kept:begin + count])
                    else:
                        target.copy_(means)
                    reference.add_(.125)
                source.copy_(seed)
                observed.fill_(float("nan"))
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                ticket = worker.submit(bundle.graph_indices[str(capacity)])
                completion = worker.poll(timeout_ms=120000)
                assert completion is not None and completion.ticket == ticket and completion.error is None
                after = torch.cuda.memory_stats()
                for key in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key]
                torch.testing.assert_close(observed, expected, rtol=0, atol=0)
                torch.testing.assert_close(source[:capacity], reference, rtol=0, atol=0)
    finally:
        worker.close()
        for graph in graphs.values():
            graph.reset()
