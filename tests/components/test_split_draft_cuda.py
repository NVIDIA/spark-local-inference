# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Confidence projection preserves checkpoint arithmetic inside captured rounds."""

import os

import pytest
import torch

from ignition.preparation.compilation import compile_forward

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@torch.no_grad()
def test_confidence_head_preserves_fp32_bias_and_preceding_token_features(with_bias, weight_dtype):
    from types import SimpleNamespace
    from ignition.components.split_draft import PreparedDraftConfidence
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.manual_seed(811)
    batch, block, hidden, rank = 4, 7, 6144, 256
    head = torch.randn(batch * block, hidden, device="cuda").bfloat16()
    markov = torch.randn(batch, block, rank, device="cuda").bfloat16()
    weight = (torch.randn(1, hidden + rank, device="cuda") * .01).to(weight_dtype)
    bias = torch.tensor([.71], device="cuda") if with_bias else None
    output = torch.empty(batch, block, device="cuda")
    with RecordedPreparationSession(device=torch.device("cuda", 0), checkpoint="confidence",
                                    workload_schema="confidence", numerical_policy="default") as session:
        source = SimpleNamespace(owner=SimpleNamespace(block_size=block, logit_offset=0, session=session),
            batch=batch, width=block + 1, draft=SimpleNamespace(head_hidden=head),
            markov=SimpleNamespace(embedding=markov[:, 0]))
        confidence = PreparedDraftConfidence(source, weight, output, bias=bias, name="confidence")

        def forward():
            for position in range(block):
                confidence.record_embedding(position, markov[:, position])
            confidence()

        session.tune()
        compiled = compile_forward(forward)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        for shift in (0., .25, -.5):
            head.add_(shift)
            markov.add_(shift)
            features = torch.cat((head.view(batch, block, hidden).float(), markov.float()), dim=-1)
            expected = (features @ weight.float().T).squeeze(-1)
            if bias is not None:
                expected.add_(bias)
            expected.sigmoid_()
            confidence.features.fill_(float("nan"))
            confidence.embeddings.fill_(float("nan"))
            confidence.logits.fill_(float("nan"))
            output.fill_(float("nan"))
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            torch.testing.assert_close(output, expected, atol=2e-6, rtol=2e-6)
        graph.reset()
