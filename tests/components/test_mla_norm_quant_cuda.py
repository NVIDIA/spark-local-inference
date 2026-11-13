# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared MLA normalization retains BF16 outputs and encoded query storage."""

import os

import pytest
import torch

from ignition.components.mla_norm_quant import PreparedMLANormQuant
from ignition.preparation.compilation import compile_forward
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.preparation.tuning import RecordedPreparationSession


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned GPU")


@pytest.mark.parametrize("functional", [False, True])
@torch.no_grad()
def test_fused_mla_norm_default_tune_compiles_and_replays(functional):
    device = torch.device("cuda:0")
    projected = torch.empty((8, 1792), dtype=torch.bfloat16, device=device)
    query = torch.empty((8, 1280), dtype=torch.bfloat16, device=device)
    kv = torch.empty((8, 512), dtype=torch.bfloat16, device=device)
    qw = torch.ones(1280, dtype=torch.bfloat16, device=device)
    kw = torch.ones(512, dtype=torch.bfloat16, device=device)
    with RecordedPreparationSession(device=device, checkpoint="mla-norm-encoded",
            workload_schema="query-kv", numerical_policy="bf16-floor1e-4", compile_workers=2) as session:
        prepared = PreparedMLANormQuant(projected, (qw, kw), query, kv, eps=1e-20,
            session=session, name="norm", release_dependents=True)
        encoded_bases, encoded = prepared.encoded_bases, prepared.encoded
        if functional:
            prepared.functional(session=session)
        session.tune()

        def run():
            if functional:
                q, k, bases = prepared(projected)
                query.copy_(q)
                kv.copy_(k)
                for observed, value in zip(encoded_bases, bases, strict=True):
                    observed.copy_(value)
            else:
                prepared(projected)

        forward = compile_forward(run)
        projected.normal_(std=.125)
        forward()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()
        session.freeze()
        bundle = PreparedGraphBundle(device=0, graphs={"norm": graph}, buffers={"projected": projected},
            resources=(projected, qw, kw, query, kv, prepared, forward, session))
        worker = bundle.start_worker()
        try:
            for scale in (.125, .25, .125):
                projected.normal_(std=scale)
                query.fill_(float("nan"))
                kv.fill_(float("nan"))
                for base in encoded_bases:
                    base.zero_()
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                ticket = worker.submit(bundle.graph_indices["norm"])
                result = worker.poll(timeout_ms=120000)
                assert result is not None and result.ticket == ticket and result.error is None
                after = torch.cuda.memory_stats()
                for counter in ("allocation.all.allocated", "allocation.all.freed",
                                "segment.all.allocated", "segment.all.freed"):
                    assert before[counter] == after[counter]
                for source, output in ((projected[:, :1280], query), (projected[:, 1280:], kv)):
                    value = source.float()
                    expected = (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-20)).bfloat16()
                    assert torch.equal(output, expected)
                assert encoded.values.view(torch.uint8).count_nonzero() > 0
        finally:
            worker.close()
            graph.reset()
