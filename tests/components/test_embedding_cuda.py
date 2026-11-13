# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint embeddings share prepared lookup and native device-count replay."""

import os
from pathlib import Path

import pytest
import torch

from ignition.components.embedding import PreparedEmbedding
from ignition.components.weights import DirectWeightLoader
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned CUDA device")


@pytest.fixture(scope="module", params=["Qwen3.8-Flash-Next", "GLM-5.3-Flash"])
def weight(request):
    snapshot = Path("/data/checkpoint-releases") / f"{request.param}-NVFP4-MXFP8-CSF-QAD"
    if not snapshot.is_dir():
        pytest.skip("model checkpoint is not available")
    name = "model.language_model.embed_tokens.weight"
    with DirectWeightLoader(inspect_indexed_checkpoint(snapshot), "cuda:0", prefixes=(name,)) as loader:
        return loader.load_tensor(name)


@pytest.mark.parametrize("device_count", [False, True])
@torch.no_grad()
def test_embedding_changed_ids_and_live_counts_native_replay(weight, device_count):
    rows, device = 4096, weight.device
    ids = torch.zeros(rows, dtype=torch.int64, device=device)
    count = torch.zeros(1, dtype=torch.int32, device=device) if device_count else None
    output = torch.zeros((rows, weight.shape[1]), dtype=weight.dtype, device=device)
    graph, worker = torch.cuda.CUDAGraph(), None
    with RecordedPreparationSession(device=device, checkpoint="embedding-component",
            workload_schema="embedding-v1", numerical_policy="default") as session:
        try:
            lookup = PreparedEmbedding(weight, ids, output, num_rows=count, session=session, name="embedding")
            session.tune()
            compiled = compile_forward(lookup)
            compiled()
            torch.cuda.synchronize(device)
            with session.capture(), torch.cuda.graph(graph):
                compiled()
            session.freeze()
            bundle = PreparedGraphBundle(device=0, graphs={"lookup": graph},
                buffers={"weight": weight, "ids": ids, "output": output}, resources=(lookup, compiled, session))
            worker = bundle.start_worker()
            for live in (1, rows, 0, 65):
                ids.random_(0, weight.shape[0])
                if count is not None:
                    count.fill_(live)
                else:
                    live = rows
                output.fill_(42)
                expected = torch.zeros_like(output)
                expected[:live].copy_(weight.index_select(0, ids[:live]))
                torch.cuda.synchronize(device)
                before = torch.cuda.memory_stats(device)
                ticket = worker.submit(0, repetitions=3)
                completion = worker.poll(timeout_ms=120000)
                assert completion is not None and completion.ticket == ticket and completion.error is None
                after = torch.cuda.memory_stats(device)
                for key in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key], key
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
        finally:
            if worker is not None:
                worker.close()
            graph.reset()
