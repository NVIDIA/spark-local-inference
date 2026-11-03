# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Designated-rank sampling preserves shared forward storage aliases."""

import os
from types import SimpleNamespace

import pytest
import torch

from tests.components.completion_support import completion_outputs

from ignition.components.paged_batch import BatchInputs
from ignition.components.paged_generation import PreparedGeneration
from ignition.components.sampling import DeviceRequestState, PreparedSampling
from ignition.preparation.aot import _signature
from ignition.preparation.scratch import GraphBufferPool, bind_scratch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


@torch.no_grad()
def test_sampling_rank_preserves_subsequent_shared_graph_aliases():
    device = torch.device("cuda:0")
    requests = DeviceRequestState.allocate(device, vocab_size=128896, capacity=16)
    inputs = BatchInputs.allocate(16, 1, device, capacity=16)
    records = torch.zeros(2, 16, 32, dtype=torch.int64, device=device)
    frame = torch.zeros(1, dtype=torch.int32, device=device)
    sequence = torch.zeros(1, dtype=torch.int64, device=device)
    logits = torch.zeros(16, requests.vocab_size, dtype=torch.bfloat16, device=device)
    decoder = SimpleNamespace(weights=SimpleNamespace(draft=None), scratch=SimpleNamespace(logits=logits),
                              head=SimpleNamespace(gather=SimpleNamespace(runtime=None)))
    pools, signatures = [], []
    for rank in (0, 1):
        pool = GraphBufferPool(device, chunk_bytes=2 << 20)
        validation = torch.zeros(1, 32, dtype=torch.bfloat16, device=device)
        batch = SimpleNamespace(inputs=inputs, rows=16, requests=requests, shard=SimpleNamespace(rank=rank),
                                validation_input=validation, validation_sum=torch.empty_like(validation))
        forward = PreparedGeneration(decoder, batch, completion_outputs(records), frame, sequence, runtime=None,
            coordinator=None, session=None, name="decode16", workspace=None, buffers=pool.cursor())
        assert forward.compilation_variant == "sample"
        shared = bind_scratch(PreparedSampling.scratch_layout(4, requests.vocab_size),
                              device=device, workspace=pool.cursor())
        signatures.append(_signature(tuple(shared.values())))
        pools.append(pool)
    assert signatures[0] == signatures[1]
    assert pools[0].nbytes == pools[1].nbytes
