# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph-level cache lengths shared by every decoder layer."""

import os
from types import SimpleNamespace

import pytest
import torch

from ignition.preparation.compilation import compile_forward
from ignition.recipes.minimax_m3 import decoder as module


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("fused", [False, True])
def test_decoder_layers_share_invocation_cache_lengths(device, fused):
    if device != "cpu" and os.environ.get("FLARE_RUN_CUDA_TESTS") != "1":
        pytest.skip("requires an assigned CUDA device")
    rows, layers = 8, 60
    hidden = torch.zeros((rows, 4), device=device)
    observed = torch.empty((layers, rows), dtype=torch.int32, device=device)
    metadata = SimpleNamespace(lengths=torch.zeros(rows, dtype=torch.int32, device=device),
        starts=torch.zeros(rows, dtype=torch.int32, device=device),
        sample_indices=torch.arange(rows, device=device), live_counts=torch.tensor([rows], device=device),
        query_start_loc=torch.arange(rows + 1, device=device, dtype=torch.int32),
        token_ids=torch.zeros(rows, dtype=torch.int64, device=device))

    class Layer:
        def __init__(self, index):
            self.output = observed[index]

        def normalize(self, residual, update, normalized):
            return residual

        def __call__(self, source, residual=None, *, metadata, lengths, schedule_storage):
            self.output.copy_(lengths)
            return (source, None) if fused else source

    decoder = module.PreparedDecoder.__new__(module.PreparedDecoder)
    decoder.fused, decoder.mode, decoder.vision, decoder.taps = fused, "verify", None, {}
    decoder.layers = [Layer(index) for index in range(layers)]
    decoder.embedding = SimpleNamespace(lookup=lambda **kwargs: hidden.clone(), gather=lambda value: value)
    decoder.schedules = lambda lengths, cu: None
    decoder.norm = lambda source, indices: source[indices]
    decoder.head = lambda source: source.clone()
    assert decoder.scratch_tensors() == ()
    forward = decoder if device == "cpu" else compile_forward(decoder)
    graph = None
    if device != "cpu":
        forward(metadata)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward(metadata)
    starts = [0, 127, 128, 4095, 4096, 1048570, 65535, 131071]
    for counts in ([1, 6, 0, 1, 6, 6, 0, 2], [0] * rows, [6, 1, 6, 0, 1, 0, 6, 6]):
        metadata.starts.copy_(torch.tensor(starts, dtype=torch.int32, device=device))
        metadata.lengths.copy_(torch.tensor(counts, dtype=torch.int32, device=device))
        expected = torch.tensor([start + count if count > 0 else 0 for start, count in zip(starts, counts)],
            dtype=torch.int32, device=device).expand(layers, -1)
        observed.fill_(-197)
        allocated = torch.cuda.memory_allocated() if graph is not None else None
        if graph is None:
            result = forward(metadata)
            assert len(result) == 4 and result[2:] == (None, None)
        else:
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    if graph is not None:
        graph.reset()


def test_attention_binds_declared_query_width_for_ragged_rows(monkeypatch):
    from ignition.components.paged_batch import Metadata

    metadata = Metadata.allocate(5, 2, 1, "cpu", max_query_tokens=4)
    source = torch.zeros((5, 8), dtype=torch.bfloat16)
    scratch = SimpleNamespace(projected=source, query=source, index_query=source,
        selected=torch.zeros((5, 1), dtype=torch.int32), attended=source, output=source)
    weights = SimpleNamespace(qkv_index=None, q_norm=None, k_norm=None,
        index_q_norm=None, index_k_norm=None, output=None)
    observed = {}

    def indexer(*args, max_q_per_request, **kwargs):
        observed["index_width"] = max_q_per_request
        return SimpleNamespace(functional=lambda **kwargs: None)

    def attention(*args, width, **kwargs):
        observed["attention_width"] = width
        return SimpleNamespace(functional=lambda **kwargs: None)

    monkeypatch.setattr(module, "projection", lambda *args, **kwargs: lambda source: None)
    monkeypatch.setattr(module, "PreparedPackedAttention", lambda *args, **kwargs: SimpleNamespace(functional=lambda **kwargs: None))
    monkeypatch.setattr(module, "PreparedMSAIndexer", indexer)
    monkeypatch.setattr(module, "PreparedPagedAttention", attention)
    module.PreparedAttention(0, weights, SimpleNamespace(text={"rms_norm_eps": 1e-6}), None,
        source, scratch, metadata, SimpleNamespace(cache=lambda *args: source, memory_capacity=torch.tensor([1, 1])), mode="verify",
        inverse_frequency=None, session=None, name="attention", workspace=None,
        cache_lengths=torch.zeros(2, dtype=torch.int32))
    assert observed == {"index_width": 4, "attention_width": 4}
