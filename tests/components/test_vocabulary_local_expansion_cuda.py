# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local vocabulary heads expand physical projection rows before distributed sampling."""

import os

import pytest
import torch

from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.components.vocabulary_logprobs import PreparedVocabularyParallelLogprobs
from ignition.components.vocabulary_sampling import PreparedVocabularySoftmax
from ignition.components.completion import CompletionOutputs
from ignition.components.sampling import DeviceRequestState, RequestAdmission
from ignition.preparation.compilation import compile_forward
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.preparation.scratch import GraphBufferPool
from tests.components.test_generation_logprobs_cuda import _assert_entry

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("physical", range(1, 9))
@pytest.mark.parametrize("local_only", [True, False])
@torch.no_grad()
def test_physical_rows_expand_to_eight_logical_rows_under_native_replay(physical, local_only):
    device, logical, hidden, columns = torch.device("cuda", 0), 8, 128, 64
    source = (torch.arange(physical * hidden, device=device).reshape(physical, hidden) % 11).bfloat16() / 64
    weight = (torch.arange(columns * hidden, device=device).reshape(columns, hidden) % 7).bfloat16() / 128
    pool = GraphBufferPool(device, chunk_bytes=1 << 20)
    cursor = pool.cursor()
    projected = cursor.zeros((physical, columns), torch.float32)
    backing = cursor.zeros((logical, columns * 4), torch.float32)
    assert projected.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr()
    expanded = backing.view(-1)[:logical * columns].view(logical, columns)
    indices = torch.arange(logical, device=device, dtype=torch.int64) % physical
    graph = worker = None
    with RecordedPreparationSession(device=device, checkpoint="local-vocabulary-expansion",
            workload_schema="physical-to-logical-rows", numerical_policy="fp32",
            autotune_effort="off") as session:
        head = PreparedVocabularyProjection(source, weight, projected, expanded,
            runtime=None, session=session, coordinator=None, name="head", row_indices=indices,
            gather=not local_only)
        session.tune(show_progress=False)
        call = compile_forward(lambda: head(source))
        call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            call()
        bundle = PreparedGraphBundle(device=0, graphs={"head": graph}, buffers={},
                                     resources=(head, call, source, weight, indices, backing, pool, session))
        worker = bundle.start_worker()
        try:
            for turn in range(4):
                source.add_(.125)
                # Negative indices retain the existing PyTorch row-index semantics.
                mapping = ((torch.arange(logical, device=device) * 3 + turn) % physical)
                mapping[::3] -= physical
                indices.copy_(mapping)
                projected.fill_(float("nan"))
                backing.fill_(float("nan"))
                expected = torch.nn.functional.linear(source.float(), weight.float())[indices]
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
                ticket = worker.submit(0)
                result = worker.poll(10000)
                assert result is not None and result.error is None and result.ticket == ticket
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
                torch.testing.assert_close(expanded, expected, rtol=1e-5, atol=1e-6)
                assert backing.view(-1)[logical * columns:].isnan().all()
                if local_only:
                    assert not hasattr(head, "gather")
        finally:
            worker.close()
            graph.reset()


@torch.no_grad()
def test_expanded_local_rows_feed_target_probabilities_and_raw_logprobs():
    from ignition.recipes.deepseek41.decoder import PreparedDecoder
    from ignition.components.paged_batch import Metadata
    from types import SimpleNamespace

    b, w, physical, columns, hidden = 2, 4, 4, 64, 128
    state = DeviceRequestState.allocate("cuda", vocab_size=columns, capacity=b)
    for slot in range(b):
        state.admit(slot, RequestAdmission(7, slot + 17, top_k=-1, top_p=1.), [0])
    slots = torch.arange(b, device="cuda", dtype=torch.int32)
    epochs = torch.full((b,), 7, device="cuda", dtype=torch.int64)
    live = torch.tensor([b], device="cuda", dtype=torch.int32)
    frame = torch.zeros(1, device="cuda", dtype=torch.int32)
    metadata = Metadata.allocate(physical, b, 1, "cuda", samples=b * w, max_query_tokens=w)
    decoder = PreparedDecoder.__new__(PreparedDecoder)
    decoder.scratch = SimpleNamespace(
        hidden=(torch.arange(b * w * hidden, device="cuda").reshape(b * w, hidden) % 11).bfloat16() / 64,
        local_logits=torch.empty(b * w, columns, device="cuda"),
        logits=torch.empty(b * w, columns * 4, device="cuda"))
    decoder.weights = SimpleNamespace(head=(torch.arange(columns * hidden, device="cuda")
        .reshape(columns, hidden) % 7).bfloat16() / 128)
    decoder.output_metadata = metadata
    p = torch.empty(b, w, columns, device="cuda")
    valid = torch.empty(b, w, device="cuda", dtype=torch.bool)
    candidates = torch.zeros(b, w - 1, device="cuda", dtype=torch.int64)
    outputs = CompletionOutputs.allocate((2, b, 32), device="cuda")
    outputs.requested.fill_(1)
    outputs.records[0, :, 0] = 7
    outputs.records[0, :, 8] = 1
    outputs.records[0, :, 16] = 3
    with RecordedPreparationSession(device=torch.device("cuda", 0), checkpoint="expanded-target-rows",
            workload_schema="packed-target-head", numerical_policy="fp32", autotune_effort="off") as session:
        decoder._prepare_head(vocabulary_parallel=True, vocabulary_runtime=None,
                              coordinator=None, session=session, name="target")
        target = PreparedVocabularySoftmax(decoder.local_logits.view(b, w, columns), p, valid,
            state, slots, epochs, live, candidates, start=0, runtime=None, session=session,
            coordinator=None, name="target.sampling")
        logprobs = PreparedVocabularyParallelLogprobs(decoder.local_logits.view(b, w, columns), outputs,
            slots, epochs, live, frame, vocabulary=columns, offset=0, runtime=None, session=session,
            coordinator=None, name="target.logprobs")
        session.tune(show_progress=False)
        target.prepare()
        logprobs.prepare()
        def forward():
            decoder.head(decoder.head_hidden)
            target()
            logprobs()
        call = compile_forward(forward)
        call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            call()
        try:
            for lengths in ([1, 3], [2, 0], [0, 4], [2, 2]):
                metadata.set(torch.zeros(b, device="cuda", dtype=torch.int64),
                    torch.tensor(lengths, device="cuda", dtype=torch.int32),
                    torch.arange(b, device="cuda", dtype=torch.int64),
                    torch.zeros(b, 1, device="cuda", dtype=torch.int32),
                    torch.zeros(b, w, device="cuda", dtype=torch.int64), live)
                decoder.scratch.local_logits.fill_(float("nan"))
                decoder.scratch.logits.fill_(float("nan"))
                graph.replay()
                expected = torch.nn.functional.linear(decoder.head_hidden.float(), decoder.weights.head.float())
                expected = expected[metadata.sample_indices].view(b, w, columns)
                torch.testing.assert_close(decoder.local_logits.view_as(expected), expected, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(p, expected.softmax(-1), rtol=3e-6, atol=1e-7)
                assert valid.all()
                for row in range(b):
                    _assert_entry(outputs.logprobs[0, row, 0], expected[row, 0], 3)
                assert decoder.scratch.local_logits[physical:].isnan().all()
        finally:
            graph.reset()
