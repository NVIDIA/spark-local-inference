# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compiled and native window metadata with layer-specific cache consumers."""

import os
from types import SimpleNamespace

import pytest
import torch

from b12x.preparation import PreparationSession

from ignition.components.compressed_mla import PreparedCompressedCacheWriter, PreparedCompressedMLA
from ignition.components.paged_batch import Metadata
from ignition.components.state import gather_rows
from ignition.components.window_indices import PreparedWindowIndices
from ignition.preparation.compilation import compile_forward
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.recipes.deepseek4.attention import AttentionIndices
from ignition.recipes.deepseek41.attention import AttentionIndices as DeepSeek41Indices
from tests.tensor_examples import TensorExamples
from .test_window_indices import assert_metadata_equal, poison, set_metadata
from .window_indices_reference import LiteralDeepSeek4WindowIndices, LiteralDeepSeek41WindowIndices


def observe_indices(operation, metadata, *, visible=None, replay_start=None):
    values = {name: getattr(operation, name)
              for name in ("window_slots", "window_indices", "window_lengths")}
    if hasattr(operation, "position"):
        values["position"] = operation.position
        values["ratios"] = {ratio: SimpleNamespace(**{name: value
            for name, value in vars(planes).items()}) for ratio, planes in operation.ratios.items()}
    output = SimpleNamespace(**values)
    operation.functional()
    def forward():
        options = dict(visible=visible)
        if replay_start is not None:
            options["replay_start"] = replay_start
        packet = operation(metadata, **options)
        for name in ("window_slots", "window_indices", "window_lengths"):
            getattr(output, name).copy_(getattr(packet, name))
        if hasattr(output, "position"):
            output.position.copy_(packet.position)
            for ratio, planes in output.ratios.items():
                for name, value in vars(planes).items():
                    value.copy_(getattr(packet.ratios[ratio], name))
    return output, forward


def poison_outputs(indices):
    poison(indices)
    if hasattr(indices, "position"):
        indices.position.fill_(-777)
        for planes in indices.ratios.values():
            for value in vars(planes).values():
                value.fill_(-777)


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("recipe,mode,tokens,rows,query_capacity,visible", [
    ("deepseek4", "prefill", 256, 1, 256, True),
    ("deepseek4", "draft", 15, 3, 5, False),
    ("deepseek4", "verify", 24, 8, 3, False),
    ("deepseek41", "decode", 1, 1, 1, False),
    ("deepseek41", "decode", 8, 8, 1, False),
    ("deepseek41", "verify", 6, 1, 6, False),
    ("deepseek41", "verify", 24, 8, 6, False),
    ("deepseek41", "verify", 128, 32, 6, False),
    ("deepseek41", "prefill", 4096, 1, 4096, False),
    ("deepseek41", "draft", 40, 8, 5, False),
])
@torch.no_grad()
def test_compiled_native_window_metadata(recipe, mode, tokens, rows, query_capacity, visible):
    torch._dynamo.reset()
    device = torch.device("cuda:0")
    metadata = Metadata.allocate(tokens, rows, 40, device, max_query_tokens=query_capacity)
    page_size = 128 if mode == "decode" else 128 + query_capacity
    width = page_size if mode == "draft" else (512 if visible else 128)
    visibility = (torch.arange(tokens, device=device) % 193,
                  torch.arange(tokens, device=device).flip(0) % 193) if visible else None
    replay_start = torch.zeros(rows, dtype=torch.int64, device=device) if mode == "prefill" and recipe == "deepseek41" else None
    if recipe == "deepseek4":
        actual = AttentionIndices(metadata, mode=mode, tokens=tokens, visible=visibility)
        reference = LiteralDeepSeek4WindowIndices
    else:
        actual = DeepSeek41Indices(metadata, mode=mode, tokens=tokens, page_size=page_size,
            width=width, replay_start=replay_start, ratios=() if mode == "draft" else (1, 2))
        reference = LiteralDeepSeek41WindowIndices
    expected = reference(metadata, mode=mode, tokens=tokens, page_size=page_size, width=width,
                         visible=visibility, replay_start=replay_start)
    set_metadata(metadata, [127] * rows, [min(query_capacity, tokens // rows)] * rows)
    actual, forward = observe_indices(actual, metadata, visible=visibility, replay_start=replay_start)
    call = compile_forward(forward)
    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    bundle = PreparedGraphBundle(device=0, graphs={"indices": graph}, buffers={}, resources=(actual, call, metadata))
    worker = bundle.start_worker()
    try:
        for start, partial in ((0, False), (127, True), (128, False), (4095, True), (4095, None), (0, True)):
            lengths = [min(query_capacity, tokens // rows)] * rows
            if partial is True:
                lengths[-1] = 0
                lengths[0] = max(0, lengths[0] - 1)
            elif partial is None:
                lengths = [0] * rows
            set_metadata(metadata, [start + i for i in range(rows)], lengths)
            if replay_start is not None:
                replay_start.copy_(metadata.starts if partial is False else torch.zeros_like(replay_start))
            expected()
            poison_outputs(actual)
            forward()
            assert_metadata_equal(actual, expected)
            if recipe == "deepseek41":
                from tests.recipes.test_deepseek41_indices import assert_compression

                assert_compression(actual, metadata)
            poison_outputs(actual)
            call()
            assert_metadata_equal(actual, expected)
            if recipe == "deepseek41":
                assert_compression(actual, metadata)
            poison_outputs(actual)
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            worker.submit(0)
            completion = worker.poll(timeout_ms=30000)
            assert completion is not None and completion.error is None
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            assert_metadata_equal(actual, expected)
            if recipe == "deepseek41":
                assert_compression(actual, metadata)
    finally:
        worker.close()
        graph.reset()


@pytest.mark.parametrize("mode,tokens,rows,query_capacity", [("verify", 24, 8, 6), ("draft", 15, 3, 5)])
@torch.no_grad()
def test_shared_window_metadata_preserves_layer_specific_attention(mode, tokens, rows, query_capacity):
    torch._dynamo.reset()
    device = torch.device("cuda:0")
    torch.manual_seed(4142)
    metadata = Metadata.allocate(tokens, rows, 1, device, max_query_tokens=query_capacity)
    page_size = 128 + query_capacity
    width = page_size if mode == "draft" else 128
    buffers = TensorExamples(device)
    shared = PreparedWindowIndices(metadata, mode=mode, tokens=tokens, page_size=page_size,
        width=width, buffers=buffers)
    layer_storage = tuple(torch.zeros((tokens, width), dtype=dtype, device=device)
                          for dtype in (torch.int64, torch.int32))
    layer_buffers = [tuple(value.view_as(value) for value in layer_storage) for _ in range(2)]
    assert layer_buffers[0][0].data_ptr() == layer_buffers[1][0].data_ptr()
    references = [LiteralDeepSeek41WindowIndices(metadata, mode=mode, tokens=tokens,
        page_size=page_size, width=width) for _ in range(2)]
    query = torch.randn((tokens, 16, 512), dtype=torch.bfloat16, device=device) * .2
    values = [torch.randn((tokens, 512), dtype=torch.bfloat16, device=device) * .2 for _ in range(2)]
    history = [torch.randint(0, 256, (rows + 2, 128 * 528), dtype=torch.uint8, device=device) for _ in range(2)]
    # Initialize valid cache records through the same prepared writer for both arms.
    history_values = [torch.randn(((rows + 2) * 128, 512), dtype=torch.bfloat16, device=device) * .2 for _ in range(2)]
    history_slots = torch.arange((rows + 2) * 128, device=device)
    caches = [[torch.zeros((rows, page_size * 528), dtype=torch.uint8, device=device) for _ in range(2)] for _ in range(2)]
    outputs = [[torch.empty_like(query) for _ in range(2)] for _ in range(2)]
    sinks = [torch.randn(16, device=device) for _ in range(2)]
    writers, attentions = [[], []], [[], []]
    set_metadata(metadata, [127] * rows, [tokens // rows] * rows)
    graph, worker = None, None
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        history_writers = [PreparedCompressedCacheWriter(history_values[i], history[i], history_slots,
            page_size=128, kind="swa", session=session, name=f"history{i}") for i in range(2)]
        for arm in range(2):
            for layer in range(2):
                indices = shared if arm == 0 else references[layer]
                writers[arm].append(PreparedCompressedCacheWriter(values[layer], caches[arm][layer],
                    indices.window_slots, page_size=page_size, kind="swa", session=session,
                    name=f"write{arm}_{layer}"))
                attentions[arm].append(PreparedCompressedMLA(query, caches[arm][layer], indices.window_indices,
                    indices.window_lengths, sinks[layer], outputs[arm][layer], window_page_size=page_size,
                    mode="decode", decode_row_capacity=tokens, compute_mode="bf16", session=session,
                    name=f"attention{arm}_{layer}", workspace=None))
        for writer, data in zip(history_writers, history_values):
            writer(data)
        shared, update_shared = observe_indices(shared, metadata)
        def forward():
            update_shared()
            for arm in range(2):
                for layer in range(2):
                    if arm:
                        references[layer]()
                    else:
                        for value in layer_buffers[layer]:
                            value.fill_(31337)
                    gather_rows(history[layer], metadata.state_rows, caches[arm][layer][:, :128 * 528])
                    writers[arm][layer](values[layer])
                    attentions[arm][layer]()
        compiled = compile_forward(forward)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        bundle = PreparedGraphBundle(device=0, graphs={"layers": graph}, buffers={},
            resources=(forward, compiled, session, shared, references, writers, attentions, history, caches, outputs))
        worker = bundle.start_worker()
        try:
            for start, inactive in ((0, False), (127, True), (128, False), (4095, True), (0, False)):
                lengths = [tokens // rows] * rows
                if inactive:
                    lengths[-1] = 0
                    lengths[0] = max(1, lengths[0] - 1)
                set_metadata(metadata, [start + i for i in range(rows)], lengths)
                for indices in (shared, *references):
                    poison_outputs(indices)
                for arm in outputs:
                    for output in arm:
                        output.fill_(float("nan"))
                for data in values:
                    data.add_(.01)
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
                worker.submit(0)
                completion = worker.poll(timeout_ms=30000)
                assert completion is not None and completion.error is None
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
                for layer in range(2):
                    assert_metadata_equal(shared, references[layer])
                    assert torch.equal(caches[0][layer], caches[1][layer])
                    assert torch.equal(outputs[0][layer], outputs[1][layer])
                assert not torch.equal(outputs[0][0][:1], outputs[0][1][:1])
        finally:
            if worker is not None:
                worker.close()
            if graph is not None:
                graph.reset()
