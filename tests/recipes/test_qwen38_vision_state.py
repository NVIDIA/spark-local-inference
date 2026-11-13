# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Native vision feature transfer and prefill injection, excluding encoder math."""

from __future__ import annotations

import os

import pytest
import torch

from tests.components.completion_support import completion_outputs

from tests.tensor_examples import TensorExamples
from ignition.recipes.qwen38.sampling import DeviceRequestState, RequestAdmission
from ignition.recipes.qwen38.vision_state import (
    PreparedEmbeddingInjection,
    PreparedVisionHandoff,
    VisionFeatureState,
    VisionInputs,
)

CUDA = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires assigned GPU"
)


def test_native_vision_staging_and_arena_geometry():
    features = VisionFeatureState.allocate("cpu")
    inputs = VisionInputs.allocate("cpu")
    assert features.arena.shape == (4, 16384, 2560)
    assert features.arena.dtype == torch.bfloat16
    assert features.arena.numel() * 2 == 335544320
    assert (
        torch.count_nonzero(features.count)
        == torch.count_nonzero(features.generation)
        == 0
    )
    assert inputs.vision_pixel_values.shape == (16384, 1536)
    assert inputs.vision_interpolation_indices.shape == (16384, 4)
    assert inputs.vision_position_ids.shape == (16384, 2)
    assert inputs.vision_slot.item() == -1
    assert inputs.vision_num_patches.item() == 0


def _device():
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    return torch.device("cuda", 0)


def _replay(worker, graph=0):
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
    ticket = worker.submit(graph)
    result = worker.poll(timeout_ms=30000)
    assert result.ticket == ticket and result.error is None
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated


@CUDA
@pytest.mark.parametrize("compiled", [False, True])
@torch.no_grad()
def test_native_vision_handoff_append_epoch_validation_and_completion(compiled):
    from ignition.preparation.graphs import PreparedGraphBundle

    device = _device()
    features, inputs = (
        VisionFeatureState.allocate(device, image_slots=2),
        VisionInputs.allocate(device),
    )
    requests = DeviceRequestState.allocate(device)
    requests.admit(2, RequestAdmission(17, 93), [1, 2, 3, 4, 5], consumed_tokens=2)
    output = torch.arange(64 * 2560, device=device).reshape(64, 2560).bfloat16()
    records = torch.zeros((2, 4, 32), dtype=torch.int64, device=device)
    frame = torch.zeros(1, dtype=torch.int32, device=device)
    sequence = torch.ones(1, dtype=torch.int64, device=device)
    pool = TensorExamples(device)
    buffers = pool
    pixels = buffers.zeros((256, 1536), torch.bfloat16)
    operation = PreparedVisionHandoff(
        inputs, features, requests, output, completion_outputs(records), frame, sequence, pixels=pixels
    )
    operation.prepare()
    metadata_example, errors = operation.metadata, torch.empty_like(operation.errors)
    operation.functional()
    def forward():
        values, metadata = operation.begin()
        operation.finish(values, output)
        errors.copy_(values["errors"])
        pixels.copy_(values["pixels"])
        for name in metadata.__dataclass_fields__:
            getattr(metadata_example, name).copy_(getattr(metadata, name))

    if compiled:
        from ignition.preparation.compilation import compile_forward

        forward = compile_forward(forward)
        forward()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
    worker = PreparedGraphBundle(
        device=0,
        graphs={"vision": graph},
        buffers={"features": features.arena},
        resources=(operation, forward),
    ).start_worker()
    try:
        features.arena.fill_(7)
        requests_before = {
            name: value.clone() for name, value in vars(requests).items()
        }
        features.feature_slot.fill_(1)
        inputs.vision_slot.fill_(2)
        inputs.vision_generation.fill_(17)
        inputs.vision_num_patches.fill_(16)
        inputs.vision_cu_seqlens[1] = 16
        inputs.vision_pixel_values[:16].normal_()
        pixels.fill_(37)
        errors.fill_(37)
        for tensor in vars(metadata_example).values():
            tensor.fill_(37)
        _replay(worker)
        assert errors.item() == 0
        assert features.count.tolist() == [0, 0, 4, 0]
        assert features.generation.tolist() == [0, 0, 17, 0]
        assert torch.equal(features.arena[1, :4], output[:4])
        assert torch.all(features.arena[1, 4:] == 7)
        assert torch.all(features.arena[0] == 7)
        assert records[0, 2][..., :8].tolist() == [17, 1, 2, 0, -1, 1, 0, 0]
        assert records[0, 0][..., :8].tolist() == [0, 0, 0, 0, -1, 0, 0, 0]
        for name, before in requests_before.items():
            assert torch.equal(getattr(requests, name), before), name
        torch.testing.assert_close(pixels[:16], inputs.vision_pixel_values[:16].bfloat16(), rtol=0, atol=0)
        assert torch.count_nonzero(pixels[16:]) == 0
        inputs.vision_feature_offset.fill_(4)
        inputs.vision_num_patches.fill_(8)
        inputs.vision_cu_seqlens[1] = 8
        frame.fill_(1)
        sequence.fill_(2)
        _replay(worker)
        assert features.count[2].item() == 6
        assert torch.equal(features.arena[1, 4:6], output[:2])
        assert records[1, 2][..., :8].tolist() == [17, 2, 2, 0, -1, 1, 0, 0]
        saved = features.arena[1].clone()
        for kind, error in (
            ("offset", 8),
            ("index", 16),
            ("pixel", 32),
            ("output", 64),
            ("epoch", 2),
            ("feature_slot", 1),
        ):
            features.feature_slot.fill_(1)
            requests.status[2] = 1
            requests.error_code[2] = 0
            inputs.vision_feature_offset.fill_(6)
            inputs.vision_generation.fill_(17)
            inputs.vision_interpolation_indices.zero_()
            inputs.vision_pixel_values.zero_()
            output[0, 0] = 0
            if kind == "feature_slot":
                features.feature_slot.fill_(2)
            if kind == "offset":
                inputs.vision_feature_offset.fill_(7)
            if kind == "index":
                inputs.vision_interpolation_indices[0, 0] = 999999
            if kind == "pixel":
                inputs.vision_pixel_values[0, 0] = float("nan")
            if kind == "output":
                output[0, 0] = float("nan")
            if kind == "epoch":
                inputs.vision_generation.fill_(16)
            _replay(worker)
            assert errors.item() & error
            assert features.count[2].item() == 6
            assert torch.equal(features.arena[1], saved)
            assert torch.all(features.arena[0] == 7)
            assert records[1, 2, 5].item() == 3
            if kind == "epoch":
                assert requests.status[2].item() == 1
                assert requests.error_code[2].item() == 0
            if kind == "index":
                assert metadata_example.num_patches.item() == 0
                assert (
                    torch.count_nonzero(metadata_example.interpolation_indices) == 0
                )
                assert torch.count_nonzero(metadata_example.position_ids) == 0
                assert torch.count_nonzero(pixels) == 0
    finally:
        worker.close()
        graph.reset()


@CUDA
@pytest.mark.parametrize("capacity", [512, 3072, 4096, 8192])
@pytest.mark.parametrize("compiled", [False, True])
@torch.no_grad()
def test_native_embedding_injection_precedes_gather_and_model_mutation(capacity, compiled):
    from ignition.components.token_state import PreparedTokenState
    from ignition.preparation.graphs import PreparedGraphBundle

    device = _device()
    features = VisionFeatureState.allocate(device, image_slots=1)
    features.feature_slot.fill_(0)
    requests = DeviceRequestState.allocate(device)
    prompt = [1, 2, 3, 4] + [1] * (capacity - 4)
    requests.admit(3, RequestAdmission(27, 5), prompt)
    indices = torch.full((capacity,), -1, dtype=torch.int32, device=device)
    slots = torch.tensor([3], dtype=torch.int32, device=device)
    epochs = torch.tensor([27], dtype=torch.int64, device=device)
    lengths = torch.tensor([capacity], dtype=torch.int32, device=device)
    live = torch.ones(1, dtype=torch.int32, device=device)
    embedding = torch.full((capacity, 2560), 7, dtype=torch.bfloat16, device=device)
    operation = PreparedEmbeddingInjection(
        features, requests, indices, slots, epochs, lengths, live, embedding
    )
    operation.prepare()
    errors = torch.empty_like(operation.errors)
    operation.functional()
    tokens = torch.tensor(prompt, dtype=torch.int64, device=device)
    starts = torch.tensor([0, capacity], dtype=torch.int32, device=device)
    token_state = PreparedTokenState(requests, tokens, starts, slots, epochs, live, history=requests.ple_history, history_pad_id=248044)
    token_state.prepare()
    def forward():
        current_errors = operation.validate()
        errors.copy_(current_errors)
        token_state.gather()
        operation.inject(embedding, current_errors)

    if compiled:
        from ignition.preparation.compilation import compile_forward

        forward = compile_forward(forward)
        forward()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
    worker = PreparedGraphBundle(
        device=0,
        graphs={"inject": graph},
        buffers={"embedding": embedding},
        resources=(operation, token_state, forward),
    ).start_worker()
    try:
        features.generation[3] = 27
        features.count[3] = 3
        features.arena[0, :3].copy_(
            torch.arange(3 * 2560, device=device).reshape(3, 2560).bfloat16()
        )
        indices[:4].copy_(
            torch.tensor([-1, 2, 0, -1], dtype=torch.int32, device=device)
        )
        indices[-1] = 1
        _replay(worker)
        assert token_state.active.item() == 1
        assert torch.all(embedding[[0, 3]] == 7)
        assert torch.equal(embedding[1], features.arena[0, 2])
        assert torch.equal(embedding[2], features.arena[0, 0])
        assert torch.all(embedding[4:-1] == 7)
        assert torch.equal(embedding[-1], features.arena[0, 1])
        assert requests.consumed_tokens[3].item() == 0
        for bad_index in (-2, 3, 16384):
            requests.status[3] = 1
            embedding.fill_(7)
            indices[-1] = bad_index
            _replay(worker)
            assert errors.item() == 2
            assert token_state.active.item() == 0
            assert requests.status[3].item() == 3
            assert torch.all(embedding == 7)
        requests.status[3] = 1
        indices[-1] = 0
        features.generation[3] = 26
        _replay(worker)
        assert requests.status[3].item() == 3
        assert token_state.active.item() == 0
        requests.status[3] = 1
        indices.fill_(-1)
        _replay(worker)
        assert requests.status[3].item() == 1
        assert token_state.active.item() == 1
        lengths.fill_(capacity - 1)
        indices[-1] = -2
        _replay(worker)
        assert requests.status[3].item() == 1
        assert errors.item() == 0
        assert torch.all(embedding == 7)
        lengths.fill_(capacity)
        epochs.fill_(26)
        indices[0] = 16384
        _replay(worker)
        assert requests.status[3].item() == 1
        assert token_state.active.item() == 0
        assert torch.all(embedding == 7)
    finally:
        worker.close()
        graph.reset()
