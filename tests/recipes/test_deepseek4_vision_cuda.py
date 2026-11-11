# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Captured image delivery against the checkpoint's ViT and aligner."""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.components.completion_support import completion_outputs
from b12x.preparation import PreparationSession

from ignition.components.paged_batch import BatchInputs
from ignition.components.parallel import TensorParallelShard
from ignition.components.sampling import DeviceRequestState
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.recipes.deepseek4.config import REVISION
from ignition.components.deepseek_vision import (
    PreparedVision,
    VisionInputs,
    VisionScratch,
)
from ignition.components.deepseek_vision import (
    PrefillVisionInputs,
    PreparedVisionGeneration,
    VisionState,
)
from ignition.recipes.deepseek4.weights import DeepSeekWeightLoader

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned CUDA device"
)


@pytest.mark.parametrize("compiler", ("fx", "aot"))
@torch.no_grad()
def test_compiled_vision_publishes_only_valid_current_generation(compiler):
    snapshot = (
        Path.home()
        / f".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-Vision-Exp/snapshots/{REVISION}"
    )
    if not (snapshot / "inference/vision.py").is_file():
        pytest.skip("DeepSeek V4 Flash Vision checkpoint is unavailable")
    device = torch.device("cuda", torch.cuda.current_device())
    with DeepSeekWeightLoader(
        inspect_indexed_checkpoint(snapshot),
        device,
        shard=TensorParallelShard(0, 2),
        prefixes=("vision.", "aligner.", "image_"),
    ) as loader:
        weights = {name: loader.load_tensor(name) for name in loader._sources}
    spec = importlib.util.spec_from_file_location(
        "deepseek4_vision_reference", snapshot / "inference/vision.py"
    )
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    config = json.loads((snapshot / "config.json").read_text())
    args = SimpleNamespace(**config, dim=config["hidden_size"])
    with torch.device("meta"):
        vit, aligner = reference.ViT(args), reference.Aligner(args)
    vit.load_state_dict(
        {
            name.removeprefix("vision."): value
            for name, value in weights.items()
            if name.startswith("vision.")
        },
        assign=True,
    )
    aligner.load_state_dict(
        {
            name.removeprefix("aligner."): value
            for name, value in weights.items()
            if name.startswith("aligner.")
        },
        assign=True,
    )
    scratch = VisionScratch.allocate(576, device, hidden=4096)
    inputs = VisionInputs.allocate(device, patches=3456)
    requests = DeviceRequestState.allocate(device, vocab_size=129280, capacity=4)
    batch = BatchInputs.allocate(512, 8, device, capacity=4)
    state = VisionState(weights, batch, PrefillVisionInputs.allocate(512, device), requests, features=1536,
        token_base=129280, sentinel_names=("image_start", "image_pad", "image_pad", "image_newline", "image_end"), feature_type=2)
    records = torch.zeros((2, 4, 32), dtype=torch.int64, device=device)
    frame, sequence = (
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.ones(1, dtype=torch.int64, device=device),
    )
    arena = ScratchArena(device)
    inputs.vision_num_patches.fill_(576)
    inputs.vision_num_features.fill_(64)
    inputs.vision_cu_seqlens[1] = 576
    inputs.vision_position_ids[:576].copy_(
        torch.stack(
            torch.meshgrid(torch.arange(24, device=device), torch.arange(24, device=device), indexing="ij"),
            -1,
        ).view(-1, 2)
    )
    indices = [
        [(row * 3 + dy) * 24 + col * 3 + dx for dy in range(3) for dx in range(3)]
        for row in range(8)
        for col in range(8)
    ]
    inputs.vision_merge_indices[:64].copy_(torch.tensor(indices, device=device))
    state.feature_slot.fill_(2)
    inputs.vision_slot.fill_(2)
    inputs.vision_generation.fill_(7)
    requests.generation[2] = 7
    requests.status[2] = 1
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        encoder = PreparedVision(weights, scratch, inputs, session=session, workspace=arena, name="vision")
        operation = PreparedVisionGeneration(encoder, state, completion_outputs(records), frame, sequence)
        if compiler == "aot":
            from ignition.preparation.aot import aot_forward

            artifact = aot_forward(operation)
            def compiled():
                return artifact(operation)
        else:
            compiled = compile_forward(operation)
        compiled()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        for iteration in range(3):
            source = torch.randn((576, 588), dtype=torch.bfloat16, device=device)
            inputs.vision_pixel_values[:576].copy_(source)
            state.arena.fill_(float("nan"))
            sequence.fill_(iteration + 2)
            graph.replay()
            torch.cuda.synchronize()
            with torch.device(device):
                expected = aligner(vit(source.view(576, 3, 14, 14), 24, 24), 24, 24)
            actual = state.arena[2, :64]
            assert torch.isfinite(actual).all()
            assert (
                torch.nn.functional.cosine_similarity(
                    actual.float().flatten(), expected.float().flatten(), dim=0
                )
                > 0.9999
            )
            torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.008)
            assert int(state.generation[2]) == 7 and int(state.count[2]) == 64
            assert torch.isnan(state.arena[0]).all()
        previous = state.arena[2].clone()
        inputs.vision_generation.fill_(6)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(state.arena[2], previous, equal_nan=True, rtol=0, atol=0)
        assert int(state.generation[2]) == 7 and int(state.count[2]) == 64
        graph.reset()
