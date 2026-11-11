# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Captured image delivery against the checkpoint's ViT and aligner."""

import importlib.util
from dataclasses import fields, is_dataclass
import json
import os
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from tests.components.completion_support import completion_outputs
from b12x.preparation import PreparationSession

from ignition.components.paged_batch import BatchInputs
from ignition.components.sampling import DeviceRequestState
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.checkpoint import IndexedCheckpoint, _read_header
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
from ignition.components.weights import DirectWeightLoader

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned CUDA device"
)


@pytest.mark.parametrize("compiler", ("fx", "aot"))
@pytest.mark.parametrize("height,width,capacity", [(24, 24, 576), (25, 31, 1152), (93, 93, 9216)])
@torch.no_grad()
def test_compiled_vision_publishes_only_valid_current_generation(compiler, height, width, capacity):
    snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170"
    )
    snapshot = Path(os.environ.get("FLARE_DS41_SNAPSHOT", snapshot))
    if not (snapshot / "inference/vision.py").is_file():
        pytest.skip("DeepSeek V4.1 Flash checkpoint is unavailable")
    device = torch.device("cuda", torch.cuda.current_device())
    prefixes = ("vision.", "aligner.", "image_")
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted({shard for key, shard in index.items() if key.startswith(prefixes)})
    if any(not (snapshot / shard).is_file() for shard in shards):
        pytest.skip("vision weight shards are unavailable")
    tensors, records = {}, []
    for shard in shards:
        record, entries = _read_header(snapshot / shard, shard)
        records.append(record)
        tensors.update({name: value for name, value in entries.items() if name.startswith(prefixes)})
    manifest = IndexedCheckpoint(snapshot, MappingProxyType(tensors), tuple(records))
    with DirectWeightLoader(manifest, device, prefixes=prefixes) as loader:
        weights = {name: loader.load_tensor(name) for name in loader._sources}
    spec = importlib.util.spec_from_file_location(
        "deepseek41_vision_reference", snapshot / "inference/vision.py"
    )
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    args = SimpleNamespace(dim=5120, vision_dim=1024, vision_n_layers=32, vision_n_heads=16,
        vision_inter_dim=2816, vision_patch_size=14, vision_rope_theta=10000., vision_downsample_ratio=3)
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
    patches = height * width
    feature_h, feature_w = (height + 2) // 3, (width + 2) // 3
    features = feature_h * feature_w
    scratch = VisionScratch.allocate(capacity, device, hidden=5120)
    inputs = VisionInputs.allocate(device, patches=9216)
    requests = DeviceRequestState.allocate(device, vocab_size=129280, capacity=4)
    batch = BatchInputs.allocate(512, 8, device, capacity=4)
    state = VisionState(weights, batch, PrefillVisionInputs.allocate(512, device), requests, features=4096,
        token_base=129280, sentinel_names=("image_start", "image_start", "image_newline", "image_end"), feature_type=1,
        image_slots=2)
    records = torch.zeros((2, 4, 32), dtype=torch.int64, device=device)
    frame, sequence = (
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.ones(1, dtype=torch.int64, device=device),
    )
    arena = ScratchArena(device)
    inputs.vision_num_patches.fill_(patches)
    inputs.vision_num_features.fill_(features)
    inputs.vision_cu_seqlens[1] = patches
    inputs.vision_position_ids[:patches].copy_(
        torch.stack(
            torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij"),
            -1,
        ).view(-1, 2)
    )
    indices = [
        [(row * 3 + dy) * width + col * 3 + dx if row * 3 + dy < height and col * 3 + dx < width else -1
         for dy in range(3) for dx in range(3)]
        for row in range(feature_h)
        for col in range(feature_w)
    ]
    inputs.vision_merge_indices[:features].copy_(torch.tensor(indices, device=device))
    inputs.vision_slot.fill_(2)
    state.feature_slot.fill_(1)
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
            source = torch.randn((patches, 588), dtype=torch.bfloat16, device=device)
            inputs.vision_pixel_values[:patches].copy_(source)
            state.arena.fill_(float("nan"))
            sequence.fill_(iteration + 2)
            def poison_examples(value):
                if is_dataclass(value):
                    for field in fields(value):
                        poison_examples(getattr(value, field.name))
                else:
                    value.fill_(float("nan") if value.is_floating_point() else -37)
            poison_examples(scratch)
            graph.replay()
            torch.cuda.synchronize()
            with torch.device(device):
                expected = aligner(vit(source.view(patches, 3, 14, 14), height, width), height, width)
            actual = state.arena[1, :features]
            assert torch.isfinite(actual).all()
            assert (
                torch.nn.functional.cosine_similarity(
                    actual.float().flatten(), expected.float().flatten(), dim=0
                )
                > 0.9999
            )
            torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.008)
            assert int(state.generation[2]) == 7 and int(state.count[2]) == features
            assert torch.isnan(state.arena[0]).all()
        previous = state.arena[1].clone()
        inputs.vision_generation.fill_(6)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(state.arena[1], previous, equal_nan=True, rtol=0, atol=0)
        assert int(state.generation[2]) == 7 and int(state.count[2]) == features
        graph.reset()
