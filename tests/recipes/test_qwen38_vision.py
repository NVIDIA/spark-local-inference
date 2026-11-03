# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Exact-checkpoint vision encoder and native CUDA graph qualification."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from tests.components.completion_support import completion_outputs

from ignition.preparation import inspect_qwen38_checkpoint
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.preparation.scratch import GraphBufferPool
from ignition.recipes.qwen38.vision import PreparedVision, VisionMetadata, VisionScratch, VisionWeights
from ignition.recipes.qwen38.weights import QwenWeightLoader
from ignition.reference.qwen38_quant import (
    CheckpointTensors, decode_mxfp8, decode_nvfp4, quantize_mxfp8_activation,
    read_nvfp4_projection,
)

CUDA = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                         reason="requires explicit GPU assignment and FLARE_RUN_CUDA_TESTS=1")


def test_vision_buffer_geometry():
    metadata = VisionMetadata.allocate(256, 2, torch.device("cpu"))
    scratch = VisionScratch.allocate(256, torch.device("cpu"))
    assert scratch.patches.shape == (256, 1536)
    assert scratch.query.shape == (256, 16, 72)
    assert scratch.output.shape == (64, 2560)
    assert metadata.position_ids.shape == (256, 2)
    assert metadata.cu_seqlens.shape == (3,)
    assert metadata.num_patches.item() == 0
    with pytest.raises(ValueError):
        VisionScratch.allocate(255, torch.device("cpu"))


@pytest.fixture(scope="module")
def manifest():
    path = Path("/data/checkpoint-releases/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD")
    path = Path(os.environ.get("FLARE_QWEN38_SNAPSHOT", path))
    if not path.exists():
        pytest.skip("exact checkpoint is unavailable")
    return inspect_qwen38_checkpoint(path)


def _module(manifest):
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpVisionConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel
    config = Qwen4ExpVisionConfig.from_dict(json.loads((manifest.snapshot / "config.json").read_text())["vision_config"])
    config._attn_implementation = "eager"
    with torch.device("meta"):
        module = Qwen4ExpVisionModel(config)
    return module.eval().requires_grad_(False)


class _ReferenceProjection(torch.nn.Module):
    def __init__(self, weight, bias, *, quantized, scale=1.):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)
        self.quantized, self.scale = quantized, scale

    def forward(self, source):
        source = quantize_mxfp8_activation(source) if self.quantized else source.float()
        return ((source @ self.weight.T) * self.scale).bfloat16() + self.bias


def _reference(manifest, weights, device):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionRotaryEmbedding
    module = _module(manifest)
    for path, value in weights.parameters.items():
        parent, _, attribute = path.rpartition(".")
        setattr(module.get_submodule(parent), attribute, torch.nn.Parameter(value, requires_grad=False))
    with CheckpointTensors(manifest, byte_budget=512 << 20) as reader:
        for path in weights.projections:
            prefix = f"model.visual.{path}"
            if ".attn." in path:
                decoded = decode_mxfp8(reader.read(prefix + ".weight"), reader.read(prefix + ".weight_scale"))
                scale, quantized = 1., True
            else:
                packed = read_nvfp4_projection(reader, prefix)
                decoded = decode_nvfp4(packed.packed, packed.scales, torch.ones_like(packed.weight_scale_2)).bfloat16().float()
                scale, quantized = packed.weight_scale_2.item(), False
            parent, _, attribute = path.rpartition(".")
            setattr(module.get_submodule(parent), attribute,
                    _ReferenceProjection(decoded.to(device), weights.parameters[path + ".bias"],
                                         quantized=quantized, scale=scale))
    module.rotary_pos_emb = Qwen4ExpVisionRotaryEmbedding(module.config).to(device)
    return module


def _input(manifest, variant):
    from PIL import Image
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(manifest.snapshot, local_files_only=True)
    width, height = 256, 256
    pixels = bytes(component for y in range(height) for x in range(width)
                   for component in ((x * 7 + variant * 31) % 256, (y * 11) % 256, (x + y * 3) % 256))
    image = Image.frombytes("RGB", (width, height), pixels)
    return processor.image_processor(images=[image], return_tensors="pt")


@CUDA
@torch.no_grad()
def test_actual_vision_encoder_native_replay(manifest, record_property):
    from b12x.preparation import PreparationSession
    from transformers.vision_utils import (
        get_vision_position_ids, get_vision_interpolation_indices_and_weights,
    )
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.fail("assign a CUDA device explicitly")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    assert torch.cuda.get_device_capability(device) in ((12, 0), (12, 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    module = _module(manifest)
    with QwenWeightLoader(manifest, device, prefixes=("model.visual.",)) as loader:
        weights = VisionWeights.load(module, loader)
    reference = _reference(manifest, weights, device)
    scratch = VisionScratch.allocate(256, device)
    metadata = VisionMetadata.allocate(256, 2, device)
    pixels = torch.zeros((256, 1536), dtype=torch.float32, device=device)
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        prepared = PreparedVision(module, scratch, metadata, weights, session=session)
        session.freeze()
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            prepared(pixels)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            prepared(pixels)
        worker = PreparedGraphBundle(device=0, graphs={"vision": graph},
            buffers={"pixels": pixels, "embeddings": scratch.output},
            resources=(prepared, weights, session, stream)).start_worker()
        errors = []
        try:
            for variant in (0, 1):
                inputs = _input(manifest, variant)
                assert inputs.pixel_values.shape == pixels.shape
                grid = inputs.image_grid_thw
                indices, weights_ref = get_vision_interpolation_indices_and_weights(
                    grid, num_grid_per_side=48, mode="bilinear", align_corners=True, spatial_merge_size=2)
                metadata.interpolation_indices.copy_(indices)
                metadata.interpolation_weights.copy_(weights_ref)
                metadata.position_ids.copy_(get_vision_position_ids(grid, 2))
                metadata.cu_seqlens.copy_(torch.tensor([0, 256, 256], dtype=torch.int32))
                metadata.num_patches.fill_(256)
                pixels.copy_(inputs.pixel_values)
                expected = reference(inputs.pixel_values.to(device), grid.to(device)).pooler_output.float()
                torch.cuda.synchronize(device)
                scratch.output.fill_(float("nan"))
                torch.cuda.synchronize(device)
                allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                ticket = worker.submit(0)
                completion = worker.poll(timeout_ms=30000)
                assert completion is not None and completion.ticket == ticket and completion.error is None
                assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
                actual = scratch.output.float()
                relative = float((actual - expected).norm() / expected.norm().clamp_min(1e-12))
                errors.append({"variant": variant, "relative_l2": relative,
                               "max_absolute": float((actual - expected).abs().max())})
                assert relative < .03, errors
                torch.testing.assert_close(actual, expected, rtol=.05, atol=float(expected.abs().max()) * .03)
        finally:
            worker.close()
        record_property("numerical_errors", json.dumps(errors))


@CUDA
@torch.no_grad()
def test_vision_generation_compiles_in_worker_and_replays_after_poisoning(manifest, tmp_path):
    from b12x.preparation import PreparationSession
    from ignition.preparation.compile_pool import CompilePool
    from ignition.preparation.scratch import ScratchArena
    from ignition.recipes.qwen38.sampling import DeviceRequestState
    from ignition.recipes.qwen38.vision_state import (
        PreparedVisionGeneration, PreparedVisionHandoff, VisionFeatureState, VisionInputs,
    )
    from transformers.vision_utils import get_vision_position_ids, get_vision_interpolation_indices_and_weights

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    module = _module(manifest)
    with QwenWeightLoader(manifest, device, prefixes=("model.visual.",)) as loader:
        weights = VisionWeights.load(module, loader)
    pool, arena = GraphBufferPool(device, chunk_bytes=2 << 20), ScratchArena(device)
    cursor = pool.cursor()
    scratch = VisionScratch.allocate(256, device, buffers=cursor)
    inputs, features = VisionInputs.allocate(device), VisionFeatureState.allocate(device, capacity=1)
    requests = DeviceRequestState.allocate(device, capacity=1)
    requests.generation.fill_(7)
    requests.status.fill_(1)
    records = torch.zeros((2, 1, 32), dtype=torch.int64, device=device)
    frame = torch.zeros(1, dtype=torch.int32, device=device)
    sequence = torch.ones(1, dtype=torch.int64, device=device)
    handoff = PreparedVisionHandoff(inputs, features, requests, scratch.output, completion_outputs(records), frame, sequence,
                                   pixels=scratch.patches, buffers=cursor)
    handoff.prepare()
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        encoder = PreparedVision(module, scratch, handoff.metadata, weights, session=session,
                                 projection_workspace=arena, attention_workspace=arena)
        session.freeze()
        forward = PreparedVisionGeneration(handoff, encoder)
        path = tmp_path / "vision.pt2"
        with CompilePool(device, compile_workers=1) as compilers:
            compilers.submit(forward, path).result()
        with path.open("rb") as file:
            compiled = torch.compiler.load_compiled_function(file)
        grid = torch.tensor([[1, 16, 16]], device=device)
        indices, factors = get_vision_interpolation_indices_and_weights(
            grid, num_grid_per_side=48, mode="bilinear", align_corners=True, spatial_merge_size=2)
        inputs.vision_interpolation_indices[:256].copy_(indices)
        inputs.vision_interpolation_weights[:256].copy_(factors)
        inputs.vision_position_ids[:256].copy_(get_vision_position_ids(grid, 2))
        inputs.vision_cu_seqlens[1] = 256
        inputs.vision_num_patches.fill_(256)
        inputs.vision_slot.zero_()
        inputs.vision_generation.fill_(7)
        features.feature_slot.zero_()
        inputs.vision_pixel_values[:256].normal_()
        def reset():
            features.count.zero_()
            features.generation.zero_()
            requests.status.fill_(1)

        reset()
        forward()
        expected = features.arena[0, :64].clone()
        reset()
        compiled(forward)
        reset()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled(forward)
        worker = PreparedGraphBundle(device=0, graphs={"vision": graph}, buffers={},
            resources=(forward, compiled, pool, arena, session)).start_worker()
        try:
            for _ in range(2):
                for chunks in pool.chunks.values():
                    for tensor in chunks:
                        tensor.fill_(float("nan") if tensor.is_floating_point() else -37)
                reset()
                features.arena.fill_(float("nan"))
                torch.cuda.synchronize()
                allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
                ticket = worker.submit(0)
                result = worker.poll(timeout_ms=30000)
                assert result is not None and result.ticket == ticket and result.error is None
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
                assert handoff.errors.item() == 0
                assert features.count.item() == 64
                torch.testing.assert_close(features.arena[0, :64], expected, rtol=0, atol=0)
        finally:
            worker.close()


@CUDA
@pytest.mark.parametrize("capacity", [256, 16384])
@pytest.mark.parametrize("compiled", [False, True])
@torch.no_grad()
def test_pooled_vision_native_replay_matches_unpooled_after_poisoning(manifest, capacity, compiled):
    from b12x.preparation import PreparationSession
    from transformers.vision_utils import get_vision_position_ids, get_vision_interpolation_indices_and_weights

    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    modules = [_module(manifest), _module(manifest)]
    with QwenWeightLoader(manifest, device, prefixes=("model.visual.",)) as loader:
        weights = VisionWeights.load(modules[0], loader)
    pool = GraphBufferPool(device, chunk_bytes=2 << 20)
    cursor = pool.cursor()
    scratch = [VisionScratch.allocate(capacity, device), VisionScratch.allocate(capacity, device, buffers=cursor)]
    metadata = [VisionMetadata.allocate(capacity, 1, device),
                VisionMetadata.allocate(capacity, 1, device, buffers=cursor)]
    pixels = torch.zeros((capacity, 1536), dtype=torch.float32, device=device)
    side = int(capacity ** .5)
    grid = torch.tensor([[1, side, side]], device=device)
    indices, factors = get_vision_interpolation_indices_and_weights(
        grid, num_grid_per_side=48, mode="bilinear", align_corners=True, spatial_merge_size=2)
    positions = get_vision_position_ids(grid, 2)
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        prepared = [PreparedVision(module, s, m, weights, session=session, name=f"vision{i}")
                    for i, (module, s, m) in enumerate(zip(modules, scratch, metadata, strict=True))]
        session.freeze()
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        graphs = {}
        owners = []
        for index, encoder in enumerate(prepared):
            forward = lambda encoder=encoder: encoder(pixels)
            if compiled and index == 1:
                import io
                from ignition.preparation.aot import aot_forward

                artifact = aot_forward(forward)
                artifact(forward)
                data = type(artifact).serialize(artifact).serialized_data
                restored = torch.compiler.load_compiled_function(io.BytesIO(data))
                owners.extend((forward, artifact, restored))
                forward = lambda owner=forward, restored=restored: restored(owner)
            with torch.cuda.stream(stream):
                forward()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                forward()
            graphs[str(index)] = graph
        worker = PreparedGraphBundle(device=0, graphs=graphs, buffers={"pixels": pixels},
            resources=(*prepared, *owners, weights, session, stream, pool)).start_worker()
        try:
            for seed in (31, 72):
                pixels.normal_(generator=torch.Generator(device=device).manual_seed(seed))
                for chunks in pool.chunks.values():
                    for chunk in chunks:
                        chunk.fill_(float("nan") if chunk.is_floating_point() else -37)
                for m in metadata:
                    m.interpolation_indices.copy_(indices)
                    m.interpolation_weights.copy_(factors)
                    m.position_ids.copy_(positions)
                    m.cu_seqlens.copy_(torch.tensor([0, capacity], dtype=torch.int32, device=device))
                    m.num_patches.fill_(capacity)
                torch.cuda.synchronize(device)
                allocations = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                for index in (0, 1):
                    ticket = worker.submit(index)
                    result = worker.poll(timeout_ms=30000)
                    assert result is not None and result.ticket == ticket and result.error is None
                assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocations
                torch.testing.assert_close(scratch[1].output, scratch[0].output, rtol=0, atol=0)
        finally:
            worker.close()
