# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared image encoding against the checkpoint's canonical RADIO implementation."""

import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an explicitly assigned CUDA device")


@pytest.fixture(scope="module")
@torch.no_grad()
def models():
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.recipes.nemotron_h.vision import PREFIX, VisionWeights

    path = Path(os.environ.get("FLARE_NEMOTRON_CHECKPOINT", "/data/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB"))
    if not path.is_dir():
        pytest.skip("requires the Nemotron Bolt checkpoint")
    weights = VisionWeights.load(inspect_indexed_checkpoint(path), "cuda:0")
    radio_class = get_class_from_dynamic_module("modeling_radio.RadioModel", str(path))
    config_class = get_class_from_dynamic_module("configuration_radio.RadioConfig", str(path))
    config = config_class(**weights.config.radio)
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        reference = radio_class(config)
    reference.make_preprocessor_external()
    state = {"embeddings.patch_projection.weight": weights.patch,
             "embeddings.position_embedding": weights.tensors[PREFIX + "patch_generator.pos_embed"],
             "embeddings.cls_register_token": weights.tensors[PREFIX + "patch_generator.cls_token.token"],
             "summary_idxs": torch.tensor([0, 1], device="cuda", dtype=torch.int64)}
    for i in range(config.num_hidden_layers):
        for key, value in weights.tensors.items():
            prefix = PREFIX + f"blocks.{i}."
            if not key.startswith(prefix):
                continue
            name = key.removeprefix(prefix)
            target = f"encoder.layer.{i}."
            if name.startswith("attn.qkv."):
                suffix = name.removeprefix("attn.qkv.")
                for part, projection in zip(value.chunk(3, 0), ("query", "key", "value"), strict=True):
                    state[target + "attention.attention." + projection + "." + suffix] = part
            else:
                name = name.replace("attn.proj.", "attention.output.dense.")
                state[target + name] = value
        for j in (1, 2):
            state[f"encoder.layer.{i}.layer_scale{j}.lambda1"] = torch.ones(1280, device="cuda", dtype=torch.bfloat16)
    missing, unexpected = reference.load_state_dict(state, strict=False, assign=True)
    assert missing == ["embeddings.video_patch_projection.weight"]
    assert unexpected == []
    reference.eval()
    return weights, reference


class VisionForward:
    def __init__(self, encoder, capacity, device):
        from ignition.recipes.nemotron_h.vision import PIXEL_WIDTH

        self.encoder = encoder
        self.patches = torch.zeros((capacity, PIXEL_WIDTH),
                                  dtype=encoder.inputs.vision_pixel_values.dtype, device=device)
        self.indices = torch.zeros((capacity, 4), dtype=torch.int64, device=device)
        self.weights = torch.zeros((capacity, 4), dtype=torch.float32, device=device)
        self.merge = torch.zeros((capacity // 4, 4), dtype=torch.int64, device=device)
        self.cu = torch.zeros((2,), dtype=torch.int32, device=device)

    def __call__(self):
        return self.encoder(self.patches, self.indices, self.weights, self.merge, self.cu)


def set_inputs(forward, pixels):
    from ignition.recipes.nemotron_h.vision import REGISTER_TOKENS

    _, _, height, width = pixels.shape
    h, w = height // 16, width // 16
    n = h * w
    packed = pixels.reshape(3, h, 16, w, 16).permute(1, 3, 0, 2, 4).reshape(n, 768)
    forward.patches.zero_()
    forward.patches[:n].copy_(packed)
    positions = torch.stack(torch.meshgrid(torch.arange(h, device="cuda"), torch.arange(w, device="cuda"), indexing="ij"), -1).reshape(-1, 2)
    source = (128. / max(h, w) * (positions.float() + .5) - .5).clamp_min(0)
    lo = source.long()
    hi = (lo + 1).clamp_max(127)
    frac = source - lo
    y, x = lo.unbind(-1)
    yy, xx = hi.unbind(-1)
    fy, fx = frac.unbind(-1)
    forward.indices.zero_()
    forward.weights.zero_()
    forward.indices[:n].copy_(torch.stack((y * 128 + x, y * 128 + xx, yy * 128 + x, yy * 128 + xx), -1))
    forward.weights[:n].copy_(torch.stack(((1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx), -1))
    merge = torch.arange(n, device="cuda").reshape(h, w).reshape(h // 2, 2, w // 2, 2).permute(0, 2, 1, 3).reshape(-1, 4)
    forward.merge.zero_()
    forward.merge[:n // 4].copy_(merge)
    forward.cu.copy_(torch.tensor([0, n + REGISTER_TOKENS], device="cuda", dtype=torch.int32))
    return n // 4


@torch.no_grad()
def reference_project(weights, features, h, w):
    import torch.nn.functional as F

    features = features.reshape(h // 2, 2, w // 2, 2, 1280).permute(0, 2, 1, 3, 4).reshape(-1, 5120)
    x = features.float()
    x = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * weights.tensors["mlp1.0.weight"].float()).bfloat16()
    x = F.linear(x, weights.tensors["mlp1.1.weight"]).relu().square()
    return F.linear(x, weights.tensors["mlp1.3.weight"])


@pytest.mark.parametrize("capacity,grids", [
    (1024, ((32, 32), (20, 48))),
    (4096, ((64, 64), (40, 100))),
    (13312, ((128, 104), (110, 120))),
])
@torch.no_grad()
def test_checkpoint_image_encoder_compiled_replay(models, capacity, grids, tmp_path):
    from b12x.preparation import PreparationSession
    from ignition.preparation.compile_pool import CompilePool
    from ignition.preparation.scratch import ScratchArena
    from ignition.recipes.nemotron_h.vision import PreparedVision, VisionInputs, VisionScratch

    weights, reference = models
    torch.manual_seed(872)
    pixels = [torch.randn((1, 3, h * 16, w * 16), device="cuda", dtype=torch.bfloat16) for h, w in grids]
    expected = [reference_project(weights, reference(x).features[0], x.shape[-2] // 16, x.shape[-1] // 16) for x in pixels]
    with PreparationSession(device="cuda:0", autotune=False) as session:
        arena = ScratchArena("cuda:0")
        encoder = PreparedVision(weights, VisionScratch.allocate(capacity, "cuda:0"), VisionInputs.allocate("cuda:0"),
                                 session=session, workspace=arena, name="radio")
        arena.finalize()
        forward = VisionForward(encoder, capacity, "cuda:0")
        artifact_path = tmp_path / "radio.bin"
        with CompilePool(pixels[0].device) as compiler:
            compiler.submit(forward, artifact_path).result()
        with artifact_path.open("rb") as file:
            artifact = torch.compiler.load_compiled_function(file)
        run = lambda: artifact(forward)
        set_inputs(forward, pixels[0])
        run()
        session.freeze()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            output = run()
        for index in (0, 1, 0):
            count = set_inputs(forward, pixels[index])
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            actual = output[:count].float()
            target = expected[index].float()
            error = actual - target
            assert error.norm() / target.norm() < .025
            assert error.abs().max() / target.abs().max() < .04
