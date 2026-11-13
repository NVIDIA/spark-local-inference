# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native RADIO preprocessing and prompt expansion against the checkpoint processor."""

import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import pytest
import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = Path(os.environ.get("FLARE_NEMOTRON_CHECKPOINT", "/data/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB"))


@pytest.fixture(scope="module")
def reference():
    if not CHECKPOINT.is_dir():
        pytest.skip("Nemotron Bolt checkpoint is unavailable")
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    cls = get_class_from_dynamic_module("image_processing_nemotron_h_omni.NemotronH_Omni_Reasoning_V3ImageProcessor", str(CHECKPOINT))
    subprocess.run(["cargo", "build", "--quiet", "--manifest-path", str(ROOT / "helios/Cargo.toml"),
                    "--features", "server,test-tools", "--bin", "test_radio_images"], check=True)
    return cls.from_pretrained(CHECKPOINT), Tokenizer.from_file(str(CHECKPOINT / "tokenizer.json"))


@pytest.mark.parametrize("sizes,mode,format", [([(37, 29)], "RGB", "png"), ([(512, 512)], "RGB", "png"),
    ([(1600, 101)], "RGB", "png"), ([(2048, 1024)], "RGB", "png"), ([(83, 127)], "RGBA", "png"),
    ([(149, 31)], "L", "png"), ([(503, 701), (149, 31)], "RGB", "png"), ([(512, 512)], "RGB", "jpeg")])
def test_native_radio_images(reference, tmp_path, sizes, mode, format):
    from ignition.recipes.nemotron_h.contract import FRONTEND, VISION
    processor, tokenizer = reference
    images, inputs = [], []
    for i, (width, height) in enumerate(sizes):
        pixels = np.random.default_rng(41 + i).integers(0, 256, (height, width, 4), dtype=np.uint8)
        image = Image.fromarray(pixels, "RGBA").convert(mode)
        path = tmp_path / f"image{i}.{format}"
        image.save(path)
        with Image.open(path) as decoded:
            images.append(decoded.convert("RGB"))
        inputs.append(dict(path=str(path), format=format))
    expected = processor(images=images, return_tensors="pt")
    prompt = "Describe " + " and ".join(["<image>"] * len(images)) + "."
    case = tmp_path / "case.json"
    case.write_text(json.dumps(dict(images=inputs, prompt=prompt, frontend=FRONTEND,
                                    vision={**VISION, "feature_slots": 4})))
    subprocess.run([str(ROOT / "helios/target/debug/test_radio_images"), str(CHECKPOINT), str(CHECKPOINT),
                    str(case), str(tmp_path)], check=True)
    actual = json.loads((tmp_path / "metadata.json").read_text())
    packed = []
    for image in expected["pixel_values"]:
        c, h, w = image.shape
        packed.append(image.reshape(c, h // 16, 16, w // 16, 16).permute(1, 3, 0, 2, 4).flatten())
    native = torch.from_numpy(np.fromfile(tmp_path / "pixels.f32", dtype=np.float32))
    torch.testing.assert_close(native, torch.cat(packed), rtol=0, atol=1e-5)
    tokens, indices, offset = [], [], 0
    prefix, suffix = tokenizer.encode("<img>", add_special_tokens=False).ids, tokenizer.encode("</img>", add_special_tokens=False).ids
    counts = iter(expected["num_tokens"])
    for token in tokenizer.encode(prompt, add_special_tokens=False).ids:
        if token != 18:
            tokens.append(token)
            indices.append(-1)
        else:
            count = next(counts)
            tokens.extend(prefix + [18] * count + suffix)
            indices.extend([-1] * len(prefix) + list(range(offset, offset + count)) + [-1] * len(suffix))
            offset += count
    assert actual["tokens"] == tokens
    assert actual["indices"] == indices
    table = torch.randn(128, 128, 3, generator=torch.Generator().manual_seed(93))
    positions = []
    for _, h, w in actual["grids"]:
        square = torch.nn.functional.interpolate(table.permute(2, 0, 1)[None], (max(h, w),) * 2,
                                                mode="bilinear", align_corners=False)
        positions.append(square[0, :, :h, :w].permute(1, 2, 0).reshape(-1, 3))
    ids = torch.tensor(actual["interpolation_indices"])
    weights = torch.tensor(actual["interpolation_weights"])
    interpolated = (table.reshape(-1, 3)[ids] * weights[:, :, None]).sum(1)
    torch.testing.assert_close(interpolated, torch.cat(positions), rtol=2e-5, atol=2e-5)
