# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Native RGB preprocessing and image sentinel layout against the checkpoint implementation."""

import importlib.util
import os
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer

from ignition.recipes.deepseek4.config import REVISION

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = (
    Path.home()
    / f".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-Vision-Exp/snapshots/{REVISION}"
)


@pytest.fixture(scope="module")
def reference():
    path = SNAPSHOT / "inference/image_processor.py"
    if not path.is_file():
        pytest.skip("canonical DeepSeek V4 image processor is not cached")
    spec = importlib.util.spec_from_file_location("deepseek4_image_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    subprocess.run(
        [
            "cargo",
            "build",
            "--quiet",
            "--manifest-path",
            str(ROOT / "helios/Cargo.toml"),
            "--features",
            "server,test-tools",
            "--bin",
            "test_deepseek4_images",
        ],
        check=True,
    )
    return module


@pytest.mark.parametrize(
    "sizes,start",
    [
        ([(1, 1)], 0),
        ([(701, 503)], 1),
        ([(67, 101)], 2),
        ([(3000, 47)], 3),
        ([(17, 3000)], 7),
        ([(1000, 1000)], 127),
        ([(503, 701), (149, 31)], 509),
    ],
)
def test_native_images_match_checkpoint(reference, tmp_path, sizes, start):
    args = SimpleNamespace(
        vision_patch_size=14,
        vision_downsample_ratio=3,
        vision_max_n_token=384,
        vision_min_pixels=147456,
        vision_max_wh_ratio=8,
    )
    tokenizer = Tokenizer.from_file(str(SNAPSHOT / "tokenizer.json"))
    placeholder = tokenizer.token_to_id("<｜deepseek_image｜>")
    prompt = "x " * start + "<｜deepseek_image｜>" * len(sizes) + "Describe."
    inputs, expected_patches, grids, counts, positions, cu = [], [], [], [], [], [0]
    for i, (width, height) in enumerate(sizes):
        rgb = np.random.default_rng(i + 51).integers(0, 256, (height, width, 3), dtype=np.uint8)
        path = tmp_path / f"image{i}.rgb"
        path.write_bytes(rgb.tobytes())
        inputs.append({"width": width, "height": height, "pixels": str(path)})
        encoded = io.BytesIO()
        Image.fromarray(rgb).save(encoded, format="PNG")
        patches, h, w, fh, fw = reference.load_image({"data": encoded.getvalue()}, args)
        count = fh * fw * 9
        expected_patches.append(
            torch.nn.functional.pad(patches.reshape(h * w, 588), (0, 0, 0, count - h * w))
        )
        grids.append([1, h, w])
        counts.append(count)
        positions.extend([[y, x] for y in range(h) for x in range(w)] + [[0, 0]] * (count - h * w))
        cu.append(cu[-1] + h * w)
    tokens, indices, spans, i, offset = [], [], [], 0, 0
    for token in tokenizer.encode(prompt, add_special_tokens=False).ids:
        if token != placeholder:
            tokens.append(token)
            indices.append(-1)
            continue
        h, w = ((grids[i][axis] + 2) // 3 for axis in (1, 2))
        types, perm = reference.build_image_block(h, w, len(tokens))
        begin = len(tokens)
        feature_indices = torch.full_like(types, -1)
        feature_indices[types == 2] = perm + offset
        spans.append([begin + int(torch.nonzero(types == 0)[0]), begin + len(types)])
        tokens.extend((129280 + types).tolist())
        indices.extend(feature_indices.tolist())
        offset += h * w
        i += 1
    case = tmp_path / "case.json"
    case.write_text(json.dumps({"images": inputs, "prompt": prompt}))
    subprocess.run(
        [
            str(Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "helios/target")) / "debug/test_deepseek4_images"),
            str(SNAPSHOT),
            str(ROOT / "ignition/recipes/deepseek4/assets"),
            str(case),
            str(tmp_path),
        ],
        check=True,
    )
    actual = json.loads((tmp_path / "metadata.json").read_text())
    assert actual == {
        "tokens": tokens,
        "indices": indices,
        "spans": spans,
        "grids": grids,
        "patches": counts,
        "positions": positions,
        "cu_seqlens": cu,
    }
    pixels = torch.from_numpy(np.fromfile(tmp_path / "pixels.f32", dtype=np.float32)).to(torch.bfloat16)
    torch.testing.assert_close(pixels, torch.cat(expected_patches).flatten(), rtol=0, atol=0)
