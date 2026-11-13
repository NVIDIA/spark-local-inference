# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Freeze Pillow JPEG/EXIF RGB and exact Qwen preprocessing fixtures."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps, features
from transformers import AutoProcessor


def digest(data):
    return hashlib.sha256(data).hexdigest()


def generate(snapshot, output):
    torch.set_num_threads(1)
    if Image.__version__ != "12.3.0":
        raise RuntimeError("fixture generation requires Pillow 12.3.0")
    assets = json.loads((output.parent / "qwen38_images.json").read_text())["assets"]
    for name, expected in assets.items():
        if digest((snapshot / name).read_bytes()) != expected:
            raise ValueError(f"checkpoint asset differs: {name}")
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True).image_processor
    directory = output.parent / "jpeg-exif-inputs"
    directory.mkdir(exist_ok=True)
    rng = np.random.default_rng(123)
    noise = Image.fromarray(rng.integers(0, 256, (29, 37, 3), dtype=np.uint8))
    specs = []
    for mode in ("RGB", "L", "CMYK"):
        for progressive in (False, True):
            for subsampling in (0, 1, 2) if mode == "RGB" else (0,):
                specs.append((f"{mode}-{int(progressive)}-{subsampling}", noise.convert(mode), "JPEG",
                    {"quality": 87, "subsampling": subsampling, "progressive": progressive}))
    for orientation in range(1, 9):
        exif = Image.Exif()
        exif[0x0112] = orientation
        for fmt in ("JPEG", "PNG"):
            specs.append((f"{fmt}-orientation-{orientation}", noise, fmt,
                {"exif": exif, **({"quality": 87, "subsampling": 2} if fmt == "JPEG" else {})}))
    cases = []
    for name, source, fmt, options in specs:
        path = directory / (name + (".jpg" if fmt == "JPEG" else ".png"))
        source.save(path, format=fmt, **options)
        with Image.open(path) as encoded:
            rgb = ImageOps.exif_transpose(encoded).convert("RGB")
        pixels = np.asarray(rgb)
        expected = path.with_suffix(".rgb")
        expected.write_bytes(pixels.tobytes())
        tensor = torch.from_numpy(pixels.copy()).permute(2, 0, 1).unsqueeze(0)
        resized = processor.resize(tensor, size=processor.size, factor=32, resample=3)
        prepared = processor(images=[rgb], return_tensors="pt")
        cases.append({"name": name, "format": fmt, "encoded": str(path.relative_to(output.parent)),
            "rgb": str(expected.relative_to(output.parent)), "width": rgb.width, "height": rgb.height,
            "max_channel_error": 2 if fmt == "JPEG" else 0,
            "grid_thw": prepared["image_grid_thw"][0].tolist(),
            "resized_rgb_sha256": digest(resized[0].permute(1, 2, 0).contiguous().numpy().tobytes()),
            "pixel_values_sha256": digest(prepared["pixel_values"].contiguous().numpy().tobytes())})
    output.write_text(json.dumps({"pillow": Image.__version__, "pillow_libjpeg_turbo": features.version_feature("libjpeg_turbo"),
        "assets": assets, "cases": cases}, indent=2) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    generate(args.snapshot, args.output)
