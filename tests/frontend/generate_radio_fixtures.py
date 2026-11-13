# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate C-RADIO preprocessing expectations from the checkpoint and PyTorch."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers.dynamic_module_utils import get_class_from_dynamic_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cls = get_class_from_dynamic_module(
        "image_processing_nemotron_h_omni.NemotronH_Omni_Reasoning_V3ImageProcessor", str(args.snapshot))
    processor = cls.from_pretrained(str(args.snapshot))
    geometry = []
    for width, height in ((1, 1), (16, 32), (37, 29), (512, 512), (513, 513),
                          (2048, 1024), (4096, 3072), (101, 1600), (1600, 101)):
        image = Image.new("RGB", (width, height))
        w, h = processor._compute_target_patches(image, processor.max_num_patches)
        geometry.append(dict(input=[width, height], grid=[h, w]))
    resize = []
    for width, height, out_w, out_h in ((7, 9, 10, 6), (19, 17, 4, 5), (5, 3, 5, 7), (5, 3, 9, 3)):
        pixels = ((np.arange(width * height * 3) * 37 + 11) % 256).astype(np.uint8)
        x = torch.from_numpy(pixels.reshape(height, width, 3)).permute(2, 0, 1)[None].float()
        y = torch.nn.functional.interpolate(x, (out_h, out_w), mode="bicubic", align_corners=False, antialias=True)
        resize.append(dict(input=[width, height], output=[out_w, out_h],
                           pixels=y[0].permute(1, 2, 0).flatten().tolist()))
    width, height = 37, 29
    pixels = ((np.arange(width * height * 3) * 37 + 11) % 256).astype(np.uint8)
    result = processor(images=Image.fromarray(pixels.reshape(height, width, 3)), return_tensors="pt")
    x = result["pixel_values"][0]
    c, h, w = x.shape
    packed = x.reshape(c, h // 16, 16, w // 16, 16).permute(1, 3, 0, 2, 4).flatten()
    indices = torch.linspace(0, packed.numel() - 1, 257).long()
    normalization = dict(input=[width, height], indices=indices.tolist(), pixels=packed[indices].tolist())
    args.output.write_text(json.dumps(dict(geometry=geometry, resize=resize, normalization=normalization)) + "\n")


if __name__ == "__main__":
    main()
