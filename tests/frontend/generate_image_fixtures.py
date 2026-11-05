# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record exact Qwen4Exp fast-processor tensors, pixels and positional metadata."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace, MethodType

import numpy as np
from PIL import Image
import torch
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpModel
from transformers.vision_utils import (
    get_vision_cu_seqlens,
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)

from ignition.preparation.qwen38 import FRONTEND_SHA256, MODEL_ID, REVISION, TRANSFORMERS_REVISION


TORCH_COMMIT = "cf30153c4c131c8164ee7798e5022d810682e2cb"
CONFIG_SHA256 = "b50dbc52a0c83b0cd2a44a372c1fb6b5e0e6827d22f2611c97d4d030f92f6b28"


def pattern(width: int, height: int, seed: int) -> np.ndarray:
    state = seed
    data = bytearray(width * height * 3)
    for index in range(len(data)):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        data[index] = state & 255
    return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)


def generate(snapshot: Path, output: Path, binary_dir: Path | None):
    distribution = importlib.metadata.distribution("transformers")
    provenance = json.loads(distribution.read_text("direct_url.json") or "{}")
    if provenance.get("vcs_info", {}).get("commit_id") != TRANSFORMERS_REVISION:
        raise RuntimeError("Transformers commit does not match checkpoint qualification pin")
    versions = {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "Pillow", "transformers", "numpy")}
    for name, expected in {"torch": "2.13.0+cu130", "torchvision": "0.28.0+cu130", "Pillow": "12.3.0"}.items():
        if versions[name] != expected:
            raise RuntimeError(f"{name} must be {expected}, got {versions[name]}")
    if torch.version.git_version != TORCH_COMMIT:
        raise RuntimeError("PyTorch source commit mismatch")
    torch.set_num_threads(1)
    assets = {**FRONTEND_SHA256, "config.json": CONFIG_SHA256}
    for name, expected in assets.items():
        if hashlib.sha256((snapshot / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"checkpoint fingerprint mismatch: {name}")
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    positional_model = SimpleNamespace(config=config)
    positional_model.get_vision_position_ids = MethodType(Qwen4ExpModel.get_vision_position_ids, positional_model)
    inputs_dir = output.parent / f"{output.stem}-inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    if binary_dir:
        binary_dir.mkdir(parents=True, exist_ok=True)

    def digest(name: str, array):
        if isinstance(array, torch.Tensor):
            array = array.cpu().contiguous().numpy()
        data = np.ascontiguousarray(array).tobytes()
        if binary_dir:
            (binary_dir / name).write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    specs = [("identity", 256, 256, 1), ("upsample", 53, 71, 2),
        ("round_up", 273, 289, 3), ("round_down", 271, 287, 4),
        ("half_even_lower", 272, 336, 5), ("half_even_upper", 304, 368, 6),
        ("width_identity", 256, 273, 7), ("height_identity", 273, 256, 8),
        ("minimum", 1, 1, 9), ("ratio_200", 200, 1, 10),
        ("portrait", 193, 417, 11), ("bucket_16384", 2033, 2033, 12)]
    inputs = {}
    for name, width, height, seed in specs:
        inputs[name] = ({"width": width, "height": height, "seed": seed}, Image.fromarray(pattern(width, height, seed)))
    raw = pattern(271, 283, 27)
    for mode in ("RGB", "RGBA", "L", "LA", "P"):
        if mode == "RGBA":
            pil = Image.fromarray(np.concatenate([raw, raw[..., :1]], axis=-1))
        elif mode == "LA":
            pil = Image.fromarray(np.stack([raw[..., 0], raw[..., 1]], axis=-1), mode="LA")
        elif mode == "L":
            pil = Image.fromarray(raw[..., 0])
        elif mode == "P":
            pil = Image.fromarray(raw).quantize(colors=32)
            pil.info["transparency"] = bytes(range(32))
        else:
            pil = Image.fromarray(raw)
        path = inputs_dir / f"{mode}.png"
        pil.save(path, format="PNG")
        inputs[f"png_{mode}"] = ({"png": str(path.relative_to(output.parent)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, Image.open(path))
    cases = []
    for names in [[name] for name in inputs] + [["portrait", "upsample", "identity"], ["identity", "identity"]]:
        name = "__".join(names)
        images = [inputs[key][1] for key in names]
        descriptors = [inputs[key][0] for key in names]
        messages = [{"role": "user", "content": [{"type": "text", "text": "Compare café 世界: "}] +
            [{"type": "image", "image": f"fixture://{key}"} for key in names]}]
        rendered = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        result = processor(text=rendered, images=images, return_tensors="pt", return_mm_token_type_ids=True)
        grids = result["image_grid_thw"]
        expanded = rendered
        for index in range(len(images)):
            expanded = expanded.replace("<|image_pad|>", f"<sg-image-{index}>", 1)
        for index in range(len(images)):
            expanded = expanded.replace(f"<sg-image-{index}>", processor.replace_image_token(result, index))
        indices, weights = get_vision_interpolation_indices_and_weights(grids, 48, mode="bilinear", align_corners=True, spatial_merge_size=2)
        positions = get_vision_position_ids(grids, 2)
        mrope, delta = Qwen4ExpModel.get_rope_index(positional_model, result["input_ids"], result["mm_token_type_ids"], image_grid_thw=grids)
        case = {"name": name, "inputs": descriptors, "messages": messages, "rendered": rendered,
            "expanded": expanded, "token_ids": result["input_ids"][0].tolist(),
            "mm_token_type_ids": result["mm_token_type_ids"][0].tolist(),
            "grid_thw": grids.tolist(), "cu_seqlens": get_vision_cu_seqlens(grids).tolist(),
            "rope_delta": delta[0, 0].item(), "resized_rgb_sha256": [], "input_rgb_sha256": []}
        for index, (pil, grid) in enumerate(zip(images, grids)):
            rgb = np.asarray(pil.convert("RGB"))
            resized = processor.image_processor.resize(torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0),
                size=processor.image_processor.size, factor=32, resample=3)
            assert tuple(resized.shape[-2:]) == (int(grid[1]) * 16, int(grid[2]) * 16)
            case["input_rgb_sha256"].append(digest(f"{name}.input{index}.u8", rgb))
            case["resized_rgb_sha256"].append(digest(f"{name}.resize{index}.u8", resized[0].permute(1, 2, 0)))
        case["hashes"] = {key: digest(f"{name}.{key}", value) for key, value in {
            "pixel_values": result["pixel_values"], "position_ids": positions,
            "interpolation_indices": indices, "interpolation_weights": weights,
            "mrope": mrope[:, 0, :],
        }.items()}
        cases.append(case)
        print(f"{name}: {list(result['pixel_values'].shape)}, exact reference recorded", flush=True)
    geometry_cases = []
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    for height, width in [(4096, 4096), (5000, 5000), (4097, 3071), (1, 201), (16, 3200), (2049, 2049)]:
        case = {"height": height, "width": width}
        try:
            case["resized"] = list(smart_resize(height, width, factor=32, min_pixels=65536, max_pixels=16777216))
        except ValueError as error:
            case["error"] = str(error)
        geometry_cases.append(case)
    result = {"model_id": MODEL_ID, "revision": REVISION, "transformers_commit": TRANSFORMERS_REVISION,
        "torch_commit": TORCH_COMMIT, "versions": versions, "assets": assets,
        "image_processor_class": f"{type(processor.image_processor).__module__}.{type(processor.image_processor).__name__}",
        "backend": processor.image_processor.backend, "cases": cases, "geometry_cases": geometry_cases}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--binary-dir", type=Path)
    args = parser.parse_args()
    generate(args.snapshot, args.output, args.binary_dir)


if __name__ == "__main__":
    main()
