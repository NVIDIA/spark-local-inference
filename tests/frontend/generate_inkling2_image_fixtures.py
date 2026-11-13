# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Record exact Inkling 2 image tensors from the Transformers reference processor and vision tower.

Run in an environment that contains only the pinned reference packages, not a Flare
environment: Transformers at the PR #50682 head below, CPU PyTorch, and Pillow. Images
open with `PIL.Image.open` and go to `Inkling2ImageProcessor` without
`ImageOps.exif_transpose`, matching the Inkling 2 decoder of SGLang PR #50186. The
vision-tower tensors are the values a reduced-width `Inkling2VisionModel` passes to
its patch embedding, rotary embedding, and first attention layer; the width does not
affect them.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, features
from torchvision.transforms.v2 import functional as tvF
from transformers.models.inkling2 import modeling_inkling2
from transformers.models.inkling2.configuration_inkling2 import Inkling2VisionConfig
from transformers.models.inkling2.image_processing_inkling2 import Inkling2ImageProcessor

MODEL_ID = "thinkingmachines/Inkling-2-Small"
REVISION = "b0e7afb8892a07897c78ce1c5b801a0d9cbd5e7b"
TRANSFORMERS_COMMIT = "1b2f625956be1416816447eba32447a1d268d095"
TORCH_COMMIT = "cf30153c4c131c8164ee7798e5022d810682e2cb"
VERSIONS = {"torch": "2.13.0+cpu", "torchvision": "0.28.0+cpu", "Pillow": "12.3.0"}
ASSETS = {
    "processor_config.json": "b326f9a5d49c037c463045bec4d80229f5463048df7d6de95b597667888228ae",
    "config.json": "95bd7a6ad4f63011c068fcdc2243d7e2323b2e22329d34707687deccdf6f5a00",
}


def pattern(width: int, height: int, seed: int) -> np.ndarray:
    """Xorshift RGB bytes; the Rust tests regenerate the same sequence."""
    state = seed
    data = bytearray(width * height * 3)
    for index in range(len(data)):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        data[index] = state & 255
    return np.frombuffer(bytes(data), dtype=np.uint8).reshape(height, width, 3)


class _Stop(Exception):
    pass


def vision_inputs(model, pixel_values, grid):
    """Return the tensors the vision tower consumes before its first encoder layer."""
    recorded = {}
    original_interpolation = modeling_inkling2.get_vision_interpolation_indices_and_weights
    original_positions = modeling_inkling2.get_vision_position_ids

    def interpolation(*args, **kwargs):
        recorded["interpolation"] = original_interpolation(*args, **kwargs)
        return recorded["interpolation"]

    def positions(*args, **kwargs):
        recorded["native_positions"] = original_positions(*args, **kwargs)
        return recorded["native_positions"]

    def patches(module, args):
        recorded["native_patches"] = args[0].detach().clone()

    def rotary(module, args):
        recorded["rotary_positions"] = args[1].detach().clone()

    def layer(module, args, kwargs):
        recorded["cu_seqlens"] = kwargs["cu_seqlens"].detach().clone()
        raise _Stop

    handles = [model.embeddings.patch_embedding.register_forward_pre_hook(patches),
        model.rotary_emb.register_forward_pre_hook(rotary),
        model.layers[0].register_forward_pre_hook(layer, with_kwargs=True)]
    modeling_inkling2.get_vision_interpolation_indices_and_weights = interpolation
    modeling_inkling2.get_vision_position_ids = positions
    try:
        with torch.no_grad():
            model(pixel_values=pixel_values, image_grid_thw=grid)
    except _Stop:
        pass
    finally:
        modeling_inkling2.get_vision_interpolation_indices_and_weights = original_interpolation
        modeling_inkling2.get_vision_position_ids = original_positions
        for handle in handles:
            handle.remove()
    return recorded


def png_inputs(directory: Path):
    rng = np.random.default_rng(2054)
    rgb = rng.integers(0, 256, (47, 53, 3), dtype=np.uint8)
    alpha = rng.integers(0, 256, (47, 53), dtype=np.uint8)
    alpha[:5] = 0
    alpha[-5:] = 255
    images = {
        "RGB": Image.fromarray(rgb),
        "L": Image.fromarray(rgb[..., 0]),
        "RGBA": Image.fromarray(np.dstack([rgb, alpha])),
        "LA": Image.fromarray(np.dstack([rgb[..., 1], alpha]), mode="LA"),
        "P": Image.fromarray(rgb).quantize(colors=48),
    }
    options = {name: {} for name in images}
    images["P-tRNS"] = Image.fromarray(rgb).quantize(colors=48)
    options["P-tRNS"] = {"transparency": bytes((index * 37) % 256 for index in range(48))}
    keyed = rgb.copy()
    keyed[10:20, 7:31] = (17, 200, 99)
    images["RGB-tRNS"] = Image.fromarray(keyed)
    options["RGB-tRNS"] = {"transparency": (17, 200, 99)}
    gray = rgb[..., 2].copy()
    gray[30:40, 3:50] = 77
    images["L-tRNS"] = Image.fromarray(gray)
    options["L-tRNS"] = {"transparency": 77}
    exif = Image.Exif()
    exif[0x0112] = 6
    images["RGBA-orientation-6"] = images["RGBA"]
    options["RGBA-orientation-6"] = {"exif": exif}
    for name, image in images.items():
        path = directory / f"{name}.png"
        image.save(path, format="PNG", **options[name])
        yield f"png_{name}", path, "PNG", 0


def jpeg_inputs(directory: Path):
    rng = np.random.default_rng(2055)
    noise = Image.fromarray(rng.integers(0, 256, (29, 61, 3), dtype=np.uint8))
    exif = Image.Exif()
    exif[0x0112] = 6
    for name, image, options in [
        ("RGB-orientation-6", noise, {"exif": exif, "quality": 87, "subsampling": 2}),
        ("CMYK", noise.convert("CMYK"), {"quality": 87, "subsampling": 0}),
    ]:
        path = directory / f"{name}.jpg"
        image.save(path, format="JPEG", **options)
        yield f"jpeg_{name}", path, "JPEG", 2


def webp_inputs(directory: Path):
    rng = np.random.default_rng(2056)
    rgb = rng.integers(0, 256, (31, 43, 3), dtype=np.uint8)
    alpha = rng.integers(0, 256, (31, 43), dtype=np.uint8)
    alpha[:, :4] = 0
    for name, image in [("RGBA-lossless", Image.fromarray(np.dstack([rgb, alpha]))),
                        ("RGB-lossless", Image.fromarray(rgb))]:
        path = directory / f"{name}.webp"
        image.save(path, format="WEBP", lossless=True, exact=True)
        yield f"webp_{name}", path, "WEBP", 0


def generate(snapshot: Path, output: Path, binary_dir: Path | None):
    distribution = importlib.metadata.distribution("transformers")
    provenance = json.loads(distribution.read_text("direct_url.json") or "{}")
    if provenance.get("vcs_info", {}).get("commit_id") != TRANSFORMERS_COMMIT:
        raise RuntimeError("Transformers must be installed from the Inkling 2 PR head commit")
    versions = {name: importlib.metadata.version(name)
                for name in ("torch", "torchvision", "Pillow", "transformers", "numpy")}
    for name, expected in VERSIONS.items():
        if versions[name] != expected:
            raise RuntimeError(f"{name} must be {expected}, got {versions[name]}")
    if torch.version.git_version != TORCH_COMMIT:
        raise RuntimeError("PyTorch source commit mismatch")
    for name, expected in ASSETS.items():
        if hashlib.sha256((snapshot / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"checkpoint fingerprint mismatch: {name}")
    torch.set_num_threads(1)
    processor_config = json.loads((snapshot / "processor_config.json").read_text())
    model_config = json.loads((snapshot / "config.json").read_text())
    settings = dict(processor_config["image_processor"])
    if settings.pop("image_processor_type") != "Inkling2ImageProcessor":
        raise ValueError("checkpoint does not declare the Inkling 2 image processor")
    processor = Inkling2ImageProcessor(**settings)
    vision = model_config["vision_config"]
    config = Inkling2VisionConfig(patch_size=vision["native_patch_size"],
        spatial_merge_size=vision["adapter_merge_size"], posemb_grid_size=vision["posemb_grid_size"],
        hidden_size=32, num_hidden_layers=1, num_attention_heads=2, projection_hidden_dim=8,
        text_hidden_size=8)
    model = modeling_inkling2.Inkling2VisionModel(config).eval()

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

    sources = {}
    for name, width, height, seed in [("one_pixel", 1, 1, 1), ("one_cell", 40, 40, 2),
            ("cell_plus_one", 41, 41, 3), ("below_cell", 39, 17, 4), ("non_multiple", 53, 71, 5),
            ("landscape", 333, 217, 6), ("wide", 1000, 3, 7), ("tall", 2, 777, 8),
            ("one_row", 4096, 1, 9), ("model_card_maximum", 4096, 4096, 10)]:
        sources[name] = ({"pattern": {"width": width, "height": height, "seed": seed}},
                         Image.fromarray(pattern(width, height, seed)))
    for generator in (png_inputs, jpeg_inputs, webp_inputs):
        for name, path, fmt, tolerance in generator(inputs_dir):
            image = Image.open(io.BytesIO(path.read_bytes()))
            image.load()
            descriptor = {"file": str(path.relative_to(output.parent)), "format": fmt,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "width": image.width, "height": image.height, "mode": image.mode}
            if fmt == "JPEG":
                rgb = path.with_suffix(".rgb")
                rgb.write_bytes(np.asarray(image.convert("RGB")).tobytes())
                descriptor.update(rgb=str(rgb.relative_to(output.parent)), max_channel_error=tolerance)
            sources[name] = (descriptor, image)

    batches = [[name] for name in sources] + [["non_multiple", "one_pixel", "wide"],
        ["one_cell", "one_cell"], ["png_RGBA", "jpeg_CMYK", "webp_RGBA-lossless"]]
    cases = []
    for names in batches:
        name = "__".join(names)
        images = [sources[key][1] for key in names]
        result = processor(images=images, return_tensors="pt")
        grid = result["image_grid_thw"]
        recorded = vision_inputs(model, result["pixel_values"], grid)
        indices, weights = recorded["interpolation"]
        composited = []
        for index, image in enumerate(images):
            converted = image.convert("RGBA" if image.has_transparency_data else "RGB")
            rgb = processor.convert_to_rgb(tvF.pil_to_tensor(converted)).permute(1, 2, 0)
            composited.append(digest(f"{name}.composited{index}.u8", rgb))
        case = {"name": name, "inputs": [sources[key][0] for key in names],
            "grid_thw": grid.tolist(), "tokens": grid.prod(-1).tolist(),
            "cu_seqlens": recorded["cu_seqlens"].tolist(), "composited_rgb_sha256": composited,
            "hashes": {key: digest(f"{name}.{key}", value) for key, value in {
                "pixel_values": result["pixel_values"],
                "native_patches": recorded["native_patches"],
                "native_positions": recorded["native_positions"],
                "rotary_positions": recorded["rotary_positions"],
                "interpolation_indices": indices,
                "interpolation_weights": weights,
            }.items()}}
        cases.append(case)
        print(f"{name}: grid {case['grid_thw']}, {int(grid.prod(-1).sum()) * 4} patches", flush=True)
    result = {"model_id": MODEL_ID, "revision": REVISION, "transformers_commit": TRANSFORMERS_COMMIT,
        "torch_commit": TORCH_COMMIT, "versions": versions,
        "pillow_libjpeg_turbo": features.version_feature("libjpeg_turbo"),
        "pillow_webp": features.version_module("webp"), "assets": ASSETS,
        "image_processor_class": f"{type(processor).__module__}.{type(processor).__name__}",
        "processor": processor_config,
        "model": {"model_type": model_config["model_type"], "vision_config": vision},
        "cases": cases}
    output.write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, help="checkpoint directory with processor_config.json and config.json")
    parser.add_argument("output", type=Path)
    parser.add_argument("--binary-dir", type=Path, help="write every reference tensor for diagnosis")
    args = parser.parse_args()
    generate(args.snapshot, args.output, args.binary_dir)


if __name__ == "__main__":
    main()
