# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isolate vision attention rounding on exact checkpoint tensors, using only CPU.

The online-softmax calculation reproduces the stored precision boundaries and
tile order of b12x, not CUDA MMA reductions, approximate exponentials, or PTX
reciprocals. Its results diagnose arithmetic differences; they do not qualify a
CUDA kernel or replace the pinned eager-attention reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from transformers import AutoProcessor
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpVisionConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpVisionRotaryEmbedding, apply_rotary_pos_emb_vision, eager_attention_forward,
)
from transformers.vision_utils import (
    get_vision_interpolation_indices_and_weights, get_vision_position_ids,
)

from ignition.preparation import inspect_qwen38_checkpoint
from ignition.preparation.qwen38 import MODEL_ID, REVISION
from ignition.reference.qwen38_quant import (
    CheckpointTensors, decode_mxfp8, decode_nvfp4, quantize_mxfp8_activation,
    read_nvfp4_projection,
)


def metrics(actual, expected):
    actual, expected = actual.float(), expected.float()
    return {
        "relative_l2": float((actual - expected).norm() / expected.norm().clamp_min(1e-12)),
        "max_absolute": float((actual - expected).abs().max()),
        "changed_elements": int((actual != expected).sum()),
        "elements": actual.numel(),
    }


def attention(query, key, value, mode):
    scale = query.shape[-1] ** -.5
    if mode in ("eager", "rounded_scores_unnormalized_probabilities"):
        scores = ((query @ key.transpose(-2, -1)) * scale).float()
    else:
        scores = (query.float() @ key.float().transpose(-2, -1)) * scale
    if mode in ("eager", "unrounded_scores_normalized_probabilities"):
        probabilities = scores.softmax(-1).bfloat16()
        return probabilities @ value
    if mode == "rounded_scores_unnormalized_probabilities":
        exponents = (scores - scores.amax(-1, keepdim=True)).exp()
        result = exponents.bfloat16().float() @ value.float()
        return (result / exponents.sum(-1, keepdim=True)).bfloat16()
    if mode != "online_64_unrounded_scores":
        raise ValueError(mode)
    maximum = torch.full((*scores.shape[:-1], 1), -torch.inf)
    denominator = torch.zeros_like(maximum)
    result = torch.zeros_like(query, dtype=torch.float32)
    for start in reversed(range(0, scores.shape[-1], 64)):
        tile = scores[..., start:start + 64]
        next_maximum = torch.maximum(maximum, tile.amax(-1, keepdim=True))
        correction = (maximum - next_maximum).exp()
        exponents = (tile - next_maximum).exp()
        result = result * correction + exponents.bfloat16().float() @ value[..., start:start + 64, :].float()
        denominator = denominator * correction + exponents.sum(-1, keepdim=True)
        maximum = next_maximum
    return (result / denominator).bfloat16()


def linear(reader, name, *, mxfp8):
    prefix = "model.visual." + name
    bias = reader.read(prefix + ".bias")
    if mxfp8:
        weight = decode_mxfp8(reader.read(prefix + ".weight"), reader.read(prefix + ".weight_scale"))
        scale = 1.
    else:
        packed = read_nvfp4_projection(reader, prefix)
        weight = decode_nvfp4(packed.packed, packed.scales, torch.ones_like(packed.weight_scale_2)).bfloat16().float()
        scale = packed.weight_scale_2.item()

    def project(source):
        source = quantize_mxfp8_activation(source) if mxfp8 else source.float()
        return ((source @ weight.T) * scale).bfloat16() + bias

    return project


@torch.no_grad()
def diagnose(snapshot, *, blocks, variant):
    manifest = inspect_qwen38_checkpoint(snapshot)
    config = Qwen4ExpVisionConfig.from_dict(json.loads((manifest.snapshot / "config.json").read_text())["vision_config"])
    if not 1 <= blocks <= config.depth:
        raise ValueError(f"blocks must be in [1, {config.depth}]")
    pixels = bytes(component for y in range(256) for x in range(256)
                   for component in ((x * 7 + variant * 31) % 256, (y * 11) % 256, (x + y * 3) % 256))
    processor = AutoProcessor.from_pretrained(manifest.snapshot, local_files_only=True)
    inputs = processor.image_processor(images=[Image.frombytes("RGB", (256, 256), pixels)], return_tensors="pt")
    grid = inputs.image_grid_thw
    with CheckpointTensors(manifest, byte_budget=32 << 20) as reader:
        patch_weight = reader.read("model.visual.patch_embed.proj.weight")
        patch_bias = reader.read("model.visual.patch_embed.proj.bias")
        hidden = torch.nn.functional.conv3d(inputs.pixel_values.bfloat16().reshape(-1, 3, 2, 16, 16),
                                            patch_weight, patch_bias, stride=(2, 16, 16)).reshape(-1, 1152)
        indices, weights = get_vision_interpolation_indices_and_weights(
            grid, num_grid_per_side=48, mode="bilinear", align_corners=True, spatial_merge_size=2)
        positional = (reader.read("model.visual.pos_embed.weight")[indices] * weights[:, :, None]).sum(1)
        hidden = hidden + positional.bfloat16()
    rope = Qwen4ExpVisionRotaryEmbedding(config)(hidden, get_vision_position_ids(grid, 2))
    modes = ("eager", "unrounded_scores_normalized_probabilities",
             "rounded_scores_unnormalized_probabilities", "online_64_unrounded_scores")
    trajectories = {mode: hidden.clone() for mode in modes}
    rows = []
    canonical_checks = []
    for layer in range(blocks):
        with CheckpointTensors(manifest, byte_budget=32 << 20) as reader:
            qkv = linear(reader, f"blocks.{layer}.attn.qkv", mxfp8=True)
            proj = linear(reader, f"blocks.{layer}.attn.proj", mxfp8=True)
            fc1 = linear(reader, f"blocks.{layer}.mlp.linear_fc1", mxfp8=False)
            fc2 = linear(reader, f"blocks.{layer}.mlp.linear_fc2", mxfp8=False)
            norms = [(reader.read(f"model.visual.blocks.{layer}.norm{i}.weight"),
                      reader.read(f"model.visual.blocks.{layer}.norm{i}.bias")) for i in (1, 2)]

        def attend(source, mode):
            normalized = torch.nn.functional.layer_norm(source, (1152,), *norms[0], eps=1e-6)
            projected = qkv(normalized).reshape(-1, 3, 16, 72)
            query, key = apply_rotary_pos_emb_vision(projected[:, 0], projected[:, 1], *rope)
            value = projected[:, 2].transpose(0, 1)
            result = attention(query.transpose(0, 1), key.transpose(0, 1), value, mode)
            if mode == "eager":
                expected, _ = eager_attention_forward(
                    SimpleNamespace(num_key_value_groups=1, training=False),
                    query.transpose(0, 1)[None], key.transpose(0, 1)[None], value[None],
                    None, scaling=72 ** -.5)
                canonical_checks.append(torch.equal(result.transpose(0, 1), expected[0]))
                if not canonical_checks[-1]:
                    raise AssertionError("CPU eager attention differs from pinned canonical implementation")
            result = result.transpose(0, 1).reshape(-1, 1152)
            return result, proj(result)

        base = trajectories["eager"]
        canonical_raw, canonical_attended = attend(base, "eager")
        row = {"block": layer, "modes": {}}
        for mode in modes:
            raw, attended = attend(base, mode)
            same = {"raw_attention": metrics(raw, canonical_raw),
                    "projected_attention": metrics(attended, canonical_attended),
                    "projection_activation_qdq": metrics(quantize_mxfp8_activation(raw),
                                                          quantize_mxfp8_activation(canonical_raw))}
            _, attended = attend(trajectories[mode], mode)
            residual = trajectories[mode] + attended
            normalized = torch.nn.functional.layer_norm(residual, (1152,), *norms[1], eps=1e-6)
            trajectories[mode] = residual + fc2(torch.nn.functional.gelu(fc1(normalized), approximate="tanh"))
            row["modes"][mode] = {"same_input": same}
        for mode in modes:
            row["modes"][mode]["sequential_block_output"] = metrics(trajectories[mode], trajectories["eager"])
        rows.append(row)
    merger = None
    if blocks == config.depth:
        with CheckpointTensors(manifest, byte_budget=32 << 20) as reader:
            norm_weight = reader.read("model.visual.merger.norm.weight")
            norm_bias = reader.read("model.visual.merger.norm.bias")
            fc1 = linear(reader, "merger.linear_fc1", mxfp8=False)
            fc2 = linear(reader, "merger.linear_fc2", mxfp8=False)
        merged = {}
        for mode, hidden in trajectories.items():
            hidden = torch.nn.functional.layer_norm(hidden, (1152,), norm_weight, norm_bias, eps=1e-6)
            merged[mode] = fc2(torch.nn.functional.gelu(fc1(hidden.reshape(-1, 4608))))
        merger = {mode: metrics(result, merged["eager"]) for mode, result in merged.items()}
    return {
        "checkpoint": {"repository": MODEL_ID, "revision": REVISION},
        "device": "cpu", "variant": variant, "blocks": blocks,
        "arithmetic": "CPU FP32 matmul with explicit BF16 boundaries; online tile64 reverse order uses torch.exp and FP32 reductions, not CUDA kernel emulation",
        "reference": "Pinned eager attention precision boundaries; identical CPU checkpoint patch embedding in all trajectories",
        "input_sha256": hashlib.sha256(inputs.pixel_values.numpy().tobytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pinned_canonical_eager_bit_exact_checks": len(canonical_checks),
        "rows": rows,
        "merger": merger,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--variant", type=int, choices=(0, 1), default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    report = diagnose(args.snapshot, blocks=args.blocks, variant=args.variant)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["rows"][-1]))


if __name__ == "__main__":
    main()
