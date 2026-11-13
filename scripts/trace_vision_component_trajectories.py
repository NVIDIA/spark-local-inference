# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure component-family numerical propagation on exact checkpoint images.

Each isolated trajectory replaces only its named component family with prepared
b12x operations. Other components retain pinned eager Transformers and the
independent checkpoint quantization adapter. These diagnostic trajectories do
not replace the complete encoder's reference or its 3% relative-L2 gate.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from b12x.attention import varlen
from b12x.preparation import PreparationSession
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    apply_rotary_pos_emb_vision, eager_attention_forward,
)
from transformers.vision_utils import (
    get_vision_interpolation_indices_and_weights, get_vision_position_ids,
)

from ignition.preparation import inspect_qwen38_checkpoint
from ignition.recipes.qwen38.vision import PreparedVision, VisionMetadata, VisionScratch, VisionWeights
from ignition.recipes.qwen38.weights import QwenWeightLoader
from trace_vision_encoder import metrics, provenance, qualification_helpers, sha256


FAMILIES = {
    "canonical": frozenset(),
    "patch_only": frozenset({"patch"}),
    "qkv_only": frozenset({"qkv"}),
    "attention_only": frozenset({"attention"}),
    "output_projection_only": frozenset({"projection"}),
    "mlp_only": frozenset({"fc1", "fc2"}),
    "merger_only": frozenset({"merger"}),
    "native_all": frozenset({"patch", "qkv", "attention", "projection", "fc1", "fc2", "merger"}),
    "canonical_patch_native_all": frozenset({"qkv", "attention", "projection", "fc1", "fc2", "merger"}),
}


def block_forward(hidden, native, canonical, rope, scratch, families):
    normalized = canonical.norm1(hidden)
    projected = (native.attn.qkv(normalized) if "qkv" in families
                 else canonical.attn.qkv(normalized)).reshape(-1, 3, 16, 72)
    query, key = apply_rotary_pos_emb_vision(projected[:, 0], projected[:, 1], *rope)
    value = projected[:, 2]
    if "attention" in families:
        scratch.query.copy_(query)
        scratch.key.copy_(key)
        scratch.value.copy_(value)
        attention, _ = varlen.run(native.attn.binding)
    else:
        attention, _ = eager_attention_forward(
            canonical.attn, query.transpose(0, 1)[None], key.transpose(0, 1)[None],
            value.transpose(0, 1)[None], None, scaling=canonical.attn.scaling)
        attention = attention[0]
    attention = attention.flatten(1)
    projected = (native.attn.proj(attention) if "projection" in families
                 else canonical.attn.proj(attention))
    hidden = hidden + projected
    normalized = canonical.norm2(hidden)
    intermediate = (native.mlp.linear_fc1(normalized) if "fc1" in families
                    else canonical.mlp.linear_fc1(normalized))
    activated = canonical.mlp.act_fn(intermediate)
    projected = (native.mlp.linear_fc2(activated) if "fc2" in families
                 else canonical.mlp.linear_fc2(activated))
    return hidden + projected


@torch.no_grad()
def trace(snapshot, output):
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or torch.cuda.device_count() != 1:
        raise RuntimeError("assign exactly one CUDA device through CUDA_VISIBLE_DEVICES")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda", 0)
    helpers = qualification_helpers()
    manifest = inspect_qwen38_checkpoint(snapshot)
    module = helpers._module(manifest)
    with QwenWeightLoader(manifest, device, prefixes=("model.visual.",)) as loader:
        weights = VisionWeights.load(module, loader)
    reference = helpers._reference(manifest, weights, device)
    scratch = VisionScratch.allocate(256, device)
    metadata = VisionMetadata.allocate(256, 2, device)
    report = provenance(manifest) | {
        "families": {name: sorted(family) for name, family in FAMILIES.items()},
        "mlp_scope": "The 27 transformer-block MLPs; the spatial merger is measured separately.",
        "variants": [], "complete": False,
    }
    report["source_sha256"][str(Path(__file__).resolve())] = sha256(__file__)

    def save():
        output.write_text(json.dumps(report, indent=2) + "\n")

    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        prepared = PreparedVision(module, scratch, metadata, weights, session=session,
                                  attention_math_mode="eager_bf16")
        session.freeze()
        for variant in (0, 1):
            inputs = helpers._input(manifest, variant)
            grid, pixels = inputs.image_grid_thw, inputs.pixel_values.to(device)
            indices, interpolation = get_vision_interpolation_indices_and_weights(
                grid, num_grid_per_side=48, mode="bilinear", align_corners=True, spatial_merge_size=2)
            metadata.interpolation_indices.copy_(indices)
            metadata.interpolation_weights.copy_(interpolation)
            metadata.position_ids.copy_(get_vision_position_ids(grid, 2))
            metadata.cu_seqlens.copy_(torch.tensor([0, 256, 256], dtype=torch.int32))
            metadata.num_patches.fill_(256)
            scratch.patches.copy_(pixels)
            native_patch = prepared.patch_embed(scratch.patches).clone()
            canonical_patch = reference.patch_embed(pixels)
            positions = (module.pos_embed(metadata.interpolation_indices)
                         * metadata.interpolation_weights[:, :, None]).sum(1).bfloat16()
            canonical = canonical_patch + positions
            trajectories = {
                name: (native_patch if "patch" in family else canonical_patch) + positions
                for name, family in FAMILIES.items()
            }
            rope = module.rotary_pos_emb(canonical, metadata.position_ids)
            case = {"variant": variant, "patch": metrics(native_patch, canonical_patch), "blocks": []}
            report["variants"].append(case)
            for index, (native, expected) in enumerate(zip(module.blocks, reference.blocks)):
                canonical = expected(canonical, metadata.cu_seqlens, position_embeddings=rope)
                row = {"block": index, "trajectories": {}}
                for name, family in FAMILIES.items():
                    hidden = block_forward(trajectories[name], native, expected, rope, scratch, family)
                    trajectories[name] = hidden
                    row["trajectories"][name] = metrics(hidden, canonical)
                torch.testing.assert_close(trajectories["canonical"], canonical, atol=0, rtol=0)
                case["blocks"].append(row)
                print(json.dumps({"variant": variant, "block": index,
                                  "relative_l2": {name: value["relative_l2"]
                                                  for name, value in row["trajectories"].items()}}), flush=True)
                save()
            canonical_output = reference.merger(canonical)
            outputs = {
                name: (module.merger(hidden).clone() if "merger" in FAMILIES[name]
                       else reference.merger(hidden))
                for name, hidden in trajectories.items()
            }
            case["outputs"] = {name: metrics(result, canonical_output) for name, result in outputs.items()}
            canonical_full = reference(pixels, grid.to(device)).pooler_output
            torch.testing.assert_close(canonical_output, canonical_full, atol=0, rtol=0)
            actual_full = prepared(pixels).clone()
            case["prepared_vs_composed_native"] = metrics(actual_full, outputs["native_all"])
            torch.testing.assert_close(actual_full, outputs["native_all"], atol=0, rtol=0)
            case["passes_encoder_relative_l2_gate"] = case["outputs"]["native_all"]["relative_l2"] < .03
            print(json.dumps({"variant": variant, "outputs": case["outputs"]}), flush=True)
            save()
    report["complete"] = True
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace(args.snapshot, args.output)


if __name__ == "__main__":
    main()
