# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trace exact-checkpoint vision arithmetic against pinned eager Transformers.

This diagnostic materializes 256-token attention scores to inspect rounding.
Serving continues to use the prepared linear-workspace attention kernel. The
reported probability reconstruction uses canonical scores and native row
statistics; it does not expose or assert equality of the kernel's QK products.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch
from b12x.attention import varlen
from b12x.preparation import PreparationSession
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
)
from transformers.vision_utils import (
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)

from ignition.preparation import inspect_qwen38_checkpoint
from ignition.preparation.qwen38 import MODEL_ID, REVISION
from ignition.recipes.qwen38.vision import (
    PreparedVision,
    VisionMetadata,
    VisionScratch,
    VisionWeights,
)
from ignition.recipes.qwen38.weights import QwenWeightLoader
from ignition.reference.qwen38_quant import quantize_mxfp8_activation

ROOT = Path(__file__).resolve().parents[1]


def metrics(actual, expected):
    actual, expected = actual.float(), expected.float()
    return {
        "relative_l2": float((actual - expected).norm() / expected.norm().clamp_min(1e-12)),
        "max_absolute": float((actual - expected).abs().max()),
        "reference_maximum": float(expected.abs().max()),
        "changed_elements": int((actual != expected).sum()),
        "elements": actual.numel(),
    }


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def qualification_helpers():
    path = ROOT / "tests/recipes/test_qwen38_vision.py"
    spec = importlib.util.spec_from_file_location("qwen38_vision_qualification", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def provenance(manifest):
    import b12x
    import transformers.models.qwen4_exp.modeling_qwen4_exp as canonical

    source_root = Path(b12x.__file__).parent
    files = [ROOT / name for name in (
        "scripts/trace_vision_encoder.py", "tests/recipes/test_qwen38_vision.py",
        "ignition/recipes/qwen38/vision.py", "ignition/components/linear.py",
        "ignition/recipes/qwen38/weights.py", "ignition/reference/qwen38_quant.py",
    )]
    files += list(source_root.rglob("*.py")) + [Path(canonical.__file__)]
    return {
        "checkpoint": {"repository": MODEL_ID, "declared_revision": REVISION,
                       "snapshot": str(manifest.snapshot),
                       "config_sha256": sha256(manifest.snapshot / "config.json"),
                       "index_sha256": sha256(manifest.snapshot / "model.safetensors.index.json")},
        "source_sha256": {str(path): sha256(path) for path in sorted(files)},
        "torch": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "command": sys.argv,
        "attention_mode": "eager_bf16",
        "reference": "Pinned eager Transformers with the independent exact-checkpoint quantization adapter from test_qwen38_vision.py",
        "score_reconstruction": "Canonical BF16 QK and scale; native maximum and reciprocal sum. Native QK products are not exported.",
        "qualification_limit_relative_l2": .03,
    }


def raw_attention(native, canonical, normalized, rope, scratch):
    qkv = canonical.qkv(normalized).reshape(-1, 3, 16, 72)
    q, k = apply_rotary_pos_emb_vision(qkv[:, 0], qkv[:, 1], *rope)
    v = qkv[:, 2]
    scratch.query.copy_(q)
    scratch.key.copy_(k)
    scratch.value.copy_(v)
    actual, lse = varlen.run(native.binding)
    actual, lse = actual.clone(), lse.clone()
    query, key, value = [tensor.transpose(0, 1)[None] for tensor in (q, k, v)]
    expected, probabilities = eager_attention_forward(
        canonical, query, key, value, None, scaling=canonical.scaling)
    expected = expected[0]
    scores = (query @ key.transpose(-2, -1)) * canonical.scaling
    maximum = scores.float().amax(-1)[0]
    exponents = (scores.float()[0] - maximum[..., None]).exp()
    inverse = exponents.sum(-1).reciprocal()
    binding = native.binding.binding
    native_maximum, native_inverse = binding.maximum.clone(), binding.inverse.clone()
    reconstructed = ((scores.float()[0] - native_maximum[..., None]).exp()
                     * native_inverse[..., None]).bfloat16()
    reconstructed_output = (reconstructed @ value[0]).transpose(0, 1)
    fp32_scores = ((query.float() @ key.float().transpose(-2, -1)).bfloat16()
                   * canonical.scaling)
    report = {
        "raw_output": metrics(actual, expected),
        "row_maximum": metrics(native_maximum, maximum),
        "reciprocal_sum": metrics(native_inverse, inverse),
        "logsumexp": metrics(lse, scores.float().logsumexp(-1)[0]),
        "fp32_qk_then_bf16_scores": metrics(fp32_scores, scores),
        "probabilities_reconstructed_with_native_statistics": metrics(reconstructed, probabilities[0]),
        "output_reconstructed_with_native_statistics": metrics(reconstructed_output, expected),
        "kernel_output_vs_statistics_reconstruction": metrics(actual, reconstructed_output),
        "activation_qdq": metrics(quantize_mxfp8_activation(actual.flatten(1)),
                                  quantize_mxfp8_activation(expected.flatten(1))),
    }
    return report, actual.flatten(1), expected.flatten(1), (q, k, v)


@torch.no_grad()
def trace(snapshot, output, operands):
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("assign exactly one CUDA device through CUDA_VISIBLE_DEVICES")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("the vision diagnostic requires exactly one visible CUDA device")
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
    report = provenance(manifest) | {"variants": [], "complete": False}
    saved_operands = {}

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
            weight = module.patch_embed.proj.weight.flatten(1)
            bias = module.patch_embed.proj.bias
            patch = {
                "native": metrics(native_patch, canonical_patch),
                "torch_linear_fused_bias": metrics(torch.nn.functional.linear(scratch.patches, weight, bias), canonical_patch),
                "torch_linear_split_bias": metrics(torch.nn.functional.linear(scratch.patches, weight) + bias, canonical_patch),
                "fp32_linear_split_bias": metrics(torch.nn.functional.linear(scratch.patches.float(), weight.float()).bfloat16() + bias, canonical_patch),
            }
            positions = (module.pos_embed(metadata.interpolation_indices)
                         * metadata.interpolation_weights[:, :, None]).sum(1).bfloat16()
            canonical = canonical_patch + positions
            trajectories = {"native_patch": native_patch + positions, "canonical_patch": canonical.clone()}
            rope = module.rotary_pos_emb(canonical, metadata.position_ids)
            case = {"variant": variant, "patch": patch, "blocks": []}
            report["variants"].append(case)
            for index, (native, expected) in enumerate(zip(module.blocks, reference.blocks)):
                normalized = expected.norm1(canonical)
                projected = expected.attn.qkv(normalized)
                row = {"block": index,
                       "qkv_same_input": metrics(native.attn.qkv(normalized).clone(), projected)}
                attention, raw, expected_raw, qkv = raw_attention(
                    native.attn, expected.attn, normalized, rope, scratch)
                row["attention_exact_qkv"] = attention
                projected_expected = expected.attn.proj(expected_raw)
                row["projection_same_input"] = metrics(native.attn.proj(expected_raw).clone(), projected_expected)
                row["projection_of_native_raw_reference_math"] = metrics(expected.attn.proj(raw), projected_expected)
                row["projection_of_native_raw_native_math"] = metrics(native.attn.proj(raw).clone(), projected_expected)
                attended = expected.attn(normalized, metadata.cu_seqlens, position_embeddings=rope)
                torch.testing.assert_close(projected_expected, attended, atol=0, rtol=0)
                row["attention_same_input"] = metrics(
                    native.attn(normalized, metadata.cu_seqlens, position_embeddings=rope).clone(), attended)
                residual = canonical + attended
                normalized_mlp = expected.norm2(residual)
                fc1 = expected.mlp.linear_fc1(normalized_mlp)
                row["fc1_same_input"] = metrics(native.mlp.linear_fc1(normalized_mlp).clone(), fc1)
                activated = expected.mlp.act_fn(fc1)
                fc2 = expected.mlp.linear_fc2(activated)
                row["fc2_same_input"] = metrics(native.mlp.linear_fc2(activated).clone(), fc2)
                next_canonical = expected(canonical, metadata.cu_seqlens, position_embeddings=rope)
                torch.testing.assert_close(residual + fc2, next_canonical, atol=0, rtol=0)
                row["block_same_input"] = metrics(
                    native(canonical, metadata.cu_seqlens, position_embeddings=rope).clone(), next_canonical)
                row["trajectories"] = {}
                for name, hidden in trajectories.items():
                    hidden = native(hidden, metadata.cu_seqlens, position_embeddings=rope).clone()
                    trajectories[name] = hidden
                    row["trajectories"][name] = metrics(hidden, next_canonical)
                if operands is not None and index in (0, 13, 26):
                    saved_operands[f"variant{variant}.block{index}"] = {
                        "query": qkv[0].cpu(), "key": qkv[1].cpu(), "value": qkv[2].cpu(),
                        "expected_raw_attention": expected_raw.cpu(), "native_raw_attention": raw.cpu(),
                        "canonical_hidden": canonical.cpu(),
                    }
                canonical = next_canonical
                case["blocks"].append(row)
                print(json.dumps({"variant": variant, "block": index,
                                  "raw_attention": attention["raw_output"]["relative_l2"],
                                  "trajectories": row["trajectories"]}), flush=True)
                save()
            merged = reference.merger(canonical)
            case["merger_same_input"] = metrics(module.merger(canonical).clone(), merged)
            final = {name: module.merger(hidden).clone() for name, hidden in trajectories.items()}
            case["outputs"] = {name: metrics(result, merged) for name, result in final.items()}
            canonical_full = reference(pixels, grid.to(device)).pooler_output
            torch.testing.assert_close(merged, canonical_full, atol=0, rtol=0)
            actual_full = prepared(pixels).clone()
            case["prepared_vs_staged_native"] = metrics(actual_full, final["native_patch"])
            case["encoder_reference"] = metrics(actual_full, canonical_full)
            case["passes_relative_l2_gate"] = case["encoder_reference"]["relative_l2"] < .03
            print(json.dumps({"variant": variant, "outputs": case["outputs"]}), flush=True)
            save()
    if operands is not None:
        torch.save(saved_operands, operands)
        report["operands"] = {"path": str(operands), "sha256": sha256(operands)}
    report["complete"] = True
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operands", type=Path)
    args = parser.parse_args()
    trace(args.snapshot, args.output, args.operands)


if __name__ == "__main__":
    main()
