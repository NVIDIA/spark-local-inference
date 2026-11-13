# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure Qwen decoder boundaries on identical teacher-forced contexts.

Layer tensors contain all 48 residual outputs, with shape [48,rows,10240].
The prompt step records every prompt row; each continuation step records one
consumed token. Measurements use the reference as the reference and apply no tolerance.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .text_oracle import QWEN_VOCAB_SIZE, LogitTrace, trace_from_artifact

LAYERS = 48
RESIDUAL_WIDTH = 10240
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_INTEGER_DTYPES = (torch.int32, torch.int64)
_DTYPE_METADATA = ("stage_storage_dtype", "stage_source_dtypes",
                   "layer_storage_dtype", "layer_source_dtypes")


@dataclass(frozen=True)
class LayerTrace:
    logits: LogitTrace
    layers: tuple[torch.Tensor, ...]
    stages: tuple[Mapping[str, torch.Tensor], ...]
    metadata: Mapping[str, Any]


def _tensor(value, name: str, *, allow_integer: bool = False) -> None:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU")
    allowed = _FLOAT_DTYPES + _INTEGER_DTYPES if allow_integer else _FLOAT_DTYPES
    if value.layout != torch.strided or value.dtype not in allowed:
        raise ValueError(f"{name} has an unsupported tensor dtype or layout")
    if value.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(value.float()).all()):
        raise ValueError(f"{name} must be finite in storage and FP32")
    if value.dtype in _INTEGER_DTYPES and not torch.equal(value, value.float().to(value.dtype)):
        raise ValueError(f"{name} integer values must be exactly representable in FP32")


def _layer_trace(artifact: Mapping[str, Any], logits: LogitTrace) -> LayerTrace:
    layers, stages = artifact.get("layer_outputs"), artifact.get("stage_outputs")
    steps = len(logits.logits)
    for name, entries in (("layer_outputs", layers), ("stage_outputs", stages)):
        if not isinstance(entries, (list, tuple)) or len(entries) != steps:
            raise ValueError(f"{name} must contain exactly {steps} logit steps")
    for step, (layer, stage) in enumerate(zip(layers, stages, strict=True)):
        rows = len(logits.prompt_tokens) if step == 0 else 1
        if not isinstance(layer, torch.Tensor) or tuple(layer.shape) != (LAYERS, rows, RESIDUAL_WIDTH):
            raise ValueError(f"layer_outputs[{step}] must have shape [{LAYERS},{rows},{RESIDUAL_WIDTH}]")
        for index in range(LAYERS):
            _tensor(layer[index], f"layer_outputs[{step}][{index}]")
        if not isinstance(stage, Mapping):
            raise ValueError(f"stage_outputs[{step}] must be a mapping")
        for key, value in stage.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"stage_outputs[{step}] keys must be nonempty strings")
            name = f"stage_outputs[{step}].{key}"
            _tensor(value, name, allow_integer=True)
            if value.ndim != 2 or value.shape[0] != rows:
                raise ValueError(f"{name} must have shape [{rows},width]")
    metadata = {key: artifact[key] for key in _DTYPE_METADATA if key in artifact}
    try:
        json.dumps(metadata, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("recorded source/bank dtype metadata must be JSON-compatible") from error
    return LayerTrace(logits, tuple(layers), tuple(dict(stage) for stage in stages), metadata)


def layer_trace_from_artifact(
    artifact: Mapping[str, Any], *, vocab_size: int = QWEN_VOCAB_SIZE,
) -> LayerTrace:
    """Validate in-memory tensors with the same logit/context parser as files."""
    return _layer_trace(artifact, trace_from_artifact(artifact, vocab_size=vocab_size))


def load_layer_trace(path: str | Path, *, vocab_size: int = QWEN_VOCAB_SIZE) -> LayerTrace:
    """Load logits and layers together and validate their shared contexts."""
    path = Path(path).resolve()
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    trace = trace_from_artifact(artifact, vocab_size=vocab_size, source={"path": str(path)})
    return _layer_trace(artifact, trace)


def _matching_contexts(actual: LogitTrace, reference: LogitTrace) -> None:
    if actual.vocab_size != reference.vocab_size:
        raise ValueError("vocabulary size differs between Flare and the reference")
    if len(actual.logits) != len(reference.logits):
        raise ValueError("logit step count differs between Flare and the reference")
    if actual.prompt_tokens != reference.prompt_tokens:
        raise ValueError("prompt token IDs differ between Flare and the reference")
    if actual.teacher_tokens != reference.teacher_tokens:
        raise ValueError("consumed teacher token IDs differ between Flare and the reference")
    for name in ("model_id", "revision"):
        identities = {getattr(trace, name) for trace in (actual, reference)} - {None}
        if len(identities) > 1:
            raise ValueError(f"recorded {name} differs between Flare and the reference")


def _measurement(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    if actual.shape != reference.shape:
        raise ValueError(f"tensor shapes differ: {tuple(actual.shape)} versus {tuple(reference.shape)}")
    if (actual.dtype in _INTEGER_DTYPES) != (reference.dtype in _INTEGER_DTYPES):
        raise ValueError("integer and floating stage representations cannot be compared")
    a, r = actual.float().contiguous(), reference.float().contiguous()
    difference = a - r
    row_ref = torch.linalg.vector_norm(r, dim=1)
    row_difference = torch.linalg.vector_norm(difference, dim=1)
    row_maximum = difference.abs().amax(dim=1)
    value_equal = (a.view(torch.int32) == r.view(torch.int32)).all(dim=1)
    storage_equal = None
    if actual.dtype == reference.dtype:
        storage_equal = (actual.contiguous().view(torch.uint8)
                         == reference.contiguous().view(torch.uint8)).all(dim=1)

    def metrics(reference_norm, difference_norm, maximum, equal, storage):
        return {
            "relative_l2": float(difference_norm / reference_norm) if reference_norm else None,
            "reference_l2_norm": float(reference_norm),
            "difference_l2_norm": float(difference_norm),
            "max_absolute_error": float(maximum),
            "fp32_value_bits_equal": bool(equal),
            "storage_bitwise_equal": None if storage is None else bool(storage),
        }

    rows = [{"row": row, **metrics(row_ref[row], row_difference[row], row_maximum[row],
             value_equal[row], None if storage_equal is None else storage_equal[row])}
            for row in range(actual.shape[0])]
    result = {
        "shape": list(actual.shape),
        "flare_storage_dtype": str(actual.dtype), "reference_storage_dtype": str(reference.dtype),
        **metrics(torch.linalg.vector_norm(r), torch.linalg.vector_norm(difference),
                  row_maximum.max(), value_equal.all(),
                  None if storage_equal is None else storage_equal.all()),
        "rows": rows, "last_row": dict(rows[-1]),
    }
    try:
        json.dumps(result, allow_nan=False)
    except ValueError as error:
        raise ValueError("FP32 metric computation produced a nonfinite value") from error
    return result


def compare_layers(*, flare: LayerTrace, reference: LayerTrace) -> dict[str, Any]:
    """Measure every layer and shared stage; report one-sided stages explicitly."""
    _matching_contexts(flare.logits, reference.logits)
    steps = []
    for step, (actual, expected, actual_stages, reference_stages) in enumerate(zip(
        flare.layers, reference.layers, flare.stages, reference.stages, strict=True
    )):
        layers = [{"layer": index, **_measurement(actual[index], expected[index])}
                  for index in range(LAYERS)]
        common = sorted(actual_stages.keys() & reference_stages.keys())
        stages = {}
        for name in common:
            try:
                stages[name] = _measurement(actual_stages[name], reference_stages[name])
            except ValueError as error:
                raise ValueError(f"step {step}, stage {name}: {error}") from error
        missing_in_flare = sorted(reference_stages.keys() - actual_stages.keys())
        missing_in_reference = sorted(actual_stages.keys() - reference_stages.keys())
        steps.append({
            "step": step, "context_length": len(reference.logits.prompt_tokens) + step,
            "input_token_ids": list(reference.logits.prompt_tokens if step == 0
                                    else reference.logits.teacher_tokens[step - 1:step]),
            "first_fp32_value_mismatch_layer": next(
                (item["layer"] for item in layers if not item["fp32_value_bits_equal"]), None),
            "first_last_row_fp32_value_mismatch_layer": next(
                (item["layer"] for item in layers if not item["last_row"]["fp32_value_bits_equal"]), None),
            "layers": layers,
            "stage_coverage": {
                "compared": common, "missing_in_flare": missing_in_flare,
                "missing_in_reference": missing_in_reference,
                "identical_key_sets": not missing_in_flare and not missing_in_reference,
                "has_comparable_stages": bool(common),
            },
            "stages": stages,
        })
    return {
        "schema_version": 1, "actual": "flare", "reference": "reference",
        "prompt_tokens": list(reference.logits.prompt_tokens),
        "teacher_tokens": list(reference.logits.teacher_tokens),
        "vocab_size": reference.logits.vocab_size, "layer_count": LAYERS,
        "residual_width": RESIDUAL_WIDTH,
        "sources": {name: {**trace.logits.source, "model_id": trace.logits.model_id,
                    "revision": trace.logits.revision, "recorded_dtype_metadata": dict(trace.metadata),
                    "layer_bank_dtypes": [str(value.dtype) for value in trace.layers],
                    "stage_bank_dtypes": [{key: str(value.dtype) for key, value in stage.items()}
                                          for stage in trace.stages]}
                    for name, trace in (("flare", flare), ("reference", reference))},
        "metrics": {
            "arithmetic": "All measurements use CPU FP32 tensors.",
            "relative_l2": "norm(flare-reference) / norm(reference); null when the reference norm is zero",
            "fp32_value_bits_equal": "Bitwise equality after FP32 conversion, including signed zero; independent of storage dtype.",
            "storage_bitwise_equal": "Original tensor bytes equal; null when storage dtypes differ.",
            "rows": "Every recorded row; last_row repeats the final prompt row or the single continuation row.",
            "first_mismatch": "First residual layer with unequal FP32 value bits; not a tolerance gate or causal attribution.",
        },
        "numerical_acceptance": "not evaluated; no thresholds supplied", "steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flare", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    result = compare_layers(flare=load_layer_trace(args.flare),
                            reference=load_layer_trace(args.reference))
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
