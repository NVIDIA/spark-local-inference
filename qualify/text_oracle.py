# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare full-vocabulary text logits on identical recorded token contexts.

The version-1 artifact contains prompt_tokens, teacher_tokens, logits, model_id,
and revision. teacher_tokens are the consumed continuation inputs, so S logit
steps require exactly S-1 teacher tokens. Historical canonical and Flare
decoder artifacts are accepted without changing their stored inputs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

QWEN_VOCAB_SIZE = 248320
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


@dataclass(frozen=True)
class LogitTrace:
    prompt_tokens: tuple[int, ...]
    teacher_tokens: tuple[int, ...]
    logits: tuple[torch.Tensor, ...]
    vocab_size: int
    source: Mapping[str, Any]
    model_id: str | None = None
    revision: str | None = None


def _token_ids(value, name: str, vocab_size: int) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.dtype != torch.int64 or value.ndim != 1:
            raise ValueError(f"{name} must be CPU int64 token IDs with shape [tokens]")
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a sequence of token IDs")
    if any(type(token) is not int or not 0 <= token < vocab_size for token in value):
        raise ValueError(f"{name} contains invalid or out-of-vocabulary token IDs")
    return tuple(value)


def _validate(trace: LogitTrace) -> None:
    if type(trace.vocab_size) is not int or trace.vocab_size < 10:
        raise ValueError("vocab_size must be an integer of at least 10")
    _token_ids(trace.prompt_tokens, "prompt_tokens", trace.vocab_size)
    _token_ids(trace.teacher_tokens, "teacher_tokens", trace.vocab_size)
    if not trace.prompt_tokens:
        raise ValueError("prompt_tokens must be nonempty")
    if not trace.logits:
        raise ValueError("logits must contain at least one step")
    if len(trace.teacher_tokens) != len(trace.logits) - 1:
        raise ValueError("S logit steps require exactly S-1 consumed teacher_tokens")
    for index, row in enumerate(trace.logits):
        if not isinstance(row, torch.Tensor):
            raise ValueError(f"logits[{index}] must be a tensor")
        if row.device.type != "cpu":
            raise ValueError(f"logits[{index}] must be on CPU")
        if row.dtype not in _FLOAT_DTYPES or row.layout != torch.strided:
            raise ValueError(f"logits[{index}] must use a supported floating strided dtype")
        if tuple(row.shape) != (trace.vocab_size,):
            raise ValueError(f"logits[{index}] must cover the full vocabulary [{trace.vocab_size}]")
        if not bool(torch.isfinite(row).all()) or not bool(torch.isfinite(row.float()).all()):
            raise ValueError(f"logits[{index}] must be finite in storage and FP32")
    for field in ("model_id", "revision"):
        value = getattr(trace, field)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"{field} must be a nonempty string when recorded")


def trace_from_artifact(
    artifact: Mapping[str, Any], *, vocab_size: int = QWEN_VOCAB_SIZE,
    source: Mapping[str, Any] | None = None,
) -> LogitTrace:
    """Validate a saved artifact; legacy generated tokens supply consumed inputs."""
    if type(vocab_size) is not int or vocab_size < 10:
        raise ValueError("vocab_size must be an integer of at least 10")
    if not isinstance(artifact, Mapping):
        raise ValueError("logit artifact must be a mapping")
    rows = artifact.get("logits")
    if isinstance(rows, torch.Tensor):
        if rows.ndim != 2:
            raise ValueError("stacked logits must have shape [steps,vocabulary]")
        rows = tuple(rows.unbind(0))
    if not isinstance(rows, (list, tuple)):
        raise ValueError("logits must be a sequence of full-vocabulary tensors")
    formats = [key for key in ("prompt_tokens", "input_ids", "prompt") if key in artifact]
    if len(formats) != 1:
        raise ValueError("artifact must declare exactly one recognized prompt representation")
    prompt_key = formats[0]
    prompt = artifact[prompt_key]
    if prompt_key == "prompt_tokens":
        if type(artifact.get("schema_version")) is not int or artifact["schema_version"] != 1:
            raise ValueError("prompt_tokens artifacts require schema_version=1")
        for key in ("model_id", "revision"):
            if not isinstance(artifact.get(key), str) or not artifact[key]:
                raise ValueError(f"version-1 artifact requires {key}")
        artifact_format = "teacher_forced_v1"
    elif prompt_key == "input_ids":
        if not isinstance(prompt, torch.Tensor) or prompt.ndim != 2 or prompt.shape[0] != 1:
            raise ValueError("canonical input_ids must have shape [1,tokens]")
        prompt = prompt[0]
        artifact_format = "canonical_decoder"
    else:
        artifact_format = "flare_decoder"
    if prompt_key == "prompt":
        if "teacher_tokens" in artifact:
            raise ValueError("legacy Flare artifact must not mix generated and teacher tokens")
        generated = _token_ids(artifact.get("tokens"), "tokens", vocab_size)
        if len(generated) != len(rows):
            raise ValueError("legacy Flare tokens count must equal logit steps")
        teachers = generated[:-1]
    else:
        teachers = _token_ids(artifact.get("teacher_tokens"), "teacher_tokens", vocab_size)
    trace = LogitTrace(
        _token_ids(prompt, "prompt_tokens", vocab_size), teachers, tuple(rows), vocab_size,
        {**(source or {}), "artifact_format": artifact_format},
        artifact.get("model_id"), artifact.get("revision"),
    )
    _validate(trace)
    return trace


def load_trace(path: str | Path, *, vocab_size: int = QWEN_VOCAB_SIZE) -> LogitTrace:
    """Load full-vocabulary CPU tensors with weights_only=True."""
    path = Path(path).resolve()
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    return trace_from_artifact(artifact, vocab_size=vocab_size, source={"path": str(path)})


def _prediction(row: torch.Tensor) -> dict[str, Any]:
    top_two = row.topk(2).values
    return {"argmax": int(row.argmax()), "top1_logit": float(top_two[0]),
            "top2_logit": float(top_two[1]),
            "top1_margin": float(top_two[0] - top_two[1])}


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    actual, reference = actual.float(), reference.float()
    difference = actual - reference
    reference_norm, difference_norm = reference.norm(), difference.norm()
    overlap = len(set(actual.topk(10).indices.tolist()) & set(reference.topk(10).indices.tolist()))
    actual_logp, reference_logp = actual.log_softmax(-1), reference.log_softmax(-1)
    result = {
        "relative_l2": float(difference_norm / reference_norm) if reference_norm else None,
        "reference_l2_norm": float(reference_norm), "difference_l2_norm": float(difference_norm),
        "max_absolute_error": float(difference.abs().max()),
        "top10_overlap_count": overlap, "top10_overlap_fraction": overlap / 10,
        "kl_reference_to_actual": float((reference_logp.exp() * (reference_logp - actual_logp)).sum()),
        "actual": _prediction(actual), "reference": _prediction(reference),
        "argmax_equal": bool(actual.argmax() == reference.argmax()),
    }
    try:
        json.dumps(result, allow_nan=False)
    except ValueError as error:
        raise ValueError("FP32 metric computation produced a nonfinite value") from error
    return result


def compare_traces(*, flare: LogitTrace, reference: LogitTrace, canonical: LogitTrace) -> dict[str, Any]:
    """Return measurements for matching contexts; no numerical gate is applied."""
    traces = {"flare": flare, "reference": reference, "canonical": canonical}
    for name, trace in traces.items():
        _validate(trace)
        if trace.vocab_size != canonical.vocab_size:
            raise ValueError(f"{name}: vocabulary size differs from canonical")
        if len(trace.logits) != len(canonical.logits):
            raise ValueError(f"{name}: logit step count differs from canonical")
        if trace.prompt_tokens != canonical.prompt_tokens:
            raise ValueError(f"{name}: prompt token IDs differ from canonical")
        if trace.teacher_tokens != canonical.teacher_tokens:
            raise ValueError(f"{name}: consumed teacher token IDs differ from canonical")
    for field in ("model_id", "revision"):
        identities = {getattr(trace, field) for trace in traces.values()} - {None}
        if len(identities) > 1:
            raise ValueError(f"recorded {field} differs across artifacts")
    pairs = {}
    for actual, reference in (("flare", "reference"), ("flare", "canonical"), ("reference", "canonical")):
        pairs[f"{actual}_vs_{reference}"] = {
            "actual": actual, "reference": reference,
            "steps": [{"step": step, **_metrics(a, r)} for step, (a, r) in enumerate(
                zip(traces[actual].logits, traces[reference].logits, strict=True))],
        }
    return {
        "schema_version": 1, "vocab_size": canonical.vocab_size,
        "prompt_tokens": list(canonical.prompt_tokens),
        "teacher_tokens": list(canonical.teacher_tokens),
        "context_lengths": [len(canonical.prompt_tokens) + step for step in range(len(canonical.logits))],
        "sources": {name: {**trace.source, "model_id": trace.model_id, "revision": trace.revision,
                            "logit_dtypes": [str(row.dtype) for row in trace.logits]}
                    for name, trace in traces.items()},
        "metrics": {
            "arithmetic": "Inputs converted to FP32; norms, top-k and log_softmax evaluated on CPU.",
            "relative_l2": "norm(actual-reference) / norm(reference); null when reference norm is zero",
            "kl_reference_to_actual": "sum(exp(log_softmax(reference)) * (log_softmax(reference)-log_softmax(actual)))",
            "top10_overlap": "Intersection count and count/10; torch.topk resolves boundary ties.",
            "top1_margin": "Largest logit minus second-largest logit; argmax uses torch.argmax.",
        },
        "numerical_acceptance": "not evaluated; no thresholds supplied",
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("flare", "reference", "canonical"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    result = compare_traces(**{name: load_trace(getattr(args, name))
                               for name in ("flare", "reference", "canonical")})
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
