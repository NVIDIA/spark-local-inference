# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Fused residual mixing against the checkpoint's attention residual and RMSNorm."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an assigned CUDA device")
CHECKPOINTS = (Path("/data/models/Kimi-K3-MXFP4-CSF"),
               Path("/data/models--moonshotai--Kimi-K3/snapshots/2496450e92e425c886db095102a52a6682ca3970"))


@torch.library.custom_op("flare_test::residual_branch", mutates_args=("out",))
def _branch(source: torch.Tensor, out: torch.Tensor) -> None:
    torch.tanh(source, out=out)


@_branch.register_fake
def _fake_branch(source, out):
    pass


def _canonical(checkpoint):
    path = checkpoint / "modeling_kimi_linear.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_apply_attn_res")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_apply_attn_res"]


def _assert_rounding_close(actual, expected):
    # FP32 reductions in another order may round a mixture to the neighbouring BF16 value.
    difference = (actual.float() - expected.float()).abs()
    ulp = expected.float().abs().clamp(min=2 ** -14) * torch.finfo(torch.bfloat16).eps
    assert bool((difference <= 4 * ulp).all()), float((difference / ulp).max())
    assert (actual != expected).float().mean().item() < 2e-3, (actual != expected).float().mean().item()


@torch.no_grad()
def test_checkpoint_residual_mixing_and_graph_replay():
    from ignition.components.weights import DirectWeightLoader
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.compilation import compile_forward
    from ignition.recipes.kimi3.model import rms_norm
    from ignition.recipes.kimi3.residual import ADD, REPLACE, ResidualStream, score_weight

    checkpoint = next((path for path in CHECKPOINTS if path.is_dir()), None)
    if checkpoint is None:
        pytest.skip("Kimi K3 checkpoint is unavailable")
    reference = _canonical(checkpoint)
    names = [f"language_model.model.layers.24.{field}.weight"
             for field in ("self_attention_res_norm", "self_attention_res_proj", "input_layernorm")]
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    with DirectWeightLoader(inspect_indexed_checkpoint(checkpoint), device, prefixes=names) as loader:
        norm, projection, output_norm = loader.load_tensors(names)
    eps = 1e-5
    canonical_norm = SimpleNamespace(weight=norm, variance_epsilon=eps)
    canonical_projection = SimpleNamespace(weight=projection)
    normalize = rms_norm(output_norm, eps)
    score = score_weight(norm, projection)
    generator = torch.Generator(device=device).manual_seed(6131)
    for rows in (1, 4, 4096):
        blocks = torch.randn((rows, 8, 7168), dtype=torch.bfloat16, device=device, generator=generator)
        prefix, delta = torch.randn((2, rows, 7168), dtype=torch.bfloat16, device=device, generator=generator)
        for sources, mode, write in ((0, REPLACE, 0), (1, ADD, -1), (1, REPLACE, -1), (4, ADD, 4), (8, ADD, -1)):
            stream = ResidualStream(prefix.clone(), blocks.clone(), torch.full_like(prefix, float("nan")))
            current = delta.clone() if mode == REPLACE else prefix + delta
            mixed = reference(current, blocks[:, :sources], canonical_projection, canonical_norm) if sources else current
            mixture = torch.empty_like(prefix)
            actual = stream.mix(delta, mode, sources=sources, write=write, score=score, norm=output_norm,
                                eps=eps, output_eps=eps, mixture=mixture)
            assert torch.equal(stream.prefix, current)
            if write >= 0:
                assert torch.equal(stream.blocks[:, write], current)
            _assert_rounding_close(actual, normalize(mixed))
            _assert_rounding_close(mixture, mixed)

    # A 93-layer residual sequence replays from caller-owned storage without allocation.
    rows = 4096
    embedded = torch.randn((rows, 7168), dtype=torch.bfloat16, device=device, generator=generator)
    stream = ResidualStream(torch.empty_like(embedded), torch.empty((rows, 8, 7168), dtype=torch.bfloat16,
                            device=device), torch.empty_like(embedded))
    branch = torch.empty_like(embedded)

    def forward():
        delta, mode = embedded, REPLACE
        for index in range(93):
            committed, boundary = -(-index // 12), index % 12 == 0
            hidden = stream.mix(delta, mode, sources=committed, write=index // 12 if boundary else -1,
                                score=score, norm=output_norm, eps=eps, output_eps=eps)
            _branch(hidden, branch)
            hidden = stream.mix(branch, REPLACE if boundary else ADD, sources=committed + boundary, write=-1,
                                score=score, norm=output_norm, eps=eps, output_eps=eps)
            _branch(hidden, branch)
            delta, mode = branch, ADD
        return stream.mix(delta, mode, sources=8, write=-1, score=score, norm=output_norm, eps=eps, output_eps=eps)

    def canonical():
        prefix, history = embedded, embedded[:, None][:, :0]
        for index in range(93):
            hidden = reference(prefix, history, canonical_projection, canonical_norm) if history.shape[1] else prefix
            if index % 12 == 0:
                history, prefix = torch.cat((history, prefix[:, None]), 1), None
            out = torch.tanh(normalize(hidden))
            prefix = out if prefix is None else prefix + out
            out = torch.tanh(normalize(reference(prefix, history, canonical_projection, canonical_norm)))
            prefix = prefix + out
        return normalize(reference(prefix, history, canonical_projection, canonical_norm))

    expected = canonical()
    compiled = compile_forward(forward)
    output = compiled()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    allocated = torch.cuda.memory_allocated()
    for _ in range(3):
        stream.output.fill_(float("nan"))
        stream.prefix.fill_(float("nan"))
        stream.blocks.fill_(float("nan"))
        graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated
    torch.testing.assert_close(stream.output, expected, atol=2e-2, rtol=2e-2)
