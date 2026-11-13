# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek target heads retain BF16 weights and FP32 logits under capture."""

from functools import lru_cache
from pathlib import Path
import os
import json

import pytest
import torch

from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.preparation.compilation import compile_forward
from b12x.preparation import PreparationSession

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1"
    or not os.environ.get("FLARE_TEST_DS41_CHECKPOINT"),
    reason="requires an explicitly assigned CUDA device and a DeepSeek V4.1 checkpoint",
)


@lru_cache(None)
def _checkpoint_shard():
    from safetensors import safe_open

    checkpoint = Path(os.environ["FLARE_TEST_DS41_CHECKPOINT"])
    if not checkpoint.is_dir():
        pytest.skip("requires the DeepSeek V4.1 source checkpoint")
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    with safe_open(
        checkpoint / index["weight_map"]["head.weight"], framework="pt", device="cpu"
    ) as shard:
        return shard.get_slice("head.weight")[:32320]


@pytest.mark.parametrize("rows", [1, 8, 32, 64])
@torch.no_grad()
def test_checkpoint_target_fp32_projection_matches_float_reference(rows):
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    weight = _checkpoint_shard().to(device)
    assert weight.dtype == torch.bfloat16
    original = weight[:128].clone()
    source = torch.empty((rows, 5120), device=device, dtype=torch.bfloat16)
    source.normal_(std=0.125)
    local = torch.empty((rows, 32320), device=device, dtype=torch.float32)
    output = torch.empty_like(local)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with PreparationSession(
            device=device, autotune=False, compile_workers=2
        ) as session:
            head = PreparedVocabularyProjection(
                source,
                weight,
                local,
                output,
                runtime=None,
                session=session,
                coordinator=None,
                name="target-fp32-head",
            )
            assert head.projection.weight is weight
            query = head.projection.plan.query
            assert query.source_dtype == query.weight_dtype == "bfloat16"
            assert query.output_dtype == "float32"
            call = compile_forward(lambda: head(source))
            call()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                call()
            session.freeze()
            try:
                generator = torch.Generator(device=device).manual_seed(56281)
                for _ in range(3):
                    source.normal_(generator=generator).mul_(0.125)
                    reference = torch.nn.functional.linear(
                        source.float(), weight.float()
                    )
                    expected = torch.mm(source, weight.T, out_dtype=torch.float32)
                    output.fill_(float("nan"))
                    local.fill_(float("nan"))
                    before = torch.cuda.memory_stats()
                    graph.replay()
                    torch.cuda.synchronize()
                    after = torch.cuda.memory_stats()
                    for key in (
                        "allocation.all.allocated",
                        "allocation.all.freed",
                        "segment.all.allocated",
                        "segment.all.freed",
                    ):
                        assert before[key] == after[key]
                    assert output.dtype == local.dtype == torch.float32
                    torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-5)
                    # FP32-input GEMM can use a different reduction order. Keep
                    # an independent normwise check in addition to the explicit
                    # BF16-input/FP32-output primitive's per-element contract.
                    relative_rms = (
                        (output - reference).double().square().mean()
                        / reference.double().square().mean()
                    ).sqrt()
                    assert relative_rms < 1e-5
                    assert (output != output.bfloat16().float()).any()
                    torch.testing.assert_close(output, local, rtol=0, atol=0)
                torch.testing.assert_close(weight[:128], original, rtol=0, atol=0)
            finally:
                graph.reset()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
