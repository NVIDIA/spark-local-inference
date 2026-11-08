# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Raced BF16 batched projections round like cuBLAS on every tactic and replay in captured graphs."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an explicitly assigned CUDA device")

SENTINEL = float("nan")


def stacks(rows, heads, major, capacity, device, generator):
    """GLM's token-major absorption (n-major weight) or expansion (k-major weight) stacks.

    Sources and outputs are the first ``rows`` tokens of ``capacity``-token buffers; the
    remaining tokens hold NaN sentinels that a correct launch leaves untouched.
    """
    width_in, width_out = (256, 512) if major == "n" else (512, 256)
    source = torch.randn((capacity, heads, width_in), generator=generator, device=device).bfloat16()
    output = torch.full((capacity, heads, width_out), SENTINEL, dtype=torch.bfloat16, device=device)
    weight = (torch.randn((heads, 256, 512), generator=generator, device=device) * 0.02).bfloat16()
    if major == "k":
        weight = weight.transpose(1, 2)
    return source[:rows].transpose(0, 1), weight, output[:rows].transpose(0, 1), source, output


def ulps(actual, expected):
    """Elementwise distance in BF16 representable values between two BF16 tensors."""
    def line(values):
        bits = values.view(torch.int16).to(torch.int32)
        return torch.where(bits < 0, -(bits & 0x7FFF), bits)
    return (line(actual) - line(expected)).abs()


@pytest.mark.parametrize("rows,heads,major", [(65, 32, "n"), (65, 32, "k"), (2, 16, "n"), (4096, 16, "k")])
@torch.no_grad()
def test_every_tactic_rounds_within_one_ulp_of_cublas_and_masks_the_row_tail(rows, heads, major):
    from ignition.components import bf16_bmm

    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device).manual_seed(rows + heads)
    source, weight, output, _, buffer = stacks(rows, heads, major, rows + 63, device, generator)
    query = bf16_bmm.query_from_call(source, weight, output)
    tactics = bf16_bmm.CONTRACT.parameters(query, None)["tactic"]
    assert tactics[0] == "cublas" and len(tactics) > 1
    expected = torch.bmm(source, weight)
    reference = torch.bmm(source.double(), weight.double())
    cublas_error = (expected.double() - reference).abs().max()
    for tactic in tactics:
        buffer.fill_(SENTINEL)
        bf16_bmm.prepared_state(query, tactic, device).run(source, weight, output)
        torch.cuda.synchronize(device)
        assert torch.isfinite(output).all(), tactic
        assert buffer[rows:].isnan().all(), tactic
        distance = ulps(output, expected)
        assert distance.max() <= 1, (tactic, distance.max().item())
        assert (distance > 0).float().mean() <= 1e-3, tactic
        assert (output.double() - reference).abs().max() <= cublas_error * 1.05 + 1e-12, tactic


@torch.no_grad()
def test_prepared_projections_race_and_replay_without_allocation():
    from ignition.components import bf16_bmm
    from ignition.components.linear import BatchedUnquantizedProjection
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.tuning import RecordedPreparationSession

    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device).manual_seed(16)
    rows, heads = 1024, 16
    absorb = stacks(rows, heads, "n", rows, device, generator)
    expand = stacks(rows, heads, "k", rows, device, generator)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with RecordedPreparationSession(device=device, checkpoint="bf16-bmm-test", workload_schema="bf16-bmm-v1",
                                    numerical_policy="default") as session:
        try:
            projections = [BatchedUnquantizedProjection(source, weight.transpose(1, 2) if major == "k" else weight,
                output, major=major, session=session, name=f"{major}-major")
                for (source, weight, output, _, _), major in zip((absorb, expand), ("n", "k"), strict=True)]
            session.tune()
            for projection, (source, weight, output, _, _) in zip(projections, (absorb, expand), strict=True):
                assert not projection.plan.memory_requirements().persistent
                selected = projection.plan.prepared.selection.config.tactic
                legal = bf16_bmm.CONTRACT.parameters(projection.plan.query, None)["tactic"]
                assert selected in legal
                print(f"{torch.cuda.get_device_name(device)} selected {selected} for {source.shape[1]} rows")

            def forward():
                for projection, (source, _, _, _, _) in zip(projections, (absorb, expand), strict=True):
                    projection(source)

            compiled = compile_forward(forward)
            compiled()
            torch.cuda.synchronize(device)
            with session.capture(), torch.cuda.graph(graph):
                compiled()
            graph.instantiate()
            session.freeze()
            for _ in range(3):
                for source, _, _, buffer, _ in (absorb, expand):
                    buffer.normal_(generator=generator)
                expected = [torch.bmm(source, weight) for source, weight, _, _, _ in (absorb, expand)]
                for _, _, _, _, buffer in (absorb, expand):
                    buffer.fill_(SENTINEL)
                torch.cuda.synchronize(device)
                before = torch.cuda.memory_stats(device)
                graph.replay()
                torch.cuda.synchronize(device)
                after = torch.cuda.memory_stats(device)
                for counter in ("allocation.all.allocated", "allocation.all.freed",
                                "segment.all.allocated", "segment.all.freed"):
                    assert before[counter] == after[counter]
                for (_, _, output, _, _), reference in zip((absorb, expand), expected, strict=True):
                    assert ulps(output, reference).max() <= 1
        finally:
            graph.reset()
