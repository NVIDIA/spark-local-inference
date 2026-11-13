# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Row-sharded and tensor-parallel attention output projections differ only by BF16 rounding.

A row-sharded prefill rank projects every output group with one prepared grouped projection, so the
group sum is accumulated in FP32 and rounded once. A tensor-parallel rank projects its two contiguous
groups into a BF16 partial, and the ranks reduce those partials. Both paths share the quantized
operands, so the remaining difference is the partial rounding plus the rounding of the reduction.
"""

import os

import pytest
import torch
from b12x.gemm import wo_projection
from b12x.gemm._shared.wo_mxfp8 import dequantize_mxfp8_rows_torch
from b12x.preparation import PreparationSession

from ignition.components.attention_output import PreparedGroupedOutputProjection

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")

# DeepSeek V4.1 Flash attention output geometry: eight output groups of 4096 columns over 64 heads of
# 512 dimensions, a 1024-wide group rank, and a 5120-wide hidden axis, over four ranks of two groups.
GROUPS, GROUP_WIDTH, RANK, HIDDEN, WORLD = 8, 4096, 1024, 5120, 4
GROUPS_PER_RANK = GROUPS // WORLD
BLOCK = 32
# Weight scale exponents in this band span 2^-12..2^-8, narrow enough that no block dominates the
# group sum while the four ranks still accumulate operands of different magnitudes.
SCALE_BAND = (115, 119)
REFERENCE_ROWS = 2048
HEADER = (f"{'rows':>6}  {'input':9} {'A!=B':5} {'ulpAB':5} {'|A-B|/rms':10} "
          f"{'A/exact diff':12} {'ulp':5} {'rms':7} {'B/exact diff':12} {'ulp':5} {'rms':7} "
          f"{'budget':6}  intermediate")
REPORTED = []


def _bf16_ulp(values):
    """BF16 spacing above the magnitude of each value, in FP32."""
    magnitude = values.abs().bfloat16()
    above = torch.nextafter(magnitude, torch.full_like(magnitude, float("inf")))
    return (above.float() - magnitude.float()).abs()


def _ulp_distance(result, reference, significant):
    """Largest BF16 ulp distance between two values, measured at their larger magnitude.

    A rank-order sum cancels to zero where four partials offset exactly, and the spacing at zero
    says nothing about the error, so each distance uses the wider of the two spacings.
    """
    spacing = torch.maximum(_bf16_ulp(result), _bf16_ulp(reference))
    return float(((result.float() - reference.float()).abs() / spacing)[significant].max())


def _weight_blocks(device, generator):
    """Random MXFP8 operands whose block scales come from the narrow exponent band."""
    def block(shape):
        values = (torch.randn(shape, device=device, generator=generator) * 16).to(torch.float8_e4m3fn)
        scales = torch.randint(SCALE_BAND[0], SCALE_BAND[1] + 1, (shape[0] // BLOCK, shape[1] // BLOCK),
            dtype=torch.uint8, device=device, generator=generator)
        return values, scales.view(torch.float8_e8m0fnu)
    return (*block((GROUPS * RANK, GROUP_WIDTH)), *block((HIDDEN, GROUPS * RANK)))


def _pack(wo_a, wo_a_scale, wo_b, wo_b_scale, groups):
    return wo_projection.pack_weights(wo_a, wo_a_scale, wo_b, wo_b_scale, groups=groups,
        group_width=GROUP_WIDTH, rank=RANK, hidden=HIDDEN, block_size=(BLOCK, BLOCK))


def _rank_pack(wo_a, wo_a_scale, wo_b, wo_b_scale, rank):
    """Pack one rank's contiguous output groups, the way the checkpoint partitions them."""
    blocks = slice(rank * GROUPS_PER_RANK * RANK // BLOCK, (rank + 1) * GROUPS_PER_RANK * RANK // BLOCK)
    columns = slice(rank * GROUPS_PER_RANK * RANK, (rank + 1) * GROUPS_PER_RANK * RANK)
    return _pack(wo_a[columns], wo_a_scale[blocks], wo_b[:, columns], wo_b_scale[:, blocks],
                 GROUPS_PER_RANK)


def _reduce(partials):
    """Accumulate BF16 partials in FP32 in rank order and round the sum once.

    The PCIe one-shot all-reduce keeps a float32 accumulator per element and adds the payloads in
    rank order starting at this rank before storing BF16 once, and the RoCE two-shot carries the
    same arithmetic, so rank zero's order ``[0, 1, 2, 3]`` is the reduction this models. The
    copy-engine ring that serves single-host prefill reductions folds its payloads one peer at a
    time in the payload dtype instead, which rounds three times per element, so the reduction
    modeled here is the most accurate tensor-parallel result the recipes produce.
    """
    total = partials[0].float()
    for partial in partials[1:]:
        total = total + partial.float()
    return total.bfloat16()


def _exact(source, plan, packed, device):
    """Dequantized FP64 result of the shared quantized operands, rounded once to BF16.

    The source quantization, the BF16 intermediate, and its MXFP8 quantization are the ones the
    prepared projection performs, so only the two GEMMs are evaluated in FP64. The projection keeps
    the intermediate as dense [groups, rows, rank] storage and its quantizer reads that layout.
    """
    rows = source.shape[0]
    x_q = wo_projection.quantize_input(source, plan=plan)
    a_deq = dequantize_mxfp8_rows_torch(packed.wo_a.values, packed.wo_a.scale_rows).double()
    intermediate = torch.empty((GROUPS, rows, RANK), dtype=torch.bfloat16, device=device)
    for start in range(0, rows, REFERENCE_ROWS):
        block = slice(start, min(start + REFERENCE_ROWS, rows))
        x_deq = dequantize_mxfp8_rows_torch(x_q.values[block].contiguous(),
                                            x_q.scale_rows[:, block].contiguous())
        intermediate[:, block] = torch.einsum("tdg,rdg->trg", x_deq.double(), a_deq
                                              ).bfloat16().permute(2, 0, 1)
        del x_deq
    b_q = wo_projection.quantize_input_b(intermediate.permute(1, 2, 0), plan=plan)
    del intermediate
    values = dequantize_mxfp8_rows_torch(b_q.values, b_q.scale_rows)
    del b_q
    b_deq = dequantize_mxfp8_rows_torch(packed.wo_b.values, packed.wo_b.scale_rows).double()
    exact = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device)
    for start in range(0, rows, REFERENCE_ROWS):
        block = slice(start, min(start + REFERENCE_ROWS, rows))
        exact[block] = (values[block].double() @ b_deq.T).bfloat16()
    return exact


def _row(rows, label, out_a, out_b, partials, exact, identical):
    """Print and return one case's measurements, its budget, and both RMS errors against exact.

    The ulp columns ignore elements within 1% of the output RMS, where a spacing carries no
    information about the group sum.
    """
    rms = float(out_a.float().square().mean().sqrt())
    difference = (out_a.float() - out_b.float()).abs()
    significant = out_a.abs() > 0.01 * rms
    # Every rounding in either path contributes at most half a BF16 ulp of its own result: one for
    # each BF16 partial, one for each path's output. The reduction adds four BF16 partials in FP32,
    # so its four additions contribute at most 4 * 2^-24 of their running magnitude, bounded here by
    # the sum of the partial magnitudes.
    budget = 0.5 * _bf16_ulp(out_a) + 0.5 * _bf16_ulp(out_b)
    extent = torch.zeros_like(out_a, dtype=torch.float32)
    for partial in partials:
        budget = budget + 0.5 * _bf16_ulp(partial)
        extent = extent + partial.float().abs()
    budget = budget + (WORLD * 2.0 ** -24) * extent
    del extent
    fields = [f"{rows:6d}  {label:9s} {float((out_a != out_b).float().mean()):5.3f}"
              f" {_ulp_distance(out_a, out_b, significant):5.1f}"
              f" {float(difference.max()) / rms:10.4f}"]
    errors = []
    for out in (out_a, out_b):
        errors.append(float((out.float() - exact.float()).square().mean().sqrt()))
        fields.append(f" {float((out != exact).float().mean()):12.4f}"
                      f" {_ulp_distance(out, exact, significant):5.1f}"
                      f" {errors[-1] / float(exact.float().square().mean().sqrt()):7.2e}")
    fields.append(f" {float((difference / budget).max()):6.2f}  {identical}")
    if not REPORTED:
        REPORTED.append(True)
        print(HEADER, flush=True)
    print("".join(fields), flush=True)
    return difference, budget, errors


# A row-sharded prefill projection sees at least 512 rows (ignition/recipes/deepseek41/decoder.py:72).
@pytest.mark.parametrize("rows", [128, 4096, 16384])
@pytest.mark.parametrize("gained", [False, True], ids=["unit", "lognormal"])
@torch.no_grad()
def test_row_shard_group_sum_matches_partitioned_reduce(rows, gained):
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(4127 + 2 * rows + int(gained))
    wo_a, wo_a_scale, wo_b, wo_b_scale = _weight_blocks(device, generator)
    full = _pack(wo_a, wo_a_scale, wo_b, wo_b_scale, GROUPS)
    partitioned = [_rank_pack(wo_a, wo_a_scale, wo_b, wo_b_scale, rank) for rank in range(WORLD)]
    source = torch.randn((rows, GROUPS, GROUP_WIDTH), dtype=torch.bfloat16, device=device,
                         generator=generator) * .125
    if gained:
        # A per-row lognormal gain gives tokens magnitudes that differ by orders of magnitude.
        gain = torch.randn(rows, device=device, generator=generator).exp().bfloat16()
        source = source * gain[:, None, None]
    shards = [source[:, rank * GROUPS_PER_RANK:(rank + 1) * GROUPS_PER_RANK].contiguous()
              for rank in range(WORLD)]
    outputs = [torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device) for _ in range(WORLD + 1)]
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        wholesale = PreparedGroupedOutputProjection(source, full, outputs[0],
            session=session, name=f"rows{rows}.groups{GROUPS}", workspace=None)
        ranked = [PreparedGroupedOutputProjection(shards[rank], partitioned[rank], outputs[rank + 1],
            session=session, name=f"rows{rows}.groups{GROUPS_PER_RANK}.rank{rank}", workspace=None)
            for rank in range(WORLD)]
        row_shard = wholesale(source).clone()
        partials = [operation(shards[rank]).clone() for rank, operation in enumerate(ranked)]
        whole_binding = wo_projection.bind(wholesale.plan, scratch=wholesale.scratch,
            source_tgd=source, weights=full, out=outputs[0])
        rank_bindings = [wo_projection.bind(operation.plan, scratch=operation.scratch,
            source_tgd=shards[rank], weights=partitioned[rank], out=outputs[rank + 1])
            for rank, operation in enumerate(ranked)]
        # The quantized intermediate of each rank's groups is the wholesale projection's slice.
        identical = []
        for rank, binding in enumerate(rank_bindings):
            columns = slice(rank * GROUPS_PER_RANK * RANK, (rank + 1) * GROUPS_PER_RANK * RANK)
            blocks = slice(columns.start // BLOCK, columns.stop // BLOCK)
            identical.append(torch.equal(
                whole_binding.tmp_q.values.contiguous().view(torch.uint8)[:, columns],
                binding.tmp_q.values.contiguous().view(torch.uint8)) and torch.equal(
                whole_binding.tmp_q.scale_rows.reshape(-1, GROUPS * RANK // BLOCK)[:, blocks],
                binding.tmp_q.scale_rows.reshape(-1, GROUPS_PER_RANK * RANK // BLOCK)))
        identical = "identical" if all(identical) else "differs"
        reduced = _reduce(partials)
        exact = _exact(source, wholesale.plan, full, device)
        difference, budget, errors = _row(rows, "lognormal" if gained else "unit", row_shard,
                                          reduced, partials, exact, identical)
        assert identical == "identical", f"rows={rows}: the quantized intermediate differs between paths"
        broken = difference > budget
        if bool(broken.any()):
            worst = int((difference / budget).flatten().argmax())
            row, column = divmod(worst, HIDDEN)
            pytest.fail(f"rows={rows} gained={gained}: {int(broken.sum())} of {broken.numel()} elements"
                        f" exceed the rounding budget, worst {float((difference / budget).max()):.3f}x at"
                        f" row {row} column {column}: |A-B|={float(difference[row, column]):.6f}"
                        f" budget={float(budget[row, column]):.6f}"
                        f" A={float(row_shard[row, column]):.6f}"
                        f" B={float(reduced[row, column]):.6f}"
                        f" partials={[float(partial[row, column]) for partial in partials]}")
        assert errors[0] <= errors[1], (f"rows={rows}: the group sum is less accurate than the"
                                        f" reduced partials ({errors[0]:.3e} > {errors[1]:.3e})")
