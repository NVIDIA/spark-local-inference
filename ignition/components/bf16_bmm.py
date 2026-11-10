# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared BF16 batched matrix multiplication over caller-owned strided stacks.

``output[g] = source[g] @ weight[g]`` for BF16 stacks ``source[G, M, K]``,
``weight[G, K, N]`` and ``output[G, M, N]`` with FP32 accumulation and one BF16
rounding per result. Operands keep the strides declared with the plan, so
token-major ``[rows, heads, width]`` activations multiply as transposed views
without staging copies, and the weight may be N- or K-contiguous.

Two implementations race through the preparation session: cuBLAS through
``torch.bmm`` and a Triton tiled kernel over the declared strides. cuBLAS
selects its kernel by device: at GLM's 8192-row head stacks it reaches about
half of GB10's bandwidth through 32x32 tiles, while RTX PRO 6000 selects wide
tensor-op tiles near its byte floor. The race keeps whichever is faster on the
preparing device.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from b12x._lib import compile_plan
from b12x._lib.compile_plan import attach_programs, compile_only_launches, launch_triton, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.preparation.fixtures.projection import input_factory
from ignition.preparation.tuning import declare

from .native_operator import native_contract, native_plan

COMPONENT_ID = "flare.bf16_bmm"

# Triton tiles as (rows, columns, depth, warps, stages). Row tiles below 64 serve decode
# stacks; 128-row tiles need the tensor-op throughput of discrete devices.
_TILES = ((64, 128, 64, 4, 3), (64, 256, 64, 8, 3), (128, 128, 64, 8, 3), (128, 256, 64, 8, 2),
          (32, 128, 64, 4, 3), (16, 128, 64, 4, 3), (16, 64, 64, 2, 3))


@dataclass(frozen=True)
class Query:
    # (shape, stride) of the source, weight and output stacks.
    operands: tuple


def _tactics(query):
    (_, rows, depth), _, (_, _, columns) = (shape for shape, _ in query.operands)
    tiles = tuple(("triton", *tile) for tile in _TILES
                  if columns % tile[1] == 0 and depth % tile[2] == 0
                  and tile[0] <= max(16, 2 * rows) and 4 * tile[0] >= min(rows, 128))
    return ("cublas", *tiles)


CONTRACT = native_contract(component_id=COMPONENT_ID, query_type=Query, tactics=_tactics)


@triton.jit
def _bmm_kernel(source, weight, output, M, N, K,
                stride_sg, stride_sm, stride_sk, stride_wg, stride_wk, stride_wn, stride_og, stride_om, stride_on,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # Consecutive programs share one source tile across the column tiles, so N / BLOCK_N
    # programs read each source row from L2 after the first. N and K are tile multiples.
    pid = tl.program_id(0)
    stack = tl.program_id(1)
    column_tiles = tl.cdiv(N, BLOCK_N)
    rows = (pid // column_tiles) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = (pid % column_tiles) * BLOCK_N + tl.arange(0, BLOCK_N)
    depth = tl.arange(0, BLOCK_K)
    source_tile = source + stack * stride_sg + rows[:, None] * stride_sm + depth[None, :] * stride_sk
    weight_tile = weight + stack * stride_wg + depth[:, None] * stride_wk + columns[None, :] * stride_wn
    live = rows[:, None] < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        acc = tl.dot(tl.load(source_tile, mask=live, other=0.0), tl.load(weight_tile), acc)
        source_tile += BLOCK_K * stride_sk
        weight_tile += BLOCK_K * stride_wk
    result = output + stack * stride_og + rows[:, None] * stride_om + columns[None, :] * stride_on
    tl.store(result, acc.to(output.dtype.element_ty), mask=live)


def _launch(tensors, tactic):
    """Positional kernel arguments, constexpr options and grid of one tile tactic."""
    source, weight, output = tensors
    _, block_m, block_n, block_k, warps, stages = tactic
    stacks, rows, depth = source.shape
    columns = output.shape[2]
    args = (source, weight, output, rows, columns, depth, *source.stride(), *weight.stride(), *output.stride())
    options = {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k}
    grid = (triton.cdiv(rows, block_m) * (columns // block_n), stacks, 1)
    return args, options, {"num_warps": warps, "num_stages": stages}, grid


@program_cache(scope="preparation")
def compile_bmm(payload, tactic, ordinal):
    query = Query(**dict(payload))
    tensors = tuple(torch.empty_strided(shape, stride, dtype=torch.bfloat16, device="meta")
                    for shape, stride in query.operands)
    args, options, launch, grid = _launch(tensors, tactic)
    # Compile planning leaves a deferred program in the kernel's Triton memo. b12x evicts those
    # entries only from kernels in its registry, so a compile worker that planned this program
    # would otherwise hand the deferred entry back instead of lowering it and never publish it.
    compile_plan._NATIVE_JITS.add(_bmm_kernel)
    with torch.cuda.device(ordinal), compile_only_launches():
        return launch_triton(_bmm_kernel, grid, *args, **options, **launch)


@dataclass(frozen=True)
class State:
    query: Query
    tactic: object
    program: object = None

    def run(self, source, weight, output):
        tensors = (source, weight, output)
        if any((tuple(t.shape), tuple(t.stride())) != declared
               for t, declared in zip(tensors, self.query.operands, strict=True)):
            raise ValueError("batched projection operands differ from the declared stacks")
        if self.program is None:
            torch.bmm(source, weight, out=output)
            return
        args, options, _, grid = _launch(tensors, self.tactic)
        self.program[grid](*args, *options.values())


@torch.library.custom_op("flare::bf16_bmm", mutates_args=("output",))
def execute(source: torch.Tensor, weight: torch.Tensor, output: torch.Tensor, handle: int) -> None:
    plan = plan_from_handle(handle)
    require_prepared(plan, COMPONENT_ID, source.device).run(source, weight, output)


@execute.register_fake
def _fake(source, weight, output, handle):
    pass


def query_from_call(source, weight, output):
    tensors = (source, weight, output)
    if (any(t.ndim != 3 or t.dtype != torch.bfloat16 or t.device != source.device for t in tensors)
            or weight.shape != (source.shape[0], source.shape[2], output.shape[2])
            or output.shape[:2] != source.shape[:2]):
        raise ValueError("batched BF16 projections require matching [G, M, K], [G, K, N] and [G, M, N] stacks")
    # Tiles vectorize along the contiguous axis, and compiled programs assume 16-byte aligned bases.
    if (source.stride(2) != 1 or output.stride(2) != 1 or 1 not in weight.stride()[1:]
            or any(t.data_ptr() % 16 for t in tensors)):
        raise ValueError("batched BF16 projections require 16-byte aligned stacks with contiguous "
                         "source and output rows and a K- or N-contiguous weight")
    return Query(tuple((tuple(t.shape), tuple(t.stride())) for t in tensors))


def prepared_state(query, tactic, device):
    """One tactic's prepared state, with its compiled program loaded."""
    if tactic == "cublas":
        return State(query, tactic)
    program = compile_bmm(CONTRACT.encode_query(query), tactic, device.index)
    load_programs(program)
    return attach_programs(State(query, tactic, program), program)


def declare_bmm(source, weight, output, *, session, name):
    """Declare one raced batched projection over the given stacks and return its plan."""
    query = query_from_call(source, weight, output)
    payload = CONTRACT.encode_query(query)
    device = source.device
    plan = native_plan(contract=CONTRACT, query=query, device=device, scratch_bytes=lambda t: 0,
        materialize=lambda tactic: prepared_state(query, tactic, device),
        compile_jobs=lambda tactic, detected: () if tactic == "cublas" else (
            CompileJob.create("ignition.components.bf16_bmm:compile_bmm", payload, tactic, detected.ordinal),))
    inputs = input_factory(source, output, storage=getattr(session, "benchmark_storage", None))

    def benchmark(state):
        operands = inputs()
        return PreparedCall(run=lambda: state.run(operands.source, weight, operands.output),
                            produce=operands.produce, output=operands.output, owners=(operands, weight))

    declare(session, (plan.request(name=name,
        prepare_call=lambda state: PreparedCall(run=lambda: state.run(source, weight, output)),
        benchmark_call=benchmark),))
    return plan
