# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared FP32 causal depthwise convolution with a raw-input residual."""

from dataclasses import dataclass, replace

import torch
import triton
import triton.language as tl
from b12x._lib.compile_plan import attach_programs, compile_only_launches, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.preparation.tuning import benchmark_resource, declare
from .native_operator import compile_triton, native_contract, native_plan
from .selected_attention import metadata_tensor, tensor_facts


@triton.jit
def _convolve(X, W, H, States, Starts, Lengths, Cu, Y,
              XS: tl.constexpr, WS: tl.constexpr, HS: tl.constexpr, HC: tl.constexpr,
              YS: tl.constexpr, C: tl.constexpr, K: tl.constexpr, T: tl.constexpr,
              BLOCK: tl.constexpr):
    token, batch, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n = tl.load(Lengths + batch)
    state = tl.load(States + batch)
    if token < n and state > 0:
        c = tile * BLOCK + tl.arange(0, BLOCK)
        begin, start = tl.load(Cu + batch), tl.load(Starts + batch)
        raw = tl.load(X + (begin + token) * XS + c, c < C, 0).to(tl.float32)
        result = tl.full((BLOCK,), 0, tl.float32)
        for i in tl.static_range(K):
            offset = token + i - (K - 1)
            current = tl.load(X + (begin + offset) * XS + c,
                              (c < C) & (offset >= 0), 0).to(tl.float32)
            previous = tl.load(H + state * HS + c * HC + offset + K - 1,
                               (c < C) & (offset < 0) & (start + offset >= 0), 0)
            value = tl.where(offset >= 0, current, previous)
            weight = tl.load(W + c * WS + i, c < C, 0)
            result = result + value * weight
        tl.store(Y + (begin + token) * YS + c, result + raw, c < C)


@triton.jit
def _commit(X, W, H, States, Starts, Lengths, Cu, Y,
            XS: tl.constexpr, WS: tl.constexpr, HS: tl.constexpr, HC: tl.constexpr,
            YS: tl.constexpr, C: tl.constexpr, K: tl.constexpr, T: tl.constexpr,
            BLOCK: tl.constexpr):
    batch, tile = tl.program_id(0), tl.program_id(1)
    n, state = tl.load(Lengths + batch), tl.load(States + batch)
    if n > 0 and state > 0:
        c = tile * BLOCK + tl.arange(0, BLOCK)
        history = tl.arange(0, triton.next_power_of_2(K - 1))
        offset = n - (K - 1) + history
        begin, start = tl.load(Cu + batch), tl.load(Starts + batch)
        current = tl.load(X + (begin + offset[None, :]) * XS + c[:, None],
                          (c[:, None] < C) & (history[None, :] < K - 1) & (offset[None, :] >= 0), 0).to(tl.float32)
        previous = tl.load(H + state * HS + c[:, None] * HC + offset[None, :] + K - 1,
                           (c[:, None] < C) & (history[None, :] < K - 1)
                           & (offset[None, :] < 0) & (start + offset[None, :] >= 0), 0)
        value = tl.where(offset[None, :] >= 0, current, previous)
        tl.store(H + state * HS + c[:, None] * HC + history[None, :], value,
                 (c[:, None] < C) & (history[None, :] < K - 1))


@dataclass(frozen=True)
class Query:
    operands: tuple


CONTRACT = native_contract(component_id="flare.residual_short_conv", query_type=Query,
                           tactics=lambda q: ((128, 4), (256, 4), (128, 8)))


def _arguments(tensors, block):
    x, w, h, states, starts, lengths, cu, y = tensors
    return (*tensors, x.stride(0), w.stride(0), h.stride(0), h.stride(1), y.stride(0),
            x.shape[1], w.shape[1], x.shape[0], block)


@program_cache(scope="preparation")
def compile_convolution(payload, tactic, ordinal):
    q = Query(**dict(payload))
    tensors = tuple(map(metadata_tensor, q.operands))
    block, warps = tactic
    x, _, _, states, *_ = tensors
    with torch.cuda.device(ordinal), compile_only_launches():
        args = _arguments(tensors, block)
        return dict(
            convolve=compile_triton(_convolve, (x.shape[0], states.numel(), triton.cdiv(x.shape[1], block)),
                *args, num_warps=warps, enable_fp_fusion=False),
            commit=compile_triton(_commit, (states.numel(), triton.cdiv(x.shape[1], block)),
                *args, num_warps=warps, enable_fp_fusion=False))


@dataclass(frozen=True)
class State:
    query: Query
    tactic: tuple
    programs: dict

    def run(self, tensors):
        x, _, _, states, _, _, cu, y = tensors
        block, _ = self.tactic
        y.zero_()
        args = _arguments(tensors, block)
        self.programs["convolve"][(x.shape[0], states.numel(), triton.cdiv(x.shape[1], block))](*args)
        self.programs["commit"][(states.numel(), triton.cdiv(x.shape[1], block), 1)](*args)


@torch.library.custom_op("flare::residual_short_conv", mutates_args=("history", "output"))
def _execute(source: torch.Tensor, weight: torch.Tensor, history: torch.Tensor,
             states: torch.Tensor, starts: torch.Tensor, lengths: torch.Tensor,
             cu: torch.Tensor, output: torch.Tensor, handle: int) -> None:
    plan = plan_from_handle(handle)
    require_prepared(plan, plan.component_id, source.device).run(
        (source, weight, history, states, starts, lengths, cu, output))


@_execute.register_fake
def _fake(source, weight, history, states, starts, lengths, cu, output, handle):
    pass


class PreparedResidualShortConvolution:
    """Commit chronological raw FP32 history after all packed outputs are computed."""

    def __init__(self, source, weight, history, metadata, output, *, session, workspace, name):
        m = metadata
        self.arguments = source, weight, history, m.state_rows, m.starts, m.lengths, m.query_start_loc, output
        if (source.ndim != 2 or source.dtype != torch.bfloat16 or source.stride(-1) != 1
                or min(source.shape) < 1 or weight.ndim != 2 or weight.shape[0] != source.shape[1]
                or not 2 <= weight.shape[1] <= 8 or weight.dtype != torch.float32 or weight.stride(-1) != 1
                or weight.stride(0) < weight.shape[1]
                or history.ndim != 3 or history.shape[1:] != (source.shape[1], weight.shape[1] - 1)
                or history.shape[0] < 2 or history.dtype != torch.float32 or history.stride(-1) != 1
                or history.stride(1) < history.shape[2]
                or history.stride(0) < (history.shape[1] - 1) * history.stride(1) + history.shape[2]
                or output.shape != source.shape or output.dtype != source.dtype or output.stride(-1) != 1
                or source.stride(0) < source.shape[1] or output.stride(0) < output.shape[1]
                or any(t.device != source.device for t in self.arguments) or source.device.type != "cuda"):
            raise ValueError("residual convolution requires BF16 matrices, FP32 depthwise weights and FP32 chronological history")
        from .window_staging import _overlap

        if any(_overlap(output, t) for t in self.arguments[:-1]) or _overlap(source, history):
            raise ValueError("residual convolution output, source and history must not overlap")
        rows = m.state_rows.numel()
        for tensor, dtype, count in ((m.state_rows, torch.int64, rows), (m.starts, torch.int64, rows),
                                    (m.lengths, torch.int32, rows), (m.query_start_loc, torch.int32, rows + 1)):
            if tensor.dtype != dtype or tensor.shape != (count,) or not tensor.is_contiguous():
                raise ValueError("residual convolution has incompatible packed metadata")
        q = Query(tuple(tensor_facts(t) for t in self.arguments))
        payload = CONTRACT.encode_query(q)

        def materialize(tactic):
            programs = compile_convolution(payload, tactic, source.device.index)
            load_programs(programs)
            return attach_programs(State(q, tactic, programs), *programs.values())

        self.plan = native_plan(contract=CONTRACT, query=q, device=source.device,
            scratch_bytes=lambda t: 0, materialize=materialize,
            compile_jobs=lambda t, d: (CompileJob.create(
                "ignition.components.residual_short_conv:compile_convolution", payload, t, d.ordinal),))

        def benchmark(state):
            def inputs():
                values = tuple(torch.empty_strided((2, *t.shape[1:]) if index == 2 else t.shape,
                                                   t.stride(), dtype=t.dtype, device=t.device)
                               if index != 1 else t for index, t in enumerate(self.arguments))
                x, _, h, states, starts, lengths, cu, _ = values
                x.normal_(std=.2)
                h.normal_(std=.2)
                states.zero_()
                states[0] = 1
                starts.fill_(weight.shape[1])
                lengths.zero_()
                lengths[0] = source.shape[0]
                cu.fill_(source.shape[0])
                cu[0] = 0
                return values, h.clone(), x.clone()
            values, history_seed, source_seed = benchmark_resource(session, ("residual_short_conv", name), inputs)
            return PreparedCall(run=lambda: state.run(values), output=values[-1],
                                produce=lambda: values[0].copy_(source_seed),
                                reset=lambda: values[2].copy_(history_seed), owners=(values, history_seed, source_seed))
        declare(session, (self.plan.request(name=name,
            prepare_call=lambda state: replace(benchmark(state), owners=()), benchmark_call=benchmark),))

    def __call__(self):
        _execute(*self.arguments, self.plan.handle)
        return self.arguments[-1]
