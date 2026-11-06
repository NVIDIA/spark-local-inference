# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paged causal attention with a finite learned bias for each backward distance."""

from dataclasses import dataclass, replace
import math

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
def _relative_attention(Q, K, V, Pages, Lengths, Cu, Bias, Out,
                        QT: tl.constexpr, QH: tl.constexpr, KP: tl.constexpr, KT: tl.constexpr,
                        KH: tl.constexpr, VP: tl.constexpr, VT: tl.constexpr, VH: tl.constexpr,
                        PT: tl.constexpr, BT: tl.constexpr, BH: tl.constexpr,
                        OT: tl.constexpr, OH: tl.constexpr, HEADS: tl.constexpr,
                        KV_HEADS: tl.constexpr, DIM: tl.constexpr, PAGE: tl.constexpr,
                        EXTENT: tl.constexpr, WINDOW: tl.constexpr, SCALE: tl.constexpr,
                        BM: tl.constexpr, BN: tl.constexpr):
    tile, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    begin, end = tl.load(Cu + batch), tl.load(Cu + batch + 1)
    length = tl.load(Lengths + batch)
    if tile * BM < end - begin:
        rows = tile * BM + tl.arange(0, BM)
        columns = tl.arange(0, BN)
        d = tl.arange(0, DIM)
        positions = length - (end - begin) + rows
        live = rows < end - begin
        q = tl.load(Q + (begin + rows[:, None]) * QT + head * QH + d[None, :], live[:, None], 0)
        kv_head = head // (HEADS // KV_HEADS)
        first = 0
        if WINDOW >= 0:
            first = tl.maximum(0, length - (end - begin) + tile * BM - WINDOW)
        stop = tl.minimum(length, length - (end - begin) + (tile + 1) * BM)
        maximum = tl.full((BM,), -1.0e30, tl.float32)
        denominator = tl.full((BM,), 0, tl.float32)
        # Pass one finds each row's max and denominator; pass two rounds normalized probabilities to BF16 before the V product.
        for start in range(first, stop, BN):
            keys = start + columns
            page = tl.load(Pages + batch * PT + keys // PAGE, keys < length, 0)
            k = tl.load(K + page[None, :] * KP + (keys[None, :] % PAGE) * KT
                        + kv_head * KH + d[:, None], keys[None, :] < length, 0)
            score = tl.dot(q, k).to(tl.bfloat16).to(tl.float32)
            score = (score * SCALE).to(tl.bfloat16).to(tl.float32)
            distance = positions[:, None] - keys[None, :]
            bias = tl.load(Bias + (begin + rows[:, None]) * BT + head * BH + distance,
                           live[:, None] & (distance >= 0) & (distance < EXTENT), 0).to(tl.float32)
            score = (score + bias).to(tl.bfloat16).to(tl.float32)
            valid = live[:, None] & (keys[None, :] < length) & (distance >= 0)
            if WINDOW >= 0:
                valid = valid & (distance <= WINDOW)
            score = tl.where(valid, score, float("-inf"))
            next_max = tl.maximum(maximum, tl.max(score, 1))
            denominator = denominator * tl.exp(maximum - next_max) + tl.sum(tl.exp(score - next_max[:, None]), 1)
            maximum = next_max
        accumulator = tl.full((BM, DIM), 0, tl.float32)
        for start in range(first, stop, BN):
            keys = start + columns
            page = tl.load(Pages + batch * PT + keys // PAGE, keys < length, 0)
            k = tl.load(K + page[None, :] * KP + (keys[None, :] % PAGE) * KT
                        + kv_head * KH + d[:, None], keys[None, :] < length, 0)
            score = tl.dot(q, k).to(tl.bfloat16).to(tl.float32)
            score = (score * SCALE).to(tl.bfloat16).to(tl.float32)
            distance = positions[:, None] - keys[None, :]
            bias = tl.load(Bias + (begin + rows[:, None]) * BT + head * BH + distance,
                           live[:, None] & (distance >= 0) & (distance < EXTENT), 0).to(tl.float32)
            score = (score + bias).to(tl.bfloat16).to(tl.float32)
            valid = live[:, None] & (keys[None, :] < length) & (distance >= 0)
            if WINDOW >= 0:
                valid = valid & (distance <= WINDOW)
            probability = tl.where(valid, tl.exp(score - maximum[:, None]) / denominator[:, None], 0).to(tl.bfloat16)
            value = tl.load(V + page[:, None] * VP + (keys[:, None] % PAGE) * VT
                            + kv_head * VH + d[None, :], keys[:, None] < length, 0)
            accumulator = tl.dot(probability, value, accumulator)
        tl.store(Out + (begin + rows[:, None]) * OT + head * OH + d[None, :], accumulator, live[:, None])


@dataclass(frozen=True)
class Query:
    operands: tuple
    window: int
    scale: float


CONTRACT = native_contract(component_id="flare.relative_paged_attention", query_type=Query,
    tactics=lambda q: ((16, 32, 4), (16, 64, 4), (32, 64, 4), (16, 64, 8)))


def _arguments(tensors, query, tactic):
    q, k, v, pages, lengths, cu, bias, out = tensors
    bm, bn, _ = tactic
    return (*tensors, *q.stride()[:2], *k.stride()[:3], *v.stride()[:3], pages.stride(0),
            *bias.stride()[:2], *out.stride()[:2], q.shape[1], k.shape[2], q.shape[2],
            k.shape[1], bias.shape[2], query.window, query.scale, bm, bn)


@program_cache(scope="preparation")
def compile_attention(payload, tactic, ordinal):
    query = Query(**dict(payload))
    tensors = tuple(map(metadata_tensor, query.operands))
    q, _, _, pages, *_ = tensors
    bm, _, warps = tactic
    with torch.cuda.device(ordinal), compile_only_launches():
        return compile_triton(_relative_attention, (triton.cdiv(q.shape[0], bm), q.shape[1], pages.shape[0]),
            *_arguments(tensors, query, tactic), num_warps=warps, enable_fp_fusion=False)


@dataclass(frozen=True)
class State:
    query: Query
    tactic: tuple
    program: object

    def run(self, tensors):
        q, _, _, pages, _, _, _, out = tensors
        out.zero_()
        self.program[(triton.cdiv(q.shape[0], self.tactic[0]), q.shape[1], pages.shape[0])](
            *_arguments(tensors, self.query, self.tactic))


@torch.library.custom_op("flare::relative_paged_attention", mutates_args=("output",))
def _execute(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
             pages: torch.Tensor, lengths: torch.Tensor, cu: torch.Tensor,
             bias: torch.Tensor, output: torch.Tensor, handle: int) -> None:
    plan = plan_from_handle(handle)
    require_prepared(plan, plan.component_id, query.device).run((query, key, value, pages, lengths, cu, bias, output))


@_execute.register_fake
def _fake(query, key, value, pages, lengths, cu, bias, output, handle):
    pass


class PreparedRelativePagedAttention:
    """Read BF16 pages and caller-projected relative logits without expanding KV."""

    def __init__(self, query, key_cache, value_cache, page_table, lengths, cu, relative_logits, output,
                 *, window=-1, softmax_scale, session, name):
        self.arguments = query, key_cache, value_cache, page_table, lengths, cu, relative_logits, output
        q, k, v, pages, lengths, cu, bias, out = self.arguments
        if (q.ndim != 3 or min(q.shape) < 1 or q.shape[2] not in (32, 64, 128, 256)
                or k.ndim != 4 or k.shape != v.shape or min(k.shape) < 1 or k.shape[3] != q.shape[2]
                or q.shape[1] % k.shape[2] or out.shape != q.shape
                or bias.ndim != 3 or bias.shape[:2] != q.shape[:2] or bias.shape[2] < 1
                or any(t.dtype != torch.bfloat16 or t.stride(-1) != 1 for t in (q, k, v, bias, out))
                or any(t.stride(1) != t.shape[2] or t.stride(0) < t.shape[1] * t.shape[2]
                       for t in (q, bias, out))
                or any(t.stride()[1:] != (t.shape[2] * t.shape[3], t.shape[3], 1)
                       or t.stride(0) < t.shape[1] * t.shape[2] * t.shape[3] for t in (k, v))
                or pages.ndim != 2 or min(pages.shape) < 1 or pages.dtype != torch.int32
                or lengths.shape != (pages.shape[0],) or lengths.dtype != torch.int32
                or cu.shape != (pages.shape[0] + 1,) or cu.dtype != torch.int32
                or any(not t.is_contiguous() for t in (pages, lengths, cu))
                or type(window) is not int or window < -1 or not math.isfinite(softmax_scale) or softmax_scale <= 0
                or any(t.device != q.device for t in self.arguments) or q.device.type != "cuda"):
            raise ValueError("relative paged attention requires matching BF16 GQA pages, per-query biases and packed int32 metadata")
        from .window_staging import _overlap

        if any(_overlap(output, t) for t in self.arguments[:-1]):
            raise ValueError("relative attention output must not overlap its inputs")
        geometry = Query(tuple(tensor_facts(t) for t in self.arguments), window, float(softmax_scale))
        payload = CONTRACT.encode_query(geometry)

        def materialize(tactic):
            program = compile_attention(payload, tactic, q.device.index)
            load_programs(program)
            return attach_programs(State(geometry, tactic, program), program)

        self.plan = native_plan(contract=CONTRACT, query=geometry, device=q.device,
            scratch_bytes=lambda t: 0, materialize=materialize,
            compile_jobs=lambda t, d: (CompileJob.create(
                "ignition.components.relative_paged_attention:compile_attention", payload, t, d.ordinal),))

        def benchmark(state):
            def inputs():
                values = []
                for index, tensor in enumerate(self.arguments):
                    shape = (1, *tensor.shape[1:]) if index in (1, 2) else tensor.shape
                    values.append(torch.empty_strided(shape, tensor.stride(), dtype=tensor.dtype, device=tensor.device))
                a, k, v, p, n, c, b, y = values
                for tensor in (a, k, v, b):
                    tensor.normal_(std=.2)
                p.zero_()
                n.zero_()
                n[0] = a.shape[0]
                c.fill_(a.shape[0])
                c[0] = 0
                return tuple(values), a.clone()
            values, seed = benchmark_resource(session, ("relative_paged_attention", name), inputs)
            return PreparedCall(run=lambda: state.run(values), output=values[-1],
                                produce=lambda: values[0].copy_(seed), owners=(values, seed))
        declare(session, (self.plan.request(name=name,
            prepare_call=lambda state: replace(benchmark(state), owners=()), benchmark_call=benchmark),))

    def __call__(self):
        _execute(*self.arguments, self.plan.handle)
        return self.arguments[-1]
