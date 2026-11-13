# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared FlashInfer checkpointing SSU with caller-owned replay records."""

from dataclasses import dataclass, replace
from functools import cache

import torch
import triton
import triton.language as tl
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared
from b12x.sequence import mamba_ssd

from ignition.preparation.persistent import persistent_copy
from ignition.preparation.scratch import planned_scratch
from ignition.preparation.tuning import benchmark_resource, declare

from .mamba import _operands
from .native_operator import native_contract, native_plan


@dataclass(frozen=True)
class VerificationQuery:
    heads: int
    groups: int
    head_dim: int
    dstate: int
    sequences: int
    tokens: int
    width: int
    ring_length: int
    state_dtype: str
    weight_dtype: str
    index_dtype: str
    state_stride: int
    x_stride: int
    dt_stride: int
    b_stride: int
    c_stride: int
    sms: int


@cache
def _tactics(query):
    from flashinfer.mamba.checkpointing_ssu import _make_tactics

    return _make_tactics(query.heads // query.groups, query.sequences, query.heads, query.sms, (1, 2))


CONTRACT = native_contract(component_id="flare.mamba_verification", query_type=VerificationQuery, tactics=_tactics)


def _scratch_shapes(q):
    tokens = (q.width + 15) // 16 * 16
    old = (q.ring_length - q.width + 7) // 8 * 8
    return (((q.sequences, q.heads, 32, tokens // 2), torch.bfloat16),
            ((q.sequences, q.heads, tokens), torch.float32),
            ((q.sequences, q.heads, 32, old // 2), torch.bfloat16))


def _scratch_bytes(q):
    import math

    return sum((math.prod(shape) * dtype.itemsize + 255) // 256 * 256 for shape, dtype in _scratch_shapes(q))


def _scratch_views(scratch, q):
    import math

    values, offset = [], 0
    for shape, dtype in _scratch_shapes(q):
        nbytes = math.prod(shape) * dtype.itemsize
        values.append(scratch[offset:offset + nbytes].view(dtype).view(shape))
        offset += (nbytes + 255) // 256 * 256
    return values


@dataclass(frozen=True)
class VerificationState:
    module: object
    query: VerificationQuery
    tactic: tuple

    def run(self, operands, records, ring_start, pending_count, indices, output, scratch):
        x, dt, A, B, C, D, bias, state, cu, *_ = operands
        heads, dim, dstate = state.shape[1:]
        stages, ctas, precompute, split = self.tactic
        cb, adt, old = _scratch_views(scratch, self.query) if stages else (None, None, None)
        self.module.checkpointing_ssu(state, x.unsqueeze(0),
            dt[None, :, :, None].expand(1, -1, -1, dim), A[:, None, None].expand(heads, dim, dstate),
            B.unsqueeze(0), C.unsqueeze(0), output.unsqueeze(0), *records, ring_start, pending_count,
            D[:, None].expand(heads, dim), None, bias[:, None].expand(heads, dim), True,
            indices, -1, None, None, split, cu, cb, adt, old, precompute, stages, ctas)


@torch.library.custom_op("flare::mamba_verification", mutates_args=("state", "x_cache", "b_cache", "dt_cache", "out", "scratch"))
def _verification(x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                  D: torch.Tensor, bias: torch.Tensor, state: torch.Tensor, cu: torch.Tensor,
                  src: torch.Tensor, dst: torch.Tensor, fresh: torch.Tensor, ns: torch.Tensor, nt: torch.Tensor,
                  x_cache: torch.Tensor, b_cache: torch.Tensor, dt_cache: torch.Tensor,
                  ring_start: torch.Tensor, pending_count: torch.Tensor, indices: torch.Tensor,
                  out: torch.Tensor, scratch: torch.Tensor, handle: int) -> None:
    plan = plan_from_handle(handle)
    require_prepared(plan, plan.component_id, x.device).run(
        (x, dt, A, B, C, D, bias, state, cu, src, dst, fresh, ns, nt),
        (x_cache, b_cache, dt_cache), ring_start, pending_count, indices, out, scratch)


@_verification.register_fake
def _verification_fake(x, dt, A, B, C, D, bias, state, cu, src, dst, fresh, ns, nt,
                       x_cache, b_cache, dt_cache, ring_start, pending_count, indices, out, scratch, handle):
    pass


class PreparedMambaVerification:
    """Evaluate speculative tokens while retaining the accepted prefix in a replay ring.

    The caller updates ring metadata after acceptance and materializes pending
    records before entering ordinary decode or prefill. Empty packed sequences
    leave state and records unchanged.
    """

    def __init__(self, args, records, ring_start, pending_count, indices, *, width, session, workspace, name):
        self.output_shape, self.output_dtype = tuple(args["out"].shape), args["out"].dtype
        device = args["out"].device
        q = mamba_ssd.query_from_call(*_operands(args), out=args["out"])
        if q.max_tokens > q.max_seqs * width:
            raise ValueError("Mamba verification token capacity exceeds its per-sequence width")
        slots = args["recurrent_state"].shape[0]
        ring_length = records[0].shape[2]
        if not 1 < width <= 8 or not width < ring_length <= width + 16:
            raise ValueError("Mamba verification requires width 2..8 and a replay window of 1..16 tokens")
        expected = ((slots, q.heads, ring_length, q.head_dim),
                    (slots, q.groups, ring_length, q.dstate), (slots, q.heads, ring_length))
        for value, shape, dtype in zip(records, expected, (torch.bfloat16, torch.bfloat16, torch.float32), strict=True):
            if value.shape != shape or value.dtype != dtype or value.stride(-1) != 1:
                raise ValueError("Mamba verification record geometry or dtype differs from its recurrence")
        for metadata in (ring_start, pending_count):
            if metadata.shape != (slots,) or metadata.dtype != torch.int32 or not metadata.is_contiguous():
                raise ValueError("Mamba replay metadata requires contiguous int32 vectors indexed by state row")
        if indices.shape != (q.max_seqs,) or indices.dtype != args["initial_state_indices"].dtype:
            raise ValueError("Mamba verification indices must match the packed sequence capacity and index dtype")
        if args["dt"].dtype != args["D"].dtype or args["dt"].dtype != args["dt_bias"].dtype:
            raise ValueError("Mamba verification requires dt, D, and dt_bias to share their dtype")
        query = VerificationQuery(q.heads, q.groups, q.head_dim, q.dstate, q.max_seqs, q.max_tokens,
            width, ring_length, q.state_dtype, q.dt_dtype, q.index_dtype,
            q.state_stride, q.x_stride, q.dt_stride, q.b_stride, q.c_stride,
            torch.cuda.get_device_properties(device).multi_processor_count)
        self.query = query

        def materialize(tactic):
            from flashinfer.mamba.checkpointing_ssu import _get_module

            module = _get_module(getattr(torch, q.state_dtype), torch.bfloat16, getattr(torch, q.dt_dtype),
                getattr(torch, q.dt_dtype), torch.float32, getattr(torch, q.index_dtype), None,
                q.head_dim, q.dstate, width, ring_length - width, q.heads // q.groups, q.groups, 0, False)
            return VerificationState(module, query, tactic)

        self.plan = native_plan(contract=CONTRACT, query=query, device=device,
            scratch_bytes=lambda tactic: _scratch_bytes(query), materialize=materialize)

        def benchmark(state):
            fixture = benchmark_resource(session, ("mamba_verification", query), lambda: _Fixture(args, query))
            scratch = planned_scratch(self.plan.scratch_specs(), workspace)
            return PreparedCall(run=lambda: state.run(_operands(fixture.args), fixture.records,
                fixture.ring_start, fixture.pending_count, fixture.indices, fixture.args["out"], scratch),
                output=fixture.args["out"], produce=fixture.produce, reset=fixture.reset,
                restore=fixture.reset, owners=(fixture, scratch))

        declare(session, (self.plan.request(name=name,
            prepare_call=lambda state: replace(benchmark(state), owners=()), benchmark_call=benchmark),))

    def __call__(self, operands, records, ring_start, pending_count, indices):
        output = torch.empty(self.output_shape, dtype=self.output_dtype, device=operands["x"].device)
        scratch = torch.empty(_scratch_bytes(self.query), dtype=torch.uint8, device=operands["x"].device)
        _verification(*_operands(operands), *records, ring_start, pending_count, indices,
                      output, scratch, self.plan.handle)
        return output


class _Fixture:
    def __init__(self, args, query):
        q, device = query, args["x"].device
        self.args = dict(args)
        for name in ("x", "dt", "B", "C", "out"):
            value = args[name]
            self.args[name] = torch.empty_strided(value.shape, value.stride(), dtype=value.dtype, device=device)
            self.args[name].normal_(std=.1)
        self.seed = self.args["x"].clone()
        slots = q.sequences + 1
        self.args["recurrent_state"] = torch.zeros((slots, q.heads, q.head_dim, q.dstate),
            dtype=getattr(torch, q.state_dtype), device=device)
        self.args["cu_seqlens"] = torch.arange(q.sequences + 1, device=device, dtype=torch.int32) * q.tokens // q.sequences
        self.records = (torch.randn((slots, q.heads, q.ring_length, q.head_dim), device=device, dtype=torch.bfloat16) * .1,
            torch.randn((slots, q.groups, q.ring_length, q.dstate), device=device, dtype=torch.bfloat16) * .1,
            torch.full((slots, q.heads, q.ring_length), .01, device=device))
        self.ring_start = torch.zeros(slots, dtype=torch.int32, device=device)
        self.pending_count = torch.zeros_like(self.ring_start)
        self.indices = torch.arange(1, slots, dtype=getattr(torch, q.index_dtype), device=device)
        self.reset()

    def produce(self):
        self.args["x"].copy_(self.seed)

    def reset(self):
        self.args["recurrent_state"].zero_()
        self.pending_count.fill_(self.records[0].shape[2] // 2)
        self.produce()


@triton.jit
def _materialize(Backing, Offsets, A, Rows, Lengths, ROW_BYTES: tl.constexpr,
                 RING_OFFSET: tl.constexpr, COUNT_OFFSET: tl.constexpr, LIMIT: tl.constexpr,
                 HEADS: tl.constexpr, GROUPS: tl.constexpr, DIM: tl.constexpr, N: tl.constexpr,
                 RING: tl.constexpr, BLOCK: tl.constexpr):
    layer, row, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    slot = tl.load(Rows + row).to(tl.int64)
    if (slot > 0) & (tl.load(Lengths + row) > 0):
        base = Backing + slot * ROW_BYTES
        count = tl.load((base + COUNT_OFFSET).to(tl.pointer_type(tl.int32)))
        if count > LIMIT:
            start = tl.load((base + RING_OFFSET).to(tl.pointer_type(tl.int32)))
            state = (base + tl.load(Offsets + layer * 4)).to(tl.pointer_type(tl.bfloat16))
            x = (base + tl.load(Offsets + layer * 4 + 1)).to(tl.pointer_type(tl.bfloat16))
            b = (base + tl.load(Offsets + layer * 4 + 2)).to(tl.pointer_type(tl.bfloat16))
            dt = (base + tl.load(Offsets + layer * 4 + 3)).to(tl.pointer_type(tl.float32))
            for part in range(tile, tl.cdiv(HEADS * DIM * N, BLOCK), tl.num_programs(2)):
                index = part * BLOCK + tl.arange(0, BLOCK)
                mask = index < HEADS * DIM * N
                head, d, n = index // (DIM * N), index // N % DIM, index % N
                group = head // (HEADS // GROUPS)
                value = tl.load(state + index, mask, 0).to(tl.float32)
                a = tl.load(A + layer * HEADS + head, mask, 0)
                for t in range(count):
                    position = (start + t) % RING
                    step = tl.load(dt + head * RING + position, mask, 0)
                    xv = tl.load(x + (head * RING + position) * DIM + d, mask, 0).to(tl.float32)
                    bv = tl.load(b + (group * RING + position) * N + n, mask, 0).to(tl.float32)
                    value = value * tl.exp(step * a) + (step * bv) * xv
                tl.store(state + index, value, mask)


@triton.jit
def _replay_metadata(Backing, Rows, Lengths, Starts, Pending, Accepted,
                     ROW_BYTES: tl.constexpr, RING_OFFSET: tl.constexpr, COUNT_OFFSET: tl.constexpr,
                     RING: tl.constexpr, LIMIT: tl.constexpr, COMMIT: tl.constexpr, WINDOW: tl.constexpr):
    row = tl.program_id(0)
    slot = tl.load(Rows + row).to(tl.int64)
    length = tl.load(Lengths + row)
    if (slot > 0) & (length > 0):
        base = Backing + slot * ROW_BYTES
        ring = (base + RING_OFFSET).to(tl.pointer_type(tl.int32))
        pending = (base + COUNT_OFFSET).to(tl.pointer_type(tl.int32))
        start, count = tl.load(ring), tl.load(pending)
        if COMMIT:
            checkpointed = count + length > WINDOW
            start = tl.where(checkpointed, (start + count) % RING, start)
            count = tl.where(checkpointed, 0, count) + tl.load(Accepted + row)
        else:
            if count > LIMIT:
                start = (start + count) % RING
                count = 0
        tl.store(ring, start)
        tl.store(pending, count)
        tl.store(Starts + slot, start)
        tl.store(Pending + slot, count)


class PreparedMambaReplay:
    """Materialize accepted records across layers before changing execution width.

    Replay fields and ring metadata live in continuation rows, so native state
    transfers preserve the complete history. Ring vectors are produced for each
    invocation and populated only for physically backed active rows.
    """

    def __init__(self, backing, layout, regions, A, *, ring_region, count_region, ring_length):
        if A.ndim != 2 or A.dtype != torch.float32 or A.shape[0] != len(regions):
            raise ValueError("Mamba replay needs one FP32 decay vector per recurrent layer")
        state, x, b, dt = regions[0]
        heads, dim, n = state.shape
        if (state.dtype != torch.bfloat16 or x.shape != (heads, ring_length, dim)
                or b.shape[1:] != (ring_length, n) or dt.shape != (heads, ring_length)):
            raise ValueError("Mamba replay regions have incompatible geometry")
        for r in regions:
            if tuple((v.shape, v.dtype) for v in r) != tuple((v.shape, v.dtype) for v in regions[0]):
                raise ValueError("batched Mamba replay requires matching layer geometry")
        self.backing, self.A = backing, A
        self.offsets = persistent_copy(torch.tensor([[r.offset for r in layer] for layer in regions],
                                                    dtype=torch.int64, device=A.device))
        self.constants = dict(ROW_BYTES=layout.row_bytes, RING_OFFSET=ring_region.offset,
            COUNT_OFFSET=count_region.offset, RING=ring_length)
        self.geometry = dict(HEADS=heads, GROUPS=b.shape[0], DIM=dim, N=n)
    def prepare(self, rows, lengths, *, width=1):
        ring_start = torch.empty((self.backing.shape[0],), dtype=torch.int32, device=self.backing.device)
        pending_count = torch.empty_like(ring_start)
        limit = 0 if width == 1 else self.constants["RING"] - width
        grid = (len(self.offsets), rows.numel(), min(256, triton.cdiv(self.geometry["HEADS"] * self.geometry["DIM"] * self.geometry["N"], 256)))
        torch.library.wrap_triton(_materialize)[grid](self.backing, self.offsets, self.A, rows, lengths,
            **self.constants, **self.geometry, LIMIT=limit, BLOCK=256, num_warps=4)
        torch.library.wrap_triton(_replay_metadata)[(rows.numel(),)](self.backing, rows, lengths,
            ring_start, pending_count, lengths, **self.constants, LIMIT=limit, COMMIT=False, WINDOW=0)
        return ring_start, pending_count

    def commit(self, counts, rows, lengths, ring_start, pending_count, *, width):
        torch.library.wrap_triton(_replay_metadata)[(rows.numel(),)](self.backing, rows, lengths,
            ring_start, pending_count, counts, **self.constants, LIMIT=0, COMMIT=True,
            WINDOW=self.constants["RING"] - width)
