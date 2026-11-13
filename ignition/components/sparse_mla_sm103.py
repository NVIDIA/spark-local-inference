# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM sparse MLA over selected latent records on SM103.

b12x's sparse MLA issues instructions SM103 does not have. Each GLM latent
record is 528 bytes: 512 E4M3 values and four FP32 scales, one per 128
channels. A row's absorbed BF16 queries ([64 heads, 512]) attend to its
selected records with the dequantized latent as both key and value, softmax at
the model's scale, and positions past the row's selected count or with
negative indices masked.

FlashInfer's Cake DSA kernel computes this attention on BF16 latents at tensor
core rate (GB300: 0.99 ms for 4096 rows of 2048 records, 14x a CUDA-core Triton
kernel); a zero query rope makes its rope product vanish. Its launches cost
about 31 us at any row count, so up to four rows attend through a Triton
kernel that splits each row's selection across programs and merges the partial
softmaxes (14 us for one row). Larger decode batches dequantize their selected
records into per-row BF16 slices for Cake; prefill dequantizes the request's
context once into a slab indexed by position and selects positions instead of
records.
"""

from __future__ import annotations

import itertools
import weakref
from dataclasses import dataclass
from math import prod

import torch
import triton
import triton.language as tl
from b12x.preparation import PreparedCall

from ignition.preparation.scratch import ScratchArena
from ignition.preparation.triton import launch_triton, warmup_triton
from ignition.preparation.tuning import declare, register_finalizer

from .native_operator import native_contract, native_plan

_RECORD_BYTES = 528
_DIM = 512
_HEADS = 64
_ROPE = 64
_PAGE_TOKENS = 256
_BLOCK_N = 32
_BLOCK_H = 16
_CONFIG = triton.Config({}, num_warps=4, num_stages=2)
_MERGE_CONFIG = triton.Config({}, num_warps=4, num_stages=1)
_COPY_CONFIG = triton.Config({}, num_warps=4, num_stages=1)
_COPY_BLOCK = 32
# Rows that split their selection, and the partial rows (splits x rows) a launch may need.
SPLIT_ROWS = 64
PARTIAL_ROWS = 512
# Rows the Triton kernel attends; more go through Cake in slices that bound gathered scratch.
TRITON_ROWS = 4
CAKE_DECODE_ROWS = 128
CAKE_PREFILL_ROWS = 2048
_OWNERS = weakref.WeakValueDictionary()
_HANDLES = itertools.count()


@dataclass(frozen=True)
class CakeQuery:
    rows: int
    slice_rows: int
    latent_rows: int


CONTRACT = native_contract(component_id="flare.sm103_cake_sparse_mla", query_type=CakeQuery,
                           tactics=lambda query: ("cake",))


def splits_for(rows: int) -> int:
    """Programs per row and head group: 32 at eight rows or fewer, halving as rows double."""
    if not 0 < rows <= SPLIT_ROWS:
        return 1
    return max(2, 256 // triton.next_power_of_2(max(rows, 8)))


@triton.jit
def _dequantize(Records, Scales, record, valid, DIM: tl.constexpr, RECORD_BYTES: tl.constexpr,
                BLOCK: tl.constexpr):
    dims = tl.arange(0, DIM)
    groups = tl.arange(0, DIM // 128)
    safe = tl.maximum(record, 0).to(tl.int64)
    values = tl.load(Records + safe[:, None] * RECORD_BYTES + dims[None, :], mask=valid[:, None], other=0)
    scales = tl.load(Scales + safe[:, None] * (RECORD_BYTES // 4) + DIM // 4 + groups[None, :],
                     mask=valid[:, None], other=0.0)
    latent = values.to(tl.float8e4nv, bitcast=True).to(tl.float32).reshape(BLOCK, DIM // 128, 128)
    return (latent * scales[:, :, None]).reshape(BLOCK, DIM).to(tl.bfloat16)


@triton.autotune(configs=[_CONFIG], key=[])
@triton.jit
def _sparse_mla(Query, Records, Scales, Selected, Counts, Output, PartialOut, PartialLse,
                query_row_stride, query_head_stride, selected_row_stride, output_row_stride, output_head_stride,
                SCALE_LOG2: tl.constexpr, HEADS: tl.constexpr, WIDTH: tl.constexpr, ROWS: tl.constexpr,
                DIM: tl.constexpr, RECORD_BYTES: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr,
                SPLITS: tl.constexpr):
    row = tl.program_id(0)
    head_block = tl.program_id(1)
    split = tl.program_id(2)
    heads = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    dims = tl.arange(0, DIM)
    columns = tl.arange(0, BLOCK_N)
    count = tl.load(Counts + row)
    query = tl.load(Query + row.to(tl.int64) * query_row_stride + heads[:, None] * query_head_stride + dims[None, :],
                    mask=heads[:, None] < HEADS, other=0.0)
    maximum = tl.full((BLOCK_H,), -1.0e20, tl.float32)
    normalizer = tl.zeros((BLOCK_H,), tl.float32)
    accumulator = tl.zeros((BLOCK_H, DIM), tl.float32)
    for tile in range(split, tl.cdiv(WIDTH, BLOCK_N), SPLITS):
        column = tile * BLOCK_N + columns
        record = tl.load(Selected + row.to(tl.int64) * selected_row_stride + column,
                         mask=(column < WIDTH) & (column < count), other=-1)
        valid = record >= 0
        latent = _dequantize(Records, Scales, record, valid, DIM, RECORD_BYTES, BLOCK_N)
        scores = tl.dot(query, tl.trans(latent)) * SCALE_LOG2
        scores = tl.where(valid[None, :], scores, -1.0e20)
        following = tl.maximum(maximum, tl.max(scores, axis=1))
        alpha = tl.math.exp2(maximum - following)
        probabilities = tl.where(valid[None, :], tl.math.exp2(scores - following[:, None]), 0.0)
        accumulator = accumulator * alpha[:, None] + tl.dot(probabilities.to(tl.bfloat16), latent)
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        maximum = following
    attended = normalizer > 0
    output = tl.where(attended[:, None], accumulator / tl.maximum(normalizer, 1.0e-20)[:, None], 0.0)
    if SPLITS == 1:
        tl.store(Output + row.to(tl.int64) * output_row_stride + heads[:, None] * output_head_stride + dims[None, :],
                 output.to(Output.dtype.element_ty), mask=heads[:, None] < HEADS)
    else:
        lse = tl.where(attended, maximum + tl.math.log2(tl.maximum(normalizer, 1.0e-20)), -float("inf"))
        partial = (split * ROWS + row).to(tl.int64)
        tl.store(PartialOut + (partial * HEADS + heads[:, None]) * DIM + dims[None, :], output,
                 mask=heads[:, None] < HEADS)
        tl.store(PartialLse + partial * HEADS + heads, lse, mask=heads < HEADS)


@triton.autotune(configs=[_MERGE_CONFIG], key=[])
@triton.jit
def _merge_splits(PartialOut, PartialLse, Output, output_row_stride, output_head_stride,
                  HEADS: tl.constexpr, DIM: tl.constexpr, ROWS: tl.constexpr, SPLITS: tl.constexpr):
    row = tl.program_id(0)
    head = tl.program_id(1)
    splits = tl.arange(0, SPLITS)
    dims = tl.arange(0, DIM)
    partials = (splits * ROWS + row).to(tl.int64)
    lse = tl.load(PartialLse + partials * HEADS + head)
    top = tl.max(lse, axis=0)
    weights = tl.where(top > -float("inf"), tl.math.exp2(lse - top), 0.0)
    denominator = tl.sum(weights, axis=0)
    values = tl.load(PartialOut + (partials[:, None] * HEADS + head) * DIM + dims[None, :])
    merged = tl.sum(values * weights[:, None], axis=0)
    merged = tl.where(denominator > 0, merged / denominator, 0.0)
    tl.store(Output + row.to(tl.int64) * output_row_stride + head * output_head_stride + dims,
             merged.to(Output.dtype.element_ty))


@triton.autotune(configs=[_COPY_CONFIG], key=[])
@triton.jit
def _gather_records(Records, Scales, Selected, Counts, Gathered, Indices, selected_row_stride,
                    WIDTH: tl.constexpr, DIM: tl.constexpr, RECORD_BYTES: tl.constexpr, BLOCK: tl.constexpr):
    # Dequantize each row's selected records into its own BF16 slice and index them there.
    row = tl.program_id(0)
    column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inside = column < WIDTH
    count = tl.load(Counts + row)
    record = tl.load(Selected + row.to(tl.int64) * selected_row_stride + column, mask=inside & (column < count),
                     other=-1)
    valid = record >= 0
    slot = row * WIDTH + column
    tl.store(Indices + slot, tl.where(valid, slot, -1), mask=inside)
    latent = _dequantize(Records, Scales, record, valid, DIM, RECORD_BYTES, BLOCK)
    dims = tl.arange(0, DIM)
    tl.store(Gathered + slot.to(tl.int64)[:, None] * DIM + dims[None, :], latent, mask=valid[:, None])


@triton.autotune(configs=[_COPY_CONFIG], key=[])
@triton.jit
def _pull_context(Records, Scales, Table, End, Slab, table_width, DIM: tl.constexpr, RECORD_BYTES: tl.constexpr,
                  PAGE_TOKENS: tl.constexpr, BLOCK: tl.constexpr):
    # Dequantize the records of positions below End into the slab, indexed by position.
    first = tl.program_id(0) * BLOCK
    end = tl.load(End)
    if first < end:
        token = first + tl.arange(0, BLOCK)
        live = token < end
        page = tl.load(Table + tl.minimum(token // PAGE_TOKENS, table_width - 1), mask=live, other=0).to(tl.int64)
        record = tl.where(live, page * PAGE_TOKENS + token % PAGE_TOKENS, -1)
        latent = _dequantize(Records, Scales, record, live, DIM, RECORD_BYTES, BLOCK)
        dims = tl.arange(0, DIM)
        tl.store(Slab + token.to(tl.int64)[:, None] * DIM + dims[None, :], latent, mask=live[:, None])


# ``reads`` lists every tensor the launches read besides the query and records, so compilation orders them.
@torch.library.custom_op("flare::sm103_cake_mla", mutates_args=("output", "scratch"))
def _cake(query: torch.Tensor, records: torch.Tensor, reads: list[torch.Tensor], output: torch.Tensor,
          scratch: list[torch.Tensor], handle: int) -> None:
    owner = _OWNERS[handle]
    if owner.query is None:
        owner.run_functional(query, records, reads, output, scratch)
    else:
        owner.run_cake()


@_cake.register_fake
def _fake_cake(query, records, reads, output, scratch, handle) -> None:
    pass


def _records(cache):
    if cache.dtype != torch.uint8 or cache.shape[-1] != _RECORD_BYTES or not cache.is_contiguous():
        raise ValueError("SM103 sparse MLA requires contiguous 528-byte uint8 latent records")
    records = cache.reshape(-1, _RECORD_BYTES)
    return records, records.view(torch.float32)


def _check_attention(query, output, lengths):
    rows, _, dim = query.shape
    if (dim != _DIM or query.dtype != torch.bfloat16 or query.stride(2) != 1
            or output.shape != query.shape or output.dtype != torch.bfloat16 or not output.is_contiguous()
            or lengths.shape != (rows,) or lengths.dtype != torch.int32):
        raise ValueError("SM103 sparse MLA requires BF16 [rows, heads, 512] queries, contiguous BF16 outputs "
                         "and int32 selected counts")


class _CakeAttention:
    """Cake runners over row slices of one binding; latents, rope, LSE and residuals live in arena scratch.

    Subclasses fill the BF16 latents and supply each slice's latent indices before ``attend``. The
    declared plan sizes that scratch; runners bind to it once the arena is final.
    """

    def __init__(self, query, lengths, output, scale, *, slice_rows, latent_rows, session, name, workspace):
        if query.shape[1] != _HEADS:
            raise ValueError("Cake sparse MLA requires 64 query heads")
        self.query, self.lengths, self.output, self.scale = query, lengths, output, scale
        self.workspace = workspace
        rows = query.shape[0]
        self.slice_rows = min(rows, slice_rows)
        self.slices = [(start, min(start + self.slice_rows, rows)) for start in range(0, rows, self.slice_rows)]
        self.specs = (((latent_rows, _DIM), torch.bfloat16), ((self.slice_rows, _HEADS, _ROPE), torch.bfloat16),
                      ((self.slice_rows, _HEADS), torch.float32), ((self.slice_rows, _HEADS, _DIM), torch.bfloat16))
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self
        # Triton programs compile against a placeholder; launches bind the arena's latents.
        self.latents = torch.empty(0, dtype=torch.bfloat16, device=query.device)
        nbytes = sum((prod(shape) * dtype.itemsize + 255) // 256 * 256 for shape, dtype in self.specs)
        self.plan = native_plan(contract=CONTRACT, query=CakeQuery(rows, self.slice_rows, latent_rows),
                                device=query.device, scratch_bytes=lambda tactic: nbytes,
                                materialize=lambda tactic: tactic)
        declare(session, (self.plan.request(name=name, prepare_call=lambda state: PreparedCall(run=lambda: None)),))
        register_finalizer(session, self.bind, workspace=workspace)

    def views(self):
        if isinstance(self.workspace, ScratchArena):
            return self.workspace.views(*self.specs)
        if not hasattr(self, "_private"):
            self._private = tuple(torch.empty(shape, dtype=dtype, device=self.query.device)
                                  for shape, dtype in self.specs)
        return self._private

    def slice_indices(self, start, end):
        raise NotImplementedError

    def bind(self):
        if self.query is None:
            return
        from flashinfer.experimental.cake_dsa_train.cake_backend import prepare_dsa_train

        self.latents, self.rope, lse, residual = self.views()
        self.lse, self.residual = lse, residual
        self.runners = []
        for start, end in self.slices:
            count = end - start
            self.runners.append(prepare_dsa_train(self.query[start:end], self.rope[:count], self.latents,
                self.latents[:, :_ROPE], self.slice_indices(start, end), topk_length=self.lengths[start:end],
                softmax_scale=self.scale, out=self.output[start:end], lse=lse[:count], o_lo=residual[:count],
                backward=False))

    def functional(self, *, session):
        self.output_shape = tuple(self.output.shape)
        register_finalizer(session, self._finalize_functional)
        return self

    def _finalize_functional(self):
        if self.query is None:
            return
        self.runners = [runner.functional() for runner in self.runners]
        self.query = self.lengths = self.output = self.workspace = None
        self.latents = self.rope = self.lse = self.residual = None
        self._private = None
        if isinstance(self, _GatheredCake):
            self.selected = self.local = None
        else:
            self.positions = self.table = self.end = None

    def local_scratch(self, device):
        return [torch.zeros(shape, dtype=dtype, device=device) for shape, dtype in self.specs]

    def attend(self, runner):
        # Other operators reuse the arena between calls, so the zero rope is rewritten every time.
        self.rope.zero_()
        runner.forward()


class _GatheredCake(_CakeAttention):
    """Decode batches: each slice gathers its rows' selected records, then Cake attends over them."""

    def __init__(self, query, records, selected, lengths, output, scale, *, session, name, workspace):
        self.records, self.selected = records, selected
        self.width = selected.shape[1]
        slice_rows = min(query.shape[0], CAKE_DECODE_ROWS)
        self.local = torch.empty((slice_rows, self.width), dtype=torch.int32, device=query.device)
        super().__init__(query, lengths, output, scale, slice_rows=slice_rows, latent_rows=slice_rows * self.width,
                         session=session, name=name, workspace=workspace)
        self.gather_options = {"WIDTH": self.width, "DIM": _DIM, "RECORD_BYTES": _RECORD_BYTES, "BLOCK": _COPY_BLOCK}
        start, end = self.slices[0]
        self.gather = warmup_triton(_gather_records.fn, self._gather_args(start, end), self._gather_grid(start, end),
            num_warps=_COPY_CONFIG.num_warps, num_stages=_COPY_CONFIG.num_stages, **self.gather_options)

    def slice_indices(self, start, end):
        return self.local[:end - start]

    def _gather_args(self, start, end):
        records, scales = self.records
        return (records, scales, self.selected[start:end], self.lengths[start:end], self.latents, self.local,
                self.selected.stride(0))

    def _gather_grid(self, start, end):
        return (end - start, triton.cdiv(self.width, _COPY_BLOCK), 1)

    def rebind_records(self, records):
        self.records = records

    def run_cake(self):
        for (start, end), runner in zip(self.slices, self.runners, strict=True):
            launch_triton(_gather_records, self.gather, self._gather_args(start, end), self._gather_grid(start, end),
                          **self.gather_options)
            self.attend(runner)

    def run_functional(self, query, records, reads, output, scratch):
        selected, lengths = reads
        latents, rope, lse, residual, local = scratch
        scales = records.view(torch.float32)
        for (start, end), runner in zip(self.slices, self.runners, strict=True):
            launch_triton(_gather_records, self.gather,
                (records, scales, selected[start:end], lengths[start:end], latents, local, selected.stride(0)),
                self._gather_grid(start, end), **self.gather_options)
            rope.zero_()
            count = end - start
            runner.forward_out(query[start:end], rope[:count], latents, latents[:, :_ROPE], local[:count],
                               lengths[start:end], out=output[start:end], lse=lse[:count], o_lo=residual[:count])


class Sm103SparseMLA:
    """Drop-in for ``PreparedSparseMLA`` on SM103: BF16 [rows, heads, 512] attention over selected records.

    ``cache`` holds 528-byte latent records (any shape whose last dimension is 528 bytes);
    ``selected`` int32 [rows, width] record indices, ``selected_lengths`` each row's count.
    Up to ``TRITON_ROWS`` rows attend through Triton, more through Cake over gathered records.
    Context-parallel LSE outputs and attention sinks are not provided.
    """

    def __init__(self, query, cache, selected, cache_lengths, selected_lengths, output, *,
                 caps, session, name, workspace, attention_sink=None, lse=None):
        rows, heads, _ = query.shape
        if attention_sink is not None or lse is not None:
            raise ValueError("SM103 sparse MLA does not provide attention sinks or LSE outputs")
        _check_attention(query, output, selected_lengths)
        if selected.dtype != torch.int32 or selected.shape[0] != rows or selected.stride(1) != 1:
            raise ValueError("SM103 sparse MLA requires int32 [rows, width] selections")
        self.query, self.selected, self.counts, self.output = query, selected, selected_lengths, output
        self.records = _records(cache)
        self.cake = None
        if rows > TRITON_ROWS:
            self.cake = _GatheredCake(query, self.records, selected, selected_lengths, output,
                                      float(caps.softmax_scale), session=session, name=name, workspace=workspace)
            return
        self.splits = splits_for(rows)
        self.options = {"SCALE_LOG2": float(caps.softmax_scale) * 1.4426950408889634, "HEADS": heads,
                        "WIDTH": selected.shape[1], "ROWS": rows, "DIM": _DIM, "RECORD_BYTES": _RECORD_BYTES,
                        "BLOCK_H": _BLOCK_H, "BLOCK_N": _BLOCK_N, "SPLITS": self.splits}
        self.grid = (rows, triton.cdiv(heads, _BLOCK_H), self.splits)
        self.partial = torch.zeros((PARTIAL_ROWS, heads, _DIM), dtype=torch.float32, device=query.device)
        self.partial_lse = torch.zeros((PARTIAL_ROWS, heads), dtype=torch.float32, device=query.device)
        if self.splits > 1 and self.splits * rows > PARTIAL_ROWS:
            raise ValueError("split sparse MLA needs more partial rows than the shared buffer holds")
        self._bind_triton()
        self.compiled = warmup_triton(_sparse_mla.fn, self.args, self.grid, num_warps=_CONFIG.num_warps,
                                      num_stages=_CONFIG.num_stages, **self.options)
        if self.splits > 1:
            self.merge_args = (self.partial, self.partial_lse, output, output.stride(0), output.stride(1))
            self.merge_options = {"HEADS": heads, "DIM": _DIM, "ROWS": rows, "SPLITS": self.splits}
            self.merge_grid = (rows, heads, 1)
            self.merge = warmup_triton(_merge_splits.fn, self.merge_args, self.merge_grid,
                                       num_warps=_MERGE_CONFIG.num_warps,
                                       num_stages=_MERGE_CONFIG.num_stages, **self.merge_options)

    def _bind_triton(self):
        q, s, o = self.query, self.selected, self.output
        records, scales = self.records
        self.args = (q, records, scales, s, self.counts, o, self.partial, self.partial_lse,
                     q.stride(0), q.stride(1), s.stride(0), o.stride(0), o.stride(1))

    def rebind_cache(self, cache):
        """Bind replacement paged storage before capture."""
        self.records = _records(cache)
        if self.cake is not None:
            self.cake.rebind_records(self.records)
        elif self.query is not None:
            self._bind_triton()

    def attend_in_scratch(self):
        """Attend into the caller's output, which no other operator shares; no LSE is produced."""
        return self(), None

    def functional(self, *, session):
        self.output_shape = tuple(self.output.shape)
        if self.cake is not None:
            self.cake.functional(session=session)
        register_finalizer(session, self._release_examples)
        return self

    def _release_examples(self):
        self.query = self.selected = self.counts = self.output = None
        self.args = self.partial = self.partial_lse = self.merge_args = None

    def __call__(self, query=None, selected=None, cache_lengths=None, selected_lengths=None, *, cache=None):
        if self.query is None:
            records = self.records if cache is None else _records(cache)
            output = torch.zeros(self.output_shape, dtype=torch.bfloat16, device=query.device)
            if self.cake is not None:
                scratch = self.cake.local_scratch(query.device)
                scratch.append(torch.zeros((self.cake.slice_rows, self.cake.width), dtype=torch.int32, device=query.device))
                _cake(query, records[0], [selected, selected_lengths], output, scratch, self.cake.handle)
                return output
            rows, heads, dim = self.output_shape
            partial = torch.zeros((self.splits * rows, heads, dim), dtype=torch.float32, device=query.device)
            lse = torch.zeros((self.splits * rows, heads), dtype=torch.float32, device=query.device)
            args = (query, *records, selected, selected_lengths, output, partial, lse,
                    query.stride(0), query.stride(1), selected.stride(0), output.stride(0), output.stride(1))
            launch_triton(_sparse_mla, self.compiled, args, self.grid, **self.options)
            if self.splits > 1:
                launch_triton(_merge_splits, self.merge,
                    (partial, lse, output, output.stride(0), output.stride(1)), self.merge_grid, **self.merge_options)
            return output
        if self.cake is not None:
            _cake(self.query, self.records[0], [self.selected, self.counts], self.output,
                  [self.cake.latents, self.cake.rope, self.cake.lse, self.cake.residual, self.cake.local], self.cake.handle)
            return self.output
        launch_triton(_sparse_mla, self.compiled, self.args, self.grid, **self.options)
        if self.splits > 1:
            launch_triton(_merge_splits, self.merge, self.merge_args, self.merge_grid, **self.merge_options)
        return self.output


class Sm103PrefillSparseMLA(_CakeAttention):
    """One request's prefill on SM103: its context dequantized into a BF16 slab, Cake over selected positions.

    ``positions`` int32 [rows, width] are the selected token positions of each row (-1 padded) and
    ``table`` the request's block table row of 256-token pages; ``end`` int64 [1] holds one past the
    last position the chunk attends. The slab holds every position the block table can address.
    """

    def __init__(self, query, cache, positions, selected_lengths, table, end, output, *, caps, session, name,
                 workspace):
        _check_attention(query, output, selected_lengths)
        if (positions.dtype != torch.int32 or positions.shape[0] != query.shape[0] or positions.stride(1) != 1
                or table.dtype != torch.int32 or table.ndim != 1 or table.stride(0) != 1
                or end.shape != (1,) or end.dtype != torch.int64):
            raise ValueError("SM103 prefill sparse MLA requires int32 positions, an int32 block table row "
                             "and an int64 end position")
        self.records, self.positions, self.table, self.end = _records(cache), positions, table, end
        super().__init__(query, selected_lengths, output, float(caps.softmax_scale), slice_rows=CAKE_PREFILL_ROWS,
                         latent_rows=table.shape[0] * _PAGE_TOKENS, session=session, name=name, workspace=workspace)
        self.pull_options = {"DIM": _DIM, "RECORD_BYTES": _RECORD_BYTES, "PAGE_TOKENS": _PAGE_TOKENS,
                             "BLOCK": _COPY_BLOCK}
        self.pull_grid = (triton.cdiv(table.shape[0] * _PAGE_TOKENS, _COPY_BLOCK), 1, 1)
        self.pull = warmup_triton(_pull_context.fn, self._pull_args(), self.pull_grid,
            num_warps=_COPY_CONFIG.num_warps, num_stages=_COPY_CONFIG.num_stages, **self.pull_options)

    def slice_indices(self, start, end):
        return self.positions[start:end]

    def _pull_args(self):
        records, scales = self.records
        return (records, scales, self.table, self.end, self.latents, self.table.shape[0])

    def rebind_cache(self, cache):
        """Bind replacement paged storage before capture."""
        self.records = _records(cache)

    def run_cake(self):
        launch_triton(_pull_context, self.pull, self._pull_args(), self.pull_grid, **self.pull_options)
        for runner in self.runners:
            self.attend(runner)

    def attend_in_scratch(self):
        """Attend into the caller's output, which no other operator shares; no LSE is produced."""
        return self(), None

    def run_functional(self, query, records, reads, output, scratch):
        positions, lengths, table, end = reads
        latents, rope, lse, residual = scratch
        launch_triton(_pull_context, self.pull,
            (records, records.view(torch.float32), table, end, latents, table.shape[0]),
            self.pull_grid, **self.pull_options)
        for (start, stop), runner in zip(self.slices, self.runners, strict=True):
            rope.zero_()
            count = stop - start
            runner.forward_out(query[start:stop], rope[:count], latents, latents[:, :_ROPE],
                               positions[start:stop], lengths[start:stop], out=output[start:stop],
                               lse=lse[:count], o_lo=residual[:count])

    def __call__(self, query=None, positions=None, selected_lengths=None, table=None, end=None, *, cache=None):
        if self.query is None:
            output = torch.zeros(self.output_shape, dtype=torch.bfloat16, device=query.device)
            records = self.records if cache is None else _records(cache)
            _cake(query, records[0], [positions, selected_lengths, table, end], output,
                  self.local_scratch(query.device), self.handle)
            return output
        _cake(self.query, self.records[0], [self.positions, self.lengths, self.table, self.end], self.output,
              [self.latents, self.rope, self.lse, self.residual], self.handle)
        return self.output
