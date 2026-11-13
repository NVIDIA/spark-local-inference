# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared mixed-format sliding-window and indexed latent attention."""

import inspect
import torch
from functools import partial
from b12x.attention import compressed_sparse_mla as mla
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle

from ignition.preparation.scratch import planned_scratch, prime_owners
from ignition.preparation.tuning import declare, register_finalizer
from ignition.preparation.fixtures.compressed_mla import Inputs, benchmark_call
from ignition.preparation.fixtures.resources import candidate_inputs


def validate_mla_compute_mode(mode):
    """Validate an arithmetic request before loading model weights."""
    if mode not in ("auto", "bf16"):
        raise ValueError("MLA compute mode must be 'auto' or 'bf16'")
    if mode != "auto" and "compute_mode" not in inspect.signature(mla.Caps).parameters:
        raise ValueError("BF16 MLA arithmetic requires FlashInfer/b12x with the"
                         " compressed_sparse_mla.Caps.compute_mode preparation API;"
                         " select a preparation environment that provides this API")


@torch.library.custom_op("flare::compressed_cache_write", mutates_args=("cache",))
def _write(kv: torch.Tensor, cache: torch.Tensor, slots: torch.Tensor, handle: int,
           history: torch.Tensor | None = None, history_rows: torch.Tensor | None = None) -> None:
    mla.write_cache(kv, cache, slots, plan=plan_from_handle(handle), history=history, history_rows=history_rows)


@_write.register_fake
def _write_fake(kv, cache, slots, handle, history=None, history_rows=None):
    pass


class PreparedCompressedCacheWriter:
    def __init__(self, values, cache, slots, *, page_size, kind, session, name, cache_format="deepseek_v41",
                 history=None, history_rows=None, snapshot_tokens=0):
        if bool(snapshot_tokens) != (history is not None and history_rows is not None):
            raise ValueError("snapshot cache writers require history, row IDs, and a prefix extent")
        if (history is None) != (history_rows is None):
            raise ValueError("history and history_rows must be supplied together")
        self.plan = mla.plan_cache_writer(mla.CacheWriterQuery(max_rows=values.shape[0],
            page_size=page_size, cache_kind=kind, cache_format=cache_format,
            slot_dtype=str(slots.dtype).removeprefix("torch."), snapshot_tokens=snapshot_tokens,
            max_requests=0 if history_rows is None else history_rows.numel()),
            device=values.device)
        self.cache, self.slots = cache, slots
        self.history, self.history_rows = history, history_rows

        def prepare(state):
            source = torch.zeros_like(values)
            rows = self.plan.query.max_requests or 1
            target = torch.empty_strided((rows, cache.shape[1]), cache.stride(), dtype=cache.dtype, device=cache.device)
            positions = torch.arange(values.shape[0], dtype=slots.dtype, device=slots.device)
            if snapshot_tokens:
                suffix = page_size - snapshot_tokens
                positions = positions // suffix * page_size + snapshot_tokens + positions % suffix
                positions.masked_fill_(positions >= rows * page_size, -1)
                seed = torch.zeros((rows + 1, snapshot_tokens * 528), dtype=torch.uint8, device=cache.device)
                selected = torch.arange(1, rows + 1, dtype=torch.int64, device=cache.device)
            else:
                positions.masked_fill_(positions >= page_size, -1)
                seed = selected = None
            return PreparedCall(run=lambda: state.run(source, target, positions, history=seed, history_rows=selected))

        declare(session, (self.plan.request(name=name, prepare_call=prepare),))

    def functional(self, *, session, cache=...):
        if cache is not ...:
            self.cache = cache
        register_finalizer(session, self._release_examples)
        return self

    def _release_examples(self):
        self.slots = self.history_rows = None

    def __call__(self, values, slots=None, *, history_rows=None, cache=None):
        _write(values, self.cache if cache is None else cache, self.slots if slots is None else slots, self.plan.handle,
               self.history, self.history_rows if history_rows is None else history_rows)


@torch.library.custom_op("flare::compressed_mla", mutates_args=("out", "scratch"))
def _attend(q: torch.Tensor, swa: torch.Tensor, indices: torch.Tensor, lengths: torch.Tensor,
            indexed: torch.Tensor | None, indexed_indices: torch.Tensor | None,
            indexed_lengths: torch.Tensor | None, page_table: torch.Tensor | None,
            sink: torch.Tensor, out: torch.Tensor, scratch: torch.Tensor, scale: float, handle: int) -> None:
    plan = plan_from_handle(handle)
    binding = mla.bind(plan, scratch=scratch, q=q, swa_indices=indices, swa_lengths=lengths,
        indexed_indices=indexed_indices, indexed_lengths=indexed_lengths, indexed_page_table=page_table)
    mla.run(binding=binding, swa_k_cache=swa, indexed_k_cache=indexed,
            attn_sink=sink, sm_scale=scale, out=out)


@_attend.register_fake
def _attend_fake(q, swa, indices, lengths, indexed, indexed_indices, indexed_lengths,
                 page_table, sink, out, scratch, scale, handle):
    pass


class PreparedCompressedMLA:
    def __init__(self, query, window, indices, lengths, sink, output, *, indexed=None,
                 indexed_indices=None, indexed_lengths=None, page_table=None,
                 window_page_size=128, indexed_page_size=64, mode="decode", session, name, workspace,
                 cache_format="deepseek_v41", decode_row_capacity=None, compute_mode="auto"):
        validate_mla_compute_mode(compute_mode)
        width = 0 if indexed_indices is None else indexed_indices.shape[1]
        caps = mla.Caps(device=query.device, num_q_heads=query.shape[1], max_q_rows=query.shape[0],
            max_width=indices.shape[1] + width, swa_width=indices.shape[1], indexed_width=width,
            max_page_table_width=1 if page_table is None else page_table.shape[1],
            swa_page_size=window_page_size, indexed_page_size=indexed_page_size,
            cache_format=cache_format, mode=mode, decode_row_capacity=decode_row_capacity,
            compute_mode=compute_mode)
        self.plan = mla.plan(caps, invocation=mla.invocation_from_tensors(q=query,
            swa_k_cache=window, indexed_k_cache=indexed, attn_sink=sink, out=output))
        self.query, self.window, self.indexed, self.output = query, window, indexed, output
        self.inputs = dict(q=query, swa_indices=indices, swa_lengths=lengths,
            indexed_indices=indexed_indices, indexed_lengths=indexed_lengths, indexed_page_table=page_table)
        self.sink, self.scale = sink, query.shape[-1] ** -.5
        self.workspace = workspace

        def prepare(state):
            inputs = benchmark_inputs()
            scratch = planned_scratch(state.scratch_specs(), workspace)
            binding = state.bind_for_preparation(scratch=scratch, **inputs.binding)
            return PreparedCall(run=lambda: state.run(binding, swa_k_cache=inputs.window, indexed_k_cache=inputs.indexed,
                attn_sink=inputs.sink, sm_scale=self.scale, out=inputs.out),
                owners=prime_owners(workspace, inputs, scratch, binding))

        benchmark_inputs = candidate_inputs(partial(Inputs, query, window, indexed, self.inputs,
            cache_format, window_page_size, indexed_page_size), session=session)
        declare(session, (self.plan.request(name=name, prepare_call=prepare,
            benchmark_call=partial(benchmark_call, inputs=benchmark_inputs, workspace=workspace)),))
        register_finalizer(session, self.rebind, workspace=workspace)

    def rebind(self):
        self.scratch = planned_scratch(self.plan.scratch_specs(), self.workspace)

    def functional(self, *, session, window=...):
        if window is not ...:
            self.window = window
        self.output_layout = tuple(self.output.shape), tuple(self.output.stride()), self.output.dtype
        register_finalizer(session, self._finalize_functional)
        return self

    def _finalize_functional(self):
        self.scratch_layout = tuple((tuple(spec.shape), spec.dtype) for spec in self.plan.scratch_specs())
        self.query = self.output = self.inputs = self.workspace = self.scratch = None

    def __call__(self, query=None, indices=None, lengths=None, *, indexed_indices=None,
                 indexed_lengths=None, page_table=None, window=None):
        if self.inputs is not None:
            return self._bound_call()
        shape, stride, dtype = self.output_layout
        output = torch.empty_strided(shape, stride, dtype=dtype, device=query.device).zero_()
        scratch, = (torch.empty(shape, dtype=dtype, device=query.device) for shape, dtype in self.scratch_layout)
        _attend(query, self.window if window is None else window, indices, lengths, self.indexed, indexed_indices, indexed_lengths,
                page_table, self.sink, output, scratch, self.scale, self.plan.handle)
        return output

    def _bound_call(self):
        args = self.inputs
        _attend(self.query, self.window, args["swa_indices"], args["swa_lengths"], self.indexed,
            args["indexed_indices"], args["indexed_lengths"], args["indexed_page_table"],
            self.sink, self.output, self.scratch, self.scale, self.plan.handle)
        return self.output
