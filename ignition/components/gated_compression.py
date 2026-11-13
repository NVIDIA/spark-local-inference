# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Gated pooling with read-only history and explicit accepted-prefix commit."""

import torch
import triton
import triton.language as tl

from ignition.components.linear import UnquantizedProjection
from ignition.components.rms_norm import PreparedRMSNorm
from ignition.preparation.examples import declaration_tensor
from ignition.preparation.tuning import register_finalizer


@triton.jit
def _pool(
    Values,
    Scores,
    Ape,
    StateValues,
    StateScores,
    Requests,
    Positions,
    Starts,
    Begins,
    StateRows,
    Output,
    SS: tl.constexpr,
    TS: tl.constexpr,
    HEAD: tl.constexpr,
    RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    dim = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    request = tl.load(Requests + token)
    position = tl.load(Positions + token)
    start = tl.load(Starts + request)
    begin = tl.load(Begins + request)
    state = tl.load(StateRows + request)
    emitted = (position >= 0) & ((position + 1) % RATIO == 0) & (state > 0)
    if emitted:
        extent: tl.constexpr = RATIO * (2 if OVERLAP else 1)
        width: tl.constexpr = HEAD * (2 if OVERLAP else 1)
        offsets = tl.arange(0, extent)
        absolute = position + 1 - extent + offsets
        channel = dim[None, :] + (tl.where(offsets[:, None] < RATIO, 0, HEAD) if OVERLAP else 0)
        valid = (absolute[:, None] >= 0) & (dim[None, :] < HEAD)
        local = absolute[:, None] >= start
        source = (begin + absolute[:, None] - start) * TS + channel
        history = state * SS + (absolute[:, None] % extent) * width + channel
        values = tl.load(Values + source, valid & local, other=0)
        values += tl.load(StateValues + history, valid & ~local, other=0)
        scores = tl.load(Scores + source, valid & local, other=0)
        scores += tl.load(StateScores + history, valid & ~local, other=0)
        scores += tl.load(Ape + (absolute[:, None] % RATIO) * width + channel, valid, other=0)
        scores = tl.where(valid, scores, -float("inf"))
        maximum = tl.max(scores, 0)
        probabilities = tl.exp(scores - maximum[None, :])
        pooled = tl.sum(values * probabilities, 0) / tl.sum(probabilities, 0)
        tl.store(Output + token * HEAD + dim, pooled, dim < HEAD)
    else:
        tl.store(Output + token * HEAD + dim, 0, dim < HEAD)


@triton.jit
def _commit(
    Values,
    Scores,
    StateValues,
    StateScores,
    Starts,
    Begins,
    Counts,
    StateRows,
    SS: tl.constexpr,
    TS: tl.constexpr,
    WIDTH: tl.constexpr,
    EXTENT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request, distance = tl.program_id(0), tl.program_id(1)
    dim = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    start, count = tl.load(Starts + request), tl.load(Counts + request)
    begin, state = tl.load(Begins + request), tl.load(StateRows + request)
    end = start + count
    absolute = end - 1 - distance
    ring_row = absolute % EXTENT
    valid = (state > 0) & (count > 0) & (absolute >= start) & (dim < WIDTH)
    source = (begin + absolute - start) * TS + dim
    destination = state * SS + ring_row * WIDTH + dim
    values = tl.load(Values + source, valid, other=0.)
    scores = tl.load(Scores + source, valid, other=0.)
    tl.store(StateValues + destination, values, valid)
    tl.store(StateScores + destination, scores, valid)


def pool_gated(values, scores, ape, state_values, state_scores, metadata, output, *, ratio):
    """Emit each complete group without changing committed compressor history."""
    head = output.shape[1]
    torch.library.wrap_triton(_pool)[(values.shape[0], triton.cdiv(head, 32))](
        values,
        scores,
        ape,
        state_values,
        state_scores,
        metadata.requests,
        metadata.positions,
        metadata.starts,
        metadata.query_start_loc,
        metadata.state_rows,
        output,
        state_values.stride(0),
        values.stride(0),
        head,
        ratio,
        ratio == 4,
        32,
        enable_fp_fusion=False,
    )


def commit_gated(values, scores, state_values, state_scores, metadata, counts):
    """Update ring slots touched by the accepted prefix, preserving inactive rows."""
    torch.library.wrap_triton(_commit)[
        (counts.numel(), min(state_values.shape[1], values.shape[0]), triton.cdiv(values.shape[1], 256))
    ](
        values,
        scores,
        state_values,
        state_scores,
        metadata.starts,
        metadata.query_start_loc,
        counts,
        metadata.state_rows,
        state_values.stride(0),
        values.stride(0),
        values.shape[1],
        state_values.shape[1],
        256,
    )


class PreparedGatedCompressor:
    """Gate and pool C4/C128 compression windows with explicit runtime intermediates."""

    def __init__(self, spec, weights, source, metadata, pool, slots, *, session, name, deferred=False,
                 buffers=None, emitted=None, emitted_slots=None, projected=None, examples=None, example_role=None):
        if spec.ratio not in (4, 128) or spec.head_dim not in (128, 512):
            raise ValueError("gated compression requires C4/C128 and a 128/512-dimensional head")
        self.spec, self.metadata, self.deferred = spec, metadata, deferred
        self.state_values = pool.reserved_region(f"{spec.prefix}.values")
        self.state_scores = pool.reserved_region(f"{spec.prefix}.scores")
        self.ape = weights["ape"]
        tokens, device = source.shape[0], source.device

        if examples is not None and (not isinstance(example_role, str) or not example_role):
            raise ValueError("shared compressor declarations require an explicit operand role")

        def zeros(role, shape, dtype):
            if examples is not None:
                return declaration_tensor(examples, example_role + "." + role, shape, dtype, device)
            return torch.zeros(shape, dtype=dtype, device=device) if buffers is None else buffers.zeros(shape, dtype)

        # A caller projecting several compressors' values and gates in one GEMM passes their outputs.
        self.projected = projected is not None
        if self.projected:
            self.values, self.scores = projected
            if any(t.shape != (tokens, spec.state_shape[1]) or t.dtype != torch.float32
                   or t.device != source.device or t.stride(1) != 1 for t in projected):
                raise ValueError("projected compressor values and scores require FP32 rows on the input device")
            if self.values.stride() != self.scores.stride():
                raise ValueError("projected compressor values and scores require matching strides")
        else:
            self.values = zeros("values", (tokens, spec.state_shape[1]), torch.float32)
            self.scores = zeros("scores", (tokens, spec.state_shape[1]), torch.float32)
        self.pooled = zeros("pooled", (tokens, spec.head_dim), torch.bfloat16)
        self.output = zeros("output", (tokens, spec.head_dim), torch.bfloat16)
        # A caller computing emission flags and slots once for several compressors passes them in.
        self.shared_slots = emitted is not None
        if self.shared_slots:
            if emitted_slots is None or emitted.shape != (tokens,) or emitted_slots.shape != slots.shape:
                raise ValueError("shared emission flags and slots must cover every source row")
            self.emitted, self.slots = emitted, emitted_slots
        else:
            self.emitted = zeros("emitted", (tokens,), torch.bool)
            self.slots = (zeros("slots", slots.shape, slots.dtype).fill_(-1) if examples is not None
                          else torch.full_like(slots, -1) if buffers is None
                          else buffers.full(slots.shape, -1, slots.dtype))
        self.destinations = slots
        self.project_values = self.project_scores = None
        if not self.projected:
            self.project_values = UnquantizedProjection(
                source, weights["values"], self.values, session=session, name=name + ".values"
            )
            self.project_scores = UnquantizedProjection(
                source, weights["gates"], self.scores, session=session, name=name + ".gates"
            )
        self.normalize = PreparedRMSNorm(
            self.pooled,
            weights["norm"],
            self.output,
            eps=pool.config.text["rms_norm_eps"],
            session=session,
            name=name + ".norm",
        )

    def functional(self, *, session):
        self.output_shape = tuple(self.output.shape)
        if not self.projected:
            self.project_values.functional(session=session)
            self.project_scores.functional(session=session)
        self.normalize = self.normalize.functional()
        register_finalizer(session, self._release_examples)
        return self

    def _release_examples(self):
        self.metadata = self.values = self.scores = self.pooled = self.output = None
        self.emitted = self.slots = self.destinations = None

    def __call__(self, source, metadata=None, slots=None, *, projected=None, emitted=None, emitted_slots=None):
        if self.metadata is None:
            values, scores = (projected if self.projected else
                              (self.project_values(source), self.project_scores(source)))
            pooled = torch.zeros(self.output_shape, dtype=torch.bfloat16, device=source.device)
            pool_gated(values, scores, self.ape, self.state_values, self.state_scores,
                       metadata, pooled, ratio=self.spec.ratio)
            output = self.normalize(pooled)
            if emitted is None:
                positions = metadata.positions
                emitted = (positions >= 0) & ((positions + 1) % self.spec.ratio == 0)
                emitted_slots = torch.where(emitted, slots, -1)
            if not self.deferred:
                self.commit(metadata.lengths, (values, scores), metadata)
            return output, values, scores, emitted, emitted_slots
        if not self.projected:
            self.project_values(source)
            self.project_scores(source)
        pool_gated(
            self.values,
            self.scores,
            self.ape,
            self.state_values,
            self.state_scores,
            self.metadata,
            self.pooled,
            ratio=self.spec.ratio,
        )
        self.normalize(self.pooled)
        if not self.shared_slots:
            positions = self.metadata.positions
            self.emitted.copy_((positions >= 0) & ((positions + 1) % self.spec.ratio == 0))
            self.slots.copy_(torch.where(self.emitted, self.destinations, -1))
        if not self.deferred:
            self.commit(self.metadata.lengths)
        return self.output

    def commit(self, counts, records=None, metadata=None):
        values, scores = (self.values, self.scores) if records is None else records
        commit_gated(values, scores, self.state_values, self.state_scores,
                     self.metadata if metadata is None else metadata, counts)
