# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Triton KDA verification records and accepted-prefix replay with b12x's record layout.

Verification reads each request's live FP32 state once and, per verified token,
writes the recurrence output and the delta-rule correction (one value vector per
head) with the token's raw BF16 key and gate; the live state is unchanged. After
acceptance, one launch over every KDA layer replays the accepted records into the
live state. Records keep b12x's layout ([slot, head, window, 128] FP32 corrections
and [slot, head, window, 256] BF16 raw key | raw gate), so either implementation's
commit replays the other's verification.

The arithmetic follows b12x's CuTe KDA recurrence: L2-normalized Q (scaled by
1/sqrt(128)) and K with epsilon 1e-6, a per-key log decay
lower * sigmoid(exp(A_log) * (g + dt_bias)), the update gate sigmoid(beta), then
per token S = S diag(exp(decay)); delta = (v - S k) sigmoid(beta); S += delta k^T;
o = S q. One warp owns a request, a head and ``ROWS`` value rows of the
[value, key] state, so each state element is read once per launch.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from ignition.preparation.triton import launch_triton, warmup_triton
from ignition.preparation.persistent import persistent_copy

ROWS = 16
# The launch configuration rides on the kernels so whole-forward compilation, which drops
# launch keywords, keeps it.
_CONFIG = triton.Config({}, num_warps=1, num_stages=2)


@triton.jit
def _gate(raw, bias, a_scale, LOWER: tl.constexpr):
    return tl.exp(LOWER / (1.0 + tl.exp(-a_scale * (raw + bias))))


@triton.autotune(configs=[_CONFIG], key=[])
@triton.jit
def _verify(Mixed, Gate, Beta, ALog, DtBias, State, Starts, Slots, NumSeqs, Output, Corrections, KeysGates,
            mixed_stride, gate_stride, beta_stride, state_stride, slot_stride, correction_stride, kg_stride,
            HEADS: tl.constexpr, SCALE: tl.constexpr, LOWER: tl.constexpr, WINDOW: tl.constexpr,
            ROWS: tl.constexpr, NULL_SLOT: tl.constexpr):
    tile, request_head = tl.program_id(0), tl.program_id(1)
    request = request_head // HEADS
    head = request_head % HEADS
    if request < tl.load(NumSeqs):
        start = tl.load(Starts + request)
        end = tl.load(Starts + request + 1)
        slot = tl.load(Slots + request * slot_stride).to(tl.int64)
        rows = tile * ROWS + tl.arange(0, ROWS)
        keys = tl.arange(0, 128)
        if slot == NULL_SLOT:
            # A null slot carries no state: its tokens decode to zero.
            for position in tl.static_range(WINDOW):
                if position < end - start:
                    token = (start + position).to(tl.int64)
                    tl.store(Output + token * (HEADS * 128) + head * 128 + rows, tl.zeros((ROWS,), tl.bfloat16))
        else:
            offsets = slot * state_stride + head * (128 * 128) + rows[:, None] * 128 + keys[None, :]
            state = tl.load(State + offsets)
            a_scale = tl.exp(tl.load(ALog + head).to(tl.float32))
            bias = tl.load(DtBias + head * 128 + keys).to(tl.float32)
            for position in tl.static_range(WINDOW):
                if position < end - start:
                    token = (start + position).to(tl.int64)
                    raw_q = tl.load(Mixed + token * mixed_stride + head * 128 + keys)
                    raw_k = tl.load(Mixed + token * mixed_stride + (HEADS + head) * 128 + keys)
                    q = raw_q.to(tl.float32)
                    k = raw_k.to(tl.float32)
                    q = q * (1.0 / tl.sqrt_rn(tl.sum(q * q, 0) + 1.0e-6)) * SCALE
                    k = k * (1.0 / tl.sqrt_rn(tl.sum(k * k, 0) + 1.0e-6))
                    raw_g = tl.load(Gate + token * gate_stride + head * 128 + keys)
                    decay = _gate(raw_g.to(tl.float32), bias, a_scale, LOWER)
                    beta = 1.0 / (1.0 + tl.exp(-tl.load(Beta + token * beta_stride + head).to(tl.float32)))
                    value = tl.load(Mixed + token * mixed_stride + (2 * HEADS + head) * 128 + rows).to(tl.float32)
                    state = state * decay[None, :]
                    delta = (value - tl.sum(state * k[None, :], 1)) * beta
                    state = state + delta[:, None] * k[None, :]
                    output = tl.sum(state * q[None, :], 1)
                    tl.store(Output + token * (HEADS * 128) + head * 128 + rows, output.to(tl.bfloat16))
                    record = (head * WINDOW + position).to(tl.int64)
                    tl.store(Corrections + slot * correction_stride + record * 128 + rows, delta)
                    if tile == 0:
                        tl.store(KeysGates + slot * kg_stride + record * 256 + keys, raw_k)
                        tl.store(KeysGates + slot * kg_stride + record * 256 + 128 + keys, raw_g)


@triton.autotune(configs=[_CONFIG], key=[])
@triton.jit
def _commit(Backing, LayerOffsets, Corrections, KeysGates, ALog, DtBias, Slots, Counts, NumSeqs,
            row_floats, slot_stride, SLOTS: tl.constexpr, HEADS: tl.constexpr, LOWER: tl.constexpr,
            WINDOW: tl.constexpr, ROWS: tl.constexpr, NULL_SLOT: tl.constexpr):
    # Every state a store reaches lies in the backing operand, so whole-forward compilation sees the
    # mutation: each layer's recurrent region sits at its offset in a row.
    tile, request, layer_head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    layer = layer_head // HEADS
    head = layer_head % HEADS
    if request < tl.load(NumSeqs):
        slot = tl.load(Slots + request * slot_stride).to(tl.int64)
        count = tl.load(Counts + request)
        if (slot != NULL_SLOT) & (count > 0):
            rows = tile * ROWS + tl.arange(0, ROWS)
            keys = tl.arange(0, 128)
            offsets = (tl.load(LayerOffsets + layer) + slot * row_floats + head * (128 * 128)
                       + rows[:, None] * 128 + keys[None, :])
            state = tl.load(Backing + offsets)
            a_scale = tl.exp(tl.load(ALog + layer * HEADS + head).to(tl.float32))
            bias = tl.load(DtBias + (layer * HEADS + head) * 128 + keys).to(tl.float32)
            # Records are stacked [layer, slot, head, position, 128 | 256].
            owner = (layer * SLOTS + slot) * HEADS + head
            for position in tl.static_range(WINDOW):
                if position < count:
                    record = owner * WINDOW + position
                    k = tl.load(KeysGates + record * 256 + keys).to(tl.float32)
                    k = k * (1.0 / tl.sqrt_rn(tl.sum(k * k, 0) + 1.0e-6))
                    decay = _gate(tl.load(KeysGates + record * 256 + 128 + keys).to(tl.float32), bias, a_scale,
                                  LOWER)
                    delta = tl.load(Corrections + record * 128 + rows)
                    state = state * decay[None, :]
                    state = state + delta[:, None] * k[None, :]
            tl.store(Backing + offsets, state)


class PreparedKDAVerification:
    """One KDA layer's recording verification over packed requests, before the gated output norm.

    Column zero of the verification metadata names each request's live state row.
    """

    def __init__(self, *, mixed_qkv, raw_g, raw_beta, A_log, dt_bias, state, metadata, output, records,
                 heads: int, lower_bound: float = -5.0, null_slot: int = 0):
        corrections, keys_gates = records
        sequences = metadata.query_start_loc.numel() - 1
        window = corrections.shape[2]
        if (state.ndim != 4 or state.shape[1:] != (heads, 128, 128) or state.dtype != torch.float32
                or state.stride()[1:] != (128 * 128, 128, 1) or output.shape[-1] != heads * 128
                or output.dtype != torch.bfloat16 or not output.is_contiguous()
                or corrections.shape[1:] != (heads, window, 128) or keys_gates.shape[1:] != (heads, window, 256)
                or corrections.stride()[1:] != (window * 128, 128, 1)
                or keys_gates.stride()[1:] != (window * 256, 256, 1)
                or metadata.state_indices.shape[1] > window or mixed_qkv.stride(1) != 1
                or raw_g.stride(-1) != 1 or raw_beta.stride(-1) != 1):
            raise ValueError("KDA verification requires packed FP32 state slots, contiguous BF16 output and "
                             "per-slot records for its head geometry")
        gate = raw_g.reshape(raw_g.shape[0], heads * 128)
        self.args = (mixed_qkv, gate, raw_beta, A_log, dt_bias, state, metadata.query_start_loc,
                     metadata.state_indices[:, 0], metadata.num_seqs, output, corrections, keys_gates,
                     mixed_qkv.stride(0), gate.stride(0), raw_beta.stride(0), state.stride(0),
                     metadata.state_indices.stride(0), corrections.stride(0), keys_gates.stride(0))
        self.options = {"HEADS": heads, "SCALE": 1.0 / math.sqrt(128), "LOWER": float(lower_bound),
                        "WINDOW": window, "ROWS": ROWS, "NULL_SLOT": null_slot}
        self.grid = (128 // ROWS, sequences * heads, 1)
        self.compiled = warmup_triton(_verify.fn, self.args, self.grid, num_warps=_CONFIG.num_warps,
                                      num_stages=_CONFIG.num_stages, **self.options)

    def functional(self):
        self.args = None
        return self

    def __call__(self, *, mixed_qkv=None, raw_g=None, raw_beta=None, A_log=None, dt_bias=None,
                 state=None, metadata=None, output=None, records=None):
        args = self.args
        if args is None:
            corrections, keys_gates = records
            gate = raw_g.reshape(raw_g.shape[0], -1)
            args = (mixed_qkv, gate, raw_beta, A_log, dt_bias, state, metadata.query_start_loc,
                    metadata.state_indices[:, 0], metadata.num_seqs, output, corrections, keys_gates,
                    mixed_qkv.stride(0), gate.stride(0), raw_beta.stride(0), state.stride(0),
                    metadata.state_indices.stride(0), corrections.stride(0), keys_gates.stride(0))
        launch_triton(_verify, self.compiled, args, self.grid, **self.options)


class PreparedKDAReplay:
    """Replay each request's accepted verification records into its live state for every KDA layer.

    ``backing`` holds the hybrid pool's byte rows; ``states`` are the recurrent views of the
    recorded layers, in record order.
    """

    def __init__(self, backing, states, records, slots, counts, num_seqs, *, lower_bound: float = -5.0,
                 null_slot: int = 0):
        floats = backing.view(torch.float32)
        rows = floats.shape[0]
        for state in states:
            if (state.untyped_storage().data_ptr() != backing.untyped_storage().data_ptr()
                    or state.stride(0) != floats.stride(0) or state.shape[0] != rows):
                raise ValueError("KDA replay requires recurrent views of the state backing rows")
        corrections = records.corrections
        if len(states) != corrections.shape[0] or not corrections.is_contiguous() \
                or not records.keys_and_gates.is_contiguous() or corrections.shape[1] != rows:
            raise ValueError("KDA replay requires contiguous records for every layer and state row")
        layers, _, heads, window, _ = corrections.shape
        offsets = persistent_copy(torch.tensor([state.storage_offset() - floats.storage_offset() for state in states],
                               dtype=torch.int64, device=backing.device))
        self.args = (floats, offsets, corrections, records.keys_and_gates, records.A_log, records.dt_bias,
                     slots, counts, num_seqs, floats.stride(0), slots.stride(0))
        self.options = {"SLOTS": rows, "HEADS": heads, "LOWER": float(lower_bound), "WINDOW": window,
                        "ROWS": ROWS, "NULL_SLOT": null_slot}
        self.grid = (128 // ROWS, counts.numel(), layers * heads)
        self.compiled = warmup_triton(_commit.fn, self.args, self.grid, num_warps=_CONFIG.num_warps,
                                      num_stages=_CONFIG.num_stages, **self.options)

    def functional(self):
        self.args = tuple(None if index in (2, 3, 6, 7, 8) else value
                          for index, value in enumerate(self.args))
        return self

    def __call__(self, counts, records, metadata):
        args = list(self.args)
        slots = metadata.state_indices[:, 0]
        args[2], args[3], args[6], args[7], args[8] = (
            records.corrections, records.keys_and_gates, slots, counts, metadata.num_seqs)
        args[-1] = slots.stride(0)
        launch_triton(_commit, self.compiled, tuple(args), self.grid, **self.options)
