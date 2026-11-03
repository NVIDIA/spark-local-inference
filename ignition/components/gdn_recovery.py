# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Gated DeltaNet verification with per-token correction records.

b12x's decode persists one FP32 recurrent checkpoint (heads x 128 x 128) per
verified token so that any accepted prefix can be resumed. Verification here
reads the committed state once and records, per token, the delta-rule
correction (one value vector per head), the normalized key and the decay.
After acceptance, one commit launch over every GDN layer replays the accepted
prefix into the committed slot, so each round reads and writes the recurrent
state once instead of writing it once per token.

The arithmetic follows b12x's CuTe Qwen recurrence: L2-normalized Q (scaled by
1/sqrt(128)) and K with epsilon 1e-6, decay exp(-exp(A_log) * softplus(a +
dt_bias)), the update gate sigmoid(b) rounded through BF16, then per token
S = decay * S; delta = (v - S k) * beta; S += delta k^T; o = S q.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from b12x.sequence._short_conv import _speculative_decode_short_conv_kernel

from ignition.preparation.triton import launch_triton, warmup_triton

# Verified tokens per round: the eight checkpoint columns of verification metadata.
WINDOW = 8
ROWS = 16
# One warp per request, value head and value-row tile. The launch configuration
# rides on the kernels so whole-forward compilation, which drops launch keywords, keeps it.
_CONFIG = triton.Config({}, num_warps=1, num_stages=2)
_CONV_CONFIG = triton.Config({}, num_warps=4, num_stages=2)


@triton.autotune(configs=[_CONFIG], key=[])
@triton.jit
def _prepare_inputs(Mixed, A, B, ALog, DtBias, Starts, Slots, NumSeqs, Queries, Keys, Decay, Beta,
                    mixed_stride, a_stride, b_stride, slot_stride,
                    KEY_HEADS: tl.constexpr, VALUE_HEADS: tl.constexpr, SCALE: tl.constexpr,
                    WINDOW: tl.constexpr, RECORD_WIDTH: tl.constexpr, GROUP: tl.constexpr):
    position, request_key = tl.program_id(0), tl.program_id(1)
    request = request_key // KEY_HEADS
    key_head = request_key % KEY_HEADS
    if request < tl.load(NumSeqs):
        start = tl.load(Starts + request)
        end = tl.load(Starts + request + 1)
        slot = tl.load(Slots + request * slot_stride)
        if (slot != 0) & (position < end - start):
            token = (start + position).to(tl.int64)
            keys = tl.arange(0, 128)
            q = tl.load(Mixed + token * mixed_stride + key_head * 128 + keys).to(tl.float32)
            k = tl.load(Mixed + token * mixed_stride + (KEY_HEADS + key_head) * 128 + keys).to(tl.float32)
            q = q * tl.rsqrt(tl.sum(q * q, 0) + 1.0e-6) * SCALE
            k = k * tl.rsqrt(tl.sum(k * k, 0) + 1.0e-6)
            tl.store(Queries + (request_key * WINDOW + position) * 128 + keys, q)
            tl.store(Keys + (request_key * RECORD_WIDTH + position) * 128 + keys, k)
            groups = tl.arange(0, GROUP)
            head = key_head * (VALUE_HEADS // KEY_HEADS) + groups
            live = groups < VALUE_HEADS // KEY_HEADS
            a_scale = tl.exp(tl.load(ALog + head, live, other=0).to(tl.float32))
            bias = tl.load(DtBias + head, live, other=0).to(tl.float32)
            a = tl.load(A + token * a_stride + head, live, other=0).to(tl.float32) + bias
            softplus = tl.where(a <= 20.0, libdevice.log1p(tl.exp(a)), a)
            decay = tl.exp(-a_scale * softplus)
            gate = tl.load(B + token * b_stride + head, live, other=0).to(tl.float32)
            beta = (1.0 / (1.0 + tl.exp(-gate))).to(tl.bfloat16)
            tl.store(Decay + (request * VALUE_HEADS + head) * RECORD_WIDTH + position, decay, live)
            tl.store(Beta + (request * VALUE_HEADS + head) * WINDOW + position, beta, live)


@triton.autotune(configs=[_CONV_CONFIG], key=[])
@triton.jit
def _prepare_convolved_inputs(Mixed, A, B, ALog, DtBias, Starts, Slots, NumSeqs, Queries, Keys, Decay, Beta,
                              mixed_stride, a_stride, b_stride, slot_stride,
                              Projected, Weight, ConvState, ConvSlots, Fresh, Accepted, NumTokens,
                              KEY_HEADS: tl.constexpr, VALUE_HEADS: tl.constexpr, SCALE: tl.constexpr,
                              WINDOW: tl.constexpr, RECORD_WIDTH: tl.constexpr,
                              INPUT_STRIDE: tl.constexpr, STATE_STRIDE: tl.constexpr,
                              STATE_CHANNEL_STRIDE: tl.constexpr, STATE_POSITION_STRIDE: tl.constexpr,
                              STATE_COLUMNS: tl.constexpr, MAX_SEQS: tl.constexpr, MAX_TOKENS: tl.constexpr, BLOCK_T: tl.constexpr):
    _speculative_decode_short_conv_kernel(
        Projected, Weight, Starts, ConvSlots, Fresh, Accepted, NumSeqs, NumTokens, ConvState, Mixed,
        CHANNELS=(2 * KEY_HEADS + VALUE_HEADS) * 128, STATE_LENGTH=3, STATE_CAPACITY=STATE_COLUMNS,
        stride_state_slot=STATE_STRIDE, stride_state_channel=STATE_CHANNEL_STRIDE,
        stride_state_position=STATE_POSITION_STRIDE,
        MAX_SPECULATIVE=STATE_COLUMNS - 3, KERNEL_SIZE=4, DILATION=1, INPUT_ROW_STRIDE=INPUT_STRIDE,
        MAX_SEQS=MAX_SEQS, MAX_TOKENS=MAX_TOKENS, BLOCK_C=128)
    # Token-row normalization redistributes channel values across producer warps.
    tl.debug_barrier()
    request, head = tl.program_id(0), tl.program_id(1)
    if request < tl.load(NumSeqs):
        slot = tl.load(Slots + request * slot_stride)
        start = tl.load(Starts + request)
        length = tl.load(Starts + request + 1) - start
        if slot != 0:
            positions = tl.arange(0, BLOCK_T)
            live = positions < length
            tokens = (start + positions).to(tl.int64)
            if head < 2 * KEY_HEADS:
                channel = tl.arange(0, 128)
                value = tl.load(Mixed + tokens[:, None] * mixed_stride + head * 128 + channel[None, :],
                                live[:, None], other=0).to(tl.float32)
                normalized = value * tl.rsqrt(tl.sum(value * value, 1) + 1.0e-6)[:, None]
                if head < KEY_HEADS:
                    tl.store(Queries + ((request * KEY_HEADS + head) * WINDOW + positions[:, None]) * 128
                             + channel[None, :], normalized * SCALE, live[:, None])
                else:
                    tl.store(Keys + ((request * KEY_HEADS + head - KEY_HEADS) * RECORD_WIDTH
                             + positions[:, None]) * 128 + channel[None, :], normalized, live[:, None])
            else:
                value_head = head - 2 * KEY_HEADS
                a_scale = tl.exp(tl.load(ALog + value_head).to(tl.float32))
                bias = tl.load(DtBias + value_head).to(tl.float32)
                a = tl.load(A + tokens * a_stride + value_head, live, other=0).to(tl.float32) + bias
                softplus = tl.where(a <= 20.0, libdevice.log1p(tl.exp(a)), a)
                decay = tl.exp(-a_scale * softplus)
                gate = tl.load(B + tokens * b_stride + value_head, live, other=0).to(tl.float32)
                beta = (1.0 / (1.0 + tl.exp(-gate))).to(tl.bfloat16)
                tl.store(Decay + (request * VALUE_HEADS + value_head) * RECORD_WIDTH + positions, decay, live)
                tl.store(Beta + (request * VALUE_HEADS + value_head) * WINDOW + positions, beta, live)


@triton.autotune(configs=[_CONFIG], key=[])
@triton.jit
def _verify(Mixed, A, B, ALog, DtBias, State, Starts, Slots, NumSeqs, Output, Delta, Keys, Decay, Queries, Beta,
            mixed_stride, a_stride, b_stride, state_stride, slot_stride,
            KEY_HEADS: tl.constexpr, VALUE_HEADS: tl.constexpr, SCALE: tl.constexpr,
            WINDOW: tl.constexpr, RECORD_WIDTH: tl.constexpr, ROWS: tl.constexpr, IN_PLACE: tl.constexpr, PRECOMPUTED: tl.constexpr):
    tile, request_head = tl.program_id(0), tl.program_id(1)
    request = request_head // VALUE_HEADS
    head = request_head % VALUE_HEADS
    key_head = head // (VALUE_HEADS // KEY_HEADS)
    if request < tl.load(NumSeqs):
        start = tl.load(Starts + request)
        end = tl.load(Starts + request + 1)
        slot = tl.load(Slots + request * slot_stride).to(tl.int64)
        rows = tile * ROWS + tl.arange(0, ROWS)
        keys = tl.arange(0, 128)
        if slot == 0:
            # A null slot carries no state: its tokens decode to zero.
            for position in tl.static_range(WINDOW):
                if position < end - start:
                    token = (start + position).to(tl.int64)
                    tl.store(Output + token * (VALUE_HEADS * 128) + head * 128 + rows,
                             tl.zeros((ROWS,), tl.bfloat16))
        else:
            offsets = slot * state_stride + head * (128 * 128) + rows[:, None] * 128 + keys[None, :]
            state = tl.load(State + offsets)
            if not PRECOMPUTED:
                a_scale = tl.exp(tl.load(ALog + head).to(tl.float32))
                bias = tl.load(DtBias + head).to(tl.float32)
            for position in tl.range(0, WINDOW, loop_unroll_factor=1):
                if position < end - start:
                    token = (start + position).to(tl.int64)
                    if PRECOMPUTED:
                        q = tl.load(Queries + ((request * KEY_HEADS + key_head) * WINDOW + position) * 128 + keys)
                        k = tl.load(Keys + ((request * KEY_HEADS + key_head) * RECORD_WIDTH + position) * 128 + keys)
                        decay = tl.load(Decay + (request * VALUE_HEADS + head) * RECORD_WIDTH + position)
                        beta = tl.load(Beta + (request * VALUE_HEADS + head) * WINDOW + position).to(tl.float32)
                    else:
                        q = tl.load(Mixed + token * mixed_stride + key_head * 128 + keys).to(tl.float32)
                        k = tl.load(Mixed + token * mixed_stride + KEY_HEADS * 128 + key_head * 128 + keys).to(tl.float32)
                        q = q * tl.rsqrt(tl.sum(q * q, 0) + 1.0e-6) * SCALE
                        k = k * tl.rsqrt(tl.sum(k * k, 0) + 1.0e-6)
                        a = tl.load(A + token * a_stride + head).to(tl.float32) + bias
                        softplus = tl.where(a <= 20.0, libdevice.log1p(tl.exp(a)), a)
                        decay = tl.exp(-a_scale * softplus)
                        gate = tl.load(B + token * b_stride + head).to(tl.float32)
                        beta = (1.0 / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
                    value = tl.load(Mixed + token * mixed_stride + 2 * KEY_HEADS * 128 + head * 128 + rows).to(tl.float32)
                    state = state * decay
                    delta = (value - tl.sum(state * k[None, :], 1)) * beta
                    state = state + delta[:, None] * k[None, :]
                    output = tl.sum(state * q[None, :], 1)
                    tl.store(Output + token * (VALUE_HEADS * 128) + head * 128 + rows, output.to(tl.bfloat16))
                    if not IN_PLACE:
                        # Records are indexed by the request's row in the round, which the commit shares.
                        tl.store(Delta + ((request * VALUE_HEADS + head) * RECORD_WIDTH + position) * 128 + rows, delta)
                        if not PRECOMPUTED:
                            if (tile == 0) & (head % (VALUE_HEADS // KEY_HEADS) == 0):
                                tl.store(Keys + ((request * KEY_HEADS + key_head) * RECORD_WIDTH + position) * 128 + keys, k)
                            if tile == 0:
                                tl.store(Decay + (request * VALUE_HEADS + head) * RECORD_WIDTH + position, decay)
            if IN_PLACE:
                tl.store(State + offsets, state)


@triton.autotune(configs=[_CONFIG], key=[])
@triton.jit
def _commit(State, LayerOffsets, Delta, KeyRecords, Decay, Slots, Counts, NumSeqs, state_stride, slot_stride,
            ROWS_CAPACITY: tl.constexpr, KEY_HEADS: tl.constexpr, VALUE_HEADS: tl.constexpr, WINDOW: tl.constexpr, RECORD_WIDTH: tl.constexpr,
            ROWS: tl.constexpr, STATE_ALIGNMENT: tl.constexpr):
    # Every argument a store reaches is a kernel operand, so whole-forward compilation sees the
    # mutation of the state backing: each layer's recurrent region sits at its offset in a row.
    tile, request, layer_head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    layer = layer_head // VALUE_HEADS
    head = layer_head % VALUE_HEADS
    key_head = head // (VALUE_HEADS // KEY_HEADS)
    if request < tl.load(NumSeqs):
        slot = tl.load(Slots + request * slot_stride).to(tl.int64)
        count = tl.load(Counts + request)
        if (slot != 0) & (count > 0):
            rows = tile * ROWS + tl.arange(0, ROWS)
            keys = tl.arange(0, 128)
            base = tl.multiple_of(tl.load(LayerOffsets + layer) + slot * state_stride, STATE_ALIGNMENT)
            offsets = base + head * (128 * 128) + rows[:, None] * 128 + keys[None, :]
            state = tl.load(State + offsets)
            # Records are stacked [layer, request row, head, position, 128].
            owner = layer * ROWS_CAPACITY + request
            for position in tl.range(0, WINDOW, loop_unroll_factor=1):
                if position < count:
                    k = tl.load(KeyRecords + ((owner * KEY_HEADS + key_head) * RECORD_WIDTH + position) * 128 + keys)
                    delta = tl.load(Delta + ((owner * VALUE_HEADS + head) * RECORD_WIDTH + position) * 128 + rows)
                    decay = tl.load(Decay + (owner * VALUE_HEADS + head) * RECORD_WIDTH + position)
                    state = state * decay
                    state = state + delta[:, None] * k[None, :]
            tl.store(State + offsets, state)


@dataclass(frozen=True)
class GDNRecords:
    """Per-request correction, normalized key and decay records of one GDN layer's verified tokens."""

    delta: torch.Tensor
    keys: torch.Tensor
    decay: torch.Tensor


@dataclass(frozen=True)
class GDNLayerRecords:
    """Records of every GDN layer in stacked tensors, with one view per layer for verification."""

    layers: tuple[int, ...]
    delta: torch.Tensor
    keys: torch.Tensor
    decay: torch.Tensor

    @classmethod
    def allocate(cls, layers, rows: int, device, *, key_heads: int, value_heads: int, width: int = WINDOW) -> GDNLayerRecords:
        if type(width) is not int or not 1 <= width <= 16:
            raise ValueError("GDN record width must be in [1,16]")
        layers = tuple(layers)
        count = len(layers)
        return cls(layers,
                   torch.zeros((count, rows, value_heads, width, 128), dtype=torch.float32, device=device),
                   torch.zeros((count, rows, key_heads, width, 128), dtype=torch.float32, device=device),
                   torch.zeros((count, rows, value_heads, width), dtype=torch.float32, device=device))

    def __getitem__(self, layer: int) -> GDNRecords:
        index = self.layers.index(layer)
        return GDNRecords(self.delta[index], self.keys[index], self.decay[index])


class PreparedRecoveredRecurrence:
    """One layer's verification over packed requests, recording instead of checkpointing.

    Column zero of the verification metadata names each request's committed state
    row; records are indexed by the request's row in the round, so they hold one
    entry per request the round can verify.
    Output is the recurrence before the gated output norm.

    Without records, the recurrence advances the committed state in place: a
    single-token round has no drafts, so its token is accepted whenever it is emitted.
    """

    def __init__(self, *, mixed_qkv, a, b, A_log, dt_bias, state, metadata, output, records: GDNRecords | None,
                 key_heads: int, value_heads: int, convolution=None):
        sequences = metadata.query_start_loc.numel() - 1
        width = metadata.state_indices.shape[1]
        record_width = width if records is None else records.delta.shape[2]
        if (state.ndim != 4 or state.shape[1:] != (value_heads, 128, 128) or state.dtype != torch.float32
                or state.stride()[1:] != (128 * 128, 128, 1) or output.shape[-1] != value_heads * 128
                or output.dtype != torch.bfloat16 or not output.is_contiguous()
                or (records is not None and (records.delta.shape[1:] != (value_heads, record_width, 128)
                    or records.keys.shape[1:] != (key_heads, record_width, 128)
                    or records.keys.shape[0] != records.delta.shape[0] or records.delta.shape[0] < sequences))
                or not 1 <= width <= record_width <= 16 or value_heads % key_heads
                or mixed_qkv.stride(1) != 1 or a.stride(1) != 1 or b.stride(1) != 1):
            raise ValueError("recovered GDN verification requires packed FP32 state slots, contiguous BF16 "
                             "output and per-request records for its head geometry")
        slots = metadata.state_indices[:, 0]
        # In place, the record operands are never stored through; the state stands in for them.
        delta, keys, decay = (state, state, state) if records is None else (records.delta, records.keys, records.decay)
        precomputed = records is not None and width > 1
        if convolution is not None and not precomputed:
            raise ValueError("fused GDN convolution requires multi-token verification records")
        queries = torch.empty((sequences, key_heads, width, 128), device=state.device, dtype=torch.float32) if precomputed else state
        beta = torch.empty((sequences, value_heads, width), device=state.device, dtype=torch.bfloat16) if precomputed else state
        self.prepare_args = None
        self.fused_convolution = convolution is not None
        if precomputed:
            prepare_kernel = _prepare_inputs
            prepare_config = _CONFIG
            self.prepare_args = (mixed_qkv, a, b, A_log, dt_bias, metadata.query_start_loc, slots,
                                 metadata.num_seqs, queries, keys, decay, beta, mixed_qkv.stride(0),
                                 a.stride(0), b.stride(0), metadata.state_indices.stride(0))
            self.prepare_options = {"KEY_HEADS": key_heads, "VALUE_HEADS": value_heads,
                                    "SCALE": 1.0 / math.sqrt(128), "WINDOW": width, "RECORD_WIDTH": record_width,
                                    "GROUP": triton.next_power_of_2(value_heads // key_heads)}
            self.prepare_grid = (width, sequences * key_heads, 1)
            if convolution is not None:
                projected, weight, history = (convolution[name] for name in ("x", "conv_weight", "conv_state"))
                channels = (2 * key_heads + value_heads) * 128
                if (projected.shape != mixed_qkv.shape or projected.dtype != torch.bfloat16 or mixed_qkv.dtype != torch.bfloat16
                        or projected.stride(1) != 1 or not mixed_qkv.is_contiguous()
                        or weight.shape != (channels, 4) or weight.dtype != torch.bfloat16 or not weight.is_contiguous()
                        or history.ndim != 3 or history.shape[1] != channels
                        or not width + 2 <= history.shape[2] <= 18 or history.dtype != torch.bfloat16
                        or convolution["out"] is not mixed_qkv
                        or convolution["query_start_loc"] is not metadata.query_start_loc
                        or convolution["num_seqs"] is not metadata.num_seqs):
                    raise ValueError("fused GDN convolution requires packed BF16 projections, four-tap weights "
                                     "and shared packed-request metadata")
                prepare_kernel = _prepare_convolved_inputs
                prepare_config = _CONV_CONFIG
                self.prepare_args += (projected, weight, history, convolution["state_slot_ids"],
                    convolution["state_is_fresh"], convolution["num_accepted_tokens"], convolution["num_tokens"])
                self.prepare_options.pop("GROUP")
                self.prepare_options.update(INPUT_STRIDE=projected.stride(0), STATE_STRIDE=history.stride(0),
                    STATE_CHANNEL_STRIDE=history.stride(1), STATE_POSITION_STRIDE=history.stride(2),
                    STATE_COLUMNS=history.shape[2], MAX_SEQS=sequences, MAX_TOKENS=projected.shape[0],
                    BLOCK_T=triton.next_power_of_2(width))
                self.prepare_grid = (sequences + 1, 2 * key_heads + value_heads, 1)
            self.prepare_compiled = warmup_triton(prepare_kernel.fn, self.prepare_args, self.prepare_grid,
                                                num_warps=prepare_config.num_warps, num_stages=prepare_config.num_stages,
                                                **self.prepare_options)
        self.args = (mixed_qkv, a, b, A_log, dt_bias, state, metadata.query_start_loc, slots, metadata.num_seqs,
                     output, delta, keys, decay, queries, beta, mixed_qkv.stride(0), a.stride(0),
                     b.stride(0), state.stride(0), metadata.state_indices.stride(0))
        verify_rows = 8 if precomputed or sequences * value_heads >= 192 else 4
        self.options = {"KEY_HEADS": key_heads, "VALUE_HEADS": value_heads, "SCALE": 1.0 / math.sqrt(128),
                        "WINDOW": width, "RECORD_WIDTH": record_width, "ROWS": verify_rows, "IN_PLACE": records is None, "PRECOMPUTED": precomputed}
        self.grid = (128 // verify_rows, sequences * value_heads, 1)
        self.compiled = warmup_triton(_verify.fn, self.args, self.grid, num_warps=_CONFIG.num_warps,
                                      num_stages=_CONFIG.num_stages, **self.options)

    def __call__(self):
        if self.prepare_args is not None:
            if self.fused_convolution:
                launch_triton(_prepare_convolved_inputs, self.prepare_compiled, self.prepare_args,
                              self.prepare_grid, **self.prepare_options)
            else:
                launch_triton(_prepare_inputs, self.prepare_compiled, self.prepare_args,
                              self.prepare_grid, **self.prepare_options)
        launch_triton(_verify, self.compiled, self.args, self.grid, **self.options)


class PreparedGDNCommit:
    """Replay each request's accepted verification records into its committed state, for every layer.

    ``backing`` is the hybrid state pool's byte rows; ``states`` are its per-layer recurrent views.
    """

    def __init__(self, backing: torch.Tensor, states: Mapping[int, torch.Tensor], records: GDNLayerRecords,
                 metadata, counts: torch.Tensor, *, key_heads: int, value_heads: int):
        if set(states) != set(records.layers) or not records.layers:
            raise ValueError("GDN commit requires records for every committed layer")
        floats = backing.view(torch.float32)
        rows = floats.shape[0]
        for layer in records.layers:
            state = states[layer]
            if (state.untyped_storage().data_ptr() != backing.untyped_storage().data_ptr()
                    or state.stride(0) != floats.stride(0) or state.shape[0] != rows):
                raise ValueError("GDN commit requires recurrent views of the state backing rows")
        if counts.shape != (metadata.query_start_loc.numel() - 1,) or counts.numel() > records.delta.shape[1]:
            raise ValueError("GDN commit requires a count and records per request")
        layer_offsets = [states[layer].storage_offset() - floats.storage_offset() for layer in records.layers]
        state_alignment = math.gcd(4, floats.data_ptr() // 4, floats.stride(0), *layer_offsets)
        offsets = torch.tensor(layer_offsets, dtype=torch.int64, device=backing.device)
        self.args = (floats, offsets, records.delta, records.keys, records.decay, metadata.state_indices[:, 0],
                     counts, metadata.num_seqs, floats.stride(0), metadata.state_indices.stride(0))
        self.options = {"ROWS_CAPACITY": records.delta.shape[1], "KEY_HEADS": key_heads, "VALUE_HEADS": value_heads,
                        "WINDOW": metadata.state_indices.shape[1], "RECORD_WIDTH": records.delta.shape[3], "ROWS": ROWS, "STATE_ALIGNMENT": state_alignment}
        self.grid = (128 // ROWS, counts.numel(), len(records.layers) * value_heads)
        self.compiled = warmup_triton(_commit.fn, self.args, self.grid, num_warps=_CONFIG.num_warps,
                                      num_stages=_CONFIG.num_stages, **self.options)

    def __call__(self):
        launch_triton(_commit, self.compiled, self.args, self.grid, **self.options)
