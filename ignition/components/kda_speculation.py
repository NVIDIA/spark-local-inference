# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recording KDA verification and accepted-prefix replay with one committed state per row."""

import itertools
import weakref
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from b12x.sequence import gdn_decode

from ignition.preparation.persistent import persistent_copy
from ignition.preparation.tuning import register_finalizer
from b12x.preparation.types import plan_from_handle


@dataclass(frozen=True)
class KDARecords:
    """Verification records of every KDA layer, so speculative rounds keep one recurrent row per request.

    Verification leaves each request's committed FP32 state unchanged and records, per verified
    token, its FP32 delta-rule correction and BF16 raw key and gate; accepting a prefix replays
    those records into the committed row for every layer in one launch. Records are indexed by
    state row like the recurrent pool. ``A_log`` and ``dt_bias`` stack every layer's gate
    parameters for that replay.
    """

    layers: tuple[int, ...]
    corrections: torch.Tensor
    keys_and_gates: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor

    @classmethod
    def allocate(cls, attention_weights, slots, heads, device, *, window=8):
        """``attention_weights`` maps each KDA layer's decoder index to its attention weights."""
        layers = tuple(sorted(attention_weights))
        count = len(layers)
        return cls(layers,
            torch.zeros((count, slots, heads, window, 128), dtype=torch.float32, device=device),
            torch.zeros((count, slots, heads, window, 256), dtype=torch.bfloat16, device=device),
            persistent_copy(torch.stack([attention_weights[layer]["A_log"] for layer in layers])),
            persistent_copy(torch.stack([attention_weights[layer]["dt_bias"].view(heads, 128) for layer in layers])))

    def layer(self, index):
        position = self.layers.index(index)
        return self.corrections[position], self.keys_and_gates[position]


_OWNERS = weakref.WeakValueDictionary()
_HANDLES = itertools.count()


@torch.library.custom_op("flare::kda_record", mutates_args=("output", "corrections", "keys_gates"))
def _record(reads: list[torch.Tensor], recurrent_state: torch.Tensor, output: torch.Tensor,
            corrections: torch.Tensor, keys_gates: torch.Tensor, scratch: torch.Tensor,
            handle: int) -> None:
    owner = _OWNERS[handle]
    owner.run_recorded(reads, recurrent_state, output, corrections, keys_gates, scratch)


@_record.register_fake
def _fake_record(reads, recurrent_state, output, corrections, keys_gates, scratch, handle):
    pass


def register_recording(owner):
    handle = next(_HANDLES)
    _OWNERS[handle] = owner
    return handle


@triton.jit
def _address_tables(Backing, Corrections, KeysGates, StateBase, StateStride,
                    CorrectionBase, CorrectionStride, KeysGatesBase, KeysGatesStride,
                    OFFSETS: tl.constexpr, STATE_STRIDES: tl.constexpr,
                    CORRECTION_LAYER_STRIDE: tl.constexpr, CORRECTION_BLOCK_STRIDE: tl.constexpr,
                    CORRECTION_ELEMENT_BYTES: tl.constexpr,
                    KEYS_GATES_LAYER_STRIDE: tl.constexpr, KEYS_GATES_BLOCK_STRIDE: tl.constexpr,
                    KEYS_GATES_ELEMENT_BYTES: tl.constexpr,
                    LAYERS: tl.constexpr):
    for layer in tl.static_range(LAYERS):
        backing_address = tl.cast(Backing, tl.uint64)
        correction_address = tl.cast(Corrections, tl.uint64)
        keys_gates_address = tl.cast(KeysGates, tl.uint64)
        tl.store(StateBase + layer, backing_address + OFFSETS[layer])
        tl.store(StateStride + layer, STATE_STRIDES[layer])
        tl.store(CorrectionBase + layer,
                 correction_address + layer * CORRECTION_LAYER_STRIDE * CORRECTION_ELEMENT_BYTES)
        tl.store(CorrectionStride + layer, CORRECTION_BLOCK_STRIDE)
        tl.store(KeysGatesBase + layer,
                 keys_gates_address + layer * KEYS_GATES_LAYER_STRIDE * KEYS_GATES_ELEMENT_BYTES)
        tl.store(KeysGatesStride + layer, KEYS_GATES_BLOCK_STRIDE)


@torch.library.custom_op("flare::kda_commit_addresses", mutates_args=("tables",))
def _pack_kda_addresses(backing: torch.Tensor, corrections: torch.Tensor, keys_gates: torch.Tensor,
                        tables: list[torch.Tensor], offsets: list[int], state_strides: list[int],
                        correction_layer_stride: int, correction_block_stride: int,
                        correction_element_bytes: int, keys_gates_layer_stride: int,
                        keys_gates_block_stride: int, keys_gates_element_bytes: int) -> None:
    _address_tables[(1,)](backing, corrections, keys_gates, *tables, tuple(offsets), tuple(state_strides),
        correction_layer_stride, correction_block_stride, correction_element_bytes,
        keys_gates_layer_stride, keys_gates_block_stride, keys_gates_element_bytes,
        len(offsets), num_warps=1)


@_pack_kda_addresses.register_fake
def _pack_kda_addresses_fake(backing, corrections, keys_gates, tables, offsets, state_strides,
                             correction_layer_stride, correction_block_stride, correction_element_bytes,
                             keys_gates_layer_stride, keys_gates_block_stride, keys_gates_element_bytes):
    pass


@torch.library.custom_op("flare::kda_commit", mutates_args=("backing",))
def _execute_kda_commit(backing: torch.Tensor, corrections: torch.Tensor, keys_gates: torch.Tensor,
                        tables: list[torch.Tensor], A_log: torch.Tensor, dt_bias: torch.Tensor,
                        state_indices: torch.Tensor, commit_lens: torch.Tensor,
                        final_state_indices: torch.Tensor, boundary_state_indices: torch.Tensor,
                        boundary_recovery_lens: torch.Tensor, scratch: torch.Tensor, handle: int) -> None:
    binding = gdn_decode.bind_kda_commit(plan_from_handle(handle), scratch=scratch,
        state_base_addrs=tables[0], state_block_strides=tables[1],
        correction_cache_base_addrs=tables[2], correction_cache_block_strides=tables[3],
        kg_cache_base_addrs=tables[4], kg_cache_block_strides=tables[5],
        A_log=A_log, dt_bias=dt_bias, state_indices=state_indices, commit_lens=commit_lens,
        final_state_indices=final_state_indices, boundary_state_indices=boundary_state_indices,
        boundary_recovery_lens=boundary_recovery_lens)
    gdn_decode.run_kda_commit(binding, lower_bound=-5.0)


@_execute_kda_commit.register_fake
def _execute_kda_commit_fake(backing, corrections, keys_gates, tables, A_log, dt_bias,
                             state_indices, commit_lens, final_state_indices,
                             boundary_state_indices, boundary_recovery_lens, scratch, handle):
    pass


class PreparedKDAReplay:
    def __init__(self, backing, states, records, plan, *, session):
        self.backing, self.states, self.plan = backing, tuple(states), plan
        self.layers = records.layers
        self.state_offsets = tuple(state.data_ptr() - backing.data_ptr() for state in states)
        self.state_strides = tuple(state.stride(0) for state in states)
        self.scratch_bytes = None
        self.session = session

    def functional(self):
        register_finalizer(self.session, self._finalize)
        return self

    def _finalize(self):
        self.scratch_bytes = sum(spec.nbytes for spec in self.plan.scratch_specs())

    def __call__(self, counts, records, metadata):
        state_indices = metadata.state_indices
        if state_indices.ndim == 2:
            state_indices = state_indices[:, 0]
        rows = state_indices.numel()
        device = self.backing.device
        corrections, kg = records.corrections, records.keys_and_gates

        tables = [torch.empty((len(self.states),), dtype=torch.int64, device=device) for _ in range(6)]
        _pack_kda_addresses(self.backing, corrections, kg, tables, list(self.state_offsets),
            list(self.state_strides), corrections.stride(0), corrections.stride(1), corrections.element_size(),
            kg.stride(0), kg.stride(1), kg.element_size())
        commit_lens = counts.to(dtype=torch.int32)
        final_state_indices = state_indices.to(dtype=torch.int32)
        boundaries = torch.zeros((rows,), dtype=torch.int32, device=device)
        boundary_lengths = torch.zeros_like(boundaries)
        scratch = torch.empty((self.scratch_bytes,), dtype=torch.uint8, device=device)
        _execute_kda_commit(self.backing, corrections, kg, tables, records.A_log, records.dt_bias,
            state_indices, commit_lens, final_state_indices, boundaries, boundary_lengths, scratch, self.plan.handle)
