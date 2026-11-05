# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Contiguous hybrid-model snapshots and captured indirect state transfers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType

import torch
import triton
import triton.language as tl

from ignition.components.state import StateLayout, StateStorage

from ignition.components.gated_delta import GDNState

GDN_LAYERS = tuple(layer for layer in range(48) if layer % 4 != 3)
QSA_LAYERS = tuple(range(3, 48, 4))
EOS_TOKEN_ID = 248044
COPY = 0
RESET = 1


@dataclass(frozen=True)
class HybridStateLayout(StateLayout):
    """Byte layout for one complete consumed-prefix continuation state."""

    speculative_tokens: int = 0
    draft_ring_capacity: int = 0

    @classmethod
    def build(cls, *, speculative_tokens: int = 0, world_size: int = 1, draft_config=None) -> HybridStateLayout:
        if type(speculative_tokens) is not int or not 0 <= speculative_tokens <= 7:
            raise ValueError("hybrid state supports zero through seven speculative tokens")
        from ignition.components.parallel import TensorParallelShard

        TensorParallelShard(0, world_size)
        specifications = []

        def add(name, shape, dtype):
            specifications.append((name, shape, dtype))

        for layer in GDN_LAYERS:
            add(f"gdn.{layer}.recurrent", (48 // world_size, 128, 128), torch.float32)
            add(f"gdn.{layer}.convolution", (10240 // world_size, 3 + speculative_tokens), torch.bfloat16)
        add("ple_convolution", (10240, 9 + speculative_tokens), torch.bfloat16)
        add("last_hidden", (2560,), torch.bfloat16)
        add("consumed_tokens", (1,), torch.int64)
        add("ple_history", (2,), torch.int64)
        add("rope_delta", (1,), torch.int64)
        from b12x.attention import qsa

        geometry = qsa.cache_requirements(main_page_size=64, position_axes=3, kv_heads=max(1, 2 // world_size),
                                         max_speculative_tokens=speculative_tokens, kv_dtype=torch.float8_e4m3fn)
        for layer in QSA_LAYERS:
            add(f"qsa.{layer}.raw_keys", geometry.raw_k_ring_shape, torch.bfloat16)
            add(f"qsa.{layer}.raw_positions", geometry.raw_logical_positions_shape, torch.int64)
            add(f"qsa.{layer}.raw_rope", geometry.raw_rope_positions_shape, torch.int64)
            add(f"qsa.{layer}.raw_anchor", (1,), torch.int64)
            add(f"qsa.{layer}.boundary_key", geometry.main_k_page_shape, geometry.kv_dtype)
            add(f"qsa.{layer}.boundary_value", geometry.main_v_page_shape, geometry.kv_dtype)
            add(f"qsa.{layer}.boundary_selector", geometry.compressed_page_shape, torch.bfloat16)
        ring = 0
        if speculative_tokens and draft_config is None:
            ring = geometry.raw_ring_capacity
            add("target_residual", (10240,), torch.bfloat16)
            add("target_residual_position", (3,), torch.int64)
            add("draft_raw_keys", geometry.raw_k_ring_shape, torch.bfloat16)
            add("draft_raw_positions", geometry.raw_logical_positions_shape, torch.int64)
            add("draft_raw_rope", geometry.raw_rope_positions_shape, torch.int64)
            add("draft_raw_anchor", (1,), torch.int64)
            add("draft_boundary_key", geometry.main_k_page_shape, geometry.kv_dtype)
            add("draft_boundary_value", geometry.main_v_page_shape, geometry.kv_dtype)
            add("draft_boundary_selector", geometry.compressed_page_shape, torch.bfloat16)
            add("draft_boundary_key_extra", geometry.main_k_page_shape, geometry.kv_dtype)
            add("draft_boundary_value_extra", geometry.main_v_page_shape, geometry.kv_dtype)
            add("draft_boundary_selector_extra", geometry.compressed_page_shape, torch.bfloat16)
        if draft_config is not None:
            from ignition.components.dflash2_geometry import draft_context_specs

            specifications.extend(draft_context_specs(draft_config, TensorParallelShard(0, world_size)))
        layout = StateLayout.from_specs(specifications)
        return cls(layout.regions, layout.row_bytes, speculative_tokens, ring)


@dataclass(frozen=True)
class HybridStatePool(StateStorage):
    """Typed disjoint views into one persistent byte row per state slot.

    Slot zero is permanently zero. Helios partitions the backed rows between
    request execution state and immutable prefix snapshots at load time.
    Snapshots carry the exact consumed endpoint, normalized recurrent/history
    state, target raw selector state and partial pages. Speculative snapshots
    also carry independently aligned draft state for one fewer consumed token.
    Sampling RNG, penalties and pending tokens remain request-local.
    """

    backing: torch.Tensor
    layout: HybridStateLayout
    memory_capacity: torch.Tensor
    gdn: Mapping[int, GDNState]
    ple_convolution: torch.Tensor
    last_hidden: torch.Tensor
    consumed_tokens: torch.Tensor
    ple_history: torch.Tensor
    rope_delta: torch.Tensor
    backed_slots: int | None = None
    reserved_backing: torch.Tensor | None = None

    @classmethod
    def allocate(cls, slots: int, device, *, speculative_tokens: int = 0, world_size: int = 1,
                 storage=None, backed_slots: int | None = None, draft_config=None) -> HybridStatePool:
        if type(slots) is not int or slots < 2:
            raise ValueError("hybrid state requires null and live slots")
        backed_slots = slots if backed_slots is None else backed_slots
        if type(backed_slots) is not int or not 2 <= backed_slots <= slots:
            raise ValueError("backed state slots must include null and live slots within the virtual extent")
        if backed_slots != slots and storage is None:
            raise ValueError("partial state backing requires package storage")
        layout = HybridStateLayout.build(speculative_tokens=speculative_tokens, world_size=world_size, draft_config=draft_config)
        backing = (torch.zeros((slots, layout.row_bytes), dtype=torch.uint8, device=device) if storage is None else
            storage.device_allocate("continuations", (slots, layout.row_bytes), torch.uint8, group="STATE_ROWS",
                                    maximum=torch.cuda.mem_get_info(device)[1] // layout.row_bytes, units=backed_slots))

        reserved = backing
        if storage is not None:
            from b12x.sequence._shared.disk_table import _tensor_from_pointer

            region = next(region for region in storage.dynamic_regions if region.address == backing.data_ptr())
            reserved = _tensor_from_pointer(region.address, shape=(region.maximum, layout.row_bytes),
                dtype=torch.uint8, device=device, nbytes=region.maximum * layout.row_bytes)

        def view(name):
            region = layout.region(name)
            return (
                backing[:, region.offset : region.offset + region.nbytes]
                .view(region.dtype)
                .view(slots, *region.shape)
            )

        gdn = MappingProxyType(
            {
                layer: GDNState(
                    view(f"gdn.{layer}.convolution"), view(f"gdn.{layer}.recurrent")
                )
                for layer in GDN_LAYERS
            }
        )
        history = view("ple_history")
        history[1:backed_slots].fill_(EOS_TOKEN_ID)
        return cls(
            backing,
            layout,
            torch.tensor([backed_slots, 0], dtype=torch.int64, device=device),
            gdn,
            view("ple_convolution"),
            view("last_hidden"),
            view("consumed_tokens"),
            history,
            view("rope_delta"),
            backed_slots,
            reserved,
        )

    def reserved_region(self, name):
        region = self.layout.region(name)
        backing = self.backing if self.reserved_backing is None else self.reserved_backing
        return backing[:, region.offset:region.offset + region.nbytes].view(region.dtype).view(
            backing.shape[0], *region.shape)

    def reset_for_preparation(self):
        """Restore empty unpublished model state after serialized benchmark use."""
        self.backing[:self.backed_slots].zero_()
        self.ple_history[1:self.backed_slots].fill_(EOS_TOKEN_ID)


@dataclass(frozen=True)
class StateTransferMetadata:
    """Device commands for complete-state copy/reset and exact cache restoration.

    Active exact commands require expected_consumed=0 for RESET and the saved
    positive consumed count for COPY. Inactive commands use -1. Error codes are
    1=operation, 2=slot bounds, 3=raw-slot bounds, 4=endpoint, 5=accepted count,
    6=batch alias conflict, 7=command count and 8=page mapping. Destinations must
    be unique and cannot alias another operation's source.
    """

    request_slots: torch.Tensor
    source_slots: torch.Tensor
    destination_slots: torch.Tensor
    operation: torch.Tensor
    qsa_slots: torch.Tensor
    accepted_tokens: torch.Tensor
    expected_consumed: torch.Tensor
    num_ops: torch.Tensor
    errors: torch.Tensor

    @classmethod
    def allocate(cls, capacity: int, device) -> StateTransferMetadata:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("state-transfer capacity must be positive")
        return cls(
            torch.full((capacity,), -1, dtype=torch.int64, device=device),
            torch.full((capacity,), -1, dtype=torch.int64, device=device),
            torch.full((capacity,), -1, dtype=torch.int64, device=device),
            torch.zeros(capacity, dtype=torch.int32, device=device),
            torch.full((capacity,), -1, dtype=torch.int64, device=device),
            torch.ones(capacity, dtype=torch.int32, device=device),
            torch.full((capacity,), -1, dtype=torch.int64, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.zeros(capacity, dtype=torch.int32, device=device),
        )


@triton.jit
def _transfer(
    Backing,
    Consumed,
    Sources,
    Destinations,
    Operations,
    RawSlots,
    Accepted,
    Count,
    Errors,
    Expected,
    PreviousAccepted,
    CopyLayout,
    Tables,
    DraftTables,
    Requests,
    StateRows,
    EXACT: tl.constexpr,
    SPECULATIVE: tl.constexpr,
    HAS_DRAFT: tl.constexpr,
    MemoryCapacity,
    TABLE_WIDTH: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    CAPACITY: tl.constexpr,
    RAW_SLOTS: tl.constexpr,
    HISTORY_OFFSET: tl.constexpr,
    CONSUMED_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    OP_BLOCK: tl.constexpr,
    LIVE: tl.constexpr = 4,
    EXCLUDED: tl.constexpr = (),
):
    command, tile = tl.program_id(0), tl.program_id(1)
    count = tl.load(Count)
    SLOTS = tl.load(MemoryCapacity)
    PAGE_COUNT = tl.load(MemoryCapacity + 1)
    source = tl.load(Sources + command)
    destination = tl.load(Destinations + command)
    operation = tl.load(Operations + command)
    raw_slot = tl.load(RawSlots + command)
    accepted = tl.load(Accepted + command)
    live = (command < count) & (destination > 0) & ((operation != 0) | (source > 0))
    error = tl.full((), 0, tl.int32)
    error = tl.where(live & ((operation < 0) | (operation > 1)), 1, error)
    error = tl.where(
        live & ((destination >= SLOTS) | ((operation == 0) & (source >= SLOTS))),
        2,
        error,
    )
    error = tl.where(live & (raw_slot >= RAW_SLOTS), 3, error)
    consumed = tl.load(
        Consumed + source * CONSUMED_STRIDE,
        mask=live & (operation == 0) & (source < SLOTS),
        other=0,
    )
    restoring = live & (raw_slot >= 0)
    if EXACT:
        expected = tl.load(Expected + command)
        error = tl.where(live & ((expected < 0) | ((operation == 1) & (expected != 0))
                         | ((operation == 0) & ((expected <= 0) | (consumed != expected)))), 4, error)
        physical_slot = tl.load(Requests + command)
        endpoint = expected // 64
        physical_valid = (physical_slot >= 0) & (physical_slot < LIVE)
        tail = live & (operation == 0) & (expected > 0) & (expected % 64 != 0)
        error = tl.where(live & ~physical_valid, 2, error)
        page = tl.load(Tables + physical_slot * TABLE_WIDTH + endpoint,
                       tail & physical_valid & (endpoint < TABLE_WIDTH), other=-1)
        error = tl.where(tail & ((endpoint >= TABLE_WIDTH) | (page < 0) | (page >= PAGE_COUNT)), 8, error)
        if HAS_DRAFT:
            for part in tl.static_range(2):
                logical = endpoint - 1 + part
                needed = live & (operation == 0) & (expected > 1) & (logical >= 0)
                if part == 1:
                    needed &= expected % 64 >= 2
                page = tl.load(DraftTables + physical_slot * TABLE_WIDTH + logical,
                               needed & physical_valid & (logical < TABLE_WIDTH), other=-1)
                error = tl.where(needed & ((logical >= TABLE_WIDTH) | (page < 0) | (page >= PAGE_COUNT)), 8, error)
        selected = tl.full((), 1, tl.int32)
        if SPECULATIVE:
            selected = tl.load(PreviousAccepted + physical_slot,
                               live & physical_valid & (raw_slot < 0) & (operation == 0), other=1)
            error = tl.where(live & ((selected < 1) | (selected > 8)), 5, error)
    else:
        error = tl.where(restoring & ((consumed < 0) | (consumed % 64 != 0)), 4, error)
    error = tl.where(restoring & (accepted <= 0), 5, error)
    others = tl.arange(0, OP_BLOCK)
    other_destination = tl.load(Destinations + others, others < CAPACITY, other=-1)
    other_source = tl.load(Sources + others, others < CAPACITY, other=-1)
    other_operation = tl.load(Operations + others, others < CAPACITY, other=-1)
    other_raw_slot = tl.load(RawSlots + others, others < CAPACITY, other=-1)
    other_live = (
        (others < count)
        & (others < CAPACITY)
        & (others != command)
        & (other_destination > 0)
    )
    other_live &= (other_operation == 1) | (other_source > 0)
    conflict = destination == other_destination
    conflict |= (operation == 0) & (source == other_destination)
    conflict |= (other_operation == 0) & (destination == other_source)
    conflict |= (raw_slot >= 0) & (raw_slot == other_raw_slot)
    error = tl.where(
        live & (tl.sum((other_live & conflict).to(tl.int32), 0) > 0), 6, error
    )
    error = tl.where((count < 0) | (count > CAPACITY), 7, error)
    if tile == 0:
        tl.store(Errors + command, error)
    valid = live & (error == 0)
    offsets = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = valid & (offsets < ROW_BYTES)
    for span in tl.static_range(len(EXCLUDED)):
        mask &= (offsets < EXCLUDED[span][0]) | (offsets >= EXCLUDED[span][1])
    if operation == 1:
        history = (offsets >= HISTORY_OFFSET) & (offsets < HISTORY_OFFSET + 16)
        eos_byte = (
            tl.full((), 248044, tl.uint64) >> (((offsets - HISTORY_OFFSET) & 7) * 8)
        ) & 255
        value = tl.where(history, eos_byte, 0).to(tl.uint8)
        tl.store(Backing + destination * ROW_BYTES + offsets, value, mask)
    elif source != destination:
        read_slot = source
        read_offsets = offsets
        keep = tl.full((BLOCK,), True, tl.int1)
        if EXACT and SPECULATIVE:
            snapshot = raw_slot < 0
            kind = tl.load(CopyLayout + tile * 4)
            region_start = tl.load(CopyLayout + tile * 4 + 1)
            history = tl.load(CopyLayout + tile * 4 + 2)
            state_history = tl.load(CopyLayout + tile * 4 + 3)
            if snapshot & (kind == 2):
                relative = offsets - region_start
                column = (relative // 2) % state_history
                keep = column < history
                read_offsets = offsets + (selected - 1) * 2
        value = tl.load(Backing + read_slot * ROW_BYTES + read_offsets, mask & keep, other=0)
        tl.store(Backing + destination * ROW_BYTES + offsets, value, mask)


@triton.jit
def _invalidate_qsa(
    Pointers,
    Consumed,
    Sources,
    Destinations,
    Operations,
    RawSlots,
    Accepted,
    Count,
    Errors,
    CONSUMED_STRIDE: tl.constexpr,
    MemoryCapacity,
    RAW_SLOTS: tl.constexpr,
    RING: tl.constexpr,
    AXES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    command, layer = tl.program_id(0), tl.program_id(1)
    SLOTS = tl.load(MemoryCapacity)
    destination = tl.load(Destinations + command)
    source = tl.load(Sources + command)
    operation = tl.load(Operations + command)
    raw_slot = tl.load(RawSlots + command)
    live = (command < tl.load(Count)) & (tl.load(Errors + command) == 0)
    live &= (
        (destination > 0)
        & (destination < SLOTS)
        & (raw_slot >= 0)
        & (raw_slot < RAW_SLOTS)
    )
    live &= (operation == 1) | ((operation == 0) & (source > 0) & (source < SLOTS))
    logical = tl.load(Pointers + layer * 3).to(tl.pointer_type(tl.int64))
    rope = tl.load(Pointers + layer * 3 + 1).to(tl.pointer_type(tl.int64))
    anchor = tl.load(Pointers + layer * 3 + 2).to(tl.pointer_type(tl.int64))
    index = tl.arange(0, BLOCK)
    tl.store(logical + raw_slot * RING + index, -1, live & (index < RING))
    tl.store(rope + raw_slot * RING * AXES + index, -1, live & (index < RING * AXES))
    consumed = tl.load(Consumed + destination * CONSUMED_STRIDE, live, other=0)
    accepted = tl.load(Accepted + command)
    tl.store(anchor + raw_slot, consumed - accepted, live)


class PreparedStateTransfers:
    """Prepared indirect copy/reset with optional QSA raw-ring invalidation.

    Preparation requires zero live commands and primes every compiled kernel.
    CUDA graph replay reads only persistent device metadata. QSA invalidation
    writes tags, RoPE tags, and anchors; raw key payloads need not be cleared.
    """

    def __init__(
        self, pool: HybridStatePool, metadata: StateTransferMetadata, *, qsa_states=(), exact=None
    ):
        device = pool.backing.device
        if device.type != "cuda":
            raise ValueError("prepared state transfers require CUDA")
        capacity = metadata.source_slots.numel()
        for name, dtype in (
            ("request_slots", torch.int64),
            ("source_slots", torch.int64),
            ("destination_slots", torch.int64),
            ("operation", torch.int32),
            ("qsa_slots", torch.int64),
            ("accepted_tokens", torch.int32),
            ("expected_consumed", torch.int64),
            ("num_ops", torch.int32),
            ("errors", torch.int32),
        ):
            tensor = getattr(metadata, name)
            shape = (1,) if name == "num_ops" else (capacity,)
            if (
                tensor.device != device
                or tensor.dtype != dtype
                or tensor.shape != shape
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"state transfer {name} has incompatible shape, dtype, or layout"
                )
        if capacity <= 0:
            raise ValueError("state transfer needs positive command capacity")
        self.pool, self.metadata, self.qsa_states = pool, metadata, tuple(qsa_states)
        self.exact = exact
        if exact is not None and exact.pool is not pool:
            raise ValueError("exact checkpoint storage must belong to the transferred pool")
        raw_slots, ring, axes = 0, 0, 0
        pointers = []
        for state in self.qsa_states:
            logical, rope, anchor = (
                state.raw_logical_positions,
                state.raw_rope_positions,
                state.raw_interval_start_positions,
            )
            if (
                logical.ndim != 2
                or rope.ndim != 3
                or rope.shape[:2] != logical.shape
                or anchor.shape != (logical.shape[0],)
            ):
                raise ValueError("QSA raw position tensors have incompatible shapes")
            shape = (logical.shape[0], logical.shape[1], rope.shape[2])
            if raw_slots and shape != (raw_slots, ring, axes):
                raise ValueError("QSA raw-state geometries must agree across layers")
            raw_slots, ring, axes = shape
            for tensor in (logical, rope, anchor):
                if (
                    tensor.device != device
                    or tensor.dtype != torch.int64
                    or not tensor.is_contiguous()
                ):
                    raise ValueError(
                        "QSA raw positions must be contiguous int64 on the pool device"
                    )
            pointers.append([logical.data_ptr(), rope.data_ptr(), anchor.data_ptr()])
        self.qsa_pointers = (
            torch.tensor(pointers, dtype=torch.int64, device=device)
            if pointers
            else None
        )
        block = exact.copy_block if exact is not None else 4096
        self._grid = (capacity, triton.cdiv(pool.layout.row_bytes, block), 1)
        self._transfer_args = (
            pool.backing,
            pool.consumed_tokens,
            metadata.source_slots,
            metadata.destination_slots,
            metadata.operation,
            metadata.qsa_slots,
            metadata.accepted_tokens,
            metadata.num_ops,
            metadata.errors,
            metadata.expected_consumed,
            exact.previous_accepted if exact is not None else None,
            exact.copy_layout if exact is not None else None,
            exact.tables if exact is not None else None,
            exact.draft_tables if exact is not None else None,
            metadata.request_slots,
            exact.state_rows if exact is not None else None,
            exact is not None,
            bool(pool.layout.speculative_tokens) if exact is None else exact.previous_accepted is not None,
            exact is not None and exact.draft is not None,
            pool.memory_capacity,
            exact.tables.shape[1] if exact is not None else 0,
            pool.layout.row_bytes,
            capacity,
            raw_slots,
            pool.layout.region("ple_history").offset,
            pool.consumed_tokens.stride(0),
            block,
            triton.next_power_of_2(capacity),
            exact.tables.shape[0] if exact is not None else capacity,
            tuple((r.offset, r.offset + r.nbytes) for r in pool.layout.regions
                  if exact is not None and exact.previous_accepted is None
                  and (r.name.startswith("draft_") or r.name.startswith("target_residual"))),
        )
        self._qsa_grid = (capacity, len(self.qsa_states), 1)
        self._qsa_args = (
            self.qsa_pointers,
            pool.consumed_tokens,
            metadata.source_slots,
            metadata.destination_slots,
            metadata.operation,
            metadata.qsa_slots,
            metadata.accepted_tokens,
            metadata.num_ops,
            metadata.errors,
            pool.consumed_tokens.stride(0),
            pool.memory_capacity,
            raw_slots,
            ring,
            axes,
            triton.next_power_of_2(max(ring * axes, 1)),
        )
        self._prepared = False

    def prepare(self) -> None:
        if self.metadata.num_ops.item() != 0:
            raise ValueError("state transfer preparation requires zero live commands")
        compiled = _transfer.warmup(*self._transfer_args, grid=self._grid, num_warps=4)
        self._transfer = partial(compiled[self._grid], *self._transfer_args)
        self._qsa = None
        if self.qsa_states and self.exact is None:
            compiled = _invalidate_qsa.warmup(
                *self._qsa_args, grid=self._qsa_grid, num_warps=4
            )
            self._qsa = partial(compiled[self._qsa_grid], *self._qsa_args)
        self._prepared = True
        self()

    def __call__(self) -> None:
        if not self._prepared:
            raise RuntimeError("state transfer prepare() must complete before capture")
        self._transfer()
        if self.exact is not None:
            self.exact()
        elif self._qsa is not None:
            self._qsa()

@triton.jit
def _checkpoint_raw(Pointers, Backing, Sources, Destinations, Operations, Expected, Count, Errors, Requests, RawSlots,
                    ROW_BYTES: tl.constexpr, ELEMENTS: tl.constexpr, RING: tl.constexpr,
                    FIELD: tl.constexpr, DRAFT: tl.constexpr, BLOCK: tl.constexpr, LIVE: tl.constexpr = 4):
    command, layer = tl.program_id(0), tl.program_id(1)
    source, destination = tl.load(Sources + command), tl.load(Destinations + command)
    operation = tl.load(Operations + command)
    active = (command < tl.load(Count)) & (destination > 0) & (tl.load(Errors + command) == 0)
    snapshot = (operation == 0) & (tl.load(RawSlots + command) < 0)
    restore = (operation == 0) & (tl.load(RawSlots + command) >= 0)
    reset = operation == 1
    slot = tl.load(Requests + command) + 1
    active &= snapshot | restore | reset
    dtype = tl.bfloat16 if FIELD == 0 else tl.int64
    pointer = tl.load(Pointers + layer * 3).to(tl.pointer_type(dtype))
    offset = tl.load(Pointers + layer * 3 + 1)
    tags = tl.load(Pointers + layer * 3 + 2).to(tl.pointer_type(tl.int64))
    index = tl.arange(0, BLOCK)
    mask = active & (index < ELEMENTS)
    endpoint = tl.load(Expected + command) - (1 if DRAFT else 0)
    endpoint = tl.maximum(endpoint, 0)
    if snapshot:
        if FIELD == 3:
            value = tl.full((BLOCK,), 0, tl.int64) + endpoint - 1
        else:
            ring_index = index // 128 if FIELD == 0 else (index // 3 if FIELD == 2 else index)
            tag = tl.load(tags + slot * RING + ring_index, mask, other=-1)
            valid = (tag >= 0) & (tag < endpoint)
            value = tl.load(pointer + slot * ELEMENTS + index, mask & valid, other=0 if FIELD == 0 else -1)
        saved = (Backing + destination * ROW_BYTES + offset).to(tl.pointer_type(dtype))
        tl.store(saved + index, value, mask)
    elif restore:
        saved = (Backing + source * ROW_BYTES + offset).to(tl.pointer_type(dtype))
        value = tl.load(saved + index, mask, other=0)
        tl.store(pointer + slot * ELEMENTS + index, value, mask)
    elif reset:
        tl.store(pointer + slot * ELEMENTS + index, 0 if FIELD == 0 else -1, mask)


@triton.jit
def _checkpoint_pages(Pointers, Backing, Tables, Sources, Destinations, Operations, Expected, Count, Errors, Requests, RawSlots,
                      ROW_BYTES: tl.constexpr, ELEMENTS: tl.constexpr, TABLE_WIDTH: tl.constexpr,
                      DRAFT_PART: tl.constexpr, BLOCK: tl.constexpr, LIVE: tl.constexpr = 4):
    command, layer, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    source, destination = tl.load(Sources + command), tl.load(Destinations + command)
    endpoint = tl.load(Expected + command)
    snapshot = tl.load(RawSlots + command) < 0
    restore = tl.load(RawSlots + command) >= 0
    active = (command < tl.load(Count)) & (tl.load(Errors + command) == 0)
    active &= (tl.load(Operations + command) == 0) & (snapshot | restore) & (endpoint > 0)
    logical = endpoint // 64
    if DRAFT_PART < 0:
        active &= endpoint % 64 != 0
    else:
        logical += DRAFT_PART - 1
        active &= (endpoint > 1) & (logical >= 0)
        if DRAFT_PART == 1:
            active &= endpoint % 64 >= 2
    slot = tl.load(Requests + command)
    page = tl.load(Tables + slot * TABLE_WIDTH + logical, active, other=-1)
    active &= page >= 0
    pointer = tl.load(Pointers + layer * 2).to(tl.pointer_type(tl.uint8))
    offset = tl.load(Pointers + layer * 2 + 1)
    index = tile * BLOCK + tl.arange(0, BLOCK)
    mask = active & (index < ELEMENTS)
    if snapshot:
        value = tl.load(pointer + page * ELEMENTS + index, mask, other=0)
        saved = Backing + destination * ROW_BYTES + offset
        tl.store(saved + index, value, mask)
    elif restore:
        saved = Backing + source * ROW_BYTES + offset
        value = tl.load(saved + index, mask, other=0)
        tl.store(pointer + page * ELEMENTS + index, value, mask)


def checkpoint_copy_layout(layout, block=4096):
    """Classify whole transfer tiles for selected history normalization.

    Recurrent state needs none: each round commits accepted verification records
    into the live slot, so snapshots copy it as is.
    """
    result = torch.zeros((triton.cdiv(layout.row_bytes, block), 4), dtype=torch.int64)
    for region in layout.regions:
        if region.name.endswith(".convolution") or region.name == "ple_convolution":
            kind, history, capacity = 2, region.shape[1] - layout.speculative_tokens, region.shape[1]
        else:
            continue
        if region.offset % block or region.nbytes % block:
            raise ValueError("state normalization regions must cover whole transfer tiles")
        result[region.offset // block:(region.offset + region.nbytes) // block] = torch.tensor(
            [kind, region.offset, history, capacity])
    return result


class PreparedCheckpointCaches:
    """Exact target/draft cache tails plus normalized recurrent snapshot storage."""

    def __init__(self, pool, metadata, qsa_states, tables, *, state_rows, speculation=None):
        from .decoder_state import compile_kernel

        slots = tables.shape[0]
        self.pool, self.metadata = pool, metadata
        self.states, self.tables = tuple(qsa_states), tables
        self.draft = speculation.cache[48] if speculation is not None and speculation.draft_tables is not None else None
        self.draft_tables = speculation.draft_tables if self.draft is not None else None
        self.previous_accepted = speculation.previous_accepted if speculation is not None else None
        self.state_rows = state_rows
        from math import gcd

        self.copy_block = 4096
        for region in pool.layout.regions:
            if region.name.endswith((".recurrent", ".convolution")) or region.name == "ple_convolution":
                self.copy_block = gcd(self.copy_block, region.offset, region.nbytes)
        self.copy_layout = checkpoint_copy_layout(pool.layout, self.copy_block).to(pool.backing.device)
        if len(self.states) != len(QSA_LAYERS):
            raise ValueError("exact checkpoints require all twelve target QSA states")
        if tables.ndim != 2 or slots <= 0 or tables.shape[1] == 0 or tables.dtype != torch.int32 or not tables.is_contiguous():
            raise ValueError("checkpoint page tables must be contiguous int32[capacity,pages]")
        if tables.device != pool.backing.device:
            raise ValueError("checkpoint page tables must share the state device")
        self.page_count = self.states[0].main_k_cache.shape[0]
        pool.memory_capacity[1] = self.page_count
        if self.draft_tables is not None and (self.draft_tables.shape != tables.shape
                or self.draft_tables.dtype != torch.int32 or not self.draft_tables.is_contiguous()
                or self.draft_tables.device != tables.device):
            raise ValueError("draft checkpoint tables must match target table geometry")
        if speculation is not None:
            previous = self.previous_accepted
            if (previous.shape != (slots,) or previous.dtype != torch.int32
                    or previous.device != pool.backing.device or not previous.is_contiguous()):
                raise ValueError("speculative checkpoints require an accepted count per live slot")
        self.operations, self.pointers = [], []
        groups = [(self.states, tuple(f"qsa.{layer}." for layer in QSA_LAYERS), False)]
        if self.draft is not None:
            groups.append(((self.draft,), ("draft_",), True))
        for states, prefixes, draft in groups:
            ring = states[0].raw_k_ring.shape[1]
            for state in states:
                if state.raw_k_ring.shape[0] < slots + 1 or state.raw_k_ring.shape[1] != ring:
                    raise ValueError("checkpoint raw-state geometry differs across layers")
                for name in ("main_k_cache", "main_v_cache", "compressed_k_cache"):
                    value = getattr(state, name)
                    if value.shape[0] != self.page_count or value.device != pool.backing.device or not value.is_contiguous():
                        raise ValueError("checkpoint page pools must share capacity and contiguous device layout")
            for field, (name, suffix) in enumerate((("raw_k_ring", "raw_keys"),
                    ("raw_logical_positions", "raw_positions"), ("raw_rope_positions", "raw_rope"),
                    ("raw_interval_start_positions", "raw_anchor"))):
                pointers = []
                for state, prefix in zip(states, prefixes, strict=True):
                    value, region = getattr(state, name), pool.layout.region(prefix + suffix)
                    if value.dtype != region.dtype or value.shape[1:] != region.shape and not (field == 3 and value.ndim == 1):
                        raise ValueError("checkpoint raw field shape or dtype differs from its storage")
                    if not value.is_contiguous() or value.device != pool.backing.device:
                        raise ValueError("checkpoint raw fields must be contiguous on the state device")
                    pointers.append((value.data_ptr(), region.offset, state.raw_logical_positions.data_ptr()))
                pointer = torch.tensor(pointers, device=pool.backing.device, dtype=torch.int64)
                self.pointers.append(pointer)
                size = pool.layout.region(prefixes[0] + suffix).nbytes // value.element_size()
                self.operations.append(compile_kernel(_checkpoint_raw,
                    (pointer, pool.backing, metadata.source_slots, metadata.destination_slots, metadata.operation,
                     metadata.expected_consumed, metadata.num_ops, metadata.errors, metadata.request_slots, metadata.qsa_slots, pool.layout.row_bytes,
                     size, ring, field, draft, triton.next_power_of_2(size), slots), (metadata.source_slots.numel(), len(states), 1)))
            for part in ((0, 1) if draft else (-1,)):
                for name, suffix in (("main_k_cache", "boundary_key"), ("main_v_cache", "boundary_value"),
                                      ("compressed_k_cache", "boundary_selector")):
                    pointers = []
                    for state, prefix in zip(states, prefixes, strict=True):
                        region = pool.layout.region(prefix + suffix + ("_extra" if part == 1 else ""))
                        value = getattr(state, name)
                        if value.shape[1:] != region.shape or value.dtype != region.dtype:
                            raise ValueError("checkpoint page field differs from its storage")
                        pointers.append((value.data_ptr(), region.offset))
                    pointer = torch.tensor(pointers, device=pool.backing.device, dtype=torch.int64)
                    self.pointers.append(pointer)
                    size = value[0].nbytes
                    self.operations.append(compile_kernel(_checkpoint_pages,
                        (pointer, pool.backing, self.draft_tables if draft else tables, metadata.source_slots,
                         metadata.destination_slots, metadata.operation, metadata.expected_consumed,
                         metadata.num_ops, metadata.errors, metadata.request_slots, metadata.qsa_slots, pool.layout.row_bytes, size, tables.shape[1], part, 1024, slots),
                        (metadata.source_slots.numel(), len(states), triton.cdiv(size, 1024))))

    def __call__(self):
        for operation in self.operations:
            operation()


__all__ = [
    "COPY",
    "EOS_TOKEN_ID",
    "GDN_LAYERS",
    "QSA_LAYERS",
    "RESET",
    "HybridStateLayout",
    "HybridStatePool",
    "PreparedCheckpointCaches",
    "PreparedStateTransfers",
    "StateTransferMetadata",
]
