# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Selected-record addressing for interleaved Kimi context shards."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from ignition.recipes.kimi3.config import PAGE_TOKENS
from ignition.recipes.kimi3.dcp import RECORD_BYTES as RECORD_BYTES, ContextShards, write_records as write_records

TOPK = 2048
PREFILL_ROWS = 128


@dataclass(frozen=True)
class AttentionConstants:
    """Immutable one-record-page attention metadata shared by serial layers and slices."""

    pages: torch.Tensor
    starts: torch.Tensor
    scale: torch.Tensor

    @classmethod
    def allocate(cls, device, *, query_capacity=PREFILL_ROWS):
        return cls(torch.arange(PREFILL_ROWS * TOPK, dtype=torch.int32, device=device).view(PREFILL_ROWS, TOPK),
                   torch.arange(max(PREFILL_ROWS, query_capacity) + 1, dtype=torch.int32, device=device),
                   torch.ones(1, dtype=torch.float32, device=device))


@dataclass(frozen=True)
class ContextParallel:
    """Latent-head and exact candidate exchanges shared by sparse attention layers."""

    shards: ContextShards
    heads: object
    coordinator: object
    peers: dict
    attention_constants: AttentionConstants
    slab: torch.Tensor | None = None
    candidates: object = None
    window: int = 0
    staging: object = None
    staged_records: torch.Tensor | None = None
    staged_address: torch.Tensor | None = None
    stage_signals: tuple = ()
    stage_runtime: object = None
    index_staging: object = None
    index_records: torch.Tensor | None = None
    index_columns: torch.Tensor | None = None
    index_signals: tuple = ()
    selections: object = None
    candidate_owners: object = None
    selection_tuning: object = None
    groups: tuple = ()


def index_pages(model_pages, output):
    """Map each 128-record model page to its two planar 64-record index pages."""
    output[:, 0::2].copy_(torch.where(model_pages >= 0, model_pages * 2, -1))
    output[:, 1::2].copy_(torch.where(model_pages >= 0, model_pages * 2 + 1, -1))


def pack_candidates(indices, scores, output, shards):
    """Pack FP32 scores and bit-preserved global int32 token IDs for BF16 transport."""
    output[:, :, 0].copy_(torch.where(indices >= 0, scores, -torch.inf))
    output.view(torch.int32)[:, :, 1].copy_(torch.where(indices >= 0,
        indices * shards.world + shards.owner, -1).to(torch.int32))


@triton.jit
def _selection_slots(Selected, Positions, Tables, Owners, Slots, Counts, Rank,
                     columns, table_stride, cache_pages, WORLD: tl.constexpr,
                     LOCAL: tl.constexpr, TOP_K: tl.constexpr, PAGE: tl.constexpr,
                     GLOBAL_RECORDS: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, TOP_K)
    token = tl.load(Selected + row * TOP_K + column).to(tl.int64)
    end = tl.load(Positions + row).to(tl.int64)
    local = tl.maximum(token, 0) // WORLD
    owner = token % WORLD
    block = local // PAGE
    live = (token >= 0) & (token <= end) & (block < columns)
    if LOCAL:
        live &= owner == tl.load(Rank)
    page = tl.load(Tables + row * table_stride + block, mask=live, other=-1).to(tl.int64)
    live &= (page >= 0) & (page < cache_pages)
    target = tl.cumsum(live.to(tl.int32), 0) - 1
    tl.store(Owners + row * TOP_K + column, -1)
    tl.store(Slots + row * TOP_K + column, -1)
    tl.debug_barrier()
    slot = page * PAGE + local % PAGE
    if GLOBAL_RECORDS:
        owner = tl.full((TOP_K,), 0, tl.int64)
        slot = token
    tl.store(Owners + row * TOP_K + target, owner.to(tl.int32), mask=live)
    tl.store(Slots + row * TOP_K + target, slot.to(tl.int32), mask=live)
    tl.store(Counts + row, tl.sum(live.to(tl.int32), 0))


def selection_slots(selected, positions, tables, owners, slots, counts, shards, cache_pages, *, local,
                    global_records=False):
    """Compact valid selections without changing their order.

    ``global_records`` addresses a local global-position slab instead of peer
    page pools; the original page-validity and causal masks still apply.
    """
    torch.library.wrap_triton(_selection_slots)[(selected.shape[0],)](
        selected, positions, tables, owners, slots, counts, shards.index,
        tables.shape[1], tables.stride(0), cache_pages, WORLD=shards.world, LOCAL=local,
        TOP_K=TOPK, PAGE=PAGE_TOKENS, GLOBAL_RECORDS=global_records, num_warps=8)
