# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Sparse-attention index selection on SM103 through FlashInfer's paged FP8 MQA logits and top-k.

b12x's indexer issues block-scaled MMA instructions that SM103 does not have.
FlashInfer's paged FP8 MQA logits compute the same scores DeepGEMM defines,
``sum_h w_h relu(q_h . k_j) s_j`` over FP8 keys with per-key scales (the query
scales are folded into ``w``), reading the 64-record index pages in place, and
its length-aware top-k keeps each row's highest-scoring logical records. Rows
with fewer visible records pad their selection with -1.
"""

from __future__ import annotations

import itertools
import weakref

import torch

_OWNERS = weakref.WeakValueDictionary()
_HANDLES = itertools.count()
_PAGE_RECORDS = 64
_RECORD_BYTES = 132


@torch.library.custom_op("flare::sm103_index_select", mutates_args=("output", "logits"))
def _select(query: torch.Tensor, weights: torch.Tensor, cache: torch.Tensor, page_table: torch.Tensor,
            lengths: torch.Tensor, output: torch.Tensor, logits: torch.Tensor, handle: int) -> None:
    _OWNERS[handle].run(query, weights, cache, page_table, lengths, output, logits)


@_select.register_fake
def _fake_select(query, weights, cache, page_table, lengths, output, logits, handle) -> None:
    pass


class Sm103SparseIndexer:
    """Drop-in for ``PreparedSparseIndexer`` on SM103: logical top-k index records per query row.

    ``query`` is FP8 ``[rows, heads, 128]``, ``weights`` FP32 ``[rows, heads]`` with query scales
    folded in, ``cache`` the uint8 index pages ``[pages, 64 * 132]``, ``page_table`` int32
    ``[rows, width]`` (prefill rows may share one row), ``lengths`` each row's visible records,
    and ``output`` int32 ``[rows, topk]``.
    """

    def __init__(self, query, weights, cache, page_table, lengths, active_width, output, *,
                 caps, session, name, workspace, scores=None):
        from flashinfer.experimental.deepgemm_dense_mqa import paged_mqa

        rows, heads, dim = query.shape
        if scores is not None:
            raise ValueError("SM103 index selection does not return scores for context-parallel ranks")
        if (dim != 128 or query.dtype != torch.float8_e4m3fn or not query.is_contiguous()
                or weights.shape != (rows, heads) or weights.dtype != torch.float32
                or cache.dtype != torch.uint8 or cache.shape[1] != _PAGE_RECORDS * _RECORD_BYTES
                or page_table.dtype != torch.int32 or lengths.shape != (rows,) or lengths.dtype != torch.int32
                or output.shape[0] != rows or output.dtype != torch.int32):
            raise ValueError("SM103 index selection requires contiguous FP8 [rows, heads, 128] queries, FP32 "
                             "weights, 64-record uint8 index pages and int32 tables, lengths and outputs")
        if not paged_mqa.paged_route_available(heads, _PAGE_RECORDS, 1):
            raise ValueError(f"FlashInfer ships no paged MQA logits route for {heads} heads")
        self.query, self.weights, self.lengths, self.output = query, weights, lengths, output
        # A shared prefill table is an expanded view; the kernel reads any row stride.
        self.page_table = page_table.expand(rows, -1) if page_table.shape[0] == 1 else page_table
        if self.page_table.shape[0] != rows:
            raise ValueError("index selection needs one page table row per query row or one shared row")
        self.topk = output.shape[1]
        self.width = page_table.shape[1] * _PAGE_RECORDS
        if paged_mqa.paged_logits_stride(self.width) != self.width:
            raise ValueError("SM103 index selection needs a page table whose record width is a multiple of 256")
        self.logits = torch.empty((rows, self.width), dtype=torch.float32, device=query.device)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self
        self.rebind_cache(cache)

    def rebind_cache(self, cache):
        """Bind a replacement index cache before capture; its page count is a runtime bound."""
        from flashinfer.experimental.deepgemm_dense_mqa import paged_mqa

        if cache.dtype != torch.uint8 or cache.shape[1] != _PAGE_RECORDS * _RECORD_BYTES:
            raise ValueError("replacement index cache must keep the 64-record page layout")
        self.cache = cache
        if self.query is None:
            return
        rows = self.query.shape[0]
        # FlashInfer's admission rules mark where its route beats DeepGEMM's; correctness holds everywhere.
        self.plan = paged_mqa.PagedMqaPlan(self.query.view(rows, 1, -1, 128),
            cache.view(cache.shape[0], _PAGE_RECORDS, 1, _RECORD_BYTES), self.weights, self.lengths.view(rows, 1),
            self.page_table, self.width, output=self.logits, enforce_admission=False)

    def run(self, query, weights, cache, page_table, lengths, output, logits):
        import flashinfer

        if self.query is None:
            self.plan.run(query.view(query.shape[0], 1, -1, 128),
                          cache.view(cache.shape[0], _PAGE_RECORDS, 1, _RECORD_BYTES), weights,
                          lengths.view(-1, 1), page_table, output=logits)
        else:
            self.plan.run()
        flashinfer.top_k_varlen(logits, lengths, self.topk, out_indices=output)

    def functional(self, *, session):
        self.rows = self.query.shape[0]
        self.plan.functional()
        self.query = self.weights = self.page_table = self.lengths = self.output = self.logits = None
        return self

    def __call__(self, query=None, weights=None, page_table=None, lengths=None, active_width=None, *, cache=None):
        if self.query is not None:
            _select(self.query, self.weights, self.cache, self.page_table, self.lengths, self.output, self.logits,
                    self.handle)
            return self.output
        output = torch.full((self.rows, self.topk), -1, dtype=torch.int32, device=query.device)
        logits = torch.empty((self.rows, self.width), dtype=torch.float32, device=query.device)
        pages = page_table.expand(self.rows, -1) if page_table.shape[0] == 1 else page_table
        _select(query, weights, self.cache if cache is None else cache, pages, lengths, output, logits, self.handle)
        return output
