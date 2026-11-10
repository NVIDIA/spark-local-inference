# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""FP8 sparse selection with b12x Hadamard quantization and caller-owned pages."""

import torch
import triton
import triton.language as tl
from b12x.attention import dsa_indexer
from b12x.attention.dsa_indexer import glm_pool
from b12x.attention.dsa_indexer._glm_pool_kernels import _write_pool
from b12x.preparation import PreparedCall

from ignition.components.indexer import PreparedSparseIndexer
from ignition.preparation.examples import declaration_tensor
from ignition.preparation.tuning import declare, register_finalizer


@triton.jit
def _write_keys(Keys, Cache, ScaleCache, Slots, stride):
    row = tl.program_id(0)
    dim = tl.arange(0, 128)
    slot = tl.load(Slots + row)
    values = tl.load(Keys + row * 128 + dim)
    _write_pool(Cache, ScaleCache, slot, values, dim, dim < 128, slot >= 0,
                stride, 64, 128, 448.0)


class PreparedFP8Indexer:
    """Rotate BF16 queries/keys and select from planar 64-record FP8 pages.

    Query scales are folded into FP32 head weights. B12x's prepared query
    operation includes normalization for 32 heads, so the input weight copy
    compensates for the declared head count before that operation runs.
    ``prefill_records_per_call`` bounds each sequential candidate window and
    its score workspace; all declared cache records participate in selection.
    """

    def __init__(self, query, weights, cache, pages, lengths, active, output, *,
                 mode, session, name, workspace, buffers=None, prefill_records_per_call=None,
                 examples=None, example_role=None):
        if prefill_records_per_call is not None and (type(prefill_records_per_call) is not int
                                                     or prefill_records_per_call <= 0):
            raise ValueError("FP8 prefill record capacity per call must be a positive integer")
        rows, heads, width = query.shape
        if width != 128 or query.dtype != torch.bfloat16 or cache.shape[1] != 64 * 132:
            raise ValueError("FP8 indexing requires BF16 128-channel queries and 64-record pages")
        self.query, self.weights, self.cache = query, weights, cache
        if examples is not None and (not isinstance(example_role, str) or not example_role):
            raise ValueError("shared indexer declarations require an explicit operand role")

        def zeros(role, shape, dtype):
            if examples is not None:
                return declaration_tensor(examples, example_role + "." + role, shape, dtype, query.device)
            return (torch.zeros(shape, dtype=dtype, device=query.device) if buffers is None
                    else buffers.zeros(shape, dtype))
        self.quantized = zeros("quantized", query.shape, torch.float8_e4m3fn)
        self.scales = zeros("scales", (rows, heads), torch.float32)
        self.scaled_weights = zeros("scaled_weights", (rows, heads), torch.float32)
        self.weight_normalization = (32 / heads) ** 0.5
        self.query_arguments = (query.view(-1, 128), self.quantized.view(-1, 128),
                                self.scales.view(-1), self.scaled_weights.view(-1))
        self.query_plan = glm_pool.plan_queries(*self.query_arguments)
        query_arguments = self.query_arguments
        declare(session, (self.query_plan.request(name=name + ".query",
            prepare_call=lambda state: PreparedCall(run=lambda: state.quantize(*query_arguments))),))
        self.pages = pages
        self.packed_pages = pages if mode == "prefill" else zeros("packed_pages", pages.shape, torch.int32)
        caps = dsa_indexer.Caps(device=query.device, num_q_heads=heads, max_q_rows=rows,
            max_page_table_width=pages.shape[1], topk=512, mode=mode,
            supertile_k=(min(prefill_records_per_call, pages.shape[1] * 64)
                         if mode == "prefill" and prefill_records_per_call is not None else 0))
        self.selection = PreparedSparseIndexer(self.quantized, self.scaled_weights, cache,
            self.packed_pages, lengths, active, output, caps=caps, session=session,
            name=name + ".select", workspace=workspace)

    def write_keys(self, keys, slots):
        # Reuse b12x's Hadamard, power-of-two scaling, and planar FP8 page encoding.
        torch.library.wrap_triton(_write_keys)[(keys.shape[0],)](
            keys, self.cache.view(torch.float8_e4m3fn), self.cache.view(torch.float32),
            slots, self.cache.stride(0), num_warps=4)

    def functional(self, *, session):
        self.query_layout = tuple(self.query.shape)
        self.copy_pages = self.packed_pages is not self.pages
        self.selection.functional(session=session)
        register_finalizer(session, self._release_examples)
        return self

    def _release_examples(self):
        self.query = self.weights = self.quantized = self.scales = self.scaled_weights = None
        self.query_arguments = self.pages = self.packed_pages = None

    def __call__(self, query=None, weights=None, pages=None, lengths=None, active=None):
        output, _, _ = self._run(query, weights, pages, lengths, active)
        return output

    def _run(self, query=None, weights=None, pages=None, lengths=None, active=None):
        if self.query is None:
            rows, heads, _ = self.query_layout
            quantized = torch.zeros(self.query_layout, dtype=torch.float8_e4m3fn, device=query.device)
            scales = torch.zeros((rows, heads), dtype=torch.float32, device=query.device)
            scaled_weights = torch.empty((rows, heads), dtype=torch.float32, device=query.device)
            scaled_weights.copy_(weights)
            scaled_weights.mul_(self.weight_normalization)
            glm_pool.quantize_queries(query.view(-1, 128), quantized.view(-1, 128), scales.view(-1),
                                      scaled_weights.view(-1), plan=self.query_plan)
            output = self.selection(quantized, scaled_weights, pages.clone() if self.copy_pages else pages,
                                    lengths, active)
            return output, quantized, scaled_weights
        self.scaled_weights.copy_(self.weights)
        self.scaled_weights.mul_(self.weight_normalization)
        glm_pool.quantize_queries(*self.query_arguments, plan=self.query_plan)
        if self.packed_pages is not self.pages:
            self.packed_pages.copy_(self.pages)
        return self.selection(), self.quantized, self.scaled_weights
