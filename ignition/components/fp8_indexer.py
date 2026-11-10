# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""FP8 sparse selection with prepared quantization and caller-owned pages."""

import math

import torch
import triton
import triton.language as tl
from b12x.attention import dsa_indexer
from b12x.attention.dsa_indexer import glm_pool
from b12x.attention.dsa_indexer._glm_pool_kernels import _write_pool
from b12x.preparation import PreparedCall

from ignition.components.indexer import PreparedSparseIndexer
from ignition.preparation.tuning import declare


@triton.jit
def _write_keys(Keys, Cache, ScaleCache, Slots, stride, HADAMARD: tl.constexpr, POWER_OF_TWO_SCALE: tl.constexpr):
    row = tl.program_id(0)
    dim = tl.arange(0, 128)
    slot = tl.load(Slots + row)
    values = tl.load(Keys + row * 128 + dim)
    _write_pool(Cache, ScaleCache, slot, values, dim, dim < 128, slot >= 0,
                stride, 64, 128, 448.0, HADAMARD=HADAMARD, POWER_OF_TWO_SCALE=POWER_OF_TWO_SCALE)


class PreparedFP8Indexer:
    """Quantize BF16 queries/keys and select from planar 64-record FP8 pages.

    Hadamard rotation and power-of-two key scales are enabled by default.
    Identity transforms and ordinary FP32 key scales can be declared explicitly.
    Query scales remain powers of two for either transform.

    Query scales are folded into FP32 head weights. B12x's prepared query
    operation includes normalization for 32 heads, so the input weight copy
    compensates for the declared head count before that operation runs.
    The default scales raw weights by ``1 / sqrt(heads)``; an explicit
    ``head_weight_scale`` replaces that multiplier. Already normalized gates
    use one, yielding ``weights * query_scale / sqrt(128)`` in FP32.
    ``prefill_records_per_call`` bounds each sequential candidate window and
    its score workspace; all declared cache records participate in selection.
    """

    def __init__(self, query, weights, cache, pages, lengths, active, output, *,
                 mode, session, name, workspace, buffers, prefill_records_per_call=None, topk=512, scores=None,
                 head_weight_scale=None, hadamard=True, key_power_of_two=True):
        if type(hadamard) is not bool or type(key_power_of_two) is not bool:
            raise ValueError("FP8 Hadamard and key power-of-two declarations must be booleans")
        if head_weight_scale is not None and (type(head_weight_scale) not in (int, float)
                or not math.isfinite(head_weight_scale) or head_weight_scale <= 0):
            raise ValueError("FP8 head-weight scale must be a finite positive scalar")
        if prefill_records_per_call is not None and (type(prefill_records_per_call) is not int
                                                     or prefill_records_per_call <= 0):
            raise ValueError("FP8 prefill record capacity per call must be a positive integer")
        if type(topk) is not int or topk <= 0 or output.shape != (query.shape[0], topk):
            raise ValueError("FP8 selection output must match a positive top-k capacity")
        rows, heads, width = query.shape
        if width != 128 or query.dtype != torch.bfloat16 or cache.shape[1] != 64 * 132:
            raise ValueError("FP8 indexing requires BF16 128-channel queries and 64-record pages")
        self.cache = cache
        self.hadamard, self.key_power_of_two = hadamard, key_power_of_two
        self.quantized = buffers.zeros(query.shape, torch.float8_e4m3fn)
        self.scales = buffers.zeros((rows, heads), torch.float32)
        self.scaled_weights = buffers.zeros((rows, heads), torch.float32)
        self.weight_normalization = ((32 / heads) ** 0.5 if head_weight_scale is None
                                     else 32 ** 0.5 * head_weight_scale)
        self.query_outputs = (self.quantized.view(-1, 128), self.scales.view(-1),
                              self.scaled_weights.view(-1))
        query_arguments = (query.view(-1, 128), *self.query_outputs)
        self.query_plan = glm_pool.plan_queries(*query_arguments, hadamard=hadamard)
        declare(session, (self.query_plan.request(name=name + ".query",
            prepare_call=lambda state: PreparedCall(run=lambda: state.quantize(*query_arguments))),))
        self.pages = pages
        self.packed_pages = pages if mode == "prefill" else buffers.zeros(pages.shape, torch.int32)
        caps = dsa_indexer.Caps(device=query.device, num_q_heads=heads, max_q_rows=rows,
            max_page_table_width=pages.shape[1], topk=topk, mode=mode,
            supertile_k=(min(prefill_records_per_call, pages.shape[1] * 64)
                         if mode == "prefill" and prefill_records_per_call is not None else 0))
        self.selection = PreparedSparseIndexer(self.quantized, self.scaled_weights, cache,
            self.packed_pages, lengths, active, output, caps=caps, session=session,
            name=name + ".select", workspace=workspace, scores=scores)

    def write_keys(self, keys, slots):
        torch.library.wrap_triton(_write_keys)[(keys.shape[0],)](
            keys, self.cache.view(torch.float8_e4m3fn), self.cache.view(torch.float32),
            slots, self.cache.stride(0), HADAMARD=self.hadamard,
            POWER_OF_TWO_SCALE=self.key_power_of_two, num_warps=4)

    def __call__(self, query, weights):
        self.scaled_weights.copy_(weights)
        self.scaled_weights.mul_(self.weight_normalization)
        glm_pool.quantize_queries(query.view(-1, 128), *self.query_outputs, plan=self.query_plan)
        if self.packed_pages is not self.pages:
            self.packed_pages.copy_(self.pages)
        return self.selection()
