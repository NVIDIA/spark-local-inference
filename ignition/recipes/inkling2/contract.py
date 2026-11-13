# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared graph inputs and cache storage of the Inkling 2 text graphs.

Each request owns one continuation row: the local layers' K/V rings, every
layer's convolution histories, partial pages of the global layers, the last
normalized hidden state and the consumed-token count. The eight global layers
keep their K and V in 128-token pages, one paged history region per layer and
kind, addressed through the request's page table.
"""

from ignition.preparation import contract as c
from ignition.preparation import runtime_core_pb2 as p

from .pool import PAGE_TOKENS


def prepare(signatures, buffers, pool, *, vocabulary, input_vocabulary, frontend):
    kinds = {signature.kind for signature in signatures.values()}
    inputs = {kind: c.text_inputs(kind) for kind in kinds}
    continuation = c.state_view("state_pool", pool.layout.row_bytes)
    regions = [c.history_region(buffers, "cache." + name, tensor, p.PAGED_KEY_VALUE)
               for name, tensor in pool.pages.items()]
    graphs = c.graph_contracts(signatures, inputs, continuation=continuation,
                               pending=c.state_view("request_pending_token", 8))
    return c.model_contract(vocabulary=vocabulary, input_vocabulary=input_vocabulary, page_tokens=PAGE_TOKENS,
        page_tables=[dict(buffer="page_tables", page_offset=0)],
        regions=[dict(kind=p.CONTINUATION_STATE, storage=continuation, tokens_per_record=1, window_tokens=0),
                 *regions],
        graphs=graphs, frontend=frontend)
