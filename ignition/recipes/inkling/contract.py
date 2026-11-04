# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text graph bindings, complete continuation rows, and paged attention histories."""

from ignition.preparation import contract as c
from ignition.preparation import runtime_package_pb2 as p

from .frontend import FRONTEND
from .pool import PAGE_TOKENS


def prepare(signatures, buffers, pool, *, vocabulary):
    inputs = {kind: c.text_inputs(kind) for kind in {s.kind for s in signatures.values()}}
    for kind in ("decode", "speculative"):
        if kind in inputs:
            inputs[kind].pop("token_ids", None)
    continuation = c.state_view("state_pool", pool.layout.row_bytes)
    regions = [c.history_region(buffers, f"cache.{name}", tensor, p.PAGED_KEY_VALUE)
               for name, tensor in pool.pages.items()]
    return c.model_contract(vocabulary=vocabulary, page_tokens=PAGE_TOKENS,
        page_tables=[dict(buffer="page_tables", page_offset=0)],
        regions=[dict(kind=p.CONTINUATION_STATE, storage=continuation,
                      tokens_per_record=1, window_tokens=0), *regions],
        graphs=c.graph_contracts(signatures, inputs, continuation=continuation,
                                 pending=c.state_view("request_pending_token", 8)), frontend=FRONTEND)
