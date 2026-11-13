# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared b12x embedding lookup with fixed output and device live counts."""

import torch
from b12x.preparation import PreparedCall
from b12x.sequence import embedding

from ignition.preparation.tuning import declare
from .collectives import PreparedChunkedAllGather


class PreparedEmbedding:
    """Gather token rows into caller storage and zero inactive output rows.

    Token IDs and the optional live count remain device inputs during graph
    replay. The fixed b12x kernel has no workspace or tuning candidates.
    """

    def __init__(self, weight, ids, output, *, session, name, num_rows=None):
        self.weight, self.ids, self.output, self.num_rows = weight, ids, output, num_rows
        self.arguments = {"out": output, "num_rows": num_rows}
        self.plan = embedding.plan(embedding.query_from_call(weight, ids, **self.arguments), device=weight.device)
        def prepare(state):
            prime_ids = torch.zeros_like(ids)
            arguments = {"out": output, "num_rows": None if num_rows is None else torch.zeros_like(num_rows)}
            return PreparedCall(run=lambda: state.run(weight, prime_ids, **arguments))

        declare(session, (self.plan.request(name=name, prepare_call=prepare),))

    def functional(self, *, session, dynamic_inputs=False):
        """Allocate embedding rows inside the traced forward."""
        if self.output is not None:
            self.output_layout = (tuple(self.output.shape), self.output.stride(), self.output.dtype)
            self.output = None
            self.arguments = None
        if dynamic_inputs:
            self.ids = self.num_rows = None
        return self

    def __call__(self, ids=None, *, token_ids=None, num_rows=None):
        if token_ids is not None:
            if ids is not None:
                raise ValueError("embedding token IDs must be supplied once")
            ids = token_ids
        ids = self.ids if ids is None else ids
        num_rows = self.num_rows if num_rows is None else num_rows
        output = self.output
        if output is None:
            shape, stride, dtype = self.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=ids.device)
        if num_rows is not None:
            output.zero_()
        return embedding.run(self.weight, ids, out=output, num_rows=num_rows, plan=self.plan)


class PreparedShardedEmbedding:
    """Gather column-sharded vocabulary rows through a prepared collective."""

    def __init__(self, weight, ids, local_output, output, *, runtime, coordinator, session, name, num_rows=None):
        self.lookup = PreparedEmbedding(weight, ids, local_output, session=session, name=name + ".lookup",
                                        num_rows=num_rows)
        self.gather = PreparedChunkedAllGather(local_output, output, runtime=runtime, coordinator=coordinator,
                                               session=session, name=name + ".gather")

    def functional(self, *, session, dynamic_inputs=False):
        """Allocate lookup and collective destinations inside the traced forward."""
        self.lookup.functional(session=session, dynamic_inputs=dynamic_inputs)
        self.gather.functional(session=session)
        return self

    def __call__(self, ids=None, *, token_ids=None, num_rows=None):
        return self.gather(self.lookup(ids, token_ids=token_ids, num_rows=num_rows))
