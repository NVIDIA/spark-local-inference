# SPDX-License-Identifier: Apache-2.0

"""Decode routing exposes all exchange destinations as invocation operands."""

from types import SimpleNamespace

import pytest
import torch

from ignition.recipes.kimi3 import routing


@pytest.mark.parametrize("rows", [1, 8])
def test_route_exchange_consumes_each_invocation_and_preserves_earlier_results(rows):
    class Runtime:
        def all_gather_pair(self, down, router, latent, logits, *, plan, threads):
            latent.copy_(down.repeat(1, 16))
            logits.copy_(router.repeat(1, 16))

        def kimi_topk16(self, logits, correction, weights, ids, *, plan):
            values, indices = (logits + correction).topk(16)
            weights.copy_(values)
            ids.copy_(indices)

        def all_gather_pair_kimi_topk(self, down, router, correction, latent, weights, ids, *, plan):
            logits = router.repeat(1, 16)
            latent.copy_(down.repeat(1, 16))
            self.kimi_topk16(logits, correction, weights, ids, plan=plan)

    owner = routing.PreparedRouteGather.__new__(routing.PreparedRouteGather)
    owner.rows, owner.runtime = rows, Runtime()
    owner.plan = owner.select_plan = SimpleNamespace()
    owner.correction_bias = torch.arange(896, dtype=torch.float32)
    owner.handle = next(routing._HANDLES)
    routing._OWNERS[owner.handle] = owner
    first = owner(torch.ones(rows, 224, dtype=torch.bfloat16), torch.ones(rows, 56))
    second = owner(torch.full((rows, 224), 7, dtype=torch.bfloat16), torch.full((rows, 56), 3.))
    torch.testing.assert_close(first[0], torch.ones(rows, 3584, dtype=torch.bfloat16))
    torch.testing.assert_close(second[0], torch.full((rows, 3584), 7, dtype=torch.bfloat16))
    torch.testing.assert_close(first[1], torch.arange(895, 879, -1, dtype=torch.int32).expand(rows, 16))
    torch.testing.assert_close(second[2], first[2] + 2)
