# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3 decode routing: latent-down and router shards gathered with expert selection.

Each rank projects its rows of the routed-expert down projection and its experts' router logits.
One b12x exchange gathers both shards; a single decode row also selects Kimi's 16 experts inside
that exchange, and two to eight rows select them with b12x's prepared top-16 afterwards. This
replaces the router all-gather, the separate top-k, and the latent all-reduce of the general path.
"""

import itertools
import weakref

import torch
from b12x.preparation import CollectiveRequirement, PreparedCall

from ignition.preparation.tuning import declare, register_finalizer
from ignition.components.moe import RouterScratch

MAX_ROWS = 8
_OWNERS = weakref.WeakValueDictionary()
_HANDLES = itertools.count()


def create_route_runtime(group, device, *, ipc=None):
    """Allocate the b12x exchange that carries 1-8 rows of K3 latent and router shards."""
    import torch.distributed as dist
    from b12x.comm.pcie import DcpAllToAll

    world_size = dist.get_world_size(group)
    if world_size < 2 or 3584 % world_size or 896 % world_size:
        raise ValueError("K3 route exchange requires an even latent and expert partition")
    return DcpAllToAll.from_exchange_group(exchange_group=group, device=device, max_batch_size=MAX_ROWS,
        total_heads=world_size, head_dim=8, query_head_dim=(3584 * 2 + 896 * 4) // world_size,
        stream_affine=False, ipc=ipc)


class PreparedRouteGather:
    """Gather latent and FP32 router shards into full rows, then select 16 experts per row.

    The latent, expert weights and expert IDs are caller-owned outputs read by the routed experts.
    Every rank must declare the same operation names and replay them serially.
    """

    def __init__(self, local_down, local_router, correction_bias, latent, weights, ids, *, runtime, session,
                 coordinator, name):
        import torch.distributed as dist
        from b12x.comm import pcie

        rows = local_down.shape[0]
        world = runtime.world_size
        expected = ((local_down, (rows, 3584 // world), torch.bfloat16), (local_router, (rows, 896 // world), torch.float32),
                    (correction_bias, (896,), torch.float32), (latent, (rows, 3584), torch.bfloat16),
                    (weights, (rows, 16), torch.float32), (ids, (rows, 16), torch.int32))
        wrong = [(tuple(t.shape), t.dtype, t.is_contiguous(), shape, dtype) for t, shape, dtype in expected
                 if t.shape != shape or t.dtype != dtype or not t.is_contiguous()]
        if not 1 <= rows <= MAX_ROWS or wrong:
            raise ValueError(f"K3 route gather requires contiguous 1-8 row latent, router and route tensors: {wrong}")
        self.runtime, self.rows = runtime, rows
        self.local_down, self.local_router, self.correction_bias = local_down, local_router, correction_bias
        self.latent, self.weights, self.ids = latent, weights, ids
        ranks = tuple(dist.get_process_group_ranks(runtime.exchange_group))
        requests = []
        if rows == 1:
            self.fused = dict(local_down=local_down, local_router=local_router, correction_bias=correction_bias,
                              out_down=latent, topk_weights=weights, topk_ids=ids)
            self.plan = pcie.plan(pcie.query_from_runtime(runtime, surface="DcpAllToAll.all_gather_pair_kimi_topk",
                                                          call=self.fused), runtime=runtime)
            requests.append(self.plan.request(name=name, prepare_call=lambda state: PreparedCall(
                run=lambda: state.run(**self.fused), output=latent, owners=(runtime,)),
                collective=CollectiveRequirement(key=name, ranks=ranks)))
        else:
            self.logits = torch.zeros((rows, 896), dtype=torch.float32, device=latent.device)
            self.pair = dict(local_first=local_down, local_second=local_router, out_first=latent,
                             out_second=self.logits, threads=512)
            self.select = dict(router_logits=self.logits, correction_bias=correction_bias,
                               output_weights=weights, output_ids=ids)
            self.plan = pcie.plan(pcie.query_from_runtime(runtime, surface="DcpAllToAll.all_gather_pair",
                                                          call=self.pair), runtime=runtime)
            self.select_plan = pcie.plan(pcie.query_from_runtime(runtime, surface="DcpAllToAll.kimi_topk16",
                                                                 call=self.select), runtime=runtime)
            requests.append(self.plan.request(name=name + "/gather", prepare_call=lambda state: PreparedCall(
                run=lambda: state.run(**self.pair), output=latent, owners=(runtime,)),
                collective=CollectiveRequirement(key=name + "/gather", ranks=ranks)))
            requests.append(self.select_plan.request(name=name + "/select", prepare_call=lambda state: PreparedCall(
                run=lambda: state.run(**self.select), output=weights, owners=(runtime,))))
        declare(session, tuple(requests), coordinator=coordinator)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self
        def release_examples():
            self.local_down = self.local_router = self.latent = self.weights = self.ids = None
            self.fused = self.pair = self.select = self.logits = None
        register_finalizer(session, release_examples)

    def __call__(self, local_down, local_router):
        latent = torch.empty((self.rows, 3584), dtype=local_down.dtype, device=local_down.device)
        weights = torch.empty((self.rows, 16), dtype=torch.float32, device=local_down.device)
        ids = torch.empty((self.rows, 16), dtype=torch.int32, device=local_down.device)
        logits = torch.empty((self.rows, 896) if self.rows > 1 else (0,), dtype=torch.float32, device=local_down.device)
        _route_gather(local_down, local_router, latent, weights, ids, logits, self.handle)
        return latent, RouterScratch(torch.zeros_like(weights), ids, weights)


@torch.library.custom_op("flare::kimi3_route_gather", mutates_args=("latent", "weights", "ids", "logits"))
def _route_gather(local_down: torch.Tensor, local_router: torch.Tensor, latent: torch.Tensor,
                  weights: torch.Tensor, ids: torch.Tensor, logits: torch.Tensor, handle: int) -> None:
    owner = _OWNERS[handle]
    runtime = owner.runtime
    if owner.rows == 1:
        runtime.all_gather_pair_kimi_topk(local_down, local_router, owner.correction_bias, latent, weights, ids,
                                          plan=owner.plan)
        return
    runtime.all_gather_pair(local_down, local_router, latent, logits, plan=owner.plan, threads=512)
    runtime.kimi_topk16(logits, owner.correction_bias, weights, ids, plan=owner.select_plan)


@_route_gather.register_fake
def _fake_route_gather(local_down, local_router, latent, weights, ids, logits, handle):
    pass
