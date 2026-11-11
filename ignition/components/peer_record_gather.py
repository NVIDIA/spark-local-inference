# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared peer reads of selected fixed-width cache records."""

import torch
import triton
import triton.language as tl
from b12x.preparation import PreparedCall, plan_from_handle

from ignition.preparation.tuning import declare, register_finalizer


@torch.library.custom_op("flare::dependent_record_gather", mutates_args=("output",))
def _dependent_gather(peers: torch.Tensor, owners: torch.Tensor, slots: torch.Tensor, counts: torch.Tensor,
                      output: torch.Tensor, dependencies: list[torch.Tensor], handle: int) -> None:
    from b12x.comm import peer_record_gather

    peer_record_gather.gather(plan_from_handle(handle), peers, owners, slots, counts, output)


@_dependent_gather.register_fake
def _fake_dependent_gather(peers, owners, slots, counts, output, dependencies, handle):
    pass


@triton.jit
def _record_address(Records, Address):
    tl.store(Address, tl.cast(Records, tl.uint64).to(tl.int64))


class PreparedPeerRecordGather:
    def __init__(self, peers, owners, slots, counts, output, *, peer_records, session, name, dependencies=()):
        from b12x.comm import peer_record_gather

        rows, topk = owners.shape
        self.arguments = (peers, owners, slots, counts, output)
        self.output = output
        self.dependencies = list(dependencies)
        query = peer_record_gather.PeerRecordGatherQuery(rows=rows, topk=topk,
            world_size=peers.numel(), peer_records=peer_records, record_bytes=output.shape[2])
        self.plan = peer_record_gather.plan(query, device=output.device)
        declare(session, (self.plan.request(name=name, prepare_call=lambda state:
            PreparedCall(run=lambda: state.run(*self.arguments), output=output,
                         owners=self.arguments)),))

    def functional(self, *, session):
        self.output_layout = tuple(self.output.shape), self.output.dtype
        self.peers = self.arguments[0]
        register_finalizer(session, self._release_examples)
        return self

    def _release_examples(self):
        self.arguments = self.output = None
        self.dependencies = []

    def __call__(self, owners=None, slots=None, counts=None, *, records=None, dependencies=()):
        from b12x.comm import peer_record_gather

        if owners is None:
            arguments, output = self.arguments, self.output
        else:
            shape, dtype = self.output_layout
            output = torch.empty(shape, dtype=dtype, device=owners.device)
            peers = self.peers
            if records is not None:
                peers = torch.empty((1,), dtype=torch.int64, device=owners.device)
                torch.library.wrap_triton(_record_address)[(1,)](records, peers)
                dependencies = (*dependencies, records)
            arguments = peers, owners, slots, counts, output
        dependencies = [*self.dependencies, *dependencies]
        if dependencies:
            _dependent_gather(*arguments, dependencies, self.plan.handle)
        else:
            peer_record_gather.gather(self.plan, *arguments)
        return output
