# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared peer reads of selected fixed-width cache records."""

import torch
from b12x.preparation import PreparedCall, plan_from_handle

from ignition.preparation.tuning import declare


@torch.library.custom_op("flare::dependent_record_gather", mutates_args=("output",))
def _dependent_gather(peers: torch.Tensor, owners: torch.Tensor, slots: torch.Tensor, counts: torch.Tensor,
                      output: torch.Tensor, dependencies: list[torch.Tensor], handle: int) -> None:
    from b12x.comm import peer_record_gather

    peer_record_gather.gather(plan_from_handle(handle), peers, owners, slots, counts, output)


@_dependent_gather.register_fake
def _fake_dependent_gather(peers, owners, slots, counts, output, dependencies, handle):
    pass


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

    def __call__(self):
        from b12x.comm import peer_record_gather

        if self.dependencies:
            _dependent_gather(*self.arguments, self.dependencies, self.plan.handle)
        else:
            peer_record_gather.gather(self.plan, *self.arguments)
        return self.output
