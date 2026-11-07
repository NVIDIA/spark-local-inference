# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared peer reads of selected fixed-width cache records."""

from b12x.preparation import PreparedCall

from ignition.preparation.tuning import declare


class PreparedPeerRecordGather:
    def __init__(self, peers, owners, slots, counts, output, *, peer_records, session, name):
        from b12x.comm import peer_record_gather

        rows, topk = owners.shape
        self.arguments = (peers, owners, slots, counts, output)
        self.output = output
        query = peer_record_gather.PeerRecordGatherQuery(rows=rows, topk=topk,
            world_size=peers.numel(), peer_records=peer_records, record_bytes=output.shape[2])
        self.plan = peer_record_gather.plan(query, device=output.device)
        declare(session, (self.plan.request(name=name, prepare_call=lambda state:
            PreparedCall(run=lambda: state.run(*self.arguments), output=output,
                         owners=self.arguments)),))

    def __call__(self):
        from b12x.comm import peer_record_gather

        peer_record_gather.gather(self.plan, *self.arguments)
        return self.output
