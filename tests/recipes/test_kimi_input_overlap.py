# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi MoE projection forks join before routing and shared scratch reuse."""

import pytest
import torch

from ignition.recipes.kimi3 import decoder


class Shared:
    def __init__(self, events, concurrent, output):
        self.events, self.concurrent, self.output = events, concurrent, output
        self.pending = None

    def start(self, source):
        assert self.pending is None
        self.events.append('shared_start')
        self.pending = source

    def finish(self, source, routed):
        assert self.pending is source
        assert torch.isfinite(routed).all()
        self.events.append('shared_join')
        self.output.copy_(self.pending * 5)
        self.pending = None
        return self.output

    def __call__(self, source):
        self.events.append('shared')
        self.output.copy_(source * 5)
        return self.output


class ProjectionFork:
    def __init__(self, events, project):
        self.events, self.project = events, project
        self.pending = None

    def start(self, source):
        assert self.pending is None
        self.events.append('down_start')
        self.pending = source

    def finish(self, source, logits):
        assert self.pending is source
        assert torch.isfinite(logits).all()
        self.project(source)
        self.events.append('down_join')
        self.pending = None


@pytest.mark.parametrize('rows,compact', [(1, True), (8, True), (9, False)])
@pytest.mark.parametrize('shared_overlap', [False, True])
@pytest.mark.parametrize('projection_overlap', [False, True])
def test_projection_and_shared_forks_preserve_outputs_and_balance_joins(
        monkeypatch, rows, compact, shared_overlap, projection_overlap):
    events = []
    source = torch.empty(rows, 4)
    logits = torch.empty(rows, 4)
    prepared = decoder.PreparedLatentMoE.__new__(decoder.PreparedLatentMoE)
    prepared.source = torch.empty_like(source)
    prepared.hidden_columns = torch.arange(4)
    prepared.latent = torch.empty_like(source)
    prepared.latent_sum = prepared.latent
    prepared.expert_output = torch.empty_like(source)
    prepared.up_input = torch.empty_like(source)
    prepared.output = torch.empty_like(source)
    prepared.latent_norm, prepared.eps, prepared.latent_begin = None, 0., None
    prepared.shared = Shared(events, shared_overlap, torch.empty_like(source))

    def router_projection(value):
        events.append('router')
        logits.copy_(value + 7)
        return logits

    def down(value):
        events.append('down')
        output = prepared.latent if compact else prepared.expert_output
        output.copy_(value * 2)
        return output

    def route():
        events.append('route')
        assert torch.isfinite(logits).all()
        assert torch.isfinite(prepared.latent).all()

    def experts():
        events.append('experts')
        prepared.expert_output.copy_(prepared.latent + 1)
        return prepared.expert_output

    def reduce(value):
        prepared.latent_sum.copy_(value)

    def normalize(value, weight, output, eps, *, begin):
        output.copy_(value)

    def up(value):
        events.append('up')
        prepared.output.copy_(value * 3)
        return prepared.output

    prepared.router_projection, prepared.down = router_projection, down
    prepared.route_gather = route if compact else None
    prepared.concurrent_down = (ProjectionFork(events, down) if compact and projection_overlap else None)
    prepared.experts, prepared.expert_reduce, prepared.up = experts, reduce, up
    if not compact:
        prepared.router_input = torch.empty_like(logits)
        prepared.router_columns = torch.empty_like(logits)
        prepared.router_output = torch.empty_like(logits)
        prepared.router_logits = prepared.router_output

        def gather(value):
            prepared.router_columns.copy_(value.view(torch.float32))

        prepared.router_gather = gather
        prepared.router = lambda value: events.append('route')
        prepared.down_reduce = reduce
    monkeypatch.setattr(decoder, 'rms_norm_columns', normalize)
    addresses = tuple(value.data_ptr() for value in (
        prepared.latent, prepared.expert_output, prepared.up_input, prepared.output, prepared.shared.output))
    for initial in (-3., 0., 17.):
        events.clear()
        source.fill_(initial)
        for value in (logits, prepared.latent, prepared.expert_output, prepared.output, prepared.shared.output):
            value.fill_(float('nan'))
        output = prepared(source)
        torch.testing.assert_close(output, source * 11 + 3, rtol=0, atol=0)
        assert output is prepared.output
        assert prepared.shared.pending is None
        if prepared.concurrent_down is not None:
            assert prepared.concurrent_down.pending is None
            assert events.index('down_start') < events.index('router') < events.index('down_join')
            assert events.index('down_join') < events.index('route')
        if shared_overlap:
            assert events.index('router') < events.index('shared_start') < events.index('route')
            assert events.index('up') < events.index('shared_join')
            if compact:
                assert events.index('down') < events.index('shared_start')
        else:
            assert events.index('up') < events.index('shared')
            assert 'shared_start' not in events and 'shared_join' not in events
        assert addresses == tuple(value.data_ptr() for value in (
            prepared.latent, prepared.expert_output, prepared.up_input, prepared.output, prepared.shared.output))
