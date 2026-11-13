# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared gate projection and stream mean with caller-owned outputs."""

from functools import partial

import torch
from b12x.gemm import hyperconnection_gate
from b12x.norm.hyperconnection import api as hc
from b12x.preparation import FrozenMapping, PreparedCall

from ignition.components.collectives import PreparedChunkedAllReduce
from ignition.components.linear import UnquantizedProjection
from ignition.preparation.fixtures import hyperconnection as hc_fixtures
from ignition.preparation.fixtures.hyperconnection_gate import HyperconnectionGateInputs
from ignition.preparation.tuning import benchmark_resource, declare, register_finalizer


class PreparedHyperconnectionGate:
    def __init__(self, source, weight, normalized, output, workspace, *, session, name):
        rows, rank = source.shape
        hidden = output.shape[1]
        streams, remainder = divmod(normalized.shape[1], hidden)
        if remainder:
            raise ValueError("normalized width must contain complete residual streams")
        self.plan = hyperconnection_gate.plan(hyperconnection_gate.Query(
            max_rows=rows, rank=rank, hidden_size=hidden, streams=streams))
        self.weight, self.output, self.workspace = weight, output, workspace

        def benchmark(state):
            inputs = benchmark_resource(session,
                ("hyperconnection_gate", str(source.device), rows, rank, hidden, streams),
                lambda: HyperconnectionGateInputs(source, normalized, output, workspace))
            return inputs.call(state, weight)

        declare(session, (self.plan.request(
            name=name,
            prepare_call=lambda state: PreparedCall(
                run=lambda: state.run(source, weight, normalized, output, workspace)),
            benchmark_call=benchmark,
        ),))

    def functional(self, *, session):
        self.output_shape, self.workspace_shape = tuple(self.output.shape), tuple(self.workspace.shape)
        self.output = self.workspace = None
        return self

    def __call__(self, source, normalized):
        output = torch.empty(self.output_shape, dtype=normalized.dtype, device=normalized.device)
        workspace = torch.empty(self.workspace_shape, dtype=normalized.dtype, device=normalized.device)
        return hyperconnection_gate.run(source, self.weight, normalized, out=output,
                                       workspace=workspace, plan=self.plan)


class PreparedShardedHyperconnectionGate:
    """Reduce BF16 partial gate logits before sigmoid and the residual stream mean.

    Each rank owns matching columns of the gate weight and bottleneck activation.
    Normalized residual streams are replicated; partial and reduced logits require
    separate caller-owned BF16 storage. Collectives execute in the same order on
    every rank, including inactive rows.
    """

    def __init__(self, source, weight, normalized, output, partial_logits, logits, *,
                 runtime, coordinator, session, name):
        rows, lowrank = source.shape
        hidden = output.shape[1]
        streams, remainder = divmod(normalized.shape[1], hidden)
        if remainder or runtime is None:
            raise ValueError("sharded gates require complete residual streams and a TP collective")
        self.up = UnquantizedProjection(source, weight, partial_logits,
            session=session, name=f"{name}.up")
        self.reduce = PreparedChunkedAllReduce(partial_logits, logits, runtime=runtime,
            session=session, coordinator=coordinator, name=f"{name}.reduce")
        self.logits, self.output = logits, output
        caps = hc.Caps(device=source.device, max_tokens=rows, hidden_size=hidden,
                       streams=streams, lowrank=lowrank)
        self.plan = hc.plan(caps, invocation=FrozenMapping({"operation": "gate_mean"}))
        arguments = (normalized, logits, output)
        declare(session, (self.plan.request(name=f"{name}.mean",
            prepare_call=lambda state: PreparedCall(run=lambda: state.launch(*arguments)),
            benchmark_call=partial(hc_fixtures.benchmark_call, operation="gate_mean",
                                   arguments=arguments, kwargs={})),))

        def finalize():
            self.binding = hc.bind(self.plan, tokens=rows, normalized=normalized,
                                   bottleneck=source, block_input=output)

        register_finalizer(session, finalize)

    def functional(self, *, session):
        self.output_shape = tuple(self.output.shape)
        self.up.functional(session=session)
        self.reduce.functional(session=session)
        self.logits = self.output = None
        register_finalizer(session, lambda: setattr(self, "binding", None))
        return self

    def __call__(self, source, normalized):
        logits = self.reduce(self.up(source))
        output = torch.empty(self.output_shape, dtype=normalized.dtype, device=normalized.device)
        torch.ops.b12x.hyperconnection_gate_mean(normalized, logits, output, self.plan.handle)
        return output
