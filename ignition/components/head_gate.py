"""Prepared BF16 head gates with explicit gate row strides."""

import torch
from b12x.norm.hyperconnection import api as hyperconnection
from b12x.preparation import PreparedCall

from ignition.preparation.fixtures.hyperconnection import HyperconnectionBenchmarkInputs
from ignition.preparation.tuning import benchmark_resource, declare


class PreparedHeadGate:
    def __init__(self, values, gates, *, session, name):
        if (values.ndim != 3 or values.dtype != torch.bfloat16 or not values.is_contiguous()
                or gates.shape != values.shape[:2] or gates.dtype != values.dtype
                or gates.device != values.device or gates.stride(1) != 1
                or gates.stride(0) < gates.shape[1]):
            raise ValueError("head gates require contiguous BF16 values and matching strided gate rows")
        value_start, gate_start = values.data_ptr(), gates.data_ptr()
        gate_bytes = ((gates.shape[0] - 1) * gates.stride(0) + gates.shape[1]) * gates.element_size()
        if value_start < gate_start + gate_bytes and gate_start < value_start + values.numel() * values.element_size():
            raise ValueError("head gate logits must not overlap mutable values")
        self.values, self.gates = values, gates
        self.plan = hyperconnection.plan(hyperconnection.Caps(device=values.device,
            max_tokens=values.shape[0], streams=values.shape[1], hidden_size=values.shape[2],
            lowrank=1, dtype=values.dtype),
            invocation={"operation": "head_gate", "gate_stride": gates.stride(0)})

        def benchmark(state):
            operands = benchmark_resource(session, ("head_gate", name),
                lambda: HyperconnectionBenchmarkInputs("head_gate", (values, gates, values)))
            v, g, _ = operands.arguments
            return PreparedCall(run=lambda: state.launch(v, g, v), output=v,
                                produce=operands.produce, owners=(operands,))

        declare(session, (self.plan.request(name=name,
            prepare_call=lambda state: PreparedCall(run=lambda: state.launch(values, gates, values)),
            benchmark_call=benchmark),))

    def functional(self, *, session):
        """Consume forward-local values and gates with the prepared row strides."""
        del session
        self.values = self.gates = None
        return self

    def __call__(self, values=None, gates=None):
        values = self.values if values is None else values
        gates = self.gates if gates is None else gates
        if values is None or gates is None:
            raise ValueError("head gate requires values and gates")
        torch.ops.b12x.hyperconnection_head_gate(values, gates, self.plan.handle)
        return values
