# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared grouped low-rank attention output projection through b12x."""

import torch
from b12x.gemm import wo_projection
from b12x.preparation import PreparedCall

from ignition.preparation.scratch import planned_scratch, prime_owners
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import declare, register_finalizer


class PreparedGroupedOutputProjection:
    def __init__(self, source, weights, output, *, session, name, workspace):
        self.weights, self.output, self.workspace = weights, output, workspace
        if source.shape[1:] != (weights.groups, weights.group_width):
            raise ValueError("attention output must match the packed projection groups and head width")
        self.plan = wo_projection.plan(wo_projection.Caps(device=source.device, max_tokens=source.shape[0],
            groups=weights.groups, group_width=weights.group_width, rank=weights.rank, hidden=weights.hidden))
        self.operands = [tensor for packed in (weights.wo_a, weights.wo_b)
                         for tensor in (packed.values, packed.scale_rows.view(torch.uint8),
                                        packed.scale_mma.view(torch.uint8))]
        for tensor in self.operands:
            record_numeric_tensor(tensor)

        def call(state, values, out, *, produce=None, owners=()):
            scratch = planned_scratch(state._scratch_state.scratch_specs(), workspace)
            binding = state.bind(source_tgd=values, weights=weights, scratch=scratch, out=out)
            return PreparedCall(run=lambda: state.run(binding), output=out, produce=produce,
                owners=(*owners, weights, *prime_owners(workspace, scratch, binding)))

        def benchmark(state):
            generator = torch.Generator(device=source.device).manual_seed(4121)
            seed = torch.randn(source.shape, dtype=source.dtype, device=source.device, generator=generator) * .125
            values, out = torch.empty_like(source), torch.empty_like(output)
            return call(state, values, out, produce=lambda: values.copy_(seed), owners=(values, out, seed))

        declare(session, (self.plan.request(name=name,
            prepare_call=lambda state: call(state, source, output), benchmark_call=benchmark),))
        register_finalizer(session, self.rebind, workspace=workspace)

    def rebind(self):
        self.scratch = planned_scratch(self.plan.scratch_specs(), self.workspace)

    def functional(self, *, session):
        self.output_layout = (tuple(self.output.shape), tuple(self.output.stride()), self.output.dtype)
        register_finalizer(session, self._finalize_functional)
        return self

    def _finalize_functional(self):
        self.scratch_layout = tuple((tuple(spec.shape), spec.dtype) for spec in self.plan.scratch_specs())
        self.output = self.workspace = self.scratch = None

    def __call__(self, source):
        output, scratch = self.output, self.scratch
        if output is None:
            shape, stride, dtype = self.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device).zero_()
            scratch, = (torch.empty(shape, dtype=dtype, device=source.device) for shape, dtype in self.scratch_layout)
        wo_projection.run_out(source, self.weights, out=output, scratch=scratch, plan=self.plan)
        return output
