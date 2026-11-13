# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Private activations and output storage for complete stream-gating trials."""

import torch
from b12x.preparation import PreparedCall


class HyperconnectionGateInputs:
    def __init__(self, source, normalized, output, workspace):
        self.source = torch.empty_like(source)
        self.normalized = torch.empty_like(normalized)
        self.output = torch.empty_like(output)
        self.workspace = torch.empty_like(workspace)
        generator = torch.Generator(device=source.device).manual_seed(3187)
        self.seeds = tuple(torch.randn(t.shape, dtype=t.dtype, device=t.device,
                                      generator=generator) * .125
                           for t in (source, normalized))
        self.nbytes = sum(t.nbytes for t in (self.source, self.normalized, self.output,
                                           self.workspace, *self.seeds))

    def produce(self):
        self.source.copy_(self.seeds[0])
        self.normalized.copy_(self.seeds[1])

    def call(self, state, weight):
        return PreparedCall(
            run=lambda: state.run(self.source, weight, self.normalized, self.output, self.workspace),
            produce=self.produce, output=self.output, owners=(self, weight),
        )
