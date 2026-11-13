# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private finite token and target-state operands for MTP feedback tuning."""

import torch
from b12x.preparation import PreparedCall

from ignition.preparation.scratch import planned_scratch
from ignition.preparation.tuning import benchmark_resource


class FeedbackInputs:
    def __init__(self, token_embedding, multi_state, output):
        self.token_embedding = torch.full_like(token_embedding, 0.125)
        self.multi_state = torch.full_like(multi_state, 0.25)
        self.output = torch.empty_like(output)

    @property
    def nbytes(self):
        return sum(x.nbytes for x in (self.token_embedding, self.multi_state, self.output))

    def produce(self):
        self.token_embedding.fill_(0.125)
        self.multi_state.fill_(0.25)


def benchmark_call(state, *, arguments, session, name, workspace, eps):
    operands = benchmark_resource(session, ("qwen_mtp_feedback", name),
        lambda: FeedbackInputs(arguments["token_embedding"], arguments["multi_state"], arguments["output"]))
    scratch = planned_scratch(state.layout.scratch_specs(), workspace)
    kwargs = {**arguments, "token_embedding": operands.token_embedding,
              "multi_state": operands.multi_state, "output": operands.output}
    binding = state.bind(scratch=scratch, **kwargs)
    return PreparedCall(run=lambda: state.run(binding, eps=eps),
        produce=operands.produce, output=operands.output, owners=(operands, scratch, binding))
