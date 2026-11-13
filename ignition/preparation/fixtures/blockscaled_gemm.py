# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Finite strided activation trials for blockscaled GEMM preparation."""

from b12x.preparation import PreparedCall


def benchmark_call(state, *, inputs, values, scales, gain, bias, weight,
                   selected_workspace, activation_scale=None):
    operands = inputs()
    storage = selected_workspace(state.required_workspace)

    def run():
        state.run(operands.source, values, scales, gain, out=operands.output, workspace=storage,
                  activation_scale=activation_scale)
        if bias is not None:
            operands.output.add_(bias)
        return operands.output

    return PreparedCall(run=run, produce=operands.produce,
                        output=operands.output, owners=(operands, storage, weight))
