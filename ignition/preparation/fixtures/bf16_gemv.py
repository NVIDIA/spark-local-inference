# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Private activation trials for BF16 projection backend selection."""

from b12x.preparation import PreparedCall


def benchmark_call(state, *, inputs, weight, bias):
    operands = inputs()
    return PreparedCall(
        run=lambda: state.run(operands.source, weight, out=operands.output, bias=bias),
        produce=operands.produce, output=operands.output,
        owners=(operands, weight, bias),
    )
