# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kimi projections retain BF16 activations with checkpoint or online MXFP8 weights."""

import torch

from ignition.components.linear import QuantizedProjection, UnquantizedProjection


def prepare_projection(source, weight, output, *, session, name, workspace):
    if isinstance(weight, torch.Tensor):
        return UnquantizedProjection(source, weight, output, session=session, name=name)
    return QuantizedProjection(source, weight, output, session=session, name=name,
                               activation_mode="a16", workspace=workspace)
