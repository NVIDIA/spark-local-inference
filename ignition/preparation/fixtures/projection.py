# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Finite strided operands shared by a projection's serial tuning candidates."""

from contextlib import nullcontext

import torch

from .resources import candidate_inputs, restore_seed


class Inputs:
    def __init__(self, source, output):
        generator = torch.Generator(device=source.device).manual_seed(0xB12)
        values = torch.randn(source.shape, device=source.device, dtype=source.dtype,
                             generator=generator).mul_(0.125)
        self.seed = torch.empty_strided(source.shape, source.stride(), dtype=source.dtype,
                                       device="cpu", pin_memory=source.is_cuda)
        self.seed.copy_(values)
        # Contiguous inputs can keep the generated allocation; strided inputs
        # preserve their ABI without changing the random sequence's logical order.
        if source.is_contiguous():
            self.source = values.as_strided(source.shape, source.stride())
        else:
            self.source = torch.empty_strided(source.shape, source.stride(),
                                             dtype=source.dtype, device=source.device).copy_(values)
        del values
        self.output = torch.empty_strided(output.shape, output.stride(),
                                         dtype=output.dtype, device=output.device)

    def produce(self):
        restore_seed(self.source, self.seed)


def input_factory(source, output, *, storage=None, session=None):
    def allocate():
        with (nullcontext() if storage is None else storage.transient()):
            return Inputs(source, output)

    return candidate_inputs(allocate, session=session)
