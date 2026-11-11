# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Independent Torch tensors for preparation declarations in tests."""

import torch


class TensorExamples:
    def __init__(self, device):
        self.device = torch.device(device)

    def empty(self, shape, dtype):
        return torch.empty(shape, dtype=dtype, device=self.device)

    def zeros(self, shape, dtype):
        return torch.zeros(shape, dtype=dtype, device=self.device)

    def full(self, shape, value, dtype):
        return torch.full(shape, value, dtype=dtype, device=self.device)
