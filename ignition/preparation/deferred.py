# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA tensor metadata for compilation before persistent storage allocation."""

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode


def deferred_tensor(shape, *, dtype, device, mode):
    """Describe a contiguous CUDA allocation without reserving device memory.

    Replace this tensor with real storage of the same shape and dtype before
    warming or capturing its compiled forward.
    """
    return FakeTensor(mode, torch.empty(shape, dtype=dtype, device="meta"), torch.device(device))


def deferred_mode():
    return FakeTensorMode(allow_non_fake_inputs=True)
