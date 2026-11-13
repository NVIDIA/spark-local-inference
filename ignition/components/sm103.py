# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compute capability detection for FlashInfer SM100-family operators."""

import torch

CAPABILITY = (10, 3)


def is_sm103(device) -> bool:
    """Whether ``device`` is a CUDA device of compute capability 10.3."""
    device = torch.device(device)
    if device.type != "cuda":
        return False
    index = torch.cuda.current_device() if device.index is None else device.index
    return torch.cuda.get_device_capability(index) == CAPABILITY
