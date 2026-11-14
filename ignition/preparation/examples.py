# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Share preparation operands while functional forwards allocate their own values."""

from math import prod

import torch


def declaration_tensor(examples, role, shape, dtype, device):
    device = torch.device(device)
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    if examples is None:
        return torch.zeros(shape, dtype=dtype, device=device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    key = role, shape, dtype, device
    if key not in examples:
        storage_key = role, dtype, device
        storage = examples.get(storage_key)
        elements = prod(shape)
        if storage is None or storage.numel() < elements:
            storage = examples[storage_key] = torch.zeros(shape, dtype=dtype, device=device)
        examples[key] = storage.view(-1)[:elements].view(shape)
    return examples[key]
