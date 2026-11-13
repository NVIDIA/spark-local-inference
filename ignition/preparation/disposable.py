# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Identify explicitly declared disposable backing storage for core export."""

from dataclasses import fields, is_dataclass

import torch


def storage_ranges(values):
    """Deduplicate backing storage for explicitly declared scratch tensor trees."""
    storage = {}

    def visit(value):
        # Integer capacities are host metadata and own no backing storage.
        if value is None or isinstance(value, int):
            return
        if isinstance(value, torch.Tensor):
            if value.is_meta:
                raise ValueError("export scratch must have materialized storage")
            backing = value.untyped_storage()
            if backing.nbytes():
                storage[backing.data_ptr()] = backing.nbytes()
        elif is_dataclass(value):
            for field in fields(value):
                visit(getattr(value, field.name))
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)
        else:
            raise TypeError(f"unsupported scratch declaration: {type(value).__name__}")

    visit(values)
    return tuple(sorted(storage.items()))
