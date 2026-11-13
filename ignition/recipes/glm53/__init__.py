# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM 5.3 Flash model construction for b12x graph preparation."""

__all__ = ["create_model_tree"]


def __getattr__(name):
    # Recipe declarations import without the model tree's PyTorch dependencies.
    if name == "create_model_tree":
        from .model import create_model_tree

        return create_model_tree
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
