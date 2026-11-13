# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preparation of the pinned Qwen3.8 mixed-precision checkpoint.

Persistent-state dataclasses own or borrow allocations retained by the prepared
model until native execution has drained. Captured storage addresses, shapes,
and dtypes stay fixed; request state changes through graph inputs and kernels.
Checkpoint publication additionally requires the corresponding completion fence.

Scratch dataclasses group temporary tensors and views into caller-owned arenas.
Sequential operators may reuse planned scratch, but values needed by another
graph or a later layer belong in persistent storage. Scratch plans must account
for simultaneous operands before capture; runtime execution cannot grow them.

Metadata dataclasses describe capacities, bindings, and ownership. Frozen fields
prevent rebinding, not writes to their tensors. Mutable tensor metadata follows
the upload/graph contract; Python must not change it after native handoff.
"""

__all__ = ["create_model_tree"]


def __getattr__(name):
    # Recipe declarations import without the model tree's PyTorch dependencies.
    if name == "create_model_tree":
        from .model import create_model_tree

        return create_model_tree
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
