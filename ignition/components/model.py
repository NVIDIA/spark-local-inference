# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical Transformers structure for prepared model components."""

import torch


def canonical_tree(model_type, config):
    """Construct geometry and component methods without allocating model weights.

    Recipes replace execution and attach checkpoint tensors before capture.
    The canonical attention setting supplies module construction semantics;
    serving executes only the recipe's captured graphs.
    """
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = model_type(config)
    return model.eval().requires_grad_(False)
