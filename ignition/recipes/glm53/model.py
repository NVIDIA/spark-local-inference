# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical GLM 5.3 Flash structure from a local checkpoint configuration."""

import json
from pathlib import Path


def create_model_tree(snapshot: Path):
    """Construct the canonical tree on meta storage, without checkpoint transfers.

    Packed weight loading, tensor-parallel partitioning, and prepared component
    replacement must precede execution. This function does not initialize CUDA
    or provide a serving forward.
    """
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig
    from transformers.models.glm5_next.modeling_glm5_next import (
        Glm5NextForConditionalGeneration,
    )

    from ignition.components.model import canonical_tree

    raw = json.loads((Path(snapshot) / "config.json").read_text())
    if (raw.get("model_type") != "glm5_next"
            or raw.get("architectures") != ["Glm5NextForConditionalGeneration"]):
        raise ValueError("GLM 5.3 Flash preparation requires Glm5NextForConditionalGeneration")
    return canonical_tree(Glm5NextForConditionalGeneration, Glm5NextConfig.from_dict(raw))
