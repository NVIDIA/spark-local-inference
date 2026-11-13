# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical Transformers module structure for Qwen graph preparation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from typing import TYPE_CHECKING

from ignition.preparation.checkpoint import CheckpointError, CheckpointManifest
from ignition.preparation.qwen38 import TRANSFORMERS_REVISION

if TYPE_CHECKING:
    from transformers import Qwen4ExpForConditionalGeneration


def create_model_tree(manifest: CheckpointManifest) -> Qwen4ExpForConditionalGeneration:
    """Construct the pinned canonical module tree on meta storage without loading weights.

    The returned tree supplies module names, geometry, and canonical component
    methods. Quantized weights and prepared forwards must be attached before
    execution; Transformers' generic ModelOpt checkpoint loader is unsupported.
    """
    distribution = importlib.metadata.distribution("transformers")
    provenance = json.loads(distribution.read_text("direct_url.json") or "{}")
    if provenance.get("vcs_info", {}).get("commit_id") != TRANSFORMERS_REVISION:
        raise RuntimeError(f"Qwen preparation requires Transformers {TRANSFORMERS_REVISION}")
    raw = (manifest.snapshot / "config.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest.metadata_sha256["config.json"]:
        raise CheckpointError("config.json changed after checkpoint inspection")

    from transformers import Qwen4ExpConfig, Qwen4ExpForConditionalGeneration

    from ignition.components.model import canonical_tree

    return canonical_tree(Qwen4ExpForConditionalGeneration,
                          Qwen4ExpConfig.from_dict(json.loads(raw)))
