# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Fetch and annotate the canonical Google frontend for each model size."""

from pathlib import Path

from ignition.preparation.frontend import downloaded_template_asset


def chat_template_asset(hidden_size: int, *, cache_dir=None):
    """Use Google's canonical template for the model size across quantization publishers."""
    if hidden_size in (1536, 2560):
        filename = "chat_template.google-e2b-e4b.patch.json"
    elif hidden_size in (2816, 3840, 5376):
        filename = "chat_template.google.patch.json"
    else:
        raise ValueError(f"unsupported Gemma 4 chat template hidden size: {hidden_size}")
    path = Path(__file__).with_name("assets").joinpath(filename).resolve()
    return downloaded_template_asset(path, cache_dir=cache_dir)
