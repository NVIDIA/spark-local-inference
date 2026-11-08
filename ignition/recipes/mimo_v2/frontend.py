# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Annotate a MiMo checkpoint's chat template with the bundled patch matching its bytes."""

import hashlib
import json
from pathlib import Path

from ignition.preparation.frontend import annotated_template_asset

ASSETS = Path(__file__).with_name("assets")


def chat_template_patch(source: bytes) -> Path:
    """Return the bundled patch whose pinned source digest matches `source`."""
    digest = hashlib.sha256(source).hexdigest()
    for path in sorted(ASSETS.glob("chat_template*.patch.json")):
        if json.loads(path.read_text())["source_sha256"] == digest:
            return path
    raise ValueError("no bundled chat template patch matches the checkpoint's chat_template.jinja")


def chat_template_asset(snapshot):
    source = (Path(snapshot) / "chat_template.jinja").read_bytes()
    return annotated_template_asset(source, chat_template_patch(source))
