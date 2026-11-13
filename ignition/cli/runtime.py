# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Locate the native Flare executable used by checkpoint distribution."""

import os
from pathlib import Path
import shutil

from ignition.errors import UsageError


def binary():
    installed = shutil.which("flare")
    if installed:
        return Path(installed)
    root = Path(__file__).resolve().parents[2]
    executable = root / "flare-cli/target/release" / ("flare.exe" if os.name == "nt" else "flare")
    if not executable.is_file():
        raise UsageError("flare executable is missing; build flare-cli/Cargo.toml or install flare on PATH")
    return executable
