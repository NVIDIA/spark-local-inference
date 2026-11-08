# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Update FlashInfer, install preparation dependencies, and rebuild native extensions."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    try:
        environment = dict(os.environ, BUILD_NVEP="0", FLASHINFER_BUILD_NO_PIP="1")
        subprocess.run(
            ["uv", "sync", "--upgrade-package", "flashinfer-python", "--extra", "prepare", "--extra", "test",
             "--reinstall-package", "flashinfer-python", "--reinstall-package", "flare-heliopause"],
            cwd=ROOT, env=environment, check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"preparation environment setup failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
