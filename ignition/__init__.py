# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python model preparation and CUDA graph capture for Flare."""

import sys
from pathlib import Path

# The TrellisMX DeepSeek checkpoint's kernels are a pinned vendor copy of the
# local-inference-lab b12x tree plus the author's ds41_trellismx driver. They
# import by their top-level package names, so expose the vendor directory
# before any recipe imports them.
_VENDOR = Path(__file__).resolve().parent.parent / "vendor"
if _VENDOR.is_dir() and str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

__version__ = "0.1.0"

__all__ = ["__version__"]
