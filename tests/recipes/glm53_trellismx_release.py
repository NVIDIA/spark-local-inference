# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Where the GLM-5.3 Flash TrellisMX release tests find the published release.

`FLARE_TRELLISMX_RELEASE` names the release root. Without it the tests look for the
recipe's pinned revision in the default Hugging Face hub cache. Only an unavailable
release skips: once the variable names a root, every release test runs, so a wrong
path, a missing carrier or a missing b12x interface fails instead of skipping.
"""

import os
from pathlib import Path

import pytest

VARIABLE = "FLARE_TRELLISMX_RELEASE"


def _release() -> Path:
    if VARIABLE in os.environ:
        return Path(os.environ[VARIABLE]).expanduser()
    from ignition.recipes import resolve

    hub_cache = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
    return resolve("glm-5.3-flash-trellismx").snapshot(hub_cache)


RELEASE = _release()
CARRIER = RELEASE / "carrier"
requires_release = pytest.mark.skipif(
    VARIABLE not in os.environ and not RELEASE.is_dir(),
    reason=f"the GLM-5.3 Flash TrellisMX release is not available; set {VARIABLE} to its root")
requires_cuda = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_CUDA_TESTS=1 and an explicitly assigned CUDA device")
