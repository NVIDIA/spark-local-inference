# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Process environment that preparation libraries read, set from Flare's configuration.

PyTorch, b12x, Triton, and the CuTe DSL read these variables when they are
imported or first used. `apply` sets the environment before installing runtime
patches. Rank and compiler processes inherit the environment and install their
own patches.
"""

import hashlib
import os
import sys
from importlib import util
from pathlib import Path

from ignition import runtime_patches

_DLL_DIRECTORIES = []


def flashinfer_workspace(kernel_cache: Path) -> Path:
    """Separate JIT build directories for independently editable installations."""
    package = util.find_spec("flashinfer")
    root = kernel_cache / "flashinfer"
    if package is None or package.origin is None:
        return root
    # FlashInfer keys build directories by version and GPU architecture. Distinct
    # checkouts can share both while requiring different sources and compiler flags.
    source = str(Path(package.origin).resolve()).encode()
    return root / hashlib.sha256(source).hexdigest()[:16]


def apply(configuration, gpus, b12x=()) -> None:
    """Select the plan's GPUs as process-local devices 0 through N-1, place every cache and set the recipe's
    b12x settings, which b12x reads only from its environment."""
    kernel_cache = configuration.directory("kernel_cache")
    os.environ.update({
        # Preserve explicitly selected dependency roots in Inductor subprocesses.
        "TORCH_CUSTOM_PYTHONPATH": os.pathsep.join(sys.path),
        "CUDA_VISIBLE_DEVICES": ",".join(gpu.uuid for gpu in gpus),
        "HF_HUB_CACHE": str(configuration.directory("hub_cache")),
        "B12X_COMPILE_CACHE_DIR": str(kernel_cache / "b12x"),
        "FLASHINFER_WORKSPACE_BASE": str(flashinfer_workspace(kernel_cache)),
        "TRITON_CACHE_DIR": str(kernel_cache / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(kernel_cache / "inductor"),
        "TORCH_EXTENSIONS_DIR": str(kernel_cache / "torch-extensions"),
        **dict(b12x),
    })
    # cuTile discovers its compiler bundle independently of the native toolkit.
    cuda = configuration.directory("cuda")
    if cuda is not None:
        os.environ["CUDA_HOME"] = str(cuda)
        if os.name == "nt":
            os.environ["CUDA_PATH"] = str(cuda)
            os.environ["TRITON_PTXAS.EXE_PATH"] = str(cuda / "bin/ptxas.exe")
            for path in (cuda / "bin", cuda / "bin/x64", cuda / "lib/x64"):
                if path.is_dir():
                    _DLL_DIRECTORIES.append(os.add_dll_directory(str(path)))
    runtime_patches.apply()
