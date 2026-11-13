# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Source compatibility identity for preparation and core rebuild metadata."""

import hashlib
from pathlib import Path


def source_identity():
    """Digest of the Python and native sources every rank must run identically."""
    from importlib import util

    # Locate FlashInfer without importing it: CUDA device selection must precede PyTorch's import.
    flashinfer = util.find_spec("flashinfer")
    if flashinfer is None:
        raise RuntimeError("the preparation environment requires FlashInfer")
    root = Path(__file__).resolve().parents[2]
    return _source_identity(root, Path(flashinfer.origin).resolve().parent)


def _source_identity(flare_root: Path, flashinfer_package: Path) -> str:
    """Hash labeled source trees independently of checkout and install layout.

    FlashInfer's ``data/include`` and ``data/csrc`` contain the native sources
    used by its JIT specs. Editable installs link those directories; wheels
    copy them. Exclude ``data`` from the Python-package tree and include the
    two resolved native trees separately, so each contributes exactly once.
    Dependency data and compilation caches are outside this source set.
    """
    import flare_heliopause

    flare_root, flashinfer_package = flare_root.resolve(), flashinfer_package.resolve()
    data = flashinfer_package / "data"
    native = {name: (data / name).resolve() for name in ("include", "csrc")}
    for name, source in native.items():
        if not source.is_dir():
            raise RuntimeError(f"FlashInfer native source directory is unavailable: {data / name}; "
                               "install the selected FlashInfer package before preparation")
    sources = (
        ("ignition", flare_root / "ignition", ()),
        ("helios", flare_root / "helios" / "src", ()),
        ("heliopause", flare_root / "heliopause" / "src", ()),
        ("flashinfer.python", flashinfer_package, (data,)),
        ("flashinfer.include", native["include"], ()),
        ("flashinfer.csrc", native["csrc"], ()),
    )
    digest = hashlib.sha256()
    for label, source, excluded in sources:
        identity = flare_heliopause.export_source_identity(
            [source], excluded_directories=list(excluded))
        digest.update(label.encode("ascii") + b"\0")
        digest.update(bytes.fromhex(identity))
    return digest.hexdigest()
