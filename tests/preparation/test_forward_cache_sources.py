# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Forward caches track execution sources independently of terminal presentation."""

from types import SimpleNamespace

import flashinfer

from ignition.preparation import aot


def test_presentation_edits_preserve_forward_cache_but_execution_edits_invalidate(tmp_path, monkeypatch):
    core = tmp_path / "ignition"
    preparation = core / "preparation"
    preparation.mkdir(parents=True)
    backend = preparation / "aot.py"
    backend.write_text("backend = 1\n")
    progress = core / "progress.py"
    progress.write_text("splash = 'sun'\n")
    entrypoint = core / "__main__.py"
    entrypoint.write_text("port = 8000\n")
    publication = preparation / "core_file.py"
    publication.write_text("threads = 1\n")
    kernels = tmp_path / "flashinfer"
    kernels.mkdir()
    operator = kernels / "operator.py"
    operator.write_text("kernel = 1\n")
    monkeypatch.setattr(aot, "__file__", str(backend))
    monkeypatch.setattr(flashinfer, "__file__", str(kernels / "__init__.py"))
    monkeypatch.setattr(aot.torch.cuda, "get_device_properties", lambda device:
        SimpleNamespace(name="SM120", major=12, minor=0, multi_processor_count=96))

    def root():
        return aot.ForwardCache(namespace={"model": "example"}, device="cuda:0").root

    original = root()
    progress.write_text("splash = 'solar flare'\n")
    entrypoint.write_text("port = 8001\n")
    publication.write_text("threads = 16\n")
    assert root() == original
    backend.write_text("backend = 2\n")
    changed = root()
    assert changed != original
    operator.write_text("kernel = 2\n")
    assert root() != changed
