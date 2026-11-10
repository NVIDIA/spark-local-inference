# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Native CUPTI recorder loading and actual driver allocation lifetimes."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from ignition.preparation.module_images import ModuleImages


@pytest.mark.parametrize("present", [False, True])
def test_native_recorder_rejects_an_unloadable_library(tmp_path, present):
    path = tmp_path / "invalid-cupti.so"
    if present:
        path.write_bytes(b"not a shared library")
    with pytest.raises((OSError, RuntimeError), match="invalid-cupti"):
        ModuleImages(path)
    assert ModuleImages.current() is None


def test_recorder_scope_restores_nested_context_and_stop_failure(monkeypatch):
    from contextvars import Context
    from types import SimpleNamespace

    class Recorder:
        fail_stop = False
        def __init__(self, library):
            pass
        def start(self):
            pass
        def stop(self):
            if self.fail_stop:
                raise RuntimeError("recorder stop failed")

    monkeypatch.setitem(sys.modules, "flare_heliopause", SimpleNamespace(NativeModuleRecorder=Recorder))
    outer, inner = ModuleImages("cupti"), ModuleImages("cupti")
    assert ModuleImages.current() is None
    with outer:
        assert ModuleImages.current() is outer
        assert Context().run(ModuleImages.current) is None
        with inner:
            assert ModuleImages.current() is inner
        assert ModuleImages.current() is outer
        inner.native.fail_stop = True
        with pytest.raises(RuntimeError, match="stop failed"), inner:
            assert ModuleImages.current() is inner
        assert ModuleImages.current() is outer
    assert ModuleImages.current() is None


def test_recorder_start_failure_does_not_replace_active_scope(monkeypatch):
    from types import SimpleNamespace

    class Recorder:
        fail_start = False
        def __init__(self, library):
            pass
        def start(self):
            if self.fail_start:
                raise RuntimeError("recorder start failed")
        def stop(self):
            pass

    monkeypatch.setitem(sys.modules, "flare_heliopause", SimpleNamespace(NativeModuleRecorder=Recorder))
    outer, inner = ModuleImages("cupti"), ModuleImages("cupti")
    inner.native.fail_start = True
    with outer:
        with pytest.raises(RuntimeError, match="start failed"):
            inner.__enter__()
        assert ModuleImages.current() is outer
    assert ModuleImages.current() is None


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires explicit CUDA test opt-in")
def test_native_recorder_tracks_real_driver_allocation_and_context_lifetimes():
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign CUDA_VISIBLE_DEVICES before running GPU tests"
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, "-c", r'''
import ctypes as C
from ignition.preparation.module_images import ModuleImages
from ignition.preparation.runtime_core import _Cuda

cuda = _Cuda()
cuda.call("cuInit", C.c_uint(0))
context, device = C.c_void_p(), C.c_int()
cuda.call("cuDeviceGet", C.byref(device), C.c_int(0))
with ModuleImages() as images:
    assert ModuleImages.current() is images
    cuda.call("cuCtxCreate_v2", C.byref(context), C.c_uint(0), device)
    try:
        pointer = C.c_uint64()
        cuda.call("cuMemAlloc_v2", C.byref(pointer), C.c_size_t(4096))
        assert images.allocations[pointer.value] == (4096, "device")
        assert images.allocation_contexts[pointer.value] == context.value
        cuda.call("cuMemFree_v2", pointer)
        assert pointer.value not in images.allocations
        cuda.call("cuMemAlloc_v2", C.byref(pointer), C.c_size_t(8192))
        assert pointer.value in images.allocations
    finally:
        cuda.call("cuCtxDestroy_v2", context)
    assert pointer.value not in images.allocations
assert ModuleImages.current() is None
'''], cwd=root, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
