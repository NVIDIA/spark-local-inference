# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint reads preserve tensor bytes and other devices' CUDA contexts."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ignition.components import weights


def _read_checkpoint_in_fresh_process(directory, existing_context):
    import ctypes
    import json

    import torch
    from safetensors.torch import save_file
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint

    driver = (ctypes.WinDLL("nvcuda.dll", winmode=0x800) if sys.platform == "win32"
              else ctypes.CDLL("libcuda.so.1"))

    def call(name, *arguments):
        status = getattr(driver, name)(*arguments)
        assert status == 0, (name, status)

    call("cuInit", 0)
    count = ctypes.c_int()
    call("cuDeviceGetCount", ctypes.byref(count))
    assert count.value >= (2 if existing_context else 1)

    def states():
        result = []
        for ordinal in range(count.value):
            flags, active = ctypes.c_uint(), ctypes.c_int()
            call("cuDevicePrimaryCtxGetState", ordinal, ctypes.byref(flags), ctypes.byref(active))
            result.append((flags.value, active.value))
        return result

    initial = states()
    assert all(active == 0 for _, active in initial), initial
    context, pointer = ctypes.c_void_p(), ctypes.c_uint64()
    payload = (ctypes.c_uint32 * 16)(*(0x9E3779B9 ^ value for value in range(16)))
    if existing_context:
        call("cuDevicePrimaryCtxRetain", ctypes.byref(context), 1)
        call("cuCtxPushCurrent_v2", context)
        call("cuMemAlloc_v2", ctypes.byref(pointer), ctypes.c_size_t(ctypes.sizeof(payload)))
        call("cuMemcpyHtoD_v2", pointer, ctypes.byref(payload), ctypes.c_size_t(ctypes.sizeof(payload)))
        popped = ctypes.c_void_p()
        call("cuCtxPopCurrent_v2", ctypes.byref(popped))
        assert popped.value == context.value
    before_read = states()
    expected_devices = {0, 1} if existing_context else {0}
    try:
        root = Path(directory)
        source = {}
        for dtype in (torch.bfloat16, torch.uint8, torch.float32):
            name = str(dtype).removeprefix("torch.")
            source[name] = torch.arange(17 * 65, dtype=torch.float32).reshape(17, 65).to(dtype)
            save_file({name: source[name]}, str(root / (name + ".safetensors")))
        (root / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": sum(tensor.nbytes for tensor in source.values())},
            "weight_map": {name: name + ".safetensors" for name in source}}))
        manifest = inspect_indexed_checkpoint(root)
        torch.cuda.set_device(0)
        actual = {}
        with weights.DirectWeightLoader(manifest, "cuda:0") as loader:
            for name in source:
                actual[name] = loader.load_tensor(name, extent=(slice(2, 13), slice(3, 62)))
            assert loader._session.read_mode == "bounce"
            assert loader._session.torch_copy_bytes == 0
            assert {device for device, (_, active) in enumerate(states()) if active} == expected_devices
        after = states()
        assert {device for device, (_, active) in enumerate(after) if active} == expected_devices
        for ordinal in range(count.value):
            if ordinal != 0:
                assert after[ordinal] == before_read[ordinal], (ordinal, before_read, after)
        for name, tensor in actual.items():
            torch.testing.assert_close(tensor.cpu(), source[name][2:13, 3:62], rtol=0, atol=0)
        if existing_context:
            assert torch.cuda.memory_reserved(1) == 0
            call("cuCtxPushCurrent_v2", context)
            restored = type(payload)()
            call("cuMemcpyDtoH_v2", ctypes.byref(restored), pointer, ctypes.c_size_t(ctypes.sizeof(restored)))
            call("cuCtxPopCurrent_v2", ctypes.byref(ctypes.c_void_p()))
            assert list(restored) == list(payload)
        print(f"visible_devices={count.value} active_devices={sorted(expected_devices)} preserved_tensor_dtypes={list(source)}")
    finally:
        if existing_context:
            call("cuCtxPushCurrent_v2", context)
            call("cuMemFree_v2", pointer)
            call("cuCtxPopCurrent_v2", ctypes.byref(ctypes.c_void_p()))
            call("cuDevicePrimaryCtxRelease_v2", 1)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned CUDA GPU")
@pytest.mark.parametrize("existing_context", [False, True])
def test_indexed_checkpoint_reads_preserve_device_contexts(tmp_path, existing_context):
    import torch

    if existing_context and os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1":
        pytest.skip("requires FLARE_RUN_TP_CUDA_TESTS=1 and two assigned visible CUDA devices")
    if existing_context and torch.cuda.device_count() < 2:
        pytest.skip("requires two assigned visible CUDA devices")
    command = "import runpy, sys; runpy.run_path(sys.argv[1])['_read_checkpoint_in_fresh_process'](sys.argv[2], sys.argv[3] == '1')"
    result = subprocess.run([sys.executable, "-c", command, str(Path(__file__).resolve()), str(tmp_path),
                             "1" if existing_context else "0"], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout, end="")


@pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1', reason='requires an assigned CUDA GPU')
@pytest.mark.parametrize('dtype', ('bfloat16', 'uint8', 'float32'))
def test_tmpfs_checkpoint_preserves_strided_tensor_bytes(dtype):
    import tempfile
    from types import MappingProxyType

    import torch
    from safetensors.torch import save_file

    from ignition.preparation.checkpoint import IndexedCheckpoint, _read_header

    if not Path('/dev/shm').is_dir():
        pytest.skip('requires Linux shared memory')
    with tempfile.TemporaryDirectory(prefix='flare-checkpoint-', dir='/dev/shm') as directory:
        root = Path(directory)
        source = torch.arange(17 * 65, dtype=torch.float32).reshape(17, 65).to(getattr(torch, dtype))
        save_file({'weight': source}, str(root / 'weights.safetensors'))
        shard, tensors = _read_header(root / 'weights.safetensors', 'weights.safetensors')
        manifest = IndexedCheckpoint(root, MappingProxyType(tensors), (shard,))
        with weights.DirectWeightLoader(manifest, 'cuda:0') as loader:
            actual = loader.load_tensor('weight', extent=(slice(2, 13), slice(3, 62)))
            assert loader._session.read_mode == 'bounce'
            assert loader._session.torch_copy_bytes == 0
        torch.testing.assert_close(actual.cpu(), source[2:13, 3:62], rtol=0, atol=0)
