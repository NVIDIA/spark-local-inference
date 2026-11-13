# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Native discovery and compiled forwards share one CUDA math-library identity."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

import ignition


@pytest.mark.parametrize("session_kind", ["recorded", "standalone"])
def test_discovery_options_survive_inductor_precision_initialization(session_kind):
    # A subprocess preserves the unset-library startup condition independently
    # of other tests that have already initialized Inductor.
    source = r'''
from pathlib import Path
import sys
from unittest.mock import patch

import torch
assert not torch.cuda.is_initialized()
def reject_cuda(*args, **kwargs):
    raise RuntimeError("this compiler-identity check must not initialize CUDA")
torch.cuda._lazy_init = reject_cuda

from torch.utils.cpp_extension import CUDA_HOME
if CUDA_HOME is None:
    raise SystemExit(77)
toolkit_library = Path(CUDA_HOME) / "nvvm/libdevice/libdevice.10.bc"
if not toolkit_library.is_file():
    raise SystemExit(77)

from triton.backends.nvidia.compiler import CUDAOptions
from torch._inductor import config
from torch._inductor.runtime.compile_tasks import _set_triton_libdevice_path
from b12x.preparation import PreparationSession
from ignition.preparation.compilation import _OPTIONS
from ignition.preparation.tuning import RecordedPreparationSession, declare

observed = []
if sys.argv[1] == "recorded":
    # Only the GPU-owning base constructor is replaced. Flare's constructor
    # and the real Inductor/Triton library selection remain under test.
    def base_initialize(self, **kwargs):
        observed.append(CUDAOptions())
    with patch.object(PreparationSession, "__init__", base_initialize):
        RecordedPreparationSession(device=0, checkpoint="compiler-library",
            workload_schema="compiler-library", numerical_policy="default",
            compile_workers=1)
else:
    class Standalone:
        def prepare(self, requests):
            observed.append(CUDAOptions())
    declare(Standalone(), ())

before = observed[0]
with config.patch(emulate_precision_casts=_OPTIONS["emulate_precision_casts"]):
    _set_triton_libdevice_path()
after = CUDAOptions()
assert before.extern_libs == after.extern_libs, (
    "native discovery", before.extern_libs, "compiled forward", after.extern_libs)
assert Path(dict(before.extern_libs)["libdevice"]) == toolkit_library, (
    before.extern_libs, toolkit_library)
assert before.hash() == after.hash(), (before.hash(), after.hash())
assert not torch.cuda.is_initialized()
'''
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    environment.pop("TRITON_LIBDEVICE_PATH", None)
    result = subprocess.run([sys.executable, "-c", source, session_kind],
        cwd=Path(ignition.__file__).resolve().parents[1], env=environment,
        capture_output=True, text=True)
    if result.returncode == 77:
        pytest.skip("requires the CUDA toolkit's libdevice bitcode")
    assert result.returncode == 0, result.stdout + result.stderr
