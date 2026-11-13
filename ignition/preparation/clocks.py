# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Temporary graphics-clock locks for kernel tuning."""

from contextlib import contextmanager
import os
import shutil
import subprocess

from ignition.errors import PreparationError


def _set_clocks(uuid, option):
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise PreparationError("--lock-gpu-clocks requires nvidia-smi")
    command = [executable, "--id", uuid, option]
    if os.geteuid() != 0:
        command = ["sudo", "-n", *command]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PreparationError(f"GPU clock control failed for {uuid}: {error}") from error
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise PreparationError(
            f"GPU clock control failed for {uuid}: {detail}. "
            "Clock control requires root or noninteractive sudo access to nvidia-smi."
        )


@contextmanager
def locked_gpu_clocks(uuid):
    """Lock the selected GPU at its maximum graphics clock, then reset driver defaults."""
    from cuda.bindings import nvml

    if not uuid:
        raise PreparationError("GPU clock locking requires the preparation device's UUID")
    try:
        nvml.init_v2()
        try:
            handle = nvml.device_get_handle_by_uuid(uuid)
            uuid = nvml.device_get_uuid(handle)
            maximum = nvml.device_get_max_clock_info(handle, nvml.ClockType.CLOCK_GRAPHICS)
        finally:
            nvml.shutdown()
    except Exception as error:
        raise PreparationError(f"Cannot query supported GPU clocks for {uuid}: {error}") from error
    _set_clocks(uuid, f"--lock-gpu-clocks={maximum},{maximum}")
    failure = None
    try:
        yield
    except BaseException as error:
        failure = error
        raise
    finally:
        try:
            _set_clocks(uuid, "--reset-gpu-clocks")
        except BaseException as error:
            if failure is not None:
                raise BaseExceptionGroup("GPU tuning and clock reset failed", [failure, error]) from None
            raise
