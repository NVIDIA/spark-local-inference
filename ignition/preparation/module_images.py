# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Preparation-scoped native CUPTI module and allocation recording."""

from contextvars import ContextVar


_CURRENT = ContextVar("flare_module_images", default=None)


def cupti_library(library=None):
    if library is None:
        from cuda.pathfinder import load_nvidia_dynamic_lib

        library = load_nvidia_dynamic_lib("cupti").abs_path
    return library


def strip_debug(image):
    from flare_heliopause import cubin_strip_debug

    return cubin_strip_debug(image)


def strip_global_source(symbol, data):
    from flare_heliopause import cubin_strip_global_source

    return cubin_strip_global_source(symbol, data)


def global_symbols(image):
    from flare_heliopause import cubin_global_symbols

    return cubin_global_symbols(image)


class ModuleImages:
    """Keep native module images and driver allocation records through export."""

    def __init__(self, library=None):
        from flare_heliopause import NativeModuleRecorder

        self.native = NativeModuleRecorder(str(cupti_library(library)))
        self._active = False
        self._token = None

    @classmethod
    def current(cls):
        return _CURRENT.get()

    @property
    def allocations(self):
        return self.native.allocations()

    @property
    def allocation_contexts(self):
        return self.native.allocation_contexts()

    def __enter__(self):
        self.native.start()
        self._token = _CURRENT.set(self)
        self._active = True
        return self

    def __exit__(self, kind, value, traceback):
        try:
            self.native.stop()
        finally:
            if self._active:
                _CURRENT.reset(self._token)
                self._token = None
                self._active = False
