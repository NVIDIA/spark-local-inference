# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Install compiler fixes in fresh dependency worker processes."""

from torch._inductor.compile_worker.subproc_pool import SubprocPickler
from torch._inductor.compile_worker.utils import _async_compile_initializer

from . import apply, apply_triton


class CompilerPickler(SubprocPickler):
    def __init__(self):
        apply_triton()

    def loads(self, data):
        apply()
        return super().loads(data)


def initialize_inductor_worker(orig_ppid, close_fds=()):
    _async_compile_initializer(orig_ppid, close_fds)
    apply()


def replacements():
    from b12x._lib import compile_pool
    from torch._inductor import async_compile
    from torch._inductor.compile_worker import subproc_pool, utils

    from .triton import initialize_b12x_worker

    return (
        (compile_pool, "_initialize_worker", initialize_b12x_worker),
        (subproc_pool, "SubprocPickler", CompilerPickler),
        (utils, "_async_compile_initializer", initialize_inductor_worker),
        (async_compile, "_async_compile_initializer", initialize_inductor_worker),
        (subproc_pool, "_async_compile_initializer", initialize_inductor_worker),
    )
