# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared Triton launches that remain visible to whole-forward compilation."""

from dataclasses import dataclass

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode


def _launch_options(options):
    """Match the one-Config launch produced by Torch's wrapped Triton call."""
    from triton.runtime.autotuner import Config

    # Ordinary constexpr and backend arguments precede the Config defaults,
    # matching Torch's wrapped launch. maxnreg/ir_override remain ordinary kwargs.
    fields = {"num_warps", "num_stages", "num_ctas"}
    selected = {name: value for name, value in options.items() if name in fields}
    if not selected:
        return dict(options)
    result = {name: value for name, value in options.items() if name not in fields}
    result.update(Config({}, **selected).all_kwargs())
    return result


@dataclass
class PreparedTriton:
    kernel: object
    compiled: object
    args: tuple
    grid: tuple
    options: dict

    def __call__(self):
        if torch.compiler.is_compiling() or get_proxy_mode() is not None:
            torch.library.wrap_triton(self.kernel)[self.grid](*self.args, **self.options)
        else:
            self.compiled[self.grid](*self.args)


def warmup_triton(kernel, args, grid, **options):
    """Prepare the same raw specialization as the later wrapped launch."""
    return kernel.warmup(*args, grid=grid, **_launch_options(options))


def prepare_triton(kernel, args, grid, **options):
    compiled = warmup_triton(kernel, args, grid, **options)
    # Torch's wrapped launch accepts caller options, not raw Config defaults.
    return PreparedTriton(kernel, compiled, args, grid, options)


def launch_triton(kernel, compiled, args, grid, **options):
    if torch.compiler.is_compiling() or get_proxy_mode() is not None:
        torch.library.wrap_triton(kernel)[grid](*args, **options)
    else:
        compiled[grid](*args)
