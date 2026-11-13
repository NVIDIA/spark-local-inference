# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Preserve Triton's cooperative launch ABI in static Inductor launchers."""

from functools import cached_property

from . import _replace


def _cooperative_launcher(self):
    from triton.backends.driver import expand_signature
    from triton.backends.nvidia.driver import (
        annotate_arguments,
        make_kernel_signature,
        wrap_handle_tensordesc,
    )
    from triton.runtime import driver

    launch = driver.active.utils.launch
    signature = dict(enumerate(self.runtime_signature))
    expanded = expand_signature(signature.values(), self.tensordesc_meta, "nvTmaDesc")
    return (
        wrap_handle_tensordesc(launch, signature, self.tensordesc_meta),
        annotate_arguments(expanded),
        make_kernel_signature(expanded),
    )


def _run_cooperative(self, function, grid_x, grid_y, grid_z, stream, args):
    if len(args) != len(self.runtime_signature):
        raise AssertionError(f"Expected {len(self.runtime_signature)} args, got {len(args)}")
    launch, annotations, signature = self._cooperative_launcher
    # Triton's launcher appends both null scratch arguments.
    launch(
        grid_x, grid_y, grid_z, stream, function, True, False,
        (self.num_warps, 1, self.shared), None, None, None, None, None,
        annotations, signature, args,
    )


def _cuda_getstate(self):
    from torch._inductor.runtime.static_triton_launcher import StaticallyLaunchedTritonKernel

    state = StaticallyLaunchedTritonKernel.__getstate__(self)
    state.pop("_cooperative_launcher", None)
    return state


def replacements():
    from torch._inductor.runtime.static_triton_launcher import (
        StaticallyLaunchedCudaKernel,
        StaticallyLaunchedTritonKernel,
    )
    from torch._inductor.runtime.triton_heuristics import (
        CachingAutotuner,
        StaticTritonCompileResult,
    )

    initialize = _replace(StaticallyLaunchedTritonKernel.__init__, (
        (
            '    self.tensordesc_meta = getattr(kernel.metadata, "tensordesc_meta", None)\n'
            '    self._has_tensordesc = False\n',
            '    self.tensordesc_meta = getattr(kernel.metadata, "tensordesc_meta", None)\n'
            '    self.launch_cooperative_grid = getattr(kernel.metadata, "launch_cooperative_grid", False)\n'
            '    self._has_tensordesc = False\n',
        ),
    ), allow_applied=True)
    argument_types = _replace(StaticallyLaunchedTritonKernel.arg_ty_from_signature, (
        (
            '    self.full_constexprs = constants\n'
            '    # Despite requiring them to be passed in, the triton CUDA launcher\n',
            '    self.full_constexprs = constants\n'
            '    self.runtime_signature = tuple(\n'
            '        signature[i] for i in sorted(signature)\n'
            '        if signature[i] != "constexpr" and i not in constants\n'
            '    )\n'
            '    # Despite requiring them to be passed in, the triton CUDA launcher\n',
        ),
    ), allow_applied=True)
    run = _replace(StaticallyLaunchedTritonKernel.run, (
        (
            '    # Get rid of constants before passing to cubin launcher\n'
            '\n'
            '    if self._has_tensordesc:\n',
            '    # Get rid of constants before passing to cubin launcher\n'
            '\n'
            '    if getattr(self, "launch_cooperative_grid", False):\n'
            '        self._run_cooperative(function, grid_x, grid_y, grid_z, stream, args)\n'
            '        return\n'
            '\n'
            '    if self._has_tensordesc:\n',
        ),
    ), allow_applied=True)
    fast_launcher = _replace(CachingAutotuner._build_fast_launcher, (
        (
            '        kernel = runner.__self__\n'
            '        # compile-on-one-rank kernels keep their loaded handles per device, so\n',
            '        kernel = runner.__self__\n'
            '        # The fast launch ABI cannot retain cooperative launch attributes.\n'
            '        if getattr(kernel, "launch_cooperative_grid", False):\n'
            '            return None\n'
            '        # compile-on-one-rank kernels keep their loaded handles per device, so\n',
        ),
    ), allow_applied=True)
    can_launch = _replace(StaticTritonCompileResult.can_statically_launch, (
        ('@staticmethod\n', ''),
        (
            '        if getattr(kernel.metadata, "launch_pdl", False) or getattr(\n'
            '            kernel.metadata, "launch_cooperative_grid", False\n'
            '        ):\n'
            '            raise CannotStaticallyLaunchKernel(\n'
            '                "static launch does not support launch attributes"\n'
            '            )\n',
            '        if getattr(kernel.metadata, "launch_pdl", False):\n'
            '            raise CannotStaticallyLaunchKernel(\n'
            '                "static launch does not support launch_pdl"\n'
            '            )\n'
            '        if getattr(kernel.metadata, "launch_cooperative_grid", False) and (\n'
            '            triton_meta.get("device_type") != "cuda" or torch.version.hip\n'
            '        ):\n'
            '            raise CannotStaticallyLaunchKernel(\n'
            '                "static cooperative launch requires CUDA"\n'
            '            )\n',
        ),
        (
            'f"{kernel.src.fn.__name__}{binary_ext}"',
            'f"{kernel.src.fn.__name__[:150]}{binary_ext}"',
        ),
    ), allow_applied=True)
    reload_path = _replace(StaticTritonCompileResult.reload_cubin_path, (
        (
            'f"{self.kernel.name}{binary_ext}"',
            'f"{self.kernel.name[:150]}{binary_ext}"',
        ),
    ), allow_applied=True)
    return [
        (StaticallyLaunchedTritonKernel, '__init__', initialize),
        (StaticallyLaunchedTritonKernel, 'arg_ty_from_signature', argument_types),
        (StaticallyLaunchedTritonKernel, 'run', run),
        (StaticallyLaunchedCudaKernel, '_cooperative_launcher', cached_property(_cooperative_launcher)),
        (StaticallyLaunchedCudaKernel, '_run_cooperative', _run_cooperative),
        (StaticallyLaunchedCudaKernel, '__getstate__', _cuda_getstate),
        (CachingAutotuner, '_build_fast_launcher', fast_launcher),
        (StaticTritonCompileResult, 'can_statically_launch', staticmethod(can_launch)),
        (StaticTritonCompileResult, 'reload_cubin_path', reload_path),
    ]
