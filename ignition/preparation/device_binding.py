# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Bind single-device compiled forwards to a tensor-parallel rank's CUDA ordinal."""

import ast
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace

import torch

_DEVICE = ContextVar("flare_compiled_device", default=None)
_FREEZE_TUNING = ContextVar("flare_compiled_tuning_winners", default=False)


@contextmanager
def bind_device(device, *, freeze_tuning=False):
    if type(freeze_tuning) is not bool:
        raise TypeError("freeze_tuning must be a boolean")
    token = _DEVICE.set(torch.device(device))
    tuning_token = _FREEZE_TUNING.set(freeze_tuning)
    try:
        yield
    finally:
        _FREEZE_TUNING.reset(tuning_token)
        _DEVICE.reset(token)


def restore_device(kind, index):
    target = _DEVICE.get()
    return target if kind == "cuda" and target is not None else torch.device(kind, index)


def restore_prepared_device(kind, state):
    value = object.__new__(kind)
    target = _DEVICE.get()
    if target is not None:
        state["ordinal"] = target.index
        if isinstance(state.get("uuid"), str):
            state["uuid"] = str(torch.cuda.get_device_properties(target).uuid).strip()
    value.__dict__.update(state)
    return value


def _device_source(source, original, target, kernel_keys=None):
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    offsets, offset = [], 0
    for line in lines:
        offsets.append(offset)
        offset += len(line.encode())
    replacements = []

    def replace(node, value):
        start = offsets[node.lineno - 1] + node.col_offset
        end = offsets[node.end_lineno - 1] + node.end_col_offset
        replacements.append((start, end, value.encode()))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func)
        if name in ("torch.cuda.set_device", "torch.cuda._DeviceGuard", "get_raw_stream"):
            if len(node.args) != 1 or not isinstance(node.args[0], ast.Constant) or node.args[0].value != original:
                raise ValueError(f"unsupported compiled CUDA device binding: {ast.unparse(node)}")
            replace(node.args[0], str(target))
        elif name == "DeviceProperties":
            for keyword in node.keywords:
                if keyword.arg == "index":
                    if not isinstance(keyword.value, ast.Constant) or keyword.value.value != original:
                        raise ValueError("compiled kernel spans multiple CUDA devices")
                    replace(keyword.value, str(target))
        elif name == "async_compile.triton":
            from torch._inductor.async_compile import CompiledTritonKernels

            kernel = node.args[1]
            rebound = _device_source(kernel.value, original, target)
            if kernel_keys is not None:
                kernel_keys[CompiledTritonKernels.key(kernel.value)] = CompiledTritonKernels.key(rebound)
            replace(kernel, repr(rebound))
    data = source.encode()
    parts, offset = [], 0
    for start, end, value in sorted(replacements):
        parts.extend((data[offset:start], value))
        offset = end
    parts.append(data[offset:])
    return b"".join(parts).decode()


def _freeze_tuning_winners(bundle):
    """Preserve completed compile-time selections while rebuilding launchers.

    Torch rechecks a serialized static winner as a one-config cache entry and
    can reopen occupancy search when it rebuilds launchers. Prepared forwards
    retain the selected config; all search and optional refinement must finish
    before serialization.
    """
    # Torch loads duplicate cache keys in list order. Only the final snapshot
    # for each key is effective; preceding snapshots can predate autotuning.
    entries = {entry.cache_key: entry for entry in bundle.static_autotuners}
    for entry in entries.values():
        kernel = entry.kernel
        if len(kernel.compile_results) != 1 or not kernel.is_statically_launchable():
            raise ValueError(f"compiled kernel {entry.kernel_name} has unresolved tuning candidates")
        result, = kernel.compile_results
        config = result.config
        if hasattr(result, '_flare_config_state'):
            config.__dict__.clear()
            config.__dict__.update(result._flare_config_state)
        metadata = kernel.inductor_meta
        if (metadata.get("coordinate_descent_tuning", False)
                and kernel._should_coordesc_tune
                and not getattr(config, "found_by_coordesc", False)):
            raise ValueError(f"compiled kernel {entry.kernel_name} has pending coordinate-descent tuning")
        if (metadata.get("combo_tuning_groups")
                and not getattr(config, "found_by_combo_autotune", False)):
            raise ValueError(f"compiled kernel {entry.kernel_name} has pending combo tuning")
        kernel.inductor_meta = {**metadata, "dynamic_scale_rblock": False,
                                "coordinate_descent_tuning": False}
        kernel.__dict__.pop("_could_rblock_scale", None)
    bundle.static_autotuners[:] = entries.values()


def relocate_inductor(entry, original, *, freeze_tuning=False):
    target = _DEVICE.get()
    relocate = target is not None and target != original
    freeze_tuning = freeze_tuning or _FREEZE_TUNING.get()
    if not relocate and not freeze_tuning:
        return
    from torch._inductor.codecache import code_hash
    from torch._inductor.output_code import CompiledFxGraph
    from torch.utils._ordered_set import OrderedSet

    graph = entry.compiled_fw.result
    if entry.compiled_bw is not None or not isinstance(graph, CompiledFxGraph):
        raise ValueError("device binding requires a compiled inference forward")
    if graph.device_idxs != OrderedSet((original.index,)):
        raise ValueError("compiled forward must use one CUDA device")
    if freeze_tuning and graph._triton_bundle is not None:
        _freeze_tuning_winners(graph._triton_bundle)
    if not relocate:
        return
    kernel_keys = {}
    graph.source_code = _device_source(graph.source_code, original.index, target.index, kernel_keys)
    graph.cache_key = code_hash(graph.source_code)
    graph.device_idxs = OrderedSet((target.index,))
    if bundle := graph._triton_bundle:
        autotuners = []
        for entry_kernel in bundle.static_autotuners:
            kernel = entry_kernel.kernel
            kernel.device_props = kernel.device_props._replace(index=target.index)
            kernel.triton_meta["device"] = target.index
            for result in kernel.compile_results:
                result.compile_meta["device"] = target.index
            autotuners.append(replace(entry_kernel, cache_key=kernel_keys[entry_kernel.cache_key]))
        graph._triton_bundle = replace(bundle,
            kernel_artifacts=[replace(item, device=target.index) for item in bundle.kernel_artifacts],
            static_autotuners=autotuners)
