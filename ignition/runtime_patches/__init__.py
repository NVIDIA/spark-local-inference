# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Process-local compiler fixes for model preparation."""

import __future__
import ast
import hashlib
import inspect
import linecache
import sys
import textwrap
from functools import cache, update_wrapper
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import RLock
from types import CodeType, FunctionType

from ignition.errors import PreparationError

_TORCH_VERSION = "2.14.1+cu132"
_TRITON_VERSION = "3.8.0"
_LOCK = RLock()
_APPLIED = False
_TRITON_APPLIED = False
_MISSING = object()


@cache
def identity() -> str:
    """Identify the patch implementation in persistent compiler cache keys."""
    key = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        key.update(path.name.encode())
        key.update(path.read_bytes())
    return "ignition-runtime-patches-" + key.hexdigest()


def _replace(function, replacements, *, allow_applied=False):
    """Compile checked edits with the original function's globals and defaults."""
    name = f"{function.__module__}.{function.__qualname__}"
    if function.__closure__:
        raise PreparationError(f"runtime patch cannot replace a closure: {name}")
    source = textwrap.dedent(inspect.getsource(function))
    for before, after in replacements:
        counts = source.count(before), source.count(after) if after else 0
        if allow_applied and counts == (0, 1):
            continue
        if (allow_applied and counts != (1, 0)) or (not allow_applied and counts[0] != 1):
            raise PreparationError(f"runtime patch context is missing or ambiguous in {name}")
        source = source.replace(before, after, 1)
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].decorator_list:
        raise PreparationError(f"runtime patch requires an undecorated function: {name}")
    filename = f"<ignition.runtime_patches:{name}>"
    flags = 0
    for feature in __future__.all_feature_names:
        flags |= getattr(__future__, feature).compiler_flag & function.__code__.co_flags
    module = compile(tree, filename, "exec", flags=flags, dont_inherit=True)
    code, = (constant for constant in module.co_consts if isinstance(constant, CodeType))
    replacement = FunctionType(code, function.__globals__, function.__name__, function.__defaults__)
    replacement.__kwdefaults__ = function.__kwdefaults__
    update_wrapper(replacement, function)
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    return replacement


def apply_triton() -> None:
    """Fix helper compilation before importing CUDA-aware preparation libraries."""
    global _TRITON_APPLIED

    with _LOCK:
        if _TRITON_APPLIED:
            return
        try:
            installed = version("triton")
        except PackageNotFoundError as error:
            if sys.platform == "win32":
                return
            raise PreparationError("Ignition runtime patches require the preparation dependencies") from error
        if installed != _TRITON_VERSION:
            raise PreparationError(f"Ignition runtime patches require triton=={_TRITON_VERSION}; found {installed}")
        from . import triton

        for owner, name, replacement in triton.replacements():
            setattr(owner, name, replacement)
        _TRITON_APPLIED = True


def apply() -> None:
    """Install all fixes once before tracing, compiling, or loading compiled forwards."""
    global _APPLIED

    apply_triton()
    import torch
    import torch.compiler.config

    with _LOCK:
        tag = identity()
        prior = torch.compiler.config.cache_key_tag
        cache_tag = prior if prior.endswith(tag) else (f"{prior};{tag}" if prior else tag)
        if not _APPLIED:
            if torch.__version__ != _TORCH_VERSION:
                raise PreparationError(f"Ignition runtime patches require torch=={_TORCH_VERSION}; "
                                       f"found {torch.__version__}")
            from . import dynamo, functionalization, memory_planning, scheduler, static_triton, workers

            replacements = (*dynamo.replacements(), *functionalization.replacements(), *scheduler.replacements(),
                            *memory_planning.replacements(), *static_triton.replacements(), *workers.replacements())
            originals = [(owner, name, vars(owner).get(name, _MISSING)) for owner, name, _ in replacements]
            # Build and check every replacement before publishing any of them.
            for owner, name, replacement in replacements:
                if set_name := getattr(type(replacement), "__set_name__", None):
                    set_name(replacement, owner, name)
            try:
                for owner, name, replacement in replacements:
                    setattr(owner, name, replacement)
                torch.compiler.config.cache_key_tag = cache_tag
            except Exception:
                for owner, name, original in originals:
                    if original is _MISSING:
                        if name in vars(owner):
                            delattr(owner, name)
                    else:
                        setattr(owner, name, original)
                raise
            _APPLIED = True
        else:
            torch.compiler.config.cache_key_tag = cache_tag
