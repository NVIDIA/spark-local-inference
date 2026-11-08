# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Triton helper cache identity requires no executable probing."""

import hashlib
import platform
from pathlib import Path
import struct
import subprocess
from types import ModuleType

import pytest
from triton.runtime import build


from ignition.runtime_patches import triton as patch


def _module(clean):
    source = Path(build.__file__).read_text()
    if clean:
        source = source.replace(
            '    from platform import machine, system\n'
            '    from struct import calcsize\n'
            '    return ",".join([machine(), system(), f"{calcsize(\'P\') * 8}bit"])',
            '    from platform import machine, system, architecture\n'
            '    return ",".join([machine(), system(), *architecture()])',
        )
    module = ModuleType("triton.runtime._test_build")
    module.__package__ = "triton.runtime"
    exec(compile(source, build.__file__, "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("clean", [False, True])
@pytest.mark.parametrize("width", [4, 8])
def test_host_identity_is_cached_without_subprocess(monkeypatch, clean, width):
    module = _module(clean)
    patch.platform_key.cache_clear()
    module.platform_key = patch.platform_key
    calls = []
    monkeypatch.setattr(platform, "machine", lambda: calls.append("machine") or "aarch64")
    monkeypatch.setattr(platform, "system", lambda: calls.append("system") or "Linux")
    calcsize = struct.calcsize
    monkeypatch.setattr(struct, "calcsize", lambda value: width if value == "P" else calcsize(value))

    def forbidden(*args, **kwargs):
        pytest.fail("platform key invoked a subprocess or platform.architecture")

    monkeypatch.setattr(platform, "architecture", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "check_output", forbidden)
    assert module.platform_key() == f"aarch64,Linux,{width * 8}bit"
    assert module.platform_key() == f"aarch64,Linux,{width * 8}bit"
    assert calls == ["machine", "system"]
    assert module.platform_key.cache_info().hits == 1
    assert module.platform_key.cache_info().misses == 1
    patch.platform_key.cache_clear()


@pytest.mark.parametrize("clean", [False, True])
def test_helper_cache_uses_pointer_width_identity(monkeypatch, clean):
    module = _module(clean)
    patch.platform_key.cache_clear()
    module.platform_key = patch.platform_key
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    calcsize = struct.calcsize
    monkeypatch.setattr(struct, "calcsize", lambda value: 8 if value == "P" else calcsize(value))
    monkeypatch.setattr(module, "get_cache_manager", lambda key: key)
    config = {"language": ["c++"], "library_dirs": None, "libraries": ["cuda"], "ccflags": ["-O3"]}
    expected = hashlib.sha256()
    expected.update(b"int helper() { return 0; }")
    expected.update(b"x86_64,Linux,64bit")
    for key, values in config.items():
        if values is None:
            continue
        expected.update(key.encode())
        for value in values:
            expected.update(value.encode())
            expected.update(b":")
    assert module._get_cache_manager(b"int helper() { return 0; }", config) == expected.hexdigest()
    assert module._get_cache_manager(b"int helper() { return 0; }", config) == expected.hexdigest()
    assert patch.platform_key.cache_info().hits == 1
    patch.platform_key.cache_clear()


def test_replacements_points_to_live_helper_module():
    assert patch.replacements() == ((build, "platform_key", patch.platform_key),)


def test_replaces_clean_architecture_probe_and_preserves_patched_key(monkeypatch):
    clean = _module(True)
    patched = _module(False)
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    calcsize = struct.calcsize
    monkeypatch.setattr(struct, "calcsize", lambda value: 8 if value == "P" else calcsize(value))
    monkeypatch.setattr(platform, "architecture", lambda: ("64bit", "ELF"))
    assert clean.platform_key() == "x86_64,Linux,64bit,ELF"
    expected = patched.platform_key()
    patch.platform_key.cache_clear()
    clean.platform_key = patch.platform_key
    assert clean.platform_key() == expected == "x86_64,Linux,64bit"
    patch.platform_key.cache_clear()
