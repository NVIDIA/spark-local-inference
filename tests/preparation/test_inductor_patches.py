# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Installation-file patching, dependency checks, and automatic preparation startup."""

import functools
import os
import platform
import re
import shutil
import stat
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
from types import SimpleNamespace

import pytest

from ignition import patches
from ignition.cli import environment
from ignition.cli.config import Configuration
from ignition.errors import PreparationError
from ignition.patches.__main__ import main


@pytest.fixture
def installation(monkeypatch, tmp_path):
    """Install distribution metadata and exact source contexts from bundled diffs."""
    for name in ("torch._inductor.runtime.triton_heuristics", "torch._inductor.runtime.static_triton_launcher",
                 "triton.runtime.build"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    sources, expected, diffs = {}, {}, []
    for name in patches._PATCHES:
        diff = files(patches).joinpath(name).read_text(encoding="utf-8")
        diffs.append(diff)
        relative = None
        for line in diff.splitlines(keepends=True):
            if line.startswith("--- a/"):
                relative = line.removeprefix("--- a/").strip()
            elif line.startswith("@@ "):
                position = int(re.match(r"@@ -(\d+)", line)[1]) - 1
            elif relative is not None and line.startswith((" ", "-")):
                sources.setdefault(relative, {})[position] = line[1:]
                position += 1
    root = tmp_path / "site-packages"
    for relative, lines in sources.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(lines.get(i, f"# source line {i}\n") for i in range(max(lines) + 1)))
        sources[relative] = path.read_bytes()
    for name, version in (("torch", "2.14.1+cu132"), ("triton", "3.8.0")):
        metadata = root / f"{name}-{version}.dist-info" / "METADATA"
        metadata.parent.mkdir()
        metadata.write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    monkeypatch.syspath_prepend(str(root))
    for relative, before, after in patches._hunks("".join(diffs)):
        expected[relative] = expected.get(relative, sources[relative].decode()).replace(before, after, 1)
    return SimpleNamespace(root=root, sources=sources, expected=expected, diffs=diffs)


def assert_patched(installation):
    for relative, expected in installation.expected.items():
        assert (installation.root / relative).read_text() == expected


def test_preparation_startup_applies_patches_without_importing_torch(installation, tmp_path, monkeypatch):
    for module in ("torch", "triton"):
        monkeypatch.delitem(sys.modules, module, raising=False)
    config = tmp_path / "config.toml"
    config.write_text(f'[paths]\nkernel_cache = "{tmp_path}/kernels"\n')
    environment.apply(Configuration.load(config), ())
    assert_patched(installation)
    assert "torch" not in sys.modules and "triton" not in sys.modules


def test_repeated_startup_keeps_files_and_reinstall_reapplies_patches(installation):
    assert len(patches.apply()) == 3
    installed = {relative: (installation.root / relative).stat() for relative in installation.sources}
    assert patches.apply() == ()
    for relative, before in installed.items():
        after = (installation.root / relative).stat()
        assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    for relative, source in installation.sources.items():
        (installation.root / relative).write_bytes(source)
    assert len(patches.apply()) == 3
    assert_patched(installation)


@pytest.mark.parametrize("patch_index", [0, 1, 2])
@pytest.mark.parametrize("partial", [False, True])
def test_existing_manual_patch_is_accepted(installation, patch_index, partial):
    for relative, before, after in patches._hunks(installation.diffs[patch_index]):
        path = installation.root / relative
        path.write_text(path.read_text().replace(before, after, 1))
        if partial:
            break
    patches.apply()
    assert_patched(installation)


def test_file_replacement_preserves_permissions_and_wheel_cache_hardlinks(installation, tmp_path):
    links = tmp_path / "wheel-cache"
    links.mkdir()
    for relative in installation.sources:
        target = installation.root / relative
        target.chmod(0o644)
        os.link(target, links / target.name)
    patches.apply()
    assert_patched(installation)
    for relative, source in installation.sources.items():
        target = installation.root / relative
        assert (links / target.name).read_bytes() == source
        assert stat.S_IMODE(target.stat().st_mode) == 0o644
        assert target.stat().st_ino != (links / target.name).stat().st_ino


@pytest.mark.parametrize("target_platform", ["linux", "win32"])
@pytest.mark.parametrize("name, version", [("torch", "2.14.0+cu132"), ("triton", "3.7.0")])
def test_incompatible_dependency_versions_fail_before_writes(installation, monkeypatch, name, version,
                                                           target_platform):
    monkeypatch.setattr(patches, "sys", SimpleNamespace(platform=target_platform, modules=sys.modules))
    metadata = next(installation.root.glob(f"{name}-*.dist-info/METADATA"))
    metadata.write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    with pytest.raises(PreparationError, match=f"require {name}=="):
        patches.apply()
    for relative, source in installation.sources.items():
        assert (installation.root / relative).read_bytes() == source


@pytest.mark.parametrize("target_platform", ["linux", "win32"])
@pytest.mark.parametrize("name", ["torch", "triton"])
def test_missing_dependency_is_allowed_only_for_triton_on_windows(installation, monkeypatch, name,
                                                                target_platform):
    shutil.rmtree(next(installation.root.glob(f"{name}-*.dist-info")))
    monkeypatch.setattr(patches, "sys", SimpleNamespace(platform=target_platform, modules=sys.modules))
    # Restrict distribution discovery to this installation, excluding the test runner's dependencies.
    monkeypatch.setattr(sys, "path", [str(installation.root)])
    if name == "triton" and target_platform == "win32":
        assert len(patches.apply()) == 2
        assert patches.apply() == ()
        for relative, expected in installation.expected.items():
            if relative.startswith("torch/"):
                assert (installation.root / relative).read_text() == expected
            else:
                assert (installation.root / relative).read_bytes() == installation.sources[relative]
    else:
        with pytest.raises(PreparationError, match="require the preparation dependencies"):
            patches.apply()
        for relative, source in installation.sources.items():
            assert (installation.root / relative).read_bytes() == source


@pytest.mark.parametrize("ambiguous", [False, True])
@pytest.mark.parametrize("relative, marker", [
    ("torch/_inductor/runtime/static_triton_launcher.py", "self.full_constexprs = constants"),
    ("triton/runtime/build.py", "from platform import machine, system, architecture"),
])
def test_incompatible_context_fails_before_any_file_is_written(installation, ambiguous, relative, marker):
    path = installation.root / relative
    source = path.read_text()
    path.write_text(source + source if ambiguous else source.replace(marker, "pass"))
    before = {relative: (installation.root / relative).read_bytes() for relative in installation.sources}
    with pytest.raises(PreparationError, match="context is missing or ambiguous"):
        patches.apply()
    for relative, source in before.items():
        assert (installation.root / relative).read_bytes() == source


@pytest.mark.parametrize("module", ["torch._inductor.runtime.static_triton_launcher", "triton.runtime.build"])
def test_imported_patch_target_requires_process_restart(installation, monkeypatch, module):
    monkeypatch.setitem(sys.modules, module, SimpleNamespace())
    with pytest.raises(PreparationError, match="restart Python"):
        patches.apply()
    for relative, source in installation.sources.items():
        assert (installation.root / relative).read_bytes() == source


def test_triton_target_uses_its_own_distribution(installation, monkeypatch, tmp_path):
    root = tmp_path / "triton-installation"
    target = root / "triton/runtime/build.py"
    target.parent.mkdir(parents=True)
    shutil.move(installation.root / "triton/runtime/build.py", target)
    metadata = next(installation.root.glob("triton-*.dist-info"))
    shutil.move(metadata, root / metadata.name)
    monkeypatch.syspath_prepend(str(root))
    changed = patches.apply()
    assert target in changed
    assert target.read_text() == installation.expected["triton/runtime/build.py"]
    assert not (installation.root / "triton/runtime/build.py").exists()


def test_already_patched_imported_triton_module_is_accepted(installation, monkeypatch):
    target = installation.root / "triton/runtime/build.py"
    target.write_text(installation.expected["triton/runtime/build.py"])
    monkeypatch.setitem(sys.modules, "triton.runtime.build", SimpleNamespace())
    assert len(patches.apply()) == 2
    assert_patched(installation)


@pytest.mark.parametrize("target", ["triton/runtime/cache.py", "triton/build.py", "numpy/runtime/build.py"])
def test_patch_targets_remain_restricted(target):
    diff = f"diff --git a/{target} b/{target}\n@@ -1 +1 @@\n-before\n+after\n"
    with pytest.raises(PreparationError, match="must modify existing"):
        list(patches._hunks(diff))


def test_patched_platform_key_does_not_launch_a_subprocess(installation, monkeypatch):
    def reject_subprocess(*args, **kwargs):
        raise AssertionError("platform_key launched a subprocess")

    monkeypatch.setattr(subprocess, "Popen", reject_subprocess)
    target = installation.root / "triton/runtime/build.py"
    if sys.platform == "linux":
        original = {"functools": functools}
        exec(compile(target.read_text(), str(target), "exec"), original)
        with pytest.raises(AssertionError, match="launched a subprocess"):
            original["platform_key"]()
    patches.apply()
    patched = {"functools": functools}
    exec(compile(target.read_text(), str(target), "exec"), patched)
    assert patched["platform_key"]() == f"{platform.machine()},{platform.system()},{struct.calcsize('P') * 8}bit"
    assert patched["platform_key"]() == patched["platform_key"]()
    assert patched["platform_key"].cache_info().hits == 2


def test_simultaneous_startup_produces_the_same_patched_files(installation):
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: patches.apply(), range(8)))
    assert_patched(installation)
    assert patches.apply() == ()


def test_explicit_entry_point_reports_errors_and_repeated_application(installation, capsys):
    assert main() == 0
    assert capsys.readouterr().out.count("Patched ") == 3
    assert main() == 0
    assert capsys.readouterr().out == ""
    next(installation.root.glob("triton-*.dist-info/METADATA")).write_text("Name: triton\nVersion: 3.7.0\n")
    assert main() == PreparationError.exit_code
    assert "require triton==3.8.0" in capsys.readouterr().err


def test_bundled_patch_results_match_system_patch(installation, tmp_path):
    executable = shutil.which("patch")
    if executable is None:
        pytest.skip("system patch is unavailable for the independent comparison")
    reference = tmp_path / "reference"
    for relative, source in installation.sources.items():
        path = reference / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source)
    for diff in installation.diffs:
        subprocess.run([executable, "-d", str(reference), "-p1"], input=diff, text=True, check=True, capture_output=True)
    patches.apply()
    for relative in installation.sources:
        assert (installation.root / relative).read_bytes() == (reference / relative).read_bytes()
