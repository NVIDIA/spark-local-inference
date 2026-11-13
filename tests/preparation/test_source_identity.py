# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Native source hashing preserves preparation's path and content identity."""

import hashlib
from pathlib import Path

import pytest


@pytest.fixture
def native_identity():
    import flare_heliopause

    return flare_heliopause.export_source_identity


def reference(roots):
    digest = hashlib.sha256()
    for source in roots:
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.suffix in (".py", ".rs", ".c", ".cu", ".cuh", ".h"):
                digest.update(str(path.relative_to(source)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def test_native_identity_matches_pathlib_order_suffixes_and_root_order(tmp_path, native_identity):
    first, second = tmp_path / "first", tmp_path / "second"
    for root in (first, second):
        (root / "nested").mkdir(parents=True)
    for name, contents in (("nested.py", b"root"), ("nested/a.py", b"child"),
                           (".hidden.cu", b"hidden"), ("UPPER.PY", b"excluded"),
                           ("data.bin", b"excluded"), ("μ.cuh", bytes(range(256)) * 1000)):
        (first / name).write_bytes(contents)
    (second / "kernel.rs").write_bytes(b"native")
    roots = [first, second]
    identity = native_identity(roots)
    assert identity == reference(roots)
    assert native_identity(roots[::-1]) == reference(roots[::-1])
    assert native_identity(roots) != native_identity(roots[::-1])
    (first / "data.bin").write_bytes(b"changed")
    assert native_identity(roots) == identity == reference(roots)
    (first / "nested.py").write_bytes(b"changed")
    assert native_identity(roots) == reference(roots)
    assert native_identity(roots) != identity


@pytest.mark.skipif(not hasattr(Path, "symlink_to"), reason="requires filesystem symlinks")
def test_native_identity_preserves_file_symlinks_and_ignores_directory_cycles(tmp_path, native_identity):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "target.py").write_bytes(b"target")
    try:
        (root / "linked.py").symlink_to(tmp_path / "target.py")
        (root / "cycle").symlink_to(tmp_path, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"filesystem symlink creation is unavailable: {error}")
    assert native_identity([root]) == reference([root])


def test_native_exact_directory_exclusion_preserves_other_roots(tmp_path, native_identity):
    package = tmp_path / "package"
    native = package / "data" / "include"
    native.mkdir(parents=True)
    (package / "api.py").write_text("python source\n")
    (native / "kernel.cuh").write_text("native source\n")
    excluded = native_identity([package], excluded_directories=[package / "data"])
    assert excluded == native_identity([package.parent / "absent", package],
                                       excluded_directories=[package / "data"])
    assert excluded != native_identity([package])
    # Exclusion is exact: an explicitly requested child root remains included.
    combined = native_identity([package, native], excluded_directories=[package / "data"])
    assert combined != excluded
    (native / "kernel.cuh").write_text("changed native source\n")
    assert native_identity([package], excluded_directories=[package / "data"]) == excluded
    assert native_identity([package, native], excluded_directories=[package / "data"]) != combined


def source_trees(root, *, editable):
    import shutil

    flare = root / "flare"
    for directory, filename in (("ignition", "model.py"), ("helios/src", "model.rs"),
                                ("heliopause/src", "export.rs")):
        path = flare / directory
        path.mkdir(parents=True)
        (path / filename).write_text(f"source for {filename}\n")
    repository = root / "flashinfer"
    package = repository / "flashinfer"
    (package / "experimental").mkdir(parents=True)
    (package / "data").mkdir()
    (package / "__init__.py").write_text("python package source\n")
    (package / "experimental" / "operator.py").write_text("prepared operator source\n")
    for directory, filename in (("include", "kernel.cuh"), ("csrc", "kernel.cu")):
        source = repository / directory
        source.mkdir()
        (source / filename).write_text(f"native source for {filename}\n")
        if editable:
            try:
                (package / "data" / directory).symlink_to(source, target_is_directory=True)
            except OSError as error:
                pytest.skip(f"filesystem symlink creation is unavailable: {error}")
        else:
            shutil.copytree(source, package / "data" / directory)
    # Wheel installs copy dependency data; editable installs link it. Neither
    # belongs to the owned Python/native trees in this compatibility identity.
    dependency = package / "data" / "dependency"
    dependency.mkdir()
    (dependency / "vendor.h").write_text("dependency source\n")
    outside = root / "unrelated.py"
    outside.write_text("outside all source roots\n")
    return flare, package, outside


def test_preparation_identity_matches_editable_wheel_and_checkout_locations(tmp_path):
    from ignition.preparation.source_identity import _source_identity

    first = source_trees(tmp_path / "first", editable=True)
    second = source_trees(tmp_path / "other-location", editable=True)
    wheel = source_trees(tmp_path / "wheel-layout", editable=False)
    expected = _source_identity(*first[:2])
    assert _source_identity(*second[:2]) == expected
    assert _source_identity(*wheel[:2]) == expected
    first[2].write_text("unrelated outside edit\n")
    (first[1] / "data" / "dependency" / "vendor.h").write_text("different dependency data\n")
    assert _source_identity(*first[:2]) == expected


@pytest.mark.parametrize("directory,filename", [("include", "kernel.cuh"), ("csrc", "kernel.cu")])
@pytest.mark.parametrize("editable", [True, False])
def test_preparation_identity_tracks_only_owned_native_source_edits(tmp_path, directory, filename, editable):
    from ignition.preparation.source_identity import _source_identity

    flare, package, outside = source_trees(tmp_path / "source", editable=editable)
    before = _source_identity(flare, package)
    source = package / "data" / directory / filename
    source.write_text("changed native implementation\n")
    assert _source_identity(flare, package) != before


def test_preparation_identity_rejects_missing_flashinfer_native_data(tmp_path):
    from ignition.preparation.source_identity import _source_identity

    package = tmp_path / "flashinfer"
    package.mkdir()
    with pytest.raises(RuntimeError, match="FlashInfer native source directory is unavailable"):
        _source_identity(tmp_path / "flare", package)
