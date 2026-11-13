# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package a native Flare executable and its license notices as tgz or zip.

The default archive holds the `flare` CLI. `--winui-packages` selects the Windows
desktop archive, which holds the single-file `Flare.exe` launcher.
"""

import argparse
from pathlib import Path
import shutil
import tarfile
import tempfile
import zipfile

from licenses import (
    CLI_MANIFESTS, ROOT, TARGETS, WINUI_MANIFESTS, WINUI_TARGETS, collect, nuget_notices, toolchain_notices,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--target", choices=sorted({*TARGETS, *WINUI_TARGETS}), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--winui-packages", type=Path, metavar="DIR",
        help="package the Windows desktop launcher named by --binary; DIR holds the NuGet packages "
             "restored from flare-winui/packages.config",
    )
    args = parser.parse_args()
    winui = args.winui_packages is not None
    if args.target not in (WINUI_TARGETS if winui else TARGETS):
        parser.error(f"--target {args.target} has no {'desktop' if winui else 'CLI'} archive")
    if not args.binary.is_file():
        parser.error("--binary must identify the executable built with the current Cargo lockfile and rustc")
    if args.output.exists():
        parser.error("--output already exists")
    if not str(args.output).endswith((".zip", ".tgz", ".tar.gz")):
        parser.error("--output must end in .zip, .tgz, or .tar.gz")
    with tempfile.TemporaryDirectory(prefix="flare-release-") as directory:
        stage = Path(directory) / "flare"
        stage.mkdir()
        if winui:
            executable = "Flare.exe"
        else:
            executable = "flare.exe" if "windows" in args.target else "flare"
        shutil.copy2(args.binary, stage / executable)
        for name in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"):
            shutil.copyfile(ROOT / name, stage / name)
        shutil.copytree(ROOT / "helios/licenses", stage / "licenses")
        collect(stage / "third_party/rust", [args.target], WINUI_MANIFESTS if winui else CLI_MANIFESTS)
        toolchain_notices(stage / "third_party/rust-standard-library")
        if winui:
            nuget_notices(stage / "third_party/nuget", args.winui_packages)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix == ".zip":
            with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for file in sorted(stage.rglob("*")):
                    if file.is_file():
                        archive.write(file, file.relative_to(stage.parent))
        else:
            with tarfile.open(args.output, "w:gz") as archive:
                archive.add(stage, arcname="flare")
    print(args.output)


if __name__ == "__main__":
    main()
