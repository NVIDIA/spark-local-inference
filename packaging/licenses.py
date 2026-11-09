# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Collect license texts for Flare's locked native dependency graph."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu", "x86_64-pc-windows-msvc")


def dependency_roles(metadata):
    """Exclude dev edges; proc macros and everything below build edges are tools."""
    packages = {p["id"]: p for p in metadata["packages"]}
    nodes = {n["id"]: n for n in metadata["resolve"]["nodes"]}
    pending = [(metadata["resolve"]["root"], "runtime")]
    seen = set()
    roles = defaultdict(set)
    while pending:
        package_id, role = pending.pop()
        if (package_id, role) in seen:
            continue
        seen.add((package_id, role))
        if packages[package_id]["source"] is not None:
            roles[package_id].add(role)
        for dep in nodes[package_id]["deps"]:
            macro = any("proc-macro" in t["kind"] for t in packages[dep["pkg"]]["targets"])
            for kind in dep["dep_kinds"]:
                if kind["kind"] == "dev":
                    continue
                child = "build" if role == "build" or kind["kind"] == "build" or macro else "runtime"
                pending.append((dep["pkg"], child))
    return roles


def license_files(root):
    return sorted(
        p for p in root.rglob("*") if p.is_file()
        and (p.name.lower().startswith(("license", "licence", "copying", "copyright", "notice"))
             or p.name.lower() == "readme.ijg")
        and p.suffix.lower() not in {".rs", ".py", ".c", ".h"}
    )


def collect(destination, targets=TARGETS):
    # Refuse to merge with old output: obsolete package notices must not survive regeneration.
    if destination.exists():
        raise ValueError(f"destination already exists: {destination}; generate into a fresh directory")
    packages = {}
    roles = defaultdict(dict)
    for target in targets:
        metadata = json.loads(subprocess.check_output([
            "cargo", "metadata", "--locked", "--format-version", "1",
            "--manifest-path", str(ROOT / "flare-cli/Cargo.toml"), "--filter-platform", target,
        ], cwd=ROOT))
        packages.update((p["id"], p) for p in metadata["packages"])
        for package_id, kinds in dependency_roles(metadata).items():
            roles[package_id][target] = sorted(kinds)

    destination.mkdir(parents=True)
    entries = []
    for package_id in sorted(roles, key=lambda k: (packages[k]["name"], packages[k]["version"])):
        package = packages[package_id]
        root = Path(package["manifest_path"]).parent
        name = f'{package["name"]}-{package["version"]}'
        target = destination / name
        files = set(license_files(root))
        if package["license_file"]:
            files.add(root / package["license_file"])
        supplement = ROOT / "packaging/license-supplements" / name
        extra = license_files(supplement) if supplement.exists() else []
        if not files and not extra:
            raise ValueError(f"missing license text for {name}")
        copied = []
        for source, relative in [(p, p.relative_to(root)) for p in sorted(files)] + [
            (p, Path("upstream") / p.relative_to(supplement)) for p in extra
        ]:
            output = target / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, output)
            copied.append(output.relative_to(destination).as_posix())
        if extra:
            shutil.copyfile(supplement / "SOURCE.txt", target / "upstream/SOURCE.txt")
        source_url = package["repository"] or package["source"]
        if package["source"].startswith("registry+"):
            source_url = f'https://crates.io/api/v1/crates/{package["name"]}/{package["version"]}/download'
        elif package["source"].startswith("git+"):
            repository, _, revision = package["source"][4:].partition("#")
            source_url = repository.split("?", 1)[0].removesuffix(".git") + "/tree/" + revision
        entries.append({
            "name": package["name"], "version": package["version"],
            "license": package["license"], "source": package["source"],
            "source_url": source_url, "targets": roles[package_id], "license_files": copied,
        })
    (destination / "packages.json").write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    groups = defaultdict(list)
    for entry in entries:
        groups[entry["license"] or "See license files"].append(entry)
    lines = ["# Native Rust dependency licenses", "",
             "Generated from the locked `flare-cli/Cargo.toml` dependency graph. Python and",
             "Heliopause are preparation/build-only and are not part of this binary release.", "",
             "This inventory includes runtime dependencies and build/proc-macro tools,",
             "with target-specific roles in `packages.json`. Build-tool notices are included",
             "conservatively; their presence does not mean those tools are shipped.", "",
             "License expressions are preserved as declared, including legacy slash notation.",
             "Embedded native code may have additional terms in the linked license files.", ""]
    for license_name, members in sorted(groups.items()):
        lines.extend([f"## {license_name}", "", "| Package | Role | Source | License texts |",
                      "| --- | --- | --- | --- |"])
        for entry in members:
            runtime = any("runtime" in kinds for kinds in entry["targets"].values())
            role = "Runtime" if runtime else "Build/proc-macro only"
            links = ", ".join(f"[{Path(p).name}]({p})" for p in entry["license_files"])
            lines.append(f'| {entry["name"]} {entry["version"]} | {role} | [source]({entry["source_url"]}) | {links} |')
        lines.append("")
    (destination / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return entries


def toolchain_notices(destination):
    """Use the notices from the compiler that built the packaged executable."""
    sysroot = Path(subprocess.check_output(["rustc", "--print", "sysroot"], text=True).strip())
    docs = sysroot / "share/doc/rust"
    candidates = [docs / "COPYRIGHT-library.html", docs / "COPYRIGHT.html"]
    notice = next((p for p in candidates if p.is_file()), None)
    if notice is None:
        raise ValueError("Rust toolchain copyright notices are missing; install the rust-docs component")
    destination.mkdir(parents=True)
    shutil.copyfile(notice, destination / notice.name)
    for name in ("LICENSE-APACHE", "LICENSE-MIT"):
        if (docs / name).is_file():
            shutil.copyfile(docs / name, destination / name)
    (destination / "version.txt").write_text(
        subprocess.check_output(["rustc", "--version", "--verbose"], text=True), encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--target", action="append", choices=TARGETS)
    args = parser.parse_args()
    entries = collect(args.destination, args.target or TARGETS)
    print(f"Collected license texts for {len(entries)} crates in {args.destination}")


if __name__ == "__main__":
    main()
