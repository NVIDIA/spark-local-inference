# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fetch pinned upstream templates and prepare external fixtures for frontend tests."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ignition.preparation.frontend import annotated_template_asset


def main():
    from huggingface_hub import hf_hub_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hub-cache", type=Path)
    parser.add_argument("--recipe", action="append", help="Limit to the named recipe(s)")
    args = parser.parse_args()
    for patch_path in sorted((ROOT / "ignition/recipes").glob("*/assets/*.patch.json")):
        recipe = patch_path.parent.parent.name
        if args.recipe and recipe not in args.recipe:
            continue
        patch = json.loads(patch_path.read_text())
        source = patch["source"]
        path = hf_hub_download(source["repository"], source["filename"],
                               revision=source["revision"], cache_dir=args.hub_cache)
        original = Path(path).read_bytes()
        if recipe == "step5":
            from ignition.recipes.step5.frontend import template_asset

            asset = template_asset(Path(path).parent)
        else:
            asset = annotated_template_asset(original, patch_path)
        output = args.output / recipe
        output.mkdir(parents=True, exist_ok=True)
        name = patch_path.name.removesuffix(".patch.json")
        (output / f"{name}.source.jinja").write_bytes(original)
        (output / f"{name}.jinja").write_bytes(asset["text"].encode("utf-8"))
        print(output / f"{name}.jinja")


if __name__ == "__main__":
    main()
