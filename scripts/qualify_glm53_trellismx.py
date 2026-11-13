# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Qualify the GLM-5.3 Flash TrellisMX release by running its tests with none skipped.

Ordinary test runs skip the GPU cases without ``FLARE_RUN_CUDA_TESTS=1`` and the
release cases when no release is found. Qualification sets both for the release's
two test modules and fails unless every collected test ran and passed: a single
skip fails it, as does a failure, an error, or an empty collection.

    CUDA_VISIBLE_DEVICES=<assigned GPU> python scripts/qualify_glm53_trellismx.py --release <release root>
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TESTS = ("tests/recipes/test_glm53_trellismx.py", "tests/recipes/test_glm53_bf16_carrier.py")


class _Outcomes:
    """Every collected test, and every test report that did not pass."""

    def __init__(self):
        self.collected = ()
        self.passed = set()
        self.unpassed = {}

    def pytest_collection_finish(self, session):
        self.collected = tuple(item.nodeid for item in session.items)

    def pytest_runtest_logreport(self, report):
        if report.skipped or report.failed:
            self.unpassed.setdefault(report.nodeid, f"{report.outcome} in {report.when}")
        elif report.when == "call":
            self.passed.add(report.nodeid)


def main(argv=None, *, tests=TESTS) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--release", type=Path, required=True, help="the release root to qualify")
    args = parser.parse_args(argv)
    if not args.release.is_dir():
        parser.error(f"--release {args.release} is not a directory")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("set CUDA_VISIBLE_DEVICES to the GPU assigned for qualification")
    os.environ["FLARE_TRELLISMX_RELEASE"] = str(args.release.resolve())
    os.environ["FLARE_RUN_CUDA_TESTS"] = "1"
    os.chdir(ROOT)
    import pytest

    outcomes = _Outcomes()
    status = pytest.main(["-q", "-rs", "-p", "no:cacheprovider", *tests], plugins=[outcomes])
    unpassed = sorted(outcomes.unpassed.items())
    missing = sorted(set(outcomes.collected) - outcomes.passed - set(outcomes.unpassed))
    print(f"qualification: {len(outcomes.collected)} collected, {len(outcomes.passed)} passed,"
          f" {len(unpassed)} not passed")
    for nodeid, outcome in unpassed:
        print(f"  {outcome}: {nodeid}")
    for nodeid in missing:
        print(f"  not run: {nodeid}")
    qualified = status == 0 and outcomes.collected and not unpassed and not missing
    print("qualification PASSED" if qualified else "qualification FAILED")
    return 0 if qualified else 1


if __name__ == "__main__":
    sys.exit(main())
