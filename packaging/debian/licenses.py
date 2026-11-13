# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Include native dependency notices in the Debian package."""

from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from licenses import collect, toolchain_notices

destination = Path(sys.argv[1])
host = next(line.split(": ", 1)[1] for line in subprocess.check_output(
    ["rustc", "--version", "--verbose"], text=True,
).splitlines() if line.startswith("host: "))
collect(destination, [host])
toolchain_notices(destination.parent / "rust-standard-library")
