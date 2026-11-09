# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Ignition's direct execution: a diagnostic and qualification path, not production serving.

Production serving loads an exported core in Helios, which owns scheduling,
input packing and graph replay without Python (see AGENTS.md, "Python
prepares; Helios serves"). At the project owner's request, `ignition debug-serve
--direct` instead prepares a model in process and drives its prepared
execution from Python: a small fixed scheduler packs the same graph inputs
Helios packs, and rank zero replays the captured graphs, or with `--eager`
calls the forwards those graphs captured, while every other rank executes the
same steps in lockstep. It exists to reproduce and bisect model-level faults
without Helios, never as a fallback; nothing selects it implicitly.

`scheduler` holds the CPU-only scheduling, cache bookkeeping and input packing,
`executor` runs one rank's steps, and `server` exposes a minimal non-streaming
OpenAI-compatible endpoint over them.
"""
