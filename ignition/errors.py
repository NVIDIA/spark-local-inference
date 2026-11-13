# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command failures and the process exit codes that report them."""


class FlareError(Exception):
    """A failure while serving; subclasses name the stage that failed."""

    exit_code = 6


class UsageError(FlareError):
    exit_code = 2


class FetchError(FlareError):
    exit_code = 3


class CapacityError(FlareError):
    exit_code = 4


class PreparationError(FlareError):
    exit_code = 5
