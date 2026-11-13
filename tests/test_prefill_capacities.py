# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prefill graph capacities shared by recipes and Helios."""

from ignition.components.capacity import prefill_capacities


def test_prefill_capacities_step_after_doubling_and_end_at_the_chunk():
    assert prefill_capacities(64) == (64,)
    assert prefill_capacities(1000) == (128, 256, 512, 1000)
    assert prefill_capacities(4096) == (128, 256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096)
    assert prefill_capacities(8192) == (128, 256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192)
