# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Disposable Qwen storage cleared before publishing a runtime core."""

from ignition.preparation.disposable import storage_ranges


def disposable_ranges(*, cache):
    values = []
    for state in cache.values():
        values.extend((state.main_k_cache, state.main_v_cache, state.compressed_k_cache, state.raw_k_ring))
    return storage_ranges(values)
