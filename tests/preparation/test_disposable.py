# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Scalar capacity metadata does not hide disposable tensor backing."""
import pytest
import torch
from ignition.components.paged_batch import Metadata
from ignition.preparation.disposable import storage_ranges


def test_scalar_capacities_preserve_every_metadata_tensor_range():
    metadata = Metadata.allocate(8, 2, 2, 'cpu', samples=12, max_query_tokens=6)
    ranges = storage_ranges(metadata)
    assert len(ranges) == 11
    assert sum(size for _, size in ranges) == 428
    assert (metadata.token_ids.data_ptr(), 64) in ranges
    assert (metadata.sample_indices.data_ptr(), 96) in ranges
    with pytest.raises(TypeError, match='unsupported scratch declaration'):
        storage_ranges((metadata, object()))


def test_tensor_views_include_unused_backing_and_deduplicate_aliases():
    backing = torch.zeros(128, dtype=torch.float32)
    first, second = backing[:3], backing[64:65]
    assert storage_ranges((first, {"second": second, "repeated": first})) == (
        (backing.data_ptr(), backing.nbytes),)
