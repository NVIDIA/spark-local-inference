# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Eligible weight views retain storage identity without allocating an eviction copy."""
import gc
import os

import pytest
import torch

from ignition.preparation.startup_tuning import resident_numeric_storage
from ignition.preparation.tensor_metadata import NumericStorage

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an assigned CUDA GPU")


def test_resident_weights_filter_size_device_and_expired_storage():
    from b12x.comm.pcie import StartupTuningStorage

    device = torch.device("cuda", 0)
    shared = StartupTuningStorage(device)
    minimum = shared.eviction_bytes
    numeric = NumericStorage()
    small = torch.empty(4096, dtype=torch.bfloat16, device=device)
    numeric.record(small)
    assert resident_numeric_storage(numeric, device=device, minimum_bytes=minimum) is None

    cpu = torch.empty(minimum + 1024, dtype=torch.uint8)
    numeric.record(cpu)
    assert resident_numeric_storage(numeric, device=device, minimum_bytes=minimum) is None

    expired = torch.empty(minimum + 512, dtype=torch.uint8, device=device)
    numeric.record(expired)
    del expired
    gc.collect()
    assert resident_numeric_storage(numeric, device=device, minimum_bytes=minimum) is None

    weight = torch.empty(minimum, dtype=torch.uint8, device=device)
    numeric.record(weight)
    before = torch.cuda.memory_allocated(device)
    source = resident_numeric_storage(numeric, device=device, minimum_bytes=minimum)
    assert source.data_ptr() == weight.data_ptr()
    assert source.nbytes == weight.nbytes
    assert torch.cuda.memory_allocated(device) == before
    assert resident_numeric_storage(numeric, device=torch.device("cuda", 1), minimum_bytes=minimum) is None
    assert resident_numeric_storage(numeric, device=device, minimum_bytes=minimum + 1) is None
    shared.use_eviction(source)
    assert torch.cuda.memory_allocated(device) == before
