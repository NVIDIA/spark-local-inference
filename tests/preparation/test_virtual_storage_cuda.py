# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""In-process runtime-sized storage keeps its full shape while only a prefix is backed."""

import os
import subprocess
import sys

import pytest


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                    reason="requires an explicitly assigned CUDA device")
def test_virtual_pages_grow_in_place_after_compilation():
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    subprocess.run([sys.executable, "-c", """
import torch
from ignition.preparation.virtual_storage import VirtualStorage

torch.empty(1, device="cuda:0")
storage = VirtualStorage("cuda:0")
unit = storage.granularity // 2
pages = storage.device_allocate("kv", (1024, unit), torch.uint8, group="KV_PAGES", maximum=1024, units=4)
index = storage.device_allocate("index", (1024, 8), torch.uint8, group="KV_PAGES", maximum=1024, units=4)
other = storage.device_allocate("other", (16, unit), torch.uint8, group="OTHER", maximum=16, units=16)
assert pages.shape == (1024, unit) and storage.nbytes == 2 * storage.granularity + storage.granularity + 8 * storage.granularity
pages[:4].fill_(7)
index[:4].fill_(3)
address = pages.data_ptr()
storage.grow("KV_PAGES", 100)
torch.cuda.synchronize()
assert pages.data_ptr() == address and storage.nbytes == 50 * storage.granularity + storage.granularity + 8 * storage.granularity
assert bool((pages[:4] == 7).all()) and bool((pages[4:100] == 0).all())
assert bool((index[:4] == 3).all()) and bool((index[4:100] == 0).all())
pages[:100].add_(1)
torch.cuda.synchronize()
storage.close()
storage.close()
assert not storage.dynamic_regions
"""], check=True)
