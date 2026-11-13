# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Root proposal blocks replace divergent peer chains before native consumers."""

import json
import os

import pytest
import torch
import torch.multiprocessing as mp

from tests.components.draft_block_broadcast_support import rank_main


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1" or torch.cuda.device_count() < 2,
                    reason="requires two explicitly assigned CUDA devices")
def test_canonical_block_under_native_tp2_replay(tmp_path):
    output = tmp_path / "results.json"
    mp.spawn(rank_main, args=((tmp_path / "rendezvous").as_uri(), output), nprocs=2, join=True)
    result = json.loads(output.read_text())
    assert set(result["summary"]) == {"1", "8", "32"}
