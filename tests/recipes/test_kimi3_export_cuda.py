# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shard index constants survive disposable-storage clearing and AOT rebinding."""

import ctypes as C
import os
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an assigned CUDA device")


class _ColumnForward:
    def __init__(self, rank):
        from ignition.components.parallel import TensorParallelShard
        from ignition.recipes.kimi3.decoder import DecoderStorage

        config = SimpleNamespace(hidden=7168, text={"routed_expert_hidden_size": 3584})
        self.storage = DecoderStorage.allocate(config, TensorParallelShard(rank, 16), "cuda:0")
        self.hidden = torch.arange(4 * 7168, dtype=torch.float32, device="cuda:0").view(4, 7168)
        self.latent = torch.arange(4 * 3584, dtype=torch.float32, device="cuda:0").view(4, 3584)
        self.storage.scratch.update(hidden=torch.empty((4, 448), device="cuda:0"),
                                    latent=torch.empty((4, 224), device="cuda:0"))

    def __call__(self):
        self.storage.scratch["hidden"].copy_(self.hidden.index_select(1, self.storage.hidden_columns))
        self.storage.scratch["latent"].copy_(self.latent.index_select(1, self.storage.latent_columns))


@torch.no_grad()
def test_shard_columns_survive_export_scratch_clearing(tmp_path):
    from ignition.preparation.aot import aot_forward
    from ignition.preparation.disposable import storage_ranges
    from ignition.preparation.runtime_core import _Cuda

    torch.cuda.set_device(0)
    first = _ColumnForward(0)
    artifact = tmp_path / "columns.bin"
    aot_forward(first).save_compiled_function(str(artifact))
    cuda = _Cuda()
    for rank in (0, 15):
        forward = _ColumnForward(rank)
        with artifact.open("rb") as source:
            compiled = torch.compiler.load_compiled_function(source)
        compiled(forward)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled(forward)
        torch.cuda.synchronize()
        for address, size in storage_ranges(forward.storage.scratch_tensors()):
            cuda.call("cuMemsetD8_v2", C.c_uint64(address), C.c_ubyte(0), C.c_size_t(size))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(forward.storage.scratch["hidden"],
                                   forward.hidden[:, rank * 448:(rank + 1) * 448], rtol=0, atol=0)
        torch.testing.assert_close(forward.storage.scratch["latent"],
                                   forward.latent[:, rank * 224:(rank + 1) * 224], rtol=0, atol=0)
        graph.reset()
