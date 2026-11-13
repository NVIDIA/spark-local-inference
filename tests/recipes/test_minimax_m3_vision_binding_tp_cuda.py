# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared vision artifacts bind each rank's feature-column tensor."""

import os
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
    reason="requires four assigned CUDA devices",
)


class _Merge(nn.Module):
    def forward(self, values):
        return values[:, :6144]


class _SelectionForward:
    def __init__(self, encoder, source, output):
        self.encoder, self.source, self.output = encoder, source, output

    def __call__(self):
        self.output.copy_(self.encoder(dict(patches=self.source)))


def _selection_fixture(rank, device):
    from ignition.recipes.minimax_m3.vision import PreparedVision

    encoder = PreparedVision.__new__(PreparedVision)
    source = (
        torch.arange(8 * 6144, device=device, dtype=torch.float32)
        .view(8, 6144)
        .bfloat16()
    )
    encoder.scratch = SimpleNamespace(
        patches=source,
        projector_reduced=torch.empty_like(source),
        merger_reduced=torch.empty(2, 6144, device=device, dtype=torch.bfloat16),
        output=torch.empty(2, 1536, device=device, dtype=torch.bfloat16),
    )
    encoder.patch = encoder.norm = encoder.projector = nn.Identity()
    encoder.blocks = []
    encoder.merger = _Merge()
    encoder.output_columns = torch.arange(rank * 1536, (rank + 1) * 1536, device=device)
    return encoder, source, torch.empty(2, 1536, device=device, dtype=torch.bfloat16)


@torch.no_grad()
def _rank(rank, rendezvous, cache):
    from ignition.preparation.capture import capture_forwards
    from ignition.preparation.collectives import CollectivePreparation
    from ignition.preparation.execution import GraphSignature
    from b12x.preparation import PreparationSession

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(minutes=10),
    )
    coordinator = CollectivePreparation(dist.group.WORLD)
    encoder, source, output = _selection_fixture(rank, device)
    forward = _SelectionForward(encoder, source, output)
    graphs = {}
    try:
        with PreparationSession(
            device=device,
            autotune=True,
            compile_workers=4,
            namespace={"checkpoint": "vision-column-binding"},
        ) as session:
            session.requested_compile_workers = 4
            graphs, compiled = capture_forwards(
                {"vision": forward},
                {"vision": GraphSignature("vision", 1, 8)},
                session=session,
                device=device,
                stream=torch.cuda.Stream(device=device),
                pool=torch.cuda.graph_pool_handle(),
                coordinator=coordinator,
                cache_root=cache,
            )
            assert getattr(encoder, "compilation_variant", "shared") == "shared"
            assert "vision" in compiled
            for iteration in range(3):
                source.add_(64)
                if iteration == 1:
                    encoder.output_columns.copy_(encoder.output_columns.flip(0))
                output.fill_(float("nan"))
                dist.barrier()
                graphs["vision"].replay()
                torch.cuda.synchronize(device)
                expected = source.reshape(2, 4, 6144)[
                    :, 0
                ].index_select(1, encoder.output_columns)
                torch.testing.assert_close(
                    output, expected, rtol=0, atol=0
                )
            dist.barrier()
    finally:
        for graph in graphs.values():
            graph.reset()
        dist.destroy_process_group()


def test_shared_vision_artifact_binds_rank_columns(tmp_path):
    assert torch.cuda.device_count() >= 4
    mp.spawn(
        _rank,
        args=((tmp_path / "rendezvous").as_uri(), tmp_path / "compiled"),
        nprocs=4,
        join=True,
    )
