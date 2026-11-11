# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint position interpolation with compiler-planned temporaries on TP16."""

from datetime import timedelta
import json
import os
from pathlib import Path

import pytest
import torch

WORLD = 16
CHECKPOINT = Path(os.environ.get("FLARE_KIMI31_CHECKPOINT", str(Path.home() /
    ".cache/huggingface/hub/models--moonshotai--Kimi-K3.1-Pro/snapshots/"
    "d9e99b0792a5e93b78ea9bea62a68167e36ddd9e")))
pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
                              reason="requires sixteen explicitly assigned CUDA devices")


class _Interpolation:
    def __init__(self, weight, indices, factors, output):
        self.weight, self.indices, self.factors, self.output = weight, indices, factors, output

    def __call__(self):
        from ignition.recipes.kimi3.vision import interpolate_positions

        values = torch.empty_like(self.output)
        interpolate_positions(self.weight, self.indices, self.factors, values)
        self.output.copy_(values)


@torch.no_grad()
def _rank(rank, rendezvous):
    import torch.distributed as dist
    from safetensors import safe_open

    from ignition.preparation.compilation import compile_forward
    from ignition.recipes.kimi3.vision import PATCH_CAPACITIES

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=WORLD,
                            timeout=timedelta(minutes=10))
    try:
        field = "vision_tower.patch_embed.pos_emb.weight"
        index = json.loads((CHECKPOINT / "model.safetensors.index.json").read_text())
        with safe_open(CHECKPOINT / index["weight_map"][field], framework="pt", device=rank) as source:
            weight = source.get_tensor(field).reshape(-1, 1024)
        patches = torch.full((max(PATCH_CAPACITIES), 588), 17, dtype=torch.bfloat16, device=device)
        destination = torch.empty((max(PATCH_CAPACITIES), 1024), dtype=torch.bfloat16, device=device)
        torch.manual_seed(31 + rank)
        indices = torch.randint(weight.shape[0], (max(PATCH_CAPACITIES), 4), device=device)
        factors = torch.rand(indices.shape, device=device)
        factors /= factors.sum(1, keepdim=True)
        stream = torch.cuda.Stream(device=device)
        pool = torch.cuda.graph_pool_handle()
        graphs = []
        compiled_forwards = []
        try:
            for rows in (*reversed(PATCH_CAPACITIES), 4100):
                output = destination[:rows]
                expected = (weight[indices[:rows]].float() * factors[:rows, :, None]).sum(1).bfloat16()
                forward = compile_forward(_Interpolation(weight, indices[:rows], factors[:rows], output))
                compiled_forwards.append(forward)
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    forward()
                stream.synchronize()
                torch.testing.assert_close(output, expected, atol=0, rtol=0)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream, pool=pool):
                    forward()
                graphs.append(graph)
                destination.fill_(float("nan"))
                before = torch.cuda.memory_stats()
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                torch.testing.assert_close(output, expected, atol=0, rtol=0)
                assert torch.all(patches == 17)
                assert after["allocation.all.allocated"] == before["allocation.all.allocated"]
                assert after["allocation.all.freed"] == before["allocation.all.freed"]
            reserved = sum(segment["total_size"] for segment in torch.cuda.memory_snapshot()
                           if segment["device"] == rank and tuple(segment["segment_pool_id"]) == pool)
            sizes = [None] * WORLD
            dist.all_gather_object(sizes, reserved)
            assert max(sizes) <= 192 << 20, sizes
            if rank == 0:
                print("position interpolation graph-pool bytes per rank", sizes, flush=True)
        finally:
            for graph in graphs:
                graph.reset()
    finally:
        dist.destroy_process_group()


def test_checkpoint_vision_position_interpolation_on_tp16(tmp_path):
    import torch.multiprocessing as mp

    assert torch.cuda.device_count() == WORLD, "assign exactly sixteen CUDA devices"
    mp.spawn(_rank, args=((tmp_path / "vision-position-interpolation").as_uri(),), nprocs=WORLD, join=True)
