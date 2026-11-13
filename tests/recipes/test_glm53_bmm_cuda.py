# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stored GLM KV projection partitions execute in prepared native BMM graphs."""

import os
from pathlib import Path

import pytest
import torch

from ignition.components.linear import BatchedQuantizedProjection
from ignition.components.parallel import TensorParallelShard
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.glm53.weights import TEXT_PREFIX, GlmWeightLoader

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("rows", [2, 4096])
@torch.no_grad()
def test_checkpoint_absorbed_projections_native_replay(world_size, rows):
    snapshot = Path(os.environ.get("FLARE_GLM53_SNAPSHOT",
        "/data/checkpoint-releases/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD"))
    manifest = inspect_indexed_checkpoint(snapshot)
    device = torch.device("cuda:0")
    shard = TensorParallelShard(world_size - 1, world_size)
    prefix = f"{TEXT_PREFIX}.layers.3.self_attn.kv_b_proj"
    with GlmWeightLoader(manifest, device, shard=shard, prefixes=(prefix + ".",)) as loader:
        packed = loader.load_linear_group((prefix,), row_ranges=(shard.partition(64 * 512, alignment=512),))
    heads = 64 // world_size
    values = packed.weight.values.view(heads, 512, 512)
    scales = packed.weight.scale_rows.view(heads, 512, 16)
    rhs = ((values[:, :256], scales[:, :256]), (values[:, 256:], scales[:, 256:]))
    inputs = [torch.zeros((rows, heads, width), dtype=torch.bfloat16, device=device).transpose(0, 1)
              for width in (256, 512)]
    outputs = [torch.empty((rows, heads, width), dtype=torch.bfloat16, device=device).transpose(0, 1)
               for width in (512, 256)]
    logical = [value.to(torch.bfloat16) * scale.view(torch.float8_e8m0fnu).to(torch.bfloat16).repeat_interleave(32, dim=-1)
               for value, scale in rhs]
    logical[1] = logical[1].transpose(1, 2)
    graph, worker = torch.cuda.CUDAGraph(), None
    with RecordedPreparationSession(device=device, checkpoint="GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD",
            workload_schema="glm-absorbed-projections-v1", numerical_policy="default") as session:
        try:
            projections = [BatchedQuantizedProjection(source, weight, output, major=major,
                session=session, name=name) for source, weight, output, major, name in
                zip(inputs, rhs, outputs, ("n", "k"), ("absorb-query", "expand-value"), strict=True)]
            for projection in projections:
                projection.functional(session=session)
            session.tune()
            assert all(projection.plan.memory_requirements().scratch_nbytes == 0 for projection in projections)

            def forward():
                for projection, source, output in zip(projections, inputs, outputs, strict=True):
                    output.copy_(projection(source))

            compiled = compile_forward(forward)
            compiled()
            torch.cuda.synchronize(device)
            with session.capture(), torch.cuda.graph(graph):
                compiled()
            session.freeze()
            bundle = PreparedGraphBundle(device=0, graphs={"projections": graph},
                buffers={name: tensor.transpose(0, 1) for name, tensor in
                    zip(("query", "value", "absorbed", "expanded"), (*inputs, *outputs), strict=True)},
                resources=(projections, packed, compiled, session))
            worker = bundle.start_worker()
            for _ in range(3):
                for source in inputs:
                    source.normal_(std=.1)
                before = torch.cuda.memory_stats(device)
                ticket = worker.submit(0, repetitions=3)
                completion = worker.poll(timeout_ms=120000)
                assert completion is not None and completion.ticket == ticket and completion.error is None
                after = torch.cuda.memory_stats(device)
                for counter in ("allocation.all.allocated", "allocation.all.freed",
                                "segment.all.allocated", "segment.all.freed"):
                    assert before[counter] == after[counter]
                for source, weight, output in zip(inputs, logical, outputs, strict=True):
                    reference = torch.bmm(source.double(), weight.double())
                    error = (torch.bmm(source, weight).double() - reference).abs().max()
                    assert (output.double() - reference).abs().max() <= error * 1.05 + 1e-12
        finally:
            if worker is not None:
                worker.close()
            graph.reset()
