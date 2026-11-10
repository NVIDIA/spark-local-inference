# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Accepted-tail staging matches an independently packed 128-row context graph."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ignition.components.dspark import PreparedContext
from ignition.components.paged_batch import Metadata
from ignition.components.parallel import TensorParallelShard
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.deepseek4.config import REVISION
from ignition.recipes.deepseek4.pool import ContinuationPool
from ignition.recipes.deepseek4.weights import DeepSeekWeightLoader

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an assigned CUDA device")


@torch.no_grad()
def test_prefill_context_tail_preserves_window_bytes_under_native_replay():
    snapshot = Path.home() / (
        ".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-Vision-Exp/snapshots/" + REVISION)
    snapshot = Path(os.environ.get("FLARE_DS4_SNAPSHOT", snapshot))
    if not (snapshot / "model.safetensors.index.json").is_file():
        pytest.skip("DeepSeek V4 Flash Vision checkpoint is unavailable")
    device = torch.device("cuda", 0)
    torch.manual_seed(1941)
    prefixes = ("mtp.0.main_proj", "mtp.0.main_norm", *(f"mtp.{i}.attn.{part}"
                for i in range(3) for part in ("wkv", "kv_norm")))
    with DeepSeekWeightLoader(inspect_indexed_checkpoint(snapshot), device,
            shard=TensorParallelShard(0, 2), prefixes=prefixes) as loader:
        config = loader.config
        draft = SimpleNamespace(main_projection=loader.load_linear("mtp.0.main_proj"),
            context_kv=loader.load_linear_group(tuple(f"mtp.{i}.attn.wkv" for i in range(3))),
            main_norm=loader.load_tensor("mtp.0.main_norm.weight"), layers=tuple(
                SimpleNamespace(attention={"kv_norm": loader.load_tensor(f"mtp.{i}.attn.kv_norm.weight")}) for i in range(3)))
        separate = [loader.load_linear(f"mtp.{i}.attn.wkv").weight for i in range(3)]
        packed = draft.context_kv.weight
        assert torch.equal(packed.values.view(torch.uint8),
                           torch.cat([weight.values.view(torch.uint8) for weight in separate]))
        assert torch.equal(packed.scale_rows[0].view(torch.uint8),
                           torch.cat([weight.scale_rows[0].view(torch.uint8) for weight in separate]))
    weights = SimpleNamespace(config=config, draft=draft)
    pool = ContinuationPool.allocate(config, 3, 2, device)
    source = torch.randn((8192, len(config.draft.target_layers) * 4096), device=device, dtype=torch.bfloat16)
    reference_source = torch.empty_like(source[:128])
    arena = ScratchArena(device)
    variants, graphs, owners = {}, {}, []
    with RecordedPreparationSession(device=device, checkpoint=REVISION,
            workload_schema="deepseek4.context-tail.v1", numerical_policy="checkpoint") as session:
        session.register_scratch(arena)
        for capacity in (128, 256, 4096, 8192):
            inputs = reference_source if capacity == 128 else torch.empty_like(reference_source)
            metadata = Metadata.allocate(capacity, 1, 1, device)
            counts = torch.zeros(1, device=device, dtype=torch.int32)
            operation = PreparedContext(weights, inputs, metadata, pool, session=session,
                name=f"context{capacity}", workspace=arena, cache_format="deepseek_v4",
                compact_prefill=capacity != 128).functional()
            variants[capacity] = inputs, metadata, counts, operation
        session.tune()
        for capacity, (inputs, metadata, counts, operation) in variants.items():
            def forward(inputs=inputs, counts=counts, operation=operation, metadata=metadata):
                operation(inputs, counts, metadata)

            compiled = compile_forward(forward)
            compiled()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                compiled()
            graphs[str(capacity)] = graph
            owners.append(compiled)
        session.freeze()
        bundle = PreparedGraphBundle(device=0, graphs=graphs,
            buffers={"source": source, "reference_source": reference_source},
            resources=(variants, weights, pool, arena, session, owners))
        worker = bundle.start_worker()

        def replay(capacity):
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            ticket = worker.submit(bundle.graph_indices[str(capacity)])
            completion = worker.poll(timeout_ms=120000)
            assert completion is not None and completion.ticket == ticket and completion.error is None
            after = torch.cuda.memory_stats()
            for key in ("allocation.all.allocated", "allocation.all.freed",
                        "segment.all.allocated", "segment.all.freed"):
                assert before[key] == after[key]

        def metadata_for(m, *, count, start, begin, row):
            m.starts.fill_(start)
            m.lengths.fill_(count)
            m.state_rows.fill_(row)
            m.query_start_loc.copy_(torch.tensor([begin, begin + count], device=device))
            m.positions.fill_(-1)
            m.positions[begin:begin + count].copy_(torch.arange(start, start + count, device=device))

        try:
            pool.backing.fill_(37)
            for capacity in (8192, 256, 4096, 256):
                inputs, m, counts, operation = variants[capacity]
                for number, count in enumerate((0, 1, 127, 128, 129, capacity - 7, capacity)):
                    begin = 7 if count <= capacity - 7 else 0
                    start = (0, 127, 999983)[number % 3]
                    row = 1 + number % 2
                    kept = min(count, 128)
                    metadata_for(m, count=count, start=start, begin=begin, row=row)
                    counts.fill_(count)
                    _, reference_metadata, reference_counts, _ = variants[128]
                    reference_source.zero_()
                    reference_source[:kept].copy_(source[begin + count - kept:begin + count])
                    metadata_for(reference_metadata, count=kept, start=start + count - kept, begin=0, row=row)
                    reference_counts.fill_(kept)
                    snapshot = pool.backing.clone()
                    replay(128)
                    expected = pool.backing.clone()
                    pool.backing.copy_(snapshot)
                    arena.tensor.fill_(0x5A)
                    inputs.copy_(reference_source)
                    replay(capacity)
                    torch.testing.assert_close(pool.backing, expected, rtol=0, atol=0)
                    torch.testing.assert_close(inputs, reference_source, rtol=0, atol=0)
        finally:
            worker.close()
            for graph in graphs.values():
                graph.reset()
