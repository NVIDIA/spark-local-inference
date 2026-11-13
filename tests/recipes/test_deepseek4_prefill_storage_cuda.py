# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Sequential C4/C128 layers retain cache state while reusing prefill intermediates."""

import os
from pathlib import Path

import pytest
import torch

from ignition.components.paged_batch import Metadata
from ignition.components.parallel import TensorParallelShard
from ignition.components.rotary import CompressedRotary
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.deepseek4.attention import AttentionIndices, AttentionScratch, PreparedAttention
from ignition.recipes.deepseek4.config import REVISION
from ignition.recipes.deepseek4.pool import PAGE_TOKENS, ContinuationPool
from ignition.recipes.deepseek4.weights import DeepSeekWeightLoader

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an assigned CUDA device")


class OrderedSelection:
    """Order equal selected sets before comparing independently compiled forwards."""

    def __init__(self, operation):
        self.operation = operation

    def write_keys(self, keys, slots):
        self.operation.write_keys(keys, slots)

    def __call__(self, *args):
        selected = self.operation(*args)
        ordered = torch.where(selected >= 0, selected, 2147483647).sort(-1).values
        selected.copy_(torch.where(ordered == 2147483647, -1, ordered))
        return selected


@pytest.mark.parametrize("capacity", (128, 4096))
@torch.no_grad()
def test_prefill_layers_reuse_storage_after_committing_state(capacity):
    snapshot = Path.home() / (
        ".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-Vision-Exp/snapshots/" + REVISION)
    if not (snapshot / "model.safetensors.index.json").is_file():
        pytest.skip("DeepSeek V4 Flash Vision checkpoint is unavailable")
    device = torch.device("cuda", 0)
    torch.manual_seed(9471 + capacity)
    layers = (2, 3)
    with DeepSeekWeightLoader(inspect_indexed_checkpoint(snapshot), device,
            shard=TensorParallelShard(0, 2), prefixes=tuple(f"layers.{i}.attn." for i in layers)) as loader:
        config = loader.config
        weights = [loader.load_attention(i) for i in layers]
    blocks = (capacity + 256 + PAGE_TOKENS - 1) // PAGE_TOKENS
    pool = ContinuationPool.allocate(config, 3, blocks, device)
    metadata = Metadata.allocate(capacity, 1, blocks, device)
    source = torch.randn((capacity, 4096), device=device, dtype=torch.bfloat16)
    scratch = AttentionScratch.allocate(capacity, 2, device, blocks=blocks)
    output = torch.empty((len(layers), capacity, 4096), device=device, dtype=torch.bfloat16)
    rotary, arena = CompressedRotary(config, device), ScratchArena(device)
    owners, graphs = [], {}
    with RecordedPreparationSession(device=device, checkpoint=REVISION,
            workload_schema="deepseek4.prefill-storage.v1", numerical_policy="checkpoint") as session:
        session.register_scratch(arena)
        forwards = []
        for variant in range(2):
            indices = AttentionIndices(metadata, mode="prefill", tokens=capacity)
            operations = []
            for i, weight in zip(layers, weights, strict=True):
                operation = PreparedAttention(i, weight, source, scratch, metadata, pool, rotary,
                    mode="prefill", world=2, session=session, name=f"variant{variant}.layer{i}",
                    workspace=arena, indices=indices)
                operation.selectors = [OrderedSelection(s) for s in operation.selectors]
                operations.append(operation)
            indices.functional()

            def forward(indices=indices, operations=operations):
                packet = indices(metadata)
                for index, operation in enumerate(operations):
                    value, _ = operation(source, metadata, packet)
                    output[index].copy_(value)

            forwards.append(forward)
            owners.extend((indices, operations))
        session.tune()
        for name, forward in zip(("first", "second"), forwards, strict=True):
            compiled = compile_forward(forward)
            compiled()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                compiled()
            graphs[name] = graph
            owners.append(compiled)
        session.freeze()
        bundle = PreparedGraphBundle(device=0, graphs=graphs, buffers={"source": source, "output": output},
            resources=(pool, metadata, scratch, arena, session, rotary, weights, owners))
        worker = bundle.start_worker()
        try:
            expected = []
            for repetition, graph_index in enumerate((0, 0, 1)):
                pool.clear_for_preparation()
                for step, (start, count, row) in enumerate(((0, capacity - 7, 1),
                        (capacity - 7, 127, 1), (capacity + 120, 0, 0))):
                    metadata.starts.fill_(start)
                    metadata.lengths.fill_(count)
                    metadata.state_rows.fill_(row)
                    metadata.query_start_loc[1] = count
                    metadata.positions.fill_(-1)
                    metadata.positions[:count].copy_(torch.arange(start, start + count, device=device))
                    metadata.page_table.copy_(torch.arange(blocks - 1, -1, -1, device=device)[None, :])
                    metadata.live_counts.copy_(torch.tensor([count, int(count > 0)], device=device))
                    torch.cuda.synchronize()
                    before = torch.cuda.memory_stats()
                    ticket = worker.submit(graph_index)
                    completion = worker.poll(timeout_ms=120000)
                    assert completion is not None and completion.ticket == ticket and completion.error is None
                    after = torch.cuda.memory_stats()
                    for key in ("allocation.all.allocated", "allocation.all.freed",
                                "segment.all.allocated", "segment.all.freed"):
                        assert before[key] == after[key]
                    observed = [output[:, :count], pool.backing, *pool.pages.values()]
                    if repetition == 0:
                        expected.append([value.clone() for value in observed])
                    else:
                        for field, (actual, wanted) in enumerate(zip(observed, expected[step], strict=True)):
                            torch.testing.assert_close(actual, wanted, rtol=0, atol=0,
                                msg=lambda error: f"graph={graph_index}, repetition={repetition}, step={step}, "
                                                  f"field={field}: {error}")
        finally:
            worker.close()
            for graph in graphs.values():
                graph.reset()
