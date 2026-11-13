# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Accepted-prefix cache writes and GPU-produced Engram requests under replay."""

import os

import pytest
import torch
from b12x.preparation import PreparationSession
from b12x.sequence import engram
from b12x.sequence.engram.reference import hash_reference
from b12x.sequence._shared.disk_table import MappedHostAllocation

from ignition.components.engram import EngramInputs, PreparedEngramRequests
from ignition.components.readiness import ReadinessTimings
from ignition.components.embedding_partition import EmbeddingHeadPartition
from ignition.components.compressed_mla import PreparedCompressedCacheWriter, PreparedCompressedMLA
from ignition.preparation.compilation import compile_forward
from ignition.recipes.deepseek41.attention import _commit_window, PreparedCompressor
from ignition.components.paged_batch import Metadata
from ignition.recipes.deepseek41.pool import ContinuationPool

from .test_deepseek41_pool import config

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


@torch.no_grad()
@pytest.mark.parametrize("functional", [False, True])
@pytest.mark.parametrize("rank", [None, 0, 1, 2, 3])
def test_graph_produced_hashes_publish_after_all_rows_are_visible(rank, functional):
    device = torch.device("cuda:0")
    tokens, rows = 12, 3
    token_map = tuple((i * 7) % 32 for i in range(32))
    geometry = engram.build_geometry(base_table_size=101, compressed_vocab_size=32)
    frame = torch.zeros(1, dtype=torch.int32, device=device)
    inputs = EngramInputs.allocate(tokens, device, timings=ReadinessTimings.allocate(frame, 1))
    ids = torch.tensor([13, 5, 7, 28, 13, 8, 2, 1, 5, 7, 9, 1], device=device)
    starts = torch.tensor([0, 3, 3, 9], dtype=torch.int32, device=device)
    history = torch.tensor([[2, 2, 2], [9, 10, 11], [5, 7, 8]], device=device)
    counts = torch.tensor([9, 3], dtype=torch.int32, device=device)
    sequence = torch.tensor([1], device=device)
    mailbox = MappedHostAllocation((tokens * 24 + 1,), torch.int64, device)
    mapping = None
    if rank is not None:
        partition = EmbeddingHeadPartition.for_rank(geometry.offsets[0], geometry.num_embeddings[0], rank, 4)
        mapping = torch.tensor(partition.row_mapping, device=device)
    try:
        with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
            operation = PreparedEngramRequests(inputs, ids, starts, history, counts, mailbox.device_view, sequence,
                layer=1, token_map=token_map, geometry=geometry, session=session, name="hash_requests",
                row_mapping=mapping)
            if functional:
                operation.functional()
                forward = lambda: operation(ids, starts, history, counts)
            else:
                forward = operation
            compiled = compile_forward(forward)
            compiled()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                compiled()
            for generation in (2, 3, 4):
                sequence.fill_(generation)
                ids.add_(1).remainder_(32)
                mailbox.host_view.fill_(-1)
                graph.replay()
                torch.cuda.synchronize()
                expected = hash_reference(ids[:9].tolist(), [True] * 9, starts.tolist(), list(range(rows)),
                    history.tolist(), token_map, geometry, 1)
                if mapping is not None:
                    for head in range(24):
                        if head not in partition.heads:
                            expected[:, head] = -1
                        else:
                            first, _, destination = partition.row_mapping[head]
                            expected[:, head] += destination - first
                    assert expected.ge(0).sum().item() == 9 * 6
                assert mailbox.host_view[-1].item() == generation
                assert torch.equal(mailbox.host_view[:-1].view(tokens, 24)[:9], expected)
                assert mailbox.host_view[:-1].view(tokens, 24)[9:].eq(-1).all()
            graph.reset()
    finally:
        mailbox.close()


@pytest.mark.parametrize("width", [6, 8, 256])
@torch.no_grad()
def test_window_commit_keeps_only_accepted_prefix_and_preserves_other_rows(width):
    device = torch.device("cuda:0")
    staged = torch.randint(0, 256, (4, (128 + width) * 528), dtype=torch.uint8, device=device)
    window = torch.full((7, 128 * 528 + 256), 231, dtype=torch.uint8, device=device)
    rows = torch.tensor([3, 1, 5, 0], device=device)
    starts = torch.tensor([127, 255, 88, 0], device=device)
    counts = torch.tensor([width, 2, 0, 0], dtype=torch.int32, device=device)
    def forward():
        torch.library.wrap_triton(_commit_window)[(128, 4)](staged, window, rows, starts, counts,
            window.stride(0), staged.stride(0), 1024)
    compiled = compile_forward(forward)
    compiled()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    for accepted in ([width, 2, 0, 0], [1, width, 3, 0]):
        counts.copy_(torch.tensor(accepted, device=device, dtype=torch.int32))
        window.fill_(231)
        graph.replay()
        torch.cuda.synchronize()
        expected = torch.full_like(window.cpu(), 231)
        source = staged.cpu().view(4, 128 + width, 528)
        for request, (row, start, count) in enumerate(zip(rows.tolist(), starts.tolist(), accepted)):
            for token in range(max(0, count - 128), count):
                offset = (start + token) % 128 * 528
                expected[row, offset:offset + 528] = source[request, 128 + token]
        assert torch.equal(window.cpu(), expected)
    graph.reset()


@pytest.mark.parametrize("compiled", [False, True])
@torch.no_grad()
def test_ratio_two_pending_state_selects_the_last_accepted_target_token(compiled):
    pool = ContinuationPool.allocate(config(), 5, 2, "cuda:0")
    metadata = Metadata.allocate(12, 3, 2, "cuda:0", samples=12)
    metadata.query_start_loc.copy_(torch.tensor([0, 4, 8, 12], device="cuda:0"))
    metadata.starts.copy_(torch.tensor([127, 8, 88], device="cuda:0"))
    metadata.state_rows.copy_(torch.tensor([1, 2, 3], device="cuda:0"))
    operator = PreparedCompressor.__new__(PreparedCompressor)
    operator.pool, operator.metadata, operator.layer, operator.ratio = pool, metadata, 2, 2
    operator.values = torch.arange(12 * 512, dtype=torch.float32, device="cuda:0").view(12, 512)
    operator.gates = -operator.values
    counts = torch.tensor([2, 2, 0], dtype=torch.int32, device="cuda:0")
    for name in ("pending_values", "pending_gates", "pending_position"):
        pool.region("layer2." + name).fill_(42)
    def forward():
        return operator.commit(counts, (operator.values, operator.gates), metadata)
    run = compile_forward(forward) if compiled else forward
    for name in ("pending_values", "pending_gates", "pending_position"):
        pool.region("layer2." + name).fill_(42)
    run()
    torch.cuda.synchronize()
    assert pool.region("layer2.pending_position")[:, 0].tolist() == [42, 128, -1, 42, 42]
    assert torch.equal(pool.region("layer2.pending_values")[1], operator.values[1])
    assert torch.equal(pool.region("layer2.pending_gates")[1], operator.gates[1])
    assert pool.region("layer2.pending_values")[2:].eq(42).all()


@pytest.mark.parametrize("block", [5, 7])
@torch.no_grad()
@pytest.mark.parametrize("mode", ("extend", "decode"))
def test_noncausal_draft_block_attends_to_future_positions_without_crossing_requests(block, mode):
    from b12x.attention._shared.mla.compressed_reference import compressed_sparse_mla_reference

    device = torch.device("cuda:0")
    torch.manual_seed(4151)
    span, rows = 128 + block, 2
    values = torch.randn((rows * span, 512), dtype=torch.bfloat16, device=device) * .2
    cache = torch.zeros((rows, span * 528), dtype=torch.uint8, device=device)
    slots = torch.arange(rows * span, device=device)
    query = torch.randn((rows * block, 16, 512), dtype=torch.bfloat16, device=device) * .2
    indices = torch.stack([torch.arange(span, device=device) + row * span
                          for row in range(rows) for _ in range(block)]).int()
    lengths = torch.full((rows * block,), span, dtype=torch.int32, device=device)
    sink = torch.randn(16, device=device)
    output = torch.empty_like(query)
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        writer = PreparedCompressedCacheWriter(values, cache, slots, page_size=span,
            kind="swa", session=session, name="draft_write")
        attention = PreparedCompressedMLA(query, cache, indices, lengths, sink, output,
            window_page_size=span, mode=mode, session=session, name="draft_attention", workspace=None)
        def forward():
            writer(values)
            attention()
        run = compile_forward(forward)
        run()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            run()
        for _ in range(2):
            values[span - 1].add_(1)
            graph.replay()
            torch.cuda.synchronize()
            expected = compressed_sparse_mla_reference(query, cache, indices, lengths,
                sm_scale=512 ** -.5, attn_sink=sink, swa_page_size=span, cache_format="deepseek_v41")
            torch.testing.assert_close(output.float(), expected.float(), atol=.012, rtol=.08)
        graph.reset()


@torch.no_grad()
@pytest.mark.parametrize("world", [3, 4])
def test_recipe_requests_replay_full_table_rows_with_tp_projection_ownership(world):
    from types import SimpleNamespace
    from ignition.components.parallel import TensorParallelShard
    from ignition.recipes.deepseek41.engram import Tables, PreparedTableRequests

    device = torch.device("cuda:0")
    tokens, rows = 12, 3
    geometry = engram.build_geometry(layer_ids=(1,), base_table_size=101, compressed_vocab_size=32)
    token_map = tuple(range(32))
    frame = torch.zeros(1, dtype=torch.int32, device=device)
    inputs = EngramInputs.allocate(tokens, device, timings=ReadinessTimings.allocate(frame, 1))
    ids = torch.tensor([13, 5, 7, 28, 13, 8, 2, 1, 5, 7, 9, 1], device=device)
    starts = torch.tensor([0, 3, 3, 9], dtype=torch.int32, device=device)
    histories = torch.tensor([[2, 2, 2], [9, 10, 11], [5, 7, 8]], device=device)
    stored_history = torch.cat((torch.zeros((1, 3), dtype=torch.int64, device=device), histories))
    counts = torch.tensor([9, rows], dtype=torch.int32, device=device)
    metadata = SimpleNamespace(token_ids=ids, query_start_loc=starts, live_counts=counts,
        starts=torch.full((rows,), 3, dtype=torch.int32, device=device),
        state_rows=torch.arange(1, rows + 1, dtype=torch.int32, device=device))
    sequence = torch.ones(1, dtype=torch.int64, device=device)
    mailbox = MappedHostAllocation((tokens * 24 + 1,), torch.int64, device)
    tables = Tables(geometry, token_map, {}, {1: inputs}, {1: mailbox}, sequence,
                    TensorParallelShard(world - 1, world), None)
    pool = SimpleNamespace(region=lambda name: stored_history)
    try:
        with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
            operation = PreparedTableRequests(tables, metadata, pool, session=session, name="shared_table_requests")
            compiled = compile_forward(lambda: operation(metadata))
            compiled()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                compiled()
            for generation in (2, 3, 4):
                sequence.fill_(generation)
                ids.add_(1).remainder_(32)
                mailbox.host_view.fill_(-1)
                before = torch.cuda.memory_stats(device)
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats(device)
                for key in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key]

                expected = hash_reference(ids[:9].tolist(), [True] * 9, starts.tolist(), list(range(rows)),
                    histories.tolist(), token_map, geometry, 1)
                actual = mailbox.host_view[:-1].view(tokens, 24)
                assert mailbox.host_view[-1].item() == generation
                assert torch.equal(actual[:9], expected)
                assert actual[9:].eq(-1).all()
                assert actual[:9].ge(0).all()
            graph.reset()
    finally:
        mailbox.close()
