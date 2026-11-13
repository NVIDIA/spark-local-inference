# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compressor state snapshots preserve accepted prefixes during native replay."""

import os

import pytest
import torch
from cuda.bindings import driver as cuda
from b12x.attention import mla_compress
from b12x.preparation import PreparationSession, PreparedCall

from ignition.components.state import gather_rows, gather_rows_three
from ignition.components.paged_batch import Metadata
from ignition.preparation.compilation import compile_forward
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.recipes.deepseek41.attention import PreparedCompressor
from ignition.recipes.deepseek41.pool import ContinuationPool

from .test_deepseek41_pool import config

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize("width", [512, 513, 1025])
@torch.no_grad()
def test_three_typed_outputs_keep_bits_bounds_and_source_lifetime(rows, width):
    device = torch.device("cuda:0")
    plane = (width * 4 + 255) // 256 * 256
    backing = torch.empty((7, plane * 2 + 256), dtype=torch.uint8, device=device)
    sources = (backing[:, :width * 4].view(torch.float32),
               backing[:, plane:plane + width * 4].view(torch.float32),
               backing[:, 2 * plane:2 * plane + 8].view(torch.int64))
    owners = [torch.empty(rows * value.shape[1] + 16, dtype=value.dtype, device=device)
              for value in sources]
    output = tuple(owner[:-16].view(rows, value.shape[1])
                   for owner, value in zip(owners, sources, strict=True))
    slots = torch.arange(rows, dtype=torch.int64, device=device)

    def forward():
        gather_rows_three(sources, slots, output)
        # All field reads must finish before their common source storage is reused.
        backing.fill_(79)

    compiled = compile_forward(forward)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    bundle = PreparedGraphBundle(device=0, graphs={"fields": graph}, buffers={}, resources=(compiled, backing, owners))
    worker = bundle.start_worker()
    try:
        for step in range(12):
            backing.random_(0, 256)
            sources[0][:, :4].view(torch.int32).copy_(torch.tensor(
                [0, -2147483648, 0x7fc01234, 0x7f800000], dtype=torch.int32, device=device).expand(7, 4))
            sources[2].copy_((torch.arange(7, device=device) + step * 31).view(7, 1) * 2**40)
            slots.copy_((torch.arange(rows, device=device) + step * 2) % 7)
            expected = tuple(value[slots].contiguous() for value in sources)
            for owner in owners:
                owner.fill_(91)
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            ticket = worker.submit(0)
            completion = worker.poll(timeout_ms=120000)
            assert completion is not None and completion.ticket == ticket and completion.error is None
            after = torch.cuda.memory_stats()
            for key in ("allocation.all.allocated", "allocation.all.freed",
                        "segment.all.allocated", "segment.all.freed"):
                assert before[key] == after[key]
            for actual, reference, owner in zip(output, expected, owners, strict=True):
                assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))
                assert owner[-16:].eq(91).all()
            assert backing.eq(79).all()
    finally:
        worker.close()
        graph.reset()


class CompressorCase:
    """Use prepared compression with precomputed projections and private snapshots."""

    def __init__(self, rows, layer, session, *, grouped):
        device = torch.device("cuda:0")
        self.rows, self.grouped = rows, grouped
        self.pool = ContinuationPool.allocate(config(), rows + 3, 1, device)
        self.metadata = Metadata.allocate(rows * 8, rows, 1, device, samples=rows * 8)
        self.metadata.query_start_loc.copy_(torch.arange(rows + 1, device=device, dtype=torch.int32) * 8)
        self.metadata.starts.fill_(128)
        self.metadata.state_rows.copy_(torch.arange(rows, device=device) + 1)
        self.metadata.live_counts.copy_(torch.tensor([rows * 8, rows], dtype=torch.int32, device=device))
        self.counts = torch.full((rows,), 8, dtype=torch.int32, device=device)
        op = PreparedCompressor.__new__(PreparedCompressor)
        self.operator = op
        op.pool, op.metadata, op.layer, op.ratio, op.deferred = self.pool, self.metadata, layer, 2, True
        op.values = torch.zeros((rows * 8, 512), device=device)
        op.gates = torch.zeros_like(op.values)
        op.output = torch.zeros_like(op.values, dtype=torch.bfloat16)
        op.emitted = torch.zeros(rows * 8, dtype=torch.bool, device=device)
        op.slots = torch.full((rows * 8,), -1, dtype=torch.int64, device=device)
        op.pending = {key: torch.zeros((rows, 512) if key != "pending_position" else (rows,),
            dtype=torch.float32 if key != "pending_position" else torch.int64, device=device)
            for key in ("pending_values", "pending_gates", "pending_position")}
        self.sources = tuple(self.pool.region(f"layer{layer}.{key}") for key in op.pending)
        self.destinations = tuple(value.view(rows, -1) for value in op.pending.values())
        self.saved = tuple(torch.empty_like(value) for value in self.destinations)
        op.project_values = op.project_gates = lambda source: None
        op.binding = None
        arguments = dict(values=op.values, gates=op.gates, weight=torch.ones(512, device=device),
            query_start_loc=self.metadata.query_start_loc, positions=self.metadata.starts,
            state_ids=torch.arange(rows, device=device, dtype=torch.int64),
            destination_slots=torch.arange(rows * 8, device=device, dtype=torch.int64),
            live_counts=self.metadata.live_counts, out=op.output, emitted=op.emitted,
            emitted_slots=op.slots, **op.pending)
        op.plan = mla_compress.plan(mla_compress.Caps(device=device, max_tokens=rows * 8,
            max_requests=rows, max_states=rows, ratio=2))

        def prepare(state):
            op.binding = state.bind(**arguments, plan=op.plan)
            return PreparedCall(run=lambda: state.run(op.binding))

        session.prepare((op.plan.request(name=f"compressor.{grouped}", prepare_call=prepare),))

    def __call__(self):
        op = self.operator
        if self.grouped:
            gather_rows_three(self.sources, self.metadata.state_rows, self.destinations)
            mla_compress.run(op.binding)
        else:
            for source, destination in zip(self.sources, self.destinations, strict=True):
                gather_rows(source, self.metadata.state_rows, destination)
            mla_compress.run(op.binding)
        for source, destination in zip(self.destinations, self.saved, strict=True):
            destination.copy_(source)
        op.commit(self.counts, (op.values, op.gates), self.metadata)
        # Acceptance no longer reads private snapshots or projected source rows.
        for value in (*self.destinations, op.values, op.gates):
            value.fill_(37)


@pytest.mark.parametrize("rows", [1, 3])
@pytest.mark.parametrize("layer", [2, 8, 14])
@torch.no_grad()
def test_compressor_snapshots_and_acceptance_are_bitwise_under_native_replay(rows, layer):
    device = torch.device("cuda:0")
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        cases = [CompressorCase(rows, layer, session, grouped=grouped) for grouped in (False, True)]
        compiled = [compile_forward(case) for case in cases]
        graphs = {}
        for name, forward in zip(("separate", "grouped"), compiled, strict=True):
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with session.capture(), torch.cuda.graph(graph):
                forward()
            graph.instantiate()
            graphs[name] = graph
        node_counts = []
        for graph in graphs.values():
            status, _, count = cuda.cuGraphGetNodes(cuda.CUgraph(graph.raw_cuda_graph()))
            assert status == cuda.CUresult.CUDA_SUCCESS
            node_counts.append(count)
        assert node_counts[0] - node_counts[1] == 2
        bundle = PreparedGraphBundle(device=0, graphs=graphs, buffers={}, resources=(cases, compiled))
        worker = bundle.start_worker()
        try:
            for step in range(36):
                accepted = step % 9
                lengths = [8 if r == 0 else (step + r) % 9 for r in range(rows)]
                slots = [(step + r) % (rows + 2) + 1 if lengths[r] else 0 for r in range(rows)]
                starts = [127 + (step // 9) % 2 + r * 16 for r in range(rows)]
                counts = [min(accepted, length) for length in lengths]
                initial = torch.full_like(cases[0].pool.backing, 173)
                values = torch.randn((rows * 8, 512), device=device) * .125
                gates = torch.randn_like(values) * .25
                for case in cases:
                    m, op = case.metadata, case.operator
                    case.pool.backing.copy_(initial)
                    for value in case.sources[:2]:
                        value.fill_(.125)
                    case.sources[2].fill_(126)
                    case.metadata.state_rows.copy_(torch.tensor(slots, device=device))
                    m.starts.copy_(torch.tensor(starts, device=device))
                    qsl = [0]
                    for length in lengths:
                        qsl.append(qsl[-1] + length)
                    m.query_start_loc.copy_(torch.tensor(qsl, dtype=torch.int32, device=device))
                    m.live_counts.copy_(torch.tensor([qsl[-1], rows], dtype=torch.int32, device=device))
                    case.counts.copy_(torch.tensor(counts, dtype=torch.int32, device=device))
                    op.values.copy_(values)
                    op.gates.copy_(gates)
                    op.output.fill_(19)
                    op.emitted.fill_(False)
                    op.slots.fill_(-1)
                    for value in case.destinations:
                        value.fill_(97)
                torch.cuda.synchronize()
                initial_state = cases[0].pool.backing.clone()
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                for name in ("separate", "grouped") if step % 2 else ("grouped", "separate"):
                    ticket = worker.submit(bundle.graph_indices[name])
                    completion = worker.poll(timeout_ms=120000)
                    assert completion is not None and completion.ticket == ticket and completion.error is None
                after = torch.cuda.memory_stats()
                for key in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key]
                control, candidate = cases
                for left, right in zip((control.pool.backing, control.operator.output,
                        control.operator.emitted, control.operator.slots, *control.saved),
                        (candidate.pool.backing, candidate.operator.output,
                         candidate.operator.emitted, candidate.operator.slots, *candidate.saved), strict=True):
                    assert torch.equal(left.view(torch.uint8), right.view(torch.uint8))
                expected = initial_state.clone()
                for request, (slot, start, count) in enumerate(zip(slots, starts, counts, strict=True)):
                    if count == 0:
                        continue
                    position = start + count - 1
                    for key, source in (("pending_values", values), ("pending_gates", gates),
                                        ("pending_position", None)):
                        region = candidate.pool.layout.region(f"layer{layer}.{key}")
                        destination = expected[:, region.offset:region.offset + region.nbytes].view(region.dtype)
                        if source is None:
                            destination[slot, 0] = position if position % 2 == 0 else -1
                        elif position % 2 == 0:
                            destination[slot].copy_(source[qsl[request] + count - 1])
                assert torch.equal(candidate.pool.backing, expected)
        finally:
            worker.close()
            for graph in graphs.values():
                graph.reset()
