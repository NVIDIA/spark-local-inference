# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compiled window snapshots preserve immutable history and accepted prefixes."""

import os

import pytest
import torch
from b12x.preparation import PreparationSession
from cuda.bindings import driver as cuda

from ignition.components.compressed_mla import PreparedCompressedCacheWriter
from ignition.components.paged_batch import Metadata
from ignition.components.ring_commit import PreparedRingCommit
from ignition.components.state import StateLayout, StateStorage, gather_rows
from ignition.components.window_indices import PreparedWindowIndices
from ignition.preparation.compilation import compile_forward
from ignition.preparation.graphs import PreparedGraphBundle

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


class SnapshotCase:
    def __init__(self, requests, width, mode, session, *, fused):
        device = torch.device("cuda:0")
        self.mode, self.fused = mode, fused
        layout = StateLayout.from_specs((("before", (256,), torch.uint8),
            ("window", (128 * 528,), torch.uint8), ("after", (256,), torch.uint8)))
        self.storage = StateStorage.allocate(layout, requests + 3, device)
        self.history = self.storage.region("window")
        self.metadata = Metadata.allocate(requests * width, requests, 1, device,
                                         max_query_tokens=width)
        self.indices = PreparedWindowIndices(self.metadata, mode=mode,
            tokens=requests * width, page_size=128 + width,
            width=128 + width if mode == "draft" else 128)
        self.values = torch.zeros((requests * width, 512), dtype=torch.bfloat16, device=device)
        self.counts = torch.zeros(requests, dtype=torch.int32, device=device)
        self.commit = PreparedRingCommit(self.storage.backing,
            (layout.region("window").offset,), self.metadata.state_rows, self.metadata.starts,
            token_capacity=width, window_tokens=128, record_bytes=528)
        self.cache = self.commit.staging[0]
        self.observed = torch.empty_like(self.cache)
        self.writer = PreparedCompressedCacheWriter(self.values, self.cache, self.indices.window_slots,
            page_size=128 + width, kind="swa", session=session, name=f"window.{fused}",
            history=self.history if fused else None,
            history_rows=self.metadata.state_rows if fused else None,
            snapshot_tokens=128 if fused else 0)

    def __call__(self):
        self.indices()
        if not self.fused:
            gather_rows(self.history, self.metadata.state_rows, self.cache[:, :128 * 528])
        self.writer(self.values)
        self.observed.copy_(self.cache)
        if self.mode == "verify":
            self.commit(self.counts)
        # Consumers and accepted commits must finish before private inputs are reused.
        self.values.fill_(37)
        self.cache.fill_(129)


@pytest.mark.parametrize("requests", [1, 3])
@pytest.mark.parametrize("mode,width", [("verify", 8), ("draft", 7)])
@torch.no_grad()
def test_compiled_native_snapshot_writer_preserves_ragged_state_and_lifetimes(requests, mode, width):
    device = torch.device("cuda:0")
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        cases = [SnapshotCase(requests, width, mode, session, fused=fused) for fused in (False, True)]
        compiled = [compile_forward(case) for case in cases]
        graphs = {}
        counts = []
        for name, forward in zip(("separate", "fused"), compiled, strict=True):
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with session.capture(), torch.cuda.graph(graph):
                forward()
            graph.instantiate()
            graphs[name] = graph
            status, _, count = cuda.cuGraphGetNodes(cuda.CUgraph(graph.raw_cuda_graph()))
            assert status == cuda.CUresult.CUDA_SUCCESS
            counts.append(count)
        assert counts[0] - counts[1] == 1
        bundle = PreparedGraphBundle(device=0, graphs=graphs, buffers={}, resources=(cases, compiled))
        worker = bundle.start_worker()
        try:
            for step in range(2 * (width + 1)):
                initial = torch.randint(0, 256, cases[0].storage.backing.shape, dtype=torch.uint8, device=device)
                values = torch.randn_like(cases[0].values) * .25
                values[0].zero_()
                values[-1].fill_(2**-20)
                lengths = [width if r == 0 else (step + r) % (width + 1) for r in range(requests)]
                selected = [(step + r) % (requests + 2) + 1 for r in range(requests)]
                starts = torch.tensor([127 + (step // (width + 1)) + r * 128
                                       for r in range(requests)], device=device)
                length_tensor = torch.tensor(lengths, dtype=torch.int32, device=device)
                accepted = [min(step % (width + 1), length) for length in lengths]
                tokens = torch.zeros((requests, width), dtype=torch.int64, device=device)
                pages = torch.zeros((requests, 1), dtype=torch.int32, device=device)
                for case in cases:
                    case.storage.backing.copy_(initial)
                    case.values.copy_(values)
                    case.cache.fill_(173)
                    case.counts.copy_(torch.tensor(accepted, dtype=torch.int32, device=device))
                    case.metadata.set(starts, length_tensor, torch.tensor(selected, device=device),
                                      pages, tokens, torch.tensor([requests], dtype=torch.int32, device=device))
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                for name in ("separate", "fused") if step % 2 else ("fused", "separate"):
                    ticket = worker.submit(bundle.graph_indices[name])
                    completion = worker.poll(timeout_ms=120000)
                    assert completion is not None and completion.ticket == ticket and completion.error is None
                after = torch.cuda.memory_stats()
                for key in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key]
                control, candidate = cases
                assert torch.equal(control.observed, candidate.observed)
                assert torch.equal(control.storage.backing, candidate.storage.backing)
                expected = initial.clone()
                if mode == "verify":
                    region = candidate.storage.layout.region("window")
                    for request, count in enumerate(accepted):
                        for token in range(count):
                            destination = region.offset + ((int(starts[request]) + token) % 128) * 528
                            source = (128 + token) * 528
                            expected[selected[request], destination:destination + 528].copy_(
                                candidate.observed[request, source:source + 528])
                assert torch.equal(candidate.storage.backing, expected)
                assert candidate.cache.eq(129).all() and candidate.values.eq(37).all()
        finally:
            worker.close()
            for graph in graphs.values():
                graph.reset()
