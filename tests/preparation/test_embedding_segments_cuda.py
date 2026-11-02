# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Native replay delivers canonical full-table rows from sealed resident planes."""

import os
import time

import pytest
import torch
import flare_heliopause
from b12x.preparation import PreparationSession
from b12x.sequence import engram
from b12x.sequence._shared.disk_table import MappedHostAllocation

from ignition.components.engram import EngramInputs, PreparedEngramLookup
from ignition.components.ngram_delivery import publish_requests
from ignition.components.parallel import TensorParallelShard
from ignition.preparation.compilation import compile_forward
from ignition.preparation.execution import GraphSignature, PreparedExecution
from ignition.preparation.runtime_package import FileSource
from ignition.recipes.deepseek41.engram import Tables


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("rank", range(4))
@torch.no_grad()
def test_native_replay_reads_full_tables_without_checkpoint_files(tmp_path, rank):
    device = torch.device("cuda:0")
    geometry = engram.build_geometry(layer_ids=(1,), base_table_size=101, compressed_vocab_size=32)
    rows, tokens = geometry.num_embeddings[0], 4
    weights = ((torch.arange(rows * 256).reshape(rows, 256) % 15) - 7).float().to(torch.float8_e4m3fn)
    scales = (torch.arange(rows * 8).reshape(rows, 8) % 5 + 125).byte()
    sources = []
    for name, tensor in (("weights", weights), ("scales", scales)):
        path = tmp_path / name
        path.write_bytes(bytes(37) + tensor.view(torch.uint8).numpy().tobytes())
        sources.append(FileSource(path, 37, tensor.nbytes))
    from ignition.components.readiness import ReadinessTimings
    frame = torch.zeros(1, dtype=torch.int32, device=device)
    inputs = EngramInputs.allocate(tokens, device, timings=ReadinessTimings.allocate(frame, 1))
    output = torch.empty((tokens, 6144), dtype=torch.bfloat16, device=device)
    count = torch.full((1,), tokens, dtype=torch.int32, device=device)
    source = torch.zeros(1, dtype=torch.int32, device=device)
    sequence = torch.zeros(1, dtype=torch.int64, device=device)
    records = torch.zeros((2, 4, 32), dtype=torch.int64, device=device)
    mailbox = MappedHostAllocation((tokens * 24 + 1,), torch.int64, device)
    mailbox.host_view.zero_()
    worker = graph = None
    try:
        with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
            lookup = PreparedEngramLookup(inputs, output, count, layer=1, token_map=tuple(range(32)),
                geometry=geometry, session=session, name="segmented_lookup")

            def forward():
                source.add_(1)
                publish_requests(inputs.hashes, mailbox.device_view, sequence)
                lookup(source)
                records.zero_()
                records[:, 0, 0] = 1
                records[:, 0, 1].copy_(sequence[0].expand(2))
                records[:, 0, 4] = -1
                records[:, 0, 5] = 1

            compiled = compile_forward(forward)
            compiled()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                compiled()
            tables = Tables(geometry, tuple(range(32)), {1: tuple(sources)}, {1: inputs}, {1: mailbox},
                            sequence, TensorParallelShard(rank, 4), None)
            execution = PreparedExecution(device=0, graphs={"decode": graph},
                buffers=dict(tables.buffers(), output=output, live_count=count, frame_index=frame,
                    submission_sequence=sequence, completion_records=records),
                signatures={"decode": GraphSignature("decode", tokens, 1, ("engram_1_hashes", "live_count"))},
                max_active=tokens, resources=(tables, lookup, compiled, session), export_events=tables.events())
            catalog = tables.resident_catalog()
            resident = flare_heliopause.EmbeddingTables(catalog)
            peer = flare_heliopause.EmbeddingTables(catalog, descriptors=resident.descriptors)
            del resident
            for span in sources:
                span.path.unlink()
            manifest = execution.manifest()
            manifest["engram"] = peer.bind(manifest, execution.export_events)
            worker = flare_heliopause.GraphExecutor(execution, manifest)
            for generation, live in enumerate((4, 2, 0, 4), 1):
                canonical = torch.full((tokens, 24), -1, dtype=torch.int64)
                for head, (first, last) in enumerate(zip(geometry.offsets[0], (*geometry.offsets[0][1:], rows), strict=True)):
                    relative = (torch.arange(tokens) * 19 + generation) % (last - first)
                    canonical[:live, head] = first + relative[:live]
                output.fill_(float("nan"))
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
                fence = worker.submit(0, [("engram_1_hashes", 0, canonical.numpy().tobytes()),
                    ("live_count", 0, torch.tensor([live], dtype=torch.int32).numpy().tobytes())])
                deadline = time.monotonic() + 30
                completion = worker.poll()
                while completion is None and time.monotonic() < deadline:
                    time.sleep(.001)
                    completion = worker.poll()
                assert completion is not None and completion.fence == fence
                assert completion.readiness_wait_ns is not None
                assert 0 <= completion.readiness_wait_ns <= max(value for value in completion.gpu_ns if value is not None)
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
                expected = (weights.float()[canonical.clamp_min(0)]
                    * scales.view(torch.float8_e8m0fnu).float()[canonical.clamp_min(0)].repeat_interleave(32, -1))
                expected.masked_fill_(canonical[:, :, None] < 0, 0)
                torch.testing.assert_close(output.cpu(), expected.bfloat16().flatten(1), rtol=0, atol=0)
    finally:
        if worker is not None:
            worker.close()
        if graph is not None:
            graph.reset()
        mailbox.close()
