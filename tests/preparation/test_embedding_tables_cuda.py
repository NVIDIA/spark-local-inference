# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Native TP4 row delivery shares RAM tables and needs no source files during replay."""

import os
from contextlib import ExitStack
from pathlib import Path
import socket
import time

import pytest
import torch
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
                              reason="requires four assigned GPUs and FLARE_RUN_TP_CUDA_TESTS=1")


def _rank(rank, sockets, root):
    from multiprocessing.reduction import recv_handle, send_handle

    import flare_heliopause
    from b12x.preparation import PreparationSession
    from b12x.sequence import engram
    from b12x.sequence._shared.disk_table import MappedHostAllocation
    from ignition.components.engram import EngramInputs, PreparedEngramLookup
    from ignition.components.parallel import TensorParallelShard
    from ignition.components.ngram_delivery import publish_requests
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.execution import GraphSignature, PreparedExecution
    from ignition.preparation.runtime_core import FileSource
    from ignition.recipes.deepseek41.engram import Tables

    peers = []
    for index, (parent, child) in enumerate(sockets, 1):
        if rank == 0:
            peers.append(parent)
            child.close()
        else:
            parent.close()
            if index == rank:
                connection = child
            else:
                child.close()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    geometry = engram.build_geometry(layer_ids=(1,), base_table_size=101, compressed_vocab_size=32)
    rows = geometry.num_embeddings[0]
    weight = ((torch.arange(rows * 256).remainder(15).float() - 7)
              .reshape(rows, 256).to(torch.float8_e4m3fn))
    scale = torch.arange(rows * 8).remainder(5).add(125).byte().reshape(rows, 8)
    sources = (FileSource(Path(root) / "weights", 4093, weight.nbytes),
               FileSource(Path(root) / "scales", 4093, scale.nbytes))
    from ignition.components.readiness import ReadinessTimings
    frame = torch.zeros(1, dtype=torch.int32, device=device)
    inputs = EngramInputs.allocate(4, device, timings=ReadinessTimings.allocate(frame, 1))
    output = torch.empty((4, 6144), dtype=torch.bfloat16, device=device)
    count = torch.full((1,), 4, dtype=torch.int32, device=device)
    source = torch.zeros(1, dtype=torch.int32, device=device)
    sequence = torch.zeros(1, dtype=torch.int64, device=device)
    mailbox = MappedHostAllocation((4 * 24 + 1,), torch.int64, device)
    mailbox.host_view.zero_()
    records = torch.zeros((2, 4, 32), dtype=torch.int64, device=device)
    worker = graph = None
    try:
        with PreparationSession(device=device, autotune=False, compile_workers=2) as session, ExitStack() as native:
            lookup = PreparedEngramLookup(inputs, output, count, layer=1, token_map=tuple(range(32)),
                geometry=geometry, session=session, name="resident_lookup")

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
            tables = Tables(geometry, tuple(range(32)), {1: sources}, {1: inputs}, {1: mailbox}, sequence,
                TensorParallelShard(rank, 4),
                torch.arange(6144, device=device))
            catalog = tables.resident_catalog()
            buffers = dict(tables.buffers(), output=output, live_count=count, frame_index=frame,
                submission_sequence=sequence, completion_records=records)
            # These independent rank lookups do not capture an all-rank readiness reduction.
            buffers.pop("readiness_timestamps")
            execution = PreparedExecution(device=rank, graphs={"decode": graph}, buffers=buffers,
                signatures={"decode": GraphSignature("decode", 4, 1, ("engram_1_hashes", "live_count"))},
                max_active=4, resources=(tables, lookup, compiled, session), export_events=tables.events())
            manifest = execution.manifest()
            if rank == 0:
                resident = flare_heliopause.EmbeddingTables(catalog)
                for peer in peers:
                    for descriptor in resident.descriptors:
                        send_handle(peer, descriptor, 0)
                for peer in peers:
                    assert peer.recv(1) == b"R"
                for source_file in sources:
                    source_file.path.unlink()
            else:
                descriptors = [recv_handle(connection) for _ in range(2)]
                try:
                    resident = flare_heliopause.EmbeddingTables(catalog, descriptors=descriptors)
                finally:
                    for descriptor in descriptors:
                        os.close(descriptor)
                connection.sendall(b"R")
            manifest["engram"] = resident.bind(manifest, execution.export_events)
            execution._resident_owner = resident
            torch.cuda.synchronize()
            if rank:
                flare_heliopause.serve_tensor_parallel_rank(execution, manifest, connection.fileno(), rank, 4)
            else:
                worker = flare_heliopause.GraphExecutor(execution, manifest, peer_fds=[peer.fileno() for peer in peers])
                native.callback(worker.close)
                before = torch.cuda.memory_stats()
                for generation, live in enumerate((4, 2, 0, 4), 1):
                    ids = (torch.arange(4 * 24).reshape(4, 24) * 19 + generation) % rows
                    ids[live:] = -1
                    output.fill_(float("nan"))
                    torch.cuda.synchronize()
                    fence = worker.submit(0, [("engram_1_hashes", 0, ids.numpy().tobytes()),
                        ("live_count", 0, torch.tensor([live], dtype=torch.int32).numpy().tobytes())])
                    completion = None
                    deadline = time.monotonic() + 30
                    while completion is None and time.monotonic() < deadline:
                        completion = worker.poll()
                        if completion is None:
                            time.sleep(.001)
                    assert completion is not None and completion.fence == fence
                    expected = (weight.float()[ids] * scale.view(torch.float8_e8m0fnu).float()[ids]
                                .repeat_interleave(32, dim=-1)).bfloat16().flatten(1)
                    expected[live:].zero_()
                    if not torch.equal(output.cpu(), expected):
                        print({"rank": rank, "generation": generation, "live": live,
                            "sequence": sequence.cpu().tolist(), "hashes": inputs.hashes.cpu()[0, :8].tolist(),
                            "weights": inputs.weights.cpu()[0, :8].tolist(),
                            "scales": inputs.scales.cpu()[0].tolist(),
                            "output": output.cpu()[0, :16].tolist(), "expected": expected[0, :16].tolist()}, flush=True)
                    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
                after = torch.cuda.memory_stats()
                for key in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key]
                worker.close()
                assert worker.drained
                worker = None
            final_generation = int(sequence.cpu().item())
            assert final_generation == 4, (rank, final_generation)
            final_ids = (torch.arange(4 * 24).reshape(4, 24) * 19 + 4) % rows
            expected = (weight.float()[final_ids] * scale.view(torch.float8_e8m0fnu).float()[final_ids]
                        .repeat_interleave(32, dim=-1)).bfloat16().flatten(1)
            torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    finally:
        if worker is not None:
            worker.close()
        if graph is not None:
            graph.reset()
        mailbox.close()
        for pair in sockets:
            for endpoint in pair:
                endpoint.close()


def test_shared_resident_tables_feed_every_native_rank_after_source_files_are_removed(tmp_path):
    from b12x.sequence import engram

    assert torch.cuda.device_count() == 4
    geometry = engram.build_geometry(layer_ids=(1,), base_table_size=101, compressed_vocab_size=32)
    rows = geometry.num_embeddings[0]
    weights = (torch.arange(rows * 256).remainder(15).float() - 7).to(torch.float8_e4m3fn)
    scales = torch.arange(rows * 8).remainder(5).add(125).byte()
    for name, tensor in (("weights", weights), ("scales", scales)):
        (tmp_path / name).write_bytes(bytes(4093) + tensor.view(torch.uint8).numpy().tobytes())
    sockets = [socket.socketpair() for _ in range(3)]
    try:
        mp.spawn(_rank, args=(sockets, str(tmp_path)), nprocs=4, join=True)
    finally:
        for pair in sockets:
            for endpoint in pair:
                endpoint.close()
