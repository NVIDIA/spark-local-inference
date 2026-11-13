# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU PLE hashes publish coherent requests for asynchronous resident row delivery."""

import os
import threading
import time
from types import SimpleNamespace

import pytest
import torch
from b12x.preparation import PreparationSession
from b12x.sequence import ple_embedding
from b12x.sequence._shared.disk_table import MappedHostAllocation
from b12x.sequence.ple_embedding.reference import lookup as reference
from b12x.sequence.ple_hash.reference import ple_hash_packed_reference
from cuda.bindings import driver

from ignition.components.ngram_delivery import PreparedPLELookup
from ignition.components.readiness import ReadinessTimings
from ignition.recipes.qwen38.ple import PLEMetadata

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an assigned CUDA device")


class Forward:
    def __init__(self, lookup, progress):
        self.lookup, self.progress = lookup, progress

    def __call__(self):
        self.lookup.publish()
        self.progress.add_(1)
        self.lookup(self.progress)


@pytest.mark.parametrize("tokens", [1, 24, 4096])
@torch.no_grad()
def test_gpu_requests_feed_resident_ple_rows(tmp_path, tokens):
    from ignition.preparation.compile_pool import CompilePool

    device = torch.device("cuda:0")
    caps = ple_embedding.Caps(device=device, max_tokens=tokens, max_seqs=1,
        vocab_size=256, eos_token_id=2, max_order=3, heads_per_order=8,
        dense_layer_ordinal=0, base_table_size=101, embedding_dim=2560, tp_size=1, tp_rank=0,
        quant_mode="nvfp4_group16", table_memory="io_uring")
    layout = ple_embedding.storage_layout(caps)
    rows, heads = layout.geometry.padded_vocab_size, caps.head_count
    weight = torch.arange(rows * 80).remainder(256).byte().view(rows, 80)
    scales = torch.arange(rows * 10).remainder(5).add(1).float().to(torch.float8_e4m3fn).view(rows, 10)
    mailbox = MappedHostAllocation((tokens * heads + 1,), torch.int64, device)
    flag = MappedHostAllocation((1,), torch.int32, device)
    progress = MappedHostAllocation((1,), torch.int32, device)
    for region in (mailbox, flag, progress):
        region.host_view.zero_()
    ready = torch.cuda.Event(external=True)
    ready.record()
    table = SimpleNamespace(layout=layout, geometry=ple_embedding.allocate_geometry(layout.geometry, device=device),
        hashes=torch.full((tokens, heads), -1, dtype=torch.int64, device=device),
        weights=torch.zeros((tokens * heads, 80), dtype=torch.uint8, device=device),
        scales=torch.zeros((tokens * heads, 10), dtype=torch.uint8, device=device),
        scale=torch.tensor([.25], dtype=torch.float32, device=device), mailbox=mailbox, ready=ready,
        sequence=torch.zeros(1, dtype=torch.int64, device=device))
    metadata = PLEMetadata.allocate(tokens, 1, device, eos_token_id=2)
    frame = torch.zeros(1, dtype=torch.int32, device=device)
    table.timings = ReadinessTimings.allocate(frame, 1)
    output = torch.empty((tokens, 2560), dtype=torch.bfloat16, device=device)
    staging_w = torch.empty_like(table.weights, device="cpu", pin_memory=True)
    staging_s = torch.empty_like(table.scales, device="cpu", pin_memory=True)
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        lookup = PreparedPLELookup(table, metadata, output, caps, session=session, name="ple")
        forward = Forward(lookup, progress.device_view)
        artifact = tmp_path / "compiled.bin"
        with CompilePool(device) as pool:
            pool.submit(forward, artifact).result()
        with artifact.open("rb") as source:
            compiled = torch.compiler.load_compiled_function(source)
        compiled(forward)
        torch.cuda.synchronize()
        compute, copies = torch.cuda.Stream(), torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph, stream=compute):
            compiled(forward)
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        for generation, live in enumerate((tokens, max(1, tokens // 2), 0), 1):
            frame.fill_(generation % 2)
            ids = (torch.arange(tokens) * 17 + generation) % 256
            metadata.token_ids.copy_(ids)
            metadata.query_start_loc.copy_(torch.tensor([0, live], dtype=torch.int32))
            metadata.num_seqs.fill_(int(live > 0))
            metadata.num_tokens.fill_(live)
            table.sequence.fill_(generation)
            progress.host_view.zero_()
            output.fill_(float("nan"))
            torch.cuda.synchronize()
            errors = []
            actual_ids = []

            def gather():
                try:
                    deadline = time.monotonic() + 10
                    while mailbox.host_view[-1].item() != generation:
                        if time.monotonic() > deadline:
                            raise RuntimeError("GPU did not publish PLE requests")
                        time.sleep(.0001)
                    requested = mailbox.host_view[:-1].clone().reshape(tokens, heads)
                    actual_ids.append(requested)
                    time.sleep(.01)
                    assert progress.host_view.item() == 1
                    safe = requested.flatten().clamp_min(0)
                    staging_w.copy_(weight[safe])
                    staging_s.copy_(scales.view(torch.uint8)[safe])
                except BaseException as error:
                    errors.append(error)
                finally:
                    flag.host_view.fill_(generation)

            with torch.cuda.stream(copies):
                status, = driver.cuStreamWaitValue32(driver.CUstream(copies.cuda_stream),
                    flag.device_view.data_ptr(), generation, driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ)
                assert int(status) == 0
                table.weights.copy_(staging_w, non_blocking=True)
                table.scales.copy_(staging_s, non_blocking=True)
                ready.record(copies)
            reader = threading.Thread(target=gather)
            reader.start()
            with torch.cuda.stream(compute):
                graph.replay()
            compute.synchronize()
            reader.join()
            if errors:
                raise errors[0]
            assert torch.cuda.memory_allocated() == allocated
            before, ready_at, synchronized = table.timings.timestamps.cpu()[generation % 2].tolist()
            assert before < ready_at <= synchronized
            assert 8_000_000 <= ready_at - before < 1_000_000_000
            expected_ids = ple_hash_packed_reference(ids[:live], torch.tensor([0, live], dtype=torch.int32),
                metadata.committed_history.cpu(), eos_token_id=2,
                multipliers=table.geometry.multipliers.cpu(), prime_sizes=table.geometry.prime_sizes.cpu(),
                table_offsets=table.geometry.table_offsets.cpu(), heads_per_order=8)
            torch.testing.assert_close(actual_ids[0][:live], expected_ids, atol=0, rtol=0)
            assert bool((actual_ids[0][live:] == -1).all())
            expected = reference(weight, scales, actual_ids[0], quant_mode="nvfp4_group16",
                weight_scale_2=table.scale.cpu(), shard_start=0, embedding_dim=2560, num_tokens=live)
            torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)
        graph.reset()
    for region in (mailbox, flag, progress):
        region.close()
