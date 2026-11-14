# SPDX-License-Identifier: Apache-2.0
"""Compiled history copies preserve active prefixes with invocation-local outputs."""

import os

import pytest
import torch

class _HistoryCopies:
    def __init__(self, latent, index, index_history, latent_history, table, end):
        self.index_history, self.latent_history = index_history, latent_history
        self.table, self.end = table, end
        self.saved_index = torch.empty_like(index)
        self.saved_latent = torch.empty_like(latent)

    def __call__(self):
        self.saved_index.copy_(self.index_history(self.table, self.end))
        self.saved_latent.copy_(self.latent_history(self.table, self.end))


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                    reason="requires an explicitly assigned CUDA GPU")
@torch.no_grad()
def test_compiled_serial_history_copies_preserve_bytes_with_changed_pages_and_tails():
    from ignition.components.interleaved_history import PreparedInterleavedHistory
    from ignition.components.planar_index_history import PreparedPlanarIndexHistory
    from ignition.preparation.compilation import compile_forward

    device = torch.device("cuda:0")
    maximum, pages = 4096, 36
    latent = torch.empty((maximum, 576), dtype=torch.uint8, device=device)
    index = torch.empty((maximum // 64, 8448), dtype=torch.uint8, device=device)
    latent_cache = torch.randint(0, 256, (pages, 128, 576), dtype=torch.uint8, device=device)
    index_cache = torch.randint(0, 256, (2 * pages, 8448), dtype=torch.uint8, device=device)
    table = torch.arange(maximum // 128, dtype=torch.int32, device=device)
    end = torch.zeros(1, dtype=torch.int64, device=device)
    rank = torch.zeros_like(end)
    latent_stage = torch.empty((maximum, 576), dtype=torch.uint8, device=device)
    index_stage = torch.empty_like(index)
    latent_addresses = torch.tensor([latent_stage.data_ptr()], dtype=torch.int64, device=device)
    index_addresses = torch.tensor([index_stage.data_ptr()], dtype=torch.int64, device=device)

    def signals():
        return tuple(torch.zeros((1, 8), dtype=torch.bfloat16, device=device) for _ in range(4))

    index_history = PreparedPlanarIndexHistory(index_cache, table, end, rank, index_stage,
        index_addresses, index, signals(), runtime=None, session=None, coordinator=None, name="index")
    latent_history = PreparedInterleavedHistory(latent_cache, table, end, rank, latent_stage,
        latent_addresses, latent.view(torch.uint8), signals(), runtime=None, session=None,
        coordinator=None, name="latent")
    index_history.functional()
    latent_history.functional()
    forward = _HistoryCopies(latent, index, index_history, latent_history, table, end)
    compiled = compile_forward(forward)
    compiled()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    source_index, source_latent = index_cache.cpu(), latent_cache.cpu()
    try:
        for step, count in enumerate((4096, 1, 63, 129, 2048, 0, 4096)):
            mapping = (torch.arange(maximum // 128) * 5 + step) % pages
            table.copy_(mapping)
            end.fill_(count)
            expected_index = torch.full_like(source_index[:maximum // 64], 173)
            for token in range(count):
                src = source_index[2 * mapping[token // 128] + token % 128 // 64]
                dst = expected_index[token // 64]
                row = token % 64
                dst[row * 128:(row + 1) * 128].copy_(src[row * 128:(row + 1) * 128])
                dst[8192 + row * 4:8196 + row * 4].copy_(src[8192 + row * 4:8196 + row * 4])
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            observed_index = forward.saved_index.cpu()
            for token in range(count):
                page, row = divmod(token, 64)
                assert torch.equal(observed_index[page, row * 128:(row + 1) * 128],
                                   expected_index[page, row * 128:(row + 1) * 128])
                assert torch.equal(observed_index[page, 8192 + row * 4:8196 + row * 4],
                                   expected_index[page, 8192 + row * 4:8196 + row * 4])
            expected_latent = torch.full((maximum, 576), 173, dtype=torch.uint8)
            expected_latent[:count].copy_(source_latent[mapping].view(maximum, 576)[:count])
            assert torch.equal(forward.saved_latent.view(torch.uint8).cpu()[:count], expected_latent[:count])
    finally:
        graph.reset()
