# SPDX-License-Identifier: Apache-2.0
"""Index selection finishes before latent-history production reuses its bytes."""

import os

import pytest
import torch

from ignition.recipes.kimi3.bootstrap import allocate_prefill_histories


def test_history_views_share_one_complete_storage_after_worker_serialization():
    from ignition.preparation.forward_metadata import dumps, loads

    latent, index = allocate_prefill_histories(1048576, "meta")
    assert latent.shape == (1048576, 576) and index.shape == (16384, 8448)
    assert latent.untyped_storage()._cdata == index.untyped_storage()._cdata
    assert latent.untyped_storage().nbytes() == index.untyped_storage().nbytes() == 576 << 20
    latent, index = loads(dumps((latent, index)))
    assert latent.untyped_storage()._cdata == index.untyped_storage()._cdata
    assert latent.untyped_storage().nbytes() == 576 << 20


class _HistoryCopies:
    def __init__(self, latent, index, index_history, latent_history):
        self.latent, self.index = latent, index
        self.index_history, self.latent_history = index_history, latent_history
        self.saved_index = torch.empty_like(index)
        self.saved_latent = torch.empty_like(latent)

    def __call__(self):
        self.index_history()
        # A material read models the consumer that must finish before overwrite.
        self.saved_index.copy_(self.index)
        self.latent_history()
        self.saved_latent.copy_(self.latent)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                    reason="requires an explicitly assigned CUDA GPU")
@torch.no_grad()
def test_compiled_serial_history_copies_preserve_bytes_with_changed_pages_and_tails():
    from ignition.components.interleaved_history import PreparedInterleavedHistory
    from ignition.components.planar_index_history import PreparedPlanarIndexHistory
    from ignition.preparation.compilation import compile_forward

    device = torch.device("cuda:0")
    maximum, pages = 4096, 36
    latent, index = allocate_prefill_histories(maximum, device)
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
    forward = _HistoryCopies(latent, index, index_history, latent_history)
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
            latent.view(torch.uint8).fill_(173)
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
            assert torch.equal(forward.saved_index.cpu(), expected_index)
            expected_latent = torch.full((maximum, 576), 173, dtype=torch.uint8)
            expected_latent.view(-1)[:expected_index.numel()].copy_(expected_index.view(-1))
            expected_latent[:count].copy_(source_latent[mapping].view(maximum, 576)[:count])
            assert torch.equal(forward.saved_latent.view(torch.uint8).cpu(), expected_latent)
    finally:
        graph.reset()
