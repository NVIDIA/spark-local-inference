# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""V4 footer-aware window transfers and orthogonal index rotation."""

import os

import pytest
import torch

from ignition.components.paged_batch import Metadata
from ignition.recipes.deepseek4.attention import rotate_index, transfer_window
from ignition.recipes.deepseek4.pool import PAGE_TOKENS, latent_page_bytes

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned CUDA device"
)


@torch.no_grad()
def test_compressed_indices_expand_both_c128_records_across_page_boundaries():
    from ignition.recipes.deepseek4.attention import AttentionIndices

    device = torch.device("cuda", 0)
    metadata = Metadata.allocate(6, 1, 2, device)
    metadata.page_table.copy_(torch.tensor([[3, 1]], device=device))
    metadata.state_rows.fill_(1)
    metadata.positions.copy_(torch.tensor([127, 128, 255, 256, 511, -1], device=device))
    indices = AttentionIndices(metadata, mode="verify", tokens=6, visible=None)
    indices.compressed(4)
    indices.compressed(128)
    dense = torch.empty((6, 4), dtype=torch.int32, device=device)
    indices.dense_pages(dense)
    indices.functional()
    packet = indices(metadata)
    c4, c128, dense = packet.ratios[4], packet.ratios[128], packet.dense
    assert c4.slots.tolist() == [223, 224, 255, 64, 127, -1]
    assert c128.slots.tolist() == [6, 7, 7, 2, 3, -1]
    assert c128.lengths.tolist() == [1, 1, 2, 2, 4, 0]
    assert dense.tolist() == [[6, 7, 2, 3]] * 6


@pytest.mark.parametrize("block,heads", ((1, 16), (3, 32), (5, 16), (5, 32)))
@torch.no_grad()
def test_draft_attention_modes_preserve_noncausal_visibility_and_request_isolation(block, heads):
    from b12x.attention._shared.mla.compressed_reference import compressed_sparse_mla_reference
    from b12x.preparation import PreparationSession

    from ignition.components.compressed_mla import PreparedCompressedCacheWriter, PreparedCompressedMLA
    from ignition.preparation.compilation import compile_forward
    from ignition.recipes.deepseek4.attention import AttentionIndices

    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(4150 + block + heads)
    rows, span = 3, 128 + block
    values = torch.randn((rows * span, 512), device=device, dtype=torch.bfloat16) * 0.2
    backing = torch.full((rows, latent_page_bytes(span) + 64), 19, dtype=torch.uint8, device=device)
    cache = backing[:, :latent_page_bytes(span)]
    slots = torch.arange(rows * span, device=device)
    query = torch.randn((rows * block, heads, 512), device=device, dtype=torch.bfloat16) * 0.2
    sink = torch.randn(heads, device=device)
    metadata = Metadata.allocate(rows * block, rows, 1, device)
    metadata.starts.copy_(torch.tensor([256, 7, 0], device=device))
    metadata.lengths.copy_(torch.tensor([block, block, 0], device=device))
    metadata.requests.copy_(torch.arange(rows, device=device).repeat_interleave(block))
    metadata.positions[:2 * block].copy_(
        (metadata.starts[:2, None] + torch.arange(block, device=device)).flatten()
    )
    indices = AttentionIndices(metadata, mode="draft", tokens=rows * block, visible=None)
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        writer = PreparedCompressedCacheWriter(values, cache, slots, page_size=span, kind="swa",
            cache_format="deepseek_v4", session=session, name="draft_write")
        outputs = {mode: torch.empty_like(query) for mode in ("extend", "decode")}
        attentions = [PreparedCompressedMLA(query, cache, indices.window_indices, indices.window_lengths,
            sink, output, window_page_size=span, mode=mode, cache_format="deepseek_v4", session=session,
            name=mode, workspace=None).functional(session=session) for mode, output in outputs.items()]
        indices.functional()

        def forward():
            packet = indices(metadata)
            writer(values)
            for attention, output in zip(attentions, outputs.values(), strict=True):
                output.copy_(attention(query, packet.window_indices, packet.window_lengths))

        compiled = compile_forward(forward)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()

        def replay():
            for output in outputs.values():
                output.fill_(float("nan"))
            graph.replay()
            packet = indices(metadata)
            expected = compressed_sparse_mla_reference(query, cache, packet.window_indices,
                packet.window_lengths, sm_scale=512**-0.5, attn_sink=sink, swa_page_size=span,
                cache_format="deepseek_v4")
            for output in outputs.values():
                torch.testing.assert_close(output.float(), expected.float(), atol=0.012, rtol=0.08)
            assert (backing[:, latent_page_bytes(span):] == 19).all()
            return {mode: output.clone() for mode, output in outputs.items()}

        before = replay()
        values[128 + block - 1].add_(2)
        after = replay()
        for mode in outputs:
            assert not torch.equal(before[mode][0], after[mode][0])
            torch.testing.assert_close(before[mode][block:], after[mode][block:], atol=0, rtol=0)
        values[span:2 * span].add_(3)
        isolated = replay()
        for mode in outputs:
            torch.testing.assert_close(after[mode][:block], isolated[mode][:block], atol=0, rtol=0)
        graph.reset()


@pytest.mark.parametrize("size,accepted", ((1, 1), (6, 0), (6, 3), (129, 127), (4096, 4096)))
@torch.inference_mode()
def test_window_roundtrip_preserves_payload_footer_and_padding(size, accepted):
    device = torch.device("cuda", torch.cuda.current_device())
    stride = latent_page_bytes(128) + 64
    storage = torch.randint(0, 256, (4, stride), dtype=torch.uint8, device=device)
    window = storage[:, : latent_page_bytes(128)]
    before = storage.clone()
    extent = 128 + size
    staged = torch.zeros((2, latent_page_bytes(extent)), dtype=torch.uint8, device=device)
    m = Metadata.allocate(2 * size, 2, 1, device)
    m.state_rows.copy_(torch.tensor([2, 0], device=device))
    m.starts.copy_(torch.tensor([127, 99], device=device))
    transfer_window(window, staged, m, extent)
    torch.testing.assert_close(staged[0, : 128 * 576], window[2, : 128 * 576])
    torch.testing.assert_close(
        staged[0, extent * 576 : extent * 576 + 128 * 8], window[2, 128 * 576 : 128 * 584]
    )
    new_values = torch.randint(0, 256, (size, 584), dtype=torch.uint8, device=device)
    staged[0, 128 * 576 : extent * 576].copy_(new_values[:, :576].flatten())
    staged[0, extent * 576 + 128 * 8 : extent * 584].copy_(new_values[:, 576:].flatten())
    counts = torch.tensor([accepted, 0], dtype=torch.int32, device=device)
    transfer_window(staged, window, m, extent, counts=counts)
    for token in range(accepted):
        slot = (127 + token) % 128
        before[2, slot * 576 : (slot + 1) * 576] = new_values[token, :576]
        before[2, 128 * 576 + slot * 8 : 128 * 576 + (slot + 1) * 8] = new_values[token, 576:]
    torch.testing.assert_close(storage, before, atol=0, rtol=0)


@torch.inference_mode()
def test_index_rotation_matches_normalized_sylvester_matrix():
    device = torch.device("cuda", torch.cuda.current_device())
    matrix = torch.ones(1, 1)
    for _ in range(7):
        matrix = torch.cat((torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0)
    source = torch.randn((3, 64, 128), dtype=torch.bfloat16, device=device)
    output = torch.empty_like(source)
    rotate_index(source, output)
    expected = (source.float() @ matrix.to(device) * 128**-0.5).bfloat16()
    torch.testing.assert_close(output, expected, atol=0.001, rtol=0.008)


@pytest.mark.parametrize("ratio", (4, 128))
@torch.no_grad()
def test_footer_cache_attention_with_shuffled_pages_and_inactive_rows(ratio):
    from b12x.attention._shared.mla.compressed_reference import (
        compressed_sparse_mla_reference,
    )
    from b12x.preparation import PreparationSession

    from ignition.components.compressed_mla import (
        PreparedCompressedCacheWriter,
        PreparedCompressedMLA,
    )
    from ignition.preparation.compilation import compile_forward

    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(4100 + ratio)
    records = PAGE_TOKENS // ratio
    window_values = torch.randn((128, 512), device=device, dtype=torch.bfloat16) * 0.2
    indexed_values = torch.randn((4 * records, 512), device=device, dtype=torch.bfloat16) * 0.2
    query = torch.randn((4, 16, 512), device=device, dtype=torch.bfloat16) * 0.2
    backing = torch.full((3, latent_page_bytes(128) + 128), 19, dtype=torch.uint8, device=device)
    window = backing[:, : latent_page_bytes(128)]
    indexed = torch.zeros((4, latent_page_bytes(records)), dtype=torch.uint8, device=device)
    window_slots = torch.arange(128, device=device) + 128
    permutation = torch.tensor([3, 1, 0, 2], device=device)
    indexed_slots = (permutation[:, None] * records + torch.arange(records, device=device)).flatten()
    window_indices = window_slots.int().repeat(4, 1)
    indexed_indices = indexed_slots.int().repeat(4, 1)
    lengths = torch.tensor([128, 83, 1, 0], dtype=torch.int32, device=device)
    indexed_lengths = torch.tensor([4 * records, 2 * records, 1, 0], dtype=torch.int32, device=device)
    sink = torch.randn(16, device=device)
    output = torch.empty_like(query)
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        write_window = PreparedCompressedCacheWriter(
            window_values,
            window,
            window_slots,
            page_size=128,
            kind="swa",
            cache_format="deepseek_v4",
            session=session,
            name="window",
        )
        write_indexed = PreparedCompressedCacheWriter(
            indexed_values,
            indexed,
            indexed_slots,
            page_size=records,
            kind="indexed",
            cache_format="deepseek_v4",
            session=session,
            name="indexed",
        )
        attention = PreparedCompressedMLA(
            query,
            window,
            window_indices,
            lengths,
            sink,
            output,
            indexed=indexed,
            indexed_indices=indexed_indices,
            indexed_lengths=indexed_lengths,
            indexed_page_size=records,
            cache_format="deepseek_v4",
            session=session,
            name="attention",
            workspace=None,
        )

        def forward():
            write_window(window_values)
            write_indexed(indexed_values)
            attention()

        compiled = compile_forward(forward)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        for _ in range(3):
            window_values.normal_(std=0.2)
            indexed_values.normal_(std=0.2)
            output.fill_(float("nan"))
            graph.replay()
            expected = compressed_sparse_mla_reference(
                query,
                window,
                window_indices,
                lengths,
                sm_scale=512**-0.5,
                attn_sink=sink,
                extra_k_cache=indexed,
                extra_indices=indexed_indices,
                extra_topk_lengths=indexed_lengths,
                swa_page_size=128,
                extra_page_size=records,
                cache_format="deepseek_v4",
            )
            torch.testing.assert_close(output.float(), expected.float(), atol=0.012, rtol=0.08)
            assert (backing[:, latent_page_bytes(128) :] == 19).all()
            assert (window[0] == 19).all() and (window[2] == 19).all()
        graph.reset()


@pytest.mark.parametrize("layer", (2, 3))
@pytest.mark.parametrize("overlap", (False, True))
@torch.no_grad()
def test_checkpoint_attention_alternates_prefill_decode_and_rejected_verification(
    layer, overlap, tmp_path, monkeypatch
):
    from pathlib import Path

    from b12x.preparation import PreparationSession

    from ignition.components.parallel import TensorParallelShard
    from ignition.components.rotary import CompressedRotary as Rotary
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.recipes.deepseek4.attention import AttentionScratch, PreparedAttention
    from ignition.recipes.deepseek4.config import REVISION
    from ignition.recipes.deepseek4.pool import ContinuationPool
    from ignition.recipes.deepseek4.weights import DeepSeekWeightLoader

    snapshot = Path.home() / (
        ".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-Vision-Exp/snapshots/" + REVISION
    )
    if not (snapshot / "model.safetensors.index.json").is_file():
        pytest.skip("DeepSeek V4 Flash Vision checkpoint is unavailable")
    device = torch.device("cuda", torch.cuda.current_device())
    monkeypatch.setattr("ignition.preparation.compilation.DEFAULT_CACHE_ROOT", tmp_path / "compiled")
    torch.manual_seed(4100 + layer)
    with DeepSeekWeightLoader(
        inspect_indexed_checkpoint(snapshot),
        device,
        shard=TensorParallelShard(0, 4),
        prefixes=(f"layers.{layer}.attn.",),
    ) as loader:
        config, weights = loader.config, loader.load_attention(layer)
    pool = ContinuationPool.allocate(config, 3, 4, device)
    arena, rotary = ScratchArena(device, overlap=overlap), Rotary(config, device)
    # The three graphs share their private buffers, as serving graphs on one rank do.
    source = torch.randn((256, 4096), dtype=torch.bfloat16, device=device)
    variants = {}
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        pending = []
        for mode, capacity in (("prefill", 256), ("decode", 1), ("verify", 6)):
            inputs = torch.zeros((capacity, 4096), dtype=torch.bfloat16, device=device)
            metadata = Metadata.allocate(capacity, 1, 8, device)
            scratch = AttentionScratch.allocate(capacity, 4, device, blocks=8)
            counts = torch.zeros(1, dtype=torch.int32, device=device)
            attention = PreparedAttention(
                layer,
                weights,
                inputs,
                scratch,
                metadata,
                pool,
                rotary,
                mode=mode,
                world=4,
                session=session,
                name=mode,
                workspace=arena,
            )
            if not overlap:
                assert attention.concurrent_compression is None
                assert attention.concurrent_index is None

            def forward(attention=attention, inputs=inputs, mode=mode, counts=counts, metadata=metadata, scratch=scratch):
                output, records = attention(inputs, metadata)
                scratch.output.copy_(output)
                if mode == "verify":
                    attention.commit(counts, records, metadata)

            pending.append((mode, inputs, metadata, scratch, counts, forward))
        arena.finalize()
        arena.freeze()
        for mode, inputs, metadata, scratch, counts, forward in pending:
            compiled = compile_forward(forward)
            compiled()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                compiled()
            variants[mode] = inputs, metadata, scratch, counts, graph

        def run(mode, start, size, accepted, *, row=1, rejected=False):
            inputs, m, scratch, counts, graph = variants[mode]
            inputs.zero_()
            inputs[:size].copy_(source[start : start + size])
            if rejected:
                inputs.normal_()
            m.starts.fill_(start)
            m.state_rows.fill_(row)
            m.lengths.fill_(size)
            m.query_start_loc[1] = size
            m.positions.fill_(-1)
            m.positions[:size].copy_(torch.arange(start, start + size, device=device))
            m.page_table.fill_(-1)
            m.page_table[0, :2].copy_(torch.tensor([3, 1], device=device))
            m.live_counts.copy_(torch.tensor([size, 1], device=device))
            counts.fill_(accepted)
            arena.tensor.fill_(0x7F if row == 1 else 0x33)
            graph.replay()
            return scratch.output[:size].clone()

        # Compare identical arithmetic paths; b12x can choose different activation
        # precision for prefill and small verification GEMMs.
        pool.clear_for_preparation()
        first = run("prefill", 0, 127, 127)
        expected = {127: run("verify", 127, 6, 2)[:2], 129: run("decode", 129, 1, 1)}
        for start in range(130, 256, 6):
            size = min(6, 256 - start)
            expected[start] = run("verify", start, size, size)
        state = pool.backing[1].clone()
        pages = {name: value.clone() for name, value in pool.pages.items()}
        pool.clear_for_preparation()
        torch.testing.assert_close(run("prefill", 0, 127, 127), first, atol=0, rtol=0)
        continuation = pool.backing[1].clone()
        run("verify", 127, 6, 0, rejected=True)
        torch.testing.assert_close(pool.backing[1], continuation, atol=0, rtol=0)
        actual = run("verify", 127, 6, 2)
        torch.testing.assert_close(actual[:2], expected[127], atol=0, rtol=0)
        actual = run("decode", 129, 1, 1)
        torch.testing.assert_close(actual, expected[129], atol=0, rtol=0)
        pool.backing[2].copy_(pool.backing[1])
        for start in range(130, 256, 6):
            size = min(6, 256 - start)
            actual = run("verify", start, size, size, row=2)
            torch.testing.assert_close(actual, expected[start], atol=0, rtol=0)
        torch.testing.assert_close(pool.backing[2], state, atol=0, rtol=0)
        for name, value in pool.pages.items():
            torch.testing.assert_close(value, pages[name], atol=0, rtol=0)
        for *_, graph in variants.values():
            graph.reset()
