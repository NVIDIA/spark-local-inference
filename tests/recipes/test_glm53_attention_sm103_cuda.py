# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM 5.3 sparse attention on SM103: FlashInfer index selection, Triton and Cake sparse MLA against b12x references."""

from __future__ import annotations

import os
import types

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_CUDA_TESTS=1 and an explicitly assigned CUDA_VISIBLE_DEVICES",
)

HEADS, WIDTH, SCALE = 64, 2051, 512 ** -0.5 * 1.1
PAGES = 40


@pytest.fixture(scope="module")
def cache():
    from b12x.attention._shared.mla.reference import pack_mla_kv_cache_reference

    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.fail("assign a CUDA device explicitly")
    if torch.cuda.get_device_capability(0) != (10, 3):
        pytest.skip("SM103 sparse attention requires a GB300/B300")
    torch.manual_seed(0)
    return pack_mla_kv_cache_reference(torch.randn(PAGES * 256, 512, device="cuda") * 2).view(-1, 64, 528)


def _prepared(factory):
    """Construct an operator in a standalone preparation session, as a model's batch would."""
    from b12x.preparation import PreparationSession

    with PreparationSession(device=torch.device("cuda"), autotune=True, compile_workers=0) as session:
        operator = factory(session)
        session.freeze()
    return operator


def _relative(output, reference):
    return ((output.float() - reference).norm() / reference.norm()).item()


def _reference(query, cache, selected, counts):
    from b12x.attention._shared.mla.reference import sparse_mla_reference

    return sparse_mla_reference(q_all=query, kv_cache=cache.view(-1, 528), page_table_1=selected,
                                active_token_counts=counts, sm_scale=SCALE, v_head_dim=512).float()


@pytest.mark.parametrize("rows", [1, 3, 8, 64, 200])
def test_decode_attention_matches_reference(cache, rows):
    from ignition.components.sparse_mla_sm103 import TRITON_ROWS, Sm103SparseMLA

    query = (torch.randn(rows, HEADS, 512, device="cuda") * 0.6).to(torch.bfloat16)
    selected = torch.stack([torch.randperm(PAGES * 256, device="cuda")[:WIDTH] for _ in range(rows)]).int()
    selected[:, 7::11] = -1
    counts = torch.randint(0, WIDTH + 1, (rows,), device="cuda", dtype=torch.int32)
    output = torch.full_like(query, 3.0)
    attention = _prepared(lambda session: Sm103SparseMLA(query, cache, selected, None, counts, output,
        caps=types.SimpleNamespace(softmax_scale=SCALE), session=session, name=f"decode{rows}", workspace=None))
    assert (attention.cake is None) == (rows <= TRITON_ROWS)
    graph = torch.cuda.CUDAGraph()
    attention()
    with torch.cuda.graph(graph):
        attention()
    output.fill_(3.0)
    graph.replay()
    torch.cuda.synchronize()
    assert _relative(output, _reference(query, cache, selected, counts)) < 6e-3


def test_prefill_slab_attention_matches_physical_selection(cache):
    from b12x.attention import sparse_mla

    from ignition.components.sparse_mla_sm103 import Sm103PrefillSparseMLA
    from ignition.recipes.glm53.dcp import expand_tokens

    table = torch.randperm(PAGES, device="cuda").int()[:32]
    context, rows = 5000, 2500
    positions = torch.arange(context - rows, context, device="cuda")
    pools = torch.full((rows, 512), -1, dtype=torch.int32, device="cuda")
    for row in range(rows):
        complete = (int(positions[row]) + 1) // 4
        pools[row, :min(512, complete)] = torch.randperm(complete, device="cuda")[:512].int()
    logical = torch.empty((rows, WIDTH), dtype=torch.int32, device="cuda")
    counts = torch.empty(rows, dtype=torch.int32, device="cuda")
    expand_tokens(pools, positions, logical, counts)
    physical, physical_counts = torch.empty_like(logical), torch.empty_like(counts)
    sparse_mla.expand_pooled_topk_to_physical_slots(pools, positions, torch.zeros_like(counts), table.view(1, -1),
        physical, physical_counts, pool_size=4, block_size=256, block_stride_rows=256, num_cache_blocks=PAGES * 4,
        plan=sparse_mla.plan_pooled_selection(device=cache.device, max_rows=rows, page_size=256,
            max_page_table_width=table.numel(), num_cache_blocks=PAGES * 4))
    assert torch.equal(counts, physical_counts)
    translated = torch.where(logical >= 0, table.long()[logical.clamp(min=0).long() // 256] * 256 + logical % 256, -1)
    for row in range(0, rows, 97):
        assert set(translated[row, :counts[row]].tolist()) == set(physical[row, :counts[row]].tolist())

    query = (torch.randn(rows, HEADS, 512, device="cuda") * 0.6).to(torch.bfloat16)
    output = torch.full_like(query, 3.0)
    end = torch.tensor([context], dtype=torch.int64, device="cuda")
    attention = _prepared(lambda session: Sm103PrefillSparseMLA(query, cache, logical, counts, table, end, output,
        caps=types.SimpleNamespace(softmax_scale=SCALE), session=session, name="prefill", workspace=None))
    attention()
    torch.cuda.synchronize()
    assert _relative(output, _reference(query, cache, physical, physical_counts)) < 6e-3


def test_index_selection_matches_exact_top_k(cache):
    from b12x.attention.dsa_indexer.reference import pack_index_k_cache_reference, paged_decode_logits_reference

    from ignition.components.indexer_sm103 import Sm103SparseIndexer

    rows, heads, pages_per_row, topk = 6, 32, 64, 512
    total = rows * pages_per_row + 3
    index_cache = pack_index_k_cache_reference(torch.randn(total * 64, 128, device="cuda"),
                                               page_size=64).view(total, 64 * 132)
    query = (torch.randn(rows, heads, 128, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    weights = torch.rand(rows, heads, device="cuda")
    table = torch.randperm(total, device="cuda")[:rows * pages_per_row].view(rows, pages_per_row).int()
    lengths = torch.tensor([4096, 3000, 512, 300, 1, 2049], device="cuda", dtype=torch.int32)
    output = torch.full((rows, topk), -7, dtype=torch.int32, device="cuda")
    selector = Sm103SparseIndexer(query, weights, index_cache, table, lengths, None, output, caps=None,
                                  session=None, name="select", workspace=None)
    selector()
    torch.cuda.synchronize()
    logits = paged_decode_logits_reference(q_fp8=query, weights=weights, index_k_cache=index_cache,
        real_page_table=table, query_row_to_batch=torch.arange(rows, device="cuda"), seqlens_per_query=lengths,
        page_size=64)
    for row in range(rows):
        count = int(lengths[row])
        expected = set(torch.topk(logits[row, :count], min(topk, count)).indices.tolist())
        selected = output[row].tolist()
        assert {index for index in selected if index >= 0} == expected
        assert sum(index < 0 for index in selected) == topk - len(expected)
