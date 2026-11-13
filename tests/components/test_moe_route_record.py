# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""A route record carries selected expert IDs and FP32 route weights between ranks.

Top-k reads one row at a time, so every rank can route its own rows and gather the selection
instead of the 384 logits per row. The native top-k kernel needs a GPU, so these cases replace it
with a row-local reference and compare the sharded decomposition against routing the gathered
logits, which is what the record replaces.
"""

from types import SimpleNamespace

import pytest
import torch

from ignition.components import moe
from ignition.components.moe import RouteRecord, RouterScratch

TOP_K = 6


def _route_topk(plan, logits, selected_logits, ids, weights, *, correction_bias=None,
                image_correction_bias=None, image_mask=None, num_valid_tokens=None):
    """Row-local reference for the native top-k: scores, tie order, and FP32 weights.

    Each row's experts come from that row alone. This reference keeps the lower ID of a tied pair
    and the packed kernel breaks ties toward the larger ID, so the cases below carry no tied
    scores. Rows at or past `num_valid_tokens` receive ID -1 and zero weight, which the b12x route
    pack skips.
    """
    query = plan.query
    scores = logits.float()
    scores = scores.sigmoid() if query.score_func == "sigmoid" else scores.softmax(-1)
    choice = scores if correction_bias is None else scores + correction_bias.float()
    if image_correction_bias is not None and image_mask is not None:
        choice = choice + image_correction_bias.float() * image_mask[:, None]
    picked = torch.argsort(choice, dim=-1, descending=True, stable=True)[:, : query.top_k]
    route = scores.gather(1, picked)
    if query.renormalize:
        route = route / route.sum(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    selected_logits.copy_(choice.gather(1, picked))
    ids.copy_(picked.to(torch.int32))
    weights.copy_(route * query.routed_scaling_factor)
    live = scores.shape[0] if num_valid_tokens is None else int(num_valid_tokens.item())
    if live < scores.shape[0]:
        selected_logits[live:] = 0
        ids[live:] = -1
        weights[live:] = 0


def _prepared(logits, scratch, **options):
    return moe.PreparedRouter(logits, scratch, session=SimpleNamespace(prepare=lambda batch: None),
                              name="routes", **options)


@pytest.fixture(autouse=True)
def reference_routes(monkeypatch):
    monkeypatch.setattr(moe.fused_moe, "route_topk", _route_topk)
    # The native plan identities and the compiler toolchain need a GPU; the geometry and the
    # caller-owned buffers are what these cases check.
    monkeypatch.setattr(moe, "declare", lambda *args, **kwargs: None)


def test_a_router_writes_the_words_the_record_transports():
    record = RouteRecord.allocate(8, TOP_K, "cpu")
    selected = torch.zeros(8, TOP_K)
    scratch = record.router_scratch(selected)
    # The router writes IDs and weights in place, so the bytes it produces are the bytes the
    # collective moves and the consumer unpacks.
    assert scratch.ids.data_ptr() == record.words.data_ptr()
    assert scratch.weights.data_ptr() == record.words.data_ptr() + 4 * TOP_K
    assert scratch.ids.stride() == scratch.weights.stride() == (2 * TOP_K, 1)
    assert scratch.ids.dtype is torch.int32 and scratch.weights.dtype is torch.float32
    assert scratch.selected_logits is selected
    assert record.packed.shape == (8, 4 * TOP_K) and record.packed.dtype is torch.bfloat16
    assert record.packed.stride(0) * record.packed.element_size() == 48


def test_a_route_record_carries_every_id_and_weight_bit_unchanged():
    """Six IDs and six FP32 weights are 48 bytes, so records travel as whole transport packs."""
    world, rows = 3, 4
    # Two's-complement words for -0.0, the smallest denormal, 1.0, the largest finite value,
    # another denormal, and -infinity.
    patterns = torch.tensor([-2147483648, 1, 1065353216, 2139095039, 4194304, -8388608],
                            dtype=torch.int32)
    records = [RouteRecord.allocate(rows, TOP_K, "cpu") for _ in range(world)]
    for rank, record in enumerate(records):
        record.words[:, :TOP_K].copy_(torch.randint(0, 384, (rows, TOP_K), dtype=torch.int32) - rank)
        # Negative zero, a denormal, one, the largest finite value, and both infinities: the
        # weight words keep their bits, so a route whose weight is unusual still arrives intact.
        record.words[:, TOP_K:] = patterns
    gathered = RouteRecord.allocate(world * rows, TOP_K, "cpu")
    for rank, record in enumerate(records):
        gathered.packed[rank * rows : (rank + 1) * rows].copy_(record.packed)
    assert gathered.packed.shape[1] * gathered.packed.element_size() % 16 == 0
    ids = torch.zeros(world * rows, TOP_K, dtype=torch.int32)
    weights = torch.zeros(world * rows, TOP_K)
    gathered.read(ids, weights)
    for rank, record in enumerate(records):
        block = slice(rank * rows, (rank + 1) * rows)
        assert torch.equal(ids[block], record.words[:, :TOP_K])
        assert torch.equal(weights[block].view(torch.int32), record.words[:, TOP_K:])


def test_a_route_record_rejects_a_row_that_is_not_whole_transport_packs():
    with pytest.raises(ValueError, match="16-byte packs"):
        RouteRecord.allocate(4, 1, "cpu")
    with pytest.raises(ValueError, match="positive rows"):
        RouteRecord.allocate(0, TOP_K, "cpu")
    record = RouteRecord.allocate(4, TOP_K, "cpu")
    assert record.rows(2).words.shape == (2, 2 * TOP_K)
    with pytest.raises(ValueError, match="exceed capacity"):
        record.rows(5)


@pytest.mark.parametrize("seed", [0, 7])
def test_gathered_rank_records_reproduce_the_full_row_routes(seed):
    """Routing each rank's rows and gathering the selection matches routing the gathered logits."""
    world, rows, experts = 4, 6, 12
    tokens = world * rows
    generator = torch.Generator().manual_seed(seed)
    # Three score levels against twelve experts tie inside and across top-k boundaries, and the
    # last row ties every expert, so each row's own tie order has to survive the decomposition.
    logits = torch.randint(0, 3, (tokens, experts), generator=generator).float()
    logits[-1].zero_()
    full = RouterScratch.allocate(tokens, "cpu", experts=experts, top_k=TOP_K)
    _prepared(logits, full)(logits)
    records = [RouteRecord.allocate(rows, TOP_K, "cpu") for _ in range(world)]
    for rank, record in enumerate(records):
        local = logits[rank * rows : (rank + 1) * rows]
        _prepared(local, record.router_scratch(torch.zeros(rows, TOP_K)))(local)
    gathered = RouteRecord.allocate(tokens, TOP_K, "cpu")
    for rank, record in enumerate(records):
        gathered.packed[rank * rows : (rank + 1) * rows].copy_(record.packed)
    ids = torch.empty_like(full.ids)
    weights = torch.empty_like(full.weights)
    gathered.read(ids, weights)
    assert torch.equal(ids, full.ids)
    # The weights cross ranks as bits, not as numbers, so they must match bit for bit.
    assert torch.equal(weights.view(torch.int32), full.weights.view(torch.int32))
    assert weights.is_contiguous() and ids.is_contiguous()


def test_joint_routing_keeps_the_correction_bias_and_scaling_of_each_row():
    """Bias correction, renormalization, and scaling all stay inside a row's own selection."""
    rows, experts = 5, 8
    logits = torch.linspace(-1, 1, rows * experts).reshape(rows, experts)
    bias = torch.linspace(1, -1, experts)
    options = dict(score_func="sigmoid", renormalize=True, routed_scaling_factor=2.5,
                   correction_bias=bias)
    full = RouterScratch.allocate(rows, "cpu", experts=experts, top_k=TOP_K)
    _prepared(logits, full, **options)(logits)
    record = RouteRecord.allocate(rows, TOP_K, "cpu")
    _prepared(logits, record.router_scratch(torch.zeros(rows, TOP_K)), **options)(logits)
    ids, weights = torch.empty_like(full.ids), torch.empty_like(full.weights)
    record.read(ids, weights)
    assert torch.equal(ids, full.ids)
    assert torch.equal(weights.view(torch.int32), full.weights.view(torch.int32))
