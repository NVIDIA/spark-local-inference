# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from ignition.components import moe
from ignition.components.moe import PreparedRouter, RouterScratch
from ignition.recipes.qwen38.moe import PreparedQwenMoe, QwenMoeScratch, QwenMoeWeights


@pytest.mark.parametrize("tokens", [1, 4, 17, 4096])
def test_router_declares_exact_fixed_library_abi(tokens):
    logits = torch.zeros((tokens, 512), dtype=torch.bfloat16)
    scratch = RouterScratch.allocate(tokens, "cpu", experts=512, top_k=10)
    requests = []
    session = SimpleNamespace(prepare=lambda batch: requests.extend(batch))
    router = PreparedRouter(logits, scratch, session=session, name="router")
    query = router.plan.query
    assert router.plan.component_id == "moe.route_topk"
    assert (query.num_tokens, query.num_experts, query.top_k) == (tokens, 512, 10)
    assert query.renormalize and query.score_func == "softmax" and query.routed_scaling_factor == 1
    assert (query.logits_row_stride, query.topk_row_stride, query.ids_row_stride, query.weights_row_stride) == (512, 10, 10, 10)
    assert not query.has_correction_bias and not query.has_image_correction_bias and not query.has_image_mask
    assert len(requests) == 1 and requests[0].benchmark_call is None
    primed = []
    call = requests[0].prepare_call(SimpleNamespace(run=lambda *args: primed.append(args)))
    call.run()
    assert primed[0] == (logits, scratch.selected_logits, scratch.ids, scratch.weights)
    assert call.owners == (logits, scratch)
    assert set(scratch.__dataclass_fields__) == {"selected_logits", "ids", "weights"}
    assert scratch.selected_logits.dtype == scratch.weights.dtype == torch.float32
    assert scratch.ids.dtype == torch.int32


def test_router_reuses_preallocated_buffers_through_public_entrypoint(monkeypatch):
    scratch = RouterScratch.allocate(2, "cpu", experts=512, top_k=10)
    pointers = tuple(getattr(scratch, name).data_ptr() for name in scratch.__dataclass_fields__)
    logits = torch.zeros(2, 512, dtype=torch.bfloat16)
    router = PreparedRouter(logits, scratch, session=SimpleNamespace(prepare=lambda _: None), name="router")
    calls = []
    monkeypatch.setattr(moe.fused_moe, "route_topk", lambda *args: calls.append(args))
    router(logits)
    logits.fill_(3)
    router(logits)
    assert len(calls) == 2
    assert all(call == (router.plan, logits, scratch.selected_logits, scratch.ids, scratch.weights) for call in calls)
    assert pointers == tuple(getattr(scratch, name).data_ptr() for name in scratch.__dataclass_fields__)


@pytest.mark.parametrize("experts,top_k", [(0, 1), (5, 10), (512, 0), (1025, 10), (512, True)])
def test_router_rejects_invalid_geometry(experts, top_k):
    with pytest.raises(ValueError, match="router requires"):
        RouterScratch.allocate(1, "cpu", experts=experts, top_k=top_k)


def test_router_rejects_changed_precision():
    scratch = RouterScratch.allocate(1, "cpu", experts=512, top_k=10)
    router = PreparedRouter(torch.zeros(1, 512, dtype=torch.bfloat16), scratch,
        session=SimpleNamespace(prepare=lambda _: None), name="router")
    with pytest.raises(ValueError, match="prepared dtype"):
        router(torch.zeros(1, 512, dtype=torch.float32))


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                   reason="requires an explicitly assigned CUDA device")
@pytest.mark.parametrize("tokens,experts", [(1, 512), (8, 512), (17, 384)])
def test_native_router_ties_fp32_weights_and_changed_input_graph(tokens, experts):
    from b12x.preparation import PreparationSession

    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign a CUDA device explicitly"
    device = torch.device("cuda", 0)
    logits = torch.zeros((tokens, experts), dtype=torch.bfloat16, device=device)
    scratch = RouterScratch.allocate(tokens, device, experts=experts, top_k=10)
    stream = torch.cuda.Stream(device=device)
    graph = torch.cuda.CUDAGraph()
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        router = PreparedRouter(logits, scratch, session=session, name="router")
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            router(logits)
        stream.synchronize()
        try:
            with session.capture(), torch.cuda.graph(graph, stream=stream):
                router(logits)
            session.freeze()
            for iteration in range(2):
                values = torch.randn((tokens, experts), generator=torch.Generator().manual_seed(13 + iteration)).bfloat16()
                values[0].zero_()
                if tokens > 1:
                    values[1, :32] = 10
                if iteration:
                    values[:, -16:] = 20
                logits.copy_(values)
                torch.cuda.synchronize(device)
                before = torch.cuda.memory_stats(device)
                with torch.cuda.stream(stream):
                    graph.replay()
                stream.synchronize()
                after = torch.cuda.memory_stats(device)
                assert before["allocation.all.allocated"] == after["allocation.all.allocated"]
                expected_ids = experts - 1 - torch.argsort(values.flip(1).float(), dim=-1, descending=True, stable=True)[:, :10]
                selected = values.gather(1, expected_ids).float()
                expected_weights = selected.softmax(-1)
                assert torch.equal(scratch.ids.cpu(), expected_ids.int())
                assert torch.equal(scratch.selected_logits.cpu(), selected)
                torch.testing.assert_close(scratch.weights.cpu(), expected_weights, rtol=2e-6, atol=1e-7)
                assert not torch.equal(scratch.weights, scratch.weights.bfloat16().float())
        finally:
            stream.synchronize()
            graph.reset()


def test_complete_recipe_rejects_component_expert_subsets():
    experts = SimpleNamespace(num_experts=2, hidden_size=2560, intermediate_size=640)
    weights = QwenMoeWeights(torch.empty((512, 2560), device="meta"), experts, None)
    with pytest.raises(ValueError, match="all 512 experts"):
        PreparedQwenMoe(torch.empty((1, 2560), device="meta", dtype=torch.bfloat16),
                        weights, session=None, name="invalid-subset")


def test_moe_capacity_views_share_all_serial_layer_intermediates():
    scratch = QwenMoeScratch.allocate(16, "cpu")
    rows = scratch.rows(4)
    assert rows.output.shape == (4, 2560)
    assert rows.output.data_ptr() == scratch.output.data_ptr()
    assert rows.routed_output.data_ptr() == scratch.routed_output.data_ptr()
    assert rows.router.ids.data_ptr() == scratch.router.ids.data_ptr()
    assert rows.shared.gate_up.data_ptr() == scratch.shared.gate_up.data_ptr()
    assert rows.shared.output.data_ptr() != rows.routed_output.data_ptr()
    for capacity in (0, 17):
        with pytest.raises(ValueError, match="capacity"):
            scratch.rows(capacity)
