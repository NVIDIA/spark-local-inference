# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Mixed routing preserves live experts and excludes padded rows from expert work."""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from ignition.components.moe import PreparedRouter, RouterScratch
from ignition.preparation.compilation import compile_forward
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.deepseek4.decoder import PreparedRouting

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an assigned CUDA device")


@pytest.mark.parametrize("rows", (1, 127, 4096, 8192))
@pytest.mark.parametrize("token_routes", (False, True))
@torch.no_grad()
def test_mixed_routing_matches_separate_text_and_image_selection(rows, token_routes):
    device = torch.device("cuda", 0)
    torch.manual_seed(2781 + rows)
    logits = torch.randn((rows, 256), device=device)
    logits[0].zero_()
    metadata = SimpleNamespace(token_ids=torch.zeros(rows, device=device, dtype=torch.int64),
                               positions=torch.arange(rows, device=device))
    weights = SimpleNamespace(correction=torch.randn(256, device=device) * .1,
        vision_correction=torch.randn(256, device=device) * .1,
        token_experts=((torch.arange(129280, device=device)[:, None]
                        + torch.arange(6, device=device)) % 256).int() if token_routes else None)
    actual, text, image = [RouterScratch.allocate(rows, device, experts=256, top_k=6) for _ in range(3)]
    graphs, owners = {}, []
    with RecordedPreparationSession(device=device, checkpoint="deepseek4-routing",
            workload_schema="mixed-expert-routing.v1", numerical_policy="fp32") as session:
        routing = PreparedRouting(weights, SimpleNamespace(router_logits=logits, router=actual),
            metadata, mode="prefill", session=session, name="mixed")
        options = dict(session=session, score_func="sqrtsoftplus", renormalize=True,
                       routed_scaling_factor=1.5)
        reference_text = (None if token_routes else PreparedRouter(logits, text, name="text",
                          correction_bias=weights.correction, **options))
        reference_image = PreparedRouter(logits, image, name="image",
                                          correction_bias=weights.vision_correction, **options)

        def reference():
            if reference_text is not None:
                reference_text(logits)
            else:
                text.ids.copy_(weights.token_experts[metadata.token_ids.clamp(0, 129279)])
                selected = F.softplus(logits).sqrt().gather(1, text.ids.long())
                text.weights.copy_(selected / selected.sum(-1, keepdim=True) * 1.5)
            reference_image(logits)
            mask = metadata.token_ids[:, None] >= 129280
            text.ids.copy_(torch.where(mask, image.ids, text.ids))
            text.weights.copy_(torch.where(mask, image.weights, text.weights))

        def mixed():
            routes = routing(logits, metadata)
            actual.ids.copy_(routes.ids)
            actual.weights.copy_(routes.weights)
        session.tune()
        for name, forward in (("reference", reference), ("mixed", mixed)):
            compiled = compile_forward(forward)
            compiled()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                compiled()
            graphs[name] = graph
            owners.append(compiled)
        session.freeze()
        bundle = PreparedGraphBundle(device=0, graphs=graphs, buffers={"logits": logits},
            resources=(session, routing, reference_text, reference_image, actual, text, image,
                       metadata, weights, owners))
        worker = bundle.start_worker()
        try:
            for images, valid in ((0, rows), (rows, rows), (rows // 2, max(0, rows - 7)), (0, 0), (0, rows)):
                metadata.token_ids.copy_(torch.arange(rows, device=device) % 129280)
                metadata.token_ids[:images].add_(129280)
                metadata.positions.fill_(-1)
                metadata.positions[:valid].copy_(torch.arange(valid, device=device))
                logits.mul_(-.75)
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                for graph_index in (0, 1):
                    ticket = worker.submit(graph_index)
                    completion = worker.poll(timeout_ms=120000)
                    assert completion is not None and completion.ticket == ticket and completion.error is None
                after = torch.cuda.memory_stats()
                for key in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key]
                torch.testing.assert_close(actual.ids[:valid], text.ids[:valid], rtol=0, atol=0)
                assert actual.ids[valid:].eq(-1).all()
                torch.testing.assert_close(actual.weights, text.weights, rtol=0, atol=0)
            timings = []
            for graph in graphs.values():
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(100):
                    graph.replay()
                end.record()
                end.synchronize()
                timings.append(begin.elapsed_time(end) * 10)
            print(f"rows={rows} token_routes={token_routes}: separate={timings[0]:.2f}us mixed={timings[1]:.2f}us")
        finally:
            worker.close()
            for graph in graphs.values():
                graph.reset()
