# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mixed target metadata and state endpoints through native CUDA graph replay."""

import os

import pytest
import torch

from ignition.preparation.graphs import PreparedGraphBundle
from ignition.recipes.qwen38.attention import QSAMetadata
from ignition.recipes.qwen38.decoder_state import DecoderInputs, PreparedDecoderState
from ignition.recipes.qwen38.mixed_state import MixedPacking, MixedScratch, PreparedMixedState, role_inputs
from ignition.recipes.qwen38.mixed_model import MixedEmbeddingInjection
from ignition.recipes.qwen38.sampling import DeviceRequestState
from ignition.recipes.qwen38.speculative_state import PreparedVerificationState
from ignition.recipes.qwen38.state import HybridStatePool
from ignition.recipes.qwen38.vision_state import VisionFeatureState

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="explicit CUDA assignment required")


@pytest.mark.parametrize("enabled,width", [(False, 1), (True, 1), (True, 3), (True, 6)])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("live", [1, 3, 4])
@torch.no_grad()
def test_mixed_metadata_and_committed_endpoints(enabled, width, compiled, live):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    device = torch.device("cuda", 0)
    batch, prefill = 4, 128
    tokens = prefill + (batch - 1) * width
    pool = HybridStatePool.allocate(5, device, speculative_tokens=7 if enabled else 0)
    requests = DeviceRequestState.allocate(device, capacity=batch)
    inputs = DecoderInputs.allocate(device, context=256, tokens=tokens, capacity=batch)
    inputs.request_state_rows[:, 0] = torch.arange(1, 5, device=device)
    inputs.page_tables.copy_(torch.arange(16, device=device).view(4, 4))
    pi, di = role_inputs(inputs, prefill), role_inputs(inputs, batch * width)
    p = PreparedDecoderState(pi, requests, pool, tokens=prefill, rows=1, mode="prefill",
                             maintain_draft_state=enabled)
    if enabled:
        candidates = torch.arange(32, device=device).view(4, 8).long() + 100
        d = PreparedVerificationState(di, requests, pool, candidates,
            torch.ones(batch, dtype=torch.int32, device=device), width=width, rows=batch,
            token_limits=di.query_lengths[:batch])
    else:
        d = PreparedDecoderState(di, requests, pool, tokens=batch, rows=batch, mode="decode",
                                 maintain_draft_state=False)
    packing = MixedPacking(inputs, pi, di, batch, width, prefill)
    scratch = MixedScratch.allocate(prefill, batch * width, device, world_size=1)
    state = PreparedMixedState(packing, p, d, scratch,
        prompt_residual=scratch.prompt_residual, decode_residual=scratch.decode_residual)
    hidden = (torch.arange(tokens, device=device) + 100).bfloat16()[:, None].expand(-1, 2560).contiguous()
    residual = (torch.arange(tokens, device=device) + 10).bfloat16()[:, None].expand(-1, 10240).contiguous()
    sample = torch.zeros((batch * width + 1, 2560), dtype=torch.bfloat16, device=device)
    state.prepare_hidden(hidden, sample)
    state.prepare_residual(residual)
    features = VisionFeatureState.allocate(device, capacity=batch)
    features.generation.fill_(7)
    features.count.fill_(1)
    features.arena[:, 0].fill_(77)
    embedding = torch.zeros((tokens, 2560), dtype=torch.bfloat16, device=device)
    injection = MixedEmbeddingInjection(packing, features, p, embedding, scratch.prompt_gdn.output)

    packing.functional()
    state.functional()
    injection.functional()
    observed = QSAMetadata.allocate(tokens=tokens, batch=batch, context=256, device=device)

    def forward():
        current = packing.runtime()
        prompt = p.begin_runtime(current.prompt)
        decode = (d.begin_runtime(candidates, current.decode) if enabled
                  else d.begin_runtime(current.decode))
        runtime = state.combine_runtime(current, prompt, decode)
        errors = injection.validate_runtime(current)
        injection.inject(embedding, (errors, current))
        sample.copy_(state.finish_runtime(runtime, hidden, residual))
        observed.request_ids.copy_(runtime.qsa.request_ids)
        observed.is_prefilling.copy_(runtime.qsa.is_prefilling)
        observed.query_start_loc.copy_(runtime.qsa.query_start_loc)

    function = forward
    if compiled:
        from ignition.preparation.compilation import compile_forward
        function = compile_forward(forward)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        function()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        function()
    worker = PreparedGraphBundle(device=0, graphs={"mixed": graph}, buffers={"sample": sample},
        resources=(pool, requests, inputs, p, d, packing, scratch, state, hidden, residual, observed, function)).start_worker()
    try:
        for prompt_row in (-1, 0, 1, 3):
            if prompt_row >= live:
                continue
            for stale in (False, True):
                pool.backing.zero_()
                requests.generation.fill_(7)
                requests.status.fill_(1)
                requests.consumed_tokens.fill_(63)
                requests.pending_token.fill_(99)
                inputs.row_slot_ids.copy_(torch.tensor([2, 0, 3, 1], device=device))
                inputs.row_generations.fill_(7)
                inputs.live_rows.fill_(live)
                inputs.query_lengths.fill_(width)
                inputs.query_lengths[2] = 1
                inputs.query_lengths[live:].zero_()
                inputs.prefill_row.fill_(prompt_row)
                inputs.token_ids.copy_(torch.arange(tokens, device=device) + 1000)
                inputs.vision_token_indices.fill_(-1)
                features.feature_slot.fill_(-1)
                embedding.fill_(-4)
                inputs.prefill_positions.copy_((torch.arange(tokens, device=device) + 20)[:, None].expand(-1, 3))
                stale_row = (prompt_row + 1) % 4
                if stale:
                    inputs.row_generations[stale_row] = 6
                if prompt_row >= 0:
                    prompt_slot = int(inputs.row_slot_ids[prompt_row])
                    requests.pending_token[prompt_slot] = -1
                    requests.consumed_tokens[prompt_slot] = 0
                    inputs.query_lengths[prompt_row] = 65
                lengths = inputs.query_lengths.tolist()
                starts = [sum(lengths[:r]) for r in range(4)]
                if prompt_row >= 0:
                    inputs.vision_token_indices[starts[prompt_row] + 17] = 0
                    features.feature_slot.fill_(0)
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()["allocation.all.allocated"]
                ticket = worker.submit(0)
                completion = worker.poll(30000)
                assert completion.ticket == ticket and completion.error is None
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
                for row, slot in enumerate([2, 0, 3, 1]):
                    start, length = starts[row], lengths[row]
                    active = row < live and not (stale and row == stale_row)
                    assert observed.request_ids[start:start + length].tolist() == [row if active else -1] * length
                    assert observed.is_prefilling[row].item() == (row == prompt_row)
                    consumed = int(requests.consumed_tokens[slot])
                    expected = 65 if row == prompt_row else 63 + int(active and not enabled)
                    assert consumed == expected
                    if row == prompt_row or active and not enabled:
                        assert pool.last_hidden[slot + 1, 0] == hidden[start + length - 1, 0]
                    else:
                        assert not pool.last_hidden[slot + 1].any()
                assert (observed.request_ids[sum(lengths):] == -1).all()
                assert observed.query_start_loc.tolist() == [0, *[sum(lengths[:r + 1]) for r in range(4)]]
                expected_embedding = torch.full_like(embedding, -4)
                if prompt_row >= 0:
                    expected_embedding[starts[prompt_row] + 17] = 77
                torch.testing.assert_close(embedding, expected_embedding, rtol=0, atol=0)
    finally:
        worker.close()
        graph.reset()
