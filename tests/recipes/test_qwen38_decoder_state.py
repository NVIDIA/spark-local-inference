# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device metadata and exact consumed endpoints through native replay."""

import os

import pytest
import torch

from ignition.preparation.graphs import PreparedGraphBundle
from ignition.recipes.qwen38.decoder_state import DecoderInputs, PreparedDecoderState
from ignition.recipes.qwen38.sampling import DeviceRequestState, RequestAdmission
from ignition.recipes.qwen38.state import HybridStatePool


def test_input_contract():
    inputs = DecoderInputs.allocate("cpu", context=256)
    assert inputs.page_tables.shape == (4, 4)
    assert inputs.prefill_positions.shape == (512, 3)
    assert (inputs.row_slot_ids == -1).all()
    assert inputs.live_rows.item() == 0
    with pytest.raises(ValueError):
        DecoderInputs.allocate("cpu", context=65)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="explicit CUDA assignment required")
@pytest.mark.parametrize("compile_forward", [False, True])
@torch.no_grad()
def test_native_metadata_exact_endpoints_and_terminal_guards(compile_forward):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    device = torch.device("cuda", 0)
    pool = HybridStatePool.allocate(5, device)
    requests = DeviceRequestState.allocate(device,
        consumed_tokens=pool.consumed_tokens[1:5, 0], ple_history=pool.ple_history[1:5])
    inputs = DecoderInputs.allocate(device, context=256)
    inputs.request_state_rows[:, 0] = torch.arange(1, 5, device=device)
    metadata = PreparedDecoderState(inputs, requests, pool, tokens=128, rows=1, mode="prefill")
    decode = PreparedDecoderState(inputs, requests, pool, tokens=4, rows=4, mode="decode")
    source = torch.arange(128, device=device, dtype=torch.float32).bfloat16()[:, None].expand(128, 10240).contiguous()
    hidden = source[:, :2560].contiguous()
    sample = torch.zeros((4, 2560), dtype=torch.bfloat16, device=device)
    metadata.prepare_hidden(hidden, sample[:1])
    decode.prepare_hidden(hidden[:4], sample)

    prefill_gdn, prefill_qsa = metadata.gdn, metadata.qsa
    decode_gdn, decode_qsa = decode.gdn, decode.qsa
    metadata.functional()
    decode.functional()

    def prefill():
        runtime = metadata.begin_runtime()
        sample[:1].copy_(metadata.finish_runtime(runtime, hidden, source))
        for name in ("checkpoint_offsets", "initial_state_indices", "final_state_indices"):
            getattr(prefill_gdn, name).copy_(getattr(runtime.gdn, name))
        prefill_qsa.slot_mapping.copy_(runtime.qsa.slot_mapping)
        prefill_qsa.request_ids.copy_(runtime.qsa.request_ids)

    def step():
        runtime = decode.begin_runtime()
        sample.copy_(decode.finish_runtime(runtime, hidden[:4], source[:4]))
        decode_qsa.rope_positions.copy_(runtime.qsa.rope_positions)
        decode_gdn.num_tokens.copy_(runtime.gdn.num_tokens)

    graphs = {}
    compiled_calls = []
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    for name, function in (("prefill", prefill), ("decode", step)):
        if compile_forward:
            from ignition.preparation.compilation import compile_forward as compile_model

            function = compile_model(function)
            compiled_calls.append(function)
        with torch.cuda.stream(stream):
            function()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            function()
        graphs[name] = graph
    worker = PreparedGraphBundle(device=0, graphs=graphs, buffers={"sample": sample},
        resources=(pool, requests, inputs, metadata, decode, source, hidden, compiled_calls)).start_worker()

    def replay(index):
        torch.cuda.synchronize()
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        ticket = worker.submit(index)
        completion = worker.poll(30000)
        assert completion.ticket == ticket
        assert completion.error is None
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before

    try:
        tokens = list(range(65))
        requests.admit(2, RequestAdmission(11, 9), tokens)
        inputs.row_slot_ids[0] = 2
        inputs.row_generations[0] = 11
        inputs.live_rows.fill_(1)
        inputs.query_lengths[0] = 65
        inputs.token_ids[:65].copy_(torch.tensor(tokens, device=device))
        inputs.prefill_positions[:65].copy_((torch.arange(65, device=device) + 17)[:, None].expand(65, 3))
        inputs.page_tables[2].copy_(torch.tensor([5, 3, 1, 7], dtype=torch.int32, device=device))
        pool.rope_delta[3] = 17
        replay(0)
        assert prefill_gdn.checkpoint_offsets.tolist() == [0]
        assert prefill_gdn.initial_state_indices.tolist() == [0]
        assert prefill_gdn.final_state_indices.tolist() == [3]
        assert pool.consumed_tokens[3].item() == 65
        assert pool.ple_history[3].tolist() == [63, 64]
        assert pool.last_hidden[3, 0].item() == 64
        assert prefill_qsa.slot_mapping[:65].tolist() == list(range(320, 384)) + [192]
        assert (prefill_qsa.request_ids[65:] == -1).all()

        # Consume the pending token at its exact position.
        requests.consumed_tokens[2] = 63
        requests.pending_token[2] = 901
        pool.gdn[0].recurrent[3].fill_(.25)
        replay(1)
        assert inputs.token_ids[0].item() == 901
        assert decode_qsa.rope_positions[0].tolist() == [80, 80, 80]
        assert requests.consumed_tokens[2].item() == 64

        # Replaying decode without a new sample rejects before any model write.
        before_state = pool.backing.clone()
        replay(1)
        assert requests.status[2].item() == 3
        assert requests.error_code[2].item() == 32
        assert decode_gdn.num_tokens.item() == 0
        assert torch.equal(before_state, pool.backing)

        requests.admit(2, RequestAdmission(12, 9), list(range(65)), consumed_tokens=65)
        inputs.row_generations[0] = 12
        inputs.prefill_positions[:65].copy_((torch.arange(65, 130, device=device) + 17)[:, None].expand(65, 3))
        before_state.copy_(pool.backing)
        replay(0)
        assert requests.error_code[2].item() == 0
        assert requests.consumed_tokens[2].item() == 130
        assert prefill_gdn.initial_state_indices.tolist() == [3]
        assert prefill_gdn.checkpoint_offsets.tolist() == [0]
    finally:
        worker.close()
        for graph in graphs.values():
            graph.reset()
