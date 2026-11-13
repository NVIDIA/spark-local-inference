# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact-weight MTP operator and graph checks on an explicitly assigned GPU."""

import os
from pathlib import Path

import pytest
import torch

from ignition.preparation import inspect_qwen38_checkpoint
from ignition.preparation.qwen38 import TEXT_PREFIX
from ignition.preparation.scratch import ScratchArena
from ignition.recipes.qwen38.attention import QSAMetadata, QSAState, RotaryTables
from ignition.recipes.qwen38.mtp import MTPScratch, PreparedMTP, PreparedMTPFeedback
from ignition.recipes.qwen38.mtp_weights import MTPWeights
from ignition.recipes.qwen38.weights import QwenWeightLoader

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an assigned CUDA_VISIBLE_DEVICES and FLARE_RUN_CUDA_TESTS=1")

FEEDBACK = Path(__file__).resolve().parents[1] / "fixtures/qwen38_mtp_feedback.pt"


@pytest.fixture(scope="module")
def weights():
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability(0) in ((12, 0), (12, 1))
    manifest = inspect_qwen38_checkpoint(os.environ.get("FLARE_QWEN38_SNAPSHOT",
        "/data/checkpoint-releases/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD"))
    with QwenWeightLoader(manifest, "cuda:0", prefixes=(f"{TEXT_PREFIX}.embed_tokens.", "lm_head.")) as loader:
        embedding = loader.load_tensor(f"{TEXT_PREFIX}.embed_tokens.weight")
        head = loader.load_tensor("lm_head.weight")
        assert loader._session.torch_copy_bytes == 0
    value = MTPWeights.load(manifest, "cuda:0", embedding_weight=embedding, lm_head_weight=head)
    assert value.embedding is embedding and value.lm_head is head
    assert value.stored_bytes == 1596720640
    return value


def inputs():
    values = torch.load(FEEDBACK, map_location="cpu", weights_only=True)
    hidden, tokens = values["hidden"], values["tokens"]
    assert hidden.shape == (len(tokens), 10240)
    return hidden, tokens


def test_exact_weight_feedback_matches_published_operator_reference(weights):
    from b12x.preparation import PreparationSession
    from b12x.sequence import mtp_feedback

    hidden, tokens = inputs()
    hidden = hidden.cuda()
    embedding = weights.embedding.index_select(0, tokens.cuda())
    output = torch.empty_like(hidden)
    arena = ScratchArena("cuda:0")
    with PreparationSession(device="cuda:0", autotune=False, compile_workers=0) as session:
        op = PreparedMTPFeedback(weights.module, embedding, hidden, output, arena, session=session, name="mtp.feedback")
        arena.finalize()
        op.functional(session=session)
        arena.freeze()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.stream(stream):
                output.copy_(op(embedding, hidden))
            stream.synchronize()
            with session.capture(), torch.cuda.graph(graph, stream=stream):
                output.copy_(op(embedding, hidden))
            session.freeze()
            expected = mtp_feedback.reference.feedback(embedding, hidden.view(-1, 4, 2560),
                weights.module.pre_fc_norm_embedding.weight, weights.module.pre_fc_norm_hidden.weight,
                weights.module.fc_embedding.weight, weights.module.fc_hidden.weight, eps=weights.module.config.rms_norm_eps)
            for poison in (0xA5, 0x5A):
                arena.tensor.fill_(poison)
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()["allocation.all.allocated"]
                graph.replay(); torch.cuda.synchronize()
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
                torch.testing.assert_close(output.view_as(expected), expected, rtol=2e-2, atol=4e-2)
        finally:
            stream.synchronize(); graph.reset()


@pytest.mark.parametrize("compiled", [False, True, "aot"])
def test_exact_weight_mtp_capture_replay_preserves_inputs_and_persistent_outputs(weights, compiled):
    from b12x.preparation import PreparationSession

    host_hidden, host_tokens = inputs()
    live, capacity = len(host_tokens), 8
    scratch = MTPScratch.allocate(capacity, 1, "cuda:0")
    hidden = torch.zeros_like(scratch.residual)
    tokens = torch.full((capacity,), -1, dtype=torch.int64, device="cuda")
    selected = torch.tensor([live - 1], dtype=torch.int64, device="cuda")
    metadata = QSAMetadata.allocate(tokens=capacity, batch=1, context=128, device=torch.device("cuda:0"))
    state = QSAState.allocate(pages=2, slots=1, device=torch.device("cuda:0"), draft_tokens=7)
    rotary = RotaryTables.allocate(weights.module.config, 128, torch.device("cuda:0"))
    arena = ScratchArena("cuda:0")
    with PreparationSession(device="cuda:0", autotune=False, compile_workers=0) as session:
        op = PreparedMTP(weights, scratch, tokens, hidden, state, metadata, rotary, arena,
            output_indices=selected, session=session, name="mtp", max_cache_pages=2)
        arena.finalize()
        op.functional(session=session)
        op.release_outputs()
        outputs = tuple(torch.empty_like(value) for value in (scratch.residual, scratch.hidden, scratch.logits))
        def execute():
            result = op(tokens, hidden, metadata, selected)
            for destination, value in zip(outputs, (result.residual, result.hidden, result.logits), strict=True):
                destination.copy_(value)

        arena.freeze()
        forward = execute
        if compiled == "aot":
            import io
            from ignition.preparation.aot import aot_forward

            artifact = aot_forward(execute)
            artifact(execute)
            data = type(artifact).serialize(artifact).serialized_data
            restored = torch.compiler.load_compiled_function(io.BytesIO(data))
            forward = lambda: restored(execute)
        elif compiled:
            from ignition.preparation.compilation import compile_forward

            forward = compile_forward(execute)
        stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.stream(stream):
                forward()
            stream.synchronize()
            with session.capture(), torch.cuda.graph(graph, stream=stream):
                forward()
            session.freeze()
            hidden[:live].copy_(host_hidden); tokens[:live].copy_(host_tokens)
            metadata.main_block_table[0].copy_(torch.tensor([1, 0], device="cuda", dtype=torch.int32))
            metadata.compressed_block_table.copy_(metadata.main_block_table)
            metadata.raw_state_slot_ids.fill_(0)
            metadata.request_ids[:live].fill_(0)
            metadata.query_positions[:live].copy_(torch.arange(live, device="cuda"))
            metadata.rope_positions[:live].copy_(torch.arange(live, device="cuda")[:, None].expand(live, 3))
            metadata.sequence_lengths.fill_(live); metadata.query_start_loc[-1] = live
            metadata.num_accepted_tokens.fill_(1); metadata.is_prefilling.fill_(True)
            metadata.slot_mapping[:live].copy_(torch.arange(64, 64 + live, device="cuda"))
            observed = []
            for poison in (0xA5, 0x5A):
                from dataclasses import fields, is_dataclass

                def poison_examples(value):
                    if isinstance(value, torch.Tensor):
                        value.view(torch.uint8).fill_(poison)
                    elif is_dataclass(value):
                        for field in fields(value):
                            poison_examples(getattr(value, field.name))

                poison_examples(scratch)
                for field in state.__dataclass_fields__:
                    value = getattr(state, field)
                    value.fill_(-1 if value.dtype == torch.int64 else 0)
                arena.tensor.fill_(poison)
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()["allocation.all.allocated"]
                graph.replay(); torch.cuda.synchronize()
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
                result = tuple(x.cpu().clone() for x in outputs)
                assert all(torch.isfinite(x).all() for x in result)
                assert torch.count_nonzero(result[0][live:]) == 0
                assert torch.count_nonzero(result[1][live:]) == 0
                assert torch.equal(hidden[:live].cpu(), host_hidden)
                assert torch.equal(tokens[:live].cpu(), host_tokens)
                observed.append(result)
            for first, second in zip(*observed):
                assert torch.equal(first.view(torch.uint8), second.view(torch.uint8))
        finally:
            stream.synchronize(); graph.reset()


@pytest.mark.parametrize("compiled", [False, "aot"])
def test_cache_only_absorption_matches_full_mtp_cache_under_replay(weights, compiled):
    from b12x.preparation import PreparationSession

    host_hidden, host_tokens = inputs()
    device = torch.device("cuda:0")
    capacity = 8
    scratch = MTPScratch.allocate(capacity, 1, device)
    hidden = host_hidden.repeat((capacity + len(host_tokens) - 1) // len(host_tokens), 1)[:capacity].to(device)
    tokens = host_tokens.repeat((capacity + len(host_tokens) - 1) // len(host_tokens))[:capacity].to(device)
    selected = torch.zeros(1, dtype=torch.int64, device=device)
    shared_hidden = torch.empty_like(hidden)
    metadata = QSAMetadata.allocate(tokens=capacity, batch=1, context=128, device=device)
    state = QSAState.allocate(pages=2, slots=1, device=device, draft_tokens=7)
    rotary = RotaryTables.allocate(weights.module.config, 128, device)
    arena = ScratchArena(device)
    graphs = []
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        variants = [(False, False), (True, False), (False, True), (True, True)]
        operators = [PreparedMTP(weights, scratch, tokens,
            shared_hidden if shared else hidden, state, metadata, rotary, arena,
            output_indices=selected, session=session, name=f"mtp.cache_only.{cache_only}.shared.{shared}",
            max_cache_pages=2, cache_only=cache_only) for cache_only, shared in variants]
        assert not hasattr(operators[1], "moe") and not hasattr(operators[1], "lm_head")
        arena.finalize()
        for op in operators:
            op.functional(session=session)
            op.release_outputs()
        outputs = tuple(torch.empty_like(value) for value in (scratch.residual, scratch.hidden, scratch.logits))
        arena.freeze()
        stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        try:
            for op, (cache_only, shared) in zip(operators, variants, strict=True):
                def execute(op=op, shared=shared):
                    result = op(tokens, shared_hidden if shared else hidden, metadata, selected)
                    if result is not None:
                        for destination, value in zip(outputs, (result.residual, result.hidden, result.logits), strict=True):
                            destination.copy_(value)
                forward = execute
                if compiled:
                    import io
                    from ignition.preparation.aot import aot_forward

                    artifact = aot_forward(execute)
                    artifact(execute)
                    data = type(artifact).serialize(artifact).serialized_data
                    restored = torch.compiler.load_compiled_function(io.BytesIO(data))
                    forward = lambda execute=execute, restored=restored: restored(execute)
                with torch.cuda.stream(stream):
                    forward()
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                graphs.append(graph)
                with session.capture(), torch.cuda.graph(graph, stream=stream):
                    forward()
            session.freeze()
            metadata.main_block_table[0].copy_(torch.tensor([1, 0], device=device, dtype=torch.int32))
            metadata.compressed_block_table.copy_(metadata.main_block_table)
            metadata.raw_state_slot_ids.fill_(0)
            for field in state.__dataclass_fields__:
                value = getattr(state, field)
                value.fill_(-1 if value.dtype == torch.int64 else 0)
            # Partial intervals, raw-ring wrap, rejected suffix replacement and page crossing.
            intervals = [(0, 6), (6, 7), (13, 8), (15, 1), (16, 7), (23, 8),
                         (31, 7), (38, 8), (46, 8), (54, 7), (61, 8), (69, 0)]
            for step, (start, live) in enumerate(intervals):
                positions = torch.arange(start, start + live, device=device)
                metadata.request_ids.fill_(-1); metadata.request_ids[:live].fill_(0)
                metadata.query_positions.fill_(-1); metadata.query_positions[:live].copy_(positions)
                metadata.rope_positions.fill_(-1)
                metadata.rope_positions[:live].copy_(positions[:, None].expand(live, 3))
                metadata.slot_mapping.fill_(-1)
                metadata.slot_mapping[:live].copy_(torch.where(positions < 64, positions + 64, positions - 64))
                metadata.sequence_lengths.fill_(start + live)
                metadata.query_start_loc[-1] = live
                metadata.num_accepted_tokens.fill_(2 if start == 15 else 1)
                metadata.is_prefilling.fill_(step == 0)
                initial = {field: getattr(state, field).clone() for field in state.__dataclass_fields__}
                graphs[0].replay()
                expected = {field: getattr(state, field).clone() for field in state.__dataclass_fields__}
                expected_outputs = tuple(value.clone() for value in outputs)
                for index, (cache_only, shared) in enumerate(variants[1:], 1):
                    for field, value in initial.items():
                        getattr(state, field).copy_(value)
                    outputs[1].fill_(19); outputs[2].fill_(23)
                    scratch.qsa.attended.fill_(29); scratch.qsa.selected_positions.fill_(-17)
                    arena.tensor.fill_(0xA5 if step % 2 else 0x5A)
                    if shared:
                        shared_hidden.copy_(hidden)
                    torch.cuda.synchronize()
                    allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
                    graphs[index].replay(); torch.cuda.synchronize()
                    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
                    for field, value in expected.items():
                        assert torch.equal(getattr(state, field), value), (compiled, start, field, shared)
                    if cache_only:
                        assert (outputs[1] == 19).all() and (outputs[2] == 23).all()
                        assert (scratch.qsa.attended == 29).all() and (scratch.qsa.selected_positions == -17).all()
                    else:
                        for observed, expected_output in zip(outputs, expected_outputs, strict=True):
                            assert torch.equal(observed, expected_output), (compiled, start)
            assert torch.count_nonzero(state.compressed_k_cache) > 0
        finally:
            stream.synchronize()
            for graph in graphs:
                graph.reset()
