# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import replace

import pytest
import torch

from tests.components.completion_support import completion_outputs
import triton
import triton.language as tl

from ignition.components.sampling import (
    DUPLICATE_SLOT,
    ERROR,
    FINISHED,
    INVALID_ADMISSION,
    INVALID_DISTRIBUTION,
    UNCONSUMED_PENDING,
    AdmissionBuffers,
    PreparedAdmission,
    PreparedSampling,
    philox4x32_reference,
)
from ignition.components.token_state import PreparedTokenState
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.recipes.qwen38.sampling import DeviceRequestState, RequestAdmission


def test_philox_random123_zero_vector_and_uniform_distribution():
    assert philox4x32_reference(0, 0) == (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)
    words = torch.tensor([philox4x32_reference(123456789, n)[0] for n in range(20000)], dtype=torch.int64)
    uniforms = (words >> 8).double() * 2**-24
    assert 0 <= uniforms.min() and uniforms.max() < 1
    counts = torch.bincount((uniforms * 20).long(), minlength=20).double()
    assert ((counts - 1000).square() / 1000).sum() < 55
    assert abs(uniforms.mean().item() - .5) < .008
    assert philox4x32_reference(123, 2**32) != philox4x32_reference(123, 0)
    assert philox4x32_reference(123 + 2**32, 0) != philox4x32_reference(123, 0)


@pytest.mark.parametrize("changes", [
    {"seed": -1}, {"seed": 2**64}, {"generation": 0}, {"generation": True},
    {"temperature": 0.}, {"temperature": -0.}, {"temperature": -1}, {"temperature": float("nan")}, {"top_p": 0},
    {"temperature": 1e-100}, {"repetition_penalty": 1e100},
    {"min_p": 1.1}, {"repetition_penalty": 0}, {"top_k": 248321},
    {"max_new_tokens": -1}, {"eos_token_ids": tuple(range(9))},
    {"eos_token_ids": (248320,)},
])
def test_admission_rejects_invalid_contract(changes):
    with pytest.raises(ValueError):
        replace(RequestAdmission(1, 1), **changes).validate()


def test_admission_full_prompt_counts_prefix_hit_explicit_eos_and_uint64_seed():
    state = DeviceRequestState.allocate("cpu", vocab_size=32)
    params = RequestAdmission(37, 2**64 - 1, eos_token_ids=(), max_new_tokens=0)
    state.admit(2, params, [4, 4, 7, 3, 4], consumed_tokens=5, ple_history=(3, 4))
    assert state.histogram[2, 4] == 3 and state.histogram[2, 7] == 1
    assert state.histogram[2].sum() == 5
    assert state.histogram[[0, 1, 3]].sum() == 0
    assert state.seed[2].item() == 2**64 - 1
    assert state.generation[2] == 37 and state.consumed_tokens[2] == 5
    assert state.generated_tokens[2] == 0 and state.pending_token[2] == -1
    assert state.status[2] == FINISHED and state.eos_count[2] == 0
    assert state.ple_history[2].tolist() == [3, 4]


def test_admission_uses_external_strided_snapshot_metadata():
    backing = torch.zeros((9, 128), dtype=torch.int64)
    state = DeviceRequestState.allocate("cpu", vocab_size=32,
        consumed_tokens=backing[1:5, 0], ple_history=backing[1:5, 1:3])
    state.admit(1, RequestAdmission(1, 13, eos_token_ids=()), [3] * 64,
                consumed_tokens=64, ple_history=(3, 3))
    assert backing[2, :3].tolist() == [64, 3, 3]
    assert backing[6].count_nonzero() == 0
    assert backing[0].sum() == 0


def test_native_admission_staging_is_contiguous_and_preserves_seed_bits():
    buffers = AdmissionBuffers.allocate("cpu", vocab_size=32)
    buffers.stage(2, 3, RequestAdmission(9, 2**64 - 1, eos_token_ids=(7,)), [4, 4, 5])
    assert buffers.admission_count.item() == 0
    assert buffers.admission_slots.tolist() == [-1, -1, 3, -1]
    assert buffers.admission_seed_bits[2].item() == -1
    assert buffers.admission_seed_bits.view(torch.uint64)[2].item() == 2**64 - 1
    assert buffers.admission_histogram[2, 4] == 2
    assert all(getattr(buffers, field).is_contiguous() for field in buffers.__dataclass_fields__)


def _reference(logits, counts, params, seed, counter, grammar=None):
    scores = logits.float().clone()
    seen = counts > 0
    scores[seen] = torch.where(scores[seen] > 0, scores[seen] / params.repetition_penalty,
                               scores[seen] * params.repetition_penalty) - params.presence_penalty
    scores -= params.frequency_penalty * counts
    if grammar is not None:
        scores[~grammar] = -torch.inf
    scores /= params.temperature
    values, ids = torch.sort(scores, descending=True, stable=True)
    if params.top_k > 0:
        values[values < values[params.top_k - 1].clone()] = -torch.inf
    probabilities = values.softmax(0)
    starts = torch.searchsorted(-values, -values, right=False)
    previous = (probabilities.cumsum(0) - probabilities)[starts]
    probabilities[previous >= params.top_p] = 0
    probabilities /= probabilities.sum()
    probabilities[probabilities < params.min_p * probabilities.max()] = 0
    probabilities /= probabilities.sum()
    ordered = torch.zeros_like(probabilities).scatter_(0, ids, probabilities)
    cdf = ordered.cumsum(0)
    uniform = _flashinfer_uniform(seed, counter)
    token = int(torch.searchsorted(cdf, torch.tensor(uniform), right=False))
    return token if token < len(cdf) else int(torch.nonzero(ordered > 0)[-1])


def _flashinfer_uniform(seed, counter):
    word = philox4x32_reference(seed, counter)[0]
    return float(torch.tensor(word, dtype=torch.float32) * 2**-32 + 2**-33)



CUDA = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_CUDA_TESTS=1 and an explicitly assigned CUDA_VISIBLE_DEVICES")


def _metadata(rows=4, fields=32):
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign a CUDA device explicitly"
    return (torch.arange(rows, dtype=torch.int32, device="cuda"),
            torch.ones(rows, dtype=torch.int64, device="cuda"),
            torch.tensor([rows], dtype=torch.int32, device="cuda"),
            torch.full((2, 4, fields), -99, dtype=torch.int64, device="cuda"),
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            torch.ones(1, dtype=torch.int64, device="cuda"))


def _worker(graphs, buffers, resources):
    return PreparedGraphBundle(device=0, graphs=graphs, buffers=buffers, resources=resources).start_worker()


def _replay(worker, index=0):
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
    ticket = worker.submit(index)
    completion = worker.poll(timeout_ms=30000)
    assert completion is not None and completion.ticket == ticket and completion.error is None
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated


@triton.jit
def _philox_probe(Output, SEED: tl.constexpr, OFFSET: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    word, _, _, _ = tl.randint4x(tl.full((), SEED, tl.uint64), index.to(tl.uint64) + OFFSET)
    tl.store(Output + index, word.to(tl.uint32).to(tl.int64), index < N)


@CUDA
def test_device_philox_matches_independent_integer_oracle():
    seed, offset, count = 2**64 - 79, 2**32 - 71, 2048
    output = torch.empty(count, dtype=torch.int64, device="cuda")
    _philox_probe[(triton.cdiv(count, 256),)](output, seed, offset, count, 256)
    assert output.cpu().tolist() == [philox4x32_reference(seed, offset + i)[0] for i in range(count)]


@CUDA
def test_native_categorical_distribution_and_exact_seed_progression():
    state = DeviceRequestState.allocate("cuda", vocab_size=8)
    metadata = _metadata()
    for slot in range(4):
        state.admit(slot, RequestAdmission(1, 9321 + slot, max_new_tokens=10000,
                    top_k=0, top_p=1., eos_token_ids=()), [])
    logits = torch.full((4, 8), -torch.inf, device="cuda")
    probabilities = torch.tensor([.1, .3, .6])
    logits[:, :3].copy_(probabilities.log().expand(4, 3))
    sampler = PreparedSampling(logits, state, *metadata[:3], completion_outputs(metadata[3]), *metadata[4:])
    sampler.prepare()
    tokens = torch.empty(4, dtype=torch.int64, device="cuda")
    spans = torch.arange(5, dtype=torch.int32, device="cuda")
    token_state = PreparedTokenState(state, tokens, spans, *metadata[:3], history=state.ple_history, history_pad_id=248044)
    token_state.prepare()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(64):
            sampler()
            token_state.gather()
            tokens.copy_(token_state.pending_tokens)
            token_state.consume()
    worker = _worker({"categorical": graph}, {"histogram": state.histogram}, (sampler, token_state, state))
    try:
        for _ in range(16):
            _replay(worker)
        counts = state.histogram.cpu()
        assert state.rng_counter.cpu().tolist() == [1024] * 4
        assert state.pending_token.cpu().tolist() == [-1] * 4
        for slot in range(4):
            # FlashInfer scans positive probabilities in vocabulary order.
            uniforms = [_flashinfer_uniform(9321 + slot, n) for n in range(1024)]
            expected = torch.bincount(torch.tensor([0 if u <= .1 else 1 if u <= .4 else 2 for u in uniforms]), minlength=8)
            assert torch.equal(counts[slot], expected)
        empirical = counts.sum(0)[:3].float() / counts.sum()
        torch.testing.assert_close(empirical, probabilities, rtol=0, atol=.018)
        assert counts[:, 3:].sum() == 0
    finally:
        worker.close()


@CUDA
@pytest.mark.parametrize("temperature", [1., .8])
@pytest.mark.parametrize("fields", [32])
def test_native_sampling_full_vocabulary_batch_reordering_and_pending_consumption(temperature, fields):
    state = DeviceRequestState.allocate("cuda")
    metadata = _metadata(fields=fields)
    slots, epochs, live, records, frame, sequence = metadata
    logits = torch.full((4, state.vocab_size), -30., device="cuda")
    params = [RequestAdmission(1, 184 + slot * 2**32, max_new_tokens=25,
        temperature=temperature, top_k=4, top_p=.8, min_p=.1,
        repetition_penalty=1.2, presence_penalty=.2, frequency_penalty=.13, eos_token_ids=()) for slot in range(4)]
    cpu_logits = torch.full((4, state.vocab_size), -30.)
    cpu_logits[:, :8] = torch.tensor([3., 3., 3., 2., 2., 1., -.3, -2.])
    cpu_counts = torch.zeros((4, state.vocab_size), dtype=torch.int32)
    for slot in range(4):
        state.admit(slot, params[slot], [0, 1, 1], consumed_tokens=3, ple_history=(1, 1))
        cpu_counts[slot, 0], cpu_counts[slot, 1] = 1, 2
    sampler = PreparedSampling(logits, state, *metadata[:3], completion_outputs(metadata[3]), *metadata[4:])
    sampler.prepare()
    tokens = torch.empty(4, dtype=torch.int64, device="cuda")
    spans = torch.arange(5, dtype=torch.int32, device="cuda")
    token_state = PreparedTokenState(state, tokens, spans, slots, epochs, live, history=state.ple_history, history_pad_id=248044)
    token_state.prepare()
    sample_graph, decode_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(sample_graph):
        sampler()
    with torch.cuda.graph(decode_graph):
        token_state.gather()
        tokens.copy_(token_state.pending_tokens)
        token_state.consume()
        sampler()
    worker = _worker({"sample": sample_graph, "decode": decode_graph}, {"logits": logits, "records": records},
                     (state, sampler, token_state))
    try:
        previous = [1] * 4
        for iteration in range(12):
            order = [3, 0, 2, 1] if iteration % 2 else [1, 2, 0, 3]
            slots.copy_(torch.tensor(order, dtype=torch.int32))
            logits.copy_(cpu_logits[order])
            frame.fill_(iteration % 2)
            sequence.fill_(iteration + 41)
            _replay(worker, 0 if iteration == 0 else 1)
            result = records[iteration % 2].cpu()
            for slot in range(4):
                expected = _reference(cpu_logits[slot], cpu_counts[slot], params[slot], params[slot].seed, iteration)
                assert result[slot, :8].tolist() == [1, iteration + 41, 3 + iteration, iteration + 1, expected, 1, 0, 0]
                if fields == 32:
                    assert result[slot, 8:].tolist() == [1, 0, 0, 0, -1, 2 + iteration,
                        0, 1, expected] + [-1] * 15
                cpu_counts[slot, expected] += 1
                assert state.pending_token[slot].item() == expected
                if iteration:
                    assert state.ple_history[slot, -1].item() == previous[slot]
                previous[slot] = expected
            assert torch.equal(state.histogram.cpu(), cpu_counts)
            assert state.rng_counter.cpu().tolist() == [iteration + 1] * 4
    finally:
        worker.close()


@CUDA
def test_terminal_grammar_errors_stale_epochs_and_inactive_lanes():
    state = DeviceRequestState.allocate("cuda", vocab_size=32)
    slots, epochs, live, records, frame, sequence = _metadata()
    logits = torch.zeros((4, 32), device="cuda")
    for slot in range(4):
        state.admit(slot, RequestAdmission(1, slot, temperature=1., eos_token_ids=(7,),
                    max_new_tokens=1 if slot == 1 else 10), [1, 2], consumed_tokens=2, ple_history=(1, 2))
    state.grammar_enabled.fill_(1)
    state.grammar_mask.zero_()
    state.grammar_mask[0, 0] = 1 << 7
    state.grammar_mask[1, 0] = 1 << 8
    state.grammar_mask[3, 0] = 1 << 9
    epochs[3] = 99
    sampler = PreparedSampling(logits, state, slots, epochs, live, completion_outputs(records), frame, sequence)
    sampler.prepare()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sampler()
    record_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(record_graph):
        sampler.record_completion()
    worker = _worker({"sampling": graph, "record": record_graph}, {"records": records}, (sampler, state))
    try:
        _replay(worker, 1)
        assert records[0, 0].cpu()[..., :8].tolist() == [1, 1, 2, 0, -1, 1, 0, 0]
        assert state.rng_counter.cpu().tolist() == [0] * 4
        _replay(worker)
        assert records[0, 0].cpu()[..., :8].tolist() == [1, 1, 2, 1, 7, FINISHED, 0, 0]
        assert records[0, 1].cpu()[..., :8].tolist() == [1, 1, 2, 1, 8, FINISHED, 0, 0]
        assert records[0, 2].cpu()[..., :8].tolist() == [1, 1, 2, 0, -1, ERROR, INVALID_DISTRIBUTION, 0]
        assert records[0, 3].cpu()[..., :8].tolist() == [0, 0, 0, 0, -1, 0, 0, 0]
        assert state.pending_token.cpu().tolist() == [7, 8, -1, -1]
        before = state.histogram.clone()
        _replay(worker)
        assert torch.equal(state.histogram, before)
        assert state.rng_counter.cpu().tolist() == [1, 1, 0, 0]
        epochs[3] = 1
        _replay(worker)
        assert state.pending_token[3] == 9
        _replay(worker)
        assert state.error_code[3] == UNCONSUMED_PENDING and state.generated_tokens[3] == 1
        live.zero_()
        frame.fill_(1)
        _replay(worker)
        assert records[1, :, 5].sum() == 0 and (records[1, :, 4] == -1).all()
    finally:
        worker.close()


@CUDA
def test_consumption_strided_live_history_and_pending_guard():
    backing = torch.full((9, 128), -22, dtype=torch.int64, device="cuda")
    state = DeviceRequestState.allocate("cuda", vocab_size=128,
        consumed_tokens=backing[1:5, 0], ple_history=backing[1:5, 1:3])
    for slot in range(4):
        state.admit(slot, RequestAdmission(3, 123, eos_token_ids=()), list(range(90)),
                    consumed_tokens=63, ple_history=(61, 62))
    slots, epochs, live, *_ = _metadata()
    epochs.fill_(3)
    tokens = torch.tensor([63, 64, 65, 66, 70, 71, 72, 90], dtype=torch.int64, device="cuda")
    spans = torch.tensor([0, 4, 4, 7, 8], dtype=torch.int32, device="cuda")
    slots.copy_(torch.tensor([2, 0, 3, 1], dtype=torch.int32))
    state.pending_token[0] = 88
    state.pending_token[1] = 91
    operation = PreparedTokenState(state, tokens, spans, slots, epochs, live, history=state.ple_history, history_pad_id=248044)
    operation.prepare()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation.gather()
        operation.consume()
    worker = _worker({"consume": graph}, {"backing": backing}, (state, operation))
    histogram = state.histogram.clone()
    try:
        _replay(worker)
        assert operation.positions.cpu().tolist() == [63] * 4
        assert operation.history.cpu().tolist() == [[61, 62]] * 4
        assert state.consumed_tokens.cpu().tolist() == [63, 63, 67, 66]
        assert state.ple_history[2].cpu().tolist() == [65, 66]
        assert state.status[1] == ERROR and state.pending_token[1] == 91
        assert state.status[0] == 1 and state.pending_token[0] == 88
        assert torch.equal(state.histogram, histogram)
        assert backing[0].eq(-22).all() and backing[:, 3:].eq(-22).all()
    finally:
        worker.close()


@CUDA
@pytest.mark.parametrize("capacity", [1, 2, 4])
def test_native_admission_scatter_preserves_restored_state_and_gates_final_sampling(capacity):
    state = DeviceRequestState.allocate("cuda", vocab_size=32)
    for slot in range(4):
        state.admit(slot, RequestAdmission(1, 100 + slot, eos_token_ids=()), [1, 2, 3],
                    consumed_tokens=3, ple_history=(2, 3))
    state.pending_token.fill_(12)
    buffers = AdmissionBuffers.allocate("cuda", vocab_size=32)
    order = [3, 1, 0, 2]
    for row, slot in enumerate(order[:capacity]):
        buffers.stage(row, slot, RequestAdmission(17 + slot, 2**64 - 1 - slot,
            temperature=1., eos_token_ids=()), [4, 4, 5, 6])
        state.ple_history[slot].copy_(torch.tensor([4, 5]))
    buffers.admission_count.fill_(capacity)
    metadata = _metadata()
    slots, epochs, live, records, frame, sequence = metadata
    slots.copy_(torch.tensor(order, dtype=torch.int32))
    epochs.copy_(torch.tensor([17 + slot for slot in order]))
    live.fill_(capacity)
    admission = PreparedAdmission(state, buffers, completion_outputs(records), frame, sequence, capacity=capacity)
    admission.prepare()
    logits = torch.zeros((4, 32), device="cuda")
    logits.fill_(-torch.inf)
    logits[:, 8] = 2
    enabled = torch.zeros(4, dtype=torch.int32, device="cuda")
    sampler = PreparedSampling(logits, state, *metadata[:3], completion_outputs(records), frame, sequence, sample_enabled=enabled)
    sampler.prepare()
    init_graph, sample_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(init_graph):
        admission()
    with torch.cuda.graph(sample_graph):
        sampler()
    worker = _worker({"init": init_graph, "sample": sample_graph}, {"records": records}, (admission, sampler, buffers, state))
    try:
        old_history = state.ple_history.clone()
        _replay(worker)
        assert torch.equal(state.ple_history, old_history)
        assert state.consumed_tokens.cpu().tolist() == [3] * 4
        for slot in order[:capacity]:
            assert records[0, slot].cpu()[..., :8].tolist() == [17 + slot, 1, 3, 0, -1, 1, 0, 0]
            assert state.seed[slot].item() == 2**64 - 1 - slot
            assert state.pending_token[slot] == -1
            assert state.histogram[slot, 4] == 2 and state.histogram[slot].sum() == 4
        for slot in order[capacity:]:
            assert state.generation[slot] == 1 and state.pending_token[slot] == 12
            assert state.histogram[slot, 1] == 1 and state.histogram[slot].sum() == 3
        before = state.histogram.clone()
        _replay(worker, 1)
        assert torch.equal(state.histogram, before)
        assert state.generated_tokens.sum() == 0 and state.rng_counter.cpu().tolist() == [0] * 4
        enabled[0] = 1
        frame.fill_(1)
        sequence.fill_(2)
        _replay(worker, 1)
        assert records[1, 3].cpu()[..., :8].tolist() == [20, 2, 3, 1, 8, 1, 0, 0]
        assert state.histogram[3, 8] == 1 and state.generated_tokens.sum() == 1
        enabled.zero_()
        _replay(worker, 1)
        assert state.status[3] == 1 and state.pending_token[3] == 8
        assert records[1, 3, 4] == -1
    finally:
        worker.close()


@CUDA
@pytest.mark.parametrize("fault", ["duplicate", "negative_histogram", "stale_epoch", "invalid_parameters", "zero_temperature", "negative_zero_temperature"])
def test_native_admission_errors_do_not_partially_mutate_resident_slots(fault):
    state = DeviceRequestState.allocate("cuda", vocab_size=32)
    state.admit(2, RequestAdmission(3, 71, eos_token_ids=()), [2, 2, 7], consumed_tokens=3, ple_history=(2, 7))
    state.pending_token[2] = 9
    buffers = AdmissionBuffers.allocate("cuda", vocab_size=32)
    buffers.stage(0, 2, RequestAdmission(4, 72, eos_token_ids=()), [6, 6, 6])
    buffers.admission_count.fill_(1)
    if fault == "duplicate":
        buffers.stage(1, 2, RequestAdmission(5, 73, eos_token_ids=()), [8])
        buffers.admission_count.fill_(2)
    elif fault == "negative_histogram":
        buffers.admission_histogram[0, 31] = -1
    elif fault == "stale_epoch":
        buffers.admission_generations[0] = 3
    elif fault == "zero_temperature":
        buffers.admission_parameters[0, 0] = 0.
    elif fault == "negative_zero_temperature":
        buffers.admission_parameters[0, 0] = -0.
    else:
        buffers.admission_parameters[0, 0] = float("nan")
    _, _, _, records, frame, sequence = _metadata()
    admission = PreparedAdmission(state, buffers, completion_outputs(records), frame, sequence)
    admission.prepare()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        admission()
    before = {field: getattr(state, field).clone() for field in state.__dataclass_fields__}
    worker = _worker({"admission": graph}, {"records": records}, (state, admission))
    try:
        _replay(worker)
        for field, expected in before.items():
            assert torch.equal(getattr(state, field), expected), field
        assert records[0, 2, 5] == ERROR
        assert records[0, 2, 6] == (DUPLICATE_SLOT if fault == "duplicate" else INVALID_ADMISSION)
        assert records[0, 2, 4] == -1
    finally:
        worker.close()
