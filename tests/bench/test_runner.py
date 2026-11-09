# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-context workload construction and latency reduction."""

import math

import pytest

from ignition.bench.runner import BenchmarkCase, prompt_tokens, summarize_case


def test_fifteen_draft_round_selects_and_packs_native_graph():
    from ignition.bench.runner import _round
    from ignition.direct.scheduler import Scheduler
    from tests.direct.test_scheduler import layout, records, request, uploads

    catalog = layout(max_active=1, widths=(1, 8, 16), graphs={
        ("init", 1, 1): "init1", ("prefill", 1, 512): "prefill512",
        ("speculative", 1, 8, 8, 7): "round8",
        ("speculative", 1, 16, 16, 15): "round16"})
    scheduler = Scheduler(catalog, prefix_cache=False)
    member = request(1, [10] * 512, limit=64)
    scheduler.submit(member)
    step = scheduler.plan()
    scheduler.complete(step, records(scheduler, step, {member: (1, 0, [], 0)}))
    step = scheduler.plan()
    scheduler.complete(step, records(scheduler, step, {member: (1, 512, [42], 0)}))
    step = _round(scheduler, BenchmarkCase(1, 16, 16, 512))
    assert (step.graph, step.width, step.packed_tokens, step.draft_tokens) == ("round16", 16, 16, 15)
    assert step.lanes[0].tokens == 16
    assert uploads(scheduler, step)["query_lengths"][0][0] == 16


@pytest.mark.parametrize("context", [0, 128, 1152])
def test_prefill_step_consumes_exact_chunk_after_prefix(context):
    from copy import deepcopy
    from ignition.bench.runner import PrefillCase, _prefill_step
    from ignition.direct.scheduler import Lane, Scheduler
    from tests.direct.test_scheduler import layout, records, request, uploads

    catalog = layout(max_active=1, prefill_chunk=512,
                     graphs={("init", 1, 1): "init1", ("prefill", 1, 128): "prefill128",
                             ("prefill", 1, 512): "prefill512"})
    scheduler = Scheduler(catalog, prefix_cache=False)
    member = request(1, [10] * (context + 512), limit=2)
    scheduler.submit(member)
    step = scheduler.plan()
    scheduler.complete(step, records(scheduler, step, {member: (1, 0, [], 0)}))
    while member.consumed < context:
        count = min(context - member.consumed, 512)
        step = scheduler._step("prefill", count, [Lane(member, member.consumed, count)])
        assert uploads(scheduler, step)["sample_enabled"][0][0] == 0
        scheduler.complete(step, records(scheduler, step, {member: (1, member.consumed + count, [], 0)}))
    baseline = deepcopy(scheduler, {id(catalog): catalog})
    for _ in range(2):
        scheduler = deepcopy(baseline, {id(catalog): catalog})
        step = _prefill_step(scheduler, PrefillCase(512, context))
        member = step.lanes[0].request
        assert (step.graph, step.width, step.lanes[0].start, step.lanes[0].tokens) == (
            "prefill512", 512, context, 512)
        assert uploads(scheduler, step)["sample_enabled"][0][0] == 1
        scheduler.complete(step, records(scheduler, step, {member: (1, context + 512, [42], 0)}))
        assert member.phase == "decode" and member.generated == [42]


def test_prefill_step_rejects_padded_graph():
    from ignition.bench.runner import PrefillCase, _prefill_step
    from ignition.direct.scheduler import Scheduler
    from tests.direct.test_scheduler import layout, records, request

    scheduler = Scheduler(layout(), prefix_cache=False)
    member = request(1, [10] * 384)
    scheduler.submit(member)
    step = scheduler.plan()
    scheduler.complete(step, records(scheduler, step, {member: (1, 0, [], 0)}))
    with pytest.raises(ValueError, match="without capacity padding"):
        _prefill_step(scheduler, PrefillCase(384, 0))


def test_case_estimate_keeps_raw_samples_and_uses_all_seeds():
    case = BenchmarkCase(2, 8, 16, 512)
    samples = [dict(seed=0, times_ms=[1., 3.]), dict(seed=1, times_ms=[2., 10.])]
    result = summarize_case(case, samples)
    assert result["median_ms"] == 2.5
    assert result["p25_ms"] == 1.75
    assert result["p75_ms"] == 4.75
    assert result["relative_iqr"] == 1.2
    assert result["samples"] is samples
    assert (result["batch"], result["packed_tokens"], result["context_tokens"]) == (2, 16, 512)


def test_eight_request_round_packs_every_lane_at_width_eight():
    from ignition.bench.runner import _round
    from ignition.direct.scheduler import Lane, Scheduler
    from tests.direct.test_scheduler import layout, records, request, uploads

    catalog = layout(capacity=8, max_active=8, widths=(1, 8), graphs={
        ("init", 8, 1): "init8", ("prefill", 1, 512): "prefill512",
        ("speculative", 8, 8, 64, 7): "round8x8"})
    scheduler = Scheduler(catalog, prefix_cache=False)
    members = [request(index + 1, [10] * 512, limit=64) for index in range(8)]
    for member in members:
        scheduler.submit(member)
    step = scheduler.plan()
    scheduler.complete(step, records(scheduler, step, {member: (1, 0, [], 0) for member in members}))
    for member in members:
        step = scheduler._step("prefill", 512, [Lane(member, 0, 512)])
        scheduler.complete(step, records(scheduler, step, {member: (1, 512, [42], 0)}))
    step = _round(scheduler, BenchmarkCase(8, 8, 64, 512))
    packed = uploads(scheduler, step)
    assert (step.graph, step.batch, step.width, step.packed_tokens, step.draft_tokens) == (
        "round8x8", 8, 8, 64, 7)
    assert [(lane.start, lane.tokens) for lane in step.lanes] == [(512, 8)] * 8
    assert packed["row_slot_ids"] == {0: list(range(8))}
    assert packed["query_lengths"] == {0: [8] * 8}
    assert packed["live_rows"] == {0: [8]}


@pytest.mark.parametrize("times", [[], [0.], [-1.], [math.inf], [math.nan]])
def test_missing_or_invalid_measurements_are_not_latency_estimates(times):
    with pytest.raises(ValueError, match="finite and positive"):
        summarize_case(BenchmarkCase(1, 8, 8, 512), [dict(seed=0, times_ms=times)])


@pytest.mark.parametrize("arguments", [(3, 8, 24, 512), (1, 0, 0, 512), (2, 8, 8, 512), (1, 8, 8, 0)])
def test_case_rejects_shapes_outside_complete_round_matrix(arguments):
    with pytest.raises(ValueError):
        BenchmarkCase(*arguments)


def test_prompts_are_exact_tokenized_prefixes_and_seeded():
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.train_from_iterator([
        "Read observations about database recovery, orbital mechanics, bounded queues and seasonal change. "
        "Record assumptions, measurements, uncertainty and independent conclusions."
    ], trainers.BpeTrainer(vocab_size=300, special_tokens=["[UNK]"]))
    first = prompt_tokens(tokenizer, 512, seed=0, lane=0)
    assert len(first) == 512
    assert first == prompt_tokens(tokenizer, 512, seed=0, lane=0)
    assert first != prompt_tokens(tokenizer, 512, seed=1, lane=0)
    assert first != prompt_tokens(tokenizer, 512, seed=0, lane=1)
    assert first == prompt_tokens(tokenizer, 2048, seed=0, lane=0)[:512]


def test_incomplete_results_preserve_partial_seed_and_have_no_aggregate():
    from ignition.bench.runner import measurement_result

    cases = (BenchmarkCase(1, 8, 8, 512), BenchmarkCase(2, 8, 16, 2048))
    measurements = [[dict(seed=0, times_ms=[1., 3.]), dict(seed=1, times_ms=[2., 4.])],
                    [dict(seed=0, times_ms=[5.])]]
    result = measurement_result(cases, measurements, warmup=1, repetitions=2, seeds=(0, 1),
                                status="incomplete", failure="RuntimeError: invalid completion")
    assert result["status"] == "incomplete" and result["ignore_eos"] is True
    assert "aggregate_ms" not in result
    assert [case["complete"] for case in result["cases"]] == [True, False]
    assert result["cases"][1]["samples"] == measurements[1]
    assert result["failure"] == "RuntimeError: invalid completion"
    with pytest.raises(ValueError, match="every case"):
        measurement_result(cases, measurements, warmup=1, repetitions=2, seeds=(0, 1), status="complete")


def test_complete_result_score_is_equal_weight_mean_of_case_medians():
    from ignition.bench.runner import measurement_result

    cases = (BenchmarkCase(1, 8, 8, 512), BenchmarkCase(2, 8, 16, 2048))
    result = measurement_result(cases, [[dict(seed=0, times_ms=[1., 3.])],
                                      [dict(seed=0, times_ms=[8., 10.])]],
                                warmup=1, repetitions=2, seeds=(0,), status="complete")
    assert result["aggregate_ms"] == 5.5
    assert all(case["complete"] for case in result["cases"])


@pytest.mark.parametrize("ignore_eos,count", [(False, 2), (True, 0)])
def test_request_stop_policy_packs_eos_count_without_changing_token_ids(ignore_eos, count):
    import numpy as np

    from ignition.direct.packing import pack
    from ignition.direct.scheduler import Lane, Layout, Request, SamplingParameters, Step
    from ignition.preparation import contract as c, runtime_package_pb2 as p

    bindings = [c.binding("admission_eos_count", p.EOS_COUNT, indexing=p.REQUEST_ROW),
                c.binding("admission_eos_token_ids", p.EOS_TOKEN_IDS, indexing=p.REQUEST_ROW)]
    layout = Layout(capacity=1, max_active=1, context=1024, page_tokens=128, table_width=8,
                    pages=8, snapshots=0, prefill_chunk=512, vocabulary=100, eos=(7, 9),
                    graphs={("init", 1, 1): "init1"}, bindings={"init1": bindings}, page_views={},
                    buffer_specs={"admission_eos_count": (np.dtype("int32"), (1,)),
                                  "admission_eos_token_ids": (np.dtype("int64"), (1, 8))})
    request = Request(1, [1, 2, 3], 16, SamplingParameters(), ignore_eos=ignore_eos)
    request.slot = 0
    uploads = {name: data for name, _, data in pack(layout, Step("init", "init1", 1, 1,
                                                                [Lane(request, 0, 1)], 1))}
    assert np.frombuffer(uploads["admission_eos_count"], dtype=np.int32).tolist() == [count]
    assert np.frombuffer(uploads["admission_eos_token_ids"], dtype=np.int64).tolist() == [7, 9, -1, -1, -1, -1, -1, -1]


def test_device_snapshot_restores_graph_mutations_and_aliased_views():
    import os

    if os.environ.get("FLARE_RUN_CUDA_TESTS") != "1":
        pytest.skip("set FLARE_RUN_CUDA_TESTS=1 on an assigned CUDA device")
    import torch

    from ignition.bench.executor import StateSnapshot

    stream = torch.cuda.Stream()
    values = torch.arange(16, dtype=torch.float32, device="cuda")
    counters = torch.arange(4, dtype=torch.int64, device="cuda")
    quantized = values.to(torch.float8_e4m3fn)
    stream.wait_stream(torch.cuda.current_stream())
    snapshot = StateSnapshot(dict(values=values, alias=values[::2], counters=counters, quantized=quantized), stream)
    expected = {name: value.cpu().clone() for name, value in
                (("values", values), ("counters", counters), ("quantized", quantized.view(torch.uint8)))}
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        values.add_(2)
        counters.add_(7)
        quantized.view(torch.uint8).fill_(0)
    for _ in range(3):
        with torch.cuda.stream(stream):
            graph.replay()
        snapshot.restore(stream)
        stream.synchronize()
        assert torch.equal(values.cpu(), expected["values"])
        assert torch.equal(counters.cpu(), expected["counters"])
        assert torch.equal(quantized.view(torch.uint8).cpu(), expected["quantized"])
    assert all(saved.device == source.device for source, saved in snapshot.entries.values())


def test_acceptance_observation_separates_verification_from_committed_clipping():
    from types import SimpleNamespace
    import torch

    from ignition.bench.runner import acceptance_observation

    acceptance = SimpleNamespace(
        offered_drafts=torch.tensor([3, 0]), accepted_drafts=torch.tensor([3, 0]),
        valid=torch.tensor([True, False]),
        survival=torch.tensor([[.75, .5, .25, -1, -1, -1, -1], [-1.] * 7]),
    )
    commit = SimpleNamespace(
        accepted_drafts=torch.tensor([1, 0]), emitted_count=torch.tensor([2, 0]),
        emitted_ids=torch.tensor([[11, 12, -1, -1], [-1, -1, -1, -1]]),
    )
    forward = SimpleNamespace(operations=(None, (), SimpleNamespace(output=acceptance), commit))
    result = acceptance_observation(forward)
    assert result["observations_per_request"] == 1
    assert result["requests"] == [
        dict(valid=True, offered_draft_count=3, accepted_draft_count=3,
             committed_draft_count=1, emitted_count=2, emitted_token_ids=[11, 12],
             proposal_survival=[.75, .5, .25]),
        dict(valid=False, offered_draft_count=0, accepted_draft_count=0,
             committed_draft_count=0, emitted_count=0, emitted_token_ids=[], proposal_survival=[]),
    ]
    sample = dict(seed=0, times_ms=[1.] * 20, acceptance=result)
    summary = summarize_case(BenchmarkCase(1, 8, 8, 8192), [sample])
    assert summary["samples"][0]["acceptance"]["observations_per_request"] == 1


def test_width_one_preserves_proposal_rng_and_advances_target_rng():
    from types import SimpleNamespace as NS
    import torch
    from ignition.bench.runner import _check_metadata

    metadata = NS(live_counts=torch.tensor([1, 1]), starts=torch.tensor([512]),
                  lengths=torch.tensor([1]), query_start_loc=torch.tensor([0, 1]))
    owner = NS(generation=torch.tensor([2]), offset=torch.tensor([28]),
               rng=NS(generation=torch.tensor([3]), target_offset=torch.tensor([4])))
    forward = NS(metadata=metadata, draft=None, owner=owner,
                 operations=(None, (), NS(draw_reservation=4), None))
    model = NS(benchmark_rounds={"round1": forward})
    step = NS(graph="round1", lanes=[NS(request=NS(slot=0, generation=3))], draft_tokens=0)
    case = BenchmarkCase(1, 1, 1, 512)
    _check_metadata(model, step, case, proposal_before=[(2, 28)])
    owner.offset.add_(4)
    with pytest.raises(RuntimeError, match="without proposals changed"):
        _check_metadata(model, step, case, proposal_before=[(2, 28)])
    owner.offset.sub_(4)
    owner.rng.target_offset.zero_()
    with pytest.raises(RuntimeError, match="draw reservations"):
        _check_metadata(model, step, case, proposal_before=[(2, 28)])
