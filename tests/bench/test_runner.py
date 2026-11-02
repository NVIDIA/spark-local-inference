# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-context workload construction and latency reduction."""

import math

import pytest

from ignition.bench.runner import BenchmarkCase, prompt_tokens, summarize_case


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


@pytest.mark.parametrize("times", [[], [0.], [-1.], [math.inf], [math.nan]])
def test_missing_or_invalid_measurements_are_not_latency_estimates(times):
    with pytest.raises(ValueError, match="finite and positive"):
        summarize_case(BenchmarkCase(1, 8, 8, 512), [dict(seed=0, times_ms=times)])


@pytest.mark.parametrize("arguments", [(3, 8, 24, 512), (1, 4, 4, 512), (2, 8, 8, 512), (1, 8, 8, 0)])
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

    from ignition.bench.runner import StateSnapshot

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
