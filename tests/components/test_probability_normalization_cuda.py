# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Vocabulary-tiled normalization preserves filtering and row validity under native replay."""

import os

import pytest
import torch

from ignition.components.probability_normalization import PreparedProbabilityNormalization
from ignition.preparation.compile_pool import CompilePool
from ignition.preparation.graphs import PreparedGraphBundle

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an assigned CUDA device")


class NormalizationForward:
    def __init__(self, operation, probabilities, params):
        self.operation, self.probabilities, self.params = operation, probabilities, params

    def __call__(self):
        return self.operation(self.probabilities, self.params)


@pytest.mark.parametrize("vocabulary", [17, 65537, 129280])
@torch.no_grad()
def test_probability_normalization_metadata_and_native_replay(vocabulary, tmp_path):
    device = torch.device("cuda", 0)
    torch.manual_seed(4813)
    probabilities = torch.randn((2, 4, vocabulary), device=device).softmax(-1)
    probabilities[0, 1].fill_(float("nan"))
    params = torch.zeros((2, 6), device=device)
    params[:, 0] = .8
    valid = torch.ones((2, 4), dtype=torch.bool, device=device)
    valid[0, 1] = False
    incoming = valid.clone()
    output = torch.empty_like(probabilities)
    operation = PreparedProbabilityNormalization(output, valid)
    forward = NormalizationForward(operation, probabilities, params)
    path = tmp_path / "probabilities.bin"
    with CompilePool(device) as compiler:
        compiler.submit(forward, path).result()
    with path.open("rb") as file:
        compiled = torch.compiler.load_compiled_function(file)
    compiled(forward)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(forward)
    bundle = PreparedGraphBundle(device=0, graphs={"normalization": graph},
        buffers={"probabilities": probabilities, "params": params, "output": output, "valid": valid},
        resources=(operation, forward, compiled))
    worker = bundle.start_worker()
    try:
        ids = torch.arange(vocabulary, device=device)
        for minimum in ((0., .05), (1.1, float("nan")), (0., 0.)):
            params[:, 2].copy_(torch.tensor(minimum, device=device))
            valid.copy_(incoming)
            output.fill_(float("nan"))
            operation.maxima.fill_(float("nan"))
            operation.masses.fill_(float("nan"))
            operation.totals.fill_(float("nan"))
            maximum = probabilities.amax(-1, keepdim=True)
            eligible = (probabilities > 0) & (probabilities >= maximum * params[:, None, 2:3])
            filtered = torch.where(eligible, probabilities, 0.)
            mass = filtered.sum(-1, keepdim=True)
            expected_valid = incoming & torch.isfinite(mass.squeeze(-1)) & (mass.squeeze(-1) > 0)
            expected = filtered / mass.clamp_min(torch.finfo(torch.float32).tiny)
            expected = torch.where(expected_valid[:, :, None], expected, (ids == 0).float())
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            ticket = worker.submit(0, repetitions=3)
            completion = worker.poll(timeout_ms=120000)
            assert completion is not None and completion.ticket == ticket and completion.error is None
            after = torch.cuda.memory_stats()
            for key in ("allocation.all.allocated", "allocation.all.freed",
                        "segment.all.allocated", "segment.all.freed"):
                assert before[key] == after[key]
            torch.testing.assert_close(valid, expected_valid, rtol=0, atol=0)
            torch.testing.assert_close(output, expected, rtol=2e-6, atol=1e-7)
    finally:
        worker.close()
        graph.reset()


@pytest.mark.parametrize("temperature", [.7, 1.])
@torch.no_grad()
def test_nucleus_distributions_match_sorted_probability_reference(temperature):
    from ignition.components.sampling import DeviceRequestState, RequestAdmission
    from ignition.components.speculation import PreparedSpeculativeDistributions
    from ignition.preparation.compilation import compile_forward

    vocabulary = 129280
    torch.manual_seed(4831)
    logits = torch.randn((1, 4, vocabulary), device="cuda") * 3
    state = DeviceRequestState.allocate("cuda", vocab_size=vocabulary)
    state.admit(0, RequestAdmission(generation=1, seed=17, max_new_tokens=100,
        temperature=temperature, top_k=0, top_p=.95, min_p=0, eos_token_ids=()), [])
    slots = torch.tensor([0], device="cuda", dtype=torch.int32)
    generations = torch.tensor([1], device="cuda", dtype=torch.int64)
    live = torch.tensor([1], device="cuda", dtype=torch.int32)
    candidates = torch.tensor([[3, 4, 5]], device="cuda", dtype=torch.int64)
    operation = PreparedSpeculativeDistributions(logits, state, slots, generations, live, candidates)
    operation.prepare()
    compiled = compile_forward(operation)
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    graph.replay()

    probabilities = (logits.double() / temperature).softmax(-1)
    ordered = probabilities.sort(descending=True).values
    crossing = (ordered.cumsum(-1) < .95).sum(-1, keepdim=True)
    cutoff = ordered.gather(-1, crossing)
    expected = torch.where(probabilities >= cutoff, probabilities, 0.)
    expected /= expected.sum(-1, keepdim=True)
    assert operation.valid.all()
    torch.testing.assert_close(operation.probabilities, expected.float(), rtol=2e-5, atol=1e-8)
    assert state.parameters[0, 1].item() == pytest.approx(.95)
    graph.reset()
