# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Local sampling tensors preserve numerical results across captured graph shapes."""

import os
from types import SimpleNamespace

import pytest
import torch

from ignition.components.draft_sampling import PreparedDraftDistribution
from ignition.components.sampling import DeviceRequestState, RequestAdmission
from ignition.components.speculation import (
    PreparedSpeculativeDistributions, PreparedStochasticVerification, SpeculativeRngState,
)
from ignition.preparation.graphs import PreparedGraphBundle

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned CUDA device")


def scratch_tensors(layout, device):
    return {name: torch.empty(shape, dtype=dtype, device=device) for name, (shape, dtype) in layout.items()}


def copy_acceptance(outputs, acceptance):
    for name, destination in outputs.items():
        destination.copy_(getattr(acceptance, name))


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("vocab,batch_limit,width_limit", [
    (10000, 4, 6), (129280, 4, 8), (129280, 32, 8), (248320, 8, 16),
])
def test_native_local_sampling_matches_independent_operands_across_graph_shapes(
        compiled, vocab, batch_limit, width_limit):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    device = torch.device("cuda:0")
    state = DeviceRequestState.allocate(device, vocab_size=vocab, capacity=batch_limit)
    for slot in range(batch_limit):
        state.admit(slot, RequestAdmission(generation=7, seed=93 + slot,
            max_new_tokens=100, temperature=1. if vocab == 129280 else .8,
            top_k=0 if vocab == 129280 else 20, top_p=.95 if vocab == 129280 else .9,
            repetition_penalty=1.1, presence_penalty=.2, eos_token_ids=()), [1, 2])
    state.status[1] = 0
    slots = torch.tensor([3, 0, 2, 1, *range(4, batch_limit)], dtype=torch.int32, device=device)
    epochs = torch.tensor([7, 7, 99, 7, *([7] * (batch_limit - 4))], dtype=torch.int64, device=device)
    live = torch.tensor([batch_limit], dtype=torch.int32, device=device)
    owners, graphs = [], {}
    generator = torch.Generator().manual_seed(17)
    for batch, width in ((batch_limit, width_limit), (1, 1), (batch_limit // 2, 3), (1, width_limit)):
        logits = torch.randn((batch, width, vocab), generator=generator).to(device)
        proposal_logits = torch.randn((batch, 1, vocab), generator=generator).to(device)
        for local in (False, True):
            ids = torch.zeros((batch, width - 1), dtype=torch.int64, device=device)
            probabilities = torch.empty((batch, width - 1, vocab), device=device)
            valid = torch.zeros_like(ids, dtype=torch.bool)
            seed = torch.arange(batch, dtype=torch.int64, device=device) + 51
            offset, sampled = torch.zeros_like(seed), torch.zeros_like(seed)
            proposals = tuple(PreparedDraftDistribution(proposal_logits, state, slots[:batch], epochs[:batch], live, ids,
                prefix_offset=index, seed=seed, offset=offset, sampled=sampled) for index in range(width - 1))
            for proposal in proposals:
                proposal.prepare()
            target = PreparedSpeculativeDistributions(logits, state, slots[:batch], epochs[:batch], live, ids)
            target.prepare()
            verify = PreparedStochasticVerification(probabilities, ids, target.probabilities, state,
                slots[:batch], epochs[:batch], live, rng=SpeculativeRngState.allocate(device, capacity=batch_limit),
                draft_valid=valid, target_valid=target.valid)
            verify.prepare()
            acceptance_outputs = {name: torch.empty_like(value) for name, value in vars(verify.output).items()}
            result = dict(ids=torch.empty_like(ids), probabilities=torch.empty_like(probabilities),
                target=torch.empty_like(target.probabilities))
            reference = (None if local else dict(
                proposals=tuple(scratch_tensors(p.layout, device) for p in proposals),
                target=scratch_tensors(target.layout, device), verify=scratch_tensors(verify.layout, device)))
            for operation in (*proposals, target, verify):
                operation.functional()
            owner = SimpleNamespace(proposals=proposals, target=target, verify=verify, seed=seed,
                logits=logits, proposal_logits=proposal_logits, batch=batch, width=width, vocab=vocab,
                reference=reference, result=result, acceptance=acceptance_outputs)

            def run(owner=owner):
                ids = torch.zeros((owner.batch, owner.width - 1), dtype=torch.int64, device=owner.logits.device)
                probabilities = torch.empty((owner.batch, owner.width - 1, owner.vocab), device=owner.logits.device)
                valid = torch.empty_like(ids, dtype=torch.bool)
                for index, proposal in enumerate(owner.proposals):
                    scratch = (scratch_tensors(proposal.layout, owner.logits.device) if owner.reference is None
                               else owner.reference["proposals"][index])
                    sampled = torch.empty_like(owner.seed)
                    q = proposal.run(scratch, owner.proposal_logits, candidates=ids, seed=owner.seed,
                        offset=torch.full_like(owner.seed, index * 4), sampled=sampled)
                    ids[:, index].copy_(sampled)
                    probabilities[:, index].copy_(q[:, 0])
                    valid[:, index].copy_(scratch["valid"][:, 0])
                target_scratch = (scratch_tensors(owner.target.layout, owner.logits.device) if owner.reference is None
                                  else owner.reference["target"])
                verify_scratch = (scratch_tensors(owner.verify.layout, owner.logits.device) if owner.reference is None
                                  else owner.reference["verify"])
                owner.target.run(target_scratch, owner.logits, candidates=ids)
                acceptance = owner.verify.run(verify_scratch, probabilities, target_scratch["probabilities"],
                    target_scratch["valid"], drafts=ids, draft_valid=valid, disjoint=True)
                owner.result["ids"].copy_(ids)
                owner.result["probabilities"].copy_(probabilities)
                owner.result["target"].copy_(target_scratch["probabilities"])
                copy_acceptance(owner.acceptance, acceptance)

            call = run
            if compiled:
                from ignition.preparation.compilation import compile_forward
                call = compile_forward(run)
            call()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                call()
            owners.append((owner, call))
            graphs[f"{batch}x{width}-{local}"] = graph
    bundle = PreparedGraphBundle(device=0, graphs=graphs, buffers={}, resources=(owners, state))
    worker = bundle.start_worker()
    try:
        for pair in (0, 3, 1, 0, 2, 1, 2, 3, 0):
            results = []
            for local in (False, True):
                index = pair * 2 + int(local)
                if local:
                    reference = owners[index - 1][0].reference
                    tensors = [*reference["target"].values(), *reference["verify"].values(),
                        *(tensor for scratch in reference["proposals"] for tensor in scratch.values())]
                    for tensor in tensors:
                        tensor.fill_(float("nan") if tensor.is_floating_point() else 255)
                torch.cuda.synchronize()
                allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
                ticket = worker.submit(index)
                completion = worker.poll(timeout_ms=30000)
                assert completion is not None and completion.ticket == ticket and completion.error is None
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
                owner, _ = owners[index]
                results.append([value.cpu().clone() for value in (*owner.result.values(), *owner.acceptance.values())])
            for independent, local in zip(*results, strict=True):
                torch.testing.assert_close(local, independent, rtol=0, atol=0)
    finally:
        worker.close()
        for graph in graphs.values():
            graph.reset()


@pytest.mark.parametrize("compiled", [False, True])
def test_local_target_and_verification_scratch_match_independent_operands(compiled):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    batch, width, vocab = 4, 3, 256
    device = torch.device("cuda:0")
    state = DeviceRequestState.allocate(device, vocab_size=vocab, capacity=batch)
    for slot in range(batch):
        state.admit(slot, RequestAdmission(generation=7, seed=93 + slot,
            top_k=20, top_p=.9, repetition_penalty=1.1, presence_penalty=.2), [1, 2])
    state.status[1] = 0
    slots = torch.arange(batch, dtype=torch.int32, device=device)
    generations = torch.full((batch,), 7, dtype=torch.int64, device=device)
    live = torch.tensor([batch], dtype=torch.int32, device=device)
    logits = torch.randn((batch, width, vocab), device=device)
    ids = torch.zeros((batch, width - 1), dtype=torch.int64, device=device)
    draft = torch.full((batch, width - 1, vocab), 1. / vocab, device=device)
    limits = torch.tensor([3, 2, 1, 3], dtype=torch.int32, device=device)
    rng = SpeculativeRngState.allocate(device, capacity=batch)
    calls, outputs = [], []
    for local in (False, True):
        target = PreparedSpeculativeDistributions(logits, state, slots, generations, live, ids)
        target.prepare()
        verify = PreparedStochasticVerification(draft, ids, target.probabilities, state, slots,
            generations, live, rng=rng, target_valid=target.valid, token_limits=limits)
        verify.prepare()
        result = {name: torch.empty_like(value) for name, value in vars(verify.output).items()}
        reference = None if local else (scratch_tensors(target.layout, device), scratch_tensors(verify.layout, device))
        target.functional()
        verify.functional()

        def run(target=target, verify=verify, reference=reference, result=result):
            target_scratch, verify_scratch = ((scratch_tensors(target.layout, logits.device),
                scratch_tensors(verify.layout, logits.device)) if reference is None else reference)
            target.run(target_scratch, logits, candidates=ids)
            acceptance = verify.run(verify_scratch, draft, target_scratch["probabilities"],
                target_scratch["valid"], drafts=ids, draft_valid=torch.ones_like(ids, dtype=torch.bool), disjoint=True)
            copy_acceptance(result, acceptance)

        if compiled:
            from ignition.preparation.compilation import compile_forward
            run = compile_forward(run)
        run()
        calls.append(run)
        outputs.append(result)
    graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
    try:
        for graph, call in zip(graphs, calls, strict=True):
            with torch.cuda.graph(graph):
                call()
        for limit in (3, 1, 2):
            limits[0] = limit
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            for graph in graphs:
                graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            for name in outputs[0]:
                torch.testing.assert_close(outputs[0][name], outputs[1][name], rtol=0, atol=0)
    finally:
        for graph in graphs:
            graph.reset()


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("source_logits", [False, True])
def test_local_proposal_scratch_matches_independent_operands(compiled, source_logits):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    batch, vocab = 4, 256
    device = torch.device("cuda:0")
    state = DeviceRequestState.allocate(device, vocab_size=vocab, capacity=batch)
    for slot in range(batch):
        state.admit(slot, RequestAdmission(generation=7, seed=93 + slot, top_k=20, top_p=.9), [1, 2])
    state.status[1] = 0
    slots = torch.arange(batch, dtype=torch.int32, device=device)
    generations = torch.full((batch,), 7, dtype=torch.int64, device=device)
    live = torch.tensor([batch], dtype=torch.int32, device=device)
    ids = torch.zeros((batch, 1), dtype=torch.int64, device=device)
    logits = torch.randn((batch, 1, vocab), device=device)
    seed = torch.arange(batch, dtype=torch.int64, device=device) + 23
    offset = torch.zeros_like(seed)
    calls, outputs = [], []
    for local in (False, True):
        proposal = PreparedDraftDistribution(logits, state, slots, generations, live, ids,
            seed=seed, offset=offset, sampled=torch.zeros_like(seed), prefix_offset=1)
        if source_logits:
            proposal.bind_logits(logits[:, 0])
        proposal.prepare()
        result = dict(probabilities=torch.empty_like(proposal.probabilities),
            valid=torch.empty_like(proposal.valid), sampled=torch.empty_like(seed))
        reference = None if local else scratch_tensors(proposal.layout, device)
        proposal.functional()

        def run(proposal=proposal, reference=reference, result=result):
            scratch = scratch_tensors(proposal.layout, logits.device) if reference is None else reference
            sampled = torch.empty_like(seed)
            probabilities = proposal.run(scratch, logits, source_logits=logits[:, 0] if source_logits else None,
                candidates=ids, seed=seed, offset=offset, sampled=sampled)
            result["probabilities"].copy_(probabilities)
            result["valid"].copy_(scratch["valid"])
            result["sampled"].copy_(sampled)

        if compiled:
            from ignition.preparation.compilation import compile_forward
            run = compile_forward(run)
        run()
        calls.append(run)
        outputs.append(result)
    graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
    try:
        for graph, call in zip(graphs, calls, strict=True):
            with torch.cuda.graph(graph):
                call()
        for count in (batch, batch - 1, batch):
            live.fill_(count)
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            for graph in graphs:
                graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            for name in outputs[0]:
                torch.testing.assert_close(outputs[0][name], outputs[1][name], rtol=0, atol=0)
    finally:
        for graph in graphs:
            graph.reset()
