# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Native proposal/rejection frequencies with independently filtered target policies."""

import json
import math
import torch
from ignition.components.markov_sampling import PreparedMarkovProposals
from ignition.components.sampling import DeviceRequestState, RUNNING
from ignition.components.speculation import PreparedStochasticVerification, SpeculativeRngState, DRAFT_RNG_DOMAIN
from ignition.preparation.compilation import compile_forward
from ignition.preparation.graphs import PreparedGraphBundle


class Pipeline:
    def __init__(self, proposal, verify, draft_count, target_count):
        self.proposal, self.verify = proposal, verify
        self.draft_count, self.target_count = draft_count, target_count
        self.ones = torch.ones(proposal.batch, device="cuda", dtype=torch.int64)

    def __call__(self):
        self.proposal()
        self.verify()
        self.draft_count.scatter_add_(0, self.proposal.tokens[:, 0], self.ones)
        self.target_count.scatter_add_(0, self.verify.output.token_ids[:, 0], self.ones)


def run(output, *, fp8=False, policy_names=None):
    b, d, v, r = 32, 1, 8, 64
    torch.cuda.set_device(0)
    state = DeviceRequestState.allocate("cuda", vocab_size=v, capacity=b)
    state.status.fill_(RUNNING)
    state.generation.fill_(7)
    state.seed.copy_(torch.arange(b, device="cuda", dtype=torch.int64) + 423894)
    state.parameters.copy_(torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0, 0.0], device="cuda"))
    slots = torch.arange(b, device="cuda", dtype=torch.int32)
    generations = torch.full((b,), 7, device="cuda", dtype=torch.int64)
    live = torch.tensor([b], device="cuda", dtype=torch.int32)
    logits = (
        torch.tensor([2.0, 1.0, 0.0, -1.0, -0.25, 0.75, -0.5, -1.5], dtype=torch.bfloat16, device="cuda")
        .expand(b, d, v)
        .contiguous()
    )
    embedding = torch.zeros((v, r), device="cuda", dtype=torch.bfloat16)
    head = torch.zeros_like(embedding)
    if fp8:
        from ignition.components.markov_sampling import fp8_markov_head

        head = fp8_markov_head(head)
    pending = torch.zeros(b, device="cuda", dtype=torch.int64)
    tokens = torch.empty((b, d), device="cuda", dtype=torch.int64)
    q = torch.empty((b, d, v), device="cuda")
    valid = torch.empty((b, d), device="cuda", dtype=torch.bool)
    seeds = state.seed.view(torch.int64) ^ DRAFT_RNG_DOMAIN
    offsets = torch.zeros_like(seeds)
    p = torch.empty((b, d + 1, v), device="cuda")
    p.fill_(1 / v)
    rng = SpeculativeRngState.allocate("cuda", capacity=b)
    rng.generation.fill_(7)
    draft_count = torch.zeros(v, device="cuda", dtype=torch.int64)
    target_count = torch.zeros_like(draft_count)
    records = []
    worker = None
    graph = None
    proposal = PreparedMarkovProposals(
        logits, embedding, head, pending, tokens, q, valid, state, slots, generations, live, seeds, offsets
    )
    proposal.prepare()
    verify = PreparedStochasticVerification(q, tokens, p, state, slots, generations, live, rng=rng, draft_valid=valid)
    verify.prepare()
    pipeline = Pipeline(proposal, verify, draft_count, target_count)
    call = compile_forward(pipeline)
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    bundle = PreparedGraphBundle(device=0, graphs={"law": graph}, buffers={}, resources=(pipeline, call))
    worker = bundle.start_worker()
    try:
        raw = torch.tensor([0.4, 1.2, -0.8, 0.5, 0.2, -1.5, 0.0, -0.3], dtype=torch.float64)
        soft = raw.softmax(0)
        topk = torch.where(raw >= raw.topk(2).values[-1], soft, 0.0)
        topk /= topk.sum()
        sorted_p, ids = soft.sort(descending=True)
        before = sorted_p.cumsum(0) - sorted_p
        cutoff = sorted_p[before < 0.65].min()
        topp = torch.where(soft >= cutoff, soft, 0.0)
        topp /= topp.sum()
        minp = torch.where(soft >= soft.max() * 0.3, soft, 0.0)
        minp /= minp.sum()
        penalized = ((raw - torch.tensor([0.5, 0.0, 0.2, 0.0, 0.4, 0.0, 0.0, 0.0])) / 0.55).softmax(0)
        policies = {
            "unfiltered": soft,
            "top_k_2": topk,
            "top_p_0.65": topp,
            "min_p_0.3": minp,
            "temperature_penalties": penalized,
            "point_mass": torch.nn.functional.one_hot(torch.tensor(5), v).double(),
        }
        if policy_names is not None:
            policies = {name: policies[name] for name in policy_names}
        steps = 4096
        n = steps * b
        alpha = 1e-7
        bound = math.sqrt(math.log(2 * v / alpha) / (2 * n))
        expected_q = logits[0, 0].double().softmax(0).cpu()
        for name, expected in policies.items():
            p.copy_(expected.to("cuda", torch.float32)[None, None, :])
            offsets.zero_()
            rng.target_offset.zero_()
            draft_count.zero_()
            target_count.zero_()
            torch.cuda.synchronize()
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            for _ in range(steps):
                offsets.add_(4)
                rng.target_offset.add_(verify.draw_reservation)
                torch.cuda.synchronize()
                ticket = worker.submit(0)
                result = worker.poll(10000)
                assert result is not None and result.error is None and result.ticket == ticket
            assert allocations == torch.cuda.memory_stats()["allocation.all.allocated"]
            draft_observed = draft_count.cpu().double() / n
            observed = target_count.cpu().double() / n
            assert int(draft_count.sum()) == n and int(target_count.sum()) == n
            assert (draft_observed - expected_q).abs().max() < bound
            assert (observed - expected).abs().max() < bound
            record = dict(
                policy=name,
                samples=n,
                simultaneous_hoeffding_bound=bound,
                expected=expected.tolist(),
                observed=observed.tolist(),
                draft_max_error=float((draft_observed - expected_q).abs().max()),
                target_max_error=float((observed - expected).abs().max()),
            )
            records.append(record)
            print(json.dumps(record), flush=True)
        output.write_text(json.dumps(records, indent=2) + "\n")
        return records
    finally:
        worker.close()
        graph.reset()
