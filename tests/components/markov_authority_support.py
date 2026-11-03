# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Native authoritative acceptance with divergent full proposal probabilities."""

import json
from datetime import timedelta
from types import SimpleNamespace
import torch
import torch.distributed as dist


def rank_main(rank, rendezvous, output):
    from ignition.components.markov_sampling import PreparedMarkovProposals, PreparedDraftTokenBroadcast
    from ignition.components.speculation import (
        SpeculativeAcceptance,
        PreparedStochasticVerification,
        PreparedSpeculativeCommit,
        SpeculativeRngState,
        DRAFT_RNG_DOMAIN,
    )
    from ignition.components.parallel_draft import PreparedParallelDraftRound
    from ignition.components.sampling import DeviceRequestState, RequestAdmission, RUNNING, ERROR
    from ignition.components.completion import CompletionOutputs
    from ignition.preparation.collectives import CollectivePreparation, create_result_broadcast_runtime
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.graphs import PreparedGraphBundle

    class Pipeline:
        def __init__(self, proposal, broadcast, verify, commit, rng_owner, pending, inputs, poison):
            self.proposal, self.broadcast, self.verify, self.commit = (proposal, broadcast, verify, commit)
            self.rng_owner, self.pending, self.inputs, self.poison = (rng_owner, pending, inputs, poison)

        def __call__(self):
            o = self.rng_owner.owner
            self.pending.copy_(o.requests.pending_token[self.proposal.slots.long()])
            self.rng_owner.base_offset.copy_(o.offset)
            self.proposal()
            self.broadcast.finish(self.rng_owner.active, self.commit.token_limits)
            if rank == 1:
                self.proposal.probabilities.copy_(
                    torch.where(self.poison, self.proposal.probabilities * float("nan"), self.proposal.probabilities)
                )
            self.inputs[:, 0].copy_(self.pending)
            self.inputs[:, 1:].copy_(self.proposal.tokens)
            PreparedParallelDraftRound.commit_draft_rng(self.rng_owner)
            self.verify()
            self.commit()

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(minutes=15))
    coordinator = CollectivePreparation(dist.group.WORLD)
    b, d, v, r = 1, 7, 1024, 256
    c = 1
    runtime = create_result_broadcast_runtime(
        dist.group.WORLD, device, layouts=(SpeculativeAcceptance.broadcast_layout(b, d + 1),)
    )
    worker = None
    graph = None
    try:
        gen = torch.Generator().manual_seed(717 + rank)
        embed = (torch.randn((v, r), generator=gen) / 8).to(device, torch.bfloat16)
        head = (torch.randn((v, r), generator=gen) / 8).to(device, torch.bfloat16)
        initial_head = head.clone()
        base = torch.randn((b, d, v), generator=gen).to(device, torch.bfloat16)
        state = DeviceRequestState.allocate(device, vocab_size=v, capacity=c)
        state.admit(0, RequestAdmission(generation=7, seed=853, max_new_tokens=8192, eos_token_ids=()), [1, 2])
        slots = torch.zeros(b, device=device, dtype=torch.int32)
        epochs = torch.full((b,), 7, device=device, dtype=torch.int64)
        live = torch.tensor([b], device=device, dtype=torch.int32)
        pending = torch.empty(b, device=device, dtype=torch.int64)
        tokens = torch.empty((b, d), device=device, dtype=torch.int64)
        q = torch.empty((b, d, v), device=device)
        valid = torch.empty((b, d), device=device, dtype=torch.bool)
        sampled = torch.empty(b, device=device, dtype=torch.int64)
        seeds = state.seed.view(torch.int64) ^ DRAFT_RNG_DOMAIN
        offset = torch.zeros(c, device=device, dtype=torch.int64)
        base_offset = torch.zeros_like(offset)
        owner = SimpleNamespace(
            requests=state,
            offset=offset,
            generation=torch.full_like(offset, 7),
            inputs=SimpleNamespace(row_slot_ids=slots),
        )
        inputs = torch.empty((b, d + 1), device=device, dtype=torch.int64)
        active = torch.ones(b, device=device, dtype=torch.bool)
        limits = torch.full((b,), d + 1, device=device, dtype=torch.int32)
        rng_owner = SimpleNamespace(owner=owner, batch=b, width=d + 1, active=active, base_offset=base_offset)
        target = torch.full((b, d + 1, v), 1 / v, device=device)
        rng = SpeculativeRngState.allocate(device, capacity=c)
        rng.generation.fill_(7)
        completion = CompletionOutputs.allocate((2, c, 32), device=device)
        frame = torch.zeros(1, device=device, dtype=torch.int32)
        sequence = torch.ones(1, device=device, dtype=torch.int64)
        poison = torch.zeros(1, device=device, dtype=torch.bool)
        with RecordedPreparationSession(
            device=device,
            checkpoint="full-q-authority",
            workload_schema="full-q-authority-v1",
            numerical_policy="default",
            coordinator=coordinator,
        ) as session:
            proposal = PreparedMarkovProposals(
                base, embed, head, pending, tokens, q, valid, state, slots, epochs, live, seeds, base_offset
            )
            proposal.prepare()
            broadcast = PreparedDraftTokenBroadcast(
                tokens,
                valid,
                sampled,
                runtime=runtime,
                session=session,
                coordinator=coordinator,
                name="canonical_tokens",
            )
            verify = PreparedStochasticVerification(
                q, tokens, target, state, slots, epochs, live, rng=rng, draft_valid=valid, token_limits=limits
            )
            verify.prepare()
            verify.prepare_broadcast(
                runtime=runtime, session=session, coordinator=coordinator, name="canonical_acceptance"
            )
            commit = PreparedSpeculativeCommit(
                state,
                verify.output,
                inputs,
                slots,
                epochs,
                live,
                completion,
                frame,
                sequence,
                logits=target,
                rng=rng,
                token_limits=limits,
                draft_context_lag=0,
            )
            commit.prepare()
            pipeline = Pipeline(proposal, broadcast, verify, commit, rng_owner, pending, inputs, poison)
            session.tune(show_progress=False)
            call = compile_forward(pipeline)
            call()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                call()
            session.freeze()
            bundle = PreparedGraphBundle(
                device=rank, graphs={"authority": graph}, buffers={}, resources=(session, pipeline, call, runtime)
            )
            worker = bundle.start_worker()
            records = []
            for mode in (
                "divergent",
                "peer_nan_q",
                "peer_nan_weights",
                "root_nan_weights",
                "prefix_limits",
                "inactive",
            ):
                head.copy_(initial_head)
                poison.fill_(mode == "peer_nan_q")
                if (mode == "peer_nan_weights" and rank == 1) or (mode == "root_nan_weights" and rank == 0):
                    head.fill_(float("nan"))
                active.fill_(mode != "inactive")
                live.fill_(0 if mode == "inactive" else b)
                state.status.fill_(RUNNING)
                state.pending_token.fill_(2)
                state.consumed_tokens.fill_(2)
                state.generated_tokens.zero_()
                offset.fill_(2**32 + 28)
                owner.generation.fill_(7)
                rng.target_offset.zero_()
                rng.generation.fill_(7)
                for repeat in range(12):
                    if mode == "root_nan_weights":
                        state.status.fill_(RUNNING)
                    limits.fill_(1 + repeat % 8 if mode == "prefix_limits" else d + 1)
                    sequence.add_(1)
                    frame.fill_(repeat % 2)
                    completion.records.fill_(-71)
                    completion.speculative_survival.fill_(-71)
                    # Every scratch value must be produced by this replay.
                    for value in proposal.scratch.values():
                        value.fill_(False if value.dtype == torch.bool else 213)
                    q.fill_(float("nan"))
                    torch.cuda.synchronize()
                    before = offset.clone()
                    allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
                    ticket = worker.submit(0)
                    result = worker.poll(10000)
                    assert result is not None and result.error is None and result.ticket == ticket
                    assert allocations == torch.cuda.memory_stats()["allocation.all.allocated"]
                    assert torch.equal(offset, before + (0 if mode == "inactive" else 4 * d))
                    fields = [
                        tokens.cpu().tolist(),
                        valid.cpu().tolist(),
                        completion.records.cpu().tolist(),
                        completion.speculative_survival.cpu().tolist(),
                        state.status.cpu().tolist(),
                        state.pending_token.cpu().tolist(),
                        rng.target_offset.cpu().tolist(),
                        *[t.cpu().tolist() for t in vars(verify.output).values()],
                    ]
                    peers = [None, None]
                    dist.all_gather_object(peers, fields)
                    assert peers[0] == peers[1], (mode, repeat, rank)
                    if mode != "inactive":
                        if mode == "root_nan_weights":
                            assert (state.status == ERROR).all()
                            assert (rng.target_offset == 0).all()
                        else:
                            assert (state.status == RUNNING).all()
                        surv = completion.speculative_survival[repeat % 2, 0]
                        visible = surv[surv >= 0]
                        assert torch.isfinite(visible).all() and (visible <= 1).all()
                        assert (visible[1:] <= visible[:-1]).all()
                records.append(dict(mode=mode, replays=12, canonical_commit=True))
                print(rank, mode, "PASS", flush=True)
            if rank == 0:
                output.write_text(json.dumps(records, indent=2) + "\n")
    finally:
        if worker is not None:
            worker.close()
        if graph is not None:
            graph.reset()
        runtime.close()
        dist.destroy_process_group()
