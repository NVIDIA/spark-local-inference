# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A finite conditional language model composed with prepared request machinery.

The model has 32 vocabulary IDs and assigns positive mass to IDs 0, 1 and 2.
Its target transition table is FP32; a BF16 Markov head defines the actual
proposal distribution. Model state is the pending predecessor and the two
consumed predecessor IDs retained by the ordinary speculative commit.
"""

from types import SimpleNamespace

import torch

from ignition.components.causal_prefix import PreparedCausalPrefixAllocation
from ignition.components.completion import CompletionOutputs
from ignition.components.parallel_draft import DraftProposalState, PreparedParallelDraftRound
from ignition.components.sampling import (
    AdmissionBuffers, DeviceRequestState, FINISHED, PreparedAdmission, PreparedSampling, RequestAdmission,
)
from ignition.components.speculation import DRAFT_RNG_DOMAIN, PreparedSpeculativeCommit, SpeculativeRngState
from ignition.components.token_state import PreparedTokenState
from ignition.components.vocabulary_logprobs import PreparedVocabularyParallelLogprobs
from ignition.components.vocabulary_sampling import PreparedVocabularyBlockVerification, PreparedVocabularySoftmax
from ignition.preparation.compilation import compile_forward
from ignition.preparation.graphs import PreparedGraphBundle

TARGET = ((.70, .20, .10), (.10, .25, .65), (.35, .60, .05))
PROPOSAL = ((.05, .85, .10), (.75, .15, .10), (.10, .15, .75))


class ToyCommittedModel:
    """Own real admission, proposal, filtering, rejection, commit and logprobs."""

    def __init__(self, batch=4, *, token_capacity=None):
        self.batch, self.width, self.vocabulary = batch, 3, 32
        b, w, v, device = batch, self.width, self.vocabulary, torch.device("cuda", 0)
        self.requests = DeviceRequestState.allocate(device, vocab_size=v, capacity=b)
        self.requests.ple_history = torch.zeros(b, 2, device=device, dtype=torch.int64)
        self.completion = CompletionOutputs.allocate((2, b, 32), device=device)
        self.frame = torch.zeros(1, device=device, dtype=torch.int32)
        self.sequence = torch.ones(1, device=device, dtype=torch.int64)
        self.slots = torch.arange(b, device=device, dtype=torch.int32).roll(1)
        self.epochs = torch.ones(b, device=device, dtype=torch.int64)
        self.live_rows = torch.full((1,), b, device=device, dtype=torch.int32)
        self.rows = torch.arange(b, device=device)
        self.slot64 = torch.zeros(b, device=device, dtype=torch.int64)
        self.admission_buffers = AdmissionBuffers.allocate(device, vocab_size=v, capacity=b)
        self.admission = PreparedAdmission(self.requests, self.admission_buffers,
            self.completion, self.frame, self.sequence, capacity=b)
        self.admission.prepare()
        self.prompt = torch.zeros(b, device=device, dtype=torch.int64)
        self.prompt_starts = torch.arange(b + 1, device=device, dtype=torch.int32)
        self.token_state = PreparedTokenState(self.requests, self.prompt, self.prompt_starts,
            self.slots, self.epochs, self.live_rows, history=self.requests.ple_history)
        self.token_state.prepare()
        initial = torch.full((b, v), -float("inf"), device=device)
        initial[:, 0] = 0.
        self.initial = initial
        self.prefill_sampler = PreparedSampling(initial, self.requests, self.slots, self.epochs,
            self.live_rows, self.completion, self.frame, self.sequence)
        self.prefill_sampler.prepare()
        self.rng = SpeculativeRngState.allocate(device, capacity=b)
        self.owner = SimpleNamespace(requests=self.requests, inputs=SimpleNamespace(row_slot_ids=self.slots),
            generation=torch.zeros(b, device=device, dtype=torch.int64),
            offset=torch.zeros(b, device=device, dtype=torch.int64))
        self.active = torch.zeros(b, device=device, dtype=torch.bool)
        self.base_offset = torch.zeros(b, device=device, dtype=torch.int64)
        self.seed = torch.zeros_like(self.base_offset)
        self.pending = torch.zeros_like(self.base_offset)
        self.drafts = torch.zeros(b, w - 1, device=device, dtype=torch.int64)
        self.candidates = torch.zeros(b, w, device=device, dtype=torch.int64)
        self.valid = torch.zeros(b, w - 1, device=device, dtype=torch.bool)
        self.target_valid = torch.zeros(b, w, device=device, dtype=torch.bool)
        self.q = torch.zeros((b, w - 1, v), dtype=torch.float32, device=device)
        self.p = torch.zeros((b, w, v), dtype=torch.float32, device=device)
        base = torch.full((b, w - 1, v), -float("inf"), device=device, dtype=torch.bfloat16)
        base[:, :, :3] = 0.
        self.proposal_base = base
        self.embedding = torch.zeros(v, 64, device=device, dtype=torch.bfloat16)
        self.embedding[:, :v] = torch.eye(v, device=device, dtype=torch.bfloat16)
        self.head = torch.zeros_like(self.embedding)
        self.head[:3, :3] = torch.tensor(PROPOSAL, device=device).log().T.bfloat16()
        self.target_table = torch.full((v, v), -float("inf"), device=device)
        self.target_table[:, 0] = 0.
        self.target_table[:3, :3] = torch.tensor(TARGET, device=device).log()
        self.logits = torch.zeros(b, w, v, device=device)
        args = (self.requests, self.slots, self.epochs, self.live_rows)
        options = dict(start=0, runtime=None, session=None, coordinator=None)
        self.proposal = PreparedVocabularySoftmax(base, self.q, self.valid, *args, self.drafts,
            **options, name="toy.proposal", embedding=self.embedding, head=self.head,
            pending=self.pending, seeds=self.seed, offsets=self.base_offset, tokens=self.drafts)
        self.target = PreparedVocabularySoftmax(self.logits, self.p, self.target_valid, *args, self.drafts,
            **options, name="toy.target")
        self.confidence = torch.full((b, w - 1), .75, device=device)
        self.submitted = torch.zeros(b, device=device, dtype=torch.bool)
        self.source_valid = torch.zeros_like(self.submitted)
        self.prefix_limits = torch.full((b,), w, device=device, dtype=torch.int32)
        self.limits = torch.zeros(b, device=device, dtype=torch.int32)
        self.admitted = torch.zeros_like(self.submitted)
        self.allocator = PreparedCausalPrefixAllocation(self.confidence, self.slot64,
            self.submitted, self.source_valid, self.prefix_limits, self.limits, self.admitted,
            token_capacity=b * w if token_capacity is None else token_capacity, request_capacity=b)
        self.verify = PreparedVocabularyBlockVerification(self.q, self.drafts, self.p, *args,
            **options, name="toy.verify", rng=self.rng, draft_valid=self.valid,
            target_valid=self.target_valid, token_limits=self.limits)
        self.logprobs = PreparedVocabularyParallelLogprobs(self.logits, self.completion,
            self.slots, self.epochs, self.live_rows, self.frame, vocabulary=v, offset=0,
            runtime=None, session=None, coordinator=None, name="toy.logprobs")
        self.commit = PreparedSpeculativeCommit(self.requests, self.verify.output, self.candidates,
            self.slots, self.epochs, self.live_rows, self.completion, self.frame, self.sequence,
            logprobs=self.logprobs, rng=self.rng, token_limits=self.limits, draft_context_lag=0)
        for operation in (self.proposal, self.target, self.verify, self.commit):
            operation.prepare()
        self.committed = SimpleNamespace(**{name: torch.empty_like(getattr(self.commit, name))
                                            for name in self.commit._output_names})
        self.token_state.functional()
        self.prefill_sampler.functional(session=None)
        self.allocator.functional()
        for operation in (self.proposal, self.target, self.verify, self.commit):
            operation.functional(session=None)
        self.worker = None
        self.graphs, self.compiled = {}, {}

    commit_draft_rng = PreparedParallelDraftRound.commit_draft_rng

    def initialize(self):
        """Apply validated admission and the toy model's empty consumed state."""
        self.admission.validate()
        rows = torch.arange(self.batch, device=self.slots.device)
        selected = ((rows < self.admission_buffers.admission_count)
                    & (self.admission.errors == 0))
        owned = (selected[:, None] & (self.admission_buffers.admission_slots[:, None]
                                     == rows[None, :])).any(0)
        self.requests.consumed_tokens.masked_fill_(owned, 0)
        self.requests.ple_history.masked_fill_(owned[:, None], 0)
        self.admission.apply()

    def prefill(self):
        self.token_state.consume(self.prompt, self.prompt_starts)
        self.prefill_sampler.run(self.initial)

    def __call__(self):
        pending, before, _, active = self.token_state.gather()
        pending = pending.clamp_min(0)
        safe = self.slots.long().clamp(0, self.batch - 1)
        active = active.bool()
        base_offset = torch.where(self.owner.generation[safe] == self.requests.generation[safe],
                                  self.owner.offset[safe], 0)
        seed = self.requests.seed.view(torch.int64)[safe] ^ DRAFT_RNG_DOMAIN
        drafts = torch.zeros((self.batch, self.width - 1), dtype=torch.int64, device=pending.device)
        q, valid, _ = self.proposal.run(self.proposal_base, drafts, pending=pending,
                                      seeds=seed, offsets=base_offset, tokens=drafts)
        candidates = torch.cat((pending[:, None], drafts), 1)
        proposal = DraftProposalState(before, active, candidates, drafts, valid, base_offset, None)
        self.commit_draft_rng(proposal=proposal)
        finished = ((self.epochs == self.requests.generation[safe])
                    & (self.requests.status[safe] == FINISHED))
        submitted = (torch.arange(self.batch, device=pending.device) < self.live_rows) & ~finished
        source_valid = active & valid.all(-1)
        limits = torch.empty_like(self.prefix_limits)
        admitted = self.allocator(self.confidence, self.slots.long(), submitted, source_valid,
                                  self.prefix_limits, limits)
        logits = self.target_table[candidates]
        p, target_valid, _ = self.target.run(logits, drafts)
        acceptance = self.verify.run(q, drafts, p, valid, target_valid, token_limits=limits)
        acceptance.valid.logical_and_(admitted)
        committed = self.commit(logits, acceptance=acceptance, inputs=candidates, token_limits=limits)
        for name, value in dict(active=active, pending=pending, base_offset=base_offset, seed=seed,
                drafts=drafts, candidates=candidates, valid=valid, target_valid=target_valid,
                q=q, p=p, logits=logits, submitted=submitted, source_valid=source_valid,
                limits=limits, admitted=admitted).items():
            getattr(self, name).copy_(value)
        for name, destination in vars(self.committed).items():
            destination.copy_(getattr(committed, name))

    def stage(self, *, generation, seed=8751, max_new_tokens=20, eos=(), rows=None):
        rows = list(range(self.batch)) if rows is None else list(rows)
        for staged, row in enumerate(rows):
            self.admission_buffers.stage(staged, int(self.slots[row]), RequestAdmission(
                generation, seed + row, max_new_tokens=max_new_tokens, top_k=-1, top_p=1.,
                eos_token_ids=eos), [0])
            self.epochs[row] = generation
        self.admission_buffers.admission_count.fill_(len(rows))

    def capture(self, extra=None):
        forwards = dict(admit=self.initialize, prefill=self.prefill, round=self)
        if extra:
            forwards.update(extra)
        for name, operation in forwards.items():
            call = compile_forward(operation)
            call()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                call()
            self.compiled[name], self.graphs[name] = call, graph
        self.bundle = PreparedGraphBundle(device=0, graphs=self.graphs, buffers={},
            resources=(self, self.compiled, self.completion))
        self.worker = self.bundle.start_worker()

    def replay(self, name):
        torch.cuda.synchronize()
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        ticket = self.worker.submit(self.bundle.graph_indices[name])
        completion = self.worker.poll(30000)
        assert completion is not None and completion.error is None and completion.ticket == ticket
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before

    def checkpoint(self):
        tensors = [value for value in vars(self.requests).values() if isinstance(value, torch.Tensor)]
        tensors += [self.owner.generation, self.owner.offset, self.rng.generation, self.rng.target_offset]
        return [(value, value.clone()) for value in tensors]

    @staticmethod
    def restore(checkpoint):
        for value, saved in checkpoint:
            value.copy_(saved)

    def close(self):
        if self.worker is not None:
            self.worker.close()
            self.worker = None
        for graph in self.graphs.values():
            graph.reset()
