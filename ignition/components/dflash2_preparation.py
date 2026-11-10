# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared DFlash2 owners for captured target-verification rounds."""

from functools import partial
from types import SimpleNamespace

import torch

from .dflash2 import DraftScratch, PreparedContext, PreparedDraft
from .speculation import SpeculativeRngState


class DFlash2Buffers:
    def __init__(self, pool, shapes, captures, *, vocabulary, columns):
        from .parallel_draft import PreparedParallelDraftRound

        self.rounds = {}
        self.dummy = torch.zeros(vocabulary, dtype=torch.float32, device=pool.device)
        self.dummy[0] = 1
        retained = pool.cursor()
        for batch, width in sorted(set(shapes), key=lambda shape: (shape[0] * shape[1], shape[1]), reverse=True):
            cursor = pool.cursor()
            self.rounds[batch, width] = PreparedParallelDraftRound.allocate_buffers(
                batch, width, vocabulary, pool.device, buffers=cursor)
            retained.join(cursor)
        self.capture = retained.fork()
        self.stage = retained
        self.pool = pool


class DFlash2Preparation:
    def __init__(self, weights, pool, requests, inputs, scratch, *, decoder, embedding, head,
                 page_tokens, completion, frame, sequence, session, workspace, buffer_layout=None,
                 verification_width=8, study_width=None, runtime=None,
                 vocabulary_runtime=None, coordinator=None, block_size=None, gather_runtime=None,
                 local_metadata=False):
        width = verification_width if study_width is None else study_width
        if type(width) is not int or not 1 <= width <= 16:
            raise ValueError("DFlash2 verification capacity must be an integer in [1,16]")
        self.study_width, self.page_tokens = study_width, page_tokens
        self.local_metadata = local_metadata
        self.weights, self.tables, self.pool = weights, None, pool
        self.requests, self.inputs, self.scratch = requests, inputs, scratch
        self.completion, self.frame, self.sequence = completion, frame, sequence
        self.session, self.workspace = session, workspace
        self.collectives = dict(runtime=runtime, vocabulary_runtime=vocabulary_runtime, coordinator=coordinator)
        self.gather_runtime = gather_runtime
        self.markov_runtime = None
        self.buffer_layout = buffer_layout
        self.owners = []
        buffers = None if buffer_layout is None else buffer_layout.stage.fork()
        self.forwards = SimpleNamespace(decoder=decoder,
            context=partial(PreparedContext, runtime=gather_runtime, coordinator=coordinator,
                buffers=None if buffer_layout is None else buffer_layout.stage.fork()),
            draft=partial(PreparedDraft, embedding=embedding, head=head,
                output_multiplier=1., final_logit_softcapping=None,
                embedding_runtime=gather_runtime, buffers=buffers), markov=None)
        self.rng = SpeculativeRngState.allocate(pool.backing.device, capacity=requests.capacity)
        self.generation, self.offset = torch.zeros_like(requests.generation), torch.zeros_like(requests.generation)
        block_size = width if block_size is None else block_size
        if type(block_size) is not int or not width <= block_size <= 16:
            raise ValueError("DFlash2 proposal block must cover the verification width through sixteen tokens")
        self.block_size, self.logit_offset = block_size, 1
        self.draft_scratch = DraftScratch.allocate(weights.config.draft, weights.shard,
            requests.capacity * block_size, pool.backing.device, buffers=buffers, block_size=block_size)

    def round(self, batch, width, name, *, target_token_capacity=None):
        from .parallel_draft import PreparedParallelDraftRound

        layout = self.buffer_layout
        forwards = SimpleNamespace(
            decoder=(self.forwards.decoder if layout is None else
                     partial(self.forwards.decoder, capture_buffers=layout.capture.fork())),
            context=self.forwards.context, draft=self.forwards.draft)
        operation = PreparedParallelDraftRound(self, batch, width, name,
            target_token_capacity=target_token_capacity, forwards=forwards,
            storage=({}, {}) if self.local_metadata else layout.rounds[batch, width],
            verification_dummy=None if layout is None else layout.dummy)
        operation.prepare()
        self.owners.append(operation)
        return operation

    def release_examples(self):
        self.scratch = None
        if self.local_metadata:
            self.draft_scratch = None

    def scratch_tensors(self):
        if self.local_metadata:
            return ()
        from .parallel_draft import PreparedParallelDraftRound

        values = [self.draft_scratch.scratch_tensors()]
        for operation in self.owners:
            if isinstance(operation, PreparedContextAbsorption):
                values.append(operation.context.scratch_tensors())
            elif isinstance(operation, PreparedParallelDraftRound):
                values.extend((operation.metadata, operation.draft_metadata,
                    operation.before, operation.active, operation.candidates, operation.drafts,
                    operation.valid, operation.seed, operation.offset, operation.base_offset,
                    operation.sampled, operation.markov_token, operation.context.scratch_tensors()))
                _, proposals, verify, _ = operation.operations
                values.append(verify.output)
                for proposal in proposals:
                    if proposal.proposal.candidates is not None:
                        values.extend((proposal.proposal.candidates, proposal.proposal.selected))
        return tuple(values)


class PreparedContextAbsorption:
    """Absorb consumed target rows after a prefill forward."""

    def __init__(self, decoder, context, metadata, *, local_metadata=False):
        self.decoder, self.context = decoder, context
        self.local_metadata = local_metadata
        self.metadata = None if local_metadata else metadata
        if local_metadata:
            context.functional()
        release_outputs = getattr(decoder, "release_outputs", None)
        if release_outputs is not None:
            release_outputs()

    def __call__(self, metadata=None):
        if self.local_metadata:
            result = self.decoder(metadata)
            self.context(result[2], metadata.lengths, metadata)
        else:
            result = self.decoder()
            auxiliary = self.decoder.auxiliary if result is None else result[2]
            self.context(auxiliary, self.metadata.lengths)
