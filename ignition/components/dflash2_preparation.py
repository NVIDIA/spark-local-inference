# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared DFlash2 owners for captured target-verification rounds."""

from functools import partial
from types import SimpleNamespace

import torch

from .dflash2 import DraftScratch, PreparedContext, PreparedDraft
from .speculation import SpeculativeRngState


class DFlash2Preparation:
    def __init__(self, weights, pool, requests, inputs, scratch, *, decoder, embedding, head,
                 page_tokens, completion, frame, sequence, session, workspace,
                 verification_width=8, study_width=None, runtime=None,
                 vocabulary_runtime=None, coordinator=None, block_size=None, gather_runtime=None, examples=None,
                 ):
        width = verification_width if study_width is None else study_width
        if type(width) is not int or not 1 <= width <= 16:
            raise ValueError("DFlash2 verification capacity must be an integer in [1,16]")
        self.study_width, self.page_tokens = study_width, page_tokens
        self.examples = {} if examples is None else examples
        self.weights, self.tables, self.pool = weights, None, pool
        self.requests, self.inputs, self.scratch = requests, inputs, scratch
        self.completion, self.frame, self.sequence = completion, frame, sequence
        self.session, self.workspace = session, workspace
        self.collectives = dict(runtime=runtime, vocabulary_runtime=vocabulary_runtime, coordinator=coordinator)
        self.gather_runtime = gather_runtime
        self.markov_runtime = None
        self.owners = []
        self.forwards = SimpleNamespace(decoder=decoder,
            context=partial(PreparedContext, runtime=gather_runtime, coordinator=coordinator, examples=self.examples),
            draft=partial(PreparedDraft, embedding=embedding, head=head,
                output_multiplier=1., final_logit_softcapping=None,
                embedding_runtime=gather_runtime, examples=self.examples), markov=None)
        self.rng = SpeculativeRngState.allocate(pool.backing.device, capacity=requests.capacity)
        self.generation, self.offset = torch.zeros_like(requests.generation), torch.zeros_like(requests.generation)
        block_size = width if block_size is None else block_size
        if type(block_size) is not int or not width <= block_size <= 16:
            raise ValueError("DFlash2 proposal block must cover the verification width through sixteen tokens")
        self.block_size, self.logit_offset = block_size, 1
        self.draft_scratch = DraftScratch.allocate(weights.config.draft, weights.shard,
            requests.capacity * block_size, pool.backing.device, block_size=block_size)

    def round(self, batch, width, name, *, target_token_capacity=None):
        from .parallel_draft import PreparedParallelDraftRound

        operation = PreparedParallelDraftRound(self, batch, width, name,
            target_token_capacity=target_token_capacity, forwards=self.forwards)
        operation.prepare()
        self.owners.append(operation)
        return operation

    def release_examples(self):
        self.scratch = self.draft_scratch = None
        self.examples.clear()
        self.examples = None


class PreparedContextAbsorption:
    """Absorb consumed target rows after a prefill forward."""

    def __init__(self, decoder, context, metadata=None):
        self.decoder, self.context = decoder, context.functional()
        decoder.release_outputs()

    def __call__(self, metadata):
        result = self.decoder(metadata)
        self.context(result[2], metadata.lengths, metadata)
        return result
