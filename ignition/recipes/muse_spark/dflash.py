# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse Spark DFlash2 composition and speculative preparation."""

from dataclasses import dataclass
from functools import partial

import torch

from ignition.components.dflash2 import (
    DraftAttention, DraftScratch, PreparedContext, PreparedDraft as _PreparedDraft,
)
from ignition.components.speculation import SpeculativeRngState
from ignition.components.dflash2_preparation import DFlash2Preparation

from .pool import PAGE_TOKENS


class PreparedDraft(_PreparedDraft):
    def __init__(self, weights, *args, **kwargs):
        options = weights.config.draft.text["dflash_config"]
        super().__init__(weights, *args, embedding=weights.embedding, head=weights.head,
            output_multiplier=options["output_multiplier"],
            final_logit_softcapping=options["final_logit_softcapping"], **kwargs)


@dataclass(frozen=True)
class DraftForwards:
    decoder: object
    context: object
    draft: object
    markov: object = None


def _target_decoder(weights, tables, scratch, metadata, pool, **kwargs):
    from .decoder import PreparedDecoder
    kwargs["buffers"] = kwargs["buffers"].fork()
    return PreparedDecoder(weights, scratch, metadata, pool, **kwargs)


class SpeculativePreparation:
    round = DFlash2Preparation.round

    def __init__(self, weights, pool, requests, inputs, scratch, *, completion, frame, sequence,
                 session, workspace, runtime, vocabulary_runtime, context_runtime, embedding_runtime,
                 preexpert_runtime, coordinator, buffer_layout, decoder_buffers):
        self.page_tokens = PAGE_TOKENS
        self.local_metadata = False
        self.weights, self.tables, self.pool = weights, None, pool
        self.requests, self.inputs, self.scratch = requests, inputs, scratch
        self.completion, self.frame, self.sequence = completion, frame, sequence
        self.session, self.workspace = session, workspace
        self.collectives = dict(runtime=runtime, vocabulary_runtime=vocabulary_runtime, coordinator=coordinator)
        self.markov_runtime = None
        self.buffer_layout, self.owners = buffer_layout, []
        buffers = buffer_layout.stage.fork()
        self.forwards = DraftForwards(
            partial(_target_decoder, embedding_runtime=embedding_runtime, preexpert_runtime=preexpert_runtime,
                    buffers=decoder_buffers),
            partial(PreparedContext, runtime=context_runtime, coordinator=coordinator, buffers=buffer_layout.stage.fork()),
            partial(PreparedDraft, embedding_runtime=embedding_runtime, buffers=buffers))
        self.rng = SpeculativeRngState.allocate(pool.backing.device, capacity=requests.capacity)
        self.generation, self.offset = torch.zeros_like(requests.generation), torch.zeros_like(requests.generation)
        self.block_size, self.logit_offset = weights.config.draft.block_size, 1
        self.draft_scratch = DraftScratch.allocate(weights.config.draft, weights.shard,
            requests.capacity * self.block_size, pool.backing.device, buffers=buffers)
