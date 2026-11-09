# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse Glimmer DFlash2 composition and single-GPU speculative preparation."""

from functools import partial

import torch

from ignition.components.dflash2 import DraftScratch, PreparedContext, PreparedDraft as _PreparedDraft
from ignition.components.speculation import SpeculativeRngState
from ignition.recipes.muse_spark.dflash import DraftForwards
from ignition.recipes.muse_spark.pool import PAGE_TOKENS


class PreparedDraft(_PreparedDraft):
    """The DFlash2 round over the target's raw mask embedding and ``lm_head``.

    Each draft layer runs a two-tap grouped convolution before its attention and
    another before its MLP; the rank-256 top-16 candidate selector traces the block
    once the layers are done. The draft head applies the target's ``output_multiplier``
    and ``final_logit_softcapping`` from ``dflash_config``, so proposal logits sit on
    the target's own scale.
    """

    def __init__(self, weights, *args, **kwargs):
        options = weights.config.draft.text["dflash_config"]
        super().__init__(weights, *args, embedding=weights.embedding, head=weights.head,
            output_multiplier=options["output_multiplier"],
            final_logit_softcapping=options["final_logit_softcapping"], **kwargs)


def _target_decoder(weights, tables, scratch, metadata, pool, **kwargs):
    from .decoder import PreparedDecoder
    kwargs["buffers"] = kwargs["buffers"].fork()
    return PreparedDecoder(weights, scratch, metadata, pool, **kwargs)


class SpeculativePreparation:
    """What ``PreparedParallelDraftRound`` reads to build one round at TP1: every runtime is ``None``."""

    def __init__(self, weights, pool, requests, inputs, scratch, *, completion, frame, sequence,
                 session, workspace, buffers, decoder_buffers, context_buffers):
        self.page_tokens = PAGE_TOKENS
        self.weights, self.tables, self.pool = weights, None, pool
        self.requests, self.inputs, self.scratch = requests, inputs, scratch
        self.completion, self.frame, self.sequence = completion, frame, sequence
        self.session, self.workspace = session, workspace
        self.collectives = dict(runtime=None, vocabulary_runtime=None, coordinator=None)
        self.markov_runtime = None
        self.forwards = DraftForwards(
            partial(_target_decoder, embedding_runtime=None, buffers=decoder_buffers),
            partial(PreparedContext, runtime=None, coordinator=None, buffers=context_buffers),
            partial(PreparedDraft, embedding_runtime=None, buffers=buffers))
        self.rng = SpeculativeRngState.allocate(pool.backing.device, capacity=requests.capacity)
        self.generation, self.offset = torch.zeros_like(requests.generation), torch.zeros_like(requests.generation)
        self.block_size, self.logit_offset = weights.config.draft.block_size, 1
        self.draft_scratch = DraftScratch.allocate(weights.config.draft, weights.shard,
            requests.capacity * self.block_size, pool.backing.device, buffers=buffers, block_size=self.block_size)
