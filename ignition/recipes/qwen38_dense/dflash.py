# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen target verification and shared DFlash2 proposals."""

from functools import partial
from types import SimpleNamespace

import torch

from ignition.components.dflash2 import DraftScratch, PreparedContext, PreparedDraft
from ignition.components.gdn_recovery import GDNLayerRecords
from ignition.components.speculation import SpeculativeRngState

from .config import PAGE_TOKENS
from .decoder import PreparedDecoder


class SpeculativePreparation:
    def __init__(self, weights, pool, requests, inputs, scratch, *, completion, frame, sequence,
                 session, workspace, buffers, decoder_buffers, context_buffers, verification_width=8, study_width=None):
        self.study_width = study_width
        width = verification_width if study_width is None else study_width
        if type(width) is not int or not 1 <= width <= 16:
            raise ValueError("Qwen verification capacity must be an integer in [1,16]")
        self.page_tokens = PAGE_TOKENS
        self.weights, self.tables, self.pool = weights, None, pool
        self.requests, self.inputs, self.scratch = requests, inputs, scratch
        self.completion, self.frame, self.sequence = completion, frame, sequence
        self.session, self.workspace = session, workspace
        self.collectives = dict(runtime=None, vocabulary_runtime=None, coordinator=None)
        self.markov_runtime = None
        self.records = GDNLayerRecords.allocate((i for i, kind in enumerate(weights.config.layers)
            if kind == "linear_attention"), requests.capacity, pool.backing.device, key_heads=16, value_heads=48, width=width)

        def decoder(weights, tables, scratch, metadata, pool, *, runtime, **kwargs):
            return PreparedDecoder(weights, scratch, metadata, pool, records=self.records,
                buffers=decoder_buffers.cursor(), **kwargs)

        self.forwards = SimpleNamespace(decoder=decoder,
            context=partial(PreparedContext, runtime=None, coordinator=None, buffers=context_buffers.cursor()),
            draft=partial(PreparedDraft, embedding=weights.tree.embed_tokens.weight,
                head=weights.head if weights.draft_head is None else weights.draft_head,
                output_multiplier=1., final_logit_softcapping=None,
                embedding_runtime=None, buffers=buffers), markov=None)
        self.rng = SpeculativeRngState.allocate(pool.backing.device, capacity=requests.capacity)
        self.generation, self.offset = torch.zeros_like(requests.generation), torch.zeros_like(requests.generation)
        self.block_size, self.logit_offset = width, 1
        self.draft_scratch = DraftScratch.allocate(weights.config.draft, weights.shard,
            requests.capacity * self.block_size, pool.backing.device, buffers=buffers, block_size=self.block_size)
