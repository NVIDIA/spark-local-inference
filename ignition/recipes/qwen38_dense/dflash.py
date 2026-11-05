# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen target verification and shared DFlash2 proposals."""

from ignition.components.dflash2_preparation import DFlash2Preparation
from ignition.components.gdn_recovery import GDNLayerRecords

from .config import PAGE_TOKENS
from .decoder import PreparedDecoder


class SpeculativePreparation(DFlash2Preparation):
    def __init__(self, weights, pool, requests, inputs, scratch, *, completion, frame, sequence,
                 session, workspace, buffers, decoder_buffers, context_buffers, verification_width=8, study_width=None):
        width = verification_width if study_width is None else study_width
        self.records = GDNLayerRecords.allocate((i for i, kind in enumerate(weights.config.layers)
            if kind == "linear_attention"), requests.capacity, pool.backing.device, key_heads=16, value_heads=48, width=width)

        def decoder(weights, tables, scratch, metadata, pool, *, runtime, **kwargs):
            return PreparedDecoder(weights, scratch, metadata, pool, records=self.records,
                buffers=decoder_buffers.cursor(), **kwargs)

        super().__init__(weights, pool, requests, inputs, scratch, decoder=decoder,
            embedding=weights.tree.embed_tokens.weight,
            head=weights.head if weights.draft_head is None else weights.draft_head,
            page_tokens=PAGE_TOKENS, completion=completion, frame=frame, sequence=sequence,
            session=session, workspace=workspace, buffers=buffers, context_buffers=context_buffers,
            verification_width=verification_width, study_width=study_width)
