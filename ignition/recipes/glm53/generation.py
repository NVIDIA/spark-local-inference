# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compose a GLM rank's text forward, request consumption, and completion."""

import torch
from dataclasses import dataclass

from ignition.components.collectives import PreparedAllReduce
from ignition.components.sampling import PreparedSampling
from ignition.components.state import save_hidden

from .batch import PreparedBatch
from .stack import PreparedDecoder
from .state import DecoderMetadata


@dataclass(frozen=True)
class GenerationOutput:
    metadata: DecoderMetadata
    hidden: torch.Tensor
    logits: torch.Tensor
    feedback: torch.Tensor | None


class PreparedGeneration:
    """Gather GPU-owned tokens and commit root-broadcast sampling results on every rank.

    Native execution waits for every rank before retiring a step or reusing its
    cache pages.
    """

    def __init__(self, decoder: PreparedDecoder, batch: PreparedBatch, completion,
                 frame_index, submission_sequence, *, pool, runtime, coordinator, session, name):
        if decoder.metadata is not batch.metadata or decoder.weights.shard != batch.shard:
            raise ValueError("GLM generation requires the decoder's metadata and rank")
        if decoder.state is not pool.decoder:
            raise ValueError("GLM generation requires the decoder's continuation pool")
        self.decoder, self.batch, self.pool = decoder, batch, pool
        validation = torch.empty((1, max(8, 2 * batch.capacity)), dtype=torch.bfloat16,
                                 device=decoder.metadata.token_ids.device)
        self.validation = PreparedAllReduce(validation, torch.empty_like(validation),
            runtime=runtime, coordinator=coordinator, session=session, name=f"{name}.validation")
        b, rows = batch.inputs, batch.rows
        arguments = (batch.requests, b.row_slot_ids[:rows], b.row_generations[:rows], b.live_rows,
                     completion, frame_index, submission_sequence)
        self.completion = PreparedSampling(
            decoder.scratch.logits, *arguments, sample_enabled=b.sample_enabled[:rows])
        self.completion.prepare()
        self.completion.prepare_broadcast(runtime=decoder.lm_head.gather.runtime, session=session,
            coordinator=coordinator, name=name + ".sampling")
        self.validation.functional(session=session)
        self.completion.functional(session=session)
        self.batch.functional()
        self.decoder.functional(session=session)
        self.compilation_variant = "sample"

    def rebind_pages(self, pages):
        self.decoder.rebind_pages()
        self.batch.pages = self.pool.memory_capacity[1:]

    def __call__(self):
        self.forward()

    def forward(self):
        validation_input, present = self.batch.validate()
        metadata = self.batch.begin(self.validation(validation_input), present)
        hidden, logits, feedback, _ = self.decoder(metadata)
        save_hidden(hidden, self.pool.last_hidden, metadata.recurrent.convolution_slots,
                    metadata.recurrent.convolution_slots >= 0)
        self.batch.finish(metadata)
        self.completion.run(logits)
        return GenerationOutput(metadata, hidden, logits, feedback)
