# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Collective text execution and request completion."""

from ignition.components.collectives import PreparedAllReduce
from ignition.components.sampling import PreparedSampling
from ignition.components.state import save_hidden


class PreparedGeneration:
    def __init__(
        self,
        decoder,
        batch,
        records,
        frame,
        sequence,
        *,
        runtime,
        coordinator,
        session,
        name,
        workspace,
        context_factory=None,
    ):
        self.decoder, self.batch = decoder, batch
        self.validation = PreparedAllReduce(
            batch.validation_input,
            batch.validation_sum,
            runtime=runtime,
            coordinator=coordinator,
            session=session,
            name=name + ".validation",
        )
        b, rows = batch.inputs, batch.rows
        args = (
            batch.requests,
            b.row_slot_ids[:rows],
            b.row_generations[:rows],
            b.live_rows,
            records,
            frame,
            sequence,
        )
        self.completion = PreparedSampling(decoder.scratch.logits, *args, sample_enabled=b.sample_enabled[:rows])
        self.completion.prepare()
        self.completion.prepare_broadcast(runtime=decoder.head.gather.runtime, session=session,
            coordinator=coordinator, name=name + ".sampling")
        self.completion.functional(session=session)
        self.compilation_variant = "sample"
        self.context = None
        if decoder.weights.draft is not None:
            if context_factory is None:
                raise ValueError("speculative generation requires a draft context factory")
            self.context = context_factory(
                decoder.weights,
                decoder.auxiliary,
                batch.metadata,
                decoder.pool,
                session=session,
                name=name + ".context",
                workspace=workspace,
            )
        decoder.release_outputs()
        self.validation.functional(session=session)
        if self.context is not None:
            self.context.functional()
        self.batch.functional()

    def __call__(self):
        self._run()

    def observe(self):
        """Return forward-local outputs and packed metadata for diagnostic consumers."""
        return self._run(observe=True)

    def _run(self, *, observe=False):
        validation, present = self.batch.validate()
        metadata = self.batch.begin(self.validation(validation), present=present)
        hidden, logits, auxiliary, _ = self.decoder(metadata)
        save_hidden(
            hidden,
            self.decoder.pool.last_hidden,
            metadata.state_rows,
            metadata.lengths > 0,
        )
        self.batch.finish(metadata)
        self.completion.run(logits)
        if self.context is not None:
            self.context(auxiliary, metadata.lengths, metadata)
        if observe:
            return {"target.hidden": hidden, "target.logits": logits,
                    **{"target." + name: getattr(metadata, name) for name in
                       ("live_counts", "starts", "lengths", "query_start_loc", "token_ids", "positions")}}
