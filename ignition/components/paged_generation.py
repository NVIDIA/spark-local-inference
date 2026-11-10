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
        buffers=None,
        local_metadata=False,
    ):
        self.decoder, self.batch = decoder, batch
        self.local_metadata = local_metadata
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
        release_outputs = getattr(decoder, "release_outputs", None)
        if release_outputs is not None:
            release_outputs()
        if local_metadata:
            self.validation.functional(session=session)
            if self.context is not None:
                self.context.functional()
            self.batch.functional()

    def __call__(self):
        self._run()

    def observe(self):
        """Execute the forward and return its packed metadata for untimed checks."""
        if not self.local_metadata:
            raise RuntimeError("forward observations require call-local metadata")
        return self._run(observe=True)

    def _run(self, *, observe=False):
        if self.local_metadata:
            validation, present = self.batch.validate()
            metadata = self.batch.begin(self.validation(validation), present=present)
            result = self.decoder(metadata)
        else:
            self.batch.begin(self.validation(self.batch.validate()))
            metadata = self.batch.metadata
            result = self.decoder()
        if result is None:
            hidden, logits, auxiliary = self.decoder.scratch.hidden, self.decoder.scratch.logits, self.decoder.auxiliary
        else:
            hidden, logits, auxiliary = result[:3]
        save_hidden(
            hidden,
            self.decoder.pool.last_hidden,
            metadata.state_rows,
            metadata.lengths > 0 if self.local_metadata else self.batch.token_state.active,
        )
        if self.local_metadata:
            self.batch.finish(metadata)
        else:
            self.batch.finish()
        self.completion.run(logits)
        if self.context is not None:
            if self.local_metadata:
                self.context(auxiliary, metadata.lengths, metadata)
            else:
                self.context(auxiliary, metadata.lengths)
        if observe:
            return {"target." + name: getattr(metadata, name) for name in
                    ("live_counts", "starts", "lengths", "query_start_loc")}
