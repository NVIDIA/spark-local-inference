# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Sample from a cached normalized hidden state without consuming a token."""

import torch

from ignition.components.collectives import PreparedAllReduce
from ignition.components.resume import PreparedResumeState
from ignition.components.sampling import PreparedSampling
from ignition.components.vocabulary import PreparedVocabularyProjection


class PreparedResume:
    def __init__(
        self,
        weights,
        requests,
        pool,
        inputs,
        records,
        frame,
        sequence,
        *,
        rows,
        runtime,
        vocabulary_runtime,
        coordinator,
        session,
        name,
        activation_mode="a16",
        logits_dtype=torch.bfloat16,
        workspace=None,
    ):
        if logits_dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("resume logits require BF16 or FP32 storage")
        self.state = PreparedResumeState(
            requests,
            pool.last_hidden,
            inputs.row_slot_ids[:rows],
            inputs.row_generations[:rows],
            inputs.live_rows,
            world_size=weights.shard.world_size,
            state_rows=inputs.request_state_rows,
        )
        self.validation = PreparedAllReduce(
            self.state.validation_input,
            self.state.validation_sum,
            runtime=runtime,
            coordinator=coordinator,
            session=session,
            name=name + ".validation",
        )
        def empty(shape):
            return torch.empty(shape, dtype=logits_dtype, device=pool.backing.device)
        self.local_logits = empty((rows, weights.head.shape[0] if isinstance(weights.head, torch.Tensor)
                                   else weights.head.out_features))
        self.logits = empty((rows, requests.vocab_size))
        self.projection = PreparedVocabularyProjection(
            self.state.hidden,
            weights.head,
            self.local_logits,
            self.logits,
            runtime=vocabulary_runtime,
            coordinator=coordinator,
            session=session,
            name=name + ".head",
            activation_mode=activation_mode,
            workspace=workspace,
        ).functional(session=session)
        args = (
            requests,
            inputs.row_slot_ids[:rows],
            inputs.row_generations[:rows],
            inputs.live_rows,
            records,
            frame,
            sequence,
        )
        self.completion = PreparedSampling(self.logits, *args)
        self.completion.prepare()
        self.completion.prepare_broadcast(runtime=vocabulary_runtime, session=session,
            coordinator=coordinator, name=name + ".sampling")
        self.completion.functional(session=session)
        self.validation.functional(session=session)
        self.state.functional()
        self.local_logits = self.logits = None
        self.compilation_variant = "sample"

    def __call__(self):
        validation, present = self.state.validate()
        totals = self.validation(validation)
        hidden = self.state.gather(totals, present=present)
        logits = self.projection(hidden)
        self.completion.run(logits)
