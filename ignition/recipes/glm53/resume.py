# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Sample from GLM's cached final normalized hidden output without token consumption."""

import torch

from ignition.components.collectives import PreparedAllReduce
from ignition.components.resume import PreparedResumeState
from ignition.components.sampling import PreparedSampling
from ignition.components.vocabulary import PreparedVocabularyProjection


class PreparedResume:
    def __init__(self, weights, requests, pool, inputs, records, frame, sequence, *, rows,
                 runtime, vocabulary_runtime, coordinator, session, name, workspace=None):
        self.state = PreparedResumeState(requests, pool.last_hidden, inputs.row_slot_ids[:rows],
            inputs.row_generations[:rows], inputs.live_rows, world_size=weights.shard.world_size, state_rows=inputs.request_state_rows)
        self.validation = PreparedAllReduce(self.state.validation_input, self.state.validation_sum,
            runtime=runtime, coordinator=coordinator, session=session, name=f"{name}.validation")
        self.local_logits = torch.empty((rows, 154880 // weights.shard.world_size),
            dtype=torch.bfloat16, device=self.state.hidden.device)
        self.logits = torch.empty((rows, 154880), dtype=torch.bfloat16, device=self.state.hidden.device)
        self.projection = PreparedVocabularyProjection(self.state.hidden, weights.head,
            self.local_logits, self.logits, runtime=vocabulary_runtime, coordinator=coordinator,
            session=session, name=f"{name}.lm_head", workspace=workspace)
        arguments = (requests, inputs.row_slot_ids[:rows], inputs.row_generations[:rows], inputs.live_rows,
                     records, frame, sequence)
        self.completion = PreparedSampling(self.logits, *arguments)
        self.completion.prepare()
        self.completion.prepare_broadcast(runtime=vocabulary_runtime, session=session,
            coordinator=coordinator, name=name + ".sampling")
        self.state.functional()
        self.validation.functional(session=session)
        self.projection.functional(session=session)
        self.completion.functional(session=session)
        self.local_logits = self.logits = None
        self.compilation_variant = "sample"

    def __call__(self):
        self.forward()

    def forward(self):
        validation, present = self.state.validate()
        hidden = self.state.gather(self.validation(validation), present=present)
        logits = self.projection(hidden)
        self.completion.run(logits)
        return hidden, logits
