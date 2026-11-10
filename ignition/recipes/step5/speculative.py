# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Step-5 causal predictors bound to shared stochastic speculative rounds."""

from ignition.components.causal_draft import PreparedCausalDraftRound

from .decoder import PreparedDecoder
from .mtp import PreparedAcceptedContext, PreparedProposals


class PreparedRound(PreparedCausalDraftRound):
    def __init__(self, owner, batch, width, name):
        def context(weights, metadata, pool, **kwargs):
            return PreparedAcceptedContext(weights, metadata, pool)

        super().__init__(owner, batch, width, name, decoder_factory=PreparedDecoder,
            context_factory=context, proposal_factory=PreparedProposals)
