# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Causal confidence heads and prefix budget selection."""

import math
import heapq

import torch

from .linear import UnquantizedProjection


def causal_prefix(confidences, threshold):
    """Admit position i using only confidence through i and stop at the first failure.

    Confidence i must depend only on history and proposals preceding i. Looking
    ahead to a later confidence when deciding whether to admit i biases sampling.
    """
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("confidence threshold must be finite and in (0, 1]")
    product, count = 1., 0
    for confidence in confidences:
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be finite and in [0, 1]")
        product *= confidence
        if product < threshold:
            break
        count += 1
    return count


def lagged_budget(confidences, costs):
    """Choose a width from a completed previous round's confidence snapshot.

    Costs cover widths one through len(confidences)+1 and include the entire
    proposal, verification and feedback step. Current sampled drafts must not
    influence this decision.
    """
    if len(costs) != len(confidences) + 1 or any(not math.isfinite(c) or c <= 0 for c in costs):
        raise ValueError("positive complete-step costs must cover every verification width")
    expected, survival = 1., 1.
    utilities = [expected / costs[0]]
    for confidence, cost in zip(confidences, costs[1:], strict=True):
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be finite and in [0, 1]")
        survival *= confidence
        expected += survival
        utilities.append(expected / cost)
    return max(range(len(utilities)), key=utilities.__getitem__) + 1


def allocate_draft_budget(confidences, budget):
    """Allocate a fixed draft count across causal request prefixes.

    The caller chooses the budget independently of these sampled proposals.
    A request's next confidence enters the frontier only after its preceding
    position has been admitted. Ties prefer the earlier position, then request.
    """
    if type(budget) is not int or budget < 0:
        raise ValueError("draft budget must be a nonnegative integer")
    rows = [tuple(row) for row in confidences]
    if any(not math.isfinite(p) or not 0 <= p <= 1 for row in rows for p in row):
        raise ValueError("confidence must be finite and in [0, 1]")
    frontier = [(-row[0], 0, slot) for slot, row in enumerate(rows) if row]
    heapq.heapify(frontier)
    counts = [0] * len(rows)
    for _ in range(min(budget, sum(map(len, rows)))):
        probability, position, slot = heapq.heappop(frontier)
        counts[slot] += 1
        if position + 1 < len(rows[slot]):
            heapq.heappush(frontier, (probability * rows[slot][position + 1], position + 1, slot))
    return tuple(counts)


class PreparedDraftConfidence:
    """Compute FP32 confidence from proposal hidden and predecessor Markov features."""

    def __init__(self, source, weight, output, *, name, bias=None):
        from .embedding import PreparedEmbedding

        owner, batch, width = source.owner, source.batch, source.width
        hidden_width = source.draft.head_hidden_shape[-1]
        embedding = source.markov.embed.weight
        if weight.shape != (1, hidden_width + embedding.shape[1]):
            raise ValueError("confidence weight must cover hidden and preceding-token Markov features")
        tokens = torch.empty((batch * (width - 1),), dtype=torch.int64, device=embedding.device)
        markov = torch.empty((tokens.numel(), embedding.shape[1]), dtype=embedding.dtype, device=embedding.device)
        features = torch.empty((tokens.numel(), weight.shape[1]), dtype=weight.dtype, device=embedding.device)
        logits = torch.empty((tokens.numel(), 1), dtype=torch.float32, device=embedding.device)
        self.embedding = PreparedEmbedding(embedding, tokens, markov, session=owner.session,
            name=name + ".canonical_predecessors").functional(session=owner.session, dynamic_inputs=True)
        self.projection = UnquantizedProjection(features, weight, logits,
            session=owner.session, name=name, bias=bias).functional(session=owner.session)
        self.batch, self.depth, self.offset = batch, width - 1, owner.logit_offset
        self.feature_dtype = weight.dtype

    def __call__(self, proposal):
        hidden = proposal.features["head_hidden"][:, self.offset:self.offset + self.depth]
        preceding = torch.cat((torch.where(proposal.active[:, None], proposal.candidates[:, :1], 0),
                               proposal.drafts[:, :self.depth - 1]), dim=1)
        embedded = self.embedding(token_ids=preceding.reshape(-1)).view(self.batch, self.depth, -1)
        features = torch.cat((hidden, embedded), dim=-1).to(self.feature_dtype)
        return self.projection(features.view(self.batch * self.depth, -1)).view(self.batch, self.depth).sigmoid()
