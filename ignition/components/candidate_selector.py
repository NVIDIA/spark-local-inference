# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Predecessor-conditioned low-rank corrections to a vocabulary shortlist."""

import torch

from .linear import UnquantizedProjection


class PreparedCandidateSelector:
    def __init__(self, hidden, weights, *, top_k, session, name):
        self.weights, self.top_k = weights, top_k
        self.projected = torch.zeros((*hidden.shape[:-1], weights["projection"].shape[0]),
                                     dtype=hidden.dtype, device=hidden.device)
        self.unary = torch.empty((*hidden.shape[:-1], top_k), dtype=torch.float32, device=hidden.device)
        self.ids = torch.empty(self.unary.shape, dtype=torch.int64, device=hidden.device)
        self.projection = UnquantizedProjection(hidden.flatten(0, -2), weights["projection"],
            self.projected.flatten(0, -2), session=session, name=name + ".hidden")
        self.functional_mode = False

    def functional(self, *, session):
        """Project selector features and shortlist candidates into forward-local tensors."""
        self.projection.functional(session=session)
        self.projected = self.unary = self.ids = None
        self.functional_mode = True
        return self

    def prepare_features(self, hidden):
        if not self.functional_mode:
            self.prepare(hidden)
            return self.projected
        return self.projection(hidden.flatten(0, -2)).view(
            *hidden.shape[:-1], self.weights["projection"].shape[0])

    def prepare(self, hidden):
        self.projection(hidden.flatten(0, -2))

    def prepare_shortlists(self, logits, *, features=None):
        if self.functional_mode:
            return logits.topk(self.top_k, dim=-1, sorted=False)
        unary, ids = logits.topk(self.top_k, dim=-1, sorted=False)
        self.unary.copy_(unary)
        self.ids.copy_(ids)
        return self.unary, self.ids

    def __call__(self, logits, predecessor, position, selector_features=None):
        if self.functional_mode:
            if selector_features is None:
                raise ValueError("functional selector correction requires selector features")
            return self.apply(logits, predecessor, position, selector_features)
        unary, ids = logits.topk(self.top_k, dim=-1, sorted=False)
        self._correct(logits, predecessor, position, unary, ids)

    def _correct(self, logits, predecessor, position, unary, ids):
        coded = self.weights["predecessor"][predecessor] * self.projected[:, position]
        successors = self.weights["successor"][ids]
        correction = torch.bmm(successors, coded.unsqueeze(-1)).squeeze(-1)
        logits.fill_(-torch.inf)
        logits.scatter_(1, ids, unary + correction.float())

    def apply_shortlist(self, logits, predecessor, position, selector_features=None, unary=None, ids=None):
        if not self.functional_mode:
            if selector_features is None:
                self._correct(logits, predecessor, position, self.unary[:, position], self.ids[:, position])
            else:
                self._correct(logits, predecessor, position, unary[:, position], ids[:, position])
            return logits
        if selector_features is None or unary is None or ids is None:
            raise ValueError("functional selector correction requires features and prepared shortlist values")
        return self._functional_correct(logits, predecessor, position, selector_features[:, position],
                                        unary[:, position], ids[:, position])

    def apply(self, logits, predecessor, position, selector_features):
        unary, ids = logits.topk(self.top_k, dim=-1, sorted=False)
        return self._functional_correct(logits, predecessor, position, selector_features[:, position], unary, ids)

    def _functional_correct(self, logits, predecessor, position, selector_features, unary, ids):
        coded = self.weights["predecessor"][predecessor] * selector_features
        successors = self.weights["successor"][ids]
        correction = torch.bmm(successors, coded.unsqueeze(-1)).squeeze(-1)
        return torch.full_like(logits, -torch.inf).scatter_(1, ids, unary + correction.float())
