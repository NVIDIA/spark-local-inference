# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared row selection with caller-supplied integer candidate IDs."""

import torch
from b12x.preparation import PreparedCall

from ignition.preparation.tuning import declare, register_finalizer


class PreparedRowTopK:
    def __init__(self, scores, lengths, indices, output_scores, output_indices, *, session, name):
        from b12x.attention import row_topk

        rows, width = scores.shape
        self.arguments = (scores, lengths, indices, output_scores, output_indices)
        self.output = output_indices
        self.plan = row_topk.plan(row_topk.RowTopKQuery(rows=rows, width=width,
            topk=output_indices.shape[1]), device=scores.device)
        declare(session, (self.plan.request(name=name, prepare_call=lambda state:
            PreparedCall(run=lambda: state.run(*self.arguments), output=output_indices,
                         owners=self.arguments)),))

    def functional(self, *, session):
        register_finalizer(session, self._release_examples)
        return self

    def _release_examples(self):
        self.arguments = self.output = None

    def __call__(self, scores=None, lengths=None, indices=None, output_scores=None, output_indices=None):
        from b12x.attention import row_topk

        arguments = self.arguments if scores is None else (scores, lengths, indices, output_scores, output_indices)
        row_topk.topk(self.plan, *arguments)
        return arguments[-1]
