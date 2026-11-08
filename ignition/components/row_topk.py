# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared row selection with caller-supplied integer candidate IDs."""

from b12x.preparation import PreparedCall

from ignition.preparation.tuning import declare


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

    def __call__(self):
        from b12x.attention import row_topk

        row_topk.topk(self.plan, *self.arguments)
        return self.output
