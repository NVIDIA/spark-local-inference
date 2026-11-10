# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit diagnostic outputs for functional sparse-attention tests."""

import torch
from torch.utils._pytree import tree_flatten, tree_map


@torch.library.custom_op("flare_test::overwrite_kimi_query", mutates_args=("query", "observed"))
def _overwrite_query(query: torch.Tensor, observed: torch.Tensor) -> None:
    query.zero_()
    observed.copy_(query)


@_overwrite_query.register_fake
def _overwrite_query_fake(query, observed):
    return None


def allocate_observations(result):
    return tree_map(lambda value: torch.empty_like(value) if isinstance(value, torch.Tensor) else value, result)


def copy_observations(destination, result):
    destinations, _ = tree_flatten(destination)
    values, _ = tree_flatten(result)
    for target, value in zip(destinations, values, strict=True):
        if isinstance(target, torch.Tensor):
            target.copy_(value)


def poison_observations(result):
    values, _ = tree_flatten(result)
    for value in values:
        if isinstance(value, torch.Tensor):
            value.view(torch.uint8).fill_(255)


class AttentionForward:
    def __init__(self, operation, source, metadata, *, observations=None):
        self.operation, self.source, self.metadata = operation, source, metadata
        self.observations = observations
        self.output = torch.empty_like(source)

    def __call__(self):
        if self.observations is None:
            self.output.copy_(self.operation(self.source, self.metadata))
        else:
            result = self.operation._run(self.source, self.metadata)
            copy_observations(self.observations, result)
            self.output.copy_(result.output)
            # The gather's source may be overwritten once its join has completed.
            for stage, observed in zip(result.stages, self.observations.stages, strict=True):
                _overwrite_query(stage.quantized, observed.quantized)
        return self.output


class AttentionTail:
    """Run prepared attention operators on an explicitly recorded selection."""

    def __init__(self, operation, observations, metadata):
        self.operation, self.observations, self.metadata = operation, observations, metadata
        self.output = torch.empty_like(observations.output)

    def __call__(self):
        from ignition.recipes.kimi31.dcp import selection_slots

        operation, result = self.operation, self.observations
        stage, = result.stages
        rows = stage.selected.shape[0]
        part = operation.parts[rows]
        owners = torch.empty_like(stage.owners)
        slots = torch.empty_like(stage.slots)
        counts = torch.empty_like(stage.counts)
        selection_slots(stage.selected, self.metadata.positions, self.metadata.token_page_table,
            owners, slots, counts, operation.shards, operation.cache.shape[0], local=True)
        partial, lse = part['attention'](stage.gathered.view(rows, operation.heads * operation.shards.world, -1),
            page_table=slots, lengths=counts, starts=part['starts'])
        attended = part['combine'](partial, lse)
        self.output.copy_(operation.finish(attended, result.projected))
        stage.owners.copy_(owners)
        stage.slots.copy_(slots)
        stage.counts.copy_(counts)
        stage.partial.copy_(partial)
        stage.lse.copy_(lse)
        result.attended.copy_(attended)
        return self.output
